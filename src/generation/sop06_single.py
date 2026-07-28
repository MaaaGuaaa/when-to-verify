"""Strict SOP05 target-motion integration for the history-only SOP06 renderer."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.contracts import (
    LONG40_FUTURE_STEPS,
    LONG40_HISTORY_STEPS,
    POSE_TIME_LAYOUT_VERSION,
    SCHEMA_VERSION,
    BaseState,
    LocalTrajectory,
    OracleContext,
    OracleWorld,
    build_grid_spec,
    validate_base_state,
    validate_oracle_context,
)

from .observation_renderer import RenderedObservation, render_observation
from .sop05r_contracts import (
    SOP05R_TEB_GENERATOR_VERSION,
    SOP05R_TEB_PLANNER_VERSION,
    SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION,
)
from .sop05r_teb_output_loader import LoadedSop05rTebOutput
from .sop05_seen_prior import SeenPriorFailure, SeenPriorResult
from .sop05_unseen_prior import (
    Long40TargetMotion,
    UnseenPriorMotherResult,
    UnseenPriorRealization,
)
from src.planning.lightweight_teb import PlannedTebRoute
from .structural_blindspot import (
    StructuralBlindSpot,
    has_continuous_emergence,
)
from src.utils.seeding import stable_digest


_BLIND_SPOT_KEYS = frozenset({"kind", "structural", "occluder_ids"})
_V5_ENVIRONMENT_BLIND_SPOT_KEYS = frozenset(
    {"kind", "occluder_ids", "blind_region_digest"}
)
_SOP05R_ENVIRONMENT_BLIND_SPOT_KEYS = frozenset(
    {"kind", "occluder_ids", "scene_template_id"}
)
_STRUCTURAL_KEYS = frozenset(
    {"forward_fov_deg", "range_m", "blind_sectors"}
)
_EVENT_KINDS = frozenset({"environment", "structural", "mixed"})
_USE_WORLD_SENSOR = object()
_SOP06_SINGLE_REGIMES = frozenset(
    {"seen_then_occluded", "unseen_in_history_window"}
)
_SOP06_SINGLE_PREFIX_SIZES = frozenset({50000, 100000, 125000})
_SOP06_SINGLE_HARD_CAP = 125000
_SOP06_SINGLE_FORBIDDEN_MODEL_METADATA_TOKENS = (
    "future",
    "oracle",
    "theta",
    "angle",
    "attempt",
    "rejection",
    "clearance",
    "risk",
    "collision",
)


@dataclass(frozen=True)
class RenderedSop06Group:
    """Rendered group with an explicit audit-certification boundary."""

    pair_group_id: str
    variant_kinds: tuple[str, ...]
    observations: tuple[RenderedObservation, ...]
    coverage_mask: tuple[bool, ...]
    is_complete: bool
    audit_certified: bool


@dataclass(frozen=True)
class Sop06SingleRendererInput:
    """History-only data allowed to cross the single-result renderer boundary."""

    sample_id: str
    mother_id: str
    split: str
    base_state: BaseState
    observed_static_occupancy: np.ndarray
    scene_dynamic_history: Mapping[str, np.ndarray]
    scene_dynamic_specs: Mapping[str, dict[str, object]]
    scene_dynamic_history_observed: Mapping[str, np.ndarray]
    sensor_config: StructuralBlindSpot | None


@dataclass(frozen=True)
class Sop06SinglePublicationContext:
    """Authenticated shared state used to adapt one A or B realization."""

    sample_id: str
    mother_id: str
    split: str
    base_state: BaseState
    trajectory: LocalTrajectory
    oracle_world: OracleWorld
    observed_static_occupancy: np.ndarray
    scene_dynamic_history: Mapping[str, np.ndarray]
    scene_dynamic_specs: Mapping[str, dict[str, object]]
    hidden_object_ids: tuple[str, ...]
    sensor_config: StructuralBlindSpot | None
    target_dynamic_object_id: str
    target_footprint_spec: Mapping[str, object]
    target_history_observed: np.ndarray
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class Sop06SinglePublication:
    """One publication entry with an explicit model-safe/oracle-only split."""

    sample_id: str
    mother_id: str
    split: str
    regime: str
    renderer_input: Sop06SingleRendererInput
    trajectory: LocalTrajectory
    oracle_world: OracleWorld
    hidden_object_ids: tuple[str, ...]
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class Sop06SingleFailureRecord:
    """One non-published mother result retained only for release accounting."""

    mother_id: str
    split: str
    regime: str
    reason: str


@dataclass(frozen=True)
class Sop06CombinedSingleRelease:
    """Deterministic A+B single-result release with bounded prefix selection."""

    entries: tuple[Sop06SinglePublication, ...]
    failures: tuple[Sop06SingleFailureRecord, ...]
    requested_prefix_size: int | None
    accepted_counts: Mapping[str, int]
    failed_counts: Mapping[str, int]


def _require_single_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _model_safe_metadata(value: Mapping[str, object], *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")

    def copy_value(item: object, *, path: str) -> object:
        if isinstance(item, Mapping):
            copied: dict[str, object] = {}
            for key, child in item.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(f"{path} keys must be non-empty strings")
                lowered = key.lower()
                if any(
                    token in lowered
                    for token in _SOP06_SINGLE_FORBIDDEN_MODEL_METADATA_TOKENS
                ):
                    raise ValueError(f"{path}.{key} is oracle-only information")
                copied[key] = copy_value(child, path=f"{path}.{key}")
            return copied
        if isinstance(item, (list, tuple)):
            return [copy_value(child, path=f"{path}[]") for child in item]
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"{path} must contain JSON-safe model metadata")

    copied = copy_value(value, path=name)
    if not isinstance(copied, dict):  # pragma: no cover - mapping invariant
        raise RuntimeError("model metadata copy lost its mapping type")
    return copied


def _model_safe_dynamic_specs(
    value: Mapping[str, object],
    *,
    name: str,
) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    copied: dict[str, dict[str, object]] = {}
    for object_id, spec in value.items():
        _require_single_identifier(object_id, name=f"{name} object ID")
        copied[object_id] = _model_safe_metadata(
            spec,
            name=f"{name}[{object_id!r}]",
        )
    return copied


def _validate_single_context(context: Sop06SinglePublicationContext) -> None:
    if not isinstance(context, Sop06SinglePublicationContext):
        raise TypeError("context must be a Sop06SinglePublicationContext")
    _require_single_identifier(context.sample_id, name="context.sample_id")
    _require_single_identifier(context.mother_id, name="context.mother_id")
    _require_single_identifier(context.split, name="context.split")
    _require_single_identifier(
        context.target_dynamic_object_id,
        name="context.target_dynamic_object_id",
    )
    if not isinstance(context.base_state, BaseState):
        raise TypeError("context.base_state must be a BaseState")
    if context.base_state.split != context.split:
        raise ValueError("context split must match base_state.split")
    if not isinstance(context.trajectory, LocalTrajectory):
        raise TypeError("context.trajectory must be a LocalTrajectory")
    if not isinstance(context.oracle_world, OracleWorld):
        raise TypeError("context.oracle_world must be an OracleWorld")
    if context.oracle_world.base_state_id != context.base_state.state_id:
        raise ValueError("context oracle_world/base_state identity mismatch")
    if not isinstance(context.scene_dynamic_history, Mapping) or not isinstance(
        context.scene_dynamic_specs, Mapping
    ):
        raise TypeError("context scene histories and specs must be mappings")
    if set(context.scene_dynamic_history) != set(context.scene_dynamic_specs):
        raise ValueError("context scene history/spec IDs must match")
    _model_safe_dynamic_specs(
        context.scene_dynamic_specs,
        name="context.scene_dynamic_specs",
    )
    _model_safe_metadata(
        context.target_footprint_spec,
        name="context.target_footprint_spec",
    )
    if context.target_dynamic_object_id in context.scene_dynamic_history:
        raise ValueError("context scene must not pre-populate the target")
    if (
        not isinstance(context.target_history_observed, np.ndarray)
        or context.target_history_observed.dtype != np.bool_
        or context.target_history_observed.shape != (8,)
    ):
        raise ValueError("context.target_history_observed must be boolean [8]")
    if not isinstance(context.hidden_object_ids, tuple):
        raise TypeError("context.hidden_object_ids must be a tuple")
    if len(set(context.hidden_object_ids)) != len(context.hidden_object_ids):
        raise ValueError("context.hidden_object_ids must be unique")
    if context.sensor_config is not None and not isinstance(
        context.sensor_config, StructuralBlindSpot
    ):
        raise TypeError("context.sensor_config must be a StructuralBlindSpot or None")
    static = np.asarray(context.observed_static_occupancy)
    if (
        static.dtype != np.float32
        or static.shape != context.oracle_world.static_occupancy.shape
        or not np.isfinite(static).all()
    ):
        raise ValueError("context.observed_static_occupancy is invalid")
    if not np.array_equal(static, context.oracle_world.static_occupancy):
        raise ValueError("observed static occupancy must equal oracle static occupancy")
    _model_safe_metadata(context.provenance, name="context.provenance")


def _single_target_arrays(
    history: object,
    future: object,
) -> tuple[np.ndarray, np.ndarray]:
    history_array = np.asarray(history)
    future_array = np.asarray(future)
    if (
        history_array.shape != (8, 3)
        or future_array.shape != (32, 3)
        or history_array.dtype.kind != "f"
        or future_array.dtype.kind != "f"
        or not np.isfinite(history_array).all()
        or not np.isfinite(future_array).all()
    ):
        raise ValueError("single-result target must use finite Long40 [8,3]/[32,3] poses")
    return (
        np.array(history_array, dtype=np.float32, order="C", copy=True),
        np.array(future_array, dtype=np.float32, order="C", copy=True),
    )


def _single_deployment_history(
    history: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Remove exact poses from frames that were not deployment-observed."""

    result = np.empty_like(history)
    last_observed = np.zeros(3, dtype=np.float32)
    for step in range(history.shape[0]):
        if observed[step]:
            last_observed = history[step]
        result[step] = last_observed
    return np.array(result, dtype=np.float32, order="C", copy=True)


def _single_oracle_world_with_target(
    context: Sop06SinglePublicationContext,
    *,
    target_future: np.ndarray | None,
    target_footprint_spec: Mapping[str, object],
) -> OracleWorld:
    trajectories = {
        object_id: np.array(value, dtype=np.float32, order="C", copy=True)
        for object_id, value in context.oracle_world.dynamic_object_trajectories.items()
    }
    specs = deepcopy(context.oracle_world.dynamic_object_specs)
    target_id = context.target_dynamic_object_id
    if target_future is None:
        trajectories.pop(target_id, None)
        specs.pop(target_id, None)
    else:
        trajectories[target_id] = np.array(
            target_future,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        specs[target_id] = deepcopy(dict(target_footprint_spec))
    return OracleWorld(
        world_id=(
            f"{context.oracle_world.world_id}::single::"
            f"{stable_digest(context.sample_id, size=16)}"
        ),
        base_state_id=context.oracle_world.base_state_id,
        static_occupancy=np.array(
            context.oracle_world.static_occupancy,
            dtype=np.float32,
            order="C",
            copy=True,
        ),
        dynamic_object_trajectories=trajectories,
        dynamic_object_specs=specs,
        occluders=tuple(deepcopy(item) for item in context.oracle_world.occluders),
        blind_spot_config=deepcopy(context.oracle_world.blind_spot_config),
        random_seed=context.oracle_world.random_seed,
        metadata=deepcopy(context.oracle_world.metadata),
    )


def _single_publication(
    *,
    context: Sop06SinglePublicationContext,
    regime: str,
    target_history: np.ndarray | None,
    target_history_observed: np.ndarray | None,
    target_future: np.ndarray | None,
    target_footprint_spec: Mapping[str, object],
) -> Sop06SinglePublication:
    _validate_single_context(context)
    if regime not in _SOP06_SINGLE_REGIMES:
        raise ValueError("single-result regime is invalid")
    histories = {
        object_id: np.array(value, dtype=np.float32, order="C", copy=True)
        for object_id, value in context.scene_dynamic_history.items()
    }
    specs = _model_safe_dynamic_specs(
        context.scene_dynamic_specs,
        name="context.scene_dynamic_specs",
    )
    observed = {
        object_id: np.ones(8, dtype=np.bool_)
        for object_id in histories
    }
    safe_target_footprint_spec = _model_safe_metadata(
        target_footprint_spec,
        name="target_footprint_spec",
    )
    target_id = context.target_dynamic_object_id
    if target_history is not None:
        if target_history_observed is None:
            raise ValueError("target history requires an observation mask")
        histories[target_id] = _single_deployment_history(
            target_history,
            target_history_observed,
        )
        specs[target_id] = safe_target_footprint_spec
        observed[target_id] = np.array(
            target_history_observed,
            dtype=np.bool_,
            order="C",
            copy=True,
        )
    elif target_history_observed is not None:
        raise ValueError("target observation mask requires target history")
    oracle_world = _single_oracle_world_with_target(
        context,
        target_future=target_future,
        target_footprint_spec=safe_target_footprint_spec,
    )
    hidden_ids = list(context.hidden_object_ids)
    if (
        target_future is not None
        and target_id not in hidden_ids
    ):
        hidden_ids.append(target_id)
    hidden_object_ids = tuple(
        object_id
        for object_id in hidden_ids
        if object_id in oracle_world.dynamic_object_trajectories
    )
    renderer_input = Sop06SingleRendererInput(
        sample_id=context.sample_id,
        mother_id=context.mother_id,
        split=context.split,
        base_state=context.base_state,
        observed_static_occupancy=np.array(
            context.observed_static_occupancy,
            dtype=np.float32,
            order="C",
            copy=True,
        ),
        scene_dynamic_history=histories,
        scene_dynamic_specs=specs,
        scene_dynamic_history_observed=observed,
        sensor_config=context.sensor_config,
    )
    return Sop06SinglePublication(
        sample_id=context.sample_id,
        mother_id=context.mother_id,
        split=context.split,
        regime=regime,
        renderer_input=renderer_input,
        trajectory=context.trajectory,
        oracle_world=oracle_world,
        hidden_object_ids=hidden_object_ids,
        provenance=_model_safe_metadata(context.provenance, name="context.provenance"),
    )


def adapt_seen_prior_result(
    result: SeenPriorResult,
    *,
    context: Sop06SinglePublicationContext,
) -> Sop06SinglePublication:
    """Connect one accepted Regime-B result without exposing its oracle fields to rendering."""

    if not isinstance(result, SeenPriorResult):
        raise TypeError("result must be a SeenPriorResult")
    if result.mother_id != context.mother_id:
        raise ValueError("seen-prior result/context mother identity mismatch")
    history, future = _single_target_arrays(result.history_poses, result.future_poses)
    observed = np.array(
        context.target_history_observed,
        dtype=np.bool_,
        order="C",
        copy=True,
    )
    if not bool(observed[0]):
        raise ValueError("seen-prior target must be observed at history index 0")
    return _single_publication(
        context=context,
        regime="seen_then_occluded",
        target_history=history,
        target_history_observed=observed,
        target_future=future,
        target_footprint_spec=context.target_footprint_spec,
    )


def adapt_finalized_sop05_scenario(
    *,
    context: Sop06SinglePublicationContext,
    regime: str,
    target_present: bool,
    history_poses: np.ndarray,
    future_poses: np.ndarray,
) -> Sop06SinglePublication:
    """Adapt one authenticated persisted SOP05 final record without resampling."""

    history, future = _single_target_arrays(history_poses, future_poses)
    if regime == "seen_then_occluded":
        if not target_present:
            raise ValueError(
                "seen-then-occluded final scenario must contain a target"
            )
        return _single_publication(
            context=context,
            regime=regime,
            target_history=history,
            target_history_observed=context.target_history_observed,
            target_future=future,
            target_footprint_spec=context.target_footprint_spec,
        )
    if regime != "unseen_in_history_window":
        raise ValueError("final scenario regime is invalid")
    return _single_publication(
        context=context,
        regime=regime,
        target_history=None,
        target_history_observed=None,
        target_future=future if target_present else None,
        target_footprint_spec=context.target_footprint_spec,
    )


def adapt_finalized_sop05_renderer_input(
    *,
    renderer_input: Sop06SingleRendererInput,
    regime: str,
    target_present: bool,
    target_dynamic_object_id: str,
    target_footprint_spec: Mapping[str, object],
    target_history_observed: np.ndarray,
    history_poses: np.ndarray,
) -> Sop06SingleRendererInput:
    """Build the causal SOP06 input without loading oracle robot trajectories."""

    if not isinstance(renderer_input, Sop06SingleRendererInput):
        raise TypeError("renderer_input must be a Sop06SingleRendererInput")
    histories = {
        object_id: np.array(value, dtype=np.float32, order="C", copy=True)
        for object_id, value in renderer_input.scene_dynamic_history.items()
    }
    specs = _model_safe_dynamic_specs(
        renderer_input.scene_dynamic_specs,
        name="renderer_input.scene_dynamic_specs",
    )
    observed = {
        object_id: np.array(value, dtype=np.bool_, order="C", copy=True)
        for object_id, value in renderer_input.scene_dynamic_history_observed.items()
    }
    if target_dynamic_object_id in histories:
        raise ValueError("renderer input must not pre-populate the target")
    if regime == "seen_then_occluded":
        if not target_present:
            raise ValueError(
                "seen-then-occluded final scenario must contain a target"
            )
        history = np.asarray(history_poses)
        if (
            history.shape != (8, 3)
            or history.dtype.kind != "f"
            or not np.isfinite(history).all()
        ):
            raise ValueError("single-result target must use finite Long40 [8,3] poses")
        mask = np.asarray(target_history_observed)
        if mask.dtype != np.bool_ or mask.shape != (8,):
            raise ValueError("target_history_observed must be boolean [8]")
        histories[target_dynamic_object_id] = _single_deployment_history(
            np.array(history, dtype=np.float32, order="C", copy=True),
            mask,
        )
        specs[target_dynamic_object_id] = _model_safe_metadata(
            target_footprint_spec,
            name="target_footprint_spec",
        )
        observed[target_dynamic_object_id] = np.array(
            mask,
            dtype=np.bool_,
            order="C",
            copy=True,
        )
    elif regime != "unseen_in_history_window":
        raise ValueError("final scenario regime is invalid")
    return Sop06SingleRendererInput(
        sample_id=renderer_input.sample_id,
        mother_id=renderer_input.mother_id,
        split=renderer_input.split,
        base_state=renderer_input.base_state,
        observed_static_occupancy=np.array(
            renderer_input.observed_static_occupancy,
            dtype=np.float32,
            order="C",
            copy=True,
        ),
        scene_dynamic_history=histories,
        scene_dynamic_specs=specs,
        scene_dynamic_history_observed=observed,
        sensor_config=renderer_input.sensor_config,
    )


def adapt_seen_prior_failure(
    failure: SeenPriorFailure,
    *,
    context: Sop06SinglePublicationContext,
) -> Sop06SingleFailureRecord:
    if not isinstance(failure, SeenPriorFailure):
        raise TypeError("failure must be a SeenPriorFailure")
    _validate_single_context(context)
    if failure.mother_id != context.mother_id:
        raise ValueError("seen-prior failure/context mother identity mismatch")
    return Sop06SingleFailureRecord(
        mother_id=context.mother_id,
        split=context.split,
        regime="seen_then_occluded",
        reason=failure.reason,
    )


def _unseen_target_arrays(
    target: Long40TargetMotion,
) -> tuple[np.ndarray, np.ndarray]:
    positions = target.positions
    headings = target.headings
    history = np.column_stack((positions[:8], headings[:8]))
    future = np.column_stack((positions[8:], headings[8:]))
    return _single_target_arrays(history, future)


def adapt_unseen_prior_realization(
    realization: UnseenPriorRealization,
    *,
    context: Sop06SinglePublicationContext,
) -> Sop06SinglePublication:
    """Connect one Regime-A empty/present realization through the same path."""

    if not isinstance(realization, UnseenPriorRealization):
        raise TypeError("realization must be an UnseenPriorRealization")
    if (
        realization.mother_id != context.mother_id
        or realization.split != context.split
    ):
        raise ValueError("unseen-prior realization/context identity mismatch")
    target = realization.target_motion
    if target is None:
        return _single_publication(
            context=context,
            regime="unseen_in_history_window",
            target_history=None,
            target_history_observed=None,
            target_future=None,
            target_footprint_spec=context.target_footprint_spec,
        )
    if target.target_dynamic_object_id != context.target_dynamic_object_id:
        raise ValueError("unseen-prior target ID differs from publication context")
    _, future = _unseen_target_arrays(target)
    return _single_publication(
        context=context,
        regime="unseen_in_history_window",
        target_history=None,
        target_history_observed=None,
        target_future=future,
        target_footprint_spec=target.footprint_spec,
    )


def adapt_unseen_prior_failure(
    result: UnseenPriorMotherResult,
    *,
    context: Sop06SinglePublicationContext,
) -> Sop06SingleFailureRecord:
    if not isinstance(result, UnseenPriorMotherResult):
        raise TypeError("result must be an UnseenPriorMotherResult")
    _validate_single_context(context)
    if result.realization is not None or result.provenance.outcome != "no_legal_angle":
        raise ValueError("only a no_legal_angle unseen-prior result is a failure")
    if (
        result.provenance.mother_id != context.mother_id
        or result.provenance.split != context.split
    ):
        raise ValueError("unseen-prior failure/context identity mismatch")
    return Sop06SingleFailureRecord(
        mother_id=context.mother_id,
        split=context.split,
        regime="unseen_in_history_window",
        reason=result.provenance.outcome,
    )


def render_sop06_single_input(
    renderer_input: Sop06SingleRendererInput,
    *,
    config: Mapping[str, Any],
) -> RenderedObservation:
    """Render only the model-safe half of a single-realization publication."""

    if not isinstance(renderer_input, Sop06SingleRendererInput):
        raise TypeError("renderer_input must be a Sop06SingleRendererInput")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    _require_single_identifier(renderer_input.sample_id, name="sample_id")
    _require_single_identifier(renderer_input.mother_id, name="mother_id")
    if renderer_input.base_state.split != renderer_input.split:
        raise ValueError("renderer input split differs from base_state.split")
    scene_dynamic_specs = _model_safe_dynamic_specs(
        renderer_input.scene_dynamic_specs,
        name="renderer_input.scene_dynamic_specs",
    )
    return render_observation(
        renderer_input.base_state,
        scene_dynamic_history=renderer_input.scene_dynamic_history,
        scene_dynamic_specs=scene_dynamic_specs,
        static_occupancy=renderer_input.observed_static_occupancy,
        sensor_config=renderer_input.sensor_config,
        config=config,
        scene_dynamic_history_observed=(
            renderer_input.scene_dynamic_history_observed
        ),
    )


def render_sop06_single_publication(
    publication: Sop06SinglePublication,
    *,
    config: Mapping[str, Any],
) -> RenderedObservation:
    if not isinstance(publication, Sop06SinglePublication):
        raise TypeError("publication must be a Sop06SinglePublication")
    return render_sop06_single_input(publication.renderer_input, config=config)


def build_sop06_single_risk_input(publication: Sop06SinglePublication):
    """Return the existing generic ``RiskBuildInput`` without a paired-group adapter."""

    if not isinstance(publication, Sop06SinglePublication):
        raise TypeError("publication must be a Sop06SinglePublication")
    if (
        publication.trajectory.poses.shape != (LONG40_FUTURE_STEPS, 3)
        or publication.trajectory.controls.shape != (LONG40_FUTURE_STEPS, 2)
    ):
        raise ValueError(
            f"SOP7 trajectory must contain exactly {LONG40_FUTURE_STEPS} "
            "future endpoints"
        )
    if (
        publication.trajectory.poses.dtype != np.float32
        or publication.trajectory.controls.dtype != np.float32
        or not np.isfinite(publication.trajectory.poses).all()
        or not np.isfinite(publication.trajectory.controls).all()
    ):
        raise ValueError("SOP7 trajectory endpoints must be finite float32 arrays")
    if (
        publication.trajectory.metadata.get("pose_time_layout_version")
        != POSE_TIME_LAYOUT_VERSION
    ):
        raise ValueError("SOP7 trajectory pose-time layout is invalid")
    if publication.renderer_input.base_state.robot_history.shape != (
        LONG40_HISTORY_STEPS,
        3,
    ):
        raise ValueError(
            f"SOP7 model history must contain exactly {LONG40_HISTORY_STEPS} frames"
        )
    if publication.oracle_world.metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"SOP7 oracle world schema_version must be {SCHEMA_VERSION}")
    for object_id, future in publication.oracle_world.dynamic_object_trajectories.items():
        if (
            not isinstance(future, np.ndarray)
            or future.shape != (LONG40_FUTURE_STEPS, 3)
        ):
            raise ValueError(
                f"SOP7 oracle future {object_id!r} must contain exactly "
                f"{LONG40_FUTURE_STEPS} future endpoints"
            )
    from src.datasets.risk_dataset import RiskBuildInput

    return RiskBuildInput(
        sample_id=publication.sample_id,
        pair_group_id=(
            f"sop06-single/{publication.regime}/{publication.split}/"
            f"{publication.mother_id}"
        ),
        event_type=publication.regime,
        base_state=publication.renderer_input.base_state,
        trajectory=publication.trajectory,
        oracle_world=publication.oracle_world,
        observed_static_occupancy=publication.renderer_input.observed_static_occupancy,
        scene_dynamic_history=publication.renderer_input.scene_dynamic_history,
        scene_dynamic_specs=publication.renderer_input.scene_dynamic_specs,
        hidden_object_ids=publication.hidden_object_ids,
        sensor_config=publication.renderer_input.sensor_config,
        provenance={
            "regime": publication.regime,
            "mother_id": publication.mother_id,
            **_model_safe_metadata(publication.provenance, name="publication.provenance"),
        },
        scene_dynamic_history_observed=(
            publication.renderer_input.scene_dynamic_history_observed
        ),
    )


def coordinate_sop06_single_release(
    entries: tuple[Sop06SinglePublication, ...] | list[Sop06SinglePublication],
    *,
    failures: tuple[Sop06SingleFailureRecord, ...] | list[Sop06SingleFailureRecord] = (),
    requested_prefix_size: int | None = None,
) -> Sop06CombinedSingleRelease:
    """Deterministically coordinate the shared A+B prefix without duplication."""

    if not isinstance(entries, (tuple, list)) or not all(
        isinstance(entry, Sop06SinglePublication) for entry in entries
    ):
        raise TypeError("entries must be a sequence of Sop06SinglePublication")
    if not isinstance(failures, (tuple, list)) or not all(
        isinstance(failure, Sop06SingleFailureRecord) for failure in failures
    ):
        raise TypeError("failures must be a sequence of Sop06SingleFailureRecord")
    if requested_prefix_size is not None:
        if (
            isinstance(requested_prefix_size, bool)
            or not isinstance(requested_prefix_size, int)
            or requested_prefix_size not in _SOP06_SINGLE_PREFIX_SIZES
        ):
            raise ValueError("requested_prefix_size must be 50000, 100000, or 125000")
    identities: set[tuple[str, str]] = set()
    sample_ids: set[str] = set()
    for entry in entries:
        if entry.regime not in _SOP06_SINGLE_REGIMES:
            raise ValueError("release entry regime is invalid")
        identity = (entry.split, entry.mother_id)
        if identity in identities or entry.sample_id in sample_ids:
            raise ValueError("release must preserve one-to-one mother/sample identity")
        identities.add(identity)
        sample_ids.add(entry.sample_id)
    for failure in failures:
        if failure.regime not in _SOP06_SINGLE_REGIMES:
            raise ValueError("release failure regime is invalid")
        identity = (failure.split, failure.mother_id)
        if identity in identities:
            raise ValueError("release mother cannot be both accepted and failed")
        identities.add(identity)
    ordered_entries = tuple(
        sorted(entries, key=lambda entry: (entry.split, entry.mother_id, entry.sample_id))
    )
    if requested_prefix_size is not None:
        if len(ordered_entries) < requested_prefix_size:
            raise ValueError("requested combined prefix is not filled")
        ordered_entries = ordered_entries[:requested_prefix_size]
    elif len(ordered_entries) > _SOP06_SINGLE_HARD_CAP:
        raise ValueError("combined release exceeds the 125000 hard cap")
    accepted_counts: dict[str, int] = {}
    for entry in ordered_entries:
        accepted_counts[entry.regime] = accepted_counts.get(entry.regime, 0) + 1
    failed_counts: dict[str, int] = {}
    for failure in failures:
        failed_counts[failure.regime] = failed_counts.get(failure.regime, 0) + 1
    return Sop06CombinedSingleRelease(
        entries=ordered_entries,
        failures=tuple(
            sorted(failures, key=lambda failure: (failure.split, failure.mother_id))
        ),
        requested_prefix_size=requested_prefix_size,
        accepted_counts=dict(sorted(accepted_counts.items())),
        failed_counts=dict(sorted(failed_counts.items())),
    )


def _same_sop06_single_risk_samples(expected: object, actual: object) -> bool:
    """Compare the immutable writer round-trip without relying on ndarray equality."""

    scalar_fields = (
        "sample_id",
        "split",
        "base_state_id",
        "pair_group_id",
        "event_type",
        "collision_label",
        "risk_severity",
        "min_clearance",
        "near_miss",
        "first_collision_time",
        "metadata",
    )
    array_fields = (
        "bev_history",
        "state_channels",
        "trajectory_channels",
        "robot_state",
    )
    return all(
        getattr(expected, name) == getattr(actual, name) for name in scalar_fields
    ) and all(
        np.array_equal(getattr(expected, name), getattr(actual, name))
        for name in array_fields
    )


def write_sop06_single_risk_shard(
    entries: tuple[Sop06SinglePublication, ...] | list[Sop06SinglePublication],
    output_dir: str | Path,
    *,
    base_config: Mapping[str, object],
    risk_config: Mapping[str, object],
    shard_index: int = 0,
) -> dict[str, Path]:
    """Build one same-split single-result shard and safely reuse an exact resume.

    The generic shard writer remains immutable.  On a resumed run this wrapper
    loads and verifies the completed shard, then accepts it only when every
    reconstructed model-safe sample exactly matches the deterministic replay.
    """

    if not isinstance(entries, (tuple, list)) or not all(
        isinstance(entry, Sop06SinglePublication) for entry in entries
    ):
        raise TypeError("entries must be a sequence of Sop06SinglePublication")
    if not entries:
        raise ValueError("cannot write an empty SOP06 single-result shard")
    if not isinstance(base_config, Mapping) or not isinstance(risk_config, Mapping):
        raise TypeError("base_config and risk_config must be mappings")

    from src.datasets.risk_dataset import build_risk_sample
    from src.datasets.shard_writer import load_risk_shard, write_risk_shard

    grid = build_grid_spec(dict(base_config))
    samples = tuple(
        build_risk_sample(
            build_sop06_single_risk_input(entry),
            base_config=base_config,
            risk_config=risk_config,
        )
        for entry in entries
    )
    expected = tuple(sorted(samples, key=lambda sample: sample.sample_id))
    output_path = Path(output_dir)

    def reuse_completed_shard() -> dict[str, Path]:
        loaded = load_risk_shard(output_path, grid=grid)
        if len(loaded.samples) != len(expected) or any(
            not _same_sop06_single_risk_samples(written, replayed)
            for written, replayed in zip(expected, loaded.samples)
        ):
            raise ValueError(
                "existing SOP06 shard does not match deterministic replay"
            )
        return {
            "directory": output_path,
            "payload": output_path / "samples.npz",
            "manifest": output_path / "metadata.jsonl",
            "summary": output_path / "summary.json",
        }

    if output_path.exists():
        return reuse_completed_shard()
    try:
        return write_risk_shard(
            samples,
            output_path,
            grid=grid,
            shard_index=shard_index,
            expected_sample_count=len(samples),
        )
    except FileExistsError:
        return reuse_completed_shard()


@dataclass(frozen=True)
class Sop05rTebSop06Handoff:
    """One strict Long40 M7 event prepared for the SOP06 history renderer."""

    event_id: str
    source_collection_digest: str
    base_config: Mapping[str, object]
    event: object
    decision_state: BaseState
    full_route: PlannedTebRoute
    nominal_trajectory: LocalTrajectory
    shared_goal_world_pose: np.ndarray


@dataclass(frozen=True)
class RenderedSop05rTebTargetPair:
    """Target-present/removed SOP06 views sharing one authenticated M7 handoff."""

    handoff: Sop05rTebSop06Handoff
    target_present: RenderedObservation
    target_removed: RenderedObservation


def resolve_sop06_teb_handoff(
    collection: LoadedSop05rTebOutput,
    *,
    event_id: str,
) -> Sop05rTebSop06Handoff:
    """Select one complete M7 event without reopening legacy trajectory stores."""

    if not isinstance(collection, LoadedSop05rTebOutput):
        raise TypeError("collection must be a LoadedSop05rTebOutput")
    if not collection.complete:
        raise ValueError("SOP06 TEB handoff requires a complete M7 collection")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be non-empty")
    if (
        collection.manifest.get("generator_algorithm_version")
        != SOP05R_TEB_GENERATOR_VERSION
        or collection.manifest.get("trajectory_collection_version")
        != SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION
    ):
        raise ValueError("SOP06 TEB handoff rejects non-current M7 collections")
    events = tuple(event for event in collection.events if event.generated_event_id == event_id)
    records = tuple(record for record in collection.trajectories.records if record.event_id == event_id)
    if len(events) != 1 or len(records) != 1:
        raise ValueError("SOP06 TEB handoff requires exactly one event and trajectory record")
    event = events[0]
    record = records[0]
    decision = collection.decision_states.get(record.decision_state_id)
    if not isinstance(decision, BaseState):
        raise ValueError("SOP06 TEB handoff decision state is missing")
    grid = build_grid_spec(dict(collection.manifest["base_config"]))
    if (
        record.planner_version != SOP05R_TEB_PLANNER_VERSION
        or event.target.history_poses.shape != (grid.history_steps, 3)
        or event.target.future_poses.shape != (grid.future_steps, 3)
        or record.full_route.band_poses_world.shape != (21, 3)
        or record.full_route.sampled_poses_world.shape != (40, 3)
        or record.nominal_trajectory.poses.shape != (32, 3)
        or event.world.base_state_id != record.decision_state_id
        or event.target_motion_record.base_state_id != record.decision_state_id
        or event.target_motion_record.trajectory_id
        != record.nominal_trajectory.trajectory_id
    ):
        raise ValueError("SOP06 TEB handoff long40 contract mismatch")
    goal = np.asarray(record.shared_goal_world_pose, dtype=np.float32)
    world_goal = np.asarray(event.world.metadata.get("shared_goal_world_pose"), dtype=np.float32)
    if goal.shape != (3,) or not np.array_equal(goal, world_goal):
        raise ValueError("SOP06 TEB handoff shared goal mismatch")
    return Sop05rTebSop06Handoff(
        event_id=event_id,
        source_collection_digest=collection.publication_semantic_digest,
        base_config=dict(collection.manifest["base_config"]),
        event=event,
        decision_state=decision,
        full_route=record.full_route,
        nominal_trajectory=record.nominal_trajectory,
        shared_goal_world_pose=goal,
    )


def render_sop06_teb_target_pair(
    handoff: Sop05rTebSop06Handoff,
) -> RenderedSop05rTebTargetPair:
    """Render current M7 target-present and target-removed views from one seam."""

    if not isinstance(handoff, Sop05rTebSop06Handoff):
        raise TypeError("handoff must be a Sop05rTebSop06Handoff")
    grid = build_grid_spec(dict(handoff.base_config))
    validate_base_state(handoff.decision_state, grid)
    event = handoff.event
    target = event.target
    if (
        event.target_motion_record.generated_event_id != handoff.event_id
        or target.target_dynamic_object_id in handoff.decision_state.dynamic_object_ids
    ):
        raise ValueError("SOP06 TEB target identity conflicts with decision state")
    histories = {
        object_id: np.array(history, dtype=np.float32, order="C", copy=True)
        for object_id, history in handoff.decision_state.visible_dynamic_object_history.items()
    }
    specs = {
        object_id: deepcopy(spec)
        for object_id, spec in handoff.decision_state.visible_dynamic_object_specs.items()
    }
    removed = render_observation(
        handoff.decision_state,
        scene_dynamic_history=histories,
        scene_dynamic_specs=specs,
        static_occupancy=event.world.static_occupancy,
        sensor_config=None,
        config=handoff.base_config,
    )
    histories[target.target_dynamic_object_id] = np.array(
        target.history_poses, dtype=np.float32, order="C", copy=True
    )
    specs[target.target_dynamic_object_id] = deepcopy(target.footprint_spec)
    present = render_observation(
        handoff.decision_state,
        scene_dynamic_history=histories,
        scene_dynamic_specs=specs,
        static_occupancy=event.world.static_occupancy,
        sensor_config=None,
        config=handoff.base_config,
    )
    return RenderedSop05rTebTargetPair(
        handoff=handoff,
        target_present=present,
        target_removed=removed,
    )
