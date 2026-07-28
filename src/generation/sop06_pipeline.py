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

from .event_sampler import SOP05_GENERATOR_ALGORITHM_VERSION
from .event_target_motion_shard import (
    EventTargetMotionRecord,
    compute_motion_array_digest,
    validate_event_target_motion_world_join,
)
from .dynamic_object_transplant import TransplantedDynamicObject
from .observation_renderer import RenderedObservation, render_observation
from .occluder_sampler import JOINT_MULTI_LOS_PLACEMENT_STRATEGY_VERSION
from .paired_variants import (
    JOINT_ENVIRONMENT_PAIR_VERSION,
    PAIRED_GENERATOR_ALGORITHM_VERSION,
    PAIRED_GROUP_CONTRACT_VERSION,
    PairedEventGroup,
    PairedVariant,
    Sop06TrajectoryHandoff,
    VARIANT_ORDER,
    compute_pair_group_id,
    _paired_target_motion_digest,
    _SOP05_JOIN_METADATA_KEYS,
)
from .sop05r_contracts import (
    SOP05R_GENERATOR_VERSION,
    SOP05R_TRAJECTORY_COLLECTION_VERSION,
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


def _validate_joint_six_pack_versions(
    mother_world: OracleWorld,
    paired_world: OracleWorld,
    *,
    trajectory_handoff: Sop06TrajectoryHandoff | None = None,
) -> None:
    paired_algorithm = paired_world.metadata.get(
        "paired_generator_algorithm_version"
    )
    if paired_algorithm == PAIRED_GENERATOR_ALGORITHM_VERSION:
        for label, world in (("mother", mother_world), ("paired", paired_world)):
            if world.metadata.get("joint_pair_generator_algorithm_version") == (
                JOINT_ENVIRONMENT_PAIR_VERSION
            ):
                raise ValueError(
                    f"{label} uses retired {JOINT_ENVIRONMENT_PAIR_VERSION}"
                )
        if paired_world.metadata.get("pair_group_contract_version") != (
            PAIRED_GROUP_CONTRACT_VERSION
        ):
            raise ValueError(
                "paired pair_group_contract_version must equal "
                f"{PAIRED_GROUP_CONTRACT_VERSION}"
            )
        _validate_formal_mother_world(
            mother_world,
            trajectory_handoff=trajectory_handoff,
        )
        return
    if mother_world.metadata.get("event_kind") != "environment":
        return
    for label, world in (
        ("mother", mother_world),
        ("paired", paired_world),
    ):
        if world.metadata.get(
            "joint_pair_generator_algorithm_version"
        ) != JOINT_ENVIRONMENT_PAIR_VERSION:
            raise ValueError(
                f"{label} joint_pair_generator_algorithm_version must equal "
                f"{JOINT_ENVIRONMENT_PAIR_VERSION}"
            )
    if len(mother_world.occluders) != 1:
        raise ValueError(
            "joint environment mother must contain exactly one occluder"
    )
    if mother_world.occluders[0].get("placement_strategy") != (
        JOINT_MULTI_LOS_PLACEMENT_STRATEGY_VERSION
    ):
        raise ValueError(
            "joint environment occluder placement_strategy must equal "
            f"{JOINT_MULTI_LOS_PLACEMENT_STRATEGY_VERSION}"
        )


def _sensor_from_world(world: OracleWorld) -> StructuralBlindSpot | None:
    config = world.blind_spot_config
    if not isinstance(config, Mapping):
        raise ValueError("world blind_spot_config must be a mapping")
    is_v5_environment = set(config) == _V5_ENVIRONMENT_BLIND_SPOT_KEYS
    is_sop05r_environment = (
        set(config) == _SOP05R_ENVIRONMENT_BLIND_SPOT_KEYS
    )
    if (
        not is_v5_environment
        and not is_sop05r_environment
        and set(config) != _BLIND_SPOT_KEYS
    ):
        raise ValueError(
            "world blind_spot_config keys do not match a frozen layout"
        )
    kind = config["kind"]
    if kind not in _EVENT_KINDS:
        raise ValueError("world blind_spot_config kind is invalid")
    if not isinstance(world.metadata, Mapping) or (
        world.metadata.get("event_kind") != kind
    ):
        raise ValueError("world metadata event_kind mismatch")
    occluder_ids = config["occluder_ids"]
    if (
        not isinstance(occluder_ids, list)
        or any(not isinstance(value, str) or not value for value in occluder_ids)
        or len(occluder_ids) != len(set(occluder_ids))
    ):
        raise ValueError("world blind_spot_config occluder_ids are invalid")

    world_occluder_ids: list[str] = []
    for index, occluder in enumerate(world.occluders):
        if not isinstance(occluder, Mapping):
            raise ValueError(f"world occluders[{index}] must be a mapping")
        occluder_id = occluder.get(
            "obstacle_id" if is_sop05r_environment else "occluder_id"
        )
        if not isinstance(occluder_id, str) or not occluder_id:
            raise ValueError(f"world occluders[{index}] id is invalid")
        world_occluder_ids.append(occluder_id)
    if len(world_occluder_ids) != len(set(world_occluder_ids)):
        raise ValueError("world occluder IDs must be unique")
    if occluder_ids != world_occluder_ids:
        raise ValueError(
            "world blind_spot_config occluder_ids mismatch world.occluders"
        )

    if is_v5_environment:
        digest = config["blind_region_digest"]
        if kind != "environment":
            raise ValueError(
                "v5 blind-region layout is valid only for environment worlds"
            )
        if not isinstance(digest, str) or not digest:
            raise ValueError("v5 blind_region_digest must be a non-empty string")
        if not world_occluder_ids:
            raise ValueError("environment world requires an occluder")
        return None

    if is_sop05r_environment:
        template_id = config["scene_template_id"]
        if kind != "environment":
            raise ValueError("SOP05R blind-spot layout requires an environment world")
        if not isinstance(template_id, str) or not template_id:
            raise ValueError("SOP05R scene_template_id must be non-empty")
        if world.metadata.get("scene_template_id") != template_id:
            raise ValueError("SOP05R scene_template_id metadata mismatch")
        if not world_occluder_ids:
            raise ValueError("SOP05R environment world requires an obstacle")
        return None

    raw = config["structural"]
    if kind == "environment":
        if not world_occluder_ids:
            raise ValueError("environment world requires an occluder")
        if raw is not None:
            raise ValueError("environment world cannot contain a structural sensor")
        return None
    if kind == "structural" and world_occluder_ids:
        raise ValueError("structural world cannot contain an occluder")
    if kind == "mixed" and not world_occluder_ids:
        raise ValueError("mixed world requires an occluder")
    if raw is None:
        raise ValueError("structural/mixed world requires a structural sensor")
    if not isinstance(raw, Mapping) or set(raw) != _STRUCTURAL_KEYS:
        raise ValueError("structural blind-spot keys are invalid")
    sectors = raw["blind_sectors"]
    if not isinstance(sectors, list):
        raise ValueError("structural blind-spot sectors must be a list")
    return StructuralBlindSpot(
        forward_fov_deg=raw["forward_fov_deg"],
        range_m=raw["range_m"],
        blind_sectors=tuple(dict(sector) for sector in sectors),
    )


def _validate_context_world_join(
    record: EventTargetMotionRecord,
    world: OracleWorld,
    oracle_context: OracleContext,
    *,
    target_present: bool = True,
) -> None:
    context_ids = set(oracle_context.dynamic_object_future)
    expected_ids = set(context_ids)
    if target_present:
        expected_ids.add(record.target_dynamic_object_id)
    if set(world.dynamic_object_trajectories) != expected_ids:
        raise ValueError("OracleWorld/context dynamic object ids mismatch")
    if set(world.dynamic_object_specs) != expected_ids:
        raise ValueError("OracleWorld/context dynamic object spec ids mismatch")
    for object_id in sorted(context_ids):
        if not np.array_equal(
            world.dynamic_object_trajectories[object_id],
            oracle_context.dynamic_object_future[object_id],
        ):
            raise ValueError(
                f"OracleWorld context future mismatch for {object_id!r}"
            )
        if (
            world.dynamic_object_specs[object_id]
            != oracle_context.dynamic_object_specs[object_id]
        ):
            raise ValueError(
                f"OracleWorld context spec mismatch for {object_id!r}"
            )


def validate_sop05r_paired_target_prefix(
    record: EventTargetMotionRecord,
    mother_world: OracleWorld,
    target: TransplantedDynamicObject,
) -> None:
    """Reject any SOP05R target-present pair that rewrites observed history."""

    if mother_world.metadata.get("generator_algorithm_version") != (
        SOP05R_GENERATOR_VERSION
    ):
        raise ValueError("SOP05R paired prefix check requires a SOP05R mother")
    if not isinstance(target, TransplantedDynamicObject):
        raise TypeError("target must be a TransplantedDynamicObject")
    if (
        target.target_dynamic_object_id != record.target_dynamic_object_id
        or target.source_object_id != record.source_object_id
        or target.snippet_id != record.source_snippet_id
        or target.object_type != record.object_type
        or target.footprint_spec != record.footprint_spec
        or target.footprint_spec_digest != record.footprint_spec_digest
    ):
        raise ValueError("SOP05R paired target identity differs from mother record")
    last_visible = mother_world.metadata.get("target_history_last_visible_index")
    if last_visible is None:
        return
    if isinstance(last_visible, (bool, np.bool_)) or not isinstance(
        last_visible, (int, np.integer)
    ):
        raise ValueError("SOP05R mother last visible index is invalid")
    last_visible = int(last_visible)
    if not 0 <= last_visible <= 5:
        raise ValueError("SOP05R mother last visible index must leave two hidden frames")
    prefix_end = last_visible + 1
    mother_prefix = np.ascontiguousarray(record.history_poses[:prefix_end])
    variant_prefix = np.ascontiguousarray(target.history_poses[:prefix_end])
    if (
        mother_prefix.dtype != variant_prefix.dtype
        or mother_prefix.shape != variant_prefix.shape
        or mother_prefix.tobytes(order="C") != variant_prefix.tobytes(order="C")
    ):
        raise ValueError("SOP05R paired visible history prefix differs from mother")


def _validate_paired_target(
    record: EventTargetMotionRecord,
    mother_world: OracleWorld,
    variant: PairedVariant,
) -> TransplantedDynamicObject | None:
    world = variant.world
    metadata = world.metadata
    if not isinstance(metadata, Mapping):
        raise ValueError("paired world metadata must be a mapping")
    if (
        not isinstance(variant.variant_kind, str)
        or variant.variant_kind not in VARIANT_ORDER
    ):
        raise ValueError("paired variant_kind must be one of six frozen kinds")
    if metadata.get("paired_variant_kind") != variant.variant_kind:
        raise ValueError("paired variant kind metadata mismatch")
    if not isinstance(metadata.get("pair_group_id"), str) or not metadata.get(
        "pair_group_id"
    ):
        raise ValueError("paired world requires pair_group_id")

    target = variant.target
    target_present = target is not None
    if metadata.get("target_present") is not target_present:
        raise ValueError("paired target_present metadata mismatch")
    if target is None:
        if variant.variant_kind != "empty_blind_spot":
            raise ValueError("only empty_blind_spot may omit the target")
        _validate_paired_visibility(variant, target_present=False)
        for key in (
            "paired_target_history_array_digest",
            "paired_target_future_array_digest",
            "paired_target_current_pose",
            "target_provenance",
        ):
            if key not in metadata or metadata[key] is not None:
                raise ValueError(f"empty paired {key} must be None")
        if metadata.get("paired_target_motion_digest") != "target-empty":
            raise ValueError(
                "empty paired paired_target_motion_digest must be target-empty"
            )
        _validate_paired_lineage(
            record,
            mother_world,
            variant,
            target_motion_digest="target-empty",
            target_current_pose=None,
            target_provenance=None,
        )
        return None
    if variant.variant_kind == "empty_blind_spot":
        raise ValueError("empty_blind_spot must omit the target")
    if not isinstance(target, TransplantedDynamicObject):
        raise TypeError("paired target must be a TransplantedDynamicObject")
    if (
        target.target_dynamic_object_id != record.target_dynamic_object_id
        or target.source_object_id != record.source_object_id
        or target.snippet_id != record.source_snippet_id
        or target.object_type != record.object_type
        or target.footprint_spec != record.footprint_spec
        or target.footprint_spec_digest != record.footprint_spec_digest
    ):
        raise ValueError("paired target identity differs from mother record")
    if target.provenance.get("target_type_policy_digest") != (
        record.target_type_policy_digest
    ):
        raise ValueError("paired target policy digest differs from mother record")
    mother_provenance = mother_world.metadata.get("target_provenance")
    if not isinstance(mother_provenance, Mapping):
        raise ValueError("mother target provenance must be a mapping")
    for field, label in (
        ("source_recording_id", "source recording"),
        ("source_session_id", "source session"),
    ):
        mother_value = mother_provenance.get(field)
        if not isinstance(mother_value, str) or not mother_value.strip():
            raise ValueError(f"mother target {label} must be non-empty")
        target_value = target.provenance.get(field)
        if not isinstance(target_value, str) or not target_value.strip():
            raise ValueError(f"paired target {label} must be non-empty")
        if target_value != mother_value:
            raise ValueError(f"paired target {label} differs from mother target")
    for name, array, shape in (
        ("history_poses", target.history_poses, (8, 3)),
        ("current_pose", target.current_pose, (3,)),
        ("future_poses", target.future_poses, (15, 3)),
    ):
        if (
            not isinstance(array, np.ndarray)
            or array.shape != shape
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"paired target {name} contract is invalid")
    if not np.array_equal(target.current_pose, target.history_poses[-1]):
        raise ValueError("paired target current/history seam mismatch")
    if mother_world.metadata.get("generator_algorithm_version") == (
        SOP05R_GENERATOR_VERSION
    ):
        validate_sop05r_paired_target_prefix(record, mother_world, target)
    history_digest = compute_motion_array_digest(
        target.history_poses, field_name="target_history_poses"
    )
    if metadata.get("paired_target_history_array_digest") != history_digest:
        raise ValueError("paired target history array digest mismatch")
    future_digest = compute_motion_array_digest(
        target.future_poses, field_name="target_future_poses"
    )
    if metadata.get("paired_target_future_array_digest") != future_digest:
        raise ValueError("paired target future array digest mismatch")
    if metadata.get("paired_target_motion_digest") != (
        _paired_target_motion_digest(target)
    ):
        raise ValueError("paired target motion digest mismatch")
    _validate_paired_lineage(
        record,
        mother_world,
        variant,
        target_motion_digest=_paired_target_motion_digest(target),
        target_current_pose=[float(value) for value in target.current_pose],
        target_provenance=dict(target.provenance),
    )
    if not np.array_equal(
        world.dynamic_object_trajectories.get(target.target_dynamic_object_id),
        target.future_poses,
    ):
        raise ValueError("paired world target future mismatch")
    if world.dynamic_object_specs.get(target.target_dynamic_object_id) != (
        target.footprint_spec
    ):
        raise ValueError("paired world target spec mismatch")
    _validate_paired_visibility(variant, target_present=True)
    return target


def _validate_paired_visibility(
    variant: PairedVariant,
    *,
    target_present: bool,
) -> None:
    metadata = variant.world.metadata
    field_shapes = {
        "target_visibility_history": 8,
        "visibility_sequence": 33,
    }
    if not target_present:
        for field_name in field_shapes:
            if getattr(variant, field_name) is not None:
                raise ValueError(
                    f"empty paired {field_name} must be None"
                )
            if field_name not in metadata or metadata[field_name] is not None:
                raise ValueError(
                    f"empty paired {field_name} metadata must be None"
                )
        return

    validated: dict[str, np.ndarray] = {}
    for field_name, length in field_shapes.items():
        value = getattr(variant, field_name)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (length,)
            or value.dtype != np.bool_
        ):
            raise ValueError(
                f"paired {field_name} must be bool[{length}]"
            )
        raw_metadata = metadata.get(field_name)
        if (
            not isinstance(raw_metadata, list)
            or len(raw_metadata) != length
            or any(type(item) is not bool for item in raw_metadata)
        ):
            raise ValueError(
                f"paired {field_name} boolean metadata is invalid"
            )
        if not np.array_equal(value, np.asarray(raw_metadata, dtype=bool)):
            raise ValueError(f"paired {field_name} metadata mismatch")
        validated[field_name] = value

    history = validated["target_visibility_history"]
    sequence = validated["visibility_sequence"]
    if history[-1] != sequence[0]:
        raise ValueError("paired target visibility seam mismatch")
    if bool(sequence[0]):
        raise ValueError("paired target current frame must remain hidden")
    if not has_continuous_emergence(sequence, min_visible_frames=2):
        raise ValueError("paired target requires continuous emergence")
    if not bool(sequence[-1]):
        raise ValueError("paired target final frame must be visible")


def _validate_paired_lineage(
    record: EventTargetMotionRecord,
    mother_world: OracleWorld,
    variant: PairedVariant,
    *,
    target_motion_digest: str,
    target_current_pose: list[float] | None,
    target_provenance: dict[str, object] | None,
) -> None:
    world = variant.world
    metadata = world.metadata
    stale_keys = sorted(_SOP05_JOIN_METADATA_KEYS.intersection(metadata))
    if stale_keys:
        raise ValueError(
            "paired world contains stale SOP05 join metadata: "
            + ", ".join(stale_keys)
        )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "world_id": world.world_id,
        "mother_generated_event_id": record.generated_event_id,
        "mother_world_id": mother_world.world_id,
        "mother_target_motion_record_digest": record.record_digest,
        "mother_source_snippet_id": record.source_snippet_id,
        "mother_source_object_id": record.source_object_id,
        "mother_target_type_policy_digest": record.target_type_policy_digest,
        "mother_target_footprint_spec_digest": record.footprint_spec_digest,
        "target_dynamic_object_id": record.target_dynamic_object_id,
        "paired_target_current_pose": target_current_pose,
        "target_provenance": target_provenance,
    }
    for key, expected_value in expected.items():
        if key not in metadata or metadata[key] != expected_value:
            raise ValueError(f"paired {key} metadata mismatch")

    pair_group_id = metadata.get("pair_group_id")
    paired_config_digest = metadata.get("paired_config_digest")
    paired_seed = metadata.get("paired_seed")
    if (
        not isinstance(paired_config_digest, str)
        or len(paired_config_digest) != 32
        or any(char not in "0123456789abcdef" for char in paired_config_digest)
    ):
        raise ValueError("paired_config_digest metadata is invalid")
    expected_pair_group_id = compute_pair_group_id(
        generated_event_id=record.generated_event_id,
        base_state_id=mother_world.base_state_id,
        trajectory_id=record.trajectory_id,
        occluders=mother_world.occluders,
        blind_spot_config=mother_world.blind_spot_config,
        source_snippet_id=record.source_snippet_id,
        target_dynamic_object_id=record.target_dynamic_object_id,
        paired_config_digest=paired_config_digest,
    )
    if pair_group_id != expected_pair_group_id:
        raise ValueError(
            "paired pair_group_id/paired_config_digest does not match "
            "trusted mother lineage"
        )
    if type(paired_seed) is not int or paired_seed != world.random_seed:
        raise ValueError("paired_seed metadata mismatch world random_seed")
    expected_world_id = "world-" + stable_digest(
        pair_group_id,
        variant.variant_kind,
        target_motion_digest,
        paired_seed,
        paired_config_digest,
        size=12,
    )
    if world.world_id != expected_world_id:
        raise ValueError(
            "paired world_id lineage mismatch (paired_config_digest)"
        )


def _build_background_scene(
    base_state: BaseState,
    oracle_context: OracleContext,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    histories: dict[str, np.ndarray] = {}
    specs: dict[str, dict[str, object]] = {}
    base_ids = set(base_state.dynamic_object_ids)
    for object_id in sorted(base_ids):
        histories[object_id] = np.array(
            base_state.visible_dynamic_object_history[object_id],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        specs[object_id] = deepcopy(
            base_state.visible_dynamic_object_specs[object_id]
        )

    for object_id in sorted(oracle_context.dynamic_object_history):
        history = oracle_context.dynamic_object_history[object_id]
        spec = oracle_context.dynamic_object_specs[object_id]
        if object_id in base_ids:
            if not np.array_equal(histories[object_id], history):
                raise ValueError(
                    "overlapping BaseState/OracleContext history mismatch"
                )
            if specs[object_id] != spec:
                raise ValueError(
                    "overlapping BaseState/OracleContext spec mismatch"
                )
            continue
        histories[object_id] = np.array(
            history, dtype=np.float32, order="C", copy=True
        )
        specs[object_id] = deepcopy(spec)
    return histories, specs


def _validate_paired_skeleton(
    mother_world: OracleWorld,
    paired_world: OracleWorld,
) -> StructuralBlindSpot | None:
    _sensor_from_world(mother_world)
    paired_sensor = _sensor_from_world(paired_world)
    if not np.array_equal(
        paired_world.static_occupancy, mother_world.static_occupancy
    ):
        raise ValueError("paired static occupancy differs from mother")
    if paired_world.occluders != mother_world.occluders:
        raise ValueError("paired occluders differ from mother")
    if paired_world.blind_spot_config != mother_world.blind_spot_config:
        raise ValueError("paired blind-spot config differs from mother")
    return paired_sensor


def render_sop06_paired_variant(
    *,
    mother_record: EventTargetMotionRecord,
    mother_world: OracleWorld,
    variant: PairedVariant,
    base_state: BaseState,
    oracle_context: OracleContext,
    config: Mapping[str, Any],
    expected_paired_config_digest: str,
    trajectory_handoff: Sop06TrajectoryHandoff | None = None,
) -> RenderedObservation:
    """Render one actual paired variant after validating its SOP05 mother.

    ``expected_paired_config_digest`` must come from the independently loaded
    paired configuration, never from ``variant.world.metadata``.
    """

    if not isinstance(variant, PairedVariant):
        raise TypeError("variant must be a PairedVariant")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if (
        not isinstance(expected_paired_config_digest, str)
        or len(expected_paired_config_digest) != 32
        or any(
            char not in "0123456789abcdef"
            for char in expected_paired_config_digest
        )
    ):
        raise ValueError(
            "expected_paired_config_digest must be a 32-character "
            "lowercase hex digest"
        )
    if (
        not isinstance(variant.world.metadata, Mapping)
        or variant.world.metadata.get("paired_config_digest")
        != expected_paired_config_digest
    ):
        raise ValueError(
            "paired_config_digest does not match expected trusted digest"
        )
    grid = build_grid_spec(dict(config))
    validate_event_target_motion_world_join(
        mother_record, mother_world, grid
    )
    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(oracle_context, OracleContext):
        raise TypeError("oracle_context must be an OracleContext")
    validate_base_state(base_state, grid)
    validate_oracle_context(oracle_context, grid)
    if base_state.state_id != mother_record.base_state_id:
        raise ValueError("base_state id does not match mother record")
    if oracle_context.base_state_id != mother_record.base_state_id:
        raise ValueError("oracle_context id does not match mother record")
    if variant.world.base_state_id != mother_record.base_state_id:
        raise ValueError("paired world base_state_id mismatch")
    sensor_config = _validate_paired_skeleton(
        mother_world, variant.world
    )
    _validate_joint_six_pack_versions(
        mother_world,
        variant.world,
        trajectory_handoff=trajectory_handoff,
    )
    target = _validate_paired_target(mother_record, mother_world, variant)
    _validate_context_world_join(
        mother_record,
        variant.world,
        oracle_context,
        target_present=target is not None,
    )
    histories, specs = _build_background_scene(base_state, oracle_context)
    if target is not None:
        if target.target_dynamic_object_id in histories:
            raise ValueError("paired target id collides with background history")
        histories[target.target_dynamic_object_id] = np.array(
            target.history_poses, dtype=np.float32, order="C", copy=True
        )
        specs[target.target_dynamic_object_id] = dict(target.footprint_spec)

    return render_observation(
        base_state,
        scene_dynamic_history=histories,
        scene_dynamic_specs=specs,
        static_occupancy=variant.world.static_occupancy,
        sensor_config=sensor_config,
        config=config,
    )


def render_sop06_variant(
    *,
    record: EventTargetMotionRecord,
    world: OracleWorld,
    base_state: BaseState,
    oracle_context: OracleContext,
    config: Mapping[str, Any],
    sensor_config_override: StructuralBlindSpot | None | object = (
        _USE_WORLD_SENSOR
    ),
) -> RenderedObservation:
    """Validate and render one target-present SOP05 mother event.

    Oracle futures are inspected only to validate the outer record/world/context
    join.  The core renderer receives a newly built history-only scene.  The
    target is always present here; paired and empty variants must use
    :func:`render_sop06_paired_variant` so their paired metadata is validated.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    grid = build_grid_spec(dict(config))
    validate_event_target_motion_world_join(record, world, grid)
    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(oracle_context, OracleContext):
        raise TypeError("oracle_context must be an OracleContext")
    validate_base_state(base_state, grid)
    validate_oracle_context(oracle_context, grid)
    if base_state.state_id != record.base_state_id:
        raise ValueError("base_state id does not match target-motion record")
    if oracle_context.base_state_id != record.base_state_id:
        raise ValueError("oracle_context id does not match target-motion record")
    _validate_context_world_join(record, world, oracle_context)
    world_sensor = _sensor_from_world(world)
    if sensor_config_override is _USE_WORLD_SENSOR:
        sensor_config = world_sensor
    elif sensor_config_override is None or isinstance(
        sensor_config_override, StructuralBlindSpot
    ):
        sensor_config = sensor_config_override
    else:
        raise TypeError(
            "sensor_config_override must be a StructuralBlindSpot or None"
        )

    histories, specs = _build_background_scene(base_state, oracle_context)
    if record.target_dynamic_object_id in histories:
        raise ValueError("target id collides with background history")
    histories[record.target_dynamic_object_id] = np.array(
        record.history_poses, dtype=np.float32, order="C", copy=True
    )
    specs[record.target_dynamic_object_id] = deepcopy(record.footprint_spec)

    return render_observation(
        base_state,
        scene_dynamic_history=histories,
        scene_dynamic_specs=specs,
        static_occupancy=world.static_occupancy,
        sensor_config=sensor_config,
        config=config,
    )


def _validate_formal_mother_world(
    world: OracleWorld,
    *,
    trajectory_handoff: Sop06TrajectoryHandoff | None = None,
) -> None:
    if not isinstance(world, OracleWorld):
        raise TypeError("mother_world must be an OracleWorld")
    joint_version = world.metadata.get(
        "joint_pair_generator_algorithm_version"
    )
    if joint_version is not None:
        if joint_version == JOINT_ENVIRONMENT_PAIR_VERSION:
            raise ValueError(
                f"mother uses retired {JOINT_ENVIRONMENT_PAIR_VERSION}"
            )
        raise ValueError("mother contains unsupported joint-pair identity")
    algorithm = world.metadata.get("generator_algorithm_version")
    if algorithm == SOP05R_GENERATOR_VERSION:
        if not isinstance(trajectory_handoff, Sop06TrajectoryHandoff):
            raise ValueError("SOP05R mother requires a trajectory handoff")
        if (
            trajectory_handoff.generator_algorithm_version
            != SOP05R_GENERATOR_VERSION
            or trajectory_handoff.event_id
            != world.metadata.get("generated_event_id")
            or trajectory_handoff.source_collection_version
            != SOP05R_TRAJECTORY_COLLECTION_VERSION
            or not isinstance(trajectory_handoff.collection_semantic_digest, str)
            or len(trajectory_handoff.collection_semantic_digest) != 64
            or trajectory_handoff.nominal_trajectory.trajectory_id
            != world.metadata.get("nominal_trajectory_id")
        ):
            raise ValueError("SOP05R trajectory handoff differs from mother metadata")
        goal = np.asarray(world.metadata.get("local_goal_world_pose"), dtype=np.float32)
        if (
            trajectory_handoff.shared_goal_world_pose is None
            or goal.shape != (3,)
            or not np.array_equal(
                trajectory_handoff.shared_goal_world_pose,
                goal,
            )
        ):
            raise ValueError("SOP05R trajectory handoff world-frame goal mismatch")
    elif algorithm != SOP05_GENERATOR_ALGORITHM_VERSION:
        raise ValueError(
            "formal SOP06 mother generator_algorithm_version must equal "
            f"{SOP05_GENERATOR_ALGORITHM_VERSION} or {SOP05R_GENERATOR_VERSION}"
        )
    elif trajectory_handoff is not None and (
        trajectory_handoff.generator_algorithm_version
        != SOP05_GENERATOR_ALGORITHM_VERSION
    ):
        raise ValueError("legacy mother trajectory handoff algorithm mismatch")
    target_provenance = world.metadata.get("target_provenance")
    if not isinstance(target_provenance, Mapping):
        raise ValueError("mother target provenance must be a mapping")
    for field, label in (
        ("source_recording_id", "source recording"),
        ("source_session_id", "source session"),
    ):
        value = target_provenance.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"mother target {label} must be non-empty")


def render_sop06_mother_event(
    *,
    record: EventTargetMotionRecord,
    world: OracleWorld,
    base_state: BaseState,
    oracle_context: OracleContext,
    config: Mapping[str, Any],
    sensor_config_override: StructuralBlindSpot | None | object = (
        _USE_WORLD_SENSOR
    ),
    trajectory_handoff: Sop06TrajectoryHandoff | None = None,
) -> RenderedObservation:
    """Render one formal v5 collision mother through the history-only path.

    The optional sensor override supports audited downstream sensor ablations;
    source-world geometry and lineage are still validated before rendering.
    """

    _validate_formal_mother_world(
        world,
        trajectory_handoff=trajectory_handoff,
    )
    return render_sop06_variant(
        record=record,
        world=world,
        base_state=base_state,
        oracle_context=oracle_context,
        config=config,
        sensor_config_override=sensor_config_override,
    )


def _validate_formal_pair_group(
    group: PairedEventGroup,
    *,
    mother_world: OracleWorld,
    expected_paired_config_digest: str,
    trajectory_handoff: Sop06TrajectoryHandoff | None = None,
) -> None:
    if not isinstance(group, PairedEventGroup):
        raise TypeError("group must be a PairedEventGroup")
    _validate_formal_mother_world(
        mother_world,
        trajectory_handoff=trajectory_handoff,
    )
    if group.paired_config_digest != expected_paired_config_digest:
        raise ValueError(
            "group paired_config_digest does not match expected trusted digest"
        )
    if not isinstance(group.pair_group_id, str) or not group.pair_group_id:
        raise ValueError("formal SOP06 pair_group_id must be non-empty")
    by_kind = group.by_kind
    if len(by_kind) != len(group.variants):
        raise ValueError("formal SOP06 group contains duplicate variant kinds")
    expected_coverage = tuple(kind in by_kind for kind in VARIANT_ORDER)
    if group.coverage_mask != expected_coverage:
        raise ValueError("formal SOP06 group coverage mask mismatch")
    if "collision" not in by_kind:
        raise ValueError("formal SOP06 group requires collision mother position")
    absent = set(VARIANT_ORDER) - set(by_kind)
    if set(group.missing_variant_reasons) != absent:
        raise ValueError("formal SOP06 group missing reasons mismatch coverage")
    if any(
        not isinstance(reason, str) or not reason
        for reason in group.missing_variant_reasons.values()
    ):
        raise ValueError(
            "formal SOP06 missing reasons must be non-empty strings"
        )
    if group.is_complete != all(expected_coverage):
        raise ValueError("formal SOP06 group completeness mismatch coverage")
    if group.eligible_for_strict_evaluation != group.is_complete:
        raise ValueError(
            "formal SOP06 strict eligibility must equal completeness"
        )
    expected_kinds = tuple(kind for kind in VARIANT_ORDER if kind in by_kind)
    observed_kinds = tuple(variant.variant_kind for variant in group.variants)
    if observed_kinds != expected_kinds:
        raise ValueError("formal SOP06 variants must follow frozen coverage order")
    expected_coverage_by_kind = {
        kind: expected_coverage[index]
        for index, kind in enumerate(VARIANT_ORDER)
    }
    sop05r_mother = mother_world.metadata.get("generator_algorithm_version") == (
        SOP05R_GENERATOR_VERSION
    )
    for variant in group.variants:
        metadata = variant.world.metadata
        if metadata.get("pair_group_id") != group.pair_group_id:
            raise ValueError("formal SOP06 variant pair_group_id mismatch")
        if metadata.get("paired_config_digest") != group.paired_config_digest:
            raise ValueError("formal SOP06 variant paired_config_digest mismatch")
        if metadata.get("paired_generator_algorithm_version") != (
            PAIRED_GENERATOR_ALGORITHM_VERSION
        ):
            raise ValueError(
                "formal SOP06 paired_generator_algorithm_version must equal "
                f"{PAIRED_GENERATOR_ALGORITHM_VERSION}"
            )
        if metadata.get("pair_group_contract_version") != (
            PAIRED_GROUP_CONTRACT_VERSION
        ):
            raise ValueError(
                "formal SOP06 pair_group_contract_version must equal "
                f"{PAIRED_GROUP_CONTRACT_VERSION}"
            )
        if metadata.get("paired_coverage_mask") != list(expected_coverage):
            raise ValueError("formal SOP06 paired_coverage_mask mismatch")
        if metadata.get("paired_coverage") != expected_coverage_by_kind:
            raise ValueError("formal SOP06 paired_coverage mismatch")
        if metadata.get("paired_missing_variant_reasons") != dict(
            group.missing_variant_reasons
        ):
            raise ValueError(
                "formal SOP06 paired_missing_variant_reasons mismatch"
            )
        if metadata.get("paired_group_complete") is not group.is_complete:
            raise ValueError("formal SOP06 paired_group_complete mismatch")
        if metadata.get("eligible_for_strict_paired_evaluation") is not (
            group.eligible_for_strict_evaluation
        ):
            raise ValueError(
                "formal SOP06 strict-evaluation metadata mismatch"
            )
        if sop05r_mother:
            expected_handoff_metadata = {
                "sop05r_trajectory_collection_version": (
                    trajectory_handoff.source_collection_version
                ),
                "sop05r_trajectory_collection_digest": (
                    trajectory_handoff.collection_semantic_digest
                ),
                "sop05r_shared_goal_world_pose": [
                    float(value)
                    for value in trajectory_handoff.shared_goal_world_pose
                ],
                "sop05r_nominal_trajectory_id": (
                    trajectory_handoff.nominal_trajectory.trajectory_id
                ),
                "sop05r_alternative_trajectory_ids": [
                    item.trajectory_id
                    for item in trajectory_handoff.alternative_trajectories
                ],
            }
            if any(
                metadata.get(key) != value
                for key, value in expected_handoff_metadata.items()
            ):
                raise ValueError("formal SOP06 SOP05R trajectory handoff metadata mismatch")
        joint_version = metadata.get("joint_pair_generator_algorithm_version")
        if joint_version is not None:
            if joint_version == JOINT_ENVIRONMENT_PAIR_VERSION:
                raise ValueError(
                    f"paired variant uses retired {JOINT_ENVIRONMENT_PAIR_VERSION}"
                )
            raise ValueError("paired variant contains unsupported joint-pair identity")


def _render_formal_pair_group(
    *,
    group: PairedEventGroup,
    mother_record: EventTargetMotionRecord,
    mother_world: OracleWorld,
    base_state: BaseState,
    oracle_context: OracleContext,
    config: Mapping[str, Any],
    expected_paired_config_digest: str,
    audit_certified: bool,
    trajectory_handoff: Sop06TrajectoryHandoff | None = None,
) -> RenderedSop06Group:
    _validate_formal_pair_group(
        group,
        mother_world=mother_world,
        expected_paired_config_digest=expected_paired_config_digest,
        trajectory_handoff=trajectory_handoff,
    )
    observations = tuple(
        render_sop06_paired_variant(
            mother_record=mother_record,
            mother_world=mother_world,
            variant=variant,
            base_state=base_state,
            oracle_context=oracle_context,
            config=config,
            expected_paired_config_digest=expected_paired_config_digest,
            trajectory_handoff=trajectory_handoff,
        )
        for variant in group.variants
    )
    return RenderedSop06Group(
        pair_group_id=group.pair_group_id,
        variant_kinds=tuple(variant.variant_kind for variant in group.variants),
        observations=observations,
        coverage_mask=group.coverage_mask,
        is_complete=group.is_complete,
        audit_certified=audit_certified,
    )


def render_sop06_partial_pair_group(
    *,
    group: PairedEventGroup,
    mother_record: EventTargetMotionRecord,
    mother_world: OracleWorld,
    base_state: BaseState,
    oracle_context: OracleContext,
    config: Mapping[str, Any],
    expected_paired_config_digest: str,
    trajectory_handoff: Sop06TrajectoryHandoff | None = None,
) -> RenderedSop06Group:
    """Render training variants without ever certifying sixpack completeness."""

    return _render_formal_pair_group(
        group=group,
        mother_record=mother_record,
        mother_world=mother_world,
        base_state=base_state,
        oracle_context=oracle_context,
        config=config,
        expected_paired_config_digest=expected_paired_config_digest,
        audit_certified=False,
        trajectory_handoff=trajectory_handoff,
    )


def render_sop06_complete_audit_group(
    *,
    group: PairedEventGroup,
    mother_record: EventTargetMotionRecord,
    mother_world: OracleWorld,
    base_state: BaseState,
    oracle_context: OracleContext,
    config: Mapping[str, Any],
    expected_paired_config_digest: str,
    trajectory_handoff: Sop06TrajectoryHandoff | None = None,
) -> RenderedSop06Group:
    """Render a conditional complete sixpack for audit/ablation only."""

    if not isinstance(group, PairedEventGroup):
        raise TypeError("group must be a PairedEventGroup")
    if (
        not group.is_complete
        or group.coverage_mask != (True,) * len(VARIANT_ORDER)
        or not group.eligible_for_strict_evaluation
    ):
        raise ValueError(
            "complete audit requires a complete six-position paired group"
        )
    return _render_formal_pair_group(
        group=group,
        mother_record=mother_record,
        mother_world=mother_world,
        base_state=base_state,
        oracle_context=oracle_context,
        config=config,
        expected_paired_config_digest=expected_paired_config_digest,
        audit_certified=True,
        trajectory_handoff=trajectory_handoff,
    )
