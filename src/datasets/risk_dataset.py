"""Schema-v3 RiskSample assembly with an explicit input/label boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from numbers import Real
from typing import Any, Mapping

import numpy as np

from src.contracts import (
    DYNAMIC_OBJECT_TYPES,
    HISTORY_CHANNELS,
    POSE_TIME_LAYOUT_VERSION,
    SCHEMA_VERSION,
    TRAJECTORY_CHANNELS,
    BaseState,
    GridSpec,
    LocalTrajectory,
    OracleContext,
    OracleWorld,
    RiskSample,
    assert_no_oracle_leakage,
    build_grid_spec,
    validate_risk_sample,
)
from src.generation.dynamic_object_transplant import footprint_from_spec
from src.generation.event_sampler import SOP05_GENERATOR_ALGORITHM_VERSION
from src.generation.event_target_motion_shard import EventTargetMotionRecord
from src.generation.observation_renderer import (
    RENDERER_LAYOUT_VERSION,
    render_observation,
)
from src.generation.paired_variants import (
    JOINT_ENVIRONMENT_PAIR_VERSION,
    PAIRED_GENERATOR_ALGORITHM_VERSION,
    PAIRED_GROUP_CONTRACT_VERSION,
    PairedVariant,
)
from src.generation.risk_gt import (
    RISK_GT_VERSION,
    compute_hidden_risk_gt,
    resolve_no_object_clearance_sentinel,
)
from src.generation.sop06_pipeline import render_sop06_paired_variant
from src.generation.structural_blindspot import StructuralBlindSpot
from src.geometry import (
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint,
)
from src.utils.seeding import stable_digest


_RISK_CONFIG_KEYS = frozenset(
    {"sigma_distance_m", "sigma_time_s", "near_miss_distance_m"}
)
_METADATA_KEYS = frozenset(
    {"schema_version", "renderer", "trajectory_id", "provenance", "label_audit"}
)
_RENDERER_METADATA_KEYS = frozenset(
    {
        "renderer_layout_version",
        "base_state_id",
        "sensor_config_digest",
        "static_occupancy_digest",
    }
)
_LABEL_AUDIT_KEYS = frozenset(
    {
        "risk_gt_version",
        "pose_time_layout_version",
        "critical_object_id",
        "critical_object_type",
        "time_to_min_clearance_s",
        "has_hidden_target",
    }
)
_FORBIDDEN_METADATA_KEY_TOKENS = (
    "future",
    "oracle",
    "clearance_sequence",
    "dynamic_object_trajectories",
    "hidden_object_ids",
)
_RISK_INPUT_ADAPTER_VERSION = "sop06_variant_to_risk_input_v1"


@dataclass(frozen=True)
class RiskBuildInput:
    """One sample source with observation history separated from label future."""

    sample_id: str
    pair_group_id: str
    event_type: str
    base_state: BaseState
    trajectory: LocalTrajectory
    oracle_world: OracleWorld
    observed_static_occupancy: np.ndarray
    scene_dynamic_history: Mapping[str, np.ndarray]
    scene_dynamic_specs: Mapping[str, dict[str, object]]
    hidden_object_ids: tuple[str, ...]
    sensor_config: StructuralBlindSpot | None
    provenance: Mapping[str, object]


def _canonical_config_digest(config: Mapping[str, object]) -> str:
    try:
        payload = json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("base_config must be finite canonical JSON") from exc
    return stable_digest(payload, size=16)


def _validate_formal_sop06_variant_identity(
    mother_world: OracleWorld,
    variant: PairedVariant,
) -> None:
    if not isinstance(mother_world.metadata, Mapping):
        raise ValueError("mother_world metadata must be a mapping")
    if not isinstance(variant.world.metadata, Mapping):
        raise ValueError("paired world metadata must be a mapping")
    for label, metadata in (
        ("mother", mother_world.metadata),
        ("paired variant", variant.world.metadata),
    ):
        joint_version = metadata.get("joint_pair_generator_algorithm_version")
        if joint_version is not None:
            if joint_version == JOINT_ENVIRONMENT_PAIR_VERSION:
                raise ValueError(
                    f"{label} uses retired {JOINT_ENVIRONMENT_PAIR_VERSION}"
                )
            raise ValueError(f"{label} contains unsupported joint identity")
    if mother_world.metadata.get("generator_algorithm_version") != (
        SOP05_GENERATOR_ALGORITHM_VERSION
    ):
        raise ValueError(
            "mother generator_algorithm_version must equal "
            f"{SOP05_GENERATOR_ALGORITHM_VERSION}"
        )
    metadata = variant.world.metadata
    if metadata.get("paired_generator_algorithm_version") != (
        PAIRED_GENERATOR_ALGORITHM_VERSION
    ):
        raise ValueError(
            "paired_generator_algorithm_version must equal "
            f"{PAIRED_GENERATOR_ALGORITHM_VERSION}"
        )
    if metadata.get("pair_group_contract_version") != (
        PAIRED_GROUP_CONTRACT_VERSION
    ):
        raise ValueError(
            "pair_group_contract_version must equal "
            f"{PAIRED_GROUP_CONTRACT_VERSION}"
        )


def _copy_sop06_scene_history(
    *,
    base_state: BaseState,
    oracle_context: OracleContext,
    variant: PairedVariant,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    histories: dict[str, np.ndarray] = {}
    specs: dict[str, dict[str, object]] = {}
    for object_id in sorted(base_state.dynamic_object_ids):
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
        if object_id in histories:
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
    if variant.target is not None:
        target_id = variant.target.target_dynamic_object_id
        if target_id in histories:
            raise ValueError("paired target id collides with scene history")
        histories[target_id] = np.array(
            variant.target.history_poses,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        specs[target_id] = deepcopy(variant.target.footprint_spec)
    return histories, specs


def build_risk_input_from_sop06_variant(
    *,
    mother_record: EventTargetMotionRecord,
    mother_world: OracleWorld,
    variant: PairedVariant,
    base_state: BaseState,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    base_config: Mapping[str, object],
    expected_paired_config_digest: str,
    source_session_id: str,
    seed_namespace: str,
) -> RiskBuildInput:
    """Adapt one formally validated SOP06 variant without leaking label future."""

    if not isinstance(mother_record, EventTargetMotionRecord):
        raise TypeError("mother_record must be an EventTargetMotionRecord")
    if not isinstance(mother_world, OracleWorld):
        raise TypeError("mother_world must be an OracleWorld")
    if not isinstance(variant, PairedVariant):
        raise TypeError("variant must be a PairedVariant")
    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(trajectory, LocalTrajectory):
        raise TypeError("trajectory must be a LocalTrajectory")
    if not isinstance(oracle_context, OracleContext):
        raise TypeError("oracle_context must be an OracleContext")
    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    config = dict(base_config)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"base_config schema_version must be {SCHEMA_VERSION}")
    source_session_id = _require_nonempty_string(
        source_session_id, name="source_session_id"
    )
    seed_namespace = _require_nonempty_string(
        seed_namespace, name="seed_namespace"
    )
    if trajectory.trajectory_id != mother_record.trajectory_id:
        raise ValueError("trajectory_id does not match mother record")
    _validate_formal_sop06_variant_identity(mother_world, variant)

    # This is the authoritative SOP06 validation/render boundary.  It checks
    # record/world/context joins, trusted config identity, retired versions,
    # paired lineage, target history/future digests, and skeleton equality.
    render_sop06_paired_variant(
        mother_record=mother_record,
        mother_world=mother_world,
        variant=variant,
        base_state=base_state,
        oracle_context=oracle_context,
        config=config,
        expected_paired_config_digest=expected_paired_config_digest,
    )
    blind_spot_config = variant.world.blind_spot_config
    if not isinstance(blind_spot_config, Mapping) or (
        blind_spot_config.get("kind") != "environment"
    ):
        raise ValueError("formal v5 risk adapter requires an environment variant")

    histories, specs = _copy_sop06_scene_history(
        base_state=base_state,
        oracle_context=oracle_context,
        variant=variant,
    )
    pair_group_id = _require_nonempty_string(
        variant.world.metadata.get("pair_group_id"), name="pair_group_id"
    )
    target_footprint = mother_record.footprint_spec.get("footprint")
    if not isinstance(target_footprint, Mapping):
        raise ValueError("mother target footprint must be a mapping")
    target_footprint_kind = _require_nonempty_string(
        target_footprint.get("kind"), name="target_footprint_kind"
    )
    base_config_digest = _canonical_config_digest(config)
    sample_id = f"{base_state.split}-" + stable_digest(
        _RISK_INPUT_ADAPTER_VERSION,
        pair_group_id,
        base_state.state_id,
        trajectory.trajectory_id,
        variant.variant_kind,
        variant.world.world_id,
        expected_paired_config_digest,
        base_config_digest,
        seed_namespace,
        size=12,
    )
    provenance = _canonical_metadata_copy(
        {
            "risk_input_adapter_version": _RISK_INPUT_ADAPTER_VERSION,
            "source_recording_id": base_state.recording_id,
            "session_id": source_session_id,
            "source_snippet_id": mother_record.source_snippet_id,
            "source_object_id": mother_record.source_object_id,
            "seed_namespace": seed_namespace,
            "target_object_type": mother_record.object_type,
            "target_footprint_kind": target_footprint_kind,
            "target_type_policy_digest": (
                mother_record.target_type_policy_digest
            ),
            "blind_spot_type": blind_spot_config["kind"],
            "generator_algorithm_version": (
                mother_world.metadata["generator_algorithm_version"]
            ),
            "paired_generator_algorithm_version": (
                variant.world.metadata[
                    "paired_generator_algorithm_version"
                ]
            ),
            "pair_group_contract_version": (
                variant.world.metadata["pair_group_contract_version"]
            ),
            "paired_config_digest": expected_paired_config_digest,
            "base_config_digest": base_config_digest,
            "variant_kind": variant.variant_kind,
            "variant_random_seed": variant.world.random_seed,
        },
        name="provenance",
    )
    hidden_object_ids = (
        ()
        if variant.target is None
        else (variant.target.target_dynamic_object_id,)
    )
    source = RiskBuildInput(
        sample_id=sample_id,
        pair_group_id=pair_group_id,
        event_type=variant.variant_kind,
        base_state=base_state,
        trajectory=trajectory,
        oracle_world=variant.world,
        observed_static_occupancy=np.array(
            variant.world.static_occupancy,
            dtype=np.float32,
            order="C",
            copy=True,
        ),
        scene_dynamic_history=histories,
        scene_dynamic_specs=specs,
        hidden_object_ids=hidden_object_ids,
        sensor_config=None,
        provenance=provenance,
    )
    _validate_source_join(source)
    return source


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _query_map(
    value: Any,
    *,
    name: str,
    grid: GridSpec,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an np.ndarray")
    if value.dtype != np.float32:
        raise TypeError(f"{name} dtype must be float32")
    if value.shape != (grid.height, grid.width):
        raise ValueError(
            f"{name} shape must be ({grid.height}, {grid.width})"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def build_trajectory_channels(
    trajectory: LocalTrajectory,
    grid: GridSpec,
) -> np.ndarray:
    """Stack the four frozen query maps without conversion or reordering."""

    if not isinstance(trajectory, LocalTrajectory):
        raise TypeError("trajectory must be a LocalTrajectory")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    by_channel = {
        "swept_volume_mask": trajectory.swept_mask,
        "time_to_arrival_map": trajectory.tta_map,
        "braking_margin_map": trajectory.braking_map,
        "centerline_map": trajectory.centerline_map,
    }
    if tuple(by_channel) != TRAJECTORY_CHANNELS:
        raise RuntimeError("trajectory query-map order violates the frozen contract")
    arrays = [
        _query_map(by_channel[name], name=name, grid=grid)
        for name in TRAJECTORY_CHANNELS
    ]
    return np.ascontiguousarray(np.stack(arrays, axis=0), dtype=np.float32)


def _validate_metadata_value(value: object, *, path: str) -> None:
    if isinstance(value, np.ndarray):
        raise TypeError(f"metadata {path} must not contain ndarray payloads")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"metadata {path} keys must be non-empty strings")
            lowered = key.lower()
            if any(token in lowered for token in _FORBIDDEN_METADATA_KEY_TOKENS):
                raise ValueError(f"metadata {path}.{key} contains a forbidden payload key")
            _validate_metadata_value(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_metadata_value(child, path=f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"metadata {path} must contain only finite values")
        return
    if isinstance(value, (np.generic, Real)):
        raise TypeError(f"metadata {path} must contain JSON-native scalar values")
    raise TypeError(f"metadata {path} contains a non-JSON value")


def _canonical_metadata_copy(value: Mapping[str, object], *, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    copied = dict(value)
    _validate_metadata_value(copied, path=name)
    return json.loads(
        json.dumps(
            copied,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _normalized_risk_config(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("risk_config must be a mapping")
    config = dict(value)
    if set(config) != _RISK_CONFIG_KEYS:
        raise ValueError(
            f"risk_config keys must be exactly {sorted(_RISK_CONFIG_KEYS)}"
        )
    return config


def _validate_source_join(source: RiskBuildInput) -> None:
    _require_nonempty_string(source.sample_id, name="sample_id")
    _require_nonempty_string(source.pair_group_id, name="pair_group_id")
    _require_nonempty_string(source.event_type, name="event_type")
    if not isinstance(source.base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(source.trajectory, LocalTrajectory):
        raise TypeError("trajectory must be a LocalTrajectory")
    if not isinstance(source.oracle_world, OracleWorld):
        raise TypeError("oracle_world must be an OracleWorld")
    if source.oracle_world.base_state_id != source.base_state.state_id:
        raise ValueError("oracle_world and base_state IDs must match")
    if not isinstance(source.scene_dynamic_history, Mapping) or not isinstance(
        source.scene_dynamic_specs, Mapping
    ):
        raise TypeError("scene history and specs must be mappings")
    history_ids = set(source.scene_dynamic_history)
    spec_ids = set(source.scene_dynamic_specs)
    world_ids = set(source.oracle_world.dynamic_object_trajectories)
    if history_ids != spec_ids or history_ids != world_ids:
        raise ValueError("scene history/spec IDs must match oracle_world object IDs")
    for object_id in sorted(history_ids):
        if source.scene_dynamic_specs[object_id] != source.oracle_world.dynamic_object_specs[
            object_id
        ]:
            raise ValueError("scene and oracle_world footprint specs must match")
    if not isinstance(source.hidden_object_ids, tuple):
        raise TypeError("hidden_object_ids must be an explicit tuple")
    if not set(source.hidden_object_ids).issubset(history_ids):
        raise ValueError("hidden_object_ids must have current history and specs")


def _validate_declared_hidden_visibility(
    source: RiskBuildInput,
    *,
    rendered_history: np.ndarray,
    grid: GridSpec,
) -> None:
    current_visible = rendered_history[
        -1, HISTORY_CHANNELS.index("past_visible_mask")
    ] > 0.5
    for object_id in sorted(source.hidden_object_ids):
        footprint = footprint_from_spec(source.scene_dynamic_specs[object_id])
        current_pose = source.scene_dynamic_history[object_id][-1]
        footprint_mask = rasterize_footprint(footprint, current_pose, grid)
        if not bool(np.any(footprint_mask)):
            raise ValueError(f"hidden object {object_id!r} has no current grid footprint")
        if bool(np.any(footprint_mask & current_visible)):
            raise ValueError(f"hidden object {object_id!r} is currently visible")


def validate_risk_sample_for_publication(
    sample: RiskSample,
    grid: GridSpec,
) -> None:
    """Validate model arrays, finite labels, and recursive metadata isolation."""

    if not isinstance(sample, RiskSample):
        raise TypeError("sample must be a RiskSample")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    validate_risk_sample(sample, grid)
    assert_no_oracle_leakage(RiskSample)
    for name in ("sample_id", "split", "base_state_id", "pair_group_id", "event_type"):
        _require_nonempty_string(getattr(sample, name), name=name)
    if isinstance(sample.collision_label, (bool, np.bool_)) or not isinstance(
        sample.collision_label, (int, np.integer)
    ):
        raise TypeError("collision_label must be an integer")
    if isinstance(sample.near_miss, (bool, np.bool_)) or not isinstance(
        sample.near_miss, (int, np.integer)
    ):
        raise TypeError("near_miss must be an integer")
    severity = _finite_float(sample.risk_severity, name="risk_severity")
    minimum = _finite_float(sample.min_clearance, name="min_clearance")
    first_collision = sample.first_collision_time
    if first_collision is not None:
        first_collision = _finite_float(
            first_collision, name="first_collision_time"
        )
        if first_collision <= 0.0:
            raise ValueError("first_collision_time must be positive")
    if sample.collision_label == 1:
        if first_collision is None:
            raise ValueError("collision requires first_collision_time")
        if severity != 1.0:
            raise ValueError("collision requires risk_severity == 1")
        if minimum > 0.0:
            raise ValueError("collision requires min_clearance <= 0")
    elif first_collision is not None:
        raise ValueError("noncollision requires first_collision_time=None")
    elif minimum <= 0.0:
        raise ValueError("noncollision requires positive min_clearance")

    if not isinstance(sample.metadata, dict):
        raise TypeError("metadata must be a dict")
    if set(sample.metadata) != _METADATA_KEYS:
        raise ValueError(f"metadata keys must be exactly {sorted(_METADATA_KEYS)}")
    _validate_metadata_value(sample.metadata, path="metadata")
    if sample.metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"metadata schema_version must be {SCHEMA_VERSION}")
    if sample.metadata["trajectory_id"] == "" or not isinstance(
        sample.metadata["trajectory_id"], str
    ):
        raise ValueError("metadata trajectory_id must be a non-empty string")

    renderer = sample.metadata["renderer"]
    if not isinstance(renderer, dict) or set(renderer) != _RENDERER_METADATA_KEYS:
        raise ValueError("renderer metadata keys violate the frozen contract")
    if renderer["renderer_layout_version"] != RENDERER_LAYOUT_VERSION:
        raise ValueError("renderer layout version mismatch")
    if renderer["base_state_id"] != sample.base_state_id:
        raise ValueError("renderer base_state_id mismatch")
    if not isinstance(sample.metadata["provenance"], dict):
        raise TypeError("provenance metadata must be a dict")

    audit = sample.metadata["label_audit"]
    if not isinstance(audit, dict) or set(audit) != _LABEL_AUDIT_KEYS:
        raise ValueError("label_audit keys violate the frozen contract")
    if audit["risk_gt_version"] != RISK_GT_VERSION:
        raise ValueError("risk_gt_version mismatch")
    if audit["pose_time_layout_version"] != POSE_TIME_LAYOUT_VERSION:
        raise ValueError("pose_time_layout_version mismatch")
    has_hidden = audit["has_hidden_target"]
    if not isinstance(has_hidden, bool):
        raise TypeError("has_hidden_target must be bool")
    critical_id = audit["critical_object_id"]
    critical_type = audit["critical_object_type"]
    time_to_minimum = audit["time_to_min_clearance_s"]
    if has_hidden:
        _require_nonempty_string(critical_id, name="critical_object_id")
        if critical_type not in DYNAMIC_OBJECT_TYPES:
            raise ValueError("critical_object_type is invalid")
        time_to_minimum = _finite_float(
            time_to_minimum, name="time_to_min_clearance_s"
        )
        if time_to_minimum <= 0.0:
            raise ValueError("time_to_min_clearance_s must be positive")
    else:
        if any(value is not None for value in (critical_id, critical_type, time_to_minimum)):
            raise ValueError("empty hidden set requires empty label_audit identity")
        if sample.collision_label != 0 or sample.near_miss != 0 or severity != 0.0:
            raise ValueError("empty hidden set requires zero risk labels")
        sentinel = resolve_no_object_clearance_sentinel(grid)
        if minimum != sentinel:
            raise ValueError("empty hidden set requires the grid-diagonal sentinel")


def build_risk_sample(
    source: RiskBuildInput,
    *,
    base_config: Mapping[str, object],
    risk_config: Mapping[str, object],
) -> RiskSample:
    """Render history-only inputs and independently compute oracle-future labels."""

    if not isinstance(source, RiskBuildInput):
        raise TypeError("source must be a RiskBuildInput")
    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    base_config_dict = dict(base_config)
    if base_config_dict.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"base_config schema_version must be {SCHEMA_VERSION}")
    normalized_risk = _normalized_risk_config(risk_config)
    grid = build_grid_spec(base_config_dict)
    _validate_source_join(source)

    rendered = render_observation(
        source.base_state,
        scene_dynamic_history=source.scene_dynamic_history,
        scene_dynamic_specs=source.scene_dynamic_specs,
        static_occupancy=source.observed_static_occupancy,
        sensor_config=source.sensor_config,
        config=base_config_dict,
    )
    if not np.array_equal(
        source.observed_static_occupancy,
        source.oracle_world.static_occupancy,
    ):
        raise ValueError("observed and oracle_world static occupancy must match")
    _validate_declared_hidden_visibility(
        source,
        rendered_history=rendered.bev_history,
        grid=grid,
    )

    robot_config = base_config_dict.get("robot")
    if not isinstance(robot_config, Mapping):
        raise TypeError("base_config.robot must be a mapping")
    robot_footprint = inflate_footprint(
        RectangleFootprint(
            robot_config.get("length_m"),
            robot_config.get("width_m"),
        ),
        robot_config.get("inflation_m"),
    )
    bev_config = base_config_dict.get("bev")
    if not isinstance(bev_config, Mapping):
        raise TypeError("base_config.bev must be a mapping")
    labels = compute_hidden_risk_gt(
        source.trajectory,
        source.oracle_world,
        hidden_object_ids=source.hidden_object_ids,
        robot_footprint=robot_footprint,
        grid=grid,
        future_dt_s=bev_config.get("future_dt_s"),
        sigma_distance_m=normalized_risk["sigma_distance_m"],
        sigma_time_s=normalized_risk["sigma_time_s"],
        near_miss_distance_m=normalized_risk["near_miss_distance_m"],
    )
    trajectory_channels = build_trajectory_channels(source.trajectory, grid)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "renderer": _canonical_metadata_copy(
            rendered.metadata, name="renderer"
        ),
        "trajectory_id": _require_nonempty_string(
            source.trajectory.trajectory_id, name="trajectory_id"
        ),
        "provenance": _canonical_metadata_copy(
            source.provenance, name="provenance"
        ),
        "label_audit": {
            "risk_gt_version": RISK_GT_VERSION,
            "pose_time_layout_version": labels.pose_time_layout_version,
            "critical_object_id": labels.critical_object_id,
            "critical_object_type": labels.critical_object_type,
            "time_to_min_clearance_s": labels.time_to_min_clearance,
            "has_hidden_target": labels.has_hidden_target,
        },
    }
    sample = RiskSample(
        sample_id=source.sample_id,
        split=source.base_state.split,
        base_state_id=source.base_state.state_id,
        pair_group_id=source.pair_group_id,
        event_type=source.event_type,
        bev_history=np.array(
            rendered.bev_history, dtype=np.float32, order="C", copy=True
        ),
        state_channels=np.array(
            rendered.state_channels, dtype=np.float32, order="C", copy=True
        ),
        trajectory_channels=trajectory_channels,
        robot_state=np.array(
            source.base_state.robot_state,
            dtype=np.float32,
            order="C",
            copy=True,
        ),
        collision_label=labels.collision_label,
        risk_severity=labels.risk_severity,
        min_clearance=labels.min_clearance,
        near_miss=labels.near_miss,
        first_collision_time=labels.first_collision_time,
        metadata=metadata,
    )
    validate_risk_sample_for_publication(sample, grid)
    return sample
