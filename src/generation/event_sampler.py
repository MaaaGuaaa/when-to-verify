"""Event-centred orchestration for typed hidden dynamic-object worlds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.contracts import (
    SCHEMA_VERSION,
    BaseState,
    LocalTrajectory,
    OracleContext,
    OracleWorld,
    build_grid_spec,
    validate_base_state,
    validate_oracle_context,
)
from src.datasets.snippet_library import SnippetLibrary
from src.geometry import (
    Footprint,
    RectangleFootprint,
    inflate_footprint,
    intersects,
    points_in_grid,
    rasterize_footprint,
    raycast_visibility,
    trajectory_signed_clearances,
)
from src.utils.seeding import derive_seed, make_rng, stable_digest

from .dynamic_object_transplant import (
    TargetTypePolicy,
    TransplantError,
    TransplantedDynamicObject,
    footprint_from_spec,
    normalize_target_type_policy,
    sample_motion_snippet,
    transplant_reachability_candidate,
    transplant_snippet,
)
from .blind_reachability import (
    BLIND_REACHABILITY_ALGORITHM_VERSION,
    ReachabilityIdentity,
    build_reachability_candidate,
    candidate_queries_mask,
    scheduled_crossing_directions,
    triage_chord,
)
from .blind_region import (
    build_blind_region,
    build_footprint_center_mask,
    check_exact_hidden_pose,
)
from .causal_occluder import (
    build_causal_occluder_context,
    build_causal_occluder_schedule,
    normalize_causal_occluder_config,
    propose_causal_occluder,
)
from .event_target_motion_shard import (
    EVENT_TARGET_MOTION_LAYOUT_VERSION,
    EventTargetMotionRecord,
    build_event_target_motion_world_metadata,
    compute_footprint_spec_digest,
    compute_motion_array_digest,
    create_event_target_motion_record,
    validate_event_target_motion_world_join,
)
from .occluder_sampler import (
    JointOccluderParameters,
    OccluderCollisionSweep,
    OccluderPlacement,
    OccluderSamplingError,
    align_environment_occluder_to_target_los,
    build_joint_occluder_schedule,
    normalize_occluder_config,
    propose_environment_occluder_geometry,
    synchronized_sweeps_intersect,
    swept_footprint_intersects_occupancy,
    validate_environment_occluder_target,
)
from .structural_blindspot import (
    StructuralBlindSpot,
    build_structural_visibility,
    footprint_visibility_sequence,
    has_continuous_emergence,
)
from .robot_sweep_cache import (
    RobotSweepCache,
    prepare_robot_collision_sweep_bundle,
)


_EVENT_TYPES = ("environment", "structural", "mixed")
SOP05_GENERATOR_ALGORITHM_VERSION = BLIND_REACHABILITY_ALGORITHM_VERSION
SOP05_GENERATOR_SCHEMA_VERSION = "3.0.0"
_GENERATED_EVENT_ID_DOMAIN = "sop05-generated-event-lineage-id-v1"
_GENERATED_EVENT_IDENTITY_VERSION = "sop05_generated_event_lineage_v1"
_WORLD_ID_DOMAIN = "sop05-generated-world-id-v1"
_WORLD_IDENTITY_VERSION = "sop05_generated_world_identity_v1"
_FLOAT32_LE_DTYPE_TOKEN = "<f4"
_ARRAY_ORDER_TOKEN = "C"


class GeneratorConfigError(ValueError):
    """Raised for an unknown, incomplete, or nonphysical SOP-05 config."""


class _EventRejection(Exception):
    def __init__(
        self,
        reason: str,
        *,
        occluder_candidate_rejection_reasons: Mapping[str, int] | None = None,
    ):
        super().__init__(reason)
        self.reason = reason
        self.occluder_candidate_rejection_reasons = dict(
            occluder_candidate_rejection_reasons or {}
        )


@dataclass(frozen=True)
class GeneratedEvent:
    """One accepted oracle world and its auditable target/visibility state."""

    generated_event_id: str
    event_kind: str
    world: OracleWorld
    target: TransplantedDynamicObject
    target_motion_record: EventTargetMotionRecord
    visibility_sequence: np.ndarray
    target_visibility_history: np.ndarray
    conflict_time_s: float
    conflict_index: int


@dataclass(frozen=True)
class EventGenerationReport:
    """Accepted events plus deterministic attempt/rejection statistics."""

    events: tuple[GeneratedEvent, ...]
    summary: dict[str, object]


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise GeneratorConfigError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise GeneratorConfigError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise GeneratorConfigError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise GeneratorConfigError(f"{name} must be positive")
    return result


def _range_pair(
    value: Any,
    *,
    name: str,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise GeneratorConfigError(f"{name} must contain [minimum, maximum]")
    lower = _finite_real(value[0], name=f"{name}[0]")
    upper = _finite_real(value[1], name=f"{name}[1]")
    if not lower_bound <= lower <= upper <= upper_bound:
        raise GeneratorConfigError(
            f"{name} must lie within [{lower_bound}, {upper_bound}]"
        )
    return lower, upper


def _normalize_event_weights(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(_EVENT_TYPES):
        raise GeneratorConfigError(
            "event_type_weights must contain environment/structural/mixed"
        )
    parsed = {
        event_type: _finite_real(
            value[event_type], name=f"event_type_weights.{event_type}"
        )
        for event_type in _EVENT_TYPES
    }
    if any(weight < 0.0 for weight in parsed.values()):
        raise GeneratorConfigError("event type weights must be non-negative")
    total = sum(parsed.values())
    if total <= 0.0:
        raise GeneratorConfigError("at least one event type weight must be positive")
    if np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
        total = 1.0
    return {event_type: parsed[event_type] / total for event_type in _EVENT_TYPES}


def build_event_type_schedule(
    weights: Mapping[str, Any],
    *,
    event_count: int,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """Allocate a deterministic batch schedule matching configured weights."""

    normalized = _normalize_event_weights(weights)
    count = _positive_integer(event_count, name="event_count")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    expected = np.asarray(
        [normalized[event_type] * count for event_type in _EVENT_TYPES],
        dtype=np.float64,
    )
    allocated = np.floor(expected).astype(np.int64)
    remaining = count - int(allocated.sum())
    order = sorted(
        range(len(_EVENT_TYPES)),
        key=lambda index: (-(expected[index] - allocated[index]), index),
    )
    for index in order[:remaining]:
        allocated[index] += 1
    schedule = [
        event_type
        for event_type, type_count in zip(_EVENT_TYPES, allocated)
        for _ in range(int(type_count))
    ]
    permutation = rng.permutation(len(schedule))
    return tuple(schedule[int(index)] for index in permutation)


def _normalize_structural_config(value: Any) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "forward_fov_deg",
        "range_m",
        "optional_blind_sectors",
    }:
        raise GeneratorConfigError(
            "structural_fov must contain forward_fov_deg, range_m, and optional_blind_sectors"
        )
    fovs = value["forward_fov_deg"]
    ranges = value["range_m"]
    sectors = value["optional_blind_sectors"]
    if not isinstance(fovs, (list, tuple)) or not fovs:
        raise GeneratorConfigError("forward_fov_deg must be a non-empty sequence")
    if not isinstance(ranges, (list, tuple)) or not ranges:
        raise GeneratorConfigError("range_m must be a non-empty sequence")
    normalized_fovs = tuple(
        _finite_real(item, name="structural_fov.forward_fov_deg") for item in fovs
    )
    normalized_ranges = tuple(
        _finite_real(item, name="structural_fov.range_m") for item in ranges
    )
    if any(item not in {160.0, 180.0, 220.0} for item in normalized_fovs):
        raise GeneratorConfigError("forward_fov_deg values must be 160/180/220")
    if any(item not in {6.0, 8.0, 10.0} for item in normalized_ranges):
        raise GeneratorConfigError("range_m values must be 6/8/10 metres")
    if not isinstance(sectors, (list, tuple)):
        raise GeneratorConfigError("optional_blind_sectors must be a sequence")
    normalized_sectors = []
    for sector in sectors:
        try:
            checked = StructuralBlindSpot(
                forward_fov_deg=normalized_fovs[0],
                range_m=normalized_ranges[0],
                blind_sectors=(dict(sector),),
            )
        except (TypeError, ValueError) as exc:
            raise GeneratorConfigError(str(exc)) from exc
        normalized_sectors.append(dict(checked.blind_sectors[0]))
    return {
        "forward_fov_deg": normalized_fovs,
        "range_m": normalized_ranges,
        "optional_blind_sectors": tuple(normalized_sectors),
    }


def normalize_generator_config(config: Mapping[str, Any]) -> dict[str, object]:
    """Strictly validate only the SOP-05 domain config, without global drift."""

    if not isinstance(config, Mapping):
        raise GeneratorConfigError("generator config must be a mapping")
    expected = {
        "schema_version",
        "production_event_kind",
        "target_type_policy",
        "conflict_time_range_s",
        "max_local_curvature_per_m",
        "crossing_angle_max_deg",
        "time_scale_range",
        "min_contiguous_visible_frames",
        "occluders",
        "blind_reachability",
    }
    if set(config) != expected:
        unknown = sorted(set(config) - expected)
        missing = sorted(expected - set(config))
        raise GeneratorConfigError(
            f"generator config keys mismatch; unknown={unknown}, missing={missing}"
        )
    if config["schema_version"] != SOP05_GENERATOR_SCHEMA_VERSION:
        raise GeneratorConfigError(
            f"schema_version must be {SOP05_GENERATOR_SCHEMA_VERSION}"
        )
    if config["production_event_kind"] != "environment":
        raise GeneratorConfigError(
            "production_event_kind must be environment"
        )
    blind = config["blind_reachability"]
    blind_expected = {
        "algorithm_version",
        "obstacle_proposals_per_trajectory",
        "interaction_range_m",
        "bearing_bin_count",
        "yaw_step_deg",
        "crossing_angle_step_deg",
        "minimum_shadow_center_cells",
        "chord_deviation_fastpath_m",
        "unresolved_exact_fallback_per_anchor",
    }
    if not isinstance(blind, Mapping) or set(blind) != blind_expected:
        raise GeneratorConfigError(
            "blind_reachability keys do not match the frozen v5 schema"
        )
    if blind["algorithm_version"] != SOP05_GENERATOR_ALGORITHM_VERSION:
        raise GeneratorConfigError(
            "blind_reachability.algorithm_version must be "
            f"{SOP05_GENERATOR_ALGORITHM_VERSION}"
        )
    occluders = config["occluders"]
    occluder_expected = {"types", "wall", "shelf", "pillar"}
    if not isinstance(occluders, Mapping) or set(occluders) != occluder_expected:
        raise GeneratorConfigError("occluders keys mismatch for the v5 schema")
    try:
        policy = (
            config["target_type_policy"]
            if isinstance(config["target_type_policy"], TargetTypePolicy)
            else normalize_target_type_policy(config["target_type_policy"])
        )
        causal_occluders = normalize_causal_occluder_config(
            {
                **occluders,
                "interaction_range_m": blind["interaction_range_m"],
                "bearing_bin_count": blind["bearing_bin_count"],
                "yaw_step_deg": blind["yaw_step_deg"],
                "minimum_shadow_center_cells": blind[
                    "minimum_shadow_center_cells"
                ],
            }
        )
    except (TypeError, ValueError) as exc:
        raise GeneratorConfigError(str(exc)) from exc
    max_curvature = _finite_real(
        config["max_local_curvature_per_m"],
        name="max_local_curvature_per_m",
    )
    if max_curvature < 0.0:
        raise GeneratorConfigError("max_local_curvature_per_m must be non-negative")
    crossing_angle = _finite_real(
        config["crossing_angle_max_deg"], name="crossing_angle_max_deg"
    )
    if not 0.0 < crossing_angle <= 35.0:
        raise GeneratorConfigError("crossing_angle_max_deg must be in (0, 35]")
    time_scale_range = _range_pair(
        config["time_scale_range"],
        name="time_scale_range",
        lower_bound=0.8,
        upper_bound=1.2,
    )
    if time_scale_range != (1.0, 1.0):
        raise GeneratorConfigError("time_scale_range must equal [1.0, 1.0]")
    crossing_step = _finite_real(
        blind["crossing_angle_step_deg"],
        name="blind_reachability.crossing_angle_step_deg",
    )
    if not 0.0 < crossing_step <= crossing_angle:
        raise GeneratorConfigError(
            "blind_reachability.crossing_angle_step_deg must be positive "
            "and no greater than crossing_angle_max_deg"
        )
    chord_deviation = _finite_real(
        blind["chord_deviation_fastpath_m"],
        name="blind_reachability.chord_deviation_fastpath_m",
    )
    if chord_deviation < 0.0:
        raise GeneratorConfigError(
            "blind_reachability.chord_deviation_fastpath_m must be non-negative"
        )
    return {
        "schema_version": SOP05_GENERATOR_SCHEMA_VERSION,
        "production_event_kind": "environment",
        "target_type_policy": policy,
        "conflict_time_range_s": _range_pair(
            config["conflict_time_range_s"],
            name="conflict_time_range_s",
            lower_bound=1.0,
            upper_bound=2.2,
        ),
        "max_local_curvature_per_m": max_curvature,
        "crossing_angle_max_deg": crossing_angle,
        "time_scale_range": time_scale_range,
        "min_contiguous_visible_frames": _positive_integer(
            config["min_contiguous_visible_frames"],
            name="min_contiguous_visible_frames",
        ),
        "occluders": {
            key: causal_occluders[key]
            for key in ("types", "wall", "shelf", "pillar")
        },
        "blind_reachability": {
            "algorithm_version": SOP05_GENERATOR_ALGORITHM_VERSION,
            "obstacle_proposals_per_trajectory": _positive_integer(
                blind["obstacle_proposals_per_trajectory"],
                name="blind_reachability.obstacle_proposals_per_trajectory",
            ),
            "interaction_range_m": causal_occluders["interaction_range_m"],
            "bearing_bin_count": causal_occluders["bearing_bin_count"],
            "yaw_step_deg": causal_occluders["yaw_step_deg"],
            "crossing_angle_step_deg": crossing_step,
            "minimum_shadow_center_cells": causal_occluders[
                "minimum_shadow_center_cells"
            ],
            "chord_deviation_fastpath_m": chord_deviation,
            "unresolved_exact_fallback_per_anchor": _positive_integer(
                blind["unresolved_exact_fallback_per_anchor"],
                name="blind_reachability.unresolved_exact_fallback_per_anchor",
            ),
        },
    }


def load_generator_config(path: str | Path) -> dict[str, object]:
    """Load one standalone SOP-05 domain config and reject unknown keys."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise GeneratorConfigError("generator config top level must be a mapping")
    return normalize_generator_config(raw)


def _jsonable_generator_config(config: Mapping[str, Any]) -> dict[str, object]:
    policy = config["target_type_policy"]
    return {
        "schema_version": SOP05_GENERATOR_SCHEMA_VERSION,
        "production_event_kind": "environment",
        "target_type_policy": policy.as_dict(),
        "conflict_time_range_s": list(config["conflict_time_range_s"]),
        "max_local_curvature_per_m": config["max_local_curvature_per_m"],
        "crossing_angle_max_deg": config["crossing_angle_max_deg"],
        "time_scale_range": list(config["time_scale_range"]),
        "min_contiguous_visible_frames": config[
            "min_contiguous_visible_frames"
        ],
        "occluders": {
            "types": list(config["occluders"]["types"]),
            **{
                kind: {
                    "length_range_m": list(
                        config["occluders"][kind]["length_range_m"]
                    ),
                    "width_range_m": list(
                        config["occluders"][kind]["width_range_m"]
                    ),
                }
                for kind in ("wall", "shelf", "pillar")
            },
        },
        "blind_reachability": {
            **config["blind_reachability"],
            "interaction_range_m": list(
                config["blind_reachability"]["interaction_range_m"]
            ),
        },
    }


def _as_normalized_generator_config(
    config: Mapping[str, Any],
) -> dict[str, object]:
    if isinstance(config.get("target_type_policy"), TargetTypePolicy):
        return normalize_generator_config(_jsonable_generator_config(config))
    return normalize_generator_config(config)


def _generator_digest(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _jsonable_generator_config(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return stable_digest(payload, size=16)


def _canonical_identity_digest(domain: str, payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    hasher = hashlib.blake2b(digest_size=16)
    for part in (domain.encode("utf-8"), canonical):
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return hasher.hexdigest()


def _require_identity_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_identity_digest(value: object, name: str) -> str:
    digest = _require_identity_string(value, name)
    if len(digest) != 32 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase BLAKE2b-128 hex digest")
    return digest


def _require_prefixed_identity(value: object, name: str, prefix: str) -> str:
    identity = _require_identity_string(value, name)
    if not identity.startswith(prefix):
        raise ValueError(f"{name} must start with {prefix!r}")
    _require_identity_digest(identity[len(prefix) :], f"{name} suffix")
    return identity


def _require_identity_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _require_identity_finite_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _canonical_identity_footprint_spec(
    footprint_spec: Mapping[str, object],
    *,
    object_type: str,
    footprint_spec_digest: str,
) -> dict[str, object]:
    if not isinstance(footprint_spec, Mapping):
        raise ValueError("footprint_spec must be a mapping")
    try:
        canonical_spec = json.loads(
            json.dumps(
                dict(footprint_spec),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("footprint_spec must be canonical JSON") from exc
    expected_digest = compute_footprint_spec_digest(canonical_spec)
    if canonical_spec["object_type"] != object_type:
        raise ValueError("object_type must match footprint_spec")
    if footprint_spec_digest != expected_digest:
        raise ValueError("footprint_spec_digest mismatch")
    return canonical_spec


def compute_generated_event_id(
    *,
    generator_algorithm_version: str,
    generator_config_digest: str,
    base_state_id: str,
    trajectory_id: str,
    event_index: int,
    attempt_index: int,
    attempt_seed: int,
    event_kind: str,
    conflict_index: int,
    conflict_time_s: float,
    target_dynamic_object_id: str,
    source_snippet_id: str,
    source_object_id: str,
    object_type: str,
    footprint_spec: Mapping[str, object],
    footprint_spec_digest: str,
    target_type_policy_digest: str,
    layout_version: str,
) -> str:
    """Build a mother-event lineage ID without variant motion summaries."""

    for name, value in (
        ("generator_algorithm_version", generator_algorithm_version),
        ("base_state_id", base_state_id),
        ("trajectory_id", trajectory_id),
        ("target_dynamic_object_id", target_dynamic_object_id),
        ("source_snippet_id", source_snippet_id),
        ("source_object_id", source_object_id),
        ("object_type", object_type),
    ):
        _require_identity_string(value, name)
    _require_identity_digest(generator_config_digest, "generator_config_digest")
    _require_identity_digest(
        footprint_spec_digest, "footprint_spec_digest"
    )
    _require_identity_digest(
        target_type_policy_digest, "target_type_policy_digest"
    )
    if event_kind not in _EVENT_TYPES:
        raise ValueError(f"event_kind must be one of {_EVENT_TYPES}")
    if layout_version != EVENT_TARGET_MOTION_LAYOUT_VERSION:
        raise ValueError("unsupported event target motion layout")
    event_index = _require_identity_nonnegative_int(event_index, "event_index")
    attempt_index = _require_identity_nonnegative_int(
        attempt_index, "attempt_index"
    )
    attempt_seed = _require_identity_nonnegative_int(attempt_seed, "attempt_seed")
    conflict_index = _require_identity_nonnegative_int(
        conflict_index, "conflict_index"
    )
    conflict_time = _require_identity_finite_real(
        conflict_time_s, "conflict_time_s"
    )
    canonical_footprint_spec = _canonical_identity_footprint_spec(
        footprint_spec,
        object_type=object_type,
        footprint_spec_digest=footprint_spec_digest,
    )
    payload = {
        "identity_version": _GENERATED_EVENT_IDENTITY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generator_algorithm_version": generator_algorithm_version,
        "generator_config_digest": generator_config_digest,
        "base_state_id": base_state_id,
        "trajectory_id": trajectory_id,
        "event_slot_index": int(event_index),
        "accepted_attempt_index": int(attempt_index),
        "attempt_seed": int(attempt_seed),
        "event_kind": event_kind,
        "conflict_index": int(conflict_index),
        "conflict_time_s_decimal_9": f"{conflict_time:.9f}",
        "target_dynamic_object_id": target_dynamic_object_id,
        "source_snippet_id": source_snippet_id,
        "source_object_id": source_object_id,
        "target_object_type": object_type,
        "target_footprint_spec": canonical_footprint_spec,
        "target_footprint_spec_digest": footprint_spec_digest,
        "target_type_policy_digest": target_type_policy_digest,
        "event_target_motion_layout_version": layout_version,
    }
    return "event-" + _canonical_identity_digest(
        _GENERATED_EVENT_ID_DOMAIN, payload
    )


_build_generated_event_id = compute_generated_event_id


def compute_generated_world_id(
    *,
    generator_algorithm_version: str,
    generator_config_digest: str,
    generated_event_id: str,
    base_state_id: str,
    trajectory_id: str,
    event_kind: str,
    target_dynamic_object_id: str,
    source_snippet_id: str,
    source_object_id: str,
    object_type: str,
    footprint_spec: Mapping[str, object],
    footprint_spec_digest: str,
    target_type_policy_digest: str,
    layout_version: str,
    history_array_digest: str,
    current_pose: np.ndarray,
    future_array_digest: str,
) -> str:
    """Build a world identity that binds lineage and exact target motion."""

    for name, value in (
        ("generator_algorithm_version", generator_algorithm_version),
        ("base_state_id", base_state_id),
        ("trajectory_id", trajectory_id),
        ("target_dynamic_object_id", target_dynamic_object_id),
        ("source_snippet_id", source_snippet_id),
        ("source_object_id", source_object_id),
        ("object_type", object_type),
    ):
        _require_identity_string(value, name)
    _require_prefixed_identity(
        generated_event_id, "generated_event_id", "event-"
    )
    _require_identity_digest(generator_config_digest, "generator_config_digest")
    _require_identity_digest(
        footprint_spec_digest, "footprint_spec_digest"
    )
    _require_identity_digest(
        target_type_policy_digest, "target_type_policy_digest"
    )
    _require_identity_digest(history_array_digest, "history_array_digest")
    _require_identity_digest(future_array_digest, "future_array_digest")
    if event_kind not in _EVENT_TYPES:
        raise ValueError(f"event_kind must be one of {_EVENT_TYPES}")
    if layout_version != EVENT_TARGET_MOTION_LAYOUT_VERSION:
        raise ValueError("unsupported event target motion layout")
    canonical_footprint_spec = _canonical_identity_footprint_spec(
        footprint_spec,
        object_type=object_type,
        footprint_spec_digest=footprint_spec_digest,
    )
    current = np.asarray(current_pose)
    if current.shape != (3,) or current.dtype != np.dtype(np.float32):
        raise ValueError("current_pose must be float32 with shape (3,)")
    if not np.isfinite(current).all():
        raise ValueError("current_pose must contain only finite values")
    current_le = np.ascontiguousarray(current, dtype=np.dtype("<f4"))
    payload = {
        "identity_version": _WORLD_IDENTITY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generator_algorithm_version": generator_algorithm_version,
        "generator_config_digest": generator_config_digest,
        "generated_event_id": generated_event_id,
        "base_state_id": base_state_id,
        "trajectory_id": trajectory_id,
        "event_kind": event_kind,
        "target_dynamic_object_id": target_dynamic_object_id,
        "source_snippet_id": source_snippet_id,
        "source_object_id": source_object_id,
        "target_object_type": object_type,
        "target_footprint_spec": canonical_footprint_spec,
        "target_footprint_spec_digest": footprint_spec_digest,
        "target_type_policy_digest": target_type_policy_digest,
        "event_target_motion_layout_version": layout_version,
        "target_history_array_digest": history_array_digest,
        "target_current_pose_dtype": _FLOAT32_LE_DTYPE_TOKEN,
        "target_current_pose_shape": [3],
        "target_current_pose_order": _ARRAY_ORDER_TOKEN,
        "target_current_pose_le_bytes_hex": current_le.tobytes(
            order="C"
        ).hex(),
        "target_future_array_digest": future_array_digest,
    }
    return "world-" + _canonical_identity_digest(_WORLD_ID_DOMAIN, payload)


_build_world_id = compute_generated_world_id


def _footprints_for_specs(
    specs: Mapping[str, dict],
) -> dict[str, Footprint]:
    return {
        object_id: footprint_from_spec(specs[object_id])
        for object_id in sorted(specs)
    }


def _round_float64_outward(value: float, *, upward: bool) -> float:
    direction = np.float64(np.inf if upward else -np.inf)
    return float(np.nextafter(np.float64(value), direction))


def _float32_quantization_interval(value: np.float32) -> tuple[float, float]:
    """Return outward half-ULP bounds for a finite stored float32 value."""
    quantized = np.float32(value)
    center = float(quantized)
    with np.errstate(over="ignore"):
        previous = float(np.nextafter(quantized, np.float32(-np.inf)))
        following = float(np.nextafter(quantized, np.float32(np.inf)))
    lower_gap = center - previous
    upper_gap = following - center
    if not np.isfinite(lower_gap):
        lower_gap = upper_gap
    if not np.isfinite(upper_gap):
        upper_gap = lower_gap
    lower = center - 0.5 * lower_gap
    upper = center + 0.5 * upper_gap
    return (
        _round_float64_outward(lower, upward=False),
        _round_float64_outward(upper, upward=True),
    )


def _interval_difference(
    left: tuple[float, float],
    right: tuple[float, float],
) -> tuple[float, float]:
    return (
        _round_float64_outward(left[0] - right[1], upward=False),
        _round_float64_outward(left[1] - right[0], upward=True),
    )


def _interval_product(
    left: tuple[float, float],
    right: tuple[float, float],
) -> tuple[float, float]:
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return (
        _round_float64_outward(min(products), upward=False),
        _round_float64_outward(max(products), upward=True),
    )


def _interval_norm_upper(
    vector: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    maximum_x = max(abs(vector[0][0]), abs(vector[0][1]))
    maximum_y = max(abs(vector[1][0]), abs(vector[1][1]))
    return _round_float64_outward(
        float(np.hypot(maximum_x, maximum_y)), upward=True
    )


def _menger_curvature_lower_bound_float32(points: np.ndarray) -> float:
    """Conservatively bound curvature below over half-ULP coordinates."""
    point_intervals = tuple(
        tuple(_float32_quantization_interval(value) for value in point)
        for point in points
    )
    before = tuple(
        _interval_difference(point_intervals[1][axis], point_intervals[0][axis])
        for axis in range(2)
    )
    after = tuple(
        _interval_difference(point_intervals[2][axis], point_intervals[1][axis])
        for axis in range(2)
    )
    opposite = tuple(
        _interval_difference(point_intervals[2][axis], point_intervals[0][axis])
        for axis in range(2)
    )
    first_product = _interval_product(before[0], after[1])
    second_product = _interval_product(before[1], after[0])
    cross_interval = _interval_difference(first_product, second_product)
    if cross_interval[0] <= 0.0 <= cross_interval[1]:
        absolute_cross_lower = 0.0
    elif cross_interval[0] > 0.0:
        absolute_cross_lower = cross_interval[0]
    else:
        absolute_cross_lower = -cross_interval[1]
    if absolute_cross_lower <= 0.0:
        return 0.0
    denominator_upper = _round_float64_outward(
        _interval_norm_upper(before) * _interval_norm_upper(after),
        upward=True,
    )
    denominator_upper = _round_float64_outward(
        denominator_upper * _interval_norm_upper(opposite),
        upward=True,
    )
    numerator_lower = _round_float64_outward(
        2.0 * absolute_cross_lower,
        upward=False,
    )
    return max(
        0.0,
        _round_float64_outward(
            numerator_lower / denominator_upper,
            upward=False,
        ),
    )


def _trajectory_geometry(
    trajectory: LocalTrajectory,
    *,
    dt_s: float,
    conflict_range: tuple[float, float],
    rng: np.random.Generator,
    max_curvature: float,
    conflict_time_quantile: float | None = None,
) -> tuple[int, float, np.ndarray, np.ndarray, np.ndarray]:
    poses = np.asarray(trajectory.poses)
    if poses.ndim != 2 or poses.shape[1:] != (3,) or poses.dtype != np.float32:
        raise _EventRejection("trajectory_contract_invalid")
    if not np.isfinite(poses).all():
        raise _EventRejection("trajectory_nonfinite")
    times = (np.arange(poses.shape[0], dtype=np.float64) + 1.0) * dt_s
    eligible = np.flatnonzero(
        (times >= conflict_range[0] - 1e-9)
        & (times <= conflict_range[1] + 1e-9)
    )
    if eligible.size == 0:
        raise _EventRejection("conflict_time_unavailable")
    if conflict_time_quantile is None:
        eligible_index = int(rng.integers(0, eligible.size))
    else:
        quantile = _finite_real(
            conflict_time_quantile, name="conflict_time_quantile"
        )
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("conflict_time_quantile must lie in [0, 1]")
        eligible_index = int(np.rint(quantile * (eligible.size - 1)))
    index = int(eligible[eligible_index])
    previous = poses[max(0, index - 1), :2].astype(np.float64)
    following = poses[min(poses.shape[0] - 1, index + 1), :2].astype(np.float64)
    tangent = following - previous
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-9:
        raise _EventRejection("conflict_tangent_degenerate")
    tangent /= norm
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    if 0 < index < poses.shape[0] - 1:
        before = poses[index, :2].astype(np.float64) - poses[index - 1, :2]
        after = poses[index + 1, :2].astype(np.float64) - poses[index, :2]
        before_norm = float(np.linalg.norm(before))
        after_norm = float(np.linalg.norm(after))
        if min(before_norm, after_norm) <= 1e-9:
            raise _EventRejection("conflict_tangent_degenerate")
        curvature_lower_bound = _menger_curvature_lower_bound_float32(
            poses[index - 1 : index + 2, :2]
        )
        if curvature_lower_bound > max_curvature:
            raise _EventRejection("conflict_curvature")
    return index, float(times[index]), poses[index, :2].astype(np.float64), tangent, normal


def _rotated_direction(
    normal: np.ndarray, *, max_angle_deg: float, rng: np.random.Generator
) -> np.ndarray:
    side = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
    angle = float(rng.uniform(-max_angle_deg, max_angle_deg))
    radians = np.deg2rad(angle)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return rotation @ (side * normal)


def _joint_crossing_direction(
    normal: np.ndarray,
    parameters: JointOccluderParameters,
    *,
    max_angle_deg: float,
) -> np.ndarray:
    base = -float(parameters.side) * normal
    radians = np.deg2rad(parameters.angle_multiplier * max_angle_deg)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return rotation @ base


def _validate_target_physics(
    target: TransplantedDynamicObject,
    *,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
) -> Footprint:
    grid = build_grid_spec(base_config)
    footprint = footprint_from_spec(target.footprint_spec)
    history = np.asarray(target.history_poses)
    current = np.asarray(target.current_pose)
    future = np.asarray(target.future_poses)
    if history.shape != (grid.history_steps, 3):
        raise _EventRejection("target_history_shape_invalid")
    if future.shape != (grid.future_steps, 3):
        raise _EventRejection("target_future_shape_invalid")
    if current.shape != (3,):
        raise _EventRejection("target_current_shape_invalid")
    if any(array.dtype != np.float32 for array in (history, current, future)):
        raise _EventRejection("target_motion_dtype_invalid")
    if not all(np.isfinite(array).all() for array in (history, current, future)):
        raise _EventRejection("target_motion_nonfinite")
    if not np.array_equal(current, history[-1]):
        raise _EventRejection("target_current_history_mismatch")
    all_poses = np.vstack((history, future))
    static = (
        np.zeros((grid.height, grid.width), dtype=np.float32)
        if base_state.static_map_local is None
        else np.asarray(base_state.static_map_local)
    )
    if swept_footprint_intersects_occupancy(
        footprint,
        all_poses,
        static,
        grid=grid,
    ):
        raise _EventRejection("target_static_collision")
    if not bool(np.any(points_in_grid(target.future_poses[:, :2], grid))):
        raise _EventRejection("target_future_out_of_bounds")
    robot_cfg = base_config["robot"]
    robot_footprint = inflate_footprint(
        RectangleFootprint(robot_cfg["length_m"], robot_cfg["width_m"]),
        robot_cfg["inflation_m"],
    )
    if intersects(
        robot_footprint,
        np.zeros(3, dtype=np.float64),
        footprint,
        target.current_pose,
    ):
        raise _EventRejection("target_current_robot_overlap")
    dt_s = float(base_config["bev"]["future_dt_s"])
    velocities = np.diff(all_poses[:, :2].astype(np.float64), axis=0) / dt_s
    speeds = np.linalg.norm(velocities, axis=1)
    accelerations = (
        np.diff(velocities, axis=0) / dt_s
        if velocities.shape[0] > 1
        else np.zeros((0, 2), dtype=np.float64)
    )
    type_config = base_config["dynamic_objects"][target.object_type]
    if np.any(speeds > float(type_config["max_speed_mps"]) + 1e-6):
        raise _EventRejection("target_speed_limit")
    if accelerations.size and np.any(
        np.linalg.norm(accelerations, axis=1)
        > float(type_config["max_acceleration_mps2"]) + 1e-5
    ):
        raise _EventRejection("target_acceleration_limit")

    context_footprints = _footprints_for_specs(oracle_context.dynamic_object_specs)
    for object_id in sorted(context_footprints):
        context_poses = np.vstack(
            (
                oracle_context.dynamic_object_history[object_id],
                oracle_context.dynamic_object_future[object_id],
            )
        )
        if synchronized_sweeps_intersect(
            footprint,
            all_poses,
            context_footprints[object_id],
            context_poses,
            grid=grid,
        ):
            raise _EventRejection("target_context_collision")
    return footprint


def _context_current_occupancy(
    oracle_context: OracleContext,
    *,
    grid,
    context_footprints: Mapping[str, Footprint],
) -> np.ndarray:
    occupancy = np.zeros((grid.height, grid.width), dtype=bool)
    for object_id in sorted(context_footprints):
        occupancy |= rasterize_footprint(
            context_footprints[object_id],
            oracle_context.dynamic_object_history[object_id][-1],
            grid,
        )
    return occupancy


def _structural_candidates(
    config: Mapping[str, Any], rng: np.random.Generator
) -> tuple[StructuralBlindSpot, ...]:
    candidates = []
    sector_options = [()] + [
        (dict(sector),) for sector in config["optional_blind_sectors"]
    ]
    for fov in config["forward_fov_deg"]:
        for sensor_range in config["range_m"]:
            for sectors in sector_options:
                candidates.append(
                    StructuralBlindSpot(
                        forward_fov_deg=fov,
                        range_m=sensor_range,
                        blind_sectors=sectors,
                    )
                )
    order = rng.permutation(len(candidates))
    return tuple(candidates[int(index)] for index in order)


def _target_visibility_history(
    *,
    event_kind: str,
    static_occupancy: np.ndarray,
    placement: OccluderPlacement | None,
    grid,
    base_state: BaseState,
    oracle_context: OracleContext,
    context_footprints: Mapping[str, Footprint],
    target: TransplantedDynamicObject,
    target_footprint: Footprint,
    structural: StructuralBlindSpot | None,
) -> np.ndarray:
    """Recompute audit-only target visibility at every observed history frame."""

    occupied_without_context = np.asarray(static_occupancy != 0, dtype=bool)
    if event_kind in {"environment", "mixed"}:
        if placement is None:
            raise _EventRejection("occluder_geometry_missing")
        occupied_without_context |= placement.mask
    if event_kind in {"structural", "mixed"} and structural is None:
        raise _EventRejection("structural_candidate_missing")

    history_visibility = np.empty(grid.history_steps, dtype=bool)
    for history_index in range(grid.history_steps):
        occupied = occupied_without_context.copy()
        for object_id in sorted(context_footprints):
            occupied |= rasterize_footprint(
                context_footprints[object_id],
                oracle_context.dynamic_object_history[object_id][history_index],
                grid,
            )
        sensor_pose = base_state.robot_history[history_index]
        if event_kind in {"structural", "mixed"}:
            visibility = build_structural_visibility(
                occupied,
                grid,
                sensor_pose=sensor_pose,
                blind_spot=structural,
            )
        else:
            visibility = raycast_visibility(
                occupied,
                grid,
                sensor_pose=sensor_pose,
            )
        history_visibility[history_index] = footprint_visibility_sequence(
            target_footprint,
            target.history_poses[history_index : history_index + 1],
            visibility,
            grid,
        )[0]
    if history_visibility.shape != (8,) or history_visibility.dtype != np.bool_:
        raise _EventRejection("target_visibility_history_contract_invalid")
    return history_visibility


def _visibility_for_event(
    *,
    event_kind: str,
    static_occupancy: np.ndarray,
    context_current_occupancy: np.ndarray,
    grid,
    sensor_pose: np.ndarray,
    target: TransplantedDynamicObject,
    target_footprint: Footprint,
    trajectory: LocalTrajectory,
    robot_footprint: Footprint,
    conflict_point: np.ndarray,
    normal: np.ndarray,
    oracle_context: OracleContext,
    generator_config: Mapping[str, Any],
    rng: np.random.Generator,
    structural_candidate: StructuralBlindSpot | None = None,
    precomputed_placement: OccluderPlacement | None = None,
) -> tuple[np.ndarray, OccluderPlacement | None, StructuralBlindSpot | None, np.ndarray]:
    placement = precomputed_placement
    structural = None
    occupied = np.asarray(static_occupancy != 0, dtype=bool)
    if event_kind in {"environment", "mixed"}:
        if placement is None:
            raise _EventRejection("occluder_geometry_missing")
        occupied |= placement.mask

    target_poses = np.vstack((target.current_pose, target.future_poses))
    occupied_with_context = occupied | context_current_occupancy
    if event_kind in {"structural", "mixed"}:
        if structural_candidate is None:
            raise _EventRejection("structural_candidate_missing")
        candidate_visibility = build_structural_visibility(
            occupied_with_context,
            grid,
            sensor_pose=sensor_pose,
            blind_spot=structural_candidate,
        )
        sequence = footprint_visibility_sequence(
            target_footprint, target_poses, candidate_visibility, grid
        )
        if not bool(sequence[0]) and has_continuous_emergence(
            sequence,
            min_visible_frames=generator_config[
                "min_contiguous_visible_frames"
            ],
        ) and bool(sequence[-1]):
            structural = structural_candidate
            return sequence, placement, structural, occupied
        raise _EventRejection(
            "structural_visibility_invalid",
            occluder_candidate_rejection_reasons=(
                {} if placement is None else placement.rejection_reasons
            ),
        )

    visibility = raycast_visibility(
        occupied_with_context,
        grid,
        sensor_pose=sensor_pose,
    )
    sequence = footprint_visibility_sequence(
        target_footprint, target_poses, visibility, grid
    )
    if bool(sequence[0]) or not has_continuous_emergence(
        sequence,
        min_visible_frames=generator_config["min_contiguous_visible_frames"],
    ) or not bool(sequence[-1]):
        raise _EventRejection(
            "environment_visibility_invalid",
            occluder_candidate_rejection_reasons=(
                {} if placement is None else placement.rejection_reasons
            ),
        )
    return sequence, placement, structural, occupied


def _bucket_key(target: TransplantedDynamicObject | None) -> tuple[str, str, str]:
    if target is None:
        return "unassigned", "unassigned", "unassigned"
    return (
        target.object_type,
        str(target.footprint_spec["footprint"]["kind"]),
        str(target.provenance.get("geometry_source", "unknown")),
    )


def _update_bucket(
    buckets: dict[str, dict[str, object]],
    key: str,
    *,
    accepted: bool,
    reason: str | None,
) -> None:
    bucket = buckets.setdefault(
        key,
        {"attempted": 0, "accepted": 0, "rejected": 0, "rejection_reasons": {}},
    )
    bucket["attempted"] += 1
    if accepted:
        bucket["accepted"] += 1
    else:
        bucket["rejected"] += 1
        reasons = bucket["rejection_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1


def _sort_buckets(buckets: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        key: {
            "attempted": bucket["attempted"],
            "accepted": bucket["accepted"],
            "rejected": bucket["rejected"],
            "attempt_acceptance_rate": (
                bucket["accepted"] / bucket["attempted"]
                if bucket["attempted"]
                else 0.0
            ),
            "rejection_reasons": dict(sorted(bucket["rejection_reasons"].items())),
        }
        for key, bucket in sorted(buckets.items())
    }


def _summarize_event_kind_buckets(
    buckets: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    summary = {}
    for event_kind in _EVENT_TYPES:
        bucket = buckets[event_kind]
        requested = int(bucket["requested"])
        attempted = int(bucket["attempted"])
        accepted = int(bucket["accepted"])
        summary[event_kind] = {
            "requested": requested,
            "attempted": attempted,
            "accepted": accepted,
            "rejected": attempted - accepted,
            "request_acceptance_rate": (
                accepted / requested if requested else 0.0
            ),
            "attempt_acceptance_rate": (
                accepted / attempted if attempted else 0.0
            ),
            "rejection_reasons": dict(
                sorted(bucket["rejection_reasons"].items())
            ),
            "rejection_stage_counts": dict(bucket["rejection_stage_counts"]),
        }
    return summary


def _merge_reason_counts(
    destination: dict[str, int], source: Mapping[str, int]
) -> None:
    for reason, count in source.items():
        destination[reason] = destination.get(reason, 0) + int(count)


def _rejection_stage(reason: str) -> str:
    if reason in {
        "occluder_los_degenerate",
        "occluder_offset_outside_line_of_sight",
        "occluder_out_of_bounds",
        "occluder_static_overlap",
        "occluder_robot_swept_overlap",
        "occluder_context_collision",
    }:
        return "occluder_geometry"
    if "visibility" in reason or "hide_current" in reason or "does_not_emerge" in reason:
        return "visibility"
    return "target_conditioning"


def _causal_config(config: Mapping[str, Any]) -> dict[str, object]:
    blind = config["blind_reachability"]
    return {
        **config["occluders"],
        "interaction_range_m": blind["interaction_range_m"],
        "bearing_bin_count": blind["bearing_bin_count"],
        "yaw_step_deg": blind["yaw_step_deg"],
        "minimum_shadow_center_cells": blind["minimum_shadow_center_cells"],
    }


def _ordered_policy_snippets(
    libraries: Mapping[str, SnippetLibrary],
    *,
    split: str,
    policy: TargetTypePolicy,
) -> tuple[object, ...]:
    snippets = []
    for object_type in policy.whitelist:
        if policy.weights[object_type] <= 0.0:
            continue
        library = libraries.get(object_type)
        if not isinstance(library, SnippetLibrary) or not library.snippets:
            raise TransplantError("snippet_library_missing")
        if library.object_type != object_type:
            raise TransplantError("snippet_library_type_mismatch")
        for snippet in library.snippets:
            if snippet.object_type != object_type:
                raise TransplantError("snippet_library_type_mismatch")
            if snippet.split != split:
                raise TransplantError("snippet_split_mismatch")
            snippets.append(snippet)
    return tuple(sorted(snippets, key=lambda item: item.snippet_id))


def _trajectory_geometry_at_index(
    trajectory: LocalTrajectory,
    *,
    index: int,
    conflict_time_s: float,
    max_curvature: float,
) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(trajectory.poses)
    previous = poses[max(0, index - 1), :2].astype(np.float64)
    following = poses[min(poses.shape[0] - 1, index + 1), :2].astype(np.float64)
    tangent = following - previous
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-9:
        raise _EventRejection("conflict_tangent_degenerate")
    tangent /= tangent_norm
    if 0 < index < poses.shape[0] - 1:
        before = poses[index, :2].astype(np.float64) - poses[index - 1, :2]
        after = poses[index + 1, :2].astype(np.float64) - poses[index, :2]
        if min(float(np.linalg.norm(before)), float(np.linalg.norm(after))) <= 1e-9:
            raise _EventRejection("conflict_tangent_degenerate")
        if (
            _menger_curvature_lower_bound_float32(
                poses[index - 1 : index + 2, :2]
            )
            > max_curvature
        ):
            raise _EventRejection("conflict_curvature")
    expected_time = float(index + 1) * 0.2
    if not np.isclose(conflict_time_s, expected_time, rtol=0.0, atol=1e-12):
        raise ValueError("trajectory conflict time metadata is misaligned")
    return tangent, np.asarray([-tangent[1], tangent[0]], dtype=np.float64)


def _circumscribed_radius(footprint: Footprint) -> float:
    if hasattr(footprint, "radius_m"):
        return float(footprint.radius_m)
    return 0.5 * float(np.hypot(footprint.length_m, footprint.width_m))


def _build_v5_event(
    *,
    accepted: Mapping[str, Any],
    event_index: int,
    base_state: BaseState,
    oracle_context: OracleContext,
    trajectory: LocalTrajectory,
    grid,
    policy: TargetTypePolicy,
    generator_digest: str,
) -> GeneratedEvent:
    target = accepted["target"]
    placement = accepted["placement"]
    visibility_sequence = accepted["visibility_sequence"]
    target_visibility_history = accepted["target_visibility_history"]
    conflict_index = int(accepted["conflict_index"])
    conflict_time_s = float(accepted["conflict_time_s"])
    attempt_seed = int(accepted["attempt_seed"])
    proposal_index = int(accepted["proposal_index"])
    history_array_digest = compute_motion_array_digest(
        target.history_poses, field_name="target_history_poses"
    )
    future_array_digest = compute_motion_array_digest(
        target.future_poses, field_name="target_future_poses"
    )
    generated_event_id = compute_generated_event_id(
        generator_algorithm_version=SOP05_GENERATOR_ALGORITHM_VERSION,
        generator_config_digest=generator_digest,
        base_state_id=base_state.state_id,
        trajectory_id=trajectory.trajectory_id,
        event_index=event_index,
        attempt_index=proposal_index,
        attempt_seed=attempt_seed,
        event_kind="environment",
        conflict_index=conflict_index,
        conflict_time_s=conflict_time_s,
        target_dynamic_object_id=target.target_dynamic_object_id,
        source_snippet_id=target.snippet_id,
        source_object_id=target.source_object_id,
        object_type=target.object_type,
        footprint_spec=target.footprint_spec,
        footprint_spec_digest=target.footprint_spec_digest,
        target_type_policy_digest=policy.digest,
        layout_version=EVENT_TARGET_MOTION_LAYOUT_VERSION,
    )
    world_id = compute_generated_world_id(
        generator_algorithm_version=SOP05_GENERATOR_ALGORITHM_VERSION,
        generator_config_digest=generator_digest,
        generated_event_id=generated_event_id,
        base_state_id=base_state.state_id,
        trajectory_id=trajectory.trajectory_id,
        event_kind="environment",
        target_dynamic_object_id=target.target_dynamic_object_id,
        source_snippet_id=target.snippet_id,
        source_object_id=target.source_object_id,
        object_type=target.object_type,
        footprint_spec=target.footprint_spec,
        footprint_spec_digest=target.footprint_spec_digest,
        target_type_policy_digest=policy.digest,
        layout_version=EVENT_TARGET_MOTION_LAYOUT_VERSION,
        history_array_digest=history_array_digest,
        current_pose=target.current_pose,
        future_array_digest=future_array_digest,
    )
    target_motion_record = create_event_target_motion_record(
        generated_event_id=generated_event_id,
        world_id=world_id,
        base_state_id=base_state.state_id,
        trajectory_id=trajectory.trajectory_id,
        target_dynamic_object_id=target.target_dynamic_object_id,
        source_snippet_id=target.snippet_id,
        source_object_id=target.source_object_id,
        object_type=target.object_type,
        footprint_spec=target.footprint_spec,
        footprint_spec_digest=target.footprint_spec_digest,
        target_type_policy_digest=policy.digest,
        history_poses=target.history_poses,
        current_pose=target.current_pose,
        future_poses=target.future_poses,
    )
    dynamic_trajectories = {
        object_id: oracle_context.dynamic_object_future[object_id].copy()
        for object_id in sorted(oracle_context.dynamic_object_future)
    }
    dynamic_specs = {
        object_id: dict(oracle_context.dynamic_object_specs[object_id])
        for object_id in sorted(oracle_context.dynamic_object_specs)
    }
    dynamic_trajectories[target.target_dynamic_object_id] = target.future_poses.copy()
    dynamic_specs[target.target_dynamic_object_id] = dict(target.footprint_spec)
    decision = accepted["decision"]
    candidate = accepted["candidate"]
    exact = accepted["exact"]
    metadata = {
        **build_event_target_motion_world_metadata(target_motion_record),
        "schema_version": SCHEMA_VERSION,
        "generator_algorithm_version": SOP05_GENERATOR_ALGORITHM_VERSION,
        "event_kind": "environment",
        "dynamic_object_snippet_id": target.snippet_id,
        "target_type_policy": policy.as_dict(),
        "generator_config_digest": generator_digest,
        "conflict_time_s": conflict_time_s,
        "conflict_index": conflict_index,
        "event_slot_index": event_index,
        "attempt_index": proposal_index,
        "target_provenance": target.provenance,
        "visibility_sequence": [bool(value) for value in visibility_sequence],
        "target_visibility_history": [
            bool(value) for value in target_visibility_history
        ],
        "target_visibility_history_layout": (
            "target_visibility_history8_current7_v1"
        ),
        "context_dynamic_object_ids": sorted(
            oracle_context.dynamic_object_future
        ),
        "causal_occluder_proposal_id": decision.proposal_id,
        "blind_region_id": accepted["region"].region_digest,
        "reachability_candidate_id": candidate.candidate_id,
        "reachability_transform_id": target.provenance["transform_id"],
        "exact_validation_id": exact.result_digest,
    }
    world = OracleWorld(
        world_id=world_id,
        base_state_id=base_state.state_id,
        static_occupancy=accepted["world_occupancy"].astype(np.float32),
        dynamic_object_trajectories=dynamic_trajectories,
        dynamic_object_specs=dynamic_specs,
        occluders=(dict(placement.occluder),),
        blind_spot_config={
            "kind": "environment",
            "occluder_ids": [placement.occluder["occluder_id"]],
            "blind_region_digest": accepted["region"].region_digest,
        },
        random_seed=attempt_seed,
        metadata=metadata,
    )
    validate_event_target_motion_world_join(target_motion_record, world, grid)
    return GeneratedEvent(
        generated_event_id=generated_event_id,
        event_kind="environment",
        world=world,
        target=target,
        target_motion_record=target_motion_record,
        visibility_sequence=visibility_sequence,
        target_visibility_history=target_visibility_history,
        conflict_time_s=conflict_time_s,
        conflict_index=conflict_index,
    )


def _generate_v5_events(
    *,
    base_state: BaseState,
    oracle_context: OracleContext,
    trajectory: LocalTrajectory,
    snippet_libraries: Mapping[str, SnippetLibrary],
    base_config: Mapping[str, Any],
    normalized: Mapping[str, Any],
    seed: int,
    event_count: int,
    attempt_index_start: int,
) -> EventGenerationReport:
    grid = build_grid_spec(base_config)
    validate_base_state(base_state, grid)
    validate_oracle_context(oracle_context, grid)
    if base_state.state_id != oracle_context.base_state_id:
        raise ValueError("base_state and oracle_context ids must match")
    desired_count = _positive_integer(event_count, name="event_count")
    proposal_budget = int(
        normalized["blind_reachability"]["obstacle_proposals_per_trajectory"]
    )
    if not 0 <= attempt_index_start < proposal_budget:
        raise ValueError("attempt_index_start must lie within obstacle proposal budget")

    static_occupancy = (
        np.zeros((grid.height, grid.width), dtype=np.bool_)
        if base_state.static_map_local is None
        else np.asarray(base_state.static_map_local != 0, dtype=np.bool_)
    )
    if base_state.static_map_local is not None and not np.isfinite(
        np.asarray(base_state.static_map_local)
    ).all():
        raise ValueError("static occupancy must contain only finite values")
    base_context_footprints = _footprints_for_specs(
        base_state.visible_dynamic_object_specs
    )
    current_context = np.zeros((grid.height, grid.width), dtype=np.bool_)
    for object_id in sorted(base_context_footprints):
        current_context |= rasterize_footprint(
            base_context_footprints[object_id],
            base_state.visible_dynamic_object_history[object_id][-1],
            grid,
        )
    context_footprints = _footprints_for_specs(oracle_context.dynamic_object_specs)
    context_sweeps = tuple(
        OccluderCollisionSweep(
            footprint=context_footprints[object_id],
            poses=np.vstack(
                (
                    oracle_context.dynamic_object_history[object_id],
                    oracle_context.dynamic_object_future[object_id],
                )
            ),
            rejection_reason="occluder_context_collision",
        )
        for object_id in sorted(context_footprints)
    )
    robot_cfg = base_config["robot"]
    robot_footprint = inflate_footprint(
        RectangleFootprint(robot_cfg["length_m"], robot_cfg["width_m"]),
        robot_cfg["inflation_m"],
    )
    future_dt_s = float(base_config["bev"]["future_dt_s"])
    sweep_cache = RobotSweepCache()
    sweep_bundle = prepare_robot_collision_sweep_bundle(
        base_state,
        trajectory,
        robot_footprint=robot_footprint,
        grid=grid,
        future_dt_s=future_dt_s,
        cache=sweep_cache,
    )
    causal_config = _causal_config(normalized)
    causal_context = build_causal_occluder_context(
        static_occupancy=static_occupancy,
        current_context_occupancy=current_context,
        interaction_poses=trajectory.poses,
        sensor_pose=base_state.robot_history[-1].astype(np.float64),
        grid=grid,
        config=causal_config,
    )
    proposal_schedule = build_causal_occluder_schedule(
        config=causal_config,
        max_candidates=proposal_budget,
        seed=int(seed),
        base_state_id=base_state.state_id,
        trajectory_id=trajectory.trajectory_id,
    )
    policy: TargetTypePolicy = normalized["target_type_policy"]
    snippets = _ordered_policy_snippets(
        snippet_libraries, split=base_state.split, policy=policy
    )
    offsets = np.asarray(
        trajectory.metadata.get("pose_time_offsets_s"), dtype=np.float64
    )
    if offsets.shape != (grid.future_steps,) or not np.isfinite(offsets).all():
        raise ValueError("trajectory pose_time_offsets_s contract invalid")
    conflict_range = normalized["conflict_time_range_s"]
    eligible_indices = tuple(
        int(index)
        for index in np.flatnonzero(
            (offsets >= conflict_range[0] - 1e-12)
            & (offsets <= conflict_range[1] + 1e-12)
            & (np.arange(offsets.size) <= 14)
        )
    )
    if not eligible_indices:
        raise ValueError("no aligned conflict time is available")

    counters = {
        "obstacle_proposal_count": 0,
        "obstacle_proposal_rejected_count": 0,
        "obstacle_proposal_passed_count": 0,
        "transform_candidate_count": 0,
        "transform_rejected_count": 0,
        "chord_certified_count": 0,
        "chord_unresolved_count": 0,
        "exact_validation_count": 0,
        "exact_validation_accepted_count": 0,
        "exact_validation_rejected_count": 0,
    }
    rejection_reasons: dict[str, int] = {}
    proposal_ids: list[str] = []
    candidate_ids: list[str] = []
    transform_ids: list[str] = []
    exact_validation_ids: list[str] = []
    accepted_candidates: list[dict[str, Any]] = []
    center_masks: dict[tuple[str, str, bytes], object] = {}
    unresolved_by_anchor: dict[tuple[str, str, int], int] = {}

    for parameters in proposal_schedule[attempt_index_start:]:
        counters["obstacle_proposal_count"] += 1
        decision = propose_causal_occluder(
            causal_context,
            collision_sweeps=(*sweep_bundle.collision_sweeps, *context_sweeps),
            config=causal_config,
            parameters=parameters,
            seed=int(seed),
            base_state_id=base_state.state_id,
            trajectory_id=trajectory.trajectory_id,
        )
        proposal_ids.append(decision.proposal_id)
        if decision.accepted is None:
            counters["obstacle_proposal_rejected_count"] += 1
            reason = decision.rejection_reason or "causal_occluder_rejected"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue
        counters["obstacle_proposal_passed_count"] += 1
        region = build_blind_region(base_state, decision, grid=grid)
        for snippet in snippets:
            footprint_spec = {
                "object_type": snippet.object_type,
                "footprint": dict(snippet.footprint),
            }
            footprint_payload = json.dumps(
                footprint_spec,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            footprint_digest = stable_digest(footprint_payload, size=16)
            footprint = footprint_from_spec(footprint_spec)
            source_current = snippet.positions[7].astype(np.float64)
            for conflict_index in eligible_indices:
                conflict_time_s = float(offsets[conflict_index])
                try:
                    _tangent, normal = _trajectory_geometry_at_index(
                        trajectory,
                        index=conflict_index,
                        conflict_time_s=conflict_time_s,
                        max_curvature=normalized["max_local_curvature_per_m"],
                    )
                except _EventRejection as exc:
                    rejection_reasons[exc.reason] = (
                        rejection_reasons.get(exc.reason, 0) + 1
                    )
                    continue
                source_anchor = snippet.positions[8 + conflict_index].astype(
                    np.float64
                )
                source_delta = source_anchor - source_current
                directions_by_side = (
                    (
                        -1,
                        scheduled_crossing_directions(
                            -normal,
                            maximum_angle_deg=normalized[
                                "crossing_angle_max_deg"
                            ],
                            angle_step_deg=normalized["blind_reachability"][
                                "crossing_angle_step_deg"
                            ],
                        ),
                    ),
                    (
                        1,
                        scheduled_crossing_directions(
                            normal,
                            maximum_angle_deg=normalized[
                                "crossing_angle_max_deg"
                            ],
                            angle_step_deg=normalized["blind_reachability"][
                                "crossing_angle_step_deg"
                            ],
                        ),
                    ),
                )
                angle_steps = int(
                    round(
                        normalized["crossing_angle_max_deg"]
                        / normalized["blind_reachability"][
                            "crossing_angle_step_deg"
                        ]
                    )
                )
                angle_offsets = np.arange(-angle_steps, angle_steps + 1) * float(
                    normalized["blind_reachability"][
                        "crossing_angle_step_deg"
                    ]
                )
                for crossing_side, directions in directions_by_side:
                    for angle_offset, direction in zip(angle_offsets, directions):
                        counters["transform_candidate_count"] += 1
                        if (
                            not np.isfinite(source_delta).all()
                            or float(np.linalg.norm(source_delta)) <= 1e-9
                        ):
                            counters["transform_rejected_count"] += 1
                            rejection_reasons["invalid_source_anchor"] = (
                                rejection_reasons.get("invalid_source_anchor", 0)
                                + 1
                            )
                            continue
                        identity = ReachabilityIdentity(
                            base_state_id=base_state.state_id,
                            trajectory_id=trajectory.trajectory_id,
                            source_snippet_id=snippet.snippet_id,
                            conflict_index=conflict_index,
                            conflict_time_s=conflict_time_s,
                            crossing_side=crossing_side,
                            angle_offset_deg=float(angle_offset),
                        )
                        candidate = build_reachability_candidate(
                            conflict_point=trajectory.poses[
                                conflict_index, :2
                            ].astype(np.float64),
                            source_current_xy=source_current,
                            source_anchor_xy=source_anchor,
                            desired_crossing_direction=direction,
                            identity=identity,
                        )
                        candidate_ids.append(candidate.candidate_id)
                        if not bool(points_in_grid(candidate.current_xy, grid)):
                            counters["transform_rejected_count"] += 1
                            rejection_reasons["transform_out_of_bounds"] = (
                                rejection_reasons.get(
                                    "transform_out_of_bounds", 0
                                )
                                + 1
                            )
                            continue
                        yaw = float(snippet.headings[7]) + candidate.rotation_rad
                        mask_key = (
                            region.region_digest,
                            footprint_digest,
                            np.asarray(yaw, dtype=">f8").tobytes(),
                        )
                        center_mask = center_masks.get(mask_key)
                        if center_mask is None:
                            center_mask = build_footprint_center_mask(
                                region,
                                footprint_spec=footprint_spec,
                                footprint_spec_digest=footprint_digest,
                                yaw_bin_rad=yaw,
                            )
                            center_masks[mask_key] = center_mask
                        if not candidate_queries_mask(
                            candidate, center_mask.center_mask, grid
                        ):
                            counters["transform_rejected_count"] += 1
                            rejection_reasons["no_arc_mask_intersection"] = (
                                rejection_reasons.get(
                                    "no_arc_mask_intersection", 0
                                )
                                + 1
                            )
                            continue
                        chord = triage_chord(
                            candidate,
                            obstacle_occupancy=region.total_current_occupancy,
                            grid=grid,
                            footprint_radius_m=_circumscribed_radius(footprint),
                            chord_deviation_bound_m=normalized[
                                "blind_reachability"
                            ]["chord_deviation_fastpath_m"],
                        )
                        if chord.outcome == "certified_clear":
                            counters["chord_certified_count"] += 1
                        else:
                            counters["chord_unresolved_count"] += 1
                            anchor_key = (
                                decision.proposal_id,
                                snippet.snippet_id,
                                conflict_index,
                            )
                            used = unresolved_by_anchor.get(anchor_key, 0)
                            if used >= normalized["blind_reachability"][
                                "unresolved_exact_fallback_per_anchor"
                            ]:
                                rejection_reasons[
                                    "chord_unresolved_fallback_exhausted"
                                ] = rejection_reasons.get(
                                    "chord_unresolved_fallback_exhausted", 0
                                ) + 1
                                continue
                            unresolved_by_anchor[anchor_key] = used + 1

                        counters["exact_validation_count"] += 1
                        attempt_seed = derive_seed(
                            int(seed),
                            decision.proposal_id,
                            candidate.candidate_id,
                        )
                        try:
                            target = transplant_reachability_candidate(
                                snippet,
                                candidate=candidate,
                                future_dt_s=future_dt_s,
                                future_steps=grid.future_steps,
                                target_type_policy_digest=policy.digest,
                                seed=attempt_seed,
                                context_object_ids=tuple(
                                    oracle_context.dynamic_object_future
                                ),
                            )
                            transform_ids.append(target.provenance["transform_id"])
                            exact = check_exact_hidden_pose(
                                region,
                                footprint_spec=target.footprint_spec,
                                footprint_spec_digest=target.footprint_spec_digest,
                                pose=target.current_pose.astype(np.float64),
                            )
                            exact_validation_ids.append(exact.result_digest)
                            if not exact.accepted:
                                raise _EventRejection(
                                    exact.rejection_reason
                                    or "exact_hidden_pose_rejected"
                                )
                            target_footprint = _validate_target_physics(
                                target,
                                base_state=base_state,
                                oracle_context=oracle_context,
                                base_config=base_config,
                            )
                            placement = validate_environment_occluder_target(
                                decision.accepted,
                                target_history_poses=target.history_poses,
                                target_future_poses=target.future_poses,
                                target_footprint=target_footprint,
                                grid=grid,
                            )
                            clearances = trajectory_signed_clearances(
                                robot_footprint,
                                trajectory.poses,
                                target_footprint,
                                target.future_poses,
                            )
                            if not bool(clearances[conflict_index] <= 0.0):
                                raise _EventRejection(
                                    "target_not_same_index_collision_mother"
                                )
                            visibility_sequence = footprint_visibility_sequence(
                                target_footprint,
                                np.vstack(
                                    (target.current_pose, target.future_poses)
                                ),
                                region.visibility_mask,
                                grid,
                            )
                            if bool(visibility_sequence[0]) or not (
                                has_continuous_emergence(
                                    visibility_sequence,
                                    min_visible_frames=normalized[
                                        "min_contiguous_visible_frames"
                                    ],
                                )
                                and bool(visibility_sequence[-1])
                            ):
                                raise _EventRejection(
                                    "environment_visibility_invalid"
                                )
                            target_visibility_history = _target_visibility_history(
                                event_kind="environment",
                                static_occupancy=static_occupancy,
                                placement=placement,
                                grid=grid,
                                base_state=base_state,
                                oracle_context=oracle_context,
                                context_footprints=context_footprints,
                                target=target,
                                target_footprint=target_footprint,
                                structural=None,
                            )
                            if bool(target_visibility_history[-1]):
                                raise _EventRejection(
                                    "target_visibility_seam_invalid"
                                )
                        except (_EventRejection, TransplantError, OccluderSamplingError) as exc:
                            counters["exact_validation_rejected_count"] += 1
                            reason = getattr(exc, "reason", str(exc))
                            rejection_reasons[reason] = (
                                rejection_reasons.get(reason, 0) + 1
                            )
                            continue
                        counters["exact_validation_accepted_count"] += 1
                        accepted_candidates.append(
                            {
                                "decision": decision,
                                "region": region,
                                "candidate": candidate,
                                "exact": exact,
                                "target": target,
                                "placement": placement,
                                "visibility_sequence": visibility_sequence,
                                "target_visibility_history": (
                                    target_visibility_history
                                ),
                                "conflict_index": conflict_index,
                                "conflict_time_s": conflict_time_s,
                                "proposal_index": parameters.proposal_index,
                                "attempt_seed": attempt_seed,
                                "world_occupancy": (
                                    static_occupancy | placement.mask
                                ),
                            }
                        )

    accepted_candidates.sort(
        key=lambda item: (
            item["decision"].proposal_id,
            item["candidate"].candidate_id,
            item["target"].provenance["transform_id"],
        )
    )
    generator_digest = _generator_digest(normalized)
    selected = accepted_candidates[:desired_count]
    events = tuple(
        _build_v5_event(
            accepted=item,
            event_index=event_index,
            base_state=base_state,
            oracle_context=oracle_context,
            trajectory=trajectory,
            grid=grid,
            policy=policy,
            generator_digest=generator_digest,
        )
        for event_index, item in enumerate(selected)
    )
    cache_stats = sweep_cache.stats
    summary = {
        "schema_version": SOP05_GENERATOR_SCHEMA_VERSION,
        "seed": int(seed),
        "requested_event_count": desired_count,
        "accepted_count": len(events),
        "rejected_count": counters["exact_validation_rejected_count"],
        "unaccepted_event_count": desired_count - len(events),
        "attempt_index_start": attempt_index_start,
        "attempt_index_stop_exclusive": proposal_budget,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        **counters,
        "proposal_ids": tuple(proposal_ids),
        "reachability_candidate_ids": tuple(candidate_ids),
        "reachability_transform_ids": tuple(transform_ids),
        "exact_validation_ids": tuple(exact_validation_ids),
        "robot_sweep_cache": {
            "size": cache_stats.size,
            "hits": cache_stats.hits,
            "misses": cache_stats.misses,
            "builds": cache_stats.builds,
        },
        "target_type_policy": policy.as_dict(),
        "target_type_policy_digest": policy.digest,
        "generator_config_digest": generator_digest,
        "generator_algorithm_version": SOP05_GENERATOR_ALGORITHM_VERSION,
        "production_event_kind": "environment",
    }
    return EventGenerationReport(events=events, summary=summary)


def generate_events(
    *,
    base_state: BaseState,
    oracle_context: OracleContext,
    trajectory: LocalTrajectory,
    snippet_libraries: Mapping[str, SnippetLibrary],
    base_config: Mapping[str, Any],
    generator_config: Mapping[str, Any],
    seed: int,
    event_count: int,
    attempt_index_start: int = 0,
) -> EventGenerationReport:
    """Generate environment mothers with the blind-reachability-first v5 path."""

    normalized = _as_normalized_generator_config(generator_config)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if isinstance(attempt_index_start, (bool, np.bool_)) or not isinstance(
        attempt_index_start, (int, np.integer)
    ):
        raise TypeError("attempt_index_start must be an integer")
    return _generate_v5_events(
        base_state=base_state,
        oracle_context=oracle_context,
        trajectory=trajectory,
        snippet_libraries=snippet_libraries,
        base_config=base_config,
        normalized=normalized,
        seed=int(seed),
        event_count=event_count,
        attempt_index_start=int(attempt_index_start),
    )
