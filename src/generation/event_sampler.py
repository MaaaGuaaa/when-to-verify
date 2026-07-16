"""Event-centred orchestration for typed hidden dynamic-object worlds."""

from __future__ import annotations

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
    validate_oracle_world,
)
from src.datasets.snippet_library import SnippetLibrary
from src.geometry import (
    Footprint,
    RectangleFootprint,
    inflate_footprint,
    intersects,
    points_in_grid,
    rasterize_footprint,
    rasterize_footprint_sweep,
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
    transplant_snippet,
)
from .occluder_sampler import (
    OccluderPlacement,
    OccluderSamplingError,
    normalize_occluder_config,
    sample_environment_occluder,
)
from .structural_blindspot import (
    StructuralBlindSpot,
    build_structural_visibility,
    footprint_visibility_sequence,
    has_continuous_emergence,
)


_EVENT_TYPES = ("environment", "structural", "mixed")


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

    event_kind: str
    world: OracleWorld
    target: TransplantedDynamicObject
    visibility_sequence: np.ndarray
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
        "target_type_policy",
        "event_type_weights",
        "conflict_time_range_s",
        "max_local_curvature_per_m",
        "crossing_angle_max_deg",
        "time_scale_range",
        "max_resample_attempts",
        "min_contiguous_visible_frames",
        "occluders",
        "structural_fov",
    }
    if set(config) != expected:
        unknown = sorted(set(config) - expected)
        missing = sorted(expected - set(config))
        raise GeneratorConfigError(
            f"generator config keys mismatch; unknown={unknown}, missing={missing}"
        )
    if config["schema_version"] != SCHEMA_VERSION:
        raise GeneratorConfigError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    try:
        policy = (
            config["target_type_policy"]
            if isinstance(config["target_type_policy"], TargetTypePolicy)
            else normalize_target_type_policy(config["target_type_policy"])
        )
        occluders = normalize_occluder_config(config["occluders"])
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
    return {
        "schema_version": SCHEMA_VERSION,
        "target_type_policy": policy,
        "event_type_weights": _normalize_event_weights(
            config["event_type_weights"]
        ),
        "conflict_time_range_s": _range_pair(
            config["conflict_time_range_s"],
            name="conflict_time_range_s",
            lower_bound=1.0,
            upper_bound=2.2,
        ),
        "max_local_curvature_per_m": max_curvature,
        "crossing_angle_max_deg": crossing_angle,
        "time_scale_range": _range_pair(
            config["time_scale_range"],
            name="time_scale_range",
            lower_bound=0.8,
            upper_bound=1.2,
        ),
        "max_resample_attempts": _positive_integer(
            config["max_resample_attempts"], name="max_resample_attempts"
        ),
        "min_contiguous_visible_frames": _positive_integer(
            config["min_contiguous_visible_frames"],
            name="min_contiguous_visible_frames",
        ),
        "occluders": occluders,
        "structural_fov": _normalize_structural_config(
            config["structural_fov"]
        ),
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
        "schema_version": SCHEMA_VERSION,
        "target_type_policy": policy.as_dict(),
        "event_type_weights": dict(config["event_type_weights"]),
        "conflict_time_range_s": list(config["conflict_time_range_s"]),
        "max_local_curvature_per_m": config["max_local_curvature_per_m"],
        "crossing_angle_max_deg": config["crossing_angle_max_deg"],
        "time_scale_range": list(config["time_scale_range"]),
        "max_resample_attempts": config["max_resample_attempts"],
        "min_contiguous_visible_frames": config[
            "min_contiguous_visible_frames"
        ],
        "occluders": {
            "types": list(config["occluders"]["types"]),
            "normal_offset_range_m": list(
                config["occluders"]["normal_offset_range_m"]
            ),
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
        "structural_fov": {
            "forward_fov_deg": list(
                config["structural_fov"]["forward_fov_deg"]
            ),
            "range_m": list(config["structural_fov"]["range_m"]),
            "optional_blind_sectors": [
                dict(sector)
                for sector in config["structural_fov"][
                    "optional_blind_sectors"
                ]
            ],
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


def _footprints_for_specs(
    specs: Mapping[str, dict],
) -> dict[str, Footprint]:
    return {
        object_id: footprint_from_spec(specs[object_id])
        for object_id in sorted(specs)
    }


def _trajectory_geometry(
    trajectory: LocalTrajectory,
    *,
    dt_s: float,
    conflict_range: tuple[float, float],
    rng: np.random.Generator,
    max_curvature: float,
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
    index = int(eligible[int(rng.integers(0, eligible.size))])
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
        cosine = float(
            np.clip(np.dot(before, after) / (before_norm * after_norm), -1.0, 1.0)
        )
        curvature = float(np.arccos(cosine) / (0.5 * (before_norm + after_norm)))
        if curvature > max_curvature + 1e-9:
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


def _sample_range(bounds: tuple[float, float], rng: np.random.Generator) -> float:
    return bounds[0] if bounds[0] == bounds[1] else float(rng.uniform(*bounds))


def _validate_target_physics(
    target: TransplantedDynamicObject,
    *,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
) -> Footprint:
    grid = build_grid_spec(base_config)
    footprint = footprint_from_spec(target.footprint_spec)
    all_poses = np.vstack((target.current_pose, target.future_poses))
    static = (
        np.zeros((grid.height, grid.width), dtype=np.float32)
        if base_state.static_map_local is None
        else np.asarray(base_state.static_map_local)
    )
    target_sweep = rasterize_footprint_sweep(footprint, all_poses, grid)
    if np.any(target_sweep & (static != 0)):
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
        context_future = oracle_context.dynamic_object_future[object_id]
        clearances = trajectory_signed_clearances(
            footprint,
            target.future_poses,
            context_footprints[object_id],
            context_future,
        )
        if np.any(clearances <= 0.0):
            raise _EventRejection("target_context_collision")
        history = oracle_context.dynamic_object_history[object_id]
        if intersects(
            footprint,
            target.current_pose,
            context_footprints[object_id],
            history[-1],
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
) -> tuple[np.ndarray, OccluderPlacement | None, StructuralBlindSpot | None, np.ndarray]:
    placement = None
    structural = None
    occupied = np.asarray(static_occupancy != 0, dtype=bool)
    context_footprints = _footprints_for_specs(oracle_context.dynamic_object_specs)
    if event_kind in {"environment", "mixed"}:
        context_trajectories = {
            object_id: np.vstack(
                (
                    oracle_context.dynamic_object_history[object_id][-1],
                    oracle_context.dynamic_object_future[object_id],
                )
            )
            for object_id in sorted(context_footprints)
        }
        placement = sample_environment_occluder(
            static_occupancy=static_occupancy,
            grid=grid,
            sensor_pose=sensor_pose,
            conflict_point=conflict_point,
            trajectory_normal=normal,
            robot_poses=trajectory.poses,
            robot_footprint=robot_footprint,
            target_current_pose=target.current_pose,
            target_future_poses=target.future_poses,
            target_footprint=target_footprint,
            context_trajectories=context_trajectories,
            context_footprints=context_footprints,
            config=generator_config["occluders"],
            rng=rng,
            max_attempts=8,
        )
        occupied |= placement.mask

    target_poses = np.vstack((target.current_pose, target.future_poses))
    occupied_with_context = occupied | context_current_occupancy
    if event_kind in {"structural", "mixed"}:
        for candidate in _structural_candidates(
            generator_config["structural_fov"], rng
        ):
            candidate_visibility = build_structural_visibility(
                occupied_with_context,
                grid,
                sensor_pose=sensor_pose,
                blind_spot=candidate,
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
                structural = candidate
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
            "rejection_reasons": dict(sorted(bucket["rejection_reasons"].items())),
        }
        for key, bucket in sorted(buckets.items())
    }


def _merge_reason_counts(
    destination: dict[str, int], source: Mapping[str, int]
) -> None:
    for reason, count in source.items():
        destination[reason] = destination.get(reason, 0) + int(count)


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
) -> EventGenerationReport:
    """Generate up to ``event_count`` physical worlds with finite retries."""

    normalized = _as_normalized_generator_config(generator_config)
    grid = build_grid_spec(base_config)
    validate_base_state(base_state, grid)
    validate_oracle_context(oracle_context, grid)
    if base_state.state_id != oracle_context.base_state_id:
        raise ValueError("base_state and oracle_context ids must match")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    desired_count = _positive_integer(event_count, name="event_count")
    event_type_schedule = build_event_type_schedule(
        normalized["event_type_weights"],
        event_count=desired_count,
        rng=make_rng(
            int(seed),
            base_state.state_id,
            trajectory.trajectory_id,
            "event_type_schedule",
        ),
    )
    static_occupancy = (
        np.zeros((grid.height, grid.width), dtype=np.float32)
        if base_state.static_map_local is None
        else np.asarray(base_state.static_map_local, dtype=np.float32)
    )
    if not np.isfinite(static_occupancy).all():
        raise ValueError("static occupancy must contain only finite values")
    context_footprints = _footprints_for_specs(oracle_context.dynamic_object_specs)
    context_current = _context_current_occupancy(
        oracle_context, grid=grid, context_footprints=context_footprints
    )
    robot_cfg = base_config["robot"]
    robot_footprint = inflate_footprint(
        RectangleFootprint(robot_cfg["length_m"], robot_cfg["width_m"]),
        robot_cfg["inflation_m"],
    )
    generator_digest = _generator_digest(normalized)
    policy: TargetTypePolicy = normalized["target_type_policy"]
    events = []
    rejection_reasons: dict[str, int] = {}
    occluder_candidate_rejection_reasons: dict[str, int] = {}
    by_object_type: dict[str, dict[str, object]] = {}
    by_footprint_kind: dict[str, dict[str, object]] = {}
    by_geometry_source: dict[str, dict[str, object]] = {}
    event_kind_counts = {event_type: 0 for event_type in _EVENT_TYPES}
    attempted_count = 0

    for event_index in range(desired_count):
        event_kind = event_type_schedule[event_index]
        accepted = False
        for attempt_index in range(normalized["max_resample_attempts"]):
            attempted_count += 1
            target = None
            attempt_seed = derive_seed(
                int(seed),
                base_state.state_id,
                trajectory.trajectory_id,
                event_index,
                attempt_index,
            )
            rng = make_rng(
                int(seed),
                base_state.state_id,
                trajectory.trajectory_id,
                event_index,
                attempt_index,
            )
            reason = None
            try:
                snippet = sample_motion_snippet(
                    snippet_libraries,
                    split=base_state.split,
                    policy=policy,
                    rng=rng,
                )
                (
                    conflict_index,
                    conflict_time_s,
                    conflict_point,
                    _tangent,
                    normal,
                ) = _trajectory_geometry(
                    trajectory,
                    dt_s=float(base_config["bev"]["future_dt_s"]),
                    conflict_range=normalized["conflict_time_range_s"],
                    rng=rng,
                    max_curvature=normalized["max_local_curvature_per_m"],
                )
                crossing_direction = _rotated_direction(
                    normal,
                    max_angle_deg=normalized["crossing_angle_max_deg"],
                    rng=rng,
                )
                time_scale = _sample_range(normalized["time_scale_range"], rng)
                target = transplant_snippet(
                    snippet,
                    conflict_point=conflict_point,
                    conflict_time_s=conflict_time_s,
                    crossing_direction=crossing_direction,
                    time_scale=time_scale,
                    future_dt_s=float(base_config["bev"]["future_dt_s"]),
                    future_steps=grid.future_steps,
                    base_state_id=base_state.state_id,
                    trajectory_id=trajectory.trajectory_id,
                    target_type_policy_digest=policy.digest,
                    seed=attempt_seed,
                    context_object_ids=tuple(oracle_context.dynamic_object_future),
                )
                target_footprint = _validate_target_physics(
                    target,
                    base_state=base_state,
                    oracle_context=oracle_context,
                    base_config=base_config,
                )
                (
                    visibility_sequence,
                    placement,
                    structural,
                    world_occupancy,
                ) = _visibility_for_event(
                    event_kind=event_kind,
                    static_occupancy=static_occupancy,
                    context_current_occupancy=context_current,
                    grid=grid,
                    sensor_pose=np.zeros(3, dtype=np.float64),
                    target=target,
                    target_footprint=target_footprint,
                    trajectory=trajectory,
                    robot_footprint=robot_footprint,
                    conflict_point=conflict_point,
                    normal=normal,
                    oracle_context=oracle_context,
                    generator_config=normalized,
                    rng=rng,
                )
                if placement is not None:
                    _merge_reason_counts(
                        occluder_candidate_rejection_reasons,
                        placement.rejection_reasons,
                    )
                dynamic_trajectories = {
                    object_id: oracle_context.dynamic_object_future[object_id].copy()
                    for object_id in sorted(oracle_context.dynamic_object_future)
                }
                dynamic_specs = {
                    object_id: dict(oracle_context.dynamic_object_specs[object_id])
                    for object_id in sorted(oracle_context.dynamic_object_specs)
                }
                dynamic_trajectories[target.target_dynamic_object_id] = (
                    target.future_poses.copy()
                )
                dynamic_specs[target.target_dynamic_object_id] = dict(
                    target.footprint_spec
                )
                occluders = () if placement is None else (dict(placement.occluder),)
                blind_spot_config = {
                    "kind": event_kind,
                    "structural": (
                        None if structural is None else structural.as_dict()
                    ),
                    "occluder_ids": [
                        occluder["occluder_id"] for occluder in occluders
                    ],
                }
                world_id = "world-" + stable_digest(
                    base_state.state_id,
                    trajectory.trajectory_id,
                    event_index,
                    attempt_index,
                    event_kind,
                    target.target_dynamic_object_id,
                    generator_digest,
                    size=12,
                )
                metadata = {
                    "schema_version": SCHEMA_VERSION,
                    "event_kind": event_kind,
                    "trajectory_id": trajectory.trajectory_id,
                    "target_dynamic_object_id": target.target_dynamic_object_id,
                    "target_object_type": target.object_type,
                    "target_footprint_spec": target.footprint_spec,
                    "target_footprint_spec_digest": target.footprint_spec_digest,
                    "source_object_id": target.source_object_id,
                    "dynamic_object_snippet_id": target.snippet_id,
                    "target_current_pose": [
                        float(value) for value in target.current_pose
                    ],
                    "target_type_policy": policy.as_dict(),
                    "target_type_policy_digest": policy.digest,
                    "generator_config_digest": generator_digest,
                    "conflict_time_s": conflict_time_s,
                    "conflict_index": conflict_index,
                    "attempt_index": attempt_index,
                    "target_provenance": target.provenance,
                    "visibility_sequence": [
                        bool(value) for value in visibility_sequence
                    ],
                    "context_dynamic_object_ids": sorted(
                        oracle_context.dynamic_object_future
                    ),
                    "occluder_candidate_rejection_reasons": (
                        {} if placement is None else placement.rejection_reasons
                    ),
                }
                world = OracleWorld(
                    world_id=world_id,
                    base_state_id=base_state.state_id,
                    static_occupancy=world_occupancy.astype(np.float32),
                    dynamic_object_trajectories=dynamic_trajectories,
                    dynamic_object_specs=dynamic_specs,
                    occluders=occluders,
                    blind_spot_config=blind_spot_config,
                    random_seed=int(attempt_seed),
                    metadata=metadata,
                )
                validate_oracle_world(world, grid)
                event = GeneratedEvent(
                    event_kind=event_kind,
                    world=world,
                    target=target,
                    visibility_sequence=visibility_sequence,
                    conflict_time_s=conflict_time_s,
                    conflict_index=conflict_index,
                )
                events.append(event)
                event_kind_counts[event_kind] += 1
                accepted = True
            except _EventRejection as exc:
                reason = exc.reason
                _merge_reason_counts(
                    occluder_candidate_rejection_reasons,
                    exc.occluder_candidate_rejection_reasons,
                )
            except TransplantError as exc:
                reason = exc.reason
            except OccluderSamplingError as exc:
                reason = exc.reason
                _merge_reason_counts(
                    occluder_candidate_rejection_reasons,
                    exc.rejection_reasons,
                )

            object_type, footprint_kind, geometry_source = _bucket_key(target)
            _update_bucket(
                by_object_type,
                object_type,
                accepted=accepted,
                reason=reason,
            )
            _update_bucket(
                by_footprint_kind,
                footprint_kind,
                accepted=accepted,
                reason=reason,
            )
            _update_bucket(
                by_geometry_source,
                geometry_source,
                accepted=accepted,
                reason=reason,
            )
            if accepted:
                break
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        if not accepted:
            continue

    rejected_count = attempted_count - len(events)
    requested_event_kind_counts = {
        event_type: event_type_schedule.count(event_type)
        for event_type in _EVENT_TYPES
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "requested_event_count": desired_count,
        "attempted_count": attempted_count,
        "accepted_count": len(events),
        "rejected_count": rejected_count,
        "acceptance_rate": len(events) / desired_count,
        "attempt_acceptance_rate": (
            len(events) / attempted_count if attempted_count else 0.0
        ),
        "unaccepted_event_count": desired_count - len(events),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "occluder_candidate_rejection_reasons": dict(
            sorted(occluder_candidate_rejection_reasons.items())
        ),
        "requested_event_kind_counts": requested_event_kind_counts,
        "event_kind_counts": event_kind_counts,
        "by_object_type": _sort_buckets(by_object_type),
        "by_footprint_kind": _sort_buckets(by_footprint_kind),
        "by_geometry_source": _sort_buckets(by_geometry_source),
        "target_type_policy": policy.as_dict(),
        "target_type_policy_digest": policy.digest,
        "generator_config_digest": generator_digest,
    }
    return EventGenerationReport(events=tuple(events), summary=summary)
