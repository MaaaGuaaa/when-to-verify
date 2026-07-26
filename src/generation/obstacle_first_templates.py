"""Deterministic obstacle-target templates for SOP05R generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product
from numbers import Real
from typing import Any, Iterator, Mapping

import numpy as np

from src.contracts import (
    BaseState,
    OracleContext,
    build_grid_spec,
    encode_dataclass,
    validate_base_state,
    validate_dynamic_object_spec,
    validate_oracle_context,
)
from src.datasets.snippet_library import MOTION_SNIPPET_LAYOUT, MotionSnippet, SnippetLibrary
from src.geometry import (
    CircleFootprint,
    RectangleFootprint,
    footprint_aabb,
    inflate_footprint,
    intersects,
    rasterize_footprint,
    raycast_candidate_visibility,
    wrap_angle,
)
from src.utils.seeding import derive_seed, stable_digest

from .dynamic_object_transplant import (
    TransplantedDynamicObject,
    footprint_from_spec,
)
from .occluder_sampler import (
    swept_footprint_intersects_occupancy,
    synchronized_sweeps_intersect,
)
from .sop05r_contracts import (
    SOP05R_TEMPLATE_VERSION,
    Sop05rConfig,
    normalize_sop05r_config,
)


SOP05R_SNIPPET_TIME_TRANSFORM_VERSION = "sop05r_bounded_snippet_time_transform_v1"
SOP05R_JOINT_TRANSFORM_VERSION = "sop05r_obstacle_target_joint_se2_v1"


class Sop05rTemplateError(ValueError):
    """Raised for invalid template inputs or one auditable rejection."""

    def __init__(self, reason: str, message: str | None = None):
        super().__init__(message or reason)
        self.reason = reason


def _readonly_array(value: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class RectangleObstacle:
    obstacle_id: str
    obstacle_type: str
    pose: np.ndarray
    length_m: float
    width_m: float
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose", _readonly_array(self.pose, dtype=np.float64))

    @property
    def footprint(self) -> RectangleFootprint:
        return RectangleFootprint(self.length_m, self.width_m)

    def as_dict(self) -> dict[str, object]:
        return {
            "obstacle_id": self.obstacle_id,
            "obstacle_type": self.obstacle_type,
            "pose": [float(value) for value in self.pose],
            "length_m": self.length_m,
            "width_m": self.width_m,
            "source": self.source,
        }


@dataclass(frozen=True)
class ObstacleTargetTemplate:
    template_id: str
    schedule_rank: tuple[int, ...]
    obstacle: RectangleObstacle
    obstacle_mask: np.ndarray
    static_occupancy: np.ndarray
    target: TransplantedDynamicObject
    target_time_scale: float
    goal_bearing_rad: float
    goal_distance_m: float
    local_goal_world_pose: np.ndarray
    provenance: dict[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "obstacle_mask",
            _readonly_array(self.obstacle_mask, dtype=np.dtype(np.bool_)),
        )
        object.__setattr__(
            self,
            "static_occupancy",
            _readonly_array(self.static_occupancy, dtype=np.dtype(np.float32)),
        )
        object.__setattr__(
            self,
            "local_goal_world_pose",
            _readonly_array(self.local_goal_world_pose, dtype=np.dtype(np.float32)),
        )


@dataclass(frozen=True)
class TemplateEvaluation:
    template_id: str
    schedule_rank: tuple[int, ...]
    selection_rank: int
    template: ObstacleTargetTemplate | None
    rejection_reason: str | None
    provenance: dict[str, object]


@dataclass(frozen=True)
class _ScheduleItem:
    template_id: str
    schedule_rank: tuple[int, ...]
    obstacle_size_index: int
    relative_layout_index: int
    goal_bearing_index: int
    goal_distance_index: int
    target_time_scale_index: int
    snippet_index: int
    snippet: MotionSnippet


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise Sop05rTemplateError(
            "snippet_contract_invalid", "template evidence must be canonical JSON"
        ) from exc


def canonical_base_state_digest(base_state: BaseState) -> str:
    """Hash every BaseState field without mutating or normalizing the source."""

    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    arrays, metadata = encode_dataclass(base_state)
    digest = hashlib.sha256()
    digest.update(b"sop05r_base_state_digest_v1\0")
    digest.update(_canonical_json_bytes(metadata))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(_canonical_json_bytes(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_snippet(snippet: MotionSnippet) -> None:
    if not isinstance(snippet, MotionSnippet):
        raise TypeError("snippet must be a MotionSnippet")
    if snippet.object_type != "human":
        raise Sop05rTemplateError(
            "snippet_contract_invalid", "SOP05R v1 supports human snippets only"
        )
    try:
        validate_dynamic_object_spec(
            {"object_type": snippet.object_type, "footprint": snippet.footprint}
        )
    except ValueError as exc:
        raise Sop05rTemplateError("snippet_contract_invalid", str(exc)) from exc
    sample_count = int(MOTION_SNIPPET_LAYOUT["sample_count"])
    if (
        snippet.positions.shape != (sample_count, 2)
        or snippet.velocities.shape != (sample_count, 2)
        or snippet.headings.shape != (sample_count,)
        or snippet.positions.dtype != np.float32
        or snippet.velocities.dtype != np.float32
        or snippet.headings.dtype != np.float32
        or not np.isfinite(snippet.positions).all()
        or not np.isfinite(snippet.velocities).all()
        or not np.isfinite(snippet.headings).all()
    ):
        raise Sop05rTemplateError(
            "snippet_contract_invalid",
            "snippet motion arrays violate the frozen 23-sample float32 layout",
        )
    for name in (
        "snippet_id",
        "split",
        "source_recording_id",
        "source_session_id",
        "source_object_id",
    ):
        if not isinstance(getattr(snippet, name), str) or not getattr(snippet, name):
            raise Sop05rTemplateError(
                "snippet_contract_invalid", f"snippet {name} must be nonempty"
            )


def resample_sop05r_snippet(
    snippet: MotionSnippet,
    *,
    time_scale: object,
) -> np.ndarray:
    """Apply bounded SOP05R time scaling and reject source extrapolation."""

    _validate_snippet(snippet)
    scale = _finite_positive(time_scale, name="time_scale")
    sample_count = int(MOTION_SNIPPET_LAYOUT["sample_count"])
    sample_dt_s = float(MOTION_SNIPPET_LAYOUT["sample_dt_s"])
    current_index = int(MOTION_SNIPPET_LAYOUT["current_index"])
    source_times = np.arange(sample_count, dtype=np.float64) * sample_dt_s
    output_times = source_times.copy()
    current_time_s = current_index * sample_dt_s
    query_times = current_time_s + (output_times - current_time_s) * scale
    tolerance = 1e-9
    if (
        float(np.min(query_times)) < float(source_times[0]) - tolerance
        or float(np.max(query_times)) > float(source_times[-1]) + tolerance
    ):
        raise Sop05rTemplateError("source_extrapolation_required")
    query_times = np.clip(query_times, source_times[0], source_times[-1])
    positions = np.column_stack(
        (
            np.interp(query_times, source_times, snippet.positions[:, 0]),
            np.interp(query_times, source_times, snippet.positions[:, 1]),
        )
    )
    unwrapped_headings = np.unwrap(snippet.headings.astype(np.float64))
    headings = wrap_angle(np.interp(query_times, source_times, unwrapped_headings))
    result = np.column_stack((positions, headings)).astype(np.float32)
    if result.shape != (sample_count, 3) or not np.isfinite(result).all():
        raise Sop05rTemplateError("snippet_contract_invalid")
    return np.array(result, dtype=np.float32, order="C", copy=True)


def _as_config(config: Sop05rConfig | Mapping[str, Any]) -> Sop05rConfig:
    if isinstance(config, Sop05rConfig):
        return config
    return normalize_sop05r_config(config)


def _seed(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError("seed must be nonnegative")
    return result


def _human_snippets(
    snippet_libraries: Mapping[str, SnippetLibrary],
    *,
    split: str,
    limit: int,
    base_state_id: str,
    seed: int,
) -> tuple[MotionSnippet, ...]:
    if not isinstance(snippet_libraries, Mapping):
        raise TypeError("snippet_libraries must be a mapping")
    library = snippet_libraries.get("human")
    if not isinstance(library, SnippetLibrary) or library.object_type != "human":
        raise Sop05rTemplateError(
            "snippet_contract_invalid", "human snippet library is required"
        )
    if library.summary.get("split") not in (None, split):
        raise Sop05rTemplateError("snippet_split_mismatch", "library split mismatch")
    if library.split_provenance.get("split") not in (None, split):
        raise Sop05rTemplateError(
            "snippet_split_mismatch", "library provenance split mismatch"
        )
    snippets = []
    identities: set[tuple[str, str]] = set()
    for snippet in library.snippets:
        _validate_snippet(snippet)
        if snippet.split != split:
            raise Sop05rTemplateError(
                "snippet_split_mismatch",
                "snippet split differs from BaseState split",
            )
        identity = (snippet.snippet_id, snippet.source_object_id)
        if identity in identities:
            raise Sop05rTemplateError(
                "snippet_contract_invalid", "snippet identities must be unique"
            )
        identities.add(identity)
        snippets.append(snippet)
    if not snippets:
        raise Sop05rTemplateError(
            "snippet_contract_invalid", "human snippet library must not be empty"
        )
    regular_order = sorted(
        snippets,
        key=lambda snippet: (
            derive_seed(
                seed,
                "sop05r-snippet-window-v2",
                base_state_id,
                snippet.snippet_id,
                snippet.source_recording_id,
                snippet.source_object_id,
            ),
            snippet.snippet_id,
            snippet.source_recording_id,
            snippet.source_object_id,
        ),
    )
    seen_candidates = []
    for snippet in snippets:
        current_index = int(MOTION_SNIPPET_LAYOUT["current_index"])
        future_index = current_index + 10
        past = (
            snippet.positions[current_index].astype(np.float64)
            - snippet.positions[0].astype(np.float64)
        )
        future = (
            snippet.positions[future_index].astype(np.float64)
            - snippet.positions[current_index].astype(np.float64)
        )
        past_distance = float(np.linalg.norm(past))
        future_distance = float(np.linalg.norm(future))
        if past_distance < 0.35 or future_distance < 0.45:
            continue
        cosine = float(
            np.clip(
                np.dot(past, future) / (past_distance * future_distance),
                -1.0,
                1.0,
            )
        )
        turn_degrees = float(np.degrees(np.arccos(cosine)))
        if turn_degrees < 80.0:
            continue
        score = (
            turn_degrees
            + 10.0 * future_distance
            - 5.0 * abs(past_distance - 0.40)
        )
        seen_candidates.append((score, snippet))
    seen_candidates.sort(
        key=lambda row: (
            -row[0],
            row[1].snippet_id,
            row[1].source_recording_id,
            row[1].source_object_id,
        )
    )
    requested_count = min(limit, len(snippets))
    seen_quota = min(len(seen_candidates), int(round(0.8 * requested_count)))
    selected: list[MotionSnippet] = []
    selected_ids: set[str] = set()
    selected_source_objects: set[str] = set()
    for _, snippet in seen_candidates:
        if len(selected) >= seen_quota:
            break
        if snippet.source_object_id in selected_source_objects:
            continue
        selected.append(snippet)
        selected_ids.add(snippet.snippet_id)
        selected_source_objects.add(snippet.source_object_id)
    if len(selected) < seen_quota:
        for _, snippet in seen_candidates:
            if len(selected) >= seen_quota:
                break
            if snippet.snippet_id in selected_ids:
                continue
            selected.append(snippet)
            selected_ids.add(snippet.snippet_id)
            selected_source_objects.add(snippet.source_object_id)
    for snippet in regular_order:
        if len(selected) >= requested_count:
            break
        if (
            snippet.snippet_id in selected_ids
            or snippet.source_object_id in selected_source_objects
        ):
            continue
        selected.append(snippet)
        selected_ids.add(snippet.snippet_id)
        selected_source_objects.add(snippet.source_object_id)
    if len(selected) < requested_count:
        for snippet in regular_order:
            if len(selected) >= requested_count:
                break
            if snippet.snippet_id in selected_ids:
                continue
            selected.append(snippet)
            selected_ids.add(snippet.snippet_id)
    return tuple(selected)


def _schedule(
    *,
    base_state: BaseState,
    snippets: tuple[MotionSnippet, ...],
    config: Sop05rConfig,
    seed: int,
) -> tuple[_ScheduleItem, ...]:
    candidates: list[tuple[str, tuple[int, ...], MotionSnippet]] = []
    axes = product(
        range(len(config.template.obstacle_sizes)),
        range(len(config.template.relative_layouts)),
        range(len(config.template.goal_bearings_deg)),
        range(len(config.template.goal_distances_m)),
        range(len(config.template.target_time_scales)),
        range(len(snippets)),
    )
    for rank in axes:
        snippet = snippets[rank[5]]
        payload = {
            "domain": "sop05r_template_identity_v1",
            "template_schedule_version": SOP05R_TEMPLATE_VERSION,
            "base_state_id": base_state.state_id,
            "split": base_state.split,
            "seed": seed,
            "config_digest": config.digest,
            "schedule_rank": list(rank),
            "source_snippet_id": snippet.snippet_id,
            "source_object_id": snippet.source_object_id,
        }
        template_digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        order_digest = hashlib.sha256(
            _canonical_json_bytes(
                {
                    "domain": "sop05r_template_order_v1",
                    "template_digest": template_digest,
                    "seed": seed,
                }
            )
        ).hexdigest()
        candidates.append((order_digest, rank, snippet))
    candidates.sort(key=lambda item: (item[0], item[1]))
    result = []
    for _, rank, snippet in candidates[: config.generation.max_templates_per_base]:
        identity = {
            "domain": "sop05r_template_identity_v1",
            "template_schedule_version": SOP05R_TEMPLATE_VERSION,
            "base_state_id": base_state.state_id,
            "split": base_state.split,
            "seed": seed,
            "config_digest": config.digest,
            "schedule_rank": list(rank),
            "source_snippet_id": snippet.snippet_id,
            "source_object_id": snippet.source_object_id,
        }
        template_id = "template-" + hashlib.sha256(
            _canonical_json_bytes(identity)
        ).hexdigest()[:24]
        result.append(
            _ScheduleItem(
                template_id=template_id,
                schedule_rank=rank,
                obstacle_size_index=rank[0],
                relative_layout_index=rank[1],
                goal_bearing_index=rank[2],
                goal_distance_index=rank[3],
                target_time_scale_index=rank[4],
                snippet_index=rank[5],
                snippet=snippet,
            )
        )
    return tuple(result)


def _footprint_within_grid(footprint: object, poses: np.ndarray, grid: object) -> bool:
    x_min = -0.5 * float(grid.width) * float(grid.resolution_m)
    x_max = 0.5 * float(grid.width) * float(grid.resolution_m)
    y_min = -0.5 * float(grid.height) * float(grid.resolution_m)
    y_max = 0.5 * float(grid.height) * float(grid.resolution_m)
    for pose in poses:
        bounds = footprint_aabb(footprint, pose)
        if (
            bounds[0] < x_min
            or bounds[1] >= x_max
            or bounds[2] < y_min
            or bounds[3] >= y_max
        ):
            return False
    return True


def _context_footprints(oracle_context: OracleContext) -> dict[str, object]:
    return {
        object_id: footprint_from_spec(spec)
        for object_id, spec in oracle_context.dynamic_object_specs.items()
    }


def _target_identity(
    *,
    item: _ScheduleItem,
    seed: int,
    transformed_poses: np.ndarray,
) -> str:
    canonical = np.ascontiguousarray(transformed_poses, dtype=np.dtype("<f4"))
    digest = hashlib.sha256()
    digest.update(b"sop05r_target_identity_v1\0")
    digest.update(item.template_id.encode("ascii"))
    digest.update(str(seed).encode("ascii"))
    digest.update(item.snippet.snippet_id.encode("utf-8"))
    digest.update(canonical.tobytes(order="C"))
    return f"generated::human::{digest.hexdigest()[:24]}"


def _build_target_and_obstacle(
    *,
    item: _ScheduleItem,
    config: Sop05rConfig,
    seed: int,
) -> tuple[RectangleObstacle, TransplantedDynamicObject, np.ndarray, dict[str, object]]:
    obstacle_size = config.template.obstacle_sizes[item.obstacle_size_index]
    layout = config.template.relative_layouts[item.relative_layout_index]
    bearing_rad = np.deg2rad(
        config.template.goal_bearings_deg[item.goal_bearing_index]
    )
    goal_distance_m = config.template.goal_distances_m[item.goal_distance_index]
    time_scale = config.template.target_time_scales[item.target_time_scale_index]
    resampled = resample_sop05r_snippet(item.snippet, time_scale=time_scale)
    task_direction = np.asarray(
        [np.cos(bearing_rad), np.sin(bearing_rad)], dtype=np.float64
    )
    task_normal = np.asarray(
        [-task_direction[1], task_direction[0]], dtype=np.float64
    )
    layout_sign = 1.0 if layout == "target_side" else -1.0
    obstacle_center = (
        0.60 * goal_distance_m * task_direction
        + layout_sign
        * (0.5 * obstacle_size.length_m + 0.42)
        * task_normal
    )
    obstacle_yaw = float(wrap_angle(bearing_rad + 0.5 * np.pi))
    target_radius_m = float(item.snippet.footprint["radius_m"])
    target_current_xy = (
        obstacle_center
        + task_direction
        * (0.5 * obstacle_size.width_m + target_radius_m + 0.30)
        + layout_sign * task_normal * (0.18 * obstacle_size.length_m)
    )
    desired_target_heading = float(
        wrap_angle(bearing_rad - layout_sign * 0.5 * np.pi)
    )
    source_current_index = int(MOTION_SNIPPET_LAYOUT["current_index"])
    rotation_angle = float(
        wrap_angle(desired_target_heading - float(resampled[source_current_index, 2]))
    )
    cosine = np.cos(rotation_angle)
    sine = np.sin(rotation_angle)
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    source_current_xy = resampled[source_current_index, :2].astype(np.float64)
    translation = target_current_xy - source_current_xy @ rotation.T
    transformed_positions = resampled[:, :2].astype(np.float64) @ rotation.T + translation
    transformed_headings = wrap_angle(
        resampled[:, 2].astype(np.float64) + rotation_angle
    )
    transformed_poses = np.column_stack(
        (transformed_positions, transformed_headings)
    ).astype(np.float32)
    canonical_obstacle_xy = (obstacle_center - translation) @ rotation
    canonical_obstacle_pose = np.asarray(
        [
            canonical_obstacle_xy[0],
            canonical_obstacle_xy[1],
            float(wrap_angle(obstacle_yaw - rotation_angle)),
        ],
        dtype=np.float64,
    )
    reconstructed_obstacle_xy = canonical_obstacle_pose[:2] @ rotation.T + translation
    obstacle_pose = np.asarray(
        [
            reconstructed_obstacle_xy[0],
            reconstructed_obstacle_xy[1],
            float(wrap_angle(canonical_obstacle_pose[2] + rotation_angle)),
        ],
        dtype=np.float64,
    )
    joint_transform = {
        "version": SOP05R_JOINT_TRANSFORM_VERSION,
        "rotation_rad": rotation_angle,
        "translation_xy_m": [float(value) for value in translation],
        "canonical_obstacle_pose": [
            float(value) for value in canonical_obstacle_pose
        ],
    }
    obstacle = RectangleObstacle(
        obstacle_id=f"obstacle-{item.template_id.removeprefix('template-')}",
        obstacle_type=obstacle_size.obstacle_type,
        pose=obstacle_pose,
        length_m=obstacle_size.length_m,
        width_m=obstacle_size.width_m,
        source="programmatic_rectangle",
    )
    footprint_spec = {
        "object_type": item.snippet.object_type,
        "footprint": dict(item.snippet.footprint),
    }
    footprint_digest = stable_digest(
        json.dumps(
            footprint_spec,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        size=16,
    )
    target_id = _target_identity(
        item=item,
        seed=seed,
        transformed_poses=transformed_poses,
    )
    target_provenance = {
        **item.snippet.provenance,
        "source_split": item.snippet.split,
        "source_recording_id": item.snippet.source_recording_id,
        "source_session_id": item.snippet.source_session_id,
        "source_object_id": item.snippet.source_object_id,
        "source_snippet_id": item.snippet.snippet_id,
        "time_transform_version": SOP05R_SNIPPET_TIME_TRANSFORM_VERSION,
        "time_scale": time_scale,
        "joint_se2_transform": joint_transform,
        "template_id": item.template_id,
        "seed": seed,
    }
    history = np.array(
        transformed_poses[:8], dtype=np.float32, order="C", copy=True
    )
    future = np.array(
        transformed_poses[8:], dtype=np.float32, order="C", copy=True
    )
    target = TransplantedDynamicObject(
        target_dynamic_object_id=target_id,
        source_object_id=item.snippet.source_object_id,
        snippet_id=item.snippet.snippet_id,
        object_type=item.snippet.object_type,
        footprint_spec=footprint_spec,
        footprint_spec_digest=footprint_digest,
        history_poses=history,
        current_pose=history[-1].copy(),
        future_poses=future,
        provenance=target_provenance,
    )
    goal_pose = np.asarray(
        [
            goal_distance_m * task_direction[0],
            goal_distance_m * task_direction[1],
            bearing_rad,
        ],
        dtype=np.float64,
    )
    return obstacle, target, goal_pose, joint_transform


def _validate_template_geometry(
    *,
    obstacle: RectangleObstacle,
    target: TransplantedDynamicObject,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    grid = build_grid_spec(dict(base_config))
    obstacle_poses = obstacle.pose.reshape(1, 3)
    if not _footprint_within_grid(obstacle.footprint, obstacle_poses, grid):
        raise Sop05rTemplateError("obstacle_out_of_bounds")
    obstacle_mask = rasterize_footprint(obstacle.footprint, obstacle.pose, grid)
    if not np.any(obstacle_mask):
        raise Sop05rTemplateError("obstacle_mask_empty")
    source_static = (
        np.zeros((grid.height, grid.width), dtype=np.bool_)
        if base_state.static_map_local is None
        else np.asarray(base_state.static_map_local != 0, dtype=np.bool_)
    )
    if np.any(obstacle_mask & source_static):
        raise Sop05rTemplateError("source_static_overlap")
    robot = base_config["robot"]
    robot_footprint = inflate_footprint(
        RectangleFootprint(
            float(robot["length_m"]), float(robot["width_m"])
        ),
        float(robot["inflation_m"]),
    )
    if any(
        intersects(robot_footprint, pose, obstacle.footprint, obstacle.pose)
        for pose in base_state.robot_history
    ):
        raise Sop05rTemplateError("robot_history_overlap")
    context_footprints = _context_footprints(oracle_context)
    for object_id in sorted(context_footprints):
        context_poses = np.vstack(
            (
                oracle_context.dynamic_object_history[object_id],
                oracle_context.dynamic_object_future[object_id],
            )
        )
        if swept_footprint_intersects_occupancy(
            context_footprints[object_id],
            context_poses,
            obstacle_mask,
            grid=grid,
        ):
            raise Sop05rTemplateError("context_overlap")
    target_footprint = footprint_from_spec(target.footprint_spec)
    target_poses = np.vstack((target.history_poses, target.future_poses))
    if not _footprint_within_grid(target_footprint, target_poses, grid):
        raise Sop05rTemplateError("target_out_of_bounds")
    if swept_footprint_intersects_occupancy(
        target_footprint,
        target_poses,
        obstacle_mask,
        grid=grid,
    ):
        raise Sop05rTemplateError("target_obstacle_collision")
    if np.any(source_static) and swept_footprint_intersects_occupancy(
        target_footprint,
        target_poses,
        source_static,
        grid=grid,
    ):
        raise Sop05rTemplateError("target_source_static_collision")
    for object_id in sorted(context_footprints):
        context_poses = np.vstack(
            (
                oracle_context.dynamic_object_history[object_id],
                oracle_context.dynamic_object_future[object_id],
            )
        )
        if synchronized_sweeps_intersect(
            target_footprint,
            target_poses,
            context_footprints[object_id],
            context_poses,
            grid=grid,
        ):
            raise Sop05rTemplateError("target_context_collision")
    current_context = np.zeros((grid.height, grid.width), dtype=np.bool_)
    for object_id in sorted(context_footprints):
        current_context |= rasterize_footprint(
            context_footprints[object_id],
            oracle_context.dynamic_object_history[object_id][-1],
            grid,
        )
    target_mask = rasterize_footprint(
        target_footprint, target.current_pose, grid
    )
    currently_visible = raycast_candidate_visibility(
        source_static | obstacle_mask | current_context,
        target_mask,
        grid,
        sensor_pose=base_state.robot_history[-1],
    )
    if bool(currently_visible.any()):
        raise Sop05rTemplateError("target_currently_visible")
    augmented_static = np.asarray(
        source_static | obstacle_mask, dtype=np.float32
    )
    return obstacle_mask, augmented_static


def _evaluate_item(
    *,
    item: _ScheduleItem,
    selection_rank: int,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    config: Sop05rConfig,
    seed: int,
) -> TemplateEvaluation:
    provenance = {
        "template_schedule_version": SOP05R_TEMPLATE_VERSION,
        "config_digest": config.digest,
        "base_state_id": base_state.state_id,
        "source_snippet_id": item.snippet.snippet_id,
        "source_object_id": item.snippet.source_object_id,
        "schedule_rank": list(item.schedule_rank),
        "selection_rank": selection_rank,
        "seed": seed,
    }
    try:
        obstacle, target, goal_pose, joint_transform = _build_target_and_obstacle(
            item=item,
            config=config,
            seed=seed,
        )
        obstacle_mask, static_occupancy = _validate_template_geometry(
            obstacle=obstacle,
            target=target,
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
        )
    except Sop05rTemplateError as exc:
        if exc.reason not in config.rejection_reasons:
            raise RuntimeError(
                f"unregistered SOP05R template rejection reason: {exc.reason}"
            ) from exc
        return TemplateEvaluation(
            template_id=item.template_id,
            schedule_rank=item.schedule_rank,
            selection_rank=selection_rank,
            template=None,
            rejection_reason=exc.reason,
            provenance=provenance,
        )
    obstacle_size = config.template.obstacle_sizes[item.obstacle_size_index]
    bearing_rad = float(
        np.deg2rad(config.template.goal_bearings_deg[item.goal_bearing_index])
    )
    target_time_scale = config.template.target_time_scales[
        item.target_time_scale_index
    ]
    template_provenance = {
        **provenance,
        "obstacle_size_template_id": obstacle_size.template_id,
        "relative_layout": config.template.relative_layouts[
            item.relative_layout_index
        ],
        "joint_se2_transform": joint_transform,
        "source_static_geometry_preserved": True,
        "base_state_digest": canonical_base_state_digest(base_state),
    }
    template = ObstacleTargetTemplate(
        template_id=item.template_id,
        schedule_rank=item.schedule_rank,
        obstacle=obstacle,
        obstacle_mask=obstacle_mask,
        static_occupancy=static_occupancy,
        target=target,
        target_time_scale=target_time_scale,
        goal_bearing_rad=bearing_rad,
        goal_distance_m=config.template.goal_distances_m[
            item.goal_distance_index
        ],
        local_goal_world_pose=goal_pose,
        provenance=template_provenance,
    )
    return TemplateEvaluation(
        template_id=item.template_id,
        schedule_rank=item.schedule_rank,
        selection_rank=selection_rank,
        template=template,
        rejection_reason=None,
        provenance=provenance,
    )


def iter_obstacle_target_templates(
    *,
    base_state: BaseState,
    oracle_context: OracleContext,
    snippet_libraries: Mapping[str, SnippetLibrary],
    base_config: Mapping[str, Any],
    config: Sop05rConfig | Mapping[str, Any],
    seed: object,
) -> Iterator[TemplateEvaluation]:
    """Yield a bounded stable schedule with explicit rejection evidence."""

    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(oracle_context, OracleContext):
        raise TypeError("oracle_context must be an OracleContext")
    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    normalized = _as_config(config)
    stable_seed = _seed(seed)
    grid = build_grid_spec(dict(base_config))
    validate_base_state(base_state, grid)
    validate_oracle_context(oracle_context, grid)
    if oracle_context.base_state_id != base_state.state_id:
        raise Sop05rTemplateError(
            "snippet_contract_invalid", "oracle_context base_state_id mismatch"
        )
    snippets = _human_snippets(
        snippet_libraries,
        split=base_state.split,
        limit=normalized.generation.max_target_snippets_per_template,
        base_state_id=base_state.state_id,
        seed=stable_seed,
    )
    schedule = _schedule(
        base_state=base_state,
        snippets=snippets,
        config=normalized,
        seed=stable_seed,
    )
    for selection_rank, item in enumerate(schedule):
        yield _evaluate_item(
            item=item,
            selection_rank=selection_rank,
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
            config=normalized,
            seed=stable_seed,
        )
