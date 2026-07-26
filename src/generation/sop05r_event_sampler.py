"""Post-plan target join and mother-event gates for SOP05R."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, Callable, Mapping

import numpy as np

from src.contracts import (
    ARRAY_DTYPE,
    BaseState,
    GridSpec,
    OracleContext,
    OracleWorld,
    build_grid_spec,
    validate_base_state,
    validate_oracle_context,
    validate_oracle_world,
)
from src.geometry import (
    CircleFootprint,
    Footprint,
    RectangleFootprint,
    footprint_aabb,
    inflate_footprint,
    intersects,
    rasterize_footprint,
    raycast_candidate_visibility,
    signed_clearance,
    wrap_angle,
)
from src.planning.obstacle_corner_planner import (
    ObstaclePlanResult,
    ObstaclePlannedRoute,
    ObstaclePlannerRequest,
    plan_obstacle_routes,
)
from src.utils.seeding import derive_seed, stable_digest

from .dynamic_object_transplant import (
    TransplantedDynamicObject,
    footprint_from_spec,
)
from .event_sampler import GeneratedEvent
from .event_target_motion_shard import (
    build_event_target_motion_world_metadata,
    compute_motion_array_digest,
    create_event_target_motion_record,
    validate_event_target_motion_world_join,
)
from .history_visibility import (
    HISTORY_VISIBILITY_REGIMES,
    SEEN_THEN_OCCLUDED,
    UNSEEN_IN_HISTORY_WINDOW,
    HistoryVisibilityAssessment,
    classify_sop05r_history,
)
from .obstacle_first_templates import ObstacleTargetTemplate
from .occluder_sampler import (
    swept_footprint_intersects_occupancy,
    synchronized_sweeps_intersect,
)
from .sop05r_contracts import (
    SOP05R_GENERATOR_VERSION,
    SOP05R_PLANNER_SLOT_IDS,
    Sop05rConfig,
    normalize_sop05r_config,
)
from .sop05r_trajectory_store import Sop05rTrajectoryRecord


SOP05R_COLLISION_EVIDENCE_VERSION = "sop05r_continuous_collision_evidence_v1"
SOP05R_ROUTE_CONTACT_ALIGNMENT_VERSION = "sop05r_route_contact_alignment_v1"
_SOP05R_TARGET_POLICY = {
    "whitelist": ["human"],
    "weights": {
        "human": 1.0,
        "carried_object": 0.0,
        "unknown_dynamic": 0.0,
    },
}
_SOP05R_TARGET_POLICY_DIGEST = stable_digest(
    json.dumps(
        _SOP05R_TARGET_POLICY,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ),
    size=16,
)


def _readonly_array(value: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


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
        raise ValueError("SOP05R event identity must be canonical JSON") from exc


@dataclass(frozen=True)
class ContinuousCollisionEvidence:
    version: str
    continuous_collision: bool
    minimum_clearance_m: float
    minimum_clearance_time_s: float
    first_collision_time_s: float | None
    robot_pose_at_first_collision: np.ndarray | None
    target_pose_at_first_collision: np.ndarray | None

    def __post_init__(self) -> None:
        if self.version != SOP05R_COLLISION_EVIDENCE_VERSION:
            raise ValueError("continuous collision evidence version mismatch")
        values = (self.minimum_clearance_m, self.minimum_clearance_time_s)
        if not np.isfinite(values).all() or self.minimum_clearance_time_s < 0.0:
            raise ValueError("continuous collision evidence scalars are invalid")
        if self.continuous_collision:
            if self.first_collision_time_s is None:
                raise ValueError("collision evidence must include first collision time")
            if self.minimum_clearance_m > 0.0:
                raise ValueError("collision evidence minimum clearance must be nonpositive")
            for name in (
                "robot_pose_at_first_collision",
                "target_pose_at_first_collision",
            ):
                pose = getattr(self, name)
                if (
                    not isinstance(pose, np.ndarray)
                    or pose.shape != (3,)
                    or not np.isfinite(pose).all()
                ):
                    raise ValueError(f"{name} must be a finite pose")
                object.__setattr__(
                    self,
                    name,
                    _readonly_array(pose, dtype=np.dtype(np.float64)),
                )
        elif any(
            value is not None
            for value in (
                self.first_collision_time_s,
                self.robot_pose_at_first_collision,
                self.target_pose_at_first_collision,
            )
        ):
            raise ValueError("noncollision evidence must not include collision fields")


@dataclass(frozen=True)
class RouteCollisionEvidence:
    route_id: str
    minimum_clearance_m: float
    first_collision_time_s: float
    conflict_index: int
    conflict_point: np.ndarray
    conflict_path_fraction: float
    goal_forward_distance_m: float
    continuous_evidence: ContinuousCollisionEvidence

    def __post_init__(self) -> None:
        point = np.asarray(self.conflict_point)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("conflict_point must be finite [2]")
        object.__setattr__(
            self,
            "conflict_point",
            _readonly_array(point, dtype=np.dtype(np.float64)),
        )


@dataclass(frozen=True)
class Sop05rMotherCandidate:
    event: GeneratedEvent
    template: ObstacleTargetTemplate
    planner_result: ObstaclePlanResult
    nominal_route: ObstaclePlannedRoute
    alternative_routes: tuple[ObstaclePlannedRoute, ...]
    collision_evidence: RouteCollisionEvidence
    history_assessment: HistoryVisibilityAssessment
    trajectory_record: Sop05rTrajectoryRecord


@dataclass(frozen=True)
class Sop05rTemplateEvaluation:
    template_id: str
    planner_result: ObstaclePlanResult
    mother: Sop05rMotherCandidate | None
    rejection_reason: str | None
    evidence: Mapping[str, object]


@dataclass(frozen=True)
class RouteContactAlignmentCandidate:
    identity: str
    route: ObstaclePlannedRoute
    conflict_index: int
    anchor_time_s: float
    estimated_path_fraction: float
    estimated_goal_forward_distance_m: float
    direction_id: str
    crossing_direction: np.ndarray
    contact_id: str
    contact_normal: np.ndarray

    def __post_init__(self) -> None:
        for name in ("crossing_direction", "contact_normal"):
            direction = np.asarray(getattr(self, name))
            if (
                direction.shape != (2,)
                or direction.dtype.kind not in "iuf"
                or not np.isfinite(direction).all()
                or not np.isclose(
                    np.linalg.norm(direction), 1.0, rtol=0.0, atol=1e-9
                )
            ):
                raise ValueError(f"{name} must be a finite unit [2] vector")
            object.__setattr__(
                self,
                name,
                _readonly_array(direction, dtype=np.dtype(np.float64)),
            )


@dataclass(frozen=True)
class _AcceptedRouteContactJoin:
    template: ObstacleTargetTemplate
    nominal: ObstaclePlannedRoute
    alternatives: tuple[ObstaclePlannedRoute, ...]
    collision: RouteCollisionEvidence
    route_evidence: Mapping[str, RouteCollisionEvidence | None]
    history_visibility: np.ndarray
    history_assessment: HistoryVisibilityAssessment
    alignment_identity: str


def _interpolate_pose(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    result = np.empty(3, dtype=np.float64)
    result[:2] = (1.0 - fraction) * start[:2] + fraction * end[:2]
    result[2] = wrap_angle(
        start[2] + fraction * float(wrap_angle(end[2] - start[2]))
    )
    return result


def _motion_radius(footprint: Footprint) -> float:
    if isinstance(footprint, CircleFootprint):
        return footprint.radius_m
    if isinstance(footprint, RectangleFootprint):
        return 0.5 * float(np.hypot(footprint.length_m, footprint.width_m))
    raise TypeError("footprint must be a supported geometry footprint")


def align_target_to_route_contact(
    *,
    target: TransplantedDynamicObject,
    route: ObstaclePlannedRoute,
    conflict_index: int,
    crossing_direction: object,
    contact_normal: object | None = None,
    base_config: Mapping[str, Any],
    template_id: str,
    seed: int,
) -> TransplantedDynamicObject:
    """Rigidly align a measured target to first contact with one planned route."""

    if not isinstance(target, TransplantedDynamicObject):
        raise TypeError("target must be a TransplantedDynamicObject")
    if not isinstance(route, ObstaclePlannedRoute):
        raise TypeError("route must be an ObstaclePlannedRoute")
    if isinstance(conflict_index, (bool, np.bool_)) or not isinstance(
        conflict_index, (int, np.integer)
    ):
        raise TypeError("conflict_index must be an integer")
    anchor_index = int(conflict_index)
    if not 0 <= anchor_index < route.poses_world.shape[0]:
        raise ValueError("conflict_index lies outside the planned route")
    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    if not isinstance(template_id, str) or not template_id:
        raise ValueError("template_id must be nonempty")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")
    direction = np.asarray(crossing_direction)
    if (
        direction.shape != (2,)
        or direction.dtype.kind not in "iuf"
        or not np.isfinite(direction).all()
    ):
        raise ValueError("crossing_direction must be a finite numeric [2] vector")
    direction = direction.astype(np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-9:
        raise ValueError("crossing_direction must be nonzero")
    direction /= direction_norm
    normal = direction if contact_normal is None else np.asarray(contact_normal)
    if (
        normal.shape != (2,)
        or normal.dtype.kind not in "iuf"
        or not np.isfinite(normal).all()
    ):
        raise ValueError("contact_normal must be a finite numeric [2] vector")
    normal = normal.astype(np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        raise ValueError("contact_normal must be nonzero")
    normal /= normal_norm

    target_footprint = footprint_from_spec(target.footprint_spec)
    if not isinstance(target_footprint, CircleFootprint):
        raise ValueError("SOP05R route contact alignment requires a circular human target")
    source_poses = np.vstack((target.history_poses, target.future_poses)).astype(
        np.float64
    )
    if source_poses.shape != (40, 3) or not np.isfinite(source_poses).all():
        raise ValueError("target motion must contain the frozen 23-pose layout")
    source_current = source_poses[7, :2]
    source_anchor = source_poses[8 + anchor_index, :2]
    source_delta = source_anchor - source_current
    source_delta_norm = float(np.linalg.norm(source_delta))
    if source_delta_norm <= 1e-9:
        raise ValueError("target source motion is stationary at the contact anchor")
    rotation_angle = float(
        wrap_angle(
            np.arctan2(direction[1], direction[0])
            - np.arctan2(source_delta[1], source_delta[0])
        )
    )
    cosine = float(np.cos(rotation_angle))
    sine = float(np.sin(rotation_angle))
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    transformed_anchor_yaw = float(
        wrap_angle(source_poses[8 + anchor_index, 2] + rotation_angle)
    )
    robot_pose = route.poses_world[anchor_index].astype(np.float64)
    robot_footprint = _robot_footprint(base_config)
    penetration_m = min(
        0.02,
        0.20 * build_grid_spec(dict(base_config)).resolution_m,
    )
    target_clearance = -penetration_m
    lower_distance = 0.0
    upper_distance = (
        _motion_radius(robot_footprint)
        + target_footprint.radius_m
        + 1.0
    )
    for _ in range(64):
        distance = 0.5 * (lower_distance + upper_distance)
        target_pose = np.asarray(
            [
                robot_pose[0] - normal[0] * distance,
                robot_pose[1] - normal[1] * distance,
                transformed_anchor_yaw,
            ],
            dtype=np.float64,
        )
        clearance = signed_clearance(
            robot_footprint,
            robot_pose,
            target_footprint,
            target_pose,
        )
        if clearance <= target_clearance:
            lower_distance = distance
        else:
            upper_distance = distance
    contact_distance = 0.5 * (lower_distance + upper_distance)
    target_anchor = robot_pose[:2] - normal * contact_distance
    transformed_positions = (
        (source_poses[:, :2] - source_anchor) @ rotation.T + target_anchor
    )
    transformed_headings = wrap_angle(source_poses[:, 2] + rotation_angle)
    transformed = np.column_stack(
        (transformed_positions, transformed_headings)
    ).astype(np.float32)
    identity_digest = hashlib.sha256()
    identity_digest.update(b"sop05r_route_contact_target_v1\0")
    identity_digest.update(
        _canonical_json_bytes(
            {
                "source_target_id": target.target_dynamic_object_id,
                "template_id": template_id,
                "route_id": route.trajectory.trajectory_id,
                "conflict_index": anchor_index,
                "crossing_direction": [float(value) for value in direction],
                "contact_normal": [float(value) for value in normal],
                "seed": int(seed),
            }
        )
    )
    identity_digest.update(
        np.ascontiguousarray(transformed, dtype=np.dtype("<f4")).tobytes(order="C")
    )
    target_id = (
        f"generated::{target.object_type}::{identity_digest.hexdigest()[:24]}"
    )
    alignment = {
        "version": SOP05R_ROUTE_CONTACT_ALIGNMENT_VERSION,
        "route_id": route.trajectory.trajectory_id,
        "conflict_index": anchor_index,
        "conflict_time_s": (anchor_index + 1)
        * float(base_config["bev"]["future_dt_s"]),
        "crossing_direction": [float(value) for value in direction],
        "contact_normal": [float(value) for value in normal],
        "rotation_rad": rotation_angle,
        "target_anchor_xy_m": [float(value) for value in target_anchor],
        "contact_distance_m": contact_distance,
        "contact_penetration_m": penetration_m,
        "source_target_id": target.target_dynamic_object_id,
        "template_id": template_id,
        "seed": int(seed),
    }
    return replace(
        target,
        target_dynamic_object_id=target_id,
        history_poses=np.array(
            transformed[:8], dtype=np.float32, order="C", copy=True
        ),
        current_pose=np.array(
            transformed[7], dtype=np.float32, order="C", copy=True
        ),
        future_poses=np.array(
            transformed[8:], dtype=np.float32, order="C", copy=True
        ),
        provenance={
            **target.provenance,
            "route_contact_alignment": alignment,
        },
    )


def compute_continuous_collision_evidence(
    *,
    robot_footprint: Footprint,
    robot_poses: np.ndarray,
    target_footprint: Footprint,
    target_poses: np.ndarray,
    dt_s: object,
    spatial_resolution_m: object,
) -> ContinuousCollisionEvidence:
    """Densely evaluate synchronized SE(2) motion and certify contact."""

    dt = _finite_positive(dt_s, name="dt_s")
    resolution = _finite_positive(
        spatial_resolution_m, name="spatial_resolution_m"
    )
    robot = np.asarray(robot_poses)
    target = np.asarray(target_poses)
    if (
        robot.ndim != 2
        or robot.shape[1:] != (3,)
        or target.shape != robot.shape
        or robot.shape[0] < 2
        or robot.dtype.kind not in "iuf"
        or target.dtype.kind not in "iuf"
        or not np.isfinite(robot).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("synchronized poses must be matching finite [T, 3] arrays")
    robot = robot.astype(np.float64)
    target = target.astype(np.float64)
    robot_radius = _motion_radius(robot_footprint)
    target_radius = _motion_radius(target_footprint)
    dense_times = [0.0]
    dense_robot = [robot[0]]
    dense_target = [target[0]]
    for interval, (robot_start, robot_end, target_start, target_end) in enumerate(
        zip(robot[:-1], robot[1:], target[:-1], target[1:], strict=True)
    ):
        robot_motion = float(np.linalg.norm(robot_end[:2] - robot_start[:2]))
        robot_motion += robot_radius * abs(
            float(wrap_angle(robot_end[2] - robot_start[2]))
        )
        target_motion = float(np.linalg.norm(target_end[:2] - target_start[:2]))
        target_motion += target_radius * abs(
            float(wrap_angle(target_end[2] - target_start[2]))
        )
        subdivisions = max(
            1,
            int(np.ceil(robot_motion / resolution)),
            int(np.ceil(target_motion / resolution)),
        )
        for subdivision in range(1, subdivisions + 1):
            fraction = subdivision / subdivisions
            dense_times.append((interval + fraction) * dt)
            dense_robot.append(_interpolate_pose(robot_start, robot_end, fraction))
            dense_target.append(_interpolate_pose(target_start, target_end, fraction))
    clearances = np.asarray(
        [
            signed_clearance(
                robot_footprint,
                robot_pose,
                target_footprint,
                target_pose,
            )
            for robot_pose, target_pose in zip(dense_robot, dense_target, strict=True)
        ],
        dtype=np.float64,
    )
    minimum_index = int(np.argmin(clearances))
    colliding = np.flatnonzero(clearances <= 0.0)
    authority_collision = synchronized_sweeps_intersect(
        robot_footprint,
        robot,
        target_footprint,
        target,
        grid=GridSpec(
            height=1,
            width=1,
            history_steps=1,
            future_steps=robot.shape[0] - 1,
            resolution_m=resolution,
        ),
    )
    if authority_collision and colliding.size == 0:
        colliding = np.asarray([minimum_index], dtype=np.int64)
        clearances[minimum_index] = min(0.0, clearances[minimum_index])
    if colliding.size:
        first = int(colliding[0])
        return ContinuousCollisionEvidence(
            version=SOP05R_COLLISION_EVIDENCE_VERSION,
            continuous_collision=True,
            minimum_clearance_m=float(clearances[minimum_index]),
            minimum_clearance_time_s=float(dense_times[minimum_index]),
            first_collision_time_s=float(dense_times[first]),
            robot_pose_at_first_collision=np.asarray(dense_robot[first]),
            target_pose_at_first_collision=np.asarray(dense_target[first]),
        )
    return ContinuousCollisionEvidence(
        version=SOP05R_COLLISION_EVIDENCE_VERSION,
        continuous_collision=False,
        minimum_clearance_m=float(clearances[minimum_index]),
        minimum_clearance_time_s=float(dense_times[minimum_index]),
        first_collision_time_s=None,
        robot_pose_at_first_collision=None,
        target_pose_at_first_collision=None,
    )


def _as_config(config: Sop05rConfig | Mapping[str, Any]) -> Sop05rConfig:
    if isinstance(config, Sop05rConfig):
        return config
    return normalize_sop05r_config(config)


def _robot_footprint(base_config: Mapping[str, Any]) -> RectangleFootprint:
    robot = base_config["robot"]
    return inflate_footprint(
        RectangleFootprint(
            float(robot["length_m"]), float(robot["width_m"])
        ),
        float(robot["inflation_m"]),
    )


def _footprint_within_grid(
    footprint: Footprint,
    poses: np.ndarray,
    grid: GridSpec,
) -> bool:
    x_min = -0.5 * grid.width * grid.resolution_m
    x_max = 0.5 * grid.width * grid.resolution_m
    y_min = -0.5 * grid.height * grid.resolution_m
    y_max = 0.5 * grid.height * grid.resolution_m
    return all(
        bounds[0] >= x_min
        and bounds[1] < x_max
        and bounds[2] >= y_min
        and bounds[3] < y_max
        for bounds in (footprint_aabb(footprint, pose) for pose in poses)
    )


def _target_physics_rejection(
    *,
    template: ObstacleTargetTemplate,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
) -> str | None:
    grid = build_grid_spec(dict(base_config))
    target = template.target
    arrays = (target.history_poses, target.current_pose, target.future_poses)
    if (
        target.history_poses.shape != (grid.history_steps, 3)
        or target.current_pose.shape != (3,)
        or target.future_poses.shape != (grid.future_steps, 3)
        or any(array.dtype != ARRAY_DTYPE for array in arrays)
        or not all(np.isfinite(array).all() for array in arrays)
        or not np.array_equal(target.current_pose, target.history_poses[-1])
    ):
        return "target_motion_contract_invalid"
    all_poses = np.vstack((target.history_poses, target.future_poses))
    target_footprint = footprint_from_spec(target.footprint_spec)
    if not _footprint_within_grid(target_footprint, all_poses, grid):
        return "target_out_of_bounds"
    if swept_footprint_intersects_occupancy(
        target_footprint,
        all_poses,
        template.obstacle_mask,
        grid=grid,
    ):
        return "target_obstacle_collision"
    source_static = (
        np.zeros((grid.height, grid.width), dtype=np.bool_)
        if base_state.static_map_local is None
        else np.asarray(base_state.static_map_local != 0, dtype=np.bool_)
    )
    if np.any(source_static) and swept_footprint_intersects_occupancy(
        target_footprint,
        all_poses,
        source_static,
        grid=grid,
    ):
        return "target_source_static_collision"
    robot = _robot_footprint(base_config)
    if intersects(robot, base_state.robot_history[-1], target_footprint, target.current_pose):
        return "target_current_robot_overlap"
    dt_s = float(base_config["bev"]["future_dt_s"])
    velocities = np.diff(all_poses[:, :2].astype(np.float64), axis=0) / dt_s
    speeds = np.linalg.norm(velocities, axis=1)
    accelerations = np.diff(velocities, axis=0) / dt_s
    dynamic_config = base_config["dynamic_objects"][target.object_type]
    if np.any(speeds > float(dynamic_config["max_speed_mps"]) + 1e-6):
        return "target_speed_limit"
    if accelerations.size and np.any(
        np.linalg.norm(accelerations, axis=1)
        > float(dynamic_config["max_acceleration_mps2"]) + 1e-5
    ):
        return "target_acceleration_limit"
    context_footprints = {
        object_id: footprint_from_spec(spec)
        for object_id, spec in oracle_context.dynamic_object_specs.items()
    }
    for object_id in sorted(context_footprints):
        context_poses = np.vstack(
            (
                oracle_context.dynamic_object_history[object_id],
                oracle_context.dynamic_object_future[object_id],
            )
        )
        if synchronized_sweeps_intersect(
            target_footprint,
            all_poses,
            context_footprints[object_id],
            context_poses,
            grid=grid,
        ):
            return "target_context_collision"
    return None


def _target_history_visibility(
    *,
    template: ObstacleTargetTemplate,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
) -> np.ndarray:
    grid = build_grid_spec(dict(base_config))
    target_footprint = footprint_from_spec(template.target.footprint_spec)
    context_footprints = {
        object_id: footprint_from_spec(spec)
        for object_id, spec in oracle_context.dynamic_object_specs.items()
    }
    result = np.empty(grid.history_steps, dtype=np.bool_)
    static = np.asarray(template.static_occupancy != 0, dtype=np.bool_)
    for index in range(grid.history_steps):
        occupied = static.copy()
        for object_id in sorted(context_footprints):
            occupied |= rasterize_footprint(
                context_footprints[object_id],
                oracle_context.dynamic_object_history[object_id][index],
                grid,
            )
        target_mask = rasterize_footprint(
            target_footprint, template.target.history_poses[index], grid
        )
        result[index] = bool(
            raycast_candidate_visibility(
                occupied,
                target_mask,
                grid,
                sensor_pose=base_state.robot_history[index],
            ).any()
        )
    return result


def _target_future_visibility(
    *,
    template: ObstacleTargetTemplate,
    nominal_route: ObstaclePlannedRoute,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
) -> np.ndarray:
    grid = build_grid_spec(dict(base_config))
    target_footprint = footprint_from_spec(template.target.footprint_spec)
    context_footprints = {
        object_id: footprint_from_spec(spec)
        for object_id, spec in oracle_context.dynamic_object_specs.items()
    }
    static = np.asarray(template.static_occupancy != 0, dtype=np.bool_)
    result = np.empty(grid.future_steps, dtype=np.bool_)
    for index in range(grid.future_steps):
        occupied = static.copy()
        for object_id in sorted(context_footprints):
            occupied |= rasterize_footprint(
                context_footprints[object_id],
                oracle_context.dynamic_object_future[object_id][index],
                grid,
            )
        target_mask = rasterize_footprint(
            target_footprint, template.target.future_poses[index], grid
        )
        result[index] = bool(
            raycast_candidate_visibility(
                occupied,
                target_mask,
                grid,
                sensor_pose=nominal_route.poses_world[index],
            ).any()
        )
    return result


def _plan_digest(result: ObstaclePlanResult) -> str:
    digest = hashlib.sha256()
    digest.update(b"sop05r_planner_candidate_set_v1\0")
    for route in result.routes:
        digest.update(route.slot_id.encode("ascii") + b"\0")
        digest.update(route.trajectory.trajectory_id.encode("utf-8") + b"\0")
        for array in (
            route.trajectory.poses,
            route.trajectory.controls,
            route.poses_world,
            route.waypoints_world,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
            digest.update(_canonical_json_bytes(list(contiguous.shape)))
            digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _collision_path_distance(
    route: ObstaclePlannedRoute,
    *,
    collision_time_s: float,
    dt_s: float,
) -> tuple[float, float]:
    controls = np.abs(route.trajectory.controls[:, 0].astype(np.float64))
    total = float(np.sum(controls) * dt_s)
    interval = min(len(controls) - 1, int(np.floor(collision_time_s / dt_s)))
    completed_time = interval * dt_s
    distance = float(np.sum(controls[:interval]) * dt_s)
    distance += float(controls[interval] * max(0.0, collision_time_s - completed_time))
    return distance, total


def _normalized_task_axes(
    *,
    template: ObstacleTargetTemplate,
    base_state: BaseState,
) -> tuple[np.ndarray, np.ndarray]:
    task_direction = (
        template.local_goal_world_pose[:2].astype(np.float64)
        - base_state.robot_history[-1, :2].astype(np.float64)
    )
    norm = float(np.linalg.norm(task_direction))
    if norm <= 1e-9:
        raise ValueError("local goal must differ from the current robot position")
    task_direction /= norm
    task_normal = np.asarray(
        [-task_direction[1], task_direction[0]], dtype=np.float64
    )
    return task_direction, task_normal


def _layout_side_sign(
    *,
    template: ObstacleTargetTemplate,
    base_state: BaseState,
    task_normal: np.ndarray,
) -> float:
    layout = template.provenance.get("relative_layout")
    if layout == "target_side":
        return 1.0
    if layout == "opposite_side":
        return -1.0
    lateral_offset = float(
        np.dot(
            template.obstacle.pose[:2]
            - base_state.robot_history[-1, :2].astype(np.float64),
            task_normal,
        )
    )
    return 1.0 if lateral_offset >= 0.0 else -1.0


def build_route_contact_alignment_schedule(
    *,
    template: ObstacleTargetTemplate,
    planner_result: ObstaclePlanResult,
    base_state: BaseState,
    base_config: Mapping[str, Any],
    config: Sop05rConfig | Mapping[str, Any],
    seed: int,
) -> tuple[RouteContactAlignmentCandidate, ...]:
    """Build a stable, bounded route/index/direction target-join schedule."""

    if not isinstance(template, ObstacleTargetTemplate):
        raise TypeError("template must be an ObstacleTargetTemplate")
    if not isinstance(planner_result, ObstaclePlanResult):
        raise TypeError("planner_result must be an ObstaclePlanResult")
    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    normalized = _as_config(config)
    dt_s = float(base_config["bev"]["future_dt_s"])
    time_range = normalized.generation.conflict_time_range_s
    fraction_range = normalized.generation.conflict_path_fraction_range
    goal_range = normalized.generation.goal_beyond_conflict_range_m
    fraction_midpoint = 0.5 * sum(fraction_range)
    goal_midpoint = 0.5 * sum(goal_range)
    fraction_span = fraction_range[1] - fraction_range[0]
    goal_span = goal_range[1] - goal_range[0]
    task_direction, task_normal = _normalized_task_axes(
        template=template,
        base_state=base_state,
    )
    layout_sign = _layout_side_sign(
        template=template,
        base_state=base_state,
        task_normal=task_normal,
    )
    corner_angle = np.deg2rad(30.0)
    corner_offset = (
        np.cos(corner_angle) * task_direction
        + layout_sign * np.sin(corner_angle) * task_normal
    )
    mirrored_corner_offset = (
        np.cos(corner_angle) * task_direction
        - layout_sign * np.sin(corner_angle) * task_normal
    )
    direction_contacts = (
        (
            "toward_robot",
            -task_direction,
            "toward_obstacle_corner_30deg",
            -corner_offset,
        ),
        (
            "toward_robot",
            -task_direction,
            "toward_mirrored_corner_30deg",
            -mirrored_corner_offset,
        ),
    )
    moving_routes = sorted(
        (route for route in planner_result.routes if route.slot_id != "stop"),
        key=lambda route: (
            route.task_score,
            SOP05R_PLANNER_SLOT_IDS.index(route.slot_id),
        ),
    )
    result: list[RouteContactAlignmentCandidate] = []
    for route in moving_routes:
        index_rows = []
        for conflict_index in range(route.poses_world.shape[0]):
            anchor_time_s = (conflict_index + 1) * dt_s
            if not time_range[0] <= anchor_time_s <= time_range[1]:
                continue
            distance, total_distance = _collision_path_distance(
                route,
                collision_time_s=anchor_time_s,
                dt_s=dt_s,
            )
            if total_distance <= 0.0:
                continue
            estimated_fraction = distance / total_distance
            estimated_goal_forward = float(
                np.dot(
                    template.local_goal_world_pose[:2].astype(np.float64)
                    - route.poses_world[conflict_index, :2].astype(np.float64),
                    task_direction,
                )
            )
            quality = (
                abs(estimated_fraction - fraction_midpoint) / fraction_span
                + abs(estimated_goal_forward - goal_midpoint) / goal_span
            )
            tie_break = derive_seed(
                int(seed),
                "sop05r-route-contact-index-order-v1",
                template.template_id,
                route.trajectory.trajectory_id,
                conflict_index,
            )
            index_rows.append(
                (
                    quality,
                    tie_break,
                    conflict_index,
                    anchor_time_s,
                    estimated_fraction,
                    estimated_goal_forward,
                )
            )
        index_rows.sort(key=lambda row: (row[0], row[1], row[2]))
        maximum_index_count = max(
            1,
            normalized.generation.max_time_alignments_per_path
            // len(direction_contacts),
        )
        selected_index_rows = []
        selected_indices: set[int] = set()
        for row in sorted(index_rows, key=lambda item: item[2], reverse=True)[
            : max(0, maximum_index_count - 1)
        ]:
            selected_index_rows.append(row)
            selected_indices.add(row[2])
        for row in index_rows:
            if len(selected_index_rows) >= maximum_index_count:
                break
            if row[2] in selected_indices:
                continue
            selected_index_rows.append(row)
            selected_indices.add(row[2])
        route_candidates: list[RouteContactAlignmentCandidate] = []
        for (
            _,
            _,
            conflict_index,
            anchor_time_s,
            estimated_fraction,
            estimated_goal_forward,
        ) in selected_index_rows:
            for (
                direction_id,
                crossing_direction,
                contact_id,
                contact_normal,
            ) in direction_contacts:
                    identity_payload = {
                        "version": SOP05R_ROUTE_CONTACT_ALIGNMENT_VERSION,
                        "template_id": template.template_id,
                        "route_id": route.trajectory.trajectory_id,
                        "conflict_index": conflict_index,
                        "direction_id": direction_id,
                        "contact_id": contact_id,
                        "seed": int(seed),
                    }
                    identity = "alignment-" + hashlib.sha256(
                        _canonical_json_bytes(identity_payload)
                    ).hexdigest()[:24]
                    route_candidates.append(
                        RouteContactAlignmentCandidate(
                            identity=identity,
                            route=route,
                            conflict_index=conflict_index,
                            anchor_time_s=anchor_time_s,
                            estimated_path_fraction=estimated_fraction,
                            estimated_goal_forward_distance_m=estimated_goal_forward,
                            direction_id=direction_id,
                            crossing_direction=crossing_direction,
                            contact_id=contact_id,
                            contact_normal=contact_normal,
                        )
                    )
        result.extend(
            route_candidates[
                : normalized.generation.max_time_alignments_per_path
            ]
        )
    return tuple(result)


def _requested_history_regime(
    *,
    template: ObstacleTargetTemplate,
    config: Sop05rConfig,
    seed: int,
) -> str:
    draw = derive_seed(
        seed,
        "sop05r-template-history-stratum-v1",
        template.template_id,
    ) / float(2**32)
    if draw < config.history_policy.seen_then_occluded_weight:
        return SEEN_THEN_OCCLUDED
    return UNSEEN_IN_HISTORY_WINDOW


def _route_collision_evidence(
    *,
    route: ObstaclePlannedRoute,
    template: ObstacleTargetTemplate,
    base_state: BaseState,
    base_config: Mapping[str, Any],
) -> RouteCollisionEvidence | None:
    grid = build_grid_spec(dict(base_config))
    dt_s = float(base_config["bev"]["future_dt_s"])
    robot_poses = np.vstack((base_state.robot_history[-1], route.poses_world))
    target_poses = np.vstack((template.target.current_pose, template.target.future_poses))
    continuous = compute_continuous_collision_evidence(
        robot_footprint=_robot_footprint(base_config),
        robot_poses=robot_poses,
        target_footprint=footprint_from_spec(template.target.footprint_spec),
        target_poses=target_poses,
        dt_s=dt_s,
        spatial_resolution_m=0.25 * grid.resolution_m,
    )
    if not continuous.continuous_collision or continuous.first_collision_time_s is None:
        return None
    collision_time = continuous.first_collision_time_s
    conflict_index = max(
        0,
        min(grid.future_steps - 1, int(np.ceil(collision_time / dt_s)) - 1),
    )
    distance, total_distance = _collision_path_distance(
        route, collision_time_s=collision_time, dt_s=dt_s
    )
    if total_distance <= 0.0:
        return None
    conflict_point = continuous.robot_pose_at_first_collision[:2]
    task_delta = (
        template.local_goal_world_pose[:2].astype(np.float64)
        - base_state.robot_history[-1, :2].astype(np.float64)
    )
    task_norm = float(np.linalg.norm(task_delta))
    if task_norm <= 1e-9:
        return None
    task_direction = task_delta / task_norm
    goal_forward = float(
        np.dot(
            template.local_goal_world_pose[:2].astype(np.float64) - conflict_point,
            task_direction,
        )
    )
    return RouteCollisionEvidence(
        route_id=route.trajectory.trajectory_id,
        minimum_clearance_m=continuous.minimum_clearance_m,
        first_collision_time_s=collision_time,
        conflict_index=conflict_index,
        conflict_point=conflict_point,
        conflict_path_fraction=distance / total_distance,
        goal_forward_distance_m=goal_forward,
        continuous_evidence=continuous,
    )


def _reject(
    *,
    template: ObstacleTargetTemplate,
    planner_result: ObstaclePlanResult,
    reason: str,
    evidence: Mapping[str, object],
) -> Sop05rTemplateEvaluation:
    return Sop05rTemplateEvaluation(
        template_id=template.template_id,
        planner_result=planner_result,
        mother=None,
        rejection_reason=reason,
        evidence=dict(evidence),
    )


def _history_policy_digest(config: Sop05rConfig) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(config.history_policy.as_dict())
    ).hexdigest()


def _build_mother(
    *,
    template: ObstacleTargetTemplate,
    planner_result: ObstaclePlanResult,
    nominal: ObstaclePlannedRoute,
    alternatives: tuple[ObstaclePlannedRoute, ...],
    collision: RouteCollisionEvidence,
    history_visibility: np.ndarray,
    history_assessment: HistoryVisibilityAssessment,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    config: Sop05rConfig,
    seed: int,
    requested_history_regime: str,
    history_regime_fallback_used: bool,
    alignment_evidence: Mapping[str, object],
) -> Sop05rMotherCandidate:
    plan_digest = _plan_digest(planner_result)
    target = template.target
    identity = {
        "generator_algorithm_version": SOP05R_GENERATOR_VERSION,
        "template_id": template.template_id,
        "base_state_id": base_state.state_id,
        "split": base_state.split,
        "seed": seed,
        "config_digest": config.digest,
        "planner_candidate_set_digest": plan_digest,
        "nominal_trajectory_id": nominal.trajectory.trajectory_id,
        "alternative_trajectory_ids": [
            route.trajectory.trajectory_id for route in alternatives
        ],
        "target_dynamic_object_id": target.target_dynamic_object_id,
        "target_history_visibility": [bool(value) for value in history_visibility],
        "requested_history_visibility_regime": requested_history_regime,
        "conflict_time_s": collision.first_collision_time_s,
        "conflict_point": [float(value) for value in collision.conflict_point],
    }
    event_id = "event-" + hashlib.sha256(
        b"sop05r_event_identity_v1\0" + _canonical_json_bytes(identity)
    ).hexdigest()[:24]
    world_id = "world-" + hashlib.sha256(
        b"sop05r_world_identity_v1\0"
        + event_id.encode("ascii")
        + compute_motion_array_digest(
            target.history_poses, field_name="target_history_poses"
        ).encode("ascii")
        + compute_motion_array_digest(
            target.future_poses, field_name="target_future_poses"
        ).encode("ascii")
    ).hexdigest()[:24]
    target_record = create_event_target_motion_record(
        generated_event_id=event_id,
        world_id=world_id,
        base_state_id=base_state.state_id,
        trajectory_id=nominal.trajectory.trajectory_id,
        target_dynamic_object_id=target.target_dynamic_object_id,
        source_snippet_id=target.snippet_id,
        source_object_id=target.source_object_id,
        object_type=target.object_type,
        footprint_spec=target.footprint_spec,
        footprint_spec_digest=target.footprint_spec_digest,
        target_type_policy_digest=_SOP05R_TARGET_POLICY_DIGEST,
        history_poses=target.history_poses,
        current_pose=target.current_pose,
        future_poses=target.future_poses,
    )
    visibility_sequence = _target_future_visibility(
        template=template,
        nominal_route=nominal,
        oracle_context=oracle_context,
        base_config=base_config,
    )
    candidate_ids = [route.trajectory.trajectory_id for route in planner_result.routes]
    alternative_ids = [route.trajectory.trajectory_id for route in alternatives]
    metadata = {
        **build_event_target_motion_world_metadata(target_record),
        "schema_version": config.schema_version,
        "generator_algorithm_version": SOP05R_GENERATOR_VERSION,
        "event_kind": "environment",
        "scene_template_id": template.template_id,
        "base_state_id": base_state.state_id,
        "target_snippet_id": target.snippet_id,
        "source_object_id": target.source_object_id,
        "obstacle": template.obstacle.as_dict(),
        "local_goal_world_pose": [
            float(value) for value in template.local_goal_world_pose
        ],
        "planner_version": planner_result.version,
        "planner_candidate_set_digest": plan_digest,
        "candidate_trajectory_ids": candidate_ids,
        "nominal_trajectory_id": nominal.trajectory.trajectory_id,
        "alternative_trajectory_ids": alternative_ids,
        "path_costs": {
            route.trajectory.trajectory_id: route.task_score
            for route in planner_result.routes
        },
        "path_lengths": {
            route.trajectory.trajectory_id: route.path_length_m
            for route in planner_result.routes
        },
        "obstacle_clearances": {
            route.trajectory.trajectory_id: (
                route.represented_obstacle_clearance_m
            )
            for route in planner_result.routes
        },
        "conflict_point": [float(value) for value in collision.conflict_point],
        "conflict_path_fraction": collision.conflict_path_fraction,
        "conflict_time_s": collision.first_collision_time_s,
        "conflict_index": collision.conflict_index,
        "minimum_clearance_m": collision.minimum_clearance_m,
        "goal_forward_distance_m": collision.goal_forward_distance_m,
        "target_history_visibility_vector": [
            bool(value) for value in history_visibility
        ],
        "target_history_visibility_regime": history_assessment.regime,
        "requested_history_visibility_regime": requested_history_regime,
        "history_regime_fallback_used": history_regime_fallback_used,
        "target_history_last_visible_index": history_assessment.last_visible_index,
        "target_history_trailing_hidden_frames": (
            history_assessment.trailing_hidden_frames
        ),
        "target_history_visibility_policy_version": config.history_policy.version,
        "target_history_visibility_policy": config.history_policy.as_dict(),
        "target_history_visibility_policy_digest": _history_policy_digest(config),
        "first_visible_time_by_verification_action": {},
        "matched_wait_visible_time": None,
        "active_revealable_action_ids": [],
        "active_revealability_status": "not_evaluated",
        "rejection_summary": dict(planner_result.rejection_counts),
        "seed": seed,
        "config_digest": config.digest,
        "target_type_policy": _SOP05R_TARGET_POLICY,
        "target_type_policy_digest": _SOP05R_TARGET_POLICY_DIGEST,
        "target_provenance": target.provenance,
        "route_contact_alignment": target.provenance["route_contact_alignment"],
        "target_alignment_evidence": dict(alignment_evidence),
        "context_dynamic_object_ids": sorted(
            oracle_context.dynamic_object_future
        ),
        "visibility_sequence": [bool(value) for value in visibility_sequence],
    }
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
    world = OracleWorld(
        world_id=world_id,
        base_state_id=base_state.state_id,
        static_occupancy=np.asarray(template.static_occupancy, dtype=np.float32),
        dynamic_object_trajectories=dynamic_trajectories,
        dynamic_object_specs=dynamic_specs,
        occluders=(template.obstacle.as_dict(),),
        blind_spot_config={
            "kind": "environment",
            "occluder_ids": [template.obstacle.obstacle_id],
            "scene_template_id": template.template_id,
        },
        random_seed=seed,
        metadata=metadata,
    )
    grid = build_grid_spec(dict(base_config))
    validate_oracle_world(world, grid)
    validate_event_target_motion_world_join(target_record, world, grid)
    event = GeneratedEvent(
        generated_event_id=event_id,
        event_kind="environment",
        world=world,
        target=target,
        target_motion_record=target_record,
        visibility_sequence=visibility_sequence,
        target_visibility_history=history_visibility,
        conflict_time_s=collision.first_collision_time_s,
        conflict_index=collision.conflict_index,
    )
    trajectory_record = Sop05rTrajectoryRecord(
        event_id=event_id,
        base_state_id=base_state.state_id,
        template_id=template.template_id,
        planner_version=planner_result.version,
        config_digest=config.digest,
        shared_goal_world_pose=template.local_goal_world_pose,
        nominal_trajectory_id=nominal.trajectory.trajectory_id,
        alternative_trajectory_ids=tuple(alternative_ids),
        routes=planner_result.routes,
    )
    return Sop05rMotherCandidate(
        event=event,
        template=template,
        planner_result=planner_result,
        nominal_route=nominal,
        alternative_routes=alternatives,
        collision_evidence=collision,
        history_assessment=history_assessment,
        trajectory_record=trajectory_record,
    )


def _collision_gate_rejection(
    *,
    collision: RouteCollisionEvidence,
    route: ObstaclePlannedRoute,
    config: Sop05rConfig,
) -> str | None:
    conflict_time_range = config.generation.conflict_time_range_s
    if not (
        conflict_time_range[0]
        <= collision.first_collision_time_s
        <= conflict_time_range[1]
    ):
        return "conflict_time_out_of_range"
    conflict_fraction_range = config.generation.conflict_path_fraction_range
    if not (
        conflict_fraction_range[0]
        <= collision.conflict_path_fraction
        <= conflict_fraction_range[1]
    ):
        return "conflict_path_fraction_out_of_range"
    goal_range = config.generation.goal_beyond_conflict_range_m
    if not goal_range[0] <= collision.goal_forward_distance_m <= goal_range[1]:
        return "goal_not_beyond_conflict"
    clearance_range = config.planner.represented_obstacle_clearance_range_m
    if not (
        clearance_range[0]
        <= route.represented_obstacle_clearance_m
        <= clearance_range[1]
    ):
        return "nominal_clearance_out_of_range"
    return None


def _terminal_alignment_rejection(rejections: Counter[str]) -> str:
    priority = (
        "no_same_goal_alternative",
        "nominal_clearance_out_of_range",
        "goal_not_beyond_conflict",
        "conflict_path_fraction_out_of_range",
        "conflict_time_out_of_range",
        "no_time_aligned_collision",
        "history_ineligible",
        "target_current_robot_overlap",
        "target_context_collision",
        "target_source_static_collision",
        "target_obstacle_collision",
        "target_out_of_bounds",
        "target_acceleration_limit",
        "target_speed_limit",
        "target_motion_contract_invalid",
    )
    for reason in priority:
        if rejections[reason]:
            return reason
    return "no_time_aligned_collision"


def evaluate_obstacle_first_template(
    *,
    template: ObstacleTargetTemplate,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    config: Sop05rConfig | Mapping[str, Any],
    seed: int,
    planner: Callable[[ObstaclePlannerRequest], ObstaclePlanResult] = plan_obstacle_routes,
) -> Sop05rTemplateEvaluation:
    """Plan from static inputs first, then join target motion and apply gates."""

    if not isinstance(template, ObstacleTargetTemplate):
        raise TypeError("template must be an ObstacleTargetTemplate")
    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(oracle_context, OracleContext):
        raise TypeError("oracle_context must be an OracleContext")
    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    normalized = _as_config(config)
    grid = build_grid_spec(dict(base_config))
    validate_base_state(base_state, grid)
    validate_oracle_context(oracle_context, grid)
    if oracle_context.base_state_id != base_state.state_id:
        raise ValueError("oracle_context base_state_id mismatch")
    request = ObstaclePlannerRequest(
        start_pose=base_state.robot_history[-1],
        initial_control=base_state.robot_state,
        static_occupancy=template.static_occupancy,
        obstacle=template.obstacle,
        local_goal_world_pose=template.local_goal_world_pose,
        base_config=base_config,
        planner_config=normalized.planner,
    )
    planner_result = planner(request)
    if not isinstance(planner_result, ObstaclePlanResult):
        raise TypeError("planner must return an ObstaclePlanResult")
    observed_slots = tuple(route.slot_id for route in planner_result.routes)
    expected_slots = tuple(
        slot for slot in SOP05R_PLANNER_SLOT_IDS if slot in observed_slots
    )
    if observed_slots != expected_slots:
        raise ValueError("planner routes violate frozen slot order")
    evidence: dict[str, object] = {
        "planner_rejection_counts": dict(planner_result.rejection_counts),
        "moving_route_count": sum(
            route.slot_id != "stop" for route in planner_result.routes
        ),
    }
    if not planner_result.direct_path_intersects_inflated_obstacle:
        return _reject(
            template=template,
            planner_result=planner_result,
            reason="direct_path_misses_obstacle",
            evidence=evidence,
        )
    moving_routes = tuple(
        route for route in planner_result.routes if route.slot_id != "stop"
    )
    if not moving_routes:
        return _reject(
            template=template,
            planner_result=planner_result,
            reason="planner_no_route",
            evidence=evidence,
        )
    alignment_schedule = build_route_contact_alignment_schedule(
        template=template,
        planner_result=planner_result,
        base_state=base_state,
        base_config=base_config,
        config=normalized,
        seed=int(seed),
    )
    attempts_by_route = {route.slot_id: 0 for route in moving_routes}
    alignment_rejections: Counter[str] = Counter()
    observed_history_regimes: Counter[str] = Counter()
    requested_history_regime = _requested_history_regime(
        template=template,
        config=normalized,
        seed=int(seed),
    )
    selected_join: _AcceptedRouteContactJoin | None = None
    fallback_join: _AcceptedRouteContactJoin | None = None
    for candidate in alignment_schedule:
        attempts_by_route[candidate.route.slot_id] += 1
        try:
            aligned_target = align_target_to_route_contact(
                target=template.target,
                route=candidate.route,
                conflict_index=candidate.conflict_index,
                crossing_direction=candidate.crossing_direction,
                contact_normal=candidate.contact_normal,
                base_config=base_config,
                template_id=template.template_id,
                seed=int(seed),
            )
        except ValueError as exc:
            if str(exc) != "target source motion is stationary at the contact anchor":
                raise
            alignment_rejections["target_motion_contract_invalid"] += 1
            continue
        aligned_template = replace(
            template,
            target=aligned_target,
            provenance={
                **template.provenance,
                "post_plan_target_join": aligned_target.provenance[
                    "route_contact_alignment"
                ],
            },
        )
        physics_rejection = _target_physics_rejection(
            template=aligned_template,
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
        )
        if physics_rejection is not None:
            alignment_rejections[physics_rejection] += 1
            continue
        history_visibility = _target_history_visibility(
            template=aligned_template,
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
        )
        history_assessment = classify_sop05r_history(history_visibility)
        if history_assessment.regime not in HISTORY_VISIBILITY_REGIMES:
            alignment_rejections["history_ineligible"] += 1
            continue
        observed_history_regimes[history_assessment.regime] += 1
        collision = _route_collision_evidence(
            route=candidate.route,
            template=aligned_template,
            base_state=base_state,
            base_config=base_config,
        )
        if collision is None:
            alignment_rejections["no_time_aligned_collision"] += 1
            continue
        gate_rejection = _collision_gate_rejection(
            collision=collision,
            route=candidate.route,
            config=normalized,
        )
        if gate_rejection is not None:
            alignment_rejections[gate_rejection] += 1
            continue
        route_evidence = {
            route.trajectory.trajectory_id: (
                collision
                if route.trajectory.trajectory_id
                == candidate.route.trajectory.trajectory_id
                else _route_collision_evidence(
                    route=route,
                    template=aligned_template,
                    base_state=base_state,
                    base_config=base_config,
                )
            )
            for route in moving_routes
        }
        alternatives = tuple(
            route
            for route in moving_routes
            if route.trajectory.trajectory_id
            != candidate.route.trajectory.trajectory_id
            and route_evidence[route.trajectory.trajectory_id] is None
        )
        if not alternatives:
            alignment_rejections["no_same_goal_alternative"] += 1
            continue
        accepted_join = _AcceptedRouteContactJoin(
            template=aligned_template,
            nominal=candidate.route,
            alternatives=alternatives,
            collision=collision,
            route_evidence=route_evidence,
            history_visibility=history_visibility,
            history_assessment=history_assessment,
            alignment_identity=candidate.identity,
        )
        if history_assessment.regime == requested_history_regime:
            selected_join = accepted_join
            break
        if fallback_join is None:
            fallback_join = accepted_join
    if selected_join is None:
        selected_join = fallback_join
    evidence.update(
        {
            "target_alignment_candidate_count": len(alignment_schedule),
            "target_alignment_attempt_count": sum(attempts_by_route.values()),
            "target_alignment_attempts_by_route": attempts_by_route,
            "target_alignment_rejection_counts": dict(
                sorted(alignment_rejections.items())
            ),
            "requested_history_visibility_regime": requested_history_regime,
            "observed_history_visibility_regimes": dict(
                sorted(observed_history_regimes.items())
            ),
        }
    )
    if selected_join is None:
        return _reject(
            template=template,
            planner_result=planner_result,
            reason=_terminal_alignment_rejection(alignment_rejections),
            evidence=evidence,
        )
    evidence["selected_alignment_identity"] = selected_join.alignment_identity
    evidence["continuous_collision_route_ids"] = [
        route_id
        for route_id, row in selected_join.route_evidence.items()
        if row is not None
    ]
    fallback_used = (
        selected_join.history_assessment.regime != requested_history_regime
    )
    evidence["history_regime_fallback_used"] = fallback_used
    mother = _build_mother(
        template=selected_join.template,
        planner_result=planner_result,
        nominal=selected_join.nominal,
        alternatives=selected_join.alternatives,
        collision=selected_join.collision,
        history_visibility=selected_join.history_visibility,
        history_assessment=selected_join.history_assessment,
        base_state=base_state,
        oracle_context=oracle_context,
        base_config=base_config,
        config=normalized,
        seed=int(seed),
        requested_history_regime=requested_history_regime,
        history_regime_fallback_used=fallback_used,
        alignment_evidence=evidence,
    )
    return Sop05rTemplateEvaluation(
        template_id=template.template_id,
        planner_result=planner_result,
        mother=mother,
        rejection_reason=None,
        evidence=evidence,
    )
