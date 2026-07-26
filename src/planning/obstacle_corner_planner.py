"""Target-blind inflated-corner planner for SOP05R obstacle-first events."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from numbers import Real
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from src.contracts import ARRAY_DTYPE, LocalTrajectory, build_grid_spec
from src.generation.obstacle_first_templates import RectangleObstacle
from src.generation.occluder_sampler import swept_footprint_intersects_occupancy
from src.generation.sop05r_contracts import (
    SOP05R_PLANNER_SLOT_IDS,
    SOP05R_PLANNER_VERSION,
    PlannerConfig,
)
from src.geometry import (
    RectangleFootprint,
    footprint_aabb,
    footprint_vertices,
    inflate_footprint,
    trajectory_signed_clearances,
    transform_poses_local_to_global,
    wrap_angle,
)

from .differential_drive import integrate_twist
from .query_maps import build_local_trajectory
from .trajectory_filters import trajectory_rejection_reasons
from .trajectory_sampler import CandidateRollout


_MOVING_SLOT_IDS = SOP05R_PLANNER_SLOT_IDS[:-1]
_YIELD_SLOT_SPEED_SCALE = 0.55


def _readonly_array(value: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ObstaclePlannerRequest:
    start_pose: np.ndarray
    initial_control: np.ndarray
    static_occupancy: np.ndarray
    obstacle: RectangleObstacle
    local_goal_world_pose: np.ndarray
    base_config: Mapping[str, Any]
    planner_config: PlannerConfig

    def __post_init__(self) -> None:
        if not isinstance(self.base_config, Mapping):
            raise TypeError("base_config must be a mapping")
        if not isinstance(self.planner_config, PlannerConfig):
            raise TypeError("planner_config must be a PlannerConfig")
        if not isinstance(self.obstacle, RectangleObstacle):
            raise TypeError("obstacle must be a RectangleObstacle")
        start = np.asarray(self.start_pose)
        initial = np.asarray(self.initial_control)
        goal = np.asarray(self.local_goal_world_pose)
        if start.shape != (3,) or start.dtype != ARRAY_DTYPE or not np.isfinite(start).all():
            raise ValueError("start_pose must be finite float32 with shape (3,)")
        if (
            initial.shape != (2,)
            or initial.dtype != ARRAY_DTYPE
            or not np.isfinite(initial).all()
        ):
            raise ValueError("initial_control must be finite float32 with shape (2,)")
        if goal.shape != (3,) or goal.dtype != ARRAY_DTYPE or not np.isfinite(goal).all():
            raise ValueError(
                "local_goal_world_pose must be finite float32 with shape (3,)"
            )
        grid = build_grid_spec(dict(self.base_config))
        occupancy = np.asarray(self.static_occupancy)
        if (
            occupancy.shape != (grid.height, grid.width)
            or occupancy.dtype != ARRAY_DTYPE
            or not np.isfinite(occupancy).all()
            or not np.isin(occupancy, (0.0, 1.0)).all()
        ):
            raise ValueError("static_occupancy must be binary finite float32 grid data")
        if self.planner_config.rollout_steps != grid.future_steps:
            raise ValueError("planner rollout_steps must match the BEV future grid")
        if self.planner_config.dt_s != float(self.base_config["bev"]["future_dt_s"]):
            raise ValueError("planner dt_s must match the BEV future time step")
        object.__setattr__(
            self, "start_pose", _readonly_array(start, dtype=np.dtype(np.float32))
        )
        object.__setattr__(
            self,
            "initial_control",
            _readonly_array(initial, dtype=np.dtype(np.float32)),
        )
        object.__setattr__(
            self,
            "static_occupancy",
            _readonly_array(occupancy, dtype=np.dtype(np.float32)),
        )
        object.__setattr__(
            self,
            "local_goal_world_pose",
            _readonly_array(goal, dtype=np.dtype(np.float32)),
        )


@dataclass(frozen=True)
class ObstaclePlanDecision:
    slot_id: str
    accepted: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class ObstaclePlannedRoute:
    slot_id: str
    trajectory: LocalTrajectory
    poses_world: np.ndarray
    waypoints_world: np.ndarray
    path_length_m: float
    represented_obstacle_clearance_m: float
    task_score: float

    def __post_init__(self) -> None:
        if self.slot_id not in SOP05R_PLANNER_SLOT_IDS:
            raise ValueError("slot_id is not a frozen SOP05R planner slot")
        if not isinstance(self.trajectory, LocalTrajectory):
            raise TypeError("trajectory must be a LocalTrajectory")
        poses_world = np.asarray(self.poses_world)
        waypoints_world = np.asarray(self.waypoints_world)
        if (
            poses_world.shape != self.trajectory.poses.shape
            or poses_world.dtype != ARRAY_DTYPE
            or not np.isfinite(poses_world).all()
        ):
            raise ValueError("poses_world must match the finite float32 trajectory")
        if (
            waypoints_world.ndim != 2
            or waypoints_world.shape[1] != 3
            or waypoints_world.dtype != ARRAY_DTYPE
            or not np.isfinite(waypoints_world).all()
        ):
            raise ValueError("waypoints_world must be finite float32 [N, 3]")
        path_length = _finite_real(self.path_length_m, name="path_length_m")
        clearance = _finite_real(
            self.represented_obstacle_clearance_m,
            name="represented_obstacle_clearance_m",
        )
        score = _finite_real(self.task_score, name="task_score")
        if path_length < 0.0 or score < 0.0:
            raise ValueError("path length and task score must be nonnegative")
        object.__setattr__(
            self,
            "poses_world",
            _readonly_array(poses_world, dtype=np.dtype(np.float32)),
        )
        object.__setattr__(
            self,
            "waypoints_world",
            _readonly_array(waypoints_world, dtype=np.dtype(np.float32)),
        )
        object.__setattr__(self, "path_length_m", path_length)
        object.__setattr__(self, "represented_obstacle_clearance_m", clearance)
        object.__setattr__(self, "task_score", score)


@dataclass(frozen=True)
class ObstaclePlanResult:
    version: str
    shared_goal_world_pose: np.ndarray
    routes: tuple[ObstaclePlannedRoute, ...]
    decisions: tuple[ObstaclePlanDecision, ...]
    rejection_counts: Mapping[str, int]
    direct_path_intersects_inflated_obstacle: bool

    def __post_init__(self) -> None:
        if self.version != SOP05R_PLANNER_VERSION:
            raise ValueError("unsupported obstacle planner version")
        if tuple(decision.slot_id for decision in self.decisions) != SOP05R_PLANNER_SLOT_IDS:
            raise ValueError("planner decisions must preserve frozen slot order")
        if len({route.slot_id for route in self.routes}) != len(self.routes):
            raise ValueError("planner routes must have unique slots")
        accepted = {decision.slot_id for decision in self.decisions if decision.accepted}
        if accepted != {route.slot_id for route in self.routes}:
            raise ValueError("planner decisions and routes disagree")
        goal = np.asarray(self.shared_goal_world_pose)
        if goal.shape != (3,) or goal.dtype != ARRAY_DTYPE or not np.isfinite(goal).all():
            raise ValueError("shared_goal_world_pose must be finite float32 [3]")
        counts = dict(self.rejection_counts)
        if any(
            not isinstance(reason, str)
            or not reason
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for reason, count in counts.items()
        ):
            raise ValueError("rejection_counts must contain positive integer counts")
        object.__setattr__(
            self,
            "shared_goal_world_pose",
            _readonly_array(goal, dtype=np.dtype(np.float32)),
        )
        object.__setattr__(self, "rejection_counts", MappingProxyType(counts))

    @property
    def by_slot(self) -> dict[str, ObstaclePlannedRoute]:
        return {route.slot_id: route for route in self.routes}


def compute_obstacle_route_task_cost(
    *,
    path_length_m: object,
    terminal_heading_error_rad: object,
    normalized_smoothness: object,
    planner_config: PlannerConfig,
) -> float:
    """Compute the frozen path-length-dominant normalized planner score."""

    if not isinstance(planner_config, PlannerConfig):
        raise TypeError("planner_config must be a PlannerConfig")
    path_length = _finite_real(path_length_m, name="path_length_m")
    heading_error = _finite_real(
        terminal_heading_error_rad, name="terminal_heading_error_rad"
    )
    smoothness = _finite_real(
        normalized_smoothness, name="normalized_smoothness"
    )
    if path_length < 0.0 or heading_error < 0.0 or not 0.0 <= smoothness <= 1.0:
        raise ValueError("task cost inputs lie outside their normalized domains")
    normalized_length = path_length / planner_config.path_length_normalizer_m
    normalized_heading = min(1.0, heading_error / np.pi)
    return float(
        normalized_length
        + planner_config.heading_cost_weight * normalized_heading
        + planner_config.smoothness_cost_weight * smoothness
    )


def _inflated_robot(request: ObstaclePlannerRequest) -> RectangleFootprint:
    robot = request.base_config["robot"]
    return inflate_footprint(
        RectangleFootprint(
            float(robot["length_m"]), float(robot["width_m"])
        ),
        float(robot["inflation_m"]),
    )


def _robot_sweep_radius(footprint: RectangleFootprint) -> float:
    return 0.5 * float(np.hypot(footprint.length_m, footprint.width_m))


def _segment_intersects_rectangle(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    rectangle: RectangleFootprint,
    rectangle_pose: np.ndarray,
) -> bool:
    cosine = float(np.cos(rectangle_pose[2]))
    sine = float(np.sin(rectangle_pose[2]))
    rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    local_start = rotation @ (start_xy - rectangle_pose[:2])
    local_end = rotation @ (end_xy - rectangle_pose[:2])
    delta = local_end - local_start
    half_extents = (0.5 * rectangle.length_m, 0.5 * rectangle.width_m)
    lower_parameter = 0.0
    upper_parameter = 1.0
    for axis, half_extent in enumerate(half_extents):
        if abs(float(delta[axis])) <= 1e-12:
            if abs(float(local_start[axis])) > half_extent:
                return False
            continue
        first = (-half_extent - local_start[axis]) / delta[axis]
        second = (half_extent - local_start[axis]) / delta[axis]
        near, far = sorted((float(first), float(second)))
        lower_parameter = max(lower_parameter, near)
        upper_parameter = min(upper_parameter, far)
        if lower_parameter > upper_parameter:
            return False
    return True


def _route_waypoints(
    request: ObstaclePlannerRequest,
    *,
    slot_id: str,
    robot_radius_m: float,
) -> np.ndarray:
    start_xy = request.start_pose[:2].astype(np.float64)
    goal = request.local_goal_world_pose.astype(np.float64)
    delta = goal[:2] - start_xy
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-9:
        raise ValueError("local goal must differ from the planner start")
    direction = delta / distance
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    vertices = footprint_vertices(request.obstacle.footprint, request.obstacle.pose)
    relative = vertices - start_xy
    longitudinal = relative @ direction
    lateral = relative @ normal
    side, band = slot_id.split("_", maxsplit=1)
    clearance = request.planner_config.corner_clearance_m
    if band == "far":
        clearance = min(
            request.planner_config.represented_obstacle_clearance_range_m[1],
            clearance + 0.20,
        )
    longitudinal_margin = robot_radius_m + 0.05
    front = float(np.min(longitudinal) - longitudinal_margin)
    back = float(np.max(longitudinal) + longitudinal_margin)
    if side == "left":
        lateral_offset = float(np.max(lateral) + robot_radius_m + clearance)
    else:
        lateral_offset = float(np.min(lateral) - robot_radius_m - clearance)
    waypoint_xy = np.asarray(
        [
            start_xy + direction * front + normal * lateral_offset,
            start_xy + direction * back + normal * lateral_offset,
            goal[:2],
        ],
        dtype=np.float64,
    )
    yaws = np.empty(3, dtype=np.float64)
    yaws[0] = np.arctan2(
        waypoint_xy[1, 1] - waypoint_xy[0, 1],
        waypoint_xy[1, 0] - waypoint_xy[0, 0],
    )
    yaws[1] = np.arctan2(
        waypoint_xy[2, 1] - waypoint_xy[1, 1],
        waypoint_xy[2, 0] - waypoint_xy[1, 0],
    )
    yaws[2] = goal[2]
    return np.column_stack((waypoint_xy, wrap_angle(yaws))).astype(np.float32)


def _polyline_samples(
    start_pose: np.ndarray,
    waypoints: np.ndarray,
    *,
    maximum_step_m: float,
) -> np.ndarray:
    points = np.vstack((start_pose[:2], waypoints[:, :2])).astype(np.float64)
    samples: list[np.ndarray] = []
    for segment_index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        delta = end - start
        distance = float(np.linalg.norm(delta))
        count = max(1, int(np.ceil(distance / maximum_step_m)))
        yaw = float(np.arctan2(delta[1], delta[0])) if distance > 0.0 else 0.0
        first_index = 0 if segment_index == 0 else 1
        for index in range(first_index, count + 1):
            fraction = index / count
            point = (1.0 - fraction) * start + fraction * end
            samples.append(np.asarray([point[0], point[1], yaw], dtype=np.float64))
    if not samples:
        return start_pose.reshape(1, 3).astype(np.float64)
    samples[-1][2] = float(waypoints[-1, 2])
    return np.asarray(samples, dtype=np.float64)


def _polyline_lengths(start_xy: np.ndarray, waypoints: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.vstack((start_xy, waypoints[:, :2])).astype(np.float64)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return cumulative, float(cumulative[-1])


def _nearest_progress_and_lookahead(
    pose_xy: np.ndarray,
    points: np.ndarray,
    cumulative: np.ndarray,
    lookahead_m: float,
) -> np.ndarray:
    best_distance = np.inf
    best_progress = 0.0
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        segment = end - start
        squared_length = float(np.dot(segment, segment))
        fraction = (
            0.0
            if squared_length <= 1e-12
            else float(np.clip(np.dot(pose_xy - start, segment) / squared_length, 0.0, 1.0))
        )
        projection = start + fraction * segment
        distance = float(np.linalg.norm(pose_xy - projection))
        if distance < best_distance:
            best_distance = distance
            best_progress = float(
                cumulative[index]
                + fraction * (cumulative[index + 1] - cumulative[index])
            )
    target_progress = min(float(cumulative[-1]), best_progress + lookahead_m)
    segment_index = int(
        min(
            len(points) - 2,
            np.searchsorted(cumulative, target_progress, side="right") - 1,
        )
    )
    segment_length = cumulative[segment_index + 1] - cumulative[segment_index]
    fraction = (
        0.0
        if segment_length <= 1e-12
        else (target_progress - cumulative[segment_index]) / segment_length
    )
    return (
        (1.0 - fraction) * points[segment_index]
        + fraction * points[segment_index + 1]
    )


def _world_to_start_local(points_world: np.ndarray, start_pose: np.ndarray) -> np.ndarray:
    delta = points_world - start_pose[:2]
    cosine = float(np.cos(start_pose[2]))
    sine = float(np.sin(start_pose[2]))
    rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return delta @ rotation.T


def _pure_pursuit_rollout(
    request: ObstaclePlannerRequest,
    *,
    slot_id: str,
    waypoints_world: np.ndarray,
) -> CandidateRollout:
    waypoint_xy_local = _world_to_start_local(
        waypoints_world[:, :2].astype(np.float64),
        request.start_pose.astype(np.float64),
    )
    points = np.vstack((np.zeros(2, dtype=np.float64), waypoint_xy_local))
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    pose = np.zeros(3, dtype=np.float64)
    previous_control = request.initial_control.astype(np.float64)
    poses = []
    controls = []
    planner = request.planner_config
    robot = request.base_config["robot"]
    dt_s = planner.dt_s
    maximum_linear_speed = float(robot["max_linear_speed_mps"])
    maximum_angular_speed = float(robot["max_angular_speed_radps"])
    speed_scale = _YIELD_SLOT_SPEED_SCALE if slot_id.endswith("_near") else 1.0
    for _ in range(planner.rollout_steps):
        target_xy = _nearest_progress_and_lookahead(
            pose[:2], points, cumulative, planner.lookahead_distance_m
        )
        target_delta = target_xy - pose[:2]
        target_heading = float(np.arctan2(target_delta[1], target_delta[0]))
        heading_error = float(wrap_angle(target_heading - pose[2]))
        turn_factor = max(0.35, float(np.cos(heading_error)) ** 2)
        desired_linear = maximum_linear_speed * speed_scale * turn_factor
        maximum_linear_delta = (
            planner.max_linear_acceleration_mps2 * dt_s * (1.0 - 1e-6)
        )
        linear = float(
            previous_control[0]
            + np.clip(
                desired_linear - previous_control[0],
                -maximum_linear_delta,
                maximum_linear_delta,
            )
        )
        linear = float(np.clip(linear, 0.0, maximum_linear_speed))
        curvature = float(
            np.clip(
                2.0 * np.sin(heading_error) / planner.lookahead_distance_m,
                -planner.max_curvature_per_m,
                planner.max_curvature_per_m,
            )
        )
        desired_angular = float(
            np.clip(curvature * linear, -maximum_angular_speed, maximum_angular_speed)
        )
        maximum_angular_delta = (
            planner.max_angular_acceleration_radps2 * dt_s * (1.0 - 1e-6)
        )
        angular = float(
            previous_control[1]
            + np.clip(
                desired_angular - previous_control[1],
                -maximum_angular_delta,
                maximum_angular_delta,
            )
        )
        angular = float(
            np.clip(
                angular,
                -min(maximum_angular_speed, planner.max_curvature_per_m * linear),
                min(maximum_angular_speed, planner.max_curvature_per_m * linear),
            )
        )
        pose = integrate_twist(pose, v=linear, omega=angular, dt_s=dt_s)
        controls.append(np.asarray([linear, angular], dtype=np.float64))
        poses.append(pose.copy())
        previous_control = np.asarray([linear, angular], dtype=np.float64)
    return CandidateRollout(
        trajectory_id=f"sop05r::{request.obstacle.obstacle_id}::{slot_id}",
        poses=np.asarray(poses, dtype=ARRAY_DTYPE),
        controls=np.asarray(controls, dtype=ARRAY_DTYPE),
        is_stop=False,
        is_reverse=False,
    )


def _within_bev(
    footprint: RectangleFootprint,
    poses: np.ndarray,
    request: ObstaclePlannerRequest,
) -> bool:
    grid = build_grid_spec(dict(request.base_config))
    x_min = -0.5 * grid.width * grid.resolution_m
    x_max = 0.5 * grid.width * grid.resolution_m
    y_min = -0.5 * grid.height * grid.resolution_m
    y_max = 0.5 * grid.height * grid.resolution_m
    return all(
        bounds[0] >= x_min
        and bounds[1] <= x_max
        and bounds[2] >= y_min
        and bounds[3] <= y_max
        for bounds in (footprint_aabb(footprint, pose) for pose in poses)
    )


def _normalized_smoothness(
    controls: np.ndarray,
    initial_control: np.ndarray,
    planner: PlannerConfig,
) -> float:
    combined = np.vstack((initial_control, controls)).astype(np.float64)
    differences = np.abs(np.diff(combined, axis=0))
    linear_scale = planner.max_linear_acceleration_mps2 * planner.dt_s
    angular_scale = planner.max_angular_acceleration_radps2 * planner.dt_s
    normalized = np.column_stack(
        (
            differences[:, 0] / linear_scale,
            differences[:, 1] / angular_scale,
        )
    )
    return float(np.mean(np.clip(normalized, 0.0, 1.0)))


def _build_stop_route(request: ObstaclePlannerRequest) -> ObstaclePlannedRoute:
    steps = request.planner_config.rollout_steps
    dt_s = request.planner_config.dt_s
    pose = np.zeros(3, dtype=np.float64)
    previous = request.initial_control.astype(np.float64)
    poses = []
    controls = []
    for _ in range(steps):
        linear_delta = (
            request.planner_config.max_linear_acceleration_mps2
            * dt_s
            * (1.0 - 1e-6)
        )
        angular_delta = (
            request.planner_config.max_angular_acceleration_radps2
            * dt_s
            * (1.0 - 1e-6)
        )
        linear = float(previous[0] - np.clip(previous[0], -linear_delta, linear_delta))
        angular = float(
            previous[1] - np.clip(previous[1], -angular_delta, angular_delta)
        )
        pose = integrate_twist(pose, v=linear, omega=angular, dt_s=dt_s)
        controls.append(np.asarray([linear, angular], dtype=np.float64))
        poses.append(pose.copy())
        previous = np.asarray([linear, angular], dtype=np.float64)
    candidate = CandidateRollout(
        trajectory_id=f"sop05r::{request.obstacle.obstacle_id}::stop",
        poses=np.asarray(poses, dtype=ARRAY_DTYPE),
        controls=np.asarray(controls, dtype=ARRAY_DTYPE),
        is_stop=True,
        is_reverse=False,
    )
    trajectory = build_local_trajectory(
        candidate,
        dict(request.base_config),
        braking_deceleration_mps2=request.planner_config.max_linear_acceleration_mps2,
        task_cost=0.0,
    )
    trajectory = replace(
        trajectory,
        metadata={
            **trajectory.metadata,
            "planner_version": SOP05R_PLANNER_VERSION,
            "planner_slot_id": "stop",
            "is_braking_stop": True,
            "shared_goal_world_pose": [
                float(value) for value in request.local_goal_world_pose
            ],
            "geometric_route_reaches_shared_goal": False,
        },
    )
    world = transform_poses_local_to_global(
        candidate.poses, request.start_pose
    ).astype(ARRAY_DTYPE)
    clearance = float(
        trajectory_signed_clearances(
            _inflated_robot(request),
            world,
            request.obstacle.footprint,
            np.tile(request.obstacle.pose, (steps, 1)),
        ).min()
    )
    return ObstaclePlannedRoute(
        slot_id="stop",
        trajectory=trajectory,
        poses_world=world,
        waypoints_world=request.local_goal_world_pose.reshape(1, 3),
        path_length_m=float(np.sum(np.abs(candidate.controls[:, 0])) * dt_s),
        represented_obstacle_clearance_m=clearance,
        task_score=0.0,
    )


def _build_moving_route(
    request: ObstaclePlannerRequest,
    *,
    slot_id: str,
    inflated_robot: RectangleFootprint,
) -> tuple[ObstaclePlannedRoute | None, str | None]:
    robot_radius = _robot_sweep_radius(inflated_robot)
    waypoints = _route_waypoints(
        request, slot_id=slot_id, robot_radius_m=robot_radius
    )
    geometric_samples = _polyline_samples(
        request.start_pose,
        waypoints,
        maximum_step_m=0.5 * build_grid_spec(dict(request.base_config)).resolution_m,
    )
    if not _within_bev(inflated_robot, geometric_samples, request):
        return None, "waypoint_out_of_bounds"
    if swept_footprint_intersects_occupancy(
        inflated_robot,
        geometric_samples,
        request.static_occupancy,
        grid=build_grid_spec(dict(request.base_config)),
    ):
        return None, "geometric_route_static_collision"
    candidate = _pure_pursuit_rollout(
        request, slot_id=slot_id, waypoints_world=waypoints
    )
    dynamics_reasons = trajectory_rejection_reasons(
        candidate,
        dict(request.base_config),
        initial_control=request.initial_control,
        max_linear_acceleration_mps2=(
            request.planner_config.max_linear_acceleration_mps2
        ),
        max_angular_acceleration_radps2=(
            request.planner_config.max_angular_acceleration_radps2
        ),
    )
    if dynamics_reasons:
        return None, "rollout_dynamics_limit:" + dynamics_reasons[0]
    poses_world = transform_poses_local_to_global(
        candidate.poses, request.start_pose
    ).astype(ARRAY_DTYPE)
    if not _within_bev(inflated_robot, poses_world, request):
        return None, "rollout_out_of_bounds"
    if swept_footprint_intersects_occupancy(
        inflated_robot,
        poses_world,
        request.static_occupancy,
        grid=build_grid_spec(dict(request.base_config)),
    ):
        return None, "rollout_static_collision"
    pose_sequence = np.vstack((request.start_pose, poses_world))
    clearances = trajectory_signed_clearances(
        inflated_robot,
        pose_sequence,
        request.obstacle.footprint,
        np.tile(request.obstacle.pose, (pose_sequence.shape[0], 1)),
    )
    minimum_clearance = float(np.min(clearances))
    lower, upper = request.planner_config.represented_obstacle_clearance_range_m
    if minimum_clearance < lower or minimum_clearance > upper:
        return None, "represented_obstacle_clearance_out_of_range"
    _, geometric_length = _polyline_lengths(request.start_pose[:2], waypoints)
    terminal_heading_error = abs(
        float(wrap_angle(waypoints[-1, 2] - candidate.poses[-1, 2]))
    )
    smoothness = _normalized_smoothness(
        candidate.controls,
        request.initial_control,
        request.planner_config,
    )
    task_score = compute_obstacle_route_task_cost(
        path_length_m=geometric_length,
        terminal_heading_error_rad=terminal_heading_error,
        normalized_smoothness=smoothness,
        planner_config=request.planner_config,
    )
    trajectory = build_local_trajectory(
        candidate,
        dict(request.base_config),
        braking_deceleration_mps2=request.planner_config.max_linear_acceleration_mps2,
        task_cost=task_score,
    )
    trajectory = replace(
        trajectory,
        metadata={
            **trajectory.metadata,
            "planner_version": SOP05R_PLANNER_VERSION,
            "planner_slot_id": slot_id,
            "shared_goal_world_pose": [
                float(value) for value in request.local_goal_world_pose
            ],
            "geometric_route_reaches_shared_goal": True,
            "geometric_path_length_m": geometric_length,
            "represented_obstacle_clearance_m": minimum_clearance,
            "start_pose_world": [float(value) for value in request.start_pose],
            "planner_speed_scale": (
                _YIELD_SLOT_SPEED_SCALE if slot_id.endswith("_near") else 1.0
            ),
        },
    )
    return (
        ObstaclePlannedRoute(
            slot_id=slot_id,
            trajectory=trajectory,
            poses_world=poses_world,
            waypoints_world=waypoints,
            path_length_m=geometric_length,
            represented_obstacle_clearance_m=minimum_clearance,
            task_score=task_score,
        ),
        None,
    )


def plan_obstacle_routes(request: ObstaclePlannerRequest) -> ObstaclePlanResult:
    """Plan fixed-slot static-map routes without any target or oracle input."""

    if not isinstance(request, ObstaclePlannerRequest):
        raise TypeError("request must be an ObstaclePlannerRequest")
    inflated_robot = _inflated_robot(request)
    inflated_obstacle = inflate_footprint(
        request.obstacle.footprint, _robot_sweep_radius(inflated_robot)
    )
    direct_intersection = _segment_intersects_rectangle(
        request.start_pose[:2].astype(np.float64),
        request.local_goal_world_pose[:2].astype(np.float64),
        inflated_obstacle,
        request.obstacle.pose,
    )
    routes: list[ObstaclePlannedRoute] = []
    decisions: list[ObstaclePlanDecision] = []
    rejection_counts: dict[str, int] = {}
    for slot_id in _MOVING_SLOT_IDS:
        if not direct_intersection:
            route = None
            reason = "direct_path_clear"
        else:
            route, reason = _build_moving_route(
                request, slot_id=slot_id, inflated_robot=inflated_robot
            )
        if route is None:
            stable_reason = str(reason)
            decisions.append(
                ObstaclePlanDecision(
                    slot_id=slot_id,
                    accepted=False,
                    rejection_reason=stable_reason,
                )
            )
            rejection_counts[stable_reason] = rejection_counts.get(stable_reason, 0) + 1
        else:
            routes.append(route)
            decisions.append(
                ObstaclePlanDecision(
                    slot_id=slot_id,
                    accepted=True,
                    rejection_reason=None,
                )
            )
    stop = _build_stop_route(request)
    routes.append(stop)
    decisions.append(
        ObstaclePlanDecision(slot_id="stop", accepted=True, rejection_reason=None)
    )
    routes.sort(key=lambda route: SOP05R_PLANNER_SLOT_IDS.index(route.slot_id))
    return ObstaclePlanResult(
        version=SOP05R_PLANNER_VERSION,
        shared_goal_world_pose=request.local_goal_world_pose,
        routes=tuple(routes),
        decisions=tuple(decisions),
        rejection_counts=rejection_counts,
        direct_path_intersects_inflated_obstacle=direct_intersection,
    )


def _planner_geometry_cache_key(request: ObstaclePlannerRequest) -> str:
    digest = hashlib.sha256()
    digest.update(b"sop05r_obstacle_planner_geometry_cache_v1\0")
    payload = {
        "obstacle_type": request.obstacle.obstacle_type,
        "obstacle_length_m": request.obstacle.length_m,
        "obstacle_width_m": request.obstacle.width_m,
        "base_config": request.base_config,
        "planner_config": request.planner_config.as_dict(),
    }
    digest.update(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )
    for array in (
        request.start_pose,
        request.initial_control,
        request.static_occupancy,
        request.obstacle.pose,
        request.local_goal_world_pose,
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
        digest.update(str(tuple(contiguous.shape)).encode("ascii") + b"\0")
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _rebind_plan_obstacle_id(
    result: ObstaclePlanResult,
    *,
    obstacle_id: str,
) -> ObstaclePlanResult:
    routes = tuple(
        replace(
            route,
            trajectory=replace(
                route.trajectory,
                trajectory_id=f"sop05r::{obstacle_id}::{route.slot_id}",
            ),
        )
        for route in result.routes
    )
    return replace(result, routes=routes)


class GeometryCachingObstaclePlanner:
    """Reuse target-blind geometry while preserving template-local identities."""

    def __init__(
        self,
        planner: Callable[[ObstaclePlannerRequest], ObstaclePlanResult] | None = None,
    ) -> None:
        self._planner = plan_obstacle_routes if planner is None else planner
        self._cache: dict[str, ObstaclePlanResult] = {}
        self.hit_count = 0
        self.miss_count = 0

    def __call__(self, request: ObstaclePlannerRequest) -> ObstaclePlanResult:
        if not isinstance(request, ObstaclePlannerRequest):
            raise TypeError("request must be an ObstaclePlannerRequest")
        key = _planner_geometry_cache_key(request)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._planner(request)
            if not isinstance(cached, ObstaclePlanResult):
                raise TypeError("planner must return an ObstaclePlanResult")
            self._cache[key] = cached
            self.miss_count += 1
        else:
            self.hit_count += 1
        return _rebind_plan_obstacle_id(
            cached,
            obstacle_id=request.obstacle.obstacle_id,
        )
