"""Deterministic target-blind static-route planner for SOP05R v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.contracts import ARRAY_DTYPE, GridSpec, build_grid_spec
from src.geometry import (
    CircleOccluder,
    RectangleFootprint,
    RectangleOccluder,
    StaticOccluder,
    inflate_footprint,
    point_signed_distance,
    rasterize_footprint,
    wrap_angle,
)
from src.generation.sop05r_contracts import LightweightTebConfig

from .differential_drive import integrate_twist


_FAILURE_COST = 1.0e12


def _readonly(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _vector(
    value: object, *, name: str, length: int, dtype: np.dtype = ARRAY_DTYPE
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,) or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric array with shape ({length},)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return _readonly(array, dtype=dtype)


def _static_occupancy(value: object, grid: GridSpec) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (grid.height, grid.width) or array.dtype != ARRAY_DTYPE:
        raise ValueError("static_occupancy must be float32 with the base grid shape")
    if not np.isfinite(array).all() or not np.isin(array, (0.0, 1.0)).all():
        raise ValueError("static_occupancy must be finite binary float32")
    return _readonly(array, dtype=ARRAY_DTYPE)


@dataclass(frozen=True)
class StaticTebRequest:
    """All and only target-blind information available to mother-route planning."""

    start_pose: np.ndarray
    initial_control: np.ndarray
    local_goal_world_pose: np.ndarray
    static_occupancy: np.ndarray
    occluders: tuple[StaticOccluder, ...]
    base_config: Mapping[str, object]
    planner_config: LightweightTebConfig

    def __post_init__(self) -> None:
        if not isinstance(self.base_config, Mapping):
            raise TypeError("base_config must be a mapping")
        if not isinstance(self.planner_config, LightweightTebConfig):
            raise TypeError("planner_config must be a LightweightTebConfig")
        grid = build_grid_spec(dict(self.base_config))
        if self.planner_config.route_sample_dt_s <= 0.0:
            raise ValueError("planner route sample interval must be positive")
        object.__setattr__(self, "start_pose", _vector(self.start_pose, name="start_pose", length=3))
        object.__setattr__(
            self,
            "initial_control",
            _vector(self.initial_control, name="initial_control", length=2),
        )
        object.__setattr__(
            self,
            "local_goal_world_pose",
            _vector(
                self.local_goal_world_pose,
                name="local_goal_world_pose",
                length=3,
            ),
        )
        object.__setattr__(
            self,
            "static_occupancy",
            _static_occupancy(self.static_occupancy, grid),
        )
        if not isinstance(self.occluders, tuple) or not all(
            isinstance(item, (RectangleOccluder, CircleOccluder))
            for item in self.occluders
        ):
            raise TypeError("occluders must be a tuple of typed static occluders")


@dataclass(frozen=True)
class PlannedTebRoute:
    """One complete source-world task route and its variable-time optimizer band."""

    planner_version: str
    goal_world_pose: np.ndarray
    band_poses_world: np.ndarray
    band_interval_dt_s: np.ndarray
    sample_times_s: np.ndarray
    sampled_poses_world: np.ndarray
    sampled_controls: np.ndarray
    goal_arrival_time_s: float
    task_cost: float

    def __post_init__(self) -> None:
        for name, shape in (
            ("goal_world_pose", (3,)),
            ("band_poses_world", (20, 3)),
            ("band_interval_dt_s", (19,)),
            ("sample_times_s", (25,)),
            ("sampled_poses_world", (25, 3)),
            ("sampled_controls", (25, 2)),
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            object.__setattr__(self, name, _readonly(value, dtype=ARRAY_DTYPE))
        for name in ("goal_arrival_time_s", "task_cost"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class TebCandidateDiagnostic:
    initialization_id: str
    valid: bool
    cost: float
    rejection_reason: str | None
    optimization_iterations: int
    cost_terms: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class TebDiagnostics:
    candidates: tuple[TebCandidateDiagnostic, ...]


@dataclass(frozen=True)
class LightweightTebResult:
    planner_version: str
    route: PlannedTebRoute | None
    goal_world_pose: np.ndarray
    diagnostics: TebDiagnostics
    rejection_reason: str | None


def _robot_radius(base_config: Mapping[str, object]) -> float:
    robot = base_config["robot"]
    if not isinstance(robot, Mapping):
        raise ValueError("base_config.robot must be a mapping")
    length = float(robot["length_m"])
    width = float(robot["width_m"])
    inflation = float(robot["inflation_m"])
    return 0.5 * float(np.hypot(length, width)) + inflation


def _limits(base_config: Mapping[str, object]) -> tuple[float, float]:
    robot = base_config["robot"]
    if not isinstance(robot, Mapping):
        raise ValueError("base_config.robot must be a mapping")
    linear = float(robot["max_linear_speed_mps"])
    angular = float(robot["max_angular_speed_radps"])
    if not np.isfinite((linear, angular)).all() or linear <= 0.0 or angular <= 0.0:
        raise ValueError("robot speed limits must be finite and positive")
    return linear, angular


def _occluder_center(occluder: StaticOccluder) -> np.ndarray:
    if isinstance(occluder, CircleOccluder):
        return np.asarray(occluder.center_xy, dtype=np.float64)
    return np.asarray(occluder.pose[:2], dtype=np.float64)


def _direct_path_blocked(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    occluders: tuple[StaticOccluder, ...],
    robot_radius: float,
    clearance_m: float,
) -> bool:
    fractions = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    points = start_xy[None, :] + fractions[:, None] * (goal_xy - start_xy)[None, :]
    return any(
        bool(np.min(point_signed_distance(occluder, points) - robot_radius) < clearance_m)
        for occluder in occluders
    )


def _straight_escape_side(request: StaticTebRequest) -> float | None:
    """Choose the unique deterministic escape side from a non-centered occluder."""

    start_xy = request.start_pose[:2].astype(np.float64)
    goal_xy = request.local_goal_world_pose[:2].astype(np.float64)
    direction = goal_xy - start_xy
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-9 or not request.occluders:
        return None
    direction /= distance
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    lateral_offsets = [
        float(np.dot(_occluder_center(occluder) - start_xy, normal))
        for occluder in request.occluders
    ]
    closest = min(lateral_offsets, key=abs)
    if abs(closest) <= 1e-9:
        return None
    return -1.0 if closest > 0.0 else 1.0


def _occluder_projection_bounds(
    occluder: StaticOccluder,
    *,
    tangent: np.ndarray,
    normal: np.ndarray,
    origin: np.ndarray,
) -> tuple[float, float, float, float]:
    center = _occluder_center(occluder)
    if isinstance(occluder, CircleOccluder):
        along = float(np.dot(center - origin, tangent))
        lateral = float(np.dot(center - origin, normal))
        return (
            along - occluder.radius_m,
            along + occluder.radius_m,
            lateral - occluder.radius_m,
            lateral + occluder.radius_m,
        )
    yaw = float(occluder.pose[2])
    forward = np.asarray([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    left = np.asarray([-forward[1], forward[0]], dtype=np.float64)
    corners = np.asarray(
        [
            center
            + along_sign * 0.5 * occluder.length_m * forward
            + lateral_sign * 0.5 * occluder.width_m * left
            for along_sign in (-1.0, 1.0)
            for lateral_sign in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    relative = corners - origin[None, :]
    along_values = relative @ tangent
    lateral_values = relative @ normal
    return (
        float(np.min(along_values)),
        float(np.max(along_values)),
        float(np.min(lateral_values)),
        float(np.max(lateral_values)),
    )


def _bypass_waypoints(
    request: StaticTebRequest, *, side: float, robot_radius: float
) -> tuple[np.ndarray, ...]:
    start_xy = request.start_pose[:2].astype(np.float64)
    goal_xy = request.local_goal_world_pose[:2].astype(np.float64)
    direction = goal_xy - start_xy
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return (goal_xy,)
    tangent = direction / length
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    clearance = request.planner_config.represented_occluder_clearance_range_m[0]
    bounds: list[tuple[float, float, float, float]] = []
    for occluder in request.occluders:
        projected = _occluder_projection_bounds(
            occluder,
            tangent=tangent,
            normal=normal,
            origin=start_xy,
        )
        if projected[1] > 0.0 and projected[0] < length:
            bounds.append(projected)
    if not bounds:
        return (goal_xy,)
    along_min = min(item[0] for item in bounds)
    along_max = max(item[1] for item in bounds)
    if side > 0.0:
        target_lateral = max(item[3] for item in bounds)
    else:
        target_lateral = min(item[2] for item in bounds)
    target_lateral += side * (
        robot_radius
        + clearance
        + request.planner_config.bypass_tracking_allowance_m
    )
    before = (
        start_xy
        + (along_min - 0.20) * tangent
        + target_lateral * normal
    )
    after = (
        start_xy
        + (along_max + 0.20) * tangent
        + target_lateral * normal
    )
    return (before, after, goal_xy)


def _resample_polyline(points: tuple[np.ndarray, ...], count: int) -> np.ndarray:
    vertices = np.asarray(points, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    samples = np.linspace(0.0, cumulative[-1], count, dtype=np.float64)
    result = np.empty((count, 2), dtype=np.float64)
    for axis in range(2):
        result[:, axis] = np.interp(samples, cumulative, vertices[:, axis])
    return result


def _optimize_teb_band(
    request: StaticTebRequest,
    *,
    side: float | None,
    robot_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the fixed-count elastic-band/time updates for one initialization."""

    config = request.planner_config
    start = request.start_pose.astype(np.float64)
    goal = request.local_goal_world_pose.astype(np.float64)
    seed_points = (
        (start[:2], goal[:2])
        if side is None
        else (start[:2], *_bypass_waypoints(request, side=side, robot_radius=robot_radius))
    )
    positions = _resample_polyline(seed_points, config.band_node_count)
    dt_s = np.full(
        config.band_node_count - 1, config.initial_band_dt_s, dtype=np.float64
    )
    clearance = config.represented_occluder_clearance_range_m[0] + robot_radius
    bounds = config.band_dt_range_s
    for _ in range(config.max_iterations):
        updated = positions.copy()
        updated[1:-1] = 0.5 * positions[1:-1] + 0.25 * (
            positions[:-2] + positions[2:]
        )
        for occluder in request.occluders:
            distances = point_signed_distance(occluder, positions[1:-1])
            violations = np.maximum(0.0, clearance - distances)
            if not np.any(violations > 0.0):
                continue
            epsilon = 1e-4
            gradient = np.empty_like(updated[1:-1])
            for axis in range(2):
                offset = np.zeros_like(updated[1:-1])
                offset[:, axis] = epsilon
                gradient[:, axis] = (
                    point_signed_distance(occluder, positions[1:-1] + offset)
                    - point_signed_distance(occluder, positions[1:-1] - offset)
                ) / (2.0 * epsilon)
            norm = np.maximum(np.linalg.norm(gradient, axis=1, keepdims=True), 1e-9)
            updated[1:-1] += 0.15 * violations[:, None] * gradient / norm
        updated[0] = start[:2]
        updated[-1] = goal[:2]
        positions = updated
        lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        target_dt = np.clip(lengths / 0.75, bounds[0], bounds[1])
        dt_s = 0.85 * dt_s + 0.15 * target_dt
        dt_s = np.clip(dt_s, bounds[0], bounds[1])
        if float(np.sum(dt_s)) > config.maximum_route_time_s:
            dt_s *= config.maximum_route_time_s / float(np.sum(dt_s))
            dt_s = np.clip(dt_s, bounds[0], bounds[1])
    band = np.empty((config.band_node_count, 3), dtype=np.float64)
    band[:, :2] = positions
    directions = np.diff(positions, axis=0)
    headings = np.arctan2(directions[:, 1], directions[:, 0])
    band[:-1, 2] = headings
    band[-1, 2] = goal[2]
    band[0] = start
    band[-1] = goal
    return band, dt_s


def _bounded_control(
    desired: float,
    previous: float,
    *,
    acceleration_limit: float,
    dt_s: float,
    magnitude_limit: float,
) -> float:
    delta = acceleration_limit * dt_s
    return float(
        np.clip(
            desired,
            max(-magnitude_limit, previous - delta),
            min(magnitude_limit, previous + delta),
        )
    )


def _task_cost_terms(
    request: StaticTebRequest,
    *,
    poses: np.ndarray,
    controls: np.ndarray,
    arrival_time_s: float,
    clearance_m: float,
) -> tuple[tuple[str, float], ...]:
    """Evaluate every frozen TEB objective component on the accepted rollout."""

    weights = dict(request.planner_config.weights)
    max_speed, max_omega = _limits(request.base_config)
    dt_s = request.planner_config.route_sample_dt_s
    initial = request.initial_control.astype(np.float64)
    pose_with_start = np.vstack((request.start_pose.astype(np.float64), poses))
    displacements = np.diff(pose_with_start[:, :2], axis=0)
    headings = pose_with_start[:-1, 2]
    lateral_slip = (
        -np.sin(headings) * displacements[:, 0]
        + np.cos(headings) * displacements[:, 1]
    )
    prior_controls = np.vstack((initial, controls[:-1]))
    accelerations = (controls - prior_controls) / dt_s
    terms = {
        "length": weights["length"]
        * float(np.sum(np.linalg.norm(displacements, axis=1))),
        "time": weights["time"] * arrival_time_s,
        "smoothness": weights["smoothness"]
        * float(np.sum(np.diff(controls, axis=0) ** 2)),
        "obstacle": (
            weights["obstacle"]
            * max(
                0.0,
                request.planner_config.represented_occluder_clearance_range_m[0]
                - clearance_m,
            )
            ** 2
            if request.occluders
            else 0.0
        ),
        "nonholonomic": weights["nonholonomic"] * float(np.sum(lateral_slip**2)),
        "velocity": weights["velocity"]
        * float(
            np.sum((controls[:, 0] / max_speed) ** 2)
            + np.sum((controls[:, 1] / max_omega) ** 2)
        ),
        "acceleration": weights["acceleration"]
        * float(
            np.sum(
                (accelerations[:, 0] / request.planner_config.max_linear_acceleration_mps2)
                ** 2
            )
            + np.sum(
                (
                    accelerations[:, 1]
                    / request.planner_config.max_angular_acceleration_radps2
                )
                ** 2
            )
        ),
        "goal_heading": weights["goal_heading"]
        * float(wrap_angle(poses[-1, 2] - request.local_goal_world_pose[2]) ** 2),
        "initial_control": weights["initial_control"]
        * float(
            ((controls[0, 0] - initial[0]) / max_speed) ** 2
            + ((controls[0, 1] - initial[1]) / max_omega) ** 2
        ),
    }
    return tuple((name, float(terms[name])) for name in weights)


def _rollout_candidate(
    request: StaticTebRequest,
    *,
    initialization_id: str,
    side: float | None,
    grid: GridSpec,
) -> tuple[PlannedTebRoute | None, str | None, float]:
    config = request.planner_config
    dt_s = config.route_sample_dt_s
    sample_count = int(round(config.maximum_route_time_s / dt_s))
    max_speed, max_omega = _limits(request.base_config)
    robot_radius = _robot_radius(request.base_config)
    clearance_floor = config.represented_occluder_clearance_range_m[0]
    start = request.start_pose.astype(np.float64)
    goal = request.local_goal_world_pose.astype(np.float64)
    direct_blocked = _direct_path_blocked(
        start[:2], goal[:2], request.occluders, robot_radius, clearance_floor
    )
    optimized_band, optimized_dt_s = _optimize_teb_band(
        request, side=side, robot_radius=robot_radius
    )
    waypoints = (
        (goal[:2],)
        if side is None
        else tuple(optimized_band[index, :2] for index in range(3, 19, 3))
        + (optimized_band[-1, :2],)
    )
    pose = start.copy()
    previous_control = request.initial_control.astype(np.float64)
    poses = np.empty((sample_count, 3), dtype=np.float64)
    controls = np.empty((sample_count, 2), dtype=np.float64)
    waypoint_index = 0
    arrival_index: int | None = None

    for index in range(sample_count):
        target = waypoints[waypoint_index]
        delta_xy = target - pose[:2]
        distance = float(np.linalg.norm(delta_xy))
        segment_start = (
            start[:2]
            if waypoint_index == 0
            else waypoints[waypoint_index - 1]
        )
        segment = target - segment_start
        has_passed_waypoint = float(np.dot(pose[:2] - segment_start, segment)) >= float(
            np.dot(segment, segment)
        )
        route_direction = goal[:2] - start[:2]
        route_direction /= max(float(np.linalg.norm(route_direction)), 1e-9)
        route_normal = np.asarray(
            [-route_direction[1], route_direction[0]], dtype=np.float64
        )
        has_reached_bypass_side = bool(
            side is not None
            and waypoint_index == 0
            and side * float(np.dot(pose[:2] - segment_start, route_normal))
            >= 0.75 * side * float(np.dot(target - segment_start, route_normal))
        )
        if (
            (distance < 0.15 or has_passed_waypoint or has_reached_bypass_side)
            and waypoint_index < len(waypoints) - 1
        ):
            waypoint_index += 1
            target = waypoints[waypoint_index]
            delta_xy = target - pose[:2]
            distance = float(np.linalg.norm(delta_xy))
        terminal_distance = float(np.linalg.norm(goal[:2] - pose[:2]))
        target_yaw = (
            float(goal[2])
            if terminal_distance <= config.goal_position_tolerance_m
            else (
                float(np.arctan2(delta_xy[1], delta_xy[0]))
                if distance > 1e-9
                else float(goal[2])
            )
        )
        heading_error = float(wrap_angle(target_yaw - pose[2]))
        desired_omega = np.clip(heading_error / dt_s, -max_omega, max_omega)
        omega = _bounded_control(
            float(desired_omega),
            float(previous_control[1]),
            acceleration_limit=config.max_angular_acceleration_radps2,
            dt_s=dt_s,
            magnitude_limit=max_omega,
        )
        if abs(previous_control[0]) > 1e-6:
            omega = float(
                np.clip(
                    omega,
                    -config.max_curvature_per_m * abs(previous_control[0]),
                    config.max_curvature_per_m * abs(previous_control[0]),
                )
            )
        stopping_speed = float(
            np.sqrt(
                max(
                    0.0,
                    2.0
                    * config.max_linear_acceleration_mps2
                    * (
                        terminal_distance
                        - 0.5 * config.goal_position_tolerance_m
                    ),
                )
            )
        )
        desired_speed = min(max_speed, stopping_speed, distance / dt_s)
        desired_speed *= max(0.0, float(np.cos(heading_error)))
        speed = _bounded_control(
            desired_speed,
            float(previous_control[0]),
            acceleration_limit=config.max_linear_acceleration_mps2,
            dt_s=dt_s,
            magnitude_limit=max_speed,
        )
        if abs(speed) > 1e-6:
            omega = float(
                np.clip(
                    omega,
                    -config.max_curvature_per_m * abs(speed),
                    config.max_curvature_per_m * abs(speed),
                )
            )
        control = np.asarray([speed, omega], dtype=np.float64)
        next_pose = integrate_twist(
            pose, v=float(control[0]), omega=float(control[1]), dt_s=dt_s
        )
        poses[index] = next_pose
        controls[index] = control
        pose = next_pose
        previous_control = control
        if (
            waypoint_index == len(waypoints) - 1
            and np.linalg.norm(pose[:2] - goal[:2]) <= config.goal_position_tolerance_m
            and abs(float(wrap_angle(pose[2] - goal[2])))
            <= config.goal_yaw_tolerance_rad
            and abs(float(control[0])) <= 1e-5
            and abs(float(control[1])) <= 1e-5
        ):
            arrival_index = index

    if arrival_index is None:
        return None, "teb_goal_unreached", _FAILURE_COST
    if _has_static_collision(
        poses,
        static_occupancy=request.static_occupancy,
        base_config=request.base_config,
        grid=grid,
    ):
        return None, "teb_static_collision", _FAILURE_COST
    if request.occluders:
        clearance = min(
            float(np.min(point_signed_distance(occluder, poses[:, :2]) - robot_radius))
            for occluder in request.occluders
        )
        if clearance < clearance_floor:
            return None, "teb_static_collision", _FAILURE_COST
    else:
        clearance = 0.0

    anchors = np.vstack((start, poses))
    arrival_time_s = (arrival_index + 1) * dt_s
    cost_terms = _task_cost_terms(
        request,
        poses=poses,
        controls=controls,
        arrival_time_s=arrival_time_s,
        clearance_m=clearance,
    )
    cost = float(sum(value for _, value in cost_terms))
    route = PlannedTebRoute(
        planner_version=config.version,
        goal_world_pose=goal,
        band_poses_world=optimized_band,
        band_interval_dt_s=optimized_dt_s,
        sample_times_s=(
            np.arange(1, sample_count + 1, dtype=np.float32)
            * np.float32(dt_s)
        ),
        sampled_poses_world=poses,
        sampled_controls=controls,
        goal_arrival_time_s=arrival_time_s,
        task_cost=cost,
    )
    return route, None, cost


def _has_static_collision(
    poses: np.ndarray,
    *,
    static_occupancy: np.ndarray,
    base_config: Mapping[str, object],
    grid: GridSpec,
) -> bool:
    robot = base_config["robot"]
    if not isinstance(robot, Mapping):
        raise ValueError("base_config.robot must be a mapping")
    footprint = inflate_footprint(
        RectangleFootprint(float(robot["length_m"]), float(robot["width_m"])),
        float(robot["inflation_m"]),
    )
    anchors = np.vstack((np.zeros((1, 3), dtype=np.float64), poses))
    for start, end in zip(anchors[:-1], anchors[1:], strict=True):
        for fraction in np.linspace(0.0, 1.0, 6):
            pose = (1.0 - fraction) * start + fraction * end
            mask = rasterize_footprint(footprint, pose, grid)
            if np.any(mask & (static_occupancy != 0.0)):
                return True
    return False


def plan_lightweight_teb(request: StaticTebRequest) -> LightweightTebResult:
    """Return the stable lowest-cost valid full route for one static-only request."""

    if not isinstance(request, StaticTebRequest):
        raise TypeError("request must be a StaticTebRequest")
    grid = build_grid_spec(dict(request.base_config))
    choices: list[tuple[float, int, PlannedTebRoute]] = []
    diagnostics: list[TebCandidateDiagnostic] = []
    for index, initialization_id in enumerate(request.planner_config.initialization_ids):
        side = (
            _straight_escape_side(request)
            if initialization_id == "straight"
            else None
        )
        route, rejection_reason, cost = _rollout_candidate(
            request,
            initialization_id=initialization_id,
            side=side,
            grid=grid,
        )
        cost_terms = (
            _task_cost_terms(
                request,
                poses=route.sampled_poses_world,
                controls=route.sampled_controls,
                arrival_time_s=route.goal_arrival_time_s,
                clearance_m=(
                    min(
                        float(
                            np.min(
                                point_signed_distance(
                                    occluder, route.sampled_poses_world[:, :2]
                                )
                                - _robot_radius(request.base_config)
                            )
                        )
                        for occluder in request.occluders
                    )
                    if request.occluders
                    else 0.0
                ),
            )
            if route is not None
            else tuple(
                (name, _FAILURE_COST / len(request.planner_config.weights))
                for name, _ in request.planner_config.weights
            )
        )
        diagnostics.append(
            TebCandidateDiagnostic(
                initialization_id=initialization_id,
                valid=route is not None,
                cost=float(cost),
                rejection_reason=rejection_reason,
                optimization_iterations=request.planner_config.max_iterations,
                cost_terms=cost_terms,
            )
        )
        if route is not None:
            choices.append((float(cost), index, route))
    if choices:
        _, _, route = min(choices, key=lambda item: (item[0], item[1]))
        return LightweightTebResult(
            planner_version=request.planner_config.version,
            route=route,
            goal_world_pose=_readonly(request.local_goal_world_pose, dtype=ARRAY_DTYPE),
            diagnostics=TebDiagnostics(candidates=tuple(diagnostics)),
            rejection_reason=None,
        )
    reasons = {item.rejection_reason for item in diagnostics}
    rejection = "teb_static_collision" if "teb_static_collision" in reasons else "teb_goal_unreached"
    return LightweightTebResult(
        planner_version=request.planner_config.version,
        route=None,
        goal_world_pose=_readonly(request.local_goal_world_pose, dtype=ARRAY_DTYPE),
        diagnostics=TebDiagnostics(candidates=tuple(diagnostics)),
        rejection_reason=rejection,
    )


__all__ = (
    "LightweightTebResult",
    "PlannedTebRoute",
    "StaticTebRequest",
    "TebCandidateDiagnostic",
    "TebDiagnostics",
    "plan_lightweight_teb",
)
