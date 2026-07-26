"""Deterministic target-blind full-route planner for SOP05R v3."""

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
_V3_BAND_NODE_COUNT = 21
_V3_ROUTE_SAMPLE_COUNT = 40
_DENSE_COLLISION_FRACTIONS = np.linspace(0.0, 1.0, 6, dtype=np.float64)
_TERMINAL_TOLERANCE_SUFFIX_COUNT = 2
_SPARSE_OBJECTIVE_GRADIENT_REFRESH = 8


def _readonly(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    return np.frombuffer(result.tobytes(order="C"), dtype=result.dtype).reshape(
        result.shape
    )


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
class ObservedDynamicObstacle:
    """Deployment-observed state propagated by a frozen constant-velocity policy."""

    object_id: str
    observed_pose: np.ndarray
    observed_velocity_xy: np.ndarray
    footprint_radius_m: float
    observation_age_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id:
            raise ValueError("object_id must be nonempty text")
        object.__setattr__(
            self,
            "observed_pose",
            _vector(self.observed_pose, name="observed_pose", length=3),
        )
        object.__setattr__(
            self,
            "observed_velocity_xy",
            _vector(
                self.observed_velocity_xy,
                name="observed_velocity_xy",
                length=2,
            ),
        )
        radius = float(self.footprint_radius_m)
        age = float(self.observation_age_s)
        if not np.isfinite((radius, age)).all() or radius <= 0.0 or age < 0.0:
            raise ValueError(
                "footprint_radius_m must be positive and observation_age_s nonnegative"
            )
        object.__setattr__(self, "footprint_radius_m", radius)
        object.__setattr__(self, "observation_age_s", age)


@dataclass(frozen=True)
class ObservedTebRequest:
    """Post-action request containing static scene and observed dynamics only."""

    start_pose: np.ndarray
    initial_control: np.ndarray
    local_goal_world_pose: np.ndarray
    static_occupancy: np.ndarray
    occluders: tuple[StaticOccluder, ...]
    observed_dynamic_obstacles: tuple[ObservedDynamicObstacle, ...]
    base_config: Mapping[str, object]
    planner_config: LightweightTebConfig

    def __post_init__(self) -> None:
        static = StaticTebRequest(
            start_pose=self.start_pose,
            initial_control=self.initial_control,
            local_goal_world_pose=self.local_goal_world_pose,
            static_occupancy=self.static_occupancy,
            occluders=self.occluders,
            base_config=self.base_config,
            planner_config=self.planner_config,
        )
        object.__setattr__(self, "start_pose", static.start_pose)
        object.__setattr__(self, "initial_control", static.initial_control)
        object.__setattr__(
            self,
            "local_goal_world_pose",
            static.local_goal_world_pose,
        )
        object.__setattr__(self, "static_occupancy", static.static_occupancy)
        if not isinstance(self.observed_dynamic_obstacles, tuple) or not all(
            isinstance(item, ObservedDynamicObstacle)
            for item in self.observed_dynamic_obstacles
        ):
            raise TypeError(
                "observed_dynamic_obstacles must contain ObservedDynamicObstacle values"
            )


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
            ("band_poses_world", (_V3_BAND_NODE_COUNT, 3)),
            ("band_interval_dt_s", (_V3_BAND_NODE_COUNT - 1,)),
            ("sample_times_s", (_V3_ROUTE_SAMPLE_COUNT,)),
            ("sampled_poses_world", (_V3_ROUTE_SAMPLE_COUNT, 3)),
            ("sampled_controls", (_V3_ROUTE_SAMPLE_COUNT, 2)),
        ):
            value = np.asarray(getattr(self, name))
            if (
                value.shape != shape
                or value.dtype.kind not in "iuf"
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"{name} must be finite with shape {shape}")
            object.__setattr__(self, name, _readonly(value, dtype=ARRAY_DTYPE))
        expected_sample_times = (
            np.arange(1, _V3_ROUTE_SAMPLE_COUNT + 1, dtype=ARRAY_DTYPE)
            * np.float32(0.2)
        )
        if not np.array_equal(self.sample_times_s, expected_sample_times):
            raise ValueError("sample_times_s must equal [0.2, ..., 8.0] seconds")
        if not np.array_equal(self.band_poses_world[-1], self.goal_world_pose):
            raise ValueError("the final band pose must equal goal_world_pose")
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

    def __post_init__(self) -> None:
        goal = _vector(self.goal_world_pose, name="goal_world_pose", length=3)
        if self.route is not None and not np.array_equal(
            self.route.goal_world_pose, goal
        ):
            raise ValueError("route and result goal_world_pose values must match")
        object.__setattr__(self, "goal_world_pose", goal)


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


def _point_signed_distance_gradient(
    occluder: StaticOccluder,
    points_xy: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    center = _occluder_center(occluder)
    if isinstance(occluder, CircleOccluder):
        delta = points - center[None, :]
        norm = np.linalg.norm(delta, axis=1, keepdims=True)
        return delta / np.maximum(norm, 1e-12)

    yaw = float(occluder.pose[2])
    cosine = float(np.cos(yaw))
    sine = float(np.sin(yaw))
    delta = points - center[None, :]
    local = np.column_stack(
        (
            delta[:, 0] * cosine + delta[:, 1] * sine,
            -delta[:, 0] * sine + delta[:, 1] * cosine,
        )
    )
    half_extents = np.asarray(
        [0.5 * occluder.length_m, 0.5 * occluder.width_m],
        dtype=np.float64,
    )
    q = np.abs(local) - half_extents[None, :]
    positive_q = np.maximum(q, 0.0)
    outside_norm = np.linalg.norm(positive_q, axis=1, keepdims=True)
    local_gradient = (
        np.sign(local)
        * positive_q
        / np.maximum(outside_norm, 1e-12)
    )
    inside_rows = np.flatnonzero(outside_norm[:, 0] <= 1e-12)
    if inside_rows.size:
        nearest_axis = np.argmax(q[inside_rows], axis=1)
        local_gradient[inside_rows] = 0.0
        local_gradient[inside_rows, nearest_axis] = np.sign(
            local[inside_rows, nearest_axis]
        )
    return np.column_stack(
        (
            local_gradient[:, 0] * cosine
            - local_gradient[:, 1] * sine,
            local_gradient[:, 0] * sine
            + local_gradient[:, 1] * cosine,
        )
    )


def _direct_path_clearance(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    occluder: StaticOccluder,
    robot_radius: float,
) -> float:
    fractions = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    points = start_xy[None, :] + fractions[:, None] * (goal_xy - start_xy)[None, :]
    return float(
        np.min(point_signed_distance(occluder, points) - robot_radius)
    )


def _directly_relevant_occluders(
    request: StaticTebRequest,
    occluders: tuple[StaticOccluder, ...],
) -> tuple[StaticOccluder, ...]:
    robot_radius = _robot_radius(request.base_config)
    maximum_clearance = (
        request.planner_config.represented_occluder_clearance_range_m[1]
    )
    start_xy = request.start_pose[:2].astype(np.float64)
    goal_xy = request.local_goal_world_pose[:2].astype(np.float64)
    return tuple(
        occluder
        for occluder in occluders
        if _direct_path_clearance(
            start_xy,
            goal_xy,
            occluder,
            robot_radius,
        )
        <= maximum_clearance
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


def _initialize_straight_band(
    request: StaticTebRequest,
) -> tuple[np.ndarray, np.ndarray]:
    config = request.planner_config
    start = request.start_pose.astype(np.float64)
    goal = request.local_goal_world_pose.astype(np.float64)
    fractions = np.linspace(
        0.0,
        1.0,
        config.band_node_count,
        dtype=np.float64,
    )
    band = start[None, :] + fractions[:, None] * (goal - start)[None, :]
    yaw_delta = float(wrap_angle(goal[2] - start[2]))
    band[:, 2] = wrap_angle(start[2] + fractions * yaw_delta)
    band[0] = start
    band[-1] = goal
    interval_dt_s = np.full(
        config.band_node_count - 1,
        config.initial_band_dt_s,
        dtype=np.float64,
    )
    return band, interval_dt_s


def _sample_polyline_positions(
    vertices_xy: tuple[np.ndarray, ...],
    sample_count: int,
) -> np.ndarray:
    vertices = np.asarray(vertices_xy, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    distances = np.linspace(0.0, cumulative[-1], sample_count, dtype=np.float64)
    positions = np.empty((sample_count, 2), dtype=np.float64)
    for axis in range(2):
        positions[:, axis] = np.interp(distances, cumulative, vertices[:, axis])
    positions[0] = vertices[0]
    positions[-1] = vertices[-1]
    return positions


def _initialize_bypass_band(
    request: StaticTebRequest,
    *,
    side: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic obstacle-front/obstacle-back bypass seed."""

    if side not in {-1.0, 1.0}:
        raise ValueError("bypass side must be -1.0 or 1.0")
    if not request.occluders:
        return _initialize_straight_band(request)
    config = request.planner_config
    start = request.start_pose.astype(np.float64)
    goal = request.local_goal_world_pose.astype(np.float64)
    route_delta = goal[:2] - start[:2]
    route_length = float(np.linalg.norm(route_delta))
    if route_length <= 1e-9:
        return _initialize_straight_band(request)
    direction = route_delta / route_length
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    robot_radius = _robot_radius(request.base_config)
    required_clearance = (
        robot_radius
        + config.represented_occluder_clearance_range_m[0]
        + config.bypass_tracking_allowance_m
    )
    relevant_occluders = _directly_relevant_occluders(
        request,
        request.occluders,
    )
    if not relevant_occluders:
        return _initialize_straight_band(request)
    bounds = tuple(
        _occluder_projection_bounds(
            occluder,
            tangent=direction,
            normal=normal,
            origin=start[:2],
        )
        for occluder in relevant_occluders
    )
    before_m = float(
        np.clip(
            min(item[0] for item in bounds) - required_clearance,
            0.05 * route_length,
            0.90 * route_length,
        )
    )
    after_m = float(
        np.clip(
            max(item[1] for item in bounds) + required_clearance,
            before_m + 0.05 * route_length,
            0.95 * route_length,
        )
    )
    obstacle_edge_m = (
        max(item[3] for item in bounds)
        if side > 0.0
        else min(item[2] for item in bounds)
    )
    maximum_offset = max(2.5, 0.75 * route_length)
    offsets = np.arange(
        0.0,
        maximum_offset + 0.025,
        0.05,
        dtype=np.float64,
    )
    lateral_m = obstacle_edge_m + side * (required_clearance + offsets[-1])
    waypoints = (
        start[:2] + before_m * direction + lateral_m * normal,
        start[:2] + after_m * direction + lateral_m * normal,
    )
    for extra_offset_m in offsets:
        lateral_m = obstacle_edge_m + side * (
            required_clearance + float(extra_offset_m)
        )
        candidate_waypoints = (
            start[:2] + before_m * direction + lateral_m * normal,
            start[:2] + after_m * direction + lateral_m * normal,
        )
        dense_positions = _sample_polyline_positions(
            (start[:2], *candidate_waypoints, goal[:2]),
            401,
        )
        if all(
            float(np.min(point_signed_distance(occluder, dense_positions)))
            >= required_clearance
            for occluder in request.occluders
        ):
            waypoints = candidate_waypoints
            break

    positions = _sample_polyline_positions(
        (start[:2], *waypoints, goal[:2]),
        config.band_node_count,
    )
    yaws = np.empty(config.band_node_count, dtype=np.float64)
    tangents = positions[2:] - positions[:-2]
    yaws[1:-1] = np.arctan2(tangents[:, 1], tangents[:, 0])
    yaws[0] = start[2]
    yaws[-1] = goal[2]
    band = np.column_stack((positions, wrap_angle(yaws)))
    band[0] = start
    band[-1] = goal
    interval_dt_s = np.full(
        config.band_node_count - 1,
        config.initial_band_dt_s,
        dtype=np.float64,
    )
    return band, interval_dt_s


def _initialize_teb_band(
    request: StaticTebRequest,
    initialization_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    if initialization_id == "straight":
        return _initialize_straight_band(request)
    if initialization_id == "bypass_left":
        return _initialize_bypass_band(request, side=1.0)
    if initialization_id == "bypass_right":
        return _initialize_bypass_band(request, side=-1.0)
    raise ValueError(f"unsupported TEB initialization: {initialization_id!r}")


def _se2_interval_kinematics(
    start_poses: np.ndarray,
    end_poses: np.ndarray,
    interval_dt_s: np.ndarray,
) -> dict[str, np.ndarray]:
    starts = np.asarray(start_poses, dtype=np.float64)
    ends = np.asarray(end_poses, dtype=np.float64)
    dt_s = np.asarray(interval_dt_s, dtype=np.float64)
    delta_xy = ends[:, :2] - starts[:, :2]
    cosine = np.cos(starts[:, 2])
    sine = np.sin(starts[:, 2])
    local_delta = np.column_stack(
        (
            cosine * delta_xy[:, 0] + sine * delta_xy[:, 1],
            -sine * delta_xy[:, 0] + cosine * delta_xy[:, 1],
        )
    )
    delta_yaw = wrap_angle(ends[:, 2] - starts[:, 2])
    linear_velocity = np.empty(len(starts), dtype=np.float64)
    residual = np.zeros((len(starts), 2), dtype=np.float64)
    straight = np.abs(delta_yaw) <= 1e-12
    linear_velocity[straight] = local_delta[straight, 0] / dt_s[straight]
    residual[straight, 1] = local_delta[straight, 1]
    turning = ~straight
    if np.any(turning):
        turning_yaw = delta_yaw[turning]
        small = np.abs(turning_yaw) <= 1e-4
        sine = np.sin(turning_yaw)
        one_minus_cosine = 2.0 * np.sin(0.5 * turning_yaw) ** 2
        if np.any(small):
            small_yaw = turning_yaw[small]
            sine[small] = small_yaw * (
                1.0 - small_yaw**2 / 6.0 + small_yaw**4 / 120.0
            )
            one_minus_cosine[small] = small_yaw**2 * (
                0.5 - small_yaw**2 / 24.0 + small_yaw**4 / 720.0
            )
        basis = np.column_stack(
            (
                sine,
                one_minus_cosine,
            )
        )
        radius = np.sum(basis * local_delta[turning], axis=1) / np.maximum(
            np.sum(basis * basis, axis=1),
            np.finfo(np.float64).tiny,
        )
        predicted = radius[:, None] * basis
        residual[turning] = local_delta[turning] - predicted
        linear_velocity[turning] = (
            radius * turning_yaw / dt_s[turning]
        )
    angular_velocity = delta_yaw / dt_s
    return {
        "linear_velocity": linear_velocity,
        "angular_velocity": angular_velocity,
        "nonholonomic_residual": residual,
        "local_delta": local_delta,
    }


def _band_kinematics(
    request: StaticTebRequest,
    band: np.ndarray,
    interval_dt_s: np.ndarray,
) -> dict[str, np.ndarray]:
    delta_xy = np.diff(band[:, :2], axis=0)
    segment_length = np.linalg.norm(delta_xy, axis=1)
    segment_heading = np.arctan2(delta_xy[:, 1], delta_xy[:, 0])
    interval_kinematics = _se2_interval_kinematics(
        band[:-1],
        band[1:],
        interval_dt_s,
    )
    linear_velocity = interval_kinematics["linear_velocity"]
    angular_velocity = interval_kinematics["angular_velocity"]
    controls = np.column_stack((linear_velocity, angular_velocity))
    prior_controls = np.vstack(
        (request.initial_control.astype(np.float64), controls[:-1])
    )
    transition_dt_s = np.concatenate(
        (
            interval_dt_s[:1],
            0.5 * (interval_dt_s[:-1] + interval_dt_s[1:]),
        )
    )
    acceleration = (controls - prior_controls) / transition_dt_s[:, None]
    curvature = np.divide(
        angular_velocity,
        linear_velocity,
        out=np.zeros_like(angular_velocity),
        where=np.abs(linear_velocity) > 1e-9,
    )
    curvature_change = np.diff(curvature)
    return {
        "delta_xy": delta_xy,
        "segment_length": segment_length,
        "segment_heading": segment_heading,
        "linear_velocity": linear_velocity,
        "angular_velocity": angular_velocity,
        "controls": controls,
        "acceleration": acceleration,
        "nonholonomic_residual": interval_kinematics[
            "nonholonomic_residual"
        ],
        "curvature": curvature,
        "curvature_change": curvature_change,
    }


def _band_objective_terms(
    request: StaticTebRequest,
    band: np.ndarray,
    interval_dt_s: np.ndarray,
    robot_radius: float,
) -> tuple[tuple[str, float], ...]:
    config = request.planner_config
    weights = dict(config.weights)
    kinematics = _band_kinematics(request, band, interval_dt_s)
    max_speed, max_omega = _limits(request.base_config)
    linear_violation = np.maximum(
        np.abs(kinematics["linear_velocity"]) - max_speed,
        0.0,
    )
    angular_violation = np.maximum(
        np.abs(kinematics["angular_velocity"]) - max_omega,
        0.0,
    )
    linear_acceleration_violation = np.maximum(
        np.abs(kinematics["acceleration"][:, 0])
        - config.max_linear_acceleration_mps2,
        0.0,
    )
    angular_acceleration_violation = np.maximum(
        np.abs(kinematics["acceleration"][:, 1])
        - config.max_angular_acceleration_radps2,
        0.0,
    )
    obstacle_hinge = 0.0
    required_center_distance = (
        config.represented_occluder_clearance_range_m[0] + robot_radius
    )
    for occluder in request.occluders:
        distance = point_signed_distance(occluder, band[1:-1, :2])
        obstacle_hinge += float(
            np.sum(np.maximum(required_center_distance - distance, 0.0) ** 2)
        )
    first_control_delta = (
        kinematics["controls"][0]
        - request.initial_control.astype(np.float64)
    )
    terminal_heading_error = float(
        wrap_angle(
            kinematics["segment_heading"][-1]
            - request.local_goal_world_pose[2]
        )
    )
    terms = {
        "length": weights["length"]
        * float(np.sum(kinematics["segment_length"])),
        "time": weights["time"] * float(np.sum(interval_dt_s)),
        "smoothness": weights["smoothness"]
        * float(np.sum(kinematics["curvature_change"] ** 2)),
        "obstacle": weights["obstacle"] * obstacle_hinge,
        "nonholonomic": weights["nonholonomic"]
        * float(np.sum(kinematics["nonholonomic_residual"] ** 2)),
        "velocity": weights["velocity"]
        * float(
            np.sum((linear_violation / max_speed) ** 2)
            + np.sum((angular_violation / max_omega) ** 2)
        ),
        "acceleration": weights["acceleration"]
        * float(
            np.sum(
                (
                    linear_acceleration_violation
                    / config.max_linear_acceleration_mps2
                )
                ** 2
            )
            + np.sum(
                (
                    angular_acceleration_violation
                    / config.max_angular_acceleration_radps2
                )
                ** 2
            )
        ),
        "goal_heading": weights["goal_heading"] * terminal_heading_error**2,
        "initial_control": weights["initial_control"]
        * float(
            (first_control_delta[0] / max_speed) ** 2
            + (first_control_delta[1] / max_omega) ** 2
        ),
    }
    return tuple((name, float(terms[name])) for name in weights)


def _local_nonholonomic_residual_cost(
    request: StaticTebRequest,
    positions: np.ndarray,
    yaws: np.ndarray,
    interval_dt_s: np.ndarray,
    node_index: int,
) -> float:
    interval_start = max(0, node_index - 1)
    interval_stop = min(len(interval_dt_s), node_index + 1)
    band = np.column_stack(
        (
            positions[interval_start : interval_stop + 1],
            yaws[interval_start : interval_stop + 1],
        )
    )
    residual = _band_kinematics(
        request,
        band,
        interval_dt_s[interval_start:interval_stop],
    )["nonholonomic_residual"]
    return float(np.sum(residual**2))


def _nonholonomic_residual_gradients(
    request: StaticTebRequest,
    positions: np.ndarray,
    yaws: np.ndarray,
    interval_dt_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position_gradient = np.zeros_like(positions)
    yaw_gradient = np.zeros_like(yaws)
    position_epsilon = 1e-4
    yaw_epsilon = 1e-5
    for index in range(1, len(positions) - 1):
        for axis in range(2):
            positive = positions.copy()
            negative = positions.copy()
            positive[index, axis] += position_epsilon
            negative[index, axis] -= position_epsilon
            position_gradient[index, axis] = (
                _local_nonholonomic_residual_cost(
                    request,
                    positive,
                    yaws,
                    interval_dt_s,
                    index,
                )
                - _local_nonholonomic_residual_cost(
                    request,
                    negative,
                    yaws,
                    interval_dt_s,
                    index,
                )
            ) / (2.0 * position_epsilon)
        positive_yaw = yaws.copy()
        negative_yaw = yaws.copy()
        positive_yaw[index] = wrap_angle(
            positive_yaw[index] + yaw_epsilon
        )
        negative_yaw[index] = wrap_angle(
            negative_yaw[index] - yaw_epsilon
        )
        yaw_gradient[index] = (
            _local_nonholonomic_residual_cost(
                request,
                positions,
                positive_yaw,
                interval_dt_s,
                index,
            )
            - _local_nonholonomic_residual_cost(
                request,
                positions,
                negative_yaw,
                interval_dt_s,
                index,
            )
        ) / (2.0 * yaw_epsilon)
    return position_gradient, yaw_gradient


def _apply_nonholonomic_residual_update(
    request: StaticTebRequest,
    positions: np.ndarray,
    yaws: np.ndarray,
    interval_dt_s: np.ndarray,
    *,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    position_gradient, yaw_gradient = _nonholonomic_residual_gradients(
        request,
        positions,
        yaws,
        interval_dt_s,
    )
    updated_positions = np.asarray(positions, dtype=np.float64).copy()
    updated_yaws = np.asarray(yaws, dtype=np.float64).copy()
    updated_positions[1:-1] -= np.clip(
        0.1 * weight * position_gradient[1:-1],
        -0.01,
        0.01,
    )
    updated_yaws[1:-1] = wrap_angle(
        updated_yaws[1:-1]
        - np.clip(
            5.0 * weight * yaw_gradient[1:-1],
            -0.1,
            0.1,
        )
    )
    return updated_positions, updated_yaws


def _curvature_change_cost(
    request: StaticTebRequest,
    positions: np.ndarray,
    yaws: np.ndarray,
    interval_dt_s: np.ndarray,
) -> float:
    band = np.column_stack((positions, yaws))
    curvature_change = _band_kinematics(
        request,
        band,
        interval_dt_s,
    )["curvature_change"]
    return float(np.sum(curvature_change**2))


def _local_curvature_change_cost(
    request: StaticTebRequest,
    positions: np.ndarray,
    yaws: np.ndarray,
    interval_dt_s: np.ndarray,
    node_index: int,
) -> float:
    interval_start = max(0, node_index - 2)
    interval_stop = min(len(interval_dt_s), node_index + 2)
    band = np.column_stack(
        (
            positions[interval_start : interval_stop + 1],
            yaws[interval_start : interval_stop + 1],
        )
    )
    curvature_change = _band_kinematics(
        request,
        band,
        interval_dt_s[interval_start:interval_stop],
    )["curvature_change"]
    return float(np.sum(curvature_change**2))


def _curvature_change_gradients(
    request: StaticTebRequest,
    positions: np.ndarray,
    yaws: np.ndarray,
    interval_dt_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position_gradient = np.zeros_like(positions)
    yaw_gradient = np.zeros_like(yaws)
    position_epsilon = 1e-4
    yaw_epsilon = 1e-5
    for index in range(1, len(positions) - 1):
        for axis in range(2):
            positive = positions.copy()
            negative = positions.copy()
            positive[index, axis] += position_epsilon
            negative[index, axis] -= position_epsilon
            position_gradient[index, axis] = (
                _local_curvature_change_cost(
                    request,
                    positive,
                    yaws,
                    interval_dt_s,
                    index,
                )
                - _local_curvature_change_cost(
                    request,
                    negative,
                    yaws,
                    interval_dt_s,
                    index,
                )
            ) / (2.0 * position_epsilon)
        positive_yaw = yaws.copy()
        negative_yaw = yaws.copy()
        positive_yaw[index] = wrap_angle(
            positive_yaw[index] + yaw_epsilon
        )
        negative_yaw[index] = wrap_angle(
            negative_yaw[index] - yaw_epsilon
        )
        yaw_gradient[index] = (
            _local_curvature_change_cost(
                request,
                positions,
                positive_yaw,
                interval_dt_s,
                index,
            )
            - _local_curvature_change_cost(
                request,
                positions,
                negative_yaw,
                interval_dt_s,
                index,
            )
        ) / (2.0 * yaw_epsilon)
    return position_gradient, yaw_gradient


def _optimize_teb_band(
    request: StaticTebRequest,
    *,
    side: float | None,
    robot_radius: float,
    initialization_id: str = "straight",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run the fixed-count elastic-band/time updates for one initialization."""

    config = request.planner_config
    weights = dict(config.weights)
    start = request.start_pose.astype(np.float64)
    goal = request.local_goal_world_pose.astype(np.float64)
    seed_band, dt_s = _initialize_teb_band(request, initialization_id)
    positions = seed_band[:, :2].copy()
    yaws = seed_band[:, 2].copy()
    clearance = config.represented_occluder_clearance_range_m[0] + robot_radius
    bounds = config.band_dt_range_s
    max_speed, max_omega = _limits(request.base_config)
    route_delta = goal[:2] - start[:2]
    route_length = float(np.linalg.norm(route_delta))
    route_normal = (
        np.asarray([-route_delta[1], route_delta[0]], dtype=np.float64)
        / route_length
        if route_length > 1e-9
        else np.asarray([0.0, 1.0], dtype=np.float64)
    )
    iterations = 0
    smoothness_position_gradient = np.zeros_like(positions)
    smoothness_yaw_gradient = np.zeros_like(yaws)
    for _ in range(config.max_iterations):
        iterations += 1
        updated = positions.copy()
        previous_delta = positions[1:-1] - positions[:-2]
        next_delta = positions[2:] - positions[1:-1]
        previous_unit = previous_delta / np.maximum(
            np.linalg.norm(previous_delta, axis=1, keepdims=True),
            1e-9,
        )
        next_unit = next_delta / np.maximum(
            np.linalg.norm(next_delta, axis=1, keepdims=True),
            1e-9,
        )
        length_gradient = previous_unit - next_unit
        if (iterations - 1) % _SPARSE_OBJECTIVE_GRADIENT_REFRESH == 0:
            (
                smoothness_position_gradient,
                smoothness_yaw_gradient,
            ) = _curvature_change_gradients(
                request,
                positions,
                yaws,
                dt_s,
            )
        objective_gradient = (
            weights["length"] * length_gradient
            + weights["smoothness"]
            * np.clip(
                smoothness_position_gradient[1:-1],
                -1.0,
                1.0,
            )
        )
        for occluder in request.occluders:
            distances = point_signed_distance(occluder, positions[1:-1])
            violations = np.maximum(0.0, clearance - distances)
            if not np.any(violations > 0.0):
                continue
            gradient = _point_signed_distance_gradient(
                occluder,
                positions[1:-1],
            )
            norm = np.linalg.norm(gradient, axis=1, keepdims=True)
            fallback_side = 1.0 if side is None else side
            gradient = np.where(
                norm > 1e-9,
                gradient / np.maximum(norm, 1e-9),
                fallback_side * route_normal[None, :],
            )
            objective_gradient -= (
                2.0
                * weights["obstacle"]
                * violations[:, None]
                * gradient
            )
        updated[1:-1] -= 0.03 * objective_gradient
        terminal_delta = goal[:2] - updated[-2]
        terminal_length = float(np.linalg.norm(terminal_delta))
        desired_terminal_start = goal[:2] - terminal_length * np.asarray(
            [np.cos(goal[2]), np.sin(goal[2])],
            dtype=np.float64,
        )
        goal_heading_blend = min(0.2, 0.01 * weights["goal_heading"])
        updated[-2] += goal_heading_blend * (
            desired_terminal_start - updated[-2]
        )
        desired_initial_pose = integrate_twist(
            start,
            v=float(request.initial_control[0]),
            omega=float(request.initial_control[1]),
            dt_s=float(dt_s[0]),
        )
        initial_control_blend = min(
            0.2,
            0.01 * weights["initial_control"],
        )
        updated[1] += initial_control_blend * (
            desired_initial_pose[:2] - updated[1]
        )
        updated[0] = start[:2]
        updated[-1] = goal[:2]
        positions = updated
        if (iterations - 1) % _SPARSE_OBJECTIVE_GRADIENT_REFRESH == 0:
            positions, yaws = _apply_nonholonomic_residual_update(
                request,
                positions,
                yaws,
                dt_s,
                weight=weights["nonholonomic"],
            )
        yaws[1] = wrap_angle(
            yaws[1]
            + initial_control_blend
            * wrap_angle(desired_initial_pose[2] - yaws[1])
        )
        yaws[-2] = wrap_angle(
            yaws[-2]
            + goal_heading_blend * wrap_angle(goal[2] - yaws[-2])
        )
        yaws[1:-1] = wrap_angle(
            yaws[1:-1]
            - 0.003
            * weights["smoothness"]
            * np.clip(smoothness_yaw_gradient[1:-1], -2.0, 2.0)
        )
        yaws[0] = start[2]
        yaws[-1] = goal[2]

        lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        heading_delta = np.abs(wrap_angle(np.diff(yaws)))
        temporary_band = np.column_stack((positions, yaws))
        kinematics = _band_kinematics(request, temporary_band, dt_s)
        linear_velocity_violation = np.maximum(
            np.abs(kinematics["linear_velocity"]) - max_speed,
            0.0,
        )
        angular_velocity_violation = np.maximum(
            np.abs(kinematics["angular_velocity"]) - max_omega,
            0.0,
        )
        velocity_pressure = (
            linear_velocity_violation / max_speed
            + angular_velocity_violation / max_omega
        )
        projected_min_dt = np.maximum.reduce(
            (
                np.full_like(lengths, bounds[0]),
                lengths / max_speed,
                heading_delta / max_omega,
            )
        )
        dt_s -= 0.01 * weights["time"]
        dt_s += (
            0.01
            * weights["velocity"]
            * np.minimum(velocity_pressure, 4.0)
        )
        velocity_projection_reserve = min(
            0.02,
            0.0025 * weights["velocity"],
        )
        dt_s = np.maximum(
            dt_s,
            projected_min_dt
            + velocity_projection_reserve
            * (projected_min_dt > bounds[0] + 1e-12),
        )
        projected_band = np.column_stack((positions, yaws))
        projected_kinematics = _band_kinematics(
            request,
            projected_band,
            np.clip(dt_s, bounds[0], bounds[1]),
        )
        linear_acceleration_pressure = np.maximum(
            np.abs(projected_kinematics["acceleration"][:, 0])
            - config.max_linear_acceleration_mps2,
            0.0,
        ) / config.max_linear_acceleration_mps2
        angular_acceleration_pressure = np.maximum(
            np.abs(projected_kinematics["acceleration"][:, 1])
            - config.max_angular_acceleration_radps2,
            0.0,
        ) / config.max_angular_acceleration_radps2
        acceleration_pressure = (
            linear_acceleration_pressure + angular_acceleration_pressure
        )
        dt_s += (
            0.002
            * weights["acceleration"]
            * np.minimum(acceleration_pressure, 4.0)
        )
        first_acceleration_blend = min(
            0.15,
            0.003
            * weights["acceleration"]
            * float(min(acceleration_pressure[0], 4.0)),
        )
        positions[1] = start[:2] + (
            1.0 - first_acceleration_blend
        ) * (positions[1] - start[:2])
        dt_s = np.clip(dt_s, bounds[0], bounds[1])
        if float(np.sum(dt_s)) > config.maximum_route_time_s:
            dt_s *= config.maximum_route_time_s / float(np.sum(dt_s))
            dt_s = np.clip(dt_s, bounds[0], bounds[1])
    # Preserve explicit clearance reserve through band-to-control projection.
    projected_clearance = clearance + config.bypass_tracking_allowance_m
    for _ in range(3):
        for occluder in request.occluders:
            distances = point_signed_distance(occluder, positions[1:-1])
            violations = np.maximum(0.0, projected_clearance - distances)
            if not np.any(violations > 0.0):
                continue
            gradient = _point_signed_distance_gradient(
                occluder,
                positions[1:-1],
            )
            norm = np.linalg.norm(gradient, axis=1, keepdims=True)
            fallback_side = 1.0 if side is None else side
            gradient = np.where(
                norm > 1e-9,
                gradient / np.maximum(norm, 1e-9),
                fallback_side * route_normal[None, :],
            )
            positions[1:-1] += (
                violations[:, None] + 1e-4
            ) * gradient
    lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    heading_delta = np.abs(wrap_angle(np.diff(yaws)))
    projected_min_dt = np.maximum.reduce(
        (
            np.full_like(lengths, bounds[0]),
            lengths / max_speed,
            heading_delta / max_omega,
        )
    )
    velocity_projection_reserve = min(
        0.02,
        0.0025 * weights["velocity"],
    )
    dt_s = np.clip(
        np.maximum(
            dt_s,
            projected_min_dt
            + velocity_projection_reserve
            * (projected_min_dt > bounds[0] + 1e-12),
        ),
        bounds[0],
        bounds[1],
    )
    band = np.column_stack((positions, yaws))
    band[0] = start
    band[-1] = goal
    return band, dt_s, iterations


def _project_control_jointly(
    desired_control: np.ndarray,
    previous_control: np.ndarray,
    *,
    base_config: Mapping[str, object],
    planner_config: LightweightTebConfig,
    dt_s: float,
) -> np.ndarray:
    desired = np.asarray(desired_control, dtype=np.float64)
    previous = np.asarray(previous_control, dtype=np.float64)
    max_speed, max_omega = _limits(base_config)
    linear_delta = planner_config.max_linear_acceleration_mps2 * dt_s
    angular_delta = planner_config.max_angular_acceleration_radps2 * dt_s
    v_min = max(-max_speed, float(previous[0] - linear_delta))
    v_max = min(max_speed, float(previous[0] + linear_delta))
    omega_min = max(-max_omega, float(previous[1] - angular_delta))
    omega_max = min(max_omega, float(previous[1] + angular_delta))
    curvature = planner_config.max_curvature_per_m

    candidates = [
        float(np.clip(desired[0], v_min, v_max)),
        float(np.clip(previous[0], v_min, v_max)),
        v_min,
        v_max,
    ]
    if v_min <= 0.0 <= v_max:
        candidates.append(0.0)
    nearest_omega_magnitude = (
        0.0
        if omega_min <= 0.0 <= omega_max
        else min(abs(omega_min), abs(omega_max))
    )
    required_speed = nearest_omega_magnitude / curvature
    for sign in (-1.0, 1.0):
        candidate = sign * required_speed
        if v_min <= candidate <= v_max:
            candidates.append(candidate)
    desired_omega = float(np.clip(desired[1], omega_min, omega_max))
    desired_curvature_speed = abs(desired_omega) / curvature
    for sign in (-1.0, 1.0):
        candidate = sign * desired_curvature_speed
        if v_min <= candidate <= v_max:
            candidates.append(candidate)

    feasible: list[tuple[float, float, float, int]] = []
    for order, velocity in enumerate(candidates):
        if abs(velocity) <= 1e-9:
            feasible_omega_min = omega_min
            feasible_omega_max = omega_max
        else:
            curvature_omega = curvature * abs(velocity)
            feasible_omega_min = max(omega_min, -curvature_omega)
            feasible_omega_max = min(omega_max, curvature_omega)
        if feasible_omega_min > feasible_omega_max + 1e-12:
            continue
        omega = float(
            np.clip(
                desired[1],
                feasible_omega_min,
                feasible_omega_max,
            )
        )
        score = (
            ((velocity - desired[0]) / max_speed) ** 2
            + ((omega - desired[1]) / max_omega) ** 2
        )
        feasible.append((float(score), velocity, omega, order))
    if not feasible:
        return np.asarray(
            [
                np.clip(desired[0], v_min, v_max),
                np.clip(desired[1], omega_min, omega_max),
            ],
            dtype=np.float64,
        )
    _, velocity, omega, _ = min(
        feasible,
        key=lambda item: (item[0], item[3]),
    )
    return np.asarray([velocity, omega], dtype=np.float64)


def _goal_tolerance_mask(
    request: StaticTebRequest,
    poses: np.ndarray,
) -> np.ndarray:
    goal = request.local_goal_world_pose.astype(np.float64)
    pose_array = np.asarray(poses, dtype=np.float64)
    return (
        np.linalg.norm(pose_array[:, :2] - goal[:2], axis=1)
        <= request.planner_config.goal_position_tolerance_m
    )


def _has_terminal_goal_suffix(
    request: StaticTebRequest,
    poses: np.ndarray,
) -> bool:
    return _stable_goal_arrival_index(request, poses) is not None


def _stable_goal_arrival_index(
    request: StaticTebRequest,
    poses: np.ndarray,
) -> int | None:
    mask = _goal_tolerance_mask(request, poses)
    if (
        len(mask) < _TERMINAL_TOLERANCE_SUFFIX_COUNT
        or not np.all(mask[-_TERMINAL_TOLERANCE_SUFFIX_COUNT:])
    ):
        return None
    outside = np.flatnonzero(~mask)
    return 0 if outside.size == 0 else int(outside[-1] + 1)


def _sampled_dynamics_rejection(
    request: StaticTebRequest,
    poses: np.ndarray,
    controls: np.ndarray,
) -> str | None:
    pose_array = np.asarray(poses, dtype=np.float64)
    control_array = np.asarray(controls, dtype=np.float64)
    if (
        pose_array.shape != (_V3_ROUTE_SAMPLE_COUNT, 3)
        or control_array.shape != (_V3_ROUTE_SAMPLE_COUNT, 2)
        or not np.isfinite(pose_array).all()
        or not np.isfinite(control_array).all()
    ):
        return "teb_dynamics_limit"
    config = request.planner_config
    dt_s = config.route_sample_dt_s
    max_speed, max_omega = _limits(request.base_config)
    tolerance = 1e-6
    if (
        np.any(np.abs(control_array[:, 0]) > max_speed + tolerance)
        or np.any(np.abs(control_array[:, 1]) > max_omega + tolerance)
    ):
        return "teb_dynamics_limit"
    prior_controls = np.vstack(
        (request.initial_control.astype(np.float64), control_array[:-1])
    )
    control_delta = np.abs(control_array - prior_controls)
    if (
        np.any(
            control_delta[:, 0]
            > config.max_linear_acceleration_mps2 * dt_s + tolerance
        )
        or np.any(
            control_delta[:, 1]
            > config.max_angular_acceleration_radps2 * dt_s + tolerance
        )
    ):
        return "teb_dynamics_limit"
    moving = np.abs(control_array[:, 0]) > 1e-6
    if np.any(
        np.abs(control_array[moving, 1] / control_array[moving, 0])
        > config.max_curvature_per_m + tolerance
    ):
        return "teb_dynamics_limit"
    prior_pose = request.start_pose.astype(np.float64)
    for control, pose in zip(control_array, pose_array, strict=True):
        expected = integrate_twist(
            prior_pose,
            v=float(control[0]),
            omega=float(control[1]),
            dt_s=dt_s,
        )
        if (
            np.linalg.norm(expected[:2] - pose[:2]) > 1e-5
            or abs(float(wrap_angle(expected[2] - pose[2]))) > 1e-5
        ):
            return "teb_dynamics_limit"
        prior_pose = pose
    return None


def _dense_control_poses(
    start_pose: np.ndarray,
    controls: np.ndarray,
    *,
    dt_s: float,
) -> np.ndarray:
    pose = np.asarray(start_pose, dtype=np.float64)
    dense: list[np.ndarray] = []
    for control in np.asarray(controls, dtype=np.float64):
        for fraction in _DENSE_COLLISION_FRACTIONS:
            if fraction == 0.0:
                dense.append(pose.copy())
            else:
                dense.append(
                    integrate_twist(
                        pose,
                        v=float(control[0]),
                        omega=float(control[1]),
                        dt_s=dt_s * float(fraction),
                    )
                )
        pose = integrate_twist(
            pose,
            v=float(control[0]),
            omega=float(control[1]),
            dt_s=dt_s,
        )
    return np.asarray(dense, dtype=np.float64)


def _route_poses_at_times(
    start_pose: np.ndarray,
    controls: np.ndarray,
    *,
    dt_s: float,
    times_s: np.ndarray,
) -> np.ndarray:
    anchors = [np.asarray(start_pose, dtype=np.float64)]
    controls64 = np.asarray(controls, dtype=np.float64)
    for control in controls64:
        anchors.append(
            integrate_twist(
                anchors[-1],
                v=float(control[0]),
                omega=float(control[1]),
                dt_s=dt_s,
            )
        )
    result: list[np.ndarray] = []
    horizon_s = len(controls64) * dt_s
    for raw_time in np.asarray(times_s, dtype=np.float64):
        time_s = float(np.clip(raw_time, 0.0, horizon_s))
        completed = min(
            int(np.floor((time_s + 1e-10) / dt_s)),
            len(controls64),
        )
        if completed == len(controls64):
            result.append(anchors[-1])
            continue
        remainder = max(0.0, time_s - completed * dt_s)
        if remainder <= 1e-12:
            result.append(anchors[completed])
        else:
            result.append(
                integrate_twist(
                    anchors[completed],
                    v=float(controls64[completed, 0]),
                    omega=float(controls64[completed, 1]),
                    dt_s=remainder,
                )
            )
    return np.asarray(result, dtype=np.float64)


def _interpolate_timed_band(
    band: np.ndarray,
    interval_dt_s: np.ndarray,
    times_s: np.ndarray,
) -> np.ndarray:
    node_times = np.concatenate(([0.0], np.cumsum(interval_dt_s)))
    clipped_times = np.clip(
        np.asarray(times_s, dtype=np.float64),
        0.0,
        node_times[-1],
    )
    result = np.empty((len(clipped_times), 3), dtype=np.float64)
    for axis in range(2):
        result[:, axis] = np.interp(clipped_times, node_times, band[:, axis])
    unwrapped_yaw = np.unwrap(band[:, 2])
    result[:, 2] = wrap_angle(
        np.interp(clipped_times, node_times, unwrapped_yaw)
    )
    return result


def _scale_intervals_to_total(
    interval_dt_s: np.ndarray,
    *,
    total_time_s: float,
    bounds: tuple[float, float],
) -> np.ndarray:
    count = len(interval_dt_s)
    target = float(
        np.clip(total_time_s, count * bounds[0], count * bounds[1])
    )
    result = np.asarray(interval_dt_s, dtype=np.float64).copy()
    result *= target / max(float(np.sum(result)), 1e-12)
    result = np.clip(result, bounds[0], bounds[1])
    for _ in range(count + 1):
        residual = target - float(np.sum(result))
        if abs(residual) <= 1e-10:
            break
        if residual > 0.0:
            free = np.flatnonzero(result < bounds[1] - 1e-12)
        else:
            free = np.flatnonzero(result > bounds[0] + 1e-12)
        if free.size == 0:
            break
        result[free] += residual / float(free.size)
        result = np.clip(result, bounds[0], bounds[1])
    return result


def _project_optimized_band(
    request: StaticTebRequest,
    optimized_band: np.ndarray,
    optimized_dt_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    config = request.planner_config
    dt_s = config.route_sample_dt_s
    sample_count = int(round(config.maximum_route_time_s / dt_s))
    max_speed, max_omega = _limits(request.base_config)
    start = request.start_pose.astype(np.float64)
    goal = request.local_goal_world_pose.astype(np.float64)
    pose = start.copy()
    previous_control = request.initial_control.astype(np.float64)
    poses = np.empty((sample_count, 3), dtype=np.float64)
    controls = np.empty((sample_count, 2), dtype=np.float64)
    arrival_index: int | None = None
    terminal_braked = False
    lookahead_s = dt_s
    target_times = (
        np.arange(1, sample_count + 1, dtype=np.float64) * dt_s
        + lookahead_s
    )
    targets = _interpolate_timed_band(
        optimized_band,
        optimized_dt_s,
        target_times,
    )

    for index, target in enumerate(targets):
        delta_xy = target[:2] - pose[:2]
        distance = float(np.linalg.norm(delta_xy))
        terminal_distance = float(np.linalg.norm(goal[:2] - pose[:2]))
        target_yaw = (
            float(goal[2])
            if terminal_distance
            <= 0.5 * config.goal_position_tolerance_m
            else (
                float(np.arctan2(delta_xy[1], delta_xy[0]))
                if distance > 1e-9
                else float(target[2])
            )
        )
        heading_error = float(wrap_angle(target_yaw - pose[2]))
        desired_omega = float(
            np.clip(heading_error / dt_s, -max_omega, max_omega)
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
        heading_alignment = float(np.cos(heading_error))
        desired_speed *= max(0.0, heading_alignment)
        if abs(heading_error) > 0.25 * np.pi:
            desired_speed = 0.0
            if abs(previous_control[0]) > 1e-6:
                desired_omega = 0.0
        elif desired_speed > 1e-6:
            desired_omega = float(
                np.clip(
                    desired_omega,
                    -config.max_curvature_per_m * desired_speed,
                    config.max_curvature_per_m * desired_speed,
                )
            )
        ready_to_settle = bool(
            arrival_index is not None
            and terminal_distance
            <= 0.5 * config.goal_position_tolerance_m
            and abs(float(wrap_angle(pose[2] - goal[2])))
            <= 0.05 * config.goal_yaw_tolerance_rad
        )
        terminal_braking = bool(
            arrival_index is not None
            and not terminal_braked
        )
        desired_control = (
            np.zeros(2, dtype=np.float64)
            if ready_to_settle or terminal_braking
            else np.asarray(
                [desired_speed, desired_omega],
                dtype=np.float64,
            )
        )
        control = _project_control_jointly(
            desired_control,
            previous_control,
            base_config=request.base_config,
            planner_config=config,
            dt_s=dt_s,
        )
        pose = integrate_twist(
            pose,
            v=float(control[0]),
            omega=float(control[1]),
            dt_s=dt_s,
        )
        poses[index] = pose
        controls[index] = control
        previous_control = control
        if (
            arrival_index is not None
            and not terminal_braked
            and abs(float(control[0])) <= 1e-6
            and abs(float(control[1])) <= 1e-6
        ):
            terminal_braked = True
        if (
            arrival_index is None
            and np.linalg.norm(pose[:2] - goal[:2])
            <= config.goal_position_tolerance_m
        ):
            arrival_index = index

    stable_arrival_index = _stable_goal_arrival_index(request, poses)
    if stable_arrival_index is None:
        return poses, controls, -1
    return poses, controls, stable_arrival_index


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
    interval_kinematics = _se2_interval_kinematics(
        pose_with_start[:-1],
        pose_with_start[1:],
        np.full(len(controls), dt_s, dtype=np.float64),
    )
    prior_controls = np.vstack((initial, controls[:-1]))
    accelerations = (controls - prior_controls) / dt_s
    curvature = np.divide(
        controls[:, 1],
        controls[:, 0],
        out=np.zeros(len(controls), dtype=np.float64),
        where=np.abs(controls[:, 0]) > 1e-6,
    )
    curvature_change = np.diff(curvature)
    linear_velocity_violation = np.maximum(
        np.abs(controls[:, 0]) - max_speed,
        0.0,
    )
    angular_velocity_violation = np.maximum(
        np.abs(controls[:, 1]) - max_omega,
        0.0,
    )
    linear_acceleration_violation = np.maximum(
        np.abs(accelerations[:, 0])
        - request.planner_config.max_linear_acceleration_mps2,
        0.0,
    )
    angular_acceleration_violation = np.maximum(
        np.abs(accelerations[:, 1])
        - request.planner_config.max_angular_acceleration_radps2,
        0.0,
    )
    terms = {
        "length": weights["length"]
        * float(np.sum(np.linalg.norm(displacements, axis=1))),
        "time": weights["time"] * arrival_time_s,
        "smoothness": weights["smoothness"]
        * float(np.sum(curvature_change**2)),
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
        "nonholonomic": weights["nonholonomic"]
        * float(
            np.sum(interval_kinematics["nonholonomic_residual"] ** 2)
        ),
        "velocity": weights["velocity"]
        * float(
            np.sum((linear_velocity_violation / max_speed) ** 2)
            + np.sum((angular_velocity_violation / max_omega) ** 2)
        ),
        "acceleration": weights["acceleration"]
        * float(
            np.sum(
                (
                    linear_acceleration_violation
                    / request.planner_config.max_linear_acceleration_mps2
                )
                ** 2
            )
            + np.sum(
                (
                    angular_acceleration_violation
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
    clearance_band_occluders: tuple[StaticOccluder, ...],
) -> tuple[
    PlannedTebRoute | None,
    str | None,
    float,
    int,
    tuple[tuple[str, float], ...] | None,
]:
    config = request.planner_config
    dt_s = config.route_sample_dt_s
    sample_count = int(round(config.maximum_route_time_s / dt_s))
    robot_radius = _robot_radius(request.base_config)
    clearance_bounds = config.represented_occluder_clearance_range_m
    start = request.start_pose.astype(np.float64)
    goal = request.local_goal_world_pose.astype(np.float64)
    optimized_band, optimized_dt_s, optimization_iterations = _optimize_teb_band(
        request,
        side=side,
        robot_radius=robot_radius,
        initialization_id=initialization_id,
    )
    poses, controls, arrival_index = _project_optimized_band(
        request,
        optimized_band,
        optimized_dt_s,
    )
    if arrival_index < 0:
        return (
            None,
            "teb_goal_unreached",
            _FAILURE_COST,
            optimization_iterations,
            None,
        )
    dynamics_rejection = _sampled_dynamics_rejection(
        request,
        poses,
        controls,
    )
    if dynamics_rejection is not None:
        return (
            None,
            dynamics_rejection,
            _FAILURE_COST,
            optimization_iterations,
            None,
        )
    if not _has_terminal_goal_suffix(request, poses):
        return (
            None,
            "teb_goal_unreached",
            _FAILURE_COST,
            optimization_iterations,
            None,
        )
    if _has_static_collision(
        start_pose=request.start_pose,
        controls=controls,
        dt_s=dt_s,
        static_occupancy=request.static_occupancy,
        base_config=request.base_config,
        grid=grid,
    ):
        return (
            None,
            "teb_static_collision",
            _FAILURE_COST,
            optimization_iterations,
            None,
        )
    dense_poses = _dense_control_poses(
        request.start_pose,
        controls,
        dt_s=dt_s,
    )
    if request.occluders:
        per_occluder_clearance = tuple(
            float(
                np.min(
                    point_signed_distance(occluder, dense_poses[:, :2])
                    - robot_radius
                )
            )
            for occluder in request.occluders
        )
        if any(
            clearance < clearance_bounds[0]
            for clearance in per_occluder_clearance
        ):
            return (
                None,
                "teb_static_collision",
                _FAILURE_COST,
                optimization_iterations,
                None,
            )
        relevant_clearances = tuple(
            float(
                np.min(
                    point_signed_distance(occluder, dense_poses[:, :2])
                    - robot_radius
                )
            )
            for occluder in clearance_band_occluders
        )
        if any(
            clearance > clearance_bounds[1]
            for clearance in relevant_clearances
        ):
            return (
                None,
                "teb_static_collision",
                _FAILURE_COST,
                optimization_iterations,
                None,
            )
        clearance = min(per_occluder_clearance)
    else:
        clearance = 0.0

    arrival_time_s = (arrival_index + 1) * dt_s
    stopped_at_goal = np.flatnonzero(
        _goal_tolerance_mask(request, poses)
        & (
            np.linalg.norm(poses[:, :2] - goal[:2], axis=1)
            <= 0.5 * config.goal_position_tolerance_m
        )
        & (
            np.abs(wrap_angle(poses[:, 2] - goal[2]))
            <= 0.05 * config.goal_yaw_tolerance_rad
        )
        & (np.abs(controls[:, 0]) <= 1e-5)
        & (np.abs(controls[:, 1]) <= 1e-5)
    )
    settled_time_s = (
        (int(stopped_at_goal[0]) + 1) * dt_s
        if stopped_at_goal.size
        else config.maximum_route_time_s
    )
    output_band_dt_s = _scale_intervals_to_total(
        optimized_dt_s,
        total_time_s=max(float(np.sum(optimized_dt_s)), settled_time_s),
        bounds=config.band_dt_range_s,
    )
    output_band_times = np.concatenate(
        ([0.0], np.cumsum(output_band_dt_s))
    )
    output_band = _route_poses_at_times(
        request.start_pose,
        controls,
        dt_s=dt_s,
        times_s=output_band_times,
    )
    output_band[0] = start
    output_band[-1] = goal
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
        band_poses_world=output_band,
        band_interval_dt_s=output_band_dt_s,
        sample_times_s=(
            np.arange(1, sample_count + 1, dtype=np.float32)
            * np.float32(dt_s)
        ),
        sampled_poses_world=poses,
        sampled_controls=controls,
        goal_arrival_time_s=arrival_time_s,
        task_cost=cost,
    )
    return route, None, cost, optimization_iterations, cost_terms


def _has_static_collision(
    *,
    start_pose: np.ndarray,
    controls: np.ndarray,
    dt_s: float,
    static_occupancy: np.ndarray,
    base_config: Mapping[str, object],
    grid: GridSpec,
) -> bool:
    if not np.any(static_occupancy):
        return False
    robot = base_config["robot"]
    if not isinstance(robot, Mapping):
        raise ValueError("base_config.robot must be a mapping")
    footprint = inflate_footprint(
        RectangleFootprint(float(robot["length_m"]), float(robot["width_m"])),
        float(robot["inflation_m"]),
    )
    for pose in _dense_control_poses(start_pose, controls, dt_s=dt_s):
        mask = rasterize_footprint(footprint, pose, grid)
        if np.any(mask & (static_occupancy != 0.0)):
            return True
    return False


def _plan_lightweight_teb(
    request: StaticTebRequest,
    *,
    clearance_band_occluders: tuple[StaticOccluder, ...],
) -> LightweightTebResult:
    if not isinstance(request, StaticTebRequest):
        raise TypeError("request must be a StaticTebRequest")
    grid = build_grid_spec(dict(request.base_config))
    choices: list[tuple[float, int, PlannedTebRoute]] = []
    diagnostics: list[TebCandidateDiagnostic] = []
    for index, initialization_id in enumerate(request.planner_config.initialization_ids):
        side = {
            "straight": _straight_escape_side(request),
            "bypass_left": 1.0,
            "bypass_right": -1.0,
        }[initialization_id]
        (
            route,
            rejection_reason,
            cost,
            optimization_iterations,
            accepted_cost_terms,
        ) = _rollout_candidate(
            request,
            initialization_id=initialization_id,
            side=side,
            grid=grid,
            clearance_band_occluders=clearance_band_occluders,
        )
        cost_terms = (
            accepted_cost_terms
            if accepted_cost_terms is not None
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
                optimization_iterations=optimization_iterations,
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
    if "teb_static_collision" in reasons:
        rejection = "teb_static_collision"
    elif "teb_dynamics_limit" in reasons:
        rejection = "teb_dynamics_limit"
    else:
        rejection = "teb_goal_unreached"
    return LightweightTebResult(
        planner_version=request.planner_config.version,
        route=None,
        goal_world_pose=_readonly(request.local_goal_world_pose, dtype=ARRAY_DTYPE),
        diagnostics=TebDiagnostics(candidates=tuple(diagnostics)),
        rejection_reason=rejection,
    )


def plan_static_lightweight_teb(
    request: StaticTebRequest,
) -> LightweightTebResult:
    """Plan one static-only full route without decision-relative products."""

    if not isinstance(request, StaticTebRequest):
        raise TypeError("request must be a StaticTebRequest")
    return _plan_lightweight_teb(
        request,
        clearance_band_occluders=_directly_relevant_occluders(
            request,
            request.occluders,
        ),
    )


plan_lightweight_teb = plan_static_lightweight_teb


def plan_observed_lightweight_teb(
    request: ObservedTebRequest,
) -> LightweightTebResult:
    """Replan with constant-velocity sweeps derived only from observed states."""

    if not isinstance(request, ObservedTebRequest):
        raise TypeError("request must be an ObservedTebRequest")
    dynamic_components: list[CircleOccluder] = []
    dt_s = request.planner_config.route_sample_dt_s
    sample_count = int(
        round(request.planner_config.maximum_route_time_s / dt_s)
    )
    start_xy = request.start_pose[:2].astype(np.float64)
    goal_xy = request.local_goal_world_pose[:2].astype(np.float64)
    direction = goal_xy - start_xy
    length = float(np.linalg.norm(direction))
    normal = (
        np.asarray([-direction[1], direction[0]], dtype=np.float64) / length
        if length > 1e-9
        else np.asarray([0.0, 1.0], dtype=np.float64)
    )
    for obstacle in request.observed_dynamic_obstacles:
        current_xy = obstacle.observed_pose[:2].astype(np.float64) + (
            obstacle.observation_age_s
            * obstacle.observed_velocity_xy.astype(np.float64)
        )
        sign = 1.0 if _sha_side(obstacle.object_id) else -1.0
        for index in range(sample_count + 1):
            center = (
                current_xy
                + index * dt_s * obstacle.observed_velocity_xy.astype(np.float64)
                + sign * 1e-4 * normal
            )
            dynamic_components.append(
                CircleOccluder(
                    occluder_id=f"observed-{obstacle.object_id}-{index:02d}",
                    semantic_type="observed_dynamic_prediction",
                    center_xy=np.asarray(center, dtype=ARRAY_DTYPE),
                    radius_m=obstacle.footprint_radius_m,
                )
            )
    static_request = StaticTebRequest(
        start_pose=request.start_pose,
        initial_control=request.initial_control,
        local_goal_world_pose=request.local_goal_world_pose,
        static_occupancy=request.static_occupancy,
        occluders=tuple(request.occluders) + tuple(dynamic_components),
        base_config=request.base_config,
        planner_config=request.planner_config,
    )
    return _plan_lightweight_teb(
        static_request,
        clearance_band_occluders=_directly_relevant_occluders(
            static_request,
            request.occluders,
        ),
    )


def _sha_side(value: str) -> bool:
    encoded = value.encode("utf-8")
    return bool(sum((index + 1) * byte for index, byte in enumerate(encoded)) % 2)


__all__ = (
    "LightweightTebResult",
    "ObservedDynamicObstacle",
    "ObservedTebRequest",
    "PlannedTebRoute",
    "StaticTebRequest",
    "TebCandidateDiagnostic",
    "TebDiagnostics",
    "plan_lightweight_teb",
    "plan_observed_lightweight_teb",
    "plan_static_lightweight_teb",
)
