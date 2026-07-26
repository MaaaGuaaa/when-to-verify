"""Label-side active-revealability audit for SOP05R mother events."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np

from src.contracts import (
    ARRAY_DTYPE,
    BaseState,
    OracleContext,
    build_grid_spec,
    validate_dynamic_object_spec,
)
from src.geometry import (
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint,
    raycast_candidate_visibility,
    trajectory_signed_clearances,
    wrap_angle,
)
from src.planning.obstacle_corner_planner import (
    ObstaclePlanResult,
    ObstaclePlannerRequest,
    ObstaclePlannedRoute,
    plan_obstacle_routes,
)
from src.planning.verification_actions import (
    CANONICAL_ACTION_IDS,
    ActionFeasibility,
    ActionTrace,
    VerificationAction,
    VerificationActionLibrary,
    check_action_trace_feasibility,
    sample_state_aware_action_trace,
)
from src.utils.config import validate_config

from .dynamic_object_transplant import footprint_from_spec
from .obstacle_first_templates import RectangleObstacle
from .sop05r_contracts import (
    SOP05R_ACTIVE_REVEALABILITY_VERSION,
    Sop05rConfig,
    normalize_sop05r_config,
)
from .sop05r_event_sampler import Sop05rMotherCandidate


Planner = Callable[[ObstaclePlannerRequest], ObstaclePlanResult]


def _readonly_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite {dtype.name} with shape {shape}")
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _immutable_pose_mapping(
    value: Mapping[str, np.ndarray],
    *,
    name: str,
    shape: tuple[int, ...],
) -> Mapping[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, np.ndarray] = {}
    for object_id in sorted(value):
        if not isinstance(object_id, str) or not object_id:
            raise ValueError(f"{name} keys must be non-empty strings")
        result[object_id] = _readonly_array(
            value[object_id],
            name=f"{name}[{object_id!r}]",
            dtype=np.dtype(ARRAY_DTYPE),
            shape=shape,
        )
    return MappingProxyType(result)


@dataclass(frozen=True)
class Sop05rRevealabilityRequest:
    event_id: str
    robot_pose: np.ndarray
    robot_state: np.ndarray
    static_occupancy: np.ndarray
    obstacle: RectangleObstacle
    local_goal_world_pose: np.ndarray
    conflict_point: np.ndarray
    conflict_target_pose: np.ndarray
    conflict_time_s: float
    target_object_id: str
    dynamic_current_poses: Mapping[str, np.ndarray]
    dynamic_future_poses: Mapping[str, np.ndarray]
    dynamic_specs: Mapping[str, Mapping[str, object]]
    base_config: Mapping[str, Any]
    config: Sop05rConfig
    action_library: VerificationActionLibrary
    sensor_range_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("event_id must be a non-empty string")
        if not isinstance(self.target_object_id, str) or not self.target_object_id:
            raise ValueError("target_object_id must be a non-empty string")
        if not isinstance(self.obstacle, RectangleObstacle):
            raise TypeError("obstacle must be a RectangleObstacle")
        if not isinstance(self.config, Sop05rConfig):
            raise TypeError("config must be a Sop05rConfig")
        if not isinstance(self.action_library, VerificationActionLibrary):
            raise TypeError("action_library must be a VerificationActionLibrary")
        action_ids = tuple(action.action_id for action in self.action_library.actions)
        if action_ids != CANONICAL_ACTION_IDS:
            raise ValueError("action_library must preserve canonical action order")
        base_config = deepcopy(dict(self.base_config))
        validate_config(base_config)
        grid = build_grid_spec(base_config)
        if self.config.planner.rollout_steps != grid.future_steps:
            raise ValueError("SOP05R planner and base future steps differ")
        if self.config.planner.dt_s != float(base_config["bev"]["future_dt_s"]):
            raise ValueError("SOP05R planner and base future dt differ")

        current = _immutable_pose_mapping(
            self.dynamic_current_poses,
            name="dynamic_current_poses",
            shape=(3,),
        )
        future = _immutable_pose_mapping(
            self.dynamic_future_poses,
            name="dynamic_future_poses",
            shape=(grid.future_steps, 3),
        )
        if set(current) != set(future) or set(current) != set(self.dynamic_specs):
            raise ValueError("dynamic current, future, and spec IDs must align")
        if self.target_object_id not in current:
            raise ValueError("target_object_id must identify a dynamic object")
        specs: dict[str, Mapping[str, object]] = {}
        for object_id in sorted(self.dynamic_specs):
            spec = deepcopy(dict(self.dynamic_specs[object_id]))
            validate_dynamic_object_spec(spec)
            specs[object_id] = MappingProxyType(spec)

        occupancy = _readonly_array(
            self.static_occupancy,
            name="static_occupancy",
            dtype=np.dtype(ARRAY_DTYPE),
            shape=(grid.height, grid.width),
        )
        if not np.isin(occupancy, (0.0, 1.0)).all():
            raise ValueError("static_occupancy must be binary")
        conflict_time = _finite_positive(self.conflict_time_s, name="conflict_time_s")
        future_horizon_s = grid.future_steps * float(base_config["bev"]["future_dt_s"])
        if conflict_time > future_horizon_s + 1e-10:
            raise ValueError("conflict_time_s exceeds the oracle future horizon")

        object.__setattr__(self, "base_config", MappingProxyType(base_config))
        object.__setattr__(self, "dynamic_current_poses", current)
        object.__setattr__(self, "dynamic_future_poses", future)
        object.__setattr__(self, "dynamic_specs", MappingProxyType(specs))
        object.__setattr__(self, "static_occupancy", occupancy)
        object.__setattr__(
            self,
            "robot_pose",
            _readonly_array(
                self.robot_pose,
                name="robot_pose",
                dtype=np.dtype(ARRAY_DTYPE),
                shape=(3,),
            ),
        )
        object.__setattr__(
            self,
            "robot_state",
            _readonly_array(
                self.robot_state,
                name="robot_state",
                dtype=np.dtype(ARRAY_DTYPE),
                shape=(2,),
            ),
        )
        object.__setattr__(
            self,
            "local_goal_world_pose",
            _readonly_array(
                self.local_goal_world_pose,
                name="local_goal_world_pose",
                dtype=np.dtype(ARRAY_DTYPE),
                shape=(3,),
            ),
        )
        object.__setattr__(
            self,
            "conflict_point",
            _readonly_array(
                np.asarray(self.conflict_point, dtype=np.float64),
                name="conflict_point",
                dtype=np.dtype(np.float64),
                shape=(2,),
            ),
        )
        object.__setattr__(
            self,
            "conflict_target_pose",
            _readonly_array(
                np.asarray(self.conflict_target_pose, dtype=np.float64),
                name="conflict_target_pose",
                dtype=np.dtype(np.float64),
                shape=(3,),
            ),
        )
        object.__setattr__(self, "conflict_time_s", conflict_time)
        object.__setattr__(
            self,
            "sensor_range_m",
            _finite_positive(self.sensor_range_m, name="sensor_range_m"),
        )


@dataclass(frozen=True)
class ActionRevealabilityEvidence:
    action_id: str
    is_stop: bool
    action_trace: ActionTrace
    matched_wait_trace: ActionTrace
    action_feasibility: ActionFeasibility
    matched_wait_feasibility: ActionFeasibility
    observation_horizon_s: float
    first_visible_time_s: float | None
    matched_wait_visible_time_s: float | None
    visibility_lead_s: float | None
    visibility_lead_lower_bound_s: float | None
    visibility_lead_is_censored: bool
    post_visibility_margin_s: float | None
    replan_goal_world_pose: np.ndarray | None
    post_action_route_ids: tuple[str, ...]
    post_action_avoiding_route_ids: tuple[str, ...]
    gate_conditions_met: bool
    active_revealable: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.active_revealable and (self.is_stop or not self.gate_conditions_met):
            raise ValueError("only a non-stop row meeting every gate may be active")
        if self.replan_goal_world_pose is not None:
            object.__setattr__(
                self,
                "replan_goal_world_pose",
                _readonly_array(
                    self.replan_goal_world_pose,
                    name="replan_goal_world_pose",
                    dtype=np.dtype(ARRAY_DTYPE),
                    shape=(3,),
                ),
            )

    @property
    def post_action_avoids_original_conflict(self) -> bool:
        return bool(self.post_action_avoiding_route_ids)

    def as_metadata(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "is_stop": self.is_stop,
            "action_feasible": self.action_feasibility.feasible,
            "action_feasibility_reason": self.action_feasibility.reason,
            "matched_wait_feasible": self.matched_wait_feasibility.feasible,
            "matched_wait_feasibility_reason": self.matched_wait_feasibility.reason,
            "action_duration_s": float(self.action_trace.times_s[-1]),
            "matched_wait_duration_s": float(self.matched_wait_trace.times_s[-1]),
            "observation_horizon_s": self.observation_horizon_s,
            "first_visible_time_s": self.first_visible_time_s,
            "matched_wait_visible_time_s": self.matched_wait_visible_time_s,
            "visibility_lead_s": self.visibility_lead_s,
            "visibility_lead_lower_bound_s": self.visibility_lead_lower_bound_s,
            "visibility_lead_is_censored": self.visibility_lead_is_censored,
            "post_visibility_margin_s": self.post_visibility_margin_s,
            "post_action_pose": [float(value) for value in self.action_trace.poses[-1]],
            "replan_goal_world_pose": (
                None
                if self.replan_goal_world_pose is None
                else [float(value) for value in self.replan_goal_world_pose]
            ),
            "post_action_route_ids": list(self.post_action_route_ids),
            "post_action_avoiding_route_ids": list(
                self.post_action_avoiding_route_ids
            ),
            "post_action_avoids_original_conflict": (
                self.post_action_avoids_original_conflict
            ),
            "gate_conditions_met": self.gate_conditions_met,
            "active_revealable": self.active_revealable,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class ActiveRevealabilityAudit:
    version: str
    event_id: str
    actions: tuple[ActionRevealabilityEvidence, ...]
    active_revealable_action_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.version != SOP05R_ACTIVE_REVEALABILITY_VERSION:
            raise ValueError("active revealability version mismatch")
        if tuple(row.action_id for row in self.actions) != CANONICAL_ACTION_IDS:
            raise ValueError("revealability rows must preserve canonical action order")
        expected = tuple(
            row.action_id for row in self.actions if row.active_revealable
        )
        if self.active_revealable_action_ids != expected:
            raise ValueError("active action summary differs from action evidence")

    @property
    def by_action(self) -> Mapping[str, ActionRevealabilityEvidence]:
        return MappingProxyType({row.action_id: row for row in self.actions})

    @property
    def active_revealable(self) -> bool:
        return bool(self.active_revealable_action_ids)

    def as_metadata(self) -> dict[str, object]:
        return {
            "active_revealability_version": self.version,
            "active_revealability_status": (
                "active_revealable" if self.active_revealable else "natural_difficult"
            ),
            "active_revealable_action_ids": list(
                self.active_revealable_action_ids
            ),
            "first_visible_time_by_verification_action": {
                row.action_id: row.first_visible_time_s for row in self.actions
            },
            "matched_wait_visible_time": {
                row.action_id: row.matched_wait_visible_time_s for row in self.actions
            },
            "active_revealability_actions": {
                row.action_id: row.as_metadata() for row in self.actions
            },
        }


def _robot_footprint(base_config: Mapping[str, Any]) -> RectangleFootprint:
    robot = base_config["robot"]
    return inflate_footprint(
        RectangleFootprint(float(robot["length_m"]), float(robot["width_m"])),
        float(robot["inflation_m"]),
    )


def _matched_wait_action(action: VerificationAction) -> VerificationAction:
    return VerificationAction(
        action_id=f"matched_wait::{action.action_id}",
        duration_s=action.duration_s,
        delta_forward_m=0.0,
        delta_yaw_rad=0.0,
    )


def _extend_with_stationary_hold(trace: ActionTrace, *, horizon_s: float) -> ActionTrace:
    if horizon_s <= float(trace.times_s[-1]) + 1e-12:
        return trace
    step_s = 0.05
    remaining_s = horizon_s - float(trace.times_s[-1])
    intervals = max(1, int(np.ceil(remaining_s / step_s)))
    tail = np.linspace(
        float(trace.times_s[-1]), horizon_s, intervals + 1, dtype=np.float64
    )[1:]
    times = np.concatenate((trace.times_s, tail))
    extra = times.size - trace.times_s.size
    return ActionTrace(
        poses=np.concatenate(
            (trace.poses, np.tile(trace.poses[-1], (extra, 1))), axis=0
        ).astype(ARRAY_DTYPE, copy=False),
        times_s=times.astype(np.float64, copy=False),
        linear_velocities_mps=np.concatenate(
            (trace.linear_velocities_mps, np.zeros(extra, dtype=np.float64))
        ),
        angular_velocities_radps=np.concatenate(
            (trace.angular_velocities_radps, np.zeros(extra, dtype=np.float64))
        ),
    )


def _dynamic_pose_at_time(
    current_pose: np.ndarray,
    future_poses: np.ndarray,
    *,
    time_s: float,
    future_dt_s: float,
) -> np.ndarray:
    all_poses = np.vstack((current_pose, future_poses)).astype(np.float64)
    if time_s <= 0.0:
        return all_poses[0]
    interval = min(
        int(np.floor(time_s / future_dt_s)), future_poses.shape[0] - 1
    )
    lower_time = interval * future_dt_s
    fraction = min(1.0, max(0.0, (time_s - lower_time) / future_dt_s))
    start = all_poses[interval]
    end = all_poses[interval + 1]
    yaw_pair = np.unwrap(np.asarray([start[2], end[2]], dtype=np.float64))
    result = (1.0 - fraction) * start + fraction * end
    result[2] = wrap_angle((1.0 - fraction) * yaw_pair[0] + fraction * yaw_pair[1])
    return result


def _first_target_visible_time(
    request: Sop05rRevealabilityRequest,
    *,
    trace: ActionTrace,
    dynamic_footprints: Mapping[str, object],
) -> float | None:
    grid = build_grid_spec(dict(request.base_config))
    future_dt_s = float(request.base_config["bev"]["future_dt_s"])
    static = np.asarray(request.static_occupancy != 0.0, dtype=np.bool_)
    target_footprint = dynamic_footprints[request.target_object_id]
    for robot_pose, time_s in zip(trace.poses, trace.times_s, strict=True):
        time_value = float(time_s)
        occupied = static.copy()
        target_pose = _dynamic_pose_at_time(
            request.dynamic_current_poses[request.target_object_id],
            request.dynamic_future_poses[request.target_object_id],
            time_s=time_value,
            future_dt_s=future_dt_s,
        )
        for object_id in sorted(request.dynamic_current_poses):
            if object_id == request.target_object_id:
                continue
            context_pose = _dynamic_pose_at_time(
                request.dynamic_current_poses[object_id],
                request.dynamic_future_poses[object_id],
                time_s=time_value,
                future_dt_s=future_dt_s,
            )
            occupied |= rasterize_footprint(
                dynamic_footprints[object_id], context_pose, grid
            )
        target_mask = rasterize_footprint(target_footprint, target_pose, grid)
        visible = raycast_candidate_visibility(
            occupied,
            target_mask,
            grid,
            sensor_pose=robot_pose,
            fov_rad=request.action_library.sensor_fov_rad,
            max_range_m=request.sensor_range_m,
        )
        if np.any(visible):
            return time_value
    return None


def _sample_polyline(
    start_pose: np.ndarray,
    waypoints: np.ndarray,
    *,
    maximum_step_m: float,
) -> np.ndarray:
    points = np.vstack((start_pose, waypoints)).astype(np.float64)
    samples: list[np.ndarray] = [points[0].copy()]
    for start, end in zip(points[:-1], points[1:], strict=True):
        delta = end[:2] - start[:2]
        distance = float(np.linalg.norm(delta))
        count = max(1, int(np.ceil(distance / maximum_step_m)))
        yaw = float(np.arctan2(delta[1], delta[0])) if distance > 1e-12 else float(end[2])
        for index in range(1, count + 1):
            fraction = index / count
            xy = (1.0 - fraction) * start[:2] + fraction * end[:2]
            samples.append(np.asarray([xy[0], xy[1], wrap_angle(yaw)]))
    return np.asarray(samples, dtype=np.float64)


def _route_avoids_original_conflict(
    route: ObstaclePlannedRoute,
    *,
    start_pose: np.ndarray,
    request: Sop05rRevealabilityRequest,
) -> bool:
    if route.slot_id == "stop":
        return False
    grid = build_grid_spec(dict(request.base_config))
    poses = _sample_polyline(
        start_pose,
        route.waypoints_world,
        maximum_step_m=0.25 * grid.resolution_m,
    )
    target_poses = np.tile(request.conflict_target_pose, (poses.shape[0], 1))
    clearances = trajectory_signed_clearances(
        _robot_footprint(request.base_config),
        poses,
        footprint_from_spec(request.dynamic_specs[request.target_object_id]),
        target_poses,
    )
    return bool(np.min(clearances) > 0.0)


def _post_action_plan(
    request: Sop05rRevealabilityRequest,
    *,
    action_trace: ActionTrace,
    planner: Planner,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    initial_control = np.asarray(
        [
            action_trace.linear_velocities_mps[-1],
            action_trace.angular_velocities_radps[-1],
        ],
        dtype=ARRAY_DTYPE,
    )
    planner_request = ObstaclePlannerRequest(
        start_pose=action_trace.poses[-1],
        initial_control=initial_control,
        static_occupancy=request.static_occupancy,
        obstacle=request.obstacle,
        local_goal_world_pose=request.local_goal_world_pose,
        base_config=request.base_config,
        planner_config=request.config.planner,
    )
    result = planner(planner_request)
    if not isinstance(result, ObstaclePlanResult):
        raise TypeError("planner must return an ObstaclePlanResult")
    if not np.array_equal(result.shared_goal_world_pose, request.local_goal_world_pose):
        raise ValueError("post-action planner changed the world-frame goal")
    moving_routes = tuple(route for route in result.routes if route.slot_id != "stop")
    route_ids = tuple(route.trajectory.trajectory_id for route in moving_routes)
    avoiding_ids = tuple(
        route.trajectory.trajectory_id
        for route in moving_routes
        if _route_avoids_original_conflict(
            route,
            start_pose=action_trace.poses[-1],
            request=request,
        )
    )
    return result.shared_goal_world_pose, route_ids, avoiding_ids


def evaluate_active_revealability(
    request: Sop05rRevealabilityRequest,
    *,
    planner: Planner = plan_obstacle_routes,
) -> ActiveRevealabilityAudit:
    if not isinstance(request, Sop05rRevealabilityRequest):
        raise TypeError("request must be a Sop05rRevealabilityRequest")
    grid = build_grid_spec(dict(request.base_config))
    future_dt_s = float(request.base_config["bev"]["future_dt_s"])
    dynamic_poses = {
        object_id: np.vstack(
            (
                request.dynamic_current_poses[object_id],
                request.dynamic_future_poses[object_id],
            )
        ).astype(ARRAY_DTYPE)
        for object_id in request.dynamic_current_poses
    }
    dynamic_footprints = {
        object_id: footprint_from_spec(request.dynamic_specs[object_id])
        for object_id in request.dynamic_specs
    }
    rows: list[ActionRevealabilityEvidence] = []

    for action in request.action_library.actions:
        action_trace = sample_state_aware_action_trace(
            request.robot_pose,
            action,
            robot_state=request.robot_state,
            braking_deceleration_mps2=(
                request.config.planner.max_linear_acceleration_mps2
            ),
        )
        matched_wait_trace = sample_state_aware_action_trace(
            request.robot_pose,
            _matched_wait_action(action),
            robot_state=request.robot_state,
            braking_deceleration_mps2=(
                request.config.planner.max_linear_acceleration_mps2
            ),
        )
        if not np.isclose(
            action_trace.times_s[-1],
            matched_wait_trace.times_s[-1],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("action and matched wait durations differ")
        action_feasibility = check_action_trace_feasibility(
            action_trace,
            robot_footprint=_robot_footprint(request.base_config),
            static_occupancy=request.static_occupancy,
            grid=grid,
            dynamic_object_poses=dynamic_poses,
            dynamic_object_footprints=dynamic_footprints,
            dynamic_dt_s=future_dt_s,
        )
        wait_feasibility = check_action_trace_feasibility(
            matched_wait_trace,
            robot_footprint=_robot_footprint(request.base_config),
            static_occupancy=request.static_occupancy,
            grid=grid,
            dynamic_object_poses=dynamic_poses,
            dynamic_object_footprints=dynamic_footprints,
            dynamic_dt_s=future_dt_s,
        )

        observation_horizon = max(
            request.conflict_time_s,
            float(action_trace.times_s[-1]),
        )
        if observation_horizon > grid.future_steps * future_dt_s + 1e-10:
            raise ValueError("verification action exceeds the oracle future horizon")
        observed_action = _extend_with_stationary_hold(
            action_trace, horizon_s=observation_horizon
        )
        observed_wait = _extend_with_stationary_hold(
            matched_wait_trace, horizon_s=observation_horizon
        )
        first_visible = _first_target_visible_time(
            request,
            trace=observed_action,
            dynamic_footprints=dynamic_footprints,
        )
        wait_visible = _first_target_visible_time(
            request,
            trace=observed_wait,
            dynamic_footprints=dynamic_footprints,
        )
        lead: float | None = None
        lead_lower_bound: float | None = None
        lead_censored = False
        if first_visible is not None:
            if wait_visible is None:
                lead_lower_bound = observation_horizon - first_visible
                lead_censored = True
            else:
                lead = wait_visible - first_visible
                lead_lower_bound = lead
        margin = (
            None
            if first_visible is None
            else request.conflict_time_s - first_visible
        )

        replan_goal: np.ndarray | None = None
        route_ids: tuple[str, ...] = ()
        avoiding_ids: tuple[str, ...] = ()
        if action_feasibility.feasible:
            replan_goal, route_ids, avoiding_ids = _post_action_plan(
                request,
                action_trace=action_trace,
                planner=planner,
            )

        reasons: list[str] = []
        if not action_feasibility.feasible:
            reasons.append(f"action_{action_feasibility.reason}")
        if first_visible is None:
            reasons.append("target_not_revealed")
        elif (
            lead_lower_bound is None
            or lead_lower_bound
            + 1e-12
            < request.config.revealability.minimum_visibility_lead_s
        ):
            reasons.append("insufficient_visibility_lead")
        if (
            margin is None
            or margin + 1e-12
            < request.config.revealability.minimum_post_visibility_margin_s
        ):
            reasons.append("insufficient_post_visibility_margin")
        if not route_ids:
            reasons.append("post_action_same_goal_route_missing")
        if not avoiding_ids:
            reasons.append("post_action_conflict_avoidance_missing")
        conditions_met = not reasons
        is_stop = action.action_id == "stop_scan"
        rows.append(
            ActionRevealabilityEvidence(
                action_id=action.action_id,
                is_stop=is_stop,
                action_trace=action_trace,
                matched_wait_trace=matched_wait_trace,
                action_feasibility=action_feasibility,
                matched_wait_feasibility=wait_feasibility,
                observation_horizon_s=observation_horizon,
                first_visible_time_s=first_visible,
                matched_wait_visible_time_s=wait_visible,
                visibility_lead_s=lead,
                visibility_lead_lower_bound_s=lead_lower_bound,
                visibility_lead_is_censored=lead_censored,
                post_visibility_margin_s=margin,
                replan_goal_world_pose=replan_goal,
                post_action_route_ids=route_ids,
                post_action_avoiding_route_ids=avoiding_ids,
                gate_conditions_met=conditions_met,
                active_revealable=conditions_met and not is_stop,
                rejection_reasons=tuple(reasons),
            )
        )

    actions = tuple(rows)
    return ActiveRevealabilityAudit(
        version=SOP05R_ACTIVE_REVEALABILITY_VERSION,
        event_id=request.event_id,
        actions=actions,
        active_revealable_action_ids=tuple(
            row.action_id for row in actions if row.active_revealable
        ),
    )


def build_active_revealability_request(
    *,
    mother: Sop05rMotherCandidate,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    config: Sop05rConfig | Mapping[str, Any],
    action_library: VerificationActionLibrary,
) -> Sop05rRevealabilityRequest:
    if not isinstance(mother, Sop05rMotherCandidate):
        raise TypeError("mother must be a Sop05rMotherCandidate")
    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(oracle_context, OracleContext):
        raise TypeError("oracle_context must be an OracleContext")
    normalized = config if isinstance(config, Sop05rConfig) else normalize_sop05r_config(config)
    if mother.event.world.base_state_id != base_state.state_id:
        raise ValueError("mother and base state IDs differ")
    if oracle_context.base_state_id != base_state.state_id:
        raise ValueError("oracle context and base state IDs differ")

    target_id = mother.event.target.target_dynamic_object_id
    current: dict[str, np.ndarray] = {}
    for object_id in sorted(mother.event.world.dynamic_object_trajectories):
        if object_id == target_id:
            pose = mother.event.target.current_pose
        elif object_id in oracle_context.dynamic_object_history:
            pose = oracle_context.dynamic_object_history[object_id][-1]
        elif object_id in base_state.visible_dynamic_object_history:
            pose = base_state.visible_dynamic_object_history[object_id][-1]
        else:
            raise ValueError(f"no current pose source for dynamic object {object_id!r}")
        current[object_id] = np.asarray(pose, dtype=ARRAY_DTYPE)

    collision = mother.collision_evidence.continuous_evidence
    if collision.target_pose_at_first_collision is None:
        raise ValueError("mother collision evidence lacks the target conflict pose")
    grid = build_grid_spec(dict(base_config))
    sensor_range = float(
        np.hypot(
            grid.height * grid.resolution_m,
            grid.width * grid.resolution_m,
        )
    )
    return Sop05rRevealabilityRequest(
        event_id=mother.event.generated_event_id,
        robot_pose=np.asarray(base_state.robot_history[-1], dtype=ARRAY_DTYPE),
        robot_state=np.asarray(base_state.robot_state, dtype=ARRAY_DTYPE),
        static_occupancy=mother.event.world.static_occupancy,
        obstacle=mother.template.obstacle,
        local_goal_world_pose=mother.template.local_goal_world_pose,
        conflict_point=mother.collision_evidence.conflict_point,
        conflict_target_pose=collision.target_pose_at_first_collision,
        conflict_time_s=mother.collision_evidence.first_collision_time_s,
        target_object_id=target_id,
        dynamic_current_poses=current,
        dynamic_future_poses=mother.event.world.dynamic_object_trajectories,
        dynamic_specs=mother.event.world.dynamic_object_specs,
        base_config=base_config,
        config=normalized,
        action_library=action_library,
        sensor_range_m=sensor_range,
    )


__all__ = (
    "ActionRevealabilityEvidence",
    "ActiveRevealabilityAudit",
    "Sop05rRevealabilityRequest",
    "build_active_revealability_request",
    "evaluate_active_revealability",
)
