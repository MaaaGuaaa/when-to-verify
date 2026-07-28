"""Causal response selection after a verification action reveals an actor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from src.contracts import ARRAY_DTYPE, GridSpec
from src.geometry import (
    CircleFootprint,
    Footprint,
    RectangleFootprint,
    rasterize_footprint,
    signed_clearance,
    wrap_angle,
)
from src.generation.counterfactual_verify import (
    CounterfactualObservationTrace,
    interpolate_dynamic_pose,
)
from src.generation.event_contracts import footprint_from_spec
from src.planning.verification_actions import ActionTrace
from src.planning.verification_responses import (
    VerificationPolicyBranch,
    build_completed_action_branch,
    build_observe_and_replan_branch,
    build_reactive_braking_branch,
)


VERIFICATION_RESPONSE_POLICY_VERSION = "verification_response_policy_v1"


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _motion_radius(footprint: Footprint) -> float:
    if isinstance(footprint, CircleFootprint):
        return float(footprint.radius_m)
    if isinstance(footprint, RectangleFootprint):
        return 0.5 * float(np.hypot(footprint.length_m, footprint.width_m))
    raise TypeError("unsupported footprint type")


def _interpolate_pose(
    start: np.ndarray,
    end: np.ndarray,
    fraction: float,
) -> np.ndarray:
    result = (1.0 - fraction) * start + fraction * end
    result = np.asarray(result, dtype=np.float64)
    result[2] = wrap_angle(
        start[2] + fraction * float(wrap_angle(end[2] - start[2]))
    )
    return result


@dataclass(frozen=True)
class VerificationResponseDecision:
    """Observable geometric evidence behind one response branch."""

    branch_kind: str
    observation_time_s: float | None
    predicted_ttc_s: float | None
    stopping_time_s: float | None
    braking_threshold_s: float | None
    brake_trace_collision_free: bool | None
    brake_trace_failure_reason: str | None

    def __post_init__(self) -> None:
        supported = {"complete", "observe_and_replan", "emergency_brake"}
        if self.branch_kind not in supported:
            raise ValueError("unsupported verification response branch")
        scalar_names = (
            "observation_time_s",
            "predicted_ttc_s",
            "stopping_time_s",
            "braking_threshold_s",
        )
        for name in scalar_names:
            value = getattr(self, name)
            if value is None:
                continue
            normalized = _finite_real(value, name=name)
            if normalized < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, normalized)
        if self.branch_kind == "complete":
            if any(
                getattr(self, name) is not None
                for name in (
                    *scalar_names,
                    "brake_trace_collision_free",
                    "brake_trace_failure_reason",
                )
            ):
                raise ValueError("complete response must not contain trigger evidence")
            return
        if (
            self.observation_time_s is None
            or self.stopping_time_s is None
            or self.braking_threshold_s is None
        ):
            raise ValueError("interrupted response requires timing evidence")
        if self.branch_kind == "observe_and_replan":
            if (
                self.brake_trace_collision_free is not None
                or self.brake_trace_failure_reason is not None
            ):
                raise ValueError(
                    "observe-and-replan response must not contain brake evidence"
                )
            return
        if self.predicted_ttc_s is None or not isinstance(
            self.brake_trace_collision_free, bool
        ):
            raise ValueError("emergency response requires TTC and brake evidence")
        if self.brake_trace_collision_free:
            if self.brake_trace_failure_reason is not None:
                raise ValueError("safe brake trace must not have a failure reason")
        elif self.brake_trace_failure_reason not in {
            "static_collision",
            "dynamic_collision",
        }:
            raise ValueError("unsafe brake trace requires a supported failure reason")


@dataclass(frozen=True)
class VerificationResponseResolution:
    version: str
    decision: VerificationResponseDecision
    branch: VerificationPolicyBranch

    def __post_init__(self) -> None:
        if self.version != VERIFICATION_RESPONSE_POLICY_VERSION:
            raise ValueError("unsupported verification response policy version")
        if not isinstance(self.decision, VerificationResponseDecision):
            raise TypeError("decision must be a VerificationResponseDecision")
        if not isinstance(self.branch, VerificationPolicyBranch):
            raise TypeError("branch must be a VerificationPolicyBranch")
        if self.decision.branch_kind != self.branch.branch_kind:
            raise ValueError("response decision and branch kinds must align")


@dataclass(frozen=True)
class _ObservedActor:
    pose: np.ndarray
    velocity_xy_mps: np.ndarray
    yaw_rate_radps: float
    footprint: Footprint


@dataclass(frozen=True)
class _ActorAssessment:
    actor: _ObservedActor
    predicted_ttc_s: float | None
    route_relevant: bool


def _project_actor_pose(actor: _ObservedActor, elapsed_s: float) -> np.ndarray:
    return np.asarray(
        [
            actor.pose[0] + actor.velocity_xy_mps[0] * elapsed_s,
            actor.pose[1] + actor.velocity_xy_mps[1] * elapsed_s,
            wrap_angle(actor.pose[2] + actor.yaw_rate_radps * elapsed_s),
        ],
        dtype=np.float64,
    )


def _observed_actor(
    *,
    object_id: str,
    time_s: float,
    dynamic_current_poses: Mapping[str, np.ndarray],
    dynamic_future_poses: Mapping[str, np.ndarray],
    footprints: Mapping[str, Footprint],
    future_dt_s: float,
    future_steps: int,
) -> _ObservedActor:
    """Materialize the actor state measurable at ``time_s``.

    Velocity is a backward finite difference ending at the observation time;
    no pose after the observation enters the response decision.
    """

    pose = interpolate_dynamic_pose(
        dynamic_current_poses[object_id],
        dynamic_future_poses[object_id],
        time_s=time_s,
        future_dt_s=future_dt_s,
        object_id=object_id,
        future_steps=future_steps,
    )
    history_time_s = max(0.0, time_s - future_dt_s)
    history = interpolate_dynamic_pose(
        dynamic_current_poses[object_id],
        dynamic_future_poses[object_id],
        time_s=history_time_s,
        future_dt_s=future_dt_s,
        object_id=object_id,
        future_steps=future_steps,
    )
    window_s = time_s - history_time_s
    if window_s <= 0.0:
        velocity = np.zeros(2, dtype=np.float64)
        yaw_rate = 0.0
    else:
        velocity = (pose[:2] - history[:2]) / window_s
        yaw_rate = float(wrap_angle(pose[2] - history[2]) / window_s)
    return _ObservedActor(
        pose=np.asarray(pose, dtype=np.float64),
        velocity_xy_mps=np.asarray(velocity, dtype=np.float64),
        yaw_rate_radps=yaw_rate,
        footprint=footprints[object_id],
    )


def _first_action_collision_ttc(
    *,
    action_trace: ActionTrace,
    start_index: int,
    actor: _ObservedActor,
    robot_footprint: Footprint,
    spatial_resolution_m: float,
) -> float | None:
    start_time_s = float(action_trace.times_s[start_index])
    robot_radius = _motion_radius(robot_footprint)
    actor_radius = _motion_radius(actor.footprint)
    first_robot = action_trace.poses[start_index].astype(np.float64)
    if signed_clearance(
        robot_footprint,
        first_robot,
        actor.footprint,
        actor.pose,
    ) <= 0.0:
        return 0.0
    maximum_step_m = 0.5 * spatial_resolution_m
    for index in range(start_index, action_trace.times_s.size - 1):
        interval_start_s = float(action_trace.times_s[index])
        interval_end_s = float(action_trace.times_s[index + 1])
        dt_s = interval_end_s - interval_start_s
        robot_start = action_trace.poses[index].astype(np.float64)
        robot_end = action_trace.poses[index + 1].astype(np.float64)
        actor_elapsed_start_s = interval_start_s - start_time_s
        robot_motion = float(np.linalg.norm(robot_end[:2] - robot_start[:2]))
        robot_motion += robot_radius * abs(
            float(wrap_angle(robot_end[2] - robot_start[2]))
        )
        actor_motion = float(np.linalg.norm(actor.velocity_xy_mps)) * dt_s
        actor_motion += actor_radius * abs(actor.yaw_rate_radps) * dt_s
        subdivisions = max(
            1,
            int(np.ceil((robot_motion + actor_motion) / maximum_step_m)),
        )
        for subdivision in range(1, subdivisions + 1):
            fraction = subdivision / subdivisions
            robot_pose = _interpolate_pose(robot_start, robot_end, fraction)
            elapsed_s = actor_elapsed_start_s + fraction * dt_s
            actor_pose = _project_actor_pose(actor, elapsed_s)
            if signed_clearance(
                robot_footprint,
                robot_pose,
                actor.footprint,
                actor_pose,
            ) <= 0.0:
                return float(interval_start_s + fraction * dt_s - start_time_s)
    return None


def _actor_intersects_route(
    *,
    actor: _ObservedActor,
    route_corridor_mask: np.ndarray,
    grid: GridSpec,
    prediction_horizon_s: float,
    future_dt_s: float,
) -> bool:
    sample_times = np.arange(
        0.0,
        prediction_horizon_s + 0.5 * future_dt_s,
        future_dt_s,
        dtype=np.float64,
    )
    if sample_times.size == 0 or sample_times[-1] < prediction_horizon_s:
        sample_times = np.append(sample_times, prediction_horizon_s)
    for elapsed_s in sample_times:
        occupied = rasterize_footprint(
            actor.footprint,
            _project_actor_pose(actor, float(elapsed_s)),
            grid,
        )
        if np.any(occupied & route_corridor_mask):
            return True
    return False


def _visible_actors(
    *,
    frame_visible_mask: np.ndarray,
    time_s: float,
    dynamic_current_poses: Mapping[str, np.ndarray],
    dynamic_future_poses: Mapping[str, np.ndarray],
    footprints: Mapping[str, Footprint],
    grid: GridSpec,
    future_dt_s: float,
) -> dict[str, _ObservedActor]:
    result: dict[str, _ObservedActor] = {}
    for object_id in sorted(dynamic_current_poses):
        actor = _observed_actor(
            object_id=object_id,
            time_s=time_s,
            dynamic_current_poses=dynamic_current_poses,
            dynamic_future_poses=dynamic_future_poses,
            footprints=footprints,
            future_dt_s=future_dt_s,
            future_steps=grid.future_steps,
        )
        occupied = rasterize_footprint(actor.footprint, actor.pose, grid)
        if np.any(occupied & frame_visible_mask):
            result[object_id] = actor
    return result


def _brake_trace_safety(
    *,
    branch: VerificationPolicyBranch,
    observation_time_s: float,
    actors: tuple[_ObservedActor, ...],
    robot_footprint: Footprint,
    static_occupancy: np.ndarray,
    grid: GridSpec,
) -> tuple[bool, str | None]:
    trace = branch.executed_trace
    start_index = max(
        0,
        int(np.searchsorted(trace.times_s, observation_time_s, side="left")),
    )
    robot_radius = _motion_radius(robot_footprint)
    actor_radii = tuple(_motion_radius(actor.footprint) for actor in actors)
    maximum_step_m = 0.5 * float(grid.resolution_m)

    def collision_reason(robot_pose: np.ndarray, absolute_time_s: float) -> str | None:
        robot_mask = rasterize_footprint(robot_footprint, robot_pose, grid)
        if np.any(static_occupancy[robot_mask] > 0.5):
            return "static_collision"
        elapsed_s = absolute_time_s - observation_time_s
        for actor in actors:
            if signed_clearance(
                robot_footprint,
                robot_pose,
                actor.footprint,
                _project_actor_pose(actor, elapsed_s),
            ) <= 0.0:
                return "dynamic_collision"
        return None

    initial_reason = collision_reason(
        trace.poses[start_index].astype(np.float64),
        float(trace.times_s[start_index]),
    )
    if initial_reason is not None:
        return False, initial_reason
    for index in range(start_index, trace.times_s.size - 1):
        start_time_s = float(trace.times_s[index])
        end_time_s = float(trace.times_s[index + 1])
        dt_s = end_time_s - start_time_s
        robot_start = trace.poses[index].astype(np.float64)
        robot_end = trace.poses[index + 1].astype(np.float64)
        robot_motion = float(np.linalg.norm(robot_end[:2] - robot_start[:2]))
        robot_motion += robot_radius * abs(
            float(wrap_angle(robot_end[2] - robot_start[2]))
        )
        actor_motion = max(
            (
                float(np.linalg.norm(actor.velocity_xy_mps)) * dt_s
                + radius * abs(actor.yaw_rate_radps) * dt_s
                for actor, radius in zip(actors, actor_radii, strict=True)
            ),
            default=0.0,
        )
        subdivisions = max(
            1,
            int(np.ceil((robot_motion + actor_motion) / maximum_step_m)),
        )
        for subdivision in range(1, subdivisions + 1):
            fraction = subdivision / subdivisions
            robot_pose = _interpolate_pose(robot_start, robot_end, fraction)
            absolute_time_s = start_time_s + fraction * dt_s
            reason = collision_reason(robot_pose, absolute_time_s)
            if reason is not None:
                return False, reason
    return True, None


def resolve_verification_response(
    *,
    action_trace: ActionTrace,
    observation_trace: CounterfactualObservationTrace,
    robot_footprint: Footprint,
    static_occupancy: np.ndarray,
    dynamic_current_poses: Mapping[str, np.ndarray],
    dynamic_future_poses: Mapping[str, np.ndarray],
    dynamic_specs: Mapping[str, dict[str, object]],
    current_visible_mask: np.ndarray,
    route_corridor_mask: np.ndarray,
    grid: GridSpec,
    future_dt_s: float,
    future_horizon_s: float,
    braking_deceleration_mps2: float,
    angular_deceleration_radps2: float,
    braking_margin_s: float,
) -> VerificationResponseResolution:
    """Choose complete, replan, or brake using only newly observed actor state."""

    if not isinstance(action_trace, ActionTrace):
        raise TypeError("action_trace must be an ActionTrace")
    if not isinstance(observation_trace, CounterfactualObservationTrace):
        raise TypeError("observation_trace must be a CounterfactualObservationTrace")
    if not isinstance(robot_footprint, (CircleFootprint, RectangleFootprint)):
        raise TypeError("robot_footprint must be a supported typed footprint")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    if not np.array_equal(observation_trace.times_s, action_trace.times_s):
        raise ValueError("observation and action trace times must match exactly")
    shape = (grid.height, grid.width)
    if (
        not isinstance(static_occupancy, np.ndarray)
        or static_occupancy.shape != shape
        or static_occupancy.dtype != ARRAY_DTYPE
        or not np.isfinite(static_occupancy).all()
        or not np.isin(static_occupancy, (0.0, 1.0)).all()
    ):
        raise ValueError("static_occupancy must be binary finite float32")
    if (
        not isinstance(current_visible_mask, np.ndarray)
        or current_visible_mask.shape != shape
        or current_visible_mask.dtype != np.bool_
    ):
        raise ValueError("current_visible_mask must be a bool grid")
    if (
        not isinstance(route_corridor_mask, np.ndarray)
        or route_corridor_mask.shape != shape
        or route_corridor_mask.dtype.kind not in "biuf"
        or not np.isfinite(route_corridor_mask).all()
        or not np.isin(route_corridor_mask, (0, 1)).all()
    ):
        raise ValueError("route_corridor_mask must be a binary grid")
    route = np.asarray(route_corridor_mask != 0, dtype=bool)
    for name, value in (
        ("dynamic_current_poses", dynamic_current_poses),
        ("dynamic_future_poses", dynamic_future_poses),
        ("dynamic_specs", dynamic_specs),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
    object_ids = set(dynamic_current_poses)
    if object_ids != set(dynamic_future_poses) or object_ids != set(dynamic_specs):
        raise ValueError("dynamic current/future/spec IDs must align")
    footprints: dict[str, Footprint] = {}
    for object_id in sorted(object_ids):
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("dynamic object IDs must be non-empty strings")
        current = dynamic_current_poses[object_id]
        future = dynamic_future_poses[object_id]
        if (
            not isinstance(current, np.ndarray)
            or current.shape != (3,)
            or current.dtype != ARRAY_DTYPE
            or not np.isfinite(current).all()
        ):
            raise ValueError("dynamic current poses must be finite float32 [3]")
        if (
            not isinstance(future, np.ndarray)
            or future.shape != (grid.future_steps, 3)
            or future.dtype != ARRAY_DTYPE
            or not np.isfinite(future).all()
        ):
            raise ValueError(
                "dynamic future poses must be finite float32 [future_steps,3]"
            )
        footprints[object_id] = footprint_from_spec(dynamic_specs[object_id])

    dt_s = _finite_real(future_dt_s, name="future_dt_s")
    horizon_s = _finite_real(future_horizon_s, name="future_horizon_s")
    linear_deceleration = _finite_real(
        braking_deceleration_mps2,
        name="braking_deceleration_mps2",
    )
    angular_deceleration = _finite_real(
        angular_deceleration_radps2,
        name="angular_deceleration_radps2",
    )
    margin_s = _finite_real(braking_margin_s, name="braking_margin_s")
    if (
        dt_s <= 0.0
        or horizon_s <= 0.0
        or linear_deceleration <= 0.0
        or angular_deceleration <= 0.0
        or margin_s < 0.0
    ):
        raise ValueError(
            "horizon, time step, and decelerations must be positive; "
            "braking margin must be non-negative"
        )
    if action_trace.times_s[-1] > horizon_s + 1e-12:
        raise ValueError("verification action trace exceeds the future horizon")

    initially_hidden: set[str] = set()
    for object_id in sorted(object_ids):
        current_occupied = rasterize_footprint(
            footprints[object_id],
            dynamic_current_poses[object_id],
            grid,
        )
        if not np.any(current_occupied & current_visible_mask):
            initially_hidden.add(object_id)

    for frame_index in range(1, action_trace.times_s.size):
        observation_time_s = float(action_trace.times_s[frame_index])
        visible = _visible_actors(
            frame_visible_mask=observation_trace.frames[frame_index].visible_mask,
            time_s=observation_time_s,
            dynamic_current_poses=dynamic_current_poses,
            dynamic_future_poses=dynamic_future_poses,
            footprints=footprints,
            grid=grid,
            future_dt_s=dt_s,
        )
        newly_observed = {
            object_id: actor
            for object_id, actor in visible.items()
            if object_id in initially_hidden
        }
        if not newly_observed:
            continue
        linear = abs(float(action_trace.linear_velocities_mps[frame_index]))
        stopping_time_s = linear / linear_deceleration
        braking_threshold_s = stopping_time_s + margin_s
        assessments: list[_ActorAssessment] = []
        prediction_horizon_s = max(0.0, horizon_s - observation_time_s)
        for actor in newly_observed.values():
            ttc_s = _first_action_collision_ttc(
                action_trace=action_trace,
                start_index=frame_index,
                actor=actor,
                robot_footprint=robot_footprint,
                spatial_resolution_m=float(grid.resolution_m),
            )
            route_relevant = ttc_s is not None or _actor_intersects_route(
                actor=actor,
                route_corridor_mask=route,
                grid=grid,
                prediction_horizon_s=prediction_horizon_s,
                future_dt_s=dt_s,
            )
            assessments.append(
                _ActorAssessment(
                    actor=actor,
                    predicted_ttc_s=ttc_s,
                    route_relevant=route_relevant,
                )
            )
        emergency = [
            item
            for item in assessments
            if item.predicted_ttc_s is not None
            and item.predicted_ttc_s <= braking_threshold_s + 1e-12
        ]
        if emergency:
            selected = min(
                emergency,
                key=lambda item: float(item.predicted_ttc_s),
            )
            branch = build_reactive_braking_branch(
                action_trace=action_trace,
                response_time_s=observation_time_s,
                braking_deceleration_mps2=linear_deceleration,
                angular_deceleration_radps2=angular_deceleration,
                future_horizon_s=horizon_s,
            )
            collision_free, failure_reason = _brake_trace_safety(
                branch=branch,
                observation_time_s=observation_time_s,
                actors=tuple(visible.values()),
                robot_footprint=robot_footprint,
                static_occupancy=static_occupancy,
                grid=grid,
            )
            decision = VerificationResponseDecision(
                branch_kind="emergency_brake",
                observation_time_s=observation_time_s,
                predicted_ttc_s=selected.predicted_ttc_s,
                stopping_time_s=stopping_time_s,
                braking_threshold_s=braking_threshold_s,
                brake_trace_collision_free=collision_free,
                brake_trace_failure_reason=failure_reason,
            )
            return VerificationResponseResolution(
                version=VERIFICATION_RESPONSE_POLICY_VERSION,
                decision=decision,
                branch=branch,
            )
        relevant = [item for item in assessments if item.route_relevant]
        if relevant:
            selected = min(
                relevant,
                key=lambda item: (
                    float("inf")
                    if item.predicted_ttc_s is None
                    else item.predicted_ttc_s
                ),
            )
            branch = build_observe_and_replan_branch(
                action_trace=action_trace,
                response_time_s=observation_time_s,
            )
            decision = VerificationResponseDecision(
                branch_kind="observe_and_replan",
                observation_time_s=observation_time_s,
                predicted_ttc_s=selected.predicted_ttc_s,
                stopping_time_s=stopping_time_s,
                braking_threshold_s=braking_threshold_s,
                brake_trace_collision_free=None,
                brake_trace_failure_reason=None,
            )
            return VerificationResponseResolution(
                version=VERIFICATION_RESPONSE_POLICY_VERSION,
                decision=decision,
                branch=branch,
            )

    branch = build_completed_action_branch(action_trace)
    return VerificationResponseResolution(
        version=VERIFICATION_RESPONSE_POLICY_VERSION,
        decision=VerificationResponseDecision(
            branch_kind="complete",
            observation_time_s=None,
            predicted_ttc_s=None,
            stopping_time_s=None,
            braking_threshold_s=None,
            brake_trace_collision_free=None,
            brake_trace_failure_reason=None,
        ),
        branch=branch,
    )


__all__ = (
    "VERIFICATION_RESPONSE_POLICY_VERSION",
    "VerificationResponseDecision",
    "VerificationResponseResolution",
    "resolve_verification_response",
)
