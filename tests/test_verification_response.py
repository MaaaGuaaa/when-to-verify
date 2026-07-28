from __future__ import annotations

import numpy as np
import pytest

from src.contracts import GridSpec
from src.geometry import CircleFootprint, rasterize_footprint
from src.generation.counterfactual_verify import (
    CounterfactualObservation,
    CounterfactualObservationTrace,
)
from src.generation.verification_response import resolve_verification_response
from src.planning.verification_actions import (
    ActionTrace,
    VerificationAction,
    sample_action_trace,
)


def _grid() -> GridSpec:
    return GridSpec(
        height=80,
        width=80,
        history_steps=8,
        future_steps=32,
        resolution_m=0.1,
    )


def _trace():
    return sample_action_trace(
        np.zeros(3, dtype=np.float32),
        VerificationAction(
            action_id="straight_probe",
            duration_s=2.0,
            delta_forward_m=1.0,
            delta_yaw_rad=0.0,
        ),
    )


def _observation_trace(action_trace, *, actor_pose: np.ndarray):
    grid = _grid()
    actor_mask = rasterize_footprint(CircleFootprint(0.20), actor_pose, grid)
    zero = np.zeros((grid.height, grid.width), dtype=bool)
    age = np.ones((grid.height, grid.width), dtype=np.float32)
    frames = []
    for time_s in action_trace.times_s:
        visible = actor_mask if time_s >= 0.2 - 1e-12 else zero
        frames.append(
            CounterfactualObservation(
                visible_mask=visible,
                visible_occupied_mask=visible,
                visible_dynamic_occupancy=visible,
                newly_visible_mask=visible,
                updated_age_map=age,
            )
        )
    return CounterfactualObservationTrace(
        aggregate=CounterfactualObservation(
            visible_mask=actor_mask,
            visible_occupied_mask=actor_mask,
            visible_dynamic_occupancy=actor_mask,
            newly_visible_mask=actor_mask,
            updated_age_map=age,
        ),
        frames=tuple(frames),
        times_s=action_trace.times_s,
    )


def _resolve(
    *,
    actor_pose: np.ndarray,
    route_relevant: bool,
    action_trace: ActionTrace | None = None,
):
    grid = _grid()
    trace = _trace() if action_trace is None else action_trace
    future = np.tile(actor_pose, (grid.future_steps, 1)).astype(np.float32)
    route = (
        rasterize_footprint(CircleFootprint(0.20), actor_pose, grid)
        if route_relevant
        else np.zeros((grid.height, grid.width), dtype=bool)
    )
    return resolve_verification_response(
        action_trace=trace,
        observation_trace=_observation_trace(trace, actor_pose=actor_pose),
        robot_footprint=CircleFootprint(0.20),
        static_occupancy=np.zeros((grid.height, grid.width), dtype=np.float32),
        dynamic_current_poses={"actor": actor_pose},
        dynamic_future_poses={"actor": future},
        dynamic_specs={
            "actor": {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.20},
            }
        },
        current_visible_mask=np.zeros((grid.height, grid.width), dtype=bool),
        route_corridor_mask=route,
        grid=grid,
        future_dt_s=0.2,
        future_horizon_s=6.4,
        braking_deceleration_mps2=1.0,
        angular_deceleration_radps2=1.6,
        braking_margin_s=0.4,
    )


def test_irrelevant_new_actor_does_not_interrupt_the_action():
    resolution = _resolve(
        actor_pose=np.asarray([1.4, 2.0, 0.0], dtype=np.float32),
        route_relevant=False,
    )

    assert resolution.decision.branch_kind == "complete"
    assert resolution.decision.observation_time_s is None
    assert resolution.branch.end_time_s == pytest.approx(2.0)


def test_route_relevant_but_non_imminent_actor_interrupts_without_braking():
    resolution = _resolve(
        actor_pose=np.asarray([1.4, 0.0, 0.0], dtype=np.float32),
        route_relevant=True,
    )

    assert resolution.decision.branch_kind == "observe_and_replan"
    assert resolution.decision.observation_time_s == pytest.approx(0.2)
    assert resolution.decision.predicted_ttc_s > (
        resolution.decision.stopping_time_s + 0.4
    )
    assert resolution.branch.end_time_s == pytest.approx(0.2)
    assert resolution.branch.executed_trace.linear_velocities_mps[-1] > 0.0


def test_angular_stopping_time_does_not_advance_emergency_braking():
    trace = _trace()
    high_angular_velocity = np.full_like(
        trace.angular_velocities_radps,
        3.2,
    )
    high_angular_velocity[-1] = 0.0
    trace = ActionTrace(
        poses=trace.poses.copy(),
        times_s=trace.times_s.copy(),
        linear_velocities_mps=trace.linear_velocities_mps.copy(),
        angular_velocities_radps=high_angular_velocity,
    )

    resolution = _resolve(
        actor_pose=np.asarray([1.4, 0.0, 0.0], dtype=np.float32),
        route_relevant=True,
        action_trace=trace,
    )

    assert resolution.decision.stopping_time_s == pytest.approx(0.5)
    assert resolution.decision.braking_threshold_s == pytest.approx(0.9)
    assert resolution.decision.branch_kind == "observe_and_replan"


def test_imminent_actor_triggers_collision_free_emergency_braking():
    resolution = _resolve(
        actor_pose=np.asarray([0.9, 0.0, 0.0], dtype=np.float32),
        route_relevant=True,
    )

    assert resolution.decision.branch_kind == "emergency_brake"
    assert resolution.decision.observation_time_s == pytest.approx(0.2)
    assert resolution.decision.predicted_ttc_s <= (
        resolution.decision.stopping_time_s + 0.4
    )
    assert resolution.decision.brake_trace_collision_free is True
    assert resolution.branch.end_time_s > 0.2
    assert resolution.branch.executed_trace.linear_velocities_mps[-1] == 0.0
    assert resolution.branch.executed_trace.angular_velocities_radps[-1] == 0.0


def test_too_late_detection_is_audited_instead_of_being_called_safe():
    resolution = _resolve(
        actor_pose=np.asarray([0.55, 0.0, 0.0], dtype=np.float32),
        route_relevant=True,
    )

    assert resolution.decision.branch_kind == "emergency_brake"
    assert resolution.decision.brake_trace_collision_free is False
    assert resolution.decision.brake_trace_failure_reason == "dynamic_collision"
