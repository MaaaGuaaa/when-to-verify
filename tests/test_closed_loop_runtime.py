from __future__ import annotations

from dataclasses import dataclass, replace

import pytest


@dataclass
class _GraphEnvironment:
    episode_id: str
    initial_state_id: str
    hazard_object_type: str
    nominal_task_time_s: float
    nominal_path_length_m: float
    frames: dict[str, object]
    execute_edges: dict[str, object]
    verify_edges: dict[tuple[str, str], object]

    def decision_frame(self, state_id: str):
        return self.frames[state_id]

    def execute(self, state_id: str, duration_s: float):
        edge = self.execute_edges[state_id]
        assert edge.duration_s == pytest.approx(duration_s)
        return edge

    def verify(self, state_id: str, action_id: str):
        return self.verify_edges[(state_id, action_id)]


def _verify_then_execute_environment(runtime, *, same_plan: bool = False):
    return _GraphEnvironment(
        episode_id="verify-then-execute",
        initial_state_id="s0",
        hazard_object_type="human",
        nominal_task_time_s=0.4,
        nominal_path_length_m=0.2,
        frames={
            "s0": runtime.DecisionFrame(
                state_id="s0",
                plan_id="plan-0",
                task_cost=0.1,
                calibrated_risk=0.9,
                reject_cost=1.2,
                action_values_by_strategy={"learned": {"forward_peek": 0.8}},
                action_durations_s={"forward_peek": 0.6},
            ),
            "s1": runtime.DecisionFrame(
                state_id="s1",
                plan_id="plan-0" if same_plan else "plan-1",
                task_cost=0.1,
                calibrated_risk=0.1,
                reject_cost=1.2,
                action_values_by_strategy={},
                action_durations_s={},
            ),
            "s2": runtime.DecisionFrame(
                state_id="s2",
                plan_id="plan-2",
                task_cost=0.1,
                calibrated_risk=0.1,
                reject_cost=1.2,
                action_values_by_strategy={},
                action_durations_s={},
            ),
        },
        execute_edges={
            "s1": runtime.EnvironmentTransition(
                kind="execute",
                source_state_id="s1",
                source_plan_id="plan-0" if same_plan else "plan-1",
                duration_s=0.2,
                path_length_m=0.1,
                next_state_id="s2",
                next_plan_id="plan-2",
                replanned=True,
                near_miss=True,
            ),
            "s2": runtime.EnvironmentTransition(
                kind="execute",
                source_state_id="s2",
                source_plan_id="plan-2",
                duration_s=0.2,
                path_length_m=0.1,
                task_complete=True,
                termination_reason="goal_reached",
            ),
        },
        verify_edges={
            ("s0", "forward_peek"): runtime.EnvironmentTransition(
                kind="verify",
                source_state_id="s0",
                source_plan_id="plan-0",
                action_id="forward_peek",
                duration_s=0.6,
                path_length_m=0.05,
                next_state_id="s1",
                next_plan_id="plan-0" if same_plan else "plan-1",
                replanned=True,
                critical_actor_revealed=True,
                verification_useful=True,
            )
        },
    )


def test_generic_closed_loop_verifies_replans_and_keeps_one_long40_window():
    from src.evaluation import closed_loop_runtime as runtime

    trace = runtime.run_closed_loop(
        _verify_then_execute_environment(runtime),
        strategy="learned",
        config=runtime.ClosedLoopRuntimeConfig(),
    )

    assert [step.decision for step in trace.steps] == [
        "verify",
        "execute",
        "execute",
    ]
    assert trace.termination_reason == "goal_reached"
    assert trace.success is True
    assert trace.elapsed_s == pytest.approx(1.0)
    assert trace.remaining_horizon_s == pytest.approx(5.4)
    assert trace.path_length_m == pytest.approx(0.25)
    assert trace.extra_time_s == pytest.approx(0.6)
    assert trace.extra_path_length_m == pytest.approx(0.05)
    assert trace.replan_count == 2
    assert trace.verification_count == 1
    assert trace.unnecessary_verification_count == 0
    assert trace.steps[0].verify_cost == pytest.approx(0.2)
    assert trace.steps[0].predicted_net_value == pytest.approx(0.8)


def test_generic_closed_loop_uses_environment_collision_not_a_risk_threshold():
    from src.evaluation import closed_loop_runtime as runtime

    environment = _GraphEnvironment(
        episode_id="false-safe",
        initial_state_id="s0",
        hazard_object_type="human",
        nominal_task_time_s=0.2,
        nominal_path_length_m=0.1,
        frames={
            "s0": runtime.DecisionFrame(
                state_id="s0",
                plan_id="plan-0",
                task_cost=0.1,
                calibrated_risk=0.01,
                reject_cost=1.2,
                action_values_by_strategy={},
                action_durations_s={},
            )
        },
        execute_edges={
            "s0": runtime.EnvironmentTransition(
                kind="execute",
                source_state_id="s0",
                source_plan_id="plan-0",
                duration_s=0.2,
                path_length_m=0.1,
                collision=True,
                termination_reason="continuous_collision",
            )
        },
        verify_edges={},
    )

    trace = runtime.run_closed_loop(
        environment,
        strategy="never",
        config=runtime.ClosedLoopRuntimeConfig(),
    )

    assert trace.collision is True
    assert trace.false_safe_execute is True
    assert trace.success is False
    assert trace.termination_reason == "continuous_collision"


def test_generic_closed_loop_rejects_reusing_the_preverification_plan():
    from src.evaluation import closed_loop_runtime as runtime

    with pytest.raises(ValueError, match="new plan"):
        runtime.run_closed_loop(
            _verify_then_execute_environment(runtime, same_plan=True),
            strategy="learned",
            config=runtime.ClosedLoopRuntimeConfig(),
        )


def test_generic_closed_loop_rejects_a_frame_that_does_not_match_claimed_replan():
    from src.evaluation import closed_loop_runtime as runtime

    environment = _verify_then_execute_environment(runtime)
    transition = environment.verify_edges[("s0", "forward_peek")]
    environment.verify_edges[("s0", "forward_peek")] = replace(
        transition,
        next_plan_id="claimed-plan",
    )

    with pytest.raises(ValueError, match="claimed next plan"):
        runtime.run_closed_loop(
            environment,
            strategy="learned",
            config=runtime.ClosedLoopRuntimeConfig(),
        )


def test_generic_summary_reports_full_sop15_metrics():
    from src.evaluation import closed_loop_runtime as runtime

    successful = runtime.run_closed_loop(
        _verify_then_execute_environment(runtime),
        strategy="learned",
        config=runtime.ClosedLoopRuntimeConfig(),
    )
    collision_environment = _GraphEnvironment(
        episode_id="collision",
        initial_state_id="c0",
        hazard_object_type="human",
        nominal_task_time_s=0.2,
        nominal_path_length_m=0.1,
        frames={
            "c0": runtime.DecisionFrame(
                state_id="c0",
                plan_id="collision-plan",
                task_cost=0.1,
                calibrated_risk=0.1,
                reject_cost=1.2,
                action_values_by_strategy={},
                action_durations_s={},
            )
        },
        execute_edges={
            "c0": runtime.EnvironmentTransition(
                kind="execute",
                source_state_id="c0",
                source_plan_id="collision-plan",
                duration_s=0.2,
                path_length_m=0.1,
                collision=True,
                termination_reason="continuous_collision",
            )
        },
        verify_edges={},
    )
    collided = runtime.run_closed_loop(
        collision_environment,
        strategy="never",
        config=runtime.ClosedLoopRuntimeConfig(),
    )

    metrics = runtime.summarize_closed_loop_episodes((successful, collided))

    assert metrics["episode_count"] == 2
    assert metrics["collision_rate"] == pytest.approx(0.5)
    assert metrics["near_miss_rate"] == pytest.approx(0.5)
    assert metrics["false_safe_execution_rate"] == pytest.approx(0.5)
    assert metrics["verification_count_mean"] == pytest.approx(0.5)
    assert metrics["unnecessary_verification_rate"] == pytest.approx(0.0)
    assert metrics["reject_rate"] == pytest.approx(0.0)
    assert metrics["success_rate"] == pytest.approx(0.5)
    assert metrics["path_length_mean_m"] == pytest.approx(0.175)
    assert metrics["extra_path_length_mean_m"] == pytest.approx(0.025)
    assert metrics["extra_time_mean_s"] == pytest.approx(0.3)
