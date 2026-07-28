import importlib
import importlib.util

import pytest


def _closed_loop_module():
    spec = importlib.util.find_spec("src.evaluation.closed_loop")
    assert spec is not None, "SOP-15 closed-loop module is missing"
    return importlib.import_module("src.evaluation.closed_loop")


def test_closed_loop_rejects_a_non_long40_future_horizon():
    closed_loop = _closed_loop_module()

    with pytest.raises(ValueError, match="6.4"):
        closed_loop.ClosedLoopConfig(future_horizon_s=3.0)


def test_toy_verify_replans_and_consumes_the_original_future_window():
    closed_loop = _closed_loop_module()
    scenario = closed_loop.ToyScenario(
        episode_id="verify-then-execute",
        initial_calibrated_risk=0.9,
        post_verify_calibrated_risk=0.1,
        task_cost=0.1,
        reject_cost=1.2,
        required_execute_time_s=0.4,
        action_values={"forward_peek": 0.8},
        hazard_object_type="human",
    )
    config = closed_loop.ClosedLoopConfig(
        future_horizon_s=6.4,
        execute_step_s=0.2,
        verify_step_s=0.6,
        risk_weight=1.0,
        verify_margin=0.01,
        collision_risk_threshold=0.7,
        max_decisions=8,
    )

    trace = closed_loop.run_toy_closed_loop(
        scenario, strategy="learned", config=config
    )

    assert [step.decision for step in trace.steps] == [
        "verify",
        "execute",
        "execute",
    ]
    assert trace.termination_reason == "success"
    assert trace.replan_count == 2
    assert trace.collision is False
    assert trace.success is True
    assert trace.hazard_object_type == "human"
    assert trace.elapsed_s == pytest.approx(1.0)
    assert trace.steps[1].remaining_before_s == pytest.approx(5.8)
    assert all(
        step.remaining_after_s < step.remaining_before_s for step in trace.steps
    )
    assert trace.remaining_horizon_s == pytest.approx(5.4)
    first_step = trace.steps[0]
    assert first_step.execute_cost == pytest.approx(1.0)
    assert first_step.no_verify_cost == pytest.approx(1.0)
    assert first_step.predicted_net_value == pytest.approx(0.8)
    assert first_step.verify_cost == pytest.approx(0.2)
    assert first_step.decision_reason == "positive_net_value_exceeds_margin"


def test_toy_reject_has_an_explicit_terminal_reason_instead_of_a_silent_skip():
    closed_loop = _closed_loop_module()
    scenario = closed_loop.ToyScenario(
        episode_id="no-candidate",
        initial_calibrated_risk=0.9,
        post_verify_calibrated_risk=0.9,
        task_cost=0.2,
        reject_cost=0.5,
        required_execute_time_s=0.4,
        action_values={},
        hazard_object_type="human",
    )

    trace = closed_loop.run_toy_closed_loop(
        scenario,
        strategy="learned",
        config=closed_loop.ClosedLoopConfig(max_decisions=4),
    )

    assert trace.termination_reason == "rejected_no_candidate"
    assert trace.reject_count == 1
    assert trace.success is False
    assert trace.collision is False


def test_terminal_non_success_step_is_not_reported_as_replanned():
    closed_loop = _closed_loop_module()
    scenario = closed_loop.ToyScenario(
        episode_id="decision-limit",
        initial_calibrated_risk=0.1,
        post_verify_calibrated_risk=0.1,
        task_cost=0.1,
        reject_cost=1.2,
        required_execute_time_s=0.4,
        action_values={},
        hazard_object_type="human",
    )

    trace = closed_loop.run_toy_closed_loop(
        scenario,
        strategy="never",
        config=closed_loop.ClosedLoopConfig(max_decisions=1),
    )

    assert trace.termination_reason == "max_decisions"
    assert trace.replan_count == 0
    assert trace.steps[0].replanned_after is False


def test_summary_separates_terminal_time_from_successful_completion_time():
    closed_loop = _closed_loop_module()
    config = closed_loop.ClosedLoopConfig()
    successful = closed_loop.run_toy_closed_loop(
        closed_loop.ToyScenario(
            episode_id="success",
            initial_calibrated_risk=0.1,
            post_verify_calibrated_risk=0.1,
            task_cost=0.1,
            reject_cost=1.2,
            required_execute_time_s=0.4,
            action_values={},
            hazard_object_type="human",
        ),
        strategy="never",
        config=config,
    )
    rejected = closed_loop.run_toy_closed_loop(
        closed_loop.ToyScenario(
            episode_id="rejected",
            initial_calibrated_risk=0.9,
            post_verify_calibrated_risk=0.9,
            task_cost=0.2,
            reject_cost=0.5,
            required_execute_time_s=0.4,
            action_values={},
            hazard_object_type="human",
        ),
        strategy="never",
        config=config,
    )

    summary = closed_loop.summarize_toy_episodes((successful, rejected))

    assert summary["terminal_time_mean_s"] == pytest.approx(0.2)
    assert summary["successful_completion_time_mean_s"] == pytest.approx(0.4)
    assert summary["successful_completion_count"] == 1
    assert "completion_time_mean_s" not in summary
