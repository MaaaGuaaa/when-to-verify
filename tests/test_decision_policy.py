import importlib
import importlib.util

import pytest


def _policy_module():
    spec = importlib.util.find_spec("src.planning.decision_policy")
    assert spec is not None, "SOP-15 decision policy module is missing"
    return importlib.import_module("src.planning.decision_policy")


def test_learned_value_is_a_net_value_and_verify_cost_is_not_double_charged():
    policy = _policy_module()
    request = policy.DecisionRequest(
        strategy="learned",
        task_cost=1.0,
        calibrated_risk=0.5,
        risk_weight=2.0,
        reject_cost=3.0,
        verify_margin=0.1,
        action_values={"forward_peek": 0.75, "stop_scan": 0.50},
    )

    result = policy.select_decision(request)

    assert result.decision == "verify"
    assert result.action_id == "forward_peek"
    assert result.execute_cost == pytest.approx(2.0)
    assert result.no_verify_cost == pytest.approx(2.0)
    assert result.predicted_net_value == pytest.approx(0.75)
    assert result.verify_cost == pytest.approx(1.25)
