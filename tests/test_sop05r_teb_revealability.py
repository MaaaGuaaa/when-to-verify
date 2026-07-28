from pathlib import Path

import pytest

import src.generation.sop05r_teb_revealability as m8
from src.generation.sop05r_teb_revealability import (
    TebPostStopRecoveryDiagnostic,
    build_teb_safe_stop_request,
    evaluate_teb_safe_stop_revealability,
    validate_teb_safe_stop_audit_payload,
)
from src.generation.sop05r_teb_safe_stop_contract import (
    load_sop05r_teb_safe_stop_config,
)
from src.planning.verification_actions import (
    ActionFeasibility,
    load_verification_actions,
)
from tests.test_sop05r_teb_event_sampler import _mother_fixture
from src.generation.sop05r_teb_event_sampler import build_sop05r_teb_mother


ROOT = Path(__file__).resolve().parents[1]


def _request():
    inputs = _mother_fixture()
    evaluation = build_sop05r_teb_mother(
        base_config=inputs[0],
        source_base_state=inputs[1],
        source_oracle_context=inputs[2],
        teb_config=inputs[3],
        task_template=inputs[4],
        placement_result=inputs[5],
        snippet=inputs[6],
        seed=43,
    )
    assert evaluation.mother is not None
    library = load_verification_actions(
        ROOT / "configs/verification_actions.yaml"
    )
    return (
        build_teb_safe_stop_request(
            mother=evaluation.mother,
            action_library=library,
            base_config=inputs[0],
            teb_config=inputs[3],
            safe_stop_config=load_sop05r_teb_safe_stop_config(
                ROOT / "configs/sop05r_m8_safe_stop.yaml"
            ),
        ),
        evaluation.mother,
    )


def _feasible(*_args, **_kwargs) -> ActionFeasibility:
    return ActionFeasibility(
        feasible=True,
        reason=None,
        critical_object_id=None,
        minimum_dynamic_clearance_m=1.0,
    )


def test_m8_v2_is_deterministic_and_serializes_a_versioned_contract() -> None:
    request, mother = _request()

    first = evaluate_teb_safe_stop_revealability(request)
    second = evaluate_teb_safe_stop_revealability(request)

    assert first == second
    assert first.label_definition_version == "sop05r_teb_safe_stop_v2"
    assert first.safe_stop_config_digest == request.safe_stop_config.digest
    assert len(first.actions) == len(request.action_library.actions)
    assert {item.action_id for item in first.actions} == {
        action.action_id for action in request.action_library.actions
    }
    for item in first.actions:
        assert item.matched_wait_action_id.endswith("-matched-wait")
        assert item.braking_margin_s == 0.4
        assert item.post_stop_recovery.shared_goal_world_pose == tuple(
            mother.trajectory_record.shared_goal_world_pose
        )
    assert "stop_scan" not in first.safe_stop_action_ids
    payload = first.as_dict()
    validate_teb_safe_stop_audit_payload(payload)
    legacy = dict(payload)
    legacy["label_definition_version"] = "sop05r_teb_revealability_v1"
    with pytest.raises(ValueError, match="label definition version"):
        validate_teb_safe_stop_audit_payload(legacy)


def test_m8_v2_does_not_gate_safe_stop_on_wait_or_same_goal_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _ = _request()
    monkeypatch.setattr(m8, "_target_hidden_at_decision", lambda **_kwargs: True)
    monkeypatch.setattr(m8, "_first_visible_time", lambda **_kwargs: 0.2)
    monkeypatch.setattr(m8, "_trace_feasibility", _feasible)

    def unavailable_recovery(*_args, **kwargs):
        return TebPostStopRecoveryDiagnostic(
            evaluated=True,
            planner_branch="observed_dynamic",
            route_available=False,
            shared_goal_world_pose=kwargs["goal"],
            replanned_task_cost=None,
            rejection_reason="teb_static_collision",
        )

    monkeypatch.setattr(
        m8,
        "_post_stop_recovery_diagnostic",
        unavailable_recovery,
    )

    audit = evaluate_teb_safe_stop_revealability(request)

    assert audit.safe_stop_action_ids == (
        "arc_left_30",
        "arc_right_30",
        "arc_left_45",
        "arc_right_45",
        "forward_peek",
    )
    for item in audit.actions:
        if item.action_id == "stop_scan":
            assert item.safe_stop_rejection_reason == "stop_scan_excluded"
            continue
        assert item.safe_stop_revealable
        assert not item.matched_wait_visibility_advantage_diagnostic
        assert item.post_stop_recovery.route_available is False
        assert item.post_stop_recovery.rejection_reason == "teb_static_collision"


def test_m8_v2_rejects_insufficient_braking_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _ = _request()
    monkeypatch.setattr(m8, "_target_hidden_at_decision", lambda **_kwargs: True)
    monkeypatch.setattr(
        m8,
        "_first_visible_time",
        lambda **_kwargs: (
            request.mother.collision.first_collision_time_after_decision_s - 0.2
        ),
    )
    monkeypatch.setattr(m8, "_trace_feasibility", _feasible)

    audit = evaluate_teb_safe_stop_revealability(request)

    assert not audit.safe_stop_action_ids
    for item in audit.actions:
        if item.action_id != "stop_scan":
            assert item.safe_stop_rejection_reason == "braking_margin_insufficient"
            assert not item.braking_margin_ok


def test_m8_v2_rejects_a_colliding_prefix_or_brake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _ = _request()
    monkeypatch.setattr(m8, "_target_hidden_at_decision", lambda **_kwargs: True)
    monkeypatch.setattr(m8, "_first_visible_time", lambda **_kwargs: 0.2)
    calls = 0

    def collision_on_safe_stop(*_args, **_kwargs) -> ActionFeasibility:
        nonlocal calls
        calls += 1
        if calls % 3 == 0:
            return ActionFeasibility(
                feasible=False,
                reason="dynamic_collision",
                critical_object_id="target",
                minimum_dynamic_clearance_m=-0.1,
            )
        return _feasible()

    monkeypatch.setattr(m8, "_trace_feasibility", collision_on_safe_stop)

    audit = evaluate_teb_safe_stop_revealability(request)

    assert not audit.safe_stop_action_ids
    for item in audit.actions:
        if item.action_id != "stop_scan":
            assert item.prefix_and_brake_feasible is False
            assert item.safe_stop_rejection_reason == "safe_stop_dynamic_collision"


def test_m8_safe_stop_config_rejects_legacy_or_nonholding_semantics(
    tmp_path: Path,
) -> None:
    config = (ROOT / "configs/sop05r_m8_safe_stop.yaml").read_text(
        encoding="utf-8"
    )
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        config.replace("sop05r_teb_safe_stop_v2", "legacy_v1"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="label definition"):
        load_sop05r_teb_safe_stop_config(legacy)

    nonholding = tmp_path / "nonholding.yaml"
    nonholding.write_text(
        config.replace("hold_until_collision: true", "hold_until_collision: false"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hold until"):
        load_sop05r_teb_safe_stop_config(nonholding)
