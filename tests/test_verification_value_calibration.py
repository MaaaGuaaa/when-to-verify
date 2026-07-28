import numpy as np
import pytest

from src.evaluation.verification_value_calibration import (
    RejectCostCalibrationCriteria,
    calibrate_reject_cost,
    revalue_reject_cost,
)
from src.generation.verification_release import VerificationRevaluationRecord
from src.planning.verification_actions import CANONICAL_ACTION_IDS


def test_revalue_reject_cost_uses_unclipped_policy_losses():
    row = revalue_reject_cost(
        nominal_execute_losses=np.asarray([0.9, 0.1], dtype=np.float64),
        unclipped_best_policy_losses=(0.1, 0.8),
        action_cost=0.05,
        reject_cost=0.3,
    )

    assert row.br_before == pytest.approx(0.3)
    assert row.mean_post_before_action_cost == pytest.approx(0.2)
    assert row.value_target == pytest.approx(0.05)
    assert row.useful_target == 1


def test_revalue_reject_cost_treats_missing_policy_as_reject():
    row = revalue_reject_cost(
        nominal_execute_losses=np.asarray([0.9, 0.1], dtype=np.float64),
        unclipped_best_policy_losses=(None, 0.8),
        action_cost=0.05,
        reject_cost=0.3,
    )

    assert row.mean_post_before_action_cost == pytest.approx(0.3)
    assert row.value_target == pytest.approx(-0.05)
    assert row.useful_target == 0


@pytest.mark.parametrize(
    ("nominal_losses", "policy_losses", "error", "match"),
    (
        (
            np.asarray([], dtype=np.float64),
            (),
            ValueError,
            "non-empty",
        ),
        (
            np.asarray([np.nan], dtype=np.float64),
            (0.1,),
            ValueError,
            "finite",
        ),
        (
            np.asarray([0.1], dtype=np.float32),
            (0.1,),
            TypeError,
            "float64",
        ),
        (
            np.asarray([0.1], dtype=np.float64),
            (),
            ValueError,
            "align",
        ),
    ),
)
def test_revalue_reject_cost_rejects_invalid_loss_vectors(
    nominal_losses,
    policy_losses,
    error,
    match,
):
    with pytest.raises(error, match=match):
        revalue_reject_cost(
            nominal_execute_losses=nominal_losses,
            unclipped_best_policy_losses=policy_losses,
            action_cost=0.05,
            reject_cost=0.3,
        )


def _calibration_records(*, split: str = "train"):
    rows = []
    for group_index in range(2):
        for action_index, action_id in enumerate(CANONICAL_ACTION_IDS):
            low_policy_loss = (action_index + group_index) % 2 == 0
            rows.append(
                VerificationRevaluationRecord(
                    release_request_identity="release-a",
                    split=split,
                    task_id=f"task-{group_index}",
                    mother_id=f"mother-{group_index}",
                    sample_id=f"sample-{group_index}-{action_index}",
                    ranking_group_id=f"group-{group_index}",
                    action_id=action_id,
                    realized_execute_loss=1.0,
                    unclipped_best_policy_loss=(
                        0.19 if low_policy_loss else 0.35
                    ),
                    action_cost=0.05,
                    original_reject_cost=0.2,
                )
            )
    return tuple(rows)


def _criteria(**changes):
    values = {
        "minimum_group_count": 2,
        "minimum_positive_fraction": 0.25,
        "maximum_positive_fraction": 0.75,
        "minimum_mixed_action_count": 2,
    }
    values.update(changes)
    return RejectCostCalibrationCriteria(**values)


def test_calibration_selects_smallest_candidate_that_passes_all_checks():
    result = calibrate_reject_cost(
        _calibration_records(),
        candidates=(0.2, 0.3, 0.5),
        criteria=_criteria(),
    )

    assert result.status == "selected"
    assert result.selected_reject_cost == pytest.approx(0.3)
    assert result.group_count == 2
    assert result.sample_count == 12
    assert result.candidate_reports["0.2"]["status"] == "fail"
    assert result.candidate_reports["0.3"]["status"] == "pass"
    assert result.candidate_reports["0.3"]["positive_fraction"] == pytest.approx(
        0.5
    )
    assert result.candidate_reports["0.3"]["mixed_action_count"] == 6
    assert result.candidate_reports["0.3"][
        "positive_risk_reduction_exceeds_action_cost"
    ] is True
    assert result.candidate_reports["0.5"]["status"] == "fail"


def test_calibration_retains_complete_diagnostics_when_no_candidate_passes():
    result = calibrate_reject_cost(
        _calibration_records(),
        candidates=(0.2, 0.3, 0.5),
        criteria=_criteria(
            minimum_positive_fraction=0.8,
            maximum_positive_fraction=0.9,
        ),
    )

    assert result.status == "no_candidate_passed"
    assert result.selected_reject_cost is None
    assert set(result.candidate_reports) == {"0.2", "0.3", "0.5"}
    assert all(
        report["failed_checks"]
        for report in result.candidate_reports.values()
    )


def test_calibration_rejects_non_train_or_incomplete_action_groups():
    with pytest.raises(ValueError, match="train"):
        calibrate_reject_cost(
            _calibration_records(split="val"),
            candidates=(0.3,),
            criteria=_criteria(),
        )

    with pytest.raises(ValueError, match="six canonical actions"):
        calibrate_reject_cost(
            _calibration_records()[:-1],
            candidates=(0.3,),
            criteria=_criteria(),
        )
