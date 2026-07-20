import math

import pytest
import torch

from src.models.verification_model import (
    VerificationLossConfig,
    VerificationPrediction,
    verification_loss,
)


def _config() -> VerificationLossConfig:
    return VerificationLossConfig(
        huber_delta=1.0,
        value_weight=1.0,
        useful_weight=1.0,
        ranking_weight=1.0,
        ranking_margin=0.5,
    )


def test_composite_loss_matches_hand_calculation_and_correct_order_has_no_hinge():
    prediction = VerificationPrediction(
        g_pred=torch.tensor([0.8, 0.2], dtype=torch.float32, requires_grad=True),
        useful_logit=torch.zeros(2, dtype=torch.float32, requires_grad=True),
    )
    result = verification_loss(
        prediction,
        value_target=torch.tensor([1.0, 0.0], dtype=torch.float32),
        useful_target=torch.tensor([1.0, 0.0], dtype=torch.float32),
        group_ids=("group-0", "group-0"),
        action_ids=("left", "right"),
        config=_config(),
    )

    assert result.value_regression.item() == pytest.approx(0.02)
    assert result.useful_classification.item() == pytest.approx(math.log(2.0))
    assert result.pairwise_ranking.item() == pytest.approx(0.0)
    assert result.total.item() == pytest.approx(0.02 + math.log(2.0))
    assert result.pair_count == 1


def test_wrong_within_group_order_increases_pairwise_loss():
    targets = torch.tensor([1.0, 0.0], dtype=torch.float32)
    useful = torch.tensor([1.0, 0.0], dtype=torch.float32)
    correct = verification_loss(
        VerificationPrediction(
            g_pred=torch.tensor([0.8, 0.2]),
            useful_logit=torch.zeros(2),
        ),
        value_target=targets,
        useful_target=useful,
        group_ids=("same", "same"),
        action_ids=("left", "right"),
        config=_config(),
    )
    wrong = verification_loss(
        VerificationPrediction(
            g_pred=torch.tensor([0.2, 0.8]),
            useful_logit=torch.zeros(2),
        ),
        value_target=targets,
        useful_target=useful,
        group_ids=("same", "same"),
        action_ids=("left", "right"),
        config=_config(),
    )

    assert correct.pairwise_ranking.item() == pytest.approx(0.0)
    assert wrong.pairwise_ranking.item() == pytest.approx(1.1)
    assert wrong.total > correct.total


def test_cross_group_target_ties_and_duplicate_actions_do_not_form_pairs():
    result = verification_loss(
        VerificationPrediction(
            g_pred=torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True),
            useful_logit=torch.zeros(4, requires_grad=True),
        ),
        value_target=torch.tensor([0.0, 1.0, 2.0, 2.0]),
        useful_target=torch.tensor([0.0, 1.0, 1.0, 1.0]),
        group_ids=("a", "b", "c", "c"),
        action_ids=("left", "right", "same", "same"),
        config=_config(),
    )

    assert result.pair_count == 0
    assert result.pairwise_ranking.item() == pytest.approx(0.0)
    result.total.backward()
    assert torch.isfinite(result.total)


def test_loss_rejects_nonbinary_targets_and_identity_length_mismatch():
    prediction = VerificationPrediction(
        g_pred=torch.zeros(2), useful_logit=torch.zeros(2)
    )
    kwargs = {
        "prediction": prediction,
        "value_target": torch.zeros(2),
        "useful_target": torch.tensor([0.0, 2.0]),
        "group_ids": ("group", "group"),
        "action_ids": ("left", "right"),
        "config": _config(),
    }
    with pytest.raises(ValueError, match="binary"):
        verification_loss(**kwargs)
    kwargs["useful_target"] = torch.zeros(2)
    kwargs["group_ids"] = ("group",)
    with pytest.raises(ValueError, match="align"):
        verification_loss(**kwargs)
