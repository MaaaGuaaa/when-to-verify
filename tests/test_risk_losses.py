"""Hand-verifiable SOP09 risk-loss tests."""

from __future__ import annotations

import math

import pytest
import torch

from src.models.losses import pinball_loss, risk_loss


def test_pinball_loss_matches_hand_calculation():
    predictions = torch.tensor(
        [[0.20, 0.40, 0.60, 0.80], [0.80, 0.60, 0.40, 0.20]],
        dtype=torch.float32,
    )
    targets = torch.tensor([1.0, 0.0], dtype=torch.float32)
    levels = (0.5, 0.8, 0.9, 0.95)

    # Row 0 under-predicts: beta * (y - q). Row 1 over-predicts:
    # (1 - beta) * (q - y). The public loss averages all eight terms.
    expected_terms = [
        0.5 * 0.8,
        0.8 * 0.6,
        0.9 * 0.4,
        0.95 * 0.2,
        0.5 * 0.8,
        0.2 * 0.6,
        0.1 * 0.4,
        0.05 * 0.2,
    ]
    assert pinball_loss(predictions, targets, levels=levels).item() == pytest.approx(
        sum(expected_terms) / len(expected_terms)
    )


def test_risk_loss_composes_pinball_and_collision_bce():
    output = {
        "quantiles": torch.zeros((2, 4), dtype=torch.float32),
        "collision_logits": torch.zeros(2, dtype=torch.float32),
        "p_collision": torch.full((2,), 0.5, dtype=torch.float32),
    }
    severity = torch.ones(2, dtype=torch.float32)
    collision = torch.tensor([0.0, 1.0], dtype=torch.float32)

    losses = risk_loss(
        output,
        risk_severity=severity,
        collision_label=collision,
        lambda_collision=2.0,
    )
    expected_pinball = (0.5 + 0.8 + 0.9 + 0.95) / 4.0
    assert losses["pinball"].item() == pytest.approx(expected_pinball)
    assert losses["collision_bce"].item() == pytest.approx(math.log(2.0))
    assert losses["occupancy_aux"].item() == pytest.approx(0.0)
    assert losses["total"].item() == pytest.approx(
        expected_pinball + 2.0 * math.log(2.0)
    )


def test_optional_occupancy_auxiliary_loss_requires_both_prediction_and_label():
    output = {
        "quantiles": torch.zeros((1, 4), dtype=torch.float32),
        "collision_logits": torch.zeros(1, dtype=torch.float32),
        "p_collision": torch.full((1,), 0.5, dtype=torch.float32),
        "occupancy_aux_logits": torch.zeros((1, 15, 4, 4), dtype=torch.float32),
    }
    target = torch.zeros((1, 15, 4, 4), dtype=torch.float32)
    with_aux = risk_loss(
        output,
        risk_severity=torch.zeros(1),
        collision_label=torch.zeros(1),
        occupancy_target=target,
        lambda_occupancy_aux=0.2,
    )
    assert with_aux["occupancy_aux"].item() == pytest.approx(math.log(2.0))

    with pytest.raises(ValueError, match="occupancy"):
        risk_loss(
            output,
            risk_severity=torch.zeros(1),
            collision_label=torch.zeros(1),
            occupancy_target=None,
            lambda_occupancy_aux=0.2,
        )


def test_loss_rejects_wrong_shapes_and_nonfinite_values():
    with pytest.raises(ValueError, match="shape"):
        pinball_loss(torch.zeros((2, 3)), torch.zeros(2))
    predictions = torch.zeros((2, 4))
    predictions[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        pinball_loss(predictions, torch.zeros(2))


def test_loss_rejects_targets_outside_frozen_probability_bounds():
    output = {
        "quantiles": torch.zeros((1, 4), dtype=torch.float32),
        "collision_logits": torch.zeros(1, dtype=torch.float32),
        "p_collision": torch.full((1,), 0.5, dtype=torch.float32),
    }
    with pytest.raises(ValueError, match="risk_severity"):
        risk_loss(
            output,
            risk_severity=torch.tensor([-0.1]),
            collision_label=torch.zeros(1),
        )
    with pytest.raises(ValueError, match="collision_label"):
        risk_loss(
            output,
            risk_severity=torch.zeros(1),
            collision_label=torch.tensor([1.1]),
        )
