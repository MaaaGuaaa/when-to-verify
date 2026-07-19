from __future__ import annotations

import numpy as np
import pytest
import torch

from src.contracts import HISTORY_CHANNELS
from src.models.occupancy_baseline import (
    ConvGRUOccupancyPredictor,
    LearnedOccupancyRiskAggregator,
)


def _history(batch_size: int = 3, grid_size: int = 8) -> torch.Tensor:
    history = torch.zeros(
        batch_size,
        8,
        len(HISTORY_CHANNELS),
        grid_size,
        grid_size,
        dtype=torch.float32,
    )
    dynamic = HISTORY_CHANNELS.index("past_dynamic_occupancy")
    visible = HISTORY_CHANNELS.index("past_visible_mask")
    for step in range(8):
        history[:, step, dynamic, 2, min(step, grid_size - 1)] = 1.0
        history[:, step, visible] = 1.0
    return history


def test_convgru_predictor_has_frozen_future_contract_and_is_deterministic() -> None:
    torch.manual_seed(11)
    model = ConvGRUOccupancyPredictor(hidden_channels=4, future_steps=15)
    history = _history()

    probability = model(history)
    repeated = model(history)
    logits = model.predict_logits(history)

    assert probability.shape == (3, 15, 8, 8)
    assert probability.dtype == torch.float32
    assert torch.isfinite(probability).all()
    assert torch.all((probability >= 0.0) & (probability <= 1.0))
    torch.testing.assert_close(probability, repeated, rtol=0.0, atol=0.0)
    torch.testing.assert_close(probability, torch.sigmoid(logits))


def test_convgru_rejects_wrong_history_channel_contract() -> None:
    model = ConvGRUOccupancyPredictor(hidden_channels=4)
    wrong = torch.zeros(1, 8, 1, 8, 8, dtype=torch.float32)

    with pytest.raises(ValueError, match="history channels"):
        model(wrong)


@pytest.mark.parametrize("history_steps", [7, 9])
def test_convgru_requires_exactly_eight_history_frames(history_steps: int) -> None:
    model = ConvGRUOccupancyPredictor(hidden_channels=4)
    wrong = torch.zeros(
        1,
        history_steps,
        len(HISTORY_CHANNELS),
        8,
        8,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="exactly 8 history frames"):
        model(wrong)


def test_learned_aggregator_only_consumes_prediction_and_query_geometry() -> None:
    torch.manual_seed(3)
    aggregator = LearnedOccupancyRiskAggregator(future_steps=15, hidden_dim=8)
    occupancy = torch.rand(4, 15, 6, 6, dtype=torch.float32)
    footprint = torch.zeros_like(occupancy)
    footprint[:, :, 2:4, 2:4] = 1.0

    probability = aggregator(occupancy, footprint)

    assert probability.shape == (4,)
    assert probability.dtype == torch.float32
    assert torch.isfinite(probability).all()
    assert torch.all((probability >= 0.0) & (probability <= 1.0))


def test_learned_aggregator_is_invariant_to_nonbinary_mask_magnitude() -> None:
    torch.manual_seed(9)
    aggregator = LearnedOccupancyRiskAggregator(future_steps=15, hidden_dim=8)
    occupancy = torch.rand(2, 15, 4, 4, dtype=torch.float32)
    footprint = torch.zeros_like(occupancy)
    footprint[:, :, 1, 1] = 1.0

    binary = aggregator(occupancy, footprint)
    repeated = aggregator(occupancy, footprint * 8.0)

    torch.testing.assert_close(binary, repeated, rtol=0.0, atol=0.0)


def test_learned_aggregator_rejects_oracle_argument_by_api() -> None:
    aggregator = LearnedOccupancyRiskAggregator(future_steps=15)
    occupancy = torch.zeros(1, 15, 4, 4, dtype=torch.float32)
    footprint = torch.ones_like(occupancy)
    labels = torch.zeros_like(occupancy)

    with pytest.raises(TypeError):
        aggregator(occupancy, footprint, labels)
