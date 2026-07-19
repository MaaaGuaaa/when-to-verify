from __future__ import annotations

import numpy as np
import pytest

from src.models.occupancy_aggregation import (
    future_endpoint_times,
    probabilistic_union_risk,
    weighted_swept_volume_risk,
)


def test_future_endpoint_times_are_dt_through_horizon() -> None:
    times = future_endpoint_times(future_steps=15, dt_s=0.2)

    assert times.dtype == np.float32
    np.testing.assert_allclose(times, np.arange(1, 16, dtype=np.float32) * 0.2)
    assert float(times[0]) == pytest.approx(0.2)
    assert float(times[-1]) == pytest.approx(3.0)


def test_weighted_sum_matches_hand_computed_normalized_score() -> None:
    occupancy = np.array(
        [[[[0.2, 0.4], [0.0, 0.0]], [[0.8, 0.0], [0.0, 0.0]]]],
        dtype=np.float32,
    )
    footprint = np.array(
        [[[[1.0, 1.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]],
        dtype=np.float32,
    )
    w0 = np.exp(-0.2 / 2.0)
    w1 = np.exp(-0.4 / 2.0)
    expected = (0.2 * w0 + 0.4 * w0 + 0.8 * w1) / (2.0 * w0 + w1)

    actual = weighted_swept_volume_risk(
        occupancy,
        footprint,
        dt_s=0.2,
        sigma_time_s=2.0,
    )

    assert actual.dtype == np.float32
    assert actual.shape == (1,)
    assert float(actual[0]) == pytest.approx(expected, abs=1e-7)


def test_union_matches_unique_cell_time_hand_calculation() -> None:
    occupancy = np.array(
        [[[[0.2, 0.4], [0.0, 0.0]], [[0.8, 0.0], [0.0, 0.0]]]],
        dtype=np.float32,
    )
    footprint = np.array(
        [[[[1.0, 1.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]],
        dtype=np.float32,
    )

    actual = probabilistic_union_risk(occupancy, footprint)

    assert actual.dtype == np.float32
    assert float(actual[0]) == pytest.approx(1.0 - (0.8 * 0.6 * 0.2), abs=1e-7)


@pytest.mark.parametrize(
    "aggregator",
    [weighted_swept_volume_risk, probabilistic_union_risk],
)
def test_aggregation_normalizes_mask_and_is_zero_for_empty_or_zero_probability(
    aggregator,
) -> None:
    occupancy = np.zeros((1, 3, 2, 2), dtype=np.float32)
    occupancy[0, 1, 0, 0] = 0.7
    binary = np.zeros_like(occupancy)
    binary[0, 1, 0, 0] = 1.0
    repeated = binary * 17.0

    np.testing.assert_allclose(aggregator(occupancy, binary), aggregator(occupancy, repeated))
    np.testing.assert_array_equal(aggregator(np.zeros_like(occupancy), binary), [0.0])
    np.testing.assert_array_equal(aggregator(occupancy, np.zeros_like(binary)), [0.0])


@pytest.mark.parametrize(
    "aggregator",
    [weighted_swept_volume_risk, probabilistic_union_risk],
)
def test_increasing_a_selected_probability_cannot_reduce_risk(aggregator) -> None:
    footprint = np.zeros((1, 3, 2, 2), dtype=np.float32)
    footprint[0, 1, 0, 0] = 1.0
    low = np.zeros_like(footprint)
    high = np.zeros_like(footprint)
    low[0, 1, 0, 0] = 0.1
    high[0, 1, 0, 0] = 0.9

    assert float(aggregator(high, footprint)[0]) >= float(aggregator(low, footprint)[0])


@pytest.mark.parametrize(
    "bad_occupancy,bad_footprint,match",
    [
        (np.zeros((2, 3, 4), np.float32), np.zeros((2, 3, 4), np.float32), "rank 4"),
        (np.zeros((1, 3, 4, 4), np.float64), np.zeros((1, 3, 4, 4), np.float32), "float32"),
        (np.zeros((1, 3, 4, 4), np.float32), np.zeros((1, 2, 4, 4), np.float32), "same shape"),
    ],
)
def test_aggregation_rejects_contract_violations(
    bad_occupancy: np.ndarray,
    bad_footprint: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        weighted_swept_volume_risk(bad_occupancy, bad_footprint)


def test_aggregation_rejects_nonfinite_and_out_of_range_probabilities() -> None:
    footprint = np.ones((1, 2, 2, 2), dtype=np.float32)
    for bad in (np.full_like(footprint, np.nan), np.full_like(footprint, 1.01)):
        with pytest.raises(ValueError, match="occupancy probabilities"):
            probabilistic_union_risk(bad, footprint)
