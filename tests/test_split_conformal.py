from __future__ import annotations

import numpy as np
import pytest

from src.calibration.grouped_calibration import (
    apply_grouped_calibration,
    fit_grouped_calibration,
    group_value,
)
from src.calibration.split_conformal import (
    CalibrationContractError,
    apply_split_conformal,
    finite_sample_conformal_quantile,
    one_sided_residuals,
)


def _row(sample_id: str, severity: float, q90: float, **overrides: object) -> dict:
    row = {
        "sample_id": sample_id,
        "split": "calibration",
        "pair_group_id": f"pair-{sample_id}",
        "event_type": "collision",
        "recording_id": f"recording-{sample_id}",
        "session_id": f"session-{sample_id}",
        "source_object_id": f"object-{sample_id}",
        "snippet_id": f"snippet-{sample_id}",
        "base_state_id": f"base-{sample_id}",
        "seed_namespace": f"seed-{sample_id}",
        "collision_label": 1,
        "risk_severity": severity,
        "min_clearance": 0.0,
        "critical_object_id": f"critical-{sample_id}",
        "p_collision": 0.8,
        "q50": 0.4,
        "q80": 0.6,
        "q90": q90,
        "q95": 0.9,
        "blind_type": "corner",
        "critical_area_fraction": 0.04,
        "age_s": 0.8,
        "density_fraction": 0.02,
        "target_object_type": "human",
        "footprint_kind": "circle",
        "ood_tag": "in_distribution",
        "pair_eligible": True,
    }
    row.update(overrides)
    return row


def test_one_sided_residuals_match_hand_calculation() -> None:
    actual = one_sided_residuals(
        np.asarray([0.2, 0.8, 0.9]),
        np.asarray([0.3, 0.5, 0.9]),
    )
    np.testing.assert_allclose(actual, [0.0, 0.3, 0.0])


def test_finite_sample_quantile_uses_explicit_ceiling_rank() -> None:
    residuals = np.asarray([0.0, 0.1, 0.2, 0.4])

    value, rank = finite_sample_conformal_quantile(residuals, alpha=0.2)

    assert rank == 4  # ceil((4 + 1) * 0.8)
    assert value == pytest.approx(0.4)


def test_finite_sample_quantile_clips_nominal_rank_to_sample_count() -> None:
    residuals = np.asarray([0.0, 0.1, 0.2, 0.4])

    value, rank = finite_sample_conformal_quantile(residuals, alpha=0.1)

    assert rank == 4  # min(4, ceil((4 + 1) * 0.9))
    assert value == pytest.approx(0.4)


def test_finite_sample_quantile_rejects_empty_or_nonfinite_input() -> None:
    with pytest.raises(CalibrationContractError, match="non-empty"):
        finite_sample_conformal_quantile(np.asarray([]), alpha=0.1)
    with pytest.raises(CalibrationContractError, match="finite"):
        finite_sample_conformal_quantile(np.asarray([0.0, np.nan]), alpha=0.1)


def test_split_conformal_clips_only_after_adding_correction() -> None:
    calibrated = apply_split_conformal(
        np.asarray([0.2, 0.95]), correction=0.1
    )
    np.testing.assert_allclose(calibrated, [0.3, 1.0])


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("critical_area_fraction", 0.00, "[0,0.05)"),
        ("critical_area_fraction", 0.05, "[0.05,0.2)"),
        ("critical_area_fraction", 1.00, "[0.2,1]"),
        ("age_s", 1.00, "[1,3)"),
        ("density_fraction", 0.01, "[0.01,0.05)"),
    ],
)
def test_continuous_group_edges_have_frozen_boundary_semantics(
    field: str, value: float, expected: str
) -> None:
    assert group_value(field, value) == expected


def test_grouped_calibration_falls_back_to_global_for_sparse_group() -> None:
    rows = [
        _row("a", 0.8, 0.4, blind_type="corner"),
        _row("b", 0.7, 0.4, blind_type="corner"),
        _row("c", 0.6, 0.4, blind_type="doorway"),
    ]

    artifact = fit_grouped_calibration(
        rows,
        alpha=0.4,
        prediction_key="q90",
        dimensions=("blind_type",),
        min_group_size=2,
    )

    global_correction = artifact["global"]["correction"]
    corner = artifact["dimensions"]["blind_type"]["corner"]
    doorway = artifact["dimensions"]["blind_type"]["doorway"]
    assert corner["fallback"] is False
    assert corner["count"] == 2
    assert corner["rank_one_based"] == 2
    assert corner["residual_min"] == pytest.approx(0.3)
    assert corner["residual_max"] == pytest.approx(0.4)
    assert doorway == {
        "correction": global_correction,
        "count": 1,
        "rank_one_based": artifact["global"]["rank_one_based"],
        "residual_min": artifact["global"]["residual_min"],
        "residual_max": artifact["global"]["residual_max"],
        "fallback": True,
        "fallback_reason": "group_count_below_minimum:1<2",
    }

    calibrated, decisions = apply_grouped_calibration(
        rows,
        artifact,
        prediction_key="q90",
        dimension="blind_type",
    )
    assert calibrated.shape == (3,)
    assert decisions[2]["fallback"] is True


def test_grouped_calibration_requires_one_dimension_at_application_time() -> None:
    rows = [_row("a", 0.8, 0.4), _row("b", 0.7, 0.4)]
    artifact = fit_grouped_calibration(
        rows,
        alpha=0.5,
        prediction_key="q90",
        dimensions=("blind_type", "target_object_type"),
        min_group_size=1,
    )
    with pytest.raises(CalibrationContractError, match="one dimension"):
        apply_grouped_calibration(
            rows,
            artifact,
            prediction_key="q90",
            dimension=("blind_type", "target_object_type"),
        )
