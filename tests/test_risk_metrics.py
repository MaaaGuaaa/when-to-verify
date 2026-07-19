from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.risk_metrics import (
    average_precision,
    binary_nll,
    brier_score,
    clearance_summary,
    compare_risk_methods,
    equal_width_ece,
    evaluate_risk_rows,
    false_safe_rate,
    pairwise_ordering_accuracy,
    quantile_coverage,
    trapezoidal_auprc,
    upper_bound_tightness,
    auroc,
)


def _value(result: dict) -> float:
    value = result["value"]
    assert isinstance(value, float)
    return value


def _row(sample_id: str, **overrides: object) -> dict:
    row = {
        "sample_id": sample_id,
        "split": "test",
        "pair_group_id": f"pair-{sample_id}",
        "event_type": "collision",
        "recording_id": f"recording-{sample_id}",
        "session_id": f"session-{sample_id}",
        "source_object_id": f"object-{sample_id}",
        "snippet_id": f"snippet-{sample_id}",
        "base_state_id": f"base-{sample_id}",
        "seed_namespace": f"seed-{sample_id}",
        "collision_label": 1,
        "risk_severity": 0.8,
        "min_clearance": 0.0,
        "critical_object_id": f"critical-{sample_id}",
        "p_collision": 0.8,
        "q50": 0.4,
        "q80": 0.6,
        "q90": 0.7,
        "q95": 0.9,
        "calibrated_upper": 0.9,
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


def test_binary_ranking_metrics_match_hand_calculation() -> None:
    labels = np.asarray([1, 0, 1])
    scores = np.asarray([0.9, 0.8, 0.7])

    assert _value(auroc(labels, scores)) == pytest.approx(0.5)
    assert _value(average_precision(labels, scores)) == pytest.approx(5.0 / 6.0)
    assert _value(trapezoidal_auprc(labels, scores)) == pytest.approx(19.0 / 24.0)
    assert _value(trapezoidal_auprc(labels, scores)) != _value(
        average_precision(labels, scores)
    )


def test_binary_metrics_return_null_with_reason_when_undefined() -> None:
    result = auroc(np.asarray([1, 1]), np.asarray([0.2, 0.8]))
    assert result["value"] is None
    assert result["reason"] == "requires_both_binary_classes"

    empty = brier_score(np.asarray([]), np.asarray([]))
    assert empty["value"] is None
    assert empty["reason"] == "empty_input"


def test_probability_metrics_match_hand_calculation() -> None:
    labels = np.asarray([0, 1])
    probabilities = np.asarray([0.2, 0.8])

    assert _value(brier_score(labels, probabilities)) == pytest.approx(0.04)
    assert _value(binary_nll(labels, probabilities)) == pytest.approx(-math.log(0.8))
    assert _value(equal_width_ece(labels, probabilities, n_bins=2)) == pytest.approx(
        0.2
    )


def test_ece_final_bin_includes_probability_one() -> None:
    result = equal_width_ece(
        np.asarray([1, 0]), np.asarray([1.0, 0.0]), n_bins=10
    )
    assert _value(result) == pytest.approx(0.0)
    assert result["bin_count"] == 10


def test_coverage_tightness_and_false_safe_match_frozen_definitions() -> None:
    severity = np.asarray([0.2, 0.8])
    upper = np.asarray([0.3, 0.7])
    coverage = quantile_coverage(severity, upper)
    tightness = upper_bound_tightness(severity, upper)
    false_safe = false_safe_rate(
        np.asarray([1, 1, 0]), np.asarray([0.2, 0.8, 0.1])
    )

    assert _value(coverage) == pytest.approx(0.5)
    assert _value(tightness["mean_nonnegative_excess"]) == pytest.approx(0.05)
    assert _value(tightness["mean_upper_bound"]) == pytest.approx(0.5)
    assert _value(false_safe) == pytest.approx(0.5)
    assert false_safe["false_safe_count"] == 1
    assert false_safe["collision_count"] == 2


def test_false_safe_is_undefined_without_true_collisions() -> None:
    result = false_safe_rate(np.asarray([0, 0]), np.asarray([0.2, 0.8]))
    assert result["value"] is None
    assert result["reason"] == "no_true_collisions"


def test_pairwise_ordering_reports_eligible_ties_missing_and_ineligible() -> None:
    rows = [
        _row("a0", pair_group_id="a", risk_severity=0.9, q90=0.8),
        _row("a1", pair_group_id="a", risk_severity=0.1, q90=0.2),
        _row("b0", pair_group_id="b", risk_severity=0.5, q90=0.8),
        _row("b1", pair_group_id="b", risk_severity=0.5, q90=0.2),
        _row("c0", pair_group_id="c", risk_severity=0.7, q90=0.6),
        _row("d0", pair_group_id="d", risk_severity=0.9, q90=0.9),
        _row(
            "d1",
            pair_group_id="d",
            risk_severity=0.1,
            q90=0.1,
            pair_eligible=False,
        ),
    ]

    result = pairwise_ordering_accuracy(rows, prediction_key="q90")

    assert _value(result) == pytest.approx(1.0)
    assert result["eligible_pair_count"] == 1
    assert result["severity_tie_count"] == 1
    assert result["missing_pair_group_count"] == 1
    assert result["ineligible_pair_count"] == 1


def test_no_object_rows_stay_in_risk_metrics_but_not_clearance() -> None:
    rows = [
        _row("object", collision_label=1, critical_object_id="obj", min_clearance=0.0),
        _row(
            "none",
            collision_label=0,
            risk_severity=0.0,
            p_collision=0.1,
            critical_object_id=None,
            min_clearance=22.627,
            event_type="empty",
        ),
    ]

    clearance = clearance_summary(rows)
    report = evaluate_risk_rows(rows)

    assert clearance["included_count"] == 1
    assert clearance["excluded_no_object_count"] == 1
    assert report["sample_count"] == 2
    assert report["clearance"]["included_count"] == 1
    assert report["classification"]["auroc"]["value"] == pytest.approx(1.0)


def test_failure_subsets_are_always_reported_with_json_safe_undefined_values() -> None:
    rows = [_row("only", event_type="collision", ood_tag="in_distribution")]

    report = evaluate_risk_rows(rows)

    assert set(report["subsets"]) == {
        "temporal_safe",
        "same_area",
        "irrelevant_hidden",
        "empty",
        "ood",
    }
    assert report["subsets"]["ood"]["sample_count"] == 0
    assert report["subsets"]["ood"]["classification"]["auroc"] == {
        "value": None,
        "reason": "empty_input",
        "count": 0,
    }


def test_probability_metrics_reject_out_of_range_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score(np.asarray([0]), np.asarray([1.1]))
    with pytest.raises(ValueError, match="finite"):
        quantile_coverage(np.asarray([0.2]), np.asarray([np.nan]))


def test_main_and_occupancy_comparison_uses_identical_rows_and_reports_gains() -> None:
    main = [
        _row("c0", collision_label=1, p_collision=0.8),
        _row("c1", collision_label=1, p_collision=0.8),
        _row(
            "safe",
            collision_label=0,
            risk_severity=0.0,
            p_collision=0.1,
            event_type="temporal_safe",
        ),
    ]
    baseline = [
        _row("c0", collision_label=1, p_collision=0.2),
        _row("c1", collision_label=1, p_collision=0.8),
        _row(
            "safe",
            collision_label=0,
            risk_severity=0.0,
            p_collision=0.6,
            event_type="temporal_safe",
        ),
    ]

    report = compare_risk_methods(main, baseline)

    assert report["sample_identity_match"] is True
    assert report["false_safe_relative_reduction"]["value"] == pytest.approx(1.0)
    temporal = report["hard_negative_subsets"]["temporal_safe"]
    assert temporal["mean_probability_improvement"]["value"] == pytest.approx(0.5)
    assert report["hard_negative_better_count"] >= 1


def test_method_comparison_rejects_selective_sample_filtering() -> None:
    with pytest.raises(ValueError, match="identical sample_id"):
        compare_risk_methods([_row("a")], [_row("b")])
