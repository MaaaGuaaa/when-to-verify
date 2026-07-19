"""Pure NumPy, JSON-safe risk metrics and failure-subset reporting."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.datasets.toy_risk_learning import TOY_MANIFEST_ROW_KEYS


def _metric(
    value: float | None,
    *,
    count: int,
    reason: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    if value is not None:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("metric value must be finite")
        value = number
        reason = None
    result: dict[str, Any] = {"value": value, "reason": reason, "count": int(count)}
    result.update(details)
    return result


def _paired_vectors(y_true: Any, prediction: Any) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=np.float64)
    values = np.asarray(prediction, dtype=np.float64)
    if labels.ndim != 1 or values.ndim != 1 or labels.shape != values.shape:
        raise ValueError("targets and predictions must be same-shape one-dimensional vectors")
    if not np.isfinite(labels).all() or not np.isfinite(values).all():
        raise ValueError("targets and predictions must be finite")
    return labels, values


def _binary_probability_vectors(
    y_true: Any, probability: Any
) -> tuple[np.ndarray, np.ndarray]:
    labels, probabilities = _paired_vectors(y_true, probability)
    if labels.size and not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("binary labels must contain only 0 and 1")
    if probabilities.size and (
        np.any(probabilities < 0.0) or np.any(probabilities > 1.0)
    ):
        raise ValueError("probabilities must be in [0, 1]")
    return labels.astype(np.int64), probabilities


def auroc(y_true: Any, score: Any) -> dict[str, Any]:
    """Tie-correct AUROC using average ranks in ``O(n log n)`` time."""

    labels, scores = _binary_probability_vectors(y_true, score)
    count = int(labels.size)
    if count == 0:
        return _metric(None, count=0, reason="empty_input")
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return _metric(None, count=count, reason="requires_both_binary_classes")
    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    ranks = np.empty(count, dtype=np.float64)
    cursor = 0
    while cursor < count:
        next_cursor = cursor + 1
        while (
            next_cursor < count
            and ordered_scores[next_cursor] == ordered_scores[cursor]
        ):
            next_cursor += 1
        # Ranks are one-based. Tied observations receive their average rank.
        average_rank = 0.5 * ((cursor + 1) + next_cursor)
        ranks[order[cursor:next_cursor]] = average_rank
        cursor = next_cursor
    positive_count = int(positive.size)
    negative_count = int(negative.size)
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    value = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return _metric(
        value,
        count=count,
        positive_count=positive_count,
        negative_count=negative_count,
    )


def _precision_recall_curve(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positive_count = int(np.sum(labels == 1))
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    recalls = [0.0]
    precisions = [1.0]
    true_positive = 0
    false_positive = 0
    cursor = 0
    while cursor < ordered_labels.size:
        next_cursor = cursor + 1
        while (
            next_cursor < ordered_labels.size
            and ordered_scores[next_cursor] == ordered_scores[cursor]
        ):
            next_cursor += 1
        group = ordered_labels[cursor:next_cursor]
        true_positive += int(np.sum(group == 1))
        false_positive += int(np.sum(group == 0))
        recalls.append(true_positive / positive_count)
        precisions.append(true_positive / (true_positive + false_positive))
        cursor = next_cursor
    return np.asarray(recalls), np.asarray(precisions)


def trapezoidal_auprc(y_true: Any, score: Any) -> dict[str, Any]:
    """Trapezoidal area under the threshold-grouped precision-recall curve."""

    labels, scores = _binary_probability_vectors(y_true, score)
    count = int(labels.size)
    if count == 0:
        return _metric(None, count=0, reason="empty_input")
    if not np.any(labels == 1):
        return _metric(None, count=count, reason="requires_positive_class")
    recall, precision = _precision_recall_curve(labels, scores)
    value = float(np.trapz(precision, recall))
    return _metric(value, count=count, curve_point_count=int(recall.size))


def average_precision(y_true: Any, score: Any) -> dict[str, Any]:
    """Step-wise average precision, reported separately from trapezoidal AUPRC."""

    labels, scores = _binary_probability_vectors(y_true, score)
    count = int(labels.size)
    if count == 0:
        return _metric(None, count=0, reason="empty_input")
    if not np.any(labels == 1):
        return _metric(None, count=count, reason="requires_positive_class")
    recall, precision = _precision_recall_curve(labels, scores)
    value = float(np.sum(np.diff(recall) * precision[1:]))
    return _metric(value, count=count, curve_point_count=int(recall.size))


def brier_score(y_true: Any, probability: Any) -> dict[str, Any]:
    labels, probabilities = _binary_probability_vectors(y_true, probability)
    if labels.size == 0:
        return _metric(None, count=0, reason="empty_input")
    return _metric(float(np.mean((probabilities - labels) ** 2)), count=labels.size)


def binary_nll(
    y_true: Any, probability: Any, *, clip_epsilon: float = 1e-7
) -> dict[str, Any]:
    labels, probabilities = _binary_probability_vectors(y_true, probability)
    if labels.size == 0:
        return _metric(None, count=0, reason="empty_input")
    if not 0.0 < clip_epsilon < 0.5:
        raise ValueError("clip_epsilon must be in (0, 0.5)")
    clipped = np.clip(probabilities, clip_epsilon, 1.0 - clip_epsilon)
    value = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    return _metric(float(value), count=labels.size, clip_epsilon=clip_epsilon)


def equal_width_ece(
    y_true: Any, probability: Any, *, n_bins: int = 10
) -> dict[str, Any]:
    labels, probabilities = _binary_probability_vectors(y_true, probability)
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    if labels.size == 0:
        return _metric(None, count=0, reason="empty_input", bin_count=n_bins)
    bin_ids = np.minimum((probabilities * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    nonempty_bins = 0
    for bin_id in range(n_bins):
        selected = bin_ids == bin_id
        if not np.any(selected):
            continue
        nonempty_bins += 1
        weight = float(np.mean(selected))
        ece += weight * abs(
            float(np.mean(probabilities[selected])) - float(np.mean(labels[selected]))
        )
    return _metric(
        ece,
        count=labels.size,
        bin_count=n_bins,
        nonempty_bin_count=nonempty_bins,
    )


def quantile_coverage(y_true: Any, upper_bound: Any) -> dict[str, Any]:
    target, upper = _paired_vectors(y_true, upper_bound)
    if target.size == 0:
        return _metric(None, count=0, reason="empty_input")
    if np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("risk severity targets must be in [0, 1]")
    if np.any(upper < 0.0) or np.any(upper > 1.0):
        raise ValueError("upper bounds must be in [0, 1]")
    return _metric(float(np.mean(target <= upper)), count=target.size)


def upper_bound_tightness(y_true: Any, upper_bound: Any) -> dict[str, Any]:
    target, upper = _paired_vectors(y_true, upper_bound)
    if target.size == 0:
        undefined = _metric(None, count=0, reason="empty_input")
        return {
            "mean_nonnegative_excess": dict(undefined),
            "mean_upper_bound": dict(undefined),
        }
    if np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("risk severity targets must be in [0, 1]")
    if np.any(upper < 0.0) or np.any(upper > 1.0):
        raise ValueError("upper bounds must be in [0, 1]")
    return {
        "mean_nonnegative_excess": _metric(
            float(np.mean(np.maximum(0.0, upper - target))), count=target.size
        ),
        "mean_upper_bound": _metric(float(np.mean(upper)), count=target.size),
    }


def false_safe_rate(
    collision_label: Any,
    probability: Any,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    labels, probabilities = _binary_probability_vectors(collision_label, probability)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    collisions = labels == 1
    collision_count = int(np.sum(collisions))
    if collision_count == 0:
        return _metric(
            None,
            count=labels.size,
            reason="no_true_collisions",
            false_safe_count=0,
            collision_count=0,
            threshold=float(threshold),
        )
    false_safe_count = int(np.sum(collisions & (probabilities < threshold)))
    return _metric(
        false_safe_count / collision_count,
        count=labels.size,
        false_safe_count=false_safe_count,
        collision_count=collision_count,
        threshold=float(threshold),
    )


def pairwise_ordering_accuracy(
    rows: Sequence[Mapping[str, Any]], *, prediction_key: str = "q90"
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_group_id"])].append(row)
    correct_credit = 0.0
    eligible_count = 0
    tie_count = 0
    prediction_tie_count = 0
    missing_group_count = 0
    ineligible_count = 0
    for group_rows in grouped.values():
        if len(group_rows) < 2:
            missing_group_count += 1
            continue
        for left, right in itertools.combinations(group_rows, 2):
            if not bool(left.get("pair_eligible", False)) or not bool(
                right.get("pair_eligible", False)
            ):
                ineligible_count += 1
                continue
            left_target = float(left["risk_severity"])
            right_target = float(right["risk_severity"])
            if not math.isfinite(left_target) or not math.isfinite(right_target):
                raise ValueError("pairwise severity must be finite")
            if left_target == right_target:
                tie_count += 1
                continue
            left_prediction = float(left[prediction_key])
            right_prediction = float(right[prediction_key])
            if not math.isfinite(left_prediction) or not math.isfinite(right_prediction):
                raise ValueError("pairwise prediction must be finite")
            eligible_count += 1
            product = (left_target - right_target) * (
                left_prediction - right_prediction
            )
            if product > 0.0:
                correct_credit += 1.0
            elif product == 0.0:
                prediction_tie_count += 1
                correct_credit += 0.5
    if eligible_count == 0:
        return _metric(
            None,
            count=len(rows),
            reason="no_eligible_non_tied_pairs",
            eligible_pair_count=0,
            severity_tie_count=tie_count,
            prediction_tie_count=prediction_tie_count,
            missing_pair_group_count=missing_group_count,
            ineligible_pair_count=ineligible_count,
        )
    return _metric(
        correct_credit / eligible_count,
        count=len(rows),
        eligible_pair_count=eligible_count,
        severity_tie_count=tie_count,
        prediction_tie_count=prediction_tie_count,
        missing_pair_group_count=missing_group_count,
        ineligible_pair_count=ineligible_count,
    )


def clearance_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    included: list[float] = []
    excluded = 0
    for row in rows:
        if row.get("critical_object_id") is None:
            excluded += 1
            continue
        value = float(row["min_clearance"])
        if not math.isfinite(value):
            raise ValueError("min_clearance must be finite")
        included.append(value)
    if not included:
        undefined = _metric(None, count=0, reason="no_rows_with_critical_object")
        return {
            "included_count": 0,
            "excluded_no_object_count": excluded,
            "mean": dict(undefined),
            "minimum": dict(undefined),
            "maximum": dict(undefined),
        }
    array = np.asarray(included, dtype=np.float64)
    return {
        "included_count": len(included),
        "excluded_no_object_count": excluded,
        "mean": _metric(float(np.mean(array)), count=len(included)),
        "minimum": _metric(float(np.min(array)), count=len(included)),
        "maximum": _metric(float(np.max(array)), count=len(included)),
    }


def _core_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str,
    upper_key: str,
) -> dict[str, Any]:
    labels = np.asarray([row["collision_label"] for row in rows], dtype=np.float64)
    probabilities = np.asarray([row[probability_key] for row in rows], dtype=np.float64)
    severity = np.asarray([row["risk_severity"] for row in rows], dtype=np.float64)
    upper = np.asarray([row[upper_key] for row in rows], dtype=np.float64)
    if probabilities.size and not np.isfinite(probabilities).all():
        raise ValueError("prediction probabilities must be finite")
    return {
        "sample_count": len(rows),
        "classification": {
            "auroc": auroc(labels, probabilities),
            "trapezoidal_auprc": trapezoidal_auprc(labels, probabilities),
            "average_precision": average_precision(labels, probabilities),
            "brier": brier_score(labels, probabilities),
            "nll": binary_nll(labels, probabilities),
            "ece": equal_width_ece(labels, probabilities, n_bins=10),
            "false_safe": false_safe_rate(labels, probabilities, threshold=0.5),
        },
        "severity": {
            "coverage": quantile_coverage(severity, upper),
            "tightness": upper_bound_tightness(severity, upper),
        },
        "pairwise_ordering": pairwise_ordering_accuracy(
            rows, prediction_key=upper_key
        ),
        "clearance": clearance_summary(rows),
    }


def _normalized_event(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def evaluate_risk_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str = "p_collision",
    upper_key: str = "calibrated_upper",
) -> dict[str, Any]:
    """Evaluate overall risk metrics and every mandatory failure subset."""

    rows_list = list(rows)
    overall = _core_report(
        rows_list, probability_key=probability_key, upper_key=upper_key
    )
    subset_predicates = {
        "temporal_safe": lambda row: _normalized_event(row.get("event_type"))
        == "temporal_safe",
        "same_area": lambda row: _normalized_event(row.get("event_type"))
        in {"same_area", "same_area_safe", "spatial_safe"},
        "irrelevant_hidden": lambda row: _normalized_event(row.get("event_type"))
        == "irrelevant_hidden",
        "empty": lambda row: _normalized_event(row.get("event_type")) == "empty",
        "ood": lambda row: _normalized_event(row.get("ood_tag"))
        not in {"", "none", "id", "in_distribution"},
    }
    overall["subsets"] = {
        name: _core_report(
            [row for row in rows_list if predicate(row)],
            probability_key=probability_key,
            upper_key=upper_key,
        )
        for name, predicate in subset_predicates.items()
    }
    overall["definitions"] = {
        "nll_clip_epsilon": 1e-7,
        "ece_bins": 10,
        "ece_binning": "equal_width_[0,1]_final_bin_inclusive",
        "false_safe": "p_collision<0.5 among all true-collision rows",
        "false_safe_scope": "raw_collision_head_probability_not_conformal",
        "false_safe_calibration_note": (
            "split conformal adjusts severity upper bounds only; false-safe does "
            "not measure a conformal improvement"
        ),
        "tightness": "mean(max(0, calibrated_upper-risk_severity))",
        "pairwise": "higher predicted upper bound should match higher severity",
        "no_object_clearance": "excluded_only_from_clearance_aggregation",
    }
    return overall


def compare_risk_methods(
    main_rows: Sequence[Mapping[str, Any]],
    occupancy_baseline_rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str = "p_collision",
) -> dict[str, Any]:
    """Compare methods without selective filtering of the shared test rows.

    Calibration provenance equality is checked by the CLI before this function
    is called.  Here, the row identities and labels are required to match
    exactly, so a favorable subset cannot be silently removed from either side.
    """

    main_by_id = {str(row["sample_id"]): row for row in main_rows}
    baseline_by_id = {
        str(row["sample_id"]): row for row in occupancy_baseline_rows
    }
    if len(main_by_id) != len(main_rows) or len(baseline_by_id) != len(
        occupancy_baseline_rows
    ):
        raise ValueError("method comparison requires unique sample_id values")
    if set(main_by_id) != set(baseline_by_id):
        raise ValueError("method comparison requires identical sample_id sets")
    ordered_ids = sorted(main_by_id)
    label_fields = tuple(sorted(TOY_MANIFEST_ROW_KEYS - {"sample_id"}))
    for sample_id in ordered_ids:
        for field in label_fields:
            if main_by_id[sample_id].get(field) != baseline_by_id[sample_id].get(field):
                raise ValueError(
                    f"method comparison label mismatch for sample {sample_id!r} "
                    f"field {field!r}"
                )
    main_ordered = [main_by_id[sample_id] for sample_id in ordered_ids]
    baseline_ordered = [baseline_by_id[sample_id] for sample_id in ordered_ids]
    labels = np.asarray(
        [row["collision_label"] for row in main_ordered], dtype=np.float64
    )
    main_probability = np.asarray(
        [row[probability_key] for row in main_ordered], dtype=np.float64
    )
    baseline_probability = np.asarray(
        [row[probability_key] for row in baseline_ordered], dtype=np.float64
    )
    main_false_safe = false_safe_rate(labels, main_probability)
    baseline_false_safe = false_safe_rate(labels, baseline_probability)
    main_value = main_false_safe["value"]
    baseline_value = baseline_false_safe["value"]
    if main_value is None or baseline_value is None:
        reduction = _metric(
            None,
            count=len(ordered_ids),
            reason="false_safe_undefined_for_one_or_both_methods",
        )
    elif baseline_value == 0.0:
        reduction = _metric(
            None,
            count=len(ordered_ids),
            reason="occupancy_baseline_false_safe_is_zero",
        )
    else:
        reduction = _metric(
            (float(baseline_value) - float(main_value)) / float(baseline_value),
            count=len(ordered_ids),
        )

    hard_negative_events = {
        "temporal_safe": {"temporal_safe"},
        "same_area": {"same_area", "same_area_safe", "spatial_safe"},
        "irrelevant_hidden": {"irrelevant_hidden"},
        "empty": {"empty"},
    }
    hard_negative_subsets: dict[str, Any] = {}
    better_count = 0
    for name, events in hard_negative_events.items():
        selected_ids = [
            sample_id
            for sample_id in ordered_ids
            if _normalized_event(main_by_id[sample_id]["event_type"]) in events
        ]
        if not selected_ids:
            undefined = _metric(None, count=0, reason="empty_subset")
            hard_negative_subsets[name] = {
                "sample_count": 0,
                "main_mean_probability": dict(undefined),
                "occupancy_baseline_mean_probability": dict(undefined),
                "mean_probability_improvement": dict(undefined),
                "main_is_better": None,
            }
            continue
        main_mean = float(
            np.mean([main_by_id[sample_id][probability_key] for sample_id in selected_ids])
        )
        baseline_mean = float(
            np.mean(
                [
                    baseline_by_id[sample_id][probability_key]
                    for sample_id in selected_ids
                ]
            )
        )
        improvement = baseline_mean - main_mean
        main_is_better = improvement > 0.0
        if main_is_better:
            better_count += 1
        hard_negative_subsets[name] = {
            "sample_count": len(selected_ids),
            "main_mean_probability": _metric(main_mean, count=len(selected_ids)),
            "occupancy_baseline_mean_probability": _metric(
                baseline_mean, count=len(selected_ids)
            ),
            "mean_probability_improvement": _metric(
                improvement, count=len(selected_ids)
            ),
            "main_is_better": main_is_better,
        }
    return {
        "sample_identity_match": True,
        "sample_count": len(ordered_ids),
        "main_false_safe": main_false_safe,
        "occupancy_baseline_false_safe": baseline_false_safe,
        "false_safe_relative_reduction": reduction,
        "false_safe_scope": "raw_collision_head_probability_not_conformal",
        "false_safe_calibration_note": (
            "comparison uses raw p_collision for both methods and is not a "
            "conformal-calibration improvement claim"
        ),
        "hard_negative_subsets": hard_negative_subsets,
        "hard_negative_better_count": better_count,
    }
