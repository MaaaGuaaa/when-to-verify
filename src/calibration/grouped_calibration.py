"""One-dimension-at-a-time grouped conformal calibration."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.calibration.split_conformal import (
    CalibrationContractError,
    apply_split_conformal,
    finite_sample_conformal_quantile,
    one_sided_residuals,
)


GROUP_DIMENSIONS: tuple[str, ...] = (
    "blind_type",
    "critical_area_fraction",
    "age_s",
    "density_fraction",
    "target_object_type",
    "footprint_kind",
)

CONTINUOUS_GROUP_BINS: dict[str, tuple[float, ...]] = {
    "critical_area_fraction": (0.0, 0.05, 0.20, 1.0),
    "age_s": (0.0, 1.0, 3.0, 5.0),
    "density_fraction": (0.0, 0.01, 0.05, 1.0),
}
GROUPED_CALIBRATION_LAYOUT_VERSION = "grouped_risk_calibration_v2"


def _format_edge(value: float) -> str:
    return f"{value:g}"


def group_value(dimension: str, value: Any) -> str:
    """Return a stable categorical key using the manifest-frozen toy bins."""

    if dimension not in GROUP_DIMENSIONS:
        raise CalibrationContractError(f"unsupported calibration group {dimension!r}")
    if dimension not in CONTINUOUS_GROUP_BINS:
        if value is None:
            return "missing"
        if not isinstance(value, str) or not value:
            raise CalibrationContractError(
                f"categorical group {dimension} must be a non-empty string or null"
            )
        return value
    if isinstance(value, (bool, np.bool_)):
        raise CalibrationContractError(f"continuous group {dimension} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationContractError(
            f"continuous group {dimension} must be numeric"
        ) from exc
    if not math.isfinite(number):
        raise CalibrationContractError(f"continuous group {dimension} must be finite")
    edges = CONTINUOUS_GROUP_BINS[dimension]
    if number < edges[0] or number > edges[-1]:
        return "out_of_range"
    for index, (left, right) in enumerate(zip(edges, edges[1:])):
        is_final = index == len(edges) - 2
        if left <= number < right or (is_final and number == right):
            close = "]" if is_final else ")"
            return f"[{_format_edge(left)},{_format_edge(right)}{close}"
    raise AssertionError("unreachable continuous group bin")


def _fit_values(
    rows: Sequence[Mapping[str, Any]], *, alpha: float, prediction_key: str
) -> dict[str, Any]:
    target = np.asarray([row["risk_severity"] for row in rows], dtype=np.float64)
    predicted = np.asarray([row[prediction_key] for row in rows], dtype=np.float64)
    residuals = one_sided_residuals(target, predicted)
    correction, rank = finite_sample_conformal_quantile(residuals, alpha=alpha)
    return {
        "correction": correction,
        "count": int(len(rows)),
        "rank_one_based": rank,
        "residual_min": float(np.min(residuals)),
        "residual_max": float(np.max(residuals)),
    }


def fit_grouped_calibration(
    rows: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    prediction_key: str,
    dimensions: Sequence[str] = GROUP_DIMENSIONS,
    min_group_size: int = 20,
) -> dict[str, Any]:
    """Fit global and independent one-dimensional corrections.

    Sparse groups copy the global correction and explicitly record why.  The
    returned artifact never combines corrections across dimensions.
    """

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise CalibrationContractError("rows must be a non-empty sequence")
    if not isinstance(min_group_size, int) or isinstance(min_group_size, bool):
        raise CalibrationContractError("min_group_size must be an integer")
    if min_group_size <= 0:
        raise CalibrationContractError("min_group_size must be positive")
    dimension_tuple = tuple(dimensions)
    if not dimension_tuple:
        raise CalibrationContractError("at least one grouped dimension is required")
    if len(set(dimension_tuple)) != len(dimension_tuple):
        raise CalibrationContractError("group dimensions must be unique")
    for dimension in dimension_tuple:
        if dimension not in GROUP_DIMENSIONS:
            raise CalibrationContractError(
                f"unsupported calibration group {dimension!r}"
            )
        if any(dimension not in row for row in rows):
            raise CalibrationContractError(
                f"calibration rows are missing group field {dimension}"
            )
        if any(prediction_key not in row or "risk_severity" not in row for row in rows):
            raise CalibrationContractError(
                f"calibration rows require {prediction_key} and risk_severity"
            )
    global_fit = _fit_values(rows, alpha=alpha, prediction_key=prediction_key)
    global_correction = float(global_fit["correction"])
    dimensions_result: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension in dimension_tuple:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[group_value(dimension, row[dimension])].append(row)
        fitted: dict[str, dict[str, Any]] = {}
        for key in sorted(grouped):
            group_rows = grouped[key]
            if len(group_rows) < min_group_size:
                fitted[key] = {
                    "correction": global_correction,
                    "count": len(group_rows),
                    "rank_one_based": int(global_fit["rank_one_based"]),
                    "residual_min": float(global_fit["residual_min"]),
                    "residual_max": float(global_fit["residual_max"]),
                    "fallback": True,
                    "fallback_reason": (
                        f"group_count_below_minimum:{len(group_rows)}<{min_group_size}"
                    ),
                }
            else:
                group_fit = _fit_values(
                    group_rows, alpha=alpha, prediction_key=prediction_key
                )
                fitted[key] = {
                    "correction": float(group_fit["correction"]),
                    "count": len(group_rows),
                    "rank_one_based": int(group_fit["rank_one_based"]),
                    "residual_min": float(group_fit["residual_min"]),
                    "residual_max": float(group_fit["residual_max"]),
                    "fallback": False,
                    "fallback_reason": None,
                }
        dimensions_result[dimension] = fitted
    return {
        "layout_version": GROUPED_CALIBRATION_LAYOUT_VERSION,
        "alpha": float(alpha),
        "prediction_key": prediction_key,
        "min_group_size": min_group_size,
        "group_dimensions": list(dimension_tuple),
        "continuous_group_bins": {
            key: list(value) for key, value in CONTINUOUS_GROUP_BINS.items()
        },
        "global": global_fit,
        "dimensions": dimensions_result,
        "combination_policy": "one_dimension_at_a_time",
    }


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise CalibrationContractError(f"{name} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationContractError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise CalibrationContractError(f"{name} must be finite")
    return number


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CalibrationContractError(f"{name} must be a positive integer")
    return int(value)


def validate_grouped_calibration_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_alpha: float | None = None,
    expected_prediction_key: str | None = None,
    expected_global: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate grouped structure and cross-field numerical invariants.

    The semantic digest detects accidental byte-level changes.  This validator
    separately rejects a malformed payload even when that digest is recomputed.
    """

    if not isinstance(artifact, Mapping):
        raise CalibrationContractError("grouped calibration artifact must be a mapping")
    result = copy.deepcopy(dict(artifact))
    required_fields = {
        "layout_version",
        "alpha",
        "prediction_key",
        "min_group_size",
        "group_dimensions",
        "continuous_group_bins",
        "global",
        "dimensions",
        "combination_policy",
    }
    if set(result) != required_fields:
        missing = sorted(required_fields - set(result))
        unexpected = sorted(set(result) - required_fields)
        raise CalibrationContractError(
            "grouped artifact fields are invalid: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if result["layout_version"] != GROUPED_CALIBRATION_LAYOUT_VERSION:
        raise CalibrationContractError("grouped artifact layout version is invalid")
    if result["combination_policy"] != "one_dimension_at_a_time":
        raise CalibrationContractError("grouped artifact combination policy is invalid")

    alpha = _finite_number(result["alpha"], "grouped alpha")
    if not 0.0 < alpha < 1.0:
        raise CalibrationContractError("grouped alpha must be strictly between 0 and 1")
    if expected_alpha is not None and alpha != float(expected_alpha):
        raise CalibrationContractError("grouped alpha does not match global artifact alpha")
    prediction_key = result["prediction_key"]
    if prediction_key not in {"q50", "q80", "q90", "q95"}:
        raise CalibrationContractError("grouped prediction_key is invalid")
    if (
        expected_prediction_key is not None
        and prediction_key != expected_prediction_key
    ):
        raise CalibrationContractError(
            "grouped prediction_key does not match global artifact prediction_key"
        )
    min_group_size = _positive_integer(
        result["min_group_size"], "grouped min_group_size"
    )

    group_dimensions = result["group_dimensions"]
    if (
        not isinstance(group_dimensions, list)
        or not group_dimensions
        or any(not isinstance(value, str) for value in group_dimensions)
        or len(set(group_dimensions)) != len(group_dimensions)
        or any(value not in GROUP_DIMENSIONS for value in group_dimensions)
    ):
        raise CalibrationContractError("grouped group_dimensions are invalid")
    expected_bins = {
        key: list(value) for key, value in CONTINUOUS_GROUP_BINS.items()
    }
    if result["continuous_group_bins"] != expected_bins:
        raise CalibrationContractError("grouped continuous_group_bins have drifted")

    global_fit = result["global"]
    if not isinstance(global_fit, Mapping) or set(global_fit) != {
        "correction",
        "count",
        "rank_one_based",
        "residual_min",
        "residual_max",
    }:
        raise CalibrationContractError("grouped global fit structure is invalid")
    global_correction = _finite_number(
        global_fit["correction"], "grouped global correction"
    )
    if not 0.0 <= global_correction <= 1.0:
        raise CalibrationContractError("grouped global correction must be in [0, 1]")
    global_count = _positive_integer(global_fit["count"], "grouped global count")
    global_rank = _positive_integer(
        global_fit["rank_one_based"], "grouped global rank_one_based"
    )
    required_rank = min(
        global_count, int(math.ceil((global_count + 1) * (1.0 - alpha)))
    )
    if global_rank != required_rank:
        raise CalibrationContractError(
            f"grouped global rank_one_based must equal {required_rank}"
        )
    global_residual_min = _finite_number(
        global_fit["residual_min"], "grouped global residual_min"
    )
    global_residual_max = _finite_number(
        global_fit["residual_max"], "grouped global residual_max"
    )
    if not 0.0 <= global_residual_min <= global_residual_max <= 1.0:
        raise CalibrationContractError("grouped global residual bounds are invalid")
    if not global_residual_min <= global_correction <= global_residual_max:
        raise CalibrationContractError(
            "grouped global correction must lie within residual bounds"
        )
    if expected_global is not None:
        for field, observed in (
            ("correction", global_correction),
            ("count", global_count),
            ("rank_one_based", global_rank),
            ("residual_min", global_residual_min),
            ("residual_max", global_residual_max),
        ):
            if observed != expected_global.get(field):
                raise CalibrationContractError(
                    f"grouped global {field} does not match global calibration"
                )

    dimensions = result["dimensions"]
    if not isinstance(dimensions, Mapping) or set(dimensions) != set(group_dimensions):
        raise CalibrationContractError(
            "grouped dimensions must exactly match group_dimensions"
        )
    for dimension in group_dimensions:
        groups = dimensions[dimension]
        if not isinstance(groups, Mapping) or not groups:
            raise CalibrationContractError(
                f"grouped dimension {dimension!r} must contain fitted groups"
            )
        dimension_count = 0
        for group, fit in groups.items():
            if not isinstance(group, str) or not group:
                raise CalibrationContractError(
                    f"grouped dimension {dimension!r} has an invalid group key"
                )
            if not isinstance(fit, Mapping) or set(fit) != {
                "correction",
                "count",
                "rank_one_based",
                "residual_min",
                "residual_max",
                "fallback",
                "fallback_reason",
            }:
                raise CalibrationContractError(
                    f"grouped dimension {dimension!r} group {group!r} structure is invalid"
                )
            correction = _finite_number(
                fit["correction"],
                f"grouped dimension {dimension!r} group {group!r} correction",
            )
            if not 0.0 <= correction <= 1.0:
                raise CalibrationContractError(
                    f"grouped dimension {dimension!r} correction must be in [0, 1]"
                )
            count = _positive_integer(
                fit["count"],
                f"grouped dimension {dimension!r} group {group!r} count",
            )
            dimension_count += count
            rank = _positive_integer(
                fit["rank_one_based"],
                f"grouped dimension {dimension!r} group {group!r} rank_one_based",
            )
            residual_min = _finite_number(
                fit["residual_min"],
                f"grouped dimension {dimension!r} group {group!r} residual_min",
            )
            residual_max = _finite_number(
                fit["residual_max"],
                f"grouped dimension {dimension!r} group {group!r} residual_max",
            )
            if not 0.0 <= residual_min <= residual_max <= 1.0:
                raise CalibrationContractError(
                    f"grouped dimension {dimension!r} residual bounds are invalid"
                )
            fallback = fit["fallback"]
            if not isinstance(fallback, bool):
                raise CalibrationContractError(
                    f"grouped dimension {dimension!r} fallback must be boolean"
                )
            if count < min_group_size:
                expected_reason = (
                    f"group_count_below_minimum:{count}<{min_group_size}"
                )
                if (
                    not fallback
                    or fit["fallback_reason"] != expected_reason
                    or correction != global_correction
                    or rank != global_rank
                    or residual_min != global_residual_min
                    or residual_max != global_residual_max
                ):
                    raise CalibrationContractError(
                        f"grouped dimension {dimension!r} sparse fallback is inconsistent"
                    )
            else:
                expected_group_rank = min(
                    count, int(math.ceil((count + 1) * (1.0 - alpha)))
                )
                if fallback or fit["fallback_reason"] is not None:
                    raise CalibrationContractError(
                        f"grouped dimension {dimension!r} dense group cannot be fallback"
                    )
                if rank != expected_group_rank:
                    raise CalibrationContractError(
                        f"grouped dimension {dimension!r} dense group rank is invalid"
                    )
                if not residual_min <= correction <= residual_max:
                    raise CalibrationContractError(
                        f"grouped dimension {dimension!r} dense correction lacks residual evidence"
                    )
        if dimension_count != global_count:
            raise CalibrationContractError(
                f"grouped dimension {dimension!r} counts must sum to global count "
                f"{global_count}, got {dimension_count}"
            )
    return result


def apply_grouped_calibration(
    rows: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
    *,
    prediction_key: str,
    dimension: str | None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply either global or exactly one grouped correction."""

    if dimension is not None and not isinstance(dimension, str):
        raise CalibrationContractError("application accepts exactly one dimension")
    if artifact.get("combination_policy") != "one_dimension_at_a_time":
        raise CalibrationContractError("grouped artifact combination policy is invalid")
    if artifact.get("prediction_key") != prediction_key:
        raise CalibrationContractError("prediction_key does not match grouped artifact")
    global_fit = artifact.get("global")
    if not isinstance(global_fit, Mapping) or "correction" not in global_fit:
        raise CalibrationContractError("grouped artifact has no global correction")
    global_correction = float(global_fit["correction"])
    dimensions = artifact.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise CalibrationContractError("grouped artifact dimensions are invalid")
    if dimension is not None and dimension not in dimensions:
        raise CalibrationContractError(
            f"requested group dimension {dimension!r} was not fitted"
        )
    calibrated_values: list[float] = []
    decisions: list[dict[str, Any]] = []
    for row in rows:
        if prediction_key not in row:
            raise CalibrationContractError(
                f"application row is missing prediction field {prediction_key}"
            )
        if dimension is None:
            correction = global_correction
            decision = {
                "dimension": None,
                "group": "global",
                "correction": correction,
                "fallback": False,
                "fallback_reason": None,
            }
        else:
            if dimension not in row:
                raise CalibrationContractError(
                    f"application row is missing group field {dimension}"
                )
            key = group_value(dimension, row[dimension])
            group_fit = dimensions[dimension].get(key)
            if group_fit is None:
                correction = global_correction
                fallback = True
                reason = "group_not_fitted"
            else:
                correction = float(group_fit["correction"])
                fallback = bool(group_fit["fallback"])
                reason = group_fit["fallback_reason"]
            decision = {
                "dimension": dimension,
                "group": key,
                "correction": correction,
                "fallback": fallback,
                "fallback_reason": reason,
            }
        calibrated = apply_split_conformal(
            np.asarray([row[prediction_key]], dtype=np.float64),
            correction=correction,
        )[0]
        calibrated_values.append(float(calibrated))
        decisions.append(decision)
    return np.asarray(calibrated_values, dtype=np.float64), decisions
