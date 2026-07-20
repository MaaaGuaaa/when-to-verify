"""Split-conformal calibration contracts for trajectory-conditioned risk.

Only rows from the calibration split may be used to fit a correction.  The
module is intentionally NumPy-only and keeps toy and production provenance
separate: toy artifacts bind a toy dataset digest, while production artifacts
bind an authenticated four-split risk-dataset family.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.datasets.risk_dataloader import RiskDataContractError
from src.datasets.risk_dataset_seal import (
    RISK_DATASET_FAMILY_LAYOUT_VERSION,
    LoadedRiskDatasetFamily,
    load_risk_dataset_family,
    risk_dataset_family_sample_ids_digest,
)
from src.datasets.toy_risk_learning import (
    TOY_MANIFEST_ROW_KEYS,
    frozen_channel_spec,
)

PREDICTION_TABLE_LAYOUT_VERSION = "risk_prediction_table_v2"
PREDICTION_COHORT_LAYOUT_VERSION = "risk_prediction_cohort_v1"
CALIBRATION_ARTIFACT_LAYOUT_VERSION = "risk_calibration_v3"
RISK_CHECKPOINT_LAYOUT_VERSION = "risk_model_checkpoint_v2"
OCCUPANCY_CHECKPOINT_LAYOUT_VERSION = "occupancy_baseline_checkpoint_v2"
CHECKPOINT_LAYOUT_VERSIONS = frozenset(
    {RISK_CHECKPOINT_LAYOUT_VERSION, OCCUPANCY_CHECKPOINT_LAYOUT_VERSION}
)

IDENTITY_FIELDS: tuple[str, ...] = (
    "sample_id",
    "pair_group_id",
    "recording_id",
    "session_id",
    "source_object_id",
    "snippet_id",
    "base_state_id",
    "seed_namespace",
)

ROW_STRING_FIELDS: tuple[str, ...] = (
    "sample_id",
    "split",
    "pair_group_id",
    "event_type",
    "recording_id",
    "session_id",
    "source_object_id",
    "snippet_id",
    "base_state_id",
    "seed_namespace",
    "trajectory_id",
    "occluder_id",
    "background_id",
    "blind_type",
    "target_object_type",
    "footprint_kind",
    "robot_footprint_kind",
    "footprint_contact_policy",
    "ood_tag",
)

PREDICTION_FIELDS: tuple[str, ...] = (
    "p_collision",
    "q50",
    "q80",
    "q90",
    "q95",
)
ROW_REQUIRED_FIELDS: tuple[str, ...] = tuple(
    sorted(set(TOY_MANIFEST_ROW_KEYS) | set(PREDICTION_FIELDS))
)

_SEMANTIC_EXCLUDED_KEYS = frozenset(
    {
        "semantic_digest",
        "generated_at",
        "generated_at_utc",
        "hostname",
        "job_id",
        "slurm_job_id",
        "output_dir",
        "absolute_output_path",
        "full_file_checksums",
    }
)


class CalibrationContractError(ValueError):
    """Raised when prediction or calibration provenance is unsafe."""


def _semantic_projection(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_projection(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _SEMANTIC_EXCLUDED_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_projection(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _semantic_digest(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            _semantic_projection(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CalibrationContractError(
            f"artifact is not finite JSON-safe data: {exc}"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def prediction_table_semantic_digest(table: Mapping[str, Any]) -> str:
    """Return the runtime-metadata-independent digest of a prediction table."""

    return _semantic_digest(table)


def calibration_artifact_semantic_digest(artifact: Mapping[str, Any]) -> str:
    """Return the runtime-metadata-independent digest of a calibration artifact."""

    return _semantic_digest(artifact)


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise CalibrationContractError(f"{name} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationContractError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise CalibrationContractError(f"{name} must be finite")
    return number


def _probability(value: Any, name: str) -> float:
    number = _finite_scalar(value, name)
    if not 0.0 <= number <= 1.0:
        raise CalibrationContractError(f"{name} must be in [0, 1]")
    return number


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationContractError(f"{name} must be a non-empty string")
    return value


def _hex_digest(value: Any, name: str, *, length: int) -> str:
    text = _nonempty_string(value, name)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise CalibrationContractError(
            f"{name} must be a canonical lowercase {length}-character hex digest"
        )
    return text


def _nonnegative_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CalibrationContractError(f"{name} must be a nonnegative integer")
    return int(value)


def _validate_channel_spec(value: Any) -> dict[str, list[str]]:
    expected = frozen_channel_spec()
    if value != expected:
        raise CalibrationContractError(
            "channel_spec must exactly match the frozen SOP00 ordered channels"
        )
    return copy.deepcopy(expected)


def _reauthenticate_dataset_family(
    dataset_family: LoadedRiskDatasetFamily | None,
    *,
    mode: str,
) -> LoadedRiskDatasetFamily | None:
    """Reload a production family from its sealed root and reject substitutes."""

    if mode == "toy":
        if dataset_family is not None:
            raise CalibrationContractError(
                "toy mode must not receive a production dataset family"
            )
        return None
    if type(dataset_family) is not LoadedRiskDatasetFamily:
        raise CalibrationContractError(
            "production mode is fail-closed without an authenticated "
            "LoadedRiskDatasetFamily"
        )
    try:
        authenticated = load_risk_dataset_family(dataset_family.root)
    except (OSError, RiskDataContractError, TypeError, ValueError) as exc:
        raise CalibrationContractError(
            "production dataset family reauthentication failed"
        ) from exc
    if authenticated != dataset_family:
        raise CalibrationContractError(
            "production dataset family reauthentication rejected a forged or "
            "stale object"
        )
    return authenticated


def _validate_production_family_binding(
    payload: Mapping[str, Any],
    *,
    dataset_family: LoadedRiskDatasetFamily,
    split: str,
    label: str,
) -> None:
    """Bind one production payload to the family and its corresponding member."""

    if payload.get("risk_dataset_family_layout_version") != (
        RISK_DATASET_FAMILY_LAYOUT_VERSION
    ):
        raise CalibrationContractError(
            f"{label} risk_dataset_family_layout_version mismatch"
        )
    family_digest = _hex_digest(
        payload.get("risk_dataset_family_digest"),
        "risk_dataset_family_digest",
        length=64,
    )
    if family_digest != dataset_family.risk_dataset_family_digest:
        raise CalibrationContractError(f"{label} family digest mismatch")
    member = dataset_family.members.get(split)
    if not isinstance(member, Mapping):
        raise CalibrationContractError(
            f"{label} split {split!r} is not a risk dataset family member"
        )
    member_digest = _hex_digest(
        payload.get("risk_dataset_manifest_digest"),
        "risk_dataset_manifest_digest",
        length=64,
    )
    if member_digest != member.get("risk_dataset_manifest_digest"):
        raise CalibrationContractError(
            f"{label} member digest mismatch for split {split}"
        )
    common = dataset_family.manifest.get("common_contract")
    if not isinstance(common, Mapping):
        raise CalibrationContractError(
            "authenticated dataset family common contract is unavailable"
        )
    provenance_lengths = {
        "g1_split_manifest_digest": 32,
        "dynamic_objects_config_digest": 64,
        "target_type_policy_digest": 32,
    }
    for field, length in provenance_lengths.items():
        value = _hex_digest(payload.get(field), field, length=length)
        if value != common.get(field):
            raise CalibrationContractError(
                f"{label} family common contract mismatch for {field}"
            )
    if payload.get("schema_version") != common.get("schema_version"):
        raise CalibrationContractError(
            f"{label} family common contract mismatch for schema_version"
        )
    common_channels = common.get("channel_spec")
    if not isinstance(common_channels, Mapping):
        raise CalibrationContractError(
            "authenticated dataset family channel contract is unavailable"
        )
    family_input_channels = _semantic_projection(
        {
            key: common_channels.get(key)
            for key in ("history", "state", "trajectory", "flat")
        }
    )
    if payload.get("channel_spec") != family_input_channels:
        raise CalibrationContractError(
            f"{label} family common contract mismatch for channel_spec"
        )


def prediction_table_cohort_digest(table: Mapping[str, Any]) -> str:
    """Hash the method-independent ordered labels, groups, and source identities."""

    if not isinstance(table, Mapping):
        raise CalibrationContractError("prediction table must be a mapping")
    mode = table.get("mode")
    if mode == "toy":
        dataset_digest = table.get("toy_dataset_manifest_digest")
    else:
        dataset_digest = table.get("risk_dataset_manifest_digest")
    rows = table.get("rows")
    if not isinstance(rows, list):
        raise CalibrationContractError("prediction table rows must be a list")
    try:
        cohort_rows = [
            {field: row[field] for field in sorted(TOY_MANIFEST_ROW_KEYS)}
            for row in rows
        ]
    except (KeyError, TypeError) as exc:
        raise CalibrationContractError(
            "prediction table cohort rows are incomplete"
        ) from exc
    return _semantic_digest(
        {
            "cohort_layout_version": PREDICTION_COHORT_LAYOUT_VERSION,
            "mode": mode,
            "schema_version": table.get("schema_version"),
            "split": table.get("split"),
            "dataset_manifest_digest": dataset_digest,
            "rows": cohort_rows,
        }
    )


def _validate_row(row: Mapping[str, Any], *, table_split: str, index: int) -> None:
    expected_fields = set(ROW_REQUIRED_FIELDS)
    actual_fields = set(row)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unexpected = sorted(actual_fields - expected_fields)
        raise CalibrationContractError(
            f"prediction row {index} fields are invalid: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for field in ROW_STRING_FIELDS:
        value = row[field]
        if not isinstance(value, str) or not value:
            raise CalibrationContractError(
                f"prediction row {index} field {field} must be a non-empty string"
            )
    if row["split"] != table_split:
        raise CalibrationContractError(
            f"prediction row {index} split {row['split']!r} does not match "
            f"table split {table_split!r}"
        )
    if row["collision_label"] not in (0, 1):
        raise CalibrationContractError(
            f"prediction row {index} collision_label must be binary"
        )
    if row["near_miss"] not in (0, 1):
        raise CalibrationContractError(
            f"prediction row {index} near_miss must be binary"
        )
    collision_time = row["first_collision_time"]
    if collision_time is not None and _finite_scalar(
        collision_time, f"prediction row {index} first_collision_time"
    ) <= 0.0:
        raise CalibrationContractError(
            f"prediction row {index} first_collision_time must be null or positive"
        )
    for field in ("risk_severity", "p_collision", "q50", "q80", "q90", "q95"):
        _probability(row[field], f"prediction row {index} {field}")
    quantiles = [float(row[field]) for field in ("q50", "q80", "q90", "q95")]
    if any(left > right for left, right in zip(quantiles, quantiles[1:])):
        raise CalibrationContractError(
            f"prediction row {index} has crossing risk quantiles"
        )
    # SOP07 clearance is a signed shape distance: overlap/collision may be
    # represented by a negative value.  Its contract is finite, not nonnegative.
    _finite_scalar(row["min_clearance"], f"prediction row {index} min_clearance")
    critical_id = row["critical_object_id"]
    if critical_id is not None and (
        not isinstance(critical_id, str) or not critical_id
    ):
        raise CalibrationContractError(
            f"prediction row {index} critical_object_id must be null or non-empty string"
        )
    for field in ("critical_area_fraction", "density_fraction"):
        _probability(row[field], f"prediction row {index} {field}")
    if _finite_scalar(row["age_s"], f"prediction row {index} age_s") < 0.0:
        raise CalibrationContractError(
            f"prediction row {index} age_s must be nonnegative"
        )
    if not isinstance(row["pair_eligible"], (bool, np.bool_)):
        raise CalibrationContractError(
            f"prediction row {index} pair_eligible must be boolean"
        )
    for field in ("footprint_dimensions_m", "robot_footprint_dimensions_m"):
        dimensions = row[field]
        if (
            not isinstance(dimensions, (list, tuple))
            or len(dimensions) != 2
            or any(_finite_scalar(value, f"prediction row {index} {field}") <= 0.0 for value in dimensions)
        ):
            raise CalibrationContractError(
                f"prediction row {index} {field} must contain two positive dimensions"
            )


def validate_prediction_table(
    table: Mapping[str, Any],
    *,
    expected_mode: str | None = None,
    expected_split: str | None = None,
    dataset_family: LoadedRiskDatasetFamily | None = None,
) -> dict[str, Any]:
    """Validate and defensively copy a prediction table."""

    if not isinstance(table, Mapping):
        raise CalibrationContractError("prediction table must be a mapping")
    result = copy.deepcopy(dict(table))
    version = result.get("prediction_table_layout_version")
    if version != PREDICTION_TABLE_LAYOUT_VERSION:
        raise CalibrationContractError(
            "prediction table layout version must be "
            f"{PREDICTION_TABLE_LAYOUT_VERSION!r}, got {version!r}"
        )
    mode = result.get("mode")
    if mode not in {"toy", "production"}:
        raise CalibrationContractError("prediction table mode must be toy or production")
    if expected_mode is not None and mode != expected_mode:
        raise CalibrationContractError(
            f"prediction table mode mismatch: expected {expected_mode!r}, got {mode!r}"
        )
    authenticated_family = _reauthenticate_dataset_family(
        dataset_family,
        mode=mode,
    )
    if result.get("schema_version") != "3.0.0":
        raise CalibrationContractError("prediction table schema_version must be '3.0.0'")
    split = result.get("split")
    if not isinstance(split, str) or not split:
        raise CalibrationContractError("prediction table split must be a non-empty string")
    if expected_split is not None and split != expected_split:
        raise CalibrationContractError(
            f"prediction table split mismatch: expected {expected_split!r}, got {split!r}"
        )
    _nonempty_string(result.get("method_id"), "method_id")
    _nonnegative_integer(result.get("seed"), "seed")
    result["channel_spec"] = _validate_channel_spec(result.get("channel_spec"))
    _hex_digest(
        result.get("config_digest_sha256"), "config_digest_sha256", length=64
    )
    checkpoint_layout_version = result.get("checkpoint_layout_version")
    if checkpoint_layout_version not in CHECKPOINT_LAYOUT_VERSIONS:
        raise CalibrationContractError(
            "checkpoint layout version must be one of "
            f"{sorted(CHECKPOINT_LAYOUT_VERSIONS)!r}"
        )
    _hex_digest(result.get("checkpoint_digest"), "checkpoint_digest", length=64)
    if checkpoint_layout_version == RISK_CHECKPOINT_LAYOUT_VERSION:
        if result.get("checkpoint_digest_kind") != (
            "risk_checkpoint_semantic_sha256"
        ):
            raise CalibrationContractError(
                "risk prediction table checkpoint_digest_kind must be "
                "risk_checkpoint_semantic_sha256"
            )
        if "prediction_semantics" in result:
            raise CalibrationContractError(
                "risk prediction table must not contain prediction_semantics"
            )
    else:
        if result.get("checkpoint_digest_kind") != (
            "occupancy_checkpoint_semantic_sha256"
        ):
            raise CalibrationContractError(
                "occupancy prediction table checkpoint_digest_kind must be "
                "occupancy_checkpoint_semantic_sha256"
            )
        if result.get("prediction_semantics") != (
            "scalar_baseline_score_repeated_for_common_calibration"
        ):
            raise CalibrationContractError(
                "occupancy prediction table prediction_semantics must be "
                "scalar_baseline_score_repeated_for_common_calibration"
            )

    production_fields = (
        "risk_dataset_family_layout_version",
        "risk_dataset_family_digest",
        "g1_split_manifest_digest",
        "risk_dataset_manifest_digest",
        "dynamic_objects_config_digest",
        "target_type_policy_digest",
    )
    if mode == "toy":
        _hex_digest(
            result.get("toy_dataset_manifest_digest"),
            "toy_dataset_manifest_digest",
            length=32,
        )
        forbidden = [field for field in production_fields if field in result]
        if forbidden:
            raise CalibrationContractError(
                f"toy prediction table contains production provenance fields: {forbidden}"
            )
    else:
        if "toy_dataset_manifest_digest" in result:
            raise CalibrationContractError(
                "production prediction table must not contain toy provenance"
            )
        assert authenticated_family is not None
        _validate_production_family_binding(
            result,
            dataset_family=authenticated_family,
            split=split,
            label="prediction table",
        )

    rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        raise CalibrationContractError("prediction table rows must be a non-empty list")
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise CalibrationContractError(f"prediction row {index} must be a mapping")
        _validate_row(row, table_split=split, index=index)
        sample_id = str(row["sample_id"])
        if sample_id in seen_ids:
            raise CalibrationContractError(
                f"prediction table has duplicate sample_id {sample_id!r}"
            )
        seen_ids.add(sample_id)
    if authenticated_family is not None:
        member = authenticated_family.members[split]
        if len(rows) != member["sample_count"]:
            raise CalibrationContractError(
                "production prediction table sample count differs from its "
                f"family member for split {split}"
            )
        observed_membership = risk_dataset_family_sample_ids_digest(
            [str(row["sample_id"]) for row in rows]
        )
        if observed_membership != member["sample_ids_digest_sha256"]:
            raise CalibrationContractError(
                "production prediction table sample ID membership differs "
                f"from its family member for split {split}"
            )

    declared_cohort_digest = _hex_digest(
        result.get("cohort_digest_sha256"), "cohort_digest_sha256", length=64
    )
    actual_cohort_digest = prediction_table_cohort_digest(result)
    if declared_cohort_digest != actual_cohort_digest:
        raise CalibrationContractError(
            "prediction table cohort digest mismatch: labels, groups, or identities may be tampered"
        )

    declared_digest = result.get("semantic_digest")
    _hex_digest(declared_digest, "semantic_digest", length=64)
    actual_digest = prediction_table_semantic_digest(result)
    if declared_digest != actual_digest:
        raise CalibrationContractError(
            "prediction table semantic digest mismatch: table may be tampered"
        )
    return result


def _as_finite_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise CalibrationContractError(f"{name} must be a one-dimensional vector")
    if not np.isfinite(array).all():
        raise CalibrationContractError(f"{name} must be finite")
    return array


def one_sided_residuals(y_true: Any, predicted_quantile: Any) -> np.ndarray:
    """Compute ``max(0, y - q)`` without using any test-set statistics."""

    target = _as_finite_vector(y_true, "y_true")
    prediction = _as_finite_vector(predicted_quantile, "predicted_quantile")
    if target.shape != prediction.shape:
        raise CalibrationContractError(
            "y_true and predicted_quantile must have identical shapes"
        )
    return np.maximum(0.0, target - prediction)


def finite_sample_conformal_quantile(
    residuals: Any, *, alpha: float
) -> tuple[float, int]:
    """Return the finite-sample correction and its one-based ceiling rank."""

    scores = _as_finite_vector(residuals, "residuals")
    if scores.size == 0:
        raise CalibrationContractError("residuals must be non-empty")
    if np.any(scores < 0.0):
        raise CalibrationContractError("residuals must be nonnegative")
    alpha_value = _finite_scalar(alpha, "alpha")
    if not 0.0 < alpha_value < 1.0:
        raise CalibrationContractError("alpha must be strictly between 0 and 1")
    rank = min(
        int(scores.size),
        int(math.ceil((int(scores.size) + 1) * (1.0 - alpha_value))),
    )
    value = float(np.partition(scores, rank - 1)[rank - 1])
    return value, rank


def apply_split_conformal(predicted_quantile: Any, *, correction: float) -> np.ndarray:
    """Add a nonnegative correction and clip the upper risk bound to ``[0, 1]``."""

    prediction = _as_finite_vector(predicted_quantile, "predicted_quantile")
    correction_value = _finite_scalar(correction, "correction")
    if correction_value < 0.0:
        raise CalibrationContractError("correction must be nonnegative")
    return np.clip(prediction + correction_value, 0.0, 1.0)


def _identity_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    return {
        field: sorted({str(row[field]) for row in rows})
        for field in IDENTITY_FIELDS
    }


def fit_calibration_artifact(
    prediction_table: Mapping[str, Any],
    *,
    alpha: float,
    prediction_key: str,
    dataset_family: LoadedRiskDatasetFamily | None = None,
) -> dict[str, Any]:
    """Fit a global correction from a validated calibration prediction table."""

    table = validate_prediction_table(
        prediction_table,
        dataset_family=dataset_family,
    )
    if table["split"] != "calibration":
        raise CalibrationContractError(
            "conformal fitting requires the calibration split, "
            f"got {table['split']!r}"
        )
    if prediction_key not in {"q50", "q80", "q90", "q95"}:
        raise CalibrationContractError(
            "prediction_key must be one of q50, q80, q90, q95"
        )
    rows = table["rows"]
    target = np.asarray([row["risk_severity"] for row in rows], dtype=np.float64)
    predicted = np.asarray([row[prediction_key] for row in rows], dtype=np.float64)
    residuals = one_sided_residuals(target, predicted)
    correction, rank = finite_sample_conformal_quantile(residuals, alpha=alpha)
    artifact: dict[str, Any] = {
        "calibration_artifact_layout_version": CALIBRATION_ARTIFACT_LAYOUT_VERSION,
        "mode": table["mode"],
        "schema_version": table["schema_version"],
        "fit_split": "calibration",
        "method_id": table["method_id"],
        "checkpoint_layout_version": table["checkpoint_layout_version"],
        "checkpoint_digest": table["checkpoint_digest"],
        "checkpoint_digest_kind": table["checkpoint_digest_kind"],
        "prediction_table_semantic_digest": table["semantic_digest"],
        "calibration_cohort_digest_sha256": table["cohort_digest_sha256"],
        "seed": table["seed"],
        "channel_spec": table["channel_spec"],
        "config_digest_sha256": table["config_digest_sha256"],
        "prediction_key": prediction_key,
        "alpha": float(alpha),
        "global": {
            "correction": correction,
            "count": int(residuals.size),
            "rank_one_based": rank,
            "residual_min": float(np.min(residuals)),
            "residual_max": float(np.max(residuals)),
        },
        "fitted_identities": _identity_manifest(rows),
    }
    if table["checkpoint_layout_version"] == OCCUPANCY_CHECKPOINT_LAYOUT_VERSION:
        artifact["prediction_semantics"] = table["prediction_semantics"]
    if table["mode"] == "toy":
        artifact["toy_dataset_manifest_digest"] = table[
            "toy_dataset_manifest_digest"
        ]
    else:
        for field in (
            "risk_dataset_family_layout_version",
            "risk_dataset_family_digest",
            "g1_split_manifest_digest",
            "risk_dataset_manifest_digest",
            "dynamic_objects_config_digest",
            "target_type_policy_digest",
        ):
            artifact[field] = table[field]
    artifact["semantic_digest"] = calibration_artifact_semantic_digest(artifact)
    return artifact


def validate_calibration_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_mode: str | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
    dataset_family: LoadedRiskDatasetFamily | None = None,
) -> dict[str, Any]:
    """Validate artifact layout, provenance, fitted identities, and digest."""

    if not isinstance(artifact, Mapping):
        raise CalibrationContractError("calibration artifact must be a mapping")
    result = copy.deepcopy(dict(artifact))
    version = result.get("calibration_artifact_layout_version")
    if version != CALIBRATION_ARTIFACT_LAYOUT_VERSION:
        raise CalibrationContractError(
            "calibration artifact layout version must be "
            f"{CALIBRATION_ARTIFACT_LAYOUT_VERSION!r}, got {version!r}"
        )
    mode = result.get("mode")
    if mode not in {"toy", "production"}:
        raise CalibrationContractError("calibration artifact mode is invalid")
    if expected_mode is not None and mode != expected_mode:
        raise CalibrationContractError(
            f"calibration artifact mode mismatch: expected {expected_mode!r}, got {mode!r}"
        )
    authenticated_family = _reauthenticate_dataset_family(
        dataset_family,
        mode=mode,
    )
    if result.get("schema_version") != "3.0.0":
        raise CalibrationContractError("calibration artifact schema_version must be '3.0.0'")
    if result.get("fit_split") != "calibration":
        raise CalibrationContractError("artifact fit_split must be calibration")
    checkpoint_layout_version = result.get("checkpoint_layout_version")
    if checkpoint_layout_version not in CHECKPOINT_LAYOUT_VERSIONS:
        raise CalibrationContractError(
            "checkpoint layout version must be one of "
            f"{sorted(CHECKPOINT_LAYOUT_VERSIONS)!r}"
        )
    _nonempty_string(result.get("method_id"), "method_id")
    for field in (
        "checkpoint_digest",
        "prediction_table_semantic_digest",
        "calibration_cohort_digest_sha256",
        "config_digest_sha256",
    ):
        _hex_digest(result.get(field), field, length=64)
    _nonnegative_integer(result.get("seed"), "seed")
    result["channel_spec"] = _validate_channel_spec(result.get("channel_spec"))
    if checkpoint_layout_version == RISK_CHECKPOINT_LAYOUT_VERSION:
        if result.get("checkpoint_digest_kind") != (
            "risk_checkpoint_semantic_sha256"
        ):
            raise CalibrationContractError(
                "risk calibration artifact checkpoint_digest_kind must be "
                "risk_checkpoint_semantic_sha256"
            )
        if "prediction_semantics" in result:
            raise CalibrationContractError(
                "risk calibration artifact must not contain prediction_semantics"
            )
    else:
        if result.get("checkpoint_digest_kind") != (
            "occupancy_checkpoint_semantic_sha256"
        ):
            raise CalibrationContractError(
                "occupancy calibration artifact checkpoint_digest_kind must be "
                "occupancy_checkpoint_semantic_sha256"
            )
        if result.get("prediction_semantics") != (
            "scalar_baseline_score_repeated_for_common_calibration"
        ):
            raise CalibrationContractError(
                "occupancy calibration artifact prediction_semantics must be "
                "scalar_baseline_score_repeated_for_common_calibration"
            )
    prediction_key = result.get("prediction_key")
    if prediction_key not in {"q50", "q80", "q90", "q95"}:
        raise CalibrationContractError(
            "artifact prediction_key must be one of q50, q80, q90, q95"
        )
    alpha_value = _finite_scalar(result.get("alpha"), "alpha")
    if not 0.0 < alpha_value < 1.0:
        raise CalibrationContractError("artifact alpha must be strictly between 0 and 1")
    if mode == "toy":
        _hex_digest(
            result.get("toy_dataset_manifest_digest"),
            "toy_dataset_manifest_digest",
            length=32,
        )
        if any(
            field in result
            for field in (
                "risk_dataset_family_layout_version",
                "risk_dataset_family_digest",
                "g1_split_manifest_digest",
                "risk_dataset_manifest_digest",
                "dynamic_objects_config_digest",
                "target_type_policy_digest",
            )
        ):
            raise CalibrationContractError(
                "toy calibration artifact contains production provenance"
            )
    else:
        if "toy_dataset_manifest_digest" in result:
            raise CalibrationContractError(
                "production calibration artifact contains toy provenance"
            )
        assert authenticated_family is not None
        _validate_production_family_binding(
            result,
            dataset_family=authenticated_family,
            split="calibration",
            label="calibration artifact",
        )
    global_fit = result.get("global")
    if not isinstance(global_fit, Mapping):
        raise CalibrationContractError("artifact global fit must be a mapping")
    correction = _finite_scalar(global_fit.get("correction"), "global correction")
    if not 0.0 <= correction <= 1.0:
        raise CalibrationContractError("global correction must be in [0, 1]")
    count = global_fit.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise CalibrationContractError("global count must be a positive integer")
    rank = global_fit.get("rank_one_based")
    expected_rank = min(count, int(math.ceil((count + 1) * (1.0 - alpha_value))))
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or not 1 <= rank <= count
        or rank != expected_rank
    ):
        raise CalibrationContractError(
            f"global rank_one_based must equal finite-sample rank {expected_rank}"
        )
    residual_min = _finite_scalar(global_fit.get("residual_min"), "residual_min")
    residual_max = _finite_scalar(global_fit.get("residual_max"), "residual_max")
    if not 0.0 <= residual_min <= residual_max <= 1.0:
        raise CalibrationContractError(
            "global residual bounds must satisfy 0 <= min <= max <= 1"
        )
    if not residual_min <= correction <= residual_max:
        raise CalibrationContractError(
            "global correction must lie within the recorded residual bounds"
        )
    identities = result.get("fitted_identities")
    if not isinstance(identities, Mapping) or set(identities) != set(IDENTITY_FIELDS):
        raise CalibrationContractError("artifact fitted_identities must be a mapping")
    for field in IDENTITY_FIELDS:
        values = identities.get(field)
        if not isinstance(values, list) or not values or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise CalibrationContractError(
                f"artifact identity field {field} must be a string list"
            )
        if values != sorted(set(values)):
            raise CalibrationContractError(
                f"artifact identity field {field} must be sorted unique"
            )
    if len(identities["sample_id"]) != count:
        raise CalibrationContractError(
            "artifact identity field sample_id count must equal global count"
        )
    if "grouped" in result:
        # Local import avoids a module-import cycle: grouped calibration uses the
        # scalar split-conformal primitives defined above.
        from src.calibration.grouped_calibration import (
            validate_grouped_calibration_artifact,
        )

        result["grouped"] = validate_grouped_calibration_artifact(
            result["grouped"],
            expected_alpha=alpha_value,
            expected_prediction_key=prediction_key,
            expected_global=global_fit,
        )
    declared_digest = result.get("semantic_digest")
    _hex_digest(declared_digest, "semantic_digest", length=64)
    if declared_digest != calibration_artifact_semantic_digest(result):
        raise CalibrationContractError(
            "calibration artifact semantic digest mismatch: artifact may be tampered"
        )
    if expected_provenance is not None:
        for field, expected in expected_provenance.items():
            if result.get(field) != expected:
                raise CalibrationContractError(
                    f"calibration artifact provenance mismatch for {field}: "
                    f"expected {expected!r}, got {result.get(field)!r}"
                )
    return result


def assert_calibration_test_isolation(
    calibration_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Reject any strict identity overlap between calibration and test rows."""

    for name, rows, expected_split in (
        ("calibration", calibration_rows, "calibration"),
        ("test", test_rows, "test"),
    ):
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise CalibrationContractError(f"{name}_rows must be a sequence")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise CalibrationContractError(f"{name} row {index} must be a mapping")
            if row.get("split") != expected_split:
                raise CalibrationContractError(
                    f"{name} row {index} must have split {expected_split!r}"
                )
            for field in IDENTITY_FIELDS:
                if field not in row:
                    raise CalibrationContractError(
                        f"{name} row {index} is missing identity field {field}"
                    )
    result: dict[str, int] = {}
    for field in IDENTITY_FIELDS:
        calibration_values = {str(row[field]) for row in calibration_rows}
        test_values = {str(row[field]) for row in test_rows}
        overlap = sorted(calibration_values & test_values)
        if overlap:
            raise CalibrationContractError(
                f"calibration/test identity overlap for {field}: {overlap[:5]}"
            )
        result[field] = 0
    return result


def assert_calibration_artifact_test_isolation(
    artifact: Mapping[str, Any],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    dataset_family: LoadedRiskDatasetFamily | None = None,
) -> dict[str, int]:
    """Check test identities against the fitted identity sets in an artifact."""

    validated = validate_calibration_artifact(
        artifact,
        dataset_family=dataset_family,
    )
    identities = validated["fitted_identities"]
    result: dict[str, int] = {}
    for index, row in enumerate(test_rows):
        if not isinstance(row, Mapping):
            raise CalibrationContractError(f"test row {index} must be a mapping")
        if row.get("split") != "test":
            raise CalibrationContractError(f"test row {index} must have split 'test'")
        for field in IDENTITY_FIELDS:
            if field not in row:
                raise CalibrationContractError(
                    f"test row {index} is missing identity field {field}"
                )
    for field in IDENTITY_FIELDS:
        fitted = set(identities[field])
        overlap = sorted(fitted & {str(row[field]) for row in test_rows})
        if overlap:
            raise CalibrationContractError(
                f"calibration/test identity overlap for {field}: {overlap[:5]}"
            )
        result[field] = 0
    return result
