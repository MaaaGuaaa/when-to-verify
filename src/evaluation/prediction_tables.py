"""Shared authenticated prediction-table and calibration-protocol contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn

from src.calibration.grouped_calibration import (
    CONTINUOUS_GROUP_BINS,
    GROUPED_CALIBRATION_LAYOUT_VERSION,
    GROUP_DIMENSIONS,
)
from src.calibration.split_conformal import (
    BASELINE_SPEC_LAYOUT_VERSION,
    FORMAL_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
    PREDICTION_TABLE_LAYOUT_VERSION,
    RISK_CHECKPOINT_LAYOUT_VERSION,
    prediction_table_cohort_digest,
    prediction_table_semantic_digest,
    validate_prediction_table,
)
from src.calibration.split_conformal import assert_calibration_test_isolation
from src.contracts import SCHEMA_VERSION
from src.datasets.risk_dataset_seal import (
    RISK_DATASET_FAMILY_LAYOUT_VERSION,
    LoadedRiskDataset,
    LoadedRiskDatasetFamily,
    load_risk_dataset_family,
)
from src.datasets.risk_evaluation_metadata import RISK_EVALUATION_RECORD_FIELDS
from src.datasets.risk_evaluation_store import (
    RISK_EVALUATION_COLLECTION_LAYOUT_VERSION,
    LoadedRiskEvaluationCollection,
)
from src.datasets.toy_risk_learning import frozen_channel_spec
from src.evaluation.risk_baselines import (
    BASELINE_SPECS,
    score_production_occupancy_baseline,
)
from src.models.occupancy_baseline import (
    ConvGRUOccupancyPredictor,
    LearnedOccupancyRiskAggregator,
)
from src.utils.atomic_publish import atomic_rename_noreplace


PREDICTION_PROTOCOL_LAYOUT_VERSION = "shared_risk_prediction_protocol_v1"
PREDICTION_COHORT_BINDING_LAYOUT_VERSION = "shared_prediction_cohort_binding_v1"
UNIFIED_PREDICTION_COLLECTION_LAYOUT_VERSION = "unified_prediction_collection_v1"
UNIFIED_PREDICTION_METHODS = ("risk-r0", "risk-r1", "B1", "B2", "B3", "B4")
_PREDICTION_FIELDS = ("p_collision", "q50", "q80", "q90", "q95")
_PROTOCOL_KEYS = frozenset(
    {
        "protocol_layout_version",
        "prediction_table_layout_version",
        "alpha",
        "prediction_key",
        "global_calibration_rule",
        "grouped_calibration",
        "baseline_quantile_proxy_policy",
        "protocol_digest_sha256",
    }
)


class PredictionTableContractError(ValueError):
    """Raised when cohorts, protocols, or method provenance diverge."""


@dataclass(frozen=True)
class LoadedUnifiedPredictionCollection:
    """One fully reauthenticated calibration or complete prediction bundle."""

    root: Path
    manifest: dict[str, object]
    protocol: dict[str, object]
    tables_by_split: Mapping[str, Mapping[str, dict[str, Any]]]
    collection_semantic_digest_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PredictionTableContractError(
            "prediction metadata must be finite canonical JSON"
        ) from exc


def _json_copy(value: object) -> object:
    return json.loads(_canonical_json_bytes(value))


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PredictionTableContractError(f"{label} must be a lowercase SHA-256")
    return value


def _semantic_digest(domain: bytes, value: object) -> str:
    payload = _canonical_json_bytes(value)
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class PredictionMethodArtifact:
    """Checkpoint or deterministic baseline specification bound to one table."""

    method_id: str
    layout_version: str
    digest_sha256: str
    digest_kind: str
    score_definition: str

    def __post_init__(self) -> None:
        if self.method_id not in UNIFIED_PREDICTION_METHODS:
            raise PredictionTableContractError("unsupported prediction method")
        if not isinstance(self.layout_version, str) or not self.layout_version:
            raise PredictionTableContractError("method artifact layout is required")
        _sha256(self.digest_sha256, label="method artifact digest")
        if not isinstance(self.digest_kind, str) or not self.digest_kind:
            raise PredictionTableContractError("method artifact digest kind is required")
        if not isinstance(self.score_definition, str) or not self.score_definition:
            raise PredictionTableContractError("score definition is required")
        expected = {
            "risk-r0": (RISK_CHECKPOINT_LAYOUT_VERSION, "risk_checkpoint_semantic_sha256"),
            "risk-r1": (RISK_CHECKPOINT_LAYOUT_VERSION, "risk_checkpoint_semantic_sha256"),
            "B1": (BASELINE_SPEC_LAYOUT_VERSION, "baseline_spec_sha256"),
            "B2": (BASELINE_SPEC_LAYOUT_VERSION, "baseline_spec_sha256"),
            "B3": (
                FORMAL_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
                "formal_occupancy_checkpoint_semantic_sha256",
            ),
            "B4": (
                FORMAL_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
                "formal_occupancy_checkpoint_semantic_sha256",
            ),
        }[self.method_id]
        if (self.layout_version, self.digest_kind) != expected:
            raise PredictionTableContractError(
                f"method artifact contract mismatch for {self.method_id}"
            )


def baseline_spec_digest(
    method_id: str,
    *,
    b2_tau_s: float,
    b2_a_max_s: float,
    sigma_time_s: float,
) -> str:
    """Digest the executable deterministic B1/B2 specification."""

    if method_id not in {"B1", "B2"}:
        raise PredictionTableContractError("baseline spec digest supports only B1/B2")
    parameters: dict[str, float] = {}
    for name, value in (
        ("b2_tau_s", b2_tau_s),
        ("b2_a_max_s", b2_a_max_s),
        ("sigma_time_s", sigma_time_s),
    ):
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise PredictionTableContractError(f"{name} must be positive and finite")
        parameters[name] = number
    return _semantic_digest(
        b"deterministic-occupancy-baseline-spec-v1\0",
        {
            "layout_version": BASELINE_SPEC_LAYOUT_VERSION,
            "method_id": method_id,
            "implementation": BASELINE_SPECS[method_id],
            "history_steps": 8,
            "future_steps": 15,
            "future_dt_s": 0.2,
            "parameters": parameters,
        },
    )


def _protocol_digest(protocol: Mapping[str, object]) -> str:
    return _semantic_digest(
        b"shared-risk-prediction-protocol-v1\0",
        {
            key: protocol[key]
            for key in _PROTOCOL_KEYS
            if key != "protocol_digest_sha256"
        },
    )


def build_prediction_protocol(
    *,
    alpha: float,
    prediction_key: str,
    min_group_size: int,
    group_dimensions: Sequence[str] = GROUP_DIMENSIONS,
) -> dict[str, object]:
    """Freeze one calibration protocol shared by all six methods."""

    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise PredictionTableContractError("alpha must be strictly between zero and one")
    if prediction_key not in {"q50", "q80", "q90", "q95"}:
        raise PredictionTableContractError("prediction_key must name a frozen quantile")
    if type(min_group_size) is not int or min_group_size < 1:
        raise PredictionTableContractError("min_group_size must be positive")
    dimensions = tuple(group_dimensions)
    if (
        not dimensions
        or len(set(dimensions)) != len(dimensions)
        or any(value not in GROUP_DIMENSIONS for value in dimensions)
    ):
        raise PredictionTableContractError("group dimensions are invalid")
    protocol: dict[str, object] = {
        "protocol_layout_version": PREDICTION_PROTOCOL_LAYOUT_VERSION,
        "prediction_table_layout_version": PREDICTION_TABLE_LAYOUT_VERSION,
        "alpha": alpha_value,
        "prediction_key": prediction_key,
        "global_calibration_rule": "one_sided_split_conformal_finite_sample",
        "grouped_calibration": {
            "layout_version": GROUPED_CALIBRATION_LAYOUT_VERSION,
            "min_group_size": min_group_size,
            "group_dimensions": list(dimensions),
            "continuous_group_bins": {
                key: list(values)
                for key, values in sorted(CONTINUOUS_GROUP_BINS.items())
            },
            "combination_policy": "one_dimension_at_a_time",
        },
        "baseline_quantile_proxy_policy": (
            "q50=q80=q90=q95=raw_score_before_conformal"
        ),
    }
    protocol["protocol_digest_sha256"] = _protocol_digest(protocol)
    return validate_prediction_protocol(protocol)


def validate_prediction_protocol(
    protocol: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(protocol, Mapping) or set(protocol) != _PROTOCOL_KEYS:
        raise PredictionTableContractError("prediction protocol fields mismatch")
    result = _json_copy(dict(protocol))
    assert isinstance(result, dict)
    if result["protocol_layout_version"] != PREDICTION_PROTOCOL_LAYOUT_VERSION:
        raise PredictionTableContractError("prediction protocol layout mismatch")
    if result["prediction_table_layout_version"] != PREDICTION_TABLE_LAYOUT_VERSION:
        raise PredictionTableContractError("prediction table layout mismatch in protocol")
    alpha = result["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise PredictionTableContractError("protocol alpha must be numeric")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise PredictionTableContractError("protocol alpha is invalid")
    if result["prediction_key"] not in {"q50", "q80", "q90", "q95"}:
        raise PredictionTableContractError("protocol prediction key is invalid")
    if result["global_calibration_rule"] != (
        "one_sided_split_conformal_finite_sample"
    ):
        raise PredictionTableContractError("global calibration rule mismatch")
    if result["baseline_quantile_proxy_policy"] != (
        "q50=q80=q90=q95=raw_score_before_conformal"
    ):
        raise PredictionTableContractError("baseline quantile proxy policy mismatch")
    grouped = result["grouped_calibration"]
    if not isinstance(grouped, dict) or set(grouped) != {
        "layout_version",
        "min_group_size",
        "group_dimensions",
        "continuous_group_bins",
        "combination_policy",
    }:
        raise PredictionTableContractError("grouped protocol fields mismatch")
    if grouped["layout_version"] != GROUPED_CALIBRATION_LAYOUT_VERSION:
        raise PredictionTableContractError("grouped protocol layout mismatch")
    if type(grouped["min_group_size"]) is not int or grouped["min_group_size"] < 1:
        raise PredictionTableContractError("grouped min_group_size is invalid")
    dimensions = grouped["group_dimensions"]
    if (
        not isinstance(dimensions, list)
        or not dimensions
        or len(set(dimensions)) != len(dimensions)
        or any(value not in GROUP_DIMENSIONS for value in dimensions)
    ):
        raise PredictionTableContractError("grouped dimensions are invalid")
    expected_bins = {
        key: list(values) for key, values in sorted(CONTINUOUS_GROUP_BINS.items())
    }
    if grouped["continuous_group_bins"] != expected_bins:
        raise PredictionTableContractError("grouped continuous bins mismatch")
    if grouped["combination_policy"] != "one_dimension_at_a_time":
        raise PredictionTableContractError("grouped combination policy mismatch")
    declared = _sha256(
        result["protocol_digest_sha256"],
        label="prediction protocol digest",
    )
    if declared != _protocol_digest(result):
        raise PredictionTableContractError("prediction protocol digest mismatch")
    return result


def _family_input_channels(dataset_family: LoadedRiskDatasetFamily) -> dict[str, object]:
    common = dataset_family.manifest.get("common_contract")
    if not isinstance(common, Mapping):
        raise PredictionTableContractError("dataset family common contract is missing")
    channels = common.get("channel_spec")
    if not isinstance(channels, Mapping):
        raise PredictionTableContractError("dataset family channels are missing")
    result = {
        key: _json_copy(channels[key])
        for key in ("history", "state", "trajectory", "flat")
    }
    if result != frozen_channel_spec():
        raise PredictionTableContractError("dataset family input channel contract mismatch")
    return result


def _validate_sources(
    *,
    dataset_family: LoadedRiskDatasetFamily,
    dataset: LoadedRiskDataset,
    evaluation_records: LoadedRiskEvaluationCollection,
) -> LoadedRiskDatasetFamily:
    if type(dataset_family) is not LoadedRiskDatasetFamily:
        raise PredictionTableContractError("authenticated dataset family is required")
    authenticated = load_risk_dataset_family(dataset_family.root)
    if authenticated != dataset_family:
        raise PredictionTableContractError("dataset family is stale or forged")
    if not isinstance(dataset, LoadedRiskDataset):
        raise PredictionTableContractError("authenticated dataset member is required")
    member = authenticated.members.get(dataset.split)
    if not isinstance(member, Mapping) or (
        member.get("risk_dataset_manifest_digest")
        != dataset.risk_dataset_manifest_digest
        or member.get("sample_count") != dataset.sample_count
        or member.get("shard_count") != len(dataset.shards)
    ):
        raise PredictionTableContractError("dataset member does not belong to family")
    if not isinstance(evaluation_records, LoadedRiskEvaluationCollection):
        raise PredictionTableContractError("authenticated evaluation records are required")
    if (
        evaluation_records.risk_dataset_manifest_digest
        != dataset.risk_dataset_manifest_digest
        or evaluation_records.sample_count != dataset.sample_count
    ):
        raise PredictionTableContractError("evaluation record collection member mismatch")
    return authenticated


def _prediction_arrays(
    predictions: Mapping[str, object],
    *,
    count: int,
    repeat_score_as_quantiles: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(predictions, Mapping) or set(predictions) != {
        "p_collision",
        "quantiles",
    }:
        raise PredictionTableContractError("predictions must contain p_collision and quantiles")
    probability = np.asarray(predictions["p_collision"], dtype=np.float64)
    quantiles = np.asarray(predictions["quantiles"], dtype=np.float64)
    if probability.shape != (count,) or quantiles.shape != (count, 4):
        raise PredictionTableContractError("prediction array shape mismatch")
    if (
        not np.isfinite(probability).all()
        or not np.isfinite(quantiles).all()
        or np.any((probability < 0.0) | (probability > 1.0))
        or np.any((quantiles < 0.0) | (quantiles > 1.0))
    ):
        raise PredictionTableContractError("prediction arrays must be finite in [0,1]")
    if repeat_score_as_quantiles:
        quantiles = np.repeat(probability[:, None], 4, axis=1)
    elif np.any(quantiles[:, :-1] > quantiles[:, 1:]):
        raise PredictionTableContractError("risk model quantiles must not cross")
    return probability, quantiles


def _cohort_binding_digest(
    *,
    family_digest: str,
    dataset_digest: str,
    evaluation_digest: str,
    sidecar_digest: str,
    sample_ids: Sequence[str],
) -> str:
    return _semantic_digest(
        b"shared-prediction-cohort-binding-v1\0",
        {
            "layout_version": PREDICTION_COHORT_BINDING_LAYOUT_VERSION,
            "risk_dataset_family_digest": family_digest,
            "risk_dataset_manifest_digest": dataset_digest,
            "evaluation_record_collection_digest_sha256": evaluation_digest,
            "occupancy_sidecar_collection_digest_sha256": sidecar_digest,
            "ordered_sample_ids": list(sample_ids),
        },
    )


def build_production_prediction_table(
    *,
    dataset_family: LoadedRiskDatasetFamily,
    dataset: LoadedRiskDataset,
    evaluation_records: LoadedRiskEvaluationCollection,
    occupancy_sidecar_collection_digest_sha256: str,
    method_artifact: PredictionMethodArtifact,
    predictions: Mapping[str, object],
    protocol: Mapping[str, object],
    seed: int,
    config_digest_sha256: str,
) -> dict[str, Any]:
    """Join one prediction vector to the shared authenticated evaluation cohort."""

    family = _validate_sources(
        dataset_family=dataset_family,
        dataset=dataset,
        evaluation_records=evaluation_records,
    )
    sidecar_digest = _sha256(
        occupancy_sidecar_collection_digest_sha256,
        label="occupancy sidecar collection digest",
    )
    config_digest = _sha256(config_digest_sha256, label="config digest")
    if type(seed) is not int or seed < 0:
        raise PredictionTableContractError("seed must be a nonnegative integer")
    checked_protocol = validate_prediction_protocol(protocol)
    probability, quantiles = _prediction_arrays(
        predictions,
        count=evaluation_records.sample_count,
        repeat_score_as_quantiles=method_artifact.method_id in {"B1", "B2", "B3", "B4"},
    )
    rows: list[dict[str, object]] = []
    for record, score, values in zip(
        evaluation_records.records,
        probability,
        quantiles,
        strict=True,
    ):
        row = _json_copy(dict(record))
        assert isinstance(row, dict)
        row.update(
            {
                "p_collision": float(score),
                "q50": float(values[0]),
                "q80": float(values[1]),
                "q90": float(values[2]),
                "q95": float(values[3]),
            }
        )
        rows.append(row)
    common = family.manifest["common_contract"]
    assert isinstance(common, Mapping)
    table: dict[str, Any] = {
        "prediction_table_layout_version": PREDICTION_TABLE_LAYOUT_VERSION,
        "mode": "production",
        "schema_version": SCHEMA_VERSION,
        "split": dataset.split,
        "method_id": method_artifact.method_id,
        "checkpoint_layout_version": method_artifact.layout_version,
        "checkpoint_digest": method_artifact.digest_sha256,
        "checkpoint_digest_kind": method_artifact.digest_kind,
        "risk_dataset_family_layout_version": RISK_DATASET_FAMILY_LAYOUT_VERSION,
        "risk_dataset_family_digest": family.risk_dataset_family_digest,
        "g1_split_manifest_digest": common["g1_split_manifest_digest"],
        "risk_dataset_manifest_digest": dataset.risk_dataset_manifest_digest,
        "dynamic_objects_config_digest": common["dynamic_objects_config_digest"],
        "target_type_policy_digest": common["target_type_policy_digest"],
        "evaluation_record_collection_layout_version": (
            RISK_EVALUATION_COLLECTION_LAYOUT_VERSION
        ),
        "evaluation_record_collection_digest_sha256": (
            evaluation_records.collection_semantic_digest_sha256
        ),
        "occupancy_sidecar_collection_digest_sha256": sidecar_digest,
        "prediction_protocol_layout_version": PREDICTION_PROTOCOL_LAYOUT_VERSION,
        "prediction_protocol_digest_sha256": checked_protocol[
            "protocol_digest_sha256"
        ],
        "cohort_binding_digest_sha256": _cohort_binding_digest(
            family_digest=family.risk_dataset_family_digest,
            dataset_digest=dataset.risk_dataset_manifest_digest,
            evaluation_digest=evaluation_records.collection_semantic_digest_sha256,
            sidecar_digest=sidecar_digest,
            sample_ids=evaluation_records.sample_ids,
        ),
        "seed": seed,
        "channel_spec": _family_input_channels(family),
        "config_digest_sha256": config_digest,
        "score_definition": method_artifact.score_definition,
        "quantile_proxy_policy": (
            "native_model_quantiles"
            if method_artifact.method_id.startswith("risk-")
            else checked_protocol["baseline_quantile_proxy_policy"]
        ),
        "rows": rows,
    }
    if not method_artifact.method_id.startswith("risk-"):
        table["prediction_semantics"] = (
            "scalar_baseline_score_repeated_for_common_calibration"
        )
    table["cohort_digest_sha256"] = prediction_table_cohort_digest(table)
    table["semantic_digest"] = prediction_table_semantic_digest(table)
    try:
        return validate_prediction_table(
            table,
            expected_mode="production",
            expected_split=dataset.split,
            dataset_family=family,
        )
    except ValueError as exc:
        raise PredictionTableContractError(str(exc)) from exc


def score_unified_prediction_batch(
    *,
    risk_models: Mapping[str, nn.Module],
    occupancy_model: ConvGRUOccupancyPredictor,
    learned_aggregator: LearnedOccupancyRiskAggregator,
    model_inputs: Mapping[str, torch.Tensor],
    query_inputs: Mapping[str, torch.Tensor],
    b2_tau_s: float,
    b2_a_max_s: float,
    sigma_time_s: float,
    device: str | torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Score R0/R1 and B1--B4 from one shared label-free input batch."""

    if set(risk_models) != {"risk-r0", "risk-r1"}:
        raise PredictionTableContractError("risk_models must contain risk-r0 and risk-r1")
    if not isinstance(occupancy_model, ConvGRUOccupancyPredictor):
        raise PredictionTableContractError("occupancy_model has the wrong type")
    if not isinstance(learned_aggregator, LearnedOccupancyRiskAggregator):
        raise PredictionTableContractError("learned_aggregator has the wrong type")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise PredictionTableContractError("allocated CUDA device is unavailable")
    if not isinstance(model_inputs, Mapping) or not isinstance(query_inputs, Mapping):
        raise PredictionTableContractError("model and query inputs must be mappings")
    moved_inputs = {
        key: value.to(device=target_device)
        for key, value in model_inputs.items()
    }
    moved_queries = {
        key: value.to(device=target_device)
        for key, value in query_inputs.items()
    }
    occupancy_model.to(target_device)
    learned_aggregator.to(target_device)
    for model in risk_models.values():
        model.to(target_device)
    result: dict[str, dict[str, torch.Tensor]] = {}
    risk_modes = {name: model.training for name, model in risk_models.items()}
    try:
        for method in ("risk-r0", "risk-r1"):
            model = risk_models[method]
            model.eval()
            with torch.no_grad():
                prediction = model(moved_inputs)
            probability = prediction.get("p_collision")
            quantiles = prediction.get("quantiles")
            if (
                not torch.is_tensor(probability)
                or not torch.is_tensor(quantiles)
                or probability.ndim != 1
                or quantiles.shape != (probability.shape[0], 4)
            ):
                raise PredictionTableContractError(
                    f"{method} returned invalid prediction shapes"
                )
            result[method] = {
                "p_collision": probability.detach().cpu(),
                "quantiles": quantiles.detach().cpu(),
            }
    finally:
        for name, model in risk_models.items():
            model.train(risk_modes[name])
    for method in ("B1", "B2", "B3", "B4"):
        score = score_production_occupancy_baseline(
            method=method,
            model_inputs=moved_inputs,
            query_inputs=moved_queries,
            occupancy_model=occupancy_model,
            learned_aggregator=learned_aggregator,
            b2_tau_s=b2_tau_s,
            b2_a_max_s=b2_a_max_s,
            sigma_time_s=sigma_time_s,
        ).detach().cpu()
        result[method] = {
            "p_collision": score,
            "quantiles": score[:, None].expand(-1, 4).clone(),
        }
    if tuple(result) != UNIFIED_PREDICTION_METHODS:
        raise PredictionTableContractError("internal prediction method order drift")
    return result


def validate_shared_prediction_tables(
    tables: Mapping[str, Mapping[str, Any]],
    *,
    dataset_family: LoadedRiskDatasetFamily,
    dataset: LoadedRiskDataset,
    evaluation_records: LoadedRiskEvaluationCollection,
    expected_protocol: Mapping[str, object],
    expected_sidecar_collection_digest_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Validate six tables against one method-independent cohort and protocol."""

    if not isinstance(tables, Mapping) or set(tables) != set(UNIFIED_PREDICTION_METHODS):
        raise PredictionTableContractError("prediction tables must cover all six methods")
    family = _validate_sources(
        dataset_family=dataset_family,
        dataset=dataset,
        evaluation_records=evaluation_records,
    )
    protocol = validate_prediction_protocol(expected_protocol)
    sidecar_digest = _sha256(
        expected_sidecar_collection_digest_sha256,
        label="expected sidecar collection digest",
    )
    expected_rows = tuple(
        _canonical_json_bytes(dict(record)) for record in evaluation_records.records
    )
    validated: dict[str, dict[str, Any]] = {}
    common_values: dict[str, object] | None = None
    for method in UNIFIED_PREDICTION_METHODS:
        if tables[method].get("prediction_protocol_digest_sha256") != protocol[
            "protocol_digest_sha256"
        ]:
            raise PredictionTableContractError("prediction protocol mismatch")
        try:
            table = validate_prediction_table(
                tables[method],
                expected_mode="production",
                expected_split=dataset.split,
                dataset_family=family,
            )
        except ValueError as exc:
            raise PredictionTableContractError(str(exc)) from exc
        if table["method_id"] != method:
            raise PredictionTableContractError("prediction table method identity mismatch")
        expected_artifact_contract = {
            "risk-r0": (
                RISK_CHECKPOINT_LAYOUT_VERSION,
                "risk_checkpoint_semantic_sha256",
                "risk_model_collision_and_quantile_heads",
                "native_model_quantiles",
            ),
            "risk-r1": (
                RISK_CHECKPOINT_LAYOUT_VERSION,
                "risk_checkpoint_semantic_sha256",
                "risk_model_collision_and_quantile_heads",
                "native_model_quantiles",
            ),
            "B1": (
                BASELINE_SPEC_LAYOUT_VERSION,
                "baseline_spec_sha256",
                "normalized_weighted_sum",
                protocol["baseline_quantile_proxy_policy"],
            ),
            "B2": (
                BASELINE_SPEC_LAYOUT_VERSION,
                "baseline_spec_sha256",
                "normalized_weighted_sum",
                protocol["baseline_quantile_proxy_policy"],
            ),
            "B3": (
                FORMAL_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
                "formal_occupancy_checkpoint_semantic_sha256",
                "normalized_weighted_sum",
                protocol["baseline_quantile_proxy_policy"],
            ),
            "B4": (
                FORMAL_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
                "formal_occupancy_checkpoint_semantic_sha256",
                "learned_occupancy_aggregator",
                protocol["baseline_quantile_proxy_policy"],
            ),
        }[method]
        actual_artifact_contract = (
            table["checkpoint_layout_version"],
            table["checkpoint_digest_kind"],
            table["score_definition"],
            table["quantile_proxy_policy"],
        )
        if actual_artifact_contract != expected_artifact_contract:
            raise PredictionTableContractError(
                f"prediction method artifact contract mismatch for {method}"
            )
        if table.get("evaluation_record_collection_layout_version") != (
            RISK_EVALUATION_COLLECTION_LAYOUT_VERSION
        ) or table.get("evaluation_record_collection_digest_sha256") != (
            evaluation_records.collection_semantic_digest_sha256
        ):
            raise PredictionTableContractError("evaluation record provenance mismatch")
        if table.get("occupancy_sidecar_collection_digest_sha256") != sidecar_digest:
            raise PredictionTableContractError("occupancy sidecar provenance mismatch")
        if table.get("prediction_protocol_layout_version") != (
            PREDICTION_PROTOCOL_LAYOUT_VERSION
        ) or table.get("prediction_protocol_digest_sha256") != protocol[
            "protocol_digest_sha256"
        ]:
            raise PredictionTableContractError("prediction protocol mismatch")
        method_rows = tuple(
            _canonical_json_bytes(
                {
                    key: row[key]
                    for key in RISK_EVALUATION_RECORD_FIELDS
                }
            )
            for row in table["rows"]
        )
        if method_rows != expected_rows:
            raise PredictionTableContractError("prediction cohort evaluation rows mismatch")
        current = {
            key: table[key]
            for key in (
                "risk_dataset_family_digest",
                "risk_dataset_manifest_digest",
                "evaluation_record_collection_digest_sha256",
                "occupancy_sidecar_collection_digest_sha256",
                "prediction_protocol_digest_sha256",
                "cohort_binding_digest_sha256",
                "cohort_digest_sha256",
                "seed",
                "config_digest_sha256",
            )
        }
        if common_values is None:
            common_values = current
        elif current != common_values:
            raise PredictionTableContractError("shared prediction cohort/protocol mismatch")
        validated[method] = table
    if validated["B1"]["checkpoint_layout_version"] != BASELINE_SPEC_LAYOUT_VERSION:
        raise PredictionTableContractError("B1 must bind a deterministic baseline spec")
    if validated["B2"]["checkpoint_layout_version"] != BASELINE_SPEC_LAYOUT_VERSION:
        raise PredictionTableContractError("B2 must bind a deterministic baseline spec")
    if validated["B3"]["checkpoint_digest"] != validated["B4"]["checkpoint_digest"]:
        raise PredictionTableContractError("B3/B4 must bind the same selected checkpoint")
    return validated


def _source_values(
    source: Mapping[str, object],
    *,
    split: str,
) -> tuple[LoadedRiskDataset, LoadedRiskEvaluationCollection, str]:
    if not isinstance(source, Mapping) or set(source) != {
        "dataset",
        "evaluation_records",
        "occupancy_sidecar_collection_digest_sha256",
    }:
        raise PredictionTableContractError(f"{split} prediction source fields mismatch")
    dataset = source["dataset"]
    evaluation = source["evaluation_records"]
    if not isinstance(dataset, LoadedRiskDataset) or dataset.split != split:
        raise PredictionTableContractError(f"{split} dataset source mismatch")
    if not isinstance(evaluation, LoadedRiskEvaluationCollection):
        raise PredictionTableContractError(f"{split} evaluation source mismatch")
    sidecar_digest = _sha256(
        source["occupancy_sidecar_collection_digest_sha256"],
        label=f"{split} occupancy sidecar digest",
    )
    return dataset, evaluation, sidecar_digest


def _collection_manifest_digest(manifest: Mapping[str, object]) -> str:
    return _semantic_digest(
        b"unified-prediction-collection-v1\0",
        {
            key: value
            for key, value in manifest.items()
            if key != "collection_semantic_digest_sha256"
        },
    )


def _build_collection_manifest(
    *,
    dataset_family: LoadedRiskDatasetFamily,
    protocol: Mapping[str, object],
    tables_by_split: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, object]:
    split_entries: dict[str, object] = {}
    method_artifacts: dict[str, object] = {}
    for split in tables_by_split:
        tables = tables_by_split[split]
        first = tables[UNIFIED_PREDICTION_METHODS[0]]
        split_entries[split] = {
            "risk_dataset_manifest_digest": first["risk_dataset_manifest_digest"],
            "evaluation_record_collection_digest_sha256": first[
                "evaluation_record_collection_digest_sha256"
            ],
            "occupancy_sidecar_collection_digest_sha256": first[
                "occupancy_sidecar_collection_digest_sha256"
            ],
            "cohort_binding_digest_sha256": first["cohort_binding_digest_sha256"],
            "cohort_digest_sha256": first["cohort_digest_sha256"],
            "sample_count": len(first["rows"]),
            "tables": {
                method: {
                    "relative_path": f"{split}/{method}.json",
                    "semantic_digest_sha256": tables[method]["semantic_digest"],
                }
                for method in UNIFIED_PREDICTION_METHODS
            },
        }
        for method in UNIFIED_PREDICTION_METHODS:
            table = tables[method]
            artifact = {
                "layout_version": table["checkpoint_layout_version"],
                "digest_sha256": table["checkpoint_digest"],
                "digest_kind": table["checkpoint_digest_kind"],
                "score_definition": table["score_definition"],
                "quantile_proxy_policy": table["quantile_proxy_policy"],
            }
            previous = method_artifacts.setdefault(method, artifact)
            if previous != artifact:
                raise PredictionTableContractError(
                    f"method artifact differs across splits for {method}"
                )
    manifest: dict[str, object] = {
        "collection_layout_version": UNIFIED_PREDICTION_COLLECTION_LAYOUT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "publication_stage": (
            "complete" if set(tables_by_split) == {"calibration", "test"} else "calibration"
        ),
        "risk_dataset_family_layout_version": RISK_DATASET_FAMILY_LAYOUT_VERSION,
        "risk_dataset_family_digest": dataset_family.risk_dataset_family_digest,
        "prediction_protocol_layout_version": PREDICTION_PROTOCOL_LAYOUT_VERSION,
        "prediction_protocol_digest_sha256": protocol["protocol_digest_sha256"],
        "methods": list(UNIFIED_PREDICTION_METHODS),
        "method_artifacts": method_artifacts,
        "splits": split_entries,
    }
    manifest["collection_semantic_digest_sha256"] = _collection_manifest_digest(
        manifest
    )
    return manifest


def _validate_collection_inputs(
    *,
    dataset_family: LoadedRiskDatasetFamily,
    protocol: Mapping[str, object],
    split_sources: Mapping[str, Mapping[str, object]],
    tables_by_split: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[str, object], dict[str, dict[str, dict[str, Any]]]]:
    if set(split_sources) not in ({"calibration"}, {"calibration", "test"}):
        raise PredictionTableContractError(
            "prediction collection splits must be calibration or calibration+test"
        )
    if set(tables_by_split) != set(split_sources):
        raise PredictionTableContractError("prediction tables/source splits mismatch")
    checked_protocol = validate_prediction_protocol(protocol)
    checked_tables: dict[str, dict[str, dict[str, Any]]] = {}
    for split in split_sources:
        dataset, evaluation, sidecar_digest = _source_values(
            split_sources[split],
            split=split,
        )
        checked_tables[split] = validate_shared_prediction_tables(
            tables_by_split[split],
            dataset_family=dataset_family,
            dataset=dataset,
            evaluation_records=evaluation,
            expected_protocol=checked_protocol,
            expected_sidecar_collection_digest_sha256=sidecar_digest,
        )
    if set(checked_tables) == {"calibration", "test"}:
        assert_calibration_test_isolation(
            checked_tables["calibration"]["risk-r0"]["rows"],
            checked_tables["test"]["risk-r0"]["rows"],
            dataset_family=dataset_family,
        )
    return checked_protocol, checked_tables


def _json_file_bytes(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise PredictionTableContractError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise PredictionTableContractError(f"{label} must be a regular file")


def _read_canonical_json(path: Path, *, label: str) -> dict[str, object]:
    _require_regular(path, label=label)
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PredictionTableContractError(f"{label} is not JSON") from exc
    if not isinstance(value, dict) or raw != _json_file_bytes(value):
        raise PredictionTableContractError(f"{label} is not canonical JSON")
    return value


def _write_checksums(root: Path) -> bytes:
    paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"checksums.sha256", ".producer-complete"}
    )
    payload = "".join(
        f"{_sha256_file(root / relative)}  {relative}\n" for relative in paths
    ).encode("ascii")
    (root / "checksums.sha256").write_bytes(payload)
    return payload


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        elif path.is_dir() and not path.is_symlink():
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def publish_unified_prediction_collection(
    output_dir: str | Path,
    *,
    dataset_family: LoadedRiskDatasetFamily,
    protocol: Mapping[str, object],
    split_sources: Mapping[str, Mapping[str, object]],
    tables_by_split: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Path:
    """Atomically publish six calibration tables or the complete two-split bundle."""

    output = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite prediction collection: {output}")
    checked_protocol, checked_tables = _validate_collection_inputs(
        dataset_family=dataset_family,
        protocol=protocol,
        split_sources=split_sources,
        tables_by_split=tables_by_split,
    )
    manifest = _build_collection_manifest(
        dataset_family=dataset_family,
        protocol=checked_protocol,
        tables_by_split=checked_tables,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        (staging / "prediction_manifest.json").write_bytes(_json_file_bytes(manifest))
        (staging / "prediction_protocol.json").write_bytes(
            _json_file_bytes(checked_protocol)
        )
        for split, tables in checked_tables.items():
            split_root = staging / split
            split_root.mkdir()
            for method, table in tables.items():
                (split_root / f"{method}.json").write_bytes(_json_file_bytes(table))
        checksums = _write_checksums(staging)
        marker = {
            "collection_layout_version": UNIFIED_PREDICTION_COLLECTION_LAYOUT_VERSION,
            "collection_semantic_digest_sha256": manifest[
                "collection_semantic_digest_sha256"
            ],
            "prediction_manifest_sha256": _sha256_file(
                staging / "prediction_manifest.json"
            ),
            "checksums_sha256": hashlib.sha256(checksums).hexdigest(),
        }
        (staging / ".producer-complete").write_bytes(_json_file_bytes(marker))
        _fsync_tree(staging)
        load_unified_prediction_collection(
            staging,
            dataset_family=dataset_family,
            split_sources=split_sources,
        )
        atomic_rename_noreplace(staging, output)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return output


def _parse_checksums(root: Path) -> dict[str, str]:
    _require_regular(root / "checksums.sha256", label="prediction checksums")
    result: dict[str, str] = {}
    for line in (root / "checksums.sha256").read_text(encoding="ascii").splitlines():
        if "  " not in line:
            raise PredictionTableContractError("prediction checksum line is malformed")
        digest, relative = line.split("  ", 1)
        _sha256(digest, label=f"checksum {relative}")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise PredictionTableContractError("prediction checksum path is unsafe")
        if relative in result:
            raise PredictionTableContractError("prediction checksum path is duplicated")
        result[relative] = digest
    return result


def load_unified_prediction_collection(
    root: str | Path,
    *,
    dataset_family: LoadedRiskDatasetFamily,
    split_sources: Mapping[str, Mapping[str, object]],
) -> LoadedUnifiedPredictionCollection:
    """Reload and reauthenticate a unified prediction publication."""

    output = Path(os.path.abspath(os.fspath(root)))
    try:
        mode = output.lstat().st_mode
    except OSError as exc:
        raise PredictionTableContractError("prediction collection root is missing") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise PredictionTableContractError("prediction collection root must be a directory")
    expected_splits = set(split_sources)
    expected_files = {
        "prediction_manifest.json",
        "prediction_protocol.json",
        "checksums.sha256",
        ".producer-complete",
        *(
            f"{split}/{method}.json"
            for split in expected_splits
            for method in UNIFIED_PREDICTION_METHODS
        ),
    }
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    actual_dirs = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_dir()
    }
    if actual_files != expected_files or actual_dirs != expected_splits:
        raise PredictionTableContractError("prediction collection file layout mismatch")
    if any(path.is_symlink() for path in output.rglob("*")):
        raise PredictionTableContractError("prediction collection forbids symlinks")
    checksums = _parse_checksums(output)
    checksum_targets = expected_files - {"checksums.sha256", ".producer-complete"}
    if set(checksums) != checksum_targets:
        raise PredictionTableContractError("prediction checksum coverage mismatch")
    for relative, digest in checksums.items():
        if _sha256_file(output / relative) != digest:
            raise PredictionTableContractError(f"prediction checksum mismatch: {relative}")
    manifest = _read_canonical_json(
        output / "prediction_manifest.json",
        label="prediction manifest",
    )
    protocol = validate_prediction_protocol(
        _read_canonical_json(
            output / "prediction_protocol.json",
            label="prediction protocol",
        )
    )
    tables: dict[str, dict[str, dict[str, Any]]] = {}
    for split in split_sources:
        tables[split] = {}
        for method in UNIFIED_PREDICTION_METHODS:
            table = _read_canonical_json(
                output / split / f"{method}.json",
                label=f"{split} {method} prediction table",
            )
            tables[split][method] = table
    checked_protocol, checked_tables = _validate_collection_inputs(
        dataset_family=dataset_family,
        protocol=protocol,
        split_sources=split_sources,
        tables_by_split=tables,
    )
    expected_manifest = _build_collection_manifest(
        dataset_family=dataset_family,
        protocol=checked_protocol,
        tables_by_split=checked_tables,
    )
    if manifest != expected_manifest:
        raise PredictionTableContractError("prediction manifest semantic mismatch")
    marker = _read_canonical_json(
        output / ".producer-complete",
        label="prediction completion marker",
    )
    expected_marker = {
        "collection_layout_version": UNIFIED_PREDICTION_COLLECTION_LAYOUT_VERSION,
        "collection_semantic_digest_sha256": manifest[
            "collection_semantic_digest_sha256"
        ],
        "prediction_manifest_sha256": _sha256_file(
            output / "prediction_manifest.json"
        ),
        "checksums_sha256": _sha256_file(output / "checksums.sha256"),
    }
    if marker != expected_marker:
        raise PredictionTableContractError("prediction completion marker mismatch")
    return LoadedUnifiedPredictionCollection(
        root=output,
        manifest=manifest,
        protocol=checked_protocol,
        tables_by_split=checked_tables,
        collection_semantic_digest_sha256=str(
            manifest["collection_semantic_digest_sha256"]
        ),
    )


__all__ = [
    "BASELINE_SPEC_LAYOUT_VERSION",
    "PREDICTION_COHORT_BINDING_LAYOUT_VERSION",
    "PREDICTION_PROTOCOL_LAYOUT_VERSION",
    "UNIFIED_PREDICTION_COLLECTION_LAYOUT_VERSION",
    "LoadedUnifiedPredictionCollection",
    "PredictionMethodArtifact",
    "PredictionTableContractError",
    "UNIFIED_PREDICTION_METHODS",
    "baseline_spec_digest",
    "build_prediction_protocol",
    "build_production_prediction_table",
    "load_unified_prediction_collection",
    "publish_unified_prediction_collection",
    "score_unified_prediction_batch",
    "validate_prediction_protocol",
    "validate_shared_prediction_tables",
]
