#!/usr/bin/env python
"""Produce shared R0/R1/B1--B4 calibration or complete prediction bundles."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any

import numpy as np
import torch


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.calibration.grouped_calibration import (  # noqa: E402
    validate_grouped_calibration_artifact,
)
from src.calibration.split_conformal import (  # noqa: E402
    validate_calibration_artifact,
)
from src.datasets.risk_dataset_seal import (  # noqa: E402
    LoadedRiskDatasetFamily,
    load_risk_dataset_family,
)
from src.datasets.risk_evaluation_store import (  # noqa: E402
    LoadedRiskEvaluationCollection,
    load_risk_evaluation_collection,
)
from src.datasets.risk_training_store import (  # noqa: E402
    AuthenticatedOccupancySnapshot,
    load_authenticated_occupancy_snapshot,
)
from src.evaluation.prediction_tables import (  # noqa: E402
    BASELINE_SPEC_LAYOUT_VERSION,
    PredictionMethodArtifact,
    PredictionTableContractError,
    UNIFIED_PREDICTION_METHODS,
    baseline_spec_digest,
    build_production_prediction_table,
    load_unified_prediction_collection,
    publish_unified_prediction_collection,
    score_unified_prediction_batch,
    validate_prediction_protocol,
)
from src.models.occupancy_baseline import (  # noqa: E402
    ConvGRUOccupancyPredictor,
    LearnedOccupancyRiskAggregator,
)
from src.models.risk_model import load_risk_checkpoint  # noqa: E402
from src.training.occupancy_trainer import (  # noqa: E402
    FORMAL_PRODUCTION_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
    load_formal_production_occupancy_checkpoint,
    validate_formal_occupancy_training_publication,
)


PREDICTION_PRODUCER_VERSION = "sop10_unified_prediction_producer_v1"


class PredictionProducerError(ValueError):
    """Raised when formal inference inputs are incomplete or inconsistent."""


@dataclass(frozen=True)
class PredictionProducerRequest:
    stage: str
    dataset_family_root: Path
    prediction_protocol: Path
    risk_r0_training_root: Path
    risk_r1_training_root: Path
    occupancy_training_root: Path
    calibration_dataset_seal_root: Path
    calibration_risk_collection_root: Path
    calibration_sidecar_collection_root: Path
    calibration_evaluation_collection_root: Path
    test_dataset_seal_root: Path | None
    test_risk_collection_root: Path | None
    test_sidecar_collection_root: Path | None
    test_evaluation_collection_root: Path | None
    training_cache_root: Path
    calibration_prediction_root: Path | None
    calibration_artifact_root: Path | None
    output_dir: Path
    batch_size: int
    seed: int
    device: str


@dataclass(frozen=True)
class _SelectedModels:
    dataset_family: LoadedRiskDatasetFamily
    protocol: dict[str, object]
    risk_models: Mapping[str, torch.nn.Module]
    occupancy_model: ConvGRUOccupancyPredictor
    learned_aggregator: LearnedOccupancyRiskAggregator
    method_artifacts: Mapping[str, PredictionMethodArtifact]
    b2_tau_s: float
    b2_a_max_s: float
    sigma_time_s: float
    config_digest_sha256: str


@dataclass(frozen=True)
class _PredictionSplitSource:
    dataset: object
    snapshot: AuthenticatedOccupancySnapshot
    evaluation_records: LoadedRiskEvaluationCollection
    occupancy_sidecar_collection_digest_sha256: str


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
        raise PredictionProducerError("producer metadata must be finite JSON") from exc


def _require_regular_file(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise PredictionProducerError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise PredictionProducerError(f"{label} must be a regular file")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require_regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PredictionProducerError(f"{label} is not JSON") from exc
    if not isinstance(value, dict):
        raise PredictionProducerError(f"{label} must contain an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PredictionProducerError(f"{label} must be a lowercase SHA-256")
    return value


def _parse_checksums(path: Path) -> dict[str, str]:
    _require_regular_file(path, label="checksum manifest")
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PredictionProducerError("unable to read checksum manifest") from exc
    result: dict[str, str] = {}
    for line in lines:
        if "  " not in line:
            raise PredictionProducerError("checksum manifest is malformed")
        digest, name = line.split("  ", 1)
        _require_sha256(digest, label=f"checksum {name}")
        if not name or Path(name).name != name or name in result:
            raise PredictionProducerError("checksum manifest path is invalid")
        result[name] = digest
    return result


def _load_selected_risk_model(
    root: Path,
    *,
    method_id: str,
    dataset_family: LoadedRiskDatasetFamily,
    seed: int,
) -> tuple[torch.nn.Module, PredictionMethodArtifact]:
    if root.is_symlink() or not root.is_dir():
        raise PredictionProducerError(f"{method_id} training root is invalid")
    manifest = _read_json(root / "training_manifest.json", label=f"{method_id} manifest")
    marker = _read_json(root / ".producer-complete", label=f"{method_id} marker")
    checksums = _parse_checksums(root / "checksums.sha256")
    if manifest.get("mode") != "production" or manifest.get("stage") != "formal_50k":
        raise PredictionProducerError(f"{method_id} is not a formal production training")
    expected_variant = method_id.removeprefix("risk-")
    if manifest.get("variant") != expected_variant:
        raise PredictionProducerError(f"{method_id} training variant mismatch")
    if marker.get("semantic_digest_sha256") != manifest.get("semantic_digest_sha256"):
        raise PredictionProducerError(f"{method_id} completion marker mismatch")
    checkpoint_path = root / "best_checkpoint.pt"
    if "best_checkpoint.pt" not in checksums or _sha256_file(checkpoint_path) != checksums[
        "best_checkpoint.pt"
    ]:
        raise PredictionProducerError(f"{method_id} best checkpoint checksum mismatch")
    model, checkpoint = load_risk_checkpoint(checkpoint_path, expected_mode="production")
    provenance = checkpoint["provenance"]
    if (
        provenance.get("risk_dataset_manifest_digest")
        != dataset_family.members["train"]["risk_dataset_manifest_digest"]
        or provenance.get("validation_risk_dataset_manifest_digest")
        != dataset_family.members["val"]["risk_dataset_manifest_digest"]
    ):
        raise PredictionProducerError(
            f"{method_id} checkpoint train/validation member mismatch"
        )
    if (
        provenance.get("model_variant") != expected_variant
        or provenance.get("risk_dataset_family_digest")
        != dataset_family.risk_dataset_family_digest
        or provenance.get("global_cross_split_leakage") != "PROVEN"
        or provenance.get("scientific_claim_eligible") is not True
        or provenance.get("seed") != seed
    ):
        raise PredictionProducerError(f"{method_id} formal checkpoint provenance mismatch")
    digest = _require_sha256(
        checkpoint.get("checkpoint_semantic_digest_sha256"),
        label=f"{method_id} checkpoint digest",
    )
    bindings = manifest.get("artifact_semantic_bindings")
    if not isinstance(bindings, Mapping) or bindings.get("best_checkpoint.pt") != digest:
        raise PredictionProducerError(f"{method_id} manifest checkpoint binding mismatch")
    return model, PredictionMethodArtifact(
        method_id=method_id,
        layout_version="risk_model_checkpoint_v2",
        digest_sha256=digest,
        digest_kind="risk_checkpoint_semantic_sha256",
        score_definition="risk_model_collision_and_quantile_heads",
    )


def _positive_float(value: object, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise PredictionProducerError(f"{label} must be positive and finite")
    return number


def _load_selected_occupancy_models(
    root: Path,
    *,
    dataset_family: LoadedRiskDatasetFamily,
    seed: int,
) -> tuple[
    ConvGRUOccupancyPredictor,
    LearnedOccupancyRiskAggregator,
    dict[str, PredictionMethodArtifact],
    dict[str, float],
]:
    manifest = validate_formal_occupancy_training_publication(root)
    checkpoint = load_formal_production_occupancy_checkpoint(root / "best_checkpoint.pt")
    provenance = checkpoint["provenance"]
    if (
        provenance.get("train_risk_dataset_manifest_digest")
        != dataset_family.members["train"]["risk_dataset_manifest_digest"]
        or provenance.get("validation_risk_dataset_manifest_digest")
        != dataset_family.members["val"]["risk_dataset_manifest_digest"]
    ):
        raise PredictionProducerError(
            "formal occupancy checkpoint train/validation member mismatch"
        )
    if (
        checkpoint["checkpoint_role"] != "best"
        or provenance.get("risk_dataset_family_digest")
        != dataset_family.risk_dataset_family_digest
        or provenance.get("global_cross_split_leakage") != "PROVEN"
        or provenance.get("scientific_claim_eligible") is not True
        or provenance.get("test_samples_used_for_training_or_selection") != 0
        or manifest.get("validation_status") != "selected_on_authenticated_val"
    ):
        raise PredictionProducerError("formal occupancy checkpoint provenance mismatch")
    config = _read_json(root / "config_snapshot.json", label="occupancy config snapshot")
    if config.get("seed") != seed:
        raise PredictionProducerError("occupancy checkpoint seed mismatch")
    model_spec = checkpoint["model_spec"]
    model = ConvGRUOccupancyPredictor(
        hidden_channels=int(model_spec["hidden_channels"]),
        future_steps=15,
        kernel_size=int(model_spec["convgru_kernel_size"]),
    )
    aggregator = LearnedOccupancyRiskAggregator(
        future_steps=15,
        hidden_dim=int(model_spec["learned_aggregator_hidden_dim"]),
    )
    model.load_state_dict(checkpoint["b3_model_state_dict"], strict=True)
    aggregator.load_state_dict(checkpoint["b4_aggregator_state_dict"], strict=True)
    digest = _require_sha256(
        checkpoint["checkpoint_semantic_digest_sha256"],
        label="formal occupancy checkpoint digest",
    )
    artifacts = {
        "B3": PredictionMethodArtifact(
            method_id="B3",
            layout_version=FORMAL_PRODUCTION_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
            digest_sha256=digest,
            digest_kind="formal_occupancy_checkpoint_semantic_sha256",
            score_definition="normalized_weighted_sum",
        ),
        "B4": PredictionMethodArtifact(
            method_id="B4",
            layout_version=FORMAL_PRODUCTION_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
            digest_sha256=digest,
            digest_kind="formal_occupancy_checkpoint_semantic_sha256",
            score_definition="learned_occupancy_aggregator",
        ),
    }
    parameters = {
        "b2_tau_s": _positive_float(config.get("b2_tau_s"), label="b2_tau_s"),
        "b2_a_max_s": _positive_float(config.get("b2_a_max_s"), label="b2_a_max_s"),
        "sigma_time_s": _positive_float(
            config.get("sigma_time_s"), label="sigma_time_s"
        ),
    }
    return model, aggregator, artifacts, parameters


def _load_selected_models(request: PredictionProducerRequest) -> _SelectedModels:
    family = load_risk_dataset_family(request.dataset_family_root)
    if family.cross_split_audit.get("global_cross_split_leakage") != "PROVEN":
        raise PredictionProducerError("dataset family leakage gate is not PROVEN")
    protocol = validate_prediction_protocol(
        _read_json(request.prediction_protocol, label="prediction protocol")
    )
    r0_model, r0_artifact = _load_selected_risk_model(
        request.risk_r0_training_root,
        method_id="risk-r0",
        dataset_family=family,
        seed=request.seed,
    )
    r1_model, r1_artifact = _load_selected_risk_model(
        request.risk_r1_training_root,
        method_id="risk-r1",
        dataset_family=family,
        seed=request.seed,
    )
    occupancy_model, aggregator, occupancy_artifacts, parameters = (
        _load_selected_occupancy_models(
            request.occupancy_training_root,
            dataset_family=family,
            seed=request.seed,
        )
    )
    b1_digest = baseline_spec_digest("B1", **parameters)
    b2_digest = baseline_spec_digest("B2", **parameters)
    artifacts: dict[str, PredictionMethodArtifact] = {
        "risk-r0": r0_artifact,
        "risk-r1": r1_artifact,
        "B1": PredictionMethodArtifact(
            method_id="B1",
            layout_version=BASELINE_SPEC_LAYOUT_VERSION,
            digest_sha256=b1_digest,
            digest_kind="baseline_spec_sha256",
            score_definition="normalized_weighted_sum",
        ),
        "B2": PredictionMethodArtifact(
            method_id="B2",
            layout_version=BASELINE_SPEC_LAYOUT_VERSION,
            digest_sha256=b2_digest,
            digest_kind="baseline_spec_sha256",
            score_definition="normalized_weighted_sum",
        ),
        **occupancy_artifacts,
    }
    config_digest = hashlib.sha256(
        _canonical_json_bytes(
            {
                "producer_version": PREDICTION_PRODUCER_VERSION,
                "seed": request.seed,
                "protocol_digest_sha256": protocol["protocol_digest_sha256"],
                "method_artifacts": {
                    method: artifact.digest_sha256
                    for method, artifact in artifacts.items()
                },
                "baseline_parameters": parameters,
            }
        )
    ).hexdigest()
    return _SelectedModels(
        dataset_family=family,
        protocol=protocol,
        risk_models={"risk-r0": r0_model, "risk-r1": r1_model},
        occupancy_model=occupancy_model,
        learned_aggregator=aggregator,
        method_artifacts=artifacts,
        b2_tau_s=parameters["b2_tau_s"],
        b2_a_max_s=parameters["b2_a_max_s"],
        sigma_time_s=parameters["sigma_time_s"],
        config_digest_sha256=config_digest,
    )


def _split_paths(request: PredictionProducerRequest, split: str) -> tuple[Path, Path, Path, Path]:
    prefix = "calibration" if split == "calibration" else "test"
    values = tuple(
        getattr(request, f"{prefix}_{suffix}")
        for suffix in (
            "dataset_seal_root",
            "risk_collection_root",
            "sidecar_collection_root",
            "evaluation_collection_root",
        )
    )
    if any(value is None for value in values):
        raise PredictionProducerError(f"{split} source roots are incomplete")
    return values  # type: ignore[return-value]


def _load_split_source(
    request: PredictionProducerRequest,
    split: str,
) -> _PredictionSplitSource:
    seal_root, risk_root, sidecar_root, evaluation_root = _split_paths(request, split)
    family = load_risk_dataset_family(request.dataset_family_root)
    expected_digest = str(family.members[split]["risk_dataset_manifest_digest"])
    dataset, snapshot = load_authenticated_occupancy_snapshot(
        seal_root,
        collection_root=risk_root,
        sidecar_root=sidecar_root,
        expected_split=split,
        cache_root=request.training_cache_root,
        expected_manifest_digest=expected_digest,
    )
    evaluation = load_risk_evaluation_collection(
        evaluation_root,
        dataset=dataset,
        expected_manifest_digest=expected_digest,
    )
    if snapshot.sample_ids != evaluation.sample_ids:
        raise PredictionProducerError(f"{split} snapshot/evaluation sample order mismatch")
    sidecar_digest = _require_sha256(
        snapshot.source_identity.get("occupancy_sidecar_collection_digest_sha256"),
        label=f"{split} sidecar collection digest",
    )
    return _PredictionSplitSource(
        dataset=dataset,
        snapshot=snapshot,
        evaluation_records=evaluation,
        occupancy_sidecar_collection_digest_sha256=sidecar_digest,
    )


def _score_split(
    request: PredictionProducerRequest,
    selected: _SelectedModels,
    source: _PredictionSplitSource,
    protocol: Mapping[str, object],
) -> dict[str, dict[str, Any]]:
    if protocol != selected.protocol:
        raise PredictionProducerError("scoring protocol differs from selected protocol")
    accumulated = {
        method: {"p_collision": [], "quantiles": []}
        for method in UNIFIED_PREDICTION_METHODS
    }
    sample_ids = source.evaluation_records.sample_ids
    for start in range(0, len(sample_ids), request.batch_size):
        batch_ids = sample_ids[start : start + request.batch_size]
        batch = source.snapshot.batch(batch_ids)
        values = score_unified_prediction_batch(
            risk_models=selected.risk_models,
            occupancy_model=selected.occupancy_model,
            learned_aggregator=selected.learned_aggregator,
            model_inputs=batch.model_inputs,
            query_inputs=batch.query_inputs,
            b2_tau_s=selected.b2_tau_s,
            b2_a_max_s=selected.b2_a_max_s,
            sigma_time_s=selected.sigma_time_s,
            device=request.device,
        )
        for method in UNIFIED_PREDICTION_METHODS:
            accumulated[method]["p_collision"].append(
                values[method]["p_collision"].numpy()
            )
            accumulated[method]["quantiles"].append(
                values[method]["quantiles"].numpy()
            )
    tables: dict[str, dict[str, Any]] = {}
    for method in UNIFIED_PREDICTION_METHODS:
        predictions = {
            key: np.concatenate(accumulated[method][key], axis=0)
            for key in ("p_collision", "quantiles")
        }
        tables[method] = build_production_prediction_table(
            dataset_family=selected.dataset_family,
            dataset=source.dataset,
            evaluation_records=source.evaluation_records,
            occupancy_sidecar_collection_digest_sha256=(
                source.occupancy_sidecar_collection_digest_sha256
            ),
            method_artifact=selected.method_artifacts[method],
            predictions=predictions,
            protocol=protocol,
            seed=request.seed,
            config_digest_sha256=selected.config_digest_sha256,
        )
    return tables


def _source_mapping(source: _PredictionSplitSource) -> dict[str, object]:
    return {
        "dataset": source.dataset,
        "evaluation_records": source.evaluation_records,
        "occupancy_sidecar_collection_digest_sha256": (
            source.occupancy_sidecar_collection_digest_sha256
        ),
    }


def _assert_current_method_artifacts(
    tables: Mapping[str, Mapping[str, Any]],
    selected: _SelectedModels,
) -> None:
    for method in UNIFIED_PREDICTION_METHODS:
        artifact = selected.method_artifacts[method]
        table = tables[method]
        if (
            table["checkpoint_layout_version"] != artifact.layout_version
            or table["checkpoint_digest"] != artifact.digest_sha256
            or table["checkpoint_digest_kind"] != artifact.digest_kind
            or table["config_digest_sha256"] != selected.config_digest_sha256
        ):
            raise PredictionProducerError(
                f"calibration predictions use a different selected artifact for {method}"
            )


def _load_calibration_prediction_gate(
    request: PredictionProducerRequest,
    selected: _SelectedModels,
    source: _PredictionSplitSource,
) -> tuple[dict[str, object], Mapping[str, Mapping[str, Any]]]:
    if request.calibration_prediction_root is None:
        raise PredictionProducerError(
            "complete stage requires --calibration-prediction-root"
        )
    loaded = load_unified_prediction_collection(
        request.calibration_prediction_root,
        dataset_family=selected.dataset_family,
        split_sources={"calibration": _source_mapping(source)},
    )
    if loaded.manifest["publication_stage"] != "calibration":
        raise PredictionProducerError("calibration prediction gate has the wrong stage")
    if loaded.protocol != selected.protocol:
        raise PredictionProducerError("calibration prediction protocol mismatch")
    tables = loaded.tables_by_split["calibration"]
    _assert_current_method_artifacts(tables, selected)
    return loaded.protocol, tables


def _verify_calibration_seal(root: Path, *, method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise PredictionProducerError(f"{method} calibration root is invalid")
    expected_names = {
        "calibration.json",
        "manifest.json",
        "checksums.sha256",
        ".producer-complete",
    }
    if {path.name for path in root.iterdir()} != expected_names:
        raise PredictionProducerError(f"{method} calibration publication is incomplete")
    checksums = _parse_checksums(root / "checksums.sha256")
    if set(checksums) != {"calibration.json", "manifest.json"}:
        raise PredictionProducerError(f"{method} calibration checksum coverage mismatch")
    for name, digest in checksums.items():
        if _sha256_file(root / name) != digest:
            raise PredictionProducerError(f"{method} calibration checksum mismatch")
    artifact = _read_json(root / "calibration.json", label=f"{method} calibration")
    manifest = _read_json(root / "manifest.json", label=f"{method} calibration manifest")
    marker = _read_json(root / ".producer-complete", label=f"{method} calibration marker")
    checksums_bytes = (root / "checksums.sha256").read_bytes()
    expected_marker = {
        "calibration_artifact_layout_version": artifact[
            "calibration_artifact_layout_version"
        ],
        "calibration_semantic_digest": artifact["semantic_digest"],
        "manifest_sha256": _sha256_file(root / "manifest.json"),
        "checksums_sha256": hashlib.sha256(checksums_bytes).hexdigest(),
    }
    if marker != expected_marker:
        raise PredictionProducerError(f"{method} calibration completion marker mismatch")
    return artifact, manifest


def _load_calibration_artifact_gate(
    request: PredictionProducerRequest,
    selected: _SelectedModels,
    source: _PredictionSplitSource,
    protocol: Mapping[str, object],
    tables: Mapping[str, Mapping[str, Any]],
) -> None:
    del source
    if request.calibration_artifact_root is None:
        raise PredictionProducerError(
            "complete stage requires --calibration-artifact-root"
        )
    grouped_protocol = protocol["grouped_calibration"]
    assert isinstance(grouped_protocol, Mapping)
    for method in UNIFIED_PREDICTION_METHODS:
        artifact, manifest = _verify_calibration_seal(
            request.calibration_artifact_root / method,
            method=method,
        )
        table = tables[method]
        expected = {
            key: table[key]
            for key in (
                "method_id",
                "checkpoint_layout_version",
                "checkpoint_digest",
                "checkpoint_digest_kind",
                "seed",
                "channel_spec",
                "config_digest_sha256",
                "risk_dataset_family_layout_version",
                "risk_dataset_family_digest",
                "evaluation_record_collection_layout_version",
                "evaluation_record_collection_digest_sha256",
                "occupancy_sidecar_collection_digest_sha256",
                "prediction_protocol_layout_version",
                "prediction_protocol_digest_sha256",
                "cohort_binding_digest_sha256",
                "score_definition",
                "quantile_proxy_policy",
            )
        }
        if method in {"B1", "B2", "B3", "B4"}:
            expected["prediction_semantics"] = table["prediction_semantics"]
        validated = validate_calibration_artifact(
            artifact,
            expected_mode="production",
            expected_provenance=expected,
            dataset_family=selected.dataset_family,
        )
        if (
            validated["prediction_table_semantic_digest"] != table["semantic_digest"]
            or validated["calibration_cohort_digest_sha256"]
            != table["cohort_digest_sha256"]
            or validated["alpha"] != protocol["alpha"]
            or validated["prediction_key"] != protocol["prediction_key"]
            or manifest.get("prediction_protocol_digest_sha256")
            != protocol["protocol_digest_sha256"]
        ):
            raise PredictionProducerError(f"{method} calibration protocol/table mismatch")
        grouped = validate_grouped_calibration_artifact(
            validated.get("grouped"),
            expected_alpha=float(protocol["alpha"]),
            expected_prediction_key=str(protocol["prediction_key"]),
            expected_global=validated["global"],
        )
        if (
            grouped["min_group_size"] != grouped_protocol["min_group_size"]
            or grouped["group_dimensions"] != grouped_protocol["group_dimensions"]
            or grouped["continuous_group_bins"]
            != grouped_protocol["continuous_group_bins"]
            or grouped["combination_policy"] != grouped_protocol["combination_policy"]
        ):
            raise PredictionProducerError(f"{method} grouped calibration mismatch")


def _publish_prediction_result(
    request: PredictionProducerRequest,
    protocol: Mapping[str, object],
    sources: Mapping[str, _PredictionSplitSource],
    tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, object]:
    family = load_risk_dataset_family(request.dataset_family_root)
    source_mappings = {split: _source_mapping(source) for split, source in sources.items()}
    output = publish_unified_prediction_collection(
        request.output_dir,
        dataset_family=family,
        protocol=protocol,
        split_sources=source_mappings,
        tables_by_split=tables,
    )
    loaded = load_unified_prediction_collection(
        output,
        dataset_family=family,
        split_sources=source_mappings,
    )
    return {
        "producer_version": PREDICTION_PRODUCER_VERSION,
        "publication_stage": loaded.manifest["publication_stage"],
        "output_dir": str(output),
        "risk_dataset_family_digest": family.risk_dataset_family_digest,
        "prediction_protocol_digest_sha256": protocol["protocol_digest_sha256"],
        "collection_semantic_digest_sha256": (
            loaded.collection_semantic_digest_sha256
        ),
    }


def run_prediction_producer(request: PredictionProducerRequest) -> dict[str, object]:
    """Run calibration-only production or complete it after calibration seals."""

    selected = _load_selected_models(request)
    calibration_source = _load_split_source(request, "calibration")
    if request.stage == "calibration":
        calibration_tables = _score_split(
            request,
            selected,
            calibration_source,
            selected.protocol,
        )
        return _publish_prediction_result(
            request,
            selected.protocol,
            {"calibration": calibration_source},
            {"calibration": calibration_tables},
        )
    if request.stage != "complete":
        raise PredictionProducerError("stage must be calibration or complete")
    protocol, calibration_tables = _load_calibration_prediction_gate(
        request,
        selected,
        calibration_source,
    )
    _load_calibration_artifact_gate(
        request,
        selected,
        calibration_source,
        protocol,
        calibration_tables,
    )
    test_source = _load_split_source(request, "test")
    test_tables = _score_split(request, selected, test_source, protocol)
    return _publish_prediction_result(
        request,
        protocol,
        {"calibration": calibration_source, "test": test_source},
        {"calibration": calibration_tables, "test": test_tables},
    )


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibration", "complete"), required=True)
    parser.add_argument("--dataset-family-root", type=Path, required=True)
    parser.add_argument("--prediction-protocol", type=Path, required=True)
    parser.add_argument("--risk-r0-training-root", type=Path, required=True)
    parser.add_argument("--risk-r1-training-root", type=Path, required=True)
    parser.add_argument("--occupancy-training-root", type=Path, required=True)
    for split in ("calibration", "test"):
        required = split == "calibration"
        parser.add_argument(f"--{split}-dataset-seal-root", type=Path, required=required)
        parser.add_argument(f"--{split}-risk-collection-root", type=Path, required=required)
        parser.add_argument(f"--{split}-sidecar-collection-root", type=Path, required=required)
        parser.add_argument(
            f"--{split}-evaluation-collection-root",
            type=Path,
            required=required,
        )
    parser.add_argument("--training-cache-root", type=Path, required=True)
    parser.add_argument("--calibration-prediction-root", type=Path)
    parser.add_argument("--calibration-artifact-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--seed", type=_nonnegative_int, required=True)
    parser.add_argument("--device", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        request = PredictionProducerRequest(**vars(args))
        report = run_prediction_producer(request)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        PredictionProducerError,
        PredictionTableContractError,
    ) as exc:
        detail = " ".join(str(exc).split()) or type(exc).__name__
        print(f"error: {detail}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
