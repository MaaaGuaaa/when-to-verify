"""Unified six-method prediction producer behavior."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from src.evaluation.prediction_tables import (
    UNIFIED_PREDICTION_METHODS,
    score_unified_prediction_batch,
)
from src.models.occupancy_baseline import (
    ConvGRUOccupancyPredictor,
    LearnedOccupancyRiskAggregator,
)
from src.models.risk_model import RiskModel


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "09_predict_risk.py"


def _model_inputs(batch_size: int = 3, grid_size: int = 8):
    return {
        "bev_history": torch.full(
            (batch_size, 8, 2, grid_size, grid_size),
            0.2,
            dtype=torch.float32,
        ),
        "state_channels": torch.full(
            (batch_size, 9, grid_size, grid_size),
            0.2,
            dtype=torch.float32,
        ),
        "trajectory_channels": torch.full(
            (batch_size, 4, grid_size, grid_size),
            0.2,
            dtype=torch.float32,
        ),
        "robot_state": torch.zeros((batch_size, 2), dtype=torch.float32),
    }


def _query_inputs(batch_size: int = 3, grid_size: int = 8):
    footprints = torch.zeros(
        (batch_size, 15, grid_size, grid_size),
        dtype=torch.float32,
    )
    footprints[:, :, 3:5, 3:5] = 1.0
    return {
        "robot_endpoint_footprints": footprints,
        "endpoint_times_s": torch.arange(1, 16, dtype=torch.float32) * 0.2,
    }


def test_one_batch_scores_all_six_methods_without_label_arguments() -> None:
    torch.manual_seed(7)
    result = score_unified_prediction_batch(
        risk_models={
            "risk-r0": RiskModel(variant="r0", hidden_channels=2),
            "risk-r1": RiskModel(variant="r1", hidden_channels=2),
        },
        occupancy_model=ConvGRUOccupancyPredictor(
            hidden_channels=2,
            future_steps=15,
            kernel_size=3,
        ),
        learned_aggregator=LearnedOccupancyRiskAggregator(
            future_steps=15,
            hidden_dim=4,
        ),
        model_inputs=_model_inputs(),
        query_inputs=_query_inputs(),
        b2_tau_s=2.0,
        b2_a_max_s=5.0,
        sigma_time_s=2.0,
        device="cpu",
    )

    assert tuple(result) == UNIFIED_PREDICTION_METHODS
    for method, values in result.items():
        assert values["p_collision"].shape == (3,)
        assert values["quantiles"].shape == (3, 4)
        assert torch.isfinite(values["p_collision"]).all()
        assert torch.isfinite(values["quantiles"]).all()
        assert torch.all(values["quantiles"][:, :-1] <= values["quantiles"][:, 1:])
        if method in {"B1", "B2", "B3", "B4"}:
            assert torch.equal(
                values["quantiles"],
                values["p_collision"][:, None].expand(-1, 4),
            )


@pytest.fixture
def producer_module():
    spec = importlib.util.spec_from_file_location("prediction_producer_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_complete_stage_authenticates_calibration_before_opening_test(
    producer_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    request = SimpleNamespace(stage="complete")
    models = object()
    calibration_source = object()
    test_source = object()
    protocol = object()
    calibration_tables = object()
    test_tables = object()

    monkeypatch.setattr(
        producer_module,
        "_load_selected_models",
        lambda value: events.append("checkpoints") or models,
    )

    def load_source(value, split):
        events.append(f"source:{split}")
        return calibration_source if split == "calibration" else test_source

    monkeypatch.setattr(producer_module, "_load_split_source", load_source)
    monkeypatch.setattr(
        producer_module,
        "_load_calibration_prediction_gate",
        lambda value, selected, source: (
            events.append("calibration_predictions")
            or (protocol, calibration_tables)
        ),
    )
    monkeypatch.setattr(
        producer_module,
        "_load_calibration_artifact_gate",
        lambda value, selected, source, checked_protocol, tables: events.append(
            "calibration_artifacts"
        ),
    )
    monkeypatch.setattr(
        producer_module,
        "_score_split",
        lambda value, selected, source, checked_protocol: (
            events.append("score:test") or test_tables
        ),
    )
    monkeypatch.setattr(
        producer_module,
        "_publish_prediction_result",
        lambda value, checked_protocol, sources, tables: events.append("publish")
        or {"publication_stage": "complete"},
    )

    report = producer_module.run_prediction_producer(request)

    assert report == {"publication_stage": "complete"}
    assert events == [
        "checkpoints",
        "source:calibration",
        "calibration_predictions",
        "calibration_artifacts",
        "source:test",
        "score:test",
        "publish",
    ]


def test_calibration_gate_rejects_symlinked_checksum_manifest(
    producer_module,
    tmp_path: Path,
) -> None:
    root = tmp_path / "risk-r0"
    root.mkdir()
    artifact = {
        "calibration_artifact_layout_version": "risk_calibration_v3",
        "semantic_digest": "a" * 64,
    }
    manifest = {"method_id": "risk-r0"}
    for name, value in (
        ("calibration.json", artifact),
        ("manifest.json", manifest),
    ):
        (root / name).write_text(
            json.dumps(value, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    checksum_bytes = "".join(
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
        for name in ("calibration.json", "manifest.json")
    ).encode("ascii")
    external_checksums = tmp_path / "external-checksums.sha256"
    external_checksums.write_bytes(checksum_bytes)
    (root / "checksums.sha256").symlink_to(external_checksums)
    marker = {
        "calibration_artifact_layout_version": "risk_calibration_v3",
        "calibration_semantic_digest": "a" * 64,
        "manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
        "checksums_sha256": hashlib.sha256(checksum_bytes).hexdigest(),
    }
    (root / ".producer-complete").write_text(
        json.dumps(marker, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(producer_module.PredictionProducerError, match="regular file"):
        producer_module._verify_calibration_seal(root, method="risk-r0")


def test_calibration_gate_binds_table_score_and_evaluation_semantics(
    producer_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol_digest = "a" * 64
    protocol = {
        "alpha": 0.1,
        "prediction_key": "q90",
        "protocol_digest_sha256": protocol_digest,
        "grouped_calibration": {
            "min_group_size": 20,
            "group_dimensions": ["blind_type"],
            "continuous_group_bins": {"age_s": [0.0, 1.0]},
            "combination_policy": "one_dimension_at_a_time",
        },
    }
    tables = {
        method: {
            "method_id": method,
            "checkpoint_layout_version": "risk_model_checkpoint_v2",
            "checkpoint_digest": "b" * 64,
            "checkpoint_digest_kind": "risk_checkpoint_semantic_sha256",
            "seed": 42,
            "channel_spec": {"flat": ["x"]},
            "config_digest_sha256": "c" * 64,
            "risk_dataset_family_layout_version": "risk_dataset_family_v1",
            "risk_dataset_family_digest": "d" * 64,
            "evaluation_record_collection_layout_version": (
                "risk_evaluation_record_collection_v1"
            ),
            "evaluation_record_collection_digest_sha256": "e" * 64,
            "occupancy_sidecar_collection_digest_sha256": "f" * 64,
            "prediction_protocol_layout_version": (
                "shared_risk_prediction_protocol_v1"
            ),
            "prediction_protocol_digest_sha256": protocol_digest,
            "cohort_binding_digest_sha256": "1" * 64,
            "score_definition": f"score-{method}",
            "quantile_proxy_policy": f"quantiles-{method}",
            "prediction_semantics": (
                "scalar_baseline_score_repeated_for_common_calibration"
            ),
            "semantic_digest": f"{UNIFIED_PREDICTION_METHODS.index(method) + 2}" * 64,
            "cohort_digest_sha256": "9" * 64,
        }
        for method in UNIFIED_PREDICTION_METHODS
    }
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(
        producer_module,
        "_verify_calibration_seal",
        lambda root, method: ({}, {"prediction_protocol_digest_sha256": protocol_digest}),
    )

    def validate_artifact(value, *, expected_provenance, **kwargs):
        del value, kwargs
        captured.append(expected_provenance)
        method = str(expected_provenance["method_id"])
        table = tables[method]
        return {
            "prediction_table_semantic_digest": table["semantic_digest"],
            "calibration_cohort_digest_sha256": table["cohort_digest_sha256"],
            "alpha": 0.1,
            "prediction_key": "q90",
            "global": {},
            "grouped": {},
        }

    monkeypatch.setattr(
        producer_module,
        "validate_calibration_artifact",
        validate_artifact,
    )
    monkeypatch.setattr(
        producer_module,
        "validate_grouped_calibration_artifact",
        lambda value, **kwargs: protocol["grouped_calibration"],
    )
    request = SimpleNamespace(calibration_artifact_root=tmp_path)
    selected = SimpleNamespace(dataset_family=object())

    producer_module._load_calibration_artifact_gate(
        request,
        selected,
        object(),
        protocol,
        tables,
    )

    for method, provenance in zip(
        UNIFIED_PREDICTION_METHODS,
        captured,
        strict=True,
    ):
        assert provenance["evaluation_record_collection_layout_version"] == (
            tables[method]["evaluation_record_collection_layout_version"]
        )
        assert provenance["score_definition"] == tables[method]["score_definition"]
        assert provenance["quantile_proxy_policy"] == tables[method][
            "quantile_proxy_policy"
        ]


def test_risk_checkpoint_must_bind_family_train_and_validation_members(
    producer_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "risk-r0"
    root.mkdir()
    family = SimpleNamespace(
        risk_dataset_family_digest="a" * 64,
        members={
            "train": {"risk_dataset_manifest_digest": "b" * 64},
            "val": {"risk_dataset_manifest_digest": "c" * 64},
        },
    )
    manifest = {
        "mode": "production",
        "stage": "formal_50k",
        "variant": "r0",
        "semantic_digest_sha256": "d" * 64,
        "artifact_semantic_bindings": {"best_checkpoint.pt": "e" * 64},
    }
    marker = {"semantic_digest_sha256": "d" * 64}
    checkpoint = {
        "provenance": {
            "model_variant": "r0",
            "risk_dataset_family_digest": "a" * 64,
            "risk_dataset_manifest_digest": "f" * 64,
            "validation_risk_dataset_manifest_digest": "1" * 64,
            "global_cross_split_leakage": "PROVEN",
            "scientific_claim_eligible": True,
            "seed": 42,
        },
        "checkpoint_semantic_digest_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        producer_module,
        "_read_json",
        lambda path, **kwargs: manifest if path.name == "training_manifest.json" else marker,
    )
    monkeypatch.setattr(
        producer_module,
        "_parse_checksums",
        lambda path: {"best_checkpoint.pt": "2" * 64},
    )
    monkeypatch.setattr(producer_module, "_sha256_file", lambda path: "2" * 64)
    monkeypatch.setattr(
        producer_module,
        "load_risk_checkpoint",
        lambda path, **kwargs: (torch.nn.Identity(), checkpoint),
    )

    with pytest.raises(
        producer_module.PredictionProducerError,
        match="train/validation member",
    ):
        producer_module._load_selected_risk_model(
            root,
            method_id="risk-r0",
            dataset_family=family,
            seed=42,
        )


def test_occupancy_checkpoint_must_bind_family_train_and_validation_members(
    producer_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    family = SimpleNamespace(
        risk_dataset_family_digest="a" * 64,
        members={
            "train": {"risk_dataset_manifest_digest": "b" * 64},
            "val": {"risk_dataset_manifest_digest": "c" * 64},
        },
    )
    occupancy_model = ConvGRUOccupancyPredictor(
        hidden_channels=2,
        future_steps=15,
        kernel_size=3,
    )
    aggregator = LearnedOccupancyRiskAggregator(future_steps=15, hidden_dim=4)
    checkpoint = {
        "checkpoint_role": "best",
        "provenance": {
            "risk_dataset_family_digest": "a" * 64,
            "train_risk_dataset_manifest_digest": "f" * 64,
            "validation_risk_dataset_manifest_digest": "1" * 64,
            "global_cross_split_leakage": "PROVEN",
            "scientific_claim_eligible": True,
            "test_samples_used_for_training_or_selection": 0,
        },
        "model_spec": {
            "hidden_channels": 2,
            "convgru_kernel_size": 3,
            "learned_aggregator_hidden_dim": 4,
        },
        "b3_model_state_dict": occupancy_model.state_dict(),
        "b4_aggregator_state_dict": aggregator.state_dict(),
        "checkpoint_semantic_digest_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        producer_module,
        "validate_formal_occupancy_training_publication",
        lambda root: {"validation_status": "selected_on_authenticated_val"},
    )
    monkeypatch.setattr(
        producer_module,
        "load_formal_production_occupancy_checkpoint",
        lambda path: checkpoint,
    )
    monkeypatch.setattr(
        producer_module,
        "_read_json",
        lambda path, **kwargs: {
            "seed": 42,
            "b2_tau_s": 2.0,
            "b2_a_max_s": 5.0,
            "sigma_time_s": 2.0,
        },
    )

    with pytest.raises(
        producer_module.PredictionProducerError,
        match="train/validation member",
    ):
        producer_module._load_selected_occupancy_models(
            tmp_path,
            dataset_family=family,
            seed=42,
        )
