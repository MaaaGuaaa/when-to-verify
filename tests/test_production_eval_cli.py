"""Focused production metadata tests for the SOP10 calibration/eval CLIs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from src.evaluation.prediction_tables import build_prediction_protocol


ROOT = Path(__file__).parents[1]
FAMILY_DIGEST = "f" * 64
CALIBRATION_MEMBER_DIGEST = "a" * 64
TEST_MEMBER_DIGEST = "b" * 64
CALIBRATION_SAMPLE_IDS_DIGEST = "6" * 64
TEST_SAMPLE_IDS_DIGEST = "7" * 64
CHECKPOINT_DIGEST = "c" * 64
TABLE_DIGEST = "d" * 64
COHORT_DIGEST = "e" * 64
ARTIFACT_DIGEST = "1" * 64
METRICS_DIGEST = "2" * 64
CHANNEL_SPEC = {
    "history": ["static", "visible_dynamic", "unknown", "occlusion_boundary"],
    "state": ["static", "visible_dynamic", "unknown", "occlusion_boundary"],
    "trajectory": ["swept_footprint", "time_to_reach"],
}
ROLE_POLICY = {
    "layout_version": "production_risk_evaluation_metadata_v1",
    "training_split": "train",
    "selection_split": "val",
    "calibration_fit_split": "calibration",
    "evaluation_split": "test",
    "test_used_for_training_or_selection": False,
    "test_used_for_calibration": False,
    "calibration_statistics_scope": "calibration_only",
}


def _load_script(name: str, relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _family() -> SimpleNamespace:
    return SimpleNamespace(
        manifest={"dataset_family_layout_version": "risk_dataset_family_v1"},
        risk_dataset_family_digest=FAMILY_DIGEST,
        members={
            "calibration": {
                "risk_dataset_manifest_digest": CALIBRATION_MEMBER_DIGEST,
                "sample_ids_digest_sha256": CALIBRATION_SAMPLE_IDS_DIGEST,
                "sample_count": 2,
                "shard_count": 1,
            },
            "test": {
                "risk_dataset_manifest_digest": TEST_MEMBER_DIGEST,
                "sample_ids_digest_sha256": TEST_SAMPLE_IDS_DIGEST,
                "sample_count": 2,
                "shard_count": 1,
            },
        },
        production_evaluation_metadata=ROLE_POLICY,
    )


def _table(
    split: str,
    *,
    method_id: str = "risk-r0",
    checkpoint_layout_version: str = "risk_model_checkpoint_v2",
    checkpoint_digest: str = CHECKPOINT_DIGEST,
) -> dict[str, Any]:
    table = {
        "mode": "production",
        "schema_version": "3.0.0",
        "split": split,
        "method_id": method_id,
        "checkpoint_layout_version": checkpoint_layout_version,
        "checkpoint_digest": checkpoint_digest,
        "checkpoint_digest_kind": (
            "occupancy_checkpoint_semantic_sha256"
            if checkpoint_layout_version == "occupancy_baseline_checkpoint_v2"
            else "risk_checkpoint_semantic_sha256"
        ),
        "risk_dataset_family_layout_version": "risk_dataset_family_v1",
        "risk_dataset_family_digest": FAMILY_DIGEST,
        "risk_dataset_manifest_digest": (
            CALIBRATION_MEMBER_DIGEST
            if split == "calibration"
            else TEST_MEMBER_DIGEST
        ),
        "seed": 17,
        "channel_spec": CHANNEL_SPEC,
        "config_digest_sha256": "3" * 64,
        "cohort_digest_sha256": COHORT_DIGEST,
        "semantic_digest": TABLE_DIGEST,
        "rows": [{"sample_id": f"{split}-0", "q90": 0.4}],
    }
    if checkpoint_layout_version == "occupancy_baseline_checkpoint_v2":
        table["prediction_semantics"] = (
            "scalar_baseline_score_repeated_for_common_calibration"
        )
    return table


def _artifact(
    *,
    method_id: str = "risk-r0",
    checkpoint_layout_version: str = "risk_model_checkpoint_v2",
    checkpoint_digest: str = CHECKPOINT_DIGEST,
    family_digest: str = FAMILY_DIGEST,
) -> dict[str, Any]:
    artifact = {
        "mode": "production",
        "schema_version": "3.0.0",
        "fit_split": "calibration",
        "method_id": method_id,
        "checkpoint_layout_version": checkpoint_layout_version,
        "checkpoint_digest": checkpoint_digest,
        "checkpoint_digest_kind": (
            "occupancy_checkpoint_semantic_sha256"
            if checkpoint_layout_version == "occupancy_baseline_checkpoint_v2"
            else "risk_checkpoint_semantic_sha256"
        ),
        "risk_dataset_family_layout_version": "risk_dataset_family_v1",
        "risk_dataset_family_digest": family_digest,
        "risk_dataset_manifest_digest": CALIBRATION_MEMBER_DIGEST,
        "prediction_table_semantic_digest": TABLE_DIGEST,
        "calibration_cohort_digest_sha256": COHORT_DIGEST,
        "seed": 17,
        "channel_spec": CHANNEL_SPEC,
        "config_digest_sha256": "3" * 64,
        "prediction_key": "q90",
        "alpha": 0.1,
        "fitted_identities": {"sample_id": ["calibration-0"]},
        "global": {"correction": 0.1},
        "grouped": {"group_dimensions": []},
        "semantic_digest": ARTIFACT_DIGEST,
    }
    if checkpoint_layout_version == "occupancy_baseline_checkpoint_v2":
        artifact["prediction_semantics"] = (
            "scalar_baseline_score_repeated_for_common_calibration"
        )
    return artifact


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_production_clis_expose_dataset_family_root() -> None:
    for relative_path in (
        "scripts/07_calibrate_risk.py",
        "scripts/10_eval_offline.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(ROOT / relative_path), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--dataset-family-root" in completed.stdout


def test_production_calibration_publishes_family_bound_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("production_calibration_cli", "scripts/07_calibrate_risk.py")
    family = _family()
    table = _table("calibration")
    artifact = _artifact()
    table_path = tmp_path / "calibration-table.json"
    _write_json(table_path, table)
    output_dir = tmp_path / "calibration-output"
    family_root = tmp_path / "family"
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            mode="production",
            prediction_table=table_path,
            dataset_family_root=family_root,
            split="calibration",
            output_dir=output_dir,
            alpha=0.1,
            prediction_key="q90",
            min_group_size=1,
        ),
    )
    monkeypatch.setattr(
        module, "load_risk_dataset_family", lambda root: family, raising=False
    )

    def validate_table(
        value: dict[str, Any],
        *,
        expected_mode: str,
        expected_split: str,
        dataset_family: object,
    ) -> dict[str, Any]:
        calls["validate_table"] = dataset_family
        assert value == table
        assert (expected_mode, expected_split) == ("production", "calibration")
        return table

    def fit_artifact(
        value: dict[str, Any],
        *,
        alpha: float,
        prediction_key: str,
        dataset_family: object,
    ) -> dict[str, Any]:
        calls["fit_artifact"] = dataset_family
        assert value is table
        assert (alpha, prediction_key) == (0.1, "q90")
        return dict(artifact)

    def validate_artifact(
        value: dict[str, Any],
        *,
        expected_mode: str,
        dataset_family: object,
    ) -> dict[str, Any]:
        calls["validate_artifact"] = dataset_family
        assert expected_mode == "production"
        return value

    monkeypatch.setattr(module, "validate_prediction_table", validate_table)
    monkeypatch.setattr(module, "fit_calibration_artifact", fit_artifact)
    monkeypatch.setattr(module, "validate_calibration_artifact", validate_artifact)
    monkeypatch.setattr(module, "fit_grouped_calibration", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        module, "calibration_artifact_semantic_digest", lambda value: ARTIFACT_DIGEST
    )

    assert module.main() == 0
    assert calls == {
        "validate_table": family,
        "fit_artifact": family,
        "validate_artifact": family,
    }
    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_kind"] == "sop10_production_calibration"
    assert manifest["risk_dataset_family_layout_version"] == (
        "risk_dataset_family_v1"
    )
    assert manifest["risk_dataset_family_digest"] == FAMILY_DIGEST
    assert manifest["calibration_risk_dataset_manifest_digest"] == (
        CALIBRATION_MEMBER_DIGEST
    )
    assert manifest["calibration_sample_ids_digest_sha256"] == (
        CALIBRATION_SAMPLE_IDS_DIGEST
    )
    assert manifest["checkpoint_digest"] == CHECKPOINT_DIGEST
    assert manifest["prediction_table_semantic_digest"] == TABLE_DIGEST
    assert manifest["calibration_cohort_digest_sha256"] == COHORT_DIGEST
    assert manifest["calibration_semantic_digest"] == ARTIFACT_DIGEST
    assert manifest["production_evaluation_metadata"] == ROLE_POLICY
    assert manifest["calibration_count"] == 1


def test_production_calibration_rejects_protocol_alpha_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(
        "production_calibration_protocol_mismatch",
        "scripts/07_calibrate_risk.py",
    )
    protocol = build_prediction_protocol(
        alpha=0.2,
        prediction_key="q90",
        min_group_size=1,
    )
    protocol_path = tmp_path / "prediction-protocol.json"
    _write_json(protocol_path, protocol)
    table = _table("calibration")
    table["prediction_protocol_digest_sha256"] = protocol[
        "protocol_digest_sha256"
    ]
    table_path = tmp_path / "calibration-table.json"
    _write_json(table_path, table)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            mode="production",
            prediction_table=table_path,
            prediction_protocol=protocol_path,
            dataset_family_root=tmp_path / "family",
            split="calibration",
            output_dir=tmp_path / "output",
            alpha=0.1,
            prediction_key="q90",
            min_group_size=1,
        ),
    )
    monkeypatch.setattr(module, "load_risk_dataset_family", lambda root: _family())
    monkeypatch.setattr(
        module,
        "validate_prediction_table",
        lambda value, **kwargs: table,
    )

    with pytest.raises(ValueError, match="alpha"):
        module.main()


def _patch_eval_dependencies(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    *,
    family: object,
) -> None:
    monkeypatch.setattr(
        module, "load_risk_dataset_family", lambda root: family, raising=False
    )
    monkeypatch.setattr(
        module,
        "_calibrated_rows",
        lambda table, artifact: [
            {
                "sample_id": table["rows"][0]["sample_id"],
                "calibrated_upper": 0.5,
            }
        ],
    )
    monkeypatch.setattr(
        module,
        "_calibration_application_report",
        lambda table, artifact: {"primary_overall": "global_split_conformal"},
    )
    monkeypatch.setattr(
        module, "evaluate_risk_rows", lambda rows: {"sample_count": len(rows)}
    )
    monkeypatch.setattr(
        module,
        "calibration_artifact_semantic_digest",
        lambda value: METRICS_DIGEST,
    )


def test_offline_protocol_gate_rejects_shared_protocol_digest_drift() -> None:
    module = _load_script(
        "production_eval_protocol_gate",
        "scripts/10_eval_offline.py",
    )
    main = {
        "alpha": 0.1,
        "prediction_key": "q90",
        "fit_split": "calibration",
        "fitted_identities": {"sample_id": ["c0"]},
        "prediction_protocol_digest_sha256": "a" * 64,
    }
    baseline = {
        **main,
        "prediction_protocol_digest_sha256": "b" * 64,
    }

    with pytest.raises(ValueError, match="prediction_protocol_digest"):
        module._assert_same_calibration_protocol(main, baseline)


def test_production_eval_publishes_family_and_isolation_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("production_eval_cli", "scripts/10_eval_offline.py")
    family = _family()
    table = _table("test")
    artifact = _artifact()
    table_path = tmp_path / "test-table.json"
    artifact_path = tmp_path / "calibration.json"
    _write_json(table_path, table)
    _write_json(artifact_path, artifact)
    output_dir = tmp_path / "evaluation-output"
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            mode="production",
            task="risk",
            prediction_table=table_path,
            calibration_artifact=artifact_path,
            dataset_family_root=tmp_path / "family",
            baseline_prediction_table=None,
            baseline_calibration_artifact=None,
            split="test",
            output_dir=output_dir,
        ),
    )
    _patch_eval_dependencies(module, monkeypatch, family=family)

    def validate_table(
        value: dict[str, Any],
        *,
        expected_mode: str,
        expected_split: str,
        dataset_family: object,
    ) -> dict[str, Any]:
        calls["validate_table"] = dataset_family
        assert (expected_mode, expected_split) == ("production", "test")
        return table

    def validate_artifact(
        value: dict[str, Any],
        *,
        expected_mode: str,
        expected_provenance: dict[str, Any],
        dataset_family: object,
    ) -> dict[str, Any]:
        calls["validate_artifact"] = dataset_family
        assert expected_mode == "production"
        assert expected_provenance["checkpoint_digest"] == CHECKPOINT_DIGEST
        assert expected_provenance["method_id"] == "risk-r0"
        return artifact

    def assert_isolation(
        value: dict[str, Any], rows: list[dict[str, Any]], *, dataset_family: object
    ) -> dict[str, int]:
        calls["assert_isolation"] = dataset_family
        assert value is artifact
        assert rows is table["rows"]
        return {"sample_id": 0}

    monkeypatch.setattr(module, "validate_prediction_table", validate_table)
    monkeypatch.setattr(module, "validate_calibration_artifact", validate_artifact)
    monkeypatch.setattr(
        module, "assert_calibration_artifact_test_isolation", assert_isolation
    )

    assert module.main() == 0
    assert calls == {
        "validate_table": family,
        "validate_artifact": family,
        "assert_isolation": family,
    }
    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_kind"] == "sop10_production_offline_evaluation"
    assert manifest["risk_dataset_family_layout_version"] == (
        "risk_dataset_family_v1"
    )
    assert manifest["risk_dataset_family_digest"] == FAMILY_DIGEST
    assert manifest["calibration_risk_dataset_manifest_digest"] == (
        CALIBRATION_MEMBER_DIGEST
    )
    assert manifest["test_risk_dataset_manifest_digest"] == TEST_MEMBER_DIGEST
    assert manifest["calibration_sample_ids_digest_sha256"] == (
        CALIBRATION_SAMPLE_IDS_DIGEST
    )
    assert manifest["test_sample_ids_digest_sha256"] == TEST_SAMPLE_IDS_DIGEST
    assert manifest["checkpoint_digest"] == CHECKPOINT_DIGEST
    assert manifest["prediction_table_semantic_digest"] == TABLE_DIGEST
    assert manifest["calibration_semantic_digest"] == ARTIFACT_DIGEST
    assert manifest["test_cohort_digest_sha256"] == COHORT_DIGEST
    assert manifest["metrics_semantic_digest"] == METRICS_DIGEST
    assert manifest["calibration_test_identity_overlap"] == {"sample_id": 0}
    assert manifest["production_evaluation_metadata"] == ROLE_POLICY
    assert manifest["sample_count"] == 1


def test_production_baseline_rejects_different_dataset_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script(
        "production_eval_baseline_cli", "scripts/10_eval_offline.py"
    )
    family = _family()
    main_table = _table("test")
    baseline_table = _table(
        "test",
        method_id="occupancy-b3",
        checkpoint_layout_version="occupancy_baseline_checkpoint_v2",
        checkpoint_digest="4" * 64,
    )
    main_artifact = _artifact()
    baseline_artifact = _artifact(
        method_id="occupancy-b3",
        checkpoint_layout_version="occupancy_baseline_checkpoint_v2",
        checkpoint_digest="4" * 64,
        family_digest="9" * 64,
    )
    paths = {
        "main_table": tmp_path / "main-table.json",
        "baseline_table": tmp_path / "baseline-table.json",
        "main_artifact": tmp_path / "main-artifact.json",
        "baseline_artifact": tmp_path / "baseline-artifact.json",
    }
    for name, value in (
        ("main_table", main_table),
        ("baseline_table", baseline_table),
        ("main_artifact", main_artifact),
        ("baseline_artifact", baseline_artifact),
    ):
        _write_json(paths[name], value)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            mode="production",
            task="risk",
            prediction_table=paths["main_table"],
            calibration_artifact=paths["main_artifact"],
            dataset_family_root=tmp_path / "family",
            baseline_prediction_table=paths["baseline_table"],
            baseline_calibration_artifact=paths["baseline_artifact"],
            split="test",
            output_dir=tmp_path / "evaluation-output",
        ),
    )
    _patch_eval_dependencies(module, monkeypatch, family=family)
    monkeypatch.setattr(
        module,
        "validate_prediction_table",
        lambda value, **kwargs: value,
    )
    monkeypatch.setattr(
        module,
        "validate_calibration_artifact",
        lambda value, **kwargs: value,
    )
    monkeypatch.setattr(
        module,
        "assert_calibration_artifact_test_isolation",
        lambda artifact, rows, **kwargs: {"sample_id": 0},
    )

    with pytest.raises(ValueError, match="risk_dataset_family_digest"):
        module.main()
