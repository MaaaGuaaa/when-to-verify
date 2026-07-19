from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from src.calibration.split_conformal import validate_prediction_table
from src.contracts import HISTORY_CHANNELS
from src.evaluation.risk_baselines import (
    fit_toy_learned_aggregator,
    fit_toy_occupancy_model,
)
from src.models.occupancy_baseline import (
    ConvGRUOccupancyPredictor,
    LearnedOccupancyRiskAggregator,
)


def _stationary_toy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = 8
    history = torch.zeros(batch_size, 8, len(HISTORY_CHANNELS), 6, 6, dtype=torch.float32)
    target = torch.zeros(batch_size, 15, 6, 6, dtype=torch.float32)
    dynamic = HISTORY_CHANNELS.index("past_dynamic_occupancy")
    visible = HISTORY_CHANNELS.index("past_visible_mask")
    for row in range(batch_size):
        y = 1 + (row % 4)
        x = 1 + ((row // 2) % 4)
        history[row, :, dynamic, y, x] = 1.0
        history[row, :, visible] = 1.0
        target[row, :, y, x] = 1.0
    return history, target


def test_convgru_materially_reduces_toy_occupancy_loss() -> None:
    torch.manual_seed(17)
    history, target = _stationary_toy_batch()
    model = ConvGRUOccupancyPredictor(hidden_channels=4, future_steps=15)

    report = fit_toy_occupancy_model(
        model,
        history,
        target,
        steps=50,
        learning_rate=0.02,
    )

    assert report["final_loss"] < report["initial_loss"] * 0.75
    assert report["steps"] == 50
    assert all(torch.isfinite(torch.tensor(report["loss_history"])))
    with torch.no_grad():
        predicted_positive = model(history) >= 0.5
    truth = target > 0.5
    true_positive = torch.logical_and(predicted_positive, truth).sum()
    positive_recall = true_positive / truth.sum()
    union = torch.logical_or(predicted_positive, truth).sum()
    intersection_over_union = true_positive / union
    assert float(positive_recall) >= 0.9
    assert float(intersection_over_union) >= 0.75


def test_b4_learned_aggregator_overfits_toy_collision_evidence() -> None:
    torch.manual_seed(23)
    occupancy = torch.full((12, 15, 4, 4), 0.01, dtype=torch.float32)
    footprint = torch.zeros_like(occupancy)
    footprint[:, :, 1, 1] = 1.0
    labels = torch.zeros(12, dtype=torch.float32)
    occupancy[6:, :, 1, 1] = 0.95
    labels[6:] = 1.0
    aggregator = LearnedOccupancyRiskAggregator(future_steps=15, hidden_dim=8)

    report = fit_toy_learned_aggregator(
        aggregator,
        occupancy,
        footprint,
        labels,
        steps=80,
        learning_rate=0.02,
    )

    assert report["final_loss"] < report["initial_loss"] * 0.5
    with torch.no_grad():
        prediction = aggregator(occupancy, footprint)
    assert torch.all(prediction[6:] > prediction[:6].max())


def _run_training_cli(
    output_dir: Path,
    *,
    mode: str,
    config_path: Path | None = None,
    code_commit: str = "commit-a",
) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts/05_train_occupancy_baseline.py"),
            "--config",
            str(config_path or root / "configs/occupancy_baseline.yaml"),
            "--mode",
            mode,
            "--output-dir",
            str(output_dir),
            "--toy-count",
            "14",
            "--validation-count",
            "7",
            "--grid-size",
            "8",
            "--training-steps",
            "10",
            "--aggregator-steps",
            "20",
            "--seed",
            "0",
            "--b2-tau-s",
            "1.5",
            "--b2-a-max-s",
            "4.0",
            "--convgru-kernel-size",
            "5",
            "--code-commit",
            code_commit,
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )


def _mutated_config(tmp_path: Path, name: str, mutate) -> Path:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/occupancy_baseline.yaml").read_text(encoding="utf-8")
    )
    mutate(config)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_training_cli_rejects_production_until_v2_sidecars_exist(tmp_path) -> None:
    output_dir = tmp_path / "production"

    result = _run_training_cli(output_dir, mode="production")

    assert result.returncode == 2
    assert "dataset-level v2 manifest" in result.stderr
    assert not output_dir.exists()


def test_training_cli_rejects_nontrain_config_split(tmp_path: Path) -> None:
    config_path = _mutated_config(
        tmp_path,
        "bad-split.yaml",
        lambda config: config["data"].__setitem__("split", "val"),
    )
    output_dir = tmp_path / "bad-split-output"

    result = _run_training_cli(output_dir, mode="toy", config_path=config_path)

    assert result.returncode == 2
    assert "data.split must be 'train'" in result.stderr
    assert not output_dir.exists()


def test_training_cli_requires_new_exact_config_fields(tmp_path: Path) -> None:
    missing_validation = _mutated_config(
        tmp_path,
        "missing-validation.yaml",
        lambda config: config["data"].pop("validation_count", None),
    )
    unknown_model_key = _mutated_config(
        tmp_path,
        "unknown-model-key.yaml",
        lambda config: config["model"].__setitem__("unfrozen_kernel", 3),
    )

    missing = _run_training_cli(
        tmp_path / "missing-validation-output",
        mode="toy",
        config_path=missing_validation,
    )
    unknown = _run_training_cli(
        tmp_path / "unknown-model-output",
        mode="toy",
        config_path=unknown_model_key,
    )

    assert missing.returncode == 2
    assert "missing keys" in missing.stderr
    assert "validation_count" in missing.stderr
    assert unknown.returncode == 2
    assert "unknown keys" in unknown.stderr
    assert "unfrozen_kernel" in unknown.stderr


def test_training_cli_writes_deterministic_toy_artifact_without_oracle_inputs(
    tmp_path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    other_commit = tmp_path / "other-commit"

    first = _run_training_cli(left, mode="toy", code_commit="commit-a")
    second = _run_training_cli(right, mode="toy", code_commit="commit-a")
    changed_commit = _run_training_cli(
        other_commit,
        mode="toy",
        code_commit="commit-b",
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert changed_commit.returncode == 0, changed_commit.stderr
    required = {
        "checkpoint.pt",
        "checksums.sha256",
        "config_snapshot.json",
        "manifest.json",
        "metrics.json",
        "occupancy_predictions.npz",
    }
    required.update(
        f"prediction_table_{split}_{method}.json"
        for split in ("val", "calibration", "test")
        for method in ("B1", "B2", "B3", "B4")
    )
    assert {path.name for path in left.iterdir()} == required
    left_manifest = json.loads((left / "manifest.json").read_text(encoding="utf-8"))
    right_manifest = json.loads((right / "manifest.json").read_text(encoding="utf-8"))
    other_manifest = json.loads(
        (other_commit / "manifest.json").read_text(encoding="utf-8")
    )
    config_snapshot = json.loads(
        (left / "config_snapshot.json").read_text(encoding="utf-8")
    )
    metrics = json.loads((left / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(left / "checkpoint.pt", map_location="cpu")
    assert left_manifest["semantic_digest_sha256"] == right_manifest["semantic_digest_sha256"]
    assert left_manifest["semantic_digest_sha256"] != other_manifest[
        "semantic_digest_sha256"
    ]
    assert left_manifest["mode"] == "toy"
    assert left_manifest["seed"] == 0
    assert left_manifest["code_commit"] == "commit-a"
    assert left_manifest["scientific_gate"] == "not_evaluated_real_data"
    assert left_manifest["evaluation_scope"] == "train_split_fit_diagnostics_only"
    assert metrics["evaluation_scope"] == "train_split_fit_diagnostics_only"
    assert metrics["evaluated_split"] == "train"
    assert set(left_manifest["baselines"]) == {"B1", "B2", "B3", "B4"}
    assert set(left_manifest["split_dataset_manifest_digests"]) == {
        "train",
        "val",
        "calibration",
        "test",
    }
    assert left_manifest["split_isolation"]["passed"] is True
    assert all(
        count == 0
        for count in left_manifest["split_isolation"]["overlap_counts"].values()
    )
    assert set(left_manifest["split_collation_provenance"]) == {
        "train",
        "val",
        "calibration",
        "test",
    }
    for provenance in left_manifest["split_collation_provenance"].values():
        assert provenance["sidecar_sample_ids_digest_sha256"] == provenance[
            "ordered_sample_ids_digest_sha256"
        ]
        assert len(provenance["ordered_sample_digest_sha256"]) == 64
    assert config_snapshot["data"]["validation_count"] == 7
    assert config_snapshot["model"]["convgru_kernel_size"] == 5
    assert config_snapshot["aggregation"]["b2_tau_s"] == pytest.approx(1.5)
    assert config_snapshot["aggregation"]["b2_a_max_s"] == pytest.approx(4.0)
    assert left_manifest["baseline_hyperparameters"] == {
        "b2_tau_s": 1.5,
        "b2_a_max_s": 4.0,
        "convgru_kernel_size": 5,
    }
    assert "hidden_risk_occupancy" not in left_manifest["model_input_keys"]
    assert "robot_future_footprints" not in left_manifest["model_input_keys"]
    assert left_manifest["future_endpoint_times_s"] == pytest.approx(
        [step * 0.2 for step in range(1, 16)]
    )
    checkpoint_digest = left_manifest["checkpoint_semantic_digest_sha256"]
    assert checkpoint["checkpoint_semantic_digest_sha256"] == checkpoint_digest
    assert checkpoint["config_digest"] == left_manifest["config_digest_sha256"]
    expected_counts = {"val": 7, "calibration": 28, "test": 28}
    for split in ("val", "calibration", "test"):
        for method in ("B1", "B2", "B3", "B4"):
            name = f"prediction_table_{split}_{method}.json"
            table = json.loads((left / name).read_text(encoding="utf-8"))
            validated = validate_prediction_table(
                table,
                expected_mode="toy",
                expected_split=split,
            )
            assert validated["method_id"] == method
            assert len(validated["rows"]) == expected_counts[split]
            assert validated["checkpoint_layout_version"] == (
                "occupancy_baseline_checkpoint_v2"
            )
            assert validated["checkpoint_digest"] == checkpoint_digest
            assert validated["checkpoint_digest_kind"] == (
                "occupancy_checkpoint_semantic_sha256"
            )
            assert validated["prediction_semantics"] == (
                "scalar_baseline_score_repeated_for_common_calibration"
            )
            assert validated["toy_dataset_manifest_digest"] == (
                left_manifest["prediction_tables"][split][method][
                    "toy_dataset_manifest_digest"
                ]
            )
