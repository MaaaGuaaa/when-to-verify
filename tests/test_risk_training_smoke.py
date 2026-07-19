"""SOP09 deterministic toy overfit, reload, and CLI smoke tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from src.calibration.split_conformal import validate_prediction_table
from src.contracts import HISTORY_CHANNELS, INPUT_CHANNELS, SCHEMA_VERSION, STATE_CHANNELS, TRAJECTORY_CHANNELS
from src.datasets.risk_dataloader import collate_risk_samples
from src.datasets.toy_risk_learning import (
    assert_toy_split_isolation,
    make_toy_risk_dataset,
)
from src.models.risk_model import (
    load_risk_checkpoint,
    save_risk_checkpoint,
    train_toy_risk_model,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts/06_train_risk_model.py"
CONFIG_PATH = ROOT / "configs/risk_model.yaml"


def _channel_spec() -> dict[str, list[str]]:
    return {
        "history": list(HISTORY_CHANNELS),
        "state": list(STATE_CHANNELS),
        "trajectory": list(TRAJECTORY_CHANNELS),
        "flat": list(INPUT_CHANNELS),
    }


@pytest.mark.parametrize("variant", ["r0", "r1"])
def test_r0_and_r1_materially_reduce_loss_on_128_toy_samples(variant):
    train = make_toy_risk_dataset(split="train", count=128, seed=101, grid_size=12)
    validation = make_toy_risk_dataset(split="val", count=28, seed=101, grid_size=12)
    assert_toy_split_isolation((train, validation))

    model, metrics = train_toy_risk_model(
        variant=variant,
        train_dataset=train,
        validation_dataset=validation,
        hidden_channels=8,
        optimization_steps=40,
        learning_rate=0.02,
        seed=101,
    )

    assert metrics["train_sample_count"] == 128
    assert metrics["validation_sample_count"] == 28
    assert metrics["training_split"] == "train"
    assert metrics["selection_split"] == "val"
    assert metrics["test_samples_used_for_training_or_selection"] == 0
    assert "test_samples_seen" not in metrics
    assert metrics["final_train_loss"] < 0.75 * metrics["initial_train_loss"]
    assert metrics["loss_history"][0] > metrics["loss_history"][-1]
    assert metrics["best_validation_loss"] == min(metrics["validation_loss_history"])
    assert metrics["validation_loss"] == metrics["best_validation_loss"]
    assert 0 <= metrics["best_validation_step"] <= metrics["optimization_steps"]
    assert metrics["quantile_crossing_rate"] == 0.0
    sensitivity = metrics["trajectory_ablation_sensitivity"]
    assert sensitivity["split"] == "val"
    assert sensitivity["sample_count"] == 28
    assert sensitivity["used_for_training_or_selection"] is False
    assert sensitivity["labels_accessed"] is False
    assert sensitivity["protocol"] == (
        "deterministic_permutation_of_validated_query"
    )
    assert sensitivity["query_components_permuted"] == [
        "trajectory_channels",
        "robot_state",
    ]
    assert sensitivity["source_dataset_manifest_digest"] == (
        validation.manifest_digest
    )
    assert sensitivity["source_manifest_rows_digest_sha256"] == (
        validation.manifest["manifest_rows_digest_sha256"]
    )
    assert sorted(sensitivity["permutation_source_indices"]) == list(range(28))
    assert all(
        source_index != target_index
        for target_index, source_index in enumerate(
            sensitivity["permutation_source_indices"]
        )
    )
    assert sensitivity["changed_query_count"] > 0
    assert sensitivity["materiality_threshold"] == 1e-8
    assert sensitivity["materially_sensitive"] is True
    assert sensitivity["combined_mean_absolute_delta"] > sensitivity[
        "materiality_threshold"
    ]
    assert sensitivity["interpretation"] == (
        "conditioning_effect_only;does_not_establish_real_world_directional_superiority"
    )
    assert all(
        isinstance(sensitivity[field], float)
        and torch.isfinite(torch.tensor(sensitivity[field])).item()
        for field in (
            "quantile_mean_absolute_delta",
            "collision_logit_mean_absolute_delta",
            "collision_probability_mean_absolute_delta",
            "combined_mean_absolute_delta",
        )
    )
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())


def test_toy_training_is_deterministic_and_reload_is_identical(tmp_path):
    train = make_toy_risk_dataset(split="train", count=28, seed=103, grid_size=10)
    validation = make_toy_risk_dataset(split="val", count=14, seed=103, grid_size=10)
    arguments = dict(
        variant="r0",
        train_dataset=train,
        validation_dataset=validation,
        hidden_channels=6,
        optimization_steps=15,
        learning_rate=0.015,
        seed=103,
    )
    first, first_metrics = train_toy_risk_model(**arguments)
    second, second_metrics = train_toy_risk_model(**arguments)
    assert first_metrics == second_metrics
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "channel_spec": _channel_spec(),
        "model_variant": "r0",
        "config_digest": "c" * 32,
        "toy_dataset_manifest_digest": train.manifest_digest,
        "validation_dataset_manifest_digest": validation.manifest_digest,
        "seed": 103,
    }
    path = tmp_path / "trained-r0.pt"
    save_risk_checkpoint(path, model=first, mode="toy", provenance=provenance)
    loaded, _ = load_risk_checkpoint(
        path, expected_mode="toy", expected_provenance=provenance
    )
    batch = collate_risk_samples(
        validation.samples,
        grid=validation.grid,
        dataset_manifest=validation.manifest,
        expected_split="val",
    )
    first.eval()
    loaded.eval()
    with torch.no_grad():
        expected = first(batch.model_inputs)
        actual = loaded(batch.model_inputs)
    assert torch.equal(expected["quantiles"], actual["quantiles"])
    assert torch.equal(expected["collision_logits"], actual["collision_logits"])


def test_training_cli_writes_auditable_r0_r1_toy_artifacts_without_test_selection(
    tmp_path,
):
    output = tmp_path / "sop09-toy-run"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(CONFIG_PATH),
        "--output-dir",
        str(output),
        "--optimization-steps",
        "3",
        "--train-count",
        "14",
        "--validation-count",
        "7",
        "--calibration-count",
        "7",
        "--test-count",
        "7",
        "--grid-size",
        "10",
        "--hidden-channels",
        "4",
        "--code-commit",
        "test-commit",
    ]
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    expected_files = {
        "config_snapshot.json",
        "r0_checkpoint.pt",
        "r1_checkpoint.pt",
        "metrics.json",
        "manifest.json",
        "r0_calibration_prediction_table.json",
        "r0_test_prediction_table.json",
        "r1_calibration_prediction_table.json",
        "r1_test_prediction_table.json",
        "checksums.json",
    }
    assert {path.name for path in output.iterdir()} == expected_files
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    checksums = json.loads((output / "checksums.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "toy"
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["selection_split"] == "val"
    assert manifest["test_samples_used_for_training_or_selection"] == 0
    assert manifest["test_prediction_rows_generated"] == 7
    assert "test_samples_seen" not in manifest
    assert manifest["real_data_status"] == "not_evaluated_real_data"
    assert set(metrics["variants"]) == {"r0", "r1"}
    assert metrics["comparison"]["selection_metric"] == "validation_loss"
    assert metrics["comparison"]["scope"] == "toy_software_validation_only"
    assert metrics["comparison"]["lower_validation_loss_variant"] in {"r0", "r1"}
    assert metrics["test_samples_used_for_training_or_selection"] == 0
    assert metrics["test_prediction_rows_generated"] == 7
    assert "test_samples_seen" not in metrics
    assert set(manifest["trajectory_ablation_sensitivity"]) == {"r0", "r1"}
    assert manifest["trajectory_ablation_sensitivity"] == {
        variant: metrics["variants"][variant]["trajectory_ablation_sensitivity"]
        for variant in ("r0", "r1")
    }
    for diagnostic in manifest["trajectory_ablation_sensitivity"].values():
        assert diagnostic["split"] == "val"
        assert diagnostic["sample_count"] == 7
        assert diagnostic["used_for_training_or_selection"] is False
        assert diagnostic["labels_accessed"] is False
        assert diagnostic["changed_query_count"] > 0
        assert diagnostic["source_dataset_manifest_digest"] == manifest[
            "validation_dataset_manifest_digest"
        ]
        assert diagnostic["source_rows_strictly_validated"] is True
        assert diagnostic["materiality_threshold"] == 1e-8
        assert diagnostic["materially_sensitive"] is True
        assert diagnostic["combined_mean_absolute_delta"] > diagnostic[
            "materiality_threshold"
        ]
    assert set(checksums["sha256"]) == expected_files - {"checksums.json"}
    assert len(manifest["semantic_digest_sha256"]) == 64
    for variant in ("r0", "r1"):
        for split in ("calibration", "test"):
            table = json.loads(
                (output / f"{variant}_{split}_prediction_table.json").read_text(
                    encoding="utf-8"
                )
            )
            validated = validate_prediction_table(
                table, expected_mode="toy", expected_split=split
            )
            assert validated["method_id"] == variant
            assert validated["checkpoint_layout_version"] == "risk_model_checkpoint_v2"
            assert (
                validated["checkpoint_digest_kind"]
                == "risk_checkpoint_semantic_sha256"
            )
            _, checkpoint_payload = load_risk_checkpoint(
                output / f"{variant}_checkpoint.pt",
                expected_mode="toy",
            )
            assert checkpoint_payload["provenance"][
                "toy_dataset_manifest_digest"
            ] == manifest["train_dataset_manifest_digest"]
            assert checkpoint_payload["provenance"][
                "validation_dataset_manifest_digest"
            ] == manifest["validation_dataset_manifest_digest"]
            assert validated["checkpoint_digest"] == (
                checkpoint_payload["checkpoint_semantic_digest_sha256"]
            )
            assert len(validated["rows"]) == 7
            assert all(row["split"] == split for row in validated["rows"])
    assert manifest["calibration_sample_count"] == 7
    assert manifest["test_sample_count"] == 7

    repeated = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr


def test_cli_model_selection_is_invariant_to_test_count(tmp_path):
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})

    def run(test_count: int):
        output = tmp_path / f"test-count-{test_count}"
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(output),
            "--optimization-steps",
            "3",
            "--train-count",
            "14",
            "--validation-count",
            "7",
            "--calibration-count",
            "7",
            "--test-count",
            str(test_count),
            "--grid-size",
            "10",
            "--hidden-channels",
            "4",
            "--code-commit",
            "test-count-isolation",
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        checkpoints = {
            variant: load_risk_checkpoint(
                output / f"{variant}_checkpoint.pt", expected_mode="toy"
            )[1]
            for variant in ("r0", "r1")
        }
        return manifest, metrics, checkpoints

    small_manifest, small_metrics, small_checkpoints = run(7)
    large_manifest, large_metrics, large_checkpoints = run(14)

    assert small_manifest["config_digest"] != large_manifest["config_digest"]
    assert small_manifest["semantic_digest_sha256"] != large_manifest[
        "semantic_digest_sha256"
    ]
    assert small_metrics["comparison"]["lower_validation_loss_variant"] == (
        large_metrics["comparison"]["lower_validation_loss_variant"]
    )
    for variant in ("r0", "r1"):
        assert small_checkpoints[variant]["model_state_digest_sha256"] == (
            large_checkpoints[variant]["model_state_digest_sha256"]
        )
        assert small_metrics["variants"][variant]["best_validation_step"] == (
            large_metrics["variants"][variant]["best_validation_step"]
        )
        assert small_metrics["variants"][variant]["best_validation_loss"] == (
            large_metrics["variants"][variant]["best_validation_loss"]
        )
        assert small_metrics["variants"][variant]["validation_loss"] == (
            large_metrics["variants"][variant]["validation_loss"]
        )
    assert small_manifest["test_samples_used_for_training_or_selection"] == 0
    assert large_manifest["test_samples_used_for_training_or_selection"] == 0
