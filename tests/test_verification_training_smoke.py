from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.datasets.verification_dataloader import write_verification_shard
from src.datasets.verification_dataset import build_verification_samples
from src.evaluation.verification_metrics import (
    build_verification_checkpoint_manifest,
)
from src.models.verification_model import load_verify_model_config
from src.models.verification_training import (
    load_verification_training_checkpoint,
    train_verification_samples,
    write_verification_training_checkpoint,
)
from tests.test_verification_dataset import _source_and_library


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/09_train_verification_model.py"


def _controlled_samples():
    grid, library, source = _source_and_library()
    original = build_verification_samples(source, library=library, grid=grid)
    targets = (0.60, 0.40, 0.20, -0.20, -0.40, -0.60)
    samples = tuple(
        replace(
            sample,
            value_target=target,
            useful_target=int(target > 0.0),
            br_before=1.0,
            post_risk=1.0 - target,
        )
        for sample, target in zip(original, targets, strict=True)
    )
    return grid, library, samples


def _fast_config():
    config = load_verify_model_config(ROOT / "configs/verify_model.yaml")
    return replace(
        config,
        training=replace(
            config.training,
            seed=19,
            epochs=100,
            batch_size=6,
            learning_rate=0.01,
        ),
    )


def test_tiny_complete_group_overfits_deterministically_on_cpu():
    grid, _, samples = _controlled_samples()
    config = _fast_config()

    first = train_verification_samples(samples, grid=grid, config=config)
    second = train_verification_samples(samples, grid=grid, config=config)

    assert first.device == "cpu"
    assert first.completed_epochs == 100
    assert np.isfinite(first.loss_history).all()
    assert first.final_loss < first.initial_loss * 0.20
    assert first.metrics["pairwise_accuracy"] == pytest.approx(1.0)
    assert first.metrics["top1_regret_mean"] == pytest.approx(0.0)
    assert first.metrics["useful_f1"] == pytest.approx(1.0)
    np.testing.assert_array_equal(first.value_prediction, second.value_prediction)
    np.testing.assert_array_equal(
        first.useful_probability, second.useful_probability
    )
    assert first.loss_history == second.loss_history


def test_checkpoint_is_immutable_and_resume_validation_rejects_context_mismatch(
    tmp_path,
):
    grid, _, samples = _controlled_samples()
    config = replace(
        _fast_config(), training=replace(_fast_config().training, epochs=2)
    )
    result = train_verification_samples(samples, grid=grid, config=config)
    manifest = build_verification_checkpoint_manifest(
        input_manifest_digest="a" * 64,
        split_digests={"train": "b" * 64},
        model_config=asdict(config),
        seed=config.training.seed,
        code_version="c" * 40,
    )
    checkpoint = tmp_path / "checkpoint.pt"

    write_verification_training_checkpoint(
        checkpoint, result=result, manifest=manifest
    )
    loaded = load_verification_training_checkpoint(
        checkpoint,
        expected_input_manifest_digest="a" * 64,
        expected_split_digests={"train": "b" * 64},
        expected_model_config=asdict(config),
        expected_seed=config.training.seed,
        expected_code_version="c" * 40,
    )

    assert loaded.completed_epochs == 2
    assert loaded.loss_history == result.loss_history
    assert set(loaded.model_state_dict) == set(result.model.state_dict())
    with pytest.raises(FileExistsError, match="overwrite"):
        write_verification_training_checkpoint(
            checkpoint, result=result, manifest=manifest
        )
    with pytest.raises(ValueError, match="code version"):
        load_verification_training_checkpoint(
            checkpoint,
            expected_input_manifest_digest="a" * 64,
            expected_split_digests={"train": "b" * 64},
            expected_model_config=asdict(config),
            expected_seed=config.training.seed,
            expected_code_version="d" * 40,
        )


def _module():
    spec = importlib.util.spec_from_file_location("train_verification_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_training_cli_writes_atomic_smoke_only_handoff_and_refuses_overwrite(
    tmp_path,
):
    grid, library, samples = _controlled_samples()
    shard = tmp_path / "input" / "shard-00000"
    write_verification_shard(
        samples,
        shard,
        grid=grid,
        library=library,
        expected_sample_count=6,
    )
    summary = json.loads((shard / "summary.json").read_text())
    handoff = {
        "schema_version": "3.0.0",
        "handoff_version": "verification_collection_handoff_v1",
        "collection_state": "complete",
        "scientific_status": "toy_smoke_only",
        "split": "train",
        "sample_count": 6,
        "group_count": 1,
        "collection_semantic_digest": "e" * 64,
        "generation_report_sha256": "f" * 64,
        "shards": [
            {
                "shard_index": 0,
                "relative_root": "shard-00000",
                "sample_count": 6,
                "semantic_digest": summary["semantic_digest"],
            }
        ],
        "limitations": ["toy data are not paper-scale evidence"],
    }
    handoff_path = tmp_path / "input" / "collection_complete_handoff.json"
    handoff_path.write_text(json.dumps(handoff, sort_keys=True) + "\n")
    base = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    base["bev"].update({"range_m": 8.0, "size": 80})
    base_path = tmp_path / "toy-base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))
    model = yaml.safe_load((ROOT / "configs/verify_model.yaml").read_text())
    model["training"].update({"epochs": 2, "batch_size": 6})
    model_path = tmp_path / "fast-model.yaml"
    model_path.write_text(yaml.safe_dump(model, sort_keys=False))
    output = tmp_path / "trained"
    args = [
        "--shard-dir",
        str(shard),
        "--collection-handoff",
        str(handoff_path),
        "--output-dir",
        str(output),
        "--base-config",
        str(base_path),
        "--actions-config",
        str(ROOT / "configs/verification_actions.yaml"),
        "--model-config",
        str(model_path),
        "--code-version",
        "1" * 40,
    ]
    module = _module()

    assert module.main(args) == 0
    assert {path.name for path in output.iterdir()} == {
        "checkpoint.pt",
        "manifest.json",
        "metrics.json",
        "training_report.json",
    }
    metrics = json.loads((output / "metrics.json").read_text())
    report = json.loads((output / "training_report.json").read_text())
    assert metrics["scientific_status"] == "toy_smoke_only"
    assert metrics["paper_thresholds_evaluated"] is False
    assert report["collection_semantic_digest"] == "e" * 64
    assert report["sample_count"] == 6
    with pytest.raises(FileExistsError, match="overwrite"):
        module.main(args)
