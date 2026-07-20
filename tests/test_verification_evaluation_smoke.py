from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.datasets.verification_dataloader import write_verification_shard
from src.datasets.verification_dataset import build_verification_samples
from src.evaluation.verification_metrics import build_verification_checkpoint_manifest
from src.models.verification_training import (
    evaluate_verification_samples,
    load_verification_training_checkpoint,
    train_verification_samples,
    write_verification_training_checkpoint,
)
from tests.test_verification_dataset import _source_and_library
from tests.test_verification_training_smoke import _controlled_samples, _fast_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/10_evaluate_verification_model.py"
CODE_VERSION = "c" * 40


def _heldout_samples(split="val"):
    grid, library, source = _source_and_library()
    source = replace(source, split=split)
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


def _checkpoint(tmp_path):
    grid, _, train_samples = _controlled_samples()
    config = replace(
        _fast_config(), training=replace(_fast_config().training, epochs=2)
    )
    training = train_verification_samples(train_samples, grid=grid, config=config)
    manifest = build_verification_checkpoint_manifest(
        input_manifest_digest="a" * 64,
        split_digests={"train": "b" * 64},
        model_config=asdict(config),
        seed=config.training.seed,
        code_version=CODE_VERSION,
    )
    path = tmp_path / "checkpoint.pt"
    write_verification_training_checkpoint(path, result=training, manifest=manifest)
    loaded = load_verification_training_checkpoint(
        path,
        expected_input_manifest_digest="a" * 64,
        expected_split_digests={"train": "b" * 64},
        expected_model_config=asdict(config),
        expected_seed=config.training.seed,
        expected_code_version=CODE_VERSION,
    )
    return path, manifest, loaded, config


def test_loaded_checkpoint_evaluation_is_deterministic_and_heldout_only(tmp_path):
    _, _, checkpoint, config = _checkpoint(tmp_path)
    grid, _, samples = _heldout_samples("val")

    first = evaluate_verification_samples(
        samples, grid=grid, config=config, checkpoint=checkpoint, split="val"
    )
    second = evaluate_verification_samples(
        samples, grid=grid, config=config, checkpoint=checkpoint, split="val"
    )

    assert first.split == "val"
    assert first.sample_count == 6
    assert first.group_count == 1
    assert first.device == "cpu"
    assert np.isfinite(list(first.losses.values())).all()
    np.testing.assert_array_equal(first.value_prediction, second.value_prediction)
    np.testing.assert_array_equal(
        first.useful_probability, second.useful_probability
    )
    assert first.metrics == second.metrics

    train_grid, _, train_samples = _controlled_samples()
    with pytest.raises(ValueError, match="held-out"):
        evaluate_verification_samples(
            train_samples,
            grid=train_grid,
            config=config,
            checkpoint=checkpoint,
            split="train",
        )


def _module():
    spec = importlib.util.spec_from_file_location("evaluate_verification_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_evaluation_cli_writes_metrics_without_refitting_or_new_checkpoint(tmp_path):
    checkpoint_path, manifest, _, config = _checkpoint(tmp_path / "trained")
    grid, library, samples = _heldout_samples("test")
    input_root = tmp_path / "input"
    shard = input_root / "shard-00000"
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
        "scientific_status": "test_smoke_only",
        "split": "test",
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
        "limitations": ["held-out smoke data are not paper-scale evidence"],
    }
    handoff_path = input_root / "collection_complete_handoff.json"
    handoff_path.write_text(json.dumps(handoff, sort_keys=True) + "\n")
    manifest_path = tmp_path / "trained" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    base = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    base["bev"].update({"range_m": 8.0, "size": 80})
    base_path = tmp_path / "toy-base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))
    model_path = tmp_path / "model.yaml"
    model_payload = json.loads(json.dumps(asdict(config)))
    model_path.write_text(yaml.safe_dump(model_payload, sort_keys=False))
    output = tmp_path / "evaluated"
    args = [
        "--split",
        "test",
        "--shard-dir",
        str(shard),
        "--collection-handoff",
        str(handoff_path),
        "--checkpoint",
        str(checkpoint_path),
        "--checkpoint-manifest",
        str(manifest_path),
        "--output-dir",
        str(output),
        "--base-config",
        str(base_path),
        "--actions-config",
        str(ROOT / "configs/verification_actions.yaml"),
        "--model-config",
        str(model_path),
        "--expected-code-version",
        CODE_VERSION,
    ]
    module = _module()
    checkpoint_before = checkpoint_path.read_bytes()

    assert module.main(args) == 0
    assert {path.name for path in output.iterdir()} == {
        "evaluation_report.json",
        "metrics.json",
    }
    assert checkpoint_path.read_bytes() == checkpoint_before
    metrics = json.loads((output / "metrics.json").read_text())
    report = json.loads((output / "evaluation_report.json").read_text())
    assert metrics["split"] == "test"
    assert metrics["paper_thresholds_evaluated"] is False
    assert report["sample_count"] == 6
    assert report["training_input_manifest_digest"] == "a" * 64
    with pytest.raises(FileExistsError, match="overwrite"):
        module.main(args)
