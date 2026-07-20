"""Production SOP09 streaming trainer and publication contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch
import yaml

from src.contracts import SCHEMA_VERSION, build_grid_spec
from src.datasets.risk_dataloader import (
    RiskDataContractError,
    iter_production_risk_batches,
    select_production_risk_subset,
)
from src.datasets.risk_dataset_seal import (
    LoadedRiskDataset,
    LoadedRiskDatasetFamily,
    load_risk_dataset_family,
    load_risk_dataset_seal,
    publish_risk_dataset_family,
    publish_risk_dataset_seal,
)
from src.datasets.shard_writer import (
    RISK_SHARD_LAYOUT_VERSION,
    load_risk_shard,
    write_risk_shard,
)
from src.datasets.sop03_publication import publish_checksum_envelope
from src.datasets.toy_risk_learning import frozen_channel_spec
from src.models.losses import risk_loss
from src.models.risk_model import (
    RiskModel,
    compute_risk_batch_loss,
    load_risk_checkpoint,
    production_trajectory_query_sensitivity,
)
import src.training.risk_trainer as trainer_module
from src.training.risk_trainer import (
    PRODUCTION_RISK_TRAINING_LAYOUT_VERSION,
    ProductionRiskTrainingConfig,
    ProductionRiskTrainingResult,
    train_production_risk_model as _train_production_risk_model,
)
from src.utils.config import load_config
from tests.fixtures.formal_risk_publication import (
    FormalRiskPublication,
    canonical_json,
    create_formal_risk_publication,
    sha256_bytes,
    sha256_file,
    write_canonical_json,
    write_formal_collection_handoff,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = ROOT / "configs" / "risk_model_production.yaml"
TRAIN_SCRIPT = ROOT / "scripts" / "06_train_risk_model.py"
CODE_COMMIT = "a" * 40


def train_production_risk_model(**kwargs):
    kwargs.setdefault("code_commit", CODE_COMMIT)
    resume_from = kwargs.get("resume_from")
    if resume_from is not None:
        manifest_path = Path(resume_from).parent / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        kwargs.setdefault(
            "resume_expected_publication_instance_digest_sha256",
            manifest.get("publication_instance_digest_sha256", "0" * 64),
        )
    return _train_production_risk_model(**kwargs)


def _publish_and_load(
    root: Path,
    *,
    split: str = "train",
) -> tuple[FormalRiskPublication, LoadedRiskDataset]:
    publication = _model_compatible_publication(root / "upstream", split=split)
    seal_root = publish_risk_dataset_seal(
        root / "seal",
        collection_root=publication.collection_root,
        base_config_path=publication.base_config_path,
        split_provenance_path=publication.split_provenance_path,
        expected_split=split,
        expected_collection_handoff_sha256=publication.handoff_sha256,
    )
    return publication, load_risk_dataset_seal(
        seal_root,
        collection_root=publication.collection_root,
        expected_split=split,
    )


def _publish_family(
    root: Path,
    *,
    train: LoadedRiskDataset,
    validation: LoadedRiskDataset,
) -> LoadedRiskDatasetFamily:
    _, calibration = _publish_and_load(root / "calibration", split="calibration")
    _, test = _publish_and_load(root / "test", split="test")
    family_root = publish_risk_dataset_family(
        root / "family",
        members={
            "train": train,
            "calibration": calibration,
            "val": validation,
            "test": test,
        },
    )
    return load_risk_dataset_family(family_root)


def _model_compatible_publication(
    root: Path, *, split: str
) -> FormalRiskPublication:
    """Upgrade the compact real-writer fixture to SOP09's frozen K=8 history."""

    handoff_dialect = "legacy" if split == "train" else "heldout"
    publication = create_formal_risk_publication(
        root,
        split=split,
        handoff_dialect=handoff_dialect,
    )
    handoff = json.loads(publication.handoff_path.read_text(encoding="utf-8"))
    old_samples = tuple(
        sample
        for shard_index in range(2)
        for sample in load_risk_shard(
            publication.collection_root / f"shard-{shard_index:05d}",
            grid=publication.grid,
        ).samples
    )

    base_config = yaml.safe_load(
        publication.base_config_path.read_text(encoding="utf-8")
    )
    base_config["bev"]["history_steps"] = 8
    base_config["bev"]["future_steps"] = 15
    publication.base_config_path.write_text(
        yaml.safe_dump(base_config, sort_keys=True), encoding="utf-8"
    )
    loaded_config = load_config(publication.base_config_path)
    grid = build_grid_spec(loaded_config)
    run_manifest = json.loads(
        publication.split_provenance_path.read_text(encoding="utf-8")
    )
    run_manifest["producer_protocol"]["config_snapshots"]["base"] = {
        "path": "configs/base.yaml",
        "sha256": sha256_file(publication.base_config_path),
        "value": loaded_config,
    }
    write_canonical_json(publication.split_provenance_path, run_manifest)
    for name in (
        ".producer-complete",
        "artifact_checksums.sha256",
        "artifact_checksum_summary.json",
    ):
        (publication.split_provenance_path.parent / name).unlink()
    publish_checksum_envelope(publication.split_provenance_path.parent, workers=1)

    rebuilt_samples = tuple(
        replace(
            sample,
            bev_history=np.repeat(sample.bev_history, repeats=4, axis=0).astype(
                np.float32, copy=False
            ),
        )
        for sample in old_samples
    )
    shutil.rmtree(publication.collection_root)
    publication.collection_root.mkdir()
    descriptors: list[dict[str, object]] = []
    for shard_index in range(2):
        shard_root = publication.collection_root / f"shard-{shard_index:05d}"
        paths = write_risk_shard(
            rebuilt_samples[shard_index * 6 : (shard_index + 1) * 6],
            shard_root,
            grid=grid,
            shard_index=shard_index,
            expected_sample_count=6,
        )
        loaded = load_risk_shard(shard_root, grid=grid)
        descriptors.append(
            {
                "array_layout_digest_sha256": sha256_bytes(
                    canonical_json(loaded.summary["array_layout"]).encode("utf-8")
                ),
                "audit_context_digest": loaded.summary["audit_context_digest"],
                "boundary": loaded.summary["boundary"],
                "formal_loader_verified": True,
                "manifest_digest": loaded.manifest_digest,
                "metadata_sha256": sha256_file(paths["manifest"]),
                "payload_sha256": sha256_file(paths["payload"]),
                "relative_root": shard_root.name,
                "sample_count": len(loaded.samples),
                "semantic_digest": loaded.semantic_digest,
                "shard_index": shard_index,
                "summary_sha256": sha256_file(paths["summary"]),
            }
        )
    collection_semantics = {
        "schema_version": SCHEMA_VERSION,
        "layout_version": RISK_SHARD_LAYOUT_VERSION,
        "split": split,
        "sample_count": 12,
        "shards": [
            {
                key: descriptor[key]
                for key in (
                    "shard_index",
                    "relative_root",
                    "sample_count",
                    "manifest_digest",
                    "semantic_digest",
                )
            }
            for descriptor in descriptors
        ],
    }
    handoff["shards"] = descriptors
    handoff["collection_semantic_digest_sha256"] = sha256_bytes(
        canonical_json(collection_semantics).encode("utf-8")
    )
    handoff["collection_instance_digest_sha256"] = sha256_bytes(
        canonical_json(
            {
                "semantics": collection_semantics,
                "runtime": handoff["runtime_metadata"],
            }
        ).encode("utf-8")
    )
    handoff_sha256 = write_formal_collection_handoff(
        publication.collection_root,
        handoff,
        handoff_dialect=handoff_dialect,
    )
    return replace(
        publication,
        handoff_sha256=handoff_sha256,
        grid=grid,
    )


def _config(
    *,
    stage: str = "real_1k_overfit",
    device: str = "cpu",
    epochs: int = 6,
    checkpoint_interval_steps: int = 1,
) -> ProductionRiskTrainingConfig:
    return ProductionRiskTrainingConfig(
        stage=stage,
        variant="r0",
        seed=42,
        device=device,
        hidden_channels=4,
        batch_size=6,
        epochs=epochs,
        gradient_accumulation_steps=1,
        learning_rate=0.02,
        weight_decay=0.0,
        lambda_collision=1.0,
        checkpoint_interval_steps=checkpoint_interval_steps,
    )


def _production_provenance(dataset: LoadedRiskDataset) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "channel_spec": frozen_channel_spec(),
        "model_variant": "r0",
        "config_digest": "a" * 64,
        **dict(dataset.provenance),
        "training_stage": "real_1k_overfit",
        "training_subset_digest_sha256": "b" * 64,
        "validation_risk_dataset_manifest_digest": None,
        "risk_dataset_family_digest": None,
        "global_cross_split_leakage": "NOT_PROVEN",
        "seed": 42,
        "code_commit": CODE_COMMIT,
        "runtime_environment_digest_sha256": "c" * 64,
        "training_data_scale": "fixture_standin",
        "scientific_claim_eligible": False,
        "selected_sample_count": 12,
        "consumed_sample_count": 12,
        "consumed_sample_ids_digest_sha256": "d" * 64,
    }


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    _, payload = load_risk_checkpoint(path, expected_mode="production")
    return payload["model_state_dict"]


def _rewrite_training_checksums(root: Path) -> None:
    lines = [
        f"{sha256_file(path)}  {path.name}\n"
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (root / "checksums.sha256").write_text("".join(lines), encoding="utf-8")


def _resign_training_publication(root: Path) -> dict[str, object]:
    manifest_path = root / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["semantic_digest_sha256"] = (
        trainer_module._training_manifest_semantic_digest(manifest)
    )
    manifest["publication_instance_digest_sha256"] = (
        trainer_module._training_publication_instance_digest(manifest)
    )
    write_canonical_json(manifest_path, manifest)
    write_canonical_json(
        root / ".producer-complete",
        {
            "training_layout_version": PRODUCTION_RISK_TRAINING_LAYOUT_VERSION,
            "semantic_digest_sha256": manifest["semantic_digest_sha256"],
            "publication_instance_digest_sha256": manifest[
                "publication_instance_digest_sha256"
            ],
        },
    )
    _rewrite_training_checksums(root)
    return manifest


def test_public_contract_config_and_loss_are_exact(tmp_path: Path) -> None:
    assert PRODUCTION_RISK_TRAINING_LAYOUT_VERSION == "sop09_production_training_v1"
    assert [field.name for field in fields(ProductionRiskTrainingConfig)] == [
        "stage",
        "variant",
        "seed",
        "device",
        "hidden_channels",
        "batch_size",
        "epochs",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "lambda_collision",
        "checkpoint_interval_steps",
    ]
    assert [field.name for field in fields(ProductionRiskTrainingResult)] == [
        "output_dir",
        "best_checkpoint",
        "final_checkpoint",
        "training_state_checkpoint",
        "metrics_path",
        "manifest_path",
        "semantic_digest_sha256",
    ]
    with pytest.raises(FrozenInstanceError):
        _config().epochs = 9  # type: ignore[misc]
    with pytest.raises(RiskDataContractError, match="stage"):
        replace(_config(), stage="real_1k")
    with pytest.raises(RiskDataContractError, match="variant"):
        replace(_config(), variant="r2")
    with pytest.raises(RiskDataContractError, match="learning_rate"):
        replace(_config(), learning_rate=0.0)
    assert trainer_module._training_data_scale(_config(), 999) == (
        "fixture_standin",
        False,
    )
    assert trainer_module._training_data_scale(_config(), 1000) == (
        "real_1k",
        True,
    )

    raw_config = yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    assert raw_config == {
        "mode": "production",
        "stage": "real_1k_overfit",
        "variant": "r0",
        "seed": 42,
        "device": "cuda",
        "hidden_channels": 8,
        "max_samples": 1000,
        "batch_size": 32,
        "epochs": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "lambda_collision": 1.0,
        "checkpoint_interval_steps": 100,
        "optimizer": "AdamW",
    }

    _, dataset = _publish_and_load(tmp_path / "loss")
    subset = select_production_risk_subset(dataset, max_samples=12, seed=42)
    batch, _ = next(
        iter(
            iter_production_risk_batches(
                dataset,
                subset=subset,
                batch_size=6,
                seed=42,
                epoch=0,
            )
        )
    )
    model = RiskModel(variant="r0", hidden_channels=4)
    output, actual = compute_risk_batch_loss(
        model, batch, lambda_collision=0.7
    )
    expected = risk_loss(
        output,
        risk_severity=batch.targets["risk_severity"],
        collision_label=batch.targets["collision_label"],
        lambda_collision=0.7,
    )
    assert set(actual) == set(expected)
    assert all(torch.equal(actual[key], expected[key]) for key in expected)


def test_gpu_one_shard_smoke_is_finite_reloadable_and_label_free(
    tmp_path: Path,
) -> None:
    assert torch.cuda.is_available(), "GPU smoke must run on an allocated CUDA device"
    _, dataset = _publish_and_load(tmp_path / "gpu")
    subset = select_production_risk_subset(dataset, max_samples=12, seed=42)
    result = train_production_risk_model(
        train_dataset=dataset,
        train_subset=subset,
        config=_config(stage="one_shard_smoke", device="cuda", epochs=20),
        output_dir=tmp_path / "gpu-output",
    )

    assert result.output_dir.is_absolute()
    assert result.best_checkpoint is None
    assert result.output_dir == (tmp_path / "gpu-output").absolute()
    assert {
        "final_checkpoint.pt",
        "training_state.pt",
        "metrics.json",
        "training_manifest.json",
        "config_snapshot.json",
        "checksums.sha256",
        ".producer-complete",
    }.issubset(path.name for path in result.output_dir.iterdir())
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["device"] == "cuda"
    assert metrics["optimizer_steps"] == 1
    assert metrics["selected_sample_count"] == 12
    assert metrics["train_sample_count"] == 6
    assert metrics["unique_consumed_sample_count"] == 6
    assert len(metrics["consumed_sample_ids_digest_sha256"]) == 64
    assert metrics["training_data_scale"] == "one_shard_smoke"
    assert metrics["scientific_claim_eligible"] is False
    assert metrics["quantile_crossing_rate"] == 0.0
    assert metrics["all_forward_loss_gradient_values_finite"] is True
    gpu_state = torch.load(
        result.training_state_checkpoint, map_location="cpu", weights_only=True
    )
    assert isinstance(gpu_state["rng_state"]["torch_cuda"], torch.Tensor)

    model, payload = load_risk_checkpoint(
        result.final_checkpoint, expected_mode="production"
    )
    provenance = payload["provenance"]
    assert provenance["training_stage"] == "one_shard_smoke"
    assert provenance["validation_risk_dataset_manifest_digest"] is None
    assert provenance["risk_dataset_family_digest"] is None
    assert provenance["global_cross_split_leakage"] == "NOT_PROVEN"
    assert provenance["code_commit"] == CODE_COMMIT
    assert provenance["selected_sample_count"] == 12
    assert provenance["consumed_sample_count"] == 6
    assert provenance["consumed_sample_ids_digest_sha256"] == metrics[
        "consumed_sample_ids_digest_sha256"
    ]
    assert provenance["training_data_scale"] == "one_shard_smoke"
    assert provenance["scientific_claim_eligible"] is False
    assert len(provenance["runtime_environment_digest_sha256"]) == 64

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime_environment"]["actual_device"].startswith("cuda:")
    assert manifest["runtime_environment_digest_sha256"] == provenance[
        "runtime_environment_digest_sha256"
    ]
    assert "cuda_visible_devices" not in manifest["runtime_environment"]
    load_risk_checkpoint(
        result.final_checkpoint.read_bytes(), expected_mode="production"
    )

    batch, _ = next(
        iter(
            iter_production_risk_batches(
                dataset,
                subset=subset,
                batch_size=6,
                seed=42,
                epoch=0,
            )
        )
    )

    class LabelTrapBatch:
        split = "train"
        sample_ids = batch.sample_ids
        model_inputs = batch.model_inputs
        provenance = batch.provenance

        @property
        def targets(self):  # pragma: no cover - accessing it is the failure
            raise AssertionError("production sensitivity must not access labels")

    sensitivity = production_trajectory_query_sensitivity(model, LabelTrapBatch())
    assert sensitivity["labels_accessed"] is False
    assert sensitivity["used_for_training_or_selection"] is False
    assert sensitivity["query_components_permuted"] == [
        "trajectory_channels",
        "robot_state",
    ]
    assert sensitivity["materially_sensitive"] is True
    assert sensitivity["combined_mean_absolute_delta"] > 0.0


def test_real_fixture_training_is_deterministic_and_reduces_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset = _publish_and_load(tmp_path / "real")
    subset = select_production_risk_subset(dataset, max_samples=1000, seed=42)
    config = _config(epochs=6)
    first = train_production_risk_model(
        train_dataset=dataset,
        train_subset=subset,
        config=config,
        output_dir=tmp_path / "first",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,3")
    second = train_production_risk_model(
        train_dataset=dataset,
        train_subset=subset,
        config=config,
        output_dir=tmp_path / "second",
    )
    first_metrics = json.loads(first.metrics_path.read_text(encoding="utf-8"))
    second_metrics = json.loads(second.metrics_path.read_text(encoding="utf-8"))

    assert first_metrics == second_metrics
    assert first_metrics["train_sample_count"] == 12
    assert first_metrics["selected_sample_count"] == 12
    assert first_metrics["unique_consumed_sample_count"] == 12
    assert first_metrics["consumed_sample_ids_digest_sha256"] == (
        trainer_module._sample_id_membership_digest(subset.sample_ids)
    )
    assert first_metrics["training_data_scale"] == "fixture_standin"
    assert first_metrics["scientific_claim_eligible"] is False
    assert first_metrics["stage"] == "real_1k_overfit"
    assert first_metrics["loss_history"][0] > first_metrics["loss_history"][-1]
    assert first_metrics["final_train_loss"] < first_metrics["initial_train_loss"]
    assert first.semantic_digest_sha256 == second.semantic_digest_sha256
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert first_manifest["runtime_environment"] == second_manifest[
        "runtime_environment"
    ]
    assert first_manifest["runtime_environment"]["actual_device"] == "cpu"
    assert "cuda_visible_devices" not in first_manifest["runtime_environment"]
    first_state = _state_dict(first.final_checkpoint)
    second_state = _state_dict(second.final_checkpoint)
    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)


def test_ragged_accumulation_matches_sample_weighted_effective_batches(
    tmp_path: Path,
) -> None:
    _, dataset = _publish_and_load(tmp_path / "ragged")
    subset = select_production_risk_subset(dataset, max_samples=1000, seed=42)
    config = replace(
        _config(epochs=1, checkpoint_interval_steps=1),
        batch_size=4,
        gradient_accumulation_steps=2,
    )

    torch.manual_seed(config.seed)
    expected_model = RiskModel(variant="r0", hidden_channels=4)
    expected_optimizer = torch.optim.AdamW(
        expected_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    batches = [
        batch
        for batch, _ in iter_production_risk_batches(
            dataset,
            subset=subset,
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=0,
        )
    ]
    assert [len(batch.sample_ids) for batch in batches] == [4, 2, 4, 2]
    expected_step_losses: list[float] = []
    for offset in range(0, len(batches), config.gradient_accumulation_steps):
        expected_optimizer.zero_grad(set_to_none=True)
        sample_count = 0
        weighted_loss_sum = 0.0
        for batch in batches[offset : offset + config.gradient_accumulation_steps]:
            _, losses = compute_risk_batch_loss(
                expected_model, batch, lambda_collision=config.lambda_collision
            )
            count = len(batch.sample_ids)
            (losses["total"] * count).backward()
            sample_count += count
            weighted_loss_sum += float(losses["total"].detach()) * count
        for parameter in expected_model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(sample_count)
        expected_optimizer.step()
        expected_step_losses.append(weighted_loss_sum / sample_count)

    result = train_production_risk_model(
        train_dataset=dataset,
        train_subset=subset,
        config=config,
        output_dir=tmp_path / "ragged-output",
    )
    actual_state = _state_dict(result.final_checkpoint)
    expected_state = expected_model.state_dict()
    assert actual_state.keys() == expected_state.keys()
    assert all(
        torch.allclose(actual_state[name], expected_state[name], rtol=0.0, atol=1e-7)
        for name in actual_state
    )
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["optimizer_steps"] == 2
    assert metrics["optimizer_step_loss_history"] == pytest.approx(
        expected_step_losses, rel=0.0, abs=1e-7
    )
    interval_state = torch.load(
        result.output_dir / "training_state_step_00000001.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert interval_state["epoch_running_sample_count"] == 6
    final_state = torch.load(
        result.training_state_checkpoint, map_location="cpu", weights_only=True
    )
    assert final_state["epoch_running_sample_count"] == 0
    resumed = train_production_risk_model(
        train_dataset=dataset,
        train_subset=subset,
        config=config,
        output_dir=tmp_path / "ragged-resumed-output",
        resume_from=result.output_dir / "training_state_step_00000001.pt",
    )
    assert json.loads(resumed.metrics_path.read_text(encoding="utf-8")) == metrics
    resumed_state = _state_dict(resumed.final_checkpoint)
    assert all(
        torch.equal(actual_state[name], resumed_state[name]) for name in actual_state
    )
    assert resumed.semantic_digest_sha256 == result.semantic_digest_sha256


def test_resume_from_optimizer_boundary_matches_uninterrupted_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset = _publish_and_load(tmp_path / "resume")
    subset = select_production_risk_subset(dataset, max_samples=1000, seed=42)
    config = _config(epochs=4, checkpoint_interval_steps=1)
    uninterrupted = train_production_risk_model(
        train_dataset=dataset,
        train_subset=subset,
        config=config,
        output_dir=tmp_path / "uninterrupted",
    )
    interval = uninterrupted.output_dir / "training_state_step_00000002.pt"
    assert interval.is_file()
    trusted_manifest = json.loads(
        uninterrupted.manifest_path.read_text(encoding="utf-8")
    )
    original_interval_semantic = trusted_manifest["artifact_semantic_bindings"][
        interval.name
    ]
    malicious_state = tmp_path / "malicious-state.pt"
    malicious_state.write_bytes(b"not a trusted training state")
    displaced_state = tmp_path / "displaced-state.pt"
    real_os_open = trainer_module.os.open
    opened: list[tuple[str, int]] = []
    swapped = False

    def _swap_after_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        descriptor = real_os_open(path, flags, mode, dir_fd=dir_fd)
        opened.append((Path(path).name, flags))
        if Path(path) == interval and not swapped:
            swapped = True
            interval.replace(displaced_state)
            shutil.copyfile(malicious_state, interval)
        return descriptor

    monkeypatch.setattr(trainer_module.os, "open", _swap_after_open)
    try:
        snapshotted = trainer_module._validate_published_resume_state(
            interval,
            expected_publication_instance_digest_sha256=trusted_manifest[
                "publication_instance_digest_sha256"
            ],
        )
    finally:
        monkeypatch.setattr(trainer_module.os, "open", real_os_open)
        if displaced_state.exists():
            interval.unlink(missing_ok=True)
            displaced_state.replace(interval)
    assert swapped is True
    assert snapshotted.target_state[
        "training_state_semantic_digest_sha256"
    ] == original_interval_semantic
    direct_filenames = {
        path.name for path in uninterrupted.output_dir.iterdir() if path.is_file()
    }
    assert {name for name, _ in opened} == direct_filenames
    assert len(opened) == len(direct_filenames)
    assert all(flags & trainer_module.os.O_NOFOLLOW for _, flags in opened)

    resumed = train_production_risk_model(
        train_dataset=dataset,
        train_subset=subset,
        config=config,
        output_dir=tmp_path / "resumed",
        resume_from=interval,
    )

    expected = _state_dict(uninterrupted.final_checkpoint)
    actual = _state_dict(resumed.final_checkpoint)
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[name], actual[name]) for name in expected)
    assert json.loads(uninterrupted.metrics_path.read_text(encoding="utf-8")) == (
        json.loads(resumed.metrics_path.read_text(encoding="utf-8"))
    )
    assert uninterrupted.semantic_digest_sha256 == resumed.semantic_digest_sha256

    isolated = tmp_path / "isolated-state.pt"
    shutil.copyfile(interval, isolated)
    with pytest.raises(RiskDataContractError, match="published"):
        _train_production_risk_model(
            train_dataset=dataset,
            train_subset=subset,
            config=config,
            output_dir=tmp_path / "isolated-resume-output",
            resume_from=isolated,
            resume_expected_publication_instance_digest_sha256="0" * 64,
            code_commit=CODE_COMMIT,
        )

    with pytest.raises(RiskDataContractError, match="supplied together"):
        _train_production_risk_model(
            train_dataset=dataset,
            train_subset=subset,
            config=config,
            output_dir=tmp_path / "missing-resume-instance",
            resume_from=interval,
            code_commit=CODE_COMMIT,
        )

    tampered_publication = tmp_path / "tampered-publication"
    shutil.copytree(uninterrupted.output_dir, tampered_publication)
    tampered_interval = tampered_publication / interval.name
    tampered_payload = torch.load(
        tampered_interval, map_location="cpu", weights_only=True
    )
    first_name = sorted(tampered_payload["model_state_dict"])[0]
    tampered_payload["model_state_dict"][first_name].reshape(-1)[0] += 1.0
    tampered_payload["training_state_semantic_digest_sha256"] = (
        trainer_module._training_state_semantic_digest(tampered_payload)
    )
    torch.save(tampered_payload, tampered_interval)
    with pytest.raises(RiskDataContractError, match="checksum|artifact_sha256"):
        train_production_risk_model(
            train_dataset=dataset,
            train_subset=subset,
            config=config,
            output_dir=tmp_path / "tampered-resume-output",
            resume_from=tampered_interval,
        )
    assert not (tmp_path / "tampered-resume-output").exists()

    resigned_publication = tmp_path / "resigned-interval-publication"
    shutil.copytree(uninterrupted.output_dir, resigned_publication)
    resigned_interval = resigned_publication / interval.name
    resigned_payload = torch.load(
        resigned_interval, map_location="cpu", weights_only=True
    )
    resigned_name = sorted(resigned_payload["model_state_dict"])[0]
    resigned_payload["model_state_dict"][resigned_name].reshape(-1)[0] += 2.0
    resigned_payload["training_state_semantic_digest_sha256"] = (
        trainer_module._training_state_semantic_digest(resigned_payload)
    )
    torch.save(resigned_payload, resigned_interval)
    resigned_manifest_path = resigned_publication / "training_manifest.json"
    resigned_manifest = json.loads(
        resigned_manifest_path.read_text(encoding="utf-8")
    )
    original_scientific_digest = resigned_manifest["semantic_digest_sha256"]
    trusted_original_instance = resigned_manifest[
        "publication_instance_digest_sha256"
    ]
    resigned_manifest["artifact_sha256"][interval.name] = sha256_file(
        resigned_interval
    )
    resigned_manifest["artifact_semantic_bindings"][interval.name] = (
        resigned_payload["training_state_semantic_digest_sha256"]
    )
    write_canonical_json(resigned_manifest_path, resigned_manifest)
    resigned_manifest = _resign_training_publication(resigned_publication)
    assert resigned_manifest["semantic_digest_sha256"] == (
        original_scientific_digest
    )
    assert resigned_manifest["publication_instance_digest_sha256"] != (
        trusted_original_instance
    )
    with pytest.raises(RiskDataContractError, match="trusted expected digest"):
        _train_production_risk_model(
            train_dataset=dataset,
            train_subset=subset,
            config=config,
            output_dir=tmp_path / "resigned-interval-resume-output",
            resume_from=resigned_interval,
            resume_expected_publication_instance_digest_sha256=(
                trusted_original_instance
            ),
            code_commit=CODE_COMMIT,
        )
    assert not (tmp_path / "resigned-interval-resume-output").exists()

    uninterrupted_six = train_production_risk_model(
        train_dataset=dataset,
        train_subset=subset,
        config=replace(config, epochs=6),
        output_dir=tmp_path / "uninterrupted-six",
    )
    resumed_six = train_production_risk_model(
        train_dataset=dataset,
        train_subset=subset,
        config=replace(config, epochs=6),
        output_dir=tmp_path / "resumed-six",
        resume_from=uninterrupted.training_state_checkpoint,
    )
    expected_six = _state_dict(uninterrupted_six.final_checkpoint)
    actual_six = _state_dict(resumed_six.final_checkpoint)
    assert all(
        torch.equal(expected_six[name], actual_six[name]) for name in expected_six
    )
    assert json.loads(
        uninterrupted_six.metrics_path.read_text(encoding="utf-8")
    ) == json.loads(resumed_six.metrics_path.read_text(encoding="utf-8"))
    expected_final_state = torch.load(
        uninterrupted_six.training_state_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    actual_final_state = torch.load(
        resumed_six.training_state_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    assert expected_final_state["training_state_semantic_digest_sha256"] == (
        actual_final_state["training_state_semantic_digest_sha256"]
    )
    assert expected_final_state["rng_state"]["torch_cuda"] is None
    assert actual_final_state["rng_state"]["torch_cuda"] is None
    assert uninterrupted_six.semantic_digest_sha256 == (
        resumed_six.semantic_digest_sha256
    )
    uninterrupted_six_manifest = json.loads(
        uninterrupted_six.manifest_path.read_text(encoding="utf-8")
    )
    resumed_six_manifest = json.loads(
        resumed_six.manifest_path.read_text(encoding="utf-8")
    )
    assert uninterrupted_six_manifest["publication_instance_digest_sha256"] != (
        resumed_six_manifest["publication_instance_digest_sha256"]
    )
    resumed_intervals = sorted(
        name
        for name in resumed_six_manifest["artifact_semantic_bindings"]
        if name.startswith("training_state_step_")
    )
    assert resumed_intervals == [
        f"training_state_step_{step:08d}.pt" for step in range(9, 13)
    ]
    parent_manifest = json.loads(
        uninterrupted.manifest_path.read_text(encoding="utf-8")
    )
    assert resumed_six_manifest["resume_lineage"] == {
        "parent_scientific_digest_sha256": parent_manifest[
            "semantic_digest_sha256"
        ],
        "parent_publication_instance_digest_sha256": parent_manifest[
            "publication_instance_digest_sha256"
        ],
        "resume_state_filename": "training_state.pt",
        "resume_state_file_sha256": parent_manifest["artifact_sha256"][
            "training_state.pt"
        ],
        "resume_state_semantic_digest_sha256": parent_manifest[
            "artifact_semantic_bindings"
        ]["training_state.pt"],
        "resume_optimizer_step": 8,
    }


def test_stage_gates_checkpoint_provenance_and_atomic_no_clobber(
    tmp_path: Path,
) -> None:
    _, train = _publish_and_load(tmp_path / "train")
    _, validation = _publish_and_load(tmp_path / "validation", split="val")
    family = _publish_family(
        tmp_path / "dataset-family",
        train=train,
        validation=validation,
    )
    full = select_production_risk_subset(train, max_samples=1000, seed=42)
    partial = select_production_risk_subset(train, max_samples=11, seed=42)

    compact_publication = create_formal_risk_publication(
        tmp_path / "compact-upstream", split="train"
    )
    compact_seal = publish_risk_dataset_seal(
        tmp_path / "compact-seal",
        collection_root=compact_publication.collection_root,
        base_config_path=compact_publication.base_config_path,
        split_provenance_path=compact_publication.split_provenance_path,
        expected_split="train",
        expected_collection_handoff_sha256=compact_publication.handoff_sha256,
    )
    compact = load_risk_dataset_seal(
        compact_seal,
        collection_root=compact_publication.collection_root,
        expected_split="train",
    )
    compact_subset = select_production_risk_subset(
        compact, max_samples=12, seed=42
    )
    with pytest.raises(RiskDataContractError, match="code_commit"):
        _train_production_risk_model(
            train_dataset=train,
            train_subset=full,
            config=_config(stage="one_shard_smoke", epochs=1),
            output_dir=tmp_path / "unversioned-rejected",
            code_commit="unversioned",
        )
    with pytest.raises(RiskDataContractError, match="history_steps=8"):
        train_production_risk_model(
            train_dataset=compact,
            train_subset=compact_subset,
            config=_config(stage="one_shard_smoke", epochs=1),
            output_dir=tmp_path / "compact-rejected",
        )

    with pytest.raises(RiskDataContractError, match="complete 12-sample fixture"):
        train_production_risk_model(
            train_dataset=train,
            train_subset=partial,
            config=_config(),
            output_dir=tmp_path / "partial",
        )
    with pytest.raises(RiskDataContractError, match="rejects validation"):
        train_production_risk_model(
            train_dataset=train,
            train_subset=full,
            config=_config(),
            output_dir=tmp_path / "unexpected-validation",
            validation_dataset=validation,
        )
    with pytest.raises(RiskDataContractError, match="validation"):
        train_production_risk_model(
            train_dataset=train,
            train_subset=full,
            config=_config(stage="formal_50k"),
            output_dir=tmp_path / "formal-no-validation",
        )
    with pytest.raises(
        RiskDataContractError, match="authenticated risk_dataset_family_v1 loader"
    ):
        train_production_risk_model(
            train_dataset=train,
            train_subset=full,
            config=_config(stage="formal_50k"),
            output_dir=tmp_path / "formal-no-audit",
            validation_dataset=validation,
        )
    with pytest.raises(
        RiskDataContractError, match="authenticated risk_dataset_family_v1 loader"
    ):
        train_production_risk_model(
            train_dataset=train,
            train_subset=full,
            config=_config(stage="formal_50k"),
            output_dir=tmp_path / "formal-forged-audit",
            validation_dataset=validation,
            dataset_family={  # type: ignore[arg-type]
                "global_cross_split_leakage": "PROVEN",
                "risk_dataset_family_digest": "f" * 64,
            },
        )
    assert trainer_module._validate_stage_gates(
        train_dataset=train,
        train_subset=full,
        config=_config(stage="formal_50k", epochs=1),
        validation_dataset=validation,
        dataset_family=family,
    ) == (
        validation.risk_dataset_manifest_digest,
        family.risk_dataset_family_digest,
        "PROVEN",
    )

    output = tmp_path / "atomic"
    result = train_production_risk_model(
        train_dataset=train,
        train_subset=full,
        config=_config(stage="one_shard_smoke", epochs=1),
        output_dir=output,
    )
    before = result.manifest_path.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        train_production_risk_model(
            train_dataset=train,
            train_subset=full,
            config=_config(stage="one_shard_smoke", epochs=1),
            output_dir=output,
        )
    assert result.manifest_path.read_bytes() == before
    assert not list(tmp_path.glob(".atomic.staging-*"))

    bad_provenance = _production_provenance(train)
    bad_provenance["g1_split_manifest_digest"] = "1" * 64
    from src.models.risk_model import save_risk_checkpoint

    with pytest.raises(RiskDataContractError, match="BLAKE2b-128"):
        save_risk_checkpoint(
            tmp_path / "bad.pt",
            model=RiskModel(variant="r0", hidden_channels=4),
            mode="production",
            provenance=bad_provenance,
        )
    bad_runtime = _production_provenance(train)
    bad_runtime["runtime_environment_digest_sha256"] = "not-a-digest"
    with pytest.raises(
        RiskDataContractError,
        match="runtime_environment_digest_sha256.*SHA-256",
    ):
        save_risk_checkpoint(
            tmp_path / "bad-runtime.pt",
            model=RiskModel(variant="r0", hidden_channels=4),
            mode="production",
            provenance=bad_runtime,
        )


def test_production_cli_returns_without_toy_fallthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication, dataset = _publish_and_load(tmp_path / "cli")
    spec = importlib.util.spec_from_file_location("sop09_train_cli", TRAIN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _forbidden_toy_constructor(*args, **kwargs):
        del args, kwargs
        raise AssertionError("production CLI fell through to toy construction")

    monkeypatch.setattr(module, "make_toy_risk_dataset", _forbidden_toy_constructor)
    unversioned_output = tmp_path / "cli-unversioned-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TRAIN_SCRIPT),
            "--config",
            str(PRODUCTION_CONFIG),
            "--output-dir",
            str(unversioned_output),
            "--train-seal-root",
            str(dataset.seal_root),
            "--train-collection-root",
            str(publication.collection_root),
            "--stage",
            "one_shard_smoke",
            "--max-samples",
            "6",
            "--batch-size",
            "6",
            "--device",
            "cuda",
        ],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert not unversioned_output.exists()

    output = tmp_path / "cli-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TRAIN_SCRIPT),
            "--config",
            str(PRODUCTION_CONFIG),
            "--output-dir",
            str(output),
            "--train-seal-root",
            str(dataset.seal_root),
            "--train-collection-root",
            str(publication.collection_root),
            "--stage",
            "one_shard_smoke",
            "--max-samples",
            "6",
            "--batch-size",
            "6",
            "--device",
            "cuda",
            "--code-commit",
            CODE_COMMIT,
        ],
    )
    assert module.main() == 0
    assert (output / ".producer-complete").is_file()

    unpaired_resume_output = tmp_path / "cli-unpaired-resume-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TRAIN_SCRIPT),
            "--config",
            str(PRODUCTION_CONFIG),
            "--output-dir",
            str(unpaired_resume_output),
            "--train-seal-root",
            str(dataset.seal_root),
            "--train-collection-root",
            str(publication.collection_root),
            "--stage",
            "one_shard_smoke",
            "--max-samples",
            "6",
            "--batch-size",
            "6",
            "--device",
            "cuda",
            "--code-commit",
            CODE_COMMIT,
            "--resume-from",
            str(output / "training_state.pt"),
        ],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert not unpaired_resume_output.exists()

    validation_publication, validation_dataset = _publish_and_load(
        tmp_path / "cli-validation", split="val"
    )
    forged_audit = tmp_path / "forged-audit.json"
    forged_audit.write_text(
        json.dumps(
            {
                "global_cross_split_leakage": "PROVEN",
                "risk_dataset_family_digest": "f" * 64,
            }
        ),
        encoding="utf-8",
    )
    formal_output = tmp_path / "cli-formal-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TRAIN_SCRIPT),
            "--config",
            str(PRODUCTION_CONFIG),
            "--output-dir",
            str(formal_output),
            "--train-seal-root",
            str(dataset.seal_root),
            "--train-collection-root",
            str(publication.collection_root),
            "--validation-seal-root",
            str(validation_dataset.seal_root),
            "--validation-collection-root",
            str(validation_publication.collection_root),
            "--stage",
            "formal_50k",
            "--max-samples",
            "12",
            "--batch-size",
            "6",
            "--device",
            "cuda",
            "--cross-split-audit",
            str(forged_audit),
            "--code-commit",
            CODE_COMMIT,
        ],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert not formal_output.exists()
