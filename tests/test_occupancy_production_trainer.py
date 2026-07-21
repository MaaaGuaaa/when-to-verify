"""Production SOP08 optimizer, scorer, and publication invariants."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from src.contracts import (
    HISTORY_CHANNELS,
    STATE_CHANNELS,
    TRAJECTORY_CHANNELS,
)
from src.datasets.risk_dataloader import (
    ProductionOccupancyBatch,
    select_production_risk_subset,
)
from src.datasets.risk_dataset_seal import (
    load_risk_dataset_seal,
    publish_risk_dataset_seal,
)
from src.datasets.risk_training_store import (
    load_authenticated_occupancy_snapshot,
)
from src.evaluation.risk_baselines import score_production_occupancy_baseline
from src.models.occupancy_baseline import (
    ConvGRUOccupancyPredictor,
    LearnedOccupancyRiskAggregator,
)
from src.training.occupancy_trainer import (
    ProductionOccupancyTrainingConfig,
    compute_weighted_binary_loss_sum,
    compute_global_binary_pos_weight,
    load_production_occupancy_checkpoint,
    production_occupancy_publication_instance_digest,
    train_production_occupancy_baselines,
    validate_production_occupancy_training_publication,
)
from tests.fixtures.formal_risk_publication import (
    create_formal_risk_publication,
    create_formal_risk_sidecar_publication,
)


def _batch(*, batch_size: int = 3, height: int = 4, width: int = 4) -> ProductionOccupancyBatch:
    history = torch.zeros(
        batch_size,
        8,
        len(HISTORY_CHANNELS),
        height,
        width,
        dtype=torch.float32,
    )
    dynamic = HISTORY_CHANNELS.index("past_dynamic_occupancy")
    history[:, -1, dynamic, 1, 1] = 0.75
    state = torch.zeros(
        batch_size,
        len(STATE_CHANNELS),
        height,
        width,
        dtype=torch.float32,
    )
    state[:, STATE_CHANNELS.index("last_seen_occupancy"), 1, 1] = 0.75
    state[:, STATE_CHANNELS.index("occlusion_age_map")] = 0.25
    query = torch.zeros(batch_size, 15, height, width, dtype=torch.float32)
    query[:, :, 1, 1] = 1.0
    hidden = torch.zeros_like(query)
    hidden[0, :, 1, 1] = 1.0
    collision = torch.zeros(batch_size, dtype=torch.float32)
    collision[0] = 1.0
    return ProductionOccupancyBatch(
        model_inputs={
            "bev_history": history,
            "state_channels": state,
            "trajectory_channels": torch.zeros(
                batch_size,
                len(TRAJECTORY_CHANNELS),
                height,
                width,
                dtype=torch.float32,
            ),
            "robot_state": torch.zeros(batch_size, 2, dtype=torch.float32),
        },
        targets={
            "collision_label": collision,
            "risk_severity": collision.clone(),
            "min_clearance": torch.ones(batch_size, dtype=torch.float32),
            "near_miss": torch.zeros(batch_size, dtype=torch.float32),
        },
        query_inputs={
            "robot_endpoint_footprints": query,
            "endpoint_times_s": torch.arange(1, 16, dtype=torch.float32) * 0.2,
        },
        occupancy_targets={"hidden_risk_occupancy": hidden},
        sample_ids=tuple(f"sample-{index}" for index in range(batch_size)),
        split="train",
        provenance={
            "mode": "production",
            "occupancy_sidecar_collection_digest_sha256": "a" * 64,
        },
    )


def test_production_score_api_is_oracle_isolated_and_b1_through_b4_are_bounded() -> None:
    batch = _batch()
    model = ConvGRUOccupancyPredictor(hidden_channels=2, future_steps=15)
    aggregator = LearnedOccupancyRiskAggregator(future_steps=15, hidden_dim=4)
    signature = inspect.signature(score_production_occupancy_baseline)
    assert "occupancy_targets" not in signature.parameters
    assert "targets" not in signature.parameters
    assert "batch" not in signature.parameters

    scores = {
        method: score_production_occupancy_baseline(
            method=method,
            model_inputs=batch.model_inputs,
            query_inputs=batch.query_inputs,
            occupancy_model=model,
            learned_aggregator=aggregator,
            b2_tau_s=2.0,
            b2_a_max_s=5.0,
            sigma_time_s=2.0,
        )
        for method in ("B1", "B2", "B3", "B4")
    }
    changed_oracle = replace(
        batch,
        occupancy_targets={
            "hidden_risk_occupancy": 1.0
            - batch.occupancy_targets["hidden_risk_occupancy"]
        },
    )
    for method, score in scores.items():
        repeated = score_production_occupancy_baseline(
            method=method,
            model_inputs=changed_oracle.model_inputs,
            query_inputs=changed_oracle.query_inputs,
            occupancy_model=model,
            learned_aggregator=aggregator,
            b2_tau_s=2.0,
            b2_a_max_s=5.0,
            sigma_time_s=2.0,
        )
        assert score.dtype == torch.float32, method
        assert score.shape == (3,), method
        assert torch.isfinite(score).all(), method
        assert torch.all((score >= 0.0) & (score <= 1.0)), method
        assert torch.equal(score, repeated), method


def test_global_class_weight_requires_both_classes_and_is_not_batch_local() -> None:
    assert compute_global_binary_pos_weight(
        positive_count=3,
        negative_count=9,
        name="occupancy",
    ) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="occupancy.*positive"):
        compute_global_binary_pos_weight(
            positive_count=0,
            negative_count=9,
            name="occupancy",
        )
    with pytest.raises(ValueError, match="collision.*negative"):
        compute_global_binary_pos_weight(
            positive_count=3,
            negative_count=0,
            name="collision",
        )


def test_production_config_freezes_engineering_stages_and_positive_hyperparameters() -> None:
    config = ProductionOccupancyTrainingConfig(
        stage="one_shard_smoke",
        seed=7,
        device="cpu",
        hidden_channels=2,
        convgru_kernel_size=3,
        learned_aggregator_hidden_dim=4,
        batch_size=5,
        occupancy_epochs=1,
        aggregator_epochs=1,
        gradient_accumulation_steps=1,
        occupancy_learning_rate=0.01,
        aggregator_learning_rate=0.01,
        weight_decay=0.0,
        b2_tau_s=2.0,
        b2_a_max_s=5.0,
        sigma_time_s=2.0,
    )
    assert config.stage == "one_shard_smoke"
    with pytest.raises(ValueError, match="stage"):
        replace(config, stage="toy")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="occupancy_learning_rate"):
        replace(config, occupancy_learning_rate=0.0)
    for field in (
        "occupancy_epochs",
        "aggregator_epochs",
        "gradient_accumulation_steps",
        "checkpoint_interval_steps",
    ):
        with pytest.raises(ValueError, match=f"{field}.*one_shard_smoke"):
            replace(config, **{field: 2})


def test_ragged_binary_loss_is_weighted_by_cells_not_microbatches() -> None:
    logits = torch.tensor([-1.0, 0.5, 1.5, -0.25, 0.75], dtype=torch.float32)
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    full_sum, full_count = compute_weighted_binary_loss_sum(
        logits,
        targets,
        pos_weight=2.0,
        name="collision",
    )
    ragged = [slice(0, 2), slice(2, 5)]
    ragged_sum = logits.new_zeros(())
    ragged_count = 0
    for current in ragged:
        loss_sum, count = compute_weighted_binary_loss_sum(
            logits[current],
            targets[current],
            pos_weight=2.0,
            name="collision",
        )
        ragged_sum = ragged_sum + loss_sum
        ragged_count += count
    assert ragged_count == full_count == 5
    assert torch.allclose(ragged_sum / ragged_count, full_sum / full_count)


def _production_fixture(tmp_path: Path):
    publication = create_formal_risk_publication(
        tmp_path / "upstream",
        history_steps=8,
        future_steps=15,
    )
    sidecars = create_formal_risk_sidecar_publication(
        publication,
        tmp_path / "sidecars",
    )
    seal_root = publish_risk_dataset_seal(
        tmp_path / "seal",
        collection_root=publication.collection_root,
        base_config_path=publication.base_config_path,
        split_provenance_path=publication.split_provenance_path,
        expected_split="train",
        expected_collection_handoff_sha256=publication.handoff_sha256,
        sidecar_root=sidecars.sidecar_root,
    )
    dataset = load_risk_dataset_seal(
        seal_root,
        collection_root=publication.collection_root,
        expected_split="train",
        sidecar_root=sidecars.sidecar_root,
    )
    subset = select_production_risk_subset(dataset, max_samples=12, seed=17)
    return dataset, subset, sidecars.sidecar_root


def _training_config(*, device: str = "cpu") -> ProductionOccupancyTrainingConfig:
    return ProductionOccupancyTrainingConfig(
        stage="one_shard_smoke",
        seed=17,
        device=device,
        hidden_channels=2,
        convgru_kernel_size=3,
        learned_aggregator_hidden_dim=4,
        batch_size=5,
        occupancy_epochs=1,
        aggregator_epochs=1,
        gradient_accumulation_steps=1,
        occupancy_learning_rate=0.02,
        aggregator_learning_rate=0.02,
        weight_decay=0.0,
        b2_tau_s=2.0,
        b2_a_max_s=5.0,
        sigma_time_s=2.0,
    )


def test_authenticated_snapshot_trainer_does_not_reopen_strict_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, subset, sidecar_root = _production_fixture(tmp_path)
    _, snapshot = load_authenticated_occupancy_snapshot(
        dataset.seal_root,
        collection_root=dataset.collection_root,
        sidecar_root=sidecar_root,
        expected_split="train",
        cache_root=tmp_path / "cache",
    )

    def forbidden_strict_iterator(*args, **kwargs):
        raise AssertionError("snapshot trainer must not reopen source sidecars")

    monkeypatch.setattr(
        "src.training.occupancy_trainer.iter_production_occupancy_batches",
        forbidden_strict_iterator,
    )
    result = train_production_occupancy_baselines(
        train_dataset=dataset,
        train_subset=subset,
        sidecar_root=sidecar_root,
        config=_training_config(),
        output_dir=tmp_path / "snapshot-output",
        code_commit="a" * 40,
        training_snapshot=snapshot,
    )

    assert result.output_dir.is_dir()


def test_real_fixture_one_shard_smoke_freezes_b3_and_publishes_bound_artifact(
    tmp_path: Path,
) -> None:
    dataset, subset, sidecar_root = _production_fixture(tmp_path)
    result = train_production_occupancy_baselines(
        train_dataset=dataset,
        train_subset=subset,
        sidecar_root=sidecar_root,
        config=_training_config(),
        output_dir=tmp_path / "training",
        code_commit="b" * 40,
    )
    manifest = validate_production_occupancy_training_publication(
        result.output_dir,
        expected_publication_instance_digest_sha256=(
            result.publication_instance_digest_sha256
        ),
    )
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    config_snapshot = json.loads(
        (result.output_dir / "config_snapshot.json").read_text(encoding="utf-8")
    )
    checkpoint = load_production_occupancy_checkpoint(
        result.final_checkpoint,
        expected_checkpoint_semantic_digest_sha256=manifest[
            "artifact_semantic_bindings"
        ]["final_checkpoint.pt"],
        expected_provenance=manifest["provenance"],
        expected_config_digest_sha256=manifest["config_digest_sha256"],
    )

    assert np.isfinite(metrics["occupancy_final_loss"])
    assert np.isfinite(metrics["aggregator_final_loss"])
    assert metrics["training_data_scale"] == "one_shard_smoke"
    assert metrics["scientific_claim_eligible"] is False
    assert config_snapshot["scientific_claim_eligible"] is False
    assert manifest["provenance"]["scientific_claim_eligible"] is False
    assert metrics["test_samples_used_for_training_or_selection"] == 0
    assert metrics["b3_state_digest_before_b4_sha256"] == metrics[
        "b3_state_digest_after_b4_sha256"
    ]
    assert metrics["b3_frozen_during_b4"] is True
    assert manifest["provenance"]["risk_dataset_manifest_digest"] == (
        dataset.risk_dataset_manifest_digest
    )
    assert manifest["provenance"][
        "occupancy_sidecar_collection_digest_sha256"
    ] == dataset.manifest["occupancy_sidecars"]["collection_digest_sha256"]
    assert manifest["provenance"]["subset_digest_sha256"] == (
        subset.sample_ids_digest_sha256
    )
    assert manifest["provenance"]["code_commit"] == "b" * 40
    assert manifest["validation_status"] == "unavailable_engineering_final_only"
    assert manifest["best_checkpoint"] is None
    assert checkpoint["b3_state_digest_sha256"] == metrics[
        "b3_state_digest_after_b4_sha256"
    ]
    assert (result.output_dir / ".producer-complete").is_file()
    assert (result.output_dir / "checksums.sha256").is_file()
    assert not list(result.output_dir.glob("training_state*.pt"))
    with pytest.raises(FileExistsError):
        train_production_occupancy_baselines(
            train_dataset=dataset,
            train_subset=subset,
            sidecar_root=sidecar_root,
            config=_training_config(),
            output_dir=result.output_dir,
            code_commit="b" * 40,
        )


def test_scale_and_resume_gates_fail_closed_without_creating_output(
    tmp_path: Path,
) -> None:
    dataset, subset, sidecar_root = _production_fixture(tmp_path)
    for stage in ("real_1k_overfit", "formal_50k"):
        output = tmp_path / stage
        with pytest.raises(ValueError, match=f"{stage}.*not implemented"):
            train_production_occupancy_baselines(
                train_dataset=dataset,
                train_subset=subset,
                sidecar_root=sidecar_root,
                config=replace(_training_config(), stage=stage),  # type: ignore[arg-type]
                output_dir=output,
                code_commit="c" * 40,
            )
        assert not output.exists()
    resume_output = tmp_path / "resume"
    with pytest.raises(ValueError, match="resume.*not implemented"):
        train_production_occupancy_baselines(
            train_dataset=dataset,
            train_subset=subset,
            sidecar_root=sidecar_root,
            config=_training_config(),
            output_dir=resume_output,
            code_commit="c" * 40,
            resume_from=tmp_path / "parent",
            resume_expected_publication_instance_digest_sha256="d" * 64,
        )
    assert not resume_output.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="allocated CUDA GPU required")
def test_one_shard_gpu_smoke_reports_actual_consumption_and_all_baseline_scores(
    tmp_path: Path,
) -> None:
    dataset, subset, sidecar_root = _production_fixture(tmp_path)
    config = replace(
        _training_config(device="cuda"),
        stage="one_shard_smoke",
        occupancy_epochs=1,
        aggregator_epochs=1,
        gradient_accumulation_steps=1,
    )
    result = train_production_occupancy_baselines(
        train_dataset=dataset,
        train_subset=subset,
        sidecar_root=sidecar_root,
        config=config,
        output_dir=tmp_path / "gpu-smoke",
        code_commit="d" * 40,
    )
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert metrics["selected_sample_count"] == 12
    assert metrics["global_statistics_sample_count"] == 12
    assert metrics["train_sample_count"] == 5
    assert metrics["optimizer_steps"] == 2
    assert set(metrics["baseline_score_summary"]) == {"B1", "B2", "B3", "B4"}
    for summary in metrics["baseline_score_summary"].values():
        assert summary["sample_count"] == 5
        assert 0.0 <= summary["minimum"] <= summary["maximum"] <= 1.0
        assert 0.0 <= summary["mean"] <= 1.0
    assert manifest["runtime_environment"]["actual_device"].startswith("cuda:")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _resign_publication(root: Path, manifest: dict[str, object]) -> str:
    artifact_hashes = manifest["artifact_sha256"]
    bindings = manifest["artifact_semantic_bindings"]
    assert isinstance(artifact_hashes, dict)
    assert isinstance(bindings, dict)
    for filename in ("config_snapshot.json", "metrics.json", "final_checkpoint.pt"):
        digest = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        artifact_hashes[filename] = digest
        if filename != "final_checkpoint.pt":
            bindings[filename] = digest
    config_digest = hashlib.sha256(
        (root / "config_snapshot.json").read_bytes()
    ).hexdigest()
    manifest["config_digest_sha256"] = config_digest
    projection = {
        key: value
        for key, value in manifest.items()
        if key != "semantic_digest_sha256"
    }
    manifest["semantic_digest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(projection)
    ).hexdigest()
    manifest_bytes = _canonical_json_bytes(manifest)
    (root / "training_manifest.json").write_bytes(manifest_bytes)
    checksum_targets = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.name not in {"checksums.sha256", ".producer-complete"}
    )
    checksums_bytes = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in checksum_targets
    ).encode("ascii")
    (root / "checksums.sha256").write_bytes(checksums_bytes)
    instance = production_occupancy_publication_instance_digest(
        manifest_bytes=manifest_bytes,
        checksums_bytes=checksums_bytes,
    )
    marker = {
        "training_layout_version": "sop08_production_training_v1",
        "semantic_digest_sha256": manifest["semantic_digest_sha256"],
        "publication_instance_digest_sha256": instance,
        "training_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "checksums_sha256": hashlib.sha256(checksums_bytes).hexdigest(),
    }
    (root / ".producer-complete").write_bytes(_canonical_json_bytes(marker))
    return instance


def test_resigned_nested_schema_and_scientific_gate_attacks_are_rejected(
    tmp_path: Path,
) -> None:
    dataset, subset, sidecar_root = _production_fixture(tmp_path)
    result = train_production_occupancy_baselines(
        train_dataset=dataset,
        train_subset=subset,
        sidecar_root=sidecar_root,
        config=_training_config(),
        output_dir=tmp_path / "source-publication",
        code_commit="1" * 40,
    )
    attacks = (
        ("config_ghost", "config snapshot keys"),
        ("config_scientific", "scientific_claim_eligible"),
        ("metrics_ghost", "metrics keys"),
        ("metrics_scientific", "scientific_claim_eligible"),
        ("provenance_ghost", "provenance keys"),
        ("provenance_scientific", "scientific_claim_eligible"),
    )
    for attack, expected_message in attacks:
        attacked = tmp_path / attack
        shutil.copytree(result.output_dir, attacked)
        manifest_path = attacked / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if attack.startswith("config_"):
            target_path = attacked / "config_snapshot.json"
            target = json.loads(target_path.read_text(encoding="utf-8"))
        elif attack.startswith("metrics_"):
            target_path = attacked / "metrics.json"
            target = json.loads(target_path.read_text(encoding="utf-8"))
        else:
            target_path = None
            target = manifest["provenance"]
            assert isinstance(target, dict)
        if attack.endswith("ghost"):
            target["ghost_nested_key"] = "resigned"
        else:
            target["scientific_claim_eligible"] = True
        if target_path is not None:
            target_path.write_bytes(_canonical_json_bytes(target))
        expected_instance = _resign_publication(attacked, manifest)
        with pytest.raises(ValueError, match=expected_message):
            validate_production_occupancy_training_publication(
                attacked,
                expected_publication_instance_digest_sha256=expected_instance,
            )


def test_checkpoint_model_spec_requires_real_integers(
    tmp_path: Path,
) -> None:
    dataset, subset, sidecar_root = _production_fixture(tmp_path)
    result = train_production_occupancy_baselines(
        train_dataset=dataset,
        train_subset=subset,
        sidecar_root=sidecar_root,
        config=_training_config(),
        output_dir=tmp_path / "checkpoint-source",
        code_commit="2" * 40,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        result.final_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    for bad_value in (True, "2", 2.0):
        attacked = copy.deepcopy(checkpoint)
        attacked["model_spec"]["hidden_channels"] = bad_value
        projection = {
            key: attacked[key]
            for key in (
                "checkpoint_layout_version",
                "config_digest_sha256",
                "provenance",
                "model_spec",
                "b3_state_digest_sha256",
                "b4_state_digest_sha256",
            )
        }
        attacked["checkpoint_semantic_digest_sha256"] = hashlib.sha256(
            _canonical_json_bytes(projection)
        ).hexdigest()
        attacked_path = tmp_path / f"bad-model-spec-{type(bad_value).__name__}.pt"
        torch.save(attacked, attacked_path)
        with pytest.raises(ValueError, match="model_spec.*integer"):
            load_production_occupancy_checkpoint(
                attacked_path,
                expected_checkpoint_semantic_digest_sha256=attacked[
                    "checkpoint_semantic_digest_sha256"
                ],
                expected_provenance=manifest["provenance"],
                expected_config_digest_sha256=manifest["config_digest_sha256"],
            )


def test_resigned_manifest_ghost_key_is_rejected_even_with_new_instance_digest(
    tmp_path: Path,
) -> None:
    dataset, subset, sidecar_root = _production_fixture(tmp_path)
    config = replace(
        _training_config(),
        stage="one_shard_smoke",
        occupancy_epochs=1,
        aggregator_epochs=1,
    )
    result = train_production_occupancy_baselines(
        train_dataset=dataset,
        train_subset=subset,
        sidecar_root=sidecar_root,
        config=config,
        output_dir=tmp_path / "ghost-publication",
        code_commit="f" * 40,
    )
    manifest_path = result.output_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ghost_review_note"] = "not part of the frozen publication"
    manifest_payload = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_payload)
    checksum_targets = sorted(
        path
        for path in result.output_dir.iterdir()
        if path.is_file() and path.name not in {"checksums.sha256", ".producer-complete"}
    )
    checksums_payload = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in checksum_targets
    ).encode("utf-8")
    (result.output_dir / "checksums.sha256").write_bytes(checksums_payload)
    new_instance = production_occupancy_publication_instance_digest(
        manifest_bytes=manifest_payload,
        checksums_bytes=checksums_payload,
    )
    marker = {
        "training_layout_version": "sop08_production_training_v1",
        "semantic_digest_sha256": manifest["semantic_digest_sha256"],
        "publication_instance_digest_sha256": new_instance,
        "training_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "checksums_sha256": hashlib.sha256(checksums_payload).hexdigest(),
    }
    (result.output_dir / ".producer-complete").write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest keys"):
        validate_production_occupancy_training_publication(
            result.output_dir,
            expected_publication_instance_digest_sha256=new_instance,
        )
