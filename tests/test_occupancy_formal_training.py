"""Formal validation-selected B3/B4 training contracts."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.training.occupancy_trainer import (
    FormalValidationSelection,
    ProductionOccupancyTrainingResult,
    load_formal_production_occupancy_checkpoint,
    train_production_occupancy_baselines,
    update_formal_validation_selection,
    validate_formal_occupancy_training_publication,
)
from src.datasets.risk_dataloader import select_production_risk_subset
from src.datasets.risk_dataset_seal import (
    load_risk_dataset_family,
    load_risk_dataset_seal,
    publish_risk_dataset_family,
    publish_risk_dataset_seal,
)
from src.datasets.risk_training_store import load_authenticated_occupancy_snapshot
from tests.fixtures.formal_risk_publication import (
    create_formal_risk_publication,
    create_formal_risk_sidecar_publication,
)


def _state(value: float) -> dict[str, torch.Tensor]:
    return {"weight": torch.tensor([value], dtype=torch.float32)}


def test_formal_selection_uses_lowest_finite_loss_and_earliest_tie() -> None:
    first = update_formal_validation_selection(
        None,
        phase="b3",
        epoch=1,
        optimizer_step=5,
        validation_loss=0.4,
        state_dict=_state(1.0),
    )
    tied = update_formal_validation_selection(
        first,
        phase="b3",
        epoch=2,
        optimizer_step=10,
        validation_loss=0.4,
        state_dict=_state(2.0),
    )
    better = update_formal_validation_selection(
        tied,
        phase="b3",
        epoch=3,
        optimizer_step=15,
        validation_loss=0.3,
        state_dict=_state(3.0),
    )

    assert isinstance(first, FormalValidationSelection)
    assert tied is first
    assert better.validation_loss == 0.3
    assert better.epoch == 3
    assert torch.equal(better.state_dict["weight"], torch.tensor([3.0]))
    with pytest.raises(ValueError, match="finite"):
        update_formal_validation_selection(
            better,
            phase="b3",
            epoch=4,
            optimizer_step=20,
            validation_loss=float("nan"),
            state_dict=_state(4.0),
        )


def test_formal_training_api_requires_validation_family_and_sidecars() -> None:
    signature = inspect.signature(train_production_occupancy_baselines)
    assert "validation_dataset" in signature.parameters
    assert "validation_sidecar_root" in signature.parameters
    assert "dataset_family" in signature.parameters
    assert "validation_snapshot" in signature.parameters


def test_formal_result_exposes_selected_and_final_checkpoints() -> None:
    assert [field.name for field in ProductionOccupancyTrainingResult.__dataclass_fields__.values()] == [
        "output_dir",
        "manifest_path",
        "metrics_path",
        "best_checkpoint",
        "final_checkpoint",
        "training_state_checkpoint",
        "semantic_digest_sha256",
        "publication_instance_digest_sha256",
    ]


def test_formal_occupancy_requires_family_val_and_exact_50k(tmp_path: Path) -> None:
    publication = create_formal_risk_publication(
        tmp_path / "upstream",
        history_steps=8,
        future_steps=15,
    )
    sidecars = create_formal_risk_sidecar_publication(
        publication,
        tmp_path / "sidecars",
    )
    seal = publish_risk_dataset_seal(
        tmp_path / "seal",
        collection_root=publication.collection_root,
        base_config_path=publication.base_config_path,
        split_provenance_path=publication.split_provenance_path,
        expected_split="train",
        expected_collection_handoff_sha256=publication.handoff_sha256,
        sidecar_root=sidecars.sidecar_root,
    )
    dataset = load_risk_dataset_seal(
        seal,
        collection_root=publication.collection_root,
        expected_split="train",
        sidecar_root=sidecars.sidecar_root,
    )
    subset = select_production_risk_subset(dataset, max_samples=12, seed=17)
    from tests.test_occupancy_production_trainer import _training_config

    with pytest.raises(
        ValueError,
        match="formal_50k.*(50,000|validation|family)",
    ):
        train_production_occupancy_baselines(
            train_dataset=dataset,
            train_subset=subset,
            sidecar_root=sidecars.sidecar_root,
            config=replace(_training_config(), stage="formal_50k"),
            output_dir=tmp_path / "formal",
            code_commit="a" * 40,
        )
    assert not (tmp_path / "formal").exists()


def _family_with_occupancy(root: Path):
    members = {}
    sidecar_roots = {}
    for split in ("train", "calibration", "val", "test"):
        publication = create_formal_risk_publication(
            root / f"upstream-{split}",
            split=split,
            handoff_dialect="legacy" if split == "train" else "heldout",
            history_steps=8,
            future_steps=15,
        )
        sidecar_root = None
        if split in {"train", "val"}:
            sidecars = create_formal_risk_sidecar_publication(
                publication,
                root / f"sidecars-{split}",
            )
            sidecar_root = sidecars.sidecar_root
            sidecar_roots[split] = sidecar_root
        seal = publish_risk_dataset_seal(
            root / f"seal-{split}",
            collection_root=publication.collection_root,
            base_config_path=publication.base_config_path,
            split_provenance_path=publication.split_provenance_path,
            expected_split=split,
            expected_collection_handoff_sha256=publication.handoff_sha256,
            sidecar_root=sidecar_root,
        )
        members[split] = load_risk_dataset_seal(
            seal,
            collection_root=publication.collection_root,
            expected_split=split,
            sidecar_root=sidecar_root,
        )
    family = load_risk_dataset_family(
        publish_risk_dataset_family(root / "family", members=members)
    )
    return family, members, sidecar_roots


def test_formal_b3_b4_selects_validation_checkpoints_and_freezes_b3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.training.occupancy_trainer as trainer_module
    from tests.test_occupancy_production_trainer import _training_config

    family, members, sidecar_roots = _family_with_occupancy(tmp_path)
    train_dataset, train_snapshot = load_authenticated_occupancy_snapshot(
        members["train"].seal_root,
        collection_root=members["train"].collection_root,
        sidecar_root=sidecar_roots["train"],
        expected_split="train",
        cache_root=tmp_path / "cache",
    )
    val_dataset, val_snapshot = load_authenticated_occupancy_snapshot(
        members["val"].seal_root,
        collection_root=members["val"].collection_root,
        sidecar_root=sidecar_roots["val"],
        expected_split="val",
        cache_root=tmp_path / "cache",
    )
    monkeypatch.setattr(trainer_module, "_FORMAL_TRAIN_SAMPLE_COUNT", 12)
    subset = train_snapshot.select_subset(max_samples=12, seed=17)
    config = replace(
        _training_config(),
        stage="formal_50k",
        batch_size=6,
        occupancy_epochs=2,
        aggregator_epochs=2,
    )

    result = train_production_occupancy_baselines(
        train_dataset=train_dataset,
        train_subset=subset,
        sidecar_root=sidecar_roots["train"],
        config=config,
        output_dir=tmp_path / "formal-training",
        code_commit="b" * 40,
        training_snapshot=train_snapshot,
        validation_dataset=val_dataset,
        validation_sidecar_root=sidecar_roots["val"],
        dataset_family=family,
        validation_snapshot=val_snapshot,
    )

    manifest = validate_formal_occupancy_training_publication(result.output_dir)
    assert result.best_checkpoint is not None
    assert result.training_state_checkpoint is not None
    checkpoint = load_formal_production_occupancy_checkpoint(
        result.best_checkpoint,
        expected_checkpoint_semantic_digest_sha256=manifest[
            "artifact_semantic_bindings"
        ]["best_checkpoint.pt"],
    )
    assert checkpoint["selection"]["b3"]["validation_loss"] == min(
        item["validation_loss"]
        for item in manifest["validation_history"]["b3"]
    )
    assert checkpoint["selection"]["b4"]["validation_loss"] == min(
        item["validation_loss"]
        for item in manifest["validation_history"]["b4"]
    )
    assert manifest["b3_frozen_during_b4"] is True
    assert manifest["b3_state_digest_before_b4_sha256"] == manifest[
        "b3_state_digest_after_b4_sha256"
    ]
    assert manifest["test_samples_used_for_training_or_selection"] == 0
