"""Production SOP08 sidecar join, query isolation, and training contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

import src.datasets.risk_dataloader as dataloader_module
from src.contracts import MODEL_INPUT_CLASSES, RiskSample
from src.datasets.risk_dataloader import (
    MODEL_INPUT_KEYS,
    TARGET_KEYS,
    OccupancyStreamCursor,
    ProductionOccupancyBatch,
    RiskDataContractError,
    iter_production_occupancy_batches,
    iter_production_risk_batches,
    production_endpoint_times_from_query_geometry,
    reconstruct_production_robot_endpoint_footprints,
    select_production_risk_subset,
)
from src.datasets.risk_dataset_seal import (
    RISK_SIDECAR_COLLECTION_LAYOUT_VERSION,
    SidecarShardDescriptor,
    load_occupancy_sidecar_collection,
    load_risk_dataset_seal,
    publish_risk_dataset_seal,
)
from src.datasets.shard_writer import load_risk_shard
from src.datasets.sidecar_writer import (
    load_risk_sidecar_shard,
    risk_sidecar_pair_completion_marker_path,
    write_risk_sidecar_pair_completion_marker,
    write_risk_sidecar_shard,
)
from src.generation.risk_sidecars import RiskLabelSidecar
from src.geometry import (
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint,
)
from src.planning.differential_drive import rollout_constant_control
from tests.fixtures.formal_risk_publication import (
    FormalRiskPublication,
    FormalRiskSidecarPublication,
    create_formal_risk_publication,
    create_formal_risk_sidecar_publication,
)


def _publication(
    root: Path,
) -> tuple[FormalRiskPublication, FormalRiskSidecarPublication]:
    risk = create_formal_risk_publication(
        root / "upstream",
        history_steps=8,
        future_steps=15,
    )
    sidecars = create_formal_risk_sidecar_publication(
        risk,
        root / "sidecars",
    )
    return risk, sidecars


def _seal(
    root: Path,
    publication: FormalRiskPublication,
    *,
    sidecar_root: Path | None,
):
    seal_root = publish_risk_dataset_seal(
        root,
        collection_root=publication.collection_root,
        base_config_path=publication.base_config_path,
        split_provenance_path=publication.split_provenance_path,
        expected_split="train",
        expected_collection_handoff_sha256=publication.handoff_sha256,
        sidecar_root=sidecar_root,
    )
    return load_risk_dataset_seal(
        seal_root,
        collection_root=publication.collection_root,
        expected_split="train",
        sidecar_root=sidecar_root,
    )


def _first_risk_sample(publication: FormalRiskPublication) -> RiskSample:
    return load_risk_shard(
        publication.collection_root / "shard-00000",
        grid=publication.grid,
    ).samples[0]


def _query_geometry(dataset) -> dict[str, object]:
    section = dataset.manifest["occupancy_sidecars"]
    assert isinstance(section, dict)
    geometry = section["query_geometry"]
    assert isinstance(geometry, dict)
    return geometry


def _expected_robot_masks(
    *,
    grid,
    v: float,
    omega: float,
) -> np.ndarray:
    footprint = inflate_footprint(RectangleFootprint(0.70, 0.55), 0.15)
    poses, _ = rollout_constant_control(
        v=v,
        omega=omega,
        dt_s=0.2,
        steps=15,
    )
    return np.stack(
        [
            rasterize_footprint(footprint, pose, grid).astype(np.float32)
            for pose in poses
        ],
        axis=0,
    )


def test_optional_sidecar_seal_is_complete_and_keeps_base_risk_digest(
    tmp_path: Path,
) -> None:
    publication, sidecars = _publication(tmp_path)
    risk_only = _seal(
        tmp_path / "risk-only-seal",
        publication,
        sidecar_root=None,
    )
    occupancy = _seal(
        tmp_path / "occupancy-seal",
        publication,
        sidecar_root=sidecars.sidecar_root,
    )

    assert RISK_SIDECAR_COLLECTION_LAYOUT_VERSION == (
        "risk_label_sidecar_collection_v1"
    )
    assert occupancy.risk_dataset_manifest_digest == (
        risk_only.risk_dataset_manifest_digest
    )
    section = occupancy.manifest["occupancy_sidecars"]
    assert section["base_risk_dataset_manifest_digest"] == (
        risk_only.risk_dataset_manifest_digest
    )
    assert section["sample_count"] == 12
    assert section["shard_count"] == 2
    assert len(section["collection_digest_sha256"]) == 64
    assert len(section["base_config_digest"]) == 32
    loaded_sidecars = load_occupancy_sidecar_collection(
        occupancy,
        sidecar_root=sidecars.sidecar_root,
    )
    assert loaded_sidecars.sample_count == 12
    assert [item.shard_index for item in loaded_sidecars.shards] == [0, 1]
    assert all(isinstance(item, SidecarShardDescriptor) for item in loaded_sidecars.shards)

    subset = select_production_risk_subset(risk_only, max_samples=12, seed=42)
    assert len(list(iter_production_risk_batches(
        risk_only,
        subset=subset,
        batch_size=5,
        seed=42,
        epoch=0,
    ))) == 4
    with pytest.raises(RiskDataContractError, match="occupancy_sidecars"):
        next(iter(iter_production_occupancy_batches(
            risk_only,
            sidecar_root=sidecars.sidecar_root,
            subset=subset,
            batch_size=5,
            seed=42,
            epoch=0,
        )))


@pytest.mark.parametrize(
    ("v", "omega"),
    ((0.0, 0.0), (0.6, 0.0), (0.5, 0.4), (-0.4, -0.4)),
)
def test_query_footprints_reconstruct_stop_straight_turn_and_reverse_exactly(
    tmp_path: Path,
    v: float,
    omega: float,
) -> None:
    publication, sidecars = _publication(tmp_path)
    dataset = _seal(
        tmp_path / "seal",
        publication,
        sidecar_root=sidecars.sidecar_root,
    )
    sample = _first_risk_sample(publication)
    metadata = dict(sample.metadata)
    provenance = dict(metadata["provenance"])
    provenance["trajectory_primitive"] = {
        "v_mps": v,
        "omega_radps": omega,
    }
    metadata["provenance"] = provenance
    candidate = replace(
        sample,
        metadata=metadata,
        trajectory_channels=np.full_like(sample.trajectory_channels, 123.0),
    )
    actual = reconstruct_production_robot_endpoint_footprints(
        candidate,
        grid=dataset.grid,
        query_geometry=_query_geometry(dataset),
    )
    expected = _expected_robot_masks(
        grid=dataset.grid,
        v=v,
        omega=omega,
    )
    assert actual.dtype == np.float32
    assert actual.shape == (15, dataset.grid.height, dataset.grid.width)
    assert np.array_equal(actual, expected)


def test_query_reconstruction_fails_closed_on_missing_or_mismatched_provenance(
    tmp_path: Path,
) -> None:
    publication, sidecars = _publication(tmp_path)
    dataset = _seal(
        tmp_path / "seal",
        publication,
        sidecar_root=sidecars.sidecar_root,
    )
    sample = _first_risk_sample(publication)
    geometry = _query_geometry(dataset)
    metadata = dict(sample.metadata)
    provenance = dict(metadata["provenance"])
    provenance.pop("trajectory_primitive")
    metadata["provenance"] = provenance
    with pytest.raises(RiskDataContractError, match="trajectory_primitive"):
        reconstruct_production_robot_endpoint_footprints(
            replace(sample, metadata=metadata),
            grid=dataset.grid,
            query_geometry=geometry,
        )

    metadata = dict(sample.metadata)
    provenance = dict(metadata["provenance"])
    provenance["base_config_digest"] = "f" * 32
    metadata["provenance"] = provenance
    with pytest.raises(RiskDataContractError, match="base_config_digest"):
        reconstruct_production_robot_endpoint_footprints(
            replace(sample, metadata=metadata),
            grid=dataset.grid,
            query_geometry=geometry,
        )

    wrong_footprint = dict(geometry)
    wrong_footprint["robot_length_m"] = 0.71
    with pytest.raises(RiskDataContractError, match="0.70x0.55"):
        reconstruct_production_robot_endpoint_footprints(
            sample,
            grid=dataset.grid,
            query_geometry=wrong_footprint,
        )


def test_occupancy_batch_namespaces_are_disjoint_and_resume_is_exact(
    tmp_path: Path,
) -> None:
    publication, sidecars = _publication(tmp_path)
    dataset = _seal(
        tmp_path / "seal",
        publication,
        sidecar_root=sidecars.sidecar_root,
    )
    subset = select_production_risk_subset(dataset, max_samples=12, seed=42)
    uninterrupted = list(iter_production_occupancy_batches(
        dataset,
        sidecar_root=sidecars.sidecar_root,
        subset=subset,
        batch_size=4,
        seed=42,
        epoch=0,
    ))
    batch, cursor = uninterrupted[0]
    assert isinstance(batch, ProductionOccupancyBatch)
    assert isinstance(cursor, OccupancyStreamCursor)
    assert cursor.sidecar_collection_digest_sha256 == (
        dataset.manifest["occupancy_sidecars"]["collection_digest_sha256"]
    )
    assert tuple(batch.model_inputs) == MODEL_INPUT_KEYS
    assert tuple(batch.targets) == TARGET_KEYS
    assert tuple(batch.query_inputs) == (
        "robot_endpoint_footprints",
        "endpoint_times_s",
    )
    assert tuple(batch.occupancy_targets) == ("hidden_risk_occupancy",)
    assert set(batch.model_inputs).isdisjoint(batch.query_inputs)
    assert set(batch.model_inputs).isdisjoint(batch.occupancy_targets)
    assert not hasattr(batch, "label_sidecars")
    assert batch.query_inputs["robot_endpoint_footprints"].shape == (
        4,
        15,
        dataset.grid.height,
        dataset.grid.width,
    )
    assert batch.occupancy_targets["hidden_risk_occupancy"].shape == (
        4,
        15,
        dataset.grid.height,
        dataset.grid.width,
    )
    for mapping in (
        batch.model_inputs,
        batch.targets,
        batch.query_inputs,
        batch.occupancy_targets,
    ):
        assert all(value.dtype == torch.float32 for value in mapping.values())
        assert all(torch.isfinite(value).all() for value in mapping.values())
    assert torch.all(
        (batch.query_inputs["robot_endpoint_footprints"] == 0.0)
        | (batch.query_inputs["robot_endpoint_footprints"] == 1.0)
    )
    assert torch.equal(
        batch.query_inputs["endpoint_times_s"],
        torch.arange(1, 16, dtype=torch.float32) * 0.2,
    )
    resumed = list(iter_production_occupancy_batches(
        dataset,
        sidecar_root=sidecars.sidecar_root,
        subset=subset,
        batch_size=4,
        seed=42,
        epoch=0,
        start_cursor=cursor,
    ))
    assert [sample_id for current, _ in uninterrupted for sample_id in current.sample_ids] == [
        *batch.sample_ids,
        *(sample_id for current, _ in resumed for sample_id in current.sample_ids),
    ]
    with pytest.raises(RiskDataContractError, match="sidecar.*digest"):
        next(iter(iter_production_occupancy_batches(
            dataset,
            sidecar_root=sidecars.sidecar_root,
            subset=subset,
            batch_size=4,
            seed=42,
            epoch=0,
            start_cursor=replace(
                cursor,
                sidecar_collection_digest_sha256="f" * 64,
            ),
        )))
    with pytest.raises(FrozenInstanceError):
        batch.split = "test"  # type: ignore[misc]


def test_re_signed_wrong_robot_masks_fail_before_first_batch(
    tmp_path: Path,
) -> None:
    publication, sidecars = _publication(tmp_path)
    dataset = _seal(
        tmp_path / "seal",
        publication,
        sidecar_root=sidecars.sidecar_root,
    )
    risk_root = publication.collection_root / "shard-00000"
    sidecar_root = sidecars.sidecar_root / "shard-00000"
    marker = risk_sidecar_pair_completion_marker_path(sidecar_root)
    risk = load_risk_shard(risk_root, grid=publication.grid)
    loaded = load_risk_sidecar_shard(
        sidecar_root,
        grid=publication.grid,
        expected_sample_ids=tuple(sample.sample_id for sample in risk.samples),
        expected_source_risk_shard_semantic_digest=risk.semantic_digest,
    )
    rewritten = tuple(
        RiskLabelSidecar(
            sample_id=sample_id,
            hidden_risk_occupancy=loaded.hidden_risk_occupancy[index].astype(np.uint8),
            robot_future_footprints=np.zeros_like(
                loaded.robot_future_footprints[index], dtype=np.uint8
            ),
            future_endpoint_times_s=loaded.future_endpoint_times_s,
        )
        for index, sample_id in enumerate(loaded.sample_ids)
    )
    shutil.rmtree(sidecar_root)
    marker.unlink()
    write_risk_sidecar_shard(
        rewritten,
        sidecar_root,
        grid=publication.grid,
        split="train",
        shard_index=0,
        source_risk_shard_semantic_digest=risk.semantic_digest,
    )
    rewritten_loaded = load_risk_sidecar_shard(
        sidecar_root,
        grid=publication.grid,
        expected_sample_ids=loaded.sample_ids,
        expected_source_risk_shard_semantic_digest=risk.semantic_digest,
    )
    write_risk_sidecar_pair_completion_marker(
        marker,
        risk_root=risk_root,
        sidecar_root=sidecar_root,
        split="train",
        shard_index=0,
        sample_ids=loaded.sample_ids,
        risk_shard_semantic_digest=risk.semantic_digest,
        sidecar_shard_semantic_digest=rewritten_loaded.semantic_digest,
    )
    subset = select_production_risk_subset(dataset, max_samples=12, seed=42)
    stream = iter_production_occupancy_batches(
        dataset,
        sidecar_root=sidecars.sidecar_root,
        subset=subset,
        batch_size=4,
        seed=42,
        epoch=0,
    )
    with pytest.raises(RiskDataContractError):
        next(iter(stream))


def test_no_occupancy_oracle_namespace_is_a_model_input_contract() -> None:
    assert all(candidate is not ProductionOccupancyBatch for candidate in MODEL_INPUT_CLASSES)
    assert [field.name for field in fields(ProductionOccupancyBatch)] == [
        "model_inputs",
        "targets",
        "query_inputs",
        "occupancy_targets",
        "sample_ids",
        "split",
        "provenance",
    ]


def test_query_endpoint_times_are_independently_derived_and_sidecar_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, sidecars = _publication(tmp_path)
    dataset = _seal(
        tmp_path / "seal",
        publication,
        sidecar_root=sidecars.sidecar_root,
    )
    geometry = _query_geometry(dataset)
    expected = production_endpoint_times_from_query_geometry(geometry)
    assert expected.dtype == np.float32
    assert expected.shape == (15,)
    assert np.array_equal(
        expected,
        np.arange(1, 16, dtype=np.float32) * np.float32(0.2),
    )

    subset = select_production_risk_subset(dataset, max_samples=12, seed=42)
    original = production_endpoint_times_from_query_geometry
    calls = 0

    def observed(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(
        dataloader_module,
        "production_endpoint_times_from_query_geometry",
        observed,
    )
    batch, _ = next(iter(iter_production_occupancy_batches(
        dataset,
        sidecar_root=sidecars.sidecar_root,
        subset=subset,
        batch_size=4,
        seed=42,
        epoch=0,
    )))
    assert calls > 0
    assert np.array_equal(
        batch.query_inputs["endpoint_times_s"].numpy(),
        expected,
    )

    monkeypatch.setattr(
        dataloader_module,
        "production_endpoint_times_from_query_geometry",
        lambda value: original(value) + np.float32(0.01),
    )
    with pytest.raises(RiskDataContractError, match="endpoint.*sidecar"):
        next(iter(iter_production_occupancy_batches(
            dataset,
            sidecar_root=sidecars.sidecar_root,
            subset=subset,
            batch_size=4,
            seed=42,
            epoch=0,
        )))


def test_re_signed_hidden_sidecar_swap_after_first_yield_fails_next_and_resume(
    tmp_path: Path,
) -> None:
    publication, sidecars = _publication(tmp_path)
    dataset = _seal(
        tmp_path / "seal",
        publication,
        sidecar_root=sidecars.sidecar_root,
    )
    subset = select_production_risk_subset(dataset, max_samples=12, seed=42)
    stream = iter(iter_production_occupancy_batches(
        dataset,
        sidecar_root=sidecars.sidecar_root,
        subset=subset,
        batch_size=4,
        seed=42,
        epoch=0,
    ))
    first_batch, first_cursor = next(stream)
    first_id = first_batch.sample_ids[0]
    shard_index = next(
        index
        for index in range(2)
        if first_id in {
            sample.sample_id
            for sample in load_risk_shard(
                publication.collection_root / f"shard-{index:05d}",
                grid=publication.grid,
            ).samples
        }
    )
    risk_root = publication.collection_root / f"shard-{shard_index:05d}"
    final_root = sidecars.sidecar_root / f"shard-{shard_index:05d}"
    final_marker = risk_sidecar_pair_completion_marker_path(final_root)
    loaded_risk = load_risk_shard(risk_root, grid=publication.grid)
    loaded_sidecar = load_risk_sidecar_shard(
        final_root,
        grid=publication.grid,
        expected_sample_ids=tuple(sample.sample_id for sample in loaded_risk.samples),
        expected_source_risk_shard_semantic_digest=loaded_risk.semantic_digest,
    )
    replacement = tuple(
        RiskLabelSidecar(
            sample_id=sample_id,
            hidden_risk_occupancy=(
                1 - loaded_sidecar.hidden_risk_occupancy[index]
            ).astype(np.uint8),
            robot_future_footprints=loaded_sidecar.robot_future_footprints[
                index
            ].astype(np.uint8),
            future_endpoint_times_s=loaded_sidecar.future_endpoint_times_s,
        )
        for index, sample_id in enumerate(loaded_sidecar.sample_ids)
    )
    staged_root = tmp_path / "replacement-sidecar-shard"
    write_risk_sidecar_shard(
        replacement,
        staged_root,
        grid=publication.grid,
        split="train",
        shard_index=shard_index,
        source_risk_shard_semantic_digest=loaded_risk.semantic_digest,
    )
    staged_loaded = load_risk_sidecar_shard(
        staged_root,
        grid=publication.grid,
        expected_sample_ids=loaded_sidecar.sample_ids,
        expected_source_risk_shard_semantic_digest=loaded_risk.semantic_digest,
    )
    staged_marker = tmp_path / "replacement-pair-marker.json"
    write_risk_sidecar_pair_completion_marker(
        staged_marker,
        risk_root=risk_root,
        sidecar_root=final_root,
        split="train",
        shard_index=shard_index,
        sample_ids=loaded_sidecar.sample_ids,
        risk_shard_semantic_digest=loaded_risk.semantic_digest,
        sidecar_shard_semantic_digest=staged_loaded.semantic_digest,
    )
    os.rename(final_root, tmp_path / "retired-sidecar-shard")
    os.rename(staged_root, final_root)
    os.rename(final_marker, tmp_path / "retired-pair-marker.json")
    os.rename(staged_marker, final_marker)

    with pytest.raises(RiskDataContractError, match="semantic digest.*seal"):
        next(stream)
    with pytest.raises(RiskDataContractError):
        next(iter(iter_production_occupancy_batches(
            dataset,
            sidecar_root=sidecars.sidecar_root,
            subset=subset,
            batch_size=4,
            seed=42,
            epoch=0,
            start_cursor=first_cursor,
        )))
