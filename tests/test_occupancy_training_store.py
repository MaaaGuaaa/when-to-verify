from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import src.datasets.risk_dataloader as risk_dataloader_module
import src.datasets.risk_dataset_seal as risk_dataset_seal_module
import src.datasets.risk_training_store as risk_training_store_module
from src.datasets.risk_dataloader import (
    MODEL_INPUT_KEYS,
    TARGET_KEYS,
    ProductionOccupancyBatch,
    RiskDataContractError,
    iter_production_occupancy_batches,
    select_production_risk_subset,
)
from tests.test_occupancy_production_training import _publication, _seal


def _assert_occupancy_batches_equal(
    actual: ProductionOccupancyBatch,
    expected: ProductionOccupancyBatch,
) -> None:
    assert actual.sample_ids == expected.sample_ids
    assert actual.split == expected.split
    for actual_values, expected_values in (
        (actual.model_inputs, expected.model_inputs),
        (actual.targets, expected.targets),
        (actual.query_inputs, expected.query_inputs),
        (actual.occupancy_targets, expected.occupancy_targets),
    ):
        assert tuple(actual_values) == tuple(expected_values)
        for name in actual_values:
            torch.testing.assert_close(actual_values[name], expected_values[name])


def _published_occupancy_dataset(tmp_path: Path):
    publication, sidecars = _publication(tmp_path / "source")
    dataset = _seal(
        tmp_path / "seal",
        publication,
        sidecar_root=sidecars.sidecar_root,
    )
    return publication, sidecars, dataset


def test_authenticated_occupancy_snapshot_uses_each_integrated_decode_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, sidecars, prior_dataset = _published_occupancy_dataset(tmp_path)
    original_risk_loader = risk_dataset_seal_module.load_risk_shard
    original_sidecar_loader = risk_dataset_seal_module.load_risk_sidecar_shard
    original_stream_loader = risk_dataloader_module.load_risk_shard
    original_reconstruct = (
        risk_dataloader_module.reconstruct_production_robot_endpoint_footprints
    )
    decoded_risk_ids: list[int] = []
    consumed_risk_ids: list[int] = []
    risk_load_count = 0
    sidecar_load_count = 0
    stream_load_count = 0
    reconstruction_count = 0

    def counted_risk_loader(*args, **kwargs):
        nonlocal risk_load_count
        risk_load_count += 1
        loaded = original_risk_loader(*args, **kwargs)
        decoded_risk_ids.append(id(loaded))
        return loaded

    def counted_sidecar_loader(*args, **kwargs):
        nonlocal sidecar_load_count
        sidecar_load_count += 1
        return original_sidecar_loader(*args, **kwargs)

    def counted_stream_loader(*args, **kwargs):
        nonlocal stream_load_count
        stream_load_count += 1
        return original_stream_loader(*args, **kwargs)

    def counted_reconstruct(*args, **kwargs):
        nonlocal reconstruction_count
        reconstruction_count += 1
        return original_reconstruct(*args, **kwargs)

    monkeypatch.setattr(
        risk_dataset_seal_module,
        "load_risk_shard",
        counted_risk_loader,
    )
    monkeypatch.setattr(
        risk_dataset_seal_module,
        "load_risk_sidecar_shard",
        counted_sidecar_loader,
    )
    monkeypatch.setattr(
        risk_dataloader_module,
        "load_risk_shard",
        counted_stream_loader,
    )
    monkeypatch.setattr(
        risk_dataloader_module,
        "reconstruct_production_robot_endpoint_footprints",
        counted_reconstruct,
    )
    materializer_type = getattr(
        risk_training_store_module,
        "_OccupancySnapshotMaterializer",
        None,
    )
    if materializer_type is not None:
        original_consume = materializer_type.consume

        def observed_consume(self, pair):
            consumed_risk_ids.append(id(pair.risk_shard))
            return original_consume(self, pair)

        monkeypatch.setattr(materializer_type, "consume", observed_consume)

    dataset, snapshot = (
        risk_training_store_module.load_authenticated_occupancy_snapshot(
            prior_dataset.seal_root,
            collection_root=publication.collection_root,
            sidecar_root=sidecars.sidecar_root,
            expected_split="train",
            cache_root=tmp_path / "cache",
        )
    )

    assert dataset.risk_dataset_manifest_digest == (
        prior_dataset.risk_dataset_manifest_digest
    )
    assert snapshot.risk_snapshot.sample_ids == snapshot.sample_ids
    assert risk_load_count == len(dataset.shards)
    assert sidecar_load_count == len(dataset.shards)
    assert stream_load_count == 0
    assert reconstruction_count == dataset.sample_count
    assert consumed_risk_ids == decoded_risk_ids


def test_authenticated_occupancy_snapshot_isolates_namespaces_and_cache_layout(
    tmp_path: Path,
) -> None:
    publication, sidecars, dataset = _published_occupancy_dataset(tmp_path)
    subset = select_production_risk_subset(dataset, max_samples=12, seed=42)
    strict_batch, _ = next(
        iter(
            iter_production_occupancy_batches(
                dataset,
                sidecar_root=sidecars.sidecar_root,
                subset=subset,
                batch_size=4,
                seed=42,
                epoch=0,
            )
        )
    )
    _, snapshot = risk_training_store_module.load_authenticated_occupancy_snapshot(
        dataset.seal_root,
        collection_root=publication.collection_root,
        sidecar_root=sidecars.sidecar_root,
        expected_split="train",
        cache_root=tmp_path / "cache",
    )

    batch = snapshot.batch(strict_batch.sample_ids)

    assert tuple(batch.model_inputs) == MODEL_INPUT_KEYS
    assert tuple(batch.targets) == TARGET_KEYS
    assert tuple(batch.query_inputs) == (
        "robot_endpoint_footprints",
        "endpoint_times_s",
    )
    assert tuple(batch.occupancy_targets) == ("hidden_risk_occupancy",)
    assert "hidden_risk_occupancy" not in batch.model_inputs
    assert "hidden_risk_occupancy" not in snapshot.risk_snapshot.targets
    _assert_occupancy_batches_equal(batch, strict_batch)
    assert (snapshot.root / "query_inputs" / "robot_endpoint_footprints.npy").is_file()
    assert (snapshot.root / "query_inputs" / "endpoint_times_s.npy").is_file()
    assert (snapshot.root / "targets" / "hidden_risk_occupancy.npy").is_file()
    assert not (snapshot.root / "robot_future_footprints.npy").exists()
    assert not (snapshot.root / "hidden_risk_occupancy.npy").exists()

    manifest = json.loads(
        (snapshot.root / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    identity = manifest["identity"]
    assert identity["risk_snapshot_digest_sha256"] == (
        snapshot.risk_snapshot.snapshot_digest_sha256
    )
    assert identity["risk_dataset_manifest_digest"] == (
        dataset.risk_dataset_manifest_digest
    )
    assert identity["split"] == "train"
    assert identity["occupancy_sidecar_collection_digest_sha256"] == (
        dataset.manifest["occupancy_sidecars"]["collection_digest_sha256"]
    )
    assert len(identity["ordered_sidecar_semantic_digests"]) == len(dataset.shards)
    assert len(identity["pair_marker_digests_sha256"]) == len(dataset.shards)
    assert set(identity["arrays"]) == {
        "query_inputs/robot_endpoint_footprints.npy",
        "query_inputs/endpoint_times_s.npy",
        "targets/hidden_risk_occupancy.npy",
    }
    assert str((tmp_path / "cache").absolute()) not in json.dumps(
        identity,
        sort_keys=True,
    )


def test_authenticated_occupancy_snapshot_matches_two_strict_epochs_without_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, sidecars, dataset = _published_occupancy_dataset(tmp_path)
    subset = select_production_risk_subset(dataset, max_samples=11, seed=17)
    strict_epochs = [
        list(
            iter_production_occupancy_batches(
                dataset,
                sidecar_root=sidecars.sidecar_root,
                subset=subset,
                batch_size=4,
                seed=17,
                epoch=epoch,
            )
        )
        for epoch in range(2)
    ]
    _, snapshot = risk_training_store_module.load_authenticated_occupancy_snapshot(
        dataset.seal_root,
        collection_root=publication.collection_root,
        sidecar_root=sidecars.sidecar_root,
        expected_split="train",
        cache_root=tmp_path / "cache",
    )
    snapshot_subset = snapshot.select_subset(max_samples=11, seed=17)
    assert snapshot_subset == subset

    def forbidden_source_load(*args, **kwargs):
        raise AssertionError("snapshot replay must not access source shards")

    monkeypatch.setattr(
        risk_dataset_seal_module,
        "load_risk_shard",
        forbidden_source_load,
    )
    monkeypatch.setattr(
        risk_dataset_seal_module,
        "load_risk_sidecar_shard",
        forbidden_source_load,
    )
    monkeypatch.setattr(
        risk_dataloader_module,
        "load_risk_shard",
        forbidden_source_load,
    )
    monkeypatch.setattr(
        risk_dataloader_module,
        "_load_validated_production_shard",
        forbidden_source_load,
    )

    for epoch, strict_batches in enumerate(strict_epochs):
        snapshot_batches = list(
            snapshot.iter_batches(
                subset=snapshot_subset,
                batch_size=4,
                seed=17,
                epoch=epoch,
            )
        )
        assert [cursor for _, cursor in snapshot_batches] == [
            cursor for _, cursor in strict_batches
        ]
        for (actual, _), (expected, _) in zip(snapshot_batches, strict_batches):
            _assert_occupancy_batches_equal(actual, expected)

    replayed = list(
        snapshot.iter_batches(
            subset=snapshot_subset,
            batch_size=4,
            seed=17,
            epoch=0,
        )
    )
    assert [batch.sample_ids for batch, _ in replayed] == [
        batch.sample_ids for batch, _ in strict_epochs[0]
    ]


def test_authenticated_occupancy_cache_reopens_without_source_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, sidecars, dataset = _published_occupancy_dataset(tmp_path)
    cache_root = tmp_path / "cache"
    _, snapshot = risk_training_store_module.load_authenticated_occupancy_snapshot(
        dataset.seal_root,
        collection_root=publication.collection_root,
        sidecar_root=sidecars.sidecar_root,
        expected_split="train",
        cache_root=cache_root,
    )

    def forbidden_source_load(*args, **kwargs):
        raise AssertionError("cache reopen must not access source shards")

    monkeypatch.setattr(
        risk_dataset_seal_module,
        "load_risk_shard",
        forbidden_source_load,
    )
    monkeypatch.setattr(
        risk_dataset_seal_module,
        "load_risk_sidecar_shard",
        forbidden_source_load,
    )
    monkeypatch.setattr(
        risk_dataloader_module,
        "_load_validated_production_shard",
        forbidden_source_load,
    )
    reopened = risk_training_store_module.open_authenticated_occupancy_snapshot(
        dataset,
        cache_root=cache_root,
    )
    assert reopened.snapshot_digest_sha256 == snapshot.snapshot_digest_sha256
    np.testing.assert_array_equal(
        reopened.batch(reopened.sample_ids[:2])
        .occupancy_targets["hidden_risk_occupancy"]
        .numpy(),
        snapshot.batch(snapshot.sample_ids[:2])
        .occupancy_targets["hidden_risk_occupancy"]
        .numpy(),
    )

    unknown = snapshot.root / "orphan.npy"
    unknown.write_bytes(b"partial")
    with pytest.raises(RiskDataContractError, match="unexpected"):
        risk_training_store_module.open_authenticated_occupancy_snapshot(
            dataset,
            cache_root=cache_root,
        )
    unknown.unlink()

    marker = snapshot.root / ".snapshot-complete"
    saved_marker = tmp_path / "saved-complete-marker"
    marker.rename(saved_marker)
    with pytest.raises(RiskDataContractError, match="missing"):
        risk_training_store_module.open_authenticated_occupancy_snapshot(
            dataset,
            cache_root=cache_root,
        )
    saved_marker.rename(marker)

    query_masks = snapshot.root / "query_inputs" / "robot_endpoint_footprints.npy"
    saved_query_masks = tmp_path / "saved-robot-endpoint-footprints.npy"
    query_masks.rename(saved_query_masks)
    query_masks.symlink_to(saved_query_masks)
    with pytest.raises(RiskDataContractError, match="regular|symlink"):
        risk_training_store_module.open_authenticated_occupancy_snapshot(
            dataset,
            cache_root=cache_root,
        )
    query_masks.unlink()
    saved_query_masks.rename(query_masks)

    hidden = snapshot.root / "targets" / "hidden_risk_occupancy.npy"
    original_hidden = hidden.read_bytes()
    corrupted_hidden = bytearray(original_hidden)
    corrupted_hidden[-1] ^= 1
    hidden.write_bytes(corrupted_hidden)
    with pytest.raises(RiskDataContractError, match="checksum"):
        risk_training_store_module.open_authenticated_occupancy_snapshot(
            dataset,
            cache_root=cache_root,
        )
    hidden.write_bytes(original_hidden)

    manifest_path = snapshot.root / "snapshot_manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    manifest["snapshot_digest_sha256"] = "0" * 64
    manifest_path.write_bytes(
        risk_training_store_module._canonical_json_bytes(manifest)
    )
    with pytest.raises(RiskDataContractError, match="digest|marker"):
        risk_training_store_module.open_authenticated_occupancy_snapshot(
            dataset,
            cache_root=cache_root,
        )
    manifest_path.write_bytes(original_manifest)

    marker_path = snapshot.root / ".snapshot-complete"
    original_marker = marker_path.read_bytes()
    marker_payload = json.loads(original_marker)
    marker_payload["layout_version"] = "unknown_occupancy_snapshot"
    marker_path.write_bytes(
        risk_training_store_module._canonical_json_bytes(marker_payload)
    )
    with pytest.raises(RiskDataContractError, match="marker|version"):
        risk_training_store_module.open_authenticated_occupancy_snapshot(
            dataset,
            cache_root=cache_root,
        )
    marker_path.write_bytes(original_marker)


def test_existing_risk_and_occupancy_winners_use_shared_full_identity_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, sidecars, dataset = _published_occupancy_dataset(tmp_path)
    cache_root = tmp_path / "cache"
    risk_training_store_module.load_authenticated_occupancy_snapshot(
        dataset.seal_root,
        collection_root=publication.collection_root,
        sidecar_root=sidecars.sidecar_root,
        expected_split="train",
        cache_root=cache_root,
    )
    helper = getattr(
        risk_training_store_module,
        "_discard_staging_if_matching_winner",
        None,
    )
    assert callable(helper)
    compared_staging_names: list[str] = []

    def observed_helper(staging: Path, winner: Path) -> None:
        compared_staging_names.append(staging.name)
        helper(staging, winner)

    monkeypatch.setattr(
        risk_training_store_module,
        "_discard_staging_if_matching_winner",
        observed_helper,
    )
    risk_training_store_module.load_authenticated_occupancy_snapshot(
        dataset.seal_root,
        collection_root=publication.collection_root,
        sidecar_root=sidecars.sidecar_root,
        expected_split="train",
        cache_root=cache_root,
    )

    assert any(name.startswith(".risk-snapshot.staging-") for name in compared_staging_names)
    assert any(
        name.startswith(".occupancy-snapshot.staging-")
        for name in compared_staging_names
    )
