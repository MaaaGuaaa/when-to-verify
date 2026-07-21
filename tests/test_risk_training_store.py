from __future__ import annotations

import json
import inspect
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

import src.datasets.risk_dataloader as risk_dataloader_module
import src.datasets.risk_dataset_seal as risk_dataset_seal_module
import src.datasets.risk_training_store as risk_training_store_module
from src.datasets.risk_dataloader import (
    MODEL_INPUT_KEYS,
    TARGET_KEYS,
    RiskDataContractError,
)
from src.datasets.risk_training_store import (
    build_authenticated_risk_training_view,
    load_authenticated_risk_snapshot,
    open_authenticated_risk_snapshot,
    open_authenticated_risk_snapshot_descriptor,
)
from src.datasets.risk_dataloader import select_production_risk_subset
from tests.test_risk_production_training import _publish_and_load


def _read_snapshot_manifest(root: Path) -> dict[str, object]:
    return json.loads(
        (root / "snapshot_manifest.json").read_text(encoding="utf-8")
    )


def _write_snapshot_json(path: Path, value: object) -> None:
    path.write_bytes(risk_training_store_module._canonical_json_bytes(value))


def _risk_array_entries(
    manifest: dict[str, object],
) -> dict[str, dict[str, object]]:
    identity = manifest["identity"]
    assert isinstance(identity, dict)
    entries = identity.get("arrays", manifest.get("arrays"))
    assert isinstance(entries, dict)
    return entries


def _risk_shard_index_entry(manifest: dict[str, object]) -> dict[str, object]:
    identity = manifest["identity"]
    assert isinstance(identity, dict)
    entry = identity.get(
        "sample_shard_indices",
        manifest.get("sample_shard_indices"),
    )
    assert isinstance(entry, dict)
    return entry


def _rewrite_risk_manifest(
    snapshot,
    manifest: dict[str, object],
    *,
    descriptor: dict[str, object] | None = None,
) -> dict[str, object] | None:
    identity = manifest["identity"]
    assert isinstance(identity, dict)
    digest = risk_training_store_module._sha256_bytes(
        risk_training_store_module._canonical_json_bytes(identity)
    )
    manifest["snapshot_digest_sha256"] = digest
    manifest_path = snapshot.root / "snapshot_manifest.json"
    _write_snapshot_json(manifest_path, manifest)
    manifest_sha256 = risk_training_store_module._sha256_file(manifest_path)
    if "snapshot_manifest_layout_version" in manifest:
        marker = {
            "layout_version": manifest["snapshot_manifest_layout_version"],
            "snapshot_digest_sha256": digest,
            "snapshot_manifest_sha256": manifest_sha256,
        }
    else:
        marker = {"snapshot_digest_sha256": digest}
    _write_snapshot_json(snapshot.root / ".snapshot-complete", marker)
    if descriptor is None:
        return None
    rewritten = json.loads(json.dumps(descriptor))
    rewritten["snapshot_digest_sha256"] = digest
    rewritten["snapshot_manifest_sha256"] = manifest_sha256
    return rewritten


def test_authenticated_snapshot_round_trips_strict_risk_tensors(tmp_path):
    _, dataset = _publish_and_load(tmp_path / "source")
    snapshot = open_authenticated_risk_snapshot(dataset, cache_root=tmp_path / "cache")

    assert len(snapshot.sample_ids) == dataset.sample_count
    batch = snapshot.batch(snapshot.sample_ids[:3])
    assert batch.provenance["loader_mode"] == "authenticated_snapshot"
    assert batch.model_inputs["bev_history"].shape[0] == 3
    reopened = open_authenticated_risk_snapshot(dataset, cache_root=tmp_path / "cache")
    assert reopened.snapshot_digest_sha256 == snapshot.snapshot_digest_sha256
    np.testing.assert_array_equal(
        reopened.batch(snapshot.sample_ids[:3]).targets["collision_label"].numpy(),
        batch.targets["collision_label"].numpy(),
    )


def test_authenticated_snapshot_rejects_partial_cache(tmp_path):
    _, dataset = _publish_and_load(tmp_path / "source")
    snapshot = open_authenticated_risk_snapshot(dataset, cache_root=tmp_path / "cache")
    (snapshot.root / ".snapshot-complete").unlink()

    with pytest.raises(RiskDataContractError, match="snapshot file is missing"):
        open_authenticated_risk_snapshot(dataset, cache_root=tmp_path / "cache")


def test_authenticated_snapshot_decodes_once_and_replays_strict_epochs(
    tmp_path, monkeypatch
):
    _, dataset = _publish_and_load(tmp_path / "source")
    subset = select_production_risk_subset(dataset, max_samples=9, seed=17)
    strict_epochs = [
        list(
            risk_dataloader_module.iter_production_risk_batches(
                dataset,
                subset=subset,
                batch_size=4,
                seed=17,
                epoch=epoch,
            )
        )
        for epoch in range(2)
    ]
    original_loader = risk_dataloader_module._load_validated_production_shard
    load_count = 0

    def counted_loader(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(
        risk_dataloader_module,
        "_load_validated_production_shard",
        counted_loader,
    )

    snapshot = open_authenticated_risk_snapshot(
        dataset,
        cache_root=tmp_path / "cache",
    )

    assert load_count == len(dataset.shards)
    snapshot_subset = snapshot.select_subset(max_samples=9, seed=17)
    assert snapshot_subset == subset
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
        assert [batch.sample_ids for batch, _ in snapshot_batches] == [
            batch.sample_ids for batch, _ in strict_batches
        ]
        for (actual, _), (expected, _) in zip(snapshot_batches, strict_batches):
            for name in actual.model_inputs:
                torch.testing.assert_close(
                    actual.model_inputs[name], expected.model_inputs[name]
                )
            for name in actual.targets:
                torch.testing.assert_close(actual.targets[name], expected.targets[name])

    reopened = open_authenticated_risk_snapshot(
        dataset,
        cache_root=tmp_path / "cache",
    )
    assert reopened.snapshot_digest_sha256 == snapshot.snapshot_digest_sha256
    assert load_count == len(dataset.shards)


def test_authenticated_snapshot_materializes_during_single_seal_pass(
    tmp_path, monkeypatch
):
    publication, prior_dataset = _publish_and_load(tmp_path / "source")
    original_seal_loader = risk_dataset_seal_module.load_risk_shard
    original_stream_loader = risk_dataloader_module.load_risk_shard
    seal_load_count = 0
    stream_load_count = 0

    def counted_seal_loader(*args, **kwargs):
        nonlocal seal_load_count
        seal_load_count += 1
        return original_seal_loader(*args, **kwargs)

    def counted_stream_loader(*args, **kwargs):
        nonlocal stream_load_count
        stream_load_count += 1
        return original_stream_loader(*args, **kwargs)

    monkeypatch.setattr(
        risk_dataset_seal_module,
        "load_risk_shard",
        counted_seal_loader,
    )
    monkeypatch.setattr(
        risk_dataloader_module,
        "load_risk_shard",
        counted_stream_loader,
    )

    dataset, snapshot = load_authenticated_risk_snapshot(
        prior_dataset.seal_root,
        collection_root=publication.collection_root,
        expected_split="train",
        cache_root=tmp_path / "cache",
    )

    assert dataset.risk_dataset_manifest_digest == prior_dataset.risk_dataset_manifest_digest
    assert len(snapshot.sample_ids) == dataset.sample_count
    assert seal_load_count == len(dataset.shards)
    assert stream_load_count == 0


def test_snapshot_descriptor_opens_without_source_or_bulk_rehash(tmp_path, monkeypatch):
    _, dataset = _publish_and_load(tmp_path / "source")
    snapshot = open_authenticated_risk_snapshot(dataset, cache_root=tmp_path / "cache")
    descriptor = snapshot.descriptor()

    def forbidden_source_load(*args, **kwargs):
        raise AssertionError("peer rank must not load source shards")

    def forbidden_bulk_hash(*args, **kwargs):
        raise AssertionError("peer rank must not hash bulk snapshot arrays")

    monkeypatch.setattr(
        risk_dataloader_module,
        "_load_validated_production_shard",
        forbidden_source_load,
    )
    monkeypatch.setattr(
        "src.datasets.risk_training_store._sha256_file",
        forbidden_bulk_hash,
    )

    peer = open_authenticated_risk_snapshot_descriptor(descriptor)

    assert peer.snapshot_digest_sha256 == snapshot.snapshot_digest_sha256
    assert peer.sample_ids == snapshot.sample_ids
    np.testing.assert_array_equal(
        peer.batch(peer.sample_ids[:2]).targets["risk_severity"].numpy(),
        snapshot.batch(snapshot.sample_ids[:2]).targets["risk_severity"].numpy(),
    )

    tampered = {**descriptor, "snapshot_digest_sha256": "0" * 64}
    with pytest.raises(RiskDataContractError, match="descriptor digest"):
        open_authenticated_risk_snapshot_descriptor(tampered)


def test_training_view_binds_partition_inputs_and_exact_epoch_order(tmp_path):
    _, dataset = _publish_and_load(tmp_path / "source")
    snapshot = open_authenticated_risk_snapshot(dataset, cache_root=tmp_path / "cache")
    subset = snapshot.select_subset(max_samples=11, seed=23)
    view = build_authenticated_risk_training_view(
        snapshot,
        subset=subset,
        split_role="train",
        world_size=3,
        batch_size=2,
        gradient_accumulation_steps=2,
    )

    epoch_zero = view.partition(epoch=0)
    epoch_one = view.partition(epoch=1)
    strict_zero = [
        sample_id
        for batch, _ in snapshot.iter_batches(
            subset=subset,
            batch_size=4,
            seed=23,
            epoch=0,
        )
        for sample_id in batch.sample_ids
    ]
    partition_zero = [
        sample_id
        for rank_batches in epoch_zero.rank_microbatches
        for batch_ids in rank_batches
        for sample_id in batch_ids
    ]

    assert partition_zero == strict_zero
    assert epoch_zero.partition_spec_digest_sha256 == (
        epoch_one.partition_spec_digest_sha256
    )
    assert epoch_zero.epoch_plan_digest_sha256 != epoch_one.epoch_plan_digest_sha256
    assert view.training_view_digest_sha256 == build_authenticated_risk_training_view(
        snapshot,
        subset=subset,
        split_role="train",
        world_size=3,
        batch_size=2,
        gradient_accumulation_steps=2,
    ).training_view_digest_sha256


def test_risk_snapshot_digest_binds_complete_manifest_projection(tmp_path):
    _, dataset = _publish_and_load(tmp_path / "source")
    snapshot = open_authenticated_risk_snapshot(
        dataset,
        cache_root=tmp_path / "cache",
    )
    manifest = _read_snapshot_manifest(snapshot.root)
    source_identity = manifest.get("source_identity")
    identity = manifest.get("identity")

    assert source_identity == snapshot.source_identity
    assert isinstance(identity, dict)
    assert identity["sample_ids"] == {
        "sha256": risk_training_store_module._sha256_file(
            snapshot.root / "sample_ids.json"
        ),
        "count": dataset.sample_count,
    }
    assert set(identity["arrays"]) == {*MODEL_INPUT_KEYS, *TARGET_KEYS}
    assert set(identity["sample_shard_indices"]) == {"sha256", "shape", "dtype"}
    assert snapshot.snapshot_digest_sha256 == (
        risk_training_store_module._sha256_bytes(
            risk_training_store_module._canonical_json_bytes(identity)
        )
    )
    request_digest = risk_training_store_module._sha256_bytes(
        risk_training_store_module._canonical_json_bytes(source_identity)
    )
    assert snapshot.root.name == request_digest
    assert snapshot.snapshot_digest_sha256 != request_digest
    marker = json.loads(
        (snapshot.root / ".snapshot-complete").read_text(encoding="utf-8")
    )
    assert marker == {
        "layout_version": risk_training_store_module.RISK_TRAINING_SNAPSHOT_LAYOUT_VERSION,
        "snapshot_digest_sha256": snapshot.snapshot_digest_sha256,
        "snapshot_manifest_sha256": snapshot.snapshot_manifest_sha256,
    }


@pytest.mark.parametrize("mutation", ("root", "manifest", "array"))
def test_risk_snapshot_rejects_unknown_cache_entries(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, dataset = _publish_and_load(tmp_path / "source")
    cache_root = tmp_path / "cache"
    snapshot = open_authenticated_risk_snapshot(dataset, cache_root=cache_root)
    manifest = _read_snapshot_manifest(snapshot.root)
    if mutation == "root":
        (snapshot.root / "unexpected.npy").write_bytes(b"partial")
    elif mutation == "manifest":
        manifest["unexpected"] = True
        _rewrite_risk_manifest(snapshot, manifest)
    else:
        _risk_array_entries(manifest)["bev_history"]["unexpected"] = True
        _rewrite_risk_manifest(snapshot, manifest)

    with pytest.raises(RiskDataContractError, match="unexpected|fields|declaration"):
        open_authenticated_risk_snapshot(dataset, cache_root=cache_root)


@pytest.mark.parametrize("mutation", ("non_list", "count"))
def test_risk_snapshot_rejects_invalid_sample_id_container_or_count(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, dataset = _publish_and_load(tmp_path / "source")
    cache_root = tmp_path / "cache"
    snapshot = open_authenticated_risk_snapshot(dataset, cache_root=cache_root)
    manifest = _read_snapshot_manifest(snapshot.root)
    ids_path = snapshot.root / "sample_ids.json"
    sample_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    assert isinstance(sample_ids, list)
    identity = manifest["identity"]
    assert isinstance(identity, dict)
    declaration = identity.get("sample_ids")
    if mutation == "non_list":
        _write_snapshot_json(
            ids_path,
            {sample_id: index for index, sample_id in enumerate(sample_ids)},
        )
        digest = risk_training_store_module._sha256_file(ids_path)
        if isinstance(declaration, dict):
            declaration["sha256"] = digest
        else:
            manifest["sample_ids_sha256"] = digest
    elif isinstance(declaration, dict):
        declaration["count"] = dataset.sample_count + 1
    else:
        manifest["sample_ids_count"] = dataset.sample_count + 1
    _rewrite_risk_manifest(snapshot, manifest)

    with pytest.raises(RiskDataContractError, match="sample ID|sample_ids|count"):
        open_authenticated_risk_snapshot(dataset, cache_root=cache_root)


def test_risk_snapshot_rejects_grid_incompatible_tensor_shape(tmp_path: Path) -> None:
    _, dataset = _publish_and_load(tmp_path / "source")
    cache_root = tmp_path / "cache"
    snapshot = open_authenticated_risk_snapshot(dataset, cache_root=cache_root)
    manifest = _read_snapshot_manifest(snapshot.root)
    path = snapshot.root / "bev_history.npy"
    values = np.load(path, allow_pickle=False)
    reshaped = values.reshape(values.shape[0], -1)
    np.save(path, reshaped, allow_pickle=False)
    entry = _risk_array_entries(manifest)["bev_history"]
    entry.update(
        {
            "sha256": risk_training_store_module._sha256_file(path),
            "shape": list(reshaped.shape),
            "dtype": str(reshaped.dtype),
        }
    )
    _rewrite_risk_manifest(snapshot, manifest)

    with pytest.raises(RiskDataContractError, match="shape|contract"):
        open_authenticated_risk_snapshot(dataset, cache_root=cache_root)


@pytest.mark.parametrize("mutation", ("out_of_range", "wrong_counts"))
def test_risk_snapshot_rejects_invalid_shard_index_partition(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, dataset = _publish_and_load(tmp_path / "source")
    cache_root = tmp_path / "cache"
    snapshot = open_authenticated_risk_snapshot(dataset, cache_root=cache_root)
    manifest = _read_snapshot_manifest(snapshot.root)
    path = snapshot.root / "sample_shard_indices.npy"
    indices = np.array(np.load(path, allow_pickle=False), dtype=np.int32, copy=True)
    if mutation == "out_of_range":
        indices[0] = len(dataset.shards)
    else:
        assert len(dataset.shards) > 1
        indices[np.flatnonzero(indices == 0)[0]] = 1
    np.save(path, indices, allow_pickle=False)
    entry = _risk_shard_index_entry(manifest)
    entry.update(
        {
            "sha256": risk_training_store_module._sha256_file(path),
            "shape": list(indices.shape),
            "dtype": str(indices.dtype),
        }
    )
    _rewrite_risk_manifest(snapshot, manifest)

    with pytest.raises(RiskDataContractError, match="shard index|shard count"):
        open_authenticated_risk_snapshot(dataset, cache_root=cache_root)


def test_peer_descriptor_rejects_unknown_descriptor_layout(tmp_path: Path) -> None:
    _, dataset = _publish_and_load(tmp_path / "source")
    snapshot = open_authenticated_risk_snapshot(
        dataset,
        cache_root=tmp_path / "cache",
    )
    descriptor = snapshot.descriptor()
    expected_layout = getattr(
        risk_training_store_module,
        "RISK_SNAPSHOT_DESCRIPTOR_LAYOUT_VERSION",
        None,
    )
    assert expected_layout == "authenticated_risk_snapshot_descriptor_v1"
    descriptor["descriptor_layout_version"] = "unknown_snapshot_descriptor"

    with pytest.raises(RiskDataContractError, match="descriptor.*layout"):
        open_authenticated_risk_snapshot_descriptor(descriptor)


def test_peer_descriptor_rejects_re_signed_unknown_source_schema(tmp_path: Path) -> None:
    _, dataset = _publish_and_load(tmp_path / "source")
    snapshot = open_authenticated_risk_snapshot(
        dataset,
        cache_root=tmp_path / "cache",
    )
    descriptor = json.loads(json.dumps(snapshot.descriptor()))
    manifest = _read_snapshot_manifest(snapshot.root)
    source_identity = manifest.get("source_identity")
    identity = manifest["identity"]
    assert isinstance(identity, dict)
    if isinstance(source_identity, dict):
        source_identity["schema_version"] = "999.0.0"
        identity["schema_version"] = "999.0.0"
        descriptor["source_identity"] = json.loads(json.dumps(source_identity))
    else:
        identity["schema_version"] = "999.0.0"
        descriptor["source_identity"] = json.loads(json.dumps(identity))
    rewritten = _rewrite_risk_manifest(
        snapshot,
        manifest,
        descriptor=descriptor,
    )
    assert rewritten is not None

    with pytest.raises(RiskDataContractError, match="source.*schema|schema_version"):
        open_authenticated_risk_snapshot_descriptor(rewritten)


@pytest.mark.parametrize("mutation", ("missing", "invalid"))
def test_peer_descriptor_validates_bulk_digest_declarations_without_rehash(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, dataset = _publish_and_load(tmp_path / "source")
    snapshot = open_authenticated_risk_snapshot(
        dataset,
        cache_root=tmp_path / "cache",
    )
    descriptor = snapshot.descriptor()
    manifest = _read_snapshot_manifest(snapshot.root)
    entry = _risk_array_entries(manifest)["bev_history"]
    if mutation == "missing":
        entry.pop("sha256")
    else:
        entry["sha256"] = "not-a-sha256"
    rewritten = _rewrite_risk_manifest(
        snapshot,
        manifest,
        descriptor=descriptor,
    )
    assert rewritten is not None

    with pytest.raises(RiskDataContractError, match="digest|declaration"):
        open_authenticated_risk_snapshot_descriptor(rewritten)


def test_shared_winner_reuse_rejects_different_full_snapshot_identity(
    tmp_path: Path,
) -> None:
    helper = getattr(
        risk_training_store_module,
        "_discard_staging_if_matching_winner",
        None,
    )
    assert callable(helper)
    staging = tmp_path / "staging"
    winner = tmp_path / "winner"
    staging.mkdir()
    winner.mkdir()

    def write_manifest(root: Path, identity: dict[str, object]) -> None:
        digest = risk_training_store_module._sha256_bytes(
            risk_training_store_module._canonical_json_bytes(identity)
        )
        _write_snapshot_json(
            root / "snapshot_manifest.json",
            {
                "snapshot_manifest_layout_version": "test_snapshot_v1",
                "source_identity": {"request": "same"},
                "identity": identity,
                "snapshot_digest_sha256": digest,
            },
        )

    write_manifest(staging, {"payload": "staged"})
    write_manifest(winner, {"payload": "winner"})
    with pytest.raises(RiskDataContractError, match="winner.*differ|identity"):
        helper(staging, winner)
    assert staging.is_dir()

    shutil.rmtree(winner)
    shutil.copytree(staging, winner)
    helper(staging, winner)
    assert not staging.exists()


def test_risk_atomic_eexist_reuses_only_through_winner_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset = _publish_and_load(tmp_path / "source")
    helper = getattr(
        risk_training_store_module,
        "_discard_staging_if_matching_winner",
        None,
    )
    assert callable(helper)
    compared: list[tuple[Path, Path]] = []

    def observed_helper(staging: Path, winner: Path) -> None:
        compared.append((staging, winner))
        helper(staging, winner)

    def racing_publish(staging: Path, winner: Path) -> None:
        shutil.copytree(staging, winner)
        raise FileExistsError(winner)

    monkeypatch.setattr(
        risk_training_store_module,
        "_discard_staging_if_matching_winner",
        observed_helper,
    )
    monkeypatch.setattr(
        risk_training_store_module,
        "atomic_rename_noreplace",
        racing_publish,
    )

    snapshot = open_authenticated_risk_snapshot(
        dataset,
        cache_root=tmp_path / "cache",
    )

    assert snapshot.sample_ids
    assert len(compared) == 1
    assert compared[0][0].name.startswith(".risk-snapshot.staging-")


def test_risk_snapshot_fsyncs_files_with_completion_marker_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset = _publish_and_load(tmp_path / "source")
    cache_root = tmp_path / "cache"
    file_fsyncs: list[Path] = []
    directory_fsyncs: list[Path] = []
    monkeypatch.setattr(
        risk_training_store_module,
        "_fsync_file",
        lambda path: file_fsyncs.append(Path(path)),
        raising=False,
    )
    monkeypatch.setattr(
        risk_training_store_module,
        "_fsync_directory",
        lambda path: directory_fsyncs.append(Path(path)),
        raising=False,
    )

    snapshot = open_authenticated_risk_snapshot(dataset, cache_root=cache_root)

    assert snapshot.sample_ids
    assert file_fsyncs, "snapshot publication must fsync staged files"
    assert file_fsyncs[-1].name == ".snapshot-complete"
    assert {path.name for path in file_fsyncs} >= {
        *(f"{name}.npy" for name in (*MODEL_INPUT_KEYS, *TARGET_KEYS)),
        "sample_ids.json",
        "sample_shard_indices.npy",
        "snapshot_manifest.json",
        ".snapshot-complete",
    }
    assert directory_fsyncs, "snapshot publication must fsync the cache parent"
    assert directory_fsyncs[-1] == cache_root.absolute()


def test_public_seal_loader_signature_and_exports_do_not_expose_consumers(
    tmp_path: Path,
) -> None:
    publication, dataset = _publish_and_load(tmp_path / "source")
    signature = inspect.signature(risk_dataset_seal_module.load_risk_dataset_seal)

    assert tuple(signature.parameters) == (
        "seal_root",
        "collection_root",
        "expected_split",
        "expected_manifest_digest",
        "sidecar_root",
    )
    assert not any("consumer" in name for name in signature.parameters)
    assert not any(
        "Consumer" in name or name.startswith("AuthenticatedRiskSidecarPair")
        for name in risk_dataset_seal_module.__all__
    )

    reloaded = risk_dataset_seal_module.load_risk_dataset_seal(
        dataset.seal_root,
        collection_root=publication.collection_root,
        expected_split="train",
    )
    assert reloaded.risk_dataset_manifest_digest == (
        dataset.risk_dataset_manifest_digest
    )
