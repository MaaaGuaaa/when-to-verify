"""Deterministic, resumable production streaming tests for ``risk_dataset_v2``."""

from __future__ import annotations

import gc
from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import json
from pathlib import Path
import weakref

import pytest
import torch

from src.contracts import FORBIDDEN_INPUT_TOKENS, SCHEMA_VERSION
import src.datasets.risk_dataloader as dataloader_module
from src.datasets.risk_dataloader import (
    MODEL_INPUT_KEYS,
    TARGET_KEYS,
    ProductionRiskSubset,
    RiskDataContractError,
    RiskStreamCursor,
    collate_production_risk_samples,
    iter_production_risk_batches,
    select_production_risk_subset,
)
from src.datasets.risk_dataset_seal import (
    LoadedRiskDataset,
    load_risk_dataset_seal,
    publish_risk_dataset_seal,
)
from src.datasets.shard_writer import load_risk_shard
from src.datasets.toy_risk_learning import frozen_channel_spec
from tests.fixtures.formal_risk_publication import (
    FormalRiskPublication,
    create_formal_risk_publication,
)


_SUBSET_MEMBERSHIP_DOMAIN = "risk-production-subset-membership-v1"
_SUBSET_DIGEST_DOMAIN = "risk-production-subset-v1"
_SHARD_ORDER_DOMAIN = "risk-production-shard-order-v1"
_ROW_ORDER_DOMAIN = "risk-production-row-order-v1"


def _publish_and_load(tmp_path: Path) -> tuple[FormalRiskPublication, LoadedRiskDataset]:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    seal_root = publish_risk_dataset_seal(
        tmp_path / "seal",
        collection_root=publication.collection_root,
        base_config_path=publication.base_config_path,
        split_provenance_path=publication.split_provenance_path,
        expected_split="train",
        expected_collection_handoff_sha256=publication.handoff_sha256,
    )
    dataset = load_risk_dataset_seal(
        seal_root,
        collection_root=publication.collection_root,
        expected_split="train",
    )
    return publication, dataset


def _expected_subset_digest(
    sample_ids: tuple[str, ...],
    *,
    dataset_manifest_digest: str,
    seed: int,
    max_samples: int,
) -> str:
    payload = json.dumps(
        {
            "dataset_manifest_digest": dataset_manifest_digest,
            "domain": _SUBSET_DIGEST_DOMAIN,
            "max_samples": max_samples,
            "sample_ids": list(sample_ids),
            "seed": seed,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_score(domain: str, **components: object) -> bytes:
    payload = json.dumps(
        {"domain": domain, **components},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _resign_subset(
    subset: ProductionRiskSubset,
    *,
    sample_ids: tuple[str, ...] | None = None,
    dataset_manifest_digest: str | None = None,
    seed: int | None = None,
    max_samples: int | None = None,
) -> ProductionRiskSubset:
    ids = subset.sample_ids if sample_ids is None else sample_ids
    dataset_digest = (
        subset.dataset_manifest_digest
        if dataset_manifest_digest is None
        else dataset_manifest_digest
    )
    selection_seed = subset.seed if seed is None else seed
    limit = subset.max_samples if max_samples is None else max_samples
    return ProductionRiskSubset(
        sample_ids=ids,
        sample_ids_digest_sha256=_expected_subset_digest(
            ids,
            dataset_manifest_digest=dataset_digest,
            seed=selection_seed,
            max_samples=limit,
        ),
        dataset_manifest_digest=dataset_digest,
        seed=selection_seed,
        max_samples=limit,
    )


def _flatten_ids(
    items: list[tuple[object, RiskStreamCursor]],
) -> tuple[str, ...]:
    return tuple(
        sample_id
        for batch, _ in items
        for sample_id in batch.sample_ids  # type: ignore[attr-defined]
    )


def test_public_subset_and_cursor_contracts_are_exact_and_frozen() -> None:
    assert [field.name for field in fields(ProductionRiskSubset)] == [
        "sample_ids",
        "sample_ids_digest_sha256",
        "dataset_manifest_digest",
        "seed",
        "max_samples",
    ]
    assert [field.name for field in fields(RiskStreamCursor)] == [
        "epoch",
        "shard_order_position",
        "shard_index",
        "row_order_position",
        "samples_yielded",
        "dataset_manifest_digest",
        "subset_digest_sha256",
    ]
    subset = ProductionRiskSubset(("sample",), "a" * 64, "b" * 64, 3, 1)
    cursor = RiskStreamCursor(0, 0, 0, 0, 0, "b" * 64, "a" * 64)
    with pytest.raises(FrozenInstanceError):
        subset.seed = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        cursor.epoch = 1  # type: ignore[misc]


def test_selection_is_deterministic_seed_sensitive_and_digest_exact(
    tmp_path: Path,
) -> None:
    _, dataset = _publish_and_load(tmp_path)

    first = select_production_risk_subset(dataset, max_samples=9, seed=42)
    repeated = select_production_risk_subset(dataset, max_samples=9, seed=42)
    another_seed = select_production_risk_subset(dataset, max_samples=9, seed=43)
    all_samples = select_production_risk_subset(dataset, max_samples=99, seed=42)

    assert first == repeated
    assert len(first.sample_ids) == 9
    assert len(set(first.sample_ids)) == 9
    assert first.sample_ids != another_seed.sample_ids
    assert len(all_samples.sample_ids) == dataset.sample_count
    all_ids = tuple(f"train-formal-{index:03d}" for index in range(12))
    expected_ids = tuple(
        sorted(
            all_ids,
            key=lambda sample_id: (
                _expected_score(
                    _SUBSET_MEMBERSHIP_DOMAIN,
                    dataset_manifest_digest=dataset.risk_dataset_manifest_digest,
                    seed=42,
                    sample_id=sample_id,
                ),
                sample_id,
            ),
        )[:9]
    )
    assert first.sample_ids == expected_ids
    assert first.dataset_manifest_digest == dataset.risk_dataset_manifest_digest
    assert first.sample_ids_digest_sha256 == _expected_subset_digest(
        first.sample_ids,
        dataset_manifest_digest=dataset.risk_dataset_manifest_digest,
        seed=42,
        max_samples=9,
    )


@pytest.mark.parametrize(
    ("max_samples", "seed", "message"),
    [
        (True, 0, "max_samples"),
        (0, 0, "max_samples"),
        (-1, 0, "max_samples"),
        (1, True, "seed"),
        (1, -1, "seed"),
    ],
)
def test_selection_rejects_invalid_integer_arguments(
    tmp_path: Path, max_samples: object, seed: object, message: str
) -> None:
    _, dataset = _publish_and_load(tmp_path)
    with pytest.raises((TypeError, RiskDataContractError), match=message):
        select_production_risk_subset(
            dataset,
            max_samples=max_samples,  # type: ignore[arg-type]
            seed=seed,  # type: ignore[arg-type]
        )


def test_stream_is_reproducible_epoch_sensitive_and_subset_invariant(
    tmp_path: Path,
) -> None:
    _, dataset = _publish_and_load(tmp_path)
    subset = select_production_risk_subset(dataset, max_samples=9, seed=42)

    epoch_zero = list(
        iter_production_risk_batches(
            dataset, subset=subset, batch_size=4, seed=42, epoch=0
        )
    )
    repeated = list(
        iter_production_risk_batches(
            dataset, subset=subset, batch_size=4, seed=42, epoch=0
        )
    )
    epoch_one = list(
        iter_production_risk_batches(
            dataset, subset=subset, batch_size=4, seed=42, epoch=1
        )
    )

    first_ids = _flatten_ids(epoch_zero)
    assert first_ids == _flatten_ids(repeated)
    assert first_ids != _flatten_ids(epoch_one)
    assert set(first_ids) == set(_flatten_ids(epoch_one)) == set(subset.sample_ids)
    assert len(first_ids) == len(set(first_ids)) == len(subset.sample_ids)
    assert select_production_risk_subset(dataset, max_samples=9, seed=42) == subset

    shard_by_id = {
        f"train-formal-{index:03d}": index // 6 for index in range(12)
    }
    original_index = {
        f"train-formal-{index:03d}": index % 6 for index in range(12)
    }
    for epoch, observed in ((0, first_ids), (1, _flatten_ids(epoch_one))):
        ordered_shards = sorted(
            range(2),
            key=lambda shard_index: (
                _expected_score(
                    _SHARD_ORDER_DOMAIN,
                    dataset_manifest_digest=dataset.risk_dataset_manifest_digest,
                    seed=42,
                    epoch=epoch,
                    shard_index=shard_index,
                ),
                shard_index,
            ),
        )
        expected_epoch_ids = tuple(
            sample_id
            for shard_index in ordered_shards
            for sample_id in sorted(
                (
                    sample_id
                    for sample_id in subset.sample_ids
                    if shard_by_id[sample_id] == shard_index
                ),
                key=lambda sample_id: (
                    _expected_score(
                        _ROW_ORDER_DOMAIN,
                        dataset_manifest_digest=dataset.risk_dataset_manifest_digest,
                        seed=42,
                        epoch=epoch,
                        sample_id=sample_id,
                    ),
                    sample_id,
                    original_index[sample_id],
                ),
            )
        )
        assert observed == expected_epoch_ids
    for batch, _ in epoch_zero:
        assert len({shard_by_id[sample_id] for sample_id in batch.sample_ids}) == 1
        assert 1 <= len(batch.sample_ids) <= 4


def test_production_batches_have_exact_inputs_float32_targets_and_provenance(
    tmp_path: Path,
) -> None:
    _, dataset = _publish_and_load(tmp_path)
    subset = select_production_risk_subset(dataset, max_samples=12, seed=7)
    batches = list(
        iter_production_risk_batches(
            dataset, subset=subset, batch_size=4, seed=7, epoch=2
        )
    )
    expected_provenance = {
        "mode": "production",
        "schema_version": SCHEMA_VERSION,
        "channel_spec": frozen_channel_spec(),
        **dict(dataset.provenance),
    }

    for batch, _ in batches:
        size = len(batch.sample_ids)
        assert batch.split == "train"
        assert set(batch.model_inputs) == set(MODEL_INPUT_KEYS)
        assert set(batch.targets) == set(TARGET_KEYS)
        assert batch.model_inputs["bev_history"].shape == (
            size,
            dataset.grid.history_steps,
            dataset.grid.n_history_channels,
            dataset.grid.height,
            dataset.grid.width,
        )
        assert batch.model_inputs["state_channels"].shape == (
            size,
            dataset.grid.n_state_channels,
            dataset.grid.height,
            dataset.grid.width,
        )
        assert batch.model_inputs["trajectory_channels"].shape == (
            size,
            dataset.grid.n_trajectory_channels,
            dataset.grid.height,
            dataset.grid.width,
        )
        assert batch.model_inputs["robot_state"].shape == (size, 2)
        for tensor in (*batch.model_inputs.values(), *batch.targets.values()):
            assert tensor.dtype == torch.float32
            assert torch.isfinite(tensor).all().item()
        assert all(batch.targets[key].shape == (size,) for key in TARGET_KEYS)
        assert batch.provenance == expected_provenance
        assert "toy_dataset_manifest_digest" not in batch.provenance
        assert not any(
            token in key.lower()
            for key in batch.model_inputs
            for token in FORBIDDEN_INPUT_TOKENS
        )


def test_temporal_safe_labels_come_from_fields_not_event_type(tmp_path: Path) -> None:
    _, dataset = _publish_and_load(tmp_path)
    subset = select_production_risk_subset(dataset, max_samples=12, seed=11)
    batches = list(
        iter_production_risk_batches(
            dataset, subset=subset, batch_size=5, seed=11, epoch=0
        )
    )

    found = 0
    for batch, _ in batches:
        for row, sample_id in enumerate(batch.sample_ids):
            if sample_id in {"train-formal-005", "train-formal-011"}:
                found += 1
                assert batch.targets["collision_label"][row].item() == 0.0
                assert batch.targets["near_miss"][row].item() == 1.0
    assert found == 2


def test_resume_cursor_reproduces_uninterrupted_suffix_and_terminal_is_empty(
    tmp_path: Path,
) -> None:
    _, dataset = _publish_and_load(tmp_path)
    subset = select_production_risk_subset(dataset, max_samples=9, seed=42)
    uninterrupted = list(
        iter_production_risk_batches(
            dataset, subset=subset, batch_size=3, seed=42, epoch=4
        )
    )
    first_batch, first_cursor = uninterrupted[0]
    resumed = list(
        iter_production_risk_batches(
            dataset,
            subset=subset,
            batch_size=3,
            seed=42,
            epoch=4,
            start_cursor=first_cursor,
        )
    )

    combined = first_batch.sample_ids + _flatten_ids(resumed)
    assert combined == _flatten_ids(uninterrupted)
    assert len(combined) == len(set(combined)) == len(subset.sample_ids)
    assert first_cursor.samples_yielded == len(first_batch.sample_ids)
    terminal = uninterrupted[-1][1]
    assert terminal.shard_order_position == len(dataset.shards)
    assert terminal.shard_index == -1
    assert terminal.row_order_position == 0
    assert terminal.samples_yielded == len(subset.sample_ids)
    assert list(
        iter_production_risk_batches(
            dataset,
            subset=subset,
            batch_size=3,
            seed=42,
            epoch=4,
            start_cursor=terminal,
        )
    ) == []


def test_tampered_cursor_fields_are_rejected(tmp_path: Path) -> None:
    _, dataset = _publish_and_load(tmp_path)
    subset = select_production_risk_subset(dataset, max_samples=9, seed=42)
    first = next(
        iter_production_risk_batches(
            dataset, subset=subset, batch_size=2, seed=42, epoch=3
        )
    )
    cursor = first[1]
    mutations = (
        replace(cursor, epoch=4),
        replace(cursor, shard_order_position=len(dataset.shards) + 1),
        replace(cursor, shard_index=99),
        replace(cursor, row_order_position=99),
        replace(cursor, samples_yielded=cursor.samples_yielded + 1),
        replace(cursor, dataset_manifest_digest="f" * 64),
        replace(cursor, subset_digest_sha256="e" * 64),
        replace(cursor, row_order_position=True),
    )
    for changed in mutations:
        with pytest.raises(RiskDataContractError, match="cursor"):
            list(
                iter_production_risk_batches(
                    dataset,
                    subset=subset,
                    batch_size=2,
                    seed=42,
                    epoch=3,
                    start_cursor=changed,
                )
            )


def test_subset_authentication_rejects_digest_count_duplicate_and_unknown_ids(
    tmp_path: Path,
) -> None:
    _, dataset = _publish_and_load(tmp_path)
    subset = select_production_risk_subset(dataset, max_samples=9, seed=42)
    duplicate_ids = (subset.sample_ids[0], subset.sample_ids[0], *subset.sample_ids[2:])
    unknown_ids = ("unknown-production-sample", *subset.sample_ids[1:])
    bad_subsets = (
        replace(subset, sample_ids_digest_sha256="A" * 64),
        _resign_subset(subset, max_samples=8),
        _resign_subset(subset, sample_ids=subset.sample_ids[:-1]),
        _resign_subset(subset, sample_ids=duplicate_ids),
        _resign_subset(subset, sample_ids=unknown_ids),
        replace(subset, dataset_manifest_digest="f" * 64),
    )
    for changed in bad_subsets:
        with pytest.raises(RiskDataContractError, match="subset|sample|digest|max_samples"):
            list(
                iter_production_risk_batches(
                    dataset,
                    subset=changed,
                    batch_size=3,
                    seed=changed.seed,
                    epoch=0,
                )
            )


@pytest.mark.parametrize(
    ("batch_size", "seed", "epoch", "message"),
    [
        (True, 42, 0, "batch_size"),
        (0, 42, 0, "batch_size"),
        (-1, 42, 0, "batch_size"),
        (2, True, 0, "seed"),
        (2, -1, 0, "seed"),
        (2, 42, True, "epoch"),
        (2, 42, -1, "epoch"),
        (2, 43, 0, "seed"),
    ],
)
def test_stream_rejects_invalid_batch_seed_and_epoch(
    tmp_path: Path,
    batch_size: object,
    seed: object,
    epoch: object,
    message: str,
) -> None:
    _, dataset = _publish_and_load(tmp_path)
    subset = select_production_risk_subset(dataset, max_samples=4, seed=42)
    with pytest.raises((TypeError, RiskDataContractError), match=message):
        list(
            iter_production_risk_batches(
                dataset,
                subset=subset,
                batch_size=batch_size,  # type: ignore[arg-type]
                seed=seed,  # type: ignore[arg-type]
                epoch=epoch,  # type: ignore[arg-type]
            )
        )


def test_wrong_dataset_digest_and_split_are_rejected(tmp_path: Path) -> None:
    _, dataset = _publish_and_load(tmp_path)
    with pytest.raises(RiskDataContractError, match="dataset.*digest"):
        select_production_risk_subset(
            replace(dataset, risk_dataset_manifest_digest="f" * 64),
            max_samples=4,
            seed=1,
        )

    subset = select_production_risk_subset(dataset, max_samples=4, seed=1)
    wrong_split = replace(dataset, split="validation")
    with pytest.raises(RiskDataContractError, match="split"):
        list(
            iter_production_risk_batches(
                wrong_split, subset=subset, batch_size=2, seed=1, epoch=0
            )
        )


def test_runtime_dataset_metadata_must_match_authenticated_manifest(
    tmp_path: Path,
) -> None:
    _, dataset = _publish_and_load(tmp_path)
    forged_provenance = {
        **dict(dataset.provenance),
        "g1_split_manifest_digest": "a" * 32,
    }
    forged_grid = replace(
        dataset.grid,
        resolution_m=dataset.grid.resolution_m * 2.0,
    )
    manifest = dict(dataset.manifest)
    manifest_shards = [dict(value) for value in dataset.manifest["shards"]]
    manifest_shards[0]["payload_sha256"] = "f" * 64
    manifest["shards"] = tuple(manifest_shards)

    for changed in (
        replace(dataset, provenance=forged_provenance),
        replace(dataset, grid=forged_grid),
        replace(dataset, manifest=manifest),
    ):
        with pytest.raises(
            RiskDataContractError, match="manifest|provenance|grid|shard"
        ):
            select_production_risk_subset(changed, max_samples=4, seed=1)


def test_collator_rejects_empty_duplicate_and_wrong_split_samples(tmp_path: Path) -> None:
    publication, dataset = _publish_and_load(tmp_path)
    loaded = load_risk_shard(
        publication.collection_root / dataset.shards[0].relative_root,
        grid=dataset.grid,
        split_audit_records=(),
    )
    sample = loaded.samples[0]
    with pytest.raises(RiskDataContractError, match="empty"):
        collate_production_risk_samples(
            (),
            grid=dataset.grid,
            expected_split=dataset.split,
            dataset_provenance=dataset.provenance,
        )
    with pytest.raises(RiskDataContractError, match="unique"):
        collate_production_risk_samples(
            (sample, sample),
            grid=dataset.grid,
            expected_split=dataset.split,
            dataset_provenance=dataset.provenance,
        )
    with pytest.raises(RiskDataContractError, match="split"):
        collate_production_risk_samples(
            (sample,),
            grid=dataset.grid,
            expected_split="validation",
            dataset_provenance=dataset.provenance,
        )


def test_tampered_shard_fails_before_its_first_batch(tmp_path: Path) -> None:
    publication, dataset = _publish_and_load(tmp_path)
    subset = select_production_risk_subset(dataset, max_samples=1, seed=5)
    summary = publication.collection_root / "shard-00000" / "summary.json"
    summary.write_bytes(summary.read_bytes() + b" ")

    stream = iter_production_risk_batches(
        dataset, subset=subset, batch_size=1, seed=5, epoch=0
    )
    with pytest.raises(RiskDataContractError, match="shard|summary|formal"):
        next(stream)


def test_stream_retains_at_most_one_formally_loaded_shard_plus_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, dataset = _publish_and_load(tmp_path)
    subset = select_production_risk_subset(dataset, max_samples=12, seed=17)
    real_loader = dataloader_module.load_risk_shard
    real_collator = dataloader_module.collate_production_risk_samples
    live_refs: list[weakref.ReferenceType[object]] = []
    batch_refs: list[weakref.ReferenceType[object]] = []
    load_names: list[str] = []
    maximum_live = 0

    def tracked_loader(root: Path, **kwargs: object) -> object:
        nonlocal maximum_live
        gc.collect()
        alive_before = sum(reference() is not None for reference in live_refs)
        assert alive_before == 0, "the prior formally loaded shard was retained"
        loaded = real_loader(root, **kwargs)
        live_refs.append(weakref.ref(loaded))
        load_names.append(Path(root).name)
        maximum_live = max(maximum_live, alive_before + 1)
        return loaded

    def tracked_collator(samples: object, **kwargs: object) -> object:
        gc.collect()
        assert all(reference() is None for reference in batch_refs), (
            "the prior emitted RiskBatch was retained while constructing the next"
        )
        batch = real_collator(samples, **kwargs)
        batch_refs.append(weakref.ref(batch))
        return batch

    monkeypatch.setattr(dataloader_module, "load_risk_shard", tracked_loader)
    monkeypatch.setattr(
        dataloader_module,
        "collate_production_risk_samples",
        tracked_collator,
    )
    stream = iter_production_risk_batches(
        dataset, subset=subset, batch_size=2, seed=17, epoch=0
    )
    flattened_ids: list[str] = []
    while True:
        try:
            batch, cursor = next(stream)
        except StopIteration:
            break
        gc.collect()
        assert sum(reference() is not None for reference in live_refs) <= 1
        flattened_ids.extend(batch.sample_ids)
        del batch
        del cursor

    gc.collect()
    assert set(flattened_ids) == set(subset.sample_ids)
    assert all(reference() is None for reference in batch_refs)
    assert maximum_live == 1
    assert set(load_names) == {"shard-00000", "shard-00001"}
