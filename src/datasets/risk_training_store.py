"""Read-only, authenticated local snapshots for production risk training."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch

from src.contracts import (
    HISTORY_CHANNELS,
    ROBOT_STATE_DIM,
    SCHEMA_VERSION,
    STATE_CHANNELS,
    TRAJECTORY_CHANNELS,
)
import src.datasets.risk_dataloader as risk_dataloader_module
from src.datasets.risk_dataloader import (
    MODEL_INPUT_KEYS,
    TARGET_KEYS,
    OccupancyStreamCursor,
    ProductionOccupancyBatch,
    ProductionRiskSubset,
    RiskBatch,
    RiskDataContractError,
    RiskStreamCursor,
)
from src.datasets.risk_dataset_seal import (
    LoadedRiskDataset,
    RiskShardDescriptor,
    _AuthenticatedRiskSidecarPair,
    _load_risk_dataset_seal_with_consumers,
)
from src.datasets.shard_writer import LoadedRiskShard
from src.utils.atomic_publish import atomic_rename_noreplace


RISK_TRAINING_SNAPSHOT_LAYOUT_VERSION = "authenticated_risk_snapshot_v1"
RISK_SNAPSHOT_DESCRIPTOR_LAYOUT_VERSION = (
    "authenticated_risk_snapshot_descriptor_v1"
)
OCCUPANCY_TRAINING_SNAPSHOT_LAYOUT_VERSION = (
    "authenticated_occupancy_snapshot_v1"
)
_MANIFEST_NAME = "snapshot_manifest.json"
_COMPLETE_NAME = ".snapshot-complete"
_IDS_NAME = "sample_ids.json"
_SHARD_INDICES_NAME = "sample_shard_indices.npy"
_OCCUPANCY_QUERY_DIRECTORY = "query_inputs"
_OCCUPANCY_TARGET_DIRECTORY = "targets"
_ROBOT_ENDPOINT_FOOTPRINTS_NAME = "robot_endpoint_footprints.npy"
_ENDPOINT_TIMES_NAME = "endpoint_times_s.npy"
_HIDDEN_RISK_OCCUPANCY_NAME = "hidden_risk_occupancy.npy"
_RISK_SOURCE_IDENTITY_KEYS = frozenset(
    {
        "layout_version",
        "schema_version",
        "g1_split_manifest_digest",
        "risk_dataset_manifest_digest",
        "dynamic_objects_config_digest",
        "target_type_policy_digest",
        "sample_count",
        "split",
        "grid",
        "channel_spec",
        "source_shard_semantic_digests",
        "source_shard_sample_counts",
    }
)
_RISK_GRID_KEYS = frozenset(
    {
        "height",
        "width",
        "history_steps",
        "future_steps",
        "resolution_m",
        "sample_dt_s",
    }
)
_SNAPSHOT_MANIFEST_KEYS = frozenset(
    {
        "snapshot_manifest_layout_version",
        "source_identity",
        "identity",
        "snapshot_digest_sha256",
    }
)
_ARRAY_DECLARATION_KEYS = frozenset({"sha256", "shape", "dtype"})


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RiskDataContractError("snapshot metadata must be finite canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_value(value: object) -> object:
    return json.loads(_canonical_json_bytes(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise RiskDataContractError(f"snapshot file is missing: {path.name}") from exc
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise RiskDataContractError(f"snapshot file must be a regular file: {path.name}")


def _snapshot_identity(dataset: LoadedRiskDataset) -> dict[str, object]:
    return dict(_canonical_json_value({
        "layout_version": RISK_TRAINING_SNAPSHOT_LAYOUT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "g1_split_manifest_digest": dataset.provenance[
            "g1_split_manifest_digest"
        ],
        "risk_dataset_manifest_digest": dataset.risk_dataset_manifest_digest,
        "dynamic_objects_config_digest": dataset.provenance[
            "dynamic_objects_config_digest"
        ],
        "target_type_policy_digest": dataset.provenance[
            "target_type_policy_digest"
        ],
        "sample_count": dataset.sample_count,
        "split": dataset.split,
        "grid": dict(dataset.manifest["grid"]),
        "channel_spec": dataset.manifest["channel_spec"],
        "source_shard_semantic_digests": [
            descriptor.semantic_digest for descriptor in dataset.shards
        ],
        "source_shard_sample_counts": [
            descriptor.sample_count for descriptor in dataset.shards
        ],
    }))


@dataclass(frozen=True)
class AuthenticatedRiskSnapshot:
    """An immutable mmap-backed representation of authenticated risk tensors."""

    root: Path
    snapshot_digest_sha256: str
    snapshot_manifest_sha256: str
    source_identity: Mapping[str, object]
    sample_ids: tuple[str, ...]
    split: str
    provenance: dict[str, object]
    model_inputs: Mapping[str, np.ndarray]
    targets: Mapping[str, np.ndarray]
    sample_shard_indices: np.ndarray
    _sample_index: Mapping[str, int] = field(repr=False, compare=False)

    def descriptor(self) -> dict[str, object]:
        return {
            "descriptor_layout_version": RISK_SNAPSHOT_DESCRIPTOR_LAYOUT_VERSION,
            "root": str(self.root),
            "source_identity": dict(self.source_identity),
            "snapshot_digest_sha256": self.snapshot_digest_sha256,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "sample_count": len(self.sample_ids),
            "split": self.split,
        }

    def batch(self, sample_ids: Sequence[str]) -> RiskBatch:
        requested = tuple(sample_ids)
        if not requested or len(set(requested)) != len(requested):
            raise RiskDataContractError("snapshot batch sample IDs must be unique and non-empty")
        unknown = [sample_id for sample_id in requested if sample_id not in self._sample_index]
        if unknown:
            raise RiskDataContractError("snapshot batch contains unknown sample IDs")
        rows = [self._sample_index[sample_id] for sample_id in requested]
        return RiskBatch(
            model_inputs={
                name: torch.from_numpy(np.array(values[rows], dtype=np.float32, copy=True))
                for name, values in self.model_inputs.items()
            },
            targets={
                name: torch.from_numpy(np.array(values[rows], dtype=np.float32, copy=True))
                for name, values in self.targets.items()
            },
            sample_ids=requested,
            split=self.split,
            provenance={**self.provenance, "loader_mode": "authenticated_snapshot"},
        )

    def select_subset(self, *, max_samples: int, seed: int) -> ProductionRiskSubset:
        limit = risk_dataloader_module._require_positive_int(max_samples, "max_samples")
        selection_seed = risk_dataloader_module._require_nonnegative_int(seed, "seed")
        ranked = sorted(
            (
                risk_dataloader_module._stable_score(
                    risk_dataloader_module._SUBSET_MEMBERSHIP_DOMAIN,
                    dataset_manifest_digest=self.provenance[
                        "risk_dataset_manifest_digest"
                    ],
                    seed=selection_seed,
                    sample_id=sample_id,
                ),
                sample_id,
            )
            for sample_id in self.sample_ids
        )
        sample_ids = tuple(item[1] for item in ranked[: min(limit, len(ranked))])
        dataset_digest = str(self.provenance["risk_dataset_manifest_digest"])
        return ProductionRiskSubset(
            sample_ids=sample_ids,
            sample_ids_digest_sha256=risk_dataloader_module._subset_digest(
                sample_ids,
                dataset_manifest_digest=dataset_digest,
                seed=selection_seed,
                max_samples=limit,
            ),
            dataset_manifest_digest=dataset_digest,
            seed=selection_seed,
            max_samples=limit,
        )

    def ordered_sample_ids(
        self,
        *,
        subset: ProductionRiskSubset,
        seed: int,
        epoch: int,
    ) -> tuple[str, ...]:
        stream_seed = risk_dataloader_module._require_nonnegative_int(seed, "seed")
        stream_epoch = risk_dataloader_module._require_nonnegative_int(epoch, "epoch")
        expected_subset = self.select_subset(
            max_samples=subset.max_samples,
            seed=subset.seed,
        )
        if subset != expected_subset:
            raise RiskDataContractError("subset does not match authenticated snapshot selection")
        if stream_seed != subset.seed:
            raise RiskDataContractError("stream seed must match subset.seed")
        selected_by_shard: dict[int, list[tuple[bytes, str, int]]] = {}
        for sample_id in subset.sample_ids:
            row = self._sample_index[sample_id]
            shard_index = int(self.sample_shard_indices[row])
            selected_by_shard.setdefault(shard_index, []).append(
                (
                    risk_dataloader_module._stable_score(
                        risk_dataloader_module._ROW_ORDER_DOMAIN,
                        dataset_manifest_digest=subset.dataset_manifest_digest,
                        seed=stream_seed,
                        epoch=stream_epoch,
                        sample_id=sample_id,
                    ),
                    sample_id,
                    row,
                )
            )
        for rows in selected_by_shard.values():
            rows.sort(key=lambda item: (item[0], item[1], item[2]))
        shard_order = tuple(
            item[1]
            for item in sorted(
                (
                    risk_dataloader_module._stable_score(
                        risk_dataloader_module._SHARD_ORDER_DOMAIN,
                        dataset_manifest_digest=subset.dataset_manifest_digest,
                        seed=stream_seed,
                        epoch=stream_epoch,
                        shard_index=shard_index,
                    ),
                    shard_index,
                )
                for shard_index in selected_by_shard
            )
        )
        return tuple(
            row[1]
            for shard_index in shard_order
            for row in selected_by_shard[shard_index]
        )

    def iter_batches(
        self,
        *,
        subset: ProductionRiskSubset,
        batch_size: int,
        seed: int,
        epoch: int,
        start_cursor: RiskStreamCursor | None = None,
    ) -> Iterator[tuple[RiskBatch, RiskStreamCursor]]:
        size = risk_dataloader_module._require_positive_int(batch_size, "batch_size")
        stream_seed = risk_dataloader_module._require_nonnegative_int(seed, "seed")
        stream_epoch = risk_dataloader_module._require_nonnegative_int(epoch, "epoch")
        if not isinstance(subset, ProductionRiskSubset):
            raise RiskDataContractError("subset must be a ProductionRiskSubset")
        if stream_seed != subset.seed:
            raise RiskDataContractError("stream seed must match subset.seed")
        if subset.dataset_manifest_digest != self.provenance["risk_dataset_manifest_digest"]:
            raise RiskDataContractError("subset dataset digest does not match snapshot")
        expected_count = min(subset.max_samples, len(self.sample_ids))
        if len(subset.sample_ids) != expected_count:
            raise RiskDataContractError("subset sample count is inconsistent with snapshot")
        if len(set(subset.sample_ids)) != len(subset.sample_ids):
            raise RiskDataContractError("subset.sample_ids must be unique")
        if any(sample_id not in self._sample_index for sample_id in subset.sample_ids):
            raise RiskDataContractError("subset contains unknown snapshot sample IDs")
        expected_digest = risk_dataloader_module._subset_digest(
            subset.sample_ids,
            dataset_manifest_digest=subset.dataset_manifest_digest,
            seed=subset.seed,
            max_samples=subset.max_samples,
        )
        if subset.sample_ids_digest_sha256 != expected_digest:
            raise RiskDataContractError("subset sample_ids_digest_sha256 mismatch")

        selected_by_shard: dict[int, list[tuple[bytes, str, int]]] = {}
        for sample_id in subset.sample_ids:
            row = self._sample_index[sample_id]
            shard_index = int(self.sample_shard_indices[row])
            selected_by_shard.setdefault(shard_index, []).append(
                (
                    risk_dataloader_module._stable_score(
                        risk_dataloader_module._ROW_ORDER_DOMAIN,
                        dataset_manifest_digest=subset.dataset_manifest_digest,
                        seed=stream_seed,
                        epoch=stream_epoch,
                        sample_id=sample_id,
                    ),
                    sample_id,
                    row,
                )
            )
        for rows in selected_by_shard.values():
            rows.sort(key=lambda item: (item[0], item[1], item[2]))
        shard_indices = sorted({int(value) for value in self.sample_shard_indices.tolist()})
        shard_order = tuple(
            item[1]
            for item in sorted(
                (
                    risk_dataloader_module._stable_score(
                        risk_dataloader_module._SHARD_ORDER_DOMAIN,
                        dataset_manifest_digest=subset.dataset_manifest_digest,
                        seed=stream_seed,
                        epoch=stream_epoch,
                        shard_index=shard_index,
                    ),
                    shard_index,
                )
                for shard_index in shard_indices
            )
        )
        selected_counts = {
            shard_index: len(selected_by_shard.get(shard_index, ()))
            for shard_index in shard_order
        }
        if start_cursor is None:
            order_position, row_position, _ = risk_dataloader_module._initial_stream_position(
                shard_order, selected_counts
            )
        else:
            order_position, row_position, _ = risk_dataloader_module._validate_start_cursor(
                start_cursor,
                epoch=stream_epoch,
                shard_order=shard_order,
                selected_counts=selected_counts,
                dataset_digest=subset.dataset_manifest_digest,
                subset_digest=subset.sample_ids_digest_sha256,
                total_samples=len(subset.sample_ids),
            )
            if order_position == len(shard_order):
                return

        for current_position in range(order_position, len(shard_order)):
            shard_index = shard_order[current_position]
            selected_rows = selected_by_shard.get(shard_index, ())
            if not selected_rows:
                continue
            current_row = row_position if current_position == order_position else 0
            samples_before_shard = sum(
                selected_counts.get(shard_order[position], 0)
                for position in range(current_position)
            )
            while current_row < len(selected_rows):
                end = min(current_row + size, len(selected_rows))
                sample_ids = tuple(
                    selected_rows[position][1] for position in range(current_row, end)
                )
                cursor = risk_dataloader_module._next_stream_cursor(
                    epoch=stream_epoch,
                    shard_order=shard_order,
                    selected_counts=selected_counts,
                    current_order_position=current_position,
                    next_row_position=end,
                    samples_yielded=samples_before_shard + end,
                    dataset_digest=subset.dataset_manifest_digest,
                    subset_digest=subset.sample_ids_digest_sha256,
                )
                yield self.batch(sample_ids), cursor
                current_row = end


@dataclass(frozen=True)
class AuthenticatedRiskTrainingView:
    snapshot: AuthenticatedRiskSnapshot = field(repr=False, compare=False)
    subset: ProductionRiskSubset
    split_role: str
    world_size: int
    batch_size: int
    gradient_accumulation_steps: int
    training_view_digest_sha256: str

    def partition(self, *, epoch: int):
        from src.training.distributed import build_synchronous_partition_plan

        ordered = self.snapshot.ordered_sample_ids(
            subset=self.subset,
            seed=self.subset.seed,
            epoch=epoch,
        )
        return build_synchronous_partition_plan(
            ordered,
            subset_digest_sha256=self.subset.sample_ids_digest_sha256,
            seed=self.subset.seed,
            epoch=epoch,
            world_size=self.world_size,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )


def build_authenticated_risk_training_view(
    snapshot: AuthenticatedRiskSnapshot,
    *,
    subset: ProductionRiskSubset,
    split_role: str,
    world_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> AuthenticatedRiskTrainingView:
    if not isinstance(snapshot, AuthenticatedRiskSnapshot):
        raise RiskDataContractError("snapshot must be an AuthenticatedRiskSnapshot")
    if split_role not in {"train", "validation"}:
        raise RiskDataContractError("split_role must be train or validation")
    expected_split = "train" if split_role == "train" else "val"
    if snapshot.split != expected_split:
        raise RiskDataContractError("training view split role does not match snapshot")
    checked_world_size = risk_dataloader_module._require_positive_int(
        world_size, "world_size"
    )
    checked_batch_size = risk_dataloader_module._require_positive_int(
        batch_size, "batch_size"
    )
    checked_accumulation = risk_dataloader_module._require_positive_int(
        gradient_accumulation_steps, "gradient_accumulation_steps"
    )
    expected_subset = snapshot.select_subset(
        max_samples=subset.max_samples,
        seed=subset.seed,
    )
    if subset != expected_subset:
        raise RiskDataContractError("training view subset does not match snapshot")
    identity = {
        "layout_version": "authenticated_risk_training_view_v1",
        "snapshot_digest_sha256": snapshot.snapshot_digest_sha256,
        "risk_dataset_manifest_digest": snapshot.provenance[
            "risk_dataset_manifest_digest"
        ],
        "subset_digest_sha256": subset.sample_ids_digest_sha256,
        "seed": subset.seed,
        "split_role": split_role,
        "world_size": checked_world_size,
        "batch_size": checked_batch_size,
        "gradient_accumulation_steps": checked_accumulation,
    }
    return AuthenticatedRiskTrainingView(
        snapshot=snapshot,
        subset=subset,
        split_role=split_role,
        world_size=checked_world_size,
        batch_size=checked_batch_size,
        gradient_accumulation_steps=checked_accumulation,
        training_view_digest_sha256=_sha256_bytes(_canonical_json_bytes(identity)),
    )


def _open_snapshot(
    root: Path,
    *,
    expected_source_identity: Mapping[str, object],
    expected_snapshot_digest: str | None = None,
    expected_manifest_sha256: str | None = None,
    verify_bulk: bool = True,
) -> AuthenticatedRiskSnapshot:
    expected_files = {
        _MANIFEST_NAME,
        _COMPLETE_NAME,
        _IDS_NAME,
        _SHARD_INDICES_NAME,
        *(f"{name}.npy" for name in (*MODEL_INPUT_KEYS, *TARGET_KEYS)),
    }
    for name in expected_files:
        _require_regular(root / name)
    _require_exact_entries(
        root,
        expected=expected_files,
        label="risk snapshot root",
    )
    source_identity = _validate_risk_source_identity(expected_source_identity)
    request_digest = _sha256_bytes(_canonical_json_bytes(source_identity))
    if root.name != request_digest:
        raise RiskDataContractError("risk snapshot root request digest mismatch")
    manifest_path = root / _MANIFEST_NAME
    marker_path = root / _COMPLETE_NAME
    manifest_bytes = manifest_path.read_bytes()
    manifest = _read_canonical_snapshot_json(
        manifest_path,
        label="risk snapshot manifest",
    )
    marker = _read_canonical_snapshot_json(
        marker_path,
        label="risk snapshot completion marker",
    )
    if set(manifest) != _SNAPSHOT_MANIFEST_KEYS:
        raise RiskDataContractError("risk snapshot manifest fields mismatch")
    if manifest.get("snapshot_manifest_layout_version") != (
        RISK_TRAINING_SNAPSHOT_LAYOUT_VERSION
    ):
        raise RiskDataContractError("risk snapshot manifest version mismatch")
    if manifest.get("source_identity") != source_identity:
        raise RiskDataContractError(
            "snapshot source identity does not match authenticated source"
        )
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise RiskDataContractError("snapshot identity is missing")
    expected_identity_keys = {
        *_RISK_SOURCE_IDENTITY_KEYS,
        "sample_ids",
        "arrays",
        "sample_shard_indices",
    }
    if set(identity) != expected_identity_keys:
        raise RiskDataContractError("snapshot identity fields mismatch")
    if {key: identity[key] for key in _RISK_SOURCE_IDENTITY_KEYS} != source_identity:
        raise RiskDataContractError("snapshot identity source projection mismatch")

    sample_ids_path = root / _IDS_NAME
    sample_ids_bytes = sample_ids_path.read_bytes()
    try:
        raw_sample_ids = json.loads(sample_ids_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RiskDataContractError("snapshot sample IDs are not valid JSON") from exc
    if sample_ids_bytes != _canonical_json_bytes(raw_sample_ids):
        raise RiskDataContractError("snapshot sample IDs must be canonical JSON")
    if not isinstance(raw_sample_ids, list):
        raise RiskDataContractError("snapshot sample IDs must be a JSON list")
    sample_ids = tuple(raw_sample_ids)
    sample_ids_entry = identity.get("sample_ids")
    if not isinstance(sample_ids_entry, Mapping) or set(sample_ids_entry) != {
        "sha256",
        "count",
    }:
        raise RiskDataContractError("snapshot sample_ids declaration mismatch")
    sample_ids_digest = _require_sha256_digest(
        sample_ids_entry.get("sha256"),
        label="snapshot sample IDs digest",
    )
    if sample_ids_digest != _sha256_bytes(sample_ids_bytes):
        raise RiskDataContractError("snapshot sample ID checksum mismatch")
    if (
        sample_ids_entry.get("count") != len(sample_ids)
        or len(sample_ids) != source_identity["sample_count"]
    ):
        raise RiskDataContractError("snapshot sample ID count mismatch")
    if not sample_ids or any(
        not isinstance(value, str) or not value for value in sample_ids
    ):
        raise RiskDataContractError("snapshot sample IDs are invalid")
    if len(sample_ids) != len(set(sample_ids)):
        raise RiskDataContractError("snapshot sample IDs are not unique")

    arrays = identity.get("arrays")
    expected_names = {*MODEL_INPUT_KEYS, *TARGET_KEYS}
    if not isinstance(arrays, Mapping) or set(arrays) != expected_names:
        raise RiskDataContractError(
            "snapshot array names do not match risk batch contract"
        )
    expected_shapes = _risk_snapshot_array_shapes(
        source_identity,
        sample_count=len(sample_ids),
    )
    loaded: dict[str, np.ndarray] = {}
    normalized_arrays: dict[str, dict[str, object]] = {}
    for name in sorted(expected_names):
        entry = arrays[name]
        if not isinstance(entry, Mapping) or set(entry) != _ARRAY_DECLARATION_KEYS:
            raise RiskDataContractError(
                f"snapshot array declaration mismatch: {name}"
            )
        declared_digest = _require_sha256_digest(
            entry.get("sha256"),
            label=f"snapshot array digest {name}",
        )
        expected_shape = expected_shapes[name]
        if entry.get("dtype") != "float32" or entry.get("shape") != list(
            expected_shape
        ):
            raise RiskDataContractError(f"snapshot array contract mismatch: {name}")
        path = root / f"{name}.npy"
        if verify_bulk and _sha256_file(path) != declared_digest:
            raise RiskDataContractError(f"snapshot array checksum mismatch: {name}")
        try:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, TypeError, ValueError) as exc:
            raise RiskDataContractError(
                f"snapshot array decode failed: {name}"
            ) from exc
        if values.dtype != np.float32 or values.shape != expected_shape:
            raise RiskDataContractError(f"snapshot array shape/dtype mismatch: {name}")
        if verify_bulk and not np.isfinite(values).all():
            raise RiskDataContractError(f"snapshot array contains NaN/Inf: {name}")
        loaded[name] = values
        normalized_arrays[name] = {
            "sha256": declared_digest,
            "shape": list(expected_shape),
            "dtype": "float32",
        }

    shard_entry = identity.get("sample_shard_indices")
    if not isinstance(shard_entry, Mapping) or set(
        shard_entry
    ) != _ARRAY_DECLARATION_KEYS:
        raise RiskDataContractError(
            "snapshot sample shard index declaration mismatch"
        )
    shard_digest = _require_sha256_digest(
        shard_entry.get("sha256"),
        label="snapshot sample shard index digest",
    )
    expected_shard_shape = (len(sample_ids),)
    if (
        shard_entry.get("dtype") != "int32"
        or shard_entry.get("shape") != list(expected_shard_shape)
    ):
        raise RiskDataContractError("snapshot sample shard index layout mismatch")
    shard_path = root / _SHARD_INDICES_NAME
    if verify_bulk and _sha256_file(shard_path) != shard_digest:
        raise RiskDataContractError("snapshot sample shard index checksum mismatch")
    try:
        sample_shard_indices = np.load(
            shard_path,
            mmap_mode="r",
            allow_pickle=False,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RiskDataContractError(
            "snapshot sample shard index decode failed"
        ) from exc
    if (
        sample_shard_indices.dtype != np.int32
        or sample_shard_indices.shape != expected_shard_shape
    ):
        raise RiskDataContractError("snapshot sample shard index layout mismatch")
    if verify_bulk:
        shard_counts = tuple(source_identity["source_shard_sample_counts"])
        if (
            bool(np.any(sample_shard_indices < 0))
            or bool(np.any(sample_shard_indices >= len(shard_counts)))
            or tuple(
                int(value)
                for value in np.bincount(
                    np.asarray(sample_shard_indices),
                    minlength=len(shard_counts),
                )
            )
            != shard_counts
        ):
            raise RiskDataContractError(
                "snapshot shard index values differ from source shard counts"
            )

    normalized_identity = {
        **source_identity,
        "sample_ids": {
            "sha256": sample_ids_digest,
            "count": len(sample_ids),
        },
        "arrays": normalized_arrays,
        "sample_shard_indices": {
            "sha256": shard_digest,
            "shape": list(expected_shard_shape),
            "dtype": "int32",
        },
    }
    if dict(identity) != normalized_identity:
        raise RiskDataContractError("snapshot identity declaration mismatch")
    digest = _sha256_bytes(_canonical_json_bytes(normalized_identity))
    if expected_snapshot_digest is not None and digest != expected_snapshot_digest:
        raise RiskDataContractError("snapshot descriptor digest mismatch")
    if manifest.get("snapshot_digest_sha256") != digest:
        raise RiskDataContractError("snapshot digest mismatch")
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256 != expected_manifest_sha256
    ):
        raise RiskDataContractError("snapshot descriptor manifest digest mismatch")
    if marker != {
        "layout_version": RISK_TRAINING_SNAPSHOT_LAYOUT_VERSION,
        "snapshot_digest_sha256": digest,
        "snapshot_manifest_sha256": manifest_sha256,
    }:
        raise RiskDataContractError("snapshot completion marker mismatch")
    return AuthenticatedRiskSnapshot(
        root=root,
        snapshot_digest_sha256=digest,
        snapshot_manifest_sha256=manifest_sha256,
        source_identity=source_identity,
        sample_ids=sample_ids,
        split=str(source_identity["split"]),
        provenance={
            "schema_version": SCHEMA_VERSION,
            "risk_dataset_manifest_digest": source_identity[
                "risk_dataset_manifest_digest"
            ],
            "snapshot_digest_sha256": digest,
        },
        model_inputs={name: loaded[name] for name in MODEL_INPUT_KEYS},
        targets={name: loaded[name] for name in TARGET_KEYS},
        sample_shard_indices=sample_shard_indices,
        _sample_index={sample_id: row for row, sample_id in enumerate(sample_ids)},
    )


class _RiskSnapshotMaterializer:
    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root).expanduser().absolute()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.staging: Path | None = None
        self.total_sample_count: int | None = None
        self.offset = 0
        self.sample_ids: list[str] = []
        self.values: dict[str, np.memmap] = {}
        self.sample_shard_indices: np.memmap | None = None

    def _initialize(self, total_sample_count: int, loaded: LoadedRiskShard) -> None:
        self.total_sample_count = total_sample_count
        self.staging = Path(
            tempfile.mkdtemp(prefix=".risk-snapshot.staging-", dir=self.cache_root)
        )
        first = loaded.samples[0]
        for name in MODEL_INPUT_KEYS:
            value = np.asarray(getattr(first, name), dtype=np.float32)
            self.values[name] = np.lib.format.open_memmap(
                self.staging / f"{name}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(total_sample_count, *value.shape),
            )
        for name in TARGET_KEYS:
            self.values[name] = np.lib.format.open_memmap(
                self.staging / f"{name}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(total_sample_count,),
            )
        self.sample_shard_indices = np.lib.format.open_memmap(
            self.staging / _SHARD_INDICES_NAME,
            mode="w+",
            dtype=np.int32,
            shape=(total_sample_count,),
        )

    def consume(
        self,
        total_sample_count: int,
        grid: object,
        descriptor: RiskShardDescriptor,
        loaded: LoadedRiskShard,
    ) -> None:
        del grid
        if not isinstance(descriptor, RiskShardDescriptor) or not isinstance(
            loaded, LoadedRiskShard
        ):
            raise RiskDataContractError("snapshot consumer received an invalid shard")
        if not loaded.samples:
            raise RiskDataContractError("snapshot consumer received an empty shard")
        if self.staging is None:
            self._initialize(total_sample_count, loaded)
        if self.total_sample_count != total_sample_count:
            raise RiskDataContractError("snapshot source sample count changed during decode")
        assert self.sample_shard_indices is not None
        for start in range(0, len(loaded.samples), 64):
            samples = loaded.samples[start : start + 64]
            count = len(samples)
            end = self.offset + count
            if end > total_sample_count:
                raise RiskDataContractError("snapshot source decoded too many samples")
            for name in MODEL_INPUT_KEYS:
                values = np.stack(
                    [np.asarray(getattr(sample, name), dtype=np.float32) for sample in samples]
                )
                if not np.isfinite(values).all():
                    raise RiskDataContractError(f"snapshot source contains NaN/Inf: {name}")
                self.values[name][self.offset:end] = values
            for name in TARGET_KEYS:
                values = np.asarray(
                    [getattr(sample, name) for sample in samples], dtype=np.float32
                )
                if not np.isfinite(values).all():
                    raise RiskDataContractError(f"snapshot source contains NaN/Inf: {name}")
                self.values[name][self.offset:end] = values
            self.sample_shard_indices[self.offset:end] = descriptor.shard_index
            self.sample_ids.extend(sample.sample_id for sample in samples)
            self.offset = end

    def abort(self) -> None:
        self.values = {}
        self.sample_shard_indices = None
        if self.staging is not None and self.staging.exists():
            shutil.rmtree(self.staging)
        self.staging = None

    def finalize(self, dataset: LoadedRiskDataset) -> AuthenticatedRiskSnapshot:
        if self.staging is None or self.sample_shard_indices is None:
            raise RiskDataContractError("snapshot materializer received no authenticated shards")
        if (
            self.offset != dataset.sample_count
            or len(self.sample_ids) != dataset.sample_count
            or len(set(self.sample_ids)) != dataset.sample_count
        ):
            raise RiskDataContractError(
                "strict snapshot source decode did not cover each sample once"
            )
        source_identity = _snapshot_identity(dataset)
        request_digest = _sha256_bytes(_canonical_json_bytes(source_identity))
        root = self.cache_root / request_digest
        for array in self.values.values():
            array.flush()
        self.sample_shard_indices.flush()
        sample_ids_path = self.staging / _IDS_NAME
        sample_ids_path.write_bytes(_canonical_json_bytes(self.sample_ids))
        array_manifest = {
            name: {
                "sha256": _sha256_file(self.staging / f"{name}.npy"),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
            for name, array in self.values.items()
        }
        shard_index_manifest = {
            "sha256": _sha256_file(self.staging / _SHARD_INDICES_NAME),
            "dtype": str(self.sample_shard_indices.dtype),
            "shape": list(self.sample_shard_indices.shape),
        }
        identity = {
            **source_identity,
            "sample_ids": {
                "sha256": _sha256_file(sample_ids_path),
                "count": len(self.sample_ids),
            },
            "arrays": array_manifest,
            "sample_shard_indices": shard_index_manifest,
        }
        digest = _sha256_bytes(_canonical_json_bytes(identity))
        manifest = {
            "snapshot_manifest_layout_version": (
                RISK_TRAINING_SNAPSHOT_LAYOUT_VERSION
            ),
            "source_identity": source_identity,
            "identity": identity,
            "snapshot_digest_sha256": digest,
        }
        manifest_path = self.staging / _MANIFEST_NAME
        manifest_path.write_bytes(_canonical_json_bytes(manifest))
        for path in (
            *(self.staging / f"{name}.npy" for name in (*MODEL_INPUT_KEYS, *TARGET_KEYS)),
            self.staging / _SHARD_INDICES_NAME,
            sample_ids_path,
            manifest_path,
        ):
            _fsync_file(path)
        marker_path = self.staging / _COMPLETE_NAME
        marker_path.write_bytes(
            _canonical_json_bytes(
                {
                    "layout_version": RISK_TRAINING_SNAPSHOT_LAYOUT_VERSION,
                    "snapshot_digest_sha256": digest,
                    "snapshot_manifest_sha256": _sha256_file(manifest_path),
                }
            )
        )
        _fsync_file(marker_path)
        _fsync_directory(self.staging)
        staging = self.staging
        self.values = {}
        self.sample_shard_indices = None
        published = False
        if root.exists():
            _discard_staging_if_matching_winner(staging, root)
        else:
            try:
                atomic_rename_noreplace(staging, root)
                published = True
            except FileExistsError:
                _discard_staging_if_matching_winner(staging, root)
        if published:
            _fsync_directory(self.cache_root)
        self.staging = None
        return _open_snapshot(root, expected_source_identity=source_identity)


def open_authenticated_risk_snapshot(
    dataset: LoadedRiskDataset, *, cache_root: str | Path
) -> AuthenticatedRiskSnapshot:
    """Materialize one already authenticated dataset into read-only memory maps."""

    loaded_dataset = risk_dataloader_module._validate_loaded_risk_dataset(dataset)
    identity = _snapshot_identity(loaded_dataset)
    root = Path(cache_root).expanduser().absolute() / _sha256_bytes(
        _canonical_json_bytes(identity)
    )
    if root.exists():
        return _open_snapshot(root, expected_source_identity=identity)
    materializer = _RiskSnapshotMaterializer(cache_root)
    try:
        for descriptor in loaded_dataset.shards:
            loaded = risk_dataloader_module._load_validated_production_shard(
                loaded_dataset, descriptor
            )
            materializer.consume(
                loaded_dataset.sample_count,
                loaded_dataset.grid,
                descriptor,
                loaded,
            )
        return materializer.finalize(loaded_dataset)
    except BaseException:
        materializer.abort()
        raise


def load_authenticated_risk_snapshot(
    seal_root: str | Path,
    *,
    collection_root: str | Path,
    expected_split: str,
    cache_root: str | Path,
    expected_manifest_digest: str | None = None,
) -> tuple[LoadedRiskDataset, AuthenticatedRiskSnapshot]:
    """Authenticate source shards once while materializing a local snapshot."""

    materializer = _RiskSnapshotMaterializer(cache_root)
    try:
        dataset = _load_risk_dataset_seal_with_consumers(
            seal_root,
            collection_root=collection_root,
            expected_split=expected_split,
            expected_manifest_digest=expected_manifest_digest,
            authenticated_shard_consumer=materializer.consume,
        )
        return dataset, materializer.finalize(dataset)
    except BaseException:
        materializer.abort()
        raise


def open_authenticated_risk_snapshot_descriptor(
    descriptor: Mapping[str, object],
) -> AuthenticatedRiskSnapshot:
    """Open rank-zero-authenticated memory maps without touching source shards."""

    expected_keys = {
        "descriptor_layout_version",
        "root",
        "source_identity",
        "snapshot_digest_sha256",
        "snapshot_manifest_sha256",
        "sample_count",
        "split",
    }
    if not isinstance(descriptor, Mapping) or set(descriptor) != expected_keys:
        raise RiskDataContractError("snapshot descriptor fields mismatch")
    if descriptor.get("descriptor_layout_version") != (
        RISK_SNAPSHOT_DESCRIPTOR_LAYOUT_VERSION
    ):
        raise RiskDataContractError("snapshot descriptor layout mismatch")
    root_value = descriptor["root"]
    identity = descriptor["source_identity"]
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise RiskDataContractError("snapshot descriptor root must be absolute")
    if not isinstance(identity, Mapping):
        raise RiskDataContractError("snapshot descriptor source identity is invalid")
    digest = _require_sha256_digest(
        descriptor["snapshot_digest_sha256"],
        label="snapshot descriptor digest",
    )
    manifest_digest = _require_sha256_digest(
        descriptor["snapshot_manifest_sha256"],
        label="snapshot descriptor manifest digest",
    )
    snapshot = _open_snapshot(
        Path(root_value),
        expected_source_identity=dict(identity),
        expected_snapshot_digest=digest,
        expected_manifest_sha256=manifest_digest,
        verify_bulk=False,
    )
    if descriptor["sample_count"] != len(snapshot.sample_ids):
        raise RiskDataContractError("snapshot descriptor sample count mismatch")
    if descriptor["split"] != snapshot.split:
        raise RiskDataContractError("snapshot descriptor split mismatch")
    return snapshot


def _require_sha256_digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RiskDataContractError(f"{label} must be a lowercase SHA256 digest")
    return value


def _require_blake2b128_digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RiskDataContractError(
            f"{label} must be a lowercase BLAKE2b-128 digest"
        )
    return value


def _require_real_directory(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise RiskDataContractError(f"{label} is missing") from exc
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise RiskDataContractError(f"{label} must be a real directory")


def _require_exact_entries(
    root: Path,
    *,
    expected: set[str],
    label: str,
) -> None:
    _require_real_directory(root, label=label)
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise RiskDataContractError(f"failed to enumerate {label}") from exc
    actual = {entry.name for entry in entries}
    if actual != expected:
        raise RiskDataContractError(
            f"{label} has missing/unexpected entries: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    if any(entry.is_symlink() for entry in entries):
        raise RiskDataContractError(f"{label} forbids symlink entries")


def _read_canonical_snapshot_json(path: Path, *, label: str) -> dict[str, object]:
    _require_regular(path)
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RiskDataContractError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RiskDataContractError(f"{label} must be a JSON object")
    if payload != _canonical_json_bytes(value):
        raise RiskDataContractError(f"{label} must be canonical JSON")
    return value


def _validate_risk_source_identity(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _RISK_SOURCE_IDENTITY_KEYS:
        raise RiskDataContractError("risk snapshot source identity fields mismatch")
    if value.get("layout_version") != RISK_TRAINING_SNAPSHOT_LAYOUT_VERSION:
        raise RiskDataContractError("risk snapshot source layout version mismatch")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise RiskDataContractError("risk snapshot source schema_version mismatch")
    _require_blake2b128_digest(
        value.get("g1_split_manifest_digest"),
        label="risk snapshot source G1 split digest",
    )
    _require_sha256_digest(
        value.get("risk_dataset_manifest_digest"),
        label="risk snapshot source dataset digest",
    )
    _require_sha256_digest(
        value.get("dynamic_objects_config_digest"),
        label="risk snapshot source dynamic-object config digest",
    )
    _require_blake2b128_digest(
        value.get("target_type_policy_digest"),
        label="risk snapshot source target policy digest",
    )
    sample_count = value.get("sample_count")
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 1
    ):
        raise RiskDataContractError(
            "risk snapshot source sample_count must be positive"
        )
    if value.get("split") not in {"train", "calibration", "val", "test"}:
        raise RiskDataContractError("risk snapshot source split is invalid")
    grid = value.get("grid")
    if not isinstance(grid, Mapping) or set(grid) != _RISK_GRID_KEYS:
        raise RiskDataContractError("risk snapshot source grid fields mismatch")
    for name in ("height", "width", "history_steps", "future_steps"):
        dimension = grid.get(name)
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 1
        ):
            raise RiskDataContractError(
                f"risk snapshot source grid.{name} must be positive"
            )
    for name in ("resolution_m", "sample_dt_s"):
        scalar = grid.get(name)
        if (
            not isinstance(scalar, (int, float))
            or isinstance(scalar, bool)
            or not math.isfinite(float(scalar))
            or float(scalar) <= 0.0
        ):
            raise RiskDataContractError(
                f"risk snapshot source grid.{name} must be finite and positive"
            )
    risk_dataloader_module._validate_production_channel_spec(
        value.get("channel_spec")
    )
    digests = value.get("source_shard_semantic_digests")
    counts = value.get("source_shard_sample_counts")
    if (
        not isinstance(digests, list)
        or not isinstance(counts, list)
        or not digests
        or len(digests) != len(counts)
    ):
        raise RiskDataContractError(
            "risk snapshot source shard declarations are invalid"
        )
    for index, digest in enumerate(digests):
        _require_sha256_digest(
            digest,
            label=f"risk snapshot source shard digest {index}",
        )
    if any(
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
        for count in counts
    ) or sum(counts) != sample_count:
        raise RiskDataContractError(
            "risk snapshot source shard counts differ from sample_count"
        )
    canonical = _canonical_json_value(dict(value))
    if not isinstance(canonical, dict):  # pragma: no cover
        raise RuntimeError("risk source identity canonicalization failed")
    return canonical


def _risk_snapshot_array_shapes(
    source_identity: Mapping[str, object],
    *,
    sample_count: int,
) -> dict[str, tuple[int, ...]]:
    grid = source_identity["grid"]
    if not isinstance(grid, Mapping):  # pragma: no cover
        raise RuntimeError("validated risk source identity lost its grid")
    height = int(grid["height"])
    width = int(grid["width"])
    history_steps = int(grid["history_steps"])
    return {
        "bev_history": (
            sample_count,
            history_steps,
            len(HISTORY_CHANNELS),
            height,
            width,
        ),
        "state_channels": (
            sample_count,
            len(STATE_CHANNELS),
            height,
            width,
        ),
        "trajectory_channels": (
            sample_count,
            len(TRAJECTORY_CHANNELS),
            height,
            width,
        ),
        "robot_state": (sample_count, ROBOT_STATE_DIM),
        **{name: (sample_count,) for name in TARGET_KEYS},
    }


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _published_snapshot_identity(root: Path) -> tuple[str, dict[str, object]]:
    _require_real_directory(root, label="snapshot publication root")
    manifest = _read_canonical_snapshot_json(
        root / _MANIFEST_NAME,
        label="snapshot publication manifest",
    )
    if set(manifest) != _SNAPSHOT_MANIFEST_KEYS:
        raise RiskDataContractError("snapshot publication manifest fields mismatch")
    if not isinstance(manifest.get("source_identity"), Mapping):
        raise RiskDataContractError(
            "snapshot publication source identity is invalid"
        )
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise RiskDataContractError("snapshot publication identity is invalid")
    declared_digest = _require_sha256_digest(
        manifest.get("snapshot_digest_sha256"),
        label="snapshot publication digest",
    )
    canonical_identity = _canonical_json_value(dict(identity))
    if not isinstance(canonical_identity, dict):  # pragma: no cover
        raise RuntimeError("snapshot publication identity canonicalization failed")
    if declared_digest != _sha256_bytes(
        _canonical_json_bytes(canonical_identity)
    ):
        raise RiskDataContractError("snapshot publication identity digest mismatch")
    return declared_digest, canonical_identity


def _discard_staging_if_matching_winner(staging: Path, winner: Path) -> None:
    staged_digest, staged_identity = _published_snapshot_identity(staging)
    winner_digest, winner_identity = _published_snapshot_identity(winner)
    if (
        staged_digest != winner_digest
        or staged_identity != winner_identity
    ):
        raise RiskDataContractError(
            "snapshot winner identity differs from staged snapshot"
        )
    shutil.rmtree(staging)


def _occupancy_source_identity(
    dataset: LoadedRiskDataset,
    risk_snapshot: AuthenticatedRiskSnapshot,
) -> dict[str, object]:
    section = dataset.manifest.get("occupancy_sidecars")
    if not isinstance(section, Mapping):
        raise RiskDataContractError(
            "dataset seal does not contain an occupancy_sidecars publication"
        )
    raw_shards = section.get("shards")
    query_geometry = section.get("query_geometry")
    grid = dataset.manifest.get("grid")
    if (
        not isinstance(raw_shards, (list, tuple))
        or not isinstance(query_geometry, Mapping)
        or not isinstance(grid, Mapping)
    ):
        raise RiskDataContractError("occupancy snapshot source metadata is invalid")
    if len(raw_shards) != len(dataset.shards):
        raise RiskDataContractError("occupancy snapshot sidecar shard count mismatch")
    sidecar_semantic_digests: list[str] = []
    pair_marker_digests: list[str] = []
    for index, raw_shard in enumerate(raw_shards):
        if not isinstance(raw_shard, Mapping):
            raise RiskDataContractError(
                f"occupancy sidecar descriptor {index} is invalid"
            )
        sidecar_semantic_digests.append(
            _require_sha256_digest(
                raw_shard.get("sidecar_semantic_digest"),
                label=f"occupancy sidecar semantic digest {index}",
            )
        )
        pair_marker_digests.append(
            _require_sha256_digest(
                raw_shard.get("pair_marker_digest_sha256"),
                label=f"occupancy pair marker digest {index}",
            )
        )
    identity = {
        "layout_version": OCCUPANCY_TRAINING_SNAPSHOT_LAYOUT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "cache_layout": {
            "query_inputs": [
                "robot_endpoint_footprints",
                "endpoint_times_s",
            ],
            "occupancy_targets": ["hidden_risk_occupancy"],
        },
        "risk_snapshot_digest_sha256": _require_sha256_digest(
            risk_snapshot.snapshot_digest_sha256,
            label="risk snapshot digest",
        ),
        "risk_dataset_manifest_digest": _require_sha256_digest(
            dataset.risk_dataset_manifest_digest,
            label="risk dataset manifest digest",
        ),
        "sample_count": dataset.sample_count,
        "shard_sample_counts": [
            descriptor.sample_count for descriptor in dataset.shards
        ],
        "split": dataset.split,
        "grid": dict(grid),
        "occupancy_sidecar_collection_digest_sha256": _require_sha256_digest(
            section.get("collection_digest_sha256"),
            label="occupancy sidecar collection digest",
        ),
        "ordered_sidecar_semantic_digests": sidecar_semantic_digests,
        "pair_marker_digests_sha256": pair_marker_digests,
        "query_geometry": dict(query_geometry),
    }
    canonical = _canonical_json_value(identity)
    if not isinstance(canonical, dict):  # pragma: no cover
        raise RuntimeError("occupancy source identity canonicalization failed")
    return canonical


def _occupancy_cache_root(
    cache_root: Path,
    *,
    source_identity: Mapping[str, object],
) -> Path:
    request_digest = _sha256_bytes(_canonical_json_bytes(dict(source_identity)))
    return cache_root / f"occupancy-{request_digest}"


@dataclass(frozen=True)
class AuthenticatedOccupancySnapshot:
    """Immutable occupancy queries and targets composed with one risk snapshot."""

    root: Path
    snapshot_digest_sha256: str
    snapshot_manifest_sha256: str
    source_identity: Mapping[str, object]
    risk_snapshot: AuthenticatedRiskSnapshot = field(repr=False, compare=False)
    sample_ids: tuple[str, ...]
    split: str
    provenance: Mapping[str, object]
    query_inputs: Mapping[str, np.ndarray]
    occupancy_targets: Mapping[str, np.ndarray]
    _sample_index: Mapping[str, int] = field(repr=False, compare=False)

    def select_subset(self, *, max_samples: int, seed: int) -> ProductionRiskSubset:
        return self.risk_snapshot.select_subset(max_samples=max_samples, seed=seed)

    def _batch_from_risk_batch(self, risk_batch: RiskBatch) -> ProductionOccupancyBatch:
        rows = [self._sample_index[sample_id] for sample_id in risk_batch.sample_ids]
        robot_masks = np.array(
            self.query_inputs["robot_endpoint_footprints"][rows],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        endpoint_times = np.array(
            self.query_inputs["endpoint_times_s"],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        hidden = np.array(
            self.occupancy_targets["hidden_risk_occupancy"][rows],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        return ProductionOccupancyBatch(
            model_inputs=dict(risk_batch.model_inputs),
            targets=dict(risk_batch.targets),
            query_inputs={
                "robot_endpoint_footprints": torch.from_numpy(robot_masks),
                "endpoint_times_s": torch.from_numpy(endpoint_times),
            },
            occupancy_targets={
                "hidden_risk_occupancy": torch.from_numpy(hidden),
            },
            sample_ids=risk_batch.sample_ids,
            split=risk_batch.split,
            provenance={
                **dict(risk_batch.provenance),
                "loader_mode": "authenticated_occupancy_snapshot",
                "occupancy_snapshot_digest_sha256": (
                    self.snapshot_digest_sha256
                ),
                "occupancy_sidecar_collection_digest_sha256": (
                    self.source_identity[
                        "occupancy_sidecar_collection_digest_sha256"
                    ]
                ),
                "occupancy_query_layout_version": self.source_identity[
                    "query_geometry"
                ]["query_layout_version"],
            },
        )

    def batch(self, sample_ids: Sequence[str]) -> ProductionOccupancyBatch:
        return self._batch_from_risk_batch(self.risk_snapshot.batch(sample_ids))

    def iter_batches(
        self,
        *,
        subset: ProductionRiskSubset,
        batch_size: int,
        seed: int,
        epoch: int,
        start_cursor: OccupancyStreamCursor | None = None,
    ) -> Iterator[tuple[ProductionOccupancyBatch, OccupancyStreamCursor]]:
        if start_cursor is None:
            risk_start_cursor = None
        else:
            if not isinstance(start_cursor, OccupancyStreamCursor):
                raise RiskDataContractError(
                    "occupancy start_cursor must be an OccupancyStreamCursor"
                )
            if start_cursor.sidecar_collection_digest_sha256 != (
                self.source_identity[
                    "occupancy_sidecar_collection_digest_sha256"
                ]
            ):
                raise RiskDataContractError(
                    "occupancy cursor sidecar collection digest mismatch"
                )
            if start_cursor.query_layout_version != self.source_identity[
                "query_geometry"
            ]["query_layout_version"]:
                raise RiskDataContractError("occupancy cursor query layout mismatch")
            risk_start_cursor = start_cursor.risk_cursor
        for risk_batch, risk_cursor in self.risk_snapshot.iter_batches(
            subset=subset,
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
            start_cursor=risk_start_cursor,
        ):
            yield (
                self._batch_from_risk_batch(risk_batch),
                OccupancyStreamCursor(
                    risk_cursor=risk_cursor,
                    sidecar_collection_digest_sha256=str(
                        self.source_identity[
                            "occupancy_sidecar_collection_digest_sha256"
                        ]
                    ),
                    query_layout_version=str(
                        self.source_identity["query_geometry"][
                            "query_layout_version"
                        ]
                    ),
                ),
            )


def _open_occupancy_snapshot(
    root: Path,
    *,
    dataset: LoadedRiskDataset,
    risk_snapshot: AuthenticatedRiskSnapshot,
    expected_source_identity: Mapping[str, object],
) -> AuthenticatedOccupancySnapshot:
    _require_exact_entries(
        root,
        expected={
            _MANIFEST_NAME,
            _COMPLETE_NAME,
            _OCCUPANCY_QUERY_DIRECTORY,
            _OCCUPANCY_TARGET_DIRECTORY,
        },
        label="occupancy snapshot root",
    )
    query_root = root / _OCCUPANCY_QUERY_DIRECTORY
    target_root = root / _OCCUPANCY_TARGET_DIRECTORY
    _require_exact_entries(
        query_root,
        expected={_ROBOT_ENDPOINT_FOOTPRINTS_NAME, _ENDPOINT_TIMES_NAME},
        label="occupancy query_inputs directory",
    )
    _require_exact_entries(
        target_root,
        expected={_HIDDEN_RISK_OCCUPANCY_NAME},
        label="occupancy targets directory",
    )
    manifest_path = root / _MANIFEST_NAME
    marker_path = root / _COMPLETE_NAME
    manifest = _read_canonical_snapshot_json(
        manifest_path,
        label="occupancy snapshot manifest",
    )
    marker = _read_canonical_snapshot_json(
        marker_path,
        label="occupancy snapshot completion marker",
    )
    if set(manifest) != {
        "snapshot_manifest_layout_version",
        "source_identity",
        "identity",
        "snapshot_digest_sha256",
    }:
        raise RiskDataContractError("occupancy snapshot manifest fields mismatch")
    if manifest.get("snapshot_manifest_layout_version") != (
        OCCUPANCY_TRAINING_SNAPSHOT_LAYOUT_VERSION
    ):
        raise RiskDataContractError("occupancy snapshot manifest version mismatch")
    if manifest.get("source_identity") != dict(expected_source_identity):
        raise RiskDataContractError(
            "occupancy snapshot source identity does not match authenticated source"
        )
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise RiskDataContractError("occupancy snapshot identity is missing")
    array_entries = identity.get("arrays")
    expected_paths = {
        "query_inputs/robot_endpoint_footprints.npy": (
            query_root / _ROBOT_ENDPOINT_FOOTPRINTS_NAME,
            (dataset.sample_count, 15, dataset.grid.height, dataset.grid.width),
        ),
        "query_inputs/endpoint_times_s.npy": (
            query_root / _ENDPOINT_TIMES_NAME,
            (15,),
        ),
        "targets/hidden_risk_occupancy.npy": (
            target_root / _HIDDEN_RISK_OCCUPANCY_NAME,
            (dataset.sample_count, 15, dataset.grid.height, dataset.grid.width),
        ),
    }
    if not isinstance(array_entries, Mapping) or set(array_entries) != set(
        expected_paths
    ):
        raise RiskDataContractError("occupancy snapshot array entries mismatch")
    loaded_arrays: dict[str, np.ndarray] = {}
    normalized_entries: dict[str, dict[str, object]] = {}
    for relative_path, (path, expected_shape) in expected_paths.items():
        entry = array_entries[relative_path]
        if not isinstance(entry, Mapping) or set(entry) != {
            "sha256",
            "shape",
            "dtype",
        }:
            raise RiskDataContractError(
                f"occupancy snapshot array declaration mismatch: {relative_path}"
            )
        declared_digest = _require_sha256_digest(
            entry.get("sha256"),
            label=f"occupancy snapshot array digest {relative_path}",
        )
        _require_regular(path)
        if _sha256_file(path) != declared_digest:
            raise RiskDataContractError(
                f"occupancy snapshot array checksum mismatch: {relative_path}"
            )
        try:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, TypeError, ValueError) as exc:
            raise RiskDataContractError(
                f"occupancy snapshot array decode failed: {relative_path}"
            ) from exc
        if (
            values.dtype != np.float32
            or values.shape != expected_shape
            or entry.get("dtype") != "float32"
            or entry.get("shape") != list(expected_shape)
        ):
            raise RiskDataContractError(
                f"occupancy snapshot array shape/dtype mismatch: {relative_path}"
            )
        if not np.isfinite(values).all():
            raise RiskDataContractError(
                f"occupancy snapshot array contains NaN/Inf: {relative_path}"
            )
        loaded_arrays[relative_path] = values
        normalized_entries[relative_path] = {
            "sha256": declared_digest,
            "shape": list(expected_shape),
            "dtype": "float32",
        }
    for relative_path in (
        "query_inputs/robot_endpoint_footprints.npy",
        "targets/hidden_risk_occupancy.npy",
    ):
        if not np.isin(loaded_arrays[relative_path], (0.0, 1.0)).all():
            raise RiskDataContractError(
                f"occupancy snapshot masks must be binary: {relative_path}"
            )
    query_geometry = expected_source_identity.get("query_geometry")
    if not isinstance(query_geometry, Mapping):
        raise RiskDataContractError("occupancy snapshot query geometry is invalid")
    expected_endpoints = (
        risk_dataloader_module.production_endpoint_times_from_query_geometry(
            query_geometry
        )
    )
    if not np.array_equal(
        loaded_arrays["query_inputs/endpoint_times_s.npy"],
        expected_endpoints,
    ):
        raise RiskDataContractError("occupancy snapshot endpoint times mismatch")
    expected_identity = {
        **dict(expected_source_identity),
        "arrays": normalized_entries,
    }
    if dict(identity) != expected_identity:
        raise RiskDataContractError("occupancy snapshot identity mismatch")
    snapshot_digest = _require_sha256_digest(
        manifest.get("snapshot_digest_sha256"),
        label="occupancy snapshot digest",
    )
    if snapshot_digest != _sha256_bytes(_canonical_json_bytes(expected_identity)):
        raise RiskDataContractError("occupancy snapshot digest mismatch")
    manifest_sha256 = _sha256_file(manifest_path)
    expected_marker = {
        "layout_version": OCCUPANCY_TRAINING_SNAPSHOT_LAYOUT_VERSION,
        "snapshot_digest_sha256": snapshot_digest,
        "snapshot_manifest_sha256": manifest_sha256,
    }
    if marker != expected_marker:
        raise RiskDataContractError("occupancy snapshot completion marker mismatch")
    return AuthenticatedOccupancySnapshot(
        root=root,
        snapshot_digest_sha256=snapshot_digest,
        snapshot_manifest_sha256=manifest_sha256,
        source_identity=dict(expected_source_identity),
        risk_snapshot=risk_snapshot,
        sample_ids=risk_snapshot.sample_ids,
        split=dataset.split,
        provenance={
            "schema_version": SCHEMA_VERSION,
            "risk_dataset_manifest_digest": dataset.risk_dataset_manifest_digest,
            "risk_snapshot_digest_sha256": risk_snapshot.snapshot_digest_sha256,
            "occupancy_snapshot_digest_sha256": snapshot_digest,
            "occupancy_sidecar_collection_digest_sha256": (
                expected_source_identity[
                    "occupancy_sidecar_collection_digest_sha256"
                ]
            ),
        },
        query_inputs={
            "robot_endpoint_footprints": loaded_arrays[
                "query_inputs/robot_endpoint_footprints.npy"
            ],
            "endpoint_times_s": loaded_arrays[
                "query_inputs/endpoint_times_s.npy"
            ],
        },
        occupancy_targets={
            "hidden_risk_occupancy": loaded_arrays[
                "targets/hidden_risk_occupancy.npy"
            ],
        },
        _sample_index={
            sample_id: row for row, sample_id in enumerate(risk_snapshot.sample_ids)
        },
    )


class _OccupancySnapshotMaterializer:
    def __init__(self, cache_root: str | Path) -> None:
        self.cache_root = Path(cache_root).expanduser().absolute()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        _require_real_directory(self.cache_root, label="snapshot cache root")
        self.staging: Path | None = None
        self.total_sample_count: int | None = None
        self.offset = 0
        self.sample_ids: list[str] = []
        self.shard_sample_counts: list[int] = []
        self.query_geometry: dict[str, object] | None = None
        self.robot_masks: np.memmap | None = None
        self.endpoint_times: np.memmap | None = None
        self.hidden_occupancy: np.memmap | None = None

    def _initialize(self, pair: _AuthenticatedRiskSidecarPair) -> None:
        self.total_sample_count = pair.total_sample_count
        canonical_geometry = _canonical_json_value(dict(pair.query_geometry))
        if not isinstance(canonical_geometry, dict):  # pragma: no cover
            raise RuntimeError("occupancy query geometry canonicalization failed")
        self.query_geometry = canonical_geometry
        self.staging = Path(
            tempfile.mkdtemp(
                prefix=".occupancy-snapshot.staging-",
                dir=self.cache_root,
            )
        )
        query_root = self.staging / _OCCUPANCY_QUERY_DIRECTORY
        target_root = self.staging / _OCCUPANCY_TARGET_DIRECTORY
        query_root.mkdir()
        target_root.mkdir()
        mask_shape = (
            pair.total_sample_count,
            15,
            pair.grid.height,
            pair.grid.width,
        )
        self.robot_masks = np.lib.format.open_memmap(
            query_root / _ROBOT_ENDPOINT_FOOTPRINTS_NAME,
            mode="w+",
            dtype=np.float32,
            shape=mask_shape,
        )
        self.endpoint_times = np.lib.format.open_memmap(
            query_root / _ENDPOINT_TIMES_NAME,
            mode="w+",
            dtype=np.float32,
            shape=(15,),
        )
        self.endpoint_times[:] = (
            risk_dataloader_module.production_endpoint_times_from_query_geometry(
                pair.query_geometry
            )
        )
        self.hidden_occupancy = np.lib.format.open_memmap(
            target_root / _HIDDEN_RISK_OCCUPANCY_NAME,
            mode="w+",
            dtype=np.float32,
            shape=mask_shape,
        )

    def consume(self, pair: _AuthenticatedRiskSidecarPair) -> None:
        if not isinstance(pair, _AuthenticatedRiskSidecarPair):
            raise RiskDataContractError(
                "occupancy snapshot consumer received an invalid pair"
            )
        if self.staging is None:
            self._initialize(pair)
        if (
            self.total_sample_count != pair.total_sample_count
            or self.query_geometry != _canonical_json_value(dict(pair.query_geometry))
            or pair.risk_descriptor.shard_index != len(self.shard_sample_counts)
        ):
            raise RiskDataContractError(
                "occupancy snapshot source changed during integrated decode"
            )
        assert self.robot_masks is not None
        assert self.endpoint_times is not None
        assert self.hidden_occupancy is not None
        risk_samples = pair.risk_shard.samples
        sample_ids = tuple(sample.sample_id for sample in risk_samples)
        if sample_ids != pair.sidecar_shard.sample_ids:
            raise RiskDataContractError("occupancy snapshot pair sample IDs mismatch")
        independent_endpoints = (
            risk_dataloader_module.production_endpoint_times_from_query_geometry(
                pair.query_geometry
            )
        )
        if not np.array_equal(
            independent_endpoints,
            pair.sidecar_shard.future_endpoint_times_s,
        ):
            raise RiskDataContractError(
                "independently derived endpoint times differ from sidecar cache"
            )
        reconstructed = np.stack(
            [
                risk_dataloader_module.reconstruct_production_robot_endpoint_footprints(
                    sample,
                    grid=pair.grid,
                    query_geometry=pair.query_geometry,
                )
                for sample in risk_samples
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        if not np.array_equal(
            reconstructed,
            pair.sidecar_shard.robot_future_footprints,
        ):
            raise RiskDataContractError(
                "Task4 robot footprint sidecar differs from query-only reconstruction"
            )
        hidden = np.asarray(
            pair.sidecar_shard.hidden_risk_occupancy,
            dtype=np.float32,
        )
        count = len(risk_samples)
        end = self.offset + count
        if end > pair.total_sample_count:
            raise RiskDataContractError(
                "occupancy snapshot source decoded too many samples"
            )
        if (
            reconstructed.shape
            != (count, 15, pair.grid.height, pair.grid.width)
            or hidden.shape != reconstructed.shape
            or not np.isfinite(reconstructed).all()
            or not np.isfinite(hidden).all()
            or not np.isin(reconstructed, (0.0, 1.0)).all()
            or not np.isin(hidden, (0.0, 1.0)).all()
        ):
            raise RiskDataContractError("occupancy snapshot pair arrays are invalid")
        self.robot_masks[self.offset:end] = reconstructed
        self.hidden_occupancy[self.offset:end] = hidden
        self.sample_ids.extend(sample_ids)
        self.shard_sample_counts.append(count)
        self.offset = end

    def abort(self) -> None:
        self.robot_masks = None
        self.endpoint_times = None
        self.hidden_occupancy = None
        if self.staging is not None and self.staging.exists():
            shutil.rmtree(self.staging)
        self.staging = None

    def finalize(
        self,
        dataset: LoadedRiskDataset,
        risk_snapshot: AuthenticatedRiskSnapshot,
    ) -> AuthenticatedOccupancySnapshot:
        if (
            self.staging is None
            or self.robot_masks is None
            or self.endpoint_times is None
            or self.hidden_occupancy is None
        ):
            raise RiskDataContractError(
                "occupancy snapshot materializer received no authenticated pairs"
            )
        if (
            self.offset != dataset.sample_count
            or tuple(self.sample_ids) != risk_snapshot.sample_ids
            or self.shard_sample_counts
            != [descriptor.sample_count for descriptor in dataset.shards]
        ):
            raise RiskDataContractError(
                "integrated occupancy decode did not cover each sample once"
            )
        source_identity = _occupancy_source_identity(dataset, risk_snapshot)
        root = _occupancy_cache_root(
            self.cache_root,
            source_identity=source_identity,
        )
        for array in (
            self.robot_masks,
            self.endpoint_times,
            self.hidden_occupancy,
        ):
            array.flush()
        array_paths = {
            "query_inputs/robot_endpoint_footprints.npy": (
                self.staging
                / _OCCUPANCY_QUERY_DIRECTORY
                / _ROBOT_ENDPOINT_FOOTPRINTS_NAME,
                self.robot_masks,
            ),
            "query_inputs/endpoint_times_s.npy": (
                self.staging / _OCCUPANCY_QUERY_DIRECTORY / _ENDPOINT_TIMES_NAME,
                self.endpoint_times,
            ),
            "targets/hidden_risk_occupancy.npy": (
                self.staging
                / _OCCUPANCY_TARGET_DIRECTORY
                / _HIDDEN_RISK_OCCUPANCY_NAME,
                self.hidden_occupancy,
            ),
        }
        array_manifest = {
            relative_path: {
                "sha256": _sha256_file(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            for relative_path, (path, array) in array_paths.items()
        }
        identity = {**source_identity, "arrays": array_manifest}
        snapshot_digest = _sha256_bytes(_canonical_json_bytes(identity))
        manifest = {
            "snapshot_manifest_layout_version": (
                OCCUPANCY_TRAINING_SNAPSHOT_LAYOUT_VERSION
            ),
            "source_identity": source_identity,
            "identity": identity,
            "snapshot_digest_sha256": snapshot_digest,
        }
        manifest_path = self.staging / _MANIFEST_NAME
        manifest_path.write_bytes(_canonical_json_bytes(manifest))
        for path, _ in array_paths.values():
            _fsync_file(path)
        _fsync_file(manifest_path)
        marker = {
            "layout_version": OCCUPANCY_TRAINING_SNAPSHOT_LAYOUT_VERSION,
            "snapshot_digest_sha256": snapshot_digest,
            "snapshot_manifest_sha256": _sha256_file(manifest_path),
        }
        marker_path = self.staging / _COMPLETE_NAME
        marker_path.write_bytes(_canonical_json_bytes(marker))
        _fsync_file(marker_path)
        _fsync_directory(self.staging / _OCCUPANCY_QUERY_DIRECTORY)
        _fsync_directory(self.staging / _OCCUPANCY_TARGET_DIRECTORY)
        _fsync_directory(self.staging)
        staging = self.staging
        self.robot_masks = None
        self.endpoint_times = None
        self.hidden_occupancy = None
        published = False
        if root.exists():
            _discard_staging_if_matching_winner(staging, root)
        else:
            try:
                atomic_rename_noreplace(staging, root)
                published = True
            except FileExistsError:
                _discard_staging_if_matching_winner(staging, root)
        if published:
            _fsync_directory(self.cache_root)
        self.staging = None
        return _open_occupancy_snapshot(
            root,
            dataset=dataset,
            risk_snapshot=risk_snapshot,
            expected_source_identity=source_identity,
        )


def load_authenticated_occupancy_snapshot(
    seal_root: str | Path,
    *,
    collection_root: str | Path,
    sidecar_root: str | Path,
    expected_split: str,
    cache_root: str | Path,
    expected_manifest_digest: str | None = None,
) -> tuple[LoadedRiskDataset, AuthenticatedOccupancySnapshot]:
    """Authenticate risk/sidecar pairs once while materializing local mmaps."""

    occupancy_materializer = _OccupancySnapshotMaterializer(cache_root)
    risk_materializer = _RiskSnapshotMaterializer(cache_root)
    try:
        dataset = _load_risk_dataset_seal_with_consumers(
            seal_root,
            collection_root=collection_root,
            expected_split=expected_split,
            expected_manifest_digest=expected_manifest_digest,
            sidecar_root=sidecar_root,
            authenticated_shard_consumer=risk_materializer.consume,
            authenticated_sidecar_pair_consumer=occupancy_materializer.consume,
        )
        risk_snapshot = risk_materializer.finalize(dataset)
        occupancy_snapshot = occupancy_materializer.finalize(
            dataset,
            risk_snapshot,
        )
        return dataset, occupancy_snapshot
    except BaseException:
        occupancy_materializer.abort()
        risk_materializer.abort()
        raise


def open_authenticated_occupancy_snapshot(
    dataset: LoadedRiskDataset,
    *,
    cache_root: str | Path,
) -> AuthenticatedOccupancySnapshot:
    """Reopen an accepted occupancy cache without accessing source shards."""

    loaded_dataset = risk_dataloader_module._validate_loaded_risk_dataset(dataset)
    cache_path = Path(cache_root).expanduser().absolute()
    _require_real_directory(cache_path, label="snapshot cache root")
    risk_identity = _snapshot_identity(loaded_dataset)
    risk_root = cache_path / _sha256_bytes(_canonical_json_bytes(risk_identity))
    if not risk_root.exists():
        raise RiskDataContractError("authenticated risk snapshot cache is missing")
    risk_snapshot = _open_snapshot(
        risk_root,
        expected_source_identity=risk_identity,
    )
    source_identity = _occupancy_source_identity(loaded_dataset, risk_snapshot)
    root = _occupancy_cache_root(
        cache_path,
        source_identity=source_identity,
    )
    return _open_occupancy_snapshot(
        root,
        dataset=loaded_dataset,
        risk_snapshot=risk_snapshot,
        expected_source_identity=source_identity,
    )
