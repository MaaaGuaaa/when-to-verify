"""Immutable NPZ/JSONL storage and deterministic loading for SOP13."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.contracts import GridSpec, SCHEMA_VERSION, VerificationSample
from src.datasets.verification_dataset import (
    audit_verification_groups,
    validate_verification_sample_for_publication,
)
from src.planning.verification_actions import VerificationActionLibrary
from src.utils.seeding import derive_seed


VERIFICATION_SHARD_LAYOUT_VERSION = "verification_shard_npz_jsonl_v1"
_PAYLOAD_NAME = "samples.npz"
_MANIFEST_NAME = "metadata.jsonl"
_SUMMARY_NAME = "summary.json"
_REQUIRED_FILES = frozenset({_PAYLOAD_NAME, _MANIFEST_NAME, _SUMMARY_NAME})
_NUMERIC_NAMES = (
    "bev_history",
    "state_channels",
    "trajectory_channels",
    "verification_fov_mask",
    "verification_action_vector",
    "value_target",
    "useful_target",
    "br_before",
    "post_risk",
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "layout_version",
        "shard_index",
        "row_index",
        "sample_id",
        "split",
        "base_state_id",
        "nominal_trajectory_id",
        "verification_action_id",
        "metadata",
    }
)
_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "layout_version",
        "shard_index",
        "split",
        "expected_sample_count",
        "boundary",
        "files",
        "checksums",
        "manifest_digest",
        "semantic_digest",
        "array_layout",
        "audit_report",
    }
)


@dataclass(frozen=True)
class LoadedVerificationShard:
    samples: tuple[VerificationSample, ...]
    manifest: tuple[dict[str, object], ...]
    manifest_digest: str
    semantic_digest: str
    action_counts: dict[str, int]
    audit_report: dict[str, object]
    summary: dict[str, object]


@dataclass(frozen=True)
class VerificationCollection:
    """Verified in-memory collection with deterministic epoch ordering."""

    samples: tuple[VerificationSample, ...]
    shard_semantic_digests: tuple[str, ...]
    audit_report: dict[str, object]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> VerificationSample:
        return self.samples[index]

    def ordered_indices(
        self,
        *,
        seed: int,
        epoch: int,
        shuffle: bool,
    ) -> tuple[int, ...]:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be boolean")
        indices = np.arange(len(self.samples), dtype=np.int64)
        if shuffle:
            rng = np.random.default_rng(
                derive_seed(
                    seed,
                    "verification-collection-order-v1",
                    epoch,
                    *self.shard_semantic_digests,
                )
            )
            indices = rng.permutation(indices)
        return tuple(int(value) for value in indices)

    def iter_batches(
        self,
        batch_size: int,
        *,
        seed: int,
        epoch: int,
        shuffle: bool,
    ) -> Iterator[tuple[VerificationSample, ...]]:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        indices = self.ordered_indices(seed=seed, epoch=epoch, shuffle=shuffle)
        for start in range(0, len(indices), batch_size):
            yield tuple(self.samples[index] for index in indices[start : start + batch_size])


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _strict_loads(payload: str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(payload, parse_constant=reject_constant)


def _canonical_copy(value: object, *, name: str) -> object:
    try:
        return _strict_loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite canonical JSON") from exc


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return ("\n".join(_canonical_json(row) for row in rows) + "\n").encode("utf-8")


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _domain_digest(domain: bytes, payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def _little_float32(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.dtype("<f4"))


def _numeric_arrays(samples: Sequence[VerificationSample]) -> dict[str, np.ndarray]:
    arrays = {
        "bev_history": _little_float32(
            np.stack([sample.bev_history for sample in samples], axis=0)
        ),
        "state_channels": _little_float32(
            np.stack([sample.state_channels for sample in samples], axis=0)
        ),
        "trajectory_channels": _little_float32(
            np.stack([sample.trajectory_channels for sample in samples], axis=0)
        ),
        "verification_fov_mask": _little_float32(
            np.stack([sample.verification_fov_mask for sample in samples], axis=0)
        ),
        "verification_action_vector": _little_float32(
            np.stack(
                [sample.verification_action_vector for sample in samples], axis=0
            )
        ),
        "value_target": np.ascontiguousarray(
            [sample.value_target for sample in samples], dtype=np.dtype("<f8")
        ),
        "useful_target": np.ascontiguousarray(
            [sample.useful_target for sample in samples], dtype=np.uint8
        ),
        "br_before": np.ascontiguousarray(
            [sample.br_before for sample in samples], dtype=np.dtype("<f8")
        ),
        "post_risk": np.ascontiguousarray(
            [sample.post_risk for sample in samples], dtype=np.dtype("<f8")
        ),
    }
    if any(array.dtype.kind not in "fiu" for array in arrays.values()):
        raise TypeError("verification NPZ arrays must be numeric")
    if any(not array.flags.c_contiguous for array in arrays.values()):
        raise RuntimeError("verification writer produced non-contiguous arrays")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("verification shard arrays must be finite")
    return arrays


def _array_layout(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        name: {
            "dtype": arrays[name].dtype.str,
            "shape": list(arrays[name].shape),
            "order": "C",
        }
        for name in _NUMERIC_NAMES
    }


def _manifest_row(
    sample: VerificationSample,
    *,
    shard_index: int,
    row_index: int,
) -> dict[str, object]:
    metadata = _canonical_copy(sample.metadata, name="sample metadata")
    if not isinstance(metadata, dict):
        raise TypeError("sample metadata must be a JSON object")
    return {
        "schema_version": SCHEMA_VERSION,
        "layout_version": VERIFICATION_SHARD_LAYOUT_VERSION,
        "shard_index": shard_index,
        "row_index": row_index,
        "sample_id": sample.sample_id,
        "split": sample.split,
        "base_state_id": sample.base_state_id,
        "nominal_trajectory_id": sample.nominal_trajectory_id,
        "verification_action_id": sample.verification_action_id,
        "metadata": metadata,
    }


def _semantic_digest(
    arrays: Mapping[str, np.ndarray],
    *,
    shard_index: int,
    split: str,
    rows: Sequence[Mapping[str, object]],
) -> str:
    header = {
        "schema_version": SCHEMA_VERSION,
        "layout_version": VERIFICATION_SHARD_LAYOUT_VERSION,
        "shard_index": shard_index,
        "split": split,
        "array_layout": _array_layout(arrays),
        "sample_ids": [row["sample_id"] for row in rows],
        "manifest_digest": _domain_digest(
            b"verification-manifest-v1\0", _jsonl_bytes(rows)
        ),
    }
    digest = hashlib.sha256()
    digest.update(b"verification-shard-semantic-v1\0")
    encoded = _canonical_json(header).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    for name in _NUMERIC_NAMES:
        name_bytes = name.encode("ascii")
        raw = arrays[name].tobytes(order="C")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _validate_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def write_verification_shard(
    samples: Sequence[VerificationSample],
    output_dir: str | Path,
    *,
    grid: GridSpec,
    library: VerificationActionLibrary,
    shard_index: int = 0,
    expected_sample_count: int,
) -> dict[str, Path]:
    """Publish one immutable shard only after a strict staging reload."""

    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    if not isinstance(library, VerificationActionLibrary):
        raise TypeError("library must be a VerificationActionLibrary")
    shard_index = _validate_integer(shard_index, name="shard_index")
    count = _validate_integer(
        expected_sample_count, name="expected_sample_count", minimum=1
    )
    ordered = tuple(sorted(samples, key=lambda sample: sample.sample_id))
    if len(ordered) != count:
        raise ValueError("expected_sample_count differs from the fixed shard boundary")
    for sample in ordered:
        validate_verification_sample_for_publication(
            sample, grid=grid, library=library
        )
    audit = audit_verification_groups(list(ordered), require_complete=True)
    splits = {sample.split for sample in ordered}
    if len(splits) != 1:
        raise ValueError("one verification shard may contain only one split")
    split = next(iter(splits))
    rows = tuple(
        _manifest_row(sample, shard_index=shard_index, row_index=index)
        for index, sample in enumerate(ordered)
    )
    manifest_bytes = _jsonl_bytes(rows)
    manifest_digest = _domain_digest(b"verification-manifest-v1\0", manifest_bytes)
    arrays = _numeric_arrays(ordered)
    semantic_digest = _semantic_digest(
        arrays, shard_index=shard_index, split=split, rows=rows
    )

    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable shard: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=output_path.parent)
    )
    try:
        payload_path = staging / _PAYLOAD_NAME
        with payload_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        payload_bytes = payload_path.read_bytes()
        (staging / _MANIFEST_NAME).write_bytes(manifest_bytes)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "layout_version": VERIFICATION_SHARD_LAYOUT_VERSION,
            "shard_index": shard_index,
            "split": split,
            "expected_sample_count": count,
            "boundary": {
                "first_sample_id": rows[0]["sample_id"],
                "last_sample_id": rows[-1]["sample_id"],
                "sample_count": count,
            },
            "files": {
                "payload": _PAYLOAD_NAME,
                "manifest": _MANIFEST_NAME,
                "summary": _SUMMARY_NAME,
            },
            "checksums": {
                "payload_sha256": _sha256(payload_bytes),
                "manifest_sha256": _sha256(manifest_bytes),
            },
            "manifest_digest": manifest_digest,
            "semantic_digest": semantic_digest,
            "array_layout": _array_layout(arrays),
            "audit_report": audit,
        }
        (staging / _SUMMARY_NAME).write_bytes(_json_bytes(summary))
        loaded = load_verification_shard(staging, grid=grid, library=library)
        if loaded.semantic_digest != semantic_digest:
            raise ValueError("staging reload semantic digest mismatch")
        os.rename(staging, output_path)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "directory": output_path,
        "payload": output_path / _PAYLOAD_NAME,
        "manifest": output_path / _MANIFEST_NAME,
        "summary": output_path / _SUMMARY_NAME,
    }


def _load_summary(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    try:
        value = _strict_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("summary is not strict finite JSON") from exc
    if not isinstance(value, dict) or set(value) != _SUMMARY_KEYS:
        raise ValueError("summary keys violate the verification shard layout")
    if _json_bytes(value) != raw:
        raise ValueError("summary must be canonical compact JSON")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("summary schema version mismatch")
    if value["layout_version"] != VERIFICATION_SHARD_LAYOUT_VERSION:
        raise ValueError("unsupported verification shard layout")
    return value, raw


def _load_manifest(path: Path) -> tuple[tuple[dict[str, object], ...], bytes]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("manifest must be non-empty newline-terminated JSONL")
    try:
        lines = raw.decode("utf-8").splitlines()
        values = tuple(_strict_loads(line) for line in lines)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("manifest is not strict finite JSONL") from exc
    if any(not isinstance(value, dict) for value in values):
        raise ValueError("manifest rows must be JSON objects")
    rows = tuple(dict(value) for value in values)
    if _jsonl_bytes(rows) != raw:
        raise ValueError("manifest must be canonical compact JSONL")
    return rows, raw


def _expected_shapes(grid: GridSpec, count: int) -> dict[str, tuple[int, ...]]:
    return {
        "bev_history": (
            count,
            grid.history_steps,
            grid.n_history_channels,
            grid.height,
            grid.width,
        ),
        "state_channels": (
            count,
            grid.n_state_channels,
            grid.height,
            grid.width,
        ),
        "trajectory_channels": (
            count,
            grid.n_trajectory_channels,
            grid.height,
            grid.width,
        ),
        "verification_fov_mask": (count, 1, grid.height, grid.width),
        "verification_action_vector": (count, 3),
        "value_target": (count,),
        "useful_target": (count,),
        "br_before": (count,),
        "post_risk": (count,),
    }


def _expected_dtypes() -> dict[str, np.dtype]:
    return {
        "bev_history": np.dtype("<f4"),
        "state_channels": np.dtype("<f4"),
        "trajectory_channels": np.dtype("<f4"),
        "verification_fov_mask": np.dtype("<f4"),
        "verification_action_vector": np.dtype("<f4"),
        "value_target": np.dtype("<f8"),
        "useful_target": np.dtype("uint8"),
        "br_before": np.dtype("<f8"),
        "post_risk": np.dtype("<f8"),
    }


def _load_arrays(path: Path, *, grid: GridSpec, count: int) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(_NUMERIC_NAMES):
                raise ValueError("NPZ keys violate the verification shard layout")
            arrays = {name: archive[name].copy(order="K") for name in _NUMERIC_NAMES}
    except (OSError, ValueError) as exc:
        raise ValueError("failed to load numeric pickle-free payload") from exc
    shapes = _expected_shapes(grid, count)
    dtypes = _expected_dtypes()
    for name, array in arrays.items():
        if array.dtype.kind not in "fiu":
            raise TypeError(f"{name} must be a numeric array")
        if array.dtype != dtypes[name]:
            raise TypeError(f"{name} dtype mismatch")
        if array.shape != shapes[name]:
            raise ValueError(f"{name} shape mismatch")
        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
    if not np.isin(arrays["useful_target"], (0, 1)).all():
        raise ValueError("useful_target must be binary")
    return arrays


def _reconstruct(
    rows: Sequence[Mapping[str, object]],
    arrays: Mapping[str, np.ndarray],
    *,
    grid: GridSpec,
    library: VerificationActionLibrary,
) -> tuple[VerificationSample, ...]:
    samples: list[VerificationSample] = []
    for index, row in enumerate(rows):
        metadata = _canonical_copy(row["metadata"], name="manifest metadata")
        if not isinstance(metadata, dict):
            raise ValueError("manifest metadata must be an object")
        sample = VerificationSample(
            sample_id=str(row["sample_id"]),
            split=str(row["split"]),
            base_state_id=str(row["base_state_id"]),
            nominal_trajectory_id=str(row["nominal_trajectory_id"]),
            verification_action_id=str(row["verification_action_id"]),
            bev_history=arrays["bev_history"][index].copy(order="C"),
            state_channels=arrays["state_channels"][index].copy(order="C"),
            trajectory_channels=arrays["trajectory_channels"][index].copy(order="C"),
            verification_fov_mask=arrays["verification_fov_mask"][index].copy(order="C"),
            verification_action_vector=arrays["verification_action_vector"][index].copy(order="C"),
            value_target=float(arrays["value_target"][index]),
            useful_target=int(arrays["useful_target"][index]),
            br_before=float(arrays["br_before"][index]),
            post_risk=float(arrays["post_risk"][index]),
            metadata=metadata,
        )
        validate_verification_sample_for_publication(
            sample, grid=grid, library=library
        )
        expected_row = _manifest_row(
            sample,
            shard_index=int(row["shard_index"]),
            row_index=index,
        )
        if dict(row) != expected_row:
            raise ValueError("manifest identity fields are inconsistent")
        samples.append(sample)
    return tuple(samples)


def load_verification_shard(
    output_dir: str | Path,
    *,
    grid: GridSpec,
    library: VerificationActionLibrary,
    recompute_value: Callable[[VerificationSample], float] | None = None,
    recompute_atol: float = 1e-9,
) -> LoadedVerificationShard:
    """Load only after file, checksum, schema, semantic, and group audits pass."""

    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    if not isinstance(library, VerificationActionLibrary):
        raise TypeError("library must be a VerificationActionLibrary")
    root = Path(output_dir)
    if not root.is_dir():
        raise ValueError(f"incomplete verification shard: {root}")
    actual = {path.name for path in root.iterdir()}
    if actual != _REQUIRED_FILES:
        raise ValueError("verification shard root file set is incomplete or unexpected")
    summary, _ = _load_summary(root / _SUMMARY_NAME)
    if summary["files"] != {
        "payload": _PAYLOAD_NAME,
        "manifest": _MANIFEST_NAME,
        "summary": _SUMMARY_NAME,
    }:
        raise ValueError("summary file layout mismatch")
    count = _validate_integer(
        summary["expected_sample_count"],
        name="expected_sample_count",
        minimum=1,
    )
    shard_index = _validate_integer(summary["shard_index"], name="shard_index")
    split = summary["split"]
    if not isinstance(split, str) or not split:
        raise ValueError("summary split must be a non-empty string")
    checksums = summary["checksums"]
    if not isinstance(checksums, dict) or set(checksums) != {
        "payload_sha256",
        "manifest_sha256",
    }:
        raise ValueError("summary checksum keys are invalid")
    payload_bytes = (root / _PAYLOAD_NAME).read_bytes()
    manifest_rows, manifest_bytes = _load_manifest(root / _MANIFEST_NAME)
    if _sha256(payload_bytes) != checksums["payload_sha256"]:
        raise ValueError("payload checksum mismatch")
    if _sha256(manifest_bytes) != checksums["manifest_sha256"]:
        raise ValueError("manifest checksum mismatch")
    if len(manifest_rows) != count:
        raise ValueError("manifest count differs from expected_sample_count")
    for index, row in enumerate(manifest_rows):
        if set(row) != _MANIFEST_KEYS:
            raise ValueError("manifest row keys violate the layout")
        if row["schema_version"] != SCHEMA_VERSION:
            raise ValueError("manifest schema version mismatch")
        if row["layout_version"] != VERIFICATION_SHARD_LAYOUT_VERSION:
            raise ValueError("manifest layout version mismatch")
        if row["shard_index"] != shard_index or row["row_index"] != index:
            raise ValueError("manifest fixed row boundary mismatch")
        if row["split"] != split:
            raise ValueError("manifest contains mixed split rows")
    sample_ids = [str(row["sample_id"]) for row in manifest_rows]
    if sample_ids != sorted(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("manifest sample IDs must be sorted and unique")
    expected_boundary = {
        "first_sample_id": sample_ids[0],
        "last_sample_id": sample_ids[-1],
        "sample_count": count,
    }
    if summary["boundary"] != expected_boundary:
        raise ValueError("summary boundary mismatch")
    manifest_digest = _domain_digest(
        b"verification-manifest-v1\0", manifest_bytes
    )
    if summary["manifest_digest"] != manifest_digest:
        raise ValueError("manifest digest mismatch")
    arrays = _load_arrays(root / _PAYLOAD_NAME, grid=grid, count=count)
    if summary["array_layout"] != _array_layout(arrays):
        raise ValueError("array layout summary mismatch")
    semantic_digest = _semantic_digest(
        arrays, shard_index=shard_index, split=split, rows=manifest_rows
    )
    if summary["semantic_digest"] != semantic_digest:
        raise ValueError("payload semantic digest mismatch")
    samples = _reconstruct(
        manifest_rows, arrays, grid=grid, library=library
    )
    audit = audit_verification_groups(list(samples), require_complete=True)
    if summary["audit_report"] != audit:
        raise ValueError("verification group audit report mismatch")
    if recompute_value is not None:
        if not callable(recompute_value):
            raise TypeError("recompute_value must be callable")
        if (
            isinstance(recompute_atol, bool)
            or not isinstance(recompute_atol, (int, float))
            or not np.isfinite(recompute_atol)
            or recompute_atol < 0.0
        ):
            raise ValueError("recompute_atol must be finite and non-negative")
        for sample in samples:
            recomputed = recompute_value(sample)
            if (
                isinstance(recomputed, bool)
                or not isinstance(recomputed, (int, float, np.integer, np.floating))
                or not np.isfinite(recomputed)
            ):
                raise ValueError("recomputed G* must be finite")
            if not np.isclose(
                float(recomputed), sample.value_target, rtol=0.0, atol=recompute_atol
            ):
                raise ValueError("recomputed G* differs from stored value_target")
    return LoadedVerificationShard(
        samples=samples,
        manifest=tuple(dict(row) for row in manifest_rows),
        manifest_digest=manifest_digest,
        semantic_digest=semantic_digest,
        action_counts=dict(audit["action_counts"]),
        audit_report=dict(audit),
        summary=dict(summary),
    )


def load_verification_collection(
    shard_dirs: Sequence[str | Path],
    *,
    grid: GridSpec,
    library: VerificationActionLibrary,
) -> VerificationCollection:
    """Load multiple shards and audit global IDs and split-isolated groups."""

    if isinstance(shard_dirs, (str, bytes)) or not isinstance(shard_dirs, Sequence):
        raise TypeError("shard_dirs must be a sequence")
    roots = tuple(sorted((Path(value) for value in shard_dirs), key=lambda path: str(path)))
    if not roots:
        raise ValueError("shard_dirs must be non-empty")
    loaded = tuple(
        load_verification_shard(root, grid=grid, library=library) for root in roots
    )
    samples = tuple(
        sorted(
            (sample for shard in loaded for sample in shard.samples),
            key=lambda sample: (sample.split, sample.sample_id),
        )
    )
    audit = audit_verification_groups(list(samples), require_complete=True)
    return VerificationCollection(
        samples=samples,
        shard_semantic_digests=tuple(shard.semantic_digest for shard in loaded),
        audit_report=audit,
    )


__all__ = (
    "LoadedVerificationShard",
    "VERIFICATION_SHARD_LAYOUT_VERSION",
    "VerificationCollection",
    "load_verification_collection",
    "load_verification_shard",
    "write_verification_shard",
)
