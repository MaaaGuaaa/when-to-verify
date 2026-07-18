"""Deterministic, fail-closed NPZ + JSONL storage for schema-v3 RiskSample shards."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np

from src.contracts import GridSpec, RiskSample, SCHEMA_VERSION
from src.datasets.risk_dataset import validate_risk_sample_for_publication
from src.datasets.split_manager import (
    SplitAuditPolicy,
    SplitLeakageError,
    assert_no_split_leakage,
)


RISK_SHARD_LAYOUT_VERSION = "risk_shard_npz_jsonl_v2"

_PAYLOAD_NAME = "samples.npz"
_MANIFEST_NAME = "metadata.jsonl"
_SUMMARY_NAME = "summary.json"
_REQUIRED_FILES = frozenset({_PAYLOAD_NAME, _MANIFEST_NAME, _SUMMARY_NAME})

_NUMERIC_ARRAY_NAMES = (
    "bev_history",
    "state_channels",
    "trajectory_channels",
    "robot_state",
    "collision_label",
    "risk_severity",
    "min_clearance",
    "near_miss",
    "first_collision_time_value",
    "first_collision_time_valid",
)
_NPZ_KEYS = frozenset((*_NUMERIC_ARRAY_NAMES, "meta_json"))
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "layout_version",
        "shard_index",
        "row_index",
        "sample_id",
        "split",
        "base_state_id",
        "pair_group_id",
        "event_type",
        "trajectory_id",
        "base_recording_id",
        "base_session_id",
        "source_recording_id",
        "source_session_id",
        "source_snippet_id",
        "seed_namespace",
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
        "manifest_digest",
        "semantic_digest",
        "array_layout",
        "audit_context_digest",
        "leakage_report",
    }
)
_META_KEYS = frozenset(
    {
        "schema_version",
        "layout_version",
        "shard_index",
        "split",
        "sample_ids",
        "manifest_digest",
        "semantic_digest",
        "array_layout",
        "audit_context_digest",
    }
)

# THÖR evaluation keeps sessions known while holding out recordings.  Session
# overlap is therefore visible in the report but is not leakage.  Recording,
# snippet, pair-group, and seed identity remain strict cross-split boundaries.
_THOR_BASE_IDENTITY_POLICY = SplitAuditPolicy(
    evaluation_scope="base_unseen_recording_within_known_sessions",
    required_fields=("recording", "session"),
    allowed_overlap_fields=("session",),
    unavailable_fields=("participant",),
)
_THOR_COMBINED_IDENTITY_POLICY = SplitAuditPolicy(
    evaluation_scope="global_recording_identity_across_base_and_source_roles",
    required_fields=("recording", "session"),
    allowed_overlap_fields=("session",),
    unavailable_fields=("participant",),
)
_THOR_SOURCE_IDENTITY_POLICY = SplitAuditPolicy(
    evaluation_scope="unseen_recording_within_known_sessions",
    required_fields=(
        "recording",
        "session",
        "snippet",
        "pair_group",
        "seed_namespace",
    ),
    allowed_overlap_fields=("session",),
    unavailable_fields=("participant",),
)


@dataclass(frozen=True)
class LoadedRiskShard:
    """A fully verified shard reconstructed into publication-safe samples."""

    samples: tuple[RiskSample, ...]
    manifest: tuple[dict[str, object], ...]
    manifest_digest: str
    semantic_digest: str
    leakage_report: dict[str, object]
    summary: dict[str, object]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_loads(payload: str) -> object:
    return json.loads(payload, parse_constant=_reject_json_constant)


def _canonical_copy(value: object, *, name: str) -> object:
    try:
        return _strict_json_loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite canonical JSON") from exc


def _serialize_jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        return b""
    return ("\n".join(_canonical_json(row) for row in rows) + "\n").encode(
        "utf-8"
    )


def _serialize_json(value: Mapping[str, object]) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _digest_bytes(domain: bytes, payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def _manifest_digest(rows: Sequence[Mapping[str, object]]) -> str:
    return _digest_bytes(b"risk-shard-manifest-v2\0", _serialize_jsonl(rows))


def _canonical_audit_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("split_audit_records must be a sequence of mappings")
    copied: list[dict[str, object]] = []
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise TypeError(f"split_audit_records[{index}] must be a mapping")
        normalized = _canonical_copy(dict(row), name=f"split_audit_records[{index}]")
        if not isinstance(normalized, dict):  # pragma: no cover - dict input above
            raise TypeError(f"split_audit_records[{index}] must be a mapping")
        copied.append(normalized)
    copied.sort(key=_canonical_json)
    return tuple(copied)


def _audit_context_digest(records: Sequence[Mapping[str, object]]) -> str:
    return _digest_bytes(
        b"risk-shard-audit-context-v2\0",
        _serialize_jsonl(records),
    )


def _require_nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing required provenance: {name}")
    return value


def _provenance_value(
    provenance: Mapping[str, object],
    aliases: tuple[str, ...],
    *,
    name: str,
) -> str:
    values = {
        value
        for alias in aliases
        if alias in provenance
        for value in (
            _require_nonempty_string(provenance[alias], name=name),
        )
    }
    if not values:
        raise ValueError(f"missing required provenance: {name}")
    if len(values) != 1:
        raise ValueError(f"conflicting required provenance aliases: {name}")
    return next(iter(values))


def _identity_audit_rows(
    records: Sequence[Mapping[str, object]],
    *,
    identity: str,
) -> tuple[dict[str, object], ...]:
    if identity not in {"base", "source"}:  # pragma: no cover - private contract
        raise ValueError("identity must be base or source")
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        split = _require_nonempty_string(
            record.get("split"), name=f"split_audit_records[{index}].split"
        )
        normalized: dict[str, object] = {
            "split": split,
            "recording_id": _require_nonempty_string(
                record.get(f"{identity}_recording_id"),
                name=f"{identity}_recording_id provenance",
            ),
            "session_id": _require_nonempty_string(
                record.get(f"{identity}_session_id"),
                name=f"{identity}_session_id provenance",
            ),
        }
        if identity == "source":
            normalized.update(
                {
                    "source_snippet_id": _require_nonempty_string(
                        record.get("source_snippet_id"),
                        name="source_snippet_id provenance",
                    ),
                    "pair_group_id": _require_nonempty_string(
                        record.get("pair_group_id"),
                        name="pair_group_id provenance",
                    ),
                    "seed_namespace": _require_nonempty_string(
                        record.get("seed_namespace"),
                        name="seed_namespace provenance",
                    ),
                }
            )
        rows.append(normalized)
    return tuple(rows)


def _identity_leakage_report(
    records: Sequence[Mapping[str, object]],
    *,
    identity: str,
    policy: SplitAuditPolicy,
) -> dict[str, object]:
    normalized = _identity_audit_rows(records, identity=identity)
    try:
        return assert_no_split_leakage(normalized, policy=policy)
    except SplitLeakageError as exc:
        message = str(exc).replace("recording", f"{identity} recording")
        message = message.replace("session", f"{identity} session")
        raise SplitLeakageError(message) from exc


def _combined_identity_leakage_report(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    expanded = tuple(
        identity_row
        for identity in ("base", "source")
        for identity_row in _identity_audit_rows(records, identity=identity)
    )
    return assert_no_split_leakage(
        expanded, policy=_THOR_COMBINED_IDENTITY_POLICY
    )


def _dual_identity_leakage_report(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    base = _identity_leakage_report(
        records, identity="base", policy=_THOR_BASE_IDENTITY_POLICY
    )
    source = _identity_leakage_report(
        records, identity="source", policy=_THOR_SOURCE_IDENTITY_POLICY
    )
    combined = _combined_identity_leakage_report(records)
    source_common_overlap_count = sum(
        source["fields"][field]["overlap_count"]
        for field in ("snippet", "pair_group", "seed_namespace")
    )
    return {
        "status": "ok",
        "evaluation_scope": "dual_base_source_unseen_recording_v1",
        "detected_overlap_count": (
            combined["detected_overlap_count"] + source_common_overlap_count
        ),
        "allowed_overlap_count": combined["allowed_overlap_count"],
        "disallowed_overlap_count": 0,
        "missing_required_row_count": 0,
        "base_identity": base,
        "source_identity": source,
        "combined_identity": combined,
    }


def _manifest_row(
    sample: RiskSample,
    *,
    shard_index: int,
    row_index: int,
) -> dict[str, object]:
    metadata = _canonical_copy(sample.metadata, name=f"{sample.sample_id}.metadata")
    if not isinstance(metadata, dict):  # pragma: no cover - validator guards this
        raise TypeError("sample metadata must be a mapping")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("missing required provenance: provenance")
    base_recording_id = _provenance_value(
        provenance,
        ("base_recording_id",),
        name="base_recording_id provenance",
    )
    base_session_id = _provenance_value(
        provenance,
        ("base_session_id",),
        name="base_session_id provenance",
    )
    source_recording_id = _provenance_value(
        provenance,
        ("source_recording_id",),
        name="source_recording_id provenance",
    )
    source_session_id = _provenance_value(
        provenance,
        ("source_session_id",),
        name="source_session_id provenance",
    )
    snippet_id = _provenance_value(
        provenance,
        ("source_snippet_id", "dynamic_object_snippet_id"),
        name="snippet",
    )
    seed_namespace = _provenance_value(
        provenance,
        ("seed_namespace", "generator_seed_namespace"),
        name="seed_namespace",
    )
    trajectory_id = _require_nonempty_string(
        metadata.get("trajectory_id"), name="trajectory_id"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "layout_version": RISK_SHARD_LAYOUT_VERSION,
        "shard_index": shard_index,
        "row_index": row_index,
        "sample_id": sample.sample_id,
        "split": sample.split,
        "base_state_id": sample.base_state_id,
        "pair_group_id": sample.pair_group_id,
        "event_type": sample.event_type,
        "trajectory_id": trajectory_id,
        "base_recording_id": base_recording_id,
        "base_session_id": base_session_id,
        "source_recording_id": source_recording_id,
        "source_session_id": source_session_id,
        "source_snippet_id": snippet_id,
        "seed_namespace": seed_namespace,
        "metadata": metadata,
    }


def _little_endian_float32(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.dtype("<f4"))


def _build_numeric_arrays(samples: Sequence[RiskSample]) -> dict[str, np.ndarray]:
    first_time_value = np.asarray(
        [
            0.0 if sample.first_collision_time is None else sample.first_collision_time
            for sample in samples
        ],
        dtype=np.dtype("<f8"),
    )
    first_time_valid = np.asarray(
        [sample.first_collision_time is not None for sample in samples],
        dtype=np.uint8,
    )
    arrays = {
        "bev_history": _little_endian_float32(
            np.stack([sample.bev_history for sample in samples], axis=0)
        ),
        "state_channels": _little_endian_float32(
            np.stack([sample.state_channels for sample in samples], axis=0)
        ),
        "trajectory_channels": _little_endian_float32(
            np.stack([sample.trajectory_channels for sample in samples], axis=0)
        ),
        "robot_state": _little_endian_float32(
            np.stack([sample.robot_state for sample in samples], axis=0)
        ),
        "collision_label": np.ascontiguousarray(
            [sample.collision_label for sample in samples], dtype=np.uint8
        ),
        "risk_severity": np.ascontiguousarray(
            [sample.risk_severity for sample in samples], dtype=np.dtype("<f8")
        ),
        "min_clearance": np.ascontiguousarray(
            [sample.min_clearance for sample in samples], dtype=np.dtype("<f8")
        ),
        "near_miss": np.ascontiguousarray(
            [sample.near_miss for sample in samples], dtype=np.uint8
        ),
        "first_collision_time_value": np.ascontiguousarray(first_time_value),
        "first_collision_time_valid": np.ascontiguousarray(first_time_valid),
    }
    if not all(array.flags.c_contiguous for array in arrays.values()):
        raise RuntimeError("writer failed to construct C-contiguous arrays")
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("risk shard arrays must contain only finite values")
    return arrays


def _array_layout(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        name: {
            "dtype": arrays[name].dtype.str,
            "shape": list(arrays[name].shape),
            "order": "C",
        }
        for name in _NUMERIC_ARRAY_NAMES
    }


def _semantic_digest(
    arrays: Mapping[str, np.ndarray],
    *,
    shard_index: int,
    split: str,
    sample_ids: Sequence[str],
    manifest_digest: str,
    audit_context_digest: str,
) -> str:
    header = {
        "schema_version": SCHEMA_VERSION,
        "layout_version": RISK_SHARD_LAYOUT_VERSION,
        "shard_index": shard_index,
        "split": split,
        "sample_ids": list(sample_ids),
        "manifest_digest": manifest_digest,
        "audit_context_digest": audit_context_digest,
        "array_layout": _array_layout(arrays),
    }
    digest = hashlib.sha256()
    digest.update(b"risk-shard-semantic-v2\0")
    header_bytes = _canonical_json(header).encode("utf-8")
    digest.update(len(header_bytes).to_bytes(8, "big"))
    digest.update(header_bytes)
    for name in _NUMERIC_ARRAY_NAMES:
        name_bytes = name.encode("utf-8")
        raw = arrays[name].tobytes(order="C")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


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
        "robot_state": (count, 2),
        "collision_label": (count,),
        "risk_severity": (count,),
        "min_clearance": (count,),
        "near_miss": (count,),
        "first_collision_time_value": (count,),
        "first_collision_time_valid": (count,),
    }


def _expected_dtypes() -> dict[str, np.dtype]:
    return {
        "bev_history": np.dtype("<f4"),
        "state_channels": np.dtype("<f4"),
        "trajectory_channels": np.dtype("<f4"),
        "robot_state": np.dtype("<f4"),
        "collision_label": np.dtype("uint8"),
        "risk_severity": np.dtype("<f8"),
        "min_clearance": np.dtype("<f8"),
        "near_miss": np.dtype("uint8"),
        "first_collision_time_value": np.dtype("<f8"),
        "first_collision_time_valid": np.dtype("uint8"),
    }


def _validate_integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validate_summary(summary: object) -> dict[str, object]:
    if not isinstance(summary, dict):
        raise ValueError("summary must be a JSON object")
    if set(summary) != _SUMMARY_KEYS:
        raise ValueError("summary keys violate the shard layout")
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"summary schema_version must be {SCHEMA_VERSION}, "
            f"got {summary.get('schema_version')!r}"
        )
    if summary.get("layout_version") != RISK_SHARD_LAYOUT_VERSION:
        raise ValueError(
            "unsupported risk shard layout: "
            f"{summary.get('layout_version')!r}"
        )
    return summary


def _load_manifest(path: Path) -> tuple[dict[str, object], ...]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("manifest must be non-empty newline-terminated JSONL")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("manifest must be UTF-8") from exc
    rows: list[dict[str, object]] = []
    for index, line in enumerate(text.splitlines()):
        try:
            row = _strict_json_loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"manifest row {index} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"manifest row {index} must be an object")
        rows.append(row)
    if _serialize_jsonl(rows) != raw:
        raise ValueError("manifest is not canonical compact JSONL")
    return tuple(rows)


def _validate_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    summary: Mapping[str, object],
) -> tuple[str, ...]:
    expected_count = _validate_integer(
        summary["expected_sample_count"],
        name="expected_sample_count",
        minimum=1,
    )
    if len(rows) != expected_count:
        raise ValueError("manifest sample count violates fixed shard boundary")
    shard_index = _validate_integer(
        summary["shard_index"], name="shard_index", minimum=0
    )
    split = summary["split"]
    if not isinstance(split, str) or not split:
        raise ValueError("summary split must be a non-empty string")
    sample_ids: list[str] = []
    for index, row in enumerate(rows):
        if set(row) != _MANIFEST_KEYS:
            raise ValueError(f"manifest row {index} keys violate the shard layout")
        if row["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"manifest row {index} schema_version mismatch")
        if row["layout_version"] != RISK_SHARD_LAYOUT_VERSION:
            raise ValueError(f"manifest row {index} layout_version mismatch")
        if row["shard_index"] != shard_index or row["row_index"] != index:
            raise ValueError(f"manifest row {index} violates fixed shard boundary")
        if row["split"] != split:
            raise ValueError("manifest contains mixed split rows")
        sample_ids.append(
            _require_nonempty_string(row["sample_id"], name="sample_id")
        )
    if sample_ids != sorted(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("manifest sample_id order must be stable and unique")
    boundary = {
        "first_sample_id": sample_ids[0],
        "last_sample_id": sample_ids[-1],
        "sample_count": len(sample_ids),
    }
    if summary["boundary"] != boundary:
        raise ValueError("summary fixed shard boundary mismatch")
    return tuple(sample_ids)


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            archive_files = set(archive.files)
            arrays = {
                name: archive[name].copy(order="K")
                for name in _NUMERIC_ARRAY_NAMES
                if name in archive_files
            }
            meta_array = (
                archive["meta_json"].copy()
                if "meta_json" in archive_files
                else None
            )
    except (OSError, ValueError) as exc:
        raise ValueError("failed to load pickle-free NPZ payload") from exc

    if archive_files != _NPZ_KEYS:
        raise ValueError("NPZ keys violate the shard layout")
    for name, array in arrays.items():
        if array.dtype.kind == "O":
            raise TypeError(f"{name} must not be an object array")
        if not array.flags.c_contiguous:
            raise ValueError(f"{name} array order must be C")
    if meta_array is None:  # pragma: no cover - exact-key check above
        raise ValueError("NPZ keys violate the shard layout")
    if meta_array.dtype.kind != "U" or meta_array.shape != ():
        raise TypeError("meta_json must be a scalar Unicode array")
    meta_text = str(meta_array)
    meta = _strict_json_loads(meta_text)
    if not isinstance(meta, dict):
        raise ValueError("meta_json must contain an object")
    return arrays, meta


def _validate_payload_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    grid: GridSpec,
    count: int,
) -> None:
    shapes = _expected_shapes(grid, count)
    dtypes = _expected_dtypes()
    for name in _NUMERIC_ARRAY_NAMES:
        array = arrays[name]
        if array.dtype != dtypes[name]:
            raise TypeError(
                f"{name} dtype mismatch: expected {dtypes[name].str}, "
                f"got {array.dtype.str}"
            )
        if array.shape != shapes[name]:
            raise ValueError(
                f"{name} shape mismatch: expected {shapes[name]}, got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
    for name in ("collision_label", "near_miss", "first_collision_time_valid"):
        if not np.isin(arrays[name], (0, 1)).all():
            raise ValueError(f"{name} must contain only 0/1")
    invalid = arrays["first_collision_time_valid"] == 0
    if not np.equal(arrays["first_collision_time_value"][invalid], 0.0).all():
        raise ValueError("invalid optional times must use finite zero values")


def _validate_payload_meta(
    meta: Mapping[str, object],
    *,
    summary: Mapping[str, object],
    sample_ids: Sequence[str],
    arrays: Mapping[str, np.ndarray],
) -> None:
    if set(meta) != _META_KEYS:
        raise ValueError("meta_json keys violate the shard layout")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "layout_version": RISK_SHARD_LAYOUT_VERSION,
        "shard_index": summary["shard_index"],
        "split": summary["split"],
        "sample_ids": list(sample_ids),
        "manifest_digest": summary["manifest_digest"],
        "semantic_digest": summary["semantic_digest"],
        "array_layout": _array_layout(arrays),
        "audit_context_digest": summary["audit_context_digest"],
    }
    if meta != expected:
        raise ValueError("meta_json does not match the shard summary/layout")


def _reconstruct_samples(
    rows: Sequence[Mapping[str, object]],
    arrays: Mapping[str, np.ndarray],
    *,
    grid: GridSpec,
) -> tuple[RiskSample, ...]:
    samples: list[RiskSample] = []
    for index, row in enumerate(rows):
        valid_time = bool(arrays["first_collision_time_valid"][index])
        first_time = (
            float(arrays["first_collision_time_value"][index])
            if valid_time
            else None
        )
        metadata = _canonical_copy(row["metadata"], name="manifest metadata")
        if not isinstance(metadata, dict):
            raise ValueError("manifest metadata must be an object")
        if metadata.get("trajectory_id") != row["trajectory_id"]:
            raise ValueError("manifest trajectory_id does not match metadata")
        sample = RiskSample(
            sample_id=str(row["sample_id"]),
            split=str(row["split"]),
            base_state_id=str(row["base_state_id"]),
            pair_group_id=str(row["pair_group_id"]),
            event_type=str(row["event_type"]),
            bev_history=arrays["bev_history"][index].copy(order="C"),
            state_channels=arrays["state_channels"][index].copy(order="C"),
            trajectory_channels=arrays["trajectory_channels"][index].copy(order="C"),
            robot_state=arrays["robot_state"][index].copy(order="C"),
            collision_label=int(arrays["collision_label"][index]),
            risk_severity=float(arrays["risk_severity"][index]),
            min_clearance=float(arrays["min_clearance"][index]),
            near_miss=int(arrays["near_miss"][index]),
            first_collision_time=first_time,
            metadata=metadata,
        )
        validate_risk_sample_for_publication(sample, grid)
        expected_row = _manifest_row(
            sample,
            shard_index=int(row["shard_index"]),
            row_index=index,
        )
        if dict(row) != expected_row:
            raise ValueError("manifest identity/provenance fields are inconsistent")
        samples.append(sample)
    return tuple(samples)


def write_risk_shard(
    samples: Sequence[RiskSample],
    output_dir: str | Path,
    *,
    grid: GridSpec,
    shard_index: int = 0,
    expected_sample_count: int,
    split_audit_records: Sequence[Mapping[str, object]] = (),
) -> dict[str, Path]:
    """Write one immutable shard after a full staging-directory reload."""

    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    shard_index = _validate_integer(shard_index, name="shard_index", minimum=0)
    expected_sample_count = _validate_integer(
        expected_sample_count, name="expected_sample_count", minimum=1
    )
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence")
    if any(not isinstance(sample, RiskSample) for sample in samples):
        raise TypeError("samples must contain only RiskSample instances")
    ordered = tuple(sorted(samples, key=lambda sample: sample.sample_id))
    if len(ordered) != expected_sample_count:
        raise ValueError(
            "expected_sample_count does not match the fixed shard boundary"
        )
    if not ordered:
        raise ValueError("a risk shard must contain at least one sample")
    for sample in ordered:
        validate_risk_sample_for_publication(sample, grid)
    sample_ids = tuple(sample.sample_id for sample in ordered)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate sample_id in risk shard")
    splits = {sample.split for sample in ordered}
    if len(splits) != 1:
        raise ValueError("mixed split samples are forbidden in one risk shard")
    split = next(iter(splits))

    rows = tuple(
        _manifest_row(sample, shard_index=shard_index, row_index=index)
        for index, sample in enumerate(ordered)
    )
    external_audit = _canonical_audit_records(split_audit_records)
    leakage_report = _dual_identity_leakage_report((*rows, *external_audit))
    audit_digest = _audit_context_digest(external_audit)
    manifest_digest = _manifest_digest(rows)
    arrays = _build_numeric_arrays(ordered)
    array_layout = _array_layout(arrays)
    semantic_digest = _semantic_digest(
        arrays,
        shard_index=shard_index,
        split=split,
        sample_ids=sample_ids,
        manifest_digest=manifest_digest,
        audit_context_digest=audit_digest,
    )
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "layout_version": RISK_SHARD_LAYOUT_VERSION,
        "shard_index": shard_index,
        "split": split,
        "expected_sample_count": expected_sample_count,
        "boundary": {
            "first_sample_id": sample_ids[0],
            "last_sample_id": sample_ids[-1],
            "sample_count": len(sample_ids),
        },
        "files": {
            "payload": _PAYLOAD_NAME,
            "manifest": _MANIFEST_NAME,
            "summary": _SUMMARY_NAME,
        },
        "manifest_digest": manifest_digest,
        "semantic_digest": semantic_digest,
        "array_layout": array_layout,
        "audit_context_digest": audit_digest,
        "leakage_report": leakage_report,
    }
    meta = {
        "schema_version": SCHEMA_VERSION,
        "layout_version": RISK_SHARD_LAYOUT_VERSION,
        "shard_index": shard_index,
        "split": split,
        "sample_ids": list(sample_ids),
        "manifest_digest": manifest_digest,
        "semantic_digest": semantic_digest,
        "array_layout": array_layout,
        "audit_context_digest": audit_digest,
    }

    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable shard: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.staging-",
            dir=output_path.parent,
        )
    )
    try:
        payload = dict(arrays)
        payload["meta_json"] = np.asarray(_canonical_json(meta))
        payload_path = staging / _PAYLOAD_NAME
        with payload_path.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        (staging / _MANIFEST_NAME).write_bytes(_serialize_jsonl(rows))
        (staging / _SUMMARY_NAME).write_bytes(_serialize_json(summary))

        loaded = load_risk_shard(
            staging,
            grid=grid,
            split_audit_records=external_audit,
        )
        if (
            loaded.manifest_digest != manifest_digest
            or loaded.semantic_digest != semantic_digest
        ):
            raise ValueError("formal reload digest mismatch")
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


def load_risk_shard(
    output_dir: str | Path,
    *,
    grid: GridSpec,
    split_audit_records: Sequence[Mapping[str, object]] = (),
) -> LoadedRiskShard:
    """Load a shard only after schema, layout, digest, and leakage checks pass."""

    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    output_path = Path(output_dir)
    if not output_path.is_dir():
        raise ValueError(f"incomplete shard: directory not found: {output_path}")
    actual_files = {path.name for path in output_path.iterdir()}
    missing = _REQUIRED_FILES - actual_files
    if missing:
        raise ValueError("incomplete shard: missing " + ", ".join(sorted(missing)))
    unexpected = actual_files - _REQUIRED_FILES
    if unexpected:
        raise ValueError(
            "unexpected shard files: " + ", ".join(sorted(unexpected))
        )

    summary_raw = (output_path / _SUMMARY_NAME).read_bytes()
    try:
        summary_text = summary_raw.decode("utf-8")
        summary_object = _strict_json_loads(summary_text)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("summary is not strict finite JSON") from exc
    summary = _validate_summary(summary_object)
    if _serialize_json(summary) != summary_raw:
        raise ValueError("summary is not canonical compact JSON")
    if summary["files"] != {
        "payload": _PAYLOAD_NAME,
        "manifest": _MANIFEST_NAME,
        "summary": _SUMMARY_NAME,
    }:
        raise ValueError("summary file layout mismatch")

    rows = _load_manifest(output_path / _MANIFEST_NAME)
    sample_ids = _validate_manifest(rows, summary=summary)
    manifest_digest = _manifest_digest(rows)
    if summary["manifest_digest"] != manifest_digest:
        raise ValueError("manifest digest mismatch")

    external_audit = _canonical_audit_records(split_audit_records)
    audit_digest = _audit_context_digest(external_audit)
    if summary["audit_context_digest"] != audit_digest:
        raise ValueError("split audit context digest mismatch")
    leakage_report = _dual_identity_leakage_report((*rows, *external_audit))
    if summary["leakage_report"] != leakage_report:
        raise ValueError("split leakage report mismatch")

    arrays, meta = _load_npz(output_path / _PAYLOAD_NAME)
    _validate_payload_arrays(arrays, grid=grid, count=len(rows))
    array_layout = _array_layout(arrays)
    if summary["array_layout"] != array_layout:
        raise ValueError("array dtype/shape/order layout mismatch")
    _validate_payload_meta(
        meta,
        summary=summary,
        sample_ids=sample_ids,
        arrays=arrays,
    )
    semantic_digest = _semantic_digest(
        arrays,
        shard_index=int(summary["shard_index"]),
        split=str(summary["split"]),
        sample_ids=sample_ids,
        manifest_digest=manifest_digest,
        audit_context_digest=audit_digest,
    )
    if summary["semantic_digest"] != semantic_digest:
        raise ValueError("payload semantic digest mismatch")

    samples = _reconstruct_samples(rows, arrays, grid=grid)
    return LoadedRiskShard(
        samples=samples,
        manifest=tuple(dict(row) for row in rows),
        manifest_digest=manifest_digest,
        semantic_digest=semantic_digest,
        leakage_report=leakage_report,
        summary=dict(summary),
    )
