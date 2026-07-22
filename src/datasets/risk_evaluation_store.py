"""Authenticated sibling storage for production evaluation records.

Evaluation records are derived at the oracle/renderer boundary, but they are
published separately from ``RiskSample`` and occupancy sidecars.  This module
binds the records to an already authenticated risk dataset without exposing
oracle fields to model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence

from src.contracts import SCHEMA_VERSION
from src.datasets.risk_dataset_seal import LoadedRiskDataset
from src.datasets.risk_evaluation_metadata import (
    validate_production_evaluation_record,
)
from src.datasets.shard_writer import load_risk_shard
from src.utils.atomic_publish import atomic_rename_noreplace


RISK_EVALUATION_COLLECTION_LAYOUT_VERSION = (
    "risk_evaluation_record_collection_v1"
)
RISK_EVALUATION_SHARD_LAYOUT_VERSION = "risk_evaluation_record_shard_v1"
RISK_EVALUATION_REPLAY_SHARD_LAYOUT_VERSION = (
    "risk_evaluation_record_replay_shard_v1"
)

_MANIFEST_NAME = "evaluation_manifest.json"
_CHECKSUMS_NAME = "checksums.sha256"
_COMPLETE_NAME = ".producer-complete"
_RECORDS_NAME = "records.jsonl"
_SUMMARY_NAME = "summary.json"
_RECORD_LABEL_FIELDS = (
    "collision_label",
    "risk_severity",
    "min_clearance",
    "near_miss",
    "first_collision_time",
)
_RECORD_IDENTITY_FIELDS = (
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
)
_MANIFEST_KEYS = frozenset(
    {
        "collection_layout_version",
        "record_layout_version",
        "schema_version",
        "split",
        "risk_dataset_manifest_digest",
        "base_config_digest",
        "grid",
        "shard_count",
        "sample_count",
        "shards",
        "ordered_sample_ids_digest_sha256",
        "collection_semantic_digest_sha256",
    }
)
_SHARD_SUMMARY_KEYS = frozenset(
    {
        "shard_layout_version",
        "record_layout_version",
        "schema_version",
        "split",
        "shard_index",
        "sample_count",
        "source_risk_shard_semantic_digest",
        "ordered_sample_ids_digest_sha256",
        "records_sha256",
        "semantic_digest_sha256",
        "files",
    }
)
_MARKER_KEYS = frozenset(
    {
        "collection_layout_version",
        "collection_semantic_digest_sha256",
        "evaluation_manifest_sha256",
        "checksums_sha256",
    }
)


class EvaluationCollectionError(ValueError):
    """Raised when a sibling evaluation-record publication is unsafe."""


@dataclass(frozen=True)
class RiskEvaluationShardDescriptor:
    """Authenticated descriptor for one evaluation-record shard."""

    shard_index: int
    relative_root: str
    sample_count: int
    source_risk_shard_semantic_digest: str
    ordered_sample_ids_digest_sha256: str
    records_sha256: str
    semantic_digest_sha256: str


@dataclass(frozen=True)
class LoadedRiskEvaluationCollection:
    """A fully reloaded evaluation-record collection."""

    root: Path
    manifest: dict[str, object]
    shards: tuple[RiskEvaluationShardDescriptor, ...]
    records_by_shard: Mapping[int, tuple[dict[str, object], ...]]
    sample_ids: tuple[str, ...]
    sample_ids_digest_sha256: str
    collection_semantic_digest_sha256: str
    risk_dataset_manifest_digest: str

    @property
    def sample_count(self) -> int:
        return len(self.sample_ids)

    @property
    def records(self) -> tuple[dict[str, object], ...]:
        return tuple(
            record
            for shard in self.shards
            for record in self.records_by_shard[shard.shard_index]
        )

    @property
    def collection_digest_sha256(self) -> str:
        """Compatibility alias used by downstream provenance builders."""

        return self.collection_semantic_digest_sha256

    @property
    def evaluation_record_collection_digest_sha256(self) -> str:
        return self.collection_semantic_digest_sha256


@dataclass(frozen=True)
class LoadedRiskEvaluationReplayShard:
    """One immutable replay shard bound to an exact risk shard."""

    root: Path
    split: str
    shard_index: int
    records: tuple[dict[str, object], ...]
    sample_ids: tuple[str, ...]
    ordered_sample_ids_digest_sha256: str
    source_risk_shard_manifest_digest: str
    source_risk_shard_semantic_digest: str
    semantic_digest_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvaluationCollectionError(
            "evaluation collection metadata must be finite canonical JSON"
        ) from exc


def _strict_json_loads(raw: bytes, *, label: str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvaluationCollectionError(f"{label} is not strict JSON") from exc


def _read_canonical_json(path: Path, *, label: str) -> dict[str, object]:
    _require_regular(path, label=label)
    value = _strict_json_loads(path.read_bytes(), label=label)
    if not isinstance(value, dict):
        raise EvaluationCollectionError(f"{label} must be an object")
    if _canonical_json_bytes(value) != path.read_bytes():
        raise EvaluationCollectionError(f"{label} is not canonical JSON")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise EvaluationCollectionError(f"failed to hash file: {path}") from exc


def _framed_digest(domain: bytes, value: object) -> str:
    payload = _canonical_json_bytes(value)
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def ordered_sample_ids_digest(sample_ids: Sequence[str]) -> str:
    """Digest an ordered, unique sample-ID sequence."""

    if isinstance(sample_ids, (str, bytes)) or not isinstance(sample_ids, Sequence):
        raise EvaluationCollectionError("sample IDs must be an ordered sequence")
    values = list(sample_ids)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise EvaluationCollectionError("sample IDs must be non-empty strings")
    if len(set(values)) != len(values):
        raise EvaluationCollectionError("sample IDs must be unique")
    return _framed_digest(
        b"risk-evaluation-ordered-sample-ids-v1\0",
        values,
    )


def _records_semantic_digest(records: Sequence[Mapping[str, object]]) -> str:
    return _framed_digest(
        b"risk-evaluation-record-shard-semantic-v1\0",
        [dict(record) for record in records],
    )


def _collection_semantic_digest(manifest_scope: Mapping[str, object]) -> str:
    return _framed_digest(
        b"risk-evaluation-record-collection-semantic-v1\0",
        dict(manifest_scope),
    )


def _require_digest(value: object, *, label: str, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluationCollectionError(
            f"{label} must be lowercase hexadecimal with length {length}"
        )
    return value


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationCollectionError(f"{label} must be a non-empty string")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise EvaluationCollectionError(f"{label} must be an integer >= {minimum}")
    return value


def _require_regular(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise EvaluationCollectionError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise EvaluationCollectionError(f"{label} must be a regular file: {path}")


def _require_directory(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise EvaluationCollectionError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise EvaluationCollectionError(f"{label} must be a real directory: {path}")


def _records_jsonl(records: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(record)) for record in records)


def _load_records(path: Path, *, expected_split: str) -> tuple[dict[str, object], ...]:
    _require_regular(path, label="evaluation records")
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise EvaluationCollectionError("evaluation records must be non-empty JSONL")
    rows: list[dict[str, object]] = []
    for index, line in enumerate(raw.splitlines()):
        value = _strict_json_loads(line, label=f"evaluation record {index}")
        if not isinstance(value, dict):
            raise EvaluationCollectionError(f"evaluation record {index} is not an object")
        try:
            normalized = validate_production_evaluation_record(value)
        except (TypeError, ValueError) as exc:
            raise EvaluationCollectionError(
                f"evaluation record {index} failed semantic validation"
            ) from exc
        if normalized["split"] != expected_split:
            raise EvaluationCollectionError(
                f"evaluation record {index} split mismatch"
            )
        rows.append(dict(normalized))
    if _records_jsonl(rows) != raw:
        raise EvaluationCollectionError("evaluation records are not canonical JSONL")
    return tuple(rows)


def _validate_sample_join(record: Mapping[str, object], sample: object) -> None:
    for field in _RECORD_LABEL_FIELDS:
        if record[field] != getattr(sample, field):
            raise EvaluationCollectionError(
                f"evaluation record label mismatch for {record['sample_id']}: {field}"
            )
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise EvaluationCollectionError("risk sample metadata is unavailable")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise EvaluationCollectionError("risk sample provenance is unavailable")
    expected_values = {
        "sample_id": getattr(sample, "sample_id"),
        "split": getattr(sample, "split"),
        "base_state_id": getattr(sample, "base_state_id"),
        "pair_group_id": getattr(sample, "pair_group_id"),
        "event_type": getattr(sample, "event_type"),
        "trajectory_id": metadata.get("trajectory_id"),
        "base_recording_id": provenance.get("base_recording_id"),
        "base_session_id": provenance.get("base_session_id"),
        "source_recording_id": provenance.get("source_recording_id"),
        "source_session_id": provenance.get("source_session_id"),
        "source_snippet_id": provenance.get("source_snippet_id"),
        "seed_namespace": provenance.get("seed_namespace"),
    }
    for field in _RECORD_IDENTITY_FIELDS:
        if record.get(field) != expected_values.get(field):
            raise EvaluationCollectionError(
                f"evaluation record identity mismatch for {record['sample_id']}: {field}"
            )
    audit = metadata.get("label_audit")
    if isinstance(audit, Mapping):
        for record_field, audit_field in (
            ("critical_object_id", "critical_object_id"),
            ("target_object_type", "critical_object_type"),
        ):
            if record.get(record_field) != audit.get(audit_field):
                raise EvaluationCollectionError(
                    f"evaluation record target identity mismatch for {record['sample_id']}"
                )


def _load_risk_shards(dataset: LoadedRiskDataset) -> tuple[object, ...]:
    loaded: list[object] = []
    for descriptor in dataset.shards:
        try:
            shard = load_risk_shard(
                dataset.collection_root / descriptor.relative_root,
                grid=dataset.grid,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise EvaluationCollectionError(
                f"failed to reload authenticated risk shard {descriptor.shard_index}"
            ) from exc
        if shard.semantic_digest != descriptor.semantic_digest:
            raise EvaluationCollectionError(
                f"risk shard semantic digest mismatch for shard {descriptor.shard_index}"
            )
        loaded.append(shard)
    return tuple(loaded)


def _normalise_records_by_shard(
    dataset: LoadedRiskDataset,
    records_by_shard: Mapping[int, Sequence[Mapping[str, object]]],
    risk_shards: Sequence[object],
) -> tuple[tuple[dict[str, object], ...], ...]:
    if not isinstance(records_by_shard, Mapping):
        raise EvaluationCollectionError("records_by_shard must be a mapping")
    expected_indices = tuple(descriptor.shard_index for descriptor in dataset.shards)
    if set(records_by_shard) != set(expected_indices):
        raise EvaluationCollectionError("records_by_shard must exactly cover risk shards")
    normalised: list[tuple[dict[str, object], ...]] = []
    seen: set[str] = set()
    for descriptor, shard in zip(dataset.shards, risk_shards, strict=True):
        raw_records = records_by_shard[descriptor.shard_index]
        if isinstance(raw_records, (str, bytes)) or not isinstance(raw_records, Sequence):
            raise EvaluationCollectionError(
                f"records for shard {descriptor.shard_index} must be a sequence"
            )
        samples = getattr(shard, "samples", ())
        if len(raw_records) != len(samples):
            raise EvaluationCollectionError(
                f"evaluation record count mismatch for shard {descriptor.shard_index}"
            )
        rows: list[dict[str, object]] = []
        for position, (raw, sample) in enumerate(zip(raw_records, samples, strict=True)):
            try:
                record = validate_production_evaluation_record(
                    raw,
                    expected_sample_id=sample.sample_id,
                )
            except (TypeError, ValueError) as exc:
                raise EvaluationCollectionError(
                    "invalid evaluation record at shard "
                    f"{descriptor.shard_index} row {position}: {exc}"
                ) from exc
            _validate_sample_join(record, sample)
            sample_id = str(record["sample_id"])
            if sample_id in seen:
                raise EvaluationCollectionError(f"duplicate evaluation sample_id: {sample_id}")
            seen.add(sample_id)
            rows.append(dict(record))
        normalised.append(tuple(rows))
    return tuple(normalised)


def _build_manifest(
    dataset: LoadedRiskDataset,
    rows_by_shard: Sequence[Sequence[Mapping[str, object]]],
    risk_shards: Sequence[object],
) -> tuple[dict[str, object], tuple[RiskEvaluationShardDescriptor, ...]]:
    base_config_digest: str | None = None
    descriptors: list[RiskEvaluationShardDescriptor] = []
    all_ids: list[str] = []
    for descriptor, rows, risk_shard in zip(
        dataset.shards, rows_by_shard, risk_shards, strict=True
    ):
        for row in rows:
            candidate = row["robot_footprint_provenance"]["base_config_digest"]
            if base_config_digest is None:
                base_config_digest = str(candidate)
            elif candidate != base_config_digest:
                raise EvaluationCollectionError("evaluation records disagree on base_config_digest")
            all_ids.append(str(row["sample_id"]))
        records_bytes = _records_jsonl(rows)
        descriptors.append(
            RiskEvaluationShardDescriptor(
                shard_index=descriptor.shard_index,
                relative_root=f"shard-{descriptor.shard_index:05d}",
                sample_count=len(rows),
                source_risk_shard_semantic_digest=str(risk_shard.semantic_digest),
                ordered_sample_ids_digest_sha256=ordered_sample_ids_digest(
                    [str(row["sample_id"]) for row in rows]
                ),
                records_sha256=_sha256_bytes(records_bytes),
                semantic_digest_sha256=_records_semantic_digest(rows),
            )
        )
    if base_config_digest is None:
        raise EvaluationCollectionError("evaluation collection cannot be empty")
    if len(all_ids) != dataset.sample_count:
        raise EvaluationCollectionError("evaluation collection sample count mismatch")
    shard_values = [
        {
            "shard_index": item.shard_index,
            "relative_root": item.relative_root,
            "sample_count": item.sample_count,
            "source_risk_shard_semantic_digest": item.source_risk_shard_semantic_digest,
            "ordered_sample_ids_digest_sha256": item.ordered_sample_ids_digest_sha256,
            "records_sha256": item.records_sha256,
            "semantic_digest_sha256": item.semantic_digest_sha256,
        }
        for item in descriptors
    ]
    ordered_ids_digest = ordered_sample_ids_digest(all_ids)
    scope = {
        "collection_layout_version": RISK_EVALUATION_COLLECTION_LAYOUT_VERSION,
        "record_layout_version": "risk_evaluation_record_v1",
        "schema_version": SCHEMA_VERSION,
        "split": dataset.split,
        "risk_dataset_manifest_digest": dataset.risk_dataset_manifest_digest,
        "base_config_digest": base_config_digest,
        "grid": dict(dataset.manifest["grid"]),
        "shard_count": len(descriptors),
        "sample_count": len(all_ids),
        "shards": shard_values,
        "ordered_sample_ids_digest_sha256": ordered_ids_digest,
    }
    manifest = {
        **scope,
        "collection_semantic_digest_sha256": _collection_semantic_digest(scope),
    }
    return manifest, tuple(descriptors)


def _summary_bytes(descriptor: RiskEvaluationShardDescriptor, *, split: str) -> bytes:
    return _canonical_json_bytes(
        {
            "shard_layout_version": RISK_EVALUATION_SHARD_LAYOUT_VERSION,
            "record_layout_version": "risk_evaluation_record_v1",
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "shard_index": descriptor.shard_index,
            "sample_count": descriptor.sample_count,
            "source_risk_shard_semantic_digest": descriptor.source_risk_shard_semantic_digest,
            "ordered_sample_ids_digest_sha256": descriptor.ordered_sample_ids_digest_sha256,
            "records_sha256": descriptor.records_sha256,
            "semantic_digest_sha256": descriptor.semantic_digest_sha256,
            "files": {"records": _RECORDS_NAME, "summary": _SUMMARY_NAME},
        }
    )


def _write_checksums(root: Path) -> bytes:
    relative_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {_CHECKSUMS_NAME, _COMPLETE_NAME}
    )
    lines = "".join(f"{_sha256_file(root / relative)}  {relative}\n" for relative in relative_files)
    payload = lines.encode("ascii")
    (root / _CHECKSUMS_NAME).write_bytes(payload)
    return payload


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            _fsync_file(path)
        elif path.is_dir() and not path.is_symlink():
            try:
                descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            except OSError:
                continue
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _manifest_scope(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        key: manifest[key]
        for key in _MANIFEST_KEYS
        if key != "collection_semantic_digest_sha256"
    }


def _risk_shard_identity(risk_shard: object) -> tuple[
    tuple[object, ...], str, int, str, str
]:
    samples = getattr(risk_shard, "samples", None)
    summary = getattr(risk_shard, "summary", None)
    if not isinstance(samples, tuple) or not samples:
        raise EvaluationCollectionError("risk shard samples must be a non-empty tuple")
    if not isinstance(summary, Mapping):
        raise EvaluationCollectionError("risk shard summary is unavailable")
    split = _require_string(summary.get("split"), label="risk shard split")
    shard_index = _require_int(
        summary.get("shard_index"), label="risk shard index"
    )
    manifest_digest = _require_digest(
        getattr(risk_shard, "manifest_digest", None),
        label="risk shard manifest digest",
    )
    semantic_digest = _require_digest(
        getattr(risk_shard, "semantic_digest", None),
        label="risk shard semantic digest",
    )
    return samples, split, shard_index, manifest_digest, semantic_digest


def _normalise_replay_records(
    risk_shard: object,
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    samples, split, _, _, _ = _risk_shard_identity(risk_shard)
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise EvaluationCollectionError("evaluation replay records must be a sequence")
    if len(records) != len(samples):
        raise EvaluationCollectionError("evaluation replay record count mismatch")
    result: list[dict[str, object]] = []
    for index, (raw, sample) in enumerate(zip(records, samples, strict=True)):
        try:
            record = validate_production_evaluation_record(
                raw,
                expected_sample_id=sample.sample_id,
            )
        except (TypeError, ValueError) as exc:
            raise EvaluationCollectionError(
                f"invalid evaluation replay record {index}: {exc}"
            ) from exc
        if record["split"] != split:
            raise EvaluationCollectionError(
                f"evaluation replay record {index} split mismatch"
            )
        _validate_sample_join(record, sample)
        result.append(dict(record))
    sample_ids = tuple(str(record["sample_id"]) for record in result)
    if len(set(sample_ids)) != len(sample_ids):
        raise EvaluationCollectionError("evaluation replay sample IDs must be unique")
    return tuple(result)


def _replay_summary(
    *,
    split: str,
    shard_index: int,
    records: Sequence[Mapping[str, object]],
    risk_manifest_digest: str,
    risk_semantic_digest: str,
) -> dict[str, object]:
    records_bytes = _records_jsonl(records)
    scope: dict[str, object] = {
        "replay_shard_layout_version": RISK_EVALUATION_REPLAY_SHARD_LAYOUT_VERSION,
        "record_layout_version": "risk_evaluation_record_v1",
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "shard_index": shard_index,
        "sample_count": len(records),
        "source_risk_shard_manifest_digest": risk_manifest_digest,
        "source_risk_shard_semantic_digest": risk_semantic_digest,
        "ordered_sample_ids_digest_sha256": ordered_sample_ids_digest(
            [str(record["sample_id"]) for record in records]
        ),
        "records_sha256": _sha256_bytes(records_bytes),
    }
    scope["semantic_digest_sha256"] = _framed_digest(
        b"risk-evaluation-replay-shard-v1\0",
        scope,
    )
    return scope


def publish_risk_evaluation_replay_shard(
    output_dir: str | Path,
    *,
    risk_shard: object,
    records: Sequence[Mapping[str, object]],
) -> Path:
    """Publish one replay shard before final collection assembly."""

    output = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(output):
        raise FileExistsError(
            f"refusing to overwrite immutable evaluation replay shard: {output}"
        )
    rows = _normalise_replay_records(risk_shard, records)
    _, split, shard_index, manifest_digest, semantic_digest = (
        _risk_shard_identity(risk_shard)
    )
    summary = _replay_summary(
        split=split,
        shard_index=shard_index,
        records=rows,
        risk_manifest_digest=manifest_digest,
        risk_semantic_digest=semantic_digest,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        (staging / _RECORDS_NAME).write_bytes(_records_jsonl(rows))
        (staging / _SUMMARY_NAME).write_bytes(_canonical_json_bytes(summary))
        checksums = _write_checksums(staging)
        marker = {
            "replay_shard_layout_version": RISK_EVALUATION_REPLAY_SHARD_LAYOUT_VERSION,
            "semantic_digest_sha256": summary["semantic_digest_sha256"],
            "summary_sha256": _sha256_file(staging / _SUMMARY_NAME),
            "checksums_sha256": _sha256_bytes(checksums),
        }
        (staging / _COMPLETE_NAME).write_bytes(_canonical_json_bytes(marker))
        _fsync_tree(staging)
        load_risk_evaluation_replay_shard(staging, risk_shard=risk_shard)
        atomic_rename_noreplace(staging, output)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return output


def load_risk_evaluation_replay_shard(
    root: str | Path,
    *,
    risk_shard: object,
) -> LoadedRiskEvaluationReplayShard:
    """Reload one replay shard and rejoin every row to the supplied risk shard."""

    output = Path(os.path.abspath(os.fspath(root)))
    _require_directory(output, label="evaluation replay shard root")
    expected_names = {_RECORDS_NAME, _SUMMARY_NAME, _CHECKSUMS_NAME, _COMPLETE_NAME}
    entries = list(output.iterdir())
    if {entry.name for entry in entries} != expected_names or any(
        entry.is_symlink() for entry in entries
    ):
        raise EvaluationCollectionError("evaluation replay shard file layout mismatch")
    for entry in entries:
        _require_regular(entry, label=f"evaluation replay member {entry.name}")
    samples, split, shard_index, manifest_digest, semantic_digest = (
        _risk_shard_identity(risk_shard)
    )
    del samples
    summary = _read_canonical_json(
        output / _SUMMARY_NAME,
        label="evaluation replay summary",
    )
    expected_summary_keys = {
        "replay_shard_layout_version",
        "record_layout_version",
        "schema_version",
        "split",
        "shard_index",
        "sample_count",
        "source_risk_shard_manifest_digest",
        "source_risk_shard_semantic_digest",
        "ordered_sample_ids_digest_sha256",
        "records_sha256",
        "semantic_digest_sha256",
    }
    if set(summary) != expected_summary_keys:
        raise EvaluationCollectionError("evaluation replay summary fields mismatch")
    rows = _load_records(output / _RECORDS_NAME, expected_split=split)
    rows = _normalise_replay_records(risk_shard, rows)
    expected_summary = _replay_summary(
        split=split,
        shard_index=shard_index,
        records=rows,
        risk_manifest_digest=manifest_digest,
        risk_semantic_digest=semantic_digest,
    )
    if summary != expected_summary:
        raise EvaluationCollectionError(
            "evaluation replay summary does not match the supplied risk shard"
        )
    checksums = _parse_checksums(output)
    if set(checksums) != {_RECORDS_NAME, _SUMMARY_NAME}:
        raise EvaluationCollectionError("evaluation replay checksum coverage mismatch")
    for relative, expected in checksums.items():
        if _sha256_file(output / relative) != expected:
            raise EvaluationCollectionError(
                f"evaluation replay checksum mismatch: {relative}"
            )
    marker = _read_canonical_json(
        output / _COMPLETE_NAME,
        label="evaluation replay completion marker",
    )
    if marker != {
        "replay_shard_layout_version": RISK_EVALUATION_REPLAY_SHARD_LAYOUT_VERSION,
        "semantic_digest_sha256": summary["semantic_digest_sha256"],
        "summary_sha256": _sha256_file(output / _SUMMARY_NAME),
        "checksums_sha256": _sha256_file(output / _CHECKSUMS_NAME),
    }:
        raise EvaluationCollectionError("evaluation replay completion marker mismatch")
    sample_ids = tuple(str(record["sample_id"]) for record in rows)
    return LoadedRiskEvaluationReplayShard(
        root=output,
        split=split,
        shard_index=shard_index,
        records=rows,
        sample_ids=sample_ids,
        ordered_sample_ids_digest_sha256=str(
            summary["ordered_sample_ids_digest_sha256"]
        ),
        source_risk_shard_manifest_digest=manifest_digest,
        source_risk_shard_semantic_digest=semantic_digest,
        semantic_digest_sha256=str(summary["semantic_digest_sha256"]),
    )


def publish_risk_evaluation_collection(
    output_dir: str | Path,
    *,
    dataset: LoadedRiskDataset,
    records_by_shard: Mapping[int, Sequence[Mapping[str, object]]],
) -> Path:
    """Publish records atomically after joining them to authenticated shards."""

    if not isinstance(dataset, LoadedRiskDataset):
        raise EvaluationCollectionError("dataset must be an authenticated LoadedRiskDataset")
    output = Path(os.path.abspath(os.fspath(output_dir)))
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite immutable evaluation collection: {output}")
    risk_shards = _load_risk_shards(dataset)
    rows_by_shard = _normalise_records_by_shard(dataset, records_by_shard, risk_shards)
    manifest, descriptors = _build_manifest(dataset, rows_by_shard, risk_shards)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        (staging / _MANIFEST_NAME).write_bytes(_canonical_json_bytes(manifest))
        for rows, descriptor in zip(rows_by_shard, descriptors, strict=True):
            shard_root = staging / descriptor.relative_root
            shard_root.mkdir()
            (shard_root / _RECORDS_NAME).write_bytes(_records_jsonl(rows))
            (shard_root / _SUMMARY_NAME).write_bytes(
                _summary_bytes(descriptor, split=dataset.split)
            )
        checksums = _write_checksums(staging)
        marker = {
            "collection_layout_version": RISK_EVALUATION_COLLECTION_LAYOUT_VERSION,
            "collection_semantic_digest_sha256": manifest[
                "collection_semantic_digest_sha256"
            ],
            "evaluation_manifest_sha256": _sha256_file(staging / _MANIFEST_NAME),
            "checksums_sha256": _sha256_bytes(checksums),
        }
        (staging / _COMPLETE_NAME).write_bytes(_canonical_json_bytes(marker))
        _fsync_tree(staging)
        loaded = load_risk_evaluation_collection(
            staging,
            dataset=dataset,
            expected_manifest_digest=dataset.risk_dataset_manifest_digest,
        )
        if loaded.collection_semantic_digest_sha256 != manifest[
            "collection_semantic_digest_sha256"
        ]:
            raise EvaluationCollectionError("staging collection digest mismatch")
        atomic_rename_noreplace(staging, output)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return output


def _parse_checksums(root: Path) -> dict[str, str]:
    _require_regular(root / _CHECKSUMS_NAME, label="evaluation checksum manifest")
    entries: dict[str, str] = {}
    for line in (root / _CHECKSUMS_NAME).read_text(encoding="ascii").splitlines():
        if "  " not in line:
            raise EvaluationCollectionError("evaluation checksum manifest is malformed")
        digest, relative = line.split("  ", 1)
        _require_digest(digest, label=f"checksum {relative}")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise EvaluationCollectionError("evaluation checksum path is unsafe")
        if relative in entries:
            raise EvaluationCollectionError("evaluation checksum paths are duplicated")
        entries[relative] = digest
    return entries


def _descriptor(value: object, *, position: int) -> RiskEvaluationShardDescriptor:
    if not isinstance(value, Mapping):
        raise EvaluationCollectionError(f"evaluation shard descriptor {position} is invalid")
    expected = {
        "shard_index",
        "relative_root",
        "sample_count",
        "source_risk_shard_semantic_digest",
        "ordered_sample_ids_digest_sha256",
        "records_sha256",
        "semantic_digest_sha256",
    }
    if set(value) != expected:
        raise EvaluationCollectionError(f"evaluation shard descriptor {position} fields mismatch")
    shard_index = _require_int(value.get("shard_index"), label="shard_index")
    if shard_index != position:
        raise EvaluationCollectionError("evaluation shard indices must be contiguous")
    relative_root = _require_string(value.get("relative_root"), label="relative_root")
    if relative_root != f"shard-{position:05d}":
        raise EvaluationCollectionError("evaluation shard relative_root order mismatch")
    if Path(relative_root).name != relative_root:
        raise EvaluationCollectionError("evaluation shard relative_root is unsafe")
    return RiskEvaluationShardDescriptor(
        shard_index=shard_index,
        relative_root=relative_root,
        sample_count=_require_int(value.get("sample_count"), label="sample_count", minimum=1),
        source_risk_shard_semantic_digest=_require_digest(
            value.get("source_risk_shard_semantic_digest"),
            label="source_risk_shard_semantic_digest",
        ),
        ordered_sample_ids_digest_sha256=_require_digest(
            value.get("ordered_sample_ids_digest_sha256"),
            label="ordered_sample_ids_digest_sha256",
        ),
        records_sha256=_require_digest(value.get("records_sha256"), label="records_sha256"),
        semantic_digest_sha256=_require_digest(
            value.get("semantic_digest_sha256"),
            label="semantic_digest_sha256",
        ),
    )


def load_risk_evaluation_collection(
    root: str | Path,
    *,
    dataset: LoadedRiskDataset,
    expected_manifest_digest: str | None = None,
) -> LoadedRiskEvaluationCollection:
    """Reload and reauthenticate an evaluation-record publication."""

    if not isinstance(dataset, LoadedRiskDataset):
        raise EvaluationCollectionError("dataset must be an authenticated LoadedRiskDataset")
    output = Path(os.path.abspath(os.fspath(root)))
    _require_directory(output, label="evaluation collection root")
    entries = list(output.iterdir())
    if any(entry.is_symlink() for entry in entries):
        raise EvaluationCollectionError("evaluation collection forbids symlink entries")
    _require_regular(
        output / _COMPLETE_NAME,
        label="evaluation completion marker",
    )
    manifest = _read_canonical_json(output / _MANIFEST_NAME, label="evaluation manifest")
    if set(manifest) != _MANIFEST_KEYS:
        raise EvaluationCollectionError("evaluation manifest fields mismatch")
    if manifest["collection_layout_version"] != RISK_EVALUATION_COLLECTION_LAYOUT_VERSION:
        raise EvaluationCollectionError("evaluation collection layout mismatch")
    if manifest["record_layout_version"] != "risk_evaluation_record_v1":
        raise EvaluationCollectionError("evaluation record layout mismatch")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["split"] != dataset.split:
        raise EvaluationCollectionError("evaluation manifest schema/split mismatch")
    source_digest = _require_digest(
        manifest["risk_dataset_manifest_digest"],
        label="risk_dataset_manifest_digest",
    )
    if source_digest != dataset.risk_dataset_manifest_digest:
        raise EvaluationCollectionError("evaluation risk dataset manifest digest mismatch")
    if expected_manifest_digest is not None and source_digest != expected_manifest_digest:
        raise EvaluationCollectionError("expected risk manifest digest mismatch")
    _require_digest(manifest["base_config_digest"], label="base_config_digest", length=32)
    shard_values = manifest["shards"]
    if not isinstance(shard_values, list) or len(shard_values) != dataset.shards.__len__():
        raise EvaluationCollectionError("evaluation manifest shard count mismatch")
    descriptors = tuple(
        _descriptor(value, position=index) for index, value in enumerate(shard_values)
    )
    if manifest["shard_count"] != len(descriptors) or manifest["sample_count"] != dataset.sample_count:
        raise EvaluationCollectionError("evaluation manifest counts mismatch")
    if manifest["shard_count"] != len(dataset.shards):
        raise EvaluationCollectionError("evaluation/risk shard count mismatch")
    scope = _manifest_scope(manifest)
    if manifest["collection_semantic_digest_sha256"] != _collection_semantic_digest(scope):
        raise EvaluationCollectionError("evaluation collection semantic digest mismatch")

    expected_files = {_MANIFEST_NAME, _CHECKSUMS_NAME, _COMPLETE_NAME}
    expected_files.update(
        f"{descriptor.relative_root}/{name}"
        for descriptor in descriptors
        for name in (_RECORDS_NAME, _SUMMARY_NAME)
    )
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    actual_dirs = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_dir()
    }
    expected_dirs = {descriptor.relative_root for descriptor in descriptors}
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise EvaluationCollectionError("evaluation collection file layout mismatch")
    checksums = _parse_checksums(output)
    checksum_targets = expected_files - {_CHECKSUMS_NAME, _COMPLETE_NAME}
    if set(checksums) != checksum_targets:
        raise EvaluationCollectionError("evaluation checksum coverage mismatch")
    for relative, expected in checksums.items():
        if _sha256_file(output / relative) != expected:
            raise EvaluationCollectionError(f"evaluation checksum mismatch: {relative}")

    marker = _read_canonical_json(output / _COMPLETE_NAME, label="evaluation completion marker")
    if set(marker) != _MARKER_KEYS:
        raise EvaluationCollectionError("evaluation completion marker fields mismatch")
    if marker["collection_layout_version"] != RISK_EVALUATION_COLLECTION_LAYOUT_VERSION:
        raise EvaluationCollectionError("evaluation completion marker layout mismatch")
    if marker["collection_semantic_digest_sha256"] != manifest[
        "collection_semantic_digest_sha256"
    ] or marker["evaluation_manifest_sha256"] != _sha256_file(output / _MANIFEST_NAME):
        raise EvaluationCollectionError("evaluation completion marker digest mismatch")
    if marker["checksums_sha256"] != _sha256_file(output / _CHECKSUMS_NAME):
        raise EvaluationCollectionError("evaluation completion marker checksum mismatch")

    risk_shards = _load_risk_shards(dataset)
    records_by_shard: dict[int, tuple[dict[str, object], ...]] = {}
    all_ids: list[str] = []
    for descriptor, risk_descriptor, risk_shard in zip(
        descriptors, dataset.shards, risk_shards, strict=True
    ):
        if descriptor.source_risk_shard_semantic_digest != risk_descriptor.semantic_digest:
            raise EvaluationCollectionError(
                f"evaluation source risk digest mismatch for shard {descriptor.shard_index}"
            )
        shard_root = output / descriptor.relative_root
        records_path = shard_root / _RECORDS_NAME
        summary = _read_canonical_json(shard_root / _SUMMARY_NAME, label="evaluation shard summary")
        if set(summary) != _SHARD_SUMMARY_KEYS:
            raise EvaluationCollectionError("evaluation shard summary fields mismatch")
        expected_summary = json.loads(_summary_bytes(descriptor, split=dataset.split))
        if summary != expected_summary:
            raise EvaluationCollectionError(
                f"evaluation shard summary mismatch for shard {descriptor.shard_index}"
            )
        rows = _load_records(records_path, expected_split=dataset.split)
        if len(rows) != descriptor.sample_count or len(rows) != len(risk_shard.samples):
            raise EvaluationCollectionError(
                f"evaluation shard sample count mismatch for shard {descriptor.shard_index}"
            )
        expected_ids = tuple(sample.sample_id for sample in risk_shard.samples)
        actual_ids = tuple(str(row["sample_id"]) for row in rows)
        if actual_ids != expected_ids:
            raise EvaluationCollectionError(
                f"evaluation shard sample order mismatch for shard {descriptor.shard_index}"
            )
        for row, sample in zip(rows, risk_shard.samples, strict=True):
            _validate_sample_join(row, sample)
        if ordered_sample_ids_digest(actual_ids) != descriptor.ordered_sample_ids_digest_sha256:
            raise EvaluationCollectionError("evaluation shard ordered sample-ID digest mismatch")
        if _sha256_file(records_path) != descriptor.records_sha256:
            raise EvaluationCollectionError("evaluation records file digest mismatch")
        if _records_semantic_digest(rows) != descriptor.semantic_digest_sha256:
            raise EvaluationCollectionError("evaluation shard semantic digest mismatch")
        records_by_shard[descriptor.shard_index] = rows
        all_ids.extend(actual_ids)
    if ordered_sample_ids_digest(all_ids) != manifest["ordered_sample_ids_digest_sha256"]:
        raise EvaluationCollectionError("evaluation ordered sample-ID digest mismatch")
    return LoadedRiskEvaluationCollection(
        root=output,
        manifest=dict(manifest),
        shards=descriptors,
        records_by_shard=records_by_shard,
        sample_ids=tuple(all_ids),
        sample_ids_digest_sha256=str(manifest["ordered_sample_ids_digest_sha256"]),
        collection_semantic_digest_sha256=str(
            manifest["collection_semantic_digest_sha256"]
        ),
        risk_dataset_manifest_digest=source_digest,
    )


__all__ = [
    "EvaluationCollectionError",
    "LoadedRiskEvaluationCollection",
    "LoadedRiskEvaluationReplayShard",
    "RISK_EVALUATION_COLLECTION_LAYOUT_VERSION",
    "RISK_EVALUATION_REPLAY_SHARD_LAYOUT_VERSION",
    "RISK_EVALUATION_SHARD_LAYOUT_VERSION",
    "RiskEvaluationShardDescriptor",
    "load_risk_evaluation_collection",
    "load_risk_evaluation_replay_shard",
    "ordered_sample_ids_digest",
    "publish_risk_evaluation_collection",
    "publish_risk_evaluation_replay_shard",
]
