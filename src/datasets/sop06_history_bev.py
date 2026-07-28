"""Deterministic immutable storage for SOP06 history-only BEV observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from types import MappingProxyType
from typing import Mapping, Sequence
import zipfile

import numpy as np

from src.contracts import (
    ARRAY_DTYPE,
    LONG40_HISTORY_STEPS,
    N_HISTORY_CHANNELS,
    N_STATE_CHANNELS,
)
from src.utils.atomic_publish import atomic_rename_noreplace


SOP06_HISTORY_SHARD_VERSION = "sop06_history_bev_shard_v1"
_PAYLOAD = "observations.npz"
_METADATA = "metadata.jsonl"
_SUMMARY = "summary.json"
_CHECKSUMS = "checksums.json"
_COMPLETE = "COMPLETE.json"
_FILES = frozenset({_PAYLOAD, _METADATA, _SUMMARY, _CHECKSUMS, _COMPLETE})
_SOURCE_FAMILIES = frozenset({"natural", "a_supplement"})
_SOURCE_MODES = frozenset({"complete_mother", "partial_m6_reconstruction"})
_SPLITS = frozenset({"train", "calibration", "val", "test"})
_REGIMES = frozenset({"seen_then_occluded", "unseen_in_history_window"})
_FORBIDDEN_METADATA_TOKENS = (
    "future",
    "oracle",
    "angle",
    "attempt",
    "rejection",
    "collision",
    "clearance",
    "risk",
)


@dataclass(frozen=True)
class Sop06HistoryBevSample:
    sample_id: str
    mother_id: str
    split: str
    regime: str
    bev_history: np.ndarray
    state_channels: np.ndarray
    renderer_metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "renderer_metadata",
            MappingProxyType(dict(self.renderer_metadata)),
        )


@dataclass(frozen=True)
class Sop06HistoryShardProvenance:
    source_family: str
    source_mode: str
    split: str
    source_publication_semantic_digest: str
    final_release_identity: str
    final_scenario_root: str


@dataclass(frozen=True)
class LoadedSop06HistoryShard:
    root: Path
    shard_index: int
    samples: tuple[Sop06HistoryBevSample, ...]
    provenance: Sop06HistoryShardProvenance
    semantic_digest: str
    summary: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary)))


@dataclass(frozen=True)
class Sop06HistoryShardCheckpoint:
    root: Path
    shard_index: int
    sample_ids: tuple[str, ...]
    mother_ids: tuple[str, ...]
    provenance: Sop06HistoryShardProvenance
    semantic_digest: str
    summary: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary)))


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("SOP06 history metadata must be canonical JSON") from exc


def _json_file(value: object) -> bytes:
    return _canonical_json(value) + b"\n"


def _sha256_file(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as exc:
        raise ValueError(f"failed to checksum SOP06 history file: {path.name}") from exc


def _require_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_relative_path(value: object, *, name: str) -> str:
    raw = _require_identifier(value, name=name)
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be repository-relative")
    return path.as_posix()


def _validate_provenance(
    provenance: Sop06HistoryShardProvenance,
) -> Sop06HistoryShardProvenance:
    if not isinstance(provenance, Sop06HistoryShardProvenance):
        raise TypeError("provenance must be a Sop06HistoryShardProvenance")
    if provenance.source_family not in _SOURCE_FAMILIES:
        raise ValueError("source_family is invalid")
    if provenance.source_mode not in _SOURCE_MODES:
        raise ValueError("source_mode is invalid")
    if provenance.split not in _SPLITS:
        raise ValueError("provenance split is invalid")
    _require_sha256(
        provenance.source_publication_semantic_digest,
        name="source_publication_semantic_digest",
    )
    _require_sha256(
        provenance.final_release_identity,
        name="final_release_identity",
    )
    _validate_relative_path(
        provenance.final_scenario_root,
        name="final_scenario_root",
    )
    return provenance


def _validate_renderer_metadata(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("renderer_metadata must be a mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or not isinstance(item, str):
            raise ValueError("renderer_metadata must contain non-empty string keys")
        if any(token in key.lower() for token in _FORBIDDEN_METADATA_TOKENS):
            raise ValueError("renderer_metadata contains oracle-only information")
        result[key] = item
    return dict(sorted(result.items()))


def _validate_sample(sample: Sop06HistoryBevSample) -> None:
    if not isinstance(sample, Sop06HistoryBevSample):
        raise TypeError("samples must contain Sop06HistoryBevSample values")
    _require_identifier(sample.sample_id, name="sample_id")
    _require_identifier(sample.mother_id, name="mother_id")
    if sample.split not in _SPLITS:
        raise ValueError("sample split is invalid")
    if sample.regime not in _REGIMES:
        raise ValueError("sample regime is invalid")
    history = np.asarray(sample.bev_history)
    state = np.asarray(sample.state_channels)
    if (
        history.ndim != 4
        or history.shape[:2] != (LONG40_HISTORY_STEPS, N_HISTORY_CHANNELS)
        or state.ndim != 3
        or state.shape[0] != N_STATE_CHANNELS
        or history.shape[2:] != state.shape[1:]
    ):
        raise ValueError("SOP06 history/state array shape is invalid")
    if history.dtype != ARRAY_DTYPE or state.dtype != ARRAY_DTYPE:
        raise ValueError("SOP06 history/state arrays must be float32")
    if not np.isfinite(history).all() or not np.isfinite(state).all():
        raise ValueError("SOP06 history/state arrays must be finite")
    _validate_renderer_metadata(sample.renderer_metadata)


def _row(
    sample: Sop06HistoryBevSample,
    *,
    shard_index: int,
    row_index: int,
) -> dict[str, object]:
    return {
        "version": SOP06_HISTORY_SHARD_VERSION,
        "shard_index": shard_index,
        "row_index": row_index,
        "sample_id": sample.sample_id,
        "scenario_id": sample.sample_id,
        "mother_id": sample.mother_id,
        "split": sample.split,
        "regime": sample.regime,
        "renderer_metadata": _validate_renderer_metadata(
            sample.renderer_metadata
        ),
    }


def _semantic_digest(
    history: np.ndarray,
    state: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    provenance: Sop06HistoryShardProvenance,
    *,
    shard_index: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(SOP06_HISTORY_SHARD_VERSION.encode("ascii") + b"\0")
    digest.update(str(shard_index).encode("ascii") + b"\0")
    digest.update(_canonical_json(asdict(provenance)) + b"\0")
    digest.update(_canonical_json(list(rows)) + b"\0")
    for name, value in (("bev_history", history), ("state_channels", state)):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(_canonical_json(list(array.shape)) + b"\0")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _write_deterministic_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for name in sorted(arrays):
            raw = np.asarray(arrays[name])
            array = raw if raw.ndim == 0 else np.ascontiguousarray(raw)
            info = zipfile.ZipInfo(
                f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            with archive.open(info, "w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, array, allow_pickle=False)


def write_sop06_history_shard(
    samples: Sequence[Sop06HistoryBevSample],
    output_dir: str | Path,
    *,
    shard_index: int,
    expected_sample_count: int,
    provenance: Sop06HistoryShardProvenance,
) -> dict[str, Path]:
    if isinstance(shard_index, bool) or not isinstance(shard_index, int) or shard_index < 0:
        raise ValueError("shard_index must be a nonnegative integer")
    if (
        isinstance(expected_sample_count, bool)
        or not isinstance(expected_sample_count, int)
        or expected_sample_count <= 0
    ):
        raise ValueError("expected_sample_count must be a positive integer")
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence")
    _validate_provenance(provenance)
    ordered = tuple(sorted(samples, key=lambda item: item.sample_id))
    if len(ordered) != expected_sample_count:
        raise ValueError("expected_sample_count differs from shard boundary")
    for sample in ordered:
        _validate_sample(sample)
        if sample.split != provenance.split:
            raise ValueError("sample split differs from shard provenance")
    sample_ids = tuple(sample.sample_id for sample in ordered)
    mother_ids = tuple(sample.mother_id for sample in ordered)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate sample_id in SOP06 history shard")
    if len(set(mother_ids)) != len(mother_ids):
        raise ValueError("duplicate mother_id in SOP06 history shard")

    history = np.stack([sample.bev_history for sample in ordered]).astype(
        ARRAY_DTYPE, copy=False
    )
    state = np.stack([sample.state_channels for sample in ordered]).astype(
        ARRAY_DTYPE, copy=False
    )
    rows = tuple(
        _row(sample, shard_index=shard_index, row_index=index)
        for index, sample in enumerate(ordered)
    )
    semantic_digest = _semantic_digest(
        history,
        state,
        rows,
        provenance,
        shard_index=shard_index,
    )
    meta = {
        "version": SOP06_HISTORY_SHARD_VERSION,
        "shard_index": shard_index,
        "sample_ids": list(sample_ids),
        "semantic_digest": semantic_digest,
        "provenance": asdict(provenance),
    }
    summary = {
        "version": SOP06_HISTORY_SHARD_VERSION,
        "shard_index": shard_index,
        "split": provenance.split,
        "expected_sample_count": expected_sample_count,
        "sample_count": len(ordered),
        "first_sample_id": sample_ids[0],
        "last_sample_id": sample_ids[-1],
        "semantic_digest": semantic_digest,
        "provenance": asdict(provenance),
        "files": {
            "payload": _PAYLOAD,
            "metadata": _METADATA,
            "summary": _SUMMARY,
        },
    }

    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable shard: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        _write_deterministic_npz(
            staging / _PAYLOAD,
            {
                "bev_history": history,
                "meta_json": np.asarray(_canonical_json(meta).decode("ascii")),
                "state_channels": state,
            },
        )
        (staging / _METADATA).write_bytes(
            b"".join(_json_file(row) for row in rows)
        )
        (staging / _SUMMARY).write_bytes(_json_file(summary))
        checksums = {
            name: _sha256_file(staging / name)
            for name in (_PAYLOAD, _METADATA, _SUMMARY)
        }
        (staging / _CHECKSUMS).write_bytes(_json_file(checksums))
        (staging / _COMPLETE).write_bytes(
            _json_file(
                {
                    "version": SOP06_HISTORY_SHARD_VERSION,
                    "shard_index": shard_index,
                    "sample_count": len(ordered),
                    "semantic_digest": semantic_digest,
                }
            )
        )
        atomic_rename_noreplace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "directory": output,
        "payload": output / _PAYLOAD,
        "metadata": output / _METADATA,
        "summary": output / _SUMMARY,
    }


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read SOP06 history JSON: {path.name}") from exc


def _read_rows(path: Path) -> tuple[dict[str, object], ...]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("failed to read SOP06 history metadata") from exc
    rows: list[dict[str, object]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("SOP06 history metadata is invalid JSONL") from exc
        if not isinstance(value, dict):
            raise ValueError("SOP06 history metadata rows must be objects")
        rows.append(value)
    return tuple(rows)


def load_sop06_history_shard_checkpoint(
    input_dir: str | Path,
) -> Sop06HistoryShardCheckpoint:
    """Validate resumable shard metadata without opening observation arrays."""

    root = Path(input_dir)
    if not root.is_dir() or {path.name for path in root.iterdir()} != _FILES:
        raise ValueError("SOP06 history shard file set mismatch")
    checksums = _read_json(root / _CHECKSUMS)
    if not isinstance(checksums, dict) or set(checksums) != {
        _PAYLOAD,
        _METADATA,
        _SUMMARY,
    }:
        raise ValueError("SOP06 history checksum schema mismatch")
    for name in (_PAYLOAD, _METADATA, _SUMMARY):
        _require_sha256(checksums.get(name), name=f"{name} checksum")
    for name in (_METADATA, _SUMMARY):
        if _sha256_file(root / name) != checksums[name]:
            raise ValueError(f"SOP06 history checksum mismatch: {name}")
    try:
        if (root / _PAYLOAD).stat().st_size <= 0:
            raise ValueError("SOP06 history payload is empty")
    except OSError as exc:
        raise ValueError("SOP06 history payload is unavailable") from exc

    summary = _read_json(root / _SUMMARY)
    complete = _read_json(root / _COMPLETE)
    if not isinstance(summary, dict) or not isinstance(complete, dict):
        raise ValueError("SOP06 history summary/complete must be objects")
    if summary.get("version") != SOP06_HISTORY_SHARD_VERSION:
        raise ValueError("SOP06 history shard version mismatch")
    try:
        provenance = Sop06HistoryShardProvenance(**summary["provenance"])
    except (KeyError, TypeError) as exc:
        raise ValueError("SOP06 history provenance schema mismatch") from exc
    _validate_provenance(provenance)
    count = summary.get("sample_count")
    shard_index = summary.get("shard_index")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or shard_index < 0
        or summary.get("expected_sample_count") != count
        or summary.get("files")
        != {
            "payload": _PAYLOAD,
            "metadata": _METADATA,
            "summary": _SUMMARY,
        }
    ):
        raise ValueError("SOP06 history checkpoint summary differs")

    rows = _read_rows(root / _METADATA)
    if len(rows) != count:
        raise ValueError("SOP06 history checkpoint row count differs")
    sample_ids: list[str] = []
    mother_ids: list[str] = []
    expected_row_keys = {
        "version",
        "shard_index",
        "row_index",
        "sample_id",
        "scenario_id",
        "mother_id",
        "split",
        "regime",
        "renderer_metadata",
    }
    for index, row in enumerate(rows):
        sample_id = _require_identifier(row.get("sample_id"), name="sample_id")
        mother_id = _require_identifier(row.get("mother_id"), name="mother_id")
        metadata = row.get("renderer_metadata")
        if (
            set(row) != expected_row_keys
            or row.get("version") != SOP06_HISTORY_SHARD_VERSION
            or row.get("shard_index") != shard_index
            or row.get("row_index") != index
            or row.get("scenario_id") != sample_id
            or row.get("split") != provenance.split
            or row.get("regime") not in _REGIMES
            or not isinstance(metadata, dict)
        ):
            raise ValueError("SOP06 history checkpoint metadata row differs")
        _validate_renderer_metadata(metadata)
        sample_ids.append(sample_id)
        mother_ids.append(mother_id)
    ordered_ids = tuple(sample_ids)
    ordered_mothers = tuple(mother_ids)
    if ordered_ids != tuple(sorted(ordered_ids)) or len(set(ordered_ids)) != count:
        raise ValueError("SOP06 history checkpoint sample IDs differ")
    if len(set(ordered_mothers)) != count:
        raise ValueError("SOP06 history checkpoint mother IDs differ")

    semantic_digest = _require_sha256(
        summary.get("semantic_digest"),
        name="semantic_digest",
    )
    if (
        summary.get("first_sample_id") != ordered_ids[0]
        or summary.get("last_sample_id") != ordered_ids[-1]
        or summary.get("split") != provenance.split
        or complete
        != {
            "version": SOP06_HISTORY_SHARD_VERSION,
            "shard_index": shard_index,
            "sample_count": count,
            "semantic_digest": semantic_digest,
        }
    ):
        raise ValueError("SOP06 history checkpoint identity differs")
    return Sop06HistoryShardCheckpoint(
        root=root,
        shard_index=shard_index,
        sample_ids=ordered_ids,
        mother_ids=ordered_mothers,
        provenance=provenance,
        semantic_digest=semantic_digest,
        summary=summary,
    )


def load_sop06_history_shard(
    input_dir: str | Path,
) -> LoadedSop06HistoryShard:
    root = Path(input_dir)
    if not root.is_dir() or {path.name for path in root.iterdir()} != _FILES:
        raise ValueError("SOP06 history shard file set mismatch")
    checksums = _read_json(root / _CHECKSUMS)
    if not isinstance(checksums, dict) or set(checksums) != {
        _PAYLOAD,
        _METADATA,
        _SUMMARY,
    }:
        raise ValueError("SOP06 history checksum schema mismatch")
    for name, expected in checksums.items():
        if not isinstance(expected, str) or _sha256_file(root / name) != expected:
            raise ValueError(f"SOP06 history checksum mismatch: {name}")
    summary = _read_json(root / _SUMMARY)
    complete = _read_json(root / _COMPLETE)
    if not isinstance(summary, dict) or not isinstance(complete, dict):
        raise ValueError("SOP06 history summary/complete must be objects")
    if summary.get("version") != SOP06_HISTORY_SHARD_VERSION:
        raise ValueError("SOP06 history shard version mismatch")
    try:
        provenance = Sop06HistoryShardProvenance(**summary["provenance"])
    except (KeyError, TypeError) as exc:
        raise ValueError("SOP06 history provenance schema mismatch") from exc
    _validate_provenance(provenance)
    rows = _read_rows(root / _METADATA)
    try:
        with np.load(root / _PAYLOAD, allow_pickle=False) as payload:
            if set(payload.files) != {"bev_history", "state_channels", "meta_json"}:
                raise ValueError("SOP06 history payload schema mismatch")
            history = np.asarray(payload["bev_history"])
            state = np.asarray(payload["state_channels"])
            raw_meta = np.asarray(payload["meta_json"])
            if raw_meta.shape != () or raw_meta.dtype.kind not in "SU":
                raise ValueError("SOP06 history payload meta_json is invalid")
            meta = json.loads(str(raw_meta.item()))
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("SOP06 history"):
            raise
        raise ValueError("failed to load SOP06 history payload") from exc
    count = summary.get("sample_count")
    shard_index = summary.get("shard_index")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or shard_index < 0
        or len(rows) != count
        or history.shape[0] != count
        or state.shape[0] != count
    ):
        raise ValueError("SOP06 history shard counts differ")
    samples: list[Sop06HistoryBevSample] = []
    for index, row in enumerate(rows):
        metadata = row.get("renderer_metadata")
        if not isinstance(metadata, dict):
            raise ValueError("SOP06 history renderer metadata is invalid")
        sample = Sop06HistoryBevSample(
            sample_id=str(row.get("sample_id")),
            mother_id=str(row.get("mother_id")),
            split=str(row.get("split")),
            regime=str(row.get("regime")),
            bev_history=np.array(
                history[index], dtype=np.float32, order="C", copy=True
            ),
            state_channels=np.array(
                state[index], dtype=np.float32, order="C", copy=True
            ),
            renderer_metadata={str(key): str(value) for key, value in metadata.items()},
        )
        _validate_sample(sample)
        if row != _row(sample, shard_index=shard_index, row_index=index):
            raise ValueError("SOP06 history metadata row differs")
        samples.append(sample)
    sample_ids = tuple(sample.sample_id for sample in samples)
    if sample_ids != tuple(sorted(sample_ids)) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("SOP06 history sample IDs are not unique and sorted")
    mother_ids = tuple(sample.mother_id for sample in samples)
    if len(set(mother_ids)) != len(mother_ids):
        raise ValueError("SOP06 history mother IDs are not unique")
    semantic_digest = _semantic_digest(
        history,
        state,
        rows,
        provenance,
        shard_index=shard_index,
    )
    if (
        summary.get("semantic_digest") != semantic_digest
        or meta.get("semantic_digest") != semantic_digest
        or meta.get("sample_ids") != list(sample_ids)
        or meta.get("provenance") != asdict(provenance)
        or complete
        != {
            "version": SOP06_HISTORY_SHARD_VERSION,
            "shard_index": shard_index,
            "sample_count": count,
            "semantic_digest": semantic_digest,
        }
    ):
        raise ValueError("SOP06 history semantic identity mismatch")
    return LoadedSop06HistoryShard(
        root=root,
        shard_index=shard_index,
        samples=tuple(samples),
        provenance=provenance,
        semantic_digest=semantic_digest,
        summary=summary,
    )


__all__ = (
    "LoadedSop06HistoryShard",
    "SOP06_HISTORY_SHARD_VERSION",
    "Sop06HistoryBevSample",
    "Sop06HistoryShardCheckpoint",
    "Sop06HistoryShardProvenance",
    "load_sop06_history_shard",
    "load_sop06_history_shard_checkpoint",
    "write_sop06_history_shard",
)
