"""Authenticated single-route trajectory storage for SOP05R TEB v2."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from src.contracts import LocalTrajectory
from src.planning.lightweight_teb import PlannedTebRoute
from src.utils.atomic_publish import atomic_rename_noreplace

from .sop05r_contracts import (
    SOP05R_TEB_PLANNER_VERSION,
    SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION,
)
from .sop05r_teb_event_sampler import Sop05rTebTrajectoryRecord


SOP05R_TEB_TRAJECTORY_STORE_VERSION = SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION
SOP05R_TEB_TRAJECTORY_RECORD_VERSION = "sop05r_teb_trajectory_record_v2"
SOP05R_TEB_TRAJECTORY_COMPLETION_VERSION = "sop05r_teb_trajectory_complete_v2"
_MANIFEST = "manifest.json"
_PAYLOAD = "trajectories.npz"
_SUMMARY = "summary.json"
_CHECKSUMS = "checksums.json"
_COMPLETE = "COMPLETE.json"
_ARRAY_NAMES = (
    "goal_world_pose",
    "band_poses_world",
    "band_interval_dt_s",
    "route_sample_times_s",
    "route_poses_world",
    "route_controls",
    "suffix_poses",
    "suffix_controls",
    "suffix_swept_mask",
    "suffix_tta_map",
    "suffix_braking_map",
    "suffix_centerline_map",
)


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
        raise ValueError("TEB trajectory evidence must be canonical JSON") from exc


def _json_file(value: object) -> bytes:
    return _canonical_json(value) + b"\n"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"failed to checksum trajectory artifact: {path.name}") from exc


def _array_digest(name: str, array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(b"sop05r_teb_trajectory_array_v2\0")
    digest.update(name.encode("ascii") + b"\0")
    digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
    digest.update(_canonical_json(list(contiguous.shape)))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _record_arrays(record: Sop05rTebTrajectoryRecord) -> dict[str, np.ndarray]:
    route = record.full_route
    suffix = record.nominal_trajectory
    return {
        "goal_world_pose": np.asarray(record.shared_goal_world_pose),
        "band_poses_world": route.band_poses_world,
        "band_interval_dt_s": route.band_interval_dt_s,
        "route_sample_times_s": route.sample_times_s,
        "route_poses_world": route.sampled_poses_world,
        "route_controls": route.sampled_controls,
        "suffix_poses": suffix.poses,
        "suffix_controls": suffix.controls,
        "suffix_swept_mask": suffix.swept_mask,
        "suffix_tta_map": suffix.tta_map,
        "suffix_braking_map": suffix.braking_map,
        "suffix_centerline_map": suffix.centerline_map,
    }


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.ascontiguousarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def _record_row(
    record: Sop05rTebTrajectoryRecord,
    *,
    row_index: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if record.planner_version != SOP05R_TEB_PLANNER_VERSION:
        raise ValueError("TEB trajectory planner_version mismatch")
    if record.full_route.planner_version != record.planner_version:
        raise ValueError("full route planner_version mismatch")
    if not record.event_id or not record.source_base_state_id or not record.decision_state_id:
        raise ValueError("TEB trajectory identities must be nonempty")
    arrays = _record_arrays(record)
    if tuple(arrays) != _ARRAY_NAMES:
        raise AssertionError("internal TEB trajectory array schema mismatch")
    stored: dict[str, np.ndarray] = {}
    array_meta: dict[str, object] = {}
    for name, value in arrays.items():
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{name} must be a numeric non-object array")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite values")
        key = f"{row_index:06d}.{name}"
        stored[key] = array
        array_meta[name] = {
            "key": key,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "semantic_digest": _array_digest(name, array),
        }
    row: dict[str, object] = {
        "record_version": SOP05R_TEB_TRAJECTORY_RECORD_VERSION,
        "row_index": row_index,
        "event_id": record.event_id,
        "source_base_state_id": record.source_base_state_id,
        "decision_state_id": record.decision_state_id,
        "template_id": record.template_id,
        "planner_version": record.planner_version,
        "config_digest": record.config_digest,
        "nominal_trajectory_id": record.nominal_trajectory.trajectory_id,
        "goal_arrival_time_s": record.full_route.goal_arrival_time_s,
        "full_route_task_cost": record.full_route.task_cost,
        "suffix_task_cost": record.nominal_trajectory.task_cost,
        "suffix_metadata": record.nominal_trajectory.metadata,
        "arrays": array_meta,
    }
    row["record_semantic_digest"] = _sha256(_canonical_json(row))
    return row, stored


def _collection_digest(rows: Sequence[Mapping[str, object]]) -> str:
    return _sha256(
        b"sop05r_teb_trajectory_collection_v2\0"
        + _canonical_json([row["record_semantic_digest"] for row in rows])
    )


@dataclass(frozen=True)
class Sop05rTebTrajectoryStore:
    records: tuple[Sop05rTebTrajectoryRecord, ...]
    manifest: Mapping[str, object]
    collection_semantic_digest: str
    complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


def publish_sop05r_teb_trajectory_store(
    records: Sequence[Sop05rTebTrajectoryRecord],
    output_dir: str | Path,
    *,
    requested_count: int,
    complete: bool,
) -> Sop05rTebTrajectoryStore:
    """Stage, strictly reload, and atomically publish one v2 trajectory store."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    if isinstance(requested_count, bool) or not isinstance(requested_count, int):
        raise TypeError("requested_count must be an integer")
    if requested_count < 0:
        raise ValueError("requested_count must be nonnegative")
    ordered = tuple(sorted(records, key=lambda record: record.event_id))
    if len({record.event_id for record in ordered}) != len(ordered):
        raise ValueError("duplicate TEB trajectory event_id")
    if complete and len(ordered) != requested_count:
        raise ValueError("complete trajectory store must meet requested_count exactly")
    if len(ordered) > requested_count:
        raise ValueError("trajectory record count exceeds requested_count")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent)
    )
    staging = staging_root / output.name
    staging.mkdir()
    try:
        rows: list[dict[str, object]] = []
        payload_arrays: dict[str, np.ndarray] = {}
        for row_index, record in enumerate(ordered):
            row, arrays = _record_row(record, row_index=row_index)
            rows.append(row)
            payload_arrays.update(arrays)
        collection_digest = _collection_digest(rows)
        manifest = {
            "store_version": SOP05R_TEB_TRAJECTORY_STORE_VERSION,
            "record_count": len(rows),
            "requested_count": requested_count,
            "collection_semantic_digest": collection_digest,
            "records": rows,
        }
        summary = {
            "store_version": SOP05R_TEB_TRAJECTORY_STORE_VERSION,
            "record_count": len(rows),
            "requested_count": requested_count,
            "quota_met": complete,
            "collection_semantic_digest": collection_digest,
        }
        (staging / _MANIFEST).write_bytes(_json_file(manifest))
        _write_deterministic_npz(staging / _PAYLOAD, payload_arrays)
        (staging / _SUMMARY).write_bytes(_json_file(summary))
        if complete:
            (staging / _COMPLETE).write_bytes(
                _json_file(
                    {
                        "completion_version": SOP05R_TEB_TRAJECTORY_COMPLETION_VERSION,
                        "record_count": len(rows),
                        "requested_count": requested_count,
                        "collection_semantic_digest": collection_digest,
                    }
                )
            )
        checksummed = [_MANIFEST, _PAYLOAD, _SUMMARY]
        if complete:
            checksummed.append(_COMPLETE)
        checksums = {
            "store_version": SOP05R_TEB_TRAJECTORY_STORE_VERSION,
            "files": {
                name: _sha256_file(staging / name) for name in sorted(checksummed)
            },
        }
        (staging / _CHECKSUMS).write_bytes(_json_file(checksums))
        loaded = load_sop05r_teb_trajectory_store(
            staging,
            require_complete=complete,
        )
        atomic_rename_noreplace(staging, output)
        return loaded
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read TEB trajectory JSON: {path.name}") from exc


def _load_array(
    payload: Mapping[str, np.ndarray],
    *,
    name: str,
    metadata: Mapping[str, object],
) -> np.ndarray:
    if set(metadata) != {"key", "dtype", "shape", "semantic_digest"}:
        raise ValueError(f"{name} array metadata schema mismatch")
    key = metadata["key"]
    if not isinstance(key, str) or key not in payload:
        raise ValueError(f"{name} array key is missing")
    array = np.ascontiguousarray(payload[key])
    if array.dtype.str != metadata["dtype"] or list(array.shape) != metadata["shape"]:
        raise ValueError(f"{name} array dtype/shape mismatch")
    if _array_digest(name, array) != metadata["semantic_digest"]:
        raise ValueError(f"{name} array semantic digest mismatch")
    array.setflags(write=False)
    return array


def load_sop05r_teb_trajectory_store(
    input_dir: str | Path,
    *,
    require_complete: bool = False,
) -> Sop05rTebTrajectoryStore:
    """Strictly load a v2 store and reject mixed, extra, or tampered artifacts."""

    root = Path(input_dir)
    if not root.is_dir():
        raise ValueError("TEB trajectory store directory does not exist")
    expected = {_MANIFEST, _PAYLOAD, _SUMMARY, _CHECKSUMS}
    complete = (root / _COMPLETE).is_file()
    if complete:
        expected.add(_COMPLETE)
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("TEB trajectory store file set mismatch")
    if require_complete and not complete:
        raise ValueError("TEB trajectory store completion marker is missing")

    checksums = _read_json(root / _CHECKSUMS)
    if (
        not isinstance(checksums, dict)
        or checksums.get("store_version") != SOP05R_TEB_TRAJECTORY_STORE_VERSION
        or not isinstance(checksums.get("files"), dict)
    ):
        raise ValueError("TEB trajectory checksum manifest schema mismatch")
    expected_checksum_names = expected - {_CHECKSUMS}
    if set(checksums["files"]) != expected_checksum_names:
        raise ValueError("TEB trajectory checksum file set mismatch")
    for name, digest in checksums["files"].items():
        if _sha256_file(root / name) != digest:
            raise ValueError(f"TEB trajectory checksum mismatch: {name}")

    manifest = _read_json(root / _MANIFEST)
    summary = _read_json(root / _SUMMARY)
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        raise ValueError("TEB trajectory manifest and summary must be mappings")
    if manifest.get("store_version") != SOP05R_TEB_TRAJECTORY_STORE_VERSION:
        raise ValueError("v1 or unknown trajectory store is not accepted by v2 loader")
    if summary.get("store_version") != SOP05R_TEB_TRAJECTORY_STORE_VERSION:
        raise ValueError("TEB trajectory summary version mismatch")
    rows = manifest.get("records")
    if not isinstance(rows, list):
        raise ValueError("TEB trajectory records must be a list")
    digest = _collection_digest(rows)
    if digest != manifest.get("collection_semantic_digest"):
        raise ValueError("TEB trajectory collection semantic digest mismatch")
    if (
        summary.get("record_count") != len(rows)
        or summary.get("collection_semantic_digest") != digest
        or bool(summary.get("quota_met")) != complete
    ):
        raise ValueError("TEB trajectory summary mismatch")

    try:
        archive = np.load(root / _PAYLOAD, allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError("failed to load TEB trajectory payload") from exc
    with archive:
        payload = {name: archive[name] for name in archive.files}
    records: list[Sop05rTebTrajectoryRecord] = []
    consumed: set[str] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("TEB trajectory row must be a mapping")
        if row.get("record_version") != SOP05R_TEB_TRAJECTORY_RECORD_VERSION:
            raise ValueError("v1 or unknown TEB trajectory record is not accepted")
        if row.get("row_index") != row_index:
            raise ValueError("TEB trajectory row index mismatch")
        semantic_row = dict(row)
        stored_digest = semantic_row.pop("record_semantic_digest", None)
        if _sha256(_canonical_json(semantic_row)) != stored_digest:
            raise ValueError("TEB trajectory record semantic digest mismatch")
        array_meta = row.get("arrays")
        if not isinstance(array_meta, dict) or set(array_meta) != set(_ARRAY_NAMES):
            raise ValueError("TEB trajectory array schema mismatch")
        arrays = {
            name: _load_array(payload, name=name, metadata=array_meta[name])
            for name in _ARRAY_NAMES
        }
        consumed.update(str(array_meta[name]["key"]) for name in _ARRAY_NAMES)
        route = PlannedTebRoute(
            planner_version=str(row["planner_version"]),
            goal_world_pose=arrays["goal_world_pose"],
            band_poses_world=arrays["band_poses_world"],
            band_interval_dt_s=arrays["band_interval_dt_s"],
            sample_times_s=arrays["route_sample_times_s"],
            sampled_poses_world=arrays["route_poses_world"],
            sampled_controls=arrays["route_controls"],
            goal_arrival_time_s=float(row["goal_arrival_time_s"]),
            task_cost=float(row["full_route_task_cost"]),
        )
        suffix_metadata = row.get("suffix_metadata")
        if not isinstance(suffix_metadata, dict):
            raise ValueError("TEB suffix metadata must be a mapping")
        suffix = LocalTrajectory(
            trajectory_id=str(row["nominal_trajectory_id"]),
            poses=arrays["suffix_poses"],
            controls=arrays["suffix_controls"],
            swept_mask=arrays["suffix_swept_mask"],
            tta_map=arrays["suffix_tta_map"],
            braking_map=arrays["suffix_braking_map"],
            centerline_map=arrays["suffix_centerline_map"],
            task_cost=float(row["suffix_task_cost"]),
            metadata=suffix_metadata,
        )
        records.append(
            Sop05rTebTrajectoryRecord(
                event_id=str(row["event_id"]),
                source_base_state_id=str(row["source_base_state_id"]),
                decision_state_id=str(row["decision_state_id"]),
                template_id=str(row["template_id"]),
                planner_version=str(row["planner_version"]),
                config_digest=str(row["config_digest"]),
                shared_goal_world_pose=arrays["goal_world_pose"],
                full_route=route,
                nominal_trajectory=suffix,
            )
        )
    if consumed != set(payload):
        raise ValueError("TEB trajectory payload contains unknown or missing arrays")
    if [record.event_id for record in records] != sorted(
        record.event_id for record in records
    ):
        raise ValueError("TEB trajectory records are not canonically ordered")
    if complete:
        marker = _read_json(root / _COMPLETE)
        if (
            not isinstance(marker, dict)
            or marker.get("completion_version")
            != SOP05R_TEB_TRAJECTORY_COMPLETION_VERSION
            or marker.get("collection_semantic_digest") != digest
            or marker.get("record_count") != len(records)
            or marker.get("requested_count") != manifest.get("requested_count")
        ):
            raise ValueError("TEB trajectory completion marker mismatch")
    return Sop05rTebTrajectoryStore(
        records=tuple(records),
        manifest=manifest,
        collection_semantic_digest=digest,
        complete=complete,
    )
