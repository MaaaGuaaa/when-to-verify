"""Dependency-free THÖR-MAGNI CSV parsing and trajectory resampling.

Raw QTM centroids/markers are millimetres in a world frame and ``Time`` is
seconds. The adapter converts them to metres, retains every valid non-robot
body, and resamples each contiguous segment independently so gaps are never
interpolated over.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.contracts import SCHEMA_VERSION, validate_dynamic_object_spec
from src.utils.config import load_config


class ThorDataError(ValueError):
    """Raised when a THÖR source recording violates the input contract."""


def _reject_json_constant(value: str) -> None:
    raise ThorDataError(f"JSON metadata must not contain {value}")


@dataclass(frozen=True)
class DynamicObjectTrack:
    """One typed, resampled non-robot trajectory in world coordinates."""

    object_id: str
    source_body_name: str
    object_type: str
    raw_role: str
    timestamps: np.ndarray  # float64 [N], seconds
    poses: np.ndarray  # float32 [N, 3], world x/y/yaw
    velocities: np.ndarray  # float32 [N, 2], metres/second
    segment_ids: np.ndarray  # int32 [N], no interpolation across ids
    footprint: dict
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RecordingIndex:
    """Resampled robot and typed dynamic-object trajectories for a recording."""

    recording_id: str
    session_id: str
    timestamps: np.ndarray  # float64 [N], seconds
    robot_pose: np.ndarray  # float32 [N, 3], world x/y/yaw
    robot_twist: np.ndarray  # float32 [N, 2], forward v/yaw rate
    robot_segment_ids: np.ndarray  # int32 [N]
    dynamic_objects: dict[str, DynamicObjectTrack]
    static_map: np.ndarray | None
    source_file: str
    dt_s: float
    coordinate_frame: str = "QTM world XY; metres; yaw radians"
    resampling_report: dict[str, float | int] = field(default_factory=dict)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ThorDataError(message)


def validate_recording_index(recording: RecordingIndex) -> None:
    """Validate shape, dtype, finite values, and segment invariants."""
    count = recording.timestamps.shape[0]
    _require(recording.timestamps.dtype == np.float64, "timestamps dtype must be float64")
    _require(recording.timestamps.shape == (count,), "timestamps must be one-dimensional")
    _require(recording.robot_pose.dtype == np.float32, "robot_pose dtype must be float32")
    _require(recording.robot_pose.shape == (count, 3), "robot_pose shape must be [N,3]")
    _require(recording.robot_twist.dtype == np.float32, "robot_twist dtype must be float32")
    _require(recording.robot_twist.shape == (count, 2), "robot_twist shape must be [N,2]")
    _require(
        recording.robot_segment_ids.dtype == np.int32,
        "robot_segment_ids dtype must be int32",
    )
    _require(
        recording.robot_segment_ids.shape == (count,),
        "robot_segment_ids shape must be [N]",
    )
    _require(count > 0, "recording must contain robot samples")
    _require(np.isfinite(recording.timestamps).all(), "timestamps contain NaN/Inf")
    _require(np.isfinite(recording.robot_pose).all(), "robot_pose contains NaN/Inf")
    _require(np.isfinite(recording.robot_twist).all(), "robot_twist contains NaN/Inf")
    _validate_resampled_grid(
        recording.timestamps,
        recording.robot_segment_ids,
        recording.dt_s,
        "robot",
    )
    _require(
        all(
            isinstance(value, int)
            or (isinstance(value, float) and math.isfinite(value))
            for value in recording.resampling_report.values()
        ),
        "resampling_report contains invalid values",
    )
    for object_id, track in recording.dynamic_objects.items():
        object_count = track.timestamps.shape[0]
        _require(track.object_id == object_id, "dynamic object key/id mismatch")
        _require(
            object_id == f"{recording.recording_id}::{track.source_body_name}",
            "dynamic object id must be recording-scoped",
        )
        _require(bool(track.source_body_name), "source_body_name must not be empty")
        _require(isinstance(track.raw_role, str), "raw_role must be a string")
        _require(
            track.timestamps.dtype == np.float64,
            "dynamic object timestamps must be float64",
        )
        _require(track.poses.dtype == np.float32, "dynamic object poses must be float32")
        _require(
            track.velocities.dtype == np.float32,
            "dynamic object velocities must be float32",
        )
        _require(
            track.segment_ids.dtype == np.int32,
            "dynamic object segment_ids must be int32",
        )
        _require(
            track.poses.shape == (object_count, 3),
            "dynamic object poses shape must be [N,3]",
        )
        _require(
            track.velocities.shape == (object_count, 2),
            "dynamic object velocities shape must be [N,2]",
        )
        _require(
            track.segment_ids.shape == (object_count,),
            "dynamic object segment_ids shape must be [N]",
        )
        _require(object_count > 0, "dynamic object track must not be empty")
        _require(
            np.isfinite(track.timestamps).all(),
            "dynamic object timestamps contain NaN/Inf",
        )
        _require(
            np.isfinite(track.poses).all(),
            "dynamic object poses contain NaN/Inf",
        )
        _require(
            np.isfinite(track.velocities).all(),
            "dynamic object velocities contain NaN/Inf",
        )
        validate_dynamic_object_spec(
            {"object_type": track.object_type, "footprint": track.footprint}
        )
        _require(isinstance(track.provenance, dict), "provenance must be a dict")
        _validate_resampled_grid(
            track.timestamps,
            track.segment_ids,
            recording.dt_s,
            f"dynamic object {object_id}",
        )


def _validate_resampled_grid(
    timestamps: np.ndarray,
    segment_ids: np.ndarray,
    dt_s: float,
    label: str,
) -> None:
    _require(math.isfinite(dt_s) and dt_s > 0.0, "dt_s must be finite and positive")
    _require(np.all(np.diff(timestamps) > 0.0), f"{label} timestamps must increase")
    _require(np.all(np.diff(segment_ids) >= 0), f"{label} segment ids must be ordered")
    for segment_id in np.unique(segment_ids):
        indices = np.flatnonzero(segment_ids == segment_id)
        _require(
            bool(indices.size) and np.all(np.diff(indices) == 1),
            f"{label} segment ids must be contiguous",
        )
        if indices.size > 1:
            _require(
                np.allclose(
                    np.diff(timestamps[indices]), dt_s, rtol=0.0, atol=1e-8
                ),
                f"{label} timestamps must follow the declared dt within segments",
            )


def save_recording_index(recording: RecordingIndex, path: str | Path) -> Path:
    """Atomically save one recording index as numeric NPZ plus JSON metadata."""
    validate_recording_index(recording)
    output_path = Path(path)
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "timestamps": recording.timestamps,
        "robot_pose": recording.robot_pose,
        "robot_twist": recording.robot_twist,
        "robot_segment_ids": recording.robot_segment_ids,
    }
    object_meta: list[dict[str, object]] = []
    for index, object_id in enumerate(sorted(recording.dynamic_objects)):
        track = recording.dynamic_objects[object_id]
        prefix = f"object_{index}"
        payload[f"{prefix}_timestamps"] = track.timestamps
        payload[f"{prefix}_poses"] = track.poses
        payload[f"{prefix}_velocities"] = track.velocities
        payload[f"{prefix}_segment_ids"] = track.segment_ids
        object_meta.append(
            {
                "object_id": object_id,
                "source_body_name": track.source_body_name,
                "object_type": track.object_type,
                "raw_role": track.raw_role,
                "footprint": track.footprint,
                "provenance": track.provenance,
                "prefix": prefix,
            }
        )
    if recording.static_map is not None:
        payload["static_map"] = recording.static_map
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "recording_id": recording.recording_id,
        "session_id": recording.session_id,
        "source_file": recording.source_file,
        "dt_s": recording.dt_s,
        "coordinate_frame": recording.coordinate_frame,
        "resampling_report": recording.resampling_report,
        "dynamic_objects": object_meta,
        "has_static_map": recording.static_map is not None,
    }
    payload["meta_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, allow_nan=False)
    )
    temporary = output_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    temporary.replace(output_path)
    return output_path


def load_recording_index(path: str | Path) -> RecordingIndex:
    """Load and validate a recording index created by :func:`save_recording_index`."""
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(
            str(payload["meta_json"]), parse_constant=_reject_json_constant
        )
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ThorDataError(
                f"schema_version must be {SCHEMA_VERSION}, got {metadata.get('schema_version')!r}"
            )
        dynamic_objects = {
            item["object_id"]: DynamicObjectTrack(
                object_id=item["object_id"],
                source_body_name=item["source_body_name"],
                object_type=item["object_type"],
                raw_role=item["raw_role"],
                timestamps=payload[f"{item['prefix']}_timestamps"].copy(),
                poses=payload[f"{item['prefix']}_poses"].copy(),
                velocities=payload[f"{item['prefix']}_velocities"].copy(),
                segment_ids=payload[f"{item['prefix']}_segment_ids"].copy(),
                footprint=item["footprint"],
                provenance=item["provenance"],
            )
            for item in metadata["dynamic_objects"]
        }
        static_map = payload["static_map"].copy() if metadata["has_static_map"] else None
        recording = RecordingIndex(
            recording_id=metadata["recording_id"],
            session_id=metadata["session_id"],
            timestamps=payload["timestamps"].copy(),
            robot_pose=payload["robot_pose"].copy(),
            robot_twist=payload["robot_twist"].copy(),
            robot_segment_ids=payload["robot_segment_ids"].copy(),
            dynamic_objects=dynamic_objects,
            static_map=static_map,
            source_file=metadata["source_file"],
            dt_s=float(metadata["dt_s"]),
            coordinate_frame=metadata["coordinate_frame"],
            resampling_report=metadata.get("resampling_report", {}),
        )
    validate_recording_index(recording)
    return recording


def _dynamic_object_type_counts(recording: RecordingIndex) -> dict[str, int]:
    counts = {name: 0 for name in ("human", "carried_object", "unknown_dynamic")}
    for track in recording.dynamic_objects.values():
        counts[track.object_type] += 1
    return counts


def _geometry_source_counts(recording: RecordingIndex) -> dict[str, int]:
    counts: dict[str, int] = {}
    for track in recording.dynamic_objects.values():
        source = str(track.provenance.get("geometry_source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def write_recording_indexes(
    recordings: list[RecordingIndex] | tuple[RecordingIndex, ...],
    *,
    split: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Atomically write one split's recording indexes and provenance manifest."""
    if split not in {"train", "calibration", "val", "test"}:
        raise ThorDataError("split must be train, calibration, val, or test")
    ordered = sorted(recordings, key=lambda item: item.recording_id)
    if not ordered:
        raise ThorDataError("recordings must not be empty")
    ids = [recording.recording_id for recording in ordered]
    if len(ids) != len(set(ids)):
        raise ThorDataError("recording ids must be unique within a split")
    for recording in ordered:
        validate_recording_index(recording)

    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_name(f".{output_path.name}.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    rows: list[dict[str, object]] = []
    try:
        for recording in ordered:
            filename = f"{recording.recording_id}.npz"
            save_recording_index(recording, staging / filename)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "split": split,
                    "recording_id": recording.recording_id,
                    "session_id": recording.session_id,
                    "dynamic_object_ids": sorted(recording.dynamic_objects),
                    "dynamic_object_type_counts": _dynamic_object_type_counts(
                        recording
                    ),
                    "geometry_source_counts": _geometry_source_counts(recording),
                    "source_file": recording.source_file,
                    "recording_index_file": filename,
                    "sample_count": int(recording.timestamps.size),
                    "resampling_report": recording.resampling_report,
                }
            )
        manifest = "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "recording_count": len(ordered),
            "robot_sample_count": sum(
                int(recording.timestamps.size) for recording in ordered
            ),
            "dynamic_object_track_count": sum(
                len(recording.dynamic_objects) for recording in ordered
            ),
        }
        (staging / "recording_manifest.jsonl").write_text(
            manifest, encoding="utf-8"
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "directory": output_path,
        "manifest": output_path / "recording_manifest.jsonl",
        "summary": output_path / "summary.json",
    }


def load_recording_indexes_from_dir(
    directory: str | Path, *, expected_split: str
) -> tuple[RecordingIndex, ...]:
    """Load a split directory strictly through its recording manifest."""
    directory_path = Path(directory)
    manifest_path = directory_path / "recording_manifest.jsonl"
    rows: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
            except json.JSONDecodeError as error:
                raise ThorDataError(
                    f"invalid recording manifest line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ThorDataError("recording manifest rows must be objects")
            if row.get("split") != expected_split:
                raise ThorDataError("recording manifest split mismatch")
            if row.get("schema_version") != SCHEMA_VERSION:
                raise ThorDataError("recording manifest schema_version mismatch")
            rows.append(row)
    if not rows:
        raise ThorDataError("recording manifest must not be empty")
    recordings: list[RecordingIndex] = []
    root = directory_path.resolve()
    for row in sorted(rows, key=lambda item: str(item.get("recording_id"))):
        declared = row.get("recording_index_file")
        if not isinstance(declared, str) or not declared:
            raise ThorDataError("recording_index_file must be a non-empty string")
        index_path = (directory_path / declared).resolve()
        if not index_path.is_relative_to(root):
            raise ThorDataError("recording index path escapes its split directory")
        recording = load_recording_index(index_path)
        if recording.recording_id != row.get("recording_id"):
            raise ThorDataError("recording id does not match its manifest row")
        recordings.append(recording)
    return tuple(recordings)


def parse_recording_id(path: str | Path) -> str:
    """Extract the stable recording id from a THÖR-MAGNI CSV filename."""
    match = re.fullmatch(r"THOR-Magni_(.+)\.csv", Path(path).name)
    if match is None:
        raise ThorDataError(f"invalid THÖR-MAGNI filename: {Path(path).name}")
    return match.group(1)


def _is_robot(body_name: str, role: str) -> bool:
    text = f"{body_name} {role}".lower()
    return "darko" in text or "robot" in text


def classify_dynamic_object(body_name: str, role: str) -> str:
    """Map raw THÖR body metadata to the frozen three-type taxonomy."""
    normalized_name = body_name.strip().lower()
    normalized_role = role.strip().lower()
    if normalized_name.startswith("helmet_") or "visitor" in normalized_role:
        return "human"
    carried_name = bool(
        re.match(r"^lo(?:\d+|\b)", normalized_name)
        or any(
            token in normalized_name
            for token in ("cart", "bin", "box", "bucket")
        )
    )
    if carried_name or normalized_role == "carried":
        return "carried_object"
    return "unknown_dynamic"


_MISSING_CSV_VALUES = {"", "N/A", "NA", "NAN"}


def _is_missing_csv_value(value: str) -> bool:
    return value.strip().upper() in _MISSING_CSV_VALUES


def _float(value: str) -> float:
    if _is_missing_csv_value(value):
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _csv_cell(row: list[str], index: int) -> str:
    return row[index].strip() if index < len(row) else ""


def _same_qtm_observation(
    left: list[str], right: list[str], qtm_column_indices: tuple[int, ...]
) -> bool:
    return all(
        _csv_cell(left, index) == _csv_cell(right, index)
        for index in qtm_column_indices
    )


def _regular_grid(start: float, end: float, dt_s: float) -> np.ndarray:
    first = math.ceil((start - 1e-9) / dt_s) * dt_s
    last = math.floor((end + 1e-9) / dt_s) * dt_s
    if last < first:
        return np.empty((0,), dtype=np.float64)
    count = int(round((last - first) / dt_s)) + 1
    return np.round(first + np.arange(count, dtype=np.float64) * dt_s, 9)


def _segment_slices(timestamps: np.ndarray, max_gap_s: float) -> list[slice]:
    if timestamps.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(timestamps) > max_gap_s + 1e-9) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [timestamps.size]))
    return [slice(int(start), int(end)) for start, end in zip(starts, ends)]


def _gradient(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    if timestamps.size < 2:
        return np.zeros_like(values, dtype=np.float64)
    return np.gradient(values, timestamps, axis=0)


def _speed_quantiles(
    timestamps: np.ndarray,
    positions: np.ndarray,
    segment_ids: np.ndarray,
) -> dict[str, float | int]:
    joined = _speed_samples(timestamps, positions, segment_ids)
    return _quantiles(joined)


def _speed_samples(
    timestamps: np.ndarray,
    positions: np.ndarray,
    segment_ids: np.ndarray,
) -> np.ndarray:
    speeds: list[np.ndarray] = []
    for segment_id in np.unique(segment_ids):
        mask = segment_ids == segment_id
        if np.count_nonzero(mask) < 2:
            continue
        velocity = _gradient(positions[mask], timestamps[mask])
        speeds.append(np.linalg.norm(velocity, axis=1))
    if not speeds:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(speeds)


def _quantiles(samples: np.ndarray) -> dict[str, float | int]:
    if samples.size == 0:
        return {"sample_count": 0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    p05, p50, p95 = np.percentile(samples, [5.0, 50.0, 95.0])
    return {
        "sample_count": int(samples.size),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
    }


def _resample_track(
    timestamps: np.ndarray,
    positions: np.ndarray,
    *,
    dt_s: float,
    max_gap_s: float,
    yaw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    grids: list[np.ndarray] = []
    sampled_positions: list[np.ndarray] = []
    sampled_yaws: list[np.ndarray] = []
    sampled_velocities: list[np.ndarray] = []
    segment_ids: list[np.ndarray] = []
    for segment_id, segment in enumerate(_segment_slices(timestamps, max_gap_s)):
        source_time = timestamps[segment]
        if source_time.size < 2:
            continue
        grid = _regular_grid(float(source_time[0]), float(source_time[-1]), dt_s)
        if grid.size == 0:
            continue
        source_position = positions[segment]
        position = np.column_stack(
            [np.interp(grid, source_time, source_position[:, axis]) for axis in range(2)]
        )
        velocity = _gradient(position, grid)
        grids.append(grid)
        sampled_positions.append(position)
        sampled_velocities.append(velocity)
        segment_ids.append(np.full(grid.shape, segment_id, dtype=np.int32))
        if yaw is not None:
            source_yaw = np.unwrap(yaw[segment])
            sampled_yaws.append(np.interp(grid, source_time, source_yaw))

    if not grids:
        empty_time = np.empty((0,), dtype=np.float64)
        empty_xy = np.empty((0, 2), dtype=np.float64)
        empty_ids = np.empty((0,), dtype=np.int32)
        return empty_time, empty_xy, None if yaw is None else empty_time, empty_xy, empty_ids
    joined_yaw = np.concatenate(sampled_yaws) if yaw is not None else None
    return (
        np.concatenate(grids),
        np.concatenate(sampled_positions),
        joined_yaw,
        np.concatenate(sampled_velocities),
        np.concatenate(segment_ids),
    )


def _motion_heading(velocities: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    speeds = np.linalg.norm(velocities, axis=1)
    moving = np.flatnonzero(speeds > 1e-6)
    if moving.size == 0:
        return np.zeros(timestamps.shape, dtype=np.float64)
    moving_heading = np.unwrap(
        np.arctan2(velocities[moving, 1], velocities[moving, 0])
    )
    return np.interp(timestamps, timestamps[moving], moving_heading)


def _resample_object_yaw(
    source_time: np.ndarray,
    source_yaw: np.ndarray,
    sampled_time: np.ndarray,
    sampled_velocity: np.ndarray,
    sampled_segments: np.ndarray,
    *,
    max_gap_s: float,
) -> tuple[np.ndarray, str, dict[str, int]]:
    yaw = np.zeros(sampled_time.shape, dtype=np.float64)
    source_counts: dict[str, int] = {}
    for segment_id, segment in enumerate(_segment_slices(source_time, max_gap_s)):
        target_mask = sampled_segments == segment_id
        if not np.any(target_mask):
            continue
        segment_time = source_time[segment]
        segment_yaw = source_yaw[segment]
        finite = np.isfinite(segment_yaw)
        if np.any(finite):
            finite_yaw = np.unwrap(segment_yaw[finite])
            yaw[target_mask] = np.interp(
                sampled_time[target_mask], segment_time[finite], finite_yaw
            )
            source = "qtm_rotation"
        else:
            target_velocity = sampled_velocity[target_mask]
            target_time = sampled_time[target_mask]
            yaw[target_mask] = _motion_heading(target_velocity, target_time)
            source = (
                "motion_fallback"
                if np.any(np.linalg.norm(target_velocity, axis=1) > 1e-6)
                else "zero_fallback"
            )
        source_counts[source] = source_counts.get(source, 0) + int(
            np.count_nonzero(target_mask)
        )
    orientation_source = (
        next(iter(source_counts)) if len(source_counts) == 1 else "mixed"
    )
    return yaw, orientation_source, dict(sorted(source_counts.items()))


def _marker_columns(
    header_index: dict[str, int], body_name: str
) -> tuple[tuple[str, str, str], ...]:
    pattern = re.compile(rf"^{re.escape(body_name)} - .+ X$")
    triplets: list[tuple[str, str, str]] = []
    for x_name in sorted(name for name in header_index if pattern.match(name)):
        stem = x_name[:-1]
        y_name = stem + "Y"
        z_name = stem + "Z"
        if y_name in header_index and z_name in header_index:
            triplets.append((x_name, y_name, z_name))
    return tuple(triplets)


def _marker_extent_local_m(
    row: list[str],
    header_index: dict[str, int],
    marker_columns: tuple[tuple[str, str, str], ...],
    centroid_m: np.ndarray,
    rotation: np.ndarray | None,
    yaw: float,
) -> tuple[float, float] | None:
    markers: list[tuple[float, float, float]] = []
    for columns in marker_columns:
        marker = tuple(_float(row[header_index[column]]) / 1000.0 for column in columns)
        if all(math.isfinite(value) for value in marker):
            markers.append(marker)
    if len(markers) < 2:
        return None
    delta = np.asarray(markers, dtype=np.float64) - centroid_m
    if rotation is not None:
        local = delta @ rotation.T
        local_xy = local[:, :2]
    elif math.isfinite(yaw):
        cosine, sine = math.cos(yaw), math.sin(yaw)
        local_xy = delta[:, :2] @ np.array(
            [[cosine, -sine], [sine, cosine]], dtype=np.float64
        )
    else:
        return None
    extent = np.ptp(local_xy, axis=0)
    if not np.isfinite(extent).all():
        return None
    return float(extent[0]), float(extent[1])


def _validated_dynamic_object_config(config: dict | None) -> dict:
    resolved = load_config()["dynamic_objects"] if config is None else config
    if not isinstance(resolved, dict):
        raise ThorDataError("dynamic object config must be a mapping")
    required = {"human", "carried_object", "unknown_dynamic", "marker_geometry"}
    if set(resolved) != required:
        raise ThorDataError("dynamic object config keys do not match schema v2")
    marker = resolved["marker_geometry"]
    quantile = float(marker["extent_quantile"])
    minimum_frames = int(marker["minimum_valid_frames"])
    min_extent = float(marker["min_extent_m"])
    max_extent = float(marker["max_extent_m"])
    if not (0.0 < quantile <= 1.0):
        raise ThorDataError("marker extent_quantile must be in (0, 1]")
    if minimum_frames < 1:
        raise ThorDataError("marker minimum_valid_frames must be positive")
    if not (0.0 < min_extent <= max_extent and math.isfinite(max_extent)):
        raise ThorDataError("marker extent bounds are invalid")
    return resolved


def _footprint_for_object(
    object_type: str,
    raw_role: str,
    marker_extents: list[tuple[float, float]],
    config: dict,
) -> tuple[dict, dict]:
    if object_type == "human":
        human = config["human"]
        radius_key = (
            "carrier_radius_m" if "carrier" in raw_role.lower() else "radius_m"
        )
        footprint = {"kind": "circle", "radius_m": float(human[radius_key])}
        return footprint, {
            "geometry_source": "config_human",
            "valid_marker_frame_count": len(marker_extents),
        }

    marker = config["marker_geometry"]
    minimum_frames = int(marker["minimum_valid_frames"])
    if len(marker_extents) >= minimum_frames:
        extents = np.asarray(marker_extents, dtype=np.float64)
        quantile = float(marker["extent_quantile"])
        length_m, width_m = np.quantile(extents, quantile, axis=0)
        lower = float(marker["min_extent_m"])
        upper = float(marker["max_extent_m"])
        footprint = {
            "kind": "rectangle",
            "length_m": float(np.clip(length_m, lower, upper)),
            "width_m": float(np.clip(width_m, lower, upper)),
        }
        return footprint, {
            "geometry_source": "qtm_marker_p95",
            "valid_marker_frame_count": len(marker_extents),
        }

    if object_type == "carried_object":
        carried = config["carried_object"]
        footprint = {
            "kind": "rectangle",
            "length_m": float(carried["fallback_length_m"]),
            "width_m": float(carried["fallback_width_m"]),
        }
    else:
        unknown = config["unknown_dynamic"]
        footprint = {
            "kind": "circle",
            "radius_m": float(unknown["fallback_radius_m"]),
        }
    return footprint, {
        "geometry_source": "config_fallback",
        "valid_marker_frame_count": len(marker_extents),
    }


def _validate_raw_timestamps(timestamps: np.ndarray) -> None:
    if timestamps.size < 2:
        raise ThorDataError("recording must contain at least two timestamped rows")
    if not np.isfinite(timestamps).all():
        raise ThorDataError("timestamps contain NaN/Inf")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ThorDataError("timestamps must be strictly increasing without duplicates")


def load_thor_recording(
    csv_path: str | Path,
    *,
    dt_s: float = 0.2,
    max_gap_s: float = 0.3,
    dynamic_object_config: dict | None = None,
) -> RecordingIndex:
    """Parse every valid non-robot QTM body from one THÖR scenario CSV."""
    path = Path(csv_path)
    recording_id = parse_recording_id(path)
    object_config = _validated_dynamic_object_config(dynamic_object_config)
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ThorDataError("dt_s must be finite and positive")
    if not math.isfinite(max_gap_s) or max_gap_s < dt_s:
        raise ThorDataError("max_gap_s must be finite and at least dt_s")

    metadata: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row and row[0] == "Frame":
                header = row
                break
            if row:
                values = row[1:]
                while values and not values[-1]:
                    values.pop()
                metadata[row[0]] = values
        else:
            raise ThorDataError(f"Frame header not found in {path.name}")

        body_names = metadata.get("BODY_NAMES", [])
        body_roles = metadata.get("BODY_ROLES", [])
        if not body_names:
            raise ThorDataError("BODY_NAMES metadata is missing")
        header_index = {name: index for index, name in enumerate(header)}
        if "Time" not in header_index:
            raise ThorDataError("Time column is missing")

        selected: dict[str, tuple[str, bool]] = {}
        for index, body_name in enumerate(body_names):
            role = body_roles[index] if index < len(body_roles) else ""
            selected[body_name] = (role, _is_robot(body_name, role))
        robots = [name for name, (_, robot) in selected.items() if robot]
        if len(robots) != 1:
            raise ThorDataError(f"expected exactly one robot body, found {len(robots)}")

        raw_time: list[float] = []
        tracks: dict[str, dict[str, list]] = {
            name: {
                "time": [],
                "x": [],
                "y": [],
                "yaw": [],
                "marker_extents": [],
            }
            for name in selected
        }
        marker_columns = {
            name: _marker_columns(header_index, name) for name in selected
        }
        qtm_column_names = {"Frame", "Time"}
        for body_name in selected:
            qtm_column_names.update(
                {
                    f"{body_name} Centroid_X",
                    f"{body_name} Centroid_Y",
                    f"{body_name} Centroid_Z",
                    *(f"{body_name} R{index}" for index in range(9)),
                }
            )
            for triplet in marker_columns[body_name]:
                qtm_column_names.update(triplet)
        qtm_column_indices = tuple(
            sorted(
                header_index[name]
                for name in qtm_column_names
                if name in header_index
            )
        )
        frame_index = header_index["Frame"]
        time_index = header_index["Time"]
        qtm_data_indices = tuple(
            index
            for index in qtm_column_indices
            if index not in {frame_index, time_index}
        )
        raw_csv_row_count = 0
        collapsed_duplicate_qtm_row_count = 0
        ignored_auxiliary_row_count = 0
        previous_frame = float("nan")
        previous_timestamp = float("nan")
        previous_qtm_row: list[str] | None = None
        for row in reader:
            if not row:
                continue
            raw_csv_row_count += 1
            time_text = _csv_cell(row, time_index)
            frame_text = _csv_cell(row, frame_index)
            timestamp = _float(time_text)
            frame = _float(frame_text)
            finite_frame = math.isfinite(frame)
            finite_timestamp = math.isfinite(timestamp)
            missing_frame = _is_missing_csv_value(frame_text)
            missing_timestamp = _is_missing_csv_value(time_text)
            if not finite_frame or not finite_timestamp:
                populated_qtm_columns = [
                    header[index]
                    for index in qtm_data_indices
                    if not _is_missing_csv_value(_csv_cell(row, index))
                ]
                if (
                    missing_frame
                    and missing_timestamp
                    and not populated_qtm_columns
                ):
                    ignored_auxiliary_row_count += 1
                    continue
                if missing_frame and missing_timestamp:
                    raise ThorDataError(
                        f"CSV row {reader.line_num} has QTM data but missing "
                        "Frame/Time; populated QTM columns: "
                        + ", ".join(populated_qtm_columns)
                    )
                raise ThorDataError(
                    f"CSV row {reader.line_num} has invalid or incomplete "
                    f"QTM Frame/Time: Frame={frame_text!r}, Time={time_text!r}"
                )
            if (
                previous_qtm_row is not None
                and frame == previous_frame
                and timestamp == previous_timestamp
            ):
                if _same_qtm_observation(
                    previous_qtm_row, row, qtm_column_indices
                ):
                    collapsed_duplicate_qtm_row_count += 1
                    continue
                differing_columns = [
                    header[index]
                    for index in qtm_column_indices
                    if _csv_cell(previous_qtm_row, index)
                    != _csv_cell(row, index)
                ]
                raise ThorDataError(
                    "conflicting duplicate QTM frame "
                    f"{frame:g} at {timestamp:g} s; differing columns: "
                    + ", ".join(differing_columns)
                )
            if previous_qtm_row is not None and (
                frame <= previous_frame or timestamp <= previous_timestamp
            ):
                raise ThorDataError(
                    "QTM Frame/Time must strictly increase after "
                    f"canonicalization; previous=({previous_frame:g}, "
                    f"{previous_timestamp:g}), current=({frame:g}, "
                    f"{timestamp:g})"
                )
            raw_time.append(timestamp)
            previous_frame = frame
            previous_timestamp = timestamp
            previous_qtm_row = row
            for body_name in selected:
                required = (
                    f"{body_name} Centroid_X",
                    f"{body_name} Centroid_Y",
                )
                if any(column not in header_index for column in required):
                    continue
                x = _float(row[header_index[required[0]]]) / 1000.0
                y = _float(row[header_index[required[1]]]) / 1000.0
                z_name = f"{body_name} Centroid_Z"
                z = (
                    _float(row[header_index[z_name]]) / 1000.0
                    if z_name in header_index
                    else 0.0
                )
                if not (math.isfinite(timestamp) and math.isfinite(x) and math.isfinite(y)):
                    continue
                rotation_values = []
                for rotation_index in range(9):
                    column = f"{body_name} R{rotation_index}"
                    rotation_values.append(
                        _float(row[header_index[column]])
                        if column in header_index
                        else float("nan")
                    )
                rotation = (
                    np.asarray(rotation_values, dtype=np.float64).reshape(3, 3)
                    if np.isfinite(rotation_values).all()
                    else None
                )
                if rotation is not None:
                    yaw = math.atan2(float(rotation[0, 1]), float(rotation[0, 0]))
                else:
                    r0, r1 = rotation_values[:2]
                    yaw = (
                        math.atan2(r1, r0)
                        if math.isfinite(r0) and math.isfinite(r1)
                        else float("nan")
                    )
                tracks[body_name]["time"].append(timestamp)
                tracks[body_name]["x"].append(x)
                tracks[body_name]["y"].append(y)
                tracks[body_name]["yaw"].append(yaw)
                extent = _marker_extent_local_m(
                    row,
                    header_index,
                    marker_columns[body_name],
                    np.asarray((x, y, z), dtype=np.float64),
                    rotation,
                    yaw,
                )
                if extent is not None:
                    tracks[body_name]["marker_extents"].append(extent)

    _validate_raw_timestamps(np.asarray(raw_time, dtype=np.float64))
    robot_name = robots[0]
    robot_source = tracks[robot_name]
    robot_time = np.asarray(robot_source["time"], dtype=np.float64)
    _validate_raw_timestamps(robot_time)
    robot_position = np.column_stack((robot_source["x"], robot_source["y"]))
    robot_yaw = np.asarray(robot_source["yaw"], dtype=np.float64)
    if not np.isfinite(robot_yaw).all():
        robot_velocity = _gradient(robot_position, robot_time)
        fallback_yaw = _motion_heading(robot_velocity, robot_time)
        robot_yaw = np.where(np.isfinite(robot_yaw), robot_yaw, fallback_yaw)
    timestamps, position, yaw, velocity, robot_segments = _resample_track(
        robot_time,
        robot_position,
        dt_s=dt_s,
        max_gap_s=max_gap_s,
        yaw=robot_yaw,
    )
    if timestamps.size == 0 or yaw is None:
        raise ThorDataError("robot track has no resampleable segment")
    forward_speed = velocity[:, 0] * np.cos(yaw) + velocity[:, 1] * np.sin(yaw)
    yaw_rate = np.zeros_like(yaw)
    for segment_id in np.unique(robot_segments):
        mask = robot_segments == segment_id
        yaw_rate[mask] = _gradient(yaw[mask], timestamps[mask])

    raw_robot_segments = np.zeros(robot_time.shape, dtype=np.int32)
    for segment_id, segment in enumerate(_segment_slices(robot_time, max_gap_s)):
        raw_robot_segments[segment] = segment_id
    raw_speed = _speed_quantiles(
        robot_time, robot_position, raw_robot_segments
    )
    resampled_speed = _speed_quantiles(
        timestamps, position, robot_segments
    )
    resampling_report: dict[str, float | int] = {
        "raw_csv_row_count": raw_csv_row_count,
        "canonical_qtm_frame_count": len(raw_time),
        "collapsed_duplicate_qtm_row_count": (
            collapsed_duplicate_qtm_row_count
        ),
        "ignored_auxiliary_row_count": ignored_auxiliary_row_count,
        "conflicting_duplicate_qtm_row_count": 0,
        "raw_robot_sample_count": raw_speed["sample_count"],
        "resampled_robot_sample_count": resampled_speed["sample_count"],
    }
    for quantile in ("p05", "p50", "p95"):
        resampling_report[f"raw_robot_speed_{quantile}_mps"] = raw_speed[
            quantile
        ]
        resampling_report[f"resampled_robot_speed_{quantile}_mps"] = (
            resampled_speed[quantile]
        )
        resampling_report[f"robot_speed_{quantile}_delta_mps"] = abs(
            float(resampled_speed[quantile]) - float(raw_speed[quantile])
        )

    dynamic_objects: dict[str, DynamicObjectTrack] = {}
    raw_object_speeds: list[np.ndarray] = []
    resampled_object_speeds: list[np.ndarray] = []
    rejection_reasons = {
        "insufficient_samples": 0,
        "no_resampleable_segment": 0,
    }
    for body_name, (role, robot) in selected.items():
        if robot:
            continue
        source = tracks[body_name]
        source_time = np.asarray(source["time"], dtype=np.float64)
        if source_time.size < 2:
            rejection_reasons["insufficient_samples"] += 1
            continue
        _validate_raw_timestamps(source_time)
        source_position = np.column_stack((source["x"], source["y"]))
        raw_object_segments = np.zeros(source_time.shape, dtype=np.int32)
        for segment_id, segment in enumerate(
            _segment_slices(source_time, max_gap_s)
        ):
            raw_object_segments[segment] = segment_id
        object_time, object_position, _, object_velocity, object_segments = (
            _resample_track(
                source_time,
                source_position,
                dt_s=dt_s,
                max_gap_s=max_gap_s,
            )
        )
        if object_time.size == 0:
            rejection_reasons["no_resampleable_segment"] += 1
            continue
        object_yaw, orientation_source, orientation_counts = _resample_object_yaw(
            source_time,
            np.asarray(source["yaw"], dtype=np.float64),
            object_time,
            object_velocity,
            object_segments,
            max_gap_s=max_gap_s,
        )
        object_type = classify_dynamic_object(body_name, role)
        footprint, geometry_provenance = _footprint_for_object(
            object_type,
            role,
            source["marker_extents"],
            object_config,
        )
        object_id = f"{recording_id}::{body_name}"
        dynamic_objects[object_id] = DynamicObjectTrack(
            object_id=object_id,
            source_body_name=body_name,
            object_type=object_type,
            raw_role=role,
            timestamps=object_time,
            poses=np.column_stack((object_position, object_yaw)).astype(np.float32),
            velocities=object_velocity.astype(np.float32),
            segment_ids=object_segments,
            footprint=footprint,
            provenance={
                **geometry_provenance,
                "orientation_source": orientation_source,
                "orientation_source_counts": orientation_counts,
            },
        )
        raw_object_speeds.append(
            _speed_samples(source_time, source_position, raw_object_segments)
        )
        resampled_object_speeds.append(
            _speed_samples(object_time, object_position, object_segments)
        )

    raw_object_speed = _quantiles(
        np.concatenate(raw_object_speeds)
        if raw_object_speeds
        else np.empty((0,), dtype=np.float64)
    )
    resampled_object_speed = _quantiles(
        np.concatenate(resampled_object_speeds)
        if resampled_object_speeds
        else np.empty((0,), dtype=np.float64)
    )
    resampling_report.update(
        {
            "raw_dynamic_object_count": len(selected) - 1,
            "indexed_dynamic_object_count": len(dynamic_objects),
            "rejected_dynamic_object_count": sum(rejection_reasons.values()),
            "raw_dynamic_object_sample_count": raw_object_speed["sample_count"],
            "resampled_dynamic_object_sample_count": resampled_object_speed[
                "sample_count"
            ],
            **{
                f"rejected_dynamic_object_{reason}": count
                for reason, count in rejection_reasons.items()
            },
        }
    )
    for quantile in ("p05", "p50", "p95"):
        resampling_report[f"raw_dynamic_object_speed_{quantile}_mps"] = (
            raw_object_speed[quantile]
        )
        resampling_report[
            f"resampled_dynamic_object_speed_{quantile}_mps"
        ] = resampled_object_speed[quantile]
        resampling_report[f"dynamic_object_speed_{quantile}_delta_mps"] = abs(
            float(resampled_object_speed[quantile])
            - float(raw_object_speed[quantile])
        )

    session_id = recording_id.split("_", 1)[0]
    recording = RecordingIndex(
        recording_id=recording_id,
        session_id=session_id,
        timestamps=timestamps,
        robot_pose=np.column_stack((position, yaw)).astype(np.float32),
        robot_twist=np.column_stack((forward_speed, yaw_rate)).astype(np.float32),
        robot_segment_ids=robot_segments,
        dynamic_objects=dynamic_objects,
        static_map=None,
        source_file=path.name,
        dt_s=float(dt_s),
        resampling_report=resampling_report,
    )
    validate_recording_index(recording)
    return recording
