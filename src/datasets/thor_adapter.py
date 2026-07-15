"""Dependency-free THÖR-MAGNI CSV parsing and trajectory resampling.

Raw QTM centroids are millimetres in a world XY frame and ``Time`` is seconds.
The adapter converts positions to metres, keeps only the robot and human
participants, and resamples each contiguous segment independently so gaps are
never interpolated over.
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

from src.contracts import SCHEMA_VERSION


class ThorDataError(ValueError):
    """Raised when a THÖR source recording violates the input contract."""


def _reject_json_constant(value: str) -> None:
    raise ThorDataError(f"JSON metadata must not contain {value}")


@dataclass(frozen=True)
class PedestrianTrack:
    """One resampled human trajectory in world metres and seconds."""

    participant_id: str
    timestamps: np.ndarray  # float64 [N], seconds
    positions: np.ndarray  # float32 [N, 2], world metres
    velocities: np.ndarray  # float32 [N, 2], metres/second
    segment_ids: np.ndarray  # int32 [N], no interpolation across ids
    role: str


@dataclass(frozen=True)
class RecordingIndex:
    """Resampled robot and pedestrian trajectories for one recording."""

    recording_id: str
    session_id: str
    timestamps: np.ndarray  # float64 [N], seconds
    robot_pose: np.ndarray  # float32 [N, 3], world x/y/yaw
    robot_twist: np.ndarray  # float32 [N, 2], forward v/yaw rate
    robot_segment_ids: np.ndarray  # int32 [N]
    pedestrians: dict[str, PedestrianTrack]
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
    for participant_id, track in recording.pedestrians.items():
        ped_count = track.timestamps.shape[0]
        _require(track.participant_id == participant_id, "pedestrian key/id mismatch")
        _require(track.timestamps.dtype == np.float64, "pedestrian timestamps must be float64")
        _require(track.positions.dtype == np.float32, "pedestrian positions must be float32")
        _require(track.velocities.dtype == np.float32, "pedestrian velocities must be float32")
        _require(track.segment_ids.dtype == np.int32, "pedestrian segment_ids must be int32")
        _require(
            track.positions.shape == (ped_count, 2),
            "pedestrian positions shape must be [N,2]",
        )
        _require(
            track.velocities.shape == (ped_count, 2),
            "pedestrian velocities shape must be [N,2]",
        )
        _require(
            track.segment_ids.shape == (ped_count,),
            "pedestrian segment_ids shape must be [N]",
        )
        _require(np.isfinite(track.timestamps).all(), "pedestrian timestamps contain NaN/Inf")
        _require(np.isfinite(track.positions).all(), "pedestrian positions contain NaN/Inf")
        _require(np.isfinite(track.velocities).all(), "pedestrian velocities contain NaN/Inf")
        _validate_resampled_grid(
            track.timestamps,
            track.segment_ids,
            recording.dt_s,
            f"pedestrian {participant_id}",
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
    pedestrian_meta: list[dict[str, str]] = []
    for index, participant_id in enumerate(sorted(recording.pedestrians)):
        track = recording.pedestrians[participant_id]
        prefix = f"ped_{index}"
        payload[f"{prefix}_timestamps"] = track.timestamps
        payload[f"{prefix}_positions"] = track.positions
        payload[f"{prefix}_velocities"] = track.velocities
        payload[f"{prefix}_segment_ids"] = track.segment_ids
        pedestrian_meta.append(
            {
                "participant_id": participant_id,
                "role": track.role,
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
        "pedestrians": pedestrian_meta,
        "has_static_map": recording.static_map is not None,
    }
    payload["meta_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
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
        pedestrians = {
            item["participant_id"]: PedestrianTrack(
                participant_id=item["participant_id"],
                timestamps=payload[f"{item['prefix']}_timestamps"].copy(),
                positions=payload[f"{item['prefix']}_positions"].copy(),
                velocities=payload[f"{item['prefix']}_velocities"].copy(),
                segment_ids=payload[f"{item['prefix']}_segment_ids"].copy(),
                role=item["role"],
            )
            for item in metadata["pedestrians"]
        }
        static_map = payload["static_map"].copy() if metadata["has_static_map"] else None
        recording = RecordingIndex(
            recording_id=metadata["recording_id"],
            session_id=metadata["session_id"],
            timestamps=payload["timestamps"].copy(),
            robot_pose=payload["robot_pose"].copy(),
            robot_twist=payload["robot_twist"].copy(),
            robot_segment_ids=payload["robot_segment_ids"].copy(),
            pedestrians=pedestrians,
            static_map=static_map,
            source_file=metadata["source_file"],
            dt_s=float(metadata["dt_s"]),
            coordinate_frame=metadata["coordinate_frame"],
            resampling_report=metadata.get("resampling_report", {}),
        )
    validate_recording_index(recording)
    return recording


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
                    "participant_ids": sorted(recording.pedestrians),
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
            "pedestrian_track_count": sum(
                len(recording.pedestrians) for recording in ordered
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


def _is_pedestrian(body_name: str, role: str) -> bool:
    text = f"{body_name} {role}".lower()
    excluded = ("carrier", "storage", "cart", "bin")
    if any(token in text for token in excluded):
        return False
    return any(token in text for token in ("helmet", "visitor", "pedestrian", "human"))


def _float(value: str) -> float:
    if value.strip().upper() in {"", "N/A", "NA", "NAN"}:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


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
) -> RecordingIndex:
    """Parse and resample one official THÖR-MAGNI scenario CSV."""
    path = Path(csv_path)
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
            if _is_robot(body_name, role):
                selected[body_name] = (role, True)
            elif _is_pedestrian(body_name, role):
                selected[body_name] = (role, False)
        robots = [name for name, (_, robot) in selected.items() if robot]
        if len(robots) != 1:
            raise ThorDataError(f"expected exactly one robot body, found {len(robots)}")

        raw_time: list[float] = []
        tracks: dict[str, dict[str, list[float]]] = {
            name: {"time": [], "x": [], "y": [], "yaw": []}
            for name in selected
        }
        for row in reader:
            if not row:
                continue
            timestamp = _float(row[header_index["Time"]])
            raw_time.append(timestamp)
            for body_name in selected:
                required = (
                    f"{body_name} Centroid_X",
                    f"{body_name} Centroid_Y",
                    f"{body_name} R0",
                    f"{body_name} R1",
                )
                if any(column not in header_index for column in required):
                    continue
                x = _float(row[header_index[required[0]]]) / 1000.0
                y = _float(row[header_index[required[1]]]) / 1000.0
                r0 = _float(row[header_index[required[2]]])
                r1 = _float(row[header_index[required[3]]])
                if not (math.isfinite(timestamp) and math.isfinite(x) and math.isfinite(y)):
                    continue
                tracks[body_name]["time"].append(timestamp)
                tracks[body_name]["x"].append(x)
                tracks[body_name]["y"].append(y)
                tracks[body_name]["yaw"].append(
                    math.atan2(r1, r0) if math.isfinite(r0) and math.isfinite(r1) else 0.0
                )

    _validate_raw_timestamps(np.asarray(raw_time, dtype=np.float64))
    robot_name = robots[0]
    robot_source = tracks[robot_name]
    robot_time = np.asarray(robot_source["time"], dtype=np.float64)
    _validate_raw_timestamps(robot_time)
    robot_position = np.column_stack((robot_source["x"], robot_source["y"]))
    robot_yaw = np.asarray(robot_source["yaw"], dtype=np.float64)
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

    pedestrians: dict[str, PedestrianTrack] = {}
    raw_pedestrian_speeds: list[np.ndarray] = []
    resampled_pedestrian_speeds: list[np.ndarray] = []
    for body_name, (role, robot) in selected.items():
        if robot:
            continue
        source = tracks[body_name]
        source_time = np.asarray(source["time"], dtype=np.float64)
        if source_time.size < 2:
            continue
        _validate_raw_timestamps(source_time)
        source_position = np.column_stack((source["x"], source["y"]))
        raw_pedestrian_segments = np.zeros(source_time.shape, dtype=np.int32)
        for segment_id, segment in enumerate(
            _segment_slices(source_time, max_gap_s)
        ):
            raw_pedestrian_segments[segment] = segment_id
        ped_time, ped_position, _, ped_velocity, ped_segments = _resample_track(
            source_time,
            source_position,
            dt_s=dt_s,
            max_gap_s=max_gap_s,
        )
        if ped_time.size == 0:
            continue
        raw_pedestrian_speeds.append(
            _speed_samples(
                source_time, source_position, raw_pedestrian_segments
            )
        )
        resampled_pedestrian_speeds.append(
            _speed_samples(ped_time, ped_position, ped_segments)
        )
        pedestrians[body_name] = PedestrianTrack(
            participant_id=body_name,
            timestamps=ped_time,
            positions=ped_position.astype(np.float32),
            velocities=ped_velocity.astype(np.float32),
            segment_ids=ped_segments,
            role=role,
        )

    raw_pedestrian_speed = _quantiles(
        np.concatenate(raw_pedestrian_speeds)
        if raw_pedestrian_speeds
        else np.empty((0,), dtype=np.float64)
    )
    resampled_pedestrian_speed = _quantiles(
        np.concatenate(resampled_pedestrian_speeds)
        if resampled_pedestrian_speeds
        else np.empty((0,), dtype=np.float64)
    )
    resampling_report.update(
        {
            "raw_pedestrian_sample_count": raw_pedestrian_speed[
                "sample_count"
            ],
            "resampled_pedestrian_sample_count": resampled_pedestrian_speed[
                "sample_count"
            ],
        }
    )
    for quantile in ("p05", "p50", "p95"):
        resampling_report[f"raw_pedestrian_speed_{quantile}_mps"] = (
            raw_pedestrian_speed[quantile]
        )
        resampling_report[
            f"resampled_pedestrian_speed_{quantile}_mps"
        ] = resampled_pedestrian_speed[quantile]
        resampling_report[f"pedestrian_speed_{quantile}_delta_mps"] = abs(
            float(resampled_pedestrian_speed[quantile])
            - float(raw_pedestrian_speed[quantile])
        )

    recording_id = parse_recording_id(path)
    session_id = recording_id.split("_", 1)[0]
    recording = RecordingIndex(
        recording_id=recording_id,
        session_id=session_id,
        timestamps=timestamps,
        robot_pose=np.column_stack((position, yaw)).astype(np.float32),
        robot_twist=np.column_stack((forward_speed, yaw_rate)).astype(np.float32),
        robot_segment_ids=robot_segments,
        pedestrians=pedestrians,
        static_map=None,
        source_file=path.name,
        dt_s=float(dt_s),
        resampling_report=resampling_report,
    )
    validate_recording_index(recording)
    return recording
