"""Build split- and type-isolated dynamic-object motion snippet libraries."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.contracts import (
    DYNAMIC_OBJECT_TYPES,
    SCHEMA_VERSION,
    validate_dynamic_object_spec,
)
from src.datasets.split_manager import audit_split_leakage
from src.geometry import (
    CircleFootprint,
    RectangleFootprint,
    inflate_footprint,
    trajectory_signed_clearances,
)
from src.utils.seeding import stable_digest

from .thor_adapter import RecordingIndex, ThorDataError, validate_recording_index


@dataclass(frozen=True)
class MotionSnippet:
    """One fixed-rate, typed local dynamic-object motion snippet."""

    snippet_id: str
    split: str
    source_recording_id: str
    source_object_id: str
    object_type: str
    footprint: dict
    start_timestamp: float
    positions: np.ndarray  # float32 [Tp, 2], first point at origin
    velocities: np.ndarray  # float32 [Tp, 2], same normalized frame
    headings: np.ndarray  # float32 [Tp], same normalized frame
    duration_s: float
    mean_speed_mps: float
    max_acceleration_mps2: float
    mean_abs_curvature_per_m: float
    provenance: dict


@dataclass(frozen=True)
class SnippetLibrary:
    """Accepted snippets and deterministic filter statistics."""

    object_type: str
    snippets: tuple[MotionSnippet, ...]
    summary: dict[str, object]


def _reject_json_constant(value: str) -> None:
    raise ThorDataError(f"JSON metadata must not contain {value}")


def _validate_snippet(snippet: MotionSnippet) -> None:
    if snippet.object_type not in DYNAMIC_OBJECT_TYPES:
        raise ThorDataError("snippet object_type is invalid")
    validate_dynamic_object_spec(
        {"object_type": snippet.object_type, "footprint": snippet.footprint}
    )
    if snippet.positions.ndim != 2 or snippet.positions.shape[1] != 2:
        raise ThorDataError("snippet positions shape must be [Tp,2]")
    if snippet.velocities.shape != snippet.positions.shape:
        raise ThorDataError("snippet velocities must match positions shape")
    if snippet.positions.dtype != np.float32:
        raise ThorDataError("snippet positions dtype must be float32")
    if snippet.velocities.dtype != np.float32:
        raise ThorDataError("snippet velocities dtype must be float32")
    if snippet.headings.shape != (snippet.positions.shape[0],):
        raise ThorDataError("snippet headings shape must be [Tp]")
    if snippet.headings.dtype != np.float32:
        raise ThorDataError("snippet headings dtype must be float32")
    if not np.isfinite(snippet.positions).all():
        raise ThorDataError("snippet positions contain NaN/Inf")
    if not np.isfinite(snippet.velocities).all():
        raise ThorDataError("snippet velocities contain NaN/Inf")
    if not np.isfinite(snippet.headings).all():
        raise ThorDataError("snippet headings contain NaN/Inf")
    if not isinstance(snippet.provenance, dict):
        raise ThorDataError("snippet provenance must be a dict")
    scalars = (
        snippet.start_timestamp,
        snippet.duration_s,
        snippet.mean_speed_mps,
        snippet.max_acceleration_mps2,
        snippet.mean_abs_curvature_per_m,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ThorDataError("snippet statistics contain NaN/Inf")


def _robot_indices(recording: RecordingIndex, timestamps: np.ndarray) -> np.ndarray | None:
    indices = np.searchsorted(recording.timestamps, timestamps)
    if np.any(indices >= recording.timestamps.size):
        return None
    if not np.allclose(
        recording.timestamps[indices], timestamps, rtol=0.0, atol=1e-8
    ):
        return None
    if not np.all(
        recording.robot_segment_ids[indices]
        == recording.robot_segment_ids[indices[0]]
    ):
        return None
    return indices.astype(np.int64, copy=False)


def _overlaps_robot(
    recording: RecordingIndex,
    timestamps: np.ndarray,
    object_poses: np.ndarray,
    footprint: dict,
) -> bool:
    indices = _robot_indices(recording, timestamps)
    if indices is None:
        return True
    robot = inflate_footprint(
        RectangleFootprint(length_m=0.70, width_m=0.55), 0.15
    )
    if footprint["kind"] == "circle":
        dynamic_object = CircleFootprint(radius_m=float(footprint["radius_m"]))
    else:
        dynamic_object = RectangleFootprint(
            length_m=float(footprint["length_m"]),
            width_m=float(footprint["width_m"]),
        )
    clearances = trajectory_signed_clearances(
        robot,
        recording.robot_pose[indices],
        dynamic_object,
        object_poses,
    )
    return bool(np.any(clearances <= 0.0))


def _motion_statistics(
    positions: np.ndarray, timestamps: np.ndarray
) -> tuple[np.ndarray, float, float, float]:
    velocities = np.gradient(positions, timestamps, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    acceleration = np.gradient(velocities, timestamps, axis=0)
    max_acceleration = float(np.max(np.linalg.norm(acceleration, axis=1)))
    if max_acceleration < 1e-5:
        max_acceleration = 0.0
    headings = np.unwrap(np.arctan2(velocities[:, 1], velocities[:, 0]))
    heading_rate = np.gradient(headings, timestamps)
    moving = speeds > 1e-6
    curvature = np.zeros_like(speeds)
    curvature[moving] = np.abs(heading_rate[moving]) / speeds[moving]
    mean_curvature = float(np.mean(curvature[moving])) if np.any(moving) else 0.0
    if mean_curvature < 1e-7:
        mean_curvature = 0.0
    return velocities, float(np.mean(speeds)), max_acceleration, mean_curvature


def _normalize_motion(
    positions: np.ndarray,
    velocities: np.ndarray,
    headings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    displacement = positions[1:] - positions[0]
    norms = np.linalg.norm(displacement, axis=1)
    moving = np.flatnonzero(norms > 1e-6)
    if moving.size == 0:
        return None
    direction = displacement[moving[0]]
    heading = math.atan2(float(direction[1]), float(direction[0]))
    cosine = math.cos(-heading)
    sine = math.sin(-heading)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    normalized_positions = (positions - positions[0]) @ rotation.T
    normalized_velocities = velocities @ rotation.T
    normalized_headings = (headings - heading + math.pi) % (2.0 * math.pi) - math.pi
    normalized_positions[np.abs(normalized_positions) < 1e-7] = 0.0
    normalized_velocities[np.abs(normalized_velocities) < 1e-7] = 0.0
    normalized_headings[np.abs(normalized_headings) < 1e-7] = 0.0
    return normalized_positions, normalized_velocities, normalized_headings


def build_snippet_library(
    recordings: list[RecordingIndex] | tuple[RecordingIndex, ...],
    *,
    split: str,
    object_type: str,
    duration_s: float = 3.0,
    stride_s: float = 1.0,
    min_mean_speed_mps: float = 0.3,
    max_mean_speed_mps: float = 2.0,
    max_acceleration_mps2: float = 2.5,
) -> SnippetLibrary:
    """Extract one deterministic split/type library from matching tracks."""
    if split not in {"train", "calibration", "val", "test"}:
        raise ThorDataError("split must be train, calibration, val, or test")
    if object_type not in DYNAMIC_OBJECT_TYPES:
        raise ThorDataError("object_type is not part of the frozen taxonomy")
    if not (3.0 <= duration_s <= 5.0):
        raise ThorDataError("duration_s must be in [3.0, 5.0]")
    if not recordings:
        raise ThorDataError("recordings must not be empty")
    if not all(
        math.isfinite(value)
        for value in (
            duration_s,
            stride_s,
            min_mean_speed_mps,
            max_mean_speed_mps,
            max_acceleration_mps2,
        )
    ):
        raise ThorDataError("snippet parameters must be finite")
    if not 0.0 <= min_mean_speed_mps < max_mean_speed_mps:
        raise ThorDataError("speed bounds are invalid")
    if stride_s <= 0.0:
        raise ThorDataError("stride_s must be positive")
    if max_acceleration_mps2 <= 0.0:
        raise ThorDataError("max_acceleration_mps2 must be positive")

    snippets: list[MotionSnippet] = []
    rejection_reasons = {
        "time_grid": 0,
        "stationary": 0,
        "speed": 0,
        "acceleration": 0,
        "robot_overlap": 0,
    }
    candidate_count = 0
    source_object_ids: set[str] = set()
    geometry_source_counts: dict[str, int] = {}
    orientation_source_counts: dict[str, int] = {}
    for recording in sorted(recordings, key=lambda item: item.recording_id):
        validate_recording_index(recording)
        window_steps = int(round(duration_s / recording.dt_s)) + 1
        stride_steps = int(round(stride_s / recording.dt_s))
        if not math.isclose(
            (window_steps - 1) * recording.dt_s,
            duration_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ThorDataError("duration_s must be a multiple of recording.dt_s")
        if stride_steps < 1 or not math.isclose(
            stride_steps * recording.dt_s,
            stride_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ThorDataError("stride_s must be a positive multiple of recording.dt_s")

        for object_id in sorted(recording.dynamic_objects):
            track = recording.dynamic_objects[object_id]
            if track.object_type != object_type:
                continue
            source_object_ids.add(object_id)
            for segment_id in np.unique(track.segment_ids):
                segment_indices = np.flatnonzero(track.segment_ids == segment_id)
                for offset in range(
                    0, segment_indices.size - window_steps + 1, stride_steps
                ):
                    candidate_count += 1
                    indices = segment_indices[offset : offset + window_steps]
                    timestamps = track.timestamps[indices]
                    if not np.allclose(
                        np.diff(timestamps),
                        recording.dt_s,
                        rtol=0.0,
                        atol=1e-8,
                    ):
                        rejection_reasons["time_grid"] += 1
                        continue
                    object_poses = track.poses[indices].astype(np.float64)
                    positions = object_poses[:, :2]
                    velocities, mean_speed, max_acceleration, curvature = (
                        _motion_statistics(positions, timestamps)
                    )
                    normalized = _normalize_motion(
                        positions, velocities, object_poses[:, 2]
                    )
                    if normalized is None:
                        rejection_reasons["stationary"] += 1
                        continue
                    if not min_mean_speed_mps <= mean_speed <= max_mean_speed_mps:
                        rejection_reasons["speed"] += 1
                        continue
                    if max_acceleration > max_acceleration_mps2 + 1e-6:
                        rejection_reasons["acceleration"] += 1
                        continue
                    if _overlaps_robot(
                        recording, timestamps, object_poses, track.footprint
                    ):
                        rejection_reasons["robot_overlap"] += 1
                        continue
                    (
                        normalized_positions,
                        normalized_velocities,
                        normalized_headings,
                    ) = normalized
                    start_timestamp = float(timestamps[0])
                    digest = stable_digest(
                        recording.recording_id,
                        object_id,
                        object_type,
                        f"{start_timestamp:.9f}",
                        f"{duration_s:.9f}",
                        size=12,
                    )
                    snippet_id = f"{split}-{object_type}-snippet-{digest}"
                    geometry_source = str(
                        track.provenance.get("geometry_source", "unknown")
                    )
                    orientation_source = str(
                        track.provenance.get("orientation_source", "unknown")
                    )
                    geometry_source_counts[geometry_source] = (
                        geometry_source_counts.get(geometry_source, 0) + 1
                    )
                    orientation_source_counts[orientation_source] = (
                        orientation_source_counts.get(orientation_source, 0) + 1
                    )
                    snippets.append(
                        MotionSnippet(
                            snippet_id=snippet_id,
                            split=split,
                            source_recording_id=recording.recording_id,
                            source_object_id=object_id,
                            object_type=object_type,
                            footprint=track.footprint,
                            start_timestamp=start_timestamp,
                            positions=normalized_positions.astype(np.float32),
                            velocities=normalized_velocities.astype(np.float32),
                            headings=normalized_headings.astype(np.float32),
                            duration_s=float(duration_s),
                            mean_speed_mps=float(mean_speed),
                            max_acceleration_mps2=float(max_acceleration),
                            mean_abs_curvature_per_m=float(curvature),
                            provenance={
                                "source_body_name": track.source_body_name,
                                "raw_role": track.raw_role,
                                "track_provenance": track.provenance,
                            },
                        )
                    )

    snippets.sort(key=lambda item: item.snippet_id)
    summary: dict[str, object] = {
        "split": split,
        "object_type": object_type,
        "recording_count": len(recordings),
        "source_object_count": len(source_object_ids),
        "candidate_count": candidate_count,
        "accepted_count": len(snippets),
        "rejected_count": sum(rejection_reasons.values()),
        "rejection_reasons": rejection_reasons,
        "duration_s": duration_s,
        "stride_s": stride_s,
        "min_mean_speed_mps": min_mean_speed_mps,
        "max_mean_speed_mps": max_mean_speed_mps,
        "max_acceleration_mps2": max_acceleration_mps2,
        "geometry_source_counts": dict(sorted(geometry_source_counts.items())),
        "orientation_source_counts": dict(
            sorted(orientation_source_counts.items())
        ),
    }
    return SnippetLibrary(
        object_type=object_type,
        snippets=tuple(snippets),
        summary=summary,
    )


def save_snippet_library(
    library: SnippetLibrary, path: str | Path
) -> Path:
    """Atomically save a fixed-rate library without object arrays/pickle."""
    snippets = tuple(sorted(library.snippets, key=lambda item: item.snippet_id))
    if library.object_type not in DYNAMIC_OBJECT_TYPES:
        raise ThorDataError("library object_type is invalid")
    for snippet in snippets:
        _validate_snippet(snippet)
        if snippet.object_type != library.object_type:
            raise ThorDataError("snippet type does not match its library")
    lengths = {snippet.positions.shape[0] for snippet in snippets}
    if len(lengths) > 1:
        raise ThorDataError("one snippet library must use one fixed time grid")

    if snippets:
        positions = np.stack([snippet.positions for snippet in snippets])
        velocities = np.stack([snippet.velocities for snippet in snippets])
        headings = np.stack([snippet.headings for snippet in snippets])
    else:
        positions = np.empty((0, 0, 2), dtype=np.float32)
        velocities = np.empty((0, 0, 2), dtype=np.float32)
        headings = np.empty((0, 0), dtype=np.float32)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "object_type": library.object_type,
        "summary": library.summary,
        "snippets": [
            {
                "snippet_id": snippet.snippet_id,
                "split": snippet.split,
                "source_recording_id": snippet.source_recording_id,
                "source_object_id": snippet.source_object_id,
                "object_type": snippet.object_type,
                "footprint": snippet.footprint,
                "start_timestamp": snippet.start_timestamp,
                "duration_s": snippet.duration_s,
                "mean_speed_mps": snippet.mean_speed_mps,
                "max_acceleration_mps2": snippet.max_acceleration_mps2,
                "mean_abs_curvature_per_m": (
                    snippet.mean_abs_curvature_per_m
                ),
                "provenance": snippet.provenance,
            }
            for snippet in snippets
        ],
    }
    output_path = Path(path)
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            positions=positions,
            velocities=velocities,
            headings=headings,
            meta_json=np.asarray(
                json.dumps(metadata, sort_keys=True, allow_nan=False)
            ),
        )
    temporary.replace(output_path)
    return output_path


def load_snippet_library(path: str | Path) -> SnippetLibrary:
    """Load and validate a library created by :func:`save_snippet_library`."""
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(
            str(payload["meta_json"]), parse_constant=_reject_json_constant
        )
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ThorDataError("snippet library schema_version mismatch")
        positions = payload["positions"].copy()
        velocities = payload["velocities"].copy()
        headings = payload["headings"].copy()
    rows = metadata["snippets"]
    if (
        positions.shape[0] != len(rows)
        or velocities.shape != positions.shape
        or headings.shape != positions.shape[:2]
    ):
        raise ThorDataError("snippet library arrays and metadata do not align")
    snippets = tuple(
        MotionSnippet(
            snippet_id=row["snippet_id"],
            split=row["split"],
            source_recording_id=row["source_recording_id"],
            source_object_id=row["source_object_id"],
            object_type=row["object_type"],
            footprint=row["footprint"],
            start_timestamp=float(row["start_timestamp"]),
            positions=positions[index],
            velocities=velocities[index],
            headings=headings[index],
            duration_s=float(row["duration_s"]),
            mean_speed_mps=float(row["mean_speed_mps"]),
            max_acceleration_mps2=float(row["max_acceleration_mps2"]),
            mean_abs_curvature_per_m=float(
                row["mean_abs_curvature_per_m"]
            ),
            provenance=row["provenance"],
        )
        for index, row in enumerate(rows)
    )
    for snippet in snippets:
        _validate_snippet(snippet)
    library = SnippetLibrary(
        object_type=metadata["object_type"],
        snippets=snippets,
        summary=metadata["summary"],
    )
    if any(snippet.object_type != library.object_type for snippet in snippets):
        raise ThorDataError("snippet type does not match its library")
    return library


def audit_snippet_source_overlap(
    libraries: list[SnippetLibrary] | tuple[SnippetLibrary, ...],
) -> dict[str, object]:
    """Audit recording, object, and snippet provenance across splits."""
    rows = [
        {
            "split": snippet.split,
            "source_recording_id": snippet.source_recording_id,
            "source_object_id": snippet.source_object_id,
            "snippet_id": snippet.snippet_id,
        }
        for library in libraries
        for snippet in library.snippets
    ]
    report = audit_split_leakage(rows)
    splits_by_object: dict[str, set[str]] = {}
    for row in rows:
        splits_by_object.setdefault(str(row["source_object_id"]), set()).add(
            str(row["split"])
        )
    overlaps = [
        {"value": object_id, "splits": sorted(splits)}
        for object_id, splits in sorted(splits_by_object.items())
        if len(splits) > 1
    ]
    report["fields"]["object"] = {
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
    }
    report["total_overlap_count"] += len(overlaps)
    report["status"] = (
        "ok" if report["total_overlap_count"] == 0 else "leakage_detected"
    )
    return report


def write_snippet_artifacts(
    library: SnippetLibrary,
    output_dir: str | Path,
    *,
    overlap_report: dict[str, object],
) -> dict[str, Path]:
    """Atomically write one split's library, provenance, and audit report."""
    if overlap_report.get("total_overlap_count") != 0:
        raise ThorDataError("refusing to write a leaking snippet library")
    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_name(f".{output_path.name}.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        save_snippet_library(library, staging / "snippet_library.npz")
        rows = [
            {
                "schema_version": SCHEMA_VERSION,
                "snippet_id": snippet.snippet_id,
                "split": snippet.split,
                "source_recording_id": snippet.source_recording_id,
                "source_object_id": snippet.source_object_id,
                "object_type": snippet.object_type,
                "footprint": snippet.footprint,
                "start_timestamp": snippet.start_timestamp,
            }
            for snippet in sorted(
                library.snippets, key=lambda item: item.snippet_id
            )
        ]
        manifest = "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        )
        summary = {"schema_version": SCHEMA_VERSION, **library.summary}
        (staging / "source_manifest.jsonl").write_text(
            manifest, encoding="utf-8"
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "source_overlap_report.json").write_text(
            json.dumps(
                overlap_report, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        staging.replace(output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "directory": output_path,
        "library": output_path / "snippet_library.npz",
        "manifest": output_path / "source_manifest.jsonl",
        "summary": output_path / "summary.json",
        "overlap_report": output_path / "source_overlap_report.json",
    }
