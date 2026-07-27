"""Standalone fixed-layout 40-frame THOR dynamic-object snippet libraries."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from src.contracts import DYNAMIC_OBJECT_TYPES, SCHEMA_VERSION, validate_dynamic_object_spec
from src.datasets.motion_snippet_utils import (
    motion_statistics,
    normalize_motion,
    overlaps_robot,
)
from src.datasets.split_manager import validate_split_provenance
from src.datasets.thor_adapter import RecordingIndex, ThorDataError, validate_recording_index
from src.utils.seeding import stable_digest


LONG_MOTION_SNIPPET_LAYOUT = MappingProxyType(
    {
        "motion_snippet_layout_version": "history8_current7_future32_v1",
        "sample_count": 40,
        "history_steps": 8,
        "future_steps": 32,
        "current_index": 7,
        "sample_dt_s": 0.2,
        "duration_s": 7.8,
    }
)
_LIBRARY_FILENAME = "snippet_library_40.npz"
_MANIFEST_FILENAME = "source_manifest_40.jsonl"
_SUMMARY_FILENAME = "summary_40.json"
_SEMANTIC_FILENAME = "semantic_digest_40.json"
_CHECKSUM_FILENAME = "artifact_checksums_40.sha256"
_CHECKSUM_SUMMARY_FILENAME = "artifact_checksum_summary_40.json"
_COMPLETE_FILENAME = ".producer-complete"
_SEMANTIC_DOMAIN = b"long_motion_snippet_library_semantic_v1\0"


@dataclass(frozen=True)
class LongMotionSnippet:
    """One fixed-rate 40-frame local dynamic-object motion snippet."""

    snippet_id: str
    split: str
    source_recording_id: str
    source_session_id: str
    source_object_id: str
    object_type: str
    footprint: dict
    start_timestamp: float
    positions: np.ndarray
    velocities: np.ndarray
    headings: np.ndarray
    duration_s: float
    mean_speed_mps: float
    max_acceleration_mps2: float
    mean_abs_curvature_per_m: float
    provenance: dict


@dataclass(frozen=True)
class LongSnippetLibrary:
    """Accepted 40-frame snippets and their immutable layout/time metadata."""

    object_type: str
    snippets: tuple[LongMotionSnippet, ...]
    relative_time_s: np.ndarray
    summary: dict[str, object]
    split_provenance: dict[str, object]


def _reject_json_constant(value: str) -> None:
    raise ThorDataError(f"JSON metadata must not contain {value}")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _layout_metadata() -> dict[str, object]:
    return dict(LONG_MOTION_SNIPPET_LAYOUT)


def _relative_time_grid() -> np.ndarray:
    layout = LONG_MOTION_SNIPPET_LAYOUT
    return (
        np.arange(int(layout["sample_count"]), dtype=np.float64)
        * float(layout["sample_dt_s"])
        - float(layout["current_index"]) * float(layout["sample_dt_s"])
    )


def _validate_layout_metadata(metadata: Mapping[str, object], *, context: str) -> None:
    for field, expected in LONG_MOTION_SNIPPET_LAYOUT.items():
        if metadata.get(field) != expected:
            raise ThorDataError(
                f"{context} LongMotionSnippet layout requires {field}={expected!r}"
            )


def _validate_relative_time_grid(relative_time_s: np.ndarray) -> None:
    expected = _relative_time_grid()
    if relative_time_s.shape != expected.shape or relative_time_s.dtype != np.float64:
        raise ThorDataError("relative time grid must be float64[40]")
    if not np.isfinite(relative_time_s).all() or not np.array_equal(
        relative_time_s, expected
    ):
        raise ThorDataError("relative time grid violates frozen 40-frame layout")


def _stack_library_arrays(
    snippets: tuple[LongMotionSnippet, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_count = int(LONG_MOTION_SNIPPET_LAYOUT["sample_count"])
    if snippets:
        return (
            np.stack([snippet.positions for snippet in snippets]),
            np.stack([snippet.velocities for snippet in snippets]),
            np.stack([snippet.headings for snippet in snippets]),
        )
    return (
        np.empty((0, sample_count, 2), dtype=np.float32),
        np.empty((0, sample_count, 2), dtype=np.float32),
        np.empty((0, sample_count), dtype=np.float32),
    )


def _array_sha256(
    positions: np.ndarray,
    velocities: np.ndarray,
    headings: np.ndarray,
    relative_time_s: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for name, array in (
        ("positions", positions),
        ("velocities", velocities),
        ("headings", headings),
        ("relative_time_s", relative_time_s),
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
        digest.update(_canonical_json(list(contiguous.shape)) + b"\0")
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _snippet_row(snippet: LongMotionSnippet) -> dict[str, object]:
    return {
        "snippet_id": snippet.snippet_id,
        "split": snippet.split,
        "source_recording_id": snippet.source_recording_id,
        "source_session_id": snippet.source_session_id,
        "source_object_id": snippet.source_object_id,
        "object_type": snippet.object_type,
        "footprint": snippet.footprint,
        "start_timestamp": snippet.start_timestamp,
        "duration_s": snippet.duration_s,
        "mean_speed_mps": snippet.mean_speed_mps,
        "max_acceleration_mps2": snippet.max_acceleration_mps2,
        "mean_abs_curvature_per_m": snippet.mean_abs_curvature_per_m,
        "provenance": snippet.provenance,
    }


def _semantic_digest(
    library: LongSnippetLibrary,
    *,
    array_sha256: str,
) -> str:
    semantic_summary = {
        key: value
        for key, value in library.summary.items()
        if key not in {"array_sha256", "semantic_digest_sha256"}
    }
    payload = {
        "object_type": library.object_type,
        "layout": _layout_metadata(),
        "relative_time_s": library.relative_time_s.tolist(),
        "array_sha256": array_sha256,
        "summary": semantic_summary,
        "split_provenance": library.split_provenance,
        "snippets": [_snippet_row(item) for item in library.snippets],
    }
    return hashlib.sha256(_SEMANTIC_DOMAIN + _canonical_json(payload)).hexdigest()


def _validate_snippet(snippet: LongMotionSnippet) -> None:
    if snippet.object_type not in DYNAMIC_OBJECT_TYPES:
        raise ThorDataError("long snippet object_type is invalid")
    validate_dynamic_object_spec(
        {"object_type": snippet.object_type, "footprint": snippet.footprint}
    )
    sample_count = int(LONG_MOTION_SNIPPET_LAYOUT["sample_count"])
    if snippet.positions.shape != (sample_count, 2):
        raise ThorDataError("long snippet positions shape must be [40,2]")
    if snippet.velocities.shape != snippet.positions.shape:
        raise ThorDataError("long snippet velocities must match positions shape")
    if snippet.headings.shape != (sample_count,):
        raise ThorDataError("long snippet headings shape must be [40]")
    if (
        snippet.positions.dtype != np.float32
        or snippet.velocities.dtype != np.float32
        or snippet.headings.dtype != np.float32
    ):
        raise ThorDataError("long snippet arrays must be float32")
    if not (
        np.isfinite(snippet.positions).all()
        and np.isfinite(snippet.velocities).all()
        and np.isfinite(snippet.headings).all()
    ):
        raise ThorDataError("long snippet arrays contain NaN/Inf")
    if not isinstance(snippet.provenance, dict):
        raise ThorDataError("long snippet provenance must be a dict")
    if not all(
        isinstance(value, str) and value
        for value in (
            snippet.snippet_id,
            snippet.split,
            snippet.source_recording_id,
            snippet.source_session_id,
            snippet.source_object_id,
        )
    ):
        raise ThorDataError("long snippet identity fields must be non-empty strings")
    if snippet.duration_s != float(LONG_MOTION_SNIPPET_LAYOUT["duration_s"]):
        raise ThorDataError("long snippet duration violates frozen 40-frame layout")
    if not all(
        math.isfinite(float(value))
        for value in (
            snippet.start_timestamp,
            snippet.duration_s,
            snippet.mean_speed_mps,
            snippet.max_acceleration_mps2,
            snippet.mean_abs_curvature_per_m,
        )
    ):
        raise ThorDataError("long snippet statistics contain NaN/Inf")


def build_long_snippet_library(
    recordings: list[RecordingIndex] | tuple[RecordingIndex, ...],
    *,
    split: str,
    object_type: str,
    stride_s: float = 1.0,
    min_mean_speed_mps: float = 0.3,
    max_mean_speed_mps: float = 2.0,
    max_acceleration_mps2: float = 2.5,
    split_provenance: Mapping[str, object],
) -> LongSnippetLibrary:
    """Re-crop complete 40-frame windows from continuous RecordingIndex tracks."""
    if split not in {"train", "calibration", "val", "test"}:
        raise ThorDataError("split must be train, calibration, val, or test")
    if object_type not in DYNAMIC_OBJECT_TYPES:
        raise ThorDataError("object_type is not part of the frozen taxonomy")
    provenance = validate_split_provenance(split_provenance)
    if not recordings:
        raise ThorDataError("recordings must not be empty")
    if not all(
        math.isfinite(value)
        for value in (
            stride_s,
            min_mean_speed_mps,
            max_mean_speed_mps,
            max_acceleration_mps2,
        )
    ):
        raise ThorDataError("long snippet parameters must be finite")
    if not 0.0 <= min_mean_speed_mps < max_mean_speed_mps:
        raise ThorDataError("speed bounds are invalid")
    if stride_s <= 0.0:
        raise ThorDataError("stride_s must be positive")
    if max_acceleration_mps2 <= 0.0:
        raise ThorDataError("max_acceleration_mps2 must be positive")

    layout = LONG_MOTION_SNIPPET_LAYOUT
    sample_count = int(layout["sample_count"])
    sample_dt_s = float(layout["sample_dt_s"])
    duration_s = float(layout["duration_s"])
    snippets: list[LongMotionSnippet] = []
    rejections = {
        "insufficient_contiguous_duration": 0,
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
        if not math.isclose(
            recording.dt_s, sample_dt_s, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ThorDataError("recording dt_s violates frozen 40-frame layout")
        stride_steps = int(round(stride_s / recording.dt_s))
        if stride_steps < 1 or not math.isclose(
            stride_steps * recording.dt_s, stride_s, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ThorDataError("stride_s must be a positive multiple of recording.dt_s")
        if not math.isclose(
            (sample_count - 1) * recording.dt_s,
            duration_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ThorDataError("recording dt_s violates frozen long duration")

        for object_id in sorted(recording.dynamic_objects):
            track = recording.dynamic_objects[object_id]
            if track.object_type != object_type:
                continue
            source_object_ids.add(object_id)
            for segment_id in np.unique(track.segment_ids):
                segment_indices = np.flatnonzero(track.segment_ids == segment_id)
                if segment_indices.size < sample_count:
                    candidate_count += 1
                    rejections["insufficient_contiguous_duration"] += 1
                    continue
                for offset in range(
                    0, segment_indices.size - sample_count + 1, stride_steps
                ):
                    candidate_count += 1
                    indices = segment_indices[offset : offset + sample_count]
                    timestamps = track.timestamps[indices]
                    if not np.allclose(
                        np.diff(timestamps),
                        recording.dt_s,
                        rtol=0.0,
                        atol=1e-8,
                    ):
                        rejections["time_grid"] += 1
                        continue
                    object_poses = track.poses[indices].astype(np.float64)
                    velocities, mean_speed, max_acceleration, curvature = (
                        motion_statistics(object_poses[:, :2], timestamps)
                    )
                    normalized = normalize_motion(
                        object_poses[:, :2], velocities, object_poses[:, 2]
                    )
                    if normalized is None:
                        rejections["stationary"] += 1
                        continue
                    if not min_mean_speed_mps <= mean_speed <= max_mean_speed_mps:
                        rejections["speed"] += 1
                        continue
                    if max_acceleration > max_acceleration_mps2 + 1e-6:
                        rejections["acceleration"] += 1
                        continue
                    if overlaps_robot(
                        recording, timestamps, object_poses, track.footprint
                    ):
                        rejections["robot_overlap"] += 1
                        continue
                    start_timestamp = float(timestamps[0])
                    digest = stable_digest(
                        recording.recording_id,
                        recording.session_id,
                        object_id,
                        object_type,
                        str(layout["motion_snippet_layout_version"]),
                        f"{start_timestamp:.9f}",
                        f"{duration_s:.9f}",
                        size=12,
                    )
                    positions, normalized_velocities, headings = normalized
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
                        LongMotionSnippet(
                            snippet_id=f"{split}-{object_type}-long40-{digest}",
                            split=split,
                            source_recording_id=recording.recording_id,
                            source_session_id=recording.session_id,
                            source_object_id=object_id,
                            object_type=object_type,
                            footprint=track.footprint,
                            start_timestamp=start_timestamp,
                            positions=positions.astype(np.float32),
                            velocities=normalized_velocities.astype(np.float32),
                            headings=headings.astype(np.float32),
                            duration_s=duration_s,
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

    ordered = tuple(sorted(snippets, key=lambda item: item.snippet_id))
    relative_time_s = _relative_time_grid()
    positions, velocities, headings = _stack_library_arrays(ordered)
    array_digest = _array_sha256(positions, velocities, headings, relative_time_s)
    summary: dict[str, object] = {
        "split": split,
        "object_type": object_type,
        "recording_count": len(recordings),
        "source_object_count": len(source_object_ids),
        "candidate_count": candidate_count,
        "accepted_count": len(ordered),
        "rejected_count": sum(rejections.values()),
        "rejection_reasons": rejections,
        "stride_s": stride_s,
        "min_mean_speed_mps": min_mean_speed_mps,
        "max_mean_speed_mps": max_mean_speed_mps,
        "max_acceleration_mps2": max_acceleration_mps2,
        "geometry_source_counts": dict(sorted(geometry_source_counts.items())),
        "orientation_source_counts": dict(sorted(orientation_source_counts.items())),
        "array_sha256": array_digest,
        "split_manifest_digest": provenance["split_manifest_digest"],
        "split_provenance": provenance,
        **_layout_metadata(),
    }
    provisional = LongSnippetLibrary(
        object_type=object_type,
        snippets=ordered,
        relative_time_s=relative_time_s,
        summary=summary,
        split_provenance=provenance,
    )
    summary["semantic_digest_sha256"] = _semantic_digest(
        provisional, array_sha256=array_digest
    )
    return LongSnippetLibrary(
        object_type=object_type,
        snippets=ordered,
        relative_time_s=relative_time_s,
        summary=summary,
        split_provenance=provenance,
    )


def _validate_library(library: LongSnippetLibrary) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if library.object_type not in DYNAMIC_OBJECT_TYPES:
        raise ThorDataError("long snippet library object_type is invalid")
    provenance = validate_split_provenance(library.split_provenance)
    _validate_layout_metadata(library.summary, context="long snippet summary")
    _validate_relative_time_grid(library.relative_time_s)
    if library.summary.get("split_provenance") != provenance:
        raise ThorDataError("long snippet summary split provenance mismatch")
    if library.summary.get("split_manifest_digest") != provenance["split_manifest_digest"]:
        raise ThorDataError("long snippet summary split_manifest_digest mismatch")
    snippets = tuple(sorted(library.snippets, key=lambda item: item.snippet_id))
    if snippets != library.snippets:
        raise ThorDataError("long snippets must be ordered by snippet_id")
    if len({item.snippet_id for item in snippets}) != len(snippets):
        raise ThorDataError("long snippet IDs must be unique")
    for snippet in snippets:
        _validate_snippet(snippet)
        if snippet.object_type != library.object_type:
            raise ThorDataError("long snippet type does not match library")
        absolute_times = snippet.start_timestamp + library.relative_time_s
        if not np.allclose(
            np.diff(absolute_times),
            float(LONG_MOTION_SNIPPET_LAYOUT["sample_dt_s"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ThorDataError("long snippet absolute time grid is not continuous")
    positions, velocities, headings = _stack_library_arrays(snippets)
    array_digest = _array_sha256(
        positions, velocities, headings, library.relative_time_s
    )
    if library.summary.get("array_sha256") != array_digest:
        raise ThorDataError("long snippet summary array_sha256 mismatch")
    if library.summary.get("semantic_digest_sha256") != _semantic_digest(
        library, array_sha256=array_digest
    ):
        raise ThorDataError("long snippet summary semantic_digest_sha256 mismatch")
    return positions, velocities, headings


def save_long_snippet_library(library: LongSnippetLibrary, path: str | Path) -> Path:
    """Atomically save a standalone 40-frame library without pickle/object arrays."""
    positions, velocities, headings = _validate_library(library)
    output_path = Path(path)
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "object_type": library.object_type,
        "summary": library.summary,
        "split_provenance": library.split_provenance,
        "split_manifest_digest": library.split_provenance["split_manifest_digest"],
        "array_sha256": library.summary["array_sha256"],
        "semantic_digest_sha256": library.summary["semantic_digest_sha256"],
        **_layout_metadata(),
        "snippets": [_snippet_row(item) for item in library.snippets],
    }
    temporary = output_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            positions=positions,
            velocities=velocities,
            headings=headings,
            relative_time_s=library.relative_time_s,
            meta_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        )
    temporary.replace(output_path)
    return output_path


def load_long_snippet_library(path: str | Path) -> LongSnippetLibrary:
    """Load only a verified standalone 40-frame library."""
    with np.load(Path(path), allow_pickle=False) as payload:
        if set(payload.files) != {
            "positions",
            "velocities",
            "headings",
            "relative_time_s",
            "meta_json",
        }:
            raise ThorDataError("long snippet library array keys mismatch")
        metadata = json.loads(
            str(payload["meta_json"]), parse_constant=_reject_json_constant
        )
        positions = payload["positions"].copy()
        velocities = payload["velocities"].copy()
        headings = payload["headings"].copy()
        relative_time_s = payload["relative_time_s"].copy()
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ThorDataError("long snippet library schema_version mismatch")
    _validate_layout_metadata(metadata, context="long snippet library")
    _validate_relative_time_grid(relative_time_s)
    rows = metadata.get("snippets")
    if not isinstance(rows, list):
        raise ThorDataError("long snippet library metadata needs snippet rows")
    sample_count = int(LONG_MOTION_SNIPPET_LAYOUT["sample_count"])
    if (
        positions.shape != (len(rows), sample_count, 2)
        or velocities.shape != positions.shape
        or headings.shape != (len(rows), sample_count)
    ):
        raise ThorDataError("long snippet library arrays and metadata do not align")
    if (
        positions.dtype != np.float32
        or velocities.dtype != np.float32
        or headings.dtype != np.float32
    ):
        raise ThorDataError("long snippet library arrays must be float32")
    if not (
        np.isfinite(positions).all()
        and np.isfinite(velocities).all()
        and np.isfinite(headings).all()
    ):
        raise ThorDataError("long snippet library arrays contain NaN/Inf")
    array_digest = _array_sha256(positions, velocities, headings, relative_time_s)
    if metadata.get("array_sha256") != array_digest:
        raise ThorDataError("long snippet library array_sha256 mismatch")
    summary = metadata.get("summary")
    if not isinstance(summary, dict):
        raise ThorDataError("long snippet summary must be an object")
    _validate_layout_metadata(summary, context="long snippet summary")
    if summary.get("array_sha256") != array_digest:
        raise ThorDataError("long snippet summary array_sha256 mismatch")
    if metadata.get("semantic_digest_sha256") != summary.get("semantic_digest_sha256"):
        raise ThorDataError("long snippet semantic_digest metadata mismatch")
    try:
        provenance = validate_split_provenance(metadata.get("split_provenance"))
    except (TypeError, ValueError) as error:
        raise ThorDataError(f"invalid long snippet split provenance: {error}") from error
    if metadata.get("split_manifest_digest") != provenance["split_manifest_digest"]:
        raise ThorDataError("long snippet split_manifest_digest mismatch")
    snippets = tuple(
        LongMotionSnippet(
            snippet_id=row["snippet_id"],
            split=row["split"],
            source_recording_id=row["source_recording_id"],
            source_session_id=row["source_session_id"],
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
            mean_abs_curvature_per_m=float(row["mean_abs_curvature_per_m"]),
            provenance=row["provenance"],
        )
        for index, row in enumerate(rows)
    )
    library = LongSnippetLibrary(
        object_type=metadata["object_type"],
        snippets=snippets,
        relative_time_s=relative_time_s,
        summary=summary,
        split_provenance=provenance,
    )
    _validate_library(library)
    return library


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_manifest_rows(library: LongSnippetLibrary) -> list[dict[str, object]]:
    return [
        {
            "schema_version": SCHEMA_VERSION,
            **_snippet_row(snippet),
            "split_manifest_digest": library.split_provenance["split_manifest_digest"],
            "split_provenance": library.split_provenance,
            **_layout_metadata(),
        }
        for snippet in library.snippets
    ]


def write_long_snippet_artifacts(
    library: LongSnippetLibrary,
    output_dir: str | Path,
    *,
    overlap_report: Mapping[str, object],
) -> dict[str, Path]:
    """Atomically publish one standalone 40-frame library and checksum envelope."""
    if overlap_report.get("status") != "ok":
        raise ThorDataError("refusing to write a leaking long snippet library")
    _validate_library(library)
    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_name(f".{output_path.name}.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        library_path = save_long_snippet_library(library, staging / _LIBRARY_FILENAME)
        rows = _source_manifest_rows(library)
        (staging / _MANIFEST_FILENAME).write_text(
            "".join(
                json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
            ),
            encoding="utf-8",
        )
        _write_json(
            staging / _SUMMARY_FILENAME,
            {"schema_version": SCHEMA_VERSION, **library.summary},
        )
        _write_json(staging / "source_overlap_report_40.json", dict(overlap_report))
        _write_json(
            staging / _SEMANTIC_FILENAME,
            {
                "schema_version": SCHEMA_VERSION,
                "semantic_digest_sha256": library.summary["semantic_digest_sha256"],
                "array_sha256": library.summary["array_sha256"],
                "library_filename": _LIBRARY_FILENAME,
                "manifest_filename": _MANIFEST_FILENAME,
                "summary_filename": _SUMMARY_FILENAME,
                **_layout_metadata(),
            },
        )
        payload_names = sorted(
            (
                _LIBRARY_FILENAME,
                _MANIFEST_FILENAME,
                _SUMMARY_FILENAME,
                _SEMANTIC_FILENAME,
                "source_overlap_report_40.json",
            )
        )
        checksums = "".join(
            f"{_sha256_file(staging / name)}  {name}\n" for name in payload_names
        )
        (staging / _CHECKSUM_FILENAME).write_text(checksums, encoding="utf-8")
        _write_json(
            staging / _CHECKSUM_SUMMARY_FILENAME,
            {
                "checksum_algorithm": "sha256",
                "checksum_manifest": _CHECKSUM_FILENAME,
                "checksum_manifest_sha256": _sha256_file(staging / _CHECKSUM_FILENAME),
                "covered_file_count": len(payload_names),
                "covered_files": payload_names,
                "status": "complete",
            },
        )
        (staging / _COMPLETE_FILENAME).write_text("", encoding="utf-8")
        staging.replace(output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "directory": output_path,
        "library": output_path / _LIBRARY_FILENAME,
        "manifest": output_path / _MANIFEST_FILENAME,
        "summary": output_path / _SUMMARY_FILENAME,
        "semantic_digest": output_path / _SEMANTIC_FILENAME,
        "checksum_manifest": output_path / _CHECKSUM_FILENAME,
        "checksum_summary": output_path / _CHECKSUM_SUMMARY_FILENAME,
        "overlap_report": output_path / "source_overlap_report_40.json",
    }


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ThorDataError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ThorDataError(f"{label} must contain an object")
    return value


def _verify_checksum_envelope(root: Path) -> None:
    if not (root / _COMPLETE_FILENAME).is_file():
        raise ThorDataError("long snippet artifact completion marker is missing")
    summary = _read_json(root / _CHECKSUM_SUMMARY_FILENAME, label="checksum summary")
    if (
        summary.get("checksum_algorithm") != "sha256"
        or summary.get("checksum_manifest") != _CHECKSUM_FILENAME
        or summary.get("status") != "complete"
    ):
        raise ThorDataError("long snippet checksum summary is invalid")
    manifest = root / _CHECKSUM_FILENAME
    if summary.get("checksum_manifest_sha256") != _sha256_file(manifest):
        raise ThorDataError("long snippet checksum manifest digest mismatch")
    expected_names = {
        _LIBRARY_FILENAME,
        _MANIFEST_FILENAME,
        _SUMMARY_FILENAME,
        _SEMANTIC_FILENAME,
        "source_overlap_report_40.json",
    }
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or name in entries
        ):
            raise ThorDataError("long snippet checksum manifest is invalid")
        entries[name] = digest
    if set(entries) != expected_names:
        raise ThorDataError("long snippet checksum manifest payload set mismatch")
    if summary.get("covered_files") != sorted(expected_names):
        raise ThorDataError("long snippet checksum summary payload set mismatch")
    for name, expected in entries.items():
        path = root / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise ThorDataError(f"long snippet checksum mismatch for {name}")


def load_long_snippet_artifact(directory: str | Path) -> LongSnippetLibrary:
    """Load an artifact only after checksum and semantic sidecar verification."""
    root = Path(directory)
    _verify_checksum_envelope(root)
    library = load_long_snippet_library(root / _LIBRARY_FILENAME)
    summary = _read_json(root / _SUMMARY_FILENAME, label="long snippet summary")
    expected_summary = {"schema_version": SCHEMA_VERSION, **library.summary}
    if summary != expected_summary:
        raise ThorDataError("long snippet outer summary mismatch")
    semantic = _read_json(root / _SEMANTIC_FILENAME, label="long snippet semantic digest")
    _validate_layout_metadata(semantic, context="long snippet semantic sidecar")
    if (
        semantic.get("schema_version") != SCHEMA_VERSION
        or semantic.get("semantic_digest_sha256")
        != library.summary["semantic_digest_sha256"]
        or semantic.get("array_sha256") != library.summary["array_sha256"]
        or semantic.get("library_filename") != _LIBRARY_FILENAME
        or semantic.get("manifest_filename") != _MANIFEST_FILENAME
        or semantic.get("summary_filename") != _SUMMARY_FILENAME
    ):
        raise ThorDataError("long snippet semantic digest sidecar mismatch")
    rows: list[dict[str, object]] = []
    for line in (root / _MANIFEST_FILENAME).read_text(encoding="utf-8").splitlines():
        row = json.loads(line, parse_constant=_reject_json_constant)
        if not isinstance(row, dict):
            raise ThorDataError("long snippet manifest row must be an object")
        rows.append(row)
    if len(rows) != len(library.snippets):
        raise ThorDataError("long snippet manifest count mismatch")
    if {str(row.get("snippet_id")) for row in rows} != {
        item.snippet_id for item in library.snippets
    }:
        raise ThorDataError("long snippet manifest IDs mismatch")
    return library
