"""Layout-independent motion-snippet extraction and split-audit helpers."""

from __future__ import annotations

import json
import math
from typing import Any, Sequence

import numpy as np

from src.datasets.split_manager import (
    SplitAuditPolicy,
    audit_split_leakage,
    validate_split_provenance,
)
from src.datasets.thor_adapter import RecordingIndex, ThorDataError
from src.geometry import (
    CircleFootprint,
    RectangleFootprint,
    inflate_footprint,
    trajectory_signed_clearances,
)


def robot_indices(
    recording: RecordingIndex,
    timestamps: np.ndarray,
) -> np.ndarray | None:
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


def overlaps_robot(
    recording: RecordingIndex,
    timestamps: np.ndarray,
    object_poses: np.ndarray,
    footprint: dict[str, object],
) -> bool:
    indices = robot_indices(recording, timestamps)
    if indices is None:
        return True
    robot = inflate_footprint(
        RectangleFootprint(length_m=0.70, width_m=0.55), 0.15
    )
    if footprint["kind"] == "circle":
        dynamic_object = CircleFootprint(
            radius_m=float(footprint["radius_m"])
        )
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


def motion_statistics(
    positions: np.ndarray,
    timestamps: np.ndarray,
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


def normalize_motion(
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


def audit_snippet_source_overlap(
    libraries: Sequence[Any],
    *,
    policy: SplitAuditPolicy | None = None,
) -> dict[str, object]:
    """Audit recording, object, and snippet provenance across splits."""

    provenances = {
        json.dumps(
            validate_split_provenance(library.split_provenance),
            sort_keys=True,
            separators=(",", ":"),
        )
        for library in libraries
    }
    if len(provenances) > 1:
        raise ThorDataError("snippet libraries use different split provenance")
    rows = [
        {
            "split": snippet.split,
            "source_recording_id": snippet.source_recording_id,
            "source_session_id": snippet.source_session_id,
            "source_object_id": snippet.source_object_id,
            "snippet_id": snippet.snippet_id,
        }
        for library in libraries
        for snippet in library.snippets
    ]
    report = audit_split_leakage(rows, policy=policy)
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
    report["detected_overlap_count"] += len(overlaps)
    report["disallowed_overlap_count"] += len(overlaps)
    if report["missing_required_row_count"]:
        report["status"] = "provenance_incomplete"
    elif report["disallowed_overlap_count"]:
        report["status"] = "leakage_detected"
    else:
        report["status"] = "ok"
    return report
