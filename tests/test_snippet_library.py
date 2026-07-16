"""Tests for split- and type-isolated dynamic-object motion snippets."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.datasets.thor_adapter import DynamicObjectTrack, RecordingIndex


def _track(
    *,
    recording_id: str,
    body_name: str,
    object_type: str,
    timestamps: np.ndarray,
    speed_mps: float,
    lateral_m: float,
    footprint: dict,
) -> DynamicObjectTrack:
    object_id = f"{recording_id}::{body_name}"
    poses = np.column_stack(
        (
            2.0 + speed_mps * timestamps,
            np.full_like(timestamps, lateral_m),
            np.zeros_like(timestamps),
        )
    ).astype(np.float32)
    return DynamicObjectTrack(
        object_id=object_id,
        source_body_name=body_name,
        object_type=object_type,
        raw_role="Visitors-Alone" if object_type == "human" else "Equipment",
        timestamps=timestamps.copy(),
        poses=poses,
        velocities=np.tile(
            np.array([speed_mps, 0.0], dtype=np.float32),
            (timestamps.size, 1),
        ),
        segment_ids=np.zeros(timestamps.shape, dtype=np.int32),
        footprint=footprint,
        provenance={
            "geometry_source": (
                "config_human" if object_type == "human" else "config_fallback"
            ),
            "orientation_source": "qtm_rotation",
        },
    )


def _recording(recording_id: str = "toy-recording") -> RecordingIndex:
    timestamps = np.arange(0.0, 6.0 + 1e-9, 0.2, dtype=np.float64)
    human = _track(
        recording_id=recording_id,
        body_name="Helmet_1",
        object_type="human",
        timestamps=timestamps,
        speed_mps=1.0,
        lateral_m=1.0,
        footprint={"kind": "circle", "radius_m": 0.30},
    )
    carried = _track(
        recording_id=recording_id,
        body_name="LO1",
        object_type="carried_object",
        timestamps=timestamps,
        speed_mps=0.10,
        lateral_m=2.0,
        footprint={"kind": "rectangle", "length_m": 0.80, "width_m": 0.20},
    )
    unknown = _track(
        recording_id=recording_id,
        body_name="Mystery_Rig",
        object_type="unknown_dynamic",
        timestamps=timestamps,
        speed_mps=0.20,
        lateral_m=3.0,
        footprint={"kind": "circle", "radius_m": 0.50},
    )
    stationary = _track(
        recording_id=recording_id,
        body_name="Storage_Unit",
        object_type="unknown_dynamic",
        timestamps=timestamps,
        speed_mps=0.0,
        lateral_m=4.0,
        footprint={"kind": "circle", "radius_m": 0.50},
    )
    return RecordingIndex(
        recording_id=recording_id,
        session_id="toy-session",
        timestamps=timestamps,
        robot_pose=np.zeros((timestamps.size, 3), dtype=np.float32),
        robot_twist=np.zeros((timestamps.size, 2), dtype=np.float32),
        robot_segment_ids=np.zeros(timestamps.shape, dtype=np.int32),
        dynamic_objects={
            track.object_id: track
            for track in (human, carried, unknown, stationary)
        },
        static_map=None,
        source_file=f"THOR-Magni_{recording_id}.csv",
        dt_s=0.2,
    )


def _build_human_library(recording: RecordingIndex | None = None, *, split="train"):
    from src.datasets.snippet_library import build_snippet_library

    return build_snippet_library(
        [recording or _recording()],
        split=split,
        object_type="human",
        duration_s=3.0,
        stride_s=1.0,
        min_mean_speed_mps=0.30,
        max_mean_speed_mps=2.00,
        max_acceleration_mps2=2.50,
    )


def test_build_motion_snippets_normalizes_pose_and_preserves_type_geometry():
    library = _build_human_library()

    assert library.object_type == "human"
    assert len(library.snippets) == 4
    snippet = library.snippets[0]
    assert snippet.split == "train"
    assert snippet.source_object_id == "toy-recording::Helmet_1"
    assert snippet.object_type == "human"
    assert snippet.footprint == {"kind": "circle", "radius_m": 0.30}
    assert snippet.positions.shape == (16, 2)
    assert snippet.positions.dtype == np.float32
    assert snippet.velocities.dtype == np.float32
    assert snippet.headings.shape == (16,)
    assert snippet.headings.dtype == np.float32
    assert np.allclose(snippet.positions[0], 0.0, atol=1e-7)
    assert np.allclose(snippet.positions[:, 1], 0.0, atol=1e-6)
    assert np.allclose(snippet.headings, 0.0, atol=1e-6)
    assert snippet.positions[-1, 0] == pytest.approx(3.0, abs=1e-6)
    assert snippet.mean_speed_mps == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(snippet.positions).all()
    assert library.summary["accepted_count"] == 4


def test_parallel_snippet_library_matches_serial():
    from src.datasets.snippet_library import build_snippet_library

    recordings = [_recording("parallel-a"), _recording("parallel-b")]
    kwargs = {
        "split": "train",
        "object_type": "human",
        "duration_s": 3.0,
        "stride_s": 1.0,
        "min_mean_speed_mps": 0.30,
        "max_mean_speed_mps": 2.00,
        "max_acceleration_mps2": 2.50,
    }
    serial = build_snippet_library(recordings, workers=1, **kwargs)
    parallel = build_snippet_library(recordings, workers=2, **kwargs)

    assert parallel.summary == serial.summary
    assert [snippet.snippet_id for snippet in parallel.snippets] == [
        snippet.snippet_id for snippet in serial.snippets
    ]
    for actual, expected in zip(parallel.snippets, serial.snippets):
        assert actual.provenance == expected.provenance
        assert actual.footprint == expected.footprint
        assert np.array_equal(actual.positions, expected.positions)
        assert np.array_equal(actual.velocities, expected.velocities)
        assert np.array_equal(actual.headings, expected.headings)


def test_snippet_library_rejects_nonpositive_workers():
    from src.datasets.snippet_library import build_snippet_library
    from src.datasets.thor_adapter import ThorDataError

    with pytest.raises(ThorDataError, match="workers must be a positive integer"):
        build_snippet_library(
            [_recording()],
            split="train",
            object_type="human",
            workers=0,
        )


def test_snippet_library_round_trip_uses_numeric_npz(tmp_path):
    from src.datasets.snippet_library import (
        load_snippet_library,
        save_snippet_library,
    )

    source = _build_human_library()
    path = save_snippet_library(source, tmp_path / "snippets.npz")
    restored = load_snippet_library(path)

    with np.load(path, allow_pickle=False) as payload:
        assert all(payload[key].dtype != object for key in payload.files)
    assert restored.object_type == "human"
    assert restored.summary == source.summary
    assert np.array_equal(
        restored.snippets[0].headings, source.snippets[0].headings
    )
    assert restored.snippets[0].provenance == source.snippets[0].provenance


def test_source_overlap_audit_detects_recording_and_object_reuse():
    from src.datasets.snippet_library import audit_snippet_source_overlap

    train = _build_human_library(split="train")
    test = _build_human_library(split="test")

    report = audit_snippet_source_overlap([train, test])

    assert report["total_overlap_count"] > 0
    assert report["fields"]["recording"]["overlap_count"] == 1
    assert report["fields"]["object"]["overlaps"] == [
        {
            "value": "toy-recording::Helmet_1",
            "splits": ["test", "train"],
        }
    ]


def test_type_specific_speed_threshold_keeps_slow_carried_object():
    from src.datasets.snippet_library import build_snippet_library

    library = build_snippet_library(
        [_recording()],
        split="train",
        object_type="carried_object",
        min_mean_speed_mps=0.05,
        max_mean_speed_mps=2.00,
        max_acceleration_mps2=2.50,
    )

    assert len(library.snippets) == 4
    assert all(item.object_type == "carried_object" for item in library.snippets)
    assert all(item.mean_speed_mps == pytest.approx(0.10) for item in library.snippets)


def test_stationary_object_remains_indexed_but_is_rejected_from_snippets():
    from src.datasets.snippet_library import build_snippet_library

    recording = _recording()
    assert "toy-recording::Storage_Unit" in recording.dynamic_objects
    library = build_snippet_library(
        [recording],
        split="train",
        object_type="unknown_dynamic",
        min_mean_speed_mps=0.05,
        max_mean_speed_mps=2.00,
        max_acceleration_mps2=2.50,
    )

    assert len(library.snippets) == 4
    assert library.summary["rejection_reasons"]["stationary"] == 4


def test_rectangle_footprint_overlap_is_rejected():
    from src.datasets.snippet_library import build_snippet_library

    recording = _recording()
    object_id = "toy-recording::LO1"
    track = recording.dynamic_objects[object_id]
    poses = track.poses.copy()
    poses[:, 0] = np.arange(poses.shape[0], dtype=np.float32) * 0.02
    poses[:, 1] = 0.50
    overlapping = replace(
        recording,
        dynamic_objects={
            **recording.dynamic_objects,
            object_id: replace(track, poses=poses),
        },
    )
    library = build_snippet_library(
        [overlapping],
        split="train",
        object_type="carried_object",
        min_mean_speed_mps=0.05,
        max_mean_speed_mps=2.00,
        max_acceleration_mps2=2.50,
    )

    assert library.summary["accepted_count"] == 0
    assert library.summary["rejection_reasons"]["robot_overlap"] == 4


def test_snippets_with_excessive_acceleration_are_rejected():
    recording = _recording()
    object_id = "toy-recording::Helmet_1"
    track = recording.dynamic_objects[object_id]
    poses = track.poses.copy()
    poses[8:, 0] += 1.0
    accelerated = replace(
        recording,
        dynamic_objects={
            **recording.dynamic_objects,
            object_id: replace(track, poses=poses),
        },
    )

    result = _build_human_library(accelerated)

    assert result.summary["rejection_reasons"]["acceleration"] > 0
    assert result.summary["accepted_count"] < result.summary["candidate_count"]


def test_build_snippet_library_cli_writes_type_scoped_artifacts(tmp_path):
    from src.datasets.snippet_library import load_snippet_library
    from src.datasets.thor_adapter import write_recording_indexes

    write_recording_indexes(
        [_recording()],
        split="train",
        output_dir=tmp_path / "indexes/train",
    )
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/02_build_snippet_library.py"),
            "--config",
            str(root / "configs/data_thor.yaml"),
            "--split",
            "train",
            "--recording-dir",
            str(tmp_path / "indexes"),
            "--output-dir",
            str(tmp_path / "snippets"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    expected_counts = {"human": 4, "carried_object": 4, "unknown_dynamic": 4}
    for object_type, count in expected_counts.items():
        path = tmp_path / f"snippets/train/{object_type}/snippet_library.npz"
        library = load_snippet_library(path)
        assert library.object_type == object_type
        assert len(library.snippets) == count
    report = json.loads(
        (
            tmp_path
            / "snippets/train/human/source_overlap_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "ok"
    assert "total_accepted_count=12" in completed.stdout
    assert "workers_requested=8" in completed.stdout
    assert "workers_used=1" in completed.stdout
