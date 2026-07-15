"""Tests for split-isolated, normalized pedestrian trajectory snippets."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.datasets.thor_adapter import PedestrianTrack, RecordingIndex


def _recording() -> RecordingIndex:
    timestamps = np.arange(0.0, 6.0 + 1e-9, 0.2, dtype=np.float64)
    robot_pose = np.column_stack(
        (timestamps, np.zeros_like(timestamps), np.zeros_like(timestamps))
    ).astype(np.float32)
    pedestrian = PedestrianTrack(
        participant_id="ped-1",
        timestamps=timestamps.copy(),
        positions=np.column_stack(
            (timestamps + 2.0, np.ones_like(timestamps))
        ).astype(np.float32),
        velocities=np.tile(
            np.array([1.0, 0.0], dtype=np.float32), (timestamps.size, 1)
        ),
        segment_ids=np.zeros(timestamps.shape, dtype=np.int32),
        role="Visitors-Alone",
    )
    return RecordingIndex(
        recording_id="toy-recording",
        session_id="toy-session",
        timestamps=timestamps,
        robot_pose=robot_pose,
        robot_twist=np.tile(
            np.array([1.0, 0.0], dtype=np.float32), (timestamps.size, 1)
        ),
        robot_segment_ids=np.zeros(timestamps.shape, dtype=np.int32),
        pedestrians={"ped-1": pedestrian},
        static_map=None,
        source_file="THOR-Magni_toy-recording.csv",
        dt_s=0.2,
    )


def test_build_snippets_normalizes_motion_and_records_statistics():
    from src.datasets.snippet_library import build_snippet_library

    result = build_snippet_library(
        [_recording()],
        split="train",
        duration_s=3.0,
        stride_s=1.0,
        min_mean_speed_mps=0.3,
        max_mean_speed_mps=2.0,
        max_acceleration_mps2=2.5,
    )

    assert len(result.snippets) == 4
    snippet = result.snippets[0]
    assert snippet.split == "train"
    assert snippet.source_recording_id == "toy-recording"
    assert snippet.participant_id == "ped-1"
    assert snippet.positions.shape == (16, 2)
    assert snippet.positions.dtype == np.float32
    assert snippet.velocities.shape == (16, 2)
    assert snippet.velocities.dtype == np.float32
    assert np.allclose(snippet.positions[0], 0.0, atol=1e-7)
    assert np.allclose(snippet.positions[:, 1], 0.0, atol=1e-6)
    assert snippet.positions[-1, 0] == np.float32(3.0)
    assert snippet.duration_s == 3.0
    assert snippet.mean_speed_mps == pytest.approx(1.0, abs=1e-6)
    assert snippet.max_acceleration_mps2 == pytest.approx(0.0, abs=1e-6)
    assert snippet.mean_abs_curvature_per_m == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(snippet.positions).all()
    assert result.summary["accepted_count"] == 4
    assert result.summary["rejected_count"] == 0


def test_snippet_library_round_trip_uses_numeric_npz(tmp_path):
    from src.datasets.snippet_library import (
        build_snippet_library,
        load_snippet_library,
        save_snippet_library,
    )

    source = build_snippet_library([_recording()], split="train")
    path = save_snippet_library(source, tmp_path / "snippets.npz")
    restored = load_snippet_library(path)

    with np.load(path, allow_pickle=False) as payload:
        assert all(payload[key].dtype != object for key in payload.files)
    assert restored.summary == source.summary
    assert len(restored.snippets) == len(source.snippets)
    assert np.array_equal(
        restored.snippets[0].positions, source.snippets[0].positions
    )


def test_source_overlap_audit_detects_cross_split_recording_reuse():
    from src.datasets.snippet_library import (
        audit_snippet_source_overlap,
        build_snippet_library,
    )

    train = build_snippet_library([_recording()], split="train")
    test = build_snippet_library([_recording()], split="test")

    report = audit_snippet_source_overlap([train, test])

    assert report["total_overlap_count"] > 0
    assert report["fields"]["recording"]["overlaps"] == [
        {"value": "toy-recording", "splits": ["test", "train"]}
    ]


def test_snippets_with_excessive_acceleration_are_rejected():
    from src.datasets.snippet_library import build_snippet_library

    recording = _recording()
    track = recording.pedestrians["ped-1"]
    positions = track.positions.copy()
    positions[8:, 0] += 1.0
    accelerated_track = replace(track, positions=positions)
    accelerated_recording = replace(
        recording, pedestrians={"ped-1": accelerated_track}
    )

    result = build_snippet_library(
        [accelerated_recording],
        split="train",
        max_acceleration_mps2=2.5,
    )

    assert result.summary["rejection_reasons"]["acceleration"] > 0
    assert result.summary["accepted_count"] < result.summary["candidate_count"]


def test_build_snippet_library_cli_writes_split_artifacts(tmp_path):
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
    library = load_snippet_library(
        tmp_path / "snippets/train/snippet_library.npz"
    )
    assert len(library.snippets) == 4
    report = json.loads(
        (tmp_path / "snippets/train/source_overlap_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "ok"
    assert "accepted_count=4" in completed.stdout
