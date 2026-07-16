"""Tests for the dependency-free THÖR-MAGNI dynamic-object adapter."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


_BODIES = (
    "DARKO_Robot",
    "Helmet_1",
    "Helmet_6",
    "LO1",
    "Storage_Cart",
    "Mystery_Rig",
)
_ROLES = (
    "Differential-Autonomous HRI Multimodal",
    "Visitors-Alone",
    "Carrier-Bucket",
    "Carried",
    "Carrier-Storage Bin HRI",
    "Equipment",
)


def _write_toy_thor_csv(
    path: Path,
    *,
    gap_after_s: float | None = None,
    sample_count: int = 21,
) -> Path:
    marker_names = tuple(f"LO1 - {index}" for index in range(1, 5))
    header = ["Frame", "Time"]
    for marker in marker_names:
        header.extend((f"{marker} X", f"{marker} Y", f"{marker} Z"))
    for body in _BODIES:
        header.extend(
            (
                f"{body} Centroid_X",
                f"{body} Centroid_Y",
                f"{body} Centroid_Z",
                *(f"{body} R{index}" for index in range(9)),
            )
        )

    marker_offsets_m = (
        (-0.40, -0.10),
        (0.40, -0.10),
        (0.40, 0.10),
        (-0.40, 0.10),
    )
    lo_yaw = math.pi / 6.0
    lo_cosine, lo_sine = math.cos(lo_yaw), math.sin(lo_yaw)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["BODY_NAMES", *_BODIES])
        writer.writerow(["BODY_ROLES", *_ROLES])
        writer.writerow(["N_FRAMES_QTM", str(sample_count)])
        writer.writerow(["MARKER_NAMES", *marker_names])
        writer.writerow(header)
        for raw_index in range(sample_count):
            timestamp = raw_index * 0.1
            if gap_after_s is not None and timestamp > gap_after_s:
                timestamp += 0.5
            positions_m = {
                "DARKO_Robot": (timestamp, 0.0),
                "Helmet_1": (2.0, 0.5 * timestamp),
                "Helmet_6": (2.5, 0.4 * timestamp),
                "LO1": (3.0 + 0.2 * timestamp, 1.5),
                "Storage_Cart": (4.0, 2.0),
                "Mystery_Rig": (5.0, 0.25 * timestamp),
            }
            row: dict[str, object] = {"Frame": raw_index, "Time": f"{timestamp:.3f}"}
            lo_x, lo_y = positions_m["LO1"]
            for marker, (dx, dy) in zip(marker_names, marker_offsets_m):
                world_dx = lo_cosine * dx - lo_sine * dy
                world_dy = lo_sine * dx + lo_cosine * dy
                row[f"{marker} X"] = 1000.0 * (lo_x + world_dx)
                row[f"{marker} Y"] = 1000.0 * (lo_y + world_dy)
                row[f"{marker} Z"] = 0.0
            for body, (x_m, y_m) in positions_m.items():
                row[f"{body} Centroid_X"] = 1000.0 * x_m
                row[f"{body} Centroid_Y"] = 1000.0 * y_m
                row[f"{body} Centroid_Z"] = 0.0
                rotation = (
                    (lo_cosine, lo_sine, 0.0, -lo_sine, lo_cosine, 0.0, 0.0, 0.0, 1.0)
                    if body == "LO1"
                    else (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
                )
                for index, value in enumerate(rotation):
                    row[f"{body} R{index}"] = "" if body == "Mystery_Rig" else value
            writer.writerow([row.get(column, "") for column in header])
    return path


def test_load_recording_retains_and_classifies_all_non_robot_bodies(tmp_path):
    from src.datasets.thor_adapter import load_thor_recording

    recording = load_thor_recording(
        _write_toy_thor_csv(tmp_path / "THOR-Magni_toy.csv"),
        dt_s=0.2,
        max_gap_s=0.3,
    )

    assert recording.recording_id == "toy"
    assert recording.timestamps.dtype == np.float64
    assert recording.robot_pose.dtype == np.float32
    assert recording.robot_twist.dtype == np.float32
    assert set(recording.dynamic_objects) == {
        "toy::Helmet_1",
        "toy::Helmet_6",
        "toy::LO1",
        "toy::Storage_Cart",
        "toy::Mystery_Rig",
    }
    assert recording.dynamic_objects["toy::Helmet_1"].object_type == "human"
    assert recording.dynamic_objects["toy::Helmet_6"].object_type == "human"
    assert recording.dynamic_objects["toy::LO1"].object_type == "carried_object"
    assert recording.dynamic_objects["toy::Storage_Cart"].object_type == "carried_object"
    assert recording.dynamic_objects["toy::Mystery_Rig"].object_type == "unknown_dynamic"
    stationary = recording.dynamic_objects["toy::Storage_Cart"]
    assert np.allclose(stationary.velocities, 0.0)
    assert recording.resampling_report["indexed_dynamic_object_count"] == 5


def test_marker_geometry_and_human_sizes_follow_frozen_policy(tmp_path):
    from src.datasets.thor_adapter import load_thor_recording

    recording = load_thor_recording(
        _write_toy_thor_csv(tmp_path / "THOR-Magni_geometry.csv")
    )

    ordinary = recording.dynamic_objects["geometry::Helmet_1"]
    carrier = recording.dynamic_objects["geometry::Helmet_6"]
    carried = recording.dynamic_objects["geometry::LO1"]
    unknown = recording.dynamic_objects["geometry::Mystery_Rig"]
    assert ordinary.footprint == {"kind": "circle", "radius_m": 0.30}
    assert carrier.footprint == {"kind": "circle", "radius_m": 0.45}
    assert carried.footprint["kind"] == "rectangle"
    assert carried.footprint["length_m"] == pytest.approx(0.80, abs=1e-6)
    assert carried.footprint["width_m"] == pytest.approx(0.20, abs=1e-6)
    assert np.allclose(carried.poses[:, 2], math.pi / 6.0, atol=1e-6)
    assert carried.provenance["geometry_source"] == "qtm_marker_p95"
    assert unknown.footprint == {"kind": "circle", "radius_m": 0.50}
    assert unknown.provenance["geometry_source"] == "config_fallback"


def test_missing_qtm_orientation_uses_motion_heading_fallback(tmp_path):
    from src.datasets.thor_adapter import load_thor_recording

    recording = load_thor_recording(
        _write_toy_thor_csv(tmp_path / "THOR-Magni_heading.csv")
    )
    track = recording.dynamic_objects["heading::Mystery_Rig"]

    assert np.allclose(track.poses[:, 2], math.pi / 2.0, atol=1e-5)
    assert track.provenance["orientation_source"] == "motion_fallback"


def test_recording_index_round_trip_uses_no_pickle(tmp_path):
    from src.datasets.thor_adapter import (
        load_recording_index,
        load_thor_recording,
        save_recording_index,
    )

    source = load_thor_recording(
        _write_toy_thor_csv(tmp_path / "THOR-Magni_roundtrip.csv")
    )
    path = save_recording_index(source, tmp_path / "recording.npz")
    restored = load_recording_index(path)

    with np.load(path, allow_pickle=False) as payload:
        assert all(payload[key].dtype != object for key in payload.files)
    assert restored.dynamic_objects.keys() == source.dynamic_objects.keys()
    for object_id in source.dynamic_objects:
        expected = source.dynamic_objects[object_id]
        actual = restored.dynamic_objects[object_id]
        assert np.array_equal(actual.poses, expected.poses)
        assert actual.poses.dtype == np.float32
        assert actual.footprint == expected.footprint
        assert actual.provenance == expected.provenance


def test_validate_recording_index_rejects_nonfinite_object_pose(tmp_path):
    from src.datasets.thor_adapter import (
        ThorDataError,
        load_thor_recording,
        validate_recording_index,
    )

    recording = load_thor_recording(
        _write_toy_thor_csv(tmp_path / "THOR-Magni_invalid.csv")
    )
    object_id = "invalid::Helmet_1"
    track = recording.dynamic_objects[object_id]
    bad_poses = track.poses.copy()
    bad_poses[0, 0] = np.nan
    broken = replace(
        recording,
        dynamic_objects={
            **recording.dynamic_objects,
            object_id: replace(track, poses=bad_poses),
        },
    )

    with pytest.raises(ThorDataError, match="NaN/Inf"):
        validate_recording_index(broken)


def test_resampling_keeps_large_time_gaps_in_separate_segments(tmp_path):
    from src.datasets.thor_adapter import load_thor_recording

    recording = load_thor_recording(
        _write_toy_thor_csv(
            tmp_path / "THOR-Magni_gap.csv", gap_after_s=1.0
        ),
        dt_s=0.2,
        max_gap_s=0.3,
    )

    assert set(recording.robot_segment_ids.tolist()) == {0, 1}
    assert not np.any(
        (recording.timestamps > 1.0) & (recording.timestamps < 1.6)
    )
    track = recording.dynamic_objects["gap::Helmet_1"]
    assert set(track.segment_ids.tolist()) == {0, 1}
    assert recording.resampling_report["robot_speed_p50_delta_mps"] < 1e-6
    assert recording.resampling_report["dynamic_object_speed_p50_delta_mps"] < 1e-6


def test_index_recordings_cli_selects_only_requested_split(tmp_path):
    from src.datasets.thor_adapter import load_recording_indexes_from_dir

    train_a_csv = _write_toy_thor_csv(tmp_path / "THOR-Magni_train_a.csv")
    train_b_csv = _write_toy_thor_csv(tmp_path / "THOR-Magni_train_b.csv")
    test_csv = _write_toy_thor_csv(tmp_path / "THOR-Magni_test.csv")
    split_manifest = tmp_path / "split_manifest.jsonl"
    split_manifest.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "recording_id": "train_a",
                    "session_id": "session-train-a",
                    "participant_ids": ["Helmet_1"],
                    "source_path": str(train_a_csv),
                    "split": "train",
                },
                {
                    "recording_id": "train_b",
                    "session_id": "session-train-b",
                    "participant_ids": ["Helmet_1"],
                    "source_path": str(train_b_csv),
                    "split": "train",
                },
                {
                    "recording_id": "test",
                    "session_id": "session-test",
                    "participant_ids": ["Helmet_1"],
                    "source_path": str(test_csv),
                    "split": "test",
                },
            )
        ),
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/01_index_recordings.py"),
            "--config",
            str(root / "configs/data_thor.yaml"),
            "--split",
            "train",
            "--split-manifest",
            str(split_manifest),
            "--raw-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "indexes"),
            "--workers",
            "2",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    restored = load_recording_indexes_from_dir(
        tmp_path / "indexes/train", expected_split="train"
    )
    assert [recording.recording_id for recording in restored] == [
        "train_a",
        "train_b",
    ]
    assert all(len(recording.dynamic_objects) == 5 for recording in restored)
    assert "dynamic_object_track_count=10" in completed.stdout
    assert "workers_requested=2" in completed.stdout
    assert "workers_used=2" in completed.stdout


def test_index_recordings_cli_rejects_nonpositive_workers():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/01_index_recordings.py"),
            "--config",
            str(root / "configs/data_thor.yaml"),
            "--split",
            "train",
            "--workers",
            "0",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "positive integer" in completed.stderr
