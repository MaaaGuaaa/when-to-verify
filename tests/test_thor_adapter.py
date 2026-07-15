"""Tests for the dependency-free THÖR-MAGNI trajectory adapter."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_toy_thor_csv(
    path: Path,
    *,
    gap_after_s: float | None = None,
    sample_count: int = 21,
) -> Path:
    bodies = ("DARKO_Robot", "Helmet_1", "Storage_Cart")
    roles = (
        "Differential-Autonomous HRI Multimodal",
        "Visitors-Alone",
        "Carrier-Storage Bin HRI",
    )
    header = ["Frame", "Time"]
    for body in bodies:
        header.extend(
            [
                f"{body} Centroid_X",
                f"{body} Centroid_Y",
                f"{body} Centroid_Z",
                f"{body} R0",
                f"{body} R1",
            ]
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["BODY_NAMES", *bodies])
        writer.writerow(["BODY_ROLES", *roles])
        writer.writerow(["N_FRAMES_QTM", str(sample_count)])
        writer.writerow(header)
        frame = 0
        for raw_index in range(sample_count):
            timestamp = raw_index * 0.1
            if gap_after_s is not None and timestamp > gap_after_s:
                timestamp += 0.5
            values = [frame, f"{timestamp:.3f}"]
            values.extend([1000.0 * timestamp, 0.0, 0.0, 1.0, 0.0])
            values.extend([2000.0, 500.0 * timestamp, 0.0, 1.0, 0.0])
            values.extend([3000.0, 0.0, 0.0, 1.0, 0.0])
            writer.writerow(values)
            frame += 1
    return path


def test_load_recording_converts_units_filters_roles_and_resamples(tmp_path):
    from src.datasets.thor_adapter import load_thor_recording

    path = _write_toy_thor_csv(tmp_path / "THOR-Magni_toy.csv")

    recording = load_thor_recording(path, dt_s=0.2, max_gap_s=0.3)

    assert recording.recording_id == "toy"
    assert recording.timestamps.dtype == np.float64
    assert recording.timestamps.shape == (11,)
    assert np.allclose(recording.timestamps, np.arange(11) * 0.2)
    assert recording.robot_pose.dtype == np.float32
    assert recording.robot_pose.shape == (11, 3)
    assert np.allclose(recording.robot_pose[:, 0], np.arange(11) * 0.2)
    assert recording.robot_twist.dtype == np.float32
    assert np.allclose(recording.robot_twist[1:-1, 0], 1.0, atol=1e-5)
    assert np.allclose(recording.robot_twist[:, 1], 0.0, atol=1e-6)
    assert set(recording.pedestrians) == {"Helmet_1"}
    pedestrian = recording.pedestrians["Helmet_1"]
    assert pedestrian.positions.dtype == np.float32
    assert pedestrian.positions.shape == (11, 2)
    assert np.allclose(pedestrian.positions[:, 0], 2.0)
    assert np.allclose(pedestrian.velocities[1:-1, 1], 0.5, atol=1e-5)
    assert np.isfinite(recording.robot_pose).all()
    assert np.isfinite(pedestrian.positions).all()


def test_recording_index_round_trip_uses_no_pickle(tmp_path):
    from src.datasets.thor_adapter import (
        load_recording_index,
        load_thor_recording,
        save_recording_index,
    )

    source = load_thor_recording(
        _write_toy_thor_csv(tmp_path / "THOR-Magni_roundtrip.csv"),
        dt_s=0.2,
        max_gap_s=0.3,
    )

    path = save_recording_index(source, tmp_path / "recording.npz")
    restored = load_recording_index(path)

    with np.load(path, allow_pickle=False) as payload:
        assert "meta_json" in payload.files
        assert all(payload[key].dtype != object for key in payload.files)
    assert restored.recording_id == source.recording_id
    assert restored.session_id == source.session_id
    assert np.array_equal(restored.timestamps, source.timestamps)
    assert np.array_equal(restored.robot_pose, source.robot_pose)
    assert np.array_equal(restored.robot_twist, source.robot_twist)
    assert restored.pedestrians.keys() == source.pedestrians.keys()
    restored_pedestrian = restored.pedestrians["Helmet_1"]
    source_pedestrian = source.pedestrians["Helmet_1"]
    assert np.array_equal(restored_pedestrian.positions, source_pedestrian.positions)
    assert restored_pedestrian.positions.dtype == np.float32


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
    pedestrian = recording.pedestrians["Helmet_1"]
    assert set(pedestrian.segment_ids.tolist()) == {0, 1}
    assert recording.resampling_report["robot_speed_p50_delta_mps"] < 1e-6
    assert recording.resampling_report["pedestrian_speed_p50_delta_mps"] < 1e-6


def test_index_recordings_cli_selects_only_requested_split(tmp_path):
    from src.datasets.thor_adapter import load_recording_indexes_from_dir

    train_csv = _write_toy_thor_csv(tmp_path / "THOR-Magni_train.csv")
    test_csv = _write_toy_thor_csv(tmp_path / "THOR-Magni_test.csv")
    split_manifest = tmp_path / "split_manifest.jsonl"
    split_manifest.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "recording_id": "train",
                    "session_id": "session-train",
                    "participant_ids": ["Helmet_1"],
                    "source_path": str(train_csv),
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
    assert [recording.recording_id for recording in restored] == ["train"]
    assert "recording_count=1" in completed.stdout
