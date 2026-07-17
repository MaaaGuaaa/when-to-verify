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


def _split_provenance() -> dict[str, object]:
    return {
        "split_manifest_digest": "0123456789abcdef0123456789abcdef",
        "evaluation_scope": "unseen_recording_within_known_sessions",
        "grouping_unit": "recording_id",
        "field_policies": {
            "recording": "forbidden",
            "session": "allowed_reported",
            "participant": "unavailable",
        },
    }


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
    duplicate_qtm_index: int | None = None,
    duplicate_qtm_conflict: bool = False,
    auxiliary_tail_row_count: int = 0,
    qtm_geometry_without_time_tail: bool = False,
    nonincreasing_qtm_frame_index: int | None = None,
    malformed_auxiliary_time_tail: bool = False,
    curved_motion: bool = False,
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
    auxiliary_column = "Helmet_1 TB2_Accelerometer_X"
    header.append(auxiliary_column)

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
            robot_y = 0.25 * timestamp**2 if curved_motion else 0.0
            helmet_1_x = 2.0 + timestamp if curved_motion else 2.0
            helmet_1_y = (
                0.25 * timestamp**2
                if curved_motion
                else 0.5 * timestamp
            )
            positions_m = {
                "DARKO_Robot": (timestamp, robot_y),
                "Helmet_1": (helmet_1_x, helmet_1_y),
                "Helmet_6": (2.5, 0.4 * timestamp),
                "LO1": (3.0 + 0.2 * timestamp, 1.5),
                "Storage_Cart": (4.0, 2.0),
                "Mystery_Rig": (5.0, 0.25 * timestamp),
            }
            row: dict[str, object] = {"Frame": raw_index, "Time": f"{timestamp:.3f}"}
            if raw_index == nonincreasing_qtm_frame_index:
                row["Frame"] = raw_index - 1
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
            row[auxiliary_column] = float(raw_index)
            writer.writerow([row.get(column, "") for column in header])
            if raw_index == duplicate_qtm_index:
                duplicate = dict(row)
                duplicate[auxiliary_column] = float(raw_index) + 0.5
                if duplicate_qtm_conflict:
                    duplicate["DARKO_Robot Centroid_X"] = (
                        float(duplicate["DARKO_Robot Centroid_X"]) + 1.0
                    )
                writer.writerow([duplicate.get(column, "") for column in header])
        for _ in range(auxiliary_tail_row_count):
            writer.writerow(
                ["N/A" if column == auxiliary_column else "" for column in header]
            )
        if qtm_geometry_without_time_tail:
            writer.writerow(
                [
                    "1000.0" if column == "DARKO_Robot Centroid_X" else ""
                    for column in header
                ]
            )
        if malformed_auxiliary_time_tail:
            writer.writerow(
                [
                    "not-a-time"
                    if column == "Time"
                    else "N/A"
                    if column == auxiliary_column
                    else ""
                    for column in header
                ]
            )
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


def test_duplicate_qtm_rows_with_identical_geometry_are_collapsed(tmp_path):
    from src.datasets.thor_adapter import load_thor_recording

    baseline = load_thor_recording(
        _write_toy_thor_csv(tmp_path / "THOR-Magni_baseline.csv")
    )
    duplicated = load_thor_recording(
        _write_toy_thor_csv(
            tmp_path / "THOR-Magni_duplicated.csv",
            duplicate_qtm_index=10,
        )
    )

    assert duplicated.resampling_report["raw_csv_row_count"] == 22
    assert duplicated.resampling_report["canonical_qtm_frame_count"] == 21
    assert duplicated.resampling_report["collapsed_duplicate_qtm_row_count"] == 1
    assert duplicated.resampling_report["ignored_auxiliary_row_count"] == 0
    assert duplicated.resampling_report["conflicting_duplicate_qtm_row_count"] == 0
    assert np.array_equal(duplicated.timestamps, baseline.timestamps)
    assert np.array_equal(duplicated.robot_pose, baseline.robot_pose)
    assert np.array_equal(duplicated.robot_twist, baseline.robot_twist)
    baseline_tracks = {
        track.source_body_name: track
        for track in baseline.dynamic_objects.values()
    }
    duplicated_tracks = {
        track.source_body_name: track
        for track in duplicated.dynamic_objects.values()
    }
    assert duplicated_tracks.keys() == baseline_tracks.keys()
    for body_name in baseline_tracks:
        actual = duplicated_tracks[body_name]
        expected = baseline_tracks[body_name]
        assert np.array_equal(actual.timestamps, expected.timestamps)
        assert np.array_equal(actual.poses, expected.poses)
        assert np.array_equal(actual.velocities, expected.velocities)
        assert np.array_equal(actual.segment_ids, expected.segment_ids)
        assert actual.footprint == expected.footprint
        assert actual.provenance == expected.provenance


def test_duplicate_qtm_rows_with_conflicting_geometry_are_rejected(tmp_path):
    from src.datasets.thor_adapter import ThorDataError, load_thor_recording

    path = _write_toy_thor_csv(
        tmp_path / "THOR-Magni_conflict.csv",
        duplicate_qtm_index=10,
        duplicate_qtm_conflict=True,
    )

    with pytest.raises(
        ThorDataError,
        match=(
            r"conflicting duplicate QTM frame 10.*"
            r"DARKO_Robot Centroid_X"
        ),
    ):
        load_thor_recording(path)


def test_rows_without_qtm_frame_time_or_geometry_are_ignored(tmp_path):
    from src.datasets.thor_adapter import load_thor_recording

    baseline = load_thor_recording(
        _write_toy_thor_csv(tmp_path / "THOR-Magni_aux_baseline.csv")
    )
    with_auxiliary_tail = load_thor_recording(
        _write_toy_thor_csv(
            tmp_path / "THOR-Magni_aux_tail.csv",
            auxiliary_tail_row_count=2,
        )
    )

    assert with_auxiliary_tail.resampling_report["raw_csv_row_count"] == 23
    assert with_auxiliary_tail.resampling_report["canonical_qtm_frame_count"] == 21
    assert with_auxiliary_tail.resampling_report["ignored_auxiliary_row_count"] == 2
    assert np.array_equal(with_auxiliary_tail.timestamps, baseline.timestamps)
    assert np.array_equal(with_auxiliary_tail.robot_pose, baseline.robot_pose)
    assert np.array_equal(with_auxiliary_tail.robot_twist, baseline.robot_twist)


def test_qtm_geometry_without_frame_and_time_is_rejected(tmp_path):
    from src.datasets.thor_adapter import ThorDataError, load_thor_recording

    path = _write_toy_thor_csv(
        tmp_path / "THOR-Magni_missing_time.csv",
        qtm_geometry_without_time_tail=True,
    )

    with pytest.raises(
        ThorDataError,
        match=(
            r"QTM data but missing Frame/Time.*"
            r"DARKO_Robot Centroid_X"
        ),
    ):
        load_thor_recording(path)


def test_qtm_frame_must_increase_after_canonicalization(tmp_path):
    from src.datasets.thor_adapter import ThorDataError, load_thor_recording

    path = _write_toy_thor_csv(
        tmp_path / "THOR-Magni_frame_order.csv",
        nonincreasing_qtm_frame_index=10,
    )

    with pytest.raises(
        ThorDataError,
        match=r"QTM Frame/Time must strictly increase after canonicalization",
    ):
        load_thor_recording(path)


def test_nonempty_invalid_qtm_frame_or_time_is_rejected(tmp_path):
    from src.datasets.thor_adapter import ThorDataError, load_thor_recording

    path = _write_toy_thor_csv(
        tmp_path / "THOR-Magni_invalid_time.csv",
        malformed_auxiliary_time_tail=True,
    )

    with pytest.raises(
        ThorDataError,
        match=r"invalid or incomplete QTM Frame/Time",
    ):
        load_thor_recording(path)


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


def test_recording_artifacts_stamp_validated_split_provenance(tmp_path):
    from src.datasets.thor_adapter import (
        load_thor_recording,
        write_recording_indexes,
    )

    recording = load_thor_recording(
        _write_toy_thor_csv(tmp_path / "THOR-Magni_provenance.csv")
    )
    paths = write_recording_indexes(
        [recording],
        split="train",
        output_dir=tmp_path / "indexes/train",
        split_provenance=_split_provenance(),
    )

    rows = [
        json.loads(line)
        for line in paths["manifest"].read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert rows[0]["split_provenance"] == _split_provenance()
    assert summary["split_provenance"] == _split_provenance()


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
    assert recording.resampling_report[
        "raw_robot_acceleration_p50_mps2"
    ] < 1e-6
    assert recording.resampling_report[
        "resampled_robot_abs_curvature_p50_per_m"
    ] < 1e-6


def test_resampling_reports_speed_acceleration_and_curvature_quantiles(tmp_path):
    from src.datasets.thor_adapter import load_thor_recording

    recording = load_thor_recording(
        _write_toy_thor_csv(
            tmp_path / "THOR-Magni_curved.csv", curved_motion=True
        ),
        dt_s=0.2,
        max_gap_s=0.3,
    )
    report = recording.resampling_report

    units = {
        "speed": "mps",
        "acceleration": "mps2",
        "abs_curvature": "per_m",
    }
    for scope in ("robot", "dynamic_object"):
        for stage in ("raw", "resampled"):
            for metric, unit in units.items():
                assert report[
                    f"{stage}_{scope}_{metric}_sample_count"
                ] > 0
                for quantile in ("p05", "p50", "p95"):
                    value = report[
                        f"{stage}_{scope}_{metric}_{quantile}_{unit}"
                    ]
                    assert math.isfinite(value)
        for metric, unit in units.items():
            for quantile in ("p05", "p50", "p95"):
                assert math.isfinite(
                    report[
                        f"{scope}_{metric}_{quantile}_delta_{unit}"
                    ]
                )

    assert report["raw_robot_acceleration_p50_mps2"] > 0.1
    assert report["raw_robot_abs_curvature_p95_per_m"] > 0.0
    assert report["raw_dynamic_object_acceleration_p95_mps2"] > 0.1
    assert report["raw_dynamic_object_abs_curvature_p95_per_m"] > 0.0


def test_index_recordings_cli_selects_only_requested_split(tmp_path):
    from src.datasets.split_manager import (
        SplitAuditPolicy,
        freeze_preassigned_split,
        write_split_artifacts,
    )
    from src.datasets.thor_adapter import load_recording_indexes_from_dir

    train_a_csv = _write_toy_thor_csv(tmp_path / "THOR-Magni_train_a.csv")
    train_b_csv = _write_toy_thor_csv(tmp_path / "THOR-Magni_train_b.csv")
    test_csv = _write_toy_thor_csv(tmp_path / "THOR-Magni_test.csv")
    rows = (
        {
            "recording_id": "train_a",
            "session_id": "session-train-a",
            "source_path": str(train_a_csv),
        },
        {
            "recording_id": "train_b",
            "session_id": "session-train-b",
            "source_path": str(train_b_csv),
        },
        {
            "recording_id": "test",
            "session_id": "session-test",
            "source_path": str(test_csv),
        },
    )
    policy = SplitAuditPolicy(
        evaluation_scope="unseen_recording_within_known_sessions",
        required_fields=("recording", "session", "seed_namespace"),
        allowed_overlap_fields=("session",),
        unavailable_fields=("participant",),
    )
    split_result = freeze_preassigned_split(
        rows,
        {"train_a": "train", "train_b": "train", "test": "test"},
        seed=42,
        policy=policy,
    )
    split_paths = write_split_artifacts(split_result, tmp_path / "split")
    split_manifest = split_paths["manifest"]
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
    recording_summary = json.loads(
        (tmp_path / "indexes/train/summary.json").read_text(encoding="utf-8")
    )
    assert recording_summary["split_provenance"] == {
        "split_manifest_digest": split_result.manifest_digest,
        "evaluation_scope": "unseen_recording_within_known_sessions",
        "grouping_unit": "recording_id",
        "field_policies": split_result.summary["field_policies"],
    }
    assert "dynamic_object_track_count=10" in completed.stdout
    assert "workers_requested=2" in completed.stdout
    assert "workers_used=2" in completed.stdout

    split_summary = json.loads(
        split_paths["summary"].read_text(encoding="utf-8")
    )
    split_summary["manifest_digest"] = "0" * 32
    split_paths["summary"].write_text(
        json.dumps(split_summary, sort_keys=True) + "\n", encoding="utf-8"
    )
    stale = subprocess.run(
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
            str(tmp_path / "stale-indexes"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert stale.returncode != 0
    assert "split manifest digest does not match" in stale.stderr


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
