"""Tests for split- and type-isolated dynamic-object motion snippets."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.datasets.thor_adapter import DynamicObjectTrack, RecordingIndex


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
        stride_s=1.0,
        min_mean_speed_mps=0.30,
        max_mean_speed_mps=2.00,
        max_acceleration_mps2=2.50,
        split_provenance=_split_provenance(),
    )


def test_build_motion_snippets_normalizes_pose_and_preserves_type_geometry():
    library = _build_human_library()

    assert library.object_type == "human"
    assert len(library.snippets) == 2
    snippet = library.snippets[0]
    assert snippet.split == "train"
    assert snippet.source_object_id == "toy-recording::Helmet_1"
    assert snippet.source_session_id == "toy-session"
    assert snippet.object_type == "human"
    assert snippet.footprint == {"kind": "circle", "radius_m": 0.30}
    assert snippet.positions.shape == (23, 2)
    assert snippet.positions.dtype == np.float32
    assert snippet.velocities.dtype == np.float32
    assert snippet.headings.shape == (23,)
    assert snippet.headings.dtype == np.float32
    assert np.allclose(snippet.positions[0], 0.0, atol=1e-7)
    assert np.allclose(snippet.positions[:, 1], 0.0, atol=1e-6)
    assert np.allclose(snippet.headings, 0.0, atol=1e-6)
    assert snippet.positions[7, 0] == pytest.approx(1.4, abs=1e-6)
    assert snippet.positions[-1, 0] == pytest.approx(4.4, abs=1e-6)
    assert snippet.positions[-1, 0] - snippet.positions[7, 0] == pytest.approx(
        3.0, abs=1e-6
    )
    assert snippet.mean_speed_mps == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(snippet.positions).all()
    assert library.summary["accepted_count"] == 2
    assert library.summary["motion_snippet_layout_version"] == (
        "history8_current7_future15_v1"
    )
    assert library.summary["sample_count"] == 23
    assert library.summary["history_steps"] == 8
    assert library.summary["future_steps"] == 15
    assert library.summary["current_index"] == 7
    assert library.summary["sample_dt_s"] == pytest.approx(0.2)
    assert library.summary["duration_s"] == pytest.approx(4.4)
    assert library.summary["candidate_count"] == (
        library.summary["accepted_count"] + library.summary["rejected_count"]
    )


def test_parallel_snippet_library_matches_serial():
    from src.datasets.snippet_library import build_snippet_library

    recordings = [_recording("parallel-a"), _recording("parallel-b")]
    kwargs = {
        "split": "train",
        "object_type": "human",
        "stride_s": 1.0,
        "min_mean_speed_mps": 0.30,
        "max_mean_speed_mps": 2.00,
        "max_acceleration_mps2": 2.50,
        "split_provenance": _split_provenance(),
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
            split_provenance=_split_provenance(),
        )


def test_snippet_library_rejects_legacy_three_second_duration():
    from src.datasets.snippet_library import build_snippet_library
    from src.datasets.thor_adapter import ThorDataError

    with pytest.raises(ThorDataError, match="frozen MotionSnippet layout"):
        build_snippet_library(
            [_recording()],
            split="train",
            object_type="human",
            duration_s=3.0,
            split_provenance=_split_provenance(),
        )


def test_short_track_is_rejected_without_extrapolation():
    from src.datasets.snippet_library import build_snippet_library

    recording = _recording()
    object_id = "toy-recording::Helmet_1"
    track = recording.dynamic_objects[object_id]
    keep = np.arange(22)
    short_track = replace(
        track,
        timestamps=track.timestamps[keep],
        poses=track.poses[keep],
        velocities=track.velocities[keep],
        segment_ids=np.zeros(keep.shape, dtype=np.int32),
    )
    short_recording = replace(
        recording,
        dynamic_objects={object_id: short_track},
    )

    library = build_snippet_library(
        [short_recording],
        split="train",
        object_type="human",
        split_provenance=_split_provenance(),
    )

    assert library.snippets == ()
    assert library.summary["candidate_count"] == 1
    assert library.summary["accepted_count"] == 0
    assert library.summary["rejected_count"] == 1
    assert library.summary["rejection_reasons"] == {
        "acceleration": 0,
        "insufficient_contiguous_duration": 1,
        "robot_overlap": 0,
        "speed": 0,
        "stationary": 0,
        "time_grid": 0,
    }


def test_snippets_never_cross_recorded_time_gap():
    from src.datasets.snippet_library import build_snippet_library

    recording = _recording()
    object_id = "toy-recording::Helmet_1"
    track = recording.dynamic_objects[object_id]
    keep = np.concatenate((np.arange(11), np.arange(15, track.timestamps.size)))
    segment_ids = np.concatenate(
        (
            np.zeros(11, dtype=np.int32),
            np.ones(track.timestamps.size - 15, dtype=np.int32),
        )
    )
    gapped_track = replace(
        track,
        timestamps=track.timestamps[keep],
        poses=track.poses[keep],
        velocities=track.velocities[keep],
        segment_ids=segment_ids,
    )
    gapped_recording = replace(
        recording,
        dynamic_objects={object_id: gapped_track},
    )

    library = build_snippet_library(
        [gapped_recording],
        split="train",
        object_type="human",
        split_provenance=_split_provenance(),
    )

    assert library.snippets == ()
    assert library.summary["candidate_count"] == 2
    assert library.summary["rejection_reasons"][
        "insufficient_contiguous_duration"
    ] == 2


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
        metadata = json.loads(str(payload["meta_json"]))
    assert restored.object_type == "human"
    assert restored.summary == source.summary
    assert restored.split_provenance == _split_provenance()
    assert np.array_equal(
        restored.snippets[0].headings, source.snippets[0].headings
    )
    assert restored.snippets[0].provenance == source.snippets[0].provenance
    assert len(source.summary["array_sha256"]) == 64
    for key, expected in {
        "motion_snippet_layout_version": "history8_current7_future15_v1",
        "sample_count": 23,
        "history_steps": 8,
        "future_steps": 15,
        "current_index": 7,
        "sample_dt_s": 0.2,
        "duration_s": 4.4,
    }.items():
        assert metadata[key] == expected


def test_loader_rejects_missing_or_legacy_layout_metadata(tmp_path):
    from src.datasets.snippet_library import (
        load_snippet_library,
        save_snippet_library,
    )
    from src.datasets.thor_adapter import ThorDataError

    source_path = save_snippet_library(
        _build_human_library(), tmp_path / "source.npz"
    )
    with np.load(source_path, allow_pickle=False) as payload:
        arrays = {
            "positions": payload["positions"].copy(),
            "velocities": payload["velocities"].copy(),
            "headings": payload["headings"].copy(),
        }
        metadata = json.loads(str(payload["meta_json"]))
    layout_fields = (
        "motion_snippet_layout_version",
        "sample_count",
        "history_steps",
        "future_steps",
        "current_index",
        "sample_dt_s",
        "duration_s",
    )

    missing = dict(metadata)
    for field in layout_fields:
        missing.pop(field, None)
    missing_path = tmp_path / "missing-layout.npz"
    np.savez(
        missing_path,
        **arrays,
        meta_json=np.asarray(json.dumps(missing, sort_keys=True)),
    )

    legacy = json.loads(json.dumps(metadata))
    legacy["motion_snippet_layout_version"] = "legacy_current_future16_v0"
    legacy["sample_count"] = 16
    legacy["history_steps"] = 1
    legacy["future_steps"] = 15
    legacy["current_index"] = 0
    legacy["duration_s"] = 3.0
    legacy["summary"].update(
        {
            "motion_snippet_layout_version": "legacy_current_future16_v0",
            "sample_count": 16,
            "history_steps": 1,
            "future_steps": 15,
            "current_index": 0,
            "duration_s": 3.0,
        }
    )
    for row in legacy["snippets"]:
        row["duration_s"] = 3.0
    legacy_path = tmp_path / "legacy-layout.npz"
    np.savez(
        legacy_path,
        positions=arrays["positions"][:, :16],
        velocities=arrays["velocities"][:, :16],
        headings=arrays["headings"][:, :16],
        meta_json=np.asarray(json.dumps(legacy, sort_keys=True)),
    )

    for path in (missing_path, legacy_path):
        with pytest.raises(ThorDataError, match="MotionSnippet layout"):
            load_snippet_library(path)


def test_empty_library_keeps_frozen_numeric_array_shapes(tmp_path):
    from src.datasets.snippet_library import (
        build_snippet_library,
        save_snippet_library,
    )

    recording = _recording()
    stationary_id = "toy-recording::Storage_Unit"
    stationary_only = replace(
        recording,
        dynamic_objects={
            stationary_id: recording.dynamic_objects[stationary_id]
        },
    )
    library = build_snippet_library(
        [stationary_only],
        split="train",
        object_type="unknown_dynamic",
        split_provenance=_split_provenance(),
    )
    path = save_snippet_library(library, tmp_path / "empty.npz")

    with np.load(path, allow_pickle=False) as payload:
        assert payload["positions"].shape == (0, 23, 2)
        assert payload["velocities"].shape == (0, 23, 2)
        assert payload["headings"].shape == (0, 23)
        assert payload["positions"].dtype == np.float32


def test_repeated_snippet_artifacts_keep_ids_manifest_and_array_digest(tmp_path):
    from src.datasets.snippet_library import (
        audit_snippet_source_overlap,
        write_snippet_artifacts,
    )

    first_library = _build_human_library()
    second_library = _build_human_library()
    first_paths = write_snippet_artifacts(
        first_library,
        tmp_path / "first",
        overlap_report=audit_snippet_source_overlap([first_library]),
    )
    second_paths = write_snippet_artifacts(
        second_library,
        tmp_path / "second",
        overlap_report=audit_snippet_source_overlap([second_library]),
    )

    assert [item.snippet_id for item in first_library.snippets] == [
        item.snippet_id for item in second_library.snippets
    ]
    assert first_paths["manifest"].read_bytes() == second_paths[
        "manifest"
    ].read_bytes()
    assert first_library.summary["array_sha256"] == second_library.summary[
        "array_sha256"
    ]


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


def test_source_overlap_audit_reports_allowed_session_overlap():
    from src.datasets.snippet_library import audit_snippet_source_overlap
    from src.datasets.split_manager import SplitAuditPolicy

    train = _build_human_library(_recording("recording-a"), split="train")
    test = _build_human_library(_recording("recording-b"), split="test")
    policy = SplitAuditPolicy(
        evaluation_scope="unseen_recording_within_known_sessions",
        required_fields=("recording", "session"),
        allowed_overlap_fields=("session",),
        unavailable_fields=("participant",),
    )

    report = audit_snippet_source_overlap([train, test], policy=policy)

    assert report["status"] == "ok"
    assert report["fields"]["recording"]["overlap_count"] == 0
    assert report["fields"]["session"]["overlaps"] == [
        {"value": "toy-session", "splits": ["test", "train"]}
    ]
    assert report["fields"]["object"]["overlap_count"] == 0
    assert report["allowed_overlap_count"] == 1
    assert report["disallowed_overlap_count"] == 0


def test_type_specific_speed_threshold_keeps_slow_carried_object():
    from src.datasets.snippet_library import build_snippet_library

    library = build_snippet_library(
        [_recording()],
        split="train",
        object_type="carried_object",
        min_mean_speed_mps=0.05,
        max_mean_speed_mps=2.00,
        max_acceleration_mps2=2.50,
        split_provenance=_split_provenance(),
    )

    assert len(library.snippets) == 2
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
        split_provenance=_split_provenance(),
    )

    assert len(library.snippets) == 2
    assert library.summary["rejection_reasons"]["stationary"] == 2


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
        split_provenance=_split_provenance(),
    )

    assert library.summary["accepted_count"] == 0
    assert library.summary["rejection_reasons"]["robot_overlap"] == 2


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
        split_provenance=_split_provenance(),
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
    repeated_output = tmp_path / "repeated-snippets"
    repeated = subprocess.run(
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
            str(repeated_output),
        ],
        cwd=root,
        env=dict(os.environ, PYTHONHASHSEED="987654"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert repeated.returncode == 0, repeated.stderr
    legacy = subprocess.run(
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
            str(tmp_path / "legacy-snippets"),
            "--duration-s",
            "3.0",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert legacy.returncode != 0
    assert "frozen MotionSnippet layout" in legacy.stderr
    expected_counts = {"human": 2, "carried_object": 2, "unknown_dynamic": 2}
    for object_type, count in expected_counts.items():
        path = tmp_path / f"snippets/train/{object_type}/snippet_library.npz"
        library = load_snippet_library(path)
        assert library.object_type == object_type
        assert library.split_provenance == _split_provenance()
        assert len(library.snippets) == count
        type_root = path.parent
        summary = json.loads(
            (type_root / "summary.json").read_text(encoding="utf-8")
        )
        rows = [
            json.loads(line)
            for line in (type_root / "source_manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert summary["split_provenance"] == _split_provenance()
        assert all(
            row["split_provenance"] == _split_provenance() for row in rows
        )
        expected_layout = {
            "motion_snippet_layout_version": "history8_current7_future15_v1",
            "sample_count": 23,
            "history_steps": 8,
            "future_steps": 15,
            "current_index": 7,
            "sample_dt_s": 0.2,
            "duration_s": 4.4,
        }
        assert all(summary[key] == value for key, value in expected_layout.items())
        assert all(
            all(row[key] == value for key, value in expected_layout.items())
            for row in rows
        )
        repeated_root = repeated_output / f"train/{object_type}"
        assert (type_root / "source_manifest.jsonl").read_bytes() == (
            repeated_root / "source_manifest.jsonl"
        ).read_bytes()
        repeated_summary = json.loads(
            (repeated_root / "summary.json").read_text(encoding="utf-8")
        )
        assert repeated_summary["array_sha256"] == summary["array_sha256"]
    report = json.loads(
        (
            tmp_path
            / "snippets/train/human/source_overlap_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "ok"
    assert "total_accepted_count=6" in completed.stdout
    assert "workers_requested=8" in completed.stdout
    assert "workers_used=1" in completed.stdout
    assert "detected_source_overlap_count=0" in completed.stdout
    assert "allowed_session_overlap_count=0" in completed.stdout
    assert "disallowed_source_overlap_count=0" in completed.stdout
