"""Tests for the standalone 40-frame THOR dynamic-object snippet library."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.datasets.thor_adapter import DynamicObjectTrack, RecordingIndex, ThorDataError


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


def _recording(*, sample_count: int = 45) -> RecordingIndex:
    timestamps = np.arange(sample_count, dtype=np.float64) * 0.2
    object_id = "toy-recording::Helmet_1"
    poses = np.column_stack(
        (
            2.0 + timestamps,
            np.full_like(timestamps, 2.0),
            np.zeros_like(timestamps),
        )
    ).astype(np.float32)
    track = DynamicObjectTrack(
        object_id=object_id,
        source_body_name="Helmet_1",
        object_type="human",
        raw_role="Visitors-Alone",
        timestamps=timestamps,
        poses=poses,
        velocities=np.tile(np.array([1.0, 0.0], dtype=np.float32), (sample_count, 1)),
        segment_ids=np.zeros(sample_count, dtype=np.int32),
        footprint={"kind": "circle", "radius_m": 0.30},
        provenance={
            "geometry_source": "config_human",
            "orientation_source": "qtm_rotation",
        },
    )
    return RecordingIndex(
        recording_id="toy-recording",
        session_id="toy-session",
        timestamps=timestamps,
        robot_pose=np.zeros((sample_count, 3), dtype=np.float32),
        robot_twist=np.zeros((sample_count, 2), dtype=np.float32),
        robot_segment_ids=np.zeros(sample_count, dtype=np.int32),
        dynamic_objects={object_id: track},
        static_map=None,
        source_file="THOR-Magni_toy-recording.csv",
        dt_s=0.2,
    )


def _build(recording: RecordingIndex | None = None):
    from src.datasets.long_snippet_library import build_long_snippet_library

    return build_long_snippet_library(
        [_recording() if recording is None else recording],
        split="train",
        object_type="human",
        stride_s=0.2,
        min_mean_speed_mps=0.30,
        max_mean_speed_mps=2.00,
        max_acceleration_mps2=2.50,
        split_provenance=_split_provenance(),
    )


def test_long_library_crops_exact_40_frame_source_window():
    library = _build()

    assert library.object_type == "human"
    assert len(library.snippets) == 6
    snippet = min(library.snippets, key=lambda item: item.start_timestamp)
    assert snippet.positions.shape == (40, 2)
    assert snippet.positions.dtype == np.float32
    assert snippet.velocities.shape == (40, 2)
    assert snippet.velocities.dtype == np.float32
    assert snippet.headings.shape == (40,)
    assert snippet.headings.dtype == np.float32
    assert snippet.start_timestamp == pytest.approx(0.0)
    assert np.allclose(snippet.positions[:, 0], np.arange(40) * 0.2)
    assert np.allclose(snippet.positions[:, 1], 0.0)
    assert snippet.footprint == {"kind": "circle", "radius_m": 0.30}
    assert snippet.provenance["track_provenance"]["geometry_source"] == "config_human"
    assert library.summary["sample_count"] == 40
    assert library.summary["duration_s"] == pytest.approx(7.8)
    assert library.summary["accepted_count"] == 6


def test_long_library_fixes_current_index_and_relative_time_range():
    library = _build()
    snippet = library.snippets[0]

    assert library.relative_time_s.shape == (40,)
    assert library.relative_time_s.dtype == np.float64
    assert library.relative_time_s[0] == pytest.approx(-1.4)
    assert library.relative_time_s[7] == pytest.approx(0.0)
    assert library.relative_time_s[-1] == pytest.approx(6.4)
    assert np.allclose(np.diff(library.relative_time_s), 0.2)
    assert snippet.positions[7, 0] == pytest.approx(1.4)
    assert snippet.positions[-1, 0] == pytest.approx(7.8)


def test_long_library_rejects_short_boundary_without_extrapolation():
    library = _build(_recording(sample_count=39))

    assert library.snippets == ()
    assert library.summary["candidate_count"] == 1
    assert library.summary["accepted_count"] == 0
    assert library.summary["rejection_reasons"]["insufficient_contiguous_duration"] == 1


def test_long_library_never_stitches_track_segments():
    recording = _recording(sample_count=40)
    track = recording.dynamic_objects["toy-recording::Helmet_1"]
    segment_ids = np.concatenate(
        (np.zeros(20, dtype=np.int32), np.ones(20, dtype=np.int32))
    )
    segmented = replace(track, segment_ids=segment_ids)
    split_recording = replace(
        recording,
        dynamic_objects={segmented.object_id: segmented},
    )

    library = _build(split_recording)

    assert library.snippets == ()
    assert library.summary["rejection_reasons"]["insufficient_contiguous_duration"] == 2


def test_long_library_is_deterministic_and_binds_source_identity_in_id():
    first = _build()
    second = _build()

    assert first.summary == second.summary
    assert first.relative_time_s.tobytes() == second.relative_time_s.tobytes()
    assert [item.snippet_id for item in first.snippets] == [
        item.snippet_id for item in second.snippets
    ]
    assert all(item.snippet_id.startswith("train-human-long40-") for item in first.snippets)
    assert len({item.snippet_id for item in first.snippets}) == len(first.snippets)


def test_long_loader_round_trip_validates_layout_time_checksum_and_semantics(tmp_path):
    from src.datasets.long_snippet_library import (
        load_long_snippet_library,
        save_long_snippet_library,
    )

    source = _build()
    path = save_long_snippet_library(source, tmp_path / "snippet_library_40.npz")
    restored = load_long_snippet_library(path)

    assert restored.summary == source.summary
    assert np.array_equal(restored.relative_time_s, source.relative_time_s)
    assert np.array_equal(restored.snippets[0].positions, source.snippets[0].positions)
    with np.load(path, allow_pickle=False) as payload:
        assert payload["relative_time_s"].dtype == np.float64
        assert payload["relative_time_s"].shape == (40,)
        metadata = json.loads(str(payload["meta_json"]))
    assert metadata["semantic_digest_sha256"] == source.summary["semantic_digest_sha256"]
    assert metadata["motion_snippet_layout_version"] == "history8_current7_future32_v1"


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("layout", "LongMotionSnippet layout"),
        ("time", "relative time grid"),
        ("array", "array_sha256"),
        ("semantic", "semantic_digest"),
    ],
)
def test_long_loader_rejects_mixed_layout_and_tampering(tmp_path, mutation, error):
    from src.datasets.long_snippet_library import (
        load_long_snippet_library,
        save_long_snippet_library,
    )

    source_path = save_long_snippet_library(
        _build(), tmp_path / "snippet_library_40.npz"
    )
    with np.load(source_path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files if key != "meta_json"}
        metadata = json.loads(str(payload["meta_json"]))
    if mutation == "layout":
        arrays["positions"] = arrays["positions"][:, :23]
        arrays["velocities"] = arrays["velocities"][:, :23]
        arrays["headings"] = arrays["headings"][:, :23]
        arrays["relative_time_s"] = arrays["relative_time_s"][:23]
        metadata["motion_snippet_layout_version"] = "history8_current7_future15_v1"
        metadata["sample_count"] = 23
        metadata["future_steps"] = 15
    elif mutation == "time":
        arrays["relative_time_s"][8] = 0.11
    elif mutation == "array":
        arrays["positions"][0, 0, 0] += 0.1
    else:
        metadata["semantic_digest_sha256"] = "0" * 64
    tampered = tmp_path / f"tampered-{mutation}.npz"
    np.savez(
        tampered,
        **arrays,
        meta_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    with pytest.raises(ThorDataError, match=error):
        load_long_snippet_library(tampered)


def test_long_artifact_loader_checks_sidecars_and_checksum_envelope(tmp_path):
    from src.datasets.long_snippet_library import (
        load_long_snippet_artifact,
        write_long_snippet_artifacts,
    )

    library = _build()
    paths = write_long_snippet_artifacts(
        library,
        tmp_path / "long40",
        overlap_report={"status": "ok", "total_overlap_count": 0},
    )
    restored = load_long_snippet_artifact(paths["directory"])

    assert restored.summary == library.summary
    assert paths["library"].name == "snippet_library_40.npz"
    assert paths["manifest"].name == "source_manifest_40.jsonl"
    assert paths["summary"].name == "summary_40.json"
    assert paths["checksum_manifest"].is_file()
    assert paths["semantic_digest"].is_file()
    paths["summary"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ThorDataError, match="checksum"):
        load_long_snippet_artifact(paths["directory"])


def test_long_snippet_library_cli_recrops_human_from_recording_indexes(tmp_path):
    from src.datasets.long_snippet_library import load_long_snippet_artifact
    from src.datasets.thor_adapter import write_recording_indexes

    source_root = tmp_path / "published-sop03"
    write_recording_indexes(
        [_recording()],
        split="train",
        output_dir=source_root / "recording_indexes/train",
        split_provenance=_split_provenance(),
    )
    root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "long40"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/02_build_long_snippet_library.py"),
            "--recording-root",
            str(source_root),
            "--output-dir",
            str(output_root),
            "--split",
            "train",
            "--stride-s",
            "0.2",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = output_root / "train/human"
    loaded = load_long_snippet_artifact(artifact)
    assert len(loaded.snippets) == 6
    assert "accepted_count[train/human]=6" in completed.stdout
