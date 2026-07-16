"""Tests for dynamic-object BaseState and isolated OracleContext extraction."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.contracts import build_grid_spec, validate_base_state, validate_oracle_context
from src.datasets.thor_adapter import DynamicObjectTrack, RecordingIndex
from src.utils.config import load_config


def _recording(
    duration_s: float = 6.0,
    *,
    recording_id: str = "toy-recording",
) -> RecordingIndex:
    timestamps = np.arange(0.0, duration_s + 1e-9, 0.2, dtype=np.float64)
    robot_pose = np.column_stack(
        (timestamps, np.zeros_like(timestamps), np.zeros_like(timestamps))
    ).astype(np.float32)
    robot_twist = np.tile(
        np.array([1.0, 0.0], dtype=np.float32), (timestamps.size, 1)
    )
    human_id = f"{recording_id}::Helmet_1"
    carried_id = f"{recording_id}::LO1"
    human = DynamicObjectTrack(
        object_id=human_id,
        source_body_name="Helmet_1",
        object_type="human",
        raw_role="Visitors-Alone",
        timestamps=timestamps.copy(),
        poses=np.column_stack(
            (timestamps + 2.0, np.ones_like(timestamps), np.zeros_like(timestamps))
        ).astype(np.float32),
        velocities=np.tile(
            np.array([1.0, 0.0], dtype=np.float32), (timestamps.size, 1)
        ),
        segment_ids=np.zeros(timestamps.shape, dtype=np.int32),
        footprint={"kind": "circle", "radius_m": 0.30},
        provenance={"geometry_source": "config_human"},
    )
    carried = DynamicObjectTrack(
        object_id=carried_id,
        source_body_name="LO1",
        object_type="carried_object",
        raw_role="Carried",
        timestamps=timestamps.copy(),
        poses=np.column_stack(
            (timestamps + 3.0, np.full_like(timestamps, 2.0), np.zeros_like(timestamps))
        ).astype(np.float32),
        velocities=np.tile(
            np.array([1.0, 0.0], dtype=np.float32), (timestamps.size, 1)
        ),
        segment_ids=np.zeros(timestamps.shape, dtype=np.int32),
        footprint={"kind": "rectangle", "length_m": 0.80, "width_m": 0.20},
        provenance={"geometry_source": "qtm_marker_p95"},
    )
    return RecordingIndex(
        recording_id=recording_id,
        session_id=f"{recording_id}-session",
        timestamps=timestamps,
        robot_pose=robot_pose,
        robot_twist=robot_twist,
        robot_segment_ids=np.zeros(timestamps.shape, dtype=np.int32),
        dynamic_objects={human_id: human, carried_id: carried},
        static_map=None,
        source_file=f"THOR-Magni_{recording_id}.csv",
        dt_s=0.2,
    )


def test_extract_base_states_separates_observed_and_oracle_dynamic_objects():
    from src.datasets.base_state_index import extract_base_states

    grid = build_grid_spec(load_config())
    result = extract_base_states(
        _recording(),
        split="train",
        grid=grid,
        stride_s=0.6,
    )

    assert len(result.base_states) == len(result.oracle_contexts) == 3
    state = result.base_states[0]
    oracle = result.oracle_contexts[0]
    assert state.state_id == oracle.base_state_id
    assert state.dynamic_object_ids == (
        "toy-recording::Helmet_1",
        "toy-recording::LO1",
    )
    assert state.robot_history.shape == (grid.history_steps, 3)
    assert state.robot_history.dtype == np.float32
    assert np.allclose(state.robot_history[-1], 0.0, atol=1e-6)
    human_history = state.visible_dynamic_object_history[
        "toy-recording::Helmet_1"
    ]
    assert human_history.shape == (grid.history_steps, 3)
    assert human_history.dtype == np.float32
    assert np.allclose(human_history[-1], [2.0, 1.0, 0.0], atol=1e-6)
    assert set(state.visible_dynamic_object_specs) == set(state.dynamic_object_ids)
    assert oracle.dynamic_object_future["toy-recording::LO1"].shape == (
        grid.future_steps,
        3,
    )
    assert oracle.dynamic_object_future["toy-recording::LO1"].dtype == np.float32
    assert set(oracle.dynamic_object_specs) == set(oracle.dynamic_object_future)
    assert not any("future" in field for field in state.__dict__)
    assert not any("oracle" in field for field in state.__dict__)
    validate_base_state(state, grid)
    validate_oracle_context(oracle, grid)


def test_empty_dynamic_base_states_are_accepted_not_rejected():
    from src.datasets.base_state_index import extract_base_states

    grid = build_grid_spec(load_config())
    result = extract_base_states(
        replace(_recording(), dynamic_objects={}),
        split="train",
        grid=grid,
        stride_s=0.6,
    )

    assert len(result.base_states) == 3
    assert result.summary["rejected_count"] == 0
    assert result.summary["empty_dynamic_count"] == 3


def test_parallel_base_state_extraction_matches_serial():
    from src.datasets.base_state_index import extract_base_state_index

    recordings = [
        _recording(recording_id="parallel-a"),
        _recording(recording_id="parallel-b"),
    ]
    grid = build_grid_spec(load_config())
    serial = extract_base_state_index(
        recordings,
        split="train",
        grid=grid,
        stride_s=0.6,
        workers=1,
    )
    parallel = extract_base_state_index(
        recordings,
        split="train",
        grid=grid,
        stride_s=0.6,
        workers=2,
    )

    assert parallel.summary == serial.summary
    assert [state.state_id for state in parallel.base_states] == [
        state.state_id for state in serial.base_states
    ]
    for actual, expected in zip(parallel.base_states, serial.base_states):
        assert actual.dynamic_object_ids == expected.dynamic_object_ids
        assert actual.visible_dynamic_object_specs == expected.visible_dynamic_object_specs
        assert np.array_equal(actual.robot_history, expected.robot_history)
        assert actual.visible_dynamic_object_history.keys() == (
            expected.visible_dynamic_object_history.keys()
        )
        for object_id in actual.visible_dynamic_object_history:
            assert np.array_equal(
                actual.visible_dynamic_object_history[object_id],
                expected.visible_dynamic_object_history[object_id],
            )
    for actual, expected in zip(
        parallel.oracle_contexts, serial.oracle_contexts
    ):
        assert actual.base_state_id == expected.base_state_id
        assert actual.dynamic_object_specs == expected.dynamic_object_specs
        assert actual.dynamic_object_future.keys() == expected.dynamic_object_future.keys()
        for object_id in actual.dynamic_object_future:
            assert np.array_equal(
                actual.dynamic_object_future[object_id],
                expected.dynamic_object_future[object_id],
            )


def test_base_state_extraction_rejects_nonpositive_workers():
    from src.datasets.base_state_index import extract_base_state_index
    from src.datasets.thor_adapter import ThorDataError

    with pytest.raises(ThorDataError, match="workers must be a positive integer"):
        extract_base_state_index(
            [_recording()],
            split="train",
            grid=build_grid_spec(load_config()),
            workers=0,
        )


def test_observed_objects_do_not_depend_on_future_availability():
    from src.datasets.base_state_index import extract_base_states

    recording = _recording()
    object_id = "toy-recording::Helmet_1"
    track = recording.dynamic_objects[object_id]
    history_only = replace(
        track,
        timestamps=track.timestamps[:20],
        poses=track.poses[:20],
        velocities=track.velocities[:20],
        segment_ids=track.segment_ids[:20],
    )
    grid = build_grid_spec(load_config())
    result = extract_base_states(
        replace(
            recording,
            dynamic_objects={**recording.dynamic_objects, object_id: history_only},
        ),
        split="train",
        grid=grid,
        stride_s=0.6,
    )

    first_state = result.base_states[0]
    first_oracle = result.oracle_contexts[0]
    assert object_id in first_state.dynamic_object_ids
    assert object_id in first_state.visible_dynamic_object_history
    assert object_id not in first_oracle.dynamic_object_future
    assert object_id not in first_oracle.dynamic_object_history
    assert object_id not in first_oracle.dynamic_object_specs


def test_out_of_view_oracle_object_is_not_exposed_in_base_state():
    from src.datasets.base_state_index import extract_base_states

    recording = _recording()
    object_id = "toy-recording::LO1"
    track = recording.dynamic_objects[object_id]
    far_poses = track.poses.copy()
    far_poses[:, 0] += 50.0
    grid = build_grid_spec(load_config())
    result = extract_base_states(
        replace(
            recording,
            dynamic_objects={
                **recording.dynamic_objects,
                object_id: replace(track, poses=far_poses),
            },
        ),
        split="train",
        grid=grid,
        stride_s=0.6,
    )

    assert object_id not in result.base_states[0].dynamic_object_ids
    assert object_id not in result.base_states[0].visible_dynamic_object_history
    assert object_id in result.oracle_contexts[0].dynamic_object_future


def test_base_and_oracle_artifacts_keep_dynamic_provenance_separate(tmp_path):
    from src.datasets.base_state_index import (
        extract_base_states,
        write_base_state_extraction,
    )

    grid = build_grid_spec(load_config())
    result = extract_base_states(
        _recording(), split="train", grid=grid, stride_s=0.6
    )
    paths = write_base_state_extraction(result, tmp_path / "base-index")

    assert paths["base_states"].name == "base_states"
    assert paths["oracle_contexts"].name == "oracle_contexts"
    manifest_rows = [
        json.loads(line)
        for line in paths["manifest"].read_text(encoding="utf-8").splitlines()
    ]
    assert len(manifest_rows) == 3
    assert manifest_rows[0]["dynamic_object_ids"] == [
        "toy-recording::Helmet_1",
        "toy-recording::LO1",
    ]
    assert all("oracle_path" not in row for row in manifest_rows)
    oracle_rows = [
        json.loads(line)
        for line in paths["oracle_manifest"].read_text(encoding="utf-8").splitlines()
    ]
    assert oracle_rows[0]["source_dynamic_object_ids"] == [
        "toy-recording::Helmet_1",
        "toy-recording::LO1",
    ]


def test_extract_base_states_cli_writes_v2_artifacts(tmp_path):
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
            str(root / "scripts/03_extract_base_states.py"),
            "--config",
            str(root / "configs/data_thor.yaml"),
            "--split",
            "train",
            "--recording-dir",
            str(tmp_path / "indexes"),
            "--output-dir",
            str(tmp_path / "states"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    split_dir = tmp_path / "states/train"
    assert len(list((split_dir / "base_states").glob("*.npz"))) == 3
    assert len(list((split_dir / "oracle_contexts").glob("*.npz"))) == 3
    summary = json.loads(
        (split_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["dynamic_object_type_counts"] == {
        "carried_object": 1,
        "human": 1,
        "unknown_dynamic": 0,
    }
    assert "accepted_count=3" in completed.stdout
    assert "workers_requested=8" in completed.stdout
    assert "workers_used=1" in completed.stdout
