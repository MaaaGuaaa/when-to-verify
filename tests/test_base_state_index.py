"""Tests for robot-centric BaseState and isolated OracleContext extraction."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.contracts import build_grid_spec
from src.datasets.thor_adapter import PedestrianTrack, RecordingIndex
from src.utils.config import load_config


def _recording(duration_s: float = 6.0) -> RecordingIndex:
    timestamps = np.arange(0.0, duration_s + 1e-9, 0.2, dtype=np.float64)
    robot_pose = np.column_stack(
        (timestamps, np.zeros_like(timestamps), np.zeros_like(timestamps))
    ).astype(np.float32)
    robot_twist = np.tile(
        np.array([1.0, 0.0], dtype=np.float32), (timestamps.size, 1)
    )
    pedestrian_positions = np.column_stack(
        (timestamps + 2.0, np.ones_like(timestamps))
    ).astype(np.float32)
    pedestrian = PedestrianTrack(
        participant_id="ped-1",
        timestamps=timestamps.copy(),
        positions=pedestrian_positions,
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
        robot_twist=robot_twist,
        robot_segment_ids=np.zeros(timestamps.shape, dtype=np.int32),
        pedestrians={"ped-1": pedestrian},
        static_map=None,
        source_file="THOR-Magni_toy-recording.csv",
        dt_s=0.2,
    )


def test_extract_base_states_separates_observed_and_oracle_data():
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
    assert state.split == "train"
    assert state.recording_id == "toy-recording"
    assert state.participant_ids == ("ped-1",)
    assert state.robot_history.shape == (grid.history_steps, 3)
    assert state.robot_history.dtype == np.float32
    assert np.allclose(state.robot_history[-1], 0.0, atol=1e-6)
    assert state.robot_state.shape == (2,)
    assert state.robot_state.dtype == np.float32
    visible = state.visible_pedestrian_history["ped-1"]
    assert visible.shape == (grid.history_steps, 2)
    assert visible.dtype == np.float32
    assert np.allclose(visible[-1], [2.0, 1.0], atol=1e-6)
    assert oracle.pedestrian_history["ped-1"].shape == (grid.history_steps, 2)
    assert oracle.pedestrian_future["ped-1"].shape == (grid.future_steps, 2)
    assert oracle.pedestrian_future["ped-1"].dtype == np.float32
    assert not any("future" in field for field in state.__dict__)
    assert not any("oracle" in field for field in state.__dict__)
    assert np.isfinite(state.robot_history).all()
    assert np.isfinite(oracle.pedestrian_future["ped-1"]).all()


def test_empty_dynamic_base_states_are_accepted_not_counted_as_rejections():
    from src.datasets.base_state_index import extract_base_states

    grid = build_grid_spec(load_config())
    result = extract_base_states(
        replace(_recording(), pedestrians={}),
        split="train",
        grid=grid,
        stride_s=0.6,
    )

    assert len(result.base_states) == 3
    assert result.summary["rejected_count"] == 0
    assert result.summary["empty_dynamic_count"] == 3


def test_observed_participants_do_not_depend_on_future_availability():
    from src.datasets.base_state_index import extract_base_states

    recording = _recording()
    track = recording.pedestrians["ped-1"]
    history_only_track = replace(
        track,
        timestamps=track.timestamps[:20],
        positions=track.positions[:20],
        velocities=track.velocities[:20],
        segment_ids=track.segment_ids[:20],
    )
    grid = build_grid_spec(load_config())
    result = extract_base_states(
        replace(recording, pedestrians={"ped-1": history_only_track}),
        split="train",
        grid=grid,
        stride_s=0.6,
    )

    first_state = result.base_states[0]
    first_oracle = result.oracle_contexts[0]
    assert first_state.participant_ids == ("ped-1",)
    assert "ped-1" in first_state.visible_pedestrian_history
    assert "ped-1" not in first_oracle.pedestrian_future


def test_out_of_view_oracle_participants_are_not_exposed_in_base_state():
    from src.datasets.base_state_index import extract_base_states

    recording = _recording()
    track = recording.pedestrians["ped-1"]
    far_track = replace(
        track,
        positions=(track.positions + np.array([50.0, 0.0], dtype=np.float32)),
    )
    grid = build_grid_spec(load_config())
    result = extract_base_states(
        replace(recording, pedestrians={"ped-1": far_track}),
        split="train",
        grid=grid,
        stride_s=0.6,
    )

    assert result.base_states[0].participant_ids == ()
    assert result.base_states[0].visible_pedestrian_history == {}
    assert "ped-1" in result.oracle_contexts[0].pedestrian_future


def test_base_and_oracle_artifacts_are_written_to_separate_directories(tmp_path):
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
    assert set(path.name for path in paths["base_states"].iterdir()) == {
        f"{state.state_id}.npz" for state in result.base_states
    }
    assert set(path.name for path in paths["oracle_contexts"].iterdir()) == {
        f"{oracle.base_state_id}.npz" for oracle in result.oracle_contexts
    }
    manifest_rows = [
        json.loads(line)
        for line in paths["manifest"].read_text(encoding="utf-8").splitlines()
    ]
    assert len(manifest_rows) == 3
    assert all("oracle_path" not in row for row in manifest_rows)
    assert json.loads(paths["summary"].read_text(encoding="utf-8"))[
        "accepted_count"
    ] == 3


def test_extract_base_states_cli_writes_observed_and_oracle_artifacts(tmp_path):
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
    assert summary["recording_count"] == 1
    assert summary["accepted_count"] == 3
    assert "accepted_count=3" in completed.stdout
