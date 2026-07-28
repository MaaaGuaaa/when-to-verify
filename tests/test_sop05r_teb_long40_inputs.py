"""Tests for the Schema 4 Long40 input path used by the TEB producer."""

from __future__ import annotations

import inspect
from pathlib import Path

import src.generation.sop05r_teb_long40_inputs as long40_inputs_module
from src.contracts import build_grid_spec
from src.datasets.long_snippet_library import write_long_snippet_artifacts
from src.datasets.thor_adapter import write_recording_indexes
from src.utils.config import load_config
from tests.test_long_snippet_library import _build, _recording, _split_provenance


ROOT = Path(__file__).resolve().parents[1]


def test_long40_inputs_validate_artifact_and_materialize_schema4_state_pairs(
    tmp_path: Path,
) -> None:
    from src.generation.sop05r_teb_long40_inputs import (
        load_sop05r_teb_long40_inputs,
    )

    recording_root = tmp_path / "recordings"
    recording = _recording()
    write_recording_indexes(
        [recording],
        split="train",
        output_dir=recording_root / "recording_indexes" / "train",
        split_provenance=_split_provenance(),
    )
    artifact = tmp_path / "long40" / "train" / "human"
    write_long_snippet_artifacts(
        _build(recording),
        artifact,
        overlap_report={"status": "ok", "total_overlap_count": 0},
    )

    inputs = load_sop05r_teb_long40_inputs(
        recording_root=recording_root,
        long40_human_artifact=artifact,
        split="train",
        grid=build_grid_spec(load_config(ROOT / "configs/base.yaml")),
        max_base_states=2,
    )

    assert len(inputs.snippets) == 6
    assert len(inputs.state_pairs) == 2
    assert all(state.split == "train" for state, _ in inputs.state_pairs)
    assert all(
        state.state_id == context.base_state_id
        for state, context in inputs.state_pairs
    )
    assert inputs.source_evidence["input_kind"] == "sop03_schema4_long40_runtime_v1"
    assert inputs.source_evidence["accepted_source_snippet_count"] == 6


def test_long40_selected_inputs_avoid_full_combined_extraction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recording_root = tmp_path / "recordings"
    recording = _recording()
    write_recording_indexes(
        [recording],
        split="train",
        output_dir=recording_root / "recording_indexes" / "train",
        split_provenance=_split_provenance(),
    )
    artifact = tmp_path / "long40" / "train" / "human"
    write_long_snippet_artifacts(
        _build(recording),
        artifact,
        overlap_report={"status": "ok", "total_overlap_count": 0},
    )
    grid = build_grid_spec(load_config(ROOT / "configs/base.yaml"))
    baseline = long40_inputs_module.load_sop05r_teb_long40_inputs(
        recording_root=recording_root,
        long40_human_artifact=artifact,
        split="train",
        grid=grid,
        max_base_states=2,
    )
    required_id = baseline.state_pairs[0][0].state_id
    assert "required_state_ids" in inspect.signature(
        long40_inputs_module.load_sop05r_teb_long40_inputs
    ).parameters
    monkeypatch.setattr(
        long40_inputs_module,
        "extract_base_state_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("selected loading used full combined extraction")
        ),
    )

    selected = long40_inputs_module.load_sop05r_teb_long40_inputs(
        recording_root=recording_root,
        long40_human_artifact=artifact,
        split="train",
        grid=grid,
        max_base_states=2,
        required_state_ids=frozenset({required_id}),
    )

    assert tuple(state.state_id for state, _ in selected.state_pairs) == (
        required_id,
    )
    assert selected.source_evidence["selected_base_state_count"] == 2
