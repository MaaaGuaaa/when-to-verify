import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
import numpy as np

import src.generation.sop05_final_scenarios as final_scenarios
from src.contracts import BaseState, OracleWorld, save_dataclass
import src.generation.sop05_partial_m6_final as partial_final
from src.generation.sop05_partial_m6_final import (
    build_partial_mother_view,
    publish_partial_m6_final_scenarios,
)
from src.generation.sop05_final_scenarios import (
    LoadedSop05rTebOutput,
    load_sop05_final_scenarios,
    publish_sop05_final_scenarios,
)
from src.generation.sop05_seen_prior import load_seen_prior_config
from src.generation.sop05_unseen_prior import normalize_unseen_prior_config
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_partial_m6_view_reconstructs_only_represented_context_and_visibility() -> None:
    robot_history = np.zeros((8, 3), dtype=np.float32)
    context_history = np.zeros((8, 3), dtype=np.float32)
    context_history[:, 1] = 2.0
    source = BaseState(
        state_id="source-0",
        split="train",
        recording_id="recording-0",
        dynamic_object_ids=("context-keep", "context-drop"),
        timestamp=10.0,
        robot_history=robot_history,
        robot_state=np.asarray([0.4, 0.0], dtype=np.float32),
        visible_dynamic_object_history={
            "context-keep": context_history,
            "context-drop": context_history + 1.0,
        },
        visible_dynamic_object_specs={
            "context-keep": {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.3},
            },
            "context-drop": {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.3},
            },
        },
        static_map_local=np.zeros((16, 16), dtype=np.float32),
    )
    target_history = np.zeros((8, 3), dtype=np.float32)
    target_history[:, 0] = 2.0
    target_future = np.repeat(target_history[-1:], 32, axis=0)
    world = OracleWorld(
        world_id="world-0",
        base_state_id="decision-0",
        static_occupancy=np.zeros((16, 16), dtype=np.float32),
        dynamic_object_trajectories={
            "context-keep": np.repeat(context_history[-1:], 32, axis=0),
            "target-0": target_future,
        },
        dynamic_object_specs={
            "context-keep": source.visible_dynamic_object_specs["context-keep"],
            "target-0": {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.3},
            },
        },
        occluders=(
            {
                "shape": "rectangle",
                "occluder_id": "wall-0",
                "semantic_type": "wall",
                "pose": [1.0, 0.0, 0.0],
                "length_m": 0.4,
                "width_m": 1.0,
            },
        ),
        blind_spot_config={},
        random_seed=1,
        metadata={},
    )
    view = build_partial_mother_view(
        trajectory_row={
            "event_id": "event-0",
            "source_base_state_id": "source-0",
            "decision_state_id": "decision-0",
            "nominal_trajectory_id": "trajectory-0",
        },
        target_row={
            "generated_event_id": "event-0",
            "world_id": "world-0",
            "base_state_id": "decision-0",
            "trajectory_id": "trajectory-0",
            "target_dynamic_object_id": "target-0",
            "source_snippet_id": "snippet-0",
            "source_object_id": "source-human-0",
            "object_type": "human",
            "footprint_spec": world.dynamic_object_specs["target-0"],
        },
        target_history=target_history,
        target_current=target_history[-1],
        target_future=target_future,
        world=world,
        source_state=source,
        source_recording_id="recording-0",
        source_session_id="session-0",
        centerline_epsilon_m=0.01,
    )

    assert view.state.state_id == "decision-0"
    assert view.state.dynamic_object_ids == ("context-keep",)
    assert set(view.state.visible_dynamic_object_history) == {"context-keep"}
    assert np.array_equal(view.state.robot_history, source.robot_history)
    assert not np.asarray(view.event.target_visibility_history).any()
    assert view.event.target.source_recording_id == "recording-0"


def test_history_regime_is_defined_by_initial_visibility() -> None:
    initially_hidden = SimpleNamespace(
        target_visibility_history=np.asarray(
            [False, True, True, False, False, True, False, False],
            dtype=np.bool_,
        )
    )
    initially_visible = SimpleNamespace(
        target_visibility_history=np.asarray(
            [True, True, True, True, False, False, False, False],
            dtype=np.bool_,
        )
    )
    initially_hidden_current_visible = SimpleNamespace(
        target_visibility_history=np.asarray(
            [False, False, False, False, False, False, False, True],
            dtype=np.bool_,
        )
    )
    initially_visible_current_visible = SimpleNamespace(
        target_visibility_history=np.asarray(
            [True, False, False, False, False, False, False, True],
            dtype=np.bool_,
        )
    )

    assert final_scenarios._history_regime(initially_hidden) == (
        "unseen_in_history_window"
    )
    assert final_scenarios._history_regime(initially_visible) == "seen_then_occluded"
    assert final_scenarios._history_regime(initially_hidden_current_visible) == (
        "unseen_in_history_window"
    )
    assert (
        final_scenarios._history_regime(initially_visible_current_visible)
        == "seen_then_occluded"
    )


def test_partial_m6_publisher_does_not_require_trajectory_payload(
    tmp_path: Path,
) -> None:
    root = tmp_path / "partial-m6"
    trajectory_root = root / "trajectory_store"
    target_root = root / "target_motion"
    world_root = target_root / "oracle_worlds"
    trajectory_root.mkdir(parents=True)
    world_root.mkdir(parents=True)
    trajectory_digest = "1" * 64
    target_manifest_digest = "2" * 32
    target_payload_digest = "3" * 32
    trajectory_row = {
        "event_id": "event-0",
        "source_base_state_id": "source-0",
        "decision_state_id": "decision-0",
        "nominal_trajectory_id": "trajectory-0",
    }
    (trajectory_root / "manifest.json").write_text(
        json.dumps(
            {
                "collection_semantic_digest": trajectory_digest,
                "record_count": 1,
                "records": [trajectory_row],
            }
        ),
        encoding="ascii",
    )
    (trajectory_root / "COMPLETE.json").write_text("{}\n", encoding="ascii")

    target_row = {
        "row_index": 0,
        "generated_event_id": "event-0",
        "world_id": "world-0",
        "base_state_id": "decision-0",
        "trajectory_id": "trajectory-0",
        "target_dynamic_object_id": "target-0",
        "source_snippet_id": "snippet-0",
        "source_object_id": "source-human-0",
        "object_type": "human",
        "footprint_spec": {
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": 0.3},
        },
        "world_file": "oracle_worlds/world-0.npz",
    }
    (target_root / "generated_event_manifest.jsonl").write_text(
        json.dumps(target_row) + "\n", encoding="ascii"
    )
    (target_root / "shard_summary.json").write_text(
        json.dumps(
            {
                "record_count": 1,
                "manifest_digest": target_manifest_digest,
                "payload_semantic_digest": target_payload_digest,
            }
        ),
        encoding="ascii",
    )
    history = np.zeros((1, 8, 3), dtype=np.float32)
    history[:, :, 0] = 2.0
    current = history[:, 7].copy()
    future = np.repeat(history[:, -1:, :], 32, axis=1)
    np.savez(
        target_root / "event_target_motion_history8_future32_v2.npz",
        history_poses=history,
        current_poses=current,
        future_poses=future,
        meta_json=np.asarray("{}"),
    )
    world = OracleWorld(
        world_id="world-0",
        base_state_id="decision-0",
        static_occupancy=np.zeros((160, 160), dtype=np.float32),
        dynamic_object_trajectories={"target-0": future[0]},
        dynamic_object_specs={"target-0": target_row["footprint_spec"]},
        occluders=(
            {
                "shape": "rectangle",
                "occluder_id": "wall-0",
                "semantic_type": "wall",
                "pose": [1.0, 0.0, 0.0],
                "length_m": 0.4,
                "width_m": 1.0,
            },
        ),
        blind_spot_config={},
        random_seed=1,
        metadata={},
    )
    save_dataclass(world, world_root / "world-0.npz")
    source = BaseState(
        state_id="source-0",
        split="train",
        recording_id="recording-0",
        dynamic_object_ids=(),
        timestamp=0.0,
        robot_history=np.zeros((8, 3), dtype=np.float32),
        robot_state=np.asarray([0.4, 0.0], dtype=np.float32),
        visible_dynamic_object_history={},
        visible_dynamic_object_specs={},
        static_map_local=np.zeros((160, 160), dtype=np.float32),
    )
    base_config = load_config(ROOT / "configs/base.yaml")
    unseen_config = normalize_unseen_prior_config(
        yaml.safe_load((ROOT / "configs/sop05_unseen_prior.yaml").read_text()),
        base_config=base_config,
    )
    output = tmp_path / "final"

    result = publish_partial_m6_final_scenarios(
        root,
        source_states={"source-0": source},
        snippet_sources={"snippet-0": ("recording-0", "session-0")},
        base_config=base_config,
        source_config_digest="source-config",
        centerline_epsilon_m=0.01,
        output_dir=output,
        unseen_config=unseen_config,
        seen_config=load_seen_prior_config(ROOT / "configs/sop05_seen_prior.yaml"),
    )

    assert result.processed_mother_count == 1
    assert result.accepted_count + result.deficit_count == 1
    assert (output / "COMPLETE.json").is_file()
    assert not (trajectory_root / "trajectories.npz").exists()


def test_final_sop05_release_accounts_for_each_mother_and_binds_source(
    tmp_path: Path, monkeypatch
) -> None:
    base_config = load_config(ROOT / "configs/base.yaml")
    source = object.__new__(LoadedSop05rTebOutput)
    object.__setattr__(source, "events", (SimpleNamespace(generated_event_id="m0"),))
    object.__setattr__(
        source,
        "manifest",
        {"config_digest": "source-config", "base_config": dict(base_config)},
    )
    object.__setattr__(source, "publication_semantic_digest", "source-digest")
    object.__setattr__(source, "complete", True)
    monkeypatch.setattr(
        final_scenarios,
        "_process_all",
        lambda *args, **kwargs: [
            final_scenarios._WorkResult(
                record={
                    "source_index": 0,
                    "mother_id": "m0",
                    "split": "train",
                    "regime": "unseen_in_history_window",
                    "status": "accepted",
                    "scenario_id": "scenario-m0",
                    "target_present": True,
                },
                provenance={
                    "source_index": 0,
                    "mother_id": "m0",
                    "status": "accepted",
                    "selected_angle_rad": 0.1,
                },
                history_poses=np.zeros((8, 3), dtype=np.float32),
                future_poses=np.zeros((32, 3), dtype=np.float32),
                target_present=True,
            )
        ],
    )
    output = tmp_path / "final-scenarios"
    unseen_config = normalize_unseen_prior_config(
        yaml.safe_load((ROOT / "configs/sop05_unseen_prior.yaml").read_text()),
        base_config=dict(source.manifest["base_config"]),
    )

    result = publish_sop05_final_scenarios(
        source,
        output_dir=output,
        unseen_config=unseen_config,
        seen_config=load_seen_prior_config(ROOT / "configs/sop05_seen_prior.yaml"),
        expected_source_config_digest="source-config",
    )
    loaded = load_sop05_final_scenarios(
        output,
        expected_source_publication_semantic_digest=source.publication_semantic_digest,
    )

    assert result.output_dir == output
    assert len(loaded.records) == len(source.events)
    assert loaded.accepted_count + loaded.deficit_count == len(source.events)
    assert loaded.history_poses.shape == (1, 8, 3)
    assert loaded.future_poses.shape == (1, 32, 3)
    assert loaded.target_present.tolist() == [True]
    assert loaded.source_record_indices.tolist() == [0]
    assert loaded.history_poses.dtype == np.float32
    assert loaded.future_poses.dtype == np.float32
    assert loaded.target_present.dtype == np.bool_
    assert loaded.source_record_indices.dtype == np.int64
    assert {record["mother_id"] for record in loaded.records} == {
        event.generated_event_id for event in source.events
    }
    assert all(
        "selected_angle_rad" not in record and "target_future" not in record
        for record in loaded.records
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_publication_semantic_digest"] == (
        source.publication_semantic_digest
    )
    assert (output / "COMPLETE.json").is_file()

    with pytest.raises(FileExistsError, match="overwrite"):
        publish_sop05_final_scenarios(
            source,
            output_dir=output,
            unseen_config=unseen_config,
            seen_config=load_seen_prior_config(ROOT / "configs/sop05_seen_prior.yaml"),
            expected_source_config_digest="source-config",
        )


def test_selected_scenarios_publish_present_and_empty_without_resampling(
    tmp_path: Path,
) -> None:
    selection_type = final_scenarios.Sop05FinalScenarioSelection
    source = object.__new__(LoadedSop05rTebOutput)
    object.__setattr__(
        source,
        "events",
        (
            SimpleNamespace(generated_event_id="m0"),
            SimpleNamespace(generated_event_id="m1"),
        ),
    )
    object.__setattr__(source, "manifest", {"config_digest": "source-config"})
    object.__setattr__(source, "publication_semantic_digest", "source-digest")
    object.__setattr__(source, "complete", True)
    present_history = np.zeros((8, 3), dtype=np.float32)
    present_future = np.ones((32, 3), dtype=np.float32)
    zero_history = np.zeros((8, 3), dtype=np.float32)
    zero_future = np.zeros((32, 3), dtype=np.float32)
    output = tmp_path / "selected-final"

    result = final_scenarios.publish_selected_sop05_final_scenarios(
        source,
        selections=(
            selection_type(
                mother_id="m0",
                split="train",
                target_present=True,
                history_poses=present_history,
                future_poses=present_future,
                provenance={"stratum": "a_present", "selected_angle_rad": 0.2},
            ),
            selection_type(
                mother_id="m1",
                split="train",
                target_present=False,
                history_poses=zero_history,
                future_poses=zero_future,
                provenance={"stratum": "a_empty"},
            ),
        ),
        output_dir=output,
        unseen_prior_config_digest="supplement-config",
        seen_prior_config_digest="not-used",
    )
    loaded = load_sop05_final_scenarios(
        output,
        expected_source_publication_semantic_digest="source-digest",
    )

    assert result.accepted_count == result.processed_mother_count == 2
    assert result.deficit_count == 0
    assert result.full_source_coverage
    assert loaded.accepted_count == 2
    assert loaded.deficit_count == 0
    assert loaded.target_present.tolist() == [True, False]
    np.testing.assert_array_equal(loaded.history_poses[0], present_history)
    np.testing.assert_array_equal(loaded.future_poses[0], present_future)
    np.testing.assert_array_equal(loaded.history_poses[1], zero_history)
    np.testing.assert_array_equal(loaded.future_poses[1], zero_future)
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "records.jsonl",
        "oracle_targets.npz",
        "provenance.jsonl",
        "checksums.json",
        "COMPLETE.json",
    }
