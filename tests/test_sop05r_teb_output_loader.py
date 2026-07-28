from pathlib import Path

import pytest

from src.generation.sop05r_teb_event_sampler import build_sop05r_teb_mother
from src.generation.sop05r_teb_output_loader import load_sop05r_teb_output
from src.generation.sop05r_teb_run import publish_sop05r_teb_run
from tests.test_sop05r_teb_event_sampler import _mother_fixture


def _publish_complete_output(tmp_path: Path) -> Path:
    inputs = _mother_fixture()
    evaluation = build_sop05r_teb_mother(
        base_config=inputs[0],
        source_base_state=inputs[1],
        source_oracle_context=inputs[2],
        teb_config=inputs[3],
        task_template=inputs[4],
        placement_result=inputs[5],
        snippet=inputs[6],
        seed=43,
    )
    assert evaluation.mother is not None
    output = tmp_path / "complete"
    publish_sop05r_teb_run(
        (evaluation.mother,),
        output,
        base_config=inputs[0],
        requested_count=1,
        config_digest=inputs[3].digest,
        verification_action_digest="b" * 64,
        source_evidence={"producer": "test"},
        denominator_counts={"m6_accepted": 1},
        rejection_counts={},
    )
    return output


def test_output_loader_strictly_round_trips_complete_long40_collection(
    tmp_path: Path,
) -> None:
    loaded = load_sop05r_teb_output(
        _publish_complete_output(tmp_path),
        require_complete=True,
    )

    assert loaded.complete
    assert len(loaded.events) == 1
    assert loaded.events[0].target.history_poses.shape == (8, 3)
    assert loaded.events[0].target.future_poses.shape == (32, 3)
    assert loaded.trajectories.records[0].full_route.sampled_poses_world.shape == (
        40,
        3,
    )
    assert loaded.trajectories.records[0].nominal_trajectory.poses.shape == (32, 3)
    evidence = loaded.event_evidence[loaded.events[0].generated_event_id]
    assert evidence["collision_point_xy"]
    assert evidence["occlusion_witness"]["sample_index"] < 8


def test_output_loader_rejects_outer_event_payload_tampering(tmp_path: Path) -> None:
    output = _publish_complete_output(tmp_path)
    events = output / "events.json"
    events.write_bytes(events.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="checksum"):
        load_sop05r_teb_output(output, require_complete=True)


def test_output_loader_requires_completion_marker_for_partial_output(
    tmp_path: Path,
) -> None:
    inputs = _mother_fixture()
    output = tmp_path / "partial"
    publish_sop05r_teb_run(
        (),
        output,
        base_config=inputs[0],
        requested_count=1,
        config_digest=inputs[3].digest,
        verification_action_digest="b" * 64,
        source_evidence={"producer": "test"},
        denominator_counts={"m6_accepted": 0},
        rejection_counts={"teb_goal_unreached": 1},
    )

    with pytest.raises(ValueError, match="completion marker"):
        load_sop05r_teb_output(output, require_complete=True)
