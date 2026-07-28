from pathlib import Path

import numpy as np
import pytest

from src.generation.sop05r_teb_event_sampler import build_sop05r_teb_mother
from src.generation.sop05r_teb_output_loader import load_sop05r_teb_output
from src.generation.sop05r_teb_run import publish_sop05r_teb_run
from src.generation.sop05r_teb_templates import iter_sop05r_teb_task_templates
from src.generation.sop06_single import (
    render_sop06_teb_target_pair,
    resolve_sop06_teb_handoff,
)
from src.generation.anchored_human_placement import solve_anchored_human_placement
from tests.test_anchored_human_placement import _m4_inputs, _snippet


def _strict_collection(
    tmp_path: Path,
    *,
    source_evidence: dict[str, object] | None = None,
):
    base_config, base_state, oracle_context, teb_config = _m4_inputs()
    placement = None
    task = None
    for template_evaluation in iter_sop05r_teb_task_templates(
        base_state=base_state,
        oracle_context=oracle_context,
        base_config=base_config,
        teb_config=teb_config,
        seed=19,
    ):
        if template_evaluation.template is None:
            continue
        candidate = solve_anchored_human_placement(
            task_template=template_evaluation.template,
            snippet=_snippet(),
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
            teb_config=teb_config,
            seed=27,
        )
        if candidate.result is not None:
            task = template_evaluation.template
            placement = candidate.result
            break
    assert task is not None and placement is not None
    evaluation = build_sop05r_teb_mother(
        base_config=base_config,
        source_base_state=base_state,
        source_oracle_context=oracle_context,
        teb_config=teb_config,
        task_template=task,
        placement_result=placement,
        snippet=_snippet(),
        seed=43,
    )
    assert evaluation.mother is not None
    output = tmp_path / "m7"
    publish_sop05r_teb_run(
        (evaluation.mother,),
        output,
        base_config=base_config,
        requested_count=1,
        config_digest=teb_config.digest,
        verification_action_digest="b" * 64,
        source_evidence=source_evidence or {"producer": "test"},
        denominator_counts={"m6_accepted": 1},
        rejection_counts={},
    )
    return load_sop05r_teb_output(output, require_complete=True)


def test_current_m7_collection_hands_one_route_to_sop06_target_pair(
    tmp_path: Path,
) -> None:
    collection = _strict_collection(tmp_path)
    event = collection.events[0]

    handoff = resolve_sop06_teb_handoff(
        collection,
        event_id=event.generated_event_id,
    )
    pair = render_sop06_teb_target_pair(handoff)

    assert handoff.full_route.band_poses_world.shape == (21, 3)
    assert handoff.full_route.sampled_poses_world.shape == (40, 3)
    assert handoff.nominal_trajectory.poses.shape == (32, 3)
    np.testing.assert_array_equal(
        handoff.shared_goal_world_pose,
        event.world.metadata["shared_goal_world_pose"],
    )
    assert pair.handoff is handoff
    assert pair.target_present.bev_history.shape == pair.target_removed.bev_history.shape
    assert not np.array_equal(pair.target_present.bev_history, pair.target_removed.bev_history)


def test_teb_handoff_rejects_unknown_event_without_legacy_fallback(
    tmp_path: Path,
) -> None:
    collection = _strict_collection(tmp_path)

    with pytest.raises(ValueError, match="exactly one event"):
        resolve_sop06_teb_handoff(collection, event_id="missing-event")
