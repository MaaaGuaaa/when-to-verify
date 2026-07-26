from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.contracts import OracleContext
from src.generation.anchored_human_placement import (
    AnchoredHumanPlacement,
    AnchoredPlacementResult,
    CenterlineOcclusionWitness,
    CollisionAnchor,
)
from src.generation.history_visibility import (
    classify_sop05r_seen_then_occluded_history,
)
from src.generation.sop05r_teb_templates import iter_sop05r_teb_task_templates


def _decision_fixture():
    from tests.test_anchored_human_placement import _m4_inputs

    base_config, base_state, oracle_context, teb_config = _m4_inputs()
    task = next(
        item.template
        for item in iter_sop05r_teb_task_templates(
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
            teb_config=teb_config,
            seed=19,
        )
        if (
            item.template is not None
            and item.template.route.goal_arrival_time_s >= 6.8
        )
    )
    decision_time_s = 0.0
    route_anchor_index = 8
    route_time_s = float(task.route.sample_times_s[route_anchor_index])
    snippet_anchor_index = 7 + int(round(route_time_s / 0.2))
    anchor = CollisionAnchor(
        route_sample_index=route_anchor_index,
        route_time_s=route_time_s,
        world_position_xy=task.route.sampled_poses_world[route_anchor_index, :2],
        snippet_anchor_index=snippet_anchor_index,
        snippet_time_s=route_time_s,
    )
    target_poses = np.zeros((40, 3), dtype=np.float32)
    frame_offsets = np.arange(40, dtype=np.float32) - snippet_anchor_index
    target_poses[:, 0] = anchor.world_position_xy[0] + 0.05 * frame_offsets
    target_poses[:, 1] = anchor.world_position_xy[1]
    target_poses[:, :2] += (
        anchor.world_position_xy - target_poses[snippet_anchor_index, :2]
    )
    placement = AnchoredHumanPlacement(
        source_snippet_id="decision-fixture-human",
        anchor=anchor,
        rotation_rad=0.0,
        translation_xy_m=np.zeros(2),
        spatial_scale=1.0,
        temporal_scale=1.0,
        history_poses=target_poses[:8],
        current_pose=target_poses[7],
        future_poses=target_poses[8:],
        provenance={"fixture": "decision"},
    )
    witness = CenterlineOcclusionWitness(
        version=teb_config.occlusion.version,
        time_s=decision_time_s,
        sample_index=7,
        robot_position_xy=base_state.robot_history[-1, :2],
        target_position_xy=target_poses[7, :2],
        blocking_occluder_id=task.occluders[0].occluder_id,
    )
    blocked = np.zeros(40, dtype=np.bool_)
    blocked[7] = True
    visibility = classify_sop05r_seen_then_occluded_history(
        blocked,
        decision_index=7,
        minimum_visible_frames=4,
        minimum_occluded_frames=1,
    )
    result = AnchoredPlacementResult(
        placement=placement,
        witness=witness,
        visibility=visibility,
        attempted_candidates=1,
        rejection_counts={},
    )
    return base_config, base_state, teb_config, task, result


def test_context_history_interpolation_uses_all_40_long40_poses() -> None:
    from src.generation.sop05r_teb_decision_state import _sample_context_history

    object_id = "recording::person"
    poses = np.zeros((40, 3), dtype=np.float32)
    poses[:, 0] = np.arange(40, dtype=np.float32) * 0.2
    context = OracleContext(
        base_state_id="base",
        dynamic_object_history={object_id: poses[:8]},
        dynamic_object_future={object_id: poses[8:]},
        dynamic_object_specs={
            object_id: {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.3},
            }
        },
    )

    sampled = _sample_context_history(
        context,
        object_id=object_id,
        sample_times_s=np.asarray([-1.4, 0.0, 6.4]),
        dt_s=0.2,
    )

    np.testing.assert_allclose(sampled[:, 0], [0.0, 1.4, 7.8], atol=1e-6)


def test_decision_state_builds_exact_6_4_second_local_suffix() -> None:
    from src.generation.sop05r_teb_decision_state import build_teb_decision_state

    base_config, base_state, teb_config, task, placement = _decision_fixture()

    decision = build_teb_decision_state(
        task_template=task,
        placement_result=placement,
        source_base_state=base_state,
        source_oracle_context=OracleContext(
            base_state_id=base_state.state_id,
            dynamic_object_history={},
            dynamic_object_future={},
            dynamic_object_specs={},
        ),
        base_config=base_config,
        teb_config=teb_config,
        seed=31,
    )

    assert decision.decision_time_s == 0.0
    assert decision.robot_history_world.shape == (8, 3)
    assert decision.target_visibility_history.sum() == 7
    assert not decision.target_visibility_history[-1]
    assert decision.suffix_poses_world.shape == (32, 3)
    assert decision.nominal_trajectory.poses.shape == (32, 3)
    assert decision.nominal_trajectory.controls.shape == (32, 2)
    np.testing.assert_allclose(decision.nominal_trajectory.poses[0], [0.0, 0.0, 0.0], atol=0.2)
    assert decision.nominal_trajectory.task_cost == task.route.task_cost
    assert decision.base_state.state_id == decision.decision_state_id
    assert decision.base_state.static_map_local.shape == task.static_occupancy.shape
    delta = placement.placement.current_pose[:2] - decision.decision_pose_world[:2]
    cosine = np.cos(decision.decision_pose_world[2])
    sine = np.sin(decision.decision_pose_world[2])
    np.testing.assert_allclose(
        decision.target_history_local[-1, :2],
        [cosine * delta[0] + sine * delta[1], -sine * delta[0] + cosine * delta[1]],
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        decision.nominal_trajectory.centerline_map,
        decision.recomputed_query_maps.centerline_map,
    )


def test_decision_state_keeps_a_terminal_hold_after_early_goal_arrival() -> None:
    from src.generation.sop05r_teb_decision_state import build_teb_decision_state

    base_config, base_state, teb_config, task, result = _decision_fixture()
    route_anchor_index = result.placement.anchor.route_sample_index
    route_time_s = float(task.route.sample_times_s[route_anchor_index])
    synchronized_anchor = replace(
        result.placement.anchor,
        snippet_anchor_index=7 + int(round(route_time_s / 0.2)),
    )
    synchronized_placement = replace(result.placement, anchor=synchronized_anchor)
    synchronized_result = replace(
        result,
        placement=synchronized_placement,
        witness=replace(
            result.witness,
            time_s=0.0,
            robot_position_xy=base_state.robot_history[-1, :2],
        ),
    )
    arrival_index = 10
    held_pose = task.route.sampled_poses_world[arrival_index].copy()
    held_poses = task.route.sampled_poses_world.copy()
    held_controls = task.route.sampled_controls.copy()
    held_poses[arrival_index:] = held_pose
    held_controls[arrival_index + 1 :] = 0.0
    early_route = replace(
        task.route,
        sampled_poses_world=held_poses,
        sampled_controls=held_controls,
        goal_arrival_time_s=float(task.route.sample_times_s[arrival_index + 1]),
    )
    early_task = replace(task, route=early_route)

    decision = build_teb_decision_state(
        task_template=early_task,
        placement_result=synchronized_result,
        source_base_state=base_state,
        source_oracle_context=OracleContext(
            base_state_id=base_state.state_id,
            dynamic_object_history={},
            dynamic_object_future={},
            dynamic_object_specs={},
        ),
        base_config=base_config,
        teb_config=teb_config,
        seed=31,
    )

    assert decision.decision_time_s == 0.0
    np.testing.assert_allclose(
        decision.suffix_poses_world[arrival_index:],
        np.broadcast_to(
            held_pose,
            decision.suffix_poses_world[arrival_index:].shape,
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        decision.suffix_controls[arrival_index + 1 :],
        0.0,
        atol=1e-6,
    )


def test_decision_state_allows_motion_after_entering_wide_goal_region() -> None:
    from src.generation.sop05r_teb_decision_state import build_teb_decision_state

    base_config, base_state, teb_config, task, result = _decision_fixture()
    arrival_index = result.placement.anchor.route_sample_index + 1
    continuing_route = replace(
        task.route,
        goal_arrival_time_s=float(task.route.sample_times_s[arrival_index]),
    )
    continuing_task = replace(task, route=continuing_route)
    assert np.any(
        np.linalg.norm(
            continuing_route.sampled_controls[arrival_index + 1 :],
            axis=1,
        )
        > 1e-5
    )

    decision = build_teb_decision_state(
        task_template=continuing_task,
        placement_result=result,
        source_base_state=base_state,
        source_oracle_context=OracleContext(
            base_state_id=base_state.state_id,
            dynamic_object_history={},
            dynamic_object_future={},
            dynamic_object_specs={},
        ),
        base_config=base_config,
        teb_config=teb_config,
        seed=31,
    )

    np.testing.assert_allclose(
        decision.suffix_poses_world,
        continuing_route.sampled_poses_world[:32],
        atol=1e-6,
    )


def test_decision_state_keeps_decision_time_when_witness_is_earlier() -> None:
    from src.generation.anchored_human_placement import _sample_robot_poses
    from src.generation.sop05r_teb_decision_state import build_teb_decision_state

    base_config, base_state, teb_config, task, result = _decision_fixture()
    decision_time_s = 0.0
    witness_index = 3
    witness_time_s = decision_time_s + (witness_index - 7) * 0.2
    witness_robot_pose = _sample_robot_poses(
        task,
        base_state,
        np.asarray([witness_time_s], dtype=np.float64),
        dt_s=0.2,
    )[0]
    earlier_witness = CenterlineOcclusionWitness(
        version=teb_config.occlusion.version,
        time_s=witness_time_s,
        sample_index=witness_index,
        robot_position_xy=witness_robot_pose[:2],
        target_position_xy=result.placement.history_poses[witness_index, :2],
        blocking_occluder_id=task.occluders[0].occluder_id,
    )
    blocked = np.zeros(40, dtype=np.bool_)
    blocked[witness_index] = True
    placement_result = replace(
        result,
        witness=earlier_witness,
        visibility=classify_sop05r_seen_then_occluded_history(
            blocked,
            decision_index=7,
            minimum_visible_frames=4,
            minimum_occluded_frames=1,
        ),
    )

    decision = build_teb_decision_state(
        task_template=task,
        placement_result=placement_result,
        source_base_state=base_state,
        source_oracle_context=OracleContext(
            base_state_id=base_state.state_id,
            dynamic_object_history={},
            dynamic_object_future={},
            dynamic_object_specs={},
        ),
        base_config=base_config,
        teb_config=teb_config,
        seed=31,
        target_dynamic_object_id="window-visible-target",
        target_dynamic_object_spec={
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": 0.3},
        },
    )

    assert decision.decision_time_s == decision_time_s
    assert decision.target_visibility_history.tolist() == [True, True, True, False, True, True, True, True]
    assert decision.provenance["occlusion_witness_time_s"] == witness_time_s
    assert decision.base_state.dynamic_object_ids == ("window-visible-target",)


def test_decision_state_rejects_occlude_then_seen_history() -> None:
    from src.generation.sop05r_teb_decision_state import build_teb_decision_state

    base_config, base_state, teb_config, task, result = _decision_fixture()
    blocked = np.zeros(40, dtype=np.bool_)
    blocked[0] = True
    invalid_result = replace(
        result,
        visibility=classify_sop05r_seen_then_occluded_history(
            blocked,
            decision_index=7,
            minimum_visible_frames=4,
            minimum_occluded_frames=1,
        ),
    )

    with pytest.raises(ValueError, match="frozen visibility rule"):
        build_teb_decision_state(
            task_template=task,
            placement_result=invalid_result,
            source_base_state=base_state,
            source_oracle_context=OracleContext(
                base_state_id=base_state.state_id,
                dynamic_object_history={},
                dynamic_object_future={},
                dynamic_object_specs={},
            ),
            base_config=base_config,
            teb_config=teb_config,
            seed=31,
        )


def test_decision_state_drops_visible_actor_without_complete_oracle_context() -> None:
    from src.generation.sop05r_teb_decision_state import build_teb_decision_state

    base_config, base_state, teb_config, task, placement = _decision_fixture()
    supported_id = "m5-recording::complete-human"
    dropped_id = "m5-recording::history-only-human"
    history = np.zeros((8, 3), dtype=np.float32)
    future = np.zeros((32, 3), dtype=np.float32)
    spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.3},
    }
    source_base_state = replace(
        base_state,
        dynamic_object_ids=(supported_id, dropped_id),
        visible_dynamic_object_history={
            supported_id: history,
            dropped_id: history,
        },
        visible_dynamic_object_specs={
            supported_id: spec,
            dropped_id: spec,
        },
    )

    decision = build_teb_decision_state(
        task_template=task,
        placement_result=placement,
        source_base_state=source_base_state,
        source_oracle_context=OracleContext(
            base_state_id=source_base_state.state_id,
            dynamic_object_history={supported_id: history},
            dynamic_object_future={supported_id: future},
            dynamic_object_specs={supported_id: spec},
        ),
        base_config=base_config,
        teb_config=teb_config,
        seed=31,
    )

    assert decision.base_state.dynamic_object_ids == (supported_id,)
    assert set(decision.base_state.visible_dynamic_object_history) == {supported_id}
    assert set(decision.base_state.visible_dynamic_object_specs) == {supported_id}
    assert decision.base_state.metadata["dropped_dynamic_object_ids"] == (dropped_id,)
    assert decision.base_state.metadata["dropped_dynamic_object_reason"] == (
        "missing_complete_source_or_oracle_context"
    )
    assert decision.provenance["dropped_dynamic_object_ids"] == (dropped_id,)
    assert decision.provenance["dropped_dynamic_object_reason"] == (
        "missing_complete_source_or_oracle_context"
    )
    assert source_base_state.dynamic_object_ids == (supported_id, dropped_id)
    np.testing.assert_array_equal(
        source_base_state.visible_dynamic_object_history[dropped_id],
        history,
    )
