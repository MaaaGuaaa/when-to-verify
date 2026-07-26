from __future__ import annotations

from dataclasses import replace

import numpy as np

import src.generation.sop05r_teb_event_sampler as sampler_module
from src.contracts import build_grid_spec, validate_base_state, validate_oracle_world
from src.datasets.long_snippet_library import LongMotionSnippet
from src.generation.anchored_human_placement import (
    AnchoredHumanPlacement,
    AnchoredPlacementResult,
    CenterlineOcclusionWitness,
    CollisionAnchor,
)
from src.generation.history_visibility import (
    classify_sop05r_seen_then_occluded_history,
)


def _mother_fixture():
    from tests.test_sop05r_teb_decision_state import _decision_fixture

    base_config, base_state, teb_config, task, _ = _decision_fixture()
    decision_time_s = 0.0
    route_anchor_index = 10
    collision_time_s = float(task.route.sample_times_s[route_anchor_index])
    snippet_anchor_index = 7 + int(round(collision_time_s / 0.2))
    collision_xy = task.route.sampled_poses_world[route_anchor_index, :2].astype(
        np.float64
    )
    crossing = np.asarray(
        [
            -np.cos(task.route.sampled_poses_world[route_anchor_index, 2]),
            -np.sin(task.route.sampled_poses_world[route_anchor_index, 2]),
        ],
        dtype=np.float64,
    )
    speed = 1.5
    positions = np.asarray(
        [
            collision_xy + (index - snippet_anchor_index) * 0.2 * speed * crossing
            for index in range(40)
        ],
        dtype=np.float32,
    )
    headings = np.full(
        40,
        np.arctan2(crossing[1], crossing[0]),
        dtype=np.float32,
    )
    velocities = np.tile((speed * crossing).astype(np.float32), (40, 1))
    snippet = LongMotionSnippet(
        snippet_id="m6-crossing-human",
        split="train",
        source_recording_id="m6-recording",
        source_session_id="m6-session",
        source_object_id="m6-recording::human",
        object_type="human",
        footprint={"kind": "circle", "radius_m": 0.1},
        start_timestamp=0.0,
        positions=positions,
        velocities=velocities,
        headings=headings,
        duration_s=7.8,
        mean_speed_mps=speed,
        max_acceleration_mps2=0.0,
        mean_abs_curvature_per_m=0.0,
        provenance={"fixture": "m6"},
    )
    anchor = CollisionAnchor(
        route_sample_index=route_anchor_index,
        route_time_s=collision_time_s,
        world_position_xy=collision_xy,
        snippet_anchor_index=snippet_anchor_index,
        snippet_time_s=collision_time_s,
    )
    placement = AnchoredHumanPlacement(
        source_snippet_id=snippet.snippet_id,
        anchor=anchor,
        rotation_rad=0.0,
        translation_xy_m=np.zeros(2),
        spatial_scale=1.0,
        temporal_scale=1.0,
        history_poses=np.column_stack((positions[:8], headings[:8])).astype(np.float32),
        current_pose=np.r_[positions[7], headings[7]].astype(np.float32),
        future_poses=np.column_stack((positions[8:], headings[8:])).astype(np.float32),
        provenance={"fixture": "m6"},
    )
    witness = CenterlineOcclusionWitness(
        version=teb_config.occlusion.version,
        time_s=decision_time_s,
        sample_index=7,
        robot_position_xy=base_state.robot_history[-1, :2],
        target_position_xy=placement.current_pose[:2],
        blocking_occluder_id=task.occluders[0].occluder_id,
    )
    placement_result = AnchoredPlacementResult(
        placement=placement,
        witness=witness,
        visibility=classify_sop05r_seen_then_occluded_history(
            np.asarray(
                [False, False, False, False, False, False, False, True]
                + [False] * 32,
                dtype=np.bool_,
            ),
            decision_index=7,
            minimum_visible_frames=4,
            minimum_occluded_frames=1,
        ),
        attempted_candidates=1,
        rejection_counts={},
    )
    oracle_context = __import__(
        "src.contracts", fromlist=["OracleContext"]
    ).OracleContext(
        base_state_id=base_state.state_id,
        dynamic_object_history={},
        dynamic_object_future={},
        dynamic_object_specs={},
    )
    return (
        base_config,
        base_state,
        oracle_context,
        teb_config,
        task,
        placement_result,
        snippet,
    )


def test_m6_constructs_deterministic_one_route_continuous_collision_mother() -> None:
    from src.generation.sop05r_teb_event_sampler import build_sop05r_teb_mother

    inputs = _mother_fixture()
    first = build_sop05r_teb_mother(
        base_config=inputs[0],
        source_base_state=inputs[1],
        source_oracle_context=inputs[2],
        teb_config=inputs[3],
        task_template=inputs[4],
        placement_result=inputs[5],
        snippet=inputs[6],
        seed=43,
    )
    second = build_sop05r_teb_mother(
        base_config=inputs[0],
        source_base_state=inputs[1],
        source_oracle_context=inputs[2],
        teb_config=inputs[3],
        task_template=inputs[4],
        placement_result=inputs[5],
        snippet=inputs[6],
        seed=43,
    )

    assert first.mother is not None, first.rejection_reason
    assert second.mother is not None
    mother = first.mother
    grid = build_grid_spec(inputs[0])
    validate_base_state(mother.decision_state.base_state, grid)
    validate_oracle_world(mother.event.world, grid)
    assert mother.event.generated_event_id == second.mother.event.generated_event_id
    assert mother.event.world.world_id == second.mother.event.world.world_id
    assert mother.collision.first_collision_time_after_decision_s >= 1.2
    assert mother.collision.first_collision_time_after_decision_s <= 6.4
    assert mother.trajectory_record.nominal_trajectory.poses.shape == (32, 3)
    assert mother.trajectory_record.full_route is inputs[4].route
    assert not hasattr(mother.trajectory_record, "alternative_trajectory_ids")


def test_m6_persists_geometry_evaluated_future_visibility(
    monkeypatch,
) -> None:
    inputs = _mother_fixture()
    blocked = np.asarray(
        [False, False, False, False, False, False, False, True]
        + [True, False, True] + [False] * 29,
        dtype=np.bool_,
    )

    def controlled_blocking(
        robot_positions_xy,
        target_positions_xy,
        occluders,
        *,
        epsilon_m,
    ):
        assert np.asarray(robot_positions_xy).shape == (40, 2)
        assert np.asarray(target_positions_xy).shape == (40, 2)
        assert occluders
        assert epsilon_m == inputs[3].occlusion.centerline_intersection_epsilon_m
        return blocked, tuple(None for _ in range(40))

    monkeypatch.setattr(
        sampler_module,
        "synchronized_centerline_blocking",
        controlled_blocking,
        raising=False,
    )
    evaluation = sampler_module.build_sop05r_teb_mother(
        base_config=inputs[0],
        source_base_state=inputs[1],
        source_oracle_context=inputs[2],
        teb_config=inputs[3],
        task_template=inputs[4],
        placement_result=inputs[5],
        snippet=inputs[6],
        seed=43,
    )

    assert evaluation.mother is not None, evaluation.rejection_reason
    np.testing.assert_array_equal(
        evaluation.mother.event.target_visibility_history,
        ~blocked[:8],
    )
    np.testing.assert_array_equal(
        evaluation.mother.event.visibility_sequence,
        ~blocked[8:],
    )


def test_m6_rejects_suffix_control_dynamics_mismatch() -> None:
    from src.generation.sop05r_teb_event_sampler import build_sop05r_teb_mother

    inputs = _mother_fixture()
    invalid_controls = np.asarray(inputs[4].route.sampled_controls).copy()
    invalid_controls[0, 0] += 0.5
    invalid_task = replace(
        inputs[4],
        route=replace(inputs[4].route, sampled_controls=invalid_controls),
    )

    evaluation = build_sop05r_teb_mother(
        base_config=inputs[0],
        source_base_state=inputs[1],
        source_oracle_context=inputs[2],
        teb_config=inputs[3],
        task_template=invalid_task,
        placement_result=inputs[5],
        snippet=inputs[6],
        seed=43,
    )

    assert evaluation.mother is None
    assert evaluation.rejection_reason == "teb_dynamics_limit"
