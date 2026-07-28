from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import src.generation.sop05r_event_sampler as sampler_module
from src.contracts import BaseState, OracleContext, build_grid_spec
from src.generation.dynamic_object_transplant import TransplantedDynamicObject
from src.generation.event_target_motion_shard import compute_footprint_spec_digest
from src.generation.obstacle_first_templates import (
    ObstacleTargetTemplate,
    RectangleObstacle,
)
from src.generation.sop05r_contracts import load_sop05r_config
from src.generation.sop05r_event_sampler import (
    compute_continuous_collision_evidence,
    evaluate_obstacle_first_template,
)
from src.geometry import (
    CircleFootprint,
    rasterize_footprint,
    signed_clearance,
)
from src.planning.obstacle_corner_planner import (
    ObstaclePlanDecision,
    ObstaclePlannerRequest,
    plan_obstacle_routes,
)
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    base_config = load_config(ROOT / "configs" / "base.yaml")
    config = load_sop05r_config(
        ROOT / "configs" / "generator_obstacle_first_train.yaml"
    )
    grid = build_grid_spec(base_config)
    base_state = BaseState(
        state_id="train-sop05r-event-base",
        split="train",
        recording_id="train-sop05r-event-recording",
        dynamic_object_ids=(),
        timestamp=8.0,
        robot_history=np.zeros((grid.history_steps, 3), dtype=np.float32),
        robot_state=np.asarray([0.4, 0.0], dtype=np.float32),
        visible_dynamic_object_history={},
        visible_dynamic_object_specs={},
        static_map_local=np.zeros((grid.height, grid.width), dtype=np.float32),
        metadata={"fixture": "sop05r-event"},
    )
    context = OracleContext(
        base_state_id=base_state.state_id,
        dynamic_object_history={},
        dynamic_object_future={},
        dynamic_object_specs={},
        metadata={"future_dt_s": 0.2},
    )
    obstacle = RectangleObstacle(
        obstacle_id="sop05r-event-wall",
        obstacle_type="wall",
        pose=np.asarray([1.5, 0.0, np.pi / 2.0], dtype=np.float64),
        length_m=1.0,
        width_m=0.25,
        source="fixture",
    )
    obstacle_mask = rasterize_footprint(
        obstacle.footprint, obstacle.pose, grid
    )
    static = obstacle_mask.astype(np.float32)
    goal = np.asarray([2.4, 0.0, 0.0], dtype=np.float32)
    plan = plan_obstacle_routes(
        ObstaclePlannerRequest(
            start_pose=base_state.robot_history[-1],
            initial_control=base_state.robot_state,
            static_occupancy=static,
            obstacle=obstacle,
            local_goal_world_pose=goal,
            base_config=base_config,
            planner_config=config.planner,
        )
    )
    collision_route = plan.by_slot["left_near"]
    conflict_time_s = 2.2
    start = np.asarray([2.1, 0.0], dtype=np.float64)
    control = np.asarray([2.1, 1.9], dtype=np.float64)
    conflict = collision_route.poses_world[10, :2].astype(np.float64)
    future_rows = []
    for index in range(grid.future_steps):
        time_s = (index + 1) * 0.2
        if time_s <= conflict_time_s + 1e-12:
            fraction = time_s / conflict_time_s
            position = (
                (1.0 - fraction) ** 2 * start
                + 2.0 * (1.0 - fraction) * fraction * control
                + fraction**2 * conflict
            )
            velocity = (
                2.0 * (1.0 - fraction) * (control - start)
                + 2.0 * fraction * (conflict - control)
            ) / conflict_time_s
        else:
            velocity = 2.0 * (conflict - control) / conflict_time_s
            position = conflict + (time_s - conflict_time_s) * velocity
        future_rows.append(
            [position[0], position[1], np.arctan2(velocity[1], velocity[0])]
        )
    future = np.asarray(future_rows, dtype=np.float32)
    initial_velocity = 2.0 * (control - start) / conflict_time_s
    history_rows = []
    for index in range(grid.history_steps):
        time_s = (index - 7) * 0.2
        position = start + time_s * initial_velocity
        history_rows.append(
            [
                position[0],
                position[1],
                np.arctan2(initial_velocity[1], initial_velocity[0]),
            ]
        )
    history = np.asarray(history_rows, dtype=np.float32)
    footprint_spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.3},
    }
    target = TransplantedDynamicObject(
        target_dynamic_object_id="generated::human::sop05r-event-target",
        source_object_id="train-source::human-1",
        snippet_id="train-sop05r-event-snippet",
        object_type="human",
        footprint_spec=footprint_spec,
        footprint_spec_digest=compute_footprint_spec_digest(footprint_spec),
        history_poses=history,
        current_pose=history[-1].copy(),
        future_poses=future,
        provenance={
            "source_split": "train",
            "source_recording_id": "train-source",
            "source_session_id": "train-source-session",
            "source_object_id": "train-source::human-1",
            "source_snippet_id": "train-sop05r-event-snippet",
            "time_scale": 1.0,
        },
    )
    template = ObstacleTargetTemplate(
        template_id="template-sop05r-event",
        schedule_rank=(0, 0, 1, 0, 1, 0),
        obstacle=obstacle,
        obstacle_mask=obstacle_mask,
        static_occupancy=static,
        target=target,
        target_time_scale=1.0,
        goal_bearing_rad=0.0,
        goal_distance_m=2.4,
        local_goal_world_pose=goal,
        provenance={"fixture": "sop05r-event"},
    )
    return base_config, config, base_state, context, template, plan


def _force_seen_then_occluded(monkeypatch) -> None:
    monkeypatch.setattr(
        sampler_module,
        "_target_history_visibility",
        lambda **kwargs: np.asarray(
            [True, True, True, True, False, False, False, False],
            dtype=np.bool_,
        ),
    )


def test_event_builder_calls_planner_before_target_join(monkeypatch) -> None:
    base_config, config, base_state, context, template, expected_plan = _fixture()
    _force_seen_then_occluded(monkeypatch)
    calls = []

    def planner(request):
        assert set(request.__dataclass_fields__).isdisjoint(
            {"target", "oracle_context", "conflict_point", "label"}
        )
        calls.append("planner")
        return expected_plan

    result = evaluate_obstacle_first_template(
        template=template,
        base_state=base_state,
        oracle_context=context,
        base_config=base_config,
        config=config,
        seed=31,
        planner=planner,
    )

    assert calls == ["planner"]
    assert result.rejection_reason is None
    assert result.mother is not None


def test_event_builder_selects_internal_collision_and_same_goal_alternatives(
    monkeypatch,
) -> None:
    base_config, config, base_state, context, template, _ = _fixture()
    _force_seen_then_occluded(monkeypatch)

    evaluation = evaluate_obstacle_first_template(
        template=template,
        base_state=base_state,
        oracle_context=context,
        base_config=base_config,
        config=config,
        seed=31,
    )

    assert evaluation.rejection_reason is None
    mother = evaluation.mother
    assert mother is not None
    evidence = mother.collision_evidence
    assert evidence.minimum_clearance_m <= 0.0
    assert 1.0 <= evidence.first_collision_time_s <= 2.2
    assert 0.4 <= evidence.conflict_path_fraction <= 0.7
    assert 1.0 <= evidence.goal_forward_distance_m <= 2.0
    assert mother.planner_result.direct_path_intersects_inflated_obstacle
    assert mother.nominal_route.slot_id == "left_near"
    assert mother.alternative_routes
    assert all(
        np.allclose(
            route.trajectory.metadata["shared_goal_world_pose"],
            [2.4, 0.0, 0.0],
            rtol=0.0,
            atol=1e-6,
        )
        for route in mother.alternative_routes
    )
    assert 0.25 <= mother.nominal_route.represented_obstacle_clearance_m <= 0.75
    assert mother.event.target is not template.target
    selected_target = mother.event.target
    assert "route_contact_alignment" in selected_target.provenance
    assert mother.event.world.dynamic_object_specs[
        selected_target.target_dynamic_object_id
    ] == selected_target.footprint_spec
    assert mother.event.target_visibility_history.tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert mother.trajectory_record.nominal_trajectory_id == (
        mother.nominal_route.trajectory.trajectory_id
    )


def test_continuous_collision_detects_between_frame_contact() -> None:
    robot_poses = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    target_poses = np.asarray(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
    )

    evidence = compute_continuous_collision_evidence(
        robot_footprint=CircleFootprint(0.2),
        robot_poses=robot_poses,
        target_footprint=CircleFootprint(0.2),
        target_poses=target_poses,
        dt_s=1.0,
        spatial_resolution_m=0.05,
    )

    assert evidence.continuous_collision
    assert evidence.minimum_clearance_m <= -0.39
    assert evidence.first_collision_time_s == pytest.approx(0.3, abs=0.06)


def test_route_contact_alignment_preserves_motion_and_uses_footprint_boundary() -> None:
    base_config, _, _, _, template, plan = _fixture()
    route = plan.by_slot["left_near"]
    source = np.vstack(
        (template.target.history_poses, template.target.future_poses)
    )

    aligned = sampler_module.align_target_to_route_contact(
        target=template.target,
        route=route,
        conflict_index=10,
        crossing_direction=np.asarray([0.0, 1.0], dtype=np.float64),
        base_config=base_config,
        template_id=template.template_id,
        seed=31,
    )
    repeated = sampler_module.align_target_to_route_contact(
        target=template.target,
        route=route,
        conflict_index=10,
        crossing_direction=np.asarray([0.0, 1.0], dtype=np.float64),
        base_config=base_config,
        template_id=template.template_id,
        seed=31,
    )

    transformed = np.vstack((aligned.history_poses, aligned.future_poses))
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(transformed[:, :2], axis=0), axis=1),
        np.linalg.norm(np.diff(source[:, :2], axis=0), axis=1),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(transformed, np.vstack(
        (repeated.history_poses, repeated.future_poses)
    ))
    assert aligned.target_dynamic_object_id == repeated.target_dynamic_object_id
    assert aligned.target_dynamic_object_id != template.target.target_dynamic_object_id
    assert aligned.provenance["route_contact_alignment"]["conflict_index"] == 10
    assert aligned.provenance["route_contact_alignment"]["route_id"] == (
        route.trajectory.trajectory_id
    )

    robot_footprint = sampler_module._robot_footprint(base_config)
    target_footprint = CircleFootprint(0.3)
    anchor_clearance = signed_clearance(
        robot_footprint,
        route.poses_world[10],
        target_footprint,
        aligned.future_poses[10],
    )
    center_distance = np.linalg.norm(
        route.poses_world[10, :2] - aligned.future_poses[10, :2]
    )
    assert anchor_clearance < 0.0
    assert anchor_clearance >= -0.021
    assert center_distance > 0.5


def test_route_contact_alignment_decouples_motion_from_contact_normal() -> None:
    base_config, _, _, _, template, plan = _fixture()
    route = plan.by_slot["left_near"]

    aligned = sampler_module.align_target_to_route_contact(
        target=template.target,
        route=route,
        conflict_index=10,
        crossing_direction=np.asarray([0.0, 1.0], dtype=np.float64),
        contact_normal=np.asarray([-1.0, 0.0], dtype=np.float64),
        base_config=base_config,
        template_id=template.template_id,
        seed=31,
    )

    motion = aligned.future_poses[10, :2] - aligned.current_pose[:2]
    motion /= np.linalg.norm(motion)
    np.testing.assert_allclose(motion, [0.0, 1.0], rtol=0.0, atol=1e-6)
    robot_to_target = (
        route.poses_world[10, :2] - aligned.future_poses[10, :2]
    )
    robot_to_target /= np.linalg.norm(robot_to_target)
    np.testing.assert_allclose(
        robot_to_target,
        [-1.0, 0.0],
        rtol=0.0,
        atol=1e-6,
    )
    assert aligned.provenance["route_contact_alignment"]["contact_normal"] == [
        -1.0,
        0.0,
    ]


def test_route_contact_schedule_is_deterministic_and_bounded_per_path() -> None:
    base_config, config, base_state, _, template, plan = _fixture()

    first = sampler_module.build_route_contact_alignment_schedule(
        template=template,
        planner_result=plan,
        base_state=base_state,
        base_config=base_config,
        config=config,
        seed=31,
    )
    repeated = sampler_module.build_route_contact_alignment_schedule(
        template=template,
        planner_result=plan,
        base_state=base_state,
        base_config=base_config,
        config=config,
        seed=31,
    )

    assert first
    assert [row.identity for row in first] == [row.identity for row in repeated]
    counts = Counter(row.route.slot_id for row in first)
    assert set(counts) == {"left_near", "left_far", "right_near", "right_far"}
    assert all(
        0 < count <= config.generation.max_time_alignments_per_path
        for count in counts.values()
    )
    for slot_id in counts:
        assert len(
            {
                row.conflict_index
                for row in first
                if row.route.slot_id == slot_id
            }
        ) >= 4
    corner_rows = [row for row in first if row.direction_id == "toward_robot"]
    assert corner_rows
    task_direction = template.local_goal_world_pose[:2].astype(np.float64)
    task_direction /= np.linalg.norm(task_direction)
    task_normal = np.asarray(
        [-task_direction[1], task_direction[0]], dtype=np.float64
    )
    expected_offset = (
        np.cos(np.deg2rad(30.0)) * task_direction
        + np.sin(np.deg2rad(30.0)) * task_normal
    )
    np.testing.assert_allclose(
        corner_rows[0].crossing_direction,
        -task_direction,
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        corner_rows[0].contact_normal,
        -expected_offset,
        rtol=0.0,
        atol=1e-9,
    )
    for row, duplicate in zip(first, repeated, strict=True):
        np.testing.assert_array_equal(row.crossing_direction, duplicate.crossing_direction)
        np.testing.assert_array_equal(row.contact_normal, duplicate.contact_normal)
        assert np.linalg.norm(row.crossing_direction) == pytest.approx(1.0)
        assert np.linalg.norm(row.contact_normal) == pytest.approx(1.0)
        assert (
            config.generation.conflict_time_range_s[0]
            <= row.anchor_time_s
            <= config.generation.conflict_time_range_s[1]
        )


def test_event_builder_aligns_target_only_after_planning(monkeypatch) -> None:
    base_config, config, base_state, context, template, expected_plan = _fixture()
    calls: list[tuple[str, object]] = []
    original_align = sampler_module.align_target_to_route_contact

    def planner(request):
        calls.append(("planner", request))
        return expected_plan

    def align(**kwargs):
        assert calls and calls[0][0] == "planner"
        calls.append(("align", kwargs["route"].slot_id))
        return original_align(**kwargs)

    monkeypatch.setattr(sampler_module, "align_target_to_route_contact", align)
    monkeypatch.setattr(
        sampler_module,
        "_target_history_visibility",
        lambda **kwargs: np.asarray(
            [True, True, True, True, False, False, False, False],
            dtype=np.bool_,
        ),
    )

    evaluation = evaluate_obstacle_first_template(
        template=template,
        base_state=base_state,
        oracle_context=context,
        base_config=base_config,
        config=config,
        seed=31,
        planner=planner,
    )

    assert calls[0][0] == "planner"
    assert any(name == "align" for name, _ in calls[1:])
    assert evaluation.rejection_reason is None
    assert evaluation.mother is not None
    selected = evaluation.mother.event.target
    assert selected is not template.target
    alignment = selected.provenance["route_contact_alignment"]
    assert alignment["route_id"] == evaluation.mother.nominal_route.trajectory.trajectory_id
    assert alignment["conflict_index"] in range(15)
    assert evaluation.evidence["target_alignment_attempt_count"] == sum(
        name == "align" for name, _ in calls
    )
    assert all(
        count <= config.generation.max_time_alignments_per_path
        for count in evaluation.evidence["target_alignment_attempts_by_route"].values()
    )


def test_route_contact_schedule_always_probes_latest_conflict_time() -> None:
    base_config, config, base_state, _, template, plan = _fixture()
    early_biased = replace(
        config,
        generation=replace(
            config.generation,
            conflict_path_fraction_range=(0.1, 0.3),
            goal_beyond_conflict_range_m=(1.8, 2.4),
        ),
    )

    schedule = sampler_module.build_route_contact_alignment_schedule(
        template=template,
        planner_result=plan,
        base_state=base_state,
        base_config=base_config,
        config=early_biased,
        seed=31,
    )

    dt_s = float(base_config["bev"]["future_dt_s"])
    latest_index = int(
        np.floor(early_biased.generation.conflict_time_range_s[1] / dt_s)
    ) - 1
    for slot_id in ("left_near", "left_far", "right_near", "right_far"):
        assert latest_index in {
            row.conflict_index
            for row in schedule
            if row.route.slot_id == slot_id
        }


def test_event_builder_rejects_missing_noncolliding_alternative(monkeypatch) -> None:
    base_config, config, base_state, context, template, plan = _fixture()
    _force_seen_then_occluded(monkeypatch)
    nominal = plan.by_slot["left_near"]
    stop = plan.by_slot["stop"]
    decisions = tuple(
        ObstaclePlanDecision(
            slot_id=slot_id,
            accepted=slot_id in {"left_near", "stop"},
            rejection_reason=(
                None if slot_id in {"left_near", "stop"} else "fixture_blocked"
            ),
        )
        for slot_id in config.planner.candidate_slot_ids
    )
    no_alternative_plan = replace(
        plan,
        routes=(nominal, stop),
        decisions=decisions,
        rejection_counts={"fixture_blocked": 3},
    )

    evaluation = evaluate_obstacle_first_template(
        template=template,
        base_state=base_state,
        oracle_context=context,
        base_config=base_config,
        config=config,
        seed=31,
        planner=lambda request: no_alternative_plan,
    )

    assert evaluation.mother is None
    assert evaluation.rejection_reason == "no_same_goal_alternative"


def test_event_builder_rejects_ineligible_history_metadata(monkeypatch) -> None:
    base_config, config, base_state, context, template, _ = _fixture()
    monkeypatch.setattr(
        sampler_module,
        "_target_history_visibility",
        lambda **kwargs: np.asarray(
            [False, False, True, False, False, False, False, False],
            dtype=np.bool_,
        ),
    )

    evaluation = evaluate_obstacle_first_template(
        template=template,
        base_state=base_state,
        oracle_context=context,
        base_config=base_config,
        config=config,
        seed=31,
    )

    assert evaluation.mother is None
    assert evaluation.rejection_reason == "history_ineligible"


def test_event_identity_is_deterministic_and_binds_nominal_selection(monkeypatch) -> None:
    base_config, config, base_state, context, template, plan = _fixture()
    _force_seen_then_occluded(monkeypatch)
    kwargs = dict(
        template=template,
        base_state=base_state,
        oracle_context=context,
        base_config=base_config,
        config=config,
        seed=31,
    )

    first = evaluate_obstacle_first_template(**kwargs)
    repeated = evaluate_obstacle_first_template(**kwargs)

    assert first.mother.event.generated_event_id == repeated.mother.event.generated_event_id
    assert first.mother.event.world.world_id == repeated.mother.event.world.world_id
    alternate_order = replace(
        plan,
        routes=tuple(
            plan.by_slot[slot]
            for slot in ("left_far", "left_near", "right_near", "right_far", "stop")
        ),
    )
    with pytest.raises(ValueError, match="slot order"):
        evaluate_obstacle_first_template(
            **kwargs,
            planner=lambda request: alternate_order,
        )
