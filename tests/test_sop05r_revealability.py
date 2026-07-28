from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.contracts import ARRAY_DTYPE, build_grid_spec
from src.generation.obstacle_first_templates import RectangleObstacle
from src.generation.sop05r_contracts import load_sop05r_config
from src.generation.sop05r_event_sampler import evaluate_obstacle_first_template
from src.generation.sop05r_revealability import (
    Sop05rRevealabilityRequest,
    build_active_revealability_request,
    evaluate_active_revealability,
)
from src.geometry import rasterize_footprint
from src.planning.obstacle_corner_planner import plan_obstacle_routes
from src.planning.verification_actions import (
    CANONICAL_ACTION_IDS,
    load_verification_actions,
)
from src.utils.config import load_config
from tests.test_sop05r_event_sampler import _fixture as event_fixture


ROOT = Path(__file__).resolve().parents[1]


def _request() -> Sop05rRevealabilityRequest:
    base_config = deepcopy(load_config(ROOT / "configs" / "base.yaml"))
    base_config["bev"]["range_m"] = 8.0
    base_config["bev"]["size"] = 80
    obstacle = RectangleObstacle(
        obstacle_id="sop05r-revealability-obstacle",
        obstacle_type="wall",
        pose=np.asarray([0.8, -1.5, 0.0], dtype=np.float64),
        length_m=1.2,
        width_m=0.25,
        source="fixture",
    )
    static = rasterize_footprint(
        obstacle.footprint, obstacle.pose, build_grid_spec(base_config)
    ).astype(ARRAY_DTYPE)
    target_id = "hidden_person"
    target_current = np.asarray([1.6, -2.0, 0.0], dtype=ARRAY_DTYPE)
    conflict_index = 10
    dynamic_future = np.tile(target_current, (15, 1)).astype(ARRAY_DTYPE)
    conflict_target_pose = np.asarray([0.5, -2.0, 0.0], dtype=ARRAY_DTYPE)
    dynamic_future[4 : conflict_index + 1] = np.linspace(
        dynamic_future[3],
        conflict_target_pose,
        conflict_index - 3,
        dtype=ARRAY_DTYPE,
    )
    dynamic_future[conflict_index + 1 :] = conflict_target_pose
    return Sop05rRevealabilityRequest(
        event_id="event-sop05r-revealability-fixture",
        robot_pose=np.zeros(3, dtype=ARRAY_DTYPE),
        robot_state=np.zeros(2, dtype=ARRAY_DTYPE),
        static_occupancy=static,
        obstacle=obstacle,
        local_goal_world_pose=np.asarray([1.6, -3.0, 0.0], dtype=ARRAY_DTYPE),
        conflict_point=conflict_target_pose[:2].astype(np.float64),
        conflict_target_pose=conflict_target_pose.astype(np.float64),
        conflict_time_s=2.2,
        target_object_id=target_id,
        dynamic_current_poses={target_id: target_current},
        dynamic_future_poses={target_id: dynamic_future},
        dynamic_specs={
            target_id: {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.3},
            }
        },
        base_config=base_config,
        config=load_sop05r_config(
            ROOT / "configs" / "generator_obstacle_first_train.yaml"
        ),
        action_library=load_verification_actions(
            ROOT / "configs" / "verification_actions.yaml"
        ),
        sensor_range_m=4.0,
    )


def test_moving_action_reveals_before_matched_wait_and_replans_same_goal() -> None:
    requests = []

    def planner(request):
        requests.append(request)
        return plan_obstacle_routes(request)

    request = _request()
    audit = evaluate_active_revealability(request, planner=planner)

    assert tuple(row.action_id for row in audit.actions) == CANONICAL_ACTION_IDS
    row = audit.by_action["arc_left_45"]
    assert row.action_feasibility.feasible
    assert row.matched_wait_feasibility.feasible
    assert row.first_visible_time_s is not None
    if row.matched_wait_visible_time_s is None:
        assert row.visibility_lead_is_censored
    else:
        assert row.first_visible_time_s <= row.matched_wait_visible_time_s - 0.2
        assert row.visibility_lead_s >= 0.2
    assert row.visibility_lead_lower_bound_s >= 0.2
    assert row.post_visibility_margin_s >= 0.4
    assert row.action_trace.times_s[-1] == row.matched_wait_trace.times_s[-1]
    np.testing.assert_array_equal(
        row.replan_goal_world_pose, request.local_goal_world_pose
    )
    assert row.post_action_route_ids
    assert row.post_action_avoiding_route_ids
    assert row.post_action_avoids_original_conflict
    assert row.active_revealable
    assert "arc_left_45" in audit.active_revealable_action_ids
    assert "stop_scan" not in audit.active_revealable_action_ids

    assert requests
    for planner_request in requests:
        np.testing.assert_array_equal(
            planner_request.local_goal_world_pose,
            request.local_goal_world_pose,
        )
        assert set(planner_request.__dataclass_fields__).isdisjoint(
            {"target", "oracle", "conflict_point", "label"}
        )


def test_action_feasibility_checks_complete_dynamic_motion() -> None:
    request = _request()
    current = dict(request.dynamic_current_poses)
    current["crossing_context"] = np.asarray(
        [1.5, 1.5, 0.0], dtype=ARRAY_DTYPE
    )
    future = {
        object_id: poses.copy()
        for object_id, poses in request.dynamic_future_poses.items()
    }
    future["crossing_context"] = np.tile(
        current["crossing_context"], (15, 1)
    ).astype(ARRAY_DTYPE)
    future["crossing_context"][:4] = np.asarray(
        [
            [1.2, 1.2, 0.0],
            [0.8, 0.7, 0.0],
            [0.5, 0.25, 0.0],
            [0.4, 0.07, 0.0],
        ],
        dtype=ARRAY_DTYPE,
    )
    specs = {
        object_id: dict(spec) for object_id, spec in request.dynamic_specs.items()
    }
    specs["crossing_context"] = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.3},
    }
    blocked = replace(
        request,
        dynamic_current_poses=current,
        dynamic_future_poses=future,
        dynamic_specs=specs,
    )

    audit = evaluate_active_revealability(blocked)

    row = audit.by_action["arc_left_45"]
    assert not row.action_feasibility.feasible
    assert row.action_feasibility.reason == "dynamic_collision"
    assert not row.active_revealable


def test_real_mother_request_uses_target_and_context_current_pose_seam() -> None:
    base_config, config, base_state, context, template, _ = event_fixture()
    evaluation = evaluate_obstacle_first_template(
        template=template,
        base_state=base_state,
        oracle_context=context,
        base_config=base_config,
        config=config,
        seed=31,
    )
    assert evaluation.mother is not None
    actions = load_verification_actions(
        ROOT / "configs" / "verification_actions.yaml"
    )

    request = build_active_revealability_request(
        mother=evaluation.mother,
        base_state=base_state,
        oracle_context=context,
        base_config=base_config,
        config=config,
        action_library=actions,
    )

    target_id = evaluation.mother.event.target.target_dynamic_object_id
    np.testing.assert_array_equal(
        request.dynamic_current_poses[target_id],
        evaluation.mother.event.target.current_pose,
    )
    for object_id in context.dynamic_object_history:
        np.testing.assert_array_equal(
            request.dynamic_current_poses[object_id],
            context.dynamic_object_history[object_id][-1],
        )
    np.testing.assert_array_equal(
        request.local_goal_world_pose,
        evaluation.mother.template.local_goal_world_pose,
    )


def test_revealability_metadata_is_deterministic_and_stop_is_never_active() -> None:
    request = _request()

    first = evaluate_active_revealability(request)
    repeated = evaluate_active_revealability(request)

    assert first.as_metadata() == repeated.as_metadata()
    assert first.by_action["stop_scan"].is_stop
    assert not first.by_action["stop_scan"].active_revealable
    assert first.active_revealable == bool(first.active_revealable_action_ids)
    for left, right in zip(first.actions, repeated.actions, strict=True):
        np.testing.assert_array_equal(left.action_trace.poses, right.action_trace.poses)
        np.testing.assert_array_equal(
            left.matched_wait_trace.poses, right.matched_wait_trace.poses
        )
