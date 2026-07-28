from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.contracts import build_grid_spec
from src.generation.obstacle_first_templates import RectangleObstacle
from src.generation.occluder_sampler import swept_footprint_intersects_occupancy
from src.generation.sop05r_contracts import load_sop05r_config
from src.geometry import RectangleFootprint, inflate_footprint, rasterize_footprint
from src.planning.obstacle_corner_planner import (
    GeometryCachingObstaclePlanner,
    ObstaclePlannerRequest,
    compute_obstacle_route_task_cost,
    plan_obstacle_routes,
)
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _request(
    *,
    obstacle: RectangleObstacle | None = None,
    static_occupancy: np.ndarray | None = None,
    goal: np.ndarray | None = None,
) -> ObstaclePlannerRequest:
    base_config = load_config(ROOT / "configs" / "base.yaml")
    grid = build_grid_spec(base_config)
    obstacle = obstacle or RectangleObstacle(
        obstacle_id="planner-obstacle",
        obstacle_type="wall",
        pose=np.asarray([2.0, 0.0, 0.0], dtype=np.float64),
        length_m=1.0,
        width_m=0.3,
        source="fixture",
    )
    if static_occupancy is None:
        static_occupancy = rasterize_footprint(
            obstacle.footprint, obstacle.pose, grid
        ).astype(np.float32)
    return ObstaclePlannerRequest(
        start_pose=np.zeros(3, dtype=np.float32),
        initial_control=np.asarray([0.4, 0.0], dtype=np.float32),
        static_occupancy=static_occupancy,
        obstacle=obstacle,
        local_goal_world_pose=(
            np.asarray([4.0, 0.0, 0.0], dtype=np.float32)
            if goal is None
            else goal
        ),
        base_config=base_config,
        planner_config=load_sop05r_config(
            ROOT / "configs" / "generator_obstacle_first_train.yaml"
        ).planner,
    )


def test_planner_request_is_target_blind() -> None:
    fields = set(ObstaclePlannerRequest.__dataclass_fields__)
    assert not fields & {
        "target",
        "oracle_context",
        "conflict_point",
        "label",
        "target_trajectory",
    }
    forbidden = {"target", "oracle", "conflict", "label"}
    assert forbidden.isdisjoint(inspect.signature(plan_obstacle_routes).parameters)


def test_corner_planner_returns_fixed_slots_to_one_shared_goal() -> None:
    request = _request()
    grid = build_grid_spec(request.base_config)

    result = plan_obstacle_routes(request)

    assert [decision.slot_id for decision in result.decisions] == [
        "left_near",
        "left_far",
        "right_near",
        "right_far",
        "stop",
    ]
    assert result.direct_path_intersects_inflated_obstacle
    np.testing.assert_array_equal(
        result.shared_goal_world_pose, request.local_goal_world_pose
    )
    assert result.by_slot["stop"].trajectory.metadata["is_stop"] is True
    moving = [route for route in result.routes if route.slot_id != "stop"]
    assert len(moving) >= 2
    assert {route.slot_id.split("_")[0] for route in moving} == {"left", "right"}
    for route in moving:
        np.testing.assert_array_equal(
            route.waypoints_world[-1], request.local_goal_world_pose
        )
        assert route.trajectory.metadata["shared_goal_world_pose"] == [4.0, 0.0, 0.0]
        assert route.waypoints_world.shape[0] >= 3
        assert route.path_length_m > 0.0
        assert 0.25 <= route.represented_obstacle_clearance_m <= 0.75
        assert route.trajectory.poses.shape == (grid.future_steps, 3)
        assert route.trajectory.controls.shape == (grid.future_steps, 2)
        assert route.trajectory.swept_mask.shape == request.static_occupancy.shape
        assert route.trajectory.tta_map.shape == request.static_occupancy.shape


def test_planner_rollouts_obey_dynamics_and_continuous_static_collision_gate() -> None:
    request = _request()
    result = plan_obstacle_routes(request)
    robot = request.base_config["robot"]
    inflated_robot = inflate_footprint(
        RectangleFootprint(robot["length_m"], robot["width_m"]),
        robot["inflation_m"],
    )
    dt_s = request.planner_config.dt_s

    for route in result.routes:
        controls = route.trajectory.controls
        assert np.max(np.abs(controls[:, 0])) <= robot["max_linear_speed_mps"] + 1e-6
        assert np.max(np.abs(controls[:, 1])) <= robot["max_angular_speed_radps"] + 1e-6
        with_initial = np.vstack((request.initial_control, controls))
        assert np.max(np.abs(np.diff(with_initial[:, 0])) / dt_s) <= (
            request.planner_config.max_linear_acceleration_mps2 + 1e-5
        )
        assert np.max(np.abs(np.diff(with_initial[:, 1])) / dt_s) <= (
            request.planner_config.max_angular_acceleration_radps2 + 1e-5
        )
        moving = np.abs(controls[:, 0]) > 1e-5
        if np.any(moving):
            curvature = np.abs(controls[moving, 1] / controls[moving, 0])
            assert np.max(curvature) <= request.planner_config.max_curvature_per_m + 1e-5
        assert not swept_footprint_intersects_occupancy(
            inflated_robot,
            route.poses_world,
            request.static_occupancy,
            grid=build_grid_spec(request.base_config),
        )


def test_long_obstacle_routes_use_two_side_waypoints_and_never_copy_slots() -> None:
    base_config = load_config(ROOT / "configs" / "base.yaml")
    grid = build_grid_spec(base_config)
    obstacle = RectangleObstacle(
        obstacle_id="long-shelf",
        obstacle_type="shelf",
        pose=np.asarray([2.0, 0.0, np.pi / 2.0], dtype=np.float64),
        length_m=2.0,
        width_m=0.4,
        source="fixture",
    )
    static = rasterize_footprint(obstacle.footprint, obstacle.pose, grid).astype(
        np.float32
    )

    result = plan_obstacle_routes(
        _request(
            obstacle=obstacle,
            static_occupancy=static,
            goal=np.asarray([4.5, 0.0, 0.0], dtype=np.float32),
        )
    )

    moving = [route for route in result.routes if route.slot_id != "stop"]
    assert moving
    for route in moving:
        assert route.waypoints_world.shape[0] >= 3
    for left in moving:
        for right in moving:
            if left.slot_id != right.slot_id:
                assert not np.array_equal(left.waypoints_world, right.waypoints_world)
                assert not np.shares_memory(
                    left.trajectory.poses, right.trajectory.poses
                )


def test_near_slots_are_fixed_target_blind_yield_alternatives() -> None:
    result = plan_obstacle_routes(_request())

    for side in ("left", "right"):
        near = result.by_slot[f"{side}_near"]
        far = result.by_slot[f"{side}_far"]
        assert near.trajectory.metadata["planner_speed_scale"] == 0.55
        assert far.trajectory.metadata["planner_speed_scale"] == 1.0
        assert np.sum(near.trajectory.controls[:, 0]) < np.sum(
            far.trajectory.controls[:, 0]
        )
        assert not np.array_equal(
            near.trajectory.controls,
            far.trajectory.controls,
        )
        assert not np.array_equal(near.poses_world, far.poses_world)


def test_blocked_side_is_reported_and_not_filled_from_opposite_side() -> None:
    request = _request()
    grid = build_grid_spec(request.base_config)
    blocker = rasterize_footprint(
        RectangleFootprint(2.4, 0.8),
        np.asarray([2.0, 1.25, 0.0], dtype=np.float64),
        grid,
    )
    blocked = np.asarray((request.static_occupancy != 0) | blocker, dtype=np.float32)

    result = plan_obstacle_routes(
        replace(request, static_occupancy=blocked)
    )

    left_decisions = [
        decision for decision in result.decisions if decision.slot_id.startswith("left")
    ]
    assert left_decisions
    assert all(not decision.accepted for decision in left_decisions)
    assert all(decision.rejection_reason for decision in left_decisions)
    assert any(
        decision.accepted
        for decision in result.decisions
        if decision.slot_id.startswith("right")
    )
    assert len({route.slot_id for route in result.routes}) == len(result.routes)


def test_lateral_gap_preserves_two_safe_route_sides() -> None:
    request = _request()
    grid = build_grid_spec(request.base_config)
    bearing = np.deg2rad(-30.0)
    task_direction = np.asarray(
        [np.cos(bearing), np.sin(bearing)], dtype=np.float64
    )
    task_normal = np.asarray(
        [-task_direction[1], task_direction[0]], dtype=np.float64
    )
    goal_distance = 2.3
    obstacle = RectangleObstacle(
        obstacle_id="offset-alternative-wall",
        obstacle_type="wall",
        pose=np.asarray(
                [
                    *(0.6 * goal_distance * task_direction + 1.02 * task_normal),
                bearing + np.pi / 2.0,
            ],
            dtype=np.float64,
        ),
        length_m=1.2,
        width_m=0.25,
        source="fixture",
    )
    static = rasterize_footprint(obstacle.footprint, obstacle.pose, grid).astype(
        np.float32
    )
    result = plan_obstacle_routes(
        replace(
            request,
            initial_control=np.zeros(2, dtype=np.float32),
            static_occupancy=static,
            obstacle=obstacle,
            local_goal_world_pose=np.asarray(
                [*(goal_distance * task_direction), bearing], dtype=np.float32
            ),
        )
    )

    assert result.direct_path_intersects_inflated_obstacle
    assert {route.slot_id for route in result.routes if route.slot_id != "stop"} == {
        "left_near",
        "left_far",
        "right_near",
        "right_far",
    }
    clearance_range = request.planner_config.represented_obstacle_clearance_range_m
    assert all(
        clearance_range[0]
        <= route.represented_obstacle_clearance_m
        <= clearance_range[1]
        for route in result.routes
        if route.slot_id != "stop"
    )


def test_direct_path_clear_request_keeps_stop_and_rejects_moving_slots() -> None:
    request = _request(goal=np.asarray([0.0, 3.0, np.pi / 2], dtype=np.float32))

    result = plan_obstacle_routes(request)

    assert not result.direct_path_intersects_inflated_obstacle
    assert set(result.by_slot) == {"stop"}
    assert all(
        decision.rejection_reason == "direct_path_clear"
        for decision in result.decisions
        if decision.slot_id != "stop"
    )


def test_path_length_remains_the_dominant_task_score_term() -> None:
    planner = _request().planner_config

    short_with_worst_secondary = compute_obstacle_route_task_cost(
        path_length_m=2.0,
        terminal_heading_error_rad=np.pi,
        normalized_smoothness=1.0,
        planner_config=planner,
    )
    longer_with_best_secondary = compute_obstacle_route_task_cost(
        path_length_m=2.0 + planner.path_length_normalizer_m,
        terminal_heading_error_rad=0.0,
        normalized_smoothness=0.0,
        planner_config=planner,
    )

    assert short_with_worst_secondary < longer_with_best_secondary


def test_planner_is_deterministic() -> None:
    request = _request()
    first = plan_obstacle_routes(request)
    second = plan_obstacle_routes(request)

    assert first.decisions == second.decisions
    assert [route.slot_id for route in first.routes] == [
        route.slot_id for route in second.routes
    ]
    for left, right in zip(first.routes, second.routes, strict=True):
        np.testing.assert_array_equal(left.trajectory.poses, right.trajectory.poses)
        np.testing.assert_array_equal(left.trajectory.controls, right.trajectory.controls)
        np.testing.assert_array_equal(left.waypoints_world, right.waypoints_world)


def test_geometry_cache_reuses_plan_and_rebinds_template_local_ids() -> None:
    request = _request()
    calls = []

    def counting_planner(candidate: ObstaclePlannerRequest):
        calls.append(candidate.obstacle.obstacle_id)
        return plan_obstacle_routes(candidate)

    planner = GeometryCachingObstaclePlanner(counting_planner)
    repeated_obstacle = replace(request.obstacle, obstacle_id="planner-obstacle-repeat")

    first = planner(request)
    repeated = planner(replace(request, obstacle=repeated_obstacle))

    assert calls == ["planner-obstacle"]
    assert planner.miss_count == 1
    assert planner.hit_count == 1
    assert [route.slot_id for route in first.routes] == [
        route.slot_id for route in repeated.routes
    ]
    for original, rebound in zip(first.routes, repeated.routes, strict=True):
        assert original.trajectory.trajectory_id == (
            f"sop05r::planner-obstacle::{original.slot_id}"
        )
        assert rebound.trajectory.trajectory_id == (
            f"sop05r::planner-obstacle-repeat::{rebound.slot_id}"
        )
        np.testing.assert_array_equal(
            original.trajectory.swept_mask,
            rebound.trajectory.swept_mask,
        )
        np.testing.assert_array_equal(original.poses_world, rebound.poses_world)

    planner(
        replace(
            request,
            local_goal_world_pose=np.asarray([4.2, 0.0, 0.0], dtype=np.float32),
        )
    )
    assert planner.miss_count == 2
