from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from src.contracts import build_grid_spec
from src.geometry import (
    CircleOccluder,
    RectangleOccluder,
    point_signed_distance,
    rasterize_occluder,
)
from src.generation.sop05r_contracts import load_sop05r_teb_config
from src.planning.differential_drive import integrate_twist
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def base_config() -> dict:
    return load_config(ROOT / "configs/base.yaml")


@pytest.fixture(scope="module")
def planner_config():
    return load_sop05r_teb_config(
        ROOT / "configs/generator_obstacle_first_teb_train.yaml"
    ).planner


def _request(
    base_config: dict,
    planner_config,
    *,
    occluders=(),
    static_occupancy: np.ndarray | None = None,
):
    from src.planning.lightweight_teb import StaticTebRequest

    grid = build_grid_spec(base_config)
    return StaticTebRequest(
        start_pose=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        initial_control=np.asarray([0.0, 0.0], dtype=np.float32),
        local_goal_world_pose=np.asarray([2.4, 0.0, 0.0], dtype=np.float32),
        static_occupancy=(
            np.zeros((grid.height, grid.width), dtype=np.float32)
            if static_occupancy is None
            else static_occupancy
        ),
        occluders=tuple(occluders),
        base_config=base_config,
        planner_config=planner_config,
    )


def _assert_dynamics(route, request, base_config: dict, planner_config) -> None:
    max_speed = float(base_config["robot"]["max_linear_speed_mps"])
    max_omega = float(base_config["robot"]["max_angular_speed_radps"])
    controls = route.sampled_controls
    poses = route.sampled_poses_world
    dt_s = planner_config.route_sample_dt_s

    assert np.max(np.abs(controls[:, 0])) <= max_speed + 1e-6
    assert np.max(np.abs(controls[:, 1])) <= max_omega + 1e-6

    prior_control = request.initial_control
    prior_pose = request.start_pose
    for control, pose in zip(controls, poses, strict=True):
        assert abs(float(control[0] - prior_control[0])) <= (
            planner_config.max_linear_acceleration_mps2 * dt_s + 1e-6
        )
        assert abs(float(control[1] - prior_control[1])) <= (
            planner_config.max_angular_acceleration_radps2 * dt_s + 1e-6
        )
        if abs(float(control[0])) > 1e-6:
            assert abs(float(control[1] / control[0])) <= (
                planner_config.max_curvature_per_m + 1e-6
            )
        expected = integrate_twist(
            prior_pose,
            v=float(control[0]),
            omega=float(control[1]),
            dt_s=dt_s,
        )
        np.testing.assert_allclose(pose, expected, atol=1e-5, rtol=0.0)
        prior_control = control
        prior_pose = pose


def test_static_request_excludes_target_oracle_collision_and_label_channels() -> None:
    from src.planning.lightweight_teb import StaticTebRequest

    forbidden = (
        "target",
        "human",
        "oracle",
        "collision",
        "label",
        "risk",
        "revealability",
    )
    assert all(
        token not in field.name.lower()
        for field in fields(StaticTebRequest)
        for token in forbidden
    )


def test_teb_public_api_is_exported_from_planning_package() -> None:
    from src.planning import PlannedTebRoute, StaticTebRequest, plan_lightweight_teb

    assert PlannedTebRoute.__name__ == "PlannedTebRoute"
    assert StaticTebRequest.__name__ == "StaticTebRequest"
    assert callable(plan_lightweight_teb)


def test_unobstructed_route_has_frozen_full_route_layout_and_dynamics(
    base_config: dict, planner_config
) -> None:
    from src.planning.lightweight_teb import plan_lightweight_teb

    request = _request(base_config, planner_config)
    result = plan_lightweight_teb(request)

    assert result.rejection_reason is None
    assert result.route is not None
    route = result.route
    assert route.band_poses_world.shape == (20, 3)
    assert route.band_interval_dt_s.shape == (19,)
    assert route.sample_times_s.shape == (25,)
    assert route.sampled_poses_world.shape == (25, 3)
    assert route.sampled_controls.shape == (25, 2)
    np.testing.assert_array_equal(route.band_poses_world[0], request.start_pose)
    np.testing.assert_array_equal(route.band_poses_world[-1], request.local_goal_world_pose)
    np.testing.assert_allclose(
        route.sample_times_s,
        np.arange(1, 26, dtype=np.float32) * 0.2,
        rtol=0.0,
        atol=1e-7,
    )
    assert np.all(route.band_interval_dt_s >= 0.1)
    assert np.all(route.band_interval_dt_s <= 0.4)
    assert float(np.sum(route.band_interval_dt_s)) <= 5.0 + 1e-6
    assert route.goal_arrival_time_s <= 5.0
    np.testing.assert_allclose(
        route.sampled_poses_world[-1],
        request.local_goal_world_pose,
        rtol=0.0,
        atol=planner_config.goal_position_tolerance_m,
    )
    assert not route.sampled_poses_world.flags.writeable
    assert not route.sampled_controls.flags.writeable
    _assert_dynamics(route, request, base_config, planner_config)


@pytest.mark.parametrize(
    "occluder",
    [
        RectangleOccluder(
            "wall",
            "wall",
            np.asarray([1.2, 0.87, 0.0], dtype=np.float64),
            1.4,
            0.35,
        ),
        CircleOccluder(
            "tree",
            "tree_trunk",
            np.asarray([1.2, 1.0], dtype=np.float64),
            0.35,
        ),
    ],
)
def test_route_bypasses_represented_occluder_with_clearance(
    base_config: dict, planner_config, occluder
) -> None:
    from src.planning.lightweight_teb import plan_lightweight_teb

    request = _request(base_config, planner_config, occluders=(occluder,))
    result = plan_lightweight_teb(request)

    assert result.route is not None, result.diagnostics
    assert len(result.diagnostics.candidates) == 1
    assert result.diagnostics.candidates[0].initialization_id == "straight"
    clearances = point_signed_distance(
        occluder, result.route.sampled_poses_world[:, :2]
    )
    robot_radius = 0.5 * float(
        np.hypot(
            base_config["robot"]["length_m"],
            base_config["robot"]["width_m"],
        )
    ) + float(base_config["robot"]["inflation_m"])
    direct_line = np.column_stack(
        (
            np.linspace(0.0, 2.4, 601, dtype=np.float64),
            np.zeros(601, dtype=np.float64),
        )
    )
    direct_clearance = float(
        np.min(point_signed_distance(occluder, direct_line) - robot_radius)
    )
    assert 0.0 < direct_clearance < (
        planner_config.represented_occluder_clearance_range_m[0]
    )
    assert result.diagnostics.candidates[0].rejection_reason is None
    route_clearance = float(np.min(clearances - robot_radius))
    assert route_clearance >= (
        planner_config.represented_occluder_clearance_range_m[0] - 1e-5
    )
    assert route_clearance <= (
        planner_config.represented_occluder_clearance_range_m[0] + 0.08
    )
    _assert_dynamics(result.route, request, base_config, planner_config)


def test_route_bypasses_l_shaped_occluder_components(
    base_config: dict, planner_config
) -> None:
    from src.planning.lightweight_teb import plan_lightweight_teb

    horizontal_arm = RectangleOccluder(
        "l-wall-horizontal",
        "wall",
        np.asarray([1.2, 0.87, 0.0], dtype=np.float64),
        1.4,
        0.35,
    )
    vertical_arm = RectangleOccluder(
        "l-wall-vertical",
        "wall",
        np.asarray([0.5, 1.37, 0.5 * np.pi], dtype=np.float64),
        1.0,
        0.35,
    )
    request = _request(
        base_config,
        planner_config,
        occluders=(horizontal_arm, vertical_arm),
    )

    result = plan_lightweight_teb(request)

    assert result.route is not None, result.diagnostics
    robot_radius = 0.5 * float(
        np.hypot(
            base_config["robot"]["length_m"],
            base_config["robot"]["width_m"],
        )
    ) + float(base_config["robot"]["inflation_m"])
    for component in (horizontal_arm, vertical_arm):
        clearance = point_signed_distance(
            component, result.route.sampled_poses_world[:, :2]
        ) - robot_radius
        assert float(np.min(clearance)) >= (
            planner_config.represented_occluder_clearance_range_m[0] - 1e-5
        )
    _assert_dynamics(result.route, request, base_config, planner_config)


def test_static_occupancy_infeasibility_returns_finite_diagnostics(
    base_config: dict, planner_config
) -> None:
    from src.planning.lightweight_teb import plan_lightweight_teb

    grid = build_grid_spec(base_config)
    request = _request(
        base_config,
        planner_config,
        static_occupancy=np.ones((grid.height, grid.width), dtype=np.float32),
    )
    result = plan_lightweight_teb(request)

    assert result.route is None
    assert result.rejection_reason == "teb_static_collision"
    assert result.diagnostics.candidates
    assert all(np.isfinite(candidate.cost) for candidate in result.diagnostics.candidates)


def test_repeated_request_returns_byte_identical_full_route(
    base_config: dict, planner_config
) -> None:
    from src.planning.lightweight_teb import plan_lightweight_teb

    occluder = CircleOccluder(
        "tree",
        "tree_trunk",
        np.asarray([1.2, 1.08], dtype=np.float64),
        0.3,
    )
    request = _request(base_config, planner_config, occluders=(occluder,))
    first = plan_lightweight_teb(request)
    second = plan_lightweight_teb(request)

    assert first.rejection_reason == second.rejection_reason
    assert first.route is not None and second.route is not None
    for name in (
        "band_poses_world",
        "band_interval_dt_s",
        "sample_times_s",
        "sampled_poses_world",
        "sampled_controls",
    ):
        assert getattr(first.route, name).tobytes() == getattr(second.route, name).tobytes()


def test_each_teb_initialization_runs_the_frozen_update_count(
    base_config: dict, planner_config
) -> None:
    from src.planning.lightweight_teb import plan_lightweight_teb

    occluder = CircleOccluder(
        "tree",
        "tree_trunk",
        np.asarray([1.2, 1.08], dtype=np.float64),
        0.3,
    )
    result = plan_lightweight_teb(
        _request(base_config, planner_config, occluders=(occluder,))
    )

    assert result.route is not None
    assert all(
        candidate.optimization_iterations == planner_config.max_iterations
        for candidate in result.diagnostics.candidates
    )


def test_teb_diagnostic_cost_contains_every_frozen_objective_term(
    base_config: dict, planner_config
) -> None:
    from src.planning.lightweight_teb import plan_lightweight_teb

    result = plan_lightweight_teb(_request(base_config, planner_config))

    candidate = result.diagnostics.candidates[0]
    terms = dict(candidate.cost_terms)
    assert tuple(terms) == tuple(dict(planner_config.weights))
    assert all(np.isfinite(value) and value >= 0.0 for value in terms.values())
    assert candidate.cost == pytest.approx(sum(terms.values()))
    assert result.route is not None
    assert result.route.task_cost == pytest.approx(candidate.cost)
