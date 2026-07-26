from __future__ import annotations

from dataclasses import fields, replace
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
    start_pose: np.ndarray | None = None,
    initial_control: np.ndarray | None = None,
    goal_pose: np.ndarray | None = None,
):
    from src.planning.lightweight_teb import StaticTebRequest

    grid = build_grid_spec(base_config)
    return StaticTebRequest(
        start_pose=(
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
            if start_pose is None
            else start_pose
        ),
        initial_control=(
            np.asarray([0.0, 0.0], dtype=np.float32)
            if initial_control is None
            else initial_control
        ),
        local_goal_world_pose=(
            np.asarray([2.4, 0.0, 0.0], dtype=np.float32)
            if goal_pose is None
            else goal_pose
        ),
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


def test_joint_control_projection_preserves_all_interval_bounds(
    base_config: dict, planner_config
) -> None:
    import src.planning.lightweight_teb as teb

    projector = getattr(teb, "_project_control_jointly", None)
    assert callable(projector)
    previous = np.asarray([0.4, -0.8], dtype=np.float64)
    projected = projector(
        np.asarray([0.04, 0.0], dtype=np.float64),
        previous,
        base_config=base_config,
        planner_config=planner_config,
        dt_s=planner_config.route_sample_dt_s,
    )

    max_speed = float(base_config["robot"]["max_linear_speed_mps"])
    max_omega = float(base_config["robot"]["max_angular_speed_radps"])
    assert abs(float(projected[0])) <= max_speed + 1e-12
    assert abs(float(projected[1])) <= max_omega + 1e-12
    assert abs(float(projected[0] - previous[0])) <= (
        planner_config.max_linear_acceleration_mps2
        * planner_config.route_sample_dt_s
        + 1e-12
    )
    assert abs(float(projected[1] - previous[1])) <= (
        planner_config.max_angular_acceleration_radps2
        * planner_config.route_sample_dt_s
        + 1e-12
    )
    assert abs(float(projected[1] / projected[0])) <= (
        planner_config.max_curvature_per_m + 1e-12
    )


def test_sampled_dynamics_postvalidator_rejects_tampered_turn(
    base_config: dict, planner_config
) -> None:
    import src.planning.lightweight_teb as teb

    occluder = CircleOccluder(
        "validator-tree",
        "tree_trunk",
        np.asarray([1.2, 1.0], dtype=np.float64),
        0.35,
    )
    request = _request(base_config, planner_config, occluders=(occluder,))
    result = teb.plan_static_lightweight_teb(request)
    assert result.route is not None, result.rejection_reason
    validator = getattr(teb, "_sampled_dynamics_rejection", None)
    assert callable(validator)
    assert (
        validator(
            request,
            result.route.sampled_poses_world,
            result.route.sampled_controls,
        )
        is None
    )

    tampered_controls = result.route.sampled_controls.copy()
    tampered_controls[1, 1] = np.clip(
        tampered_controls[0, 1]
        + 2.0
        * planner_config.max_angular_acceleration_radps2
        * planner_config.route_sample_dt_s,
        -float(base_config["robot"]["max_angular_speed_radps"]),
        float(base_config["robot"]["max_angular_speed_radps"]),
    )
    assert (
        validator(
            request,
            result.route.sampled_poses_world,
            tampered_controls,
        )
        == "teb_dynamics_limit"
    )


def _integrate_twist_oracle(
    pose: np.ndarray, control: np.ndarray, dt_s: float
) -> np.ndarray:
    start = np.asarray(pose, dtype=np.float64)
    v = float(control[0])
    omega = float(control[1])
    yaw = float(start[2])
    result = start.copy()
    if abs(omega) <= 1e-12:
        result[0] += v * dt_s * np.cos(yaw)
        result[1] += v * dt_s * np.sin(yaw)
    else:
        end_yaw = yaw + omega * dt_s
        radius = v / omega
        result[0] += radius * (np.sin(end_yaw) - np.sin(yaw))
        result[1] -= radius * (np.cos(end_yaw) - np.cos(yaw))
        result[2] = end_yaw
    result[2] = (result[2] + np.pi) % (2.0 * np.pi) - np.pi
    return result


def _dense_route_poses(request, route) -> np.ndarray:
    fractions = np.linspace(0.0, 1.0, 6, dtype=np.float64)
    pose = request.start_pose.astype(np.float64)
    dense: list[np.ndarray] = []
    for control in route.sampled_controls:
        dense.extend(
            _integrate_twist_oracle(
                pose,
                control,
                request.planner_config.route_sample_dt_s * float(fraction),
            )
            for fraction in fractions
        )
        pose = _integrate_twist_oracle(
            pose,
            control,
            request.planner_config.route_sample_dt_s,
        )
    return np.asarray(dense, dtype=np.float64)


def _route_poses_at_times_oracle(request, route, times_s: np.ndarray) -> np.ndarray:
    dt_s = request.planner_config.route_sample_dt_s
    anchors = [request.start_pose.astype(np.float64)]
    for control in route.sampled_controls:
        anchors.append(_integrate_twist_oracle(anchors[-1], control, dt_s))
    result: list[np.ndarray] = []
    for raw_time in np.asarray(times_s, dtype=np.float64):
        time_s = float(np.clip(raw_time, 0.0, len(route.sampled_controls) * dt_s))
        completed = min(int(np.floor((time_s + 1e-10) / dt_s)), len(route.sampled_controls))
        if completed == len(route.sampled_controls):
            result.append(anchors[-1])
            continue
        remainder = max(0.0, time_s - completed * dt_s)
        result.append(
            _integrate_twist_oracle(
                anchors[completed],
                route.sampled_controls[completed],
                remainder,
            )
        )
    return np.asarray(result, dtype=np.float64)


def _poses_from_controls_oracle(
    start_pose: np.ndarray,
    controls: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    pose = np.asarray(start_pose, dtype=np.float64)
    poses: list[np.ndarray] = []
    for control in np.asarray(controls, dtype=np.float64):
        pose = _integrate_twist_oracle(pose, control, dt_s)
        poses.append(pose)
    return np.asarray(poses, dtype=np.float64)


def _config_with_weight(planner_config, name: str, value: float):
    return replace(
        planner_config,
        weights=tuple(
            (term, value if term == name else weight)
            for term, weight in planner_config.weights
        ),
    )


def _assert_irreversibly_readonly(array: np.ndarray) -> None:
    assert not array.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        array.setflags(write=True)


def test_observed_teb_uses_only_observation_derived_dynamic_state(
    base_config: dict,
    planner_config,
) -> None:
    from src.planning.lightweight_teb import (
        ObservedDynamicObstacle,
        ObservedTebRequest,
        plan_observed_lightweight_teb,
    )

    forbidden = ("oracle", "future", "risk", "label", "conflict")
    request_fields = {field.name for field in fields(ObservedTebRequest)}
    obstacle_fields = {field.name for field in fields(ObservedDynamicObstacle)}
    assert not any(token in name for name in request_fields for token in forbidden)
    assert not any(token in name for name in obstacle_fields for token in forbidden)

    static = _request(base_config, planner_config)
    obstacle = ObservedDynamicObstacle(
        object_id="observed-human",
        observed_pose=np.asarray([1.2, 1.0, 0.0], dtype=np.float32),
        observed_velocity_xy=np.zeros(2, dtype=np.float32),
        footprint_radius_m=0.2,
        observation_age_s=0.0,
    )
    request = ObservedTebRequest(
        start_pose=static.start_pose,
        initial_control=static.initial_control,
        local_goal_world_pose=static.local_goal_world_pose,
        static_occupancy=static.static_occupancy,
        occluders=(),
        observed_dynamic_obstacles=(obstacle,),
        base_config=base_config,
        planner_config=planner_config,
    )
    for array in (
        obstacle.observed_pose,
        obstacle.observed_velocity_xy,
        request.start_pose,
        request.initial_control,
        request.local_goal_world_pose,
        request.static_occupancy,
    ):
        _assert_irreversibly_readonly(array)

    result = plan_observed_lightweight_teb(request)

    assert result.route is not None, result.rejection_reason
    assert result.route.band_poses_world.shape == (21, 3)
    assert result.route.band_interval_dt_s.shape == (20,)
    assert result.route.sampled_poses_world.shape == (40, 3)
    assert result.route.sampled_controls.shape == (40, 2)
    np.testing.assert_array_equal(
        result.route.goal_world_pose,
        request.local_goal_world_pose,
    )
    distances = np.linalg.norm(
        result.route.sampled_poses_world[:, :2] - obstacle.observed_pose[:2],
        axis=1,
    )
    assert float(np.min(distances)) > obstacle.footprint_radius_m


def test_far_observed_obstacle_does_not_trigger_static_clearance_ceiling(
    base_config: dict,
    planner_config,
) -> None:
    from src.planning.lightweight_teb import (
        ObservedDynamicObstacle,
        ObservedTebRequest,
        plan_observed_lightweight_teb,
    )

    static = _request(base_config, planner_config)
    request = ObservedTebRequest(
        start_pose=static.start_pose,
        initial_control=static.initial_control,
        local_goal_world_pose=static.local_goal_world_pose,
        static_occupancy=static.static_occupancy,
        occluders=(),
        observed_dynamic_obstacles=(
            ObservedDynamicObstacle(
                object_id="far-observed-human",
                observed_pose=np.asarray([6.0, 6.0, 0.0], dtype=np.float32),
                observed_velocity_xy=np.zeros(2, dtype=np.float32),
                footprint_radius_m=0.2,
                observation_age_s=0.0,
            ),
        ),
        base_config=base_config,
        planner_config=planner_config,
    )

    result = plan_observed_lightweight_teb(request)

    assert result.route is not None, result.rejection_reason
    np.testing.assert_array_equal(
        result.route.goal_world_pose,
        request.local_goal_world_pose,
    )


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
        "conflict",
    )
    assert all(
        token not in field.name.lower()
        for field in fields(StaticTebRequest)
        for token in forbidden
    )


def test_teb_public_api_is_exported_from_planning_package() -> None:
    from src.planning import (
        ObservedDynamicObstacle,
        ObservedTebRequest,
        PlannedTebRoute,
        StaticTebRequest,
        plan_lightweight_teb,
        plan_observed_lightweight_teb,
        plan_static_lightweight_teb,
    )

    assert PlannedTebRoute.__name__ == "PlannedTebRoute"
    assert StaticTebRequest.__name__ == "StaticTebRequest"
    assert ObservedTebRequest.__name__ == "ObservedTebRequest"
    assert ObservedDynamicObstacle.__name__ == "ObservedDynamicObstacle"
    assert plan_lightweight_teb is plan_static_lightweight_teb
    assert callable(plan_static_lightweight_teb)
    assert callable(plan_observed_lightweight_teb)


def test_straight_initialization_is_unbent_before_optimizer_updates(
    base_config: dict, planner_config
) -> None:
    import src.planning.lightweight_teb as teb

    occluder = CircleOccluder(
        "offset-tree",
        "tree_trunk",
        np.asarray([1.2, 1.0], dtype=np.float64),
        0.35,
    )
    request = _request(
        base_config,
        planner_config,
        occluders=(occluder,),
        start_pose=np.asarray([0.0, 0.0, 0.3], dtype=np.float32),
        goal_pose=np.asarray([2.4, 0.0, -0.5], dtype=np.float32),
    )
    seed_band, seed_dt_s = teb._initialize_straight_band(request)

    expected_xy = np.linspace(
        request.start_pose[:2],
        request.local_goal_world_pose[:2],
        planner_config.band_node_count,
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        seed_band[:, :2],
        expected_xy,
        rtol=0.0,
        atol=2e-7,
    )
    fractions = np.linspace(0.0, 1.0, planner_config.band_node_count)
    expected_yaw = (
        request.start_pose[2]
        + fractions
        * np.arctan2(
            np.sin(request.local_goal_world_pose[2] - request.start_pose[2]),
            np.cos(request.local_goal_world_pose[2] - request.start_pose[2]),
        )
    )
    np.testing.assert_allclose(
        seed_band[:, 2],
        expected_yaw,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        seed_dt_s,
        np.full(20, planner_config.initial_band_dt_s, dtype=np.float64),
    )


def test_bypass_initializations_form_opposite_waypoint_bands(
    base_config: dict, planner_config
) -> None:
    import src.planning.lightweight_teb as teb

    occluder = CircleOccluder(
        "centered-tree",
        "tree_trunk",
        np.asarray([2.0, 0.0], dtype=np.float64),
        0.35,
    )
    request = _request(
        base_config,
        planner_config,
        occluders=(occluder,),
        goal_pose=np.asarray([4.0, 0.0, 0.0], dtype=np.float32),
    )

    left, left_dt_s = teb._initialize_bypass_band(request, side=1.0)
    right, right_dt_s = teb._initialize_bypass_band(request, side=-1.0)

    np.testing.assert_allclose(left[[0, -1]], right[[0, -1]])
    np.testing.assert_allclose(left[:, 0], right[:, 0], atol=1e-9)
    np.testing.assert_allclose(left[:, 1], -right[:, 1], atol=1e-9)
    assert float(np.max(left[:, 1])) > 0.0
    assert float(np.min(right[:, 1])) < 0.0
    left_bypass = left[left[:, 1] >= 0.9 * np.max(left[:, 1])]
    right_bypass = right[right[:, 1] <= 0.9 * np.min(right[:, 1])]
    assert np.ptp(left_bypass[:, 0]) > 2.0 * occluder.radius_m
    assert np.ptp(right_bypass[:, 0]) > 2.0 * occluder.radius_m
    np.testing.assert_array_equal(left_dt_s, right_dt_s)


def test_unobstructed_route_has_frozen_full_route_layout_and_dynamics(
    base_config: dict, planner_config
) -> None:
    from src.planning.lightweight_teb import plan_lightweight_teb

    request = _request(base_config, planner_config)
    result = plan_lightweight_teb(request)

    assert result.rejection_reason is None
    assert result.route is not None
    route = result.route
    assert route.band_poses_world.shape == (21, 3)
    assert route.band_interval_dt_s.shape == (20,)
    assert route.sample_times_s.shape == (40,)
    assert route.sampled_poses_world.shape == (40, 3)
    assert route.sampled_controls.shape == (40, 2)
    np.testing.assert_array_equal(route.band_poses_world[0], request.start_pose)
    np.testing.assert_array_equal(route.band_poses_world[-1], request.local_goal_world_pose)
    np.testing.assert_array_equal(
        route.sample_times_s,
        np.arange(1, 41, dtype=np.float32) * np.float32(0.2),
    )
    for array in (
        route.goal_world_pose,
        route.band_poses_world,
        route.band_interval_dt_s,
        route.sample_times_s,
        route.sampled_poses_world,
        route.sampled_controls,
    ):
        assert array.dtype == np.float32
    assert np.all(route.band_interval_dt_s >= 0.1)
    assert np.all(route.band_interval_dt_s <= 0.4)
    assert (
        float(np.sum(route.band_interval_dt_s))
        <= planner_config.maximum_route_time_s + 1e-6
    )
    assert route.goal_arrival_time_s <= planner_config.maximum_route_time_s
    np.testing.assert_allclose(
        route.sampled_poses_world[-1],
        request.local_goal_world_pose,
        rtol=0.0,
        atol=planner_config.goal_position_tolerance_m,
    )
    assert not route.sampled_poses_world.flags.writeable
    assert not route.sampled_controls.flags.writeable
    _assert_dynamics(route, request, base_config, planner_config)


def test_returned_timed_band_tracks_projected_route_at_every_node(
    base_config: dict, planner_config
) -> None:
    from src.planning.lightweight_teb import plan_static_lightweight_teb

    occluder = CircleOccluder(
        "tree",
        "tree_trunk",
        np.asarray([1.2, 1.0], dtype=np.float64),
        0.35,
    )
    request = _request(base_config, planner_config, occluders=(occluder,))
    result = plan_static_lightweight_teb(request)

    assert result.route is not None, result.rejection_reason
    route = result.route
    band_times = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.cumsum(route.band_interval_dt_s, dtype=np.float64),
        )
    )
    projected = _route_poses_at_times_oracle(request, route, band_times)
    np.testing.assert_allclose(
        route.band_poses_world[1:-1],
        projected[1:-1],
        rtol=0.0,
        atol=2e-4,
    )
    assert (
        np.linalg.norm(route.band_poses_world[-1, :2] - projected[-1, :2])
        <= 0.6 * planner_config.goal_position_tolerance_m
    )
    assert abs(
        float(
            np.arctan2(
                np.sin(route.band_poses_world[-1, 2] - projected[-1, 2]),
                np.cos(route.band_poses_world[-1, 2] - projected[-1, 2]),
            )
        )
    ) <= min(planner_config.goal_yaw_tolerance_rad, 0.02)


def test_optimizer_weights_change_band_and_report_actual_iterations(
    base_config: dict, planner_config
) -> None:
    import src.planning.lightweight_teb as teb

    occluder = CircleOccluder(
        "weighted-tree",
        "tree_trunk",
        np.asarray([1.2, 1.0], dtype=np.float64),
        0.35,
    )
    low_config = _config_with_weight(planner_config, "obstacle", 0.25)
    high_config = _config_with_weight(planner_config, "obstacle", 8.0)
    low_request = _request(
        base_config,
        replace(low_config, initialization_ids=("straight",)),
        occluders=(occluder,),
    )
    high_request = _request(
        base_config,
        replace(high_config, initialization_ids=("straight",)),
        occluders=(occluder,),
    )
    robot_radius = teb._robot_radius(base_config)
    low = teb._optimize_teb_band(
        low_request,
        side=teb._straight_escape_side(low_request),
        robot_radius=robot_radius,
    )
    high = teb._optimize_teb_band(
        high_request,
        side=teb._straight_escape_side(high_request),
        robot_radius=robot_radius,
    )

    low_lateral_deformation = float(np.sum(np.abs(low[0][1:-1, 1])))
    high_lateral_deformation = float(np.sum(np.abs(high[0][1:-1, 1])))
    assert high_lateral_deformation > low_lateral_deformation + 0.1
    assert low[0].tobytes() != high[0].tobytes()
    assert low[2] == low_config.max_iterations
    assert high[2] == high_config.max_iterations
    low_result = teb.plan_static_lightweight_teb(low_request)
    high_result = teb.plan_static_lightweight_teb(high_request)
    assert low_result.route is not None, low_result.rejection_reason
    assert high_result.route is not None, high_result.rejection_reason
    assert (
        low_result.route.band_poses_world.tobytes()
        != high_result.route.band_poses_world.tobytes()
    )


def test_band_objective_terms_have_frozen_semantic_sensitivity(
    base_config: dict, planner_config
) -> None:
    import src.planning.lightweight_teb as teb

    objective = getattr(teb, "_band_objective_terms", None)
    assert callable(objective)
    request = _request(
        base_config,
        planner_config,
        initial_control=np.asarray([0.48, 0.0], dtype=np.float32),
    )
    seed_band, seed_dt_s = teb._initialize_straight_band(request)
    robot_radius = teb._robot_radius(base_config)
    baseline = dict(objective(request, seed_band, seed_dt_s, robot_radius))

    assert baseline["length"] > 0.0
    assert baseline["time"] > 0.0
    for name in (
        "smoothness",
        "obstacle",
        "nonholonomic",
        "velocity",
        "acceleration",
        "goal_heading",
        "initial_control",
    ):
        assert baseline[name] == pytest.approx(0.0, abs=1e-10)

    curved_band = seed_band.copy()
    curved_band[5:16, 1] = 0.3 * np.sin(np.linspace(0.0, np.pi, 11))
    curved_delta = np.diff(curved_band[:, :2], axis=0)
    curved_band[:-1, 2] = np.arctan2(
        curved_delta[:, 1],
        curved_delta[:, 0],
    )
    curved_band[-1, 2] = curved_band[-2, 2]
    assert dict(objective(request, curved_band, seed_dt_s, robot_radius))[
        "smoothness"
    ] > baseline["smoothness"]

    slipping_band = seed_band.copy()
    slipping_band[1:-1, 2] = 0.4
    assert dict(objective(request, slipping_band, seed_dt_s, robot_radius))[
        "nonholonomic"
    ] > baseline["nonholonomic"]

    fast_dt_s = np.full_like(seed_dt_s, 0.1)
    assert dict(objective(request, seed_band, fast_dt_s, robot_radius))[
        "velocity"
    ] > baseline["velocity"]

    accelerating_dt_s = seed_dt_s.copy()
    accelerating_dt_s[:2] = (0.4, 0.1)
    assert dict(
        objective(request, seed_band, accelerating_dt_s, robot_radius)
    )["acceleration"] > baseline["acceleration"]

    heading_band = seed_band.copy()
    heading_band[-2, 1] = 0.5
    assert dict(objective(request, heading_band, seed_dt_s, robot_radius))[
        "goal_heading"
    ] > baseline["goal_heading"]

    discontinuous_request = _request(base_config, planner_config)
    discontinuous_seed, discontinuous_dt_s = teb._initialize_straight_band(
        discontinuous_request
    )
    assert dict(
        objective(
            discontinuous_request,
            discontinuous_seed,
            discontinuous_dt_s,
            robot_radius,
        )
    )["initial_control"] > 0.0

    occluder = CircleOccluder(
        "objective-tree",
        "tree_trunk",
        np.asarray([1.2, 1.0], dtype=np.float64),
        0.35,
    )
    obstacle_request = _request(
        base_config,
        planner_config,
        occluders=(occluder,),
        initial_control=np.asarray([0.48, 0.0], dtype=np.float32),
    )
    obstacle_seed, obstacle_dt_s = teb._initialize_straight_band(
        obstacle_request
    )
    assert dict(
        objective(
            obstacle_request,
            obstacle_seed,
            obstacle_dt_s,
            robot_radius,
        )
    )["obstacle"] > 0.0


def test_band_kinematics_preserves_reverse_twist_sign(
    base_config: dict, planner_config
) -> None:
    import src.planning.lightweight_teb as teb

    controls = np.tile(np.asarray([-0.4, 0.0], dtype=np.float64), (4, 1))
    dt_s = np.full(4, 0.25, dtype=np.float64)
    poses = _poses_from_controls_oracle(
        np.zeros(3, dtype=np.float64),
        controls,
        0.25,
    )
    request = _request(
        base_config,
        planner_config,
        initial_control=np.asarray([-0.4, 0.0], dtype=np.float32),
        goal_pose=poses[-1].astype(np.float32),
    )
    band = np.vstack((request.start_pose, poses))
    kinematics = teb._band_kinematics(request, band, dt_s)

    np.testing.assert_allclose(
        kinematics["linear_velocity"],
        -0.4,
        rtol=0.0,
        atol=1e-10,
    )
    residual = kinematics.get("nonholonomic_residual")
    assert residual is not None
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=1e-10)


def test_constant_curvature_arc_has_zero_residual_and_curvature_change(
    base_config: dict, planner_config
) -> None:
    import src.planning.lightweight_teb as teb

    controls = np.tile(np.asarray([0.5, 0.2], dtype=np.float64), (20, 1))
    dt_s = np.full(20, 0.25, dtype=np.float64)
    poses = _poses_from_controls_oracle(
        np.zeros(3, dtype=np.float64),
        controls,
        0.25,
    )
    request = _request(
        base_config,
        planner_config,
        initial_control=np.asarray([0.5, 0.2], dtype=np.float32),
        goal_pose=poses[-1].astype(np.float32),
    )
    band = np.vstack((request.start_pose, poses))
    kinematics = teb._band_kinematics(request, band, dt_s)

    np.testing.assert_allclose(
        kinematics["linear_velocity"],
        0.5,
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        kinematics["angular_velocity"],
        0.2,
        rtol=0.0,
        atol=1e-10,
    )
    residual = kinematics.get("nonholonomic_residual")
    assert residual is not None
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=1e-10)
    terms = dict(
        teb._band_objective_terms(
            request,
            band,
            dt_s,
            teb._robot_radius(base_config),
        )
    )
    assert terms["smoothness"] == pytest.approx(0.0, abs=1e-10)
    assert terms["nonholonomic"] == pytest.approx(0.0, abs=1e-10)
    updater = getattr(teb, "_apply_nonholonomic_residual_update", None)
    assert callable(updater)
    updated_positions, updated_yaws = updater(
        request,
        band[:, :2],
        band[:, 2],
        dt_s,
        weight=dict(planner_config.weights)["nonholonomic"],
    )
    np.testing.assert_allclose(
        updated_positions,
        band[:, :2],
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        updated_yaws,
        band[:, 2],
        rtol=0.0,
        atol=1e-9,
    )


def test_near_zero_yaw_twist_inversion_uses_stable_limit() -> None:
    import src.planning.lightweight_teb as teb

    dt_s = 0.25
    start = np.zeros(3, dtype=np.float64)
    control = np.asarray([0.5, 1e-8], dtype=np.float64)
    delta_yaw = float(control[1] * dt_s)
    radius = float(control[0] / control[1])
    end = np.asarray(
        [
            radius * np.sin(delta_yaw),
            radius * 2.0 * np.sin(0.5 * delta_yaw) ** 2,
            delta_yaw,
        ],
        dtype=np.float64,
    )
    kinematics = teb._se2_interval_kinematics(
        start[None, :],
        end[None, :],
        np.asarray([dt_s], dtype=np.float64),
    )

    assert float(kinematics["linear_velocity"][0]) == pytest.approx(
        0.5,
        abs=1e-8,
    )
    assert float(kinematics["angular_velocity"][0]) == pytest.approx(
        1e-8,
        abs=1e-12,
    )
    np.testing.assert_allclose(
        kinematics["nonholonomic_residual"],
        0.0,
        rtol=0.0,
        atol=1e-10,
    )


def test_curvature_gradient_avoids_repeated_full_band_evaluations(
    base_config: dict,
    planner_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.planning.lightweight_teb as teb

    request = _request(base_config, planner_config)
    band, dt_s = teb._initialize_straight_band(request)
    full_cost_calls = 0
    original = teb._curvature_change_cost

    def counted(*args, **kwargs):
        nonlocal full_cost_calls
        full_cost_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(teb, "_curvature_change_cost", counted)
    teb._curvature_change_gradients(
        request,
        band[:, :2],
        band[:, 2],
        dt_s,
    )

    assert full_cost_calls <= 1


def test_task_cost_uses_curvature_change_and_limit_hinges(
    base_config: dict, planner_config
) -> None:
    import src.planning.lightweight_teb as teb

    request = _request(
        base_config,
        planner_config,
        initial_control=np.asarray([0.2, 0.1], dtype=np.float32),
    )
    dt_s = planner_config.route_sample_dt_s
    constant_curvature_controls = np.asarray(
        [[0.2, 0.1], [0.4, 0.2], [0.6, 0.3]],
        dtype=np.float64,
    )
    constant_curvature_poses = _poses_from_controls_oracle(
        request.start_pose,
        constant_curvature_controls,
        dt_s,
    )
    baseline = dict(
        teb._task_cost_terms(
            request,
            poses=constant_curvature_poses,
            controls=constant_curvature_controls,
            arrival_time_s=3.0 * dt_s,
            clearance_m=0.0,
        )
    )

    assert baseline["length"] > 0.0
    assert baseline["time"] > 0.0
    assert baseline["smoothness"] == pytest.approx(0.0, abs=1e-10)
    assert baseline["velocity"] == pytest.approx(0.0, abs=1e-10)
    assert baseline["acceleration"] == pytest.approx(0.0, abs=1e-10)
    assert baseline["initial_control"] == pytest.approx(0.0, abs=1e-10)

    changing_curvature_controls = np.asarray(
        [[0.4, 0.1], [0.4, 0.1], [0.4, 0.3]],
        dtype=np.float64,
    )
    changing_curvature_poses = _poses_from_controls_oracle(
        request.start_pose,
        changing_curvature_controls,
        dt_s,
    )
    assert dict(
        teb._task_cost_terms(
            request,
            poses=changing_curvature_poses,
            controls=changing_curvature_controls,
            arrival_time_s=3.0 * dt_s,
            clearance_m=0.0,
        )
    )["smoothness"] > 0.0

    velocity_violation = np.asarray([[1.0, 0.9]], dtype=np.float64)
    assert dict(
        teb._task_cost_terms(
            request,
            poses=_poses_from_controls_oracle(
                request.start_pose,
                velocity_violation,
                dt_s,
            ),
            controls=velocity_violation,
            arrival_time_s=dt_s,
            clearance_m=0.0,
        )
    )["velocity"] > 0.0

    acceleration_violation = np.asarray(
        [[0.8, 0.1], [0.8, 0.1]],
        dtype=np.float64,
    )
    acceleration_terms = dict(
        teb._task_cost_terms(
            request,
            poses=_poses_from_controls_oracle(
                request.start_pose,
                acceleration_violation,
                dt_s,
            ),
            controls=acceleration_violation,
            arrival_time_s=2.0 * dt_s,
            clearance_m=0.0,
        )
    )
    assert acceleration_terms["velocity"] == pytest.approx(0.0, abs=1e-10)
    assert acceleration_terms["acceleration"] > 0.0


@pytest.mark.parametrize(
    "weight_name",
    [
        "length",
        "time",
        "smoothness",
        "obstacle",
        "nonholonomic",
        "velocity",
        "acceleration",
        "goal_heading",
        "initial_control",
    ],
)
def test_each_frozen_weight_changes_preprojection_optimization(
    base_config: dict,
    planner_config,
    weight_name: str,
) -> None:
    import src.planning.lightweight_teb as teb

    request_kwargs: dict[str, object] = {}
    if weight_name in {"length", "smoothness", "obstacle"}:
        request_kwargs["occluders"] = (
            CircleOccluder(
                "weighted-tree",
                "tree_trunk",
                np.asarray([1.2, 1.0], dtype=np.float64),
                0.35,
            ),
        )
    elif weight_name == "nonholonomic":
        request_kwargs["start_pose"] = np.asarray(
            [0.0, 0.0, 0.8],
            dtype=np.float32,
        )
    elif weight_name in {"velocity", "acceleration"}:
        request_kwargs["goal_pose"] = np.asarray(
            [5.0, 0.0, 0.0],
            dtype=np.float32,
        )
    elif weight_name == "goal_heading":
        request_kwargs["goal_pose"] = np.asarray(
            [2.4, 1.0, 0.0],
            dtype=np.float32,
        )
    elif weight_name == "initial_control":
        request_kwargs["initial_control"] = np.asarray(
            [0.8, 0.4],
            dtype=np.float32,
        )

    low_config = _config_with_weight(planner_config, weight_name, 0.05)
    high_config = _config_with_weight(planner_config, weight_name, 8.0)
    low_request = _request(base_config, low_config, **request_kwargs)
    high_request = _request(base_config, high_config, **request_kwargs)
    robot_radius = teb._robot_radius(base_config)
    low = teb._optimize_teb_band(
        low_request,
        side=teb._straight_escape_side(low_request),
        robot_radius=robot_radius,
    )
    high = teb._optimize_teb_band(
        high_request,
        side=teb._straight_escape_side(high_request),
        robot_radius=robot_radius,
    )

    assert (
        low[0].tobytes() != high[0].tobytes()
        or low[1].tobytes() != high[1].tobytes()
    )
def test_straight_five_meter_goal_is_reachable_within_full_route_horizon(
    base_config: dict, planner_config
) -> None:
    from src.planning.lightweight_teb import plan_lightweight_teb

    request = _request(
        base_config,
        planner_config,
        goal_pose=np.asarray([5.0, 0.0, 0.0], dtype=np.float32),
    )
    result = plan_lightweight_teb(request)

    assert result.route is not None, result.rejection_reason
    assert result.route.sample_times_s.shape == (40,)
    assert result.route.goal_arrival_time_s <= planner_config.maximum_route_time_s
    np.testing.assert_allclose(
        result.route.sampled_poses_world[-1],
        request.local_goal_world_pose,
        rtol=0.0,
        atol=planner_config.goal_position_tolerance_m,
    )
    _assert_dynamics(result.route, request, base_config, planner_config)


def test_goal_arrival_time_records_first_arrival_before_stationary_hold(
    base_config: dict, planner_config
) -> None:
    from src.planning.lightweight_teb import plan_lightweight_teb

    request = _request(
        base_config,
        planner_config,
        goal_pose=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )
    result = plan_lightweight_teb(request)

    assert result.route is not None, result.rejection_reason
    route = result.route
    position_error = np.linalg.norm(
        route.sampled_poses_world[:, :2] - request.local_goal_world_pose[:2],
        axis=1,
    )
    heading_error = np.abs(
        np.arctan2(
            np.sin(route.sampled_poses_world[:, 2] - request.local_goal_world_pose[2]),
            np.cos(route.sampled_poses_world[:, 2] - request.local_goal_world_pose[2]),
        )
    )
    arrived = (
        (position_error <= planner_config.goal_position_tolerance_m)
        & (heading_error <= planner_config.goal_yaw_tolerance_rad)
    )
    arrival_indices = np.flatnonzero(arrived)
    assert arrival_indices.size > 1
    first_entry_index = int(arrival_indices[0])
    final_suffix_start = int(np.flatnonzero(~arrived)[-1] + 1)
    assert first_entry_index <= final_suffix_start
    assert final_suffix_start < route.sampled_poses_world.shape[0]
    assert np.linalg.norm(route.sampled_controls[first_entry_index]) > 1e-5
    stopped_indices = np.flatnonzero(
        arrived
        & (np.abs(route.sampled_controls[:, 0]) <= 1e-5)
        & (np.abs(route.sampled_controls[:, 1]) <= 1e-5)
    )
    assert stopped_indices.size > 1
    first_stopped_index = int(stopped_indices[0])
    assert first_stopped_index > first_entry_index
    assert route.goal_arrival_time_s == pytest.approx(
        (final_suffix_start + 1) * planner_config.route_sample_dt_s
    )
    assert route.goal_arrival_time_s < planner_config.maximum_route_time_s
    terminal_count = 2
    np.testing.assert_allclose(
        np.diff(route.sampled_poses_world[-terminal_count:], axis=0),
        np.zeros_like(route.sampled_poses_world[-terminal_count + 1 :]),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        route.sampled_controls[-terminal_count:],
        np.zeros_like(route.sampled_controls[-terminal_count:]),
        rtol=0.0,
        atol=1e-7,
    )
    import src.planning.lightweight_teb as teb

    assert teb._has_terminal_goal_suffix(request, route.sampled_poses_world)


def test_transient_goal_entry_does_not_satisfy_terminal_suffix(
    base_config: dict, planner_config
) -> None:
    import src.planning.lightweight_teb as teb

    request = _request(
        base_config,
        planner_config,
        goal_pose=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )
    poses = np.tile(
        np.asarray([1.6, 0.0, 2.239], dtype=np.float64),
        (40, 1),
    )
    poses[4] = request.local_goal_world_pose
    mask = teb._goal_tolerance_mask(request, poses)
    assert int(np.flatnonzero(mask)[0]) == 4
    assert not teb._has_terminal_goal_suffix(request, poses)

    poses[-2] = np.asarray([0.9, 0.0, 0.1], dtype=np.float64)
    poses[-1] = np.asarray([0.95, 0.0, 0.05], dtype=np.float64)
    assert teb._has_terminal_goal_suffix(request, poses)
    assert teb._stable_goal_arrival_index(request, poses) == 38


def test_terminal_goal_suffix_uses_wide_position_region_without_yaw_gate(
    base_config: dict, planner_config
) -> None:
    import src.planning.lightweight_teb as teb

    request = _request(
        base_config,
        planner_config,
        goal_pose=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )
    poses = np.tile(
        np.asarray([1.6, 0.0, 0.0], dtype=np.float64),
        (40, 1),
    )
    poses[-2] = np.asarray([1.49, 0.0, np.pi], dtype=np.float64)
    poses[-1] = np.asarray([1.45, 0.0, -2.8], dtype=np.float64)

    assert planner_config.goal_position_tolerance_m == 0.5
    np.testing.assert_array_equal(
        teb._goal_tolerance_mask(request, poses)[-2:],
        np.asarray([True, True]),
    )
    assert teb._has_terminal_goal_suffix(request, poses)
    assert teb._stable_goal_arrival_index(request, poses) == 38


def test_planner_request_and_result_arrays_are_irreversibly_readonly(
    base_config: dict, planner_config
) -> None:
    from src.planning.lightweight_teb import plan_lightweight_teb

    request = _request(base_config, planner_config)
    for array in (
        request.start_pose,
        request.initial_control,
        request.local_goal_world_pose,
        request.static_occupancy,
    ):
        _assert_irreversibly_readonly(array)

    result = plan_lightweight_teb(request)
    assert result.route is not None, result.rejection_reason
    _assert_irreversibly_readonly(result.goal_world_pose)
    for array in (
        result.route.goal_world_pose,
        result.route.band_poses_world,
        result.route.band_interval_dt_s,
        result.route.sample_times_s,
        result.route.sampled_poses_world,
        result.route.sampled_controls,
    ):
        _assert_irreversibly_readonly(array)


def test_dense_static_collision_check_anchors_at_nonzero_request_start(
    base_config: dict, planner_config
) -> None:
    from src.planning.lightweight_teb import plan_lightweight_teb

    grid = build_grid_spec(base_config)
    origin_obstacle = CircleOccluder(
        "unrelated-origin-obstacle",
        "column",
        np.asarray([0.0, 0.0], dtype=np.float64),
        0.25,
    )
    request = _request(
        base_config,
        planner_config,
        start_pose=np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
        goal_pose=np.asarray([2.4, 2.0, 0.0], dtype=np.float32),
        static_occupancy=rasterize_occluder(origin_obstacle, grid).astype(np.float32),
    )

    result = plan_lightweight_teb(request)

    assert result.route is not None, result.rejection_reason
    np.testing.assert_array_equal(result.route.band_poses_world[0], request.start_pose)
    _assert_dynamics(result.route, request, base_config, planner_config)


def test_empty_static_occupancy_skips_rasterization(
    base_config: dict,
    planner_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.planning.lightweight_teb as teb

    calls = 0

    def forbidden_rasterization(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("empty static occupancy must skip rasterization")

    monkeypatch.setattr(teb, "rasterize_footprint", forbidden_rasterization)
    result = teb.plan_static_lightweight_teb(
        _request(base_config, planner_config)
    )

    assert result.route is not None, result.rejection_reason
    assert calls == 0


def test_dense_collision_validation_samples_constant_control_arcs(
    base_config: dict,
    planner_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.planning.lightweight_teb as teb

    grid = build_grid_spec(base_config)
    captured: list[np.ndarray] = []

    def capture_pose(_footprint, pose, _grid):
        captured.append(np.asarray(pose, dtype=np.float64).copy())
        return np.zeros((grid.height, grid.width), dtype=bool)

    monkeypatch.setattr(teb, "rasterize_footprint", capture_pose)
    occluder = CircleOccluder(
        "arc-tree",
        "tree_trunk",
        np.asarray([1.2, 1.0], dtype=np.float64),
        0.35,
    )
    nonempty_static = np.zeros((grid.height, grid.width), dtype=np.float32)
    nonempty_static[0, 0] = 1.0
    request = _request(
        base_config,
        replace(planner_config, initialization_ids=("straight",)),
        occluders=(occluder,),
        static_occupancy=nonempty_static,
    )
    result = teb.plan_lightweight_teb(request)

    assert result.route is not None, result.rejection_reason
    assert np.any(np.abs(result.route.sampled_controls[:, 1]) > 1e-3)
    expected = _dense_route_poses(request, result.route)
    np.testing.assert_allclose(
        np.asarray(captured),
        expected,
        rtol=0.0,
        atol=1e-6,
    )
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
    assert tuple(
        candidate.initialization_id for candidate in result.diagnostics.candidates
    ) == ("straight", "bypass_left", "bypass_right")
    assert any(candidate.valid for candidate in result.diagnostics.candidates)
    clearances = point_signed_distance(
        occluder, _dense_route_poses(request, result.route)[:, :2]
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
    route_clearance = float(np.min(clearances - robot_radius))
    assert route_clearance >= (
        planner_config.represented_occluder_clearance_range_m[0] - 1e-5
    )
    assert route_clearance <= (
        planner_config.represented_occluder_clearance_range_m[1] + 1e-5
    )
    _assert_dynamics(result.route, request, base_config, planner_config)


def test_route_ignores_upper_clearance_for_directly_irrelevant_occluder(
    base_config: dict, planner_config
) -> None:
    from src.planning.lightweight_teb import plan_lightweight_teb

    far_occluder = CircleOccluder(
        "far-column",
        "column",
        np.asarray([6.0, 2.0], dtype=np.float64),
        0.2,
    )
    result = plan_lightweight_teb(
        _request(base_config, planner_config, occluders=(far_occluder,))
    )

    assert result.route is not None, result.rejection_reason
    np.testing.assert_array_equal(
        result.route.goal_world_pose,
        np.asarray([2.4, 0.0, 0.0], dtype=np.float32),
    )


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
    dense_poses = _dense_route_poses(request, result.route)
    for component in (horizontal_arm, vertical_arm):
        clearance = point_signed_distance(
            component, dense_poses[:, :2]
        ) - robot_radius
        assert float(np.min(clearance)) >= (
            planner_config.represented_occluder_clearance_range_m[0] - 1e-5
        )
        assert float(np.min(clearance)) <= (
            planner_config.represented_occluder_clearance_range_m[1] + 1e-5
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
    repeated = plan_lightweight_teb(request)

    assert result.route is None
    assert result.rejection_reason == "teb_static_collision"
    assert result.diagnostics.candidates
    assert all(np.isfinite(candidate.cost) for candidate in result.diagnostics.candidates)
    assert result.diagnostics == repeated.diagnostics
    assert result.goal_world_pose.tobytes() == repeated.goal_world_pose.tobytes()


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
        "goal_world_pose",
        "band_poses_world",
        "band_interval_dt_s",
        "sample_times_s",
        "sampled_poses_world",
        "sampled_controls",
    ):
        assert getattr(first.route, name).tobytes() == getattr(second.route, name).tobytes()
    assert first.route.goal_arrival_time_s == second.route.goal_arrival_time_s
    assert first.route.task_cost == second.route.task_cost
    assert first.diagnostics == second.diagnostics


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
    assert tuple(
        candidate.initialization_id for candidate in result.diagnostics.candidates
    ) == ("straight", "bypass_left", "bypass_right")
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
