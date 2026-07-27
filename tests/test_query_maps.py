"""Long40 trajectory query-map tests against the canonical geometry API."""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts import ARRAY_DTYPE, GridSpec, LocalTrajectory, build_grid_spec
from src.geometry import (
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint,
    rasterize_footprint_sweep,
    world_to_grid,
)
from src.planning.differential_drive import integrate_twist, rollout_constant_control
from src.planning.query_maps import (
    build_local_trajectory,
    build_trajectory_query_maps,
)
from src.planning.trajectory_contracts import CandidateRollout
from src.utils.config import load_config


LONG40_STEPS = 32
LONG40_DT_S = 0.2


def _candidate(*, v: float = 0.6, omega: float = 0.0) -> CandidateRollout:
    poses, controls = rollout_constant_control(
        v=v,
        omega=omega,
        dt_s=LONG40_DT_S,
        steps=LONG40_STEPS,
    )
    return CandidateRollout(
        trajectory_id=f"constant-{v:.2f}-{omega:.2f}",
        poses=poses,
        controls=controls,
        is_stop=v == 0.0 and omega == 0.0,
        is_reverse=v < 0.0,
    )


def _base_geometry() -> tuple[dict, GridSpec, RectangleFootprint]:
    config = load_config()
    grid = build_grid_spec(config)
    robot = config["robot"]
    footprint = inflate_footprint(
        RectangleFootprint(robot["length_m"], robot["width_m"]),
        robot["inflation_m"],
    )
    return config, grid, footprint


def _rollout_control_sequence(controls: np.ndarray) -> np.ndarray:
    pose = np.zeros(3, dtype=ARRAY_DTYPE)
    poses = []
    for v, omega in controls:
        pose = integrate_twist(
            pose,
            v=float(v),
            omega=float(omega),
            dt_s=LONG40_DT_S,
        )
        poses.append(pose)
    return np.asarray(poses, dtype=ARRAY_DTYPE)


def test_swept_mask_covers_every_discrete_inflated_footprint() -> None:
    config, grid, footprint = _base_geometry()
    candidate = _candidate()

    maps = build_trajectory_query_maps(
        candidate.poses,
        candidate.controls,
        grid=grid,
        footprint=footprint,
        dt_s=config["bev"]["future_dt_s"],
        braking_deceleration_mps2=1.0,
    )

    swept = maps.swept_mask.astype(bool)
    anchored_poses = np.vstack(
        (np.zeros((1, 3), dtype=ARRAY_DTYPE), candidate.poses)
    )
    assert np.all(
        rasterize_footprint_sweep(footprint, anchored_poses, grid) <= swept
    )
    for pose in anchored_poses:
        assert np.all(rasterize_footprint(footprint, pose, grid) <= swept)


def test_tta_is_minus_one_exactly_outside_swept_volume() -> None:
    config, grid, footprint = _base_geometry()
    candidate = _candidate()

    maps = build_trajectory_query_maps(
        candidate.poses,
        candidate.controls,
        grid=grid,
        footprint=footprint,
        dt_s=config["bev"]["future_dt_s"],
        braking_deceleration_mps2=1.0,
    )

    swept = maps.swept_mask.astype(bool)
    assert np.all(maps.tta_map[~swept] == -1.0)
    assert np.all(maps.tta_map[swept] >= 0.0)


def test_centerline_map_contains_every_rollout_pose_center() -> None:
    config, grid, footprint = _base_geometry()
    candidate = _candidate(v=0.7, omega=0.2)

    maps = build_trajectory_query_maps(
        candidate.poses,
        candidate.controls,
        grid=grid,
        footprint=footprint,
        dt_s=config["bev"]["future_dt_s"],
        braking_deceleration_mps2=1.0,
    )

    indices = world_to_grid(candidate.poses[:, :2], grid)
    assert np.all(maps.centerline_map[indices[:, 0], indices[:, 1]] == 1.0)
    origin = world_to_grid(np.array([[0.0, 0.0]]), grid)[0]
    assert maps.centerline_map[origin[0], origin[1]] == 1.0


def test_first_control_interval_is_present_in_all_query_maps() -> None:
    grid = GridSpec(
        height=41,
        width=41,
        history_steps=8,
        future_steps=LONG40_STEPS,
        resolution_m=0.1,
    )
    controls = np.zeros((LONG40_STEPS, 2), dtype=ARRAY_DTYPE)
    controls[0] = [2.25, 0.0]
    poses = _rollout_control_sequence(controls)

    maps = build_trajectory_query_maps(
        poses,
        controls,
        grid=grid,
        footprint=RectangleFootprint(0.02, 0.02),
        dt_s=LONG40_DT_S,
        braking_deceleration_mps2=1.0,
    )

    origin = world_to_grid(np.array([[0.0, 0.0]]), grid)[0]
    midpoint = world_to_grid(np.array([[0.2, 0.0]]), grid)[0]
    leading_edge = world_to_grid(np.array([[0.5, 0.0]]), grid)[0]
    assert maps.swept_mask[origin[0], origin[1]] == 1.0
    assert maps.swept_mask[midpoint[0], midpoint[1]] == 1.0
    assert maps.centerline_map[midpoint[0], midpoint[1]] == 1.0
    assert maps.tta_map[origin[0], origin[1]] == pytest.approx(0.0)
    assert maps.tta_map[leading_edge[0], leading_edge[1]] == pytest.approx(0.2)
    expected_braking_margin = 0.45 - 2.25**2 / 2.0
    assert maps.braking_map[
        leading_edge[0], leading_edge[1]
    ] == pytest.approx(expected_braking_margin)


def test_turning_first_interval_uses_differential_drive_arc_not_chord() -> None:
    grid = GridSpec(
        height=61,
        width=61,
        history_steps=8,
        future_steps=LONG40_STEPS,
        resolution_m=0.05,
    )
    yaw_rate = np.pi / (2.0 * LONG40_DT_S)
    controls = np.zeros((LONG40_STEPS, 2), dtype=ARRAY_DTYPE)
    controls[0] = [4.0, yaw_rate]
    poses = _rollout_control_sequence(controls)

    maps = build_trajectory_query_maps(
        poses,
        controls,
        grid=grid,
        footprint=RectangleFootprint(0.02, 0.02),
        dt_s=LONG40_DT_S,
        braking_deceleration_mps2=1.0,
    )

    radius = 4.0 / yaw_rate
    arc_midpoint = np.array(
        [[radius * np.sin(np.pi / 4.0), radius * (1.0 - np.cos(np.pi / 4.0))]]
    )
    midpoint_index = world_to_grid(arc_midpoint, grid)[0]
    assert maps.swept_mask[midpoint_index[0], midpoint_index[1]] == 1.0
    assert maps.centerline_map[midpoint_index[0], midpoint_index[1]] == 1.0


def test_full_rotation_sweep_is_not_lost_to_wrapped_endpoint_yaw() -> None:
    grid = GridSpec(
        height=41,
        width=41,
        history_steps=8,
        future_steps=LONG40_STEPS,
        resolution_m=0.05,
    )
    controls = np.zeros((LONG40_STEPS, 2), dtype=ARRAY_DTYPE)
    controls[0] = [0.0, 2.0 * np.pi / LONG40_DT_S]
    poses = _rollout_control_sequence(controls)

    maps = build_trajectory_query_maps(
        poses,
        controls,
        grid=grid,
        footprint=RectangleFootprint(0.4, 0.05),
        dt_s=LONG40_DT_S,
        braking_deceleration_mps2=1.0,
    )

    rotated_tip = world_to_grid(np.array([[0.0, 0.2]]), grid)[0]
    assert maps.swept_mask[rotated_tip[0], rotated_tip[1]] == 1.0


def test_query_map_arrivals_span_dt_through_long40_horizon() -> None:
    grid = GridSpec(
        height=160,
        width=160,
        history_steps=8,
        future_steps=LONG40_STEPS,
        resolution_m=0.1,
    )
    poses, controls = rollout_constant_control(
        v=1.0,
        omega=0.0,
        dt_s=LONG40_DT_S,
        steps=LONG40_STEPS,
    )

    maps = build_trajectory_query_maps(
        poses,
        controls,
        grid=grid,
        footprint=RectangleFootprint(0.02, 0.02),
        dt_s=LONG40_DT_S,
        braking_deceleration_mps2=1.0,
    )

    first_endpoint_cell = world_to_grid(np.array([[0.25, 0.0]]), grid)[0]
    final_endpoint_cell = world_to_grid(np.array([[6.4, 0.0]]), grid)[0]
    assert maps.tta_map[
        first_endpoint_cell[0], first_endpoint_cell[1]
    ] == pytest.approx(0.2)
    assert maps.tta_map[
        final_endpoint_cell[0], final_endpoint_cell[1]
    ] == pytest.approx(6.4)
    assert maps.braking_map[
        final_endpoint_cell[0], final_endpoint_cell[1]
    ] == pytest.approx(6.4 - 0.5)


def test_braking_map_is_path_distance_minus_stopping_distance() -> None:
    config, grid, footprint = _base_geometry()
    candidate = _candidate()
    deceleration = 1.0

    maps = build_trajectory_query_maps(
        candidate.poses,
        candidate.controls,
        grid=grid,
        footprint=footprint,
        dt_s=config["bev"]["future_dt_s"],
        braking_deceleration_mps2=deceleration,
    )

    swept = maps.swept_mask.astype(bool)
    speed = float(abs(candidate.controls[0, 0]))
    expected = speed * maps.tta_map[swept] - speed**2 / (2.0 * deceleration)
    np.testing.assert_allclose(maps.braking_map[swept], expected, atol=1e-6)
    assert np.all(maps.braking_map[~swept] == 0.0)


@pytest.mark.parametrize("deceleration", [0.0, -1.0, np.nan, np.inf])
def test_query_maps_reject_invalid_braking_deceleration(
    deceleration: float,
) -> None:
    config, grid, footprint = _base_geometry()
    candidate = _candidate()

    with pytest.raises(ValueError, match="deceleration"):
        build_trajectory_query_maps(
            candidate.poses,
            candidate.controls,
            grid=grid,
            footprint=footprint,
            dt_s=config["bev"]["future_dt_s"],
            braking_deceleration_mps2=deceleration,
        )


def test_query_maps_reject_pose_control_length_mismatch() -> None:
    config, grid, footprint = _base_geometry()
    candidate = _candidate()

    with pytest.raises(ValueError, match="same length"):
        build_trajectory_query_maps(
            candidate.poses,
            candidate.controls[:-1],
            grid=grid,
            footprint=footprint,
            dt_s=config["bev"]["future_dt_s"],
            braking_deceleration_mps2=1.0,
        )


@pytest.mark.parametrize("dt_s", [0.0, -0.2, np.nan, np.inf])
def test_query_maps_reject_invalid_timestep(dt_s: float) -> None:
    _, grid, footprint = _base_geometry()
    candidate = _candidate()

    with pytest.raises(ValueError, match="dt_s"):
        build_trajectory_query_maps(
            candidate.poses,
            candidate.controls,
            grid=grid,
            footprint=footprint,
            dt_s=dt_s,
            braking_deceleration_mps2=1.0,
        )


def test_query_maps_reject_nonfinite_controls() -> None:
    config, grid, footprint = _base_geometry()
    candidate = _candidate()
    controls = candidate.controls.copy()
    controls[4, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        build_trajectory_query_maps(
            candidate.poses,
            controls,
            grid=grid,
            footprint=footprint,
            dt_s=config["bev"]["future_dt_s"],
            braking_deceleration_mps2=1.0,
        )


def test_candidate_is_materialized_as_long40_local_trajectory() -> None:
    config = load_config()
    candidate = _candidate()

    trajectory = build_local_trajectory(
        candidate,
        config,
        braking_deceleration_mps2=1.0,
        task_cost=2.5,
    )

    assert isinstance(trajectory, LocalTrajectory)
    assert trajectory.trajectory_id == candidate.trajectory_id
    assert trajectory.poses.shape == (LONG40_STEPS, 3)
    assert trajectory.controls.shape == (LONG40_STEPS, 2)
    for array in (
        trajectory.swept_mask,
        trajectory.tta_map,
        trajectory.braking_map,
        trajectory.centerline_map,
    ):
        assert array.shape == (160, 160)
        assert array.dtype == ARRAY_DTYPE
        assert np.isfinite(array).all()
    assert trajectory.task_cost == 2.5
    assert trajectory.metadata["is_stop"] is False
    assert trajectory.metadata["is_reverse"] is False
    assert (
        trajectory.metadata["pose_time_layout_version"]
        == "future_endpoints_dt_to_horizon_v1"
    )
    assert trajectory.metadata["first_pose_time_s"] == pytest.approx(0.2)
    assert trajectory.metadata["last_pose_time_s"] == pytest.approx(6.4)
    assert trajectory.metadata["dt_s"] == pytest.approx(0.2)
    assert trajectory.metadata["trajectory_steps"] == LONG40_STEPS


def test_local_trajectory_rejects_nonfinite_task_cost() -> None:
    with pytest.raises(ValueError, match="task_cost"):
        build_local_trajectory(
            _candidate(),
            load_config(),
            braking_deceleration_mps2=1.0,
            task_cost=np.nan,
        )
