"""Trajectory query-map tests against the canonical SOP-02 geometry API."""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts import ARRAY_DTYPE, LocalTrajectory, build_grid_spec
from src.geometry import (
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint,
    rasterize_footprint_sweep,
    world_to_grid,
)
from src.planning.query_maps import (
    build_local_trajectory,
    build_trajectory_query_maps,
)
from src.planning.trajectory_sampler import sample_candidate_rollouts
from src.utils.config import load_config


def test_swept_mask_covers_every_discrete_inflated_footprint() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot = config["robot"]
    footprint = inflate_footprint(
        RectangleFootprint(robot["length_m"], robot["width_m"]),
        robot["inflation_m"],
    )
    candidate = sample_candidate_rollouts(config)[12]

    maps = build_trajectory_query_maps(
        candidate.poses,
        candidate.controls,
        grid=grid,
        footprint=footprint,
        dt_s=config["trajectories"]["dt_s"],
        braking_deceleration_mps2=1.0,
    )

    swept = maps.swept_mask.astype(bool)
    np.testing.assert_array_equal(
        swept,
        rasterize_footprint_sweep(footprint, candidate.poses, grid),
    )
    for pose in candidate.poses:
        footprint_mask = rasterize_footprint(footprint, pose, grid)
        assert np.all(footprint_mask <= swept)


def test_tta_is_minus_one_exactly_outside_swept_volume() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot = config["robot"]
    footprint = inflate_footprint(
        RectangleFootprint(robot["length_m"], robot["width_m"]),
        robot["inflation_m"],
    )
    candidate = sample_candidate_rollouts(config)[12]

    maps = build_trajectory_query_maps(
        candidate.poses,
        candidate.controls,
        grid=grid,
        footprint=footprint,
        dt_s=config["trajectories"]["dt_s"],
        braking_deceleration_mps2=1.0,
    )

    swept = maps.swept_mask.astype(bool)
    assert np.all(maps.tta_map[~swept] == -1.0)
    assert np.all(maps.tta_map[swept] >= 0.0)


def test_centerline_map_contains_every_rollout_pose_center() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot = config["robot"]
    footprint = inflate_footprint(
        RectangleFootprint(robot["length_m"], robot["width_m"]),
        robot["inflation_m"],
    )
    candidate = sample_candidate_rollouts(config)[14]

    maps = build_trajectory_query_maps(
        candidate.poses,
        candidate.controls,
        grid=grid,
        footprint=footprint,
        dt_s=config["trajectories"]["dt_s"],
        braking_deceleration_mps2=1.0,
    )

    indices = world_to_grid(candidate.poses[:, :2], grid)
    assert np.all(maps.centerline_map[indices[:, 0], indices[:, 1]] == 1.0)


def test_braking_map_is_path_distance_minus_stopping_distance() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot = config["robot"]
    footprint = inflate_footprint(
        RectangleFootprint(robot["length_m"], robot["width_m"]),
        robot["inflation_m"],
    )
    candidate = sample_candidate_rollouts(config)[12]
    deceleration = 1.0

    maps = build_trajectory_query_maps(
        candidate.poses,
        candidate.controls,
        grid=grid,
        footprint=footprint,
        dt_s=config["trajectories"]["dt_s"],
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
    config = load_config()
    grid = build_grid_spec(config)
    robot = config["robot"]
    footprint = inflate_footprint(
        RectangleFootprint(robot["length_m"], robot["width_m"]),
        robot["inflation_m"],
    )
    candidate = sample_candidate_rollouts(config)[12]

    with pytest.raises(ValueError, match="deceleration"):
        build_trajectory_query_maps(
            candidate.poses,
            candidate.controls,
            grid=grid,
            footprint=footprint,
            dt_s=config["trajectories"]["dt_s"],
            braking_deceleration_mps2=deceleration,
        )


def test_query_maps_reject_pose_control_length_mismatch() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot = config["robot"]
    footprint = inflate_footprint(
        RectangleFootprint(robot["length_m"], robot["width_m"]),
        robot["inflation_m"],
    )
    candidate = sample_candidate_rollouts(config)[12]

    with pytest.raises(ValueError, match="same length"):
        build_trajectory_query_maps(
            candidate.poses,
            candidate.controls[:-1],
            grid=grid,
            footprint=footprint,
            dt_s=config["trajectories"]["dt_s"],
            braking_deceleration_mps2=1.0,
        )


@pytest.mark.parametrize("dt_s", [0.0, -0.2, np.nan, np.inf])
def test_query_maps_reject_invalid_timestep(dt_s: float) -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot = config["robot"]
    footprint = inflate_footprint(
        RectangleFootprint(robot["length_m"], robot["width_m"]),
        robot["inflation_m"],
    )
    candidate = sample_candidate_rollouts(config)[12]

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
    config = load_config()
    grid = build_grid_spec(config)
    robot = config["robot"]
    footprint = inflate_footprint(
        RectangleFootprint(robot["length_m"], robot["width_m"]),
        robot["inflation_m"],
    )
    candidate = sample_candidate_rollouts(config)[12]
    controls = candidate.controls.copy()
    controls[4, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        build_trajectory_query_maps(
            candidate.poses,
            controls,
            grid=grid,
            footprint=footprint,
            dt_s=config["trajectories"]["dt_s"],
            braking_deceleration_mps2=1.0,
        )


def test_candidate_is_materialized_as_frozen_local_trajectory_contract() -> None:
    config = load_config()
    candidate = sample_candidate_rollouts(config)[12]

    trajectory = build_local_trajectory(
        candidate,
        config,
        braking_deceleration_mps2=1.0,
        task_cost=2.5,
    )

    assert isinstance(trajectory, LocalTrajectory)
    assert trajectory.trajectory_id == candidate.trajectory_id
    assert trajectory.poses.shape == (15, 3)
    assert trajectory.controls.shape == (15, 2)
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


def test_tta_is_nondecreasing_along_straight_centerline_direction() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot = config["robot"]
    footprint = inflate_footprint(
        RectangleFootprint(robot["length_m"], robot["width_m"]),
        robot["inflation_m"],
    )
    candidate = sample_candidate_rollouts(config)[12]

    maps = build_trajectory_query_maps(
        candidate.poses,
        candidate.controls,
        grid=grid,
        footprint=footprint,
        dt_s=config["trajectories"]["dt_s"],
        braking_deceleration_mps2=1.0,
    )

    indices = world_to_grid(candidate.poses[:, :2], grid)
    tta_along_centerline = maps.tta_map[indices[:, 0], indices[:, 1]]
    assert np.all(np.diff(tta_along_centerline) >= 0.0)


def test_straight_centerline_map_has_no_grid_cell_gaps() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot = config["robot"]
    footprint = inflate_footprint(
        RectangleFootprint(robot["length_m"], robot["width_m"]),
        robot["inflation_m"],
    )
    candidate = sample_candidate_rollouts(config)[17]

    maps = build_trajectory_query_maps(
        candidate.poses,
        candidate.controls,
        grid=grid,
        footprint=footprint,
        dt_s=config["trajectories"]["dt_s"],
        braking_deceleration_mps2=1.0,
    )

    indices = world_to_grid(candidate.poses[:, :2], grid)
    row = indices[0, 0]
    start_column = int(indices[:, 1].min())
    end_column = int(indices[:, 1].max())
    assert np.all(maps.centerline_map[row, start_column : end_column + 1] == 1.0)


def test_local_trajectory_rejects_nonfinite_task_cost() -> None:
    config = load_config()
    candidate = sample_candidate_rollouts(config)[12]

    with pytest.raises(ValueError, match="task_cost"):
        build_local_trajectory(
            candidate,
            config,
            braking_deceleration_mps2=1.0,
            task_cost=np.nan,
        )
