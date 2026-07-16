"""Trajectory-contract and pure-dynamics filter tests for SOP-04."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.contracts import ARRAY_DTYPE, build_grid_spec
from src.geometry import world_to_grid
from src.planning.trajectory_filters import (
    filter_trajectory_candidates,
    trajectory_rejection_reasons,
)
from src.planning.trajectory_sampler import sample_candidate_rollouts
from src.utils.config import load_config


def test_dynamics_filter_rejects_linear_speed_above_robot_limit() -> None:
    config = load_config()
    candidate = sample_candidate_rollouts(config)[0]
    controls = candidate.controls.copy()
    controls[:, 0] = config["robot"]["max_linear_speed_mps"] + 0.1
    candidate = replace(candidate, controls=controls)

    reasons = trajectory_rejection_reasons(candidate, config)

    assert "linear_speed_limit" in reasons


def test_dynamics_filter_rejects_yaw_rate_above_robot_limit() -> None:
    config = load_config()
    candidate = sample_candidate_rollouts(config)[0]
    controls = candidate.controls.copy()
    controls[:, 1] = config["robot"]["max_angular_speed_radps"] + 0.1
    candidate = replace(candidate, controls=controls)

    reasons = trajectory_rejection_reasons(candidate, config)

    assert "angular_speed_limit" in reasons


def test_filter_rejects_nonfinite_trajectory_arrays() -> None:
    config = load_config()
    candidate = sample_candidate_rollouts(config)[0]
    poses = candidate.poses.copy()
    poses[3, 1] = np.nan
    candidate = replace(candidate, poses=poses)

    reasons = trajectory_rejection_reasons(candidate, config)

    assert "nonfinite" in reasons


def test_filter_rejects_shape_that_breaks_frozen_horizon() -> None:
    config = load_config()
    candidate = sample_candidate_rollouts(config)[0]
    candidate = replace(candidate, controls=candidate.controls[:-1])

    reasons = trajectory_rejection_reasons(candidate, config)

    assert reasons == ("shape_mismatch",)


def test_filter_rejects_arrays_outside_frozen_float_dtype() -> None:
    config = load_config()
    candidate = sample_candidate_rollouts(config)[0]
    assert candidate.controls.dtype == ARRAY_DTYPE
    candidate = replace(candidate, controls=candidate.controls.astype(np.float64))

    reasons = trajectory_rejection_reasons(candidate, config)

    assert "dtype_mismatch" in reasons


def test_float32_control_exactly_at_limit_is_accepted() -> None:
    config = load_config()
    candidate = sample_candidate_rollouts(config)[0]
    assert np.max(np.abs(candidate.controls[:, 1])) == np.float32(0.8)

    reasons = trajectory_rejection_reasons(candidate, config)

    assert reasons == ()


def test_dynamics_filter_rejects_configured_acceleration_limit() -> None:
    config = load_config()
    candidate = sample_candidate_rollouts(config)[0]
    controls = candidate.controls.copy()
    controls[1:, 0] = 0.8
    candidate = replace(candidate, controls=controls)

    reasons = trajectory_rejection_reasons(
        candidate,
        config,
        max_linear_acceleration_mps2=1.0,
    )

    assert "linear_acceleration_limit" in reasons


def test_dynamics_filter_rejects_configured_yaw_acceleration_limit() -> None:
    config = load_config()
    candidate = sample_candidate_rollouts(config)[0]
    controls = candidate.controls.copy()
    controls[1:, 1] = 0.8
    candidate = replace(candidate, controls=controls)

    reasons = trajectory_rejection_reasons(
        candidate,
        config,
        max_angular_acceleration_radps2=1.0,
    )

    assert "angular_acceleration_limit" in reasons


def test_acceleration_filter_includes_transition_from_current_control() -> None:
    config = load_config()
    candidate = next(
        item
        for item in sample_candidate_rollouts(config)
        if np.allclose(item.controls[0], np.array([0.8, 0.0]))
    )

    reasons = trajectory_rejection_reasons(
        candidate,
        config,
        initial_control=np.array([0.0, 0.0], dtype=np.float32),
        max_linear_acceleration_mps2=1.0,
    )

    assert "linear_acceleration_limit" in reasons


def test_stationary_candidate_requires_explicit_stop_marker() -> None:
    config = load_config()
    candidates = sample_candidate_rollouts(config)
    candidate = candidates[0]
    stationary = replace(
        candidate,
        poses=np.zeros_like(candidate.poses),
        controls=np.zeros_like(candidate.controls),
    )
    stop = next(item for item in candidates if item.is_stop)

    assert "invalid_stagnation" in trajectory_rejection_reasons(stationary, config)
    assert trajectory_rejection_reasons(stop, config) == ()


def test_stop_marker_cannot_hide_nonzero_motion() -> None:
    config = load_config()
    candidate = sample_candidate_rollouts(config)[0]
    incorrectly_marked = replace(candidate, is_stop=True)

    reasons = trajectory_rejection_reasons(incorrectly_marked, config)

    assert "invalid_stop_marker" in reasons


def test_reverse_marker_must_match_control_direction() -> None:
    config = load_config()
    config["trajectories"]["reverse_probability"] = 1.0
    candidates = sample_candidate_rollouts(
        config,
        reverse_stress=True,
        rng=np.random.default_rng(3),
    )
    reverse = next(item for item in candidates if item.is_reverse)
    forward = next(
        item for item in candidates if not item.is_reverse and not item.is_stop
    )

    assert "invalid_reverse_marker" in trajectory_rejection_reasons(
        replace(reverse, is_reverse=False), config
    )
    assert "invalid_reverse_marker" in trajectory_rejection_reasons(
        replace(forward, is_reverse=True), config
    )


def test_filter_report_tracks_acceptance_and_rejection_reasons() -> None:
    config = load_config()
    candidates = list(sample_candidate_rollouts(config))
    invalid = replace(
        candidates[0],
        controls=np.zeros_like(candidates[0].controls),
        poses=np.zeros_like(candidates[0].poses),
    )
    candidates[0] = invalid

    report = filter_trajectory_candidates(candidates, config)

    assert len(report.accepted) >= 6
    assert len(report.rejected) == 1
    assert report.acceptance_rate >= 0.70
    assert report.rejection_counts == {"invalid_stagnation": 1}


def test_filter_report_applies_requested_initial_acceleration_limit() -> None:
    config = load_config()
    candidate = next(
        item
        for item in sample_candidate_rollouts(config)
        if np.allclose(item.controls[0], np.array([0.8, 0.0]))
    )

    report = filter_trajectory_candidates(
        [candidate],
        config,
        initial_control=np.zeros(2, dtype=np.float32),
        max_linear_acceleration_mps2=1.0,
    )

    assert report.acceptance_rate == 0.0
    assert report.rejection_counts == {"linear_acceleration_limit": 1}


def test_filter_rejects_inflated_footprint_outside_bev() -> None:
    config = load_config()
    candidate = sample_candidate_rollouts(config)[12]
    poses = candidate.poses.copy()
    poses[-1] = np.array([7.6, 0.0, 0.0], dtype=np.float32)
    candidate = replace(candidate, poses=poses)

    reasons = trajectory_rejection_reasons(candidate, config)

    assert "bev_out_of_bounds" in reasons


def test_static_collision_filters_every_intersecting_trajectory() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    occupancy = np.zeros((grid.height, grid.width), dtype=bool)
    origin = world_to_grid([[0.0, 0.0]], grid)[0]
    occupancy[origin[0], origin[1]] = True
    candidates = sample_candidate_rollouts(config)[:6]

    report = filter_trajectory_candidates(
        candidates,
        config,
        static_occupancy=occupancy,
    )

    assert len(report.accepted) == 0
    assert len(report.rejected) == len(candidates)
    assert report.rejection_counts == {"static_collision": len(candidates)}


@pytest.mark.parametrize(
    "limits",
    [
        {"max_linear_acceleration_mps2": -1.0},
        {"max_linear_acceleration_mps2": np.nan},
        {"max_angular_acceleration_radps2": np.inf},
    ],
)
def test_filter_rejects_invalid_acceleration_thresholds(limits: dict) -> None:
    config = load_config()
    candidate = sample_candidate_rollouts(config)[0]

    with pytest.raises(ValueError, match="acceleration"):
        trajectory_rejection_reasons(candidate, config, **limits)
