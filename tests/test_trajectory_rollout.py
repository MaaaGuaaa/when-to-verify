"""Analytic and contract tests for SOP-04 trajectory rollout and sampling."""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts import ARRAY_DTYPE
from src.planning.differential_drive import rollout_constant_control
from src.planning.trajectory_sampler import (
    sample_candidate_rollouts,
    sample_trajectory_primitives,
)
from src.utils.config import load_config


def test_straight_rollout_matches_analytic_solution() -> None:
    poses, controls = rollout_constant_control(
        v=0.6,
        omega=0.0,
        dt_s=0.2,
        steps=15,
    )

    expected_x = 0.6 * np.arange(15, dtype=np.float64) * 0.2
    np.testing.assert_allclose(poses[:, 0], expected_x, atol=1e-6)
    np.testing.assert_allclose(poses[:, 1:], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        controls,
        np.tile(np.array([0.6, 0.0]), (15, 1)),
        atol=1e-6,
    )


@pytest.mark.parametrize("omega", [-0.4, 0.4])
def test_constant_turn_rollout_matches_analytic_arc(omega: float) -> None:
    v = 0.6
    dt_s = 0.2
    poses, _ = rollout_constant_control(v=v, omega=omega, dt_s=dt_s, steps=15)

    times = np.arange(15, dtype=np.float64) * dt_s
    expected = np.column_stack(
        (
            (v / omega) * np.sin(omega * times),
            (v / omega) * (1.0 - np.cos(omega * times)),
            omega * times,
        )
    )
    np.testing.assert_allclose(poses, expected, atol=1e-6)


def test_rollout_matches_frozen_shape_and_dtype_contract() -> None:
    poses, controls = rollout_constant_control(
        v=0.4,
        omega=0.0,
        dt_s=0.2,
        steps=15,
    )

    assert poses.shape == (15, 3)
    assert controls.shape == (15, 2)
    assert poses.dtype == ARRAY_DTYPE
    assert controls.dtype == ARRAY_DTYPE
    assert np.isfinite(poses).all()
    assert np.isfinite(controls).all()


@pytest.mark.parametrize(
    ("v", "omega"),
    [(np.nan, 0.0), (0.4, np.inf)],
)
def test_rollout_rejects_nonfinite_controls(v: float, omega: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        rollout_constant_control(v=v, omega=omega, dt_s=0.2, steps=15)


@pytest.mark.parametrize(
    ("dt_s", "steps"),
    [(0.0, 15), (-0.2, 15), (0.2, 0), (0.2, 1.5)],
)
def test_rollout_rejects_invalid_time_grid(dt_s: float, steps: int) -> None:
    with pytest.raises(ValueError, match="time grid"):
        rollout_constant_control(v=0.4, omega=0.0, dt_s=dt_s, steps=steps)


def test_default_sampler_builds_configured_forward_grid_and_marked_stop() -> None:
    primitives = sample_trajectory_primitives(load_config())

    forward = [item for item in primitives if not item.is_stop]
    stops = [item for item in primitives if item.is_stop]
    assert len(forward) == 4 * 5
    assert len(stops) == 1
    assert not any(item.is_reverse for item in primitives)
    assert (stops[0].v, stops[0].omega) == (0.0, 0.0)


def test_reverse_primitives_require_stress_gate_and_probability() -> None:
    config = load_config()
    config["trajectories"]["reverse_probability"] = 1.0

    primitives = sample_trajectory_primitives(
        config,
        reverse_stress=True,
        rng=np.random.default_rng(7),
    )

    reverse = [item for item in primitives if item.is_reverse]
    assert len(reverse) == 2 * 5
    assert all(item.v < 0.0 and not item.is_stop for item in reverse)


def test_sampled_candidates_roll_out_on_configured_time_grid() -> None:
    candidates = sample_candidate_rollouts(load_config())

    assert len(candidates) >= 6
    assert all(item.poses.shape == (15, 3) for item in candidates)
    assert all(item.controls.shape == (15, 2) for item in candidates)
    assert all(item.poses.dtype == ARRAY_DTYPE for item in candidates)
    assert all(item.controls.dtype == ARRAY_DTYPE for item in candidates)
    stop = next(item for item in candidates if item.is_stop)
    np.testing.assert_array_equal(stop.poses, 0.0)
    np.testing.assert_array_equal(stop.controls, 0.0)


def test_reverse_rollout_keeps_explicit_reverse_marker() -> None:
    config = load_config()
    config["trajectories"]["reverse_probability"] = 1.0

    candidates = sample_candidate_rollouts(
        config,
        reverse_stress=True,
        rng=np.random.default_rng(11),
    )

    reverse = [item for item in candidates if item.is_reverse]
    assert reverse
    assert all(np.all(item.controls[:, 0] < 0.0) for item in reverse)


def test_sampler_rejects_time_grid_that_conflicts_with_frozen_bev() -> None:
    config = load_config()
    config["trajectories"]["horizon_s"] = 3.2

    with pytest.raises(ValueError, match="BEV contract"):
        sample_candidate_rollouts(config)


def test_reverse_sampler_rejects_invalid_probability() -> None:
    config = load_config()
    config["trajectories"]["reverse_probability"] = 1.1

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sample_trajectory_primitives(config, reverse_stress=True)
