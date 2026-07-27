"""Analytic and contract tests for the retained Long40 rollout primitive."""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts import ARRAY_DTYPE
from src.planning.differential_drive import rollout_constant_control


LONG40_STEPS = 32
LONG40_DT_S = 0.2


def test_straight_rollout_matches_analytic_solution() -> None:
    poses, controls = rollout_constant_control(
        v=0.6,
        omega=0.0,
        dt_s=LONG40_DT_S,
        steps=LONG40_STEPS,
    )

    expected_x = (
        0.6 * (np.arange(LONG40_STEPS, dtype=np.float64) + 1.0) * LONG40_DT_S
    )
    np.testing.assert_allclose(poses[:, 0], expected_x, atol=1e-6)
    np.testing.assert_allclose(poses[:, 1:], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        controls,
        np.tile(np.array([0.6, 0.0]), (LONG40_STEPS, 1)),
        atol=1e-6,
    )


@pytest.mark.parametrize("omega", [-0.4, 0.4])
def test_constant_turn_rollout_matches_analytic_arc(omega: float) -> None:
    v = 0.6
    poses, _ = rollout_constant_control(
        v=v,
        omega=omega,
        dt_s=LONG40_DT_S,
        steps=LONG40_STEPS,
    )

    times = (
        np.arange(LONG40_STEPS, dtype=np.float64) + 1.0
    ) * LONG40_DT_S
    expected = np.column_stack(
        (
            (v / omega) * np.sin(omega * times),
            (v / omega) * (1.0 - np.cos(omega * times)),
            omega * times,
        )
    )
    np.testing.assert_allclose(poses, expected, atol=1e-6)


def test_rollout_matches_long40_shape_and_dtype_contract() -> None:
    poses, controls = rollout_constant_control(
        v=0.4,
        omega=0.0,
        dt_s=LONG40_DT_S,
        steps=LONG40_STEPS,
    )

    assert poses.shape == (LONG40_STEPS, 3)
    assert controls.shape == (LONG40_STEPS, 2)
    assert poses.dtype == ARRAY_DTYPE
    assert controls.dtype == ARRAY_DTYPE
    assert np.isfinite(poses).all()
    assert np.isfinite(controls).all()


def test_rollout_poses_are_future_endpoints_through_long40_horizon() -> None:
    poses, controls = rollout_constant_control(
        v=1.0,
        omega=0.0,
        dt_s=LONG40_DT_S,
        steps=LONG40_STEPS,
    )

    assert poses[0, 0] == pytest.approx(0.2)
    assert poses[-1, 0] == pytest.approx(6.4)
    np.testing.assert_array_equal(
        controls,
        np.tile(
            np.array([1.0, 0.0], dtype=ARRAY_DTYPE),
            (LONG40_STEPS, 1),
        ),
    )


@pytest.mark.parametrize(
    ("v", "omega"),
    [(np.nan, 0.0), (0.4, np.inf)],
)
def test_rollout_rejects_nonfinite_controls(v: float, omega: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        rollout_constant_control(
            v=v,
            omega=omega,
            dt_s=LONG40_DT_S,
            steps=LONG40_STEPS,
        )


@pytest.mark.parametrize(
    ("dt_s", "steps"),
    [
        (0.0, LONG40_STEPS),
        (-LONG40_DT_S, LONG40_STEPS),
        (LONG40_DT_S, 0),
        (LONG40_DT_S, 1.5),
    ],
)
def test_rollout_rejects_invalid_time_grid(dt_s: float, steps: int) -> None:
    with pytest.raises(ValueError, match="time grid"):
        rollout_constant_control(v=0.4, omega=0.0, dt_s=dt_s, steps=steps)
