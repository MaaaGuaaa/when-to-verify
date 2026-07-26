"""Constant-control differential-drive rollout."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np

from src.contracts import ARRAY_DTYPE, POSE_TIME_LAYOUT_VERSION
from src.geometry import wrap_angle


DEFAULT_ANGULAR_DECELERATION_RADPS2 = 1.6


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def integrate_twist(
    pose: np.ndarray, *, v: float, omega: float, dt_s: float
) -> np.ndarray:
    """Integrate one constant differential-drive control interval."""

    array = np.asarray(pose)
    if array.shape != (3,) or array.dtype.kind not in "iuf":
        raise ValueError("pose must be a numeric pose with shape (3,)")
    start = np.asarray(array, dtype=np.float64)
    if not np.isfinite(start).all():
        raise ValueError("pose must be finite")
    linear = _finite_real(v, name="v")
    angular = _finite_real(omega, name="omega")
    dt = _finite_real(dt_s, name="dt_s")
    if dt <= 0.0:
        raise ValueError("dt_s must be positive")
    yaw = float(start[2])
    result = start.copy()
    if abs(angular) <= 1e-12:
        distance = linear * dt
        result[0] += distance * np.cos(yaw)
        result[1] += distance * np.sin(yaw)
    else:
        end_yaw = yaw + angular * dt
        radius = linear / angular
        result[0] += radius * (np.sin(end_yaw) - np.sin(yaw))
        result[1] -= radius * (np.cos(end_yaw) - np.cos(yaw))
        result[2] = end_yaw
    result[2] = wrap_angle(result[2])
    return result


def rollout_constant_control(
    *, v: float, omega: float, dt_s: float, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out future endpoints for constant control intervals.

    ``controls[i]`` acts on ``[i * dt_s, (i + 1) * dt_s]`` and ``poses[i]``
    is the resulting pose at ``(i + 1) * dt_s``.
    """
    if not np.isfinite([v, omega]).all():
        raise ValueError("v and omega must be finite")
    if (
        not np.isfinite(dt_s)
        or dt_s <= 0.0
        or isinstance(steps, (bool, np.bool_))
        or not isinstance(steps, (int, np.integer))
        or steps <= 0
    ):
        raise ValueError("time grid requires finite dt_s > 0 and integer steps > 0")
    times = (np.arange(steps, dtype=np.float64) + 1.0) * dt_s
    poses = np.zeros((steps, 3), dtype=np.float64)
    if omega == 0.0:
        poses[:, 0] = v * times
    else:
        yaw = omega * times
        poses[:, 0] = (v / omega) * np.sin(yaw)
        poses[:, 1] = (v / omega) * (1.0 - np.cos(yaw))
        poses[:, 2] = yaw
    controls = np.tile(np.asarray([v, omega], dtype=ARRAY_DTYPE), (steps, 1))
    return poses.astype(ARRAY_DTYPE), controls


__all__ = (
    "DEFAULT_ANGULAR_DECELERATION_RADPS2",
    "integrate_twist",
    "rollout_constant_control",
)
