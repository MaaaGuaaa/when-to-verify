"""Constant-control differential-drive rollout."""

from __future__ import annotations

import numpy as np

from src.contracts import ARRAY_DTYPE


def rollout_constant_control(
    *, v: float, omega: float, dt_s: float, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out one constant ``(v, omega)`` command from the local origin."""
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
    times = np.arange(steps, dtype=np.float64) * dt_s
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
