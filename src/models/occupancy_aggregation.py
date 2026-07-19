"""Trajectory-risk aggregation over predicted hidden-object occupancy.

The future axis uses endpoint semantics: index ``t`` represents
``(t + 1) * dt_s``.  Query masks are converted to boolean masks before any
calculation, so each batch/time/cell tuple contributes at most once.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def future_endpoint_times(*, future_steps: int = 15, dt_s: float = 0.2) -> np.ndarray:
    """Return float32 endpoint times ``dt_s .. future_steps * dt_s``."""
    if isinstance(future_steps, bool) or not isinstance(future_steps, int) or future_steps < 1:
        raise ValueError("future_steps must be a positive integer")
    if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        raise ValueError("dt_s must be positive and finite")
    return np.arange(1, future_steps + 1, dtype=np.float32) * np.float32(dt_s)


def _is_torch(value: Any) -> bool:
    return torch.is_tensor(value)


def _validate_inputs(occupancy: Any, robot_future_footprints: Any) -> tuple[Any, Any]:
    if _is_torch(occupancy) != _is_torch(robot_future_footprints):
        raise TypeError("occupancy and robot_future_footprints must use the same backend")
    if getattr(occupancy, "ndim", None) != 4 or getattr(robot_future_footprints, "ndim", None) != 4:
        raise ValueError("occupancy and robot_future_footprints must have rank 4 [B,T,H,W]")
    if tuple(occupancy.shape) != tuple(robot_future_footprints.shape):
        raise ValueError("occupancy and robot_future_footprints must have the same shape")

    if _is_torch(occupancy):
        if occupancy.dtype != torch.float32 or robot_future_footprints.dtype != torch.float32:
            raise ValueError("occupancy and robot_future_footprints must be float32")
        if not bool(torch.isfinite(occupancy).all()):
            raise ValueError("occupancy probabilities must be finite and in [0,1]")
        if bool(((occupancy < 0.0) | (occupancy > 1.0)).any()):
            raise ValueError("occupancy probabilities must be finite and in [0,1]")
        if not bool(torch.isfinite(robot_future_footprints).all()) or bool(
            (robot_future_footprints < 0.0).any()
        ):
            raise ValueError("robot_future_footprints must be finite and nonnegative")
    else:
        if not isinstance(occupancy, np.ndarray) or not isinstance(
            robot_future_footprints, np.ndarray
        ):
            raise TypeError("occupancy and robot_future_footprints must be NumPy arrays or tensors")
        if occupancy.dtype != np.float32 or robot_future_footprints.dtype != np.float32:
            raise ValueError("occupancy and robot_future_footprints must be float32")
        if not np.isfinite(occupancy).all() or np.logical_or(
            occupancy < 0.0, occupancy > 1.0
        ).any():
            raise ValueError("occupancy probabilities must be finite and in [0,1]")
        if not np.isfinite(robot_future_footprints).all() or (
            robot_future_footprints < 0.0
        ).any():
            raise ValueError("robot_future_footprints must be finite and nonnegative")

    return occupancy, robot_future_footprints > 0.0


def weighted_swept_volume_risk(
    occupancy: Any,
    robot_future_footprints: Any,
    *,
    dt_s: float = 0.2,
    sigma_time_s: float = 2.0,
) -> Any:
    """Return normalized time-weighted mean occupancy inside the footprint.

    Frame weight is ``exp(-((t + 1) * dt_s) / sigma_time_s)``.  The
    denominator contains the same weight for every selected cell, which keeps
    the score in ``[0,1]``.  Samples with an empty query mask return zero.
    """
    occupancy, mask = _validate_inputs(occupancy, robot_future_footprints)
    if not math.isfinite(float(sigma_time_s)) or float(sigma_time_s) <= 0.0:
        raise ValueError("sigma_time_s must be positive and finite")
    times = future_endpoint_times(future_steps=int(occupancy.shape[1]), dt_s=dt_s)

    if _is_torch(occupancy):
        weights = torch.as_tensor(times, device=occupancy.device, dtype=occupancy.dtype)
        weights = torch.exp(-weights / float(sigma_time_s)).view(1, -1, 1, 1)
        selected_weights = mask.to(occupancy.dtype) * weights
        numerator = (occupancy * selected_weights).sum(dim=(1, 2, 3))
        denominator = selected_weights.sum(dim=(1, 2, 3))
        return torch.where(denominator > 0.0, numerator / denominator.clamp_min(1e-12), 0.0)

    weights = np.exp(-times.astype(np.float64) / float(sigma_time_s)).astype(np.float32)
    selected_weights = mask.astype(np.float32) * weights.reshape(1, -1, 1, 1)
    numerator = np.sum(occupancy * selected_weights, axis=(1, 2, 3), dtype=np.float32)
    denominator = np.sum(selected_weights, axis=(1, 2, 3), dtype=np.float32)
    result = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return result


def probabilistic_union_risk(occupancy: Any, robot_future_footprints: Any) -> Any:
    """Return ``1 - product(1-p)`` over unique selected time/cell tuples."""
    occupancy, mask = _validate_inputs(occupancy, robot_future_footprints)
    if _is_torch(occupancy):
        selected = torch.where(mask, occupancy, torch.zeros_like(occupancy))
        return 1.0 - torch.prod(1.0 - selected.flatten(start_dim=1), dim=1)
    selected = np.where(mask, occupancy, np.float32(0.0)).astype(np.float32, copy=False)
    return (1.0 - np.prod(1.0 - selected, axis=(1, 2, 3), dtype=np.float32)).astype(
        np.float32,
        copy=False,
    )


# Explicit aliases used in reports and older experiment notes.
swept_volume_weighted_sum = weighted_swept_volume_risk
probabilistic_union = probabilistic_union_risk


__all__ = [
    "future_endpoint_times",
    "probabilistic_union",
    "probabilistic_union_risk",
    "swept_volume_weighted_sum",
    "weighted_swept_volume_risk",
]
