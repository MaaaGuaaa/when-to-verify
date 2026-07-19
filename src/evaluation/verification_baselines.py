"""Leakage-safe geometric and entropy baselines for verification actions."""

from __future__ import annotations

import numpy as np

from src.contracts import ARRAY_DTYPE, STATE_CHANNELS, TRAJECTORY_CHANNELS


def _legal_arrays(
    *,
    state_channels: np.ndarray,
    verification_fov_mask: np.ndarray,
    trajectory_channels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not isinstance(state_channels, np.ndarray):
        raise TypeError("state_channels must be an ndarray")
    if state_channels.dtype != ARRAY_DTYPE:
        raise TypeError("state_channels must be float32")
    if (
        state_channels.ndim != 3
        or state_channels.shape[0] != len(STATE_CHANNELS)
        or min(state_channels.shape[1:]) <= 0
    ):
        raise ValueError("state_channels shape is invalid")
    spatial_shape = state_channels.shape[1:]
    if (
        not isinstance(verification_fov_mask, np.ndarray)
        or verification_fov_mask.dtype != ARRAY_DTYPE
        or verification_fov_mask.shape != (1, *spatial_shape)
    ):
        raise ValueError("verification_fov_mask must be float32 [1,H,W]")
    if trajectory_channels is not None and (
        not isinstance(trajectory_channels, np.ndarray)
        or trajectory_channels.dtype != ARRAY_DTYPE
        or trajectory_channels.shape
        != (len(TRAJECTORY_CHANNELS), *spatial_shape)
    ):
        raise ValueError("trajectory_channels must be float32 [4,H,W]")
    arrays = [state_channels, verification_fov_mask]
    if trajectory_channels is not None:
        arrays.append(trajectory_channels)
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("verification baseline inputs must be finite")
    fov = verification_fov_mask[0]
    unobservable = state_channels[STATE_CHANNELS.index("current_unobservable_mask")]
    if not np.isin(fov, (0.0, 1.0)).all() or not np.isin(
        unobservable, (0.0, 1.0)
    ).all():
        raise ValueError("FOV and current-unobservable masks must be binary")
    return state_channels, fov != 0.0, trajectory_channels


def _newly_visible(
    state_channels: np.ndarray, verification_fov_mask: np.ndarray
) -> np.ndarray:
    unobservable = (
        state_channels[STATE_CHANNELS.index("current_unobservable_mask")] != 0.0
    )
    return verification_fov_mask & unobservable


def visible_area_score(
    *, state_channels: np.ndarray, verification_fov_mask: np.ndarray
) -> float:
    """Count cells expected to become visible from the current blind region."""

    state, fov, _ = _legal_arrays(
        state_channels=state_channels,
        verification_fov_mask=verification_fov_mask,
    )
    return float(np.count_nonzero(_newly_visible(state, fov)))


def critical_swept_coverage_score(
    *,
    state_channels: np.ndarray,
    trajectory_channels: np.ndarray,
    verification_fov_mask: np.ndarray,
) -> float:
    """Fraction of the nominal swept mask newly exposed by an action."""

    state, fov, trajectory = _legal_arrays(
        state_channels=state_channels,
        trajectory_channels=trajectory_channels,
        verification_fov_mask=verification_fov_mask,
    )
    assert trajectory is not None
    swept = trajectory[TRAJECTORY_CHANNELS.index("swept_volume_mask")]
    if not np.isin(swept, (0.0, 1.0)).all():
        raise ValueError("swept_volume_mask must be binary")
    swept_mask = swept != 0.0
    denominator = int(np.count_nonzero(swept_mask))
    if denominator == 0:
        return 0.0
    numerator = np.count_nonzero(_newly_visible(state, fov) & swept_mask)
    return float(numerator / denominator)


def occupancy_entropy_reduction_score(
    *, state_channels: np.ndarray, verification_fov_mask: np.ndarray
) -> float:
    """Prior occupancy entropy removed in newly visible cells.

    The legal last-seen occupancy is decayed toward an uninformative 0.5 using
    the normalized occlusion-age channel. Observation is assumed certain, so
    its posterior entropy is zero. No post-action occupancy is consumed.
    """

    state, fov, _ = _legal_arrays(
        state_channels=state_channels,
        verification_fov_mask=verification_fov_mask,
    )
    last_seen = state[STATE_CHANNELS.index("last_seen_occupancy")]
    age = state[STATE_CHANNELS.index("occlusion_age_map")]
    if (
        np.any(last_seen < 0.0)
        or np.any(last_seen > 1.0)
        or np.any(age < 0.0)
        or np.any(age > 1.0)
    ):
        raise ValueError("last-seen occupancy and occlusion age must lie in [0,1]")
    probability = (1.0 - age.astype(np.float64)) * last_seen + 0.5 * age
    entropy = np.zeros_like(probability, dtype=np.float64)
    uncertain = (probability > 0.0) & (probability < 1.0)
    values = probability[uncertain]
    entropy[uncertain] = -values * np.log(values) - (1.0 - values) * np.log(
        1.0 - values
    )
    return float(np.sum(entropy[_newly_visible(state, fov)], dtype=np.float64))


__all__ = (
    "critical_swept_coverage_score",
    "occupancy_entropy_reduction_score",
    "visible_area_score",
)
