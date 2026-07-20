"""Leakage-safe geometric and entropy baselines for verification actions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.contracts import (
    ARRAY_DTYPE,
    STATE_CHANNELS,
    TRAJECTORY_CHANNELS,
    VerificationSample,
)
from src.evaluation.verification_metrics import evaluate_verification_predictions


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


def evaluate_verification_baselines(
    samples: Sequence[VerificationSample], *, huber_delta: float
) -> dict[str, object]:
    """Evaluate all legal deployment-side baselines on complete sample groups."""

    rows = tuple(samples)
    if not rows or any(not isinstance(sample, VerificationSample) for sample in rows):
        raise ValueError("baseline evaluation requires VerificationSample values")
    score_functions = {
        "visible_area": lambda sample: visible_area_score(
            state_channels=sample.state_channels,
            verification_fov_mask=sample.verification_fov_mask,
        ),
        "critical_swept_coverage": lambda sample: critical_swept_coverage_score(
            state_channels=sample.state_channels,
            trajectory_channels=sample.trajectory_channels,
            verification_fov_mask=sample.verification_fov_mask,
        ),
        "occupancy_entropy": lambda sample: occupancy_entropy_reduction_score(
            state_channels=sample.state_channels,
            verification_fov_mask=sample.verification_fov_mask,
        ),
    }
    values = np.asarray([sample.value_target for sample in rows], dtype=np.float64)
    useful = np.asarray([sample.useful_target for sample in rows], dtype=np.int64)
    groups = tuple(str(sample.metadata["ranking_group_id"]) for sample in rows)
    actions = tuple(sample.verification_action_id for sample in rows)
    result: dict[str, object] = {}
    for name, function in score_functions.items():
        scores = np.asarray([function(sample) for sample in rows], dtype=np.float64)
        report = evaluate_verification_predictions(
            value_prediction=scores,
            useful_probability=np.full(scores.shape, 0.5, dtype=np.float64),
            value_target=values,
            useful_target=useful,
            group_ids=groups,
            action_ids=actions,
            huber_delta=huber_delta,
        )
        result[name] = {
            key: report[key]
            for key in (
                "pairwise_accuracy",
                "pair_count",
                "top1_regret_mean",
                "top_two_selection_rate",
                "selected_action_counts",
                "selected_action_proportions",
            )
        }
    return result


__all__ = (
    "critical_swept_coverage_score",
    "evaluate_verification_baselines",
    "occupancy_entropy_reduction_score",
    "visible_area_score",
)
