"""Current Long40 history-visibility classification for SOP05R."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_HISTORY_STEPS = 8
_PREFERRED_CLASS = "prefix4_visible_then_occluded"
_FALLBACK_CLASS = "fallback_seen_then_occluded"
_INELIGIBLE_CLASS = "ineligible"


@dataclass(frozen=True)
class SeenThenOccludedHistoryAssessment:
    """One shared M5/M6 classification of a synchronized history window."""

    decision_index: int
    observed_class: str
    blocked_indices: tuple[int, ...]
    visible_frames: int
    occluded_frames: int
    preferred: bool
    eligible: bool


def classify_sop05r_seen_then_occluded_history(
    blocked: object,
    *,
    decision_index: int,
    minimum_visible_frames: int,
    minimum_occluded_frames: int = 1,
) -> SeenThenOccludedHistoryAssessment:
    """Classify one Long40 history as preferred, fallback, or ineligible."""

    if not isinstance(blocked, np.ndarray):
        raise TypeError("centerline blocked sequence must be a numpy array")
    if blocked.ndim != 1 or blocked.dtype != np.bool_:
        raise ValueError("centerline blocked sequence must be a boolean vector")
    if (
        isinstance(decision_index, (bool, np.bool_))
        or not isinstance(decision_index, (int, np.integer))
    ):
        raise TypeError("decision_index must be an integer")
    decision_index = int(decision_index)
    if decision_index < _HISTORY_STEPS - 1 or decision_index >= blocked.size:
        raise ValueError("decision_index must have eight synchronized history frames")
    if (
        isinstance(minimum_visible_frames, (bool, np.bool_))
        or not isinstance(minimum_visible_frames, (int, np.integer))
    ):
        raise TypeError("minimum_visible_frames must be an integer")
    minimum_visible_frames = int(minimum_visible_frames)
    if not 1 <= minimum_visible_frames < _HISTORY_STEPS:
        raise ValueError("minimum_visible_frames must lie within [1, 7]")
    if (
        isinstance(minimum_occluded_frames, (bool, np.bool_))
        or not isinstance(minimum_occluded_frames, (int, np.integer))
    ):
        raise TypeError("minimum_occluded_frames must be an integer")
    minimum_occluded_frames = int(minimum_occluded_frames)
    if not 1 <= minimum_occluded_frames < _HISTORY_STEPS:
        raise ValueError("minimum_occluded_frames must lie within [1, 7]")

    history_start = decision_index - _HISTORY_STEPS + 1
    history = blocked[history_start : decision_index + 1]
    visible_frames = int(np.count_nonzero(~history))
    occluded_frames = int(np.count_nonzero(history))
    blocked_indices = tuple(
        history_start + int(index) for index in np.flatnonzero(history)
    )
    preferred = bool((~history[:4]).all() and history[4:].any())
    last_occluded_index = int(np.flatnonzero(history)[-1]) if occluded_frames else -1
    seen_before_occlusion = bool(
        last_occluded_index > 0 and np.any(~history[:last_occluded_index])
    )
    fallback = bool(
        visible_frames >= minimum_visible_frames
        and occluded_frames >= minimum_occluded_frames
        and seen_before_occlusion
    )
    observed_class = (
        _PREFERRED_CLASS
        if preferred
        else (_FALLBACK_CLASS if fallback else _INELIGIBLE_CLASS)
    )
    return SeenThenOccludedHistoryAssessment(
        decision_index=decision_index,
        observed_class=observed_class,
        blocked_indices=blocked_indices,
        visible_frames=visible_frames,
        occluded_frames=occluded_frames,
        preferred=preferred,
        eligible=preferred or fallback,
    )
