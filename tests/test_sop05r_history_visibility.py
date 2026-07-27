from __future__ import annotations

import numpy as np
import pytest

from src.generation.history_visibility import (
    classify_sop05r_seen_then_occluded_history,
)


def test_m5_history_classifier_prefers_four_initial_visible_frames() -> None:
    assessment = classify_sop05r_seen_then_occluded_history(
        np.asarray(
            [False, False, False, False, True, False, False, False],
            dtype=np.bool_,
        ),
        decision_index=7,
        minimum_visible_frames=4,
        minimum_occluded_frames=1,
    )

    assert assessment.eligible
    assert assessment.preferred
    assert assessment.observed_class == "prefix4_visible_then_occluded"
    assert assessment.blocked_indices == (4,)


def test_m5_history_classifier_accepts_nonprefix_seen_then_occluded_as_fallback() -> None:
    assessment = classify_sop05r_seen_then_occluded_history(
        np.asarray(
            [False, True, False, False, False, False, True, False],
            dtype=np.bool_,
        ),
        decision_index=7,
        minimum_visible_frames=4,
        minimum_occluded_frames=1,
    )

    assert assessment.eligible
    assert not assessment.preferred
    assert assessment.observed_class == "fallback_seen_then_occluded"
    assert assessment.blocked_indices == (1, 6)


@pytest.mark.parametrize(
    "blocked",
    [
        [True, True, True, True, False, False, False, False],
        [True, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False],
        [True, True, True, True, True, True, True, True],
    ],
)
def test_m5_history_classifier_rejects_histories_without_seen_then_occluded_order(
    blocked: list[bool],
) -> None:
    assessment = classify_sop05r_seen_then_occluded_history(
        np.asarray(blocked, dtype=np.bool_),
        decision_index=7,
        minimum_visible_frames=4,
        minimum_occluded_frames=1,
    )

    assert not assessment.eligible
    assert not assessment.preferred
    assert assessment.observed_class == "ineligible"
