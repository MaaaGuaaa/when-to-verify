from __future__ import annotations

import numpy as np
import pytest

from src.generation.history_visibility import (
    INELIGIBLE_HISTORY_VISIBILITY,
    SEEN_THEN_OCCLUDED,
    UNSEEN_IN_HISTORY_WINDOW,
    classify_sop05r_history,
    validate_sop05r_history_metadata,
)
from src.generation.sop05r_contracts import SOP05R_HISTORY_POLICY_VERSION


@pytest.mark.parametrize(
    "vector",
    [
        [True, True, True, True, False, False, False, False],
        [False, True, True, True, True, False, False, False],
    ],
)
def test_sop05r_seen_history_accepts_one_contiguous_early_visible_run(
    vector: list[bool],
) -> None:
    assessment = classify_sop05r_history(np.asarray(vector, dtype=np.bool_))

    assert assessment.regime == SEEN_THEN_OCCLUDED
    assert assessment.last_visible_index == max(
        index for index, visible in enumerate(vector) if visible
    )
    assert assessment.trailing_hidden_frames >= 2


def test_sop05r_unseen_history_requires_all_eight_frames_hidden() -> None:
    assessment = classify_sop05r_history(np.zeros(8, dtype=np.bool_))

    assert assessment.regime == UNSEEN_IN_HISTORY_WINDOW
    assert assessment.last_visible_index is None
    assert assessment.trailing_hidden_frames == 8


@pytest.mark.parametrize(
    "vector",
    [
        [False, False, True, False, False, False, False, False],
        [True, True, False, True, False, False, False, False],
        [True, True, True, True, True, True, True, False],
        [False, True, True, False, True, False, False, False],
    ],
)
def test_sop05r_seen_history_rejects_late_flickering_or_short_occlusion(
    vector: list[bool],
) -> None:
    assessment = classify_sop05r_history(np.asarray(vector, dtype=np.bool_))

    assert assessment.regime == INELIGIBLE_HISTORY_VISIBILITY


@pytest.mark.parametrize(
    "vector",
    [
        np.zeros(7, dtype=np.bool_),
        np.zeros(8, dtype=np.uint8),
        [False] * 8,
        np.zeros((8, 1), dtype=np.bool_),
    ],
)
def test_sop05r_history_rejects_noncanonical_vectors(vector: object) -> None:
    with pytest.raises((TypeError, ValueError), match="visibility"):
        classify_sop05r_history(vector)


def test_sop05r_history_metadata_must_match_recomputed_assessment() -> None:
    vector = np.asarray(
        [False, True, True, True, True, False, False, False],
        dtype=np.bool_,
    )
    metadata = {
        "target_history_visibility_policy_version": SOP05R_HISTORY_POLICY_VERSION,
        "target_history_visibility_regime": SEEN_THEN_OCCLUDED,
        "target_history_last_visible_index": 4,
        "target_history_trailing_hidden_frames": 3,
    }

    assessment = validate_sop05r_history_metadata(vector, metadata)

    assert assessment.last_visible_index == 4
    for key in metadata:
        changed = dict(metadata)
        changed[key] = "drift" if key.endswith("version") else -1
        with pytest.raises(ValueError, match="metadata"):
            validate_sop05r_history_metadata(vector, changed)


def test_m5_history_classifier_prefers_four_initial_visible_frames() -> None:
    from src.generation.history_visibility import (
        classify_sop05r_seen_then_occluded_history,
    )

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
    from src.generation.history_visibility import (
        classify_sop05r_seen_then_occluded_history,
    )

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
    from src.generation.history_visibility import (
        classify_sop05r_seen_then_occluded_history,
    )

    assessment = classify_sop05r_seen_then_occluded_history(
        np.asarray(blocked, dtype=np.bool_),
        decision_index=7,
        minimum_visible_frames=4,
        minimum_occluded_frames=1,
    )

    assert not assessment.eligible
    assert not assessment.preferred
    assert assessment.observed_class == "ineligible"
