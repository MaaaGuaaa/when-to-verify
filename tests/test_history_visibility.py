from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.generation.event_sampler import load_generator_config
from src.generation.history_visibility import (
    HISTORY_VISIBILITY_POLICY_VERSION,
    INELIGIBLE_HISTORY_VISIBILITY,
    SEEN_THEN_OCCLUDED,
    UNSEEN_IN_HISTORY_WINDOW,
    allocate_history_visibility_counts,
    classify_history_visibility,
    normalize_history_visibility_policy,
)


def _policy() -> dict[str, object]:
    return {
        "policy_version": HISTORY_VISIBILITY_POLICY_VERSION,
        "min_trailing_hidden_frames": 2,
        "weights": {
            SEEN_THEN_OCCLUDED: 0.8,
            UNSEEN_IN_HISTORY_WINDOW: 0.2,
        },
    }


def test_classifies_unseen_in_finite_history_window() -> None:
    assessment = classify_history_visibility(
        np.zeros(8, dtype=np.bool_),
        normalize_history_visibility_policy(_policy()),
    )

    assert assessment.regime == UNSEEN_IN_HISTORY_WINDOW
    assert assessment.last_visible_index is None
    assert assessment.trailing_hidden_frames == 8


def test_classifies_seen_then_contiguously_occluded() -> None:
    visibility = np.asarray(
        [False, True, True, True, True, True, False, False],
        dtype=np.bool_,
    )

    assessment = classify_history_visibility(
        visibility,
        normalize_history_visibility_policy(_policy()),
    )

    assert assessment.regime == SEEN_THEN_OCCLUDED
    assert assessment.last_visible_index == 5
    assert assessment.trailing_hidden_frames == 2


def test_rejects_history_hidden_only_at_current_frame() -> None:
    visibility = np.asarray(
        [False, True, True, True, True, True, True, False],
        dtype=np.bool_,
    )

    assessment = classify_history_visibility(
        visibility,
        normalize_history_visibility_policy(_policy()),
    )

    assert assessment.regime == INELIGIBLE_HISTORY_VISIBILITY
    assert assessment.last_visible_index == 6
    assert assessment.trailing_hidden_frames == 1


@pytest.mark.parametrize(
    "visibility",
    [
        np.zeros(7, dtype=np.bool_),
        np.zeros(8, dtype=np.uint8),
        [False] * 8,
    ],
)
def test_classification_rejects_noncanonical_visibility_vectors(
    visibility: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="visibility"):
        classify_history_visibility(
            visibility,
            normalize_history_visibility_policy(_policy()),
        )


def test_normalizes_and_authenticates_history_visibility_policy() -> None:
    raw = _policy()
    raw["weights"] = {
        SEEN_THEN_OCCLUDED: 8.0,
        UNSEEN_IN_HISTORY_WINDOW: 2.0,
    }

    policy = normalize_history_visibility_policy(raw)

    assert policy.as_dict() == {
        "policy_version": HISTORY_VISIBILITY_POLICY_VERSION,
        "min_trailing_hidden_frames": 2,
        "weights": {
            SEEN_THEN_OCCLUDED: 0.8,
            UNSEEN_IN_HISTORY_WINDOW: 0.2,
        },
    }
    assert len(policy.digest) == 64
    assert policy.digest == normalize_history_visibility_policy(raw).digest


def test_stratified_allocation_is_exact_for_ten_event_eighty_twenty_mix() -> None:
    policy = normalize_history_visibility_policy(_policy())

    counts = allocate_history_visibility_counts(
        10,
        policy,
        seed=42,
        namespace="pair-a",
    )

    assert counts == {
        SEEN_THEN_OCCLUDED: 8,
        UNSEEN_IN_HISTORY_WINDOW: 2,
    }
    assert counts == allocate_history_visibility_counts(
        10,
        policy,
        seed=42,
        namespace="pair-a",
    )


def test_production_generator_configs_freeze_eighty_twenty_history_policy() -> None:
    for relative_path in (
        "configs/generator_train.yaml",
        "configs/generator_test.yaml",
    ):
        config = load_generator_config(Path(relative_path))
        policy = config["target_history_visibility"]

        assert policy.as_dict() == _policy()
        assert len(policy.digest) == 64


def test_single_event_allocations_cover_both_strata_across_pair_seeds() -> None:
    policy = normalize_history_visibility_policy(_policy())
    selected = {
        next(
            regime
            for regime, count in allocate_history_visibility_counts(
                1,
                policy,
                seed=seed,
                namespace=f"pair-{seed}",
            ).items()
            if count == 1
        )
        for seed in range(64)
    }

    assert selected == {SEEN_THEN_OCCLUDED, UNSEEN_IN_HISTORY_WINDOW}


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"policy_version": "wrong"}, "policy_version"),
        ({"min_trailing_hidden_frames": 0}, "trailing"),
        ({"min_trailing_hidden_frames": 8}, "trailing"),
        ({"weights": {SEEN_THEN_OCCLUDED: 0.0, UNSEEN_IN_HISTORY_WINDOW: 0.0}}, "weight"),
    ],
)
def test_policy_rejects_invalid_contract_values(
    update: dict[str, object],
    message: str,
) -> None:
    raw = _policy()
    raw.update(update)

    with pytest.raises((TypeError, ValueError), match=message):
        normalize_history_visibility_policy(raw)
