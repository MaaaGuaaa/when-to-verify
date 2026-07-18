from __future__ import annotations

import hashlib
import json

import pytest

from src.generation import sop05_selection as selection


SEED = 60_505
SELECTION_VERSION = "sop05_diversity_total_selection_v1"


def _candidate(
    index: int,
    *,
    base_state_id: str = "base-a",
    trajectory_id: str = "trajectory-a",
    object_type: str = "human",
    occluder_type: str = "wall",
    crossing_side: int = -1,
    conflict_index: int = 4,
) -> selection.Sop05SelectionCandidate:
    return selection.Sop05SelectionCandidate(
        base_state_id=base_state_id,
        trajectory_id=trajectory_id,
        generated_event_id=f"event-{index:02d}",
        object_type=object_type,
        occluder_type=occluder_type,
        crossing_side=crossing_side,
        conflict_index=conflict_index,
    )


def _expected_key(candidate: selection.Sop05SelectionCandidate) -> tuple[str, ...]:
    payload = json.dumps(
        [
            SELECTION_VERSION,
            SEED,
            candidate.base_state_id,
            candidate.trajectory_id,
            candidate.generated_event_id,
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return (
        hashlib.blake2b(payload, digest_size=16).hexdigest(),
        candidate.base_state_id,
        candidate.trajectory_id,
        candidate.generated_event_id,
    )


def test_diversity_total_selector_versions_and_key_are_frozen() -> None:
    candidate = _candidate(7)
    assert selection.SOP05_DIVERSITY_TOTAL_SELECTION_VERSION == SELECTION_VERSION
    assert selection.SOP05_TOTAL_QUOTA_SELECTION_VERSION == SELECTION_VERSION
    assert selection.SOP05_PAIR_REPORT_VERSION == "sop05_pair_generation_report_v3"
    assert selection.SOP05_RUN_PRODUCER_VERSION == "sop05_generation_run_v5"
    assert selection.sop05_selection_key(SEED, candidate) == _expected_key(candidate)


def test_diversity_selector_is_independent_of_worker_candidate_order() -> None:
    candidates = (
        _candidate(1),
        _candidate(2, base_state_id="base-b"),
        _candidate(3, trajectory_id="trajectory-b"),
        _candidate(4, object_type="vehicle"),
    )
    expected = selection.select_sop05_event_ids(
        candidates, seed=SEED, accepted_quota=3
    )
    assert selection.select_sop05_event_ids(
        tuple(reversed(candidates)), seed=SEED, accepted_quota=3
    ) == expected


def test_diversity_is_a_soft_ranking_and_never_a_hard_quota() -> None:
    duplicate_pattern = tuple(_candidate(index) for index in range(7))
    assert len(
        selection.select_sop05_event_ids(
            duplicate_pattern, seed=SEED, accepted_quota=7
        )
    ) == 7


def test_greedy_ranking_prefers_new_diversity_before_stable_tie_breaks() -> None:
    candidates = (
        _candidate(1),
        _candidate(2),
        _candidate(
            3,
            base_state_id="base-b",
            trajectory_id="trajectory-b",
            object_type="vehicle",
            occluder_type="pillar",
            crossing_side=1,
            conflict_index=8,
        ),
    )
    selected = selection.select_sop05_event_ids(
        candidates, seed=SEED, accepted_quota=2
    )
    assert "event-03" in selected


def test_selector_returns_every_available_candidate_below_requested_total() -> None:
    candidates = (_candidate(1), _candidate(2))
    assert set(
        selection.select_sop05_event_ids(
            candidates, seed=SEED, accepted_quota=5
        )
    ) == {"event-01", "event-02"}


@pytest.mark.parametrize("seed", [-1, True, 1.5, "7"])
def test_selector_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        selection.select_sop05_event_ids((), seed=seed, accepted_quota=1)


@pytest.mark.parametrize("accepted_quota", [0, -1, True, 1.5, "3"])
def test_selector_rejects_invalid_total(accepted_quota: object) -> None:
    with pytest.raises(ValueError, match="quota"):
        selection.select_sop05_event_ids(
            (), seed=SEED, accepted_quota=accepted_quota
        )


def test_selector_rejects_duplicate_event_or_canonical_keys() -> None:
    candidate = _candidate(1)
    with pytest.raises(ValueError, match="unique"):
        selection.select_sop05_event_ids(
            (candidate, candidate), seed=SEED, accepted_quota=1
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"base_state_id": ""},
        {"trajectory_id": ""},
        {"generated_event_id": ""},
        {"object_type": ""},
        {"occluder_type": ""},
        {"crossing_side": 0},
        {"conflict_index": -1},
    ],
)
def test_canonical_candidate_rejects_malformed_identity_or_diversity(
    updates: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "base_state_id": "base-a",
        "trajectory_id": "trajectory-a",
        "generated_event_id": "event-a",
        "object_type": "human",
        "occluder_type": "wall",
        "crossing_side": -1,
        "conflict_index": 4,
    }
    values.update(updates)
    with pytest.raises((TypeError, ValueError)):
        selection.Sop05SelectionCandidate(**values)
