"""Frozen deterministic history-stratified SOP-05 diversity selection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

from src.generation.history_visibility import (
    HISTORY_VISIBILITY_REGIMES,
    HistoryVisibilityPolicy,
    allocate_history_visibility_counts,
)


SOP05_DIVERSITY_TOTAL_SELECTION_VERSION = (
    "sop05_history_stratified_selection_v2"
)
# Kept as the public import used by the producer and consumer.  The value is
# intentionally the new formal token; the previous quota selector is retired.
SOP05_TOTAL_QUOTA_SELECTION_VERSION = (
    SOP05_DIVERSITY_TOTAL_SELECTION_VERSION
)
SOP05_PAIR_REPORT_VERSION = "sop05_pair_generation_report_v5"
SOP05_RUN_PRODUCER_VERSION = "sop05_generation_run_v7"
SOP05_EVENT_KIND_ORDER = ("environment",)


@dataclass(frozen=True)
class Sop05SelectionCandidate:
    """Canonical identity plus soft-diversity attributes for one candidate."""

    base_state_id: str
    trajectory_id: str
    generated_event_id: str
    object_type: str
    occluder_type: str
    crossing_side: int
    conflict_index: int
    history_visibility_regime: str

    def __post_init__(self) -> None:
        for name in (
            "base_state_id",
            "trajectory_id",
            "generated_event_id",
            "object_type",
            "occluder_type",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"SOP05 selection {name} must be a string")
            if not value:
                raise ValueError(f"SOP05 selection {name} must be nonempty")
        if type(self.crossing_side) is not int or self.crossing_side not in {
            -1,
            1,
        }:
            raise ValueError("SOP05 selection crossing_side must be -1 or 1")
        if type(self.conflict_index) is not int or self.conflict_index < 0:
            raise ValueError(
                "SOP05 selection conflict_index must be a nonnegative int"
            )
        if self.history_visibility_regime not in HISTORY_VISIBILITY_REGIMES:
            raise ValueError(
                "SOP05 selection history_visibility_regime is invalid"
            )


@dataclass(frozen=True)
class Sop05SelectionResult:
    """Selected identities plus hard history-stratum quota evidence."""

    event_ids: tuple[str, ...]
    required_counts: dict[str, int]
    selected_counts: dict[str, int]
    deficits: dict[str, int]
    quota_met: bool


def _validate_selection_seed(seed: object) -> int:
    if type(seed) is not int or seed < 0:
        raise ValueError("SOP05 selection seed must be a nonnegative integer")
    return seed


def sop05_selection_key(
    seed: int,
    candidate: Sop05SelectionCandidate,
) -> tuple[str, str, str, str]:
    """Return the frozen cross-process tie-break key for one candidate."""

    validated_seed = _validate_selection_seed(seed)
    if not isinstance(candidate, Sop05SelectionCandidate):
        raise TypeError("SOP05 selection candidate has the wrong type")
    payload = json.dumps(
        [
            SOP05_DIVERSITY_TOTAL_SELECTION_VERSION,
            validated_seed,
            candidate.base_state_id,
            candidate.trajectory_id,
            candidate.generated_event_id,
            candidate.history_visibility_regime,
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


def _diversity_values(candidate: Sop05SelectionCandidate) -> tuple[object, ...]:
    return (
        candidate.base_state_id,
        candidate.trajectory_id,
        candidate.object_type,
        candidate.occluder_type,
        candidate.crossing_side,
        candidate.conflict_index,
        candidate.history_visibility_regime,
    )


def _select_diverse_event_ids(
    candidates: list[Sop05SelectionCandidate],
    *,
    seed: int,
    accepted_quota: int,
) -> tuple[str, ...]:
    validated_seed = _validate_selection_seed(seed)
    candidates.sort(
        key=lambda candidate: (
            candidate.base_state_id,
            candidate.trajectory_id,
            candidate.generated_event_id,
        )
    )
    remaining = sorted(
        candidates,
        key=lambda candidate: sop05_selection_key(validated_seed, candidate),
    )
    selected: list[str] = []
    seen_diversity: tuple[set[object], ...] = tuple(set() for _ in range(7))
    while remaining and len(selected) < accepted_quota:
        def ranking(candidate: Sop05SelectionCandidate) -> tuple[object, ...]:
            novelty = sum(
                value not in seen_values
                for value, seen_values in zip(
                    _diversity_values(candidate),
                    seen_diversity,
                    strict=True,
                )
            )
            return (-novelty, *sop05_selection_key(validated_seed, candidate))

        chosen = min(remaining, key=ranking)
        remaining.remove(chosen)
        selected.append(chosen.generated_event_id)
        for value, seen_values in zip(
            _diversity_values(chosen), seen_diversity, strict=True
        ):
            seen_values.add(value)

    return tuple(selected)


def select_sop05_event_ids(
    accepted_events: Iterable[Sop05SelectionCandidate],
    *,
    seed: int,
    accepted_quota: int,
    history_visibility_policy: HistoryVisibilityPolicy,
) -> Sop05SelectionResult:
    """Select an exact history-stratified total with soft within-stratum diversity."""

    validated_seed = _validate_selection_seed(seed)
    if type(accepted_quota) is not int or accepted_quota <= 0:
        raise ValueError("SOP05 accepted quota must be a positive integer")
    if not isinstance(history_visibility_policy, HistoryVisibilityPolicy):
        raise TypeError("SOP05 history visibility policy has the wrong type")

    candidates: list[Sop05SelectionCandidate] = []
    seen_event_ids: set[str] = set()
    seen_keys: set[tuple[str, str, str]] = set()
    for candidate in accepted_events:
        if not isinstance(candidate, Sop05SelectionCandidate):
            raise TypeError("SOP05 selection candidate has the wrong type")
        canonical_key = (
            candidate.base_state_id,
            candidate.trajectory_id,
            candidate.generated_event_id,
        )
        if (
            candidate.generated_event_id in seen_event_ids
            or canonical_key in seen_keys
        ):
            raise ValueError("SOP05 selection candidate identities must be unique")
        seen_event_ids.add(candidate.generated_event_id)
        seen_keys.add(canonical_key)
        candidates.append(candidate)

    required_counts = allocate_history_visibility_counts(
        accepted_quota,
        history_visibility_policy,
        seed=validated_seed,
        namespace="sop05-global-selection",
    )
    selected_by_regime: dict[str, tuple[str, ...]] = {}
    for regime in HISTORY_VISIBILITY_REGIMES:
        selected_by_regime[regime] = _select_diverse_event_ids(
            [
                candidate
                for candidate in candidates
                if candidate.history_visibility_regime == regime
            ],
            seed=validated_seed,
            accepted_quota=required_counts[regime],
        ) if required_counts[regime] > 0 else ()

    selected_candidates = {
        candidate.generated_event_id: candidate
        for candidate in candidates
        if candidate.generated_event_id
        in {
            event_id
            for event_ids in selected_by_regime.values()
            for event_id in event_ids
        }
    }
    event_ids = tuple(
        candidate.generated_event_id
        for candidate in sorted(
            selected_candidates.values(),
            key=lambda candidate: sop05_selection_key(
                validated_seed, candidate
            ),
        )
    )
    selected_counts = {
        regime: len(selected_by_regime[regime])
        for regime in HISTORY_VISIBILITY_REGIMES
    }
    deficits = {
        regime: required_counts[regime] - selected_counts[regime]
        for regime in HISTORY_VISIBILITY_REGIMES
    }
    return Sop05SelectionResult(
        event_ids=event_ids,
        required_counts=required_counts,
        selected_counts=selected_counts,
        deficits=deficits,
        quota_met=(
            len(event_ids) == accepted_quota
            and all(deficit == 0 for deficit in deficits.values())
        ),
    )
