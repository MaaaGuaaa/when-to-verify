"""Frozen deterministic total-first SOP-05 diversity selection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass


SOP05_DIVERSITY_TOTAL_SELECTION_VERSION = (
    "sop05_diversity_total_selection_v1"
)
# Kept as the public import used by the producer and consumer.  The value is
# intentionally the new formal token; the previous quota selector is retired.
SOP05_TOTAL_QUOTA_SELECTION_VERSION = (
    SOP05_DIVERSITY_TOTAL_SELECTION_VERSION
)
SOP05_PAIR_REPORT_VERSION = "sop05_pair_generation_report_v4"
SOP05_RUN_PRODUCER_VERSION = "sop05_generation_run_v6"
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
    )


def select_sop05_event_ids(
    accepted_events: Iterable[Sop05SelectionCandidate],
    *,
    seed: int,
    accepted_quota: int,
) -> tuple[str, ...]:
    """Select the requested total with deterministic soft diversity ranking.

    Diversity affects rank only.  It never reserves per-kind slots and never
    reduces the returned total when enough unique candidates are available.
    """

    validated_seed = _validate_selection_seed(seed)
    if type(accepted_quota) is not int or accepted_quota <= 0:
        raise ValueError("SOP05 accepted quota must be a positive integer")

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
    seen_diversity: tuple[set[object], ...] = tuple(set() for _ in range(6))
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
