"""Frozen deterministic total-quota selection shared by SOP-05 I/O."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable


SOP05_TOTAL_QUOTA_SELECTION_VERSION = "sop05_total_quota_selection_v1"
SOP05_PAIR_REPORT_VERSION = "sop05_pair_generation_report_v2"
SOP05_RUN_PRODUCER_VERSION = "sop05_generation_run_v4"
SOP05_EVENT_KIND_ORDER = ("environment", "structural", "mixed")


def _validate_selection_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("SOP05 selection seed must be a nonnegative integer")
    return seed


def sop05_selection_key(seed: int, generated_event_id: str) -> tuple[str, str]:
    """Return the frozen cross-process ordering key for one generated event."""

    validated_seed = _validate_selection_seed(seed)
    if not isinstance(generated_event_id, str) or not generated_event_id:
        raise ValueError("SOP05 selection event ID must be a nonempty string")
    payload = json.dumps(
        [
            SOP05_TOTAL_QUOTA_SELECTION_VERSION,
            validated_seed,
            generated_event_id,
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return (
        hashlib.blake2b(payload, digest_size=16).hexdigest(),
        generated_event_id,
    )


def select_sop05_event_ids(
    accepted_events: Iterable[tuple[str, str]],
    *,
    seed: int,
    accepted_quota: int,
) -> tuple[str, ...]:
    """Select a total quota in the frozen producer selection order."""

    validated_seed = _validate_selection_seed(seed)
    if (
        isinstance(accepted_quota, bool)
        or not isinstance(accepted_quota, int)
        or accepted_quota <= 0
    ):
        raise ValueError("SOP05 accepted quota must be a positive integer")

    event_ids: list[str] = []
    seen: set[str] = set()
    for entry in accepted_events:
        try:
            generated_event_id, event_kind = entry
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SOP05 selection entries must contain exactly "
                "generated_event_id and event_kind"
            ) from exc
        if not isinstance(generated_event_id, str) or not generated_event_id:
            raise ValueError("SOP05 selection event ID must be a nonempty string")
        if event_kind not in SOP05_EVENT_KIND_ORDER:
            raise ValueError("SOP05 selection event kind is unsupported")
        if generated_event_id in seen:
            raise ValueError("SOP05 selection event IDs must be unique")
        seen.add(generated_event_id)
        event_ids.append(generated_event_id)

    return tuple(
        sorted(
            event_ids,
            key=lambda event_id: sop05_selection_key(
                validated_seed,
                event_id,
            ),
        )[:accepted_quota]
    )
