from __future__ import annotations

import hashlib
import json

import pytest

from src.generation import sop05_selection as selection


SEED = 60_505
SELECTION_VERSION = "sop05_total_quota_selection_v1"


def _expected_key(event_id: str) -> tuple[str, str]:
    payload = json.dumps(
        [SELECTION_VERSION, SEED, event_id],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest(), event_id


def _ranked_ids(entries: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    return tuple(
        event_id
        for event_id, _ in sorted(
            entries,
            key=lambda item: _expected_key(item[0]),
        )
    )


def test_global_total_selector_versions_and_key_are_frozen() -> None:
    assert (
        getattr(selection, "SOP05_TOTAL_QUOTA_SELECTION_VERSION", None)
        == SELECTION_VERSION
    )
    assert selection.SOP05_RUN_PRODUCER_VERSION == "sop05_generation_run_v4"
    assert selection.sop05_selection_key(SEED, "event-07") == _expected_key(
        "event-07"
    )


def test_global_total_selector_is_independent_of_input_order() -> None:
    entries = tuple(
        (f"event-{index:02d}", "structural") for index in range(12)
    )
    expected = _ranked_ids(entries)[:10]

    assert selection.select_sop05_event_ids(
        entries,
        seed=SEED,
        accepted_quota=10,
    ) == expected
    assert selection.select_sop05_event_ids(
        reversed(entries),
        seed=SEED,
        accepted_quota=10,
    ) == expected


def test_global_total_selector_does_not_reserve_event_kind_quotas() -> None:
    ids = tuple(f"event-{index:02d}" for index in range(15))
    mixed_entries = tuple(
        (event_id, ("environment", "structural", "mixed")[index % 3])
        for index, event_id in enumerate(ids)
    )
    structural_entries = tuple((event_id, "structural") for event_id in ids)

    expected = _ranked_ids(mixed_entries)[:10]
    assert selection.select_sop05_event_ids(
        mixed_entries,
        seed=SEED,
        accepted_quota=10,
    ) == expected
    assert selection.select_sop05_event_ids(
        structural_entries,
        seed=SEED,
        accepted_quota=10,
    ) == expected


def test_global_total_selector_returns_all_available_when_quota_is_larger() -> None:
    entries = (
        ("event-environment", "environment"),
        ("event-structural", "structural"),
        ("event-mixed", "mixed"),
    )

    assert selection.select_sop05_event_ids(
        entries,
        seed=SEED,
        accepted_quota=10,
    ) == _ranked_ids(entries)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "60505"])
def test_global_total_selector_rejects_invalid_seed_for_empty_input(
    seed: object,
) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        selection.select_sop05_event_ids(
            (),
            seed=seed,
            accepted_quota=10,
        )


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (
            (("event-01", "structural"), ("event-01", "mixed")),
            "unique",
        ),
        ((("event-01", "unknown"),), "kind"),
        ((('', "structural"),), "nonempty string"),
        (((7, "structural"),), "nonempty string"),
    ],
)
def test_global_total_selector_rejects_malformed_entries(
    entries: tuple[tuple[object, str], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        selection.select_sop05_event_ids(
            entries,
            seed=SEED,
            accepted_quota=1,
        )


@pytest.mark.parametrize(
    "entry",
    [("event-01",), ("event-01", "structural", "unexpected")],
)
def test_global_total_selector_rejects_malformed_entry_arity(
    entry: tuple[str, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match="exactly generated_event_id and event_kind",
    ):
        selection.select_sop05_event_ids(
            (entry,),
            seed=SEED,
            accepted_quota=1,
        )


@pytest.mark.parametrize("accepted_quota", [0, -1, True, 1.0, "10"])
def test_global_total_selector_rejects_nonpositive_or_noninteger_quota(
    accepted_quota: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        selection.select_sop05_event_ids(
            (("event-01", "structural"),),
            seed=SEED,
            accepted_quota=accepted_quota,
        )
