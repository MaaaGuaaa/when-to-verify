"""Leakage-audit tests over hand-constructed provenance manifests."""

from __future__ import annotations

import copy

import pytest


def _clean_provenance_manifest() -> list[dict[str, object]]:
    return [
        {
            "split": split,
            "source_recording_id": f"recording-{split}",
            "session_id": f"session-{split}",
            "source_participant_id": f"participant-{split}",
            "ped_snippet_id": f"snippet-{split}",
            "pair_group_id": f"pair-{split}",
            "seed_namespace": f"split/{split}/generator",
        }
        for split in ("train", "calibration", "val", "test")
    ]


def test_clean_manifest_has_zero_overlap_for_every_audited_source():
    from src.datasets.split_manager import audit_split_leakage

    report = audit_split_leakage(_clean_provenance_manifest())

    assert report["status"] == "ok"
    assert report["total_overlap_count"] == 0
    assert set(report["fields"]) == {
        "recording",
        "session",
        "participant",
        "snippet",
        "pair_group",
        "seed_namespace",
    }
    assert all(
        field_report["overlap_count"] == 0
        for field_report in report["fields"].values()
    )


@pytest.mark.parametrize(
    "audit_field, source_field",
    [
        ("recording", "source_recording_id"),
        ("session", "session_id"),
        ("participant", "source_participant_id"),
        ("snippet", "ped_snippet_id"),
        ("pair_group", "pair_group_id"),
        ("seed_namespace", "seed_namespace"),
    ],
)
def test_injected_source_overlap_is_reported_and_rejected(audit_field, source_field):
    from src.datasets.split_manager import (
        SplitLeakageError,
        assert_no_split_leakage,
        audit_split_leakage,
    )

    manifest = copy.deepcopy(_clean_provenance_manifest())
    manifest[1][source_field] = manifest[0][source_field]

    report = audit_split_leakage(manifest)

    assert report["status"] == "leakage_detected"
    assert report["total_overlap_count"] == 1
    assert report["fields"][audit_field] == {
        "overlap_count": 1,
        "overlaps": [
            {
                "value": manifest[0][source_field],
                "splits": ["calibration", "train"],
            }
        ],
    }
    with pytest.raises(SplitLeakageError, match=audit_field):
        assert_no_split_leakage(manifest)


def test_audit_rejects_unknown_split_name():
    from src.datasets.split_manager import SplitIndexError, audit_split_leakage

    manifest = _clean_provenance_manifest()
    manifest[0]["split"] = "validation"

    with pytest.raises(SplitIndexError, match="one of train, calibration, val, test"):
        audit_split_leakage(manifest)
