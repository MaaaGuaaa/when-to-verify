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


def _recording_generalization_policy():
    from src.datasets.split_manager import SplitAuditPolicy

    return SplitAuditPolicy(
        evaluation_scope="unseen_recording_within_known_sessions",
        required_fields=("recording", "session", "seed_namespace"),
        allowed_overlap_fields=("session",),
        unavailable_fields=("participant",),
    )


def test_recording_generalization_reports_allowed_session_overlap():
    from src.datasets.split_manager import (
        assert_no_split_leakage,
        audit_split_leakage,
    )

    rows = [
        {
            "split": "train",
            "recording_id": "recording-train",
            "session_id": "120522",
            "seed_namespace": "split/train/generator",
        },
        {
            "split": "test",
            "recording_id": "recording-test",
            "session_id": "120522",
            "seed_namespace": "split/test/generator",
        },
    ]

    report = audit_split_leakage(
        rows, policy=_recording_generalization_policy()
    )

    assert report["status"] == "ok"
    assert report["detected_overlap_count"] == 1
    assert report["total_overlap_count"] == 1
    assert report["allowed_overlap_count"] == 1
    assert report["disallowed_overlap_count"] == 0
    assert report["missing_required_row_count"] == 0
    assert report["field_policies"]["session"] == "allowed_reported"
    assert report["field_policies"]["participant"] == "unavailable"
    assert report["field_coverage"]["session"] == {
        "row_count": 2,
        "rows_with_values": 2,
        "missing_row_count": 0,
        "unique_value_count": 1,
        "status": "complete",
    }
    assert report["fields"]["session"] == {
        "overlap_count": 1,
        "overlaps": [
            {"value": "120522", "splits": ["test", "train"]}
        ],
    }
    assert_no_split_leakage(
        rows, policy=_recording_generalization_policy()
    )


def test_required_session_coverage_cannot_pass_vacuously():
    from src.datasets.split_manager import (
        SplitLeakageError,
        assert_no_split_leakage,
        audit_split_leakage,
    )

    rows = [
        {
            "split": "train",
            "recording_id": "recording-train",
            "session_id": "120522",
            "seed_namespace": "split/train/generator",
        },
        {
            "split": "test",
            "recording_id": "recording-test",
            "seed_namespace": "split/test/generator",
        },
    ]

    report = audit_split_leakage(
        rows, policy=_recording_generalization_policy()
    )

    assert report["status"] == "provenance_incomplete"
    assert report["missing_required_row_count"] == 1
    assert report["field_coverage"]["session"]["missing_row_count"] == 1
    with pytest.raises(
        SplitLeakageError, match="missing required provenance: session"
    ):
        assert_no_split_leakage(
            rows, policy=_recording_generalization_policy()
        )
