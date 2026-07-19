from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.datasets.sop03_publication import (
    Sop03PublicationError,
    _schema_and_digest,
    publish_checksum_envelope,
    validate_commit,
)
from src.generation.sop05_input_adapter import _validate_sop03_checksum_envelope


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checksum_envelope_covers_payload_and_marker_last(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"scientific payload")

    summary = publish_checksum_envelope(root, workers=2)

    marker = root / ".producer-complete"
    assert marker.read_bytes() == b""
    lines = (root / "artifact_checksums.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    assert lines == sorted(lines, key=lambda line: line.split("  ", 1)[1])
    assert f"{_sha256(marker)}  .producer-complete" in lines
    assert f"{_sha256(payload)}  payload.bin" in lines
    assert summary["covered_file_count"] == 2
    verified, manifest_digest = _validate_sop03_checksum_envelope(root, 2)
    assert set(verified) == {".producer-complete", "payload.bin"}
    assert manifest_digest == summary["checksum_manifest_sha256"]


def test_checksum_envelope_refuses_existing_publication(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    marker = root / ".producer-complete"
    marker.write_bytes(b"")

    with pytest.raises(Sop03PublicationError, match="already exists"):
        publish_checksum_envelope(root, workers=1)

    assert marker.read_bytes() == b""
    assert not (root / "artifact_checksums.sha256").exists()


def test_checksum_failure_never_leaves_complete_marker(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "unsafe-link").symlink_to(tmp_path / "missing")

    with pytest.raises(Sop03PublicationError, match="symlink"):
        publish_checksum_envelope(root, workers=1)

    assert not (root / ".producer-complete").exists()


def test_checksum_envelope_does_not_steal_another_finalizer_lock(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    lock = tmp_path / ".artifact.sop03-publication.lock"
    lock.write_text("owner=fixture\n", encoding="utf-8")

    with pytest.raises(Sop03PublicationError, match="another finalizer"):
        publish_checksum_envelope(root, workers=1)

    assert lock.read_text(encoding="utf-8") == "owner=fixture\n"
    assert not (root / ".producer-complete").exists()


@pytest.mark.parametrize("value", ["abc", "G" * 40, "1" * 39, "1" * 41])
def test_validate_commit_rejects_noncanonical_identity(value: str) -> None:
    with pytest.raises(Sop03PublicationError, match="40 lowercase"):
        validate_commit(value, "producer commit")


def test_validate_commit_accepts_canonical_identity() -> None:
    assert validate_commit("a" * 40, "producer commit") == "a" * 40


def test_schema_audit_rejects_conflicting_direct_and_nested_split_digest() -> None:
    with pytest.raises(Sop03PublicationError, match="direct split digest"):
        _schema_and_digest(
            {
                "schema_version": "3.0.0",
                "split_manifest_digest": "wrong",
                "split_provenance": {"split_manifest_digest": "accepted"},
            },
            "accepted",
            "fixture",
        )
