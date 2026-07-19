"""Stable external identity for one complete SOP-05 publication."""

from __future__ import annotations

import hashlib
import json
import re


SOP05_PUBLICATION_IDENTITY_VERSION = (
    "sop05_publication_semantic_digest_v2"
)
_DOMAIN_PREFIX = SOP05_PUBLICATION_IDENTITY_VERSION.encode("ascii") + b"\0"
_RUN_ID_PATTERN = re.compile(r"sop05-run-[0-9a-f]{32}\Z")
_LOWER_HEX_PATTERNS = {
    32: re.compile(r"[0-9a-f]{32}\Z"),
    64: re.compile(r"[0-9a-f]{64}\Z"),
}


def _validate_run_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("run_id must be a string")
    if _RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("run_id must match sop05-run- plus 32 lowercase hex")
    return value


def _validate_lower_hex(name: str, value: object, length: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if _LOWER_HEX_PATTERNS[length].fullmatch(value) is None:
        raise ValueError(f"{name} must be {length} lowercase hex characters")
    return value


def compute_sop05_publication_semantic_digest(
    *,
    run_id: object,
    run_manifest_sha256: object,
    checksums_sha256: object,
    target_motion_manifest_digest: object,
    target_motion_payload_semantic_digest: object,
) -> str:
    """Return the domain-separated digest handed off outside the publication.

    The digest deliberately binds both ordinary file checksums and the target
    motion shard's semantic identity.  A consumer must obtain the returned
    value from a trusted producer handoff rather than from the directory it is
    validating.
    """

    payload = {
        "run_id": _validate_run_id(run_id),
        "run_manifest_sha256": _validate_lower_hex(
            "run_manifest_sha256", run_manifest_sha256, 64
        ),
        "checksums_sha256": _validate_lower_hex(
            "checksums_sha256", checksums_sha256, 64
        ),
        "target_motion_manifest_digest": _validate_lower_hex(
            "target_motion_manifest_digest",
            target_motion_manifest_digest,
            32,
        ),
        "target_motion_payload_semantic_digest": _validate_lower_hex(
            "target_motion_payload_semantic_digest",
            target_motion_payload_semantic_digest,
            32,
        ),
    }
    canonical_payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.blake2b(
        _DOMAIN_PREFIX + canonical_payload, digest_size=32
    ).hexdigest()
