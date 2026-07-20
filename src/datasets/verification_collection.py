"""Strict collection handoff validation and deterministic provenance digests."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from src.contracts import SCHEMA_VERSION
from src.datasets.verification_dataloader import LoadedVerificationShard
from src.planning.verification_actions import CANONICAL_ACTION_IDS


_HANDOFF_KEYS = frozenset(
    {
        "schema_version",
        "handoff_version",
        "collection_state",
        "scientific_status",
        "split",
        "sample_count",
        "group_count",
        "collection_semantic_digest",
        "generation_report_sha256",
        "shards",
        "limitations",
    }
)
_ROW_KEYS = frozenset(
    {"shard_index", "relative_root", "sample_count", "semantic_digest"}
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_STATUS_BY_SPLIT = {
    "train": frozenset({"toy_smoke_only", "train_smoke_only", "publication_ready"}),
    "calibration": frozenset({"calibration_smoke_only", "publication_ready"}),
    "val": frozenset({"val_smoke_only", "publication_ready"}),
    "test": frozenset({"test_smoke_only", "publication_ready"}),
}


def _strict_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("collection handoff must be a real file")

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid collection handoff: {exc}") from exc
    if not isinstance(value, dict) or set(value) != _HANDOFF_KEYS:
        raise ValueError("collection handoff keys are invalid")
    return value


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _relative_shard_root(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("collection handoff relative_root must be a string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or re.fullmatch(r"shard-[0-9]{5}", value) is None
    ):
        raise ValueError("collection handoff relative_root is unsafe")
    return value


def validate_verification_collection_handoff(
    path: str | Path,
    *,
    shard_dirs: Sequence[str | Path],
    loaded_shards: Sequence[LoadedVerificationShard],
    expected_split: str,
) -> dict[str, object]:
    """Bind one immutable handoff to already validated shard payloads."""

    root = Path(path)
    handoff = _strict_json(root)
    if expected_split not in _STATUS_BY_SPLIT:
        raise ValueError("expected_split is invalid")
    if handoff["schema_version"] != SCHEMA_VERSION:
        raise ValueError("collection handoff schema mismatch")
    if handoff["handoff_version"] != "verification_collection_handoff_v1":
        raise ValueError("unsupported verification collection handoff")
    if handoff["collection_state"] != "complete":
        raise ValueError("verification collection is incomplete")
    if handoff["split"] != expected_split:
        raise ValueError("verification collection split mismatch")
    if handoff["scientific_status"] not in _STATUS_BY_SPLIT[expected_split]:
        raise ValueError("collection scientific_status is invalid for split")
    _digest(handoff["collection_semantic_digest"], name="collection digest")
    _digest(handoff["generation_report_sha256"], name="generation report digest")

    roots = tuple(Path(value) for value in shard_dirs)
    loaded = tuple(loaded_shards)
    if not roots or len(roots) != len(loaded):
        raise ValueError("collection handoff loaded shard count mismatch")
    rows = handoff["shards"]
    if not isinstance(rows, list) or len(rows) != len(roots):
        raise ValueError("collection handoff shard count mismatch")
    expected_rows: dict[Path, Mapping[str, object]] = {}
    for position, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != _ROW_KEYS:
            raise ValueError("collection handoff shard row keys are invalid")
        if raw["shard_index"] != position:
            raise ValueError("collection handoff shard indices must be ordered")
        relative = _relative_shard_root(raw["relative_root"])
        resolved = (root.parent / relative).resolve()
        if resolved in expected_rows:
            raise ValueError("collection handoff contains duplicate shard roots")
        expected_rows[resolved] = raw
    observed_roots = {value.resolve() for value in roots}
    if set(expected_rows) != observed_roots:
        raise ValueError("collection handoff shard roots mismatch")
    for shard_root, shard in zip(roots, loaded, strict=True):
        row = expected_rows[shard_root.resolve()]
        if row["sample_count"] != len(shard.samples):
            raise ValueError("collection handoff shard sample count mismatch")
        if row["semantic_digest"] != shard.semantic_digest:
            raise ValueError("collection handoff shard semantic digest mismatch")
        if shard.summary.get("split") != expected_split:
            raise ValueError("loaded shard split differs from collection handoff")

    sample_count = sum(len(value.samples) for value in loaded)
    if handoff["sample_count"] != sample_count:
        raise ValueError("collection handoff total sample count mismatch")
    group_size = len(CANONICAL_ACTION_IDS)
    if (
        isinstance(handoff["group_count"], bool)
        or not isinstance(handoff["group_count"], int)
        or handoff["group_count"] <= 0
        or handoff["group_count"] * group_size != sample_count
    ):
        raise ValueError("collection handoff group count mismatch")
    if not isinstance(handoff["limitations"], list) or any(
        not isinstance(value, str) or not value for value in handoff["limitations"]
    ):
        raise ValueError("collection handoff limitations are invalid")
    return dict(handoff)


def _domain_digest(domain: bytes, values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    for value in sorted(values):
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def verification_input_digests(
    loaded_shards: Sequence[LoadedVerificationShard],
) -> tuple[str, dict[str, str]]:
    """Aggregate shard manifests and semantic digests without path dependence."""

    loaded = tuple(loaded_shards)
    if not loaded:
        raise ValueError("loaded_shards must be non-empty")
    manifests = [value.manifest_digest for value in loaded]
    input_manifest = (
        manifests[0]
        if len(manifests) == 1
        else _domain_digest(b"verification-input-manifests-v1\0", manifests)
    )
    by_split: dict[str, list[str]] = {}
    for shard in loaded:
        split = str(shard.summary["split"])
        if split not in _STATUS_BY_SPLIT:
            raise ValueError("loaded shard contains an invalid split")
        by_split.setdefault(split, []).append(shard.semantic_digest)
    split_digests = {
        split: (
            values[0]
            if len(values) == 1
            else _domain_digest(
                b"verification-split-shards-v1\0" + split.encode("ascii") + b"\0",
                values,
            )
        )
        for split, values in sorted(by_split.items())
    }
    return input_manifest, split_digests


__all__ = (
    "validate_verification_collection_handoff",
    "verification_input_digests",
)
