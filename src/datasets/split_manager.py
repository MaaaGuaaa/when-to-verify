"""Deterministic connected-group splits for source indexes.

Splitting happens before snippets, base states, or generated samples exist.
Rows connected by a recording, session, or participant identifier are treated
as one indivisible component.  A scene identifier is used only when none of
those stronger identifiers is available.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src.contracts import SCHEMA_VERSION
from src.utils.seeding import derive_seed, stable_digest

SPLIT_NAMES: tuple[str, ...] = ("train", "calibration", "val", "test")
DEFAULT_SPLIT_RATIOS: dict[str, float] = {
    "train": 0.70,
    "calibration": 0.10,
    "val": 0.10,
    "test": 0.10,
}

_AUDIT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "recording": ("recording_id", "source_recording_id", "source_recording_ids"),
    "session": ("session_id", "source_session_id", "source_session_ids"),
    "participant": (
        "participant_id",
        "participant_ids",
        "source_participant_id",
        "source_participant_ids",
    ),
    "snippet": (
        "snippet_id",
        "ped_snippet_id",
        "source_snippet_id",
        "source_snippet_ids",
    ),
    "pair_group": ("pair_group_id",),
    "seed_namespace": (
        "seed_namespace",
        "generator_seed_namespace",
        "generator_seed",
    ),
}

_RESERVED_OUTPUT_FIELDS = {
    "split",
    "group_id",
    "seed_namespace",
    "generator_seed",
    "participant_check",
}


class SplitIndexError(ValueError):
    """Raised when a source index cannot satisfy the split data contract."""


class SplitLeakageError(ValueError):
    """Raised when audited provenance values occur in multiple splits."""


@dataclass(frozen=True)
class SplitResult:
    """In-memory split artifacts; rows are stored in canonical order."""

    manifest: tuple[dict[str, object], ...]
    summary: dict[str, object]
    overlap_report: dict[str, object]
    manifest_digest: str


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _audit_values(row: Mapping[str, object], aliases: tuple[str, ...]) -> set[str]:
    containers: list[Mapping[str, object]] = [row]
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        containers.append(metadata)
    values: set[str] = set()
    for container in containers:
        for alias in aliases:
            if alias not in container:
                continue
            raw = container[alias]
            if isinstance(raw, (list, tuple)):
                values.update(str(value) for value in raw if str(value))
            elif raw is not None and str(raw):
                values.add(str(raw))
    return values


def audit_split_leakage(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return deterministic cross-split overlaps for source provenance fields."""
    values_by_field: dict[str, dict[str, set[str]]] = {
        field: {} for field in _AUDIT_FIELD_ALIASES
    }
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise SplitIndexError(f"row {index} must be a mapping")
        split = row.get("split")
        if not isinstance(split, str) or not split:
            raise SplitIndexError(f"row {index} split must be a non-empty string")
        if split not in SPLIT_NAMES:
            raise SplitIndexError(
                f"row {index} split must be one of " + ", ".join(SPLIT_NAMES)
            )
        for field, aliases in _AUDIT_FIELD_ALIASES.items():
            for value in _audit_values(row, aliases):
                values_by_field[field].setdefault(value, set()).add(split)

    field_reports: dict[str, dict[str, object]] = {}
    total_overlap_count = 0
    for field in _AUDIT_FIELD_ALIASES:
        overlaps = [
            {"value": value, "splits": sorted(splits)}
            for value, splits in sorted(values_by_field[field].items())
            if len(splits) > 1
        ]
        total_overlap_count += len(overlaps)
        field_reports[field] = {
            "overlap_count": len(overlaps),
            "overlaps": overlaps,
        }
    return {
        "status": "ok" if total_overlap_count == 0 else "leakage_detected",
        "total_overlap_count": total_overlap_count,
        "fields": field_reports,
    }


def assert_no_split_leakage(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return the clean report or raise with all leaking provenance fields."""
    report = audit_split_leakage(records)
    if report["total_overlap_count"]:
        fields = report["fields"]
        leaking_fields = [
            field
            for field in _AUDIT_FIELD_ALIASES
            if fields[field]["overlap_count"]
        ]
        raise SplitLeakageError(
            "cross-split leakage detected for: " + ", ".join(leaking_fields)
        )
    return report


def _canonical_row(row: Mapping[str, object]) -> str:
    return json.dumps(
        row,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def serialize_manifest(manifest: Sequence[Mapping[str, object]]) -> bytes:
    """Serialize manifest rows as canonical, newline-terminated JSONL bytes."""
    lines = sorted(_canonical_row(row) for row in manifest)
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def _serialize_json(payload: Mapping[str, object]) -> bytes:
    text = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def _write_identical_or_new(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite different artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_split_artifacts(
    result: SplitResult, output_dir: str | Path
) -> dict[str, Path]:
    """Atomically write deterministic manifest, summary, and overlap report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output_path / "split_manifest.jsonl",
        "summary": output_path / "split_summary.json",
        "overlap_report": output_path / "overlap_report.json",
    }
    _write_identical_or_new(paths["manifest"], serialize_manifest(result.manifest))
    _write_identical_or_new(paths["summary"], _serialize_json(result.summary))
    _write_identical_or_new(
        paths["overlap_report"], _serialize_json(result.overlap_report)
    )
    return paths


def _participant_ids(row: Mapping[str, object]) -> tuple[str, ...]:
    values: list[str] = []
    participant_id = row.get("participant_id")
    if isinstance(participant_id, str) and participant_id:
        values.append(participant_id)
    participant_ids = row.get("participant_ids", ())
    if isinstance(participant_ids, (list, tuple)):
        values.extend(
            value
            for value in participant_ids
            if isinstance(value, str) and value
        )
    return tuple(sorted(set(values)))


def _validate_json_value(value: object, path: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SplitIndexError(f"{path} must not contain NaN/Inf")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise SplitIndexError(f"{path} keys must be strings")
            child_path = f"{path}.{key}" if path else key
            _validate_json_value(child, child_path)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
        return
    raise SplitIndexError(f"{path} has unsupported type {type(value).__name__}")


def _validate_source_index(
    records: Sequence[Mapping[str, object]], seed: int
) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitIndexError("seed must be an integer")
    if not records:
        raise SplitIndexError("source index must not be empty")
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise SplitIndexError(f"row {index} must be a mapping")
        reserved = _RESERVED_OUTPUT_FIELDS.intersection(row)
        if reserved:
            raise SplitIndexError(
                "source index contains reserved output field(s): "
                + ", ".join(sorted(reserved))
            )
        if "schema_version" in row and row["schema_version"] != SCHEMA_VERSION:
            raise SplitIndexError(
                f"schema_version must be {SCHEMA_VERSION}, got {row['schema_version']!r}"
            )
        for field in ("recording_id", "session_id", "participant_id", "scene_id"):
            if field in row and (
                not isinstance(row[field], str) or not row[field]
            ):
                raise SplitIndexError(f"{field} must be a non-empty string")
        if "participant_ids" in row:
            participant_ids = row["participant_ids"]
            if not isinstance(participant_ids, (list, tuple)):
                raise SplitIndexError("participant_ids must be a list or tuple")
            if any(not isinstance(value, str) or not value for value in participant_ids):
                raise SplitIndexError(
                    "participant_ids entries must be non-empty strings"
                )
        for key, value in row.items():
            if not isinstance(key, str):
                raise SplitIndexError(f"row {index} keys must be strings")
            _validate_json_value(value, key)


def _group_tokens(row: Mapping[str, object]) -> tuple[str, ...]:
    tokens: list[str] = []
    for field in ("recording_id", "session_id"):
        value = row.get(field)
        if isinstance(value, str) and value:
            tokens.append(f"{field}:{value}")
    tokens.extend(f"participant_id:{value}" for value in _participant_ids(row))
    if not tokens:
        scene_id = row.get("scene_id")
        if isinstance(scene_id, str) and scene_id:
            tokens.append(f"scene_id:{scene_id}")
    return tuple(tokens)


def _validate_ratios(ratios: Mapping[str, float] | None) -> dict[str, float]:
    if ratios is None:
        return dict(DEFAULT_SPLIT_RATIOS)
    if set(ratios) != set(SPLIT_NAMES):
        raise SplitIndexError(
            "split ratios must contain exactly: " + ", ".join(SPLIT_NAMES)
        )
    normalized: dict[str, float] = {}
    for split in SPLIT_NAMES:
        value = ratios[split]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SplitIndexError(f"ratio for {split} must be numeric")
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise SplitIndexError(f"ratio for {split} must be finite and non-negative")
        normalized[split] = value
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SplitIndexError("split ratios must sum to 1.0")
    return normalized


def _group_quotas(group_count: int, ratios: Mapping[str, float]) -> dict[str, int]:
    raw = {split: group_count * ratios[split] for split in SPLIT_NAMES}
    quotas = {split: int(raw[split]) for split in SPLIT_NAMES}
    remaining = group_count - sum(quotas.values())
    remainder_order = sorted(
        SPLIT_NAMES,
        key=lambda split: (-(raw[split] - quotas[split]), SPLIT_NAMES.index(split)),
    )
    for split in remainder_order[:remaining]:
        quotas[split] += 1
    return quotas


def make_split_manifest(
    records: Sequence[Mapping[str, object]],
    seed: int = 42,
    ratios: Mapping[str, float] | None = None,
) -> SplitResult:
    """Assign source-index rows to deterministic, indivisible components."""
    _validate_source_index(records, seed)
    requested_ratios = _validate_ratios(ratios)
    ordered = [dict(row) for row in sorted(records, key=_canonical_row)]
    union_find = _UnionFind(len(ordered))
    first_by_token: dict[str, int] = {}
    for index, row in enumerate(ordered):
        tokens = _group_tokens(row)
        if not tokens:
            raise SplitIndexError(
                "each index row needs recording_id, session_id, participant_id(s), "
                "or scene_id"
            )
        for token in tokens:
            if token in first_by_token:
                union_find.union(index, first_by_token[token])
            else:
                first_by_token[token] = index

    members_by_root: dict[int, list[int]] = {}
    for index in range(len(ordered)):
        members_by_root.setdefault(union_find.find(index), []).append(index)

    component_rows: dict[str, list[dict[str, object]]] = {}
    for member_indices in members_by_root.values():
        component_tokens = sorted(
            {token for index in member_indices for token in _group_tokens(ordered[index])}
        )
        group_id = f"group-{stable_digest(*component_tokens, size=12)}"
        component_rows[group_id] = [ordered[index] for index in member_indices]

    ordered_group_ids = sorted(
        component_rows,
        key=lambda group_id: (
            derive_seed(seed, "split-order", group_id),
            group_id,
        ),
    )
    quotas = _group_quotas(len(ordered_group_ids), requested_ratios)
    split_by_group: dict[str, str] = {}
    cursor = 0
    for split in SPLIT_NAMES:
        next_cursor = cursor + quotas[split]
        for group_id in ordered_group_ids[cursor:next_cursor]:
            split_by_group[group_id] = split
        cursor = next_cursor

    manifest: list[dict[str, object]] = []
    for group_id, rows in component_rows.items():
        split = split_by_group[group_id]
        seed_namespace = f"split/{split}/generator"
        generator_seed = derive_seed(seed, "split", split, "generator")
        participant_check = (
            "available" if all(_participant_ids(row) for row in rows) else "unavailable"
        )
        for row in rows:
            manifest.append(
                {
                    **row,
                    "group_id": group_id,
                    "split": split,
                    "seed_namespace": seed_namespace,
                    "generator_seed": generator_seed,
                    "participant_check": participant_check,
                    "schema_version": SCHEMA_VERSION,
                }
            )

    manifest.sort(key=_canonical_row)
    frozen_manifest = tuple(manifest)
    manifest_payload = serialize_manifest(frozen_manifest)
    manifest_digest = hashlib.blake2b(manifest_payload, digest_size=16).hexdigest()

    groups_by_split = {
        split: {
            row["group_id"] for row in frozen_manifest if row["split"] == split
        }
        for split in SPLIT_NAMES
    }
    records_by_split = {
        split: sum(row["split"] == split for row in frozen_manifest)
        for split in SPLIT_NAMES
    }
    group_count = len(component_rows)
    record_count = len(frozen_manifest)
    split_statistics = {
        split: {
            "requested_ratio": requested_ratios[split],
            "group_count": len(groups_by_split[split]),
            "record_count": records_by_split[split],
            "actual_group_ratio": len(groups_by_split[split]) / group_count,
            "actual_record_ratio": records_by_split[split] / record_count,
        }
        for split in SPLIT_NAMES
    }
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "requested_ratios": requested_ratios,
        "source_record_count": record_count,
        "connected_group_count": group_count,
        "participant_check": (
            "available"
            if all(row["participant_check"] == "available" for row in frozen_manifest)
            else "unavailable"
        ),
        "manifest_digest": manifest_digest,
        "split_statistics": split_statistics,
    }
    overlap_report = audit_split_leakage(frozen_manifest)
    overlap_report["schema_version"] = SCHEMA_VERSION
    overlap_report["manifest_digest"] = manifest_digest
    return SplitResult(
        manifest=frozen_manifest,
        summary=summary,
        overlap_report=overlap_report,
        manifest_digest=manifest_digest,
    )
