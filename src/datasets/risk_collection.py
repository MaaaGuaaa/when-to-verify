"""Immutable global leakage evidence for schema-v3 SOP07 risk shards."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from src.contracts import GridSpec, SCHEMA_VERSION
from src.datasets.shard_writer import load_risk_shard
from src.datasets.split_manager import SPLIT_NAMES, serialize_manifest


RISK_COLLECTION_LAYOUT_VERSION = "risk_collection_v1"

_MEMBERS_NAME = "members.jsonl"
_LEAKAGE_REPORT_NAME = "leakage_report.json"
_SUMMARY_NAME = "summary.json"
_REQUIRED_FILES = frozenset(
    {_MEMBERS_NAME, _LEAKAGE_REPORT_NAME, _SUMMARY_NAME}
)
_SPLIT_FILES = frozenset(
    {"split_manifest.jsonl", "split_summary.json", "overlap_report.json"}
)
_MEMBER_KEYS = frozenset(
    {
        "schema_version",
        "layout_version",
        "relative_path",
        "split",
        "shard_index",
        "sample_count",
        "manifest_digest",
        "semantic_digest",
        "audit_context_digest",
    }
)
_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "layout_version",
        "required_splits",
        "split_manifest_digest",
        "member_count",
        "sample_count",
        "split_counts",
        "files",
        "members_digest",
        "leakage_report_digest",
        "collection_semantic_digest",
    }
)
_STRICT_IDENTITY_FIELDS = (
    "recording",
    "source_snippet",
    "pair_group",
    "sample",
    "seed_namespace",
)


class RiskCollectionError(ValueError):
    """Raised when collection membership or leakage evidence is unsafe."""


@dataclass(frozen=True)
class RiskCollectionMemberRequest:
    """Trusted scheduler declaration for one immutable SOP07 shard."""

    relative_path: str
    split: str
    shard_index: int
    expected_sample_count: int
    expected_manifest_digest: str
    expected_semantic_digest: str


@dataclass(frozen=True)
class LoadedRiskCollection:
    """A formally reloaded four-split risk collection."""

    members: tuple[dict[str, object], ...]
    leakage_report: dict[str, object]
    summary: dict[str, object]
    collection_semantic_digest: str


@dataclass(frozen=True)
class _SplitAuthority:
    manifest_digest: str
    recording_split: dict[str, str]
    recording_session: dict[str, str]


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_loads(payload: str) -> object:
    return json.loads(payload, parse_constant=_reject_json_constant)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _serialize_json(value: Mapping[str, object]) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _serialize_jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        return b""
    return ("\n".join(_canonical_json(row) for row in rows) + "\n").encode(
        "utf-8"
    )


def _digest(domain: bytes, payload: bytes) -> str:
    value = hashlib.sha256()
    value.update(domain)
    value.update(len(payload).to_bytes(8, "big"))
    value.update(payload)
    return value.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RiskCollectionError(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise RiskCollectionError(f"{label} must be an object")
    return value


def _load_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise RiskCollectionError(f"invalid {label}") from exc
    if not raw or not raw.endswith(b"\n"):
        raise RiskCollectionError(f"{label} must be non-empty newline-terminated JSONL")
    rows: list[dict[str, object]] = []
    for index, line in enumerate(text.splitlines()):
        try:
            value = _strict_json_loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RiskCollectionError(f"invalid {label} row {index}") from exc
        if not isinstance(value, dict):
            raise RiskCollectionError(f"{label} row {index} must be an object")
        rows.append(value)
    return tuple(rows)


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RiskCollectionError(f"{label} must be a non-empty string")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RiskCollectionError(f"{label} must be a non-negative integer")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    result = _require_nonnegative_int(value, label=label)
    if result == 0:
        raise RiskCollectionError(f"{label} must be positive")
    return result


def _require_sha256(value: object, *, label: str) -> str:
    result = _require_string(value, label=label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise RiskCollectionError(f"{label} must be a lowercase SHA-256")
    return result


def _require_blake2b128(value: object, *, label: str) -> str:
    result = _require_string(value, label=label)
    if len(result) != 32 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise RiskCollectionError(f"{label} must be a lowercase BLAKE2b-128")
    return result


def _load_split_authority(
    directory: str | Path, *, expected_manifest_digest: str
) -> _SplitAuthority:
    trusted_digest = _require_blake2b128(
        expected_manifest_digest, label="expected split manifest digest"
    )
    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise RiskCollectionError("split artifact directory must be a real directory")
    actual = {path.name for path in root.iterdir()}
    if actual != _SPLIT_FILES:
        raise RiskCollectionError("split artifact file layout mismatch")
    manifest_path = root / "split_manifest.jsonl"
    rows = _load_jsonl(manifest_path, label="split manifest")
    raw = manifest_path.read_bytes()
    if serialize_manifest(rows) != raw:
        raise RiskCollectionError("split manifest is not canonical")
    digest = hashlib.blake2b(raw, digest_size=16).hexdigest()
    if digest != trusted_digest:
        raise RiskCollectionError("expected split manifest digest mismatch")
    summary = _load_json(root / "split_summary.json", label="split summary")
    overlap = _load_json(root / "overlap_report.json", label="split overlap report")
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise RiskCollectionError("split summary schema_version mismatch")
    if summary.get("manifest_digest") != digest:
        raise RiskCollectionError("split summary manifest digest mismatch")
    if overlap.get("manifest_digest") != digest:
        raise RiskCollectionError("split overlap manifest digest mismatch")
    if overlap.get("status") != "ok" or overlap.get("disallowed_overlap_count") != 0:
        raise RiskCollectionError("authoritative split leakage audit is not clean")
    if summary.get("evaluation_scope") != "unseen_recording_within_known_sessions":
        raise RiskCollectionError("split evaluation scope mismatch")
    if summary.get("grouping_unit") != "recording_id":
        raise RiskCollectionError("split grouping unit mismatch")

    mapping: dict[str, str] = {}
    sessions: dict[str, str] = {}
    for index, row in enumerate(rows):
        recording_id = _require_string(
            row.get("recording_id"), label=f"split manifest row {index} recording_id"
        )
        split = _require_string(
            row.get("split"), label=f"split manifest row {index} split"
        )
        session = _require_string(
            row.get("session_id"), label=f"split manifest row {index} session_id"
        )
        if split not in SPLIT_NAMES:
            raise RiskCollectionError("split manifest contains an unknown split")
        if recording_id in mapping:
            raise RiskCollectionError("split manifest contains duplicate recording_id")
        mapping[recording_id] = split
        sessions[recording_id] = session
    return _SplitAuthority(
        manifest_digest=digest,
        recording_split=mapping,
        recording_session=sessions,
    )


def _safe_member_path(root: Path, relative_path: object) -> tuple[str, Path]:
    text = _require_string(relative_path, label="member relative_path")
    relative = Path(text)
    if relative.is_absolute() or text != relative.as_posix() or ".." in relative.parts:
        raise RiskCollectionError("member relative_path must be a safe POSIX relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise RiskCollectionError("member relative_path escapes shard_root")
    return text, resolved


def _field_report(
    values_by_split: Mapping[str, Mapping[str, None]], *, policy: str
) -> dict[str, object]:
    splits_by_value: dict[str, set[str]] = {}
    for split, values in values_by_split.items():
        for value in values:
            splits_by_value.setdefault(value, set()).add(split)
    overlaps = [
        {"value": value, "splits": sorted(splits)}
        for value, splits in sorted(splits_by_value.items())
        if len(splits) > 1
    ]
    return {
        "policy": policy,
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "unique_value_count": len(splits_by_value),
        "per_split_unique_counts": {
            split: len(values_by_split[split]) for split in SPLIT_NAMES
        },
    }


def _build_collection(
    requests: Sequence[RiskCollectionMemberRequest],
    *,
    shard_root: str | Path,
    split_artifact_dir: str | Path,
    expected_split_manifest_digest: str,
    grid: GridSpec,
) -> tuple[
    tuple[dict[str, object], ...],
    dict[str, object],
    dict[str, object],
]:
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
        raise TypeError("members must be a sequence")
    if any(not isinstance(item, RiskCollectionMemberRequest) for item in requests):
        raise TypeError("members must contain RiskCollectionMemberRequest values")
    authority = _load_split_authority(
        split_artifact_dir,
        expected_manifest_digest=expected_split_manifest_digest,
    )
    root = Path(shard_root)
    if not root.is_dir() or root.is_symlink():
        raise RiskCollectionError("shard_root must be a real directory")

    identity_values: dict[str, dict[str, dict[str, None]]] = {
        field: {split: {} for split in SPLIT_NAMES}
        for field in (*_STRICT_IDENTITY_FIELDS, "session")
    }
    member_rows: list[dict[str, object]] = []
    seen_boundaries: set[tuple[str, int]] = set()
    seen_sample_ids: set[str] = set()
    for request_index, request in enumerate(requests):
        split = _require_string(request.split, label="member split")
        if split not in SPLIT_NAMES:
            raise RiskCollectionError("member split is outside the frozen split set")
        shard_index = _require_nonnegative_int(
            request.shard_index, label="member shard_index"
        )
        boundary = (split, shard_index)
        if boundary in seen_boundaries:
            raise RiskCollectionError("duplicate split/shard_index member boundary")
        seen_boundaries.add(boundary)
        relative_path, shard_path = _safe_member_path(root, request.relative_path)
        expected_count = _require_positive_int(
            request.expected_sample_count, label="member expected_sample_count"
        )
        expected_manifest = _require_sha256(
            request.expected_manifest_digest, label="member expected_manifest_digest"
        )
        expected_semantic = _require_sha256(
            request.expected_semantic_digest, label="member expected_semantic_digest"
        )
        try:
            loaded = load_risk_shard(shard_path, grid=grid)
        except (TypeError, ValueError) as exc:
            raise RiskCollectionError(
                f"formal risk shard load failed for member {request_index}"
            ) from exc
        summary = loaded.summary
        if summary.get("split") != split or summary.get("shard_index") != shard_index:
            raise RiskCollectionError("member split/shard_index declaration mismatch")
        if len(loaded.manifest) != expected_count:
            raise RiskCollectionError("member expected sample count mismatch")
        if loaded.manifest_digest != expected_manifest:
            raise RiskCollectionError("member expected manifest digest mismatch")
        if loaded.semantic_digest != expected_semantic:
            raise RiskCollectionError("member expected semantic digest mismatch")

        for row_index, row in enumerate(loaded.manifest):
            if row.get("split") != split:
                raise RiskCollectionError("verified shard contains a mixed split row")
            base_recording = _require_string(
                row.get("base_recording_id"),
                label=f"member {request_index} row {row_index} base_recording_id",
            )
            source_recording = _require_string(
                row.get("source_recording_id"),
                label=f"member {request_index} row {row_index} source_recording_id",
            )
            base_session = _require_string(
                row.get("base_session_id"),
                label=f"member {request_index} row {row_index} base_session_id",
            )
            source_session = _require_string(
                row.get("source_session_id"),
                label=f"member {request_index} row {row_index} source_session_id",
            )
            for recording, session in (
                (base_recording, base_session),
                (source_recording, source_session),
            ):
                assigned = authority.recording_split.get(recording)
                if assigned is None:
                    raise RiskCollectionError(
                        f"recording {recording!r} is absent from authoritative split manifest"
                    )
                if assigned != split:
                    raise RiskCollectionError(
                        f"recording {recording!r} authoritative split mismatch"
                    )
                if authority.recording_session[recording] != session:
                    raise RiskCollectionError(
                        f"recording {recording!r} authoritative session mismatch"
                    )
                identity_values["recording"][split][recording] = None
                identity_values["session"][split][session] = None
            for field, row_key in (
                ("source_snippet", "source_snippet_id"),
                ("pair_group", "pair_group_id"),
                ("sample", "sample_id"),
                ("seed_namespace", "seed_namespace"),
            ):
                value = _require_string(
                    row.get(row_key),
                    label=f"member {request_index} row {row_index} {row_key}",
                )
                if field == "sample":
                    if value in seen_sample_ids:
                        raise RiskCollectionError(
                            f"duplicate sample_id across collection members: {value!r}"
                        )
                    seen_sample_ids.add(value)
                identity_values[field][split][value] = None
        member_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "layout_version": RISK_COLLECTION_LAYOUT_VERSION,
                "relative_path": relative_path,
                "split": split,
                "shard_index": shard_index,
                "sample_count": len(loaded.manifest),
                "manifest_digest": loaded.manifest_digest,
                "semantic_digest": loaded.semantic_digest,
                "audit_context_digest": _require_sha256(
                    summary.get("audit_context_digest"),
                    label="shard audit_context_digest",
                ),
            }
        )

    covered = {row["split"] for row in member_rows}
    if covered != set(SPLIT_NAMES):
        missing = sorted(set(SPLIT_NAMES) - covered)
        raise RiskCollectionError(
            "global collection requires all four splits; missing=" + ",".join(missing)
        )
    split_order = {split: index for index, split in enumerate(SPLIT_NAMES)}
    member_rows.sort(
        key=lambda row: (
            split_order[str(row["split"])],
            int(row["shard_index"]),
            str(row["relative_path"]),
        )
    )
    fields = {
        field: _field_report(
            identity_values[field],
            policy=("allowed_reported" if field == "session" else "forbidden"),
        )
        for field in (*_STRICT_IDENTITY_FIELDS, "session")
    }
    leaking = [
        field
        for field in _STRICT_IDENTITY_FIELDS
        if fields[field]["overlap_count"]
    ]
    if leaking:
        raise RiskCollectionError(
            "cross-split leakage detected for: " + ", ".join(leaking)
        )
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "layout_version": RISK_COLLECTION_LAYOUT_VERSION,
        "status": "ok",
        "evaluation_scope": "global_four_split_unseen_recording_v1",
        "disallowed_overlap_count": 0,
        "allowed_overlap_count": fields["session"]["overlap_count"],
        "fields": fields,
    }
    members_payload = _serialize_jsonl(member_rows)
    report_payload = _serialize_json(report)
    members_digest = _digest(b"risk-collection-members-v1\0", members_payload)
    report_digest = _digest(b"risk-collection-leakage-v1\0", report_payload)
    split_counts = {
        split: {
            "member_count": sum(row["split"] == split for row in member_rows),
            "sample_count": sum(
                int(row["sample_count"])
                for row in member_rows
                if row["split"] == split
            ),
        }
        for split in SPLIT_NAMES
    }
    semantic_payload = _canonical_json(
        {
            "layout_version": RISK_COLLECTION_LAYOUT_VERSION,
            "required_splits": list(SPLIT_NAMES),
            "split_manifest_digest": authority.manifest_digest,
            "members_digest": members_digest,
            "leakage_report_digest": report_digest,
        }
    ).encode("utf-8")
    collection_digest = _digest(
        b"risk-collection-semantic-v1\0", semantic_payload
    )
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "layout_version": RISK_COLLECTION_LAYOUT_VERSION,
        "required_splits": list(SPLIT_NAMES),
        "split_manifest_digest": authority.manifest_digest,
        "member_count": len(member_rows),
        "sample_count": sum(int(row["sample_count"]) for row in member_rows),
        "split_counts": split_counts,
        "files": {
            "members": _MEMBERS_NAME,
            "leakage_report": _LEAKAGE_REPORT_NAME,
            "summary": _SUMMARY_NAME,
        },
        "members_digest": members_digest,
        "leakage_report_digest": report_digest,
        "collection_semantic_digest": collection_digest,
    }
    return tuple(member_rows), report, summary


def _requests_from_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[RiskCollectionMemberRequest, ...]:
    requests: list[RiskCollectionMemberRequest] = []
    for index, row in enumerate(rows):
        if set(row) != _MEMBER_KEYS:
            raise RiskCollectionError(f"collection member row {index} keys mismatch")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise RiskCollectionError("collection member schema_version mismatch")
        if row.get("layout_version") != RISK_COLLECTION_LAYOUT_VERSION:
            raise RiskCollectionError("collection member layout_version mismatch")
        _require_sha256(
            row.get("audit_context_digest"), label="member audit_context_digest"
        )
        requests.append(
            RiskCollectionMemberRequest(
                relative_path=_require_string(
                    row.get("relative_path"), label="member relative_path"
                ),
                split=_require_string(row.get("split"), label="member split"),
                shard_index=_require_nonnegative_int(
                    row.get("shard_index"), label="member shard_index"
                ),
                expected_sample_count=_require_positive_int(
                    row.get("sample_count"), label="member sample_count"
                ),
                expected_manifest_digest=_require_sha256(
                    row.get("manifest_digest"), label="member manifest_digest"
                ),
                expected_semantic_digest=_require_sha256(
                    row.get("semantic_digest"), label="member semantic_digest"
                ),
            )
        )
    return tuple(requests)


def write_risk_collection(
    members: Sequence[RiskCollectionMemberRequest],
    output_dir: str | Path,
    *,
    shard_root: str | Path,
    split_artifact_dir: str | Path,
    expected_split_manifest_digest: str,
    grid: GridSpec,
) -> dict[str, Path]:
    """Publish one immutable four-split collection after formal staging reload."""

    rows, report, summary = _build_collection(
        members,
        shard_root=shard_root,
        split_artifact_dir=split_artifact_dir,
        expected_split_manifest_digest=expected_split_manifest_digest,
        grid=grid,
    )
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable collection: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        (staging / _MEMBERS_NAME).write_bytes(_serialize_jsonl(rows))
        (staging / _LEAKAGE_REPORT_NAME).write_bytes(_serialize_json(report))
        (staging / _SUMMARY_NAME).write_bytes(_serialize_json(summary))
        loaded = load_risk_collection(
            staging,
            shard_root=shard_root,
            split_artifact_dir=split_artifact_dir,
            expected_split_manifest_digest=expected_split_manifest_digest,
            grid=grid,
        )
        if loaded.collection_semantic_digest != summary["collection_semantic_digest"]:
            raise RiskCollectionError("formal collection reload digest mismatch")
        os.rename(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "members": output / _MEMBERS_NAME,
        "leakage_report": output / _LEAKAGE_REPORT_NAME,
        "summary": output / _SUMMARY_NAME,
    }


def load_risk_collection(
    output_dir: str | Path,
    *,
    shard_root: str | Path,
    split_artifact_dir: str | Path,
    expected_split_manifest_digest: str,
    grid: GridSpec,
) -> LoadedRiskCollection:
    """Reload all collection members and recompute the global leakage proof."""

    output = Path(output_dir)
    if not output.is_dir() or output.is_symlink():
        raise RiskCollectionError("collection directory must be a real directory")
    actual = {path.name for path in output.iterdir()}
    if actual != _REQUIRED_FILES:
        raise RiskCollectionError("collection file layout mismatch")
    raw_members = (output / _MEMBERS_NAME).read_bytes()
    rows = _load_jsonl(output / _MEMBERS_NAME, label="collection members")
    if _serialize_jsonl(rows) != raw_members:
        raise RiskCollectionError("collection members are not canonical")
    stored_report = _load_json(
        output / _LEAKAGE_REPORT_NAME, label="collection leakage report"
    )
    stored_summary = _load_json(output / _SUMMARY_NAME, label="collection summary")
    if set(stored_summary) != _SUMMARY_KEYS:
        raise RiskCollectionError("collection summary keys mismatch")
    if _serialize_json(stored_report) != (output / _LEAKAGE_REPORT_NAME).read_bytes():
        raise RiskCollectionError("collection leakage report is not canonical")
    if _serialize_json(stored_summary) != (output / _SUMMARY_NAME).read_bytes():
        raise RiskCollectionError("collection summary is not canonical")
    requests = _requests_from_rows(rows)
    expected_rows, expected_report, expected_summary = _build_collection(
        requests,
        shard_root=shard_root,
        split_artifact_dir=split_artifact_dir,
        expected_split_manifest_digest=expected_split_manifest_digest,
        grid=grid,
    )
    if tuple(rows) != expected_rows:
        raise RiskCollectionError("collection member evidence mismatch")
    if stored_report != expected_report:
        raise RiskCollectionError("collection leakage report mismatch")
    if stored_summary != expected_summary:
        raise RiskCollectionError("collection summary mismatch")
    return LoadedRiskCollection(
        members=expected_rows,
        leakage_report=expected_report,
        summary=expected_summary,
        collection_semantic_digest=str(
            expected_summary["collection_semantic_digest"]
        ),
    )
