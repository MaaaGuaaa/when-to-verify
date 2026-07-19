"""Strict, manifest-driven adapters for the frozen SOP-03/SOP-04 inputs.

The adapters in this module deliberately treat producer artifacts as untrusted
input.  They validate completion evidence, the complete checksum envelope, and
cross-file contracts before exposing any data to SOP-05.  Base-state/oracle
pairs remain lazy: their serialized contracts are checked when requested.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from src.contracts import (
    DYNAMIC_OBJECT_TYPES,
    SCHEMA_VERSION,
    BaseState,
    ContractError,
    GridSpec,
    LocalTrajectory,
    OracleContext,
    load_dataclass,
    validate_base_state,
    validate_oracle_context,
)
from src.datasets.snippet_library import SnippetLibrary, load_snippet_library
from src.utils.seeding import derive_seed, stable_digest


_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SPLITS = frozenset({"train", "calibration", "val", "test"})
_SNIPPET_STEPS = 23
_HISTORY_STEPS = 8
_CURRENT_INDEX = 7
_FUTURE_STEPS = 15
_SAMPLE_DT_S = 0.2
_SNIPPET_DURATION_S = 4.4
_LAYOUT_VERSION = "history8_current7_future15_v1"
SOP04_TRAJECTORY_BANK_VERSION = "sop04_audited_bank_v2"
SOP04_POSE_TIME_LAYOUT_VERSION = "future_endpoints_dt_to_horizon_v1"
SOP04_COMPLETION_POLICY = SOP04_TRAJECTORY_BANK_VERSION
_SOP04_EXTERNAL_HANDOFF_LABEL = "sop04_audited_bank_v2_envelope"
_SOP04_EXTERNAL_HANDOFF_DOMAIN = b"sop04_audited_bank_v2_external_handoff\0"
_SOP04_CORE_PAYLOADS = frozenset(
    {"trajectory_bank.npz", "trajectory_manifest.jsonl", "summary.json"}
)


class Sop05InputError(ValueError):
    """Raised when an upstream bundle is unsafe or internally inconsistent."""


@dataclass(frozen=True)
class ProducerEvidence:
    """Validated producer evidence retained for downstream provenance."""

    root: Path
    code_commit: str
    checksum_manifest_sha256: str
    audit_sha256: str
    completion_policy: str
    payload_checksums: dict[str, str]


@dataclass(frozen=True)
class Sop03ManifestRecord:
    """Resolved paths and immutable identity fields for one paired state."""

    state_id: str
    recording_id: str
    split: str
    base_state_path: Path
    oracle_context_path: Path
    timestamp: float
    dynamic_object_ids: tuple[str, ...]
    oracle_dynamic_object_ids: tuple[str, ...]


@dataclass(frozen=True)
class Sop03SplitInputs:
    """One validated SOP-03 split with lazy base/oracle payload loading."""

    root: Path
    split: str
    manifest_index: dict[str, Sop03ManifestRecord]
    typed_libraries: dict[str, SnippetLibrary]
    producer_evidence: ProducerEvidence

    def load_pair(
        self, state_id: str, grid: GridSpec
    ) -> tuple[BaseState, OracleContext]:
        """Load and validate exactly one paired BaseState/OracleContext."""
        record = self.manifest_index.get(state_id)
        if record is None:
            raise Sop05InputError(f"unknown SOP-03 state_id: {state_id!r}")
        try:
            state = load_dataclass(record.base_state_path)
        except Exception as exc:
            raise Sop05InputError(
                f"failed to load BaseState for {state_id!r}: {exc}"
            ) from exc
        if not isinstance(state, BaseState):
            raise Sop05InputError(
                f"BaseState payload for {state_id!r} has type {type(state).__name__}"
            )
        try:
            validate_base_state(state, grid)
        except (ContractError, KeyError, TypeError, ValueError) as exc:
            raise Sop05InputError(
                f"BaseState contract failed for {state_id!r}: {exc}"
            ) from exc

        try:
            context = load_dataclass(record.oracle_context_path)
        except Exception as exc:
            raise Sop05InputError(
                f"failed to load OracleContext for {state_id!r}: {exc}"
            ) from exc
        if not isinstance(context, OracleContext):
            raise Sop05InputError(
                f"OracleContext payload for {state_id!r} has type "
                f"{type(context).__name__}"
            )
        try:
            validate_oracle_context(context, grid)
        except (ContractError, KeyError, TypeError, ValueError) as exc:
            raise Sop05InputError(
                f"OracleContext contract failed for {state_id!r}: {exc}"
            ) from exc

        _require(state.state_id == record.state_id, "BaseState state_id mismatch")
        _require(state.split == record.split, "BaseState split mismatch")
        _require(
            state.recording_id == record.recording_id,
            "BaseState recording_id mismatch",
        )
        _require(
            math.isclose(
                float(state.timestamp), record.timestamp, rel_tol=0.0, abs_tol=1e-9
            ),
            "BaseState timestamp mismatch",
        )
        _require(
            state.dynamic_object_ids == record.dynamic_object_ids,
            "BaseState dynamic_object_ids mismatch",
        )
        _require(
            context.base_state_id == record.state_id,
            "OracleContext base_state_id mismatch",
        )
        _require(
            tuple(sorted(context.dynamic_object_future))
            == record.oracle_dynamic_object_ids,
            "OracleContext dynamic_object_ids mismatch",
        )
        source_recording = context.metadata.get("source_recording_id")
        _require(
            source_recording in (None, record.recording_id),
            "OracleContext source_recording_id mismatch",
        )
        return state, context


@dataclass(frozen=True)
class Sop04TrajectoryBank:
    """Validated canonical trajectory bank in manifest array order."""

    root: Path
    trajectories: tuple[LocalTrajectory, ...]
    by_id: dict[str, LocalTrajectory]
    producer_evidence: ProducerEvidence
    trajectory_bank_version: str
    pose_time_layout_version: str
    pose_time_offsets_s: tuple[float, ...]
    pose_time_offsets_sha256: str
    bank_semantic_digest_sha256: str
    external_handoff_digest_sha256: str


@dataclass(frozen=True)
class StablePair:
    """One stable SOP-05 base-state/trajectory pair."""

    state_id: str
    trajectory_id: str
    seed: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Sop05InputError(message)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON must not contain {value}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise Sop05InputError(f"invalid {label} at {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Sop05InputError(f"failed to read {label} at {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        _require(bool(line.strip()), f"{label} contains blank line {line_number}")
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise Sop05InputError(
                f"invalid {label} JSON at line {line_number}: {exc}"
            ) from exc
        _require(
            isinstance(value, dict),
            f"{label} line {line_number} must be a JSON object",
        )
        rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Sop05InputError(f"failed to hash {path}: {exc}") from exc
    return digest.hexdigest()


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _safe_relative_path(root: Path, raw: object, label: str) -> tuple[str, Path]:
    _require(isinstance(raw, str) and bool(raw), f"{label} must be a path string")
    _require("\\" not in raw, f"unsafe {label}: {raw!r}")
    relative = PurePosixPath(raw)
    _require(
        not relative.is_absolute()
        and relative.parts
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"unsafe {label}: {raw!r}",
    )
    canonical = relative.as_posix()
    _require(canonical == raw, f"unsafe {label}: {raw!r}")
    candidate = root.joinpath(*relative.parts)
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise Sop05InputError(f"{label} does not resolve: {raw!r}: {exc}") from exc
    _require(
        resolved == resolved_root or resolved_root in resolved.parents,
        f"unsafe {label}: {raw!r}",
    )
    _require(candidate.is_file(), f"{label} is not a file: {raw!r}")
    _require(not candidate.is_symlink(), f"unsafe {label}: symlink {raw!r}")
    return canonical, candidate


def _parse_checksum_manifest(
    root: Path, manifest_path: Path
) -> dict[str, tuple[str, Path]]:
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Sop05InputError(f"failed to read checksum manifest: {exc}") from exc
    _require(lines, "checksum manifest must not be empty")
    entries: dict[str, tuple[str, Path]] = {}
    for line_number, line in enumerate(lines, start=1):
        match = _CHECKSUM_LINE.fullmatch(line)
        _require(
            match is not None,
            f"malformed checksum manifest line {line_number}",
        )
        assert match is not None
        expected, raw = match.groups()
        canonical, path = _safe_relative_path(root, raw, "checksum path")
        _require(
            canonical not in entries,
            f"duplicate checksum path: {canonical!r}",
        )
        entries[canonical] = (expected, path)
    return entries


def _actual_payload_files(root: Path, excluded: set[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    try:
        candidates = root.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative in excluded:
                continue
            _require(not path.is_symlink(), f"payload set contains symlink: {relative}")
            files[relative] = path
    except OSError as exc:
        raise Sop05InputError(f"failed to enumerate payload set: {exc}") from exc
    return files


def _verify_checksum_entries(
    entries: dict[str, tuple[str, Path]], checksum_workers: int
) -> dict[str, str]:
    workers = _positive_int(checksum_workers, "checksum_workers")
    ordered = sorted(entries.items())
    with ThreadPoolExecutor(max_workers=workers) as executor:
        actual = list(executor.map(lambda item: _sha256_file(item[1][1]), ordered))
    verified: dict[str, str] = {}
    for (relative, (expected, _)), observed in zip(ordered, actual):
        _require(
            observed == expected,
            f"checksum mismatch for {relative}: expected {expected}, got {observed}",
        )
        verified[relative] = observed
    return verified


def _validate_sop03_checksum_envelope(
    root: Path, checksum_workers: int
) -> tuple[dict[str, str], str]:
    manifest_path = root / "artifact_checksums.sha256"
    summary_path = root / "artifact_checksum_summary.json"
    summary = _load_json(summary_path, "SOP-03 checksum summary")
    _require(summary.get("status") == "complete", "checksum summary status")
    _require(
        summary.get("checksum_algorithm") == "sha256",
        "checksum algorithm must be sha256",
    )
    _require(
        summary.get("checksum_manifest") == manifest_path.name,
        "checksum manifest name mismatch",
    )
    manifest_sha256 = _sha256_file(manifest_path)
    _require(
        summary.get("checksum_manifest_sha256") == manifest_sha256,
        "checksum manifest digest mismatch",
    )
    excluded_raw = summary.get("excluded_paths")
    _require(isinstance(excluded_raw, list), "checksum excluded_paths must be a list")
    _require(
        all(isinstance(item, str) for item in excluded_raw),
        "checksum excluded_paths must contain strings",
    )
    excluded = set(excluded_raw)
    required = {summary_path.name, manifest_path.name}
    allowed = required | {".manual-review-incomplete"}
    _require(
        required <= excluded <= allowed and len(excluded) == len(excluded_raw),
        "checksum excluded_paths policy mismatch",
    )
    entries = _parse_checksum_manifest(root, manifest_path)
    actual = _actual_payload_files(root, excluded)
    _require(
        set(entries) == set(actual),
        "checksum payload set does not exactly match files on disk",
    )
    _require(
        type(summary.get("covered_file_count")) is int
        and summary["covered_file_count"] == len(entries),
        "checksum covered_file_count mismatch",
    )
    verified = _verify_checksum_entries(entries, checksum_workers)
    total_bytes = sum(path.stat().st_size for path in actual.values())
    _require(
        type(summary.get("covered_total_bytes")) is int
        and summary["covered_total_bytes"] == total_bytes,
        "checksum covered_total_bytes mismatch",
    )
    return verified, manifest_sha256


def _split_digest(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} split digest missing")
    return value


def _provenance_digest(value: object, label: str) -> str:
    _require(isinstance(value, dict), f"{label} split provenance must be an object")
    return _split_digest(value.get("split_manifest_digest"), label)


def _validate_sop03_readiness(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    marker = root / ".producer-complete"
    _require(marker.is_file(), "SOP-03 .producer-complete marker is missing")
    _require(not marker.is_symlink(), "SOP-03 .producer-complete marker is unsafe")
    _require(marker.stat().st_size == 0, "SOP-03 .producer-complete must be empty")
    run = _load_json(root / "run_manifest.json", "SOP-03 run manifest")
    audit = _load_json(root / "audit_report.json", "SOP-03 audit report")
    _require(run.get("status") == "complete", "SOP-03 run status is not complete")
    validation = run.get("validation")
    _require(isinstance(validation, dict), "SOP-03 validation evidence missing")
    _require(validation.get("status") == "passed", "SOP-03 validation did not pass")
    _require(
        validation.get("audit_report") == "audit_report.json",
        "SOP-03 validation audit report mismatch",
    )
    _require(audit.get("status") == "ok", "SOP-03 audit status is not ok")
    repository = run.get("repository")
    _require(isinstance(repository, dict), "SOP-03 repository evidence missing")
    run_commit = repository.get("code_commit")
    audit_commit = audit.get("code_commit")
    _require(
        isinstance(run_commit, str) and _COMMIT.fullmatch(run_commit) is not None,
        "SOP-03 run commit is invalid",
    )
    _require(
        isinstance(audit_commit, str) and _COMMIT.fullmatch(audit_commit) is not None,
        "SOP-03 audit commit is invalid",
    )
    _require(run_commit == audit_commit, "SOP-03 producer commit mismatch")
    base_audit = audit.get("base_states")
    snippet_audit = audit.get("snippets")
    split_audit = audit.get("split")
    _require(isinstance(base_audit, dict), "SOP-03 base state audit missing")
    _require(isinstance(snippet_audit, dict), "SOP-03 snippet audit missing")
    _require(isinstance(split_audit, dict), "SOP-03 split audit missing")
    _require(
        base_audit.get("shape_dtype_finite_contract_validation") == "passed_all",
        "SOP-03 base state contract audit did not pass",
    )
    _require(
        snippet_audit.get("strict_source_window_and_array_validation")
        == "passed_all",
        "SOP-03 snippet contract audit did not pass",
    )
    _require(
        split_audit.get("disallowed_overlap_count") == 0,
        "SOP-03 split overlap audit did not pass",
    )
    inputs = run.get("inputs")
    _require(isinstance(inputs, dict), "SOP-03 run inputs missing")
    run_digest = _split_digest(inputs.get("split_manifest_digest"), "run")
    audit_digest = _split_digest(split_audit.get("manifest_digest"), "audit")
    _require(run_digest == audit_digest, "SOP-03 split digest mismatch")
    return run, audit, run_commit, run_digest


def _validate_schema(row: dict[str, Any], label: str) -> None:
    _require(
        row.get("schema_version") == SCHEMA_VERSION,
        f"{label} schema_version mismatch",
    )


def _require_string(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be non-empty")
    return value


def _require_string_list(value: object, label: str) -> tuple[str, ...]:
    _require(isinstance(value, list), f"{label} must be a list")
    _require(
        all(isinstance(item, str) and bool(item) for item in value),
        f"{label} must contain non-empty strings",
    )
    result = tuple(value)
    _require(result == tuple(sorted(set(result))), f"{label} must be sorted and unique")
    return result


def _validate_layout(summary: dict[str, Any], label: str) -> None:
    expected: tuple[tuple[str, object], ...] = (
        ("sample_count", _SNIPPET_STEPS),
        ("history_steps", _HISTORY_STEPS),
        ("current_index", _CURRENT_INDEX),
        ("future_steps", _FUTURE_STEPS),
        ("motion_snippet_layout_version", _LAYOUT_VERSION),
    )
    for key, value in expected:
        _require(summary.get(key) == value, f"{label} layout mismatch for {key}")
    for key, expected_float in (
        ("sample_dt_s", _SAMPLE_DT_S),
        ("duration_s", _SNIPPET_DURATION_S),
    ):
        value = summary.get(key)
        _require(
            type(value) in {int, float}
            and math.isfinite(float(value))
            and math.isclose(
                float(value), expected_float, rel_tol=0.0, abs_tol=1e-9
            ),
            f"{label} layout mismatch for {key}",
        )


def _validate_sop03_manifests(
    root: Path,
    split: str,
    grid: GridSpec,
    audit: dict[str, Any],
    expected_digest: str,
) -> dict[str, Sop03ManifestRecord]:
    split_root = root / "base_states" / split
    base_rows = _load_jsonl(
        split_root / "base_state_manifest.jsonl", "base state manifest"
    )
    oracle_rows = _load_jsonl(
        split_root / "oracle_context_manifest.jsonl", "oracle context manifest"
    )
    summary = _load_json(split_root / "summary.json", "base state summary")
    _validate_schema(summary, "base state summary")
    _require(summary.get("split") == split, "base state summary split mismatch")
    _require(summary.get("history_steps") == _HISTORY_STEPS, "history layout mismatch")
    _require(summary.get("future_steps") == _FUTURE_STEPS, "future layout mismatch")
    _require(grid.history_steps == _HISTORY_STEPS, "grid history layout mismatch")
    _require(grid.future_steps == _FUTURE_STEPS, "grid future layout mismatch")
    summary_digest = _provenance_digest(
        summary.get("split_provenance"), "base state summary"
    )
    _require(summary_digest == expected_digest, "base state split digest mismatch")
    _require(
        type(summary.get("accepted_count")) is int
        and summary["accepted_count"] == len(base_rows),
        "base state count mismatch",
    )
    _require(len(base_rows) == len(oracle_rows), "base/oracle manifest count mismatch")

    base_by_id: dict[str, dict[str, Any]] = {}
    for row in base_rows:
        _validate_schema(row, "base state manifest row")
        _require(row.get("split") == split, "base state manifest split mismatch")
        state_id = _require_string(row.get("state_id"), "state_id")
        _require(
            state_id not in base_by_id,
            f"duplicate base state state_id: {state_id!r}",
        )
        _require(
            _provenance_digest(row.get("split_provenance"), "base state row")
            == expected_digest,
            "base state split digest mismatch",
        )
        base_by_id[state_id] = row

    oracle_by_id: dict[str, dict[str, Any]] = {}
    for row in oracle_rows:
        _validate_schema(row, "oracle context manifest row")
        state_id = _require_string(row.get("base_state_id"), "base_state_id")
        _require(
            state_id not in oracle_by_id,
            f"duplicate oracle base_state_id: {state_id!r}",
        )
        _require(
            _provenance_digest(row.get("split_provenance"), "oracle context row")
            == expected_digest,
            "oracle context split digest mismatch",
        )
        oracle_by_id[state_id] = row
    _require(
        set(base_by_id) == set(oracle_by_id),
        "base/oracle manifests contain unpaired state IDs",
    )

    base_audit = audit["base_states"]
    accepted_counts = base_audit.get("accepted_counts")
    _require(isinstance(accepted_counts, dict), "base state audit counts missing")
    _require(
        accepted_counts.get(split) == len(base_rows),
        "base state count disagrees with audit",
    )

    result: dict[str, Sop03ManifestRecord] = {}
    for state_id in sorted(base_by_id):
        base_row = base_by_id[state_id]
        oracle_row = oracle_by_id[state_id]
        recording_id = _require_string(
            base_row.get("recording_id"), "recording_id"
        )
        _require(
            oracle_row.get("source_recording_id") == recording_id,
            "base/oracle source recording mismatch",
        )
        _, base_path = _safe_relative_path(
            split_root, base_row.get("base_state_file"), "base state file path"
        )
        _, oracle_path = _safe_relative_path(
            split_root,
            oracle_row.get("oracle_context_file"),
            "oracle context file path",
        )
        timestamp = base_row.get("timestamp")
        _require(
            type(timestamp) in {int, float} and math.isfinite(float(timestamp)),
            "base state timestamp must be finite",
        )
        result[state_id] = Sop03ManifestRecord(
            state_id=state_id,
            recording_id=recording_id,
            split=split,
            base_state_path=base_path,
            oracle_context_path=oracle_path,
            timestamp=float(timestamp),
            dynamic_object_ids=_require_string_list(
                base_row.get("dynamic_object_ids"), "dynamic_object_ids"
            ),
            oracle_dynamic_object_ids=_require_string_list(
                oracle_row.get("source_dynamic_object_ids"),
                "source_dynamic_object_ids",
            ),
        )
    return result


def _validate_source_row(
    row: dict[str, Any], split: str, object_type: str, expected_digest: str
) -> str:
    _validate_schema(row, "snippet source row")
    _require(row.get("split") == split, "snippet source split mismatch")
    _require(row.get("object_type") == object_type, "snippet source type mismatch")
    _validate_layout(row, "snippet source")
    _require(
        _split_digest(row.get("split_manifest_digest"), "snippet source")
        == expected_digest,
        "snippet source split digest mismatch",
    )
    _require(
        _provenance_digest(row.get("split_provenance"), "snippet source")
        == expected_digest,
        "snippet source split digest mismatch",
    )
    return _require_string(row.get("snippet_id"), "snippet_id")


def _validate_sop03_libraries(
    root: Path,
    split: str,
    audit: dict[str, Any],
    expected_digest: str,
) -> dict[str, SnippetLibrary]:
    snippet_audit = audit["snippets"]
    accepted_counts = snippet_audit.get("accepted_counts")
    _require(isinstance(accepted_counts, dict), "snippet audit counts missing")
    split_counts = accepted_counts.get(split)
    _require(isinstance(split_counts, dict), "snippet split audit counts missing")
    libraries: dict[str, SnippetLibrary] = {}
    for object_type in DYNAMIC_OBJECT_TYPES:
        directory = root / "snippets" / split / object_type
        summary = _load_json(directory / "summary.json", "snippet summary")
        _validate_schema(summary, "snippet summary")
        _require(summary.get("split") == split, "snippet summary split mismatch")
        _require(
            summary.get("object_type") == object_type,
            "snippet summary object_type mismatch",
        )
        _validate_layout(summary, "snippet summary")
        _require(
            _split_digest(summary.get("split_manifest_digest"), "snippet summary")
            == expected_digest,
            "snippet summary split digest mismatch",
        )
        _require(
            _provenance_digest(summary.get("split_provenance"), "snippet summary")
            == expected_digest,
            "snippet summary split digest mismatch",
        )
        _require(
            isinstance(summary.get("array_sha256"), str)
            and _SHA256.fullmatch(summary["array_sha256"]) is not None,
            "snippet summary array_sha256 is invalid",
        )
        library_path = directory / "snippet_library.npz"
        try:
            with np.load(library_path, allow_pickle=False) as payload:
                _require(
                    set(payload.files)
                    == {"positions", "velocities", "headings", "meta_json"},
                    "snippet library array keys mismatch",
                )
                positions = payload["positions"]
                velocities = payload["velocities"]
                headings = payload["headings"]
                metadata = json.loads(
                    str(payload["meta_json"]),
                    parse_constant=_reject_json_constant,
                )
                _require(
                    positions.dtype == np.float32
                    and velocities.dtype == np.float32
                    and headings.dtype == np.float32,
                    "snippet library array dtype mismatch",
                )
                count = positions.shape[0] if positions.ndim == 3 else -1
                _require(
                    positions.shape == (count, _SNIPPET_STEPS, 2)
                    and velocities.shape == (count, _SNIPPET_STEPS, 2)
                    and headings.shape == (count, _SNIPPET_STEPS),
                    "snippet library array shape mismatch",
                )
                _require(
                    np.isfinite(positions).all()
                    and np.isfinite(velocities).all()
                    and np.isfinite(headings).all(),
                    "snippet library arrays contain NaN/Inf",
                )
        except Sop05InputError:
            raise
        except Exception as exc:
            raise Sop05InputError(f"invalid snippet library arrays: {exc}") from exc
        _require(isinstance(metadata, dict), "snippet metadata must be an object")
        _validate_schema(metadata, "snippet metadata")
        _require(
            metadata.get("object_type") == object_type,
            "snippet metadata object_type mismatch",
        )
        _require(
            metadata.get("summary")
            == {
                key: value
                for key, value in summary.items()
                if key != "schema_version"
            }
            or metadata.get("summary") == summary,
            "snippet embedded summary mismatch",
        )
        metadata_rows = metadata.get("snippets")
        _require(isinstance(metadata_rows, list), "snippet metadata rows missing")
        _require(len(metadata_rows) == count, "snippet metadata count mismatch")
        _require(
            type(summary.get("accepted_count")) is int
            and summary["accepted_count"] == count,
            "snippet accepted count mismatch",
        )
        _require(
            split_counts.get(object_type) == count,
            "snippet count disagrees with audit",
        )
        source_rows = _load_jsonl(
            directory / "source_manifest.jsonl", "snippet source manifest"
        )
        _require(len(source_rows) == count, "snippet source count mismatch")
        source_ids = [
            _validate_source_row(row, split, object_type, expected_digest)
            for row in source_rows
        ]
        _require(
            len(source_ids) == len(set(source_ids)),
            "duplicate snippet IDs in source manifest",
        )
        try:
            library = load_snippet_library(library_path)
        except Exception as exc:
            raise Sop05InputError(f"snippet library contract failed: {exc}") from exc
        _require(library.object_type == object_type, "snippet library type mismatch")
        _require(
            library.summary
            == {
                key: value
                for key, value in summary.items()
                if key != "schema_version"
            }
            or library.summary == summary,
            "snippet library summary mismatch",
        )
        library_ids = [snippet.snippet_id for snippet in library.snippets]
        _require(
            len(library_ids) == len(set(library_ids)),
            "duplicate snippet IDs in library",
        )
        _require(
            set(source_ids) == set(library_ids),
            "snippet IDs disagree between source manifest and library",
        )
        for snippet in library.snippets:
            _require(snippet.split == split, "snippet split mismatch")
            _require(snippet.object_type == object_type, "snippet object_type mismatch")
            _require(
                math.isclose(
                    float(snippet.duration_s),
                    _SNIPPET_DURATION_S,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
                "snippet duration layout mismatch",
            )
        libraries[object_type] = library
    return libraries


def load_sop03_split_inputs(
    root: str | Path,
    split: str,
    grid: GridSpec,
    *,
    checksum_workers: int = 1,
) -> Sop03SplitInputs:
    """Validate and expose one SOP-03 split without eagerly loading pairs."""
    root_path = Path(root)
    _require(root_path.is_dir(), f"SOP-03 root is not a directory: {root_path}")
    _require(not root_path.is_symlink(), "SOP-03 root must not be a symlink")
    _require(split in _SPLITS, f"unsupported split: {split!r}")
    _require(isinstance(grid, GridSpec), "grid must be a GridSpec")
    run, audit, code_commit, split_digest = _validate_sop03_readiness(root_path)
    checksums, checksum_manifest_sha256 = _validate_sop03_checksum_envelope(
        root_path, checksum_workers
    )
    manifest_index = _validate_sop03_manifests(
        root_path, split, grid, audit, split_digest
    )
    typed_libraries = _validate_sop03_libraries(
        root_path, split, audit, split_digest
    )
    counts = run.get("counts")
    _require(isinstance(counts, dict), "SOP-03 run counts missing")
    base_audit = audit["base_states"]
    snippet_audit = audit["snippets"]
    _require(
        counts.get("base_states") == base_audit.get("total_accepted_count"),
        "SOP-03 total base state count mismatch",
    )
    _require(
        counts.get("oracle_contexts") == base_audit.get("total_accepted_count"),
        "SOP-03 total oracle context count mismatch",
    )
    _require(
        counts.get("snippets") == snippet_audit.get("total_accepted_count"),
        "SOP-03 total snippet count mismatch",
    )
    evidence = ProducerEvidence(
        root=root_path.resolve(),
        code_commit=code_commit,
        checksum_manifest_sha256=checksum_manifest_sha256,
        audit_sha256=_sha256_file(root_path / "audit_report.json"),
        completion_policy="sop03_complete_marker_v1",
        payload_checksums=checksums,
    )
    return Sop03SplitInputs(
        root=root_path.resolve(),
        split=split,
        manifest_index=manifest_index,
        typed_libraries=typed_libraries,
        producer_evidence=evidence,
    )


def _sop04_pose_time_offsets() -> tuple[float, ...]:
    return tuple(
        float(value)
        for value in (np.arange(_FUTURE_STEPS, dtype=np.float64) + 1.0)
        * _SAMPLE_DT_S
    )


def _sop04_pose_time_offsets_sha256(offsets: tuple[float, ...]) -> str:
    payload = json.dumps(
        list(offsets),
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(b"sop04_pose_time_offsets_v1\0" + payload).hexdigest()


def _validate_sop04_time_contract(
    value: dict[str, Any],
    label: str,
    *,
    require_bank_version: bool,
) -> tuple[float, ...]:
    if require_bank_version:
        _require(
            value.get("trajectory_bank_version")
            == SOP04_TRAJECTORY_BANK_VERSION,
            f"{label} trajectory bank version mismatch",
        )
    _require(
        value.get("pose_time_layout_version")
        == SOP04_POSE_TIME_LAYOUT_VERSION,
        f"{label} pose-time layout version mismatch",
    )
    _require(
        type(value.get("trajectory_steps")) is int
        and value["trajectory_steps"] == _FUTURE_STEPS,
        f"{label} trajectory_steps mismatch",
    )
    expected = {
        "dt_s": _SAMPLE_DT_S,
        "first_pose_time_s": _SAMPLE_DT_S,
        "last_pose_time_s": _FUTURE_STEPS * _SAMPLE_DT_S,
    }
    for name, expected_value in expected.items():
        observed = value.get(name)
        _require(
            type(observed) in {int, float}
            and math.isfinite(float(observed))
            and math.isclose(
                float(observed), expected_value, rel_tol=0.0, abs_tol=1e-12
            ),
            f"{label} {name} mismatch",
        )
    return _sop04_pose_time_offsets()


def _validate_sop04_readiness(
    root: Path,
) -> tuple[dict[str, Any], str, str]:
    audit = _load_json(root / "audit_report.json", "SOP-04 audit report")
    _require(audit.get("status") == "ok", "SOP-04 audit status is not ok")
    _require(
        audit.get("schema_version") == SCHEMA_VERSION,
        "SOP-04 audit schema_version mismatch",
    )
    _validate_sop04_time_contract(
        audit, "SOP-04 audit", require_bank_version=True
    )
    for name in (
        "artifact_reload_validation",
        "shape_dtype_finite_validation",
        "future_endpoint_kinematics",
        "query_map_invariants",
        "manifest_array_alignment",
        "summary_npz_alignment",
        "checksum_verification",
    ):
        expected = (
            "passed"
            if name == "artifact_reload_validation"
            else "passed_all"
        )
        _require(
            audit.get(name) == expected,
            f"SOP-04 {name.replace('_', ' ')} audit did not pass",
        )
    _require(
        audit.get("determinism_reference_exact_match") is True,
        "SOP-04 determinism reference exact-match audit did not pass",
    )
    _require(
        audit.get("serial_parallel_exact_match") is True,
        "SOP-04 serial_parallel exact-match audit did not pass",
    )
    _require(audit.get("trajectory_count") == 21, "SOP-04 trajectory count mismatch")
    provenance = audit.get("provenance")
    _require(isinstance(provenance, dict), "SOP-04 audit provenance missing")
    _require(
        provenance.get("canonical_shared_bank") is True,
        "SOP-04 audit is not the canonical shared bank",
    )
    code_commit = provenance.get("code_commit")
    _require(
        isinstance(code_commit, str) and _COMMIT.fullmatch(code_commit) is not None,
        "SOP-04 audit commit is invalid",
    )
    _require(
        audit.get("checksum_file") == "artifact_checksums.sha256",
        "SOP-04 checksum file evidence mismatch",
    )
    semantic_digest = audit.get("bank_semantic_digest_sha256")
    _require(
        isinstance(semantic_digest, str)
        and _SHA256.fullmatch(semantic_digest) is not None,
        "SOP-04 bank semantic digest is invalid",
    )
    return audit, code_commit, semantic_digest


def _validate_sop04_checksum_envelope(
    root: Path, audit: dict[str, Any], checksum_workers: int
) -> tuple[dict[str, str], str]:
    manifest_path = root / "artifact_checksums.sha256"
    manifest_sha256 = _sha256_file(manifest_path)
    _require(
        audit.get("checksum_manifest_sha256") == manifest_sha256,
        "SOP-04 checksum manifest digest mismatch",
    )
    entries = _parse_checksum_manifest(root, manifest_path)
    excluded = {
        "artifact_checksums.sha256",
        "audit_report.json",
        "external_handoff_digest.sha256",
    }
    actual = _actual_payload_files(root, excluded)
    _require(
        set(entries) == _SOP04_CORE_PAYLOADS
        and set(actual) == _SOP04_CORE_PAYLOADS,
        "SOP-04 checksum payload set must contain exactly the v2 core files",
    )
    _require(
        type(audit.get("checksummed_payload_file_count")) is int
        and audit["checksummed_payload_file_count"] == len(entries),
        "SOP-04 checksummed payload file count mismatch",
    )
    return _verify_checksum_entries(entries, checksum_workers), manifest_sha256


def _validate_sop04_external_handoff(
    root: Path,
    *,
    expected_digest: object,
) -> str:
    _require(
        isinstance(expected_digest, str)
        and _SHA256.fullmatch(expected_digest) is not None,
        "expected SOP-04 external handoff digest must be 64 lowercase hex",
    )
    path = root / "external_handoff_digest.sha256"
    _require(path.is_file(), "SOP-04 external handoff digest is missing")
    _require(not path.is_symlink(), "SOP-04 external handoff digest is unsafe")
    try:
        line = path.read_text(encoding="utf-8").rstrip("\n")
    except (OSError, UnicodeError) as exc:
        raise Sop05InputError("failed to read SOP-04 external handoff digest") from exc
    match = re.fullmatch(
        rf"([0-9a-f]{{64}})  {_SOP04_EXTERNAL_HANDOFF_LABEL}", line
    )
    _require(match is not None, "SOP-04 external handoff digest is malformed")
    assert match is not None
    observed = match.group(1)
    hasher = hashlib.sha256()
    try:
        hasher.update(_SOP04_EXTERNAL_HANDOFF_DOMAIN)
        hasher.update((root / "artifact_checksums.sha256").read_bytes())
        hasher.update(b"\0")
        hasher.update((root / "audit_report.json").read_bytes())
    except OSError as exc:
        raise Sop05InputError(
            "failed to read SOP-04 external handoff envelope"
        ) from exc
    computed = hasher.hexdigest()
    _require(
        observed == computed,
        "SOP-04 external handoff digest does not match its envelope",
    )
    _require(
        observed == expected_digest,
        "SOP-04 external handoff digest does not match the trusted handoff",
    )
    return observed


def _load_sop04_arrays(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    expected_keys = {
        "poses",
        "controls",
        "swept_masks",
        "tta_maps",
        "braking_maps",
        "centerline_maps",
        "task_costs",
        "meta_json",
    }
    try:
        with np.load(path, allow_pickle=False) as payload:
            _require(
                set(payload.files) == expected_keys,
                "SOP-04 trajectory bank array keys mismatch",
            )
            metadata = json.loads(
                str(payload["meta_json"]), parse_constant=_reject_json_constant
            )
            arrays = {
                key: payload[key].copy(order="C")
                for key in expected_keys
                if key != "meta_json"
            }
    except Sop05InputError:
        raise
    except Exception as exc:
        raise Sop05InputError(f"invalid SOP-04 trajectory bank: {exc}") from exc
    _require(isinstance(metadata, dict), "SOP-04 metadata must be an object")
    _validate_schema(metadata, "SOP-04 embedded metadata")
    _require(
        metadata.get("trajectory_bank_version")
        == SOP04_TRAJECTORY_BANK_VERSION,
        "SOP-04 embedded metadata trajectory bank version mismatch",
    )
    _require(
        metadata.get("pose_time_layout_version")
        == SOP04_POSE_TIME_LAYOUT_VERSION,
        "SOP-04 embedded metadata pose-time layout version mismatch",
    )
    return arrays, metadata


def _validate_sop04_array_contracts(
    arrays: dict[str, np.ndarray], grid: GridSpec, count: int
) -> None:
    shapes = {
        "poses": (count, _FUTURE_STEPS, 3),
        "controls": (count, _FUTURE_STEPS, 2),
        "swept_masks": (count, grid.height, grid.width),
        "tta_maps": (count, grid.height, grid.width),
        "braking_maps": (count, grid.height, grid.width),
        "centerline_maps": (count, grid.height, grid.width),
        "task_costs": (count,),
    }
    for key, shape in shapes.items():
        array = arrays[key]
        _require(array.dtype == np.float32, f"SOP-04 {key} dtype mismatch")
        _require(array.shape == shape, f"SOP-04 {key} shape mismatch")
        _require(np.isfinite(array).all(), f"SOP-04 {key} contains NaN/Inf")
    swept = arrays["swept_masks"]
    _require(
        np.logical_or(swept == 0.0, swept == 1.0).all(),
        "SOP-04 swept-mask query invariant failed",
    )
    tta = arrays["tta_maps"]
    _require(
        np.equal(tta[swept == 0.0], -1.0).all()
        and np.greater_equal(tta[swept == 1.0], 0.0).all(),
        "SOP-04 TTA query invariant failed",
    )
    centerline = arrays["centerline_maps"]
    _require(
        np.logical_or(centerline == 0.0, centerline == 1.0).all(),
        "SOP-04 centerline query invariant failed",
    )


def _manifest_int(value: object, label: str) -> int:
    _require(type(value) is int, f"SOP-04 {label} must be an int")
    return value


def _manifest_float(value: object, label: str) -> float:
    _require(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value)),
        f"SOP-04 {label} must be finite",
    )
    return float(value)


def _close(left: object, right: object, label: str) -> None:
    left_float = _manifest_float(left, label)
    right_float = _manifest_float(right, label)
    _require(
        math.isclose(left_float, right_float, rel_tol=1e-6, abs_tol=1e-7),
        f"SOP-04 {label} mismatch",
    )


def _build_sop04_trajectories(
    root: Path,
    grid: GridSpec,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[LocalTrajectory, ...]:
    pose_time_offsets = _validate_sop04_time_contract(
        summary, "SOP-04 summary", require_bank_version=True
    )
    ids_raw = metadata.get("trajectory_ids")
    rows_metadata = metadata.get("trajectory_metadata")
    _require(isinstance(ids_raw, list), "SOP-04 trajectory_ids must be a list")
    _require(
        all(isinstance(value, str) and bool(value) for value in ids_raw),
        "SOP-04 trajectory_ids must be non-empty strings",
    )
    trajectory_ids = tuple(ids_raw)
    count = len(trajectory_ids)
    _require(count == 21, "SOP-04 trajectory count must be 21")
    _require(len(set(trajectory_ids)) == count, "duplicate SOP-04 trajectory_id")
    _require(isinstance(rows_metadata, list), "SOP-04 trajectory metadata missing")
    _require(
        len(rows_metadata) == count
        and all(isinstance(item, dict) for item in rows_metadata),
        "SOP-04 trajectory metadata count mismatch",
    )
    _validate_sop04_array_contracts(arrays, grid, count)

    manifest_rows = _load_jsonl(
        root / "trajectory_manifest.jsonl", "SOP-04 trajectory manifest"
    )
    indices = [
        _manifest_int(row.get("array_index"), "array_index")
        for row in manifest_rows
    ]
    _require(
        len(indices) == count and sorted(indices) == list(range(count)),
        "SOP-04 manifest array_index must cover 0..20 exactly",
    )
    _require(
        len(indices) == len(set(indices)),
        "duplicate SOP-04 manifest array_index",
    )
    ordered_rows = sorted(manifest_rows, key=lambda row: int(row["array_index"]))
    trajectories: list[LocalTrajectory] = []
    for index, row in enumerate(ordered_rows):
        _validate_schema(row, "SOP-04 trajectory manifest row")
        trajectory_id = _require_string(row.get("trajectory_id"), "trajectory_id")
        _require(
            trajectory_id == trajectory_ids[index],
            "SOP-04 manifest trajectory_id alignment mismatch",
        )
        _require(
            row.get("trajectory_steps") == _FUTURE_STEPS,
            "SOP-04 manifest trajectory_steps mismatch",
        )
        _validate_sop04_time_contract(
            row,
            f"SOP-04 trajectory manifest row {index}",
            require_bank_version=True,
        )
        _require(
            row.get("query_map_shape") == [grid.height, grid.width],
            "SOP-04 manifest query map shape mismatch",
        )
        item_metadata = rows_metadata[index]
        assert isinstance(item_metadata, dict)
        _validate_sop04_time_contract(
            item_metadata,
            f"SOP-04 trajectory metadata {index}",
            require_bank_version=False,
        )
        _close(row.get("v_mps"), item_metadata.get("v"), "velocity")
        _close(row.get("omega_radps"), item_metadata.get("omega"), "yaw rate")
        _close(row.get("task_cost"), arrays["task_costs"][index], "task cost")
        for flag in ("is_stop", "is_reverse"):
            _require(type(row.get(flag)) is bool, f"SOP-04 manifest {flag} invalid")
            _require(
                type(item_metadata.get(flag)) is bool
                and row[flag] == item_metadata[flag],
                f"SOP-04 manifest {flag} mismatch",
            )
        controls = arrays["controls"][index]
        _require(
            np.array_equal(
                controls,
                np.repeat(controls[:1], _FUTURE_STEPS, axis=0),
            ),
            "SOP-04 canonical trajectory controls must be constant",
        )
        v = float(controls[0, 0])
        omega = float(controls[0, 1])
        times = np.asarray(pose_time_offsets, dtype=np.float64)
        expected_poses = np.zeros((_FUTURE_STEPS, 3), dtype=np.float64)
        if omega == 0.0:
            expected_poses[:, 0] = v * times
        else:
            yaw = omega * times
            expected_poses[:, 0] = (v / omega) * np.sin(yaw)
            expected_poses[:, 1] = (v / omega) * (1.0 - np.cos(yaw))
            expected_poses[:, 2] = yaw
        _require(
            np.allclose(
                arrays["poses"][index],
                expected_poses,
                rtol=0.0,
                atol=1e-6,
            ),
            "SOP-04 poses do not follow future endpoint time semantics",
        )
        trajectories.append(
            LocalTrajectory(
                trajectory_id=trajectory_id,
                poses=arrays["poses"][index].copy(order="C"),
                controls=arrays["controls"][index].copy(order="C"),
                swept_mask=arrays["swept_masks"][index].copy(order="C"),
                tta_map=arrays["tta_maps"][index].copy(order="C"),
                braking_map=arrays["braking_maps"][index].copy(order="C"),
                centerline_map=arrays["centerline_maps"][index].copy(order="C"),
                task_cost=float(arrays["task_costs"][index]),
                metadata={
                    **item_metadata,
                    "array_index": index,
                    "trajectory_bank_version": SOP04_TRAJECTORY_BANK_VERSION,
                    "pose_time_offsets_s": pose_time_offsets,
                },
            )
        )
    _require(
        sum(item.metadata["is_stop"] is True for item in trajectories) == 1,
        "SOP-04 bank must contain one stop trajectory",
    )
    _require(
        any(
            item.trajectory_id == "stop" and item.metadata["is_stop"] is True
            for item in trajectories
        ),
        "SOP-04 stop trajectory identity mismatch",
    )
    _require(summary.get("accepted_count") == count, "SOP-04 summary count mismatch")
    return tuple(trajectories)


def load_sop04_trajectory_bank(
    root: str | Path,
    grid: GridSpec,
    *,
    expected_external_handoff_digest_sha256: str,
    checksum_workers: int = 1,
) -> Sop04TrajectoryBank:
    """Validate and load the canonical SOP-04 trajectory bank."""
    root_path = Path(root)
    _require(root_path.is_dir(), f"SOP-04 root is not a directory: {root_path}")
    _require(not root_path.is_symlink(), "SOP-04 root must not be a symlink")
    _require(isinstance(grid, GridSpec), "grid must be a GridSpec")
    _require(grid.future_steps == _FUTURE_STEPS, "SOP-04 trajectory layout mismatch")
    _require(grid.height > 0 and grid.width > 0, "SOP-04 grid shape is invalid")
    audit, code_commit, bank_semantic_digest = _validate_sop04_readiness(
        root_path
    )
    checksums, checksum_manifest_sha256 = _validate_sop04_checksum_envelope(
        root_path, audit, checksum_workers
    )
    external_handoff_digest = _validate_sop04_external_handoff(
        root_path,
        expected_digest=expected_external_handoff_digest_sha256,
    )
    summary = _load_json(root_path / "summary.json", "SOP-04 summary")
    _validate_schema(summary, "SOP-04 summary")
    pose_time_offsets = _validate_sop04_time_contract(
        summary, "SOP-04 summary", require_bank_version=True
    )
    provenance = summary.get("provenance")
    _require(isinstance(provenance, dict), "SOP-04 summary provenance missing")
    _require(
        provenance.get("canonical_shared_bank") is True,
        "SOP-04 summary is not the canonical shared bank",
    )
    _require(
        provenance.get("code_commit") == code_commit,
        "SOP-04 producer commit mismatch",
    )
    _require(
        audit.get("provenance") == provenance,
        "SOP-04 audit/summary provenance mismatch",
    )
    _require(summary.get("accepted_count") == 21, "SOP-04 summary count mismatch")
    _require(summary.get("candidate_count") == 21, "SOP-04 candidate count mismatch")
    _require(summary.get("rejected_count") == 0, "SOP-04 rejected count mismatch")
    _require(summary.get("array_dtype") == "float32", "SOP-04 array dtype mismatch")
    _require(
        summary.get("trajectory_steps") == _FUTURE_STEPS,
        "SOP-04 trajectory step count mismatch",
    )
    _require(summary.get("grid_height") == grid.height, "SOP-04 grid height mismatch")
    _require(summary.get("grid_width") == grid.width, "SOP-04 grid width mismatch")
    _require(
        type(summary.get("grid_resolution_m")) in {int, float}
        and math.isclose(
            float(summary["grid_resolution_m"]),
            float(grid.resolution_m),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "SOP-04 grid resolution mismatch",
    )
    _require(
        summary.get("meets_minimum_acceptance_rate") is True,
        "SOP-04 acceptance threshold was not met",
    )
    arrays, metadata = _load_sop04_arrays(root_path / "trajectory_bank.npz")
    embedded_summary = metadata.get("summary")
    external_embedded = {
        key: value
        for key, value in summary.items()
        if key not in {"schema_version", "provenance"}
    }
    _require(
        embedded_summary == external_embedded,
        "SOP-04 embedded summary disagrees with external summary",
    )
    trajectories = _build_sop04_trajectories(
        root_path, grid, arrays, metadata, summary
    )
    _require(
        audit.get("trajectory_count") == len(trajectories),
        "SOP-04 audit trajectory count mismatch",
    )
    by_id = {item.trajectory_id: item for item in trajectories}
    evidence = ProducerEvidence(
        root=root_path.resolve(),
        code_commit=code_commit,
        checksum_manifest_sha256=checksum_manifest_sha256,
        audit_sha256=_sha256_file(root_path / "audit_report.json"),
        completion_policy=SOP04_COMPLETION_POLICY,
        payload_checksums=checksums,
    )
    return Sop04TrajectoryBank(
        root=root_path.resolve(),
        trajectories=trajectories,
        by_id=by_id,
        producer_evidence=evidence,
        trajectory_bank_version=SOP04_TRAJECTORY_BANK_VERSION,
        pose_time_layout_version=SOP04_POSE_TIME_LAYOUT_VERSION,
        pose_time_offsets_s=pose_time_offsets,
        pose_time_offsets_sha256=_sop04_pose_time_offsets_sha256(
            pose_time_offsets
        ),
        bank_semantic_digest_sha256=bank_semantic_digest,
        external_handoff_digest_sha256=external_handoff_digest,
    )


def build_stable_pair_schedule(
    sop03: Sop03SplitInputs,
    sop04: Sop04TrajectoryBank,
    *,
    seed: int,
    max_base_states: int,
    trajectory_count: int,
) -> tuple[StablePair, ...]:
    """Build a stable, order-independent base-state/trajectory schedule."""
    if not isinstance(sop03, Sop03SplitInputs):
        raise TypeError("sop03 must be Sop03SplitInputs")
    if not isinstance(sop04, Sop04TrajectoryBank):
        raise TypeError("sop04 must be Sop04TrajectoryBank")
    if type(seed) is not int:
        raise TypeError("seed must be an int")
    state_limit = _positive_int(max_base_states, "max_base_states")
    trajectory_limit = _positive_int(trajectory_count, "trajectory_count")

    def rank(kind: str, identifier: str) -> tuple[str, str]:
        return (
            stable_digest(
                "sop05_pair_schedule_order_v1",
                kind,
                seed,
                sop03.split,
                identifier,
            ),
            identifier,
        )

    state_ids = tuple(
        sorted(sop03.manifest_index, key=lambda value: rank("base_state", value))[
            :state_limit
        ]
    )
    trajectory_ids = tuple(
        sorted(sop04.by_id, key=lambda value: rank("trajectory", value))[
            :trajectory_limit
        ]
    )
    pairs = tuple(
        StablePair(
            state_id=state_id,
            trajectory_id=trajectory_id,
            seed=int(
                derive_seed(
                    seed,
                    "sop05_pair_seed_v1",
                    sop03.split,
                    state_id,
                    trajectory_id,
                )
            ),
        )
        for state_id in state_ids
        for trajectory_id in trajectory_ids
    )
    _require(
        len({(pair.state_id, pair.trajectory_id) for pair in pairs}) == len(pairs),
        "stable pair schedule contains duplicate pairs",
    )
    return pairs
