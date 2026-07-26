"""Strict Schema 4 THOR Long40 inputs for the lightweight-TEB producer."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.contracts import BaseState, GridSpec, OracleContext, SCHEMA_VERSION
from src.datasets.base_state_index import extract_base_state_index
from src.datasets.long_snippet_library import (
    LongMotionSnippet,
    load_long_snippet_artifact,
)
from src.datasets.thor_adapter import load_recording_index


class Sop05rTebLong40InputError(ValueError):
    """Raised when the split Long40 inputs cannot be used by M4--M6."""


@dataclass(frozen=True)
class Sop05rTebLong40Inputs:
    """Validated Long40 snippets and materialized Schema 4 source pairs."""

    snippets: tuple[LongMotionSnippet, ...]
    state_pairs: tuple[tuple[BaseState, OracleContext], ...]
    source_evidence: Mapping[str, object]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Sop05rTebLong40InputError(message)


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Sop05rTebLong40InputError(f"invalid {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise Sop05rTebLong40InputError(f"failed to read {label}: {exc}") from exc
    _require(lines, f"{label} must not be empty")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Sop05rTebLong40InputError(
                f"invalid {label} row {line_number}: {exc}"
            ) from exc
        _require(
            isinstance(row, dict),
            f"{label} row {line_number} must be a JSON object",
        )
        rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise Sop05rTebLong40InputError(f"failed to hash {path}: {exc}") from exc


def _split_digest(value: object, label: str) -> str:
    _require(isinstance(value, dict), f"{label} split provenance is missing")
    digest = value.get("split_manifest_digest")
    _require(
        isinstance(digest, str) and len(digest) == 32,
        f"{label} split manifest digest is invalid",
    )
    return digest


def _safe_file(root: Path, raw: object, label: str) -> Path:
    _require(isinstance(raw, str) and bool(raw), f"{label} must be non-empty")
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise Sop05rTebLong40InputError(f"unsafe {label}: {raw!r}") from exc
    _require(candidate.is_file(), f"{label} is not a file: {raw!r}")
    _require(not candidate.is_symlink(), f"unsafe {label}: symlink {raw!r}")
    return candidate


def load_sop05r_teb_long40_inputs(
    *,
    recording_root: str | Path,
    long40_human_artifact: str | Path,
    split: str,
    grid: GridSpec,
    max_base_states: int,
) -> Sop05rTebLong40Inputs:
    """Load a standalone Long40 library with matching THOR recording indexes."""

    _require(split in {"train", "calibration", "val", "test"}, "invalid split")
    _require(
        isinstance(max_base_states, int) and not isinstance(max_base_states, bool)
        and max_base_states > 0,
        "max_base_states must be positive",
    )
    _require(grid.history_steps == 8, "Long40 inputs require eight history steps")
    _require(grid.future_steps == 32, "Long40 inputs require 32 future steps")

    artifact_root = Path(long40_human_artifact).resolve()
    _require(artifact_root.is_dir(), "Long40 human artifact is not a directory")
    _require(not artifact_root.is_symlink(), "Long40 human artifact is unsafe")
    try:
        library = load_long_snippet_artifact(artifact_root)
    except Exception as exc:
        raise Sop05rTebLong40InputError(
            f"Long40 human artifact validation failed: {exc}"
        ) from exc
    _require(library.object_type == "human", "Long40 artifact must contain humans")
    _require(library.summary.get("split") == split, "Long40 artifact split mismatch")
    _require(
        library.summary.get("sample_count") == 40
        and library.summary.get("history_steps") == 8
        and library.summary.get("future_steps") == 32,
        "Long40 artifact layout mismatch",
    )
    library_digest = _split_digest(
        library.split_provenance, "Long40 artifact"
    )

    root = Path(recording_root).resolve()
    _require(root.is_dir(), "recording index root is not a directory")
    _require(not root.is_symlink(), "recording index root is unsafe")
    split_root = root / "recording_indexes" / split
    _require(split_root.is_dir(), "recording index split directory is missing")
    summary_path = split_root / "summary.json"
    manifest_path = split_root / "recording_manifest.jsonl"
    summary = _read_json(summary_path, "recording index summary")
    _require(
        summary.get("schema_version") == SCHEMA_VERSION,
        "recording index summary schema mismatch",
    )
    _require(summary.get("split") == split, "recording index summary split mismatch")
    _require(
        _split_digest(summary.get("split_provenance"), "recording index summary")
        == library_digest,
        "Long40 and recording-index split provenance disagree",
    )
    rows = _read_jsonl(manifest_path, "recording index manifest")
    recordings = []
    recording_ids: set[str] = set()
    for row in rows:
        _require(
            row.get("schema_version") == SCHEMA_VERSION,
            "recording index manifest schema mismatch",
        )
        _require(row.get("split") == split, "recording index manifest split mismatch")
        _require(
            _split_digest(row.get("split_provenance"), "recording index manifest")
            == library_digest,
            "Long40 and recording-index manifest provenance disagree",
        )
        recording_id = row.get("recording_id")
        _require(
            isinstance(recording_id, str) and bool(recording_id),
            "recording index manifest recording_id is invalid",
        )
        _require(recording_id not in recording_ids, "duplicate recording id")
        recording_ids.add(recording_id)
        path = _safe_file(
            split_root,
            row.get("recording_index_file"),
            "recording index file",
        )
        try:
            recording = load_recording_index(path)
        except Exception as exc:
            raise Sop05rTebLong40InputError(
                f"recording index validation failed for {recording_id!r}: {exc}"
            ) from exc
        _require(
            recording.recording_id == recording_id,
            "recording index identity mismatch",
        )
        _require(
            math.isclose(recording.dt_s, 0.2, rel_tol=0.0, abs_tol=1e-9),
            "Long40 recording index dt_s must be 0.2",
        )
        recordings.append(recording)

    extraction = extract_base_state_index(
        tuple(recordings),
        split=split,
        grid=grid,
        stride_s=0.6,
        workers=1,
    )
    oracle_by_id = {
        context.base_state_id: context for context in extraction.oracle_contexts
    }
    pairs = tuple(
        (state, oracle_by_id[state.state_id])
        for state in extraction.base_states[:max_base_states]
    )
    _require(pairs, "recording indexes produced no usable base states")
    snippets = tuple(sorted(library.snippets, key=lambda item: item.snippet_id))
    _require(snippets, "Long40 human artifact contains no accepted snippets")

    return Sop05rTebLong40Inputs(
        snippets=snippets,
        state_pairs=pairs,
        source_evidence={
            "input_kind": "sop03_schema4_long40_runtime_v1",
            "recording_index_root": str(root),
            "recording_index_manifest_sha256": _sha256_file(manifest_path),
            "recording_index_summary_sha256": _sha256_file(summary_path),
            "recording_count": len(recordings),
            "materialized_base_state_count": len(pairs),
            "long40_human_artifact": str(artifact_root),
            "long40_checksum_manifest_sha256": _sha256_file(
                artifact_root / "artifact_checksums_40.sha256"
            ),
            "long40_semantic_digest": library.summary["semantic_digest_sha256"],
            "accepted_source_snippet_count": len(snippets),
        },
    )
