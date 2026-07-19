"""Independent audit and fail-closed publication for complete SOP-03 bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.contracts import (
    DYNAMIC_OBJECT_TYPES,
    SCHEMA_VERSION,
    BaseState,
    OracleContext,
    build_grid_spec,
    load_dataclass,
    validate_base_state,
    validate_oracle_context,
)
from src.datasets.snippet_library import (
    MotionSnippet,
    _motion_statistics,
    _normalize_motion,
    load_snippet_library,
)
from src.datasets.split_manager import SPLIT_NAMES
from src.datasets.thor_adapter import load_recording_indexes_from_dir
from src.utils.config import load_config


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256_EMPTY = hashlib.sha256(b"").hexdigest()
_CHECKSUM_MANIFEST = "artifact_checksums.sha256"
_CHECKSUM_SUMMARY = "artifact_checksum_summary.json"
_COMPLETE_MARKER = ".producer-complete"
_FORMAL_OUTPUTS = (
    _COMPLETE_MARKER,
    _CHECKSUM_MANIFEST,
    _CHECKSUM_SUMMARY,
    "audit_report.json",
    "rejection_report.json",
    "run_manifest.json",
)
_LAYOUT = {
    "sample_count": 23,
    "history_steps": 8,
    "current_index": 7,
    "future_steps": 15,
    "sample_dt_s": 0.2,
    "duration_s": 4.4,
    "motion_snippet_layout_version": "history8_current7_future15_v1",
}


class Sop03PublicationError(ValueError):
    """Raised when a SOP-03 directory cannot be safely published."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Sop03PublicationError(message)


def validate_commit(value: str, label: str) -> str:
    """Return a canonical Git identity or fail closed."""
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise Sop03PublicationError(f"{label} must be 40 lowercase hex characters")
    return value


def _positive_int(value: int, label: str) -> int:
    if type(value) is not int or value < 1:
        raise Sop03PublicationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise Sop03PublicationError(f"{label} must be a non-negative integer")
    return value


def _validate_filter_counts(
    summary: dict[str, Any], *, accepted: int, label: str
) -> tuple[int, int, dict[str, int]]:
    candidate = _nonnegative_int(summary.get("candidate_count"), f"{label} candidate")
    declared_accepted = _nonnegative_int(
        summary.get("accepted_count"), f"{label} accepted"
    )
    rejected = _nonnegative_int(summary.get("rejected_count"), f"{label} rejected")
    _require(declared_accepted == accepted, f"{label} accepted count mismatch")
    _require(candidate == accepted + rejected, f"{label} count identity mismatch")
    reasons_raw = summary.get("rejection_reasons")
    _require(isinstance(reasons_raw, dict), f"{label} rejection reasons missing")
    reasons = {
        str(reason): _nonnegative_int(count, f"{label} rejection {reason}")
        for reason, count in reasons_raw.items()
    }
    _require(sum(reasons.values()) == rejected, f"{label} rejection sum mismatch")
    return candidate, rejected, reasons


def _reject_constant(value: str) -> None:
    raise Sop03PublicationError(f"JSON must not contain {value}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Sop03PublicationError(f"invalid {label} at {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must contain an object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Sop03PublicationError(f"failed to read {label}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        _require(bool(line.strip()), f"{label} contains blank line {line_number}")
        try:
            value = json.loads(line, parse_constant=_reject_constant)
        except json.JSONDecodeError as exc:
            raise Sop03PublicationError(
                f"invalid {label} line {line_number}: {exc}"
            ) from exc
        _require(isinstance(value, dict), f"{label} line must be an object")
        rows.append(value)
    return rows


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Sop03PublicationError(f"failed to hash {path}: {exc}") from exc
    return digest.hexdigest()


def _payload_files(root: Path) -> list[tuple[str, Path]]:
    excluded = {_CHECKSUM_MANIFEST, _CHECKSUM_SUMMARY, _COMPLETE_MARKER}
    result: list[tuple[str, Path]] = []
    try:
        for path in root.rglob("*"):
            if path.is_symlink():
                raise Sop03PublicationError(
                    f"publication payload contains symlink: {path.relative_to(root)}"
                )
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative not in excluded:
                result.append((relative, path))
    except OSError as exc:
        raise Sop03PublicationError(f"failed to enumerate payloads: {exc}") from exc
    return sorted(result)


@contextmanager
def _publication_lock(root: Path) -> Iterable[None]:
    lock_path = root.parent / f".{root.name}.sop03-publication.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise Sop03PublicationError(
            f"another finalizer owns publication lock: {lock_path}"
        ) from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _publish_checksum_envelope_locked(
    root_path: Path, *, workers: int
) -> dict[str, Any]:
    _require(root_path.is_dir(), f"artifact root is not a directory: {root_path}")
    _require(not root_path.is_symlink(), "artifact root must not be a symlink")
    worker_count = _positive_int(workers, "workers")
    for name in (_COMPLETE_MARKER, _CHECKSUM_MANIFEST, _CHECKSUM_SUMMARY):
        _require(not (root_path / name).exists(), f"{name} already exists")

    payloads = _payload_files(root_path)
    manifest_path = root_path / _CHECKSUM_MANIFEST
    summary_path = root_path / _CHECKSUM_SUMMARY
    marker_path = root_path / _COMPLETE_MARKER
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            digests = list(executor.map(lambda item: _sha256_file(item[1]), payloads))
        entries = [(_COMPLETE_MARKER, _SHA256_EMPTY)] + [
            (relative, digest)
            for (relative, _), digest in zip(payloads, digests)
        ]
        entries.sort()
        manifest_text = "".join(
            f"{digest}  {relative}\n" for relative, digest in entries
        )
        _atomic_text(manifest_path, manifest_text)
        summary = {
            "checksum_algorithm": "sha256",
            "checksum_manifest": _CHECKSUM_MANIFEST,
            "checksum_manifest_sha256": _sha256_file(manifest_path),
            "covered_file_count": len(entries),
            "covered_total_bytes": sum(path.stat().st_size for _, path in payloads),
            "excluded_paths": [_CHECKSUM_SUMMARY, _CHECKSUM_MANIFEST],
            "hash_workers": worker_count,
            "status": "complete",
        }
        _atomic_json(summary_path, summary)
        with marker_path.open("xb"):
            pass
        return summary
    except Exception:
        marker_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise


def publish_checksum_envelope(root: str | Path, *, workers: int) -> dict[str, Any]:
    """Hash every payload and publish the empty completion marker last."""
    root_path = Path(root)
    with _publication_lock(root_path):
        return _publish_checksum_envelope_locked(root_path, workers=workers)


def _schema_and_digest(
    value: dict[str, Any],
    expected_digest: str,
    label: str,
    *,
    require_direct: bool = False,
) -> None:
    _require(value.get("schema_version") == SCHEMA_VERSION, f"{label} schema")
    direct = value.get("split_manifest_digest")
    provenance = value.get("split_provenance")
    nested = (
        provenance.get("split_manifest_digest")
        if isinstance(provenance, dict)
        else None
    )
    _require(nested == expected_digest, f"{label} nested split digest mismatch")
    if require_direct:
        _require(direct == expected_digest, f"{label} direct split digest mismatch")
    elif direct is not None:
        _require(direct == expected_digest, f"{label} direct split digest mismatch")


def _safe_file(root: Path, raw: object, label: str) -> Path:
    _require(isinstance(raw, str) and bool(raw), f"{label} path missing")
    candidate = root / raw
    _require(not candidate.is_symlink(), f"{label} must not be a symlink")
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    _require(resolved.is_relative_to(resolved_root), f"{label} escapes its root")
    _require(resolved.is_file(), f"{label} file missing")
    return resolved


def _audit_pair(
    split_root: Path,
    base_row: dict[str, Any],
    oracle_row: dict[str, Any],
    grid: Any,
) -> None:
    state = load_dataclass(
        _safe_file(split_root, base_row.get("base_state_file"), "BaseState")
    )
    context = load_dataclass(
        _safe_file(
            split_root, oracle_row.get("oracle_context_file"), "OracleContext"
        )
    )
    _require(isinstance(state, BaseState), "BaseState payload class mismatch")
    _require(isinstance(context, OracleContext), "OracleContext payload class mismatch")
    validate_base_state(state, grid)
    validate_oracle_context(context, grid)
    state_id = base_row.get("state_id")
    recording_id = base_row.get("recording_id")
    _require(state.state_id == state_id, "BaseState state_id mismatch")
    _require(context.base_state_id == state_id, "OracleContext base_state_id mismatch")
    _require(state.recording_id == recording_id, "BaseState recording_id mismatch")
    _require(
        oracle_row.get("source_recording_id") == recording_id,
        "OracleContext recording_id mismatch",
    )
    _require(
        context.metadata.get("source_recording_id") in (None, recording_id),
        "OracleContext embedded recording_id mismatch",
    )
    _require(state.split == base_row.get("split"), "BaseState split mismatch")
    _require(
        math.isclose(
            float(state.timestamp),
            float(base_row.get("timestamp")),
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "BaseState timestamp mismatch",
    )
    _require(
        list(state.dynamic_object_ids) == base_row.get("dynamic_object_ids"),
        "BaseState dynamic object IDs mismatch",
    )
    _require(
        sorted(context.dynamic_object_future)
        == oracle_row.get("source_dynamic_object_ids"),
        "OracleContext dynamic object IDs mismatch",
    )
    _require(
        context.metadata.get("source_dynamic_object_ids")
        == oracle_row.get("source_dynamic_object_ids"),
        "OracleContext embedded dynamic object IDs mismatch",
    )


def _unique_rows(
    rows: Iterable[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        _require(isinstance(value, str) and bool(value), f"{label} {key} missing")
        _require(value not in result, f"duplicate {label} {key}: {value}")
        result[value] = row
    return result


def _audit_snippet_source(
    snippet: MotionSnippet, recordings_by_id: dict[str, Any]
) -> None:
    recording = recordings_by_id.get(snippet.source_recording_id)
    _require(recording is not None, "snippet source recording is missing")
    _require(
        snippet.source_session_id == recording.session_id,
        "snippet source session mismatch",
    )
    track = recording.dynamic_objects.get(snippet.source_object_id)
    _require(track is not None, "snippet source object is missing")
    _require(track.object_type == snippet.object_type, "snippet source type mismatch")
    _require(track.footprint == snippet.footprint, "snippet source footprint mismatch")
    sample_count = int(_LAYOUT["sample_count"])
    sample_dt = float(_LAYOUT["sample_dt_s"])
    expected_times = snippet.start_timestamp + np.arange(sample_count) * sample_dt
    start = int(np.searchsorted(track.timestamps, snippet.start_timestamp))
    stop = start + sample_count
    _require(stop <= track.timestamps.size, "snippet source window would extrapolate")
    indices = np.arange(start, stop)
    _require(
        np.allclose(
            track.timestamps[indices], expected_times, rtol=0.0, atol=1e-8
        ),
        "snippet source window is not on the frozen time grid",
    )
    _require(
        np.all(track.segment_ids[indices] == track.segment_ids[start]),
        "snippet source window crosses a track gap",
    )
    object_poses = track.poses[indices].astype(np.float64)
    source_positions = object_poses[:, :2]
    velocities, mean_speed, max_acceleration, curvature = _motion_statistics(
        source_positions, track.timestamps[indices]
    )
    normalized = _normalize_motion(
        source_positions, velocities, object_poses[:, 2]
    )
    _require(normalized is not None, "snippet source window is stationary")
    assert normalized is not None
    expected_arrays = tuple(item.astype(np.float32) for item in normalized)
    for label, actual, expected in zip(
        ("positions", "velocities", "headings"),
        (snippet.positions, snippet.velocities, snippet.headings),
        expected_arrays,
    ):
        _require(
            np.array_equal(actual, expected),
            f"snippet {label} differs from its measured source window",
        )
    for label, actual, expected in (
        ("mean_speed_mps", snippet.mean_speed_mps, mean_speed),
        ("max_acceleration_mps2", snippet.max_acceleration_mps2, max_acceleration),
        ("mean_abs_curvature_per_m", snippet.mean_abs_curvature_per_m, curvature),
    ):
        _require(
            math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12),
            f"snippet {label} differs from its measured source window",
        )


def audit_sop03_artifact(
    root: str | Path, *, base_config_path: str | Path, workers: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reopen and validate every scientific payload in an unpublished bundle."""
    root_path = Path(root)
    _require(root_path.is_dir(), f"artifact root is not a directory: {root_path}")
    _require(not root_path.is_symlink(), "artifact root must not be a symlink")
    worker_count = _positive_int(workers, "workers")
    for name in _FORMAL_OUTPUTS:
        _require(not (root_path / name).exists(), f"{name} already exists")

    split_summary = _read_json(root_path / "splits/split_summary.json", "split summary")
    overlap = _read_json(root_path / "splits/overlap_report.json", "split overlap")
    _require(split_summary.get("schema_version") == SCHEMA_VERSION, "split schema")
    split_manifest = root_path / "splits/split_manifest.jsonl"
    split_digest = hashlib.blake2b(
        split_manifest.read_bytes(), digest_size=16
    ).hexdigest()
    _require(split_summary.get("manifest_digest") == split_digest, "split digest")
    _require(overlap.get("manifest_digest") == split_digest, "overlap split digest")
    _require(overlap.get("status") == "ok", "split overlap status")
    _require(overlap.get("disallowed_overlap_count") == 0, "disallowed overlap")
    _require(
        overlap.get("allowed_overlap_count") == 5,
        "allowed session overlap count must remain five",
    )

    grid = build_grid_spec(load_config(base_config_path))
    base_counts: dict[str, int] = {}
    base_candidates: dict[str, int] = {}
    base_rejections: dict[str, Any] = {}
    snippet_counts: dict[str, dict[str, int]] = {}
    snippet_candidates: dict[str, dict[str, int]] = {}
    snippet_rejections: dict[str, dict[str, Any]] = {}
    recording_counts: dict[str, int] = {}
    recording_track_count = 0
    declared_npz: set[Path] = set()

    for split in SPLIT_NAMES:
        recording_root = root_path / "recording_indexes" / split
        recording_rows = _read_jsonl(
            recording_root / "recording_manifest.jsonl", "recording manifest"
        )
        for row in recording_rows:
            _schema_and_digest(row, split_digest, "recording row")
            _require(row.get("split") == split, "recording row split mismatch")
            declared_npz.add(
                _safe_file(
                    recording_root,
                    row.get("recording_index_file"),
                    "recording index",
                )
            )
        recordings = load_recording_indexes_from_dir(
            recording_root, expected_split=split
        )
        _require(
            len(recording_rows) == len(recordings),
            "recording manifest count mismatch",
        )
        recordings_by_id = {item.recording_id: item for item in recordings}
        _require(
            len(recordings_by_id) == len(recordings),
            "duplicate recording ID within split",
        )
        recording_counts[split] = len(recordings)
        recording_track_count += sum(len(item.dynamic_objects) for item in recordings)

        split_snippets: dict[str, int] = {}
        split_snippet_candidates: dict[str, int] = {}
        split_snippet_rejections: dict[str, Any] = {}
        for object_type in DYNAMIC_OBJECT_TYPES:
            directory = root_path / "snippets" / split / object_type
            summary = _read_json(directory / "summary.json", "snippet summary")
            _schema_and_digest(
                summary, split_digest, "snippet summary", require_direct=True
            )
            _require(summary.get("split") == split, "snippet summary split")
            _require(summary.get("object_type") == object_type, "snippet summary type")
            for key, expected in _LAYOUT.items():
                _require(summary.get(key) == expected, f"snippet layout {key}")
            library_path = directory / "snippet_library.npz"
            declared_npz.add(library_path.resolve())
            library = load_snippet_library(library_path)
            _require(
                library.summary
                == {
                    key: value
                    for key, value in summary.items()
                    if key != "schema_version"
                }
                or library.summary == summary,
                "snippet outer and embedded summaries disagree",
            )
            rows = _read_jsonl(directory / "source_manifest.jsonl", "source manifest")
            _require(
                len(rows) == len(library.snippets),
                "snippet source count mismatch",
            )
            row_ids: set[str] = set()
            for row in rows:
                _schema_and_digest(
                    row, split_digest, "snippet source", require_direct=True
                )
                _require(row.get("split") == split, "snippet source split")
                _require(row.get("object_type") == object_type, "snippet source type")
                for key, expected in _LAYOUT.items():
                    _require(row.get(key) == expected, f"snippet source layout {key}")
                snippet_id = row.get("snippet_id")
                _require(
                    isinstance(snippet_id, str) and snippet_id not in row_ids,
                    "duplicate or missing snippet_id",
                )
                row_ids.add(snippet_id)
                session_id = row.get("source_session_id")
                _require(
                    isinstance(session_id, str) and bool(session_id),
                    "snippet source_session_id missing",
                )
            _require(
                row_ids == {item.snippet_id for item in library.snippets},
                "snippet IDs disagree with source manifest",
            )
            rows_by_id = {str(row["snippet_id"]): row for row in rows}
            for snippet in library.snippets:
                row = rows_by_id[snippet.snippet_id]
                for field in (
                    "split",
                    "source_recording_id",
                    "source_session_id",
                    "source_object_id",
                    "object_type",
                ):
                    _require(
                        row.get(field) == getattr(snippet, field),
                        f"snippet embedded {field} mismatch",
                    )
                _require(
                    math.isclose(
                        float(row.get("start_timestamp")),
                        float(snippet.start_timestamp),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ),
                    "snippet embedded start_timestamp mismatch",
                )
                _require(
                    row.get("footprint") == snippet.footprint,
                    "snippet embedded footprint mismatch",
                )
                _audit_snippet_source(snippet, recordings_by_id)
            count = len(library.snippets)
            candidate, rejected, reasons = _validate_filter_counts(
                summary, accepted=count, label=f"snippet {split}/{object_type}"
            )
            split_snippets[object_type] = count
            split_snippet_candidates[object_type] = candidate
            split_snippet_rejections[object_type] = {
                "count": rejected,
                "reasons": reasons,
            }
        snippet_counts[split] = split_snippets
        snippet_candidates[split] = split_snippet_candidates
        snippet_rejections[split] = split_snippet_rejections

        split_root = root_path / "base_states" / split
        summary = _read_json(split_root / "summary.json", "base state summary")
        _schema_and_digest(summary, split_digest, "base state summary")
        _require(summary.get("split") == split, "base state summary split")
        _require(summary.get("history_steps") == grid.history_steps, "history steps")
        _require(summary.get("future_steps") == grid.future_steps, "future steps")
        base_rows = _read_jsonl(
            split_root / "base_state_manifest.jsonl", "base state manifest"
        )
        oracle_rows = _read_jsonl(
            split_root / "oracle_context_manifest.jsonl", "oracle context manifest"
        )
        base_by_id = _unique_rows(base_rows, "state_id", "BaseState")
        oracle_by_id = _unique_rows(oracle_rows, "base_state_id", "OracleContext")
        _require(set(base_by_id) == set(oracle_by_id), "base/oracle ID mismatch")
        for row in base_rows:
            _schema_and_digest(row, split_digest, "BaseState row")
            _require(row.get("split") == split, "BaseState row split")
        for row in oracle_rows:
            _schema_and_digest(row, split_digest, "OracleContext row")
        ordered = [
            (split_root, base_by_id[state_id], oracle_by_id[state_id], grid)
            for state_id in sorted(base_by_id)
        ]
        for _, base_row, oracle_row, _ in ordered:
            declared_npz.add(
                _safe_file(
                    split_root, base_row.get("base_state_file"), "BaseState"
                )
            )
            declared_npz.add(
                _safe_file(
                    split_root,
                    oracle_row.get("oracle_context_file"),
                    "OracleContext",
                )
            )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(lambda args: _audit_pair(*args), ordered))
        count = len(base_rows)
        candidate, rejected, reasons = _validate_filter_counts(
            summary, accepted=count, label=f"BaseState {split}"
        )
        base_counts[split] = count
        base_candidates[split] = candidate
        base_rejections[split] = {
            "count": rejected,
            "reasons": reasons,
        }

    actual_npz = {path.resolve() for path in root_path.rglob("*.npz")}
    _require(
        actual_npz == declared_npz,
        "NPZ payload set does not exactly match scientific manifests",
    )
    total_base = sum(base_counts.values())
    total_snippets = sum(sum(value.values()) for value in snippet_counts.values())
    total_recordings = sum(recording_counts.values())
    split_statistics = split_summary.get("split_statistics")
    _require(isinstance(split_statistics, dict), "split statistics are missing")
    for split, count in recording_counts.items():
        statistic = split_statistics.get(split)
        _require(isinstance(statistic, dict), f"split statistic missing for {split}")
        _require(
            statistic.get("record_count") == count,
            f"recording count disagrees for {split}",
        )
    _require(
        total_recordings == split_summary.get("metadata_record_count"),
        "recording count disagrees with split summary",
    )
    audit = {
        "status": "ok",
        "split": {
            "manifest_digest": split_digest,
            "recording_counts": recording_counts,
            "allowed_session_overlap_count": overlap.get("allowed_overlap_count"),
            "disallowed_overlap_count": 0,
        },
        "recording_indexes": {
            "recording_count": total_recordings,
            "dynamic_object_track_count": recording_track_count,
            "shape_dtype_finite_time_grid_validation": "passed_all",
        },
        "snippets": {
            "accepted_counts": snippet_counts,
            "candidate_counts": snippet_candidates,
            "total_accepted_count": total_snippets,
            "strict_source_window_and_array_validation": "passed_all",
        },
        "base_states": {
            "accepted_counts": base_counts,
            "candidate_counts": base_candidates,
            "total_accepted_count": total_base,
            "base_oracle_id_alignment": "passed_all",
            "shape_dtype_finite_contract_validation": "passed_all",
        },
    }
    counts = {
        "recordings": total_recordings,
        "dynamic_object_tracks": recording_track_count,
        "snippets": total_snippets,
        "base_states": total_base,
        "oracle_contexts": total_base,
    }
    rejection_report = {
        "base_states": base_rejections,
        "snippets": snippet_rejections,
        "status": "complete",
    }
    return audit, counts, rejection_report


def _finalize_sop03_artifact_locked(
    root: str | Path,
    *,
    base_config_path: str | Path,
    producer_commit: str,
    finalizer_commit: str,
    workers: int,
    producer_job_id: str | None = None,
) -> dict[str, Any]:
    """Audit a complete producer directory and publish its formal envelope."""
    root_path = Path(root)
    producer_identity = validate_commit(producer_commit, "producer commit")
    finalizer_identity = validate_commit(finalizer_commit, "finalizer commit")
    audit, counts, rejection_report = audit_sop03_artifact(
        root_path, base_config_path=base_config_path, workers=workers
    )
    base_config = load_config(base_config_path)
    base_config_file = Path(base_config_path)
    split_summary = _read_json(
        root_path / "splits/split_summary.json", "split summary"
    )
    split_digest = audit["split"]["manifest_digest"]
    published_at = datetime.now(timezone.utc).isoformat()
    audit = {
        **audit,
        "code_commit": producer_identity,
        "publication": {
            "finalizer_commit": finalizer_identity,
            "full_payload_audit": True,
            "published_at_utc": published_at,
            "schema_version": SCHEMA_VERSION,
        },
    }
    run_manifest = {
        "run_id": root_path.name,
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "repository": {
            "code_commit": producer_identity,
            "finalizer_commit": finalizer_identity,
        },
        "inputs": {
            "dataset": "THOR-MAGNI",
            "evaluation_scope": split_summary.get("evaluation_scope"),
            "raw_data_modified": False,
            "raw_recording_count": counts["recordings"],
            "source_assignment_sha256": split_summary.get(
                "source_assignment_sha256"
            ),
            "split_manifest_digest": split_digest,
        },
        "counts": counts,
        "parameters": {
            **_LAYOUT,
            "audit_workers": workers,
            "base_state_stride_s": 0.6,
            "recording_dt_s": 0.2,
            "recording_max_gap_s": 0.3,
            "snippet_stride_s": 1.0,
        },
        "producer_protocol": {
            "base_config_path": str(base_config_file),
            "base_config_sha256": _sha256_file(base_config_file),
            "base_config_snapshot": base_config,
            "scripts": [
                "scripts/00_freeze_thor_recording_split.py",
                "scripts/01_index_recordings.py",
                "scripts/02_build_snippet_library.py",
                "scripts/03_extract_base_states.py",
                "scripts/03_finalize_sop03_artifact.py",
            ],
            "source_data_policy": "read_only",
            "workers": workers,
        },
        "environment": {
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "published_at_utc": published_at,
            "slurm_producer_job_id": producer_job_id,
            "slurm_audit_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "outputs": {
            "audit_report": "audit_report.json",
            "rejection_report": "rejection_report.json",
            "artifact_checksums": _CHECKSUM_MANIFEST,
            "artifact_checksum_summary": _CHECKSUM_SUMMARY,
        },
        "validation": {
            "status": "passed",
            "audit_report": "audit_report.json",
        },
    }
    created: list[Path] = []
    try:
        for name, payload in (
            ("rejection_report.json", rejection_report),
            ("audit_report.json", audit),
            ("run_manifest.json", run_manifest),
        ):
            path = root_path / name
            _atomic_json(path, payload)
            created.append(path)
        checksum_summary = _publish_checksum_envelope_locked(
            root_path, workers=workers
        )
    except Exception:
        if not (root_path / _COMPLETE_MARKER).exists():
            for path in created:
                path.unlink(missing_ok=True)
        raise
    return {
        "audit": audit,
        "counts": counts,
        "run_manifest": run_manifest,
        "checksum_summary": checksum_summary,
    }


def finalize_sop03_artifact(
    root: str | Path,
    *,
    base_config_path: str | Path,
    producer_commit: str,
    finalizer_commit: str,
    workers: int,
    producer_job_id: str | None = None,
) -> dict[str, Any]:
    """Audit a complete producer directory and publish its formal envelope."""
    root_path = Path(root)
    with _publication_lock(root_path):
        return _finalize_sop03_artifact_locked(
            root_path,
            base_config_path=base_config_path,
            producer_commit=producer_commit,
            finalizer_commit=finalizer_commit,
            workers=workers,
            producer_job_id=producer_job_id,
        )
