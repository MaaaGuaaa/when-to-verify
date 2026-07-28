"""Structured statistics and atomic visual-audit publication for SOP05R."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from src.contracts import build_grid_spec
from src.generation.paired_variants import (
    PairGenerationError,
    generate_paired_variants,
    load_paired_variant_config,
)
from src.generation.sop05_input_adapter import load_sop03_split_inputs
from src.generation.sop05r_output_loader import (
    LoadedSop05rEvents,
    load_sop05r_events,
)
from src.planning.verification_actions import (
    VerificationAction,
    VerificationActionLibrary,
)
from src.utils.atomic_publish import atomic_rename_noreplace

from .sop05r_visuals import (
    Sop05rVisualRequest,
    render_sop05r_visual_artifacts,
)


SOP05R_AUDIT_METRICS_VERSION = "sop05r_audit_metrics_v1"
SOP05R_AUDIT_COLLECTION_VERSION = "sop05r_visual_audit_collection_v1"
SOP05R_AUDIT_COMPLETION_VERSION = "sop05r_visual_audit_complete_v1"


class Sop05rAuditError(ValueError):
    """Raised when SOP05R audit inputs or artifacts violate their contract."""


@dataclass(frozen=True)
class Sop05rAuditMetrics:
    version: str
    sample_ids: tuple[str, ...]
    sample_id_digest: str
    counts: Mapping[str, int]
    rates: Mapping[str, float]
    history_counts: Mapping[str, int]
    history_fractions: Mapping[str, float]
    active_revealable_fraction: float
    attempts: Mapping[str, float | int | None]
    action_leads: Mapping[str, Mapping[str, float | int | None]]

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "sample_ids": list(self.sample_ids),
            "sample_id_digest": self.sample_id_digest,
            "counts": dict(self.counts),
            "rates": dict(self.rates),
            "history_counts": dict(self.history_counts),
            "history_fractions": dict(self.history_fractions),
            "active_revealable_fraction": self.active_revealable_fraction,
            "attempts": dict(self.attempts),
            "action_leads": {
                action_id: dict(values)
                for action_id, values in self.action_leads.items()
            },
        }


@dataclass(frozen=True)
class Sop05rAuditRequest:
    sop05r_root: Path
    sop03_root: Path
    paired_config_path: Path
    output_dir: Path
    sample_count: int
    seed: int
    checksum_workers: int


@dataclass(frozen=True)
class Sop05rAuditResult:
    status: str
    output_dir: Path
    selected_event_ids: tuple[str, ...]
    manifest_sha256: str
    checksum_manifest_sha256: str
    exit_code: int


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise Sop05rAuditError("SOP05R audit evidence must be canonical JSON") from exc


def _json_file_bytes(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise Sop05rAuditError(f"failed to hash audit artifact: {path}") from exc


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "minimum": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def build_sop05r_audit_metrics(
    source: LoadedSop05rEvents,
) -> Sop05rAuditMetrics:
    if not isinstance(source, LoadedSop05rEvents):
        raise TypeError("source must be LoadedSop05rEvents")
    summary = source.summary
    sample_ids = tuple(event.generated_event_id for event in source.events)
    if list(sample_ids) != summary["selected_event_ids"]:
        raise Sop05rAuditError("source event order differs from strict summary")
    count_names = (
        "input_base_count",
        "geometry_eligible_base_count",
        "template_count",
        "geometry_eligible_template_count",
        "planner_feasible_template_count",
        "exact_history_qualified_count",
        "time_aligned_collision_count",
        "active_revealable_count",
        "accepted_count",
        "selected_count",
    )
    counts = {name: int(summary[name]) for name in count_names}
    rates = {
        "geometry_eligible_base_fraction": _ratio(
            counts["geometry_eligible_base_count"],
            counts["input_base_count"],
        ),
        "geometry_eligible_template_fraction": _ratio(
            counts["geometry_eligible_template_count"],
            counts["template_count"],
        ),
        "planner_success_given_geometry": _ratio(
            counts["planner_feasible_template_count"],
            counts["geometry_eligible_template_count"],
        ),
        "collision_given_geometry": _ratio(
            counts["time_aligned_collision_count"],
            counts["geometry_eligible_template_count"],
        ),
        "end_to_end_acceptance": _ratio(
            counts["accepted_count"],
            counts["template_count"],
        ),
        "selection_fraction": _ratio(
            counts["selected_count"],
            counts["accepted_count"],
        ),
    }
    history = Counter(
        str(event.world.metadata["target_history_visibility_regime"])
        for event in source.events
    )
    history_counts = {
        "seen_then_occluded": history["seen_then_occluded"],
        "unseen_in_history_window": history["unseen_in_history_window"],
    }
    history_fractions = {
        regime: _ratio(count, len(source.events))
        for regime, count in history_counts.items()
    }
    active_count = sum(
        bool(event.world.metadata["active_revealable_action_ids"])
        for event in source.events
    )
    action_values: dict[str, list[float]] = {}
    for event in source.events:
        metadata = event.world.metadata
        actions = metadata.get("active_revealability_actions")
        if not isinstance(actions, Mapping):
            continue
        for action_id in metadata["active_revealable_action_ids"]:
            row = actions.get(action_id)
            if not isinstance(row, Mapping):
                raise Sop05rAuditError("active action evidence is missing")
            value = row.get("visibility_lead_lower_bound_s")
            if value is None:
                value = row.get("visibility_lead_s")
            if value is not None:
                lead = float(value)
                if not np.isfinite(lead):
                    raise Sop05rAuditError("active action lead is not finite")
                action_values.setdefault(action_id, []).append(lead)
    attempts = [float(value) for value in summary["attempts_per_accepted_event"]]
    return Sop05rAuditMetrics(
        version=SOP05R_AUDIT_METRICS_VERSION,
        sample_ids=sample_ids,
        sample_id_digest=_sha256_bytes(
            b"sop05r_audit_sample_ids_v1\0"
            + _canonical_json_bytes(list(sample_ids))
        ),
        counts=counts,
        rates=rates,
        history_counts=history_counts,
        history_fractions=history_fractions,
        active_revealable_fraction=_ratio(active_count, len(source.events)),
        attempts=_distribution(attempts),
        action_leads={
            action_id: _distribution(action_values[action_id])
            for action_id in sorted(action_values)
        },
    )


def _checksum_manifest_bytes(root: Path) -> bytes:
    rows = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise Sop05rAuditError(f"audit artifact must not be a symlink: {relative}")
        if not path.is_file() or relative in {
            "artifact_checksums.sha256",
            ".sop05r-audit-complete",
        }:
            continue
        rows.append(f"{_sha256_file(path)}  {relative}\n")
    return "".join(sorted(rows)).encode("ascii")


def _sample_directory_name(index: int, event_id: str) -> str:
    digest = _sha256_bytes(event_id.encode("utf-8"))[:12]
    return f"sample-{index:04d}-{digest}"


def publish_sop05r_audit(
    output_dir: str | Path,
    *,
    source: LoadedSop05rEvents,
    visual_requests: tuple[Sop05rVisualRequest, ...],
    required_sample_count: int,
    attempts: tuple[Mapping[str, object], ...] = (),
) -> Sop05rAuditResult:
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite audit output: {destination}")
    if not isinstance(source, LoadedSop05rEvents):
        raise TypeError("source must be LoadedSop05rEvents")
    if type(required_sample_count) is not int or required_sample_count <= 0:
        raise Sop05rAuditError("required_sample_count must be a positive integer")
    if not isinstance(visual_requests, tuple) or any(
        not isinstance(request, Sop05rVisualRequest) for request in visual_requests
    ):
        raise TypeError("visual_requests must be a tuple of Sop05rVisualRequest")
    source_ids = {event.generated_event_id for event in source.events}
    request_ids = tuple(request.event.generated_event_id for request in visual_requests)
    if len(request_ids) != len(set(request_ids)) or any(
        event_id not in source_ids for event_id in request_ids
    ):
        raise Sop05rAuditError("visual request IDs must be unique source event IDs")
    selected_requests = visual_requests[:required_sample_count]
    status = (
        "complete"
        if len(selected_requests) == required_sample_count
        else "insufficient_samples"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.sop05r-audit-staging-",
            dir=destination.parent,
        )
    )
    try:
        samples_dir = staging / "samples"
        samples_dir.mkdir()
        sample_rows = []
        for index, request in enumerate(selected_requests):
            directory_name = _sample_directory_name(
                index, request.event.generated_event_id
            )
            sample_dir = samples_dir / directory_name
            result = render_sop05r_visual_artifacts(request, sample_dir)
            audit_row = {
                "audit_version": SOP05R_AUDIT_COLLECTION_VERSION,
                "sample_index": index,
                "generated_event_id": request.event.generated_event_id,
                "pair_group_id": request.pair_group.pair_group_id,
                "base_state_id": request.base_state.state_id,
                "nominal_trajectory_id": request.trajectory_record.nominal_trajectory_id,
                "alternative_trajectory_ids": list(
                    request.trajectory_record.alternative_trajectory_ids
                ),
                "verification_action_id": request.verification_action_id,
                "event_replay": result.event_replay_metadata,
                "paired_events": result.paired_events_metadata,
            }
            audit_bytes = _json_file_bytes(audit_row)
            (sample_dir / "sample_audit.json").write_bytes(audit_bytes)
            sample_rows.append(
                {
                    "sample_index": index,
                    "generated_event_id": request.event.generated_event_id,
                    "directory": f"samples/{directory_name}",
                    "sample_audit_sha256": _sha256_bytes(audit_bytes),
                    "event_replay": result.event_replay_metadata,
                    "paired_events": result.paired_events_metadata,
                }
            )
        metrics = build_sop05r_audit_metrics(source)
        metrics_bytes = _json_file_bytes(metrics.as_dict())
        (staging / "audit_metrics.json").write_bytes(metrics_bytes)
        attempt_rows = [dict(row) for row in attempts]
        attempt_bytes = b"".join(
            _canonical_json_bytes(row) + b"\n" for row in attempt_rows
        )
        (staging / "pair_attempts.jsonl").write_bytes(attempt_bytes)
        manifest = {
            "manifest_version": SOP05R_AUDIT_COLLECTION_VERSION,
            "status": status,
            "source_run_id": source.manifest["run_id"],
            "source_publication_semantic_digest": (
                source.publication_semantic_digest
            ),
            "required_sample_count": required_sample_count,
            "sample_count": len(sample_rows),
            "selected_event_ids": [
                row["generated_event_id"] for row in sample_rows
            ],
            "source_sample_id_digest": metrics.sample_id_digest,
            "audit_metrics_sha256": _sha256_bytes(metrics_bytes),
            "pair_attempts_sha256": _sha256_bytes(attempt_bytes),
            "samples": sample_rows,
        }
        manifest_bytes = _json_file_bytes(manifest)
        (staging / "manifest.json").write_bytes(manifest_bytes)
        checksums = _checksum_manifest_bytes(staging)
        (staging / "artifact_checksums.sha256").write_bytes(checksums)
        if status == "complete":
            marker = {
                "marker_version": SOP05R_AUDIT_COMPLETION_VERSION,
                "status": "complete",
                "source_publication_semantic_digest": (
                    source.publication_semantic_digest
                ),
                "manifest_sha256": _sha256_bytes(manifest_bytes),
                "artifact_checksums_sha256": _sha256_bytes(checksums),
            }
            (staging / ".sop05r-audit-complete").write_bytes(
                _json_file_bytes(marker)
            )
        load_sop05r_audit(staging, require_complete=status == "complete")
        atomic_rename_noreplace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return Sop05rAuditResult(
        status=status,
        output_dir=destination,
        selected_event_ids=tuple(manifest["selected_event_ids"]),
        manifest_sha256=_sha256_bytes(manifest_bytes),
        checksum_manifest_sha256=_sha256_bytes(checksums),
        exit_code=0 if status == "complete" else 3,
    )


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid SOP05R audit {label}") from exc
    if not isinstance(value, dict) or payload != _json_file_bytes(value):
        raise ValueError(f"SOP05R audit {label} is not canonical")
    return value


def load_sop05r_audit(
    root: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("SOP05R audit root must be a real directory")
    observed = {path.name for path in directory.iterdir()}
    complete = ".sop05r-audit-complete" in observed
    expected = {
        "manifest.json",
        "audit_metrics.json",
        "pair_attempts.jsonl",
        "samples",
        "artifact_checksums.sha256",
    }
    if complete:
        expected.add(".sop05r-audit-complete")
    if observed != expected:
        raise ValueError("SOP05R audit root entries mismatch")
    if require_complete and not complete:
        raise ValueError("SOP05R audit completion marker is required")
    checksums = _checksum_manifest_bytes(directory)
    if (directory / "artifact_checksums.sha256").read_bytes() != checksums:
        raise ValueError("SOP05R audit checksum manifest mismatch")
    manifest = _load_json(directory / "manifest.json", label="manifest")
    expected_manifest_keys = {
        "manifest_version",
        "status",
        "source_run_id",
        "source_publication_semantic_digest",
        "required_sample_count",
        "sample_count",
        "selected_event_ids",
        "source_sample_id_digest",
        "audit_metrics_sha256",
        "pair_attempts_sha256",
        "samples",
    }
    if set(manifest) != expected_manifest_keys or manifest[
        "manifest_version"
    ] != SOP05R_AUDIT_COLLECTION_VERSION:
        raise ValueError("SOP05R audit manifest contract mismatch")
    if manifest["status"] not in {"complete", "insufficient_samples"}:
        raise ValueError("SOP05R audit status is invalid")
    if (manifest["status"] == "complete") != complete:
        raise ValueError("SOP05R audit status/completion mismatch")
    required = manifest["required_sample_count"]
    count = manifest["sample_count"]
    if type(required) is not int or required <= 0 or type(count) is not int or count < 0:
        raise ValueError("SOP05R audit sample counts are invalid")
    if complete and count != required:
        raise ValueError("complete SOP05R audit sample count mismatch")
    samples = manifest["samples"]
    selected_ids = manifest["selected_event_ids"]
    if (
        not isinstance(samples, list)
        or len(samples) != count
        or not isinstance(selected_ids, list)
        or selected_ids != [row.get("generated_event_id") for row in samples]
        or len(selected_ids) != len(set(selected_ids))
    ):
        raise ValueError("SOP05R audit sample identity mismatch")
    if _sha256_file(directory / "audit_metrics.json") != manifest[
        "audit_metrics_sha256"
    ]:
        raise ValueError("SOP05R audit metrics checksum mismatch")
    if _sha256_file(directory / "pair_attempts.jsonl") != manifest[
        "pair_attempts_sha256"
    ]:
        raise ValueError("SOP05R pair-attempt checksum mismatch")
    sample_dirs = set()
    for index, row in enumerate(samples):
        if not isinstance(row, dict) or set(row) != {
            "sample_index",
            "generated_event_id",
            "directory",
            "sample_audit_sha256",
            "event_replay",
            "paired_events",
        }:
            raise ValueError("SOP05R audit sample row keys mismatch")
        if row["sample_index"] != index:
            raise ValueError("SOP05R audit sample order mismatch")
        relative = row["directory"]
        if not isinstance(relative, str) or not relative.startswith("samples/sample-"):
            raise ValueError("SOP05R audit sample directory is invalid")
        sample_dir = directory / relative
        if sample_dir.is_symlink() or not sample_dir.is_dir():
            raise ValueError("SOP05R audit sample directory is missing")
        sample_dirs.add(sample_dir.name)
        if {path.name for path in sample_dir.iterdir()} != {
            "event_replay.gif",
            "paired_events.png",
            "sample_audit.json",
        }:
            raise ValueError("SOP05R audit sample files mismatch")
        if _sha256_file(sample_dir / "sample_audit.json") != row[
            "sample_audit_sha256"
        ]:
            raise ValueError("SOP05R sample audit checksum mismatch")
        for field, filename in (
            ("event_replay", "event_replay.gif"),
            ("paired_events", "paired_events.png"),
        ):
            metadata = row[field]
            if not isinstance(metadata, dict) or _sha256_file(
                sample_dir / filename
            ) != metadata.get("sha256"):
                raise ValueError("SOP05R visual checksum mismatch")
    observed_dirs = {
        path.name for path in (directory / "samples").iterdir() if path.is_dir()
    }
    if observed_dirs != sample_dirs:
        raise ValueError("SOP05R audit sample directory set mismatch")
    if complete:
        marker = _load_json(
            directory / ".sop05r-audit-complete",
            label="completion marker",
        )
        expected_marker = {
            "marker_version": SOP05R_AUDIT_COMPLETION_VERSION,
            "status": "complete",
            "source_publication_semantic_digest": manifest[
                "source_publication_semantic_digest"
            ],
            "manifest_sha256": _sha256_file(directory / "manifest.json"),
            "artifact_checksums_sha256": _sha256_file(
                directory / "artifact_checksums.sha256"
            ),
        }
        if marker != expected_marker:
            raise ValueError("SOP05R audit completion marker mismatch")
    return manifest


def _action_library_from_manifest(
    source: LoadedSop05rEvents,
) -> VerificationActionLibrary:
    raw = source.manifest["verification_action_config"]
    actions = tuple(
        VerificationAction(
            action_id=row["action_id"],
            duration_s=row["duration_s"],
            delta_forward_m=row["delta_forward_m"],
            delta_yaw_rad=float(np.deg2rad(row["delta_yaw_deg"])),
        )
        for row in raw["actions"]
    )
    return VerificationActionLibrary(
        schema_version=raw["schema_version"],
        library_version=raw["library_version"],
        sensor_fov_rad=float(np.deg2rad(raw["sensor_fov_deg"])),
        actions=actions,
    )


def _stable_event_order(
    events: tuple[object, ...],
    *,
    seed: int,
) -> tuple[object, ...]:
    return tuple(
        sorted(
            events,
            key=lambda event: _sha256_bytes(
                f"{seed}|{event.generated_event_id}".encode("utf-8")
            ),
        )
    )


def run_sop05r_audit(request: Sop05rAuditRequest) -> Sop05rAuditResult:
    if not isinstance(request, Sop05rAuditRequest):
        raise TypeError("request must be a Sop05rAuditRequest")
    if type(request.sample_count) is not int or request.sample_count <= 0:
        raise Sop05rAuditError("sample_count must be a positive integer")
    if type(request.seed) is not int or request.seed < 0:
        raise Sop05rAuditError("seed must be a nonnegative integer")
    if type(request.checksum_workers) is not int or request.checksum_workers <= 0:
        raise Sop05rAuditError("checksum_workers must be a positive integer")
    if request.output_dir.exists() or request.output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite audit output: {request.output_dir}")
    source = load_sop05r_events(request.sop05r_root, require_complete=True)
    grid = build_grid_spec(source.manifest["base_config"])
    sop03 = load_sop03_split_inputs(
        request.sop03_root,
        source.manifest["split"],
        grid,
        checksum_workers=request.checksum_workers,
    )
    expected_sop03 = source.manifest["input_lock"]["sop03"]
    observed_sop03 = {
        "code_commit": sop03.producer_evidence.code_commit,
        "checksum_manifest_sha256": (
            sop03.producer_evidence.checksum_manifest_sha256
        ),
        "audit_sha256": sop03.producer_evidence.audit_sha256,
        "completion_policy": sop03.producer_evidence.completion_policy,
    }
    if observed_sop03 != expected_sop03:
        raise Sop05rAuditError("SOP03 evidence differs from SOP05R input lock")
    paired_config = load_paired_variant_config(request.paired_config_path)
    action_library = _action_library_from_manifest(source)
    trajectory_by_event = {
        record.event_id: record for record in source.trajectory_store.records
    }
    snippets = {
        snippet.snippet_id: snippet
        for library in sop03.typed_libraries.values()
        for snippet in library.snippets
    }
    visual_requests = []
    attempt_rows = []
    for event in _stable_event_order(source.events, seed=request.seed):
        if len(visual_requests) >= request.sample_count:
            break
        if event.world.metadata["target_history_visibility_regime"] != (
            "seen_then_occluded"
        ):
            attempt_rows.append(
                {
                    "generated_event_id": event.generated_event_id,
                    "status": "skipped",
                    "reason": "unseen_history_not_visualized_v1",
                }
            )
            continue
        base_state, oracle_context = sop03.load_pair(event.world.base_state_id, grid)
        snippet = snippets.get(event.target.snippet_id)
        if snippet is None:
            raise Sop05rAuditError("source target snippet is missing from SOP03")
        try:
            group = generate_paired_variants(
                mother_event=event,
                source_snippet=snippet,
                base_state=base_state,
                trajectory=None,
                oracle_context=oracle_context,
                base_config=source.manifest["base_config"],
                paired_config=paired_config,
                seed=request.seed,
                sop05r_trajectory_store=source.trajectory_store,
            )
        except PairGenerationError as exc:
            attempt_rows.append(
                {
                    "generated_event_id": event.generated_event_id,
                    "status": "rejected",
                    "reason": exc.reason,
                }
            )
            continue
        if not group.is_complete or not group.eligible_for_strict_evaluation:
            attempt_rows.append(
                {
                    "generated_event_id": event.generated_event_id,
                    "status": "rejected",
                    "reason": "incomplete_pair_group",
                }
            )
            continue
        active_ids = event.world.metadata["active_revealable_action_ids"]
        action_id = active_ids[0] if active_ids else "forward_peek"
        visual_requests.append(
            Sop05rVisualRequest(
                event=event,
                trajectory_record=trajectory_by_event[event.generated_event_id],
                base_state=base_state,
                oracle_context=oracle_context,
                pair_group=group,
                base_config=source.manifest["base_config"],
                action_library=action_library,
                verification_action_id=action_id,
            )
        )
        attempt_rows.append(
            {
                "generated_event_id": event.generated_event_id,
                "status": "selected",
                "reason": "complete_pair_group",
            }
        )
    return publish_sop05r_audit(
        request.output_dir,
        source=source,
        visual_requests=tuple(visual_requests),
        required_sample_count=request.sample_count,
        attempts=tuple(attempt_rows),
    )
