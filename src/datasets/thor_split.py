"""THÖR metadata indexing and frozen recording-generalization splits."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Mapping

from src.datasets.split_manager import (
    SPLIT_NAMES,
    SplitAuditPolicy,
    SplitIndexError,
    SplitResult,
    freeze_preassigned_split,
    serialize_manifest,
    write_split_artifacts,
)
from src.datasets.thor_adapter import parse_recording_id


THOR_RECORDING_GENERALIZATION_POLICY = SplitAuditPolicy(
    evaluation_scope="unseen_recording_within_known_sessions",
    required_fields=("recording", "session", "seed_namespace"),
    allowed_overlap_fields=("session",),
    unavailable_fields=("participant",),
)


@dataclass(frozen=True)
class ThorRecordingSplitBuild:
    """All deterministic inputs and outputs for one frozen THÖR split."""

    metadata: tuple[dict[str, object], ...]
    result: SplitResult
    source_assignment_sha256: str
    metadata_digest: str


def _reject_json_constant(value: str) -> None:
    raise SplitIndexError(f"assignment manifest must not contain {value}")


def index_thor_recording_metadata(
    raw_root: str | Path,
) -> tuple[dict[str, object], ...]:
    """Read official FILE_ID rows without loading trajectory samples."""
    root = Path(raw_root)
    if not root.is_dir():
        raise FileNotFoundError(f"THÖR raw root was not found: {root}")
    paths = sorted(root.glob("Scenario_*/THOR-Magni_*.csv"))
    if not paths:
        raise SplitIndexError("THÖR raw root contains no scenario CSV files")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            first = next(csv.reader(handle), None)
        if first is None or len(first) < 2 or first[0].strip() != "FILE_ID":
            raise SplitIndexError(f"missing official FILE_ID header in {path.name}")
        file_id = first[1].strip()
        filename_id = parse_recording_id(path)
        if file_id != filename_id:
            raise SplitIndexError(
                f"FILE_ID does not match filename for {path.name}: {file_id!r}"
            )
        match = re.match(r"^(\d{6})_", file_id)
        if match is None:
            raise SplitIndexError(
                f"FILE_ID must start with a six-digit recording day: {file_id!r}"
            )
        if file_id in seen:
            raise SplitIndexError(f"duplicate THÖR recording id: {file_id}")
        seen.add(file_id)
        rows.append(
            {
                "recording_id": file_id,
                "session_id": match.group(1),
                "source_path": path.relative_to(root).as_posix(),
            }
        )
    return tuple(sorted(rows, key=lambda row: str(row["recording_id"])))


def load_frozen_recording_assignment(
    path: str | Path,
) -> dict[str, str]:
    """Load the approved legacy recording mapping without changing assignments."""
    assignment_path = Path(path)
    try:
        payload = json.loads(
            assignment_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise SplitIndexError(f"invalid assignment manifest: {error}") from error
    if not isinstance(payload, dict):
        raise SplitIndexError("assignment manifest must be a JSON object")
    keys = set(payload)
    validation_keys = keys & {"val", "validation"}
    if len(validation_keys) != 1:
        raise SplitIndexError("assignment manifest needs exactly one of val/validation")
    expected = {"train", "calibration", "test", *validation_keys}
    if keys != expected:
        raise SplitIndexError(
            "assignment manifest split keys must be train, calibration, "
            "val/validation, and test"
        )
    assignments: dict[str, str] = {}
    for declared_split, values in payload.items():
        if not isinstance(values, list):
            raise SplitIndexError(
                f"assignment values for {declared_split} must be a list"
            )
        split = "val" if declared_split == "validation" else declared_split
        for value in values:
            if not isinstance(value, str) or not value:
                raise SplitIndexError("assignment recording ids must be strings")
            prefix = "thor_magni::"
            recording_id = value[len(prefix) :] if value.startswith(prefix) else value
            if not recording_id:
                raise SplitIndexError("assignment recording ids must not be empty")
            if recording_id in assignments:
                raise SplitIndexError(
                    f"duplicate assignment for recording {recording_id}"
                )
            assignments[recording_id] = split
    if set(assignments.values()) != set(SPLIT_NAMES):
        raise SplitIndexError("all four splits must contain at least one recording")
    return dict(sorted(assignments.items()))


def build_thor_recording_split(
    *,
    raw_root: str | Path,
    assignment_manifest: str | Path,
    seed: int,
) -> ThorRecordingSplitBuild:
    """Enrich and audit the approved recording assignment."""
    metadata = index_thor_recording_metadata(raw_root)
    assignment_path = Path(assignment_manifest)
    assignment_payload = assignment_path.read_bytes()
    assignments = load_frozen_recording_assignment(assignment_path)
    result = freeze_preassigned_split(
        metadata,
        assignments,
        seed=seed,
        policy=THOR_RECORDING_GENERALIZATION_POLICY,
    )
    metadata_payload = serialize_manifest(metadata)
    assignment_sha256 = hashlib.sha256(assignment_payload).hexdigest()
    metadata_digest = hashlib.blake2b(
        metadata_payload, digest_size=16
    ).hexdigest()
    summary = {
        **result.summary,
        "source_assignment_sha256": assignment_sha256,
        "metadata_digest": metadata_digest,
        "metadata_record_count": len(metadata),
    }
    overlap_report = {
        **result.overlap_report,
        "source_assignment_sha256": assignment_sha256,
        "metadata_digest": metadata_digest,
    }
    enriched_result = SplitResult(
        manifest=result.manifest,
        summary=summary,
        overlap_report=overlap_report,
        manifest_digest=result.manifest_digest,
    )
    return ThorRecordingSplitBuild(
        metadata=metadata,
        result=enriched_result,
        source_assignment_sha256=assignment_sha256,
        metadata_digest=metadata_digest,
    )


def write_thor_recording_split_artifacts(
    build: ThorRecordingSplitBuild,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Atomically write metadata, manifest, summary, and policy audit."""
    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_name(f".{output_path.name}.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        paths = write_split_artifacts(build.result, staging)
        metadata_path = staging / "recording_metadata.jsonl"
        metadata_path.write_bytes(serialize_manifest(build.metadata))
        staging.replace(output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "metadata": output_path / "recording_metadata.jsonl",
        "manifest": output_path / paths["manifest"].name,
        "summary": output_path / paths["summary"].name,
        "overlap_report": output_path / paths["overlap_report"].name,
    }
