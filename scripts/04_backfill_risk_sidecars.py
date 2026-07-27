#!/usr/bin/env python
"""Replay one accepted SOP07 shard and adopt its authenticated SOP08 sidecar."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.contracts import SCHEMA_VERSION, build_grid_spec  # noqa: E402
from src.datasets.shard_writer import load_risk_shard  # noqa: E402
from src.datasets.sidecar_writer import (  # noqa: E402
    load_risk_sidecar_pair_completion_marker,
    load_risk_sidecar_shard,
    risk_sidecar_pair_completion_marker_path,
)
from src.datasets.split_manager import SPLIT_NAMES  # noqa: E402
from src.utils.config import load_config  # noqa: E402


BACKFILL_PRODUCER_VERSION = "sop08_accepted_sidecar_backfill_v1"
ACCEPTED_SOP07_PRODUCER_VERSION = "sop07_risk_dataset_cli_v3"
REPLAY_SOP07_PRODUCER_VERSION = "sop07_risk_dataset_cli_v4"
SOP05_TRAIN_HANDOFF_VERSION = "sop05_batch_index_handoff_v1"
SOP05_HELDOUT_HANDOFF_VERSION = "sop05_heldout_batch_complete_handoff_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHARD_ROOT = re.compile(r"shard-[0-9]{5}")


class BackfillError(ValueError):
    """Raised when an accepted shard cannot be safely replayed or adopted."""


@dataclass(frozen=True)
class BackfillTask:
    """One report/handoff-bound immutable shard replay task."""

    split: str
    shard_index: int
    relative_root: str
    sample_count: int
    event_count: int
    manifest_digest: str
    semantic_digest: str
    sop05_relative_root: str
    sop05_publication_digest: str
    trajectory_id: str


@dataclass(frozen=True)
class BackfillRequest:
    """Caller-controlled roots and frozen inputs for one array task."""

    batch_generation_report: Path
    sop05_batch_handoff: Path
    accepted_risk_root: Path
    sop03_root: Path
    sop04_root: Path
    sop04_handoff_digest: str
    config_path: Path
    paired_config_path: Path
    seed: int
    replay_risk_root: Path
    sidecar_root: Path
    task_index: int
    checksum_workers: int
    python_executable: Path
    producer_script: Path


@dataclass(frozen=True)
class _VerifiedPair:
    accepted: object
    replay: object
    sidecar: object
    marker: object
    marker_path: Path


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _strict_json_object(path: Path, *, name: str) -> dict[str, object]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise BackfillError(f"{name} not found: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise BackfillError(f"{name} must be a direct regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise BackfillError(f"{name} is not strict finite JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BackfillError(f"{name} must contain a JSON object")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise BackfillError(f"{name} must be a string-keyed mapping")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise BackfillError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise BackfillError(f"{name} must be a non-negative integer")
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise BackfillError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BackfillError(f"{name} must be a lowercase SHA-256")
    return value


def _relative_root(value: object, *, index: int, name: str) -> str:
    text = _nonempty_string(value, name=name)
    path = PurePosixPath(text)
    expected = f"shard-{index:05d}"
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or _SHARD_ROOT.fullmatch(text) is None
        or text != expected
    ):
        raise BackfillError(
            f"{name} must equal the canonical shard root {expected!r}"
        )
    return text


def _load_backfill_tasks(
    batch_generation_report: str | Path,
    sop05_batch_handoff: str | Path,
) -> tuple[BackfillTask, ...]:
    report_path = Path(batch_generation_report)
    handoff_path = Path(sop05_batch_handoff)
    if report_path.name != "batch_generation_report.json":
        raise BackfillError(
            "SOP07 authority must be named batch_generation_report.json"
        )
    if handoff_path.name != "batch_complete_handoff.json":
        raise BackfillError(
            "SOP05 authority must be named batch_complete_handoff.json"
        )
    report = _strict_json_object(
        report_path, name="SOP07 batch generation report"
    )
    handoff = _strict_json_object(
        handoff_path, name="SOP05 batch complete handoff"
    )

    if report.get("schema_version") != SCHEMA_VERSION:
        raise BackfillError("SOP07 batch report schema_version mismatch")
    if report.get("producer_version") != ACCEPTED_SOP07_PRODUCER_VERSION:
        raise BackfillError("SOP07 batch report producer_version mismatch")
    if report.get("generation_state") != "complete":
        raise BackfillError("SOP07 batch report is not complete")
    split = report.get("split")
    if split not in SPLIT_NAMES:
        raise BackfillError(f"unsupported SOP07 batch split: {split!r}")
    shard_count = _positive_int(
        report.get("shard_count"), name="SOP07 shard_count"
    )
    report_sample_count = _positive_int(
        report.get("sample_count"), name="SOP07 sample_count"
    )
    report_event_count = _positive_int(
        report.get("event_count"), name="SOP07 event_count"
    )
    raw_report_rows = report.get("shards")
    if not isinstance(raw_report_rows, list) or len(raw_report_rows) != shard_count:
        raise BackfillError("SOP07 shard list length differs from shard_count")

    report_rows: list[dict[str, object]] = []
    for position, raw_row in enumerate(raw_report_rows):
        row = _mapping(raw_row, name=f"SOP07 shard row {position}")
        index = _nonnegative_int(
            row.get("shard_index"), name="SOP07 shard_index"
        )
        if index != position:
            raise BackfillError(
                "SOP07 shard indices must be contiguous and ordered from zero"
            )
        report_rows.append(
            {
                "shard_index": index,
                "relative_root": _relative_root(
                    row.get("relative_root"),
                    index=index,
                    name="SOP07 relative_root",
                ),
                "sample_count": _positive_int(
                    row.get("sample_count"), name="SOP07 shard sample_count"
                ),
                "manifest_digest": _sha256(
                    row.get("manifest_digest"), name="SOP07 manifest digest"
                ),
                "semantic_digest": _sha256(
                    row.get("semantic_digest"), name="SOP07 semantic digest"
                ),
                "sop05_publication_digest": _sha256(
                    row.get("sop05_publication_digest"),
                    name="SOP07 source SOP05 publication digest",
                ),
                "trajectory_id": _nonempty_string(
                    row.get("trajectory_id"), name="SOP07 trajectory_id"
                ),
            }
        )
    if sum(int(row["sample_count"]) for row in report_rows) != report_sample_count:
        raise BackfillError(
            "SOP07 shard sample counts do not conserve the batch sample count"
        )

    if handoff.get("schema_version") != SCHEMA_VERSION:
        raise BackfillError("SOP05 batch handoff schema_version mismatch")
    if split == "train":
        if (
            handoff.get("artifact_role") != "sop05_train_batch_complete_index"
            or handoff.get("handoff_version") != SOP05_TRAIN_HANDOFF_VERSION
            or handoff.get("producer_version") != "sop05_generation_run_v6"
        ):
            raise BackfillError("SOP05 train batch handoff identity mismatch")
    elif (
        handoff.get("artifact_role") != "sop05_heldout_batch_complete_index"
        or handoff.get("handoff_version") != SOP05_HELDOUT_HANDOFF_VERSION
        or "producer_version" in handoff
        or "generator_algorithm_version" in handoff
    ):
        raise BackfillError("SOP05 heldout batch handoff identity mismatch")
    if handoff.get("batch_state") != "complete":
        raise BackfillError("SOP05 batch handoff is not complete")
    if handoff.get("split") != split:
        raise BackfillError("SOP05/SOP07 batch split mismatch")
    common = _mapping(
        handoff.get("common_contracts"), name="SOP05 common_contracts"
    )
    input_lock = _mapping(common.get("input_lock"), name="SOP05 input_lock")
    if input_lock.get("split") != split:
        raise BackfillError("SOP05 input_lock split mismatch")
    counts = _mapping(handoff.get("counts"), name="SOP05 counts")
    handoff_shard_count = _positive_int(
        counts.get("shards"), name="SOP05 shard count"
    )
    handoff_event_count = _positive_int(
        counts.get("events"), name="SOP05 event count"
    )
    if handoff_shard_count != shard_count:
        raise BackfillError("SOP05/SOP07 shard count mismatch")
    if handoff_event_count != report_event_count:
        raise BackfillError("SOP05/SOP07 event count mismatch")
    raw_handoff_rows = handoff.get("shards")
    if not isinstance(raw_handoff_rows, list) or len(raw_handoff_rows) != shard_count:
        raise BackfillError("SOP05 shard list length differs from shard count")

    handoff_rows: list[dict[str, object]] = []
    for position, raw_row in enumerate(raw_handoff_rows):
        row = _mapping(raw_row, name=f"SOP05 shard row {position}")
        index = _nonnegative_int(
            row.get("shard_index"), name="SOP05 shard_index"
        )
        if index != position:
            raise BackfillError(
                "SOP05 shard indices must be contiguous and ordered from zero"
            )
        handoff_rows.append(
            {
                "shard_index": index,
                "relative_root": _relative_root(
                    row.get("relative_root"),
                    index=index,
                    name="SOP05 relative_root",
                ),
                "event_count": _positive_int(
                    row.get("event_count"), name="SOP05 shard event_count"
                ),
                "publication_semantic_digest": _sha256(
                    row.get("publication_semantic_digest"),
                    name="SOP05 publication digest",
                ),
                "trajectory_id": _nonempty_string(
                    row.get("trajectory_id"), name="SOP05 trajectory_id"
                ),
            }
        )
    if sum(int(row["event_count"]) for row in handoff_rows) != handoff_event_count:
        raise BackfillError(
            "SOP05 shard event counts do not conserve the batch event count"
        )

    tasks: list[BackfillTask] = []
    for report_row, handoff_row in zip(report_rows, handoff_rows):
        if report_row["relative_root"] != handoff_row["relative_root"]:
            raise BackfillError("SOP05/SOP07 shard relative_root mismatch")
        if (
            report_row["sop05_publication_digest"]
            != handoff_row["publication_semantic_digest"]
        ):
            raise BackfillError("SOP05/SOP07 publication digest mismatch")
        if report_row["trajectory_id"] != handoff_row["trajectory_id"]:
            raise BackfillError("SOP05/SOP07 trajectory_id mismatch")
        tasks.append(
            BackfillTask(
                split=str(split),
                shard_index=int(report_row["shard_index"]),
                relative_root=str(report_row["relative_root"]),
                sample_count=int(report_row["sample_count"]),
                event_count=int(handoff_row["event_count"]),
                manifest_digest=str(report_row["manifest_digest"]),
                semantic_digest=str(report_row["semantic_digest"]),
                sop05_relative_root=str(handoff_row["relative_root"]),
                sop05_publication_digest=str(
                    handoff_row["publication_semantic_digest"]
                ),
                trajectory_id=str(report_row["trajectory_id"]),
            )
        )
    return tuple(tasks)


def load_backfill_task(
    batch_generation_report: str | Path,
    sop05_batch_handoff: str | Path,
    *,
    task_index: int,
) -> BackfillTask:
    """Load both complete batch authorities and select one ordered task."""

    tasks = _load_backfill_tasks(
        batch_generation_report,
        sop05_batch_handoff,
    )
    index = _nonnegative_int(task_index, name="task_index")
    if index >= len(tasks):
        raise BackfillError(
            f"task_index out of range: {index}; expected 0..{len(tasks) - 1}"
        )
    return tasks[index]


def resolve_task_index(
    explicit_task_index: int | None,
    *,
    environ: Mapping[str, str],
    task_count: int,
) -> int:
    """Resolve and bound an explicit or Slurm-provided zero-based array index."""

    count = _positive_int(task_count, name="task_count")
    if explicit_task_index is None:
        raw_value = environ.get("SLURM_ARRAY_TASK_ID")
        if raw_value is None or raw_value == "":
            raise BackfillError(
                "SLURM_ARRAY_TASK_ID is required when --task-index is omitted"
            )
        try:
            index = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise BackfillError("SLURM_ARRAY_TASK_ID must be an integer") from exc
    else:
        index = _nonnegative_int(explicit_task_index, name="task_index")
    if index < 0 or index >= count:
        raise BackfillError(
            f"task index out of range: {index}; expected 0..{count - 1}"
        )
    return index


def _validate_request(request: BackfillRequest) -> None:
    if not isinstance(request, BackfillRequest):
        raise TypeError("request must be a BackfillRequest")
    _nonnegative_int(request.task_index, name="task_index")
    _nonnegative_int(request.seed, name="seed")
    _positive_int(request.checksum_workers, name="checksum_workers")
    _sha256(request.sop04_handoff_digest, name="SOP04 handoff digest")
    for name in (
        "batch_generation_report",
        "sop05_batch_handoff",
        "accepted_risk_root",
        "sop03_root",
        "sop04_root",
        "config_path",
        "paired_config_path",
        "replay_risk_root",
        "sidecar_root",
        "python_executable",
        "producer_script",
    ):
        if not isinstance(getattr(request, name), Path):
            raise TypeError(f"{name} must be a Path")
    accepted = request.accepted_risk_root.resolve(strict=False)
    replay = request.replay_risk_root.resolve(strict=False)
    sidecar = request.sidecar_root.resolve(strict=False)
    if replay == sidecar or replay in sidecar.parents or sidecar in replay.parents:
        raise BackfillError("replay risk and sidecar roots must not be nested")
    for output, name in ((replay, "replay risk"), (sidecar, "sidecar")):
        if output == accepted or output in accepted.parents or accepted in output.parents:
            raise BackfillError(f"{name} root must not overlap accepted risk root")


def _require_direct_directory(path: Path, *, name: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise BackfillError(f"{name} not found: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise BackfillError(f"{name} must be a direct directory: {path}")


def _risk_identity(
    loaded: object,
    *,
    task: BackfillTask,
    name: str,
) -> tuple[str, ...]:
    samples = getattr(loaded, "samples", None)
    summary = getattr(loaded, "summary", None)
    if not isinstance(samples, tuple) or not isinstance(summary, Mapping):
        raise BackfillError(f"{name} formal load is incomplete")
    sample_ids = tuple(getattr(sample, "sample_id", None) for sample in samples)
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise BackfillError(f"{name} sample IDs are invalid")
    if len(sample_ids) != task.sample_count:
        raise BackfillError(f"{name} sample count mismatch")
    if (
        summary.get("split") != task.split
        or summary.get("shard_index") != task.shard_index
        or summary.get("expected_sample_count") != task.sample_count
    ):
        raise BackfillError(f"{name} split/index/count mismatch")
    return sample_ids


def _load_accepted(
    path: Path,
    *,
    grid: object,
    task: BackfillTask,
):
    if path.name != task.relative_root:
        raise BackfillError("accepted risk shard basename mismatch")
    _require_direct_directory(path, name="accepted risk shard")
    loaded = load_risk_shard(path, grid=grid)
    _risk_identity(loaded, task=task, name="accepted risk shard")
    if loaded.manifest_digest != task.manifest_digest:
        raise BackfillError("accepted risk shard manifest digest mismatch")
    if loaded.semantic_digest != task.semantic_digest:
        raise BackfillError("accepted risk shard semantic digest mismatch")
    return loaded


def _verify_complete_pair(
    *,
    accepted_path: Path,
    replay_path: Path,
    sidecar_path: Path,
    grid: object,
    task: BackfillTask,
) -> _VerifiedPair:
    accepted = _load_accepted(accepted_path, grid=grid, task=task)
    accepted_ids = _risk_identity(
        accepted, task=task, name="accepted risk shard"
    )
    if replay_path.name != task.relative_root:
        raise BackfillError("replay risk shard basename mismatch")
    if sidecar_path.name != task.relative_root:
        raise BackfillError("sidecar shard basename mismatch")
    _require_direct_directory(replay_path, name="replay risk shard")
    _require_direct_directory(sidecar_path, name="sidecar shard")
    replay = load_risk_shard(replay_path, grid=grid)
    replay_ids = _risk_identity(replay, task=task, name="replay risk shard")
    if replay_ids != accepted_ids:
        raise BackfillError("accepted/replay ordered sample IDs mismatch")
    if replay.manifest_digest != accepted.manifest_digest:
        raise BackfillError("accepted/replay manifest digest mismatch")
    if replay.semantic_digest != accepted.semantic_digest:
        raise BackfillError("accepted/replay semantic digest mismatch")

    sidecar = load_risk_sidecar_shard(
        sidecar_path,
        grid=grid,
        expected_sample_ids=accepted_ids,
        expected_source_risk_shard_semantic_digest=accepted.semantic_digest,
    )
    if (
        sidecar.sample_ids != accepted_ids
        or sidecar.split != task.split
        or sidecar.shard_index != task.shard_index
        or len(sidecar.sample_ids) != task.sample_count
        or sidecar.source_risk_shard_semantic_digest != accepted.semantic_digest
    ):
        raise BackfillError("sidecar accepted-risk identity mismatch")
    marker_path = risk_sidecar_pair_completion_marker_path(sidecar_path)
    marker = load_risk_sidecar_pair_completion_marker(
        marker_path,
        expected_risk_root=accepted_path,
        expected_sidecar_root=sidecar_path,
        expected_split=task.split,
        expected_shard_index=task.shard_index,
        expected_sample_ids=accepted_ids,
        expected_risk_shard_semantic_digest=accepted.semantic_digest,
        expected_sidecar_shard_semantic_digest=sidecar.semantic_digest,
    )
    final_accepted = _load_accepted(accepted_path, grid=grid, task=task)
    final_replay = load_risk_shard(replay_path, grid=grid)
    if (
        final_accepted.manifest_digest != accepted.manifest_digest
        or final_accepted.semantic_digest != accepted.semantic_digest
        or final_replay.manifest_digest != replay.manifest_digest
        or final_replay.semantic_digest != replay.semantic_digest
    ):
        raise BackfillError("risk shard identity changed during pair verification")
    return _VerifiedPair(
        accepted=accepted,
        replay=replay,
        sidecar=sidecar,
        marker=marker,
        marker_path=marker_path,
    )


def _producer_command(
    request: BackfillRequest,
    *,
    task: BackfillTask,
    replay_path: Path,
    sidecar_path: Path,
) -> tuple[str, ...]:
    return (
        str(request.python_executable),
        str(request.producer_script),
        "--sop03-root",
        str(request.sop03_root),
        "--sop04-root",
        str(request.sop04_root),
        "--sop04-handoff-digest",
        request.sop04_handoff_digest,
        "--sop05-root",
        str(request.sop05_batch_handoff.parent / task.sop05_relative_root),
        "--sop05-publication-digest",
        task.sop05_publication_digest,
        "--split",
        task.split,
        "--config",
        str(request.config_path),
        "--paired-config",
        str(request.paired_config_path),
        "--seed",
        str(request.seed),
        "--output-dir",
        str(replay_path),
        "--sidecar-output-dir",
        str(sidecar_path),
        "--shard-index",
        str(task.shard_index),
        "--expected-event-count",
        str(task.event_count),
        "--expected-sample-count",
        str(task.sample_count),
        "--checksum-workers",
        str(request.checksum_workers),
    )


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(command),
        check=False,
        capture_output=True,
        text=True,
    )


def _load_producer_report(
    result: object,
    *,
    task: BackfillTask,
    replay_path: Path,
    sidecar_path: Path,
    verified: _VerifiedPair,
) -> None:
    returncode = getattr(result, "returncode", None)
    stdout = getattr(result, "stdout", None)
    stderr = getattr(result, "stderr", None)
    if type(returncode) is not int or not isinstance(stdout, str) or not isinstance(
        stderr, str
    ):
        raise BackfillError("SOP07 producer runner returned an invalid result")
    if returncode != 0:
        detail = stderr.strip() or f"exit {returncode}"
        raise BackfillError(f"SOP07 replay producer failed: {detail}")
    if not stdout.endswith("\n") or stdout.count("\n") != 1:
        raise BackfillError("SOP07 replay producer did not emit exactly one JSON line")
    try:
        value = json.loads(
            stdout,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise BackfillError("SOP07 replay producer output is not strict JSON") from exc
    if not isinstance(value, dict) or stdout != _canonical_json(value) + "\n":
        raise BackfillError("SOP07 replay producer output is not canonical JSON")
    expected = {
        "producer_version": REPLAY_SOP07_PRODUCER_VERSION,
        "publication_status": "complete",
        "split": task.split,
        "shard_index": task.shard_index,
        "sample_count": task.sample_count,
        "output_dir": str(replay_path),
        "sidecar_output_dir": str(sidecar_path),
        "manifest_digest": verified.replay.manifest_digest,
        "semantic_digest": verified.replay.semantic_digest,
        "risk_shard_semantic_digest": verified.replay.semantic_digest,
        "sidecar_shard_semantic_digest": verified.sidecar.semantic_digest,
        "pair_completion_marker_path": str(verified.marker_path),
        "pair_completion_marker_digest": verified.marker.marker_digest_sha256,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            label = name.replace("_", " ")
            raise BackfillError(f"SOP07 replay producer {label} mismatch")


def _output_presence(
    replay_path: Path,
    sidecar_path: Path,
) -> tuple[bool, bool, bool]:
    marker_path = risk_sidecar_pair_completion_marker_path(sidecar_path)
    return tuple(
        os.path.lexists(path) for path in (replay_path, sidecar_path, marker_path)
    )  # type: ignore[return-value]


def _report(
    *,
    status: str,
    task: BackfillTask,
    accepted_path: Path,
    replay_path: Path,
    sidecar_path: Path,
    verified: _VerifiedPair,
) -> dict[str, object]:
    report = {
        "accepted_manifest_digest": verified.accepted.manifest_digest,
        "accepted_risk_root": str(accepted_path),
        "accepted_semantic_digest": verified.accepted.semantic_digest,
        "pair_completion_marker_digest": verified.marker.marker_digest_sha256,
        "pair_completion_marker_path": str(verified.marker_path),
        "producer_version": BACKFILL_PRODUCER_VERSION,
        "relative_root": task.relative_root,
        "replay_manifest_digest": verified.replay.manifest_digest,
        "replay_risk_root": str(replay_path),
        "replay_semantic_digest": verified.replay.semantic_digest,
        "sample_count": task.sample_count,
        "schema_version": SCHEMA_VERSION,
        "shard_index": task.shard_index,
        "sidecar_root": str(sidecar_path),
        "sidecar_semantic_digest": verified.sidecar.semantic_digest,
        "sop05_publication_digest": task.sop05_publication_digest,
        "sop05_relative_root": task.sop05_relative_root,
        "split": task.split,
        "status": status,
        "trajectory_id": task.trajectory_id,
    }
    _canonical_json(report)
    return report


def run_backfill(
    request: BackfillRequest,
    *,
    runner: Callable[[Sequence[str]], object] = _default_runner,
) -> dict[str, object]:
    """Replay or fully revalidate one accepted shard and its sidecar pair."""

    _validate_request(request)
    task = load_backfill_task(
        request.batch_generation_report,
        request.sop05_batch_handoff,
        task_index=request.task_index,
    )
    grid = build_grid_spec(load_config(request.config_path))
    accepted_path = request.accepted_risk_root / task.relative_root
    replay_path = request.replay_risk_root / task.relative_root
    sidecar_path = request.sidecar_root / task.relative_root
    _load_accepted(accepted_path, grid=grid, task=task)

    presence = _output_presence(replay_path, sidecar_path)
    if all(presence):
        verified = _verify_complete_pair(
            accepted_path=accepted_path,
            replay_path=replay_path,
            sidecar_path=sidecar_path,
            grid=grid,
            task=task,
        )
        return _report(
            status="already_complete",
            task=task,
            accepted_path=accepted_path,
            replay_path=replay_path,
            sidecar_path=sidecar_path,
            verified=verified,
        )
    if any(presence):
        names = ("replay risk shard", "sidecar shard", "pair marker")
        present = ", ".join(name for name, exists in zip(names, presence) if exists)
        missing = ", ".join(name for name, exists in zip(names, presence) if not exists)
        raise BackfillError(
            f"partial outputs are forbidden; present: {present}; missing: {missing}"
        )

    command = _producer_command(
        request,
        task=task,
        replay_path=replay_path,
        sidecar_path=sidecar_path,
    )
    result = runner(command)
    returncode = getattr(result, "returncode", None)
    stdout = getattr(result, "stdout", None)
    stderr = getattr(result, "stderr", None)
    if type(returncode) is not int or not isinstance(stdout, str) or not isinstance(
        stderr, str
    ):
        raise BackfillError("SOP07 producer runner returned an invalid result")
    if returncode != 0:
        detail = stderr.strip() or f"exit {returncode}"
        raise BackfillError(f"SOP07 replay producer failed: {detail}")
    generated_presence = _output_presence(replay_path, sidecar_path)
    if not all(generated_presence):
        names = ("replay risk shard", "sidecar shard", "pair marker")
        missing = ", ".join(
            name for name, exists in zip(names, generated_presence) if not exists
        )
        raise BackfillError(f"SOP07 replay producer left partial outputs; missing: {missing}")
    verified = _verify_complete_pair(
        accepted_path=accepted_path,
        replay_path=replay_path,
        sidecar_path=sidecar_path,
        grid=grid,
        task=task,
    )
    _load_producer_report(
        result,
        task=task,
        replay_path=replay_path,
        sidecar_path=sidecar_path,
        verified=verified,
    )
    return _report(
        status="complete",
        task=task,
        accepted_path=accepted_path,
        replay_path=replay_path,
        sidecar_path=sidecar_path,
        verified=verified,
    )


def _argument_int(text: str, *, minimum: int, name: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
    if value < minimum:
        raise argparse.ArgumentTypeError(f"{name} must be >= {minimum}")
    return value


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise BackfillError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description=(
            "Replay one accepted SOP07 shard, verify exact identity, and adopt "
            "its immutable SOP08 sidecar."
        )
    )
    parser.add_argument("--batch-generation-report", type=Path, required=True)
    parser.add_argument("--sop05-batch-handoff", type=Path, required=True)
    parser.add_argument("--accepted-risk-root", type=Path, required=True)
    parser.add_argument("--sop03-root", type=Path, required=True)
    parser.add_argument("--sop04-root", type=Path, required=True)
    parser.add_argument("--sop04-handoff-digest", required=True)
    parser.add_argument("--config", dest="config_path", type=Path, required=True)
    parser.add_argument(
        "--paired-config", dest="paired_config_path", type=Path, required=True
    )
    parser.add_argument(
        "--seed",
        type=lambda text: _argument_int(text, minimum=0, name="seed"),
        required=True,
    )
    parser.add_argument("--replay-risk-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument(
        "--task-index",
        type=lambda text: _argument_int(text, minimum=0, name="task_index"),
    )
    parser.add_argument(
        "--checksum-workers",
        type=lambda text: _argument_int(text, minimum=1, name="checksum_workers"),
        default=8,
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument(
        "--producer-script",
        type=Path,
        default=_ROOT / "scripts" / "04_generate_risk_dataset.py",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    runner: Callable[[Sequence[str]], object] = _default_runner,
) -> int:
    try:
        args = _parser().parse_args(argv)
        tasks = _load_backfill_tasks(
            args.batch_generation_report,
            args.sop05_batch_handoff,
        )
        task_index = resolve_task_index(
            args.task_index,
            environ=os.environ if environ is None else environ,
            task_count=len(tasks),
        )
        request = BackfillRequest(
            batch_generation_report=args.batch_generation_report,
            sop05_batch_handoff=args.sop05_batch_handoff,
            accepted_risk_root=args.accepted_risk_root,
            sop03_root=args.sop03_root,
            sop04_root=args.sop04_root,
            sop04_handoff_digest=args.sop04_handoff_digest,
            config_path=args.config_path,
            paired_config_path=args.paired_config_path,
            seed=args.seed,
            replay_risk_root=args.replay_risk_root,
            sidecar_root=args.sidecar_root,
            task_index=task_index,
            checksum_workers=args.checksum_workers,
            python_executable=args.python_executable,
            producer_script=args.producer_script,
        )
        report = run_backfill(request, runner=runner)
    except (BackfillError, OSError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        detail = " ".join(str(exc).split()) or type(exc).__name__
        print(f"error: {detail}", file=sys.stderr)
        return 2
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
