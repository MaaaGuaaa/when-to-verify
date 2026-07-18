"""Deterministic SOP-05 run orchestration over audited SOP-03/SOP-04 inputs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.contracts import SCHEMA_VERSION, GridSpec, build_grid_spec
from src.generation.dynamic_object_transplant import TargetTypePolicy
from src.generation.event_sampler import (
    EventGenerationReport,
    GeneratedEvent,
    SOP05_GENERATOR_ALGORITHM_VERSION,
    _generator_digest,
    generate_events,
    load_generator_config,
)
from src.generation.event_target_motion_shard import (
    create_event_target_motion_record,
    load_event_target_motion_shard,
    validate_event_target_motion_world_join,
    write_event_target_motion_shard,
)
from src.generation.sop05_input_adapter import (
    ProducerEvidence,
    SOP04_COMPLETION_POLICY,
    SOP04_POSE_TIME_LAYOUT_VERSION,
    SOP04_TRAJECTORY_BANK_VERSION,
    Sop03SplitInputs,
    Sop04TrajectoryBank,
    build_stable_pair_schedule,
    load_sop03_split_inputs,
    load_sop04_trajectory_bank,
)
from src.generation.sop05_publication_identity import (
    SOP05_PUBLICATION_IDENTITY_VERSION,
    compute_sop05_publication_semantic_digest,
)
from src.generation.sop05_selection import (
    SOP05_EVENT_KIND_ORDER,
    SOP05_PAIR_REPORT_VERSION,
    SOP05_RUN_PRODUCER_VERSION,
    SOP05_TOTAL_QUOTA_SELECTION_VERSION,
    select_sop05_event_ids,
)
from src.utils.config import load_config


SOP05_RUN_VERSION = SOP05_RUN_PRODUCER_VERSION
SOP05_RUN_MANIFEST_VERSION = "sop05_run_manifest_v2"
SOP05_GENERATION_SUMMARY_VERSION = "sop05_generation_summary_v2"
SOP05_INPUT_LOCK_VERSION = "sop05_input_lock_v2"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_IDENTITY_VERSION = "sop05_producer_source_identity_v1"
_EVENT_KIND_WEIGHTS = dict(
    zip(SOP05_EVENT_KIND_ORDER, (0.6, 0.3, 0.1), strict=True)
)


class Sop05RunError(ValueError):
    """Raised when a run request or generated report violates its contract."""


@dataclass(frozen=True)
class Sop05RunRequest:
    sop03_root: Path
    sop04_root: Path
    sop04_external_handoff_digest_sha256: str
    split: str
    base_config_path: Path
    generator_config_path: Path
    output_dir: Path
    seed: int
    accepted_quota: int
    events_per_pair: int
    max_base_states: int
    trajectory_count: int
    max_pairs: int
    checksum_workers: int
    workers: int
    git_executable: Path


@dataclass(frozen=True)
class RankedPair:
    rank: int
    state_id: str
    trajectory_id: str
    pair_seed: int


@dataclass(frozen=True)
class PreparedSop05Run:
    request: Sop05RunRequest
    base_config: dict[str, Any]
    generator_config: dict[str, object]
    base_config_snapshot: bytes
    generator_config_snapshot: bytes
    base_config_sha256: str
    generator_config_sha256: str
    generator_config_semantic_digest: str
    target_type_policy: dict[str, object]
    target_type_policy_digest: str
    grid: GridSpec
    sop03: Sop03SplitInputs
    sop04: Sop04TrajectoryBank
    schedule: tuple[RankedPair, ...]
    input_lock: dict[str, object]
    runtime_provenance: dict[str, object]
    producer_source_identity: dict[str, object]
    run_id: str


@dataclass(frozen=True)
class PairGenerationReport:
    rank: int
    state_id: str
    trajectory_id: str
    pair_seed: int
    report: EventGenerationReport


@dataclass(frozen=True)
class Sop05GenerationCollection:
    pair_reports: tuple[PairGenerationReport, ...]
    all_events: tuple[GeneratedEvent, ...]
    selected_events: tuple[GeneratedEvent, ...]
    generation_summary: dict[str, object]


@dataclass(frozen=True)
class Sop05RunResult:
    run_state: str
    run_id: str
    output_dir: Path
    generation_summary: dict[str, object]
    publication_semantic_digest: str | None
    exit_code: int


@dataclass(frozen=True)
class _EventTransport:
    generated_event_id: str
    event_kind: str
    world: object
    target: object
    record_payload: dict[str, object]
    expected_record_identity: tuple[str, str, str, str, str]
    visibility_sequence: object
    target_visibility_history: object
    conflict_time_s: float
    conflict_index: int


@dataclass(frozen=True)
class _PairReportTransport:
    rank: int
    state_id: str
    trajectory_id: str
    pair_seed: int
    events: tuple[_EventTransport, ...]
    summary: dict[str, object]


_PAIR_WORKER_PREPARED: PreparedSop05Run | None = None


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Sop05RunError("run evidence must be canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise Sop05RunError(f"failed to hash file: {path}") from exc


def _read_config_snapshot(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise Sop05RunError(f"failed to snapshot config: {path}") from exc


def _git_repository_layout() -> tuple[Path, Path]:
    marker = _PROJECT_ROOT / ".git"
    if marker.is_symlink():
        raise Sop05RunError("repository .git marker must not be a symlink")
    if marker.is_dir():
        git_directory = marker.resolve()
        common_directory = git_directory
    elif marker.is_file():
        try:
            line = marker.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise Sop05RunError("failed to read repository .git marker") from exc
        prefix = "gitdir: "
        if not line.startswith(prefix) or not line[len(prefix) :]:
            raise Sop05RunError("repository .git marker is invalid")
        git_directory = Path(line[len(prefix) :])
        if not git_directory.is_absolute():
            git_directory = marker.parent / git_directory
        git_directory = git_directory.resolve()
        common_marker = git_directory / "commondir"
        if common_marker.is_file():
            try:
                common_value = common_marker.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise Sop05RunError("failed to read Git commondir") from exc
            common_directory = Path(common_value)
            if not common_directory.is_absolute():
                common_directory = git_directory / common_directory
            common_directory = common_directory.resolve()
        else:
            common_directory = git_directory
    else:
        raise Sop05RunError("producer source is not inside a Git repository")
    if not git_directory.is_dir() or not common_directory.is_dir():
        raise Sop05RunError("Git repository metadata is incomplete")
    return git_directory, common_directory


def _validate_git_executable(value: object) -> Path:
    if not isinstance(value, Path):
        raise Sop05RunError("git_executable must be a Path")
    if not value.is_absolute():
        raise Sop05RunError("git_executable must be an absolute path")
    if value.is_symlink():
        raise Sop05RunError("git_executable must not be a symlink")
    if not value.is_file():
        raise Sop05RunError("git_executable must be an existing regular file")
    if not os.access(value, os.X_OK):
        raise Sop05RunError("git_executable must be executable")
    return value


def _run_git_read_only(
    git_executable: Path,
    git_directory: Path,
    common_directory: Path,
    arguments: list[str],
) -> bytes:
    git_executable = _validate_git_executable(git_executable)
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_DIR": str(common_directory),
            "GIT_WORK_TREE": str(_PROJECT_ROOT),
            "GIT_INDEX_FILE": str(git_directory / "index"),
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    try:
        completed = subprocess.run(
            [str(git_executable), *arguments],
            cwd=_PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise Sop05RunError("failed to execute git_executable") from exc
    if completed.returncode:
        raise Sop05RunError("failed to resolve producer Git identity")
    return completed.stdout


def _validate_producer_source_identity(
    value: Mapping[str, object],
) -> dict[str, object]:
    expected_keys = {
        "version",
        "git_commit",
        "worktree_state",
        "dirty_tree_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise Sop05RunError("producer source identity keys mismatch")
    if value.get("version") != _SOURCE_IDENTITY_VERSION:
        raise Sop05RunError("producer source identity version mismatch")
    commit = value.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise Sop05RunError("producer git_commit must be a lowercase SHA-1")
    state = value.get("worktree_state")
    digest = value.get("dirty_tree_sha256")
    if state not in {"clean", "dirty"}:
        raise Sop05RunError("producer worktree_state must be clean or dirty")
    if state == "clean" and digest is not None:
        raise Sop05RunError("clean source identity must not have dirty_tree_sha256")
    if state == "dirty" and (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise Sop05RunError("dirty_tree_sha256 must bind dirty source content")
    return {
        "version": _SOURCE_IDENTITY_VERSION,
        "git_commit": commit,
        "worktree_state": state,
        "dirty_tree_sha256": digest,
    }


def _load_producer_source_identity(
    git_executable: Path,
) -> dict[str, object]:
    """Resolve HEAD plus a deterministic dirty-content digest without writes."""

    git_executable = _validate_git_executable(git_executable)
    git_directory, common_directory = _git_repository_layout()
    try:
        head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise Sop05RunError("failed to read producer Git HEAD") from exc
    revision = head[5:] if head.startswith("ref: ") else head
    commit = _run_git_read_only(
        git_executable,
        git_directory,
        common_directory,
        ["rev-parse", "--verify", f"{revision}^{{commit}}"],
    ).decode("ascii").strip()
    tracked_diff = _run_git_read_only(
        git_executable,
        git_directory,
        common_directory,
        ["diff", "--binary", "--no-ext-diff", commit, "--"],
    )
    untracked_output = _run_git_read_only(
        git_executable,
        git_directory,
        common_directory,
        ["ls-files", "--others", "--exclude-standard", "-z"],
    )
    untracked_rows: list[dict[str, object]] = []
    for raw_path in sorted(item for item in untracked_output.split(b"\0") if item):
        relative_text = os.fsdecode(raw_path)
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise Sop05RunError("Git returned an unsafe untracked path")
        path = _PROJECT_ROOT / relative
        if path.is_symlink():
            row = {
                "path": relative.as_posix(),
                "kind": "symlink",
                "target": os.readlink(path),
            }
        elif path.is_file():
            row = {
                "path": relative.as_posix(),
                "kind": "file",
                "sha256": _sha256_file(path),
                "executable": bool(path.stat().st_mode & 0o111),
            }
        else:
            raise Sop05RunError("untracked source entry is not a file")
        untracked_rows.append(row)
    dirty = bool(tracked_diff or untracked_rows)
    dirty_digest = None
    if dirty:
        hasher = hashlib.sha256()
        hasher.update(b"sop05-producer-dirty-tree-v1\0")
        hasher.update(len(tracked_diff).to_bytes(8, "big"))
        hasher.update(tracked_diff)
        hasher.update(_canonical_json_bytes(untracked_rows))
        dirty_digest = hasher.hexdigest()
    return _validate_producer_source_identity(
        {
            "version": _SOURCE_IDENTITY_VERSION,
            "git_commit": commit,
            "worktree_state": "dirty" if dirty else "clean",
            "dirty_tree_sha256": dirty_digest,
        }
    )


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise Sop05RunError(f"{name} must be a positive int")
    return value


def _validate_request(request: Sop05RunRequest) -> None:
    if not isinstance(request, Sop05RunRequest):
        raise TypeError("request must be Sop05RunRequest")
    if request.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {request.output_dir}"
        )
    _validate_git_executable(request.git_executable)
    if type(request.seed) is not int or request.seed < 0:
        raise Sop05RunError("seed must be a non-negative int")
    for name in (
        "accepted_quota",
        "events_per_pair",
        "max_base_states",
        "trajectory_count",
        "max_pairs",
        "checksum_workers",
        "workers",
    ):
        _positive_int(getattr(request, name), name)
    if request.events_per_pair % 10:
        raise Sop05RunError("events_per_pair must be a multiple of 10")
    if not isinstance(request.split, str) or not request.split:
        raise Sop05RunError("split must be a non-empty string")
    digest = request.sop04_external_handoff_digest_sha256
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise Sop05RunError(
            "trusted SOP-04 external handoff digest must be 64 lowercase hex"
        )


def _evidence_payload(evidence: ProducerEvidence) -> dict[str, object]:
    if not isinstance(evidence, ProducerEvidence):
        raise Sop05RunError("upstream producer evidence has the wrong type")
    return {
        "code_commit": evidence.code_commit,
        "checksum_manifest_sha256": evidence.checksum_manifest_sha256,
        "audit_sha256": evidence.audit_sha256,
        "completion_policy": evidence.completion_policy,
    }


def _evidence_identity(evidence: ProducerEvidence) -> dict[str, str]:
    """Return content identity only; resolved paths are runtime provenance."""

    if not isinstance(evidence, ProducerEvidence):
        raise Sop05RunError("upstream producer evidence has the wrong type")
    return {
        "code_commit": evidence.code_commit,
        "checksum_manifest_sha256": evidence.checksum_manifest_sha256,
        "audit_sha256": evidence.audit_sha256,
    }


def _sop04_evidence_payload(sop04: Sop04TrajectoryBank) -> dict[str, object]:
    expected_offsets = (
        np.arange(15, dtype=np.float64) + 1.0
    ) * 0.2
    observed_offsets = np.asarray(sop04.pose_time_offsets_s, dtype=np.float64)
    if (
        sop04.trajectory_bank_version != SOP04_TRAJECTORY_BANK_VERSION
        or sop04.pose_time_layout_version != SOP04_POSE_TIME_LAYOUT_VERSION
        or observed_offsets.shape != (15,)
        or not np.array_equal(observed_offsets, expected_offsets)
    ):
        raise Sop05RunError("SOP-04 corrected future-time contract mismatch")
    digests = {
        "pose_time_offsets_sha256": sop04.pose_time_offsets_sha256,
        "bank_semantic_digest_sha256": sop04.bank_semantic_digest_sha256,
        "external_handoff_digest_sha256": (
            sop04.external_handoff_digest_sha256
        ),
    }
    for name, value in digests.items():
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise Sop05RunError(f"SOP-04 {name} is invalid")
    if sop04.producer_evidence.completion_policy != SOP04_COMPLETION_POLICY:
        raise Sop05RunError("SOP-04 completion policy mismatch")
    return {
        **_evidence_payload(sop04.producer_evidence),
        "trajectory_bank_version": SOP04_TRAJECTORY_BANK_VERSION,
        "pose_time_layout_version": SOP04_POSE_TIME_LAYOUT_VERSION,
        "trajectory_steps": 15,
        "dt_s": 0.2,
        "first_pose_time_s": 0.2,
        "last_pose_time_s": 3.0,
        **digests,
    }


def _sop04_evidence_identity(
    evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: value
        for key, value in evidence.items()
        if key != "completion_policy"
    }


def _validate_generator_mix(generator_config: Mapping[str, object]) -> None:
    observed = generator_config.get("event_type_weights")
    if observed != _EVENT_KIND_WEIGHTS:
        raise Sop05RunError("generator event_type_weights must equal 60/30/10")


def prepare_sop05_run(request: Sop05RunRequest) -> PreparedSop05Run:
    """Load and validate all run inputs without creating output files."""

    _validate_request(request)
    producer_source_identity = _validate_producer_source_identity(
        _load_producer_source_identity(request.git_executable)
    )
    base_config_snapshot = _read_config_snapshot(request.base_config_path)
    generator_config_snapshot = _read_config_snapshot(
        request.generator_config_path
    )
    base_config = load_config(request.base_config_path)
    generator_config = load_generator_config(request.generator_config_path)
    _validate_generator_mix(generator_config)
    target_type_policy = generator_config.get("target_type_policy")
    if not isinstance(target_type_policy, TargetTypePolicy):
        raise Sop05RunError("normalized target_type_policy is missing")
    target_type_policy_payload = target_type_policy.as_dict()
    target_type_policy_digest = target_type_policy.digest
    generator_config_semantic_digest = _generator_digest(generator_config)
    grid = build_grid_spec(base_config)
    sop03 = load_sop03_split_inputs(
        request.sop03_root,
        request.split,
        grid,
        checksum_workers=request.checksum_workers,
    )
    sop04 = load_sop04_trajectory_bank(
        request.sop04_root,
        grid,
        expected_external_handoff_digest_sha256=(
            request.sop04_external_handoff_digest_sha256
        ),
        checksum_workers=request.checksum_workers,
    )
    sop04_evidence = _sop04_evidence_payload(sop04)
    if (
        sop04_evidence["external_handoff_digest_sha256"]
        != request.sop04_external_handoff_digest_sha256
    ):
        raise Sop05RunError("SOP-04 trusted external handoff digest mismatch")
    raw_schedule = build_stable_pair_schedule(
        sop03,
        sop04,
        seed=request.seed,
        max_base_states=request.max_base_states,
        trajectory_count=request.trajectory_count,
    )[: request.max_pairs]
    if (
        _read_config_snapshot(request.base_config_path) != base_config_snapshot
        or _read_config_snapshot(request.generator_config_path)
        != generator_config_snapshot
    ):
        raise Sop05RunError("configuration changed during preflight")
    if _validate_producer_source_identity(
        _load_producer_source_identity(request.git_executable)
    ) != producer_source_identity:
        raise Sop05RunError("producer source changed during preflight")
    base_config_sha256 = _sha256_bytes(base_config_snapshot)
    generator_config_sha256 = _sha256_bytes(generator_config_snapshot)
    schedule = tuple(
        RankedPair(
            rank=rank,
            state_id=pair.state_id,
            trajectory_id=pair.trajectory_id,
            pair_seed=pair.seed,
        )
        for rank, pair in enumerate(raw_schedule)
    )
    pair_keys = {(item.state_id, item.trajectory_id) for item in schedule}
    if len(pair_keys) != len(schedule):
        raise Sop05RunError("stable pair schedule contains duplicate pairs")
    theoretical_capacity = len(schedule) * request.events_per_pair
    if theoretical_capacity < request.accepted_quota:
        raise Sop05RunError(
            "stable pair theoretical capacity is below accepted_quota"
        )
    schedule_rows = [
        {
            "rank": item.rank,
            "state_id": item.state_id,
            "trajectory_id": item.trajectory_id,
            "pair_seed": item.pair_seed,
        }
        for item in schedule
    ]
    input_lock = {
        "version": SOP05_INPUT_LOCK_VERSION,
        "split": request.split,
        "sop03": _evidence_payload(sop03.producer_evidence),
        "sop04": sop04_evidence,
        "selection": {
            "seed": request.seed,
            "max_base_states": request.max_base_states,
            "trajectory_count": request.trajectory_count,
            "max_pairs": request.max_pairs,
            "pair_count": len(schedule),
            "pair_schedule_sha256": _sha256_bytes(
                _canonical_json_bytes(schedule_rows)
            ),
        },
    }
    runtime_provenance = {
        "workers": request.workers,
        "checksum_workers": request.checksum_workers,
        "git_executable": str(request.git_executable),
        "resolved_input_roots": {
            "sop03": str(sop03.producer_evidence.root),
            "sop04": str(sop04.producer_evidence.root),
        },
    }
    identity_payload = {
        "version": SOP05_RUN_VERSION,
        "selection_version": SOP05_TOTAL_QUOTA_SELECTION_VERSION,
        "producer_source_identity": producer_source_identity,
        "split": request.split,
        "sop03": _evidence_identity(sop03.producer_evidence),
        "sop04": _sop04_evidence_identity(sop04_evidence),
        "selection": input_lock["selection"],
        "base_config_sha256": base_config_sha256,
        "generator_config_sha256": generator_config_sha256,
        "generator_config_semantic_digest": generator_config_semantic_digest,
        "target_type_policy": target_type_policy_payload,
        "target_type_policy_digest": target_type_policy_digest,
        "accepted_quota": request.accepted_quota,
        "events_per_pair": request.events_per_pair,
    }
    run_digest = hashlib.blake2b(
        _canonical_json_bytes(identity_payload), digest_size=16
    ).hexdigest()
    return PreparedSop05Run(
        request=request,
        base_config=base_config,
        generator_config=generator_config,
        base_config_snapshot=base_config_snapshot,
        generator_config_snapshot=generator_config_snapshot,
        base_config_sha256=base_config_sha256,
        generator_config_sha256=generator_config_sha256,
        generator_config_semantic_digest=generator_config_semantic_digest,
        target_type_policy=target_type_policy_payload,
        target_type_policy_digest=target_type_policy_digest,
        grid=grid,
        sop03=sop03,
        sop04=sop04,
        schedule=schedule,
        input_lock=input_lock,
        runtime_provenance=runtime_provenance,
        producer_source_identity=producer_source_identity,
        run_id=f"sop05-run-{run_digest}",
    )


def preflight_summary(prepared: PreparedSop05Run) -> dict[str, object]:
    """Return a JSON-safe summary without mutating the prepared run."""

    if not isinstance(prepared, PreparedSop05Run):
        raise TypeError("prepared must be PreparedSop05Run")
    return {
        "status": "preflight_ok",
        "run_id": prepared.run_id,
        "producer_source_identity": prepared.producer_source_identity,
        "output_dir": str(prepared.request.output_dir),
        "pair_count": len(prepared.schedule),
        "theoretical_capacity": (
            len(prepared.schedule) * prepared.request.events_per_pair
        ),
        "accepted_quota": prepared.request.accepted_quota,
        "selection_version": SOP05_TOTAL_QUOTA_SELECTION_VERSION,
    }


def _generate_pair(
    prepared: PreparedSop05Run, pair: RankedPair
) -> PairGenerationReport:
    base_state, oracle_context = prepared.sop03.load_pair(
        pair.state_id, prepared.grid
    )
    try:
        trajectory = prepared.sop04.by_id[pair.trajectory_id]
    except KeyError as exc:
        raise Sop05RunError(
            f"unknown scheduled trajectory_id: {pair.trajectory_id}"
        ) from exc
    report = generate_events(
        base_state=base_state,
        oracle_context=oracle_context,
        trajectory=trajectory,
        snippet_libraries=prepared.sop03.typed_libraries,
        base_config=prepared.base_config,
        generator_config=prepared.generator_config,
        seed=pair.pair_seed,
        event_count=prepared.request.events_per_pair,
    )
    return PairGenerationReport(
        rank=pair.rank,
        state_id=pair.state_id,
        trajectory_id=pair.trajectory_id,
        pair_seed=pair.pair_seed,
        report=report,
    )


def _canonical_json_copy(value: object) -> object:
    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


def _transport_pair_report(item: PairGenerationReport) -> _PairReportTransport:
    """Remove the deliberately unpicklable frozen record mapping at IPC."""

    events: list[_EventTransport] = []
    for event in item.report.events:
        record = event.target_motion_record
        footprint_spec = _canonical_json_copy(record.footprint_spec)
        if not isinstance(footprint_spec, dict):
            raise Sop05RunError("record footprint_spec must be an object")
        events.append(
            _EventTransport(
                generated_event_id=event.generated_event_id,
                event_kind=event.event_kind,
                world=event.world,
                target=event.target,
                record_payload={
                    "generated_event_id": record.generated_event_id,
                    "world_id": record.world_id,
                    "base_state_id": record.base_state_id,
                    "trajectory_id": record.trajectory_id,
                    "target_dynamic_object_id": (
                        record.target_dynamic_object_id
                    ),
                    "source_snippet_id": record.source_snippet_id,
                    "source_object_id": record.source_object_id,
                    "object_type": record.object_type,
                    "footprint_spec": footprint_spec,
                    "footprint_spec_digest": record.footprint_spec_digest,
                    "target_type_policy_digest": (
                        record.target_type_policy_digest
                    ),
                    "history_poses": record.history_poses,
                    "current_pose": record.current_pose,
                    "future_poses": record.future_poses,
                },
                expected_record_identity=(
                    record.schema_version,
                    record.layout_version,
                    record.history_array_digest,
                    record.future_array_digest,
                    record.record_digest,
                ),
                visibility_sequence=event.visibility_sequence,
                target_visibility_history=event.target_visibility_history,
                conflict_time_s=event.conflict_time_s,
                conflict_index=event.conflict_index,
            )
        )
    return _PairReportTransport(
        rank=item.rank,
        state_id=item.state_id,
        trajectory_id=item.trajectory_id,
        pair_seed=item.pair_seed,
        events=tuple(events),
        summary=dict(item.report.summary),
    )


def _restore_pair_report(transport: _PairReportTransport) -> PairGenerationReport:
    if not isinstance(transport, _PairReportTransport):
        raise Sop05RunError("pair worker returned the wrong transport type")
    events: list[GeneratedEvent] = []
    for transported_event in transport.events:
        record = create_event_target_motion_record(
            **transported_event.record_payload
        )
        observed_identity = (
            record.schema_version,
            record.layout_version,
            record.history_array_digest,
            record.future_array_digest,
            record.record_digest,
        )
        if observed_identity != transported_event.expected_record_identity:
            raise Sop05RunError("record identity changed across process transport")
        events.append(
            GeneratedEvent(
                generated_event_id=transported_event.generated_event_id,
                event_kind=transported_event.event_kind,
                world=transported_event.world,
                target=transported_event.target,
                target_motion_record=record,
                visibility_sequence=transported_event.visibility_sequence,
                target_visibility_history=(
                    transported_event.target_visibility_history
                ),
                conflict_time_s=transported_event.conflict_time_s,
                conflict_index=transported_event.conflict_index,
            )
        )
    return PairGenerationReport(
        rank=transport.rank,
        state_id=transport.state_id,
        trajectory_id=transport.trajectory_id,
        pair_seed=transport.pair_seed,
        report=EventGenerationReport(
            events=tuple(events), summary=dict(transport.summary)
        ),
    )


def _summary_int(summary: Mapping[str, object], name: str) -> int:
    value = summary.get(name)
    if type(value) is not int or value < 0:
        raise Sop05RunError(f"report {name} must be a non-negative int")
    return value


def _summary_count_map(
    summary: Mapping[str, object], name: str
) -> dict[str, int]:
    value = summary.get(name)
    if not isinstance(value, Mapping):
        raise Sop05RunError(f"report {name} must be a mapping")
    result: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or type(count) is not int or count < 0:
            raise Sop05RunError(f"report {name} contains an invalid count")
        result[key] = count
    return result


def _summary_rate(summary: Mapping[str, object], name: str) -> float:
    value = summary.get(name)
    if type(value) is not float or not math.isfinite(value):
        raise Sop05RunError(f"report {name} must be a finite float")
    return value


def _summary_digest(summary: Mapping[str, object], name: str) -> str:
    value = summary.get(name)
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Sop05RunError(f"report {name} must be a lowercase digest")
    return value


def _validate_pair_report(
    prepared: PreparedSop05Run, item: PairGenerationReport
) -> None:
    report = item.report
    if not isinstance(report, EventGenerationReport):
        raise Sop05RunError("generate_events returned the wrong report type")
    summary = report.summary
    if not isinstance(summary, Mapping):
        raise Sop05RunError("report summary must be a mapping")
    if summary.get("seed") != item.pair_seed:
        raise Sop05RunError("report seed mismatch")
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise Sop05RunError("report schema_version mismatch")
    requested = _summary_int(summary, "requested_event_count")
    if requested != prepared.request.events_per_pair:
        raise Sop05RunError("report requested_event_count mismatch")
    accepted = _summary_int(summary, "accepted_count")
    if accepted != len(report.events):
        raise Sop05RunError("report accepted_count mismatch")
    if accepted > requested:
        raise Sop05RunError("report accepted_count exceeds requested count")
    attempted = _summary_int(summary, "attempted_count")
    for alias in (
        "complete_joint_candidates_attempted",
        "joint_candidate_attempted_count",
    ):
        if _summary_int(summary, alias) != attempted:
            raise Sop05RunError("report attempted_count aliases mismatch")
    rejected = _summary_int(summary, "rejected_count")
    if rejected != attempted - accepted:
        raise Sop05RunError("report rejected_count mismatch")
    if _summary_int(summary, "unaccepted_event_count") != requested - accepted:
        raise Sop05RunError("report unaccepted_event_count mismatch")
    expected_attempt_rate = accepted / attempted if attempted else 0.0
    expected_request_rate = accepted / requested
    if _summary_rate(summary, "attempt_acceptance_rate") != expected_attempt_rate:
        raise Sop05RunError("report attempt_acceptance_rate mismatch")
    for name in ("acceptance_rate", "request_acceptance_rate"):
        if _summary_rate(summary, name) != expected_request_rate:
            raise Sop05RunError(f"report {name} mismatch")
    rejection_reasons = _summary_count_map(summary, "rejection_reasons")
    if sum(rejection_reasons.values()) != rejected:
        raise Sop05RunError("report rejection_reasons total mismatch")
    rejection_stages = _summary_count_map(summary, "rejection_stage_counts")
    expected_stage_names = {
        "occluder_geometry",
        "target_conditioning",
        "visibility",
    }
    if set(rejection_stages) != expected_stage_names:
        raise Sop05RunError("report rejection_stage_counts keys mismatch")
    if sum(rejection_stages.values()) != rejected:
        raise Sop05RunError("report rejection_stage_counts total mismatch")
    _summary_count_map(summary, "occluder_candidate_rejection_reasons")
    report_policy_digest = _summary_digest(
        summary, "target_type_policy_digest"
    )
    if report_policy_digest != prepared.target_type_policy_digest:
        raise Sop05RunError(
            "report target_type_policy_digest does not match prepared config"
        )
    if summary.get("target_type_policy") != prepared.target_type_policy:
        raise Sop05RunError(
            "report target_type_policy does not match prepared config"
        )
    report_generator_digest = _summary_digest(
        summary, "generator_config_digest"
    )
    if report_generator_digest != prepared.generator_config_semantic_digest:
        raise Sop05RunError(
            "report generator_config_digest does not match prepared config"
        )
    generator_version = summary.get("generator_algorithm_version")
    if generator_version != SOP05_GENERATOR_ALGORITHM_VERSION:
        raise Sop05RunError(
            "report generator_algorithm_version does not match the frozen "
            f"{SOP05_GENERATOR_ALGORITHM_VERSION!r} contract"
        )
    expected_requested = {
        kind: prepared.request.events_per_pair * numerator // 10
        for kind, numerator in (
            ("environment", 6),
            ("structural", 3),
            ("mixed", 1),
        )
    }
    if _summary_count_map(
        summary, "requested_event_kind_counts"
    ) != expected_requested:
        raise Sop05RunError("report requested_event_kind_counts mismatch")
    observed_kind_counts = Counter(event.event_kind for event in report.events)
    expected_observed = {
        kind: observed_kind_counts[kind] for kind in _EVENT_KIND_WEIGHTS
    }
    if _summary_count_map(summary, "event_kind_counts") != expected_observed:
        raise Sop05RunError("report event_kind_counts mismatch")
    for event in report.events:
        if not isinstance(event, GeneratedEvent):
            raise Sop05RunError("report events must contain GeneratedEvent")
        if event.event_kind not in _EVENT_KIND_WEIGHTS:
            raise Sop05RunError("report event_kind is invalid")
        record = event.target_motion_record
        if record.target_type_policy_digest != prepared.target_type_policy_digest:
            raise Sop05RunError(
                "record target_type_policy_digest does not match prepared config"
            )
        if event.generated_event_id != record.generated_event_id:
            raise Sop05RunError("event/record generated_event_id mismatch")
        if record.base_state_id != item.state_id:
            raise Sop05RunError("event base_state_id does not match pair")
        if record.trajectory_id != item.trajectory_id:
            raise Sop05RunError("event trajectory_id does not match pair")
        target = event.target
        if (
            target.target_dynamic_object_id != record.target_dynamic_object_id
            or target.source_object_id != record.source_object_id
            or target.snippet_id != record.source_snippet_id
            or target.object_type != record.object_type
            or target.footprint_spec != record.footprint_spec
            or target.footprint_spec_digest != record.footprint_spec_digest
            or not np.array_equal(target.history_poses, record.history_poses)
            or not np.array_equal(target.current_pose, record.current_pose)
            or not np.array_equal(target.future_poses, record.future_poses)
        ):
            raise Sop05RunError("event target does not match target-motion record")
        validate_event_target_motion_world_join(record, event.world, prepared.grid)


def _initialize_pair_worker(prepared: PreparedSop05Run) -> None:
    if not isinstance(prepared, PreparedSop05Run):
        raise TypeError("prepared must be PreparedSop05Run")
    global _PAIR_WORKER_PREPARED
    _PAIR_WORKER_PREPARED = prepared


def _generate_pair_in_worker(pair: RankedPair) -> _PairReportTransport:
    prepared = _PAIR_WORKER_PREPARED
    if prepared is None:
        raise RuntimeError("pair process worker was not initialized")
    item = _generate_pair(prepared, pair)
    _validate_pair_report(prepared, item)
    return _transport_pair_report(item)


def _make_pair_process_pool(prepared: PreparedSop05Run) -> ProcessPoolExecutor:
    """Build a true CPU process pool with forked read-only audited inputs."""

    return ProcessPoolExecutor(
        max_workers=prepared.request.workers,
        mp_context=get_context("fork"),
        initializer=_initialize_pair_worker,
        initargs=(prepared,),
    )


def _build_generation_summary(
    prepared: PreparedSop05Run,
    pair_reports: tuple[PairGenerationReport, ...] | list[PairGenerationReport],
    all_events: tuple[GeneratedEvent, ...] | list[GeneratedEvent],
    selected_events: tuple[GeneratedEvent, ...] | list[GeneratedEvent],
) -> dict[str, object]:
    """Recompute every published aggregate from validated pair reports."""

    invariant_values: dict[str, object] | None = None
    rejection_reasons: dict[str, int] = {}
    rejection_stages: dict[str, int] = {}
    attempted_count = 0
    requested_count = 0
    for item in pair_reports:
        summary = item.report.summary
        invariants = {
            name: summary.get(name)
            for name in (
                "schema_version",
                "target_type_policy_digest",
                "generator_config_digest",
                "generator_algorithm_version",
            )
        }
        if invariant_values is None:
            invariant_values = invariants
        elif invariants != invariant_values:
            raise Sop05RunError("generator report invariants differ")
        requested_count += _summary_int(summary, "requested_event_count")
        attempted_count += _summary_int(summary, "attempted_count")
        for reason, count in _summary_count_map(
            summary, "rejection_reasons"
        ).items():
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + count
        for stage, count in _summary_count_map(
            summary, "rejection_stage_counts"
        ).items():
            rejection_stages[stage] = rejection_stages.get(stage, 0) + count

    generated_counts = Counter(event.event_kind for event in all_events)
    selected_counts = Counter(event.event_kind for event in selected_events)
    generated_kind_counts = {
        kind: generated_counts[kind] for kind in _EVENT_KIND_WEIGHTS
    }
    selected_kind_counts = {
        kind: selected_counts[kind] for kind in _EVENT_KIND_WEIGHTS
    }
    return {
        "processed_pair_count": len(pair_reports),
        "requested_event_count": requested_count,
        "attempted_count": attempted_count,
        "generator_accepted_count": len(all_events),
        "selected_count": len(selected_events),
        "quota_trimmed_count": len(all_events) - len(selected_events),
        "generated_event_kind_counts": generated_kind_counts,
        "selected_event_kind_counts": selected_kind_counts,
        "quota_met": len(selected_events) == prepared.request.accepted_quota,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "rejection_stage_counts": dict(sorted(rejection_stages.items())),
        "generator_invariants": invariant_values or {},
    }


def collect_sop05_generation(
    prepared: PreparedSop05Run,
) -> Sop05GenerationCollection:
    """Generate the full schedule and select one deterministic total quota."""

    if not isinstance(prepared, PreparedSop05Run):
        raise TypeError("prepared must be PreparedSop05Run")
    pair_reports: list[PairGenerationReport] = []
    all_events: list[GeneratedEvent] = []
    seen_event_ids: set[str] = set()
    seen_world_ids: set[str] = set()

    executor_context = (
        _make_pair_process_pool(prepared)
        if prepared.request.workers > 1
        else nullcontext(None)
    )
    with executor_context as executor:
        if executor is None:
            completed = (
                _generate_pair(prepared, pair) for pair in prepared.schedule
            )
        else:
            completed = (
                _restore_pair_report(transport)
                for transport in executor.map(
                    _generate_pair_in_worker, prepared.schedule
                )
        )
        for item in completed:
            _validate_pair_report(prepared, item)
            pair_reports.append(item)
            for event in item.report.events:
                if event.generated_event_id in seen_event_ids:
                    raise Sop05RunError("duplicate generated_event_id")
                if event.world.world_id in seen_world_ids:
                    raise Sop05RunError("duplicate world_id")
                seen_event_ids.add(event.generated_event_id)
                seen_world_ids.add(event.world.world_id)
                all_events.append(event)

    selected_ids = select_sop05_event_ids(
        (
            (event.generated_event_id, event.event_kind)
            for event in all_events
        ),
        seed=prepared.request.seed,
        accepted_quota=prepared.request.accepted_quota,
    )
    events_by_id = {
        event.generated_event_id: event for event in all_events
    }
    selected = [events_by_id[event_id] for event_id in selected_ids]
    summary = _build_generation_summary(
        prepared, pair_reports, all_events, selected
    )
    return Sop05GenerationCollection(
        pair_reports=tuple(pair_reports),
        all_events=tuple(all_events),
        selected_events=tuple(selected),
        generation_summary=summary,
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _pair_report_rows(
    collection: Sop05GenerationCollection,
) -> list[dict[str, object]]:
    return [
        {
            "report_version": SOP05_PAIR_REPORT_VERSION,
            "selection_version": SOP05_TOTAL_QUOTA_SELECTION_VERSION,
            "rank": item.rank,
            "state_id": item.state_id,
            "trajectory_id": item.trajectory_id,
            "seed": item.pair_seed,
            "summary": item.report.summary,
            "accepted_events": [
                {
                    "generated_event_id": event.generated_event_id,
                    "event_kind": event.event_kind,
                }
                for event in item.report.events
            ],
        }
        for item in collection.pair_reports
    ]


def _pair_report_bytes(collection: Sop05GenerationCollection) -> bytes:
    return b"".join(
        _canonical_json_bytes(row) + b"\n"
        for row in _pair_report_rows(collection)
    )


def _checksum_manifest_bytes(root: Path) -> bytes:
    rows: list[bytes] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise Sop05RunError(f"staged artifact must not be a symlink: {relative}")
        if not path.is_file() or relative in {
            "checksums.sha256",
            ".producer-complete",
        }:
            continue
        rows.append(f"{_sha256_file(path)}  {relative}\n".encode("utf-8"))
    return b"".join(rows)


def _completion_marker(
    prepared: PreparedSop05Run,
    staging: Path,
    loaded_shard: object,
) -> dict[str, object]:
    identity_fields = {
        "run_id": prepared.run_id,
        "run_manifest_sha256": _sha256_file(staging / "run_manifest.json"),
        "checksums_sha256": _sha256_file(staging / "checksums.sha256"),
        "target_motion_manifest_digest": loaded_shard.manifest_digest,
        "target_motion_payload_semantic_digest": (
            loaded_shard.payload_semantic_digest
        ),
    }
    return {
        "marker_version": "sop05_producer_complete_v2",
        "publication_identity_version": SOP05_PUBLICATION_IDENTITY_VERSION,
        **identity_fields,
        "publication_semantic_digest": (
            compute_sop05_publication_semantic_digest(**identity_fields)
        ),
    }


def _validate_collection_for_publication(
    prepared: PreparedSop05Run,
    collection: Sop05GenerationCollection,
) -> bool:
    if not isinstance(collection, Sop05GenerationCollection):
        raise TypeError("collection must be Sop05GenerationCollection")
    expected_pairs = [
        (pair.rank, pair.state_id, pair.trajectory_id, pair.pair_seed)
        for pair in prepared.schedule
    ]
    observed_pairs = [
        (item.rank, item.state_id, item.trajectory_id, item.pair_seed)
        for item in collection.pair_reports
    ]
    if observed_pairs != expected_pairs:
        raise Sop05RunError("collection does not cover the fixed pair schedule")
    for item in collection.pair_reports:
        _validate_pair_report(prepared, item)
    all_event_ids = [event.generated_event_id for event in collection.all_events]
    selected_event_ids = [
        event.generated_event_id for event in collection.selected_events
    ]
    reported_event_objects = [
        event
        for item in collection.pair_reports
        for event in item.report.events
    ]
    reported_events = [
        (event.generated_event_id, event.event_kind)
        for event in reported_event_objects
    ]
    observed_events = [
        (event.generated_event_id, event.event_kind)
        for event in collection.all_events
    ]
    if reported_events != observed_events:
        raise Sop05RunError(
            "collection generated events differ from pair reports"
        )
    if any(
        reported is not observed
        for reported, observed in zip(
            reported_event_objects, collection.all_events, strict=True
        )
    ):
        raise Sop05RunError(
            "collection generated event payload differs from pair reports"
        )
    if len(all_event_ids) != len(set(all_event_ids)):
        raise Sop05RunError("collection contains duplicate generated_event_id")
    if len(selected_event_ids) != len(set(selected_event_ids)):
        raise Sop05RunError("selection contains duplicate generated_event_id")
    if not set(selected_event_ids).issubset(all_event_ids):
        raise Sop05RunError("selection is not a subset of generated events")
    all_events_by_id = {
        event.generated_event_id: event for event in collection.all_events
    }
    if any(
        selected is not all_events_by_id[selected.generated_event_id]
        for selected in collection.selected_events
    ):
        raise Sop05RunError(
            "selected event payload differs from pair reports"
        )
    expected_selected_event_ids = select_sop05_event_ids(
        observed_events,
        seed=prepared.request.seed,
        accepted_quota=prepared.request.accepted_quota,
    )
    if tuple(selected_event_ids) != expected_selected_event_ids:
        raise Sop05RunError(
            "selection differs from the frozen total-quota selection"
        )
    for event in collection.selected_events:
        validate_event_target_motion_world_join(
            event.target_motion_record, event.world, prepared.grid
        )
    summary = collection.generation_summary
    if not isinstance(summary, Mapping):
        raise Sop05RunError("generation summary must be a mapping")
    expected_summary = _build_generation_summary(
        prepared,
        list(collection.pair_reports),
        list(collection.all_events),
        list(collection.selected_events),
    )
    if set(summary) != set(expected_summary):
        raise Sop05RunError("generation summary schema mismatch")
    for field_name, expected_value in expected_summary.items():
        if summary[field_name] != expected_value:
            raise Sop05RunError(
                f"generation summary {field_name} mismatch"
            )
    return bool(expected_summary["quota_met"])


def _run_manifest(
    prepared: PreparedSop05Run,
    *,
    run_state: str,
    shard_directory_name: str | None,
) -> dict[str, object]:
    schedule = [
        {
            "rank": pair.rank,
            "state_id": pair.state_id,
            "trajectory_id": pair.trajectory_id,
            "pair_seed": pair.pair_seed,
        }
        for pair in prepared.schedule
    ]
    return {
        "manifest_version": SOP05_RUN_MANIFEST_VERSION,
        "producer_version": SOP05_RUN_VERSION,
        "producer_source_identity": prepared.producer_source_identity,
        "run_id": prepared.run_id,
        "run_state": run_state,
        "split": prepared.request.split,
        "input_lock": prepared.input_lock,
        "scientific_request": {
            "seed": prepared.request.seed,
            "selection_version": SOP05_TOTAL_QUOTA_SELECTION_VERSION,
            "accepted_quota": prepared.request.accepted_quota,
            "events_per_pair": prepared.request.events_per_pair,
            "max_base_states": prepared.request.max_base_states,
            "trajectory_count": prepared.request.trajectory_count,
            "max_pairs": prepared.request.max_pairs,
            "base_config_sha256": prepared.base_config_sha256,
            "generator_config_sha256": prepared.generator_config_sha256,
            "generator_config_semantic_digest": (
                prepared.generator_config_semantic_digest
            ),
            "target_type_policy": prepared.target_type_policy,
            "target_type_policy_digest": prepared.target_type_policy_digest,
            "pair_schedule": schedule,
        },
        "runtime": prepared.runtime_provenance,
        "artifacts": {
            "base_config_snapshot": "configs/base.yaml",
            "generator_config_snapshot": "configs/generator.yaml",
            "generation_summary": "generation_summary.json",
            "pair_generation_reports": "pair_generation_reports.jsonl",
            "checksums": "checksums.sha256",
            "target_motion_shard": shard_directory_name,
            "producer_complete": (
                ".producer-complete" if run_state == "complete" else None
            ),
        },
    }


def _validate_staged_publication(
    staging: Path,
    prepared: PreparedSop05Run,
    collection: Sop05GenerationCollection,
    *,
    run_state: str,
    shard_directory_name: str | None,
    manifest: Mapping[str, object],
    generation_summary: Mapping[str, object],
) -> None:
    expected_entries = {
        "checksums.sha256",
        "configs",
        "generation_summary.json",
        "pair_generation_reports.jsonl",
        "run_manifest.json",
    }
    if shard_directory_name is not None:
        expected_entries.add(shard_directory_name)
    if run_state == "complete":
        expected_entries.add(".producer-complete")
    if {path.name for path in staging.iterdir()} != expected_entries:
        raise Sop05RunError("staged publication root file set mismatch")
    loaded_manifest = json.loads(
        (staging / "run_manifest.json").read_text(encoding="utf-8")
    )
    loaded_summary = json.loads(
        (staging / "generation_summary.json").read_text(encoding="utf-8")
    )
    if loaded_manifest != manifest or loaded_summary != generation_summary:
        raise Sop05RunError("staged publication JSON round-trip mismatch")
    base_snapshot = staging / "configs/base.yaml"
    generator_snapshot = staging / "configs/generator.yaml"
    if (
        base_snapshot.read_bytes() != prepared.base_config_snapshot
        or generator_snapshot.read_bytes()
        != prepared.generator_config_snapshot
        or _sha256_file(base_snapshot) != prepared.base_config_sha256
        or _sha256_file(generator_snapshot)
        != prepared.generator_config_sha256
    ):
        raise Sop05RunError("staged config snapshot mismatch")
    if (
        staging / "pair_generation_reports.jsonl"
    ).read_bytes() != _pair_report_bytes(collection):
        raise Sop05RunError("staged pair generation reports mismatch")
    expected_checksums = _checksum_manifest_bytes(staging)
    if (staging / "checksums.sha256").read_bytes() != expected_checksums:
        raise Sop05RunError("staged checksum manifest mismatch")
    loaded_shard = None
    if shard_directory_name is not None:
        loaded_shard = load_event_target_motion_shard(
            staging / shard_directory_name,
            grid=prepared.grid,
            expected_generated_event_ids={
                event.generated_event_id
                for event in collection.selected_events
            },
            expected_base_state_ids={
                event.target_motion_record.base_state_id
                for event in collection.selected_events
            },
            expected_trajectory_ids={
                event.target_motion_record.trajectory_id
                for event in collection.selected_events
            },
        )
    if run_state == "complete":
        marker = staging / ".producer-complete"
        if marker.is_symlink() or not marker.is_file() or not marker.stat().st_size:
            raise Sop05RunError("invalid staged producer-complete marker")
        if loaded_shard is None:
            raise Sop05RunError("complete publication is missing target shard")
        loaded_marker = json.loads(marker.read_text(encoding="utf-8"))
        expected_marker = _completion_marker(
            prepared, staging, loaded_shard
        )
        if loaded_marker != expected_marker:
            raise Sop05RunError("producer-complete marker digest mismatch")
    elif (staging / ".producer-complete").exists():
        raise Sop05RunError("quota-unmet publication must not have a marker")


def publish_sop05_generation(
    prepared: PreparedSop05Run,
    collection: Sop05GenerationCollection,
) -> Sop05RunResult:
    """Validate, stage, and atomically publish a complete or partial run."""

    if not isinstance(prepared, PreparedSop05Run):
        raise TypeError("prepared must be PreparedSop05Run")
    quota_met = _validate_collection_for_publication(prepared, collection)
    output_dir = prepared.request.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    current_source_identity = _validate_producer_source_identity(
        _load_producer_source_identity(prepared.request.git_executable)
    )
    if current_source_identity != prepared.producer_source_identity:
        raise Sop05RunError("producer source changed after generation")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
        )
    )
    run_state = "complete" if quota_met else "quota_unmet"
    shard_directory_name = (
        "target_motions"
        if quota_met
        else ("partial_target_motions" if collection.selected_events else None)
    )
    manifest = _run_manifest(
        prepared,
        run_state=run_state,
        shard_directory_name=shard_directory_name,
    )
    generation_summary = {
        "summary_version": SOP05_GENERATION_SUMMARY_VERSION,
        "run_id": prepared.run_id,
        "run_state": run_state,
        **collection.generation_summary,
    }
    publication_semantic_digest: str | None = None
    try:
        loaded_shard = None
        if shard_directory_name is not None:
            write_event_target_motion_shard(
                [
                    event.target_motion_record
                    for event in collection.selected_events
                ],
                [event.world for event in collection.selected_events],
                staging / shard_directory_name,
                grid=prepared.grid,
            )
            loaded_shard = load_event_target_motion_shard(
                staging / shard_directory_name,
                grid=prepared.grid,
                expected_generated_event_ids={
                    event.generated_event_id
                    for event in collection.selected_events
                },
            )
        config_directory = staging / "configs"
        config_directory.mkdir()
        (config_directory / "base.yaml").write_bytes(
            prepared.base_config_snapshot
        )
        (config_directory / "generator.yaml").write_bytes(
            prepared.generator_config_snapshot
        )
        (staging / "pair_generation_reports.jsonl").write_bytes(
            _pair_report_bytes(collection)
        )
        _write_json(staging / "generation_summary.json", generation_summary)
        _write_json(staging / "run_manifest.json", manifest)
        (staging / "checksums.sha256").write_bytes(
            _checksum_manifest_bytes(staging)
        )
        if quota_met:
            if loaded_shard is None:
                raise Sop05RunError("complete run is missing target shard")
            marker_payload = _completion_marker(
                prepared, staging, loaded_shard
            )
            publication_semantic_digest = str(
                marker_payload["publication_semantic_digest"]
            )
            _write_json(
                staging / ".producer-complete",
                marker_payload,
            )
        _validate_staged_publication(
            staging,
            prepared,
            collection,
            run_state=run_state,
            shard_directory_name=shard_directory_name,
            manifest=manifest,
            generation_summary=generation_summary,
        )
        if quota_met:
            if publication_semantic_digest is None:
                raise Sop05RunError(
                    "complete publication is missing its semantic digest"
                )
            from src.generation.sop05_output_loader import (
                load_complete_sop05_events,
            )

            loaded_publication = load_complete_sop05_events(
                staging,
                grid=prepared.grid,
                expected_publication_semantic_digest=(
                    publication_semantic_digest
                ),
                expected_run_id=prepared.run_id,
            )
            if len(loaded_publication.events) != len(
                collection.selected_events
            ):
                raise Sop05RunError(
                    "consumer round-trip selected event count mismatch"
                )
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite existing output: {output_dir}"
            )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return Sop05RunResult(
        run_state=run_state,
        run_id=prepared.run_id,
        output_dir=output_dir,
        generation_summary=generation_summary,
        publication_semantic_digest=publication_semantic_digest,
        exit_code=0 if quota_met else 4,
    )


def execute_sop05_run(request: Sop05RunRequest) -> Sop05RunResult:
    """Prepare once, execute the fixed schedule, and atomically publish."""

    prepared = prepare_sop05_run(request)
    collection = collect_sop05_generation(prepared)
    return publish_sop05_generation(prepared, collection)
