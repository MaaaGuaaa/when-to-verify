"""Deterministic SOP05R orchestration, selection, and atomic publication."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Executor, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TypeVar

import numpy as np
import yaml

from src.contracts import GridSpec, build_grid_spec
from src.datasets.split_manager import SPLIT_NAMES
from src.generation.event_sampler import GeneratedEvent
from src.generation.event_target_motion_shard import (
    create_event_target_motion_record,
    load_event_target_motion_shard,
    write_event_target_motion_shard,
)
from src.generation.history_visibility import HISTORY_VISIBILITY_REGIMES
from src.planning.obstacle_corner_planner import GeometryCachingObstaclePlanner
from src.planning.verification_actions import (
    VerificationActionLibrary,
    load_verification_actions,
)
from src.utils.atomic_publish import atomic_rename_noreplace
from src.utils.config import load_config
from src.utils.seeding import derive_seed

from .obstacle_first_templates import iter_obstacle_target_templates
from .sop05_input_adapter import ProducerEvidence, Sop03SplitInputs, load_sop03_split_inputs
from .sop05_run import _load_producer_source_identity
from .sop05r_contracts import (
    SOP05R_COMPLETION_MARKER_VERSION,
    SOP05R_GENERATOR_VERSION,
    SOP05R_MANIFEST_VERSION,
    SOP05R_REPORT_VERSION,
    SOP05R_RUN_VERSION,
    SOP05R_SELECTION_VERSION,
    SOP05R_SUMMARY_VERSION,
    Sop05rConfig,
    load_sop05r_config,
)
from .sop05r_event_sampler import evaluate_obstacle_first_template
from .sop05r_revealability import (
    build_active_revealability_request,
    evaluate_active_revealability,
)
from .sop05r_trajectory_store import (
    Sop05rTrajectoryRecord,
    Sop05rTrajectoryStore,
    load_sop05r_trajectory_store,
    publish_sop05r_trajectory_store,
)


SOP05R_INPUT_LOCK_VERSION = "sop05r_input_lock_v1"
SOP05R_EMPTY_TARGET_MOTION_VERSION = "sop05r_empty_target_motion_v1"
SOP05R_EMPTY_TRAJECTORY_STORE_VERSION = "sop05r_empty_trajectory_store_v1"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_T = TypeVar("_T")


class Sop05rRunError(ValueError):
    """Raised when a SOP05R run or publication violates its contract."""


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
        raise Sop05rRunError("SOP05R run evidence must be canonical JSON") from exc


def _canonical_json_copy(value: object) -> object:
    return json.loads(_canonical_json_bytes(value).decode("ascii"))


def _json_file_bytes(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise Sop05rRunError(f"failed to hash artifact: {path}") from exc


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise Sop05rRunError(f"{name} must be a positive integer")
    return value


def _seed(value: object) -> int:
    if type(value) is not int or value < 0 or value >= 2**32:
        raise Sop05rRunError("seed must be an unsigned 32-bit integer")
    return value


@dataclass(frozen=True)
class Sop05rRunRequest:
    sop03_root: Path
    split: str
    base_config_path: Path
    generator_config_path: Path
    verification_action_config_path: Path
    output_dir: Path
    seed: int
    accepted_quota: int
    max_base_states: int
    checksum_workers: int
    workers: int
    git_executable: Path


@dataclass(frozen=True)
class Sop05rScheduleEntry:
    rank: int
    state_id: str
    base_seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "state_id": self.state_id,
            "base_seed": self.base_seed,
        }


@dataclass(frozen=True)
class Sop05rSelectionCandidate:
    generated_event_id: str
    base_state_id: str
    template_id: str
    history_visibility_regime: str
    active_revealable: bool
    schedule_rank: int


@dataclass(frozen=True)
class Sop05rSelectionResult:
    event_ids: tuple[str, ...]
    requested_history_counts: Mapping[str, int]
    exact_qualified_history_counts: Mapping[str, int]
    accepted_history_counts: Mapping[str, int]
    selected_history_counts: Mapping[str, int]
    active_revealable_requested_count: int
    active_revealable_accepted_count: int
    active_revealable_selected_count: int
    natural_difficult_selected_count: int
    deficits: Mapping[str, int]
    quota_met: bool


@dataclass(frozen=True)
class Sop05rAcceptedEvent:
    event: GeneratedEvent
    trajectory_record: Sop05rTrajectoryRecord
    template_id: str
    schedule_rank: int
    attempts_before_acceptance: int
    active_revealable: bool


@dataclass(frozen=True)
class Sop05rTemplateReport:
    report_version: str
    base_rank: int
    state_id: str
    template_id: str
    template_schedule_rank: tuple[int, ...]
    attempt_index: int
    geometry_eligible: bool
    planner_feasible: bool
    exact_history_qualified: bool
    time_aligned_collision: bool
    active_revealable: bool
    generated_event_id: str | None
    history_visibility_regime: str | None
    rejection_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "report_version": self.report_version,
            "base_rank": self.base_rank,
            "state_id": self.state_id,
            "template_id": self.template_id,
            "template_schedule_rank": list(self.template_schedule_rank),
            "attempt_index": self.attempt_index,
            "geometry_eligible": self.geometry_eligible,
            "planner_feasible": self.planner_feasible,
            "exact_history_qualified": self.exact_history_qualified,
            "time_aligned_collision": self.time_aligned_collision,
            "active_revealable": self.active_revealable,
            "generated_event_id": self.generated_event_id,
            "history_visibility_regime": self.history_visibility_regime,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class Sop05rBaseGenerationReport:
    schedule_entry: Sop05rScheduleEntry
    template_reports: tuple[Sop05rTemplateReport, ...]
    accepted_events: tuple[Sop05rAcceptedEvent, ...]


@dataclass(frozen=True)
class _Sop05rEventTransport:
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
class _Sop05rAcceptedEventTransport:
    event: _Sop05rEventTransport
    trajectory_record: Sop05rTrajectoryRecord
    template_id: str
    schedule_rank: int
    attempts_before_acceptance: int
    active_revealable: bool


@dataclass(frozen=True)
class _Sop05rBaseReportTransport:
    schedule_entry: Sop05rScheduleEntry
    template_reports: tuple[Sop05rTemplateReport, ...]
    accepted_events: tuple[_Sop05rAcceptedEventTransport, ...]


@dataclass(frozen=True)
class Sop05rGenerationCollection:
    base_reports: tuple[Sop05rBaseGenerationReport, ...]
    accepted_events: tuple[Sop05rAcceptedEvent, ...]
    selected_events: tuple[Sop05rAcceptedEvent, ...]
    selection: Sop05rSelectionResult
    generation_summary: Mapping[str, object]


@dataclass(frozen=True)
class Sop05rPublicationContext:
    run_id: str
    split: str
    seed: int
    accepted_quota: int
    base_config: Mapping[str, Any]
    config: Sop05rConfig
    verification_action_config: Mapping[str, object]
    input_lock: Mapping[str, object]
    producer_source_identity: Mapping[str, object]
    schedule: tuple[Sop05rScheduleEntry, ...]


@dataclass(frozen=True)
class PreparedSop05rRun:
    request: Sop05rRunRequest
    publication_context: Sop05rPublicationContext
    grid: GridSpec
    sop03: Sop03SplitInputs
    action_library: VerificationActionLibrary


@dataclass(frozen=True)
class Sop05rRunResult:
    run_state: str
    run_id: str
    output_dir: Path
    generation_summary: Mapping[str, object]
    publication_semantic_digest: str
    exit_code: int


def build_sop05r_schedule(
    state_ids: Iterable[str],
    *,
    seed: int,
    max_base_states: int,
) -> tuple[Sop05rScheduleEntry, ...]:
    stable_seed = _seed(seed)
    limit = _positive_int(max_base_states, name="max_base_states")
    ordered = tuple(sorted(state_ids))
    if not ordered or len(ordered) != len(set(ordered)):
        raise Sop05rRunError("SOP05R state IDs must be nonempty and unique")
    if any(not isinstance(state_id, str) or not state_id for state_id in ordered):
        raise Sop05rRunError("SOP05R state IDs must be nonempty strings")
    return tuple(
        Sop05rScheduleEntry(
            rank=rank,
            state_id=state_id,
            base_seed=derive_seed(stable_seed, "sop05r-base", state_id),
        )
        for rank, state_id in enumerate(ordered[:limit])
    )


def _selection_key(seed: int, candidate: Sop05rSelectionCandidate) -> tuple[str, ...]:
    payload = _canonical_json_bytes(
        {
            "seed": seed,
            "event_id": candidate.generated_event_id,
            "base_state_id": candidate.base_state_id,
            "template_id": candidate.template_id,
            "schedule_rank": candidate.schedule_rank,
        }
    )
    return (
        _sha256_bytes(b"sop05r_selection_v1\0" + payload),
        candidate.base_state_id,
        candidate.template_id,
        candidate.generated_event_id,
    )


def _history_requested_counts(total: int, config: Sop05rConfig) -> dict[str, int]:
    weights = config.history_policy.weights
    raw = {regime: total * weights[regime] for regime in HISTORY_VISIBILITY_REGIMES}
    result = {regime: int(math.floor(raw[regime])) for regime in raw}
    remainder = total - sum(result.values())
    order = sorted(
        HISTORY_VISIBILITY_REGIMES,
        key=lambda regime: (
            -(raw[regime] - result[regime]),
            HISTORY_VISIBILITY_REGIMES.index(regime),
        ),
    )
    for regime in order[:remainder]:
        result[regime] += 1
    return result


def select_sop05r_event_ids(
    candidates: Iterable[Sop05rSelectionCandidate],
    *,
    accepted_quota: int,
    seed: int,
    config: Sop05rConfig,
) -> Sop05rSelectionResult:
    total = _positive_int(accepted_quota, name="accepted_quota")
    stable_seed = _seed(seed)
    if not isinstance(config, Sop05rConfig):
        raise TypeError("config must be a Sop05rConfig")
    rows = tuple(candidates)
    if any(not isinstance(row, Sop05rSelectionCandidate) for row in rows):
        raise TypeError("candidates must contain Sop05rSelectionCandidate values")
    event_ids = [row.generated_event_id for row in rows]
    if len(event_ids) != len(set(event_ids)):
        raise Sop05rRunError("SOP05R selection event IDs must be unique")
    if any(row.history_visibility_regime not in HISTORY_VISIBILITY_REGIMES for row in rows):
        raise Sop05rRunError("SOP05R selection contains an ineligible history regime")

    requested = _history_requested_counts(total, config)
    grouped = {
        regime: sorted(
            [row for row in rows if row.history_visibility_regime == regime],
            key=partial(_selection_key, stable_seed),
        )
        for regime in HISTORY_VISIBILITY_REGIMES
    }
    selected: list[Sop05rSelectionCandidate] = []
    active_requested = 0
    if config.revealability.selection_filtering:
        active_requested = int(
            math.ceil(
                total * config.revealability.training_min_active_fraction - 1e-12
            )
        )
        target_slots = {
            regime: min(requested[regime], len(grouped[regime]))
            for regime in HISTORY_VISIBILITY_REGIMES
        }
        active_rows = {
            regime: [row for row in grouped[regime] if row.active_revealable]
            for regime in HISTORY_VISIBILITY_REGIMES
        }
        natural_rows = {
            regime: [row for row in grouped[regime] if not row.active_revealable]
            for regime in HISTORY_VISIBILITY_REGIMES
        }
        minimum_active = {
            regime: max(0, target_slots[regime] - len(natural_rows[regime]))
            for regime in HISTORY_VISIBILITY_REGIMES
        }
        maximum_active = {
            regime: min(target_slots[regime], len(active_rows[regime]))
            for regime in HISTORY_VISIBILITY_REGIMES
        }
        active_counts = dict(minimum_active)
        desired_active = min(active_requested, sum(maximum_active.values()))
        while sum(active_counts.values()) < desired_active:
            eligible = [
                regime
                for regime in HISTORY_VISIBILITY_REGIMES
                if active_counts[regime] < maximum_active[regime]
            ]
            if not eligible:
                break
            chosen_regime = min(
                eligible,
                key=lambda regime: _selection_key(
                    stable_seed,
                    active_rows[regime][active_counts[regime]],
                ),
            )
            active_counts[chosen_regime] += 1
        for regime in HISTORY_VISIBILITY_REGIMES:
            active_count = active_counts[regime]
            natural_count = target_slots[regime] - active_count
            selected.extend(active_rows[regime][:active_count])
            selected.extend(natural_rows[regime][:natural_count])
    else:
        for regime in HISTORY_VISIBILITY_REGIMES:
            selected.extend(grouped[regime][: requested[regime]])

    selected.sort(key=partial(_selection_key, stable_seed))
    selected_history = Counter(row.history_visibility_regime for row in selected)
    accepted_history = Counter(row.history_visibility_regime for row in rows)
    active_selected = sum(row.active_revealable for row in selected)
    deficits = {
        regime: requested[regime] - selected_history[regime]
        for regime in HISTORY_VISIBILITY_REGIMES
    }
    deficits["active_revealable"] = max(0, active_requested - active_selected)
    return Sop05rSelectionResult(
        event_ids=tuple(row.generated_event_id for row in selected),
        requested_history_counts=dict(requested),
        exact_qualified_history_counts={
            regime: accepted_history[regime] for regime in HISTORY_VISIBILITY_REGIMES
        },
        accepted_history_counts={
            regime: accepted_history[regime] for regime in HISTORY_VISIBILITY_REGIMES
        },
        selected_history_counts={
            regime: selected_history[regime] for regime in HISTORY_VISIBILITY_REGIMES
        },
        active_revealable_requested_count=active_requested,
        active_revealable_accepted_count=sum(row.active_revealable for row in rows),
        active_revealable_selected_count=active_selected,
        natural_difficult_selected_count=len(selected) - active_selected,
        deficits=deficits,
        quota_met=len(selected) == total and not any(deficits.values()),
    )


def collect_ranked_sop05r_reports(
    schedule: tuple[Sop05rScheduleEntry, ...],
    *,
    evaluate: Callable[[Sop05rScheduleEntry], _T],
    workers: int,
    executor_factory: Callable[..., Executor] | None = None,
) -> tuple[_T, ...]:
    worker_count = _positive_int(workers, name="workers")
    if not isinstance(schedule, tuple) or tuple(row.rank for row in schedule) != tuple(
        range(len(schedule))
    ):
        raise Sop05rRunError("SOP05R schedule ranks must be contiguous")
    if worker_count == 1:
        return tuple(evaluate(row) for row in schedule)
    factory = ProcessPoolExecutor if executor_factory is None else executor_factory
    completed: dict[int, _T] = {}
    iterator = iter(schedule)
    with factory(max_workers=worker_count) as executor:
        pending = {}
        for _ in range(min(worker_count, len(schedule))):
            row = next(iterator, None)
            if row is not None:
                pending[executor.submit(evaluate, row)] = row.rank
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                rank = pending.pop(future)
                completed[rank] = future.result()
                row = next(iterator, None)
                if row is not None:
                    pending[executor.submit(evaluate, row)] = row.rank
    return tuple(completed[rank] for rank in range(len(schedule)))


def _evidence_payload(evidence: ProducerEvidence) -> dict[str, object]:
    return {
        "code_commit": evidence.code_commit,
        "checksum_manifest_sha256": evidence.checksum_manifest_sha256,
        "audit_sha256": evidence.audit_sha256,
        "completion_policy": evidence.completion_policy,
    }


def _load_action_snapshot(path: Path) -> tuple[dict[str, object], VerificationActionLibrary]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise Sop05rRunError(f"failed to load verification action config: {exc}") from exc
    if not isinstance(raw, dict):
        raise Sop05rRunError("verification action config must be a mapping")
    try:
        library = load_verification_actions(path)
    except (TypeError, ValueError) as exc:
        raise Sop05rRunError(str(exc)) from exc
    snapshot = json.loads(_canonical_json_bytes(raw).decode("ascii"))
    return snapshot, library


def _validate_request(request: Sop05rRunRequest) -> None:
    if not isinstance(request, Sop05rRunRequest):
        raise TypeError("request must be a Sop05rRunRequest")
    if request.split not in SPLIT_NAMES:
        raise Sop05rRunError(f"unsupported split: {request.split!r}")
    _seed(request.seed)
    for name in ("accepted_quota", "max_base_states", "checksum_workers", "workers"):
        _positive_int(getattr(request, name), name=name)
    for name in (
        "sop03_root",
        "base_config_path",
        "generator_config_path",
        "verification_action_config_path",
        "output_dir",
        "git_executable",
    ):
        if not isinstance(getattr(request, name), Path):
            raise TypeError(f"{name} must be a Path")
    if request.output_dir.exists() or request.output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {request.output_dir}")


def prepare_sop05r_run(request: Sop05rRunRequest) -> PreparedSop05rRun:
    _validate_request(request)
    try:
        base_config = load_config(request.base_config_path)
        config = load_sop05r_config(request.generator_config_path)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise Sop05rRunError(str(exc)) from exc
    grid = build_grid_spec(base_config)
    if config.planner.rollout_steps != grid.future_steps or not np.isclose(
        config.planner.dt_s,
        float(base_config["bev"]["future_dt_s"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise Sop05rRunError("SOP05R planner and base future layout differ")
    if request.split != "train" and config.revealability.selection_filtering:
        raise Sop05rRunError("revealability filtering is forbidden outside train")
    action_snapshot, action_library = _load_action_snapshot(
        request.verification_action_config_path
    )
    sop03 = load_sop03_split_inputs(
        request.sop03_root,
        request.split,
        grid,
        checksum_workers=request.checksum_workers,
    )
    schedule = build_sop05r_schedule(
        sop03.manifest_index,
        seed=request.seed,
        max_base_states=request.max_base_states,
    )
    source_identity = _load_producer_source_identity(request.git_executable)
    input_lock = {
        "version": SOP05R_INPUT_LOCK_VERSION,
        "split": request.split,
        "sop03": _evidence_payload(sop03.producer_evidence),
        "sop04_trajectory_bank_is_input": False,
        "base_config_sha256": _sha256_bytes(_canonical_json_bytes(base_config)),
        "sop05r_config_digest": config.digest,
        "verification_action_config_sha256": _sha256_bytes(
            _canonical_json_bytes(action_snapshot)
        ),
        "schedule_sha256": _sha256_bytes(
            _canonical_json_bytes([row.as_dict() for row in schedule])
        ),
        "versions": {
            "generator_algorithm_version": SOP05R_GENERATOR_VERSION,
            "run_producer_version": SOP05R_RUN_VERSION,
            "selection_version": SOP05R_SELECTION_VERSION,
            "report_version": SOP05R_REPORT_VERSION,
        },
    }
    run_identity = {
        "split": request.split,
        "seed": request.seed,
        "accepted_quota": request.accepted_quota,
        "max_base_states": request.max_base_states,
        "input_lock": input_lock,
        "producer_source_identity": source_identity,
    }
    run_id = "sop05r-run-" + _sha256_bytes(
        b"sop05r_run_identity_v1\0" + _canonical_json_bytes(run_identity)
    )[:24]
    context = Sop05rPublicationContext(
        run_id=run_id,
        split=request.split,
        seed=request.seed,
        accepted_quota=request.accepted_quota,
        base_config=base_config,
        config=config,
        verification_action_config=action_snapshot,
        input_lock=input_lock,
        producer_source_identity=source_identity,
        schedule=schedule,
    )
    return PreparedSop05rRun(
        request=request,
        publication_context=context,
        grid=grid,
        sop03=sop03,
        action_library=action_library,
    )


def preflight_summary(prepared: PreparedSop05rRun) -> dict[str, object]:
    if not isinstance(prepared, PreparedSop05rRun):
        raise TypeError("prepared must be a PreparedSop05rRun")
    return {
        "status": "preflight_ok",
        "run_id": prepared.publication_context.run_id,
        "output_dir": str(prepared.request.output_dir),
        "split": prepared.request.split,
        "base_count": len(prepared.publication_context.schedule),
        "schedule_sha256": prepared.publication_context.input_lock[
            "schedule_sha256"
        ],
        "sop04_trajectory_bank_is_input": False,
    }


def _moving_route_exists(evaluation: object) -> bool:
    planner_result = evaluation.planner_result
    return any(route.slot_id != "stop" for route in planner_result.routes)


def _generate_base_report(
    prepared: PreparedSop05rRun,
    entry: Sop05rScheduleEntry,
) -> Sop05rBaseGenerationReport:
    context = prepared.publication_context
    base_state, oracle_context = prepared.sop03.load_pair(entry.state_id, prepared.grid)
    reports: list[Sop05rTemplateReport] = []
    accepted: list[Sop05rAcceptedEvent] = []
    planner = GeometryCachingObstaclePlanner()
    templates = iter_obstacle_target_templates(
        base_state=base_state,
        oracle_context=oracle_context,
        snippet_libraries=prepared.sop03.typed_libraries,
        base_config=context.base_config,
        config=context.config,
        seed=entry.base_seed,
    )
    for attempt_index, template_row in enumerate(templates):
        template = template_row.template
        if template is None:
            reports.append(
                Sop05rTemplateReport(
                    report_version=SOP05R_REPORT_VERSION,
                    base_rank=entry.rank,
                    state_id=entry.state_id,
                    template_id=template_row.template_id,
                    template_schedule_rank=template_row.schedule_rank,
                    attempt_index=attempt_index,
                    geometry_eligible=False,
                    planner_feasible=False,
                    exact_history_qualified=False,
                    time_aligned_collision=False,
                    active_revealable=False,
                    generated_event_id=None,
                    history_visibility_regime=None,
                    rejection_reason=template_row.rejection_reason,
                )
            )
            continue
        evaluation = evaluate_obstacle_first_template(
            template=template,
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=context.base_config,
            config=context.config,
            seed=entry.base_seed,
            planner=planner,
        )
        mother = evaluation.mother
        if mother is None:
            reports.append(
                Sop05rTemplateReport(
                    report_version=SOP05R_REPORT_VERSION,
                    base_rank=entry.rank,
                    state_id=entry.state_id,
                    template_id=template.template_id,
                    template_schedule_rank=template.schedule_rank,
                    attempt_index=attempt_index,
                    geometry_eligible=True,
                    planner_feasible=_moving_route_exists(evaluation),
                    exact_history_qualified=False,
                    time_aligned_collision=False,
                    active_revealable=False,
                    generated_event_id=None,
                    history_visibility_regime=None,
                    rejection_reason=evaluation.rejection_reason,
                )
            )
            continue
        revealability = evaluate_active_revealability(
            build_active_revealability_request(
                mother=mother,
                base_state=base_state,
                oracle_context=oracle_context,
                base_config=context.base_config,
                config=context.config,
                action_library=prepared.action_library,
            )
        )
        event = replace(
            mother.event,
            world=replace(
                mother.event.world,
                metadata={
                    **mother.event.world.metadata,
                    **revealability.as_metadata(),
                    "run_id": context.run_id,
                    "producer_source_identity": dict(
                        context.producer_source_identity
                    ),
                },
            ),
        )
        accepted_row = Sop05rAcceptedEvent(
            event=event,
            trajectory_record=mother.trajectory_record,
            template_id=template.template_id,
            schedule_rank=entry.rank,
            attempts_before_acceptance=attempt_index + 1,
            active_revealable=revealability.active_revealable,
        )
        accepted.append(accepted_row)
        reports.append(
            Sop05rTemplateReport(
                report_version=SOP05R_REPORT_VERSION,
                base_rank=entry.rank,
                state_id=entry.state_id,
                template_id=template.template_id,
                template_schedule_rank=template.schedule_rank,
                attempt_index=attempt_index,
                geometry_eligible=True,
                planner_feasible=True,
                exact_history_qualified=True,
                time_aligned_collision=True,
                active_revealable=revealability.active_revealable,
                generated_event_id=event.generated_event_id,
                history_visibility_regime=mother.history_assessment.regime,
                rejection_reason=None,
            )
        )
    return Sop05rBaseGenerationReport(
        schedule_entry=entry,
        template_reports=tuple(reports),
        accepted_events=tuple(accepted),
    )


def _transport_base_report(
    report: Sop05rBaseGenerationReport,
) -> _Sop05rBaseReportTransport:
    if not isinstance(report, Sop05rBaseGenerationReport):
        raise TypeError("report must be a Sop05rBaseGenerationReport")
    accepted = []
    for row in report.accepted_events:
        event = row.event
        record = event.target_motion_record
        footprint_spec = _canonical_json_copy(record.footprint_spec)
        if not isinstance(footprint_spec, dict):
            raise Sop05rRunError("record footprint_spec must be an object")
        transported_event = _Sop05rEventTransport(
            generated_event_id=event.generated_event_id,
            event_kind=event.event_kind,
            world=event.world,
            target=event.target,
            record_payload={
                "generated_event_id": record.generated_event_id,
                "world_id": record.world_id,
                "base_state_id": record.base_state_id,
                "trajectory_id": record.trajectory_id,
                "target_dynamic_object_id": record.target_dynamic_object_id,
                "source_snippet_id": record.source_snippet_id,
                "source_object_id": record.source_object_id,
                "object_type": record.object_type,
                "footprint_spec": footprint_spec,
                "footprint_spec_digest": record.footprint_spec_digest,
                "target_type_policy_digest": record.target_type_policy_digest,
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
        accepted.append(
            _Sop05rAcceptedEventTransport(
                event=transported_event,
                trajectory_record=row.trajectory_record,
                template_id=row.template_id,
                schedule_rank=row.schedule_rank,
                attempts_before_acceptance=row.attempts_before_acceptance,
                active_revealable=row.active_revealable,
            )
        )
    return _Sop05rBaseReportTransport(
        schedule_entry=report.schedule_entry,
        template_reports=report.template_reports,
        accepted_events=tuple(accepted),
    )


def _restore_base_report(
    transport: _Sop05rBaseReportTransport,
) -> Sop05rBaseGenerationReport:
    if not isinstance(transport, _Sop05rBaseReportTransport):
        raise Sop05rRunError("base worker returned the wrong transport type")
    accepted = []
    for row in transport.accepted_events:
        event_payload = row.event
        record = create_event_target_motion_record(**event_payload.record_payload)
        observed_identity = (
            record.schema_version,
            record.layout_version,
            record.history_array_digest,
            record.future_array_digest,
            record.record_digest,
        )
        if observed_identity != event_payload.expected_record_identity:
            raise Sop05rRunError("record identity changed across process transport")
        event = GeneratedEvent(
            generated_event_id=event_payload.generated_event_id,
            event_kind=event_payload.event_kind,
            world=event_payload.world,
            target=event_payload.target,
            target_motion_record=record,
            visibility_sequence=event_payload.visibility_sequence,
            target_visibility_history=event_payload.target_visibility_history,
            conflict_time_s=event_payload.conflict_time_s,
            conflict_index=event_payload.conflict_index,
        )
        accepted.append(
            Sop05rAcceptedEvent(
                event=event,
                trajectory_record=row.trajectory_record,
                template_id=row.template_id,
                schedule_rank=row.schedule_rank,
                attempts_before_acceptance=row.attempts_before_acceptance,
                active_revealable=row.active_revealable,
            )
        )
    return Sop05rBaseGenerationReport(
        schedule_entry=transport.schedule_entry,
        template_reports=transport.template_reports,
        accepted_events=tuple(accepted),
    )


def _generate_base_report_transport(
    prepared: PreparedSop05rRun,
    entry: Sop05rScheduleEntry,
) -> _Sop05rBaseReportTransport:
    return _transport_base_report(_generate_base_report(prepared, entry))


def _selection_candidate(row: Sop05rAcceptedEvent) -> Sop05rSelectionCandidate:
    metadata = row.event.world.metadata
    return Sop05rSelectionCandidate(
        generated_event_id=row.event.generated_event_id,
        base_state_id=row.event.world.base_state_id,
        template_id=row.template_id,
        history_visibility_regime=str(
            metadata["target_history_visibility_regime"]
        ),
        active_revealable=row.active_revealable,
        schedule_rank=row.schedule_rank,
    )


def build_sop05r_generation_collection(
    reports: tuple[Sop05rBaseGenerationReport, ...],
    *,
    accepted_quota: int,
    seed: int,
    config: Sop05rConfig,
) -> Sop05rGenerationCollection:
    if not isinstance(reports, tuple) or tuple(
        report.schedule_entry.rank for report in reports
    ) != tuple(range(len(reports))):
        raise Sop05rRunError("SOP05R base reports must preserve schedule order")
    accepted = tuple(row for report in reports for row in report.accepted_events)
    ids = tuple(row.event.generated_event_id for row in accepted)
    if len(ids) != len(set(ids)):
        raise Sop05rRunError("SOP05R accepted event IDs must be unique")
    selection = select_sop05r_event_ids(
        tuple(_selection_candidate(row) for row in accepted),
        accepted_quota=accepted_quota,
        seed=seed,
        config=config,
    )
    by_id = {row.event.generated_event_id: row for row in accepted}
    selected = tuple(by_id[event_id] for event_id in selection.event_ids)
    template_rows = tuple(row for report in reports for row in report.template_reports)
    rejection_reasons = Counter(
        row.rejection_reason for row in template_rows if row.rejection_reason is not None
    )
    summary = {
        "summary_version": SOP05R_SUMMARY_VERSION,
        "selection_version": SOP05R_SELECTION_VERSION,
        "input_base_count": len(reports),
        "geometry_eligible_base_count": sum(
            any(row.geometry_eligible for row in report.template_reports)
            for report in reports
        ),
        "template_count": len(template_rows),
        "geometry_eligible_template_count": sum(
            row.geometry_eligible for row in template_rows
        ),
        "planner_feasible_template_count": sum(
            row.planner_feasible for row in template_rows
        ),
        "exact_history_qualified_count": sum(
            row.exact_history_qualified for row in template_rows
        ),
        "time_aligned_collision_count": sum(
            row.time_aligned_collision for row in template_rows
        ),
        "active_revealable_count": sum(row.active_revealable for row in accepted),
        "accepted_count": len(accepted),
        "selected_count": len(selected),
        "accepted_quota": accepted_quota,
        "quota_met": selection.quota_met,
        "history_visibility": {
            "requested": dict(selection.requested_history_counts),
            "exact_qualified": dict(selection.exact_qualified_history_counts),
            "accepted": dict(selection.accepted_history_counts),
            "selected": dict(selection.selected_history_counts),
            "deficits": {
                regime: selection.deficits[regime]
                for regime in HISTORY_VISIBILITY_REGIMES
            },
        },
        "revealability": {
            "selection_filtering": config.revealability.selection_filtering,
            "requested_active_count": selection.active_revealable_requested_count,
            "accepted_active_count": selection.active_revealable_accepted_count,
            "selected_active_count": selection.active_revealable_selected_count,
            "selected_natural_difficult_count": (
                selection.natural_difficult_selected_count
            ),
            "active_deficit": selection.deficits["active_revealable"],
        },
        "selected_event_ids": list(selection.event_ids),
        "attempts_per_accepted_event": [
            row.attempts_before_acceptance for row in accepted
        ],
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }
    return Sop05rGenerationCollection(
        base_reports=reports,
        accepted_events=accepted,
        selected_events=selected,
        selection=selection,
        generation_summary=summary,
    )


def collect_sop05r_generation(prepared: PreparedSop05rRun) -> Sop05rGenerationCollection:
    if not isinstance(prepared, PreparedSop05rRun):
        raise TypeError("prepared must be a PreparedSop05rRun")
    transported = collect_ranked_sop05r_reports(
        prepared.publication_context.schedule,
        evaluate=partial(_generate_base_report_transport, prepared),
        workers=prepared.request.workers,
    )
    reports = tuple(_restore_base_report(row) for row in transported)
    return build_sop05r_generation_collection(
        reports,
        accepted_quota=prepared.request.accepted_quota,
        seed=prepared.request.seed,
        config=prepared.publication_context.config,
    )


def _bind_publication_metadata(
    row: Sop05rAcceptedEvent,
    context: Sop05rPublicationContext,
) -> Sop05rAcceptedEvent:
    event = row.event
    world = replace(
        event.world,
        metadata={
            **event.world.metadata,
            "run_id": context.run_id,
            "producer_source_identity": dict(context.producer_source_identity),
        },
    )
    return replace(row, event=replace(event, world=world))


def _template_report_bytes(collection: Sop05rGenerationCollection) -> bytes:
    rows = sorted(
        (
            row
            for report in collection.base_reports
            for row in report.template_reports
        ),
        key=lambda row: (row.base_rank, row.attempt_index, row.template_id),
    )
    return b"".join(_canonical_json_bytes(row.as_dict()) + b"\n" for row in rows)


def _event_row(row: Sop05rAcceptedEvent) -> dict[str, object]:
    event = row.event
    metadata = event.world.metadata
    return {
        "generated_event_id": event.generated_event_id,
        "event_kind": event.event_kind,
        "world_id": event.world.world_id,
        "base_state_id": event.world.base_state_id,
        "trajectory_id": event.target_motion_record.trajectory_id,
        "template_id": row.template_id,
        "target_motion_record_digest": event.target_motion_record.record_digest,
        "visibility_sequence": [bool(value) for value in event.visibility_sequence],
        "target_visibility_history": [
            bool(value) for value in event.target_visibility_history
        ],
        "conflict_time_s": event.conflict_time_s,
        "conflict_index": event.conflict_index,
        "history_visibility_regime": metadata[
            "target_history_visibility_regime"
        ],
        "active_revealable": row.active_revealable,
        "active_revealable_action_ids": list(
            metadata["active_revealable_action_ids"]
        ),
    }


def _checksum_manifest_bytes(root: Path) -> bytes:
    rows = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise Sop05rRunError(f"SOP05R artifact must not be a symlink: {relative}")
        if not path.is_file() or relative in {
            "artifact_checksums.sha256",
            ".sop05r-complete",
        }:
            continue
        rows.append(f"{_sha256_file(path)}  {relative}\n")
    return "".join(sorted(rows)).encode("ascii")


def _publication_digest(
    *,
    context: Sop05rPublicationContext,
    event_rows: list[dict[str, object]],
    reports_sha256: str,
    trajectory_digest: str,
    target_manifest_digest: str,
    target_payload_digest: str,
) -> str:
    payload = {
        "run_id": context.run_id,
        "split": context.split,
        "seed": context.seed,
        "accepted_quota": context.accepted_quota,
        "config_digest": context.config.digest,
        "input_lock": context.input_lock,
        "producer_source_identity": context.producer_source_identity,
        "event_rows": event_rows,
        "template_reports_sha256": reports_sha256,
        "trajectory_collection_semantic_digest": trajectory_digest,
        "target_motion_manifest_digest": target_manifest_digest,
        "target_motion_payload_semantic_digest": target_payload_digest,
    }
    return _sha256_bytes(
        b"sop05r_publication_identity_v1\0" + _canonical_json_bytes(payload)
    )


def _publish_empty_nested_collections(
    staging: Path,
) -> tuple[Sop05rTrajectoryStore, object]:
    trajectory_dir = staging / "planner_trajectories"
    trajectory_dir.mkdir()
    trajectory_digest = _sha256_bytes(
        b"sop05r_empty_trajectory_store_v1\0"
    )
    trajectory_manifest = {
        "version": SOP05R_EMPTY_TRAJECTORY_STORE_VERSION,
        "record_count": 0,
        "collection_semantic_digest": trajectory_digest,
    }
    (trajectory_dir / "empty.json").write_bytes(
        _json_file_bytes(trajectory_manifest)
    )
    trajectory_store = Sop05rTrajectoryStore(
        records=(),
        manifest=trajectory_manifest,
        collection_semantic_digest=trajectory_digest,
    )

    target_dir = staging / "target_motion"
    target_dir.mkdir()
    target_manifest_digest = _sha256_bytes(
        b"sop05r_empty_target_motion_manifest_v1\0"
    )
    target_payload_digest = _sha256_bytes(
        b"sop05r_empty_target_motion_payload_v1\0"
    )
    target_summary = {
        "version": SOP05R_EMPTY_TARGET_MOTION_VERSION,
        "record_count": 0,
        "manifest_digest": target_manifest_digest,
        "payload_semantic_digest": target_payload_digest,
    }
    (target_dir / "empty.json").write_bytes(_json_file_bytes(target_summary))
    from src.generation.event_target_motion_shard import LoadedEventTargetMotionShard

    target_motion = LoadedEventTargetMotionShard(
        records=(),
        worlds={},
        manifest_digest=target_manifest_digest,
        payload_semantic_digest=target_payload_digest,
        summary=target_summary,
    )
    (staging / "worlds").mkdir()
    return trajectory_store, target_motion


def publish_sop05r_generation(
    output_dir: str | Path,
    context: Sop05rPublicationContext,
    collection: Sop05rGenerationCollection,
) -> Sop05rRunResult:
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    if not isinstance(context, Sop05rPublicationContext):
        raise TypeError("context must be a Sop05rPublicationContext")
    if not isinstance(collection, Sop05rGenerationCollection):
        raise TypeError("collection must be a Sop05rGenerationCollection")
    selected = tuple(
        _bind_publication_metadata(row, context) for row in collection.selected_events
    )
    event_ids = tuple(row.event.generated_event_id for row in selected)
    if event_ids != collection.selection.event_ids:
        raise Sop05rRunError("selected event order differs from selection evidence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.sop05r-staging-",
            dir=destination.parent,
        )
    )
    try:
        if selected:
            trajectory_store_path = staging / "planner_trajectories"
            publish_sop05r_trajectory_store(
                trajectory_store_path,
                tuple(row.trajectory_record for row in selected),
                base_config=context.base_config,
            )
            trajectory_store = load_sop05r_trajectory_store(trajectory_store_path)
            target_motion_path = staging / "target_motion"
            write_event_target_motion_shard(
                [row.event.target_motion_record for row in selected],
                [row.event.world for row in selected],
                target_motion_path,
                grid=build_grid_spec(dict(context.base_config)),
            )
            target_motion = load_event_target_motion_shard(
                target_motion_path,
                grid=build_grid_spec(dict(context.base_config)),
                expected_generated_event_ids=set(event_ids),
            )
            worlds_dir = staging / "worlds"
            worlds_dir.mkdir()
            for path in sorted((target_motion_path / "oracle_worlds").iterdir()):
                if not path.is_file() or path.is_symlink():
                    raise Sop05rRunError("target motion world artifact is invalid")
                shutil.copyfile(path, worlds_dir / path.name)
        else:
            trajectory_store, target_motion = _publish_empty_nested_collections(
                staging
            )

        report_bytes = _template_report_bytes(collection)
        (staging / "template_reports.jsonl").write_bytes(report_bytes)
        event_rows = [_event_row(row) for row in selected]
        event_bytes = b"".join(
            _canonical_json_bytes(row) + b"\n" for row in event_rows
        )
        (staging / "events.jsonl").write_bytes(event_bytes)
        publication_digest = _publication_digest(
            context=context,
            event_rows=event_rows,
            reports_sha256=_sha256_bytes(report_bytes),
            trajectory_digest=trajectory_store.collection_semantic_digest,
            target_manifest_digest=target_motion.manifest_digest,
            target_payload_digest=target_motion.payload_semantic_digest,
        )
        quota_met = collection.selection.quota_met
        run_state = "complete" if quota_met else "quota_unmet"
        summary = {
            **dict(collection.generation_summary),
            "run_id": context.run_id,
            "run_state": run_state,
            "split": context.split,
            "publication_semantic_digest": publication_digest,
        }
        (staging / "generation_summary.json").write_bytes(_json_file_bytes(summary))
        manifest = {
            "schema_version": context.config.schema_version,
            "manifest_version": SOP05R_MANIFEST_VERSION,
            "producer_version": SOP05R_RUN_VERSION,
            "generator_algorithm_version": SOP05R_GENERATOR_VERSION,
            "selection_version": SOP05R_SELECTION_VERSION,
            "report_version": SOP05R_REPORT_VERSION,
            "run_id": context.run_id,
            "run_state": run_state,
            "split": context.split,
            "seed": context.seed,
            "accepted_quota": context.accepted_quota,
            "input_lock": context.input_lock,
            "producer_source_identity": context.producer_source_identity,
            "base_config": context.base_config,
            "sop05r_config": {
                **context.config.as_dict(),
                "digest": context.config.digest,
            },
            "verification_action_config": context.verification_action_config,
            "schedule": [row.as_dict() for row in context.schedule],
            "selected_event_ids": list(event_ids),
            "publication_semantic_digest": publication_digest,
            "artifacts": {
                "generation_summary": "generation_summary.json",
                "template_reports": "template_reports.jsonl",
                "events": "events.jsonl",
                "worlds": "worlds",
                "target_motion": "target_motion",
                "planner_trajectories": "planner_trajectories",
                "checksums": "artifact_checksums.sha256",
                "completion_marker": ".sop05r-complete",
                "trajectory_collection_semantic_digest": (
                    trajectory_store.collection_semantic_digest
                ),
                "target_motion_manifest_digest": target_motion.manifest_digest,
                "target_motion_payload_semantic_digest": (
                    target_motion.payload_semantic_digest
                ),
            },
        }
        manifest_bytes = _json_file_bytes(manifest)
        (staging / "manifest.json").write_bytes(manifest_bytes)
        checksums = _checksum_manifest_bytes(staging)
        (staging / "artifact_checksums.sha256").write_bytes(checksums)
        if quota_met:
            marker = {
                "marker_version": SOP05R_COMPLETION_MARKER_VERSION,
                "run_id": context.run_id,
                "publication_semantic_digest": publication_digest,
                "manifest_sha256": _sha256_bytes(manifest_bytes),
                "artifact_checksums_sha256": _sha256_bytes(checksums),
                "trajectory_collection_semantic_digest": (
                    trajectory_store.collection_semantic_digest
                ),
                "target_motion_manifest_digest": target_motion.manifest_digest,
                "target_motion_payload_semantic_digest": (
                    target_motion.payload_semantic_digest
                ),
            }
            (staging / ".sop05r-complete").write_bytes(_json_file_bytes(marker))

        from .sop05r_output_loader import load_sop05r_events

        loaded = load_sop05r_events(
            staging,
            require_complete=quota_met,
            expected_publication_semantic_digest=publication_digest,
            expected_run_id=context.run_id,
        )
        if tuple(event.generated_event_id for event in loaded.events) != event_ids:
            raise Sop05rRunError("SOP05R publication self-reload event order mismatch")
        atomic_rename_noreplace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return Sop05rRunResult(
        run_state=run_state,
        run_id=context.run_id,
        output_dir=destination,
        generation_summary=summary,
        publication_semantic_digest=publication_digest,
        exit_code=0 if quota_met else 4,
    )


def execute_sop05r_run(request: Sop05rRunRequest) -> Sop05rRunResult:
    prepared = prepare_sop05r_run(request)
    collection = collect_sop05r_generation(prepared)
    return publish_sop05r_generation(
        request.output_dir,
        prepared.publication_context,
        collection,
    )
