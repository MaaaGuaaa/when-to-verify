"""Resumable bounded-memory releases for SOP11--13 verification data."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
import hashlib
import json
import multiprocessing
from numbers import Real
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any

import numpy as np

from src.contracts import VerificationSample, build_grid_spec
from src.datasets.verification_dataloader import (
    VERIFICATION_SHARD_LAYOUT_VERSION,
    write_verification_shard,
)
from src.datasets.verification_dataset import VERIFICATION_DATASET_VERSION
from src.generation.sop05r_contracts import load_sop05r_teb_config
from src.planning.verification_actions import (
    ACTION_LIBRARY_VERSION,
    CANONICAL_ACTION_IDS,
    VerificationActionLibrary,
    load_verification_actions,
)
from src.utils.atomic_publish import atomic_rename_noreplace
from src.utils.config import config_digest, load_config

from .sop06_finalized_source import (
    Sop06AcceptedFinalRecord,
    Sop06FinalizedSource,
    load_sop06_finalized_source,
)
from .verification_gt import (
    SAMPLED_REALIZATION_GT_VERSION,
    VerificationGTConfig,
    VERIFICATION_GT_VERSION,
    load_verification_gt_config,
)
from .verification_pipeline import (
    VERIFICATION_PIPELINE_VERSION,
    VerificationGroupResult,
    VerificationSourceIneligibleError,
    build_finalized_verification_input,
    generate_verification_group,
)


VERIFICATION_RELEASE_VERSION = "verification_release_v2"
_REQUEST = "request.json"
_MANIFEST = "manifest.json"
_CHECKSUMS = "checksums.json"
_COMPLETE = "COMPLETE.json"
_SHARDS = "shards"
_TASK_SUMMARY = "task_summary.json"
_DATA = "data"
_ROOT_FILES = frozenset({_REQUEST, _MANIFEST, _CHECKSUMS, _COMPLETE, _SHARDS})
_SOURCE_FAMILIES = frozenset({"natural", "a_supplement"})
_SOURCE_MODES = frozenset({"complete_mother", "partial_m6_reconstruction"})
_SPLITS = frozenset({"train", "calibration", "val", "test"})
_REVALUATION_ROW_KEYS = frozenset(
    {
        "split",
        "task_id",
        "mother_id",
        "sample_id",
        "ranking_group_id",
        "action_id",
        "realized_execute_loss",
        "unclipped_best_policy_loss",
        "action_cost",
        "original_reject_cost",
    }
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class VerificationReleaseRequest:
    source_family: str
    source_mode: str
    source_root: Path
    final_scenario_root: Path
    split: str
    output_dir: Path
    actions_config_path: Path
    gt_config_path: Path
    source_cache_root: Path | None = None
    workers: int = 4
    groups_per_shard: int = 16
    max_replan_candidates: int = 4
    max_tasks: int | None = None
    sop03_root: Path | None = None
    long40_human_artifact: Path | None = None
    base_state_start: int | None = None
    max_base_states: int | None = None
    base_config_path: Path | None = None
    generator_config_path: Path | None = None


@dataclass(frozen=True)
class VerificationReleaseResult:
    output_dir: Path
    split: str
    task_count: int
    accepted_group_count: int
    rejected_task_count: int
    sample_count: int
    shard_count: int
    reused_shard_count: int
    manifest_digest: str


@dataclass(frozen=True)
class LoadedVerificationRelease:
    root: Path
    request_identity: str
    split: str
    task_count: int
    accepted_group_count: int
    rejected_task_count: int
    sample_count: int
    shard_count: int
    manifest_digest: str
    request: Mapping[str, object]
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "request", MappingProxyType(dict(self.request)))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


@dataclass(frozen=True)
class VerificationRevaluationRecord:
    """Authenticated label-side scalars sufficient to change reject cost."""

    release_request_identity: str
    split: str
    task_id: str
    mother_id: str
    sample_id: str
    ranking_group_id: str
    action_id: str
    realized_execute_loss: float
    unclipped_best_policy_loss: float | None
    action_cost: float
    original_reject_cost: float

    def __post_init__(self) -> None:
        for name in (
            "release_request_identity",
            "task_id",
            "mother_id",
            "sample_id",
            "ranking_group_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.split not in _SPLITS:
            raise ValueError("split is invalid")
        if self.action_id not in CANONICAL_ACTION_IDS:
            raise ValueError("action_id is not canonical")
        for name in (
            "realized_execute_loss",
            "action_cost",
            "original_reject_cost",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        if self.unclipped_best_policy_loss is not None:
            object.__setattr__(
                self,
                "unclipped_best_policy_loss",
                _finite_nonnegative(
                    self.unclipped_best_policy_loss,
                    name="unclipped_best_policy_loss",
                ),
            )


@dataclass(frozen=True)
class _TaskOutcome:
    task_id: str
    mother_id: str
    samples: tuple[VerificationSample, ...]
    revaluation_records: tuple[dict[str, object], ...]
    sampled_child_world_id: str | None
    rejection_reason: str | None
    rejection_detail: str | None

    @property
    def accepted(self) -> bool:
        return self.rejection_reason is None


BuildOne = Callable[
    [
        Sop06FinalizedSource,
        Sop06AcceptedFinalRecord,
        VerificationActionLibrary,
        VerificationGTConfig,
        int,
    ],
    VerificationGroupResult,
]
SourceLoader = Callable[[VerificationReleaseRequest], Sop06FinalizedSource]
ProgressCallback = Callable[[int, int, bool], None]


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("verification release metadata must be canonical JSON") from exc


def _json_file(value: object) -> bytes:
    return _canonical_json(value) + b"\n"


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read verification release JSON: {path}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"failed to checksum verification release file: {path}") from exc
    return digest.hexdigest()


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _task_revaluation_row(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _REVALUATION_ROW_KEYS:
        raise ValueError("revaluation record keys are invalid")
    result = dict(value)
    for name in (
        "task_id",
        "mother_id",
        "sample_id",
        "ranking_group_id",
    ):
        if not isinstance(result[name], str) or not result[name]:
            raise ValueError(f"revaluation record {name} must be non-empty")
    if result["split"] not in _SPLITS:
        raise ValueError("revaluation record split is invalid")
    if result["action_id"] not in CANONICAL_ACTION_IDS:
        raise ValueError("revaluation record action_id is not canonical")
    for name in (
        "realized_execute_loss",
        "action_cost",
        "original_reject_cost",
    ):
        result[name] = _finite_nonnegative(
            result[name],
            name=f"revaluation record {name}",
        )
    if result["unclipped_best_policy_loss"] is not None:
        result["unclipped_best_policy_loss"] = _finite_nonnegative(
            result["unclipped_best_policy_loss"],
            name="revaluation record unclipped_best_policy_loss",
        )
    return result


def _repository_relative(path: Path, *, name: str) -> str:
    candidate = path if path.is_absolute() else _REPOSITORY_ROOT / path
    resolved = candidate.resolve()
    try:
        return resolved.relative_to(_REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{name} must be inside the repository") from exc


def _paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.resolve()
    second_resolved = second.resolve()
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def _request_payload(request: VerificationReleaseRequest) -> dict[str, object]:
    if not isinstance(request, VerificationReleaseRequest):
        raise TypeError("request must be a VerificationReleaseRequest")
    if request.source_family not in _SOURCE_FAMILIES:
        raise ValueError("source_family is invalid")
    if request.source_mode not in _SOURCE_MODES:
        raise ValueError("source_mode is invalid")
    if request.split not in _SPLITS:
        raise ValueError("split is invalid")
    _positive_int(request.workers, name="workers")
    _positive_int(request.groups_per_shard, name="groups_per_shard")
    _positive_int(
        request.max_replan_candidates,
        name="max_replan_candidates",
    )
    if request.max_tasks is not None:
        _positive_int(request.max_tasks, name="max_tasks")
    input_roots = tuple(
        path
        for path in (
            request.source_root,
            request.final_scenario_root,
            request.source_cache_root,
            request.sop03_root,
            request.long40_human_artifact,
        )
        if path is not None
    )
    if any(_paths_overlap(request.output_dir, root) for root in input_roots):
        raise ValueError("output_dir must not overlap input roots")
    partial_values = (
        request.sop03_root,
        request.long40_human_artifact,
        request.base_state_start,
        request.max_base_states,
        request.base_config_path,
        request.generator_config_path,
    )
    if request.source_mode == "partial_m6_reconstruction":
        if any(value is None for value in partial_values):
            raise ValueError("partial-M6 release arguments are required")
        if (
            isinstance(request.base_state_start, bool)
            or not isinstance(request.base_state_start, int)
            or request.base_state_start < 0
        ):
            raise ValueError("base_state_start must be a nonnegative integer")
        _positive_int(request.max_base_states, name="max_base_states")
    elif any(value is not None for value in partial_values):
        raise ValueError("partial-M6 arguments are forbidden for complete mothers")
    payload: dict[str, object] = {
        "version": VERIFICATION_RELEASE_VERSION,
        "source_family": request.source_family,
        "source_mode": request.source_mode,
        "source_root": _repository_relative(
            request.source_root,
            name="source_root",
        ),
        "final_scenario_root": _repository_relative(
            request.final_scenario_root,
            name="final_scenario_root",
        ),
        "split": request.split,
        "groups_per_shard": request.groups_per_shard,
        "max_replan_candidates": request.max_replan_candidates,
        "max_tasks": request.max_tasks,
        "actions_config": _repository_relative(
            request.actions_config_path,
            name="actions_config_path",
        ),
        "actions_config_sha256": _sha256_file(request.actions_config_path),
        "gt_config": _repository_relative(
            request.gt_config_path,
            name="gt_config_path",
        ),
        "gt_config_sha256": _sha256_file(request.gt_config_path),
        "scientific_versions": {
            "action_library": ACTION_LIBRARY_VERSION,
            "dataset": VERIFICATION_DATASET_VERSION,
            "ground_truth": VERIFICATION_GT_VERSION,
            "pipeline": VERIFICATION_PIPELINE_VERSION,
            "sampled_realization_ground_truth": (
                SAMPLED_REALIZATION_GT_VERSION
            ),
            "shard_layout": VERIFICATION_SHARD_LAYOUT_VERSION,
        },
    }
    if request.source_mode == "partial_m6_reconstruction":
        assert request.sop03_root is not None
        assert request.long40_human_artifact is not None
        assert request.base_config_path is not None
        assert request.generator_config_path is not None
        payload["partial_m6"] = {
            "sop03_root": _repository_relative(
                request.sop03_root,
                name="sop03_root",
            ),
            "long40_human_artifact": _repository_relative(
                request.long40_human_artifact,
                name="long40_human_artifact",
            ),
            "base_state_start": request.base_state_start,
            "max_base_states": request.max_base_states,
            "base_config": _repository_relative(
                request.base_config_path,
                name="base_config_path",
            ),
            "base_config_sha256": _sha256_file(request.base_config_path),
            "generator_config": _repository_relative(
                request.generator_config_path,
                name="generator_config_path",
            ),
            "generator_config_sha256": _sha256_file(
                request.generator_config_path
            ),
        }
    return payload


def _request_document(
    request: VerificationReleaseRequest,
    *,
    source: Sop06FinalizedSource,
) -> dict[str, object]:
    payload = _request_payload(request)
    payload["resolved_source"] = {
        "source_publication_semantic_digest": (
            source.source_publication_semantic_digest
        ),
        "final_release_identity": source.final_release_identity,
        "base_config_digest": config_digest(dict(source.base_config)),
    }
    identity = hashlib.sha256(
        b"verification_release_request_v2\0" + _canonical_json(payload)
    ).hexdigest()
    return {
        "version": VERIFICATION_RELEASE_VERSION,
        "request_identity": identity,
        "request": payload,
    }


def _load_source(request: VerificationReleaseRequest) -> Sop06FinalizedSource:
    read_source_root = request.source_cache_root or request.source_root
    if request.source_mode == "complete_mother":
        return load_sop06_finalized_source(
            source_mode=request.source_mode,
            source_root=read_source_root,
            final_scenario_root=request.final_scenario_root,
            split=request.split,
        )
    assert request.sop03_root is not None
    assert request.long40_human_artifact is not None
    assert request.base_state_start is not None
    assert request.max_base_states is not None
    assert request.base_config_path is not None
    assert request.generator_config_path is not None
    base_config = load_config(request.base_config_path)
    generator = load_sop05r_teb_config(request.generator_config_path)
    return load_sop06_finalized_source(
        source_mode=request.source_mode,
        source_root=read_source_root,
        final_scenario_root=request.final_scenario_root,
        split=request.split,
        sop03_root=request.sop03_root,
        long40_human_artifact=request.long40_human_artifact,
        base_state_start=request.base_state_start,
        max_base_states=request.max_base_states,
        base_config=base_config,
        source_config_digest=generator.digest,
        centerline_epsilon_m=(
            generator.occlusion.centerline_intersection_epsilon_m
        ),
    )


def _default_build_one(
    source: Sop06FinalizedSource,
    accepted: Sop06AcceptedFinalRecord,
    library: VerificationActionLibrary,
    gt_config: VerificationGTConfig,
    max_replan_candidates: int,
) -> VerificationGroupResult:
    publication = source.resolve(accepted).publication
    if (
        publication.sample_id != accepted.scenario_id
        or publication.mother_id != accepted.mother_id
        or publication.split != accepted.split
    ):
        raise ValueError("resolved publication identity differs from task")
    target_current_pose = (
        np.array(
            source.finalized.history_poses[accepted.target_row, -1],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        if accepted.target_present
        else None
    )
    pipeline_input = build_finalized_verification_input(
        publication,
        base_config=source.base_config,
        action_library=library,
        target_current_pose=target_current_pose,
        source_publication_semantic_digest=(
            source.source_publication_semantic_digest
        ),
        final_release_identity=source.final_release_identity,
    )
    return generate_verification_group(
        pipeline_input,
        base_config=source.base_config,
        action_library=library,
        gt_config=gt_config,
        max_replan_candidates=max_replan_candidates,
    )


def _revaluation_rows(
    result: VerificationGroupResult,
    accepted: Sop06AcceptedFinalRecord,
) -> tuple[dict[str, object], ...]:
    values = dict(result.values)
    if set(values) != set(CANONICAL_ACTION_IDS):
        raise ValueError("accepted task value results must contain six actions")
    rows: list[dict[str, object]] = []
    for sample in result.samples:
        value = values[sample.verification_action_id]
        if (
            value.verification_action_id != sample.verification_action_id
            or value.sampled_child_world_id != result.sampled_child_world_id
            or value.nominal_trajectory_id != sample.nominal_trajectory_id
        ):
            raise ValueError("accepted task value/sample identity differs")
        ranking_group_id = sample.metadata.get("ranking_group_id")
        row = _task_revaluation_row(
            {
                "split": sample.split,
                "task_id": accepted.scenario_id,
                "mother_id": accepted.mother_id,
                "sample_id": sample.sample_id,
                "ranking_group_id": ranking_group_id,
                "action_id": sample.verification_action_id,
                "realized_execute_loss": value.realized_execute_loss,
                "unclipped_best_policy_loss": (
                    value.unclipped_best_policy_loss
                ),
                "action_cost": value.action_cost,
                "original_reject_cost": value.reject_cost,
            }
        )
        rows.append(row)
    if tuple(row["action_id"] for row in rows) != CANONICAL_ACTION_IDS:
        raise ValueError("accepted task revaluation action order differs")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError("accepted task revaluation sample IDs are not unique")
    return tuple(rows)


def _task_outcome(
    source: Sop06FinalizedSource,
    accepted: Sop06AcceptedFinalRecord,
    library: VerificationActionLibrary,
    gt_config: VerificationGTConfig,
    max_replan_candidates: int,
    build_one: BuildOne,
) -> _TaskOutcome:
    try:
        result = build_one(
            source,
            accepted,
            library,
            gt_config,
            max_replan_candidates,
        )
    except VerificationSourceIneligibleError as exc:
        return _TaskOutcome(
            task_id=accepted.scenario_id,
            mother_id=accepted.mother_id,
            samples=(),
            revaluation_records=(),
            sampled_child_world_id=None,
            rejection_reason=exc.reason,
            rejection_detail=exc.detail,
        )
    if not isinstance(result, VerificationGroupResult):
        raise TypeError("build_one must return VerificationGroupResult")
    if result.version != VERIFICATION_PIPELINE_VERSION:
        raise ValueError("accepted task verification pipeline version differs")
    if len(result.samples) != len(CANONICAL_ACTION_IDS):
        raise ValueError("accepted verification task must contain six samples")
    if (
        not isinstance(result.sampled_child_world_id, str)
        or not result.sampled_child_world_id
    ):
        raise ValueError("accepted task sampled child world ID must be non-empty")
    if tuple(
        sample.verification_action_id for sample in result.samples
    ) != CANONICAL_ACTION_IDS:
        raise ValueError("accepted verification task action order differs")
    if len({sample.split for sample in result.samples}) != 1 or any(
        sample.split != accepted.split for sample in result.samples
    ):
        raise ValueError("accepted verification task split differs")
    if any(
        sample.metadata["label_audit"]["sampled_child_world_id"]
        != result.sampled_child_world_id
        for sample in result.samples
    ):
        raise ValueError("accepted task sampled child audit differs")
    return _TaskOutcome(
        task_id=accepted.scenario_id,
        mother_id=accepted.mother_id,
        samples=result.samples,
        revaluation_records=_revaluation_rows(result, accepted),
        sampled_child_world_id=result.sampled_child_world_id,
        rejection_reason=None,
        rejection_detail=None,
    )


_FORK_SOURCE: Sop06FinalizedSource | None = None
_FORK_BOUNDARY: tuple[Sop06AcceptedFinalRecord, ...] = ()
_FORK_LIBRARY: VerificationActionLibrary | None = None
_FORK_GT_CONFIG: VerificationGTConfig | None = None
_FORK_MAX_REPLAN_CANDIDATES: int | None = None
_FORK_BUILD_ONE: BuildOne | None = None


def _fork_task(index: int) -> _TaskOutcome:
    if (
        _FORK_SOURCE is None
        or _FORK_LIBRARY is None
        or _FORK_GT_CONFIG is None
        or _FORK_MAX_REPLAN_CANDIDATES is None
        or _FORK_BUILD_ONE is None
    ):
        raise RuntimeError("verification fork context is unavailable")
    return _task_outcome(
        _FORK_SOURCE,
        _FORK_BOUNDARY[index],
        _FORK_LIBRARY,
        _FORK_GT_CONFIG,
        _FORK_MAX_REPLAN_CANDIDATES,
        _FORK_BUILD_ONE,
    )


def _evaluate_boundary(
    source: Sop06FinalizedSource,
    boundary: tuple[Sop06AcceptedFinalRecord, ...],
    *,
    library: VerificationActionLibrary,
    gt_config: VerificationGTConfig,
    max_replan_candidates: int,
    workers: int,
    build_one: BuildOne,
) -> tuple[_TaskOutcome, ...]:
    if not boundary:
        raise ValueError("verification task boundary must be non-empty")
    prepared = source.prepare_boundary(boundary)
    if workers == 1 or len(boundary) == 1:
        return tuple(
            _task_outcome(
                prepared,
                accepted,
                library,
                gt_config,
                max_replan_candidates,
                build_one,
            )
            for accepted in boundary
        )
    if multiprocessing.get_start_method(allow_none=True) not in (None, "fork"):
        raise ValueError("parallel verification release requires fork")
    global _FORK_SOURCE
    global _FORK_BOUNDARY
    global _FORK_LIBRARY
    global _FORK_GT_CONFIG
    global _FORK_MAX_REPLAN_CANDIDATES
    global _FORK_BUILD_ONE
    previous = (
        _FORK_SOURCE,
        _FORK_BOUNDARY,
        _FORK_LIBRARY,
        _FORK_GT_CONFIG,
        _FORK_MAX_REPLAN_CANDIDATES,
        _FORK_BUILD_ONE,
    )
    _FORK_SOURCE = prepared
    _FORK_BOUNDARY = boundary
    _FORK_LIBRARY = library
    _FORK_GT_CONFIG = gt_config
    _FORK_MAX_REPLAN_CANDIDATES = max_replan_candidates
    _FORK_BUILD_ONE = build_one
    limit = min(workers, len(boundary))
    executor: ProcessPoolExecutor | None = None
    futures: dict[Future[_TaskOutcome], int] = {}
    try:
        executor = ProcessPoolExecutor(
            max_workers=limit,
            mp_context=multiprocessing.get_context("fork"),
        )
        for index in range(limit):
            futures[executor.submit(_fork_task, index)] = index
        ready: dict[int, _TaskOutcome] = {}
        outcomes: list[_TaskOutcome] = []
        next_submit = limit
        next_emit = 0
        while futures:
            completed, _ = wait(
                tuple(futures),
                return_when=FIRST_COMPLETED,
            )
            for future in completed:
                if future.exception() is not None:
                    future.result()
            for future in completed:
                index = futures.pop(future)
                ready[index] = future.result()
            while next_emit in ready:
                outcomes.append(ready.pop(next_emit))
                next_emit += 1
            while (
                next_submit < len(boundary)
                and len(futures) + len(ready) < limit
            ):
                future = executor.submit(_fork_task, next_submit)
                futures[future] = next_submit
                next_submit += 1
        executor.shutdown(wait=True)
        executor = None
        if ready or len(outcomes) != len(boundary):
            raise RuntimeError("verification worker result ordering is incomplete")
        return tuple(outcomes)
    except BaseException:
        if executor is not None:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        (
            _FORK_SOURCE,
            _FORK_BOUNDARY,
            _FORK_LIBRARY,
            _FORK_GT_CONFIG,
            _FORK_MAX_REPLAN_CANDIDATES,
            _FORK_BUILD_ONE,
        ) = previous


def _value_records(
    outcomes: Sequence[_TaskOutcome],
) -> list[dict[str, object]]:
    return [
        {
            "sample_id": sample.sample_id,
            "action_id": sample.verification_action_id,
            "value_target": sample.value_target,
            "useful_target": sample.useful_target,
        }
        for outcome in outcomes
        for sample in outcome.samples
    ]


def _task_summary(
    outcomes: tuple[_TaskOutcome, ...],
    *,
    request_identity: str,
    shard_index: int,
    verification_semantic_digest: str | None,
) -> dict[str, object]:
    accepted = tuple(outcome for outcome in outcomes if outcome.accepted)
    rejected = tuple(outcome for outcome in outcomes if not outcome.accepted)
    revaluation_records = [
        dict(record)
        for outcome in accepted
        for record in outcome.revaluation_records
    ]
    return {
        "version": VERIFICATION_RELEASE_VERSION,
        "request_identity": request_identity,
        "shard_index": shard_index,
        "task_count": len(outcomes),
        "task_ids": [outcome.task_id for outcome in outcomes],
        "mother_ids": [outcome.mother_id for outcome in outcomes],
        "accepted_group_count": len(accepted),
        "rejected_task_count": len(rejected),
        "sample_count": sum(len(outcome.samples) for outcome in accepted),
        "sampled_child_world_ids": [
            outcome.sampled_child_world_id for outcome in accepted
        ],
        "rejections": [
            {
                "task_id": outcome.task_id,
                "mother_id": outcome.mother_id,
                "reason": outcome.rejection_reason,
                "detail": outcome.rejection_detail,
            }
            for outcome in rejected
        ],
        "verification_semantic_digest": verification_semantic_digest,
        "value_records": _value_records(outcomes),
        "revaluation_records": revaluation_records,
        "revaluation_records_digest": hashlib.sha256(
            b"verification_release_revaluation_records_v2\0"
            + _canonical_json(revaluation_records)
        ).hexdigest(),
    }


def _write_task_shard(
    outcomes: tuple[_TaskOutcome, ...],
    output_path: Path,
    *,
    request_identity: str,
    shard_index: int,
    grid,
    library: VerificationActionLibrary,
) -> dict[str, object]:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite task shard: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.parent / f".{output_path.name}.staging"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"stale verification shard staging path: {staging}")
    staging.mkdir()
    try:
        samples = tuple(
            sample for outcome in outcomes for sample in outcome.samples
        )
        semantic_digest: str | None = None
        if samples:
            write_verification_shard(
                samples,
                staging / _DATA,
                grid=grid,
                library=library,
                shard_index=shard_index,
                expected_sample_count=len(samples),
            )
            data_summary = _read_json(staging / _DATA / "summary.json")
            if not isinstance(data_summary, Mapping):
                raise ValueError("verification data summary must be a mapping")
            semantic_digest = str(data_summary["semantic_digest"])
        summary = _task_summary(
            outcomes,
            request_identity=request_identity,
            shard_index=shard_index,
            verification_semantic_digest=semantic_digest,
        )
        summary_bytes = _json_file(summary)
        (staging / _TASK_SUMMARY).write_bytes(summary_bytes)
        (staging / _COMPLETE).write_bytes(
            _json_file(
                {
                    "version": VERIFICATION_RELEASE_VERSION,
                    "request_identity": request_identity,
                    "shard_index": shard_index,
                    "task_summary_sha256": hashlib.sha256(
                        summary_bytes
                    ).hexdigest(),
                }
            )
        )
        atomic_rename_noreplace(staging, output_path)
        return summary
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _load_task_checkpoint(
    root: Path,
    *,
    request_identity: str,
    shard_index: int,
    expected_task_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"incomplete verification task shard: {root}")
    actual = {path.name for path in root.iterdir()}
    if actual not in (
        {_TASK_SUMMARY, _COMPLETE},
        {_TASK_SUMMARY, _COMPLETE, _DATA},
    ):
        raise ValueError("verification task shard file set is invalid")
    summary = _read_json(root / _TASK_SUMMARY)
    complete = _read_json(root / _COMPLETE)
    if not isinstance(summary, dict) or not isinstance(complete, dict):
        raise ValueError("verification task shard metadata must be mappings")
    if (
        summary.get("version") != VERIFICATION_RELEASE_VERSION
        or complete.get("version") != VERIFICATION_RELEASE_VERSION
        or summary.get("request_identity") != request_identity
        or complete.get("request_identity") != request_identity
        or summary.get("shard_index") != shard_index
        or complete.get("shard_index") != shard_index
    ):
        raise ValueError("verification task shard identity differs")
    summary_sha256 = _sha256_file(root / _TASK_SUMMARY)
    if complete.get("task_summary_sha256") != summary_sha256:
        raise ValueError("verification task shard summary checksum differs")
    task_ids = summary.get("task_ids")
    mother_ids = summary.get("mother_ids")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(value, str) or not value for value in task_ids)
        or len(set(task_ids)) != len(task_ids)
        or not isinstance(mother_ids, list)
        or len(mother_ids) != len(task_ids)
        or any(not isinstance(value, str) or not value for value in mother_ids)
    ):
        raise ValueError("verification task shard task boundary is invalid")
    if expected_task_ids is not None and tuple(task_ids) != expected_task_ids:
        raise ValueError("verification task shard fixed boundary differs")
    task_count = summary.get("task_count")
    accepted_count = summary.get("accepted_group_count")
    rejected_count = summary.get("rejected_task_count")
    sample_count = summary.get("sample_count")
    if (
        task_count != len(task_ids)
        or not isinstance(accepted_count, int)
        or not isinstance(rejected_count, int)
        or accepted_count < 0
        or rejected_count < 0
        or accepted_count + rejected_count != task_count
        or sample_count != accepted_count * len(CANONICAL_ACTION_IDS)
    ):
        raise ValueError("verification task shard counts are inconsistent")
    data_exists = _DATA in actual
    semantic_digest = summary.get("verification_semantic_digest")
    if data_exists != (sample_count > 0):
        raise ValueError("verification task shard data presence differs from count")
    if data_exists:
        data_summary = _read_json(root / _DATA / "summary.json")
        if (
            not isinstance(data_summary, Mapping)
            or data_summary.get("expected_sample_count") != sample_count
            or data_summary.get("shard_index") != shard_index
            or data_summary.get("semantic_digest") != semantic_digest
        ):
            raise ValueError("verification nested shard summary differs")
    elif semantic_digest is not None:
        raise ValueError("empty verification task shard has a data digest")
    records = summary.get("value_records")
    revaluation_records = summary.get("revaluation_records")
    world_ids = summary.get("sampled_child_world_ids")
    rejections = summary.get("rejections")
    if (
        not isinstance(records, list)
        or len(records) != sample_count
        or not isinstance(revaluation_records, list)
        or len(revaluation_records) != sample_count
        or not isinstance(world_ids, list)
        or len(world_ids) != accepted_count
        or any(not isinstance(value, str) or not value for value in world_ids)
        or not isinstance(rejections, list)
        or len(rejections) != rejected_count
    ):
        raise ValueError("verification task shard audit records differ from counts")
    normalized_revaluation = [
        _task_revaluation_row(record) for record in revaluation_records
    ]
    expected_revaluation_digest = hashlib.sha256(
        b"verification_release_revaluation_records_v2\0"
        + _canonical_json(normalized_revaluation)
    ).hexdigest()
    if summary.get("revaluation_records_digest") != expected_revaluation_digest:
        raise ValueError("revaluation records digest differs")
    if len(
        {str(record["sample_id"]) for record in normalized_revaluation}
    ) != sample_count:
        raise ValueError("revaluation record sample IDs are not unique")
    value_identity = [
        (record.get("sample_id"), record.get("action_id"))
        for record in records
        if isinstance(record, Mapping)
    ]
    revaluation_identity = [
        (record["sample_id"], record["action_id"])
        for record in normalized_revaluation
    ]
    task_to_mother = dict(zip(task_ids, mother_ids, strict=True))
    revaluation_task_counts = Counter(
        str(record["task_id"]) for record in normalized_revaluation
    )
    if (
        len(value_identity) != sample_count
        or value_identity != revaluation_identity
        or (
            sample_count > 0
            and len({record["split"] for record in normalized_revaluation}) != 1
        )
        or len(revaluation_task_counts) != accepted_count
        or set(revaluation_task_counts) - set(task_ids)
        or any(
            count != len(CANONICAL_ACTION_IDS)
            for count in revaluation_task_counts.values()
        )
        or any(
            task_to_mother[record["task_id"]] != record["mother_id"]
            for record in normalized_revaluation
        )
    ):
        raise ValueError("revaluation records differ from value records")
    summary["revaluation_records"] = normalized_revaluation
    return summary


def _manifest_from_shards(
    *,
    source: Sop06FinalizedSource,
    request_identity: str,
    request: VerificationReleaseRequest,
    shard_summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    task_ids = [
        str(task_id)
        for summary in shard_summaries
        for task_id in summary["task_ids"]
    ]
    mother_ids = [
        str(mother_id)
        for summary in shard_summaries
        for mother_id in summary["mother_ids"]
    ]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("verification release task IDs are not unique")
    if len(set(mother_ids)) != len(mother_ids):
        raise ValueError("verification release mother IDs are not unique")
    records = [
        record
        for summary in shard_summaries
        for record in summary["value_records"]
    ]
    if any(
        not isinstance(record, Mapping)
        or set(record)
        != {"sample_id", "action_id", "value_target", "useful_target"}
        or not isinstance(record["sample_id"], str)
        or not record["sample_id"]
        or record["action_id"] not in CANONICAL_ACTION_IDS
        or record["useful_target"] not in (0, 1)
        for record in records
    ):
        raise ValueError("verification release value records are invalid")
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("verification release sample IDs are not unique")
    values = np.asarray(
        [float(record["value_target"]) for record in records],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("verification release values must be finite")
    action_counts: Counter[str] = Counter(
        str(record["action_id"]) for record in records
    )
    positive_counts: Counter[str] = Counter(
        str(record["action_id"])
        for record in records
        if int(record["useful_target"]) == 1
    )
    negative_counts = {
        action_id: action_counts[action_id] - positive_counts[action_id]
        for action_id in CANONICAL_ACTION_IDS
    }
    accepted_count = sum(
        int(summary["accepted_group_count"]) for summary in shard_summaries
    )
    rejected_count = sum(
        int(summary["rejected_task_count"]) for summary in shard_summaries
    )
    rejection_counts: Counter[str] = Counter(
        str(row["reason"])
        for summary in shard_summaries
        for row in summary["rejections"]
    )
    if any(
        action_counts[action_id] != accepted_count
        for action_id in CANONICAL_ACTION_IDS
    ):
        raise ValueError("verification release action groups are incomplete")
    world_ids = [
        str(world_id)
        for summary in shard_summaries
        for world_id in summary["sampled_child_world_ids"]
    ]
    if len(world_ids) != accepted_count or len(set(world_ids)) != len(world_ids):
        raise ValueError("verification release sampled child IDs are invalid")
    quantile_levels = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    quantiles = (
        np.quantile(values, quantile_levels)
        if values.size
        else np.full(len(quantile_levels), np.nan, dtype=np.float64)
    )
    value_quantiles = (
        {
            str(level): float(value)
            for level, value in zip(
                quantile_levels,
                quantiles,
                strict=True,
            )
        }
        if values.size
        else None
    )
    return {
        "version": VERIFICATION_RELEASE_VERSION,
        "request_identity": request_identity,
        "source_family": request.source_family,
        "source_mode": request.source_mode,
        "source_publication_semantic_digest": (
            source.source_publication_semantic_digest
        ),
        "final_release_identity": source.final_release_identity,
        "split": request.split,
        "task_count": len(task_ids),
        "accepted_group_count": accepted_count,
        "rejected_task_count": rejected_count,
        "sample_count": len(records),
        "shard_count": len(shard_summaries),
        "groups_per_shard": request.groups_per_shard,
        "max_replan_candidates": request.max_replan_candidates,
        "task_ids_digest": hashlib.sha256(
            b"verification_release_task_ids_v2\0"
            + _canonical_json(task_ids)
        ).hexdigest(),
        "mother_ids_digest": hashlib.sha256(
            b"verification_release_mother_ids_v2\0"
            + _canonical_json(mother_ids)
        ).hexdigest(),
        "sampled_child_world_id_count": sum(
            len(summary["sampled_child_world_ids"])
            for summary in shard_summaries
        ),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "positive_count": int(
            sum(int(record["useful_target"]) for record in records)
        ),
        "negative_count": int(
            len(records)
            - sum(int(record["useful_target"]) for record in records)
        ),
        "action_counts": {
            action_id: action_counts[action_id]
            for action_id in CANONICAL_ACTION_IDS
        },
        "per_action_positive_counts": {
            action_id: positive_counts[action_id]
            for action_id in CANONICAL_ACTION_IDS
        },
        "per_action_negative_counts": negative_counts,
        "value_quantiles": value_quantiles,
        "shards": [
            {
                "shard_index": int(summary["shard_index"]),
                "relative_root": (
                    f"{_SHARDS}/shard-{int(summary['shard_index']):05d}"
                ),
                "task_count": int(summary["task_count"]),
                "accepted_group_count": int(
                    summary["accepted_group_count"]
                ),
                "rejected_task_count": int(summary["rejected_task_count"]),
                "sample_count": int(summary["sample_count"]),
                "verification_semantic_digest": summary[
                    "verification_semantic_digest"
                ],
                "revaluation_records_digest": summary[
                    "revaluation_records_digest"
                ],
            }
            for summary in shard_summaries
        ],
    }


def _load_verification_release(
    input_dir: str | Path,
    *,
    allow_inprogress: bool,
) -> LoadedVerificationRelease:
    root = Path(input_dir)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"verification release is not a real directory: {root}")
    if {path.name for path in root.iterdir()} != _ROOT_FILES:
        raise ValueError("verification release root file set is invalid")
    request_document = _read_json(root / _REQUEST)
    manifest = _read_json(root / _MANIFEST)
    checksums = _read_json(root / _CHECKSUMS)
    complete = _read_json(root / _COMPLETE)
    if not all(
        isinstance(value, dict)
        for value in (request_document, manifest, checksums, complete)
    ):
        raise ValueError("verification release metadata must be mappings")
    request_payload = request_document.get("request")
    if not isinstance(request_payload, Mapping):
        raise ValueError("verification release request payload is invalid")
    request_identity = request_document.get("request_identity")
    expected_request_identity = hashlib.sha256(
        b"verification_release_request_v2\0"
        + _canonical_json(request_payload)
    ).hexdigest()
    if (
        request_document.get("version") != VERIFICATION_RELEASE_VERSION
        or request_payload.get("version") != VERIFICATION_RELEASE_VERSION
        or manifest.get("version") != VERIFICATION_RELEASE_VERSION
        or complete.get("version") != VERIFICATION_RELEASE_VERSION
        or not isinstance(request_identity, str)
        or not request_identity
        or request_identity != expected_request_identity
        or manifest.get("request_identity") != request_identity
        or complete.get("request_identity") != request_identity
    ):
        raise ValueError("verification release identity differs")
    expected_checksums = {
        _REQUEST: _sha256_file(root / _REQUEST),
        _MANIFEST: _sha256_file(root / _MANIFEST),
    }
    if checksums != expected_checksums:
        raise ValueError("verification release metadata checksums differ")
    manifest_digest = hashlib.sha256(
        b"verification_release_manifest_v2\0" + _canonical_json(manifest)
    ).hexdigest()
    scalar_names = (
        "task_count",
        "accepted_group_count",
        "rejected_task_count",
        "sample_count",
        "shard_count",
    )
    if any(
        not isinstance(manifest.get(name), int) or manifest[name] < 0
        for name in scalar_names
    ):
        raise ValueError("verification release manifest counts are invalid")
    if (
        complete.get("manifest_digest") != manifest_digest
        or complete.get("task_count") != manifest["task_count"]
        or complete.get("sample_count") != manifest["sample_count"]
        or complete.get("shard_count") != manifest["shard_count"]
        or manifest["accepted_group_count"] * len(CANONICAL_ACTION_IDS)
        != manifest["sample_count"]
        or manifest["accepted_group_count"] + manifest["rejected_task_count"]
        != manifest["task_count"]
    ):
        raise ValueError("verification release completion counts differ")
    descriptors = manifest.get("shards")
    if (
        not isinstance(descriptors, list)
        or len(descriptors) != manifest["shard_count"]
    ):
        raise ValueError("verification release shard descriptors are invalid")
    observed_tasks = 0
    observed_samples = 0
    for index, descriptor in enumerate(descriptors):
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("shard_index") != index
            or descriptor.get("relative_root")
            != f"{_SHARDS}/shard-{index:05d}"
        ):
            raise ValueError("verification release shard ordering differs")
        summary = _load_task_checkpoint(
            root / str(descriptor["relative_root"]),
            request_identity=request_identity,
            shard_index=index,
        )
        if any(
            descriptor.get(name) != summary.get(name)
            for name in (
                "task_count",
                "accepted_group_count",
                "rejected_task_count",
                "sample_count",
                "verification_semantic_digest",
                "revaluation_records_digest",
            )
        ):
            raise ValueError("verification release shard descriptor differs")
        observed_tasks += int(summary["task_count"])
        observed_samples += int(summary["sample_count"])
    if (
        observed_tasks != manifest["task_count"]
        or observed_samples != manifest["sample_count"]
    ):
        raise ValueError("verification release shard totals differ")
    if not allow_inprogress and root.name.startswith("."):
        raise ValueError("in-progress verification release is not publishable input")
    return LoadedVerificationRelease(
        root=root,
        request_identity=request_identity,
        split=str(manifest["split"]),
        task_count=int(manifest["task_count"]),
        accepted_group_count=int(manifest["accepted_group_count"]),
        rejected_task_count=int(manifest["rejected_task_count"]),
        sample_count=int(manifest["sample_count"]),
        shard_count=int(manifest["shard_count"]),
        manifest_digest=manifest_digest,
        request=dict(request_payload),
        manifest=dict(manifest),
    )


def load_verification_release(
    input_dir: str | Path,
) -> LoadedVerificationRelease:
    """Load a complete release without decompressing verification payloads."""

    return _load_verification_release(input_dir, allow_inprogress=False)


def load_verification_revaluation_records(
    input_dir: str | Path,
) -> tuple[VerificationRevaluationRecord, ...]:
    """Load authenticated label-side scalars without decompressing model inputs."""

    release = load_verification_release(input_dir)
    records: list[VerificationRevaluationRecord] = []
    descriptors = release.manifest.get("shards")
    if not isinstance(descriptors, list):
        raise ValueError("verification release shard descriptors are invalid")
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, Mapping):
            raise ValueError("verification release shard descriptor is invalid")
        summary = _load_task_checkpoint(
            release.root / str(descriptor["relative_root"]),
            request_identity=release.request_identity,
            shard_index=index,
        )
        for raw in summary["revaluation_records"]:
            row = _task_revaluation_row(raw)
            records.append(
                VerificationRevaluationRecord(
                    release_request_identity=release.request_identity,
                    split=str(row["split"]),
                    task_id=str(row["task_id"]),
                    mother_id=str(row["mother_id"]),
                    sample_id=str(row["sample_id"]),
                    ranking_group_id=str(row["ranking_group_id"]),
                    action_id=str(row["action_id"]),
                    realized_execute_loss=float(
                        row["realized_execute_loss"]
                    ),
                    unclipped_best_policy_loss=(
                        None
                        if row["unclipped_best_policy_loss"] is None
                        else float(row["unclipped_best_policy_loss"])
                    ),
                    action_cost=float(row["action_cost"]),
                    original_reject_cost=float(
                        row["original_reject_cost"]
                    ),
                )
            )
    if len(records) != release.sample_count:
        raise ValueError("verification revaluation record count differs")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("verification revaluation sample IDs are not unique")
    if records and (
        {record.split for record in records} != {release.split}
        or len({record.original_reject_cost for record in records}) != 1
    ):
        raise ValueError("verification revaluation release context differs")
    return tuple(records)


def _result(
    loaded: LoadedVerificationRelease,
    *,
    reused_shard_count: int,
) -> VerificationReleaseResult:
    return VerificationReleaseResult(
        output_dir=loaded.root,
        split=loaded.split,
        task_count=loaded.task_count,
        accepted_group_count=loaded.accepted_group_count,
        rejected_task_count=loaded.rejected_task_count,
        sample_count=loaded.sample_count,
        shard_count=loaded.shard_count,
        reused_shard_count=reused_shard_count,
        manifest_digest=loaded.manifest_digest,
    )


def publish_verification_release(
    request: VerificationReleaseRequest,
    *,
    build_one: BuildOne = _default_build_one,
    source_loader: SourceLoader = _load_source,
    progress_callback: ProgressCallback | None = None,
) -> VerificationReleaseResult:
    """Evaluate each fixed SOP5 task once and publish resumable shards."""

    source = source_loader(request)
    if not isinstance(source, Sop06FinalizedSource) and any(
        not hasattr(source, name)
        for name in (
            "accepted",
            "base_config",
            "source_mode",
            "source_publication_semantic_digest",
            "final_release_identity",
            "prepare_boundary",
        )
    ):
        raise TypeError("source_loader result does not implement finalized source API")
    if source.source_mode != request.source_mode:
        raise ValueError("verification source mode differs from request")
    request_document = _request_document(request, source=source)
    request_identity = str(request_document["request_identity"])
    accepted = tuple(source.accepted)
    if request.max_tasks is not None:
        accepted = accepted[: request.max_tasks]
    if not accepted:
        raise ValueError("verification release contains no source tasks")
    if any(item.split != request.split for item in accepted):
        raise ValueError("verification source split differs from request")
    if len({item.scenario_id for item in accepted}) != len(accepted):
        raise ValueError("verification source task IDs are not unique")
    if len({item.mother_id for item in accepted}) != len(accepted):
        raise ValueError("verification source mother IDs are not unique")
    if (
        any(item.source_index < 0 for item in accepted)
        or len({item.source_index for item in accepted}) != len(accepted)
    ):
        raise ValueError("verification source indices are invalid")
    library = load_verification_actions(request.actions_config_path)
    gt_config = load_verification_gt_config(request.gt_config_path)
    grid = build_grid_spec(dict(source.base_config))
    boundaries = tuple(
        accepted[start : start + request.groups_per_shard]
        for start in range(0, len(accepted), request.groups_per_shard)
    )
    output = request.output_dir
    in_progress = output.parent / f".{output.name}.inprogress"
    if output.exists() or output.is_symlink():
        if in_progress.exists() or in_progress.is_symlink():
            raise ValueError("complete and in-progress verification releases coexist")
        loaded = load_verification_release(output)
        if loaded.request_identity != request_identity:
            raise ValueError("existing verification release differs from request")
        return _result(loaded, reused_shard_count=loaded.shard_count)
    if in_progress.exists() or in_progress.is_symlink():
        if not in_progress.is_dir() or in_progress.is_symlink():
            raise ValueError("in-progress verification release is not a real directory")
        actual = {path.name for path in in_progress.iterdir()}
        if (
            not {_REQUEST, _SHARDS}.issubset(actual)
            or actual - _ROOT_FILES
            or _read_json(in_progress / _REQUEST) != request_document
        ):
            raise ValueError("in-progress verification release differs from request")
        for name in (_MANIFEST, _CHECKSUMS, _COMPLETE):
            metadata_path = in_progress / name
            if (
                name in actual
                and (
                    not metadata_path.is_file()
                    or metadata_path.is_symlink()
                )
            ):
                raise ValueError(
                    "in-progress verification metadata is not a real file"
                )
        if actual == _ROOT_FILES:
            try:
                staged = _load_verification_release(
                    in_progress,
                    allow_inprogress=True,
                )
            except ValueError:
                # Root metadata may be truncated by interruption. Completed
                # immutable shards remain the source of truth and are checked
                # below before the root metadata is deterministically rebuilt.
                pass
            else:
                if staged.request_identity != request_identity:
                    raise ValueError("completed staging request differs")
                atomic_rename_noreplace(in_progress, output)
                return _result(
                    replace(staged, root=output),
                    reused_shard_count=staged.shard_count,
                )
    else:
        in_progress.mkdir(parents=True)
        (in_progress / _SHARDS).mkdir()
        (in_progress / _REQUEST).write_bytes(_json_file(request_document))
    shards_root = in_progress / _SHARDS
    expected_names = {
        f"shard-{index:05d}" for index in range(len(boundaries))
    }
    for name in expected_names:
        stale = shards_root / f".{name}.staging"
        if stale.exists() or stale.is_symlink():
            if not stale.is_dir() or stale.is_symlink():
                raise ValueError("verification shard staging path is not a real directory")
            shutil.rmtree(stale)
    if {path.name for path in shards_root.iterdir()} - expected_names:
        raise ValueError("in-progress verification release has unexpected shards")
    summaries: list[dict[str, object]] = []
    reused = 0
    for index, boundary in enumerate(boundaries):
        shard_path = shards_root / f"shard-{index:05d}"
        expected_ids = tuple(item.scenario_id for item in boundary)
        was_reused = shard_path.exists()
        if was_reused:
            summary = _load_task_checkpoint(
                shard_path,
                request_identity=request_identity,
                shard_index=index,
                expected_task_ids=expected_ids,
            )
            reused += 1
        else:
            outcomes = _evaluate_boundary(
                source,
                boundary,
                library=library,
                gt_config=gt_config,
                max_replan_candidates=request.max_replan_candidates,
                workers=request.workers,
                build_one=build_one,
            )
            if tuple(outcome.task_id for outcome in outcomes) != expected_ids:
                raise ValueError("verification worker task order differs")
            summary = _write_task_shard(
                outcomes,
                shard_path,
                request_identity=request_identity,
                shard_index=index,
                grid=grid,
                library=library,
            )
            del outcomes
        summaries.append(summary)
        if progress_callback is not None:
            progress_callback(index + 1, len(boundaries), was_reused)
    manifest = _manifest_from_shards(
        source=source,
        request_identity=request_identity,
        request=request,
        shard_summaries=summaries,
    )
    if manifest["task_count"] != len(accepted):
        raise ValueError("verification release task count differs from source boundary")
    manifest_digest = hashlib.sha256(
        b"verification_release_manifest_v2\0" + _canonical_json(manifest)
    ).hexdigest()
    (in_progress / _MANIFEST).write_bytes(_json_file(manifest))
    (in_progress / _CHECKSUMS).write_bytes(
        _json_file(
            {
                _REQUEST: _sha256_file(in_progress / _REQUEST),
                _MANIFEST: _sha256_file(in_progress / _MANIFEST),
            }
        )
    )
    (in_progress / _COMPLETE).write_bytes(
        _json_file(
            {
                "version": VERIFICATION_RELEASE_VERSION,
                "request_identity": request_identity,
                "task_count": manifest["task_count"],
                "sample_count": manifest["sample_count"],
                "shard_count": manifest["shard_count"],
                "manifest_digest": manifest_digest,
            }
        )
    )
    staged = _load_verification_release(in_progress, allow_inprogress=True)
    if staged.manifest_digest != manifest_digest:
        raise ValueError("staged verification release manifest differs")
    atomic_rename_noreplace(in_progress, output)
    published = load_verification_release(output)
    if published.manifest_digest != manifest_digest:
        raise ValueError("published verification release differs from staging")
    return _result(published, reused_shard_count=reused)


__all__ = (
    "LoadedVerificationRelease",
    "VERIFICATION_RELEASE_VERSION",
    "VerificationRevaluationRecord",
    "VerificationReleaseRequest",
    "VerificationReleaseResult",
    "load_verification_release",
    "load_verification_revaluation_records",
    "publish_verification_release",
)
