"""Atomic publication and deterministic collection helpers for SOP05R TEB v2."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np

from src.contracts import build_grid_spec, save_dataclass
from src.planning.verification_actions import load_verification_actions
from src.utils.atomic_publish import atomic_rename_noreplace
from src.utils.config import load_config
from src.utils.seeding import derive_seed

from .event_target_motion_shard import (
    LoadedEventTargetMotionShard,
    load_event_target_motion_shard,
    write_event_target_motion_shard,
)
from .anchored_human_placement import (
    PLACEMENT_SELECTION_MODES,
    solve_anchored_human_placement,
)
from .sop05r_contracts import (
    SOP05R_TEB_COMPLETION_MARKER_VERSION,
    SOP05R_TEB_GENERATOR_VERSION,
    SOP05R_TEB_MANIFEST_VERSION,
    SOP05R_TEB_RUN_VERSION,
    SOP05R_TEB_SUMMARY_VERSION,
    SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION,
    Sop05rTebConfig,
    load_sop05r_teb_config,
)
from .sop05r_teb_event_sampler import (
    Sop05rTebMotherCandidate,
    build_sop05r_teb_mother,
)
from .sop05r_teb_long40_inputs import (
    Sop05rTebLong40InputError,
    load_sop05r_teb_long40_inputs,
)
from .sop05r_teb_output_loader import (
    SOP05R_TEB_EMPTY_TARGET_MOTION_VERSION,
    SOP05R_TEB_EVENT_ROW_VERSION,
    LoadedSop05rTebOutput,
    compute_sop05r_teb_publication_semantic_digest,
    load_sop05r_teb_output,
)
from .sop05r_teb_templates import canonical_sop05r_teb_base_state_digest
from .sop05r_teb_templates import iter_sop05r_teb_task_templates
from .sop05r_teb_trajectory_store import (
    load_sop05r_teb_trajectory_store,
    publish_sop05r_teb_trajectory_store,
)


class Sop05rTebRunError(ValueError):
    """Raised when a v2 run violates publication or quota invariants."""


Sop05rTebProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class Sop05rTebRunRequest:
    sop03_root: Path
    long40_human_artifact: Path
    split: str
    base_config_path: Path
    generator_config_path: Path
    verification_action_config_path: Path
    output_dir: Path
    seed: int
    accepted_quota: int | None
    max_base_states: int
    checksum_workers: int
    workers: int
    git_executable: Path
    base_state_start: int = 0
    exclude_existing_output: Path | None = None
    resume_staging_root: Path | None = None
    placement_selection_mode: str = "seen_first"


@dataclass(frozen=True)
class Sop05rTebRunResult:
    output_dir: Path
    run_state: str
    exit_code: int
    accepted_count: int
    requested_count: int
    publication_semantic_digest: str
    generation_summary: Mapping[str, object]
    complete: bool


@dataclass(frozen=True)
class _BaseStateGenerationContext:
    base_config: Mapping[str, object]
    teb_config: Sop05rTebConfig
    snippets: Sequence[object]
    seed: int
    split: str
    placement_selection_mode: str


@dataclass(frozen=True)
class _TemplateGenerationResult:
    mother: Sop05rTebMotherCandidate | None
    counters: Mapping[str, int]
    rejections: Mapping[str, int]
    m5_candidate_counts: Mapping[str, int]
    m5_candidate_rejections: Mapping[str, int]


@dataclass(frozen=True)
class _BaseStateGenerationResult:
    base_rank: int
    templates: tuple[_TemplateGenerationResult, ...]


_BASE_STATE_GENERATION_CONTEXT: _BaseStateGenerationContext | None = None


def _generate_base_state_result(
    indexed_state_pair: object,
) -> _BaseStateGenerationResult:
    context = _BASE_STATE_GENERATION_CONTEXT
    if context is None:
        raise RuntimeError("base-state generation context is not initialized")
    base_rank, state_pair = indexed_state_pair
    base_state, oracle_context = state_pair
    base_seed = derive_seed(
        context.seed,
        context.split,
        base_state.state_id,
        base_rank,
    )
    max_snippets = context.teb_config.generation.max_target_snippets_per_template
    template_results: list[_TemplateGenerationResult] = []
    for template_evaluation in iter_sop05r_teb_task_templates(
        base_state=base_state,
        oracle_context=oracle_context,
        base_config=context.base_config,
        teb_config=context.teb_config,
        seed=base_seed,
    ):
        counters: Counter[str] = Counter(m4_attempted=1)
        rejections: Counter[str] = Counter()
        m5_candidate_counts: Counter[str] = Counter()
        m5_candidate_rejections: Counter[str] = Counter()
        template = template_evaluation.template
        if template is None:
            rejections[
                template_evaluation.rejection_reason or "m4_rejected"
            ] += 1
            template_results.append(
                _TemplateGenerationResult(
                    mother=None,
                    counters=dict(counters),
                    rejections=dict(rejections),
                    m5_candidate_counts={},
                    m5_candidate_rejections={},
                )
            )
            continue
        counters["m4_accepted"] += 1
        if not context.snippets:
            rejections["human_snippet_library_empty"] += 1
            template_results.append(
                _TemplateGenerationResult(
                    mother=None,
                    counters=dict(counters),
                    rejections=dict(rejections),
                    m5_candidate_counts={},
                    m5_candidate_rejections={},
                )
            )
            continue
        rng_seed = derive_seed(
            base_seed,
            template.template_id,
            context.teb_config.digest,
        )
        start = rng_seed % len(context.snippets)
        snippet_order = tuple(
            context.snippets[(start + offset) % len(context.snippets)]
            for offset in range(min(max_snippets, len(context.snippets)))
        )
        accepted_mother: Sop05rTebMotherCandidate | None = None
        for snippet_rank, snippet in enumerate(snippet_order):
            counters["m5_attempted"] += 1
            placement = solve_anchored_human_placement(
                task_template=template,
                snippet=snippet,
                base_state=base_state,
                oracle_context=oracle_context,
                base_config=context.base_config,
                teb_config=context.teb_config,
                seed=derive_seed(rng_seed, snippet.snippet_id, snippet_rank),
                selection_mode=context.placement_selection_mode,
            )
            m5_candidate_counts.update(placement.candidate_counts)
            m5_candidate_rejections.update(placement.rejection_counts)
            if placement.result is None:
                rejections[placement.rejection_reason or "m5_rejected"] += 1
                continue
            counters["m5_accepted"] += 1
            counters["m6_attempted"] += 1
            mother_evaluation = build_sop05r_teb_mother(
                base_config=context.base_config,
                source_base_state=base_state,
                source_oracle_context=oracle_context,
                teb_config=context.teb_config,
                task_template=template,
                placement_result=placement.result,
                snippet=snippet,
                seed=derive_seed(rng_seed, "mother", snippet.snippet_id),
            )
            if mother_evaluation.mother is None:
                rejections[
                    mother_evaluation.rejection_reason or "m6_rejected"
                ] += 1
                continue
            counters["m6_accepted"] += 1
            accepted_mother = mother_evaluation.mother
            break
        if accepted_mother is None:
            counters["templates_without_mother"] += 1
        template_results.append(
            _TemplateGenerationResult(
                mother=accepted_mother,
                counters=dict(counters),
                rejections=dict(rejections),
                m5_candidate_counts=dict(m5_candidate_counts),
                m5_candidate_rejections=dict(m5_candidate_rejections),
            )
        )
    return _BaseStateGenerationResult(
        base_rank=int(base_rank),
        templates=tuple(template_results),
    )


def _base_state_worker(indexed_state_pair: object) -> object:
    return _generate_base_state_result(indexed_state_pair)


def _restore_worker_array_immutability(value: object) -> object:
    """Restore array flags that NumPy pickle transport does not preserve."""

    seen: set[int] = set()

    def visit(item: object) -> None:
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(item, np.ndarray):
            item.setflags(write=False)
        elif is_dataclass(item):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (tuple, list)):
            for nested in item:
                visit(nested)

    visit(value)
    return value


def _ordered_base_state_results(
    indexed_state_pairs: Sequence[object],
    *,
    context: object,
    workers: int,
) -> Iterator[object]:
    """Yield independent base-state results in source rank order."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not indexed_state_pairs:
        return
    global _BASE_STATE_GENERATION_CONTEXT
    previous_context = _BASE_STATE_GENERATION_CONTEXT
    _BASE_STATE_GENERATION_CONTEXT = context
    try:
        if workers == 1:
            for indexed_state_pair in indexed_state_pairs:
                yield _generate_base_state_result(indexed_state_pair)
            return
        process_context = multiprocessing.get_context("fork")
        pool = process_context.Pool(
            processes=min(workers, len(indexed_state_pairs)),
        )
        try:
            for result in pool.imap(
                _base_state_worker, indexed_state_pairs, chunksize=1
            ):
                yield _restore_worker_array_immutability(result)
        finally:
            pool.terminate()
            pool.join()
    finally:
        _BASE_STATE_GENERATION_CONTEXT = previous_context


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
        raise Sop05rTebRunError("SOP05R TEB run evidence must be canonical JSON") from exc


def _canonical_copy(value: object) -> object:
    return json.loads(_canonical_json(value).decode("ascii"))


def _json_file(value: object) -> bytes:
    return _canonical_json(value) + b"\n"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _event_row(
    mother: Sop05rTebMotherCandidate,
    *,
    row_index: int,
    decision_state_digest: str,
) -> dict[str, object]:
    event = mother.event
    record = mother.trajectory_record
    row: dict[str, object] = {
        "row_version": SOP05R_TEB_EVENT_ROW_VERSION,
        "row_index": row_index,
        "event_id": event.generated_event_id,
        "event_kind": event.event_kind,
        "world_id": event.world.world_id,
        "source_base_state_id": record.source_base_state_id,
        "decision_state_id": record.decision_state_id,
        "template_id": record.template_id,
        "trajectory_id": record.nominal_trajectory.trajectory_id,
        "decision_state_file": f"{record.decision_state_id}.npz",
        "decision_state_digest": decision_state_digest,
        "visibility_sequence": [
            bool(value) for value in event.visibility_sequence
        ],
        "target_visibility_history": [
            bool(value) for value in event.target_visibility_history
        ],
        "conflict_time_s": float(event.conflict_time_s),
        "conflict_index": int(event.conflict_index),
        "target_provenance": _canonical_copy(event.target.provenance),
        "first_collision_time_after_decision_s": (
            mother.collision.first_collision_time_after_decision_s
        ),
        "collision_point_xy": [
            float(value) for value in mother.collision.collision_point_xy
        ],
        "occlusion_witness": {
            "time_s": mother.placement_result.witness.time_s,
            "sample_index": mother.placement_result.witness.sample_index,
            "occluder_id": (
                mother.placement_result.witness.blocking_occluder_id
            ),
        },
    }
    row["record_semantic_digest"] = _sha256(
        b"sop05r_teb_event_row_v1\0" + _canonical_json(row)
    )
    return row


def _empty_target_motion(path: Path) -> LoadedEventTargetMotionShard:
    path.mkdir()
    manifest_digest = _sha256(b"sop05r_teb_empty_target_manifest_v1\0")
    payload_digest = _sha256(b"sop05r_teb_empty_target_payload_v1\0")
    summary = {
        "version": SOP05R_TEB_EMPTY_TARGET_MOTION_VERSION,
        "record_count": 0,
        "manifest_digest": manifest_digest,
        "payload_semantic_digest": payload_digest,
    }
    (path / "empty.json").write_bytes(_json_file(summary))
    return LoadedEventTargetMotionShard(
        records=(),
        worlds={},
        manifest_digest=manifest_digest,
        payload_semantic_digest=payload_digest,
        summary=summary,
    )


def _write_checksums(staging: Path) -> None:
    files = {
        path.relative_to(staging).as_posix(): _sha256_file(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    (staging / "checksums.json").write_bytes(
        _json_file(
            {
                "manifest_version": SOP05R_TEB_MANIFEST_VERSION,
                "files": files,
            }
        )
    )


def _resume_staged_artifacts(
    staging: Path,
    *,
    selected: Sequence[Sop05rTebMotherCandidate],
    base_config: Mapping[str, object],
    requested_count: int,
    complete: bool,
) -> tuple[object, LoadedEventTargetMotionShard]:
    if {path.name for path in staging.iterdir()} != {
        "trajectory_store",
        "target_motion",
    }:
        raise Sop05rTebRunError("resume staging must contain only completed stores")
    trajectories = load_sop05r_teb_trajectory_store(
        staging / "trajectory_store",
        require_complete=complete,
    )
    expected_by_event = {
        mother.event.generated_event_id: mother for mother in selected
    }
    if tuple(record.event_id for record in trajectories.records) != tuple(
        sorted(expected_by_event)
    ):
        raise Sop05rTebRunError("resume trajectory event IDs differ from replay")
    for record in trajectories.records:
        mother = expected_by_event[record.event_id]
        expected = mother.trajectory_record
        if (
            record.source_base_state_id != expected.source_base_state_id
            or record.decision_state_id != expected.decision_state_id
            or record.nominal_trajectory.trajectory_id
            != expected.nominal_trajectory.trajectory_id
            or record.config_digest != expected.config_digest
        ):
            raise Sop05rTebRunError("resume trajectory identity differs from replay")
    target_motion = load_event_target_motion_shard(
        staging / "target_motion",
        grid=build_grid_spec(dict(base_config)),
        expected_generated_event_ids=set(expected_by_event),
        expected_base_state_ids={
            mother.trajectory_record.decision_state_id for mother in selected
        },
        expected_trajectory_ids={
            mother.trajectory_record.nominal_trajectory.trajectory_id
            for mother in selected
        },
    )
    target_by_event = {
        record.generated_event_id: record for record in target_motion.records
    }
    for event_id, mother in expected_by_event.items():
        expected = mother.event.target_motion_record
        actual = target_by_event.get(event_id)
        if actual is None or actual.record_digest != expected.record_digest:
            raise Sop05rTebRunError("resume target motion differs from replay")
    if len(trajectories.records) != len(selected) or len(target_motion.records) != len(
        selected
    ):
        raise Sop05rTebRunError("resume stores do not meet the requested count")
    return trajectories, target_motion


def publish_sop05r_teb_run(
    mothers: Sequence[Sop05rTebMotherCandidate],
    output_dir: str | Path,
    *,
    base_config: Mapping[str, object],
    requested_count: int,
    config_digest: str,
    verification_action_digest: str,
    source_evidence: Mapping[str, object],
    denominator_counts: Mapping[str, int],
    rejection_counts: Mapping[str, int],
    m5_candidate_counts: Mapping[str, int] | None = None,
    m5_candidate_rejection_counts: Mapping[str, int] | None = None,
    resume_staging_root: str | Path | None = None,
) -> LoadedSop05rTebOutput:
    """Publish selected M6 mothers, including useful partial/empty diagnostics."""

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    if isinstance(requested_count, bool) or not isinstance(requested_count, int):
        raise TypeError("requested_count must be an integer")
    if requested_count < 0:
        raise ValueError("requested_count must be nonnegative")
    selected = tuple(
        sorted(mothers, key=lambda mother: mother.event.generated_event_id)
    )
    if any(not isinstance(mother, Sop05rTebMotherCandidate) for mother in selected):
        raise TypeError("mothers must contain Sop05rTebMotherCandidate values")
    event_ids = [mother.event.generated_event_id for mother in selected]
    if len(event_ids) != len(set(event_ids)):
        raise Sop05rTebRunError("duplicate selected event ID")
    if len(selected) > requested_count:
        raise Sop05rTebRunError("accepted count exceeds requested quota")
    complete = len(selected) == requested_count
    base_snapshot = _canonical_copy(dict(base_config))
    if not isinstance(base_snapshot, dict):
        raise Sop05rTebRunError("base_config must normalize to a mapping")
    build_grid_spec(base_snapshot)
    source_snapshot = _canonical_copy(dict(source_evidence))
    if not isinstance(source_snapshot, dict):
        raise Sop05rTebRunError("source_evidence must normalize to a mapping")

    destination.parent.mkdir(parents=True, exist_ok=True)
    created_staging = resume_staging_root is None
    if created_staging:
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.sop05r-teb-stage-",
                dir=destination.parent,
            )
        )
        staging = staging_root / destination.name
        staging.mkdir()
    else:
        staging_root = Path(resume_staging_root)
        staging = staging_root / destination.name
        if (
            staging_root.is_symlink()
            or not staging_root.is_dir()
            or staging.is_symlink()
            or not staging.is_dir()
            or {path.name for path in staging_root.iterdir()} != {destination.name}
        ):
            raise Sop05rTebRunError("resume staging root layout is invalid")
    try:
        if created_staging:
            trajectories = publish_sop05r_teb_trajectory_store(
                tuple(mother.trajectory_record for mother in selected),
                staging / "trajectory_store",
                requested_count=requested_count,
                complete=complete,
            )
            if selected:
                write_event_target_motion_shard(
                    [mother.event.target_motion_record for mother in selected],
                    [mother.event.world for mother in selected],
                    staging / "target_motion",
                    grid=build_grid_spec(base_snapshot),
                )
                target_motion = load_event_target_motion_shard(
                    staging / "target_motion",
                    grid=build_grid_spec(base_snapshot),
                    expected_generated_event_ids=set(event_ids),
                )
            else:
                target_motion = _empty_target_motion(staging / "target_motion")
        else:
            trajectories, target_motion = _resume_staged_artifacts(
                staging,
                selected=selected,
                base_config=base_snapshot,
                requested_count=requested_count,
                complete=complete,
            )

        decisions_dir = staging / "decision_states"
        decisions_dir.mkdir()
        decision_digests: list[str] = []
        rows: list[dict[str, object]] = []
        for row_index, mother in enumerate(selected):
            state = mother.decision_state.base_state
            if state.state_id != mother.trajectory_record.decision_state_id:
                raise Sop05rTebRunError("decision state identity mismatch")
            digest = canonical_sop05r_teb_base_state_digest(state)
            save_dataclass(state, decisions_dir / state.state_id)
            decision_digests.append(digest)
            rows.append(
                _event_row(
                    mother,
                    row_index=row_index,
                    decision_state_digest=digest,
                )
            )
        publication_digest = compute_sop05r_teb_publication_semantic_digest(
            event_rows=rows,
            trajectory_collection_digest=trajectories.collection_semantic_digest,
            target_motion_payload_digest=target_motion.payload_semantic_digest,
            decision_state_digests=decision_digests,
            config_digest=config_digest,
        )
        manifest = {
            "manifest_version": SOP05R_TEB_MANIFEST_VERSION,
            "run_version": SOP05R_TEB_RUN_VERSION,
            "generator_algorithm_version": SOP05R_TEB_GENERATOR_VERSION,
            "trajectory_collection_version": (
                SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION
            ),
            "trajectory_collection_digest": (
                trajectories.collection_semantic_digest
            ),
            "target_motion_payload_digest": target_motion.payload_semantic_digest,
            "config_digest": config_digest,
            "verification_action_digest": verification_action_digest,
            "base_config": base_snapshot,
            "source_evidence": source_snapshot,
            "requested_count": requested_count,
            "accepted_count": len(selected),
            "event_ids": event_ids,
            "publication_semantic_digest": publication_digest,
        }
        summary = {
            "summary_version": SOP05R_TEB_SUMMARY_VERSION,
            "requested_count": requested_count,
            "accepted_count": len(selected),
            "quota_met": complete,
            "denominator_counts": {
                str(key): int(value)
                for key, value in sorted(denominator_counts.items())
            },
            "rejection_counts": {
                str(key): int(value)
                for key, value in sorted(rejection_counts.items())
            },
            "m5_candidate_counts": {
                str(key): int(value)
                for key, value in sorted((m5_candidate_counts or {}).items())
            },
            "m5_candidate_rejection_counts": {
                str(key): int(value)
                for key, value in sorted(
                    (m5_candidate_rejection_counts or {}).items()
                )
            },
            "publication_semantic_digest": publication_digest,
        }
        (staging / "events.json").write_bytes(_json_file(rows))
        (staging / "manifest.json").write_bytes(_json_file(manifest))
        (staging / "generation_summary.json").write_bytes(_json_file(summary))
        if complete:
            (staging / "COMPLETE.json").write_bytes(
                _json_file(
                    {
                        "completion_marker_version": (
                            SOP05R_TEB_COMPLETION_MARKER_VERSION
                        ),
                        "accepted_count": len(selected),
                        "requested_count": requested_count,
                        "publication_semantic_digest": publication_digest,
                    }
                )
            )
        _write_checksums(staging)
        loaded = load_sop05r_teb_output(
            staging,
            require_complete=complete,
        )
        atomic_rename_noreplace(staging, destination)
        return loaded
    finally:
        if created_staging:
            shutil.rmtree(staging_root, ignore_errors=True)
        elif not staging.exists():
            staging_root.rmdir()


def preflight_summary(request: Sop05rTebRunRequest) -> dict[str, object]:
    """Return the immutable CLI request envelope without starting generation."""

    return {
        "status": "preflight_ok",
        "producer_version": SOP05R_TEB_RUN_VERSION,
        "generator_algorithm_version": SOP05R_TEB_GENERATOR_VERSION,
        "sop03_root": str(request.sop03_root),
        "long40_human_artifact": str(request.long40_human_artifact),
        "split": request.split,
        "output_dir": str(request.output_dir),
        "seed": request.seed,
        "accepted_quota": request.accepted_quota,
        "accepted_selection": (
            "all_accepted_v1"
            if request.accepted_quota is None
            else "accepted_quota_v1"
        ),
        "max_base_states": request.max_base_states,
        "base_state_start": request.base_state_start,
        "exclude_existing_output": (
            None
            if request.exclude_existing_output is None
            else "provided"
        ),
        "resume_staging_root": (
            None if request.resume_staging_root is None else "provided"
        ),
        "placement_selection_mode": request.placement_selection_mode,
        "workers": request.workers,
        "publication_semantic_digest": None,
    }


def _progress_snapshot(
    *,
    processed_base_states: int,
    total_base_states: int,
    accepted_count: int,
    requested_count: int | None,
    started_at_s: float,
    denominator_counts: Mapping[str, int],
    rejection_counts: Mapping[str, int],
) -> dict[str, object]:
    elapsed_seconds = max(0.0, time.monotonic() - started_at_s)
    accepted_per_second = (
        accepted_count / elapsed_seconds if elapsed_seconds > 0.0 else None
    )
    estimated_remaining_seconds = (
        None
        if requested_count is None
        else (
            0.0
            if accepted_count >= requested_count
            else (
                None
                if accepted_per_second is None or accepted_per_second <= 0.0
                else (requested_count - accepted_count) / accepted_per_second
            )
        )
    )
    return {
        "progress_version": "sop05r_teb_progress_v1",
        "processed_base_states": processed_base_states,
        "total_base_states": total_base_states,
        "accepted_count": accepted_count,
        "requested_count": requested_count,
        "completion_fraction": (
            None
            if requested_count is None
            else (1.0 if requested_count == 0 else accepted_count / requested_count)
        ),
        "denominator_counts": {
            str(key): int(value) for key, value in sorted(denominator_counts.items())
        },
        "rejection_counts": {
            str(key): int(value) for key, value in sorted(rejection_counts.items())
        },
        "elapsed_seconds": elapsed_seconds,
        "accepted_per_second": accepted_per_second,
        "estimated_remaining_seconds": estimated_remaining_seconds,
    }


def _excluded_event_ids(
    request: Sop05rTebRunRequest,
    *,
    config_digest: str,
) -> tuple[frozenset[str], Mapping[str, object]]:
    """Strictly load an earlier compatible collection used for deduplication."""

    if request.exclude_existing_output is None:
        return frozenset(), {}
    try:
        loaded = load_sop05r_teb_output(
            request.exclude_existing_output,
            require_complete=True,
        )
    except ValueError as exc:
        raise Sop05rTebRunError(
            f"failed to load excluded SOP05R TEB output: {exc}"
        ) from exc
    if loaded.manifest.get("config_digest") != config_digest:
        raise Sop05rTebRunError(
            "excluded SOP05R TEB output uses a different generator config"
        )
    return (
        frozenset(event.generated_event_id for event in loaded.events),
        {
            "excluded_existing_event_count": len(loaded.events),
            "excluded_existing_publication_semantic_digest": (
                loaded.publication_semantic_digest
            ),
        },
    )


def execute_sop05r_teb_run(
    request: Sop05rTebRunRequest,
    *,
    progress_callback: Sop05rTebProgressCallback | None = None,
) -> Sop05rTebRunResult:
    """Run the deterministic M4→M6 search and publish the selected prefix."""

    if not isinstance(request, Sop05rTebRunRequest):
        raise TypeError("request must be a Sop05rTebRunRequest")
    if (
        request.accepted_quota is not None
        and (
            isinstance(request.accepted_quota, bool)
            or not isinstance(request.accepted_quota, int)
            or request.accepted_quota < 0
        )
    ) or request.max_base_states <= 0 or request.base_state_start < 0:
        raise Sop05rTebRunError("quota must be nonnegative and max_base_states positive")
    if request.placement_selection_mode not in PLACEMENT_SELECTION_MODES:
        raise Sop05rTebRunError(
            "placement_selection_mode must be one of "
            + ", ".join(PLACEMENT_SELECTION_MODES)
        )
    started_at_s = time.monotonic()
    base_config = load_config(request.base_config_path)
    grid = build_grid_spec(base_config)
    teb_config = load_sop05r_teb_config(request.generator_config_path)
    load_verification_actions(request.verification_action_config_path)
    try:
        inputs = load_sop05r_teb_long40_inputs(
            recording_root=request.sop03_root,
            long40_human_artifact=request.long40_human_artifact,
            split=request.split,
            grid=grid,
            max_base_states=request.max_base_states,
            base_state_start=request.base_state_start,
        )
    except Sop05rTebLong40InputError as exc:
        raise Sop05rTebRunError(str(exc)) from exc
    snippets = inputs.snippets
    counters: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    m5_candidate_counts: Counter[str] = Counter()
    m5_candidate_rejections: Counter[str] = Counter()
    accepted: list[Sop05rTebMotherCandidate] = []
    generation_context = _BaseStateGenerationContext(
        base_config=base_config,
        teb_config=teb_config,
        snippets=snippets,
        seed=request.seed,
        split=request.split,
        placement_selection_mode=request.placement_selection_mode,
    )
    excluded_event_ids, exclusion_evidence = _excluded_event_ids(
        request,
        config_digest=teb_config.digest,
    )
    indexed_state_pairs = tuple(
        enumerate(
            inputs.state_pairs,
            start=getattr(inputs, "base_state_start", request.base_state_start),
        )
    )
    processed_base_states = 0
    if request.accepted_quota is None or request.accepted_quota > 0:
        for base_result in _ordered_base_state_results(
            indexed_state_pairs,
            context=generation_context,
            workers=request.workers,
        ):
            if not isinstance(base_result, _BaseStateGenerationResult):
                raise Sop05rTebRunError("base-state worker returned an invalid result")
            processed_base_states += 1
            for template_result in base_result.templates:
                counters.update(template_result.counters)
                rejections.update(template_result.rejections)
                m5_candidate_counts.update(template_result.m5_candidate_counts)
                m5_candidate_rejections.update(
                    template_result.m5_candidate_rejections
                )
                if template_result.mother is not None:
                    event_id = template_result.mother.event.generated_event_id
                    if event_id in excluded_event_ids:
                        counters["m6_excluded_existing"] += 1
                    else:
                        accepted.append(template_result.mother)
                if (
                    request.accepted_quota is not None
                    and len(accepted) >= request.accepted_quota
                ):
                    break
            if progress_callback is not None:
                progress_callback(
                    _progress_snapshot(
                        processed_base_states=processed_base_states,
                        total_base_states=len(indexed_state_pairs),
                        accepted_count=len(accepted),
                        requested_count=request.accepted_quota,
                        started_at_s=started_at_s,
                        denominator_counts=counters,
                        rejection_counts=rejections,
                    )
                )
            if (
                request.accepted_quota is not None
                and len(accepted) >= request.accepted_quota
            ):
                break

    requested_count = (
        len(accepted)
        if request.accepted_quota is None
        else request.accepted_quota
    )
    source_evidence = {
        **inputs.source_evidence,
        **exclusion_evidence,
        "base_state_start": request.base_state_start,
        "accepted_selection": (
            "all_accepted_v1"
            if request.accepted_quota is None
            else "accepted_quota_v1"
        ),
        "placement_selection_mode": request.placement_selection_mode,
    }

    loaded = publish_sop05r_teb_run(
        tuple(
            accepted
            if request.accepted_quota is None
            else accepted[: request.accepted_quota]
        ),
        request.output_dir,
        base_config=base_config,
        requested_count=requested_count,
        config_digest=teb_config.digest,
        verification_action_digest=_sha256_file(
            request.verification_action_config_path
        ),
        source_evidence=source_evidence,
        denominator_counts=counters,
        rejection_counts=rejections,
        m5_candidate_counts=m5_candidate_counts,
        m5_candidate_rejection_counts=m5_candidate_rejections,
        resume_staging_root=request.resume_staging_root,
    )
    return Sop05rTebRunResult(
        output_dir=request.output_dir,
        run_state="complete" if loaded.complete else "quota_unmet",
        exit_code=0 if loaded.complete else 4,
        accepted_count=len(loaded.events),
        requested_count=requested_count,
        publication_semantic_digest=loaded.publication_semantic_digest,
        generation_summary=loaded.summary,
        complete=loaded.complete,
    )
