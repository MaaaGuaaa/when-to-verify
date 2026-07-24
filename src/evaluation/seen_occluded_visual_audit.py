"""Audited search and publication for real seen-then-occluded visuals."""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np

from src.contracts import BaseState, LocalTrajectory, OracleContext, build_grid_spec
from src.datasets.snippet_library import MotionSnippet
from src.generation.dynamic_object_transplant import footprint_from_spec
from src.generation.event_sampler import GeneratedEvent, generate_events
from src.generation.history_visibility import (
    SEEN_THEN_OCCLUDED,
    classify_history_visibility,
    normalize_history_visibility_policy,
)
from src.generation.occluder_sampler import (
    swept_footprint_intersects_occupancy,
    synchronized_sweeps_intersect,
)
from src.generation.paired_variants import (
    VARIANT_ORDER,
    PairGenerationError,
    PairedEventGroup,
    PairedVariantConfig,
    generate_paired_variants,
    load_paired_variant_config,
)
from src.generation.sop06_pipeline import (
    RenderedSop06Group,
    render_sop06_complete_audit_group,
)
from src.generation.sop05_run import (
    PairGenerationReport,
    PreparedSop05Run,
    Sop05RunRequest,
    _iter_pair_reports_bounded,
    _make_pair_process_pool,
    _prewarm_robot_sweep_cache,
    prepare_sop05_run,
)
from src.generation.structural_blindspot import has_continuous_emergence
from src.geometry import (
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint,
    trajectory_signed_clearances,
)
from src.utils.atomic_publish import atomic_rename_noreplace
from src.utils.seeding import derive_seed

from .seen_occluded_visuals import (
    VisualArtifactResult,
    VisualAuditBundle,
    VisualVariant,
    render_visual_artifacts,
)
from .seen_occluded_joint_search import (
    JointAuditSearchConfig,
    JointAuditSearchResult,
    load_joint_audit_search_config,
    search_joint_audit_group,
)


AUDIT_COLLECTION_VERSION = "seen_then_occluded_visual_audit_v1"
_JOINT_REPAIRABLE_COVERAGE = (
    True,
    True,
    False,
    True,
    True,
    True,
)
_JOINT_REPAIRABLE_SPATIAL_COVERAGE = (
    True,
    True,
    False,
    False,
    True,
    True,
)
_JOINT_REPAIRABLE_TEMPORAL_REASONS = frozenset(
    {"target_current_visible", "target_occluder_collision"}
)


class SeenOccludedAuditError(ValueError):
    """Raised when visual-audit inputs or evidence violate the contract."""


@dataclass(frozen=True)
class SeenOccludedAuditRequest:
    sop03_root: Path
    sop04_root: Path
    sop04_handoff_digest: str
    split: str
    base_config_path: Path
    generator_config_path: Path
    paired_config_path: Path
    output_dir: Path
    seed: int
    sample_count: int
    events_per_pair: int
    max_base_states: int
    trajectory_count: int
    max_pairs: int
    max_seen_mothers: int
    checksum_workers: int
    workers: int
    git_executable: Path
    joint_config_path: Path = Path(
        "configs/seen_occluded_joint_visual_audit.yaml"
    )


@dataclass(frozen=True)
class SearchAttempt:
    sequence_index: int
    schedule_rank: int
    state_id: str
    trajectory_id: str
    pair_seed: int
    event_id: str | None
    status: str
    reason: str
    history_regime: str | None
    coverage_mask: tuple[bool, ...] | None
    missing_variant_reasons: dict[str, str]
    joint_search_summary: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence_index": self.sequence_index,
            "schedule_rank": self.schedule_rank,
            "state_id": self.state_id,
            "trajectory_id": self.trajectory_id,
            "pair_seed": self.pair_seed,
            "event_id": self.event_id,
            "status": self.status,
            "reason": self.reason,
            "history_regime": self.history_regime,
            "coverage_mask": (
                None
                if self.coverage_mask is None
                else list(self.coverage_mask)
            ),
            "missing_variant_reasons": dict(
                sorted(self.missing_variant_reasons.items())
            ),
            "joint_search_summary": (
                None
                if self.joint_search_summary is None
                else dict(self.joint_search_summary)
            ),
        }


@dataclass(frozen=True)
class SearchEvaluation:
    attempt: SearchAttempt
    candidate: object | None
    seen_mother_increment: int


@dataclass(frozen=True)
class SeenOccludedSearchResult:
    attempts: tuple[SearchAttempt, ...]
    candidates: tuple[object, ...]
    selected_event_ids: tuple[str, ...]
    seen_mother_count: int
    complete: bool


@dataclass(frozen=True)
class SelectedAuditCandidate:
    schedule_rank: int
    pair_seed: int
    paired_seed: int
    event: GeneratedEvent
    group: PairedEventGroup
    rendered_group: RenderedSop06Group
    base_state: BaseState
    oracle_context: OracleContext
    trajectory: LocalTrajectory
    source_snippet: MotionSnippet
    joint_search_summary: dict[str, object] | None = None


@dataclass(frozen=True)
class CandidateEvidence:
    visual_bundle: VisualAuditBundle
    audit: dict[str, object]


@dataclass(frozen=True)
class AuditCollectionResult:
    status: str
    output_dir: Path
    manifest_sha256: str
    checksum_manifest_sha256: str
    exit_code: int


@dataclass(frozen=True)
class PreparedRealAudit:
    request: SeenOccludedAuditRequest
    sop05: PreparedSop05Run
    paired_config: PairedVariantConfig
    snippets_by_id: dict[str, MotionSnippet]
    context: dict[str, object]
    joint_config: JointAuditSearchConfig | None = None


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SeenOccludedAuditError(f"{name} must be a positive integer")
    return value


def validate_audit_request(
    request: SeenOccludedAuditRequest,
) -> SeenOccludedAuditRequest:
    """Validate immutable scalar and publication constraints."""

    if not isinstance(request, SeenOccludedAuditRequest):
        raise TypeError("request must be a SeenOccludedAuditRequest")
    if request.sample_count != 3:
        raise SeenOccludedAuditError("sample_count must equal 3")
    if request.output_dir.exists():
        raise SeenOccludedAuditError(
            f"refusing to overwrite existing output: {request.output_dir}"
        )
    if request.split != "train":
        raise SeenOccludedAuditError("visual audit split must equal train")
    if (
        not isinstance(request.seed, int)
        or isinstance(request.seed, bool)
        or request.seed < 0
    ):
        raise SeenOccludedAuditError("seed must be a non-negative integer")
    for name in (
        "events_per_pair",
        "max_base_states",
        "trajectory_count",
        "max_pairs",
        "max_seen_mothers",
        "checksum_workers",
        "workers",
    ):
        _positive_int(getattr(request, name), name=name)
    digest = request.sop04_handoff_digest
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise SeenOccludedAuditError(
            "sop04_handoff_digest must be 64 lowercase hexadecimal characters"
        )
    return request


def prepare_real_audit(request: SeenOccludedAuditRequest) -> PreparedRealAudit:
    """Authenticate all real inputs without generating or publishing samples."""

    validate_audit_request(request)
    sop05_request = Sop05RunRequest(
        sop03_root=request.sop03_root,
        sop04_root=request.sop04_root,
        sop04_external_handoff_digest_sha256=request.sop04_handoff_digest,
        split=request.split,
        base_config_path=request.base_config_path,
        generator_config_path=request.generator_config_path,
        output_dir=request.output_dir,
        seed=request.seed,
        accepted_quota=request.sample_count,
        events_per_pair=request.events_per_pair,
        max_base_states=request.max_base_states,
        trajectory_count=request.trajectory_count,
        max_pairs=request.max_pairs,
        checksum_workers=request.checksum_workers,
        workers=request.workers,
        git_executable=request.git_executable,
    )
    prepared_sop05 = prepare_sop05_run(sop05_request)
    paired_config = load_paired_variant_config(request.paired_config_path)
    joint_config = load_joint_audit_search_config(request.joint_config_path)
    snippets_by_id: dict[str, MotionSnippet] = {}
    for object_type in sorted(prepared_sop05.sop03.typed_libraries):
        library = prepared_sop05.sop03.typed_libraries[object_type]
        for snippet in library.snippets:
            if snippet.snippet_id in snippets_by_id:
                raise SeenOccludedAuditError("duplicate source snippet ID")
            snippets_by_id[snippet.snippet_id] = snippet
    context = {
        "collection_version": AUDIT_COLLECTION_VERSION,
        "sop05_run_id": prepared_sop05.run_id,
        "input_lock": _canonical_copy(prepared_sop05.input_lock),
        "producer_source_identity": _canonical_copy(
            prepared_sop05.producer_source_identity
        ),
        "base_config_sha256": prepared_sop05.base_config_sha256,
        "generator_config_sha256": prepared_sop05.generator_config_sha256,
        "generator_config_semantic_digest": (
            prepared_sop05.generator_config_semantic_digest
        ),
        "history_visibility_policy_digest": (
            prepared_sop05.target_history_visibility_policy_digest
        ),
        "paired_config": paired_config.as_dict(),
        "paired_config_digest": paired_config.digest,
        "joint_config": joint_config.as_dict(),
        "joint_config_digest": joint_config.digest,
    }
    return PreparedRealAudit(
        request=request,
        sop05=prepared_sop05,
        paired_config=paired_config,
        snippets_by_id=snippets_by_id,
        context=context,
        joint_config=joint_config,
    )


def real_audit_preflight_summary(
    prepared: PreparedRealAudit,
) -> dict[str, object]:
    """Return a finite read-only summary of authenticated real inputs."""

    if not isinstance(prepared, PreparedRealAudit):
        raise TypeError("prepared must be a PreparedRealAudit")
    return {
        "status": "preflight_ok",
        "collection_version": AUDIT_COLLECTION_VERSION,
        "sop05_run_id": prepared.sop05.run_id,
        "split": prepared.request.split,
        "schedule_count": len(prepared.sop05.schedule),
        "snippet_count": len(prepared.snippets_by_id),
        "trajectory_count": len(prepared.sop05.sop04.by_id),
        "history_steps": prepared.sop05.grid.history_steps,
        "future_steps": prepared.sop05.grid.future_steps,
        "paired_config_digest": prepared.paired_config.digest,
        "joint_config_digest": (
            None
            if prepared.joint_config is None
            else prepared.joint_config.digest
        ),
        "history_visibility_policy_digest": (
            prepared.sop05.target_history_visibility_policy_digest
        ),
        "output_dir": str(prepared.request.output_dir),
    }


def consume_search_evaluations(
    evaluations: Iterable[SearchEvaluation],
    *,
    sample_count: int,
    max_seen_mothers: int,
) -> SeenOccludedSearchResult:
    """Consume a stable lazy prefix until success or the seen-mother bound."""

    _positive_int(sample_count, name="sample_count")
    _positive_int(max_seen_mothers, name="max_seen_mothers")
    attempts: list[SearchAttempt] = []
    candidates: list[object] = []
    event_ids: list[str] = []
    seen_mother_count = 0
    for evaluation in evaluations:
        if not isinstance(evaluation, SearchEvaluation):
            raise TypeError("evaluations must yield SearchEvaluation values")
        increment = evaluation.seen_mother_increment
        if (
            isinstance(increment, bool)
            or not isinstance(increment, int)
            or increment not in {0, 1}
        ):
            raise SeenOccludedAuditError(
                "seen_mother_increment must equal zero or one"
            )
        attempts.append(evaluation.attempt)
        seen_mother_count += increment
        if evaluation.candidate is not None:
            event_id = evaluation.attempt.event_id
            if not event_id:
                raise SeenOccludedAuditError(
                    "accepted candidate requires a non-empty event_id"
                )
            if event_id in event_ids:
                raise SeenOccludedAuditError("duplicate selected event_id")
            candidates.append(evaluation.candidate)
            event_ids.append(event_id)
            if len(candidates) == sample_count:
                break
        if seen_mother_count >= max_seen_mothers:
            break
    return SeenOccludedSearchResult(
        attempts=tuple(attempts),
        candidates=tuple(candidates),
        selected_event_ids=tuple(event_ids),
        seen_mother_count=seen_mother_count,
        complete=len(candidates) == sample_count,
    )


def _joint_fallback_is_eligible(
    group: PairedEventGroup,
    *,
    joint_config: JointAuditSearchConfig | None,
) -> bool:
    if joint_config is None:
        return False
    reasons = group.missing_variant_reasons
    if (
        reasons.get("temporal_safe")
        not in _JOINT_REPAIRABLE_TEMPORAL_REASONS
    ):
        return False
    if group.coverage_mask == _JOINT_REPAIRABLE_COVERAGE:
        return set(reasons) == {"temporal_safe"}
    return (
        group.coverage_mask == _JOINT_REPAIRABLE_SPATIAL_COVERAGE
        and set(reasons) == {"temporal_safe", "spatial_safe"}
        and reasons.get("spatial_safe") == "target_occluder_collision"
    )


def iter_real_search_evaluations(
    prepared: PreparedRealAudit,
) -> Iterable[SearchEvaluation]:
    """Yield one deterministic attempt stream from authenticated real inputs."""

    if not isinstance(prepared, PreparedRealAudit):
        raise TypeError("prepared must be a PreparedRealAudit")
    sequence_index = 0
    accepted_event_ids: set[str] = set()
    accepted_base_state_ids: set[str] = set()
    schedule_by_rank = {
        pair.rank: pair for pair in prepared.sop05.schedule
    }
    for pair_report in iter_generated_pair_reports(prepared):
        try:
            pair = schedule_by_rank[pair_report.rank]
        except KeyError as exc:
            raise SeenOccludedAuditError(
                "pair report rank is absent from stable schedule"
            ) from exc
        base_state, oracle_context = prepared.sop05.sop03.load_pair(
            pair.state_id, prepared.sop05.grid
        )
        try:
            trajectory = prepared.sop05.sop04.by_id[pair.trajectory_id]
        except KeyError as exc:
            raise SeenOccludedAuditError(
                f"scheduled trajectory is missing: {pair.trajectory_id}"
            ) from exc
        report = pair_report.report
        yield SearchEvaluation(
            attempt=SearchAttempt(
                sequence_index=sequence_index,
                schedule_rank=pair.rank,
                state_id=pair.state_id,
                trajectory_id=pair.trajectory_id,
                pair_seed=pair.pair_seed,
                event_id=None,
                status="pair_generated",
                reason=(
                    "pair_generated"
                    if report.events
                    else "generator_deficit"
                ),
                history_regime=None,
                coverage_mask=None,
                missing_variant_reasons={},
            ),
            candidate=None,
            seen_mother_increment=0,
        )
        sequence_index += 1
        for event in report.events:
            raw_policy = event.world.metadata.get(
                "target_history_visibility_policy"
            )
            try:
                policy = normalize_history_visibility_policy(raw_policy)
                assessment = classify_history_visibility(
                    event.target_visibility_history, policy
                )
            except (TypeError, ValueError) as exc:
                raise SeenOccludedAuditError(
                    "generated event history visibility contract is invalid"
                ) from exc
            if assessment.regime != SEEN_THEN_OCCLUDED:
                yield SearchEvaluation(
                    attempt=SearchAttempt(
                        sequence_index=sequence_index,
                        schedule_rank=pair.rank,
                        state_id=pair.state_id,
                        trajectory_id=pair.trajectory_id,
                        pair_seed=pair.pair_seed,
                        event_id=event.generated_event_id,
                        status="history_regime_not_selected",
                        reason=assessment.regime,
                        history_regime=assessment.regime,
                        coverage_mask=None,
                        missing_variant_reasons={},
                    ),
                    candidate=None,
                    seen_mother_increment=0,
                )
                sequence_index += 1
                continue
            if event.generated_event_id in accepted_event_ids:
                yield SearchEvaluation(
                    attempt=SearchAttempt(
                        sequence_index=sequence_index,
                        schedule_rank=pair.rank,
                        state_id=pair.state_id,
                        trajectory_id=pair.trajectory_id,
                        pair_seed=pair.pair_seed,
                        event_id=event.generated_event_id,
                        status="duplicate_identity_rejected",
                        reason="duplicate_event_id",
                        history_regime=assessment.regime,
                        coverage_mask=None,
                        missing_variant_reasons={},
                    ),
                    candidate=None,
                    seen_mother_increment=1,
                )
                sequence_index += 1
                continue
            if base_state.state_id in accepted_base_state_ids:
                yield SearchEvaluation(
                    attempt=SearchAttempt(
                        sequence_index=sequence_index,
                        schedule_rank=pair.rank,
                        state_id=pair.state_id,
                        trajectory_id=pair.trajectory_id,
                        pair_seed=pair.pair_seed,
                        event_id=event.generated_event_id,
                        status="duplicate_identity_rejected",
                        reason="duplicate_base_state_id",
                        history_regime=assessment.regime,
                        coverage_mask=None,
                        missing_variant_reasons={},
                    ),
                    candidate=None,
                    seen_mother_increment=1,
                )
                sequence_index += 1
                continue
            snippet_id = event.target_motion_record.source_snippet_id
            source_snippet = prepared.snippets_by_id.get(snippet_id)
            if source_snippet is None:
                yield SearchEvaluation(
                    attempt=SearchAttempt(
                        sequence_index=sequence_index,
                        schedule_rank=pair.rank,
                        state_id=pair.state_id,
                        trajectory_id=pair.trajectory_id,
                        pair_seed=pair.pair_seed,
                        event_id=event.generated_event_id,
                        status="source_join_rejected",
                        reason="source_snippet_missing",
                        history_regime=assessment.regime,
                        coverage_mask=None,
                        missing_variant_reasons={},
                    ),
                    candidate=None,
                    seen_mother_increment=1,
                )
                sequence_index += 1
                continue
            paired_seed = derive_seed(
                prepared.request.seed,
                "seen-occluded-visual-pair",
                event.generated_event_id,
            )
            try:
                group = generate_paired_variants(
                    mother_event=event,
                    source_snippet=source_snippet,
                    base_state=base_state,
                    trajectory=trajectory,
                    oracle_context=oracle_context,
                    base_config=prepared.sop05.base_config,
                    paired_config=prepared.paired_config,
                    seed=paired_seed,
                )
            except PairGenerationError as exc:
                yield SearchEvaluation(
                    attempt=SearchAttempt(
                        sequence_index=sequence_index,
                        schedule_rank=pair.rank,
                        state_id=pair.state_id,
                        trajectory_id=pair.trajectory_id,
                        pair_seed=pair.pair_seed,
                        event_id=event.generated_event_id,
                        status="paired_generation_rejected",
                        reason=exc.reason,
                        history_regime=assessment.regime,
                        coverage_mask=None,
                        missing_variant_reasons={},
                    ),
                    candidate=None,
                    seen_mother_increment=1,
                )
                sequence_index += 1
                continue
            joint_search_summary = None
            if not group.is_complete or not group.eligible_for_strict_evaluation:
                if _joint_fallback_is_eligible(
                    group,
                    joint_config=prepared.joint_config,
                ):
                    joint_result = search_joint_audit_group(
                        mother_event=event,
                        source_snippet=source_snippet,
                        base_state=base_state,
                        trajectory=trajectory,
                        oracle_context=oracle_context,
                        base_config=prepared.sop05.base_config,
                        generator_config=prepared.sop05.generator_config,
                        paired_config=prepared.paired_config,
                        joint_config=prepared.joint_config,
                        pair_seed=pair.pair_seed,
                    )
                    if not isinstance(joint_result, JointAuditSearchResult):
                        raise SeenOccludedAuditError(
                            "joint search returned an invalid result"
                        )
                    joint_search_summary = dict(joint_result.summary)
                    if joint_result.mother_event is not None:
                        if (
                            joint_result.group is None
                            or joint_result.paired_seed is None
                            or not joint_result.group.is_complete
                            or not joint_result.group.eligible_for_strict_evaluation
                            or joint_result.mother_event.generated_event_id
                            != event.generated_event_id
                        ):
                            raise SeenOccludedAuditError(
                                "joint search success contract is invalid"
                            )
                        event = joint_result.mother_event
                        group = joint_result.group
                        paired_seed = joint_result.paired_seed
                        assessment = classify_history_visibility(
                            event.target_visibility_history, policy
                        )
                        if assessment.regime != SEEN_THEN_OCCLUDED:
                            raise SeenOccludedAuditError(
                                "joint search changed the selected history regime"
                            )
                    elif (
                        joint_result.group is not None
                        or joint_result.paired_seed is not None
                    ):
                        raise SeenOccludedAuditError(
                            "joint search failure contract is invalid"
                        )
                if (
                    not group.is_complete
                    or not group.eligible_for_strict_evaluation
                ):
                    yield SearchEvaluation(
                        attempt=SearchAttempt(
                            sequence_index=sequence_index,
                            schedule_rank=pair.rank,
                            state_id=pair.state_id,
                            trajectory_id=pair.trajectory_id,
                            pair_seed=pair.pair_seed,
                            event_id=event.generated_event_id,
                            status="partial_pair_group",
                            reason="partial_pair_group",
                            history_regime=assessment.regime,
                            coverage_mask=group.coverage_mask,
                            missing_variant_reasons=dict(
                                group.missing_variant_reasons
                            ),
                            joint_search_summary=joint_search_summary,
                        ),
                        candidate=None,
                        seen_mother_increment=1,
                    )
                    sequence_index += 1
                    continue
            try:
                rendered = render_sop06_complete_audit_group(
                    group=group,
                    mother_record=event.target_motion_record,
                    mother_world=event.world,
                    base_state=base_state,
                    oracle_context=oracle_context,
                    config=prepared.sop05.base_config,
                    expected_paired_config_digest=prepared.paired_config.digest,
                )
            except (TypeError, ValueError) as exc:
                yield SearchEvaluation(
                    attempt=SearchAttempt(
                        sequence_index=sequence_index,
                        schedule_rank=pair.rank,
                        state_id=pair.state_id,
                        trajectory_id=pair.trajectory_id,
                        pair_seed=pair.pair_seed,
                        event_id=event.generated_event_id,
                        status="formal_render_rejected",
                        reason=f"formal_render_rejected:{type(exc).__name__}",
                        history_regime=assessment.regime,
                        coverage_mask=group.coverage_mask,
                        missing_variant_reasons={},
                        joint_search_summary=joint_search_summary,
                    ),
                    candidate=None,
                    seen_mother_increment=1,
                )
                sequence_index += 1
                continue
            if event.generated_event_id in accepted_event_ids:
                status = "duplicate_identity_rejected"
                reason = "duplicate_event_id"
                evidence = None
            elif base_state.state_id in accepted_base_state_ids:
                status = "duplicate_identity_rejected"
                reason = "duplicate_base_state_id"
                evidence = None
            else:
                selected = SelectedAuditCandidate(
                    schedule_rank=pair.rank,
                    pair_seed=pair.pair_seed,
                    paired_seed=paired_seed,
                    event=event,
                    group=group,
                    rendered_group=rendered,
                    base_state=base_state,
                    oracle_context=oracle_context,
                    trajectory=trajectory,
                    source_snippet=source_snippet,
                    joint_search_summary=joint_search_summary,
                )
                try:
                    evidence = build_candidate_evidence(
                        selected,
                        base_config=prepared.sop05.base_config,
                        paired_config=prepared.paired_config,
                    )
                except SeenOccludedAuditError as exc:
                    status = "independent_audit_rejected"
                    reason = f"independent_audit_rejected:{exc}"
                    evidence = None
                else:
                    status = "accepted"
                    reason = "accepted"
                    accepted_event_ids.add(event.generated_event_id)
                    accepted_base_state_ids.add(base_state.state_id)
            yield SearchEvaluation(
                attempt=SearchAttempt(
                    sequence_index=sequence_index,
                    schedule_rank=pair.rank,
                    state_id=pair.state_id,
                    trajectory_id=pair.trajectory_id,
                    pair_seed=pair.pair_seed,
                    event_id=event.generated_event_id,
                    status=status,
                    reason=reason,
                    history_regime=assessment.regime,
                    coverage_mask=group.coverage_mask,
                    missing_variant_reasons={},
                    joint_search_summary=joint_search_summary,
                ),
                candidate=evidence,
                seen_mother_increment=1,
            )
            sequence_index += 1


def iter_generated_pair_reports(
    prepared: PreparedRealAudit,
) -> Iterable[PairGenerationReport]:
    """Generate pair reports serially or with SOP05's rank-ordered process pool."""

    if not isinstance(prepared, PreparedRealAudit):
        raise TypeError("prepared must be a PreparedRealAudit")
    workers = prepared.request.workers
    if prepared.sop05.request.workers != workers:
        raise SeenOccludedAuditError("prepared SOP05 worker count drift")
    if workers == 1:
        for pair in prepared.sop05.schedule:
            base_state, oracle_context = prepared.sop05.sop03.load_pair(
                pair.state_id, prepared.sop05.grid
            )
            try:
                trajectory = prepared.sop05.sop04.by_id[pair.trajectory_id]
            except KeyError as exc:
                raise SeenOccludedAuditError(
                    f"scheduled trajectory is missing: {pair.trajectory_id}"
                ) from exc
            report = generate_events(
                base_state=base_state,
                oracle_context=oracle_context,
                trajectory=trajectory,
                snippet_libraries=prepared.sop05.sop03.typed_libraries,
                base_config=prepared.sop05.base_config,
                generator_config=prepared.sop05.generator_config,
                seed=pair.pair_seed,
                event_count=prepared.request.events_per_pair,
            )
            yield PairGenerationReport(
                rank=pair.rank,
                state_id=pair.state_id,
                trajectory_id=pair.trajectory_id,
                pair_seed=pair.pair_seed,
                report=report,
            )
        return
    cache = _prewarm_robot_sweep_cache(prepared.sop05)
    with _make_pair_process_pool(prepared.sop05, cache) as executor:
        with closing(
            _iter_pair_reports_bounded(
                executor,
                prepared.sop05.schedule,
                max_in_flight=workers,
            )
        ) as reports:
            yield from reports


def search_real_audit(prepared: PreparedRealAudit) -> SeenOccludedSearchResult:
    """Search the authenticated stable schedule for the exact audit prefix."""

    return consume_search_evaluations(
        iter_real_search_evaluations(prepared),
        sample_count=prepared.request.sample_count,
        max_seen_mothers=prepared.request.max_seen_mothers,
    )


def run_real_audit(request: SeenOccludedAuditRequest) -> AuditCollectionResult:
    """Authenticate, search, render, and publish one visual audit collection."""

    prepared = prepare_real_audit(request)
    search_result = search_real_audit(prepared)
    return publish_audit_collection(
        request,
        search_result,
        context=prepared.context,
    )


def _canonical_copy(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _canonical_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_copy(item) for item in value]
    return value


def _canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        payload = json.dumps(
            _canonical_copy(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SeenOccludedAuditError("audit evidence is not finite JSON") from exc
    return payload + (b"\n" if newline else b"")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shared_skeleton(group: PairedEventGroup) -> bool:
    reference = group.variants[0].world
    for variant in group.variants[1:]:
        world = variant.world
        if not np.array_equal(world.static_occupancy, reference.static_occupancy):
            return False
        if world.occluders != reference.occluders:
            return False
        if world.blind_spot_config != reference.blind_spot_config:
            return False
    return True


def _robot_footprint(base_config: Mapping[str, Any]):
    robot = base_config["robot"]
    return inflate_footprint(
        RectangleFootprint(
            length_m=float(robot["length_m"]),
            width_m=float(robot["width_m"]),
        ),
        float(robot["inflation_m"]),
    )


def _label_predicate(
    kind: str,
    minimum: float,
    paired_config: PairedVariantConfig,
) -> bool:
    if kind == "collision":
        return minimum <= 0.0
    if kind == "near_miss":
        lower, upper = paired_config.near_miss_clearance_range_m
        return lower <= minimum <= upper
    if kind == "temporal_safe":
        return minimum > 0.0
    if kind == "spatial_safe":
        lower, upper = paired_config.spatial_safe_clearance_range_m
        return lower <= minimum <= upper
    if kind == "irrelevant_hidden":
        return minimum >= paired_config.irrelevant_min_clearance_m
    raise SeenOccludedAuditError(f"unsupported variant kind: {kind}")


def _represented_geometry_checks(
    *,
    candidate: SelectedAuditCandidate,
    variant,
    target_footprint,
    target_poses: np.ndarray,
    grid,
) -> dict[str, bool]:
    if candidate.base_state.static_map_local is None:
        base_static = np.zeros((grid.height, grid.width), dtype=np.bool_)
    else:
        base_static = np.asarray(
            candidate.base_state.static_map_local != 0, dtype=np.bool_
        )
    occluder_mask = np.zeros_like(base_static)
    exact_occluder_collision = False
    for row in variant.world.occluders:
        footprint = RectangleFootprint(
            length_m=float(row["length_m"]),
            width_m=float(row["width_m"]),
        )
        pose = np.asarray(row["pose"], dtype=np.float32)
        occluder_mask |= rasterize_footprint(footprint, pose, grid)
        exact_occluder_collision = exact_occluder_collision or (
            synchronized_sweeps_intersect(
                target_footprint,
                target_poses,
                footprint,
                np.tile(pose, (target_poses.shape[0], 1)),
                grid=grid,
            )
        )
    expected_world_static = base_static | occluder_mask
    static_layout_matches = np.array_equal(
        np.asarray(variant.world.static_occupancy != 0, dtype=np.bool_),
        expected_world_static,
    )
    base_static_collision = bool(base_static.any()) and (
        swept_footprint_intersects_occupancy(
            target_footprint,
            target_poses,
            base_static,
            grid=grid,
        )
    )
    return {
        "static_layout_matches": static_layout_matches,
        "base_static_collision": base_static_collision,
        "exact_occluder_collision": exact_occluder_collision,
    }


def _variant_evidence(
    *,
    kind: str,
    variant,
    candidate: SelectedAuditCandidate,
    paired_config: PairedVariantConfig,
    history_policy,
    robot_footprint,
    grid,
) -> tuple[VisualVariant, dict[str, object]]:
    if kind == "empty_blind_spot":
        if any(
            value is not None
            for value in (
                variant.target,
                variant.target_visibility_history,
                variant.visibility_sequence,
                variant.clearance_sequence_m,
                variant.min_clearance_m,
                variant.time_to_min_clearance_s,
            )
        ):
            raise SeenOccludedAuditError("empty blind spot did not remove target")
        return (
            VisualVariant(
                kind=kind,
                target_history=None,
                target_future=None,
                visibility_history=None,
                min_clearance_m=None,
                time_to_min_clearance_s=None,
                temporal_offset_s=None,
            ),
            {
                "target_present": False,
                "target_removed": True,
                "label_predicate_passed": True,
            },
        )
    if variant.target is None:
        raise SeenOccludedAuditError(f"{kind} target is missing")
    assessment = classify_history_visibility(
        variant.target_visibility_history, history_policy
    )
    if assessment.regime != SEEN_THEN_OCCLUDED:
        raise SeenOccludedAuditError(f"{kind} history regime changed")
    visibility = np.asarray(variant.visibility_sequence)
    if visibility.shape != (16,) or visibility.dtype != np.bool_:
        raise SeenOccludedAuditError(f"{kind} visibility sequence is invalid")
    current_hidden = not bool(visibility[0])
    emerges = has_continuous_emergence(visibility, min_visible_frames=2)
    target_footprint = footprint_from_spec(variant.target.footprint_spec)
    recomputed = trajectory_signed_clearances(
        robot_footprint,
        candidate.trajectory.poses,
        target_footprint,
        variant.target.future_poses,
    )
    stored = np.asarray(variant.clearance_sequence_m, dtype=np.float64)
    if stored.shape != (15,) or not np.isfinite(stored).all():
        raise SeenOccludedAuditError(f"{kind} clearance sequence is invalid")
    if not np.allclose(recomputed, stored, rtol=0.0, atol=1e-6):
        raise SeenOccludedAuditError(f"{kind} clearance sequence drift")
    minimum_index = int(np.argmin(recomputed))
    minimum = float(recomputed[minimum_index])
    time_to_minimum = float((minimum_index + 1) * 0.2)
    if not np.isclose(minimum, variant.min_clearance_m, rtol=0.0, atol=1e-6):
        raise SeenOccludedAuditError(f"{kind} minimum clearance drift")
    if not np.isclose(
        time_to_minimum,
        variant.time_to_min_clearance_s,
        rtol=0.0,
        atol=1e-9,
    ):
        raise SeenOccludedAuditError(f"{kind} time-to-clearance drift")
    label_passed = _label_predicate(kind, minimum, paired_config)
    if not current_hidden or not emerges or not label_passed:
        raise SeenOccludedAuditError(f"{kind} scientific predicate failed")
    all_target_poses = np.vstack(
        (variant.target.history_poses, variant.target.future_poses)
    )
    geometry = _represented_geometry_checks(
        candidate=candidate,
        variant=variant,
        target_footprint=target_footprint,
        target_poses=all_target_poses,
        grid=grid,
    )
    if not geometry["static_layout_matches"]:
        raise SeenOccludedAuditError(f"{kind} represented static layout drift")
    if geometry["base_static_collision"] or geometry["exact_occluder_collision"]:
        raise SeenOccludedAuditError(f"{kind} intersects represented static geometry")
    visual = VisualVariant(
        kind=kind,
        target_history=variant.target.history_poses.copy(),
        target_future=variant.target.future_poses.copy(),
        visibility_history=variant.target_visibility_history.copy(),
        min_clearance_m=minimum,
        time_to_min_clearance_s=time_to_minimum,
        temporal_offset_s=variant.temporal_offset_s,
    )
    return (
        visual,
        {
            "target_present": True,
            "history_regime": assessment.regime,
            "history_visibility": variant.target_visibility_history.tolist(),
            "visibility_sequence": visibility.tolist(),
            "current_hidden": current_hidden,
            "emerges_for_two_frames": emerges,
            "visible_at_final": bool(visibility[-1]),
            "clearance_sequence_m": recomputed.tolist(),
            "min_clearance_m": minimum,
            "time_to_min_clearance_s": time_to_minimum,
            "temporal_offset_s": variant.temporal_offset_s,
            "lateral_offset_m": variant.lateral_offset_m,
            "radial_shift_m": variant.radial_shift_m,
            "rotation_rad": variant.rotation_rad,
            "label_predicate_passed": label_passed,
            **geometry,
            "represented_static_collision": False,
        },
    )


def build_candidate_evidence(
    candidate: SelectedAuditCandidate,
    *,
    base_config: Mapping[str, Any],
    paired_config: PairedVariantConfig,
) -> CandidateEvidence:
    """Independently recompute one selected mother's visual audit evidence."""

    if not isinstance(candidate, SelectedAuditCandidate):
        raise TypeError("candidate must be a SelectedAuditCandidate")
    if not isinstance(paired_config, PairedVariantConfig):
        raise TypeError("paired_config must be a PairedVariantConfig")
    group = candidate.group
    if (
        not group.is_complete
        or not group.eligible_for_strict_evaluation
        or group.coverage_mask != (True,) * len(VARIANT_ORDER)
        or tuple(variant.variant_kind for variant in group.variants)
        != VARIANT_ORDER
        or group.missing_variant_reasons
    ):
        raise SeenOccludedAuditError("candidate is not a complete sixpack")
    if group.paired_config_digest != paired_config.digest:
        raise SeenOccludedAuditError("paired config digest mismatch")
    rendered = candidate.rendered_group
    if (
        not rendered.audit_certified
        or not rendered.is_complete
        or rendered.coverage_mask != (True,) * len(VARIANT_ORDER)
        or rendered.variant_kinds != VARIANT_ORDER
        or rendered.pair_group_id != group.pair_group_id
    ):
        raise SeenOccludedAuditError("formal rendered group is not certified")
    if not _shared_skeleton(group):
        raise SeenOccludedAuditError("paired static skeleton changed")
    event = candidate.event
    if event.generated_event_id != event.target_motion_record.generated_event_id:
        raise SeenOccludedAuditError("event target-motion identity mismatch")
    if event.target.snippet_id != candidate.source_snippet.snippet_id:
        raise SeenOccludedAuditError("source snippet join mismatch")
    if event.world.base_state_id != candidate.base_state.state_id or (
        candidate.oracle_context.base_state_id != candidate.base_state.state_id
    ):
        raise SeenOccludedAuditError("base-state join mismatch")
    if event.target_motion_record.trajectory_id != candidate.trajectory.trajectory_id:
        raise SeenOccludedAuditError("trajectory join mismatch")
    raw_policy = event.world.metadata.get("target_history_visibility_policy")
    history_policy = normalize_history_visibility_policy(raw_policy)
    if event.world.metadata.get("target_history_visibility_policy_digest") != (
        history_policy.digest
    ):
        raise SeenOccludedAuditError("history visibility policy digest mismatch")
    assessment = classify_history_visibility(
        event.target_visibility_history, history_policy
    )
    if assessment.regime != SEEN_THEN_OCCLUDED:
        raise SeenOccludedAuditError("collision mother is not seen_then_occluded")
    if not np.array_equal(
        event.target_visibility_history,
        group.by_kind["collision"].target_visibility_history,
    ):
        raise SeenOccludedAuditError("collision history visibility drift")
    grid = build_grid_spec(dict(base_config))
    robot_footprint = _robot_footprint(base_config)
    visual_variants = []
    variant_rows: dict[str, object] = {}
    for kind in VARIANT_ORDER:
        visual, row = _variant_evidence(
            kind=kind,
            variant=group.by_kind[kind],
            candidate=candidate,
            paired_config=paired_config,
            history_policy=history_policy,
            robot_footprint=robot_footprint,
            grid=grid,
        )
        visual_variants.append(visual)
        variant_rows[kind] = row
    robot = base_config["robot"]
    bundle = VisualAuditBundle(
        event_id=event.generated_event_id,
        base_state=candidate.base_state,
        oracle_context=candidate.oracle_context,
        trajectory=candidate.trajectory,
        static_occupancy=group.variants[0].world.static_occupancy.copy(),
        occluders=tuple(dict(item) for item in group.variants[0].world.occluders),
        variants=tuple(visual_variants),
        grid=grid,
        robot_length_m=float(robot["length_m"]),
        robot_width_m=float(robot["width_m"]),
    )
    audit = {
        "audit_version": AUDIT_COLLECTION_VERSION,
        "generated_event_id": event.generated_event_id,
        "pair_group_id": group.pair_group_id,
        "base_state_id": candidate.base_state.state_id,
        "trajectory_id": candidate.trajectory.trajectory_id,
        "schedule_rank": candidate.schedule_rank,
        "pair_seed": candidate.pair_seed,
        "paired_seed": candidate.paired_seed,
        "joint_search_summary": (
            None
            if candidate.joint_search_summary is None
            else _canonical_copy(candidate.joint_search_summary)
        ),
        "source": {
            "snippet_id": candidate.source_snippet.snippet_id,
            "object_id": candidate.source_snippet.source_object_id,
            "recording_id": candidate.source_snippet.source_recording_id,
            "session_id": candidate.source_snippet.source_session_id,
            "object_type": candidate.source_snippet.object_type,
        },
        "history_visibility": {
            "vector": event.target_visibility_history.tolist(),
            "regime": assessment.regime,
            "last_visible_index": assessment.last_visible_index,
            "trailing_hidden_frames": assessment.trailing_hidden_frames,
            "policy": history_policy.as_dict(),
            "policy_digest": history_policy.digest,
        },
        "scientific_checks": {
            "complete_sixpack": True,
            "formal_render_certified": True,
            "shared_skeleton": True,
            "current_hidden_all_nonempty": True,
            "future_emergence_all_nonempty": True,
            "represented_geometry_passed": True,
            "unmodeled_floorplan": "unknown",
        },
        "variants": variant_rows,
        "occluders": _canonical_copy(group.variants[0].world.occluders),
        "blind_spot_config": _canonical_copy(
            group.variants[0].world.blind_spot_config
        ),
    }
    return CandidateEvidence(visual_bundle=bundle, audit=audit)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _checksum_payload(root: Path) -> tuple[bytes, dict[str, str]]:
    excluded = {"artifact_checksums.sha256", ".audit-complete"}
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    )
    entries = {
        path.relative_to(root).as_posix(): _sha256_file(path) for path in files
    }
    lines = sorted(f"{digest}  {relative}\n" for relative, digest in entries.items())
    return "".join(lines).encode("utf-8"), entries


def _sample_directory_name(index: int, event_id: str) -> str:
    safe_prefix = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in event_id[:8]
    )
    if not safe_prefix:
        raise SeenOccludedAuditError("event ID has no safe directory prefix")
    return f"sample_{index:02d}_{safe_prefix}"


def _validate_render_result(
    result: VisualArtifactResult,
    *,
    sample_dir: Path,
) -> None:
    expected = {
        "event_replay.gif": result.event_replay_path,
        "paired_events.png": result.paired_events_path,
    }
    for name, path in expected.items():
        if path != sample_dir / name or not path.is_file() or path.is_symlink():
            raise SeenOccludedAuditError(f"renderer returned invalid {name} path")
    for metadata, path in (
        (result.event_replay_metadata, result.event_replay_path),
        (result.paired_events_metadata, result.paired_events_path),
    ):
        if metadata.get("bytes") != path.stat().st_size:
            raise SeenOccludedAuditError("rendered artifact byte count mismatch")
        if metadata.get("sha256") != _sha256_file(path):
            raise SeenOccludedAuditError("rendered artifact checksum mismatch")


def publish_audit_collection(
    request: SeenOccludedAuditRequest,
    search_result: SeenOccludedSearchResult,
    *,
    context: Mapping[str, object],
    renderer=render_visual_artifacts,
) -> AuditCollectionResult:
    """Render, verify, and atomically publish one immutable audit collection."""

    validate_audit_request(request)
    if not isinstance(search_result, SeenOccludedSearchResult):
        raise TypeError("search_result must be a SeenOccludedSearchResult")
    if not isinstance(context, Mapping):
        raise TypeError("context must be a mapping")
    if len(search_result.candidates) > request.sample_count:
        raise SeenOccludedAuditError("search result exceeds requested sample count")
    status = (
        "complete"
        if search_result.complete
        and len(search_result.candidates) == request.sample_count
        else "insufficient_complete_samples"
    )
    request.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{request.output_dir.name}.staging-",
            dir=request.output_dir.parent,
        )
    )
    try:
        sample_rows = []
        for index, raw_evidence in enumerate(search_result.candidates, start=1):
            if not isinstance(raw_evidence, CandidateEvidence):
                raise SeenOccludedAuditError(
                    "search candidates must be CandidateEvidence values"
                )
            event_id = raw_evidence.visual_bundle.event_id
            if raw_evidence.audit.get("generated_event_id") != event_id:
                raise SeenOccludedAuditError("candidate evidence event ID mismatch")
            directory_name = _sample_directory_name(index, event_id)
            sample_dir = staging / directory_name
            render_result = renderer(raw_evidence.visual_bundle, sample_dir)
            if not isinstance(render_result, VisualArtifactResult):
                raise SeenOccludedAuditError(
                    "renderer must return VisualArtifactResult"
                )
            _validate_render_result(render_result, sample_dir=sample_dir)
            audit = deepcopy_json = json.loads(
                _canonical_json_bytes(raw_evidence.audit, newline=False).decode(
                    "utf-8"
                )
            )
            audit["collection_context"] = _canonical_copy(context)
            audit["files"] = {
                "event_replay.gif": _canonical_copy(
                    render_result.event_replay_metadata
                ),
                "paired_events.png": _canonical_copy(
                    render_result.paired_events_metadata
                ),
            }
            audit_path = sample_dir / "audit.json"
            _write_bytes(audit_path, _canonical_json_bytes(audit))
            sample_rows.append(
                {
                    "sample_index": index,
                    "generated_event_id": event_id,
                    "directory": directory_name,
                    "audit_sha256": _sha256_file(audit_path),
                    "event_replay_sha256": render_result.event_replay_metadata[
                        "sha256"
                    ],
                    "paired_events_sha256": render_result.paired_events_metadata[
                        "sha256"
                    ],
                }
            )
        attempt_bytes = b"".join(
            _canonical_json_bytes(attempt.as_dict())
            for attempt in search_result.attempts
        )
        _write_bytes(staging / "search_attempts.jsonl", attempt_bytes)
        reason_counts = Counter(attempt.reason for attempt in search_result.attempts)
        manifest = {
            "manifest_version": AUDIT_COLLECTION_VERSION,
            "status": status,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sample_count": len(sample_rows),
            "required_sample_count": request.sample_count,
            "selected_event_ids": list(search_result.selected_event_ids),
            "seen_mother_count": search_result.seen_mother_count,
            "search_attempt_count": len(search_result.attempts),
            "search_reason_counts": dict(sorted(reason_counts.items())),
            "search_attempts_sha256": _sha256_bytes(attempt_bytes),
            "request": {
                "split": request.split,
                "seed": request.seed,
                "events_per_pair": request.events_per_pair,
                "max_base_states": request.max_base_states,
                "trajectory_count": request.trajectory_count,
                "max_pairs": request.max_pairs,
                "max_seen_mothers": request.max_seen_mothers,
            },
            "context": _canonical_copy(context),
            "samples": sample_rows,
        }
        manifest_path = staging / "visual_audit_manifest.json"
        _write_bytes(manifest_path, _canonical_json_bytes(manifest))
        checksum_payload, _ = _checksum_payload(staging)
        checksum_path = staging / "artifact_checksums.sha256"
        _write_bytes(checksum_path, checksum_payload)
        manifest_sha256 = _sha256_file(manifest_path)
        checksum_sha256 = _sha256_file(checksum_path)
        if status == "complete":
            marker = {
                "marker_version": AUDIT_COLLECTION_VERSION,
                "status": "complete",
                "manifest_sha256": manifest_sha256,
                "checksum_manifest_sha256": checksum_sha256,
            }
            _write_bytes(staging / ".audit-complete", _canonical_json_bytes(marker))
        load_audit_collection(
            staging,
            require_complete=status == "complete",
        )
        atomic_rename_noreplace(staging, request.output_dir)
        staging = None
    except Exception:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        raise
    return AuditCollectionResult(
        status=status,
        output_dir=request.output_dir,
        manifest_sha256=manifest_sha256,
        checksum_manifest_sha256=checksum_sha256,
        exit_code=0 if status == "complete" else 3,
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SeenOccludedAuditError(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise SeenOccludedAuditError(f"{label} must contain a JSON object")
    return value


def _parse_checksums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SeenOccludedAuditError("invalid checksum manifest") from exc
    if lines != sorted(lines):
        raise SeenOccludedAuditError("checksum manifest order mismatch")
    for line in lines:
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise SeenOccludedAuditError("invalid checksum manifest line")
        digest, relative = parts
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in entries
        ):
            raise SeenOccludedAuditError("invalid checksum manifest entry")
        entries[relative] = digest
    return entries


def load_audit_collection(
    root: str | Path,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    """Strictly reload and authenticate a visual-audit collection."""

    from PIL import Image

    root_path = Path(root)
    if not root_path.is_dir() or root_path.is_symlink():
        raise SeenOccludedAuditError("audit collection root is invalid")
    manifest_path = root_path / "visual_audit_manifest.json"
    checksum_path = root_path / "artifact_checksums.sha256"
    manifest = _load_json(manifest_path, label="visual audit manifest")
    if manifest.get("manifest_version") != AUDIT_COLLECTION_VERSION:
        raise SeenOccludedAuditError("visual audit manifest version mismatch")
    status = manifest.get("status")
    if status not in {"complete", "insufficient_complete_samples"}:
        raise SeenOccludedAuditError("visual audit status is invalid")
    if require_complete and status != "complete":
        raise SeenOccludedAuditError("audit collection is not complete")
    entries = _parse_checksums(checksum_path)
    actual_payloads = {
        path.relative_to(root_path).as_posix()
        for path in root_path.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_checksums.sha256", ".audit-complete"}
    }
    if set(entries) != actual_payloads:
        raise SeenOccludedAuditError("checksum payload file set mismatch")
    for relative, expected in entries.items():
        if _sha256_file(root_path / relative) != expected:
            raise SeenOccludedAuditError(f"checksum mismatch: {relative}")
    marker_path = root_path / ".audit-complete"
    if status == "complete":
        if not marker_path.is_file() or marker_path.is_symlink():
            raise SeenOccludedAuditError("complete audit marker is missing")
        marker = _load_json(marker_path, label="audit completion marker")
        expected_marker = {
            "marker_version": AUDIT_COLLECTION_VERSION,
            "status": "complete",
            "manifest_sha256": _sha256_file(manifest_path),
            "checksum_manifest_sha256": _sha256_file(checksum_path),
        }
        if marker != expected_marker:
            raise SeenOccludedAuditError("audit completion marker digest mismatch")
    elif marker_path.exists():
        raise SeenOccludedAuditError("insufficient audit must not have a marker")
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise SeenOccludedAuditError("manifest samples must be a list")
    selected = manifest.get("selected_event_ids")
    if not isinstance(selected, list) or selected != [
        row.get("generated_event_id") for row in samples if isinstance(row, dict)
    ]:
        raise SeenOccludedAuditError("manifest selected event IDs mismatch")
    if status == "complete" and len(samples) != 3:
        raise SeenOccludedAuditError("complete audit must contain three samples")
    for row in samples:
        if not isinstance(row, dict):
            raise SeenOccludedAuditError("manifest sample row is invalid")
        directory = row.get("directory")
        if not isinstance(directory, str) or Path(directory).name != directory:
            raise SeenOccludedAuditError("manifest sample directory is invalid")
        sample_dir = root_path / directory
        audit_path = sample_dir / "audit.json"
        audit = _load_json(audit_path, label="sample audit")
        if audit.get("generated_event_id") != row.get("generated_event_id"):
            raise SeenOccludedAuditError("sample audit event ID mismatch")
        if _sha256_file(audit_path) != row.get("audit_sha256"):
            raise SeenOccludedAuditError("sample audit checksum binding mismatch")
        gif_path = sample_dir / "event_replay.gif"
        png_path = sample_dir / "paired_events.png"
        files = audit.get("files")
        if not isinstance(files, dict):
            raise SeenOccludedAuditError("sample audit files mapping is invalid")
        if files.get("event_replay.gif", {}).get("sha256") != _sha256_file(gif_path):
            raise SeenOccludedAuditError("GIF audit checksum binding mismatch")
        if files.get("paired_events.png", {}).get("sha256") != _sha256_file(png_path):
            raise SeenOccludedAuditError("PNG audit checksum binding mismatch")
        with Image.open(gif_path) as gif:
            if gif.format != "GIF" or gif.size != (1200, 900) or gif.n_frames != 23:
                raise SeenOccludedAuditError("GIF image contract mismatch")
        with Image.open(png_path) as png:
            if png.format != "PNG" or png.size != (2100, 1200):
                raise SeenOccludedAuditError("PNG image contract mismatch")
    attempt_path = root_path / "search_attempts.jsonl"
    if _sha256_file(attempt_path) != manifest.get("search_attempts_sha256"):
        raise SeenOccludedAuditError("search attempt digest mismatch")
    return manifest
