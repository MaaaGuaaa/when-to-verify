"""Scientific and publication contracts for seen-occluded visual audits."""

from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.contracts import build_grid_spec
from src.evaluation.seen_occluded_visual_audit import (
    CandidateEvidence,
    PreparedRealAudit,
    SearchAttempt,
    SearchEvaluation,
    SelectedAuditCandidate,
    SeenOccludedSearchResult,
    SeenOccludedAuditRequest,
    SeenOccludedAuditError,
    build_candidate_evidence,
    consume_search_evaluations,
    iter_real_search_evaluations,
    iter_generated_pair_reports,
    load_audit_collection,
    publish_audit_collection,
    prepare_real_audit,
    validate_audit_request,
)
from src.evaluation.seen_occluded_visuals import VisualArtifactResult
from src.evaluation.seen_occluded_joint_search import (
    JointAuditSearchResult,
    load_joint_audit_search_config,
)
from src.generation.paired_variants import (
    VARIANT_ORDER,
    assemble_paired_event_group,
    generate_paired_variants,
    load_paired_variant_config,
)
from src.generation.event_sampler import load_generator_config
from src.generation.sop06_pipeline import RenderedSop06Group
from tests.test_pair_variants import _mother_inputs, _paired_source_inputs


ROOT = Path(__file__).resolve().parents[1]


def test_visual_audit_generator_config_focuses_only_seen_then_occluded() -> None:
    audit_config = load_generator_config(
        ROOT / "configs/generator_seen_occluded_visual_audit.yaml"
    )
    formal_config = load_generator_config(ROOT / "configs/generator_test.yaml")

    assert audit_config["target_history_visibility"].weights == {
        "seen_then_occluded": 1.0,
        "unseen_in_history_window": 0.0,
    }
    assert audit_config["blind_reachability"][
        "obstacle_proposals_per_trajectory"
    ] == 8
    assert audit_config["blind_reachability"][
        "snippet_candidates_per_proposal"
    ] == 8
    comparable_audit = {
        key: value
        for key, value in audit_config.items()
        if key != "target_history_visibility"
    }
    comparable_audit["blind_reachability"] = {
        **comparable_audit["blind_reachability"],
        "obstacle_proposals_per_trajectory": formal_config[
            "blind_reachability"
        ]["obstacle_proposals_per_trajectory"],
        "snippet_candidates_per_proposal": formal_config[
            "blind_reachability"
        ]["snippet_candidates_per_proposal"],
    }
    assert comparable_audit == {
        key: value
        for key, value in formal_config.items()
        if key != "target_history_visibility"
    }


def test_visual_audit_paired_config_changes_only_spatial_search_grid() -> None:
    audit_config = load_paired_variant_config(
        ROOT / "configs/paired_variants_visual_audit.yaml"
    )
    formal_config = load_paired_variant_config(
        ROOT / "configs/paired_variants.yaml"
    )

    assert audit_config.lateral_offset_step_m == 0.1
    assert audit_config.lateral_offset_max_m == 2.0
    assert {
        key: value
        for key, value in audit_config.as_dict().items()
        if key not in {"lateral_offset_step_m", "lateral_offset_max_m"}
    } == {
        key: value
        for key, value in formal_config.as_dict().items()
        if key not in {"lateral_offset_step_m", "lateral_offset_max_m"}
    }


def _request(tmp_path: Path, **changes: object) -> SeenOccludedAuditRequest:
    values = {
        "sop03_root": tmp_path / "sop03-schema3",
        "sop04_root": tmp_path / "sop04-schema3",
        "sop04_handoff_digest": "a" * 64,
        "split": "train",
        "base_config_path": tmp_path / "base.yaml",
        "generator_config_path": tmp_path / "generator.yaml",
        "paired_config_path": tmp_path / "paired.yaml",
        "joint_config_path": tmp_path / "joint.yaml",
        "output_dir": tmp_path / "visual-audit",
        "seed": 42,
        "sample_count": 3,
        "events_per_pair": 10,
        "max_base_states": 512,
        "trajectory_count": 21,
        "max_pairs": 512,
        "max_seen_mothers": 4096,
        "checksum_workers": 8,
        "workers": 8,
        "git_executable": Path("/usr/bin/git"),
    }
    values.update(changes)
    return SeenOccludedAuditRequest(**values)


def _attempt(index: int, status: str, event_id: str | None) -> SearchAttempt:
    return SearchAttempt(
        sequence_index=index,
        schedule_rank=index,
        state_id=f"state-{index}",
        trajectory_id=f"trajectory-{index}",
        pair_seed=100 + index,
        event_id=event_id,
        status=status,
        reason=status,
        history_regime=(
            "seen_then_occluded" if event_id is not None else None
        ),
        coverage_mask=(True,) * 6 if status == "accepted" else None,
        missing_variant_reasons={},
    )


def test_validate_audit_request_requires_three_samples_and_new_output(
    tmp_path: Path,
) -> None:
    with pytest.raises(SeenOccludedAuditError, match="sample_count"):
        validate_audit_request(_request(tmp_path, sample_count=2))

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(SeenOccludedAuditError, match="refusing to overwrite"):
        validate_audit_request(_request(tmp_path, output_dir=existing))


def test_consume_search_evaluations_stops_at_third_complete_candidate() -> None:
    visited: list[str] = []

    def evaluations():
        rows = (
            SearchEvaluation(
                attempt=_attempt(0, "generator_deficit", None),
                candidate=None,
                seen_mother_increment=0,
            ),
            SearchEvaluation(
                attempt=_attempt(1, "partial_pair_group", "event-partial"),
                candidate=None,
                seen_mother_increment=1,
            ),
            SearchEvaluation(
                attempt=_attempt(2, "accepted", "event-a"),
                candidate={"event_id": "event-a"},
                seen_mother_increment=1,
            ),
            SearchEvaluation(
                attempt=_attempt(3, "accepted", "event-b"),
                candidate={"event_id": "event-b"},
                seen_mother_increment=1,
            ),
            SearchEvaluation(
                attempt=_attempt(4, "accepted", "event-c"),
                candidate={"event_id": "event-c"},
                seen_mother_increment=1,
            ),
            SearchEvaluation(
                attempt=_attempt(5, "accepted", "event-after-stop"),
                candidate={"event_id": "event-after-stop"},
                seen_mother_increment=1,
            ),
        )
        for row in rows:
            visited.append(row.attempt.reason)
            yield row

    result = consume_search_evaluations(
        evaluations(), sample_count=3, max_seen_mothers=20
    )

    assert [row.status for row in result.attempts] == [
        "generator_deficit",
        "partial_pair_group",
        "accepted",
        "accepted",
        "accepted",
    ]
    assert result.selected_event_ids == ("event-a", "event-b", "event-c")
    assert result.complete is True
    assert "event-after-stop" not in visited


def test_consume_search_evaluations_stops_at_seen_mother_bound() -> None:
    evaluations = (
        SearchEvaluation(
            attempt=_attempt(index, "partial_pair_group", f"event-{index}"),
            candidate=None,
            seen_mother_increment=1,
        )
        for index in range(5)
    )

    result = consume_search_evaluations(
        evaluations, sample_count=3, max_seen_mothers=2
    )

    assert result.complete is False
    assert result.seen_mother_count == 2
    assert len(result.attempts) == 2


@pytest.fixture(scope="module")
def audited_candidate() -> tuple[SelectedAuditCandidate, dict, object]:
    config, _, base, oracle, trajectory, snippet, mother = _mother_inputs()
    paired_config = load_paired_variant_config("configs/paired_variants.yaml")
    partial = generate_paired_variants(
        mother_event=mother,
        source_snippet=snippet,
        base_state=base,
        trajectory=trajectory,
        oracle_context=oracle,
        base_config=config,
        paired_config=paired_config,
        seed=20260716,
    )
    source = partial.by_kind["near_miss"]
    temporal = replace(
        source,
        variant_kind="temporal_safe",
        world=replace(
            source.world,
            metadata={
                **source.world.metadata,
                "paired_variant_kind": "temporal_safe",
            },
        ),
        temporal_offset_s=0.8,
        lateral_offset_m=None,
        radial_shift_m=None,
        rotation_rad=None,
    )
    variants = {variant.variant_kind: variant for variant in partial.variants}
    variants["temporal_safe"] = temporal
    group = assemble_paired_event_group(
        pair_group_id=partial.pair_group_id,
        variants=variants,
        missing_variant_reasons={},
        paired_config=paired_config,
    )
    rendered = RenderedSop06Group(
        pair_group_id=group.pair_group_id,
        variant_kinds=VARIANT_ORDER,
        observations=tuple(object() for _ in VARIANT_ORDER),
        coverage_mask=(True,) * 6,
        is_complete=True,
        audit_certified=True,
    )
    candidate = SelectedAuditCandidate(
        schedule_rank=7,
        pair_seed=20260716,
        paired_seed=20260717,
        event=mother,
        group=group,
        rendered_group=rendered,
        base_state=base,
        oracle_context=oracle,
        trajectory=trajectory,
        source_snippet=snippet,
    )
    return candidate, config, paired_config


def test_build_candidate_evidence_recomputes_history_labels_and_skeleton(
    audited_candidate,
) -> None:
    candidate, config, paired_config = audited_candidate

    evidence = build_candidate_evidence(
        candidate,
        base_config=config,
        paired_config=paired_config,
    )

    assert evidence.audit["history_visibility"]["vector"] == [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert evidence.audit["history_visibility"]["regime"] == (
        "seen_then_occluded"
    )
    assert evidence.audit["history_visibility"]["trailing_hidden_frames"] == 2
    assert evidence.audit["scientific_checks"]["shared_skeleton"] is True
    assert evidence.audit["scientific_checks"]["complete_sixpack"] is True
    assert all(
        row["label_predicate_passed"]
        for kind, row in evidence.audit["variants"].items()
        if kind != "empty_blind_spot"
    )
    assert tuple(variant.kind for variant in evidence.visual_bundle.variants) == (
        VARIANT_ORDER
    )


def test_build_candidate_evidence_carries_joint_search_summary(
    audited_candidate,
) -> None:
    candidate, config, paired_config = audited_candidate
    joint_summary = {
        "algorithm_version": "seen_occluded_joint_visual_audit_v4",
        "complete": True,
        "rejection_counts": {"placement:static_collision": 4},
    }

    evidence = build_candidate_evidence(
        replace(candidate, joint_search_summary=joint_summary),
        base_config=config,
        paired_config=paired_config,
    )

    assert evidence.audit["joint_search_summary"] == joint_summary


def test_build_candidate_evidence_rejects_changed_static_skeleton(
    audited_candidate,
) -> None:
    candidate, config, paired_config = audited_candidate
    changed_variant = candidate.group.variants[1]
    changed_static = changed_variant.world.static_occupancy.copy()
    changed_static[0, 0] = np.float32(1.0)
    changed_variant = replace(
        changed_variant,
        world=replace(changed_variant.world, static_occupancy=changed_static),
    )
    changed_group = replace(
        candidate.group,
        variants=(candidate.group.variants[0], changed_variant, *candidate.group.variants[2:]),
    )

    with pytest.raises(SeenOccludedAuditError, match="static skeleton"):
        build_candidate_evidence(
            replace(candidate, group=changed_group),
            base_config=config,
            paired_config=paired_config,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_renderer(bundle, output_dir) -> VisualArtifactResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "event_replay.gif"
    png_path = output_dir / "paired_events.png"
    frames = []
    for index in range(23):
        frame = Image.new("P", (1200, 900), color=index + 1)
        frame.putpixel((index, index), 255)
        frames.append(frame)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=250,
        loop=0,
        disposal=2,
    )
    png = Image.new("RGB", (2100, 1200), color=(245, 245, 245))
    for index in range(200):
        png.putpixel((index, index), (10, 80, 120))
    png.save(png_path)
    return VisualArtifactResult(
        event_replay_path=gif_path,
        paired_events_path=png_path,
        event_replay_metadata={
            "format": "GIF",
            "width": 1200,
            "height": 900,
            "frame_count": 23,
            "frame_duration_ms": 250,
            "loop": 0,
            "bytes": gif_path.stat().st_size,
            "sha256": _sha256(gif_path),
        },
        paired_events_metadata={
            "format": "PNG",
            "width": 2100,
            "height": 1200,
            "panel_order": list(VARIANT_ORDER),
            "empty_target_removed": True,
            "shared_axes": [-4.0, 4.0, -4.0, 4.0],
            "bytes": png_path.stat().st_size,
            "sha256": _sha256(png_path),
        },
    )


def _publication_search_result(
    evidence: CandidateEvidence,
    *,
    count: int,
) -> SeenOccludedSearchResult:
    candidates = []
    attempts = []
    event_ids = []
    for index in range(count):
        event_id = f"event-publish-{index}"
        audit = deepcopy(evidence.audit)
        audit["generated_event_id"] = event_id
        visual = replace(evidence.visual_bundle, event_id=event_id)
        candidates.append(CandidateEvidence(visual_bundle=visual, audit=audit))
        event_ids.append(event_id)
        attempts.append(_attempt(index, "accepted", event_id))
    return SeenOccludedSearchResult(
        attempts=tuple(attempts),
        candidates=tuple(candidates),
        selected_event_ids=tuple(event_ids),
        seen_mother_count=count,
        complete=count == 3,
    )


def test_publish_complete_collection_binds_checksums_and_rejects_tampering(
    tmp_path: Path,
    audited_candidate,
) -> None:
    candidate, config, paired_config = audited_candidate
    evidence = build_candidate_evidence(
        candidate, base_config=config, paired_config=paired_config
    )
    request = _request(tmp_path)
    result = publish_audit_collection(
        request,
        _publication_search_result(evidence, count=3),
        context={"input_lock": {"schema_version": "3.0.0"}},
        renderer=_fake_renderer,
    )

    assert result.status == "complete"
    assert result.exit_code == 0
    assert (request.output_dir / ".audit-complete").is_file()
    loaded = load_audit_collection(request.output_dir, require_complete=True)
    assert loaded["selected_event_ids"] == [
        "event-publish-0",
        "event-publish-1",
        "event-publish-2",
    ]
    checksum_lines = (
        request.output_dir / "artifact_checksums.sha256"
    ).read_text(encoding="utf-8").splitlines()
    assert checksum_lines == sorted(checksum_lines)

    paired = (
        request.output_dir
        / "sample_01_event-pu"
        / "paired_events.png"
    )
    paired.write_bytes(paired.read_bytes() + b"tamper")
    with pytest.raises(SeenOccludedAuditError, match="checksum"):
        load_audit_collection(request.output_dir, require_complete=True)


def test_publish_insufficient_collection_has_no_completion_marker(
    tmp_path: Path,
    audited_candidate,
) -> None:
    candidate, config, paired_config = audited_candidate
    evidence = build_candidate_evidence(
        candidate, base_config=config, paired_config=paired_config
    )
    request = _request(tmp_path, output_dir=tmp_path / "insufficient")
    result = publish_audit_collection(
        request,
        _publication_search_result(evidence, count=2),
        context={"input_lock": {"schema_version": "3.0.0"}},
        renderer=_fake_renderer,
    )

    assert result.status == "insufficient_complete_samples"
    assert result.exit_code == 3
    assert not (request.output_dir / ".audit-complete").exists()
    loaded = load_audit_collection(request.output_dir, require_complete=False)
    assert loaded["status"] == "insufficient_complete_samples"
    with pytest.raises(SeenOccludedAuditError, match="not complete"):
        load_audit_collection(request.output_dir, require_complete=True)


def test_prepare_real_audit_maps_request_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.seen_occluded_visual_audit as audit_module

    config, grid, _, _, _, snippet, libraries = _paired_source_inputs()
    captured = []
    fake_prepared = SimpleNamespace(
        request=None,
        base_config=config,
        grid=grid,
        sop03=SimpleNamespace(typed_libraries=libraries),
        sop04=SimpleNamespace(by_id={}),
        schedule=(SimpleNamespace(rank=0),),
        input_lock={"version": "input-lock"},
        producer_source_identity={"commit": "abc"},
        run_id="sop05-run-test",
        base_config_sha256="b" * 64,
        generator_config_sha256="c" * 64,
        generator_config_semantic_digest="d" * 32,
        target_history_visibility_policy_digest="e" * 32,
    )

    def fake_prepare(request):
        captured.append(request)
        fake_prepared.request = request
        return fake_prepared

    monkeypatch.setattr(audit_module, "prepare_sop05_run", fake_prepare)
    request = _request(
        tmp_path,
        base_config_path=ROOT / "configs/base.yaml",
        generator_config_path=ROOT / "configs/generator_test.yaml",
        paired_config_path=ROOT / "configs/paired_variants.yaml",
        joint_config_path=(
            ROOT / "configs/seen_occluded_joint_visual_audit.yaml"
        ),
    )

    prepared = prepare_real_audit(request)

    assert len(captured) == 1
    assert captured[0].accepted_quota == 3
    assert captured[0].workers == 8
    assert captured[0].output_dir == request.output_dir
    assert prepared.snippets_by_id == {snippet.snippet_id: snippet}
    assert prepared.context["sop05_run_id"] == "sop05-run-test"
    assert prepared.context["paired_config_digest"] == (
        prepared.paired_config.digest
    )
    assert prepared.context["joint_config_digest"] == (
        prepared.joint_config.digest
    )
    assert not request.output_dir.exists()


def test_real_search_records_exact_partial_sixpack_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.seen_occluded_visual_audit as audit_module

    config, _, base, oracle, trajectory, snippet, mother = _mother_inputs()
    paired_config = load_paired_variant_config("configs/paired_variants.yaml")
    pair = SimpleNamespace(
        rank=0,
        state_id=base.state_id,
        trajectory_id=trajectory.trajectory_id,
        pair_seed=20260716,
    )
    sop05 = SimpleNamespace(
        request=SimpleNamespace(workers=1),
        base_config=config,
        generator_config={"normalized": True},
        grid=build_grid_spec(config),
        schedule=(pair,),
        sop03=SimpleNamespace(
            load_pair=lambda *_: (base, oracle),
            typed_libraries={},
        ),
        sop04=SimpleNamespace(by_id={trajectory.trajectory_id: trajectory}),
    )
    prepared = PreparedRealAudit(
        request=_request(tmp_path, workers=1),
        sop05=sop05,
        paired_config=paired_config,
        snippets_by_id={snippet.snippet_id: snippet},
        context={},
        joint_config=load_joint_audit_search_config(
            "configs/seen_occluded_joint_visual_audit.yaml"
        ),
    )
    monkeypatch.setattr(
        audit_module,
        "generate_events",
        lambda **_: SimpleNamespace(
            events=(mother,),
            summary={"history_visibility_deficits": {}},
        ),
    )
    monkeypatch.setattr(
        audit_module,
        "search_joint_audit_group",
        lambda **_: pytest.fail("non-repairable group used joint fallback"),
    )

    evaluations = list(iter_real_search_evaluations(prepared))

    assert [row.attempt.status for row in evaluations] == [
        "pair_generated",
        "partial_pair_group",
    ]
    assert evaluations[1].attempt.history_regime == "seen_then_occluded"
    assert evaluations[1].attempt.coverage_mask == (
        True,
        True,
        False,
        True,
        True,
        True,
    )
    assert evaluations[1].attempt.missing_variant_reasons == {
        "temporal_safe": "temporal_variant_still_collides"
    }


def test_joint_fallback_accepts_only_occluder_repairable_spatial_gap() -> None:
    import src.evaluation.seen_occluded_visual_audit as audit_module

    config, _, base, oracle, trajectory, snippet, mother = _mother_inputs()
    paired_config = load_paired_variant_config(
        "configs/paired_variants_visual_audit.yaml"
    )
    partial = generate_paired_variants(
        mother_event=mother,
        source_snippet=snippet,
        base_state=base,
        trajectory=trajectory,
        oracle_context=oracle,
        base_config=config,
        paired_config=paired_config,
        seed=20260716,
    )
    joint_config = load_joint_audit_search_config(
        "configs/seen_occluded_joint_visual_audit.yaml"
    )
    repairable = replace(
        partial,
        coverage_mask=(True, True, False, False, True, True),
        missing_variant_reasons={
            "temporal_safe": "target_current_visible",
            "spatial_safe": "target_occluder_collision",
        },
    )
    nonrepairable = replace(
        repairable,
        missing_variant_reasons={
            "temporal_safe": "target_current_visible",
            "spatial_safe": "target_context_collision",
        },
    )

    assert audit_module._joint_fallback_is_eligible(
        repairable, joint_config=joint_config
    )
    assert not audit_module._joint_fallback_is_eligible(
        nonrepairable, joint_config=joint_config
    )


@pytest.mark.parametrize(
    "repair_reason",
    ("target_current_visible", "target_occluder_collision"),
)
def test_real_search_uses_joint_fallback_for_repairable_temporal_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repair_reason: str,
) -> None:
    import src.evaluation.seen_occluded_visual_audit as audit_module

    config, _, base, oracle, trajectory, snippet, mother = _mother_inputs()
    paired_config = load_paired_variant_config(
        "configs/paired_variants_visual_audit.yaml"
    )
    formal_partial = generate_paired_variants(
        mother_event=mother,
        source_snippet=snippet,
        base_state=base,
        trajectory=trajectory,
        oracle_context=oracle,
        base_config=config,
        paired_config=paired_config,
        seed=20260716,
    )
    assert formal_partial.coverage_mask == (
        True,
        True,
        False,
        True,
        True,
        True,
    )
    repairable = replace(
        formal_partial,
        missing_variant_reasons={"temporal_safe": repair_reason},
    )
    complete_group = replace(
        formal_partial,
        coverage_mask=(True,) * 6,
        missing_variant_reasons={},
        is_complete=True,
        eligible_for_strict_evaluation=True,
    )
    pair = SimpleNamespace(
        rank=0,
        state_id=base.state_id,
        trajectory_id=trajectory.trajectory_id,
        pair_seed=20260716,
    )
    sop05 = SimpleNamespace(
        request=SimpleNamespace(workers=1),
        base_config=config,
        generator_config=load_generator_config(
            "configs/generator_seen_occluded_visual_audit.yaml"
        ),
        grid=build_grid_spec(config),
        schedule=(pair,),
        sop03=SimpleNamespace(
            load_pair=lambda *_: (base, oracle),
            typed_libraries={},
        ),
        sop04=SimpleNamespace(by_id={trajectory.trajectory_id: trajectory}),
    )
    joint_config = load_joint_audit_search_config(
        "configs/seen_occluded_joint_visual_audit.yaml"
    )
    prepared = PreparedRealAudit(
        request=_request(tmp_path, workers=1),
        sop05=sop05,
        paired_config=paired_config,
        snippets_by_id={snippet.snippet_id: snippet},
        context={},
        joint_config=joint_config,
    )
    monkeypatch.setattr(
        audit_module,
        "generate_events",
        lambda **_: SimpleNamespace(events=(mother,), summary={}),
    )
    monkeypatch.setattr(
        audit_module,
        "generate_paired_variants",
        lambda **_: repairable,
    )
    joint_calls = []
    joint_summary = {
        "algorithm_version": joint_config.algorithm_version,
        "complete": True,
    }

    def joint_search(**kwargs):
        joint_calls.append(kwargs)
        return JointAuditSearchResult(
            mother_event=mother,
            group=complete_group,
            paired_seed=303,
            summary=joint_summary,
        )

    monkeypatch.setattr(audit_module, "search_joint_audit_group", joint_search)
    rendered = object()
    monkeypatch.setattr(
        audit_module,
        "render_sop06_complete_audit_group",
        lambda **_: rendered,
    )
    evidence = object()

    def build(selected, **_kwargs):
        assert selected.event is mother
        assert selected.group is complete_group
        assert selected.rendered_group is rendered
        assert selected.paired_seed == 303
        assert selected.joint_search_summary == joint_summary
        return evidence

    monkeypatch.setattr(audit_module, "build_candidate_evidence", build)

    evaluations = list(iter_real_search_evaluations(prepared))

    assert [row.attempt.status for row in evaluations] == [
        "pair_generated",
        "accepted",
    ]
    assert evaluations[1].candidate is evidence
    assert evaluations[1].attempt.joint_search_summary == joint_summary
    assert len(joint_calls) == 1
    assert joint_calls[0]["joint_config"] is joint_config


def test_generated_pair_reports_reuses_ordered_sop05_process_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluation.seen_occluded_visual_audit as audit_module

    schedule = (SimpleNamespace(rank=0), SimpleNamespace(rank=1))
    sop05 = SimpleNamespace(
        request=SimpleNamespace(workers=2),
        schedule=schedule,
    )
    prepared = PreparedRealAudit(
        request=_request(tmp_path, workers=2),
        sop05=sop05,
        paired_config=load_paired_variant_config("configs/paired_variants.yaml"),
        snippets_by_id={},
        context={},
    )
    calls = []
    cache = object()
    executor = SimpleNamespace()

    class Pool:
        def __enter__(self):
            calls.append("pool_enter")
            return executor

        def __exit__(self, *_):
            calls.append("pool_exit")

    monkeypatch.setattr(
        audit_module,
        "_prewarm_robot_sweep_cache",
        lambda value: cache if value is sop05 else pytest.fail("wrong sop05"),
    )
    monkeypatch.setattr(
        audit_module,
        "_make_pair_process_pool",
        lambda value, value_cache: Pool()
        if value is sop05 and value_cache is cache
        else pytest.fail("wrong process-pool arguments"),
    )

    def ordered(value_executor, value_schedule, *, max_in_flight):
        assert value_executor is executor
        assert value_schedule is schedule
        assert max_in_flight == 2
        yield "report-0"
        yield "report-1"

    monkeypatch.setattr(audit_module, "_iter_pair_reports_bounded", ordered)

    assert list(iter_generated_pair_reports(prepared)) == [
        "report-0",
        "report-1",
    ]
    assert calls == ["pool_enter", "pool_exit"]
