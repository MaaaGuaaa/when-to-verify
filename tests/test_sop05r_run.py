from __future__ import annotations

import hashlib
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

import src.generation.sop05r_run as run_module
from src.generation.sop05r_contracts import (
    SOP05R_ACTIVE_REVEALABILITY_VERSION,
    load_sop05r_config,
)
from src.generation.sop05r_event_sampler import evaluate_obstacle_first_template
from src.generation.sop05r_run import (
    Sop05rAcceptedEvent,
    Sop05rBaseGenerationReport,
    Sop05rPublicationContext,
    Sop05rScheduleEntry,
    Sop05rSelectionCandidate,
    Sop05rTemplateReport,
    build_sop05r_generation_collection,
    build_sop05r_schedule,
    collect_ranked_sop05r_reports,
    publish_sop05r_generation,
    select_sop05r_event_ids,
)
from src.planning.verification_actions import CANONICAL_ACTION_IDS
from src.utils.config import load_config
from tests.test_sop05r_event_sampler import _fixture as event_fixture


ROOT = Path(__file__).resolve().parents[1]


def _selection_candidate(
    index: int,
    *,
    regime: str,
    active: bool,
) -> Sop05rSelectionCandidate:
    return Sop05rSelectionCandidate(
        generated_event_id=f"event-selection-{index:03d}",
        base_state_id=f"base-selection-{index // 3:03d}",
        template_id=f"template-selection-{index:03d}",
        history_visibility_regime=regime,
        active_revealable=active,
        schedule_rank=index,
    )


def test_schedule_and_parallel_collection_are_rank_stable() -> None:
    schedule = build_sop05r_schedule(
        ("state-c", "state-a", "state-b", "state-d"),
        seed=29,
        max_base_states=3,
    )

    assert tuple(row.state_id for row in schedule) == (
        "state-a",
        "state-b",
        "state-c",
    )
    assert tuple(row.rank for row in schedule) == (0, 1, 2)
    assert schedule == build_sop05r_schedule(
        ("state-d", "state-b", "state-c", "state-a"),
        seed=29,
        max_base_states=3,
    )

    def evaluate(row: Sop05rScheduleEntry) -> tuple[int, int]:
        return row.rank, row.base_seed

    serial = collect_ranked_sop05r_reports(
        schedule,
        evaluate=evaluate,
        workers=1,
    )
    parallel = collect_ranked_sop05r_reports(
        schedule,
        evaluate=evaluate,
        workers=3,
        executor_factory=ThreadPoolExecutor,
    )
    assert serial == parallel
    assert tuple(rank for rank, _ in parallel) == (0, 1, 2)


def test_selection_enforces_history_and_training_revealability_quotas() -> None:
    config = load_sop05r_config(
        ROOT / "configs/generator_obstacle_first_train.yaml"
    )
    candidates = tuple(
        [
            _selection_candidate(
                index,
                regime="seen_then_occluded",
                active=index < 6,
            )
            for index in range(8)
        ]
        + [
            _selection_candidate(
                8 + index,
                regime="unseen_in_history_window",
                active=index == 0,
            )
            for index in range(2)
        ]
    )

    selected = select_sop05r_event_ids(
        candidates,
        accepted_quota=10,
        seed=41,
        config=config,
    )

    assert selected.requested_history_counts == {
        "seen_then_occluded": 8,
        "unseen_in_history_window": 2,
    }
    assert selected.selected_history_counts == selected.requested_history_counts
    assert selected.active_revealable_requested_count == 7
    assert selected.active_revealable_selected_count == 7
    assert selected.natural_difficult_selected_count == 3
    assert selected.deficits == {
        "seen_then_occluded": 0,
        "unseen_in_history_window": 0,
        "active_revealable": 0,
    }
    assert selected.quota_met


def test_selection_never_backfills_history_or_hides_active_deficits() -> None:
    config = load_sop05r_config(
        ROOT / "configs/generator_obstacle_first_train.yaml"
    )
    history_shortfall = tuple(
        _selection_candidate(
            index,
            regime="seen_then_occluded",
            active=True,
        )
        for index in range(10)
    ) + (
        _selection_candidate(
            20,
            regime="unseen_in_history_window",
            active=True,
        ),
    )
    selected = select_sop05r_event_ids(
        history_shortfall,
        accepted_quota=10,
        seed=7,
        config=config,
    )
    assert selected.selected_history_counts == {
        "seen_then_occluded": 8,
        "unseen_in_history_window": 1,
    }
    assert selected.deficits["unseen_in_history_window"] == 1
    assert len(selected.event_ids) == 9
    assert not selected.quota_met

    active_shortfall = tuple(
        _selection_candidate(
            index,
            regime=(
                "seen_then_occluded"
                if index < 8
                else "unseen_in_history_window"
            ),
            active=index < 6,
        )
        for index in range(10)
    )
    selected = select_sop05r_event_ids(
        active_shortfall,
        accepted_quota=10,
        seed=7,
        config=config,
    )
    assert len(selected.event_ids) == 10
    assert selected.active_revealable_selected_count == 6
    assert selected.deficits["active_revealable"] == 1
    assert not selected.quota_met


def test_validation_selection_records_revealability_without_filtering() -> None:
    config = load_sop05r_config(
        ROOT / "configs/generator_obstacle_first_test.yaml"
    )
    candidates = tuple(
        _selection_candidate(
            index,
            regime=(
                "seen_then_occluded"
                if index < 8
                else "unseen_in_history_window"
            ),
            active=False,
        )
        for index in range(10)
    )

    selected = select_sop05r_event_ids(
        candidates,
        accepted_quota=10,
        seed=3,
        config=config,
    )

    assert len(selected.event_ids) == 10
    assert selected.active_revealable_requested_count == 0
    assert selected.active_revealable_selected_count == 0
    assert selected.deficits["active_revealable"] == 0
    assert selected.quota_met


def _accepted_event(*, active: bool = True) -> Sop05rAcceptedEvent:
    base_config, config, base_state, context, template, _ = event_fixture()
    with patch(
        "src.generation.sop05r_event_sampler._target_history_visibility",
        return_value=np.asarray(
            [True, True, True, True, False, False, False, False],
            dtype=np.bool_,
        ),
    ), patch(
        "src.generation.sop05r_event_sampler._target_physics_rejection",
        return_value=None,
    ):
        evaluation = evaluate_obstacle_first_template(
            template=template,
            base_state=base_state,
            oracle_context=context,
            base_config=base_config,
            config=config,
            seed=31,
        )
    assert evaluation.mother is not None
    mother = evaluation.mother
    active_ids = ["arc_left_30"] if active else []
    first_visible = {
        action_id: (0.4 if action_id == "arc_left_30" and active else None)
        for action_id in CANONICAL_ACTION_IDS
    }
    matched_wait = {
        action_id: (0.8 if action_id == "arc_left_30" and active else None)
        for action_id in CANONICAL_ACTION_IDS
    }
    metadata = {
        **mother.event.world.metadata,
        "active_revealability_version": SOP05R_ACTIVE_REVEALABILITY_VERSION,
        "active_revealability_status": (
            "active_revealable" if active else "natural_difficult"
        ),
        "active_revealable_action_ids": active_ids,
        "first_visible_time_by_verification_action": first_visible,
        "matched_wait_visible_time": matched_wait,
        "active_revealability_actions": {
            action_id: {
                "active_revealable": action_id in active_ids,
                "first_visible_time_s": first_visible[action_id],
                "matched_wait_visible_time_s": matched_wait[action_id],
            }
            for action_id in CANONICAL_ACTION_IDS
        },
    }
    event = replace(
        mother.event,
        world=replace(mother.event.world, metadata=metadata),
    )
    return Sop05rAcceptedEvent(
        event=event,
        trajectory_record=mother.trajectory_record,
        template_id=template.template_id,
        schedule_rank=0,
        attempts_before_acceptance=1,
        active_revealable=active,
    )


def _complete_publication_inputs(
    *,
    accepted_quota: int = 1,
) -> tuple[Sop05rPublicationContext, object]:
    accepted = _accepted_event(active=True)
    event = accepted.event
    row = Sop05rTemplateReport(
        report_version="sop05r_template_report_v1",
        base_rank=0,
        state_id=event.world.base_state_id,
        template_id=accepted.template_id,
        template_schedule_rank=(0, 0, 1, 0, 1, 0),
        attempt_index=0,
        geometry_eligible=True,
        planner_feasible=True,
        exact_history_qualified=True,
        time_aligned_collision=True,
        active_revealable=True,
        generated_event_id=event.generated_event_id,
        history_visibility_regime="seen_then_occluded",
        rejection_reason=None,
    )
    entry = Sop05rScheduleEntry(
        rank=0,
        state_id=event.world.base_state_id,
        base_seed=31,
    )
    report = Sop05rBaseGenerationReport(
        schedule_entry=entry,
        template_reports=(row,),
        accepted_events=(accepted,),
    )
    config = load_sop05r_config(
        ROOT / "configs/generator_obstacle_first_train.yaml"
    )
    collection = build_sop05r_generation_collection(
        (report,),
        accepted_quota=accepted_quota,
        seed=31,
        config=config,
    )
    verification_config = yaml.safe_load(
        (ROOT / "configs/verification_actions.yaml").read_text(encoding="utf-8")
    )
    base_config = load_config(ROOT / "configs/base.yaml")
    canonical = lambda value: json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    input_lock = {
        "version": "sop05r_input_lock_v1",
        "split": "train",
        "sop03": {
            "code_commit": "1" * 40,
            "checksum_manifest_sha256": "2" * 64,
            "audit_sha256": "3" * 64,
            "completion_policy": "sop03_complete_marker_v1",
        },
        "sop04_trajectory_bank_is_input": False,
        "base_config_sha256": hashlib.sha256(canonical(base_config)).hexdigest(),
        "sop05r_config_digest": config.digest,
        "verification_action_config_sha256": hashlib.sha256(
            canonical(verification_config)
        ).hexdigest(),
        "schedule_sha256": hashlib.sha256(
            canonical([entry.as_dict()])
        ).hexdigest(),
        "versions": {
            "generator_algorithm_version": "obstacle_first_event_generation_v1",
            "run_producer_version": "sop05r_generation_run_v1",
            "selection_version": "sop05r_stratified_selection_v1",
            "report_version": "sop05r_template_report_v1",
        },
    }
    context = Sop05rPublicationContext(
        run_id="sop05r-run-fixture",
        split="train",
        seed=31,
        accepted_quota=accepted_quota,
        base_config=base_config,
        config=config,
        verification_action_config=verification_config,
        input_lock=input_lock,
        producer_source_identity={
            "version": "sop05_producer_source_identity_v1",
            "git_commit": "4" * 40,
            "worktree_state": "clean",
            "dirty_tree_sha256": None,
        },
        schedule=(entry,),
    )
    return context, collection


def test_process_transport_round_trips_accepted_event_identity() -> None:
    accepted = _accepted_event(active=True)
    entry = Sop05rScheduleEntry(
        rank=0,
        state_id=accepted.event.world.base_state_id,
        base_seed=31,
    )
    report = Sop05rBaseGenerationReport(
        schedule_entry=entry,
        template_reports=(),
        accepted_events=(accepted,),
    )

    transported = pickle.loads(
        pickle.dumps(run_module._transport_base_report(report))
    )
    restored = run_module._restore_base_report(transported)

    assert restored.schedule_entry == report.schedule_entry
    assert restored.accepted_events[0].event.generated_event_id == (
        accepted.event.generated_event_id
    )
    assert (
        restored.accepted_events[0].event.target_motion_record.record_digest
        == accepted.event.target_motion_record.record_digest
    )
    assert restored.accepted_events[0].trajectory_record.candidate_trajectory_ids == (
        accepted.trajectory_record.candidate_trajectory_ids
    )


def test_publication_round_trips_all_denominators_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    context, collection = _complete_publication_inputs()
    output = tmp_path / "sop05r"

    result = publish_sop05r_generation(output, context, collection)

    assert result.run_state == "complete"
    assert result.exit_code == 0
    assert len(result.publication_semantic_digest) == 64
    assert (output / ".sop05r-complete").is_file()
    assert {
        "input_base_count",
        "geometry_eligible_base_count",
        "template_count",
        "planner_feasible_template_count",
        "exact_history_qualified_count",
        "time_aligned_collision_count",
        "active_revealable_count",
        "accepted_count",
        "selected_count",
        "attempts_per_accepted_event",
        "rejection_reasons",
    } <= set(result.generation_summary)
    checksum_lines = (output / "artifact_checksums.sha256").read_text(
        encoding="ascii"
    ).splitlines()
    assert checksum_lines == sorted(checksum_lines)
    with pytest.raises(FileExistsError):
        publish_sop05r_generation(output, context, collection)


def test_quota_unmet_publication_has_no_completion_marker(tmp_path: Path) -> None:
    context, collection = _complete_publication_inputs(accepted_quota=2)
    output = tmp_path / "sop05r-partial"

    result = publish_sop05r_generation(output, context, collection)

    assert result.run_state == "quota_unmet"
    assert result.exit_code == 4
    assert not (output / ".sop05r-complete").exists()
    summary = json.loads(
        (output / "generation_summary.json").read_text(encoding="ascii")
    )
    assert summary["selected_count"] == 1
    assert summary["quota_met"] is False


def test_zero_acceptance_publishes_loadable_diagnostic(tmp_path: Path) -> None:
    context, _ = _complete_publication_inputs()
    entry = context.schedule[0]
    rejected = Sop05rTemplateReport(
        report_version="sop05r_template_report_v1",
        base_rank=0,
        state_id=entry.state_id,
        template_id="template-rejected",
        template_schedule_rank=(0, 0, 0, 0, 0, 0),
        attempt_index=0,
        geometry_eligible=False,
        planner_feasible=False,
        exact_history_qualified=False,
        time_aligned_collision=False,
        active_revealable=False,
        generated_event_id=None,
        history_visibility_regime=None,
        rejection_reason="obstacle_out_of_bounds",
    )
    collection = build_sop05r_generation_collection(
        (
            Sop05rBaseGenerationReport(
                schedule_entry=entry,
                template_reports=(rejected,),
                accepted_events=(),
            ),
        ),
        accepted_quota=1,
        seed=context.seed,
        config=context.config,
    )
    output = tmp_path / "sop05r-empty"

    result = publish_sop05r_generation(output, context, collection)

    assert result.run_state == "quota_unmet"
    assert result.generation_summary["selected_count"] == 0
    assert not (output / ".sop05r-complete").exists()
    from src.generation.sop05r_output_loader import load_sop05r_events

    loaded = load_sop05r_events(output, require_complete=False)
    assert loaded.events == ()
    assert loaded.trajectory_store.records == ()
    assert loaded.target_motion.records == ()


def test_publication_semantic_digest_is_independent_of_output_path(
    tmp_path: Path,
) -> None:
    context, collection = _complete_publication_inputs()

    first = publish_sop05r_generation(tmp_path / "first", context, collection)
    second = publish_sop05r_generation(tmp_path / "second", context, collection)

    assert first.publication_semantic_digest == second.publication_semantic_digest
    assert hashlib.sha256(
        (first.output_dir / "events.jsonl").read_bytes()
    ).hexdigest() == hashlib.sha256(
        (second.output_dir / "events.jsonl").read_bytes()
    ).hexdigest()
    assert (first.output_dir / "manifest.json").read_bytes() == (
        second.output_dir / "manifest.json"
    ).read_bytes()
