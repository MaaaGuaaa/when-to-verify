import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
import numpy as np

from src.contracts import build_grid_spec
from src.generation.event_target_motion_shard import write_event_target_motion_shard
from src.generation.sop05r_teb_output_loader import load_sop05r_teb_output
from src.generation.sop05r_teb_event_sampler import build_sop05r_teb_mother
from src.generation.sop05r_teb_run import publish_sop05r_teb_run
from src.generation.sop05r_teb_trajectory_store import (
    publish_sop05r_teb_trajectory_store,
)
from tests.test_sop05r_teb_event_sampler import _mother_fixture


def _base_config() -> dict[str, object]:
    with Path("configs/base.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _run_request(
    tmp_path: Path,
    *,
    workers: int,
    accepted_quota: int | None = 1,
):
    from src.generation.sop05r_teb_run import Sop05rTebRunRequest

    return Sop05rTebRunRequest(
        sop03_root=tmp_path / "unused-sop03",
        long40_human_artifact=tmp_path / "unused-long40",
        split="train",
        base_config_path=Path("configs/base.yaml"),
        generator_config_path=Path(
            "configs/generator_obstacle_first_teb_train.yaml"
        ),
        verification_action_config_path=Path("configs/verification_actions.yaml"),
        output_dir=tmp_path / f"run-w{workers}-q{accepted_quota}",
        seed=310725,
        accepted_quota=accepted_quota,
        max_base_states=2,
        checksum_workers=1,
        workers=workers,
        git_executable=Path("git"),
    )


def test_ordered_base_results_use_requested_processes_and_preserve_rank(
    monkeypatch,
) -> None:
    from src.generation import sop05r_teb_run as run_module

    ordered_results = getattr(run_module, "_ordered_base_state_results", None)
    assert callable(ordered_results)

    def identify_process(indexed_state_pair):
        time.sleep(0.05)
        return indexed_state_pair[0], os.getpid()

    monkeypatch.setattr(
        run_module,
        "_generate_base_state_result",
        identify_process,
    )
    indexed = tuple((rank, None) for rank in range(8))
    serial = tuple(
        ordered_results(indexed, context=None, workers=1)
    )
    parallel = tuple(
        ordered_results(indexed, context=None, workers=2)
    )

    assert [rank for rank, _ in serial] == list(range(8))
    assert [rank for rank, _ in parallel] == list(range(8))
    assert {pid for _, pid in serial} == {os.getpid()}
    assert len({pid for _, pid in parallel}) == 2


def test_ordered_base_results_accept_empty_input_with_multiple_workers() -> None:
    from src.generation.sop05r_teb_run import _ordered_base_state_results

    assert tuple(
        _ordered_base_state_results((), context=None, workers=2)
    ) == ()


def test_zero_quota_skips_base_state_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.generation import sop05r_teb_run as run_module

    monkeypatch.setattr(run_module, "load_verification_actions", lambda _: None)
    monkeypatch.setattr(
        run_module,
        "load_sop05r_teb_long40_inputs",
        lambda **_: SimpleNamespace(
            snippets=(),
            state_pairs=((None, None),),
            source_evidence={"producer": "test"},
        ),
    )

    def fail_generation(*args, **kwargs):
        raise AssertionError("zero quota started base-state generation")

    monkeypatch.setattr(run_module, "_ordered_base_state_results", fail_generation)

    result = run_module.execute_sop05r_teb_run(
        _run_request(tmp_path, workers=2, accepted_quota=0)
    )

    assert result.complete
    assert result.accepted_count == 0
    assert result.generation_summary["denominator_counts"] == {}


def test_all_accepted_processes_every_base_state_and_publishes_actual_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.generation import sop05r_teb_run as run_module

    inputs = _mother_fixture()
    mother_evaluation = build_sop05r_teb_mother(
        base_config=inputs[0],
        source_base_state=inputs[1],
        source_oracle_context=inputs[2],
        teb_config=inputs[3],
        task_template=inputs[4],
        placement_result=inputs[5],
        snippet=inputs[6],
        seed=43,
    )
    assert mother_evaluation.mother is not None

    monkeypatch.setattr(run_module, "load_verification_actions", lambda _: None)
    monkeypatch.setattr(
        run_module,
        "load_sop05r_teb_long40_inputs",
        lambda **_: SimpleNamespace(
            snippets=(),
            state_pairs=((None, None), (None, None)),
            source_evidence={"producer": "test"},
        ),
    )

    def base_result(indexed_state_pair):
        rank, _ = indexed_state_pair
        return run_module._BaseStateGenerationResult(
            base_rank=rank,
            templates=(
                run_module._TemplateGenerationResult(
                    mother=mother_evaluation.mother if rank == 0 else None,
                    counters={"m4_attempted": 1, "m6_accepted": int(rank == 0)},
                    rejections={},
                    m5_candidate_counts={},
                    m5_candidate_rejections={},
                ),
            ),
        )

    monkeypatch.setattr(run_module, "_generate_base_state_result", base_result)
    progress: list[dict[str, object]] = []
    result = run_module.execute_sop05r_teb_run(
        _run_request(tmp_path, workers=1, accepted_quota=None),
        progress_callback=progress.append,
    )

    assert result.complete
    assert result.accepted_count == result.requested_count == 1
    assert [item["processed_base_states"] for item in progress] == [1, 2]
    assert all(item["requested_count"] is None for item in progress)
    assert all(item["completion_fraction"] is None for item in progress)
    loaded = load_sop05r_teb_output(result.output_dir, require_complete=True)
    assert loaded.manifest["source_evidence"]["accepted_selection"] == "all_accepted_v1"


def test_all_accepted_excludes_a_completed_existing_collection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.generation import sop05r_teb_run as run_module

    inputs = _mother_fixture()
    mother_evaluation = build_sop05r_teb_mother(
        base_config=inputs[0],
        source_base_state=inputs[1],
        source_oracle_context=inputs[2],
        teb_config=inputs[3],
        task_template=inputs[4],
        placement_result=inputs[5],
        snippet=inputs[6],
        seed=43,
    )
    assert mother_evaluation.mother is not None
    existing_root = tmp_path / "existing"
    publish_sop05r_teb_run(
        (mother_evaluation.mother,),
        existing_root,
        base_config=inputs[0],
        requested_count=1,
        config_digest=inputs[3].digest,
        verification_action_digest="a" * 64,
        source_evidence={"producer": "test"},
        denominator_counts={},
        rejection_counts={},
    )
    monkeypatch.setattr(run_module, "load_verification_actions", lambda _: None)
    monkeypatch.setattr(
        run_module,
        "load_sop05r_teb_long40_inputs",
        lambda **_: SimpleNamespace(
            snippets=(),
            state_pairs=((None, None),),
            source_evidence={"producer": "test"},
        ),
    )
    monkeypatch.setattr(
        run_module,
        "_generate_base_state_result",
        lambda indexed: run_module._BaseStateGenerationResult(
            base_rank=indexed[0],
            templates=(
                run_module._TemplateGenerationResult(
                    mother=mother_evaluation.mother,
                    counters={"m4_attempted": 1, "m6_accepted": 1},
                    rejections={},
                    m5_candidate_counts={},
                    m5_candidate_rejections={},
                ),
            ),
        ),
    )

    request = _run_request(tmp_path, workers=1, accepted_quota=None)
    request = replace(request, exclude_existing_output=existing_root)
    result = run_module.execute_sop05r_teb_run(request)

    assert result.complete
    assert result.accepted_count == 0
    assert result.generation_summary["denominator_counts"]["m6_excluded_existing"] == 1
    loaded = load_sop05r_teb_output(result.output_dir, require_complete=True)
    assert loaded.manifest["source_evidence"]["excluded_existing_event_count"] == 1


def test_run_reports_parent_aggregated_progress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.generation import sop05r_teb_run as run_module

    inputs = _mother_fixture()
    mother_evaluation = build_sop05r_teb_mother(
        base_config=inputs[0],
        source_base_state=inputs[1],
        source_oracle_context=inputs[2],
        teb_config=inputs[3],
        task_template=inputs[4],
        placement_result=inputs[5],
        snippet=inputs[6],
        seed=43,
    )
    assert mother_evaluation.mother is not None

    monkeypatch.setattr(run_module, "load_verification_actions", lambda _: None)
    monkeypatch.setattr(
        run_module,
        "load_sop05r_teb_long40_inputs",
        lambda **_: SimpleNamespace(
            snippets=(),
            state_pairs=((None, None), (None, None)),
            source_evidence={"producer": "test"},
        ),
    )

    def deterministic_base_result(indexed_state_pair):
        rank, _ = indexed_state_pair
        return run_module._BaseStateGenerationResult(
            base_rank=rank,
            templates=(
                run_module._TemplateGenerationResult(
                    mother=mother_evaluation.mother if rank == 0 else None,
                    counters={
                        "m4_attempted": 1,
                        "m4_accepted": 1,
                        "m5_attempted": 1,
                        "m5_accepted": int(rank == 0),
                        "m6_attempted": int(rank == 0),
                        "m6_accepted": int(rank == 0),
                    },
                    rejections={} if rank == 0 else {"m5_rejected": 1},
                    m5_candidate_counts={"tested_candidates": 1},
                    m5_candidate_rejections={},
                ),
            ),
        )

    monkeypatch.setattr(
        run_module,
        "_generate_base_state_result",
        deterministic_base_result,
    )
    progress: list[dict[str, object]] = []

    result = run_module.execute_sop05r_teb_run(
        _run_request(tmp_path, workers=1),
        progress_callback=progress.append,
    )

    assert result.complete
    assert progress == [
        {
            "progress_version": "sop05r_teb_progress_v1",
            "processed_base_states": 1,
            "total_base_states": 2,
            "accepted_count": 1,
            "requested_count": 1,
            "completion_fraction": 1.0,
            "denominator_counts": {
                "m4_accepted": 1,
                "m4_attempted": 1,
                "m5_accepted": 1,
                "m5_attempted": 1,
                "m6_accepted": 1,
                "m6_attempted": 1,
            },
            "rejection_counts": {},
            "elapsed_seconds": pytest.approx(progress[0]["elapsed_seconds"]),
            "accepted_per_second": pytest.approx(
                progress[0]["accepted_per_second"]
            ),
            "estimated_remaining_seconds": 0.0,
        }
    ]


def test_worker_count_preserves_semantic_digest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.generation import sop05r_teb_run as run_module

    inputs = _mother_fixture()
    mother_evaluation = build_sop05r_teb_mother(
        base_config=inputs[0],
        source_base_state=inputs[1],
        source_oracle_context=inputs[2],
        teb_config=inputs[3],
        task_template=inputs[4],
        placement_result=inputs[5],
        snippet=inputs[6],
        seed=43,
    )
    assert mother_evaluation.mother is not None
    mother = mother_evaluation.mother

    monkeypatch.setattr(run_module, "load_verification_actions", lambda _: None)
    monkeypatch.setattr(
        run_module,
        "load_sop05r_teb_long40_inputs",
        lambda **_: SimpleNamespace(
            snippets=(),
            state_pairs=((None, None), (None, None)),
            source_evidence={"producer": "test"},
        ),
    )

    def deterministic_base_result(indexed_state_pair):
        rank, _ = indexed_state_pair
        rejected = run_module._TemplateGenerationResult(
            mother=None,
            counters={"m4_attempted": 1},
            rejections={"teb_goal_unreached": 1},
            m5_candidate_counts={"tested_candidates": 3},
            m5_candidate_rejections={"decision_not_blocked": 3},
        )
        accepted = run_module._TemplateGenerationResult(
            mother=mother if rank == 0 else None,
            counters={
                "m4_attempted": 1,
                "m4_accepted": 1,
                "m5_attempted": 1,
                "m5_accepted": int(rank == 0),
                "m6_attempted": int(rank == 0),
                "m6_accepted": int(rank == 0),
            },
            rejections={} if rank == 0 else {"occlusion_witness_missing": 1},
            m5_candidate_counts={"tested_candidates": 1},
            m5_candidate_rejections={},
        )
        return run_module._BaseStateGenerationResult(
            base_rank=rank,
            templates=(rejected, accepted),
        )

    monkeypatch.setattr(
        run_module,
        "_generate_base_state_result",
        deterministic_base_result,
    )

    serial = run_module.execute_sop05r_teb_run(
        _run_request(tmp_path, workers=1)
    )
    parallel = run_module.execute_sop05r_teb_run(
        _run_request(tmp_path, workers=2)
    )
    serial_loaded = load_sop05r_teb_output(serial.output_dir)
    parallel_loaded = load_sop05r_teb_output(parallel.output_dir)

    assert [event.generated_event_id for event in serial_loaded.events] == [
        event.generated_event_id for event in parallel_loaded.events
    ]
    assert (
        serial.publication_semantic_digest
        == parallel.publication_semantic_digest
    )
    for key in (
        "denominator_counts",
        "rejection_counts",
        "m5_candidate_counts",
        "m5_candidate_rejection_counts",
    ):
        assert serial.generation_summary[key] == parallel.generation_summary[key]


def test_run_publishes_later_mother_after_m6_dynamics_rejection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.generation import sop05r_teb_run as run_module

    inputs = _mother_fixture()
    invalid_controls = np.asarray(inputs[4].route.sampled_controls).copy()
    invalid_controls[0, 0] += 0.5
    invalid_task = replace(
        inputs[4],
        route=replace(inputs[4].route, sampled_controls=invalid_controls),
    )
    template_evaluations = (
        SimpleNamespace(template=invalid_task, rejection_reason=None),
        SimpleNamespace(template=inputs[4], rejection_reason=None),
    )

    monkeypatch.setattr(run_module, "load_verification_actions", lambda _: None)
    monkeypatch.setattr(
        run_module,
        "load_sop05r_teb_long40_inputs",
        lambda **_: SimpleNamespace(
            snippets=(inputs[6],),
            state_pairs=((inputs[1], inputs[2]),),
            source_evidence={"producer": "test"},
        ),
    )
    monkeypatch.setattr(
        run_module,
        "iter_sop05r_teb_task_templates",
        lambda **_: iter(template_evaluations),
    )
    monkeypatch.setattr(
        run_module,
        "solve_anchored_human_placement",
        lambda **_: SimpleNamespace(
            result=inputs[5],
            rejection_reason=None,
            candidate_counts={},
            rejection_counts={},
        ),
    )

    result = run_module.execute_sop05r_teb_run(
        _run_request(tmp_path, workers=1)
    )
    loaded = load_sop05r_teb_output(result.output_dir, require_complete=True)

    assert result.complete
    assert len(loaded.events) == 1
    assert result.generation_summary["denominator_counts"]["m6_attempted"] == 2
    assert result.generation_summary["denominator_counts"]["m6_accepted"] == 1
    assert result.generation_summary["rejection_counts"] == {
        "teb_dynamics_limit": 1
    }


def test_run_propagates_h0_hidden_placement_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.generation import sop05r_teb_run as run_module

    inputs = _mother_fixture()
    template_evaluation = SimpleNamespace(template=inputs[4], rejection_reason=None)
    observed_modes: list[str] = []
    monkeypatch.setattr(run_module, "load_verification_actions", lambda _: None)
    monkeypatch.setattr(
        run_module,
        "load_sop05r_teb_long40_inputs",
        lambda **_: SimpleNamespace(
            snippets=(inputs[6],),
            state_pairs=((inputs[1], inputs[2]),),
            source_evidence={"producer": "test"},
        ),
    )
    monkeypatch.setattr(
        run_module,
        "iter_sop05r_teb_task_templates",
        lambda **_: iter((template_evaluation,)),
    )

    def solve(**kwargs):
        observed_modes.append(kwargs["selection_mode"])
        return SimpleNamespace(
            result=inputs[5],
            rejection_reason=None,
            candidate_counts={},
            rejection_counts={},
        )

    monkeypatch.setattr(run_module, "solve_anchored_human_placement", solve)
    request = replace(
        _run_request(tmp_path, workers=1),
        placement_selection_mode="h0_hidden",
    )

    result = run_module.execute_sop05r_teb_run(request)
    loaded = load_sop05r_teb_output(result.output_dir, require_complete=True)

    assert observed_modes == ["h0_hidden"]
    assert run_module.preflight_summary(request)["placement_selection_mode"] == (
        "h0_hidden"
    )
    assert loaded.manifest["source_evidence"]["placement_selection_mode"] == (
        "h0_hidden"
    )


def test_partial_v2_run_publishes_diagnostics_without_completion_marker(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    published = publish_sop05r_teb_run(
        (),
        output,
        base_config=_base_config(),
        requested_count=1,
        config_digest="a" * 64,
        verification_action_digest="b" * 64,
        source_evidence={"producer": "test"},
        denominator_counts={
            "m4_attempted": 3,
            "m4_accepted": 0,
            "m5_attempted": 0,
            "m5_accepted": 0,
            "m6_attempted": 0,
            "m6_accepted": 0,
        },
        rejection_counts={"teb_goal_unreached": 3},
    )

    loaded = load_sop05r_teb_output(output)

    assert not published.complete
    assert not loaded.complete
    assert loaded.events == ()
    assert loaded.summary["requested_count"] == 1
    assert loaded.summary["accepted_count"] == 0
    assert loaded.summary["rejection_counts"] == {"teb_goal_unreached": 3}
    assert not (output / "COMPLETE.json").exists()


def test_run_publishes_m5_candidate_evidence(tmp_path: Path) -> None:
    output = tmp_path / "run"
    published = publish_sop05r_teb_run(
        (),
        output,
        base_config=_base_config(),
        requested_count=1,
        config_digest="a" * 64,
        verification_action_digest="b" * 64,
        source_evidence={"producer": "test"},
        denominator_counts={"m5_attempted": 2},
        rejection_counts={"occlusion_witness_missing": 2},
        m5_candidate_counts={
            "visibility_preferred": 9,
            "visibility_fallback": 4,
        },
        m5_candidate_rejection_counts={"decision_not_blocked": 17},
    )

    assert published.summary["m5_candidate_counts"] == {
        "visibility_fallback": 4,
        "visibility_preferred": 9,
    }
    assert published.summary["m5_candidate_rejection_counts"] == {
        "decision_not_blocked": 17,
    }


def test_v2_run_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "run"
    kwargs = dict(
        base_config=_base_config(),
        requested_count=0,
        config_digest="a" * 64,
        verification_action_digest="b" * 64,
        source_evidence={"producer": "test"},
        denominator_counts={},
        rejection_counts={},
    )
    publish_sop05r_teb_run((), output, **kwargs)

    try:
        publish_sop05r_teb_run((), output, **kwargs)
    except FileExistsError:
        pass
    else:
        raise AssertionError("v2 run overwrote an existing output")


def test_complete_v2_run_reloads_one_m6_mother(tmp_path: Path) -> None:
    inputs = _mother_fixture()
    evaluation = build_sop05r_teb_mother(
        base_config=inputs[0],
        source_base_state=inputs[1],
        source_oracle_context=inputs[2],
        teb_config=inputs[3],
        task_template=inputs[4],
        placement_result=inputs[5],
        snippet=inputs[6],
        seed=43,
    )
    assert evaluation.mother is not None

    loaded = publish_sop05r_teb_run(
        (evaluation.mother,),
        tmp_path / "complete",
        base_config=inputs[0],
        requested_count=1,
        config_digest=inputs[3].digest,
        verification_action_digest="b" * 64,
        source_evidence={"producer": "test"},
        denominator_counts={"m6_accepted": 1},
        rejection_counts={},
    )

    assert loaded.complete
    assert len(loaded.events) == 1
    assert (
        loaded.events[0].generated_event_id
        == evaluation.mother.event.generated_event_id
    )
    assert len(loaded.trajectories.records) == 1


def test_publish_resumes_verified_trajectory_and_target_staging(tmp_path: Path) -> None:
    inputs = _mother_fixture()
    evaluation = build_sop05r_teb_mother(
        base_config=inputs[0],
        source_base_state=inputs[1],
        source_oracle_context=inputs[2],
        teb_config=inputs[3],
        task_template=inputs[4],
        placement_result=inputs[5],
        snippet=inputs[6],
        seed=43,
    )
    assert evaluation.mother is not None
    output = tmp_path / "recovered"
    staging_root = tmp_path / ".recovered.sop05r-teb-stage"
    staging = staging_root / output.name
    staging.mkdir(parents=True)
    publish_sop05r_teb_trajectory_store(
        (evaluation.mother.trajectory_record,),
        staging / "trajectory_store",
        requested_count=1,
        complete=True,
    )
    write_event_target_motion_shard(
        [evaluation.mother.event.target_motion_record],
        [evaluation.mother.event.world],
        staging / "target_motion",
        grid=build_grid_spec(inputs[0]),
    )

    loaded = publish_sop05r_teb_run(
        (evaluation.mother,),
        output,
        base_config=inputs[0],
        requested_count=1,
        config_digest=inputs[3].digest,
        verification_action_digest="b" * 64,
        source_evidence={"producer": "test"},
        denominator_counts={"m6_accepted": 1},
        rejection_counts={},
        resume_staging_root=staging_root,
    )

    assert loaded.complete
    assert (output / "COMPLETE.json").is_file()
    assert not staging_root.exists()
