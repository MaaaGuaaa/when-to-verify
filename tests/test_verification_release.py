"""Focused resume, rejection, and fork-order tests for verification releases."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from src.contracts import build_grid_spec
from src.datasets.verification_dataloader import load_verification_shard
from src.datasets.verification_dataset import build_verification_samples
from src.generation.sop06_finalized_source import Sop06AcceptedFinalRecord
from src.generation.verification_pipeline import (
    VERIFICATION_PIPELINE_VERSION,
    VerificationGroupResult,
    VerificationSourceIneligibleError,
)
import src.generation.verification_release as release_module
from src.generation.verification_release import (
    VerificationReleaseRequest,
    load_verification_release,
    load_verification_revaluation_records,
    publish_verification_release,
)
from src.planning.verification_actions import CANONICAL_ACTION_IDS
from tests.test_verification_dataset import _source_and_library


def _accepted(index: int) -> Sop06AcceptedFinalRecord:
    return Sop06AcceptedFinalRecord(
        source_index=index,
        mother_id=f"mother-{index:03d}",
        scenario_id=f"scenario-{index:03d}",
        split="train",
        regime="unseen_in_history_window",
        target_present=True,
        target_row=index,
    )


def _source(count: int, grid):
    source = SimpleNamespace(
        source_mode="complete_mother",
        source_publication_semantic_digest="a" * 64,
        final_release_identity="b" * 64,
        base_config={
            "bev": {
                "size": grid.height,
                "history_steps": grid.history_steps,
                "future_steps": grid.future_steps,
                "resolution_m": grid.resolution_m,
            }
        },
        accepted=tuple(_accepted(index) for index in range(count)),
    )
    source.prepare_boundary = lambda boundary: source
    return source


def _build_one_factory(template):
    def build_one(
        source,
        accepted,
        library,
        gt_config,
        max_replan_candidates,
    ) -> VerificationGroupResult:
        del gt_config, max_replan_candidates
        nominal = replace(
            template.nominal_trajectory,
            trajectory_id=f"nominal-{accepted.scenario_id}",
        )
        sampled_child_id = f"sampled-child-{accepted.scenario_id}"
        values = {
            action_id: replace(
                value,
                sampled_child_world_id=sampled_child_id,
                nominal_trajectory_id=nominal.trajectory_id,
            )
            for action_id, value in template.value_results.items()
        }
        group_input = replace(
            template,
            split=accepted.split,
            base_state_id=f"base-{accepted.scenario_id}",
            nominal_trajectory=nominal,
            value_results=values,
            provenance={
                **template.provenance,
                "release_task_id": accepted.scenario_id,
            },
        )
        samples = build_verification_samples(
            group_input,
            library=library,
            grid=build_grid_spec(dict(source.base_config)),
        )
        return VerificationGroupResult(
            version=VERIFICATION_PIPELINE_VERSION,
            samples=samples,
            values=values,
            sampled_child_world_id=sampled_child_id,
            infeasible_action_ids=(),
        )

    return build_one


@pytest.fixture
def release_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    grid, library, template = _source_and_library()
    config_root = tmp_path / "configs"
    config_root.mkdir()
    actions_path = config_root / "verification_actions.yaml"
    gt_path = config_root / "verification_gt.yaml"
    actions_path.write_text("fixture: actions\n", encoding="utf-8")
    gt_path.write_text("fixture: gt\n", encoding="utf-8")
    monkeypatch.setattr(release_module, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        release_module,
        "load_verification_actions",
        lambda path: library,
    )
    monkeypatch.setattr(
        release_module,
        "load_verification_gt_config",
        lambda path: object(),
    )
    return SimpleNamespace(
        root=tmp_path,
        grid=grid,
        library=library,
        template=template,
        actions_path=actions_path,
        gt_path=gt_path,
    )


def _request(
    harness,
    *,
    groups_per_shard: int,
    workers: int = 1,
) -> VerificationReleaseRequest:
    return VerificationReleaseRequest(
        source_family="natural",
        source_mode="complete_mother",
        source_root=harness.root / "inputs" / "source",
        final_scenario_root=harness.root / "inputs" / "final",
        split="train",
        output_dir=harness.root / "outputs" / "verification",
        actions_config_path=harness.actions_path,
        gt_config_path=harness.gt_path,
        workers=workers,
        groups_per_shard=groups_per_shard,
    )


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    assert isinstance(value, dict)
    return value


def test_release_resumes_one_completed_fixed_shard_then_replays_idempotently(
    release_harness,
) -> None:
    source = _source(5, release_harness.grid)
    request = _request(release_harness, groups_per_shard=2)
    build_one = _build_one_factory(release_harness.template)
    interrupted_calls: list[str] = []

    def interrupt(*args, **kwargs):
        accepted = args[1]
        interrupted_calls.append(accepted.scenario_id)
        if len(interrupted_calls) == 3:
            raise RuntimeError("intentional interruption")
        return build_one(*args, **kwargs)

    with pytest.raises(RuntimeError, match="intentional interruption"):
        publish_verification_release(
            request,
            source_loader=lambda value: source,
            build_one=interrupt,
        )

    in_progress = request.output_dir.parent / (
        f".{request.output_dir.name}.inprogress"
    )
    first_summary = _json(
        in_progress / "shards" / "shard-00000" / "task_summary.json"
    )
    assert interrupted_calls == [
        "scenario-000",
        "scenario-001",
        "scenario-002",
    ]
    assert first_summary["task_ids"] == ["scenario-000", "scenario-001"]
    assert first_summary["task_count"] == 2
    assert first_summary["accepted_group_count"] == 2
    assert first_summary["sample_count"] == 12
    stale = in_progress / "shards" / ".shard-00001.staging"
    stale.mkdir()
    (stale / "partial.tmp").write_bytes(b"incomplete")
    # Simulate interruption while publishing the root metadata. Completed
    # immutable task shards must remain reusable even if all metadata names
    # exist with truncated contents.
    (in_progress / "manifest.json").write_bytes(b'{"partial":')
    (in_progress / "checksums.json").write_bytes(b'{"partial":')
    (in_progress / "COMPLETE.json").write_bytes(b'{"partial":')

    resumed_calls: list[str] = []
    progress: list[tuple[int, int, bool]] = []

    def resumed(*args, **kwargs):
        resumed_calls.append(args[1].scenario_id)
        return build_one(*args, **kwargs)

    result = publish_verification_release(
        request,
        source_loader=lambda value: source,
        build_one=resumed,
        progress_callback=lambda *row: progress.append(row),
    )

    def fail_if_built(*args, **kwargs):
        raise AssertionError("idempotent replay rebuilt a completed task")

    replay = publish_verification_release(
        request,
        source_loader=lambda value: source,
        build_one=fail_if_built,
    )
    loaded = load_verification_release(request.output_dir)
    manifest = _json(request.output_dir / "manifest.json")
    summaries = [
        _json(
            request.output_dir
            / "shards"
            / f"shard-{index:05d}"
            / "task_summary.json"
        )
        for index in range(3)
    ]

    assert resumed_calls == ["scenario-002", "scenario-003", "scenario-004"]
    assert progress == [(1, 3, True), (2, 3, False), (3, 3, False)]
    assert result.reused_shard_count == 1
    assert replay.reused_shard_count == 3
    assert replay.manifest_digest == result.manifest_digest
    assert loaded.manifest_digest == result.manifest_digest
    assert manifest["task_count"] == 5
    assert manifest["accepted_group_count"] == 5
    assert manifest["rejected_task_count"] == 0
    assert manifest["sample_count"] == 30
    assert manifest["shard_count"] == 3
    assert [summary["task_ids"] for summary in summaries] == [
        ["scenario-000", "scenario-001"],
        ["scenario-002", "scenario-003"],
        ["scenario-004"],
    ]
    assert [summary["sample_count"] for summary in summaries] == [12, 12, 6]
    assert not stale.exists()
    assert not in_progress.exists()


def test_typed_ineligibility_persists_one_task_without_replacement(
    release_harness,
) -> None:
    source = _source(2, release_harness.grid)
    request = _request(release_harness, groups_per_shard=2)
    accepted_build = _build_one_factory(release_harness.template)
    calls: list[str] = []

    def reject_first(*args, **kwargs):
        accepted = args[1]
        calls.append(accepted.scenario_id)
        if accepted.source_index == 0:
            raise VerificationSourceIneligibleError(
                "infeasible_actions",
                "fixture rejected the fixed task",
            )
        return accepted_build(*args, **kwargs)

    result = publish_verification_release(
        request,
        source_loader=lambda value: source,
        build_one=reject_first,
    )
    manifest = _json(request.output_dir / "manifest.json")
    summary = _json(
        request.output_dir
        / "shards"
        / "shard-00000"
        / "task_summary.json"
    )
    loaded_shard = load_verification_shard(
        request.output_dir / "shards" / "shard-00000" / "data",
        grid=release_harness.grid,
        library=release_harness.library,
    )

    assert calls == ["scenario-000", "scenario-001"]
    assert result.task_count == 2
    assert result.accepted_group_count == 1
    assert result.rejected_task_count == 1
    assert result.sample_count == 6
    assert manifest["task_count"] == 2
    assert manifest["accepted_group_count"] == 1
    assert manifest["rejected_task_count"] == 1
    assert manifest["sample_count"] == 6
    assert manifest["rejection_counts"] == {"infeasible_actions": 1}
    assert manifest["action_counts"] == {
        action_id: 1 for action_id in CANONICAL_ACTION_IDS
    }
    assert summary["task_ids"] == ["scenario-000", "scenario-001"]
    assert summary["accepted_group_count"] == 1
    assert summary["rejected_task_count"] == 1
    assert summary["sample_count"] == 6
    assert summary["sampled_child_world_ids"] == [
        "sampled-child-scenario-001"
    ]
    assert summary["rejections"] == [
        {
            "task_id": "scenario-000",
            "mother_id": "mother-000",
            "reason": "infeasible_actions",
            "detail": "fixture rejected the fixed task",
        }
    ]
    assert tuple(
        record["action_id"] for record in summary["value_records"]
    ) == CANONICAL_ACTION_IDS
    assert len(loaded_shard.samples) == 6
    assert loaded_shard.action_counts == {
        action_id: 1 for action_id in CANONICAL_ACTION_IDS
    }


def test_release_persists_strict_revaluation_records(release_harness) -> None:
    source = _source(1, release_harness.grid)
    request = _request(release_harness, groups_per_shard=1)
    publish_verification_release(
        request,
        source_loader=lambda value: source,
        build_one=_build_one_factory(release_harness.template),
    )

    loaded = load_verification_release(request.output_dir)
    records = load_verification_revaluation_records(request.output_dir)
    expected = release_harness.template.value_results

    assert len(records) == loaded.sample_count == 6
    assert len({record.sample_id for record in records}) == 6
    assert {record.release_request_identity for record in records} == {
        loaded.request_identity
    }
    assert {record.split for record in records} == {"train"}
    assert {record.task_id for record in records} == {"scenario-000"}
    assert {record.mother_id for record in records} == {"mother-000"}
    assert len({record.ranking_group_id for record in records}) == 1
    assert tuple(record.action_id for record in records) == CANONICAL_ACTION_IDS
    for record in records:
        value = expected[record.action_id]
        assert record.realized_execute_loss == value.realized_execute_loss
        assert (
            record.unclipped_best_policy_loss
            == value.unclipped_best_policy_loss
        )
        assert record.action_cost == value.action_cost
        assert record.original_reject_cost == value.reject_cost


def test_revaluation_record_semantic_tampering_is_rejected(
    release_harness,
) -> None:
    source = _source(1, release_harness.grid)
    request = _request(release_harness, groups_per_shard=1)
    publish_verification_release(
        request,
        source_loader=lambda value: source,
        build_one=_build_one_factory(release_harness.template),
    )
    shard = request.output_dir / "shards" / "shard-00000"
    summary_path = shard / "task_summary.json"
    complete_path = shard / "COMPLETE.json"
    summary = _json(summary_path)
    summary["revaluation_records"][0]["action_cost"] = -1.0
    payload = (
        json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    summary_path.write_bytes(payload)
    complete = _json(complete_path)
    complete["task_summary_sha256"] = hashlib.sha256(payload).hexdigest()
    complete_path.write_text(
        json.dumps(
            complete,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="action_cost"):
        load_verification_revaluation_records(request.output_dir)


def test_single_and_fork_releases_match_digests_and_preserve_source_order(
    release_harness,
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork multiprocessing is unavailable")
    start_method = multiprocessing.get_start_method(allow_none=True)
    if start_method not in (None, "fork"):
        pytest.skip(f"global multiprocessing start method is {start_method!r}")

    source = _source(4, release_harness.grid)
    source.accepted = (
        source.accepted[2],
        source.accepted[0],
        source.accepted[3],
        source.accepted[1],
    )
    accepted_build = _build_one_factory(release_harness.template)

    def fork_inherited_build(*args, **kwargs):
        accepted = args[1]
        if accepted.source_index == 1:
            raise VerificationSourceIneligibleError(
                "scenario_current_static_overlap",
                f"rejected {accepted.scenario_id}",
            )
        return accepted_build(*args, **kwargs)

    single_request = _request(
        release_harness,
        groups_per_shard=4,
        workers=1,
    )
    parallel_request = replace(
        single_request,
        output_dir=release_harness.root
        / "outputs"
        / "verification-parallel",
        workers=2,
    )
    single = publish_verification_release(
        single_request,
        source_loader=lambda value: source,
        build_one=fork_inherited_build,
    )
    parallel = publish_verification_release(
        parallel_request,
        source_loader=lambda value: source,
        build_one=fork_inherited_build,
    )
    single_loaded = load_verification_release(single_request.output_dir)
    parallel_loaded = load_verification_release(parallel_request.output_dir)
    single_summary = _json(
        single_request.output_dir
        / "shards"
        / "shard-00000"
        / "task_summary.json"
    )
    parallel_summary = _json(
        parallel_request.output_dir
        / "shards"
        / "shard-00000"
        / "task_summary.json"
    )
    expected_order = [
        "scenario-002",
        "scenario-000",
        "scenario-003",
        "scenario-001",
    ]

    assert single.manifest_digest == parallel.manifest_digest
    assert single_loaded.request_identity == parallel_loaded.request_identity
    assert single_loaded.manifest == parallel_loaded.manifest
    assert single_summary == parallel_summary
    assert single_summary["task_ids"] == expected_order
    assert isinstance(single_summary["verification_semantic_digest"], str)
    assert single_summary["verification_semantic_digest"]


def test_parallel_unexpected_failure_does_not_wait_for_earlier_task(
    release_harness,
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork multiprocessing is unavailable")
    start_method = multiprocessing.get_start_method(allow_none=True)
    if start_method not in (None, "fork"):
        pytest.skip(f"global multiprocessing start method is {start_method!r}")

    source = _source(2, release_harness.grid)
    accepted_build = _build_one_factory(release_harness.template)
    context = multiprocessing.get_context("fork")
    first_started = context.Event()
    release_first = context.Event()

    def fail_later_index(*args, **kwargs):
        accepted = args[1]
        if accepted.source_index == 0:
            first_started.set()
            if not release_first.wait(timeout=8.0):
                raise RuntimeError("earlier worker gate timed out")
            return accepted_build(*args, **kwargs)
        if not first_started.wait(timeout=8.0):
            raise RuntimeError("earlier worker did not start")
        raise RuntimeError("later worker failed")

    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="later worker failed"):
            release_module._evaluate_boundary(
                source,
                source.accepted,
                library=release_harness.library,
                gt_config=object(),
                max_replan_candidates=4,
                workers=2,
                build_one=fail_later_index,
            )
    finally:
        release_first.set()
    assert time.monotonic() - started < 4.0


def test_loader_recomputes_request_identity_from_payload(
    release_harness,
) -> None:
    source = _source(1, release_harness.grid)
    request = _request(release_harness, groups_per_shard=1)
    publish_verification_release(
        request,
        source_loader=lambda value: source,
        build_one=_build_one_factory(release_harness.template),
    )
    request_path = request.output_dir / "request.json"
    request_document = _json(request_path)
    request_document["request"]["max_replan_candidates"] = 99
    request_path.write_bytes(release_module._json_file(request_document))
    checksums_path = request.output_dir / "checksums.json"
    checksums = _json(checksums_path)
    checksums["request.json"] = release_module._sha256_file(request_path)
    checksums_path.write_bytes(release_module._json_file(checksums))

    with pytest.raises(ValueError, match="identity differs"):
        load_verification_release(request.output_dir)


@pytest.mark.parametrize(
    "relation",
    ["inside", "ancestor", "cache", "partial"],
)
def test_release_output_cannot_overlap_read_only_inputs(
    release_harness,
    relation: str,
) -> None:
    request = _request(release_harness, groups_per_shard=1)
    if relation == "inside":
        request = replace(
            request,
            output_dir=request.source_root / "verification",
        )
    elif relation == "ancestor":
        request = replace(
            request,
            output_dir=request.source_root.parent,
        )
    elif relation == "cache":
        cache = release_harness.root / "cache"
        request = replace(
            request,
            source_cache_root=cache,
            output_dir=cache / "verification",
        )
    else:
        sop03 = release_harness.root / "sop03"
        request = replace(
            request,
            source_mode="partial_m6_reconstruction",
            sop03_root=sop03,
            long40_human_artifact=release_harness.root / "long40",
            base_state_start=0,
            max_base_states=1,
            base_config_path=release_harness.actions_path,
            generator_config_path=release_harness.gt_path,
            output_dir=sop03 / "verification",
        )

    with pytest.raises(ValueError, match="must not overlap"):
        release_module._request_payload(request)
