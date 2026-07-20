"""Behavioral tests for accepted SOP07 risk-sidecar replay adoption."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

import numpy as np
import pytest

from src.contracts import GridSpec
from src.datasets.shard_writer import load_risk_shard, write_risk_shard
from src.datasets.sidecar_writer import (
    load_risk_sidecar_shard,
    risk_sidecar_pair_completion_marker_path,
    write_risk_sidecar_pair_completion_marker,
    write_risk_sidecar_shard,
)
from src.generation.risk_sidecars import RiskLabelSidecar
from tests.fixtures.formal_risk_publication import (
    FormalRiskPublication,
    create_formal_risk_publication,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "04_backfill_risk_sidecars.py"
GENERATOR = ROOT / "scripts" / "04_generate_risk_dataset.py"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@pytest.fixture
def cli_module():
    assert SCRIPT.is_file(), "the accepted-sidecar backfill CLI does not exist"
    module_name = "_test_04_backfill_risk_sidecars"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


@dataclass(frozen=True)
class _Case:
    publication: FormalRiskPublication
    report_path: Path
    handoff_path: Path
    report: dict[str, object]
    handoff: dict[str, object]
    replay_root: Path
    sidecar_root: Path
    paired_config_path: Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload) + "\n", encoding="utf-8")


def _make_case(tmp_path: Path) -> _Case:
    publication = create_formal_risk_publication(
        tmp_path / "accepted",
        history_steps=8,
        future_steps=15,
    )
    digests = ("d" * 64, "e" * 64)
    trajectories = ("forward_v00_w02", "forward_v01_w01")
    report_rows: list[dict[str, object]] = []
    handoff_rows: list[dict[str, object]] = []
    for shard_index in range(2):
        relative_root = f"shard-{shard_index:05d}"
        loaded = load_risk_shard(
            publication.collection_root / relative_root,
            grid=publication.grid,
        )
        report_rows.append(
            {
                "manifest_digest": loaded.manifest_digest,
                "relative_root": relative_root,
                "sample_count": len(loaded.samples),
                "semantic_digest": loaded.semantic_digest,
                "shard_index": shard_index,
                "sop05_publication_digest": digests[shard_index],
                "trajectory_id": trajectories[shard_index],
            }
        )
        handoff_rows.append(
            {
                "event_count": 1,
                "publication_semantic_digest": digests[shard_index],
                "relative_root": relative_root,
                "shard_index": shard_index,
                "trajectory_id": trajectories[shard_index],
            }
        )
    report: dict[str, object] = {
        "event_count": 2,
        "generation_state": "complete",
        "producer_version": "sop07_risk_dataset_cli_v3",
        "sample_count": 12,
        "schema_version": "3.0.0",
        "shard_count": 2,
        "shards": report_rows,
        "split": "train",
    }
    handoff: dict[str, object] = {
        "artifact_role": "sop05_train_batch_complete_index",
        "batch_state": "complete",
        "common_contracts": {"input_lock": {"split": "train"}},
        "counts": {"events": 2, "shards": 2},
        "handoff_version": "sop05_batch_index_handoff_v1",
        "producer_version": "sop05_generation_run_v6",
        "schema_version": "3.0.0",
        "shards": handoff_rows,
        "split": "train",
    }
    report_path = publication.collection_root / "batch_generation_report.json"
    handoff_path = tmp_path / "sop05" / "batch_complete_handoff.json"
    _write_json(report_path, report)
    _write_json(handoff_path, handoff)
    paired_config_path = tmp_path / "paired.yaml"
    paired_config_path.write_text("fixture: true\n", encoding="utf-8")
    return _Case(
        publication=publication,
        report_path=report_path,
        handoff_path=handoff_path,
        report=report,
        handoff=handoff,
        replay_root=tmp_path / "replay",
        sidecar_root=tmp_path / "sidecars",
        paired_config_path=paired_config_path,
    )


def _request(module, case: _Case, *, task_index: int = 0):
    return module.BackfillRequest(
        batch_generation_report=case.report_path,
        sop05_batch_handoff=case.handoff_path,
        accepted_risk_root=case.publication.collection_root,
        sop03_root=case.publication.root / "sop03",
        sop04_root=case.publication.root / "sop04",
        sop04_handoff_digest="a" * 64,
        config_path=case.publication.base_config_path,
        paired_config_path=case.paired_config_path,
        seed=42,
        replay_risk_root=case.replay_root,
        sidecar_root=case.sidecar_root,
        task_index=task_index,
        checksum_workers=2,
        python_executable=Path(sys.executable),
        producer_script=GENERATOR,
    )


def _option(command: tuple[str, ...], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def _sidecars(sample_ids: tuple[str, ...], grid: GridSpec) -> tuple[RiskLabelSidecar, ...]:
    shape = (grid.future_steps, grid.height, grid.width)
    endpoint_times = (
        np.arange(1, grid.future_steps + 1, dtype=np.float32)
        * np.float32(0.2)
    )
    values: list[RiskLabelSidecar] = []
    for index, sample_id in enumerate(sample_ids):
        hidden = np.zeros(shape, dtype=np.uint8)
        robot = np.zeros(shape, dtype=np.uint8)
        hidden[:, index % grid.height, index % grid.width] = np.uint8(1)
        robot[:, (index + 1) % grid.height, index % grid.width] = np.uint8(1)
        values.append(
            RiskLabelSidecar(
                sample_id=sample_id,
                hidden_risk_occupancy=hidden,
                robot_future_footprints=robot,
                future_endpoint_times_s=endpoint_times,
            )
        )
    return tuple(values)


def _real_io_runner(
    case: _Case,
    *,
    mutate_samples: Callable[[tuple[object, ...]], tuple[object, ...]] | None = None,
    reorder_manifest: bool = False,
    omit_marker: bool = False,
    wrong_marker_basename: bool = False,
    wrong_marker_index: bool = False,
    tamper_sidecar: bool = False,
):
    calls: list[tuple[str, ...]] = []

    def runner(command):
        normalized = tuple(str(value) for value in command)
        calls.append(normalized)
        assert Path(normalized[0]) == Path(sys.executable)
        assert Path(normalized[1]) == GENERATOR
        shard_index = int(_option(normalized, "--shard-index"))
        relative_root = f"shard-{shard_index:05d}"
        accepted_root = case.publication.collection_root / relative_root
        replay_root = Path(_option(normalized, "--output-dir"))
        sidecar_root = Path(_option(normalized, "--sidecar-output-dir"))
        assert replay_root == case.replay_root / relative_root
        assert sidecar_root == case.sidecar_root / relative_root
        assert Path(_option(normalized, "--sop05-root")) == (
            case.handoff_path.parent / relative_root
        )
        assert _option(normalized, "--sop05-publication-digest") in {
            "d" * 64,
            "e" * 64,
        }

        accepted = load_risk_shard(accepted_root, grid=case.publication.grid)
        samples = tuple(accepted.samples)
        if mutate_samples is not None:
            samples = mutate_samples(samples)
        write_risk_shard(
            samples,
            replay_root,
            grid=case.publication.grid,
            shard_index=shard_index,
            expected_sample_count=len(samples),
        )
        replay = load_risk_shard(replay_root, grid=case.publication.grid)
        sample_ids = tuple(sample.sample_id for sample in replay.samples)
        write_risk_sidecar_shard(
            _sidecars(sample_ids, case.publication.grid),
            sidecar_root,
            grid=case.publication.grid,
            split="train",
            shard_index=shard_index,
            source_risk_shard_semantic_digest=replay.semantic_digest,
        )
        sidecar = load_risk_sidecar_shard(
            sidecar_root,
            grid=case.publication.grid,
            expected_sample_ids=sample_ids,
            expected_source_risk_shard_semantic_digest=replay.semantic_digest,
        )
        marker_path = risk_sidecar_pair_completion_marker_path(sidecar_root)
        if not omit_marker:
            marker_risk_root = (
                replay_root.with_name("wrong-risk-basename")
                if wrong_marker_basename
                else replay_root
            )
            write_risk_sidecar_pair_completion_marker(
                marker_path,
                risk_root=marker_risk_root,
                sidecar_root=sidecar_root,
                split="train",
                shard_index=shard_index + 1 if wrong_marker_index else shard_index,
                sample_ids=sample_ids,
                risk_shard_semantic_digest=replay.semantic_digest,
                sidecar_shard_semantic_digest=sidecar.semantic_digest,
            )
        if reorder_manifest:
            manifest_path = replay_root / "metadata.jsonl"
            rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
            rows.reverse()
            for row_index, row in enumerate(rows):
                row["row_index"] = row_index
            manifest_path.write_text(
                "".join(_canonical_json(row) + "\n" for row in rows),
                encoding="utf-8",
            )
        if tamper_sidecar:
            summary_path = sidecar_root / "summary.json"
            summary_path.write_bytes(summary_path.read_bytes() + b"\n")
        child_report = {
            "manifest_digest": replay.manifest_digest,
            "output_dir": str(replay_root),
            "pair_completion_marker_digest": (
                None
                if omit_marker
                else json.loads(marker_path.read_text())["marker_digest_sha256"]
            ),
            "pair_completion_marker_path": str(marker_path),
            "producer_version": "sop07_risk_dataset_cli_v4",
            "publication_status": "complete",
            "risk_shard_semantic_digest": replay.semantic_digest,
            "sample_count": len(samples),
            "semantic_digest": replay.semantic_digest,
            "shard_index": shard_index,
            "sidecar_output_dir": str(sidecar_root),
            "sidecar_shard_semantic_digest": sidecar.semantic_digest,
            "split": "train",
        }
        return subprocess.CompletedProcess(
            normalized,
            0,
            stdout=_canonical_json(child_report) + "\n",
            stderr="",
        )

    runner.calls = calls
    return runner


def _changed_semantics(samples: tuple[object, ...]) -> tuple[object, ...]:
    changed = list(samples)
    first = changed[0]
    bev_history = first.bev_history.copy()
    bev_history[0, 0, 0, 0] += np.float32(0.01)
    changed[0] = replace(first, bev_history=bev_history)
    return tuple(changed)


def test_report_and_handoff_parser_selects_one_exact_task(cli_module, tmp_path: Path) -> None:
    case = _make_case(tmp_path)

    task = cli_module.load_backfill_task(
        case.report_path,
        case.handoff_path,
        task_index=1,
    )

    assert task.split == "train"
    assert task.shard_index == 1
    assert task.relative_root == "shard-00001"
    assert task.sample_count == 6
    assert task.event_count == 1
    assert task.manifest_digest == case.report["shards"][1]["manifest_digest"]
    assert task.semantic_digest == case.report["shards"][1]["semantic_digest"]
    assert task.sop05_relative_root == "shard-00001"
    assert task.sop05_publication_digest == "e" * 64
    assert task.trajectory_id == "forward_v01_w01"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("report_reorder", "contiguous and ordered"),
        ("handoff_reorder", "contiguous and ordered"),
        ("digest", "publication digest"),
        ("trajectory", "trajectory_id"),
        ("handoff_version", "identity"),
        ("sample_total", "sample count"),
        ("event_total", "event count"),
    ),
)
def test_report_and_handoff_parser_rejects_cross_contract_mismatch(
    cli_module,
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    case = _make_case(tmp_path)
    report = deepcopy(case.report)
    handoff = deepcopy(case.handoff)
    if mutation == "report_reorder":
        report["shards"].reverse()
    elif mutation == "handoff_reorder":
        handoff["shards"].reverse()
    elif mutation == "digest":
        handoff["shards"][0]["publication_semantic_digest"] = "f" * 64
    elif mutation == "trajectory":
        handoff["shards"][0]["trajectory_id"] = "forward_v03_w00"
    elif mutation == "handoff_version":
        handoff["handoff_version"] = "sop05_batch_index_handoff_v0"
    elif mutation == "sample_total":
        report["sample_count"] = 13
    elif mutation == "event_total":
        handoff["counts"]["events"] = 3
    _write_json(case.report_path, report)
    _write_json(case.handoff_path, handoff)

    with pytest.raises(ValueError, match=message):
        cli_module.load_backfill_task(
            case.report_path,
            case.handoff_path,
            task_index=0,
        )


@pytest.mark.parametrize("authority", ("report", "handoff"))
def test_authority_json_rejects_duplicate_keys_recursively(
    cli_module,
    tmp_path: Path,
    authority: str,
) -> None:
    case = _make_case(tmp_path)
    if authority == "report":
        payload = case.report_path.read_text(encoding="utf-8").replace(
            '"split":"train"',
            '"split":"train","split":"train"',
            1,
        )
        case.report_path.write_text(payload, encoding="utf-8")
    else:
        payload = case.handoff_path.read_text(encoding="utf-8").replace(
            '"input_lock":{"split":"train"}',
            '"input_lock":{"split":"train","split":"train"}',
            1,
        )
        case.handoff_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        cli_module.load_backfill_task(
            case.report_path,
            case.handoff_path,
            task_index=0,
        )


@pytest.mark.parametrize(
    ("environment", "message"),
    (
        ({}, "SLURM_ARRAY_TASK_ID"),
        ({"SLURM_ARRAY_TASK_ID": "not-an-int"}, "integer"),
        ({"SLURM_ARRAY_TASK_ID": "-1"}, "out of range"),
        ({"SLURM_ARRAY_TASK_ID": "2"}, "out of range"),
    ),
)
def test_slurm_array_task_id_is_required_and_bounded(
    cli_module,
    environment: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        cli_module.resolve_task_index(None, environ=environment, task_count=2)
    assert cli_module.resolve_task_index(
        None,
        environ={"SLURM_ARRAY_TASK_ID": "1"},
        task_count=2,
    ) == 1


def test_parser_and_command_accept_the_distinct_heldout_handoff_identity(
    cli_module,
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    report = deepcopy(case.report)
    handoff = deepcopy(case.handoff)
    report["split"] = "calibration"
    handoff["split"] = "calibration"
    handoff["common_contracts"]["input_lock"]["split"] = "calibration"
    handoff["artifact_role"] = "sop05_heldout_batch_complete_index"
    handoff["handoff_version"] = "sop05_heldout_batch_complete_handoff_v1"
    handoff.pop("producer_version")
    _write_json(case.report_path, report)
    _write_json(case.handoff_path, handoff)

    task = cli_module.load_backfill_task(
        case.report_path,
        case.handoff_path,
        task_index=0,
    )

    assert task.split == "calibration"
    command = cli_module._producer_command(
        _request(cli_module, case),
        task=task,
        replay_path=case.replay_root / task.relative_root,
        sidecar_path=case.sidecar_root / task.relative_root,
    )
    assert _option(command, "--split") == "calibration"
    assert _option(command, "--sop05-publication-digest") == "d" * 64
    assert "--sop05-publication-semantic-digest" not in command


def test_backfill_adopts_real_replay_sidecar_and_marker_and_resumes(
    cli_module,
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    request = _request(cli_module, case)
    runner = _real_io_runner(case)

    report = cli_module.run_backfill(request, runner=runner)
    resumed = cli_module.run_backfill(
        request,
        runner=lambda command: pytest.fail(f"resume invoked producer: {command}"),
    )

    assert report["status"] == "complete"
    assert resumed["status"] == "already_complete"
    assert len(runner.calls) == 1
    assert report["accepted_manifest_digest"] == report["replay_manifest_digest"]
    assert report["accepted_semantic_digest"] == report["replay_semantic_digest"]
    assert report["sample_count"] == 6
    assert report["shard_index"] == 0
    accepted = load_risk_shard(
        case.publication.collection_root / "shard-00000",
        grid=case.publication.grid,
    )
    replay = load_risk_shard(
        case.replay_root / "shard-00000",
        grid=case.publication.grid,
    )
    assert tuple(sample.sample_id for sample in accepted.samples) == tuple(
        sample.sample_id for sample in replay.samples
    )
    assert accepted.manifest_digest == replay.manifest_digest
    assert accepted.semantic_digest == replay.semantic_digest


def test_partial_existing_outputs_are_rejected_without_overwrite(
    cli_module,
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    accepted = load_risk_shard(
        case.publication.collection_root / "shard-00000",
        grid=case.publication.grid,
    )
    replay_path = case.replay_root / "shard-00000"
    write_risk_shard(
        accepted.samples,
        replay_path,
        grid=case.publication.grid,
        shard_index=0,
        expected_sample_count=6,
    )

    with pytest.raises(ValueError, match="partial"):
        cli_module.run_backfill(
            _request(cli_module, case),
            runner=lambda command: pytest.fail(f"partial resume invoked: {command}"),
        )

    assert replay_path.is_dir()


def test_replay_semantic_mismatch_is_rejected_after_real_formal_io(
    cli_module,
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)

    with pytest.raises(ValueError, match="semantic digest"):
        cli_module.run_backfill(
            _request(cli_module, case),
            runner=_real_io_runner(case, mutate_samples=_changed_semantics),
        )


def test_reordered_replay_manifest_is_rejected_by_formal_loader(
    cli_module,
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)

    with pytest.raises(ValueError, match="order"):
        cli_module.run_backfill(
            _request(cli_module, case),
            runner=_real_io_runner(case, reorder_manifest=True),
        )


@pytest.mark.parametrize(
    ("runner_kwargs", "message"),
    (
        ({"omit_marker": True}, "marker"),
        ({"wrong_marker_basename": True}, "marker evidence mismatch"),
        ({"wrong_marker_index": True}, "marker evidence mismatch"),
        ({"tamper_sidecar": True}, "sidecar summary"),
    ),
)
def test_missing_wrong_basename_or_tampered_pair_is_rejected(
    cli_module,
    tmp_path: Path,
    runner_kwargs: dict[str, bool],
    message: str,
) -> None:
    case = _make_case(tmp_path)

    with pytest.raises(ValueError, match=message):
        cli_module.run_backfill(
            _request(cli_module, case),
            runner=_real_io_runner(case, **runner_kwargs),
        )


def test_cli_emits_exactly_one_canonical_json_line(cli_module, tmp_path: Path, capsys) -> None:
    case = _make_case(tmp_path)
    runner = _real_io_runner(case)
    arguments = [
        "--batch-generation-report",
        str(case.report_path),
        "--sop05-batch-handoff",
        str(case.handoff_path),
        "--accepted-risk-root",
        str(case.publication.collection_root),
        "--sop03-root",
        str(case.publication.root / "sop03"),
        "--sop04-root",
        str(case.publication.root / "sop04"),
        "--sop04-handoff-digest",
        "a" * 64,
        "--config",
        str(case.publication.base_config_path),
        "--paired-config",
        str(case.paired_config_path),
        "--seed",
        "42",
        "--replay-risk-root",
        str(case.replay_root),
        "--sidecar-root",
        str(case.sidecar_root),
        "--checksum-workers",
        "2",
    ]

    exit_code = cli_module.main(
        arguments,
        environ={"SLURM_ARRAY_TASK_ID": "0"},
        runner=runner,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.endswith("\n")
    assert captured.out.count("\n") == 1
    assert captured.out == _canonical_json(json.loads(captured.out)) + "\n"


def test_cli_errors_use_one_stderr_line_and_exit_two(cli_module, tmp_path: Path, capsys) -> None:
    case = _make_case(tmp_path)

    exit_code = cli_module.main(
        [
            "--batch-generation-report",
            str(case.report_path),
            "--sop05-batch-handoff",
            str(case.handoff_path),
        ],
        environ={"SLURM_ARRAY_TASK_ID": "0"},
        runner=lambda command: pytest.fail(f"invalid CLI invoked: {command}"),
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert captured.err.count("\n") == 1


def test_cli_collapses_multiline_producer_errors_to_one_stderr_line(
    cli_module,
    tmp_path: Path,
    capsys,
) -> None:
    case = _make_case(tmp_path)
    arguments = [
        "--batch-generation-report",
        str(case.report_path),
        "--sop05-batch-handoff",
        str(case.handoff_path),
        "--accepted-risk-root",
        str(case.publication.collection_root),
        "--sop03-root",
        str(case.publication.root / "sop03"),
        "--sop04-root",
        str(case.publication.root / "sop04"),
        "--sop04-handoff-digest",
        "a" * 64,
        "--config",
        str(case.publication.base_config_path),
        "--paired-config",
        str(case.paired_config_path),
        "--seed",
        "42",
        "--replay-risk-root",
        str(case.replay_root),
        "--sidecar-root",
        str(case.sidecar_root),
    ]

    exit_code = cli_module.main(
        arguments,
        environ={"SLURM_ARRAY_TASK_ID": "0"},
        runner=lambda command: subprocess.CompletedProcess(
            command,
            2,
            stdout="",
            stderr="first failure line\nsecond failure line\n",
        ),
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == (
        "error: SOP07 replay producer failed: first failure line "
        "second failure line\n"
    )
