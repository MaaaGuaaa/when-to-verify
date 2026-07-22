"""Authenticated evaluation-record replay and publication CLI tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from src.datasets.risk_dataset_seal import (
    load_risk_dataset_seal,
    publish_risk_dataset_seal,
)
from src.datasets.risk_evaluation_store import (
    load_risk_evaluation_collection,
    publish_risk_evaluation_replay_shard,
)
from src.datasets.shard_writer import load_risk_shard
from tests.test_04_backfill_risk_sidecars import _make_case
from tests.test_risk_evaluation_store import _record


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "04_publish_risk_evaluation_records.py"


@pytest.fixture
def cli_module():
    spec = importlib.util.spec_from_file_location("evaluation_record_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path: Path):
    case = _make_case(tmp_path)
    seal_root = publish_risk_dataset_seal(
        tmp_path / "seal",
        collection_root=case.publication.collection_root,
        base_config_path=case.publication.base_config_path,
        split_provenance_path=case.publication.split_provenance_path,
        expected_split="train",
        expected_collection_handoff_sha256=case.publication.handoff_sha256,
    )
    dataset = load_risk_dataset_seal(
        seal_root,
        collection_root=case.publication.collection_root,
        expected_split="train",
    )
    replay_risk_root = tmp_path / "replay-risk"
    replay_sidecar_root = tmp_path / "replay-sidecar"
    replay_evaluation_root = tmp_path / "replay-evaluation"
    for descriptor in dataset.shards:
        accepted_root = dataset.collection_root / descriptor.relative_root
        replay_root = replay_risk_root / descriptor.relative_root
        replay_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(accepted_root, replay_root)
        replay = load_risk_shard(replay_root, grid=dataset.grid)
        publish_risk_evaluation_replay_shard(
            replay_evaluation_root / descriptor.relative_root,
            risk_shard=replay,
            records=tuple(_record(sample) for sample in replay.samples),
        )
    return (
        case,
        dataset,
        replay_risk_root,
        replay_sidecar_root,
        replay_evaluation_root,
    )


def _request(module, tmp_path: Path):
    case, dataset, replay_risk, replay_sidecar, replay_evaluation = _case(tmp_path)
    return module.EvaluationReplayRequest(
        dataset_seal_root=dataset.seal_root,
        risk_collection_root=dataset.collection_root,
        batch_generation_report=case.report_path,
        sop05_batch_handoff=case.handoff_path,
        sop05_root=case.handoff_path.parent,
        sop03_root=tmp_path / "sop03",
        sop04_root=tmp_path / "sop04",
        sop04_handoff_digest="a" * 64,
        config_path=case.publication.base_config_path,
        paired_config_path=case.paired_config_path,
        seed=42,
        split="train",
        replay_risk_root=replay_risk,
        replay_sidecar_root=replay_sidecar,
        replay_evaluation_root=replay_evaluation,
        output_dir=tmp_path / "evaluation-collection",
        shard_start=0,
        shard_end=len(dataset.shards),
        checksum_workers=2,
        python_executable=Path(sys.executable),
        producer_script=ROOT / "scripts" / "04_generate_risk_dataset.py",
    ), dataset


def test_replay_cli_publishes_records_without_changing_risk_collection(
    cli_module,
    tmp_path: Path,
) -> None:
    request, dataset = _request(cli_module, tmp_path)
    before = {
        path.relative_to(dataset.collection_root).as_posix(): _sha256(path)
        for path in dataset.collection_root.rglob("*")
        if path.is_file()
    }

    report = cli_module.run_evaluation_replay(
        request,
        runner=lambda command: pytest.fail(f"complete replay reran producer: {command}"),
    )

    loaded = load_risk_evaluation_collection(
        request.output_dir,
        dataset=dataset,
    )
    after = {
        path.relative_to(dataset.collection_root).as_posix(): _sha256(path)
        for path in dataset.collection_root.rglob("*")
        if path.is_file()
    }
    assert before == after
    expected_ids = tuple(
        sample.sample_id
        for descriptor in dataset.shards
        for sample in load_risk_shard(
            dataset.collection_root / descriptor.relative_root,
            grid=dataset.grid,
        ).samples
    )
    assert loaded.sample_ids == expected_ids
    assert report["sample_count"] == dataset.sample_count
    assert report["risk_dataset_manifest_digest"] == (
        dataset.risk_dataset_manifest_digest
    )
    assert report["evaluation_collection_semantic_digest_sha256"] == (
        loaded.collection_semantic_digest_sha256
    )


def test_replay_publisher_refuses_risk_semantic_digest_mismatch(
    cli_module,
    tmp_path: Path,
) -> None:
    request, dataset = _request(cli_module, tmp_path)
    first = dataset.shards[0].relative_root
    shutil.rmtree(request.replay_risk_root / first)
    shutil.copytree(
        dataset.collection_root / dataset.shards[1].relative_root,
        request.replay_risk_root / first,
    )

    with pytest.raises(cli_module.EvaluationReplayError, match="semantic|sample|shard"):
        cli_module.run_evaluation_replay(
            request,
            runner=lambda command: pytest.fail(f"mismatch reran producer: {command}"),
        )
    assert not request.output_dir.exists()


def test_replay_cli_errors_are_one_line_and_exit_two(
    cli_module,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_module.main(
        [
            "--dataset-seal-root",
            str(tmp_path / "missing-seal"),
            "--risk-collection-root",
            str(tmp_path / "missing-risk"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert captured.err.count("\n") == 1
