#!/usr/bin/env python
"""Replay accepted risk shards and publish one evaluation-record collection."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.contracts import SCHEMA_VERSION  # noqa: E402
from src.datasets.risk_dataset_seal import (  # noqa: E402
    LoadedRiskDataset,
    load_risk_dataset_seal,
)
from src.datasets.risk_evaluation_store import (  # noqa: E402
    EvaluationCollectionError,
    load_risk_evaluation_collection,
    load_risk_evaluation_replay_shard,
    publish_risk_evaluation_collection,
)
from src.datasets.shard_writer import load_risk_shard  # noqa: E402


EVALUATION_REPLAY_PRODUCER_VERSION = "sop08_evaluation_replay_v1"


class EvaluationReplayError(ValueError):
    """Raised when accepted and replayed shard identities diverge."""


@dataclass(frozen=True)
class EvaluationReplayRequest:
    dataset_seal_root: Path
    risk_collection_root: Path
    batch_generation_report: Path
    sop05_batch_handoff: Path
    sop03_root: Path
    sop04_root: Path
    sop04_handoff_digest: str
    sop05_root: Path
    config_path: Path
    paired_config_path: Path
    seed: int
    split: str
    replay_risk_root: Path
    replay_sidecar_root: Path
    replay_evaluation_root: Path
    output_dir: Path
    shard_start: int
    shard_end: int
    checksum_workers: int
    python_executable: Path
    producer_script: Path


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluationReplayError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_request(request: EvaluationReplayRequest) -> None:
    if not isinstance(request, EvaluationReplayRequest):
        raise TypeError("request must be an EvaluationReplayRequest")
    path_fields = (
        "dataset_seal_root",
        "risk_collection_root",
        "batch_generation_report",
        "sop05_batch_handoff",
        "sop03_root",
        "sop04_root",
        "sop05_root",
        "config_path",
        "paired_config_path",
        "replay_risk_root",
        "replay_sidecar_root",
        "replay_evaluation_root",
        "output_dir",
        "python_executable",
        "producer_script",
    )
    for field in path_fields:
        if not isinstance(getattr(request, field), Path):
            raise TypeError(f"{field} must be a Path")
    _sha256(request.sop04_handoff_digest, label="SOP04 handoff digest")
    for field in ("seed", "shard_start"):
        value = getattr(request, field)
        if type(value) is not int or value < 0:
            raise EvaluationReplayError(f"{field} must be a nonnegative integer")
    for field in ("shard_end", "checksum_workers"):
        value = getattr(request, field)
        if type(value) is not int or value < 1:
            raise EvaluationReplayError(f"{field} must be a positive integer")
    if request.shard_start >= request.shard_end:
        raise EvaluationReplayError("shard range must be non-empty")
    if not isinstance(request.split, str) or not request.split:
        raise EvaluationReplayError("split must be a non-empty string")
    if os.path.lexists(request.output_dir):
        raise FileExistsError(
            f"refusing to overwrite immutable evaluation collection: {request.output_dir}"
        )
    immutable_risk = request.risk_collection_root.resolve(strict=False)
    mutable_outputs = tuple(
        path.resolve(strict=False)
        for path in (
            request.replay_risk_root,
            request.replay_sidecar_root,
            request.replay_evaluation_root,
            request.output_dir,
        )
    )
    for output in mutable_outputs:
        if (
            output == immutable_risk
            or output in immutable_risk.parents
            or immutable_risk in output.parents
        ):
            raise EvaluationReplayError(
                "replay/evaluation outputs must not overlap the accepted risk collection"
            )
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(mutable_outputs)
        for right in mutable_outputs[index + 1 :]
    ):
        raise EvaluationReplayError("replay/evaluation output roots must not overlap")


def _load_tasks(request: EvaluationReplayRequest) -> tuple[object, ...]:
    module = importlib.import_module("scripts.04_backfill_risk_sidecars")
    try:
        tasks = module._load_backfill_tasks(  # noqa: SLF001
            request.batch_generation_report,
            request.sop05_batch_handoff,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise EvaluationReplayError(f"failed to load replay authorities: {exc}") from exc
    if not isinstance(tasks, tuple) or not tasks:
        raise EvaluationReplayError("replay authorities contain no shard tasks")
    return tasks


def _producer_command(
    request: EvaluationReplayRequest,
    *,
    task: object,
) -> tuple[str, ...]:
    relative_root = str(task.relative_root)
    return (
        str(request.python_executable),
        str(request.producer_script),
        "--sop03-root",
        str(request.sop03_root),
        "--sop04-root",
        str(request.sop04_root),
        "--sop04-handoff-digest",
        request.sop04_handoff_digest,
        "--sop05-root",
        str(request.sop05_root / str(task.sop05_relative_root)),
        "--sop05-publication-digest",
        str(task.sop05_publication_digest),
        "--split",
        request.split,
        "--config",
        str(request.config_path),
        "--paired-config",
        str(request.paired_config_path),
        "--seed",
        str(request.seed),
        "--output-dir",
        str(request.replay_risk_root / relative_root),
        "--sidecar-output-dir",
        str(request.replay_sidecar_root / relative_root),
        "--evaluation-record-output-dir",
        str(request.replay_evaluation_root / relative_root),
        "--shard-index",
        str(task.shard_index),
        "--expected-event-count",
        str(task.event_count),
        "--expected-sample-count",
        str(task.sample_count),
        "--checksum-workers",
        str(request.checksum_workers),
    )


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(command),
        check=False,
        capture_output=True,
        text=True,
    )


def _run_producer(
    request: EvaluationReplayRequest,
    *,
    task: object,
    runner: Callable[[Sequence[str]], object],
) -> None:
    command = _producer_command(request, task=task)
    result = runner(command)
    returncode = getattr(result, "returncode", None)
    stderr = getattr(result, "stderr", None)
    if type(returncode) is not int or not isinstance(stderr, str):
        raise EvaluationReplayError("replay producer returned an invalid result")
    if returncode != 0:
        detail = " ".join(stderr.split()) or f"exit {returncode}"
        raise EvaluationReplayError(f"evaluation replay producer failed: {detail}")


def _verify_task(
    *,
    dataset: LoadedRiskDataset,
    task: object,
    replay_risk_root: Path,
    replay_evaluation_root: Path,
) -> tuple[dict[str, object], ...]:
    descriptor = dataset.shards[int(task.shard_index)]
    if (
        task.split != dataset.split
        or task.relative_root != descriptor.relative_root
        or task.sample_count != descriptor.sample_count
        or task.semantic_digest != descriptor.semantic_digest
    ):
        raise EvaluationReplayError(
            f"replay authority differs from sealed shard {task.shard_index}"
        )
    accepted = load_risk_shard(
        dataset.collection_root / descriptor.relative_root,
        grid=dataset.grid,
    )
    replay_path = replay_risk_root / descriptor.relative_root
    evaluation_path = replay_evaluation_root / descriptor.relative_root
    try:
        replay = load_risk_shard(replay_path, grid=dataset.grid)
    except (OSError, TypeError, ValueError) as exc:
        raise EvaluationReplayError(
            f"failed to load replay risk shard {task.shard_index}: {exc}"
        ) from exc
    accepted_ids = tuple(sample.sample_id for sample in accepted.samples)
    replay_ids = tuple(sample.sample_id for sample in replay.samples)
    if (
        replay.summary.get("split") != dataset.split
        or replay.summary.get("shard_index") != task.shard_index
        or replay.manifest_digest != accepted.manifest_digest
        or replay.semantic_digest != accepted.semantic_digest
        or replay_ids != accepted_ids
    ):
        raise EvaluationReplayError(
            f"accepted/replay risk semantic or sample identity mismatch for shard {task.shard_index}"
        )
    try:
        evaluation = load_risk_evaluation_replay_shard(
            evaluation_path,
            risk_shard=accepted,
        )
    except (OSError, TypeError, ValueError, EvaluationCollectionError) as exc:
        raise EvaluationReplayError(
            f"evaluation replay shard {task.shard_index} failed authentication: {exc}"
        ) from exc
    if evaluation.sample_ids != accepted_ids:
        raise EvaluationReplayError(
            f"evaluation replay sample order mismatch for shard {task.shard_index}"
        )
    return evaluation.records


def run_evaluation_replay(
    request: EvaluationReplayRequest,
    *,
    runner: Callable[[Sequence[str]], object] = _default_runner,
) -> dict[str, object]:
    """Replay selected missing shards, then publish only after full verification."""

    _validate_request(request)
    try:
        dataset = load_risk_dataset_seal(
            request.dataset_seal_root,
            collection_root=request.risk_collection_root,
            expected_split=request.split,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise EvaluationReplayError(f"failed to authenticate risk dataset: {exc}") from exc
    tasks = _load_tasks(request)
    if len(tasks) != len(dataset.shards) or request.shard_end > len(tasks):
        raise EvaluationReplayError("replay shard range/authority count mismatch")
    records_by_shard: dict[int, tuple[dict[str, object], ...]] = {}
    for task in tasks:
        index = int(task.shard_index)
        replay_path = request.replay_risk_root / str(task.relative_root)
        evaluation_path = request.replay_evaluation_root / str(task.relative_root)
        presence = (os.path.lexists(replay_path), os.path.lexists(evaluation_path))
        if not all(presence):
            if any(presence):
                raise EvaluationReplayError(
                    f"partial replay outputs are forbidden for shard {index}"
                )
            if request.shard_start <= index < request.shard_end:
                _run_producer(request, task=task, runner=runner)
            else:
                raise EvaluationReplayError(
                    f"replay outputs are missing outside requested shard range: {index}"
                )
        records_by_shard[index] = _verify_task(
            dataset=dataset,
            task=task,
            replay_risk_root=request.replay_risk_root,
            replay_evaluation_root=request.replay_evaluation_root,
        )
    publish_risk_evaluation_collection(
        request.output_dir,
        dataset=dataset,
        records_by_shard=records_by_shard,
    )
    loaded = load_risk_evaluation_collection(
        request.output_dir,
        dataset=dataset,
        expected_manifest_digest=dataset.risk_dataset_manifest_digest,
    )
    reloaded_dataset = load_risk_dataset_seal(
        request.dataset_seal_root,
        collection_root=request.risk_collection_root,
        expected_split=request.split,
        expected_manifest_digest=dataset.risk_dataset_manifest_digest,
    )
    if reloaded_dataset != dataset:
        raise EvaluationReplayError("accepted risk dataset changed during replay")
    report = {
        "schema_version": SCHEMA_VERSION,
        "producer_version": EVALUATION_REPLAY_PRODUCER_VERSION,
        "split": dataset.split,
        "shard_count": len(dataset.shards),
        "sample_count": loaded.sample_count,
        "risk_dataset_manifest_digest": dataset.risk_dataset_manifest_digest,
        "evaluation_collection_semantic_digest_sha256": (
            loaded.collection_semantic_digest_sha256
        ),
        "output_dir": str(request.output_dir),
    }
    _canonical_json(report)
    return report


def _nonnegative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def _positive_int(text: str) -> int:
    value = _nonnegative_int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise EvaluationReplayError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-seal-root", type=Path, required=True)
    parser.add_argument("--risk-collection-root", type=Path, required=True)
    parser.add_argument("--batch-generation-report", type=Path, required=True)
    parser.add_argument("--sop05-batch-handoff", type=Path, required=True)
    parser.add_argument("--sop03-root", type=Path, required=True)
    parser.add_argument("--sop04-root", type=Path, required=True)
    parser.add_argument("--sop04-handoff-digest", required=True)
    parser.add_argument("--sop05-root", type=Path, required=True)
    parser.add_argument("--config", dest="config_path", type=Path, required=True)
    parser.add_argument(
        "--paired-config",
        dest="paired_config_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=_nonnegative_int, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--replay-risk-root", type=Path, required=True)
    parser.add_argument("--replay-sidecar-root", type=Path, required=True)
    parser.add_argument("--replay-evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-start", type=_nonnegative_int, default=0)
    parser.add_argument("--shard-end", type=_positive_int, required=True)
    parser.add_argument("--checksum-workers", type=_positive_int, default=8)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--producer-script",
        type=Path,
        default=_ROOT / "scripts" / "04_generate_risk_dataset.py",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[Sequence[str]], object] = _default_runner,
) -> int:
    try:
        args = _parser().parse_args(argv)
        request = EvaluationReplayRequest(
            dataset_seal_root=args.dataset_seal_root,
            risk_collection_root=args.risk_collection_root,
            batch_generation_report=args.batch_generation_report,
            sop05_batch_handoff=args.sop05_batch_handoff,
            sop03_root=args.sop03_root,
            sop04_root=args.sop04_root,
            sop04_handoff_digest=args.sop04_handoff_digest,
            sop05_root=args.sop05_root,
            config_path=args.config_path,
            paired_config_path=args.paired_config_path,
            seed=args.seed,
            split=args.split,
            replay_risk_root=args.replay_risk_root,
            replay_sidecar_root=args.replay_sidecar_root,
            replay_evaluation_root=args.replay_evaluation_root,
            output_dir=args.output_dir,
            shard_start=args.shard_start,
            shard_end=args.shard_end,
            checksum_workers=args.checksum_workers,
            python_executable=args.python_executable,
            producer_script=args.producer_script,
        )
        report = run_evaluation_replay(request, runner=runner)
    except (
        EvaluationReplayError,
        EvaluationCollectionError,
        FileExistsError,
        OSError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        detail = " ".join(str(exc).split()) or type(exc).__name__
        print(f"error: {detail}", file=sys.stderr)
        return 2
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
