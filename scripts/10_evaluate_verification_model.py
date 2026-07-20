#!/usr/bin/env python3
"""Evaluate one frozen SOP14 checkpoint on an immutable held-out collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import SCHEMA_VERSION, build_grid_spec
from src.datasets.verification_collection import (
    validate_verification_collection_handoff,
    verification_input_digests,
)
from src.datasets.verification_dataloader import (
    load_verification_collection,
    load_verification_shard,
)
from src.evaluation.verification_baselines import evaluate_verification_baselines
from src.models.verification_model import load_verify_model_config
from src.models.verification_training import (
    evaluate_verification_samples,
    load_verification_training_checkpoint,
)
from src.planning.verification_actions import load_verification_actions
from src.utils.config import load_config


EVALUATION_CLI_VERSION = "verification_evaluation_cli_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=("calibration", "val", "test"), required=True
    )
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--collection-handoff", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--actions-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--expected-code-version", required=True)
    return parser


def _strict_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a real file")

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"provenance input must be a real file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot hash provenance input: {path}") from exc
    return digest.hexdigest()


def _implementation_digest(config_paths: Sequence[Path]) -> str:
    relative_files = (
        Path("scripts/10_evaluate_verification_model.py"),
        Path("src/datasets/verification_collection.py"),
        Path("src/evaluation/verification_baselines.py"),
        Path("src/evaluation/verification_metrics.py"),
        Path("src/models/verification_model.py"),
        Path("src/models/verification_training.py"),
    )
    digest = hashlib.sha256()
    digest.update(b"verification-heldout-evaluation-implementation-v1\0")
    for path in (*[ROOT / value for value in relative_files], *config_paths):
        payload = path.read_bytes()
        label = str(path).encode("utf-8")
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable output: {args.output_dir}"
        )
    if not args.expected_code_version:
        raise ValueError("expected_code_version must be non-empty")
    started = time.perf_counter()
    grid = build_grid_spec(load_config(args.base_config))
    library = load_verification_actions(args.actions_config)
    config = load_verify_model_config(args.model_config)
    shard_dirs = tuple(args.shard_dir)
    loaded_shards = tuple(
        load_verification_shard(path, grid=grid, library=library)
        for path in shard_dirs
    )
    handoff = validate_verification_collection_handoff(
        args.collection_handoff,
        shard_dirs=shard_dirs,
        loaded_shards=loaded_shards,
        expected_split=args.split,
    )
    collection = load_verification_collection(
        shard_dirs, grid=grid, library=library
    )
    evaluation_input_digest, evaluation_split_digests = (
        verification_input_digests(loaded_shards)
    )

    external_manifest = _strict_json(
        args.checkpoint_manifest, label="checkpoint manifest"
    )
    if external_manifest.get("code_version") != args.expected_code_version:
        raise ValueError("checkpoint manifest code version differs from trust anchor")
    training_input_digest = _digest(
        external_manifest.get("input_manifest_digest"),
        name="training input manifest digest",
    )
    raw_training_splits = external_manifest.get("split_digests")
    if not isinstance(raw_training_splits, dict) or set(raw_training_splits) != {
        "train"
    }:
        raise ValueError("checkpoint must be bound to exactly one train split")
    training_split_digests = {
        "train": _digest(
            raw_training_splits["train"], name="training split digest"
        )
    }
    config_payload = asdict(config)
    checkpoint = load_verification_training_checkpoint(
        args.checkpoint,
        expected_input_manifest_digest=training_input_digest,
        expected_split_digests=training_split_digests,
        expected_model_config=config_payload,
        expected_seed=config.training.seed,
        expected_code_version=args.expected_code_version,
    )
    if checkpoint.manifest != external_manifest:
        raise ValueError("external and embedded checkpoint manifests differ")

    evaluated = evaluate_verification_samples(
        collection.samples,
        grid=grid,
        config=config,
        checkpoint=checkpoint,
        split=args.split,
    )
    baselines = evaluate_verification_baselines(
        collection.samples, huber_delta=config.loss.huber_delta
    )
    limitations = list(handoff["limitations"])
    limitations.extend(
        [
            "held-out smoke metrics do not estimate paper-scale uncertainty",
            "paper F1/ranking/regret thresholds are not evaluated",
            "target object, footprint, and blind-type slices are unavailable in the current verification manifest",
        ]
    )
    if sum(
        value > 0 for value in evaluated.metrics["oracle_best_action_counts"].values()
    ) <= 1:
        limitations.append("oracle-best action lacks diversity in this smoke collection")
    if sum(
        value > 0 for value in evaluated.metrics["selected_action_counts"].values()
    ) <= 1:
        limitations.append("model selections collapse to one action in this smoke")

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_cli_version": EVALUATION_CLI_VERSION,
        "scientific_status": handoff["scientific_status"],
        "split": args.split,
        "paper_thresholds_evaluated": False,
        "losses": evaluated.losses,
        "learned": evaluated.metrics,
        "baselines": baselines,
        "limitations": limitations,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_cli_version": EVALUATION_CLI_VERSION,
        "scientific_status": handoff["scientific_status"],
        "split": args.split,
        "sample_count": evaluated.sample_count,
        "group_count": evaluated.group_count,
        "collection_semantic_digest": handoff["collection_semantic_digest"],
        "evaluation_input_manifest_digest": evaluation_input_digest,
        "evaluation_split_digests": evaluation_split_digests,
        "training_input_manifest_digest": training_input_digest,
        "training_split_digests": training_split_digests,
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "checkpoint_manifest_sha256": _sha256_file(args.checkpoint_manifest),
        "checkpoint_code_version": args.expected_code_version,
        "evaluation_implementation_digest_sha256": _implementation_digest(
            (args.base_config, args.actions_config, args.model_config)
        ),
        "completed_training_epochs": checkpoint.completed_epochs,
        "elapsed_seconds": time.perf_counter() - started,
        "device": evaluated.device,
        "limitations": limitations,
    }

    output = args.output_dir
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        (staging / "metrics.json").write_bytes(_json_bytes(metrics))
        (staging / "evaluation_report.json").write_bytes(_json_bytes(report))
        os.rename(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "split": args.split,
                "sample_count": evaluated.sample_count,
                "scientific_status": handoff["scientific_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
