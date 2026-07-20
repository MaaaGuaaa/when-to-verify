#!/usr/bin/env python3
"""Train the schema-3 SOP14 V0 model on an immutable train collection."""

from __future__ import annotations

import argparse
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
from src.evaluation.verification_metrics import (
    build_verification_checkpoint_manifest,
)
from src.models.verification_model import load_verify_model_config
from src.models.verification_training import (
    load_verification_training_checkpoint,
    train_verification_samples,
    write_verification_training_checkpoint,
)
from src.planning.verification_actions import load_verification_actions
from src.utils.config import load_config


TRAINING_CLI_VERSION = "verification_training_cli_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--collection-handoff", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--actions-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser


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
        raise FileExistsError(f"refusing to overwrite immutable output: {args.output_dir}")
    if not isinstance(args.code_version, str) or not args.code_version:
        raise ValueError("code_version must be non-empty")
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
        expected_split="train",
    )
    collection = load_verification_collection(
        shard_dirs, grid=grid, library=library
    )
    input_manifest_digest, split_digests = verification_input_digests(loaded_shards)
    config_payload = asdict(config)
    manifest = build_verification_checkpoint_manifest(
        input_manifest_digest=input_manifest_digest,
        split_digests=split_digests,
        model_config=config_payload,
        seed=config.training.seed,
        code_version=args.code_version,
    )
    resume = None
    if args.resume_checkpoint is not None:
        resume = load_verification_training_checkpoint(
            args.resume_checkpoint,
            expected_input_manifest_digest=input_manifest_digest,
            expected_split_digests=split_digests,
            expected_model_config=config_payload,
            expected_seed=config.training.seed,
            expected_code_version=args.code_version,
        )
    training = train_verification_samples(
        collection.samples,
        grid=grid,
        config=config,
        resume=resume,
    )
    learned = dict(training.metrics)
    baselines = evaluate_verification_baselines(
        collection.samples, huber_delta=config.loss.huber_delta
    )
    scientific_status = str(handoff["scientific_status"])
    limitations = list(handoff["limitations"])
    limitations.extend(
        [
            "metrics are train-fit smoke metrics, not validation/test estimates",
            "paper F1/ranking/regret thresholds are not evaluated",
        ]
    )
    oracle_best_counts = learned["oracle_best_action_counts"]
    selected_counts = learned["selected_action_counts"]
    if sum(value > 0 for value in oracle_best_counts.values()) <= 1:
        limitations.append(
            "oracle-best action lacks diversity in this smoke collection"
        )
    if sum(value > 0 for value in selected_counts.values()) <= 1:
        limitations.append(
            "model selections collapse to one action on this train-fit smoke"
        )
    limitations.append(
        "target object, footprint, and blind-type slices are unavailable in the current verification manifest"
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "training_cli_version": TRAINING_CLI_VERSION,
        "scientific_status": scientific_status,
        "paper_thresholds_evaluated": False,
        "learned": learned,
        "baselines": baselines,
        "limitations": limitations,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "training_cli_version": TRAINING_CLI_VERSION,
        "scientific_status": scientific_status,
        "split": "train",
        "sample_count": len(collection.samples),
        "group_count": collection.audit_report["group_count"],
        "collection_semantic_digest": handoff["collection_semantic_digest"],
        "input_manifest_digest": input_manifest_digest,
        "split_digests": split_digests,
        "completed_epochs": training.completed_epochs,
        "initial_loss": training.initial_loss,
        "final_loss": training.final_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "code_version": args.code_version,
        "resumed_from": (
            None if args.resume_checkpoint is None else str(args.resume_checkpoint)
        ),
        "limitations": limitations,
    }

    output = args.output_dir
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        checkpoint = write_verification_training_checkpoint(
            staging / "checkpoint.pt", result=training, manifest=manifest
        )
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        (staging / "metrics.json").write_bytes(_json_bytes(metrics))
        (staging / "training_report.json").write_bytes(_json_bytes(report))
        load_verification_training_checkpoint(
            checkpoint,
            expected_input_manifest_digest=input_manifest_digest,
            expected_split_digests=split_digests,
            expected_model_config=config_payload,
            expected_seed=config.training.seed,
            expected_code_version=args.code_version,
        )
        os.rename(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "sample_count": len(collection.samples),
                "scientific_status": scientific_status,
                "initial_loss": training.initial_loss,
                "final_loss": training.final_loss,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
