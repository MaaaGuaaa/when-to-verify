#!/usr/bin/env python3
"""Train the Schema-4 SOP14 V0 model on an immutable train collection."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, replace
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
from src.datasets.verification_release_collection import (
    load_calibrated_verification_release,
)
from src.evaluation.verification_baselines import evaluate_verification_baselines
from src.evaluation.verification_metrics import (
    build_verification_checkpoint_manifest,
    verification_slice_fields,
)
from src.evaluation.verification_run_artifacts import (
    VERIFICATION_TRAINING_RUN_LAYOUT_VERSION,
    canonical_json_bytes,
    load_authenticated_run_directory,
    seal_authenticated_run_staging,
)
from src.evaluation.verification_value_calibration import (
    load_reject_cost_calibration,
)
from src.models.verification_model import load_verify_model_config
from src.models.verification_training import (
    load_verification_training_checkpoint,
    train_verification_samples,
    write_verification_training_checkpoint,
)
from src.planning.verification_actions import load_verification_actions
from src.utils.atomic_publish import atomic_rename_noreplace
from src.utils.config import load_config


TRAINING_CLI_VERSION = "verification_training_cli_v2"
_TRAINING_PAYLOADS = frozenset(
    {
        "checkpoint.pt",
        "manifest.json",
        "metrics.json",
        "training_report.json",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--release-dir", type=Path)
    inputs.add_argument("--shard-dir", type=Path, action="append")
    parser.add_argument("--value-calibration", type=Path)
    parser.add_argument("--collection-handoff", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--actions-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--seed", type=int)
    return parser


def _validate_input_args(args: argparse.Namespace) -> None:
    if args.release_dir is not None:
        if args.value_calibration is None:
            raise ValueError(
                "--release-dir requires --value-calibration"
            )
        if args.collection_handoff is not None:
            raise ValueError(
                "--collection-handoff is only valid with --shard-dir"
            )
    else:
        if not args.shard_dir or args.collection_handoff is None:
            raise ValueError(
                "--shard-dir requires --collection-handoff"
            )
        if args.value_calibration is not None:
            raise ValueError(
                "--value-calibration is only valid with --release-dir"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_input_args(args)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable output: {args.output_dir}")
    if not isinstance(args.code_version, str) or not args.code_version:
        raise ValueError("code_version must be non-empty")
    started = time.perf_counter()
    grid = build_grid_spec(load_config(args.base_config))
    library = load_verification_actions(args.actions_config)
    config = load_verify_model_config(args.model_config)
    if args.seed is not None:
        config = replace(
            config,
            training=replace(config.training, seed=args.seed),
        )
    if args.release_dir is not None:
        calibration = load_reject_cost_calibration(args.value_calibration)
        collection = load_calibrated_verification_release(
            args.release_dir,
            grid=grid,
            library=library,
            expected_split="train",
            calibration=calibration,
        )
        verification_slice_fields(collection.samples, require_complete=True)
        samples = collection.samples
        audit_report = collection.audit_report
        input_manifest_digest = collection.input_manifest_digest
        split_digests = dict(collection.split_digests)
        scientific_status = "production_release"
        limitations: list[str] = []
        collection_semantic_digest = collection.split_digest
        release_manifest_digest = collection.release_manifest_digest
        value_calibration_digest = collection.calibration_digest
        reject_cost = collection.reject_cost
        data_mode = "release"
    else:
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
        samples = collection.samples
        audit_report = collection.audit_report
        input_manifest_digest, split_digests = verification_input_digests(
            loaded_shards
        )
        scientific_status = str(handoff["scientific_status"])
        limitations = list(handoff["limitations"])
        collection_semantic_digest = str(
            handoff["collection_semantic_digest"]
        )
        release_manifest_digest = None
        value_calibration_digest = None
        reject_cost = None
        data_mode = "bounded_fixture"
    config_payload = asdict(config)
    manifest = build_verification_checkpoint_manifest(
        input_manifest_digest=input_manifest_digest,
        split_digests=split_digests,
        model_config=config_payload,
        seed=config.training.seed,
        code_version=args.code_version,
        value_calibration_digest=value_calibration_digest,
        reject_cost=reject_cost,
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
            expected_value_calibration_digest=value_calibration_digest,
            expected_reject_cost=reject_cost,
        )
    training = train_verification_samples(
        samples,
        grid=grid,
        config=config,
        resume=resume,
    )
    learned = dict(training.metrics)
    baselines = evaluate_verification_baselines(
        samples,
        huber_delta=config.loss.huber_delta,
        slice_fields=verification_slice_fields(samples),
    )
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
    if data_mode == "bounded_fixture":
        limitations.append(
            "production provenance slices may be unavailable in bounded fixtures"
        )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "training_cli_version": TRAINING_CLI_VERSION,
        "scientific_status": scientific_status,
        "paper_thresholds_evaluated": False,
        "seed": config.training.seed,
        "value_calibration_digest": value_calibration_digest,
        "reject_cost": reject_cost,
        "learned": learned,
        "baselines": baselines,
        "limitations": limitations,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "training_cli_version": TRAINING_CLI_VERSION,
        "scientific_status": scientific_status,
        "run_layout_version": VERIFICATION_TRAINING_RUN_LAYOUT_VERSION,
        "data_mode": data_mode,
        "split": "train",
        "sample_count": len(samples),
        "group_count": audit_report["group_count"],
        "collection_semantic_digest": collection_semantic_digest,
        "release_manifest_digest": release_manifest_digest,
        "input_manifest_digest": input_manifest_digest,
        "split_digests": split_digests,
        "value_calibration_digest": value_calibration_digest,
        "reject_cost": reject_cost,
        "seed": config.training.seed,
        "model_config": config_payload,
        "model_config_digest": manifest["model_config_digest"],
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
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        (staging / "metrics.json").write_bytes(canonical_json_bytes(metrics))
        (staging / "training_report.json").write_bytes(
            canonical_json_bytes(report)
        )
        load_verification_training_checkpoint(
            checkpoint,
            expected_input_manifest_digest=input_manifest_digest,
            expected_split_digests=split_digests,
            expected_model_config=config_payload,
            expected_seed=config.training.seed,
            expected_code_version=args.code_version,
            expected_value_calibration_digest=value_calibration_digest,
            expected_reject_cost=reject_cost,
        )
        seal_authenticated_run_staging(
            staging,
            layout_version=VERIFICATION_TRAINING_RUN_LAYOUT_VERSION,
            required_payloads=_TRAINING_PAYLOADS,
        )
        atomic_rename_noreplace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    load_authenticated_run_directory(
        output,
        expected_layout_version=VERIFICATION_TRAINING_RUN_LAYOUT_VERSION,
        required_payloads=_TRAINING_PAYLOADS,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "sample_count": len(samples),
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
