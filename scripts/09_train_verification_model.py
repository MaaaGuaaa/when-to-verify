#!/usr/bin/env python3
"""Train the schema-3 SOP14 V0 model on an immutable train collection."""

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

import numpy as np

from src.contracts import SCHEMA_VERSION, build_grid_spec
from src.datasets.verification_dataloader import (
    load_verification_collection,
    load_verification_shard,
)
from src.evaluation.verification_baselines import (
    critical_swept_coverage_score,
    occupancy_entropy_reduction_score,
    visible_area_score,
)
from src.evaluation.verification_metrics import (
    build_verification_checkpoint_manifest,
    evaluate_verification_predictions,
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
_HANDOFF_KEYS = frozenset(
    {
        "schema_version",
        "handoff_version",
        "collection_state",
        "scientific_status",
        "split",
        "sample_count",
        "group_count",
        "collection_semantic_digest",
        "generation_report_sha256",
        "shards",
        "limitations",
    }
)


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


def _strict_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid collection handoff: {exc}") from exc
    if not isinstance(value, dict) or set(value) != _HANDOFF_KEYS:
        raise ValueError("collection handoff keys are invalid")
    return value


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_handoff(
    path: Path,
    *,
    shard_dirs: tuple[Path, ...],
    loaded_shards,
) -> dict[str, object]:
    handoff = _strict_json(path)
    if handoff["schema_version"] != SCHEMA_VERSION:
        raise ValueError("collection handoff schema mismatch")
    if handoff["handoff_version"] != "verification_collection_handoff_v1":
        raise ValueError("unsupported verification collection handoff")
    if handoff["collection_state"] != "complete":
        raise ValueError("verification collection is incomplete")
    if handoff["split"] != "train":
        raise ValueError("verification model fitting requires a train handoff")
    if handoff["scientific_status"] not in {
        "toy_smoke_only",
        "train_smoke_only",
        "publication_ready",
    }:
        raise ValueError("collection scientific_status is invalid")
    _digest(handoff["collection_semantic_digest"], name="collection digest")
    _digest(handoff["generation_report_sha256"], name="generation report digest")
    rows = handoff["shards"]
    if not isinstance(rows, list) or len(rows) != len(shard_dirs):
        raise ValueError("collection handoff shard count mismatch")
    expected_roots = {
        (path.parent / str(row["relative_root"])).resolve(): row for row in rows
    }
    if set(expected_roots) != {value.resolve() for value in shard_dirs}:
        raise ValueError("collection handoff shard roots mismatch")
    for root, loaded in zip(shard_dirs, loaded_shards, strict=True):
        row = expected_roots[root.resolve()]
        if not isinstance(row, dict) or set(row) != {
            "shard_index",
            "relative_root",
            "sample_count",
            "semantic_digest",
        }:
            raise ValueError("collection handoff shard row keys are invalid")
        if row["sample_count"] != len(loaded.samples):
            raise ValueError("collection handoff shard sample count mismatch")
        if row["semantic_digest"] != loaded.semantic_digest:
            raise ValueError("collection handoff shard semantic digest mismatch")
    sample_count = sum(len(value.samples) for value in loaded_shards)
    if handoff["sample_count"] != sample_count:
        raise ValueError("collection handoff total sample count mismatch")
    if not isinstance(handoff["limitations"], list) or any(
        not isinstance(value, str) or not value for value in handoff["limitations"]
    ):
        raise ValueError("collection handoff limitations are invalid")
    return handoff


def _domain_digest(domain: bytes, values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    for value in sorted(values):
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _input_digests(loaded_shards) -> tuple[str, dict[str, str]]:
    manifests = [value.manifest_digest for value in loaded_shards]
    input_manifest = (
        manifests[0]
        if len(manifests) == 1
        else _domain_digest(b"verification-input-manifests-v1\0", manifests)
    )
    by_split: dict[str, list[str]] = {}
    for loaded in loaded_shards:
        split = str(loaded.summary["split"])
        by_split.setdefault(split, []).append(loaded.semantic_digest)
    split_digests = {
        split: (
            values[0]
            if len(values) == 1
            else _domain_digest(
                b"verification-split-shards-v1\0" + split.encode("ascii") + b"\0",
                values,
            )
        )
        for split, values in sorted(by_split.items())
    }
    return input_manifest, split_digests


def _baseline_metrics(samples, *, huber_delta: float) -> dict[str, object]:
    score_functions = {
        "visible_area": lambda sample: visible_area_score(
            state_channels=sample.state_channels,
            verification_fov_mask=sample.verification_fov_mask,
        ),
        "critical_swept_coverage": lambda sample: critical_swept_coverage_score(
            state_channels=sample.state_channels,
            trajectory_channels=sample.trajectory_channels,
            verification_fov_mask=sample.verification_fov_mask,
        ),
        "occupancy_entropy": lambda sample: occupancy_entropy_reduction_score(
            state_channels=sample.state_channels,
            verification_fov_mask=sample.verification_fov_mask,
        ),
    }
    result: dict[str, object] = {}
    values = np.asarray([sample.value_target for sample in samples], dtype=np.float64)
    useful = np.asarray([sample.useful_target for sample in samples], dtype=np.int64)
    groups = tuple(str(sample.metadata["ranking_group_id"]) for sample in samples)
    actions = tuple(sample.verification_action_id for sample in samples)
    for name, function in score_functions.items():
        scores = np.asarray([function(sample) for sample in samples], dtype=np.float64)
        report = evaluate_verification_predictions(
            value_prediction=scores,
            useful_probability=np.full(scores.shape, 0.5, dtype=np.float64),
            value_target=values,
            useful_target=useful,
            group_ids=groups,
            action_ids=actions,
            huber_delta=huber_delta,
        )
        result[name] = {
            key: report[key]
            for key in (
                "pairwise_accuracy",
                "pair_count",
                "top1_regret_mean",
                "top_two_selection_rate",
                "selected_action_counts",
                "selected_action_proportions",
            )
        }
    return result


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
    handoff = _validate_handoff(
        args.collection_handoff,
        shard_dirs=shard_dirs,
        loaded_shards=loaded_shards,
    )
    collection = load_verification_collection(
        shard_dirs, grid=grid, library=library
    )
    input_manifest_digest, split_digests = _input_digests(loaded_shards)
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
    baselines = _baseline_metrics(
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
