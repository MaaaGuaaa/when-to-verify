#!/usr/bin/env python3
"""Evaluate one frozen SOP14 checkpoint on an immutable held-out collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from src.evaluation.verification_metrics import verification_slice_fields
from src.evaluation.verification_run_artifacts import (
    VERIFICATION_EVALUATION_RUN_LAYOUT_VERSION,
    canonical_json_bytes,
    publish_authenticated_run_directory,
)
from src.evaluation.verification_value_calibration import (
    load_reject_cost_calibration,
)
from src.models.verification_model import load_verify_model_config
from src.models.verification_training import (
    evaluate_verification_samples,
    load_verification_training_checkpoint,
)
from src.planning.verification_actions import load_verification_actions
from src.utils.config import load_config


EVALUATION_CLI_VERSION = "verification_evaluation_cli_v2"
_EVALUATION_PAYLOADS = frozenset(
    {"evaluation_report.json", "metrics.json", "predictions.jsonl"}
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=("calibration", "val", "test"), required=True
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--release-dir", type=Path)
    inputs.add_argument("--shard-dir", type=Path, action="append")
    parser.add_argument("--value-calibration", type=Path)
    parser.add_argument("--collection-handoff", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--actions-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--expected-code-version", required=True)
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
    return canonical_json_bytes(value)


def _json_lines_bytes(rows: Sequence[dict[str, object]]) -> bytes:
    if not rows:
        raise ValueError("prediction rows must be non-empty")
    return b"".join(canonical_json_bytes(row) for row in rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_input_args(args)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite immutable output: {args.output_dir}"
        )
    if not args.expected_code_version:
        raise ValueError("expected_code_version must be non-empty")
    started = time.perf_counter()
    grid = build_grid_spec(load_config(args.base_config))
    library = load_verification_actions(args.actions_config)
    config = load_verify_model_config(args.model_config)
    external_manifest = _strict_json(
        args.checkpoint_manifest, label="checkpoint manifest"
    )
    if external_manifest.get("code_version") != args.expected_code_version:
        raise ValueError("checkpoint manifest code version differs from trust anchor")
    checkpoint_seed = external_manifest.get("seed")
    if isinstance(checkpoint_seed, bool) or not isinstance(checkpoint_seed, int):
        raise ValueError("checkpoint manifest seed is invalid")
    config = replace(
        config,
        training=replace(config.training, seed=checkpoint_seed),
    )
    if args.release_dir is not None:
        calibration = load_reject_cost_calibration(args.value_calibration)
        collection = load_calibrated_verification_release(
            args.release_dir,
            grid=grid,
            library=library,
            expected_split=args.split,
            calibration=calibration,
        )
        slices = verification_slice_fields(
            collection.samples,
            require_complete=True,
        )
        samples = collection.samples
        evaluation_input_digest = collection.input_manifest_digest
        evaluation_split_digests = dict(collection.split_digests)
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
            expected_split=args.split,
        )
        collection = load_verification_collection(
            shard_dirs, grid=grid, library=library
        )
        samples = collection.samples
        evaluation_input_digest, evaluation_split_digests = (
            verification_input_digests(loaded_shards)
        )
        slices = verification_slice_fields(samples)
        scientific_status = str(handoff["scientific_status"])
        limitations = list(handoff["limitations"])
        collection_semantic_digest = str(
            handoff["collection_semantic_digest"]
        )
        release_manifest_digest = None
        value_calibration_digest = None
        reject_cost = None
        data_mode = "bounded_fixture"
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
        expected_seed=checkpoint_seed,
        expected_code_version=args.expected_code_version,
        expected_value_calibration_digest=value_calibration_digest,
        expected_reject_cost=reject_cost,
    )
    if checkpoint.manifest != external_manifest:
        raise ValueError("external and embedded checkpoint manifests differ")

    evaluated = evaluate_verification_samples(
        samples,
        grid=grid,
        config=config,
        checkpoint=checkpoint,
        split=args.split,
    )
    baselines = evaluate_verification_baselines(
        samples,
        huber_delta=config.loss.huber_delta,
        slice_fields=slices,
    )
    limitations.extend(
        [
            "paper F1/ranking/regret thresholds are not evaluated",
        ]
    )
    if data_mode == "bounded_fixture":
        limitations.extend(
            [
                "held-out smoke metrics do not estimate paper-scale uncertainty",
                "production provenance slices may be unavailable in bounded fixtures",
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
        "scientific_status": scientific_status,
        "split": args.split,
        "paper_thresholds_evaluated": False,
        "seed": checkpoint_seed,
        "value_calibration_digest": value_calibration_digest,
        "reject_cost": reject_cost,
        "losses": evaluated.losses,
        "learned": evaluated.metrics,
        "baselines": baselines,
        "limitations": limitations,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_cli_version": EVALUATION_CLI_VERSION,
        "scientific_status": scientific_status,
        "run_layout_version": VERIFICATION_EVALUATION_RUN_LAYOUT_VERSION,
        "data_mode": data_mode,
        "split": args.split,
        "sample_count": evaluated.sample_count,
        "group_count": evaluated.group_count,
        "collection_semantic_digest": collection_semantic_digest,
        "release_manifest_digest": release_manifest_digest,
        "evaluation_input_manifest_digest": evaluation_input_digest,
        "evaluation_split_digests": evaluation_split_digests,
        "training_input_manifest_digest": training_input_digest,
        "training_split_digests": training_split_digests,
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "checkpoint_manifest_sha256": _sha256_file(args.checkpoint_manifest),
        "checkpoint_code_version": args.expected_code_version,
        "seed": checkpoint_seed,
        "model_config": config_payload,
        "model_config_digest": external_manifest["model_config_digest"],
        "value_calibration_digest": value_calibration_digest,
        "reject_cost": reject_cost,
        "evaluation_implementation_digest_sha256": _implementation_digest(
            (args.base_config, args.actions_config, args.model_config)
        ),
        "completed_training_epochs": checkpoint.completed_epochs,
        "elapsed_seconds": time.perf_counter() - started,
        "device": evaluated.device,
        "limitations": limitations,
    }
    ordered_samples = tuple(sorted(samples, key=lambda row: row.sample_id))
    prediction_rows = [
        {
            "sample_id": sample.sample_id,
            "split": sample.split,
            "ranking_group_id": str(
                sample.metadata["ranking_group_id"]
            ),
            "action_id": sample.verification_action_id,
            "value_target": sample.value_target,
            "value_prediction": float(evaluated.value_prediction[index]),
            "useful_target": sample.useful_target,
            "useful_probability": float(
                evaluated.useful_probability[index]
            ),
            **{
                field: slices[field][index]
                for field in (
                    "source_mode",
                    "blind_type",
                    "target_object_type",
                    "target_footprint_kind",
                )
            },
        }
        for index, sample in enumerate(ordered_samples)
    ]
    run_identity_payload = {
        "split": args.split,
        "evaluation_split_digests": evaluation_split_digests,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "model_config_digest": report["model_config_digest"],
        "seed": checkpoint_seed,
        "value_calibration_digest": value_calibration_digest,
    }
    report["run_identity"] = hashlib.sha256(
        b"verification-evaluation-run-v2\0"
        + canonical_json_bytes(run_identity_payload)
    ).hexdigest()

    output = args.output_dir
    publish_authenticated_run_directory(
        output,
        layout_version=VERIFICATION_EVALUATION_RUN_LAYOUT_VERSION,
        payloads={
            "metrics.json": _json_bytes(metrics),
            "evaluation_report.json": _json_bytes(report),
            "predictions.jsonl": _json_lines_bytes(prediction_rows),
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "split": args.split,
                "sample_count": evaluated.sample_count,
                "scientific_status": scientific_status,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
