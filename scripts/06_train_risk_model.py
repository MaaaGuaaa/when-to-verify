#!/usr/bin/env python
"""Train deterministic SOP09 R0/R1 models in explicit toy mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Mapping

import yaml
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.contracts import SCHEMA_VERSION  # noqa: E402
from src.calibration.split_conformal import (  # noqa: E402
    PREDICTION_TABLE_LAYOUT_VERSION,
    prediction_table_cohort_digest,
    prediction_table_semantic_digest,
    validate_prediction_table,
)
from src.datasets.risk_dataloader import (  # noqa: E402
    RiskDataContractError,
    collate_risk_samples,
    load_production_risk_dataset,
    select_production_risk_subset,
)
from src.datasets.toy_risk_learning import (  # noqa: E402
    assert_toy_split_isolation,
    frozen_channel_spec,
    make_toy_risk_dataset,
    validate_toy_risk_dataset_publication,
)
from src.models.risk_model import (  # noqa: E402
    RISK_CHECKPOINT_LAYOUT_VERSION,
    load_risk_checkpoint,
    save_risk_checkpoint,
    train_toy_risk_model,
)
from src.training.risk_trainer import (  # noqa: E402
    ProductionRiskTrainingConfig,
    train_production_risk_model,
)

ARTIFACT_LAYOUT_VERSION = "sop09_toy_risk_training_v2"
_CONFIG_KEYS = {
    "mode",
    "seed",
    "grid_size",
    "train_count",
    "validation_count",
    "calibration_count",
    "test_count",
    "variants",
    "hidden_channels",
    "optimization_steps",
    "learning_rate",
    "lambda_collision",
    "lambda_occupancy_aux",
    "optimizer",
}
_PRODUCTION_CONFIG_KEYS = {
    "mode",
    "stage",
    "variant",
    "seed",
    "device",
    "hidden_channels",
    "max_samples",
    "batch_size",
    "epochs",
    "gradient_accumulation_steps",
    "learning_rate",
    "weight_decay",
    "lambda_collision",
    "checkpoint_interval_steps",
    "optimizer",
}


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise RiskDataContractError("risk model config must be a mapping")
    return dict(value)


def _load_config(path: Path) -> dict[str, object]:
    value = _load_yaml_mapping(path)
    unknown = sorted(set(value) - _CONFIG_KEYS)
    missing = sorted(_CONFIG_KEYS - set(value))
    if unknown:
        raise RiskDataContractError(f"unknown risk model config keys: {unknown}")
    if missing:
        raise RiskDataContractError(f"missing risk model config keys: {missing}")
    if value["mode"] != "toy":
        raise RiskDataContractError("toy risk model config mode must be toy")
    if value["variants"] != ["r0", "r1"]:
        raise RiskDataContractError("toy comparison variants must be exactly [r0, r1]")
    if value["optimizer"] != "AdamW":
        raise RiskDataContractError("SOP09 optimizer must be AdamW")
    if float(value["lambda_occupancy_aux"]) != 0.0:
        raise RiskDataContractError(
            "occupancy auxiliary training is unavailable without a bound label head"
        )
    for field in (
        "seed",
        "grid_size",
        "train_count",
        "validation_count",
        "calibration_count",
        "test_count",
        "hidden_channels",
        "optimization_steps",
    ):
        if not isinstance(value[field], int) or isinstance(value[field], bool):
            raise RiskDataContractError(f"risk model config {field} must be an integer")
    for field in ("learning_rate", "lambda_collision"):
        if not isinstance(value[field], (int, float)) or isinstance(value[field], bool):
            raise RiskDataContractError(f"risk model config {field} must be numeric")
    return dict(value)


def _load_production_config(path: Path) -> dict[str, object]:
    value = _load_yaml_mapping(path)
    unknown = sorted(set(value) - _PRODUCTION_CONFIG_KEYS)
    missing = sorted(_PRODUCTION_CONFIG_KEYS - set(value))
    if unknown:
        raise RiskDataContractError(
            f"unknown production risk model config keys: {unknown}"
        )
    if missing:
        raise RiskDataContractError(
            f"missing production risk model config keys: {missing}"
        )
    if value["mode"] != "production":
        raise RiskDataContractError("production risk model config mode must be production")
    if value["optimizer"] != "AdamW":
        raise RiskDataContractError("SOP09 production optimizer must be AdamW")
    max_samples = value["max_samples"]
    if type(max_samples) is not int or max_samples < 1:
        raise RiskDataContractError("production max_samples must be a positive integer")
    ProductionRiskTrainingConfig(
        **{
            key: value[key]
            for key in (
                "stage",
                "variant",
                "seed",
                "device",
                "hidden_channels",
                "batch_size",
                "epochs",
                "gradient_accumulation_steps",
                "learning_rate",
                "weight_decay",
                "lambda_collision",
                "checkpoint_interval_steps",
            )
        }
    )
    return dict(value)


def _effective_config(args: argparse.Namespace) -> dict[str, object]:
    raw = _load_yaml_mapping(args.config)
    if raw.get("mode") == "production":
        config = _load_production_config(args.config)
        for argument, field in (
            (args.variant, "variant"),
            (args.stage, "stage"),
            (args.max_samples, "max_samples"),
            (args.batch_size, "batch_size"),
            (args.device, "device"),
            (args.hidden_channels, "hidden_channels"),
        ):
            if argument is not None:
                config[field] = argument
        ProductionRiskTrainingConfig(
            **{
                key: config[key]
                for key in (
                    "stage",
                    "variant",
                    "seed",
                    "device",
                    "hidden_channels",
                    "batch_size",
                    "epochs",
                    "gradient_accumulation_steps",
                    "learning_rate",
                    "weight_decay",
                    "lambda_collision",
                    "checkpoint_interval_steps",
                )
            }
        )
        return config
    config = _load_config(args.config)
    for argument, field in (
        (args.optimization_steps, "optimization_steps"),
        (args.train_count, "train_count"),
        (args.validation_count, "validation_count"),
        (args.calibration_count, "calibration_count"),
        (args.test_count, "test_count"),
        (args.grid_size, "grid_size"),
        (args.hidden_channels, "hidden_channels"),
    ):
        if argument is not None:
            config[field] = argument
    return config


def _config_digest(config: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(config)).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SOP09 R0/R1 with strict toy/production provenance."
    )
    parser.add_argument(
        "--config", type=Path, default=_ROOT / "configs/risk_model.yaml"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-commit", default="unversioned")
    parser.add_argument("--optimization-steps", type=_positive_int)
    parser.add_argument("--train-count", type=_positive_int)
    parser.add_argument("--validation-count", type=_positive_int)
    parser.add_argument("--calibration-count", type=_positive_int)
    parser.add_argument("--test-count", type=_positive_int)
    parser.add_argument("--grid-size", type=_positive_int)
    parser.add_argument("--hidden-channels", type=_positive_int)
    parser.add_argument("--train-seal-root", type=Path)
    parser.add_argument("--train-collection-root", type=Path)
    parser.add_argument("--validation-seal-root", type=Path)
    parser.add_argument("--validation-collection-root", type=Path)
    parser.add_argument("--variant", choices=("r0", "r1"))
    parser.add_argument(
        "--stage",
        choices=("one_shard_smoke", "real_1k_overfit", "formal_50k"),
    )
    parser.add_argument("--max-samples", type=_positive_int)
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--device")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-publication-instance-digest")
    parser.add_argument("--cross-split-audit", type=Path)
    return parser


def _load_cross_split_audit(path: Path | None) -> Mapping[str, object] | None:
    if path is None:
        return None
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RiskDataContractError(
            f"unable to load finite cross-split audit JSON: {error}"
        ) from error
    if not isinstance(value, dict):
        raise RiskDataContractError("cross-split audit must be a JSON mapping")
    return value


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite existing output: {args.output_dir}")
    try:
        config = _effective_config(args)
        if config["mode"] == "production":
            if args.train_seal_root is None or args.train_collection_root is None:
                raise RiskDataContractError(
                    "production mode requires --train-seal-root and --train-collection-root"
                )
            validation_paths = (
                args.validation_seal_root,
                args.validation_collection_root,
            )
            if (validation_paths[0] is None) != (validation_paths[1] is None):
                raise RiskDataContractError(
                    "validation seal and collection roots must be supplied together"
                )
            train_dataset = load_production_risk_dataset(
                args.train_seal_root,
                collection_root=args.train_collection_root,
                expected_split="train",
            )
            validation_dataset = (
                None
                if validation_paths[0] is None
                else load_production_risk_dataset(
                    validation_paths[0],
                    collection_root=validation_paths[1],
                    expected_split="val",
                )
            )
            subset = select_production_risk_subset(
                train_dataset,
                max_samples=int(config["max_samples"]),
                seed=int(config["seed"]),
            )
            training_config = ProductionRiskTrainingConfig(
                **{
                    key: config[key]
                    for key in (
                        "stage",
                        "variant",
                        "seed",
                        "device",
                        "hidden_channels",
                        "batch_size",
                        "epochs",
                        "gradient_accumulation_steps",
                        "learning_rate",
                        "weight_decay",
                        "lambda_collision",
                        "checkpoint_interval_steps",
                    )
                }
            )
            result = train_production_risk_model(
                train_dataset=train_dataset,
                train_subset=subset,
                config=training_config,
                output_dir=args.output_dir,
                code_commit=args.code_commit,
                validation_dataset=validation_dataset,
                resume_from=args.resume_from,
                resume_expected_publication_instance_digest_sha256=(
                    args.resume_publication_instance_digest
                ),
                cross_split_audit=_load_cross_split_audit(args.cross_split_audit),
            )
            print(
                json.dumps(
                    {
                        "output_dir": str(result.output_dir),
                        "semantic_digest_sha256": result.semantic_digest_sha256,
                    },
                    sort_keys=True,
                )
            )
            return 0
    except (OSError, ValueError, RiskDataContractError) as error:
        parser.error(str(error))

    seed = int(config["seed"])
    grid_size = int(config["grid_size"])
    train_dataset = make_toy_risk_dataset(
        split="train",
        count=int(config["train_count"]),
        seed=seed,
        grid_size=grid_size,
    )
    validation_dataset = make_toy_risk_dataset(
        split="val",
        count=int(config["validation_count"]),
        seed=seed,
        grid_size=grid_size,
    )
    calibration_dataset = make_toy_risk_dataset(
        split="calibration",
        count=int(config["calibration_count"]),
        seed=seed,
        grid_size=grid_size,
    )
    test_dataset = make_toy_risk_dataset(
        split="test",
        count=int(config["test_count"]),
        seed=seed,
        grid_size=grid_size,
    )
    for dataset in (
        train_dataset,
        validation_dataset,
        calibration_dataset,
        test_dataset,
    ):
        validate_toy_risk_dataset_publication(dataset)
    isolation = assert_toy_split_isolation(
        (train_dataset, validation_dataset, calibration_dataset, test_dataset)
    )
    config_snapshot: dict[str, object] = {
        **config,
        "schema_version": SCHEMA_VERSION,
        "channel_spec": frozen_channel_spec(),
    }
    config_digest = _config_digest(config_snapshot)
    staging = args.output_dir.with_name(
        f".{args.output_dir.name}.staging-{os.getpid()}"
    )
    if staging.exists():
        parser.error(f"refusing to overwrite existing staging output: {staging}")
    staging.mkdir(parents=True)
    try:
        _write_json(staging / "config_snapshot.json", config_snapshot)
        variant_metrics: dict[str, object] = {}
        prediction_table_digests: dict[str, dict[str, str]] = {}
        for variant in config["variants"]:
            model, metrics = train_toy_risk_model(
                variant=str(variant),
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                hidden_channels=int(config["hidden_channels"]),
                optimization_steps=int(config["optimization_steps"]),
                learning_rate=float(config["learning_rate"]),
                lambda_collision=float(config["lambda_collision"]),
                seed=seed,
            )
            provenance = {
                "schema_version": SCHEMA_VERSION,
                "channel_spec": frozen_channel_spec(),
                "model_variant": str(variant),
                "config_digest": config_digest,
                "toy_dataset_manifest_digest": train_dataset.manifest_digest,
                "validation_dataset_manifest_digest": (
                    validation_dataset.manifest_digest
                ),
                "seed": seed,
            }
            checkpoint_path = staging / f"{variant}_checkpoint.pt"
            save_risk_checkpoint(
                checkpoint_path,
                model=model,
                mode="toy",
                provenance=provenance,
            )
            variant_metrics[str(variant)] = metrics
            loaded_model, checkpoint_payload = load_risk_checkpoint(
                checkpoint_path,
                expected_mode="toy",
                expected_provenance=provenance,
            )
            checkpoint_digest = str(
                checkpoint_payload["checkpoint_semantic_digest_sha256"]
            )
            prediction_table_digests[str(variant)] = {}
            for prediction_dataset in (calibration_dataset, test_dataset):
                prediction_batch = collate_risk_samples(
                    prediction_dataset.samples,
                    grid=prediction_dataset.grid,
                    dataset_manifest=prediction_dataset.manifest,
                    expected_split=prediction_dataset.split,
                )
                loaded_model.eval()
                with torch.no_grad():
                    prediction = loaded_model(prediction_batch.model_inputs)
                probabilities = prediction["p_collision"].cpu().tolist()
                quantiles = prediction["quantiles"].cpu().tolist()
                rows: list[dict[str, object]] = []
                for row, probability, sample_quantiles in zip(
                    prediction_dataset.manifest_rows,
                    probabilities,
                    quantiles,
                ):
                    rows.append(
                        {
                            **dict(row),
                            "p_collision": float(probability),
                            "q50": float(sample_quantiles[0]),
                            "q80": float(sample_quantiles[1]),
                            "q90": float(sample_quantiles[2]),
                            "q95": float(sample_quantiles[3]),
                        }
                    )
                table: dict[str, object] = {
                    "prediction_table_layout_version": PREDICTION_TABLE_LAYOUT_VERSION,
                    "mode": "toy",
                    "schema_version": SCHEMA_VERSION,
                    "split": prediction_dataset.split,
                    "method_id": str(variant),
                    "checkpoint_layout_version": RISK_CHECKPOINT_LAYOUT_VERSION,
                    "checkpoint_digest": checkpoint_digest,
                    "checkpoint_digest_kind": "risk_checkpoint_semantic_sha256",
                    "toy_dataset_manifest_digest": prediction_dataset.manifest_digest,
                    "seed": seed,
                    "channel_spec": frozen_channel_spec(),
                    "config_digest_sha256": config_digest,
                    "rows": rows,
                }
                table["cohort_digest_sha256"] = prediction_table_cohort_digest(
                    table
                )
                table["semantic_digest"] = prediction_table_semantic_digest(table)
                table = validate_prediction_table(
                    table,
                    expected_mode="toy",
                    expected_split=prediction_dataset.split,
                )
                table_path = staging / (
                    f"{variant}_{prediction_dataset.split}_prediction_table.json"
                )
                _write_json(table_path, table)
                prediction_table_digests[str(variant)][
                    prediction_dataset.split
                ] = str(table["semantic_digest"])

        validation_losses = {
            variant: float(metrics["validation_loss"])
            for variant, metrics in variant_metrics.items()
        }
        lower_validation_loss_variant = min(
            validation_losses, key=lambda name: (validation_losses[name], name)
        )
        comparison = {
            "scope": "toy_software_validation_only",
            "selection_metric": "validation_loss",
            "validation_loss_by_variant": validation_losses,
            "lower_validation_loss_variant": lower_validation_loss_variant,
            "absolute_validation_loss_difference": abs(
                validation_losses["r0"] - validation_losses["r1"]
            ),
        }
        trajectory_ablation = {
            variant: metrics["trajectory_ablation_sensitivity"]
            for variant, metrics in variant_metrics.items()
        }
        metrics_payload = {
            "mode": "toy",
            "variants": variant_metrics,
            "comparison_scope": "toy_software_validation_only",
            "comparison": comparison,
            "trajectory_ablation_sensitivity": trajectory_ablation,
            "selection_split": "val",
            "test_samples_used_for_training_or_selection": 0,
            "test_prediction_rows_generated": len(test_dataset.samples),
        }
        _write_json(staging / "metrics.json", metrics_payload)
        semantic_evidence = {
            "artifact_layout_version": ARTIFACT_LAYOUT_VERSION,
            "config_snapshot": config_snapshot,
            "train_dataset_manifest_digest": train_dataset.manifest_digest,
            "validation_dataset_manifest_digest": validation_dataset.manifest_digest,
            "calibration_dataset_manifest_digest": calibration_dataset.manifest_digest,
            "test_dataset_manifest_digest": test_dataset.manifest_digest,
            "metrics": metrics_payload,
            "prediction_table_digests": prediction_table_digests,
            "source_isolation": isolation,
            "code_commit": args.code_commit,
        }
        semantic_digest = hashlib.sha256(_canonical_json(semantic_evidence)).hexdigest()
        manifest = {
            "artifact_layout_version": ARTIFACT_LAYOUT_VERSION,
            "mode": "toy",
            "schema_version": SCHEMA_VERSION,
            "channel_spec": frozen_channel_spec(),
            "config_digest": config_digest,
            "train_dataset_manifest_digest": train_dataset.manifest_digest,
            "validation_dataset_manifest_digest": validation_dataset.manifest_digest,
            "calibration_dataset_manifest_digest": calibration_dataset.manifest_digest,
            "test_dataset_manifest_digest": test_dataset.manifest_digest,
            "train_sample_count": len(train_dataset.samples),
            "validation_sample_count": len(validation_dataset.samples),
            "calibration_sample_count": len(calibration_dataset.samples),
            "test_sample_count": len(test_dataset.samples),
            "selection_split": "val",
            "test_samples_used_for_training_or_selection": 0,
            "test_prediction_rows_generated": len(test_dataset.samples),
            "source_isolation": isolation,
            "model_variants": list(config["variants"]),
            "trajectory_ablation_sensitivity": trajectory_ablation,
            "prediction_table_digests": prediction_table_digests,
            "code_commit": args.code_commit,
            "real_data_status": "not_evaluated_real_data",
            "semantic_digest_sha256": semantic_digest,
        }
        _write_json(staging / "manifest.json", manifest)
        artifact_names = (
            "config_snapshot.json",
            "r0_checkpoint.pt",
            "r1_checkpoint.pt",
            "metrics.json",
            "manifest.json",
            "r0_calibration_prediction_table.json",
            "r0_test_prediction_table.json",
            "r1_calibration_prediction_table.json",
            "r1_test_prediction_table.json",
        )
        _write_json(
            staging / "checksums.json",
            {"sha256": {name: _sha256(staging / name) for name in artifact_names}},
        )
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"output_dir={args.output_dir}")
    print(f"semantic_digest_sha256={semantic_digest}")
    print(f"train_sample_count={len(train_dataset.samples)}")
    print(f"validation_sample_count={len(validation_dataset.samples)}")
    print(f"calibration_sample_count={len(calibration_dataset.samples)}")
    print(f"test_sample_count={len(test_dataset.samples)}")
    print("model_variants=r0,r1")
    print("real_data_status=not_evaluated_real_data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
