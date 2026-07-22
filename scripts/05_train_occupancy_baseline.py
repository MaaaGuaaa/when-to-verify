#!/usr/bin/env python
"""Train and publish deterministic toy or authenticated production SOP08 baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.contracts import INPUT_CHANNELS, SCHEMA_VERSION  # noqa: E402
from src.calibration.split_conformal import (  # noqa: E402
    PREDICTION_TABLE_LAYOUT_VERSION,
    prediction_table_cohort_digest,
    prediction_table_semantic_digest,
    validate_prediction_table,
)
from src.datasets.toy_risk_learning import (  # noqa: E402
    TOY_DATASET_LAYOUT_VERSION,
    assert_toy_split_isolation,
    frozen_channel_spec,
    make_toy_risk_dataset,
)
from src.datasets.risk_dataloader import (  # noqa: E402
    select_production_risk_subset,
)
from src.datasets.risk_dataset_seal import (  # noqa: E402
    load_risk_dataset_family,
    load_risk_dataset_seal,
)
from src.datasets.risk_training_store import (  # noqa: E402
    load_authenticated_occupancy_snapshot,
)
from src.evaluation.risk_baselines import (  # noqa: E402
    BASELINE_SPECS,
    OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
    ProductionOccupancyContractUnavailable,
    build_occupancy_checkpoint,
    collate_occupancy_toy_dataset,
    fit_toy_learned_aggregator,
    fit_toy_occupancy_model,
    hand_trajectory_risk_scores,
    occupancy_binary_metrics,
    save_occupancy_checkpoint,
    validate_occupancy_dataset_manifest,
)
from src.models.occupancy_baseline import (  # noqa: E402
    AgeDecay,
    ConvGRUOccupancyPredictor,
    LastObservationHold,
    LearnedOccupancyRiskAggregator,
)
from src.training.occupancy_trainer import (  # noqa: E402
    ProductionOccupancyTrainingConfig,
    train_production_occupancy_baselines,
)
from src.training.distributed import discover_distributed_runtime  # noqa: E402


_CONFIG_SCHEMA: dict[str, Any] = {
    "schema_version": None,
    "mode": None,
    "seed": None,
    "data": {
        "split": None,
        "toy_count": None,
        "validation_count": None,
        "calibration_count": None,
        "test_count": None,
        "grid_size": None,
        "history_steps": None,
        "future_steps": None,
        "future_dt_s": None,
    },
    "model": {
        "hidden_channels": None,
        "convgru_kernel_size": None,
        "learned_aggregator_hidden_dim": None,
    },
    "training": {
        "occupancy_steps": None,
        "occupancy_learning_rate": None,
        "aggregator_steps": None,
        "aggregator_learning_rate": None,
    },
    "aggregation": {
        "b2_tau_s": None,
        "b2_a_max_s": None,
        "sigma_time_s": None,
        "occupancy_threshold": None,
    },
    "artifact": {
        "checkpoint_layout_version": None,
        "calibration_status": None,
    },
}
_PRODUCTION_CONFIG_KEYS = frozenset(
    {
        "mode",
        "stage",
        "seed",
        "device",
        "hidden_channels",
        "convgru_kernel_size",
        "learned_aggregator_hidden_dim",
        "max_samples",
        "batch_size",
        "occupancy_epochs",
        "aggregator_epochs",
        "gradient_accumulation_steps",
        "occupancy_learning_rate",
        "aggregator_learning_rate",
        "weight_decay",
        "checkpoint_interval_steps",
        "b2_tau_s",
        "b2_a_max_s",
        "sigma_time_s",
        "optimizer",
    }
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _validate_known_keys(value: Any, schema: Any, path: str = "") -> None:
    if not isinstance(schema, dict):
        return
    if not isinstance(value, dict):
        raise ValueError(f"config {path or '/'} must be a mapping")
    missing = sorted(set(schema).difference(value))
    unknown = sorted(set(value).difference(schema))
    if missing:
        raise ValueError(f"config {path or '/'} missing keys: {missing}")
    if unknown:
        raise ValueError(f"config {path or '/'} has unknown keys: {unknown}")
    for key, child_schema in schema.items():
        _validate_known_keys(value[key], child_schema, f"{path}.{key}" if path else key)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _odd_positive_int(value: Any, name: str) -> int:
    parsed = _positive_int(value, name)
    if parsed % 2 == 0:
        raise ValueError(f"{name} must be odd")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return parsed


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    _validate_known_keys(value, _CONFIG_SCHEMA)
    assert isinstance(value, dict)
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if value["mode"] not in {"toy", "production"}:
        raise ValueError("config mode must be toy or production")
    if value["data"]["split"] != "train":
        raise ValueError("data.split must be 'train'")
    if value["data"]["history_steps"] != 8:
        raise ValueError("data.history_steps must be 8")
    if value["data"]["future_steps"] != 15:
        raise ValueError("data.future_steps must be 15")
    if not math.isclose(float(value["data"]["future_dt_s"]), 0.2, abs_tol=1e-9):
        raise ValueError("data.future_dt_s must be 0.2")
    if value["artifact"]["checkpoint_layout_version"] != OCCUPANCY_CHECKPOINT_LAYOUT_VERSION:
        raise ValueError(
            f"artifact.checkpoint_layout_version must be {OCCUPANCY_CHECKPOINT_LAYOUT_VERSION}"
        )
    _nonnegative_int(value["seed"], "seed")
    _positive_int(value["data"]["toy_count"], "data.toy_count")
    _positive_int(value["data"]["validation_count"], "data.validation_count")
    _positive_int(value["data"]["calibration_count"], "data.calibration_count")
    _positive_int(value["data"]["test_count"], "data.test_count")
    _positive_int(value["data"]["grid_size"], "data.grid_size")
    _positive_int(value["model"]["hidden_channels"], "model.hidden_channels")
    _odd_positive_int(
        value["model"]["convgru_kernel_size"],
        "model.convgru_kernel_size",
    )
    _positive_int(
        value["model"]["learned_aggregator_hidden_dim"],
        "model.learned_aggregator_hidden_dim",
    )
    for key in ("occupancy_steps", "aggregator_steps"):
        _positive_int(value["training"][key], f"training.{key}")
    for key in ("occupancy_learning_rate", "aggregator_learning_rate"):
        _positive_float(value["training"][key], f"training.{key}")
    _positive_float(value["aggregation"]["b2_tau_s"], "aggregation.b2_tau_s")
    _positive_float(value["aggregation"]["b2_a_max_s"], "aggregation.b2_a_max_s")
    _positive_float(value["aggregation"]["sigma_time_s"], "aggregation.sigma_time_s")
    threshold = float(value["aggregation"]["occupancy_threshold"])
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("aggregation.occupancy_threshold must be in [0,1]")
    return value


def _load_production_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("production config must be a mapping")
    missing = sorted(_PRODUCTION_CONFIG_KEYS.difference(value))
    unknown = sorted(set(value).difference(_PRODUCTION_CONFIG_KEYS))
    if missing:
        raise ValueError(f"production config missing keys: {missing}")
    if unknown:
        raise ValueError(f"production config has unknown keys: {unknown}")
    if value["mode"] != "production":
        raise ValueError("production config mode must be production")
    if value["optimizer"] != "AdamW":
        raise ValueError("production optimizer must be AdamW")
    _positive_int(value["max_samples"], "max_samples")
    ProductionOccupancyTrainingConfig(
        **{
            key: value[key]
            for key in _PRODUCTION_CONFIG_KEYS
            if key not in {"mode", "max_samples", "optimizer"}
        }
    )
    return dict(value)


def _run_production(args: argparse.Namespace) -> dict[str, object]:
    for field, option in (
        ("dataset_seal_root", "--dataset-seal-root"),
        ("risk_collection_root", "--risk-collection-root"),
        ("sidecar_collection_root", "--sidecar-collection-root"),
    ):
        if getattr(args, field) is None:
            raise ValueError(f"{option} is required in production mode")
    config = _load_production_config(args.config)
    if args.stage is not None:
        config["stage"] = args.stage
        if args.stage == "one_shard_smoke":
            config["occupancy_epochs"] = 1
            config["aggregator_epochs"] = 1
            config["gradient_accumulation_steps"] = 1
            config["checkpoint_interval_steps"] = 1
    if args.seed is not None:
        config["seed"] = _nonnegative_int(args.seed, "--seed")
    if args.max_samples is not None:
        config["max_samples"] = _positive_int(args.max_samples, "--max-samples")
    if args.batch_size is not None:
        config["batch_size"] = _positive_int(args.batch_size, "--batch-size")
    if args.device is not None:
        config["device"] = args.device
    training_config = ProductionOccupancyTrainingConfig(
        **{
            key: config[key]
            for key in _PRODUCTION_CONFIG_KEYS
            if key not in {"mode", "max_samples", "optimizer"}
        }
    )
    runtime = discover_distributed_runtime(training_config.device)
    if runtime.is_distributed:
        raise ValueError(
            "production occupancy training does not support WORLD_SIZE>1"
        )
    validation_paths = (
        args.validation_dataset_seal_root,
        args.validation_risk_collection_root,
        args.validation_sidecar_collection_root,
    )
    if any(value is None for value in validation_paths) and any(
        value is not None for value in validation_paths
    ):
        raise ValueError(
            "validation dataset seal, risk collection, and sidecar collection "
            "roots must be supplied together"
        )
    formal = training_config.stage == "formal_50k"
    if formal:
        if args.dataset_family_root is None or any(
            value is None for value in validation_paths
        ):
            raise ValueError(
                "formal_50k requires --dataset-family-root and all validation roots"
            )
        if args.training_cache_mode != "authenticated_snapshot":
            raise ValueError(
                "formal_50k requires --training-cache-mode authenticated_snapshot"
            )
    elif args.dataset_family_root is not None or any(
        value is not None for value in validation_paths
    ):
        raise ValueError(
            "smoke/overfit occupancy training rejects validation and family inputs"
        )

    dataset_family = (
        load_risk_dataset_family(args.dataset_family_root) if formal else None
    )
    expected_train_digest = (
        str(dataset_family.members["train"]["risk_dataset_manifest_digest"])
        if dataset_family is not None
        else None
    )
    expected_validation_digest = (
        str(dataset_family.members["val"]["risk_dataset_manifest_digest"])
        if dataset_family is not None
        else None
    )
    training_snapshot = None
    validation_dataset = None
    validation_snapshot = None
    if args.training_cache_mode == "authenticated_snapshot":
        if args.training_cache_root is None:
            raise ValueError(
                "authenticated_snapshot requires --training-cache-root"
            )
        dataset, training_snapshot = load_authenticated_occupancy_snapshot(
            args.dataset_seal_root,
            collection_root=args.risk_collection_root,
            sidecar_root=args.sidecar_collection_root,
            expected_split="train",
            cache_root=args.training_cache_root,
            expected_manifest_digest=expected_train_digest,
        )
        subset = training_snapshot.select_subset(
            max_samples=int(config["max_samples"]),
            seed=training_config.seed,
        )
        if formal:
            assert validation_paths[0] is not None
            assert validation_paths[1] is not None
            assert validation_paths[2] is not None
            validation_dataset, validation_snapshot = (
                load_authenticated_occupancy_snapshot(
                    validation_paths[0],
                    collection_root=validation_paths[1],
                    sidecar_root=validation_paths[2],
                    expected_split="val",
                    cache_root=args.training_cache_root,
                    expected_manifest_digest=expected_validation_digest,
                )
            )
    else:
        if args.training_cache_root is not None:
            raise ValueError(
                "--training-cache-root requires authenticated_snapshot mode"
            )
        dataset = load_risk_dataset_seal(
            args.dataset_seal_root,
            collection_root=args.risk_collection_root,
            expected_split="train",
            sidecar_root=args.sidecar_collection_root,
        )
        subset = select_production_risk_subset(
            dataset,
            max_samples=int(config["max_samples"]),
            seed=training_config.seed,
        )
    result = train_production_occupancy_baselines(
        train_dataset=dataset,
        train_subset=subset,
        sidecar_root=args.sidecar_collection_root,
        config=training_config,
        output_dir=args.output_dir,
        code_commit=args.code_commit,
        resume_from=args.resume_from,
        resume_expected_publication_instance_digest_sha256=(
            args.resume_publication_instance_digest
        ),
        training_snapshot=training_snapshot,
        validation_dataset=validation_dataset,
        validation_sidecar_root=args.validation_sidecar_collection_root,
        dataset_family=dataset_family,
        validation_snapshot=validation_snapshot,
    )
    return {
        "artifact": str(result.output_dir),
        "semantic_digest_sha256": result.semantic_digest_sha256,
        "publication_instance_digest_sha256": (
            result.publication_instance_digest_sha256
        ),
        "risk_dataset_manifest_digest": dataset.risk_dataset_manifest_digest,
        "scientific_gate": (
            "formal_validation_selected"
            if formal
            else "engineering_only_without_validation_family"
        ),
    }


def _risk_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    probability64 = np.asarray(probability, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    if probability64.shape != target64.shape or probability64.ndim != 1:
        raise ValueError("risk probability and target must have equal [B] shape")
    if not np.isfinite(probability64).all() or np.any((probability64 < 0) | (probability64 > 1)):
        raise ValueError("risk probability must be finite and in [0,1]")
    clipped = np.clip(probability64, 1e-7, 1.0 - 1e-7)
    return {
        "brier": float(np.mean((probability64 - target64) ** 2)),
        "binary_cross_entropy": float(
            -np.mean(target64 * np.log(clipped) + (1.0 - target64) * np.log(1.0 - clipped))
        ),
    }


def _array_semantic_digest(
    *,
    config: dict[str, Any],
    dataset_digest: str,
    predictions: dict[str, np.ndarray],
    checkpoint_semantic_digest: str,
    prediction_table_digests: dict[str, dict[str, str]],
    code_commit: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(config))
    digest.update(dataset_digest.encode("ascii"))
    digest.update(checkpoint_semantic_digest.encode("ascii"))
    digest.update(_canonical_json_bytes(prediction_table_digests))
    digest.update(_canonical_json_bytes({"code_commit": code_commit}))
    for name in sorted(predictions):
        value = np.ascontiguousarray(predictions[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _baseline_scores_for_dataset(
    *,
    batch: dict[str, object],
    model: ConvGRUOccupancyPredictor,
    learned_aggregator: LearnedOccupancyRiskAggregator,
    b2_tau_s: float,
    b2_a_max_s: float,
    sigma_time_s: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Predict B1--B4 scores without routing labels into either model."""
    model_inputs = batch["model_inputs"]
    sidecars = batch["label_sidecars"]
    assert isinstance(model_inputs, dict)
    assert isinstance(sidecars, dict)
    history = np.asarray(model_inputs["bev_history"], dtype=np.float32)
    state = np.asarray(model_inputs["state_channels"], dtype=np.float32)
    footprints = np.asarray(sidecars["robot_future_footprints"], dtype=np.float32)
    b1_occupancy = LastObservationHold(future_steps=15)(history)
    b2_occupancy = AgeDecay(
        future_steps=15,
        dt_s=0.2,
        tau_s=b2_tau_s,
        a_max_s=b2_a_max_s,
    )(state)
    with torch.no_grad():
        history_tensor = torch.from_numpy(history)
        footprint_tensor = torch.from_numpy(footprints)
        b3_tensor = model(history_tensor)
        b4_score = learned_aggregator(b3_tensor, footprint_tensor)
    b3_occupancy = b3_tensor.cpu().numpy().astype(np.float32, copy=False)
    b1_scores = hand_trajectory_risk_scores(
        b1_occupancy,
        footprints,
        sigma_time_s=sigma_time_s,
    )
    b2_scores = hand_trajectory_risk_scores(
        b2_occupancy,
        footprints,
        sigma_time_s=sigma_time_s,
    )
    b3_scores = hand_trajectory_risk_scores(
        b3_occupancy,
        footprints,
        sigma_time_s=sigma_time_s,
    )
    scores = {
        "B1": np.asarray(b1_scores["normalized_weighted_sum"], dtype=np.float32),
        "B2": np.asarray(b2_scores["normalized_weighted_sum"], dtype=np.float32),
        "B3": np.asarray(b3_scores["normalized_weighted_sum"], dtype=np.float32),
        "B4": b4_score.cpu().numpy().astype(np.float32, copy=False),
    }
    return scores, batch


def _build_prediction_table(
    *,
    dataset: Any,
    batch: dict[str, Any],
    method_id: str,
    score: np.ndarray,
    checkpoint_digest: str,
    config_digest: str,
    seed: int,
) -> dict[str, Any]:
    rows = batch["manifest_rows"]
    if len(rows) != int(score.shape[0]):
        raise ValueError("prediction score count does not match manifest rows")
    prediction_rows: list[dict[str, Any]] = []
    for source_row, value in zip(rows, score):
        scalar = float(value)
        row = dict(source_row)
        row.update(
            {
                "p_collision": scalar,
                "q50": scalar,
                "q80": scalar,
                "q90": scalar,
                "q95": scalar,
            }
        )
        prediction_rows.append(row)
    table: dict[str, Any] = {
        "prediction_table_layout_version": PREDICTION_TABLE_LAYOUT_VERSION,
        "mode": "toy",
        "schema_version": SCHEMA_VERSION,
        "split": dataset.split,
        "method_id": method_id,
        "checkpoint_layout_version": OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
        "checkpoint_digest": checkpoint_digest,
        "checkpoint_digest_kind": "occupancy_checkpoint_semantic_sha256",
        "toy_dataset_manifest_digest": dataset.manifest_digest,
        "seed": seed,
        "channel_spec": frozen_channel_spec(),
        "config_digest_sha256": config_digest,
        "score_definition": (
            "normalized_weighted_sum" if method_id in {"B1", "B2", "B3"}
            else "learned_occupancy_aggregator"
        ),
        "quantile_proxy_policy": "q50=q80=q90=q95=raw_score_before_conformal",
        "prediction_semantics": (
            "scalar_baseline_score_repeated_for_common_calibration"
        ),
        "rows": prediction_rows,
    }
    table["cohort_digest_sha256"] = prediction_table_cohort_digest(table)
    table["semantic_digest"] = prediction_table_semantic_digest(table)
    return validate_prediction_table(
        table,
        expected_mode="toy",
        expected_split=dataset.split,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _publish_toy_artifact(
    *,
    config: dict[str, Any],
    output_dir: Path,
    code_commit: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale staging directory exists: {staging}")

    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    split_counts = {
        "train": int(config["data"]["toy_count"]),
        "val": int(config["data"]["validation_count"]),
        "calibration": int(config["data"]["calibration_count"]),
        "test": int(config["data"]["test_count"]),
    }
    datasets = {
        split: make_toy_risk_dataset(
            split=split,
            count=count,
            seed=seed,
            grid_size=int(config["data"]["grid_size"]),
        )
        for split, count in split_counts.items()
    }
    split_isolation = assert_toy_split_isolation(datasets.values())
    split_batches = {
        split: collate_occupancy_toy_dataset(dataset)
        for split, dataset in datasets.items()
    }
    dataset = datasets["train"]
    batch = split_batches["train"]
    model_inputs = batch["model_inputs"]
    label_sidecars = batch["label_sidecars"]
    labels = batch["labels"]
    assert isinstance(model_inputs, dict)
    assert isinstance(label_sidecars, dict)
    assert isinstance(labels, dict)

    history_np = np.asarray(model_inputs["bev_history"], dtype=np.float32)
    state_np = np.asarray(model_inputs["state_channels"], dtype=np.float32)
    target_np = np.asarray(label_sidecars["hidden_risk_occupancy"], dtype=np.float32)
    footprints_np = np.asarray(label_sidecars["robot_future_footprints"], dtype=np.float32)
    collision_np = np.asarray(labels["collision_label"], dtype=np.float32)
    history = torch.from_numpy(history_np)
    target = torch.from_numpy(target_np)
    footprints = torch.from_numpy(footprints_np)
    collision = torch.from_numpy(collision_np)

    model = ConvGRUOccupancyPredictor(
        hidden_channels=int(config["model"]["hidden_channels"]),
        future_steps=15,
        kernel_size=int(config["model"]["convgru_kernel_size"]),
    )
    aggregator = LearnedOccupancyRiskAggregator(
        future_steps=15,
        hidden_dim=int(config["model"]["learned_aggregator_hidden_dim"]),
    )
    occupancy_training = fit_toy_occupancy_model(
        model,
        history,
        target,
        steps=int(config["training"]["occupancy_steps"]),
        learning_rate=float(config["training"]["occupancy_learning_rate"]),
    )
    with torch.no_grad():
        learned_occupancy_torch = model(history)
    aggregator_training = fit_toy_learned_aggregator(
        aggregator,
        learned_occupancy_torch.detach(),
        footprints,
        collision,
        steps=int(config["training"]["aggregator_steps"]),
        learning_rate=float(config["training"]["aggregator_learning_rate"]),
    )

    last_observation = LastObservationHold(future_steps=15)(history_np)
    b2_tau_s = float(config["aggregation"]["b2_tau_s"])
    b2_a_max_s = float(config["aggregation"]["b2_a_max_s"])
    age_decay = AgeDecay(
        future_steps=15,
        dt_s=0.2,
        tau_s=b2_tau_s,
        a_max_s=b2_a_max_s,
    )(state_np)
    learned_occupancy = learned_occupancy_torch.detach().cpu().numpy().astype(np.float32)
    with torch.no_grad():
        learned_risk = (
            aggregator(learned_occupancy_torch, footprints)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    sigma_time_s = float(config["aggregation"]["sigma_time_s"])
    hand_scores = {
        "B1": hand_trajectory_risk_scores(
            last_observation,
            footprints_np,
            sigma_time_s=sigma_time_s,
        ),
        "B2": hand_trajectory_risk_scores(
            age_decay,
            footprints_np,
            sigma_time_s=sigma_time_s,
        ),
        "B3": hand_trajectory_risk_scores(
            learned_occupancy,
            footprints_np,
            sigma_time_s=sigma_time_s,
        ),
    }
    predictions: dict[str, np.ndarray] = {
        "sample_ids": np.asarray(batch["sample_ids"], dtype="U64"),
        "B1_occupancy": last_observation,
        "B2_occupancy": age_decay,
        "B3_occupancy": learned_occupancy,
        "B1_weighted_risk": np.asarray(
            hand_scores["B1"]["normalized_weighted_sum"], dtype=np.float32
        ),
        "B1_union_risk": np.asarray(hand_scores["B1"]["probabilistic_union"], dtype=np.float32),
        "B2_weighted_risk": np.asarray(
            hand_scores["B2"]["normalized_weighted_sum"], dtype=np.float32
        ),
        "B2_union_risk": np.asarray(hand_scores["B2"]["probabilistic_union"], dtype=np.float32),
        "B3_weighted_risk": np.asarray(
            hand_scores["B3"]["normalized_weighted_sum"], dtype=np.float32
        ),
        "B3_union_risk": np.asarray(hand_scores["B3"]["probabilistic_union"], dtype=np.float32),
        "B4_learned_risk": learned_risk,
        "collision_label": collision_np,
    }
    threshold = float(config["aggregation"]["occupancy_threshold"])
    metrics = {
        "mode": "toy",
        "scientific_gate": "not_evaluated_real_data",
        "evaluation_scope": "train_split_fit_diagnostics_only",
        "evaluated_split": dataset.split,
        "occupancy": {
            "B1": occupancy_binary_metrics(last_observation, target_np, threshold=threshold),
            "B2": occupancy_binary_metrics(age_decay, target_np, threshold=threshold),
            "B3": occupancy_binary_metrics(learned_occupancy, target_np, threshold=threshold),
        },
        "trajectory_risk": {
            "B1_weighted": _risk_metrics(predictions["B1_weighted_risk"], collision_np),
            "B1_union": _risk_metrics(predictions["B1_union_risk"], collision_np),
            "B2_weighted": _risk_metrics(predictions["B2_weighted_risk"], collision_np),
            "B2_union": _risk_metrics(predictions["B2_union_risk"], collision_np),
            "B3_weighted": _risk_metrics(predictions["B3_weighted_risk"], collision_np),
            "B3_union": _risk_metrics(predictions["B3_union_risk"], collision_np),
            "B4_learned": _risk_metrics(predictions["B4_learned_risk"], collision_np),
        },
        "training": {
            "occupancy": occupancy_training,
            "learned_aggregator": aggregator_training,
        },
    }
    config_digest = hashlib.sha256(_canonical_json_bytes(config)).hexdigest()
    checkpoint = build_occupancy_checkpoint(
        model=model,
        learned_aggregator=aggregator,
        toy_dataset_manifest_digest=dataset.manifest_digest,
        config_digest=config_digest,
        seed=seed,
    )
    checkpoint_state_digest = str(checkpoint["model_state_digest_sha256"])
    checkpoint_digest = str(checkpoint["checkpoint_semantic_digest_sha256"])
    prediction_table_payloads: dict[str, dict[str, Any]] = {}
    prediction_table_digests: dict[str, dict[str, str]] = {}
    prediction_table_manifest: dict[str, dict[str, dict[str, Any]]] = {}
    for split in ("val", "calibration", "test"):
        split_dataset = datasets[split]
        split_scores, split_batch = _baseline_scores_for_dataset(
            batch=split_batches[split],
            model=model,
            learned_aggregator=aggregator,
            b2_tau_s=b2_tau_s,
            b2_a_max_s=b2_a_max_s,
            sigma_time_s=sigma_time_s,
        )
        prediction_table_digests[split] = {}
        prediction_table_manifest[split] = {}
        for method_id in BASELINE_SPECS:
            filename = f"prediction_table_{split}_{method_id}.json"
            table = _build_prediction_table(
                dataset=split_dataset,
                batch=split_batch,
                method_id=method_id,
                score=split_scores[method_id],
                checkpoint_digest=checkpoint_digest,
                config_digest=config_digest,
                seed=seed,
            )
            prediction_table_payloads[filename] = table
            prediction_table_digests[split][method_id] = str(table["semantic_digest"])
            prediction_table_manifest[split][method_id] = {
                "path": filename,
                "semantic_digest_sha256": str(table["semantic_digest"]),
                "toy_dataset_manifest_digest": split_dataset.manifest_digest,
                "sample_count": len(table["rows"]),
            }
    semantic_digest = _array_semantic_digest(
        config=config,
        dataset_digest=dataset.manifest_digest,
        predictions=predictions,
        checkpoint_semantic_digest=checkpoint_digest,
        prediction_table_digests=prediction_table_digests,
        code_commit=code_commit,
    )
    manifest = {
        "artifact_layout_version": "sop08_occupancy_baselines_v1",
        "mode": "toy",
        "schema_version": SCHEMA_VERSION,
        "channel_spec": list(INPUT_CHANNELS),
        "model_input_keys": sorted(model_inputs),
        "label_sidecar_keys": sorted(label_sidecars),
        "toy_dataset_layout_version": TOY_DATASET_LAYOUT_VERSION,
        "toy_dataset_manifest_digest": dataset.manifest_digest,
        "split_dataset_manifest_digests": {
            split: split_dataset.manifest_digest
            for split, split_dataset in datasets.items()
        },
        "split_collation_provenance": {
            split: dict(split_batch["strict_provenance"])
            for split, split_batch in split_batches.items()
        },
        "split_isolation": split_isolation,
        "config_digest_sha256": config_digest,
        "semantic_digest_sha256": semantic_digest,
        "checkpoint_layout_version": OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
        "checkpoint_semantic_digest_sha256": checkpoint_digest,
        "checkpoint_model_state_digest_sha256": checkpoint_state_digest,
        "baselines": BASELINE_SPECS,
        "sample_count": int(history_np.shape[0]),
        "split": dataset.split,
        "seed": seed,
        "future_steps": 15,
        "future_dt_s": 0.2,
        "future_time_layout": "endpoint_dt_to_horizon",
        "future_endpoint_times_s": [round(step * 0.2, 7) for step in range(1, 16)],
        "baseline_hyperparameters": {
            "b2_tau_s": b2_tau_s,
            "b2_a_max_s": b2_a_max_s,
            "convgru_kernel_size": int(config["model"]["convgru_kernel_size"]),
        },
        "aggregation": {
            "normalized_weighted_sum": (
                "sum(p*mask*exp(-endpoint_time/sigma))"
                "/sum(mask*exp(-endpoint_time/sigma))"
            ),
            "probabilistic_union": "1-product(1-p) over unique selected batch/time/cell tuples",
            "mask_normalization": "strictly-positive values become one boolean selection",
        },
        "prediction_tables": prediction_table_manifest,
        "calibration_status": config["artifact"]["calibration_status"],
        "scientific_gate": "not_evaluated_real_data",
        "evaluation_scope": "train_split_fit_diagnostics_only",
        "code_commit": code_commit,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }

    try:
        staging.mkdir(parents=True)
        _write_json(staging / "config_snapshot.json", config)
        _write_json(staging / "metrics.json", metrics)
        _write_json(staging / "manifest.json", manifest)
        np.savez_compressed(staging / "occupancy_predictions.npz", **predictions)
        save_occupancy_checkpoint(staging / "checkpoint.pt", checkpoint)
        for filename, table in sorted(prediction_table_payloads.items()):
            _write_json(staging / filename, table)
        checksums = {
            path.name: _sha256_file(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        (staging / "checksums.sha256").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("toy", "production"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--toy-count", type=int)
    parser.add_argument("--validation-count", type=int)
    parser.add_argument("--calibration-count", type=int)
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--training-steps", type=int)
    parser.add_argument("--aggregator-steps", type=int)
    parser.add_argument("--b2-tau-s", type=float)
    parser.add_argument("--b2-a-max-s", type=float)
    parser.add_argument("--convgru-kernel-size", type=int)
    parser.add_argument("--code-commit", default="unversioned")
    parser.add_argument("--dataset-seal-root", type=Path)
    parser.add_argument("--risk-collection-root", type=Path)
    parser.add_argument("--sidecar-collection-root", type=Path)
    parser.add_argument("--validation-dataset-seal-root", type=Path)
    parser.add_argument("--validation-risk-collection-root", type=Path)
    parser.add_argument("--validation-sidecar-collection-root", type=Path)
    parser.add_argument("--dataset-family-root", type=Path)
    parser.add_argument(
        "--stage",
        choices=("one_shard_smoke", "real_1k_overfit", "formal_50k"),
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-publication-instance-digest")
    parser.add_argument(
        "--training-cache-mode",
        choices=("strict", "authenticated_snapshot"),
        default="strict",
    )
    parser.add_argument("--training-cache-root", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.mode == "production":
            report = _run_production(args)
            print(f"artifact={report['artifact']}")
            print(f"semantic_digest_sha256={report['semantic_digest_sha256']}")
            print(
                "publication_instance_digest_sha256="
                f"{report['publication_instance_digest_sha256']}"
            )
            print(
                "risk_dataset_manifest_digest="
                f"{report['risk_dataset_manifest_digest']}"
            )
            print(f"scientific_gate={report['scientific_gate']}")
            return 0
        config = _load_config(args.config)
        if args.mode is not None:
            config["mode"] = args.mode
        if args.seed is not None:
            config["seed"] = _nonnegative_int(args.seed, "--seed")
        if args.toy_count is not None:
            config["data"]["toy_count"] = _positive_int(args.toy_count, "--toy-count")
        if args.validation_count is not None:
            config["data"]["validation_count"] = _positive_int(
                args.validation_count,
                "--validation-count",
            )
        if args.calibration_count is not None:
            config["data"]["calibration_count"] = _positive_int(
                args.calibration_count,
                "--calibration-count",
            )
        if args.test_count is not None:
            config["data"]["test_count"] = _positive_int(
                args.test_count,
                "--test-count",
            )
        if args.grid_size is not None:
            config["data"]["grid_size"] = _positive_int(args.grid_size, "--grid-size")
        if args.training_steps is not None:
            config["training"]["occupancy_steps"] = _positive_int(
                args.training_steps,
                "--training-steps",
            )
        if args.aggregator_steps is not None:
            config["training"]["aggregator_steps"] = _positive_int(
                args.aggregator_steps,
                "--aggregator-steps",
            )
        if args.b2_tau_s is not None:
            config["aggregation"]["b2_tau_s"] = _positive_float(
                args.b2_tau_s,
                "--b2-tau-s",
            )
        if args.b2_a_max_s is not None:
            config["aggregation"]["b2_a_max_s"] = _positive_float(
                args.b2_a_max_s,
                "--b2-a-max-s",
            )
        if args.convgru_kernel_size is not None:
            config["model"]["convgru_kernel_size"] = _odd_positive_int(
                args.convgru_kernel_size,
                "--convgru-kernel-size",
            )
        if config["mode"] == "production":
            raise ValueError("production mode must be selected with --mode production")
        manifest = _publish_toy_artifact(
            config=config,
            output_dir=args.output_dir,
            code_commit=args.code_commit,
        )
    except (
        ValueError,
        FileExistsError,
        OSError,
        ProductionOccupancyContractUnavailable,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"artifact={args.output_dir}")
    print(f"semantic_digest_sha256={manifest['semantic_digest_sha256']}")
    print(f"toy_dataset_manifest_digest={manifest['toy_dataset_manifest_digest']}")
    print("scientific_gate=not_evaluated_real_data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
