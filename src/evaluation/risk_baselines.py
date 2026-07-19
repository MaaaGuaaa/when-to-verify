"""Shared SOP08 baseline definitions, validation, training, and metrics."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.contracts import INPUT_CHANNELS, SCHEMA_VERSION
from src.datasets.risk_dataloader import (
    collate_risk_samples,
    validate_toy_dataset_manifest,
)
from src.datasets.toy_risk_learning import (
    TOY_DATASET_LAYOUT_VERSION,
    TOY_DT_S,
    TOY_FUTURE_STEPS,
    TOY_SPLITS,
    ToyRiskDataset,
    make_toy_batch,
    toy_sample_id_sequence_digest,
    validate_toy_risk_dataset_publication,
)
from src.models.occupancy_aggregation import (
    future_endpoint_times,
    probabilistic_union_risk,
    weighted_swept_volume_risk,
)


OCCUPANCY_CHECKPOINT_LAYOUT_VERSION = "occupancy_baseline_checkpoint_v2"
OCCUPANCY_MODEL_VARIANT = "convgru_hidden_occupancy+B4_learned_aggregator"
_CHECKPOINT_SEMANTIC_FIELDS: tuple[str, ...] = (
    "checkpoint_layout_version",
    "mode",
    "schema_version",
    "channel_spec",
    "toy_dataset_manifest_digest",
    "config_digest",
    "seed",
    "model_variant",
    "model_state_digest_sha256",
)
_CHECKPOINT_PRODUCTION_PROVENANCE_FIELDS = frozenset(
    {
        "g1_split_manifest_digest",
        "risk_dataset_manifest_digest",
        "dynamic_objects_config_digest",
        "target_type_policy_digest",
    }
)
_TOY_MANIFEST_FIELDS = frozenset(
    {
        "mode",
        "dataset_layout_version",
        "schema_version",
        "channel_spec",
        "split",
        "sample_count",
        "seed",
        "ordered_sample_ids",
        "ordered_sample_ids_digest_sha256",
        "model_input_digest_sha256",
        "label_digest_sha256",
        "ordered_sample_digest_sha256",
        "manifest_rows_digest_sha256",
        "label_sidecars_digest_sha256",
        "future_endpoint_times_s",
        "grid",
        "toy_dataset_manifest_digest",
    }
)
_CHECKPOINT_TOP_LEVEL_FIELDS = frozenset(
    {
        *_CHECKPOINT_SEMANTIC_FIELDS,
        "checkpoint_semantic_digest_sha256",
        "model_state_dict",
        "learned_aggregator_state_dict",
    }
)
BASELINE_SPECS: dict[str, str] = {
    "B1": "last_observation_hold+hand_aggregation",
    "B2": "age_decay+hand_aggregation",
    "B3": "convgru_occupancy+hand_aggregation",
    "B4": "convgru_occupancy+learned_aggregation",
}


class ProductionOccupancyContractUnavailable(RuntimeError):
    """Raised because the production v2 occupancy-label contract is unpublished."""


def occupancy_checkpoint_semantic_digest(checkpoint: Mapping[str, Any]) -> str:
    """Digest exactly the frozen checkpoint identity and model-state digest.

    The digest field itself and serialized tensor bytes are deliberately not
    projected here. Tensor bytes are covered by ``model_state_digest_sha256``;
    :func:`validate_occupancy_checkpoint_provenance` validates both layers.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    missing = sorted(set(_CHECKPOINT_SEMANTIC_FIELDS).difference(checkpoint))
    if missing:
        raise ValueError(f"checkpoint semantic identity missing fields: {missing}")
    payload = {field: checkpoint[field] for field in _CHECKPOINT_SEMANTIC_FIELDS}
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint semantic identity must be finite JSON data") from error
    return hashlib.sha256(encoded).hexdigest()


def _expected_endpoints() -> np.ndarray:
    return future_endpoint_times(future_steps=15, dt_s=0.2)


def _channel_spec_is_current(value: Any) -> bool:
    if isinstance(value, Mapping):
        flattened: list[str] = []
        for key in ("history", "state", "trajectory"):
            current = value.get(key)
            if not isinstance(current, (list, tuple)):
                return False
            flattened.extend(str(item) for item in current)
        flat = value.get("flat", flattened)
        return tuple(flattened) == INPUT_CHANNELS and tuple(flat) == INPUT_CHANNELS
    return isinstance(value, (list, tuple)) and tuple(value) == INPUT_CHANNELS


def validate_occupancy_dataset_manifest(
    manifest: Mapping[str, Any],
    *,
    mode: str,
    expected_manifest_digest: str,
) -> dict[str, Any]:
    """Validate the supported toy manifest or fail closed for production."""
    if mode == "production":
        raise ProductionOccupancyContractUnavailable(
            "production occupancy training requires a dataset-level v2 manifest "
            "with label-only future occupancy and robot-footprint sidecars"
        )
    if mode != "toy":
        raise ValueError("mode must be 'toy' or 'production'")
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    result = copy.deepcopy(dict(manifest))
    production_fields = sorted(
        _CHECKPOINT_PRODUCTION_PROVENANCE_FIELDS.intersection(result)
    )
    if production_fields:
        raise ValueError(
            "toy manifest must not contain production provenance fields: "
            f"{production_fields}"
        )
    unexpected = sorted(set(result).difference(_TOY_MANIFEST_FIELDS))
    if unexpected:
        raise ValueError(f"toy manifest has unexpected top-level fields: {unexpected}")
    split = result.get("split")
    if split not in TOY_SPLITS:
        raise ValueError(f"toy manifest split must be one of {TOY_SPLITS}")
    validate_toy_dataset_manifest(result, expected_split=str(split))
    if result.get("mode") != "toy":
        raise ValueError("toy manifest must declare mode='toy'")
    if result.get("dataset_layout_version") != TOY_DATASET_LAYOUT_VERSION:
        raise ValueError(f"dataset_layout_version must be {TOY_DATASET_LAYOUT_VERSION}")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if not _channel_spec_is_current(result.get("channel_spec")):
        raise ValueError("channel_spec does not match the frozen ordered input channels")
    grid = result.get("grid")
    if not isinstance(grid, Mapping):
        raise ValueError("toy manifest must contain grid provenance")
    if grid.get("future_steps") != TOY_FUTURE_STEPS:
        raise ValueError(f"grid.future_steps must be {TOY_FUTURE_STEPS}")
    if not math.isclose(float(grid.get("sample_dt_s", -1.0)), TOY_DT_S, abs_tol=1e-9):
        raise ValueError(f"grid.sample_dt_s must be {TOY_DT_S}")
    if grid.get("future_time_layout") != "endpoint_dt_to_horizon":
        raise ValueError("grid.future_time_layout must be endpoint_dt_to_horizon")
    if "future_endpoint_times_s" not in result:
        raise ValueError("toy manifest must record future_endpoint_times_s")
    endpoints = np.asarray(result["future_endpoint_times_s"], dtype=np.float64)
    expected = _expected_endpoints().astype(np.float64)
    if endpoints.shape != expected.shape or not np.allclose(
        endpoints,
        expected,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("future_endpoint_times_s must contain 0.2 .. 3.0 endpoint times")
    actual_digest = result.get("toy_dataset_manifest_digest")
    if not isinstance(actual_digest, str) or actual_digest != expected_manifest_digest:
        raise ValueError("toy_dataset_manifest_digest mismatch")
    for field in (
        "ordered_sample_ids_digest_sha256",
        "model_input_digest_sha256",
        "label_digest_sha256",
        "ordered_sample_digest_sha256",
    ):
        if result[field] == "0" * 64:
            raise ValueError(f"{field} must not be an all-zero placeholder digest")
    return result


def collate_occupancy_toy_dataset(dataset: ToyRiskDataset) -> dict[str, object]:
    """Strictly collate one toy split and bind its ordered label sidecars."""
    if not isinstance(dataset, ToyRiskDataset):
        raise TypeError("dataset must be ToyRiskDataset")
    publication_digests = validate_toy_risk_dataset_publication(dataset)
    validate_occupancy_dataset_manifest(
        dataset.manifest,
        mode="toy",
        expected_manifest_digest=dataset.manifest_digest,
    )
    strict_batch = collate_risk_samples(
        dataset.samples,
        grid=dataset.grid,
        dataset_manifest=dataset.manifest,
        expected_split=dataset.split,
    )
    sidecar_sample_ids = tuple(sidecar.sample_id for sidecar in dataset.sidecars)
    if sidecar_sample_ids != strict_batch.sample_ids:
        raise ValueError(
            "sidecar sample IDs must exactly match the ordered RiskSample IDs"
        )
    batch = make_toy_batch(dataset)
    if tuple(batch["sample_ids"]) != strict_batch.sample_ids:
        raise ValueError("collated sample IDs do not match strict RiskSample IDs")
    label_sidecars = batch["label_sidecars"]
    if not isinstance(label_sidecars, Mapping):
        raise ValueError("label_sidecars must be a mapping")
    if tuple(label_sidecars.get("sample_ids", ())) != strict_batch.sample_ids:
        raise ValueError("sidecar sample IDs were not preserved during collation")
    strict_provenance = dict(strict_batch.provenance)
    strict_provenance.update(publication_digests)
    strict_provenance["sidecar_sample_ids_digest_sha256"] = (
        toy_sample_id_sequence_digest(sidecar_sample_ids)
    )
    batch["strict_provenance"] = strict_provenance
    return batch


def validate_occupancy_checkpoint_provenance(
    checkpoint: Mapping[str, Any],
    *,
    mode: str,
    expected_manifest_digest: str,
    expected_config_digest: str | None = None,
    expected_seed: int | None = None,
    expected_model_variant: str = OCCUPANCY_MODEL_VARIANT,
    expected_model_state_digest: str | None = None,
) -> dict[str, Any]:
    """Validate checkpoint-v2 identity without accepting legacy or cross-mode data."""
    if mode == "production":
        raise ProductionOccupancyContractUnavailable(
            "production occupancy checkpoints require the unpublished dataset-level v2 manifest"
        )
    if mode != "toy":
        raise ValueError("mode must be 'toy' or 'production'")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    result = copy.deepcopy(dict(checkpoint))
    production_fields = sorted(
        _CHECKPOINT_PRODUCTION_PROVENANCE_FIELDS.intersection(result)
    )
    if production_fields:
        raise ValueError(
            "toy checkpoint must not contain production provenance fields: "
            f"{production_fields}"
        )
    missing = sorted(_CHECKPOINT_TOP_LEVEL_FIELDS.difference(result))
    if missing:
        raise ValueError(f"checkpoint missing required top-level fields: {missing}")
    unexpected = sorted(set(result).difference(_CHECKPOINT_TOP_LEVEL_FIELDS))
    if unexpected:
        raise ValueError(f"checkpoint has unexpected top-level fields: {unexpected}")
    if result["checkpoint_layout_version"] != OCCUPANCY_CHECKPOINT_LAYOUT_VERSION:
        raise ValueError("legacy or unsupported occupancy checkpoint layout")
    if result["mode"] != "toy":
        raise ValueError("toy loader rejects non-toy checkpoints")
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(result["channel_spec"], list) or tuple(
        result["channel_spec"]
    ) != INPUT_CHANNELS:
        raise ValueError("checkpoint channel_spec does not match frozen ordering")
    if result["toy_dataset_manifest_digest"] != expected_manifest_digest:
        raise ValueError("checkpoint toy_dataset_manifest_digest mismatch")
    if expected_config_digest is not None and result["config_digest"] != expected_config_digest:
        raise ValueError("checkpoint config_digest mismatch")
    if (
        isinstance(result["seed"], bool)
        or not isinstance(result["seed"], int)
        or result["seed"] < 0
    ):
        raise ValueError("checkpoint seed must be a nonnegative integer")
    if expected_seed is not None and result["seed"] != expected_seed:
        raise ValueError("checkpoint seed mismatch")
    if result["model_variant"] != expected_model_variant:
        raise ValueError("checkpoint model_variant mismatch")
    state_digest = result["model_state_digest_sha256"]
    if (
        not isinstance(state_digest, str)
        or len(state_digest) != 64
        or any(character not in "0123456789abcdef" for character in state_digest)
    ):
        raise ValueError("model_state_digest_sha256 must be a lowercase SHA-256 hex digest")
    if expected_model_state_digest is not None and state_digest != expected_model_state_digest:
        raise ValueError("checkpoint model_state_digest_sha256 mismatch")
    declared_semantic_digest = result["checkpoint_semantic_digest_sha256"]
    actual_semantic_digest = occupancy_checkpoint_semantic_digest(result)
    if declared_semantic_digest != actual_semantic_digest:
        raise ValueError("checkpoint semantic digest mismatch")

    model_state = result["model_state_dict"]
    aggregator_state = result["learned_aggregator_state_dict"]
    if not isinstance(model_state, Mapping) or not isinstance(aggregator_state, Mapping):
        raise ValueError("checkpoint model state dictionaries must be mappings")
    if _state_dict_digest(model_state, aggregator_state) != state_digest:
        raise ValueError("checkpoint model state digest mismatch")
    return result


def _validate_metric_arrays(probability: np.ndarray, target: np.ndarray) -> None:
    if not isinstance(probability, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("probability and target must be NumPy arrays")
    if probability.dtype != np.float32 or target.dtype != np.float32:
        raise ValueError("probability and target must be float32")
    if probability.shape != target.shape:
        raise ValueError("probability and target must have the same shape")
    if probability.ndim != 4:
        raise ValueError("probability and target must have rank 4 [B,T,H,W]")
    if not np.isfinite(probability).all() or np.logical_or(
        probability < 0.0, probability > 1.0
    ).any():
        raise ValueError("probability must be finite and in [0,1]")
    if not np.isfinite(target).all() or not np.logical_or(target == 0.0, target == 1.0).all():
        raise ValueError("target must be a finite binary array")


def occupancy_binary_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return finite occupancy metrics over every batch/time/cell element."""
    _validate_metric_arrays(probability, target)
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be finite and in [0,1]")
    prediction = probability >= float(threshold)
    truth = target > 0.5
    intersection = int(np.logical_and(prediction, truth).sum())
    union = int(np.logical_or(prediction, truth).sum())
    predicted_positive = int(prediction.sum())
    target_positive = int(truth.sum())
    iou = float(intersection / union) if union else 1.0
    precision = (
        float(intersection / predicted_positive)
        if predicted_positive
        else (1.0 if target_positive == 0 else 0.0)
    )
    recall = float(intersection / target_positive) if target_positive else 1.0
    clipped = np.clip(probability.astype(np.float64), 1e-7, 1.0 - 1e-7)
    truth64 = target.astype(np.float64)
    bce = -np.mean(truth64 * np.log(clipped) + (1.0 - truth64) * np.log(1.0 - clipped))
    return {
        "brier": float(np.mean((probability.astype(np.float64) - truth64) ** 2)),
        "binary_cross_entropy": float(bce),
        "intersection_over_union": iou,
        "binary_accuracy": float(np.mean(prediction == truth)),
        "positive_precision": precision,
        "positive_recall": recall,
        "target_positive_fraction": float(np.mean(truth)),
    }


def hand_trajectory_risk_scores(
    occupancy_probability: np.ndarray | torch.Tensor,
    robot_future_footprints: np.ndarray | torch.Tensor,
    *,
    dt_s: float = 0.2,
    sigma_time_s: float = 2.0,
) -> dict[str, np.ndarray | torch.Tensor]:
    """Apply both auditable hand aggregators to one occupancy prediction."""
    return {
        "normalized_weighted_sum": weighted_swept_volume_risk(
            occupancy_probability,
            robot_future_footprints,
            dt_s=dt_s,
            sigma_time_s=sigma_time_s,
        ),
        "probabilistic_union": probabilistic_union_risk(
            occupancy_probability,
            robot_future_footprints,
        ),
    }


def fit_toy_occupancy_model(
    model: nn.Module,
    bev_history: torch.Tensor,
    hidden_risk_occupancy: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Deterministically optimize a bounded toy batch for a smoke-test closure."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    if not torch.is_tensor(hidden_risk_occupancy) or hidden_risk_occupancy.dtype != torch.float32:
        raise ValueError("hidden_risk_occupancy must be a float32 tensor")
    if not bool(torch.isfinite(hidden_risk_occupancy).all()) or bool(
        ((hidden_risk_occupancy < 0.0) | (hidden_risk_occupancy > 1.0)).any()
    ):
        raise ValueError("hidden_risk_occupancy must be finite and in [0,1]")

    model.train()
    positive_count = hidden_risk_occupancy.sum()
    negative_count = hidden_risk_occupancy.numel() - positive_count
    if float(positive_count) <= 0.0 or float(negative_count) <= 0.0:
        raise ValueError("toy occupancy batch must contain both positive and negative cells")
    positive_weight = (negative_count / positive_count).detach()
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    with torch.no_grad():
        initial_loss = float(
            loss_function(model.predict_logits(bev_history), hidden_risk_occupancy)
        )
    loss_history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model.predict_logits(bev_history)
        if tuple(logits.shape) != tuple(hidden_risk_occupancy.shape):
            raise ValueError("model logits and hidden_risk_occupancy must have the same shape")
        loss = loss_function(logits, hidden_risk_occupancy)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        final_loss = float(loss_function(model.predict_logits(bev_history), hidden_risk_occupancy))
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "steps": steps,
        "loss_history": loss_history,
    }


def fit_toy_learned_aggregator(
    aggregator: nn.Module,
    occupancy_probability: torch.Tensor,
    robot_future_footprints: torch.Tensor,
    collision_labels: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Train B4 without exposing future occupancy labels to its forward API."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    if not torch.is_tensor(collision_labels) or collision_labels.ndim != 1:
        raise ValueError("collision_labels must be float32 [B]")
    if collision_labels.dtype != torch.float32 or not bool(torch.isfinite(collision_labels).all()):
        raise ValueError("collision_labels must be finite float32 [B]")
    if bool(((collision_labels < 0.0) | (collision_labels > 1.0)).any()):
        raise ValueError("collision_labels must be in [0,1]")
    if int(occupancy_probability.shape[0]) != int(collision_labels.shape[0]):
        raise ValueError("occupancy_probability and collision_labels batch sizes differ")

    optimizer = torch.optim.Adam(aggregator.parameters(), lr=float(learning_rate))
    loss_fn = nn.BCELoss()
    aggregator.train()
    with torch.no_grad():
        initial = float(
            loss_fn(
                aggregator(occupancy_probability, robot_future_footprints),
                collision_labels,
            )
        )
    loss_history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = aggregator(occupancy_probability, robot_future_footprints)
        loss = loss_fn(prediction, collision_labels)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach()))
    aggregator.eval()
    with torch.no_grad():
        final = float(
            loss_fn(
                aggregator(occupancy_probability, robot_future_footprints),
                collision_labels,
            )
        )
    return {
        "initial_loss": initial,
        "final_loss": final,
        "steps": steps,
        "loss_history": loss_history,
    }


def _state_dict_digest(*state_dicts: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for index, state_dict in enumerate(state_dicts):
        digest.update(f"state_dict:{index}\n".encode("utf-8"))
        for name in sorted(state_dict):
            tensor = state_dict[name].detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_occupancy_checkpoint(
    *,
    model: nn.Module,
    learned_aggregator: nn.Module,
    toy_dataset_manifest_digest: str,
    config_digest: str,
    seed: int,
) -> dict[str, Any]:
    """Build checkpoint-v2 with toy provenance and deterministic state digest."""
    model_state = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    aggregator_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in learned_aggregator.state_dict().items()
    }
    checkpoint: dict[str, Any] = {
        "checkpoint_layout_version": OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
        "mode": "toy",
        "schema_version": SCHEMA_VERSION,
        "channel_spec": list(INPUT_CHANNELS),
        "toy_dataset_manifest_digest": str(toy_dataset_manifest_digest),
        "config_digest": str(config_digest),
        "seed": int(seed),
        "model_variant": OCCUPANCY_MODEL_VARIANT,
        "model_state_dict": model_state,
        "learned_aggregator_state_dict": aggregator_state,
        "model_state_digest_sha256": _state_dict_digest(model_state, aggregator_state),
    }
    checkpoint["checkpoint_semantic_digest_sha256"] = (
        occupancy_checkpoint_semantic_digest(checkpoint)
    )
    validate_occupancy_checkpoint_provenance(
        checkpoint,
        mode="toy",
        expected_manifest_digest=str(toy_dataset_manifest_digest),
        expected_config_digest=str(config_digest),
        expected_seed=int(seed),
        expected_model_state_digest=str(checkpoint["model_state_digest_sha256"]),
    )
    return checkpoint


def save_occupancy_checkpoint(path: str | Path, checkpoint: Mapping[str, Any]) -> Path:
    """Atomically write a new checkpoint without overwriting an existing one."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale checkpoint staging file exists: {temporary}")
    try:
        torch.save(dict(checkpoint), temporary)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_occupancy_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    learned_aggregator: nn.Module,
    mode: str,
    expected_manifest_digest: str,
    expected_config_digest: str,
    expected_seed: int,
) -> dict[str, Any]:
    """Validate and restore checkpoint-v2 into caller-constructed modules."""
    checkpoint = torch.load(Path(path), map_location="cpu")
    validated = validate_occupancy_checkpoint_provenance(
        checkpoint,
        mode=mode,
        expected_manifest_digest=expected_manifest_digest,
        expected_config_digest=expected_config_digest,
        expected_seed=expected_seed,
    )
    model_state = validated.get("model_state_dict")
    aggregator_state = validated.get("learned_aggregator_state_dict")
    if not isinstance(model_state, Mapping) or not isinstance(aggregator_state, Mapping):
        raise ValueError("checkpoint is missing model state dictionaries")
    actual_digest = _state_dict_digest(model_state, aggregator_state)
    if validated.get("model_state_digest_sha256") != actual_digest:
        raise ValueError("checkpoint model state digest mismatch")
    model.load_state_dict(model_state, strict=True)
    learned_aggregator.load_state_dict(aggregator_state, strict=True)
    model.eval()
    learned_aggregator.eval()
    return validated


__all__ = [
    "BASELINE_SPECS",
    "OCCUPANCY_CHECKPOINT_LAYOUT_VERSION",
    "OCCUPANCY_MODEL_VARIANT",
    "ProductionOccupancyContractUnavailable",
    "build_occupancy_checkpoint",
    "collate_occupancy_toy_dataset",
    "fit_toy_learned_aggregator",
    "fit_toy_occupancy_model",
    "hand_trajectory_risk_scores",
    "load_occupancy_checkpoint",
    "occupancy_binary_metrics",
    "occupancy_checkpoint_semantic_digest",
    "save_occupancy_checkpoint",
    "validate_occupancy_checkpoint_provenance",
    "validate_occupancy_dataset_manifest",
]
