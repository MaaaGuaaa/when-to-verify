"""Deterministic group-preserving CPU training for SOP14 verification value."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.contracts import GridSpec, VerificationSample, validate_verification_sample
from src.datasets.verification_dataset import (
    audit_verification_groups,
    verification_model_inputs,
)
from src.evaluation.verification_metrics import (
    evaluate_verification_predictions,
    validate_verification_checkpoint_manifest,
)
from src.models.verification_model import (
    VerificationValueModel,
    VerifyModelConfig,
    verification_loss,
)
from src.planning.verification_actions import CANONICAL_ACTION_IDS
from src.utils.seeding import derive_seed


VERIFICATION_TRAINING_CHECKPOINT_VERSION = "verification_training_checkpoint_v1"
_CHECKPOINT_KEYS = frozenset(
    {
        "checkpoint_version",
        "manifest",
        "model_state_dict",
        "optimizer_state_dict",
        "completed_epochs",
        "loss_history",
    }
)


@dataclass(frozen=True)
class LoadedVerificationTrainingCheckpoint:
    manifest: dict[str, object]
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, object]
    completed_epochs: int
    loss_history: tuple[float, ...]


@dataclass(frozen=True)
class VerificationTrainingResult:
    model: VerificationValueModel
    optimizer_state_dict: dict[str, object]
    completed_epochs: int
    loss_history: tuple[float, ...]
    initial_loss: float
    final_loss: float
    metrics: dict[str, object]
    value_prediction: np.ndarray
    useful_probability: np.ndarray
    device: str


@dataclass(frozen=True)
class VerificationEvaluationResult:
    split: str
    sample_count: int
    group_count: int
    losses: dict[str, float | int]
    metrics: dict[str, object]
    value_prediction: np.ndarray
    useful_probability: np.ndarray
    device: str


def _validated_samples(
    samples: Sequence[VerificationSample], *, grid: GridSpec, expected_split: str
) -> tuple[VerificationSample, ...]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    ordered = tuple(sorted(samples, key=lambda sample: sample.sample_id))
    if not ordered:
        raise ValueError("samples must be non-empty")
    for sample in ordered:
        if not isinstance(sample, VerificationSample):
            raise TypeError("samples must contain VerificationSample values")
        validate_verification_sample(sample, grid)
        if sample.split != expected_split:
            if expected_split == "train":
                raise ValueError("verification model fitting accepts train split only")
            raise ValueError(
                "verification model evaluation requires one declared held-out split"
            )
        verification_model_inputs(sample)
    audit_verification_groups(list(ordered), require_complete=True)
    return ordered


def _groups(
    samples: tuple[VerificationSample, ...],
) -> tuple[tuple[VerificationSample, ...], ...]:
    grouped: dict[str, list[VerificationSample]] = {}
    for sample in samples:
        group_id = str(sample.metadata["ranking_group_id"])
        grouped.setdefault(group_id, []).append(sample)
    action_priority = {
        action_id: index for index, action_id in enumerate(CANONICAL_ACTION_IDS)
    }
    result: list[tuple[VerificationSample, ...]] = []
    for group_id in sorted(grouped):
        rows = tuple(
            sorted(
                grouped[group_id],
                key=lambda sample: action_priority[sample.verification_action_id],
            )
        )
        if len(rows) != len(CANONICAL_ACTION_IDS):
            raise ValueError("training requires complete six-action groups")
        result.append(rows)
    return tuple(result)


def _batch(
    samples: Sequence[VerificationSample], *, device: torch.device
) -> dict[str, object]:
    rows = tuple(samples)
    arrays = {
        name: torch.from_numpy(
            np.stack([getattr(sample, name) for sample in rows], axis=0)
        ).to(device=device, dtype=torch.float32)
        for name in (
            "bev_history",
            "state_channels",
            "trajectory_channels",
            "verification_fov_mask",
            "verification_action_vector",
        )
    }
    arrays["value_target"] = torch.tensor(
        [sample.value_target for sample in rows],
        dtype=torch.float32,
        device=device,
    )
    arrays["useful_target"] = torch.tensor(
        [sample.useful_target for sample in rows],
        dtype=torch.float32,
        device=device,
    )
    arrays["group_ids"] = tuple(
        str(sample.metadata["ranking_group_id"]) for sample in rows
    )
    arrays["action_ids"] = tuple(sample.verification_action_id for sample in rows)
    return arrays


def _predict_and_loss(
    model: VerificationValueModel,
    samples: Sequence[VerificationSample],
    *,
    config: VerifyModelConfig,
    device: torch.device,
    gradients: bool,
):
    batch = _batch(samples, device=device)
    context = torch.enable_grad() if gradients else torch.no_grad()
    with context:
        prediction = model(
            bev_history=batch["bev_history"],
            state_channels=batch["state_channels"],
            trajectory_channels=batch["trajectory_channels"],
            verification_fov_mask=batch["verification_fov_mask"],
            verification_action_vector=batch["verification_action_vector"],
        )
        loss = verification_loss(
            prediction,
            value_target=batch["value_target"],
            useful_target=batch["useful_target"],
            group_ids=batch["group_ids"],
            action_ids=batch["action_ids"],
            config=config.loss,
        )
    return prediction, loss


def _metric_report(
    samples: tuple[VerificationSample, ...],
    *,
    value_prediction: np.ndarray,
    useful_probability: np.ndarray,
    huber_delta: float,
) -> dict[str, object]:
    source_modes = tuple(
        str(sample.metadata["provenance"].get("source_mode", "unknown"))
        for sample in samples
    )
    return evaluate_verification_predictions(
        value_prediction=value_prediction,
        useful_probability=useful_probability,
        value_target=np.asarray(
            [sample.value_target for sample in samples], dtype=np.float64
        ),
        useful_target=np.asarray(
            [sample.useful_target for sample in samples], dtype=np.int64
        ),
        group_ids=tuple(
            str(sample.metadata["ranking_group_id"]) for sample in samples
        ),
        action_ids=tuple(sample.verification_action_id for sample in samples),
        huber_delta=huber_delta,
        slice_fields={
            "action": tuple(sample.verification_action_id for sample in samples),
            "source_mode": source_modes,
        },
    )


def train_verification_samples(
    samples: Sequence[VerificationSample],
    *,
    grid: GridSpec,
    config: VerifyModelConfig,
    resume: LoadedVerificationTrainingCheckpoint | None = None,
) -> VerificationTrainingResult:
    """Fit V0 on complete train groups using explicit deterministic epoch order."""

    if not isinstance(config, VerifyModelConfig):
        raise TypeError("config must be a VerifyModelConfig")
    ordered = _validated_samples(samples, grid=grid, expected_split="train")
    grouped = _groups(ordered)
    group_size = len(CANONICAL_ACTION_IDS)
    if config.training.batch_size % group_size != 0:
        raise ValueError("training batch_size must be divisible by six actions")
    groups_per_batch = max(1, config.training.batch_size // group_size)
    device = torch.device("cpu")
    model = VerificationValueModel(
        grid=grid,
        config=config.model,
        initialization_seed=config.training.seed,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    start_epoch = 0
    if resume is not None:
        if not isinstance(resume, LoadedVerificationTrainingCheckpoint):
            raise TypeError("resume must be a LoadedVerificationTrainingCheckpoint")
        model.load_state_dict(resume.model_state_dict, strict=True)
        optimizer.load_state_dict(resume.optimizer_state_dict)
        start_epoch = resume.completed_epochs
        history = list(resume.loss_history)
        if start_epoch > config.training.epochs:
            raise ValueError("resume checkpoint exceeds configured total epochs")
    else:
        model.eval()
        _, initial = _predict_and_loss(
            model, ordered, config=config, device=device, gradients=False
        )
        history = [float(initial.total.item())]
    if len(history) != start_epoch + 1 or not np.isfinite(history).all():
        raise ValueError("resume loss history is inconsistent")

    for epoch in range(start_epoch, config.training.epochs):
        rng = np.random.default_rng(
            derive_seed(config.training.seed, "verification-training-epoch", epoch)
        )
        group_order = rng.permutation(len(grouped))
        model.train()
        for start in range(0, len(group_order), groups_per_batch):
            indices = group_order[start : start + groups_per_batch]
            rows = tuple(
                sample for index in indices for sample in grouped[int(index)]
            )
            optimizer.zero_grad(set_to_none=True)
            _, loss = _predict_and_loss(
                model, rows, config=config, device=device, gradients=True
            )
            loss.total.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if not gradients or any(
                not torch.isfinite(value).all() for value in gradients
            ):
                raise ValueError("verification training gradients must be finite")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip_norm
            )
            optimizer.step()
        model.eval()
        _, evaluated = _predict_and_loss(
            model, ordered, config=config, device=device, gradients=False
        )
        epoch_loss = float(evaluated.total.item())
        if not np.isfinite(epoch_loss):
            raise ValueError("verification training loss must be finite")
        history.append(epoch_loss)

    model.eval()
    prediction, final_loss = _predict_and_loss(
        model, ordered, config=config, device=device, gradients=False
    )
    values = prediction.g_pred.detach().cpu().numpy().astype(np.float64)
    probabilities = (
        prediction.p_useful.detach().cpu().numpy().astype(np.float64)
    )
    metrics = _metric_report(
        ordered,
        value_prediction=values,
        useful_probability=probabilities,
        huber_delta=config.loss.huber_delta,
    )
    return VerificationTrainingResult(
        model=model,
        optimizer_state_dict=optimizer.state_dict(),
        completed_epochs=config.training.epochs,
        loss_history=tuple(history),
        initial_loss=float(history[0]),
        final_loss=float(final_loss.total.item()),
        metrics=metrics,
        value_prediction=values,
        useful_probability=probabilities,
        device=str(device),
    )


def evaluate_verification_samples(
    samples: Sequence[VerificationSample],
    *,
    grid: GridSpec,
    config: VerifyModelConfig,
    checkpoint: LoadedVerificationTrainingCheckpoint,
    split: str,
) -> VerificationEvaluationResult:
    """Run deterministic CPU-only forward evaluation without optimizer updates."""

    if split not in {"calibration", "val", "test"}:
        raise ValueError("verification checkpoint evaluation is held-out only")
    if not isinstance(config, VerifyModelConfig):
        raise TypeError("config must be a VerifyModelConfig")
    if not isinstance(checkpoint, LoadedVerificationTrainingCheckpoint):
        raise TypeError("checkpoint must be a LoadedVerificationTrainingCheckpoint")
    ordered = _validated_samples(samples, grid=grid, expected_split=split)
    grouped = _groups(ordered)
    device = torch.device("cpu")
    model = VerificationValueModel(
        grid=grid,
        config=config.model,
        initialization_seed=config.training.seed,
    ).to(device)
    try:
        model.load_state_dict(checkpoint.model_state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError("checkpoint model state is incompatible with config") from exc
    model.eval()
    prediction, loss = _predict_and_loss(
        model, ordered, config=config, device=device, gradients=False
    )
    values = prediction.g_pred.detach().cpu().numpy().astype(np.float64)
    probabilities = prediction.p_useful.detach().cpu().numpy().astype(np.float64)
    losses: dict[str, float | int] = {
        "total": float(loss.total.item()),
        "value_regression": float(loss.value_regression.item()),
        "useful_classification": float(loss.useful_classification.item()),
        "pairwise_ranking": float(loss.pairwise_ranking.item()),
        "pair_count": int(loss.pair_count),
    }
    if not np.isfinite(
        [value for key, value in losses.items() if key != "pair_count"]
    ).all():
        raise ValueError("verification evaluation losses must be finite")
    metrics = _metric_report(
        ordered,
        value_prediction=values,
        useful_probability=probabilities,
        huber_delta=config.loss.huber_delta,
    )
    return VerificationEvaluationResult(
        split=split,
        sample_count=len(ordered),
        group_count=len(grouped),
        losses=losses,
        metrics=metrics,
        value_prediction=values,
        useful_probability=probabilities,
        device=str(device),
    )


def write_verification_training_checkpoint(
    path: str | Path,
    *,
    result: VerificationTrainingResult,
    manifest: Mapping[str, object],
) -> Path:
    """Atomically publish one immutable checkpoint file."""

    if not isinstance(result, VerificationTrainingResult):
        raise TypeError("result must be a VerificationTrainingResult")
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().clone()
        for key, value in result.model.state_dict().items()
    }
    payload = {
        "checkpoint_version": VERIFICATION_TRAINING_CHECKPOINT_VERSION,
        "manifest": dict(manifest),
        "model_state_dict": state,
        "optimizer_state_dict": result.optimizer_state_dict,
        "completed_epochs": result.completed_epochs,
        "loss_history": list(result.loss_history),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination


def load_verification_training_checkpoint(
    path: str | Path,
    *,
    expected_input_manifest_digest: str,
    expected_split_digests: Mapping[str, str],
    expected_model_config: Mapping[str, object],
    expected_seed: int,
    expected_code_version: str,
) -> LoadedVerificationTrainingCheckpoint:
    """Load only a finite checkpoint whose embedded v2 manifest matches exactly."""

    try:
        payload = torch.load(Path(path), map_location="cpu")
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid verification checkpoint: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_KEYS:
        raise ValueError("verification checkpoint keys are invalid")
    if payload["checkpoint_version"] != VERIFICATION_TRAINING_CHECKPOINT_VERSION:
        raise ValueError("unsupported verification training checkpoint version")
    manifest = validate_verification_checkpoint_manifest(
        payload["manifest"],
        expected_input_manifest_digest=expected_input_manifest_digest,
        expected_split_digests=expected_split_digests,
        expected_model_config=expected_model_config,
        expected_seed=expected_seed,
        expected_code_version=expected_code_version,
    )
    state = payload["model_state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint model_state_dict is invalid")
    copied_state: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value, torch.Tensor)
            or not torch.isfinite(value).all()
        ):
            raise ValueError("checkpoint model state must contain finite tensors")
        copied_state[name] = value.detach().cpu().clone()
    optimizer = payload["optimizer_state_dict"]
    if not isinstance(optimizer, dict):
        raise ValueError("checkpoint optimizer_state_dict is invalid")
    completed = payload["completed_epochs"]
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise ValueError("checkpoint completed_epochs is invalid")
    try:
        history = tuple(float(value) for value in payload["loss_history"])
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint loss_history is invalid") from exc
    if len(history) != completed + 1 or not np.isfinite(history).all():
        raise ValueError("checkpoint loss_history is inconsistent")
    return LoadedVerificationTrainingCheckpoint(
        manifest=manifest,
        model_state_dict=copied_state,
        optimizer_state_dict=optimizer,
        completed_epochs=completed,
        loss_history=history,
    )


__all__ = (
    "LoadedVerificationTrainingCheckpoint",
    "VERIFICATION_TRAINING_CHECKPOINT_VERSION",
    "VerificationEvaluationResult",
    "VerificationTrainingResult",
    "evaluate_verification_samples",
    "load_verification_training_checkpoint",
    "train_verification_samples",
    "write_verification_training_checkpoint",
)
