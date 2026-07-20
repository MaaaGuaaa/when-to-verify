"""Leakage-safe V0 concat CNN and grouped verification-value losses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from src.contracts import GridSpec, SCHEMA_VERSION


VERIFICATION_MODEL_VERSION = "verification_concat_cnn_v0"
_TOP_KEYS = frozenset({"schema_version", "model", "loss", "training"})
_MODEL_KEYS = frozenset(
    {
        "version",
        "spatial_channels",
        "action_hidden_dim",
        "fusion_hidden_dim",
    }
)
_LOSS_KEYS = frozenset(
    {
        "huber_delta",
        "value_weight",
        "useful_weight",
        "ranking_weight",
        "ranking_margin",
    }
)
_TRAINING_KEYS = frozenset(
    {
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
    }
)


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_real(
    value: Any, *, name: str, positive: bool = False, nonnegative: bool = False
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class VerificationModelConfig:
    version: str
    spatial_channels: tuple[int, ...]
    action_hidden_dim: int
    fusion_hidden_dim: int

    def __post_init__(self) -> None:
        if self.version != VERIFICATION_MODEL_VERSION:
            raise ValueError("unsupported verification model version")
        if (
            not isinstance(self.spatial_channels, tuple)
            or len(self.spatial_channels) < 2
        ):
            raise ValueError("spatial_channels must contain at least two stages")
        channels = tuple(
            _positive_integer(value, name=f"spatial_channels[{index}]")
            for index, value in enumerate(self.spatial_channels)
        )
        object.__setattr__(self, "spatial_channels", channels)
        object.__setattr__(
            self,
            "action_hidden_dim",
            _positive_integer(self.action_hidden_dim, name="action_hidden_dim"),
        )
        object.__setattr__(
            self,
            "fusion_hidden_dim",
            _positive_integer(self.fusion_hidden_dim, name="fusion_hidden_dim"),
        )


@dataclass(frozen=True)
class VerificationLossConfig:
    huber_delta: float
    value_weight: float
    useful_weight: float
    ranking_weight: float
    ranking_margin: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "huber_delta",
            _finite_real(self.huber_delta, name="huber_delta", positive=True),
        )
        for name in ("value_weight", "useful_weight", "ranking_weight"):
            object.__setattr__(
                self,
                name,
                _finite_real(getattr(self, name), name=name, nonnegative=True),
            )
        if self.value_weight + self.useful_weight + self.ranking_weight <= 0.0:
            raise ValueError("at least one verification loss weight must be positive")
        object.__setattr__(
            self,
            "ranking_margin",
            _finite_real(
                self.ranking_margin, name="ranking_margin", nonnegative=True
            ),
        )


@dataclass(frozen=True)
class VerificationTrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float

    def __post_init__(self) -> None:
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (Integral, np.integer)
        ):
            raise TypeError("training seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "epochs", _positive_integer(self.epochs, name="epochs"))
        object.__setattr__(
            self, "batch_size", _positive_integer(self.batch_size, name="batch_size")
        )
        object.__setattr__(
            self,
            "learning_rate",
            _finite_real(self.learning_rate, name="learning_rate", positive=True),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _finite_real(self.weight_decay, name="weight_decay", nonnegative=True),
        )
        object.__setattr__(
            self,
            "gradient_clip_norm",
            _finite_real(
                self.gradient_clip_norm,
                name="gradient_clip_norm",
                positive=True,
            ),
        )


@dataclass(frozen=True)
class VerifyModelConfig:
    schema_version: str
    model: VerificationModelConfig
    loss: VerificationLossConfig
    training: VerificationTrainingConfig

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"verification model schema must be {SCHEMA_VERSION}")


@dataclass(frozen=True)
class VerificationPrediction:
    g_pred: torch.Tensor
    useful_logit: torch.Tensor

    @property
    def p_useful(self) -> torch.Tensor:
        return torch.sigmoid(self.useful_logit)


@dataclass(frozen=True)
class VerificationLossResult:
    total: torch.Tensor
    value_regression: torch.Tensor
    useful_classification: torch.Tensor
    pairwise_ranking: torch.Tensor
    pair_count: int


def _strict_section(
    value: object, *, name: str, expected_keys: frozenset[str]
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"{name} keys are invalid")
    return value


def load_verify_model_config(path: str | Path) -> VerifyModelConfig:
    """Load the schema-3 V0 architecture/loss/training configuration strictly."""

    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid verification model config: {exc}") from exc
    top = _strict_section(raw, name="verification model config", expected_keys=_TOP_KEYS)
    if top["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"verification model schema must be {SCHEMA_VERSION}")
    model = _strict_section(top["model"], name="model", expected_keys=_MODEL_KEYS)
    loss = _strict_section(top["loss"], name="loss", expected_keys=_LOSS_KEYS)
    training = _strict_section(
        top["training"], name="training", expected_keys=_TRAINING_KEYS
    )
    raw_channels = model["spatial_channels"]
    if not isinstance(raw_channels, list):
        raise ValueError("model spatial_channels must be a list")
    return VerifyModelConfig(
        schema_version=str(top["schema_version"]),
        model=VerificationModelConfig(
            version=model["version"],
            spatial_channels=tuple(raw_channels),
            action_hidden_dim=model["action_hidden_dim"],
            fusion_hidden_dim=model["fusion_hidden_dim"],
        ),
        loss=VerificationLossConfig(**loss),
        training=VerificationTrainingConfig(**training),
    )


def _initialization_seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError("initialization_seed must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError("initialization_seed must be non-negative")
    return result


class VerificationValueModel(nn.Module):
    """V0 concat CNN accepting deployment-available verification inputs only."""

    version = VERIFICATION_MODEL_VERSION

    def __init__(
        self,
        *,
        grid: GridSpec,
        config: VerificationModelConfig,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec")
        if not isinstance(config, VerificationModelConfig):
            raise TypeError("config must be a VerificationModelConfig")
        seed = _initialization_seed(initialization_seed)
        self.grid = grid
        self.config = config
        spatial_input_channels = (
            grid.history_steps * grid.n_history_channels
            + grid.n_state_channels
            + grid.n_trajectory_channels
            + 1
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            stages: list[nn.Module] = []
            input_channels = spatial_input_channels
            for index, output_channels in enumerate(config.spatial_channels):
                stages.extend(
                    (
                        nn.Conv2d(
                            input_channels,
                            output_channels,
                            kernel_size=5 if index == 0 else 3,
                            stride=4 if index == 0 else 2,
                            padding=2 if index == 0 else 1,
                        ),
                        nn.ReLU(inplace=False),
                    )
                )
                input_channels = output_channels
            stages.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.spatial_encoder = nn.Sequential(*stages)
            self.action_encoder = nn.Sequential(
                nn.Linear(3, config.action_hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(config.action_hidden_dim, config.action_hidden_dim),
                nn.ReLU(inplace=False),
            )
            self.fusion = nn.Sequential(
                nn.Linear(
                    config.spatial_channels[-1] + config.action_hidden_dim,
                    config.fusion_hidden_dim,
                ),
                nn.ReLU(inplace=False),
            )
            self.value_head = nn.Linear(config.fusion_hidden_dim, 1)
            self.useful_head = nn.Linear(config.fusion_hidden_dim, 1)

    def _validate_inputs(
        self,
        *,
        bev_history: torch.Tensor,
        state_channels: torch.Tensor,
        trajectory_channels: torch.Tensor,
        verification_fov_mask: torch.Tensor,
        verification_action_vector: torch.Tensor,
    ) -> int:
        values = {
            "bev_history": bev_history,
            "state_channels": state_channels,
            "trajectory_channels": trajectory_channels,
            "verification_fov_mask": verification_fov_mask,
            "verification_action_vector": verification_action_vector,
        }
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.dtype != torch.float32:
                raise TypeError(f"{name} must be float32")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        batch = int(bev_history.shape[0]) if bev_history.ndim > 0 else -1
        expected = {
            "bev_history": (
                batch,
                self.grid.history_steps,
                self.grid.n_history_channels,
                self.grid.height,
                self.grid.width,
            ),
            "state_channels": (
                batch,
                self.grid.n_state_channels,
                self.grid.height,
                self.grid.width,
            ),
            "trajectory_channels": (
                batch,
                self.grid.n_trajectory_channels,
                self.grid.height,
                self.grid.width,
            ),
            "verification_fov_mask": (
                batch,
                1,
                self.grid.height,
                self.grid.width,
            ),
            "verification_action_vector": (batch, 3),
        }
        if batch <= 0:
            raise ValueError("verification model batch must be non-empty")
        for name, shape in expected.items():
            if tuple(values[name].shape) != shape:
                raise ValueError(f"{name} shape must be {shape}")
        devices = {value.device for value in values.values()}
        if len(devices) != 1:
            raise ValueError("verification model inputs must share one device")
        return batch

    def forward(
        self,
        *,
        bev_history: torch.Tensor,
        state_channels: torch.Tensor,
        trajectory_channels: torch.Tensor,
        verification_fov_mask: torch.Tensor,
        verification_action_vector: torch.Tensor,
    ) -> VerificationPrediction:
        batch = self._validate_inputs(
            bev_history=bev_history,
            state_channels=state_channels,
            trajectory_channels=trajectory_channels,
            verification_fov_mask=verification_fov_mask,
            verification_action_vector=verification_action_vector,
        )
        flattened_history = bev_history.reshape(
            batch,
            self.grid.history_steps * self.grid.n_history_channels,
            self.grid.height,
            self.grid.width,
        )
        spatial = torch.cat(
            (
                flattened_history,
                state_channels,
                trajectory_channels,
                verification_fov_mask,
            ),
            dim=1,
        )
        spatial_embedding = self.spatial_encoder(spatial).flatten(start_dim=1)
        action_embedding = self.action_encoder(verification_action_vector)
        fused = self.fusion(torch.cat((spatial_embedding, action_embedding), dim=1))
        return VerificationPrediction(
            g_pred=self.value_head(fused).squeeze(-1),
            useful_logit=self.useful_head(fused).squeeze(-1),
        )


def _target_tensor(
    value: object, *, name: str, batch_size: int, device: torch.device
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != (batch_size,) or not value.dtype.is_floating_point:
        raise ValueError(f"{name} must be a floating tensor with shape ({batch_size},)")
    if value.device != device:
        raise ValueError(f"{name} must share the prediction device")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _identity_sequence(
    value: object, *, name: str, batch_size: int
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(value)
    if len(result) != batch_size:
        raise ValueError(f"{name} must align with predictions")
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain non-empty strings")
    return result


def verification_loss(
    prediction: VerificationPrediction,
    *,
    value_target: torch.Tensor,
    useful_target: torch.Tensor,
    group_ids: Sequence[str],
    action_ids: Sequence[str],
    config: VerificationLossConfig,
) -> VerificationLossResult:
    """Compute Huber+BCE+within-group pairwise hinge ranking loss."""

    if not isinstance(prediction, VerificationPrediction):
        raise TypeError("prediction must be a VerificationPrediction")
    if not isinstance(config, VerificationLossConfig):
        raise TypeError("config must be a VerificationLossConfig")
    g_pred = prediction.g_pred
    useful_logit = prediction.useful_logit
    if not isinstance(g_pred, torch.Tensor) or not isinstance(
        useful_logit, torch.Tensor
    ):
        raise TypeError("prediction fields must be tensors")
    if g_pred.ndim != 1 or useful_logit.shape != g_pred.shape or g_pred.numel() == 0:
        raise ValueError("prediction fields must be aligned non-empty [B] tensors")
    if (
        not g_pred.dtype.is_floating_point
        or useful_logit.dtype != g_pred.dtype
        or useful_logit.device != g_pred.device
        or not torch.isfinite(g_pred).all()
        or not torch.isfinite(useful_logit).all()
    ):
        raise ValueError("prediction tensors must be finite and aligned")
    batch_size = int(g_pred.shape[0])
    values = _target_tensor(
        value_target, name="value_target", batch_size=batch_size, device=g_pred.device
    )
    useful = _target_tensor(
        useful_target,
        name="useful_target",
        batch_size=batch_size,
        device=g_pred.device,
    )
    if not torch.all((useful == 0.0) | (useful == 1.0)):
        raise ValueError("useful_target must be binary")
    groups = _identity_sequence(group_ids, name="group_ids", batch_size=batch_size)
    actions = _identity_sequence(action_ids, name="action_ids", batch_size=batch_size)

    value_regression = F.huber_loss(
        g_pred, values.to(dtype=g_pred.dtype), reduction="mean", delta=config.huber_delta
    )
    useful_classification = F.binary_cross_entropy_with_logits(
        useful_logit, useful.to(dtype=useful_logit.dtype), reduction="mean"
    )
    pair_losses: list[torch.Tensor] = []
    for left in range(batch_size):
        for right in range(left + 1, batch_size):
            if groups[left] != groups[right] or actions[left] == actions[right]:
                continue
            target_difference = values[left] - values[right]
            if bool(target_difference == 0.0):
                continue
            direction = torch.sign(target_difference).to(dtype=g_pred.dtype)
            predicted_difference = g_pred[left] - g_pred[right]
            pair_losses.append(
                torch.relu(config.ranking_margin - direction * predicted_difference)
            )
    if pair_losses:
        pairwise_ranking = torch.stack(pair_losses).mean()
    else:
        pairwise_ranking = g_pred.sum() * 0.0
    total = (
        config.value_weight * value_regression
        + config.useful_weight * useful_classification
        + config.ranking_weight * pairwise_ranking
    )
    if not torch.isfinite(total):
        raise ValueError("verification loss must be finite")
    return VerificationLossResult(
        total=total,
        value_regression=value_regression,
        useful_classification=useful_classification,
        pairwise_ranking=pairwise_ranking,
        pair_count=len(pair_losses),
    )


__all__ = (
    "VERIFICATION_MODEL_VERSION",
    "VerificationLossConfig",
    "VerificationLossResult",
    "VerificationModelConfig",
    "VerificationPrediction",
    "VerificationTrainingConfig",
    "VerificationValueModel",
    "VerifyModelConfig",
    "load_verify_model_config",
    "verification_loss",
)
