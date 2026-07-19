"""SOP08 analytic and learned hidden-occupancy predictors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from src.contracts import HISTORY_CHANNELS, STATE_CHANNELS

from .occupancy_aggregation import future_endpoint_times


FUTURE_STEPS = 15
FUTURE_DT_S = 0.2
HISTORY_STEPS = 8


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_probability_array(
    value: Any,
    *,
    name: str,
    rank: int,
    channels: int | None = None,
    channel_axis: int = 1,
    require_probability_range: bool = True,
) -> None:
    if getattr(value, "ndim", None) != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if channels is not None and int(value.shape[channel_axis]) != channels:
        raise ValueError(f"{name} must contain exactly {channels} channels")
    if torch.is_tensor(value):
        if value.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        if require_probability_range and bool(((value < 0.0) | (value > 1.0)).any()):
            raise ValueError(f"{name} probability channels must be in [0,1]")
    elif isinstance(value, np.ndarray):
        if value.dtype != np.float32:
            raise ValueError(f"{name} must be float32")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
        if require_probability_range and np.logical_or(value < 0.0, value > 1.0).any():
            raise ValueError(f"{name} probability channels must be in [0,1]")
    else:
        raise TypeError(f"{name} must be a NumPy array or torch tensor")


class LastObservationHold:
    """Repeat the last observed dynamic-occupancy channel into the future."""

    def __init__(self, *, future_steps: int = FUTURE_STEPS) -> None:
        self.future_steps = _positive_integer("future_steps", future_steps)
        self.dynamic_channel_index = HISTORY_CHANNELS.index("past_dynamic_occupancy")

    def __call__(self, bev_history: Any) -> Any:
        _validate_probability_array(
            bev_history,
            name="bev_history",
            rank=5,
            channels=len(HISTORY_CHANNELS),
            channel_axis=2,
        )
        if int(bev_history.shape[1]) != HISTORY_STEPS:
            raise ValueError(
                f"bev_history must contain exactly {HISTORY_STEPS} history frames"
            )
        last = bev_history[:, -1, self.dynamic_channel_index]
        if torch.is_tensor(last):
            return last.unsqueeze(1).repeat(1, self.future_steps, 1, 1)
        return np.repeat(last[:, None], self.future_steps, axis=1).astype(
            np.float32,
            copy=False,
        )

    predict = __call__


class AgeDecay:
    """Decay last-seen occupancy using normalized age and endpoint time."""

    def __init__(
        self,
        *,
        future_steps: int = FUTURE_STEPS,
        dt_s: float = FUTURE_DT_S,
        tau_s: float = 2.0,
        a_max_s: float = 5.0,
    ) -> None:
        self.future_steps = _positive_integer("future_steps", future_steps)
        for name, value in (("dt_s", dt_s), ("tau_s", tau_s), ("a_max_s", a_max_s)):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        self.dt_s = float(dt_s)
        self.tau_s = float(tau_s)
        self.a_max_s = float(a_max_s)
        self.last_seen_index = STATE_CHANNELS.index("last_seen_occupancy")
        self.age_index = STATE_CHANNELS.index("occlusion_age_map")

    def __call__(self, state_channels: Any) -> Any:
        _validate_probability_array(
            state_channels,
            name="state_channels",
            rank=4,
            channels=len(STATE_CHANNELS),
            channel_axis=1,
            require_probability_range=False,
        )
        last_seen = state_channels[:, self.last_seen_index]
        age_seconds = state_channels[:, self.age_index] * self.a_max_s
        normalized_age = state_channels[:, self.age_index]
        if torch.is_tensor(state_channels):
            if bool(((last_seen < 0.0) | (last_seen > 1.0)).any()):
                raise ValueError("last_seen_occupancy must be in [0,1]")
            if bool(((normalized_age < 0.0) | (normalized_age > 1.0)).any()):
                raise ValueError("occlusion_age_map must be normalized to [0,1]")
        else:
            if np.logical_or(last_seen < 0.0, last_seen > 1.0).any():
                raise ValueError("last_seen_occupancy must be in [0,1]")
            if np.logical_or(normalized_age < 0.0, normalized_age > 1.0).any():
                raise ValueError("occlusion_age_map must be normalized to [0,1]")
        times = future_endpoint_times(future_steps=self.future_steps, dt_s=self.dt_s)
        if torch.is_tensor(state_channels):
            endpoint = torch.as_tensor(
                times,
                dtype=state_channels.dtype,
                device=state_channels.device,
            ).view(1, -1, 1, 1)
            probability = last_seen[:, None] * torch.exp(
                -(age_seconds[:, None] + endpoint) / self.tau_s
            )
            return probability.clamp(0.0, 1.0)
        endpoint = times.reshape(1, -1, 1, 1)
        probability = last_seen[:, None] * np.exp(
            -(age_seconds[:, None] + endpoint) / np.float32(self.tau_s)
        )
        return np.clip(probability, 0.0, 1.0).astype(np.float32, copy=False)

    predict = __call__


class ConvGRUCell(nn.Module):
    """Small convolutional GRU cell with same-size spatial padding."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        _positive_integer("input_channels", input_channels)
        _positive_integer("hidden_channels", hidden_channels)
        _positive_integer("kernel_size", kernel_size)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        joined = input_channels + hidden_channels
        self.gates = nn.Conv2d(joined, 2 * hidden_channels, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(joined, hidden_channels, kernel_size, padding=padding)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        reset, update = torch.sigmoid(self.gates(torch.cat((inputs, hidden), dim=1))).chunk(
            2,
            dim=1,
        )
        candidate = torch.tanh(
            self.candidate(torch.cat((inputs, reset * hidden), dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


class ConvGRUOccupancyPredictor(nn.Module):
    """Encode eight observed BEV frames and autoregress 15 occupancy logits."""

    def __init__(
        self,
        *,
        history_channels: int = len(HISTORY_CHANNELS),
        hidden_channels: int = 8,
        future_steps: int = FUTURE_STEPS,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if history_channels != len(HISTORY_CHANNELS):
            raise ValueError(
                f"history_channels must match frozen contract ({len(HISTORY_CHANNELS)})"
            )
        self.history_channels = history_channels
        self.hidden_channels = _positive_integer("hidden_channels", hidden_channels)
        self.future_steps = _positive_integer("future_steps", future_steps)
        self.encoder_cell = ConvGRUCell(history_channels, hidden_channels, kernel_size)
        self.decoder_cell = ConvGRUCell(1, hidden_channels, kernel_size)
        self.output_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.dynamic_channel_index = HISTORY_CHANNELS.index("past_dynamic_occupancy")

    def _validate_history(self, bev_history: torch.Tensor) -> None:
        if not torch.is_tensor(bev_history):
            raise TypeError("bev_history must be a torch tensor")
        if bev_history.ndim != 5:
            raise ValueError("bev_history must have rank 5 [B,K,C,H,W]")
        if int(bev_history.shape[1]) != HISTORY_STEPS:
            raise ValueError(
                f"bev_history must contain exactly {HISTORY_STEPS} history frames"
            )
        if int(bev_history.shape[2]) != self.history_channels:
            raise ValueError(
                f"bev_history history channels must be {self.history_channels}, "
                f"got {int(bev_history.shape[2])}"
            )
        if bev_history.dtype != torch.float32:
            raise ValueError("bev_history must be float32")
        if not bool(torch.isfinite(bev_history).all()):
            raise ValueError("bev_history must be finite")
        if bool(((bev_history < 0.0) | (bev_history > 1.0)).any()):
            raise ValueError("bev_history probability channels must be in [0,1]")

    def predict_logits(self, bev_history: torch.Tensor) -> torch.Tensor:
        self._validate_history(bev_history)
        batch_size, _, _, height, width = bev_history.shape
        hidden = bev_history.new_zeros(batch_size, self.hidden_channels, height, width)
        for step in range(int(bev_history.shape[1])):
            hidden = self.encoder_cell(bev_history[:, step], hidden)

        previous = bev_history[:, -1, self.dynamic_channel_index : self.dynamic_channel_index + 1]
        logits: list[torch.Tensor] = []
        for _ in range(self.future_steps):
            hidden = self.decoder_cell(previous, hidden)
            current_logits = self.output_head(hidden)
            logits.append(current_logits[:, 0])
            previous = torch.sigmoid(current_logits)
        return torch.stack(logits, dim=1)

    def forward(self, bev_history: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.predict_logits(bev_history))


class LearnedOccupancyRiskAggregator(nn.Module):
    """B4 head over predicted occupancy and robot future footprints only."""

    def __init__(self, *, future_steps: int = FUTURE_STEPS, hidden_dim: int = 32) -> None:
        super().__init__()
        self.future_steps = _positive_integer("future_steps", future_steps)
        hidden_dim = _positive_integer("hidden_dim", hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.future_steps * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _features(
        self,
        occupancy: torch.Tensor,
        robot_future_footprints: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(occupancy) or not torch.is_tensor(robot_future_footprints):
            raise TypeError("occupancy and robot_future_footprints must be torch tensors")
        if occupancy.ndim != 4 or robot_future_footprints.ndim != 4:
            raise ValueError("occupancy and robot_future_footprints must have rank 4 [B,T,H,W]")
        if tuple(occupancy.shape) != tuple(robot_future_footprints.shape):
            raise ValueError("occupancy and robot_future_footprints must have the same shape")
        if int(occupancy.shape[1]) != self.future_steps:
            raise ValueError(f"future axis must contain {self.future_steps} endpoint frames")
        if occupancy.dtype != torch.float32 or robot_future_footprints.dtype != torch.float32:
            raise ValueError("occupancy and robot_future_footprints must be float32")
        if not bool(torch.isfinite(occupancy).all()) or bool(
            ((occupancy < 0.0) | (occupancy > 1.0)).any()
        ):
            raise ValueError("occupancy probabilities must be finite and in [0,1]")
        if not bool(torch.isfinite(robot_future_footprints).all()) or bool(
            (robot_future_footprints < 0.0).any()
        ):
            raise ValueError("robot_future_footprints must be finite and nonnegative")

        mask = robot_future_footprints > 0.0
        mask_float = mask.to(occupancy.dtype)
        count = mask_float.sum(dim=(2, 3))
        mean = (occupancy * mask_float).sum(dim=(2, 3)) / count.clamp_min(1.0)
        selected_for_max = torch.where(mask, occupancy, torch.full_like(occupancy, -1.0))
        maximum = selected_for_max.amax(dim=(2, 3)).clamp_min(0.0)
        selected = torch.where(mask, occupancy, torch.zeros_like(occupancy))
        union = 1.0 - torch.prod(1.0 - selected.flatten(start_dim=2), dim=2)
        return torch.stack((mean, maximum, union), dim=2)

    def forward(
        self,
        occupancy: torch.Tensor,
        robot_future_footprints: torch.Tensor,
    ) -> torch.Tensor:
        features = self._features(occupancy, robot_future_footprints)
        return torch.sigmoid(self.network(features.flatten(start_dim=1))[:, 0])


__all__ = [
    "AgeDecay",
    "ConvGRUCell",
    "ConvGRUOccupancyPredictor",
    "FUTURE_DT_S",
    "FUTURE_STEPS",
    "HISTORY_STEPS",
    "LastObservationHold",
    "LearnedOccupancyRiskAggregator",
]
