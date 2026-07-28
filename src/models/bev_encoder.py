"""Small spatial and temporal encoders used by SOP09 R0/R1."""

from __future__ import annotations

import torch
from torch import nn


class BEVEncoder(nn.Module):
    """Two-layer same-resolution CNN suitable for the small risk models."""

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        if in_channels < 1 or hidden_channels < 1:
            raise ValueError("encoder channel counts must be positive")
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("BEVEncoder inputs must have shape [B,C,H,W]")
        return self.network(inputs)


class ConvGRUCell(nn.Module):
    """Minimal convolutional GRU cell with a spatial hidden state."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        if input_channels < 1 or hidden_channels < 1:
            raise ValueError("ConvGRU channel counts must be positive")
        merged = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(merged, 2 * hidden_channels, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(merged, hidden_channels, kernel_size=3, padding=1)

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("ConvGRU inputs must have shape [B,C,H,W]")
        if hidden is None:
            hidden = inputs.new_zeros(
                inputs.shape[0], self.hidden_channels, inputs.shape[2], inputs.shape[3]
            )
        if hidden.shape != (
            inputs.shape[0],
            self.hidden_channels,
            inputs.shape[2],
            inputs.shape[3],
        ):
            raise ValueError("ConvGRU hidden state shape mismatch")
        reset, update = torch.sigmoid(
            self.gates(torch.cat((inputs, hidden), dim=1))
        ).chunk(2, dim=1)
        proposal = torch.tanh(
            self.candidate(torch.cat((inputs, reset * hidden), dim=1))
        )
        return (1.0 - update) * hidden + update * proposal


__all__ = ["BEVEncoder", "ConvGRUCell"]
