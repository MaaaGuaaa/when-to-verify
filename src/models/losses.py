"""SOP09 quantile, collision, and optional occupancy losses."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch.nn import functional as F

from src.contracts import QUANTILE_LEVELS


def pinball_loss(
    predicted_quantiles: torch.Tensor,
    target: torch.Tensor,
    *,
    levels: Sequence[float] = QUANTILE_LEVELS,
) -> torch.Tensor:
    """Return the mean quantile pinball loss over batch and levels."""

    if predicted_quantiles.ndim != 2 or predicted_quantiles.shape[1] != len(levels):
        raise ValueError(
            f"predicted quantile shape must be [B,{len(levels)}]"
        )
    if target.ndim != 1 or target.shape[0] != predicted_quantiles.shape[0]:
        raise ValueError("target shape must be [B] and align with predictions")
    if not torch.is_floating_point(predicted_quantiles) or not torch.is_floating_point(
        target
    ):
        raise ValueError("pinball inputs must be floating-point tensors")
    if not torch.isfinite(predicted_quantiles).all().item() or not torch.isfinite(
        target
    ).all().item():
        raise ValueError("pinball inputs must be finite")
    if any(not 0.0 < float(level) < 1.0 for level in levels):
        raise ValueError("quantile levels must lie strictly inside (0,1)")
    level_tensor = predicted_quantiles.new_tensor(tuple(float(v) for v in levels))
    residual = target.unsqueeze(1) - predicted_quantiles
    return torch.maximum(level_tensor * residual, (level_tensor - 1.0) * residual).mean()


def risk_loss(
    output: Mapping[str, torch.Tensor],
    *,
    risk_severity: torch.Tensor,
    collision_label: torch.Tensor,
    lambda_collision: float = 1.0,
    occupancy_target: torch.Tensor | None = None,
    lambda_occupancy_aux: float = 0.0,
    levels: Sequence[float] = QUANTILE_LEVELS,
) -> dict[str, torch.Tensor]:
    """Compose pinball + collision BCE + optional occupancy BCE."""

    required = {"quantiles", "collision_logits"}
    if not isinstance(output, Mapping) or not required.issubset(output):
        raise ValueError("risk model output lacks quantiles/collision_logits")
    if lambda_collision < 0.0 or lambda_occupancy_aux < 0.0:
        raise ValueError("loss weights must be nonnegative")
    quantiles = output["quantiles"]
    logits = output["collision_logits"]
    if logits.ndim != 1 or collision_label.shape != logits.shape:
        raise ValueError("collision logits/label shape must both be [B]")
    if risk_severity.ndim != 1 or risk_severity.shape != logits.shape:
        raise ValueError("risk_severity shape must be [B]")
    if not torch.isfinite(logits).all().item() or not torch.isfinite(
        collision_label
    ).all().item():
        raise ValueError("collision logits/labels must be finite")
    if torch.any((risk_severity < 0.0) | (risk_severity > 1.0)).item():
        raise ValueError("risk_severity must be in [0,1]")
    if torch.any((collision_label < 0.0) | (collision_label > 1.0)).item():
        raise ValueError("collision_label must be in [0,1]")
    pinball = pinball_loss(quantiles, risk_severity, levels=levels)
    collision_bce = F.binary_cross_entropy_with_logits(logits, collision_label)

    occupancy_prediction = output.get("occupancy_aux_logits")
    uses_occupancy = (
        occupancy_prediction is not None
        or occupancy_target is not None
        or lambda_occupancy_aux > 0.0
    )
    if uses_occupancy:
        if occupancy_prediction is None or occupancy_target is None:
            raise ValueError(
                "occupancy auxiliary prediction and target must be supplied together"
            )
        if occupancy_prediction.shape != occupancy_target.shape:
            raise ValueError("occupancy auxiliary prediction/target shape mismatch")
        if not torch.isfinite(occupancy_prediction).all().item() or not torch.isfinite(
            occupancy_target
        ).all().item():
            raise ValueError("occupancy auxiliary tensors must be finite")
        if torch.any((occupancy_target < 0.0) | (occupancy_target > 1.0)).item():
            raise ValueError("occupancy auxiliary target must be in [0,1]")
        occupancy_aux = F.binary_cross_entropy_with_logits(
            occupancy_prediction, occupancy_target
        )
    else:
        occupancy_aux = pinball.new_zeros(())
    total = (
        pinball
        + float(lambda_collision) * collision_bce
        + float(lambda_occupancy_aux) * occupancy_aux
    )
    return {
        "total": total,
        "pinball": pinball,
        "collision_bce": collision_bce,
        "occupancy_aux": occupancy_aux,
    }


__all__ = ["pinball_loss", "risk_loss"]
