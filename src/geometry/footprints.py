"""Geometric footprint definitions and pose-dependent bounds."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Union

import numpy as np


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite_real(value: Any, name: str) -> float:
    result = _finite_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class CircleFootprint:
    """A circular footprint centered on a pose."""

    radius_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "radius_m", _positive_finite_real(self.radius_m, "radius_m")
        )


@dataclass(frozen=True)
class RectangleFootprint:
    """A rectangle whose length and width follow local +x and +y."""

    length_m: float
    width_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "length_m", _positive_finite_real(self.length_m, "length_m")
        )
        object.__setattr__(
            self, "width_m", _positive_finite_real(self.width_m, "width_m")
        )


Footprint = Union[CircleFootprint, RectangleFootprint]


def _validate_footprint(footprint: Any) -> Footprint:
    if not isinstance(footprint, (CircleFootprint, RectangleFootprint)):
        raise TypeError("footprint must be a CircleFootprint or RectangleFootprint")
    return footprint


def _validate_pose(pose: Any) -> np.ndarray:
    try:
        array = np.asarray(pose)
    except (TypeError, ValueError) as exc:
        raise TypeError("pose must be a numeric array") from exc
    if array.shape != (3,):
        raise ValueError("pose must have shape (3,)")
    if array.dtype.kind not in "iuf":
        raise TypeError("pose must contain real numbers")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("pose must contain only finite values")
    return result


def inflate_footprint(footprint: Footprint, margin_m: float) -> Footprint:
    """Return a new footprint expanded outwards by ``margin_m``."""

    footprint = _validate_footprint(footprint)
    margin = _finite_real(margin_m, "margin_m")
    if margin < 0.0:
        raise ValueError("margin_m must be non-negative")
    if isinstance(footprint, CircleFootprint):
        return CircleFootprint(footprint.radius_m + margin)
    return RectangleFootprint(
        footprint.length_m + 2.0 * margin,
        footprint.width_m + 2.0 * margin,
    )


def footprint_vertices(
    rectangle: RectangleFootprint, pose: Any
) -> np.ndarray:
    """Return rectangle corners in stable counter-clockwise order."""

    if not isinstance(rectangle, RectangleFootprint):
        raise TypeError("footprint_vertices only supports RectangleFootprint")
    x, y, yaw = _validate_pose(pose)
    half_length = 0.5 * rectangle.length_m
    half_width = 0.5 * rectangle.width_m
    local = np.array(
        [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    rotation = np.array(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    return np.asarray(local @ rotation.T + np.array([x, y]), dtype=np.float64)


def footprint_aabb(footprint: Footprint, pose: Any) -> tuple[float, float, float, float]:
    """Return ``(x_min, x_max, y_min, y_max)`` in world coordinates."""

    footprint = _validate_footprint(footprint)
    validated_pose = _validate_pose(pose)
    if isinstance(footprint, CircleFootprint):
        x, y = validated_pose[:2]
        radius = footprint.radius_m
        return (float(x - radius), float(x + radius), float(y - radius), float(y + radius))

    vertices = footprint_vertices(footprint, validated_pose)
    return (
        float(np.min(vertices[:, 0])),
        float(np.max(vertices[:, 0])),
        float(np.min(vertices[:, 1])),
        float(np.max(vertices[:, 1])),
    )
