"""Immutable analytic and raster geometry for v2 static occluders."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Union

import numpy as np

from src.contracts import GridSpec

from .footprints import CircleFootprint, RectangleFootprint
from .rasterization import grid_bounds, rasterize_footprint


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_real(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _readonly_vector(value: object, *, name: str, length: int) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array") from exc
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real values")
    result = np.asarray(array, dtype=np.float64).copy()
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.frombuffer(result.tobytes(), dtype=np.float64)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be nonempty text")
    return value


@dataclass(frozen=True)
class RectangleOccluder:
    """An oriented wall, shelf, or cabinet occluder in world coordinates."""

    occluder_id: str
    semantic_type: str
    pose: np.ndarray
    length_m: float
    width_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "occluder_id", _text(self.occluder_id, name="occluder_id")
        )
        object.__setattr__(
            self, "semantic_type", _text(self.semantic_type, name="semantic_type")
        )
        object.__setattr__(
            self, "pose", _readonly_vector(self.pose, name="pose", length=3)
        )
        object.__setattr__(
            self, "length_m", _positive_real(self.length_m, name="length_m")
        )
        object.__setattr__(
            self, "width_m", _positive_real(self.width_m, name="width_m")
        )


@dataclass(frozen=True)
class CircleOccluder:
    """A circular tree-trunk or column occluder in world coordinates."""

    occluder_id: str
    semantic_type: str
    center_xy: np.ndarray
    radius_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "occluder_id", _text(self.occluder_id, name="occluder_id")
        )
        object.__setattr__(
            self, "semantic_type", _text(self.semantic_type, name="semantic_type")
        )
        object.__setattr__(
            self,
            "center_xy",
            _readonly_vector(self.center_xy, name="center_xy", length=2),
        )
        object.__setattr__(
            self, "radius_m", _positive_real(self.radius_m, name="radius_m")
        )


StaticOccluder = Union[RectangleOccluder, CircleOccluder]


def _validate_occluder(occluder: object) -> StaticOccluder:
    if not isinstance(occluder, (RectangleOccluder, CircleOccluder)):
        raise TypeError("occluder must be a RectangleOccluder or CircleOccluder")
    return occluder


def _points(value: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric [N, 2] array") from exc
    if array.ndim != 2 or array.shape[1:] != (2,):
        raise ValueError(f"{name} must have shape [N, 2]")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real values")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def occluder_payload(occluder: StaticOccluder) -> dict[str, object]:
    """Return the stable, JSON-safe geometry payload used in v2 metadata."""

    occluder = _validate_occluder(occluder)
    if isinstance(occluder, RectangleOccluder):
        return {
            "occluder_id": occluder.occluder_id,
            "semantic_type": occluder.semantic_type,
            "shape": "rectangle",
            "pose": [float(value) for value in occluder.pose],
            "length_m": occluder.length_m,
            "width_m": occluder.width_m,
        }
    return {
        "occluder_id": occluder.occluder_id,
        "semantic_type": occluder.semantic_type,
        "shape": "circle",
        "center_xy": [float(value) for value in occluder.center_xy],
        "radius_m": occluder.radius_m,
    }


def occluder_bounds(occluder: StaticOccluder) -> tuple[float, float, float, float]:
    """Return the world-frame `(x_min, x_max, y_min, y_max)` bounds."""

    occluder = _validate_occluder(occluder)
    if isinstance(occluder, CircleOccluder):
        x, y = occluder.center_xy
        radius = occluder.radius_m
        return (float(x - radius), float(x + radius), float(y - radius), float(y + radius))
    x, y, yaw = occluder.pose
    half_length = 0.5 * occluder.length_m
    half_width = 0.5 * occluder.width_m
    extent_x = abs(np.cos(yaw)) * half_length + abs(np.sin(yaw)) * half_width
    extent_y = abs(np.sin(yaw)) * half_length + abs(np.cos(yaw)) * half_width
    return (
        float(x - extent_x),
        float(x + extent_x),
        float(y - extent_y),
        float(y + extent_y),
    )


def inflate_occluder(occluder: StaticOccluder, margin_m: object) -> StaticOccluder:
    """Return a same-type occluder expanded by a non-negative world margin."""

    occluder = _validate_occluder(occluder)
    margin = _finite_real(margin_m, name="margin_m")
    if margin < 0.0:
        raise ValueError("margin_m must be non-negative")
    if isinstance(occluder, CircleOccluder):
        return CircleOccluder(
            occluder_id=occluder.occluder_id,
            semantic_type=occluder.semantic_type,
            center_xy=occluder.center_xy,
            radius_m=occluder.radius_m + margin,
        )
    return RectangleOccluder(
        occluder_id=occluder.occluder_id,
        semantic_type=occluder.semantic_type,
        pose=occluder.pose,
        length_m=occluder.length_m + 2.0 * margin,
        width_m=occluder.width_m + 2.0 * margin,
    )


def _rectangle_local_points(
    occluder: RectangleOccluder, points_xy: np.ndarray
) -> np.ndarray:
    delta = points_xy - occluder.pose[:2]
    cosine = float(np.cos(occluder.pose[2]))
    sine = float(np.sin(occluder.pose[2]))
    return np.column_stack(
        (delta[:, 0] * cosine + delta[:, 1] * sine, -delta[:, 0] * sine + delta[:, 1] * cosine)
    )


def point_signed_distance(
    occluder: StaticOccluder, points_xy: object
) -> np.ndarray:
    """Return vectorized signed distance: negative inside, zero on, positive outside."""

    occluder = _validate_occluder(occluder)
    points = _points(points_xy, name="points_xy")
    if isinstance(occluder, CircleOccluder):
        return np.linalg.norm(points - occluder.center_xy, axis=1) - occluder.radius_m
    local = _rectangle_local_points(occluder, points)
    half_extents = np.asarray(
        [0.5 * occluder.length_m, 0.5 * occluder.width_m], dtype=np.float64
    )
    q = np.abs(local) - half_extents
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def segment_intersects_occluder(
    occluder: StaticOccluder,
    starts_xy: object,
    ends_xy: object,
    *,
    epsilon_m: object,
) -> np.ndarray:
    """Return whether each `[N,2]` centerline segment intersects an occluder."""

    occluder = _validate_occluder(occluder)
    starts = _points(starts_xy, name="starts_xy")
    ends = _points(ends_xy, name="ends_xy")
    if starts.shape != ends.shape:
        raise ValueError("starts_xy and ends_xy must have matching shapes")
    epsilon = _finite_real(epsilon_m, name="epsilon_m")
    if epsilon < 0.0:
        raise ValueError("epsilon_m must be non-negative")
    if isinstance(occluder, CircleOccluder):
        direction = ends - starts
        squared_length = np.sum(direction * direction, axis=1)
        offset = occluder.center_xy - starts
        fraction = np.divide(
            np.sum(offset * direction, axis=1),
            squared_length,
            out=np.zeros_like(squared_length),
            where=squared_length > 0.0,
        )
        closest = starts + np.clip(fraction, 0.0, 1.0)[:, None] * direction
        squared_distance = np.sum((closest - occluder.center_xy) ** 2, axis=1)
        radius = occluder.radius_m + epsilon
        return squared_distance <= radius * radius

    starts_local = _rectangle_local_points(occluder, starts)
    ends_local = _rectangle_local_points(occluder, ends)
    direction = ends_local - starts_local
    half_extents = np.asarray(
        [
            0.5 * occluder.length_m + epsilon,
            0.5 * occluder.width_m + epsilon,
        ],
        dtype=np.float64,
    )
    parallel = direction == 0.0
    outside_parallel = parallel & (np.abs(starts_local) > half_extents)
    valid = ~np.any(outside_parallel, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        first = (-half_extents - starts_local) / direction
        second = (half_extents - starts_local) / direction
    lower = np.where(parallel, -np.inf, np.minimum(first, second))
    upper = np.where(parallel, np.inf, np.maximum(first, second))
    entry = np.max(lower, axis=1)
    exit = np.min(upper, axis=1)
    return valid & (np.maximum(entry, 0.0) <= np.minimum(exit, 1.0))


def rasterize_occluder(occluder: StaticOccluder, grid: GridSpec) -> np.ndarray:
    """Rasterize an in-grid static occluder with the existing occupancy authority."""

    occluder = _validate_occluder(occluder)
    x_min, x_max, y_min, y_max = grid_bounds(grid)
    bounds = occluder_bounds(occluder)
    if (
        bounds[0] < x_min
        or bounds[1] > x_max
        or bounds[2] < y_min
        or bounds[3] > y_max
    ):
        raise ValueError("occluder must lie inside the grid")
    if isinstance(occluder, CircleOccluder):
        return rasterize_footprint(
            CircleFootprint(occluder.radius_m),
            np.asarray([*occluder.center_xy, 0.0], dtype=np.float64),
            grid,
        )
    return rasterize_footprint(
        RectangleFootprint(occluder.length_m, occluder.width_m),
        occluder.pose,
        grid,
    )
