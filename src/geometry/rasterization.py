"""Canonical mapping between centred world coordinates and BEV grid cells.

Occupancy arrays use ``occupancy[row, column]``. Columns increase with world
``x`` and rows increase with world ``y``. World points use ``[..., x, y]``;
grid indices use ``[..., row, column]``.
"""

from __future__ import annotations

import numpy as np

from src.contracts import GridSpec

from .collision import intersects
from .footprints import (
    Footprint,
    RectangleFootprint,
    _validate_footprint,
    footprint_aabb,
)


def _validate_grid(grid: GridSpec) -> None:
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    for name in ("height", "width"):
        value = getattr(grid, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"GridSpec.{name} must be a positive integer")
        if value <= 0:
            raise ValueError(f"GridSpec.{name} must be a positive integer")
    resolution = grid.resolution_m
    if isinstance(resolution, (bool, np.bool_)) or not isinstance(
        resolution, (int, float, np.integer, np.floating)
    ):
        raise ValueError("GridSpec.resolution_m must be a positive finite number")
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("GridSpec.resolution_m must be a positive finite number")
    if not np.isfinite(float(grid.width) * float(resolution)) or not np.isfinite(
        float(grid.height) * float(resolution)
    ):
        raise ValueError("GridSpec spatial extent must be finite")


def _points(values, *, name: str) -> np.ndarray:
    try:
        points = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if points.ndim == 0 or points.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (..., 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points


def _indices(values) -> np.ndarray:
    indices = np.asarray(values)
    if indices.ndim == 0 or indices.shape[-1] != 2:
        raise ValueError("indices must have shape (..., 2)")
    try:
        numeric_indices = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("indices must contain numeric integer values") from exc
    if not np.all(np.isfinite(numeric_indices)):
        raise ValueError("indices must contain only finite values")
    if np.issubdtype(indices.dtype, np.bool_) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must have an integer dtype")
    return indices


def grid_bounds(grid: GridSpec) -> tuple[float, float, float, float]:
    """Return ``(x_min, x_max, y_min, y_max)`` for the centred half-open grid."""
    _validate_grid(grid)
    half_width = float(grid.width) * float(grid.resolution_m) / 2.0
    half_height = float(grid.height) * float(grid.resolution_m) / 2.0
    return (-half_width, half_width, -half_height, half_height)


def points_in_grid(points_world, grid: GridSpec) -> np.ndarray:
    """Return whether each finite world point lies in the half-open grid bounds."""
    _validate_grid(grid)
    points = _points(points_world, name="points_world")
    x_min, x_max, y_min, y_max = grid_bounds(grid)
    return (
        (points[..., 0] >= x_min)
        & (points[..., 0] < x_max)
        & (points[..., 1] >= y_min)
        & (points[..., 1] < y_max)
    )


def world_to_grid(points_world, grid: GridSpec, *, clip: bool = False) -> np.ndarray:
    """Map ``[..., x, y]`` points to int64 ``[..., row, column]`` indices."""
    _validate_grid(grid)
    points = _points(points_world, name="points_world")
    if not isinstance(clip, (bool, np.bool_)):
        raise TypeError("clip must be a boolean")
    in_bounds = points_in_grid(points, grid)
    if not clip and not np.all(in_bounds):
        raise ValueError("points_world contains point(s) outside the grid")

    x_min, _, y_min, _ = grid_bounds(grid)
    columns = (points[..., 0] - x_min) / float(grid.resolution_m)
    rows = (points[..., 1] - y_min) / float(grid.resolution_m)
    if clip:
        columns = np.clip(columns, 0.0, np.nextafter(float(grid.width), -np.inf))
        rows = np.clip(rows, 0.0, np.nextafter(float(grid.height), -np.inf))
    row_indices = np.clip(np.floor(rows), 0, grid.height - 1)
    column_indices = np.clip(np.floor(columns), 0, grid.width - 1)
    indices = np.stack((row_indices, column_indices), axis=-1)
    return indices.astype(np.int64, copy=False)


def grid_to_world(indices, grid: GridSpec, *, clip: bool = False) -> np.ndarray:
    """Map ``[..., row, column]`` indices to float64 ``[..., x, y]`` centres."""
    _validate_grid(grid)
    grid_indices = _indices(indices)
    if not isinstance(clip, (bool, np.bool_)):
        raise TypeError("clip must be a boolean")
    rows = grid_indices[..., 0]
    columns = grid_indices[..., 1]
    in_bounds = (
        (rows >= 0)
        & (rows < grid.height)
        & (columns >= 0)
        & (columns < grid.width)
    )
    if not clip and not np.all(in_bounds):
        raise ValueError("indices contains cell(s) outside the grid")
    if clip:
        rows = np.clip(rows, 0, grid.height - 1)
        columns = np.clip(columns, 0, grid.width - 1)

    x_min, _, y_min, _ = grid_bounds(grid)
    x_world = x_min + (columns.astype(np.float64) + 0.5) * float(grid.resolution_m)
    y_world = y_min + (rows.astype(np.float64) + 0.5) * float(grid.resolution_m)
    return np.stack((x_world, y_world), axis=-1).astype(np.float64, copy=False)


def grid_cell_centers(grid: GridSpec) -> np.ndarray:
    """Return all cell centres as a float64 ``[height, width, 2]`` array."""
    _validate_grid(grid)
    indices = np.indices((grid.height, grid.width), dtype=np.int64).transpose(1, 2, 0)
    return grid_to_world(indices, grid)


def rasterize_footprint(footprint: Footprint, pose, grid: GridSpec) -> np.ndarray:
    """Rasterize every closed grid cell square touched by ``footprint``."""
    _validate_grid(grid)
    bounds = footprint_aabb(footprint, pose)
    centers = grid_cell_centers(grid)
    half_cell = 0.5 * float(grid.resolution_m)
    candidates = (
        (centers[..., 0] >= bounds[0] - half_cell)
        & (centers[..., 0] <= bounds[1] + half_cell)
        & (centers[..., 1] >= bounds[2] - half_cell)
        & (centers[..., 1] <= bounds[3] + half_cell)
    )
    result = np.zeros((grid.height, grid.width), dtype=bool)
    cell = RectangleFootprint(grid.resolution_m, grid.resolution_m)
    for row, column in np.argwhere(candidates):
        center = centers[row, column]
        cell_pose = np.array([center[0], center[1], 0.0], dtype=np.float64)
        result[row, column] = intersects(footprint, pose, cell, cell_pose)
    return result


def rasterize_footprint_sweep(
    footprint: Footprint, poses, grid: GridSpec
) -> np.ndarray:
    """Return the union of footprint masks at the supplied discrete poses."""
    _validate_grid(grid)
    _validate_footprint(footprint)
    try:
        pose_array = np.asarray(poses)
    except (TypeError, ValueError) as exc:
        raise TypeError("poses must be a numeric array") from exc
    if pose_array.ndim != 2 or pose_array.shape[1:] != (3,):
        raise ValueError("poses must have shape (T, 3)")
    if pose_array.dtype.kind not in "iuf":
        raise TypeError("poses must contain real numbers")
    validated_poses = np.asarray(pose_array, dtype=np.float64)
    if not np.all(np.isfinite(validated_poses)):
        raise ValueError("poses must contain only finite values")

    result = np.zeros((grid.height, grid.width), dtype=bool)
    for pose in validated_poses:
        result |= rasterize_footprint(footprint, pose, grid)
    return result
