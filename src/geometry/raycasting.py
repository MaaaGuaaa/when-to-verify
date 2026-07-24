"""Deterministic visibility ray casting on the canonical BEV grid."""

from __future__ import annotations

from numbers import Real

import numpy as np

from src.contracts import GridSpec

from .rasterization import (
    _validate_grid,
    grid_bounds,
    grid_cell_centers,
    points_in_grid,
    world_to_grid,
)
from .transforms import wrap_angle


def _real_parameter(value, *, name: str, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or result < minimum or (
        maximum is not None and result > maximum
    ):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be finite and {interval}")
    return result


def _occupancy_mask(occupancy, grid: GridSpec) -> np.ndarray:
    try:
        array = np.asarray(occupancy)
    except (TypeError, ValueError) as exc:
        raise TypeError("occupancy must be a numeric array") from exc
    if array.shape != (grid.height, grid.width):
        raise ValueError("occupancy must have shape (grid.height, grid.width)")
    if array.dtype.kind not in "biuf":
        raise TypeError("occupancy must have a boolean or real numeric dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError("occupancy must contain only finite values")
    return np.asarray(array != 0, dtype=bool)


def _sensor_pose(sensor_pose, grid: GridSpec) -> np.ndarray:
    try:
        array = np.asarray(sensor_pose)
    except (TypeError, ValueError) as exc:
        raise TypeError("sensor_pose must be a numeric array") from exc
    if array.shape != (3,):
        raise ValueError("sensor_pose must have shape (3,)")
    if array.dtype.kind not in "iuf":
        raise TypeError("sensor_pose must contain real numbers")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("sensor_pose must contain only finite values")
    if not bool(points_in_grid(result[:2], grid)):
        raise ValueError("sensor_pose x/y must lie inside the grid")
    return result


def _candidate_visibility(
    occupancy: np.ndarray,
    grid: GridSpec,
    sensor_xy: np.ndarray,
    centers: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Run the exact supercover DDA for all candidate cells in parallel."""
    visible = np.zeros((grid.height, grid.width), dtype=bool)
    targets = np.argwhere(candidates)
    if targets.size == 0:
        return visible
    if not occupancy.any():
        return candidates.copy()

    sensor_index = world_to_grid(sensor_xy, grid)
    sensor_row = int(sensor_index[0])
    sensor_column = int(sensor_index[1])
    x_min, _, y_min, _ = grid_bounds(grid)
    resolution = float(grid.resolution_m)
    target_rows = targets[:, 0]
    target_columns = targets[:, 1]
    target_centers = centers[target_rows, target_columns]
    delta_x = target_centers[:, 0] - sensor_xy[0]
    delta_y = target_centers[:, 1] - sensor_xy[1]
    step_columns = np.sign(delta_x).astype(np.int64)
    step_rows = np.sign(delta_y).astype(np.int64)
    rows = np.full(targets.shape[0], sensor_row, dtype=np.int64)
    columns = np.full(targets.shape[0], sensor_column, dtype=np.int64)

    t_max_x = np.full(targets.shape[0], np.inf, dtype=np.float64)
    t_delta_x = np.full(targets.shape[0], np.inf, dtype=np.float64)
    moving_x = step_columns != 0
    boundary_x = x_min + (
        sensor_column + (step_columns[moving_x] > 0)
    ) * resolution
    t_max_x[moving_x] = (boundary_x - sensor_xy[0]) / delta_x[moving_x]
    t_delta_x[moving_x] = resolution / np.abs(delta_x[moving_x])

    t_max_y = np.full(targets.shape[0], np.inf, dtype=np.float64)
    t_delta_y = np.full(targets.shape[0], np.inf, dtype=np.float64)
    moving_y = step_rows != 0
    boundary_y = y_min + (sensor_row + (step_rows[moving_y] > 0)) * resolution
    t_max_y[moving_y] = (boundary_y - sensor_xy[1]) / delta_y[moving_y]
    t_delta_y[moving_y] = resolution / np.abs(delta_y[moving_y])

    ray_visible = np.ones(targets.shape[0], dtype=bool)
    active = (rows != target_rows) | (columns != target_columns)

    def terminate_blocked(ray_ids: np.ndarray, blocked: np.ndarray) -> None:
        blocked_ids = ray_ids[blocked]
        ray_visible[blocked_ids] = False
        active[blocked_ids] = False

    def check_current_cells(ray_ids: np.ndarray) -> None:
        if ray_ids.size == 0:
            return
        reached = (rows[ray_ids] == target_rows[ray_ids]) & (
            columns[ray_ids] == target_columns[ray_ids]
        )
        active[ray_ids[reached]] = False
        traversing = ray_ids[~reached]
        if traversing.size:
            in_bounds = (
                (rows[traversing] >= 0)
                & (rows[traversing] < grid.height)
                & (columns[traversing] >= 0)
                & (columns[traversing] < grid.width)
            )
            if not np.all(in_bounds):
                raise RuntimeError("ray traversal left the grid before reaching its target")
            terminate_blocked(
                traversing, occupancy[rows[traversing], columns[traversing]]
            )

    max_iterations = grid.height + grid.width + 2
    epsilon = np.finfo(np.float64).eps
    for _ in range(max_iterations):
        ray_ids = np.flatnonzero(active)
        if ray_ids.size == 0:
            break
        current_t_x = t_max_x[ray_ids]
        current_t_y = t_max_y[ray_ids]
        finite_both = np.isfinite(current_t_x) & np.isfinite(current_t_y)
        tied = np.zeros(ray_ids.size, dtype=bool)
        tie_scale = np.maximum(
            1.0, np.maximum(np.abs(current_t_x[finite_both]), np.abs(current_t_y[finite_both]))
        )
        tied[finite_both] = (
            np.abs(current_t_x[finite_both] - current_t_y[finite_both])
            <= 8.0 * epsilon * tie_scale
        )
        x_only = ~tied & (current_t_x < current_t_y)
        y_only = ~tied & (current_t_y < current_t_x)

        x_ids = ray_ids[x_only]
        columns[x_ids] += step_columns[x_ids]
        t_max_x[x_ids] += t_delta_x[x_ids]
        check_current_cells(x_ids)

        y_ids = ray_ids[y_only]
        rows[y_ids] += step_rows[y_ids]
        t_max_y[y_ids] += t_delta_y[y_ids]
        check_current_cells(y_ids)

        tie_ids = ray_ids[~x_only & ~y_only]
        if tie_ids.size:
            side_columns = columns[tie_ids] + step_columns[tie_ids]
            side_rows = rows[tie_ids] + step_rows[tie_ids]
            side_blocked = np.zeros(tie_ids.size, dtype=bool)

            horizontal_valid = (
                (rows[tie_ids] >= 0)
                & (rows[tie_ids] < grid.height)
                & (side_columns >= 0)
                & (side_columns < grid.width)
            )
            horizontal_sensor = (rows[tie_ids] == sensor_row) & (
                side_columns == sensor_column
            )
            horizontal_target = (rows[tie_ids] == target_rows[tie_ids]) & (
                side_columns == target_columns[tie_ids]
            )
            check_horizontal = horizontal_valid & ~horizontal_sensor & ~horizontal_target
            side_blocked[check_horizontal] |= occupancy[
                rows[tie_ids][check_horizontal], side_columns[check_horizontal]
            ]

            vertical_valid = (
                (side_rows >= 0)
                & (side_rows < grid.height)
                & (columns[tie_ids] >= 0)
                & (columns[tie_ids] < grid.width)
            )
            vertical_sensor = (side_rows == sensor_row) & (
                columns[tie_ids] == sensor_column
            )
            vertical_target = (side_rows == target_rows[tie_ids]) & (
                columns[tie_ids] == target_columns[tie_ids]
            )
            check_vertical = vertical_valid & ~vertical_sensor & ~vertical_target
            side_blocked[check_vertical] |= occupancy[
                side_rows[check_vertical], columns[tie_ids][check_vertical]
            ]
            terminate_blocked(tie_ids, side_blocked)

            advancing = tie_ids[~side_blocked]
            rows[advancing] += step_rows[advancing]
            columns[advancing] += step_columns[advancing]
            t_max_x[advancing] += t_delta_x[advancing]
            t_max_y[advancing] += t_delta_y[advancing]
            check_current_cells(advancing)
    else:
        raise RuntimeError("ray traversal exceeded its deterministic iteration bound")

    if np.any(active):
        raise RuntimeError("ray traversal did not reach every target")
    visible[target_rows, target_columns] = ray_visible
    return visible


def _filtered_candidates(
    candidates: np.ndarray,
    centers: np.ndarray,
    sensor: np.ndarray,
    *,
    fov: float,
    maximum_range: float | None,
) -> np.ndarray:
    selected = candidates.copy()
    deltas = centers - sensor[:2]
    distances = np.hypot(deltas[..., 0], deltas[..., 1])
    if maximum_range is not None:
        selected &= distances <= maximum_range
    if fov < 2.0 * np.pi:
        bearings = np.arctan2(deltas[..., 1], deltas[..., 0])
        differences = np.abs(wrap_angle(bearings - sensor[2]))
        selected &= differences <= 0.5 * fov
    return selected


def raycast_candidate_visibility(
    occupancy,
    candidate_mask,
    grid: GridSpec,
    *,
    sensor_pose=(0.0, 0.0, 0.0),
    fov_rad=2.0 * np.pi,
    max_range_m=None,
) -> np.ndarray:
    """Raycast only selected cells with the exact full-map visibility semantics."""

    _validate_grid(grid)
    occupied = _occupancy_mask(occupancy, grid)
    candidates = np.asarray(candidate_mask)
    if candidates.shape != (grid.height, grid.width):
        raise ValueError("candidate_mask must match the grid shape")
    if candidates.dtype != np.bool_:
        raise TypeError("candidate_mask must have boolean dtype")
    sensor = _sensor_pose(sensor_pose, grid)
    fov = _real_parameter(
        fov_rad,
        name="fov_rad",
        minimum=0.0,
        maximum=2.0 * np.pi,
    )
    maximum_range = (
        None
        if max_range_m is None
        else _real_parameter(max_range_m, name="max_range_m", minimum=0.0)
    )
    centers = grid_cell_centers(grid)
    selected = _filtered_candidates(
        candidates,
        centers,
        sensor,
        fov=fov,
        maximum_range=maximum_range,
    )
    return _candidate_visibility(occupied, grid, sensor[:2], centers, selected)


def raycast_visibility(
    occupancy,
    grid: GridSpec,
    *,
    sensor_pose=(0.0, 0.0, 0.0),
    fov_rad=2.0 * np.pi,
    max_range_m=None,
) -> np.ndarray:
    """Return cells visible from a finite in-grid sensor pose.

    Every cell, including the sensor's containing cell, is first selected using
    its centre relative to the sensor's actual ``x, y`` for FOV and range.
    """
    _validate_grid(grid)
    occupied = _occupancy_mask(occupancy, grid)
    sensor = _sensor_pose(sensor_pose, grid)
    fov = _real_parameter(fov_rad, name="fov_rad", minimum=0.0, maximum=2.0 * np.pi)
    maximum_range = (
        None
        if max_range_m is None
        else _real_parameter(max_range_m, name="max_range_m", minimum=0.0)
    )

    centers = grid_cell_centers(grid)
    candidates = _filtered_candidates(
        np.ones((grid.height, grid.width), dtype=bool),
        centers,
        sensor,
        fov=fov,
        maximum_range=maximum_range,
    )
    return _candidate_visibility(occupied, grid, sensor[:2], centers, candidates)
