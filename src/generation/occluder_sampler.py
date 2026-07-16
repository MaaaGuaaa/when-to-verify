"""Physical environment-occluder placement with finite deterministic retries."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

import numpy as np

from src.contracts import GridSpec
from src.geometry import (
    Footprint,
    RectangleFootprint,
    footprint_aabb,
    grid_bounds,
    rasterize_footprint,
    rasterize_footprint_sweep,
    raycast_visibility,
    trajectory_signed_clearances,
    wrap_angle,
)
from src.utils.seeding import stable_digest

from .structural_blindspot import (
    footprint_visibility_sequence,
    has_continuous_emergence,
)


class OccluderSamplingError(ValueError):
    """Raised after the configured finite placement budget is exhausted."""

    def __init__(
        self,
        reason: str,
        *,
        attempts: int,
        rejection_reasons: Mapping[str, int] | None = None,
    ):
        super().__init__(f"{reason} after {attempts} attempts")
        self.reason = reason
        self.attempts = attempts
        self.rejection_reasons = dict(sorted((rejection_reasons or {}).items()))


@dataclass(frozen=True)
class OccluderPlacement:
    """One accepted rectangular environment occluder and its BEV mask."""

    occluder: dict[str, object]
    footprint: RectangleFootprint
    pose: np.ndarray
    mask: np.ndarray
    attempt: int
    rejection_reasons: dict[str, int]


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _vector(value: Any, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric vector with shape ({size},)")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _poses(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1:] != (3,) or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric array with shape (T, 3)")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _range_pair(value: Any, *, name: str, positive: bool = True) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain [minimum, maximum]")
    lower = _finite_real(value[0], name=f"{name}[0]")
    upper = _finite_real(value[1], name=f"{name}[1]")
    if lower > upper or (positive and lower <= 0.0):
        raise ValueError(f"{name} is not a valid positive range")
    return lower, upper


def normalize_occluder_config(config: Mapping[str, Any]) -> dict[str, object]:
    if not isinstance(config, Mapping):
        raise TypeError("occluder config must be a mapping")
    expected = {
        "types",
        "normal_offset_range_m",
        "wall",
        "shelf",
        "pillar",
    }
    if set(config) != expected:
        raise ValueError("occluder config keys do not match the frozen SOP-05 schema")
    types = config["types"]
    if not isinstance(types, (list, tuple)) or not types:
        raise ValueError("occluder types must be a non-empty list")
    allowed = ("wall", "shelf", "pillar")
    if len(set(types)) != len(types) or any(kind not in allowed for kind in types):
        raise ValueError("occluder types must be unique wall/shelf/pillar names")
    normalized: dict[str, object] = {
        "types": tuple(types),
        "normal_offset_range_m": _range_pair(
            config["normal_offset_range_m"], name="normal_offset_range_m"
        ),
    }
    for kind in allowed:
        node = config[kind]
        if not isinstance(node, Mapping) or set(node) != {
            "length_range_m",
            "width_range_m",
        }:
            raise ValueError(
                f"occluder {kind} config must contain length_range_m and width_range_m"
            )
        normalized[kind] = {
            "length_range_m": _range_pair(
                node["length_range_m"], name=f"{kind}.length_range_m"
            ),
            "width_range_m": _range_pair(
                node["width_range_m"], name=f"{kind}.width_range_m"
            ),
        }
    return normalized


def _range_quantile(bounds: tuple[float, float], quantile: float) -> float:
    lower, upper = bounds
    return lower + quantile * (upper - lower)


def _inside_grid(footprint: Footprint, pose: np.ndarray, grid: GridSpec) -> bool:
    x_min, x_max, y_min, y_max = footprint_aabb(footprint, pose)
    grid_x_min, grid_x_max, grid_y_min, grid_y_max = grid_bounds(grid)
    return bool(
        x_min >= grid_x_min
        and x_max < grid_x_max
        and y_min >= grid_y_min
        and y_max < grid_y_max
    )


def _record_rejection(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def sample_environment_occluder(
    *,
    static_occupancy: Any,
    grid: GridSpec,
    sensor_pose: Any,
    conflict_point: Any,
    trajectory_normal: Any,
    robot_poses: Any,
    robot_footprint: Footprint,
    target_current_pose: Any,
    target_future_poses: Any,
    target_footprint: Footprint,
    context_trajectories: Mapping[str, np.ndarray],
    context_footprints: Mapping[str, Footprint],
    config: Mapping[str, Any],
    rng: np.random.Generator,
    max_attempts: int,
) -> OccluderPlacement:
    """Place an LOS-blocking obstacle without blocking any physical sweep.

    Each candidate is anchored at a configured normal offset from the conflict
    point. Its tangential coordinate is the intersection with the sensor-to-
    target line of sight, which makes blocking likely without putting the
    obstacle directly on the target crossing path.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    attempts = _positive_integer(max_attempts, name="max_attempts")
    normalized = normalize_occluder_config(config)
    occupancy = np.asarray(static_occupancy)
    if occupancy.shape != (grid.height, grid.width) or occupancy.dtype.kind not in "biuf":
        raise ValueError("static_occupancy must be a numeric grid-shaped array")
    if not np.isfinite(occupancy).all():
        raise ValueError("static_occupancy must contain only finite values")
    occupied = np.asarray(occupancy != 0, dtype=bool)
    sensor = _vector(sensor_pose, name="sensor_pose", size=3)
    conflict = _vector(conflict_point, name="conflict_point", size=2)
    normal = _vector(trajectory_normal, name="trajectory_normal", size=2)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        raise ValueError("trajectory_normal must be non-zero")
    normal /= normal_norm
    robot_pose_array = _poses(robot_poses, name="robot_poses")
    robot_pose_array = np.vstack((sensor, robot_pose_array))
    target_current = _vector(target_current_pose, name="target_current_pose", size=3)
    target_future = _poses(target_future_poses, name="target_future_poses")
    if set(context_trajectories) != set(context_footprints):
        raise ValueError("context trajectory and footprint keys must match")
    context_pose_arrays = {
        object_id: _poses(poses, name=f"context_trajectories[{object_id!r}]")
        for object_id, poses in sorted(context_trajectories.items())
    }

    robot_sweep = rasterize_footprint_sweep(
        robot_footprint, robot_pose_array, grid
    )
    target_poses = np.vstack((target_current, target_future))
    target_side = float(np.dot(target_current[:2] - conflict, normal))
    los = target_current[:2] - sensor[:2]
    los_normal_component = float(np.dot(los, normal))
    rejection_reasons: dict[str, int] = {}
    if abs(target_side) <= 1e-9 or abs(los_normal_component) <= 1e-9:
        raise OccluderSamplingError(
            "occluder_no_valid_placement",
            attempts=attempts,
            rejection_reasons={"occluder_los_degenerate": attempts},
        )
    side_sign = 1.0 if target_side > 0.0 else -1.0
    type_values = tuple(normalized["types"])
    type_order = tuple(
        type_values[int(index)] for index in rng.permutation(len(type_values))
    )
    offset_quantiles = (0.30, 0.50, 0.70, 0.10, 0.90, 0.20, 0.40, 0.60)

    for attempt in range(1, attempts + 1):
        cycle, type_index = divmod(attempt - 1, len(type_order))
        kind = type_order[type_index]
        dimensions = normalized[kind]
        dimension_quantile = min(1.0, 0.25 * cycle)
        offset_quantile = offset_quantiles[cycle % len(offset_quantiles)]
        length_m = _range_quantile(
            dimensions["length_range_m"], dimension_quantile
        )
        width_m = _range_quantile(
            dimensions["width_range_m"], dimension_quantile
        )
        normal_offset = side_sign * _range_quantile(
            normalized["normal_offset_range_m"], offset_quantile
        )
        desired_normal_coordinate = float(np.dot(conflict, normal)) + normal_offset
        fraction = (
            desired_normal_coordinate - float(np.dot(sensor[:2], normal))
        ) / los_normal_component
        if not 0.0 < fraction < 1.0:
            _record_rejection(
                rejection_reasons, "occluder_offset_outside_line_of_sight"
            )
            continue
        center = sensor[:2] + fraction * los
        los_yaw = float(np.arctan2(los[1], los[0]))
        pose = np.asarray(
            [center[0], center[1], wrap_angle(los_yaw + 0.5 * np.pi)],
            dtype=np.float64,
        )
        footprint = RectangleFootprint(length_m=length_m, width_m=width_m)
        if not _inside_grid(footprint, pose, grid):
            _record_rejection(rejection_reasons, "occluder_out_of_bounds")
            continue
        mask = rasterize_footprint(footprint, pose, grid)
        if np.any(mask & occupied):
            _record_rejection(rejection_reasons, "occluder_static_overlap")
            continue
        if np.any(mask & robot_sweep):
            _record_rejection(
                rejection_reasons, "occluder_robot_swept_overlap"
            )
            continue
        target_clearances = trajectory_signed_clearances(
            footprint,
            np.tile(pose, (target_poses.shape[0], 1)),
            target_footprint,
            target_poses,
        )
        if np.any(target_clearances <= 0.0):
            _record_rejection(rejection_reasons, "occluder_target_collision")
            continue
        if any(
            np.any(
                trajectory_signed_clearances(
                    footprint,
                    np.tile(pose, (poses.shape[0], 1)),
                    context_footprints[object_id],
                    poses,
                )
                <= 0.0
            )
            for object_id, poses in context_pose_arrays.items()
        ):
            _record_rejection(rejection_reasons, "occluder_context_collision")
            continue
        visibility = raycast_visibility(
            occupied | mask,
            grid,
            sensor_pose=sensor,
        )
        sequence = footprint_visibility_sequence(
            target_footprint, target_poses, visibility, grid
        )
        if bool(sequence[0]):
            _record_rejection(
                rejection_reasons, "occluder_does_not_hide_current_target"
            )
            continue
        if not has_continuous_emergence(sequence, min_visible_frames=2):
            _record_rejection(
                rejection_reasons, "occluder_target_does_not_emerge"
            )
            continue
        pose32 = pose.astype(np.float32)
        occluder_id = "occluder-" + stable_digest(
            kind,
            *(f"{value:.9f}" for value in pose32),
            f"{length_m:.9f}",
            f"{width_m:.9f}",
            size=12,
        )
        return OccluderPlacement(
            occluder={
                "occluder_id": occluder_id,
                "type": kind,
                "pose": [float(value) for value in pose32],
                "length_m": float(length_m),
                "width_m": float(width_m),
                "geometry_source": "generator_config",
                "placement_strategy": "conflict_normal_los_intersection",
                "normal_offset_m": float(normal_offset),
                "line_of_sight_fraction": float(fraction),
                "dimension_quantile": float(dimension_quantile),
                "offset_quantile": float(offset_quantile),
            },
            footprint=footprint,
            pose=pose32,
            mask=mask,
            attempt=attempt,
            rejection_reasons=dict(sorted(rejection_reasons.items())),
        )

    raise OccluderSamplingError(
        "occluder_no_valid_placement",
        attempts=attempts,
        rejection_reasons=rejection_reasons,
    )
