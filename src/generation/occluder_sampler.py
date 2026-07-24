"""Physical environment-occluder placement with finite deterministic retries."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Mapping

import numpy as np

from src.contracts import GridSpec
from src.geometry import (
    CircleFootprint,
    Footprint,
    RectangleFootprint,
    footprint_aabb,
    grid_bounds,
    grid_to_world,
    rasterize_footprint,
    raycast_visibility,
    signed_clearance,
    trajectory_signed_clearances,
    wrap_angle,
)
from src.utils.seeding import stable_digest

from .structural_blindspot import (
    footprint_visibility_sequence,
    has_continuous_emergence,
)


JOINT_MULTI_LOS_PLACEMENT_STRATEGY_VERSION = "joint_multi_los_envelope_v2"
OCCLUDER_COLLISION_SWEEP_PREPARATION_VERSION = (
    "occluder_collision_sweep_preparation_v2"
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


@dataclass(frozen=True)
class JointOccluderParameters:
    """One deterministic point in the joint occluder/target proposal design."""

    occluder_type: str
    side: int
    offset_quantile: float
    dimension_quantile: float
    angle_multiplier: float
    conflict_time_quantile: float = 0.5


@dataclass(frozen=True)
class OccluderGeometryCandidate:
    """Target-independent geometry that already clears the static world."""

    occluder: dict[str, object]
    footprint: RectangleFootprint
    pose: np.ndarray
    mask: np.ndarray
    proposal_index: int


@dataclass(frozen=True)
class OccluderCollisionSweep:
    """One named physical motion sweep that a candidate must clear."""

    footprint: Footprint
    poses: np.ndarray
    rejection_reason: str


@dataclass(frozen=True)
class PreparedOccluderCollisionSweep:
    """Immutable candidate-independent geometry for one continuous sweep."""

    footprint: Footprint
    dense_poses: np.ndarray
    interval_motion_bounds_m: np.ndarray
    rejection_reason: str
    grid: GridSpec
    preparation_version: str = OCCLUDER_COLLISION_SWEEP_PREPARATION_VERSION
    _dense_pose_storage: bytes = field(init=False, repr=False, compare=False)
    _interval_motion_bound_storage: bytes = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.footprint, (CircleFootprint, RectangleFootprint)):
            raise TypeError("footprint must be a Footprint")
        dense_poses = _poses(self.dense_poses, name="dense_poses")
        if dense_poses.shape[0] == 0:
            raise ValueError("dense_poses must not be empty")
        canonical_dense_poses = np.array(
            dense_poses,
            dtype=np.dtype("<f8"),
            order="C",
            copy=True,
        )

        bounds = np.asarray(self.interval_motion_bounds_m)
        if (
            bounds.shape != (canonical_dense_poses.shape[0] - 1,)
            or bounds.dtype.kind not in "iuf"
        ):
            raise ValueError(
                "interval_motion_bounds_m must have shape (len(dense_poses) - 1,)"
            )
        canonical_bounds = np.array(
            bounds,
            dtype=np.dtype("<f8"),
            order="C",
            copy=True,
        )
        if not np.isfinite(canonical_bounds).all() or np.any(canonical_bounds < 0.0):
            raise ValueError(
                "interval_motion_bounds_m must contain finite non-negative values"
            )
        motion_radius_m = _rotation_motion_radius(self.footprint)
        expected_bounds = np.asarray(
            [
                _pose_interval_motion_bound(
                    start_pose,
                    end_pose,
                    motion_radius_m=motion_radius_m,
                )
                for start_pose, end_pose in zip(
                    canonical_dense_poses[:-1],
                    canonical_dense_poses[1:],
                    strict=True,
                )
            ],
            dtype=np.dtype("<f8"),
        )
        if not np.array_equal(canonical_bounds, expected_bounds):
            raise ValueError(
                "interval_motion_bounds_m must match canonical interval motion bounds"
            )

        if not isinstance(self.rejection_reason, str) or not self.rejection_reason:
            raise ValueError("rejection_reason must be non-empty")
        if not isinstance(self.grid, GridSpec):
            raise TypeError("grid must be a GridSpec")
        if not isinstance(self.preparation_version, str) or not self.preparation_version:
            raise ValueError("preparation_version must be non-empty")

        dense_pose_storage = canonical_dense_poses.tobytes(order="C")
        interval_motion_bound_storage = expected_bounds.tobytes(order="C")
        immutable_dense_poses = np.frombuffer(
            dense_pose_storage,
            dtype=np.dtype("<f8"),
        ).reshape(canonical_dense_poses.shape)
        immutable_bounds = np.frombuffer(
            interval_motion_bound_storage,
            dtype=np.dtype("<f8"),
        )
        object.__setattr__(self, "_dense_pose_storage", dense_pose_storage)
        object.__setattr__(
            self,
            "_interval_motion_bound_storage",
            interval_motion_bound_storage,
        )
        object.__setattr__(self, "dense_poses", immutable_dense_poses)
        object.__setattr__(self, "interval_motion_bounds_m", immutable_bounds)


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


def _densify_pose_sequence(
    poses: np.ndarray,
    *,
    max_translation_step_m: float,
    max_yaw_step_rad: float,
) -> np.ndarray:
    """Interpolate a pose path so exact collision checks include frame gaps."""

    dense = [poses[0].copy()]
    for start, end in zip(poses[:-1], poses[1:], strict=True):
        translation = float(np.linalg.norm(end[:2] - start[:2]))
        yaw_delta = float(wrap_angle(end[2] - start[2]))
        steps = max(
            1,
            int(np.ceil(translation / max_translation_step_m)),
            int(np.ceil(abs(yaw_delta) / max_yaw_step_rad)),
        )
        fractions = np.arange(1, steps + 1, dtype=np.float64) / steps
        positions = start[:2] + fractions[:, None] * (end[:2] - start[:2])
        yaws = wrap_angle(start[2] + fractions * yaw_delta)
        dense.extend(
            np.column_stack((positions, np.asarray(yaws, dtype=np.float64)))
        )
    return np.asarray(dense, dtype=np.float64)


def _sweep_motion_radius(footprint: Footprint) -> float:
    if isinstance(footprint, CircleFootprint):
        return footprint.radius_m
    return 0.5 * float(np.hypot(footprint.length_m, footprint.width_m))


def _rotation_motion_radius(footprint: Footprint) -> float:
    if isinstance(footprint, CircleFootprint):
        return 0.0
    return _sweep_motion_radius(footprint)


def _pose_interval_motion_bound(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    *,
    motion_radius_m: float,
) -> float:
    yaw_delta = float(wrap_angle(end_pose[2] - start_pose[2]))
    result = float(np.linalg.norm(end_pose[:2] - start_pose[:2]))
    return result + motion_radius_m * abs(yaw_delta)


def _densify_synchronized_pose_sequences(
    poses_a: np.ndarray,
    poses_b: np.ndarray,
    *,
    max_translation_step_m: float,
    max_yaw_step_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    if poses_a.shape != poses_b.shape:
        raise ValueError("synchronized pose sequences must have the same shape")
    if poses_a.shape[0] == 0:
        raise ValueError("synchronized pose sequences must not be empty")

    dense_a = [poses_a[0].copy()]
    dense_b = [poses_b[0].copy()]
    for start_a, end_a, start_b, end_b in zip(
        poses_a[:-1],
        poses_a[1:],
        poses_b[:-1],
        poses_b[1:],
        strict=True,
    ):
        yaw_delta_a = float(wrap_angle(end_a[2] - start_a[2]))
        yaw_delta_b = float(wrap_angle(end_b[2] - start_b[2]))
        steps = max(
            1,
            int(
                np.ceil(
                    float(np.linalg.norm(end_a[:2] - start_a[:2]))
                    / max_translation_step_m
                )
            ),
            int(
                np.ceil(
                    float(np.linalg.norm(end_b[:2] - start_b[:2]))
                    / max_translation_step_m
                )
            ),
            int(np.ceil(abs(yaw_delta_a) / max_yaw_step_rad)),
            int(np.ceil(abs(yaw_delta_b) / max_yaw_step_rad)),
        )
        fractions = np.arange(1, steps + 1, dtype=np.float64) / steps
        positions_a = start_a[:2] + fractions[:, None] * (
            end_a[:2] - start_a[:2]
        )
        positions_b = start_b[:2] + fractions[:, None] * (
            end_b[:2] - start_b[:2]
        )
        yaws_a = wrap_angle(start_a[2] + fractions * yaw_delta_a)
        yaws_b = wrap_angle(start_b[2] + fractions * yaw_delta_b)
        dense_a.extend(np.column_stack((positions_a, yaws_a)))
        dense_b.extend(np.column_stack((positions_b, yaws_b)))
    return (
        np.asarray(dense_a, dtype=np.float64),
        np.asarray(dense_b, dtype=np.float64),
    )


def _dense_synchronized_sweeps_intersect(
    footprint_a: Footprint,
    dense_poses_a: np.ndarray,
    footprint_b: Footprint,
    dense_poses_b: np.ndarray,
) -> bool:
    clearances = trajectory_signed_clearances(
        footprint_a,
        dense_poses_a,
        footprint_b,
        dense_poses_b,
    )
    if np.any(clearances <= 0.0):
        return True

    rotation_radius_a = _rotation_motion_radius(footprint_a)
    rotation_radius_b = _rotation_motion_radius(footprint_b)

    def interval_intersects(
        start_a: np.ndarray,
        end_a: np.ndarray,
        start_b: np.ndarray,
        end_b: np.ndarray,
        start_clearance: float,
        end_clearance: float,
        depth: int,
    ) -> bool:
        relative_translation = (end_b[:2] - start_b[:2]) - (
            end_a[:2] - start_a[:2]
        )
        motion_bound = float(np.linalg.norm(relative_translation)) + (
            rotation_radius_a
            * abs(float(wrap_angle(end_a[2] - start_a[2])))
        ) + (
            rotation_radius_b
            * abs(float(wrap_angle(end_b[2] - start_b[2])))
        )
        if max(start_clearance, end_clearance) > motion_bound:
            return False
        if depth >= 20:
            return True

        midpoint_a = np.empty(3, dtype=np.float64)
        midpoint_b = np.empty(3, dtype=np.float64)
        midpoint_a[:2] = 0.5 * (start_a[:2] + end_a[:2])
        midpoint_b[:2] = 0.5 * (start_b[:2] + end_b[:2])
        midpoint_a[2] = wrap_angle(
            start_a[2] + 0.5 * float(wrap_angle(end_a[2] - start_a[2]))
        )
        midpoint_b[2] = wrap_angle(
            start_b[2] + 0.5 * float(wrap_angle(end_b[2] - start_b[2]))
        )
        midpoint_clearance = signed_clearance(
            footprint_a,
            midpoint_a,
            footprint_b,
            midpoint_b,
        )
        if midpoint_clearance <= 0.0:
            return True
        return interval_intersects(
            start_a,
            midpoint_a,
            start_b,
            midpoint_b,
            start_clearance,
            midpoint_clearance,
            depth + 1,
        ) or interval_intersects(
            midpoint_a,
            end_a,
            midpoint_b,
            end_b,
            midpoint_clearance,
            end_clearance,
            depth + 1,
        )

    return any(
        interval_intersects(
            dense_poses_a[index],
            dense_poses_a[index + 1],
            dense_poses_b[index],
            dense_poses_b[index + 1],
            float(clearances[index]),
            float(clearances[index + 1]),
            0,
        )
        for index in range(dense_poses_a.shape[0] - 1)
    )


def synchronized_sweeps_intersect(
    footprint_a: Footprint,
    poses_a: Any,
    footprint_b: Footprint,
    poses_b: Any,
    *,
    grid: GridSpec,
) -> bool:
    """Conservatively reject contact anywhere between synchronized SE(2) frames."""

    if not isinstance(
        footprint_a, (CircleFootprint, RectangleFootprint)
    ) or not isinstance(footprint_b, (CircleFootprint, RectangleFootprint)):
        raise TypeError("footprints must be CircleFootprint or RectangleFootprint")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    resolution_m = _finite_real(grid.resolution_m, name="grid.resolution_m")
    if resolution_m <= 0.0:
        raise ValueError("grid.resolution_m must be positive")
    pose_array_a = _poses(poses_a, name="poses_a")
    pose_array_b = _poses(poses_b, name="poses_b")
    dense_a, dense_b = _densify_synchronized_pose_sequences(
        pose_array_a,
        pose_array_b,
        max_translation_step_m=0.5 * resolution_m,
        max_yaw_step_rad=np.deg2rad(5.0),
    )
    return _dense_synchronized_sweeps_intersect(
        footprint_a,
        dense_a,
        footprint_b,
        dense_b,
    )


def swept_footprint_intersects_occupancy(
    footprint: Footprint,
    poses: Any,
    occupancy: Any,
    *,
    grid: GridSpec,
) -> bool:
    """Certify a continuous sweep against closed occupied grid-cell rectangles."""

    if not isinstance(footprint, (CircleFootprint, RectangleFootprint)):
        raise TypeError("footprint must be CircleFootprint or RectangleFootprint")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    resolution_m = _finite_real(grid.resolution_m, name="grid.resolution_m")
    if resolution_m <= 0.0:
        raise ValueError("grid.resolution_m must be positive")
    pose_array = _poses(poses, name="poses")
    if pose_array.shape[0] == 0:
        raise ValueError("poses must not be empty")
    occupancy_array = np.asarray(occupancy)
    if (
        occupancy_array.shape != (grid.height, grid.width)
        or occupancy_array.dtype.kind not in "biuf"
    ):
        raise ValueError("occupancy must be a numeric grid-shaped array")
    if occupancy_array.dtype.kind in "iuf" and not np.isfinite(
        occupancy_array
    ).all():
        raise ValueError("occupancy must contain only finite values")
    occupied_indices = np.argwhere(occupancy_array != 0)
    if occupied_indices.size == 0:
        return False

    dense_poses = _densify_pose_sequence(
        pose_array,
        max_translation_step_m=0.5 * resolution_m,
        max_yaw_step_rad=np.deg2rad(5.0),
    )
    occupied_centers = grid_to_world(occupied_indices, grid)
    broadphase_margin = _sweep_motion_radius(footprint) + (
        resolution_m / np.sqrt(2.0)
    )
    cell_footprint = RectangleFootprint(
        resolution_m,
        resolution_m,
    )
    pose_segments = (
        (dense_poses,)
        if dense_poses.shape[0] == 1
        else tuple(
            dense_poses[index : index + 2]
            for index in range(dense_poses.shape[0] - 1)
        )
    )
    for segment_poses in pose_segments:
        candidate_mask = (
            (
                occupied_centers[:, 0]
                >= np.min(segment_poses[:, 0]) - broadphase_margin
            )
            & (
                occupied_centers[:, 0]
                <= np.max(segment_poses[:, 0]) + broadphase_margin
            )
            & (
                occupied_centers[:, 1]
                >= np.min(segment_poses[:, 1]) - broadphase_margin
            )
            & (
                occupied_centers[:, 1]
                <= np.max(segment_poses[:, 1]) + broadphase_margin
            )
        )
        for center in occupied_centers[candidate_mask]:
            cell_poses = np.zeros_like(segment_poses)
            cell_poses[:, :2] = center
            if _dense_synchronized_sweeps_intersect(
                footprint,
                segment_poses,
                cell_footprint,
                cell_poses,
            ):
                return True
    return False


def prepare_occluder_collision_sweep(
    sweep: OccluderCollisionSweep,
    *,
    grid: GridSpec,
) -> PreparedOccluderCollisionSweep:
    """Prepare reusable dense SE(2) and interval geometry for one raw sweep."""

    if not isinstance(sweep, OccluderCollisionSweep):
        raise TypeError("sweep must be an OccluderCollisionSweep")
    if not isinstance(sweep.footprint, (CircleFootprint, RectangleFootprint)):
        raise TypeError("sweep.footprint must be a Footprint")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    resolution_m = _finite_real(grid.resolution_m, name="grid.resolution_m")
    if resolution_m <= 0.0:
        raise ValueError("grid.resolution_m must be positive")
    poses = _poses(sweep.poses, name="sweep.poses")
    if poses.shape[0] == 0:
        raise ValueError("sweep.poses must not be empty")
    if not isinstance(sweep.rejection_reason, str) or not sweep.rejection_reason:
        raise ValueError("sweep.rejection_reason must be non-empty")

    dense_poses = _densify_pose_sequence(
        poses,
        max_translation_step_m=0.5 * resolution_m,
        max_yaw_step_rad=(
            np.inf
            if isinstance(sweep.footprint, CircleFootprint)
            else np.deg2rad(5.0)
        ),
    )
    motion_radius_m = _rotation_motion_radius(sweep.footprint)
    interval_motion_bounds_m = np.asarray(
        [
            _pose_interval_motion_bound(
                start_pose,
                end_pose,
                motion_radius_m=motion_radius_m,
            )
            for start_pose, end_pose in zip(
                dense_poses[:-1], dense_poses[1:], strict=True
            )
        ],
        dtype=np.float64,
    )
    return PreparedOccluderCollisionSweep(
        footprint=sweep.footprint,
        dense_poses=dense_poses,
        interval_motion_bounds_m=interval_motion_bounds_m,
        rejection_reason=sweep.rejection_reason,
        grid=grid,
    )


def _validate_prepared_collision_sweep(
    sweep: PreparedOccluderCollisionSweep,
    *,
    grid: GridSpec,
) -> None:
    if sweep.preparation_version != OCCLUDER_COLLISION_SWEEP_PREPARATION_VERSION:
        raise ValueError("prepared collision sweep preparation version mismatch")
    if sweep.grid != grid:
        raise ValueError("prepared collision sweep grid mismatch")


def _prepared_intersects_robot_sweep(
    occluder_footprint: Footprint,
    occluder_pose: np.ndarray,
    sweep: PreparedOccluderCollisionSweep,
) -> bool:
    """Apply the unchanged signed-clearance recursion to prepared geometry."""

    dense_robot_poses = sweep.dense_poses
    sweep_radius_m = _sweep_motion_radius(sweep.footprint)
    occluder_radius_m = _sweep_motion_radius(occluder_footprint)
    center_min = np.min(dense_robot_poses[:, :2], axis=0)
    center_max = np.max(dense_robot_poses[:, :2], axis=0)
    coordinate_scale = max(
        1.0,
        float(np.max(np.abs(dense_robot_poses[:, :2]))),
        float(np.max(np.abs(occluder_pose[:2]))),
        sweep_radius_m + occluder_radius_m,
    )
    expanded_radius_m = (
        sweep_radius_m
        + occluder_radius_m
        + 64.0 * np.finfo(np.float64).eps * coordinate_scale
    )
    if bool(
        occluder_pose[0] < center_min[0] - expanded_radius_m
        or occluder_pose[0] > center_max[0] + expanded_radius_m
        or occluder_pose[1] < center_min[1] - expanded_radius_m
        or occluder_pose[1] > center_max[1] + expanded_radius_m
    ):
        return False
    center_distances_m = np.linalg.norm(
        dense_robot_poses[:, :2] - occluder_pose[:2],
        axis=1,
    )
    clearance_lower_bounds_m = center_distances_m - expanded_radius_m
    if dense_robot_poses.shape[0] == 1:
        if clearance_lower_bounds_m[0] > 0.0:
            return False
        return signed_clearance(
            occluder_footprint,
            occluder_pose,
            sweep.footprint,
            dense_robot_poses[0],
        ) <= 0.0

    unresolved_intervals = (
        np.maximum(
            clearance_lower_bounds_m[:-1],
            clearance_lower_bounds_m[1:],
        )
        <= sweep.interval_motion_bounds_m
    )
    exact_endpoint_mask = clearance_lower_bounds_m <= 0.0
    exact_endpoint_mask[:-1] |= unresolved_intervals
    exact_endpoint_mask[1:] |= unresolved_intervals
    clearances = clearance_lower_bounds_m.copy()
    for index in np.flatnonzero(exact_endpoint_mask):
        clearance = signed_clearance(
            occluder_footprint,
            occluder_pose,
            sweep.footprint,
            dense_robot_poses[index],
        )
        if clearance <= 0.0:
            return True
        clearances[index] = clearance

    robot_radius = _rotation_motion_radius(sweep.footprint)

    def interval_intersects(
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        start_clearance: float,
        end_clearance: float,
        depth: int,
        motion_bound: float | None = None,
    ) -> bool:
        yaw_delta = float(wrap_angle(end_pose[2] - start_pose[2]))
        if motion_bound is None:
            motion_bound = _pose_interval_motion_bound(
                start_pose,
                end_pose,
                motion_radius_m=robot_radius,
            )
        # Signed clearance is 1-Lipschitz under this conservative rigid-body
        # displacement bound. Either endpoint can therefore certify the whole
        # interval without additional samples.
        if max(start_clearance, end_clearance) > motion_bound:
            return False
        if depth >= 20:
            return True

        midpoint = np.empty(3, dtype=np.float64)
        midpoint[:2] = 0.5 * (start_pose[:2] + end_pose[:2])
        midpoint[2] = wrap_angle(start_pose[2] + 0.5 * yaw_delta)
        midpoint_clearance = signed_clearance(
            occluder_footprint,
            occluder_pose,
            sweep.footprint,
            midpoint,
        )
        if midpoint_clearance <= 0.0:
            return True
        return interval_intersects(
            start_pose,
            midpoint,
            start_clearance,
            midpoint_clearance,
            depth + 1,
        ) or interval_intersects(
            midpoint,
            end_pose,
            midpoint_clearance,
            end_clearance,
            depth + 1,
        )

    return any(
        interval_intersects(
            dense_robot_poses[index],
            dense_robot_poses[index + 1],
            float(clearances[index]),
            float(clearances[index + 1]),
            0,
            float(sweep.interval_motion_bounds_m[index]),
        )
        for index in np.flatnonzero(unresolved_intervals)
    )


def _intersects_robot_sweep(
    occluder_footprint: Footprint,
    occluder_pose: np.ndarray,
    robot_footprint: Footprint,
    robot_poses: np.ndarray,
    *,
    grid: GridSpec,
) -> bool:
    """Conservatively certify clearance over an interpolated robot sweep."""

    prepared = prepare_occluder_collision_sweep(
        OccluderCollisionSweep(
            footprint=robot_footprint,
            poses=robot_poses,
            rejection_reason="_intersects_robot_sweep",
        ),
        grid=grid,
    )
    return _prepared_intersects_robot_sweep(
        occluder_footprint,
        occluder_pose,
        prepared,
    )


def _normalize_collision_sweeps(
    collision_sweeps: Any,
) -> tuple[OccluderCollisionSweep | PreparedOccluderCollisionSweep, ...]:
    if not isinstance(collision_sweeps, (list, tuple)) or not collision_sweeps:
        raise ValueError("collision_sweeps must be a non-empty sequence")
    normalized = []
    for index, sweep in enumerate(collision_sweeps):
        if isinstance(sweep, PreparedOccluderCollisionSweep):
            normalized.append(sweep)
            continue
        if not isinstance(sweep, OccluderCollisionSweep):
            raise TypeError(
                f"collision_sweeps[{index}] must be raw or prepared "
                "OccluderCollisionSweep"
            )
        if not isinstance(sweep.footprint, (CircleFootprint, RectangleFootprint)):
            raise TypeError(
                f"collision_sweeps[{index}].footprint must be a Footprint"
            )
        poses = _poses(
            sweep.poses,
            name=f"collision_sweeps[{index}].poses",
        )
        if poses.shape[0] == 0:
            raise ValueError(
                f"collision_sweeps[{index}].poses must not be empty"
            )
        if not isinstance(sweep.rejection_reason, str) or not sweep.rejection_reason:
            raise ValueError(
                f"collision_sweeps[{index}].rejection_reason must be non-empty"
            )
        normalized.append(
            OccluderCollisionSweep(
                footprint=sweep.footprint,
                poses=poses,
                rejection_reason=sweep.rejection_reason,
            )
        )
    return tuple(normalized)


def _normalized_collision_sweep_rejection_reason(
    occluder_footprint: Footprint,
    occluder_pose: np.ndarray,
    collision_sweeps: tuple[
        OccluderCollisionSweep | PreparedOccluderCollisionSweep, ...
    ],
    *,
    grid: GridSpec,
) -> str | None:
    for sweep in collision_sweeps:
        if isinstance(sweep, PreparedOccluderCollisionSweep):
            _validate_prepared_collision_sweep(sweep, grid=grid)
    for sweep in collision_sweeps:
        if isinstance(sweep, PreparedOccluderCollisionSweep):
            intersects = _prepared_intersects_robot_sweep(
                occluder_footprint,
                occluder_pose,
                sweep,
            )
        else:
            intersects = _intersects_robot_sweep(
                occluder_footprint,
                occluder_pose,
                sweep.footprint,
                sweep.poses,
                grid=grid,
            )
        if intersects:
            return sweep.rejection_reason
    return None


def occluder_collision_sweep_rejection_reason(
    occluder_footprint: Footprint,
    occluder_pose: Any,
    collision_sweeps: Any,
    *,
    grid: GridSpec,
) -> str | None:
    """Return the first deterministic full-motion collision reason, if any."""

    pose = _vector(occluder_pose, name="occluder_pose", size=3)
    normalized = _normalize_collision_sweeps(collision_sweeps)
    return _normalized_collision_sweep_rejection_reason(
        occluder_footprint,
        pose,
        normalized,
        grid=grid,
    )


def build_joint_occluder_schedule(
    *,
    types: tuple[str, ...] | list[str],
    max_candidates: int,
    rng: np.random.Generator,
) -> tuple[JointOccluderParameters, ...]:
    """Build a seeded, stratified stream over the frozen joint design."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    count = _positive_integer(max_candidates, name="max_candidates")
    type_values = tuple(types)
    if not type_values or len(set(type_values)) != len(type_values):
        raise ValueError("types must be a non-empty unique sequence")
    allowed = {"wall", "shelf", "pillar"}
    if any(kind not in allowed for kind in type_values):
        raise ValueError("types must contain only wall/shelf/pillar")

    offsets = (0.5, 0.3, 0.7, 0.1, 0.9)
    dimensions = (0.0, 0.5, 1.0, 0.25, 0.75)
    angles = (0.0, -0.5, 0.5, -0.95, 0.95)
    conflict_times = (0.5, 0.25, 0.75, 0.0, 1.0)
    base = []
    for side_index, side in enumerate((-1, 1)):
        for type_index, kind in enumerate(type_values):
            for rank in range(5):
                base.append(
                    JointOccluderParameters(
                        occluder_type=kind,
                        side=side,
                        offset_quantile=offsets[rank],
                        dimension_quantile=dimensions[
                            (rank + type_index + side_index) % 5
                        ],
                        angle_multiplier=angles[
                            (rank + 2 * type_index + side_index) % 5
                        ],
                        conflict_time_quantile=conflict_times[
                            (rank + 2 * type_index + side_index) % 5
                        ],
                    )
                )

    feasible_templates = (
        (0.3, 0.0),
        (0.3, 0.25),
        (0.3, 0.5),
        (0.3, 0.7),
        (0.4, 0.0),
        (0.4, 0.25),
        (0.4, 0.5),
        (0.4, 0.7),
    )
    scheduled = []
    if "pillar" in type_values:
        for template_index in rng.permutation(len(feasible_templates)):
            offset_quantile, angle_magnitude = feasible_templates[
                int(template_index)
            ]
            for side in (-1, 1):
                scheduled.append(
                    JointOccluderParameters(
                        occluder_type="pillar",
                        side=side,
                        offset_quantile=offset_quantile,
                        dimension_quantile=0.0,
                        angle_multiplier=-angle_magnitude * side,
                        conflict_time_quantile=1.0,
                    )
                )
    while len(scheduled) < count:
        order = rng.permutation(len(base))
        scheduled.extend(base[int(index)] for index in order)
    return tuple(scheduled[:count])


def propose_environment_occluder_geometry(
    *,
    static_occupancy: Any,
    grid: GridSpec,
    sensor_pose: Any,
    conflict_point: Any,
    trajectory_normal: Any,
    collision_sweeps: Any,
    config: Mapping[str, Any],
    parameters: JointOccluderParameters,
    proposal_index: int,
) -> OccluderGeometryCandidate:
    """Construct and cheaply validate one target-independent occluder."""

    if not isinstance(parameters, JointOccluderParameters):
        raise TypeError("parameters must be JointOccluderParameters")
    if parameters.side not in (-1, 1):
        raise ValueError("parameters.side must be -1 or 1")
    if isinstance(proposal_index, (bool, np.bool_)) or not isinstance(
        proposal_index, (int, np.integer)
    ):
        raise TypeError("proposal_index must be an integer")
    proposal_index = int(proposal_index)
    if proposal_index < 0:
        raise ValueError("proposal_index must be non-negative")
    quantiles = (
        parameters.offset_quantile,
        parameters.dimension_quantile,
        parameters.conflict_time_quantile,
    )
    if any(not 0.0 <= float(value) <= 1.0 for value in quantiles):
        raise ValueError("joint proposal quantiles must lie in [0, 1]")

    normalized = normalize_occluder_config(config)
    if parameters.occluder_type not in normalized["types"]:
        raise ValueError("proposal occluder type is disabled by config")
    occupancy = np.asarray(static_occupancy)
    if occupancy.shape != (grid.height, grid.width) or occupancy.dtype.kind not in "biuf":
        raise ValueError("static_occupancy must be a numeric grid-shaped array")
    if not np.isfinite(occupancy).all():
        raise ValueError("static_occupancy must contain only finite values")
    sensor = _vector(sensor_pose, name="sensor_pose", size=3)
    conflict = _vector(conflict_point, name="conflict_point", size=2)
    normal = _vector(trajectory_normal, name="trajectory_normal", size=2)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        raise ValueError("trajectory_normal must be non-zero")
    normal /= normal_norm
    normalized_collision_sweeps = _normalize_collision_sweeps(
        collision_sweeps
    )

    kind = parameters.occluder_type
    dimensions = normalized[kind]
    length_m = _range_quantile(
        dimensions["length_range_m"], parameters.dimension_quantile
    )
    width_m = _range_quantile(
        dimensions["width_range_m"], parameters.dimension_quantile
    )
    normal_offset = parameters.side * _range_quantile(
        normalized["normal_offset_range_m"], parameters.offset_quantile
    )
    center = conflict + normal_offset * normal
    sensor_to_center = center - sensor[:2]
    if float(np.linalg.norm(sensor_to_center)) <= 1e-9:
        raise OccluderSamplingError(
            "occluder_los_degenerate", attempts=1,
            rejection_reasons={"occluder_los_degenerate": 1},
        )
    yaw = wrap_angle(float(np.arctan2(sensor_to_center[1], sensor_to_center[0])) + 0.5 * np.pi)
    pose = np.asarray([center[0], center[1], yaw], dtype=np.float64)
    footprint = RectangleFootprint(length_m=length_m, width_m=width_m)
    mask = rasterize_footprint(footprint, pose, grid)
    reason = None
    if not _inside_grid(footprint, pose, grid):
        reason = "occluder_out_of_bounds"
    elif np.any(mask & (occupancy != 0)):
        reason = "occluder_static_overlap"
    else:
        reason = _normalized_collision_sweep_rejection_reason(
            footprint,
            pose,
            normalized_collision_sweeps,
            grid=grid,
        )
    if reason is not None:
        raise OccluderSamplingError(
            reason, attempts=1, rejection_reasons={reason: 1}
        )

    pose32 = pose.astype(np.float32)
    occluder_id = "occluder-" + stable_digest(
        kind,
        proposal_index,
        *(f"{value:.9f}" for value in pose32),
        f"{length_m:.9f}",
        f"{width_m:.9f}",
        size=12,
    )
    return OccluderGeometryCandidate(
        occluder={
            "occluder_id": occluder_id,
            "type": kind,
            "pose": [float(value) for value in pose32],
            "length_m": float(length_m),
            "width_m": float(width_m),
            "geometry_source": "generator_config",
            "placement_strategy": "joint_occluder_first_v2",
            "normal_offset_m": float(normal_offset),
            "offset_quantile": float(parameters.offset_quantile),
            "dimension_quantile": float(parameters.dimension_quantile),
            "conflict_time_quantile": float(parameters.conflict_time_quantile),
            "proposal_index": proposal_index,
        },
        footprint=footprint,
        pose=pose32,
        mask=mask,
        proposal_index=proposal_index,
    )


def validate_environment_occluder_target(
    candidate: OccluderGeometryCandidate,
    *,
    target_history_poses: Any,
    target_future_poses: Any,
    target_footprint: Footprint,
    grid: GridSpec,
) -> OccluderPlacement:
    """Promote a geometry candidate after exact target-sweep clearance checks."""

    if not isinstance(candidate, OccluderGeometryCandidate):
        raise TypeError("candidate must be OccluderGeometryCandidate")
    target_history = _poses(
        target_history_poses, name="target_history_poses"
    )
    target_future = _poses(target_future_poses, name="target_future_poses")
    if target_history.shape != (grid.history_steps, 3):
        raise ValueError(
            "target_history_poses must have shape "
            f"({grid.history_steps}, 3)"
        )
    if target_future.shape != (grid.future_steps, 3):
        raise ValueError(
            "target_future_poses must have shape "
            f"({grid.future_steps}, 3)"
        )
    target_poses = np.vstack((target_history, target_future))
    endpoint_clearances = trajectory_signed_clearances(
        candidate.footprint,
        np.broadcast_to(candidate.pose, target_poses.shape),
        target_footprint,
        target_poses,
    )
    if np.any(endpoint_clearances <= 0.0):
        raise OccluderSamplingError(
            "occluder_target_collision",
            attempts=1,
            rejection_reasons={"occluder_target_collision": 1},
        )
    reason = occluder_collision_sweep_rejection_reason(
        candidate.footprint,
        candidate.pose,
        (
            OccluderCollisionSweep(
                footprint=target_footprint,
                poses=target_poses,
                rejection_reason="occluder_target_collision",
            ),
        ),
        grid=grid,
    )
    if reason is not None:
        raise OccluderSamplingError(
            reason,
            attempts=1,
            rejection_reasons={reason: 1},
        )
    return OccluderPlacement(
        occluder=dict(candidate.occluder),
        footprint=candidate.footprint,
        pose=candidate.pose.copy(),
        mask=candidate.mask.copy(),
        attempt=candidate.proposal_index + 1,
        rejection_reasons={},
    )


def align_environment_occluder_to_target_los(
    candidate: OccluderGeometryCandidate,
    *,
    static_occupancy: Any,
    grid: GridSpec,
    sensor_pose: Any,
    trajectory_normal: Any,
    target_current_pose: Any,
    collision_sweeps: Any,
) -> OccluderGeometryCandidate:
    """Keep a proposed normal band while solving its LOS tangential coordinate."""

    if not isinstance(candidate, OccluderGeometryCandidate):
        raise TypeError("candidate must be OccluderGeometryCandidate")
    occupancy = np.asarray(static_occupancy)
    if occupancy.shape != (grid.height, grid.width) or occupancy.dtype.kind not in "biuf":
        raise ValueError("static_occupancy must be a numeric grid-shaped array")
    if not np.isfinite(occupancy).all():
        raise ValueError("static_occupancy must contain only finite values")
    sensor = _vector(sensor_pose, name="sensor_pose", size=3)
    normal = _vector(trajectory_normal, name="trajectory_normal", size=2)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        raise ValueError("trajectory_normal must be non-zero")
    normal /= normal_norm
    target_current = _vector(
        target_current_pose, name="target_current_pose", size=3
    )
    los = target_current[:2] - sensor[:2]
    normal_component = float(np.dot(los, normal))
    if abs(normal_component) <= 1e-9:
        raise OccluderSamplingError(
            "occluder_los_degenerate",
            attempts=1,
            rejection_reasons={"occluder_los_degenerate": 1},
        )
    desired_coordinate = float(np.dot(candidate.pose[:2], normal))
    fraction = (
        desired_coordinate - float(np.dot(sensor[:2], normal))
    ) / normal_component
    if not 0.0 < fraction < 1.0:
        raise OccluderSamplingError(
            "occluder_offset_outside_line_of_sight",
            attempts=1,
            rejection_reasons={"occluder_offset_outside_line_of_sight": 1},
        )
    center = sensor[:2] + fraction * los
    yaw = wrap_angle(float(np.arctan2(los[1], los[0])) + 0.5 * np.pi)
    pose = np.asarray([center[0], center[1], yaw], dtype=np.float64)
    mask = rasterize_footprint(candidate.footprint, pose, grid)
    normalized_collision_sweeps = _normalize_collision_sweeps(
        collision_sweeps
    )
    reason = None
    if not _inside_grid(candidate.footprint, pose, grid):
        reason = "occluder_out_of_bounds"
    elif np.any(mask & (occupancy != 0)):
        reason = "occluder_static_overlap"
    else:
        reason = _normalized_collision_sweep_rejection_reason(
            candidate.footprint,
            pose,
            normalized_collision_sweeps,
            grid=grid,
        )
    if reason is not None:
        raise OccluderSamplingError(
            reason, attempts=1, rejection_reasons={reason: 1}
        )

    pose32 = pose.astype(np.float32)
    occluder = dict(candidate.occluder)
    occluder.update(
        {
            "occluder_id": "occluder-"
            + stable_digest(
                occluder["type"],
                candidate.proposal_index,
                *(f"{value:.9f}" for value in pose32),
                f"{occluder['length_m']:.9f}",
                f"{occluder['width_m']:.9f}",
                size=12,
            ),
            "pose": [float(value) for value in pose32],
            "line_of_sight_fraction": float(fraction),
        }
    )
    return OccluderGeometryCandidate(
        occluder=occluder,
        footprint=candidate.footprint,
        pose=pose32,
        mask=mask,
        proposal_index=candidate.proposal_index,
    )


def align_environment_occluder_to_target_los_envelope(
    *,
    occluder_type: str,
    normal_offset_m: float,
    proposal_index: int,
    static_occupancy: Any,
    grid: GridSpec,
    sensor_pose: Any,
    conflict_point: Any,
    trajectory_normal: Any,
    target_visibility_pose_sequences: Any,
    target_footprint: Footprint,
    current_context_poses: Mapping[str, np.ndarray],
    current_context_footprints: Mapping[str, Footprint],
    collision_sweeps: Any,
    config: Mapping[str, Any],
    min_contiguous_visible_frames: int,
) -> tuple[OccluderPlacement, tuple[np.ndarray, ...]]:
    """Find one thin, asymmetric occluder covering multiple current LOS rays.

    Length and width remain inside the configured range for ``occluder_type``.
    Candidate centres lie on the envelope between the per-target LOS
    intersections at one frozen trajectory-normal coordinate.  Every returned
    candidate clears every supplied full-motion sweep and the static map while
    keeping every target currently hidden and eventually visible.  Visibility
    paths are deliberately separate from collision sweeps because they begin
    at the current frame rather than the oldest history frame.
    """

    normalized = normalize_occluder_config(config)
    if occluder_type not in normalized["types"]:
        raise ValueError("occluder_type is disabled by config")
    offset = _finite_real(normal_offset_m, name="normal_offset_m")
    offset_bounds = normalized["normal_offset_range_m"]
    if not offset_bounds[0] <= abs(offset) <= offset_bounds[1]:
        raise ValueError("normal_offset_m lies outside the configured range")
    if isinstance(proposal_index, (bool, np.bool_)) or not isinstance(
        proposal_index, (int, np.integer)
    ):
        raise TypeError("proposal_index must be an integer")
    proposal_index = int(proposal_index)
    if proposal_index < 0:
        raise ValueError("proposal_index must be non-negative")
    visible_frames = _positive_integer(
        min_contiguous_visible_frames,
        name="min_contiguous_visible_frames",
    )

    occupancy = np.asarray(static_occupancy)
    if (
        occupancy.shape != (grid.height, grid.width)
        or occupancy.dtype.kind not in "biuf"
    ):
        raise ValueError("static_occupancy must be a numeric grid-shaped array")
    if not np.isfinite(occupancy).all():
        raise ValueError("static_occupancy must contain only finite values")
    base_occupied = np.asarray(occupancy != 0, dtype=bool)
    sensor = _vector(sensor_pose, name="sensor_pose", size=3)
    conflict = _vector(conflict_point, name="conflict_point", size=2)
    normal = _vector(trajectory_normal, name="trajectory_normal", size=2)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        raise ValueError("trajectory_normal must be non-zero")
    normal /= normal_norm
    if set(current_context_poses) != set(current_context_footprints):
        raise ValueError("current context pose and footprint keys must match")
    current_context_pose_arrays = {
        object_id: _vector(
            pose,
            name=f"current_context_poses[{object_id!r}]",
            size=3,
        )
        for object_id, pose in sorted(current_context_poses.items())
    }
    pose_sequences = tuple(
        _poses(poses, name="target_visibility_pose_sequences")
        for poses in target_visibility_pose_sequences
    )
    if len(pose_sequences) < 2:
        raise ValueError(
            "target_visibility_pose_sequences must contain at least two paths"
        )
    if any(
        poses.shape != (grid.future_steps + 1, 3)
        for poses in pose_sequences
    ):
        raise ValueError(
            "target_visibility_pose_sequences must contain current+future paths "
            f"with shape ({grid.future_steps + 1}, 3)"
        )
    normalized_collision_sweeps = _normalize_collision_sweeps(
        collision_sweeps
    )

    desired_coordinate = float(
        np.dot(conflict + offset * normal, normal)
    )
    sensor_coordinate = float(np.dot(sensor[:2], normal))
    intersections = []
    los_directions = []
    for poses in pose_sequences:
        los = poses[0, :2] - sensor[:2]
        los_norm = float(np.linalg.norm(los))
        denominator = float(np.dot(los, normal))
        if los_norm <= 1e-9 or abs(denominator) <= 1e-9:
            raise OccluderSamplingError(
                "occluder_multilos_los_degenerate",
                attempts=1,
                rejection_reasons={"occluder_multilos_los_degenerate": 1},
            )
        fraction = (desired_coordinate - sensor_coordinate) / denominator
        if not 0.0 < fraction < 1.0:
            raise OccluderSamplingError(
                "occluder_multilos_offset_outside_line_of_sight",
                attempts=1,
                rejection_reasons={
                    "occluder_multilos_offset_outside_line_of_sight": 1
                },
            )
        intersections.append(sensor[:2] + fraction * los)
        los_directions.append(los / los_norm)

    mean_direction = np.sum(los_directions, axis=0)
    yaw_directions = list(los_directions)
    if float(np.linalg.norm(mean_direction)) > 1e-9:
        yaw_directions.append(mean_direction / np.linalg.norm(mean_direction))
    yaw_candidates = []
    for direction in yaw_directions:
        yaw = float(
            wrap_angle(np.arctan2(direction[1], direction[0]) + 0.5 * np.pi)
        )
        if not any(
            np.isclose(yaw, existing, atol=1e-12, rtol=0.0)
            for existing in yaw_candidates
        ):
            yaw_candidates.append(yaw)

    dimensions = normalized[occluder_type]
    quantiles = (1.0, 0.75, 0.5, 0.25, 0.0)
    length_candidates = tuple(
        dict.fromkeys(
            _range_quantile(dimensions["length_range_m"], quantile)
            for quantile in quantiles
        )
    )
    width_candidates = tuple(
        dict.fromkeys(
            _range_quantile(dimensions["width_range_m"], quantile)
            for quantile in reversed(quantiles)
        )
    )
    alpha_candidates = (
        0.5,
        0.4,
        0.6,
        0.3,
        0.7,
        0.2,
        0.8,
        0.1,
        0.9,
        0.0,
        1.0,
    )
    context_current = np.zeros((grid.height, grid.width), dtype=bool)
    for object_id, pose in current_context_pose_arrays.items():
        context_current |= rasterize_footprint(
            current_context_footprints[object_id], pose, grid
        )

    rejection_reasons: dict[str, int] = {}
    attempts = 0
    for length_m in length_candidates:
        for width_m in width_candidates:
            footprint = RectangleFootprint(length_m, width_m)
            for yaw_index, yaw in enumerate(yaw_candidates):
                for alpha in alpha_candidates:
                    attempts += 1
                    center = (
                        alpha * intersections[0]
                        + (1.0 - alpha) * intersections[1]
                    )
                    pose = np.asarray(
                        [center[0], center[1], yaw], dtype=np.float64
                    )
                    mask = rasterize_footprint(footprint, pose, grid)
                    reason = None
                    if not _inside_grid(footprint, pose, grid):
                        reason = "occluder_out_of_bounds"
                    elif np.any(mask & base_occupied):
                        reason = "occluder_static_overlap"
                    else:
                        reason = _normalized_collision_sweep_rejection_reason(
                            footprint,
                            pose,
                            normalized_collision_sweeps,
                            grid=grid,
                        )
                    if reason is not None:
                        _record_rejection(rejection_reasons, reason)
                        continue

                    visibility = raycast_visibility(
                        base_occupied | context_current | mask,
                        grid,
                        sensor_pose=sensor,
                    )
                    sequences = tuple(
                        footprint_visibility_sequence(
                            target_footprint,
                            poses,
                            visibility,
                            grid,
                        )
                        for poses in pose_sequences
                    )
                    if any(
                        bool(sequence[0])
                        or not bool(sequence[-1])
                        or not has_continuous_emergence(
                            sequence,
                            min_visible_frames=visible_frames,
                        )
                        for sequence in sequences
                    ):
                        _record_rejection(
                            rejection_reasons,
                            "occluder_multilos_visibility_invalid",
                        )
                        continue

                    pose32 = pose.astype(np.float32)
                    occluder = {
                        "occluder_id": "occluder-"
                        + stable_digest(
                            occluder_type,
                            proposal_index,
                            JOINT_MULTI_LOS_PLACEMENT_STRATEGY_VERSION,
                            *(f"{value:.9f}" for value in pose32),
                            f"{length_m:.9f}",
                            f"{width_m:.9f}",
                            size=12,
                        ),
                        "type": occluder_type,
                        "pose": [float(value) for value in pose32],
                        "length_m": float(length_m),
                        "width_m": float(width_m),
                        "geometry_source": "generator_config",
                        "placement_strategy": (
                            JOINT_MULTI_LOS_PLACEMENT_STRATEGY_VERSION
                        ),
                        "normal_offset_m": float(offset),
                        "proposal_index": proposal_index,
                        "hidden_los_count": len(pose_sequences),
                        "center_alpha": float(alpha),
                        "yaw_source_index": yaw_index,
                    }
                    return (
                        OccluderPlacement(
                            occluder=occluder,
                            footprint=footprint,
                            pose=pose32,
                            mask=mask.copy(),
                            attempt=attempts,
                            rejection_reasons=dict(
                                sorted(rejection_reasons.items())
                            ),
                        ),
                        tuple(sequence.copy() for sequence in sequences),
                    )

    raise OccluderSamplingError(
        "occluder_multilos_unavailable",
        attempts=attempts,
        rejection_reasons=rejection_reasons,
    )


def sample_environment_occluder(
    *,
    static_occupancy: Any,
    grid: GridSpec,
    sensor_pose: Any,
    conflict_point: Any,
    trajectory_normal: Any,
    target_history_poses: Any,
    target_current_pose: Any,
    target_future_poses: Any,
    target_footprint: Footprint,
    collision_sweeps: Any,
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
    target_history = _poses(
        target_history_poses, name="target_history_poses"
    )
    target_current = _vector(target_current_pose, name="target_current_pose", size=3)
    target_future = _poses(target_future_poses, name="target_future_poses")
    if target_history.shape != (grid.history_steps, 3):
        raise ValueError(
            "target_history_poses must have shape "
            f"({grid.history_steps}, 3)"
        )
    if target_future.shape != (grid.future_steps, 3):
        raise ValueError(
            "target_future_poses must have shape "
            f"({grid.future_steps}, 3)"
        )
    if not np.array_equal(target_current, target_history[-1]):
        raise ValueError(
            "target_current_pose must equal the final target history pose"
        )
    base_collision_sweeps = _normalize_collision_sweeps(
        collision_sweeps
    )
    normalized_collision_sweeps = (
        *base_collision_sweeps,
        OccluderCollisionSweep(
            footprint=target_footprint,
            poses=np.vstack((target_history, target_future)),
            rejection_reason="occluder_target_collision",
        ),
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
        collision_reason = _normalized_collision_sweep_rejection_reason(
            footprint,
            pose,
            normalized_collision_sweeps,
            grid=grid,
        )
        if collision_reason is not None:
            _record_rejection(rejection_reasons, collision_reason)
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
