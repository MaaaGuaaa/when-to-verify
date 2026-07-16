"""Physical environment-occluder placement with finite deterministic retries."""

from __future__ import annotations

from dataclasses import dataclass
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
    rasterize_footprint,
    rasterize_footprint_sweep,
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
    time_scale_quantile: float
    conflict_time_quantile: float = 0.5


@dataclass(frozen=True)
class OccluderGeometryCandidate:
    """Target-independent geometry that already clears the static world."""

    occluder: dict[str, object]
    footprint: RectangleFootprint
    pose: np.ndarray
    mask: np.ndarray
    proposal_index: int


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


def _intersects_robot_sweep(
    occluder_footprint: Footprint,
    occluder_pose: np.ndarray,
    robot_footprint: Footprint,
    robot_poses: np.ndarray,
    *,
    grid: GridSpec,
) -> bool:
    """Conservatively certify clearance over an interpolated robot sweep."""

    dense_robot_poses = _densify_pose_sequence(
        robot_poses,
        max_translation_step_m=0.5 * grid.resolution_m,
        max_yaw_step_rad=np.deg2rad(5.0),
    )
    clearances = trajectory_signed_clearances(
        occluder_footprint,
        np.tile(occluder_pose, (dense_robot_poses.shape[0], 1)),
        robot_footprint,
        dense_robot_poses,
    )
    if np.any(clearances <= 0.0):
        return True

    if isinstance(robot_footprint, CircleFootprint):
        robot_radius = robot_footprint.radius_m
    else:
        robot_radius = 0.5 * float(
            np.hypot(robot_footprint.length_m, robot_footprint.width_m)
        )

    def interval_intersects(
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        start_clearance: float,
        end_clearance: float,
        depth: int,
    ) -> bool:
        yaw_delta = float(wrap_angle(end_pose[2] - start_pose[2]))
        motion_bound = float(np.linalg.norm(end_pose[:2] - start_pose[:2]))
        motion_bound += robot_radius * abs(yaw_delta)
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
            robot_footprint,
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
        )
        for index in range(dense_robot_poses.shape[0] - 1)
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
    time_scales = (0.5, 0.25, 0.75, 0.0, 1.0)
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
                        time_scale_quantile=time_scales[
                            (rank + type_index + 2 * side_index) % 5
                        ],
                        conflict_time_quantile=conflict_times[
                            (rank + 2 * type_index + side_index) % 5
                        ],
                    )
                )

    feasible_templates = (
        (0.3, 0.0, 0.0),
        (0.3, 0.25, 0.16),
        (0.3, 0.5, 0.25),
        (0.3, 0.7, 0.5),
        (0.4, 0.0, 0.16),
        (0.4, 0.25, 0.25),
        (0.4, 0.5, 0.5),
        (0.4, 0.7, 0.0),
    )
    scheduled = []
    if "pillar" in type_values:
        for template_index in rng.permutation(len(feasible_templates)):
            (
                offset_quantile,
                angle_magnitude,
                time_scale_quantile,
            ) = feasible_templates[int(template_index)]
            for side in (-1, 1):
                scheduled.append(
                    JointOccluderParameters(
                        occluder_type="pillar",
                        side=side,
                        offset_quantile=offset_quantile,
                        dimension_quantile=0.0,
                        angle_multiplier=-angle_magnitude * side,
                        time_scale_quantile=time_scale_quantile,
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
    robot_poses: Any,
    robot_footprint: Footprint,
    context_trajectories: Mapping[str, np.ndarray],
    context_footprints: Mapping[str, Footprint],
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
        parameters.time_scale_quantile,
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
    robot_pose_array = np.vstack(
        (sensor, _poses(robot_poses, name="robot_poses"))
    )
    if set(context_trajectories) != set(context_footprints):
        raise ValueError("context trajectory and footprint keys must match")
    context_pose_arrays = {
        object_id: _poses(poses, name=f"context_trajectories[{object_id!r}]")
        for object_id, poses in sorted(context_trajectories.items())
    }

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
    elif _intersects_robot_sweep(
        footprint,
        pose,
        robot_footprint,
        robot_pose_array,
        grid=grid,
    ):
        reason = "occluder_robot_swept_overlap"
    elif any(
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
        reason = "occluder_context_collision"
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
    target_current_pose: Any,
    target_future_poses: Any,
    target_footprint: Footprint,
) -> OccluderPlacement:
    """Promote a geometry candidate after exact target-sweep clearance checks."""

    if not isinstance(candidate, OccluderGeometryCandidate):
        raise TypeError("candidate must be OccluderGeometryCandidate")
    target_current = _vector(
        target_current_pose, name="target_current_pose", size=3
    )
    target_future = _poses(target_future_poses, name="target_future_poses")
    target_poses = np.vstack((target_current, target_future))
    clearances = trajectory_signed_clearances(
        candidate.footprint,
        np.tile(candidate.pose, (target_poses.shape[0], 1)),
        target_footprint,
        target_poses,
    )
    if np.any(clearances <= 0.0):
        raise OccluderSamplingError(
            "occluder_target_collision",
            attempts=1,
            rejection_reasons={"occluder_target_collision": 1},
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
    robot_poses: Any,
    robot_footprint: Footprint,
    target_current_pose: Any,
    context_trajectories: Mapping[str, np.ndarray],
    context_footprints: Mapping[str, Footprint],
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
    robot_pose_array = np.vstack(
        (sensor, _poses(robot_poses, name="robot_poses"))
    )
    if set(context_trajectories) != set(context_footprints):
        raise ValueError("context trajectory and footprint keys must match")
    context_pose_arrays = {
        object_id: _poses(poses, name=f"context_trajectories[{object_id!r}]")
        for object_id, poses in sorted(context_trajectories.items())
    }
    reason = None
    if not _inside_grid(candidate.footprint, pose, grid):
        reason = "occluder_out_of_bounds"
    elif np.any(mask & (occupancy != 0)):
        reason = "occluder_static_overlap"
    elif _intersects_robot_sweep(
        candidate.footprint,
        pose,
        robot_footprint,
        robot_pose_array,
        grid=grid,
    ):
        reason = "occluder_robot_swept_overlap"
    elif any(
        np.any(
            trajectory_signed_clearances(
                candidate.footprint,
                np.tile(pose, (poses.shape[0], 1)),
                context_footprints[object_id],
                poses,
            )
            <= 0.0
        )
        for object_id, poses in context_pose_arrays.items()
    ):
        reason = "occluder_context_collision"
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
