"""Continuous synchronized collision evidence shared by Long40 generators."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from src.contracts import GridSpec
from src.geometry import (
    CircleFootprint,
    Footprint,
    RectangleFootprint,
    signed_clearance,
    wrap_angle,
)

from .occluder_sampler import synchronized_sweeps_intersect


CONTINUOUS_COLLISION_EVIDENCE_VERSION = (
    "sop05r_continuous_collision_evidence_v1"
)


def _readonly_array(value: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class ContinuousCollisionEvidence:
    version: str
    continuous_collision: bool
    minimum_clearance_m: float
    minimum_clearance_time_s: float
    first_collision_time_s: float | None
    robot_pose_at_first_collision: np.ndarray | None
    target_pose_at_first_collision: np.ndarray | None

    def __post_init__(self) -> None:
        if self.version != CONTINUOUS_COLLISION_EVIDENCE_VERSION:
            raise ValueError("continuous collision evidence version mismatch")
        values = (self.minimum_clearance_m, self.minimum_clearance_time_s)
        if not np.isfinite(values).all() or self.minimum_clearance_time_s < 0.0:
            raise ValueError("continuous collision evidence scalars are invalid")
        if self.continuous_collision:
            if self.first_collision_time_s is None:
                raise ValueError("collision evidence must include first collision time")
            if self.minimum_clearance_m > 0.0:
                raise ValueError(
                    "collision evidence minimum clearance must be nonpositive"
                )
            for name in (
                "robot_pose_at_first_collision",
                "target_pose_at_first_collision",
            ):
                pose = getattr(self, name)
                if (
                    not isinstance(pose, np.ndarray)
                    or pose.shape != (3,)
                    or not np.isfinite(pose).all()
                ):
                    raise ValueError(f"{name} must be a finite pose")
                object.__setattr__(
                    self,
                    name,
                    _readonly_array(pose, dtype=np.dtype(np.float64)),
                )
        elif any(
            value is not None
            for value in (
                self.first_collision_time_s,
                self.robot_pose_at_first_collision,
                self.target_pose_at_first_collision,
            )
        ):
            raise ValueError("noncollision evidence must not include collision fields")


def _interpolate_pose(
    start: np.ndarray,
    end: np.ndarray,
    fraction: float,
) -> np.ndarray:
    result = np.empty(3, dtype=np.float64)
    result[:2] = (1.0 - fraction) * start[:2] + fraction * end[:2]
    result[2] = wrap_angle(
        start[2] + fraction * float(wrap_angle(end[2] - start[2]))
    )
    return result


def _motion_radius(footprint: Footprint) -> float:
    if isinstance(footprint, CircleFootprint):
        return footprint.radius_m
    if isinstance(footprint, RectangleFootprint):
        return 0.5 * float(np.hypot(footprint.length_m, footprint.width_m))
    raise TypeError("footprint must be a supported geometry footprint")


def compute_continuous_collision_evidence(
    *,
    robot_footprint: Footprint,
    robot_poses: np.ndarray,
    target_footprint: Footprint,
    target_poses: np.ndarray,
    dt_s: object,
    spatial_resolution_m: object,
) -> ContinuousCollisionEvidence:
    """Densely evaluate synchronized SE(2) motion and certify contact."""

    dt = _finite_positive(dt_s, name="dt_s")
    resolution = _finite_positive(
        spatial_resolution_m, name="spatial_resolution_m"
    )
    robot = np.asarray(robot_poses)
    target = np.asarray(target_poses)
    if (
        robot.ndim != 2
        or robot.shape[1:] != (3,)
        or target.shape != robot.shape
        or robot.shape[0] < 2
        or robot.dtype.kind not in "iuf"
        or target.dtype.kind not in "iuf"
        or not np.isfinite(robot).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("synchronized poses must be matching finite [T, 3] arrays")
    robot = robot.astype(np.float64)
    target = target.astype(np.float64)
    robot_radius = _motion_radius(robot_footprint)
    target_radius = _motion_radius(target_footprint)
    dense_times = [0.0]
    dense_robot = [robot[0]]
    dense_target = [target[0]]
    for interval, (robot_start, robot_end, target_start, target_end) in enumerate(
        zip(robot[:-1], robot[1:], target[:-1], target[1:], strict=True)
    ):
        robot_motion = float(np.linalg.norm(robot_end[:2] - robot_start[:2]))
        robot_motion += robot_radius * abs(
            float(wrap_angle(robot_end[2] - robot_start[2]))
        )
        target_motion = float(np.linalg.norm(target_end[:2] - target_start[:2]))
        target_motion += target_radius * abs(
            float(wrap_angle(target_end[2] - target_start[2]))
        )
        subdivisions = max(
            1,
            int(np.ceil(robot_motion / resolution)),
            int(np.ceil(target_motion / resolution)),
        )
        for subdivision in range(1, subdivisions + 1):
            fraction = subdivision / subdivisions
            dense_times.append((interval + fraction) * dt)
            dense_robot.append(
                _interpolate_pose(robot_start, robot_end, fraction)
            )
            dense_target.append(
                _interpolate_pose(target_start, target_end, fraction)
            )
    clearances = np.asarray(
        [
            signed_clearance(
                robot_footprint,
                robot_pose,
                target_footprint,
                target_pose,
            )
            for robot_pose, target_pose in zip(
                dense_robot, dense_target, strict=True
            )
        ],
        dtype=np.float64,
    )
    minimum_index = int(np.argmin(clearances))
    colliding = np.flatnonzero(clearances <= 0.0)
    authority_collision = synchronized_sweeps_intersect(
        robot_footprint,
        robot,
        target_footprint,
        target,
        grid=GridSpec(
            height=1,
            width=1,
            history_steps=1,
            future_steps=robot.shape[0] - 1,
            resolution_m=resolution,
        ),
    )
    if authority_collision and colliding.size == 0:
        colliding = np.asarray([minimum_index], dtype=np.int64)
        clearances[minimum_index] = min(0.0, clearances[minimum_index])
    if colliding.size:
        first = int(colliding[0])
        return ContinuousCollisionEvidence(
            version=CONTINUOUS_COLLISION_EVIDENCE_VERSION,
            continuous_collision=True,
            minimum_clearance_m=float(clearances[minimum_index]),
            minimum_clearance_time_s=float(dense_times[minimum_index]),
            first_collision_time_s=float(dense_times[first]),
            robot_pose_at_first_collision=np.asarray(dense_robot[first]),
            target_pose_at_first_collision=np.asarray(dense_target[first]),
        )
    return ContinuousCollisionEvidence(
        version=CONTINUOUS_COLLISION_EVIDENCE_VERSION,
        continuous_collision=False,
        minimum_clearance_m=float(clearances[minimum_index]),
        minimum_clearance_time_s=float(dense_times[minimum_index]),
        first_collision_time_s=None,
        robot_pose_at_first_collision=None,
        target_pose_at_first_collision=None,
    )
