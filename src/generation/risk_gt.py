"""Schema-v3 hidden-object risk ground truth for one candidate trajectory."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from src.contracts import (
    POSE_TIME_LAYOUT_VERSION,
    SCHEMA_VERSION,
    GridSpec,
    LocalTrajectory,
    OracleWorld,
    validate_oracle_world,
)
from src.geometry import (
    CircleFootprint,
    Footprint,
    RectangleFootprint,
    trajectory_signed_clearances,
)

from .dynamic_object_transplant import footprint_from_spec


RISK_GT_VERSION = "hidden_risk_gt_schema3_v1"
_SCHEMA3_VERSION = "3.0.0"


@dataclass(frozen=True)
class RiskGroundTruth:
    """Finite scalar labels and label-only audit identity for one trajectory."""

    schema_version: str
    pose_time_layout_version: str
    collision_label: int
    risk_severity: float
    min_clearance: float
    near_miss: int
    first_collision_time: float | None
    time_to_min_clearance: float | None
    critical_object_id: str | None
    critical_object_type: str | None
    has_hidden_target: bool


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_real(value: Any, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_real(value: Any, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_grid(grid: GridSpec) -> GridSpec:
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    _positive_integer(grid.height, name="grid.height")
    _positive_integer(grid.width, name="grid.width")
    _positive_integer(grid.future_steps, name="grid.future_steps")
    _positive_real(grid.resolution_m, name="grid.resolution_m")
    return grid


def resolve_no_object_clearance_sentinel(grid: GridSpec) -> float:
    """Return the finite physical diagonal of the complete BEV grid."""

    validated = _validate_grid(grid)
    try:
        width_m = float(validated.width) * float(validated.resolution_m)
        height_m = float(validated.height) * float(validated.resolution_m)
        sentinel = float(math.hypot(width_m, height_m))
    except (OverflowError, ValueError) as exc:
        raise ValueError("grid physical diagonal must be finite") from exc
    if not np.isfinite(sentinel) or sentinel <= 0.0:
        raise ValueError("grid physical diagonal must be finite and positive")
    return sentinel


def _validate_trajectory(
    trajectory: LocalTrajectory,
    *,
    grid: GridSpec,
) -> np.ndarray:
    if not isinstance(trajectory, LocalTrajectory):
        raise TypeError("trajectory must be a LocalTrajectory")
    poses = trajectory.poses
    if not isinstance(poses, np.ndarray):
        raise TypeError("trajectory.poses must be an np.ndarray")
    if poses.dtype != np.float32:
        raise TypeError("trajectory.poses dtype must be float32")
    if poses.shape != (grid.future_steps, 3):
        raise ValueError(
            f"trajectory.poses shape must be ({grid.future_steps}, 3)"
        )
    if not np.isfinite(poses).all():
        raise ValueError("trajectory.poses must contain only finite values")
    if trajectory.metadata.get("pose_time_layout_version") != POSE_TIME_LAYOUT_VERSION:
        raise ValueError(
            "trajectory pose_time_layout_version must use schema3 future endpoints"
        )
    return poses


def _ordered_hidden_ids(
    hidden_object_ids: tuple[str, ...],
    *,
    world: OracleWorld,
) -> tuple[str, ...]:
    if not isinstance(hidden_object_ids, tuple):
        raise TypeError("hidden_object_ids must be an explicit tuple")
    if not all(isinstance(object_id, str) and object_id for object_id in hidden_object_ids):
        raise ValueError("hidden_object_ids must contain non-empty strings")
    if len(set(hidden_object_ids)) != len(hidden_object_ids):
        raise ValueError("hidden_object_ids must be unique")
    missing = set(hidden_object_ids) - set(world.dynamic_object_trajectories)
    if missing:
        raise ValueError(f"hidden_object_ids missing from OracleWorld: {sorted(missing)}")
    return tuple(sorted(hidden_object_ids))


def compute_hidden_risk_gt(
    trajectory: LocalTrajectory,
    world: OracleWorld,
    *,
    hidden_object_ids: tuple[str, ...],
    robot_footprint: Footprint,
    grid: GridSpec,
    future_dt_s: float,
    sigma_distance_m: float,
    sigma_time_s: float,
    near_miss_distance_m: float,
) -> RiskGroundTruth:
    """Compute risk using only caller-declared currently hidden dynamic objects.

    Future pose index ``k`` is the schema-v3 endpoint at
    ``tau = (k + 1) * future_dt_s``.  Objects present in ``world`` but absent
    from ``hidden_object_ids`` cannot affect any returned label.
    """

    if SCHEMA_VERSION != _SCHEMA3_VERSION:
        raise RuntimeError(f"risk GT requires schema {_SCHEMA3_VERSION}")
    validated_grid = _validate_grid(grid)
    robot_poses = _validate_trajectory(trajectory, grid=validated_grid)
    if not isinstance(world, OracleWorld):
        raise TypeError("world must be an OracleWorld")
    validate_oracle_world(world, validated_grid)
    if world.metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"world metadata schema_version must be {SCHEMA_VERSION}")
    if not isinstance(robot_footprint, (CircleFootprint, RectangleFootprint)):
        raise TypeError(
            "robot_footprint must be a CircleFootprint or RectangleFootprint"
        )

    dt_s = _positive_real(future_dt_s, name="future_dt_s")
    sigma_distance = _positive_real(
        sigma_distance_m, name="sigma_distance_m"
    )
    sigma_time = _positive_real(sigma_time_s, name="sigma_time_s")
    near_miss_distance = _nonnegative_real(
        near_miss_distance_m, name="near_miss_distance_m"
    )
    endpoint_times = (
        np.arange(1, validated_grid.future_steps + 1, dtype=np.float64) * dt_s
    )
    if not np.isfinite(endpoint_times).all():
        raise ValueError("schema3 future endpoint times must be finite")

    ordered_ids = _ordered_hidden_ids(hidden_object_ids, world=world)
    sentinel = resolve_no_object_clearance_sentinel(validated_grid)
    if not ordered_ids:
        return RiskGroundTruth(
            schema_version=SCHEMA_VERSION,
            pose_time_layout_version=POSE_TIME_LAYOUT_VERSION,
            collision_label=0,
            risk_severity=0.0,
            min_clearance=sentinel,
            near_miss=0,
            first_collision_time=None,
            time_to_min_clearance=None,
            critical_object_id=None,
            critical_object_type=None,
            has_hidden_target=False,
        )

    minimum_clearance = float("inf")
    minimum_time: float | None = None
    critical_object_id: str | None = None
    first_collision_index: int | None = None
    maximum_severity = 0.0

    for object_id in ordered_ids:
        object_footprint = footprint_from_spec(world.dynamic_object_specs[object_id])
        clearances = trajectory_signed_clearances(
            robot_footprint,
            robot_poses,
            object_footprint,
            world.dynamic_object_trajectories[object_id],
        )
        if clearances.shape != (validated_grid.future_steps,) or not np.isfinite(
            clearances
        ).all():
            raise ValueError("signed clearances must be finite at every future endpoint")

        local_minimum_index = int(np.argmin(clearances))
        local_minimum = float(clearances[local_minimum_index])
        if local_minimum < minimum_clearance:
            minimum_clearance = local_minimum
            minimum_time = float(endpoint_times[local_minimum_index])
            critical_object_id = object_id

        collision_indices = np.flatnonzero(clearances <= 0.0)
        if collision_indices.size:
            candidate = int(collision_indices[0])
            if first_collision_index is None or candidate < first_collision_index:
                first_collision_index = candidate

        severity = np.exp(-np.maximum(clearances, 0.0) / sigma_distance)
        severity *= np.exp(-endpoint_times / sigma_time)
        object_maximum = float(np.max(severity))
        if not np.isfinite(object_maximum):
            raise ValueError("risk severity must be finite")
        maximum_severity = max(maximum_severity, object_maximum)

    if critical_object_id is None or minimum_time is None:
        raise RuntimeError("hidden risk reduction produced no critical object")
    collision = first_collision_index is not None
    if collision:
        maximum_severity = 1.0
    first_collision_time = (
        None
        if first_collision_index is None
        else float(endpoint_times[first_collision_index])
    )
    near_miss = int(
        not collision and minimum_clearance < near_miss_distance
    )
    return RiskGroundTruth(
        schema_version=SCHEMA_VERSION,
        pose_time_layout_version=POSE_TIME_LAYOUT_VERSION,
        collision_label=int(collision),
        risk_severity=float(maximum_severity),
        min_clearance=float(minimum_clearance),
        near_miss=near_miss,
        first_collision_time=first_collision_time,
        time_to_min_clearance=minimum_time,
        critical_object_id=critical_object_id,
        critical_object_type=str(
            world.dynamic_object_specs[critical_object_id]["object_type"]
        ),
        has_hidden_target=True,
    )
