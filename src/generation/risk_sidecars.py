"""Oracle-only SOP-08 label sidecars for future occupancy supervision."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
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
    rasterize_footprint,
)

from .dynamic_object_transplant import footprint_from_spec


RISK_LABEL_SIDECAR_VERSION = "risk_label_sidecar_v1"


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite positive real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number")
    return result


def _owned_binary_mask(value: Any, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an np.ndarray")
    if value.dtype != np.uint8:
        raise TypeError(f"{name} dtype must be uint8")
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [T, H, W]")
    if not np.isin(value, (0, 1)).all():
        raise ValueError(f"{name} must be binary")
    result = np.array(value, dtype=np.uint8, order="C", copy=True)
    result.setflags(write=False)
    return result


def _owned_endpoint_times(value: Any, *, expected_steps: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("future_endpoint_times_s must be an np.ndarray")
    if value.dtype != np.float32:
        raise TypeError("future_endpoint_times_s dtype must be float32")
    if value.shape != (expected_steps,):
        raise ValueError(
            f"future_endpoint_times_s shape must be ({expected_steps},)"
        )
    if not np.isfinite(value).all():
        raise ValueError("future_endpoint_times_s must be finite")
    if value[0] <= 0.0 or np.any(np.diff(value.astype(np.float64)) <= 0.0):
        raise ValueError("future endpoint times must be strictly positive/increasing")
    result = np.array(value, dtype=np.float32, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RiskLabelSidecar:
    """Label-only future masks bound to one ``RiskSample.sample_id``.

    The uint8 masks remain outside every model-input contract.  Construction
    snapshots each array and makes it read-only so later mutation of an oracle
    world cannot alter a published label.
    """

    sample_id: str
    hidden_risk_occupancy: np.ndarray
    robot_future_footprints: np.ndarray
    future_endpoint_times_s: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_id",
            _nonempty_string(self.sample_id, name="sample_id"),
        )
        hidden = _owned_binary_mask(
            self.hidden_risk_occupancy, name="hidden_risk_occupancy"
        )
        robot = _owned_binary_mask(
            self.robot_future_footprints, name="robot_future_footprints"
        )
        if hidden.shape != robot.shape:
            raise ValueError("hidden and robot sidecar mask shapes must match")
        times = _owned_endpoint_times(
            self.future_endpoint_times_s, expected_steps=hidden.shape[0]
        )
        object.__setattr__(self, "hidden_risk_occupancy", hidden)
        object.__setattr__(self, "robot_future_footprints", robot)
        object.__setattr__(self, "future_endpoint_times_s", times)


def _validate_grid(grid: Any) -> GridSpec:
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    for name in ("height", "width", "future_steps"):
        value = getattr(grid, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ) or int(value) <= 0:
            raise ValueError(f"grid.{name} must be a positive integer")
    _positive_finite(grid.resolution_m, name="grid.resolution_m")
    return grid


def _validate_trajectory(
    trajectory: Any, *, grid: GridSpec
) -> LocalTrajectory:
    if not isinstance(trajectory, LocalTrajectory):
        raise TypeError("trajectory must be a LocalTrajectory")
    poses = trajectory.poses
    if not isinstance(poses, np.ndarray) or poses.dtype != np.float32:
        raise TypeError("trajectory.poses must be a float32 np.ndarray")
    if poses.shape != (grid.future_steps, 3):
        raise ValueError(
            f"trajectory.poses shape must be ({grid.future_steps}, 3)"
        )
    if not np.isfinite(poses).all():
        raise ValueError("trajectory.poses must be finite")
    if trajectory.metadata.get("pose_time_layout_version") != POSE_TIME_LAYOUT_VERSION:
        raise ValueError("trajectory must use schema3 future endpoint semantics")
    return trajectory


def _ordered_hidden_ids(
    hidden_object_ids: Any, *, world: OracleWorld
) -> tuple[str, ...]:
    if not isinstance(hidden_object_ids, tuple):
        raise TypeError("hidden_object_ids must be an explicit tuple")
    if not all(isinstance(object_id, str) and object_id for object_id in hidden_object_ids):
        raise ValueError("hidden_object_ids must contain non-empty strings")
    if len(hidden_object_ids) != len(set(hidden_object_ids)):
        raise ValueError("hidden_object_ids must be unique")
    missing = set(hidden_object_ids) - set(world.dynamic_object_trajectories)
    if missing:
        raise ValueError(
            f"hidden_object_ids missing from OracleWorld: {sorted(missing)}"
        )
    return tuple(sorted(hidden_object_ids))


def build_risk_label_sidecar(
    *,
    sample_id: str,
    trajectory: LocalTrajectory,
    world: OracleWorld,
    hidden_object_ids: tuple[str, ...],
    robot_footprint: Footprint,
    grid: GridSpec,
    future_dt_s: float,
) -> RiskLabelSidecar:
    """Rasterize oracle labels without exposing them to a ``RiskSample``.

    Only caller-declared hidden object IDs contribute to hidden occupancy;
    other actors in the oracle world are intentionally ignored.  Pose index
    ``k`` is the strict future endpoint ``(k + 1) * future_dt_s``.
    """

    _nonempty_string(sample_id, name="sample_id")
    validated_grid = _validate_grid(grid)
    validated_trajectory = _validate_trajectory(
        trajectory, grid=validated_grid
    )
    if not isinstance(world, OracleWorld):
        raise TypeError("world must be an OracleWorld")
    validate_oracle_world(world, validated_grid)
    if world.metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"world metadata schema_version must be {SCHEMA_VERSION}")
    if not isinstance(robot_footprint, (CircleFootprint, RectangleFootprint)):
        raise TypeError(
            "robot_footprint must be a CircleFootprint or RectangleFootprint"
        )
    dt_s = _positive_finite(future_dt_s, name="future_dt_s")
    ordered_ids = _ordered_hidden_ids(hidden_object_ids, world=world)

    shape = (
        validated_grid.future_steps,
        validated_grid.height,
        validated_grid.width,
    )
    hidden_occupancy = np.zeros(shape, dtype=np.uint8)
    for object_id in ordered_ids:
        footprint = footprint_from_spec(world.dynamic_object_specs[object_id])
        object_poses = world.dynamic_object_trajectories[object_id]
        for index, pose in enumerate(object_poses):
            hidden_occupancy[index] |= rasterize_footprint(
                footprint, pose, validated_grid
            ).astype(np.uint8, copy=False)

    robot_masks = np.zeros(shape, dtype=np.uint8)
    for index, pose in enumerate(validated_trajectory.poses):
        robot_masks[index] = rasterize_footprint(
            robot_footprint, pose, validated_grid
        ).astype(np.uint8, copy=False)

    endpoint_times = (
        np.arange(1, validated_grid.future_steps + 1, dtype=np.float32)
        * np.float32(dt_s)
    )
    if not np.isfinite(endpoint_times).all():
        raise ValueError("future endpoint times must be finite")
    return RiskLabelSidecar(
        sample_id=sample_id,
        hidden_risk_occupancy=hidden_occupancy,
        robot_future_footprints=robot_masks,
        future_endpoint_times_s=endpoint_times,
    )


__all__ = (
    "RISK_LABEL_SIDECAR_VERSION",
    "RiskLabelSidecar",
    "build_risk_label_sidecar",
)
