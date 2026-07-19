"""Canonical verification primitives and conservative motion feasibility."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from src.contracts import ACTION_VECTOR_DIM, ARRAY_DTYPE, GridSpec, SCHEMA_VERSION
from src.geometry import (
    Footprint,
    rasterize_footprint,
    signed_clearance,
    wrap_angle,
)


ACTION_LIBRARY_VERSION = "verification_actions_v1"
CANONICAL_ACTION_IDS = (
    "yaw_left_10",
    "yaw_right_10",
    "yaw_left_20",
    "yaw_right_20",
    "forward_peek",
    "stop_scan",
)
_EXPECTED_DELTAS = {
    "yaw_left_10": (0.0, 10.0),
    "yaw_right_10": (0.0, -10.0),
    "yaw_left_20": (0.0, 20.0),
    "yaw_right_20": (0.0, -20.0),
    "forward_peek": (0.30, 0.0),
    "stop_scan": (0.0, 0.0),
}
_TOP_LEVEL_KEYS = frozenset({"schema_version", "library_version", "actions"})
_ACTION_KEYS = frozenset(
    {"action_id", "duration_s", "delta_forward_m", "delta_yaw_deg"}
)


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _pose(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (3,) or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric array with shape (3,)")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class VerificationAction:
    """One robot-centric sensing primitive with a frozen three-value vector."""

    action_id: str
    duration_s: float
    delta_forward_m: float
    delta_yaw_rad: float

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id:
            raise ValueError("action_id must be a non-empty string")
        duration = _finite_real(self.duration_s, name="duration_s")
        forward = _finite_real(self.delta_forward_m, name="delta_forward_m")
        yaw = _finite_real(self.delta_yaw_rad, name="delta_yaw_rad")
        if duration <= 0.0:
            raise ValueError("duration_s must be positive")
        if forward < 0.0:
            raise ValueError("delta_forward_m must be non-negative")
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "delta_forward_m", forward)
        object.__setattr__(self, "delta_yaw_rad", yaw)

    @property
    def vector(self) -> np.ndarray:
        result = np.asarray(
            [self.duration_s, self.delta_forward_m, self.delta_yaw_rad],
            dtype=ARRAY_DTYPE,
        )
        if result.shape != (ACTION_VECTOR_DIM,):
            raise RuntimeError("verification action vector contract changed")
        return result


@dataclass(frozen=True)
class VerificationActionLibrary:
    schema_version: str
    library_version: str
    actions: tuple[VerificationAction, ...]

    @property
    def by_id(self) -> Mapping[str, VerificationAction]:
        return MappingProxyType({action.action_id: action for action in self.actions})


@dataclass(frozen=True)
class ActionTrace:
    """Continuous-motion approximation including both t=0 and the endpoint."""

    poses: np.ndarray
    times_s: np.ndarray


@dataclass(frozen=True)
class ActionFeasibility:
    feasible: bool
    reason: str | None
    critical_object_id: str | None
    minimum_dynamic_clearance_m: float


def load_verification_actions(path: str | Path) -> VerificationActionLibrary:
    """Load only the frozen six-action YAML layout; reject alternate libraries."""

    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid verification action config: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != _TOP_LEVEL_KEYS:
        raise ValueError("verification action config keys are invalid")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"verification action schema must be {SCHEMA_VERSION}")
    if raw["library_version"] != ACTION_LIBRARY_VERSION:
        raise ValueError("unsupported verification action library version")
    rows = raw["actions"]
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or set(row) != _ACTION_KEYS for row in rows
    ):
        raise ValueError("verification action rows have invalid keys")
    ids = tuple(row["action_id"] for row in rows)
    if ids != CANONICAL_ACTION_IDS:
        raise ValueError("actions must contain the canonical action IDs in order")

    actions: list[VerificationAction] = []
    for row in rows:
        action_id = row["action_id"]
        expected_forward, expected_yaw_deg = _EXPECTED_DELTAS[action_id]
        forward = _finite_real(
            row["delta_forward_m"], name=f"{action_id}.delta_forward_m"
        )
        yaw_deg = _finite_real(row["delta_yaw_deg"], name=f"{action_id}.delta_yaw_deg")
        if not np.isclose(forward, expected_forward, rtol=0.0, atol=1e-12) or not np.isclose(
            yaw_deg, expected_yaw_deg, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"{action_id} delta differs from the frozen action")
        actions.append(
            VerificationAction(
                action_id=action_id,
                duration_s=row["duration_s"],
                delta_forward_m=forward,
                delta_yaw_rad=np.deg2rad(yaw_deg),
            )
        )
    return VerificationActionLibrary(
        schema_version=SCHEMA_VERSION,
        library_version=ACTION_LIBRARY_VERSION,
        actions=tuple(actions),
    )


def _pose_at_fraction(
    start_pose: np.ndarray, action: VerificationAction, fraction: float
) -> np.ndarray:
    yaw_delta = action.delta_yaw_rad * fraction
    distance = action.delta_forward_m * fraction
    if abs(action.delta_yaw_rad) <= 1e-12:
        local_x = distance
        local_y = 0.0
    else:
        radius = action.delta_forward_m / action.delta_yaw_rad
        local_x = radius * np.sin(yaw_delta)
        local_y = radius * (1.0 - np.cos(yaw_delta))
    cosine = np.cos(start_pose[2])
    sine = np.sin(start_pose[2])
    result = np.asarray(
        [
            start_pose[0] + cosine * local_x - sine * local_y,
            start_pose[1] + sine * local_x + cosine * local_y,
            wrap_angle(start_pose[2] + yaw_delta),
        ],
        dtype=np.float64,
    )
    return result


def action_endpoint(start_pose: Any, action: VerificationAction) -> np.ndarray:
    if not isinstance(action, VerificationAction):
        raise TypeError("action must be a VerificationAction")
    return _pose_at_fraction(_pose(start_pose, name="start_pose"), action, 1.0).astype(
        ARRAY_DTYPE
    )


def sample_action_trace(
    start_pose: Any,
    action: VerificationAction,
    *,
    max_time_step_s: float = 0.05,
    max_translation_step_m: float = 0.025,
    max_yaw_step_rad: float = float(np.deg2rad(2.0)),
) -> ActionTrace:
    """Sample a bounded-error action trace, including the current pose."""

    if not isinstance(action, VerificationAction):
        raise TypeError("action must be a VerificationAction")
    start = _pose(start_pose, name="start_pose")
    time_step = _finite_real(max_time_step_s, name="max_time_step_s")
    translation_step = _finite_real(
        max_translation_step_m, name="max_translation_step_m"
    )
    yaw_step = _finite_real(max_yaw_step_rad, name="max_yaw_step_rad")
    if min(time_step, translation_step, yaw_step) <= 0.0:
        raise ValueError("action trace sampling steps must be positive")
    intervals = max(
        1,
        int(np.ceil(action.duration_s / time_step)),
        int(np.ceil(action.delta_forward_m / translation_step)),
        int(np.ceil(abs(action.delta_yaw_rad) / yaw_step)),
    )
    fractions = np.linspace(0.0, 1.0, intervals + 1, dtype=np.float64)
    poses = np.stack(
        [_pose_at_fraction(start, action, float(value)) for value in fractions],
        axis=0,
    ).astype(ARRAY_DTYPE)
    times = np.linspace(0.0, action.duration_s, intervals + 1, dtype=np.float64)
    return ActionTrace(poses=poses, times_s=times)


def action_cost(action: VerificationAction, cost_config: Mapping[str, Any]) -> float:
    """Compute the primitive cost exactly once from the frozen three terms."""

    if not isinstance(action, VerificationAction):
        raise TypeError("action must be a VerificationAction")
    if not isinstance(cost_config, Mapping) or set(cost_config) != {
        "lambda_time",
        "lambda_distance",
        "lambda_yaw_per_deg",
    }:
        raise ValueError("verification cost config keys are invalid")
    weights = {
        key: _finite_real(value, name=key) for key, value in cost_config.items()
    }
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("verification cost weights must be non-negative")
    result = (
        weights["lambda_time"] * action.duration_s
        + weights["lambda_distance"] * abs(action.delta_forward_m)
        + weights["lambda_yaw_per_deg"]
        * abs(float(np.rad2deg(action.delta_yaw_rad)))
    )
    if not np.isfinite(result):
        raise ValueError("verification action cost must be finite")
    return float(result)


def _interpolated_dynamic_pose(
    poses: np.ndarray, *, time_s: float, dt_s: float, object_id: str
) -> np.ndarray:
    if (
        not isinstance(poses, np.ndarray)
        or poses.ndim != 2
        or poses.shape[1] != 3
        or poses.shape[0] < 2
        or poses.dtype != ARRAY_DTYPE
        or not np.isfinite(poses).all()
    ):
        raise ValueError(
            f"dynamic_object_poses[{object_id!r}] must be finite float32 [N>=2,3]"
        )
    last_time = (poses.shape[0] - 1) * dt_s
    if time_s > last_time + 1e-10:
        raise ValueError(f"dynamic trajectory for {object_id!r} is too short")
    index = min(int(np.floor(time_s / dt_s)), poses.shape[0] - 2)
    start_time = index * dt_s
    fraction = min(1.0, max(0.0, (time_s - start_time) / dt_s))
    start = poses[index].astype(np.float64)
    end = poses[index + 1].astype(np.float64)
    yaw_pair = np.unwrap(np.asarray([start[2], end[2]], dtype=np.float64))
    result = (1.0 - fraction) * start + fraction * end
    result[2] = wrap_angle((1.0 - fraction) * yaw_pair[0] + fraction * yaw_pair[1])
    return result


def check_action_feasibility(
    start_pose: Any,
    action: VerificationAction,
    *,
    robot_footprint: Footprint,
    static_occupancy: np.ndarray,
    grid: GridSpec,
    dynamic_object_poses: Mapping[str, np.ndarray],
    dynamic_object_footprints: Mapping[str, Footprint],
    dynamic_dt_s: float,
) -> ActionFeasibility:
    """Check sampled robot motion against static and typed dynamic geometry.

    Dynamic arrays include the current pose at index 0 followed by future
    endpoints. Counterfactual label code is responsible for constructing that
    explicit seam; no oracle data is inferred here.
    """

    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    if (
        not isinstance(static_occupancy, np.ndarray)
        or static_occupancy.shape != (grid.height, grid.width)
        or static_occupancy.dtype != ARRAY_DTYPE
        or not np.isfinite(static_occupancy).all()
    ):
        raise ValueError("static_occupancy must be finite float32 with grid shape")
    if np.any((static_occupancy < 0.0) | (static_occupancy > 1.0)):
        raise ValueError("static_occupancy values must be in [0,1]")
    if not isinstance(dynamic_object_poses, Mapping) or not isinstance(
        dynamic_object_footprints, Mapping
    ):
        raise TypeError("dynamic objects must be mappings")
    if set(dynamic_object_poses) != set(dynamic_object_footprints):
        raise ValueError("dynamic pose and footprint IDs must align")
    dt_s = _finite_real(dynamic_dt_s, name="dynamic_dt_s")
    if dt_s <= 0.0:
        raise ValueError("dynamic_dt_s must be positive")

    trace = sample_action_trace(start_pose, action)
    finite_sentinel = float(
        np.hypot(grid.height * grid.resolution_m, grid.width * grid.resolution_m)
    )
    minimum = finite_sentinel
    critical: str | None = None
    for robot_pose, time_s in zip(trace.poses, trace.times_s, strict=True):
        robot_mask = rasterize_footprint(robot_footprint, robot_pose, grid)
        if np.any(static_occupancy[robot_mask] > 0.5):
            return ActionFeasibility(
                feasible=False,
                reason="static_collision",
                critical_object_id=None,
                minimum_dynamic_clearance_m=minimum,
            )
        for object_id in sorted(dynamic_object_poses):
            dynamic_pose = _interpolated_dynamic_pose(
                dynamic_object_poses[object_id],
                time_s=float(time_s),
                dt_s=dt_s,
                object_id=object_id,
            )
            clearance = signed_clearance(
                robot_footprint,
                robot_pose,
                dynamic_object_footprints[object_id],
                dynamic_pose,
            )
            if clearance < minimum:
                minimum = float(clearance)
                critical = object_id
            if clearance <= 0.0:
                return ActionFeasibility(
                    feasible=False,
                    reason="dynamic_collision",
                    critical_object_id=object_id,
                    minimum_dynamic_clearance_m=float(clearance),
                )
    return ActionFeasibility(
        feasible=True,
        reason=None,
        critical_object_id=critical,
        minimum_dynamic_clearance_m=float(minimum),
    )


__all__ = (
    "ACTION_LIBRARY_VERSION",
    "CANONICAL_ACTION_IDS",
    "ActionFeasibility",
    "ActionTrace",
    "VerificationAction",
    "VerificationActionLibrary",
    "action_cost",
    "action_endpoint",
    "check_action_feasibility",
    "load_verification_actions",
    "sample_action_trace",
)
