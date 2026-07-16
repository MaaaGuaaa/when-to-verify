"""Deterministic filters for candidate differential-drive trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.contracts import ARRAY_DTYPE, build_grid_spec
from src.geometry import (
    Footprint,
    RectangleFootprint,
    footprint_aabb,
    grid_bounds,
    inflate_footprint,
    rasterize_footprint,
)

from .trajectory_sampler import CandidateRollout


def _inflated_robot_footprint(config: dict) -> Footprint:
    """Build the configured rectangular robot footprint with safety inflation."""
    robot = config["robot"]
    return inflate_footprint(
        RectangleFootprint(robot["length_m"], robot["width_m"]),
        robot["inflation_m"],
    )


def _optional_acceleration_limit(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} acceleration limit must be finite and non-negative")
    return value


@dataclass(frozen=True)
class TrajectoryFilterDecision:
    """Decision and stable rejection reasons for one candidate."""

    trajectory_id: str
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TrajectoryFilterReport:
    """Accepted/rejected candidates and auditable reason statistics."""

    accepted: tuple[CandidateRollout, ...]
    rejected: tuple[CandidateRollout, ...]
    decisions: tuple[TrajectoryFilterDecision, ...]
    rejection_counts: dict[str, int]
    acceptance_rate: float


def trajectory_rejection_reasons(
    candidate: CandidateRollout,
    config: dict,
    *,
    initial_control: np.ndarray | None = None,
    max_linear_acceleration_mps2: float | None = None,
    max_angular_acceleration_radps2: float | None = None,
    static_occupancy: np.ndarray | None = None,
) -> tuple[str, ...]:
    """Return stable reason codes for trajectory-contract and dynamics failures."""
    max_linear_acceleration_mps2 = _optional_acceleration_limit(
        max_linear_acceleration_mps2, "linear"
    )
    max_angular_acceleration_radps2 = _optional_acceleration_limit(
        max_angular_acceleration_radps2, "angular"
    )
    reasons = []
    expected_steps = int(config["bev"]["future_steps"])
    if (
        candidate.poses.shape != (expected_steps, 3)
        or candidate.controls.shape != (expected_steps, 2)
    ):
        return ("shape_mismatch",)
    if (
        candidate.poses.dtype != ARRAY_DTYPE
        or candidate.controls.dtype != ARRAY_DTYPE
    ):
        return ("dtype_mismatch",)
    robot = config["robot"]
    if not (
        np.isfinite(candidate.poses).all()
        and np.isfinite(candidate.controls).all()
    ):
        reasons.append("nonfinite")
    if np.any(
        np.abs(candidate.controls[:, 0]) > float(robot["max_linear_speed_mps"])
    ):
        reasons.append("linear_speed_limit")
    if np.any(
        np.abs(candidate.controls[:, 1]) > float(robot["max_angular_speed_radps"])
    ):
        reasons.append("angular_speed_limit")
    if candidate.is_stop and np.any(np.abs(candidate.controls) > 1e-6):
        reasons.append("invalid_stop_marker")
    if not candidate.is_stop and np.all(np.abs(candidate.controls) <= 1e-6):
        reasons.append("invalid_stagnation")
    has_reverse_control = bool(np.any(candidate.controls[:, 0] < -1e-6))
    has_forward_control = bool(np.any(candidate.controls[:, 0] > 1e-6))
    if candidate.is_reverse != has_reverse_control or (
        candidate.is_reverse and has_forward_control
    ):
        reasons.append("invalid_reverse_marker")
    controls_for_acceleration = candidate.controls
    if initial_control is not None:
        initial_control = np.asarray(initial_control)
        if initial_control.shape != (2,) or not np.isfinite(initial_control).all():
            raise ValueError("initial_control must be a finite [2] array")
        controls_for_acceleration = np.vstack((initial_control, candidate.controls))
    if max_linear_acceleration_mps2 is not None:
        dt_s = float(config["trajectories"]["dt_s"])
        linear_acceleration = np.abs(np.diff(controls_for_acceleration[:, 0])) / dt_s
        if np.any(linear_acceleration > max_linear_acceleration_mps2):
            reasons.append("linear_acceleration_limit")
    if max_angular_acceleration_radps2 is not None:
        dt_s = float(config["trajectories"]["dt_s"])
        angular_acceleration = (
            np.abs(np.diff(controls_for_acceleration[:, 1])) / dt_s
        )
        if np.any(angular_acceleration > max_angular_acceleration_radps2):
            reasons.append("angular_acceleration_limit")
    if "nonfinite" not in reasons:
        grid = build_grid_spec(config)
        x_min, x_max, y_min, y_max = grid_bounds(grid)
        footprint = _inflated_robot_footprint(config)
        for pose in candidate.poses:
            pose_x_min, pose_x_max, pose_y_min, pose_y_max = footprint_aabb(
                footprint, pose
            )
            if (
                pose_x_min < x_min
                or pose_x_max > x_max
                or pose_y_min < y_min
                or pose_y_max > y_max
            ):
                reasons.append("bev_out_of_bounds")
                break
        if static_occupancy is not None:
            occupancy = np.asarray(static_occupancy)
            if occupancy.shape != (grid.height, grid.width):
                raise ValueError("static_occupancy shape must match the BEV grid")
            if occupancy.dtype.kind not in "buif":
                raise TypeError("static_occupancy must be a numeric or boolean array")
            if not np.isfinite(occupancy).all():
                raise ValueError("static_occupancy must contain only finite values")
            occupied = occupancy != 0
            for pose in candidate.poses:
                footprint_mask = rasterize_footprint(footprint, pose, grid)
                if np.any(footprint_mask & occupied):
                    reasons.append("static_collision")
                    break
    return tuple(reasons)


def filter_trajectory_candidates(
    candidates: list[CandidateRollout] | tuple[CandidateRollout, ...],
    config: dict,
    *,
    initial_control: np.ndarray | None = None,
    max_linear_acceleration_mps2: float | None = None,
    max_angular_acceleration_radps2: float | None = None,
    static_occupancy: np.ndarray | None = None,
) -> TrajectoryFilterReport:
    """Filter candidates and retain an auditable reason distribution."""
    accepted = []
    rejected = []
    decisions = []
    rejection_counts: dict[str, int] = {}
    for candidate in candidates:
        reasons = trajectory_rejection_reasons(
            candidate,
            config,
            initial_control=initial_control,
            max_linear_acceleration_mps2=max_linear_acceleration_mps2,
            max_angular_acceleration_radps2=max_angular_acceleration_radps2,
            static_occupancy=static_occupancy,
        )
        is_accepted = not reasons
        decisions.append(
            TrajectoryFilterDecision(
                trajectory_id=candidate.trajectory_id,
                accepted=is_accepted,
                reasons=reasons,
            )
        )
        if is_accepted:
            accepted.append(candidate)
        else:
            rejected.append(candidate)
            for reason in reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    total = len(candidates)
    return TrajectoryFilterReport(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        decisions=tuple(decisions),
        rejection_counts=rejection_counts,
        acceptance_rate=len(accepted) / total if total else 0.0,
    )
