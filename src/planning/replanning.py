"""Verification-aware replanning from the post-action robot frame."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from src.contracts import ARRAY_DTYPE, LocalTrajectory, build_grid_spec
from src.geometry import (
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint_sweep,
    transform_poses_local_to_global,
    wrap_angle,
)

from .query_maps import build_local_trajectory
from .trajectory_filters import filter_trajectory_candidates
from .trajectory_sampler import CandidateRollout, sample_candidate_rollouts


REPLANNING_VERSION = "post_action_anchored_sampler_v1"


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _float32_pose(value: Any, *, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (3,)
        or value.dtype != ARRAY_DTYPE
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be finite float32 with shape (3,)")
    return np.array(value, dtype=ARRAY_DTYPE, order="C", copy=True)


def _owned_array(
    value: np.ndarray, *, name: str, ndim: int, final_dim: int | None = None
) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != ndim
        or (final_dim is not None and value.shape[-1] != final_dim)
        or value.dtype != ARRAY_DTYPE
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} violates the finite float32 array contract")
    result = np.array(value, dtype=ARRAY_DTYPE, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ReplannedCandidate:
    """A post-action-local trajectory plus its explicit parent-frame geometry."""

    trajectory: LocalTrajectory
    implicit_start_pose: np.ndarray
    poses_in_parent_frame: np.ndarray
    swept_mask_in_parent_frame: np.ndarray
    intent_error: float

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, LocalTrajectory):
            raise TypeError("trajectory must be a LocalTrajectory")
        start = _float32_pose(self.implicit_start_pose, name="implicit_start_pose")
        start.setflags(write=False)
        parent = _owned_array(
            self.poses_in_parent_frame,
            name="poses_in_parent_frame",
            ndim=2,
            final_dim=3,
        )
        mask = _owned_array(
            self.swept_mask_in_parent_frame,
            name="swept_mask_in_parent_frame",
            ndim=2,
        )
        if parent.shape != self.trajectory.poses.shape:
            raise ValueError("local and parent pose arrays must align")
        error = _finite_real(self.intent_error, name="intent_error")
        if error < 0.0:
            raise ValueError("intent_error must be non-negative")
        object.__setattr__(self, "implicit_start_pose", start)
        object.__setattr__(self, "poses_in_parent_frame", parent)
        object.__setattr__(self, "swept_mask_in_parent_frame", mask)
        object.__setattr__(self, "intent_error", error)

    @property
    def pose_sequence_in_parent_frame(self) -> np.ndarray:
        """Return `[q0,q1,...,qT]` with q0 equal to the post-action pose."""

        return np.vstack((self.implicit_start_pose, self.poses_in_parent_frame)).astype(
            ARRAY_DTYPE, copy=False
        )


@dataclass(frozen=True)
class ReplanningResult:
    version: str
    post_action_pose: np.ndarray
    task_anchor_pose: np.ndarray
    candidates: tuple[ReplannedCandidate, ...]
    reject_available: bool
    rejection_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.version != REPLANNING_VERSION:
            raise ValueError("unsupported replanning version")
        post = _float32_pose(self.post_action_pose, name="post_action_pose")
        anchor = _float32_pose(self.task_anchor_pose, name="task_anchor_pose")
        post.setflags(write=False)
        anchor.setflags(write=False)
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(item, ReplannedCandidate) for item in self.candidates
        ):
            raise TypeError("candidates must be a tuple of ReplannedCandidate")
        if self.reject_available is not True:
            raise ValueError("reject must remain available after verification")
        counts = dict(self.rejection_counts)
        if any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in counts.items()
        ):
            raise ValueError("rejection_counts must be non-negative integer counts")
        object.__setattr__(self, "post_action_pose", post)
        object.__setattr__(self, "task_anchor_pose", anchor)
        object.__setattr__(self, "rejection_counts", MappingProxyType(counts))


def _validate_nominal(trajectory: LocalTrajectory, *, steps: int) -> None:
    if not isinstance(trajectory, LocalTrajectory):
        raise TypeError("nominal_trajectory must be a LocalTrajectory")
    arrays = (
        (trajectory.poses, (steps, 3), "nominal poses"),
        (trajectory.controls, (steps, 2), "nominal controls"),
    )
    for array, shape, name in arrays:
        if (
            not isinstance(array, np.ndarray)
            or array.shape != shape
            or array.dtype != ARRAY_DTYPE
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"{name} violate the trajectory contract")


def _inflated_robot(config: Mapping[str, Any]):
    robot = config["robot"]
    return inflate_footprint(
        RectangleFootprint(float(robot["length_m"]), float(robot["width_m"])),
        float(robot["inflation_m"]),
    )


def _dense_local_poses(
    candidate: CandidateRollout,
    *,
    dt_s: float,
    resolution_m: float,
    sweep_radius_m: float,
) -> np.ndarray:
    """Densify the same constant-control intervals used by SOP04."""

    anchors = np.vstack(
        (np.zeros((1, 3), dtype=np.float64), candidate.poses.astype(np.float64))
    )
    dense = [anchors[0]]
    maximum_step = 0.5 * resolution_m
    for interval, (start, endpoint) in enumerate(zip(anchors[:-1], anchors[1:])):
        v = float(candidate.controls[interval, 0])
        omega = float(candidate.controls[interval, 1])
        bound = abs(v) * dt_s + abs(omega) * dt_s * sweep_radius_m
        subdivisions = max(1, int(np.ceil(bound / maximum_step)))
        for subdivision in range(1, subdivisions + 1):
            elapsed = subdivision / subdivisions * dt_s
            yaw = start[2] + omega * elapsed
            if omega == 0.0:
                x = start[0] + v * elapsed * np.cos(start[2])
                y = start[1] + v * elapsed * np.sin(start[2])
            else:
                radius = v / omega
                x = start[0] + radius * (np.sin(yaw) - np.sin(start[2]))
                y = start[1] - radius * (np.cos(yaw) - np.cos(start[2]))
            dense.append(np.asarray([x, y, yaw], dtype=np.float64))
        if not np.allclose(dense[-1], endpoint, rtol=0.0, atol=1e-5):
            raise ValueError("sampled replan violates differential-drive dynamics")
    return np.asarray(dense, dtype=np.float64)


def _intent_error(endpoint: np.ndarray, anchor: np.ndarray) -> float:
    position_error = float(np.linalg.norm(endpoint[:2] - anchor[:2]))
    heading_error = abs(float(wrap_angle(endpoint[2] - anchor[2])))
    return position_error + 0.25 * heading_error


def generate_replanned_candidates(
    *,
    post_action_pose: np.ndarray,
    nominal_trajectory: LocalTrajectory,
    action_id: str,
    config: Mapping[str, Any],
    static_occupancy: np.ndarray,
    braking_deceleration_mps2: float,
    max_candidates: int | None = None,
) -> ReplanningResult:
    """Freshly sample in the post-action frame and filter in the parent frame."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    config_dict = dict(config)
    grid = build_grid_spec(config_dict)
    _validate_nominal(nominal_trajectory, steps=grid.future_steps)
    post = _float32_pose(post_action_pose, name="post_action_pose")
    if not isinstance(action_id, str) or not action_id:
        raise ValueError("action_id must be a non-empty string")
    deceleration = _finite_real(
        braking_deceleration_mps2, name="braking_deceleration_mps2"
    )
    if deceleration <= 0.0:
        raise ValueError("braking_deceleration_mps2 must be positive")
    if max_candidates is not None and (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or max_candidates <= 0
    ):
        raise ValueError("max_candidates must be a positive integer or None")
    if (
        not isinstance(static_occupancy, np.ndarray)
        or static_occupancy.shape != (grid.height, grid.width)
        or static_occupancy.dtype != ARRAY_DTYPE
        or not np.isfinite(static_occupancy).all()
        or not np.isin(static_occupancy, (0.0, 1.0)).all()
    ):
        raise ValueError("static_occupancy must be binary finite float32 grid data")

    sampled = sample_candidate_rollouts(config_dict, reverse_stress=False)
    local_by_id: dict[str, tuple[CandidateRollout, str]] = {}
    parent_candidates: list[CandidateRollout] = []
    for primitive in sampled:
        trajectory_id = f"replan::{action_id}::{primitive.trajectory_id}"
        local = CandidateRollout(
            trajectory_id=trajectory_id,
            poses=np.array(primitive.poses, dtype=ARRAY_DTYPE, order="C", copy=True),
            controls=np.array(
                primitive.controls, dtype=ARRAY_DTYPE, order="C", copy=True
            ),
            is_stop=primitive.is_stop,
            is_reverse=primitive.is_reverse,
        )
        parent_poses = transform_poses_local_to_global(local.poses, post).astype(
            ARRAY_DTYPE
        )
        parent_candidates.append(
            CandidateRollout(
                trajectory_id=trajectory_id,
                poses=parent_poses,
                controls=local.controls,
                is_stop=local.is_stop,
                is_reverse=local.is_reverse,
            )
        )
        local_by_id[trajectory_id] = (local, primitive.trajectory_id)

    report = filter_trajectory_candidates(
        tuple(parent_candidates),
        config_dict,
        static_occupancy=static_occupancy,
    )
    accepted_parent = {item.trajectory_id: item for item in report.accepted}
    footprint = _inflated_robot(config_dict)
    sweep_radius = 0.5 * float(
        np.hypot(footprint.length_m, footprint.width_m)
    )
    dt_s = float(config_dict["trajectories"]["dt_s"])
    anchor = np.array(nominal_trajectory.poses[-1], dtype=ARRAY_DTYPE, copy=True)
    candidates: list[ReplannedCandidate] = []
    for trajectory_id in sorted(accepted_parent):
        local, primitive_id = local_by_id[trajectory_id]
        parent = accepted_parent[trajectory_id]
        error = _intent_error(parent.poses[-1], anchor)
        trajectory = build_local_trajectory(
            local,
            config_dict,
            braking_deceleration_mps2=deceleration,
            task_cost=error,
        )
        metadata = {
            **trajectory.metadata,
            "replanning_version": REPLANNING_VERSION,
            "sampling_origin": "post_action_pose",
            "parent_frame": "pre_verification_robot",
            "source_primitive_id": primitive_id,
            "action_id": action_id,
            "implicit_start_pose": [float(value) for value in post],
            "task_anchor_pose": [float(value) for value in anchor],
            "nominal_trajectory_id": nominal_trajectory.trajectory_id,
            "nominal_suffix_used": False,
        }
        trajectory = replace(trajectory, metadata=metadata)
        dense_local = _dense_local_poses(
            local,
            dt_s=dt_s,
            resolution_m=float(grid.resolution_m),
            sweep_radius_m=sweep_radius,
        )
        dense_parent = transform_poses_local_to_global(dense_local, post)
        parent_sweep = rasterize_footprint_sweep(
            footprint, dense_parent, grid
        ).astype(ARRAY_DTYPE)
        candidates.append(
            ReplannedCandidate(
                trajectory=trajectory,
                implicit_start_pose=post,
                poses_in_parent_frame=parent.poses,
                swept_mask_in_parent_frame=parent_sweep,
                intent_error=error,
            )
        )

    candidates.sort(key=lambda item: (item.intent_error, item.trajectory.trajectory_id))
    stop_candidates = [item for item in candidates if item.trajectory.metadata["is_stop"]]
    if len(stop_candidates) != 1:
        raise RuntimeError("post-action replanning must retain exactly one stop")
    stop = stop_candidates[0]
    non_stop = [item for item in candidates if item is not stop]
    if max_candidates is None:
        selected = tuple(non_stop + [stop])
    elif max_candidates == 1:
        selected = (stop,)
    else:
        selected = tuple(non_stop[: max_candidates - 1] + [stop])
    return ReplanningResult(
        version=REPLANNING_VERSION,
        post_action_pose=post,
        task_anchor_pose=anchor,
        candidates=selected,
        reject_available=True,
        rejection_counts=report.rejection_counts,
    )


__all__ = (
    "REPLANNING_VERSION",
    "ReplannedCandidate",
    "ReplanningResult",
    "generate_replanned_candidates",
)
