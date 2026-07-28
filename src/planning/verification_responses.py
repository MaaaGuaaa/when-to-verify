"""Deterministic label-side response rollouts for verification policies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real
from typing import Any

import numpy as np

from src.contracts import ARRAY_DTYPE, LocalTrajectory, POSE_TIME_LAYOUT_VERSION
from src.geometry import wrap_angle
from src.planning.differential_drive import (
    DEFAULT_ANGULAR_DECELERATION_RADPS2,
    integrate_twist,
)
from src.planning.verification_actions import ActionTrace


VERIFICATION_RESPONSE_VERSION = "verification_responses_v2"


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
        raise ValueError(f"{name} must be a numeric pose with shape (3,)")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _approach_zero(value: float, amount: float) -> float:
    if value > 0.0:
        return max(0.0, value - amount)
    if value < 0.0:
        return min(0.0, value + amount)
    return 0.0


@dataclass(frozen=True)
class VerificationPolicyBranch:
    """The motion actually executed before post-verification replanning."""

    branch_kind: str
    executed_trace: ActionTrace
    trigger_time_s: float | None
    planned_action_end_time_s: float

    def __post_init__(self) -> None:
        if self.branch_kind not in {
            "complete",
            "observe_and_replan",
            "emergency_brake",
        }:
            raise ValueError("unsupported verification policy branch kind")
        if not isinstance(self.executed_trace, ActionTrace):
            raise TypeError("executed_trace must be an ActionTrace")
        planned_end = _finite_real(
            self.planned_action_end_time_s,
            name="planned_action_end_time_s",
        )
        if planned_end <= 0.0:
            raise ValueError("planned action end time must be positive")
        object.__setattr__(self, "planned_action_end_time_s", planned_end)
        trigger = self.trigger_time_s
        if self.branch_kind == "complete":
            if trigger is not None:
                raise ValueError("complete branch must not record a trigger time")
            if not np.isclose(
                self.executed_trace.times_s[-1],
                planned_end,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("complete branch must execute the planned action")
        else:
            trigger = _finite_real(trigger, name="trigger_time_s")
            if (
                trigger < 0.0
                or trigger > planned_end + 1e-12
                or trigger > self.executed_trace.times_s[-1] + 1e-12
            ):
                raise ValueError("trigger time must lie within the executed trace")
            object.__setattr__(self, "trigger_time_s", trigger)
            if self.branch_kind == "observe_and_replan" and not np.isclose(
                self.executed_trace.times_s[-1],
                trigger,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "observe-and-replan branch must end at the trigger time"
                )
        if self.branch_kind != "observe_and_replan":
            if (
                self.executed_trace.linear_velocities_mps[-1] != 0.0
                or self.executed_trace.angular_velocities_radps[-1] != 0.0
            ):
                raise ValueError(
                    "complete and emergency-brake branches must end at rest"
                )

    @property
    def end_time_s(self) -> float:
        return float(self.executed_trace.times_s[-1])

    @property
    def end_pose(self) -> np.ndarray:
        return np.array(
            self.executed_trace.poses[-1],
            dtype=ARRAY_DTYPE,
            order="C",
            copy=True,
        )

    @property
    def end_control(self) -> np.ndarray:
        return np.asarray(
            [
                self.executed_trace.linear_velocities_mps[-1],
                self.executed_trace.angular_velocities_radps[-1],
            ],
            dtype=ARRAY_DTYPE,
        )


def build_completed_action_branch(
    action_trace: ActionTrace,
) -> VerificationPolicyBranch:
    if not isinstance(action_trace, ActionTrace):
        raise TypeError("action_trace must be an ActionTrace")
    return VerificationPolicyBranch(
        branch_kind="complete",
        executed_trace=action_trace,
        trigger_time_s=None,
        planned_action_end_time_s=float(action_trace.times_s[-1]),
    )


def _trace_state_at_time(
    action_trace: ActionTrace, time_s: float
) -> tuple[np.ndarray, float, float, int, bool]:
    time_value = _finite_real(time_s, name="response_time_s")
    if time_value < 0.0 or time_value > action_trace.times_s[-1] + 1e-12:
        raise ValueError("response_time_s must lie within the action trace")
    close = np.flatnonzero(
        np.isclose(action_trace.times_s, time_value, rtol=0.0, atol=1e-12)
    )
    if close.size:
        index = int(close[0])
        return (
            action_trace.poses[index].astype(np.float64),
            float(action_trace.linear_velocities_mps[index]),
            float(action_trace.angular_velocities_radps[index]),
            index,
            True,
        )
    upper = int(np.searchsorted(action_trace.times_s, time_value, side="right"))
    lower = upper - 1
    start_time = float(action_trace.times_s[lower])
    end_time = float(action_trace.times_s[upper])
    fraction = (time_value - start_time) / (end_time - start_time)
    start_pose = action_trace.poses[lower].astype(np.float64)
    end_pose = action_trace.poses[upper].astype(np.float64)
    yaw_pair = np.unwrap(np.asarray([start_pose[2], end_pose[2]], dtype=np.float64))
    pose = (1.0 - fraction) * start_pose + fraction * end_pose
    pose[2] = wrap_angle((1.0 - fraction) * yaw_pair[0] + fraction * yaw_pair[1])
    linear = (1.0 - fraction) * action_trace.linear_velocities_mps[lower]
    linear += fraction * action_trace.linear_velocities_mps[upper]
    angular = (1.0 - fraction) * action_trace.angular_velocities_radps[lower]
    angular += fraction * action_trace.angular_velocities_radps[upper]
    return pose, float(linear), float(angular), upper, False


def _trace_prefix_at_time(
    action_trace: ActionTrace,
    time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pose, linear, angular, seam_index, exact = _trace_state_at_time(
        action_trace, time_s
    )
    if exact:
        prefix_end = seam_index + 1
        return (
            action_trace.poses[:prefix_end].copy(),
            action_trace.times_s[:prefix_end].copy(),
            action_trace.linear_velocities_mps[:prefix_end].copy(),
            action_trace.angular_velocities_radps[:prefix_end].copy(),
        )
    return (
        np.concatenate(
            (action_trace.poses[:seam_index], pose[None, :].astype(ARRAY_DTYPE)),
            axis=0,
        ),
        np.concatenate(
            (
                action_trace.times_s[:seam_index],
                np.asarray([time_s], dtype=np.float64),
            )
        ),
        np.concatenate(
            (
                action_trace.linear_velocities_mps[:seam_index],
                np.asarray([linear], dtype=np.float64),
            )
        ),
        np.concatenate(
            (
                action_trace.angular_velocities_radps[:seam_index],
                np.asarray([angular], dtype=np.float64),
            )
        ),
    )


def build_observe_and_replan_branch(
    *,
    action_trace: ActionTrace,
    response_time_s: float,
) -> VerificationPolicyBranch:
    """Interrupt at a new relevant observation without forcing a stop."""

    if not isinstance(action_trace, ActionTrace):
        raise TypeError("action_trace must be an ActionTrace")
    trigger = _finite_real(response_time_s, name="response_time_s")
    if trigger <= 0.0:
        raise ValueError("observe-and-replan response time must be positive")
    prefix_poses, prefix_times, prefix_linear, prefix_angular = (
        _trace_prefix_at_time(action_trace, trigger)
    )
    executed = ActionTrace(
        poses=np.asarray(prefix_poses, dtype=ARRAY_DTYPE),
        times_s=np.asarray(prefix_times, dtype=np.float64),
        linear_velocities_mps=np.asarray(prefix_linear, dtype=np.float64),
        angular_velocities_radps=np.asarray(prefix_angular, dtype=np.float64),
    )
    return VerificationPolicyBranch(
        branch_kind="observe_and_replan",
        executed_trace=executed,
        trigger_time_s=trigger,
        planned_action_end_time_s=float(action_trace.times_s[-1]),
    )


def build_reactive_braking_branch(
    *,
    action_trace: ActionTrace,
    response_time_s: float,
    braking_deceleration_mps2: float,
    future_horizon_s: float,
    angular_deceleration_radps2: float = DEFAULT_ANGULAR_DECELERATION_RADPS2,
    max_time_step_s: float = 0.05,
) -> VerificationPolicyBranch:
    """Follow the real action trace until detection, then brake from that state."""

    if not isinstance(action_trace, ActionTrace):
        raise TypeError("action_trace must be an ActionTrace")
    linear_deceleration = _finite_real(
        braking_deceleration_mps2, name="braking_deceleration_mps2"
    )
    angular_deceleration = _finite_real(
        angular_deceleration_radps2,
        name="angular_deceleration_radps2",
    )
    horizon = _finite_real(future_horizon_s, name="future_horizon_s")
    time_step = _finite_real(max_time_step_s, name="max_time_step_s")
    if min(linear_deceleration, angular_deceleration, horizon, time_step) <= 0.0:
        raise ValueError("horizon, decelerations, and sampling step must be positive")
    trigger = _finite_real(response_time_s, name="response_time_s")
    pose, linear, angular, _, _ = _trace_state_at_time(
        action_trace, trigger
    )
    prefix_poses, prefix_times, prefix_linear, prefix_angular = (
        _trace_prefix_at_time(action_trace, trigger)
    )

    linear_stop_s = abs(linear) / linear_deceleration
    angular_stop_s = abs(angular) / angular_deceleration
    braking_duration_s = max(linear_stop_s, angular_stop_s)
    end_time_s = trigger + braking_duration_s
    if end_time_s > horizon + 1e-10:
        raise ValueError("reactive braking exceeds the future horizon")
    if braking_duration_s == 0.0:
        if prefix_times.size < 2:
            epsilon = min(time_step, horizon) * 1e-9
            prefix_poses = np.concatenate(
                (prefix_poses, prefix_poses[-1:]), axis=0
            )
            prefix_times = np.concatenate(
                (prefix_times, np.asarray([trigger + epsilon], dtype=np.float64))
            )
            prefix_linear = np.concatenate(
                (prefix_linear, np.asarray([0.0], dtype=np.float64))
            )
            prefix_angular = np.concatenate(
                (prefix_angular, np.asarray([0.0], dtype=np.float64))
            )
        prefix_linear[-1] = 0.0
        prefix_angular[-1] = 0.0
        executed = ActionTrace(
            poses=np.asarray(prefix_poses, dtype=ARRAY_DTYPE),
            times_s=np.asarray(prefix_times, dtype=np.float64),
            linear_velocities_mps=np.asarray(prefix_linear, dtype=np.float64),
            angular_velocities_radps=np.asarray(prefix_angular, dtype=np.float64),
        )
        return VerificationPolicyBranch(
            branch_kind="emergency_brake",
            executed_trace=executed,
            trigger_time_s=trigger,
            planned_action_end_time_s=float(action_trace.times_s[-1]),
        )

    intervals = max(1, int(np.ceil(braking_duration_s / time_step)))
    relative_times = np.unique(
        np.concatenate(
            (
                np.linspace(
                    0.0, braking_duration_s, intervals + 1, dtype=np.float64
                ),
                np.asarray([linear_stop_s, angular_stop_s], dtype=np.float64),
            )
        )
    )
    braking_linear = np.sign(linear) * np.maximum(
        abs(linear) - linear_deceleration * relative_times, 0.0
    )
    braking_angular = np.sign(angular) * np.maximum(
        abs(angular) - angular_deceleration * relative_times, 0.0
    )
    braking_poses = np.empty((relative_times.size, 3), dtype=ARRAY_DTYPE)
    braking_poses[0] = pose.astype(ARRAY_DTYPE)
    current_pose = pose
    for index in range(1, relative_times.size):
        dt_s = float(relative_times[index] - relative_times[index - 1])
        current_pose = integrate_twist(
            current_pose,
            v=0.5 * float(braking_linear[index - 1] + braking_linear[index]),
            omega=0.5
            * float(braking_angular[index - 1] + braking_angular[index]),
            dt_s=dt_s,
        )
        braking_poses[index] = current_pose.astype(ARRAY_DTYPE)
    executed = ActionTrace(
        poses=np.concatenate((prefix_poses, braking_poses[1:]), axis=0).astype(
            ARRAY_DTYPE, copy=False
        ),
        times_s=np.concatenate(
            (prefix_times, trigger + relative_times[1:])
        ).astype(np.float64, copy=False),
        linear_velocities_mps=np.concatenate(
            (prefix_linear, braking_linear[1:])
        ).astype(np.float64, copy=False),
        angular_velocities_radps=np.concatenate(
            (prefix_angular, braking_angular[1:])
        ).astype(np.float64, copy=False),
    )
    return VerificationPolicyBranch(
        branch_kind="emergency_brake",
        executed_trace=executed,
        trigger_time_s=trigger,
        planned_action_end_time_s=float(action_trace.times_s[-1]),
    )


def _interpolated_pose(
    poses: np.ndarray, times_s: np.ndarray, time_s: float
) -> np.ndarray:
    if time_s <= times_s[0] + 1e-12:
        return poses[0].astype(np.float64)
    if time_s >= times_s[-1] - 1e-12:
        return poses[-1].astype(np.float64)
    upper = int(np.searchsorted(times_s, time_s, side="right"))
    lower = upper - 1
    fraction = (time_s - times_s[lower]) / (times_s[upper] - times_s[lower])
    start = poses[lower].astype(np.float64)
    end = poses[upper].astype(np.float64)
    yaw_pair = np.unwrap(np.asarray([start[2], end[2]], dtype=np.float64))
    result = (1.0 - fraction) * start + fraction * end
    result[2] = wrap_angle((1.0 - fraction) * yaw_pair[0] + fraction * yaw_pair[1])
    return result


def compose_time_aligned_policy_trajectory(
    *,
    template_trajectory: LocalTrajectory,
    branch: VerificationPolicyBranch,
    future_dt_s: float,
    trajectory_id: str,
    source_action_id: str,
    source_nominal_trajectory_id: str,
    suffix_trajectory: LocalTrajectory | None = None,
    suffix_poses_in_parent_frame: np.ndarray | None = None,
) -> LocalTrajectory:
    """Compose an executed branch and optional replan on the original time grid."""

    if not isinstance(template_trajectory, LocalTrajectory):
        raise TypeError("template_trajectory must be a LocalTrajectory")
    if not isinstance(branch, VerificationPolicyBranch):
        raise TypeError("branch must be a VerificationPolicyBranch")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("trajectory_id must be non-empty")
    if not isinstance(source_action_id, str) or not source_action_id:
        raise ValueError("source_action_id must be non-empty")
    if not isinstance(source_nominal_trajectory_id, str) or not source_nominal_trajectory_id:
        raise ValueError("source_nominal_trajectory_id must be non-empty")
    dt_s = _finite_real(future_dt_s, name="future_dt_s")
    if dt_s <= 0.0:
        raise ValueError("future_dt_s must be positive")
    template_poses = template_trajectory.poses
    template_controls = template_trajectory.controls
    if (
        not isinstance(template_poses, np.ndarray)
        or not isinstance(template_controls, np.ndarray)
        or template_poses.dtype != ARRAY_DTYPE
        or template_controls.dtype != ARRAY_DTYPE
        or template_poses.ndim != 2
        or template_poses.shape[1] != 3
        or template_controls.shape != (template_poses.shape[0], 2)
        or not np.isfinite(template_poses).all()
        or not np.isfinite(template_controls).all()
    ):
        raise ValueError("template trajectory violates the finite float32 contract")
    if (suffix_trajectory is None) != (suffix_poses_in_parent_frame is None):
        raise ValueError("suffix trajectory and parent poses must be provided together")
    suffix_poses = None
    suffix_controls = None
    if suffix_trajectory is not None:
        if not isinstance(suffix_trajectory, LocalTrajectory):
            raise TypeError("suffix_trajectory must be a LocalTrajectory")
        suffix_poses = suffix_poses_in_parent_frame
        suffix_controls = suffix_trajectory.controls
        if (
            not isinstance(suffix_poses, np.ndarray)
            or suffix_poses.dtype != ARRAY_DTYPE
            or suffix_poses.shape != template_poses.shape
            or not np.isfinite(suffix_poses).all()
            or not isinstance(suffix_controls, np.ndarray)
            or suffix_controls.dtype != ARRAY_DTYPE
            or suffix_controls.shape != template_controls.shape
            or not np.isfinite(suffix_controls).all()
        ):
            raise ValueError("suffix trajectory violates the finite float32 contract")

    horizon_s = template_poses.shape[0] * dt_s
    if branch.end_time_s > horizon_s + 1e-10:
        raise ValueError("verification branch exceeds the policy horizon")
    endpoint_times = (
        np.arange(1, template_poses.shape[0] + 1, dtype=np.float64) * dt_s
    )
    output_poses = np.empty_like(template_poses)
    output_controls = np.zeros_like(template_controls)
    trace = branch.executed_trace
    suffix_times = np.arange(
        0, template_poses.shape[0] + 1, dtype=np.float64
    ) * dt_s
    suffix_anchors = None
    if suffix_poses is not None:
        suffix_anchors = np.concatenate(
            (branch.end_pose[None, :], suffix_poses), axis=0
        ).astype(ARRAY_DTYPE, copy=False)
    for index, time_s in enumerate(endpoint_times):
        if time_s <= branch.end_time_s + 1e-12:
            output_poses[index] = _interpolated_pose(
                trace.poses, trace.times_s, float(time_s)
            ).astype(ARRAY_DTYPE)
            output_controls[index, 0] = np.float32(
                np.interp(time_s, trace.times_s, trace.linear_velocities_mps)
            )
            output_controls[index, 1] = np.float32(
                np.interp(time_s, trace.times_s, trace.angular_velocities_radps)
            )
            continue
        elapsed_s = float(time_s - branch.end_time_s)
        if suffix_anchors is None:
            output_poses[index] = branch.end_pose
            continue
        output_poses[index] = _interpolated_pose(
            suffix_anchors, suffix_times, elapsed_s
        ).astype(ARRAY_DTYPE)
        control_index = min(
            suffix_controls.shape[0] - 1,
            max(0, int(np.ceil(elapsed_s / dt_s - 1e-12)) - 1),
        )
        output_controls[index] = suffix_controls[control_index]

    metadata = {
        **template_trajectory.metadata,
        "pose_time_layout_version": POSE_TIME_LAYOUT_VERSION,
        "response_version": VERIFICATION_RESPONSE_VERSION,
        "label_side_policy_trajectory": True,
        "absolute_time_aligned": True,
        "source_action_id": source_action_id,
        "source_nominal_trajectory_id": source_nominal_trajectory_id,
        "branch_kind": branch.branch_kind,
        "branch_trigger_time_s": branch.trigger_time_s,
        "branch_end_time_s": branch.end_time_s,
        "planned_action_end_time_s": branch.planned_action_end_time_s,
        "branch_end_pose": [float(value) for value in branch.end_pose],
        "branch_end_control": [float(value) for value in branch.end_control],
        "suffix_trajectory_id": (
            None if suffix_trajectory is None else suffix_trajectory.trajectory_id
        ),
    }
    if branch.branch_kind == "emergency_brake" and suffix_trajectory is None:
        metadata["response_id"] = "emergency_brake"
    return replace(
        template_trajectory,
        trajectory_id=trajectory_id,
        poses=np.ascontiguousarray(output_poses, dtype=ARRAY_DTYPE),
        controls=np.ascontiguousarray(output_controls, dtype=ARRAY_DTYPE),
        metadata=metadata,
    )


def build_reactive_braking_trajectory(
    *,
    nominal_trajectory: LocalTrajectory,
    start_pose: Any,
    response_time_s: float,
    braking_deceleration_mps2: float,
    future_dt_s: float,
    angular_deceleration_radps2: float = DEFAULT_ANGULAR_DECELERATION_RADPS2,
) -> LocalTrajectory:
    """Follow nominal controls until response time, then brake to a full stop.

    Response time is conservatively quantized to the next frozen future endpoint.
    This keeps the returned trajectory on the exact Long40 time grid used by
    the existing risk ground truth.
    """

    if not isinstance(nominal_trajectory, LocalTrajectory):
        raise TypeError("nominal_trajectory must be a LocalTrajectory")
    poses = nominal_trajectory.poses
    controls = nominal_trajectory.controls
    if (
        not isinstance(poses, np.ndarray)
        or not isinstance(controls, np.ndarray)
        or poses.dtype != ARRAY_DTYPE
        or controls.dtype != ARRAY_DTYPE
        or poses.ndim != 2
        or controls.shape != (poses.shape[0], 2)
        or poses.shape[1] != 3
        or not np.isfinite(poses).all()
        or not np.isfinite(controls).all()
    ):
        raise ValueError("nominal trajectory arrays violate the float32 contract")
    start = _pose(start_pose, name="start_pose")
    response = _finite_real(response_time_s, name="response_time_s")
    dt = _finite_real(future_dt_s, name="future_dt_s")
    linear_deceleration = _finite_real(
        braking_deceleration_mps2, name="braking_deceleration_mps2"
    )
    angular_deceleration = _finite_real(
        angular_deceleration_radps2,
        name="angular_deceleration_radps2",
    )
    horizon = poses.shape[0] * dt
    if response < 0.0 or response > horizon:
        raise ValueError("response_time_s must lie within the future horizon")
    if dt <= 0.0 or linear_deceleration <= 0.0 or angular_deceleration <= 0.0:
        raise ValueError("time step and braking decelerations must be positive")

    response_interval = min(
        poses.shape[0], int(np.ceil(response / dt - 1e-12))
    )
    output_poses = np.empty_like(poses)
    output_controls = np.empty_like(controls)
    current_pose = start
    braking_linear: float | None = None
    braking_angular: float | None = None
    for index in range(poses.shape[0]):
        if index < response_interval:
            average_linear = float(controls[index, 0])
            average_angular = float(controls[index, 1])
        else:
            if braking_linear is None or braking_angular is None:
                source_index = min(index, controls.shape[0] - 1)
                braking_linear = float(controls[source_index, 0])
                braking_angular = float(controls[source_index, 1])
            next_linear = _approach_zero(
                braking_linear, linear_deceleration * dt
            )
            next_angular = _approach_zero(
                braking_angular, angular_deceleration * dt
            )
            average_linear = 0.5 * (braking_linear + next_linear)
            average_angular = 0.5 * (braking_angular + next_angular)
            braking_linear = next_linear
            braking_angular = next_angular
        current_pose = integrate_twist(
            current_pose,
            v=average_linear,
            omega=average_angular,
            dt_s=dt,
        )
        output_poses[index] = current_pose.astype(ARRAY_DTYPE)
        output_controls[index] = np.asarray(
            [average_linear, average_angular], dtype=ARRAY_DTYPE
        )

    response_endpoint_s = response_interval * dt
    metadata = {
        **nominal_trajectory.metadata,
        "pose_time_layout_version": POSE_TIME_LAYOUT_VERSION,
        "response_version": VERIFICATION_RESPONSE_VERSION,
        "response_id": "emergency_brake",
        "source_nominal_trajectory_id": nominal_trajectory.trajectory_id,
        "requested_response_time_s": response,
        "response_endpoint_time_s": response_endpoint_s,
        "braking_deceleration_mps2": linear_deceleration,
        "angular_deceleration_radps2": angular_deceleration,
        "label_side_policy_trajectory": True,
    }
    return replace(
        nominal_trajectory,
        trajectory_id=(
            f"policy::brake@q{response_interval:04d}::"
            f"{nominal_trajectory.trajectory_id}"
        ),
        poses=np.ascontiguousarray(output_poses, dtype=ARRAY_DTYPE),
        controls=np.ascontiguousarray(output_controls, dtype=ARRAY_DTYPE),
        metadata=metadata,
    )


__all__ = (
    "DEFAULT_ANGULAR_DECELERATION_RADPS2",
    "VERIFICATION_RESPONSE_VERSION",
    "VerificationPolicyBranch",
    "build_completed_action_branch",
    "build_observe_and_replan_branch",
    "build_reactive_braking_branch",
    "build_reactive_braking_trajectory",
    "compose_time_aligned_policy_trajectory",
    "integrate_twist",
)
