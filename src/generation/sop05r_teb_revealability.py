"""M8 v2 safe-stop verification actions with non-gating recovery diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.contracts import ARRAY_DTYPE, build_grid_spec
from src.geometry import (
    CircleOccluder,
    RectangleFootprint,
    RectangleOccluder,
    StaticOccluder,
    inflate_footprint,
)
from src.planning.lightweight_teb import (
    ObservedDynamicObstacle,
    ObservedTebRequest,
    plan_observed_lightweight_teb,
)
from src.planning.verification_actions import (
    ActionFeasibility,
    ActionTrace,
    VerificationAction,
    VerificationActionLibrary,
    check_action_trace_feasibility,
    sample_state_aware_action_trace,
)
from src.planning.verification_responses import build_reactive_braking_branch

from .anchored_human_placement import synchronized_centerline_blocking
from .event_contracts import footprint_from_spec
from .sop05r_contracts import Sop05rTebConfig
from .sop05r_teb_decision_state import _to_decision_frame
from .sop05r_teb_event_sampler import Sop05rTebMotherCandidate
from .sop05r_teb_safe_stop_contract import (
    SOP05R_TEB_SAFE_STOP_LABEL_VERSION,
    Sop05rTebSafeStopConfig,
)


SOP05R_TEB_SAFE_STOP_AUDIT_VERSION = "sop05r_teb_safe_stop_audit_v2"


@dataclass(frozen=True)
class Sop05rTebSafeStopRequest:
    mother: Sop05rTebMotherCandidate
    action_library: VerificationActionLibrary
    base_config: Mapping[str, object]
    teb_config: Sop05rTebConfig
    safe_stop_config: Sop05rTebSafeStopConfig


@dataclass(frozen=True)
class TebPostStopRecoveryDiagnostic:
    """Observed same-goal replanning result after a verified safe stop.

    This is intentionally diagnostic-only: a blocked recovery route must never
    negate a successful hazard revelation and safe emergency stop.
    """

    evaluated: bool
    planner_branch: str | None
    route_available: bool | None
    shared_goal_world_pose: tuple[float, float, float]
    replanned_task_cost: float | None
    rejection_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "evaluated": self.evaluated,
            "planner_branch": self.planner_branch,
            "route_available": self.route_available,
            "shared_goal_world_pose": list(self.shared_goal_world_pose),
            "replanned_task_cost": self.replanned_task_cost,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class TebActionSafeStopEvidence:
    action_id: str
    target_hidden_at_decision: bool
    first_visible_time_s: float | None
    completed_action_feasible_diagnostic: bool
    matched_wait_action_id: str
    matched_wait_feasible_diagnostic: bool
    matched_wait_first_visible_time_s: float | None
    matched_wait_visibility_advantage_diagnostic: bool
    braking_margin_s: float
    braking_margin_ok: bool
    braking_end_time_s: float | None
    prefix_and_brake_feasible: bool | None
    safe_stop_revealable: bool
    safe_stop_rejection_reason: str | None
    post_stop_recovery: TebPostStopRecoveryDiagnostic

    def as_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "target_hidden_at_decision": self.target_hidden_at_decision,
            "first_visible_time_s": self.first_visible_time_s,
            "completed_action_feasible_diagnostic": (
                self.completed_action_feasible_diagnostic
            ),
            "matched_wait_action_id": self.matched_wait_action_id,
            "matched_wait_feasible_diagnostic": (
                self.matched_wait_feasible_diagnostic
            ),
            "matched_wait_first_visible_time_s": (
                self.matched_wait_first_visible_time_s
            ),
            "matched_wait_visibility_advantage_diagnostic": (
                self.matched_wait_visibility_advantage_diagnostic
            ),
            "braking_margin_s": self.braking_margin_s,
            "braking_margin_ok": self.braking_margin_ok,
            "braking_end_time_s": self.braking_end_time_s,
            "prefix_and_brake_feasible": self.prefix_and_brake_feasible,
            "safe_stop_revealable": self.safe_stop_revealable,
            "safe_stop_rejection_reason": self.safe_stop_rejection_reason,
            "post_stop_recovery": self.post_stop_recovery.as_dict(),
        }


@dataclass(frozen=True)
class TebSafeStopAudit:
    version: str
    label_definition_version: str
    safe_stop_config_digest: str
    event_id: str
    actions: tuple[TebActionSafeStopEvidence, ...]
    safe_stop_action_ids: tuple[str, ...]
    natural_difficult: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "label_definition_version": self.label_definition_version,
            "safe_stop_config_digest": self.safe_stop_config_digest,
            "event_id": self.event_id,
            "actions": [action.as_dict() for action in self.actions],
            "safe_stop_action_ids": list(self.safe_stop_action_ids),
            "natural_difficult": self.natural_difficult,
        }


def build_teb_safe_stop_request(
    *,
    mother: Sop05rTebMotherCandidate,
    action_library: VerificationActionLibrary,
    base_config: Mapping[str, object],
    teb_config: Sop05rTebConfig,
    safe_stop_config: Sop05rTebSafeStopConfig,
) -> Sop05rTebSafeStopRequest:
    """Build the only supported M8 request; v1 is deliberately not accepted."""

    if not isinstance(mother, Sop05rTebMotherCandidate):
        raise TypeError("mother must be a Sop05rTebMotherCandidate")
    if not isinstance(action_library, VerificationActionLibrary):
        raise TypeError("action_library must be a VerificationActionLibrary")
    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    if not isinstance(teb_config, Sop05rTebConfig):
        raise TypeError("teb_config must be a Sop05rTebConfig")
    if not isinstance(safe_stop_config, Sop05rTebSafeStopConfig):
        raise TypeError("safe_stop_config must be a Sop05rTebSafeStopConfig")
    if safe_stop_config.label_definition_version != (
        SOP05R_TEB_SAFE_STOP_LABEL_VERSION
    ):
        raise ValueError("unsupported M8 safe-stop label definition")
    if not np.isclose(
        safe_stop_config.braking_margin_s,
        teb_config.occlusion.braking_margin_s,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "M8 braking margin must match the source TEB occlusion contract"
        )
    build_grid_spec(dict(base_config))
    return Sop05rTebSafeStopRequest(
        mother=mother,
        action_library=action_library,
        base_config=dict(base_config),
        teb_config=teb_config,
        safe_stop_config=safe_stop_config,
    )


def _typed_occluders(
    metadata: tuple[dict, ...],
) -> tuple[StaticOccluder, ...]:
    result: list[StaticOccluder] = []
    for item in metadata:
        shape = item.get("shape")
        if shape == "circle":
            result.append(
                CircleOccluder(
                    occluder_id=str(item["occluder_id"]),
                    semantic_type=str(item["semantic_type"]),
                    center_xy=np.asarray(item["center_xy"], dtype=ARRAY_DTYPE),
                    radius_m=float(item["radius_m"]),
                )
            )
        elif shape == "rectangle":
            result.append(
                RectangleOccluder(
                    occluder_id=str(item["occluder_id"]),
                    semantic_type=str(item["semantic_type"]),
                    pose=np.asarray(item["pose"], dtype=ARRAY_DTYPE),
                    length_m=float(item["length_m"]),
                    width_m=float(item["width_m"]),
                )
            )
        else:
            raise ValueError("unsupported v2 occluder metadata")
    return tuple(result)


def _pose_at_times(
    current_pose: np.ndarray,
    future_poses: np.ndarray,
    times_s: np.ndarray,
    *,
    dt_s: float,
) -> np.ndarray:
    anchors = np.vstack((current_pose, future_poses)).astype(np.float64)
    anchor_times = np.arange(anchors.shape[0], dtype=np.float64) * dt_s
    times = np.clip(np.asarray(times_s, dtype=np.float64), 0.0, anchor_times[-1])
    result = np.empty((times.size, 3), dtype=np.float64)
    result[:, 0] = np.interp(times, anchor_times, anchors[:, 0])
    result[:, 1] = np.interp(times, anchor_times, anchors[:, 1])
    result[:, 2] = np.interp(times, anchor_times, np.unwrap(anchors[:, 2]))
    return result


def _first_visible_time(
    *,
    action_trace: ActionTrace,
    target_current: np.ndarray,
    target_future: np.ndarray,
    occluders: tuple[StaticOccluder, ...],
    dt_s: float,
    epsilon_m: float,
) -> float | None:
    target = _pose_at_times(
        target_current,
        target_future,
        action_trace.times_s,
        dt_s=dt_s,
    )
    blocked, _ = synchronized_centerline_blocking(
        action_trace.poses[:, :2],
        target[:, :2],
        occluders,
        epsilon_m=epsilon_m,
    )
    visible = np.flatnonzero(~blocked)
    return None if visible.size == 0 else float(action_trace.times_s[int(visible[0])])


def _target_hidden_at_decision(
    *,
    action_trace: ActionTrace,
    target_current: np.ndarray,
    occluders: tuple[StaticOccluder, ...],
    epsilon_m: float,
) -> bool:
    blocked, _ = synchronized_centerline_blocking(
        action_trace.poses[:1, :2],
        np.asarray(target_current[None, :2], dtype=np.float64),
        occluders,
        epsilon_m=epsilon_m,
    )
    return bool(blocked[0])


def _matched_wait(action: VerificationAction) -> VerificationAction:
    return VerificationAction(
        action_id=f"{action.action_id}-matched-wait",
        duration_s=action.duration_s,
        delta_forward_m=0.0,
        delta_yaw_rad=0.0,
    )


def _action_trace(
    request: Sop05rTebSafeStopRequest,
    action: VerificationAction,
) -> ActionTrace:
    state = request.mother.decision_state.base_state
    return sample_state_aware_action_trace(
        state.robot_history[-1],
        action,
        robot_state=state.robot_state,
        braking_deceleration_mps2=float(
            request.teb_config.planner.max_linear_acceleration_mps2
        ),
        angular_deceleration_radps2=(
            request.teb_config.planner.max_angular_acceleration_radps2
        ),
    )


def _trace_feasibility(
    request: Sop05rTebSafeStopRequest,
    trace: ActionTrace,
) -> ActionFeasibility:
    mother = request.mother
    grid = build_grid_spec(dict(request.base_config))
    robot = request.base_config["robot"]
    footprint = inflate_footprint(
        RectangleFootprint(
            float(robot["length_m"]),
            float(robot["width_m"]),
        ),
        float(robot["inflation_m"]),
    )
    target = mother.event.target
    target_poses = np.vstack(
        (target.current_pose, target.future_poses)
    ).astype(ARRAY_DTYPE)
    return check_action_trace_feasibility(
        trace,
        robot_footprint=footprint,
        static_occupancy=mother.decision_state.static_occupancy_local,
        grid=grid,
        dynamic_object_poses={
            target.target_dynamic_object_id: target_poses,
        },
        dynamic_object_footprints={
            target.target_dynamic_object_id: footprint_from_spec(
                target.footprint_spec
            ),
        },
        dynamic_dt_s=float(request.base_config["bev"]["future_dt_s"]),
    )


def _hold_trace_until(
    trace: ActionTrace,
    *,
    end_time_s: float,
) -> ActionTrace:
    """Keep a stopped robot in place until the original collision horizon."""

    final_time = float(trace.times_s[-1])
    if end_time_s < final_time - 1e-12:
        raise ValueError("hold horizon must not precede the brake completion")
    if (
        trace.linear_velocities_mps[-1] != 0.0
        or trace.angular_velocities_radps[-1] != 0.0
    ):
        raise ValueError("safe-stop trace must reach rest before holding")
    if np.isclose(end_time_s, final_time, rtol=0.0, atol=1e-12):
        return trace
    return ActionTrace(
        poses=np.concatenate((trace.poses, trace.poses[-1:]), axis=0).astype(
            ARRAY_DTYPE, copy=False
        ),
        times_s=np.concatenate(
            (trace.times_s, np.asarray([end_time_s], dtype=np.float64))
        ),
        linear_velocities_mps=np.concatenate(
            (trace.linear_velocities_mps, np.asarray([0.0], dtype=np.float64))
        ),
        angular_velocities_radps=np.concatenate(
            (trace.angular_velocities_radps, np.asarray([0.0], dtype=np.float64))
        ),
    )


def _not_evaluated_recovery(
    goal: tuple[float, float, float],
) -> TebPostStopRecoveryDiagnostic:
    return TebPostStopRecoveryDiagnostic(
        evaluated=False,
        planner_branch=None,
        route_available=None,
        shared_goal_world_pose=goal,
        replanned_task_cost=None,
        rejection_reason=None,
    )


def _post_stop_recovery_diagnostic(
    request: Sop05rTebSafeStopRequest,
    *,
    stop_trace: ActionTrace,
    occluders: tuple[StaticOccluder, ...],
    goal: tuple[float, float, float],
) -> TebPostStopRecoveryDiagnostic:
    """Attempt same-goal recovery after stopping; this never gates the label."""

    mother = request.mother
    decision = mother.decision_state
    stop_time_s = float(stop_trace.times_s[-1])
    local_goal = _to_decision_frame(
        np.asarray([decision.shared_goal_world_pose], dtype=np.float64),
        decision.decision_pose_world,
    )[0].astype(ARRAY_DTYPE)
    dt_s = float(request.base_config["bev"]["future_dt_s"])
    target = mother.event.target
    observed = _pose_at_times(
        target.current_pose,
        target.future_poses,
        np.asarray([stop_time_s], dtype=np.float64),
        dt_s=dt_s,
    )[0]
    before_time_s = max(0.0, stop_time_s - dt_s)
    before = _pose_at_times(
        target.current_pose,
        target.future_poses,
        np.asarray([before_time_s], dtype=np.float64),
        dt_s=dt_s,
    )[0]
    elapsed_s = max(dt_s, stop_time_s - before_time_s)
    velocity = (observed[:2] - before[:2]) / elapsed_s
    footprint_spec = target.footprint_spec["footprint"]
    radius = (
        float(footprint_spec["radius_m"])
        if footprint_spec.get("kind") == "circle"
        else 0.5
        * float(
            np.hypot(footprint_spec["length_m"], footprint_spec["width_m"])
        )
    )
    result = plan_observed_lightweight_teb(
        ObservedTebRequest(
            start_pose=np.asarray(stop_trace.poses[-1], dtype=ARRAY_DTYPE),
            initial_control=np.asarray([0.0, 0.0], dtype=ARRAY_DTYPE),
            local_goal_world_pose=local_goal,
            static_occupancy=decision.static_occupancy_local,
            occluders=occluders,
            base_config=request.base_config,
            planner_config=request.teb_config.planner,
            observed_dynamic_obstacles=(
                ObservedDynamicObstacle(
                    object_id=target.target_dynamic_object_id,
                    observed_pose=np.asarray(observed, dtype=ARRAY_DTYPE),
                    observed_velocity_xy=np.asarray(velocity, dtype=ARRAY_DTYPE),
                    footprint_radius_m=radius,
                    observation_age_s=0.0,
                ),
            ),
        )
    )
    return TebPostStopRecoveryDiagnostic(
        evaluated=True,
        planner_branch="observed_dynamic",
        route_available=result.route is not None,
        shared_goal_world_pose=goal,
        replanned_task_cost=(
            None if result.route is None else result.route.task_cost
        ),
        rejection_reason=result.rejection_reason,
    )


def evaluate_teb_safe_stop_revealability(
    request: Sop05rTebSafeStopRequest,
) -> TebSafeStopAudit:
    """Evaluate M8 v2: reveal a hidden target, then safely brake and hold.

    Same-goal TEB recovery and matched-wait observations are retained only as
    diagnostics.  Neither may invalidate a safe-stop verification action.
    """

    if not isinstance(request, Sop05rTebSafeStopRequest):
        raise TypeError("request must be a Sop05rTebSafeStopRequest")
    mother = request.mother
    occluders = _typed_occluders(mother.event.world.occluders)
    target = mother.event.target
    dt_s = float(request.base_config["bev"]["future_dt_s"])
    horizon_s = float(request.base_config["bev"]["future_steps"]) * dt_s
    collision_time_s = float(
        mother.collision.first_collision_time_after_decision_s
    )
    if collision_time_s > horizon_s + 1e-12:
        raise ValueError("M8 collision horizon exceeds available target motion")
    goal = tuple(
        float(value) for value in mother.trajectory_record.shared_goal_world_pose
    )
    evidence: list[TebActionSafeStopEvidence] = []
    safe_stop_action_ids: list[str] = []

    for action in request.action_library.actions:
        trace = _action_trace(request, action)
        wait = _matched_wait(action)
        wait_trace = _action_trace(request, wait)
        hidden_at_decision = _target_hidden_at_decision(
            action_trace=trace,
            target_current=target.current_pose,
            occluders=occluders,
            epsilon_m=(
                request.teb_config.occlusion.centerline_intersection_epsilon_m
            ),
        )
        visible = _first_visible_time(
            action_trace=trace,
            target_current=target.current_pose,
            target_future=target.future_poses,
            occluders=occluders,
            dt_s=dt_s,
            epsilon_m=(
                request.teb_config.occlusion.centerline_intersection_epsilon_m
            ),
        )
        wait_visible = _first_visible_time(
            action_trace=wait_trace,
            target_current=target.current_pose,
            target_future=target.future_poses,
            occluders=occluders,
            dt_s=dt_s,
            epsilon_m=(
                request.teb_config.occlusion.centerline_intersection_epsilon_m
            ),
        )
        completed_feasibility = _trace_feasibility(request, trace)
        wait_feasibility = _trace_feasibility(request, wait_trace)
        visibility_advantage = visible is not None and (
            wait_visible is None
            or wait_visible - visible
            >= request.teb_config.revealability.minimum_visibility_lead_s
        )
        braking_margin_ok = visible is not None and (
            collision_time_s - visible
            >= request.safe_stop_config.braking_margin_s
        )
        rejection: str | None = None
        braking_end_time_s: float | None = None
        prefix_and_brake_feasible: bool | None = None
        recovery = _not_evaluated_recovery(goal)
        safe_stop = False

        if action.action_id == "stop_scan" and not request.safe_stop_config.allow_stop_scan:
            rejection = "stop_scan_excluded"
        elif request.safe_stop_config.require_hidden_at_decision and not hidden_at_decision:
            rejection = "target_visible_at_decision"
        elif visible is None:
            rejection = "target_not_revealed"
        elif not braking_margin_ok:
            rejection = "braking_margin_insufficient"
        else:
            branch = build_reactive_braking_branch(
                action_trace=trace,
                response_time_s=visible,
                braking_deceleration_mps2=(
                    request.teb_config.planner.max_linear_acceleration_mps2
                ),
                angular_deceleration_radps2=(
                    request.teb_config.planner.max_angular_acceleration_radps2
                ),
                future_horizon_s=horizon_s,
            )
            braking_end_time_s = branch.end_time_s
            if braking_end_time_s > collision_time_s + 1e-12:
                rejection = "braking_completion_after_collision"
            else:
                safe_trace = _hold_trace_until(
                    branch.executed_trace,
                    end_time_s=collision_time_s,
                )
                stop_feasibility = _trace_feasibility(request, safe_trace)
                prefix_and_brake_feasible = stop_feasibility.feasible
                if not stop_feasibility.feasible:
                    rejection = f"safe_stop_{stop_feasibility.reason}"
                else:
                    safe_stop = True
                    recovery = _post_stop_recovery_diagnostic(
                        request,
                        stop_trace=branch.executed_trace,
                        occluders=occluders,
                        goal=goal,
                    )
        if safe_stop:
            safe_stop_action_ids.append(action.action_id)
        evidence.append(
            TebActionSafeStopEvidence(
                action_id=action.action_id,
                target_hidden_at_decision=hidden_at_decision,
                first_visible_time_s=visible,
                completed_action_feasible_diagnostic=completed_feasibility.feasible,
                matched_wait_action_id=wait.action_id,
                matched_wait_feasible_diagnostic=wait_feasibility.feasible,
                matched_wait_first_visible_time_s=wait_visible,
                matched_wait_visibility_advantage_diagnostic=visibility_advantage,
                braking_margin_s=request.safe_stop_config.braking_margin_s,
                braking_margin_ok=braking_margin_ok,
                braking_end_time_s=braking_end_time_s,
                prefix_and_brake_feasible=prefix_and_brake_feasible,
                safe_stop_revealable=safe_stop,
                safe_stop_rejection_reason=rejection,
                post_stop_recovery=recovery,
            )
        )
    return TebSafeStopAudit(
        version=SOP05R_TEB_SAFE_STOP_AUDIT_VERSION,
        label_definition_version=(
            request.safe_stop_config.label_definition_version
        ),
        safe_stop_config_digest=request.safe_stop_config.digest,
        event_id=mother.event.generated_event_id,
        actions=tuple(evidence),
        safe_stop_action_ids=tuple(safe_stop_action_ids),
        natural_difficult=not safe_stop_action_ids,
    )


def validate_teb_safe_stop_audit_payload(payload: Mapping[str, object]) -> None:
    """Fail closed for legacy, unknown, or structurally incomplete M8 payloads."""

    expected_keys = {
        "version",
        "label_definition_version",
        "safe_stop_config_digest",
        "event_id",
        "actions",
        "safe_stop_action_ids",
        "natural_difficult",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError("M8 safe-stop audit payload keys are invalid")
    if payload["version"] != SOP05R_TEB_SAFE_STOP_AUDIT_VERSION:
        raise ValueError("unsupported M8 audit version")
    if payload["label_definition_version"] != SOP05R_TEB_SAFE_STOP_LABEL_VERSION:
        raise ValueError("unsupported M8 label definition version")
    digest = payload["safe_stop_config_digest"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("M8 safe-stop config digest is invalid")
    if not isinstance(payload["event_id"], str) or not payload["event_id"]:
        raise ValueError("M8 event ID is invalid")
    actions = payload["actions"]
    action_ids = payload["safe_stop_action_ids"]
    if not isinstance(actions, list) or not isinstance(action_ids, list):
        raise ValueError("M8 action payload fields are invalid")
    if not all(isinstance(item, Mapping) for item in actions):
        raise ValueError("M8 action evidence must be mappings")
    expected_active = [
        item["action_id"]
        for item in actions
        if item.get("safe_stop_revealable") is True
    ]
    if action_ids != expected_active:
        raise ValueError("M8 safe-stop action IDs disagree with evidence")
    if (
        not isinstance(payload["natural_difficult"], bool)
        or payload["natural_difficult"] != (not bool(action_ids))
    ):
        raise ValueError("M8 natural-difficult flag disagrees with evidence")


__all__ = (
    "SOP05R_TEB_SAFE_STOP_AUDIT_VERSION",
    "Sop05rTebSafeStopRequest",
    "TebActionSafeStopEvidence",
    "TebPostStopRecoveryDiagnostic",
    "TebSafeStopAudit",
    "build_teb_safe_stop_request",
    "evaluate_teb_safe_stop_revealability",
    "validate_teb_safe_stop_audit_payload",
)
