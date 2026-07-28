"""SOP05R adapters for deterministic replay and paired-event visuals."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.contracts import BaseState, OracleContext, build_grid_spec
from src.evaluation.seen_occluded_visuals import (
    PAIRED_PANEL_ORDER,
    VisualArtifactResult,
    VisualAuditBundle,
    VisualVariant,
    render_visual_artifacts,
)
from src.generation.paired_variants import PairedEventGroup
from src.generation.event_sampler import GeneratedEvent
from src.generation.sop05r_contracts import SOP05R_GENERATOR_VERSION
from src.generation.sop05r_trajectory_store import Sop05rTrajectoryRecord
from src.planning.verification_actions import (
    VerificationActionLibrary,
    sample_state_aware_action_trace,
)


@dataclass(frozen=True)
class Sop05rVisualRequest:
    event: GeneratedEvent
    trajectory_record: Sop05rTrajectoryRecord
    base_state: BaseState
    oracle_context: OracleContext
    pair_group: PairedEventGroup
    base_config: Mapping[str, Any]
    action_library: VerificationActionLibrary
    verification_action_id: str

    def with_event(self, event: GeneratedEvent) -> "Sop05rVisualRequest":
        if not isinstance(event, GeneratedEvent):
            raise TypeError("event must be a GeneratedEvent")
        if event.generated_event_id != self.event.generated_event_id:
            raise ValueError("replacement event ID differs from visual request")
        return replace(self, event=event)


def _validate_request(request: Sop05rVisualRequest) -> None:
    if not isinstance(request, Sop05rVisualRequest):
        raise TypeError("request must be a Sop05rVisualRequest")
    event = request.event
    if event.world.metadata.get("generator_algorithm_version") != (
        SOP05R_GENERATOR_VERSION
    ):
        raise ValueError("visual event is not a SOP05R mother")
    if event.world.metadata.get("target_history_visibility_regime") != (
        "seen_then_occluded"
    ):
        raise ValueError("SOP05R replay visuals require seen_then_occluded")
    record = request.trajectory_record
    if (
        record.event_id != event.generated_event_id
        or record.base_state_id != event.world.base_state_id
        or record.nominal_trajectory_id
        != event.world.metadata.get("nominal_trajectory_id")
    ):
        raise ValueError("SOP05R visual event/trajectory join mismatch")
    if (
        request.base_state.state_id != event.world.base_state_id
        or request.oracle_context.base_state_id != event.world.base_state_id
    ):
        raise ValueError("SOP05R visual base/context join mismatch")
    group = request.pair_group
    if (
        not group.is_complete
        or not group.eligible_for_strict_evaluation
        or group.coverage_mask != (True,) * len(PAIRED_PANEL_ORDER)
        or tuple(variant.variant_kind for variant in group.variants)
        != PAIRED_PANEL_ORDER
        or group.missing_variant_reasons
    ):
        raise ValueError("SOP05R visual pair group must be a complete sixpack")
    collision = group.by_kind["collision"]
    if collision.target is None or not np.array_equal(
        collision.target.history_poses,
        event.target.history_poses,
    ):
        raise ValueError("SOP05R visual collision mother differs from event")
    if not isinstance(request.action_library, VerificationActionLibrary):
        raise TypeError("action_library must be a VerificationActionLibrary")
    action = request.action_library.by_id.get(request.verification_action_id)
    if action is None or action.action_id == "stop_scan":
        raise ValueError("SOP05R visual requires a moving verification action")
    build_grid_spec(dict(request.base_config))


def _visual_variants(group: PairedEventGroup) -> tuple[VisualVariant, ...]:
    rows = []
    for variant in group.variants:
        if variant.variant_kind == "empty_blind_spot":
            rows.append(
                VisualVariant(
                    kind=variant.variant_kind,
                    target_history=None,
                    target_future=None,
                    visibility_history=None,
                    min_clearance_m=None,
                    time_to_min_clearance_s=None,
                    temporal_offset_s=None,
                )
            )
            continue
        if variant.target is None or variant.target_visibility_history is None:
            raise ValueError(f"SOP05R {variant.variant_kind} visual target is missing")
        rows.append(
            VisualVariant(
                kind=variant.variant_kind,
                target_history=variant.target.history_poses.copy(),
                target_future=variant.target.future_poses.copy(),
                visibility_history=variant.target_visibility_history.copy(),
                min_clearance_m=variant.min_clearance_m,
                time_to_min_clearance_s=variant.time_to_min_clearance_s,
                temporal_offset_s=variant.temporal_offset_s,
            )
        )
    return tuple(rows)


def build_sop05r_visual_bundle(
    request: Sop05rVisualRequest,
) -> VisualAuditBundle:
    _validate_request(request)
    event = request.event
    record = request.trajectory_record
    routes_by_id = {
        route.trajectory.trajectory_id: route for route in record.routes
    }
    nominal = routes_by_id[record.nominal_trajectory_id]
    action = request.action_library.by_id[request.verification_action_id]
    trace = sample_state_aware_action_trace(
        request.base_state.robot_history[-1],
        action,
        robot_state=request.base_state.robot_state,
        braking_deceleration_mps2=float(
            nominal.trajectory.metadata["braking_deceleration_mps2"]
        ),
    )
    robot = request.base_config["robot"]
    inflated_length = float(robot["length_m"]) + 2.0 * float(
        robot["inflation_m"]
    )
    inflated_width = float(robot["width_m"]) + 2.0 * float(
        robot["inflation_m"]
    )
    obstacle_margin = 0.5 * float(np.hypot(inflated_length, inflated_width))
    return VisualAuditBundle(
        event_id=event.generated_event_id,
        base_state=request.base_state,
        oracle_context=request.oracle_context,
        trajectory=nominal.trajectory,
        static_occupancy=event.world.static_occupancy.copy(),
        occluders=tuple(dict(item) for item in event.world.occluders),
        variants=_visual_variants(request.pair_group),
        grid=build_grid_spec(dict(request.base_config)),
        robot_length_m=float(robot["length_m"]),
        robot_width_m=float(robot["width_m"]),
        source_static_occupancy=request.base_state.static_map_local.copy(),
        planner_routes_world=tuple(route.poses_world.copy() for route in record.routes),
        planner_slot_ids=tuple(route.slot_id for route in record.routes),
        planner_trajectory_ids=record.candidate_trajectory_ids,
        nominal_trajectory_id=record.nominal_trajectory_id,
        alternative_trajectory_ids=record.alternative_trajectory_ids,
        shared_goal_world_pose=record.shared_goal_world_pose.copy(),
        conflict_point=np.asarray(
            event.world.metadata["conflict_point"], dtype=np.float64
        ),
        inflated_obstacle_margin_m=obstacle_margin,
        verification_action_id=action.action_id,
        verification_trace=trace.poses.copy(),
    )


def render_sop05r_visual_artifacts(
    request: Sop05rVisualRequest,
    output_dir: str | Path,
) -> VisualArtifactResult:
    return render_visual_artifacts(
        build_sop05r_visual_bundle(request),
        output_dir,
    )
