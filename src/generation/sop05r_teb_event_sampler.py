"""M6 one-route mother construction with continuous robot-human collision."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.contracts import (
    ARRAY_DTYPE,
    BaseState,
    LocalTrajectory,
    OracleContext,
    OracleWorld,
    build_grid_spec,
)
from src.datasets.long_snippet_library import LongMotionSnippet
from src.geometry import (
    CircleOccluder,
    RectangleFootprint,
    RectangleOccluder,
    inflate_footprint,
)
from src.generation.event_sampler import GeneratedEvent
from src.planning.lightweight_teb import PlannedTebRoute
from src.utils.seeding import stable_digest

from .anchored_human_placement import (
    AnchoredPlacementResult,
    _sample_robot_poses,
    synchronized_centerline_blocking,
)
from .dynamic_object_transplant import (
    TransplantedDynamicObject,
    footprint_from_spec,
)
from .event_target_motion_shard import (
    EventTargetMotionRecord,
    build_event_target_motion_world_metadata,
    compute_footprint_spec_digest,
    compute_motion_array_digest,
    create_event_target_motion_record,
)
from .occluder_sampler import (
    swept_footprint_intersects_occupancy,
    synchronized_sweeps_intersect,
)
from .sop05r_contracts import Sop05rTebConfig
from .sop05r_event_sampler import (
    ContinuousCollisionEvidence,
    compute_continuous_collision_evidence,
)
from .sop05r_teb_decision_state import (
    Sop05rTebDecisionState,
    Sop05rTebDecisionStateCandidateRejected,
    _sample_context_history,
    _to_decision_frame,
    build_teb_decision_state,
)
from .sop05r_teb_templates import Sop05rTebTaskTemplate


SOP05R_TEB_MOTHER_VERSION = "sop05r_teb_one_route_mother_v2"
_TARGET_POLICY = {
    "whitelist": ["human"],
    "weights": {"human": 1.0, "carried_object": 0.0, "unknown_dynamic": 0.0},
}
_TARGET_POLICY_DIGEST = stable_digest(
    json.dumps(_TARGET_POLICY, sort_keys=True, separators=(",", ":")),
    size=16,
)


@dataclass(frozen=True)
class Sop05rTebCollisionEvidence:
    continuous: ContinuousCollisionEvidence
    first_collision_time_world_s: float
    first_collision_time_after_decision_s: float
    conflict_index: int
    collision_point_xy: np.ndarray

    def __post_init__(self) -> None:
        point = np.asarray(self.collision_point_xy)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("collision_point_xy must be finite [2]")
        point = np.array(point, dtype=np.float64, order="C", copy=True)
        point.setflags(write=False)
        object.__setattr__(self, "collision_point_xy", point)


@dataclass(frozen=True)
class Sop05rTebTrajectoryRecord:
    event_id: str
    source_base_state_id: str
    decision_state_id: str
    template_id: str
    planner_version: str
    config_digest: str
    shared_goal_world_pose: np.ndarray
    full_route: PlannedTebRoute
    nominal_trajectory: LocalTrajectory


@dataclass(frozen=True)
class Sop05rTebMotherCandidate:
    event: GeneratedEvent
    decision_state: Sop05rTebDecisionState
    placement_result: AnchoredPlacementResult
    collision: Sop05rTebCollisionEvidence
    trajectory_record: Sop05rTebTrajectoryRecord


@dataclass(frozen=True)
class Sop05rTebMotherEvaluation:
    mother: Sop05rTebMotherCandidate | None
    rejection_reason: str | None
    evidence: Mapping[str, object]


def _occluder_metadata(
    occluder: object,
    *,
    decision_pose_world: np.ndarray,
) -> dict[str, object]:
    if isinstance(occluder, CircleOccluder):
        local_center = _to_decision_frame(
            np.asarray([[*occluder.center_xy, 0.0]], dtype=np.float64),
            decision_pose_world,
        )[0, :2]
        return {
            "occluder_id": occluder.occluder_id,
            "semantic_type": occluder.semantic_type,
            "shape": "circle",
            "center_xy": [float(value) for value in local_center],
            "radius_m": occluder.radius_m,
        }
    local_pose = _to_decision_frame(
        np.asarray([occluder.pose], dtype=np.float64),
        decision_pose_world,
    )[0]
    return {
        "occluder_id": occluder.occluder_id,
        "semantic_type": occluder.semantic_type,
        "shape": "rectangle",
        "pose": [float(value) for value in local_pose],
        "length_m": occluder.length_m,
        "width_m": occluder.width_m,
    }


def _occluders_in_decision_frame(
    task_template: Sop05rTebTaskTemplate,
    *,
    decision_pose_world: np.ndarray,
) -> tuple[CircleOccluder | RectangleOccluder, ...]:
    """Rebase M4 occluders for the serialized decision-frame visibility trace."""

    local_occluders: list[CircleOccluder | RectangleOccluder] = []
    for occluder in task_template.occluders:
        if isinstance(occluder, CircleOccluder):
            center = _to_decision_frame(
                np.asarray([[*occluder.center_xy, 0.0]], dtype=np.float64),
                decision_pose_world,
            )[0, :2]
            local_occluders.append(
                CircleOccluder(
                    occluder_id=occluder.occluder_id,
                    semantic_type=occluder.semantic_type,
                    center_xy=np.asarray(center, dtype=ARRAY_DTYPE),
                    radius_m=occluder.radius_m,
                )
            )
        else:
            pose = _to_decision_frame(
                np.asarray([occluder.pose], dtype=np.float64),
                decision_pose_world,
            )[0]
            local_occluders.append(
                RectangleOccluder(
                    occluder_id=occluder.occluder_id,
                    semantic_type=occluder.semantic_type,
                    pose=np.asarray(pose, dtype=ARRAY_DTYPE),
                    length_m=occluder.length_m,
                    width_m=occluder.width_m,
                )
            )
    return tuple(local_occluders)


def _full_centerline_blockage(
    *,
    decision: Sop05rTebDecisionState,
    target: TransplantedDynamicObject,
    task_template: Sop05rTebTaskTemplate,
    epsilon_m: float,
) -> np.ndarray:
    robot = np.vstack(
        (decision.base_state.robot_history, decision.nominal_trajectory.poses)
    )
    target_poses = np.vstack((target.history_poses, target.future_poses))
    blocked, _ = synchronized_centerline_blocking(
        robot[:, :2],
        target_poses[:, :2],
        _occluders_in_decision_frame(
            task_template,
            decision_pose_world=decision.decision_pose_world,
        ),
        epsilon_m=epsilon_m,
    )
    if blocked.shape != (40,):
        raise ValueError("M6 centerline blocking must cover all Long40 samples")
    return blocked


def _target_spec(snippet: MotionSnippet) -> dict[str, object]:
    return {
        "object_type": snippet.object_type,
        "footprint": dict(snippet.footprint),
    }


def _target_dynamic_object_id(
    *,
    source_base_state: BaseState,
    task_template: Sop05rTebTaskTemplate,
    placement_result: AnchoredPlacementResult,
    snippet: LongMotionSnippet,
    teb_config: Sop05rTebConfig,
    seed: int,
) -> str:
    """Derive the target ID before rebasing a currently visible target."""

    placement = placement_result.placement
    return "sop05r-teb-target-" + stable_digest(
        "sop05r-teb-target-v2",
        source_base_state.state_id,
        task_template.template_id,
        snippet.snippet_id,
        placement.anchor.route_sample_index,
        placement.anchor.snippet_anchor_index,
        placement.rotation_rad,
        placement.temporal_scale,
        teb_config.digest,
        int(seed),
        size=16,
    )


def _collision_evidence(
    *,
    task_template: Sop05rTebTaskTemplate,
    placement_result: AnchoredPlacementResult,
    source_base_state: BaseState,
    snippet: LongMotionSnippet,
    base_config: Mapping[str, object],
    teb_config: Sop05rTebConfig,
    decision_time_s: float,
) -> tuple[Sop05rTebCollisionEvidence | None, str | None]:
    dt_s = float(base_config["bev"]["future_dt_s"])
    offsets = (np.arange(40, dtype=np.float64) - 7) * dt_s
    robot_poses = _sample_robot_poses(
        task_template,
        source_base_state,
        decision_time_s + offsets,
        dt_s=dt_s,
    )
    placement = placement_result.placement
    target_poses = np.vstack(
        (placement.history_poses, placement.future_poses)
    ).astype(np.float64)
    robot = base_config["robot"]
    robot_footprint = inflate_footprint(
        RectangleFootprint(
            float(robot["length_m"]),
            float(robot["width_m"]),
        ),
        float(robot["inflation_m"]),
    )
    continuous = compute_continuous_collision_evidence(
        robot_footprint=robot_footprint,
        robot_poses=robot_poses,
        target_footprint=footprint_from_spec(_target_spec(snippet)),
        target_poses=target_poses,
        dt_s=dt_s,
        spatial_resolution_m=0.25 * build_grid_spec(dict(base_config)).resolution_m,
    )
    if not continuous.continuous_collision:
        return None, "no_continuous_collision"
    assert continuous.first_collision_time_s is not None
    after_decision_s = continuous.first_collision_time_s - 7 * dt_s
    if after_decision_s <= 0.0:
        return None, "premature_robot_contact"
    if (
        after_decision_s
        < teb_config.occlusion.minimum_decision_to_collision_margin_s - 1e-6
    ):
        return None, "decision_margin_insufficient"
    if after_decision_s > teb_config.trajectory.future_horizon_s + 1e-6:
        return None, "collision_outside_model_suffix"
    if np.isclose(
        after_decision_s,
        teb_config.trajectory.future_horizon_s,
        rtol=0.0,
        atol=1e-6,
    ):
        return None, "endpoint_only_collision"
    world_time_s = decision_time_s + after_decision_s
    conflict_index = max(0, int(np.ceil(after_decision_s / dt_s - 1e-9)) - 1)
    robot_pose = continuous.robot_pose_at_first_collision
    target_pose = continuous.target_pose_at_first_collision
    assert robot_pose is not None and target_pose is not None
    collision_point = 0.5 * (robot_pose[:2] + target_pose[:2])
    return (
        Sop05rTebCollisionEvidence(
            continuous=continuous,
            first_collision_time_world_s=world_time_s,
            first_collision_time_after_decision_s=after_decision_s,
            conflict_index=conflict_index,
            collision_point_xy=collision_point,
        ),
        None,
    )


def _target_environment_rejection(
    *,
    decision: Sop05rTebDecisionState,
    source_oracle_context: OracleContext,
    snippet: LongMotionSnippet,
    base_config: Mapping[str, object],
) -> str | None:
    dt_s = float(base_config["bev"]["future_dt_s"])
    grid = build_grid_spec(dict(base_config))
    footprint = footprint_from_spec(_target_spec(snippet))
    target_poses = np.vstack(
        (decision.target_history_local, decision.target_future_local)
    ).astype(np.float64)
    if swept_footprint_intersects_occupancy(
        footprint,
        target_poses,
        decision.static_occupancy_local != 0.0,
        grid=grid,
    ):
        return "target_static_collision"
    offsets = (np.arange(40, dtype=np.float64) - 7) * dt_s
    sample_times = decision.decision_time_s + offsets
    for object_id in sorted(source_oracle_context.dynamic_object_history):
        context_poses_world = _sample_context_history(
            source_oracle_context,
            object_id=object_id,
            sample_times_s=sample_times,
            dt_s=dt_s,
        )
        context_poses_local = _to_decision_frame(
            context_poses_world,
            decision.decision_pose_world,
        )
        if synchronized_sweeps_intersect(
            footprint,
            target_poses,
            footprint_from_spec(source_oracle_context.dynamic_object_specs[object_id]),
            context_poses_local,
            grid=grid,
        ):
            return "target_context_collision"
    return None


def build_sop05r_teb_mother(
    *,
    base_config: Mapping[str, object],
    source_base_state: BaseState,
    source_oracle_context: OracleContext,
    teb_config: Sop05rTebConfig,
    task_template: Sop05rTebTaskTemplate,
    placement_result: AnchoredPlacementResult,
    snippet: LongMotionSnippet,
    seed: int,
) -> Sop05rTebMotherEvaluation:
    """Build one in-memory window-occlusion mother or return one rejection reason."""

    target_spec = _target_spec(snippet)
    target_id = _target_dynamic_object_id(
        source_base_state=source_base_state,
        task_template=task_template,
        placement_result=placement_result,
        snippet=snippet,
        teb_config=teb_config,
        seed=seed,
    )
    try:
        decision = build_teb_decision_state(
            task_template=task_template,
            placement_result=placement_result,
            source_base_state=source_base_state,
            source_oracle_context=source_oracle_context,
            base_config=base_config,
            teb_config=teb_config,
            seed=seed,
            target_dynamic_object_id=target_id,
            target_dynamic_object_spec=target_spec,
        )
    except Sop05rTebDecisionStateCandidateRejected as exc:
        return Sop05rTebMotherEvaluation(
            mother=None,
            rejection_reason=exc.rejection_reason,
            evidence={"template_id": task_template.template_id},
        )
    environment_rejection = _target_environment_rejection(
        decision=decision,
        source_oracle_context=source_oracle_context,
        snippet=snippet,
        base_config=base_config,
    )
    if environment_rejection is not None:
        return Sop05rTebMotherEvaluation(
            mother=None,
            rejection_reason=environment_rejection,
            evidence={
                "template_id": task_template.template_id,
                "decision_state_id": decision.decision_state_id,
            },
        )
    collision, rejection = _collision_evidence(
        task_template=task_template,
        placement_result=placement_result,
        source_base_state=source_base_state,
        snippet=snippet,
        base_config=base_config,
        teb_config=teb_config,
        decision_time_s=decision.decision_time_s,
    )
    if rejection is not None:
        return Sop05rTebMotherEvaluation(
            mother=None,
            rejection_reason=rejection,
            evidence={
                "template_id": task_template.template_id,
                "decision_state_id": decision.decision_state_id,
            },
        )
    assert collision is not None

    placement = placement_result.placement
    footprint_digest = compute_footprint_spec_digest(target_spec)
    history_poses = np.asarray(decision.target_history_local, dtype=ARRAY_DTYPE)
    current_pose = np.asarray(decision.target_history_local[-1], dtype=ARRAY_DTYPE)
    future_poses = np.asarray(decision.target_future_local, dtype=ARRAY_DTYPE)
    history_digest = compute_motion_array_digest(
        history_poses,
        field_name="target_history_poses",
    )
    future_digest = compute_motion_array_digest(
        future_poses,
        field_name="target_future_poses",
    )
    identity = stable_digest(
        SOP05R_TEB_MOTHER_VERSION,
        source_base_state.state_id,
        decision.decision_state_id,
        task_template.template_id,
        snippet.snippet_id,
        history_digest,
        future_digest,
        target_id,
        teb_config.digest,
        int(seed),
        size=16,
    )
    event_id = f"sop05r-teb-event-{identity}"
    world_id = f"sop05r-teb-world-{identity}"
    target = TransplantedDynamicObject(
        target_dynamic_object_id=target_id,
        source_object_id=snippet.source_object_id,
        snippet_id=snippet.snippet_id,
        object_type=snippet.object_type,
        footprint_spec=target_spec,
        footprint_spec_digest=footprint_digest,
        history_poses=history_poses,
        current_pose=current_pose,
        future_poses=future_poses,
        provenance={
            **dict(snippet.provenance),
            **dict(placement.provenance),
            "decision_state_id": decision.decision_state_id,
        },
    )
    target_record: EventTargetMotionRecord = create_event_target_motion_record(
        generated_event_id=event_id,
        world_id=world_id,
        base_state_id=decision.decision_state_id,
        trajectory_id=decision.nominal_trajectory.trajectory_id,
        target_dynamic_object_id=target_id,
        source_snippet_id=snippet.snippet_id,
        source_object_id=snippet.source_object_id,
        object_type=snippet.object_type,
        footprint_spec=target_spec,
        footprint_spec_digest=footprint_digest,
        target_type_policy_digest=_TARGET_POLICY_DIGEST,
        history_poses=history_poses,
        current_pose=current_pose,
        future_poses=future_poses,
    )
    dt_s = float(base_config["bev"]["future_dt_s"])
    future_times = decision.decision_time_s + (
        np.arange(1, teb_config.trajectory.future_steps + 1, dtype=np.float64) * dt_s
    )
    context_trajectories = {
        object_id: np.asarray(
            _to_decision_frame(
                _sample_context_history(
                    source_oracle_context,
                    object_id=object_id,
                    sample_times_s=future_times,
                    dt_s=dt_s,
                ),
                decision.decision_pose_world,
            ),
            dtype=ARRAY_DTYPE,
        )
        for object_id in source_oracle_context.dynamic_object_future
    }
    context_specs = {
        object_id: dict(spec)
        for object_id, spec in source_oracle_context.dynamic_object_specs.items()
    }
    world = OracleWorld(
        world_id=world_id,
        base_state_id=decision.decision_state_id,
        static_occupancy=np.asarray(
            decision.static_occupancy_local,
            dtype=ARRAY_DTYPE,
        ),
        dynamic_object_trajectories={
            **context_trajectories,
            target_id: future_poses,
        },
        dynamic_object_specs={**context_specs, target_id: target_spec},
        occluders=tuple(
            _occluder_metadata(
                occluder,
                decision_pose_world=decision.decision_pose_world,
            )
            for occluder in task_template.occluders
        ),
        blind_spot_config={
            "generator_algorithm_version": teb_config.generator_algorithm_version,
            "template_id": task_template.template_id,
        },
        random_seed=int(seed),
        metadata={
            **build_event_target_motion_world_metadata(target_record),
            "mother_version": SOP05R_TEB_MOTHER_VERSION,
            "source_base_state_id": source_base_state.state_id,
            "decision_state_id": decision.decision_state_id,
            "config_digest": teb_config.digest,
            "shared_goal_world_pose": [
                float(value) for value in task_template.local_goal_world_pose
            ],
            "first_collision_time_after_decision_s": (
                collision.first_collision_time_after_decision_s
            ),
        },
    )
    blocked = _full_centerline_blockage(
        decision=decision,
        target=target,
        task_template=task_template,
        epsilon_m=teb_config.occlusion.centerline_intersection_epsilon_m,
    )
    event = GeneratedEvent(
        generated_event_id=event_id,
        event_kind="obstacle_first_teb",
        world=world,
        target=target,
        target_motion_record=target_record,
        visibility_sequence=np.asarray(~blocked[8:], dtype=np.bool_),
        target_visibility_history=np.asarray(~blocked[:8], dtype=np.bool_),
        conflict_time_s=collision.first_collision_time_after_decision_s,
        conflict_index=collision.conflict_index,
    )
    trajectory_record = Sop05rTebTrajectoryRecord(
        event_id=event_id,
        source_base_state_id=source_base_state.state_id,
        decision_state_id=decision.decision_state_id,
        template_id=task_template.template_id,
        planner_version=task_template.route.planner_version,
        config_digest=teb_config.digest,
        shared_goal_world_pose=np.asarray(
            task_template.local_goal_world_pose,
            dtype=ARRAY_DTYPE,
        ),
        full_route=task_template.route,
        nominal_trajectory=decision.nominal_trajectory,
    )
    return Sop05rTebMotherEvaluation(
        mother=Sop05rTebMotherCandidate(
            event=event,
            decision_state=decision,
            placement_result=placement_result,
            collision=collision,
            trajectory_record=trajectory_record,
        ),
        rejection_reason=None,
        evidence={
            "event_id": event_id,
            "world_id": world_id,
            "decision_state_id": decision.decision_state_id,
            "target_id": target_id,
        },
    )
