"""M6 decision rebasing and frozen 6.4-second nominal suffix construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.contracts import (
    ARRAY_DTYPE,
    BaseState,
    LocalTrajectory,
    OracleContext,
    build_grid_spec,
)
from src.geometry import (
    RectangleFootprint,
    grid_to_world,
    inflate_footprint,
    points_in_grid,
    world_to_grid,
    wrap_angle,
)
from src.planning.query_maps import (
    TrajectoryQueryMaps,
    TrajectoryDynamicsError,
    build_local_trajectory,
    build_trajectory_query_maps,
)
from src.planning.trajectory_sampler import CandidateRollout
from src.utils.seeding import stable_digest

from .anchored_human_placement import (
    AnchoredPlacementResult,
    _sample_robot_poses,
    synchronized_long40_anchor_index,
)
from .history_visibility import classify_sop05r_seen_then_occluded_history
from .sop05r_contracts import Sop05rTebConfig
from .sop05r_teb_templates import Sop05rTebTaskTemplate


_HISTORY_STEPS = 8
_CURRENT_INDEX = 7
_FUTURE_STEPS = 32


class Sop05rTebDecisionStateCandidateRejected(ValueError):
    """Known M6 trajectory validation failure that rejects one candidate."""

    def __init__(self, rejection_reason: str) -> None:
        self.rejection_reason = rejection_reason
        super().__init__(rejection_reason)


def _readonly_array(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"expected finite array with shape {shape}")
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _route_controls_at_endpoints(
    task_template: Sop05rTebTaskTemplate,
    endpoint_times_s: np.ndarray,
    *,
    dt_s: float,
) -> np.ndarray:
    route = task_template.route
    times = np.asarray(endpoint_times_s, dtype=np.float64)
    controls = np.zeros((times.size, 2), dtype=np.float64)
    for output_index, time_s in enumerate(times):
        interval_index = int(round(float(time_s) / dt_s)) - 1
        reconstructed_time = (interval_index + 1) * dt_s
        if not np.isclose(time_s, reconstructed_time, rtol=0.0, atol=1e-5):
            raise ValueError("suffix endpoint times must lie on the frozen route grid")
        if 0 <= interval_index < route.sampled_controls.shape[0]:
            controls[output_index] = route.sampled_controls[interval_index]
    return controls


def _to_decision_frame(poses_world: np.ndarray, decision_pose: np.ndarray) -> np.ndarray:
    cosine = float(np.cos(decision_pose[2]))
    sine = float(np.sin(decision_pose[2]))
    inverse_rotation = np.asarray(((cosine, sine), (-sine, cosine)))
    local = np.empty_like(poses_world, dtype=np.float64)
    local[:, :2] = (poses_world[:, :2] - decision_pose[:2]) @ inverse_rotation.T
    local[:, 2] = wrap_angle(poses_world[:, 2] - decision_pose[2])
    return local


@dataclass(frozen=True)
class Sop05rTebDecisionState:
    """One source-bound hidden decision and its exact Schema 4.0.0 suffix."""

    decision_state_id: str
    source_base_state_id: str
    decision_time_s: float
    decision_pose_world: np.ndarray
    decision_control: np.ndarray
    robot_history_world: np.ndarray
    robot_history_local: np.ndarray
    target_history_world: np.ndarray
    target_history_local: np.ndarray
    target_future_world: np.ndarray
    target_future_local: np.ndarray
    target_visibility_history: np.ndarray
    static_occupancy_local: np.ndarray
    suffix_poses_world: np.ndarray
    suffix_controls: np.ndarray
    shared_goal_world_pose: np.ndarray
    nominal_trajectory: LocalTrajectory
    recomputed_query_maps: TrajectoryQueryMaps
    base_state: BaseState
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        specs = (
            ("decision_pose_world", (3,), ARRAY_DTYPE),
            ("decision_control", (2,), ARRAY_DTYPE),
            ("robot_history_world", (8, 3), ARRAY_DTYPE),
            ("robot_history_local", (8, 3), ARRAY_DTYPE),
            ("target_history_world", (8, 3), ARRAY_DTYPE),
            ("target_history_local", (8, 3), ARRAY_DTYPE),
            ("target_future_world", (32, 3), ARRAY_DTYPE),
            ("target_future_local", (32, 3), ARRAY_DTYPE),
            ("suffix_poses_world", (32, 3), ARRAY_DTYPE),
            ("suffix_controls", (32, 2), ARRAY_DTYPE),
            ("shared_goal_world_pose", (3,), ARRAY_DTYPE),
        )
        for name, shape, dtype in specs:
            object.__setattr__(
                self,
                name,
                _readonly_array(getattr(self, name), shape=shape, dtype=dtype),
            )
        visibility = np.asarray(self.target_visibility_history)
        if visibility.shape != (8,) or visibility.dtype != np.bool_:
            raise ValueError("target_visibility_history must be boolean [8]")
        visibility = np.array(visibility, dtype=np.bool_, copy=True)
        visibility.setflags(write=False)
        object.__setattr__(self, "target_visibility_history", visibility)
        static = np.asarray(self.static_occupancy_local)
        expected_static_shape = self.nominal_trajectory.swept_mask.shape
        if static.shape != expected_static_shape or static.dtype != ARRAY_DTYPE:
            raise ValueError("static_occupancy_local violates the decision grid")
        static = np.array(static, dtype=ARRAY_DTYPE, order="C", copy=True)
        static.setflags(write=False)
        object.__setattr__(self, "static_occupancy_local", static)


def _sample_context_history(
    oracle_context: OracleContext,
    *,
    object_id: str,
    sample_times_s: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    poses = np.vstack(
        (
            oracle_context.dynamic_object_history[object_id],
            oracle_context.dynamic_object_future[object_id],
        )
    ).astype(np.float64)
    expected_shape = (_HISTORY_STEPS + _FUTURE_STEPS, 3)
    if poses.shape != expected_shape:
        raise ValueError(
            f"context history/future must combine to shape {expected_shape}"
        )
    source_times = (
        np.arange(poses.shape[0], dtype=np.float64) - _CURRENT_INDEX
    ) * dt_s
    query = np.asarray(sample_times_s, dtype=np.float64)
    sampled = np.empty((query.size, 3), dtype=np.float64)
    sampled[:, 0] = np.interp(query, source_times, poses[:, 0])
    sampled[:, 1] = np.interp(query, source_times, poses[:, 1])
    sampled[:, 2] = wrap_angle(
        np.interp(query, source_times, np.unwrap(poses[:, 2]))
    )
    return sampled


def _rebase_static_occupancy(
    occupancy_world: np.ndarray,
    *,
    decision_pose: np.ndarray,
    base_config: Mapping[str, object],
) -> np.ndarray:
    grid = build_grid_spec(dict(base_config))
    rows, columns = np.indices((grid.height, grid.width), dtype=np.int64)
    local_points = grid_to_world(
        np.stack((rows, columns), axis=-1).reshape(-1, 2),
        grid,
    )
    cosine = float(np.cos(decision_pose[2]))
    sine = float(np.sin(decision_pose[2]))
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    world_points = local_points @ rotation.T + decision_pose[:2]
    valid = points_in_grid(world_points, grid)
    local_static = np.zeros(world_points.shape[0], dtype=ARRAY_DTYPE)
    source_indices = world_to_grid(world_points[valid], grid)
    source = np.asarray(occupancy_world)
    local_static[valid] = source[
        source_indices[:, 0],
        source_indices[:, 1],
    ]
    return local_static.reshape(grid.height, grid.width)


def build_teb_decision_state(
    *,
    task_template: Sop05rTebTaskTemplate,
    placement_result: AnchoredPlacementResult,
    source_base_state: BaseState,
    source_oracle_context: OracleContext,
    base_config: Mapping[str, object],
    teb_config: Sop05rTebConfig,
    seed: int,
    target_dynamic_object_id: str | None = None,
    target_dynamic_object_spec: Mapping[str, object] | None = None,
) -> Sop05rTebDecisionState:
    """Rebase the M4 route and M5 target around its fixed decision timestamp."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if source_oracle_context.base_state_id != source_base_state.state_id:
        raise ValueError("source oracle context does not match the source BaseState")
    dt_s = teb_config.trajectory.future_dt_s
    if not np.isclose(dt_s, 0.2, rtol=0.0, atol=1e-9):
        raise ValueError("M6 requires the frozen 0.2-second endpoint grid")
    witness = placement_result.witness
    placement = placement_result.placement
    anchor = placement.anchor
    decision_time_s = 0.0
    expected_anchor_index = synchronized_long40_anchor_index(
        route_time_s=anchor.route_time_s,
        current_index=_CURRENT_INDEX,
        future_dt_s=dt_s,
        sample_count=_HISTORY_STEPS + _FUTURE_STEPS,
    )
    if anchor.snippet_anchor_index != expected_anchor_index:
        raise ValueError("collision anchor must use one synchronized Long40 index")
    if (
        anchor.route_sample_index >= task_template.route.sample_times_s.size
        or not np.isclose(
            task_template.route.sample_times_s[anchor.route_sample_index],
            anchor.route_time_s,
            rtol=0.0,
            atol=1e-6,
        )
        or not np.allclose(
            task_template.route.sampled_poses_world[
                anchor.route_sample_index, :2
            ],
            anchor.world_position_xy,
            rtol=0.0,
            atol=1e-6,
        )
        or anchor.route_time_s
        >= task_template.route.goal_arrival_time_s - 1e-6
    ):
        raise ValueError("collision anchor must be a pre-goal fixed route endpoint")
    expected_witness_time_s = decision_time_s + (
        witness.sample_index - _CURRENT_INDEX
    ) * dt_s
    if not np.isclose(
        witness.time_s, expected_witness_time_s, rtol=0.0, atol=1e-6
    ):
        raise ValueError("occlusion witness time does not align with the history grid")
    history_offsets = (
        np.arange(_HISTORY_STEPS, dtype=np.float64) - _CURRENT_INDEX
    ) * dt_s
    robot_history_world = _sample_robot_poses(
        task_template,
        source_base_state,
        decision_time_s + history_offsets,
        dt_s=dt_s,
    )
    decision_pose = robot_history_world[-1]
    if not np.allclose(
        robot_history_world[witness.sample_index, :2],
        witness.robot_position_xy,
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError("persisted witness robot position does not match the route")
    if not np.allclose(
        placement.history_poses[witness.sample_index, :2],
        witness.target_position_xy,
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError("persisted witness target position does not match placement")

    blocked = np.zeros(_HISTORY_STEPS + _FUTURE_STEPS, dtype=np.bool_)
    blocked[list(placement_result.visibility.blocked_indices)] = True
    history_assessment = classify_sop05r_seen_then_occluded_history(
        blocked,
        decision_index=_CURRENT_INDEX,
        minimum_visible_frames=teb_config.occlusion.minimum_visible_history_frames,
        minimum_occluded_frames=(
            teb_config.occlusion.minimum_occluded_history_frames
        ),
    )
    if not history_assessment.eligible:
        raise ValueError("decision history does not satisfy the frozen visibility rule")
    if (
        history_assessment.observed_class
        != placement_result.visibility.observed_class
        or history_assessment.blocked_indices
        != placement_result.visibility.blocked_indices
    ):
        raise ValueError("persisted history visibility does not match recomputed evidence")
    target_visibility_history = ~blocked[:_HISTORY_STEPS]
    target_history_world = np.asarray(placement.history_poses, dtype=np.float64)
    target_future_world = np.asarray(placement.future_poses, dtype=np.float64)
    target_poses_world = np.vstack((target_history_world, target_future_world))
    if not np.allclose(
        target_poses_world[anchor.snippet_anchor_index, :2],
        anchor.world_position_xy,
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError("collision anchor does not match the transformed target pose")
    target_history_local = _to_decision_frame(target_history_world, decision_pose)
    target_future_local = _to_decision_frame(target_future_world, decision_pose)
    target_currently_visible = bool(target_visibility_history[-1])
    if target_currently_visible and (
        target_dynamic_object_id is None or target_dynamic_object_spec is None
    ):
        raise ValueError(
            "a currently visible target requires observed BaseState identity and spec"
        )

    suffix_offsets = np.arange(1, _FUTURE_STEPS + 1, dtype=np.float64) * dt_s
    suffix_times = decision_time_s + suffix_offsets
    suffix_poses_world = _sample_robot_poses(
        task_template,
        source_base_state,
        suffix_times,
        dt_s=dt_s,
    )
    suffix_controls = _route_controls_at_endpoints(
        task_template,
        suffix_times,
        dt_s=dt_s,
    )
    suffix_poses_local = _to_decision_frame(suffix_poses_world, decision_pose)
    trajectory_id = "teb-nominal-" + stable_digest(
        source_base_state.state_id,
        task_template.template_id,
        placement.source_snippet_id,
        decision_time_s,
        teb_config.digest,
        int(seed),
        size=12,
    )
    trajectory_base_config = dict(base_config)
    trajectory_base_config["bev"] = dict(
        base_config["bev"],
        history_steps=teb_config.trajectory.history_steps,
        future_steps=teb_config.trajectory.future_steps,
        future_dt_s=teb_config.trajectory.future_dt_s,
    )
    candidate = CandidateRollout(
        trajectory_id=trajectory_id,
        poses=np.asarray(suffix_poses_local, dtype=ARRAY_DTYPE),
        controls=np.asarray(suffix_controls, dtype=ARRAY_DTYPE),
        is_stop=bool(np.allclose(suffix_controls, 0.0, atol=1e-7)),
        is_reverse=bool(np.any(suffix_controls[:, 0] < -1e-7)),
    )
    try:
        nominal = build_local_trajectory(
            candidate,
            trajectory_base_config,
            braking_deceleration_mps2=(
                teb_config.planner.max_linear_acceleration_mps2
            ),
            task_cost=task_template.route.task_cost,
        )
        robot = base_config["robot"]
        footprint = inflate_footprint(
            RectangleFootprint(
                float(robot["length_m"]),
                float(robot["width_m"]),
            ),
            float(robot["inflation_m"]),
        )
        recomputed = build_trajectory_query_maps(
            nominal.poses,
            nominal.controls,
            grid=build_grid_spec(trajectory_base_config),
            footprint=footprint,
            dt_s=dt_s,
            braking_deceleration_mps2=(
                teb_config.planner.max_linear_acceleration_mps2
            ),
        )
    except TrajectoryDynamicsError as exc:
        raise Sop05rTebDecisionStateCandidateRejected(
            "teb_dynamics_limit"
        ) from exc
    decision_state_id = "decision-" + stable_digest(
        "sop05r-teb-decision-v1",
        source_base_state.state_id,
        task_template.template_id,
        trajectory_id,
        decision_time_s,
        size=12,
    )
    current_control = np.asarray(source_base_state.robot_state, dtype=ARRAY_DTYPE)
    supported_context_ids = (
        set(source_base_state.dynamic_object_ids)
        & set(source_base_state.visible_dynamic_object_history)
        & set(source_base_state.visible_dynamic_object_specs)
        & set(source_oracle_context.dynamic_object_history)
        & set(source_oracle_context.dynamic_object_future)
        & set(source_oracle_context.dynamic_object_specs)
    )
    supported_dynamic_object_ids = tuple(
        object_id
        for object_id in source_base_state.dynamic_object_ids
        if object_id in supported_context_ids
    )
    dropped_dynamic_object_ids = tuple(
        object_id
        for object_id in source_base_state.dynamic_object_ids
        if object_id not in supported_context_ids
    )
    visible_histories_local = {
        object_id: np.asarray(
            _to_decision_frame(
                _sample_context_history(
                    source_oracle_context,
                    object_id=object_id,
                    sample_times_s=decision_time_s + history_offsets,
                    dt_s=dt_s,
                ),
                decision_pose,
            ),
            dtype=ARRAY_DTYPE,
        )
        for object_id in supported_dynamic_object_ids
    }
    visible_specs_local = {
        object_id: dict(source_base_state.visible_dynamic_object_specs[object_id])
        for object_id in supported_dynamic_object_ids
    }
    if target_currently_visible:
        assert target_dynamic_object_id is not None
        assert target_dynamic_object_spec is not None
        if target_dynamic_object_id in visible_histories_local:
            raise ValueError("observed target object ID collides with source context")
        visible_histories_local[target_dynamic_object_id] = np.asarray(
            target_history_local,
            dtype=ARRAY_DTYPE,
        )
        visible_specs_local[target_dynamic_object_id] = dict(
            target_dynamic_object_spec
        )
        supported_dynamic_object_ids = tuple(
            sorted((*supported_dynamic_object_ids, target_dynamic_object_id))
        )
    static_occupancy_local = _rebase_static_occupancy(
        task_template.static_occupancy,
        decision_pose=decision_pose,
        base_config=base_config,
    )
    decision_base_state = BaseState(
        state_id=decision_state_id,
        split=source_base_state.split,
        recording_id=source_base_state.recording_id,
        dynamic_object_ids=supported_dynamic_object_ids,
        timestamp=source_base_state.timestamp + decision_time_s,
        robot_history=np.asarray(
            _to_decision_frame(robot_history_world, decision_pose),
            dtype=ARRAY_DTYPE,
        ),
        robot_state=current_control,
        visible_dynamic_object_history=visible_histories_local,
        visible_dynamic_object_specs=visible_specs_local,
        static_map_local=np.asarray(static_occupancy_local, dtype=ARRAY_DTYPE),
        metadata={
            "source_base_state_id": source_base_state.state_id,
            "decision_time_s": decision_time_s,
            "template_id": task_template.template_id,
            "dropped_dynamic_object_ids": dropped_dynamic_object_ids,
            "dropped_dynamic_object_reason": (
                "missing_complete_source_or_oracle_context"
            ),
            "target_currently_visible": target_currently_visible,
            "target_dynamic_object_id": target_dynamic_object_id,
        },
    )
    return Sop05rTebDecisionState(
        decision_state_id=decision_state_id,
        source_base_state_id=source_base_state.state_id,
        decision_time_s=decision_time_s,
        decision_pose_world=np.asarray(decision_pose, dtype=ARRAY_DTYPE),
        decision_control=current_control,
        robot_history_world=np.asarray(robot_history_world, dtype=ARRAY_DTYPE),
        robot_history_local=np.asarray(
            _to_decision_frame(robot_history_world, decision_pose),
            dtype=ARRAY_DTYPE,
        ),
        target_history_world=np.asarray(target_history_world, dtype=ARRAY_DTYPE),
        target_history_local=np.asarray(target_history_local, dtype=ARRAY_DTYPE),
        target_future_world=np.asarray(target_future_world, dtype=ARRAY_DTYPE),
        target_future_local=np.asarray(target_future_local, dtype=ARRAY_DTYPE),
        target_visibility_history=target_visibility_history,
        static_occupancy_local=np.asarray(static_occupancy_local, dtype=ARRAY_DTYPE),
        suffix_poses_world=np.asarray(suffix_poses_world, dtype=ARRAY_DTYPE),
        suffix_controls=np.asarray(suffix_controls, dtype=ARRAY_DTYPE),
        shared_goal_world_pose=task_template.local_goal_world_pose,
        nominal_trajectory=nominal,
        recomputed_query_maps=recomputed,
        base_state=decision_base_state,
        provenance={
            "source_base_state_id": source_base_state.state_id,
            "template_id": task_template.template_id,
            "occlusion_witness_time_s": float(witness.time_s),
            "occlusion_witness_sample_index": witness.sample_index,
            "route_prefix_end_time_s": decision_time_s,
            "visible_history_frames": history_assessment.visible_frames,
                "occluded_history_frames": history_assessment.occluded_frames,
            "decision_visible": target_currently_visible,
            "dropped_dynamic_object_ids": dropped_dynamic_object_ids,
            "dropped_dynamic_object_reason": (
                "missing_complete_source_or_oracle_context"
            ),
            "config_digest": teb_config.digest,
            "seed": int(seed),
        },
    )
