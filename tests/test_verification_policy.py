from dataclasses import replace
from pathlib import Path

import numpy as np

from src.contracts import (
    GridSpec,
    LONG40_FUTURE_HORIZON_S,
    LONG40_FUTURE_STEPS,
    LocalTrajectory,
    OracleWorld,
    POSE_TIME_LAYOUT_VERSION,
    SCHEMA_VERSION,
)
from src.generation.counterfactual_verify import (
    simulate_counterfactual_observation_trace,
)
from src.generation.risk_gt import compute_hidden_risk_gt
from src.geometry import (
    CircleFootprint,
    rasterize_footprint,
    transform_poses_local_to_global,
)
from src.planning.verification_actions import (
    VerificationAction,
    load_verification_actions,
    sample_state_aware_action_trace,
)
from src.planning.verification_responses import (
    build_completed_action_branch,
    build_observe_and_replan_branch,
    build_reactive_braking_branch,
    build_reactive_braking_trajectory,
    compose_time_aligned_policy_trajectory,
)


ROOT = Path(__file__).resolve().parents[1]


def _trajectory(*, speed_mps: float, grid: GridSpec) -> LocalTrajectory:
    dt_s = 0.2
    controls = np.tile(
        np.asarray([speed_mps, 0.0], dtype=np.float32),
        (grid.future_steps, 1),
    )
    times = np.arange(1, grid.future_steps + 1, dtype=np.float32) * np.float32(
        dt_s
    )
    poses = np.column_stack(
        (
            times * np.float32(speed_mps),
            np.zeros(grid.future_steps, dtype=np.float32),
            np.zeros(grid.future_steps, dtype=np.float32),
        )
    ).astype(np.float32)
    zeros = np.zeros((grid.height, grid.width), dtype=np.float32)
    return LocalTrajectory(
        trajectory_id="straight",
        poses=poses,
        controls=controls,
        swept_mask=zeros.copy(),
        tta_map=zeros.copy(),
        braking_map=zeros.copy(),
        centerline_map=zeros.copy(),
        task_cost=0.0,
        metadata={
            "pose_time_layout_version": POSE_TIME_LAYOUT_VERSION,
            "dt_s": dt_s,
        },
    )


def test_state_aware_arc_trace_brakes_first_and_ends_at_rest():
    action = load_verification_actions(
        ROOT / "configs/verification_actions.yaml"
    ).by_id["arc_left_30"]

    trace = sample_state_aware_action_trace(
        np.zeros(3, dtype=np.float32),
        action,
        robot_state=np.asarray([0.4, 0.0], dtype=np.float32),
        braking_deceleration_mps2=1.0,
        angular_deceleration_radps2=1.6,
    )

    assert trace.times_s[-1] > action.duration_s
    assert trace.poses[-1, 0] > 0.0
    np.testing.assert_allclose(trace.poses[-1, 2], action.delta_yaw_rad, atol=1e-6)
    assert trace.poses[-1, 1] > 0.0
    assert trace.linear_velocities_mps[0] == np.float32(0.4)
    assert trace.linear_velocities_mps[-1] == np.float32(0.0)
    assert trace.angular_velocities_radps[-1] == np.float32(0.0)


def test_trace_observation_keeps_hazard_seen_mid_arc_when_endpoint_misses_it():
    grid = GridSpec(
        height=80,
        width=80,
        history_steps=8,
        future_steps=32,
        resolution_m=0.1,
    )
    action = load_verification_actions(
        ROOT / "configs/verification_actions.yaml"
    ).by_id["arc_left_30"]
    trace = sample_state_aware_action_trace(
        np.zeros(3, dtype=np.float32),
        action,
        robot_state=np.zeros(2, dtype=np.float32),
        braking_deceleration_mps2=1.0,
        angular_deceleration_radps2=1.6,
    )
    angle = np.deg2rad(10.0)
    current_pose = np.asarray(
        [2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0], dtype=np.float32
    )
    future = np.tile(current_pose, (grid.future_steps, 1)).astype(np.float32)
    static = np.zeros((grid.height, grid.width), dtype=np.float32)

    observed = simulate_counterfactual_observation_trace(
        action_trace=trace,
        static_occupancy=static,
        dynamic_current_poses={"person": current_pose},
        dynamic_future_poses={"person": future},
        dynamic_specs={
            "person": {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.30},
            }
        },
        current_visible_mask=np.zeros_like(static, dtype=bool),
        current_age_map=np.ones_like(static, dtype=np.float32),
        grid=grid,
        future_dt_s=0.2,
        age_max_s=5.0,
        fov_rad=np.deg2rad(4.0),
        max_range_m=4.0,
    )

    assert observed.aggregate.visible_dynamic_occupancy.any()
    assert not observed.frames[-1].visible_dynamic_occupancy.any()


def test_reactive_branch_observation_excludes_unexecuted_action_tail():
    grid = GridSpec(
        height=80,
        width=80,
        history_steps=8,
        future_steps=LONG40_FUTURE_STEPS,
        resolution_m=0.1,
    )
    action = VerificationAction(
        action_id="arc_left_probe",
        duration_s=1.0,
        delta_forward_m=0.0,
        delta_yaw_rad=float(np.deg2rad(30.0)),
    )
    trace = sample_state_aware_action_trace(
        np.zeros(3, dtype=np.float32),
        action,
        robot_state=np.zeros(2, dtype=np.float32),
        braking_deceleration_mps2=1.0,
    )

    def pose_at_angle(angle_deg: float) -> np.ndarray:
        angle = np.deg2rad(angle_deg)
        return np.asarray(
            [2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0],
            dtype=np.float32,
        )

    early_pose = pose_at_angle(10.0)
    late_pose = pose_at_angle(25.0)
    future = {
        object_id: np.tile(pose, (grid.future_steps, 1)).astype(np.float32)
        for object_id, pose in {
            "early": early_pose,
            "late": late_pose,
        }.items()
    }
    common = {
        "static_occupancy": np.zeros((grid.height, grid.width), dtype=np.float32),
        "dynamic_current_poses": {
            "early": early_pose,
            "late": late_pose,
        },
        "dynamic_future_poses": future,
        "dynamic_specs": {
            object_id: {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.10},
            }
            for object_id in future
        },
        "current_visible_mask": np.zeros((grid.height, grid.width), dtype=bool),
        "current_age_map": np.ones((grid.height, grid.width), dtype=np.float32),
        "grid": grid,
        "future_dt_s": 0.2,
        "age_max_s": 5.0,
        "fov_rad": np.deg2rad(4.0),
        "max_range_m": 4.0,
    }
    full = simulate_counterfactual_observation_trace(
        action_trace=trace,
        **common,
    )
    first_visible_time_s = next(
        float(time_s)
        for frame, time_s in zip(full.frames, full.times_s, strict=True)
        if np.any(frame.visible_dynamic_occupancy & rasterize_footprint(
            CircleFootprint(0.10), early_pose, grid
        ))
    )
    branch = build_reactive_braking_branch(
        action_trace=trace,
        response_time_s=first_visible_time_s,
        braking_deceleration_mps2=1.0,
        future_horizon_s=LONG40_FUTURE_HORIZON_S,
    )
    executed = simulate_counterfactual_observation_trace(
        action_trace=branch.executed_trace,
        **common,
    )
    late_mask = rasterize_footprint(CircleFootprint(0.10), late_pose, grid)

    assert branch.end_time_s < trace.times_s[-1]
    assert np.any(full.aggregate.visible_dynamic_occupancy & late_mask)
    assert not np.any(executed.aggregate.visible_dynamic_occupancy & late_mask)


def test_early_reactive_braking_reuses_risk_gt_and_avoids_collision():
    grid = GridSpec(
        height=80,
        width=80,
        history_steps=8,
        future_steps=LONG40_FUTURE_STEPS,
        resolution_m=0.1,
    )
    nominal = _trajectory(speed_mps=0.8, grid=grid)
    person_pose = np.asarray([1.55, 0.0, 0.0], dtype=np.float32)
    world = OracleWorld(
        world_id="braking-world",
        base_state_id="braking-base",
        static_occupancy=np.zeros((grid.height, grid.width), dtype=np.float32),
        dynamic_object_trajectories={
            "person": np.tile(person_pose, (grid.future_steps, 1)).astype(np.float32)
        },
        dynamic_object_specs={
            "person": {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.30},
            }
        },
        occluders=(),
        blind_spot_config={"kind": "structural", "occluder_ids": []},
        random_seed=7,
        metadata={"schema_version": SCHEMA_VERSION},
    )
    braking = build_reactive_braking_trajectory(
        nominal_trajectory=nominal,
        start_pose=np.zeros(3, dtype=np.float32),
        response_time_s=0.4,
        braking_deceleration_mps2=1.0,
        future_dt_s=0.2,
    )
    common = {
        "world": world,
        "hidden_object_ids": ("person",),
        "robot_footprint": CircleFootprint(0.30),
        "grid": grid,
        "future_dt_s": 0.2,
        "sigma_distance_m": 0.5,
        "sigma_time_s": 2.0,
        "near_miss_distance_m": 0.35,
    }

    nominal_risk = compute_hidden_risk_gt(nominal, **common)
    braking_risk = compute_hidden_risk_gt(braking, **common)

    assert nominal_risk.collision_label == 1
    assert nominal_risk.risk_severity == 1.0
    assert braking_risk.collision_label == 0
    assert braking_risk.risk_severity < nominal_risk.risk_severity
    assert braking.poses.shape == nominal.poses.shape
    assert braking.controls.shape == nominal.controls.shape
    assert braking.poses.dtype == np.float32
    assert braking.controls.dtype == np.float32
    assert np.isfinite(braking.poses).all()
    assert np.isfinite(braking.controls).all()
    np.testing.assert_allclose(braking.poses[0, 0], 0.16, atol=1e-6)
    assert braking.metadata["source_nominal_trajectory_id"] == nominal.trajectory_id
    assert braking.metadata["response_id"] == "emergency_brake"
    assert braking.metadata["response_endpoint_time_s"] == 0.4
    assert braking.metadata["label_side_policy_trajectory"] is True


def test_reactive_branch_brakes_from_actual_arc_trace_state():
    action = VerificationAction(
        action_id="arc_left_probe",
        duration_s=1.0,
        delta_forward_m=0.4,
        delta_yaw_rad=float(np.deg2rad(30.0)),
    )
    trace = sample_state_aware_action_trace(
        np.zeros(3, dtype=np.float32),
        action,
        robot_state=np.zeros(2, dtype=np.float32),
        braking_deceleration_mps2=1.0,
    )
    trigger_time_s = 0.5
    trigger_index = int(
        np.flatnonzero(np.isclose(trace.times_s, trigger_time_s))[0]
    )

    branch = build_reactive_braking_branch(
        action_trace=trace,
        response_time_s=trigger_time_s,
        braking_deceleration_mps2=1.0,
        angular_deceleration_radps2=1.6,
        future_horizon_s=LONG40_FUTURE_HORIZON_S,
    )

    branch_trigger_index = int(
        np.flatnonzero(
            np.isclose(branch.executed_trace.times_s, trigger_time_s)
        )[0]
    )
    np.testing.assert_array_equal(
        branch.executed_trace.poses[branch_trigger_index],
        trace.poses[trigger_index],
    )
    assert branch.branch_kind == "emergency_brake"
    assert branch.trigger_time_s == trigger_time_s
    assert branch.end_time_s > trigger_time_s
    assert branch.end_pose[1] > trace.poses[trigger_index, 1]
    assert branch.end_pose[2] > trace.poses[trigger_index, 2]
    assert branch.executed_trace.linear_velocities_mps[-1] == 0.0
    assert branch.executed_trace.angular_velocities_radps[-1] == 0.0


def test_time_aligned_policy_keeps_action_prefix_and_delays_replan_suffix():
    grid = GridSpec(
        height=80,
        width=80,
        history_steps=8,
        future_steps=LONG40_FUTURE_STEPS,
        resolution_m=0.1,
    )
    action = VerificationAction(
        action_id="short_forward_probe",
        duration_s=0.4,
        delta_forward_m=0.2,
        delta_yaw_rad=0.0,
    )
    action_trace = sample_state_aware_action_trace(
        np.zeros(3, dtype=np.float32),
        action,
        robot_state=np.zeros(2, dtype=np.float32),
        braking_deceleration_mps2=1.0,
    )
    branch = build_completed_action_branch(action_trace)
    suffix = _trajectory(speed_mps=0.5, grid=grid)
    suffix_parent = transform_poses_local_to_global(
        suffix.poses, branch.end_pose
    ).astype(np.float32)

    combined = compose_time_aligned_policy_trajectory(
        template_trajectory=suffix,
        branch=branch,
        future_dt_s=0.2,
        trajectory_id="policy::complete::suffix",
        source_action_id=action.action_id,
        source_nominal_trajectory_id="nominal",
        suffix_trajectory=suffix,
        suffix_poses_in_parent_frame=suffix_parent,
    )

    np.testing.assert_allclose(combined.poses[0], [0.1, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(combined.poses[1], branch.end_pose, atol=1e-6)
    np.testing.assert_allclose(combined.poses[2], [0.3, 0.0, 0.0], atol=1e-6)
    assert combined.metadata["absolute_time_aligned"] is True
    assert combined.metadata["branch_end_time_s"] == 0.4


def test_hazard_and_clear_branches_have_distinct_replanning_start_states():
    action = VerificationAction(
        action_id="arc_left_probe",
        duration_s=1.0,
        delta_forward_m=0.4,
        delta_yaw_rad=float(np.deg2rad(30.0)),
    )
    trace = sample_state_aware_action_trace(
        np.zeros(3, dtype=np.float32),
        action,
        robot_state=np.zeros(2, dtype=np.float32),
        braking_deceleration_mps2=1.0,
    )
    clear_branch = build_completed_action_branch(trace)
    hazard_branch = build_reactive_braking_branch(
        action_trace=trace,
        response_time_s=0.5,
        braking_deceleration_mps2=1.0,
        angular_deceleration_radps2=1.6,
        future_horizon_s=LONG40_FUTURE_HORIZON_S,
    )

    assert hazard_branch.end_time_s < clear_branch.end_time_s
    assert not np.allclose(hazard_branch.end_pose, clear_branch.end_pose)


def test_observe_and_replan_interrupts_without_forcing_zero_control():
    action = VerificationAction(
        action_id="forward_probe",
        duration_s=1.0,
        delta_forward_m=0.5,
        delta_yaw_rad=0.0,
    )
    trace = sample_state_aware_action_trace(
        np.zeros(3, dtype=np.float32),
        action,
        robot_state=np.zeros(2, dtype=np.float32),
        braking_deceleration_mps2=1.0,
    )

    branch = build_observe_and_replan_branch(
        action_trace=trace,
        response_time_s=0.4,
    )

    assert branch.branch_kind == "observe_and_replan"
    assert branch.trigger_time_s == 0.4
    assert branch.end_time_s == 0.4
    assert branch.planned_action_end_time_s == 1.0
    np.testing.assert_allclose(branch.end_control, [0.5, 0.0], atol=1e-7)
