from collections import Counter

import numpy as np

from src.contracts import build_grid_spec
from src.evaluation.sop05r_teb_audit import audit_sop05r_teb_collection
from src.generation.anchored_human_placement import (
    _sample_robot_poses,
    apply_anchored_rigid_transform,
    resample_long_motion_snippet,
    solve_anchored_human_placement,
    synchronized_centerline_blocking,
)
from src.generation.history_visibility import classify_sop05r_seen_then_occluded_history
from src.generation.sop05r_teb_event_sampler import build_sop05r_teb_mother
from src.generation.sop05r_teb_output_loader import load_sop05r_teb_output
from src.generation.sop05r_teb_run import publish_sop05r_teb_run
from src.generation.sop05r_teb_templates import (
    canonical_sop05r_teb_base_state_digest,
    iter_sop05r_teb_task_templates,
)
from src.geometry import RectangleFootprint, inflate_footprint
from src.planning.query_maps import build_trajectory_query_maps
from tests.test_anchored_human_placement import _m4_inputs, _snippet


_FAMILY_QUOTA = {"rectangle": 4, "l_shape": 4, "circle": 2}


def _ten_fixture_mothers():
    base_config, base_state, oracle_context, teb_config = _m4_inputs()
    source_digest = canonical_sop05r_teb_base_state_digest(base_state)
    selected = []
    counts: Counter[str] = Counter()
    snippet = _snippet()
    for template_evaluation in iter_sop05r_teb_task_templates(
        base_state=base_state,
        oracle_context=oracle_context,
        base_config=base_config,
        teb_config=teb_config,
        seed=19,
    ):
        task = template_evaluation.template
        if task is None or counts[task.family] >= _FAMILY_QUOTA[task.family]:
            continue
        placement_evaluation = solve_anchored_human_placement(
            task_template=task,
            snippet=snippet,
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
            teb_config=teb_config,
            seed=27,
        )
        if placement_evaluation.result is None:
            continue
        mother_evaluation = build_sop05r_teb_mother(
            base_config=base_config,
            source_base_state=base_state,
            source_oracle_context=oracle_context,
            teb_config=teb_config,
            task_template=task,
            placement_result=placement_evaluation.result,
            snippet=snippet,
            seed=43,
        )
        if mother_evaluation.mother is None:
            continue
        selected.append((mother_evaluation.mother, task, placement_evaluation.result))
        counts[task.family] += 1
        if dict(counts) == _FAMILY_QUOTA:
            break
    assert dict(counts) == _FAMILY_QUOTA
    assert canonical_sop05r_teb_base_state_digest(base_state) == source_digest
    return base_config, base_state, teb_config, snippet, tuple(selected)


def test_ten_template_gate(tmp_path) -> None:
    base_config, base_state, teb_config, snippet, fixtures = _ten_fixture_mothers()
    mothers = tuple(item[0] for item in fixtures)
    assert len(mothers) == 10
    assert Counter(task.family for _, task, _ in fixtures) == _FAMILY_QUOTA

    rotation_solver_needed = False
    for mother, task, placement_result in fixtures:
        record = mother.trajectory_record
        event = mother.event
        assert task.direct_corridor_intrusion_m >= 0.15
        assert record.full_route.band_poses_world.shape == (21, 3)
        assert record.full_route.sampled_poses_world.shape == (40, 3)
        assert record.nominal_trajectory.poses.shape == (32, 3)
        assert event.target.history_poses.shape == (8, 3)
        assert event.target.future_poses.shape == (32, 3)
        assert placement_result.visibility.eligible
        assert 1.2 <= event.conflict_time_s < 6.4
        assert event.conflict_time_s < record.full_route.goal_arrival_time_s
        assert np.linalg.norm(
            record.full_route.sampled_poses_world[-1, :2]
            - record.shared_goal_world_pose[:2]
        ) <= teb_config.planner.goal_position_tolerance_m
        recomputed_maps = build_trajectory_query_maps(
            record.nominal_trajectory.poses,
            record.nominal_trajectory.controls,
            grid=build_grid_spec(base_config),
            footprint=inflate_footprint(
                RectangleFootprint(
                    float(base_config["robot"]["length_m"]),
                    float(base_config["robot"]["width_m"]),
                ),
                float(base_config["robot"]["inflation_m"]),
            ),
            dt_s=teb_config.trajectory.future_dt_s,
            braking_deceleration_mps2=(
                teb_config.planner.max_linear_acceleration_mps2
            ),
        )
        np.testing.assert_array_equal(
            recomputed_maps.swept_mask,
            record.nominal_trajectory.swept_mask,
        )
        np.testing.assert_allclose(
            recomputed_maps.tta_map,
            record.nominal_trajectory.tta_map,
            rtol=0.0,
            atol=1e-6,
        )
        if abs(placement_result.placement.rotation_rad) > 1e-6:
            source_poses, source_velocities = resample_long_motion_snippet(
                snippet,
                time_scale=placement_result.placement.temporal_scale,
            )
            unrotated, _, _ = apply_anchored_rigid_transform(
                source_poses=source_poses,
                source_velocities=source_velocities,
                anchor=placement_result.placement.anchor,
                rotation_rad=0.0,
            )
            times = (np.arange(40, dtype=np.float64) - 7) * 0.2
            robot = _sample_robot_poses(
                task,
                base_state,
                times,
                dt_s=teb_config.trajectory.future_dt_s,
            )
            blocked, _ = synchronized_centerline_blocking(
                robot[:, :2],
                unrotated[:, :2],
                task.occluders,
                epsilon_m=teb_config.occlusion.centerline_intersection_epsilon_m,
            )
            unrotated_history = classify_sop05r_seen_then_occluded_history(
                blocked,
                decision_index=7,
                minimum_visible_frames=(
                    teb_config.occlusion.minimum_visible_history_frames
                ),
                minimum_occluded_frames=(
                    teb_config.occlusion.minimum_occluded_history_frames
                ),
            )
            rotation_solver_needed |= not unrotated_history.eligible

    assert rotation_solver_needed
    output = tmp_path / "ten-template-gate"
    publish_sop05r_teb_run(
        mothers,
        output,
        base_config=base_config,
        requested_count=10,
        config_digest=teb_config.digest,
        verification_action_digest="b" * 64,
        source_evidence={"producer": "ten-template-gate"},
        denominator_counts={"m6_accepted": 10},
        rejection_counts={},
    )
    collection = load_sop05r_teb_output(output, require_complete=True)
    metrics = audit_sop05r_teb_collection(collection)

    assert collection.complete
    assert len(collection.events) == 10
    assert metrics.event_count == metrics.recomputed_witness_count == 10
