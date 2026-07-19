from pathlib import Path

import numpy as np
import pytest

from src.contracts import LocalTrajectory, build_grid_spec
from src.geometry import RectangleFootprint, rasterize_footprint
from src.planning.query_maps import build_local_trajectory
from src.planning.replanning import (
    REPLANNING_VERSION,
    generate_replanned_candidates,
)
from src.planning.trajectory_sampler import sample_candidate_rollouts
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def base_config():
    return load_config(ROOT / "configs/base.yaml")


@pytest.fixture(scope="module")
def nominal(base_config) -> LocalTrajectory:
    candidate = next(
        candidate
        for candidate in sample_candidate_rollouts(base_config)
        if candidate.trajectory_id == "forward_v00_w02"
    )
    return build_local_trajectory(
        candidate,
        base_config,
        braking_deceleration_mps2=1.0,
        task_cost=0.0,
    )


def test_replanned_candidates_start_at_post_pose_and_use_fresh_sampler(
    base_config, nominal
):
    grid = build_grid_spec(base_config)
    static = np.zeros((grid.height, grid.width), dtype=np.float32)
    post_pose = np.asarray([0.25, 0.05, np.deg2rad(20.0)], dtype=np.float32)

    result = generate_replanned_candidates(
        post_action_pose=post_pose,
        nominal_trajectory=nominal,
        action_id="yaw_left_20",
        config=base_config,
        static_occupancy=static,
        braking_deceleration_mps2=1.0,
        max_candidates=6,
    )

    assert result.version == REPLANNING_VERSION
    assert result.reject_available
    np.testing.assert_array_equal(result.post_action_pose, post_pose)
    np.testing.assert_array_equal(result.task_anchor_pose, nominal.poses[-1])
    assert len(result.candidates) == 6
    assert any(candidate.trajectory.metadata["is_stop"] for candidate in result.candidates)
    assert [candidate.intent_error for candidate in result.candidates[:-1]] == sorted(
        candidate.intent_error for candidate in result.candidates[:-1]
    )

    sampled_ids = {
        candidate.trajectory_id for candidate in sample_candidate_rollouts(base_config)
    }
    for candidate in result.candidates:
        np.testing.assert_array_equal(
            candidate.pose_sequence_in_parent_frame[0], post_pose
        )
        np.testing.assert_array_equal(candidate.implicit_start_pose, post_pose)
        assert candidate.trajectory.metadata["replanning_version"] == REPLANNING_VERSION
        assert candidate.trajectory.metadata["sampling_origin"] == "post_action_pose"
        assert candidate.trajectory.metadata["nominal_suffix_used"] is False
        assert candidate.trajectory.metadata["source_primitive_id"] in sampled_ids
        assert not np.shares_memory(candidate.trajectory.poses, nominal.poses)
        assert candidate.trajectory.poses.shape == nominal.poses.shape
        assert candidate.poses_in_parent_frame.dtype == np.float32
        assert candidate.swept_mask_in_parent_frame.dtype == np.float32
        assert np.isfinite(candidate.poses_in_parent_frame).all()
        assert np.isfinite(candidate.swept_mask_in_parent_frame).all()
        assert candidate.swept_mask_in_parent_frame.any()
        for array in (
            candidate.trajectory.poses,
            candidate.trajectory.controls,
            candidate.trajectory.swept_mask,
            candidate.trajectory.tta_map,
            candidate.trajectory.braking_map,
            candidate.trajectory.centerline_map,
        ):
            assert array.dtype == np.float32
            assert np.isfinite(array).all()


def test_replanning_filters_parent_frame_static_collisions_and_retains_stop(
    base_config, nominal
):
    grid = build_grid_spec(base_config)
    static = rasterize_footprint(
        RectangleFootprint(0.40, 2.00),
        np.asarray([0.80, 0.0, 0.0], dtype=np.float32),
        grid,
    ).astype(np.float32)
    result = generate_replanned_candidates(
        post_action_pose=np.zeros(3, dtype=np.float32),
        nominal_trajectory=nominal,
        action_id="stop_scan",
        config=base_config,
        static_occupancy=static,
        braking_deceleration_mps2=1.0,
        max_candidates=8,
    )

    assert result.rejection_counts.get("static_collision", 0) > 0
    stop = [
        candidate
        for candidate in result.candidates
        if candidate.trajectory.metadata["is_stop"]
    ]
    assert len(stop) == 1
    assert result.reject_available


def test_replanning_is_deterministic(base_config, nominal):
    grid = build_grid_spec(base_config)
    static = np.zeros((grid.height, grid.width), dtype=np.float32)
    kwargs = dict(
        post_action_pose=np.asarray([0.15, -0.05, -0.1], dtype=np.float32),
        nominal_trajectory=nominal,
        action_id="forward_peek",
        config=base_config,
        static_occupancy=static,
        braking_deceleration_mps2=1.0,
        max_candidates=5,
    )
    first = generate_replanned_candidates(**kwargs)
    second = generate_replanned_candidates(**kwargs)

    assert [item.trajectory.trajectory_id for item in first.candidates] == [
        item.trajectory.trajectory_id for item in second.candidates
    ]
    for left, right in zip(first.candidates, second.candidates, strict=True):
        np.testing.assert_array_equal(left.poses_in_parent_frame, right.poses_in_parent_frame)
        np.testing.assert_array_equal(
            left.swept_mask_in_parent_frame, right.swept_mask_in_parent_frame
        )
