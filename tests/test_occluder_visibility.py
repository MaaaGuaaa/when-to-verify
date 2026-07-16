"""Behavioral tests for physical environment-occluder sampling."""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts import build_grid_spec
import src.generation.occluder_sampler as occluder_sampler
from src.generation.occluder_sampler import (
    OccluderSamplingError,
    sample_environment_occluder,
)
from src.generation.structural_blindspot import (
    footprint_visibility_sequence,
    has_continuous_emergence,
)
from src.geometry import (
    CircleFootprint,
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint_sweep,
    raycast_visibility,
    trajectory_signed_clearances,
)
from src.utils.config import load_config


def _occluder_config() -> dict:
    return {
        "types": ["pillar"],
        "normal_offset_range_m": [1.0, 1.0],
        "wall": {"length_range_m": [1.0, 1.0], "width_range_m": [0.2, 0.2]},
        "shelf": {"length_range_m": [1.0, 1.0], "width_range_m": [0.4, 0.4]},
        "pillar": {"length_range_m": [0.5, 0.5], "width_range_m": [0.5, 0.5]},
    }


def _straight_robot_poses() -> np.ndarray:
    x = np.linspace(0.2, 3.0, 15, dtype=np.float32)
    return np.column_stack(
        (x, np.zeros_like(x), np.zeros_like(x))
    ).astype(np.float32)


def _curved_target_poses() -> np.ndarray:
    times = np.arange(16, dtype=np.float64) * 0.2
    source_positions = np.column_stack((times, 0.08 * times**2))
    source_headings = np.arctan2(0.16 * times, np.ones_like(times))
    conflict_time = 2.0
    conflict_point = np.asarray([0.72, 0.0], dtype=np.float64)
    crossing_direction = np.asarray([-0.415, 0.910], dtype=np.float64)
    crossing_direction /= np.linalg.norm(crossing_direction)
    source_angle = float(np.arctan2(0.32, 1.0))
    target_angle = float(
        np.arctan2(crossing_direction[1], crossing_direction[0])
    )
    rotation_angle = target_angle - source_angle
    rotation = np.asarray(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )
    anchor = np.asarray([conflict_time, 0.08 * conflict_time**2])
    target_positions = (
        ((source_positions - anchor) / 0.865) @ rotation.T + conflict_point
    )
    return np.column_stack(
        (target_positions, source_headings + rotation_angle)
    ).astype(np.float32)


@pytest.mark.parametrize(
    "target_footprint",
    [CircleFootprint(0.20), RectangleFootprint(0.40, 0.20)],
)
@pytest.mark.parametrize("occluder_type", ["wall", "shelf", "pillar"])
def test_sampled_occluder_hides_target_without_blocking_robot_or_target(
    target_footprint,
    occluder_type: str,
) -> None:
    config = load_config()
    grid = build_grid_spec(config)
    static_occupancy = np.zeros((grid.height, grid.width), dtype=np.float32)
    robot_poses = _straight_robot_poses()
    robot_footprint = inflate_footprint(
        RectangleFootprint(
            config["robot"]["length_m"], config["robot"]["width_m"]
        ),
        config["robot"]["inflation_m"],
    )
    target_current_pose = np.asarray([1.6, -2.0, np.pi / 2.0], dtype=np.float32)
    target_future_poses = np.column_stack(
        (
            np.full(15, 1.6, dtype=np.float32),
            np.linspace(-1.8, 1.0, 15, dtype=np.float32),
            np.linspace(0.2, 1.1, 15, dtype=np.float32),
        )
    ).astype(np.float32)

    occluder_config = _occluder_config()
    occluder_config["types"] = [occluder_type]
    placement = sample_environment_occluder(
        static_occupancy=static_occupancy,
        grid=grid,
        sensor_pose=np.zeros(3, dtype=np.float32),
        conflict_point=np.asarray([1.6, 0.0], dtype=np.float32),
        trajectory_normal=np.asarray([0.0, 1.0], dtype=np.float32),
        robot_poses=robot_poses,
        robot_footprint=robot_footprint,
        target_current_pose=target_current_pose,
        target_future_poses=target_future_poses,
        target_footprint=target_footprint,
        context_trajectories={},
        context_footprints={},
        config=occluder_config,
        rng=np.random.default_rng(7),
        max_attempts=8,
    )

    assert placement.occluder["type"] == occluder_type
    assert placement.mask.dtype == np.bool_
    assert not np.any(
        placement.mask & rasterize_footprint_sweep(robot_footprint, robot_poses, grid)
    )
    target_poses = np.vstack((target_current_pose, target_future_poses))
    target_clearances = trajectory_signed_clearances(
        placement.footprint,
        np.tile(placement.pose, (target_poses.shape[0], 1)),
        target_footprint,
        target_poses,
    )
    assert np.all(target_clearances > 0.0)
    visibility = raycast_visibility(
        static_occupancy.astype(bool) | placement.mask,
        grid,
        sensor_pose=(0.0, 0.0, 0.0),
        max_range_m=8.0,
    )
    sequence = footprint_visibility_sequence(
        target_footprint, target_poses, visibility, grid
    )
    assert not bool(sequence[0])
    assert has_continuous_emergence(sequence, min_visible_frames=2)


def test_occluder_sampling_fails_with_explicit_reason_when_swept_volume_is_blocked() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    static_occupancy = np.ones((grid.height, grid.width), dtype=np.float32)
    robot_footprint = inflate_footprint(
        RectangleFootprint(
            config["robot"]["length_m"], config["robot"]["width_m"]
        ),
        config["robot"]["inflation_m"],
    )

    with pytest.raises(OccluderSamplingError) as exc_info:
        sample_environment_occluder(
            static_occupancy=static_occupancy,
            grid=grid,
            sensor_pose=(0.0, 0.0, 0.0),
            conflict_point=(1.6, 0.0),
            trajectory_normal=(0.0, 1.0),
            robot_poses=_straight_robot_poses(),
            robot_footprint=robot_footprint,
            target_current_pose=(1.6, -2.0, np.pi / 2.0),
            target_future_poses=np.tile(
                np.asarray([1.6, -1.0, np.pi / 2.0], dtype=np.float32),
                (15, 1),
            ),
            target_footprint=CircleFootprint(0.2),
            context_trajectories={},
            context_footprints={},
            config=_occluder_config(),
            rng=np.random.default_rng(7),
            max_attempts=3,
        )

    assert exc_info.value.reason == "occluder_no_valid_placement"
    assert exc_info.value.attempts == 3
    assert exc_info.value.rejection_reasons == {"occluder_static_overlap": 3}


def test_occluder_rejects_overlap_with_current_robot_before_future_rollout() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot_footprint = inflate_footprint(
        RectangleFootprint(
            config["robot"]["length_m"], config["robot"]["width_m"]
        ),
        config["robot"]["inflation_m"],
    )
    future_robot_poses = np.column_stack(
        (
            np.linspace(1.5, 3.0, 15, dtype=np.float32),
            np.zeros(15, dtype=np.float32),
            np.zeros(15, dtype=np.float32),
        )
    )
    occluder_config = _occluder_config()
    occluder_config["normal_offset_range_m"] = [0.35, 0.35]
    occluder_config["pillar"] = {
        "length_range_m": [0.2, 0.2],
        "width_range_m": [0.2, 0.2],
    }

    with pytest.raises(OccluderSamplingError) as exc_info:
        sample_environment_occluder(
            static_occupancy=np.zeros(
                (grid.height, grid.width), dtype=np.float32
            ),
            grid=grid,
            sensor_pose=(0.0, 0.0, 0.0),
            conflict_point=(1.6, 0.0),
            trajectory_normal=(0.0, 1.0),
            robot_poses=future_robot_poses,
            robot_footprint=robot_footprint,
            target_current_pose=(0.0, -0.8, np.pi / 2.0),
            target_future_poses=np.column_stack(
                (
                    np.full(15, 1.6, dtype=np.float32),
                    np.linspace(-0.6, 1.0, 15, dtype=np.float32),
                    np.full(15, np.pi / 2.0, dtype=np.float32),
                )
            ),
            target_footprint=CircleFootprint(0.1),
            context_trajectories={},
            context_footprints={},
            config=occluder_config,
            rng=np.random.default_rng(11),
            max_attempts=2,
        )

    assert exc_info.value.reason == "occluder_no_valid_placement"
    assert exc_info.value.rejection_reasons == {
        "occluder_robot_swept_overlap": 2
    }


def test_occluder_uses_exact_target_clearance_at_raster_cell_boundary() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot_footprint = inflate_footprint(
        RectangleFootprint(
            config["robot"]["length_m"], config["robot"]["width_m"]
        ),
        config["robot"]["inflation_m"],
    )
    times = np.arange(16, dtype=np.float64) * 0.2
    source_positions = np.column_stack((times, 0.08 * times**2))
    source_headings = np.arctan2(0.16 * times, np.ones_like(times))
    conflict_time = 2.0
    conflict_point = np.asarray([0.72, 0.0], dtype=np.float64)
    crossing_direction = np.asarray([-0.415, 0.910], dtype=np.float64)
    crossing_direction /= np.linalg.norm(crossing_direction)
    source_angle = float(np.arctan2(0.32, 1.0))
    target_angle = float(
        np.arctan2(crossing_direction[1], crossing_direction[0])
    )
    rotation_angle = target_angle - source_angle
    rotation = np.asarray(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )
    anchor = np.asarray([conflict_time, 0.08 * conflict_time**2])
    target_positions = (
        ((source_positions - anchor) / 0.865) @ rotation.T + conflict_point
    )
    target_poses = np.column_stack(
        (target_positions, source_headings + rotation_angle)
    ).astype(np.float32)
    target_footprint = CircleFootprint(0.30)
    occluder_config = _occluder_config()
    occluder_config["normal_offset_range_m"] = [0.8, 0.8]
    occluder_config["pillar"] = {
        "length_range_m": [0.4, 0.4],
        "width_range_m": [0.4, 0.4],
    }

    placement = sample_environment_occluder(
        static_occupancy=np.zeros(
            (grid.height, grid.width), dtype=np.float32
        ),
        grid=grid,
        sensor_pose=(0.0, 0.0, 0.0),
        conflict_point=conflict_point,
        trajectory_normal=(0.0, 1.0),
        robot_poses=_straight_robot_poses(),
        robot_footprint=robot_footprint,
        target_current_pose=target_poses[0],
        target_future_poses=target_poses[1:],
        target_footprint=target_footprint,
        context_trajectories={},
        context_footprints={},
        config=occluder_config,
        rng=np.random.default_rng(19),
        max_attempts=1,
    )

    repeated_occluder_pose = np.tile(placement.pose, (target_poses.shape[0], 1))
    clearances = trajectory_signed_clearances(
        placement.footprint,
        repeated_occluder_pose,
        target_footprint,
        target_poses,
    )
    assert np.min(clearances) > 0.0
    assert np.any(
        placement.mask
        & rasterize_footprint_sweep(target_footprint, target_poses, grid)
    )


def test_occluder_search_covers_feasible_band_with_broad_default_ranges() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot_footprint = inflate_footprint(
        RectangleFootprint(
            config["robot"]["length_m"], config["robot"]["width_m"]
        ),
        config["robot"]["inflation_m"],
    )
    target_poses = _curved_target_poses()
    broad_config = {
        "types": ["wall", "shelf", "pillar"],
        "normal_offset_range_m": [0.5, 1.5],
        "wall": {
            "length_range_m": [1.0, 3.0],
            "width_range_m": [0.2, 0.5],
        },
        "shelf": {
            "length_range_m": [1.0, 2.5],
            "width_range_m": [0.4, 0.8],
        },
        "pillar": {
            "length_range_m": [0.4, 0.8],
            "width_range_m": [0.4, 0.8],
        },
    }

    placement = sample_environment_occluder(
        static_occupancy=np.zeros(
            (grid.height, grid.width), dtype=np.float32
        ),
        grid=grid,
        sensor_pose=(0.0, 0.0, 0.0),
        conflict_point=(0.72, 0.0),
        trajectory_normal=(0.0, 1.0),
        robot_poses=_straight_robot_poses(),
        robot_footprint=robot_footprint,
        target_current_pose=target_poses[0],
        target_future_poses=target_poses[1:],
        target_footprint=CircleFootprint(0.30),
        context_trajectories={},
        context_footprints={},
        config=broad_config,
        rng=np.random.default_rng(5),
        max_attempts=8,
    )

    assert placement.attempt <= 8
    assert placement.occluder["type"] in {"wall", "shelf", "pillar"}
    assert sum(placement.rejection_reasons.values()) == placement.attempt - 1


def test_joint_schedule_covers_both_sides_types_and_full_quantiles() -> None:
    first = occluder_sampler.build_joint_occluder_schedule(
        types=("wall", "shelf", "pillar"),
        max_candidates=64,
        rng=np.random.default_rng(19),
    )
    repeated = occluder_sampler.build_joint_occluder_schedule(
        types=("wall", "shelf", "pillar"),
        max_candidates=64,
        rng=np.random.default_rng(19),
    )
    different_seed = occluder_sampler.build_joint_occluder_schedule(
        types=("wall", "shelf", "pillar"),
        max_candidates=64,
        rng=np.random.default_rng(23),
    )

    assert first == repeated
    assert first[:16] != different_seed[:16]
    assert len(first) == 64
    assert {item.side for item in first} == {-1, 1}
    assert {item.occluder_type for item in first} == {"wall", "shelf", "pillar"}
    assert {item.offset_quantile for item in first} == {
        0.1,
        0.3,
        0.4,
        0.5,
        0.7,
        0.9,
    }
    assert {item.dimension_quantile for item in first} == {
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    }
    assert {item.angle_multiplier for item in first} >= {
        -0.95,
        -0.5,
        0.0,
        0.5,
        0.95,
    }
    assert {item.time_scale_quantile for item in first} >= {
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    }
    physics_prefix = first[:16]
    assert all(item.occluder_type == "pillar" for item in physics_prefix)
    assert all(item.dimension_quantile == 0.0 for item in physics_prefix)
    assert {item.side for item in physics_prefix} == {-1, 1}
    assert {item.offset_quantile for item in physics_prefix} == {0.3, 0.4}
    assert {abs(item.angle_multiplier) for item in physics_prefix} == {
        0.0,
        0.25,
        0.5,
        0.7,
    }
    assert all(
        item.angle_multiplier == 0.0
        or np.sign(item.angle_multiplier) == -item.side
        for item in physics_prefix
    )
    assert {item.time_scale_quantile for item in physics_prefix} == {
        0.0,
        0.16,
        0.25,
        0.5,
    }
    assert all(item.conflict_time_quantile == 1.0 for item in physics_prefix)
    for first_side, second_side in zip(
        physics_prefix[::2], physics_prefix[1::2], strict=True
    ):
        assert first_side.side == -1
        assert second_side.side == 1
        assert first_side.occluder_type == second_side.occluder_type
        assert first_side.offset_quantile == second_side.offset_quantile
        assert first_side.dimension_quantile == second_side.dimension_quantile
        assert first_side.angle_multiplier == -second_side.angle_multiplier
        assert first_side.time_scale_quantile == second_side.time_scale_quantile
        assert (
            first_side.conflict_time_quantile
            == second_side.conflict_time_quantile
        )

    wall_only = occluder_sampler.build_joint_occluder_schedule(
        types=("wall",),
        max_candidates=10,
        rng=np.random.default_rng(19),
    )
    assert all(item.occluder_type == "wall" for item in wall_only)
    assert {item.offset_quantile for item in wall_only} == {
        0.1,
        0.3,
        0.5,
        0.7,
        0.9,
    }
    assert {item.dimension_quantile for item in wall_only} == {
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    }


def test_joint_occluder_uses_exact_robot_clearance_when_raster_masks_touch() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot_footprint = inflate_footprint(
        RectangleFootprint(
            config["robot"]["length_m"], config["robot"]["width_m"]
        ),
        config["robot"]["inflation_m"],
    )
    robot_poses = np.vstack(
        (np.zeros(3, dtype=np.float32), _straight_robot_poses())
    )
    occluder_footprint = RectangleFootprint(0.4, 0.4)
    center = np.asarray([1.6, -0.75], dtype=np.float64)
    occluder_pose = np.asarray(
        [
            center[0],
            center[1],
            np.arctan2(center[1], center[0]) + 0.5 * np.pi,
        ],
        dtype=np.float64,
    )
    exact_clearances = trajectory_signed_clearances(
        occluder_footprint,
        np.tile(occluder_pose, (robot_poses.shape[0], 1)),
        robot_footprint,
        robot_poses,
    )
    occluder_mask = rasterize_footprint_sweep(
        occluder_footprint, occluder_pose[None, :], grid
    )
    robot_mask = rasterize_footprint_sweep(robot_footprint, robot_poses, grid)
    assert float(np.min(exact_clearances)) > 0.0
    assert np.any(occluder_mask & robot_mask)

    candidate = occluder_sampler.propose_environment_occluder_geometry(
        static_occupancy=np.zeros((grid.height, grid.width), dtype=np.float32),
        grid=grid,
        sensor_pose=(0.0, 0.0, 0.0),
        conflict_point=(1.6, 0.0),
        trajectory_normal=(0.0, 1.0),
        robot_poses=_straight_robot_poses(),
        robot_footprint=robot_footprint,
        context_trajectories={},
        context_footprints={},
        config={
            **_occluder_config(),
            "normal_offset_range_m": [0.75, 0.75],
            "pillar": {
                "length_range_m": [0.4, 0.4],
                "width_range_m": [0.4, 0.4],
            },
        },
        parameters=occluder_sampler.JointOccluderParameters(
            occluder_type="pillar",
            side=-1,
            offset_quantile=0.5,
            dimension_quantile=0.0,
            angle_multiplier=0.7,
            time_scale_quantile=0.16,
        ),
        proposal_index=0,
    )

    np.testing.assert_allclose(candidate.pose, occluder_pose, atol=1e-6)


def test_joint_occluder_rejects_collision_between_robot_samples() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot_footprint = inflate_footprint(
        RectangleFootprint(
            config["robot"]["length_m"], config["robot"]["width_m"]
        ),
        config["robot"]["inflation_m"],
    )
    future_robot_poses = np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32)
    occluder_footprint = RectangleFootprint(0.2, 0.2)
    occluder_pose = np.asarray([1.0, 0.0, 0.5 * np.pi], dtype=np.float64)
    endpoint_clearances = trajectory_signed_clearances(
        occluder_footprint,
        np.tile(occluder_pose, (2, 1)),
        robot_footprint,
        np.vstack((np.zeros(3), future_robot_poses)),
    )
    assert np.all(endpoint_clearances > 0.0)

    with pytest.raises(OccluderSamplingError) as exc_info:
        occluder_sampler.propose_environment_occluder_geometry(
            static_occupancy=np.zeros(
                (grid.height, grid.width), dtype=np.float32
            ),
            grid=grid,
            sensor_pose=(0.0, 0.0, 0.0),
            conflict_point=(0.25, 0.0),
            trajectory_normal=(1.0, 0.0),
            robot_poses=future_robot_poses,
            robot_footprint=robot_footprint,
            context_trajectories={},
            context_footprints={},
            config={
                **_occluder_config(),
                "normal_offset_range_m": [0.75, 0.75],
                "pillar": {
                    "length_range_m": [0.2, 0.2],
                    "width_range_m": [0.2, 0.2],
                },
            },
            parameters=occluder_sampler.JointOccluderParameters(
                occluder_type="pillar",
                side=1,
                offset_quantile=0.5,
                dimension_quantile=0.0,
                angle_multiplier=-0.7,
                time_scale_quantile=0.16,
            ),
            proposal_index=0,
        )

    assert exc_info.value.reason == "occluder_robot_swept_overlap"


def test_joint_occluder_rejects_narrow_collision_between_dense_samples() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot_footprint = RectangleFootprint(0.002, 0.002)
    future_robot_poses = np.asarray([[0.04, 0.0, 0.0]], dtype=np.float32)
    occluder_footprint = RectangleFootprint(0.002, 0.002)
    occluder_pose = np.asarray([0.02, 0.0, 0.5 * np.pi], dtype=np.float64)
    endpoint_clearances = trajectory_signed_clearances(
        occluder_footprint,
        np.tile(occluder_pose, (2, 1)),
        robot_footprint,
        np.vstack((np.zeros(3), future_robot_poses)),
    )
    assert np.all(endpoint_clearances > 0.0)

    with pytest.raises(OccluderSamplingError) as exc_info:
        occluder_sampler.propose_environment_occluder_geometry(
            static_occupancy=np.zeros(
                (grid.height, grid.width), dtype=np.float32
            ),
            grid=grid,
            sensor_pose=(0.0, 0.0, 0.0),
            conflict_point=(0.0, 0.0),
            trajectory_normal=(1.0, 0.0),
            robot_poses=future_robot_poses,
            robot_footprint=robot_footprint,
            context_trajectories={},
            context_footprints={},
            config={
                **_occluder_config(),
                "normal_offset_range_m": [0.02, 0.02],
                "pillar": {
                    "length_range_m": [0.002, 0.002],
                    "width_range_m": [0.002, 0.002],
                },
            },
            parameters=occluder_sampler.JointOccluderParameters(
                occluder_type="pillar",
                side=1,
                offset_quantile=0.5,
                dimension_quantile=0.0,
                angle_multiplier=-0.7,
                time_scale_quantile=0.16,
            ),
            proposal_index=0,
        )

    assert exc_info.value.reason == "occluder_robot_swept_overlap"


def test_occluder_geometry_is_proposed_before_target_exists() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot_footprint = inflate_footprint(
        RectangleFootprint(
            config["robot"]["length_m"], config["robot"]["width_m"]
        ),
        config["robot"]["inflation_m"],
    )
    static = np.zeros((grid.height, grid.width), dtype=np.float32)

    candidate = occluder_sampler.propose_environment_occluder_geometry(
        static_occupancy=static,
        grid=grid,
        sensor_pose=(0.0, 0.0, 0.0),
        conflict_point=(1.6, 0.0),
        trajectory_normal=(0.0, 1.0),
        robot_poses=_straight_robot_poses(),
        robot_footprint=robot_footprint,
        context_trajectories={},
        context_footprints={},
        config=_occluder_config(),
        parameters=occluder_sampler.JointOccluderParameters(
            occluder_type="pillar",
            side=-1,
            offset_quantile=0.5,
            dimension_quantile=0.0,
            angle_multiplier=0.0,
            time_scale_quantile=0.5,
        ),
        proposal_index=0,
    )

    np.testing.assert_allclose(candidate.pose[:2], [1.6, -1.0], atol=1e-6)
    assert candidate.occluder["placement_strategy"] == "joint_occluder_first_v2"
    assert candidate.occluder["proposal_index"] == 0
    assert candidate.mask.dtype == np.bool_
    assert not np.any(
        candidate.mask
        & rasterize_footprint_sweep(
            robot_footprint,
            np.vstack((np.zeros(3, dtype=np.float32), _straight_robot_poses())),
            grid,
        )
    )


def test_joint_occluder_keeps_normal_band_and_aligns_tangentially_to_target_los() -> None:
    config = load_config()
    grid = build_grid_spec(config)
    robot_footprint = inflate_footprint(
        RectangleFootprint(
            config["robot"]["length_m"], config["robot"]["width_m"]
        ),
        config["robot"]["inflation_m"],
    )
    static = np.zeros((grid.height, grid.width), dtype=np.float32)
    candidate = occluder_sampler.propose_environment_occluder_geometry(
        static_occupancy=static,
        grid=grid,
        sensor_pose=(0.0, 0.0, 0.0),
        conflict_point=(1.6, 0.0),
        trajectory_normal=(0.0, 1.0),
        robot_poses=_straight_robot_poses(),
        robot_footprint=robot_footprint,
        context_trajectories={},
        context_footprints={},
        config=_occluder_config(),
        parameters=occluder_sampler.JointOccluderParameters(
            occluder_type="pillar",
            side=-1,
            offset_quantile=0.5,
            dimension_quantile=0.0,
            angle_multiplier=0.95,
            time_scale_quantile=0.5,
        ),
        proposal_index=4,
    )

    aligned = occluder_sampler.align_environment_occluder_to_target_los(
        candidate,
        static_occupancy=static,
        grid=grid,
        sensor_pose=(0.0, 0.0, 0.0),
        trajectory_normal=(0.0, 1.0),
        robot_poses=_straight_robot_poses(),
        robot_footprint=robot_footprint,
        target_current_pose=(1.6, -2.0, np.pi / 2.0),
        context_trajectories={},
        context_footprints={},
    )

    np.testing.assert_allclose(aligned.pose[:2], [0.8, -1.0], atol=1e-6)
    assert aligned.occluder["normal_offset_m"] == -1.0
    assert aligned.occluder["line_of_sight_fraction"] == pytest.approx(0.5)
    assert aligned.occluder["placement_strategy"] == "joint_occluder_first_v2"
