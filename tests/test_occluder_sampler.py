"""Focused contract tests for reusable continuous collision sweeps."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

import src.generation.occluder_sampler as occluder_sampler_module
from src.contracts import GridSpec
from src.generation.occluder_sampler import (
    OCCLUDER_COLLISION_SWEEP_PREPARATION_VERSION,
    OccluderCollisionSweep,
    PreparedOccluderCollisionSweep,
    occluder_collision_sweep_rejection_reason,
    prepare_occluder_collision_sweep,
    synchronized_sweeps_intersect,
    swept_footprint_intersects_occupancy,
)
from src.geometry import (
    CircleFootprint,
    RectangleFootprint,
    grid_to_world,
    signed_clearance,
    trajectory_signed_clearances,
)


def _grid(*, resolution_m: float = 0.1) -> GridSpec:
    return GridSpec(
        height=40,
        width=40,
        history_steps=2,
        future_steps=2,
        resolution_m=resolution_m,
    )


def _raw_sweep(
    poses: np.ndarray,
    *,
    footprint=None,
    reason: str = "occluder_robot_swept_overlap",
) -> OccluderCollisionSweep:
    return OccluderCollisionSweep(
        footprint=footprint or RectangleFootprint(0.002, 0.002),
        poses=poses,
        rejection_reason=reason,
    )


@pytest.mark.parametrize(
    ("footprint_a", "footprint_b"),
    (
        (CircleFootprint(0.1), CircleFootprint(0.1)),
        (CircleFootprint(0.1), RectangleFootprint(0.2, 0.1)),
        (RectangleFootprint(0.2, 0.1), RectangleFootprint(0.2, 0.1)),
    ),
    ids=("circle-circle", "circle-rectangle", "rectangle-rectangle"),
)
def test_synchronized_sweeps_reject_between_frame_position_exchange(
    footprint_a,
    footprint_b,
) -> None:
    poses_a = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    poses_b = np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    assert signed_clearance(footprint_a, poses_a[0], footprint_b, poses_b[0]) > 0.0
    assert signed_clearance(footprint_a, poses_a[1], footprint_b, poses_b[1]) > 0.0

    assert synchronized_sweeps_intersect(
        footprint_a,
        poses_a,
        footprint_b,
        poses_b,
        grid=_grid(),
    )


def test_synchronized_common_translation_preserves_tiny_positive_gap() -> None:
    footprint = CircleFootprint(0.1)
    poses_a = np.asarray([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    poses_b = poses_a.copy()
    poses_b[:, 1] += 0.20000005
    assert np.all(
        trajectory_signed_clearances(
            footprint,
            poses_a,
            footprint,
            poses_b,
        )
        > 0.0
    )
    assert not synchronized_sweeps_intersect(
        footprint,
        poses_a,
        footprint,
        poses_b,
        grid=_grid(),
    )


def test_circle_yaw_does_not_consume_clearance_margin() -> None:
    footprint = CircleFootprint(0.1)
    poses_a = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, np.deg2rad(5.0)]]
    )
    poses_b = np.asarray(
        [[0.0, 0.200000001, 0.0], [0.0, 0.200000001, 0.0]]
    )
    assert np.all(
        trajectory_signed_clearances(
            footprint,
            poses_a,
            footprint,
            poses_b,
        )
        > 0.0
    )

    assert not synchronized_sweeps_intersect(
        footprint,
        poses_a,
        footprint,
        poses_b,
        grid=_grid(),
    )


def test_static_occupancy_rejects_contact_between_rotation_samples() -> None:
    grid = GridSpec(
        height=200,
        width=200,
        history_steps=2,
        future_steps=2,
        resolution_m=0.02,
    )
    cell_index = np.asarray([102, 149], dtype=np.int64)
    cell_center = grid_to_world(cell_index, grid)
    contact_yaw = float(np.arctan2(cell_center[1], cell_center[0]))
    footprint = RectangleFootprint(
        2.0 * float(np.linalg.norm(cell_center)),
        0.002,
    )
    poses = np.asarray(
        [
            [0.0, 0.0, contact_yaw - np.deg2rad(2.25)],
            [0.0, 0.0, contact_yaw + np.deg2rad(6.75)],
        ]
    )
    fixed_samples = np.asarray(
        [
            poses[0],
            [0.0, 0.0, contact_yaw + np.deg2rad(2.25)],
            poses[1],
        ]
    )
    cell = RectangleFootprint(grid.resolution_m, grid.resolution_m)
    cell_pose = np.asarray([*cell_center, 0.0])
    assert all(
        signed_clearance(footprint, pose, cell, cell_pose) > 0.0
        for pose in fixed_samples
    )
    occupancy = np.zeros((grid.height, grid.width), dtype=bool)
    occupancy[tuple(cell_index)] = True

    assert swept_footprint_intersects_occupancy(
        footprint,
        poses,
        occupancy,
        grid=grid,
    )


def test_static_occupancy_broadphase_is_local_to_each_dense_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = GridSpec(
        height=100,
        width=100,
        history_steps=2,
        future_steps=2,
        resolution_m=0.1,
    )
    poses = np.asarray([[-4.0, -4.0, 0.0], [4.0, 4.0, 0.0]])
    occupancy = np.zeros((grid.height, grid.width), dtype=bool)
    occupancy[10:30, 70:90] = True
    occupancy[70:90, 10:30] = True
    real_intersection = (
        occluder_sampler_module._dense_synchronized_sweeps_intersect
    )
    checked_cells = 0

    def count_checked_cells(*args, **kwargs) -> bool:
        nonlocal checked_cells
        checked_cells += 1
        return real_intersection(*args, **kwargs)

    monkeypatch.setattr(
        occluder_sampler_module,
        "_dense_synchronized_sweeps_intersect",
        count_checked_cells,
    )

    assert not swept_footprint_intersects_occupancy(
        CircleFootprint(0.05),
        poses,
        occupancy,
        grid=grid,
    )
    assert checked_cells <= 2


def test_prepare_collision_sweep_owns_canonical_read_only_interval_geometry() -> None:
    source = np.asfortranarray(
        np.asarray(
            [[0.0, 0.0, 0.0], [0.13, 0.0, np.deg2rad(12.0)]],
            dtype=np.float32,
        )
    )
    original = source.copy()
    grid = _grid(resolution_m=0.1)
    footprint = RectangleFootprint(0.4, 0.2)

    prepared = prepare_occluder_collision_sweep(
        _raw_sweep(source, footprint=footprint, reason="robot-contact"),
        grid=grid,
    )
    source[:] = 99.0

    assert isinstance(prepared, PreparedOccluderCollisionSweep)
    assert prepared.footprint == footprint
    assert prepared.rejection_reason == "robot-contact"
    assert prepared.grid == grid
    assert (
        prepared.preparation_version
        == OCCLUDER_COLLISION_SWEEP_PREPARATION_VERSION
        == "occluder_collision_sweep_preparation_v2"
    )
    assert prepared.dense_poses.dtype == np.float64
    assert prepared.dense_poses.flags.c_contiguous
    assert not prepared.dense_poses.flags.writeable
    assert prepared.interval_motion_bounds_m.dtype == np.float64
    assert prepared.interval_motion_bounds_m.flags.c_contiguous
    assert not prepared.interval_motion_bounds_m.flags.writeable
    assert prepared.interval_motion_bounds_m.shape == (
        prepared.dense_poses.shape[0] - 1,
    )
    assert np.isfinite(prepared.dense_poses).all()
    assert np.isfinite(prepared.interval_motion_bounds_m).all()
    np.testing.assert_array_equal(prepared.dense_poses[0], original[0])
    np.testing.assert_array_equal(prepared.dense_poses[-1], original[-1])
    with pytest.raises(ValueError, match="read-only"):
        prepared.dense_poses[0, 0] = 1.0
    for values in (
        prepared.dense_poses,
        prepared.interval_motion_bounds_m,
    ):
        with pytest.raises(ValueError, match="WRITEABLE"):
            values.setflags(write=True)
        assert not values.flags.owndata
    with pytest.raises(FrozenInstanceError):
        prepared.rejection_reason = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("poses", "occluder_pose", "expected_reason"),
    [
        (
            np.asarray([[0.0, 1.0, 0.0], [0.04, 1.0, 0.0]]),
            np.asarray([0.02, 0.0, 0.0]),
            None,
        ),
        (
            np.asarray([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0]]),
            np.asarray([0.0, 0.0, 0.0]),
            "occluder_robot_swept_overlap",
        ),
        (
            np.asarray([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0]]),
            np.asarray([0.02, 0.0, 0.0]),
            "occluder_robot_swept_overlap",
        ),
    ],
    ids=("clear", "sample-frame-contact", "narrow-between-frame-contact"),
)
def test_prepared_and_raw_sweeps_have_identical_continuous_verdicts(
    poses: np.ndarray,
    occluder_pose: np.ndarray,
    expected_reason: str | None,
) -> None:
    grid = _grid()
    raw = _raw_sweep(poses)
    prepared = prepare_occluder_collision_sweep(raw, grid=grid)
    occluder = RectangleFootprint(0.002, 0.002)

    raw_reason = occluder_collision_sweep_rejection_reason(
        occluder,
        occluder_pose,
        (raw,),
        grid=grid,
    )
    prepared_reason = occluder_collision_sweep_rejection_reason(
        occluder,
        occluder_pose,
        (prepared,),
        grid=grid,
    )

    assert raw_reason == prepared_reason == expected_reason


def test_collision_validator_accepts_mixed_raw_and_prepared_sweeps_in_order() -> None:
    grid = _grid()
    clear_raw = _raw_sweep(
        np.asarray([[0.0, 1.0, 0.0], [0.04, 1.0, 0.0]]),
        reason="clear-first",
    )
    collision_raw = _raw_sweep(
        np.asarray([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0]]),
        reason="prepared-second",
    )
    prepared_collision = prepare_occluder_collision_sweep(
        collision_raw,
        grid=grid,
    )

    reason = occluder_collision_sweep_rejection_reason(
        RectangleFootprint(0.002, 0.002),
        np.asarray([0.02, 0.0, 0.0]),
        (clear_raw, prepared_collision),
        grid=grid,
    )

    assert reason == "prepared-second"


def test_prepared_sweep_rejects_stale_grid_and_preparation_version() -> None:
    grid = _grid()
    prepared = prepare_occluder_collision_sweep(
        _raw_sweep(np.asarray([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0]])),
        grid=grid,
    )
    occluder = CircleFootprint(0.01)

    with pytest.raises(ValueError, match="grid mismatch"):
        occluder_collision_sweep_rejection_reason(
            occluder,
            (1.0, 1.0, 0.0),
            (prepared,),
            grid=replace(grid, resolution_m=0.2),
        )

    stale = replace(
        prepared,
        preparation_version="occluder_collision_sweep_preparation_v1",
    )
    with pytest.raises(ValueError, match="preparation version mismatch"):
        occluder_collision_sweep_rejection_reason(
            occluder,
            (1.0, 1.0, 0.0),
            (stale,),
            grid=grid,
        )


def test_circle_yaw_only_preparation_has_no_dense_substeps_or_motion_bound() -> None:
    grid = _grid()
    poses = np.asarray(
        [[0.0, 0.0, -0.75 * np.pi], [0.0, 0.0, 0.75 * np.pi]],
        dtype=np.float64,
    )

    prepared = prepare_occluder_collision_sweep(
        _raw_sweep(poses, footprint=CircleFootprint(0.1)),
        grid=grid,
    )

    assert prepared.dense_poses.shape == (2, 3)
    np.testing.assert_array_equal(prepared.dense_poses, poses)
    np.testing.assert_array_equal(
        prepared.interval_motion_bounds_m,
        np.zeros(1, dtype=np.float64),
    )


def test_circle_preparation_interval_bounds_keep_translation_only() -> None:
    grid = _grid()
    poses = np.asarray(
        [[0.0, 0.0, -0.75 * np.pi], [0.13, 0.04, 0.75 * np.pi]],
        dtype=np.float64,
    )

    prepared = prepare_occluder_collision_sweep(
        _raw_sweep(poses, footprint=CircleFootprint(0.1)),
        grid=grid,
    )
    dense_translations = np.linalg.norm(
        np.diff(prepared.dense_poses[:, :2], axis=0),
        axis=1,
    )

    assert prepared.dense_poses.shape == (4, 3)
    np.testing.assert_array_equal(
        prepared.interval_motion_bounds_m,
        dense_translations,
    )
    assert np.sum(prepared.interval_motion_bounds_m) == pytest.approx(
        np.linalg.norm(poses[1, :2] - poses[0, :2])
    )


def test_circle_yaw_only_prepared_and_raw_sweeps_preserve_tiny_gap() -> None:
    grid = _grid()
    footprint = CircleFootprint(0.1)
    poses = np.asarray(
        [[0.0, 0.0, -0.75 * np.pi], [0.0, 0.0, 0.75 * np.pi]],
        dtype=np.float64,
    )
    raw = _raw_sweep(poses, footprint=footprint)
    prepared = prepare_occluder_collision_sweep(raw, grid=grid)
    candidate_pose = np.asarray([0.0, 0.200000001, 0.0])
    candidate = CircleFootprint(0.1)
    assert np.all(
        trajectory_signed_clearances(
            candidate,
            np.tile(candidate_pose, (poses.shape[0], 1)),
            footprint,
            poses,
        )
        > 0.0
    )

    raw_reason = occluder_collision_sweep_rejection_reason(
        candidate,
        candidate_pose,
        (raw,),
        grid=grid,
    )
    prepared_reason = occluder_collision_sweep_rejection_reason(
        candidate,
        candidate_pose,
        (prepared,),
        grid=grid,
    )

    assert raw_reason is None
    assert prepared_reason == raw_reason


def test_rectangle_yaw_preparation_rejects_between_endpoint_contact() -> None:
    grid = _grid()
    footprint = RectangleFootprint(2.0, 0.04)
    poses = np.asarray(
        [[0.0, 0.0, -0.25 * np.pi], [0.0, 0.0, 0.25 * np.pi]],
        dtype=np.float64,
    )
    candidate = CircleFootprint(0.02)
    candidate_pose = np.asarray([0.9, 0.0, 0.0])
    assert all(
        signed_clearance(candidate, candidate_pose, footprint, pose) > 0.0
        for pose in poses
    )

    prepared = prepare_occluder_collision_sweep(
        _raw_sweep(poses, footprint=footprint),
        grid=grid,
    )

    assert prepared.dense_poses.shape[0] > poses.shape[0]
    assert occluder_collision_sweep_rejection_reason(
        candidate,
        candidate_pose,
        (prepared,),
        grid=grid,
    ) == "occluder_robot_swept_overlap"


@pytest.mark.parametrize(
    "moving_footprint",
    (CircleFootprint(0.12), RectangleFootprint(0.30, 0.16)),
    ids=("circle", "rectangle"),
)
def test_prepared_random_sweeps_do_not_miss_dense_reference_contacts(
    moving_footprint,
) -> None:
    grid = _grid(resolution_m=0.08)
    candidate = CircleFootprint(0.06)
    rng = np.random.default_rng(20260719)
    dense_fractions = np.linspace(0.0, 1.0, 2001, dtype=np.float64)
    reference_contacts = 0

    for _ in range(48):
        start = rng.uniform((-0.6, -0.6, -np.pi), (0.6, 0.6, np.pi))
        end = rng.uniform((-0.6, -0.6, -np.pi), (0.6, 0.6, np.pi))
        poses = np.vstack((start, end))
        yaw_delta = float(
            occluder_sampler_module.wrap_angle(end[2] - start[2])
        )
        dense_poses = np.empty((dense_fractions.size, 3), dtype=np.float64)
        dense_poses[:, :2] = start[:2] + dense_fractions[:, None] * (
            end[:2] - start[:2]
        )
        dense_poses[:, 2] = occluder_sampler_module.wrap_angle(
            start[2] + dense_fractions * yaw_delta
        )
        candidate_pose = rng.uniform(
            (-0.8, -0.8, -np.pi),
            (0.8, 0.8, np.pi),
        )
        dense_candidate_poses = np.tile(
            candidate_pose,
            (dense_poses.shape[0], 1),
        )
        reference_intersects = bool(
            np.any(
                trajectory_signed_clearances(
                    candidate,
                    dense_candidate_poses,
                    moving_footprint,
                    dense_poses,
                )
                <= 0.0
            )
        )
        if not reference_intersects:
            continue
        reference_contacts += 1
        prepared = prepare_occluder_collision_sweep(
            _raw_sweep(poses, footprint=moving_footprint),
            grid=grid,
        )
        assert occluder_collision_sweep_rejection_reason(
            candidate,
            candidate_pose,
            (prepared,),
            grid=grid,
        ) == "occluder_robot_swept_overlap"

    assert reference_contacts >= 4


def test_prepared_sweep_rejects_forged_zero_interval_motion_bounds() -> None:
    grid = _grid()
    raw = _raw_sweep(
        np.asarray([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0]]),
    )
    prepared = prepare_occluder_collision_sweep(raw, grid=grid)
    occluder = RectangleFootprint(0.002, 0.002)
    assert occluder_collision_sweep_rejection_reason(
        occluder,
        (0.02, 0.0, 0.0),
        (raw,),
        grid=grid,
    ) == "occluder_robot_swept_overlap"

    with pytest.raises(ValueError, match="canonical interval motion bounds"):
        replace(
            prepared,
            interval_motion_bounds_m=np.zeros_like(
                prepared.interval_motion_bounds_m
            ),
        )


def test_prepared_sweep_keeps_clearance_candidate_specific() -> None:
    grid = _grid()
    prepared = prepare_occluder_collision_sweep(
        OccluderCollisionSweep(
            footprint=CircleFootprint(0.05),
            poses=np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            rejection_reason="robot-contact",
        ),
        grid=grid,
    )
    occluder = CircleFootprint(0.05)

    colliding = occluder_collision_sweep_rejection_reason(
        occluder,
        (0.25, 0.0, 0.0),
        (prepared,),
        grid=grid,
    )
    clear = occluder_collision_sweep_rejection_reason(
        occluder,
        (0.25, 1.0, 0.0),
        (prepared,),
        grid=grid,
    )

    assert colliding == "robot-contact"
    assert clear is None


@pytest.mark.parametrize(
    "bad_poses",
    [
        np.empty((0, 3), dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        np.asarray([[0.0, np.nan, 0.0]], dtype=np.float64),
        np.asarray([[0.0, np.inf, 0.0]], dtype=np.float64),
    ],
)
def test_prepare_collision_sweep_rejects_invalid_pose_arrays(
    bad_poses: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        prepare_occluder_collision_sweep(
            _raw_sweep(bad_poses),
            grid=_grid(),
        )
