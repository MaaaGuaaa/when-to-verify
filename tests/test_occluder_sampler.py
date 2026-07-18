"""Focused contract tests for reusable continuous collision sweeps."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from src.contracts import GridSpec
from src.generation.occluder_sampler import (
    OCCLUDER_COLLISION_SWEEP_PREPARATION_VERSION,
    OccluderCollisionSweep,
    PreparedOccluderCollisionSweep,
    occluder_collision_sweep_rejection_reason,
    prepare_occluder_collision_sweep,
)
from src.geometry import CircleFootprint, RectangleFootprint


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
        == "occluder_collision_sweep_preparation_v1"
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

    stale = replace(prepared, preparation_version="legacy_preparation_v0")
    with pytest.raises(ValueError, match="preparation version mismatch"):
        occluder_collision_sweep_rejection_reason(
            occluder,
            (1.0, 1.0, 0.0),
            (stale,),
            grid=grid,
        )


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
