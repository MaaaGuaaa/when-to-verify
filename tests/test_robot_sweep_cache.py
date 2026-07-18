"""Behavioral tests for the strict in-process SOP04 robot-sweep cache."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from src.contracts import GridSpec, LocalTrajectory
from src.generation.occluder_sampler import (
    OCCLUDER_COLLISION_SWEEP_PREPARATION_VERSION,
    occluder_collision_sweep_rejection_reason,
)
from src.generation.robot_sweep_cache import (
    ROBOT_SWEEP_CACHE_VERSION,
    RobotSweepCache,
    RobotSweepCacheIdentityError,
)
from src.geometry import CircleFootprint, RectangleFootprint


_LAYOUT_VERSION = "future_endpoints_dt_to_horizon_v1"
_REJECTION_REASON = "occluder_robot_swept_overlap"


def _grid() -> GridSpec:
    return GridSpec(
        height=8,
        width=10,
        history_steps=3,
        future_steps=3,
        resolution_m=0.1,
    )


def _trajectory(
    *,
    trajectory_id: str = "traj-001",
    grid: GridSpec | None = None,
    future_dt_s: float = 0.2,
    poses: np.ndarray | None = None,
    swept_mask: np.ndarray | None = None,
    metadata: dict | None = None,
) -> LocalTrajectory:
    grid = grid or _grid()
    if poses is None:
        poses = np.asarray(
            [[0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0]],
            dtype=np.float32,
        )
    if swept_mask is None:
        swept_mask = np.zeros((grid.height, grid.width), dtype=np.float32)
        swept_mask[grid.height // 2, 1:4] = 1.0
    if metadata is None:
        metadata = {
            "pose_time_layout_version": _LAYOUT_VERSION,
            "pose_time_offsets_s": (
                np.arange(grid.future_steps, dtype=np.float64) + 1.0
            )
            * future_dt_s,
        }
    maps = np.zeros((grid.height, grid.width), dtype=np.float32)
    return LocalTrajectory(
        trajectory_id=trajectory_id,
        poses=poses,
        controls=np.zeros((grid.future_steps, 2), dtype=np.float32),
        swept_mask=swept_mask,
        tta_map=maps.copy(),
        braking_map=maps.copy(),
        centerline_map=maps.copy(),
        task_cost=0.0,
        metadata=metadata,
    )


def _get(
    cache: RobotSweepCache,
    trajectory: LocalTrajectory,
    *,
    footprint=None,
    grid: GridSpec | None = None,
    future_dt_s: float = 0.2,
    rejection_reason: str = _REJECTION_REASON,
):
    return cache.get(
        trajectory,
        robot_footprint=footprint or RectangleFootprint(0.4, 0.2),
        grid=grid or _grid(),
        future_dt_s=future_dt_s,
        rejection_reason=rejection_reason,
    )


def test_repeated_get_builds_once_and_returns_same_immutable_entry() -> None:
    cache = RobotSweepCache()
    trajectory = _trajectory()

    cold = _get(cache, trajectory)
    warm = _get(cache, trajectory)

    assert cold is warm
    assert cache.stats.size == 1
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.builds == 1
    assert cold.trajectory_id == trajectory.trajectory_id
    assert cold.key.trajectory_id == trajectory.trajectory_id
    assert cold.key.cache_version == ROBOT_SWEEP_CACHE_VERSION == "robot_sweep_cache_v1"
    assert (
        cold.key.preparation_version
        == OCCLUDER_COLLISION_SWEEP_PREPARATION_VERSION
    )
    assert cold.key.pose_time_layout_version == _LAYOUT_VERSION
    assert cold.key.rejection_reason == _REJECTION_REASON
    assert cold.key.grid == _grid()
    assert cold.key.footprint == RectangleFootprint(0.4, 0.2)
    assert cold.swept_mask.dtype == np.float32
    assert cold.swept_mask.flags.c_contiguous
    assert cold.swept_mask.flags.owndata
    assert not cold.swept_mask.flags.writeable
    assert cold.prepared_future_sweep.dense_poses.dtype == np.float64
    assert cold.prepared_future_sweep.dense_poses.flags.c_contiguous
    assert cold.prepared_future_sweep.dense_poses.flags.owndata
    assert not cold.prepared_future_sweep.dense_poses.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        cold.swept_mask[0, 0] = 1.0
    with pytest.raises(FrozenInstanceError):
        cold.trajectory_id = "changed"  # type: ignore[misc]


def test_key_binds_exact_canonical_bytes_and_cold_warm_builds_are_identical() -> None:
    trajectory = _trajectory()
    footprint = CircleFootprint(0.25)
    first_cache = RobotSweepCache()
    first = _get(first_cache, trajectory, footprint=footprint)
    warm = _get(first_cache, trajectory, footprint=footprint)
    cold_again = _get(RobotSweepCache(), trajectory, footprint=footprint)
    expected_poses = np.array(trajectory.poses, dtype=np.float64, order="C", copy=True)
    expected_mask = np.array(
        trajectory.swept_mask,
        dtype=np.float32,
        order="C",
        copy=True,
    )
    expected_offsets = np.array(
        trajectory.metadata["pose_time_offsets_s"],
        dtype=np.float64,
        order="C",
        copy=True,
    )

    assert first.key.pose_bytes == expected_poses.tobytes(order="C")
    assert first.key.swept_mask_bytes == expected_mask.tobytes(order="C")
    assert first.key.pose_time_offsets_bytes == expected_offsets.tobytes(order="C")
    assert first.key.footprint == footprint
    assert first.key.canonical_digest == warm.key.canonical_digest
    assert first.key.canonical_digest == cold_again.key.canonical_digest
    assert first.swept_mask.tobytes() == warm.swept_mask.tobytes()
    assert first.swept_mask.tobytes() == cold_again.swept_mask.tobytes()
    assert (
        first.prepared_future_sweep.dense_poses.tobytes()
        == warm.prepared_future_sweep.dense_poses.tobytes()
        == cold_again.prepared_future_sweep.dense_poses.tobytes()
    )
    assert (
        first.prepared_future_sweep.interval_motion_bounds_m.tobytes()
        == cold_again.prepared_future_sweep.interval_motion_bounds_m.tobytes()
    )


def test_cache_owns_inputs_and_detects_caller_mutation_under_same_identity() -> None:
    poses = np.asarray(
        [[0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0]],
        dtype=np.float32,
    )
    mask = np.zeros((_grid().height, _grid().width), dtype=np.float32)
    mask[2, 2] = 1.0
    trajectory = _trajectory(poses=poses, swept_mask=mask)
    cache = RobotSweepCache()
    entry = _get(cache, trajectory)
    entry_pose_bytes = entry.prepared_future_sweep.dense_poses.tobytes()
    entry_mask_bytes = entry.swept_mask.tobytes()

    poses[0, 0] += 1.0
    mask[2, 2] = 0.0

    assert entry.prepared_future_sweep.dense_poses.tobytes() == entry_pose_bytes
    assert entry.swept_mask.tobytes() == entry_mask_bytes
    with pytest.raises(RobotSweepCacheIdentityError, match="traj-001.*binding changed"):
        _get(cache, trajectory)


@pytest.mark.parametrize(
    "changed",
    ("poses", "swept-mask", "footprint", "grid", "future-dt", "reason"),
)
def test_same_trajectory_id_with_changed_binding_is_a_hard_identity_error(
    changed: str,
) -> None:
    cache = RobotSweepCache()
    grid = _grid()
    original = _trajectory(grid=grid)
    _get(cache, original, grid=grid)

    trajectory = _trajectory(grid=grid)
    footprint = RectangleFootprint(0.4, 0.2)
    active_grid = grid
    future_dt_s = 0.2
    reason = _REJECTION_REASON
    if changed == "poses":
        modified = np.asarray(trajectory.poses).copy()
        modified[1, 1] = 0.01
        trajectory = replace(trajectory, poses=modified)
    elif changed == "swept-mask":
        modified = np.asarray(trajectory.swept_mask).copy()
        modified[0, 0] = 1.0
        trajectory = replace(trajectory, swept_mask=modified)
    elif changed == "footprint":
        footprint = RectangleFootprint(0.5, 0.2)
    elif changed == "grid":
        active_grid = replace(grid, resolution_m=0.2)
    elif changed == "future-dt":
        future_dt_s = 0.1
        trajectory = _trajectory(
            grid=grid,
            future_dt_s=future_dt_s,
            trajectory_id=trajectory.trajectory_id,
        )
    else:
        reason = "different-rejection-reason"

    with pytest.raises(RobotSweepCacheIdentityError, match="traj-001.*binding changed"):
        _get(
            cache,
            trajectory,
            footprint=footprint,
            grid=active_grid,
            future_dt_s=future_dt_s,
            rejection_reason=reason,
        )

    assert cache.stats.size == 1
    assert cache.stats.builds == 1


def test_distinct_trajectory_ids_never_alias_even_with_identical_arrays() -> None:
    cache = RobotSweepCache()
    first = _get(cache, _trajectory(trajectory_id="traj-a"))
    second = _get(cache, _trajectory(trajectory_id="traj-b"))

    assert first is not second
    assert first.key.canonical_digest != second.key.canonical_digest
    assert cache.stats.size == 2
    assert cache.stats.hits == 0
    assert cache.stats.misses == 2
    assert cache.stats.builds == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("poses", np.zeros((2, 3), dtype=np.float32), "poses.*shape"),
        (
            "poses",
            np.asarray(
                [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
            "poses.*finite",
        ),
        (
            "poses",
            np.asarray(
                [[0.0, 0.0, 0.0], [np.inf, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
            "poses.*finite",
        ),
        ("swept_mask", np.zeros((8, 9), dtype=np.float32), "swept_mask.*shape"),
        ("swept_mask", np.zeros((8, 10), dtype=np.float64), "swept_mask.*float32"),
        (
            "swept_mask",
            np.full((8, 10), 0.5, dtype=np.float32),
            "swept_mask.*binary",
        ),
        (
            "swept_mask",
            np.full((8, 10), np.nan, dtype=np.float32),
            "swept_mask.*finite",
        ),
        (
            "swept_mask",
            np.full((8, 10), np.inf, dtype=np.float32),
            "swept_mask.*finite",
        ),
    ],
)
def test_cache_rejects_invalid_pose_and_persisted_mask_arrays(
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    trajectory = replace(_trajectory(), **{field: value})

    with pytest.raises((TypeError, ValueError), match=message):
        _get(RobotSweepCache(), trajectory)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"pose_time_offsets_s": np.asarray([0.2, 0.4, 0.6])}, "layout version"),
        (
            {
                "pose_time_layout_version": "legacy_t0_to_horizon_minus_dt_v0",
                "pose_time_offsets_s": np.asarray([0.0, 0.2, 0.4]),
            },
            "layout version",
        ),
        ({"pose_time_layout_version": _LAYOUT_VERSION}, "offsets"),
        (
            {
                "pose_time_layout_version": _LAYOUT_VERSION,
                "pose_time_offsets_s": np.asarray([0.2, np.nan, 0.6]),
            },
            "offsets.*finite",
        ),
        (
            {
                "pose_time_layout_version": _LAYOUT_VERSION,
                "pose_time_offsets_s": np.asarray([0.4, 0.2, 0.6]),
            },
            "offsets.*mismatch",
        ),
    ],
)
def test_cache_rejects_missing_stale_or_invalid_pose_time_layout(
    metadata: dict,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _get(RobotSweepCache(), _trajectory(metadata=metadata))


@pytest.mark.parametrize("future_dt_s", [0.0, -0.2, np.nan, np.inf, True])
def test_cache_rejects_invalid_future_dt(future_dt_s) -> None:
    with pytest.raises((TypeError, ValueError), match="future_dt_s"):
        _get(RobotSweepCache(), _trajectory(), future_dt_s=future_dt_s)


def test_persisted_mask_is_broad_phase_evidence_not_a_collision_verdict() -> None:
    grid = _grid()
    all_hit_mask = np.ones((grid.height, grid.width), dtype=np.float32)
    trajectory = _trajectory(grid=grid, swept_mask=all_hit_mask)
    entry = _get(
        RobotSweepCache(),
        trajectory,
        grid=grid,
        footprint=CircleFootprint(0.02),
    )
    occluder = CircleFootprint(0.02)

    clear = occluder_collision_sweep_rejection_reason(
        occluder,
        (0.4, 1.0, 0.0),
        (entry.prepared_future_sweep,),
        grid=grid,
    )
    contact = occluder_collision_sweep_rejection_reason(
        occluder,
        (0.4, 0.0, 0.0),
        (entry.prepared_future_sweep,),
        grid=grid,
    )

    assert np.all(entry.swept_mask == 1.0)
    assert clear is None
    assert contact == _REJECTION_REASON
    # The cache contains only canonical future poses; base history/context must
    # be prepared separately by each caller and cannot alias through this entry.
    np.testing.assert_array_equal(
        entry.prepared_future_sweep.dense_poses[0],
        np.asarray(trajectory.poses, dtype=np.float64)[0],
    )
