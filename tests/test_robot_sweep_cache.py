"""Behavioral tests for the strict in-process SOP04 robot-sweep cache."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from src.contracts import BaseState, GridSpec, LocalTrajectory
import src.generation.robot_sweep_cache as robot_sweep_cache
from src.generation.occluder_sampler import (
    OCCLUDER_COLLISION_SWEEP_PREPARATION_VERSION,
    OccluderCollisionSweep,
    occluder_collision_sweep_rejection_reason,
)
from src.generation.robot_sweep_cache import (
    ROBOT_SWEEP_CACHE_VERSION,
    RobotSweepCache,
    RobotSweepCacheEntry,
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
            # SOP05 adapter injects JSON-safe tuple offsets into each v2 item.
            "pose_time_offsets_s": tuple(
                float(value)
                for value in (
                    (np.arange(grid.future_steps, dtype=np.float64) + 1.0)
                    * future_dt_s
                )
            ),
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


def _base_state(
    *,
    state_id: str = "base-001",
    grid: GridSpec | None = None,
    robot_history: np.ndarray | None = None,
) -> BaseState:
    grid = grid or _grid()
    if robot_history is None:
        robot_history = np.asarray(
            [[-0.4, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
    return BaseState(
        state_id=state_id,
        split="train",
        recording_id="recording-001",
        dynamic_object_ids=(),
        timestamp=1.0,
        robot_history=robot_history,
        robot_state=np.zeros((2,), dtype=np.float32),
        visible_dynamic_object_history={},
        visible_dynamic_object_specs={},
        static_map_local=np.zeros((grid.height, grid.width), dtype=np.float32),
    )


def _bundle(
    *,
    base_state: BaseState | None = None,
    trajectory: LocalTrajectory | None = None,
    footprint=None,
    grid: GridSpec | None = None,
    future_dt_s: float = 0.2,
    cache: RobotSweepCache | None = None,
    rejection_reason: str = _REJECTION_REASON,
):
    active_grid = _grid() if grid is None else grid
    fixture_grid = active_grid if isinstance(active_grid, GridSpec) else _grid()
    return robot_sweep_cache.prepare_robot_collision_sweep_bundle(
        _base_state(grid=fixture_grid) if base_state is None else base_state,
        (
            _trajectory(grid=fixture_grid, future_dt_s=future_dt_s)
            if trajectory is None
            else trajectory
        ),
        robot_footprint=(
            CircleFootprint(0.02) if footprint is None else footprint
        ),
        grid=active_grid,
        future_dt_s=future_dt_s,
        cache=cache,
        rejection_reason=rejection_reason,
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
    assert not cold.swept_mask.flags.writeable
    assert cold.prepared_future_sweep.dense_poses.dtype == np.float64
    assert cold.prepared_future_sweep.dense_poses.flags.c_contiguous
    assert not cold.prepared_future_sweep.dense_poses.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        cold.swept_mask[0, 0] = 1.0
    with pytest.raises(FrozenInstanceError):
        cold.trajectory_id = "changed"  # type: ignore[misc]


def test_cached_arrays_cannot_be_reenabled_or_pollute_warm_cold_results() -> None:
    trajectory = _trajectory()
    cache = RobotSweepCache()
    cold = _get(cache, trajectory, footprint=CircleFootprint(0.02))
    arrays = (
        cold.swept_mask,
        cold.prepared_future_sweep.dense_poses,
        cold.prepared_future_sweep.interval_motion_bounds_m,
    )
    snapshots = tuple(values.tobytes() for values in arrays)
    occluder = CircleFootprint(0.02)

    for values in arrays:
        with pytest.raises(ValueError, match="WRITEABLE"):
            values.setflags(write=True)
        assert not values.flags.owndata

    warm = _get(cache, trajectory, footprint=CircleFootprint(0.02))
    cold_again = _get(
        RobotSweepCache(),
        trajectory,
        footprint=CircleFootprint(0.02),
    )
    assert tuple(values.tobytes() for values in arrays) == snapshots
    assert warm is cold
    assert warm.swept_mask.tobytes() == cold_again.swept_mask.tobytes()
    assert (
        warm.prepared_future_sweep.dense_poses.tobytes()
        == cold_again.prepared_future_sweep.dense_poses.tobytes()
    )
    assert (
        occluder_collision_sweep_rejection_reason(
            occluder,
            (0.4, 0.0, 0.0),
            (warm.prepared_future_sweep,),
            grid=_grid(),
        )
        == occluder_collision_sweep_rejection_reason(
            occluder,
            (0.4, 0.0, 0.0),
            (cold_again.prepared_future_sweep,),
            grid=_grid(),
        )
        == _REJECTION_REASON
    )


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


def test_entry_rejects_spliced_prepared_future_with_different_interior_pose() -> None:
    grid = _grid()
    first_poses = np.asarray(
        [[0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0]],
        dtype=np.float64,
    )
    second_poses = first_poses.copy()
    second_poses[1, 1] = 0.05
    cache = RobotSweepCache()
    first = _get(
        cache,
        _trajectory(trajectory_id="splice-a", poses=first_poses, grid=grid),
        grid=grid,
    )
    second = _get(
        cache,
        _trajectory(trajectory_id="splice-b", poses=second_poses, grid=grid),
        grid=grid,
    )
    np.testing.assert_array_equal(
        first.prepared_future_sweep.dense_poses[[0, -1]],
        second.prepared_future_sweep.dense_poses[[0, -1]],
    )

    for key_entry, prepared_entry in ((first, second), (second, first)):
        with pytest.raises(ValueError, match="prepared future sweep.*cache key"):
            RobotSweepCacheEntry(
                trajectory_id=key_entry.trajectory_id,
                key=key_entry.key,
                swept_mask=key_entry.swept_mask,
                prepared_future_sweep=prepared_entry.prepared_future_sweep,
            )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"canonical_digest": "0" * 64}, "canonical_digest"),
        ({"cache_version": "robot_sweep_cache_v0"}, "cache_version"),
        (
            {"preparation_version": "occluder_collision_sweep_preparation_v0"},
            "preparation_version",
        ),
    ],
)
def test_cache_key_rejects_forged_digest_or_versions(
    changes: dict[str, str],
    message: str,
) -> None:
    entry = _get(RobotSweepCache(), _trajectory())

    with pytest.raises(ValueError, match=message):
        replace(entry.key, **changes)


def test_entry_rejects_trajectory_id_that_disagrees_with_key() -> None:
    entry = _get(RobotSweepCache(), _trajectory())

    with pytest.raises(ValueError, match="trajectory_id.*cache key"):
        replace(entry, trajectory_id="forged-trajectory-id")


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


def test_collision_bundle_closes_current_to_first_future_seam() -> None:
    poses = np.asarray(
        [[0.4, 0.0, 0.0], [0.6, 0.0, 0.0], [0.8, 0.0, 0.0]],
        dtype=np.float32,
    )
    trajectory = _trajectory(poses=poses)
    robot_footprint = CircleFootprint(0.02)
    obstacle_footprint = CircleFootprint(0.02)
    obstacle_pose = (0.2, 0.0, 0.0)
    bundle = _bundle(trajectory=trajectory, footprint=robot_footprint)

    without_seam = occluder_collision_sweep_rejection_reason(
        obstacle_footprint,
        obstacle_pose,
        (bundle.history_sweep, bundle.future_sweep),
        grid=_grid(),
    )
    seam_only = occluder_collision_sweep_rejection_reason(
        obstacle_footprint,
        obstacle_pose,
        (bundle.seam_sweep,),
        grid=_grid(),
    )
    complete = occluder_collision_sweep_rejection_reason(
        obstacle_footprint,
        obstacle_pose,
        bundle.collision_sweeps,
        grid=_grid(),
    )

    assert without_seam is None
    assert seam_only == _REJECTION_REASON
    assert complete == _REJECTION_REASON


def test_collision_bundle_has_exact_ordered_interval_coverage() -> None:
    base_state = _base_state()
    trajectory = _trajectory()
    bundle = _bundle(base_state=base_state, trajectory=trajectory)
    history, seam, future = bundle.collision_sweeps

    assert history is bundle.history_sweep
    assert seam is bundle.seam_sweep
    assert future is bundle.future_sweep
    assert future is bundle.future_entry.prepared_future_sweep
    np.testing.assert_array_equal(history.dense_poses[0], base_state.robot_history[0])
    np.testing.assert_array_equal(history.dense_poses[-1], base_state.robot_history[-1])
    np.testing.assert_array_equal(seam.dense_poses[0], base_state.robot_history[-1])
    np.testing.assert_array_equal(seam.dense_poses[-1], trajectory.poses[0])
    np.testing.assert_array_equal(future.dense_poses[0], trajectory.poses[0])
    np.testing.assert_array_equal(future.dense_poses[-1], trajectory.poses[-1])

    stitched = np.vstack(
        (history.dense_poses, seam.dense_poses[1:], future.dense_poses[1:])
    )
    interval_count = sum(
        sweep.interval_motion_bounds_m.shape[0] for sweep in bundle.collision_sweeps
    )
    assert interval_count == stitched.shape[0] - 1
    assert np.all(np.linalg.norm(np.diff(stitched[:, :2], axis=0), axis=1) > 0.0)
    np.testing.assert_array_equal(
        np.frombuffer(
            bundle.future_entry.key.pose_time_offsets_bytes,
            dtype=np.dtype("<f8"),
        ),
        (np.arange(_grid().future_steps, dtype=np.float64) + 1.0) * 0.2,
    )


def test_collision_bundle_accepts_equivalent_unwrapped_seam_endpoint() -> None:
    poses = np.asarray(
        [
            [0.2, 0.0, 2.0 * np.pi + 0.1],
            [0.4, 0.0, 2.0 * np.pi + 0.2],
            [0.6, 0.0, 2.0 * np.pi + 0.3],
        ],
        dtype=np.float64,
    )
    trajectory = _trajectory(poses=poses)

    bundle = _bundle(trajectory=trajectory)

    np.testing.assert_array_equal(
        bundle.seam_sweep.dense_poses[-1, :2],
        poses[0, :2],
    )
    assert np.isclose(
        np.sin(
            bundle.seam_sweep.dense_poses[-1, 2]
            - bundle.future_sweep.dense_poses[0, 2]
        ),
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.cos(
        bundle.seam_sweep.dense_poses[-1, 2]
        - bundle.future_sweep.dense_poses[0, 2]
    ) > 0.0


def test_explicit_cache_reuses_only_future_and_preserves_cold_warm_verdicts() -> None:
    cache = RobotSweepCache()
    base_state = _base_state()
    trajectory = _trajectory()
    cold = _bundle(base_state=base_state, trajectory=trajectory, cache=cache)
    warm = _bundle(base_state=base_state, trajectory=trajectory, cache=cache)
    changed_history = np.asarray(base_state.robot_history).copy()
    changed_history[0, 1] = 0.1
    changed_base = replace(base_state, robot_history=changed_history)
    changed = _bundle(base_state=changed_base, trajectory=trajectory, cache=cache)
    obstacle = CircleFootprint(0.02)

    assert cache.stats.builds == 1
    assert cache.stats.hits == 2
    assert cold.future_entry is warm.future_entry is changed.future_entry
    assert cold.future_sweep is warm.future_sweep is changed.future_sweep
    assert cold.history_sweep is not warm.history_sweep
    assert cold.seam_sweep is not warm.seam_sweep
    assert (
        cold.history_sweep.dense_poses.tobytes()
        == warm.history_sweep.dense_poses.tobytes()
    )
    assert (
        cold.seam_sweep.dense_poses.tobytes()
        == warm.seam_sweep.dense_poses.tobytes()
    )
    assert (
        cold.history_sweep.dense_poses.tobytes()
        != changed.history_sweep.dense_poses.tobytes()
    )
    assert (
        cold.seam_sweep.dense_poses.tobytes()
        == changed.seam_sweep.dense_poses.tobytes()
    )
    for obstacle_pose in ((0.1, 0.0, 0.0), (0.0, 1.0, 0.0)):
        cold_verdict = occluder_collision_sweep_rejection_reason(
            obstacle, obstacle_pose, cold.collision_sweeps, grid=_grid()
        )
        warm_verdict = occluder_collision_sweep_rejection_reason(
            obstacle, obstacle_pose, warm.collision_sweeps, grid=_grid()
        )
        assert cold_verdict == warm_verdict

    uncached_first = _bundle(base_state=base_state, trajectory=trajectory)
    uncached_second = _bundle(base_state=base_state, trajectory=trajectory)
    assert uncached_first.future_entry is not uncached_second.future_entry
    assert (
        uncached_first.future_sweep.dense_poses.tobytes()
        == uncached_second.future_sweep.dense_poses.tobytes()
    )
    assert uncached_first.canonical_digest == uncached_second.canonical_digest


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("base_state", object(), "base_state"),
        ("trajectory", object(), "trajectory"),
        ("footprint", object(), "robot_footprint"),
        ("grid", object(), "grid"),
        ("future_dt_s", 0.0, "future_dt_s"),
        ("future_dt_s", np.nan, "future_dt_s"),
        ("cache", object(), "cache"),
        ("rejection_reason", "", "rejection_reason"),
    ],
)
def test_collision_bundle_rejects_invalid_public_inputs(
    field: str,
    value,
    message: str,
) -> None:
    kwargs = {field: value}

    with pytest.raises((TypeError, ValueError), match=message):
        _bundle(**kwargs)


@pytest.mark.parametrize("invalid_current", [0.01, np.nan, np.inf])
def test_collision_bundle_rejects_nonlocal_or_nonfinite_base_current(
    invalid_current: float,
) -> None:
    history = np.asarray(_base_state().robot_history).copy()
    history[-1, 0] = invalid_current

    with pytest.raises((TypeError, ValueError), match="robot_history|local origin"):
        _bundle(base_state=_base_state(robot_history=history))


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "pose_time_layout_version": "legacy_t0_v0",
            "pose_time_offsets_s": np.asarray([0.0, 0.2, 0.4]),
        },
        {
            "pose_time_layout_version": _LAYOUT_VERSION,
            "pose_time_offsets_s": np.asarray([0.2, 0.4, np.nan]),
        },
        {
            "pose_time_layout_version": _LAYOUT_VERSION,
            "pose_time_offsets_s": np.asarray([0.0, 0.2, 0.4]),
        },
    ],
)
def test_collision_bundle_rejects_stale_or_invalid_future_time_layout(
    metadata: dict,
) -> None:
    with pytest.raises((TypeError, ValueError), match="layout|offsets"):
        _bundle(trajectory=_trajectory(metadata=metadata))


@pytest.mark.parametrize(
    ("obstacle_pose", "expected"),
    [
        ((0.0, 1.0, 0.0), None),
        ((-0.2, 0.0, 0.0), _REJECTION_REASON),
        ((0.1, 0.0, 0.0), _REJECTION_REASON),
    ],
    ids=("clear", "frame-contact", "between-frame-contact"),
)
def test_collision_bundle_matches_equivalent_raw_full_stack_verdict(
    obstacle_pose: tuple[float, float, float],
    expected: str | None,
) -> None:
    base_state = _base_state()
    trajectory = _trajectory()
    footprint = CircleFootprint(0.02)
    bundle = _bundle(
        base_state=base_state,
        trajectory=trajectory,
        footprint=footprint,
    )
    raw = OccluderCollisionSweep(
        footprint=footprint,
        poses=np.vstack((base_state.robot_history, trajectory.poses)),
        rejection_reason=_REJECTION_REASON,
    )
    obstacle = CircleFootprint(0.02)

    prepared_verdict = occluder_collision_sweep_rejection_reason(
        obstacle, obstacle_pose, bundle.collision_sweeps, grid=_grid()
    )
    raw_verdict = occluder_collision_sweep_rejection_reason(
        obstacle, obstacle_pose, (raw,), grid=_grid()
    )

    assert prepared_verdict == raw_verdict == expected


def test_collision_bundle_is_immutable_and_rejects_component_splices() -> None:
    base_state = _base_state()
    trajectory = _trajectory()
    bundle = _bundle(base_state=base_state, trajectory=trajectory)
    other_base = _base_state(state_id="base-002")
    other_trajectory = _trajectory(trajectory_id="traj-002")
    other_bundle = _bundle(base_state=other_base, trajectory=other_trajectory)

    with pytest.raises(FrozenInstanceError):
        bundle.base_state_id = "base-002"  # type: ignore[misc]
    for sweep in bundle.collision_sweeps:
        assert not sweep.dense_poses.flags.writeable
        assert not sweep.interval_motion_bounds_m.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            sweep.dense_poses.setflags(write=True)

    for changes, message in (
        ({"base_state_id": "base-002"}, "canonical_digest"),
        ({"trajectory_id": "traj-002"}, "future entry|canonical_digest"),
        ({"robot_footprint": CircleFootprint(0.03)}, "future entry|canonical_digest"),
        ({"future_entry": other_bundle.future_entry}, "future entry"),
        ({"canonical_digest": "0" * 64}, "canonical_digest"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(bundle, **changes)

    with pytest.raises(ValueError, match="init=False"):
        replace(bundle, history_sweep=other_bundle.history_sweep)
    with pytest.raises(ValueError, match="init=False"):
        replace(bundle, seam_sweep=other_bundle.seam_sweep)
