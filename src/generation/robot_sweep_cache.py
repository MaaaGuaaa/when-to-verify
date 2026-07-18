"""Strict process-local preparation cache for canonical robot future sweeps.

The persisted ``swept_mask`` is retained only as broad-phase evidence.  It is
never consulted for a physical collision verdict.  Entries contain only the
canonical trajectory future; robot history and context motion are base-specific
and callers must prepare them separately rather than aliasing them here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from numbers import Real
import struct
from typing import Any, Mapping

import numpy as np

from src.contracts import BaseState, GridSpec, LocalTrajectory, validate_base_state
from src.geometry import CircleFootprint, Footprint, RectangleFootprint, wrap_angle

from .occluder_sampler import (
    OCCLUDER_COLLISION_SWEEP_PREPARATION_VERSION,
    OccluderCollisionSweep,
    PreparedOccluderCollisionSweep,
    prepare_occluder_collision_sweep,
)


ROBOT_SWEEP_CACHE_VERSION = "robot_sweep_cache_v1"
ROBOT_COLLISION_SWEEP_BUNDLE_VERSION = "robot_collision_sweep_bundle_v1"
SOP04_POSE_TIME_LAYOUT_VERSION = "future_endpoints_dt_to_horizon_v1"


class RobotSweepCacheIdentityError(ValueError):
    """Raised when one trajectory ID is reused with different bound content."""


@dataclass(frozen=True)
class RobotSweepCacheKey:
    """Complete scientific binding for one canonical future-sweep entry."""

    trajectory_id: str
    pose_bytes: bytes
    swept_mask_bytes: bytes
    footprint: Footprint
    grid: GridSpec
    future_dt_s: float
    pose_time_layout_version: str
    pose_time_offsets_bytes: bytes
    cache_version: str
    preparation_version: str
    rejection_reason: str
    canonical_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory_id, str) or not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty")
        grid = _validate_grid(self.grid)
        if not isinstance(self.footprint, (CircleFootprint, RectangleFootprint)):
            raise TypeError("footprint must be a Footprint")
        future_dt_s = _finite_float(
            self.future_dt_s,
            name="future_dt_s",
            positive=True,
        )
        if self.pose_time_layout_version != SOP04_POSE_TIME_LAYOUT_VERSION:
            raise ValueError("pose_time_layout_version is not the frozen v1 layout")
        if self.cache_version != ROBOT_SWEEP_CACHE_VERSION:
            raise ValueError("cache_version mismatch")
        if (
            self.preparation_version
            != OCCLUDER_COLLISION_SWEEP_PREPARATION_VERSION
        ):
            raise ValueError("preparation_version mismatch")
        if not isinstance(self.rejection_reason, str) or not self.rejection_reason:
            raise ValueError("rejection_reason must be non-empty")

        if not isinstance(self.pose_bytes, bytes):
            raise TypeError("pose_bytes must be canonical bytes")
        expected_pose_nbytes = grid.future_steps * 3 * np.dtype("<f8").itemsize
        if len(self.pose_bytes) != expected_pose_nbytes:
            raise ValueError("pose_bytes length does not match grid.future_steps")
        poses = np.frombuffer(self.pose_bytes, dtype=np.dtype("<f8")).reshape(
            (grid.future_steps, 3)
        )
        if not np.isfinite(poses).all():
            raise ValueError("pose_bytes must encode only finite float64 poses")

        if not isinstance(self.swept_mask_bytes, bytes):
            raise TypeError("swept_mask_bytes must be canonical bytes")
        expected_mask_nbytes = (
            grid.height * grid.width * np.dtype("<f4").itemsize
        )
        if len(self.swept_mask_bytes) != expected_mask_nbytes:
            raise ValueError("swept_mask_bytes length does not match grid shape")
        swept_mask = np.frombuffer(
            self.swept_mask_bytes,
            dtype=np.dtype("<f4"),
        ).reshape((grid.height, grid.width))
        if not np.isfinite(swept_mask).all():
            raise ValueError("swept_mask_bytes must encode finite float32 values")
        if not np.all((swept_mask == 0.0) | (swept_mask == 1.0)):
            raise ValueError("swept_mask_bytes must encode a binary mask")

        if not isinstance(self.pose_time_offsets_bytes, bytes):
            raise TypeError("pose_time_offsets_bytes must be canonical bytes")
        expected_offsets_nbytes = grid.future_steps * np.dtype("<f8").itemsize
        if len(self.pose_time_offsets_bytes) != expected_offsets_nbytes:
            raise ValueError(
                "pose_time_offsets_bytes length does not match grid.future_steps"
            )
        offsets = np.frombuffer(
            self.pose_time_offsets_bytes,
            dtype=np.dtype("<f8"),
        )
        if not np.isfinite(offsets).all():
            raise ValueError("pose_time_offsets_bytes must encode finite values")
        expected_offsets = (
            np.arange(grid.future_steps, dtype=np.float64) + np.float64(1.0)
        ) * np.float64(future_dt_s)
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError("pose_time_offsets_bytes mismatch future_dt_s")

        expected_digest = _canonical_key_digest(
            trajectory_id=self.trajectory_id,
            pose_bytes=self.pose_bytes,
            swept_mask_bytes=self.swept_mask_bytes,
            footprint=self.footprint,
            grid=grid,
            future_dt_s=future_dt_s,
            layout_version=self.pose_time_layout_version,
            offsets_bytes=self.pose_time_offsets_bytes,
            rejection_reason=self.rejection_reason,
        )
        if self.canonical_digest != expected_digest:
            raise ValueError("canonical_digest does not match bound key fields")


@dataclass(frozen=True)
class RobotSweepCacheEntry:
    """Immutable persisted mask evidence plus prepared canonical future motion."""

    trajectory_id: str
    key: RobotSweepCacheKey
    swept_mask: np.ndarray
    prepared_future_sweep: PreparedOccluderCollisionSweep
    _swept_mask_storage: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.key, RobotSweepCacheKey):
            raise TypeError("key must be a RobotSweepCacheKey")
        if self.trajectory_id != self.key.trajectory_id:
            raise ValueError("entry trajectory_id must match cache key trajectory_id")
        incoming_bytes = np.asarray(self.swept_mask).tobytes(order="C")
        if incoming_bytes != self.key.swept_mask_bytes:
            raise ValueError("swept_mask bytes must match the cache key")
        prepared = self.prepared_future_sweep
        if not isinstance(prepared, PreparedOccluderCollisionSweep):
            raise TypeError(
                "prepared_future_sweep must be a PreparedOccluderCollisionSweep"
            )
        prepared_binding_matches = (
            prepared.footprint == self.key.footprint
            and prepared.grid == self.key.grid
            and prepared.rejection_reason == self.key.rejection_reason
            and prepared.preparation_version == self.key.preparation_version
        )
        raw_poses = np.frombuffer(
            self.key.pose_bytes,
            dtype=np.dtype("<f8"),
        ).reshape((self.key.grid.future_steps, 3))
        expected_prepared = prepare_occluder_collision_sweep(
            OccluderCollisionSweep(
                footprint=self.key.footprint,
                poses=raw_poses,
                rejection_reason=self.key.rejection_reason,
            ),
            grid=self.key.grid,
        )
        prepared_geometry_matches = (
            prepared.dense_poses.shape == expected_prepared.dense_poses.shape
            and prepared.dense_poses.tobytes(order="C")
            == expected_prepared.dense_poses.tobytes(order="C")
            and prepared.interval_motion_bounds_m.shape
            == expected_prepared.interval_motion_bounds_m.shape
            and prepared.interval_motion_bounds_m.tobytes(order="C")
            == expected_prepared.interval_motion_bounds_m.tobytes(order="C")
        )
        if not prepared_binding_matches or not prepared_geometry_matches:
            raise ValueError(
                "prepared future sweep does not match cache key canonical preparation"
            )
        storage = self.key.swept_mask_bytes
        immutable_mask = np.frombuffer(
            storage,
            dtype=np.dtype("<f4"),
        ).reshape((self.key.grid.height, self.key.grid.width))
        object.__setattr__(self, "_swept_mask_storage", storage)
        object.__setattr__(self, "swept_mask", immutable_mask)


@dataclass(frozen=True)
class RobotSweepCacheStats:
    """Inspectable process-local cache counters."""

    size: int
    hits: int
    misses: int
    builds: int


def _finite_float(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Real, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validate_grid(grid: Any) -> GridSpec:
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    for field_name in (
        "height",
        "width",
        "history_steps",
        "future_steps",
        "n_history_channels",
        "n_state_channels",
        "n_trajectory_channels",
    ):
        _positive_integer(getattr(grid, field_name), name=f"grid.{field_name}")
    _finite_float(grid.resolution_m, name="grid.resolution_m", positive=True)
    return grid


def _canonical_poses(value: Any, *, grid: GridSpec) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("trajectory.poses must be a numeric array") from exc
    expected_shape = (grid.future_steps, 3)
    if array.shape != expected_shape:
        raise ValueError(f"trajectory.poses must have shape {expected_shape}")
    if array.dtype.kind not in "iuf":
        raise TypeError("trajectory.poses must contain real numbers")
    result = np.array(array, dtype=np.dtype("<f8"), order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("trajectory.poses must contain only finite values")
    result.setflags(write=False)
    return result


def _canonical_swept_mask(value: Any, *, grid: GridSpec) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("trajectory.swept_mask must be an array") from exc
    expected_shape = (grid.height, grid.width)
    if array.shape != expected_shape:
        raise ValueError(f"trajectory.swept_mask must have shape {expected_shape}")
    if array.dtype != np.dtype(np.float32):
        raise TypeError("trajectory.swept_mask must have dtype float32")
    result = np.array(array, dtype=np.dtype("<f4"), order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("trajectory.swept_mask must contain only finite values")
    if not np.all((result == 0.0) | (result == 1.0)):
        raise ValueError("trajectory.swept_mask must be binary")
    result.setflags(write=False)
    return result


def _canonical_time_offsets(
    metadata: Any,
    *,
    grid: GridSpec,
    future_dt_s: float,
) -> tuple[str, np.ndarray]:
    if not isinstance(metadata, Mapping):
        raise TypeError("trajectory.metadata must be a mapping")
    layout_version = metadata.get("pose_time_layout_version")
    if layout_version != SOP04_POSE_TIME_LAYOUT_VERSION:
        raise ValueError(
            "trajectory pose-time layout version must be "
            f"{SOP04_POSE_TIME_LAYOUT_VERSION}"
        )
    if "pose_time_offsets_s" not in metadata:
        raise ValueError("trajectory pose-time offsets are missing")
    try:
        offsets = np.asarray(metadata["pose_time_offsets_s"])
    except (TypeError, ValueError) as exc:
        raise TypeError("trajectory pose-time offsets must be numeric") from exc
    if offsets.shape != (grid.future_steps,):
        raise ValueError(
            "trajectory pose-time offsets must have shape "
            f"({grid.future_steps},)"
        )
    if offsets.dtype.kind not in "iuf":
        raise TypeError("trajectory pose-time offsets must contain real numbers")
    canonical = np.array(offsets, dtype=np.dtype("<f8"), order="C", copy=True)
    if not np.isfinite(canonical).all():
        raise ValueError("trajectory pose-time offsets must contain finite values")
    expected = (
        np.arange(grid.future_steps, dtype=np.float64) + np.float64(1.0)
    ) * np.float64(future_dt_s)
    if not np.array_equal(canonical, expected):
        raise ValueError("trajectory pose-time offsets mismatch future_dt_s")
    canonical.setflags(write=False)
    return layout_version, canonical


def _digest_part(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _footprint_bytes(footprint: Footprint) -> bytes:
    if isinstance(footprint, CircleFootprint):
        return b"circle\0" + struct.pack(">d", footprint.radius_m)
    return b"rectangle\0" + struct.pack(
        ">dd",
        footprint.length_m,
        footprint.width_m,
    )


def _grid_bytes(grid: GridSpec) -> bytes:
    return struct.pack(
        ">qqqqdqqq",
        grid.height,
        grid.width,
        grid.history_steps,
        grid.future_steps,
        grid.resolution_m,
        grid.n_history_channels,
        grid.n_state_channels,
        grid.n_trajectory_channels,
    )


def _canonical_key_digest(
    *,
    trajectory_id: str,
    pose_bytes: bytes,
    swept_mask_bytes: bytes,
    footprint: Footprint,
    grid: GridSpec,
    future_dt_s: float,
    layout_version: str,
    offsets_bytes: bytes,
    rejection_reason: str,
) -> str:
    digest = hashlib.sha256()
    for part in (
        ROBOT_SWEEP_CACHE_VERSION.encode("utf-8"),
        OCCLUDER_COLLISION_SWEEP_PREPARATION_VERSION.encode("utf-8"),
        trajectory_id.encode("utf-8"),
        pose_bytes,
        swept_mask_bytes,
        _footprint_bytes(footprint),
        _grid_bytes(grid),
        struct.pack(">d", future_dt_s),
        layout_version.encode("utf-8"),
        offsets_bytes,
        rejection_reason.encode("utf-8"),
    ):
        _digest_part(digest, part)
    return digest.hexdigest()


class RobotSweepCache:
    """Explicit immutable cache for canonical futures within one worker process.

    No global instance is created.  ``get`` never uses the persisted mask as a
    collision decision and never accepts base history or context motion.
    """

    def __init__(self) -> None:
        self._entries_by_digest: dict[str, RobotSweepCacheEntry] = {}
        self._keys_by_trajectory_id: dict[str, RobotSweepCacheKey] = {}
        self._hits = 0
        self._misses = 0
        self._builds = 0

    @property
    def stats(self) -> RobotSweepCacheStats:
        return RobotSweepCacheStats(
            size=len(self._entries_by_digest),
            hits=self._hits,
            misses=self._misses,
            builds=self._builds,
        )

    def get(
        self,
        trajectory: LocalTrajectory,
        *,
        robot_footprint: Footprint,
        grid: GridSpec,
        future_dt_s: float,
        rejection_reason: str = "occluder_robot_swept_overlap",
    ) -> RobotSweepCacheEntry:
        """Return one strict-key entry, building its future geometry once."""

        if not isinstance(trajectory, LocalTrajectory):
            raise TypeError("trajectory must be a LocalTrajectory")
        if not isinstance(trajectory.trajectory_id, str) or not trajectory.trajectory_id:
            raise ValueError("trajectory.trajectory_id must be non-empty")
        if not isinstance(robot_footprint, (CircleFootprint, RectangleFootprint)):
            raise TypeError("robot_footprint must be a Footprint")
        grid = _validate_grid(grid)
        dt_s = _finite_float(future_dt_s, name="future_dt_s", positive=True)
        if not isinstance(rejection_reason, str) or not rejection_reason:
            raise ValueError("rejection_reason must be non-empty")

        poses = _canonical_poses(trajectory.poses, grid=grid)
        swept_mask = _canonical_swept_mask(trajectory.swept_mask, grid=grid)
        layout_version, offsets = _canonical_time_offsets(
            trajectory.metadata,
            grid=grid,
            future_dt_s=dt_s,
        )
        pose_bytes = poses.tobytes(order="C")
        mask_bytes = swept_mask.tobytes(order="C")
        offsets_bytes = offsets.tobytes(order="C")
        canonical_digest = _canonical_key_digest(
            trajectory_id=trajectory.trajectory_id,
            pose_bytes=pose_bytes,
            swept_mask_bytes=mask_bytes,
            footprint=robot_footprint,
            grid=grid,
            future_dt_s=dt_s,
            layout_version=layout_version,
            offsets_bytes=offsets_bytes,
            rejection_reason=rejection_reason,
        )
        key = RobotSweepCacheKey(
            trajectory_id=trajectory.trajectory_id,
            pose_bytes=pose_bytes,
            swept_mask_bytes=mask_bytes,
            footprint=robot_footprint,
            grid=grid,
            future_dt_s=dt_s,
            pose_time_layout_version=layout_version,
            pose_time_offsets_bytes=offsets_bytes,
            cache_version=ROBOT_SWEEP_CACHE_VERSION,
            preparation_version=OCCLUDER_COLLISION_SWEEP_PREPARATION_VERSION,
            rejection_reason=rejection_reason,
            canonical_digest=canonical_digest,
        )

        previous_key = self._keys_by_trajectory_id.get(trajectory.trajectory_id)
        if previous_key is not None and previous_key != key:
            raise RobotSweepCacheIdentityError(
                f"trajectory_id {trajectory.trajectory_id!r} binding changed "
                "within one RobotSweepCache instance"
            )
        existing = self._entries_by_digest.get(canonical_digest)
        if existing is not None:
            if existing.key != key:
                raise RuntimeError("robot sweep cache canonical digest collision")
            self._hits += 1
            return existing

        self._misses += 1
        prepared = prepare_occluder_collision_sweep(
            OccluderCollisionSweep(
                footprint=robot_footprint,
                poses=poses,
                rejection_reason=rejection_reason,
            ),
            grid=grid,
        )
        entry = RobotSweepCacheEntry(
            trajectory_id=trajectory.trajectory_id,
            key=key,
            swept_mask=swept_mask,
            prepared_future_sweep=prepared,
        )
        self._entries_by_digest[canonical_digest] = entry
        self._keys_by_trajectory_id[trajectory.trajectory_id] = key
        self._builds += 1
        return entry


def _canonical_bundle_digest(
    *,
    base_state_id: str,
    trajectory_id: str,
    base_history_pose_bytes: bytes,
    future_entry_digest: str,
    robot_footprint: Footprint,
    grid: GridSpec,
    future_dt_s: float,
    rejection_reason: str,
) -> str:
    digest = hashlib.sha256()
    for part in (
        ROBOT_COLLISION_SWEEP_BUNDLE_VERSION.encode("utf-8"),
        base_state_id.encode("utf-8"),
        trajectory_id.encode("utf-8"),
        base_history_pose_bytes,
        future_entry_digest.encode("ascii"),
        _footprint_bytes(robot_footprint),
        _grid_bytes(grid),
        struct.pack(">d", future_dt_s),
        rejection_reason.encode("utf-8"),
    ):
        _digest_part(digest, part)
    return digest.hexdigest()


def _same_se2_endpoint(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(
        np.array_equal(first[:2], second[:2])
        and np.isclose(
            float(wrap_angle(first[2] - second[2])),
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
    )


@dataclass(frozen=True)
class RobotCollisionSweepBundle:
    """Immutable no-gap robot sweep ordered as history, seam, and future.

    The canonical future is owned by ``future_entry``.  Base history and the
    current-to-first-future seam are derived afresh from immutable canonical
    bytes, so no base-specific geometry enters the process-local cache.
    """

    base_state_id: str
    trajectory_id: str
    robot_footprint: Footprint
    grid: GridSpec
    future_dt_s: float
    rejection_reason: str
    base_history_pose_bytes: bytes
    future_entry: RobotSweepCacheEntry
    canonical_digest: str
    bundle_version: str = ROBOT_COLLISION_SWEEP_BUNDLE_VERSION
    history_sweep: PreparedOccluderCollisionSweep = field(init=False)
    seam_sweep: PreparedOccluderCollisionSweep = field(init=False)
    collision_sweeps: tuple[PreparedOccluderCollisionSweep, ...] = field(
        init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.base_state_id, str) or not self.base_state_id:
            raise ValueError("base_state_id must be non-empty")
        if not isinstance(self.trajectory_id, str) or not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty")
        if not isinstance(
            self.robot_footprint,
            (CircleFootprint, RectangleFootprint),
        ):
            raise TypeError("robot_footprint must be a Footprint")
        grid = _validate_grid(self.grid)
        future_dt_s = _finite_float(
            self.future_dt_s,
            name="future_dt_s",
            positive=True,
        )
        if not isinstance(self.rejection_reason, str) or not self.rejection_reason:
            raise ValueError("rejection_reason must be non-empty")
        if self.bundle_version != ROBOT_COLLISION_SWEEP_BUNDLE_VERSION:
            raise ValueError("bundle_version mismatch")
        if not isinstance(self.base_history_pose_bytes, bytes):
            raise TypeError("base_history_pose_bytes must be canonical bytes")
        expected_history_nbytes = (
            grid.history_steps * 3 * np.dtype("<f8").itemsize
        )
        if len(self.base_history_pose_bytes) != expected_history_nbytes:
            raise ValueError(
                "base_history_pose_bytes length does not match grid.history_steps"
            )
        history_poses = np.frombuffer(
            self.base_history_pose_bytes,
            dtype=np.dtype("<f8"),
        ).reshape((grid.history_steps, 3))
        if not np.isfinite(history_poses).all():
            raise ValueError("base_history_pose_bytes must encode finite poses")
        if not np.allclose(
            history_poses[-1],
            0.0,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("base_state current robot pose must be the local origin")

        entry = self.future_entry
        if not isinstance(entry, RobotSweepCacheEntry):
            raise TypeError("future_entry must be a RobotSweepCacheEntry")
        key = entry.key
        if (
            entry.trajectory_id != self.trajectory_id
            or key.trajectory_id != self.trajectory_id
            or key.footprint != self.robot_footprint
            or key.grid != grid
            or key.future_dt_s != future_dt_s
            or key.rejection_reason != self.rejection_reason
            or key.pose_time_layout_version != SOP04_POSE_TIME_LAYOUT_VERSION
        ):
            raise ValueError("future entry does not match bundle components")

        expected_digest = _canonical_bundle_digest(
            base_state_id=self.base_state_id,
            trajectory_id=self.trajectory_id,
            base_history_pose_bytes=self.base_history_pose_bytes,
            future_entry_digest=key.canonical_digest,
            robot_footprint=self.robot_footprint,
            grid=grid,
            future_dt_s=future_dt_s,
            rejection_reason=self.rejection_reason,
        )
        if self.canonical_digest != expected_digest:
            raise ValueError("canonical_digest does not match bound bundle fields")

        future_poses = np.frombuffer(
            key.pose_bytes,
            dtype=np.dtype("<f8"),
        ).reshape((grid.future_steps, 3))
        history_sweep = prepare_occluder_collision_sweep(
            OccluderCollisionSweep(
                footprint=self.robot_footprint,
                poses=history_poses,
                rejection_reason=self.rejection_reason,
            ),
            grid=grid,
        )
        seam_poses = np.vstack((history_poses[-1], future_poses[0]))
        seam_sweep = prepare_occluder_collision_sweep(
            OccluderCollisionSweep(
                footprint=self.robot_footprint,
                poses=seam_poses,
                rejection_reason=self.rejection_reason,
            ),
            grid=grid,
        )
        future_sweep = entry.prepared_future_sweep
        if not (
            _same_se2_endpoint(
                history_sweep.dense_poses[-1],
                seam_sweep.dense_poses[0],
            )
            and _same_se2_endpoint(
                seam_sweep.dense_poses[-1],
                future_sweep.dense_poses[0],
            )
        ):
            raise ValueError("robot collision sweep endpoint seam mismatch")
        object.__setattr__(self, "history_sweep", history_sweep)
        object.__setattr__(self, "seam_sweep", seam_sweep)
        object.__setattr__(
            self,
            "collision_sweeps",
            (history_sweep, seam_sweep, future_sweep),
        )

    @property
    def future_sweep(self) -> PreparedOccluderCollisionSweep:
        """Return the cache-owned canonical future sweep."""

        return self.future_entry.prepared_future_sweep


def prepare_robot_collision_sweep_bundle(
    base_state: BaseState,
    trajectory: LocalTrajectory,
    *,
    robot_footprint: Footprint,
    grid: GridSpec,
    future_dt_s: float,
    cache: RobotSweepCache | None = None,
    rejection_reason: str = "occluder_robot_swept_overlap",
) -> RobotCollisionSweepBundle:
    """Prepare exact history/seam/future robot sweeps without time gaps."""

    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(trajectory, LocalTrajectory):
        raise TypeError("trajectory must be a LocalTrajectory")
    if not isinstance(robot_footprint, (CircleFootprint, RectangleFootprint)):
        raise TypeError("robot_footprint must be a Footprint")
    grid = _validate_grid(grid)
    future_dt_s = _finite_float(
        future_dt_s,
        name="future_dt_s",
        positive=True,
    )
    if cache is not None and not isinstance(cache, RobotSweepCache):
        raise TypeError("cache must be a RobotSweepCache or None")
    if not isinstance(rejection_reason, str) or not rejection_reason:
        raise ValueError("rejection_reason must be non-empty")
    if not isinstance(base_state.state_id, str) or not base_state.state_id:
        raise ValueError("base_state.state_id must be non-empty")
    validate_base_state(base_state, grid)
    if not np.allclose(
        base_state.robot_history[-1],
        0.0,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("base_state current robot pose must be the local origin")

    base_history = np.array(
        base_state.robot_history,
        dtype=np.dtype("<f8"),
        order="C",
        copy=True,
    )
    base_history_pose_bytes = base_history.tobytes(order="C")
    active_cache = RobotSweepCache() if cache is None else cache
    future_entry = active_cache.get(
        trajectory,
        robot_footprint=robot_footprint,
        grid=grid,
        future_dt_s=future_dt_s,
        rejection_reason=rejection_reason,
    )
    canonical_digest = _canonical_bundle_digest(
        base_state_id=base_state.state_id,
        trajectory_id=trajectory.trajectory_id,
        base_history_pose_bytes=base_history_pose_bytes,
        future_entry_digest=future_entry.key.canonical_digest,
        robot_footprint=robot_footprint,
        grid=grid,
        future_dt_s=future_dt_s,
        rejection_reason=rejection_reason,
    )
    return RobotCollisionSweepBundle(
        base_state_id=base_state.state_id,
        trajectory_id=trajectory.trajectory_id,
        robot_footprint=robot_footprint,
        grid=grid,
        future_dt_s=future_dt_s,
        rejection_reason=rejection_reason,
        base_history_pose_bytes=base_history_pose_bytes,
        future_entry=future_entry,
        canonical_digest=canonical_digest,
    )
