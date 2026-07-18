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

from src.contracts import GridSpec, LocalTrajectory
from src.geometry import CircleFootprint, Footprint, RectangleFootprint

from .occluder_sampler import (
    OCCLUDER_COLLISION_SWEEP_PREPARATION_VERSION,
    OccluderCollisionSweep,
    PreparedOccluderCollisionSweep,
    prepare_occluder_collision_sweep,
)


ROBOT_SWEEP_CACHE_VERSION = "robot_sweep_cache_v1"
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


@dataclass(frozen=True)
class RobotSweepCacheEntry:
    """Immutable persisted mask evidence plus prepared canonical future motion."""

    trajectory_id: str
    key: RobotSweepCacheKey
    swept_mask: np.ndarray
    prepared_future_sweep: PreparedOccluderCollisionSweep
    _swept_mask_storage: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        incoming_bytes = np.asarray(self.swept_mask).tobytes(order="C")
        if incoming_bytes != self.key.swept_mask_bytes:
            raise ValueError("swept_mask bytes must match the cache key")
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
