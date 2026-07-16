"""Structural FOV, range, and blind-sector visibility constraints."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from src.contracts import GridSpec
from src.geometry import (
    Footprint,
    grid_cell_centers,
    rasterize_footprint,
    raycast_visibility,
    wrap_angle,
)


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class StructuralBlindSpot:
    """One deterministic structural sensor-visibility definition.

    Blind-sector centres and widths are expressed in degrees relative to the
    sensor's forward yaw. Sectors remove visibility after the canonical
    raycaster has applied map occlusion, forward FOV, and range.
    """

    forward_fov_deg: float
    range_m: float
    blind_sectors: tuple[dict[str, float], ...] = ()

    def __post_init__(self) -> None:
        forward_fov = _finite_real(
            self.forward_fov_deg, name="forward_fov_deg"
        )
        if not 0.0 < forward_fov <= 360.0:
            raise ValueError("forward_fov_deg must be in (0, 360]")
        sensor_range = _finite_real(self.range_m, name="range_m")
        if sensor_range <= 0.0:
            raise ValueError("range_m must be positive")
        if not isinstance(self.blind_sectors, tuple):
            raise TypeError("blind_sectors must be a tuple")
        normalized = []
        for index, sector in enumerate(self.blind_sectors):
            if not isinstance(sector, dict) or set(sector) != {
                "center_deg",
                "width_deg",
            }:
                raise ValueError(
                    "each blind sector must contain center_deg and width_deg"
                )
            center = _finite_real(
                sector["center_deg"], name=f"blind_sectors[{index}].center_deg"
            )
            width = _finite_real(
                sector["width_deg"], name=f"blind_sectors[{index}].width_deg"
            )
            if not 0.0 < width <= 360.0:
                raise ValueError("blind-sector width_deg must be in (0, 360]")
            center = (center + 180.0) % 360.0 - 180.0
            normalized.append({"center_deg": center, "width_deg": width})
        object.__setattr__(self, "forward_fov_deg", forward_fov)
        object.__setattr__(self, "range_m", sensor_range)
        object.__setattr__(self, "blind_sectors", tuple(normalized))

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-safe representation for ``OracleWorld``."""

        return {
            "forward_fov_deg": self.forward_fov_deg,
            "range_m": self.range_m,
            "blind_sectors": [dict(sector) for sector in self.blind_sectors],
        }


def _sensor_pose(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (3,) or array.dtype.kind not in "iuf":
        raise ValueError("sensor_pose must be a numeric array with shape (3,)")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("sensor_pose must contain only finite values")
    return result


def build_structural_visibility(
    occupancy: np.ndarray,
    grid: GridSpec,
    *,
    sensor_pose: Any,
    blind_spot: StructuralBlindSpot,
) -> np.ndarray:
    """Return the canonical raycast mask after structural-sector removal."""

    if not isinstance(blind_spot, StructuralBlindSpot):
        raise TypeError("blind_spot must be a StructuralBlindSpot")
    sensor = _sensor_pose(sensor_pose)
    visible = raycast_visibility(
        occupancy,
        grid,
        sensor_pose=sensor,
        fov_rad=np.deg2rad(blind_spot.forward_fov_deg),
        max_range_m=blind_spot.range_m,
    )
    if not blind_spot.blind_sectors:
        return visible

    centers = grid_cell_centers(grid)
    deltas = centers - sensor[:2]
    relative_bearings = wrap_angle(
        np.arctan2(deltas[..., 1], deltas[..., 0]) - sensor[2]
    )
    for sector in blind_spot.blind_sectors:
        center = np.deg2rad(sector["center_deg"])
        half_width = 0.5 * np.deg2rad(sector["width_deg"])
        inside = np.abs(wrap_angle(relative_bearings - center)) <= half_width
        visible[inside] = False
    return visible


def footprint_visibility_sequence(
    footprint: Footprint,
    poses: Any,
    visibility_mask: Any,
    grid: GridSpec,
) -> np.ndarray:
    """Return whether any cell touched by a footprint is visible per pose."""

    pose_array = np.asarray(poses)
    if pose_array.ndim != 2 or pose_array.shape[1:] != (3,):
        raise ValueError("poses must have shape (T, 3)")
    if pose_array.dtype.kind not in "iuf" or not np.isfinite(pose_array).all():
        raise ValueError("poses must contain only finite real values")
    mask = np.asarray(visibility_mask)
    if mask.shape != (grid.height, grid.width):
        raise ValueError("visibility_mask shape must match grid")
    if mask.dtype != np.bool_:
        raise TypeError("visibility_mask must have boolean dtype")
    return np.asarray(
        [
            bool(np.any(rasterize_footprint(footprint, pose, grid) & mask))
            for pose in pose_array
        ],
        dtype=bool,
    )


def has_continuous_emergence(
    visibility_sequence: Any, *, min_visible_frames: int = 2
) -> bool:
    """Check that a currently hidden object later has a visible run."""

    sequence = np.asarray(visibility_sequence)
    if sequence.ndim != 1 or sequence.dtype != np.bool_ or sequence.size == 0:
        raise ValueError("visibility_sequence must be a non-empty boolean vector")
    if isinstance(min_visible_frames, (bool, np.bool_)) or not isinstance(
        min_visible_frames, (int, np.integer)
    ):
        raise TypeError("min_visible_frames must be an integer")
    if min_visible_frames <= 0:
        raise ValueError("min_visible_frames must be positive")
    if bool(sequence[0]):
        return False
    run = 0
    for is_visible in sequence[1:]:
        run = run + 1 if bool(is_visible) else 0
        if run >= min_visible_frames:
            return True
    return False
