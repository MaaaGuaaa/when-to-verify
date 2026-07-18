"""Current-causal, renderer-identical blind-region construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from numbers import Integral
import struct
from typing import Any

import numpy as np

from src.contracts import ARRAY_DTYPE, BaseState, GridSpec, validate_base_state
from src.geometry import (
    RectangleFootprint,
    footprint_aabb,
    grid_bounds,
    rasterize_footprint,
    raycast_visibility,
)
from src.generation.causal_occluder import (
    CAUSAL_OCCLUDER_PROPOSAL_VERSION,
    CAUSAL_OCCLUDER_SCHEDULE_VERSION,
)
from src.generation.dynamic_object_transplant import footprint_from_spec
from src.generation.occluder_sampler import OccluderGeometryCandidate


BLIND_REGION_VERSION = "blind_region_v1"

_ENVIRONMENT_FOV_RAD = 2.0 * np.pi
_ENVIRONMENT_MAX_RANGE_M = None
_CAUSAL_OCCLUDER_TYPES = frozenset(("wall", "shelf", "pillar"))


def _digest_parts(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, byteorder="big", signed=False))
        digest.update(part)
    return digest.hexdigest()


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


def _array_digest(name: str, values: np.ndarray) -> str:
    shape = struct.pack(">q" + "q" * values.ndim, values.ndim, *values.shape)
    return _digest_parts(
        name.encode("ascii"),
        values.dtype.str.encode("ascii"),
        shape,
        values.tobytes(order="C"),
    )


def _immutable_array(values: np.ndarray) -> tuple[bytes, np.ndarray]:
    contiguous = np.ascontiguousarray(values)
    storage = contiguous.tobytes(order="C")
    immutable = np.frombuffer(storage, dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    return storage, immutable


def _canonical_bool_grid(value: Any, *, name: str, grid: GridSpec) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an np.ndarray")
    if value.shape != (grid.height, grid.width):
        raise ValueError(f"{name} shape must match grid")
    if value.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must have bool dtype")
    return np.array(value, dtype=np.bool_, order="C", copy=True)


def _canonical_pose(value: Any, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an np.ndarray")
    if value.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if value.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have canonical float64 dtype")
    pose = np.array(value, dtype=np.dtype("<f8"), order="C", copy=True)
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} must contain only finite values")
    return pose


def _inside_grid(
    footprint: RectangleFootprint,
    pose: np.ndarray,
    grid: GridSpec,
) -> bool:
    x_min, x_max, y_min, y_max = footprint_aabb(footprint, pose)
    grid_x_min, grid_x_max, grid_y_min, grid_y_max = grid_bounds(grid)
    return bool(
        x_min >= grid_x_min
        and x_max < grid_x_max
        and y_min >= grid_y_min
        and y_max < grid_y_max
    )


def _validate_causal_occluder(
    causal_occluder: Any,
    *,
    base_state_id: str,
    grid: GridSpec,
) -> tuple[str, RectangleFootprint, np.ndarray, np.ndarray]:
    if not isinstance(causal_occluder, OccluderGeometryCandidate):
        raise TypeError("causal_occluder must be an OccluderGeometryCandidate")
    if not isinstance(causal_occluder.footprint, RectangleFootprint):
        raise TypeError("causal_occluder footprint must be a RectangleFootprint")
    footprint = causal_occluder.footprint
    pose = _canonical_pose(causal_occluder.pose, name="causal_occluder.pose")
    if not _inside_grid(footprint, pose, grid):
        raise ValueError("causal_occluder footprint must lie fully inside grid")
    mask = _canonical_bool_grid(
        causal_occluder.mask,
        name="causal_occluder.mask",
        grid=grid,
    )
    expected_mask = rasterize_footprint(footprint, pose, grid)
    if not np.array_equal(mask, expected_mask):
        raise ValueError("causal_occluder.mask must equal rasterized geometry")

    if isinstance(causal_occluder.proposal_index, (bool, np.bool_)) or not isinstance(
        causal_occluder.proposal_index,
        (Integral, np.integer),
    ):
        raise TypeError("causal_occluder proposal_index must be an integer")
    proposal_index = int(causal_occluder.proposal_index)
    if proposal_index < 0:
        raise ValueError("causal_occluder proposal_index must be non-negative")

    metadata = causal_occluder.occluder
    if not isinstance(metadata, Mapping):
        raise TypeError("causal_occluder metadata must be a mapping")
    proposal_id = metadata.get("proposal_id")
    occluder_id = metadata.get("occluder_id")
    if (
        not isinstance(proposal_id, str)
        or not proposal_id
        or proposal_id != occluder_id
    ):
        raise ValueError("causal_occluder proposal_id and occluder_id must match")
    if metadata.get("base_state_id") != base_state_id:
        raise ValueError("causal_occluder base_state_id must match BaseState")
    if metadata.get("schedule_version") != CAUSAL_OCCLUDER_SCHEDULE_VERSION:
        raise ValueError("causal_occluder schedule_version does not match")
    if metadata.get("proposal_version") != CAUSAL_OCCLUDER_PROPOSAL_VERSION:
        raise ValueError("causal_occluder proposal_version does not match")
    if metadata.get("type") not in _CAUSAL_OCCLUDER_TYPES:
        raise ValueError("causal_occluder occluder type is not supported")
    if metadata.get("geometry_source") != "generator_config":
        raise ValueError("causal_occluder geometry_source does not match")
    if metadata.get("placement_strategy") != "causal_free_space_schedule_v1":
        raise ValueError("causal_occluder placement_strategy does not match")
    metadata_index = metadata.get("proposal_index")
    if (
        isinstance(metadata_index, (bool, np.bool_))
        or not isinstance(metadata_index, (Integral, np.integer))
        or int(metadata_index) != proposal_index
    ):
        raise ValueError("causal_occluder proposal_index metadata does not match")

    metadata_pose = metadata.get("pose")
    if (
        not isinstance(metadata_pose, tuple)
        or len(metadata_pose) != 3
        or any(type(value) is not float for value in metadata_pose)
    ):
        raise ValueError("causal_occluder metadata pose must be canonical")
    metadata_pose_bytes = np.asarray(
        metadata_pose,
        dtype=np.dtype(">f8"),
    ).tobytes(order="C")
    expected_pose_bytes = np.asarray(pose, dtype=np.dtype(">f8")).tobytes(
        order="C"
    )
    if metadata_pose_bytes != expected_pose_bytes:
        raise ValueError("causal_occluder metadata pose must match geometry")
    metadata_length = metadata.get("length_m")
    metadata_width = metadata.get("width_m")
    if type(metadata_length) is not float or type(metadata_width) is not float:
        raise ValueError("causal_occluder metadata dimensions must be canonical")
    if struct.pack(">dd", metadata_length, metadata_width) != struct.pack(
        ">dd",
        footprint.length_m,
        footprint.width_m,
    ):
        raise ValueError("causal_occluder metadata dimensions must match footprint")
    return proposal_id, footprint, pose, mask


@dataclass(frozen=True)
class BlindRegion:
    """Immutable current-frame occupancy and renderer-derived blind evidence."""

    base_state_id: str
    causal_occluder_id: str
    grid: GridSpec
    sensor_pose: np.ndarray
    static_occupancy: np.ndarray
    current_context_occupancy: np.ndarray
    causal_occluder_mask: np.ndarray
    version: str = field(init=False, default=BLIND_REGION_VERSION)
    total_current_occupancy: np.ndarray = field(init=False, repr=False)
    visibility_mask: np.ndarray = field(init=False, repr=False)
    raw_unobservable_mask: np.ndarray = field(init=False, repr=False)
    blind_free_mask: np.ndarray = field(init=False, repr=False)
    static_occupancy_digest: str = field(init=False)
    current_context_occupancy_digest: str = field(init=False)
    causal_occluder_mask_digest: str = field(init=False)
    total_current_occupancy_digest: str = field(init=False)
    visibility_digest: str = field(init=False)
    raw_unobservable_digest: str = field(init=False)
    blind_free_digest: str = field(init=False)
    blind_free_count: int = field(init=False)
    region_digest: str = field(init=False)
    _array_storage: tuple[bytes, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.base_state_id, str) or not self.base_state_id:
            raise ValueError("base_state_id must be a non-empty string")
        if not isinstance(self.causal_occluder_id, str) or not self.causal_occluder_id:
            raise ValueError("causal_occluder_id must be a non-empty string")
        if not isinstance(self.grid, GridSpec):
            raise TypeError("grid must be a GridSpec")
        grid_bounds(self.grid)
        sensor_pose = _canonical_pose(self.sensor_pose, name="sensor_pose")
        static = _canonical_bool_grid(
            self.static_occupancy,
            name="static_occupancy",
            grid=self.grid,
        )
        context = _canonical_bool_grid(
            self.current_context_occupancy,
            name="current_context_occupancy",
            grid=self.grid,
        )
        causal = _canonical_bool_grid(
            self.causal_occluder_mask,
            name="causal_occluder_mask",
            grid=self.grid,
        )
        total = np.asarray(static | context | causal, dtype=np.bool_, order="C")
        visibility = np.asarray(
            raycast_visibility(
                total,
                self.grid,
                sensor_pose=sensor_pose,
                fov_rad=_ENVIRONMENT_FOV_RAD,
                max_range_m=_ENVIRONMENT_MAX_RANGE_M,
            ),
            dtype=np.bool_,
            order="C",
        )
        if visibility.shape != total.shape:
            raise ValueError("raycast_visibility returned an invalid shape")
        raw_unobservable = np.asarray(~visibility, dtype=np.bool_, order="C")
        blind_free = np.asarray(
            raw_unobservable & ~total,
            dtype=np.bool_,
            order="C",
        )

        arrays = {
            "sensor_pose": sensor_pose,
            "static_occupancy": static,
            "current_context_occupancy": context,
            "causal_occluder_mask": causal,
            "total_current_occupancy": total,
            "visibility_mask": visibility,
            "raw_unobservable_mask": raw_unobservable,
            "blind_free_mask": blind_free,
        }
        digests = {
            name: _array_digest(name, values) for name, values in arrays.items()
        }
        canonical_identity = json.dumps(
            {
                "version": BLIND_REGION_VERSION,
                "base_state_id": self.base_state_id,
                "causal_occluder_id": self.causal_occluder_id,
                "fov_rad": _ENVIRONMENT_FOV_RAD,
                "max_range_m": _ENVIRONMENT_MAX_RANGE_M,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        region_digest = _digest_parts(
            canonical_identity,
            _grid_bytes(self.grid),
            *(digests[name].encode("ascii") for name in arrays),
        )

        storages: list[bytes] = []
        for name, values in arrays.items():
            storage, immutable = _immutable_array(values)
            storages.append(storage)
            object.__setattr__(self, name, immutable)
        object.__setattr__(
            self,
            "static_occupancy_digest",
            digests["static_occupancy"],
        )
        object.__setattr__(
            self,
            "current_context_occupancy_digest",
            digests["current_context_occupancy"],
        )
        object.__setattr__(
            self,
            "causal_occluder_mask_digest",
            digests["causal_occluder_mask"],
        )
        object.__setattr__(
            self,
            "total_current_occupancy_digest",
            digests["total_current_occupancy"],
        )
        object.__setattr__(self, "visibility_digest", digests["visibility_mask"])
        object.__setattr__(
            self,
            "raw_unobservable_digest",
            digests["raw_unobservable_mask"],
        )
        object.__setattr__(
            self,
            "blind_free_digest",
            digests["blind_free_mask"],
        )
        object.__setattr__(
            self,
            "blind_free_count",
            int(np.count_nonzero(blind_free)),
        )
        object.__setattr__(self, "region_digest", region_digest)
        object.__setattr__(self, "_array_storage", tuple(storages))


def build_blind_region(
    base_state: BaseState,
    causal_occluder: OccluderGeometryCandidate,
    *,
    grid: GridSpec,
) -> BlindRegion:
    """Build current-frame blind evidence using formal renderer semantics."""

    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    grid_bounds(grid)
    validate_base_state(base_state, grid)
    if not isinstance(base_state.state_id, str) or not base_state.state_id:
        raise ValueError("base_state.state_id must be a non-empty string")
    if not np.allclose(
        base_state.robot_history[-1],
        0.0,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("base_state current robot pose must be the local origin")

    if base_state.static_map_local is None:
        static = np.zeros((grid.height, grid.width), dtype=np.bool_)
    else:
        static_source = base_state.static_map_local
        if static_source.dtype != ARRAY_DTYPE:
            raise TypeError("base_state.static_map_local must have float32 dtype")
        if not np.isin(static_source, (0.0, 1.0)).all():
            raise ValueError("base_state.static_map_local must be binary")
        static = np.asarray(static_source != 0.0, dtype=np.bool_, order="C")

    context = np.zeros((grid.height, grid.width), dtype=np.bool_)
    for object_id in base_state.dynamic_object_ids:
        footprint = footprint_from_spec(
            base_state.visible_dynamic_object_specs[object_id]
        )
        context |= rasterize_footprint(
            footprint,
            base_state.visible_dynamic_object_history[object_id][-1],
            grid,
        )

    occluder_id, _footprint, _pose, causal_mask = _validate_causal_occluder(
        causal_occluder,
        base_state_id=base_state.state_id,
        grid=grid,
    )
    if np.any(causal_mask & static):
        raise ValueError("causal_occluder must not overlap base static occupancy")
    if np.any(causal_mask & context):
        raise ValueError("causal_occluder must not overlap current context occupancy")

    return BlindRegion(
        base_state_id=base_state.state_id,
        causal_occluder_id=occluder_id,
        grid=grid,
        sensor_pose=np.asarray(base_state.robot_history[-1], dtype=np.float64),
        static_occupancy=static,
        current_context_occupancy=context,
        causal_occluder_mask=causal_mask,
    )


__all__ = (
    "BLIND_REGION_VERSION",
    "BlindRegion",
    "build_blind_region",
)
