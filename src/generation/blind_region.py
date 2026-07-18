"""Current-causal, renderer-identical blind-region construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from numbers import Integral, Real
import struct
from typing import Any

import numpy as np

from src.contracts import ARRAY_DTYPE, BaseState, GridSpec, validate_base_state
from src.geometry import (
    CircleFootprint,
    Footprint,
    RectangleFootprint,
    footprint_aabb,
    grid_bounds,
    grid_cell_centers,
    rasterize_footprint,
    raycast_visibility,
    wrap_angle,
)
from src.generation.causal_occluder import (
    CAUSAL_OCCLUDER_PROPOSAL_VERSION,
    CAUSAL_OCCLUDER_SCHEDULE_VERSION,
    CausalOccluderDecision,
)
from src.generation.dynamic_object_transplant import footprint_from_spec
from src.generation.event_target_motion_shard import compute_footprint_spec_digest
from src.generation.observation_renderer import RENDERER_LAYOUT_VERSION
from src.generation.occluder_sampler import OccluderGeometryCandidate
from src.generation.structural_blindspot import footprint_visibility_sequence


BLIND_REGION_VERSION = "blind_region_causal_delta_v2"
CENTER_MASK_VERSION = "footprint_center_mask_causal_delta_v2"
EXACT_HIDDEN_POSE_VERSION = "exact_hidden_pose_v1"
VISIBILITY_ALGORITHM_VERSION = "raycast_visibility_environment_v1"

_ENVIRONMENT_FOV_RAD = 2.0 * np.pi
_ENVIRONMENT_MAX_RANGE_M = None
_CAUSAL_OCCLUDER_TYPES = frozenset(("wall", "shelf", "pillar"))


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Real, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


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


def _validate_causal_candidate_surface(
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


def _validate_causal_decision(
    decision: Any,
    *,
    base_state_id: str,
    grid: GridSpec,
) -> tuple[
    str,
    RectangleFootprint,
    np.ndarray,
    np.ndarray,
    str,
    str,
]:
    if not isinstance(decision, CausalOccluderDecision):
        raise TypeError("causal_occluder must be a CausalOccluderDecision")
    if decision.accepted is None:
        raise ValueError("causal_occluder decision must be accepted")
    if decision.grid != grid:
        raise ValueError("causal_occluder decision grid must match blind-region grid")
    if decision.base_state_id != base_state_id:
        raise ValueError("causal_occluder decision base_state_id must match BaseState")
    binding = decision._proposal_binding
    if not isinstance(binding, bytes):
        raise TypeError("causal_occluder proposal binding must be bytes")
    binding_digest = hashlib.sha256(binding).hexdigest()
    expected_proposal_id = f"causal-occluder-{binding_digest[:32]}"
    if decision.proposal_id != expected_proposal_id:
        raise ValueError("causal_occluder proposal_id must match complete binding")
    proposal_id, footprint, pose, mask = _validate_causal_candidate_surface(
        decision.accepted,
        base_state_id=base_state_id,
        grid=grid,
    )
    if proposal_id != decision.proposal_id:
        raise ValueError("accepted causal candidate must match decision proposal_id")
    return (
        proposal_id,
        footprint,
        pose,
        mask,
        decision.context_digest,
        binding_digest,
    )


@dataclass(frozen=True)
class BlindRegion:
    """Immutable current-frame occupancy and renderer-derived blind evidence."""

    base_state_id: str
    causal_decision: CausalOccluderDecision = field(repr=False, compare=False)
    grid: GridSpec
    sensor_pose: np.ndarray
    static_occupancy: np.ndarray
    current_context_occupancy: np.ndarray
    version: str = field(init=False, default=BLIND_REGION_VERSION)
    visibility_algorithm_version: str = field(
        init=False,
        default=VISIBILITY_ALGORITHM_VERSION,
    )
    renderer_layout_version: str = field(
        init=False,
        default=RENDERER_LAYOUT_VERSION,
    )
    causal_occluder_id: str = field(init=False)
    causal_context_digest: str = field(init=False)
    causal_proposal_binding_digest: str = field(init=False)
    causal_occluder_mask: np.ndarray = field(init=False, repr=False)
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
        (
            causal_occluder_id,
            _footprint,
            _pose,
            causal,
            causal_context_digest,
            causal_proposal_binding_digest,
        ) = _validate_causal_decision(
            self.causal_decision,
            base_state_id=self.base_state_id,
            grid=self.grid,
        )
        if np.any(causal & static):
            raise ValueError("causal_occluder must not overlap base static occupancy")
        if np.any(causal & context):
            raise ValueError(
                "causal_occluder must not overlap current context occupancy"
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
        baseline_occupancy = np.asarray(
            static | context,
            dtype=np.bool_,
            order="C",
        )
        baseline_visibility = np.asarray(
            raycast_visibility(
                baseline_occupancy,
                self.grid,
                sensor_pose=sensor_pose,
                fov_rad=_ENVIRONMENT_FOV_RAD,
                max_range_m=_ENVIRONMENT_MAX_RANGE_M,
            ),
            dtype=np.bool_,
            order="C",
        )
        if baseline_visibility.shape != total.shape:
            raise ValueError("raycast_visibility returned an invalid shape")
        renderer_delta = np.asarray(
            baseline_visibility & raw_unobservable & ~total,
            dtype=np.bool_,
            order="C",
        )
        causal_shadow = _canonical_bool_grid(
            self.causal_decision.useful_shadow_mask,
            name="causal_occluder.useful_shadow_mask",
            grid=self.grid,
        )
        if np.any(causal_shadow & total):
            raise ValueError("causal useful shadow must contain only free cells")
        if np.any(causal_shadow & visibility):
            raise ValueError("causal useful shadow must be currently unobservable")
        if np.any(causal_shadow & ~renderer_delta):
            raise ValueError(
                "causal useful shadow must be a subset of the renderer delta"
            )
        blind_free = causal_shadow

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
                "causal_occluder_id": causal_occluder_id,
                "causal_context_digest": causal_context_digest,
                "causal_proposal_binding_digest": (
                    causal_proposal_binding_digest
                ),
                "visibility_algorithm_version": VISIBILITY_ALGORITHM_VERSION,
                "renderer_layout_version": RENDERER_LAYOUT_VERSION,
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
        object.__setattr__(self, "causal_occluder_id", causal_occluder_id)
        object.__setattr__(self, "causal_context_digest", causal_context_digest)
        object.__setattr__(
            self,
            "causal_proposal_binding_digest",
            causal_proposal_binding_digest,
        )
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
    causal_occluder: CausalOccluderDecision,
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

    return BlindRegion(
        base_state_id=base_state.state_id,
        causal_decision=causal_occluder,
        grid=grid,
        sensor_pose=np.asarray(base_state.robot_history[-1], dtype=np.float64),
        static_occupancy=static,
        current_context_occupancy=context,
    )


def _canonical_footprint_spec_bytes(
    footprint_spec: Any,
    *,
    footprint_spec_digest: Any,
) -> tuple[bytes, Footprint, str]:
    if not isinstance(footprint_spec, Mapping):
        raise TypeError("footprint_spec must be a mapping")
    try:
        canonical_bytes = json.dumps(
            dict(footprint_spec),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        canonical_spec = json.loads(canonical_bytes.decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("footprint_spec must be canonical JSON-safe data") from exc
    footprint = footprint_from_spec(canonical_spec)
    expected_digest = compute_footprint_spec_digest(canonical_spec)
    if (
        not isinstance(footprint_spec_digest, str)
        or footprint_spec_digest != expected_digest
    ):
        raise ValueError("footprint_spec_digest does not match footprint_spec")
    return canonical_bytes, footprint, expected_digest


def _footprint_from_canonical_bytes(
    canonical_bytes: bytes,
    *,
    expected_digest: str,
) -> Footprint:
    if not isinstance(canonical_bytes, bytes):
        raise TypeError("_footprint_spec_bytes must be canonical bytes")
    try:
        spec = json.loads(canonical_bytes.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("_footprint_spec_bytes must contain canonical JSON") from exc
    recoded = json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if recoded != canonical_bytes:
        raise ValueError("_footprint_spec_bytes must use canonical JSON encoding")
    footprint = footprint_from_spec(spec)
    if compute_footprint_spec_digest(spec) != expected_digest:
        raise ValueError("footprint_spec_digest does not match canonical bytes")
    return footprint


def _center_bounds_mask(
    footprint: Footprint,
    *,
    yaw_rad: float,
    grid: GridSpec,
    centers: np.ndarray,
) -> np.ndarray:
    relative_bounds = footprint_aabb(
        footprint,
        np.asarray([0.0, 0.0, yaw_rad], dtype=np.float64),
    )
    grid_x_min, grid_x_max, grid_y_min, grid_y_max = grid_bounds(grid)
    return np.asarray(
        (centers[..., 0] + relative_bounds[0] >= grid_x_min)
        & (centers[..., 0] + relative_bounds[1] < grid_x_max)
        & (centers[..., 1] + relative_bounds[2] >= grid_y_min)
        & (centers[..., 1] + relative_bounds[3] < grid_y_max),
        dtype=np.bool_,
        order="C",
    )


def _shift_for_offsets(
    source: np.ndarray,
    *,
    row_offset: int,
    column_offset: int,
) -> np.ndarray:
    """Return source[r+dr,c+dc] at each prospective centre cell."""

    height, width = source.shape
    shifted = np.zeros_like(source, dtype=np.bool_, order="C")
    destination_row_start = max(0, -row_offset)
    destination_row_end = min(height, height - row_offset)
    destination_column_start = max(0, -column_offset)
    destination_column_end = min(width, width - column_offset)
    if (
        destination_row_start >= destination_row_end
        or destination_column_start >= destination_column_end
    ):
        return shifted
    shifted[
        destination_row_start:destination_row_end,
        destination_column_start:destination_column_end,
    ] = source[
        destination_row_start + row_offset : destination_row_end + row_offset,
        destination_column_start
        + column_offset : destination_column_end
        + column_offset,
    ]
    return shifted


def _build_center_mask(
    region: BlindRegion,
    footprint: Footprint,
    *,
    yaw_rad: float,
) -> np.ndarray:
    centers = grid_cell_centers(region.grid)
    in_bounds = _center_bounds_mask(
        footprint,
        yaw_rad=yaw_rad,
        grid=region.grid,
        centers=centers,
    )
    if not np.any(in_bounds):
        return np.zeros_like(in_bounds, dtype=np.bool_, order="C")

    distance_squared = centers[..., 0] ** 2 + centers[..., 1] ** 2
    reference_flat = int(
        np.argmin(np.where(in_bounds, distance_squared, np.inf))
    )
    reference_row, reference_column = np.unravel_index(
        reference_flat,
        in_bounds.shape,
    )
    reference_xy = centers[reference_row, reference_column]
    reference_pose = np.asarray(
        [reference_xy[0], reference_xy[1], yaw_rad],
        dtype=np.float64,
    )
    reference_mask = rasterize_footprint(footprint, reference_pose, region.grid)
    stencil_indices = np.argwhere(reference_mask)
    if stencil_indices.size == 0:
        raise ValueError("footprint rasterization must touch at least one grid cell")

    result = np.array(in_bounds, dtype=np.bool_, order="C", copy=True)
    for row, column in stencil_indices:
        result &= _shift_for_offsets(
            region.blind_free_mask,
            row_offset=int(row) - int(reference_row),
            column_offset=int(column) - int(reference_column),
        )
    return result


@dataclass(frozen=True)
class FootprintCenterMask:
    """One footprint/yaw-specific broad-phase mask over grid-cell centres."""

    region: BlindRegion = field(repr=False, compare=False)
    footprint_spec_digest: str
    yaw_bin_rad: float
    _footprint_spec_bytes: bytes = field(repr=False, compare=False)
    version: str = field(init=False, default=CENTER_MASK_VERSION)
    region_digest: str = field(init=False)
    footprint_kind: str = field(init=False)
    center_mask: np.ndarray = field(init=False, repr=False)
    center_mask_digest: str = field(init=False)
    valid_cell_count: int = field(init=False)
    identity_digest: str = field(init=False)
    _center_mask_storage: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.region, BlindRegion):
            raise TypeError("region must be a BlindRegion")
        footprint = _footprint_from_canonical_bytes(
            self._footprint_spec_bytes,
            expected_digest=self.footprint_spec_digest,
        )
        requested_yaw = _finite_real(self.yaw_bin_rad, name="yaw_bin_rad")
        yaw = 0.0 if isinstance(footprint, CircleFootprint) else wrap_angle(
            requested_yaw
        )
        mask = _build_center_mask(self.region, footprint, yaw_rad=yaw)
        mask_digest = _array_digest("center_mask", mask)
        footprint_kind = (
            "circle" if isinstance(footprint, CircleFootprint) else "rectangle"
        )
        identity_digest = _digest_parts(
            CENTER_MASK_VERSION.encode("ascii"),
            self.region.region_digest.encode("ascii"),
            self.footprint_spec_digest.encode("ascii"),
            struct.pack(">d", yaw),
            mask_digest.encode("ascii"),
        )
        storage, immutable_mask = _immutable_array(mask)
        object.__setattr__(self, "yaw_bin_rad", yaw)
        object.__setattr__(self, "region_digest", self.region.region_digest)
        object.__setattr__(self, "footprint_kind", footprint_kind)
        object.__setattr__(self, "center_mask", immutable_mask)
        object.__setattr__(self, "center_mask_digest", mask_digest)
        object.__setattr__(
            self,
            "valid_cell_count",
            int(np.count_nonzero(mask)),
        )
        object.__setattr__(self, "identity_digest", identity_digest)
        object.__setattr__(self, "_center_mask_storage", storage)


def build_footprint_center_mask(
    region: BlindRegion,
    *,
    footprint_spec: Mapping[str, object],
    footprint_spec_digest: str,
    yaw_bin_rad: float,
) -> FootprintCenterMask:
    """Build a complete-footprint hidden/free mask for one orientation bin."""

    if not isinstance(region, BlindRegion):
        raise TypeError("region must be a BlindRegion")
    canonical_bytes, _footprint, digest = _canonical_footprint_spec_bytes(
        footprint_spec,
        footprint_spec_digest=footprint_spec_digest,
    )
    return FootprintCenterMask(
        region=region,
        footprint_spec_digest=digest,
        yaw_bin_rad=yaw_bin_rad,
        _footprint_spec_bytes=canonical_bytes,
    )


@dataclass(frozen=True)
class ExactHiddenPoseResult:
    """Exact continuous-pose authority after grid/yaw broad phases."""

    region: BlindRegion = field(repr=False, compare=False)
    footprint_spec_digest: str
    pose: np.ndarray
    _footprint_spec_bytes: bytes = field(repr=False, compare=False)
    version: str = field(init=False, default=EXACT_HIDDEN_POSE_VERSION)
    region_digest: str = field(init=False)
    footprint_kind: str = field(init=False)
    footprint_mask: np.ndarray = field(init=False, repr=False)
    footprint_mask_digest: str = field(init=False)
    in_bounds: bool = field(init=False)
    collision_free: bool = field(init=False)
    fully_hidden: bool = field(init=False)
    accepted: bool = field(init=False)
    rejection_reason: str | None = field(init=False)
    result_digest: str = field(init=False)
    _array_storage: tuple[bytes, bytes] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.region, BlindRegion):
            raise TypeError("region must be a BlindRegion")
        footprint = _footprint_from_canonical_bytes(
            self._footprint_spec_bytes,
            expected_digest=self.footprint_spec_digest,
        )
        pose = _canonical_pose(self.pose, name="pose")
        bounds = footprint_aabb(footprint, pose)
        grid_x_min, grid_x_max, grid_y_min, grid_y_max = grid_bounds(
            self.region.grid
        )
        in_bounds = bool(
            bounds[0] >= grid_x_min
            and bounds[1] < grid_x_max
            and bounds[2] >= grid_y_min
            and bounds[3] < grid_y_max
        )
        footprint_mask = rasterize_footprint(footprint, pose, self.region.grid)
        collision_free = not bool(
            np.any(footprint_mask & self.region.total_current_occupancy)
        )
        any_visible = bool(
            footprint_visibility_sequence(
                footprint,
                pose[None, :],
                self.region.visibility_mask,
                self.region.grid,
            )[0]
        )
        fully_hidden = not any_visible
        if not in_bounds:
            reason = "hidden_pose_out_of_bounds"
        elif not collision_free:
            reason = "hidden_pose_current_collision"
        elif not fully_hidden:
            reason = "hidden_pose_partially_visible"
        else:
            reason = None
        accepted = reason is None
        footprint_kind = (
            "circle" if isinstance(footprint, CircleFootprint) else "rectangle"
        )
        mask_digest = _array_digest("exact_footprint_mask", footprint_mask)
        result_digest = _digest_parts(
            EXACT_HIDDEN_POSE_VERSION.encode("ascii"),
            self.region.region_digest.encode("ascii"),
            self.footprint_spec_digest.encode("ascii"),
            np.asarray(pose, dtype=np.dtype(">f8")).tobytes(order="C"),
            mask_digest.encode("ascii"),
            bytes((int(in_bounds), int(collision_free), int(fully_hidden))),
            b"accepted" if accepted else reason.encode("ascii"),
        )
        pose_storage, immutable_pose = _immutable_array(pose)
        mask_storage, immutable_mask = _immutable_array(footprint_mask)
        object.__setattr__(self, "pose", immutable_pose)
        object.__setattr__(self, "region_digest", self.region.region_digest)
        object.__setattr__(self, "footprint_kind", footprint_kind)
        object.__setattr__(self, "footprint_mask", immutable_mask)
        object.__setattr__(self, "footprint_mask_digest", mask_digest)
        object.__setattr__(self, "in_bounds", in_bounds)
        object.__setattr__(self, "collision_free", collision_free)
        object.__setattr__(self, "fully_hidden", fully_hidden)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "rejection_reason", reason)
        object.__setattr__(self, "result_digest", result_digest)
        object.__setattr__(
            self,
            "_array_storage",
            (pose_storage, mask_storage),
        )


def check_exact_hidden_pose(
    region: BlindRegion,
    *,
    footprint_spec: Mapping[str, object],
    footprint_spec_digest: str,
    pose: np.ndarray,
) -> ExactHiddenPoseResult:
    """Check full-footprint hiding/collision at an exact continuous pose."""

    if not isinstance(region, BlindRegion):
        raise TypeError("region must be a BlindRegion")
    canonical_bytes, _footprint, digest = _canonical_footprint_spec_bytes(
        footprint_spec,
        footprint_spec_digest=footprint_spec_digest,
    )
    return ExactHiddenPoseResult(
        region=region,
        footprint_spec_digest=digest,
        pose=pose,
        _footprint_spec_bytes=canonical_bytes,
    )


__all__ = (
    "BLIND_REGION_VERSION",
    "CENTER_MASK_VERSION",
    "EXACT_HIDDEN_POSE_VERSION",
    "VISIBILITY_ALGORITHM_VERSION",
    "BlindRegion",
    "FootprintCenterMask",
    "ExactHiddenPoseResult",
    "build_blind_region",
    "build_footprint_center_mask",
    "check_exact_hidden_pose",
)
