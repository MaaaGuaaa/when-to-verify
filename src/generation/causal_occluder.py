"""Target-independent deterministic causal environment-occluder proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from numbers import Real
import struct
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from src.contracts import GridSpec
from src.geometry import (
    RectangleFootprint,
    footprint_aabb,
    grid_bounds,
    grid_cell_centers,
    rasterize_footprint,
    raycast_visibility,
    wrap_angle,
)
from src.generation.occluder_sampler import (
    OccluderGeometryCandidate,
    occluder_collision_sweep_rejection_reason,
)
from src.utils.seeding import make_rng


CAUSAL_OCCLUDER_SCHEDULE_VERSION = "causal_occluder_schedule_v1"
CAUSAL_OCCLUDER_PROPOSAL_VERSION = "causal_occluder_proposal_v2"

_OCCLUDER_TYPES = ("wall", "shelf", "pillar")
_ANCHOR_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _quantile(value: Any, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} quantile must lie in [0, 1]")
    return result


def _positive_range(value: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain [minimum, maximum]")
    lower = _finite_real(value[0], name=f"{name}[0]")
    upper = _finite_real(value[1], name=f"{name}[1]")
    if lower <= 0.0 or lower > upper:
        raise ValueError(f"{name} must be a positive ordered range")
    return lower, upper


@dataclass(frozen=True)
class CausalOccluderParameters:
    """One auditable point in the deterministic causal proposal schedule."""

    proposal_index: int
    anchor_index: int
    anchor_quantile: float
    bearing_bin: int
    range_quantile: float
    yaw_index: int
    yaw_offset_rad: float
    occluder_type: str
    dimension_quantile: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proposal_index",
            _integer(self.proposal_index, name="proposal_index", minimum=0),
        )
        object.__setattr__(
            self,
            "anchor_index",
            _integer(self.anchor_index, name="anchor_index", minimum=0),
        )
        object.__setattr__(
            self,
            "anchor_quantile",
            _quantile(self.anchor_quantile, name="anchor_quantile"),
        )
        if (
            self.anchor_index >= len(_ANCHOR_QUANTILES)
            or self.anchor_quantile != _ANCHOR_QUANTILES[self.anchor_index]
        ):
            raise ValueError(
                "anchor_index and anchor_quantile must identify one frozen stratum"
            )
        object.__setattr__(
            self,
            "bearing_bin",
            _integer(self.bearing_bin, name="bearing_bin", minimum=0),
        )
        object.__setattr__(
            self,
            "range_quantile",
            _quantile(self.range_quantile, name="range_quantile"),
        )
        if isinstance(self.yaw_index, (bool, np.bool_)) or not isinstance(
            self.yaw_index, (int, np.integer)
        ):
            raise TypeError("yaw_index must be an integer")
        object.__setattr__(self, "yaw_index", int(self.yaw_index))
        object.__setattr__(
            self,
            "yaw_offset_rad",
            _finite_real(self.yaw_offset_rad, name="yaw_offset_rad"),
        )
        if self.occluder_type not in _OCCLUDER_TYPES:
            raise ValueError("occluder_type must be wall, shelf, or pillar")
        object.__setattr__(
            self,
            "dimension_quantile",
            _quantile(self.dimension_quantile, name="dimension_quantile"),
        )


def normalize_causal_occluder_config(
    config: Mapping[str, Any],
) -> dict[str, object]:
    """Validate and canonicalize the exact causal-occluder config schema."""

    if not isinstance(config, Mapping):
        raise TypeError("causal occluder config must be a mapping")
    expected = {
        "types",
        "interaction_range_m",
        "bearing_bin_count",
        "yaw_step_deg",
        "minimum_shadow_center_cells",
        *_OCCLUDER_TYPES,
    }
    if set(config) != expected:
        raise ValueError(
            "config keys do not match the frozen causal occluder schema"
        )
    types = config["types"]
    if not isinstance(types, (list, tuple)) or not types:
        raise ValueError("types must be a non-empty sequence")
    if len(set(types)) != len(types) or any(kind not in _OCCLUDER_TYPES for kind in types):
        raise ValueError("types must be unique wall/shelf/pillar names")

    bearing_bin_count = _integer(
        config["bearing_bin_count"],
        name="bearing_bin_count",
        minimum=4,
    )
    yaw_step_deg = _finite_real(config["yaw_step_deg"], name="yaw_step_deg")
    if yaw_step_deg <= 0.0:
        raise ValueError("yaw_step_deg must be positive")
    minimum_shadow_center_cells = _integer(
        config["minimum_shadow_center_cells"],
        name="minimum_shadow_center_cells",
        minimum=1,
    )
    normalized: dict[str, object] = {
        "types": tuple(types),
        "interaction_range_m": _positive_range(
            config["interaction_range_m"], name="interaction_range_m"
        ),
        "bearing_bin_count": bearing_bin_count,
        "yaw_step_deg": yaw_step_deg,
        "minimum_shadow_center_cells": minimum_shadow_center_cells,
    }
    for kind in _OCCLUDER_TYPES:
        node = config[kind]
        if not isinstance(node, Mapping) or set(node) != {
            "length_range_m",
            "width_range_m",
        }:
            raise ValueError(
                f"{kind} must contain exactly length_range_m and width_range_m"
            )
        normalized[kind] = {
            "length_range_m": _positive_range(
                node["length_range_m"], name=f"{kind}.length_range_m"
            ),
            "width_range_m": _positive_range(
                node["width_range_m"], name=f"{kind}.width_range_m"
            ),
        }
    return normalized


def _digest_parts(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, byteorder="big", signed=False))
        digest.update(part)
    return digest.hexdigest()


def _canonical_config_bytes(normalized: Mapping[str, Any]) -> bytes:
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


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
    shape_bytes = struct.pack(
        ">q" + "q" * values.ndim,
        values.ndim,
        *values.shape,
    )
    return _digest_parts(
        name.encode("ascii"),
        values.dtype.str.encode("ascii"),
        shape_bytes,
        values.tobytes(order="C"),
    )


def _context_digest(
    *,
    grid: GridSpec,
    config_digest: str,
    static_occupancy_digest: str,
    current_context_occupancy_digest: str,
    baseline_occupancy_digest: str,
    baseline_visibility_digest: str,
    interaction_region_digest: str,
    interaction_poses_digest: str,
    sensor_pose_digest: str,
) -> str:
    return _digest_parts(
        CAUSAL_OCCLUDER_PROPOSAL_VERSION.encode("ascii"),
        _grid_bytes(grid),
        config_digest.encode("ascii"),
        static_occupancy_digest.encode("ascii"),
        current_context_occupancy_digest.encode("ascii"),
        baseline_occupancy_digest.encode("ascii"),
        baseline_visibility_digest.encode("ascii"),
        interaction_region_digest.encode("ascii"),
        interaction_poses_digest.encode("ascii"),
        sensor_pose_digest.encode("ascii"),
    )


def _canonical_bool_grid(
    value: Any,
    *,
    name: str,
    grid: GridSpec,
    input_dtype: bool,
) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an array") from exc
    if array.shape != (grid.height, grid.width):
        raise ValueError(f"{name} must have grid shape")
    if input_dtype:
        if array.dtype not in (np.dtype(np.bool_), np.dtype(np.float32)):
            raise TypeError(f"{name} must have bool or float32 dtype")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError(f"{name} must be binary")
    elif array.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must have bool dtype")
    return np.array(array != 0, dtype=np.bool_, order="C", copy=True)


def _canonical_float_array(
    value: Any,
    *,
    name: str,
    shape_tail: tuple[int, ...],
    vector: bool = False,
) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an array") from exc
    if vector:
        valid_shape = array.shape == shape_tail
    else:
        valid_shape = array.ndim == len(shape_tail) + 1 and array.shape[1:] == shape_tail
    if not valid_shape or (not vector and array.shape[0] == 0):
        expected = shape_tail if vector else ("T", *shape_tail)
        raise ValueError(f"{name} must have shape {expected}")
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{name} must have float32 or float64 dtype")
    result = np.array(array, dtype=np.dtype("<f8"), order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _immutable_array(values: np.ndarray) -> tuple[bytes, np.ndarray]:
    storage = values.tobytes(order="C")
    immutable = np.frombuffer(storage, dtype=values.dtype).reshape(values.shape)
    return storage, immutable


def _length_prefixed_parts(binding: bytes) -> tuple[bytes, ...]:
    parts = []
    offset = 0
    while offset < len(binding):
        if len(binding) - offset < 8:
            raise ValueError("proposal binding has a truncated length prefix")
        size = int.from_bytes(
            binding[offset : offset + 8],
            byteorder="big",
            signed=False,
        )
        offset += 8
        end = offset + size
        if end > len(binding):
            raise ValueError("proposal binding has a truncated part")
        parts.append(binding[offset:end])
        offset = end
    return tuple(parts)


def _binding_utf8_identity(part: bytes, *, name: str) -> str:
    try:
        value = part.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"proposal binding {name} must be strict UTF-8"
        ) from exc
    if not value or value.encode("utf-8") != part:
        raise ValueError(
            f"proposal binding {name} must be non-empty canonical UTF-8"
        )
    return value


def _binding_seed(part: bytes) -> int:
    try:
        encoded_seed = part.decode("ascii", errors="strict")
        seed = int(encoded_seed, 10)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(
            "proposal binding seed must be a canonical decimal integer"
        ) from exc
    if str(seed) != encoded_seed:
        raise ValueError(
            "proposal binding seed must be a canonical decimal integer"
        )
    return seed


def _canonical_digest(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _binding_config_bytes(part: bytes) -> bytes:
    try:
        decoded = json.loads(part.decode("ascii", errors="strict"))
        normalized = normalize_causal_occluder_config(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(
            "proposal binding config must contain the frozen canonical schema"
        ) from exc
    canonical = _canonical_config_bytes(normalized)
    if canonical != part:
        raise ValueError("proposal binding config must use canonical JSON bytes")
    return canonical


def _same_canonical_value(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, tuple):
        return len(actual) == len(expected) and all(
            _same_canonical_value(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


@dataclass(frozen=True)
class CausalOccluderContext:
    """Immutable current-causal occupancy and renderer baseline evidence."""

    static_occupancy: np.ndarray
    current_context_occupancy: np.ndarray
    baseline_occupancy: np.ndarray
    baseline_visibility: np.ndarray = field(init=False)
    interaction_region: np.ndarray
    interaction_poses: np.ndarray
    sensor_pose: np.ndarray
    interaction_range_m: tuple[float, float]
    grid: GridSpec
    config_canonical_bytes: bytes
    config_digest: str
    static_occupancy_digest: str
    current_context_occupancy_digest: str
    baseline_occupancy_digest: str
    baseline_visibility_digest: str = field(init=False)
    interaction_region_digest: str
    interaction_poses_digest: str
    sensor_pose_digest: str
    context_digest: str = field(init=False)
    _array_storage: tuple[bytes, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.grid, GridSpec):
            raise TypeError("grid must be a GridSpec")
        grid_bounds(self.grid)
        if not isinstance(self.config_canonical_bytes, bytes):
            raise TypeError("config_canonical_bytes must be bytes")
        try:
            decoded_config = json.loads(self.config_canonical_bytes.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("config_canonical_bytes must encode canonical JSON") from exc
        normalized_config = normalize_causal_occluder_config(decoded_config)
        if _canonical_config_bytes(normalized_config) != self.config_canonical_bytes:
            raise ValueError("config_canonical_bytes are not canonical")
        expected_config_digest = _digest_parts(self.config_canonical_bytes)
        if self.config_digest != expected_config_digest:
            raise ValueError("config_digest does not match config_canonical_bytes")
        interaction_range = _positive_range(
            self.interaction_range_m,
            name="interaction_range_m",
        )
        if interaction_range != normalized_config["interaction_range_m"]:
            raise ValueError("interaction_range_m does not match config")

        arrays = {
            "static_occupancy": _canonical_bool_grid(
                self.static_occupancy,
                name="static_occupancy",
                grid=self.grid,
                input_dtype=False,
            ),
            "current_context_occupancy": _canonical_bool_grid(
                self.current_context_occupancy,
                name="current_context_occupancy",
                grid=self.grid,
                input_dtype=False,
            ),
            "baseline_occupancy": _canonical_bool_grid(
                self.baseline_occupancy,
                name="baseline_occupancy",
                grid=self.grid,
                input_dtype=False,
            ),
            "interaction_region": _canonical_bool_grid(
                self.interaction_region,
                name="interaction_region",
                grid=self.grid,
                input_dtype=False,
            ),
            "interaction_poses": _canonical_float_array(
                self.interaction_poses,
                name="interaction_poses",
                shape_tail=(3,),
            ),
            "sensor_pose": _canonical_float_array(
                self.sensor_pose,
                name="sensor_pose",
                shape_tail=(3,),
                vector=True,
            ),
        }
        for name, values in arrays.items():
            field_name = f"{name}_digest"
            expected_digest = _array_digest(name, values)
            if getattr(self, field_name) != expected_digest:
                raise ValueError(f"{field_name} does not match {name}")
        if not np.array_equal(
            arrays["baseline_occupancy"],
            arrays["static_occupancy"] | arrays["current_context_occupancy"],
        ):
            raise ValueError("baseline_occupancy must equal static | current context")
        x_min, x_max, y_min, y_max = grid_bounds(self.grid)
        sensor = arrays["sensor_pose"]
        if not (x_min <= sensor[0] < x_max and y_min <= sensor[1] < y_max):
            raise ValueError("sensor_pose x/y must lie inside the grid")

        centers = grid_cell_centers(self.grid)
        expected_region = np.zeros(
            (self.grid.height, self.grid.width), dtype=np.bool_
        )
        for pose in arrays["interaction_poses"]:
            distances = np.linalg.norm(centers - pose[:2], axis=-1)
            expected_region |= (distances >= interaction_range[0]) & (
                distances <= interaction_range[1]
            )
        if not np.array_equal(arrays["interaction_region"], expected_region):
            raise ValueError(
                "interaction_region must match grid centres and interaction poses"
            )

        arrays["baseline_visibility"] = np.asarray(
            raycast_visibility(
                arrays["baseline_occupancy"],
                self.grid,
                sensor_pose=arrays["sensor_pose"],
                fov_rad=2.0 * np.pi,
                max_range_m=None,
            ),
            dtype=np.bool_,
            order="C",
        )
        baseline_visibility_digest = _array_digest(
            "baseline_visibility",
            arrays["baseline_visibility"],
        )
        context_digest = _context_digest(
            grid=self.grid,
            config_digest=self.config_digest,
            static_occupancy_digest=self.static_occupancy_digest,
            current_context_occupancy_digest=self.current_context_occupancy_digest,
            baseline_occupancy_digest=self.baseline_occupancy_digest,
            baseline_visibility_digest=baseline_visibility_digest,
            interaction_region_digest=self.interaction_region_digest,
            interaction_poses_digest=self.interaction_poses_digest,
            sensor_pose_digest=self.sensor_pose_digest,
        )

        storages = []
        for name, values in arrays.items():
            storage, immutable = _immutable_array(values)
            storages.append(storage)
            object.__setattr__(self, name, immutable)
        object.__setattr__(self, "interaction_range_m", interaction_range)
        object.__setattr__(
            self,
            "baseline_visibility_digest",
            baseline_visibility_digest,
        )
        object.__setattr__(self, "context_digest", context_digest)
        object.__setattr__(self, "_array_storage", tuple(storages))


@dataclass(frozen=True)
class CausalOccluderDecision:
    """One auditable proposal verdict, including rejected proposal identity."""

    proposal_id: str
    proposal_index: int
    seed: int
    base_state_id: str
    trajectory_id: str
    parameters: CausalOccluderParameters
    config_digest: str
    context_digest: str
    interaction_region: np.ndarray
    interaction_region_digest: str
    proposal_pose: np.ndarray
    proposal_length_m: float
    proposal_width_m: float
    proposal_mask: np.ndarray
    grid: GridSpec
    accepted: OccluderGeometryCandidate | None
    useful_shadow_mask: np.ndarray
    useful_shadow_count: int
    rejection_stage: str | None
    rejection_reason: str | None
    _proposal_binding: bytes = field(repr=False, compare=False)
    _proposal_geometry_storage: tuple[bytes, bytes] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _useful_shadow_storage: bytes = field(
        init=False,
        repr=False,
        compare=False,
    )
    _interaction_region_storage: bytes = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_id, str) or not self.proposal_id:
            raise ValueError("proposal_id must be a non-empty string")
        if not isinstance(self._proposal_binding, bytes):
            raise TypeError("_proposal_binding must be canonical bytes")
        bound_proposal_id = (
            "causal-occluder-"
            + hashlib.sha256(self._proposal_binding).hexdigest()[:32]
        )
        if self.proposal_id != bound_proposal_id:
            raise ValueError("proposal_id does not match proposal binding")
        proposal_index = _integer(
            self.proposal_index,
            name="proposal_index",
            minimum=0,
        )
        if not isinstance(self.grid, GridSpec):
            raise TypeError("grid must be a GridSpec")
        grid_bounds(self.grid)
        try:
            proposal_pose_array = np.asarray(self.proposal_pose)
        except (TypeError, ValueError) as exc:
            raise TypeError("proposal_pose must be an array") from exc
        if (
            proposal_pose_array.shape != (3,)
            or proposal_pose_array.dtype != np.dtype(np.float64)
        ):
            raise ValueError("proposal_pose must be canonical float64 with shape (3,)")
        proposal_pose = np.array(
            proposal_pose_array,
            dtype=np.dtype("<f8"),
            order="C",
            copy=True,
        )
        if not np.isfinite(proposal_pose).all():
            raise ValueError("proposal_pose must contain only finite values")
        proposal_length_m = _finite_real(
            self.proposal_length_m,
            name="proposal_length_m",
        )
        proposal_width_m = _finite_real(
            self.proposal_width_m,
            name="proposal_width_m",
        )
        if proposal_length_m <= 0.0 or proposal_width_m <= 0.0:
            raise ValueError("proposal dimensions must be positive")
        proposal_mask = _canonical_bool_grid(
            self.proposal_mask,
            name="proposal_mask",
            grid=self.grid,
            input_dtype=False,
        )
        interaction_region = _canonical_bool_grid(
            self.interaction_region,
            name="interaction_region",
            grid=self.grid,
            input_dtype=False,
        )
        interaction_region_digest = _canonical_digest(
            self.interaction_region_digest,
            name="interaction_region_digest",
        )
        if interaction_region_digest != _array_digest(
            "interaction_region",
            interaction_region,
        ):
            raise ValueError(
                "interaction_region_digest does not match interaction_region"
            )
        binding_parts = _length_prefixed_parts(self._proposal_binding)
        if len(binding_parts) != 10:
            raise ValueError("proposal binding must contain ten canonical parts")
        if binding_parts[0] != CAUSAL_OCCLUDER_SCHEDULE_VERSION.encode("ascii"):
            raise ValueError("proposal binding schedule version does not match")
        if binding_parts[1] != CAUSAL_OCCLUDER_PROPOSAL_VERSION.encode("ascii"):
            raise ValueError("proposal binding proposal version does not match")
        bound_seed = _binding_seed(binding_parts[2])
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer")
        seed = int(self.seed)
        if seed != bound_seed:
            raise ValueError("proposal identity seed does not match proposal binding")
        bound_base_state_id = _binding_utf8_identity(
            binding_parts[3],
            name="base_state_id",
        )
        if (
            not isinstance(self.base_state_id, str)
            or not self.base_state_id
            or self.base_state_id != bound_base_state_id
        ):
            raise ValueError(
                "proposal identity base_state_id does not match proposal binding"
            )
        bound_trajectory_id = _binding_utf8_identity(
            binding_parts[4],
            name="trajectory_id",
        )
        if (
            not isinstance(self.trajectory_id, str)
            or not self.trajectory_id
            or self.trajectory_id != bound_trajectory_id
        ):
            raise ValueError(
                "proposal identity trajectory_id does not match proposal binding"
            )
        bound_parameters = _parameters_from_bytes(binding_parts[5])
        if not isinstance(self.parameters, CausalOccluderParameters):
            raise TypeError("parameters must be CausalOccluderParameters")
        if _parameter_bytes(self.parameters) != binding_parts[5]:
            raise ValueError(
                "proposal identity parameters do not match proposal binding"
            )
        if proposal_index != bound_parameters.proposal_index:
            raise ValueError("proposal_index must match parameters.proposal_index")
        canonical_config_bytes = _binding_config_bytes(binding_parts[7])
        config_digest = _canonical_digest(
            self.config_digest,
            name="config_digest",
        )
        if config_digest != _digest_parts(canonical_config_bytes):
            raise ValueError(
                "proposal identity config_digest does not match proposal binding"
            )
        try:
            bound_context_digest = binding_parts[8].decode(
                "ascii",
                errors="strict",
            )
        except UnicodeDecodeError as exc:
            raise ValueError(
                "proposal binding context digest must be canonical ASCII"
            ) from exc
        bound_context_digest = _canonical_digest(
            bound_context_digest,
            name="proposal binding context digest",
        )
        context_digest = _canonical_digest(
            self.context_digest,
            name="context_digest",
        )
        if context_digest != bound_context_digest:
            raise ValueError(
                "proposal identity context_digest does not match proposal binding"
            )
        try:
            bound_interaction_region_digest = binding_parts[9].decode(
                "ascii",
                errors="strict",
            )
        except UnicodeDecodeError as exc:
            raise ValueError(
                "proposal binding interaction region digest must be canonical ASCII"
            ) from exc
        bound_interaction_region_digest = _canonical_digest(
            bound_interaction_region_digest,
            name="proposal binding interaction region digest",
        )
        if interaction_region_digest != bound_interaction_region_digest:
            raise ValueError(
                "interaction region digest does not match proposal binding"
            )
        expected_geometry_bytes = np.asarray(
            [
                proposal_pose[0],
                proposal_pose[1],
                proposal_pose[2],
                proposal_length_m,
                proposal_width_m,
            ],
            dtype=np.dtype(">f8"),
        ).tobytes(order="C")
        if binding_parts[6] != expected_geometry_bytes:
            raise ValueError("proposal geometry does not match proposal binding")
        proposal_footprint = RectangleFootprint(
            proposal_length_m,
            proposal_width_m,
        )
        expected_proposal_mask = rasterize_footprint(
            proposal_footprint,
            proposal_pose,
            self.grid,
        )
        if not np.array_equal(proposal_mask, expected_proposal_mask):
            raise ValueError("proposal_mask must equal the rasterized proposal geometry")
        try:
            shadow = np.asarray(self.useful_shadow_mask)
        except (TypeError, ValueError) as exc:
            raise TypeError("useful_shadow_mask must be an array") from exc
        if shadow.shape != (self.grid.height, self.grid.width):
            raise ValueError("useful_shadow_mask must have grid shape")
        if shadow.dtype != np.dtype(np.bool_):
            raise TypeError("useful_shadow_mask must have bool dtype")
        canonical_shadow = np.array(
            shadow,
            dtype=np.bool_,
            order="C",
            copy=True,
        )
        expected_count = int(np.count_nonzero(canonical_shadow))
        count = _integer(
            self.useful_shadow_count,
            name="useful_shadow_count",
            minimum=0,
        )
        if count != expected_count:
            raise ValueError(
                "useful_shadow_count must equal the useful_shadow_mask count"
            )
        proposal_pose_storage, immutable_proposal_pose = _immutable_array(
            proposal_pose
        )
        proposal_mask_storage, immutable_proposal_mask = _immutable_array(
            proposal_mask
        )

        accepted = self.accepted
        if accepted is None:
            if self.rejection_stage not in {
                "bounds",
                "static",
                "continuous_clearance",
                "shadow",
            }:
                raise ValueError("rejected decision requires a known rejection_stage")
            if not isinstance(self.rejection_reason, str) or not self.rejection_reason:
                raise ValueError("rejected decision requires a rejection_reason")
            exact_reasons = {
                "bounds": "occluder_out_of_bounds",
                "static": "occluder_static_overlap",
                "shadow": "occluder_no_useful_shadow",
            }
            expected_reason = exact_reasons.get(self.rejection_stage)
            if expected_reason is not None and self.rejection_reason != expected_reason:
                raise ValueError(
                    "rejection_reason does not match the frozen rejection_stage"
                )
        else:
            if not isinstance(accepted, OccluderGeometryCandidate):
                raise TypeError("accepted must be an OccluderGeometryCandidate or None")
            if self.rejection_stage is not None or self.rejection_reason is not None:
                raise ValueError("accepted decision rejection fields must be None")
            if accepted.proposal_index != proposal_index:
                raise ValueError("accepted proposal_index must match decision")
            if accepted.occluder.get("proposal_id") != self.proposal_id:
                raise ValueError("proposal_id must match accepted candidate metadata")
            if accepted.occluder.get("occluder_id") != self.proposal_id:
                raise ValueError("proposal_id must match accepted occluder_id")
            expected_proposal_parameters = tuple(
                (field_name, getattr(bound_parameters, field_name))
                for field_name in bound_parameters.__dataclass_fields__
            )
            expected_metadata_identity = {
                "schedule_version": CAUSAL_OCCLUDER_SCHEDULE_VERSION,
                "proposal_version": CAUSAL_OCCLUDER_PROPOSAL_VERSION,
                "seed": seed,
                "base_state_id": bound_base_state_id,
                "trajectory_id": bound_trajectory_id,
                "proposal_index": bound_parameters.proposal_index,
                "proposal_parameters": expected_proposal_parameters,
                "config_digest": config_digest,
                "context_digest": context_digest,
                "type": bound_parameters.occluder_type,
            }
            for field_name, expected_value in expected_metadata_identity.items():
                if not _same_canonical_value(
                    accepted.occluder.get(field_name),
                    expected_value,
                ):
                    raise ValueError(
                        f"accepted metadata {field_name} must match proposal identity"
                    )
            pose = np.asarray(accepted.pose)
            mask = np.asarray(accepted.mask)
            if (
                pose.shape != (3,)
                or pose.dtype != np.dtype(np.float64)
                or not np.isfinite(pose).all()
            ):
                raise ValueError("accepted pose must be finite canonical float64")
            canonical_accepted_pose = np.array(
                pose,
                dtype=np.dtype("<f8"),
                order="C",
                copy=True,
            )
            if canonical_accepted_pose.tobytes(order="C") != proposal_pose.tobytes(
                order="C"
            ):
                raise ValueError("accepted pose must match proposal geometry")
            if mask.shape != proposal_mask.shape or mask.dtype != np.dtype(np.bool_):
                raise ValueError("accepted mask must be a bool proposal grid")
            canonical_accepted_mask = np.array(
                mask,
                dtype=np.bool_,
                order="C",
                copy=True,
            )
            if canonical_accepted_mask.tobytes(order="C") != proposal_mask.tobytes(
                order="C"
            ):
                raise ValueError("accepted mask must match proposal geometry")
            if not isinstance(accepted.footprint, RectangleFootprint):
                raise TypeError("accepted footprint must be a RectangleFootprint")
            accepted_dimensions = struct.pack(
                ">dd",
                accepted.footprint.length_m,
                accepted.footprint.width_m,
            )
            proposal_dimensions = struct.pack(
                ">dd",
                proposal_length_m,
                proposal_width_m,
            )
            if accepted_dimensions != proposal_dimensions:
                raise ValueError("accepted footprint must match proposal geometry")
            metadata_pose = accepted.occluder.get("pose")
            metadata_length_m = accepted.occluder.get("length_m")
            metadata_width_m = accepted.occluder.get("width_m")
            if (
                not isinstance(metadata_pose, tuple)
                or len(metadata_pose) != 3
                or any(type(value) is not float for value in metadata_pose)
                or type(metadata_length_m) is not float
                or type(metadata_width_m) is not float
            ):
                raise ValueError(
                    "accepted metadata geometry must contain canonical primitives"
                )
            metadata_geometry_bytes = np.asarray(
                [*metadata_pose, metadata_length_m, metadata_width_m],
                dtype=np.dtype(">f8"),
            ).tobytes(order="C")
            if metadata_geometry_bytes != expected_geometry_bytes:
                raise ValueError("accepted metadata geometry must match proposal")
            metadata = MappingProxyType(dict(accepted.occluder))
            accepted = OccluderGeometryCandidate(
                occluder=metadata,
                footprint=proposal_footprint,
                pose=immutable_proposal_pose,
                mask=immutable_proposal_mask,
                proposal_index=proposal_index,
            )

        interaction_region_storage, immutable_interaction_region = (
            _immutable_array(interaction_region)
        )
        storage, immutable_shadow = _immutable_array(canonical_shadow)
        object.__setattr__(self, "proposal_index", proposal_index)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "base_state_id", bound_base_state_id)
        object.__setattr__(self, "trajectory_id", bound_trajectory_id)
        object.__setattr__(self, "parameters", bound_parameters)
        object.__setattr__(self, "config_digest", config_digest)
        object.__setattr__(self, "context_digest", context_digest)
        object.__setattr__(
            self,
            "interaction_region",
            immutable_interaction_region,
        )
        object.__setattr__(
            self,
            "interaction_region_digest",
            interaction_region_digest,
        )
        object.__setattr__(self, "proposal_pose", immutable_proposal_pose)
        object.__setattr__(self, "proposal_length_m", proposal_length_m)
        object.__setattr__(self, "proposal_width_m", proposal_width_m)
        object.__setattr__(self, "proposal_mask", immutable_proposal_mask)
        object.__setattr__(self, "useful_shadow_count", count)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "useful_shadow_mask", immutable_shadow)
        object.__setattr__(
            self,
            "_proposal_geometry_storage",
            (proposal_pose_storage, proposal_mask_storage),
        )
        object.__setattr__(self, "_useful_shadow_storage", storage)
        object.__setattr__(
            self,
            "_interaction_region_storage",
            interaction_region_storage,
        )


def build_causal_occluder_schedule(
    *,
    config: Mapping[str, Any],
    max_candidates: int,
    seed: int,
    base_state_id: str,
    trajectory_id: str,
) -> tuple[CausalOccluderParameters, ...]:
    """Build a finite stable schedule with a balanced bearing prefix."""

    normalized = normalize_causal_occluder_config(config)
    count = _integer(max_candidates, name="max_candidates", minimum=1)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")
    if not isinstance(base_state_id, str) or not base_state_id:
        raise ValueError("base_state_id must be a non-empty string")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("trajectory_id must be a non-empty string")
    def dimension_rng(name: str) -> np.random.Generator:
        return make_rng(
            int(seed),
            CAUSAL_OCCLUDER_SCHEDULE_VERSION,
            base_state_id,
            trajectory_id,
            name,
        )

    bearing_count = int(normalized["bearing_bin_count"])
    bearing_rng = dimension_rng("bearing")

    def one_balanced_bearing_cycle() -> list[int]:
        bins_by_quadrant = [
            [
                bearing_bin
                for bearing_bin in range(bearing_count)
                if bearing_bin * 4 // bearing_count == quadrant
            ]
            for quadrant in range(4)
        ]
        for quadrant, values in enumerate(bins_by_quadrant):
            order = bearing_rng.permutation(len(values))
            bins_by_quadrant[quadrant] = [values[int(index)] for index in order]
        result: list[int] = []
        depth = 0
        while any(depth < len(values) for values in bins_by_quadrant):
            quadrant_order = bearing_rng.permutation(4)
            for quadrant in quadrant_order:
                values = bins_by_quadrant[int(quadrant)]
                if depth < len(values):
                    result.append(values[depth])
            depth += 1
        return result

    bearing_stream: list[int] = []
    while len(bearing_stream) < count:
        bearing_stream.extend(one_balanced_bearing_cycle())

    def stratified_stream(
        values: tuple[Any, ...],
        *,
        rng: np.random.Generator,
    ) -> list[Any]:
        result: list[Any] = []
        while len(result) < count:
            order = rng.permutation(len(values))
            result.extend(values[int(index)] for index in order)
        return result[:count]

    quantiles = (0.0, 0.25, 0.5, 0.75, 1.0)
    anchors = tuple(enumerate(quantiles))
    anchor_stream = stratified_stream(anchors, rng=dimension_rng("anchor"))
    range_stream = stratified_stream(quantiles, rng=dimension_rng("range"))
    yaw_stream = stratified_stream(
        (-2, -1, 0, 1, 2), rng=dimension_rng("yaw")
    )
    type_stream = stratified_stream(
        tuple(normalized["types"]), rng=dimension_rng("type")
    )
    dimension_stream = stratified_stream(
        quantiles, rng=dimension_rng("dimension")
    )
    yaw_step_rad = np.deg2rad(float(normalized["yaw_step_deg"]))

    return tuple(
        CausalOccluderParameters(
            proposal_index=proposal_index,
            anchor_index=int(anchor_stream[proposal_index][0]),
            anchor_quantile=float(anchor_stream[proposal_index][1]),
            bearing_bin=int(bearing_stream[proposal_index]),
            range_quantile=float(range_stream[proposal_index]),
            yaw_index=int(yaw_stream[proposal_index]),
            yaw_offset_rad=(
                int(yaw_stream[proposal_index]) * float(yaw_step_rad)
            ),
            occluder_type=str(type_stream[proposal_index]),
            dimension_quantile=float(dimension_stream[proposal_index]),
        )
        for proposal_index in range(count)
    )


def build_causal_occluder_context(
    *,
    static_occupancy: Any,
    current_context_occupancy: Any,
    interaction_poses: Any,
    sensor_pose: Any,
    grid: Any,
    config: Mapping[str, Any],
) -> CausalOccluderContext:
    """Build immutable current-causal occupancy and one formal baseline."""

    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    grid_bounds(grid)
    normalized = normalize_causal_occluder_config(config)
    static = _canonical_bool_grid(
        static_occupancy,
        name="static_occupancy",
        grid=grid,
        input_dtype=True,
    )
    current_context = _canonical_bool_grid(
        current_context_occupancy,
        name="current_context_occupancy",
        grid=grid,
        input_dtype=True,
    )
    poses = _canonical_float_array(
        interaction_poses,
        name="interaction_poses",
        shape_tail=(3,),
    )
    sensor = _canonical_float_array(
        sensor_pose,
        name="sensor_pose",
        shape_tail=(3,),
        vector=True,
    )
    baseline_occupancy = np.asarray(static | current_context, dtype=np.bool_)
    interaction_range = normalized["interaction_range_m"]
    centers = grid_cell_centers(grid)
    interaction_region = np.zeros((grid.height, grid.width), dtype=np.bool_)
    for pose in poses:
        distances = np.linalg.norm(centers - pose[:2], axis=-1)
        interaction_region |= (distances >= interaction_range[0]) & (
            distances <= interaction_range[1]
        )

    arrays = {
        "static_occupancy": static,
        "current_context_occupancy": current_context,
        "baseline_occupancy": baseline_occupancy,
        "interaction_region": interaction_region,
        "interaction_poses": poses,
        "sensor_pose": sensor,
    }
    digests = {
        f"{name}_digest": _array_digest(name, values)
        for name, values in arrays.items()
    }
    config_bytes = _canonical_config_bytes(normalized)
    config_digest = _digest_parts(config_bytes)
    return CausalOccluderContext(
        **arrays,
        interaction_range_m=interaction_range,
        grid=grid,
        config_canonical_bytes=config_bytes,
        config_digest=config_digest,
        **digests,
    )


def _range_quantile(bounds: tuple[float, float], quantile: float) -> float:
    return bounds[0] + quantile * (bounds[1] - bounds[0])


def _parameter_bytes(parameters: CausalOccluderParameters) -> bytes:
    return b"".join(
        (
            struct.pack(">q", parameters.proposal_index),
            struct.pack(">q", parameters.anchor_index),
            struct.pack(">d", parameters.anchor_quantile),
            struct.pack(">q", parameters.bearing_bin),
            struct.pack(">d", parameters.range_quantile),
            struct.pack(">q", parameters.yaw_index),
            struct.pack(">d", parameters.yaw_offset_rad),
            len(parameters.occluder_type.encode("ascii")).to_bytes(
                8, byteorder="big", signed=False
            ),
            parameters.occluder_type.encode("ascii"),
            struct.pack(">d", parameters.dimension_quantile),
        )
    )


def _parameters_from_bytes(part: bytes) -> CausalOccluderParameters:
    if len(part) < 72:
        raise ValueError("proposal binding parameters have invalid length")
    type_size = int.from_bytes(
        part[56:64],
        byteorder="big",
        signed=False,
    )
    type_end = 64 + type_size
    if type_end + 8 != len(part):
        raise ValueError("proposal binding parameters have invalid length")
    try:
        occluder_type = part[64:type_end].decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "proposal binding parameters type must be canonical UTF-8 ASCII"
        ) from exc
    try:
        parameters = CausalOccluderParameters(
            proposal_index=struct.unpack_from(">q", part, 0)[0],
            anchor_index=struct.unpack_from(">q", part, 8)[0],
            anchor_quantile=struct.unpack_from(">d", part, 16)[0],
            bearing_bin=struct.unpack_from(">q", part, 24)[0],
            range_quantile=struct.unpack_from(">d", part, 32)[0],
            yaw_index=struct.unpack_from(">q", part, 40)[0],
            yaw_offset_rad=struct.unpack_from(">d", part, 48)[0],
            occluder_type=occluder_type,
            dimension_quantile=struct.unpack_from(">d", part, type_end)[0],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("proposal binding parameters are invalid") from exc
    if _parameter_bytes(parameters) != part:
        raise ValueError("proposal binding parameters are not canonical")
    return parameters


def _proposal_id(
    *,
    parameters: CausalOccluderParameters,
    pose: np.ndarray,
    length_m: float,
    width_m: float,
    config_canonical_bytes: bytes,
    context_digest: str,
    interaction_region_digest: str,
    seed: int,
    base_state_id: str,
    trajectory_id: str,
) -> tuple[str, bytes]:
    pose_and_dimensions = np.asarray(
        [pose[0], pose[1], pose[2], length_m, width_m],
        dtype=np.dtype(">f8"),
    ).tobytes(order="C")
    parts = (
        CAUSAL_OCCLUDER_SCHEDULE_VERSION.encode("ascii"),
        CAUSAL_OCCLUDER_PROPOSAL_VERSION.encode("ascii"),
        str(seed).encode("ascii"),
        base_state_id.encode("utf-8"),
        trajectory_id.encode("utf-8"),
        _parameter_bytes(parameters),
        pose_and_dimensions,
        config_canonical_bytes,
        context_digest.encode("ascii"),
        interaction_region_digest.encode("ascii"),
    )
    binding = b"".join(
        len(part).to_bytes(8, byteorder="big", signed=False) + part
        for part in parts
    )
    digest = hashlib.sha256(binding).hexdigest()
    return f"causal-occluder-{digest[:32]}", binding


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


def _decision(
    *,
    proposal_id: str,
    proposal_binding: bytes,
    proposal_index: int,
    seed: int,
    base_state_id: str,
    trajectory_id: str,
    parameters: CausalOccluderParameters,
    config_digest: str,
    context_digest: str,
    interaction_region: np.ndarray,
    interaction_region_digest: str,
    proposal_pose: np.ndarray,
    proposal_length_m: float,
    proposal_width_m: float,
    proposal_mask: np.ndarray,
    grid: GridSpec,
    accepted: OccluderGeometryCandidate | None = None,
    useful_shadow_mask: np.ndarray | None = None,
    rejection_stage: str | None = None,
    rejection_reason: str | None = None,
) -> CausalOccluderDecision:
    shadow = (
        np.zeros((grid.height, grid.width), dtype=np.bool_)
        if useful_shadow_mask is None
        else np.asarray(useful_shadow_mask, dtype=np.bool_)
    )
    return CausalOccluderDecision(
        proposal_id=proposal_id,
        proposal_index=proposal_index,
        seed=seed,
        base_state_id=base_state_id,
        trajectory_id=trajectory_id,
        parameters=parameters,
        config_digest=config_digest,
        context_digest=context_digest,
        interaction_region=interaction_region,
        interaction_region_digest=interaction_region_digest,
        proposal_pose=proposal_pose,
        proposal_length_m=proposal_length_m,
        proposal_width_m=proposal_width_m,
        proposal_mask=proposal_mask,
        grid=grid,
        accepted=accepted,
        useful_shadow_mask=shadow,
        useful_shadow_count=int(np.count_nonzero(shadow)),
        rejection_stage=rejection_stage,
        rejection_reason=rejection_reason,
        _proposal_binding=proposal_binding,
    )


def propose_causal_occluder(
    context: CausalOccluderContext,
    *,
    collision_sweeps: Any,
    config: Mapping[str, Any],
    parameters: CausalOccluderParameters,
    seed: int,
    base_state_id: str,
    trajectory_id: str,
) -> CausalOccluderDecision:
    """Place and validate one target-independent causal occluder proposal."""

    if not isinstance(context, CausalOccluderContext):
        raise TypeError("context must be a CausalOccluderContext")
    if not isinstance(parameters, CausalOccluderParameters):
        raise TypeError("parameters must be CausalOccluderParameters")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if not isinstance(base_state_id, str) or not base_state_id:
        raise ValueError("base_state_id must be a non-empty string")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("trajectory_id must be a non-empty string")

    normalized = normalize_causal_occluder_config(config)
    config_bytes = _canonical_config_bytes(normalized)
    if config_bytes != context.config_canonical_bytes:
        raise ValueError("config does not match causal occluder context")
    bearing_count = int(normalized["bearing_bin_count"])
    if parameters.bearing_bin >= bearing_count:
        raise ValueError("parameters.bearing_bin exceeds bearing_bin_count")
    expected_yaw_offset = parameters.yaw_index * float(
        np.deg2rad(float(normalized["yaw_step_deg"]))
    )
    if struct.pack(">d", parameters.yaw_offset_rad) != struct.pack(
        ">d", expected_yaw_offset
    ):
        raise ValueError("parameters yaw index/offset identity mismatch")
    if parameters.occluder_type not in normalized["types"]:
        raise ValueError("parameters.occluder_type is disabled by config")

    anchor_pose_index = int(
        np.floor(
            parameters.anchor_quantile
            * (context.interaction_poses.shape[0] - 1)
            + 0.5
        )
    )
    anchor_pose = context.interaction_poses[anchor_pose_index]
    relative_bearing = 2.0 * np.pi * parameters.bearing_bin / bearing_count
    world_bearing = float(wrap_angle(anchor_pose[2] + relative_bearing))
    placement_range_m = _range_quantile(
        normalized["interaction_range_m"],
        parameters.range_quantile,
    )
    center = anchor_pose[:2] + placement_range_m * np.asarray(
        [np.cos(world_bearing), np.sin(world_bearing)],
        dtype=np.float64,
    )
    yaw = float(
        wrap_angle(
            world_bearing + 0.5 * np.pi + parameters.yaw_offset_rad
        )
    )
    pose = np.asarray([center[0], center[1], yaw], dtype=np.dtype("<f8"))
    dimensions = normalized[parameters.occluder_type]
    length_m = _range_quantile(
        dimensions["length_range_m"],
        parameters.dimension_quantile,
    )
    width_m = _range_quantile(
        dimensions["width_range_m"],
        parameters.dimension_quantile,
    )
    proposal_id, proposal_binding = _proposal_id(
        parameters=parameters,
        pose=pose,
        length_m=length_m,
        width_m=width_m,
        config_canonical_bytes=config_bytes,
        context_digest=context.context_digest,
        interaction_region_digest=context.interaction_region_digest,
        seed=seed,
        base_state_id=base_state_id,
        trajectory_id=trajectory_id,
    )
    footprint = RectangleFootprint(length_m=length_m, width_m=width_m)
    mask = rasterize_footprint(footprint, pose, context.grid)
    proposal_geometry = {
        "proposal_pose": pose,
        "proposal_length_m": length_m,
        "proposal_width_m": width_m,
        "proposal_mask": mask,
    }
    proposal_identity = {
        "seed": seed,
        "base_state_id": base_state_id,
        "trajectory_id": trajectory_id,
        "parameters": parameters,
        "config_digest": context.config_digest,
        "context_digest": context.context_digest,
        "interaction_region": context.interaction_region,
        "interaction_region_digest": context.interaction_region_digest,
    }

    if not _inside_grid(footprint, pose, context.grid):
        return _decision(
            proposal_id=proposal_id,
            proposal_binding=proposal_binding,
            proposal_index=parameters.proposal_index,
            grid=context.grid,
            **proposal_identity,
            **proposal_geometry,
            rejection_stage="bounds",
            rejection_reason="occluder_out_of_bounds",
        )
    if np.any(mask & context.static_occupancy):
        return _decision(
            proposal_id=proposal_id,
            proposal_binding=proposal_binding,
            proposal_index=parameters.proposal_index,
            grid=context.grid,
            **proposal_identity,
            **proposal_geometry,
            rejection_stage="static",
            rejection_reason="occluder_static_overlap",
        )
    clearance_reason = occluder_collision_sweep_rejection_reason(
        footprint,
        pose,
        collision_sweeps,
        grid=context.grid,
    )
    if clearance_reason is not None:
        return _decision(
            proposal_id=proposal_id,
            proposal_binding=proposal_binding,
            proposal_index=parameters.proposal_index,
            grid=context.grid,
            **proposal_identity,
            **proposal_geometry,
            rejection_stage="continuous_clearance",
            rejection_reason=clearance_reason,
        )

    visibility_with_obstacle = raycast_visibility(
        context.baseline_occupancy | mask,
        context.grid,
        sensor_pose=context.sensor_pose,
        fov_rad=2.0 * np.pi,
        max_range_m=None,
    )
    useful_shadow = (
        context.baseline_visibility
        & ~visibility_with_obstacle
        & ~context.baseline_occupancy
        & ~mask
        & context.interaction_region
    )
    useful_shadow_count = int(np.count_nonzero(useful_shadow))
    if useful_shadow_count < int(normalized["minimum_shadow_center_cells"]):
        return _decision(
            proposal_id=proposal_id,
            proposal_binding=proposal_binding,
            proposal_index=parameters.proposal_index,
            grid=context.grid,
            **proposal_identity,
            **proposal_geometry,
            useful_shadow_mask=useful_shadow,
            rejection_stage="shadow",
            rejection_reason="occluder_no_useful_shadow",
        )

    proposal_parameters = tuple(
        (field_name, getattr(parameters, field_name))
        for field_name in parameters.__dataclass_fields__
    )
    metadata = {
        "occluder_id": proposal_id,
        "proposal_id": proposal_id,
        "type": parameters.occluder_type,
        "pose": tuple(float(value) for value in pose),
        "length_m": float(length_m),
        "width_m": float(width_m),
        "geometry_source": "generator_config",
        "placement_strategy": "causal_free_space_schedule_v1",
        "schedule_version": CAUSAL_OCCLUDER_SCHEDULE_VERSION,
        "proposal_version": CAUSAL_OCCLUDER_PROPOSAL_VERSION,
        "seed": seed,
        "base_state_id": base_state_id,
        "trajectory_id": trajectory_id,
        "proposal_index": parameters.proposal_index,
        "proposal_parameters": proposal_parameters,
        "interaction_anchor_pose_index": anchor_pose_index,
        "placement_range_m": float(placement_range_m),
        "bearing_rad": world_bearing,
        "context_digest": context.context_digest,
        "config_digest": context.config_digest,
    }
    pose_storage, immutable_pose = _immutable_array(pose)
    mask_storage, immutable_mask = _immutable_array(
        np.asarray(mask, dtype=np.bool_, order="C")
    )
    candidate = OccluderGeometryCandidate(
        occluder=MappingProxyType(metadata),
        footprint=footprint,
        pose=immutable_pose,
        mask=immutable_mask,
        proposal_index=parameters.proposal_index,
    )
    _ = (pose_storage, mask_storage)
    return _decision(
        proposal_id=proposal_id,
        proposal_binding=proposal_binding,
        proposal_index=parameters.proposal_index,
        grid=context.grid,
        **proposal_identity,
        **proposal_geometry,
        accepted=candidate,
        useful_shadow_mask=useful_shadow,
    )
