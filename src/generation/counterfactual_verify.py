"""Leakage-safe expected FOV geometry and label-side verification observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from src.contracts import ARRAY_DTYPE, GridSpec
from src.geometry import raycast_visibility, rasterize_footprint, wrap_angle
from src.generation.dynamic_object_transplant import footprint_from_spec


OBSERVATION_SIGNATURE_DIM = 7


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _grid_array(
    value: Any,
    grid: GridSpec,
    *,
    name: str,
    dtype: np.dtype | type,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an np.ndarray")
    if value.shape != (grid.height, grid.width):
        raise ValueError(f"{name} must have grid shape")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    if dtype == np.float32:
        if value.dtype != ARRAY_DTYPE:
            raise TypeError(f"{name} dtype must be float32")
        return np.array(value, dtype=ARRAY_DTYPE, order="C", copy=True)
    if value.dtype.kind not in "biuf":
        raise TypeError(f"{name} must have boolean or numeric dtype")
    return np.asarray(value != 0, dtype=bool, order="C")


def _binary_static(value: Any, grid: GridSpec) -> np.ndarray:
    static = _grid_array(
        value, grid, name="static_occupancy", dtype=np.float32
    )
    if not np.isin(static, (0.0, 1.0)).all():
        raise ValueError("static_occupancy must be binary")
    return static


def _pose(value: Any, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an np.ndarray")
    if value.shape != (3,) or value.dtype != ARRAY_DTYPE:
        raise ValueError(f"{name} must be float32 with shape (3,)")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value.astype(np.float64)


def _owned_bool(value: np.ndarray, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional ndarray")
    if value.dtype != np.bool_:
        raise TypeError(f"{name} dtype must be bool")
    result = np.array(value, dtype=bool, order="C", copy=True)
    result.setflags(write=False)
    return result


def _owned_float(value: np.ndarray, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional ndarray")
    if value.dtype != ARRAY_DTYPE or not np.isfinite(value).all():
        raise TypeError(f"{name} must be finite float32")
    result = np.array(value, dtype=ARRAY_DTYPE, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CounterfactualObservation:
    """Only content observable after an action; no world/object identity fields."""

    visible_mask: np.ndarray
    visible_occupied_mask: np.ndarray
    visible_dynamic_occupancy: np.ndarray
    newly_visible_mask: np.ndarray
    updated_age_map: np.ndarray

    def __post_init__(self) -> None:
        boolean_names = (
            "visible_mask",
            "visible_occupied_mask",
            "visible_dynamic_occupancy",
            "newly_visible_mask",
        )
        arrays = []
        for name in boolean_names:
            owned = _owned_bool(getattr(self, name), name=name)
            object.__setattr__(self, name, owned)
            arrays.append(owned)
        age = _owned_float(self.updated_age_map, name="updated_age_map")
        object.__setattr__(self, "updated_age_map", age)
        shape = arrays[0].shape
        if any(array.shape != shape for array in (*arrays[1:], age)):
            raise ValueError("counterfactual observation grid shapes must align")
        if np.any(arrays[1] & ~arrays[0]):
            raise ValueError("visible occupied cells must be visible")
        if np.any(arrays[2] & ~arrays[1]):
            raise ValueError("visible dynamic cells must be visible occupied")
        if np.any(arrays[3] & ~arrays[0]):
            raise ValueError("newly visible cells must be visible")
        if np.any((age < 0.0) | (age > 1.0)):
            raise ValueError("updated_age_map values must be in [0,1]")


@dataclass(frozen=True)
class SignatureNormalizer:
    """Seven-feature normalization statistics fitted on train only."""

    mean: np.ndarray
    scale: np.ndarray
    fit_split: str

    def __post_init__(self) -> None:
        if self.fit_split != "train":
            raise ValueError("signature normalizer must be fitted on train")
        for name in ("mean", "scale"):
            value = getattr(self, name)
            if (
                not isinstance(value, np.ndarray)
                or value.shape != (OBSERVATION_SIGNATURE_DIM,)
                or value.dtype != ARRAY_DTYPE
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"normalizer {name} violates the signature contract")
            if name == "scale" and np.any(value <= 0.0):
                raise ValueError("normalizer scale must be positive")
            owned = np.array(value, dtype=ARRAY_DTYPE, order="C", copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)

    def transform(self, signatures: np.ndarray) -> np.ndarray:
        values = _signature_matrix(signatures)
        return np.asarray((values - self.mean) / self.scale, dtype=ARRAY_DTYPE)


def expected_verification_fov_mask(
    static_occupancy: np.ndarray,
    grid: GridSpec,
    *,
    sensor_pose: np.ndarray,
    fov_rad: float,
    max_range_m: float,
) -> np.ndarray:
    """Return `[1,H,W]` static-only expected visibility for model input."""

    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    static = _binary_static(static_occupancy, grid)
    pose = _pose(sensor_pose, name="sensor_pose")
    visible = raycast_visibility(
        static,
        grid,
        sensor_pose=pose,
        fov_rad=fov_rad,
        max_range_m=max_range_m,
    )
    return visible.astype(ARRAY_DTYPE, copy=False)[None, ...]


def _pose_at_time(
    current_pose: np.ndarray,
    future_poses: np.ndarray,
    *,
    time_s: float,
    future_dt_s: float,
    object_id: str,
    future_steps: int,
) -> np.ndarray:
    current = _pose(current_pose, name=f"dynamic_current_poses[{object_id!r}]")
    if (
        not isinstance(future_poses, np.ndarray)
        or future_poses.shape != (future_steps, 3)
        or future_poses.dtype != ARRAY_DTYPE
        or not np.isfinite(future_poses).all()
    ):
        raise ValueError(
            f"dynamic_future_poses[{object_id!r}] must be finite float32 "
            f"[{future_steps},3]"
        )
    if time_s > future_steps * future_dt_s + 1e-10:
        raise ValueError("verification action exceeds the oracle future horizon")
    all_poses = np.vstack((current[None, :], future_poses.astype(np.float64)))
    if time_s <= 0.0:
        return all_poses[0]
    interval = min(int(np.floor(time_s / future_dt_s)), future_steps - 1)
    lower_time = interval * future_dt_s
    fraction = min(1.0, max(0.0, (time_s - lower_time) / future_dt_s))
    start = all_poses[interval]
    end = all_poses[interval + 1]
    yaw_pair = np.unwrap(np.asarray([start[2], end[2]], dtype=np.float64))
    result = (1.0 - fraction) * start + fraction * end
    result[2] = wrap_angle((1.0 - fraction) * yaw_pair[0] + fraction * yaw_pair[1])
    return result


def simulate_counterfactual_observation(
    *,
    post_action_pose: np.ndarray,
    action_duration_s: float,
    static_occupancy: np.ndarray,
    dynamic_current_poses: Mapping[str, np.ndarray],
    dynamic_future_poses: Mapping[str, np.ndarray],
    dynamic_specs: Mapping[str, dict[str, object]],
    current_visible_mask: np.ndarray,
    current_age_map: np.ndarray,
    grid: GridSpec,
    future_dt_s: float,
    age_max_s: float,
    fov_rad: float,
    max_range_m: float,
) -> CounterfactualObservation:
    """Ray cast a full typed oracle world only on the label-generation side."""

    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    duration = _finite_real(action_duration_s, name="action_duration_s")
    dt_s = _finite_real(future_dt_s, name="future_dt_s")
    age_max = _finite_real(age_max_s, name="age_max_s")
    if duration < 0.0 or dt_s <= 0.0 or age_max <= 0.0:
        raise ValueError("time values must be positive, except duration may be zero")
    post_pose = _pose(post_action_pose, name="post_action_pose")
    static = _binary_static(static_occupancy, grid)
    current_visible = _grid_array(
        current_visible_mask,
        grid,
        name="current_visible_mask",
        dtype=bool,
    )
    age = _grid_array(
        current_age_map, grid, name="current_age_map", dtype=np.float32
    )
    if np.any((age < 0.0) | (age > 1.0)):
        raise ValueError("current_age_map values must be in [0,1]")
    for name, value in (
        ("dynamic_current_poses", dynamic_current_poses),
        ("dynamic_future_poses", dynamic_future_poses),
        ("dynamic_specs", dynamic_specs),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
    object_ids = set(dynamic_current_poses)
    if object_ids != set(dynamic_future_poses) or object_ids != set(dynamic_specs):
        raise ValueError("dynamic current/future/spec IDs must align")

    dynamic_occupancy = np.zeros(
        (grid.height, grid.width), dtype=bool, order="C"
    )
    for object_id in sorted(object_ids):
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("dynamic object IDs must be non-empty strings")
        footprint = footprint_from_spec(dynamic_specs[object_id])
        pose = _pose_at_time(
            dynamic_current_poses[object_id],
            dynamic_future_poses[object_id],
            time_s=duration,
            future_dt_s=dt_s,
            object_id=object_id,
            future_steps=grid.future_steps,
        )
        dynamic_occupancy |= rasterize_footprint(footprint, pose, grid)

    total_occupancy = (static != 0.0) | dynamic_occupancy
    visible = raycast_visibility(
        total_occupancy,
        grid,
        sensor_pose=post_pose,
        fov_rad=fov_rad,
        max_range_m=max_range_m,
    )
    visible_occupied = visible & total_occupancy
    visible_dynamic = visible & dynamic_occupancy
    newly_visible = visible & ~current_visible
    updated_age = np.minimum(age + duration / age_max, 1.0).astype(ARRAY_DTYPE)
    updated_age[visible] = np.float32(0.0)
    return CounterfactualObservation(
        visible_mask=visible,
        visible_occupied_mask=visible_occupied,
        visible_dynamic_occupancy=visible_dynamic,
        newly_visible_mask=newly_visible,
        updated_age_map=updated_age,
    )


def _mask(value: Any, grid: GridSpec, *, name: str) -> np.ndarray:
    return _grid_array(value, grid, name=name, dtype=bool)


def _minimum_mask_distance_m(
    source: np.ndarray, target: np.ndarray, *, resolution_m: float, sentinel_m: float
) -> float:
    source_indices = np.argwhere(source)
    target_indices = np.argwhere(target)
    if source_indices.size == 0 or target_indices.size == 0:
        return sentinel_m
    minimum_squared = np.inf
    for start in range(0, source_indices.shape[0], 256):
        chunk = source_indices[start : start + 256]
        deltas = chunk[:, None, :] - target_indices[None, :, :]
        minimum_squared = min(
            minimum_squared,
            float(np.min(np.sum(deltas.astype(np.float64) ** 2, axis=2))),
        )
        if minimum_squared == 0.0:
            break
    return float(np.sqrt(minimum_squared) * resolution_m)


def make_observation_signature(
    observation: CounterfactualObservation,
    *,
    grid: GridSpec,
    original_swept_mask: np.ndarray,
    replanned_swept_masks: Sequence[np.ndarray],
    local_goal_corridor_mask: np.ndarray,
    critical_region_mask: np.ndarray,
    previous_age_map: np.ndarray,
) -> np.ndarray:
    """Build the seven recommended features from observable masks only."""

    if not isinstance(observation, CounterfactualObservation):
        raise TypeError("observation must be a CounterfactualObservation")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    shape = (grid.height, grid.width)
    if observation.visible_mask.shape != shape:
        raise ValueError("observation shape differs from grid")
    original = _mask(original_swept_mask, grid, name="original_swept_mask")
    if not isinstance(replanned_swept_masks, Sequence):
        raise TypeError("replanned_swept_masks must be a sequence")
    replanned_union = np.zeros(shape, dtype=bool)
    for index, value in enumerate(replanned_swept_masks):
        replanned_union |= _mask(value, grid, name=f"replanned_swept_masks[{index}]")
    corridor = _mask(
        local_goal_corridor_mask, grid, name="local_goal_corridor_mask"
    )
    critical = _mask(critical_region_mask, grid, name="critical_region_mask")
    previous_age = _grid_array(
        previous_age_map, grid, name="previous_age_map", dtype=np.float32
    )
    if np.any((previous_age < 0.0) | (previous_age > 1.0)):
        raise ValueError("previous_age_map values must be in [0,1]")

    newly = observation.newly_visible_mask
    cell_area = float(grid.resolution_m) ** 2
    sentinel = float(
        np.hypot(grid.height * grid.resolution_m, grid.width * grid.resolution_m)
    )
    minimum_actor_distance = _minimum_mask_distance_m(
        observation.visible_dynamic_occupancy,
        corridor,
        resolution_m=float(grid.resolution_m),
        sentinel_m=sentinel,
    )
    if critical.any():
        age_reduction = float(
            np.mean(
                np.maximum(
                    previous_age[critical] - observation.updated_age_map[critical],
                    0.0,
                )
            )
        )
    else:
        age_reduction = 0.0
    values = np.asarray(
        [
            np.count_nonzero(newly) * cell_area,
            np.count_nonzero(newly & original) * cell_area,
            np.count_nonzero(newly & replanned_union) * cell_area,
            float(np.count_nonzero(newly & observation.visible_occupied_mask)),
            minimum_actor_distance,
            float(observation.visible_dynamic_occupancy.any()),
            age_reduction,
        ],
        dtype=ARRAY_DTYPE,
    )
    if values.shape != (OBSERVATION_SIGNATURE_DIM,) or not np.isfinite(values).all():
        raise RuntimeError("observation signature violates its finite shape contract")
    return values


def _signature_matrix(signatures: Any) -> np.ndarray:
    if not isinstance(signatures, np.ndarray) or signatures.dtype != ARRAY_DTYPE:
        raise TypeError("signatures must be a float32 ndarray")
    if signatures.ndim == 1:
        if signatures.shape != (OBSERVATION_SIGNATURE_DIM,):
            raise ValueError("signature must have seven features")
    elif signatures.ndim == 2:
        if signatures.shape[1] != OBSERVATION_SIGNATURE_DIM:
            raise ValueError("signature matrix must have seven columns")
    else:
        raise ValueError("signatures must have one or two dimensions")
    if not np.isfinite(signatures).all():
        raise ValueError("signatures must be finite")
    return np.asarray(signatures, dtype=ARRAY_DTYPE)


def fit_signature_normalizer(
    signatures: np.ndarray, *, split: str
) -> SignatureNormalizer:
    if split != "train":
        raise ValueError("signature normalizer statistics may be fitted on train only")
    values = _signature_matrix(signatures)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("train signature matrix requires at least two rows")
    mean = np.mean(values, axis=0, dtype=np.float64)
    scale = np.std(values, axis=0, dtype=np.float64)
    scale[scale <= np.finfo(np.float32).eps] = 1.0
    return SignatureNormalizer(
        mean=mean.astype(ARRAY_DTYPE),
        scale=scale.astype(ARRAY_DTYPE),
        fit_split="train",
    )


__all__ = (
    "OBSERVATION_SIGNATURE_DIM",
    "CounterfactualObservation",
    "SignatureNormalizer",
    "expected_verification_fov_mask",
    "fit_signature_normalizer",
    "make_observation_signature",
    "simulate_counterfactual_observation",
)
