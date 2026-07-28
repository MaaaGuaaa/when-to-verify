"""Leakage-safe expected FOV geometry and label-side verification observations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from src.contracts import ARRAY_DTYPE, GridSpec
from src.geometry import raycast_visibility, rasterize_footprint, wrap_angle
from src.generation.event_contracts import footprint_from_spec
from src.planning.verification_actions import ActionTrace


OBSERVATION_SIGNATURE_DIM = 7
SIGNATURE_NORMALIZER_VERSION = "verification_signature_normalizer_v1"


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
class CounterfactualObservationTrace:
    """Label-side frame sequence plus the cumulative observable evidence."""

    aggregate: CounterfactualObservation
    frames: tuple[CounterfactualObservation, ...]
    times_s: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.aggregate, CounterfactualObservation):
            raise TypeError("aggregate must be a CounterfactualObservation")
        if (
            not isinstance(self.frames, tuple)
            or not self.frames
            or any(
                not isinstance(frame, CounterfactualObservation)
                for frame in self.frames
            )
        ):
            raise TypeError("frames must be a non-empty observation tuple")
        if (
            not isinstance(self.times_s, np.ndarray)
            or self.times_s.dtype != np.float64
            or self.times_s.shape != (len(self.frames),)
            or not np.isfinite(self.times_s).all()
            or self.times_s[0] != 0.0
            or np.any(np.diff(self.times_s) <= 0.0)
        ):
            raise ValueError("trace times must be finite float64 starting at zero")
        shape = self.aggregate.visible_mask.shape
        if any(frame.visible_mask.shape != shape for frame in self.frames):
            raise ValueError("trace observation grid shapes must align")
        for name in (
            "visible_mask",
            "visible_occupied_mask",
            "visible_dynamic_occupancy",
            "newly_visible_mask",
        ):
            union = np.logical_or.reduce(
                tuple(getattr(frame, name) for frame in self.frames)
            )
            if not np.array_equal(getattr(self.aggregate, name), union):
                raise ValueError(f"aggregate {name} must equal the frame union")
        if not np.array_equal(
            self.aggregate.updated_age_map, self.frames[-1].updated_age_map
        ):
            raise ValueError("aggregate age map must equal the final frame age map")
        times = np.array(self.times_s, dtype=np.float64, order="C", copy=True)
        times.setflags(write=False)
        object.__setattr__(self, "times_s", times)


@dataclass(frozen=True)
class SignatureNormalizer:
    """Versioned seven-feature statistics fitted on and attributable to train."""

    version: str
    mean: np.ndarray
    scale: np.ndarray
    fit_split: str
    sample_count: int
    statistics_digest: str

    def __post_init__(self) -> None:
        if self.version != SIGNATURE_NORMALIZER_VERSION:
            raise ValueError("unsupported signature normalizer version")
        if self.fit_split != "train":
            raise ValueError("signature normalizer must be fitted on train")
        if isinstance(self.sample_count, (bool, np.bool_)) or not isinstance(
            self.sample_count, (Integral, np.integer)
        ):
            raise TypeError("normalizer sample_count must be an integer")
        count = int(self.sample_count)
        if count < 2:
            raise ValueError("normalizer sample_count must be at least two")
        object.__setattr__(self, "sample_count", count)
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
        digest = self.statistics_digest
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("normalizer statistics digest is invalid")
        expected = _signature_normalizer_digest(
            mean=self.mean,
            scale=self.scale,
            sample_count=count,
        )
        if digest != expected:
            raise ValueError("normalizer statistics digest mismatch")

    def transform(self, signatures: np.ndarray) -> np.ndarray:
        values = _signature_matrix(signatures)
        return np.asarray((values - self.mean) / self.scale, dtype=ARRAY_DTYPE)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "fit_split": self.fit_split,
            "sample_count": self.sample_count,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "statistics_digest": self.statistics_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SignatureNormalizer:
        expected_keys = {
            "version",
            "fit_split",
            "sample_count",
            "mean",
            "scale",
            "statistics_digest",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected_keys:
            raise ValueError("signature normalizer payload keys are invalid")
        try:
            mean = np.asarray(payload["mean"], dtype=ARRAY_DTYPE)
            scale = np.asarray(payload["scale"], dtype=ARRAY_DTYPE)
        except (TypeError, ValueError) as exc:
            raise ValueError("signature normalizer arrays are invalid") from exc
        return cls(
            version=payload["version"],
            mean=mean,
            scale=scale,
            fit_split=payload["fit_split"],
            sample_count=payload["sample_count"],
            statistics_digest=payload["statistics_digest"],
        )


def _signature_normalizer_digest(
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    sample_count: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(SIGNATURE_NORMALIZER_VERSION.encode("ascii"))
    digest.update(b"\0train\0")
    digest.update(str(sample_count).encode("ascii"))
    for name, value in (("mean", mean), ("scale", scale)):
        digest.update(b"\0")
        digest.update(name.encode("ascii"))
        digest.update(
            np.ascontiguousarray(value, dtype=ARRAY_DTYPE).tobytes(order="C")
        )
    return digest.hexdigest()


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


def expected_verification_fov_trace_mask(
    static_occupancy: np.ndarray,
    grid: GridSpec,
    *,
    action_trace: ActionTrace,
    fov_rad: float,
    max_range_m: float,
) -> np.ndarray:
    """Return the static-only visibility union along a complete action trace."""

    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    if not isinstance(action_trace, ActionTrace):
        raise TypeError("action_trace must be an ActionTrace")
    static = _binary_static(static_occupancy, grid)
    visible_union = np.zeros((grid.height, grid.width), dtype=bool, order="C")
    for pose in action_trace.poses:
        visible_union |= raycast_visibility(
            static,
            grid,
            sensor_pose=pose,
            fov_rad=fov_rad,
            max_range_m=max_range_m,
        )
    return visible_union.astype(ARRAY_DTYPE, copy=False)[None, ...]


def interpolate_dynamic_pose(
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
        pose = interpolate_dynamic_pose(
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


def simulate_counterfactual_observation_trace(
    *,
    action_trace: ActionTrace,
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
) -> CounterfactualObservationTrace:
    """Ray cast every action-trace frame and retain cumulative evidence."""

    if not isinstance(action_trace, ActionTrace):
        raise TypeError("action_trace must be an ActionTrace")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    dt_s = _finite_real(future_dt_s, name="future_dt_s")
    age_max = _finite_real(age_max_s, name="age_max_s")
    if dt_s <= 0.0 or age_max <= 0.0:
        raise ValueError("future_dt_s and age_max_s must be positive")
    if action_trace.times_s[-1] > grid.future_steps * dt_s + 1e-10:
        raise ValueError("verification action exceeds the oracle future horizon")
    static = _binary_static(static_occupancy, grid)
    if (
        not isinstance(current_visible_mask, np.ndarray)
        or current_visible_mask.dtype != np.bool_
    ):
        raise TypeError("current_visible_mask dtype must be bool")
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
    if any(not isinstance(object_id, str) or not object_id for object_id in object_ids):
        raise ValueError("dynamic object IDs must be non-empty strings")
    footprints = {}
    for object_id in sorted(object_ids):
        _pose(
            dynamic_current_poses[object_id],
            name=f"dynamic_current_poses[{object_id!r}]",
        )
        future = dynamic_future_poses[object_id]
        if (
            not isinstance(future, np.ndarray)
            or future.shape != (grid.future_steps, 3)
            or future.dtype != ARRAY_DTYPE
            or not np.isfinite(future).all()
        ):
            raise ValueError(
                f"dynamic_future_poses[{object_id!r}] must be finite float32 "
                f"[{grid.future_steps},3]"
            )
        footprints[object_id] = footprint_from_spec(dynamic_specs[object_id])

    frames: list[CounterfactualObservation] = []
    visible_union = np.zeros((grid.height, grid.width), dtype=bool, order="C")
    occupied_union = np.zeros_like(visible_union)
    dynamic_union = np.zeros_like(visible_union)
    newly_visible_union = np.zeros_like(visible_union)
    previous_time_s = 0.0
    for robot_pose, time_s in zip(action_trace.poses, action_trace.times_s):
        time_value = float(time_s)
        dynamic_occupancy = np.zeros_like(visible_union)
        for object_id in sorted(object_ids):
            pose = interpolate_dynamic_pose(
                dynamic_current_poses[object_id],
                dynamic_future_poses[object_id],
                time_s=time_value,
                future_dt_s=dt_s,
                object_id=object_id,
                future_steps=grid.future_steps,
            )
            occupied = rasterize_footprint(footprints[object_id], pose, grid)
            dynamic_occupancy |= occupied
        total_occupancy = (static != 0.0) | dynamic_occupancy
        visible = raycast_visibility(
            total_occupancy,
            grid,
            sensor_pose=robot_pose,
            fov_rad=fov_rad,
            max_range_m=max_range_m,
        )
        visible_occupied = visible & total_occupancy
        visible_dynamic = visible & dynamic_occupancy
        newly_visible = visible & ~current_visible
        age = np.minimum(
            age + (time_value - previous_time_s) / age_max,
            1.0,
        ).astype(ARRAY_DTYPE)
        age[visible] = np.float32(0.0)
        frame = CounterfactualObservation(
            visible_mask=visible,
            visible_occupied_mask=visible_occupied,
            visible_dynamic_occupancy=visible_dynamic,
            newly_visible_mask=newly_visible,
            updated_age_map=age,
        )
        frames.append(frame)
        visible_union |= frame.visible_mask
        occupied_union |= frame.visible_occupied_mask
        dynamic_union |= frame.visible_dynamic_occupancy
        newly_visible_union |= frame.newly_visible_mask
        previous_time_s = time_value

    aggregate = CounterfactualObservation(
        visible_mask=visible_union,
        visible_occupied_mask=occupied_union,
        visible_dynamic_occupancy=dynamic_union,
        newly_visible_mask=newly_visible_union,
        updated_age_map=frames[-1].updated_age_map,
    )
    return CounterfactualObservationTrace(
        aggregate=aggregate,
        frames=tuple(frames),
        times_s=action_trace.times_s,
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
    newly_occupied = newly & observation.visible_occupied_mask
    newly_dynamic = newly & observation.visible_dynamic_occupancy
    cell_area = float(grid.resolution_m) ** 2
    sentinel = float(
        np.hypot(grid.height * grid.resolution_m, grid.width * grid.resolution_m)
    )
    minimum_actor_distance = _minimum_mask_distance_m(
        newly_occupied,
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
            float(np.count_nonzero(newly_occupied)),
            minimum_actor_distance,
            float(newly_dynamic.any()),
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
    mean_array = mean.astype(ARRAY_DTYPE)
    scale_array = scale.astype(ARRAY_DTYPE)
    return SignatureNormalizer(
        version=SIGNATURE_NORMALIZER_VERSION,
        mean=mean_array,
        scale=scale_array,
        fit_split="train",
        sample_count=values.shape[0],
        statistics_digest=_signature_normalizer_digest(
            mean=mean_array,
            scale=scale_array,
            sample_count=values.shape[0],
        ),
    )


__all__ = (
    "OBSERVATION_SIGNATURE_DIM",
    "SIGNATURE_NORMALIZER_VERSION",
    "CounterfactualObservation",
    "CounterfactualObservationTrace",
    "SignatureNormalizer",
    "expected_verification_fov_mask",
    "expected_verification_fov_trace_mask",
    "fit_signature_normalizer",
    "interpolate_dynamic_pose",
    "make_observation_signature",
    "simulate_counterfactual_observation",
    "simulate_counterfactual_observation_trace",
)
