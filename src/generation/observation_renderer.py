"""Render deployment-observable BEV inputs from complete scene history."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from src.contracts import (
    ARRAY_DTYPE,
    HISTORY_CHANNELS,
    SCHEMA_VERSION,
    STATE_CHANNELS,
    BaseState,
    GridSpec,
    build_grid_spec,
    validate_base_state,
)
from src.geometry.footprints import RectangleFootprint
from src.geometry.rasterization import rasterize_footprint
from src.geometry.raycasting import raycast_visibility
from src.generation.dynamic_object_transplant import footprint_from_spec
from src.generation.structural_blindspot import (
    StructuralBlindSpot,
    build_structural_visibility,
)
from src.utils.config import validate_config

RENDERER_LAYOUT_VERSION = "bev_history2_state9_v1"


@dataclass(frozen=True)
class RenderedObservation:
    """Owned model-input arrays plus whitelisted renderer provenance."""

    bev_history: np.ndarray
    state_channels: np.ndarray
    metadata: dict[str, str]


def _require_config_section(
    config: dict,
    section: str,
    required_keys: set[str],
) -> dict:
    value = config.get(section)
    if not isinstance(value, dict):
        raise TypeError(f"config.{section} must be a dict")
    missing = required_keys - set(value)
    if missing:
        raise ValueError(
            f"config.{section} is missing required keys: {sorted(missing)}"
        )
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a positive finite real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


def _unit_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number in [0, 1]")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite real number in [0, 1]")
    return result


def _validated_config_and_grid(config: Any) -> tuple[dict, GridSpec]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    config_dict = dict(config)
    validate_config(config_dict)
    if config_dict.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"config.schema_version must be {SCHEMA_VERSION}")
    bev = _require_config_section(
        config_dict,
        "bev",
        {
            "range_m",
            "size",
            "resolution_m",
            "history_steps",
            "history_dt_s",
            "future_steps",
        },
    )
    robot = _require_config_section(config_dict, "robot", {"length_m", "width_m"})
    age_map = _require_config_section(
        config_dict,
        "age_map",
        {"a_max_s", "never_seen_value", "visible_value"},
    )
    size = _positive_integer(bev["size"], name="config.bev.size")
    _positive_integer(
        bev["history_steps"], name="config.bev.history_steps"
    )
    _positive_integer(bev["future_steps"], name="config.bev.future_steps")
    range_m = _positive_real(bev["range_m"], name="config.bev.range_m")
    resolution_m = _positive_real(
        bev["resolution_m"], name="config.bev.resolution_m"
    )
    _positive_real(bev["history_dt_s"], name="config.bev.history_dt_s")
    _positive_real(robot["length_m"], name="config.robot.length_m")
    _positive_real(robot["width_m"], name="config.robot.width_m")
    _positive_real(age_map["a_max_s"], name="config.age_map.a_max_s")
    for key in ("never_seen_value", "visible_value"):
        _unit_real(age_map[key], name=f"config.age_map.{key}")
    if not np.isclose(
        range_m,
        size * resolution_m,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "config.bev.range_m must equal size * resolution_m"
        )

    grid = build_grid_spec(config_dict)
    if grid.n_history_channels != len(HISTORY_CHANNELS):
        raise ValueError("grid.n_history_channels violates the frozen contract")
    if grid.n_state_channels != len(STATE_CHANNELS):
        raise ValueError("grid.n_state_channels violates the frozen contract")
    return config_dict, grid


def _validate_scene(
    base_state: Any,
    scene_dynamic_history: Any,
    scene_dynamic_specs: Any,
    grid: GridSpec,
) -> dict[str, Any]:
    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    validate_base_state(base_state, grid)
    if not isinstance(base_state.state_id, str) or not base_state.state_id:
        raise ValueError("base_state.state_id must be a non-empty string")
    if not np.allclose(base_state.robot_history[-1], 0.0, rtol=0.0, atol=1e-6):
        raise ValueError("base_state current robot pose must be the local origin")
    if not isinstance(scene_dynamic_history, Mapping):
        raise TypeError("scene_dynamic_history must be a mapping")
    if not isinstance(scene_dynamic_specs, Mapping):
        raise TypeError("scene_dynamic_specs must be a mapping")
    if set(scene_dynamic_history) != set(scene_dynamic_specs):
        raise ValueError("scene dynamic history and spec keys must match")
    object_ids = set(scene_dynamic_history)
    if not all(
        isinstance(object_id, str) and object_id for object_id in object_ids
    ):
        raise ValueError("scene dynamic object IDs must be non-empty strings")

    footprints: dict[str, Any] = {}
    for object_id in sorted(object_ids):
        history = scene_dynamic_history[object_id]
        name = f"scene_dynamic_history[{object_id!r}]"
        if not isinstance(history, np.ndarray):
            raise TypeError(f"{name} must be an np.ndarray")
        if history.dtype != ARRAY_DTYPE:
            raise TypeError(f"{name} dtype must be float32")
        if history.shape != (grid.history_steps, 3):
            raise ValueError(
                f"{name} shape must be ({grid.history_steps}, 3)"
            )
        if not np.isfinite(history).all():
            raise ValueError(f"{name} must contain only finite values")
        footprints[object_id] = footprint_from_spec(
            scene_dynamic_specs[object_id]
        )

    context_ids = set(base_state.dynamic_object_ids)
    if not context_ids.issubset(object_ids):
        raise ValueError("BaseState context objects must be present in the scene")
    for object_id in base_state.dynamic_object_ids:
        if not np.array_equal(
            scene_dynamic_history[object_id],
            base_state.visible_dynamic_object_history[object_id],
        ):
            raise ValueError(
                "BaseState context history must be exactly preserved"
            )
        if (
            scene_dynamic_specs[object_id]
            != base_state.visible_dynamic_object_specs[object_id]
        ):
            raise ValueError("BaseState context spec must be exactly preserved")
    return footprints


def _copy_static_occupancy(static_occupancy: Any, grid: GridSpec) -> np.ndarray:
    if not isinstance(static_occupancy, np.ndarray):
        raise TypeError("static_occupancy must be an np.ndarray")
    if static_occupancy.dtype != ARRAY_DTYPE:
        raise TypeError("static_occupancy dtype must be float32")
    if static_occupancy.shape != (grid.height, grid.width):
        raise ValueError("static_occupancy shape must match the grid")
    if not np.isfinite(static_occupancy).all():
        raise ValueError("static_occupancy must contain only finite values")
    if not np.isin(static_occupancy, (0.0, 1.0)).all():
        raise ValueError("static_occupancy must be binary")
    return np.array(static_occupancy, dtype=ARRAY_DTYPE, order="C", copy=True)


def _occupancy_digest(static_occupancy: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"float32:{static_occupancy.shape}".encode("ascii"))
    digest.update(static_occupancy.tobytes(order="C"))
    return digest.hexdigest()


def _sensor_config_digest(sensor_config: StructuralBlindSpot | None) -> str:
    payload = None if sensor_config is None else sensor_config.as_dict()
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def render_observation(
    base_state: BaseState,
    *,
    scene_dynamic_history: Mapping[str, np.ndarray],
    scene_dynamic_specs: Mapping[str, dict[str, object]],
    static_occupancy: np.ndarray,
    sensor_config: StructuralBlindSpot | None,
    config: Mapping[str, Any],
) -> RenderedObservation:
    """Render the frozen two-history/nine-state layout from scene history."""
    config_dict, grid = _validated_config_and_grid(config)
    footprints = _validate_scene(
        base_state,
        scene_dynamic_history,
        scene_dynamic_specs,
        grid,
    )
    static = _copy_static_occupancy(static_occupancy, grid)
    if sensor_config is not None and not isinstance(
        sensor_config, StructuralBlindSpot
    ):
        raise TypeError("sensor_config must be a StructuralBlindSpot or None")
    if base_state.static_map_local is not None and np.any(
        (base_state.static_map_local != 0.0) & (static == 0.0)
    ):
        raise ValueError(
            "BaseState static occupied cells must remain occupied"
        )

    dynamic_occupancy = np.zeros(
        (grid.history_steps, grid.height, grid.width), dtype=bool, order="C"
    )
    for object_id in sorted(footprints):
        footprint = footprints[object_id]
        for step, pose in enumerate(scene_dynamic_history[object_id]):
            dynamic_occupancy[step] |= rasterize_footprint(
                footprint,
                pose,
                grid,
            )
    total_occupancy = dynamic_occupancy | (static != 0.0)

    history = np.empty(
        (grid.history_steps, len(HISTORY_CHANNELS), grid.height, grid.width),
        dtype=ARRAY_DTYPE,
        order="C",
    )
    visibility_history = np.empty(
        (grid.history_steps, grid.height, grid.width), dtype=bool, order="C"
    )
    for step, robot_pose in enumerate(base_state.robot_history):
        if sensor_config is None:
            visible = raycast_visibility(
                total_occupancy[step],
                grid,
                sensor_pose=robot_pose,
            )
        else:
            visible = build_structural_visibility(
                total_occupancy[step],
                grid,
                sensor_pose=robot_pose,
                blind_spot=sensor_config,
            )
        visibility_history[step] = visible
        history[
            step, HISTORY_CHANNELS.index("past_dynamic_occupancy")
        ] = dynamic_occupancy[step] & visible
        history[step, HISTORY_CHANNELS.index("past_visible_mask")] = visible

    current_visible = visibility_history[-1]
    current_occupied = current_visible & total_occupancy[-1]
    current_free = current_visible & ~current_occupied
    state = np.empty(
        (len(STATE_CHANNELS), grid.height, grid.width),
        dtype=ARRAY_DTYPE,
        order="C",
    )
    state[STATE_CHANNELS.index("current_visible_free")] = current_free
    state[STATE_CHANNELS.index("current_visible_occupied")] = current_occupied
    state[STATE_CHANNELS.index("current_unobservable_mask")] = ~current_visible
    last_seen_dynamic = state[STATE_CHANNELS.index("last_seen_occupancy")]
    last_seen_dynamic.fill(0.0)

    last_visible_index = np.full(
        (grid.height, grid.width), -1, dtype=np.int64
    )
    for step, visible in enumerate(visibility_history):
        last_seen_dynamic[visible] = dynamic_occupancy[step][visible]
        last_visible_index[visible] = step
    age_config = config_dict["age_map"]
    age = np.full(
        (grid.height, grid.width),
        np.float32(age_config["never_seen_value"]),
        dtype=ARRAY_DTYPE,
    )
    seen = last_visible_index >= 0
    elapsed = (grid.history_steps - 1 - last_visible_index[seen]) * float(
        config_dict["bev"]["history_dt_s"]
    )
    age[seen] = np.minimum(elapsed / float(age_config["a_max_s"]), 1.0)
    age[current_visible] = np.float32(age_config["visible_value"])
    state[STATE_CHANNELS.index("occlusion_age_map")] = age
    state[STATE_CHANNELS.index("static_obstacle_map")] = static

    robot_footprint = RectangleFootprint(
        float(config_dict["robot"]["length_m"]),
        float(config_dict["robot"]["width_m"]),
    )
    state[STATE_CHANNELS.index("robot_footprint")] = rasterize_footprint(
        robot_footprint,
        base_state.robot_history[-1],
        grid,
    )
    state[STATE_CHANNELS.index("robot_velocity_channel")].fill(
        base_state.robot_state[0]
    )
    state[STATE_CHANNELS.index("robot_yaw_rate_channel")].fill(
        base_state.robot_state[1]
    )

    return RenderedObservation(
        bev_history=history,
        state_channels=state,
        metadata={
            "renderer_layout_version": RENDERER_LAYOUT_VERSION,
            "base_state_id": base_state.state_id,
            "sensor_config_digest": _sensor_config_digest(sensor_config),
            "static_occupancy_digest": _occupancy_digest(static),
        },
    )
