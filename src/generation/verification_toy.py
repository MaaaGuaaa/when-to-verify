"""Small deterministic mixed-footprint scene used as the SOP11–14 oracle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.contracts import GridSpec
from src.geometry import (
    CircleFootprint,
    RectangleFootprint,
    rasterize_footprint,
    raycast_visibility,
)


@dataclass(frozen=True)
class VerificationToyWorld:
    grid: GridSpec
    static_occupancy: np.ndarray
    current_visible_mask: np.ndarray
    current_age_map: np.ndarray
    dynamic_current_poses: dict[str, np.ndarray]
    dynamic_future_poses: dict[str, np.ndarray]
    dynamic_specs: dict[str, dict[str, object]]
    critical_mask: np.ndarray
    irrelevant_mask: np.ndarray


def build_verification_toy_world() -> VerificationToyWorld:
    grid = GridSpec(
        height=80,
        width=80,
        history_steps=8,
        future_steps=32,
        resolution_m=0.1,
    )
    static = rasterize_footprint(
        RectangleFootprint(0.20, 0.25),
        np.asarray([1.10, -0.70, 0.0], dtype=np.float32),
        grid,
    ).astype(np.float32)
    critical_pose = np.asarray([2.40, -1.05, 0.0], dtype=np.float32)
    irrelevant_pose = np.asarray([2.00, 1.20, 0.0], dtype=np.float32)
    current = {
        "critical_cart": critical_pose.copy(),
        "irrelevant_person": irrelevant_pose.copy(),
    }
    critical_future = np.tile(critical_pose, (grid.future_steps, 1)).astype(
        np.float32
    )
    motion_start = 5
    critical_future[motion_start:, 0] = np.linspace(
        critical_pose[0],
        2.05,
        grid.future_steps - motion_start,
        dtype=np.float32,
    )
    crossing_start = 11
    critical_future[crossing_start:, 1] = np.linspace(
        critical_pose[1],
        0.0,
        grid.future_steps - crossing_start,
        dtype=np.float32,
    )
    future = {
        "critical_cart": critical_future,
        "irrelevant_person": np.tile(
            irrelevant_pose, (grid.future_steps, 1)
        ).astype(np.float32),
    }
    specs = {
        "critical_cart": {
            "object_type": "carried_object",
            "footprint": {
                "kind": "rectangle",
                "length_m": 0.80,
                "width_m": 0.25,
            },
        },
        "irrelevant_person": {
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": 0.30},
        },
    }
    critical_mask = rasterize_footprint(
        RectangleFootprint(0.80, 0.25), critical_pose, grid
    )
    irrelevant_mask = rasterize_footprint(
        CircleFootprint(0.30), irrelevant_pose, grid
    )
    current_visible = raycast_visibility(
        (static != 0.0) | critical_mask | irrelevant_mask,
        grid,
        sensor_pose=np.zeros(3, dtype=np.float32),
        fov_rad=2.0 * np.pi,
        max_range_m=4.0,
    )
    age = np.ones((grid.height, grid.width), dtype=np.float32)
    age[current_visible] = 0.0
    return VerificationToyWorld(
        grid=grid,
        static_occupancy=static,
        current_visible_mask=current_visible,
        current_age_map=age,
        dynamic_current_poses=current,
        dynamic_future_poses=future,
        dynamic_specs=specs,
        critical_mask=critical_mask,
        irrelevant_mask=irrelevant_mask,
    )


__all__ = ("VerificationToyWorld", "build_verification_toy_world")
