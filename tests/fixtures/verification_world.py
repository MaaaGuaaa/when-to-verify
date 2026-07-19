"""Independent hand-checkable scene for SOP11–14 verification tests."""

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
        future_steps=15,
        resolution_m=0.1,
    )
    static = np.zeros((grid.height, grid.width), dtype=np.float32)
    current_visible = raycast_visibility(
        static,
        grid,
        sensor_pose=np.zeros(3, dtype=np.float32),
        fov_rad=np.deg2rad(20.0),
        max_range_m=4.0,
    )
    age = np.ones((grid.height, grid.width), dtype=np.float32)
    age[current_visible] = 0.0

    critical_pose = np.asarray([2.50, 0.85, np.pi / 4.0], dtype=np.float32)
    irrelevant_pose = np.asarray([2.50, -0.85, 0.0], dtype=np.float32)
    current = {
        "critical_cart": critical_pose.copy(),
        "irrelevant_person": irrelevant_pose.copy(),
    }
    future = {
        object_id: np.tile(pose, (grid.future_steps, 1)).astype(np.float32)
        for object_id, pose in current.items()
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
