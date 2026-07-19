"""Behavioral tests for structural FOV/range/blind-sector blind spots."""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts import build_grid_spec
from src.generation.structural_blindspot import (
    StructuralBlindSpot,
    build_structural_visibility,
    footprint_visibility_sequence,
    has_continuous_emergence,
)
from src.geometry import CircleFootprint, world_to_grid
from src.utils.config import load_config


def test_structural_visibility_applies_fov_range_and_blind_sector() -> None:
    grid = build_grid_spec(load_config())
    occupancy = np.zeros((grid.height, grid.width), dtype=np.float32)
    blind_spot = StructuralBlindSpot(
        forward_fov_deg=180.0,
        range_m=6.0,
        blind_sectors=({"center_deg": 90.0, "width_deg": 40.0},),
    )

    visible = build_structural_visibility(
        occupancy,
        grid,
        sensor_pose=np.zeros(3, dtype=np.float32),
        blind_spot=blind_spot,
    )
    indices = world_to_grid(
        np.asarray(
            [
                [2.0, 0.0],   # in front
                [-2.0, 0.0],  # behind the forward FOV
                [0.0, 2.0],   # explicitly blinded sector
                [7.0, 0.0],   # beyond range
            ],
            dtype=np.float32,
        ),
        grid,
    )

    assert bool(visible[tuple(indices[0])])
    assert not bool(visible[tuple(indices[1])])
    assert not bool(visible[tuple(indices[2])])
    assert not bool(visible[tuple(indices[3])])


def test_hidden_circle_emerges_continuously_from_structural_boundary() -> None:
    grid = build_grid_spec(load_config())
    occupancy = np.zeros((grid.height, grid.width), dtype=np.float32)
    blind_spot = StructuralBlindSpot(
        forward_fov_deg=160.0,
        range_m=6.0,
        blind_sectors=({"center_deg": -60.0, "width_deg": 80.0},),
    )
    visible = build_structural_visibility(
        occupancy,
        grid,
        sensor_pose=(0.0, 0.0, 0.0),
        blind_spot=blind_spot,
    )
    poses = np.asarray(
        [
            [1.0, -2.0, np.pi / 2.0],
            [1.0, -1.6, np.pi / 2.0],
            [1.0, -1.2, np.pi / 2.0],
            [1.0, -0.6, np.pi / 2.0],
            [1.0, 0.0, np.pi / 2.0],
            [1.0, 0.6, np.pi / 2.0],
        ],
        dtype=np.float32,
    )

    sequence = footprint_visibility_sequence(
        CircleFootprint(0.20), poses, visible, grid
    )

    assert sequence.dtype == np.bool_
    assert not bool(sequence[0])
    assert bool(sequence[-1])
    assert has_continuous_emergence(sequence, min_visible_frames=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forward_fov_deg": 0.0, "range_m": 6.0},
        {"forward_fov_deg": 361.0, "range_m": 6.0},
        {"forward_fov_deg": 180.0, "range_m": 0.0},
        {
            "forward_fov_deg": 180.0,
            "range_m": 6.0,
            "blind_sectors": ({"center_deg": 0.0, "width_deg": 0.0},),
        },
    ],
)
def test_structural_blindspot_rejects_nonphysical_parameters(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        StructuralBlindSpot(**kwargs)
