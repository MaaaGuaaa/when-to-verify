from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.contracts import GridSpec
from src.geometry.static_occluders import (
    CircleOccluder,
    RectangleOccluder,
    inflate_occluder,
    occluder_bounds,
    occluder_payload,
    point_signed_distance,
    rasterize_occluder,
    segment_intersects_occluder,
)


@pytest.fixture
def grid() -> GridSpec:
    return GridSpec(
        height=16,
        width=16,
        history_steps=1,
        future_steps=1,
        resolution_m=0.5,
    )


def test_occluders_are_immutable_and_have_stable_payloads() -> None:
    rectangle = RectangleOccluder(
        occluder_id="wall-1",
        semantic_type="wall",
        pose=np.asarray([1.0, -2.0, np.pi / 2.0], dtype=np.float64),
        length_m=4.0,
        width_m=2.0,
    )
    circle = CircleOccluder(
        occluder_id="tree-1",
        semantic_type="tree_trunk",
        center_xy=np.asarray([0.5, 0.25], dtype=np.float64),
        radius_m=0.3,
    )

    assert rectangle.pose.dtype == np.float64
    assert not rectangle.pose.flags.writeable
    assert occluder_payload(rectangle) == {
        "occluder_id": "wall-1",
        "semantic_type": "wall",
        "shape": "rectangle",
        "pose": [1.0, -2.0, pytest.approx(np.pi / 2.0)],
        "length_m": 4.0,
        "width_m": 2.0,
    }
    assert occluder_payload(circle) == {
        "occluder_id": "tree-1",
        "semantic_type": "tree_trunk",
        "shape": "circle",
        "center_xy": [0.5, 0.25],
        "radius_m": 0.3,
    }
    with pytest.raises(FrozenInstanceError):
        circle.radius_m = 1.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        rectangle.pose[0] = 0.0


def test_bounds_inflation_and_signed_distances_are_analytic() -> None:
    rectangle = RectangleOccluder(
        "wall-1",
        "wall",
        np.asarray([0.0, 0.0, np.pi / 2.0]),
        4.0,
        2.0,
    )
    circle = CircleOccluder("tree-1", "tree_trunk", np.asarray([0.0, 0.0]), 1.0)

    assert occluder_bounds(rectangle) == pytest.approx((-1.0, 1.0, -2.0, 2.0))
    assert occluder_bounds(inflate_occluder(rectangle, 0.5)) == pytest.approx(
        (-1.5, 1.5, -2.5, 2.5)
    )
    assert occluder_bounds(circle) == pytest.approx((-1.0, 1.0, -1.0, 1.0))
    assert occluder_bounds(inflate_occluder(circle, 0.5)) == pytest.approx(
        (-1.5, 1.5, -1.5, 1.5)
    )

    np.testing.assert_allclose(
        point_signed_distance(rectangle, [[0.0, 0.0], [0.0, 3.0], [1.0, 0.0]]),
        [-1.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(
        point_signed_distance(circle, [[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]]),
        [-1.0, 1.0, 0.0],
    )


def test_segment_intersection_handles_crossing_tangency_miss_degenerate_and_batching() -> None:
    rectangle = RectangleOccluder(
        "wall-1",
        "wall",
        np.asarray([0.0, 0.0, np.pi / 4.0]),
        2.0,
        1.0,
    )
    circle = CircleOccluder("tree-1", "tree_trunk", np.asarray([0.0, 0.0]), 1.0)
    starts = np.asarray(
        [[-3.0, 0.0], [-3.0, 1.5], [-3.0, 3.0], [0.0, 0.0]], dtype=np.float64
    )
    ends = np.asarray(
        [[3.0, 0.0], [3.0, 1.5], [3.0, 3.0], [0.0, 0.0]], dtype=np.float64
    )

    np.testing.assert_array_equal(
        segment_intersects_occluder(rectangle, starts, ends, epsilon_m=1e-9),
        [True, False, False, True],
    )
    np.testing.assert_array_equal(
        segment_intersects_occluder(circle, starts, ends, epsilon_m=1e-9),
        [True, False, False, True],
    )
    assert segment_intersects_occluder(
        circle,
        np.asarray([[-3.0, 1.0]]),
        np.asarray([[3.0, 1.0]]),
        epsilon_m=0.0,
    )[0]
    assert segment_intersects_occluder(
        circle,
        np.asarray([[-3.0, 1.0 + 1e-6]]),
        np.asarray([[3.0, 1.0 + 1e-6]]),
        epsilon_m=1e-5,
    )[0]


def test_rasterization_is_bounded_nonempty_and_deterministic(grid: GridSpec) -> None:
    occluders = (
        RectangleOccluder(
            "wall-1",
            "wall",
            np.asarray([0.0, 0.0, np.pi / 6.0]),
            2.0,
            0.5,
        ),
        CircleOccluder(
            "tree-1",
            "tree_trunk",
            np.asarray([1.0, 1.0]),
            0.6,
        ),
    )
    for occluder in occluders:
        first = rasterize_occluder(occluder, grid)
        second = rasterize_occluder(occluder, grid)
        assert first.shape == (grid.height, grid.width)
        assert first.dtype == np.bool_
        assert first.any()
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RectangleOccluder("", "wall", np.zeros(3), 1.0, 1.0),
        lambda: RectangleOccluder("wall", "wall", [0.0, 0.0], 1.0, 1.0),
        lambda: RectangleOccluder("wall", "wall", [0.0, 0.0, np.nan], 1.0, 1.0),
        lambda: RectangleOccluder("wall", "wall", np.zeros(3), 0.0, 1.0),
        lambda: CircleOccluder("", "tree_trunk", np.zeros(2), 1.0),
        lambda: CircleOccluder("tree", "tree_trunk", [0.0, np.inf], 1.0),
        lambda: CircleOccluder("tree", "tree_trunk", np.zeros(2), True),
    ],
)
def test_occluders_reject_invalid_values(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()
