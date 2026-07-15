from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.geometry.footprints import (
    CircleFootprint,
    RectangleFootprint,
    footprint_aabb,
    footprint_vertices,
    inflate_footprint,
)


@pytest.mark.parametrize("value", [True, 0.0, -1.0, np.nan, np.inf, "1", 1 + 0j])
def test_footprints_reject_invalid_dimensions(value):
    with pytest.raises((TypeError, ValueError)):
        CircleFootprint(value)
    with pytest.raises((TypeError, ValueError)):
        RectangleFootprint(value, 0.5)
    with pytest.raises((TypeError, ValueError)):
        RectangleFootprint(0.7, value)


def test_footprints_are_frozen_and_store_float_dimensions():
    circle = CircleFootprint(1)
    rectangle = RectangleFootprint(2, 3)

    assert circle.radius_m == 1.0
    assert rectangle == RectangleFootprint(length_m=2.0, width_m=3.0)
    with pytest.raises(FrozenInstanceError):
        circle.radius_m = 2.0


def test_inflate_footprint_returns_new_objects_without_mutating_inputs():
    circle = CircleFootprint(0.30)
    rectangle = RectangleFootprint(0.70, 0.55)

    inflated_circle = inflate_footprint(circle, 0.15)
    inflated_rectangle = inflate_footprint(rectangle, 0.15)

    assert inflated_circle.radius_m == pytest.approx(0.45)
    assert inflated_rectangle.length_m == pytest.approx(1.00)
    assert inflated_rectangle.width_m == pytest.approx(0.85)
    assert inflated_circle is not circle
    assert inflated_rectangle is not rectangle
    assert circle == CircleFootprint(0.30)
    assert rectangle == RectangleFootprint(0.70, 0.55)


@pytest.mark.parametrize("margin", [True, -0.1, np.nan, np.inf, "0.1"])
def test_inflate_footprint_rejects_invalid_margin(margin):
    with pytest.raises((TypeError, ValueError)):
        inflate_footprint(CircleFootprint(0.3), margin)


def test_inflate_footprint_rejects_unknown_shape():
    with pytest.raises(TypeError):
        inflate_footprint(object(), 0.1)


def test_rectangle_vertices_are_rotated_translated_float64_and_deterministic():
    rectangle = RectangleFootprint(4.0, 2.0)
    pose = np.array([3.0, -1.0, np.pi / 2.0])
    expected = np.array(
        [[4.0, -3.0], [4.0, 1.0], [2.0, 1.0], [2.0, -3.0]],
        dtype=np.float64,
    )

    first = footprint_vertices(rectangle, pose)
    second = footprint_vertices(rectangle, pose)

    np.testing.assert_allclose(first, expected, atol=1e-14)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 2)
    assert first.dtype == np.float64


def test_footprint_aabb_supports_circle_and_rotated_rectangle():
    assert footprint_aabb(CircleFootprint(0.5), [2.0, -1.0, np.pi]) == (
        1.5,
        2.5,
        -1.5,
        -0.5,
    )
    np.testing.assert_allclose(
        footprint_aabb(RectangleFootprint(4.0, 2.0), [3.0, -1.0, np.pi / 2.0]),
        (2.0, 4.0, -3.0, 1.0),
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "pose",
    [[0.0, 0.0], [[0.0, 0.0, 0.0]], [0.0, 0.0, np.nan], [0.0, np.inf, 0.0]],
)
def test_geometry_helpers_reject_invalid_pose(pose):
    rectangle = RectangleFootprint(1.0, 1.0)
    with pytest.raises(ValueError):
        footprint_vertices(rectangle, pose)
    with pytest.raises(ValueError):
        footprint_aabb(rectangle, pose)


def test_vertices_reject_circle_and_helpers_reject_unknown_shape():
    with pytest.raises(TypeError):
        footprint_vertices(CircleFootprint(1.0), [0.0, 0.0, 0.0])
    with pytest.raises(TypeError):
        footprint_aabb(object(), [0.0, 0.0, 0.0])
