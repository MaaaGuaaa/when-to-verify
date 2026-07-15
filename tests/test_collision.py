import warnings

import numpy as np
import pytest

from src.geometry.collision import (
    first_collision_index,
    intersects,
    segments_intersect,
    signed_clearance,
    trajectory_signed_clearances,
)
from src.geometry.footprints import CircleFootprint, RectangleFootprint


def test_circle_circle_clearance_matches_analytic_toy_result_and_is_symmetric():
    a = CircleFootprint(0.30)
    b = CircleFootprint(0.30)
    pose_a = [0.0, 0.0, 0.0]

    for distance, expected in [(1.0, 0.4), (0.6, 0.0), (0.2, -0.4)]:
        pose_b = [distance, 0.0, np.pi]
        assert signed_clearance(a, pose_a, b, pose_b) == pytest.approx(expected)
        assert signed_clearance(b, pose_b, a, pose_a) == pytest.approx(expected)

    assert intersects(a, pose_a, b, [0.6, 0.0, 0.0])
    assert not intersects(a, pose_a, b, [0.61, 0.0, 0.0])
    assert intersects(a, pose_a, b, [0.61, 0.0, 0.0], atol=0.011)


def test_circle_rectangle_clearance_handles_outside_touching_inside_and_rotation():
    circle = CircleFootprint(0.5)
    rectangle = RectangleFootprint(4.0, 2.0)
    rect_pose = [0.0, 0.0, np.pi / 2.0]

    for center, expected in [([0.0, 3.0, 0.0], 0.5), ([0.0, 2.5, 0.0], 0.0), ([0.0, 0.0, 0.0], -1.5)]:
        actual = signed_clearance(circle, center, rectangle, rect_pose)
        reverse = signed_clearance(rectangle, rect_pose, circle, center)
        assert actual == pytest.approx(expected)
        assert reverse == pytest.approx(expected)


def test_rectangle_clearance_uses_euclidean_separation_and_containment_mtd():
    unit_box = RectangleFootprint(2.0, 2.0)
    assert signed_clearance(unit_box, [0.0, 0.0, 0.0], unit_box, [3.0, 3.0, 0.0]) == pytest.approx(
        np.sqrt(2.0)
    )
    assert signed_clearance(unit_box, [0.0, 0.0, 0.0], unit_box, [2.0, 0.0, 0.0]) == pytest.approx(0.0)

    outer = RectangleFootprint(4.0, 4.0)
    inner = RectangleFootprint(2.0, 2.0)
    assert signed_clearance(outer, [0.0, 0.0, 0.0], inner, [0.0, 0.0, 0.0]) == pytest.approx(-3.0)


def test_rotated_rectangles_are_symmetric_and_pi_yaw_is_equivalent():
    rectangle = RectangleFootprint(2.0, 1.0)
    pose_a = [0.0, 0.0, np.pi / 4.0]
    pose_b = [1.0, 0.0, -np.pi / 4.0]

    forward = signed_clearance(rectangle, pose_a, rectangle, pose_b)
    reverse = signed_clearance(rectangle, pose_b, rectangle, pose_a)
    assert forward == pytest.approx(reverse, abs=1e-14)
    assert forward < 0.0
    assert signed_clearance(rectangle, [0.0, 0.0, np.pi], rectangle, [3.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert signed_clearance(rectangle, [0.0, 0.0, -np.pi], rectangle, [3.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_rotated_rectangles_snap_roundoff_touch_to_zero_but_keep_real_gap():
    rectangle = RectangleFootprint(2.0, 1.0)
    yaw = 0.3
    local_x = np.array([np.cos(yaw), np.sin(yaw)])
    pose_a = np.array([0.0, 0.0, yaw])
    touching_pose = np.array([*(2.0 * local_x), yaw])
    separated_pose = np.array([*((2.0 + 1e-10) * local_x), yaw])

    assert signed_clearance(rectangle, pose_a, rectangle, touching_pose) == 0.0
    assert intersects(rectangle, pose_a, rectangle, touching_pose)
    assert signed_clearance(rectangle, pose_a, rectangle, separated_pose) > 0.0
    assert not intersects(rectangle, pose_a, rectangle, separated_pose)


def test_circle_clearance_remains_finite_at_large_and_small_scales():
    large_circle = CircleFootprint(1e190)
    large_clearance = signed_clearance(
        large_circle,
        [0.0, 0.0, 0.0],
        large_circle,
        [1e200, 1e200, 0.0],
    )
    assert np.isfinite(large_clearance)
    assert large_clearance == np.hypot(1e200, 1e200) - 2e190

    small_circle = CircleFootprint(1e-210)
    small_clearance = signed_clearance(
        small_circle,
        [0.0, 0.0, 0.0],
        small_circle,
        [1e-200, 1e-200, 0.0],
    )
    assert small_clearance == np.hypot(1e-200, 1e-200) - 2e-210


def test_rectangle_clearance_preserves_huge_finite_gap_without_roundoff_snap():
    rectangle = RectangleFootprint(1e308, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        clearance = signed_clearance(
            rectangle, [0.0, 0.0, 0.0], rectangle, [1.5e308, 0.0, 0.0]
        )
        collision = intersects(
            rectangle, [0.0, 0.0, 0.0], rectangle, [1.5e308, 0.0, 0.0]
        )

    assert np.isfinite(clearance)
    assert clearance == pytest.approx(5e307)
    assert not collision


def test_rectangle_clearance_opposite_huge_centers_never_false_collides_or_warns():
    rectangle = RectangleFootprint(1.0, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        clearance = signed_clearance(
            rectangle, [-1e308, 0.0, 0.0], rectangle, [1e308, 0.0, 0.0]
        )
        collision = intersects(
            rectangle, [-1e308, 0.0, 0.0], rectangle, [1e308, 0.0, 0.0]
        )

    assert not np.isnan(clearance)
    assert clearance > 0.0
    assert not collision


def test_rectangle_clearance_small_coordinates_is_finite_without_warnings():
    small_rectangle = RectangleFootprint(0.6, 0.4)
    cell_rectangle = RectangleFootprint(1.0, 1.0)
    pose = [0.2, -0.2, 0.0]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        clearance = signed_clearance(small_rectangle, pose, cell_rectangle, pose)

    assert np.isfinite(clearance)
    assert clearance == pytest.approx(-0.7)


@pytest.mark.parametrize(
    ("start_a", "end_a", "start_b", "end_b", "expected"),
    [
        ([0, 0], [2, 2], [0, 2], [2, 0], True),
        ([0, 0], [1, 0], [1, 0], [2, 1], True),
        ([0, 0], [3, 0], [1, 0], [4, 0], True),
        ([0, 0], [1, 0], [0, 1], [1, 1], False),
        ([1, 1], [1, 1], [0, 1], [2, 1], True),
        ([1, 2], [1, 2], [0, 1], [2, 1], False),
    ],
)
def test_segments_intersect_covers_crossing_touch_collinear_parallel_and_degenerate(
    start_a, end_a, start_b, end_b, expected
):
    assert segments_intersect(start_a, end_a, start_b, end_b) is expected


def test_segments_intersect_is_stable_at_large_and_small_finite_scales():
    large_a = ([0.0, 0.0], [1e200, 0.0])
    large_b = ([0.0, 1e190], [1e200, 1e190])
    assert not segments_intersect(*large_a, *large_b)
    assert segments_intersect(*large_a, *large_b, atol=1e191)

    small_a = ([0.0, 0.0], [1e-200, 0.0])
    small_b = ([0.0, 1e-210], [1e-200, 1e-210])
    assert not segments_intersect(*small_a, *small_b)
    assert segments_intersect(*small_a, *small_b, atol=1e-209)

    assert segments_intersect(
        [0.0, 0.0], [1e200, 1e200], [0.0, 1e200], [1e200, 0.0]
    )


def test_fifteen_frame_trajectory_returns_float64_clearances_and_first_collision():
    footprint = CircleFootprint(0.30)
    poses_a = np.zeros((15, 3), dtype=np.float64)
    poses_b = np.zeros((15, 3), dtype=np.float64)
    poses_b[:, 0] = 1.3 - 0.1 * np.arange(15)

    clearances = trajectory_signed_clearances(footprint, poses_a, footprint, poses_b)

    expected = np.abs(poses_b[:, 0]) - 0.60
    np.testing.assert_allclose(clearances, expected, atol=1e-15)
    assert clearances.shape == (15,)
    assert clearances.dtype == np.float64
    assert first_collision_index(clearances, atol=1e-14) == 7
    assert clearances[7] == pytest.approx(0.0, abs=1e-14)


def test_trajectory_is_temporally_safe_when_spatial_paths_cross_at_different_times():
    footprint = CircleFootprint(0.30)
    poses_a = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    poses_b = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -3.0, 0.0]])

    clearances = trajectory_signed_clearances(footprint, poses_a, footprint, poses_b)

    assert np.all(clearances > 0.0)
    assert first_collision_index(clearances) is None


def test_empty_trajectory_and_first_collision_edge_cases():
    footprint = CircleFootprint(0.3)
    result = trajectory_signed_clearances(
        footprint, np.empty((0, 3)), footprint, np.empty((0, 3))
    )
    assert result.shape == (0,)
    assert result.dtype == np.float64
    assert first_collision_index(result) is None
    assert first_collision_index(np.array([0.2, 0.01, -0.1]), atol=0.01) == 1


@pytest.mark.parametrize("bad_pose", [[0.0, 0.0], [[0.0, 0.0, 0.0]], [0.0, np.nan, 0.0]])
def test_collision_rejects_invalid_pose(bad_pose):
    circle = CircleFootprint(0.3)
    with pytest.raises(ValueError):
        signed_clearance(circle, bad_pose, circle, [0.0, 0.0, 0.0])


@pytest.mark.parametrize("atol", [True, -1.0, np.nan, np.inf, "0"])
def test_collision_apis_reject_invalid_atol(atol):
    circle = CircleFootprint(0.3)
    with pytest.raises((TypeError, ValueError)):
        intersects(circle, [0, 0, 0], circle, [1, 0, 0], atol=atol)
    with pytest.raises((TypeError, ValueError)):
        segments_intersect([0, 0], [1, 0], [0, 1], [1, 1], atol=atol)
    with pytest.raises((TypeError, ValueError)):
        first_collision_index([1.0], atol=atol)


def test_collision_apis_reject_invalid_shapes_arrays_and_nonfinite_values():
    circle = CircleFootprint(0.3)
    with pytest.raises(TypeError):
        signed_clearance(object(), [0, 0, 0], circle, [0, 0, 0])
    with pytest.raises(ValueError):
        segments_intersect([0, 0, 0], [1, 0], [0, 1], [1, 1])
    with pytest.raises(ValueError):
        segments_intersect([0, 0], [1, np.inf], [0, 1], [1, 1])
    with pytest.raises(ValueError):
        trajectory_signed_clearances(circle, np.zeros((2, 3)), circle, np.zeros((3, 3)))
    with pytest.raises(ValueError):
        trajectory_signed_clearances(circle, np.zeros((2, 2)), circle, np.zeros((2, 3)))
    with pytest.raises(ValueError):
        trajectory_signed_clearances(circle, np.array([[0.0, 0.0, np.nan]]), circle, np.zeros((1, 3)))
    with pytest.raises(ValueError):
        first_collision_index([[0.0]])
    with pytest.raises(ValueError):
        first_collision_index([0.0, np.inf])
