import numpy as np
import pytest

from src.geometry.transforms import (
    global_to_local,
    interpolate_poses,
    local_to_global,
    transform_poses_global_to_local,
    transform_poses_local_to_global,
    unwrap_yaws,
    wrap_angle,
)


def test_wrap_angle_uses_half_open_interval_at_pi() -> None:
    angles = np.array([-3.0 * np.pi, -np.pi, np.pi, 3.0 * np.pi])

    wrapped = wrap_angle(angles)

    np.testing.assert_allclose(wrapped, -np.pi)
    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    assert wrap_angle(np.pi) == pytest.approx(-np.pi)


def test_wrap_angle_keeps_representable_values_adjacent_to_pi_half_open() -> None:
    angles = np.array(
        [
            np.nextafter(np.pi, np.inf),
            np.nextafter(np.pi, -np.inf),
            np.nextafter(-np.pi, np.inf),
            np.nextafter(-np.pi, -np.inf),
        ]
    )

    wrapped = wrap_angle(angles)

    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    np.testing.assert_allclose(np.sin(wrapped), np.sin(angles), atol=1e-15)
    np.testing.assert_allclose(np.cos(wrapped), np.cos(angles), atol=1e-15)


def test_point_transform_round_trip_and_pairwise_distances() -> None:
    points_global = np.array(
        [
            [[2.0, -1.0], [4.0, 3.0]],
            [[-0.5, 2.5], [8.0, -4.0]],
        ]
    )
    reference_pose = np.array([1.25, -2.5, 0.73])

    points_local = global_to_local(points_global, reference_pose)
    reconstructed = local_to_global(points_local, reference_pose)

    assert points_local.shape == points_global.shape
    assert points_local.dtype == np.float64
    np.testing.assert_allclose(reconstructed, points_global, atol=1e-8)
    global_distances = np.linalg.norm(points_global[..., 0, :] - points_global[..., 1, :], axis=-1)
    local_distances = np.linalg.norm(points_local[..., 0, :] - points_local[..., 1, :], axis=-1)
    np.testing.assert_allclose(local_distances, global_distances, atol=1e-10)


def test_global_to_local_uses_forward_x_and_left_y_axes() -> None:
    reference_pose = np.array([10.0, 20.0, np.pi / 2.0])
    points_global = np.array([[10.0, 21.0], [9.0, 20.0], [10.0, 20.0]])

    points_local = global_to_local(points_global, reference_pose)

    np.testing.assert_allclose(points_local, [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], atol=1e-12)


def test_pose_transform_round_trip_meets_error_tolerances() -> None:
    poses_global = np.array(
        [
            [2.0, -1.0, np.pi - 2e-6],
            [4.0, 3.0, -np.pi + 3e-6],
            [-0.5, 2.5, 1.7],
        ]
    )
    reference_pose = np.array([1.25, -2.5, -2.4])

    poses_local = transform_poses_global_to_local(poses_global, reference_pose)
    reconstructed = transform_poses_local_to_global(poses_local, reference_pose)

    position_error = np.linalg.norm(reconstructed[:, :2] - poses_global[:, :2], axis=1)
    angle_error = np.abs(wrap_angle(reconstructed[:, 2] - poses_global[:, 2]))
    assert np.max(position_error) < 1e-4
    assert np.max(angle_error) < 1e-5
    assert poses_local.dtype == np.float64


@pytest.mark.parametrize(
    ("function", "values", "reference"),
    [
        (global_to_local, np.ones((3, 3)), np.zeros(3)),
        (local_to_global, np.ones(2), np.zeros((1, 3))),
        (transform_poses_global_to_local, np.ones((2, 2)), np.zeros(3)),
        (transform_poses_local_to_global, np.ones(3), np.zeros(2)),
    ],
)
def test_transform_functions_reject_invalid_shapes(function, values, reference) -> None:
    with pytest.raises(ValueError, match="shape"):
        function(values, reference)


@pytest.mark.parametrize(
    ("function", "values", "reference"),
    [
        (global_to_local, [[np.nan, 0.0]], [0.0, 0.0, 0.0]),
        (local_to_global, [[0.0, np.inf]], [0.0, 0.0, 0.0]),
        (transform_poses_global_to_local, [[0.0, 0.0, 0.0]], [0.0, np.inf, 0.0]),
        (transform_poses_local_to_global, [[0.0, np.nan, 0.0]], [0.0, 0.0, 0.0]),
    ],
)
def test_transform_functions_reject_non_finite_inputs(function, values, reference) -> None:
    with pytest.raises(ValueError, match="finite"):
        function(values, reference)


def test_yaw_unwrap_and_pose_interpolation_cross_pi_continuously() -> None:
    source_times = np.array([0.0, 2.0])
    source_poses = np.array(
        [
            [0.0, 2.0, np.deg2rad(179.0)],
            [4.0, -2.0, np.deg2rad(-179.0)],
        ]
    )
    query_times = np.array([0.0, 1.0, 2.0])

    result = interpolate_poses(source_times, source_poses, query_times)

    assert result.shape == (3, 3)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result[:, :2], [[0.0, 2.0], [2.0, 0.0], [4.0, -2.0]])
    np.testing.assert_allclose(np.rad2deg(result[:, 2]), [179.0, 180.0, 181.0], atol=1e-10)
    assert np.all(np.abs(np.diff(result[:, 2])) < np.pi)
    np.testing.assert_allclose(unwrap_yaws(source_poses[:, 2]), result[[0, 2], 2])


@pytest.mark.parametrize(
    ("source_times", "query_times", "message"),
    [
        ([0.0, 0.0, 1.0], [0.5], "strictly increasing"),
        ([0.0, 2.0, 1.0], [0.5], "strictly increasing"),
        ([0.0, np.nan, 1.0], [0.5], "finite"),
        ([0.0, 1.0, 2.0], [1.0, 1.0], "strictly increasing"),
        ([0.0, 1.0, 2.0], [1.5, 1.0], "strictly increasing"),
        ([0.0, 1.0, 2.0], [np.inf], "finite"),
        ([0.0, 1.0, 2.0], [-0.1], "source range"),
        ([0.0, 1.0, 2.0], [2.1], "source range"),
    ],
)
def test_interpolate_poses_rejects_invalid_timestamps(source_times, query_times, message) -> None:
    source_poses = np.zeros((len(source_times), 3))

    with pytest.raises(ValueError, match=message):
        interpolate_poses(source_times, source_poses, query_times)


def test_interpolate_poses_rejects_pose_shape_and_non_finite_yaw() -> None:
    with pytest.raises(ValueError, match="shape"):
        interpolate_poses([0.0, 1.0], np.zeros((2, 2)), [0.5])
    with pytest.raises(ValueError, match="finite"):
        interpolate_poses([0.0, 1.0], [[0.0, 0.0, 0.0], [1.0, 1.0, np.nan]], [0.5])


def test_transform_and_interpolation_results_are_deterministic() -> None:
    poses = np.array([[0.0, 0.0, 3.0], [1.0, 2.0, -3.0]])
    reference = np.array([0.5, -0.25, 0.9])

    first_transform = transform_poses_global_to_local(poses, reference)
    second_transform = transform_poses_global_to_local(poses, reference)
    first_interpolation = interpolate_poses([0.0, 1.0], poses, [0.25, 0.75])
    second_interpolation = interpolate_poses([0.0, 1.0], poses, [0.25, 0.75])

    assert np.array_equal(first_transform, second_transform)
    assert np.array_equal(first_interpolation, second_interpolation)
