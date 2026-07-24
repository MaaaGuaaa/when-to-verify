import numpy as np
import pytest

from src.contracts import GridSpec
from src.geometry.raycasting import (
    raycast_candidate_visibility,
    raycast_visibility,
)


@pytest.fixture
def grid() -> GridSpec:
    return GridSpec(height=5, width=5, history_steps=1, future_steps=1, resolution_m=1.0)


def test_empty_map_visibility_equals_full_fov_and_range_candidates(grid: GridSpec) -> None:
    occupancy = np.zeros((5, 5), dtype=bool)

    full = raycast_visibility(occupancy, grid)
    ranged = raycast_visibility(occupancy, grid, max_range_m=1.0)

    np.testing.assert_array_equal(full, np.ones((5, 5), dtype=bool))
    expected_range = np.zeros((5, 5), dtype=bool)
    expected_range[2, 2] = True
    expected_range[1, 2] = expected_range[3, 2] = True
    expected_range[2, 1] = expected_range[2, 3] = True
    np.testing.assert_array_equal(ranged, expected_range)
    assert full.dtype == np.bool_


def test_zero_fov_keeps_only_sensor_and_cells_on_positive_x_bearing(grid: GridSpec) -> None:
    visible = raycast_visibility(
        np.zeros((5, 5), dtype=bool), grid, fov_rad=0.0
    )
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2:] = True

    np.testing.assert_array_equal(visible, expected)


def test_explicit_full_fov_matches_default(grid: GridSpec) -> None:
    occupancy = np.zeros((5, 5), dtype=bool)
    occupancy[2, 3] = True

    default = raycast_visibility(occupancy, grid)
    explicit = raycast_visibility(occupancy, grid, fov_rad=2.0 * np.pi)

    np.testing.assert_array_equal(explicit, default)


def test_zero_range_at_cell_center_keeps_only_sensor_cell(grid: GridSpec) -> None:
    visible = raycast_visibility(
        np.zeros((5, 5), dtype=bool), grid, max_range_m=0.0
    )
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = True

    np.testing.assert_array_equal(visible, expected)


def test_noncentral_sensor_cell_is_excluded_when_no_cell_center_is_in_zero_range(
    grid: GridSpec,
) -> None:
    visible = raycast_visibility(
        np.zeros((5, 5), dtype=bool),
        grid,
        sensor_pose=(0.49, 0.49, 0.0),
        max_range_m=0.0,
    )

    np.testing.assert_array_equal(visible, np.zeros((5, 5), dtype=bool))


def test_noncentral_sensor_cell_is_not_forced_into_back_facing_zero_fov(
    grid: GridSpec,
) -> None:
    visible = raycast_visibility(
        np.zeros((5, 5), dtype=bool),
        grid,
        sensor_pose=(0.49, 0.49, np.pi / 4.0),
        fov_rad=0.0,
    )

    assert not visible[2, 2]


def test_sensor_at_minimum_grid_boundary_sees_full_empty_map(grid: GridSpec) -> None:
    x_min = -0.5 * grid.width * grid.resolution_m
    y_min = -0.5 * grid.height * grid.resolution_m

    visible = raycast_visibility(
        np.zeros((5, 5), dtype=bool),
        grid,
        sensor_pose=(x_min, y_min, 0.0),
    )

    np.testing.assert_array_equal(visible, np.ones((5, 5), dtype=bool))


def test_fov_is_yaw_centred_and_includes_exact_angular_edges(grid: GridSpec) -> None:
    visible = raycast_visibility(
        np.zeros((5, 5)), grid, sensor_pose=(0.0, 0.0, 0.0), fov_rad=np.pi / 2.0
    )
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2:] = True
    expected[1, 3:] = True
    expected[0, 4] = True
    expected[3, 3:] = True
    expected[4, 4] = True

    np.testing.assert_array_equal(visible, expected)


def test_obstacle_is_visible_and_blocks_cells_behind_on_same_ray(grid: GridSpec) -> None:
    occupancy = np.zeros((5, 5), dtype=np.float64)
    occupancy[2, 3] = 2.0

    visible = raycast_visibility(occupancy, grid)

    assert visible[2, 3]
    assert not visible[2, 4]
    assert visible[2, 2]
    assert visible[0, 0]


def test_noncentral_sensor_uses_actual_xy_for_same_row_blocker(grid: GridSpec) -> None:
    occupancy = np.zeros((5, 5), dtype=bool)
    occupancy[2, 3] = True

    visible = raycast_visibility(
        occupancy, grid, sensor_pose=(0.2, 0.0, 0.0)
    )

    assert visible[2, 3]
    assert not visible[2, 4]


def test_vertical_wall_has_hand_computed_supercover_visibility(grid: GridSpec) -> None:
    occupancy = np.zeros((5, 5), dtype=bool)
    occupancy[:, 3] = True
    expected = np.zeros((5, 5), dtype=bool)
    expected[:, :3] = True
    expected[2, 3] = True

    visible = raycast_visibility(occupancy, grid)

    np.testing.assert_array_equal(visible, expected)


def test_first_of_two_collinear_obstacles_is_visible_and_second_is_hidden(
    grid: GridSpec,
) -> None:
    occupancy = np.zeros((5, 5), dtype=bool)
    occupancy[2, 3:5] = True

    visible = raycast_visibility(occupancy, grid)

    assert visible[2, 3]
    assert not visible[2, 4]


@pytest.mark.parametrize("blocking_cell", [(2, 3), (1, 2)])
def test_supercover_corner_cells_prevent_diagonal_light_leaks(
    grid: GridSpec, blocking_cell: tuple[int, int]
) -> None:
    occupancy = np.zeros((5, 5), dtype=bool)
    occupancy[blocking_cell] = True

    visible = raycast_visibility(occupancy, grid)

    assert visible[blocking_cell]
    assert not visible[0, 4]


@pytest.mark.parametrize("blocking_cell", [(0, 1), (1, 0)])
def test_minimum_boundary_sensor_corner_side_blocks_diagonal_target(
    grid: GridSpec, blocking_cell: tuple[int, int]
) -> None:
    occupancy = np.zeros((5, 5), dtype=bool)
    occupancy[blocking_cell] = True
    x_min = -0.5 * grid.width * grid.resolution_m
    y_min = -0.5 * grid.height * grid.resolution_m

    visible = raycast_visibility(
        occupancy, grid, sensor_pose=(x_min, y_min, 0.0)
    )

    assert visible[blocking_cell]
    assert not visible[2, 2]


def test_sensor_cell_never_blocks_and_all_occupied_has_near_obstacles_visible(
    grid: GridSpec,
) -> None:
    sensor_only = np.zeros((5, 5), dtype=bool)
    sensor_only[2, 2] = True
    sensor_visible = raycast_visibility(sensor_only, grid)
    all_occupied = raycast_visibility(np.ones((5, 5), dtype=np.int8), grid)

    np.testing.assert_array_equal(sensor_visible, np.ones((5, 5), dtype=bool))
    assert all_occupied[2, 2]
    assert all_occupied[2, 3]
    assert not all_occupied[2, 4]


def test_plus_and_minus_pi_yaw_are_equivalent(grid: GridSpec) -> None:
    occupancy = np.zeros((5, 5), dtype=bool)

    positive = raycast_visibility(
        occupancy, grid, sensor_pose=(0.0, 0.0, np.pi), fov_rad=np.pi / 3.0
    )
    negative = raycast_visibility(
        occupancy, grid, sensor_pose=(0.0, 0.0, -np.pi), fov_rad=np.pi / 3.0
    )

    np.testing.assert_array_equal(positive, negative)


@pytest.mark.parametrize(
    "occupancy",
    [
        np.zeros((4, 5)),
        np.zeros((5, 5), dtype=np.complex128),
        np.full((5, 5), "free", dtype=object),
        np.full((5, 5), np.nan),
        np.full((5, 5), np.inf),
    ],
)
def test_rejects_invalid_occupancy(occupancy, grid: GridSpec) -> None:
    with pytest.raises((TypeError, ValueError)):
        raycast_visibility(occupancy, grid)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sensor_pose": (0.0, 0.0)}, "shape"),
        ({"sensor_pose": (0.0, 0.0, np.nan)}, "finite"),
        ({"sensor_pose": (3.0, 0.0, 0.0)}, "grid"),
        ({"fov_rad": True}, "fov_rad"),
        ({"fov_rad": -0.1}, "fov_rad"),
        ({"fov_rad": 2.0 * np.pi + 1e-6}, "fov_rad"),
        ({"max_range_m": True}, "max_range_m"),
        ({"max_range_m": -0.1}, "max_range_m"),
        ({"max_range_m": np.inf}, "max_range_m"),
    ],
)
def test_rejects_invalid_sensor_and_visibility_parameters(
    kwargs: dict, message: str, grid: GridSpec
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        raycast_visibility(np.zeros((5, 5)), grid, **kwargs)


def test_visibility_is_deterministic_for_noncentral_sensor_pose(grid: GridSpec) -> None:
    occupancy = np.zeros((5, 5), dtype=np.int64)
    occupancy[1, 3] = -2
    kwargs = {
        "sensor_pose": (0.2, -0.3, 0.4),
        "fov_rad": 1.7,
        "max_range_m": 2.3,
    }

    first = raycast_visibility(occupancy, grid, **kwargs)
    second = raycast_visibility(occupancy, grid, **kwargs)

    assert np.array_equal(first, second)


def test_candidate_visibility_exactly_matches_full_raycast_on_selected_cells() -> None:
    grid = GridSpec(
        height=17,
        width=17,
        history_steps=8,
        future_steps=15,
        resolution_m=0.25,
    )
    rng = np.random.default_rng(20260723)
    for _ in range(20):
        occupancy = rng.random((17, 17)) < 0.12
        candidates = rng.random((17, 17)) < 0.08
        sensor_pose = np.asarray(
            [
                rng.uniform(-1.8, 1.8),
                rng.uniform(-1.8, 1.8),
                rng.uniform(-np.pi, np.pi),
            ],
            dtype=np.float64,
        )
        fov_rad = float(rng.uniform(0.3, 2.0 * np.pi))
        max_range_m = float(rng.uniform(0.5, 3.0))

        full = raycast_visibility(
            occupancy,
            grid,
            sensor_pose=sensor_pose,
            fov_rad=fov_rad,
            max_range_m=max_range_m,
        )
        targeted = raycast_candidate_visibility(
            occupancy,
            candidates,
            grid,
            sensor_pose=sensor_pose,
            fov_rad=fov_rad,
            max_range_m=max_range_m,
        )

        np.testing.assert_array_equal(targeted, full & candidates)


def test_candidate_visibility_rejects_nonboolean_or_wrong_shape_mask(
    grid: GridSpec,
) -> None:
    occupancy = np.zeros((5, 5), dtype=bool)

    with pytest.raises(TypeError, match="candidate_mask"):
        raycast_candidate_visibility(
            occupancy, np.zeros((5, 5), dtype=np.float32), grid
        )
    with pytest.raises(ValueError, match="candidate_mask"):
        raycast_candidate_visibility(
            occupancy, np.zeros((4, 5), dtype=bool), grid
        )


def test_geometry_package_has_stable_central_public_api() -> None:
    import src.geometry as geometry

    expected = {
        "GridSpec",
        "wrap_angle",
        "global_to_local",
        "local_to_global",
        "transform_poses_global_to_local",
        "transform_poses_local_to_global",
        "unwrap_yaws",
        "interpolate_poses",
        "CircleFootprint",
        "RectangleFootprint",
        "Footprint",
        "inflate_footprint",
        "footprint_vertices",
        "footprint_aabb",
        "grid_bounds",
        "points_in_grid",
        "world_to_grid",
        "grid_to_world",
        "grid_cell_centers",
        "rasterize_footprint",
        "rasterize_footprint_sweep",
        "raycast_visibility",
        "raycast_candidate_visibility",
        "signed_clearance",
        "intersects",
        "segments_intersect",
        "trajectory_signed_clearances",
        "first_collision_index",
    }

    assert set(geometry.__all__) == expected
    assert all(getattr(geometry, name) is not None for name in expected)
