import warnings

import numpy as np
import pytest

from src.contracts import GridSpec
import src.geometry.rasterization as rasterization_module
from src.geometry.collision import intersects
from src.geometry.rasterization import (
    grid_bounds,
    grid_cell_centers,
    grid_to_world,
    points_in_grid,
    rasterize_footprint,
    rasterize_footprint_sweep,
    world_to_grid,
)
from src.geometry.footprints import (
    CircleFootprint,
    RectangleFootprint,
    footprint_aabb,
    inflate_footprint,
)


@pytest.fixture
def grid() -> GridSpec:
    return GridSpec(height=2, width=4, history_steps=3, future_steps=5, resolution_m=0.5)


def test_grid_bounds_are_centered_and_half_open(grid: GridSpec) -> None:
    assert grid_bounds(grid) == pytest.approx((-1.0, 1.0, -0.5, 0.5))


def test_world_to_grid_uses_row_y_and_column_x(grid: GridSpec) -> None:
    points = np.array(
        [
            [-1.0, -0.5],
            [-0.75, -0.25],
            [0.25, 0.25],
            [1.0 - 1e-12, 0.5 - 1e-12],
        ]
    )

    indices = world_to_grid(points, grid)

    np.testing.assert_array_equal(indices, [[0, 0], [0, 0], [1, 2], [1, 3]])
    assert indices.dtype == np.int64
    assert indices.shape == points.shape


def test_world_to_grid_maps_representable_values_below_maximum_to_last_cells(
    grid: GridSpec,
) -> None:
    x_min, x_max, y_min, y_max = grid_bounds(grid)
    points = np.array(
        [
            [np.nextafter(x_max, -np.inf), y_min],
            [x_min, np.nextafter(y_max, -np.inf)],
        ]
    )

    indices = world_to_grid(points, grid)

    np.testing.assert_array_equal(indices, [[0, grid.width - 1], [grid.height - 1, 0]])


def test_grid_cell_centers_round_trip_exactly(grid: GridSpec) -> None:
    centers = grid_cell_centers(grid)
    indices = np.indices((grid.height, grid.width), dtype=np.int64).transpose(1, 2, 0)

    assert centers.shape == (grid.height, grid.width, 2)
    assert centers.dtype == np.float64
    np.testing.assert_array_equal(world_to_grid(centers, grid), indices)
    np.testing.assert_allclose(grid_to_world(indices, grid), centers, atol=0.0)
    np.testing.assert_allclose(centers[0, 0], [-0.75, -0.25])
    np.testing.assert_allclose(centers[-1, -1], [0.75, 0.25])


def test_points_in_grid_includes_minimum_and_excludes_maximum(grid: GridSpec) -> None:
    points = np.array(
        [
            [-1.0, -0.5],
            [1.0, 0.0],
            [0.0, 0.5],
            [-1.0 - 1e-12, 0.0],
            [0.0, -0.5 - 1e-12],
            [0.999999, 0.499999],
        ]
    )

    mask = points_in_grid(points, grid)

    np.testing.assert_array_equal(mask, [True, False, False, False, False, True])
    assert mask.dtype == np.bool_


def test_out_of_bounds_raises_by_default(grid: GridSpec) -> None:
    with pytest.raises(ValueError, match="outside"):
        world_to_grid([[1.0, 0.0]], grid)
    with pytest.raises(ValueError, match="outside"):
        grid_to_world([[2, 0]], grid)


def test_explicit_clip_maps_to_nearest_edge_cell(grid: GridSpec) -> None:
    clipped_indices = world_to_grid([[-2.0, -2.0], [2.0, 2.0]], grid, clip=True)
    clipped_centers = grid_to_world([[-2, -3], [4, 8]], grid, clip=True)

    np.testing.assert_array_equal(clipped_indices, [[0, 0], [1, 3]])
    np.testing.assert_allclose(clipped_centers, [[-0.75, -0.25], [0.75, 0.25]])


@pytest.mark.parametrize("function", [world_to_grid, points_in_grid])
def test_point_grid_functions_reject_invalid_shape_and_non_finite_points(function, grid: GridSpec) -> None:
    with pytest.raises(ValueError, match="shape"):
        function(np.ones((2, 3)), grid)
    with pytest.raises(ValueError, match="finite"):
        function([[np.nan, 0.0]], grid)
    with pytest.raises(ValueError, match="finite"):
        function([[0.0, np.inf]], grid)


def test_grid_to_world_rejects_invalid_indices(grid: GridSpec) -> None:
    with pytest.raises(ValueError, match="shape"):
        grid_to_world(np.ones((2, 3), dtype=np.int64), grid)
    with pytest.raises(ValueError, match="integer"):
        grid_to_world([[0.0, 1.0]], grid)
    with pytest.raises(ValueError, match="finite"):
        grid_to_world([[np.nan, 1.0]], grid)


@pytest.mark.parametrize(
    "invalid_grid",
    [
        object(),
        GridSpec(height=0, width=4, history_steps=3, future_steps=5, resolution_m=0.5),
        GridSpec(height=2, width=4, history_steps=3, future_steps=5, resolution_m=np.nan),
    ],
)
def test_grid_apis_require_a_valid_grid_spec(invalid_grid) -> None:
    error = TypeError if not isinstance(invalid_grid, GridSpec) else ValueError
    with pytest.raises(error):
        grid_bounds(invalid_grid)


def test_grid_mapping_is_deterministic(grid: GridSpec) -> None:
    points = np.array([[-0.75, -0.25], [0.75, 0.25]])

    first_indices = world_to_grid(points, grid)
    second_indices = world_to_grid(points, grid)
    first_centers = grid_cell_centers(grid)
    second_centers = grid_cell_centers(grid)

    assert np.array_equal(first_indices, second_indices)
    assert np.array_equal(first_centers, second_centers)


@pytest.fixture
def square_grid() -> GridSpec:
    return GridSpec(height=5, width=5, history_steps=1, future_steps=1, resolution_m=1.0)


def test_circle_rasterization_conservatively_includes_cells_touched_away_from_centres(
    square_grid: GridSpec,
) -> None:
    mask = rasterize_footprint(CircleFootprint(0.5), [0.0, 0.0, 0.0], square_grid)
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = True
    expected[1, 2] = expected[3, 2] = True
    expected[2, 1] = expected[2, 3] = True

    np.testing.assert_array_equal(mask, expected)
    assert mask.shape == (square_grid.height, square_grid.width)
    assert mask.dtype == np.bool_


def test_rotated_rectangle_rasterization_uses_closed_cell_square_intersection(
    square_grid: GridSpec,
) -> None:
    mask = rasterize_footprint(
        RectangleFootprint(1.0, 1.0), [0.0, 0.0, np.pi / 4.0], square_grid
    )
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = True
    expected[1, 2] = expected[3, 2] = True
    expected[2, 1] = expected[2, 3] = True

    np.testing.assert_array_equal(mask, expected)


def test_rectangle_rasterization_avoids_scalar_per_cell_collision_queries(
    square_grid: GridSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    footprint = RectangleFootprint(1.37, 0.63)
    pose = np.asarray([0.17, -0.23, 0.41], dtype=np.float64)
    expected = rasterize_footprint(footprint, pose, square_grid)

    def unexpected_scalar_query(*args: object, **kwargs: object) -> bool:
        del args, kwargs
        raise AssertionError("rasterization must batch candidate cell geometry")

    monkeypatch.setattr(
        rasterization_module,
        "intersects",
        unexpected_scalar_query,
    )

    np.testing.assert_array_equal(
        rasterize_footprint(footprint, pose, square_grid),
        expected,
    )


@pytest.mark.parametrize(
    ("footprint", "pose"),
    [
        (CircleFootprint(0.37), np.asarray([0.13, -0.21, 0.7])),
        (CircleFootprint(0.5), np.asarray([-3.0, 0.0, 0.0])),
        (RectangleFootprint(1.37, 0.63), np.asarray([0.17, -0.23, 0.41])),
        (RectangleFootprint(1.0, 1.0), np.asarray([0.0, 0.0, np.pi / 4.0])),
        (RectangleFootprint(0.7, 1.3), np.asarray([1.11, -1.27, -0.83])),
    ],
)
def test_batched_rasterization_matches_scalar_closed_cell_authority(
    square_grid: GridSpec,
    footprint: CircleFootprint | RectangleFootprint,
    pose: np.ndarray,
) -> None:
    centers = grid_cell_centers(square_grid)
    cell = RectangleFootprint(
        square_grid.resolution_m,
        square_grid.resolution_m,
    )
    expected = np.zeros((square_grid.height, square_grid.width), dtype=bool)
    for row in range(square_grid.height):
        for column in range(square_grid.width):
            center = centers[row, column]
            expected[row, column] = intersects(
                footprint,
                pose,
                cell,
                np.asarray([center[0], center[1], 0.0], dtype=np.float64),
            )

    np.testing.assert_array_equal(
        rasterize_footprint(footprint, pose, square_grid),
        expected,
    )


def test_batched_rectangle_rasterization_defers_numeric_boundary_to_authority() -> None:
    boundary_grid = GridSpec(
        height=160,
        width=160,
        history_steps=1,
        future_steps=1,
        resolution_m=0.125,
    )
    footprint = RectangleFootprint(0.5, 0.25)
    pose = np.asarray([-10.0, 0.0, np.pi / 2.0], dtype=np.float64)
    centers = grid_cell_centers(boundary_grid)
    bounds = footprint_aabb(footprint, pose)
    half_cell = 0.5 * boundary_grid.resolution_m
    candidates = (
        (centers[..., 0] >= bounds[0] - half_cell)
        & (centers[..., 0] <= bounds[1] + half_cell)
        & (centers[..., 1] >= bounds[2] - half_cell)
        & (centers[..., 1] <= bounds[3] + half_cell)
    )
    cell = RectangleFootprint(
        boundary_grid.resolution_m,
        boundary_grid.resolution_m,
    )
    expected = np.zeros((boundary_grid.height, boundary_grid.width), dtype=bool)
    for row, column in np.argwhere(candidates):
        center = centers[row, column]
        expected[row, column] = intersects(
            footprint,
            pose,
            cell,
            np.asarray([center[0], center[1], 0.0], dtype=np.float64),
        )

    np.testing.assert_array_equal(
        rasterize_footprint(footprint, pose, boundary_grid),
        expected,
    )


def test_rasterization_supports_inflation_and_clips_at_grid_bounds(
    square_grid: GridSpec,
) -> None:
    base = rasterize_footprint(CircleFootprint(0.25), [0.0, 0.0, 0.0], square_grid)
    inflated = rasterize_footprint(
        inflate_footprint(CircleFootprint(0.25), 0.25), [0.0, 0.0, 1.7], square_grid
    )
    clipped = rasterize_footprint(CircleFootprint(0.5), [-2.5, 0.0, 0.0], square_grid)
    outside = rasterize_footprint(CircleFootprint(0.5), [10.0, 0.0, 0.0], square_grid)

    assert base.sum() == 1
    assert inflated.sum() == 5
    np.testing.assert_array_equal(np.flatnonzero(clipped), [5, 10, 15])
    assert not outside.any()


def test_circle_touching_closed_outer_grid_boundary_is_rasterized(
    square_grid: GridSpec,
) -> None:
    touching = rasterize_footprint(
        CircleFootprint(0.5), [-3.0, 0.0, 0.0], square_grid
    )
    separated = rasterize_footprint(
        CircleFootprint(0.5), [-3.0 - 1e-6, 0.0, 0.0], square_grid
    )
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 0] = True

    np.testing.assert_array_equal(touching, expected)
    assert not separated.any()


def test_sweep_is_exact_discrete_union_and_empty_sweep_is_false(square_grid: GridSpec) -> None:
    footprint = RectangleFootprint(0.6, 0.4)
    poses = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, np.pi / 2.0], [1.0, 0.0, np.pi]])
    individual = [rasterize_footprint(footprint, pose, square_grid) for pose in poses]

    sweep = rasterize_footprint_sweep(footprint, poses, square_grid)

    np.testing.assert_array_equal(sweep, np.logical_or.reduce(individual))
    assert all(np.all(mask <= sweep) for mask in individual)
    empty = rasterize_footprint_sweep(footprint, np.empty((0, 3)), square_grid)
    np.testing.assert_array_equal(empty, np.zeros((5, 5), dtype=bool))


@pytest.mark.parametrize(
    "poses",
    [
        [0.0, 0.0, 0.0],
        np.zeros((2, 2)),
        np.array([[0.0, np.nan, 0.0]]),
        np.array([[0.0, 0.0, np.inf]]),
    ],
)
def test_sweep_rejects_invalid_pose_arrays(poses, square_grid: GridSpec) -> None:
    with pytest.raises((TypeError, ValueError)):
        rasterize_footprint_sweep(CircleFootprint(0.2), poses, square_grid)


def test_footprint_rasterization_rejects_invalid_inputs(square_grid: GridSpec) -> None:
    with pytest.raises(TypeError):
        rasterize_footprint(object(), [0.0, 0.0, 0.0], square_grid)
    with pytest.raises(ValueError, match="shape"):
        rasterize_footprint(CircleFootprint(0.2), [0.0, 0.0], square_grid)
    with pytest.raises(ValueError, match="finite"):
        rasterize_footprint(CircleFootprint(0.2), [np.nan, 0.0, 0.0], square_grid)
    with pytest.raises(TypeError):
        rasterize_footprint(CircleFootprint(0.2), [0.0, 0.0, 0.0], object())


def test_footprint_rasterization_is_deterministic(square_grid: GridSpec) -> None:
    footprint = RectangleFootprint(1.3, 0.7)
    pose = [0.2, -0.3, 0.61]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        first = rasterize_footprint(footprint, pose, square_grid)
        second = rasterize_footprint(footprint, pose, square_grid)

    assert np.array_equal(first, second)


def test_footprint_rasterization_emits_no_runtime_warning(square_grid: GridSpec) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rasterize_footprint(
            RectangleFootprint(1.3, 0.7), [0.2, -0.3, 0.61], square_grid
        )
