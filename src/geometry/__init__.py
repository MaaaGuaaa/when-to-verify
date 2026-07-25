"""Public, side-effect-free geometry API."""

from src.contracts import GridSpec

from .collision import (
    first_collision_index,
    intersects,
    segments_intersect,
    signed_clearance,
    trajectory_signed_clearances,
)
from .footprints import (
    CircleFootprint,
    Footprint,
    RectangleFootprint,
    footprint_aabb,
    footprint_vertices,
    inflate_footprint,
)
from .rasterization import (
    grid_bounds,
    grid_cell_centers,
    grid_to_world,
    points_in_grid,
    rasterize_footprint,
    rasterize_footprint_sweep,
    world_to_grid,
)
from .static_occluders import (
    CircleOccluder,
    RectangleOccluder,
    StaticOccluder,
    inflate_occluder,
    occluder_bounds,
    occluder_payload,
    point_signed_distance,
    rasterize_occluder,
    segment_intersects_occluder,
)
from .raycasting import raycast_candidate_visibility, raycast_visibility
from .transforms import (
    global_to_local,
    interpolate_poses,
    local_to_global,
    transform_poses_global_to_local,
    transform_poses_local_to_global,
    unwrap_yaws,
    wrap_angle,
)

__all__ = (
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
    "RectangleOccluder",
    "CircleOccluder",
    "StaticOccluder",
    "occluder_payload",
    "occluder_bounds",
    "inflate_occluder",
    "point_signed_distance",
    "segment_intersects_occluder",
    "rasterize_occluder",
    "raycast_visibility",
    "raycast_candidate_visibility",
    "signed_clearance",
    "intersects",
    "segments_intersect",
    "trajectory_signed_clearances",
    "first_collision_index",
)
