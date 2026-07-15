"""Signed-clearance and collision queries for supported footprints."""

from __future__ import annotations

from typing import Any

import numpy as np

from .footprints import (
    CircleFootprint,
    Footprint,
    RectangleFootprint,
    _finite_real,
    _validate_footprint,
    _validate_pose,
    footprint_vertices,
)


def _nonnegative_tolerance(atol: Any) -> float:
    tolerance = _finite_real(atol, "atol")
    if tolerance < 0.0:
        raise ValueError("atol must be non-negative")
    return tolerance


def _circle_circle_clearance(
    circle_a: CircleFootprint,
    pose_a: np.ndarray,
    circle_b: CircleFootprint,
    pose_b: np.ndarray,
) -> float:
    delta = pose_b[:2] - pose_a[:2]
    center_distance = np.hypot(delta[0], delta[1])
    return float(center_distance - (circle_a.radius_m + circle_b.radius_m))


def _circle_rectangle_clearance(
    circle: CircleFootprint,
    circle_pose: np.ndarray,
    rectangle: RectangleFootprint,
    rectangle_pose: np.ndarray,
) -> float:
    delta = circle_pose[:2] - rectangle_pose[:2]
    cosine = np.cos(rectangle_pose[2])
    sine = np.sin(rectangle_pose[2])
    local_center = np.array(
        [cosine * delta[0] + sine * delta[1], -sine * delta[0] + cosine * delta[1]],
        dtype=np.float64,
    )
    half_extents = np.array(
        [0.5 * rectangle.length_m, 0.5 * rectangle.width_m], dtype=np.float64
    )
    offsets = np.abs(local_center) - half_extents
    outside = np.maximum(offsets, 0.0)
    outside_distance = np.hypot(outside[0], outside[1])
    inside_distance = min(float(np.max(offsets)), 0.0)
    return float(outside_distance + inside_distance - circle.radius_m)


def _rectangle_axes(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.cos(pose[2])
    sine = np.sin(pose[2])
    return (
        np.array([cosine, sine], dtype=np.float64),
        np.array([-sine, cosine], dtype=np.float64),
    )


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    scale = float(
        max(np.max(np.abs(point)), np.max(np.abs(start)), np.max(np.abs(end)))
    )
    if scale == 0.0:
        return 0.0
    scaled_point = point / scale
    scaled_start = start / scale
    scaled_end = end / scale
    segment = scaled_end - scaled_start
    squared_length = float(np.dot(segment, segment))
    if squared_length == 0.0:
        parameter = 0.0
    else:
        parameter = float(np.dot(scaled_point - scaled_start, segment) / squared_length)
        parameter = min(1.0, max(0.0, parameter))
    closest = (1.0 - parameter) * scaled_start + parameter * scaled_end
    delta = scaled_point - closest
    scaled_distance = float(np.hypot(delta[0], delta[1]))
    if scale > 1.0 and scaled_distance > np.finfo(np.float64).max / scale:
        return float("inf")
    return scaled_distance * scale


def _orientation_sign(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
    coordinate_scale = float(
        max(np.max(np.abs(a)), np.max(np.abs(b)), np.max(np.abs(c)))
    )
    if coordinate_scale == 0.0:
        return 0
    scaled_a = a / coordinate_scale
    ab = b / coordinate_scale - scaled_a
    ac = c / coordinate_scale - scaled_a
    scale_ab = float(np.max(np.abs(ab)))
    scale_ac = float(np.max(np.abs(ac)))
    if scale_ab == 0.0 or scale_ac == 0.0:
        return 0
    normalized_ab = ab / scale_ab
    normalized_ac = ac / scale_ac
    cross = normalized_ab[0] * normalized_ac[1] - normalized_ab[1] * normalized_ac[0]
    return int(cross > 0.0) - int(cross < 0.0)


def _point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    return bool(
        _orientation_sign(start, end, point) == 0
        and min(start[0], end[0]) <= point[0] <= max(start[0], end[0])
        and min(start[1], end[1]) <= point[1] <= max(start[1], end[1])
    )


def _segments_intersect_exact(
    start_a: np.ndarray,
    end_a: np.ndarray,
    start_b: np.ndarray,
    end_b: np.ndarray,
) -> bool:
    cross_a_start = _orientation_sign(start_a, end_a, start_b)
    cross_a_end = _orientation_sign(start_a, end_a, end_b)
    cross_b_start = _orientation_sign(start_b, end_b, start_a)
    cross_b_end = _orientation_sign(start_b, end_b, end_a)

    if (cross_a_start > 0.0 > cross_a_end or cross_a_start < 0.0 < cross_a_end) and (
        cross_b_start > 0.0 > cross_b_end or cross_b_start < 0.0 < cross_b_end
    ):
        return True
    return bool(
        (cross_a_start == 0.0 and _point_on_segment(start_b, start_a, end_a))
        or (cross_a_end == 0.0 and _point_on_segment(end_b, start_a, end_a))
        or (cross_b_start == 0.0 and _point_on_segment(start_a, start_b, end_b))
        or (cross_b_end == 0.0 and _point_on_segment(end_a, start_b, end_b))
    )


def _segment_distance(
    start_a: np.ndarray,
    end_a: np.ndarray,
    start_b: np.ndarray,
    end_b: np.ndarray,
) -> float:
    if _segments_intersect_exact(start_a, end_a, start_b, end_b):
        return 0.0
    return min(
        _point_segment_distance(start_a, start_b, end_b),
        _point_segment_distance(end_a, start_b, end_b),
        _point_segment_distance(start_b, start_a, end_a),
        _point_segment_distance(end_b, start_a, end_a),
    )


def _polygon_distance(vertices_a: np.ndarray, vertices_b: np.ndarray) -> float:
    minimum = np.inf
    for index_a in range(4):
        start_a = vertices_a[index_a]
        end_a = vertices_a[(index_a + 1) % 4]
        for index_b in range(4):
            start_b = vertices_b[index_b]
            end_b = vertices_b[(index_b + 1) % 4]
            minimum = min(
                minimum, _segment_distance(start_a, end_a, start_b, end_b)
            )
    return float(minimum)


def _absolute_center_projection(
    center_a: np.ndarray, center_b: np.ndarray, axis: np.ndarray
) -> float:
    coordinate_scale = float(
        max(np.max(np.abs(center_a)), np.max(np.abs(center_b)))
    )
    if coordinate_scale == 0.0:
        return 0.0
    scaled_delta = center_b / coordinate_scale - center_a / coordinate_scale
    scaled_projection = abs(float(np.dot(scaled_delta, axis)))
    if (
        coordinate_scale > 1.0
        and scaled_projection > np.finfo(np.float64).max / coordinate_scale
    ):
        return float("inf")
    return scaled_projection * coordinate_scale


def _rectangle_rectangle_clearance(
    rectangle_a: RectangleFootprint,
    pose_a: np.ndarray,
    rectangle_b: RectangleFootprint,
    pose_b: np.ndarray,
) -> float:
    axes_a = _rectangle_axes(pose_a)
    axes_b = _rectangle_axes(pose_b)
    half_extents_a = (0.5 * rectangle_a.length_m, 0.5 * rectangle_a.width_m)
    half_extents_b = (0.5 * rectangle_b.length_m, 0.5 * rectangle_b.width_m)
    maximum_gap = -np.inf

    for axis in (*axes_a, *axes_b):
        center_projection = _absolute_center_projection(pose_a[:2], pose_b[:2], axis)
        support_a = sum(
            extent * abs(float(np.dot(local_axis, axis)))
            for extent, local_axis in zip(half_extents_a, axes_a)
        )
        support_b = sum(
            extent * abs(float(np.dot(local_axis, axis)))
            for extent, local_axis in zip(half_extents_b, axes_b)
        )
        support_sum = support_a + support_b
        gap = center_projection - support_sum
        roundoff = (
            8.0
            * np.finfo(np.float64).eps
            * max(center_projection, support_sum)
        )
        if np.isfinite(gap) and np.isfinite(roundoff) and abs(gap) <= roundoff:
            gap = 0.0
        maximum_gap = max(maximum_gap, gap)

    if np.isposinf(maximum_gap):
        return float("inf")
    if maximum_gap > 0.0:
        rebased_pose_a = pose_a.copy()
        rebased_pose_b = pose_b.copy()
        rebased_pose_a[:2] = 0.5 * pose_a[:2] - 0.5 * pose_b[:2]
        rebased_pose_b[:2] = -rebased_pose_a[:2]
        vertices_a = footprint_vertices(rectangle_a, rebased_pose_a)
        vertices_b = footprint_vertices(rectangle_b, rebased_pose_b)
        return _polygon_distance(vertices_a, vertices_b)
    return float(maximum_gap)


def signed_clearance(
    footprint_a: Footprint,
    pose_a: Any,
    footprint_b: Footprint,
    pose_b: Any,
) -> float:
    """Return positive separation or negative penetration between two shapes."""

    footprint_a = _validate_footprint(footprint_a)
    footprint_b = _validate_footprint(footprint_b)
    validated_pose_a = _validate_pose(pose_a)
    validated_pose_b = _validate_pose(pose_b)

    if isinstance(footprint_a, CircleFootprint) and isinstance(
        footprint_b, CircleFootprint
    ):
        return _circle_circle_clearance(
            footprint_a, validated_pose_a, footprint_b, validated_pose_b
        )
    if isinstance(footprint_a, CircleFootprint) and isinstance(
        footprint_b, RectangleFootprint
    ):
        return _circle_rectangle_clearance(
            footprint_a, validated_pose_a, footprint_b, validated_pose_b
        )
    if isinstance(footprint_a, RectangleFootprint) and isinstance(
        footprint_b, CircleFootprint
    ):
        return _circle_rectangle_clearance(
            footprint_b, validated_pose_b, footprint_a, validated_pose_a
        )
    return _rectangle_rectangle_clearance(
        footprint_a, validated_pose_a, footprint_b, validated_pose_b
    )


def intersects(
    footprint_a: Footprint,
    pose_a: Any,
    footprint_b: Footprint,
    pose_b: Any,
    *,
    atol: float = 0.0,
) -> bool:
    """Return whether two footprints touch or overlap within ``atol``."""

    tolerance = _nonnegative_tolerance(atol)
    return bool(signed_clearance(footprint_a, pose_a, footprint_b, pose_b) <= tolerance)


def _validate_point(point: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(point)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array") from exc
    if array.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def segments_intersect(
    start_a: Any,
    end_a: Any,
    start_b: Any,
    end_b: Any,
    *,
    atol: float = 0.0,
) -> bool:
    """Return whether two finite 2-D line segments meet within ``atol``."""

    tolerance = _nonnegative_tolerance(atol)
    validated_start_a = _validate_point(start_a, "start_a")
    validated_end_a = _validate_point(end_a, "end_a")
    validated_start_b = _validate_point(start_b, "start_b")
    validated_end_b = _validate_point(end_b, "end_b")
    distance = _segment_distance(
        validated_start_a, validated_end_a, validated_start_b, validated_end_b
    )
    return bool(distance <= tolerance)


def _validate_trajectory(poses: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(poses)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array") from exc
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (T, 3)")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def trajectory_signed_clearances(
    footprint_a: Footprint,
    poses_a: Any,
    footprint_b: Footprint,
    poses_b: Any,
) -> np.ndarray:
    """Compute synchronous per-frame signed clearances for two trajectories."""

    footprint_a = _validate_footprint(footprint_a)
    footprint_b = _validate_footprint(footprint_b)
    validated_poses_a = _validate_trajectory(poses_a, "poses_a")
    validated_poses_b = _validate_trajectory(poses_b, "poses_b")
    if validated_poses_a.shape[0] != validated_poses_b.shape[0]:
        raise ValueError("poses_a and poses_b must have the same length")

    result = np.empty(validated_poses_a.shape[0], dtype=np.float64)
    for index in range(result.size):
        result[index] = signed_clearance(
            footprint_a,
            validated_poses_a[index],
            footprint_b,
            validated_poses_b[index],
        )
    return result


def first_collision_index(clearances: Any, *, atol: float = 0.0) -> int | None:
    """Return the first index whose signed clearance is at most ``atol``."""

    tolerance = _nonnegative_tolerance(atol)
    try:
        array = np.asarray(clearances)
    except (TypeError, ValueError) as exc:
        raise TypeError("clearances must be a numeric array") from exc
    if array.ndim != 1:
        raise ValueError("clearances must be one-dimensional")
    if array.dtype.kind not in "iuf":
        raise TypeError("clearances must contain real numbers")
    values = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("clearances must contain only finite values")
    matches = np.flatnonzero(values <= tolerance)
    return None if matches.size == 0 else int(matches[0])
