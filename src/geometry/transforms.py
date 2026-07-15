"""Deterministic SE(2) transforms and pose interpolation utilities.

Local coordinates are robot-centric: the reference pose is the origin, ``+x``
points forward, and ``+y`` points left. Positions are measured in metres and
yaw angles in radians.
"""

from __future__ import annotations

import numpy as np


def _finite_array(values, *, name: str, final_dimension: int | None = None) -> np.ndarray:
    """Convert numeric input to float64 and enforce shape/finite-value rules."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if final_dimension is not None and (array.ndim == 0 or array.shape[-1] != final_dimension):
        raise ValueError(f"{name} must have shape (..., {final_dimension})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _reference_pose(reference_pose) -> np.ndarray:
    reference = _finite_array(reference_pose, name="reference_pose")
    if reference.shape != (3,):
        raise ValueError("reference_pose must have shape (3,)")
    return reference


def wrap_angle(angles):
    """Wrap finite angle(s) to the half-open interval ``[-pi, pi)``."""
    angle_array = _finite_array(angles, name="angles")
    wrapped = np.remainder(angle_array + np.pi, 2.0 * np.pi) - np.pi
    wrapped = np.where(wrapped >= np.pi, wrapped - 2.0 * np.pi, wrapped)
    if wrapped.ndim == 0:
        return float(wrapped)
    return wrapped


def global_to_local(points_global, reference_pose) -> np.ndarray:
    """Transform global ``[..., x, y]`` points into one reference pose's frame."""
    points = _finite_array(points_global, name="points_global", final_dimension=2)
    reference = _reference_pose(reference_pose)
    delta = points - reference[:2]
    cosine = np.cos(reference[2])
    sine = np.sin(reference[2])
    x_local = cosine * delta[..., 0] + sine * delta[..., 1]
    y_local = -sine * delta[..., 0] + cosine * delta[..., 1]
    return np.stack((x_local, y_local), axis=-1).astype(np.float64, copy=False)


def local_to_global(points_local, reference_pose) -> np.ndarray:
    """Transform local ``[..., x, y]`` points into the global frame."""
    points = _finite_array(points_local, name="points_local", final_dimension=2)
    reference = _reference_pose(reference_pose)
    cosine = np.cos(reference[2])
    sine = np.sin(reference[2])
    x_global = cosine * points[..., 0] - sine * points[..., 1] + reference[0]
    y_global = sine * points[..., 0] + cosine * points[..., 1] + reference[1]
    return np.stack((x_global, y_global), axis=-1).astype(np.float64, copy=False)


def transform_poses_global_to_local(poses_global, reference_pose) -> np.ndarray:
    """Transform global ``[..., x, y, yaw]`` poses into a local frame."""
    poses = _finite_array(poses_global, name="poses_global", final_dimension=3)
    reference = _reference_pose(reference_pose)
    positions = global_to_local(poses[..., :2], reference)
    yaws = wrap_angle(poses[..., 2] - reference[2])
    return np.concatenate((positions, np.asarray(yaws)[..., None]), axis=-1).astype(
        np.float64, copy=False
    )


def transform_poses_local_to_global(poses_local, reference_pose) -> np.ndarray:
    """Transform local ``[..., x, y, yaw]`` poses into the global frame."""
    poses = _finite_array(poses_local, name="poses_local", final_dimension=3)
    reference = _reference_pose(reference_pose)
    positions = local_to_global(poses[..., :2], reference)
    yaws = wrap_angle(poses[..., 2] + reference[2])
    return np.concatenate((positions, np.asarray(yaws)[..., None]), axis=-1).astype(
        np.float64, copy=False
    )


def unwrap_yaws(yaws) -> np.ndarray:
    """Unwrap a one-dimensional finite yaw sequence in temporal order."""
    yaw_array = _finite_array(yaws, name="yaws")
    if yaw_array.ndim != 1:
        raise ValueError("yaws must have shape (N,)")
    return np.unwrap(yaw_array).astype(np.float64, copy=False)


def _timestamps(values, *, name: str, minimum_size: int) -> np.ndarray:
    timestamps = _finite_array(values, name=name)
    if timestamps.ndim != 1:
        raise ValueError(f"{name} must have shape (N,)")
    if timestamps.size < minimum_size:
        raise ValueError(f"{name} must contain at least {minimum_size} timestamp(s)")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing with no duplicates")
    return timestamps


def interpolate_poses(source_timestamps, source_poses, query_timestamps) -> np.ndarray:
    """Linearly interpolate positions and unwrapped yaw at ordered query times.

    The source and query timestamps are never sorted implicitly, and queries
    outside the source range are rejected rather than extrapolated. The returned
    yaw remains unwrapped so a crossing of ``+/-pi`` is continuous.
    """
    source_times = _timestamps(source_timestamps, name="source_timestamps", minimum_size=2)
    query_times = _timestamps(query_timestamps, name="query_timestamps", minimum_size=1)
    poses = _finite_array(source_poses, name="source_poses")
    if poses.ndim != 2 or poses.shape != (source_times.size, 3):
        raise ValueError("source_poses must have shape (len(source_timestamps), 3)")
    if query_times[0] < source_times[0] or query_times[-1] > source_times[-1]:
        raise ValueError("query_timestamps must lie within the source range")

    unwrapped_yaws = unwrap_yaws(poses[:, 2])
    result = np.column_stack(
        tuple(np.interp(query_times, source_times, poses[:, axis]) for axis in range(2))
        + (np.interp(query_times, source_times, unwrapped_yaws),)
    )
    return result.astype(np.float64, copy=False)
