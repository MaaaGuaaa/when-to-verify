"""Trajectory-conditioned BEV query maps built on the canonical geometry API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.contracts import (
    ARRAY_DTYPE,
    POSE_TIME_LAYOUT_VERSION,
    GridSpec,
    LocalTrajectory,
    build_grid_spec,
)
from src.geometry import (
    CircleFootprint,
    Footprint,
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint,
    world_to_grid,
)

from .trajectory_sampler import CandidateRollout


@dataclass(frozen=True)
class TrajectoryQueryMaps:
    """Four float32 ``[H, W]`` channels in the frozen contract order.

    ``tta_map`` is the first discrete footprint-arrival time and is ``-1``
    outside the swept volume. ``braking_map`` is zero outside the sweep; on
    traversed cells it stores distance travelled before first arrival minus
    the instantaneous stopping distance ``|v|**2 / (2 * deceleration)``.
    """

    swept_mask: np.ndarray
    tta_map: np.ndarray
    braking_map: np.ndarray
    centerline_map: np.ndarray


def _footprint_sweep_radius(footprint: Footprint) -> float:
    """Return a conservative radius for rotational sweep sampling."""
    if isinstance(footprint, RectangleFootprint):
        return 0.5 * float(
            np.hypot(footprint.length_m, footprint.width_m)
        )
    if isinstance(footprint, CircleFootprint):
        return float(footprint.radius_m)
    raise TypeError("footprint must be a CircleFootprint or RectangleFootprint")


def _densify_trajectory(
    poses: np.ndarray,
    controls: np.ndarray,
    *,
    footprint: Footprint,
    resolution_m: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Include the current origin and sample every future control interval."""
    anchors = np.concatenate(
        (np.zeros((1, 3), dtype=np.float64), poses.astype(np.float64)),
        axis=0,
    )
    sweep_radius = _footprint_sweep_radius(footprint)
    maximum_step = 0.5 * resolution_m
    dense_poses = [anchors[0]]
    arrival_times = [0.0]
    path_distances = [0.0]
    speeds = [abs(float(controls[0, 0])) if controls.shape[0] else 0.0]
    cumulative_distance = 0.0
    for interval_index, (start, end) in enumerate(
        zip(anchors[:-1], anchors[1:])
    ):
        signed_speed = float(controls[interval_index, 0])
        yaw_rate = float(controls[interval_index, 1])
        speed = abs(signed_speed)
        corner_motion_bound = (
            speed * dt_s + abs(yaw_rate) * dt_s * sweep_radius
        )
        subdivisions = max(1, int(np.ceil(corner_motion_bound / maximum_step)))
        interval_path_distance = speed * dt_s
        for subdivision in range(1, subdivisions + 1):
            fraction = subdivision / subdivisions
            elapsed = fraction * dt_s
            pose = np.empty(3, dtype=np.float64)
            pose[2] = start[2] + yaw_rate * elapsed
            if yaw_rate == 0.0:
                distance = signed_speed * elapsed
                pose[0] = start[0] + distance * np.cos(start[2])
                pose[1] = start[1] + distance * np.sin(start[2])
            else:
                radius = signed_speed / yaw_rate
                pose[0] = start[0] + radius * (
                    np.sin(pose[2]) - np.sin(start[2])
                )
                pose[1] = start[1] - radius * (
                    np.cos(pose[2]) - np.cos(start[2])
                )
            dense_poses.append(pose)
            arrival_times.append((interval_index + fraction) * dt_s)
            path_distances.append(
                cumulative_distance + fraction * interval_path_distance
            )
            speeds.append(speed)
        cumulative_distance += interval_path_distance
        if not np.allclose(
            dense_poses[-1], end, rtol=0.0, atol=1e-5
        ):
            raise ValueError(
                "poses and controls violate differential-drive interval dynamics"
            )
    return (
        np.asarray(dense_poses, dtype=np.float64),
        np.asarray(arrival_times, dtype=np.float64),
        np.asarray(path_distances, dtype=np.float64),
        np.asarray(speeds, dtype=np.float64),
    )


def build_trajectory_query_maps(
    poses: np.ndarray,
    controls: np.ndarray,
    *,
    grid: GridSpec,
    footprint: Footprint,
    dt_s: float,
    braking_deceleration_mps2: float,
) -> TrajectoryQueryMaps:
    """Build query maps using SOP-02 footprint rasterization and grid indexing."""
    dt_s = float(dt_s)
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    braking_deceleration_mps2 = float(braking_deceleration_mps2)
    if (
        not np.isfinite(braking_deceleration_mps2)
        or braking_deceleration_mps2 <= 0.0
    ):
        raise ValueError("braking deceleration must be finite and positive")
    poses_array = np.asarray(poses)
    controls_array = np.asarray(controls)
    if poses_array.ndim != 2 or poses_array.shape[1:] != (3,):
        raise ValueError("poses must have shape (T, 3)")
    if controls_array.ndim != 2 or controls_array.shape[1:] != (2,):
        raise ValueError("controls must have shape (T, 2)")
    if poses_array.shape[0] != controls_array.shape[0]:
        raise ValueError("poses and controls must have the same length")
    if poses_array.dtype.kind not in "iuf" or controls_array.dtype.kind not in "iuf":
        raise TypeError("poses and controls must contain real numbers")
    if not np.isfinite(poses_array).all() or not np.isfinite(controls_array).all():
        raise ValueError("poses and controls must contain only finite values")
    resolution_m = float(grid.resolution_m)
    if not np.isfinite(resolution_m) or resolution_m <= 0.0:
        raise ValueError("grid resolution_m must be finite and positive")
    (
        sampled_poses,
        arrival_times,
        path_distances,
        sampled_speeds,
    ) = _densify_trajectory(
        poses_array,
        controls_array,
        footprint=footprint,
        resolution_m=resolution_m,
        dt_s=dt_s,
    )
    footprint_masks = tuple(
        rasterize_footprint(footprint, pose, grid) for pose in sampled_poses
    )
    if footprint_masks:
        swept_mask = np.logical_or.reduce(footprint_masks)
    else:
        swept_mask = np.zeros((grid.height, grid.width), dtype=bool)
    tta_map = np.full((grid.height, grid.width), -1.0, dtype=ARRAY_DTYPE)
    braking_map = np.zeros((grid.height, grid.width), dtype=ARRAY_DTYPE)
    for index, footprint_mask in enumerate(footprint_masks):
        first_arrival = footprint_mask & (tta_map < 0.0)
        tta_map[first_arrival] = arrival_times[index]
        speed = sampled_speeds[index]
        stopping_distance = speed * speed / (2.0 * braking_deceleration_mps2)
        braking_map[first_arrival] = (
            path_distances[index] - stopping_distance
        )
    centerline_map = np.zeros((grid.height, grid.width), dtype=ARRAY_DTYPE)
    centerline_points = sampled_poses[:, :2]
    centerline_indices = world_to_grid(centerline_points, grid)
    centerline_map[centerline_indices[:, 0], centerline_indices[:, 1]] = 1.0
    return TrajectoryQueryMaps(
        swept_mask=swept_mask.astype(ARRAY_DTYPE),
        tta_map=tta_map,
        braking_map=braking_map,
        centerline_map=centerline_map,
    )


def build_local_trajectory(
    candidate: CandidateRollout,
    config: dict,
    *,
    braking_deceleration_mps2: float,
    task_cost: float = 0.0,
) -> LocalTrajectory:
    """Attach query maps and materialize the frozen trajectory contract."""
    task_cost = float(task_cost)
    if not np.isfinite(task_cost):
        raise ValueError("task_cost must be finite")
    grid = build_grid_spec(config)
    robot = config["robot"]
    footprint = inflate_footprint(
        RectangleFootprint(robot["length_m"], robot["width_m"]),
        robot["inflation_m"],
    )
    dt_s = float(config["trajectories"]["dt_s"])
    maps = build_trajectory_query_maps(
        candidate.poses,
        candidate.controls,
        grid=grid,
        footprint=footprint,
        dt_s=dt_s,
        braking_deceleration_mps2=braking_deceleration_mps2,
    )
    return LocalTrajectory(
        trajectory_id=candidate.trajectory_id,
        poses=candidate.poses,
        controls=candidate.controls,
        swept_mask=maps.swept_mask,
        tta_map=maps.tta_map,
        braking_map=maps.braking_map,
        centerline_map=maps.centerline_map,
        task_cost=task_cost,
        metadata={
            "is_stop": candidate.is_stop,
            "is_reverse": candidate.is_reverse,
            "v": float(candidate.controls[0, 0]),
            "omega": float(candidate.controls[0, 1]),
            "pose_time_layout_version": POSE_TIME_LAYOUT_VERSION,
            "first_pose_time_s": dt_s,
            "last_pose_time_s": dt_s * int(candidate.poses.shape[0]),
            "dt_s": dt_s,
            "trajectory_steps": int(candidate.poses.shape[0]),
            "braking_deceleration_mps2": float(braking_deceleration_mps2),
        },
    )
