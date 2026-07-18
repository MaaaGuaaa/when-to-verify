"""Behavioral tests for schema-v3 hidden-risk ground truth."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from src.contracts import GridSpec, LocalTrajectory, OracleWorld
from src.geometry import CircleFootprint, RectangleFootprint
from src.generation.risk_gt import (
    RISK_GT_VERSION,
    compute_hidden_risk_gt,
    resolve_no_object_clearance_sentinel,
)


def _grid() -> GridSpec:
    return GridSpec(
        height=40,
        width=30,
        history_steps=8,
        future_steps=15,
        resolution_m=0.2,
    )


def _trajectory(*, x: float = 0.0) -> LocalTrajectory:
    grid = _grid()
    poses = np.zeros((grid.future_steps, 3), dtype=np.float32)
    poses[:, 0] = np.float32(x)
    zeros = np.zeros((grid.height, grid.width), dtype=np.float32)
    return LocalTrajectory(
        trajectory_id="trajectory-risk-toy",
        poses=poses,
        controls=np.zeros((grid.future_steps, 2), dtype=np.float32),
        swept_mask=zeros.copy(),
        tta_map=np.full_like(zeros, -1.0),
        braking_map=zeros.copy(),
        centerline_map=zeros.copy(),
        task_cost=0.0,
        metadata={"pose_time_layout_version": "future_endpoints_dt_to_horizon_v1"},
    )


def _circle_spec(*, radius_m: float = 0.5) -> dict[str, object]:
    return {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": radius_m},
    }


def _rectangle_spec(
    *, length_m: float = 2.0, width_m: float = 0.4
) -> dict[str, object]:
    return {
        "object_type": "carried_object",
        "footprint": {
            "kind": "rectangle",
            "length_m": length_m,
            "width_m": width_m,
        },
    }


def _constant_poses(x: float, y: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    poses = np.empty((_grid().future_steps, 3), dtype=np.float32)
    poses[:] = np.asarray([x, y, yaw], dtype=np.float32)
    return poses


def _world(
    trajectories: dict[str, np.ndarray],
    specs: dict[str, dict[str, object]],
) -> OracleWorld:
    grid = _grid()
    return OracleWorld(
        world_id="world-risk-toy",
        base_state_id="base-risk-toy",
        static_occupancy=np.zeros((grid.height, grid.width), dtype=np.float32),
        dynamic_object_trajectories=trajectories,
        dynamic_object_specs=specs,
        occluders=(),
        blind_spot_config={"kind": "toy"},
        random_seed=17,
        metadata={"schema_version": "3.0.0"},
    )


def _compute(
    world: OracleWorld,
    *,
    hidden_object_ids: tuple[str, ...],
    robot_footprint=CircleFootprint(0.5),
    **changes,
):
    kwargs = {
        "hidden_object_ids": hidden_object_ids,
        "robot_footprint": robot_footprint,
        "grid": _grid(),
        "future_dt_s": 0.2,
        "sigma_distance_m": 0.5,
        "sigma_time_s": 2.0,
        "near_miss_distance_m": 0.35,
    }
    kwargs.update(changes)
    return compute_hidden_risk_gt(_trajectory(), world, **kwargs)


def test_collision_uses_schema3_future_endpoint_time_and_forces_severity_one() -> None:
    world = _world(
        {"hidden": _constant_poses(0.8)},
        {"hidden": _circle_spec()},
    )

    result = _compute(world, hidden_object_ids=("hidden",))

    assert result.schema_version == "3.0.0"
    assert result.pose_time_layout_version == "future_endpoints_dt_to_horizon_v1"
    assert RISK_GT_VERSION == "hidden_risk_gt_schema3_v1"
    assert result.collision_label == 1
    assert result.near_miss == 0
    assert result.risk_severity == 1.0
    assert result.first_collision_time == pytest.approx(0.2)
    assert result.time_to_min_clearance == pytest.approx(0.2)
    assert result.min_clearance == pytest.approx(-0.2)


def test_zero_clearance_is_collision_and_near_miss_threshold_is_strict() -> None:
    touching = _world(
        {"hidden": _constant_poses(1.0)},
        {"hidden": _circle_spec()},
    )
    threshold = _world(
        {"hidden": _constant_poses(1.5)},
        {"hidden": _circle_spec()},
    )

    collision = _compute(touching, hidden_object_ids=("hidden",))
    safe = _compute(
        threshold,
        hidden_object_ids=("hidden",),
        near_miss_distance_m=0.5,
    )

    assert collision.min_clearance == pytest.approx(0.0)
    assert collision.collision_label == 1
    assert collision.near_miss == 0
    assert safe.min_clearance == pytest.approx(0.5)
    assert safe.collision_label == 0
    assert safe.near_miss == 0


@pytest.mark.parametrize(
    ("robot_footprint", "target_x", "target_radius", "expected_clearance"),
    [
        (CircleFootprint(0.5), 1.2, 0.5, 0.2),
        (RectangleFootprint(1.0, 1.0), 1.0, 0.25, 0.25),
    ],
    ids=("circle-circle", "circle-rectangle"),
)
def test_noncollision_circle_risk_matches_clearance_and_authoritative_formula(
    robot_footprint,
    target_x: float,
    target_radius: float,
    expected_clearance: float,
) -> None:
    world = _world(
        {"hidden": _constant_poses(target_x)},
        {"hidden": _circle_spec(radius_m=target_radius)},
    )

    result = _compute(
        world,
        hidden_object_ids=("hidden",),
        robot_footprint=robot_footprint,
    )

    expected_severity = math.exp(-expected_clearance / 0.5) * math.exp(-0.2 / 2.0)
    assert result.collision_label == 0
    assert result.near_miss == 1
    assert result.min_clearance == pytest.approx(expected_clearance)
    assert result.risk_severity == pytest.approx(expected_severity)
    assert result.first_collision_time is None
    assert result.time_to_min_clearance == pytest.approx(0.2)


def test_rotated_rectangle_yaw_is_used_for_signed_clearance() -> None:
    world = _world(
        {"hidden-rectangle": _constant_poses(0.9, yaw=np.pi / 2.0)},
        {"hidden-rectangle": _rectangle_spec()},
    )

    result = _compute(
        world,
        hidden_object_ids=("hidden-rectangle",),
        robot_footprint=RectangleFootprint(1.0, 1.0),
    )

    assert result.collision_label == 0
    assert result.near_miss == 1
    assert result.min_clearance == pytest.approx(0.2, abs=1e-6)
    assert result.critical_object_type == "carried_object"


def test_only_explicit_hidden_objects_contribute_to_labels() -> None:
    world = _world(
        {
            "visible-collision": _constant_poses(0.0),
            "hidden-safe": _constant_poses(2.0),
        },
        {
            "visible-collision": _circle_spec(),
            "hidden-safe": _circle_spec(),
        },
    )

    result = _compute(world, hidden_object_ids=("hidden-safe",))

    assert result.collision_label == 0
    assert result.near_miss == 0
    assert result.min_clearance == pytest.approx(1.0)
    assert result.critical_object_id == "hidden-safe"
    assert result.critical_object_type == "human"


def test_empty_hidden_set_uses_finite_grid_diagonal_sentinel() -> None:
    grid = _grid()
    expected = math.hypot(grid.width * grid.resolution_m, grid.height * grid.resolution_m)
    world = _world({}, {})

    result = _compute(world, hidden_object_ids=())

    assert resolve_no_object_clearance_sentinel(grid) == pytest.approx(expected)
    assert np.isfinite(result.min_clearance)
    assert result.min_clearance == pytest.approx(expected)
    assert result.collision_label == 0
    assert result.near_miss == 0
    assert result.risk_severity == 0.0
    assert result.first_collision_time is None
    assert result.time_to_min_clearance is None
    assert result.critical_object_id is None
    assert result.critical_object_type is None


def test_equal_minimum_clearance_uses_sorted_object_id_tie_break() -> None:
    world = _world(
        {
            "z-hidden": _constant_poses(2.0),
            "a-hidden": _constant_poses(2.0),
        },
        {
            "z-hidden": _circle_spec(),
            "a-hidden": _circle_spec(),
        },
    )

    result = _compute(
        world,
        hidden_object_ids=("z-hidden", "a-hidden"),
    )

    assert result.critical_object_id == "a-hidden"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"future_dt_s": 0.0}, "future_dt_s"),
        ({"future_dt_s": np.nan}, "future_dt_s"),
        ({"sigma_distance_m": 0.0}, "sigma_distance_m"),
        ({"sigma_time_s": np.inf}, "sigma_time_s"),
        ({"near_miss_distance_m": -0.1}, "near_miss_distance_m"),
    ],
)
def test_invalid_numeric_parameters_fail_closed(changes, message: str) -> None:
    world = _world(
        {"hidden": _constant_poses(2.0)},
        {"hidden": _circle_spec()},
    )

    with pytest.raises((TypeError, ValueError), match=message):
        _compute(world, hidden_object_ids=("hidden",), **changes)


@pytest.mark.parametrize(
    ("hidden_object_ids", "message"),
    [
        ("hidden", "hidden_object_ids"),
        (("hidden", "hidden"), "unique"),
        (("",), "non-empty"),
        (("missing",), "missing"),
    ],
)
def test_invalid_explicit_hidden_object_ids_fail_closed(
    hidden_object_ids,
    message: str,
) -> None:
    world = _world(
        {"hidden": _constant_poses(2.0)},
        {"hidden": _circle_spec()},
    )

    with pytest.raises((TypeError, ValueError), match=message):
        _compute(world, hidden_object_ids=hidden_object_ids)


@pytest.mark.parametrize("field", ["trajectory", "world"])
def test_nonfinite_trajectory_or_hidden_future_fails_closed(field: str) -> None:
    trajectory = _trajectory()
    world = _world(
        {"hidden": _constant_poses(2.0)},
        {"hidden": _circle_spec()},
    )
    if field == "trajectory":
        poses = trajectory.poses.copy()
        poses[0, 0] = np.nan
        trajectory = replace(trajectory, poses=poses)
    else:
        future = world.dynamic_object_trajectories["hidden"].copy()
        future[0, 0] = np.inf
        world = replace(world, dynamic_object_trajectories={"hidden": future})

    with pytest.raises(ValueError, match="finite|NaN/Inf"):
        compute_hidden_risk_gt(
            trajectory,
            world,
            hidden_object_ids=("hidden",),
            robot_footprint=CircleFootprint(0.5),
            grid=_grid(),
            future_dt_s=0.2,
            sigma_distance_m=0.5,
            sigma_time_s=2.0,
            near_miss_distance_m=0.35,
        )
