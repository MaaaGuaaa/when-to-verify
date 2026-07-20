"""Geometry and isolation tests for SOP-08 risk-label sidecars."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from src.contracts import (
    POSE_TIME_LAYOUT_VERSION,
    SCHEMA_VERSION,
    GridSpec,
    LocalTrajectory,
    OracleWorld,
    RiskSample,
)
from src.generation.risk_sidecars import (
    RiskLabelSidecar,
    build_risk_label_sidecar,
)
from src.geometry import (
    CircleFootprint,
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint,
)


HIDDEN_ID = "generated::human::hidden"
CONTEXT_ID = "recording::human::context"


def _grid() -> GridSpec:
    return GridSpec(
        height=17,
        width=17,
        history_steps=8,
        future_steps=15,
        resolution_m=0.25,
    )


def _poses(x_values: np.ndarray, *, y: float) -> np.ndarray:
    poses = np.empty((15, 3), dtype=np.float32)
    poses[:, 0] = np.asarray(x_values, dtype=np.float32)
    poses[:, 1] = np.float32(y)
    poses[:, 2] = np.float32(0.0)
    return poses


def _trajectory(grid: GridSpec) -> LocalTrajectory:
    poses = _poses(np.linspace(-1.4, 1.4, grid.future_steps), y=0.0)
    shape = (grid.height, grid.width)
    return LocalTrajectory(
        trajectory_id="trajectory-sidecar-toy",
        poses=poses,
        controls=np.zeros((grid.future_steps, 2), dtype=np.float32),
        swept_mask=np.zeros(shape, dtype=np.float32),
        tta_map=np.full(shape, -1.0, dtype=np.float32),
        braking_map=np.zeros(shape, dtype=np.float32),
        centerline_map=np.zeros(shape, dtype=np.float32),
        task_cost=0.0,
        metadata={"pose_time_layout_version": POSE_TIME_LAYOUT_VERSION},
    )


def _world(grid: GridSpec) -> OracleWorld:
    hidden = _poses(np.linspace(-1.0, 1.0, grid.future_steps), y=0.75)
    context = _poses(np.linspace(1.0, -1.0, grid.future_steps), y=-0.75)
    return OracleWorld(
        world_id="world-sidecar-toy",
        base_state_id="base-sidecar-toy",
        static_occupancy=np.zeros(
            (grid.height, grid.width), dtype=np.float32
        ),
        dynamic_object_trajectories={
            HIDDEN_ID: hidden,
            CONTEXT_ID: context,
        },
        dynamic_object_specs={
            HIDDEN_ID: {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.20},
            },
            CONTEXT_ID: {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.45},
            },
        },
        occluders=(),
        blind_spot_config={"kind": "test"},
        random_seed=13,
        metadata={"schema_version": SCHEMA_VERSION},
    )


def test_sidecar_rasterizes_only_declared_hidden_objects_and_inflated_robot() -> None:
    grid = _grid()
    trajectory = _trajectory(grid)
    world = _world(grid)
    base_robot = RectangleFootprint(length_m=0.50, width_m=0.30)
    inflated_robot = inflate_footprint(base_robot, 0.20)

    sidecar = build_risk_label_sidecar(
        sample_id="sample-sidecar-toy",
        trajectory=trajectory,
        world=world,
        hidden_object_ids=(HIDDEN_ID,),
        robot_footprint=inflated_robot,
        grid=grid,
        future_dt_s=0.2,
    )

    expected_hidden = np.stack(
        [
            rasterize_footprint(
                CircleFootprint(0.20), pose, grid
            ).astype(np.uint8)
            for pose in world.dynamic_object_trajectories[HIDDEN_ID]
        ]
    )
    context_only = np.stack(
        [
            rasterize_footprint(
                CircleFootprint(0.45), pose, grid
            ).astype(np.uint8)
            for pose in world.dynamic_object_trajectories[CONTEXT_ID]
        ]
    )
    expected_robot = np.stack(
        [
            rasterize_footprint(inflated_robot, pose, grid).astype(np.uint8)
            for pose in trajectory.poses
        ]
    )
    base_robot_masks = np.stack(
        [
            rasterize_footprint(base_robot, pose, grid).astype(np.uint8)
            for pose in trajectory.poses
        ]
    )

    np.testing.assert_array_equal(sidecar.hidden_risk_occupancy, expected_hidden)
    np.testing.assert_array_equal(sidecar.robot_future_footprints, expected_robot)
    assert np.any(context_only != expected_hidden)
    assert np.all(sidecar.hidden_risk_occupancy[context_only.astype(bool)] == 0)
    assert int(expected_robot.sum()) > int(base_robot_masks.sum())


def test_empty_blind_spot_sidecar_has_zero_hidden_occupancy() -> None:
    grid = _grid()

    sidecar = build_risk_label_sidecar(
        sample_id="sample-empty-blind-spot",
        trajectory=_trajectory(grid),
        world=_world(grid),
        hidden_object_ids=(),
        robot_footprint=RectangleFootprint(0.90, 0.70),
        grid=grid,
        future_dt_s=0.2,
    )

    assert sidecar.hidden_risk_occupancy.shape == (
        grid.future_steps,
        grid.height,
        grid.width,
    )
    assert not np.any(sidecar.hidden_risk_occupancy)
    assert np.any(sidecar.robot_future_footprints)


def test_sidecar_arrays_are_binary_uint8_owned_read_only_and_endpoint_float32() -> None:
    grid = _grid()
    sidecar = build_risk_label_sidecar(
        sample_id="sample-immutable-sidecar",
        trajectory=_trajectory(grid),
        world=_world(grid),
        hidden_object_ids=(HIDDEN_ID,),
        robot_footprint=RectangleFootprint(0.90, 0.70),
        grid=grid,
        future_dt_s=0.2,
    )

    expected_times = (
        np.arange(1, grid.future_steps + 1, dtype=np.float32)
        * np.float32(0.2)
    )
    for array in (
        sidecar.hidden_risk_occupancy,
        sidecar.robot_future_footprints,
        sidecar.future_endpoint_times_s,
    ):
        assert array.flags.owndata
        assert array.flags.c_contiguous
        assert not array.flags.writeable
    assert sidecar.hidden_risk_occupancy.dtype == np.uint8
    assert sidecar.robot_future_footprints.dtype == np.uint8
    assert sidecar.future_endpoint_times_s.dtype == np.float32
    assert set(np.unique(sidecar.hidden_risk_occupancy)).issubset({0, 1})
    assert set(np.unique(sidecar.robot_future_footprints)).issubset({0, 1})
    np.testing.assert_array_equal(sidecar.future_endpoint_times_s, expected_times)
    assert sidecar.future_endpoint_times_s[0] == pytest.approx(0.2)
    assert sidecar.future_endpoint_times_s[-1] == pytest.approx(3.0)


def test_sidecar_constructor_snapshots_arrays_and_rejects_nonbinary_masks() -> None:
    hidden = np.zeros((2, 3, 4), dtype=np.uint8)
    robot = np.ones((2, 3, 4), dtype=np.uint8)
    times = np.asarray([0.2, 0.4], dtype=np.float32)

    sidecar = RiskLabelSidecar(
        sample_id="sample-owned-arrays",
        hidden_risk_occupancy=hidden,
        robot_future_footprints=robot,
        future_endpoint_times_s=times,
    )
    hidden[0, 0, 0] = 1
    robot[0, 0, 0] = 0
    times[0] = 9.0

    assert sidecar.hidden_risk_occupancy[0, 0, 0] == 0
    assert sidecar.robot_future_footprints[0, 0, 0] == 1
    assert sidecar.future_endpoint_times_s[0] == pytest.approx(0.2)
    bad = np.zeros((2, 3, 4), dtype=np.uint8)
    bad[0, 0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        RiskLabelSidecar(
            sample_id="sample-invalid-mask",
            hidden_risk_occupancy=bad,
            robot_future_footprints=robot,
            future_endpoint_times_s=np.asarray([0.2, 0.4], dtype=np.float32),
        )


def test_sidecar_is_a_separate_label_type_not_a_risk_model_input() -> None:
    risk_fields = {field.name for field in fields(RiskSample)}

    assert not issubclass(RiskLabelSidecar, RiskSample)
    assert "hidden_risk_occupancy" not in risk_fields
    assert "robot_future_footprints" not in risk_fields
    assert "future_endpoint_times_s" not in risk_fields
