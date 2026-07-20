from pathlib import Path

import numpy as np
import pytest

from src.contracts import GridSpec
from src.geometry import (
    CircleFootprint,
    RectangleFootprint,
    rasterize_footprint,
)
from src.planning.verification_actions import (
    CANONICAL_ACTION_IDS,
    action_cost,
    action_endpoint,
    check_action_feasibility,
    load_verification_actions,
    sample_action_trace,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "verification_actions.yaml"


def _grid() -> GridSpec:
    return GridSpec(
        height=40,
        width=40,
        history_steps=8,
        future_steps=15,
        resolution_m=0.1,
    )


def _future_pose(x: float, y: float, yaw: float) -> np.ndarray:
    return np.tile(
        np.asarray([x, y, yaw], dtype=np.float32),
        (16, 1),
    )


def test_canonical_action_order_vectors_and_analytic_endpoints():
    library = load_verification_actions(CONFIG)
    actions = library.actions

    assert tuple(action.action_id for action in actions) == CANONICAL_ACTION_IDS
    assert CANONICAL_ACTION_IDS == (
        "yaw_left_10",
        "yaw_right_10",
        "yaw_left_20",
        "yaw_right_20",
        "forward_peek",
        "stop_scan",
    )
    assert all(action.vector.shape == (3,) for action in actions)
    assert all(action.vector.dtype == np.float32 for action in actions)

    start = np.asarray([1.0, 2.0, np.pi / 2.0], dtype=np.float32)
    by_id = library.by_id
    np.testing.assert_allclose(
        action_endpoint(start, by_id["yaw_left_10"]),
        np.asarray([1.0, 2.0, np.pi / 2.0 + np.deg2rad(10.0)]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        action_endpoint(start, by_id["yaw_right_20"]),
        np.asarray([1.0, 2.0, np.pi / 2.0 - np.deg2rad(20.0)]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        action_endpoint(start, by_id["forward_peek"]),
        np.asarray([1.0, 2.30, np.pi / 2.0]),
        atol=1e-6,
    )
    np.testing.assert_allclose(action_endpoint(start, by_id["stop_scan"]), start)

    yaw_trace = sample_action_trace(start, by_id["yaw_left_20"])
    assert np.all(yaw_trace.poses[:, :2] == start[:2])
    np.testing.assert_allclose(yaw_trace.poses[-1], action_endpoint(start, by_id["yaw_left_20"]))
    assert yaw_trace.times_s[0] == 0.0
    assert yaw_trace.times_s[-1] == by_id["yaw_left_20"].duration_s


def test_action_cost_uses_duration_distance_and_yaw_once():
    action = load_verification_actions(CONFIG).by_id["yaw_left_20"]
    cost = action_cost(
        action,
        {
            "lambda_time": 0.04,
            "lambda_distance": 0.05,
            "lambda_yaw_per_deg": 0.0015,
        },
    )
    assert cost == pytest.approx(0.04 * 0.75 + 0.0015 * 20.0)


def test_static_and_typed_dynamic_feasibility_cover_rotated_rectangles():
    grid = _grid()
    library = load_verification_actions(CONFIG)
    robot = RectangleFootprint(length_m=0.70, width_m=0.55)
    empty = np.zeros((grid.height, grid.width), dtype=np.float32)

    static_blocked = rasterize_footprint(
        CircleFootprint(0.12), np.asarray([0.30, 0.0, 0.0]), grid
    ).astype(np.float32)
    blocked = check_action_feasibility(
        np.zeros(3, dtype=np.float32),
        library.by_id["forward_peek"],
        robot_footprint=robot,
        static_occupancy=static_blocked,
        grid=grid,
        dynamic_object_poses={},
        dynamic_object_footprints={},
        dynamic_dt_s=0.2,
    )
    assert not blocked.feasible
    assert blocked.reason == "static_collision"

    dynamic_blocked = check_action_feasibility(
        np.zeros(3, dtype=np.float32),
        library.by_id["stop_scan"],
        robot_footprint=robot,
        static_occupancy=empty,
        grid=grid,
        dynamic_object_poses={
            "cart": _future_pose(0.38, 0.0, np.pi / 4.0),
        },
        dynamic_object_footprints={
            "cart": RectangleFootprint(length_m=0.80, width_m=0.20),
        },
        dynamic_dt_s=0.2,
    )
    assert not dynamic_blocked.feasible
    assert dynamic_blocked.reason == "dynamic_collision"
    assert dynamic_blocked.critical_object_id == "cart"

    safe = check_action_feasibility(
        np.zeros(3, dtype=np.float32),
        library.by_id["stop_scan"],
        robot_footprint=robot,
        static_occupancy=empty,
        grid=grid,
        dynamic_object_poses={"person": _future_pose(1.5, 0.0, 0.0)},
        dynamic_object_footprints={"person": CircleFootprint(0.30)},
        dynamic_dt_s=0.2,
    )
    assert safe.feasible
    assert safe.reason is None
    assert safe.minimum_dynamic_clearance_m > 0.0


def test_loader_rejects_duplicate_or_noncanonical_action_ids(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
schema_version: "3.0.0"
library_version: verification_actions_v1
actions:
  - {action_id: yaw_left_10, duration_s: 0.5, delta_forward_m: 0.0, delta_yaw_deg: 10.0}
  - {action_id: yaw_left_10, duration_s: 0.5, delta_forward_m: 0.0, delta_yaw_deg: -10.0}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical action IDs"):
        load_verification_actions(bad)
