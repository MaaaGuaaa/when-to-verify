"""Structural guardrails for target-blind Long40 planning."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_static_teb_execution_does_not_construct_decision_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.contracts as contracts
    import src.planning.lightweight_teb as teb
    import src.planning.query_maps as query_maps
    from src.contracts import build_grid_spec
    from src.generation.sop05r_contracts import load_sop05r_teb_config
    from src.utils.config import load_config

    source = inspect.getsource(teb)
    assert "LocalTrajectory" not in source
    assert "query_maps" not in source
    planner = getattr(teb, "plan_static_lightweight_teb", None)
    assert callable(planner)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("M3 must not construct decision-relative products")

    monkeypatch.setattr(contracts, "LocalTrajectory", forbidden)
    monkeypatch.setattr(query_maps, "build_local_trajectory", forbidden)
    monkeypatch.setattr(query_maps, "build_trajectory_query_maps", forbidden)
    base_config = load_config(ROOT / "configs/base.yaml")
    planner_config = load_sop05r_teb_config(
        ROOT / "configs/generator_obstacle_first_teb_train.yaml"
    ).planner
    grid = build_grid_spec(base_config)
    request = teb.StaticTebRequest(
        start_pose=np.zeros(3, dtype=np.float32),
        initial_control=np.zeros(2, dtype=np.float32),
        local_goal_world_pose=np.asarray([2.4, 0.0, 0.0], dtype=np.float32),
        static_occupancy=np.zeros((grid.height, grid.width), dtype=np.float32),
        occluders=(),
        base_config=base_config,
        planner_config=planner_config,
    )

    result = planner(request)

    assert result.route is not None, result.rejection_reason
