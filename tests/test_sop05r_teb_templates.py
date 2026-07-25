from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from src.contracts import BaseState, OracleContext, build_grid_spec
from src.geometry import RectangleOccluder, point_signed_distance
from src.generation.sop05r_contracts import load_sop05r_teb_config
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    base_config = load_config(ROOT / "configs/base.yaml")
    grid = build_grid_spec(base_config)
    base_state = BaseState(
        state_id="m4-template-base",
        split="train",
        recording_id="m4-template-recording",
        dynamic_object_ids=(),
        timestamp=12.0,
        robot_history=np.zeros((grid.history_steps, 3), dtype=np.float32),
        robot_state=np.asarray([0.0, 0.0], dtype=np.float32),
        visible_dynamic_object_history={},
        visible_dynamic_object_specs={},
        static_map_local=np.zeros((grid.height, grid.width), dtype=np.float32),
    )
    oracle_context = OracleContext(
        base_state_id=base_state.state_id,
        dynamic_object_history={},
        dynamic_object_future={},
        dynamic_object_specs={},
    )
    teb_config = load_sop05r_teb_config(
        ROOT / "configs/generator_obstacle_first_teb_train.yaml"
    )
    return base_config, base_state, oracle_context, teb_config


def _evaluate(*, base_state=None, oracle_context=None, seed=31):
    from src.generation.sop05r_teb_templates import iter_sop05r_teb_task_templates

    base_config, default_base_state, default_oracle_context, teb_config = _inputs()
    return tuple(
        iter_sop05r_teb_task_templates(
            base_state=default_base_state if base_state is None else base_state,
            oracle_context=(
                default_oracle_context
                if oracle_context is None
                else oracle_context
            ),
            base_config=base_config,
            teb_config=teb_config,
            seed=seed,
        )
    )


def test_m4_templates_are_deterministic_target_blind_and_preserve_inputs() -> None:
    from src.generation.sop05r_teb_templates import (
        canonical_sop05r_teb_base_state_digest,
        canonical_sop05r_teb_oracle_context_digest,
    )

    _, base_state, oracle_context, teb_config = _inputs()
    before_history = base_state.robot_history.copy()
    before_static = base_state.static_map_local.copy()
    before_base_digest = canonical_sop05r_teb_base_state_digest(base_state)
    before_oracle_digest = canonical_sop05r_teb_oracle_context_digest(oracle_context)

    first = _evaluate()
    second = _evaluate()

    assert [(item.template_id, item.rejection_reason) for item in first] == [
        (item.template_id, item.rejection_reason) for item in second
    ]
    assert np.array_equal(base_state.robot_history, before_history)
    assert np.array_equal(base_state.static_map_local, before_static)
    assert oracle_context.dynamic_object_history == {}
    assert canonical_sop05r_teb_base_state_digest(base_state) == before_base_digest
    assert (
        canonical_sop05r_teb_oracle_context_digest(oracle_context)
        == before_oracle_digest
    )

    accepted = [item.template for item in first if item.template is not None]
    assert accepted
    assert {item.family for item in accepted} == {"rectangle", "l_shape", "circle"}
    assert all(item.route is not None for item in accepted)
    assert {
        item.source_base_state_digest for item in accepted
    } == {before_base_digest}
    assert {
        item.source_oracle_context_digest for item in accepted
    } == {before_oracle_digest}

    min_clearance = teb_config.planner.represented_occluder_clearance_range_m[0]
    yaw_min, yaw_max = np.deg2rad(teb_config.template.relative_yaw_abs_range_deg)
    for item in accepted:
        assert 0.05 <= item.direct_corridor_intrusion_m <= 0.15
        if item.family != "circle":
            assert yaw_min <= abs(item.relative_yaw_rad) <= yaw_max
        if item.family == "l_shape":
            assert len(item.occluders) == 2
            assert all(isinstance(component, RectangleOccluder) for component in item.occluders)
        for component in item.occluders:
            clearance = point_signed_distance(
                component, item.route.sampled_poses_world[:, :2]
            ) - item.robot_radius_m
            assert float(np.min(clearance)) >= min_clearance - 1e-5


def test_m4_rejects_source_static_overlap_before_planning() -> None:
    _, base_state, _, _ = _inputs()
    blocked_base_state = replace(
        base_state,
        static_map_local=np.ones_like(base_state.static_map_local),
    )

    evaluations = _evaluate(base_state=blocked_base_state)

    assert evaluations
    assert all(item.template is None for item in evaluations)
    assert {item.rejection_reason for item in evaluations} == {"source_static_overlap"}
