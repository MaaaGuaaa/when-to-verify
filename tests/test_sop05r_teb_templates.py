from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from src.contracts import BaseState, OracleContext, build_grid_spec
from src.geometry import (
    CircleOccluder,
    RectangleOccluder,
    point_signed_distance,
    segment_intersects_occluder,
)
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


def _evaluate(*, base_state=None, oracle_context=None, base_config=None, seed=31):
    from src.generation.sop05r_teb_templates import iter_sop05r_teb_task_templates

    default_base_config, default_base_state, default_oracle_context, teb_config = _inputs()
    return tuple(
        iter_sop05r_teb_task_templates(
            base_state=default_base_state if base_state is None else base_state,
            oracle_context=(
                default_oracle_context
                if oracle_context is None
                else oracle_context
            ),
            base_config=default_base_config if base_config is None else base_config,
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
    template_fields = {field.name.lower() for field in fields(accepted[0])}
    assert not any(
        forbidden in field
        for field in template_fields
        for forbidden in ("target", "snippet", "human")
    )
    assert {
        item.source_base_state_digest for item in accepted
    } == {before_base_digest}
    assert {
        item.source_oracle_context_digest for item in accepted
    } == {before_oracle_digest}

    min_clearance, _ = (
        teb_config.planner.represented_occluder_clearance_range_m
    )
    yaw_min, yaw_max = np.deg2rad(teb_config.template.relative_yaw_abs_range_deg)
    for item in accepted:
        assert item.direct_corridor_intrusion_m >= (
            teb_config.generation.minimum_direct_corridor_intrusion_m - 1e-6
        )
        assert any(
            segment_intersects_occluder(
                component,
                base_state.robot_history[-1:, :2],
                item.local_goal_world_pose[None, :2],
                epsilon_m=1e-9,
            )[0]
            for component in item.occluders
        )
        assert item.route.sampled_poses_world.shape == (40, 3)
        assert item.route.sampled_controls.shape == (40, 2)
        assert item.route.goal_arrival_time_s <= 8.0
        goal_distance = float(
            np.linalg.norm(
                item.local_goal_world_pose[:2] - base_state.robot_history[-1, :2]
            )
        )
        assert goal_distance in teb_config.template.goal_distances_m
        if item.family != "circle":
            assert yaw_min <= abs(item.relative_yaw_rad) <= yaw_max
        if item.family == "l_shape":
            assert len(item.occluders) == 2
            assert all(isinstance(component, RectangleOccluder) for component in item.occluders)
        for component in item.occluders:
            center_xy = (
                component.center_xy
                if isinstance(component, CircleOccluder)
                else component.pose[:2]
            )
            route_direction = item.local_goal_world_pose[:2].astype(np.float64)
            route_direction /= np.linalg.norm(route_direction)
            route_normal = np.asarray(
                [-route_direction[1], route_direction[0]], dtype=np.float64
            )
            assert abs(float(np.dot(center_xy, route_normal))) > 1e-6
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


def test_m4_rejects_out_of_grid_occluders_as_template_evaluations() -> None:
    base_config, base_state, oracle_context, _ = _inputs()
    constrained_config = dict(base_config)
    constrained_config["bev"] = dict(base_config["bev"], size=10)
    constrained_grid = build_grid_spec(constrained_config)
    constrained_state = replace(
        base_state,
        static_map_local=np.zeros(
            (constrained_grid.height, constrained_grid.width), dtype=np.float32
        ),
    )

    evaluations = _evaluate(
        base_state=constrained_state,
        oracle_context=oracle_context,
        base_config=constrained_config,
    )

    assert evaluations
    assert all(item.template is None for item in evaluations)
    assert {item.rejection_reason for item in evaluations} == {"occluder_out_of_bounds"}
