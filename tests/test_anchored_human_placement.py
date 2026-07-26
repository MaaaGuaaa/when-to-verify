from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.contracts import BaseState, OracleContext, build_grid_spec
from src.datasets.long_snippet_library import LongMotionSnippet
from src.generation.sop05r_contracts import load_sop05r_teb_config
from src.generation.sop05r_teb_templates import iter_sop05r_teb_task_templates
from src.geometry import CircleOccluder, RectangleOccluder
from src.utils.config import load_config
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _m4_inputs():
    base_config = load_config(ROOT / "configs/base.yaml")
    grid = build_grid_spec(base_config)
    base_state = BaseState(
        state_id="m5-base",
        split="train",
        recording_id="m5-recording",
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


def _snippet() -> LongMotionSnippet:
    times = np.arange(40, dtype=np.float32) * np.float32(0.2)
    positions = np.column_stack((0.7 * times, 0.0 * times)).astype(np.float32)
    velocities = np.gradient(positions, 0.2, axis=0).astype(np.float32)
    return LongMotionSnippet(
        snippet_id="m5-human",
        split="train",
        source_recording_id="m5-recording",
        source_session_id="m5-session",
        source_object_id="m5-recording::human",
        object_type="human",
        footprint={"kind": "circle", "radius_m": 0.2},
        start_timestamp=0.0,
        positions=positions,
        velocities=velocities,
        headings=np.zeros(40, dtype=np.float32),
        duration_s=7.8,
        mean_speed_mps=0.7,
        max_acceleration_mps2=0.0,
        mean_abs_curvature_per_m=0.0,
        provenance={"fixture": "m5"},
    )


def test_rigid_anchor_rotation_preserves_anchor_distances_and_velocities() -> None:
    from src.generation.anchored_human_placement import (
        CollisionAnchor,
        apply_anchored_rigid_transform,
    )

    source_poses = np.zeros((23, 3), dtype=np.float32)
    source_poses[:, 0] = np.linspace(0.0, 2.2, 23, dtype=np.float32)
    source_poses[:, 1] = np.linspace(-0.5, 0.5, 23, dtype=np.float32)
    source_poses[:, 2] = 0.2
    source_velocities = np.tile(
        np.asarray([0.5, 0.25], dtype=np.float32),
        (23, 1),
    )
    anchor = CollisionAnchor(
        route_sample_index=6,
        route_time_s=1.4,
        world_position_xy=np.asarray([1.0, 2.0], dtype=np.float64),
        snippet_anchor_index=7,
        snippet_time_s=1.4,
    )

    transformed, velocities, translation = apply_anchored_rigid_transform(
        source_poses=source_poses,
        source_velocities=source_velocities,
        anchor=anchor,
        rotation_rad=0.5 * np.pi,
    )

    np.testing.assert_allclose(transformed[7, :2], anchor.world_position_xy, atol=1e-6)
    np.testing.assert_allclose(
        np.linalg.norm(
            source_poses[:, None, :2] - source_poses[None, :, :2],
            axis=-1,
        ),
        np.linalg.norm(
            transformed[:, None, :2] - transformed[None, :, :2],
            axis=-1,
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.linalg.norm(velocities, axis=1),
        np.linalg.norm(source_velocities, axis=1),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        transformed[:, 2],
        source_poses[:, 2] + 0.5 * np.pi,
        atol=1e-6,
    )
    np.testing.assert_allclose(translation, [1.0 + source_poses[7, 1], 1.3], atol=1e-6)


def test_synchronized_long40_anchor_index_uses_the_same_future_frame() -> None:
    from src.generation.anchored_human_placement import (
        synchronized_long40_anchor_index,
    )

    assert (
        synchronized_long40_anchor_index(
            route_time_s=0.2,
            current_index=7,
            future_dt_s=0.2,
            sample_count=40,
        )
        == 8
    )
    assert (
        synchronized_long40_anchor_index(
            route_time_s=6.4,
            current_index=7,
            future_dt_s=0.2,
            sample_count=40,
        )
        == 39
    )
    with pytest.raises(ValueError, match="future support"):
        synchronized_long40_anchor_index(
            route_time_s=6.6,
            current_index=7,
            future_dt_s=0.2,
            sample_count=40,
        )


def test_long40_resampling_preserves_layout_and_rejects_extrapolation() -> None:
    from src.generation.anchored_human_placement import resample_long_motion_snippet

    poses, velocities = resample_long_motion_snippet(_snippet(), time_scale=1.0)

    assert poses.shape == (40, 3)
    assert poses.dtype == np.float32
    assert velocities.shape == (40, 2)
    assert velocities.dtype == np.float32
    np.testing.assert_allclose(poses[:, :2], _snippet().positions)
    np.testing.assert_allclose(velocities, _snippet().velocities)
    with pytest.raises(ValueError, match="source_extrapolation_required"):
        resample_long_motion_snippet(_snippet(), time_scale=1.1)


def test_centerline_blocking_never_pairs_robot_and_target_across_times() -> None:
    from src.generation.anchored_human_placement import (
        synchronized_centerline_blocking,
    )

    circle = CircleOccluder(
        "circle",
        "tree_trunk",
        np.asarray([1.0, 1.0]),
        0.1,
    )
    robot_xy = np.asarray([[0.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    target_xy = np.asarray([[2.0, 0.0], [2.0, 2.0]], dtype=np.float64)

    blocked, blocker_ids = synchronized_centerline_blocking(
        robot_xy,
        target_xy,
        (circle,),
        epsilon_m=0.01,
    )

    assert blocked.tolist() == [False, False]
    assert blocker_ids == (None, None)


@pytest.mark.parametrize(
    "occluder",
    [
        CircleOccluder("circle", "tree_trunk", np.asarray([1.0, 1.0]), 0.1),
        RectangleOccluder(
            "rectangle",
            "wall",
            np.asarray([1.0, 1.0, 0.0]),
            0.2,
            0.2,
        ),
    ],
)
def test_centerline_blocking_accepts_one_later_witness(
    occluder: CircleOccluder | RectangleOccluder,
) -> None:
    from src.generation.anchored_human_placement import (
        synchronized_centerline_blocking,
    )

    blocked, blocker_ids = synchronized_centerline_blocking(
        np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        np.asarray([[2.0, 0.0], [2.0, 2.0]], dtype=np.float64),
        (occluder,),
        epsilon_m=0.01,
    )

    assert blocked.tolist() == [False, True]
    assert blocker_ids == (None, occluder.occluder_id)


def test_route_anchor_indices_prioritize_latest_valid_route_samples() -> None:
    from src.generation.anchored_human_placement import _route_anchor_indices

    route_times_s = np.arange(1, 11, dtype=np.float64) * 0.2
    task_template = SimpleNamespace(
        route=SimpleNamespace(
            band_poses_world=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            sampled_poses_world=np.column_stack(
                (
                    np.arange(1, 11, dtype=np.float64),
                    np.zeros(10, dtype=np.float64),
                    np.zeros(10, dtype=np.float64),
                )
            ),
            sample_times_s=route_times_s,
            goal_arrival_time_s=3.0,
        )
    )
    config = SimpleNamespace(
        generation=SimpleNamespace(
            collision_route_path_fraction_range=(0.2, 0.95),
            max_route_anchor_candidates=3,
        ),
        trajectory=SimpleNamespace(future_horizon_s=6.4),
    )

    assert _route_anchor_indices(task_template, config) == (8, 7, 6)


def test_b_outward_primary_and_expansion_cover_the_full_a_relative_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.generation import anchored_human_placement as placement_module
    from src.generation.anchored_human_placement import (
        CollisionAnchor,
        construct_visibility_guided_rotation_angles,
    )

    source_poses = np.zeros((40, 3), dtype=np.float32)
    source_poses[3, :2] = np.asarray([1.0, 0.0], dtype=np.float32)
    anchor = CollisionAnchor(
        route_sample_index=8,
        route_time_s=1.8,
        world_position_xy=np.asarray([0.0, 0.0], dtype=np.float64),
        snippet_anchor_index=7,
        snippet_time_s=1.8,
    )
    robot_xy = np.zeros((40, 2), dtype=np.float64)
    robot_xy[3] = np.asarray([1.0, 0.0], dtype=np.float64)
    obstacle_angle_deg = 89.416
    occluder = CircleOccluder(
        "neighborhood-shadow",
        "tree_trunk",
        np.asarray(
            [
                0.5 * np.cos(np.deg2rad(obstacle_angle_deg)),
                0.5 * np.sin(np.deg2rad(obstacle_angle_deg)),
            ],
            dtype=np.float64,
        ),
        0.05,
    )
    monkeypatch.setattr(
        placement_module,
        "segment_intersects_occluder",
        lambda *_args, **_kwargs: pytest.fail(
            "the half-plane constructor must not scan trajectory intersections"
        ),
    )

    primary = construct_visibility_guided_rotation_angles(
        source_poses=source_poses,
        anchor=anchor,
        robot_positions_xy=robot_xy,
        occluders=(occluder,),
        guide_history_index=3,
        occluder_angular_margin_step_deg=10.0,
        epsilon_m=0.01,
    )
    secondary = construct_visibility_guided_rotation_angles(
        source_poses=source_poses,
        anchor=anchor,
        robot_positions_xy=robot_xy,
        occluders=(occluder,),
        guide_history_index=3,
        occluder_angular_margin_step_deg=10.0,
        search_stage="secondary",
        epsilon_m=0.01,
    )

    grid_degrees = tuple(range(-180, 180, 10))
    side = 1 if obstacle_angle_deg >= 0.0 else -1

    def circular_distance_deg(angle_deg: int) -> float:
        return abs((angle_deg - obstacle_angle_deg + 180.0) % 360.0 - 180.0)

    expected_primary = tuple(
        sorted(
            (
                angle_deg
                for angle_deg in grid_degrees
                if side * angle_deg > abs(obstacle_angle_deg)
            ),
            key=lambda angle_deg: side * angle_deg,
        )
    )
    expected_secondary = tuple(
        sorted(
            (
                angle_deg
                for angle_deg in grid_degrees
                if angle_deg not in expected_primary
            ),
            key=lambda angle_deg: (circular_distance_deg(angle_deg), angle_deg),
        )
    )

    assert primary.rejection_reason is None
    assert secondary.rejection_reason is None
    assert primary.theta_align_rad == pytest.approx(0.0, abs=1e-8)
    assert np.rad2deg(primary.signed_ab_rad) == pytest.approx(obstacle_angle_deg)
    assert np.rad2deg(np.asarray(primary.angles_rad)) == pytest.approx(
        expected_primary
    )
    assert np.rad2deg(np.asarray(secondary.angles_rad)) == pytest.approx(
        expected_secondary
    )
    assert np.rad2deg(np.asarray(primary.angles_rad))[0] == pytest.approx(90.0)
    assert np.any(
        np.isclose(np.rad2deg(np.asarray(primary.angles_rad)), 120.0, atol=1e-8)
    )
    assert len(primary.angles_rad) + len(secondary.angles_rad) == len(grid_degrees)


def test_first_fit_expands_b_neighborhood_after_primary_physics_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.generation import anchored_human_placement as placement_module
    from src.generation.anchored_human_placement import (
        VisibilityGuidedRotationAngles,
        solve_anchored_human_placement,
    )

    base_config, base_state, oracle_context, teb_config = _m4_inputs()
    template = next(
        item.template
        for item in iter_sop05r_teb_task_templates(
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
            teb_config=teb_config,
            seed=9,
        )
        if item.template is not None
    )
    stages: list[str | None] = []
    active_stage: str | None = None

    def staged_angles(**kwargs):
        nonlocal active_stage
        active_stage = kwargs.get("search_stage")
        stages.append(active_stage)
        return VisibilityGuidedRotationAngles(
            theta_align_rad=0.0,
            signed_ab_rad=0.2,
            half_plane_side=1,
            angles_rad=(0.0,),
            angular_margins_deg=(0.0,),
            rejection_reason=None,
        )

    def eligible_history(robot_positions, target_positions, occluders, *, epsilon_m):
        del robot_positions, occluders, epsilon_m
        blocked = np.zeros(target_positions.shape[:2], dtype=np.bool_)
        blocked[:, 4] = True
        blocker_indices = np.full(blocked.shape, -1, dtype=np.int16)
        blocker_indices[:, 4] = 0
        return blocked, blocker_indices

    monkeypatch.setattr(
        placement_module,
        "construct_visibility_guided_rotation_angles",
        staged_angles,
    )
    monkeypatch.setattr(placement_module, "_batched_centerline_blocking", eligible_history)
    monkeypatch.setattr(
        placement_module,
        "_physics_rejection",
        lambda **_: None if active_stage == "secondary" else "target_occluder_collision",
    )

    evaluation = solve_anchored_human_placement(
        task_template=template,
        snippet=_snippet(),
        base_state=base_state,
        oracle_context=oracle_context,
        base_config=base_config,
        teb_config=teb_config,
        seed=27,
    )

    assert evaluation.result is not None
    assert stages[:2] == ["primary", "secondary"]


def test_first_fit_keeps_later_route_anchor_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.generation import anchored_human_placement as placement_module
    from src.generation.anchored_human_placement import (
        VisibilityGuidedRotationAngles,
        solve_anchored_human_placement,
    )

    base_config, base_state, oracle_context, teb_config = _m4_inputs()
    template = next(
        item.template
        for item in iter_sop05r_teb_task_templates(
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
            teb_config=teb_config,
            seed=9,
        )
        if item.template is not None
    )

    class ReversePermutationRng:
        def permutation(self, values):
            if isinstance(values, int):
                return np.arange(values)[::-1]
            return np.asarray(values)[::-1]

    def eligible_history(robot_positions, target_positions, occluders, *, epsilon_m):
        del robot_positions, occluders, epsilon_m
        blocked = np.zeros(target_positions.shape[:2], dtype=np.bool_)
        blocked[:, 4] = True
        blocker_indices = np.full(blocked.shape, -1, dtype=np.int16)
        blocker_indices[:, 4] = 0
        return blocked, blocker_indices

    monkeypatch.setattr(placement_module, "_route_anchor_indices", lambda *_: (8, 7))
    monkeypatch.setattr(
        placement_module.np.random,
        "default_rng",
        lambda *_: ReversePermutationRng(),
    )
    monkeypatch.setattr(
        placement_module,
        "construct_visibility_guided_rotation_angles",
        lambda **_: VisibilityGuidedRotationAngles(
            theta_align_rad=0.0,
            signed_ab_rad=0.2,
            half_plane_side=1,
            angles_rad=(0.0,),
            angular_margins_deg=(10.0,),
            rejection_reason=None,
        ),
    )
    monkeypatch.setattr(placement_module, "_batched_centerline_blocking", eligible_history)
    monkeypatch.setattr(placement_module, "_physics_rejection", lambda **_: None)

    evaluation = solve_anchored_human_placement(
        task_template=template,
        snippet=_snippet(),
        base_state=base_state,
        oracle_context=oracle_context,
        base_config=base_config,
        teb_config=teb_config,
        seed=27,
    )

    assert evaluation.result is not None
    assert evaluation.result.placement.anchor.route_sample_index == 8


def test_first_fit_constructs_one_half_plane_margin_batch_per_anchor_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.generation import anchored_human_placement as placement_module
    from src.generation.anchored_human_placement import (
        VisibilityGuidedRotationAngles,
        solve_anchored_human_placement,
    )

    base_config, base_state, oracle_context, teb_config = _m4_inputs()
    template = next(
        item.template
        for item in iter_sop05r_teb_task_templates(
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
            teb_config=teb_config,
            seed=9,
        )
        if item.template is not None
    )
    construction_calls = 0

    def margin_angles(**_):
        nonlocal construction_calls
        construction_calls += 1
        return VisibilityGuidedRotationAngles(
            theta_align_rad=0.0,
            signed_ab_rad=0.2,
            half_plane_side=1,
            angles_rad=(0.0,),
            angular_margins_deg=(10.0,),
            rejection_reason=None,
        )

    def primary_history(robot_positions, target_positions, occluders, *, epsilon_m):
        del robot_positions, occluders, epsilon_m
        blocked = np.zeros(target_positions.shape[:2], dtype=np.bool_)
        blocked[:, 4] = True
        blocker_indices = np.full(blocked.shape, -1, dtype=np.int16)
        blocker_indices[:, 4] = 0
        return blocked, blocker_indices

    monkeypatch.setattr(
        placement_module,
        "construct_visibility_guided_rotation_angles",
        margin_angles,
    )
    monkeypatch.setattr(
        placement_module,
        "_batched_centerline_blocking",
        primary_history,
    )
    monkeypatch.setattr(placement_module, "_physics_rejection", lambda **_: None)

    evaluation = solve_anchored_human_placement(
        task_template=template,
        snippet=_snippet(),
        base_state=base_state,
        oracle_context=oracle_context,
        base_config=base_config,
        teb_config=teb_config,
        seed=27,
    )

    assert evaluation.result is not None
    assert construction_calls == 1


def test_first_fit_orders_preferred_history_before_fallback_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.generation import anchored_human_placement as placement_module
    from src.generation.anchored_human_placement import (
        VisibilityGuidedRotationAngles,
        solve_anchored_human_placement,
    )

    base_config, base_state, oracle_context, teb_config = _m4_inputs()
    template = next(
        item.template
        for item in iter_sop05r_teb_task_templates(
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
            teb_config=teb_config,
            seed=9,
        )
        if item.template is not None
    )

    monkeypatch.setattr(
        placement_module,
        "construct_visibility_guided_rotation_angles",
        lambda **_: VisibilityGuidedRotationAngles(
            theta_align_rad=0.0,
            signed_ab_rad=0.2,
            half_plane_side=1,
            angles_rad=(0.0, 1.0),
            angular_margins_deg=(10.0, 20.0),
            rejection_reason=None,
        ),
    )

    def fallback_before_primary(robot_positions, target_positions, occluders, *, epsilon_m):
        del robot_positions, occluders, epsilon_m
        blocked = np.zeros(target_positions.shape[:2], dtype=np.bool_)
        blocked[0, 1] = True
        blocked[1, 4] = True
        blocker_indices = np.full(blocked.shape, -1, dtype=np.int16)
        blocker_indices[0, 1] = 0
        blocker_indices[1, 4] = 0
        return blocked, blocker_indices

    monkeypatch.setattr(
        placement_module,
        "_batched_centerline_blocking",
        fallback_before_primary,
    )
    monkeypatch.setattr(placement_module, "_physics_rejection", lambda **_: None)

    evaluation = solve_anchored_human_placement(
        task_template=template,
        snippet=_snippet(),
        base_state=base_state,
        oracle_context=oracle_context,
        base_config=base_config,
        teb_config=teb_config,
        seed=27,
    )

    assert evaluation.result is not None
    assert evaluation.result.placement.rotation_rad == pytest.approx(1.0)
    assert evaluation.result.visibility.preferred


def test_first_fit_rejects_history_without_seen_then_occluded_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.generation import anchored_human_placement as placement_module
    from src.generation.anchored_human_placement import (
        VisibilityGuidedRotationAngles,
        solve_anchored_human_placement,
    )

    base_config, base_state, oracle_context, teb_config = _m4_inputs()
    template = next(
        item.template
        for item in iter_sop05r_teb_task_templates(
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
            teb_config=teb_config,
            seed=9,
        )
        if item.template is not None
    )
    assert template is not None

    def always_hidden(robot_positions, target_positions, occluders, *, epsilon_m):
        del robot_positions, occluders, epsilon_m
        blocked = np.ones(target_positions.shape[:2], dtype=np.bool_)
        return blocked, np.zeros(blocked.shape, dtype=np.int16)

    monkeypatch.setattr(
        placement_module,
        "_batched_centerline_blocking",
        always_hidden,
    )
    monkeypatch.setattr(
        placement_module,
        "construct_visibility_guided_rotation_angles",
        lambda **_: VisibilityGuidedRotationAngles(
            theta_align_rad=0.0,
            signed_ab_rad=0.2,
            half_plane_side=1,
            angles_rad=(0.0,),
            angular_margins_deg=(10.0,),
            rejection_reason=None,
        ),
    )
    monkeypatch.setattr(placement_module, "_physics_rejection", lambda **_: None)

    evaluation = solve_anchored_human_placement(
        task_template=template,
        snippet=_snippet(),
        base_state=base_state,
        oracle_context=oracle_context,
        base_config=base_config,
        teb_config=teb_config,
        seed=27,
    )

    assert evaluation.result is None


def test_first_fit_synchronizes_route_and_human_long40_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.generation import anchored_human_placement as placement_module
    from src.generation.anchored_human_placement import (
        VisibilityGuidedRotationAngles,
        solve_anchored_human_placement,
    )

    base_config, base_state, oracle_context, teb_config = _m4_inputs()
    template = next(
        item.template
        for item in iter_sop05r_teb_task_templates(
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
            teb_config=teb_config,
            seed=9,
        )
        if item.template is not None
    )
    assert template is not None

    def only_mid_history_is_blocked(
        robot_positions, target_positions, occluders, *, epsilon_m
    ):
        del robot_positions, occluders, epsilon_m
        blocked = np.zeros(target_positions.shape[:2], dtype=np.bool_)
        blocked[:, 3] = True
        blocker_indices = np.full(blocked.shape, -1, dtype=np.int16)
        blocker_indices[:, 3] = 0
        return blocked, blocker_indices

    monkeypatch.setattr(
        placement_module,
        "_route_anchor_indices",
        lambda *_: (8,),
    )
    monkeypatch.setattr(
        placement_module,
        "_batched_centerline_blocking",
        only_mid_history_is_blocked,
    )
    monkeypatch.setattr(
        placement_module,
        "construct_visibility_guided_rotation_angles",
        lambda **_: VisibilityGuidedRotationAngles(
            theta_align_rad=0.0,
            signed_ab_rad=0.2,
            half_plane_side=1,
            angles_rad=(0.0,),
            angular_margins_deg=(10.0,),
            rejection_reason=None,
        ),
    )
    monkeypatch.setattr(placement_module, "_physics_rejection", lambda **_: None)

    evaluation = solve_anchored_human_placement(
        task_template=template,
        snippet=_snippet(),
        base_state=base_state,
        oracle_context=oracle_context,
        base_config=base_config,
        teb_config=teb_config,
        seed=27,
    )

    assert evaluation.result is not None
    anchor = evaluation.result.placement.anchor
    assert anchor.snippet_anchor_index == 7 + int(
        round(anchor.route_time_s / 0.2)
    )
    assert evaluation.result.placement.provenance["decision_time_s"] == 0.0


def test_first_fit_solver_is_deterministic_and_aligns_any_acceptance() -> None:
    from src.generation.anchored_human_placement import solve_anchored_human_placement

    base_config, base_state, oracle_context, teb_config = _m4_inputs()
    templates = [
        item.template
        for item in iter_sop05r_teb_task_templates(
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
            teb_config=teb_config,
            seed=9,
        )
        if item.template is not None
    ]
    evaluations = [
        solve_anchored_human_placement(
            task_template=template,
            snippet=_snippet(),
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
            teb_config=teb_config,
            seed=27,
        )
        for template in templates
    ]
    repeated = [
        solve_anchored_human_placement(
            task_template=template,
            snippet=_snippet(),
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
            teb_config=teb_config,
            seed=27,
        )
        for template in templates
    ]

    def stable_outcome(item):
        anchor = None if item.result is None else item.result.placement.anchor
        return (
            item.rejection_reason,
            item.attempted_candidates,
            None
            if anchor is None
            else (
                anchor.route_sample_index,
                anchor.route_time_s,
                tuple(anchor.world_position_xy.tolist()),
                anchor.snippet_anchor_index,
                anchor.snippet_time_s,
            ),
        )

    assert [stable_outcome(item) for item in evaluations] == [
        stable_outcome(item) for item in repeated
    ]
    for accepted in (item.result for item in evaluations if item.result is not None):
        placement = accepted.placement
        witness = accepted.witness
        assert 0 <= witness.sample_index <= 7
        assert witness.time_s == pytest.approx(
            placement.provenance["decision_time_s"]
            + (witness.sample_index - 7) * 0.2
        )
        assert (
            placement.anchor.route_time_s - placement.provenance["decision_time_s"]
            >= 1.2
        )
        assert accepted.visibility.eligible
