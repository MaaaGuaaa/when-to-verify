"""Audit-only joint occluder search contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.evaluation.seen_occluded_joint_search as joint_module
from src.contracts import build_grid_spec
from src.generation.event_sampler import load_generator_config
from src.generation.paired_variants import load_paired_variant_config
from tests.test_pair_variants import _mother_inputs


ROOT = Path(__file__).resolve().parents[1]


def test_joint_audit_config_is_strict_and_deterministic() -> None:
    first = joint_module.load_joint_audit_search_config(
        ROOT / "configs" / "seen_occluded_joint_visual_audit.yaml"
    )
    repeated = joint_module.load_joint_audit_search_config(
        ROOT / "configs" / "seen_occluded_joint_visual_audit.yaml"
    )

    assert first == repeated
    assert first.algorithm_version == (
        "seen_occluded_joint_visual_audit_v4"
    )
    assert first.shared_los_fractions == (0.2, 0.35, 0.5, 0.65, 0.8, 0.9)
    assert first.length_quantiles == (1.0, 0.5, 0.0)
    assert first.width_quantiles == (0.0,)
    assert first.longitudinal_center_quantiles == (
        0.5,
        0.25,
        0.75,
        0.1,
        0.9,
    )
    assert first.occluder_type_order == (
        "pillar",
        "source",
        "wall",
        "shelf",
    )
    assert len(first.digest) == 32


def test_shared_los_offsets_stay_inside_both_target_rays() -> None:
    offsets = joint_module.shared_los_normal_offsets(
        collision_current_pose=np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
        temporal_current_pose=np.asarray([3.0, 0.8, 0.0], dtype=np.float32),
        trajectory_normal=np.asarray([0.0, 1.0], dtype=np.float64),
        source_normal_coordinate_m=0.3,
        fractions=(0.2, 0.5, 0.9),
    )

    assert offsets == pytest.approx((0.3, 0.16, 0.4, 0.72))
    assert all(0.0 < value < 0.8 for value in offsets)


def test_joint_placement_uses_robot_frame_los_normal_coordinate(
    monkeypatch,
) -> None:
    config, _, _, _, _, _, mother = _mother_inputs()
    generator_config = load_generator_config(
        ROOT / "configs" / "generator_seen_occluded_visual_audit.yaml"
    )
    joint_config = joint_module.load_joint_audit_search_config(
        ROOT / "configs" / "seen_occluded_joint_visual_audit.yaml"
    )
    collision_target = replace(
        mother.target,
        current_pose=np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
    )
    temporal_target = replace(
        mother.target,
        current_pose=np.asarray([3.0, 0.8, 0.0], dtype=np.float32),
    )
    candidate = SimpleNamespace(
        target=temporal_target,
        prepared_sweeps=(),
        trajectory_normal=np.asarray([0.0, 1.0], dtype=np.float64),
        conflict_point=np.asarray([0.0, 10.0], dtype=np.float64),
        normal_coordinates_m=(0.4,),
    )
    environment = SimpleNamespace(
        base_static_occupancy=np.zeros_like(
            mother.world.static_occupancy, dtype=np.bool_
        ),
        scene_history_footprints={},
        scene_dynamic_history={},
        grid=build_grid_spec(config),
    )
    diagnostics = joint_module.Counter()
    monkeypatch.setattr(
        joint_module,
        "points_in_grid",
        lambda *_args, **_kwargs: np.asarray([False], dtype=np.bool_),
    )

    placements = list(
        joint_module._iter_certified_joint_placements(
            mother_event=replace(mother, target=collision_target),
            temporal_candidate=candidate,
            environment=environment,
            generator_config=generator_config,
            joint_config=joint_config,
            diagnostics=diagnostics,
        )
    )

    assert placements == []
    assert diagnostics["placement:offset_outside_los"] == 0
    assert diagnostics["placement:candidate"] > 0


def test_longitudinal_centers_stay_in_shared_los_coverage_interval() -> None:
    intersections = np.asarray(
        [[1.0, -0.2], [1.0, 0.2]], dtype=np.float64
    )

    centers = joint_module.longitudinal_center_candidates(
        base_center=np.asarray([1.0, 0.0], dtype=np.float64),
        intersections=intersections,
        yaw=0.5 * np.pi,
        length_m=1.0,
        quantiles=(0.0, 0.5, 1.0),
    )

    np.testing.assert_allclose(
        np.asarray(centers),
        np.asarray(((1.0, 0.0), (1.0, -0.3), (1.0, 0.3))),
        rtol=0.0,
        atol=1e-12,
    )
    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    for center in centers:
        offsets = np.abs((intersections - center) @ axis)
        assert np.all(offsets <= 0.5 + 1e-12)


def test_longitudinal_refinement_waits_for_certified_base_center(
    monkeypatch,
) -> None:
    config, _, _, _, _, _, mother = _mother_inputs()
    generator_config = load_generator_config(
        ROOT / "configs" / "generator_seen_occluded_visual_audit.yaml"
    )
    joint_config = joint_module.load_joint_audit_search_config(
        ROOT / "configs" / "seen_occluded_joint_visual_audit.yaml"
    )
    collision_target = replace(
        mother.target,
        current_pose=np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
    )
    temporal_target = replace(
        mother.target,
        current_pose=np.asarray([3.0, 0.8, 0.0], dtype=np.float32),
    )
    candidate = SimpleNamespace(
        target=temporal_target,
        prepared_sweeps=(),
        trajectory_normal=np.asarray([0.0, 1.0], dtype=np.float64),
        normal_coordinates_m=(0.4,),
    )
    environment = SimpleNamespace(
        base_static_occupancy=np.zeros_like(
            mother.world.static_occupancy, dtype=np.bool_
        ),
        scene_history_footprints={},
        scene_dynamic_history={},
        grid=build_grid_spec(config),
    )
    checked_centers = []

    def centers(*, base_center, **_kwargs):
        base = np.asarray(base_center, dtype=np.float64)
        return (base, base + np.asarray([100.0, 0.0]))

    def vertices(_footprint, pose):
        return np.asarray([pose[:2]], dtype=np.float64)

    def reject_grid(points, _grid):
        checked_centers.append(np.asarray(points[0], dtype=np.float64))
        return np.asarray([False], dtype=np.bool_)

    monkeypatch.setattr(
        joint_module, "longitudinal_center_candidates", centers
    )
    monkeypatch.setattr(joint_module, "footprint_vertices", vertices)
    monkeypatch.setattr(joint_module, "points_in_grid", reject_grid)

    placements = list(
        joint_module._iter_certified_joint_placements(
            mother_event=replace(mother, target=collision_target),
            temporal_candidate=candidate,
            environment=environment,
            generator_config=generator_config,
            joint_config=joint_config,
            diagnostics=joint_module.Counter(),
        )
    )

    assert placements == []
    assert checked_centers
    assert all(abs(center[0]) < 50.0 for center in checked_centers)


def test_joint_search_continues_after_unseen_rebound(
    monkeypatch,
) -> None:
    config, _, base, oracle, trajectory, snippet, mother = _mother_inputs()
    paired_config = load_paired_variant_config(
        ROOT / "configs" / "paired_variants_visual_audit.yaml"
    )
    generator_config = load_generator_config(
        ROOT / "configs" / "generator_seen_occluded_visual_audit.yaml"
    )
    joint_config = joint_module.load_joint_audit_search_config(
        ROOT / "configs" / "seen_occluded_joint_visual_audit.yaml"
    )
    temporal = SimpleNamespace(
        offset_s=-1.2,
        target=mother.target,
        prepared_sweeps=(),
        normal_coordinates_m=(0.4,),
    )
    placements = (
        SimpleNamespace(
            mask=np.zeros_like(mother.world.static_occupancy, dtype=np.bool_),
            occluder={"occluder_id": "occluder-unseen"},
        ),
        SimpleNamespace(
            mask=np.zeros_like(mother.world.static_occupancy, dtype=np.bool_),
            occluder={"occluder_id": "occluder-seen"},
        ),
    )
    monkeypatch.setattr(
        joint_module,
        "_iter_joint_temporal_candidates",
        lambda **_kwargs: (temporal,),
    )
    monkeypatch.setattr(
        joint_module,
        "_iter_certified_joint_placements",
        lambda **_kwargs: placements,
    )
    rebound_calls = 0

    def rebound_mother(*, placement, **_kwargs):
        nonlocal rebound_calls
        rebound_calls += 1
        history = (
            np.zeros(8, dtype=np.bool_)
            if placement.occluder["occluder_id"] == "occluder-unseen"
            else mother.target_visibility_history.copy()
        )
        return replace(
            mother,
            target_visibility_history=history,
            world=replace(
                mother.world,
                world_id=f"world-joint-{rebound_calls}",
            ),
        )

    monkeypatch.setattr(
        joint_module,
        "_rebind_audit_mother",
        rebound_mother,
    )
    complete_group = SimpleNamespace(
        is_complete=True,
        eligible_for_strict_evaluation=True,
        coverage_mask=(True,) * 6,
    )
    generated_mothers = []

    def generate_group(**kwargs):
        generated_mothers.append(kwargs["mother_event"])
        return complete_group

    monkeypatch.setattr(
        joint_module,
        "generate_paired_variants",
        generate_group,
    )

    outcome = joint_module.search_joint_audit_group(
        mother_event=mother,
        source_snippet=snippet,
        base_state=base,
        trajectory=trajectory,
        oracle_context=oracle,
        base_config=config,
        generator_config=generator_config,
        paired_config=paired_config,
        joint_config=joint_config,
        pair_seed=20260723,
    )

    assert outcome is not None
    assert rebound_calls == 2
    assert outcome.mother_event.generated_event_id == mother.generated_event_id
    assert outcome.mother_event.world.world_id == "world-joint-2"
    assert outcome.group is complete_group
    assert len(generated_mothers) == 2
    assert "audit_joint_search_summary" not in (
        generated_mothers[0].world.metadata
    )
    assert generated_mothers[1] is outcome.mother_event
    assert generated_mothers[1].world.metadata[
        "audit_joint_search_summary"
    ] == outcome.summary
    assert outcome.summary["history_regime_rejections"] == {
        "unseen_in_history_window": 1
    }
    metadata = outcome.mother_event.world.metadata
    assert metadata["audit_only"] is True
    assert metadata["audit_joint_occluder_algorithm_version"] == (
        joint_config.algorithm_version
    )
    assert metadata["audit_joint_occluder_config_digest"] == joint_config.digest
    assert metadata["audit_source_world_id"] == mother.world.world_id
