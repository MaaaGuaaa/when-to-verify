"""Configured SOP-06 paired counterfactual generation tests."""

from __future__ import annotations

from dataclasses import replace

from pathlib import Path

import numpy as np
import pytest

from src.contracts import BaseState, OracleContext, build_grid_spec, validate_oracle_world
from src.datasets.snippet_library import MotionSnippet, SnippetLibrary
from src.generation.dynamic_object_transplant import footprint_from_spec
from src.generation.event_sampler import generate_events, normalize_generator_config
from src.generation.paired_variants import (
    PairGenerationError,
    PairedVariantConfigError,
    assemble_paired_event_group,
    generate_paired_variants,
    load_paired_variant_config,
    normalize_paired_variant_config,
    summarize_paired_groups,
)
from src.geometry import (
    RectangleFootprint,
    inflate_footprint,
    signed_clearance,
    trajectory_signed_clearances,
)
from src.planning.query_maps import build_local_trajectory
from src.planning.trajectory_sampler import sample_candidate_rollouts
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _snippet() -> MotionSnippet:
    times = np.linspace(0.0, 3.0, 16, dtype=np.float32)
    positions = np.column_stack((times, 0.08 * times**2)).astype(np.float32)
    velocities = np.column_stack((np.ones_like(times), 0.16 * times)).astype(
        np.float32
    )
    headings = np.arctan2(velocities[:, 1], velocities[:, 0]).astype(np.float32)
    return MotionSnippet(
        snippet_id="train-human-snippet-paired",
        split="train",
        source_recording_id="source-recording",
        source_object_id="source-recording::paired-human",
        object_type="human",
        footprint={"kind": "circle", "radius_m": 0.30},
        start_timestamp=2.0,
        positions=positions,
        velocities=velocities,
        headings=headings,
        duration_s=3.0,
        mean_speed_mps=1.02,
        max_acceleration_mps2=0.16,
        mean_abs_curvature_per_m=0.10,
        provenance={
            "source_body_name": "paired-human",
            "raw_role": "Visitors-Alone",
            "track_provenance": {
                "geometry_source": "marker_extent_p95",
                "orientation_source": "qtm_rotation",
            },
        },
    )


def _mother_inputs():
    config = load_config()
    grid = build_grid_spec(config)
    context_id = "context-recording::LO1"
    context_history = np.tile(
        np.asarray([-2.0, 2.0, np.pi / 4.0], dtype=np.float32),
        (grid.history_steps, 1),
    )
    context_future = np.tile(
        np.asarray([-2.0, 2.0, np.pi / 4.0], dtype=np.float32),
        (grid.future_steps, 1),
    )
    context_spec = {
        "object_type": "carried_object",
        "footprint": {"kind": "rectangle", "length_m": 0.8, "width_m": 0.2},
    }
    base_state = BaseState(
        state_id="train-base-paired-fixture",
        split="train",
        recording_id="context-recording",
        dynamic_object_ids=(context_id,),
        timestamp=10.0,
        robot_history=np.zeros((grid.history_steps, 3), dtype=np.float32),
        robot_state=np.asarray([0.8, 0.0], dtype=np.float32),
        visible_dynamic_object_history={context_id: context_history.copy()},
        visible_dynamic_object_specs={context_id: context_spec},
        static_map_local=np.zeros((grid.height, grid.width), dtype=np.float32),
        metadata={"fixture": "sop06"},
    )
    oracle_context = OracleContext(
        base_state_id=base_state.state_id,
        dynamic_object_history={context_id: context_history},
        dynamic_object_future={context_id: context_future},
        dynamic_object_specs={context_id: context_spec},
        metadata={"future_dt_s": 0.2},
    )
    trajectory = build_local_trajectory(
        sample_candidate_rollouts(config)[17],
        config,
        braking_deceleration_mps2=1.0,
    )
    snippet = _snippet()
    libraries = {
        "human": SnippetLibrary(
            object_type="human",
            snippets=(snippet,),
            summary={"split": "train", "accepted_count": 1},
        )
    }
    generator_config = normalize_generator_config(
        {
            "schema_version": "2.0.0",
            "target_type_policy": {
                "whitelist": ["human"],
                "weights": {
                    "human": 1.0,
                    "carried_object": 0.0,
                    "unknown_dynamic": 0.0,
                },
            },
            "event_type_weights": {
                "environment": 0.0,
                "structural": 1.0,
                "mixed": 0.0,
            },
            "conflict_time_range_s": [1.0, 1.0],
            "max_local_curvature_per_m": 1.0,
            "crossing_angle_max_deg": 35.0,
            "time_scale_range": [1.0, 1.0],
            "max_resample_attempts": 32,
            "min_contiguous_visible_frames": 2,
            "occluders": {
                "types": ["pillar"],
                "normal_offset_range_m": [0.8, 1.2],
                "wall": {
                    "length_range_m": [1.0, 1.4],
                    "width_range_m": [0.2, 0.3],
                },
                "shelf": {
                    "length_range_m": [1.0, 1.4],
                    "width_range_m": [0.4, 0.5],
                },
                "pillar": {
                    "length_range_m": [0.4, 0.5],
                    "width_range_m": [0.4, 0.5],
                },
            },
            "structural_fov": {
                "forward_fov_deg": [160.0],
                "range_m": [6.0],
                "optional_blind_sectors": [
                    {"center_deg": -90.0, "width_deg": 110.0},
                    {"center_deg": 90.0, "width_deg": 110.0},
                ],
            },
        }
    )
    report = generate_events(
        base_state=base_state,
        oracle_context=oracle_context,
        trajectory=trajectory,
        snippet_libraries=libraries,
        base_config=config,
        generator_config=generator_config,
        seed=23,
        event_count=1,
    )
    assert len(report.events) == 1
    return config, grid, oracle_context, trajectory, snippet, report.events[0]


@pytest.fixture(scope="module")
def complete_pair():
    config, grid, oracle, trajectory, snippet, mother = _mother_inputs()
    paired_config = load_paired_variant_config(
        ROOT / "configs" / "paired_variants.yaml"
    )
    group = generate_paired_variants(
        mother_event=mother,
        source_snippet=snippet,
        trajectory=trajectory,
        oracle_context=oracle,
        base_config=config,
        paired_config=paired_config,
        seed=20260716,
    )
    return config, grid, oracle, trajectory, mother, paired_config, group


def test_paired_config_freezes_thresholds_coverage_and_digest() -> None:
    first = load_paired_variant_config(ROOT / "configs" / "paired_variants.yaml")
    repeated = load_paired_variant_config(ROOT / "configs" / "paired_variants.yaml")

    assert first == repeated
    assert first.near_miss_clearance_range_m == (0.05, 0.35)
    assert first.temporal_offset_candidates_s == (
        0.8,
        -0.8,
        1.0,
        -1.0,
        1.2,
        -1.2,
        1.5,
        -1.5,
    )
    assert first.spatial_safe_clearance_range_m == (0.5, 1.0)
    assert first.irrelevant_min_clearance_m == 1.5
    assert first.lateral_offset_step_m == 0.025
    assert first.lateral_offset_max_m == 3.0
    assert first.minimum_required_variants == (
        "collision",
        "empty_blind_spot",
    )
    assert first.minimum_contrast_variants == (
        "near_miss",
        "temporal_safe",
        "spatial_safe",
    )
    assert first.minimum_contrast_count == 1
    assert first.complete_evaluation_requires_all_variants is True
    assert len(first.digest) == 32


def test_paired_config_rejects_unknown_keys_and_weak_irrelevant_threshold() -> None:
    valid = {
        "schema_version": "2.0.0",
        "near_miss_clearance_range_m": [0.05, 0.35],
        "temporal_offset_candidates_s": [
            0.8,
            -0.8,
            1.0,
            -1.0,
            1.2,
            -1.2,
            1.5,
            -1.5,
        ],
        "spatial_safe_clearance_range_m": [0.5, 1.0],
        "irrelevant_min_clearance_m": 1.5,
        "lateral_offset_step_m": 0.025,
        "lateral_offset_max_m": 3.0,
        "minimum_required_variants": ["collision", "empty_blind_spot"],
        "minimum_contrast_variants": [
            "near_miss",
            "temporal_safe",
            "spatial_safe",
        ],
        "minimum_contrast_count": 1,
        "complete_evaluation_requires_all_variants": True,
    }
    with pytest.raises(PairedVariantConfigError, match="unknown"):
        normalize_paired_variant_config({**valid, "unknown": 1})
    with pytest.raises(PairedVariantConfigError, match="greater than spatial-safe"):
        normalize_paired_variant_config(
            {**valid, "irrelevant_min_clearance_m": 1.0}
        )


def test_six_pack_preserves_skeleton_and_meets_exact_risk_geometry(
    complete_pair,
) -> None:
    config, grid, oracle, trajectory, mother, paired_config, group = complete_pair
    assert group.is_complete is True
    assert group.eligible_for_strict_evaluation is True
    assert group.coverage_mask == (True, True, True, True, True, True)
    assert group.missing_variant_reasons == {}
    assert tuple(variant.variant_kind for variant in group.variants) == (
        "collision",
        "near_miss",
        "temporal_safe",
        "spatial_safe",
        "irrelevant_hidden",
        "empty_blind_spot",
    )

    robot_footprint = inflate_footprint(
        RectangleFootprint(
            config["robot"]["length_m"], config["robot"]["width_m"]
        ),
        config["robot"]["inflation_m"],
    )
    target_footprint = footprint_from_spec(mother.target.footprint_spec)
    variants = group.by_kind
    assert variants["collision"].min_clearance_m <= 0.0
    assert paired_config.near_miss_clearance_range_m[0] <= (
        variants["near_miss"].min_clearance_m
    ) <= paired_config.near_miss_clearance_range_m[1]
    assert paired_config.spatial_safe_clearance_range_m[0] <= (
        variants["spatial_safe"].min_clearance_m
    ) <= paired_config.spatial_safe_clearance_range_m[1]
    assert variants["irrelevant_hidden"].min_clearance_m >= (
        paired_config.irrelevant_min_clearance_m
    )
    assert variants["temporal_safe"].min_clearance_m > 0.0
    assert variants["temporal_safe"].temporal_offset_s in (
        paired_config.temporal_offset_candidates_s
    )
    spatial_minimum_index = int(
        round(
            variants["spatial_safe"].time_to_min_clearance_s
            / config["bev"]["future_dt_s"]
        )
        - 1
    )
    assert abs(spatial_minimum_index - mother.conflict_index) <= 1

    temporal_target = variants["temporal_safe"].target
    assert temporal_target is not None
    asynchronous_clearances = [
        signed_clearance(robot_footprint, robot_pose, target_footprint, target_pose)
        for robot_pose in trajectory.poses
        for target_pose in temporal_target.future_poses
    ]
    assert min(asynchronous_clearances) <= 0.0
    np.testing.assert_array_equal(
        trajectory_signed_clearances(
            robot_footprint,
            trajectory.poses,
            target_footprint,
            temporal_target.future_poses,
        ),
        variants["temporal_safe"].clearance_sequence_m,
    )

    non_target_ids = set(oracle.dynamic_object_future)
    for variant in group.variants:
        validate_oracle_world(variant.world, grid)
        assert variant.world.base_state_id == mother.world.base_state_id
        assert variant.world.occluders == mother.world.occluders
        assert variant.world.blind_spot_config == mother.world.blind_spot_config
        np.testing.assert_array_equal(
            variant.world.static_occupancy, mother.world.static_occupancy
        )
        assert variant.world.metadata["pair_group_id"] == group.pair_group_id
        assert variant.world.metadata["paired_variant_kind"] == variant.variant_kind
        for object_id in non_target_ids:
            np.testing.assert_array_equal(
                variant.world.dynamic_object_trajectories[object_id],
                oracle.dynamic_object_future[object_id],
            )
            assert (
                variant.world.dynamic_object_specs[object_id]
                == oracle.dynamic_object_specs[object_id]
            )

    target_id = mother.target.target_dynamic_object_id
    for kind in (
        "collision",
        "near_miss",
        "temporal_safe",
        "spatial_safe",
        "irrelevant_hidden",
    ):
        variant = variants[kind]
        assert variant.target is not None
        assert variant.target.target_dynamic_object_id == target_id
        assert variant.target.object_type == mother.target.object_type
        assert variant.target.footprint_spec == mother.target.footprint_spec
        assert variant.target.source_object_id == mother.target.source_object_id
        assert variant.world.metadata["target_provenance"] == (
            variant.target.provenance
        )
        assert not bool(variant.visibility_sequence[0])
        assert bool(variant.visibility_sequence[-1])
    empty = variants["empty_blind_spot"]
    assert empty.target is None
    assert empty.visibility_sequence is None
    assert target_id not in empty.world.dynamic_object_trajectories
    assert set(empty.world.dynamic_object_trajectories) == non_target_ids


def test_pair_generation_is_elementwise_deterministic(complete_pair) -> None:
    config, _, oracle, trajectory, mother, paired_config, first = complete_pair
    second = generate_paired_variants(
        mother_event=mother,
        source_snippet=_snippet(),
        trajectory=trajectory,
        oracle_context=oracle,
        base_config=config,
        paired_config=paired_config,
        seed=20260716,
    )

    assert first.pair_group_id == second.pair_group_id
    assert first.coverage_mask == second.coverage_mask
    assert first.missing_variant_reasons == second.missing_variant_reasons
    for left, right in zip(first.variants, second.variants):
        assert left.variant_kind == right.variant_kind
        assert left.world.world_id == right.world.world_id
        assert left.world.metadata == right.world.metadata
        if left.target is not None:
            np.testing.assert_array_equal(
                left.target.future_poses, right.target.future_poses
            )
            np.testing.assert_array_equal(
                left.visibility_sequence, right.visibility_sequence
            )


def test_partial_group_policy_records_fixed_coverage_and_excludes_strict_eval(
    complete_pair,
) -> None:
    *_, paired_config, complete = complete_pair
    variants = {
        variant.variant_kind: variant
        for variant in complete.variants
        if variant.variant_kind != "irrelevant_hidden"
    }
    partial = assemble_paired_event_group(
        pair_group_id=complete.pair_group_id,
        variants=variants,
        missing_variant_reasons={"irrelevant_hidden": "clearance_unavailable"},
        paired_config=paired_config,
    )

    assert partial.coverage_mask == (True, True, True, True, False, True)
    assert partial.is_complete is False
    assert partial.eligible_for_strict_evaluation is False
    assert partial.missing_variant_reasons == {
        "irrelevant_hidden": "clearance_unavailable"
    }
    summary = summarize_paired_groups((complete, partial))
    assert summary["group_count"] == 2
    assert summary["complete_group_count"] == 1
    assert summary["partial_group_count"] == 1
    assert summary["coverage_counts"]["irrelevant_hidden"] == 1

    minimum_only = {
        kind: complete.by_kind[kind]
        for kind in ("collision", "empty_blind_spot")
    }
    with pytest.raises(PairGenerationError, match="minimum contrast"):
        assemble_paired_event_group(
            pair_group_id=complete.pair_group_id,
            variants=minimum_only,
            missing_variant_reasons={
                kind: "not_generated"
                for kind in (
                    "near_miss",
                    "temporal_safe",
                    "spatial_safe",
                    "irrelevant_hidden",
                )
            },
            paired_config=paired_config,
        )


def test_pair_rejects_context_actor_that_is_itself_critical(complete_pair) -> None:
    config, _, oracle, trajectory, mother, paired_config, _ = complete_pair
    context_id = next(iter(oracle.dynamic_object_future))
    colliding_context = replace(
        oracle,
        dynamic_object_history={
            context_id: np.zeros_like(oracle.dynamic_object_history[context_id])
        },
        dynamic_object_future={context_id: trajectory.poses.copy()},
    )
    colliding_world_trajectories = {
        **mother.world.dynamic_object_trajectories,
        context_id: trajectory.poses.copy(),
    }
    colliding_mother = replace(
        mother,
        world=replace(
            mother.world,
            dynamic_object_trajectories=colliding_world_trajectories,
        ),
    )
    with pytest.raises(PairGenerationError, match="multi_object_context"):
        generate_paired_variants(
            mother_event=colliding_mother,
            source_snippet=_snippet(),
            trajectory=trajectory,
            oracle_context=colliding_context,
            base_config=config,
            paired_config=paired_config,
            seed=20260716,
        )
