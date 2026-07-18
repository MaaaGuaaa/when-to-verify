"""Configured SOP-06 paired counterfactual generation tests."""

from __future__ import annotations

from dataclasses import replace

from pathlib import Path

import numpy as np
import pytest

import src.generation.occluder_sampler as occluder_sampler_module
import src.generation.paired_variants as paired_variants_module
from src.contracts import (
    SCHEMA_VERSION,
    BaseState,
    OracleContext,
    OracleWorld,
    build_grid_spec,
    validate_oracle_world,
)
from src.datasets.snippet_library import MotionSnippet, SnippetLibrary
from src.generation.dynamic_object_transplant import (
    TransplantError,
    footprint_from_spec,
)
from src.generation.event_target_motion_shard import (
    build_event_target_motion_world_metadata,
    compute_footprint_spec_digest,
    compute_motion_array_digest,
    create_event_target_motion_record,
    validate_event_target_motion_world_join,
)
from src.generation.event_sampler import (
    generate_events,
    load_generator_config,
    normalize_generator_config,
)
from src.generation.paired_variants import (
    PairGenerationError,
    PairedVariantConfigError,
    VARIANT_ORDER,
    assemble_paired_event_group,
    generate_paired_variants,
    load_paired_variant_config,
    normalize_paired_variant_config,
    summarize_paired_groups,
)
from src.geometry import (
    RectangleFootprint,
    grid_to_world,
    inflate_footprint,
    rasterize_footprint,
    signed_clearance,
    trajectory_signed_clearances,
    wrap_angle,
)
from src.planning.query_maps import build_local_trajectory
from src.planning.trajectory_sampler import sample_candidate_rollouts
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _snippet() -> MotionSnippet:
    times = np.linspace(0.0, 4.4, 23, dtype=np.float32)
    positions = np.column_stack((1.4 * times, 0.08 * times**2)).astype(
        np.float32
    )
    velocities = np.column_stack(
        (np.full_like(times, 1.4), 0.16 * times)
    ).astype(np.float32)
    headings = np.arctan2(velocities[:, 1], velocities[:, 0]).astype(np.float32)
    return MotionSnippet(
        snippet_id="train-human-snippet-paired",
        split="train",
        source_recording_id="source-recording",
        source_session_id="source-session",
        source_object_id="source-recording::paired-human",
        object_type="human",
        footprint={"kind": "circle", "radius_m": 0.30},
        start_timestamp=2.0,
        positions=positions,
        velocities=velocities,
        headings=headings,
        duration_s=4.4,
        mean_speed_mps=float(np.linalg.norm(velocities, axis=1).mean()),
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


def _exact_safe_curve_snippet() -> MotionSnippet:
    """23-sample real curve used by the v5 reachability mother fixture."""

    snippet = _snippet()
    relative_times = (np.arange(23, dtype=np.float64) - 7.0) * 0.2
    conflict_time_s = 2.2
    lateral_acceleration = 0.8
    initial_lateral_velocity = lateral_acceleration * conflict_time_s
    lateral = np.where(
        relative_times <= 0.0,
        initial_lateral_velocity * relative_times,
        np.where(
            relative_times <= conflict_time_s,
            lateral_acceleration
            * relative_times
            * (conflict_time_s - relative_times),
            -initial_lateral_velocity
            * (relative_times - conflict_time_s)
            + 1.1 * (relative_times - conflict_time_s) ** 2,
        ),
    )
    lateral_velocity = np.where(
        relative_times <= 0.0,
        initial_lateral_velocity,
        np.where(
            relative_times <= conflict_time_s,
            lateral_acceleration
            * (conflict_time_s - 2.0 * relative_times),
            -initial_lateral_velocity
            + 2.2 * (relative_times - conflict_time_s),
        ),
    )
    positions = np.column_stack((0.8 * relative_times, lateral))
    positions -= positions[0]
    velocities = np.column_stack(
        (np.full(23, 0.8, dtype=np.float64), lateral_velocity)
    )
    headings = np.arctan2(velocities[:, 1], velocities[:, 0])
    return replace(
        snippet,
        snippet_id="train-human-snippet-exact-safe-curve",
        positions=positions.astype(np.float32),
        velocities=velocities.astype(np.float32),
        headings=headings.astype(np.float32),
        mean_speed_mps=1.5,
        max_acceleration_mps2=2.2,
        mean_abs_curvature_per_m=0.5,
    )


def _paired_source_inputs():
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
        state_id="train-base-event-fixture",
        split="train",
        recording_id="context-recording",
        dynamic_object_ids=(context_id,),
        timestamp=10.0,
        robot_history=np.zeros((grid.history_steps, 3), dtype=np.float32),
        robot_state=np.asarray([0.4, 0.0], dtype=np.float32),
        visible_dynamic_object_history={context_id: context_history.copy()},
        visible_dynamic_object_specs={context_id: context_spec},
        static_map_local=np.zeros((grid.height, grid.width), dtype=np.float32),
        metadata={"fixture": "sop05"},
    )
    oracle_context = OracleContext(
        base_state_id=base_state.state_id,
        dynamic_object_history={context_id: context_history},
        dynamic_object_future={context_id: context_future},
        dynamic_object_specs={context_id: context_spec},
        metadata={"future_dt_s": 0.2},
    )
    trajectory = build_local_trajectory(
        sample_candidate_rollouts(config)[7],
        config,
        braking_deceleration_mps2=1.0,
    )
    trajectory = replace(
        trajectory,
        metadata={
            **trajectory.metadata,
            "pose_time_layout_version": "future_endpoints_dt_to_horizon_v1",
            "pose_time_offsets_s": (
                (np.arange(grid.future_steps, dtype=np.float64) + 1.0)
                * float(config["bev"]["future_dt_s"])
            ).tolist(),
        },
    )
    snippet = _exact_safe_curve_snippet()
    libraries = {
        "human": SnippetLibrary(
            object_type="human",
            snippets=(snippet,),
            summary={"split": "train", "accepted_count": 1},
            split_provenance={"split": "train"},
        )
    }
    return config, grid, base_state, oracle_context, trajectory, snippet, libraries


def _mother_inputs():
    (
        config,
        grid,
        base_state,
        oracle_context,
        trajectory,
        snippet,
        libraries,
    ) = _paired_source_inputs()
    generator_config = load_generator_config(
        ROOT / "configs" / "generator_test.yaml"
    )
    generator_config["conflict_time_range_s"] = (2.2, 2.2)
    generator_config["occluders"] = {
        "types": ("pillar",),
        "wall": {
            "length_range_m": (1.0, 1.4),
            "width_range_m": (0.2, 0.3),
        },
        "shelf": {
            "length_range_m": (1.0, 1.4),
            "width_range_m": (0.4, 0.5),
        },
        "pillar": {
            "length_range_m": (0.4, 0.5),
            "width_range_m": (0.4, 0.5),
        },
    }
    generator_config["blind_reachability"] = {
        **generator_config["blind_reachability"],
        "obstacle_proposals_per_trajectory": 8,
    }
    report = generate_events(
        base_state=base_state,
        oracle_context=oracle_context,
        trajectory=trajectory,
        snippet_libraries=libraries,
        base_config=config,
        generator_config=generator_config,
        seed=20260716,
        event_count=1,
    )
    assert len(report.events) == 1
    return (
        config,
        grid,
        base_state,
        oracle_context,
        trajectory,
        snippet,
        report.events[0],
    )


def _mother_with_rebuilt_motion_record(mother, **overrides):
    source = mother.target_motion_record
    values = {
        "generated_event_id": source.generated_event_id,
        "world_id": source.world_id,
        "base_state_id": source.base_state_id,
        "trajectory_id": source.trajectory_id,
        "target_dynamic_object_id": source.target_dynamic_object_id,
        "source_snippet_id": source.source_snippet_id,
        "source_object_id": source.source_object_id,
        "object_type": source.object_type,
        "footprint_spec": source.footprint_spec,
        "footprint_spec_digest": source.footprint_spec_digest,
        "target_type_policy_digest": source.target_type_policy_digest,
        "history_poses": source.history_poses,
        "current_pose": source.current_pose,
        "future_poses": source.future_poses,
    }
    values.update(overrides)
    record = create_event_target_motion_record(**values)
    trajectories = dict(mother.world.dynamic_object_trajectories)
    specs = dict(mother.world.dynamic_object_specs)
    old_target_id = source.target_dynamic_object_id
    if record.target_dynamic_object_id != old_target_id:
        trajectories.pop(old_target_id)
        specs.pop(old_target_id)
    trajectories[record.target_dynamic_object_id] = record.future_poses.copy()
    specs[record.target_dynamic_object_id] = dict(record.footprint_spec)
    world = replace(
        mother.world,
        dynamic_object_trajectories=trajectories,
        dynamic_object_specs=specs,
        metadata={
            **mother.world.metadata,
            **build_event_target_motion_world_metadata(record),
        },
    )
    return replace(mother, world=world, target_motion_record=record)


@pytest.fixture(scope="module")
def complete_pair():
    config, grid, base, oracle, trajectory, snippet, mother = _mother_inputs()
    paired_config = load_paired_variant_config(
        ROOT / "configs" / "paired_variants.yaml"
    )
    group = generate_paired_variants(
        mother_event=mother,
        source_snippet=snippet,
        base_state=base,
        trajectory=trajectory,
        oracle_context=oracle,
        base_config=config,
        paired_config=paired_config,
        seed=20260716,
    )
    return config, grid, base, oracle, trajectory, mother, paired_config, group


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
    assert first.paired_generator_algorithm_version == (
        "independent_partial_pairs_v1"
    )
    assert first.group_contract_version == "sop06_partial_pair_group_v1"
    assert first.mother_required_variants == ("collision",)
    assert first.training_minimum_contrast_count == 0
    assert first.audit_requires_all_variants is True
    assert len(first.digest) == 32


def test_partial_pair_versions_are_public_frozen_constants() -> None:
    assert getattr(
        paired_variants_module,
        "PAIRED_GENERATOR_ALGORITHM_VERSION",
        None,
    ) == "independent_partial_pairs_v1"
    assert paired_variants_module.PAIRED_GROUP_CONTRACT_VERSION == (
        "sop06_partial_pair_group_v1"
    )


def test_paired_config_rejects_unknown_keys_and_weak_irrelevant_threshold() -> None:
    valid = {
        "schema_version": SCHEMA_VERSION,
        "paired_generator_algorithm_version": (
            "independent_partial_pairs_v1"
        ),
        "group_contract_version": "sop06_partial_pair_group_v1",
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
        "mother_required_variants": ["collision"],
        "training_minimum_contrast_count": 0,
        "audit_requires_all_variants": True,
    }
    with pytest.raises(PairedVariantConfigError, match="unknown"):
        normalize_paired_variant_config({**valid, "unknown": 1})
    with pytest.raises(PairedVariantConfigError, match="greater than spatial-safe"):
        normalize_paired_variant_config(
            {**valid, "irrelevant_min_clearance_m": 1.0}
        )


def test_singleton_collision_group_is_training_eligible_partial() -> None:
    paired_config = load_paired_variant_config(
        ROOT / "configs" / "paired_variants.yaml"
    )
    pair_group_id = "pair-singleton-collision"
    world = OracleWorld(
        world_id="world-singleton-collision",
        base_state_id="base-singleton-collision",
        static_occupancy=np.zeros((1, 1), dtype=np.float32),
        dynamic_object_trajectories={},
        dynamic_object_specs={},
        occluders=(),
        blind_spot_config={
            "kind": "environment",
            "structural": None,
            "occluder_ids": [],
        },
        random_seed=7,
        metadata={"pair_group_id": pair_group_id},
    )
    collision = paired_variants_module.PairedVariant(
        variant_kind="collision",
        world=world,
        target=None,
        target_visibility_history=None,
        visibility_sequence=None,
        clearance_sequence_m=np.asarray([-0.1], dtype=np.float32),
        min_clearance_m=-0.1,
        time_to_min_clearance_s=0.2,
    )
    missing = {
        kind: "not_generated"
        for kind in paired_variants_module.VARIANT_ORDER
        if kind != "collision"
    }

    group = assemble_paired_event_group(
        pair_group_id=pair_group_id,
        variants={"collision": collision},
        missing_variant_reasons=missing,
        paired_config=paired_config,
    )

    assert group.coverage_mask == (True, False, False, False, False, False)
    assert group.is_complete is False
    assert group.eligible_for_strict_evaluation is False
    assert group.by_kind == {"collision": group.variants[0]}
    assert group.variants[0].world.metadata[
        "paired_generator_algorithm_version"
    ] == "independent_partial_pairs_v1"
    assert group.variants[0].world.metadata["pair_group_contract_version"] == (
        "sop06_partial_pair_group_v1"
    )


def test_formal_v5_config_rejects_retired_joint_sixpack_producer() -> None:
    (
        config,
        _,
        base_state,
        oracle,
        trajectory,
        _,
        libraries,
    ) = _paired_source_inputs()
    paired_config = load_paired_variant_config(
        ROOT / "configs" / "paired_variants.yaml"
    )
    generator_config = load_generator_config(
        ROOT / "configs" / "generator_train.yaml"
    )

    with pytest.raises(
        PairGenerationError, match="joint_environment_pair_v2 is retired"
    ) as exc_info:
        paired_variants_module.generate_joint_environment_pair(
            base_state=base_state,
            oracle_context=oracle,
            trajectory=trajectory,
            snippet_libraries=libraries,
            base_config=config,
            generator_config=generator_config,
            paired_config=paired_config,
            seed=20260716,
        )

    assert exc_info.value.reason == "joint_environment_pair_v2_retired"


def test_partial_generator_rejects_retired_joint_mother(complete_pair) -> None:
    (
        config,
        _,
        base_state,
        oracle,
        trajectory,
        mother,
        paired_config,
        _,
    ) = complete_pair
    retired_world = replace(
        mother.world,
        metadata={
            **mother.world.metadata,
            "joint_pair_generator_algorithm_version": (
                "joint_environment_pair_v2"
            ),
        },
    )

    with pytest.raises(
        PairGenerationError, match="retired joint_environment_pair_v2"
    ) as exc_info:
        generate_paired_variants(
            mother_event=replace(mother, world=retired_world),
            source_snippet=_exact_safe_curve_snippet(),
            base_state=base_state,
            trajectory=trajectory,
            oracle_context=oracle,
            base_config=config,
            paired_config=paired_config,
            seed=20260716,
        )

    assert exc_info.value.reason == "retired_joint_mother"


def test_optional_variant_failures_preserve_collision_and_empty(
    complete_pair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        config,
        _,
        base_state,
        oracle,
        trajectory,
        mother,
        paired_config,
        _,
    ) = complete_pair

    def reject_geometric(*, variant_kind: str, **_: object):
        return None, f"{variant_kind}_unavailable"

    monkeypatch.setattr(
        paired_variants_module,
        "_geometric_variant",
        reject_geometric,
    )
    monkeypatch.setattr(
        paired_variants_module,
        "_temporal_variant",
        lambda **_: (
            None,
            "temporal_transplant:snippet_stationary_at_conflict",
        ),
    )

    group = generate_paired_variants(
        mother_event=mother,
        source_snippet=_exact_safe_curve_snippet(),
        base_state=base_state,
        trajectory=trajectory,
        oracle_context=oracle,
        base_config=config,
        paired_config=paired_config,
        seed=20260716,
    )

    assert tuple(variant.variant_kind for variant in group.variants) == (
        "collision",
        "empty_blind_spot",
    )
    assert group.coverage_mask == (
        True,
        False,
        False,
        False,
        False,
        True,
    )
    assert group.is_complete is False
    assert group.eligible_for_strict_evaluation is False
    assert group.missing_variant_reasons == {
        "near_miss": "near_miss_unavailable",
        "temporal_safe": (
            "temporal_transplant:snippet_stationary_at_conflict"
        ),
        "spatial_safe": "spatial_safe_unavailable",
        "irrelevant_hidden": "irrelevant_hidden_unavailable",
    }


@pytest.mark.parametrize(
    "failed_kind",
    ("near_miss", "spatial_safe", "irrelevant_hidden"),
)
def test_each_geometric_failure_preserves_other_successful_variants(
    complete_pair,
    monkeypatch: pytest.MonkeyPatch,
    failed_kind: str,
) -> None:
    (
        config,
        _,
        base_state,
        oracle,
        trajectory,
        mother,
        paired_config,
        baseline,
    ) = complete_pair
    real_geometric_variant = paired_variants_module._geometric_variant

    def reject_one(*, variant_kind: str, **kwargs):
        if variant_kind == failed_kind:
            return None, f"{failed_kind}_forced_unavailable"
        return real_geometric_variant(
            variant_kind=variant_kind,
            **kwargs,
        )

    monkeypatch.setattr(
        paired_variants_module,
        "_geometric_variant",
        reject_one,
    )
    group = generate_paired_variants(
        mother_event=mother,
        source_snippet=_exact_safe_curve_snippet(),
        base_state=base_state,
        trajectory=trajectory,
        oracle_context=oracle,
        base_config=config,
        paired_config=paired_config,
        seed=20260716,
    )

    assert group.pair_group_id == baseline.pair_group_id
    assert "collision" in group.by_kind
    assert "empty_blind_spot" in group.by_kind
    assert failed_kind not in group.by_kind
    assert group.missing_variant_reasons[failed_kind] == (
        f"{failed_kind}_forced_unavailable"
    )
    for retained_kind in {
        "near_miss",
        "spatial_safe",
        "irrelevant_hidden",
    } - {failed_kind}:
        assert retained_kind in group.by_kind


def test_temporal_variant_audits_transplant_errors_and_exhausts_schedule(
    complete_pair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, grid, base, oracle, trajectory, mother, paired_config, group = (
        complete_pair
    )
    stationary = replace(
        _snippet(),
        positions=np.zeros((23, 2), dtype=np.float32),
        velocities=np.zeros((23, 2), dtype=np.float32),
        headings=np.zeros(23, dtype=np.float32),
        mean_speed_mps=0.0,
        max_acceleration_mps2=0.0,
        mean_abs_curvature_per_m=0.0,
    )
    environment = paired_variants_module._pair_environment(
        mother_event=mother,
        trajectory=trajectory,
        base_state=base,
        oracle_context=oracle,
        base_config=config,
        critical_clearance_threshold_m=(
            paired_config.near_miss_clearance_range_m[1]
        ),
    )
    attempted_conflict_times: list[float] = []
    real_transplant = paired_variants_module.transplant_snippet

    def recording_transplant(*args, **kwargs):
        attempted_conflict_times.append(float(kwargs["conflict_time_s"]))
        return real_transplant(*args, **kwargs)

    monkeypatch.setattr(
        paired_variants_module,
        "transplant_snippet",
        recording_transplant,
    )

    variant, reason = paired_variants_module._temporal_variant(
        mother_event=mother,
        source_snippet=stationary,
        trajectory=trajectory,
        base_state=base,
        oracle_context=oracle,
        base_config=config,
        paired_config=paired_config,
        pair_group_id=group.pair_group_id,
        seed=20260716,
        environment=environment,
    )

    expected_conflict_times = [
        mother.conflict_time_s + offset
        for offset in paired_config.temporal_offset_candidates_s
        if 0.0 < mother.conflict_time_s + offset <= (
            grid.future_steps * config["bev"]["future_dt_s"]
        )
    ]
    assert variant is None
    assert reason == "temporal_transplant:snippet_stationary_at_conflict"
    assert attempted_conflict_times == pytest.approx(expected_conflict_times)


def test_temporal_transplant_failure_becomes_group_missing_reason(
    complete_pair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        config,
        _,
        base_state,
        oracle,
        trajectory,
        mother,
        paired_config,
        baseline,
    ) = complete_pair
    attempted_conflict_times: list[float] = []

    def reject_retiming(*args, **kwargs):
        attempted_conflict_times.append(float(kwargs["conflict_time_s"]))
        raise TransplantError("snippet_stationary_at_conflict")

    monkeypatch.setattr(
        paired_variants_module,
        "transplant_snippet",
        reject_retiming,
    )

    group = generate_paired_variants(
        mother_event=mother,
        source_snippet=_exact_safe_curve_snippet(),
        base_state=base_state,
        trajectory=trajectory,
        oracle_context=oracle,
        base_config=config,
        paired_config=paired_config,
        seed=20260716,
    )

    assert group.pair_group_id == baseline.pair_group_id
    assert group.by_kind["collision"].world.world_id == (
        baseline.by_kind["collision"].world.world_id
    )
    assert "temporal_safe" not in group.by_kind
    assert group.missing_variant_reasons["temporal_safe"] == (
        "temporal_transplant:snippet_stationary_at_conflict"
    )
    assert len(attempted_conflict_times) > 1
    assert group.is_complete is False


def test_optional_failure_does_not_resample_or_change_mother_identity(
    complete_pair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        config,
        _,
        base_state,
        oracle,
        trajectory,
        mother,
        paired_config,
        baseline,
    ) = complete_pair
    real_geometric_variant = paired_variants_module._geometric_variant

    def forbid_mother_resample(**_: object):
        raise AssertionError("SOP06 must not resample an accepted SOP05 mother")

    def reject_near_miss(*, variant_kind: str, **kwargs):
        if variant_kind == "near_miss":
            return None, "near_miss_unavailable"
        return real_geometric_variant(
            variant_kind=variant_kind,
            **kwargs,
        )

    monkeypatch.setattr(
        paired_variants_module,
        "generate_events",
        forbid_mother_resample,
    )
    monkeypatch.setattr(
        paired_variants_module,
        "_geometric_variant",
        reject_near_miss,
    )

    group = generate_paired_variants(
        mother_event=mother,
        source_snippet=_exact_safe_curve_snippet(),
        base_state=base_state,
        trajectory=trajectory,
        oracle_context=oracle,
        base_config=config,
        paired_config=paired_config,
        seed=20260716,
    )

    assert group.pair_group_id == baseline.pair_group_id
    assert group.by_kind["collision"].world.world_id == (
        baseline.by_kind["collision"].world.world_id
    )
    assert group.by_kind["collision"].target is mother.target
    assert "near_miss" not in group.by_kind
    assert group.missing_variant_reasons["near_miss"] == (
        "near_miss_unavailable"
    )
    assert "spatial_safe" in group.by_kind
    assert "irrelevant_hidden" in group.by_kind
    assert "empty_blind_spot" in group.by_kind


def test_partial_pair_preserves_skeleton_and_meets_exact_risk_geometry(
    complete_pair,
) -> None:
    (
        config,
        grid,
        _,
        oracle,
        trajectory,
        mother,
        paired_config,
        group,
    ) = complete_pair
    assert group.is_complete is False
    assert group.eligible_for_strict_evaluation is False
    assert group.coverage_mask == (True, True, False, True, True, True)
    assert group.missing_variant_reasons == {
        "temporal_safe": "temporal_variant_still_collides"
    }
    assert tuple(variant.variant_kind for variant in group.variants) == (
        "collision",
        "near_miss",
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
    spatial_minimum_index = int(
        round(
            variants["spatial_safe"].time_to_min_clearance_s
            / config["bev"]["future_dt_s"]
        )
        - 1
    )
    assert abs(spatial_minimum_index - mother.conflict_index) <= 1

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


def test_v5_environment_partial_group_renders_without_joint_pair_version(
    complete_pair,
) -> None:
    from src.generation.sop06_pipeline import render_sop06_partial_pair_group

    config, _, base, oracle, _, mother, paired_config, group = complete_pair
    assert mother.event_kind == "environment"
    assert mother.world.metadata["generator_algorithm_version"] == (
        "blind_reachability_first_v1"
    )
    assert "joint_pair_generator_algorithm_version" not in (
        mother.world.metadata
    )
    assert group.is_complete is False

    rendered = render_sop06_partial_pair_group(
        group=group,
        mother_record=mother.target_motion_record,
        mother_world=mother.world,
        base_state=base,
        oracle_context=oracle,
        config=config,
        expected_paired_config_digest=paired_config.digest,
    )

    assert rendered.variant_kinds == tuple(
        variant.variant_kind for variant in group.variants
    )
    assert rendered.audit_certified is False
    assert all(
        observation.bev_history.shape[0] == config["bev"]["history_steps"]
        for observation in rendered.observations
    )


def test_spatial_variants_apply_one_se2_to_complete_23_point_motion(
    complete_pair,
) -> None:
    _, _, _, _, _, mother, _, group = complete_pair
    mother_motion = np.vstack(
        (mother.target.history_poses, mother.target.future_poses)
    ).astype(np.float64)
    mother_distances = np.linalg.norm(
        mother_motion[:, None, :2] - mother_motion[None, :, :2], axis=2
    )

    for kind in ("near_miss", "spatial_safe", "irrelevant_hidden"):
        variant = group.by_kind[kind]
        assert variant.target is not None
        target = variant.target
        assert target.history_poses.shape == (8, 3)
        assert target.future_poses.shape == (15, 3)
        assert target.history_poses.dtype == np.float32
        assert target.future_poses.dtype == np.float32
        np.testing.assert_array_equal(target.history_poses[7], target.current_pose)

        transformed = np.vstack(
            (target.history_poses, target.future_poses)
        ).astype(np.float64)
        transformed_distances = np.linalg.norm(
            transformed[:, None, :2] - transformed[None, :, :2], axis=2
        )
        np.testing.assert_allclose(
            transformed_distances,
            mother_distances,
            rtol=0.0,
            atol=2e-6,
        )
        yaw_delta = wrap_angle(transformed[:, 2] - mother_motion[:, 2])
        np.testing.assert_allclose(
            yaw_delta,
            np.full_like(yaw_delta, yaw_delta[0]),
            rtol=0.0,
            atol=2e-6,
        )


def test_variant_world_identity_binds_history_and_future_motion(
    complete_pair,
) -> None:
    *_, group = complete_pair
    world_ids = set()
    for variant in group.variants:
        assert variant.world.world_id not in world_ids
        world_ids.add(variant.world.world_id)
        if variant.target is None:
            assert variant.world.metadata["paired_target_motion_digest"] == (
                "target-empty"
            )
            continue
        target = variant.target
        assert variant.world.metadata[
            "paired_target_history_array_digest"
        ] == compute_motion_array_digest(
            target.history_poses, field_name="target_history_poses"
        )
        assert variant.world.metadata[
            "paired_target_future_array_digest"
        ] == compute_motion_array_digest(
            target.future_poses, field_name="target_future_poses"
        )
        expected = paired_variants_module._paired_target_motion_digest(target)
        assert variant.world.metadata["paired_target_motion_digest"] == expected

        changed_history = target.history_poses.copy()
        changed_history[0, 0] += np.float32(0.01)
        changed = replace(target, history_poses=changed_history)
        assert paired_variants_module._paired_target_motion_digest(changed) != (
            expected
        )


def test_public_pair_group_id_reproduces_existing_lineage_and_binds_inputs(
    complete_pair,
) -> None:
    *_, trajectory, mother, paired_config, group = complete_pair
    inputs = {
        "generated_event_id": mother.generated_event_id,
        "base_state_id": mother.world.base_state_id,
        "trajectory_id": trajectory.trajectory_id,
        "occluders": mother.world.occluders,
        "blind_spot_config": mother.world.blind_spot_config,
        "source_snippet_id": mother.target.snippet_id,
        "target_dynamic_object_id": mother.target.target_dynamic_object_id,
        "paired_config_digest": paired_config.digest,
    }

    expected = paired_variants_module.compute_pair_group_id(**inputs)

    assert expected == group.pair_group_id
    mutations = (
        {"generated_event_id": "event-different"},
        {"base_state_id": "base-different"},
        {"trajectory_id": "trajectory-different"},
        {"occluders": ({"occluder_id": "different"},)},
        {"blind_spot_config": {"kind": "different"}},
        {"source_snippet_id": "snippet-different"},
        {"target_dynamic_object_id": "target-different"},
        {"paired_config_digest": "0" * 32},
    )
    for mutation in mutations:
        assert paired_variants_module.compute_pair_group_id(
            **{**inputs, **mutation}
        ) != expected


def test_variant_metadata_separates_mother_join_evidence_from_paired_world(
    complete_pair,
) -> None:
    *_, mother, _, group = complete_pair
    stale_sop05_join_keys = {
        "generated_event_id",
        "source_snippet_id",
        "source_object_id",
        "target_object_type",
        "target_footprint_spec",
        "target_footprint_spec_digest",
        "target_type_policy_digest",
        "event_target_motion_layout_version",
        "target_history_array_digest",
        "target_future_array_digest",
        "target_motion_record_digest",
        "target_current_pose",
    }
    for variant in group.variants:
        metadata = variant.world.metadata
        assert not stale_sop05_join_keys.intersection(metadata)
        assert metadata["world_id"] == variant.world.world_id
        assert metadata["mother_generated_event_id"] == mother.generated_event_id
        assert metadata["mother_world_id"] == mother.world.world_id
        assert metadata["mother_target_motion_record_digest"] == (
            mother.target_motion_record.record_digest
        )
        assert metadata["mother_source_snippet_id"] == mother.target.snippet_id
        assert metadata["mother_source_object_id"] == mother.target.source_object_id
        assert metadata["mother_target_type_policy_digest"] == (
            mother.target_motion_record.target_type_policy_digest
        )


def test_pair_recomputes_auditable_history_visibility_for_every_variant(
    complete_pair,
) -> None:
    *_, group = complete_pair
    for variant in group.variants:
        if variant.target is None:
            assert variant.target_visibility_history is None
            assert variant.visibility_sequence is None
            continue
        assert variant.target_visibility_history is not None
        assert variant.target_visibility_history.shape == (8,)
        assert variant.target_visibility_history.dtype == np.bool_
        assert variant.visibility_sequence is not None
        assert variant.visibility_sequence.shape == (16,)
        assert variant.visibility_sequence.dtype == np.bool_
        assert not bool(variant.target_visibility_history[-1])
        assert variant.target_visibility_history[-1] == variant.visibility_sequence[0]


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    (
        ("out_of_bounds", "target_out_of_bounds"),
        ("static", "target_static_collision"),
        ("occluder", "target_occluder_collision"),
        ("context", "target_context_collision"),
    ),
)
def test_candidate_validation_covers_history_before_current(
    complete_pair,
    case: str,
    expected_reason: str,
) -> None:
    config, grid, base, oracle, trajectory, mother, paired_config, _ = complete_pair
    environment = paired_variants_module._pair_environment(
        mother_event=mother,
        trajectory=trajectory,
        base_state=base,
        oracle_context=oracle,
        base_config=config,
        critical_clearance_threshold_m=(
            paired_config.near_miss_clearance_range_m[1]
        ),
    )
    history = mother.target.history_poses.copy()
    if case == "out_of_bounds":
        history[0, :2] = np.asarray([100.0, 100.0], dtype=np.float32)
    elif case in {"static", "occluder"}:
        history[0] = np.asarray([-5.0, -5.0, 0.0], dtype=np.float32)
        if case == "static":
            occupied = environment.base_static_occupancy.copy()
            occupied |= rasterize_footprint(
                environment.target_footprint,
                history[0],
                grid,
            )
            environment = replace(
                environment, base_static_occupancy=occupied
            )
        else:
            environment = replace(
                environment,
                occluder_geometry=(
                    (
                        RectangleFootprint(1.0, 1.0),
                        history[0].copy(),
                    ),
                ),
            )
    else:
        context_id = next(iter(oracle.dynamic_object_history))
        history[0] = oracle.dynamic_object_history[context_id][0]
    candidate = replace(mother.target, history_poses=history)

    with pytest.raises(
        paired_variants_module._CandidateRejected
    ) as exc_info:
        paired_variants_module._validate_target_candidate(
            candidate,
            trajectory=trajectory,
            base_state=base,
            oracle_context=oracle,
            base_config=config,
            environment=environment,
        )
    assert exc_info.value.reason == expected_reason


def test_candidate_validation_rejects_context_collision_between_safe_frames(
    complete_pair,
) -> None:
    config, _, base, oracle, trajectory, mother, paired_config, _ = complete_pair
    environment = paired_variants_module._pair_environment(
        mother_event=mother,
        trajectory=trajectory,
        base_state=base,
        oracle_context=oracle,
        base_config=config,
        critical_clearance_threshold_m=(
            paired_config.near_miss_clearance_range_m[1]
        ),
    )
    target_poses = np.vstack(
        (mother.target.history_poses, mother.target.future_poses)
    )
    context_id = next(iter(oracle.dynamic_object_history))
    offsets = np.ones((target_poses.shape[0], 1), dtype=np.float32)
    offsets[target_poses.shape[0] // 2 :] = np.float32(-1.0)
    context_poses = target_poses.copy()
    context_poses[:, 0:1] += offsets
    context_poses[:, 2] = np.float32(0.0)
    endpoint_clearances = trajectory_signed_clearances(
        environment.target_footprint,
        target_poses,
        environment.context_footprints[context_id],
        context_poses,
    )
    assert np.all(endpoint_clearances > 0.0)
    crossing_context = replace(
        oracle,
        dynamic_object_history={context_id: context_poses[:8].copy()},
        dynamic_object_future={context_id: context_poses[8:].copy()},
    )

    with pytest.raises(paired_variants_module._CandidateRejected) as exc_info:
        paired_variants_module._validate_target_candidate(
            mother.target,
            trajectory=trajectory,
            base_state=base,
            oracle_context=crossing_context,
            base_config=config,
            environment=environment,
        )

    assert exc_info.value.reason == "target_context_collision"


def test_candidate_validation_rejects_static_contact_between_rotation_samples(
    complete_pair,
) -> None:
    config, grid, base, oracle, trajectory, mother, paired_config, _ = complete_pair
    environment = paired_variants_module._pair_environment(
        mother_event=mother,
        trajectory=trajectory,
        base_state=base,
        oracle_context=oracle,
        base_config=config,
        critical_clearance_threshold_m=(
            paired_config.near_miss_clearance_range_m[1]
        ),
    )
    target_center = grid_to_world(np.asarray([110, 80], dtype=np.int64), grid)
    occupied_index = np.asarray([111, 110], dtype=np.int64)
    occupied_center = grid_to_world(occupied_index, grid)
    relative_center = occupied_center - target_center
    contact_yaw = float(np.arctan2(relative_center[1], relative_center[0]))
    target_footprint = RectangleFootprint(
        2.0 * float(np.linalg.norm(relative_center)),
        0.002,
    )
    start_pose = np.asarray(
        [*target_center, contact_yaw - np.deg2rad(2.25)], dtype=np.float32
    )
    end_pose = np.asarray(
        [*target_center, contact_yaw + np.deg2rad(6.75)], dtype=np.float32
    )
    poses = np.tile(end_pose, (23, 1)).astype(np.float32)
    poses[0] = start_pose
    target = replace(
        mother.target,
        object_type="carried_object",
        history_poses=poses[:8].copy(),
        current_pose=poses[7].copy(),
        future_poses=poses[8:].copy(),
    )
    occupancy = np.zeros((grid.height, grid.width), dtype=bool)
    occupancy[tuple(occupied_index)] = True
    cell_footprint = RectangleFootprint(grid.resolution_m, grid.resolution_m)
    cell_pose = np.asarray([*occupied_center, 0.0])
    fixed_samples = np.asarray(
        [
            start_pose,
            [*target_center, contact_yaw + np.deg2rad(2.25)],
            end_pose,
        ],
        dtype=np.float64,
    )
    assert all(
        signed_clearance(target_footprint, pose, cell_footprint, cell_pose) > 0.0
        for pose in fixed_samples
    )
    environment = replace(
        environment,
        target_footprint=target_footprint,
        base_static_occupancy=occupancy,
    )

    with pytest.raises(paired_variants_module._CandidateRejected) as exc_info:
        paired_variants_module._validate_target_candidate(
            target,
            trajectory=trajectory,
            base_state=base,
            oracle_context=oracle,
            base_config=config,
            environment=environment,
        )

    assert exc_info.value.reason == "target_static_collision"


def test_base_state_only_actor_participates_in_history_collision_without_future(
    complete_pair,
) -> None:
    config, _, base, oracle, trajectory, mother, paired_config, _ = complete_pair
    actor_id = "base-only-visible-actor"
    actor_spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.3},
    }
    base_with_actor = replace(
        base,
        dynamic_object_ids=base.dynamic_object_ids + (actor_id,),
        visible_dynamic_object_history={
            **base.visible_dynamic_object_history,
            actor_id: mother.target.history_poses.copy(),
        },
        visible_dynamic_object_specs={
            **base.visible_dynamic_object_specs,
            actor_id: actor_spec,
        },
    )

    with pytest.raises(PairGenerationError) as exc_info:
        generate_paired_variants(
            mother_event=mother,
            source_snippet=_exact_safe_curve_snippet(),
            base_state=base_with_actor,
            trajectory=trajectory,
            oracle_context=oracle,
            base_config=config,
            paired_config=paired_config,
            seed=20260716,
        )
    assert exc_info.value.reason == "collision_mother_invalid"
    assert "target_base_history_collision" in str(exc_info.value)


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    (
        ("robot_history", "occluder_robot_history_overlap"),
        ("base_only_history", "occluder_base_history_collision"),
    ),
)
def test_joint_replacement_occluder_checks_history_only_sweeps(
    complete_pair,
    case: str,
    expected_reason: str,
) -> None:
    config, grid, base, oracle, trajectory, mother, paired_config, group = (
        complete_pair
    )
    pose = np.asarray([5.0, -5.0, 0.0], dtype=np.float32)
    checked_base = base
    if case == "robot_history":
        robot_history = base.robot_history.copy()
        robot_history[0] = pose
        checked_base = replace(base, robot_history=robot_history)
    else:
        actor_id = "base-only-occluder-history"
        actor_history = np.tile(pose, (grid.history_steps, 1)).astype(np.float32)
        actor_spec = {
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": 0.3},
        }
        checked_base = replace(
            base,
            dynamic_object_ids=base.dynamic_object_ids + (actor_id,),
            visible_dynamic_object_history={
                **base.visible_dynamic_object_history,
                actor_id: actor_history,
            },
            visible_dynamic_object_specs={
                **base.visible_dynamic_object_specs,
                actor_id: actor_spec,
            },
        )
    environment = paired_variants_module._pair_environment(
        mother_event=mother,
        trajectory=trajectory,
        base_state=checked_base,
        oracle_context=oracle,
        base_config=config,
        critical_clearance_threshold_m=(
            paired_config.near_miss_clearance_range_m[1]
        ),
    )
    footprint = RectangleFootprint(0.4, 0.4)
    placement = occluder_sampler_module.OccluderPlacement(
        occluder={
            "occluder_id": "history-only-test-occluder",
            "type": "pillar",
            "pose": [float(value) for value in pose],
            "length_m": 0.4,
            "width_m": 0.4,
        },
        footprint=footprint,
        pose=pose.copy(),
        mask=rasterize_footprint(footprint, pose, grid),
        attempt=1,
        rejection_reasons={},
    )

    reason = paired_variants_module._joint_occluder_full_motion_rejection_reason(
        placement=placement,
        base_state=checked_base,
        trajectory=trajectory,
        oracle_context=oracle,
        environment=environment,
        collision_target=mother.target,
        temporal_target=group.by_kind["spatial_safe"].target,
    )
    assert reason == expected_reason


def test_joint_collision_sweeps_cover_exact_available_motion(complete_pair) -> None:
    config, grid, base, oracle, trajectory, mother, paired_config, group = (
        complete_pair
    )
    base_only_id = "base-only-collision-sweep"
    base_only_history = np.tile(
        np.asarray([-6.0, -6.0, 0.0], dtype=np.float32),
        (grid.history_steps, 1),
    )
    base_with_history_only_actor = replace(
        base,
        dynamic_object_ids=base.dynamic_object_ids + (base_only_id,),
        visible_dynamic_object_history={
            **base.visible_dynamic_object_history,
            base_only_id: base_only_history,
        },
        visible_dynamic_object_specs={
            **base.visible_dynamic_object_specs,
            base_only_id: {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.3},
            },
        },
    )
    environment = paired_variants_module._pair_environment(
        mother_event=mother,
        trajectory=trajectory,
        base_state=base_with_history_only_actor,
        oracle_context=oracle,
        base_config=config,
        critical_clearance_threshold_m=(
            paired_config.near_miss_clearance_range_m[1]
        ),
    )
    temporal_target = group.by_kind["spatial_safe"].target
    assert temporal_target is not None

    sweeps = paired_variants_module._joint_occluder_collision_sweeps(
        base_state=base_with_history_only_actor,
        trajectory=trajectory,
        oracle_context=oracle,
        environment=environment,
        collision_target=mother.target,
        temporal_target=temporal_target,
    )

    assert all(
        isinstance(sweep, occluder_sampler_module.OccluderCollisionSweep)
        for sweep in sweeps
    )
    expected_count = 1 + len(environment.context_footprints) + 1 + 2
    assert len(sweeps) == expected_count
    cursor = 0
    robot_sweep = sweeps[cursor]
    cursor += 1
    assert robot_sweep.rejection_reason == "occluder_robot_history_overlap"
    np.testing.assert_array_equal(
        robot_sweep.poses,
        np.vstack((base_with_history_only_actor.robot_history, trajectory.poses)),
    )
    assert robot_sweep.poses.shape == (23, 3)

    for object_id in sorted(environment.context_footprints):
        context_sweep = sweeps[cursor]
        cursor += 1
        assert context_sweep.rejection_reason == (
            "occluder_context_full_motion_collision"
        )
        np.testing.assert_array_equal(
            context_sweep.poses,
            np.vstack(
                (
                    oracle.dynamic_object_history[object_id],
                    oracle.dynamic_object_future[object_id],
                )
            ),
        )
        assert context_sweep.poses.shape == (23, 3)

    base_only_sweep = sweeps[cursor]
    cursor += 1
    assert base_only_sweep.rejection_reason == (
        "occluder_base_history_collision"
    )
    np.testing.assert_array_equal(base_only_sweep.poses, base_only_history)
    assert base_only_sweep.poses.shape == (8, 3)

    for target in (mother.target, temporal_target):
        target_sweep = sweeps[cursor]
        cursor += 1
        assert target_sweep.rejection_reason == (
            "occluder_target_full_motion_collision"
        )
        np.testing.assert_array_equal(
            target_sweep.poses,
            np.vstack((target.history_poses, target.future_poses)),
        )
        assert target_sweep.poses.shape == (23, 3)
    assert cursor == len(sweeps)


def test_joint_replacement_occluder_certifies_between_frame_robot_rotation(
    complete_pair,
) -> None:
    config, grid, base, oracle, trajectory, mother, paired_config, _ = (
        complete_pair
    )
    environment = paired_variants_module._pair_environment(
        mother_event=mother,
        trajectory=trajectory,
        base_state=base,
        oracle_context=oracle,
        base_config=config,
        critical_clearance_threshold_m=(
            paired_config.near_miss_clearance_range_m[1]
        ),
    )
    maximum_support_yaw = np.arctan2(0.425, 0.5)
    start_yaw = np.float32(maximum_support_yaw - np.deg2rad(2.5))
    end_yaw = np.float32(maximum_support_yaw + np.deg2rad(2.5))
    robot_history = np.tile(
        np.asarray([0.0, 0.0, start_yaw], dtype=np.float32),
        (grid.history_steps, 1),
    )
    robot_future = np.tile(
        np.asarray([0.0, 0.0, end_yaw], dtype=np.float32),
        (grid.future_steps, 1),
    )
    checked_base = replace(base, robot_history=robot_history)
    checked_trajectory = replace(trajectory, poses=robot_future)
    occluder_pose = np.asarray([0.856, 0.0, 0.0], dtype=np.float32)
    occluder_footprint = RectangleFootprint(0.4, 0.4)
    placement = occluder_sampler_module.OccluderPlacement(
        occluder={
            "occluder_id": "between-frame-rotation-occluder",
            "type": "pillar",
            "pose": [float(value) for value in occluder_pose],
            "length_m": 0.4,
            "width_m": 0.4,
        },
        footprint=occluder_footprint,
        pose=occluder_pose,
        mask=rasterize_footprint(occluder_footprint, occluder_pose, grid),
        attempt=1,
        rejection_reasons={},
    )
    endpoint_poses = np.vstack((robot_history[-1], robot_future[0]))
    endpoint_clearances = trajectory_signed_clearances(
        occluder_footprint,
        np.tile(occluder_pose, (2, 1)),
        environment.robot_footprint,
        endpoint_poses,
    )
    assert np.all(endpoint_clearances > 0.0)
    assert occluder_sampler_module._intersects_robot_sweep(
        occluder_footprint,
        occluder_pose,
        environment.robot_footprint,
        endpoint_poses,
        grid=grid,
    )
    far_history = np.tile(
        np.asarray([5.0, -5.0, 0.0], dtype=np.float32),
        (grid.history_steps, 1),
    )
    far_future = np.tile(
        np.asarray([5.0, -5.0, 0.0], dtype=np.float32),
        (grid.future_steps, 1),
    )
    far_target = replace(
        mother.target,
        history_poses=far_history,
        current_pose=far_history[-1].copy(),
        future_poses=far_future,
    )

    reason = paired_variants_module._joint_occluder_full_motion_rejection_reason(
        placement=placement,
        base_state=checked_base,
        trajectory=checked_trajectory,
        oracle_context=oracle,
        environment=environment,
        collision_target=far_target,
        temporal_target=far_target,
    )
    assert reason == "occluder_robot_history_overlap"


def test_overlapping_base_and_oracle_actor_history_and_spec_must_match(
    complete_pair,
) -> None:
    config, _, base, oracle, trajectory, mother, paired_config, _ = complete_pair
    actor_id = next(iter(oracle.dynamic_object_history))
    changed_history = base.visible_dynamic_object_history[actor_id].copy()
    changed_history[0, 0] += np.float32(0.01)
    inconsistent_base = replace(
        base,
        visible_dynamic_object_history={
            **base.visible_dynamic_object_history,
            actor_id: changed_history,
        },
    )

    with pytest.raises(PairGenerationError) as exc_info:
        generate_paired_variants(
            mother_event=mother,
            source_snippet=_exact_safe_curve_snippet(),
            base_state=inconsistent_base,
            trajectory=trajectory,
            oracle_context=oracle,
            base_config=config,
            paired_config=paired_config,
            seed=20260716,
        )
    assert exc_info.value.reason == "base_oracle_history_mismatch"


def test_rebinding_joint_mother_rebuilds_strict_target_motion_join(
    complete_pair,
) -> None:
    config, grid, base, oracle, trajectory, mother, paired_config, _ = (
        complete_pair
    )
    new_world_id = "world-" + "1" * 24
    changed_world = replace(
        mother.world,
        world_id=new_world_id,
        metadata={
            **mother.world.metadata,
            "world_id": new_world_id,
            "joint_pair_generator_algorithm_version": (
                "joint_environment_pair_v2"
            ),
        },
    )

    rebound = paired_variants_module._rebind_mother_world(
        mother,
        changed_world,
        base_state=base,
        trajectory=trajectory,
        oracle_context=oracle,
        base_config=config,
        critical_clearance_threshold_m=(
            paired_config.near_miss_clearance_range_m[1]
        ),
        grid=grid,
    )

    assert rebound.generated_event_id == mother.generated_event_id
    assert rebound.world.world_id == new_world_id
    assert rebound.target_motion_record.world_id == new_world_id
    np.testing.assert_array_equal(
        rebound.target_motion_record.history_poses,
        mother.target.history_poses,
    )
    np.testing.assert_array_equal(
        rebound.target_motion_record.future_poses,
        mother.target.future_poses,
    )
    validate_event_target_motion_world_join(
        rebound.target_motion_record,
        rebound.world,
        grid,
    )


def test_pair_rejects_tampered_mother_target_motion_record_join(
    complete_pair,
) -> None:
    config, _, base, oracle, trajectory, mother, paired_config, _ = complete_pair
    changed_history = mother.target_motion_record.history_poses.copy()
    changed_history[0, 0] += np.float32(0.01)
    tampered = replace(
        mother,
        target_motion_record=replace(
            mother.target_motion_record,
            history_poses=changed_history,
        ),
    )

    with pytest.raises(PairGenerationError) as exc_info:
        generate_paired_variants(
            mother_event=tampered,
            source_snippet=_exact_safe_curve_snippet(),
            base_state=base,
            trajectory=trajectory,
            oracle_context=oracle,
            base_config=config,
            paired_config=paired_config,
            seed=20260716,
        )
    assert exc_info.value.reason == "mother_target_motion_join_invalid"


@pytest.mark.parametrize(
    "case",
    (
        "target_id",
        "object_type",
        "source_snippet_id",
        "source_object_id",
        "footprint_spec",
        "target_type_policy_digest",
    ),
)
def test_pair_rejects_valid_record_whose_target_identity_differs_from_mother(
    complete_pair,
    case: str,
) -> None:
    config, _, base, oracle, trajectory, mother, paired_config, _ = complete_pair
    overrides = {}
    if case == "target_id":
        overrides["target_dynamic_object_id"] = "different-target-id"
    elif case == "object_type":
        footprint_spec = {
            "object_type": "carried_object",
            "footprint": {
                "kind": "rectangle",
                "length_m": 0.8,
                "width_m": 0.2,
            },
        }
        overrides.update(
            object_type="carried_object",
            footprint_spec=footprint_spec,
            footprint_spec_digest=compute_footprint_spec_digest(
                footprint_spec
            ),
        )
    elif case == "source_snippet_id":
        overrides["source_snippet_id"] = "different-source-snippet"
    elif case == "source_object_id":
        overrides["source_object_id"] = "different-source-object"
    elif case == "footprint_spec":
        footprint_spec = {
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": 0.31},
        }
        overrides.update(
            footprint_spec=footprint_spec,
            footprint_spec_digest=compute_footprint_spec_digest(
                footprint_spec
            ),
        )
    else:
        overrides["target_type_policy_digest"] = "0" * 32
    changed_mother = _mother_with_rebuilt_motion_record(mother, **overrides)

    with pytest.raises(PairGenerationError) as exc_info:
        generate_paired_variants(
            mother_event=changed_mother,
            source_snippet=_exact_safe_curve_snippet(),
            base_state=base,
            trajectory=trajectory,
            oracle_context=oracle,
            base_config=config,
            paired_config=paired_config,
            seed=20260716,
        )
    assert exc_info.value.reason == "mother_target_motion_record_mismatch"


def test_pair_rejects_oracle_world_object_not_declared_by_context_or_target(
    complete_pair,
) -> None:
    config, _, base, oracle, trajectory, mother, paired_config, _ = complete_pair
    extra_id = "undeclared-object-colliding-with-robot"
    extra_spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.3},
    }
    changed_world = replace(
        mother.world,
        dynamic_object_trajectories={
            **mother.world.dynamic_object_trajectories,
            extra_id: trajectory.poses.copy(),
        },
        dynamic_object_specs={
            **mother.world.dynamic_object_specs,
            extra_id: extra_spec,
        },
    )

    with pytest.raises(PairGenerationError) as exc_info:
        generate_paired_variants(
            mother_event=replace(mother, world=changed_world),
            source_snippet=_exact_safe_curve_snippet(),
            base_state=base,
            trajectory=trajectory,
            oracle_context=oracle,
            base_config=config,
            paired_config=paired_config,
            seed=20260716,
        )
    assert exc_info.value.reason == "mother_dynamic_object_ids_mismatch"


def test_pair_generation_is_elementwise_deterministic(complete_pair) -> None:
    (
        config,
        _,
        base,
        oracle,
        trajectory,
        mother,
        paired_config,
        first,
    ) = complete_pair
    second = generate_paired_variants(
        mother_event=mother,
        source_snippet=_exact_safe_curve_snippet(),
        base_state=base,
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
        missing_variant_reasons={
            "temporal_safe": "temporal_variant_still_collides",
            "irrelevant_hidden": "clearance_unavailable",
        },
        paired_config=paired_config,
    )

    assert partial.coverage_mask == (True, True, False, True, False, True)
    assert partial.is_complete is False
    assert partial.eligible_for_strict_evaluation is False
    assert partial.missing_variant_reasons == {
        "temporal_safe": "temporal_variant_still_collides",
        "irrelevant_hidden": "clearance_unavailable"
    }
    summary = summarize_paired_groups((complete, partial))
    assert summary["group_count"] == 2
    assert summary["complete_group_count"] == 0
    assert summary["partial_group_count"] == 2
    assert summary["coverage_counts"]["irrelevant_hidden"] == 1

    collision_only = {"collision": complete.by_kind["collision"]}
    singleton = assemble_paired_event_group(
        pair_group_id=complete.pair_group_id,
        variants=collision_only,
        missing_variant_reasons={
            kind: "not_generated"
            for kind in VARIANT_ORDER
            if kind != "collision"
        },
        paired_config=paired_config,
    )
    assert singleton.coverage_mask == (
        True,
        False,
        False,
        False,
        False,
        False,
    )
    assert singleton.is_complete is False

    with pytest.raises(PairGenerationError, match="mother required"):
        assemble_paired_event_group(
            pair_group_id=complete.pair_group_id,
            variants={
                "empty_blind_spot": complete.by_kind["empty_blind_spot"]
            },
            missing_variant_reasons={
                kind: "not_generated"
                for kind in VARIANT_ORDER
                if kind != "empty_blind_spot"
            },
            paired_config=paired_config,
        )


def test_pair_rejects_context_actor_that_is_itself_critical(complete_pair) -> None:
    (
        config,
        _,
        base,
        oracle,
        trajectory,
        mother,
        paired_config,
        _,
    ) = complete_pair
    context_id = next(iter(oracle.dynamic_object_future))
    colliding_context = replace(
        oracle,
        dynamic_object_history={
            context_id: np.zeros_like(oracle.dynamic_object_history[context_id])
        },
        dynamic_object_future={context_id: trajectory.poses.copy()},
    )
    colliding_base = replace(
        base,
        visible_dynamic_object_history={
            context_id: colliding_context.dynamic_object_history[context_id].copy()
        },
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
            source_snippet=_exact_safe_curve_snippet(),
            base_state=colliding_base,
            trajectory=trajectory,
            oracle_context=colliding_context,
            base_config=config,
            paired_config=paired_config,
            seed=20260716,
        )
