from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.contracts import SCHEMA_VERSION
from src.generation import sop05r_contracts as contracts
from src.generation.sop05r_contracts import (
    SOP05R_TEB_COMPLETION_MARKER_VERSION,
    SOP05R_TEB_GENERATOR_VERSION,
    SOP05R_TEB_MANIFEST_VERSION,
    SOP05R_TEB_OCCLUSION_VERSION,
    SOP05R_TEB_PLACEMENT_VERSION,
    SOP05R_TEB_PLANNER_VERSION,
    SOP05R_TEB_RUN_VERSION,
    SOP05R_TEB_SUMMARY_VERSION,
    SOP05R_TEB_TEMPLATE_VERSION,
    SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION,
    load_sop05r_config,
    load_sop05r_teb_config,
    normalize_sop05r_config,
    normalize_sop05r_teb_config,
)


def _valid_config() -> dict[str, object]:
    return {
        "schema_version": "4.0.0",
        "generator_algorithm_version": SOP05R_TEB_GENERATOR_VERSION,
        "trajectory": {
            "layout_version": "history8_current7_future32_v1",
            "history_steps": 8,
            "current_index": 7,
            "future_steps": 32,
            "future_dt_s": 0.2,
            "future_horizon_s": 6.4,
        },
        "template": {
            "version": SOP05R_TEB_TEMPLATE_VERSION,
            "occluders": [
                {
                    "template_id": "wall_small",
                    "shape": "rectangle",
                    "semantic_type": "wall",
                    "length_m": 1.4,
                    "width_m": 0.35,
                },
                {
                    "template_id": "shelf_medium",
                    "shape": "rectangle",
                    "semantic_type": "shelf",
                    "length_m": 2.1,
                    "width_m": 0.55,
                },
                {
                    "template_id": "l_wall_small",
                    "shape": "l_shape",
                    "semantic_type": "wall",
                    "arm_lengths_m": [1.4, 1.0],
                    "arm_width_m": 0.35,
                },
                {
                    "template_id": "l_shelf_medium",
                    "shape": "l_shape",
                    "semantic_type": "shelf",
                    "arm_lengths_m": [2.0, 1.4],
                    "arm_width_m": 0.5,
                },
                {
                    "template_id": "tree_trunk_small",
                    "shape": "circle",
                    "semantic_type": "tree_trunk",
                    "radius_m": 0.35,
                },
            ],
            "family_weights": {
                "rectangle": 0.4,
                "l_shape": 0.4,
                "circle": 0.2,
            },
            "relative_yaw_abs_range_deg": [15.0, 45.0],
            "goal_bearings_deg": [-30.0, 0.0, 30.0],
            "goal_distances_m": [4.0, 4.5],
        },
        "planner": {
            "version": SOP05R_TEB_PLANNER_VERSION,
            "band_node_count": 21,
            "max_iterations": 40,
            "initialization_ids": ["straight", "bypass_left", "bypass_right"],
            "initial_band_dt_s": 0.25,
            "band_dt_range_s": [0.1, 0.4],
            "maximum_route_time_s": 8.0,
            "route_sample_dt_s": 0.2,
            "goal_position_tolerance_m": 0.5,
            "goal_yaw_tolerance_rad": 0.35,
            "max_linear_acceleration_mps2": 1.0,
            "max_angular_acceleration_radps2": 2.0,
            "max_curvature_per_m": 2.0,
            "represented_occluder_clearance_range_m": [0.15, 0.75],
            "bypass_tracking_allowance_m": 0.08,
            "weights": {
                "length": 1.0,
                "time": 0.1,
                "smoothness": 0.1,
                "obstacle": 4.0,
                "nonholonomic": 1.0,
                "velocity": 1.0,
                "acceleration": 1.0,
                "goal_heading": 0.1,
                "initial_control": 1.0,
            },
        },
        "placement": {
            "version": SOP05R_TEB_PLACEMENT_VERSION,
            "temporal_scales": [0.9, 1.0, 1.1],
            "spatial_scale": 1.0,
            "occluder_angular_margin_step_deg": 5.0,
            "internal_snippet_anchor_margin_frames": 0,
        },
        "occlusion": {
            "version": SOP05R_TEB_OCCLUSION_VERSION,
            "centerline_intersection_epsilon_m": 0.01,
            "minimum_visible_history_frames": 4,
            "minimum_occluded_history_frames": 1,
            "minimum_decision_to_collision_margin_s": 1.2,
            "braking_margin_s": 0.4,
            "replanning_margin_s": 0.3,
        },
        "generation": {
            "max_templates_per_base": 32,
            "max_target_snippets_per_template": 10,
            "max_route_anchor_candidates": 8,
            "collision_route_path_fraction_range": [0.2, 0.95],
            "minimum_direct_corridor_intrusion_m": 0.15,
        },
        "revealability": {
            "minimum_visibility_lead_s": 0.2,
            "training_min_active_fraction": 0.7,
            "training_max_natural_difficult_fraction": 0.3,
            "selection_filtering": True,
        },
        "publication": {
            "trajectory_collection_version": SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION,
            "run_producer_version": SOP05R_TEB_RUN_VERSION,
            "manifest_version": SOP05R_TEB_MANIFEST_VERSION,
            "summary_version": SOP05R_TEB_SUMMARY_VERSION,
            "completion_marker_version": SOP05R_TEB_COMPLETION_MARKER_VERSION,
        },
        "rejection_reasons": [
            "occluder_out_of_bounds",
            "source_static_overlap",
            "robot_history_overlap",
            "context_overlap",
            "direct_path_clear",
            "teb_no_route",
            "teb_dynamics_limit",
            "teb_static_collision",
            "teb_goal_unreached",
            "target_out_of_bounds",
            "target_occluder_collision",
            "target_source_static_collision",
            "target_context_collision",
            "target_speed_limit",
            "target_acceleration_limit",
            "guide_ray_degenerate",
            "half_plane_margin_missing",
            "occlusion_witness_missing",
            "decision_margin_insufficient",
            "no_continuous_collision",
            "endpoint_only_collision",
            "active_revealability_quota_deficit",
            "quota_unmet",
        ],
    }


def test_long40_semantic_versions_are_exact_and_canonical_schema_is_four() -> None:
    assert SCHEMA_VERSION == "4.0.0"
    assert getattr(contracts, "SOP05R_LONG40_SCHEMA_VERSION", None) == "4.0.0"
    assert (
        getattr(contracts, "SOP05R_LONG40_LAYOUT_VERSION", None)
        == "history8_current7_future32_v1"
    )
    assert SOP05R_TEB_GENERATOR_VERSION == "obstacle_first_lightweight_teb_v8"
    assert SOP05R_TEB_TEMPLATE_VERSION == "goal_occluder_template_schedule_v3"
    assert SOP05R_TEB_PLANNER_VERSION == "lightweight_teb_planner_v3"
    assert SOP05R_TEB_PLACEMENT_VERSION == "anchored_human_half_plane_step_long40_v8"
    assert SOP05R_TEB_OCCLUSION_VERSION == "seen_then_occlude_prefix4_v4"
    assert (
        SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION
        == "sop05r_nominal_trajectory_collection_v8"
    )
    assert SOP05R_TEB_RUN_VERSION == "sop05r_lightweight_teb_generation_run_v7"
    assert SOP05R_TEB_MANIFEST_VERSION == "sop05r_lightweight_teb_manifest_v7"
    assert SOP05R_TEB_SUMMARY_VERSION == "sop05r_lightweight_teb_summary_v8"
    assert (
        SOP05R_TEB_COMPLETION_MARKER_VERSION
        == "sop05r_lightweight_teb_producer_complete_v7"
    )


def test_teb_config_freezes_long40_layout_versions_and_digest() -> None:
    config = normalize_sop05r_teb_config(_valid_config())

    assert config.schema_version == "4.0.0"
    assert config.generator_algorithm_version == SOP05R_TEB_GENERATOR_VERSION
    assert config.trajectory.layout_version == "history8_current7_future32_v1"
    assert config.trajectory.history_steps == 8
    assert config.trajectory.current_index == 7
    assert config.trajectory.future_steps == 32
    assert config.trajectory.future_dt_s == 0.2
    assert config.trajectory.future_horizon_s == 6.4
    assert config.template.version == SOP05R_TEB_TEMPLATE_VERSION
    assert config.planner.version == SOP05R_TEB_PLANNER_VERSION
    assert config.placement.version == SOP05R_TEB_PLACEMENT_VERSION
    assert config.occlusion.version == SOP05R_TEB_OCCLUSION_VERSION
    assert config.occlusion.minimum_visible_history_frames == 4
    assert config.occlusion.minimum_occluded_history_frames == 1
    assert config.occlusion.minimum_decision_to_collision_margin_s == 1.2
    assert config.generation.collision_route_path_fraction_range == (0.2, 0.95)
    assert config.placement.spatial_scale == 1.0
    assert config.placement.internal_snippet_anchor_margin_frames == 0
    assert set(config.placement.as_dict()) == {
        "version",
        "temporal_scales",
        "spatial_scale",
        "occluder_angular_margin_step_deg",
        "internal_snippet_anchor_margin_frames",
    }
    assert set(config.generation.as_dict()) == {
        "max_templates_per_base",
        "max_target_snippets_per_template",
        "max_route_anchor_candidates",
        "collision_route_path_fraction_range",
        "minimum_direct_corridor_intrusion_m",
    }
    assert config.template.occluder_shapes == ("rectangle", "l_shape", "circle")
    assert dict(config.template.family_weights) == {
        "rectangle": 0.4,
        "l_shape": 0.4,
        "circle": 0.2,
    }
    assert config.template.relative_yaw_abs_range_deg == (15.0, 45.0)
    assert config.template.goal_distances_m == (4.0, 4.5)
    assert config.template.occluders[2].arm_lengths_m == (1.4, 1.0)
    assert config.template.occluders[2].arm_width_m == 0.35
    assert config.placement.occluder_angular_margin_step_deg == 5.0
    assert config.revealability.training_min_active_fraction == 0.7
    assert config.generation.minimum_direct_corridor_intrusion_m == 0.15
    assert config.planner.band_node_count == 21
    assert config.planner.band_node_count - 1 == 20
    assert config.planner.initial_band_dt_s == 0.25
    assert config.planner.band_dt_range_s == (0.1, 0.4)
    assert config.planner.maximum_route_time_s == 8.0
    assert config.planner.route_sample_dt_s == 0.2
    assert config.planner.goal_position_tolerance_m == 0.5
    assert config.planner.initialization_ids == (
        "straight",
        "bypass_left",
        "bypass_right",
    )
    assert (
        config.planner.maximum_route_time_s / config.planner.route_sample_dt_s
        == 40
    )
    assert config.planner.represented_occluder_clearance_range_m == (0.15, 0.75)
    assert config.planner.bypass_tracking_allowance_m == 0.08

    canonical = json.dumps(
        config.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    assert config.digest == hashlib.sha256(canonical).hexdigest()

    changed = _valid_config()
    changed["placement"]["temporal_scales"] = [1.0]  # type: ignore[index]
    assert normalize_sop05r_teb_config(changed).digest != config.digest


def test_teb_config_freezes_half_plane_step_without_initial_visibility_quotas() -> None:
    raw = _valid_config()

    config = normalize_sop05r_teb_config(raw)

    assert config.placement.occluder_angular_margin_step_deg == 5.0
    assert "initial_visible_weight" not in config.occlusion.as_dict()
    assert "initially_hidden_weight" not in config.occlusion.as_dict()


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (None, "schema_version"),
        ("trajectory", "future_steps"),
        ("template", "occluders"),
        ("planner", "weights"),
        ("placement", "spatial_scale"),
        ("occlusion", "centerline_intersection_epsilon_m"),
        ("generation", "max_route_anchor_candidates"),
        ("revealability", "selection_filtering"),
        ("publication", "manifest_version"),
    ],
)
def test_teb_config_rejects_missing_and_unknown_keys(
    section: str | None,
    key: str,
) -> None:
    missing = _valid_config()
    node = missing if section is None else missing[section]
    del node[key]  # type: ignore[index]
    with pytest.raises(ValueError, match="keys"):
        normalize_sop05r_teb_config(missing)

    unknown = _valid_config()
    node = unknown if section is None else unknown[section]
    node["oracle_hint"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="keys"):
        normalize_sop05r_teb_config(unknown)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("trajectory", "history_steps"),
        ("trajectory", "future_dt_s"),
        ("planner", "band_node_count"),
        ("planner", "initial_band_dt_s"),
        ("planner", "maximum_route_time_s"),
        ("placement", "occluder_angular_margin_step_deg"),
        ("occlusion", "centerline_intersection_epsilon_m"),
        ("generation", "max_templates_per_base"),
        ("revealability", "minimum_visibility_lead_s"),
    ],
)
def test_teb_config_rejects_boolean_numeric_limits(section: str, key: str) -> None:
    raw = _valid_config()
    raw[section][key] = True  # type: ignore[index]

    with pytest.raises(TypeError, match=key):
        normalize_sop05r_teb_config(raw)


def test_teb_config_rejects_noncanonical_half_plane_margin_step() -> None:
    raw = _valid_config()
    raw["placement"]["occluder_angular_margin_step_deg"] = 10.0  # type: ignore[index]

    with pytest.raises(ValueError, match="occluder_angular_margin_step_deg"):
        normalize_sop05r_teb_config(raw)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("future_dt_s", float("nan")),
        ("future_horizon_s", float("inf")),
    ],
)
def test_teb_trajectory_config_rejects_nonfinite_values(
    key: str,
    value: float,
) -> None:
    raw = _valid_config()
    raw["trajectory"][key] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=key):
        normalize_sop05r_teb_config(raw)


def test_teb_config_rejects_invalid_shapes_scales_weights_and_ranges() -> None:
    invalid_shape = _valid_config()
    invalid_shape["template"]["occluders"][0]["shape"] = "triangle"  # type: ignore[index]
    with pytest.raises(ValueError, match="rectangle, l_shape, or circle"):
        normalize_sop05r_teb_config(invalid_shape)

    invalid_family_mix = _valid_config()
    invalid_family_mix["template"]["family_weights"]["circle"] = 0.4  # type: ignore[index]
    with pytest.raises(ValueError, match="rectangle=0.4, l_shape=0.4, circle=0.2"):
        normalize_sop05r_teb_config(invalid_family_mix)

    invalid_yaw_range = _valid_config()
    invalid_yaw_range["template"]["relative_yaw_abs_range_deg"] = [0.0, 45.0]  # type: ignore[index]
    with pytest.raises(ValueError, match="relative_yaw_abs_range_deg"):
        normalize_sop05r_teb_config(invalid_yaw_range)

    invalid_scale = _valid_config()
    invalid_scale["placement"]["spatial_scale"] = 0.9  # type: ignore[index]
    with pytest.raises(ValueError, match="spatial_scale"):
        normalize_sop05r_teb_config(invalid_scale)

    invalid_goal_distances = _valid_config()
    invalid_goal_distances["template"]["goal_distances_m"] = [4.0, 5.0]  # type: ignore[index]
    with pytest.raises(ValueError, match="goal_distances_m"):
        normalize_sop05r_teb_config(invalid_goal_distances)

    invalid_visible_frames = _valid_config()
    invalid_visible_frames["occlusion"]["minimum_visible_history_frames"] = 3  # type: ignore[index]
    with pytest.raises(ValueError, match="minimum_visible_history_frames"):
        normalize_sop05r_teb_config(invalid_visible_frames)

    invalid_occluded_frames = _valid_config()
    invalid_occluded_frames["occlusion"]["minimum_occluded_history_frames"] = 2  # type: ignore[index]
    with pytest.raises(ValueError, match="minimum_occluded_history_frames"):
        normalize_sop05r_teb_config(invalid_occluded_frames)

    invalid_collision_margin = _valid_config()
    invalid_collision_margin["occlusion"]["minimum_decision_to_collision_margin_s"] = 1.5  # type: ignore[index]
    with pytest.raises(ValueError, match="minimum_decision_to_collision_margin_s"):
        normalize_sop05r_teb_config(invalid_collision_margin)

    invalid_range = _valid_config()
    invalid_range["generation"]["collision_route_path_fraction_range"] = [0.95, 0.2]  # type: ignore[index]
    with pytest.raises(ValueError, match="collision_route_path_fraction_range"):
        normalize_sop05r_teb_config(invalid_range)

    invalid_minimum_intrusion = _valid_config()
    invalid_minimum_intrusion["generation"]["minimum_direct_corridor_intrusion_m"] = 0.149  # type: ignore[index]
    with pytest.raises(ValueError, match="minimum_direct_corridor_intrusion_m"):
        normalize_sop05r_teb_config(invalid_minimum_intrusion)


def test_teb_config_rejects_invalid_dual_horizon_schedule() -> None:
    wrong_band_count = _valid_config()
    wrong_band_count["planner"]["band_node_count"] = 20  # type: ignore[index]
    with pytest.raises(ValueError, match="band_node_count"):
        normalize_sop05r_teb_config(wrong_band_count)

    initial_dt_outside_bounds = _valid_config()
    initial_dt_outside_bounds["planner"]["initial_band_dt_s"] = 0.5  # type: ignore[index]
    with pytest.raises(ValueError, match="initial_band_dt_s"):
        normalize_sop05r_teb_config(initial_dt_outside_bounds)

    insufficient_time = _valid_config()
    insufficient_time["planner"]["band_dt_range_s"] = [0.1, 0.2]  # type: ignore[index]
    with pytest.raises(ValueError, match="band_dt_range_s"):
        normalize_sop05r_teb_config(insufficient_time)

    non_integral_sample_grid = _valid_config()
    non_integral_sample_grid["planner"]["route_sample_dt_s"] = 0.3  # type: ignore[index]
    with pytest.raises(ValueError, match="route_sample_dt_s"):
        normalize_sop05r_teb_config(non_integral_sample_grid)

    inconsistent_suffix = _valid_config()
    inconsistent_suffix["trajectory"]["future_horizon_s"] = 6.2  # type: ignore[index]
    with pytest.raises(ValueError, match="future_horizon_s"):
        normalize_sop05r_teb_config(inconsistent_suffix)


@pytest.mark.parametrize(
    ("section", "obsolete_key", "value"),
    [
        ("generation", "conflict_time_range_s", [1.0, 2.2]),
        ("generation", "conflict_path_fraction_range", [0.4, 0.7]),
        ("placement", "refinement_radius_deg", 10.0),
        ("placement", "refinement_step_deg", 2.0),
        ("placement", "refined_candidate_count", 2),
        ("placement", "coarse_rotation_step_deg", 10.0),
        ("placement", "boundary_refinement_step_deg", 1.0),
        ("placement", "maximum_sweep_deg", 180.0),
        ("placement", "occluder_angular_margins_deg", [60.0, 30.0, 10.0]),
    ],
)
def test_teb_config_rejects_removed_pre_long40_keys(
    section: str,
    obsolete_key: str,
    value: object,
) -> None:
    raw = _valid_config()
    raw[section][obsolete_key] = value  # type: ignore[index]

    with pytest.raises(ValueError, match="keys"):
        normalize_sop05r_teb_config(raw)


def test_v1_normalizer_rejects_v4_config_at_the_mode_boundary() -> None:
    with pytest.raises(ValueError, match="long40 v4.*obstacle_first_teb mode"):
        normalize_sop05r_config(_valid_config())


def test_teb_config_is_immutable_and_production_configs_only_differ_by_selection() -> None:
    config = normalize_sop05r_teb_config(_valid_config())
    with pytest.raises(FrozenInstanceError):
        config.planner.band_node_count = 12  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.trajectory.future_steps = 15  # type: ignore[misc]

    train = load_sop05r_teb_config(
        Path("configs/generator_obstacle_first_teb_train.yaml")
    )
    test = load_sop05r_teb_config(
        Path("configs/generator_obstacle_first_teb_test.yaml")
    )
    assert train.revealability.selection_filtering is True
    assert test.revealability.selection_filtering is False
    assert train.planner.goal_position_tolerance_m == 0.5
    assert test.planner.goal_position_tolerance_m == 0.5
    train_payload = train.as_dict()
    train_payload["revealability"]["selection_filtering"] = False
    assert train_payload == test.as_dict()


def test_retired_v1_generator_config_is_not_shipped() -> None:
    assert not Path("configs/generator_obstacle_first_train.yaml").exists()


def test_v4_normalizer_identifies_v1_config_boundary() -> None:
    retired = {
        "schema_version": "4.0.0",
        "generator_algorithm_version": "obstacle_first_event_generation_v1",
    }
    with pytest.raises(ValueError, match="SOP05R v1.*obstacle_first mode"):
        normalize_sop05r_teb_config(retired)


def test_v4_normalizer_identifies_pre_long40_v2_without_misrouting() -> None:
    pre_long40 = _valid_config()
    pre_long40["schema_version"] = "3.0.0"
    pre_long40["generator_algorithm_version"] = "obstacle_first_lightweight_teb_v2"
    with pytest.raises(
        ValueError,
        match="pre-long40 v2.*not accepted by the long40 v4 normalizer",
    ):
        normalize_sop05r_teb_config(pre_long40)


def test_v1_normalizer_identifies_pre_long40_v2_without_misrouting() -> None:
    pre_long40 = _valid_config()
    pre_long40["schema_version"] = "3.0.0"
    pre_long40["generator_algorithm_version"] = "obstacle_first_lightweight_teb_v2"
    with pytest.raises(
        ValueError,
        match="pre-long40 v2.*not accepted by the SOP05R v1 normalizer",
    ):
        normalize_sop05r_config(pre_long40)
