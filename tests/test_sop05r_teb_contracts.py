from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.generation.sop05r_contracts import (
    SOP05R_TEB_GENERATOR_VERSION,
    SOP05R_TEB_OCCLUSION_VERSION,
    SOP05R_TEB_PLACEMENT_VERSION,
    SOP05R_TEB_PLANNER_VERSION,
    SOP05R_TEB_TEMPLATE_VERSION,
    load_sop05r_teb_config,
    normalize_sop05r_config,
    normalize_sop05r_teb_config,
)


def _valid_config() -> dict[str, object]:
    return {
        "schema_version": "3.0.0",
        "generator_algorithm_version": SOP05R_TEB_GENERATOR_VERSION,
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
            "goal_distances_m": [2.4, 2.8],
        },
        "planner": {
            "version": SOP05R_TEB_PLANNER_VERSION,
            "band_node_count": 20,
            "max_iterations": 40,
            "initialization_ids": ["straight"],
            "initial_band_dt_s": 0.25,
            "band_dt_range_s": [0.1, 0.4],
            "maximum_route_time_s": 5.0,
            "route_sample_dt_s": 0.2,
            "goal_position_tolerance_m": 0.25,
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
            "coarse_rotation_step_deg": 10.0,
            "refinement_radius_deg": 10.0,
            "refinement_step_deg": 2.0,
            "refined_candidate_count": 2,
            "internal_snippet_anchor_margin_frames": 2,
        },
        "occlusion": {
            "version": SOP05R_TEB_OCCLUSION_VERSION,
            "centerline_intersection_epsilon_m": 0.01,
            "initial_visible_weight": 0.8,
            "initially_hidden_weight": 0.2,
            "minimum_decision_to_collision_margin_s": 1.2,
            "braking_margin_s": 0.4,
            "replanning_margin_s": 0.3,
        },
        "generation": {
            "max_templates_per_base": 32,
            "max_target_snippets_per_template": 10,
            "max_route_anchor_candidates": 8,
            "conflict_path_fraction_range": [0.4, 0.7],
            "conflict_time_range_s": [1.0, 2.2],
            "direct_corridor_intrusion_range_m": [0.05, 0.15],
        },
        "revealability": {
            "minimum_visibility_lead_s": 0.2,
            "training_min_active_fraction": 0.7,
            "training_max_natural_difficult_fraction": 0.3,
            "selection_filtering": True,
        },
        "publication": {
            "trajectory_collection_version": "sop05r_nominal_trajectory_collection_v2",
            "run_producer_version": "sop05r_lightweight_teb_generation_run_v1",
            "manifest_version": "sop05r_lightweight_teb_manifest_v1",
            "summary_version": "sop05r_lightweight_teb_summary_v1",
            "completion_marker_version": "sop05r_lightweight_teb_producer_complete_v1",
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
            "initial_visibility_missing",
            "occlusion_witness_missing",
            "decision_margin_insufficient",
            "no_continuous_collision",
            "endpoint_only_collision",
            "active_revealability_quota_deficit",
            "quota_unmet",
        ],
    }


def test_teb_config_freezes_versions_shapes_scales_and_digest() -> None:
    config = normalize_sop05r_teb_config(_valid_config())

    assert config.generator_algorithm_version == SOP05R_TEB_GENERATOR_VERSION
    assert config.template.version == SOP05R_TEB_TEMPLATE_VERSION
    assert config.planner.version == SOP05R_TEB_PLANNER_VERSION
    assert config.placement.version == SOP05R_TEB_PLACEMENT_VERSION
    assert config.occlusion.version == SOP05R_TEB_OCCLUSION_VERSION
    assert config.placement.spatial_scale == 1.0
    assert config.template.occluder_shapes == ("rectangle", "l_shape", "circle")
    assert dict(config.template.family_weights) == {
        "rectangle": 0.4,
        "l_shape": 0.4,
        "circle": 0.2,
    }
    assert config.template.relative_yaw_abs_range_deg == (15.0, 45.0)
    assert config.template.occluders[2].arm_lengths_m == (1.4, 1.0)
    assert config.template.occluders[2].arm_width_m == 0.35
    assert config.occlusion.initial_visible_weight == 0.8
    assert config.occlusion.initially_hidden_weight == 0.2
    assert config.revealability.training_min_active_fraction == 0.7
    assert config.generation.direct_corridor_intrusion_range_m == (0.05, 0.15)
    assert config.planner.band_node_count == 20
    assert config.planner.initial_band_dt_s == 0.25
    assert config.planner.band_dt_range_s == (0.1, 0.4)
    assert config.planner.maximum_route_time_s == 5.0
    assert config.planner.route_sample_dt_s == 0.2
    assert config.planner.represented_occluder_clearance_range_m == (0.15, 0.75)
    assert config.planner.bypass_tracking_allowance_m == 0.08

    changed = _valid_config()
    changed["placement"]["coarse_rotation_step_deg"] = 15.0  # type: ignore[index]
    assert normalize_sop05r_teb_config(changed).digest != config.digest


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (None, "schema_version"),
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
        ("planner", "band_node_count"),
        ("planner", "initial_band_dt_s"),
        ("planner", "maximum_route_time_s"),
        ("placement", "coarse_rotation_step_deg"),
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

    invalid_weights = _valid_config()
    invalid_weights["occlusion"]["initial_visible_weight"] = 0.7  # type: ignore[index]
    invalid_weights["occlusion"]["initially_hidden_weight"] = 0.3  # type: ignore[index]
    with pytest.raises(ValueError, match="0.8.*0.2"):
        normalize_sop05r_teb_config(invalid_weights)

    invalid_range = _valid_config()
    invalid_range["generation"]["conflict_path_fraction_range"] = [0.7, 0.4]  # type: ignore[index]
    with pytest.raises(ValueError, match="conflict_path_fraction_range"):
        normalize_sop05r_teb_config(invalid_range)

    invalid_intrusion_range = _valid_config()
    invalid_intrusion_range["generation"]["direct_corridor_intrusion_range_m"] = [0.15, 0.05]  # type: ignore[index]
    with pytest.raises(ValueError, match="direct_corridor_intrusion_range_m"):
        normalize_sop05r_teb_config(invalid_intrusion_range)


def test_teb_config_rejects_invalid_dual_horizon_schedule() -> None:
    wrong_band_count = _valid_config()
    wrong_band_count["planner"]["band_node_count"] = 19  # type: ignore[index]
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


def test_v1_normalizer_rejects_v2_config_at_the_mode_boundary() -> None:
    with pytest.raises(ValueError, match="requires obstacle_first_teb mode"):
        normalize_sop05r_config(_valid_config())


def test_teb_config_is_immutable_and_production_configs_only_differ_by_selection() -> None:
    config = normalize_sop05r_teb_config(_valid_config())
    with pytest.raises(FrozenInstanceError):
        config.planner.band_node_count = 12  # type: ignore[misc]

    train = load_sop05r_teb_config(
        Path("configs/generator_obstacle_first_teb_train.yaml")
    )
    test = load_sop05r_teb_config(
        Path("configs/generator_obstacle_first_teb_test.yaml")
    )
    assert train.revealability.selection_filtering is True
    assert test.revealability.selection_filtering is False
    train_payload = train.as_dict()
    train_payload["revealability"]["selection_filtering"] = False
    assert train_payload == test.as_dict()
