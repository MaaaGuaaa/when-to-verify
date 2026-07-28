from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from src.generation.sop05r_contracts import (
    SOP05R_ACTIVE_REVEALABILITY_VERSION,
    SOP05R_COMPLETION_MARKER_VERSION,
    SOP05R_GENERATOR_VERSION,
    SOP05R_HISTORY_POLICY_VERSION,
    SOP05R_MANIFEST_VERSION,
    SOP05R_PLANNER_VERSION,
    SOP05R_REPORT_VERSION,
    SOP05R_RUN_VERSION,
    SOP05R_SELECTION_VERSION,
    SOP05R_SUMMARY_VERSION,
    SOP05R_TEMPLATE_VERSION,
    SOP05R_TRAJECTORY_COLLECTION_VERSION,
    SOP05R_TEB_GENERATOR_VERSION,
    load_sop05r_config,
    load_sop05r_teb_config,
    normalize_sop05r_config,
    normalize_sop05r_teb_config,
)


def _valid_config() -> dict[str, object]:
    return {
        "schema_version": "4.0.0",
        "generator_algorithm_version": SOP05R_GENERATOR_VERSION,
        "template": {
            "version": SOP05R_TEMPLATE_VERSION,
            "obstacle_types": ["wall", "shelf"],
            "obstacle_sizes": [
                {
                    "template_id": "wall_small",
                    "obstacle_type": "wall",
                    "length_m": 1.2,
                    "width_m": 0.25,
                },
                {
                    "template_id": "wall_medium",
                    "obstacle_type": "wall",
                    "length_m": 2.0,
                    "width_m": 0.35,
                },
                {
                    "template_id": "shelf_small",
                    "obstacle_type": "shelf",
                    "length_m": 1.0,
                    "width_m": 0.4,
                },
                {
                    "template_id": "shelf_medium",
                    "obstacle_type": "shelf",
                    "length_m": 1.8,
                    "width_m": 0.55,
                },
            ],
            "relative_layouts": ["target_side", "opposite_side"],
            "goal_bearings_deg": [-30.0, 0.0, 30.0],
            "goal_distances_m": [2.3, 2.4],
            "target_time_scales": [1.0],
        },
        "planner": {
            "version": SOP05R_PLANNER_VERSION,
            "candidate_slot_ids": [
                "left_near",
                "left_far",
                "right_near",
                "right_far",
                "stop",
            ],
            "rollout_steps": 32,
            "dt_s": 0.2,
            "corner_clearance_m": 0.35,
            "lookahead_distance_m": 0.5,
            "goal_tolerance_m": 0.3,
            "max_linear_acceleration_mps2": 1.0,
            "max_angular_acceleration_radps2": 2.0,
            "max_curvature_per_m": 2.0,
            "represented_obstacle_clearance_range_m": [0.25, 0.75],
            "path_length_normalizer_m": 4.0,
            "heading_cost_weight": 0.1,
            "smoothness_cost_weight": 0.1,
        },
        "history_policy": {
            "version": SOP05R_HISTORY_POLICY_VERSION,
            "history_steps": 8,
            "min_trailing_hidden_frames": 2,
            "weights": {
                "seen_then_occluded": 0.8,
                "unseen_in_history_window": 0.2,
            },
        },
        "revealability": {
            "version": SOP05R_ACTIVE_REVEALABILITY_VERSION,
            "minimum_visibility_lead_s": 0.2,
            "minimum_post_visibility_margin_s": 0.4,
            "training_min_active_fraction": 0.7,
            "training_max_natural_difficult_fraction": 0.3,
            "selection_filtering": True,
        },
        "generation": {
            "max_templates_per_base": 32,
            "max_target_snippets_per_template": 10,
            "max_time_alignments_per_path": 8,
            "conflict_path_fraction_range": [0.4, 0.7],
            "conflict_time_range_s": [1.0, 2.2],
            "goal_beyond_conflict_range_m": [1.0, 2.0],
            "fallback_to_legacy_generator": False,
        },
        "publication": {
            "trajectory_collection_version": SOP05R_TRAJECTORY_COLLECTION_VERSION,
            "run_producer_version": SOP05R_RUN_VERSION,
            "pair_report_version": SOP05R_REPORT_VERSION,
            "selection_version": SOP05R_SELECTION_VERSION,
            "manifest_version": SOP05R_MANIFEST_VERSION,
            "summary_version": SOP05R_SUMMARY_VERSION,
            "completion_marker_version": SOP05R_COMPLETION_MARKER_VERSION,
        },
        "rejection_reasons": [
            "obstacle_out_of_bounds",
            "obstacle_mask_empty",
            "target_out_of_bounds",
            "source_static_overlap",
            "robot_history_overlap",
            "context_overlap",
            "source_extrapolation_required",
            "target_obstacle_collision",
            "target_source_static_collision",
            "target_context_collision",
            "target_currently_visible",
            "target_motion_contract_invalid",
            "target_speed_limit",
            "target_acceleration_limit",
            "target_current_robot_overlap",
            "no_reachable_obstacle_channel",
            "planner_no_route",
            "planner_dynamics_limit",
            "planner_static_collision",
            "planner_goal_unreached",
            "history_ineligible",
            "no_time_aligned_collision",
            "conflict_path_fraction_out_of_range",
            "conflict_time_out_of_range",
            "goal_not_beyond_conflict",
            "direct_path_misses_obstacle",
            "nominal_clearance_out_of_range",
            "no_same_goal_alternative",
            "active_revealability_quota_deficit",
            "quota_unmet",
        ],
    }


def test_obstacle_first_config_freezes_versions_and_scientific_digest() -> None:
    config = normalize_sop05r_config(_valid_config())

    assert config.generator_algorithm_version == SOP05R_GENERATOR_VERSION
    assert config.template.version == SOP05R_TEMPLATE_VERSION
    assert config.planner.version == SOP05R_PLANNER_VERSION
    assert config.history_policy.weights == {
        "seen_then_occluded": 0.8,
        "unseen_in_history_window": 0.2,
    }
    assert config.publication.as_dict() == _valid_config()["publication"]
    canonical = json.dumps(
        config.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    assert config.digest == hashlib.sha256(canonical).hexdigest()

    changed = _valid_config()
    changed["template"]["goal_distances_m"] = [2.2, 2.4]  # type: ignore[index]
    assert normalize_sop05r_config(changed).digest != config.digest


def test_config_is_immutable_and_uses_finite_template_axes() -> None:
    config = normalize_sop05r_config(_valid_config())

    assert config.template.obstacle_types == ("wall", "shelf")
    assert [row.template_id for row in config.template.obstacle_sizes] == [
        "wall_small",
        "wall_medium",
        "shelf_small",
        "shelf_medium",
    ]
    assert config.template.relative_layouts == ("target_side", "opposite_side")
    assert config.template.goal_bearings_deg == (-30.0, 0.0, 30.0)
    assert config.template.goal_distances_m == (2.3, 2.4)
    assert config.template.target_time_scales == (1.0,)
    with pytest.raises(FrozenInstanceError):
        config.planner.dt_s = 0.1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        replace(config.template.obstacle_sizes[0], length_m=1.3).length_m = 1.4  # type: ignore[misc]


def test_config_requires_exact_planner_slots_and_distance_dominant_score() -> None:
    config = normalize_sop05r_config(_valid_config())

    assert config.planner.candidate_slot_ids == (
        "left_near",
        "left_far",
        "right_near",
        "right_far",
        "stop",
    )
    assert (
        config.planner.heading_cost_weight
        + config.planner.smoothness_cost_weight
        < 1.0
    )

    wrong_slots = _valid_config()
    wrong_slots["planner"]["candidate_slot_ids"] = [  # type: ignore[index]
        "left_near",
        "right_near",
        "stop",
    ]
    with pytest.raises(ValueError, match="candidate_slot_ids"):
        normalize_sop05r_config(wrong_slots)

    non_dominant = _valid_config()
    non_dominant["planner"]["heading_cost_weight"] = 0.6  # type: ignore[index]
    non_dominant["planner"]["smoothness_cost_weight"] = 0.4  # type: ignore[index]
    with pytest.raises(ValueError, match="secondary.*less than one"):
        normalize_sop05r_config(non_dominant)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (None, "schema_version"),
        ("template", "goal_bearings_deg"),
        ("planner", "rollout_steps"),
        ("history_policy", "weights"),
        ("revealability", "selection_filtering"),
        ("generation", "max_templates_per_base"),
        ("publication", "manifest_version"),
    ],
)
def test_config_rejects_missing_and_unknown_keys(
    section: str | None,
    key: str,
) -> None:
    missing = _valid_config()
    node = missing if section is None else missing[section]
    del node[key]  # type: ignore[index]
    with pytest.raises(ValueError, match="keys"):
        normalize_sop05r_config(missing)

    unknown = _valid_config()
    node = unknown if section is None else unknown[section]
    node["oracle_hint"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="keys"):
        normalize_sop05r_config(unknown)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("planner", "rollout_steps"),
        ("planner", "dt_s"),
        ("history_policy", "history_steps"),
        ("revealability", "minimum_visibility_lead_s"),
        ("generation", "max_templates_per_base"),
    ],
)
def test_config_rejects_boolean_numeric_limits(section: str, key: str) -> None:
    raw = _valid_config()
    raw[section][key] = True  # type: ignore[index]

    with pytest.raises(TypeError, match=key):
        normalize_sop05r_config(raw)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("template", "goal_distances_m", [2.3, float("nan")]),
        ("template", "target_time_scales", [1.0, float("inf")]),
        ("planner", "represented_obstacle_clearance_range_m", [0.75, 0.25]),
        ("generation", "conflict_path_fraction_range", [0.7, 0.4]),
        ("generation", "conflict_time_range_s", [2.2, 1.0]),
        ("generation", "goal_beyond_conflict_range_m", [0.0, 2.0]),
    ],
)
def test_config_rejects_nonfinite_or_unordered_ranges(
    section: str,
    key: str,
    value: object,
) -> None:
    raw = _valid_config()
    raw[section][key] = value  # type: ignore[index]

    with pytest.raises((TypeError, ValueError), match=key):
        normalize_sop05r_config(raw)


def test_config_requires_wall_shelf_and_exact_eighty_twenty_history() -> None:
    bad_type = _valid_config()
    bad_type["template"]["obstacle_types"] = ["wall", "pillar"]  # type: ignore[index]
    with pytest.raises(ValueError, match="wall.*shelf"):
        normalize_sop05r_config(bad_type)

    bad_weights = _valid_config()
    bad_weights["history_policy"]["weights"] = {  # type: ignore[index]
        "seen_then_occluded": 0.7,
        "unseen_in_history_window": 0.3,
    }
    with pytest.raises(ValueError, match="0.8.*0.2"):
        normalize_sop05r_config(bad_weights)


def test_production_configs_differ_only_in_selection_filtering() -> None:
    train = load_sop05r_config(Path("configs/generator_obstacle_first_train.yaml"))
    test = load_sop05r_config(Path("configs/generator_obstacle_first_test.yaml"))

    assert train.revealability.selection_filtering is True
    assert test.revealability.selection_filtering is False
    assert train.template.goal_distances_m == (2.3, 2.4)
    assert test.template.goal_distances_m == (2.3, 2.4)
    assert train.template.target_time_scales == (1.0,)
    assert test.template.target_time_scales == (1.0,)
    assert train.revealability.training_min_active_fraction == 0.7
    assert test.revealability.training_min_active_fraction == 0.7
    train_payload = train.as_dict()
    test_payload = test.as_dict()
    train_payload["revealability"]["selection_filtering"] = False
    assert train_payload == test_payload


def test_v1_and_v2_configs_are_rejected_across_normalizer_boundaries() -> None:
    teb_config = load_sop05r_teb_config(
        Path("configs/generator_obstacle_first_teb_train.yaml")
    )
    assert teb_config.generator_algorithm_version == SOP05R_TEB_GENERATOR_VERSION
    with pytest.raises(ValueError, match="requires obstacle_first_teb mode"):
        normalize_sop05r_config(teb_config.as_dict())
    with pytest.raises(ValueError, match="obstacle_first mode"):
        normalize_sop05r_teb_config(_valid_config())
