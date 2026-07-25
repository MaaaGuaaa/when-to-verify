"""Strict immutable contracts for SOP05R obstacle-first generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.contracts import SCHEMA_VERSION


SOP05R_GENERATOR_VERSION = "obstacle_first_event_generation_v1"
SOP05R_TEMPLATE_VERSION = "obstacle_target_template_schedule_v1"
SOP05R_PLANNER_VERSION = "inflated_corner_waypoint_planner_v1"
SOP05R_TRAJECTORY_COLLECTION_VERSION = (
    "sop05r_planner_trajectory_collection_v1"
)
SOP05R_RUN_VERSION = "sop05r_generation_run_v1"
SOP05R_REPORT_VERSION = "sop05r_template_report_v1"
SOP05R_SELECTION_VERSION = "sop05r_stratified_selection_v1"
SOP05R_MANIFEST_VERSION = "sop05r_run_manifest_v1"
SOP05R_SUMMARY_VERSION = "sop05r_generation_summary_v1"
SOP05R_COMPLETION_MARKER_VERSION = "sop05r_producer_complete_v1"
SOP05R_HISTORY_POLICY_VERSION = "target_history_visibility_policy_v2_sop05r"
SOP05R_ACTIVE_REVEALABILITY_VERSION = "sop05r_active_revealability_v1"

SOP05R_PLANNER_SLOT_IDS = (
    "left_near",
    "left_far",
    "right_near",
    "right_far",
    "stop",
)
SOP05R_OBSTACLE_TYPES = ("wall", "shelf")
SOP05R_RELATIVE_LAYOUTS = ("target_side", "opposite_side")
SOP05R_OBSTACLE_SIZE_TEMPLATE_IDS = (
    "wall_small",
    "wall_medium",
    "shelf_small",
    "shelf_medium",
)
SOP05R_REJECTION_REASONS = (
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
)


class Sop05rConfigError(ValueError):
    """Raised when a SOP05R configuration violates the frozen contract."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise Sop05rConfigError("SOP05R config must be finite canonical ASCII JSON") from exc


def _strict_mapping(
    value: object,
    *,
    name: str,
    expected_keys: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != expected_keys:
        raise Sop05rConfigError(f"{name} keys do not match the frozen schema")
    return value


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise Sop05rConfigError(f"{name} must be finite")
    return result


def _positive_real(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise Sop05rConfigError(f"{name} must be positive")
    return result


def _integer(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise Sop05rConfigError(f"{name} must be at least {minimum}")
    return result


def _fraction(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise Sop05rConfigError(f"{name} must lie in [0, 1]")
    return result


def _ordered_range(
    value: object,
    *,
    name: str,
    positive: bool = False,
    unit_interval: bool = False,
) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise Sop05rConfigError(f"{name} must contain [minimum, maximum]")
    lower = _finite_real(value[0], name=f"{name}[0]")
    upper = _finite_real(value[1], name=f"{name}[1]")
    if lower >= upper:
        raise Sop05rConfigError(f"{name} must be a strictly ordered range")
    if positive and lower <= 0.0:
        raise Sop05rConfigError(f"{name} must be a positive ordered range")
    if unit_interval and (lower < 0.0 or upper > 1.0):
        raise Sop05rConfigError(f"{name} must lie in [0, 1]")
    return lower, upper


def _finite_axis(
    value: object,
    *,
    name: str,
    length: int,
    positive: bool = False,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise Sop05rConfigError(f"{name} must contain exactly {length} values")
    result = tuple(
        _finite_real(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if positive and any(item <= 0.0 for item in result):
        raise Sop05rConfigError(f"{name} values must be positive")
    if any(left >= right for left, right in zip(result, result[1:])):
        raise Sop05rConfigError(f"{name} must be strictly increasing")
    return result


def _exact_string_sequence(
    value: object,
    *,
    name: str,
    expected: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or tuple(value) != expected:
        expected_text = ", ".join(expected)
        raise Sop05rConfigError(f"{name} must equal [{expected_text}]")
    return expected


@dataclass(frozen=True)
class ObstacleSizeTemplate:
    template_id: str
    obstacle_type: str
    length_m: float
    width_m: float

    def as_dict(self) -> dict[str, object]:
        return {
            "template_id": self.template_id,
            "obstacle_type": self.obstacle_type,
            "length_m": self.length_m,
            "width_m": self.width_m,
        }


@dataclass(frozen=True)
class TemplateConfig:
    version: str
    obstacle_types: tuple[str, ...]
    obstacle_sizes: tuple[ObstacleSizeTemplate, ...]
    relative_layouts: tuple[str, ...]
    goal_bearings_deg: tuple[float, ...]
    goal_distances_m: tuple[float, ...]
    target_time_scales: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "obstacle_types": list(self.obstacle_types),
            "obstacle_sizes": [row.as_dict() for row in self.obstacle_sizes],
            "relative_layouts": list(self.relative_layouts),
            "goal_bearings_deg": list(self.goal_bearings_deg),
            "goal_distances_m": list(self.goal_distances_m),
            "target_time_scales": list(self.target_time_scales),
        }


@dataclass(frozen=True)
class PlannerConfig:
    version: str
    candidate_slot_ids: tuple[str, ...]
    rollout_steps: int
    dt_s: float
    corner_clearance_m: float
    lookahead_distance_m: float
    goal_tolerance_m: float
    max_linear_acceleration_mps2: float
    max_angular_acceleration_radps2: float
    max_curvature_per_m: float
    represented_obstacle_clearance_range_m: tuple[float, float]
    path_length_normalizer_m: float
    heading_cost_weight: float
    smoothness_cost_weight: float

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "candidate_slot_ids": list(self.candidate_slot_ids),
            "rollout_steps": self.rollout_steps,
            "dt_s": self.dt_s,
            "corner_clearance_m": self.corner_clearance_m,
            "lookahead_distance_m": self.lookahead_distance_m,
            "goal_tolerance_m": self.goal_tolerance_m,
            "max_linear_acceleration_mps2": self.max_linear_acceleration_mps2,
            "max_angular_acceleration_radps2": self.max_angular_acceleration_radps2,
            "max_curvature_per_m": self.max_curvature_per_m,
            "represented_obstacle_clearance_range_m": list(
                self.represented_obstacle_clearance_range_m
            ),
            "path_length_normalizer_m": self.path_length_normalizer_m,
            "heading_cost_weight": self.heading_cost_weight,
            "smoothness_cost_weight": self.smoothness_cost_weight,
        }


@dataclass(frozen=True)
class Sop05rHistoryPolicy:
    version: str
    history_steps: int
    min_trailing_hidden_frames: int
    seen_then_occluded_weight: float
    unseen_in_history_window_weight: float

    @property
    def weights(self) -> dict[str, float]:
        return {
            "seen_then_occluded": self.seen_then_occluded_weight,
            "unseen_in_history_window": self.unseen_in_history_window_weight,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "history_steps": self.history_steps,
            "min_trailing_hidden_frames": self.min_trailing_hidden_frames,
            "weights": self.weights,
        }


@dataclass(frozen=True)
class RevealabilityConfig:
    version: str
    minimum_visibility_lead_s: float
    minimum_post_visibility_margin_s: float
    training_min_active_fraction: float
    training_max_natural_difficult_fraction: float
    selection_filtering: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "minimum_visibility_lead_s": self.minimum_visibility_lead_s,
            "minimum_post_visibility_margin_s": self.minimum_post_visibility_margin_s,
            "training_min_active_fraction": self.training_min_active_fraction,
            "training_max_natural_difficult_fraction": (
                self.training_max_natural_difficult_fraction
            ),
            "selection_filtering": self.selection_filtering,
        }


@dataclass(frozen=True)
class GenerationLimits:
    max_templates_per_base: int
    max_target_snippets_per_template: int
    max_time_alignments_per_path: int
    conflict_path_fraction_range: tuple[float, float]
    conflict_time_range_s: tuple[float, float]
    goal_beyond_conflict_range_m: tuple[float, float]
    fallback_to_legacy_generator: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "max_templates_per_base": self.max_templates_per_base,
            "max_target_snippets_per_template": self.max_target_snippets_per_template,
            "max_time_alignments_per_path": self.max_time_alignments_per_path,
            "conflict_path_fraction_range": list(self.conflict_path_fraction_range),
            "conflict_time_range_s": list(self.conflict_time_range_s),
            "goal_beyond_conflict_range_m": list(
                self.goal_beyond_conflict_range_m
            ),
            "fallback_to_legacy_generator": self.fallback_to_legacy_generator,
        }


@dataclass(frozen=True)
class PublicationVersions:
    trajectory_collection_version: str
    run_producer_version: str
    pair_report_version: str
    selection_version: str
    manifest_version: str
    summary_version: str
    completion_marker_version: str

    def as_dict(self) -> dict[str, str]:
        return {
            "trajectory_collection_version": self.trajectory_collection_version,
            "run_producer_version": self.run_producer_version,
            "pair_report_version": self.pair_report_version,
            "selection_version": self.selection_version,
            "manifest_version": self.manifest_version,
            "summary_version": self.summary_version,
            "completion_marker_version": self.completion_marker_version,
        }


@dataclass(frozen=True)
class Sop05rConfig:
    schema_version: str
    generator_algorithm_version: str
    template: TemplateConfig
    planner: PlannerConfig
    history_policy: Sop05rHistoryPolicy
    revealability: RevealabilityConfig
    generation: GenerationLimits
    publication: PublicationVersions
    rejection_reasons: tuple[str, ...]
    digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generator_algorithm_version": self.generator_algorithm_version,
            "template": self.template.as_dict(),
            "planner": self.planner.as_dict(),
            "history_policy": self.history_policy.as_dict(),
            "revealability": self.revealability.as_dict(),
            "generation": self.generation.as_dict(),
            "publication": self.publication.as_dict(),
            "rejection_reasons": list(self.rejection_reasons),
        }


def _normalize_obstacle_sizes(value: object) -> tuple[ObstacleSizeTemplate, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise Sop05rConfigError("template.obstacle_sizes must contain four templates")
    result: list[ObstacleSizeTemplate] = []
    expected_keys = {"template_id", "obstacle_type", "length_m", "width_m"}
    for index, raw in enumerate(value):
        node = _strict_mapping(
            raw,
            name=f"template.obstacle_sizes[{index}]",
            expected_keys=expected_keys,
        )
        template_id = node["template_id"]
        obstacle_type = node["obstacle_type"]
        if not isinstance(template_id, str) or not template_id:
            raise TypeError("template.obstacle_sizes template_id must be nonempty text")
        if obstacle_type not in SOP05R_OBSTACLE_TYPES:
            raise Sop05rConfigError("obstacle_type must be wall or shelf")
        length_m = _positive_real(
            node["length_m"], name=f"template.obstacle_sizes[{index}].length_m"
        )
        width_m = _positive_real(
            node["width_m"], name=f"template.obstacle_sizes[{index}].width_m"
        )
        if not 1.0 <= length_m <= 2.5:
            raise Sop05rConfigError("obstacle length_m must lie in [1.0, 2.5]")
        if not 0.2 <= width_m <= 0.6:
            raise Sop05rConfigError("obstacle width_m must lie in [0.2, 0.6]")
        result.append(
            ObstacleSizeTemplate(
                template_id=template_id,
                obstacle_type=str(obstacle_type),
                length_m=length_m,
                width_m=width_m,
            )
        )
    if tuple(row.template_id for row in result) != SOP05R_OBSTACLE_SIZE_TEMPLATE_IDS:
        raise Sop05rConfigError(
            "template.obstacle_sizes must use the frozen small/medium IDs"
        )
    if tuple(row.obstacle_type for row in result) != (
        "wall",
        "wall",
        "shelf",
        "shelf",
    ):
        raise Sop05rConfigError(
            "template.obstacle_sizes IDs must bind wall/shelf types"
        )
    for small, medium in ((result[0], result[1]), (result[2], result[3])):
        if small.length_m >= medium.length_m or small.width_m >= medium.width_m:
            raise Sop05rConfigError(
                "medium obstacle templates must exceed matching small templates"
            )
    return tuple(result)


def _normalize_template(value: object) -> TemplateConfig:
    node = _strict_mapping(
        value,
        name="template",
        expected_keys={
            "version",
            "obstacle_types",
            "obstacle_sizes",
            "relative_layouts",
            "goal_bearings_deg",
            "goal_distances_m",
            "target_time_scales",
        },
    )
    if node["version"] != SOP05R_TEMPLATE_VERSION:
        raise Sop05rConfigError(f"template.version must equal {SOP05R_TEMPLATE_VERSION}")
    obstacle_types = _exact_string_sequence(
        node["obstacle_types"],
        name="template.obstacle_types",
        expected=SOP05R_OBSTACLE_TYPES,
    )
    relative_layouts = _exact_string_sequence(
        node["relative_layouts"],
        name="template.relative_layouts",
        expected=SOP05R_RELATIVE_LAYOUTS,
    )
    goal_bearings_deg = _finite_axis(
        node["goal_bearings_deg"],
        name="template.goal_bearings_deg",
        length=3,
    )
    if not (
        -90.0 <= goal_bearings_deg[0] < 0.0
        and goal_bearings_deg[1] == 0.0
        and 0.0 < goal_bearings_deg[2] <= 90.0
    ):
        raise Sop05rConfigError(
            "template.goal_bearings_deg must encode left, straight, right"
        )
    return TemplateConfig(
        version=SOP05R_TEMPLATE_VERSION,
        obstacle_types=obstacle_types,
        obstacle_sizes=_normalize_obstacle_sizes(node["obstacle_sizes"]),
        relative_layouts=relative_layouts,
        goal_bearings_deg=goal_bearings_deg,
        goal_distances_m=_finite_axis(
            node["goal_distances_m"],
            name="template.goal_distances_m",
            length=2,
            positive=True,
        ),
        target_time_scales=_finite_axis(
            node["target_time_scales"],
            name="template.target_time_scales",
            length=1,
            positive=True,
        ),
    )


def _normalize_planner(value: object) -> PlannerConfig:
    expected_keys = {
        "version",
        "candidate_slot_ids",
        "rollout_steps",
        "dt_s",
        "corner_clearance_m",
        "lookahead_distance_m",
        "goal_tolerance_m",
        "max_linear_acceleration_mps2",
        "max_angular_acceleration_radps2",
        "max_curvature_per_m",
        "represented_obstacle_clearance_range_m",
        "path_length_normalizer_m",
        "heading_cost_weight",
        "smoothness_cost_weight",
    }
    node = _strict_mapping(value, name="planner", expected_keys=expected_keys)
    if node["version"] != SOP05R_PLANNER_VERSION:
        raise Sop05rConfigError(f"planner.version must equal {SOP05R_PLANNER_VERSION}")
    rollout_steps = _integer(node["rollout_steps"], name="planner.rollout_steps")
    if rollout_steps != 15:
        raise Sop05rConfigError("planner.rollout_steps must equal 15")
    dt_s = _positive_real(node["dt_s"], name="planner.dt_s")
    if dt_s != 0.2:
        raise Sop05rConfigError("planner.dt_s must equal 0.2")
    heading_weight = _fraction(
        node["heading_cost_weight"], name="planner.heading_cost_weight"
    )
    smoothness_weight = _fraction(
        node["smoothness_cost_weight"], name="planner.smoothness_cost_weight"
    )
    if heading_weight + smoothness_weight >= 1.0:
        raise Sop05rConfigError(
            "planner secondary maximum contributions must sum to less than one"
        )
    return PlannerConfig(
        version=SOP05R_PLANNER_VERSION,
        candidate_slot_ids=_exact_string_sequence(
            node["candidate_slot_ids"],
            name="planner.candidate_slot_ids",
            expected=SOP05R_PLANNER_SLOT_IDS,
        ),
        rollout_steps=rollout_steps,
        dt_s=dt_s,
        corner_clearance_m=_positive_real(
            node["corner_clearance_m"], name="planner.corner_clearance_m"
        ),
        lookahead_distance_m=_positive_real(
            node["lookahead_distance_m"], name="planner.lookahead_distance_m"
        ),
        goal_tolerance_m=_positive_real(
            node["goal_tolerance_m"], name="planner.goal_tolerance_m"
        ),
        max_linear_acceleration_mps2=_positive_real(
            node["max_linear_acceleration_mps2"],
            name="planner.max_linear_acceleration_mps2",
        ),
        max_angular_acceleration_radps2=_positive_real(
            node["max_angular_acceleration_radps2"],
            name="planner.max_angular_acceleration_radps2",
        ),
        max_curvature_per_m=_positive_real(
            node["max_curvature_per_m"], name="planner.max_curvature_per_m"
        ),
        represented_obstacle_clearance_range_m=_ordered_range(
            node["represented_obstacle_clearance_range_m"],
            name="planner.represented_obstacle_clearance_range_m",
            positive=True,
        ),
        path_length_normalizer_m=_positive_real(
            node["path_length_normalizer_m"],
            name="planner.path_length_normalizer_m",
        ),
        heading_cost_weight=heading_weight,
        smoothness_cost_weight=smoothness_weight,
    )


def _normalize_history_policy(value: object) -> Sop05rHistoryPolicy:
    node = _strict_mapping(
        value,
        name="history_policy",
        expected_keys={
            "version",
            "history_steps",
            "min_trailing_hidden_frames",
            "weights",
        },
    )
    if node["version"] != SOP05R_HISTORY_POLICY_VERSION:
        raise Sop05rConfigError(
            f"history_policy.version must equal {SOP05R_HISTORY_POLICY_VERSION}"
        )
    history_steps = _integer(
        node["history_steps"], name="history_policy.history_steps"
    )
    if history_steps != 8:
        raise Sop05rConfigError("history_policy.history_steps must equal 8")
    trailing = _integer(
        node["min_trailing_hidden_frames"],
        name="history_policy.min_trailing_hidden_frames",
    )
    if trailing != 2:
        raise Sop05rConfigError(
            "history_policy.min_trailing_hidden_frames must equal 2"
        )
    weights = _strict_mapping(
        node["weights"],
        name="history_policy.weights",
        expected_keys={"seen_then_occluded", "unseen_in_history_window"},
    )
    seen_weight = _fraction(
        weights["seen_then_occluded"],
        name="history_policy.weights.seen_then_occluded",
    )
    unseen_weight = _fraction(
        weights["unseen_in_history_window"],
        name="history_policy.weights.unseen_in_history_window",
    )
    if seen_weight != 0.8 or unseen_weight != 0.2:
        raise Sop05rConfigError("history_policy weights must equal 0.8 and 0.2")
    return Sop05rHistoryPolicy(
        version=SOP05R_HISTORY_POLICY_VERSION,
        history_steps=history_steps,
        min_trailing_hidden_frames=trailing,
        seen_then_occluded_weight=seen_weight,
        unseen_in_history_window_weight=unseen_weight,
    )


def _normalize_revealability(value: object) -> RevealabilityConfig:
    node = _strict_mapping(
        value,
        name="revealability",
        expected_keys={
            "version",
            "minimum_visibility_lead_s",
            "minimum_post_visibility_margin_s",
            "training_min_active_fraction",
            "training_max_natural_difficult_fraction",
            "selection_filtering",
        },
    )
    if node["version"] != SOP05R_ACTIVE_REVEALABILITY_VERSION:
        raise Sop05rConfigError(
            "revealability.version must equal "
            f"{SOP05R_ACTIVE_REVEALABILITY_VERSION}"
        )
    active_fraction = _fraction(
        node["training_min_active_fraction"],
        name="revealability.training_min_active_fraction",
    )
    difficult_fraction = _fraction(
        node["training_max_natural_difficult_fraction"],
        name="revealability.training_max_natural_difficult_fraction",
    )
    if active_fraction < 0.7 or difficult_fraction > 0.3:
        raise Sop05rConfigError(
            "revealability training fractions must retain at least 0.7 active and at most 0.3 difficult"
        )
    if active_fraction + difficult_fraction != 1.0:
        raise Sop05rConfigError("revealability training fractions must sum to one")
    selection_filtering = node["selection_filtering"]
    if not isinstance(selection_filtering, bool):
        raise TypeError("revealability.selection_filtering must be boolean")
    return RevealabilityConfig(
        version=SOP05R_ACTIVE_REVEALABILITY_VERSION,
        minimum_visibility_lead_s=_positive_real(
            node["minimum_visibility_lead_s"],
            name="revealability.minimum_visibility_lead_s",
        ),
        minimum_post_visibility_margin_s=_positive_real(
            node["minimum_post_visibility_margin_s"],
            name="revealability.minimum_post_visibility_margin_s",
        ),
        training_min_active_fraction=active_fraction,
        training_max_natural_difficult_fraction=difficult_fraction,
        selection_filtering=selection_filtering,
    )


def _normalize_generation(value: object) -> GenerationLimits:
    node = _strict_mapping(
        value,
        name="generation",
        expected_keys={
            "max_templates_per_base",
            "max_target_snippets_per_template",
            "max_time_alignments_per_path",
            "conflict_path_fraction_range",
            "conflict_time_range_s",
            "goal_beyond_conflict_range_m",
            "fallback_to_legacy_generator",
        },
    )
    fallback = node["fallback_to_legacy_generator"]
    if not isinstance(fallback, bool):
        raise TypeError("generation.fallback_to_legacy_generator must be boolean")
    if fallback:
        raise Sop05rConfigError(
            "generation.fallback_to_legacy_generator must be false; legacy is a separate run"
        )
    return GenerationLimits(
        max_templates_per_base=_integer(
            node["max_templates_per_base"],
            name="generation.max_templates_per_base",
        ),
        max_target_snippets_per_template=_integer(
            node["max_target_snippets_per_template"],
            name="generation.max_target_snippets_per_template",
        ),
        max_time_alignments_per_path=_integer(
            node["max_time_alignments_per_path"],
            name="generation.max_time_alignments_per_path",
        ),
        conflict_path_fraction_range=_ordered_range(
            node["conflict_path_fraction_range"],
            name="generation.conflict_path_fraction_range",
            unit_interval=True,
        ),
        conflict_time_range_s=_ordered_range(
            node["conflict_time_range_s"],
            name="generation.conflict_time_range_s",
            positive=True,
        ),
        goal_beyond_conflict_range_m=_ordered_range(
            node["goal_beyond_conflict_range_m"],
            name="generation.goal_beyond_conflict_range_m",
            positive=True,
        ),
        fallback_to_legacy_generator=False,
    )


def _normalize_publication(value: object) -> PublicationVersions:
    expected = {
        "trajectory_collection_version": SOP05R_TRAJECTORY_COLLECTION_VERSION,
        "run_producer_version": SOP05R_RUN_VERSION,
        "pair_report_version": SOP05R_REPORT_VERSION,
        "selection_version": SOP05R_SELECTION_VERSION,
        "manifest_version": SOP05R_MANIFEST_VERSION,
        "summary_version": SOP05R_SUMMARY_VERSION,
        "completion_marker_version": SOP05R_COMPLETION_MARKER_VERSION,
    }
    node = _strict_mapping(
        value,
        name="publication",
        expected_keys=set(expected),
    )
    for key, frozen_value in expected.items():
        if node[key] != frozen_value:
            raise Sop05rConfigError(f"publication.{key} must equal {frozen_value}")
    return PublicationVersions(**expected)


def normalize_sop05r_config(config: Mapping[str, Any]) -> Sop05rConfig:
    """Validate and normalize the exact SOP05R configuration schema."""

    if config.get("generator_algorithm_version") == SOP05R_TEB_GENERATOR_VERSION:
        raise Sop05rConfigError(
            "generator_algorithm_version "
            f"{SOP05R_TEB_GENERATOR_VERSION} requires obstacle_first_teb mode"
        )
    node = _strict_mapping(
        config,
        name="SOP05R config",
        expected_keys={
            "schema_version",
            "generator_algorithm_version",
            "template",
            "planner",
            "history_policy",
            "revealability",
            "generation",
            "publication",
            "rejection_reasons",
        },
    )
    if node["schema_version"] != SCHEMA_VERSION:
        raise Sop05rConfigError(f"schema_version must equal {SCHEMA_VERSION}")
    if node["generator_algorithm_version"] != SOP05R_GENERATOR_VERSION:
        raise Sop05rConfigError(
            "generator_algorithm_version must equal "
            f"{SOP05R_GENERATOR_VERSION}"
        )
    rejection_reasons = _exact_string_sequence(
        node["rejection_reasons"],
        name="rejection_reasons",
        expected=SOP05R_REJECTION_REASONS,
    )
    template = _normalize_template(node["template"])
    planner = _normalize_planner(node["planner"])
    history_policy = _normalize_history_policy(node["history_policy"])
    revealability = _normalize_revealability(node["revealability"])
    generation = _normalize_generation(node["generation"])
    publication = _normalize_publication(node["publication"])
    normalized_without_digest = {
        "schema_version": SCHEMA_VERSION,
        "generator_algorithm_version": SOP05R_GENERATOR_VERSION,
        "template": template.as_dict(),
        "planner": planner.as_dict(),
        "history_policy": history_policy.as_dict(),
        "revealability": revealability.as_dict(),
        "generation": generation.as_dict(),
        "publication": publication.as_dict(),
        "rejection_reasons": list(rejection_reasons),
    }
    return Sop05rConfig(
        schema_version=SCHEMA_VERSION,
        generator_algorithm_version=SOP05R_GENERATOR_VERSION,
        template=template,
        planner=planner,
        history_policy=history_policy,
        revealability=revealability,
        generation=generation,
        publication=publication,
        rejection_reasons=rejection_reasons,
        digest=hashlib.sha256(_canonical_json_bytes(normalized_without_digest)).hexdigest(),
    )


def load_sop05r_config(path: str | Path) -> Sop05rConfig:
    """Load one standalone SOP05R config without defaults or legacy merging."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise Sop05rConfigError(f"failed to load SOP05R config: {config_path}") from exc
    if not isinstance(raw, Mapping):
        raise Sop05rConfigError("SOP05R config root must be a mapping")
    return normalize_sop05r_config(raw)


# SOP05R v2: lightweight TEB scene-generation contract.  This is intentionally
# separate from the v1 obstacle-first contract above: v1 artifacts must retain
# their normalization and semantic identities unchanged.
SOP05R_TEB_GENERATOR_VERSION = "obstacle_first_lightweight_teb_v2"
SOP05R_TEB_TEMPLATE_VERSION = "goal_occluder_template_schedule_v2"
SOP05R_TEB_PLANNER_VERSION = "lightweight_teb_planner_v2"
SOP05R_TEB_PLACEMENT_VERSION = "anchored_human_rotation_v1"
SOP05R_TEB_OCCLUSION_VERSION = "synchronized_centerline_occlusion_v1"
SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION = (
    "sop05r_nominal_trajectory_collection_v2"
)
SOP05R_TEB_RUN_VERSION = "sop05r_lightweight_teb_generation_run_v1"
SOP05R_TEB_MANIFEST_VERSION = "sop05r_lightweight_teb_manifest_v1"
SOP05R_TEB_SUMMARY_VERSION = "sop05r_lightweight_teb_summary_v1"
SOP05R_TEB_COMPLETION_MARKER_VERSION = (
    "sop05r_lightweight_teb_producer_complete_v1"
)

SOP05R_TEB_OCCLUDER_SHAPES = ("rectangle", "l_shape", "circle")
SOP05R_TEB_OCCLUDER_FAMILY_WEIGHTS = (
    ("rectangle", 0.4),
    ("l_shape", 0.4),
    ("circle", 0.2),
)
SOP05R_TEB_RECTANGLE_SEMANTIC_TYPES = ("wall", "shelf", "cabinet")
SOP05R_TEB_CIRCLE_SEMANTIC_TYPES = ("tree_trunk", "column")
SOP05R_TEB_INITIALIZATION_IDS = ("straight",)
SOP05R_TEB_WEIGHT_NAMES = (
    "length",
    "time",
    "smoothness",
    "obstacle",
    "nonholonomic",
    "velocity",
    "acceleration",
    "goal_heading",
    "initial_control",
)
SOP05R_TEB_REJECTION_REASONS = (
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
)


@dataclass(frozen=True)
class TebOccluderTemplate:
    template_id: str
    shape: str
    semantic_type: str
    length_m: float | None
    width_m: float | None
    arm_lengths_m: tuple[float, float] | None
    arm_width_m: float | None
    radius_m: float | None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "template_id": self.template_id,
            "shape": self.shape,
            "semantic_type": self.semantic_type,
        }
        if self.shape == "rectangle":
            result["length_m"] = self.length_m
            result["width_m"] = self.width_m
        elif self.shape == "l_shape":
            result["arm_lengths_m"] = list(self.arm_lengths_m or ())
            result["arm_width_m"] = self.arm_width_m
        else:
            result["radius_m"] = self.radius_m
        return result


@dataclass(frozen=True)
class TebTemplateConfig:
    version: str
    occluders: tuple[TebOccluderTemplate, ...]
    family_weights: tuple[tuple[str, float], ...]
    relative_yaw_abs_range_deg: tuple[float, float]
    goal_bearings_deg: tuple[float, ...]
    goal_distances_m: tuple[float, ...]

    @property
    def occluder_shapes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(row.shape for row in self.occluders))

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "occluders": [row.as_dict() for row in self.occluders],
            "family_weights": dict(self.family_weights),
            "relative_yaw_abs_range_deg": list(self.relative_yaw_abs_range_deg),
            "goal_bearings_deg": list(self.goal_bearings_deg),
            "goal_distances_m": list(self.goal_distances_m),
        }


@dataclass(frozen=True)
class LightweightTebConfig:
    version: str
    band_node_count: int
    max_iterations: int
    initialization_ids: tuple[str, ...]
    initial_band_dt_s: float
    band_dt_range_s: tuple[float, float]
    maximum_route_time_s: float
    route_sample_dt_s: float
    goal_position_tolerance_m: float
    goal_yaw_tolerance_rad: float
    max_linear_acceleration_mps2: float
    max_angular_acceleration_radps2: float
    max_curvature_per_m: float
    represented_occluder_clearance_range_m: tuple[float, float]
    bypass_tracking_allowance_m: float
    weights: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "band_node_count": self.band_node_count,
            "max_iterations": self.max_iterations,
            "initialization_ids": list(self.initialization_ids),
            "initial_band_dt_s": self.initial_band_dt_s,
            "band_dt_range_s": list(self.band_dt_range_s),
            "maximum_route_time_s": self.maximum_route_time_s,
            "route_sample_dt_s": self.route_sample_dt_s,
            "goal_position_tolerance_m": self.goal_position_tolerance_m,
            "goal_yaw_tolerance_rad": self.goal_yaw_tolerance_rad,
            "max_linear_acceleration_mps2": self.max_linear_acceleration_mps2,
            "max_angular_acceleration_radps2": self.max_angular_acceleration_radps2,
            "max_curvature_per_m": self.max_curvature_per_m,
            "represented_occluder_clearance_range_m": list(
                self.represented_occluder_clearance_range_m
            ),
            "bypass_tracking_allowance_m": self.bypass_tracking_allowance_m,
            "weights": dict(self.weights),
        }


@dataclass(frozen=True)
class AnchoredPlacementConfig:
    version: str
    temporal_scales: tuple[float, ...]
    spatial_scale: float
    coarse_rotation_step_deg: float
    refinement_radius_deg: float
    refinement_step_deg: float
    refined_candidate_count: int
    internal_snippet_anchor_margin_frames: int

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "temporal_scales": list(self.temporal_scales),
            "spatial_scale": self.spatial_scale,
            "coarse_rotation_step_deg": self.coarse_rotation_step_deg,
            "refinement_radius_deg": self.refinement_radius_deg,
            "refinement_step_deg": self.refinement_step_deg,
            "refined_candidate_count": self.refined_candidate_count,
            "internal_snippet_anchor_margin_frames": (
                self.internal_snippet_anchor_margin_frames
            ),
        }


@dataclass(frozen=True)
class CenterlineOcclusionConfig:
    version: str
    centerline_intersection_epsilon_m: float
    initial_visible_weight: float
    initially_hidden_weight: float
    minimum_decision_to_collision_margin_s: float
    braking_margin_s: float
    replanning_margin_s: float

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "centerline_intersection_epsilon_m": self.centerline_intersection_epsilon_m,
            "initial_visible_weight": self.initial_visible_weight,
            "initially_hidden_weight": self.initially_hidden_weight,
            "minimum_decision_to_collision_margin_s": (
                self.minimum_decision_to_collision_margin_s
            ),
            "braking_margin_s": self.braking_margin_s,
            "replanning_margin_s": self.replanning_margin_s,
        }


@dataclass(frozen=True)
class TebGenerationLimits:
    max_templates_per_base: int
    max_target_snippets_per_template: int
    max_route_anchor_candidates: int
    conflict_path_fraction_range: tuple[float, float]
    conflict_time_range_s: tuple[float, float]
    direct_corridor_intrusion_range_m: tuple[float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "max_templates_per_base": self.max_templates_per_base,
            "max_target_snippets_per_template": self.max_target_snippets_per_template,
            "max_route_anchor_candidates": self.max_route_anchor_candidates,
            "conflict_path_fraction_range": list(self.conflict_path_fraction_range),
            "conflict_time_range_s": list(self.conflict_time_range_s),
            "direct_corridor_intrusion_range_m": list(
                self.direct_corridor_intrusion_range_m
            ),
        }


@dataclass(frozen=True)
class TebRevealabilityConfig:
    minimum_visibility_lead_s: float
    training_min_active_fraction: float
    training_max_natural_difficult_fraction: float
    selection_filtering: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "minimum_visibility_lead_s": self.minimum_visibility_lead_s,
            "training_min_active_fraction": self.training_min_active_fraction,
            "training_max_natural_difficult_fraction": (
                self.training_max_natural_difficult_fraction
            ),
            "selection_filtering": self.selection_filtering,
        }


@dataclass(frozen=True)
class TebPublicationVersions:
    trajectory_collection_version: str
    run_producer_version: str
    manifest_version: str
    summary_version: str
    completion_marker_version: str

    def as_dict(self) -> dict[str, str]:
        return {
            "trajectory_collection_version": self.trajectory_collection_version,
            "run_producer_version": self.run_producer_version,
            "manifest_version": self.manifest_version,
            "summary_version": self.summary_version,
            "completion_marker_version": self.completion_marker_version,
        }


@dataclass(frozen=True)
class Sop05rTebConfig:
    schema_version: str
    generator_algorithm_version: str
    template: TebTemplateConfig
    planner: LightweightTebConfig
    placement: AnchoredPlacementConfig
    occlusion: CenterlineOcclusionConfig
    generation: TebGenerationLimits
    revealability: TebRevealabilityConfig
    publication: TebPublicationVersions
    rejection_reasons: tuple[str, ...]
    digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generator_algorithm_version": self.generator_algorithm_version,
            "template": self.template.as_dict(),
            "planner": self.planner.as_dict(),
            "placement": self.placement.as_dict(),
            "occlusion": self.occlusion.as_dict(),
            "generation": self.generation.as_dict(),
            "revealability": self.revealability.as_dict(),
            "publication": self.publication.as_dict(),
            "rejection_reasons": list(self.rejection_reasons),
        }


def _teb_axis(
    value: object,
    *,
    name: str,
    minimum_length: int,
    positive: bool = False,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) < minimum_length:
        raise Sop05rConfigError(f"{name} must contain at least {minimum_length} values")
    result = tuple(
        _finite_real(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if positive and any(item <= 0.0 for item in result):
        raise Sop05rConfigError(f"{name} values must be positive")
    if any(left >= right for left, right in zip(result, result[1:])):
        raise Sop05rConfigError(f"{name} must be strictly increasing")
    return result


def _normalize_teb_occluders(value: object) -> tuple[TebOccluderTemplate, ...]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise Sop05rConfigError("template.occluders must contain at least two templates")
    result: list[TebOccluderTemplate] = []
    template_ids: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TypeError(f"template.occluders[{index}] must be a mapping")
        shape = raw.get("shape")
        common = {"template_id", "shape", "semantic_type"}
        if shape == "rectangle":
            expected = common | {"length_m", "width_m"}
        elif shape == "l_shape":
            expected = common | {"arm_lengths_m", "arm_width_m"}
        elif shape == "circle":
            expected = common | {"radius_m"}
        else:
            raise Sop05rConfigError(
                f"template.occluders[{index}].shape must be rectangle, l_shape, or circle"
            )
        node = _strict_mapping(
            raw,
            name=f"template.occluders[{index}]",
            expected_keys=expected,
        )
        template_id = node["template_id"]
        semantic_type = node["semantic_type"]
        if not isinstance(template_id, str) or not template_id:
            raise TypeError(f"template.occluders[{index}].template_id must be text")
        if template_id in template_ids:
            raise Sop05rConfigError("template.occluders template_id values must be unique")
        template_ids.add(template_id)
        if not isinstance(semantic_type, str) or not semantic_type:
            raise TypeError(f"template.occluders[{index}].semantic_type must be text")
        if shape == "rectangle":
            if semantic_type not in SOP05R_TEB_RECTANGLE_SEMANTIC_TYPES:
                raise Sop05rConfigError(
                    "rectangle semantic_type must be wall, shelf, or cabinet"
                )
            result.append(
                TebOccluderTemplate(
                    template_id=template_id,
                    shape="rectangle",
                    semantic_type=semantic_type,
                    length_m=_positive_real(
                        node["length_m"],
                        name=f"template.occluders[{index}].length_m",
                    ),
                    width_m=_positive_real(
                        node["width_m"],
                        name=f"template.occluders[{index}].width_m",
                    ),
                    arm_lengths_m=None,
                    arm_width_m=None,
                    radius_m=None,
                )
            )
        elif shape == "l_shape":
            if semantic_type not in SOP05R_TEB_RECTANGLE_SEMANTIC_TYPES:
                raise Sop05rConfigError(
                    "l_shape semantic_type must be wall, shelf, or cabinet"
                )
            raw_arm_lengths = node["arm_lengths_m"]
            if not isinstance(raw_arm_lengths, (list, tuple)) or len(raw_arm_lengths) != 2:
                raise Sop05rConfigError(
                    "l_shape arm_lengths_m must contain exactly two values"
                )
            arm_lengths = tuple(
                _positive_real(
                    value,
                    name=f"template.occluders[{index}].arm_lengths_m[{arm_index}]",
                )
                for arm_index, value in enumerate(raw_arm_lengths)
            )
            arm_width = _positive_real(
                node["arm_width_m"],
                name=f"template.occluders[{index}].arm_width_m",
            )
            if arm_width >= min(arm_lengths):
                raise Sop05rConfigError(
                    "l_shape arm_width_m must be smaller than both arm lengths"
                )
            result.append(
                TebOccluderTemplate(
                    template_id=template_id,
                    shape="l_shape",
                    semantic_type=semantic_type,
                    length_m=None,
                    width_m=None,
                    arm_lengths_m=(arm_lengths[0], arm_lengths[1]),
                    arm_width_m=arm_width,
                    radius_m=None,
                )
            )
        else:
            if semantic_type not in SOP05R_TEB_CIRCLE_SEMANTIC_TYPES:
                raise Sop05rConfigError(
                    "circle semantic_type must be tree_trunk or column"
                )
            result.append(
                TebOccluderTemplate(
                    template_id=template_id,
                    shape="circle",
                    semantic_type=semantic_type,
                    length_m=None,
                    width_m=None,
                    arm_lengths_m=None,
                    arm_width_m=None,
                    radius_m=_positive_real(
                        node["radius_m"],
                        name=f"template.occluders[{index}].radius_m",
                    ),
                )
            )
    shapes = tuple(dict.fromkeys(row.shape for row in result))
    if shapes != SOP05R_TEB_OCCLUDER_SHAPES:
        raise Sop05rConfigError(
            "template.occluders must include rectangle templates before circle templates"
        )
    return tuple(result)


def _normalize_teb_template(value: object) -> TebTemplateConfig:
    node = _strict_mapping(
        value,
        name="template",
        expected_keys={
            "version",
            "occluders",
            "family_weights",
            "relative_yaw_abs_range_deg",
            "goal_bearings_deg",
            "goal_distances_m",
        },
    )
    if node["version"] != SOP05R_TEB_TEMPLATE_VERSION:
        raise Sop05rConfigError(
            f"template.version must equal {SOP05R_TEB_TEMPLATE_VERSION}"
        )
    bearings = _teb_axis(
        node["goal_bearings_deg"],
        name="template.goal_bearings_deg",
        minimum_length=3,
    )
    if not (bearings[0] < 0.0 and 0.0 in bearings and bearings[-1] > 0.0):
        raise Sop05rConfigError(
            "template.goal_bearings_deg must contain left, straight, and right"
        )
    family_weights = _strict_mapping(
        node["family_weights"],
        name="template.family_weights",
        expected_keys=set(SOP05R_TEB_OCCLUDER_SHAPES),
    )
    normalized_family_weights = tuple(
        (
            family,
            _positive_real(
                family_weights[family],
                name=f"template.family_weights.{family}",
            ),
        )
        for family in SOP05R_TEB_OCCLUDER_SHAPES
    )
    if normalized_family_weights != SOP05R_TEB_OCCLUDER_FAMILY_WEIGHTS:
        raise Sop05rConfigError(
            "template.family_weights must equal rectangle=0.4, l_shape=0.4, circle=0.2"
        )
    relative_yaw_abs_range_deg = _ordered_range(
        node["relative_yaw_abs_range_deg"],
        name="template.relative_yaw_abs_range_deg",
        positive=True,
    )
    if relative_yaw_abs_range_deg != (15.0, 45.0):
        raise Sop05rConfigError(
            "template.relative_yaw_abs_range_deg must equal [15.0, 45.0]"
        )
    return TebTemplateConfig(
        version=SOP05R_TEB_TEMPLATE_VERSION,
        occluders=_normalize_teb_occluders(node["occluders"]),
        family_weights=normalized_family_weights,
        relative_yaw_abs_range_deg=relative_yaw_abs_range_deg,
        goal_bearings_deg=bearings,
        goal_distances_m=_teb_axis(
            node["goal_distances_m"],
            name="template.goal_distances_m",
            minimum_length=1,
            positive=True,
        ),
    )


def _normalize_teb_planner(value: object) -> LightweightTebConfig:
    node = _strict_mapping(
        value,
        name="planner",
        expected_keys={
            "version",
            "band_node_count",
            "max_iterations",
            "initialization_ids",
            "initial_band_dt_s",
            "band_dt_range_s",
            "maximum_route_time_s",
            "route_sample_dt_s",
            "goal_position_tolerance_m",
            "goal_yaw_tolerance_rad",
            "max_linear_acceleration_mps2",
            "max_angular_acceleration_radps2",
            "max_curvature_per_m",
            "represented_occluder_clearance_range_m",
            "bypass_tracking_allowance_m",
            "weights",
        },
    )
    if node["version"] != SOP05R_TEB_PLANNER_VERSION:
        raise Sop05rConfigError(
            f"planner.version must equal {SOP05R_TEB_PLANNER_VERSION}"
        )
    weights = _strict_mapping(
        node["weights"],
        name="planner.weights",
        expected_keys=set(SOP05R_TEB_WEIGHT_NAMES),
    )
    normalized_weights = tuple(
        (name, _positive_real(weights[name], name=f"planner.weights.{name}"))
        for name in SOP05R_TEB_WEIGHT_NAMES
    )
    band_node_count = _integer(
        node["band_node_count"], name="planner.band_node_count", minimum=3
    )
    if band_node_count != 20:
        raise Sop05rConfigError("planner.band_node_count must equal 20")
    interval_bounds = _ordered_range(
        node["band_dt_range_s"],
        name="planner.band_dt_range_s",
        positive=True,
    )
    if interval_bounds != (0.1, 0.4):
        raise Sop05rConfigError(
            "planner.band_dt_range_s must equal [0.1, 0.4]"
        )
    initial_band_dt_s = _positive_real(
        node["initial_band_dt_s"], name="planner.initial_band_dt_s"
    )
    if initial_band_dt_s != 0.25:
        raise Sop05rConfigError("planner.initial_band_dt_s must equal 0.25")
    if not interval_bounds[0] <= initial_band_dt_s <= interval_bounds[1]:
        raise Sop05rConfigError(
            "planner.initial_band_dt_s must lie within planner.band_dt_range_s"
        )
    maximum_route_time_s = _positive_real(
        node["maximum_route_time_s"], name="planner.maximum_route_time_s"
    )
    if maximum_route_time_s != 5.0:
        raise Sop05rConfigError("planner.maximum_route_time_s must equal 5.0")
    interval_count = band_node_count - 1
    if interval_count * interval_bounds[1] < maximum_route_time_s:
        raise Sop05rConfigError(
            "planner.maximum_route_time_s exceeds maximum interval support"
        )
    if interval_count * initial_band_dt_s > maximum_route_time_s:
        raise Sop05rConfigError(
            "planner.initial_band_dt_s produces an initial route beyond maximum_route_time_s"
        )
    route_sample_dt_s = _positive_real(
        node["route_sample_dt_s"], name="planner.route_sample_dt_s"
    )
    if route_sample_dt_s != 0.2:
        raise Sop05rConfigError("planner.route_sample_dt_s must equal 0.2")
    sample_count = maximum_route_time_s / route_sample_dt_s
    if not np.isclose(sample_count, round(sample_count), rtol=0.0, atol=1e-12):
        raise Sop05rConfigError(
            "planner.maximum_route_time_s / planner.route_sample_dt_s must be integral"
        )
    if int(round(sample_count)) != 25:
        raise Sop05rConfigError(
            "planner route sample grid must contain exactly 25 future endpoints"
        )
    bypass_tracking_allowance_m = _positive_real(
        node["bypass_tracking_allowance_m"],
        name="planner.bypass_tracking_allowance_m",
    )
    if bypass_tracking_allowance_m != 0.08:
        raise Sop05rConfigError(
            "planner.bypass_tracking_allowance_m must equal 0.08"
        )
    return LightweightTebConfig(
        version=SOP05R_TEB_PLANNER_VERSION,
        band_node_count=band_node_count,
        max_iterations=_integer(
            node["max_iterations"], name="planner.max_iterations"
        ),
        initialization_ids=_exact_string_sequence(
            node["initialization_ids"],
            name="planner.initialization_ids",
            expected=SOP05R_TEB_INITIALIZATION_IDS,
        ),
        initial_band_dt_s=initial_band_dt_s,
        band_dt_range_s=interval_bounds,
        maximum_route_time_s=maximum_route_time_s,
        route_sample_dt_s=route_sample_dt_s,
        goal_position_tolerance_m=_positive_real(
            node["goal_position_tolerance_m"],
            name="planner.goal_position_tolerance_m",
        ),
        goal_yaw_tolerance_rad=_positive_real(
            node["goal_yaw_tolerance_rad"],
            name="planner.goal_yaw_tolerance_rad",
        ),
        max_linear_acceleration_mps2=_positive_real(
            node["max_linear_acceleration_mps2"],
            name="planner.max_linear_acceleration_mps2",
        ),
        max_angular_acceleration_radps2=_positive_real(
            node["max_angular_acceleration_radps2"],
            name="planner.max_angular_acceleration_radps2",
        ),
        max_curvature_per_m=_positive_real(
            node["max_curvature_per_m"], name="planner.max_curvature_per_m"
        ),
        represented_occluder_clearance_range_m=_ordered_range(
            node["represented_occluder_clearance_range_m"],
            name="planner.represented_occluder_clearance_range_m",
            positive=True,
        ),
        bypass_tracking_allowance_m=bypass_tracking_allowance_m,
        weights=normalized_weights,
    )


def _normalize_teb_placement(value: object) -> AnchoredPlacementConfig:
    node = _strict_mapping(
        value,
        name="placement",
        expected_keys={
            "version",
            "temporal_scales",
            "spatial_scale",
            "coarse_rotation_step_deg",
            "refinement_radius_deg",
            "refinement_step_deg",
            "refined_candidate_count",
            "internal_snippet_anchor_margin_frames",
        },
    )
    if node["version"] != SOP05R_TEB_PLACEMENT_VERSION:
        raise Sop05rConfigError(
            f"placement.version must equal {SOP05R_TEB_PLACEMENT_VERSION}"
        )
    spatial_scale = _finite_real(
        node["spatial_scale"], name="placement.spatial_scale"
    )
    if spatial_scale != 1.0:
        raise Sop05rConfigError("placement.spatial_scale must equal 1.0")
    coarse_step = _positive_real(
        node["coarse_rotation_step_deg"],
        name="placement.coarse_rotation_step_deg",
    )
    if not np.isclose(360.0 / coarse_step, round(360.0 / coarse_step)):
        raise Sop05rConfigError(
            "placement.coarse_rotation_step_deg must divide 360 exactly"
        )
    refinement_radius = _positive_real(
        node["refinement_radius_deg"], name="placement.refinement_radius_deg"
    )
    refinement_step = _positive_real(
        node["refinement_step_deg"], name="placement.refinement_step_deg"
    )
    if refinement_step > refinement_radius:
        raise Sop05rConfigError(
            "placement.refinement_step_deg must not exceed refinement_radius_deg"
        )
    return AnchoredPlacementConfig(
        version=SOP05R_TEB_PLACEMENT_VERSION,
        temporal_scales=_teb_axis(
            node["temporal_scales"],
            name="placement.temporal_scales",
            minimum_length=1,
            positive=True,
        ),
        spatial_scale=spatial_scale,
        coarse_rotation_step_deg=coarse_step,
        refinement_radius_deg=refinement_radius,
        refinement_step_deg=refinement_step,
        refined_candidate_count=_integer(
            node["refined_candidate_count"],
            name="placement.refined_candidate_count",
        ),
        internal_snippet_anchor_margin_frames=_integer(
            node["internal_snippet_anchor_margin_frames"],
            name="placement.internal_snippet_anchor_margin_frames",
        ),
    )


def _normalize_teb_occlusion(value: object) -> CenterlineOcclusionConfig:
    node = _strict_mapping(
        value,
        name="occlusion",
        expected_keys={
            "version",
            "centerline_intersection_epsilon_m",
            "initial_visible_weight",
            "initially_hidden_weight",
            "minimum_decision_to_collision_margin_s",
            "braking_margin_s",
            "replanning_margin_s",
        },
    )
    if node["version"] != SOP05R_TEB_OCCLUSION_VERSION:
        raise Sop05rConfigError(
            f"occlusion.version must equal {SOP05R_TEB_OCCLUSION_VERSION}"
        )
    initial_visible_weight = _fraction(
        node["initial_visible_weight"], name="occlusion.initial_visible_weight"
    )
    initially_hidden_weight = _fraction(
        node["initially_hidden_weight"], name="occlusion.initially_hidden_weight"
    )
    if initial_visible_weight != 0.8 or initially_hidden_weight != 0.2:
        raise Sop05rConfigError(
            "occlusion weights must equal 0.8 and 0.2"
        )
    return CenterlineOcclusionConfig(
        version=SOP05R_TEB_OCCLUSION_VERSION,
        centerline_intersection_epsilon_m=_positive_real(
            node["centerline_intersection_epsilon_m"],
            name="occlusion.centerline_intersection_epsilon_m",
        ),
        initial_visible_weight=initial_visible_weight,
        initially_hidden_weight=initially_hidden_weight,
        minimum_decision_to_collision_margin_s=_positive_real(
            node["minimum_decision_to_collision_margin_s"],
            name="occlusion.minimum_decision_to_collision_margin_s",
        ),
        braking_margin_s=_positive_real(
            node["braking_margin_s"], name="occlusion.braking_margin_s"
        ),
        replanning_margin_s=_positive_real(
            node["replanning_margin_s"], name="occlusion.replanning_margin_s"
        ),
    )


def _normalize_teb_generation(value: object) -> TebGenerationLimits:
    node = _strict_mapping(
        value,
        name="generation",
        expected_keys={
            "max_templates_per_base",
            "max_target_snippets_per_template",
            "max_route_anchor_candidates",
            "conflict_path_fraction_range",
            "conflict_time_range_s",
            "direct_corridor_intrusion_range_m",
        },
    )
    direct_corridor_intrusion_range_m = _ordered_range(
        node["direct_corridor_intrusion_range_m"],
        name="generation.direct_corridor_intrusion_range_m",
        positive=True,
    )
    if direct_corridor_intrusion_range_m != (0.05, 0.15):
        raise Sop05rConfigError(
            "generation.direct_corridor_intrusion_range_m must equal [0.05, 0.15]"
        )
    return TebGenerationLimits(
        max_templates_per_base=_integer(
            node["max_templates_per_base"],
            name="generation.max_templates_per_base",
        ),
        max_target_snippets_per_template=_integer(
            node["max_target_snippets_per_template"],
            name="generation.max_target_snippets_per_template",
        ),
        max_route_anchor_candidates=_integer(
            node["max_route_anchor_candidates"],
            name="generation.max_route_anchor_candidates",
        ),
        conflict_path_fraction_range=_ordered_range(
            node["conflict_path_fraction_range"],
            name="generation.conflict_path_fraction_range",
            unit_interval=True,
        ),
        conflict_time_range_s=_ordered_range(
            node["conflict_time_range_s"],
            name="generation.conflict_time_range_s",
            positive=True,
        ),
        direct_corridor_intrusion_range_m=direct_corridor_intrusion_range_m,
    )


def _normalize_teb_revealability(value: object) -> TebRevealabilityConfig:
    node = _strict_mapping(
        value,
        name="revealability",
        expected_keys={
            "minimum_visibility_lead_s",
            "training_min_active_fraction",
            "training_max_natural_difficult_fraction",
            "selection_filtering",
        },
    )
    active_fraction = _fraction(
        node["training_min_active_fraction"],
        name="revealability.training_min_active_fraction",
    )
    difficult_fraction = _fraction(
        node["training_max_natural_difficult_fraction"],
        name="revealability.training_max_natural_difficult_fraction",
    )
    if active_fraction != 0.7 or difficult_fraction != 0.3:
        raise Sop05rConfigError(
            "revealability training fractions must equal 0.7 and 0.3"
        )
    selection_filtering = node["selection_filtering"]
    if not isinstance(selection_filtering, bool):
        raise TypeError("revealability.selection_filtering must be boolean")
    return TebRevealabilityConfig(
        minimum_visibility_lead_s=_positive_real(
            node["minimum_visibility_lead_s"],
            name="revealability.minimum_visibility_lead_s",
        ),
        training_min_active_fraction=active_fraction,
        training_max_natural_difficult_fraction=difficult_fraction,
        selection_filtering=selection_filtering,
    )


def _normalize_teb_publication(value: object) -> TebPublicationVersions:
    expected = {
        "trajectory_collection_version": SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION,
        "run_producer_version": SOP05R_TEB_RUN_VERSION,
        "manifest_version": SOP05R_TEB_MANIFEST_VERSION,
        "summary_version": SOP05R_TEB_SUMMARY_VERSION,
        "completion_marker_version": SOP05R_TEB_COMPLETION_MARKER_VERSION,
    }
    node = _strict_mapping(
        value, name="publication", expected_keys=set(expected)
    )
    for name, expected_value in expected.items():
        if node[name] != expected_value:
            raise Sop05rConfigError(
                f"publication.{name} must equal {expected_value}"
            )
    return TebPublicationVersions(**expected)


def normalize_sop05r_teb_config(config: Mapping[str, Any]) -> Sop05rTebConfig:
    """Validate one immutable v2 lightweight-TEB generation configuration."""

    if config.get("generator_algorithm_version") != SOP05R_TEB_GENERATOR_VERSION:
        raise Sop05rConfigError(
            "generator_algorithm_version must equal "
            f"{SOP05R_TEB_GENERATOR_VERSION}; v1 configs require obstacle_first mode"
        )
    node = _strict_mapping(
        config,
        name="SOP05R lightweight TEB config",
        expected_keys={
            "schema_version",
            "generator_algorithm_version",
            "template",
            "planner",
            "placement",
            "occlusion",
            "generation",
            "revealability",
            "publication",
            "rejection_reasons",
        },
    )
    if node["schema_version"] != SCHEMA_VERSION:
        raise Sop05rConfigError(f"schema_version must equal {SCHEMA_VERSION}")
    if node["generator_algorithm_version"] != SOP05R_TEB_GENERATOR_VERSION:
        raise Sop05rConfigError(
            "generator_algorithm_version must equal "
            f"{SOP05R_TEB_GENERATOR_VERSION}"
        )
    rejection_reasons = _exact_string_sequence(
        node["rejection_reasons"],
        name="rejection_reasons",
        expected=SOP05R_TEB_REJECTION_REASONS,
    )
    template = _normalize_teb_template(node["template"])
    planner = _normalize_teb_planner(node["planner"])
    placement = _normalize_teb_placement(node["placement"])
    occlusion = _normalize_teb_occlusion(node["occlusion"])
    generation = _normalize_teb_generation(node["generation"])
    revealability = _normalize_teb_revealability(node["revealability"])
    publication = _normalize_teb_publication(node["publication"])
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "generator_algorithm_version": SOP05R_TEB_GENERATOR_VERSION,
        "template": template.as_dict(),
        "planner": planner.as_dict(),
        "placement": placement.as_dict(),
        "occlusion": occlusion.as_dict(),
        "generation": generation.as_dict(),
        "revealability": revealability.as_dict(),
        "publication": publication.as_dict(),
        "rejection_reasons": list(rejection_reasons),
    }
    return Sop05rTebConfig(
        schema_version=SCHEMA_VERSION,
        generator_algorithm_version=SOP05R_TEB_GENERATOR_VERSION,
        template=template,
        planner=planner,
        placement=placement,
        occlusion=occlusion,
        generation=generation,
        revealability=revealability,
        publication=publication,
        rejection_reasons=rejection_reasons,
        digest=hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest(),
    )


def load_sop05r_teb_config(path: str | Path) -> Sop05rTebConfig:
    """Load a standalone v2 config without v1 defaults or merging."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise Sop05rConfigError(
            f"failed to load SOP05R lightweight TEB config: {config_path}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise Sop05rConfigError("SOP05R lightweight TEB config root must be a mapping")
    return normalize_sop05r_teb_config(raw)
