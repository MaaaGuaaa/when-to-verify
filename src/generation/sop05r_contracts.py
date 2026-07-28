"""Strict immutable contracts for current Long40 SOP05R generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

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


# Current SOP05R Long40 lightweight-TEB scene-generation contract.
SOP05R_LONG40_SCHEMA_VERSION = "4.0.0"
SOP05R_LONG40_LAYOUT_VERSION = "history8_current7_future32_v1"
SOP05R_TEB_GENERATOR_VERSION = "obstacle_first_lightweight_teb_v8"
SOP05R_TEB_TEMPLATE_VERSION = "goal_occluder_template_schedule_v3"
SOP05R_TEB_PLANNER_VERSION = "lightweight_teb_planner_v3"
SOP05R_TEB_PLACEMENT_VERSION = "anchored_human_half_plane_step_long40_v8"
SOP05R_TEB_OCCLUSION_VERSION = "seen_then_occlude_prefix4_v4"
SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION = (
    "sop05r_nominal_trajectory_collection_v8"
)
SOP05R_TEB_RUN_VERSION = "sop05r_lightweight_teb_generation_run_v7"
SOP05R_TEB_MANIFEST_VERSION = "sop05r_lightweight_teb_manifest_v7"
SOP05R_TEB_SUMMARY_VERSION = "sop05r_lightweight_teb_summary_v8"
SOP05R_TEB_COMPLETION_MARKER_VERSION = (
    "sop05r_lightweight_teb_producer_complete_v7"
)
SOP05R_TEB_OCCLUDER_ANGULAR_MARGIN_STEP_DEG = 5.0

SOP05R_TEB_OCCLUDER_SHAPES = ("rectangle", "l_shape", "circle")
SOP05R_TEB_OCCLUDER_FAMILY_WEIGHTS = (
    ("rectangle", 0.4),
    ("l_shape", 0.4),
    ("circle", 0.2),
)
SOP05R_TEB_RECTANGLE_SEMANTIC_TYPES = ("wall", "shelf", "cabinet")
SOP05R_TEB_CIRCLE_SEMANTIC_TYPES = ("tree_trunk", "column")
SOP05R_TEB_INITIALIZATION_IDS = (
    "straight",
    "bypass_left",
    "bypass_right",
)
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
    "guide_ray_degenerate",
    "half_plane_margin_missing",
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
class Long40TrajectoryConfig:
    layout_version: str
    history_steps: int
    current_index: int
    future_steps: int
    future_dt_s: float
    future_horizon_s: float

    def as_dict(self) -> dict[str, object]:
        return {
            "layout_version": self.layout_version,
            "history_steps": self.history_steps,
            "current_index": self.current_index,
            "future_steps": self.future_steps,
            "future_dt_s": self.future_dt_s,
            "future_horizon_s": self.future_horizon_s,
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
    occluder_angular_margin_step_deg: float
    internal_snippet_anchor_margin_frames: int

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "temporal_scales": list(self.temporal_scales),
            "spatial_scale": self.spatial_scale,
            "occluder_angular_margin_step_deg": (
                self.occluder_angular_margin_step_deg
            ),
            "internal_snippet_anchor_margin_frames": (
                self.internal_snippet_anchor_margin_frames
            ),
        }


@dataclass(frozen=True)
class CenterlineOcclusionConfig:
    version: str
    centerline_intersection_epsilon_m: float
    minimum_visible_history_frames: int
    minimum_occluded_history_frames: int
    minimum_decision_to_collision_margin_s: float
    braking_margin_s: float
    replanning_margin_s: float

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "centerline_intersection_epsilon_m": self.centerline_intersection_epsilon_m,
            "minimum_visible_history_frames": self.minimum_visible_history_frames,
            "minimum_occluded_history_frames": self.minimum_occluded_history_frames,
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
    collision_route_path_fraction_range: tuple[float, float]
    minimum_direct_corridor_intrusion_m: float

    def as_dict(self) -> dict[str, object]:
        return {
            "max_templates_per_base": self.max_templates_per_base,
            "max_target_snippets_per_template": self.max_target_snippets_per_template,
            "max_route_anchor_candidates": self.max_route_anchor_candidates,
            "collision_route_path_fraction_range": list(
                self.collision_route_path_fraction_range
            ),
            "minimum_direct_corridor_intrusion_m": (
                self.minimum_direct_corridor_intrusion_m
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
    trajectory: Long40TrajectoryConfig
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
            "trajectory": self.trajectory.as_dict(),
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


def _normalize_long40_trajectory(value: object) -> Long40TrajectoryConfig:
    node = _strict_mapping(
        value,
        name="trajectory",
        expected_keys={
            "layout_version",
            "history_steps",
            "current_index",
            "future_steps",
            "future_dt_s",
            "future_horizon_s",
        },
    )
    if node["layout_version"] != SOP05R_LONG40_LAYOUT_VERSION:
        raise Sop05rConfigError(
            "trajectory.layout_version must equal "
            f"{SOP05R_LONG40_LAYOUT_VERSION}"
        )
    history_steps = _integer(
        node["history_steps"],
        name="trajectory.history_steps",
    )
    current_index = _integer(
        node["current_index"],
        name="trajectory.current_index",
        minimum=0,
    )
    future_steps = _integer(
        node["future_steps"],
        name="trajectory.future_steps",
    )
    future_dt_s = _positive_real(
        node["future_dt_s"],
        name="trajectory.future_dt_s",
    )
    future_horizon_s = _positive_real(
        node["future_horizon_s"],
        name="trajectory.future_horizon_s",
    )
    if history_steps != 8:
        raise Sop05rConfigError("trajectory.history_steps must equal 8")
    if current_index != 7 or current_index != history_steps - 1:
        raise Sop05rConfigError(
            "trajectory.current_index must equal 7 and terminate history"
        )
    if future_steps != 32:
        raise Sop05rConfigError("trajectory.future_steps must equal 32")
    if future_dt_s != 0.2:
        raise Sop05rConfigError("trajectory.future_dt_s must equal 0.2")
    if not np.isclose(
        future_steps * future_dt_s,
        future_horizon_s,
        rtol=0.0,
        atol=1e-12,
    ):
        raise Sop05rConfigError(
            "trajectory.future_steps * trajectory.future_dt_s must equal "
            "trajectory.future_horizon_s"
        )
    if future_horizon_s != 6.4:
        raise Sop05rConfigError("trajectory.future_horizon_s must equal 6.4")
    return Long40TrajectoryConfig(
        layout_version=SOP05R_LONG40_LAYOUT_VERSION,
        history_steps=history_steps,
        current_index=current_index,
        future_steps=future_steps,
        future_dt_s=future_dt_s,
        future_horizon_s=future_horizon_s,
    )


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
    goal_distances_m = _teb_axis(
        node["goal_distances_m"],
        name="template.goal_distances_m",
        minimum_length=1,
        positive=True,
    )
    if goal_distances_m != (4.0, 4.5):
        raise Sop05rConfigError(
            "template.goal_distances_m must equal [4.0, 4.5]"
        )
    return TebTemplateConfig(
        version=SOP05R_TEB_TEMPLATE_VERSION,
        occluders=_normalize_teb_occluders(node["occluders"]),
        family_weights=normalized_family_weights,
        relative_yaw_abs_range_deg=relative_yaw_abs_range_deg,
        goal_bearings_deg=bearings,
        goal_distances_m=goal_distances_m,
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
    if band_node_count != 21:
        raise Sop05rConfigError("planner.band_node_count must equal 21")
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
    if maximum_route_time_s != 8.0:
        raise Sop05rConfigError("planner.maximum_route_time_s must equal 8.0")
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
    if int(round(sample_count)) != 40:
        raise Sop05rConfigError(
            "planner route sample grid must contain exactly 40 future endpoints"
        )
    bypass_tracking_allowance_m = _positive_real(
        node["bypass_tracking_allowance_m"],
        name="planner.bypass_tracking_allowance_m",
    )
    if bypass_tracking_allowance_m != 0.08:
        raise Sop05rConfigError(
            "planner.bypass_tracking_allowance_m must equal 0.08"
        )
    represented_clearance_range_m = _ordered_range(
        node["represented_occluder_clearance_range_m"],
        name="planner.represented_occluder_clearance_range_m",
        positive=True,
    )
    if represented_clearance_range_m != (0.15, 0.75):
        raise Sop05rConfigError(
            "planner.represented_occluder_clearance_range_m must equal "
            "[0.15, 0.75]"
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
        represented_occluder_clearance_range_m=represented_clearance_range_m,
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
            "occluder_angular_margin_step_deg",
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
    angular_margin_step_deg = _positive_real(
        node["occluder_angular_margin_step_deg"],
        name="placement.occluder_angular_margin_step_deg",
    )
    if angular_margin_step_deg != SOP05R_TEB_OCCLUDER_ANGULAR_MARGIN_STEP_DEG:
        raise Sop05rConfigError(
            "placement.occluder_angular_margin_step_deg must equal 5.0"
        )
    anchor_margin_frames = _integer(
        node["internal_snippet_anchor_margin_frames"],
        name="placement.internal_snippet_anchor_margin_frames",
        minimum=0,
    )
    if anchor_margin_frames != 0:
        raise Sop05rConfigError(
            "placement.internal_snippet_anchor_margin_frames must equal 0"
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
        occluder_angular_margin_step_deg=angular_margin_step_deg,
        internal_snippet_anchor_margin_frames=anchor_margin_frames,
    )


def _normalize_teb_occlusion(value: object) -> CenterlineOcclusionConfig:
    node = _strict_mapping(
        value,
        name="occlusion",
        expected_keys={
            "version",
            "centerline_intersection_epsilon_m",
            "minimum_visible_history_frames",
            "minimum_occluded_history_frames",
            "minimum_decision_to_collision_margin_s",
            "braking_margin_s",
            "replanning_margin_s",
        },
    )
    if node["version"] != SOP05R_TEB_OCCLUSION_VERSION:
        raise Sop05rConfigError(
            f"occlusion.version must equal {SOP05R_TEB_OCCLUSION_VERSION}"
        )
    minimum_visible_history_frames = _integer(
        node["minimum_visible_history_frames"],
        name="occlusion.minimum_visible_history_frames",
        minimum=1,
    )
    if minimum_visible_history_frames != 4:
        raise Sop05rConfigError(
            "occlusion.minimum_visible_history_frames must equal 4"
        )
    minimum_occluded_history_frames = _integer(
        node["minimum_occluded_history_frames"],
        name="occlusion.minimum_occluded_history_frames",
        minimum=1,
    )
    if minimum_occluded_history_frames != 1:
        raise Sop05rConfigError(
            "occlusion.minimum_occluded_history_frames must equal 1"
        )
    minimum_decision_to_collision_margin_s = _positive_real(
        node["minimum_decision_to_collision_margin_s"],
        name="occlusion.minimum_decision_to_collision_margin_s",
    )
    if minimum_decision_to_collision_margin_s != 1.2:
        raise Sop05rConfigError(
            "occlusion.minimum_decision_to_collision_margin_s must equal 1.2"
        )
    return CenterlineOcclusionConfig(
        version=SOP05R_TEB_OCCLUSION_VERSION,
        centerline_intersection_epsilon_m=_positive_real(
            node["centerline_intersection_epsilon_m"],
            name="occlusion.centerline_intersection_epsilon_m",
        ),
        minimum_visible_history_frames=minimum_visible_history_frames,
        minimum_occluded_history_frames=minimum_occluded_history_frames,
        minimum_decision_to_collision_margin_s=minimum_decision_to_collision_margin_s,
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
            "collision_route_path_fraction_range",
            "minimum_direct_corridor_intrusion_m",
        },
    )
    minimum_direct_corridor_intrusion_m = _positive_real(
        node["minimum_direct_corridor_intrusion_m"],
        name="generation.minimum_direct_corridor_intrusion_m",
    )
    if minimum_direct_corridor_intrusion_m != 0.15:
        raise Sop05rConfigError(
            "generation.minimum_direct_corridor_intrusion_m must equal 0.15"
        )
    collision_route_path_fraction_range = _ordered_range(
        node["collision_route_path_fraction_range"],
        name="generation.collision_route_path_fraction_range",
        unit_interval=True,
    )
    if collision_route_path_fraction_range != (0.2, 0.95):
        raise Sop05rConfigError(
            "generation.collision_route_path_fraction_range must equal "
            "[0.20, 0.95]"
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
        collision_route_path_fraction_range=(
            collision_route_path_fraction_range
        ),
        minimum_direct_corridor_intrusion_m=minimum_direct_corridor_intrusion_m,
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
    """Validate one immutable current Long40 lightweight-TEB configuration."""

    generator_version = config.get("generator_algorithm_version")
    if generator_version != SOP05R_TEB_GENERATOR_VERSION:
        raise Sop05rConfigError(
            "generator_algorithm_version must equal "
            f"{SOP05R_TEB_GENERATOR_VERSION} for obstacle_first_teb mode"
        )
    node = _strict_mapping(
        config,
        name="SOP05R lightweight TEB config",
        expected_keys={
            "schema_version",
            "generator_algorithm_version",
            "trajectory",
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
    if node["schema_version"] != SOP05R_LONG40_SCHEMA_VERSION:
        raise Sop05rConfigError(
            "schema_version must equal "
            f"{SOP05R_LONG40_SCHEMA_VERSION} for obstacle_first_teb"
        )
    if node["generator_algorithm_version"] != SOP05R_TEB_GENERATOR_VERSION:
        raise Sop05rConfigError(
            "generator_algorithm_version must equal "
            f"{SOP05R_TEB_GENERATOR_VERSION}"
        )
    trajectory = _normalize_long40_trajectory(node["trajectory"])
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
        "schema_version": SOP05R_LONG40_SCHEMA_VERSION,
        "generator_algorithm_version": SOP05R_TEB_GENERATOR_VERSION,
        "trajectory": trajectory.as_dict(),
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
        schema_version=SOP05R_LONG40_SCHEMA_VERSION,
        generator_algorithm_version=SOP05R_TEB_GENERATOR_VERSION,
        trajectory=trajectory,
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
    """Load a standalone current Long40 config without defaults or merging."""

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
