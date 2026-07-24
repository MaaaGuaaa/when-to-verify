"""Configured SOP-06 paired counterfactual variants built from one mother event."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from src.contracts import (
    SCHEMA_VERSION,
    BaseState,
    LocalTrajectory,
    OracleContext,
    OracleWorld,
    build_grid_spec,
    validate_oracle_context,
    validate_oracle_world,
)
from src.datasets.snippet_library import MotionSnippet, SnippetLibrary
from src.geometry import (
    Footprint,
    RectangleFootprint,
    inflate_footprint,
    intersects,
    points_in_grid,
    rasterize_footprint,
    raycast_visibility,
    signed_clearance,
    trajectory_signed_clearances,
    wrap_angle,
)
from src.utils.seeding import derive_seed, stable_digest

from .blind_reachability import BLIND_REACHABILITY_ALGORITHM_VERSION
from .dynamic_object_transplant import (
    TransplantError,
    TransplantedDynamicObject,
    footprint_from_spec,
    transplant_snippet,
)
from .event_sampler import (
    GeneratedEvent,
    SOP05_GENERATOR_ALGORITHM_VERSION,
    generate_events,
    normalize_generator_config,
)
from .event_target_motion_shard import (
    build_event_target_motion_world_metadata,
    compute_motion_array_digest,
    create_event_target_motion_record,
    validate_event_target_motion_world_join,
)
from .history_visibility import (
    HISTORY_VISIBILITY_POLICY_VERSION,
    HISTORY_VISIBILITY_REGIMES,
    HistoryVisibilityAssessment,
    HistoryVisibilityPolicy,
    classify_history_visibility,
    normalize_history_visibility_policy,
)
from .occluder_sampler import (
    JOINT_MULTI_LOS_PLACEMENT_STRATEGY_VERSION,
    OccluderCollisionSweep,
    OccluderSamplingError,
    align_environment_occluder_to_target_los_envelope,
    occluder_collision_sweep_rejection_reason,
    synchronized_sweeps_intersect,
    swept_footprint_intersects_occupancy,
)
from .structural_blindspot import (
    StructuralBlindSpot,
    build_structural_visibility,
    footprint_visibility_sequence,
    has_continuous_emergence,
)


VARIANT_ORDER = (
    "collision",
    "near_miss",
    "temporal_safe",
    "spatial_safe",
    "irrelevant_hidden",
    "empty_blind_spot",
)
_TEMPORAL_OFFSETS = (0.8, -0.8, 1.0, -1.0, 1.2, -1.2, 1.5, -1.5)
_MOTHER_REQUIRED_VARIANTS = ("collision",)
PAIRED_GENERATOR_ALGORITHM_VERSION = "independent_partial_pairs_v2"
PAIRED_GROUP_CONTRACT_VERSION = "sop06_partial_pair_group_v2"
# Historical identity retained only by the explicitly named legacy joint
# search function.  Formal v5 producers/consumers use the two constants above.
JOINT_ENVIRONMENT_PAIR_VERSION = "joint_environment_pair_v2"
_JOINT_ENVIRONMENT_MOTHER_BATCH_SIZE = 16
_SOP05_JOIN_METADATA_KEYS = frozenset(
    {
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
)


class PairedVariantConfigError(ValueError):
    """Raised when the standalone SOP-06 paired config drifts."""


class PairGenerationError(ValueError):
    """Raised when a mother event cannot satisfy the paired-event contract."""

    def __init__(self, reason: str, message: str | None = None):
        super().__init__(message or reason)
        self.reason = reason


class _CandidateRejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _mother_history_visibility_contract(
    mother_event: GeneratedEvent,
) -> tuple[HistoryVisibilityPolicy, HistoryVisibilityAssessment]:
    metadata = mother_event.world.metadata
    try:
        policy = normalize_history_visibility_policy(
            metadata.get("target_history_visibility_policy")
        )
        assessment = classify_history_visibility(
            mother_event.target_visibility_history,
            policy,
        )
    except (TypeError, ValueError) as exc:
        raise PairGenerationError(
            "mother_history_visibility_policy_invalid",
            f"mother history visibility contract is invalid: {exc}",
        ) from exc
    if metadata.get("target_history_visibility_policy_digest") != policy.digest:
        raise PairGenerationError(
            "mother_history_visibility_policy_digest_mismatch"
        )
    if assessment.regime not in HISTORY_VISIBILITY_REGIMES:
        raise PairGenerationError("mother_history_visibility_ineligible")
    expected = {
        "target_history_visibility_regime": assessment.regime,
        "target_history_last_visible_index": assessment.last_visible_index,
        "target_history_trailing_hidden_frames": (
            assessment.trailing_hidden_frames
        ),
        "target_history_visibility_policy_version": (
            HISTORY_VISIBILITY_POLICY_VERSION
        ),
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise PairGenerationError(
            "mother_history_visibility_metadata_mismatch"
        )
    return policy, assessment


@dataclass(frozen=True)
class PairedVariantConfig:
    schema_version: str
    paired_generator_algorithm_version: str
    group_contract_version: str
    near_miss_clearance_range_m: tuple[float, float]
    temporal_offset_candidates_s: tuple[float, ...]
    spatial_safe_clearance_range_m: tuple[float, float]
    irrelevant_min_clearance_m: float
    lateral_offset_step_m: float
    lateral_offset_max_m: float
    mother_required_variants: tuple[str, ...]
    training_minimum_contrast_count: int
    audit_requires_all_variants: bool
    digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "paired_generator_algorithm_version": (
                self.paired_generator_algorithm_version
            ),
            "group_contract_version": self.group_contract_version,
            "near_miss_clearance_range_m": list(
                self.near_miss_clearance_range_m
            ),
            "temporal_offset_candidates_s": list(
                self.temporal_offset_candidates_s
            ),
            "spatial_safe_clearance_range_m": list(
                self.spatial_safe_clearance_range_m
            ),
            "irrelevant_min_clearance_m": self.irrelevant_min_clearance_m,
            "lateral_offset_step_m": self.lateral_offset_step_m,
            "lateral_offset_max_m": self.lateral_offset_max_m,
            "mother_required_variants": list(self.mother_required_variants),
            "training_minimum_contrast_count": (
                self.training_minimum_contrast_count
            ),
            "audit_requires_all_variants": self.audit_requires_all_variants,
        }


@dataclass(frozen=True)
class PairedVariant:
    """One auditable world in a paired counterfactual group."""

    variant_kind: str
    world: OracleWorld
    target: TransplantedDynamicObject | None
    target_visibility_history: np.ndarray | None
    visibility_sequence: np.ndarray | None
    clearance_sequence_m: np.ndarray | None
    min_clearance_m: float | None
    time_to_min_clearance_s: float | None
    temporal_offset_s: float | None = None
    lateral_offset_m: float | None = None
    radial_shift_m: float | None = None
    rotation_rad: float | None = None


@dataclass(frozen=True)
class PairedEventGroup:
    """A complete or policy-compliant partial six-position paired group."""

    pair_group_id: str
    variants: tuple[PairedVariant, ...]
    coverage_mask: tuple[bool, ...]
    missing_variant_reasons: dict[str, str]
    is_complete: bool
    eligible_for_strict_evaluation: bool
    paired_config_digest: str

    @property
    def by_kind(self) -> dict[str, PairedVariant]:
        return {variant.variant_kind: variant for variant in self.variants}


@dataclass(frozen=True)
class JointEnvironmentPairReport:
    """One bounded environment-mother search and its complete paired group."""

    mother_event: GeneratedEvent | None
    group: PairedEventGroup | None
    summary: dict[str, object]


@dataclass(frozen=True)
class _PairEnvironment:
    grid: object
    future_dt_s: float
    robot_footprint: Footprint
    target_footprint: Footprint
    context_footprints: dict[str, Footprint]
    scene_history_footprints: dict[str, Footprint]
    scene_dynamic_history: dict[str, np.ndarray]
    base_static_occupancy: np.ndarray
    occluder_geometry: tuple[tuple[RectangleFootprint, np.ndarray], ...]
    visibility_history: np.ndarray
    visibility_mask: np.ndarray


@dataclass(frozen=True)
class _GeometricCandidatePool:
    pivot_radius_m: float
    ray_direction: np.ndarray
    candidates: tuple[tuple[float, float, float, int], ...]


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise PairedVariantConfigError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise PairedVariantConfigError(f"{name} must be finite")
    return result


def _range_pair(value: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise PairedVariantConfigError(f"{name} must contain [minimum, maximum]")
    lower = _finite_real(value[0], name=f"{name}[0]")
    upper = _finite_real(value[1], name=f"{name}[1]")
    if lower < 0.0 or lower > upper:
        raise PairedVariantConfigError(f"{name} must be a non-negative range")
    return lower, upper


def normalize_paired_variant_config(
    config: Mapping[str, Any],
) -> PairedVariantConfig:
    """Strictly normalize the standalone SOP-06 paired-variant config."""

    if not isinstance(config, Mapping):
        raise PairedVariantConfigError("paired config must be a mapping")
    expected = {
        "schema_version",
        "paired_generator_algorithm_version",
        "group_contract_version",
        "near_miss_clearance_range_m",
        "temporal_offset_candidates_s",
        "spatial_safe_clearance_range_m",
        "irrelevant_min_clearance_m",
        "lateral_offset_step_m",
        "lateral_offset_max_m",
        "mother_required_variants",
        "training_minimum_contrast_count",
        "audit_requires_all_variants",
    }
    if set(config) != expected:
        unknown = sorted(set(config) - expected)
        missing = sorted(expected - set(config))
        raise PairedVariantConfigError(
            f"paired config keys mismatch; unknown={unknown}, missing={missing}"
        )
    if config["schema_version"] != SCHEMA_VERSION:
        raise PairedVariantConfigError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    if (
        config["paired_generator_algorithm_version"]
        != PAIRED_GENERATOR_ALGORITHM_VERSION
    ):
        raise PairedVariantConfigError(
            "paired_generator_algorithm_version must equal "
            f"{PAIRED_GENERATOR_ALGORITHM_VERSION}"
        )
    if config["group_contract_version"] != PAIRED_GROUP_CONTRACT_VERSION:
        raise PairedVariantConfigError(
            "group_contract_version must equal "
            f"{PAIRED_GROUP_CONTRACT_VERSION}"
        )
    near_miss = _range_pair(
        config["near_miss_clearance_range_m"],
        name="near_miss_clearance_range_m",
    )
    spatial_safe = _range_pair(
        config["spatial_safe_clearance_range_m"],
        name="spatial_safe_clearance_range_m",
    )
    if near_miss[1] >= spatial_safe[0]:
        raise PairedVariantConfigError(
            "near-miss upper bound must be below spatial-safe lower bound"
        )
    temporal = tuple(
        _finite_real(value, name="temporal_offset_candidates_s")
        for value in config["temporal_offset_candidates_s"]
    )
    if temporal != _TEMPORAL_OFFSETS:
        raise PairedVariantConfigError(
            "temporal_offset_candidates_s must match the frozen SOP-06 order"
        )
    irrelevant = _finite_real(
        config["irrelevant_min_clearance_m"],
        name="irrelevant_min_clearance_m",
    )
    if irrelevant <= spatial_safe[1]:
        raise PairedVariantConfigError(
            "irrelevant minimum must be greater than spatial-safe upper bound"
        )
    step = _finite_real(
        config["lateral_offset_step_m"], name="lateral_offset_step_m"
    )
    maximum = _finite_real(
        config["lateral_offset_max_m"], name="lateral_offset_max_m"
    )
    if step <= 0.0 or maximum <= irrelevant or step > maximum:
        raise PairedVariantConfigError(
            "lateral offset step/maximum do not cover the irrelevant threshold"
        )
    required = tuple(config["mother_required_variants"])
    if required != _MOTHER_REQUIRED_VARIANTS:
        raise PairedVariantConfigError(
            "mother_required_variants must equal [collision]"
        )
    count = config["training_minimum_contrast_count"]
    if isinstance(count, (bool, np.bool_)) or not isinstance(
        count, (int, np.integer)
    ):
        raise PairedVariantConfigError(
            "training_minimum_contrast_count must be an integer"
        )
    count = int(count)
    if count != 0:
        raise PairedVariantConfigError(
            "training_minimum_contrast_count must equal zero"
        )
    complete = config["audit_requires_all_variants"]
    if not isinstance(complete, (bool, np.bool_)) or not bool(complete):
        raise PairedVariantConfigError(
            "audit_requires_all_variants must be true"
        )
    normalized_payload = {
        "schema_version": SCHEMA_VERSION,
        "paired_generator_algorithm_version": (
            PAIRED_GENERATOR_ALGORITHM_VERSION
        ),
        "group_contract_version": PAIRED_GROUP_CONTRACT_VERSION,
        "near_miss_clearance_range_m": list(near_miss),
        "temporal_offset_candidates_s": list(temporal),
        "spatial_safe_clearance_range_m": list(spatial_safe),
        "irrelevant_min_clearance_m": irrelevant,
        "lateral_offset_step_m": step,
        "lateral_offset_max_m": maximum,
        "mother_required_variants": list(required),
        "training_minimum_contrast_count": count,
        "audit_requires_all_variants": True,
    }
    payload = json.dumps(
        normalized_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return PairedVariantConfig(
        schema_version=SCHEMA_VERSION,
        paired_generator_algorithm_version=(
            PAIRED_GENERATOR_ALGORITHM_VERSION
        ),
        group_contract_version=PAIRED_GROUP_CONTRACT_VERSION,
        near_miss_clearance_range_m=near_miss,
        temporal_offset_candidates_s=temporal,
        spatial_safe_clearance_range_m=spatial_safe,
        irrelevant_min_clearance_m=irrelevant,
        lateral_offset_step_m=step,
        lateral_offset_max_m=maximum,
        mother_required_variants=required,
        training_minimum_contrast_count=count,
        audit_requires_all_variants=True,
        digest=stable_digest(payload, size=16),
    )


def load_paired_variant_config(path: str | Path) -> PairedVariantConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise PairedVariantConfigError("paired config top level must be a mapping")
    return normalize_paired_variant_config(raw)


def _as_paired_config(
    config: PairedVariantConfig | Mapping[str, Any],
) -> PairedVariantConfig:
    if isinstance(config, PairedVariantConfig):
        return config
    return normalize_paired_variant_config(config)


def _robot_footprint(base_config: Mapping[str, Any]) -> Footprint:
    robot = base_config["robot"]
    return inflate_footprint(
        RectangleFootprint(robot["length_m"], robot["width_m"]),
        robot["inflation_m"],
    )


def _structural_blind_spot(world: OracleWorld) -> StructuralBlindSpot | None:
    raw = world.blind_spot_config.get("structural")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != {
        "forward_fov_deg",
        "range_m",
        "blind_sectors",
    }:
        raise PairGenerationError("mother_blind_spot_invalid")
    return StructuralBlindSpot(
        forward_fov_deg=raw["forward_fov_deg"],
        range_m=raw["range_m"],
        blind_sectors=tuple(dict(sector) for sector in raw["blind_sectors"]),
    )


def _occluder_geometry(
    world: OracleWorld,
) -> tuple[tuple[RectangleFootprint, np.ndarray], ...]:
    result = []
    for occluder in world.occluders:
        required = {"pose", "length_m", "width_m"}
        if not isinstance(occluder, Mapping) or not required <= set(occluder):
            raise PairGenerationError("mother_occluder_invalid")
        pose = np.asarray(occluder["pose"], dtype=np.float64)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise PairGenerationError("mother_occluder_invalid")
        result.append(
            (
                RectangleFootprint(
                    float(occluder["length_m"]),
                    float(occluder["width_m"]),
                ),
                pose,
            )
        )
    return tuple(result)


def _context_current_occupancy(
    oracle_context: OracleContext,
    *,
    context_footprints: Mapping[str, Footprint],
    grid,
) -> np.ndarray:
    result = np.zeros((grid.height, grid.width), dtype=bool)
    for object_id in sorted(context_footprints):
        result |= rasterize_footprint(
            context_footprints[object_id],
            oracle_context.dynamic_object_history[object_id][-1],
            grid,
        )
    return result


def _pair_environment(
    *,
    mother_event: GeneratedEvent,
    trajectory: LocalTrajectory,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    critical_clearance_threshold_m: float,
) -> _PairEnvironment:
    grid = build_grid_spec(dict(base_config))
    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if base_state.state_id != mother_event.world.base_state_id:
        raise PairGenerationError("mother_base_state_id_mismatch")
    validate_oracle_context(oracle_context, grid)
    validate_oracle_world(mother_event.world, grid)
    if mother_event.world.base_state_id != oracle_context.base_state_id:
        raise PairGenerationError("mother_context_id_mismatch")
    if mother_event.world.metadata.get("trajectory_id") != trajectory.trajectory_id:
        raise PairGenerationError("mother_trajectory_id_mismatch")
    try:
        validate_event_target_motion_world_join(
            mother_event.target_motion_record,
            mother_event.world,
            grid,
        )
    except (TypeError, ValueError) as exc:
        raise PairGenerationError(
            "mother_target_motion_join_invalid", str(exc)
        ) from exc
    target_id = mother_event.target.target_dynamic_object_id
    target = mother_event.target
    record = mother_event.target_motion_record
    if (
        record.generated_event_id != mother_event.generated_event_id
        or record.target_dynamic_object_id != target_id
        or record.source_snippet_id != target.snippet_id
        or record.source_object_id != target.source_object_id
        or record.object_type != target.object_type
        or record.footprint_spec != target.footprint_spec
        or record.footprint_spec_digest != target.footprint_spec_digest
        or record.target_type_policy_digest
        != target.provenance.get("target_type_policy_digest")
        or not np.array_equal(record.history_poses, target.history_poses)
        or not np.array_equal(record.current_pose, target.current_pose)
        or not np.array_equal(record.future_poses, target.future_poses)
    ):
        raise PairGenerationError("mother_target_motion_record_mismatch")
    context_ids = set(oracle_context.dynamic_object_future)
    if target_id in context_ids:
        raise PairGenerationError("mother_target_context_id_collision")
    expected_world_ids = context_ids | {target_id}
    if set(mother_event.world.dynamic_object_trajectories) != expected_world_ids:
        raise PairGenerationError("mother_dynamic_object_ids_mismatch")
    if not np.array_equal(
        mother_event.world.dynamic_object_trajectories[target_id],
        target.future_poses,
    ):
        raise PairGenerationError("mother_target_changed")
    if mother_event.world.dynamic_object_specs[target_id] != target.footprint_spec:
        raise PairGenerationError("mother_target_changed")
    for object_id in sorted(context_ids):
        if not np.array_equal(
            mother_event.world.dynamic_object_trajectories[object_id],
            oracle_context.dynamic_object_future[object_id],
        ):
            raise PairGenerationError("mother_context_changed")
        if (
            mother_event.world.dynamic_object_specs[object_id]
            != oracle_context.dynamic_object_specs[object_id]
        ):
            raise PairGenerationError("mother_context_changed")

    base_ids = set(base_state.dynamic_object_ids)
    if (
        base_ids != set(base_state.visible_dynamic_object_history)
        or base_ids != set(base_state.visible_dynamic_object_specs)
    ):
        raise PairGenerationError("base_dynamic_object_contract_invalid")
    oracle_ids = set(oracle_context.dynamic_object_history)
    for object_id in sorted(base_ids & oracle_ids):
        if not np.array_equal(
            base_state.visible_dynamic_object_history[object_id],
            oracle_context.dynamic_object_history[object_id],
        ):
            raise PairGenerationError("base_oracle_history_mismatch")
        if (
            base_state.visible_dynamic_object_specs[object_id]
            != oracle_context.dynamic_object_specs[object_id]
        ):
            raise PairGenerationError("base_oracle_spec_mismatch")
    if target_id in base_ids:
        raise PairGenerationError("mother_target_base_id_collision")

    context_footprints = {
        object_id: footprint_from_spec(oracle_context.dynamic_object_specs[object_id])
        for object_id in sorted(oracle_context.dynamic_object_specs)
    }
    scene_dynamic_history = {
        object_id: oracle_context.dynamic_object_history[object_id]
        for object_id in sorted(oracle_ids)
    }
    scene_history_footprints = dict(context_footprints)
    for object_id in sorted(base_ids - oracle_ids):
        scene_dynamic_history[object_id] = (
            base_state.visible_dynamic_object_history[object_id]
        )
        scene_history_footprints[object_id] = footprint_from_spec(
            base_state.visible_dynamic_object_specs[object_id]
        )
    robot_footprint = _robot_footprint(base_config)
    for object_id in sorted(context_footprints):
        clearances = trajectory_signed_clearances(
            robot_footprint,
            trajectory.poses,
            context_footprints[object_id],
            oracle_context.dynamic_object_future[object_id],
        )
        if float(np.min(clearances)) <= critical_clearance_threshold_m:
            raise PairGenerationError("multi_object_context")

    occluders = _occluder_geometry(mother_event.world)
    base_static = np.asarray(
        mother_event.world.static_occupancy != 0, dtype=bool
    ).copy()
    for footprint, pose in occluders:
        base_static &= ~rasterize_footprint(footprint, pose, grid)
    sensor_static_occupancy = np.asarray(
        mother_event.world.static_occupancy != 0, dtype=bool
    )
    structural = _structural_blind_spot(mother_event.world)
    visibility_history = np.empty(
        (grid.history_steps, grid.height, grid.width),
        dtype=bool,
        order="C",
    )
    for history_index, robot_pose in enumerate(base_state.robot_history):
        occupied = sensor_static_occupancy.copy()
        for object_id in sorted(scene_history_footprints):
            occupied |= rasterize_footprint(
                scene_history_footprints[object_id],
                scene_dynamic_history[object_id][history_index],
                grid,
            )
        if structural is None:
            visibility = raycast_visibility(
                occupied, grid, sensor_pose=robot_pose
            )
        else:
            visibility = build_structural_visibility(
                occupied,
                grid,
                sensor_pose=robot_pose,
                blind_spot=structural,
            )
        visibility_history[history_index] = visibility
    return _PairEnvironment(
        grid=grid,
        future_dt_s=float(base_config["bev"]["future_dt_s"]),
        robot_footprint=robot_footprint,
        target_footprint=footprint_from_spec(mother_event.target.footprint_spec),
        context_footprints=context_footprints,
        scene_history_footprints=scene_history_footprints,
        scene_dynamic_history=scene_dynamic_history,
        base_static_occupancy=base_static,
        occluder_geometry=occluders,
        visibility_history=visibility_history,
        visibility_mask=visibility_history[-1].copy(),
    )


def _validate_source_snippet(
    mother_event: GeneratedEvent, source_snippet: MotionSnippet
) -> None:
    if not isinstance(source_snippet, MotionSnippet):
        raise TypeError("source_snippet must be a MotionSnippet")
    target = mother_event.target
    expected_spec = {
        "object_type": source_snippet.object_type,
        "footprint": dict(source_snippet.footprint),
    }
    if (
        target.snippet_id != source_snippet.snippet_id
        or target.source_object_id != source_snippet.source_object_id
        or target.object_type != source_snippet.object_type
        or target.footprint_spec != expected_spec
    ):
        raise PairGenerationError("mother_snippet_contract_mismatch")
    provenance = target.provenance
    if not isinstance(provenance, Mapping):
        raise PairGenerationError("mother_target_provenance_invalid")
    for field, label in (
        ("source_recording_id", "source recording"),
        ("source_session_id", "source session"),
    ):
        reason_field = field.removesuffix("_id")
        source_value = getattr(source_snippet, field)
        target_value = provenance.get(field)
        if (
            not isinstance(source_value, str)
            or not source_value.strip()
            or not isinstance(target_value, str)
            or not target_value.strip()
        ):
            raise PairGenerationError(
                f"mother_snippet_{reason_field}_invalid",
                f"mother target/source snippet {label} must be non-empty",
            )
        if target_value != source_value:
            raise PairGenerationError(
                f"mother_snippet_{reason_field}_mismatch",
                f"mother target/source snippet {label} mismatch",
            )


def _target_clearances(
    target: TransplantedDynamicObject,
    *,
    trajectory: LocalTrajectory,
    environment: _PairEnvironment,
) -> np.ndarray:
    return trajectory_signed_clearances(
        environment.robot_footprint,
        trajectory.poses,
        environment.target_footprint,
        target.future_poses,
    )


def _validate_target_candidate(
    target: TransplantedDynamicObject,
    *,
    trajectory: LocalTrajectory,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    environment: _PairEnvironment,
) -> tuple[np.ndarray, np.ndarray]:
    history = np.asarray(target.history_poses)
    current = np.asarray(target.current_pose)
    future = np.asarray(target.future_poses)
    if (
        history.shape != (environment.grid.history_steps, 3)
        or current.shape != (3,)
        or future.shape != (environment.grid.future_steps, 3)
        or any(array.dtype != np.float32 for array in (history, current, future))
        or not all(np.isfinite(array).all() for array in (history, current, future))
        or not np.array_equal(history[-1], current)
    ):
        raise _CandidateRejected("target_contract_invalid")
    all_poses = np.vstack((history, future))
    if not bool(np.all(points_in_grid(all_poses[:, :2], environment.grid))):
        raise _CandidateRejected("target_out_of_bounds")
    if intersects(
        environment.robot_footprint,
        np.zeros(3, dtype=np.float64),
        environment.target_footprint,
        target.current_pose,
    ):
        raise _CandidateRejected("target_current_robot_overlap")

    if (
        environment.base_static_occupancy.any()
        and swept_footprint_intersects_occupancy(
            environment.target_footprint,
            all_poses,
            environment.base_static_occupancy,
            grid=environment.grid,
        )
    ):
        raise _CandidateRejected("target_static_collision")
    for occluder_footprint, occluder_pose in environment.occluder_geometry:
        if synchronized_sweeps_intersect(
            environment.target_footprint,
            all_poses,
            occluder_footprint,
            np.tile(occluder_pose, (all_poses.shape[0], 1)),
            grid=environment.grid,
        ):
            raise _CandidateRejected("target_occluder_collision")

    for object_id in sorted(environment.context_footprints):
        context_footprint = environment.context_footprints[object_id]
        context_poses = np.vstack(
            (
                oracle_context.dynamic_object_history[object_id],
                oracle_context.dynamic_object_future[object_id],
            )
        )
        if synchronized_sweeps_intersect(
            environment.target_footprint,
            all_poses,
            context_footprint,
            context_poses,
            grid=environment.grid,
        ):
            raise _CandidateRejected("target_context_collision")

    base_only_ids = set(environment.scene_history_footprints) - set(
        environment.context_footprints
    )
    for object_id in sorted(base_only_ids):
        if synchronized_sweeps_intersect(
            environment.target_footprint,
            history,
            environment.scene_history_footprints[object_id],
            environment.scene_dynamic_history[object_id],
            grid=environment.grid,
        ):
            raise _CandidateRejected("target_base_history_collision")

    if synchronized_sweeps_intersect(
        environment.robot_footprint,
        base_state.robot_history,
        environment.target_footprint,
        history,
        grid=environment.grid,
    ):
        raise _CandidateRejected("target_robot_history_collision")

    velocities = np.diff(all_poses[:, :2].astype(np.float64), axis=0) / (
        environment.future_dt_s
    )
    accelerations = np.diff(velocities, axis=0) / environment.future_dt_s
    dynamic_config = base_config["dynamic_objects"][target.object_type]
    if np.any(
        np.linalg.norm(velocities, axis=1)
        > float(dynamic_config["max_speed_mps"]) + 1e-6
    ):
        raise _CandidateRejected("target_speed_limit")
    if accelerations.size and np.any(
        np.linalg.norm(accelerations, axis=1)
        > float(dynamic_config["max_acceleration_mps2"]) + 1e-5
    ):
        raise _CandidateRejected("target_acceleration_limit")

    history_visibility = np.asarray(
        [
            footprint_visibility_sequence(
                environment.target_footprint,
                history[index : index + 1],
                environment.visibility_history[index],
                environment.grid,
            )[0]
            for index in range(environment.grid.history_steps)
        ],
        dtype=bool,
    )
    sequence = footprint_visibility_sequence(
        environment.target_footprint,
        np.vstack((current, future)),
        environment.visibility_mask,
        environment.grid,
    )
    if history_visibility.shape != (environment.grid.history_steps,):
        raise _CandidateRejected("target_visibility_history_contract_invalid")
    if history_visibility[-1] != sequence[0]:
        raise _CandidateRejected("target_visibility_seam_invalid")
    if bool(sequence[0]):
        raise _CandidateRejected("target_current_visible")
    if not has_continuous_emergence(sequence, min_visible_frames=2):
        raise _CandidateRejected("target_does_not_emerge")
    if not bool(sequence[-1]):
        raise _CandidateRejected("target_final_not_visible")
    return history_visibility, sequence


def compute_pair_group_id(
    *,
    generated_event_id: str,
    base_state_id: str,
    trajectory_id: str,
    occluders: Iterable[Mapping[str, Any]],
    blind_spot_config: Mapping[str, Any],
    source_snippet_id: str,
    target_dynamic_object_id: str,
    paired_config_digest: str,
) -> str:
    """Compute the stable SOP06 group ID from trusted mother lineage."""

    geometry_payload = json.dumps(
        {
            "occluders": list(occluders),
            "blind_spot_config": blind_spot_config,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = stable_digest(
        generated_event_id,
        base_state_id,
        trajectory_id,
        stable_digest(geometry_payload, size=16),
        source_snippet_id,
        target_dynamic_object_id,
        paired_config_digest,
        size=12,
    )
    return f"pair-{digest}"


def _pair_group_identifier(
    mother_event: GeneratedEvent,
    *,
    trajectory: LocalTrajectory,
    paired_config: PairedVariantConfig,
) -> str:
    return compute_pair_group_id(
        generated_event_id=mother_event.generated_event_id,
        base_state_id=mother_event.world.base_state_id,
        trajectory_id=trajectory.trajectory_id,
        occluders=mother_event.world.occluders,
        blind_spot_config=mother_event.world.blind_spot_config,
        source_snippet_id=mother_event.target.snippet_id,
        target_dynamic_object_id=(
            mother_event.target.target_dynamic_object_id
        ),
        paired_config_digest=paired_config.digest,
    )


def _paired_target_motion_digest(target: TransplantedDynamicObject) -> str:
    """Bind one paired target identity to its complete observed/future motion."""

    history_digest = compute_motion_array_digest(
        target.history_poses, field_name="target_history_poses"
    )
    future_digest = compute_motion_array_digest(
        target.future_poses, field_name="target_future_poses"
    )
    if not np.array_equal(target.history_poses[-1], target.current_pose):
        raise PairGenerationError("paired_target_current_history_mismatch")
    return stable_digest(
        "paired_target_motion_history8_future15_v1",
        history_digest,
        target.current_pose.tobytes(order="C"),
        future_digest,
        target.footprint_spec_digest,
        size=16,
    )


def _variant_world(
    *,
    mother_event: GeneratedEvent,
    pair_group_id: str,
    variant_kind: str,
    target: TransplantedDynamicObject | None,
    target_visibility_history: np.ndarray | None,
    visibility_sequence: np.ndarray | None,
    min_clearance_m: float | None,
    time_to_min_clearance_s: float | None,
    paired_config: PairedVariantConfig,
    history_visibility_policy: HistoryVisibilityPolicy,
    seed: int,
    transform_metadata: Mapping[str, object],
    environment: _PairEnvironment,
) -> OracleWorld:
    target_id = mother_event.target.target_dynamic_object_id
    if target is not None:
        mother_provenance = mother_event.target.provenance
        for field, label in (
            ("source_recording_id", "source recording"),
            ("source_session_id", "source session"),
        ):
            target_value = target.provenance.get(field)
            mother_value = mother_provenance.get(field)
            if (
                not isinstance(target_value, str)
                or not target_value.strip()
                or not isinstance(mother_value, str)
                or not mother_value.strip()
            ):
                raise PairGenerationError(
                    f"paired_target_{field}_invalid",
                    f"paired/mother target {label} must be non-empty",
                )
            if target_value != mother_value:
                raise PairGenerationError(
                    f"paired_target_{field}_mismatch",
                    f"paired target {label} differs from mother target",
                )
    trajectories = {
        object_id: poses.copy()
        for object_id, poses in sorted(
            mother_event.world.dynamic_object_trajectories.items()
        )
        if object_id != target_id
    }
    specs = {
        object_id: dict(spec)
        for object_id, spec in sorted(mother_event.world.dynamic_object_specs.items())
        if object_id != target_id
    }
    if target is not None:
        trajectories[target_id] = target.future_poses.copy()
        specs[target_id] = dict(target.footprint_spec)
        target_history_digest = compute_motion_array_digest(
            target.history_poses, field_name="target_history_poses"
        )
        target_future_digest = compute_motion_array_digest(
            target.future_poses, field_name="target_future_poses"
        )
        target_digest = _paired_target_motion_digest(target)
    else:
        target_history_digest = None
        target_future_digest = None
        target_digest = "target-empty"
    history_assessment = (
        None
        if target_visibility_history is None
        else classify_history_visibility(
            target_visibility_history,
            history_visibility_policy,
        )
    )
    world_id = "world-" + stable_digest(
        pair_group_id,
        variant_kind,
        target_digest,
        int(seed),
        paired_config.digest,
        size=12,
    )
    mother_metadata = {
        key: value
        for key, value in mother_event.world.metadata.items()
        if key not in _SOP05_JOIN_METADATA_KEYS
    }
    metadata = {
        **mother_metadata,
        "schema_version": SCHEMA_VERSION,
        "paired_generator_algorithm_version": (
            paired_config.paired_generator_algorithm_version
        ),
        "pair_group_contract_version": paired_config.group_contract_version,
        "world_id": world_id,
        "mother_generated_event_id": mother_event.generated_event_id,
        "mother_world_id": mother_event.world.world_id,
        "mother_target_motion_record_digest": (
            mother_event.target_motion_record.record_digest
        ),
        "mother_source_snippet_id": mother_event.target.snippet_id,
        "mother_source_object_id": mother_event.target.source_object_id,
        "mother_target_type_policy_digest": (
            mother_event.target_motion_record.target_type_policy_digest
        ),
        "mother_target_footprint_spec_digest": (
            mother_event.target.footprint_spec_digest
        ),
        "pair_group_id": pair_group_id,
        "paired_variant_kind": variant_kind,
        "paired_config_digest": paired_config.digest,
        "paired_seed": int(seed),
        "target_dynamic_object_id": target_id,
        "target_present": target is not None,
        "paired_target_history_array_digest": target_history_digest,
        "paired_target_future_array_digest": target_future_digest,
        "paired_target_motion_digest": target_digest,
        "paired_target_current_pose": (
            None
            if target is None
            else [float(value) for value in target.current_pose]
        ),
        "target_provenance": (
            None if target is None else dict(target.provenance)
        ),
        "visibility_sequence": (
            None
            if visibility_sequence is None
            else [bool(value) for value in visibility_sequence]
        ),
        "target_visibility_history": (
            None
            if target_visibility_history is None
            else [bool(value) for value in target_visibility_history]
        ),
        "target_history_visibility_regime": (
            None if history_assessment is None else history_assessment.regime
        ),
        "target_history_last_visible_index": (
            None
            if history_assessment is None
            else history_assessment.last_visible_index
        ),
        "target_history_trailing_hidden_frames": (
            None
            if history_assessment is None
            else history_assessment.trailing_hidden_frames
        ),
        "target_history_visibility_policy_version": (
            HISTORY_VISIBILITY_POLICY_VERSION
        ),
        "target_history_visibility_policy": (
            history_visibility_policy.as_dict()
        ),
        "target_history_visibility_policy_digest": (
            history_visibility_policy.digest
        ),
        "min_clearance_m": min_clearance_m,
        "time_to_min_clearance_s": time_to_min_clearance_s,
        "paired_transform": dict(transform_metadata),
    }
    world = OracleWorld(
        world_id=world_id,
        base_state_id=mother_event.world.base_state_id,
        static_occupancy=mother_event.world.static_occupancy.copy(),
        dynamic_object_trajectories=trajectories,
        dynamic_object_specs=specs,
        occluders=tuple(dict(item) for item in mother_event.world.occluders),
        blind_spot_config=dict(mother_event.world.blind_spot_config),
        random_seed=int(seed),
        metadata=metadata,
    )
    validate_oracle_world(world, environment.grid)
    return world


def _make_variant(
    *,
    variant_kind: str,
    target: TransplantedDynamicObject | None,
    target_visibility_history: np.ndarray | None,
    visibility_sequence: np.ndarray | None,
    mother_event: GeneratedEvent,
    trajectory: LocalTrajectory,
    pair_group_id: str,
    paired_config: PairedVariantConfig,
    history_visibility_policy: HistoryVisibilityPolicy,
    seed: int,
    environment: _PairEnvironment,
    transform_metadata: Mapping[str, object],
    temporal_offset_s: float | None = None,
    lateral_offset_m: float | None = None,
    radial_shift_m: float | None = None,
    rotation_rad: float | None = None,
) -> PairedVariant:
    if target is None:
        clearance_sequence = None
        minimum = None
        time_to_minimum = None
    else:
        clearance_sequence = _target_clearances(
            target, trajectory=trajectory, environment=environment
        )
        minimum_index = int(np.argmin(clearance_sequence))
        minimum = float(clearance_sequence[minimum_index])
        time_to_minimum = float(
            (minimum_index + 1) * environment.future_dt_s
        )
    world = _variant_world(
        mother_event=mother_event,
        pair_group_id=pair_group_id,
        variant_kind=variant_kind,
        target=target,
        target_visibility_history=target_visibility_history,
        visibility_sequence=visibility_sequence,
        min_clearance_m=minimum,
        time_to_min_clearance_s=time_to_minimum,
        paired_config=paired_config,
        history_visibility_policy=history_visibility_policy,
        seed=seed,
        transform_metadata=transform_metadata,
        environment=environment,
    )
    return PairedVariant(
        variant_kind=variant_kind,
        world=world,
        target=target,
        target_visibility_history=(
            None
            if target_visibility_history is None
            else target_visibility_history.copy()
        ),
        visibility_sequence=(
            None if visibility_sequence is None else visibility_sequence.copy()
        ),
        clearance_sequence_m=(
            None if clearance_sequence is None else clearance_sequence.copy()
        ),
        min_clearance_m=minimum,
        time_to_min_clearance_s=time_to_minimum,
        temporal_offset_s=temporal_offset_s,
        lateral_offset_m=lateral_offset_m,
        radial_shift_m=radial_shift_m,
        rotation_rad=rotation_rad,
    )


def _transformed_target(
    mother_target: TransplantedDynamicObject,
    *,
    radial_shift_m: float,
    signed_arc_offset_m: float,
    pivot_radius_m: float,
    ray_direction: np.ndarray,
) -> TransplantedDynamicObject:
    angle = signed_arc_offset_m / pivot_radius_m
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    poses = np.vstack(
        (mother_target.history_poses, mother_target.future_poses)
    ).astype(np.float64)
    pivot = poses[mother_target.history_poses.shape[0] - 1, :2].copy()
    poses[:, :2] = (
        (poses[:, :2] - pivot) @ rotation.T
        + pivot
        + radial_shift_m * ray_direction
    )
    poses[:, 2] = wrap_angle(poses[:, 2] + angle)
    poses = poses.astype(np.float32)
    provenance = {
        **mother_target.provenance,
        "paired_transform": {
            "kind": "hidden_pose_pivot_v1",
            "radial_shift_m": float(radial_shift_m),
            "signed_arc_offset_m": float(signed_arc_offset_m),
            "rotation_rad": float(angle),
        },
    }
    return replace(
        mother_target,
        history_poses=poses[: mother_target.history_poses.shape[0]],
        current_pose=poses[mother_target.history_poses.shape[0] - 1].copy(),
        future_poses=poses[mother_target.history_poses.shape[0] :],
        provenance=provenance,
    )


def _transform_parameter_grid(
    mother_event: GeneratedEvent,
    paired_config: PairedVariantConfig,
) -> tuple[float, np.ndarray, tuple[tuple[float, float], ...]]:
    all_poses = np.vstack(
        (mother_event.target.current_pose, mother_event.target.future_poses)
    ).astype(np.float64)
    pivot = all_poses[0, :2]
    conflict_pose_index = min(
        mother_event.conflict_index + 1, all_poses.shape[0] - 1
    )
    pivot_radius = float(
        np.linalg.norm(all_poses[conflict_pose_index, :2] - pivot)
    )
    if pivot_radius <= 1e-6:
        pivot_radius = float(
            np.max(np.linalg.norm(all_poses[:, :2] - pivot, axis=1))
        )
    if pivot_radius <= 1e-6:
        raise PairGenerationError("target_pivot_degenerate")
    sensor_distance = float(np.linalg.norm(pivot))
    if sensor_distance <= 1e-6:
        raise PairGenerationError("target_blind_ray_degenerate")
    ray = pivot / sensor_distance

    step = paired_config.lateral_offset_step_m
    maximum = paired_config.lateral_offset_max_m
    count = int(np.floor(maximum / step + 1e-9))
    exact_offsets = tuple(float(step * index) for index in range(1, count + 1))
    parameters: list[tuple[float, float]] = []
    for arc in exact_offsets:
        parameters.extend(((0.0, -arc), (0.0, arc)))

    stride = max(1, int(np.ceil(0.05 / step)))
    coarse_offsets = list(exact_offsets[stride - 1 :: stride])
    if not coarse_offsets or not np.isclose(coarse_offsets[-1], maximum):
        coarse_offsets.append(maximum)
    for radial in coarse_offsets:
        for arc in coarse_offsets:
            parameters.extend(((radial, -arc), (radial, arc)))
    return pivot_radius, ray, tuple(dict.fromkeys(parameters))


def _geometric_candidate_pool(
    *,
    mother_event: GeneratedEvent,
    trajectory: LocalTrajectory,
    environment: _PairEnvironment,
    paired_config: PairedVariantConfig,
) -> _GeometricCandidatePool:
    pivot_radius, ray, parameters = _transform_parameter_grid(
        mother_event, paired_config
    )
    candidates = []
    for radial, signed_arc in parameters:
        target = _transformed_target(
            mother_event.target,
            radial_shift_m=radial,
            signed_arc_offset_m=signed_arc,
            pivot_radius_m=pivot_radius,
            ray_direction=ray,
        )
        clearances = _target_clearances(
            target, trajectory=trajectory, environment=environment
        )
        minimum_index = int(np.argmin(clearances))
        minimum = float(clearances[minimum_index])
        candidates.append((radial, signed_arc, minimum, minimum_index))
    return _GeometricCandidatePool(
        pivot_radius_m=pivot_radius,
        ray_direction=ray,
        candidates=tuple(candidates),
    )


def _ranked_geometric_candidates(
    *,
    mother_event: GeneratedEvent,
    paired_config: PairedVariantConfig,
    variant_kind: str,
    pool: _GeometricCandidatePool,
) -> tuple[tuple[float, float, float, int], ...]:
    candidates = []
    for radial, signed_arc, minimum, minimum_index in pool.candidates:
        if variant_kind == "near_miss":
            lower, upper = paired_config.near_miss_clearance_range_m
            eligible = lower <= minimum <= upper
            midpoint = 0.5 * (lower + upper)
            score = abs(minimum - midpoint)
        elif variant_kind == "spatial_safe":
            lower, upper = paired_config.spatial_safe_clearance_range_m
            eligible = (
                lower <= minimum <= upper
                and abs(minimum_index - mother_event.conflict_index) <= 1
            )
            midpoint = 0.5 * (lower + upper)
            score = abs(minimum - midpoint)
        elif variant_kind == "irrelevant_hidden":
            eligible = minimum >= paired_config.irrelevant_min_clearance_m
            score = minimum - paired_config.irrelevant_min_clearance_m
        else:  # pragma: no cover - private caller is frozen
            raise ValueError(f"unknown geometric paired variant: {variant_kind}")
        if eligible:
            candidates.append(
                (
                    float(score),
                    float(radial),
                    float(signed_arc),
                    int(minimum_index),
                )
            )
    candidates.sort(
        key=lambda item: (
            item[0],
            abs(item[1]) + abs(item[2]),
            item[1],
            item[2],
            item[3],
        )
    )
    return tuple(candidates)


def _geometric_variant(
    *,
    variant_kind: str,
    mother_event: GeneratedEvent,
    trajectory: LocalTrajectory,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    paired_config: PairedVariantConfig,
    history_visibility_policy: HistoryVisibilityPolicy,
    required_history_regime: str | None = None,
    pair_group_id: str,
    seed: int,
    environment: _PairEnvironment,
    candidate_pool: _GeometricCandidatePool,
) -> tuple[PairedVariant | None, str | None]:
    candidates = _ranked_geometric_candidates(
        mother_event=mother_event,
        paired_config=paired_config,
        variant_kind=variant_kind,
        pool=candidate_pool,
    )
    last_reason = f"{variant_kind}_clearance_unavailable"
    for _, radial, signed_arc, _ in candidates[:256]:
        target = _transformed_target(
            mother_event.target,
            radial_shift_m=radial,
            signed_arc_offset_m=signed_arc,
            pivot_radius_m=candidate_pool.pivot_radius_m,
            ray_direction=candidate_pool.ray_direction,
        )
        try:
            history_visibility, visibility = _validate_target_candidate(
                target,
                trajectory=trajectory,
                base_state=base_state,
                oracle_context=oracle_context,
                base_config=base_config,
                environment=environment,
            )
        except _CandidateRejected as exc:
            last_reason = exc.reason
            continue
        if required_history_regime is not None and classify_history_visibility(
            history_visibility,
            history_visibility_policy,
        ).regime != required_history_regime:
            last_reason = "target_history_visibility_regime_changed"
            continue
        angle = signed_arc / candidate_pool.pivot_radius_m
        transform_metadata = {
            "kind": "hidden_pose_pivot_v1",
            "radial_shift_m": radial,
            "signed_arc_offset_m": signed_arc,
            "rotation_rad": angle,
        }
        return (
            _make_variant(
                variant_kind=variant_kind,
                target=target,
                target_visibility_history=history_visibility,
                visibility_sequence=visibility,
                mother_event=mother_event,
                trajectory=trajectory,
                pair_group_id=pair_group_id,
                paired_config=paired_config,
                history_visibility_policy=history_visibility_policy,
                seed=seed,
                environment=environment,
                transform_metadata=transform_metadata,
                lateral_offset_m=abs(signed_arc),
                radial_shift_m=radial,
                rotation_rad=angle,
            ),
            None,
        )
    return None, last_reason


def _paths_spatially_intersect(
    *,
    trajectory: LocalTrajectory,
    target: TransplantedDynamicObject,
    environment: _PairEnvironment,
) -> bool:
    return any(
        signed_clearance(
            environment.robot_footprint,
            robot_pose,
            environment.target_footprint,
            target_pose,
        )
        <= 0.0
        for robot_pose in trajectory.poses
        for target_pose in target.future_poses
    )


def _temporal_variant(
    *,
    mother_event: GeneratedEvent,
    source_snippet: MotionSnippet,
    trajectory: LocalTrajectory,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    paired_config: PairedVariantConfig,
    history_visibility_policy: HistoryVisibilityPolicy,
    required_history_regime: str | None = None,
    pair_group_id: str,
    seed: int,
    environment: _PairEnvironment,
) -> tuple[PairedVariant | None, str | None]:
    provenance = mother_event.target.provenance
    required = {
        "conflict_point",
        "time_scale",
        "target_type_policy_digest",
        "seed",
    }
    crossing_direction = provenance.get(
        "desired_crossing_direction", provenance.get("crossing_direction")
    )
    if not required <= set(provenance) or crossing_direction is None:
        return None, "mother_target_provenance_incomplete"
    horizon_s = environment.grid.future_steps * environment.future_dt_s
    last_reason = "temporal_offset_unavailable"
    last_transplant_reason: str | None = None
    transplanted_candidate_count = 0
    for offset in paired_config.temporal_offset_candidates_s:
        conflict_time = mother_event.conflict_time_s + offset
        if not 0.0 < conflict_time <= horizon_s:
            last_reason = "temporal_offset_out_of_horizon"
            continue
        try:
            target = transplant_snippet(
                source_snippet,
                conflict_point=provenance["conflict_point"],
                conflict_time_s=conflict_time,
                crossing_direction=crossing_direction,
                time_scale=provenance["time_scale"],
                future_dt_s=environment.future_dt_s,
                future_steps=environment.grid.future_steps,
                base_state_id=mother_event.world.base_state_id,
                trajectory_id=trajectory.trajectory_id,
                target_type_policy_digest=provenance[
                    "target_type_policy_digest"
                ],
                seed=provenance["seed"],
                context_object_ids=tuple(
                    oracle_context.dynamic_object_future
                ),
            )
        except TransplantError as exc:
            last_reason = f"temporal_transplant:{exc.reason}"
            last_transplant_reason = last_reason
            continue
        transplanted_candidate_count += 1
        target = replace(
            target,
            target_dynamic_object_id=mother_event.target.target_dynamic_object_id,
            provenance={
                **target.provenance,
                "paired_transform": {
                    "kind": "temporal_offset_v1",
                    "temporal_offset_s": float(offset),
                    "mother_conflict_time_s": mother_event.conflict_time_s,
                },
            },
        )
        clearances = _target_clearances(
            target, trajectory=trajectory, environment=environment
        )
        if np.any(clearances <= 0.0):
            last_reason = "temporal_variant_still_collides"
            continue
        if not _paths_spatially_intersect(
            trajectory=trajectory, target=target, environment=environment
        ):
            last_reason = "temporal_spatial_paths_do_not_intersect"
            continue
        try:
            history_visibility, visibility = _validate_target_candidate(
                target,
                trajectory=trajectory,
                base_state=base_state,
                oracle_context=oracle_context,
                base_config=base_config,
                environment=environment,
            )
        except _CandidateRejected as exc:
            last_reason = exc.reason
            continue
        if required_history_regime is not None and classify_history_visibility(
            history_visibility,
            history_visibility_policy,
        ).regime != required_history_regime:
            last_reason = "target_history_visibility_regime_changed"
            continue
        transform_metadata = {
            "kind": "temporal_offset_v1",
            "temporal_offset_s": float(offset),
            "mother_conflict_time_s": mother_event.conflict_time_s,
            "variant_conflict_time_s": float(conflict_time),
        }
        return (
            _make_variant(
                variant_kind="temporal_safe",
                target=target,
                target_visibility_history=history_visibility,
                visibility_sequence=visibility,
                mother_event=mother_event,
                trajectory=trajectory,
                pair_group_id=pair_group_id,
                paired_config=paired_config,
                history_visibility_policy=history_visibility_policy,
                seed=seed,
                environment=environment,
                transform_metadata=transform_metadata,
                temporal_offset_s=float(offset),
            ),
            None,
        )
    if transplanted_candidate_count == 0 and last_transplant_reason is not None:
        return None, last_transplant_reason
    return None, last_reason


def _stamp_group_metadata(
    group: PairedEventGroup,
) -> PairedEventGroup:
    coverage = {
        kind: bool(group.coverage_mask[index])
        for index, kind in enumerate(VARIANT_ORDER)
    }
    variants = []
    for variant in group.variants:
        metadata = {
            **variant.world.metadata,
            "paired_generator_algorithm_version": (
                PAIRED_GENERATOR_ALGORITHM_VERSION
            ),
            "pair_group_contract_version": PAIRED_GROUP_CONTRACT_VERSION,
            "paired_coverage_mask": list(group.coverage_mask),
            "paired_coverage": coverage,
            "paired_missing_variant_reasons": dict(
                group.missing_variant_reasons
            ),
            "paired_group_complete": group.is_complete,
            "eligible_for_strict_paired_evaluation": (
                group.eligible_for_strict_evaluation
            ),
        }
        variants.append(
            replace(variant, world=replace(variant.world, metadata=metadata))
        )
    return replace(group, variants=tuple(variants))


def assemble_paired_event_group(
    *,
    pair_group_id: str,
    variants: Mapping[str, PairedVariant],
    missing_variant_reasons: Mapping[str, str],
    paired_config: PairedVariantConfig | Mapping[str, Any],
) -> PairedEventGroup:
    """Enforce the frozen minimum partial-group and strict-eval policies."""

    normalized = _as_paired_config(paired_config)
    if not isinstance(pair_group_id, str) or not pair_group_id:
        raise ValueError("pair_group_id must be a non-empty string")
    if not isinstance(variants, Mapping):
        raise TypeError("variants must be a mapping")
    unknown = set(variants) - set(VARIANT_ORDER)
    if unknown:
        raise PairGenerationError("unknown_paired_variant")
    for kind, variant in variants.items():
        if not isinstance(variant, PairedVariant) or variant.variant_kind != kind:
            raise PairGenerationError("paired_variant_kind_mismatch")
        if variant.world.metadata.get("pair_group_id") != pair_group_id:
            raise PairGenerationError("paired_group_id_mismatch")
    absent = set(VARIANT_ORDER) - set(variants)
    reasons = dict(missing_variant_reasons)
    if set(reasons) != absent or any(
        not isinstance(reason, str) or not reason for reason in reasons.values()
    ):
        raise PairGenerationError("missing_variant_reasons_incomplete")
    missing_required = set(normalized.mother_required_variants) - set(variants)
    if missing_required:
        raise PairGenerationError(
            "mother_required_variants_missing",
            f"mother required variants missing: {sorted(missing_required)}",
        )
    coverage = tuple(kind in variants for kind in VARIANT_ORDER)
    complete = all(coverage)
    group = PairedEventGroup(
        pair_group_id=pair_group_id,
        variants=tuple(variants[kind] for kind in VARIANT_ORDER if kind in variants),
        coverage_mask=coverage,
        missing_variant_reasons={
            kind: reasons[kind] for kind in VARIANT_ORDER if kind in reasons
        },
        is_complete=complete,
        eligible_for_strict_evaluation=(
            complete and normalized.audit_requires_all_variants
        ),
        paired_config_digest=normalized.digest,
    )
    return _stamp_group_metadata(group)


def generate_paired_variants(
    *,
    mother_event: GeneratedEvent,
    source_snippet: MotionSnippet,
    base_state: BaseState,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    paired_config: PairedVariantConfig | Mapping[str, Any],
    seed: int,
) -> PairedEventGroup:
    """Generate one six-position group, retaining policy-compliant partials."""

    if not isinstance(mother_event, GeneratedEvent):
        raise TypeError("mother_event must be a GeneratedEvent")
    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(trajectory, LocalTrajectory):
        raise TypeError("trajectory must be a LocalTrajectory")
    if not isinstance(oracle_context, OracleContext):
        raise TypeError("oracle_context must be an OracleContext")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")
    mother_generator_version = mother_event.world.metadata.get(
        "generator_algorithm_version"
    )
    if mother_generator_version != SOP05_GENERATOR_ALGORITHM_VERSION:
        reason = (
            "retired_mother_generator_version"
            if mother_generator_version == "blind_reachability_first_v1"
            else "unsupported_mother_generator_version"
        )
        raise PairGenerationError(
            reason,
            "mother generator_algorithm_version must equal "
            f"{SOP05_GENERATOR_ALGORITHM_VERSION}",
        )
    joint_version = mother_event.world.metadata.get(
        "joint_pair_generator_algorithm_version"
    )
    if joint_version is not None:
        if joint_version == JOINT_ENVIRONMENT_PAIR_VERSION:
            raise PairGenerationError(
                "retired_joint_mother",
                f"retired {JOINT_ENVIRONMENT_PAIR_VERSION} mother is invalid",
            )
        raise PairGenerationError(
            "unsupported_joint_mother",
            "mother contains unsupported joint-pair identity",
        )
    normalized = _as_paired_config(paired_config)
    history_visibility_policy, mother_history_assessment = (
        _mother_history_visibility_contract(mother_event)
    )
    _validate_source_snippet(mother_event, source_snippet)
    environment = _pair_environment(
        mother_event=mother_event,
        trajectory=trajectory,
        base_state=base_state,
        oracle_context=oracle_context,
        base_config=base_config,
        critical_clearance_threshold_m=(
            normalized.near_miss_clearance_range_m[1]
        ),
    )
    pair_id = _pair_group_identifier(
        mother_event, trajectory=trajectory, paired_config=normalized
    )

    try:
        mother_history_visibility, mother_visibility = _validate_target_candidate(
            mother_event.target,
            trajectory=trajectory,
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
            environment=environment,
        )
    except _CandidateRejected as exc:
        raise PairGenerationError(
            "collision_mother_invalid", f"collision mother invalid: {exc.reason}"
        ) from exc
    mother_clearances = _target_clearances(
        mother_event.target, trajectory=trajectory, environment=environment
    )
    if not np.any(mother_clearances <= 0.0):
        raise PairGenerationError("collision_mother_does_not_collide")

    variants: dict[str, PairedVariant] = {}
    variants["collision"] = _make_variant(
        variant_kind="collision",
        target=mother_event.target,
        target_visibility_history=mother_history_visibility,
        visibility_sequence=mother_visibility,
        mother_event=mother_event,
        trajectory=trajectory,
        pair_group_id=pair_id,
        paired_config=normalized,
        history_visibility_policy=history_visibility_policy,
        seed=int(seed),
        environment=environment,
        transform_metadata={"kind": "collision_mother"},
    )

    missing: dict[str, str] = {}
    candidate_pool = _geometric_candidate_pool(
        mother_event=mother_event,
        trajectory=trajectory,
        environment=environment,
        paired_config=normalized,
    )
    for kind in ("near_miss", "spatial_safe", "irrelevant_hidden"):
        variant, reason = _geometric_variant(
            variant_kind=kind,
            mother_event=mother_event,
            trajectory=trajectory,
            base_state=base_state,
            oracle_context=oracle_context,
            base_config=base_config,
            paired_config=normalized,
            history_visibility_policy=history_visibility_policy,
            required_history_regime=mother_history_assessment.regime,
            pair_group_id=pair_id,
            seed=int(seed),
            environment=environment,
            candidate_pool=candidate_pool,
        )
        if variant is None:
            missing[kind] = str(reason)
        elif classify_history_visibility(
            variant.target_visibility_history,
            history_visibility_policy,
        ).regime != mother_history_assessment.regime:
            missing[kind] = "target_history_visibility_regime_changed"
        else:
            variants[kind] = variant

    temporal, reason = _temporal_variant(
        mother_event=mother_event,
        source_snippet=source_snippet,
        trajectory=trajectory,
        base_state=base_state,
        oracle_context=oracle_context,
        base_config=base_config,
        paired_config=normalized,
        history_visibility_policy=history_visibility_policy,
        required_history_regime=mother_history_assessment.regime,
        pair_group_id=pair_id,
        seed=int(seed),
        environment=environment,
    )
    if temporal is None:
        missing["temporal_safe"] = str(reason)
    elif classify_history_visibility(
        temporal.target_visibility_history,
        history_visibility_policy,
    ).regime != mother_history_assessment.regime:
        missing["temporal_safe"] = (
            "target_history_visibility_regime_changed"
        )
    else:
        variants["temporal_safe"] = temporal

    variants["empty_blind_spot"] = _make_variant(
        variant_kind="empty_blind_spot",
        target=None,
        target_visibility_history=None,
        visibility_sequence=None,
        mother_event=mother_event,
        trajectory=trajectory,
        pair_group_id=pair_id,
        paired_config=normalized,
        history_visibility_policy=history_visibility_policy,
        seed=int(seed),
        environment=environment,
        transform_metadata={
            "kind": "target_removal",
            "removed_target_dynamic_object_id": (
                mother_event.target.target_dynamic_object_id
            ),
        },
    )
    return assemble_paired_event_group(
        pair_group_id=pair_id,
        variants=variants,
        missing_variant_reasons=missing,
        paired_config=normalized,
    )


def _joint_environment_anchor_schedule(
    *,
    trajectory: LocalTrajectory,
    generator_config: Mapping[str, Any],
    paired_config: PairedVariantConfig,
    future_dt_s: float,
) -> tuple[float, ...]:
    times = (
        np.arange(trajectory.poses.shape[0], dtype=np.float64) + 1.0
    ) * future_dt_s
    lower, upper = generator_config["conflict_time_range_s"]
    eligible = tuple(
        float(time)
        for time in times
        if lower - 1e-9 <= time <= upper + 1e-9
    )
    positive_offsets = tuple(
        offset
        for offset in paired_config.temporal_offset_candidates_s
        if offset > 0.0
    )
    horizon_s = trajectory.poses.shape[0] * future_dt_s
    if not eligible or not positive_offsets:
        raise PairGenerationError("joint_environment_anchor_unavailable")

    def maximum_available_offset(anchor: float) -> float:
        available = tuple(
            offset
            for offset in positive_offsets
            if anchor + offset <= horizon_s + 1e-9
        )
        return max(available, default=-np.inf)

    scheduled = tuple(
        sorted(
            eligible,
            key=lambda anchor: (
                -maximum_available_offset(anchor),
                -anchor,
            ),
        )
    )
    if not np.isfinite(maximum_available_offset(scheduled[0])):
        raise PairGenerationError("joint_environment_anchor_unavailable")
    return scheduled


def _joint_environment_normal_offset_schedule(
    original_offset_m: float,
    configured_range_m: tuple[float, float],
) -> tuple[float, ...]:
    """Deterministically search the configured normal band around the mother."""

    lower, upper = (float(value) for value in configured_range_m)
    sign = -1.0 if original_offset_m < 0.0 else 1.0
    magnitudes = tuple(
        float(value) for value in np.linspace(lower, upper, 5, dtype=np.float64)
    )
    ranked = tuple(
        sorted(
            magnitudes,
            key=lambda value: (abs(value - abs(original_offset_m)), value),
        )
    )
    values = (float(original_offset_m),) + tuple(
        sign * value for value in ranked
    ) + tuple(-sign * value for value in ranked)
    return tuple(dict.fromkeys(values))


def _source_snippet_for_event(
    mother_event: GeneratedEvent,
    snippet_libraries: Mapping[str, SnippetLibrary],
) -> MotionSnippet:
    library = snippet_libraries.get(mother_event.target.object_type)
    if not isinstance(library, SnippetLibrary):
        raise PairGenerationError("mother_snippet_library_missing")
    matches = tuple(
        snippet
        for snippet in library.snippets
        if snippet.snippet_id == mother_event.target.snippet_id
        and snippet.source_object_id == mother_event.target.source_object_id
    )
    if len(matches) != 1:
        raise PairGenerationError("mother_source_snippet_not_unique")
    return matches[0]


def _retimed_target_candidate(
    *,
    mother_event: GeneratedEvent,
    source_snippet: MotionSnippet,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    environment: _PairEnvironment,
    offset_s: float,
) -> tuple[TransplantedDynamicObject | None, str | None]:
    provenance = mother_event.target.provenance
    required = {
        "conflict_point",
        "time_scale",
        "target_type_policy_digest",
        "seed",
    }
    crossing_direction = provenance.get(
        "desired_crossing_direction", provenance.get("crossing_direction")
    )
    if not required <= set(provenance) or crossing_direction is None:
        raise PairGenerationError("mother_target_provenance_incomplete")
    conflict_time = mother_event.conflict_time_s + offset_s
    horizon_s = environment.grid.future_steps * environment.future_dt_s
    if not 0.0 < conflict_time <= horizon_s:
        return None, None
    try:
        target = transplant_snippet(
            source_snippet,
            conflict_point=provenance["conflict_point"],
            conflict_time_s=conflict_time,
            crossing_direction=crossing_direction,
            time_scale=provenance["time_scale"],
            future_dt_s=environment.future_dt_s,
            future_steps=environment.grid.future_steps,
            base_state_id=mother_event.world.base_state_id,
            trajectory_id=trajectory.trajectory_id,
            target_type_policy_digest=provenance[
                "target_type_policy_digest"
            ],
            seed=provenance["seed"],
            context_object_ids=tuple(oracle_context.dynamic_object_future),
        )
    except TransplantError as exc:
        return None, f"temporal_transplant:{exc.reason}"
    return (
        replace(
            target,
            target_dynamic_object_id=(
                mother_event.target.target_dynamic_object_id
            ),
            provenance={
                **target.provenance,
                "paired_transform": {
                    "kind": "temporal_offset_v1",
                    "temporal_offset_s": float(offset_s),
                    "mother_conflict_time_s": mother_event.conflict_time_s,
                },
            },
        ),
        None,
    )


def _event_trajectory_normal(
    mother_event: GeneratedEvent, trajectory: LocalTrajectory
) -> np.ndarray:
    index = mother_event.conflict_index
    poses = trajectory.poses
    previous = poses[max(0, index - 1), :2].astype(np.float64)
    following = poses[min(poses.shape[0] - 1, index + 1), :2].astype(
        np.float64
    )
    tangent = following - previous
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-9:
        raise PairGenerationError("joint_environment_tangent_degenerate")
    tangent /= norm
    return np.asarray([-tangent[1], tangent[0]], dtype=np.float64)


def _joint_occluder_full_motion_rejection_reason(
    *,
    placement,
    base_state: BaseState,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    environment: _PairEnvironment,
    collision_target: TransplantedDynamicObject,
    temporal_target: TransplantedDynamicObject,
) -> str | None:
    """Reject a replacement occluder against every available 23-point sweep."""

    collision_sweeps = _joint_occluder_collision_sweeps(
        base_state=base_state,
        trajectory=trajectory,
        oracle_context=oracle_context,
        environment=environment,
        collision_target=collision_target,
        temporal_target=temporal_target,
    )
    return occluder_collision_sweep_rejection_reason(
        placement.footprint,
        placement.pose,
        collision_sweeps,
        grid=environment.grid,
    )


def _joint_occluder_collision_sweeps(
    *,
    base_state: BaseState,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    environment: _PairEnvironment,
    collision_target: TransplantedDynamicObject,
    temporal_target: TransplantedDynamicObject,
) -> tuple[OccluderCollisionSweep, ...]:
    """Build the deterministic complete-motion closure for one LOS search."""

    sweeps = [
        OccluderCollisionSweep(
            footprint=environment.robot_footprint,
            poses=np.vstack((base_state.robot_history, trajectory.poses)),
            rejection_reason="occluder_robot_history_overlap",
        )
    ]
    for object_id in sorted(environment.context_footprints):
        sweeps.append(
            OccluderCollisionSweep(
                footprint=environment.context_footprints[object_id],
                poses=np.vstack(
                    (
                        oracle_context.dynamic_object_history[object_id],
                        oracle_context.dynamic_object_future[object_id],
                    )
                ),
                rejection_reason="occluder_context_full_motion_collision",
            )
        )
    base_only_ids = set(environment.scene_history_footprints) - set(
        environment.context_footprints
    )
    for object_id in sorted(base_only_ids):
        sweeps.append(
            OccluderCollisionSweep(
                footprint=environment.scene_history_footprints[object_id],
                poses=environment.scene_dynamic_history[object_id],
                rejection_reason="occluder_base_history_collision",
            )
        )
    for target in (collision_target, temporal_target):
        sweeps.append(
            OccluderCollisionSweep(
                footprint=environment.target_footprint,
                poses=np.vstack((target.history_poses, target.future_poses)),
                rejection_reason="occluder_target_full_motion_collision",
            )
        )
    return tuple(sweeps)


def _rebind_mother_world(
    mother_event: GeneratedEvent,
    world: OracleWorld,
    *,
    base_state: BaseState,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    critical_clearance_threshold_m: float,
    grid,
) -> GeneratedEvent:
    """Rebuild the strict join and visibility after joint world replacement."""

    source_record = mother_event.target_motion_record
    target = mother_event.target
    record = create_event_target_motion_record(
        generated_event_id=mother_event.generated_event_id,
        world_id=world.world_id,
        base_state_id=world.base_state_id,
        trajectory_id=source_record.trajectory_id,
        target_dynamic_object_id=target.target_dynamic_object_id,
        source_snippet_id=target.snippet_id,
        source_object_id=target.source_object_id,
        object_type=target.object_type,
        footprint_spec=target.footprint_spec,
        footprint_spec_digest=target.footprint_spec_digest,
        target_type_policy_digest=source_record.target_type_policy_digest,
        history_poses=target.history_poses,
        current_pose=target.current_pose,
        future_poses=target.future_poses,
    )
    metadata = dict(world.metadata)
    metadata.update(build_event_target_motion_world_metadata(record))
    provisional_world = replace(world, metadata=metadata)
    validate_event_target_motion_world_join(record, provisional_world, grid)
    provisional_event = replace(
        mother_event,
        world=provisional_world,
        target_motion_record=record,
    )
    environment = _pair_environment(
        mother_event=provisional_event,
        trajectory=trajectory,
        base_state=base_state,
        oracle_context=oracle_context,
        base_config=base_config,
        critical_clearance_threshold_m=critical_clearance_threshold_m,
    )
    try:
        target_visibility_history, visibility_sequence = (
            _validate_target_candidate(
                target,
                trajectory=trajectory,
                base_state=base_state,
                oracle_context=oracle_context,
                base_config=base_config,
                environment=environment,
            )
        )
    except _CandidateRejected as exc:
        raise PairGenerationError(
            "joint_rebound_mother_invalid",
            f"joint rebound mother invalid: {exc.reason}",
        ) from exc
    metadata.update(
        {
            "target_visibility_history": [
                bool(value) for value in target_visibility_history
            ],
            "visibility_sequence": [
                bool(value) for value in visibility_sequence
            ],
        }
    )
    rebound_world = replace(provisional_world, metadata=metadata)
    validate_event_target_motion_world_join(record, rebound_world, grid)
    return replace(
        provisional_event,
        world=rebound_world,
        target_visibility_history=target_visibility_history.copy(),
        visibility_sequence=visibility_sequence.copy(),
    )


def _joint_environment_summary(
    *,
    seed: int,
    anchor_schedule: tuple[float, ...],
    attempted_count: int,
    mother_accepted_count: int,
    pair_candidate_count: int,
    occluder_candidate_count: int,
    rejection_reasons: Mapping[str, int],
    mother_event: GeneratedEvent | None,
    group: PairedEventGroup | None,
) -> dict[str, object]:
    accepted = int(group is not None and group.is_complete)
    temporal = None if group is None else group.by_kind.get("temporal_safe")
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_algorithm_version": JOINT_ENVIRONMENT_PAIR_VERSION,
        "seed": int(seed),
        "requested_count": 1,
        "attempted_count": attempted_count,
        "mother_accepted_count": mother_accepted_count,
        "pair_candidate_count": pair_candidate_count,
        "occluder_candidate_count": occluder_candidate_count,
        "accepted_count": accepted,
        "complete_group_count": accepted,
        "request_acceptance_rate": float(accepted),
        "attempt_acceptance_rate": (
            accepted / attempted_count if attempted_count else 0.0
        ),
        "anchor_schedule_s": list(anchor_schedule),
        "selected_conflict_time_s": (
            None if mother_event is None else mother_event.conflict_time_s
        ),
        "selected_temporal_offset_s": (
            None if temporal is None else temporal.temporal_offset_s
        ),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }


def generate_joint_environment_pair(
    *,
    base_state: BaseState,
    oracle_context: OracleContext,
    trajectory: LocalTrajectory,
    snippet_libraries: Mapping[str, SnippetLibrary],
    base_config: Mapping[str, Any],
    generator_config: Mapping[str, Any],
    paired_config: PairedVariantConfig | Mapping[str, Any],
    seed: int,
) -> JointEnvironmentPairReport:
    """Jointly search one environment mother, temporal target, and LOS envelope.

    One top-level attempt creates at most one SOP-05 mother candidate.  Only a
    physically valid, complete SOP-06 six-pack is accepted; partial groups are
    counted with explicit reasons and the bounded search continues.
    """

    if (
        isinstance(generator_config, Mapping)
        and generator_config.get("production_event_kind") == "environment"
        and isinstance(generator_config.get("blind_reachability"), Mapping)
        and generator_config["blind_reachability"].get("algorithm_version")
        == BLIND_REACHABILITY_ALGORITHM_VERSION
    ):
        raise PairGenerationError(
            "joint_environment_pair_v2_retired",
            "joint_environment_pair_v2 is retired for formal v5 input",
        )
    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(oracle_context, OracleContext):
        raise TypeError("oracle_context must be an OracleContext")
    if not isinstance(trajectory, LocalTrajectory):
        raise TypeError("trajectory must be a LocalTrajectory")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")
    normalized_generator = normalize_generator_config(generator_config)
    normalized_paired = _as_paired_config(paired_config)
    future_dt_s = float(base_config["bev"]["future_dt_s"])
    anchor_schedule = _joint_environment_anchor_schedule(
        trajectory=trajectory,
        generator_config=normalized_generator,
        paired_config=normalized_paired,
        future_dt_s=future_dt_s,
    )
    max_attempts = int(normalized_generator["max_resample_attempts"])
    grid = build_grid_spec(dict(base_config))
    base_static = (
        np.zeros((grid.height, grid.width), dtype=bool)
        if base_state.static_map_local is None
        else np.asarray(base_state.static_map_local != 0, dtype=bool)
    )
    rejection_reasons: dict[str, int] = {}
    mother_accepted_count = 0
    pair_candidate_count = 0
    occluder_candidate_count = 0
    attempted_count = 0
    search_round = 0
    anchor_attempt_index = 0
    anchor_attempt_limit = min(
        _JOINT_ENVIRONMENT_MOTHER_BATCH_SIZE,
        max_attempts,
    )
    while attempted_count < max_attempts:
        if anchor_attempt_index >= anchor_attempt_limit:
            search_round += 1
            anchor_attempt_index = 0
            anchor_attempt_limit = min(
                _JOINT_ENVIRONMENT_MOTHER_BATCH_SIZE,
                max_attempts - attempted_count,
            )
        round_index = search_round
        anchor = anchor_schedule[round_index % len(anchor_schedule)]
        event_seed = (
            int(seed)
            if round_index == 0
            else derive_seed(
                int(seed),
                base_state.state_id,
                trajectory.trajectory_id,
                "joint_environment_pair",
                round_index,
            )
        )
        # The SOP-05 joint schedule starts with eight pillar templates on both
        # sides. Consume that complete high-feasibility prefix before moving to
        # the next conflict-time anchor; shorter batches repeatedly reseed a
        # truncated prefix and substantially reduce acceptance.
        candidate_generator = {
            **normalized_generator,
            "event_type_weights": {
                "environment": 1.0,
                "structural": 0.0,
                "mixed": 0.0,
            },
            "conflict_time_range_s": (anchor, anchor),
            "max_resample_attempts": anchor_attempt_limit,
        }
        segment_start = anchor_attempt_index
        event_report = generate_events(
            base_state=base_state,
            oracle_context=oracle_context,
            trajectory=trajectory,
            snippet_libraries=snippet_libraries,
            base_config=base_config,
            generator_config=candidate_generator,
            seed=event_seed,
            event_count=1,
            attempt_index_start=segment_start,
        )
        segment_attempted = int(event_report.summary["attempted_count"])
        segment_stop = event_report.summary.get(
            "attempt_index_stop_exclusive"
        )
        if (
            event_report.summary.get("attempt_index_start") != segment_start
            or segment_stop != segment_start + segment_attempted
            or not segment_start < segment_stop <= anchor_attempt_limit
        ):
            raise PairGenerationError(
                "joint_mother_attempt_summary_invalid"
            )
        anchor_attempt_index = int(segment_stop)
        attempted_count += segment_attempted
        for reason, count in event_report.summary[
            "rejection_reasons"
        ].items():
            key = f"mother:{reason}"
            rejection_reasons[key] = rejection_reasons.get(key, 0) + int(
                count
            )
        if not event_report.events:
            continue

        mother = event_report.events[0]
        mother_accepted_count += 1
        try:
            source_snippet = _source_snippet_for_event(
                mother, snippet_libraries
            )
            environment = _pair_environment(
                mother_event=mother,
                trajectory=trajectory,
                base_state=base_state,
                oracle_context=oracle_context,
                base_config=base_config,
                critical_clearance_threshold_m=(
                    normalized_paired.near_miss_clearance_range_m[1]
                ),
            )
            normal = _event_trajectory_normal(mother, trajectory)
        except PairGenerationError as exc:
            key = f"pair:{exc.reason}"
            rejection_reasons[key] = rejection_reasons.get(key, 0) + 1
            continue

        if len(mother.world.occluders) != 1:
            key = "pair:environment_occluder_count_invalid"
            rejection_reasons[key] = rejection_reasons.get(key, 0) + 1
            continue
        original_occluder = mother.world.occluders[0]
        for offset_s in normalized_paired.temporal_offset_candidates_s:
            temporal_target, temporal_rejection_reason = (
                _retimed_target_candidate(
                    mother_event=mother,
                    source_snippet=source_snippet,
                    trajectory=trajectory,
                    oracle_context=oracle_context,
                    environment=environment,
                    offset_s=offset_s,
                )
            )
            if temporal_rejection_reason is not None:
                rejection_reasons[temporal_rejection_reason] = (
                    rejection_reasons.get(temporal_rejection_reason, 0) + 1
                )
                continue
            if temporal_target is None:
                continue
            temporal_clearances = _target_clearances(
                temporal_target,
                trajectory=trajectory,
                environment=environment,
            )
            if np.any(temporal_clearances <= 0.0):
                continue
            if not _paths_spatially_intersect(
                trajectory=trajectory,
                target=temporal_target,
                environment=environment,
            ):
                continue
            pair_candidate_count += 1
            placement = None
            visibility_sequences = None
            offset_rejection_reasons: dict[str, int] = {}
            offset_candidate_attempts = 0
            normal_offset_schedule = _joint_environment_normal_offset_schedule(
                float(original_occluder["normal_offset_m"]),
                normalized_generator["occluders"][
                    "normal_offset_range_m"
                ],
            )
            collision_sweeps = _joint_occluder_collision_sweeps(
                base_state=base_state,
                trajectory=trajectory,
                oracle_context=oracle_context,
                environment=environment,
                collision_target=mother.target,
                temporal_target=temporal_target,
            )
            current_context_poses = {
                object_id: environment.scene_dynamic_history[object_id][-1]
                for object_id in sorted(environment.scene_history_footprints)
            }
            for normal_offset_m in normal_offset_schedule:
                try:
                    candidate_placement, candidate_visibility = (
                        align_environment_occluder_to_target_los_envelope(
                            occluder_type=str(original_occluder["type"]),
                            normal_offset_m=normal_offset_m,
                            proposal_index=int(
                                original_occluder["proposal_index"]
                            ),
                            static_occupancy=base_static,
                            grid=grid,
                            sensor_pose=np.zeros(3, dtype=np.float64),
                            conflict_point=mother.target.provenance[
                                "conflict_point"
                            ],
                            trajectory_normal=normal,
                            target_visibility_pose_sequences=(
                                np.vstack(
                                    (
                                        mother.target.current_pose,
                                        mother.target.future_poses,
                                    )
                                ),
                                np.vstack(
                                    (
                                        temporal_target.current_pose,
                                        temporal_target.future_poses,
                                    )
                                ),
                            ),
                            target_footprint=environment.target_footprint,
                            current_context_poses=current_context_poses,
                            current_context_footprints=(
                                environment.scene_history_footprints
                            ),
                            collision_sweeps=collision_sweeps,
                            config=normalized_generator["occluders"],
                            min_contiguous_visible_frames=int(
                                normalized_generator[
                                    "min_contiguous_visible_frames"
                                ]
                            ),
                        )
                    )
                except OccluderSamplingError as exc:
                    offset_candidate_attempts += exc.attempts
                    detailed = exc.rejection_reasons or {exc.reason: 1}
                    for reason, count in detailed.items():
                        offset_rejection_reasons[reason] = (
                            offset_rejection_reasons.get(reason, 0)
                            + int(count)
                        )
                    continue
                offset_candidate_attempts += candidate_placement.attempt
                for reason, count in (
                    candidate_placement.rejection_reasons.items()
                ):
                    offset_rejection_reasons[reason] = (
                        offset_rejection_reasons.get(reason, 0) + int(count)
                    )
                full_motion_reason = occluder_collision_sweep_rejection_reason(
                    candidate_placement.footprint,
                    candidate_placement.pose,
                    collision_sweeps,
                    grid=environment.grid,
                )
                if full_motion_reason is not None:
                    raise PairGenerationError(
                        "joint_occluder_full_motion_certification_mismatch",
                        "inner LOS search returned an uncertified candidate: "
                        f"{full_motion_reason}",
                    )
                placement = replace(
                    candidate_placement,
                    attempt=offset_candidate_attempts,
                    rejection_reasons=dict(
                        sorted(offset_rejection_reasons.items())
                    ),
                )
                visibility_sequences = candidate_visibility
                break
            occluder_candidate_count += offset_candidate_attempts
            for reason, count in offset_rejection_reasons.items():
                key = f"occluder:{reason}"
                rejection_reasons[key] = rejection_reasons.get(key, 0) + int(
                    count
                )
            if placement is None or visibility_sequences is None:
                continue
            if placement.occluder.get("placement_strategy") != (
                JOINT_MULTI_LOS_PLACEMENT_STRATEGY_VERSION
            ):
                raise PairGenerationError(
                    "joint_occluder_placement_strategy_mismatch"
                )
            world_id = "world-" + stable_digest(
                mother.world.world_id,
                JOINT_ENVIRONMENT_PAIR_VERSION,
                placement.occluder["occluder_id"],
                event_seed,
                size=12,
            )
            metadata = {
                **mother.world.metadata,
                "joint_pair_generator_algorithm_version": (
                    JOINT_ENVIRONMENT_PAIR_VERSION
                ),
                "joint_pair_attempt_index": attempted_count - 1,
                "joint_pair_anchor_round_index": round_index,
                "joint_pair_anchor_attempt_index": int(
                    mother.world.metadata["attempt_index"]
                ),
                "joint_pair_anchor_candidate_limit": anchor_attempt_limit,
                "joint_pair_anchor_schedule_s": list(anchor_schedule),
                "joint_pair_temporal_offset_s": float(offset_s),
                "joint_pair_normal_offset_schedule_m": list(
                    normal_offset_schedule
                ),
                "joint_pair_selected_normal_offset_m": float(
                    placement.occluder["normal_offset_m"]
                ),
                "joint_pair_occluder_candidate_attempts": placement.attempt,
                "joint_pair_occluder_rejection_reasons": dict(
                    placement.rejection_reasons
                ),
                "visibility_sequence": [
                    bool(value) for value in visibility_sequences[0]
                ],
            }
            blind_spot_config = dict(mother.world.blind_spot_config)
            blind_spot_config["occluder_ids"] = [
                placement.occluder["occluder_id"]
            ]
            updated_world = replace(
                mother.world,
                world_id=world_id,
                static_occupancy=(base_static | placement.mask).astype(
                    np.float32
                ),
                occluders=(dict(placement.occluder),),
                blind_spot_config=blind_spot_config,
                random_seed=event_seed,
                metadata=metadata,
            )
            updated_mother = _rebind_mother_world(
                mother,
                updated_world,
                base_state=base_state,
                trajectory=trajectory,
                oracle_context=oracle_context,
                base_config=base_config,
                critical_clearance_threshold_m=(
                    normalized_paired.near_miss_clearance_range_m[1]
                ),
                grid=grid,
            )
            try:
                group = generate_paired_variants(
                    mother_event=updated_mother,
                    source_snippet=source_snippet,
                    base_state=base_state,
                    trajectory=trajectory,
                    oracle_context=oracle_context,
                    base_config=base_config,
                    paired_config=normalized_paired,
                    seed=int(seed),
                )
            except PairGenerationError as exc:
                key = f"pair:{exc.reason}"
                rejection_reasons[key] = rejection_reasons.get(key, 0) + 1
                continue
            if group.is_complete:
                summary = _joint_environment_summary(
                    seed=int(seed),
                    anchor_schedule=anchor_schedule,
                    attempted_count=attempted_count,
                    mother_accepted_count=mother_accepted_count,
                    pair_candidate_count=pair_candidate_count,
                    occluder_candidate_count=occluder_candidate_count,
                    rejection_reasons=rejection_reasons,
                    mother_event=updated_mother,
                    group=group,
                )
                return JointEnvironmentPairReport(
                    mother_event=updated_mother,
                    group=group,
                    summary=summary,
                )
            for kind, reason in group.missing_variant_reasons.items():
                key = f"variant:{kind}:{reason}"
                rejection_reasons[key] = rejection_reasons.get(key, 0) + 1

    summary = _joint_environment_summary(
        seed=int(seed),
        anchor_schedule=anchor_schedule,
        attempted_count=attempted_count,
        mother_accepted_count=mother_accepted_count,
        pair_candidate_count=pair_candidate_count,
        occluder_candidate_count=occluder_candidate_count,
        rejection_reasons=rejection_reasons,
        mother_event=None,
        group=None,
    )
    return JointEnvironmentPairReport(
        mother_event=None,
        group=None,
        summary=summary,
    )


def summarize_paired_groups(
    groups: Iterable[PairedEventGroup],
) -> dict[str, object]:
    """Aggregate deterministic coverage and missing-reason audit counts."""

    values = tuple(groups)
    if any(not isinstance(group, PairedEventGroup) for group in values):
        raise TypeError("groups must contain only PairedEventGroup values")
    coverage_counts = {
        kind: sum(bool(group.coverage_mask[index]) for group in values)
        for index, kind in enumerate(VARIANT_ORDER)
    }
    missing_reason_counts: dict[str, dict[str, int]] = {
        kind: {} for kind in VARIANT_ORDER
    }
    for group in values:
        for kind, reason in group.missing_variant_reasons.items():
            counts = missing_reason_counts[kind]
            counts[reason] = counts.get(reason, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "group_count": len(values),
        "complete_group_count": sum(group.is_complete for group in values),
        "partial_group_count": sum(not group.is_complete for group in values),
        "strict_evaluation_group_count": sum(
            group.eligible_for_strict_evaluation for group in values
        ),
        "coverage_counts": coverage_counts,
        "coverage_rates": {
            kind: (count / len(values) if values else 0.0)
            for kind, count in coverage_counts.items()
        },
        "missing_reason_counts": {
            kind: dict(sorted(counts.items()))
            for kind, counts in missing_reason_counts.items()
        },
    }
