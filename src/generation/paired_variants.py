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
    rasterize_footprint_sweep,
    raycast_visibility,
    signed_clearance,
    trajectory_signed_clearances,
    wrap_angle,
)
from src.utils.seeding import derive_seed, stable_digest

from .dynamic_object_transplant import (
    TransplantedDynamicObject,
    footprint_from_spec,
    transplant_snippet,
)
from .event_sampler import (
    GeneratedEvent,
    generate_events,
    normalize_generator_config,
)
from .occluder_sampler import (
    OccluderSamplingError,
    align_environment_occluder_to_target_los_envelope,
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
_REQUIRED_VARIANTS = ("collision", "empty_blind_spot")
_CONTRAST_VARIANTS = ("near_miss", "temporal_safe", "spatial_safe")
_JOINT_ENVIRONMENT_PAIR_VERSION = "joint_environment_pair_v1"
_JOINT_ENVIRONMENT_MOTHER_BATCH_SIZE = 16


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


@dataclass(frozen=True)
class PairedVariantConfig:
    schema_version: str
    near_miss_clearance_range_m: tuple[float, float]
    temporal_offset_candidates_s: tuple[float, ...]
    spatial_safe_clearance_range_m: tuple[float, float]
    irrelevant_min_clearance_m: float
    lateral_offset_step_m: float
    lateral_offset_max_m: float
    minimum_required_variants: tuple[str, ...]
    minimum_contrast_variants: tuple[str, ...]
    minimum_contrast_count: int
    complete_evaluation_requires_all_variants: bool
    digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
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
            "minimum_required_variants": list(self.minimum_required_variants),
            "minimum_contrast_variants": list(self.minimum_contrast_variants),
            "minimum_contrast_count": self.minimum_contrast_count,
            "complete_evaluation_requires_all_variants": (
                self.complete_evaluation_requires_all_variants
            ),
        }


@dataclass(frozen=True)
class PairedVariant:
    """One auditable world in a paired counterfactual group."""

    variant_kind: str
    world: OracleWorld
    target: TransplantedDynamicObject | None
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
    base_static_occupancy: np.ndarray
    occluder_geometry: tuple[tuple[RectangleFootprint, np.ndarray], ...]
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
        "near_miss_clearance_range_m",
        "temporal_offset_candidates_s",
        "spatial_safe_clearance_range_m",
        "irrelevant_min_clearance_m",
        "lateral_offset_step_m",
        "lateral_offset_max_m",
        "minimum_required_variants",
        "minimum_contrast_variants",
        "minimum_contrast_count",
        "complete_evaluation_requires_all_variants",
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
    required = tuple(config["minimum_required_variants"])
    contrasts = tuple(config["minimum_contrast_variants"])
    if required != _REQUIRED_VARIANTS:
        raise PairedVariantConfigError(
            "minimum_required_variants must match the frozen SOP-06 policy"
        )
    if contrasts != _CONTRAST_VARIANTS:
        raise PairedVariantConfigError(
            "minimum_contrast_variants must match the frozen SOP-06 policy"
        )
    count = config["minimum_contrast_count"]
    if isinstance(count, (bool, np.bool_)) or not isinstance(
        count, (int, np.integer)
    ):
        raise PairedVariantConfigError("minimum_contrast_count must be an integer")
    count = int(count)
    if not 1 <= count <= len(contrasts):
        raise PairedVariantConfigError(
            "minimum_contrast_count must lie within the contrast variant count"
        )
    complete = config["complete_evaluation_requires_all_variants"]
    if not isinstance(complete, (bool, np.bool_)) or not bool(complete):
        raise PairedVariantConfigError(
            "complete_evaluation_requires_all_variants must be true"
        )
    normalized_payload = {
        "schema_version": SCHEMA_VERSION,
        "near_miss_clearance_range_m": list(near_miss),
        "temporal_offset_candidates_s": list(temporal),
        "spatial_safe_clearance_range_m": list(spatial_safe),
        "irrelevant_min_clearance_m": irrelevant,
        "lateral_offset_step_m": step,
        "lateral_offset_max_m": maximum,
        "minimum_required_variants": list(required),
        "minimum_contrast_variants": list(contrasts),
        "minimum_contrast_count": count,
        "complete_evaluation_requires_all_variants": True,
    }
    payload = json.dumps(
        normalized_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return PairedVariantConfig(
        schema_version=SCHEMA_VERSION,
        near_miss_clearance_range_m=near_miss,
        temporal_offset_candidates_s=temporal,
        spatial_safe_clearance_range_m=spatial_safe,
        irrelevant_min_clearance_m=irrelevant,
        lateral_offset_step_m=step,
        lateral_offset_max_m=maximum,
        minimum_required_variants=required,
        minimum_contrast_variants=contrasts,
        minimum_contrast_count=count,
        complete_evaluation_requires_all_variants=True,
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
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    critical_clearance_threshold_m: float,
) -> _PairEnvironment:
    grid = build_grid_spec(dict(base_config))
    validate_oracle_context(oracle_context, grid)
    validate_oracle_world(mother_event.world, grid)
    if mother_event.world.base_state_id != oracle_context.base_state_id:
        raise PairGenerationError("mother_context_id_mismatch")
    if mother_event.world.metadata.get("trajectory_id") != trajectory.trajectory_id:
        raise PairGenerationError("mother_trajectory_id_mismatch")
    target_id = mother_event.target.target_dynamic_object_id
    if target_id not in mother_event.world.dynamic_object_trajectories:
        raise PairGenerationError("mother_target_missing")
    if not np.array_equal(
        mother_event.world.dynamic_object_trajectories[target_id],
        mother_event.target.future_poses,
    ):
        raise PairGenerationError("mother_target_changed")
    for object_id in sorted(oracle_context.dynamic_object_future):
        if object_id not in mother_event.world.dynamic_object_trajectories:
            raise PairGenerationError("mother_context_missing")
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

    context_footprints = {
        object_id: footprint_from_spec(oracle_context.dynamic_object_specs[object_id])
        for object_id in sorted(oracle_context.dynamic_object_specs)
    }
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
    occupied = np.asarray(
        mother_event.world.static_occupancy != 0, dtype=bool
    ) | _context_current_occupancy(
        oracle_context,
        context_footprints=context_footprints,
        grid=grid,
    )
    structural = _structural_blind_spot(mother_event.world)
    if structural is None:
        visibility = raycast_visibility(
            occupied, grid, sensor_pose=np.zeros(3, dtype=np.float64)
        )
    else:
        visibility = build_structural_visibility(
            occupied,
            grid,
            sensor_pose=np.zeros(3, dtype=np.float64),
            blind_spot=structural,
        )
    return _PairEnvironment(
        grid=grid,
        future_dt_s=float(base_config["bev"]["future_dt_s"]),
        robot_footprint=robot_footprint,
        target_footprint=footprint_from_spec(mother_event.target.footprint_spec),
        context_footprints=context_footprints,
        base_static_occupancy=base_static,
        occluder_geometry=occluders,
        visibility_mask=visibility,
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
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    environment: _PairEnvironment,
) -> np.ndarray:
    all_poses = np.vstack((target.current_pose, target.future_poses))
    if (
        all_poses.shape != (environment.grid.future_steps + 1, 3)
        or all_poses.dtype != np.float32
        or not np.isfinite(all_poses).all()
    ):
        raise _CandidateRejected("target_contract_invalid")
    if not bool(np.all(points_in_grid(all_poses[:, :2], environment.grid))):
        raise _CandidateRejected("target_out_of_bounds")
    if intersects(
        environment.robot_footprint,
        np.zeros(3, dtype=np.float64),
        environment.target_footprint,
        target.current_pose,
    ):
        raise _CandidateRejected("target_current_robot_overlap")

    if environment.base_static_occupancy.any():
        sweep = rasterize_footprint_sweep(
            environment.target_footprint, all_poses, environment.grid
        )
        if np.any(sweep & environment.base_static_occupancy):
            raise _CandidateRejected("target_static_collision")
    for occluder_footprint, occluder_pose in environment.occluder_geometry:
        clearances = trajectory_signed_clearances(
            occluder_footprint,
            np.tile(occluder_pose, (all_poses.shape[0], 1)),
            environment.target_footprint,
            all_poses,
        )
        if np.any(clearances <= 0.0):
            raise _CandidateRejected("target_occluder_collision")

    for object_id in sorted(environment.context_footprints):
        context_footprint = environment.context_footprints[object_id]
        if intersects(
            environment.target_footprint,
            target.current_pose,
            context_footprint,
            oracle_context.dynamic_object_history[object_id][-1],
        ):
            raise _CandidateRejected("target_context_collision")
        clearances = trajectory_signed_clearances(
            environment.target_footprint,
            target.future_poses,
            context_footprint,
            oracle_context.dynamic_object_future[object_id],
        )
        if np.any(clearances <= 0.0):
            raise _CandidateRejected("target_context_collision")

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

    sequence = footprint_visibility_sequence(
        environment.target_footprint,
        all_poses,
        environment.visibility_mask,
        environment.grid,
    )
    if bool(sequence[0]):
        raise _CandidateRejected("target_current_visible")
    if not has_continuous_emergence(sequence, min_visible_frames=2):
        raise _CandidateRejected("target_does_not_emerge")
    if not bool(sequence[-1]):
        raise _CandidateRejected("target_final_not_visible")
    return sequence


def _pair_group_identifier(
    mother_event: GeneratedEvent,
    *,
    trajectory: LocalTrajectory,
    paired_config: PairedVariantConfig,
) -> str:
    geometry_payload = json.dumps(
        {
            "occluders": list(mother_event.world.occluders),
            "blind_spot_config": mother_event.world.blind_spot_config,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = stable_digest(
        mother_event.world.base_state_id,
        trajectory.trajectory_id,
        stable_digest(geometry_payload, size=16),
        mother_event.target.snippet_id,
        mother_event.target.target_dynamic_object_id,
        paired_config.digest,
        size=12,
    )
    return f"pair-{digest}"


def _variant_world(
    *,
    mother_event: GeneratedEvent,
    pair_group_id: str,
    variant_kind: str,
    target: TransplantedDynamicObject | None,
    visibility_sequence: np.ndarray | None,
    min_clearance_m: float | None,
    time_to_min_clearance_s: float | None,
    paired_config: PairedVariantConfig,
    seed: int,
    transform_metadata: Mapping[str, object],
    environment: _PairEnvironment,
) -> OracleWorld:
    target_id = mother_event.target.target_dynamic_object_id
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
        target_digest = stable_digest(
            target.current_pose.tobytes(), target.future_poses.tobytes(), size=16
        )
    else:
        target_digest = "target-empty"
    world_id = "world-" + stable_digest(
        pair_group_id,
        variant_kind,
        target_digest,
        int(seed),
        paired_config.digest,
        size=12,
    )
    metadata = {
        **mother_event.world.metadata,
        "schema_version": SCHEMA_VERSION,
        "pair_group_id": pair_group_id,
        "paired_variant_kind": variant_kind,
        "paired_config_digest": paired_config.digest,
        "paired_seed": int(seed),
        "target_dynamic_object_id": target_id,
        "target_present": target is not None,
        "target_current_pose": (
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
    visibility_sequence: np.ndarray | None,
    mother_event: GeneratedEvent,
    trajectory: LocalTrajectory,
    pair_group_id: str,
    paired_config: PairedVariantConfig,
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
        visibility_sequence=visibility_sequence,
        min_clearance_m=minimum,
        time_to_min_clearance_s=time_to_minimum,
        paired_config=paired_config,
        seed=seed,
        transform_metadata=transform_metadata,
        environment=environment,
    )
    return PairedVariant(
        variant_kind=variant_kind,
        world=world,
        target=target,
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
        (mother_target.current_pose, mother_target.future_poses)
    ).astype(np.float64)
    pivot = poses[0, :2].copy()
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
        current_pose=poses[0],
        future_poses=poses[1:],
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
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    paired_config: PairedVariantConfig,
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
            visibility = _validate_target_candidate(
                target,
                trajectory=trajectory,
                oracle_context=oracle_context,
                base_config=base_config,
                environment=environment,
            )
        except _CandidateRejected as exc:
            last_reason = exc.reason
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
                visibility_sequence=visibility,
                mother_event=mother_event,
                trajectory=trajectory,
                pair_group_id=pair_group_id,
                paired_config=paired_config,
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
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    paired_config: PairedVariantConfig,
    pair_group_id: str,
    seed: int,
    environment: _PairEnvironment,
) -> tuple[PairedVariant | None, str | None]:
    provenance = mother_event.target.provenance
    required = {
        "conflict_point",
        "crossing_direction",
        "time_scale",
        "target_type_policy_digest",
        "seed",
    }
    if not required <= set(provenance):
        return None, "mother_target_provenance_incomplete"
    horizon_s = environment.grid.future_steps * environment.future_dt_s
    last_reason = "temporal_offset_unavailable"
    for offset in paired_config.temporal_offset_candidates_s:
        conflict_time = mother_event.conflict_time_s + offset
        if not 0.0 < conflict_time <= horizon_s:
            last_reason = "temporal_offset_out_of_horizon"
            continue
        target = transplant_snippet(
            source_snippet,
            conflict_point=provenance["conflict_point"],
            conflict_time_s=conflict_time,
            crossing_direction=provenance["crossing_direction"],
            time_scale=provenance["time_scale"],
            future_dt_s=environment.future_dt_s,
            future_steps=environment.grid.future_steps,
            base_state_id=mother_event.world.base_state_id,
            trajectory_id=trajectory.trajectory_id,
            target_type_policy_digest=provenance["target_type_policy_digest"],
            seed=provenance["seed"],
            context_object_ids=tuple(oracle_context.dynamic_object_future),
        )
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
            visibility = _validate_target_candidate(
                target,
                trajectory=trajectory,
                oracle_context=oracle_context,
                base_config=base_config,
                environment=environment,
            )
        except _CandidateRejected as exc:
            last_reason = exc.reason
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
                visibility_sequence=visibility,
                mother_event=mother_event,
                trajectory=trajectory,
                pair_group_id=pair_group_id,
                paired_config=paired_config,
                seed=seed,
                environment=environment,
                transform_metadata=transform_metadata,
                temporal_offset_s=float(offset),
            ),
            None,
        )
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
    missing_required = set(normalized.minimum_required_variants) - set(variants)
    if missing_required:
        raise PairGenerationError(
            "minimum_required_variants_missing",
            f"minimum required variants missing: {sorted(missing_required)}",
        )
    contrast_count = sum(
        kind in variants for kind in normalized.minimum_contrast_variants
    )
    if contrast_count < normalized.minimum_contrast_count:
        raise PairGenerationError(
            "minimum_contrast_missing",
            "minimum contrast coverage is not satisfied",
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
            complete and normalized.complete_evaluation_requires_all_variants
        ),
        paired_config_digest=normalized.digest,
    )
    return _stamp_group_metadata(group)


def generate_paired_variants(
    *,
    mother_event: GeneratedEvent,
    source_snippet: MotionSnippet,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    paired_config: PairedVariantConfig | Mapping[str, Any],
    seed: int,
) -> PairedEventGroup:
    """Generate one six-position group, retaining policy-compliant partials."""

    if not isinstance(mother_event, GeneratedEvent):
        raise TypeError("mother_event must be a GeneratedEvent")
    if not isinstance(trajectory, LocalTrajectory):
        raise TypeError("trajectory must be a LocalTrajectory")
    if not isinstance(oracle_context, OracleContext):
        raise TypeError("oracle_context must be an OracleContext")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")
    normalized = _as_paired_config(paired_config)
    _validate_source_snippet(mother_event, source_snippet)
    environment = _pair_environment(
        mother_event=mother_event,
        trajectory=trajectory,
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
        mother_visibility = _validate_target_candidate(
            mother_event.target,
            trajectory=trajectory,
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
        visibility_sequence=mother_visibility,
        mother_event=mother_event,
        trajectory=trajectory,
        pair_group_id=pair_id,
        paired_config=normalized,
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
            oracle_context=oracle_context,
            base_config=base_config,
            paired_config=normalized,
            pair_group_id=pair_id,
            seed=int(seed),
            environment=environment,
            candidate_pool=candidate_pool,
        )
        if variant is None:
            missing[kind] = str(reason)
        else:
            variants[kind] = variant

    temporal, reason = _temporal_variant(
        mother_event=mother_event,
        source_snippet=source_snippet,
        trajectory=trajectory,
        oracle_context=oracle_context,
        base_config=base_config,
        paired_config=normalized,
        pair_group_id=pair_id,
        seed=int(seed),
        environment=environment,
    )
    if temporal is None:
        missing["temporal_safe"] = str(reason)
    else:
        variants["temporal_safe"] = temporal

    variants["empty_blind_spot"] = _make_variant(
        variant_kind="empty_blind_spot",
        target=None,
        visibility_sequence=None,
        mother_event=mother_event,
        trajectory=trajectory,
        pair_group_id=pair_id,
        paired_config=normalized,
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
) -> TransplantedDynamicObject | None:
    provenance = mother_event.target.provenance
    required = {
        "conflict_point",
        "crossing_direction",
        "time_scale",
        "target_type_policy_digest",
        "seed",
    }
    if not required <= set(provenance):
        raise PairGenerationError("mother_target_provenance_incomplete")
    conflict_time = mother_event.conflict_time_s + offset_s
    horizon_s = environment.grid.future_steps * environment.future_dt_s
    if not 0.0 < conflict_time <= horizon_s:
        return None
    target = transplant_snippet(
        source_snippet,
        conflict_point=provenance["conflict_point"],
        conflict_time_s=conflict_time,
        crossing_direction=provenance["crossing_direction"],
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
    return replace(
        target,
        target_dynamic_object_id=mother_event.target.target_dynamic_object_id,
        provenance={
            **target.provenance,
            "paired_transform": {
                "kind": "temporal_offset_v1",
                "temporal_offset_s": float(offset_s),
                "mother_conflict_time_s": mother_event.conflict_time_s,
            },
        },
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
        "generator_algorithm_version": _JOINT_ENVIRONMENT_PAIR_VERSION,
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
    context_footprints = {
        object_id: footprint_from_spec(
            oracle_context.dynamic_object_specs[object_id]
        )
        for object_id in sorted(oracle_context.dynamic_object_specs)
    }
    context_trajectories = {
        object_id: np.vstack(
            (
                oracle_context.dynamic_object_history[object_id][-1],
                oracle_context.dynamic_object_future[object_id],
            )
        )
        for object_id in sorted(context_footprints)
    }

    rejection_reasons: dict[str, int] = {}
    mother_accepted_count = 0
    pair_candidate_count = 0
    occluder_candidate_count = 0
    attempted_count = 0
    search_round = 0
    while attempted_count < max_attempts:
        round_index = search_round
        search_round += 1
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
        mother_candidate_budget = min(
            _JOINT_ENVIRONMENT_MOTHER_BATCH_SIZE,
            max_attempts - attempted_count,
        )
        candidate_generator = {
            **normalized_generator,
            "event_type_weights": {
                "environment": 1.0,
                "structural": 0.0,
                "mixed": 0.0,
            },
            "conflict_time_range_s": (anchor, anchor),
            "max_resample_attempts": mother_candidate_budget,
        }
        event_report = generate_events(
            base_state=base_state,
            oracle_context=oracle_context,
            trajectory=trajectory,
            snippet_libraries=snippet_libraries,
            base_config=base_config,
            generator_config=candidate_generator,
            seed=event_seed,
            event_count=1,
        )
        attempted_count += int(event_report.summary["attempted_count"])
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
            temporal_target = _retimed_target_candidate(
                mother_event=mother,
                source_snippet=source_snippet,
                trajectory=trajectory,
                oracle_context=oracle_context,
                environment=environment,
                offset_s=offset_s,
            )
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
            try:
                placement, visibility_sequences = (
                    align_environment_occluder_to_target_los_envelope(
                        occluder_type=str(original_occluder["type"]),
                        normal_offset_m=float(
                            original_occluder["normal_offset_m"]
                        ),
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
                        robot_poses=trajectory.poses,
                        robot_footprint=environment.robot_footprint,
                        target_pose_sequences=(
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
                        context_trajectories=context_trajectories,
                        context_footprints=context_footprints,
                        config=normalized_generator["occluders"],
                        min_contiguous_visible_frames=int(
                            normalized_generator[
                                "min_contiguous_visible_frames"
                            ]
                        ),
                    )
                )
            except OccluderSamplingError as exc:
                occluder_candidate_count += exc.attempts
                detailed_reasons = exc.rejection_reasons or {exc.reason: 1}
                for reason, count in detailed_reasons.items():
                    key = f"occluder:{reason}"
                    rejection_reasons[key] = rejection_reasons.get(
                        key, 0
                    ) + int(count)
                continue
            occluder_candidate_count += placement.attempt
            for reason, count in placement.rejection_reasons.items():
                key = f"occluder:{reason}"
                rejection_reasons[key] = rejection_reasons.get(key, 0) + int(
                    count
                )
            world_id = "world-" + stable_digest(
                mother.world.world_id,
                _JOINT_ENVIRONMENT_PAIR_VERSION,
                placement.occluder["occluder_id"],
                event_seed,
                size=12,
            )
            metadata = {
                **mother.world.metadata,
                "joint_pair_generator_algorithm_version": (
                    _JOINT_ENVIRONMENT_PAIR_VERSION
                ),
                "joint_pair_attempt_index": attempted_count - 1,
                "joint_pair_anchor_schedule_s": list(anchor_schedule),
                "joint_pair_temporal_offset_s": float(offset_s),
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
            updated_mother = replace(
                mother,
                world=updated_world,
                visibility_sequence=visibility_sequences[0].copy(),
            )
            try:
                group = generate_paired_variants(
                    mother_event=updated_mother,
                    source_snippet=source_snippet,
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
