"""Audit-only joint occluder search for complete seen-occluded sixpacks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from src.contracts import BaseState, LocalTrajectory, OracleContext
from src.datasets.snippet_library import MotionSnippet
from src.generation.dynamic_object_transplant import TransplantedDynamicObject
from src.generation.event_sampler import GeneratedEvent
from src.generation.history_visibility import (
    HISTORY_VISIBILITY_POLICY_VERSION,
    SEEN_THEN_OCCLUDED,
    classify_history_visibility,
    normalize_history_visibility_policy,
)
from src.generation.occluder_sampler import (
    PreparedOccluderCollisionSweep,
    occluder_collision_sweep_rejection_reason,
    prepare_occluder_collision_sweep,
)
from src.generation.paired_variants import (
    PairGenerationError,
    PairedEventGroup,
    PairedVariantConfig,
    generate_paired_variants,
)
import src.generation.paired_variants as paired_variants
from src.generation.structural_blindspot import (
    footprint_visibility_sequence,
    has_continuous_emergence,
)
from src.geometry import (
    RectangleFootprint,
    footprint_vertices,
    points_in_grid,
    rasterize_footprint,
    raycast_visibility,
    wrap_angle,
)
from src.utils.seeding import derive_seed, stable_digest


JOINT_AUDIT_ALGORITHM_VERSION = "seen_occluded_joint_visual_audit_v4"
JOINT_AUDIT_CONFIG_SCHEMA_VERSION = "2.0.0"
_OCCLUDER_TYPES = ("wall", "shelf", "pillar")
_TYPE_TOKENS = (*_OCCLUDER_TYPES, "source")


class JointAuditSearchError(ValueError):
    """Raised when joint-search inputs violate the audit contract."""


@dataclass(frozen=True)
class JointAuditSearchConfig:
    schema_version: str
    algorithm_version: str
    shared_los_fractions: tuple[float, ...]
    center_alphas: tuple[float, ...]
    length_quantiles: tuple[float, ...]
    width_quantiles: tuple[float, ...]
    longitudinal_center_quantiles: tuple[float, ...]
    occluder_type_order: tuple[str, ...]
    digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "shared_los_fractions": list(self.shared_los_fractions),
            "center_alphas": list(self.center_alphas),
            "length_quantiles": list(self.length_quantiles),
            "width_quantiles": list(self.width_quantiles),
            "longitudinal_center_quantiles": list(
                self.longitudinal_center_quantiles
            ),
            "occluder_type_order": list(self.occluder_type_order),
        }


@dataclass(frozen=True)
class JointTemporalCandidate:
    offset_s: float
    target: TransplantedDynamicObject
    prepared_sweeps: tuple[PreparedOccluderCollisionSweep, ...]
    trajectory_normal: np.ndarray
    normal_coordinates_m: tuple[float, ...]


@dataclass(frozen=True)
class JointOccluderPlacement:
    mask: np.ndarray
    occluder: dict[str, object]


@dataclass(frozen=True)
class JointAuditSearchResult:
    mother_event: GeneratedEvent | None
    group: PairedEventGroup | None
    paired_seed: int | None
    summary: dict[str, object]

    @property
    def complete(self) -> bool:
        return (
            self.mother_event is not None
            and self.group is not None
            and self.paired_seed is not None
        )


def _finite_quantile_sequence(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise JointAuditSearchError(f"{name} must be a non-empty sequence")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
            raise JointAuditSearchError(f"{name}[{index}] must be a finite real")
        parsed = float(item)
        if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise JointAuditSearchError(f"{name}[{index}] must lie in [0, 1]")
        result.append(parsed)
    if len(set(result)) != len(result):
        raise JointAuditSearchError(f"{name} must not contain duplicates")
    return tuple(result)


def normalize_joint_audit_search_config(
    value: Mapping[str, Any],
) -> JointAuditSearchConfig:
    """Strictly normalize the compact audit-only joint search config."""

    if not isinstance(value, Mapping):
        raise JointAuditSearchError("joint audit config must be a mapping")
    expected = {
        "schema_version",
        "algorithm_version",
        "shared_los_fractions",
        "center_alphas",
        "length_quantiles",
        "width_quantiles",
        "longitudinal_center_quantiles",
        "occluder_type_order",
    }
    if set(value) != expected:
        raise JointAuditSearchError("joint audit config keys mismatch")
    if value["schema_version"] != JOINT_AUDIT_CONFIG_SCHEMA_VERSION:
        raise JointAuditSearchError("joint audit config schema version mismatch")
    if value["algorithm_version"] != JOINT_AUDIT_ALGORITHM_VERSION:
        raise JointAuditSearchError("joint audit algorithm version mismatch")
    fractions = _finite_quantile_sequence(
        value["shared_los_fractions"], name="shared_los_fractions"
    )
    if any(item <= 0.0 or item >= 1.0 for item in fractions):
        raise JointAuditSearchError("shared_los_fractions must lie in (0, 1)")
    alphas = _finite_quantile_sequence(value["center_alphas"], name="center_alphas")
    lengths = _finite_quantile_sequence(
        value["length_quantiles"], name="length_quantiles"
    )
    widths = _finite_quantile_sequence(
        value["width_quantiles"], name="width_quantiles"
    )
    longitudinal_centers = _finite_quantile_sequence(
        value["longitudinal_center_quantiles"],
        name="longitudinal_center_quantiles",
    )
    type_order = value["occluder_type_order"]
    if (
        not isinstance(type_order, (list, tuple))
        or not type_order
        or len(set(type_order)) != len(type_order)
        or any(item not in _TYPE_TOKENS for item in type_order)
        or "source" not in type_order
    ):
        raise JointAuditSearchError("occluder_type_order is invalid")
    payload = {
        "schema_version": JOINT_AUDIT_CONFIG_SCHEMA_VERSION,
        "algorithm_version": JOINT_AUDIT_ALGORITHM_VERSION,
        "shared_los_fractions": list(fractions),
        "center_alphas": list(alphas),
        "length_quantiles": list(lengths),
        "width_quantiles": list(widths),
        "longitudinal_center_quantiles": list(longitudinal_centers),
        "occluder_type_order": list(type_order),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return JointAuditSearchConfig(
        schema_version=JOINT_AUDIT_CONFIG_SCHEMA_VERSION,
        algorithm_version=JOINT_AUDIT_ALGORITHM_VERSION,
        shared_los_fractions=fractions,
        center_alphas=alphas,
        length_quantiles=lengths,
        width_quantiles=widths,
        longitudinal_center_quantiles=longitudinal_centers,
        occluder_type_order=tuple(type_order),
        digest=stable_digest(canonical, size=16),
    )


def load_joint_audit_search_config(path: str | Path) -> JointAuditSearchConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return normalize_joint_audit_search_config(raw)


def _pose(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (3,) or array.dtype.kind not in "iuf":
        raise JointAuditSearchError(f"{name} must have shape (3,)")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise JointAuditSearchError(f"{name} contains NaN/Inf")
    return result


def shared_los_normal_offsets(
    *,
    collision_current_pose: object,
    temporal_current_pose: object,
    trajectory_normal: object,
    source_normal_coordinate_m: float,
    fractions: Iterable[float],
) -> tuple[float, ...]:
    """Return stable normal coordinates strictly inside both current LOS rays."""

    collision = _pose(collision_current_pose, name="collision_current_pose")
    temporal = _pose(temporal_current_pose, name="temporal_current_pose")
    normal = np.asarray(trajectory_normal, dtype=np.float64)
    if normal.shape != (2,) or not np.isfinite(normal).all():
        raise JointAuditSearchError("trajectory_normal must have shape (2,)")
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        raise JointAuditSearchError("trajectory_normal must be non-zero")
    normal /= norm
    coordinates = np.asarray(
        [np.dot(collision[:2], normal), np.dot(temporal[:2], normal)],
        dtype=np.float64,
    )
    if coordinates[0] * coordinates[1] <= 0.0:
        return ()
    parsed_fractions = _finite_quantile_sequence(tuple(fractions), name="fractions")
    if any(value <= 0.0 or value >= 1.0 for value in parsed_fractions):
        raise JointAuditSearchError("fractions must lie in (0, 1)")
    sign = -1.0 if coordinates[0] < 0.0 else 1.0
    extent = float(np.min(np.abs(coordinates)))
    source = float(source_normal_coordinate_m)
    if not np.isfinite(source):
        raise JointAuditSearchError(
            "source_normal_coordinate_m must be finite"
        )
    values = []
    if 0.0 < sign * source < extent:
        values.append(source)
    values.extend(sign * extent * fraction for fraction in parsed_fractions)
    return tuple(dict.fromkeys(float(round(value, 12)) for value in values))


def _range_quantile(value: object, quantile: float, *, name: str) -> float:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise JointAuditSearchError(f"{name} must contain two values")
    lower, upper = (float(item) for item in value)
    if not np.isfinite([lower, upper]).all() or lower <= 0.0 or lower > upper:
        raise JointAuditSearchError(f"{name} is invalid")
    return float(lower + quantile * (upper - lower))


def longitudinal_center_candidates(
    *,
    base_center: object,
    intersections: object,
    yaw: float,
    length_m: float,
    quantiles: Iterable[float],
) -> tuple[np.ndarray, ...]:
    """Shift a center along its long axis while covering both LOS points."""

    center = np.asarray(base_center, dtype=np.float64)
    points = np.asarray(intersections, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise JointAuditSearchError("base_center must be a finite 2-vector")
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise JointAuditSearchError(
            "intersections must be a finite (2, 2) array"
        )
    parsed_yaw = float(yaw)
    parsed_length = float(length_m)
    if not np.isfinite(parsed_yaw):
        raise JointAuditSearchError("yaw must be finite")
    if not np.isfinite(parsed_length) or parsed_length <= 0.0:
        raise JointAuditSearchError("length_m must be positive and finite")
    parsed_quantiles = _finite_quantile_sequence(
        tuple(quantiles), name="quantiles"
    )
    axis = np.asarray(
        [np.cos(parsed_yaw), np.sin(parsed_yaw)], dtype=np.float64
    )
    projections = points @ axis
    half_length = 0.5 * parsed_length
    lower = float(np.max(projections) - half_length)
    upper = float(np.min(projections) + half_length)
    candidates = [center.copy()]
    if lower > upper:
        return tuple(candidates)
    base_projection = float(np.dot(center, axis))
    seen = {tuple(float(round(value, 12)) for value in center)}
    for quantile in parsed_quantiles:
        desired_projection = lower + quantile * (upper - lower)
        candidate = center + (desired_projection - base_projection) * axis
        key = tuple(float(round(value, 12)) for value in candidate)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return tuple(candidates)


def _effective_occluder_types(
    source_type: str,
    config: JointAuditSearchConfig,
) -> tuple[str, ...]:
    values = (
        source_type if token == "source" else token
        for token in config.occluder_type_order
    )
    return tuple(dict.fromkeys(values))


def _iter_joint_temporal_candidates(
    *,
    mother_event: GeneratedEvent,
    source_snippet: MotionSnippet,
    base_state: BaseState,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    environment,
    paired_config: PairedVariantConfig,
    joint_config: JointAuditSearchConfig,
    diagnostics: Counter[str],
) -> Iterable[JointTemporalCandidate]:
    normal = paired_variants._event_trajectory_normal(mother_event, trajectory)
    source_occluder = mother_event.world.occluders[0]
    source_pose = np.asarray(source_occluder["pose"], dtype=np.float64)
    source_normal_coordinate = float(np.dot(source_pose[:2], normal))
    for offset_s in paired_config.temporal_offset_candidates_s:
        target, reason = paired_variants._retimed_target_candidate(
            mother_event=mother_event,
            source_snippet=source_snippet,
            trajectory=trajectory,
            oracle_context=oracle_context,
            environment=environment,
            offset_s=offset_s,
        )
        if target is None:
            diagnostics[f"temporal:{reason or 'out_of_horizon'}"] += 1
            continue
        clearances = paired_variants._target_clearances(
            target, trajectory=trajectory, environment=environment
        )
        if np.any(clearances <= 0.0):
            diagnostics["temporal:still_collides"] += 1
            continue
        if not paired_variants._paths_spatially_intersect(
            trajectory=trajectory,
            target=target,
            environment=environment,
        ):
            diagnostics["temporal:paths_disjoint"] += 1
            continue
        normal_coordinates = shared_los_normal_offsets(
            collision_current_pose=mother_event.target.current_pose,
            temporal_current_pose=target.current_pose,
            trajectory_normal=normal,
            source_normal_coordinate_m=source_normal_coordinate,
            fractions=joint_config.shared_los_fractions,
        )
        if not normal_coordinates:
            diagnostics["temporal:no_shared_los_halfspace"] += 1
            continue
        raw_sweeps = paired_variants._joint_occluder_collision_sweeps(
            base_state=base_state,
            trajectory=trajectory,
            oracle_context=oracle_context,
            environment=environment,
            collision_target=mother_event.target,
            temporal_target=target,
        )
        prepared_sweeps = tuple(
            prepare_occluder_collision_sweep(sweep, grid=environment.grid)
            for sweep in raw_sweeps
        )
        diagnostics["temporal:candidate"] += 1
        yield JointTemporalCandidate(
            offset_s=float(offset_s),
            target=target,
            prepared_sweeps=prepared_sweeps,
            trajectory_normal=normal.copy(),
            normal_coordinates_m=normal_coordinates,
        )


def _current_context_occupancy(environment) -> np.ndarray:
    result = np.zeros_like(environment.base_static_occupancy, dtype=np.bool_)
    for object_id in sorted(environment.scene_history_footprints):
        result |= rasterize_footprint(
            environment.scene_history_footprints[object_id],
            environment.scene_dynamic_history[object_id][-1],
            environment.grid,
        )
    return result


def _iter_certified_joint_placements(
    *,
    mother_event: GeneratedEvent,
    temporal_candidate: JointTemporalCandidate,
    environment,
    generator_config: Mapping[str, Any],
    joint_config: JointAuditSearchConfig,
    diagnostics: Counter[str],
) -> Iterable[JointOccluderPlacement]:
    source_type = str(mother_event.world.occluders[0]["type"])
    enabled_types = tuple(generator_config["occluders"]["types"])
    type_order = tuple(
        value
        for value in _effective_occluder_types(source_type, joint_config)
        if value in enabled_types
    )
    if not type_order:
        raise JointAuditSearchError("joint occluder type schedule is empty")
    context_occupancy = _current_context_occupancy(environment)
    targets = (mother_event.target, temporal_candidate.target)
    for type_index, occluder_type in enumerate(type_order):
        dimensions = generator_config["occluders"][occluder_type]
        for coordinate_index, normal_coordinate_m in enumerate(
            temporal_candidate.normal_coordinates_m
        ):
            desired_coordinate = float(normal_coordinate_m)
            intersections = []
            directions = []
            for target in targets:
                line_of_sight = target.current_pose[:2].astype(np.float64)
                distance = float(np.linalg.norm(line_of_sight))
                denominator = float(
                    np.dot(line_of_sight, temporal_candidate.trajectory_normal)
                )
                if distance <= 1e-9 or abs(denominator) <= 1e-9:
                    diagnostics["placement:los_degenerate"] += 1
                    intersections = []
                    break
                fraction = desired_coordinate / denominator
                if not 0.0 < fraction < 1.0:
                    diagnostics["placement:offset_outside_los"] += 1
                    intersections = []
                    break
                intersections.append(fraction * line_of_sight)
                directions.append(line_of_sight / distance)
            if len(intersections) != 2:
                continue
            mean_direction = np.sum(directions, axis=0)
            if float(np.linalg.norm(mean_direction)) <= 1e-9:
                diagnostics["placement:mean_direction_degenerate"] += 1
                continue
            mean_direction /= np.linalg.norm(mean_direction)
            yaw_candidates = tuple(
                dict.fromkeys(
                    float(
                        wrap_angle(
                            np.arctan2(direction[1], direction[0])
                            + 0.5 * np.pi
                        )
                    )
                    for direction in (mean_direction, *directions)
                )
            )
            for length_quantile in joint_config.length_quantiles:
                length_m = _range_quantile(
                    dimensions["length_range_m"],
                    length_quantile,
                    name=f"{occluder_type}.length_range_m",
                )
                for width_quantile in joint_config.width_quantiles:
                    width_m = _range_quantile(
                        dimensions["width_range_m"],
                        width_quantile,
                        name=f"{occluder_type}.width_range_m",
                    )
                    footprint = RectangleFootprint(length_m, width_m)
                    for yaw_index, yaw in enumerate(yaw_candidates):
                        seen_centers: set[tuple[float, float]] = set()
                        for alpha_index, alpha in enumerate(
                            joint_config.center_alphas
                        ):
                            base_center = (
                                alpha * intersections[0]
                                + (1.0 - alpha) * intersections[1]
                            )
                            centers = longitudinal_center_candidates(
                                base_center=base_center,
                                intersections=intersections,
                                yaw=yaw,
                                length_m=length_m,
                                quantiles=(
                                    joint_config.longitudinal_center_quantiles
                                ),
                            )
                            base_center_certified = False
                            for center_index, center in enumerate(centers):
                                center_key = tuple(
                                    float(round(value, 12)) for value in center
                                )
                                if center_key in seen_centers:
                                    diagnostics[
                                        "placement:duplicate_center"
                                    ] += 1
                                    if center_index == 0:
                                        break
                                    continue
                                seen_centers.add(center_key)
                                diagnostics["placement:candidate"] += 1
                                pose = np.asarray(
                                    [center[0], center[1], yaw],
                                    dtype=np.float64,
                                )
                                if not bool(
                                    np.all(
                                        points_in_grid(
                                            footprint_vertices(footprint, pose),
                                            environment.grid,
                                        )
                                    )
                                ):
                                    diagnostics["placement:out_of_grid"] += 1
                                    if center_index == 0:
                                        break
                                    continue
                                mask = rasterize_footprint(
                                    footprint, pose, environment.grid
                                )
                                if np.any(
                                    mask & environment.base_static_occupancy
                                ):
                                    diagnostics["placement:static_overlap"] += 1
                                    if center_index == 0:
                                        break
                                    continue
                                reason = (
                                    occluder_collision_sweep_rejection_reason(
                                        footprint,
                                        pose,
                                        temporal_candidate.prepared_sweeps,
                                        grid=environment.grid,
                                    )
                                )
                                if reason is not None:
                                    diagnostics[f"placement:{reason}"] += 1
                                    if center_index == 0:
                                        break
                                    continue
                                visibility = raycast_visibility(
                                    environment.base_static_occupancy
                                    | context_occupancy
                                    | mask,
                                    environment.grid,
                                    sensor_pose=np.zeros(3, dtype=np.float64),
                                )
                                sequences = tuple(
                                    footprint_visibility_sequence(
                                        environment.target_footprint,
                                        np.vstack(
                                            (
                                                target.current_pose,
                                                target.future_poses,
                                            )
                                        ),
                                        visibility,
                                        environment.grid,
                                    )
                                    for target in targets
                                )
                                if any(
                                    bool(sequence[0])
                                    or not bool(sequence[-1])
                                    or not has_continuous_emergence(
                                        sequence, min_visible_frames=2
                                    )
                                    for sequence in sequences
                                ):
                                    diagnostics[
                                        "placement:visibility_invalid"
                                    ] += 1
                                    if center_index == 0:
                                        break
                                    continue
                                pose32 = pose.astype(np.float32)
                                schedule_rank = (
                                    type_index,
                                    coordinate_index,
                                    length_quantile,
                                    width_quantile,
                                    yaw_index,
                                    alpha_index,
                                    center_index,
                                )
                                occluder_id = "occluder-" + stable_digest(
                                    joint_config.algorithm_version,
                                    joint_config.digest,
                                    mother_event.generated_event_id,
                                    *schedule_rank,
                                    *(f"{item:.9f}" for item in pose32),
                                    f"{length_m:.9f}",
                                    f"{width_m:.9f}",
                                    size=12,
                                )
                                diagnostics["placement:certified"] += 1
                                if center_index == 0:
                                    base_center_certified = True
                                axis = np.asarray(
                                    [np.cos(yaw), np.sin(yaw)],
                                    dtype=np.float64,
                                )
                                yield JointOccluderPlacement(
                                    mask=mask.copy(),
                                    occluder={
                                        "occluder_id": occluder_id,
                                        "type": occluder_type,
                                        "pose": [
                                            float(item) for item in pose32
                                        ],
                                        "length_m": length_m,
                                        "width_m": width_m,
                                        "geometry_source": "generator_config",
                                        "placement_strategy": (
                                            joint_config.algorithm_version
                                        ),
                                        "joint_config_digest": (
                                            joint_config.digest
                                        ),
                                        "los_normal_coordinate_m": (
                                            normal_coordinate_m
                                        ),
                                        "schedule_rank": list(schedule_rank),
                                        "hidden_los_count": 2,
                                        "center_alpha": alpha,
                                        "longitudinal_shift_m": float(
                                            np.dot(center - base_center, axis)
                                        ),
                                    },
                                )
                            if not base_center_certified:
                                continue


def _rebind_audit_mother(
    *,
    mother_event: GeneratedEvent,
    placement: JointOccluderPlacement,
    temporal_offset_s: float,
    base_state: BaseState,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    paired_config: PairedVariantConfig,
    joint_config: JointAuditSearchConfig,
    environment,
) -> GeneratedEvent:
    world_id = "world-" + stable_digest(
        mother_event.world.world_id,
        joint_config.algorithm_version,
        joint_config.digest,
        placement.occluder["occluder_id"],
        size=12,
    )
    blind_spot_config = dict(mother_event.world.blind_spot_config)
    blind_spot_config["occluder_ids"] = [placement.occluder["occluder_id"]]
    candidate_world = replace(
        mother_event.world,
        world_id=world_id,
        static_occupancy=(
            environment.base_static_occupancy | placement.mask
        ).astype(np.float32),
        occluders=(dict(placement.occluder),),
        blind_spot_config=blind_spot_config,
        metadata={
            **mother_event.world.metadata,
            "audit_only": True,
            "audit_joint_occluder_algorithm_version": (
                joint_config.algorithm_version
            ),
            "audit_joint_occluder_config_digest": joint_config.digest,
            "audit_source_world_id": mother_event.world.world_id,
            "audit_joint_temporal_offset_s": temporal_offset_s,
        },
    )
    rebound = paired_variants._rebind_mother_world(
        mother_event,
        candidate_world,
        base_state=base_state,
        trajectory=trajectory,
        oracle_context=oracle_context,
        base_config=base_config,
        critical_clearance_threshold_m=(
            paired_config.near_miss_clearance_range_m[1]
        ),
        grid=environment.grid,
    )
    policy = normalize_history_visibility_policy(
        mother_event.world.metadata["target_history_visibility_policy"]
    )
    assessment = classify_history_visibility(
        rebound.target_visibility_history, policy
    )
    return replace(
        rebound,
        world=replace(
            rebound.world,
            metadata={
                **rebound.world.metadata,
                "target_history_visibility_regime": assessment.regime,
                "target_history_last_visible_index": assessment.last_visible_index,
                "target_history_trailing_hidden_frames": (
                    assessment.trailing_hidden_frames
                ),
                "target_history_visibility_policy_version": (
                    HISTORY_VISIBILITY_POLICY_VERSION
                ),
                "target_history_visibility_policy": policy.as_dict(),
                "target_history_visibility_policy_digest": policy.digest,
            },
        ),
    )


def _summary(
    *,
    source_world_id: str,
    joint_config: JointAuditSearchConfig,
    diagnostics: Counter[str],
    history_rejections: Counter[str],
    selected_offset_s: float | None,
    selected_occluder_id: str | None,
    paired_seed: int | None,
    complete: bool,
) -> dict[str, object]:
    return {
        "algorithm_version": joint_config.algorithm_version,
        "config_digest": joint_config.digest,
        "source_world_id": source_world_id,
        "complete": complete,
        "selected_temporal_offset_s": selected_offset_s,
        "selected_occluder_id": selected_occluder_id,
        "paired_seed": paired_seed,
        "history_regime_rejections": dict(sorted(history_rejections.items())),
        "rejection_counts": dict(sorted(diagnostics.items())),
    }


def _stamp_audit_metadata(
    event: GeneratedEvent,
    *,
    source_world_id: str,
    joint_config: JointAuditSearchConfig,
    temporal_offset_s: float,
    paired_seed: int,
    summary: Mapping[str, object],
) -> GeneratedEvent:
    return replace(
        event,
        world=replace(
            event.world,
            metadata={
                **event.world.metadata,
                "audit_only": True,
                "audit_joint_occluder_algorithm_version": (
                    joint_config.algorithm_version
                ),
                "audit_joint_occluder_config_digest": joint_config.digest,
                "audit_source_world_id": source_world_id,
                "audit_joint_temporal_offset_s": temporal_offset_s,
                "audit_joint_paired_seed": paired_seed,
                "audit_joint_search_summary": dict(summary),
            },
        ),
    )


def search_joint_audit_group(
    *,
    mother_event: GeneratedEvent,
    source_snippet: MotionSnippet,
    base_state: BaseState,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    base_config: Mapping[str, Any],
    generator_config: Mapping[str, Any],
    paired_config: PairedVariantConfig,
    joint_config: JointAuditSearchConfig,
    pair_seed: int,
) -> JointAuditSearchResult:
    """Jointly replace one audit occluder and require a formal complete group."""

    if not isinstance(joint_config, JointAuditSearchConfig):
        raise TypeError("joint_config must be JointAuditSearchConfig")
    if not isinstance(pair_seed, int) or isinstance(pair_seed, bool):
        raise TypeError("pair_seed must be an integer")
    if len(mother_event.world.occluders) != 1:
        raise JointAuditSearchError("joint audit mother must have one occluder")
    environment = paired_variants._pair_environment(
        mother_event=mother_event,
        trajectory=trajectory,
        base_state=base_state,
        oracle_context=oracle_context,
        base_config=base_config,
        critical_clearance_threshold_m=(
            paired_config.near_miss_clearance_range_m[1]
        ),
    )
    policy = normalize_history_visibility_policy(
        mother_event.world.metadata["target_history_visibility_policy"]
    )
    source_world_id = mother_event.world.world_id
    diagnostics: Counter[str] = Counter()
    history_rejections: Counter[str] = Counter()
    temporal_candidates = _iter_joint_temporal_candidates(
        mother_event=mother_event,
        source_snippet=source_snippet,
        base_state=base_state,
        trajectory=trajectory,
        oracle_context=oracle_context,
        environment=environment,
        paired_config=paired_config,
        joint_config=joint_config,
        diagnostics=diagnostics,
    )
    for temporal in temporal_candidates:
        placements = _iter_certified_joint_placements(
            mother_event=mother_event,
            temporal_candidate=temporal,
            environment=environment,
            generator_config=generator_config,
            joint_config=joint_config,
            diagnostics=diagnostics,
        )
        for placement in placements:
            try:
                rebound = _rebind_audit_mother(
                    mother_event=mother_event,
                    placement=placement,
                    temporal_offset_s=temporal.offset_s,
                    base_state=base_state,
                    trajectory=trajectory,
                    oracle_context=oracle_context,
                    base_config=base_config,
                    paired_config=paired_config,
                    joint_config=joint_config,
                    environment=environment,
                )
            except PairGenerationError as exc:
                diagnostics[f"rebind:{exc.reason}"] += 1
                continue
            assessment = classify_history_visibility(
                rebound.target_visibility_history, policy
            )
            if assessment.regime != SEEN_THEN_OCCLUDED:
                history_rejections[assessment.regime] += 1
                continue
            occluder_id = str(placement.occluder["occluder_id"])
            paired_seed = derive_seed(
                pair_seed,
                joint_config.algorithm_version,
                joint_config.digest,
                occluder_id,
            )
            try:
                probe_group = generate_paired_variants(
                    mother_event=rebound,
                    source_snippet=source_snippet,
                    base_state=base_state,
                    trajectory=trajectory,
                    oracle_context=oracle_context,
                    base_config=base_config,
                    paired_config=paired_config,
                    seed=paired_seed,
                )
            except PairGenerationError as exc:
                diagnostics[f"pair:{exc.reason}"] += 1
                continue
            if (
                not probe_group.is_complete
                or not probe_group.eligible_for_strict_evaluation
            ):
                for kind, reason in probe_group.missing_variant_reasons.items():
                    diagnostics[f"pair:{kind}:{reason}"] += 1
                continue
            final_summary = _summary(
                source_world_id=source_world_id,
                joint_config=joint_config,
                diagnostics=diagnostics,
                history_rejections=history_rejections,
                selected_offset_s=temporal.offset_s,
                selected_occluder_id=occluder_id,
                paired_seed=paired_seed,
                complete=True,
            )
            rebound = _stamp_audit_metadata(
                rebound,
                source_world_id=source_world_id,
                joint_config=joint_config,
                temporal_offset_s=temporal.offset_s,
                paired_seed=paired_seed,
                summary=final_summary,
            )
            try:
                group = generate_paired_variants(
                    mother_event=rebound,
                    source_snippet=source_snippet,
                    base_state=base_state,
                    trajectory=trajectory,
                    oracle_context=oracle_context,
                    base_config=base_config,
                    paired_config=paired_config,
                    seed=paired_seed,
                )
            except PairGenerationError as exc:
                diagnostics[f"final_pair:{exc.reason}"] += 1
                continue
            if not group.is_complete or not group.eligible_for_strict_evaluation:
                for kind, reason in group.missing_variant_reasons.items():
                    diagnostics[f"final_pair:{kind}:{reason}"] += 1
                continue
            return JointAuditSearchResult(
                mother_event=rebound,
                group=group,
                paired_seed=paired_seed,
                summary=final_summary,
            )
    return JointAuditSearchResult(
        mother_event=None,
        group=None,
        paired_seed=None,
        summary=_summary(
            source_world_id=source_world_id,
            joint_config=joint_config,
            diagnostics=diagnostics,
            history_rejections=history_rejections,
            selected_offset_s=None,
            selected_occluder_id=None,
            paired_seed=None,
            complete=False,
        ),
    )
