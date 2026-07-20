"""Behavioral tests for schema-v3 RiskSample assembly and isolation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import math
import os
from pathlib import Path

import numpy as np
import pytest

from src.contracts import (
    SCHEMA_VERSION,
    TRAJECTORY_CHANNELS,
    BaseState,
    LocalTrajectory,
    OracleContext,
    OracleWorld,
    RiskSample,
    build_grid_spec,
)
import src.datasets.risk_dataset as risk_dataset_module
from src.datasets.risk_dataset import (
    RiskBuildInput,
    build_risk_sample,
    build_trajectory_channels,
    validate_risk_sample_for_publication,
)
from src.datasets.snippet_library import MotionSnippet
from src.datasets.shard_writer import load_risk_shard, write_risk_shard
from src.generation.dynamic_object_transplant import (
    TransplantedDynamicObject,
    footprint_from_spec,
    transplant_snippet,
)
from src.generation.event_sampler import (
    SOP05_GENERATOR_ALGORITHM_VERSION,
    GeneratedEvent,
)
from src.generation.event_target_motion_shard import (
    build_event_target_motion_world_metadata,
    compute_footprint_spec_digest,
    compute_motion_array_digest,
    create_event_target_motion_record,
)
import src.generation.paired_variants as paired_variants_module
from src.generation.paired_variants import (
    PAIRED_GENERATOR_ALGORITHM_VERSION,
    PAIRED_GROUP_CONTRACT_VERSION,
    PairedEventGroup,
    PairedVariant,
    PairedVariantConfig,
    assemble_paired_event_group,
    compute_pair_group_id,
    normalize_paired_variant_config,
)
from src.generation.structural_blindspot import StructuralBlindSpot
from src.geometry import (
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint,
    trajectory_signed_clearances,
)
from src.utils.seeding import stable_digest


TARGET_ID = "generated::human::hidden-target"
CONTEXT_ID = "recording::visible-context"


def _config() -> dict[str, object]:
    return {
        "schema_version": "3.0.0",
        "bev": {
            "range_m": 9.0,
            "resolution_m": 1.0,
            "size": 9,
            "history_steps": 8,
            "history_dt_s": 0.2,
            "future_steps": 15,
            "future_dt_s": 0.2,
        },
        "robot": {
            "model": "differential_drive",
            "length_m": 0.70,
            "width_m": 0.55,
            "inflation_m": 0.15,
            "max_linear_speed_mps": 0.9,
            "max_angular_speed_radps": 0.8,
        },
        "age_map": {
            "a_max_s": 5.0,
            "never_seen_value": 1.0,
            "visible_value": 0.0,
        },
    }


def _risk_config() -> dict[str, float]:
    return {
        "sigma_distance_m": 0.5,
        "sigma_time_s": 2.0,
        "near_miss_distance_m": 0.35,
    }


def _circle_spec(radius_m: float = 0.2) -> dict[str, object]:
    return {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": radius_m},
    }


def _constant_motion(x: float, y: float = 0.0) -> np.ndarray:
    poses = np.empty((15, 3), dtype=np.float32)
    poses[:] = np.asarray([x, y, 0.0], dtype=np.float32)
    return poses


def _base_state(static: np.ndarray) -> BaseState:
    context_history = np.empty((8, 3), dtype=np.float32)
    context_history[:] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    return BaseState(
        state_id="base-risk-dataset-toy",
        split="train",
        recording_id="recording-risk-dataset-toy",
        dynamic_object_ids=(CONTEXT_ID,),
        timestamp=1.4,
        robot_history=np.zeros((8, 3), dtype=np.float32),
        robot_state=np.asarray([0.4, 0.0], dtype=np.float32),
        visible_dynamic_object_history={CONTEXT_ID: context_history},
        visible_dynamic_object_specs={CONTEXT_ID: _circle_spec()},
        static_map_local=static.copy(),
        metadata={
            "coordinate_frame": "robot_current_local",
            "session_id": "base-session-risk-dataset-toy",
        },
    )


def _trajectory() -> LocalTrajectory:
    grid = build_grid_spec(_config())
    shape = (grid.height, grid.width)
    return LocalTrajectory(
        trajectory_id="trajectory-risk-dataset-toy",
        poses=np.zeros((grid.future_steps, 3), dtype=np.float32),
        controls=np.zeros((grid.future_steps, 2), dtype=np.float32),
        swept_mask=np.full(shape, 1.0, dtype=np.float32),
        tta_map=np.full(shape, 2.0, dtype=np.float32),
        braking_map=np.full(shape, 3.0, dtype=np.float32),
        centerline_map=np.full(shape, 4.0, dtype=np.float32),
        task_cost=0.0,
        metadata={
            "pose_time_layout_version": "future_endpoints_dt_to_horizon_v1",
            "v": 0.0,
            "omega": 0.0,
        },
    )


def _sensor() -> StructuralBlindSpot:
    return StructuralBlindSpot(
        forward_fov_deg=360.0,
        range_m=9.0,
        blind_sectors=({"center_deg": 90.0, "width_deg": 90.0},),
    )


def _source(
    *,
    target_future: np.ndarray | None = None,
    event_type: str = "collision",
) -> RiskBuildInput:
    config = _config()
    grid = build_grid_spec(config)
    static = np.zeros((grid.height, grid.width), dtype=np.float32)
    base = _base_state(static)
    context_history = base.visible_dynamic_object_history[CONTEXT_ID].copy()
    scene_history = {CONTEXT_ID: context_history}
    scene_specs = {CONTEXT_ID: _circle_spec()}
    trajectories = {CONTEXT_ID: _constant_motion(0.0)}
    specs = {CONTEXT_ID: _circle_spec()}
    hidden_ids: tuple[str, ...] = ()
    if target_future is not None:
        target_history = np.empty((grid.history_steps, 3), dtype=np.float32)
        target_history[:] = np.asarray([0.0, 2.0, 0.0], dtype=np.float32)
        scene_history[TARGET_ID] = target_history
        scene_specs[TARGET_ID] = _circle_spec()
        trajectories[TARGET_ID] = target_future
        specs[TARGET_ID] = _circle_spec()
        hidden_ids = (TARGET_ID,)
    world = OracleWorld(
        world_id=f"label-source-{event_type}",
        base_state_id=base.state_id,
        static_occupancy=static.copy(),
        dynamic_object_trajectories=trajectories,
        dynamic_object_specs=specs,
        occluders=(),
        blind_spot_config={"kind": "structural"},
        random_seed=23,
        metadata={"schema_version": SCHEMA_VERSION},
    )
    return RiskBuildInput(
        sample_id=f"risk-sample-{event_type}",
        pair_group_id="pair-risk-dataset-toy",
        event_type=event_type,
        base_state=base,
        trajectory=_trajectory(),
        oracle_world=world,
        observed_static_occupancy=static,
        scene_dynamic_history=scene_history,
        scene_dynamic_specs=scene_specs,
        hidden_object_ids=hidden_ids,
        sensor_config=_sensor(),
        provenance={
            "source_recording_id": base.recording_id,
            "session_id": "session-risk-toy",
            "dynamic_object_snippet_id": "snippet-risk-toy",
            "seed_namespace": "split/train/risk-dataset",
        },
    )


def _build(source: RiskBuildInput) -> RiskSample:
    return build_risk_sample(
        source,
        base_config=_config(),
        risk_config=_risk_config(),
    )


def _paired_config() -> PairedVariantConfig:
    return normalize_paired_variant_config(
        {
            "schema_version": SCHEMA_VERSION,
            "paired_generator_algorithm_version": (
                PAIRED_GENERATOR_ALGORITHM_VERSION
            ),
            "group_contract_version": PAIRED_GROUP_CONTRACT_VERSION,
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
            "lateral_offset_step_m": 0.05,
            "lateral_offset_max_m": 2.0,
            "mother_required_variants": ["collision"],
            "training_minimum_contrast_count": 0,
            "audit_requires_all_variants": True,
        }
    )


def _formal_sop06_group_inputs(
    *,
    include_empty: bool = False,
    include_context: bool = False,
    extra_variant_kind: str | None = None,
    paired_seed: int = 23,
    target_future_y_m: float = 0.0,
    base_recording_id: str = "base-recording-risk-adapter",
    base_session_id: str = "base-session-risk-adapter",
    source_recording_id: str = "source-recording-risk-adapter",
    source_session_id: str = "source-session-risk-adapter",
    sop05_transplant_seed: int = 17,
    dataset_seed: int = 41,
) -> dict[str, object]:
    """Build one formally stamped partial SOP06 group for atomic adapter tests."""

    config = _config()
    grid = build_grid_spec(config)
    trajectory = _trajectory()
    if extra_variant_kind == "temporal_safe":
        moving_poses = np.column_stack(
            (
                np.arange(2, 17, dtype=np.float32) * np.float32(0.25),
                np.zeros(grid.future_steps, dtype=np.float32),
                np.zeros(grid.future_steps, dtype=np.float32),
            )
        ).astype(np.float32)
        moving_controls = np.tile(
            np.asarray([1.25, 0.0], dtype=np.float32),
            (grid.future_steps, 1),
        )
        trajectory = replace(
            trajectory,
            poses=moving_poses,
            controls=moving_controls,
            metadata={**trajectory.metadata, "v": 1.25, "omega": 0.0},
        )
    target_spec = _circle_spec(radius_m=0.2)
    target_history = np.tile(
        np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
        (grid.history_steps, 1),
    )
    if extra_variant_kind == "temporal_safe":
        mother_future = trajectory.poses.copy()
    else:
        mother_future = np.column_stack(
            (
                np.arange(14, -1, -1, dtype=np.float32) * np.float32(0.125),
                np.full(
                    grid.future_steps,
                    np.float32(target_future_y_m),
                    dtype=np.float32,
                ),
                np.zeros(grid.future_steps, dtype=np.float32),
            )
        ).astype(np.float32)
    source_snippet_id = "train-human-snippet-risk-adapter"
    source_object_id = f"{source_recording_id}::source-risk-adapter"
    record = create_event_target_motion_record(
        generated_event_id="event-risk-adapter",
        world_id="world-risk-adapter-mother",
        base_state_id="base-risk-adapter",
        trajectory_id="trajectory-risk-dataset-toy",
        target_dynamic_object_id=TARGET_ID,
        source_snippet_id=source_snippet_id,
        source_object_id=source_object_id,
        object_type="human",
        footprint_spec=target_spec,
        footprint_spec_digest=compute_footprint_spec_digest(target_spec),
        target_type_policy_digest="a" * 32,
        history_poses=target_history,
        current_pose=target_history[-1].copy(),
        future_poses=mother_future,
    )
    target_provenance = {
        "snippet_id": record.source_snippet_id,
        "source_recording_id": source_recording_id,
        "source_session_id": source_session_id,
        "source_object_id": record.source_object_id,
        "source_current_index": 7,
        "candidate_current_xy": [2.0, 0.0],
        "rotation_rad": 0.0,
        "conflict_point": (
            [float(value) for value in trajectory.poses[7, :2]]
            if extra_variant_kind == "temporal_safe"
            else [0.0, 0.0]
        ),
        "crossing_direction": [1.0, 0.0],
        "desired_crossing_direction": [1.0, 0.0],
        "time_scale": 1.0,
        "target_type_policy_digest": record.target_type_policy_digest,
        "seed": sop05_transplant_seed,
    }
    mother_target = TransplantedDynamicObject(
        target_dynamic_object_id=record.target_dynamic_object_id,
        source_object_id=record.source_object_id,
        snippet_id=record.source_snippet_id,
        object_type=record.object_type,
        footprint_spec=dict(record.footprint_spec),
        footprint_spec_digest=record.footprint_spec_digest,
        history_poses=record.history_poses.copy(),
        current_pose=record.current_pose.copy(),
        future_poses=record.future_poses.copy(),
        provenance=dict(target_provenance),
    )
    mother_poses = np.vstack(
        (mother_target.history_poses, mother_target.future_poses)
    )
    snippet_positions = (
        mother_poses[:, :2] - mother_target.current_pose[:2]
    ).astype(np.float32)
    snippet_velocities = np.gradient(snippet_positions, 0.2, axis=0).astype(
        np.float32
    )
    source_snippet = MotionSnippet(
        snippet_id=record.source_snippet_id,
        split="train",
        source_recording_id=source_recording_id,
        source_session_id=source_session_id,
        source_object_id=record.source_object_id,
        object_type=record.object_type,
        footprint=dict(target_spec["footprint"]),
        start_timestamp=0.0,
        positions=snippet_positions,
        velocities=snippet_velocities,
        headings=mother_poses[:, 2].copy(),
        duration_s=4.4,
        mean_speed_mps=0.0,
        max_acceleration_mps2=0.0,
        mean_abs_curvature_per_m=0.0,
        provenance={},
    )
    context_history = np.tile(
        np.asarray([0.0, 3.0, 0.0], dtype=np.float32),
        (grid.history_steps, 1),
    )
    context_future = np.tile(
        np.asarray([0.0, 3.0, 0.0], dtype=np.float32),
        (grid.future_steps, 1),
    )
    context_spec = _circle_spec(radius_m=0.25)
    background_trajectories = (
        {CONTEXT_ID: context_future.copy()} if include_context else {}
    )
    background_specs = (
        {CONTEXT_ID: dict(context_spec)} if include_context else {}
    )
    occluder_pose = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    world_static = rasterize_footprint(
        RectangleFootprint(0.8, 0.8), occluder_pose, grid
    ).astype(np.float32)
    occluders = (
        {
            "occluder_id": "occluder::risk-adapter",
            "type": "pillar",
            "pose": [float(value) for value in occluder_pose],
            "length_m": 0.8,
            "width_m": 0.8,
            "placement_strategy": SOP05_GENERATOR_ALGORITHM_VERSION,
        },
    )
    blind_spot_config = {
        "kind": "environment",
        "occluder_ids": [occluders[0]["occluder_id"]],
        "blind_region_digest": "blind-region-risk-adapter",
    }
    target_history_visibility = np.zeros(grid.history_steps, dtype=bool)
    visibility_sequence = np.asarray([False] * 12 + [True] * 4, dtype=bool)
    mother_world = OracleWorld(
        world_id=record.world_id,
        base_state_id=record.base_state_id,
        static_occupancy=world_static.copy(),
        dynamic_object_trajectories={
            **background_trajectories,
            TARGET_ID: record.future_poses.copy(),
        },
        dynamic_object_specs={
            **background_specs,
            TARGET_ID: dict(record.footprint_spec),
        },
        occluders=occluders,
        blind_spot_config=blind_spot_config,
        random_seed=sop05_transplant_seed,
        metadata={
            **build_event_target_motion_world_metadata(record),
            "schema_version": SCHEMA_VERSION,
            "event_kind": "environment",
            "generator_algorithm_version": SOP05_GENERATOR_ALGORITHM_VERSION,
            "conflict_index": 7,
            "conflict_time_s": 1.6,
            "target_provenance": dict(target_provenance),
        },
    )
    mother_event = GeneratedEvent(
        generated_event_id=record.generated_event_id,
        event_kind="environment",
        world=mother_world,
        target=mother_target,
        target_motion_record=record,
        visibility_sequence=visibility_sequence.copy(),
        target_visibility_history=target_history_visibility.copy(),
        conflict_time_s=1.6,
        conflict_index=7,
    )
    base_state = BaseState(
        state_id=record.base_state_id,
        split="train",
        recording_id=base_recording_id,
        dynamic_object_ids=((CONTEXT_ID,) if include_context else ()),
        timestamp=4.0,
        robot_history=np.zeros((grid.history_steps, 3), dtype=np.float32),
        robot_state=np.asarray([0.4, 0.0], dtype=np.float32),
        visible_dynamic_object_history=(
            {CONTEXT_ID: context_history.copy()} if include_context else {}
        ),
        visible_dynamic_object_specs=(
            {CONTEXT_ID: dict(context_spec)} if include_context else {}
        ),
        static_map_local=np.zeros_like(world_static),
        metadata={
            "coordinate_frame": "robot_current_local",
            "session_id": base_session_id,
        },
    )
    oracle_context = OracleContext(
        base_state_id=record.base_state_id,
        dynamic_object_history=(
            {CONTEXT_ID: context_history.copy()} if include_context else {}
        ),
        dynamic_object_future=(
            {CONTEXT_ID: context_future.copy()} if include_context else {}
        ),
        dynamic_object_specs=(
            {CONTEXT_ID: dict(context_spec)} if include_context else {}
        ),
        metadata={"future_dt_s": 0.2, "source_recording_id": base_recording_id},
    )
    paired_config = _paired_config()
    paired_config_digest = paired_config.digest
    pair_group_id = compute_pair_group_id(
        generated_event_id=record.generated_event_id,
        base_state_id=record.base_state_id,
        trajectory_id=record.trajectory_id,
        occluders=mother_world.occluders,
        blind_spot_config=mother_world.blind_spot_config,
        source_snippet_id=record.source_snippet_id,
        target_dynamic_object_id=record.target_dynamic_object_id,
        paired_config_digest=paired_config_digest,
    )

    robot_footprint = inflate_footprint(
        RectangleFootprint(0.70, 0.55), 0.15
    )

    def spatial_target(
        kind: str,
    ) -> tuple[TransplantedDynamicObject, float, float, float]:
        pivot_radius, ray, parameters = (
            paired_variants_module._transform_parameter_grid(
                mother_event, paired_config
            )
        )
        for radial, signed_arc in parameters:
            candidate = paired_variants_module._transformed_target(
                mother_target,
                radial_shift_m=radial,
                signed_arc_offset_m=signed_arc,
                pivot_radius_m=pivot_radius,
                ray_direction=ray,
            )
            clearances = trajectory_signed_clearances(
                robot_footprint,
                trajectory.poses,
                footprint_from_spec(candidate.footprint_spec),
                candidate.future_poses,
            )
            minimum = float(np.min(clearances))
            if kind == "near_miss":
                lower, upper = paired_config.near_miss_clearance_range_m
                eligible = lower <= minimum <= upper
            elif kind == "spatial_safe":
                lower, upper = paired_config.spatial_safe_clearance_range_m
                eligible = lower <= minimum <= upper
            else:
                eligible = minimum >= paired_config.irrelevant_min_clearance_m
            if eligible:
                return candidate, radial, signed_arc, signed_arc / pivot_radius
        raise AssertionError(f"test fixture lacks {kind} spatial candidate")

    def make_variant(kind: str) -> PairedVariant:
        if kind == "empty_blind_spot":
            target = None
            trajectories = {
                object_id: value.copy()
                for object_id, value in background_trajectories.items()
            }
            specs = {
                object_id: dict(value) for object_id, value in background_specs.items()
            }
            history_visibility = None
            future_visibility = None
            history_digest = None
            future_digest = None
            motion_digest = "target-empty"
            current_pose = None
            provenance = None
            temporal_offset_s = None
            lateral_offset_m = None
            radial_shift_m = None
            rotation_rad = None
            transform_metadata = {
                "kind": "target_removal",
                "removed_target_dynamic_object_id": TARGET_ID,
            }
        else:
            if kind == "collision":
                target = mother_target
                temporal_offset_s = None
                lateral_offset_m = None
                radial_shift_m = None
                rotation_rad = None
                transform_metadata = {"kind": "collision_mother"}
            elif kind == "temporal_safe":
                temporal_offset_s = -0.8
                target = transplant_snippet(
                    source_snippet,
                    conflict_point=target_provenance["conflict_point"],
                    conflict_time_s=mother_event.conflict_time_s + temporal_offset_s,
                    crossing_direction=target_provenance[
                        "desired_crossing_direction"
                    ],
                    time_scale=target_provenance["time_scale"],
                    future_dt_s=0.2,
                    future_steps=grid.future_steps,
                    base_state_id=record.base_state_id,
                    trajectory_id=trajectory.trajectory_id,
                    target_type_policy_digest=record.target_type_policy_digest,
                    seed=sop05_transplant_seed,
                    context_object_ids=tuple(oracle_context.dynamic_object_future),
                )
                target = replace(
                    target,
                    target_dynamic_object_id=TARGET_ID,
                    provenance={
                        **target.provenance,
                        "paired_transform": {
                            "kind": "temporal_offset_v1",
                            "temporal_offset_s": temporal_offset_s,
                            "mother_conflict_time_s": mother_event.conflict_time_s,
                        },
                    },
                )
                lateral_offset_m = None
                radial_shift_m = None
                rotation_rad = None
                transform_metadata = {
                    "kind": "temporal_offset_v1",
                    "temporal_offset_s": temporal_offset_s,
                    "mother_conflict_time_s": mother_event.conflict_time_s,
                    "variant_conflict_time_s": (
                        mother_event.conflict_time_s + temporal_offset_s
                    ),
                }
            elif kind in {"near_miss", "spatial_safe", "irrelevant_hidden"}:
                target, radial, signed_arc, angle = spatial_target(kind)
                temporal_offset_s = None
                lateral_offset_m = abs(signed_arc)
                radial_shift_m = radial
                rotation_rad = angle
                transform_metadata = {
                    "kind": "hidden_pose_pivot_v1",
                    "signed_arc_offset_m": signed_arc,
                    "radial_shift_m": radial,
                    "rotation_rad": angle,
                }
            else:  # pragma: no cover - test helper is frozen to formal kinds
                raise AssertionError(kind)
            trajectories = {
                **{
                    object_id: value.copy()
                    for object_id, value in background_trajectories.items()
                },
                TARGET_ID: target.future_poses.copy(),
            }
            specs = {
                **{
                    object_id: dict(value)
                    for object_id, value in background_specs.items()
                },
                TARGET_ID: dict(target.footprint_spec),
            }
            history_visibility = target_history_visibility.copy()
            future_visibility = visibility_sequence.copy()
            history_digest = compute_motion_array_digest(
                target.history_poses, field_name="target_history_poses"
            )
            future_digest = compute_motion_array_digest(
                target.future_poses, field_name="target_future_poses"
            )
            motion_digest = paired_variants_module._paired_target_motion_digest(target)
            current_pose = [float(value) for value in target.current_pose]
            provenance = dict(target.provenance)
            target_clearances = trajectory_signed_clearances(
                robot_footprint,
                trajectory.poses,
                footprint_from_spec(target.footprint_spec),
                target.future_poses,
            )
            minimum_index = int(np.argmin(target_clearances))
            minimum_clearance = float(target_clearances[minimum_index])
            minimum_time = float((minimum_index + 1) * 0.2)
        world_id = "world-" + stable_digest(
            pair_group_id,
            kind,
            motion_digest,
            paired_seed,
            paired_config_digest,
            size=12,
        )
        paired_world = OracleWorld(
            world_id=world_id,
            base_state_id=record.base_state_id,
            static_occupancy=world_static.copy(),
            dynamic_object_trajectories=trajectories,
            dynamic_object_specs=specs,
            occluders=tuple(dict(item) for item in occluders),
            blind_spot_config=dict(blind_spot_config),
            random_seed=paired_seed,
            metadata={
                "schema_version": SCHEMA_VERSION,
                "event_kind": "environment",
                "generator_algorithm_version": SOP05_GENERATOR_ALGORITHM_VERSION,
                "paired_generator_algorithm_version": PAIRED_GENERATOR_ALGORITHM_VERSION,
                "pair_group_contract_version": PAIRED_GROUP_CONTRACT_VERSION,
                "world_id": world_id,
                "mother_generated_event_id": record.generated_event_id,
                "mother_world_id": mother_world.world_id,
                "mother_target_motion_record_digest": record.record_digest,
                "mother_source_snippet_id": record.source_snippet_id,
                "mother_source_object_id": record.source_object_id,
                "mother_target_type_policy_digest": record.target_type_policy_digest,
                "mother_target_footprint_spec_digest": record.footprint_spec_digest,
                "pair_group_id": pair_group_id,
                "paired_variant_kind": kind,
                "paired_config_digest": paired_config_digest,
                "paired_seed": paired_seed,
                "target_dynamic_object_id": record.target_dynamic_object_id,
                "target_present": target is not None,
                "paired_target_history_array_digest": history_digest,
                "paired_target_future_array_digest": future_digest,
                "paired_target_motion_digest": motion_digest,
                "paired_target_current_pose": current_pose,
                "target_provenance": provenance,
                "visibility_sequence": (
                    None
                    if future_visibility is None
                    else [bool(value) for value in future_visibility]
                ),
                "target_visibility_history": (
                    None
                    if history_visibility is None
                    else [bool(value) for value in history_visibility]
                ),
                "min_clearance_m": (
                    None if target is None else minimum_clearance
                ),
                "time_to_min_clearance_s": (
                    None if target is None else minimum_time
                ),
                "paired_transform": transform_metadata,
            },
        )
        return PairedVariant(
            variant_kind=kind,
            world=paired_world,
            target=target,
            target_visibility_history=history_visibility,
            visibility_sequence=future_visibility,
            clearance_sequence_m=(
                None
                if target is None
                else target_clearances.copy()
            ),
            min_clearance_m=None if target is None else minimum_clearance,
            time_to_min_clearance_s=None if target is None else minimum_time,
            temporal_offset_s=temporal_offset_s,
            lateral_offset_m=lateral_offset_m,
            radial_shift_m=radial_shift_m,
            rotation_rad=rotation_rad,
        )

    variants = {"collision": make_variant("collision")}
    if extra_variant_kind is not None:
        variants[extra_variant_kind] = make_variant(extra_variant_kind)
    if include_empty:
        variants["empty_blind_spot"] = make_variant("empty_blind_spot")
    missing = {
        kind: f"{kind}_unavailable"
        for kind in paired_variants_module.VARIANT_ORDER
        if kind not in variants
    }
    group = assemble_paired_event_group(
        pair_group_id=pair_group_id,
        variants=variants,
        missing_variant_reasons=missing,
        paired_config=paired_config,
    )
    return {
        "group": group,
        "mother_event": mother_event,
        "source_snippet": source_snippet,
        "base_state": base_state,
        "trajectory": trajectory,
        "oracle_context": oracle_context,
        "base_config": config,
        "paired_config": paired_config,
        "risk_config": _risk_config(),
        "dataset_seed": dataset_seed,
    }


def _build_formal_group_samples(inputs: dict[str, object]) -> tuple[RiskSample, ...]:
    return risk_dataset_module.build_risk_samples_from_sop06_group(**inputs)


def test_formal_group_adapter_is_atomic_and_keeps_valid_partial_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _formal_sop06_group_inputs(include_empty=True, include_context=True)
    real_render = risk_dataset_module.render_sop06_partial_pair_group
    render_calls = []

    def render_once(**kwargs):
        render_calls.append(kwargs["group"].pair_group_id)
        return real_render(**kwargs)

    monkeypatch.setattr(
        risk_dataset_module, "render_sop06_partial_pair_group", render_once
    )
    monkeypatch.setattr(
        risk_dataset_module,
        "render_observation",
        lambda *args, **kwargs: pytest.fail("formal group must reuse SOP06 render"),
    )

    samples = _build_formal_group_samples(inputs)

    assert tuple(sample.event_type for sample in samples) == (
        "collision",
        "empty_blind_spot",
    )
    assert samples[0].collision_label == 1
    assert samples[1].collision_label == 0
    assert samples[1].risk_severity == 0.0
    assert samples[0].pair_group_id == samples[1].pair_group_id
    assert render_calls == [samples[0].pair_group_id]


def test_formal_group_sidecar_api_preserves_risk_samples_and_keeps_labels_separate() -> None:
    combined_inputs = _formal_sop06_group_inputs(
        include_empty=True, include_context=True
    )
    legacy_inputs = _formal_sop06_group_inputs(
        include_empty=True, include_context=True
    )

    samples, sidecars = (
        risk_dataset_module.build_risk_samples_and_sidecars_from_sop06_group(
            **combined_inputs
        )
    )
    legacy_samples = _build_formal_group_samples(legacy_inputs)

    assert tuple(sidecar.sample_id for sidecar in sidecars) == tuple(
        sample.sample_id for sample in samples
    )
    assert len(samples) == len(legacy_samples) == len(sidecars)
    for sample, legacy in zip(samples, legacy_samples, strict=True):
        for field in fields(RiskSample):
            actual = getattr(sample, field.name)
            expected = getattr(legacy, field.name)
            if isinstance(actual, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            else:
                assert actual == expected
        assert "hidden_risk_occupancy" not in sample.metadata
        assert "robot_future_footprints" not in sample.metadata
    by_kind = {
        sample.event_type: sidecar
        for sample, sidecar in zip(samples, sidecars, strict=True)
    }
    assert np.any(by_kind["collision"].hidden_risk_occupancy)
    assert not np.any(by_kind["empty_blind_spot"].hidden_risk_occupancy)
    assert all(not sidecar.hidden_risk_occupancy.flags.writeable for sidecar in sidecars)


def test_risk_only_group_adapter_never_builds_oracle_sidecars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _formal_sop06_group_inputs(
        include_empty=True, include_context=True
    )

    def forbidden_sidecar_builder(**kwargs):
        raise AssertionError("risk-only assembly must not rasterize sidecars")

    monkeypatch.setattr(
        risk_dataset_module,
        "build_risk_label_sidecar",
        forbidden_sidecar_builder,
    )

    samples = risk_dataset_module.build_risk_samples_from_sop06_group(**inputs)

    assert tuple(sample.event_type for sample in samples) == (
        "collision",
        "empty_blind_spot",
    )


def test_combined_group_adapter_builds_one_sidecar_per_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _formal_sop06_group_inputs(
        include_empty=True, include_context=True
    )
    real_builder = risk_dataset_module.build_risk_label_sidecar
    built_ids: list[str] = []

    def counted_sidecar_builder(**kwargs):
        built_ids.append(kwargs["sample_id"])
        return real_builder(**kwargs)

    monkeypatch.setattr(
        risk_dataset_module,
        "build_risk_label_sidecar",
        counted_sidecar_builder,
    )

    samples, sidecars = (
        risk_dataset_module.build_risk_samples_and_sidecars_from_sop06_group(
            **inputs
        )
    )

    assert built_ids == [sample.sample_id for sample in samples]
    assert tuple(built_ids) == tuple(sidecar.sample_id for sidecar in sidecars)


def test_formal_group_risk_shard_matches_pre_task4_semantic_golden(
    tmp_path,
) -> None:
    inputs = _formal_sop06_group_inputs(
        include_empty=True, include_context=True
    )
    samples = risk_dataset_module.build_risk_samples_from_sop06_group(**inputs)
    grid = build_grid_spec(inputs["base_config"])
    root = tmp_path / "risk-shard-golden"

    write_risk_shard(
        samples,
        root,
        grid=grid,
        shard_index=7,
        expected_sample_count=2,
    )
    loaded = load_risk_shard(root, grid=grid)

    assert tuple(sample.sample_id for sample in loaded.samples) == (
        "train-269373dcdaab245d19563b96",
        "train-ede80befc273c7aa98c3b3a8",
    )
    assert loaded.manifest_digest == (
        "7e00c65a7e1e3ecbfd043470879dbdbfcb057f95b66bf8a07f798b2280637ae7"
    )
    assert loaded.semantic_digest == (
        "bc4f2ed6a1029a29635f5c46edd4fbc0079a8816a4ce4539432c44c8380f50e4"
    )


def _write_formal_risk_shard_for_hardened_load(
    tmp_path: Path,
) -> tuple[Path, object]:
    inputs = _formal_sop06_group_inputs(
        include_empty=True, include_context=True
    )
    samples = risk_dataset_module.build_risk_samples_from_sop06_group(**inputs)
    grid = build_grid_spec(inputs["base_config"])
    root = tmp_path / "risk-shard-hardened"
    write_risk_shard(
        samples,
        root,
        grid=grid,
        shard_index=7,
        expected_sample_count=2,
    )
    return root, grid


@pytest.mark.parametrize(
    "member_name", ("summary.json", "metadata.jsonl", "samples.npz")
)
def test_hardened_risk_loader_rejects_member_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_name: str,
) -> None:
    root, grid = _write_formal_risk_shard_for_hardened_load(tmp_path)
    root_inode = os.lstat(root).st_ino
    member = root / member_name
    displaced = root / f"{member_name}.displaced"
    real_open = risk_dataset_module._open_risk_snapshot_member_nofollow
    swapped = False

    def swap_member_then_open(root_fd: int, name: str):
        nonlocal swapped
        if name == member_name and not swapped:
            member.rename(displaced)
            member.symlink_to(displaced.name)
            swapped = True
        return real_open(root_fd, name)

    monkeypatch.setattr(
        risk_dataset_module,
        "_open_risk_snapshot_member_nofollow",
        swap_member_then_open,
    )

    with pytest.raises(ValueError, match="symlink"):
        risk_dataset_module.load_hardened_risk_shard_snapshot(
            root, grid=grid
        )

    assert swapped
    assert os.lstat(root).st_ino == root_inode
    assert member.is_symlink()
    assert displaced.is_file()


@pytest.mark.parametrize(
    "member_name", ("summary.json", "metadata.jsonl", "samples.npz")
)
def test_hardened_risk_loader_detects_same_inode_member_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_name: str,
) -> None:
    root, grid = _write_formal_risk_shard_for_hardened_load(tmp_path)
    root_inode = os.lstat(root).st_ino
    member = root / member_name
    member_inode = os.lstat(member).st_ino
    real_load = risk_dataset_module._load_risk_shard_from_snapshot_directory
    mutated = False

    def mutate_original_then_load(snapshot_root: Path, *, grid):
        nonlocal mutated
        with member.open("r+b") as mutable:
            mutable.seek(0)
            mutable.write(b"BORK")
            mutable.flush()
            os.fsync(mutable.fileno())
        assert os.lstat(member).st_ino == member_inode
        mutated = True
        return real_load(snapshot_root, grid=grid)

    monkeypatch.setattr(
        risk_dataset_module,
        "_load_risk_shard_from_snapshot_directory",
        mutate_original_then_load,
    )

    with pytest.raises(ValueError, match="content changed"):
        risk_dataset_module.load_hardened_risk_shard_snapshot(
            root, grid=grid
        )

    assert mutated
    assert os.lstat(root).st_ino == root_inode
    assert os.lstat(member).st_ino == member_inode


def test_formal_group_adapter_replaces_the_single_variant_bypass() -> None:
    assert not hasattr(risk_dataset_module, "build_risk_input_from_sop06_variant")


@pytest.mark.parametrize(
    ("variant_kind", "expected_near_miss"),
    [
        ("near_miss", 1),
        ("temporal_safe", 1),
        ("spatial_safe", 0),
        ("irrelevant_hidden", 0),
    ],
)
def test_formal_group_adapter_validates_actual_counterfactual_semantics(
    variant_kind: str,
    expected_near_miss: int,
) -> None:
    inputs = _formal_sop06_group_inputs(extra_variant_kind=variant_kind)

    samples = _build_formal_group_samples(inputs)

    by_kind = {sample.event_type: sample for sample in samples}
    assert by_kind[variant_kind].collision_label == 0
    assert by_kind[variant_kind].near_miss == expected_near_miss


@pytest.mark.parametrize(
    "variant_kind",
    [
        "collision",
        "near_miss",
        "temporal_safe",
        "spatial_safe",
        "irrelevant_hidden",
    ],
)
def test_formal_group_adapter_rejects_counterfeit_transform_geometry(
    variant_kind: str,
) -> None:
    inputs = _formal_sop06_group_inputs(extra_variant_kind=variant_kind)
    group = inputs["group"]
    variant = group.by_kind[variant_kind]
    future = variant.target.future_poses.copy()
    future[0, 2] += np.float32(0.2)
    target = replace(variant.target, future_poses=future)
    motion_digest = paired_variants_module._paired_target_motion_digest(target)
    world_id = "world-" + stable_digest(
        group.pair_group_id,
        variant_kind,
        motion_digest,
        variant.world.random_seed,
        group.paired_config_digest,
        size=12,
    )
    trajectories = dict(variant.world.dynamic_object_trajectories)
    trajectories[TARGET_ID] = future.copy()
    world = replace(
        variant.world,
        world_id=world_id,
        dynamic_object_trajectories=trajectories,
        metadata={
            **variant.world.metadata,
            "world_id": world_id,
            "paired_target_future_array_digest": compute_motion_array_digest(
                future, field_name="target_future_poses"
            ),
            "paired_target_motion_digest": motion_digest,
            "target_provenance": dict(target.provenance),
        },
    )
    tampered = replace(variant, target=target, world=world)
    inputs["group"] = replace(
        group,
        variants=tuple(
            tampered if item.variant_kind == variant_kind else item
            for item in group.variants
        ),
    )

    with pytest.raises(ValueError, match="does not reconstruct"):
        _build_formal_group_samples(inputs)


def test_formal_group_adapter_rejects_forged_paired_config_digest() -> None:
    inputs = _formal_sop06_group_inputs()
    paired_config = inputs["paired_config"]
    inputs["paired_config"] = replace(
        paired_config,
        irrelevant_min_clearance_m=paired_config.irrelevant_min_clearance_m + 0.1,
    )

    with pytest.raises(ValueError, match="canonical paired config"):
        _build_formal_group_samples(inputs)


def test_formal_group_adapter_snapshots_inputs_before_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _formal_sop06_group_inputs()
    original_group = inputs["group"]
    real_render = risk_dataset_module.render_sop06_partial_pair_group

    def render_then_mutate_external_input(**kwargs):
        rendered = real_render(**kwargs)
        collision = original_group.by_kind["collision"]
        collision.target.future_poses[:, 1] += np.float32(3.0)
        collision.world.dynamic_object_trajectories[TARGET_ID][:, 1] += np.float32(
            3.0
        )
        return rendered

    monkeypatch.setattr(
        risk_dataset_module,
        "render_sop06_partial_pair_group",
        render_then_mutate_external_input,
    )

    sample = _build_formal_group_samples(inputs)[0]

    assert sample.collision_label == 1


def test_formal_group_adapter_rejects_unconfigured_temporal_offset() -> None:
    inputs = _formal_sop06_group_inputs(extra_variant_kind="temporal_safe")
    group = inputs["group"]
    inputs["group"] = replace(
        group,
        variants=tuple(
            replace(variant, temporal_offset_s=0.7)
            if variant.variant_kind == "temporal_safe"
            else variant
            for variant in group.variants
        ),
    )

    with pytest.raises(ValueError, match="absent from paired_config"):
        _build_formal_group_samples(inputs)


def test_formal_group_adapter_rejects_missing_spatial_transform() -> None:
    inputs = _formal_sop06_group_inputs(extra_variant_kind="spatial_safe")
    group = inputs["group"]
    inputs["group"] = replace(
        group,
        variants=tuple(
            replace(variant, lateral_offset_m=None)
            if variant.variant_kind == "spatial_safe"
            else variant
            for variant in group.variants
        ),
    )

    with pytest.raises(TypeError, match="lateral_offset_m"):
        _build_formal_group_samples(inputs)


def test_formal_group_adapter_rejects_collision_name_with_safe_future() -> None:
    inputs = _formal_sop06_group_inputs(target_future_y_m=1.5)

    with pytest.raises(ValueError, match="collision variant must collide"):
        _build_formal_group_samples(inputs)


def test_formal_group_adapter_allows_visible_history_without_oracle_future() -> None:
    inputs = _formal_sop06_group_inputs()
    base = inputs["base_state"]
    history = np.tile(
        np.asarray([0.0, 3.0, 0.0], dtype=np.float32), (8, 1)
    )
    inputs["base_state"] = replace(
        base,
        dynamic_object_ids=(CONTEXT_ID,),
        visible_dynamic_object_history={CONTEXT_ID: history},
        visible_dynamic_object_specs={CONTEXT_ID: _circle_spec(radius_m=0.25)},
    )

    samples = _build_formal_group_samples(inputs)

    assert len(samples) == 1
    assert samples[0].collision_label == 1


def test_formal_group_adapter_allows_same_sample_base_source_identity() -> None:
    inputs = _formal_sop06_group_inputs(
        base_recording_id="shared-recording",
        source_recording_id="shared-recording",
        base_session_id="shared-session",
        source_session_id="shared-session",
    )

    sample = _build_formal_group_samples(inputs)[0]

    provenance = sample.metadata["provenance"]
    assert provenance["base_recording_id"] == "shared-recording"
    assert provenance["source_recording_id"] == "shared-recording"
    assert provenance["base_session_id"] == "shared-session"
    assert provenance["source_session_id"] == "shared-session"


def test_formal_group_adapter_binds_full_identity_and_safe_audit_provenance() -> None:
    baseline_inputs = _formal_sop06_group_inputs()
    baseline = _build_formal_group_samples(baseline_inputs)[0]
    repeat = _build_formal_group_samples(_formal_sop06_group_inputs())[0]
    primitive_inputs = _formal_sop06_group_inputs()
    primitive = primitive_inputs["trajectory"]
    primitive_inputs["trajectory"] = replace(
        primitive,
        controls=np.tile(
            np.asarray([0.1, 0.0], dtype=np.float32),
            (primitive.controls.shape[0], 1),
        ),
        metadata={**primitive.metadata, "v": 0.1},
    )
    changed = (
        _build_formal_group_samples(
            _formal_sop06_group_inputs(dataset_seed=43)
        )[0],
        _build_formal_group_samples(
            _formal_sop06_group_inputs(
                base_recording_id="base-recording-other",
                base_session_id="base-session-other",
            )
        )[0],
        _build_formal_group_samples(
            _formal_sop06_group_inputs(
                source_recording_id="source-recording-other",
                source_session_id="source-session-other",
            )
        )[0],
        _build_formal_group_samples(
            _formal_sop06_group_inputs(sop05_transplant_seed=19)
        )[0],
        _build_formal_group_samples(
            _formal_sop06_group_inputs(paired_seed=29)
        )[0],
        _build_formal_group_samples(primitive_inputs)[0],
    )

    assert baseline.sample_id == repeat.sample_id
    assert len({baseline.sample_id, *(sample.sample_id for sample in changed)}) == 7
    provenance = baseline.metadata["provenance"]
    assert provenance["base_recording_id"] == "base-recording-risk-adapter"
    assert provenance["base_session_id"] == "base-session-risk-adapter"
    assert provenance["source_recording_id"] == "source-recording-risk-adapter"
    assert provenance["source_session_id"] == "source-session-risk-adapter"
    assert provenance["source_snippet_id"] == "train-human-snippet-risk-adapter"
    assert provenance["sop05_transplant_seed"] == 17
    assert provenance["sop06_paired_seed"] == 23
    assert provenance["sop07_dataset_seed"] == 41
    assert provenance["trajectory_primitive"] == {
        "v_mps": 0.0,
        "omega_radps": 0.0,
    }
    assert provenance["occluders"] == [
        {
            "length_m": 0.8,
            "occluder_id": "occluder::risk-adapter",
            "placement_strategy": SOP05_GENERATOR_ALGORITHM_VERSION,
            "pose": [1.0, 0.0, 0.0],
            "type": "pillar",
            "width_m": 0.8,
        }
    ]
    assert not _metadata_has_forbidden_payload(provenance)


def test_formal_group_adapter_rejects_motion_snippet_identity_tampering() -> None:
    inputs = _formal_sop06_group_inputs()
    snippet = inputs["source_snippet"]
    inputs["source_snippet"] = replace(
        snippet, source_recording_id="source-recording-tampered"
    )

    with pytest.raises(ValueError, match="source recording"):
        _build_formal_group_samples(inputs)


def test_formal_group_adapter_rejects_source_session_lineage_tampering() -> None:
    inputs = _formal_sop06_group_inputs()
    snippet = inputs["source_snippet"]
    inputs["source_snippet"] = replace(
        snippet, source_session_id="source-session-tampered"
    )

    with pytest.raises(ValueError, match="source session"):
        _build_formal_group_samples(inputs)


def test_formal_group_adapter_rejects_blank_source_session_lineage() -> None:
    inputs = _formal_sop06_group_inputs(source_session_id="   ")

    with pytest.raises(ValueError, match="source session must be a non-empty string"):
        _build_formal_group_samples(inputs)


@pytest.mark.parametrize(
    "variant_kind",
    ["collision", "near_miss", "temporal_safe", "spatial_safe", "irrelevant_hidden"],
)
def test_formal_group_adapter_rejects_variant_source_session_drift(
    variant_kind: str,
) -> None:
    inputs = _formal_sop06_group_inputs(
        extra_variant_kind=None if variant_kind == "collision" else variant_kind
    )
    group = inputs["group"]
    variant = group.by_kind[variant_kind]
    target = replace(
        variant.target,
        provenance={
            **variant.target.provenance,
            "source_session_id": "variant-session-tampered",
        },
    )
    world = replace(
        variant.world,
        metadata={
            **variant.world.metadata,
            "target_provenance": dict(target.provenance),
        },
    )
    inputs["group"] = replace(
        group,
        variants=tuple(
            replace(item, target=target, world=world)
            if item.variant_kind == variant_kind
            else item
            for item in group.variants
        ),
    )

    with pytest.raises(ValueError, match="source session"):
        _build_formal_group_samples(inputs)


def test_formal_group_adapter_rejects_base_recording_identity_tampering() -> None:
    inputs = _formal_sop06_group_inputs()
    base_state = inputs["base_state"]
    inputs["base_state"] = replace(
        base_state, recording_id="base-recording-tampered"
    )

    with pytest.raises(ValueError, match="oracle_context/base recording"):
        _build_formal_group_samples(inputs)


def test_formal_group_adapter_rejects_same_id_with_tampered_snippet_motion() -> None:
    inputs = _formal_sop06_group_inputs()
    snippet = inputs["source_snippet"]
    positions = snippet.positions.copy()
    positions[3, 0] += np.float32(0.25)
    inputs["source_snippet"] = replace(snippet, positions=positions)

    with pytest.raises(ValueError, match="does not reconstruct mother target"):
        _build_formal_group_samples(inputs)


def test_formal_group_adapter_rejects_group_coverage_tampering() -> None:
    inputs = _formal_sop06_group_inputs()
    group = inputs["group"]
    inputs["group"] = replace(
        group,
        coverage_mask=(False,) + group.coverage_mask[1:],
    )

    with pytest.raises(ValueError, match="coverage mask"):
        _build_formal_group_samples(inputs)


def test_formal_group_adapter_rejects_retired_joint_identity() -> None:
    inputs = _formal_sop06_group_inputs()
    mother = inputs["mother_event"]
    inputs["mother_event"] = replace(
        mother,
        world=replace(
            mother.world,
            metadata={
                **mother.world.metadata,
                "joint_pair_generator_algorithm_version": "joint_environment_pair_v2",
            },
        ),
    )

    with pytest.raises(ValueError, match="retired joint_environment_pair_v2"):
        _build_formal_group_samples(inputs)


def _metadata_has_forbidden_payload(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(
                token in lowered
                for token in (
                    "future",
                    "oracle",
                    "clearance_sequence",
                    "dynamic_object_trajectories",
                    "hidden_object_ids",
                )
            ):
                return True
            if _metadata_has_forbidden_payload(child):
                return True
    if isinstance(value, (list, tuple)):
        return any(_metadata_has_forbidden_payload(child) for child in value)
    return False


def test_trajectory_channels_use_frozen_order_shape_dtype_and_owned_copy() -> None:
    trajectory = _trajectory()
    grid = build_grid_spec(_config())

    channels = build_trajectory_channels(trajectory, grid)

    assert TRAJECTORY_CHANNELS == (
        "swept_volume_mask",
        "time_to_arrival_map",
        "braking_margin_map",
        "centerline_map",
    )
    assert channels.shape == (4, grid.height, grid.width)
    assert channels.dtype == np.float32
    assert channels.flags.c_contiguous
    for index, expected in enumerate((1.0, 2.0, 3.0, 4.0)):
        np.testing.assert_array_equal(channels[index], expected)
    assert not np.shares_memory(channels, trajectory.swept_mask)


@pytest.mark.parametrize("defect", ["shape", "dtype", "nonfinite"])
def test_trajectory_channels_reject_invalid_query_maps(defect: str) -> None:
    trajectory = _trajectory()
    if defect == "shape":
        bad = np.zeros((8, 9), dtype=np.float32)
    elif defect == "dtype":
        bad = trajectory.tta_map.astype(np.float64)
    else:
        bad = trajectory.tta_map.copy()
        bad[0, 0] = np.nan
    trajectory = replace(trajectory, tta_map=bad)

    with pytest.raises((TypeError, ValueError), match="time_to_arrival_map"):
        build_trajectory_channels(trajectory, build_grid_spec(_config()))


def test_changing_only_hidden_future_changes_labels_not_observation_arrays() -> None:
    collision_source = _source(target_future=_constant_motion(0.0))
    safe_world = replace(
        collision_source.oracle_world,
        dynamic_object_trajectories={
            CONTEXT_ID: _constant_motion(0.0),
            TARGET_ID: _constant_motion(3.0),
        },
    )
    safe_source = replace(collision_source, oracle_world=safe_world)

    collision = _build(collision_source)
    safe = _build(safe_source)

    assert collision.collision_label == 1
    assert collision.risk_severity == 1.0
    assert collision.first_collision_time == pytest.approx(0.2)
    assert safe.collision_label == 0
    assert safe.near_miss == 0
    for field in (
        "bev_history",
        "state_channels",
        "trajectory_channels",
        "robot_state",
    ):
        np.testing.assert_array_equal(getattr(collision, field), getattr(safe, field))


def test_label_audit_is_the_only_location_for_critical_object_metadata() -> None:
    sample = _build(_source(target_future=_constant_motion(0.0)))

    assert sample.metadata["schema_version"] == "3.0.0"
    assert set(sample.metadata) == {
        "schema_version",
        "renderer",
        "trajectory_id",
        "provenance",
        "label_audit",
    }
    audit = sample.metadata["label_audit"]
    assert audit["critical_object_id"] == TARGET_ID
    assert audit["critical_object_type"] == "human"
    assert audit["time_to_min_clearance_s"] == pytest.approx(0.2)
    assert audit["has_hidden_target"] is True
    assert "critical_object_id" not in {
        key for key in sample.metadata if key != "label_audit"
    }
    assert not _metadata_has_forbidden_payload(sample.metadata)
    validate_risk_sample_for_publication(sample, build_grid_spec(_config()))


def test_empty_hidden_set_ignores_colliding_visible_context_and_uses_sentinel() -> None:
    sample = _build(_source(target_future=None, event_type="empty_blind_spot"))
    grid = build_grid_spec(_config())

    assert sample.event_type == "empty_blind_spot"
    assert sample.collision_label == 0
    assert sample.near_miss == 0
    assert sample.risk_severity == 0.0
    assert sample.min_clearance == pytest.approx(
        math.hypot(grid.width * grid.resolution_m, grid.height * grid.resolution_m)
    )
    assert sample.first_collision_time is None
    assert sample.metadata["label_audit"] == {
        "risk_gt_version": "hidden_risk_gt_schema3_v1",
        "pose_time_layout_version": "future_endpoints_dt_to_horizon_v1",
        "critical_object_id": None,
        "critical_object_type": None,
        "time_to_min_clearance_s": None,
        "has_hidden_target": False,
    }


def test_irrelevant_hidden_target_keeps_real_safe_label_and_excludes_context() -> None:
    sample = _build(
        _source(
            target_future=_constant_motion(3.0),
            event_type="irrelevant_hidden",
        )
    )

    assert sample.collision_label == 0
    assert sample.near_miss == 0
    assert sample.min_clearance > 0.35
    assert 0.0 < sample.risk_severity < 1.0
    assert sample.metadata["label_audit"]["critical_object_id"] == TARGET_ID
    assert sample.metadata["label_audit"]["critical_object_id"] != CONTEXT_ID


def test_visible_context_cannot_be_redeclared_as_hidden() -> None:
    source = _source(target_future=_constant_motion(3.0))
    source = replace(source, hidden_object_ids=(CONTEXT_ID,))

    with pytest.raises(ValueError, match="currently visible"):
        _build(source)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda metadata: metadata["provenance"].update(
                {"hidden_future_poses": [[0.0, 0.0, 0.0]]}
            ),
            "forbidden",
        ),
        (
            lambda metadata: metadata["provenance"].update(
                {"oracle_payload": {"x": 1.0}}
            ),
            "forbidden",
        ),
        (
            lambda metadata: metadata["provenance"].update(
                {"debug_array": np.zeros(1, dtype=np.float32)}
            ),
            "ndarray",
        ),
        (
            lambda metadata: metadata["provenance"].update(
                {"debug_score": float("nan")}
            ),
            "finite",
        ),
        (
            lambda metadata: metadata.update(
                {"critical_object_id": TARGET_ID}
            ),
            "metadata keys",
        ),
    ],
)
def test_publication_validator_rejects_metadata_leakage(
    mutation,
    message: str,
) -> None:
    sample = _build(_source(target_future=_constant_motion(0.0)))
    metadata = deepcopy(sample.metadata)
    mutation(metadata)

    with pytest.raises((TypeError, ValueError), match=message):
        validate_risk_sample_for_publication(
            replace(sample, metadata=metadata),
            build_grid_spec(_config()),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"min_clearance": float("inf")}, "min_clearance"),
        ({"risk_severity": float("nan")}, "risk_severity"),
        ({"first_collision_time": None}, "first_collision_time"),
    ],
)
def test_publication_validator_rejects_nonfinite_or_inconsistent_labels(
    changes: dict[str, object],
    message: str,
) -> None:
    sample = _build(_source(target_future=_constant_motion(0.0)))

    with pytest.raises(ValueError, match=message):
        validate_risk_sample_for_publication(
            replace(sample, **changes),
            build_grid_spec(_config()),
        )
