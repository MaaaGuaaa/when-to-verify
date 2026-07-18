"""Behavioral tests for schema-v3 RiskSample assembly and isolation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math

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
from src.datasets.risk_dataset import (
    RiskBuildInput,
    build_risk_input_from_sop06_variant,
    build_risk_sample,
    build_trajectory_channels,
    validate_risk_sample_for_publication,
)
from src.generation.dynamic_object_transplant import TransplantedDynamicObject
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
    PairedVariant,
    compute_pair_group_id,
)
from src.generation.structural_blindspot import StructuralBlindSpot
from src.geometry import RectangleFootprint, rasterize_footprint
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
        metadata={"coordinate_frame": "robot_current_local"},
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
        metadata={"pose_time_layout_version": "future_endpoints_dt_to_horizon_v1"},
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


def _formal_sop06_variant_inputs(
    *,
    empty: bool = False,
    include_context: bool = False,
    paired_config_digest: str = "b" * 32,
    paired_seed: int = 23,
    target_future_y_m: float = 0.0,
) -> dict[str, object]:
    """Build one fully joined formal-v5 environment variant for adapter tests."""

    config = _config()
    grid = build_grid_spec(config)
    target_spec = _circle_spec(radius_m=0.2)
    target_history = np.tile(
        np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
        (grid.history_steps, 1),
    )
    mother_future = np.column_stack(
        (
            np.linspace(2.1, 3.5, grid.future_steps, dtype=np.float32),
            np.zeros(grid.future_steps, dtype=np.float32),
            np.zeros(grid.future_steps, dtype=np.float32),
        )
    ).astype(np.float32)
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
    record = create_event_target_motion_record(
        generated_event_id="event-risk-adapter",
        world_id="world-risk-adapter-mother",
        base_state_id="base-risk-adapter",
        trajectory_id="trajectory-risk-dataset-toy",
        target_dynamic_object_id=TARGET_ID,
        source_snippet_id="snippet-risk-adapter",
        source_object_id="recording::source-risk-adapter",
        object_type="human",
        footprint_spec=target_spec,
        footprint_spec_digest=compute_footprint_spec_digest(target_spec),
        target_type_policy_digest="a" * 32,
        history_poses=target_history,
        current_pose=target_history[-1].copy(),
        future_poses=mother_future,
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
            "placement_strategy": "blind_reachability_first_v1",
        },
    )
    blind_spot_config = {
        "kind": "environment",
        "occluder_ids": [occluders[0]["occluder_id"]],
        "blind_region_digest": "blind-region-risk-adapter",
    }
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
        random_seed=17,
        metadata={
            **build_event_target_motion_world_metadata(record),
            "schema_version": SCHEMA_VERSION,
            "event_kind": "environment",
            "generator_algorithm_version": "blind_reachability_first_v1",
        },
    )
    base_state = BaseState(
        state_id=record.base_state_id,
        split="train",
        recording_id="recording-risk-adapter",
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
        metadata={"coordinate_frame": "robot_current_local"},
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
        metadata={"future_dt_s": 0.2},
    )
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
    if empty:
        variant_kind = "empty_blind_spot"
        target = None
        trajectories = {
            object_id: value.copy()
            for object_id, value in background_trajectories.items()
        }
        specs = {
            object_id: dict(value)
            for object_id, value in background_specs.items()
        }
        target_history_visibility = None
        visibility_sequence = None
        target_history_digest = None
        target_future_digest = None
        target_motion_digest = "target-empty"
        target_current_pose = None
        target_provenance = None
    else:
        variant_kind = "collision"
        target_future = np.column_stack(
            (
                np.linspace(1.8, 0.0, grid.future_steps, dtype=np.float32),
                np.full(
                    grid.future_steps,
                    np.float32(target_future_y_m),
                    dtype=np.float32,
                ),
                np.zeros(grid.future_steps, dtype=np.float32),
            )
        ).astype(np.float32)
        target = TransplantedDynamicObject(
            target_dynamic_object_id=record.target_dynamic_object_id,
            source_object_id=record.source_object_id,
            snippet_id=record.source_snippet_id,
            object_type=record.object_type,
            footprint_spec=dict(record.footprint_spec),
            footprint_spec_digest=record.footprint_spec_digest,
            history_poses=record.history_poses.copy(),
            current_pose=record.current_pose.copy(),
            future_poses=target_future,
            provenance={
                "target_type_policy_digest": record.target_type_policy_digest,
            },
        )
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
        target_history_visibility = np.zeros(grid.history_steps, dtype=bool)
        visibility_sequence = np.asarray(
            [False] * 12 + [True] * 4, dtype=bool
        )
        target_history_digest = compute_motion_array_digest(
            target.history_poses, field_name="target_history_poses"
        )
        target_future_digest = compute_motion_array_digest(
            target.future_poses, field_name="target_future_poses"
        )
        target_motion_digest = (
            paired_variants_module._paired_target_motion_digest(target)
        )
        target_current_pose = [float(value) for value in target.current_pose]
        target_provenance = dict(target.provenance)
    world_id = "world-" + stable_digest(
        pair_group_id,
        variant_kind,
        target_motion_digest,
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
            "generator_algorithm_version": "blind_reachability_first_v1",
            "paired_generator_algorithm_version": (
                PAIRED_GENERATOR_ALGORITHM_VERSION
            ),
            "pair_group_contract_version": PAIRED_GROUP_CONTRACT_VERSION,
            "world_id": world_id,
            "mother_generated_event_id": record.generated_event_id,
            "mother_world_id": mother_world.world_id,
            "mother_target_motion_record_digest": record.record_digest,
            "mother_source_snippet_id": record.source_snippet_id,
            "mother_source_object_id": record.source_object_id,
            "mother_target_type_policy_digest": (
                record.target_type_policy_digest
            ),
            "mother_target_footprint_spec_digest": (
                record.footprint_spec_digest
            ),
            "pair_group_id": pair_group_id,
            "paired_variant_kind": variant_kind,
            "paired_config_digest": paired_config_digest,
            "paired_seed": paired_seed,
            "target_dynamic_object_id": record.target_dynamic_object_id,
            "target_present": not empty,
            "paired_target_history_array_digest": target_history_digest,
            "paired_target_future_array_digest": target_future_digest,
            "paired_target_motion_digest": target_motion_digest,
            "paired_target_current_pose": target_current_pose,
            "target_provenance": target_provenance,
            "visibility_sequence": (
                None
                if visibility_sequence is None
                else [bool(value) for value in visibility_sequence]
            ),
            "target_visibility_history": (
                None
                if target_history_visibility is None
                else [bool(value) for value in target_history_visibility]
            ),
        },
    )
    variant = PairedVariant(
        variant_kind=variant_kind,
        world=paired_world,
        target=target,
        target_visibility_history=target_history_visibility,
        visibility_sequence=visibility_sequence,
        clearance_sequence_m=(
            None if empty else np.zeros(grid.future_steps, dtype=np.float32)
        ),
        min_clearance_m=None if empty else -0.1,
        time_to_min_clearance_s=None if empty else 1.0,
    )
    return {
        "mother_record": record,
        "mother_world": mother_world,
        "variant": variant,
        "base_state": base_state,
        "trajectory": _trajectory(),
        "oracle_context": oracle_context,
        "base_config": config,
        "expected_paired_config_digest": paired_config_digest,
        "source_session_id": "session-risk-adapter",
        "seed_namespace": "split/train/risk-adapter",
    }


def test_sop06_adapter_builds_target_present_history_only_risk_input() -> None:
    inputs = _formal_sop06_variant_inputs(include_context=True)

    source = build_risk_input_from_sop06_variant(**inputs)

    variant = inputs["variant"]
    assert source.pair_group_id == variant.world.metadata["pair_group_id"]
    assert source.event_type == "collision"
    assert source.hidden_object_ids == (TARGET_ID,)
    assert set(source.scene_dynamic_history) == {CONTEXT_ID, TARGET_ID}
    assert set(source.scene_dynamic_specs) == {CONTEXT_ID, TARGET_ID}
    np.testing.assert_array_equal(
        source.scene_dynamic_history[CONTEXT_ID],
        inputs["oracle_context"].dynamic_object_history[CONTEXT_ID],
    )
    np.testing.assert_array_equal(
        source.scene_dynamic_history[TARGET_ID],
        variant.target.history_poses,
    )
    assert source.sensor_config is None
    np.testing.assert_array_equal(
        source.observed_static_occupancy,
        variant.world.static_occupancy,
    )
    assert not np.shares_memory(
        source.observed_static_occupancy,
        variant.world.static_occupancy,
    )
    assert source.provenance["source_recording_id"] == (
        inputs["base_state"].recording_id
    )
    assert source.provenance["session_id"] == "session-risk-adapter"
    assert source.provenance["source_snippet_id"] == "snippet-risk-adapter"
    assert source.provenance["seed_namespace"] == (
        "split/train/risk-adapter"
    )
    assert not _metadata_has_forbidden_payload(source.provenance)

    sample = build_risk_sample(
        source,
        base_config=inputs["base_config"],
        risk_config=_risk_config(),
    )
    assert sample.collision_label == 1


def test_sop06_adapter_empty_variant_keeps_context_and_has_no_hidden_target(
) -> None:
    inputs = _formal_sop06_variant_inputs(empty=True, include_context=True)

    source = build_risk_input_from_sop06_variant(**inputs)

    assert source.event_type == "empty_blind_spot"
    assert source.hidden_object_ids == ()
    assert set(source.scene_dynamic_history) == {CONTEXT_ID}
    assert TARGET_ID not in source.scene_dynamic_history
    sample = build_risk_sample(
        source,
        base_config=inputs["base_config"],
        risk_config=_risk_config(),
    )
    assert sample.collision_label == 0
    assert sample.near_miss == 0
    assert sample.risk_severity == 0.0


def test_sop06_adapter_target_future_changes_labels_not_model_input_or_provenance(
) -> None:
    first_inputs = _formal_sop06_variant_inputs(target_future_y_m=0.0)
    second_inputs = _formal_sop06_variant_inputs(target_future_y_m=1.5)

    first_source = build_risk_input_from_sop06_variant(**first_inputs)
    second_source = build_risk_input_from_sop06_variant(**second_inputs)
    first = build_risk_sample(
        first_source,
        base_config=first_inputs["base_config"],
        risk_config=_risk_config(),
    )
    second = build_risk_sample(
        second_source,
        base_config=second_inputs["base_config"],
        risk_config=_risk_config(),
    )

    assert first_source.sample_id != second_source.sample_id
    assert first_source.provenance == second_source.provenance
    assert not _metadata_has_forbidden_payload(first_source.provenance)
    for field in (
        "bev_history",
        "state_channels",
        "trajectory_channels",
        "robot_state",
    ):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    assert first.collision_label == 1
    assert second.collision_label == 0


def test_sop06_adapter_sample_id_is_stable_and_binds_config_world_and_namespace(
) -> None:
    inputs = _formal_sop06_variant_inputs()
    reordered = dict(inputs)
    reordered["base_config"] = dict(
        reversed(list(inputs["base_config"].items()))
    )

    baseline = build_risk_input_from_sop06_variant(**inputs)
    repeat = build_risk_input_from_sop06_variant(**inputs)
    reordered_source = build_risk_input_from_sop06_variant(**reordered)
    changed_namespace = build_risk_input_from_sop06_variant(
        **{**inputs, "seed_namespace": "split/train/risk-adapter-v2"}
    )
    changed_config = build_risk_input_from_sop06_variant(
        **_formal_sop06_variant_inputs(paired_config_digest="c" * 32)
    )
    changed_world = build_risk_input_from_sop06_variant(
        **_formal_sop06_variant_inputs(paired_seed=29)
    )

    assert baseline.sample_id == repeat.sample_id == reordered_source.sample_id
    assert len(
        {
            baseline.sample_id,
            changed_namespace.sample_id,
            changed_config.sample_id,
            changed_world.sample_id,
        }
    ) == 4


def test_sop06_adapter_rejects_retired_joint_identity() -> None:
    inputs = _formal_sop06_variant_inputs()
    mother = inputs["mother_world"]
    inputs["mother_world"] = replace(
        mother,
        metadata={
            **mother.metadata,
            "joint_pair_generator_algorithm_version": (
                "joint_environment_pair_v2"
            ),
        },
    )

    with pytest.raises(ValueError, match="retired joint_environment_pair_v2"):
        build_risk_input_from_sop06_variant(**inputs)


def test_sop06_adapter_rejects_record_base_join_mismatch() -> None:
    inputs = _formal_sop06_variant_inputs()
    inputs["base_state"] = replace(
        inputs["base_state"], state_id="base-risk-adapter-tampered"
    )

    with pytest.raises(ValueError, match="base_state id"):
        build_risk_input_from_sop06_variant(**inputs)


def test_sop06_adapter_rejects_paired_lineage_tampering() -> None:
    inputs = _formal_sop06_variant_inputs()
    variant = inputs["variant"]
    inputs["variant"] = replace(
        variant,
        world=replace(
            variant.world,
            metadata={
                **variant.world.metadata,
                "mother_world_id": "world-risk-adapter-tampered",
            },
        ),
    )

    with pytest.raises(ValueError, match="mother_world_id"):
        build_risk_input_from_sop06_variant(**inputs)


def test_sop06_adapter_rejects_untrusted_paired_config_digest() -> None:
    inputs = _formal_sop06_variant_inputs()
    inputs["expected_paired_config_digest"] = "c" * 32

    with pytest.raises(ValueError, match="paired_config_digest"):
        build_risk_input_from_sop06_variant(**inputs)


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
