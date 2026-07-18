"""Strict SOP05-to-SOP06 observation-pipeline integration tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import src.generation.paired_variants as paired_variants_module
from src.contracts import (
    HISTORY_CHANNELS,
    BaseState,
    OracleContext,
    OracleWorld,
    build_grid_spec,
)
from src.generation.event_target_motion_shard import (
    build_event_target_motion_world_metadata,
    compute_motion_array_digest,
    compute_footprint_spec_digest,
    create_event_target_motion_record,
)
from src.generation.dynamic_object_transplant import (
    TransplantedDynamicObject,
    footprint_from_spec,
)
from src.generation.paired_variants import PairedVariant
from src.generation.sop06_pipeline import render_sop06_variant
from src.generation.structural_blindspot import (
    StructuralBlindSpot,
    footprint_visibility_sequence,
    has_continuous_emergence,
)
from src.geometry import (
    RectangleFootprint,
    rasterize_footprint,
    raycast_visibility,
    world_to_grid,
)
from src.utils.seeding import stable_digest


TARGET_ID = "generated::human::sop06-pipeline"
CONTEXT_ID = "recording::context-human"
BASE_ONLY_ID = "recording::base-only-visible"
ORACLE_ONLY_ID = "recording::oracle-only-context"
OCCLUDER_ID = "occluder::pipeline-wall"
PAIRED_CONFIG_DIGEST = "b" * 32


def _config() -> dict[str, object]:
    return {
        "schema_version": "2.0.0",
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


def _inputs(
    *,
    suffix: str = "a",
    future_y_offset_m: float = 0.0,
    structural: dict[str, object] | None = None,
) -> dict[str, object]:
    config = _config()
    grid = build_grid_spec(config)
    target_history = np.tile(
        np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
        (grid.history_steps, 1),
    )
    target_future = np.column_stack(
        (
            np.float32(2.0)
            + np.arange(1, grid.future_steps + 1, dtype=np.float32)
            * np.float32(0.1),
            np.full(
                grid.future_steps,
                np.float32(future_y_offset_m),
                dtype=np.float32,
            ),
            np.zeros(grid.future_steps, dtype=np.float32),
        )
    ).astype(np.float32)
    target_spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.25},
    }
    record = create_event_target_motion_record(
        generated_event_id=f"event-{suffix}",
        world_id=f"world-{suffix}",
        base_state_id="base-shared",
        trajectory_id="trajectory-shared",
        target_dynamic_object_id=TARGET_ID,
        source_snippet_id="snippet-shared",
        source_object_id="recording::source-human",
        object_type="human",
        footprint_spec=target_spec,
        footprint_spec_digest=compute_footprint_spec_digest(target_spec),
        target_type_policy_digest="a" * 32,
        history_poses=target_history,
        current_pose=target_history[-1].copy(),
        future_poses=target_future,
    )

    context_history = np.tile(
        np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
        (grid.history_steps, 1),
    )
    context_future = np.tile(
        np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
        (grid.future_steps, 1),
    )
    context_spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.2},
    }
    base_static = np.zeros((grid.height, grid.width), dtype=np.float32)
    if structural is None:
        occluder_pose = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        occluder_footprint = RectangleFootprint(0.8, 0.8)
        world_static = rasterize_footprint(
            occluder_footprint, occluder_pose, grid
        ).astype(np.float32)
        occluders = (
            {
                "occluder_id": OCCLUDER_ID,
                "type": "pillar",
                "pose": [float(value) for value in occluder_pose],
                "length_m": 0.8,
                "width_m": 0.8,
                "placement_strategy": "joint_multi_los_envelope_v2",
            },
        )
        event_kind = "environment"
    else:
        world_static = base_static.copy()
        occluders = ()
        event_kind = "structural"
    base_state = BaseState(
        state_id=record.base_state_id,
        split="train",
        recording_id="recording",
        dynamic_object_ids=(CONTEXT_ID,),
        timestamp=4.0,
        robot_history=np.zeros((grid.history_steps, 3), dtype=np.float32),
        robot_state=np.asarray([0.5, 0.1], dtype=np.float32),
        visible_dynamic_object_history={CONTEXT_ID: context_history.copy()},
        visible_dynamic_object_specs={CONTEXT_ID: dict(context_spec)},
        static_map_local=base_static.copy(),
        metadata={"coordinate_frame": "robot_current"},
    )
    oracle_context = OracleContext(
        base_state_id=record.base_state_id,
        dynamic_object_history={CONTEXT_ID: context_history.copy()},
        dynamic_object_future={CONTEXT_ID: context_future.copy()},
        dynamic_object_specs={CONTEXT_ID: dict(context_spec)},
        metadata={"future_dt_s": 0.2},
    )
    world_metadata = {
        **build_event_target_motion_world_metadata(record),
        "schema_version": "2.0.0",
        "event_kind": event_kind,
        "joint_pair_generator_algorithm_version": (
            "joint_environment_pair_v2"
        ),
    }
    world = OracleWorld(
        world_id=record.world_id,
        base_state_id=record.base_state_id,
        static_occupancy=world_static,
        dynamic_object_trajectories={
            CONTEXT_ID: context_future.copy(),
            TARGET_ID: record.future_poses.copy(),
        },
        dynamic_object_specs={
            CONTEXT_ID: dict(context_spec),
            TARGET_ID: dict(record.footprint_spec),
        },
        occluders=occluders,
        blind_spot_config={
            "kind": event_kind,
            "structural": structural,
            "occluder_ids": [
                occluder["occluder_id"] for occluder in occluders
            ],
        },
        random_seed=17,
        metadata=world_metadata,
    )
    return {
        "record": record,
        "world": world,
        "base_state": base_state,
        "oracle_context": oracle_context,
        "config": config,
    }


def test_present_variant_uses_only_history_at_the_core_renderer_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.generation.sop06_pipeline as pipeline

    first = _inputs(suffix="a", future_y_offset_m=0.0)
    second = _inputs(suffix="b", future_y_offset_m=0.75)
    real_renderer = pipeline.render_observation
    calls: list[dict[str, object]] = []

    def guarded_renderer(base_state: BaseState, **kwargs: object):
        assert set(kwargs) == {
            "scene_dynamic_history",
            "scene_dynamic_specs",
            "static_occupancy",
            "sensor_config",
            "config",
        }
        assert not any(
            token in key
            for key in kwargs
            for token in ("future", "oracle", "world", "trajectory")
        )
        calls.append(kwargs)
        return real_renderer(base_state, **kwargs)

    monkeypatch.setattr(pipeline, "render_observation", guarded_renderer)
    rendered_first = render_sop06_variant(**first)
    rendered_second = render_sop06_variant(**second)

    assert len(calls) == 2
    assert set(calls[0]["scene_dynamic_history"]) == {CONTEXT_ID, TARGET_ID}
    np.testing.assert_array_equal(
        calls[0]["scene_dynamic_history"][TARGET_ID],
        first["record"].history_poses,
    )
    np.testing.assert_array_equal(
        rendered_first.bev_history,
        rendered_second.bev_history,
    )
    np.testing.assert_array_equal(
        rendered_first.state_channels,
        rendered_second.state_channels,
    )
    assert rendered_first.metadata == rendered_second.metadata


def test_mother_entry_always_includes_target_and_independently_rerenders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.generation.sop06_pipeline as pipeline

    inputs = _inputs()
    real_renderer = pipeline.render_observation
    scene_ids: list[tuple[str, ...]] = []

    def recording_renderer(base_state: BaseState, **kwargs: object):
        histories = kwargs["scene_dynamic_history"]
        scene_ids.append(tuple(histories))
        return real_renderer(base_state, **kwargs)

    monkeypatch.setattr(pipeline, "render_observation", recording_renderer)
    first = render_sop06_variant(**inputs)
    second = render_sop06_variant(**inputs)

    assert scene_ids == [
        (CONTEXT_ID, TARGET_ID),
        (CONTEXT_ID, TARGET_ID),
    ]
    np.testing.assert_array_equal(
        first.bev_history,
        second.bev_history,
    )
    np.testing.assert_array_equal(
        first.state_channels,
        second.state_channels,
    )
    assert first.bev_history.flags.owndata
    assert second.bev_history.flags.owndata
    assert not np.shares_memory(first.bev_history, second.bev_history)


def test_pipeline_recovers_structural_sensor_and_passes_world_occupancy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.generation.sop06_pipeline as pipeline

    structural = {
        "forward_fov_deg": 180.0,
        "range_m": 4.0,
        "blind_sectors": [{"center_deg": 90.0, "width_deg": 30.0}],
    }
    inputs = _inputs(structural=structural)
    real_renderer = pipeline.render_observation
    captured: dict[str, object] = {}

    def recording_renderer(base_state: BaseState, **kwargs: object):
        captured.update(kwargs)
        return real_renderer(base_state, **kwargs)

    monkeypatch.setattr(pipeline, "render_observation", recording_renderer)
    rendered = render_sop06_variant(**inputs)

    sensor = captured["sensor_config"]
    assert isinstance(sensor, StructuralBlindSpot)
    assert sensor.as_dict() == structural
    assert captured["static_occupancy"] is inputs["world"].static_occupancy
    assert not np.shares_memory(
        rendered.state_channels,
        inputs["world"].static_occupancy,
    )


def test_mother_entry_rejects_target_present_switch() -> None:
    with pytest.raises(
        TypeError,
        match="unexpected keyword argument.*target_present",
    ):
        render_sop06_variant(**_inputs(), target_present=False)


@pytest.mark.parametrize("join_kind", ["world", "base", "context"])
def test_pipeline_rejects_identity_join_tampering(join_kind: str) -> None:
    inputs = _inputs()
    if join_kind == "world":
        metadata = dict(inputs["world"].metadata)
        metadata["generated_event_id"] = "event-tampered"
        inputs["world"] = replace(inputs["world"], metadata=metadata)
        expected = "generated_event_id"
    elif join_kind == "base":
        inputs["base_state"] = replace(
            inputs["base_state"], state_id="base-tampered"
        )
        expected = "base_state"
    else:
        inputs["oracle_context"] = replace(
            inputs["oracle_context"], base_state_id="base-tampered"
        )
        expected = "oracle_context"

    with pytest.raises(ValueError, match=expected):
        render_sop06_variant(**inputs)


def test_pipeline_rejects_context_future_drift_before_rendering() -> None:
    inputs = _inputs()
    trajectories = {
        object_id: poses.copy()
        for object_id, poses in inputs["world"].dynamic_object_trajectories.items()
    }
    trajectories[CONTEXT_ID][0, 0] += np.float32(0.5)
    inputs["world"] = replace(
        inputs["world"], dynamic_object_trajectories=trajectories
    )

    with pytest.raises(ValueError, match="context future"):
        render_sop06_variant(**inputs)


def test_pipeline_rejects_malformed_structural_sensor_contract() -> None:
    inputs = _inputs(structural={
        "forward_fov_deg": 180.0,
        "range_m": 4.0,
        "blind_sectors": [],
    })
    blind_spot = dict(inputs["world"].blind_spot_config)
    blind_spot["structural"] = {
        **blind_spot["structural"],
        "unexpected": 1,
    }
    inputs["world"] = replace(
        inputs["world"], blind_spot_config=blind_spot
    )

    with pytest.raises(ValueError, match="structural blind-spot keys"):
        render_sop06_variant(**inputs)


def _paired_variant(
    inputs: dict[str, object], *, empty: bool
) -> PairedVariant:
    record = inputs["record"]
    mother_world = inputs["world"]
    paired_config_digest = PAIRED_CONFIG_DIGEST
    paired_seed = 23
    pair_group_id = paired_variants_module.compute_pair_group_id(
        generated_event_id=record.generated_event_id,
        base_state_id=mother_world.base_state_id,
        trajectory_id=record.trajectory_id,
        occluders=mother_world.occluders,
        blind_spot_config=mother_world.blind_spot_config,
        source_snippet_id=record.source_snippet_id,
        target_dynamic_object_id=record.target_dynamic_object_id,
        paired_config_digest=paired_config_digest,
    )
    if empty:
        trajectories = {
            object_id: poses.copy()
            for object_id, poses in mother_world.dynamic_object_trajectories.items()
            if object_id != record.target_dynamic_object_id
        }
        specs = {
            object_id: dict(spec)
            for object_id, spec in mother_world.dynamic_object_specs.items()
            if object_id != record.target_dynamic_object_id
        }
        target = None
        kind = "empty_blind_spot"
        history_visibility = None
        visibility_sequence = None
        history_digest = None
        future_digest = None
        motion_digest = "target-empty"
        current_pose = None
    else:
        future = record.future_poses.copy()
        future[:, 0] = np.linspace(
            1.9, 2.0, future.shape[0], dtype=np.float32
        )
        future[:, 1] = np.linspace(
            -0.15, -3.0, future.shape[0], dtype=np.float32
        )
        target = TransplantedDynamicObject(
            target_dynamic_object_id=record.target_dynamic_object_id,
            source_object_id=record.source_object_id,
            snippet_id=record.source_snippet_id,
            object_type=record.object_type,
            footprint_spec=dict(record.footprint_spec),
            footprint_spec_digest=record.footprint_spec_digest,
            history_poses=record.history_poses.copy(),
            current_pose=record.current_pose.copy(),
            future_poses=future,
            provenance={
                "target_type_policy_digest": record.target_type_policy_digest,
            },
        )
        kind = "collision"
        trajectories = {
            object_id: poses.copy()
            for object_id, poses in mother_world.dynamic_object_trajectories.items()
        }
        trajectories[record.target_dynamic_object_id] = future.copy()
        specs = {
            object_id: dict(spec)
            for object_id, spec in mother_world.dynamic_object_specs.items()
        }
        grid = build_grid_spec(inputs["config"])
        occupied = mother_world.static_occupancy != 0.0
        for object_id in sorted(inputs["oracle_context"].dynamic_object_history):
            occupied |= rasterize_footprint(
                footprint_from_spec(
                    inputs["oracle_context"].dynamic_object_specs[object_id]
                ),
                inputs["oracle_context"].dynamic_object_history[object_id][-1],
                grid,
            )
        visible = raycast_visibility(
            occupied,
            grid,
            sensor_pose=np.zeros(3, dtype=np.float32),
        )
        footprint = footprint_from_spec(target.footprint_spec)
        history_visibility = footprint_visibility_sequence(
            footprint, target.history_poses, visible, grid
        )
        visibility_sequence = footprint_visibility_sequence(
            footprint,
            np.vstack((target.current_pose, target.future_poses)),
            visible,
            grid,
        )
        assert not bool(visibility_sequence[0])
        assert has_continuous_emergence(
            visibility_sequence, min_visible_frames=2
        )
        assert bool(visibility_sequence[-1])
        assert history_visibility[-1] == visibility_sequence[0]
        history_digest = compute_motion_array_digest(
            target.history_poses, field_name="target_history_poses"
        )
        future_digest = compute_motion_array_digest(
            target.future_poses, field_name="target_future_poses"
        )
        motion_digest = paired_variants_module._paired_target_motion_digest(
            target
        )
        current_pose = [float(value) for value in target.current_pose]
    world_id = "world-" + stable_digest(
        pair_group_id,
        kind,
        motion_digest,
        paired_seed,
        paired_config_digest,
        size=12,
    )
    mother_metadata = {
        key: value
        for key, value in mother_world.metadata.items()
        if key not in paired_variants_module._SOP05_JOIN_METADATA_KEYS
    }
    pair_world = replace(
        mother_world,
        world_id=world_id,
        dynamic_object_trajectories=trajectories,
        dynamic_object_specs=specs,
        random_seed=paired_seed,
        metadata={
            **mother_metadata,
            "schema_version": "2.0.0",
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
            "paired_variant_kind": kind,
            "paired_config_digest": paired_config_digest,
            "paired_seed": paired_seed,
            "target_dynamic_object_id": record.target_dynamic_object_id,
            "target_present": not empty,
            "paired_target_history_array_digest": history_digest,
            "paired_target_future_array_digest": future_digest,
            "paired_target_motion_digest": motion_digest,
            "paired_target_current_pose": current_pose,
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
                if history_visibility is None
                else [bool(value) for value in history_visibility]
            ),
            "min_clearance_m": None if empty else -0.1,
            "time_to_min_clearance_s": None if empty else 1.0,
            "paired_transform": {"kind": "minimal_real_fixture"},
        },
    )
    return PairedVariant(
        variant_kind=kind,
        world=pair_world,
        target=target,
        target_visibility_history=history_visibility,
        visibility_sequence=visibility_sequence,
        clearance_sequence_m=(
            None if empty else np.zeros(15, dtype=np.float32)
        ),
        min_clearance_m=None if empty else -0.1,
        time_to_min_clearance_s=None if empty else 1.0,
    )


def test_pipeline_accepts_real_paired_variant_and_empty_without_boolean_switch(
    monkeypatch: pytest.MonkeyPatch,
    ) -> None:
    import src.generation.sop06_pipeline as pipeline
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    present_variant = _paired_variant(inputs, empty=False)
    empty_variant = _paired_variant(inputs, empty=True)
    real_renderer = pipeline.render_observation
    scene_ids: list[tuple[str, ...]] = []

    def recording_renderer(base_state: BaseState, **kwargs: object):
        scene_ids.append(tuple(kwargs["scene_dynamic_history"]))
        return real_renderer(base_state, **kwargs)

    monkeypatch.setattr(pipeline, "render_observation", recording_renderer)
    common = {
        "mother_record": inputs["record"],
        "mother_world": inputs["world"],
        "base_state": inputs["base_state"],
        "oracle_context": inputs["oracle_context"],
        "config": inputs["config"],
        "expected_paired_config_digest": PAIRED_CONFIG_DIGEST,
    }
    present = render_sop06_paired_variant(
        **common, variant=present_variant
    )
    empty = render_sop06_paired_variant(**common, variant=empty_variant)

    assert scene_ids == [(CONTEXT_ID, TARGET_ID), (CONTEXT_ID,)]
    assert present.bev_history.flags.owndata
    assert empty.bev_history.flags.owndata
    assert not np.shares_memory(present.bev_history, empty.bev_history)


def _retag_joint_pair_versions(
    inputs: dict[str, object],
    variant: PairedVariant,
    *,
    algorithm_version: str,
    placement_strategy: str,
) -> tuple[OracleWorld, PairedVariant]:
    mother_world = inputs["world"]
    mother_occluder = {
        **mother_world.occluders[0],
        "placement_strategy": placement_strategy,
    }
    mother_metadata = {
        **mother_world.metadata,
        "joint_pair_generator_algorithm_version": algorithm_version,
    }
    changed_mother = replace(
        mother_world,
        occluders=(mother_occluder,),
        metadata=mother_metadata,
    )
    metadata = dict(variant.world.metadata)
    pair_group_id = paired_variants_module.compute_pair_group_id(
        generated_event_id=inputs["record"].generated_event_id,
        base_state_id=changed_mother.base_state_id,
        trajectory_id=inputs["record"].trajectory_id,
        occluders=changed_mother.occluders,
        blind_spot_config=changed_mother.blind_spot_config,
        source_snippet_id=inputs["record"].source_snippet_id,
        target_dynamic_object_id=inputs["record"].target_dynamic_object_id,
        paired_config_digest=metadata["paired_config_digest"],
    )
    world_id = "world-" + stable_digest(
        pair_group_id,
        variant.variant_kind,
        metadata["paired_target_motion_digest"],
        metadata["paired_seed"],
        metadata["paired_config_digest"],
        size=12,
    )
    changed_variant = replace(
        variant,
        world=replace(
            variant.world,
            world_id=world_id,
            occluders=(dict(mother_occluder),),
            metadata={
                **metadata,
                "world_id": world_id,
                "pair_group_id": pair_group_id,
                "joint_pair_generator_algorithm_version": algorithm_version,
            },
        ),
    )
    return changed_mother, changed_variant


@pytest.mark.parametrize(
    ("algorithm_version", "placement_strategy", "message"),
    [
        (
            "joint_environment_pair_v1",
            "joint_multi_los_envelope_v2",
            "joint_pair_generator_algorithm_version",
        ),
        (
            "joint_environment_pair_v2",
            "joint_multi_los_envelope_v1",
            "placement_strategy",
        ),
    ],
)
def test_strict_six_pack_consumer_rejects_self_consistent_old_joint_versions(
    algorithm_version: str,
    placement_strategy: str,
    message: str,
) -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    mother_world, variant = _retag_joint_pair_versions(
        inputs,
        variant,
        algorithm_version=algorithm_version,
        placement_strategy=placement_strategy,
    )

    with pytest.raises(ValueError, match=message):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=mother_world,
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


def _inputs_with_split_background() -> dict[str, object]:
    inputs = _inputs()
    grid = build_grid_spec(inputs["config"])
    base_only_history = np.tile(
        np.asarray([-2.0, 0.0, 0.0], dtype=np.float32),
        (grid.history_steps, 1),
    )
    oracle_only_history = np.tile(
        np.asarray([0.0, -2.0, 0.0], dtype=np.float32),
        (grid.history_steps, 1),
    )
    oracle_only_future = np.tile(
        np.asarray([0.0, -2.0, 0.0], dtype=np.float32),
        (grid.future_steps, 1),
    )
    spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.2},
    }
    base = inputs["base_state"]
    inputs["base_state"] = replace(
        base,
        dynamic_object_ids=tuple(
            sorted(base.dynamic_object_ids + (BASE_ONLY_ID,))
        ),
        visible_dynamic_object_history={
            **base.visible_dynamic_object_history,
            BASE_ONLY_ID: base_only_history,
        },
        visible_dynamic_object_specs={
            **base.visible_dynamic_object_specs,
            BASE_ONLY_ID: dict(spec),
        },
    )
    context = inputs["oracle_context"]
    inputs["oracle_context"] = replace(
        context,
        dynamic_object_history={
            **context.dynamic_object_history,
            ORACLE_ONLY_ID: oracle_only_history,
        },
        dynamic_object_future={
            **context.dynamic_object_future,
            ORACLE_ONLY_ID: oracle_only_future,
        },
        dynamic_object_specs={
            **context.dynamic_object_specs,
            ORACLE_ONLY_ID: dict(spec),
        },
    )
    world = inputs["world"]
    inputs["world"] = replace(
        world,
        dynamic_object_trajectories={
            **world.dynamic_object_trajectories,
            ORACLE_ONLY_ID: oracle_only_future.copy(),
        },
        dynamic_object_specs={
            **world.dynamic_object_specs,
            ORACLE_ONLY_ID: dict(spec),
        },
    )
    return inputs


def test_paired_scene_preserves_base_only_and_merges_oracle_only_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.generation.sop06_pipeline as pipeline
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs_with_split_background()
    present_variant = _paired_variant(inputs, empty=False)
    empty_variant = _paired_variant(inputs, empty=True)
    real_renderer = pipeline.render_observation
    scene_ids: list[tuple[str, ...]] = []

    def recording_renderer(base_state: BaseState, **kwargs: object):
        scene_ids.append(tuple(kwargs["scene_dynamic_history"]))
        return real_renderer(base_state, **kwargs)

    monkeypatch.setattr(pipeline, "render_observation", recording_renderer)
    common = {
        "mother_record": inputs["record"],
        "mother_world": inputs["world"],
        "base_state": inputs["base_state"],
        "oracle_context": inputs["oracle_context"],
        "config": inputs["config"],
        "expected_paired_config_digest": PAIRED_CONFIG_DIGEST,
    }
    render_sop06_paired_variant(**common, variant=present_variant)
    render_sop06_paired_variant(**common, variant=empty_variant)

    assert scene_ids == [
        (BASE_ONLY_ID, CONTEXT_ID, ORACLE_ONLY_ID, TARGET_ID),
        (BASE_ONLY_ID, CONTEXT_ID, ORACLE_ONLY_ID),
    ]


def test_pipeline_rejects_overlap_history_drift_before_core_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.generation.sop06_pipeline as pipeline
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs_with_split_background()
    context = inputs["oracle_context"]
    histories = {
        object_id: poses.copy()
        for object_id, poses in context.dynamic_object_history.items()
    }
    histories[CONTEXT_ID][0, 0] += np.float32(0.25)
    inputs["oracle_context"] = replace(
        context, dynamic_object_history=histories
    )
    variant = _paired_variant(inputs, empty=False)

    def forbidden_renderer(*args: object, **kwargs: object):
        raise AssertionError("core renderer must not receive a bad overlap")

    monkeypatch.setattr(pipeline, "render_observation", forbidden_renderer)
    with pytest.raises(ValueError, match="overlapping.*history"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


def test_environment_fixture_has_real_occluder_and_hides_current_target() -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    rendered = render_sop06_paired_variant(
        mother_record=inputs["record"],
        mother_world=inputs["world"],
        variant=variant,
        base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )
    grid = build_grid_spec(inputs["config"])
    occluder_cell, target_cell = world_to_grid(
        np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32), grid
    )
    visible_channel = HISTORY_CHANNELS.index("past_visible_mask")

    assert inputs["world"].occluders[0]["occluder_id"] == OCCLUDER_ID
    assert inputs["world"].static_occupancy[tuple(occluder_cell)] == 1.0
    assert rendered.bev_history[-1, visible_channel][tuple(target_cell)] == 0.0


@pytest.mark.parametrize(
    ("tamper_kind", "message"),
    [
        ("metadata_event_kind", "event_kind"),
        ("occluder_id", "occluder_ids"),
        ("environment_missing_occluder", "environment.*occluder"),
        ("structural_has_occluder", "structural.*occluder"),
    ],
)
def test_pipeline_closes_event_kind_and_occluder_identity_contract(
    tamper_kind: str,
    message: str,
) -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    if tamper_kind == "structural_has_occluder":
        inputs = _inputs()
        variant = _paired_variant(inputs, empty=False)
        blind_spot = {
            "kind": "structural",
            "structural": {
                "forward_fov_deg": 180.0,
                "range_m": 4.0,
                "blind_sectors": [
                    {"center_deg": 0.0, "width_deg": 30.0}
                ],
            },
            "occluder_ids": [OCCLUDER_ID],
        }
        metadata = dict(inputs["world"].metadata)
        metadata["event_kind"] = "structural"
        inputs["world"] = replace(
            inputs["world"],
            blind_spot_config=blind_spot,
            metadata=metadata,
        )
    else:
        inputs = _inputs()
        variant = _paired_variant(inputs, empty=False)
        if tamper_kind == "metadata_event_kind":
            metadata = dict(inputs["world"].metadata)
            metadata["event_kind"] = "mixed"
            inputs["world"] = replace(inputs["world"], metadata=metadata)
        elif tamper_kind == "occluder_id":
            blind_spot = dict(inputs["world"].blind_spot_config)
            blind_spot["occluder_ids"] = ["occluder::wrong"]
            inputs["world"] = replace(
                inputs["world"], blind_spot_config=blind_spot
            )
        else:
            blind_spot = dict(inputs["world"].blind_spot_config)
            blind_spot["occluder_ids"] = []
            inputs["world"] = replace(
                inputs["world"],
                occluders=(),
                blind_spot_config=blind_spot,
            )

    with pytest.raises(ValueError, match=message):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


def test_pipeline_rejects_paired_occluder_skeleton_drift_from_mother() -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    changed_occluder = dict(variant.world.occluders[0])
    changed_occluder["pose"] = [1.0, 0.25, 0.0]
    variant = replace(
        variant,
        world=replace(variant.world, occluders=(changed_occluder,)),
    )

    with pytest.raises(ValueError, match="paired occluders.*mother"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


def test_pipeline_rejects_early_paired_history_tamper_against_motion_digest(
) -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    changed_history = variant.target.history_poses.copy()
    changed_history[0, 1] += np.float32(0.5)
    variant = replace(
        variant,
        target=replace(variant.target, history_poses=changed_history),
    )

    with pytest.raises(ValueError, match="history.*digest"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


@pytest.mark.parametrize(
    ("metadata_key", "tampered_value"),
    [
        ("world_id", "world-tampered"),
        ("mother_generated_event_id", "event-tampered"),
        ("mother_world_id", "world-mother-tampered"),
        ("mother_target_motion_record_digest", "c" * 32),
        ("mother_source_snippet_id", "snippet-tampered"),
        ("mother_source_object_id", "recording::source-tampered"),
        ("mother_target_type_policy_digest", "c" * 32),
        ("mother_target_footprint_spec_digest", "c" * 32),
        ("target_dynamic_object_id", "generated::human::tampered"),
        ("paired_target_current_pose", [2.25, 0.0, 0.0]),
        ("target_provenance", {"target_type_policy_digest": "c" * 32}),
        ("paired_seed", 24),
        ("paired_config_digest", "c" * 32),
    ],
)
def test_pipeline_rejects_paired_lineage_metadata_tampering(
    metadata_key: str,
    tampered_value: object,
) -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    metadata = dict(variant.world.metadata)
    metadata[metadata_key] = tampered_value
    variant = replace(
        variant,
        world=replace(variant.world, metadata=metadata),
    )

    with pytest.raises(ValueError, match=metadata_key):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


def test_pipeline_rejects_paired_world_id_not_derived_from_lineage() -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    metadata = dict(variant.world.metadata)
    metadata["world_id"] = "world-tampered"
    variant = replace(
        variant,
        world=replace(
            variant.world,
            world_id="world-tampered",
            metadata=metadata,
        ),
    )

    with pytest.raises(ValueError, match="world_id.*lineage"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


def test_pipeline_rejects_pair_group_and_world_id_synchronized_tamper() -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    metadata = dict(variant.world.metadata)
    metadata["pair_group_id"] = "pair-" + "c" * 24
    changed_world_id = "world-" + stable_digest(
        metadata["pair_group_id"],
        variant.variant_kind,
        metadata["paired_target_motion_digest"],
        metadata["paired_seed"],
        metadata["paired_config_digest"],
        size=12,
    )
    metadata["world_id"] = changed_world_id
    variant = replace(
        variant,
        world=replace(
            variant.world,
            world_id=changed_world_id,
            metadata=metadata,
        ),
    )

    with pytest.raises(ValueError, match="pair_group_id.*mother lineage"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


def test_pipeline_rejects_config_pair_and_world_id_synchronized_tamper() -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    metadata = dict(variant.world.metadata)
    expected_paired_config_digest = PAIRED_CONFIG_DIGEST
    changed_config_digest = "c" * 32
    changed_pair_group_id = paired_variants_module.compute_pair_group_id(
        generated_event_id=inputs["record"].generated_event_id,
        base_state_id=inputs["world"].base_state_id,
        trajectory_id=inputs["record"].trajectory_id,
        occluders=inputs["world"].occluders,
        blind_spot_config=inputs["world"].blind_spot_config,
        source_snippet_id=inputs["record"].source_snippet_id,
        target_dynamic_object_id=(
            inputs["record"].target_dynamic_object_id
        ),
        paired_config_digest=changed_config_digest,
    )
    changed_world_id = "world-" + stable_digest(
        changed_pair_group_id,
        variant.variant_kind,
        metadata["paired_target_motion_digest"],
        metadata["paired_seed"],
        changed_config_digest,
        size=12,
    )
    metadata.update(
        {
            "paired_config_digest": changed_config_digest,
            "pair_group_id": changed_pair_group_id,
            "world_id": changed_world_id,
        }
    )
    variant = replace(
        variant,
        world=replace(
            variant.world,
            world_id=changed_world_id,
            metadata=metadata,
        ),
    )

    with pytest.raises(ValueError, match="paired_config_digest.*expected"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=expected_paired_config_digest,
        )


def test_pipeline_rejects_variant_kind_and_world_id_synchronized_tamper() -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    changed_kind = "counterfeit_variant"
    metadata = dict(variant.world.metadata)
    metadata["paired_variant_kind"] = changed_kind
    changed_world_id = "world-" + stable_digest(
        metadata["pair_group_id"],
        changed_kind,
        metadata["paired_target_motion_digest"],
        metadata["paired_seed"],
        metadata["paired_config_digest"],
        size=12,
    )
    metadata["world_id"] = changed_world_id
    variant = replace(
        variant,
        variant_kind=changed_kind,
        world=replace(
            variant.world,
            world_id=changed_world_id,
            metadata=metadata,
        ),
    )

    with pytest.raises(ValueError, match="variant_kind.*six frozen kinds"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


def test_pipeline_rejects_stale_sop05_join_metadata_in_paired_world() -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    metadata = dict(variant.world.metadata)
    metadata["generated_event_id"] = inputs["record"].generated_event_id
    variant = replace(
        variant,
        world=replace(variant.world, metadata=metadata),
    )

    with pytest.raises(ValueError, match="stale SOP05.*generated_event_id"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


@pytest.mark.parametrize(
    "field_name",
    ["target_visibility_history", "visibility_sequence"],
)
def test_pipeline_rejects_paired_visibility_array_metadata_drift(
    field_name: str,
) -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    changed = getattr(variant, field_name).copy()
    changed[0] = ~changed[0]
    variant = replace(variant, **{field_name: changed})

    with pytest.raises(ValueError, match=f"{field_name}.*metadata"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


@pytest.mark.parametrize(
    ("field_name", "mutation", "expected_shape"),
    [
        ("target_visibility_history", "integer_dtype", 8),
        ("target_visibility_history", "short_shape", 8),
        ("visibility_sequence", "integer_dtype", 16),
        ("visibility_sequence", "short_shape", 16),
    ],
)
def test_pipeline_rejects_paired_visibility_array_contract_drift(
    field_name: str,
    mutation: str,
    expected_shape: int,
) -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    changed = getattr(variant, field_name).copy()
    if mutation == "integer_dtype":
        changed = changed.astype(np.uint8)
    else:
        changed = changed[:-1]
    variant = replace(variant, **{field_name: changed})

    with pytest.raises(
        ValueError,
        match=rf"{field_name}.*bool\[{expected_shape}\]",
    ):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


def test_pipeline_rejects_non_boolean_visibility_metadata_values() -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    metadata = dict(variant.world.metadata)
    metadata["target_visibility_history"] = list(
        metadata["target_visibility_history"]
    )
    metadata["target_visibility_history"][0] = int(
        metadata["target_visibility_history"][0]
    )
    variant = replace(
        variant,
        world=replace(variant.world, metadata=metadata),
    )

    with pytest.raises(
        ValueError,
        match="target_visibility_history.*boolean metadata",
    ):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


def test_pipeline_rejects_paired_visibility_seam_drift() -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    history = variant.target_visibility_history.copy()
    history[-1] = ~variant.visibility_sequence[0]
    metadata = dict(variant.world.metadata)
    metadata["target_visibility_history"] = [
        bool(value) for value in history
    ]
    variant = replace(
        variant,
        target_visibility_history=history,
        world=replace(variant.world, metadata=metadata),
    )

    with pytest.raises(ValueError, match="visibility seam"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


def test_pipeline_rejects_paired_target_visible_at_current_frame() -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    history = variant.target_visibility_history.copy()
    sequence = variant.visibility_sequence.copy()
    history[-1] = True
    sequence[0] = True
    metadata = dict(variant.world.metadata)
    metadata["target_visibility_history"] = [
        bool(value) for value in history
    ]
    metadata["visibility_sequence"] = [bool(value) for value in sequence]
    variant = replace(
        variant,
        target_visibility_history=history,
        visibility_sequence=sequence,
        world=replace(variant.world, metadata=metadata),
    )

    with pytest.raises(ValueError, match="current.*hidden"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


def test_pipeline_rejects_paired_target_without_continuous_emergence() -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    sequence = np.zeros(16, dtype=bool)
    metadata = dict(variant.world.metadata)
    metadata["visibility_sequence"] = [bool(value) for value in sequence]
    variant = replace(
        variant,
        visibility_sequence=sequence,
        world=replace(variant.world, metadata=metadata),
    )

    with pytest.raises(ValueError, match="continuous emergence"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


def test_pipeline_rejects_paired_target_not_visible_at_final_frame() -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=False)
    sequence = np.zeros(16, dtype=bool)
    sequence[1:3] = True
    metadata = dict(variant.world.metadata)
    metadata["visibility_sequence"] = [bool(value) for value in sequence]
    variant = replace(
        variant,
        visibility_sequence=sequence,
        world=replace(variant.world, metadata=metadata),
    )

    with pytest.raises(ValueError, match="final.*visible"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


@pytest.mark.parametrize(
    "field_name",
    ["target_visibility_history", "visibility_sequence"],
)
def test_pipeline_rejects_empty_variant_visibility_arrays(
    field_name: str,
) -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=True)
    length = 8 if field_name == "target_visibility_history" else 16
    changed = np.zeros(length, dtype=bool)
    metadata = dict(variant.world.metadata)
    metadata[field_name] = [bool(value) for value in changed]
    variant = replace(
        variant,
        **{field_name: changed},
        world=replace(variant.world, metadata=metadata),
    )

    with pytest.raises(ValueError, match=rf"empty.*{field_name}.*None"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


@pytest.mark.parametrize(
    "metadata_key",
    ["target_visibility_history", "visibility_sequence"],
)
def test_pipeline_rejects_empty_variant_visibility_metadata(
    metadata_key: str,
) -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=True)
    length = 8 if metadata_key == "target_visibility_history" else 16
    metadata = dict(variant.world.metadata)
    metadata[metadata_key] = [False] * length
    variant = replace(
        variant,
        world=replace(variant.world, metadata=metadata),
    )

    with pytest.raises(ValueError, match=rf"empty.*{metadata_key}.*None"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )


def test_pipeline_rejects_empty_variant_with_target_still_in_world() -> None:
    from src.generation.sop06_pipeline import render_sop06_paired_variant

    inputs = _inputs()
    variant = _paired_variant(inputs, empty=True)
    trajectories = dict(variant.world.dynamic_object_trajectories)
    trajectories[TARGET_ID] = inputs["record"].future_poses.copy()
    specs = dict(variant.world.dynamic_object_specs)
    specs[TARGET_ID] = dict(inputs["record"].footprint_spec)
    variant = replace(
        variant,
        world=replace(
            variant.world,
            dynamic_object_trajectories=trajectories,
            dynamic_object_specs=specs,
        ),
    )

    with pytest.raises(ValueError, match="dynamic object ids"):
        render_sop06_paired_variant(
            mother_record=inputs["record"],
            mother_world=inputs["world"],
            variant=variant,
            base_state=inputs["base_state"],
            oracle_context=inputs["oracle_context"],
            config=inputs["config"],
            expected_paired_config_digest=PAIRED_CONFIG_DIGEST,
        )
