"""Strict SOP05 target-motion integration for the history-only SOP06 renderer."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.contracts import (
    SCHEMA_VERSION,
    BaseState,
    OracleContext,
    OracleWorld,
    build_grid_spec,
    validate_base_state,
    validate_oracle_context,
)

from .event_target_motion_shard import (
    EventTargetMotionRecord,
    compute_motion_array_digest,
    validate_event_target_motion_world_join,
)
from .dynamic_object_transplant import TransplantedDynamicObject
from .observation_renderer import RenderedObservation, render_observation
from .occluder_sampler import JOINT_MULTI_LOS_PLACEMENT_STRATEGY_VERSION
from .paired_variants import (
    JOINT_ENVIRONMENT_PAIR_VERSION,
    PAIRED_GENERATOR_ALGORITHM_VERSION,
    PAIRED_GROUP_CONTRACT_VERSION,
    PairedEventGroup,
    PairedVariant,
    VARIANT_ORDER,
    compute_pair_group_id,
    _paired_target_motion_digest,
    _SOP05_JOIN_METADATA_KEYS,
)
from .structural_blindspot import (
    StructuralBlindSpot,
    has_continuous_emergence,
)
from src.utils.seeding import stable_digest


_BLIND_SPOT_KEYS = frozenset({"kind", "structural", "occluder_ids"})
_V5_ENVIRONMENT_BLIND_SPOT_KEYS = frozenset(
    {"kind", "occluder_ids", "blind_region_digest"}
)
_STRUCTURAL_KEYS = frozenset(
    {"forward_fov_deg", "range_m", "blind_sectors"}
)
_EVENT_KINDS = frozenset({"environment", "structural", "mixed"})
_FORMAL_SOP05_GENERATOR_VERSION = "blind_reachability_first_v1"


@dataclass(frozen=True)
class RenderedSop06Group:
    """Rendered group with an explicit audit-certification boundary."""

    pair_group_id: str
    variant_kinds: tuple[str, ...]
    observations: tuple[RenderedObservation, ...]
    coverage_mask: tuple[bool, ...]
    is_complete: bool
    audit_certified: bool


def _validate_joint_six_pack_versions(
    mother_world: OracleWorld,
    paired_world: OracleWorld,
) -> None:
    paired_algorithm = paired_world.metadata.get(
        "paired_generator_algorithm_version"
    )
    if paired_algorithm == PAIRED_GENERATOR_ALGORITHM_VERSION:
        for label, world in (("mother", mother_world), ("paired", paired_world)):
            if world.metadata.get("joint_pair_generator_algorithm_version") == (
                JOINT_ENVIRONMENT_PAIR_VERSION
            ):
                raise ValueError(
                    f"{label} uses retired {JOINT_ENVIRONMENT_PAIR_VERSION}"
                )
        if paired_world.metadata.get("pair_group_contract_version") != (
            PAIRED_GROUP_CONTRACT_VERSION
        ):
            raise ValueError(
                "paired pair_group_contract_version must equal "
                f"{PAIRED_GROUP_CONTRACT_VERSION}"
            )
        if mother_world.metadata.get("generator_algorithm_version") != (
            _FORMAL_SOP05_GENERATOR_VERSION
        ):
            raise ValueError(
                "formal SOP06 mother generator_algorithm_version must equal "
                f"{_FORMAL_SOP05_GENERATOR_VERSION}"
            )
        return
    if mother_world.metadata.get("event_kind") != "environment":
        return
    for label, world in (
        ("mother", mother_world),
        ("paired", paired_world),
    ):
        if world.metadata.get(
            "joint_pair_generator_algorithm_version"
        ) != JOINT_ENVIRONMENT_PAIR_VERSION:
            raise ValueError(
                f"{label} joint_pair_generator_algorithm_version must equal "
                f"{JOINT_ENVIRONMENT_PAIR_VERSION}"
            )
    if len(mother_world.occluders) != 1:
        raise ValueError(
            "joint environment mother must contain exactly one occluder"
    )
    if mother_world.occluders[0].get("placement_strategy") != (
        JOINT_MULTI_LOS_PLACEMENT_STRATEGY_VERSION
    ):
        raise ValueError(
            "joint environment occluder placement_strategy must equal "
            f"{JOINT_MULTI_LOS_PLACEMENT_STRATEGY_VERSION}"
        )


def _sensor_from_world(world: OracleWorld) -> StructuralBlindSpot | None:
    config = world.blind_spot_config
    if not isinstance(config, Mapping):
        raise ValueError("world blind_spot_config must be a mapping")
    is_v5_environment = set(config) == _V5_ENVIRONMENT_BLIND_SPOT_KEYS
    if not is_v5_environment and set(config) != _BLIND_SPOT_KEYS:
        raise ValueError(
            "world blind_spot_config keys do not match a frozen layout"
        )
    kind = config["kind"]
    if kind not in _EVENT_KINDS:
        raise ValueError("world blind_spot_config kind is invalid")
    if not isinstance(world.metadata, Mapping) or (
        world.metadata.get("event_kind") != kind
    ):
        raise ValueError("world metadata event_kind mismatch")
    occluder_ids = config["occluder_ids"]
    if (
        not isinstance(occluder_ids, list)
        or any(not isinstance(value, str) or not value for value in occluder_ids)
        or len(occluder_ids) != len(set(occluder_ids))
    ):
        raise ValueError("world blind_spot_config occluder_ids are invalid")

    world_occluder_ids: list[str] = []
    for index, occluder in enumerate(world.occluders):
        if not isinstance(occluder, Mapping):
            raise ValueError(f"world occluders[{index}] must be a mapping")
        occluder_id = occluder.get("occluder_id")
        if not isinstance(occluder_id, str) or not occluder_id:
            raise ValueError(f"world occluders[{index}] id is invalid")
        world_occluder_ids.append(occluder_id)
    if len(world_occluder_ids) != len(set(world_occluder_ids)):
        raise ValueError("world occluder IDs must be unique")
    if occluder_ids != world_occluder_ids:
        raise ValueError(
            "world blind_spot_config occluder_ids mismatch world.occluders"
        )

    if is_v5_environment:
        digest = config["blind_region_digest"]
        if kind != "environment":
            raise ValueError(
                "v5 blind-region layout is valid only for environment worlds"
            )
        if not isinstance(digest, str) or not digest:
            raise ValueError("v5 blind_region_digest must be a non-empty string")
        if not world_occluder_ids:
            raise ValueError("environment world requires an occluder")
        return None

    raw = config["structural"]
    if kind == "environment":
        if not world_occluder_ids:
            raise ValueError("environment world requires an occluder")
        if raw is not None:
            raise ValueError("environment world cannot contain a structural sensor")
        return None
    if kind == "structural" and world_occluder_ids:
        raise ValueError("structural world cannot contain an occluder")
    if kind == "mixed" and not world_occluder_ids:
        raise ValueError("mixed world requires an occluder")
    if raw is None:
        raise ValueError("structural/mixed world requires a structural sensor")
    if not isinstance(raw, Mapping) or set(raw) != _STRUCTURAL_KEYS:
        raise ValueError("structural blind-spot keys are invalid")
    sectors = raw["blind_sectors"]
    if not isinstance(sectors, list):
        raise ValueError("structural blind-spot sectors must be a list")
    return StructuralBlindSpot(
        forward_fov_deg=raw["forward_fov_deg"],
        range_m=raw["range_m"],
        blind_sectors=tuple(dict(sector) for sector in sectors),
    )


def _validate_context_world_join(
    record: EventTargetMotionRecord,
    world: OracleWorld,
    oracle_context: OracleContext,
    *,
    target_present: bool = True,
) -> None:
    context_ids = set(oracle_context.dynamic_object_future)
    expected_ids = set(context_ids)
    if target_present:
        expected_ids.add(record.target_dynamic_object_id)
    if set(world.dynamic_object_trajectories) != expected_ids:
        raise ValueError("OracleWorld/context dynamic object ids mismatch")
    if set(world.dynamic_object_specs) != expected_ids:
        raise ValueError("OracleWorld/context dynamic object spec ids mismatch")
    for object_id in sorted(context_ids):
        if not np.array_equal(
            world.dynamic_object_trajectories[object_id],
            oracle_context.dynamic_object_future[object_id],
        ):
            raise ValueError(
                f"OracleWorld context future mismatch for {object_id!r}"
            )
        if (
            world.dynamic_object_specs[object_id]
            != oracle_context.dynamic_object_specs[object_id]
        ):
            raise ValueError(
                f"OracleWorld context spec mismatch for {object_id!r}"
            )


def _validate_paired_target(
    record: EventTargetMotionRecord,
    mother_world: OracleWorld,
    variant: PairedVariant,
) -> TransplantedDynamicObject | None:
    world = variant.world
    metadata = world.metadata
    if not isinstance(metadata, Mapping):
        raise ValueError("paired world metadata must be a mapping")
    if (
        not isinstance(variant.variant_kind, str)
        or variant.variant_kind not in VARIANT_ORDER
    ):
        raise ValueError("paired variant_kind must be one of six frozen kinds")
    if metadata.get("paired_variant_kind") != variant.variant_kind:
        raise ValueError("paired variant kind metadata mismatch")
    if not isinstance(metadata.get("pair_group_id"), str) or not metadata.get(
        "pair_group_id"
    ):
        raise ValueError("paired world requires pair_group_id")

    target = variant.target
    target_present = target is not None
    if metadata.get("target_present") is not target_present:
        raise ValueError("paired target_present metadata mismatch")
    if target is None:
        if variant.variant_kind != "empty_blind_spot":
            raise ValueError("only empty_blind_spot may omit the target")
        _validate_paired_visibility(variant, target_present=False)
        for key in (
            "paired_target_history_array_digest",
            "paired_target_future_array_digest",
            "paired_target_current_pose",
            "target_provenance",
        ):
            if key not in metadata or metadata[key] is not None:
                raise ValueError(f"empty paired {key} must be None")
        if metadata.get("paired_target_motion_digest") != "target-empty":
            raise ValueError(
                "empty paired paired_target_motion_digest must be target-empty"
            )
        _validate_paired_lineage(
            record,
            mother_world,
            variant,
            target_motion_digest="target-empty",
            target_current_pose=None,
            target_provenance=None,
        )
        return None
    if variant.variant_kind == "empty_blind_spot":
        raise ValueError("empty_blind_spot must omit the target")
    if not isinstance(target, TransplantedDynamicObject):
        raise TypeError("paired target must be a TransplantedDynamicObject")
    if (
        target.target_dynamic_object_id != record.target_dynamic_object_id
        or target.source_object_id != record.source_object_id
        or target.snippet_id != record.source_snippet_id
        or target.object_type != record.object_type
        or target.footprint_spec != record.footprint_spec
        or target.footprint_spec_digest != record.footprint_spec_digest
    ):
        raise ValueError("paired target identity differs from mother record")
    if target.provenance.get("target_type_policy_digest") != (
        record.target_type_policy_digest
    ):
        raise ValueError("paired target policy digest differs from mother record")
    for name, array, shape in (
        ("history_poses", target.history_poses, (8, 3)),
        ("current_pose", target.current_pose, (3,)),
        ("future_poses", target.future_poses, (15, 3)),
    ):
        if (
            not isinstance(array, np.ndarray)
            or array.shape != shape
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"paired target {name} contract is invalid")
    if not np.array_equal(target.current_pose, target.history_poses[-1]):
        raise ValueError("paired target current/history seam mismatch")
    history_digest = compute_motion_array_digest(
        target.history_poses, field_name="target_history_poses"
    )
    if metadata.get("paired_target_history_array_digest") != history_digest:
        raise ValueError("paired target history array digest mismatch")
    future_digest = compute_motion_array_digest(
        target.future_poses, field_name="target_future_poses"
    )
    if metadata.get("paired_target_future_array_digest") != future_digest:
        raise ValueError("paired target future array digest mismatch")
    if metadata.get("paired_target_motion_digest") != (
        _paired_target_motion_digest(target)
    ):
        raise ValueError("paired target motion digest mismatch")
    _validate_paired_lineage(
        record,
        mother_world,
        variant,
        target_motion_digest=_paired_target_motion_digest(target),
        target_current_pose=[float(value) for value in target.current_pose],
        target_provenance=dict(target.provenance),
    )
    if not np.array_equal(
        world.dynamic_object_trajectories.get(target.target_dynamic_object_id),
        target.future_poses,
    ):
        raise ValueError("paired world target future mismatch")
    if world.dynamic_object_specs.get(target.target_dynamic_object_id) != (
        target.footprint_spec
    ):
        raise ValueError("paired world target spec mismatch")
    _validate_paired_visibility(variant, target_present=True)
    return target


def _validate_paired_visibility(
    variant: PairedVariant,
    *,
    target_present: bool,
) -> None:
    metadata = variant.world.metadata
    field_shapes = {
        "target_visibility_history": 8,
        "visibility_sequence": 16,
    }
    if not target_present:
        for field_name in field_shapes:
            if getattr(variant, field_name) is not None:
                raise ValueError(
                    f"empty paired {field_name} must be None"
                )
            if field_name not in metadata or metadata[field_name] is not None:
                raise ValueError(
                    f"empty paired {field_name} metadata must be None"
                )
        return

    validated: dict[str, np.ndarray] = {}
    for field_name, length in field_shapes.items():
        value = getattr(variant, field_name)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (length,)
            or value.dtype != np.bool_
        ):
            raise ValueError(
                f"paired {field_name} must be bool[{length}]"
            )
        raw_metadata = metadata.get(field_name)
        if (
            not isinstance(raw_metadata, list)
            or len(raw_metadata) != length
            or any(type(item) is not bool for item in raw_metadata)
        ):
            raise ValueError(
                f"paired {field_name} boolean metadata is invalid"
            )
        if not np.array_equal(value, np.asarray(raw_metadata, dtype=bool)):
            raise ValueError(f"paired {field_name} metadata mismatch")
        validated[field_name] = value

    history = validated["target_visibility_history"]
    sequence = validated["visibility_sequence"]
    if history[-1] != sequence[0]:
        raise ValueError("paired target visibility seam mismatch")
    if bool(sequence[0]):
        raise ValueError("paired target current frame must remain hidden")
    if not has_continuous_emergence(sequence, min_visible_frames=2):
        raise ValueError("paired target requires continuous emergence")
    if not bool(sequence[-1]):
        raise ValueError("paired target final frame must be visible")


def _validate_paired_lineage(
    record: EventTargetMotionRecord,
    mother_world: OracleWorld,
    variant: PairedVariant,
    *,
    target_motion_digest: str,
    target_current_pose: list[float] | None,
    target_provenance: dict[str, object] | None,
) -> None:
    world = variant.world
    metadata = world.metadata
    stale_keys = sorted(_SOP05_JOIN_METADATA_KEYS.intersection(metadata))
    if stale_keys:
        raise ValueError(
            "paired world contains stale SOP05 join metadata: "
            + ", ".join(stale_keys)
        )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "world_id": world.world_id,
        "mother_generated_event_id": record.generated_event_id,
        "mother_world_id": mother_world.world_id,
        "mother_target_motion_record_digest": record.record_digest,
        "mother_source_snippet_id": record.source_snippet_id,
        "mother_source_object_id": record.source_object_id,
        "mother_target_type_policy_digest": record.target_type_policy_digest,
        "mother_target_footprint_spec_digest": record.footprint_spec_digest,
        "target_dynamic_object_id": record.target_dynamic_object_id,
        "paired_target_current_pose": target_current_pose,
        "target_provenance": target_provenance,
    }
    for key, expected_value in expected.items():
        if key not in metadata or metadata[key] != expected_value:
            raise ValueError(f"paired {key} metadata mismatch")

    pair_group_id = metadata.get("pair_group_id")
    paired_config_digest = metadata.get("paired_config_digest")
    paired_seed = metadata.get("paired_seed")
    if (
        not isinstance(paired_config_digest, str)
        or len(paired_config_digest) != 32
        or any(char not in "0123456789abcdef" for char in paired_config_digest)
    ):
        raise ValueError("paired_config_digest metadata is invalid")
    expected_pair_group_id = compute_pair_group_id(
        generated_event_id=record.generated_event_id,
        base_state_id=mother_world.base_state_id,
        trajectory_id=record.trajectory_id,
        occluders=mother_world.occluders,
        blind_spot_config=mother_world.blind_spot_config,
        source_snippet_id=record.source_snippet_id,
        target_dynamic_object_id=record.target_dynamic_object_id,
        paired_config_digest=paired_config_digest,
    )
    if pair_group_id != expected_pair_group_id:
        raise ValueError(
            "paired pair_group_id/paired_config_digest does not match "
            "trusted mother lineage"
        )
    if type(paired_seed) is not int or paired_seed != world.random_seed:
        raise ValueError("paired_seed metadata mismatch world random_seed")
    expected_world_id = "world-" + stable_digest(
        pair_group_id,
        variant.variant_kind,
        target_motion_digest,
        paired_seed,
        paired_config_digest,
        size=12,
    )
    if world.world_id != expected_world_id:
        raise ValueError(
            "paired world_id lineage mismatch (paired_config_digest)"
        )


def _build_background_scene(
    base_state: BaseState,
    oracle_context: OracleContext,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    histories: dict[str, np.ndarray] = {}
    specs: dict[str, dict[str, object]] = {}
    base_ids = set(base_state.dynamic_object_ids)
    for object_id in sorted(base_ids):
        histories[object_id] = np.array(
            base_state.visible_dynamic_object_history[object_id],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        specs[object_id] = deepcopy(
            base_state.visible_dynamic_object_specs[object_id]
        )

    for object_id in sorted(oracle_context.dynamic_object_history):
        history = oracle_context.dynamic_object_history[object_id]
        spec = oracle_context.dynamic_object_specs[object_id]
        if object_id in base_ids:
            if not np.array_equal(histories[object_id], history):
                raise ValueError(
                    "overlapping BaseState/OracleContext history mismatch"
                )
            if specs[object_id] != spec:
                raise ValueError(
                    "overlapping BaseState/OracleContext spec mismatch"
                )
            continue
        histories[object_id] = np.array(
            history, dtype=np.float32, order="C", copy=True
        )
        specs[object_id] = deepcopy(spec)
    return histories, specs


def _validate_paired_skeleton(
    mother_world: OracleWorld,
    paired_world: OracleWorld,
) -> StructuralBlindSpot | None:
    _sensor_from_world(mother_world)
    paired_sensor = _sensor_from_world(paired_world)
    if not np.array_equal(
        paired_world.static_occupancy, mother_world.static_occupancy
    ):
        raise ValueError("paired static occupancy differs from mother")
    if paired_world.occluders != mother_world.occluders:
        raise ValueError("paired occluders differ from mother")
    if paired_world.blind_spot_config != mother_world.blind_spot_config:
        raise ValueError("paired blind-spot config differs from mother")
    return paired_sensor


def render_sop06_paired_variant(
    *,
    mother_record: EventTargetMotionRecord,
    mother_world: OracleWorld,
    variant: PairedVariant,
    base_state: BaseState,
    oracle_context: OracleContext,
    config: Mapping[str, Any],
    expected_paired_config_digest: str,
) -> RenderedObservation:
    """Render one actual paired variant after validating its SOP05 mother.

    ``expected_paired_config_digest`` must come from the independently loaded
    paired configuration, never from ``variant.world.metadata``.
    """

    if not isinstance(variant, PairedVariant):
        raise TypeError("variant must be a PairedVariant")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if (
        not isinstance(expected_paired_config_digest, str)
        or len(expected_paired_config_digest) != 32
        or any(
            char not in "0123456789abcdef"
            for char in expected_paired_config_digest
        )
    ):
        raise ValueError(
            "expected_paired_config_digest must be a 32-character "
            "lowercase hex digest"
        )
    if (
        not isinstance(variant.world.metadata, Mapping)
        or variant.world.metadata.get("paired_config_digest")
        != expected_paired_config_digest
    ):
        raise ValueError(
            "paired_config_digest does not match expected trusted digest"
        )
    grid = build_grid_spec(dict(config))
    validate_event_target_motion_world_join(
        mother_record, mother_world, grid
    )
    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(oracle_context, OracleContext):
        raise TypeError("oracle_context must be an OracleContext")
    validate_base_state(base_state, grid)
    validate_oracle_context(oracle_context, grid)
    if base_state.state_id != mother_record.base_state_id:
        raise ValueError("base_state id does not match mother record")
    if oracle_context.base_state_id != mother_record.base_state_id:
        raise ValueError("oracle_context id does not match mother record")
    if variant.world.base_state_id != mother_record.base_state_id:
        raise ValueError("paired world base_state_id mismatch")
    sensor_config = _validate_paired_skeleton(
        mother_world, variant.world
    )
    _validate_joint_six_pack_versions(mother_world, variant.world)
    target = _validate_paired_target(mother_record, mother_world, variant)
    _validate_context_world_join(
        mother_record,
        variant.world,
        oracle_context,
        target_present=target is not None,
    )
    histories, specs = _build_background_scene(base_state, oracle_context)
    if target is not None:
        if target.target_dynamic_object_id in histories:
            raise ValueError("paired target id collides with background history")
        histories[target.target_dynamic_object_id] = np.array(
            target.history_poses, dtype=np.float32, order="C", copy=True
        )
        specs[target.target_dynamic_object_id] = dict(target.footprint_spec)

    return render_observation(
        base_state,
        scene_dynamic_history=histories,
        scene_dynamic_specs=specs,
        static_occupancy=variant.world.static_occupancy,
        sensor_config=sensor_config,
        config=config,
    )


def render_sop06_variant(
    *,
    record: EventTargetMotionRecord,
    world: OracleWorld,
    base_state: BaseState,
    oracle_context: OracleContext,
    config: Mapping[str, Any],
) -> RenderedObservation:
    """Validate and render one target-present SOP05 mother event.

    Oracle futures are inspected only to validate the outer record/world/context
    join.  The core renderer receives a newly built history-only scene.  The
    target is always present here; paired and empty variants must use
    :func:`render_sop06_paired_variant` so their paired metadata is validated.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    grid = build_grid_spec(dict(config))
    validate_event_target_motion_world_join(record, world, grid)
    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(oracle_context, OracleContext):
        raise TypeError("oracle_context must be an OracleContext")
    validate_base_state(base_state, grid)
    validate_oracle_context(oracle_context, grid)
    if base_state.state_id != record.base_state_id:
        raise ValueError("base_state id does not match target-motion record")
    if oracle_context.base_state_id != record.base_state_id:
        raise ValueError("oracle_context id does not match target-motion record")
    _validate_context_world_join(record, world, oracle_context)
    sensor_config = _sensor_from_world(world)

    histories, specs = _build_background_scene(base_state, oracle_context)
    if record.target_dynamic_object_id in histories:
        raise ValueError("target id collides with background history")
    histories[record.target_dynamic_object_id] = np.array(
        record.history_poses, dtype=np.float32, order="C", copy=True
    )
    specs[record.target_dynamic_object_id] = deepcopy(record.footprint_spec)

    return render_observation(
        base_state,
        scene_dynamic_history=histories,
        scene_dynamic_specs=specs,
        static_occupancy=world.static_occupancy,
        sensor_config=sensor_config,
        config=config,
    )


def _validate_formal_mother_world(world: OracleWorld) -> None:
    if not isinstance(world, OracleWorld):
        raise TypeError("mother_world must be an OracleWorld")
    joint_version = world.metadata.get(
        "joint_pair_generator_algorithm_version"
    )
    if joint_version is not None:
        if joint_version == JOINT_ENVIRONMENT_PAIR_VERSION:
            raise ValueError(
                f"mother uses retired {JOINT_ENVIRONMENT_PAIR_VERSION}"
            )
        raise ValueError("mother contains unsupported joint-pair identity")
    if world.metadata.get("generator_algorithm_version") != (
        _FORMAL_SOP05_GENERATOR_VERSION
    ):
        raise ValueError(
            "formal SOP06 mother generator_algorithm_version must equal "
            f"{_FORMAL_SOP05_GENERATOR_VERSION}"
        )


def render_sop06_mother_event(
    *,
    record: EventTargetMotionRecord,
    world: OracleWorld,
    base_state: BaseState,
    oracle_context: OracleContext,
    config: Mapping[str, Any],
) -> RenderedObservation:
    """Render one formal v5 collision mother through the history-only path."""

    _validate_formal_mother_world(world)
    return render_sop06_variant(
        record=record,
        world=world,
        base_state=base_state,
        oracle_context=oracle_context,
        config=config,
    )


def _validate_formal_pair_group(
    group: PairedEventGroup,
    *,
    mother_world: OracleWorld,
    expected_paired_config_digest: str,
) -> None:
    if not isinstance(group, PairedEventGroup):
        raise TypeError("group must be a PairedEventGroup")
    _validate_formal_mother_world(mother_world)
    if group.paired_config_digest != expected_paired_config_digest:
        raise ValueError(
            "group paired_config_digest does not match expected trusted digest"
        )
    if not isinstance(group.pair_group_id, str) or not group.pair_group_id:
        raise ValueError("formal SOP06 pair_group_id must be non-empty")
    by_kind = group.by_kind
    if len(by_kind) != len(group.variants):
        raise ValueError("formal SOP06 group contains duplicate variant kinds")
    expected_coverage = tuple(kind in by_kind for kind in VARIANT_ORDER)
    if group.coverage_mask != expected_coverage:
        raise ValueError("formal SOP06 group coverage mask mismatch")
    if "collision" not in by_kind:
        raise ValueError("formal SOP06 group requires collision mother position")
    absent = set(VARIANT_ORDER) - set(by_kind)
    if set(group.missing_variant_reasons) != absent:
        raise ValueError("formal SOP06 group missing reasons mismatch coverage")
    if any(
        not isinstance(reason, str) or not reason
        for reason in group.missing_variant_reasons.values()
    ):
        raise ValueError(
            "formal SOP06 missing reasons must be non-empty strings"
        )
    if group.is_complete != all(expected_coverage):
        raise ValueError("formal SOP06 group completeness mismatch coverage")
    if group.eligible_for_strict_evaluation != group.is_complete:
        raise ValueError(
            "formal SOP06 strict eligibility must equal completeness"
        )
    expected_kinds = tuple(kind for kind in VARIANT_ORDER if kind in by_kind)
    observed_kinds = tuple(variant.variant_kind for variant in group.variants)
    if observed_kinds != expected_kinds:
        raise ValueError("formal SOP06 variants must follow frozen coverage order")
    expected_coverage_by_kind = {
        kind: expected_coverage[index]
        for index, kind in enumerate(VARIANT_ORDER)
    }
    for variant in group.variants:
        metadata = variant.world.metadata
        if metadata.get("pair_group_id") != group.pair_group_id:
            raise ValueError("formal SOP06 variant pair_group_id mismatch")
        if metadata.get("paired_config_digest") != group.paired_config_digest:
            raise ValueError("formal SOP06 variant paired_config_digest mismatch")
        if metadata.get("paired_generator_algorithm_version") != (
            PAIRED_GENERATOR_ALGORITHM_VERSION
        ):
            raise ValueError(
                "formal SOP06 paired_generator_algorithm_version must equal "
                f"{PAIRED_GENERATOR_ALGORITHM_VERSION}"
            )
        if metadata.get("pair_group_contract_version") != (
            PAIRED_GROUP_CONTRACT_VERSION
        ):
            raise ValueError(
                "formal SOP06 pair_group_contract_version must equal "
                f"{PAIRED_GROUP_CONTRACT_VERSION}"
            )
        if metadata.get("paired_coverage_mask") != list(expected_coverage):
            raise ValueError("formal SOP06 paired_coverage_mask mismatch")
        if metadata.get("paired_coverage") != expected_coverage_by_kind:
            raise ValueError("formal SOP06 paired_coverage mismatch")
        if metadata.get("paired_missing_variant_reasons") != dict(
            group.missing_variant_reasons
        ):
            raise ValueError(
                "formal SOP06 paired_missing_variant_reasons mismatch"
            )
        if metadata.get("paired_group_complete") is not group.is_complete:
            raise ValueError("formal SOP06 paired_group_complete mismatch")
        if metadata.get("eligible_for_strict_paired_evaluation") is not (
            group.eligible_for_strict_evaluation
        ):
            raise ValueError(
                "formal SOP06 strict-evaluation metadata mismatch"
            )
        joint_version = metadata.get("joint_pair_generator_algorithm_version")
        if joint_version is not None:
            if joint_version == JOINT_ENVIRONMENT_PAIR_VERSION:
                raise ValueError(
                    f"paired variant uses retired {JOINT_ENVIRONMENT_PAIR_VERSION}"
                )
            raise ValueError("paired variant contains unsupported joint-pair identity")


def _render_formal_pair_group(
    *,
    group: PairedEventGroup,
    mother_record: EventTargetMotionRecord,
    mother_world: OracleWorld,
    base_state: BaseState,
    oracle_context: OracleContext,
    config: Mapping[str, Any],
    expected_paired_config_digest: str,
    audit_certified: bool,
) -> RenderedSop06Group:
    _validate_formal_pair_group(
        group,
        mother_world=mother_world,
        expected_paired_config_digest=expected_paired_config_digest,
    )
    observations = tuple(
        render_sop06_paired_variant(
            mother_record=mother_record,
            mother_world=mother_world,
            variant=variant,
            base_state=base_state,
            oracle_context=oracle_context,
            config=config,
            expected_paired_config_digest=expected_paired_config_digest,
        )
        for variant in group.variants
    )
    return RenderedSop06Group(
        pair_group_id=group.pair_group_id,
        variant_kinds=tuple(variant.variant_kind for variant in group.variants),
        observations=observations,
        coverage_mask=group.coverage_mask,
        is_complete=group.is_complete,
        audit_certified=audit_certified,
    )


def render_sop06_partial_pair_group(
    *,
    group: PairedEventGroup,
    mother_record: EventTargetMotionRecord,
    mother_world: OracleWorld,
    base_state: BaseState,
    oracle_context: OracleContext,
    config: Mapping[str, Any],
    expected_paired_config_digest: str,
) -> RenderedSop06Group:
    """Render training variants without ever certifying sixpack completeness."""

    return _render_formal_pair_group(
        group=group,
        mother_record=mother_record,
        mother_world=mother_world,
        base_state=base_state,
        oracle_context=oracle_context,
        config=config,
        expected_paired_config_digest=expected_paired_config_digest,
        audit_certified=False,
    )


def render_sop06_complete_audit_group(
    *,
    group: PairedEventGroup,
    mother_record: EventTargetMotionRecord,
    mother_world: OracleWorld,
    base_state: BaseState,
    oracle_context: OracleContext,
    config: Mapping[str, Any],
    expected_paired_config_digest: str,
) -> RenderedSop06Group:
    """Render a conditional complete sixpack for audit/ablation only."""

    if not isinstance(group, PairedEventGroup):
        raise TypeError("group must be a PairedEventGroup")
    if (
        not group.is_complete
        or group.coverage_mask != (True,) * len(VARIANT_ORDER)
        or not group.eligible_for_strict_evaluation
    ):
        raise ValueError(
            "complete audit requires a complete six-position paired group"
        )
    return _render_formal_pair_group(
        group=group,
        mother_record=mother_record,
        mother_world=mother_world,
        base_state=base_state,
        oracle_context=oracle_context,
        config=config,
        expected_paired_config_digest=expected_paired_config_digest,
        audit_certified=True,
    )
