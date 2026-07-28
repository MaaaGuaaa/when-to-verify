"""Shared SOP11–13 geometry/value pipeline for toy and audited train inputs."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any

import numpy as np

from src.contracts import (
    ARRAY_DTYPE,
    SCHEMA_VERSION,
    STATE_CHANNELS,
    BaseState,
    GridSpec,
    LocalTrajectory,
    OracleWorld,
    VerificationSample,
    build_grid_spec,
)
from src.datasets.verification_dataset import (
    VerificationGroupInput,
    build_verification_samples,
)
from src.datasets.verification_sources import VerificationSourceEvent
from src.geometry import (
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint,
)
from src.generation.counterfactual_verify import (
    expected_verification_fov_trace_mask,
    simulate_counterfactual_observation_trace,
)
from src.generation.event_contracts import footprint_from_spec
from src.generation.observation_renderer import render_observation
from src.generation.sop06_pipeline import (
    Sop06SinglePublication,
    render_sop06_mother_event,
)
from src.generation.structural_blindspot import StructuralBlindSpot
from src.generation.verification_gt import (
    SampledVerificationValueResult,
    TypedFootprintRiskLoss,
    VerificationGTConfig,
    evaluate_sampled_realization_value,
)
from src.generation.verification_response import (
    resolve_verification_response,
)
from src.generation.verification_toy import build_verification_toy_world
from src.planning.query_maps import build_local_trajectory
from src.planning.replanning import (
    POST_PLAN_STATUS_SAFE_STOP_NO_FEASIBLE_PLAN,
    generate_replanned_candidates,
)
from src.planning.trajectory_sampler import sample_candidate_rollouts
from src.planning.verification_actions import (
    VerificationActionLibrary,
    check_action_trace_feasibility,
    sample_state_aware_action_trace,
)
from src.planning.verification_responses import (
    VerificationPolicyBranch,
    compose_time_aligned_policy_trajectory,
)
from src.utils.config import validate_config
from src.utils.seeding import stable_digest


VERIFICATION_PIPELINE_VERSION = "verification_pipeline_v6"


class VerificationSourceIneligibleError(ValueError):
    """An audited source cannot form a complete, physically valid action group."""

    def __init__(self, reason: str, detail: str) -> None:
        if not isinstance(reason, str) or not reason:
            raise ValueError("source-ineligibility reason must be non-empty")
        if not isinstance(detail, str) or not detail:
            raise ValueError("source-ineligibility detail must be non-empty")
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class VerificationPipelineInput:
    split: str
    base_state_id: str
    source_namespace: str
    grid: GridSpec
    nominal_trajectory: LocalTrajectory
    current_world: OracleWorld
    current_dynamic_poses: Mapping[str, np.ndarray]
    target_object_id: str | None
    robot_pose: np.ndarray
    robot_state: np.ndarray
    current_visible_mask: np.ndarray
    current_age_map: np.ndarray
    bev_history: np.ndarray
    state_channels: np.ndarray
    sensor_fov_rad: float
    sensor_range_m: float
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class VerificationGroupResult:
    version: str
    samples: tuple[VerificationSample, ...]
    values: Mapping[str, SampledVerificationValueResult]
    sampled_child_world_id: str
    infeasible_action_ids: tuple[str, ...]


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _robot_footprint(config: Mapping[str, Any]):
    robot = config["robot"]
    return inflate_footprint(
        RectangleFootprint(float(robot["length_m"]), float(robot["width_m"])),
        float(robot["inflation_m"]),
    )


def _current_pose_map(
    source: VerificationSourceEvent,
) -> dict[str, np.ndarray]:
    record = source.event.target_motion_record
    world = source.event.world
    result: dict[str, np.ndarray] = {}
    for object_id in sorted(world.dynamic_object_trajectories):
        if object_id == record.target_dynamic_object_id:
            value = record.current_pose
        elif object_id in source.oracle_context.dynamic_object_history:
            value = source.oracle_context.dynamic_object_history[object_id][-1]
        elif object_id in source.base_state.visible_dynamic_object_history:
            value = source.base_state.visible_dynamic_object_history[object_id][-1]
        else:
            raise ValueError(f"no current pose source for dynamic object {object_id!r}")
        result[object_id] = np.array(
            value, dtype=ARRAY_DTYPE, order="C", copy=True
        )
    return result


def _verification_sensor_geometry(
    world: OracleWorld,
    grid: GridSpec,
    *,
    action_library: VerificationActionLibrary,
) -> tuple[float, float]:
    if not isinstance(action_library, VerificationActionLibrary):
        raise TypeError("action_library must be a VerificationActionLibrary")
    config = world.blind_spot_config
    if not isinstance(config, Mapping):
        raise ValueError("world blind_spot_config must be a mapping")
    structural = config.get("structural")
    if structural is not None:
        if not isinstance(structural, Mapping):
            raise ValueError("structural blind-spot config must be a mapping")
        sensor_range = float(structural["range_m"])
    else:
        sensor_range = float(
            np.hypot(
                grid.height * grid.resolution_m,
                grid.width * grid.resolution_m,
            )
        )
    fov = float(action_library.sensor_fov_rad)
    if not np.isfinite([fov, sensor_range]).all() or fov <= 0.0 or sensor_range <= 0.0:
        raise ValueError("verification sensor geometry must be finite and positive")
    return fov, sensor_range


def build_real_verification_input(
    source: VerificationSourceEvent,
    *,
    base_config: Mapping[str, Any],
    action_library: VerificationActionLibrary,
    sop05_batch_digest: str,
    sop07_collection_digest: str,
    scientific_status: str,
    cross_split_status: str,
) -> VerificationPipelineInput:
    """Render deployment history and retain oracle state only in label fields."""

    if not isinstance(source, VerificationSourceEvent):
        raise TypeError("source must be a VerificationSourceEvent")
    if not isinstance(action_library, VerificationActionLibrary):
        raise TypeError("action_library must be a VerificationActionLibrary")
    config = dict(base_config)
    validate_config(config)
    grid = build_grid_spec(config)
    sensor_fov, sensor_range = _verification_sensor_geometry(
        source.event.world,
        grid,
        action_library=action_library,
    )
    verification_sensor = StructuralBlindSpot(
        forward_fov_deg=float(np.rad2deg(sensor_fov)),
        range_m=sensor_range,
    )
    rendered = render_sop06_mother_event(
        record=source.event.target_motion_record,
        world=source.event.world,
        base_state=source.base_state,
        oracle_context=source.oracle_context,
        config=config,
        sensor_config_override=verification_sensor,
    )
    visible = (
        rendered.state_channels[STATE_CHANNELS.index("current_visible_free")] != 0.0
    ) | (
        rendered.state_channels[
            STATE_CHANNELS.index("current_visible_occupied")
        ]
        != 0.0
    )
    age = rendered.state_channels[STATE_CHANNELS.index("occlusion_age_map")]
    event_id = source.event.generated_event_id
    split = source.base_state.split
    if split not in {"train", "calibration", "val", "test"}:
        raise ValueError("real verification source split is unsupported")
    expected_status = f"{split}_smoke_only"
    if scientific_status != expected_status:
        raise ValueError("real verification scientific status differs from split")
    if cross_split_status not in {"NOT_PROVEN", "PROVEN"}:
        raise ValueError("real verification cross-split status is invalid")
    source_mode = "sop05-train" if split == "train" else "sop05-heldout"
    namespace = f"sop05/{split}/{event_id}"
    provenance = {
        "source_mode": source_mode,
        "scientific_status": scientific_status,
        "cross_split_status": cross_split_status,
        "source_event_id": event_id,
        "source_snippet_id": source.source_snippet.snippet_id,
        "source_trajectory_id": source.nominal_trajectory.trajectory_id,
        "sop05_batch_digest": sop05_batch_digest,
        "sop07_collection_digest": sop07_collection_digest,
        "source_artifact_digest": stable_digest(
            sop05_batch_digest,
            source.shard.publication_semantic_digest,
            event_id,
            size=16,
        ),
        "verification_sensor_fov_deg": float(np.rad2deg(sensor_fov)),
        "verification_sensor_range_m": sensor_range,
    }
    return VerificationPipelineInput(
        split=split,
        base_state_id=source.base_state.state_id,
        source_namespace=namespace,
        grid=grid,
        nominal_trajectory=source.nominal_trajectory,
        current_world=source.event.world,
        current_dynamic_poses=_current_pose_map(source),
        target_object_id=source.event.target.target_dynamic_object_id,
        robot_pose=np.array(
            source.base_state.robot_history[-1],
            dtype=ARRAY_DTYPE,
            order="C",
            copy=True,
        ),
        robot_state=np.array(
            source.base_state.robot_state,
            dtype=ARRAY_DTYPE,
            order="C",
            copy=True,
        ),
        current_visible_mask=np.asarray(visible, dtype=bool, order="C"),
        current_age_map=np.array(age, dtype=ARRAY_DTYPE, order="C", copy=True),
        bev_history=np.array(
            rendered.bev_history, dtype=ARRAY_DTYPE, order="C", copy=True
        ),
        state_channels=np.array(
            rendered.state_channels, dtype=ARRAY_DTYPE, order="C", copy=True
        ),
        sensor_fov_rad=sensor_fov,
        sensor_range_m=sensor_range,
        provenance=provenance,
    )


def build_finalized_verification_input(
    publication: Sop06SinglePublication,
    *,
    base_config: Mapping[str, Any],
    action_library: VerificationActionLibrary,
    target_current_pose: np.ndarray | None,
    source_publication_semantic_digest: str,
    final_release_identity: str,
) -> VerificationPipelineInput:
    """Adapt one finalized SOP5 sampled child without creating new worlds."""

    if not isinstance(publication, Sop06SinglePublication):
        raise TypeError("publication must be a Sop06SinglePublication")
    if not isinstance(action_library, VerificationActionLibrary):
        raise TypeError("action_library must be a VerificationActionLibrary")
    for name, value in (
        (
            "source_publication_semantic_digest",
            source_publication_semantic_digest,
        ),
        ("final_release_identity", final_release_identity),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be non-empty")
    config = dict(base_config)
    validate_config(config)
    grid = build_grid_spec(config)
    world = publication.oracle_world
    renderer = publication.renderer_input
    sensor_fov, sensor_range = _verification_sensor_geometry(
        world,
        grid,
        action_library=action_library,
    )
    sensor = StructuralBlindSpot(
        forward_fov_deg=float(np.rad2deg(sensor_fov)),
        range_m=sensor_range,
    )
    rendered = render_observation(
        renderer.base_state,
        scene_dynamic_history=renderer.scene_dynamic_history,
        scene_dynamic_specs=renderer.scene_dynamic_specs,
        scene_dynamic_history_observed=(
            renderer.scene_dynamic_history_observed
        ),
        static_occupancy=renderer.observed_static_occupancy,
        sensor_config=sensor,
        config=config,
    )
    visible = (
        rendered.state_channels[
            STATE_CHANNELS.index("current_visible_free")
        ]
        != 0.0
    ) | (
        rendered.state_channels[
            STATE_CHANNELS.index("current_visible_occupied")
        ]
        != 0.0
    )
    age = rendered.state_channels[STATE_CHANNELS.index("occlusion_age_map")]
    if len(publication.hidden_object_ids) != 1:
        raise VerificationSourceIneligibleError(
            "hidden_target_count",
            "finalized sampled child must contain exactly one hidden target",
        )
    target_id = publication.hidden_object_ids[0]
    target_spec = world.dynamic_object_specs[target_id]
    target_object_type = target_spec.get("object_type")
    target_footprint = target_spec.get("footprint")
    if (
        not isinstance(target_object_type, str)
        or not target_object_type
        or not isinstance(target_footprint, Mapping)
        or target_footprint.get("kind") not in {"circle", "rectangle"}
    ):
        raise ValueError("finalized target type/footprint metadata is invalid")
    target_footprint_kind = str(target_footprint["kind"])
    if (
        not isinstance(target_current_pose, np.ndarray)
        or target_current_pose.shape != (3,)
        or target_current_pose.dtype != ARRAY_DTYPE
        or not np.isfinite(target_current_pose).all()
    ):
        raise ValueError("target_current_pose must be finite float32 [3]")
    future_ids = set(world.dynamic_object_trajectories)
    if (
        future_ids != set(world.dynamic_object_specs)
        or target_id not in future_ids
        or not (future_ids - {target_id}).issubset(
            renderer.scene_dynamic_history
        )
    ):
        raise ValueError("oracle future IDs lack aligned history/spec fields")
    current_poses = {
        object_id: np.array(
            (
                target_current_pose
                if object_id == target_id
                else renderer.scene_dynamic_history[object_id][-1]
            ),
            dtype=ARRAY_DTYPE,
            order="C",
            copy=True,
        )
        for object_id in sorted(future_ids)
    }
    return VerificationPipelineInput(
        split=publication.split,
        base_state_id=renderer.base_state.state_id,
        source_namespace=(
            f"sop05-final/{publication.split}/{publication.sample_id}"
        ),
        grid=grid,
        nominal_trajectory=publication.trajectory,
        current_world=world,
        current_dynamic_poses=current_poses,
        target_object_id=target_id,
        robot_pose=np.array(
            renderer.base_state.robot_history[-1],
            dtype=ARRAY_DTYPE,
            order="C",
            copy=True,
        ),
        robot_state=np.array(
            renderer.base_state.robot_state,
            dtype=ARRAY_DTYPE,
            order="C",
            copy=True,
        ),
        current_visible_mask=np.asarray(visible, dtype=np.bool_, order="C"),
        current_age_map=np.array(
            age,
            dtype=ARRAY_DTYPE,
            order="C",
            copy=True,
        ),
        bev_history=np.array(
            rendered.bev_history,
            dtype=ARRAY_DTYPE,
            order="C",
            copy=True,
        ),
        state_channels=np.array(
            rendered.state_channels,
            dtype=ARRAY_DTYPE,
            order="C",
            copy=True,
        ),
        sensor_fov_rad=sensor_fov,
        sensor_range_m=sensor_range,
        provenance={
            "source_mode": "sop05-final",
            "source_sample_id": publication.sample_id,
            "source_mother_id": publication.mother_id,
            "blind_type": publication.regime,
            "target_object_type": target_object_type,
            "target_footprint_kind": target_footprint_kind,
            "source_artifact_digest": stable_digest(
                source_publication_semantic_digest,
                final_release_identity,
                publication.sample_id,
                size=16,
            ),
            "verification_sensor_fov_deg": float(np.rad2deg(sensor_fov)),
            "verification_sensor_range_m": sensor_range,
        },
    )


def build_verification_toy_input(
    base_config: Mapping[str, Any],
    *,
    action_library: VerificationActionLibrary,
    group_index: int,
) -> tuple[VerificationPipelineInput, dict[str, Any]]:
    """Create one distinct toy identity while preserving hand-checkable geometry."""

    index = _positive_integer(group_index + 1, name="group_index_plus_one") - 1
    if not isinstance(action_library, VerificationActionLibrary):
        raise TypeError("action_library must be a VerificationActionLibrary")
    config = deepcopy(dict(base_config))
    config["bev"]["range_m"] = 8.0
    config["bev"]["size"] = 80
    validate_config(config)
    toy = build_verification_toy_world()
    if build_grid_spec(config) != toy.grid:
        raise RuntimeError("toy config and toy grid differ")
    base_id = f"toy-base-{index:04d}"
    sensor = StructuralBlindSpot(
        forward_fov_deg=float(np.rad2deg(action_library.sensor_fov_rad)),
        range_m=4.0,
    )
    robot_history = np.zeros((toy.grid.history_steps, 3), dtype=ARRAY_DTYPE)
    base_state = BaseState(
        state_id=base_id,
        split="train",
        recording_id=f"toy-recording-{index:04d}",
        dynamic_object_ids=(),
        timestamp=float(index),
        robot_history=robot_history,
        robot_state=np.zeros(2, dtype=ARRAY_DTYPE),
        visible_dynamic_object_history={},
        visible_dynamic_object_specs={},
        static_map_local=toy.static_occupancy.copy(),
        metadata={"schema_version": SCHEMA_VERSION, "source_mode": "toy"},
    )
    scene_history = {
        object_id: np.tile(pose, (toy.grid.history_steps, 1)).astype(ARRAY_DTYPE)
        for object_id, pose in toy.dynamic_current_poses.items()
    }
    rendered = render_observation(
        base_state,
        scene_dynamic_history=scene_history,
        scene_dynamic_specs=toy.dynamic_specs,
        static_occupancy=toy.static_occupancy,
        sensor_config=sensor,
        config=config,
    )
    visible = (
        rendered.state_channels[STATE_CHANNELS.index("current_visible_free")] != 0.0
    ) | (
        rendered.state_channels[
            STATE_CHANNELS.index("current_visible_occupied")
        ]
        != 0.0
    )
    age = rendered.state_channels[STATE_CHANNELS.index("occlusion_age_map")]
    primitive = next(
        item
        for item in sample_candidate_rollouts(config, reverse_stress=False)
        if item.trajectory_id == "forward_v01_w02"
    )
    nominal = build_local_trajectory(
        primitive,
        config,
        braking_deceleration_mps2=1.0,
        task_cost=0.05,
    )
    namespace = f"toy/train/group-{index:04d}"
    world = OracleWorld(
        world_id=f"toy-current-{index:04d}",
        base_state_id=base_id,
        static_occupancy=toy.static_occupancy.copy(),
        dynamic_object_trajectories={
            key: value.copy() for key, value in toy.dynamic_future_poses.items()
        },
        dynamic_object_specs=deepcopy(toy.dynamic_specs),
        occluders=(),
        blind_spot_config={
            "kind": "structural",
            "occluder_ids": [],
            "structural": sensor.as_dict(),
        },
        random_seed=index,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "split": "train",
            "source_namespace": namespace,
        },
    )
    case_digest = stable_digest(VERIFICATION_PIPELINE_VERSION, namespace, size=16)
    source = VerificationPipelineInput(
        split="train",
        base_state_id=base_id,
        source_namespace=namespace,
        grid=toy.grid,
        nominal_trajectory=nominal,
        current_world=world,
        current_dynamic_poses={
            key: value.copy() for key, value in toy.dynamic_current_poses.items()
        },
        target_object_id="critical_cart",
        robot_pose=np.zeros(3, dtype=ARRAY_DTYPE),
        robot_state=np.array(
            base_state.robot_state, dtype=ARRAY_DTYPE, order="C", copy=True
        ),
        current_visible_mask=np.asarray(visible, dtype=bool, order="C"),
        current_age_map=np.array(age, dtype=ARRAY_DTYPE, order="C", copy=True),
        bev_history=np.array(
            rendered.bev_history, dtype=ARRAY_DTYPE, order="C", copy=True
        ),
        state_channels=np.array(
            rendered.state_channels, dtype=ARRAY_DTYPE, order="C", copy=True
        ),
        sensor_fov_rad=action_library.sensor_fov_rad,
        sensor_range_m=4.0,
        provenance={
            "source_mode": "toy",
            "scientific_status": "toy_smoke_only",
            "source_artifact_digest": case_digest,
            "toy_case_digest": case_digest,
            "verification_sensor_fov_deg": float(
                np.rad2deg(action_library.sensor_fov_rad)
            ),
            "verification_sensor_range_m": 4.0,
        },
    )
    return source, config


def _hidden_object_ids(source: VerificationPipelineInput) -> tuple[str, ...]:
    hidden: list[str] = []
    for object_id in sorted(source.current_dynamic_poses):
        footprint = footprint_from_spec(source.current_world.dynamic_object_specs[object_id])
        occupied = rasterize_footprint(
            footprint, source.current_dynamic_poses[object_id], source.grid
        )
        if not np.any(occupied & source.current_visible_mask):
            hidden.append(object_id)
    if (
        source.target_object_id is not None
        and source.target_object_id not in hidden
    ):
        raise VerificationSourceIneligibleError(
            "target_visible_under_verification_sensor",
            "sampled target is visible in the configured verification observation",
        )
    return tuple(hidden)


def _policy_trajectory_id(
    *,
    action_id: str,
    branch: VerificationPolicyBranch,
    suffix_trajectory_id: str | None,
) -> str:
    if suffix_trajectory_id is None:
        if branch.branch_kind != "emergency_brake":
            raise ValueError("only an emergency branch may omit a replan suffix")
        return (
            f"policy::brake@{branch.trigger_time_s:.6f}::"
            f"{action_id}"
        )
    return (
        f"policy::{branch.branch_kind}@{branch.end_time_s:.6f}::"
        f"{suffix_trajectory_id}"
    )


def generate_verification_group(
    source: VerificationPipelineInput,
    *,
    base_config: Mapping[str, Any],
    action_library: VerificationActionLibrary,
    gt_config: VerificationGTConfig,
    max_replan_candidates: int,
) -> VerificationGroupResult:
    """Evaluate all six actions on the one child realization sampled by SOP5."""

    if not isinstance(source, VerificationPipelineInput):
        raise TypeError("source must be a VerificationPipelineInput")
    config = dict(base_config)
    validate_config(config)
    if build_grid_spec(config) != source.grid:
        raise ValueError("base_config grid differs from verification source")
    if not isinstance(action_library, VerificationActionLibrary):
        raise TypeError("action_library must be a VerificationActionLibrary")
    if not np.isclose(
        source.sensor_fov_rad,
        action_library.sensor_fov_rad,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("verification source FOV differs from the action library")
    if not isinstance(gt_config, VerificationGTConfig):
        raise TypeError("gt_config must be a VerificationGTConfig")
    candidate_count = _positive_integer(
        max_replan_candidates, name="max_replan_candidates"
    )

    hidden_ids = _hidden_object_ids(source)
    robot_footprint = _robot_footprint(config)
    dynamic_footprints = {
        object_id: footprint_from_spec(spec)
        for object_id, spec in source.current_world.dynamic_object_specs.items()
    }
    dynamic_poses = {
        object_id: np.vstack(
            (
                source.current_dynamic_poses[object_id][None, :],
                source.current_world.dynamic_object_trajectories[object_id],
            )
        ).astype(ARRAY_DTYPE)
        for object_id in source.current_world.dynamic_object_trajectories
    }
    visible_dynamic_ids = set(dynamic_poses) - set(hidden_ids)
    risk_config = config["risk_gt"]
    risk_loss = TypedFootprintRiskLoss(
        hidden_object_ids=hidden_ids,
        robot_footprint=robot_footprint,
        grid=source.grid,
        future_dt_s=float(config["bev"]["future_dt_s"]),
        sigma_distance_m=float(risk_config["sigma_distance_m"]),
        sigma_time_s=float(risk_config["sigma_time_s"]),
        near_miss_distance_m=float(risk_config["near_miss_distance_m"]),
    )
    fov_masks: dict[str, np.ndarray] = {}
    values: dict[str, SampledVerificationValueResult] = {}
    infeasible: list[str] = []
    future_dt_s = float(config["bev"]["future_dt_s"])
    future_horizon_s = source.grid.future_steps * future_dt_s
    for action in action_library.actions:
        action_trace = sample_state_aware_action_trace(
            source.robot_pose,
            action,
            robot_state=source.robot_state,
            braking_deceleration_mps2=gt_config.braking_deceleration_mps2,
            angular_deceleration_radps2=(
                gt_config.angular_deceleration_radps2
            ),
        )
        feasibility = check_action_trace_feasibility(
            action_trace,
            robot_footprint=robot_footprint,
            static_occupancy=source.current_world.static_occupancy,
            grid=source.grid,
            dynamic_object_poses={
                object_id: dynamic_poses[object_id]
                for object_id in visible_dynamic_ids
            },
            dynamic_object_footprints={
                object_id: dynamic_footprints[object_id]
                for object_id in visible_dynamic_ids
            },
            dynamic_dt_s=future_dt_s,
        )
        if not feasibility.feasible:
            infeasible.append(action.action_id)
            continue
        fov_masks[action.action_id] = expected_verification_fov_trace_mask(
            source.current_world.static_occupancy,
            source.grid,
            action_trace=action_trace,
            fov_rad=source.sensor_fov_rad,
            max_range_m=source.sensor_range_m,
        )
        planned_observation_trace = simulate_counterfactual_observation_trace(
            action_trace=action_trace,
            static_occupancy=source.current_world.static_occupancy,
            dynamic_current_poses=source.current_dynamic_poses,
            dynamic_future_poses=(
                source.current_world.dynamic_object_trajectories
            ),
            dynamic_specs=source.current_world.dynamic_object_specs,
            current_visible_mask=source.current_visible_mask,
            current_age_map=source.current_age_map,
            grid=source.grid,
            future_dt_s=future_dt_s,
            age_max_s=float(config["age_map"]["a_max_s"]),
            fov_rad=source.sensor_fov_rad,
            max_range_m=source.sensor_range_m,
        )
        resolution = resolve_verification_response(
            action_trace=action_trace,
            observation_trace=planned_observation_trace,
            robot_footprint=robot_footprint,
            static_occupancy=source.current_world.static_occupancy,
            dynamic_current_poses=source.current_dynamic_poses,
            dynamic_future_poses=(
                source.current_world.dynamic_object_trajectories
            ),
            dynamic_specs=source.current_world.dynamic_object_specs,
            current_visible_mask=source.current_visible_mask,
            route_corridor_mask=source.nominal_trajectory.swept_mask,
            grid=source.grid,
            future_dt_s=future_dt_s,
            future_horizon_s=future_horizon_s,
            braking_deceleration_mps2=gt_config.braking_deceleration_mps2,
            angular_deceleration_radps2=(
                gt_config.angular_deceleration_radps2
            ),
            braking_margin_s=gt_config.braking_margin_s,
        )
        branch = resolution.branch
        replanning = generate_replanned_candidates(
            post_action_pose=branch.end_pose,
            nominal_trajectory=source.nominal_trajectory,
            action_id=action.action_id,
            config=config,
            static_occupancy=source.current_world.static_occupancy,
            braking_deceleration_mps2=gt_config.braking_deceleration_mps2,
            max_candidates=candidate_count,
        )
        if (
            replanning.plan_status
            == POST_PLAN_STATUS_SAFE_STOP_NO_FEASIBLE_PLAN
        ):
            aligned: list[LocalTrajectory] = []
            if (
                branch.branch_kind == "emergency_brake"
                and resolution.decision.brake_trace_collision_free is not True
            ):
                raise VerificationSourceIneligibleError(
                    "unsafe_emergency_brake",
                    "no-feasible-plan bypass requires a collision-free "
                    "emergency brake",
                )
            if not np.allclose(
                branch.end_control,
                0.0,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "no-feasible-plan bypass requires a safe stopped branch"
                )
        else:
            aligned = [
                compose_time_aligned_policy_trajectory(
                    template_trajectory=candidate.trajectory,
                    branch=branch,
                    future_dt_s=future_dt_s,
                    trajectory_id=_policy_trajectory_id(
                        action_id=action.action_id,
                        branch=branch,
                        suffix_trajectory_id=candidate.trajectory.trajectory_id,
                    ),
                    source_action_id=action.action_id,
                    source_nominal_trajectory_id=(
                        source.nominal_trajectory.trajectory_id
                    ),
                    suffix_trajectory=candidate.trajectory,
                    suffix_poses_in_parent_frame=(
                        candidate.poses_in_parent_frame
                    ),
                )
                for candidate in replanning.candidates
            ]
            if branch.branch_kind == "emergency_brake":
                aligned.append(
                    compose_time_aligned_policy_trajectory(
                        template_trajectory=source.nominal_trajectory,
                        branch=branch,
                        future_dt_s=future_dt_s,
                        trajectory_id=_policy_trajectory_id(
                            action_id=action.action_id,
                            branch=branch,
                            suffix_trajectory_id=None,
                        ),
                        source_action_id=action.action_id,
                        source_nominal_trajectory_id=(
                            source.nominal_trajectory.trajectory_id
                        ),
                    )
                )
        values[action.action_id] = evaluate_sampled_realization_value(
            realized_world=source.current_world,
            nominal_trajectory=source.nominal_trajectory,
            action=action,
            replanning_result=replanning,
            risk_loss=risk_loss,
            reject_cost=gt_config.reject_cost,
            risk_weight=gt_config.risk_weight,
            action_cost_config=config["verification_cost"],
            time_aligned_policy_trajectories=tuple(aligned),
        )
    if infeasible:
        raise VerificationSourceIneligibleError(
            "infeasible_actions",
            "complete six-action group blocked by " + ", ".join(infeasible),
        )
    samples = build_verification_samples(
        VerificationGroupInput(
            split=source.split,
            base_state_id=source.base_state_id,
            nominal_trajectory=source.nominal_trajectory,
            bev_history=source.bev_history,
            state_channels=source.state_channels,
            expected_fov_masks=fov_masks,
            value_results=values,
            provenance=source.provenance,
        ),
        library=action_library,
        grid=source.grid,
    )
    return VerificationGroupResult(
        version=VERIFICATION_PIPELINE_VERSION,
        samples=samples,
        values=MappingProxyType(dict(values)),
        sampled_child_world_id=source.current_world.world_id,
        infeasible_action_ids=(),
    )


__all__ = (
    "VERIFICATION_PIPELINE_VERSION",
    "VerificationGroupResult",
    "VerificationPipelineInput",
    "VerificationSourceIneligibleError",
    "build_finalized_verification_input",
    "build_real_verification_input",
    "build_verification_toy_input",
    "generate_verification_group",
)
