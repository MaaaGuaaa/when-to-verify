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
    expected_verification_fov_mask,
    fit_signature_normalizer,
    make_observation_signature,
    simulate_counterfactual_observation,
)
from src.generation.dynamic_object_transplant import footprint_from_spec
from src.generation.observation_renderer import render_observation
from src.generation.scenario_bank import (
    ScenarioBankConfig,
    ScenarioBankGeometryError,
    build_scenario_bank,
)
from src.generation.sop06_pipeline import render_sop06_mother_event
from src.generation.structural_blindspot import StructuralBlindSpot
from src.generation.verification_gt import (
    TypedFootprintRiskLoss,
    VerificationGTConfig,
    VerificationValueResult,
    evaluate_verification_value,
)
from src.generation.verification_toy import build_verification_toy_world
from src.planning.query_maps import build_local_trajectory
from src.planning.replanning import generate_replanned_candidates
from src.planning.trajectory_sampler import sample_candidate_rollouts
from src.planning.verification_actions import (
    VerificationActionLibrary,
    action_endpoint,
    check_action_feasibility,
)
from src.utils.config import validate_config
from src.utils.seeding import stable_digest


VERIFICATION_PIPELINE_VERSION = "verification_pipeline_v1"


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
    target_object_id: str
    robot_pose: np.ndarray
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
    values: Mapping[str, VerificationValueResult]
    scenario_bank_digest: str
    bank_size: int
    posterior_mode: str
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
    world: OracleWorld, grid: GridSpec
) -> tuple[float, float]:
    config = world.blind_spot_config
    if not isinstance(config, Mapping):
        raise ValueError("world blind_spot_config must be a mapping")
    structural = config.get("structural")
    if structural is not None:
        if not isinstance(structural, Mapping):
            raise ValueError("structural blind-spot config must be a mapping")
        fov = float(np.deg2rad(structural["forward_fov_deg"]))
        sensor_range = float(structural["range_m"])
    else:
        fov = float(2.0 * np.pi)
        sensor_range = float(
            np.hypot(
                grid.height * grid.resolution_m,
                grid.width * grid.resolution_m,
            )
        )
    if not np.isfinite([fov, sensor_range]).all() or fov <= 0.0 or sensor_range <= 0.0:
        raise ValueError("verification sensor geometry must be finite and positive")
    return fov, sensor_range


def build_real_verification_input(
    source: VerificationSourceEvent,
    *,
    base_config: Mapping[str, Any],
    sop05_batch_digest: str,
    sop07_collection_digest: str,
    scientific_status: str,
    cross_split_status: str,
) -> VerificationPipelineInput:
    """Render deployment history and retain oracle state only in label fields."""

    if not isinstance(source, VerificationSourceEvent):
        raise TypeError("source must be a VerificationSourceEvent")
    config = dict(base_config)
    validate_config(config)
    grid = build_grid_spec(config)
    rendered = render_sop06_mother_event(
        record=source.event.target_motion_record,
        world=source.event.world,
        base_state=source.base_state,
        oracle_context=source.oracle_context,
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
    sensor_fov, sensor_range = _verification_sensor_geometry(
        source.event.world, grid
    )
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


def build_verification_toy_input(
    base_config: Mapping[str, Any],
    *,
    group_index: int,
) -> tuple[VerificationPipelineInput, dict[str, Any]]:
    """Create one distinct toy identity while preserving hand-checkable geometry."""

    index = _positive_integer(group_index + 1, name="group_index_plus_one") - 1
    config = deepcopy(dict(base_config))
    config["bev"]["range_m"] = 8.0
    config["bev"]["size"] = 80
    validate_config(config)
    toy = build_verification_toy_world()
    if build_grid_spec(config) != toy.grid:
        raise RuntimeError("toy config and toy grid differ")
    base_id = f"toy-base-{index:04d}"
    sensor = StructuralBlindSpot(forward_fov_deg=20.0, range_m=4.0)
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
        current_visible_mask=np.asarray(visible, dtype=bool, order="C"),
        current_age_map=np.array(age, dtype=ARRAY_DTYPE, order="C", copy=True),
        bev_history=np.array(
            rendered.bev_history, dtype=ARRAY_DTYPE, order="C", copy=True
        ),
        state_channels=np.array(
            rendered.state_channels, dtype=ARRAY_DTYPE, order="C", copy=True
        ),
        sensor_fov_rad=float(np.deg2rad(20.0)),
        sensor_range_m=4.0,
        provenance={
            "source_mode": "toy",
            "scientific_status": "toy_smoke_only",
            "source_artifact_digest": case_digest,
            "toy_case_digest": case_digest,
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
    if source.target_object_id not in hidden:
        raise ValueError("scenario target must be hidden in the current observation")
    return tuple(hidden)


def generate_verification_group(
    source: VerificationPipelineInput,
    *,
    base_config: Mapping[str, Any],
    action_library: VerificationActionLibrary,
    gt_config: VerificationGTConfig,
    scenario_config: ScenarioBankConfig,
    bank_size: int,
    posterior_mode: str,
    posterior_temperature: float | None,
    seed: int,
    max_replan_candidates: int,
) -> VerificationGroupResult:
    """Run all six actions through the same simulator-defined value path."""

    if not isinstance(source, VerificationPipelineInput):
        raise TypeError("source must be a VerificationPipelineInput")
    config = dict(base_config)
    validate_config(config)
    if build_grid_spec(config) != source.grid:
        raise ValueError("base_config grid differs from verification source")
    if not isinstance(action_library, VerificationActionLibrary):
        raise TypeError("action_library must be a VerificationActionLibrary")
    if not isinstance(gt_config, VerificationGTConfig):
        raise TypeError("gt_config must be a VerificationGTConfig")
    if not isinstance(scenario_config, ScenarioBankConfig):
        raise TypeError("scenario_config must be a ScenarioBankConfig")
    bank_count = _positive_integer(bank_size, name="bank_size")
    candidate_count = _positive_integer(
        max_replan_candidates, name="max_replan_candidates"
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if posterior_mode == "exact" and posterior_temperature is not None:
        raise ValueError("exact posterior does not use a temperature")
    if posterior_mode == "soft" and posterior_temperature is None:
        raise ValueError("soft posterior requires a temperature")

    try:
        bank = build_scenario_bank(
            current_world=source.current_world,
            target_object_id=source.target_object_id,
            current_dynamic_poses=source.current_dynamic_poses,
            current_visible_mask=source.current_visible_mask,
            grid=source.grid,
            split=source.split,
            source_namespace=source.source_namespace,
            seed=seed,
            size=bank_count,
            config=scenario_config,
        )
    except ScenarioBankGeometryError as exc:
        reason = (
            "scenario_current_static_overlap"
            if exc.variant_kind == "current"
            else "scenario_variant_static_overlap"
        )
        raise VerificationSourceIneligibleError(reason, str(exc)) from exc
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
    hidden_ids = _hidden_object_ids(source)
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
    values: dict[str, VerificationValueResult] = {}
    infeasible: list[str] = []
    for action in action_library.actions:
        feasibility = check_action_feasibility(
            source.robot_pose,
            action,
            robot_footprint=robot_footprint,
            static_occupancy=source.current_world.static_occupancy,
            grid=source.grid,
            dynamic_object_poses=dynamic_poses,
            dynamic_object_footprints=dynamic_footprints,
            dynamic_dt_s=float(config["bev"]["future_dt_s"]),
        )
        if not feasibility.feasible:
            infeasible.append(action.action_id)
            continue
        post_pose = action_endpoint(source.robot_pose, action)
        replanning = generate_replanned_candidates(
            post_action_pose=post_pose,
            nominal_trajectory=source.nominal_trajectory,
            action_id=action.action_id,
            config=config,
            static_occupancy=source.current_world.static_occupancy,
            braking_deceleration_mps2=gt_config.braking_deceleration_mps2,
            max_candidates=candidate_count,
        )
        fov_masks[action.action_id] = expected_verification_fov_mask(
            source.current_world.static_occupancy,
            source.grid,
            sensor_pose=post_pose,
            fov_rad=source.sensor_fov_rad,
            max_range_m=source.sensor_range_m,
        )
        observations = tuple(
            simulate_counterfactual_observation(
                post_action_pose=post_pose,
                action_duration_s=action.duration_s,
                static_occupancy=hypothesis.world.static_occupancy,
                dynamic_current_poses=hypothesis.current_dynamic_poses,
                dynamic_future_poses=hypothesis.world.dynamic_object_trajectories,
                dynamic_specs=hypothesis.world.dynamic_object_specs,
                current_visible_mask=source.current_visible_mask,
                current_age_map=source.current_age_map,
                grid=source.grid,
                future_dt_s=float(config["bev"]["future_dt_s"]),
                age_max_s=float(config["age_map"]["a_max_s"]),
                fov_rad=source.sensor_fov_rad,
                max_range_m=source.sensor_range_m,
            )
            for hypothesis in bank.hypotheses
        )
        replanned_masks = tuple(
            candidate.swept_mask_in_parent_frame
            for candidate in replanning.candidates
        )
        signatures = np.stack(
            [
                make_observation_signature(
                    observation,
                    grid=source.grid,
                    original_swept_mask=source.nominal_trajectory.swept_mask,
                    replanned_swept_masks=replanned_masks,
                    local_goal_corridor_mask=source.nominal_trajectory.swept_mask,
                    critical_region_mask=source.nominal_trajectory.swept_mask,
                    previous_age_map=source.current_age_map,
                )
                for observation in observations
            ],
            axis=0,
        ).astype(ARRAY_DTYPE)
        normalizer = (
            fit_signature_normalizer(signatures, split="train")
            if posterior_mode == "soft"
            else None
        )
        values[action.action_id] = evaluate_verification_value(
            bank=bank,
            nominal_trajectory=source.nominal_trajectory,
            action=action,
            observations=observations,
            signatures=signatures,
            replanning_results=(replanning,) * bank.size,
            risk_loss=risk_loss,
            posterior_mode=posterior_mode,
            signature_normalizer=normalizer,
            posterior_temperature=posterior_temperature,
            reject_cost=gt_config.reject_cost,
            risk_weight=gt_config.risk_weight,
            action_cost_config=config["verification_cost"],
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
        scenario_bank_digest=bank.semantic_digest,
        bank_size=bank.size,
        posterior_mode=posterior_mode,
        infeasible_action_ids=(),
    )


__all__ = (
    "VERIFICATION_PIPELINE_VERSION",
    "VerificationGroupResult",
    "VerificationPipelineInput",
    "VerificationSourceIneligibleError",
    "build_real_verification_input",
    "build_verification_toy_input",
    "generate_verification_group",
)
