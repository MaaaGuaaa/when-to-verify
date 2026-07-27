"""Deterministic, target-blind M4 goal and static-occluder templates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np

from src.contracts import (
    ARRAY_DTYPE,
    BaseState,
    OracleContext,
    build_grid_spec,
    encode_dataclass,
    validate_base_state,
    validate_oracle_context,
)
from src.geometry import (
    CircleFootprint,
    CircleOccluder,
    RectangleFootprint,
    RectangleOccluder,
    StaticOccluder,
    inflate_footprint,
    intersects,
    point_signed_distance,
    rasterize_occluder,
    segment_intersects_occluder,
    wrap_angle,
)
from src.planning import (
    PlannedTebRoute,
    StaticTebRequest,
    plan_static_lightweight_teb,
)
from src.utils.seeding import derive_seed

from .event_contracts import footprint_from_spec
from .occluder_sampler import swept_footprint_intersects_occupancy
from .sop05r_contracts import Sop05rTebConfig, TebOccluderTemplate


_DIRECT_LINE_SAMPLES = 1001
_OFFSET_BISECTION_ITERATIONS = 40
_MAX_LATERAL_OFFSET_M = 3.0
_DIRECT_SEGMENT_EPSILON_M = 1e-9
_PLANNER_CACHE: dict[str, Any] = {}


class Sop05rTebTemplateError(ValueError):
    """One auditable M4 task-template rejection."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _readonly_array(value: object, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _dataclass_digest(value: object, *, namespace: str) -> str:
    arrays, metadata = encode_dataclass(value)
    digest = hashlib.sha256()
    digest.update(namespace.encode("ascii") + b"\0")
    digest.update(_canonical_json_bytes(metadata))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(_canonical_json_bytes(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_sop05r_teb_base_state_digest(base_state: BaseState) -> str:
    """Return the immutable source-state identity persisted by M4."""

    return _dataclass_digest(base_state, namespace="sop05r_teb_base_state_v1")


def canonical_sop05r_teb_oracle_context_digest(oracle_context: OracleContext) -> str:
    """Return the immutable protected-context identity persisted by M4."""

    return _dataclass_digest(
        oracle_context,
        namespace="sop05r_teb_oracle_context_v1",
    )


@dataclass(frozen=True)
class Sop05rTebTaskTemplate:
    """One target-blind source-state, goal, occluder-set, and valid M3 route."""

    template_id: str
    schedule_rank: tuple[int, int, int]
    family: str
    base_state_id: str
    source_base_state_digest: str
    source_oracle_context_digest: str
    occluders: tuple[StaticOccluder, ...]
    static_occupancy: np.ndarray
    local_goal_world_pose: np.ndarray
    route: PlannedTebRoute
    direct_line_clearance_m: float
    direct_corridor_intrusion_m: float
    relative_yaw_rad: float
    robot_radius_m: float
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "static_occupancy",
            _readonly_array(self.static_occupancy, dtype=ARRAY_DTYPE),
        )
        object.__setattr__(
            self,
            "local_goal_world_pose",
            _readonly_array(self.local_goal_world_pose, dtype=ARRAY_DTYPE),
        )
        if self.static_occupancy.ndim != 2 or not np.isin(
            self.static_occupancy, (0.0, 1.0)
        ).all():
            raise ValueError("static_occupancy must be a binary 2-D float32 array")
        if self.local_goal_world_pose.shape != (3,):
            raise ValueError("local_goal_world_pose must have shape [3]")
        if not self.occluders:
            raise ValueError("occluders must not be empty")
        if not np.isfinite(
            (
                self.direct_line_clearance_m,
                self.direct_corridor_intrusion_m,
                self.relative_yaw_rad,
                self.robot_radius_m,
            )
        ).all():
            raise ValueError("task-template geometry scalars must be finite")


@dataclass(frozen=True)
class Sop05rTebTemplateEvaluation:
    """One deterministic M4 schedule item and its accepted template or rejection."""

    template_id: str
    schedule_rank: tuple[int, int, int]
    template: Sop05rTebTaskTemplate | None
    rejection_reason: str | None
    evidence: Mapping[str, object]


def _robot_radius(base_config: Mapping[str, object]) -> float:
    robot = base_config["robot"]
    if not isinstance(robot, Mapping):
        raise ValueError("base_config.robot must be a mapping")
    return 0.5 * float(
        np.hypot(float(robot["length_m"]), float(robot["width_m"]))
    ) + float(robot["inflation_m"])


def _component_footprint(component: StaticOccluder):
    if isinstance(component, CircleOccluder):
        return CircleFootprint(component.radius_m), np.r_[component.center_xy, 0.0]
    return RectangleFootprint(component.length_m, component.width_m), component.pose


def _component_payload(component: StaticOccluder) -> dict[str, object]:
    if isinstance(component, CircleOccluder):
        return {
            "kind": "circle",
            "id": component.occluder_id,
            "semantic_type": component.semantic_type,
            "center_xy": [float(value) for value in component.center_xy],
            "radius_m": component.radius_m,
        }
    return {
        "kind": "rectangle",
        "id": component.occluder_id,
        "semantic_type": component.semantic_type,
        "pose": [float(value) for value in component.pose],
        "length_m": component.length_m,
        "width_m": component.width_m,
    }


def _planner_cache_key(
    *,
    source_base_digest: str,
    config_digest: str,
    goal_pose: np.ndarray,
    initial_control: np.ndarray,
    components: tuple[StaticOccluder, ...],
) -> str:
    payload = {
        "source_base_digest": source_base_digest,
        "config_digest": config_digest,
        "goal_pose": [float(value) for value in goal_pose],
        "initial_control": [float(value) for value in initial_control],
        "occluders": [_component_payload(component) for component in components],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _direct_clearance(
    components: tuple[StaticOccluder, ...],
    *,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    robot_radius_m: float,
) -> float:
    fractions = np.linspace(0.0, 1.0, _DIRECT_LINE_SAMPLES, dtype=np.float64)
    points = start_xy[None, :] + fractions[:, None] * (goal_xy - start_xy)[None, :]
    return min(
        float(np.min(point_signed_distance(component, points) - robot_radius_m))
        for component in components
    )


def _direct_segment_intersects(
    components: tuple[StaticOccluder, ...],
    *,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
) -> bool:
    starts = np.asarray(start_xy, dtype=np.float64)[None, :]
    ends = np.asarray(goal_xy, dtype=np.float64)[None, :]
    return any(
        bool(
            segment_intersects_occluder(
                component,
                starts,
                ends,
                epsilon_m=_DIRECT_SEGMENT_EPSILON_M,
            )[0]
        )
        for component in components
    )


def _make_components(
    *,
    template: TebOccluderTemplate,
    template_id: str,
    anchor_xy: np.ndarray,
    route_yaw_rad: float,
    relative_yaw_rad: float,
    side: float,
    lateral_offset_m: float,
) -> tuple[StaticOccluder, ...]:
    route_normal = np.asarray(
        [-np.sin(route_yaw_rad), np.cos(route_yaw_rad)],
        dtype=np.float64,
    )
    center = anchor_xy + side * lateral_offset_m * route_normal
    if template.shape == "circle":
        assert template.radius_m is not None
        return (
            CircleOccluder(
                occluder_id=f"{template_id}-circle",
                semantic_type=template.semantic_type,
                center_xy=center,
                radius_m=template.radius_m,
            ),
        )
    if template.shape == "rectangle":
        assert template.length_m is not None and template.width_m is not None
        return (
            RectangleOccluder(
                occluder_id=f"{template_id}-rectangle",
                semantic_type=template.semantic_type,
                pose=np.r_[center, route_yaw_rad + relative_yaw_rad],
                length_m=template.length_m,
                width_m=template.width_m,
            ),
        )

    assert template.arm_lengths_m is not None and template.arm_width_m is not None
    horizontal_length, vertical_length = template.arm_lengths_m
    arm_width = template.arm_width_m
    horizontal_yaw = route_yaw_rad + relative_yaw_rad
    horizontal_axis = np.asarray(
        [np.cos(horizontal_yaw), np.sin(horizontal_yaw)],
        dtype=np.float64,
    )
    vertical_axis = np.asarray(
        [-np.sin(horizontal_yaw), np.cos(horizontal_yaw)],
        dtype=np.float64,
    )
    vertical_sign = (
        1.0
        if float(np.dot(vertical_axis, side * route_normal)) >= 0.0
        else -1.0
    )
    vertical_center = (
        center
        - 0.5 * (horizontal_length - arm_width) * horizontal_axis
        + vertical_sign * 0.5 * (vertical_length - arm_width) * vertical_axis
    )
    return (
        RectangleOccluder(
            occluder_id=f"{template_id}-l-horizontal",
            semantic_type=template.semantic_type,
            pose=np.r_[center, horizontal_yaw],
            length_m=horizontal_length,
            width_m=arm_width,
        ),
        RectangleOccluder(
            occluder_id=f"{template_id}-l-vertical",
            semantic_type=template.semantic_type,
            pose=np.r_[vertical_center, horizontal_yaw + 0.5 * np.pi],
            length_m=vertical_length,
            width_m=arm_width,
        ),
    )


def _place_components(
    *,
    template: TebOccluderTemplate,
    template_id: str,
    anchor_xy: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    route_yaw_rad: float,
    relative_yaw_rad: float,
    side: float,
    required_direct_clearance_m: float,
    robot_radius_m: float,
) -> tuple[tuple[StaticOccluder, ...], float]:
    def at_offset(offset_m: float) -> tuple[StaticOccluder, ...]:
        return _make_components(
            template=template,
            template_id=template_id,
            anchor_xy=anchor_xy,
            route_yaw_rad=route_yaw_rad,
            relative_yaw_rad=relative_yaw_rad,
            side=side,
            lateral_offset_m=offset_m,
        )

    lower = 0.0
    upper = _MAX_LATERAL_OFFSET_M
    if (
        _direct_clearance(
            at_offset(upper),
            start_xy=start_xy,
            goal_xy=goal_xy,
            robot_radius_m=robot_radius_m,
        )
        < required_direct_clearance_m
    ):
        raise Sop05rTebTemplateError("occluder_out_of_bounds")
    for _ in range(_OFFSET_BISECTION_ITERATIONS):
        midpoint = 0.5 * (lower + upper)
        clearance = _direct_clearance(
            at_offset(midpoint),
            start_xy=start_xy,
            goal_xy=goal_xy,
            robot_radius_m=robot_radius_m,
        )
        if clearance < required_direct_clearance_m:
            lower = midpoint
        else:
            upper = midpoint
    # ``lower`` is the greatest offset still satisfying the minimum
    # direct-corridor intrusion; ``upper`` is the first offset that misses it.
    components = at_offset(lower)
    if not _direct_segment_intersects(
        components,
        start_xy=start_xy,
        goal_xy=goal_xy,
    ):
        if not _direct_segment_intersects(
            at_offset(0.0),
            start_xy=start_xy,
            goal_xy=goal_xy,
        ):
            raise Sop05rTebTemplateError("direct_path_clear")
        intersection_lower = 0.0
        intersection_upper = lower
        for _ in range(_OFFSET_BISECTION_ITERATIONS):
            midpoint = 0.5 * (intersection_lower + intersection_upper)
            if _direct_segment_intersects(
                at_offset(midpoint),
                start_xy=start_xy,
                goal_xy=goal_xy,
            ):
                intersection_lower = midpoint
            else:
                intersection_upper = midpoint
        components = at_offset(intersection_lower)
    return (
        components,
        _direct_clearance(
            components,
            start_xy=start_xy,
            goal_xy=goal_xy,
            robot_radius_m=robot_radius_m,
        ),
    )


def _validate_static_geometry(
    *,
    components: tuple[StaticOccluder, ...],
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    grid = build_grid_spec(dict(base_config))
    try:
        component_masks = tuple(
            rasterize_occluder(component, grid) for component in components
        )
    except ValueError as exc:
        raise Sop05rTebTemplateError("occluder_out_of_bounds") from exc
    occluder_mask = np.logical_or.reduce(component_masks)
    source_static = (
        np.zeros((grid.height, grid.width), dtype=bool)
        if base_state.static_map_local is None
        else np.asarray(base_state.static_map_local != 0, dtype=bool)
    )
    if np.any(occluder_mask & source_static):
        raise Sop05rTebTemplateError("source_static_overlap")

    robot = base_config["robot"]
    robot_footprint = inflate_footprint(
        RectangleFootprint(
            float(robot["length_m"]),
            float(robot["width_m"]),
        ),
        float(robot["inflation_m"]),
    )
    for component in components:
        component_footprint, component_pose = _component_footprint(component)
        if any(
            intersects(robot_footprint, pose, component_footprint, component_pose)
            for pose in base_state.robot_history
        ):
            raise Sop05rTebTemplateError("robot_history_overlap")

    for object_id in sorted(oracle_context.dynamic_object_history):
        footprint = footprint_from_spec(oracle_context.dynamic_object_specs[object_id])
        poses = np.vstack(
            (
                oracle_context.dynamic_object_history[object_id],
                oracle_context.dynamic_object_future[object_id],
            )
        )
        if swept_footprint_intersects_occupancy(
            footprint,
            poses,
            occluder_mask,
            grid=grid,
        ):
            raise Sop05rTebTemplateError("context_overlap")
    return (
        np.asarray(source_static, dtype=ARRAY_DTYPE),
        np.asarray(source_static | occluder_mask, dtype=ARRAY_DTYPE),
    )


def _schedule(
    teb_config: Sop05rTebConfig,
) -> Iterator[tuple[int, int, int, TebOccluderTemplate, float, float]]:
    count = 0
    for template_index, template in enumerate(teb_config.template.occluders):
        for bearing_index, bearing_deg in enumerate(teb_config.template.goal_bearings_deg):
            for distance_index, distance_m in enumerate(teb_config.template.goal_distances_m):
                if count >= teb_config.generation.max_templates_per_base:
                    return
                yield (
                    template_index,
                    bearing_index,
                    distance_index,
                    template,
                    bearing_deg,
                    distance_m,
                )
                count += 1


def iter_sop05r_teb_task_templates(
    *,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, object],
    teb_config: Sop05rTebConfig,
    seed: int,
) -> Iterator[Sop05rTebTemplateEvaluation]:
    """Yield deterministic M4 task templates before any snippet or target is read."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if not isinstance(teb_config, Sop05rTebConfig):
        raise TypeError("teb_config must be a Sop05rTebConfig")
    if oracle_context.base_state_id != base_state.state_id:
        raise ValueError("oracle_context.base_state_id must match base_state.state_id")
    grid = build_grid_spec(dict(base_config))
    validate_base_state(base_state, grid)
    validate_oracle_context(oracle_context, grid)

    source_base_digest = canonical_sop05r_teb_base_state_digest(base_state)
    source_oracle_digest = canonical_sop05r_teb_oracle_context_digest(oracle_context)
    start_pose = np.asarray(base_state.robot_history[-1], dtype=np.float64)
    start_xy = start_pose[:2]
    robot_radius_m = _robot_radius(base_config)
    min_clearance, max_clearance = (
        teb_config.planner.represented_occluder_clearance_range_m
    )
    yaw_min_deg, yaw_max_deg = teb_config.template.relative_yaw_abs_range_deg

    for (
        template_index,
        bearing_index,
        distance_index,
        template,
        bearing_deg,
        distance_m,
    ) in _schedule(teb_config):
        schedule_rank = (template_index, bearing_index, distance_index)
        template_id = (
            f"teb-{template.template_id}-bearing-{bearing_index}-distance-{distance_index}"
        )
        rng = np.random.default_rng(
            derive_seed(seed, base_state.state_id, teb_config.digest, template_id)
        )
        route_yaw_rad = float(wrap_angle(start_pose[2] + np.deg2rad(bearing_deg)))
        route_direction = np.asarray(
            [np.cos(route_yaw_rad), np.sin(route_yaw_rad)],
            dtype=np.float64,
        )
        goal_pose = np.asarray(
            [
                *(start_xy + distance_m * route_direction),
                route_yaw_rad,
            ],
            dtype=ARRAY_DTYPE,
        )
        fraction = float(
            rng.uniform(*teb_config.generation.collision_route_path_fraction_range)
        )
        anchor_xy = start_xy + fraction * (goal_pose[:2] - start_xy)
        side = float(rng.choice((-1.0, 1.0)))
        if template.shape == "circle":
            relative_yaw_rad = 0.0
        else:
            relative_yaw_rad = float(
                rng.choice((-1.0, 1.0))
                * np.deg2rad(rng.uniform(yaw_min_deg, yaw_max_deg))
            )
        minimum_intrusion_m = (
            teb_config.generation.minimum_direct_corridor_intrusion_m
        )
        required_direct_clearance_m = min_clearance - minimum_intrusion_m
        evidence: dict[str, object] = {
            "config_digest": teb_config.digest,
            "base_state_id": base_state.state_id,
            "source_base_state_digest": source_base_digest,
            "source_oracle_context_digest": source_oracle_digest,
            "schedule_rank": list(schedule_rank),
            "seed": int(seed),
            "family": template.shape,
            "relative_yaw_rad": relative_yaw_rad,
            "minimum_direct_corridor_intrusion_m": minimum_intrusion_m,
        }
        try:
            components, direct_clearance_m = _place_components(
                template=template,
                template_id=template_id,
                anchor_xy=anchor_xy,
                start_xy=start_xy,
                goal_xy=goal_pose[:2],
                route_yaw_rad=route_yaw_rad,
                relative_yaw_rad=relative_yaw_rad,
                side=side,
                required_direct_clearance_m=required_direct_clearance_m,
                robot_radius_m=robot_radius_m,
            )
            planner_static, output_static = _validate_static_geometry(
                components=components,
                base_state=base_state,
                oracle_context=oracle_context,
                base_config=base_config,
            )
            request = StaticTebRequest(
                start_pose=np.asarray(start_pose, dtype=ARRAY_DTYPE),
                initial_control=np.asarray(base_state.robot_state, dtype=ARRAY_DTYPE),
                local_goal_world_pose=goal_pose,
                static_occupancy=planner_static,
                occluders=components,
                base_config=base_config,
                planner_config=teb_config.planner,
            )
            cache_key = _planner_cache_key(
                source_base_digest=source_base_digest,
                config_digest=teb_config.digest,
                goal_pose=goal_pose,
                initial_control=request.initial_control,
                components=components,
            )
            result = _PLANNER_CACHE.get(cache_key)
            if result is None:
                result = plan_static_lightweight_teb(request)
                _PLANNER_CACHE[cache_key] = result
            if result.route is None:
                raise Sop05rTebTemplateError(result.rejection_reason or "teb_no_route")
            route_clearance_m = min(
                float(
                    np.min(
                        point_signed_distance(
                            component,
                            result.route.sampled_poses_world[:, :2],
                        )
                        - robot_radius_m
                    )
                )
                for component in components
            )
            if not min_clearance - 1e-5 <= route_clearance_m <= max_clearance + 1e-5:
                raise Sop05rTebTemplateError("teb_static_collision")
            task_template = Sop05rTebTaskTemplate(
                template_id=template_id,
                schedule_rank=schedule_rank,
                family=template.shape,
                base_state_id=base_state.state_id,
                source_base_state_digest=source_base_digest,
                source_oracle_context_digest=source_oracle_digest,
                occluders=components,
                static_occupancy=output_static,
                local_goal_world_pose=goal_pose,
                route=result.route,
                direct_line_clearance_m=direct_clearance_m,
                direct_corridor_intrusion_m=min_clearance - direct_clearance_m,
                relative_yaw_rad=relative_yaw_rad,
                robot_radius_m=robot_radius_m,
                provenance={
                    **evidence,
                    "goal_bearing_deg": bearing_deg,
                    "goal_distance_m": distance_m,
                    "lateral_side": int(side),
                    "planner_cache_key": cache_key,
                    "direct_line_clearance_m": direct_clearance_m,
                    "route_clearance_m": route_clearance_m,
                },
            )
        except Sop05rTebTemplateError as exc:
            yield Sop05rTebTemplateEvaluation(
                template_id=template_id,
                schedule_rank=schedule_rank,
                template=None,
                rejection_reason=exc.reason,
                evidence=evidence,
            )
            continue
        yield Sop05rTebTemplateEvaluation(
            template_id=template_id,
            schedule_rank=schedule_rank,
            template=task_template,
            rejection_reason=None,
            evidence=task_template.provenance,
        )
