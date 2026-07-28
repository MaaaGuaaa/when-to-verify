"""M1 input and rigid-rotation primitives for the SOP05 unseen-history prior."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from numbers import Real
from types import MappingProxyType

import numpy as np

from src.contracts import GridSpec
from src.geometry import (
    CircleFootprint,
    RectangleFootprint,
    footprint_aabb,
    grid_bounds,
    raycast_visibility,
    rasterize_footprint,
)
from src.generation.event_contracts import footprint_from_spec
from src.generation.occluder_sampler import synchronized_sweeps_intersect, swept_footprint_intersects_occupancy
from src.generation.structural_blindspot import (
    StructuralBlindSpot,
    build_structural_visibility,
    footprint_visibility_sequence,
)
from src.utils.config import config_digest


UNSEEN_PRIOR_GENERATOR_VERSION = "sop05_unseen_history_prior_v1"
UNSEEN_PRIOR_CONTRACT_VERSION = "sop05_unseen_history_prior_contract_v1"
UNSEEN_PRIOR_REGIME = "unseen_in_history_window"
LONG40_SCHEMA_VERSION = "4.0.0"
LONG40_LAYOUT_VERSION = "history8_current7_future32_v1"

_EXPECTED_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "generator_version",
        "contract_version",
        "p_hidden_human",
        "max_attempts_per_mother",
        "max_variants_per_mother",
        "hard_total_sample_cap",
        "manifest_targets",
        "seed",
    }
)
_MANIFEST_TARGETS = (50000, 100000, 125000)


class UnseenPriorConfigError(ValueError):
    """Raised when the frozen unseen-prior configuration is invalid."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class UnseenPriorInputError(ValueError):
    """Raised when a Long40 source input violates the public contract."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _require_nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise UnseenPriorInputError(f"{name} must be a non-empty string")
    return value


def _owned_footprint(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise UnseenPriorInputError("footprint_spec must be a non-empty mapping")
    try:
        return deepcopy(dict(value))
    except (TypeError, ValueError) as exc:
        raise UnseenPriorInputError("footprint_spec must be copyable") from exc


def _long40_array(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise UnseenPriorInputError(f"{name} must be a numeric array") from exc
    if array.dtype != np.dtype(np.float32):
        raise UnseenPriorInputError(f"{name} must have dtype float32")
    if array.shape != shape:
        raise UnseenPriorInputError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise UnseenPriorInputError(f"{name} must contain only finite values")
    result = np.array(array, dtype=np.float32, order="C", copy=True)
    result.flags.writeable = False
    return result


def _finite_real(value: object, *, name: str, error_type: type[ValueError]) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise error_type(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise error_type(f"{name} must be finite")
    return result


def _integer(
    value: object, *, name: str, minimum: int = 0, error_type: type[ValueError]
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise error_type(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise error_type(f"{name} must be at least {minimum}")
    return result


def _float32_wrapped_headings(headings: np.ndarray, *, angle_rad: float) -> np.ndarray:
    """Wrap with the repository's ``[-pi, pi)`` convention in float32."""

    wrapped64 = np.remainder(
        headings.astype(np.float64) + angle_rad + np.pi, 2.0 * np.pi
    ) - np.pi
    wrapped64 = np.where(wrapped64 >= np.pi, wrapped64 - 2.0 * np.pi, wrapped64)
    wrapped = np.asarray(wrapped64, dtype=np.float32)
    lower = np.nextafter(np.float32(-np.pi), np.float32(np.inf))
    upper = np.nextafter(np.float32(np.pi), np.float32(-np.inf))
    return np.clip(wrapped, lower, upper)


@dataclass(frozen=True)
class Long40TargetMotion:
    """Authenticated target motion using the fixed 8-history/32-future layout."""

    target_dynamic_object_id: str
    source_recording_id: str
    source_session_id: str
    source_snippet_id: str
    source_object_id: str
    object_type: str
    footprint_spec: dict[str, object]
    layout_version: str
    positions: np.ndarray
    velocities: np.ndarray
    headings: np.ndarray

    def __post_init__(self) -> None:
        for field_name in (
            "target_dynamic_object_id",
            "source_recording_id",
            "source_session_id",
            "source_snippet_id",
            "source_object_id",
            "object_type",
        ):
            _require_nonempty_string(getattr(self, field_name), name=field_name)
        if self.layout_version != LONG40_LAYOUT_VERSION:
            raise UnseenPriorInputError(
                f"layout_version must equal {LONG40_LAYOUT_VERSION!r}"
            )
        object.__setattr__(self, "footprint_spec", _owned_footprint(self.footprint_spec))
        object.__setattr__(
            self,
            "positions",
            _long40_array(self.positions, name="positions", shape=(40, 2)),
        )
        object.__setattr__(
            self,
            "velocities",
            _long40_array(self.velocities, name="velocities", shape=(40, 2)),
        )
        object.__setattr__(
            self,
            "headings",
            _long40_array(self.headings, name="headings", shape=(40,)),
        )


def transform_long40_target(
    target: Long40TargetMotion, *, angle_rad: float
) -> Long40TargetMotion:
    """Rigidly rotate all 40 target frames about the index-7 target position."""

    if not isinstance(target, Long40TargetMotion):
        raise UnseenPriorInputError("target must be a Long40TargetMotion")
    angle = _finite_real(
        angle_rad, name="angle_rad", error_type=UnseenPriorInputError
    )
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    source_positions = target.positions.astype(np.float64)
    pivot = source_positions[7]
    positions = (
        pivot + (source_positions - pivot) @ rotation.T
    ).astype(np.float32)
    velocities = (target.velocities.astype(np.float64) @ rotation.T).astype(np.float32)
    headings = _float32_wrapped_headings(target.headings, angle_rad=angle)

    return Long40TargetMotion(
        target_dynamic_object_id=target.target_dynamic_object_id,
        source_recording_id=target.source_recording_id,
        source_session_id=target.source_session_id,
        source_snippet_id=target.source_snippet_id,
        source_object_id=target.source_object_id,
        object_type=target.object_type,
        footprint_spec=target.footprint_spec,
        layout_version=target.layout_version,
        positions=positions,
        velocities=velocities,
        headings=headings,
    )


def _readonly_float32_array(
    value: object, *, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise UnseenPriorInputError(f"{name} must be a numeric array") from exc
    if array.dtype != np.dtype(np.float32) or array.shape != shape:
        raise UnseenPriorInputError(f"{name} must be float32 with shape {shape}")
    if not np.isfinite(array).all():
        raise UnseenPriorInputError(f"{name} must contain only finite values")
    result = np.array(array, dtype=np.float32, order="C", copy=True)
    result.flags.writeable = False
    return result


def _readonly_occupancy(
    value: object, *, name: str, grid: GridSpec
) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise UnseenPriorInputError(f"{name} must be a numeric occupancy grid") from exc
    if array.shape != (grid.height, grid.width) or array.dtype.kind not in "biuf":
        raise UnseenPriorInputError(
            f"{name} must be numeric with shape ({grid.height}, {grid.width})"
        )
    if not np.isfinite(array).all():
        raise UnseenPriorInputError(f"{name} must contain only finite values")
    result = np.asarray(array != 0, dtype=bool).copy(order="C")
    result.flags.writeable = False
    return result


def _target_footprint(target: Long40TargetMotion) -> CircleFootprint | RectangleFootprint:
    try:
        footprint = footprint_from_spec(target.footprint_spec)
    except (TypeError, ValueError) as exc:
        raise UnseenPriorInputError("target footprint_spec is not a production footprint") from exc
    if not isinstance(footprint, (CircleFootprint, RectangleFootprint)):
        raise UnseenPriorInputError("target footprint is unsupported")
    return footprint


@dataclass(frozen=True)
class UnseenPriorContextObstacle:
    """One represented dynamic context obstacle over the Long40 interval."""

    object_id: str
    footprint_spec: dict[str, object]
    poses: np.ndarray

    def __post_init__(self) -> None:
        _require_nonempty_string(self.object_id, name="object_id")
        object.__setattr__(self, "footprint_spec", _owned_footprint(self.footprint_spec))
        object.__setattr__(
            self,
            "poses",
            _readonly_float32_array(self.poses, name="poses", shape=(40, 3)),
        )
        try:
            footprint_from_spec(self.footprint_spec)
        except (TypeError, ValueError) as exc:
            raise UnseenPriorInputError(
                "context footprint_spec is not a production footprint"
            ) from exc


@dataclass(frozen=True)
class UnseenPriorMother:
    """Validated geometry and sensor state required for one M2 decision."""

    mother_id: str
    split: str
    target_motion: Long40TargetMotion
    grid: GridSpec
    robot_footprint: CircleFootprint | RectangleFootprint
    robot_history: np.ndarray
    static_occupancy: np.ndarray
    occluder_occupancy: np.ndarray
    context_obstacles: tuple[UnseenPriorContextObstacle, ...]
    sensor_config: StructuralBlindSpot | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.mother_id, name="mother_id")
        _require_nonempty_string(self.split, name="split")
        if not isinstance(self.target_motion, Long40TargetMotion):
            raise UnseenPriorInputError("target_motion must be a Long40TargetMotion")
        if not isinstance(self.grid, GridSpec):
            raise UnseenPriorInputError("grid must be a GridSpec")
        if self.grid.history_steps != 8 or self.grid.future_steps != 32:
            raise UnseenPriorInputError("grid must use the Long40 8/32 layout")
        try:
            grid_bounds(self.grid)
        except (TypeError, ValueError) as exc:
            raise UnseenPriorInputError("grid is invalid") from exc
        if not isinstance(self.robot_footprint, (CircleFootprint, RectangleFootprint)):
            raise UnseenPriorInputError("robot_footprint must be a production footprint")
        object.__setattr__(
            self,
            "robot_history",
            _readonly_float32_array(
                self.robot_history,
                name="robot_history",
                shape=(self.grid.history_steps, 3),
            ),
        )
        object.__setattr__(
            self,
            "static_occupancy",
            _readonly_occupancy(
                self.static_occupancy, name="static_occupancy", grid=self.grid
            ),
        )
        object.__setattr__(
            self,
            "occluder_occupancy",
            _readonly_occupancy(
                self.occluder_occupancy,
                name="occluder_occupancy",
                grid=self.grid,
            ),
        )
        if not isinstance(self.context_obstacles, tuple) or not all(
            isinstance(obstacle, UnseenPriorContextObstacle)
            for obstacle in self.context_obstacles
        ):
            raise UnseenPriorInputError(
                "context_obstacles must be a tuple of UnseenPriorContextObstacle"
            )
        context_ids = tuple(obstacle.object_id for obstacle in self.context_obstacles)
        if len(set(context_ids)) != len(context_ids):
            raise UnseenPriorInputError("context_obstacles object IDs must be unique")
        if self.target_motion.target_dynamic_object_id in context_ids:
            raise UnseenPriorInputError("target ID must not appear in context_obstacles")
        if self.sensor_config is not None and not isinstance(
            self.sensor_config, StructuralBlindSpot
        ):
            raise UnseenPriorInputError(
                "sensor_config must be a StructuralBlindSpot or None"
            )
        _target_footprint(self.target_motion)


@dataclass(frozen=True)
class UnseenPriorCandidateContext:
    """One mother-local cache for angle-invariant history visibility work."""

    mother_id: str
    target_dynamic_object_id: str
    grid_shape: tuple[int, int, int]
    target_footprint: CircleFootprint | RectangleFootprint
    history_visibility_masks: np.ndarray

    def __post_init__(self) -> None:
        _require_nonempty_string(self.mother_id, name="mother_id")
        _require_nonempty_string(
            self.target_dynamic_object_id,
            name="target_dynamic_object_id",
        )
        if (
            not isinstance(self.grid_shape, tuple)
            or len(self.grid_shape) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.grid_shape
            )
        ):
            raise UnseenPriorInputError("candidate context grid_shape is invalid")
        if not isinstance(self.target_footprint, (CircleFootprint, RectangleFootprint)):
            raise UnseenPriorInputError("candidate context target footprint is invalid")
        expected = (
            self.grid_shape[0],
            self.grid_shape[1],
            self.grid_shape[2],
        )
        masks = np.asarray(self.history_visibility_masks)
        if masks.shape != expected or masks.dtype != np.bool_:
            raise UnseenPriorInputError(
                "candidate context history visibility masks have invalid shape or dtype"
            )
        owned = np.array(masks, dtype=np.bool_, order="C", copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "history_visibility_masks", owned)


@dataclass(frozen=True)
class UnseenPriorCandidateDecision:
    """One deterministic legality result with no future-robot filtering."""

    legal: bool
    rejection_reason: str | None
    accepted_target: Long40TargetMotion | None

    def __post_init__(self) -> None:
        if self.legal:
            if self.rejection_reason is not None or self.accepted_target is None:
                raise UnseenPriorInputError("legal decision must contain one accepted target")
        elif self.rejection_reason not in {
            "nonfinite_or_out_of_bounds",
            "obstacle_collision",
            "robot_history_collision",
            "history_visible",
        } or self.accepted_target is not None:
            raise UnseenPriorInputError("illegal decision must contain one rejection reason")


def _candidate_matches_mother(
    mother: UnseenPriorMother, target: Long40TargetMotion
) -> None:
    source = mother.target_motion
    for field_name in (
        "target_dynamic_object_id",
        "source_recording_id",
        "source_session_id",
        "source_snippet_id",
        "source_object_id",
        "object_type",
        "layout_version",
    ):
        if getattr(target, field_name) != getattr(source, field_name):
            raise UnseenPriorInputError(
                f"transformed_target.{field_name} does not match mother target"
            )
    if target.footprint_spec != source.footprint_spec:
        raise UnseenPriorInputError(
            "transformed_target.footprint_spec does not match mother target"
        )


def _candidate_poses(target: Long40TargetMotion) -> np.ndarray | None:
    arrays = (
        ("positions", (40, 2)),
        ("velocities", (40, 2)),
        ("headings", (40,)),
    )
    for name, shape in arrays:
        array = np.asarray(getattr(target, name))
        if array.dtype != np.dtype(np.float32) or array.shape != shape:
            raise UnseenPriorInputError(
                f"transformed_target.{name} must be float32 with shape {shape}"
            )
        if not np.isfinite(array).all():
            return None
    poses = np.empty((40, 3), dtype=np.float32)
    poses[:, :2] = target.positions
    poses[:, 2] = target.headings
    return poses


def _target_within_grid(
    footprint: CircleFootprint | RectangleFootprint,
    poses: np.ndarray,
    grid: GridSpec,
) -> bool:
    x_min, x_max, y_min, y_max = grid_bounds(grid)
    for pose in poses:
        footprint_x_min, footprint_x_max, footprint_y_min, footprint_y_max = (
            footprint_aabb(footprint, pose)
        )
        if (
            footprint_x_min < x_min
            or footprint_x_max >= x_max
            or footprint_y_min < y_min
            or footprint_y_max >= y_max
        ):
            return False
    return True


def _history_visibility_masks(mother: UnseenPriorMother) -> np.ndarray:
    occupancy = mother.static_occupancy | mother.occluder_occupancy
    context_footprints = {
        obstacle.object_id: footprint_from_spec(obstacle.footprint_spec)
        for obstacle in mother.context_obstacles
    }
    masks: list[np.ndarray] = []
    for index in range(mother.grid.history_steps):
        occupied = occupancy.copy()
        for obstacle in mother.context_obstacles:
            occupied |= rasterize_footprint(
                context_footprints[obstacle.object_id], obstacle.poses[index], mother.grid
            )
        if mother.sensor_config is None:
            visible = raycast_visibility(
                occupied, mother.grid, sensor_pose=mother.robot_history[index]
            )
        else:
            visible = build_structural_visibility(
                occupied,
                mother.grid,
                sensor_pose=mother.robot_history[index],
                blind_spot=mother.sensor_config,
            )
        masks.append(np.asarray(visible, dtype=np.bool_))
    return np.stack(masks, axis=0)


def prepare_unseen_candidate_context(
    mother: UnseenPriorMother,
) -> UnseenPriorCandidateContext:
    """Precompute angle-invariant visibility masks for one mother."""

    if not isinstance(mother, UnseenPriorMother):
        raise UnseenPriorInputError("mother must be an UnseenPriorMother")
    return UnseenPriorCandidateContext(
        mother_id=mother.mother_id,
        target_dynamic_object_id=mother.target_motion.target_dynamic_object_id,
        grid_shape=(mother.grid.history_steps, mother.grid.height, mother.grid.width),
        target_footprint=_target_footprint(mother.target_motion),
        history_visibility_masks=_history_visibility_masks(mother),
    )


def _history_target_visible(
    mother: UnseenPriorMother,
    *,
    footprint: CircleFootprint | RectangleFootprint,
    poses: np.ndarray,
    visibility_masks: np.ndarray | None = None,
) -> bool:
    history_visibility = np.empty(mother.grid.history_steps, dtype=bool)
    masks = (
        _history_visibility_masks(mother)
        if visibility_masks is None
        else np.asarray(visibility_masks)
    )
    if masks.shape != (
        mother.grid.history_steps,
        mother.grid.height,
        mother.grid.width,
    ) or masks.dtype != np.bool_:
        raise UnseenPriorInputError("prepared history visibility masks are invalid")
    for index in range(mother.grid.history_steps):
        history_visibility[index] = footprint_visibility_sequence(
            footprint,
            poses[index : index + 1],
            masks[index],
            mother.grid,
        )[0]
    return bool(history_visibility.any())


def evaluate_candidate(
    mother: UnseenPriorMother,
    *,
    transformed_target: Long40TargetMotion,
    prepared_context: UnseenPriorCandidateContext | None = None,
) -> UnseenPriorCandidateDecision:
    """Apply the four M2 legality gates to one transformed Long40 target."""

    if not isinstance(mother, UnseenPriorMother):
        raise UnseenPriorInputError("mother must be an UnseenPriorMother")
    if not isinstance(transformed_target, Long40TargetMotion):
        raise UnseenPriorInputError("transformed_target must be a Long40TargetMotion")
    _candidate_matches_mother(mother, transformed_target)
    if prepared_context is not None:
        if not isinstance(prepared_context, UnseenPriorCandidateContext):
            raise UnseenPriorInputError(
                "prepared_context must be an UnseenPriorCandidateContext"
            )
        expected_shape = (
            mother.grid.history_steps,
            mother.grid.height,
            mother.grid.width,
        )
        if (
            prepared_context.mother_id != mother.mother_id
            or prepared_context.target_dynamic_object_id
            != mother.target_motion.target_dynamic_object_id
            or prepared_context.grid_shape != expected_shape
        ):
            raise UnseenPriorInputError("prepared_context does not belong to mother")
    poses = _candidate_poses(transformed_target)
    if poses is None:
        return UnseenPriorCandidateDecision(
            legal=False,
            rejection_reason="nonfinite_or_out_of_bounds",
            accepted_target=None,
        )
    footprint = (
        _target_footprint(transformed_target)
        if prepared_context is None
        else prepared_context.target_footprint
    )
    if not _target_within_grid(footprint, poses, mother.grid):
        return UnseenPriorCandidateDecision(
            legal=False,
            rejection_reason="nonfinite_or_out_of_bounds",
            accepted_target=None,
        )
    if (
        swept_footprint_intersects_occupancy(
            footprint, poses, mother.static_occupancy, grid=mother.grid
        )
        or swept_footprint_intersects_occupancy(
            footprint, poses, mother.occluder_occupancy, grid=mother.grid
        )
    ):
        return UnseenPriorCandidateDecision(
            legal=False,
            rejection_reason="obstacle_collision",
            accepted_target=None,
        )
    for obstacle in mother.context_obstacles:
        if synchronized_sweeps_intersect(
            footprint,
            poses,
            footprint_from_spec(obstacle.footprint_spec),
            obstacle.poses,
            grid=mother.grid,
        ):
            return UnseenPriorCandidateDecision(
                legal=False,
                rejection_reason="obstacle_collision",
                accepted_target=None,
            )
    if synchronized_sweeps_intersect(
        mother.robot_footprint,
        mother.robot_history,
        footprint,
        poses[: mother.grid.history_steps],
        grid=mother.grid,
    ):
        return UnseenPriorCandidateDecision(
            legal=False,
            rejection_reason="robot_history_collision",
            accepted_target=None,
        )
    if _history_target_visible(
        mother,
        footprint=footprint,
        poses=poses,
        visibility_masks=(
            None
            if prepared_context is None
            else prepared_context.history_visibility_masks
        ),
    ):
        return UnseenPriorCandidateDecision(
            legal=False,
            rejection_reason="history_visible",
            accepted_target=None,
        )
    return UnseenPriorCandidateDecision(
        legal=True,
        rejection_reason=None,
        accepted_target=transformed_target,
    )


@dataclass(frozen=True)
class UnseenPriorConfig:
    """Normalized release configuration; angles are sampled continuously in M3."""

    schema_version: str
    generator_version: str
    contract_version: str
    p_hidden_human: float
    max_attempts_per_mother: int
    max_variants_per_mother: int
    hard_total_sample_cap: int
    manifest_targets: tuple[int, int, int]
    seed: int
    config_digest: str


def validate_long40_base_config(base_config: Mapping[str, object]) -> None:
    """Check that the supplied base configuration declares the Long40 layout."""

    if not isinstance(base_config, Mapping):
        raise UnseenPriorConfigError("base_config must be a mapping")
    if base_config.get("schema_version") != LONG40_SCHEMA_VERSION:
        raise UnseenPriorConfigError(
            f"base_config.schema_version must equal {LONG40_SCHEMA_VERSION!r}"
        )
    bev = base_config.get("bev")
    if not isinstance(bev, Mapping):
        raise UnseenPriorConfigError("base_config.bev must be a mapping")
    for name, expected in (
        ("history_steps", 8),
        ("future_steps", 32),
    ):
        value = _integer(
            bev.get(name),
            name=f"base_config.bev.{name}",
            minimum=1,
            error_type=UnseenPriorConfigError,
        )
        if value != expected:
            raise UnseenPriorConfigError(
                f"base_config.bev.{name} must equal {expected}"
            )
    for name in ("history_dt_s", "future_dt_s"):
        value = _finite_real(
            bev.get(name), name=f"base_config.bev.{name}", error_type=UnseenPriorConfigError
        )
        if value != 0.2:
            raise UnseenPriorConfigError(f"base_config.bev.{name} must equal 0.2")


def normalize_unseen_prior_config(
    raw: Mapping[str, object], *, base_config: Mapping[str, object] | None = None
) -> UnseenPriorConfig:
    """Validate the minimal frozen configuration for regime A."""

    if not isinstance(raw, Mapping):
        raise UnseenPriorConfigError("unseen prior config must be a mapping")
    if set(raw) != _EXPECTED_CONFIG_KEYS:
        raise UnseenPriorConfigError(
            "unseen prior config keys do not match the frozen schema"
        )
    if base_config is not None:
        validate_long40_base_config(base_config)

    schema_version = raw["schema_version"]
    if schema_version != LONG40_SCHEMA_VERSION:
        raise UnseenPriorConfigError(
            f"schema_version must equal {LONG40_SCHEMA_VERSION!r}"
        )
    generator_version = raw["generator_version"]
    if generator_version != UNSEEN_PRIOR_GENERATOR_VERSION:
        raise UnseenPriorConfigError(
            f"generator_version must equal {UNSEEN_PRIOR_GENERATOR_VERSION!r}"
        )
    contract_version = raw["contract_version"]
    if contract_version != UNSEEN_PRIOR_CONTRACT_VERSION:
        raise UnseenPriorConfigError(
            f"contract_version must equal {UNSEEN_PRIOR_CONTRACT_VERSION!r}"
        )

    p_hidden_human = _finite_real(
        raw["p_hidden_human"],
        name="p_hidden_human",
        error_type=UnseenPriorConfigError,
    )
    if p_hidden_human != 0.30:
        raise UnseenPriorConfigError("p_hidden_human must equal 0.30")
    max_attempts_per_mother = _integer(
        raw["max_attempts_per_mother"],
        name="max_attempts_per_mother",
        minimum=1,
        error_type=UnseenPriorConfigError,
    )
    if max_attempts_per_mother != 32:
        raise UnseenPriorConfigError("max_attempts_per_mother must equal 32")
    max_variants_per_mother = _integer(
        raw["max_variants_per_mother"],
        name="max_variants_per_mother",
        minimum=1,
        error_type=UnseenPriorConfigError,
    )
    if max_variants_per_mother != 1:
        raise UnseenPriorConfigError("max_variants_per_mother must equal 1")
    hard_total_sample_cap = _integer(
        raw["hard_total_sample_cap"],
        name="hard_total_sample_cap",
        minimum=1,
        error_type=UnseenPriorConfigError,
    )
    if hard_total_sample_cap != 125000:
        raise UnseenPriorConfigError("hard_total_sample_cap must equal 125000")

    manifest_raw = raw["manifest_targets"]
    if not isinstance(manifest_raw, (list, tuple)) or len(manifest_raw) != 3:
        raise UnseenPriorConfigError("manifest_targets must contain exactly three targets")
    manifest_targets = tuple(
        _integer(
            value,
            name=f"manifest_targets[{index}]",
            minimum=1,
            error_type=UnseenPriorConfigError,
        )
        for index, value in enumerate(manifest_raw)
    )
    if manifest_targets != _MANIFEST_TARGETS:
        raise UnseenPriorConfigError(
            "manifest_targets must equal [50000, 100000, 125000]"
        )
    seed = _integer(
        raw["seed"], name="seed", minimum=0, error_type=UnseenPriorConfigError
    )

    normalized_payload = {
        "schema_version": schema_version,
        "generator_version": generator_version,
        "contract_version": contract_version,
        "p_hidden_human": p_hidden_human,
        "max_attempts_per_mother": max_attempts_per_mother,
        "max_variants_per_mother": max_variants_per_mother,
        "hard_total_sample_cap": hard_total_sample_cap,
        "manifest_targets": list(manifest_targets),
        "seed": seed,
    }
    try:
        digest = config_digest(normalized_payload)
    except (TypeError, ValueError) as exc:
        raise UnseenPriorConfigError(
            "unseen prior config must be finite canonical JSON"
        ) from exc
    return UnseenPriorConfig(
        schema_version=schema_version,
        generator_version=generator_version,
        contract_version=contract_version,
        p_hidden_human=p_hidden_human,
        max_attempts_per_mother=max_attempts_per_mother,
        max_variants_per_mother=max_variants_per_mother,
        hard_total_sample_cap=hard_total_sample_cap,
        manifest_targets=manifest_targets,
        seed=seed,
        config_digest=digest,
    )


_CANDIDATE_REJECTION_REASONS = frozenset(
    {
        "nonfinite_or_out_of_bounds",
        "obstacle_collision",
        "robot_history_collision",
        "history_visible",
    }
)


def _readonly_rejection_counts(value: Mapping[str, int]) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or not set(value).issubset(
        _CANDIDATE_REJECTION_REASONS
    ):
        raise UnseenPriorInputError("rejection_reason_counts has invalid keys")
    normalized: dict[str, int] = {}
    for reason, count in value.items():
        normalized[reason] = _integer(
            count,
            name=f"rejection_reason_counts[{reason!r}]",
            minimum=0,
            error_type=UnseenPriorInputError,
        )
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class UnseenPriorRealization:
    """One M3 scene realization before M4 creates model-safe and oracle views."""

    mother_id: str
    split: str
    grid: GridSpec
    robot_footprint: CircleFootprint | RectangleFootprint
    robot_history: np.ndarray
    static_occupancy: np.ndarray
    occluder_occupancy: np.ndarray
    context_obstacles: tuple[UnseenPriorContextObstacle, ...]
    sensor_config: StructuralBlindSpot | None
    target_motion: Long40TargetMotion | None


@dataclass(frozen=True)
class UnseenPriorSamplingProvenance:
    """Audit-only branch and rejection information for one mother event."""

    mother_id: str
    split: str
    seed: int
    presence_branch: str
    outcome: str
    attempted_angle_count: int
    selected_angle_rad: float | None
    rejection_reason_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.mother_id, name="mother_id")
        _require_nonempty_string(self.split, name="split")
        _integer(self.seed, name="seed", minimum=0, error_type=UnseenPriorInputError)
        if self.presence_branch not in {"empty", "present"}:
            raise UnseenPriorInputError("presence_branch must be empty or present")
        if self.outcome not in {"empty", "present", "no_legal_angle"}:
            raise UnseenPriorInputError("outcome is invalid")
        attempts = _integer(
            self.attempted_angle_count,
            name="attempted_angle_count",
            minimum=0,
            error_type=UnseenPriorInputError,
        )
        object.__setattr__(self, "attempted_angle_count", attempts)
        if self.selected_angle_rad is not None:
            angle = _finite_real(
                self.selected_angle_rad,
                name="selected_angle_rad",
                error_type=UnseenPriorInputError,
            )
            if not -np.pi <= angle < np.pi:
                raise UnseenPriorInputError("selected_angle_rad must lie in [-pi, pi)")
            object.__setattr__(self, "selected_angle_rad", angle)
        object.__setattr__(
            self,
            "rejection_reason_counts",
            _readonly_rejection_counts(self.rejection_reason_counts),
        )
        if self.outcome == "empty":
            if (
                self.presence_branch != "empty"
                or attempts != 0
                or self.selected_angle_rad is not None
                or self.rejection_reason_counts
            ):
                raise UnseenPriorInputError("empty provenance must not contain attempts")
        elif self.outcome == "present":
            if (
                self.presence_branch != "present"
                or attempts < 1
                or self.selected_angle_rad is None
            ):
                raise UnseenPriorInputError("present provenance is inconsistent")
        elif (
            self.presence_branch != "present"
            or attempts != 32
            or self.selected_angle_rad is not None
        ):
            raise UnseenPriorInputError("no_legal_angle provenance is inconsistent")


@dataclass(frozen=True)
class UnseenPriorMotherResult:
    """Zero-or-one M3 realization and its audit provenance."""

    realization: UnseenPriorRealization | None
    provenance: UnseenPriorSamplingProvenance

    def __post_init__(self) -> None:
        if self.provenance.outcome == "no_legal_angle":
            if self.realization is not None:
                raise UnseenPriorInputError("no_legal_angle must not emit a realization")
            return
        if self.realization is None:
            raise UnseenPriorInputError("successful branch must emit one realization")
        if (
            self.realization.mother_id != self.provenance.mother_id
            or self.realization.split != self.provenance.split
        ):
            raise UnseenPriorInputError("realization/provenance mother identity mismatch")
        if self.provenance.outcome == "empty":
            if self.realization.target_motion is not None:
                raise UnseenPriorInputError("empty realization must remove the target")
        elif self.realization.target_motion is None:
            raise UnseenPriorInputError("present realization must contain one target")


@dataclass(frozen=True)
class UnseenPriorRun:
    """Deterministic M3 collection plus aggregate deficits and rejections."""

    results: tuple[UnseenPriorMotherResult, ...]
    realizations: tuple[UnseenPriorRealization, ...]
    deficit_count: int
    rejection_reason_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.results, tuple) or not isinstance(self.realizations, tuple):
            raise UnseenPriorInputError("run outputs must be tuples")
        emitted = tuple(
            result.realization
            for result in self.results
            if result.realization is not None
        )
        if len(emitted) != len(self.realizations) or any(
            left is not right for left, right in zip(emitted, self.realizations)
        ):
            raise UnseenPriorInputError("run realizations must preserve result order")
        deficit_count = _integer(
            self.deficit_count,
            name="deficit_count",
            minimum=0,
            error_type=UnseenPriorInputError,
        )
        if deficit_count != sum(
            result.provenance.outcome == "no_legal_angle" for result in self.results
        ):
            raise UnseenPriorInputError("deficit_count does not match mother results")
        object.__setattr__(self, "deficit_count", deficit_count)
        object.__setattr__(
            self,
            "rejection_reason_counts",
            _readonly_rejection_counts(self.rejection_reason_counts),
        )


def _stable_mother_rng(
    mother: UnseenPriorMother, *, seed: int
) -> np.random.Generator:
    payload = json.dumps(
        {
            "namespace": "sop05/unseen-history-prior/m3/v1",
            "seed": seed,
            "split": mother.split,
            "mother_id": mother.mother_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    entropy = int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")
    return np.random.default_rng(entropy)


def _validate_sampling_inputs(
    mother: UnseenPriorMother,
    config: UnseenPriorConfig,
    seed: int,
) -> int:
    if not isinstance(mother, UnseenPriorMother):
        raise UnseenPriorInputError("mother must be an UnseenPriorMother")
    if not isinstance(config, UnseenPriorConfig):
        raise UnseenPriorConfigError("config must be an UnseenPriorConfig")
    normalized_seed = _integer(
        seed, name="seed", minimum=0, error_type=UnseenPriorConfigError
    )
    if normalized_seed != config.seed:
        raise UnseenPriorConfigError("seed must equal config.seed")
    return normalized_seed


def _realization_from_mother(
    mother: UnseenPriorMother, *, target_motion: Long40TargetMotion | None
) -> UnseenPriorRealization:
    return UnseenPriorRealization(
        mother_id=mother.mother_id,
        split=mother.split,
        grid=mother.grid,
        robot_footprint=mother.robot_footprint,
        robot_history=mother.robot_history,
        static_occupancy=mother.static_occupancy,
        occluder_occupancy=mother.occluder_occupancy,
        context_obstacles=mother.context_obstacles,
        sensor_config=mother.sensor_config,
        target_motion=target_motion,
    )


def generate_unseen_prior_mother(
    mother: UnseenPriorMother, *, config: UnseenPriorConfig, seed: int
) -> UnseenPriorMotherResult:
    """Sample one target-empty or first-legal target-present realization."""

    normalized_seed = _validate_sampling_inputs(mother, config, seed)
    rng = _stable_mother_rng(mother, seed=normalized_seed)
    presence_draw = float(rng.random())
    if not np.isfinite(presence_draw) or not 0.0 <= presence_draw < 1.0:
        raise UnseenPriorInputError("RNG presence draw must lie in [0, 1)")
    if presence_draw >= config.p_hidden_human:
        return UnseenPriorMotherResult(
            realization=_realization_from_mother(mother, target_motion=None),
            provenance=UnseenPriorSamplingProvenance(
                mother_id=mother.mother_id,
                split=mother.split,
                seed=normalized_seed,
                presence_branch="empty",
                outcome="empty",
                attempted_angle_count=0,
                selected_angle_rad=None,
                rejection_reason_counts={},
            ),
        )

    rejection_counts: dict[str, int] = {}
    prepared_context = prepare_unseen_candidate_context(mother)
    for attempt in range(1, config.max_attempts_per_mother + 1):
        angle = float(rng.uniform(-np.pi, np.pi))
        if not np.isfinite(angle) or not -np.pi <= angle < np.pi:
            raise UnseenPriorInputError("RNG angle draw must lie in [-pi, pi)")
        transformed_target = transform_long40_target(
            mother.target_motion, angle_rad=angle
        )
        decision = evaluate_candidate(
            mother,
            transformed_target=transformed_target,
            prepared_context=prepared_context,
        )
        if decision.legal:
            return UnseenPriorMotherResult(
                realization=_realization_from_mother(
                    mother, target_motion=decision.accepted_target
                ),
                provenance=UnseenPriorSamplingProvenance(
                    mother_id=mother.mother_id,
                    split=mother.split,
                    seed=normalized_seed,
                    presence_branch="present",
                    outcome="present",
                    attempted_angle_count=attempt,
                    selected_angle_rad=angle,
                    rejection_reason_counts=rejection_counts,
                ),
            )
        reason = decision.rejection_reason
        if reason not in _CANDIDATE_REJECTION_REASONS:
            raise RuntimeError("candidate decision returned an unknown rejection reason")
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    return UnseenPriorMotherResult(
        realization=None,
        provenance=UnseenPriorSamplingProvenance(
            mother_id=mother.mother_id,
            split=mother.split,
            seed=normalized_seed,
            presence_branch="present",
            outcome="no_legal_angle",
            attempted_angle_count=config.max_attempts_per_mother,
            selected_angle_rad=None,
            rejection_reason_counts=rejection_counts,
        ),
    )


def run_unseen_prior(
    mothers: tuple[UnseenPriorMother, ...] | list[UnseenPriorMother],
    *,
    config: UnseenPriorConfig,
    seed: int,
) -> UnseenPriorRun:
    """Generate deterministic, independent M3 outcomes in caller-supplied order."""

    if not isinstance(mothers, (tuple, list)) or not all(
        isinstance(mother, UnseenPriorMother) for mother in mothers
    ):
        raise UnseenPriorInputError("mothers must be a sequence of UnseenPriorMother")
    if not isinstance(config, UnseenPriorConfig):
        raise UnseenPriorConfigError("config must be an UnseenPriorConfig")
    normalized_seed = (
        _validate_sampling_inputs(mothers[0], config, seed)
        if mothers
        else _integer(seed, name="seed", minimum=0, error_type=UnseenPriorConfigError)
    )
    if normalized_seed != config.seed:
        raise UnseenPriorConfigError("seed must equal config.seed")
    identities = tuple((mother.split, mother.mother_id) for mother in mothers)
    if len(set(identities)) != len(identities):
        raise UnseenPriorInputError("mothers must have unique (split, mother_id) pairs")
    results = tuple(
        generate_unseen_prior_mother(mother, config=config, seed=normalized_seed)
        for mother in mothers
    )
    rejection_counts: dict[str, int] = {}
    for result in results:
        for reason, count in result.provenance.rejection_reason_counts.items():
            rejection_counts[reason] = rejection_counts.get(reason, 0) + count
    realizations = tuple(
        result.realization for result in results if result.realization is not None
    )
    return UnseenPriorRun(
        results=results,
        realizations=realizations,
        deficit_count=sum(
            result.provenance.outcome == "no_legal_angle" for result in results
        ),
        rejection_reason_counts=rejection_counts,
    )
