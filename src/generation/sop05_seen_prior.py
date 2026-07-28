"""SOP05 Regime-B input, continuous-prior, and future-rotation primitives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType

import numpy as np
import yaml

from src.contracts import GridSpec
from src.geometry import (
    CircleFootprint,
    Footprint,
    RectangleFootprint,
    footprint_aabb,
    grid_bounds,
    wrap_angle,
)

from .occluder_sampler import (
    swept_footprint_intersects_occupancy,
    synchronized_sweeps_intersect,
)
SEEN_PRIOR_CONFIG_VERSION = "sop05_seen_prior_config_v1"
SEEN_PRIOR_GENERATOR_VERSION = "sop05_seen_occluded_scenario_v1"
SEEN_PRIOR_ANGLE_PRIOR_VERSION = "truncated_normal_zero_mean_v1"
SEEN_PRIOR_SAMPLING_VERSION = "sha256_pcg64_per_mother_v1"
SEEN_PRIOR_TRANSFORM_VERSION = "future32_rigid_rotation_about_history7_v1"
SEEN_PRIOR_M2_ENVIRONMENT_GATE_VERSION = "future32_environment_legality_v1"
SEEN_PRIOR_M2_REJECTION_REASONS = (
    "future_nonfinite",
    "future_out_of_bounds",
    "future_static_collision",
    "future_occluder_collision",
    "future_context_collision",
)
SEEN_PRIOR_M3_SELECTION_VERSION = "first_legal_future_v1"
SEEN_PRIOR_M3_FAILURE_REASON = "no_legal_future"
_ANGLE_OUT_OF_RANGE_REASON = "angle_out_of_range"

_SCHEMA_VERSION = "4.0.0"
_HISTORY_REGIME = "seen_then_occluded"
_HISTORY_STEPS = 8
_CURRENT_INDEX = 7
_FUTURE_STEPS = 32
_DT_S = 0.2
_ANGLE_PRIOR_KIND = "truncated_normal"
_ANGLE_MEAN_RAD = 0.0
_ANGLE_SIGMA_RAD = math.pi / 12.0
_MIN_ANGLE_RAD = -math.pi
_MAX_ANGLE_RAD = math.pi
_SEED_NAMESPACE = "sop05/seen-prior/continuous/v1"
_MAX_ATTEMPTS_PER_MOTHER = 32
_MAX_VARIANTS_PER_MOTHER = 1


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader which refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found unhashable key {key!r}",
                key_node.start_mark,
            ) from exc
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class SeenPriorConfig:
    """Validated configuration containing only M1 method parameters."""

    schema_version: str
    required_history_regime: str
    history_steps: int
    current_index: int
    future_steps: int
    dt_s: float
    angle_prior_kind: str
    mean_rad: float
    sigma_rad: float
    min_rad_inclusive: float
    max_rad_exclusive: float
    seed_namespace: str
    max_attempts_per_mother: int
    max_variants_per_mother: int
    digest: str


@dataclass(frozen=True)
class SeenPriorSource:
    """One authenticated Long40 Regime-B mother used by M1 and later stages."""

    mother_id: str
    split: str
    source_collection_identity: str
    history_regime: str
    target_history_poses: np.ndarray
    target_current_pose: np.ndarray
    target_future_poses: np.ndarray
    target_visibility_history: np.ndarray


@dataclass(frozen=True)
class SeenPriorFuture:
    """Future-only rotation result; history and current are copied byte-for-byte."""

    history_poses: np.ndarray
    current_pose: np.ndarray
    future_poses: np.ndarray
    theta_rad: float


@dataclass(frozen=True)
class SeenPriorContextSweep:
    """One protected object's current-plus-future synchronized trajectory."""

    context_object_id: str
    footprint: Footprint
    poses: np.ndarray


@dataclass(frozen=True)
class SeenPriorEnvironment:
    """The non-robot scene geometry used exclusively by the M2 legality gate."""

    grid: GridSpec
    target_footprint: Footprint
    static_occupancy: np.ndarray
    occluder_occupancy: np.ndarray
    context_sweeps: tuple[SeenPriorContextSweep, ...]


@dataclass(frozen=True)
class SeenPriorEnvironmentValidation:
    """Focused accept/reject result for one candidate's environment legality."""

    accepted: bool
    reason: str | None


@dataclass(frozen=True)
class SeenPriorResult:
    """The one SOP05-selected future before SOP7 computes risk labels."""

    mother_id: str
    history_poses: np.ndarray
    current_pose: np.ndarray
    future_poses: np.ndarray
    theta_rad: float
    accepted_attempt: int

    def __post_init__(self) -> None:
        _require_identifier(self.mother_id, name="mother_id")
        _require_floating_pose_array(
            self.history_poses,
            name="history_poses",
            shape=(_HISTORY_STEPS, 3),
            finite=True,
        )
        current = _require_floating_pose_array(
            self.current_pose,
            name="current_pose",
            shape=(3,),
            finite=True,
        )
        history = self.history_poses
        if current.dtype != history.dtype or current.tobytes(order="C") != history[
            _CURRENT_INDEX
        ].tobytes(order="C"):
            raise ValueError("current_pose must be byte-equal to history index 7")
        _require_floating_pose_array(
            self.future_poses,
            name="future_poses",
            shape=(_FUTURE_STEPS, 3),
            finite=True,
        )
        theta = _require_finite_real(self.theta_rad, name="theta_rad")
        if not _MIN_ANGLE_RAD <= theta < _MAX_ANGLE_RAD:
            raise ValueError("theta_rad must be in [-pi, pi)")
        if (
            isinstance(self.accepted_attempt, bool)
            or not isinstance(self.accepted_attempt, Integral)
            or not 1 <= int(self.accepted_attempt) <= _MAX_ATTEMPTS_PER_MOTHER
        ):
            raise ValueError("accepted_attempt must be in [1, 32]")


@dataclass(frozen=True)
class SeenPriorFailure:
    """A bounded M3 failure with no rejected future retained or published."""

    mother_id: str
    reason: str
    attempts: int
    rejection_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.reason != SEEN_PRIOR_M3_FAILURE_REASON:
            raise ValueError("SeenPriorFailure reason must be no_legal_future")
        if self.attempts != _MAX_ATTEMPTS_PER_MOTHER:
            raise ValueError("SeenPriorFailure attempts must be 32")
        if not isinstance(self.rejection_counts, Mapping):
            raise ValueError("rejection_counts must be a mapping")
        normalized: dict[str, int] = {}
        for reason, count in self.rejection_counts.items():
            if reason not in (*SEEN_PRIOR_M2_REJECTION_REASONS, _ANGLE_OUT_OF_RANGE_REASON):
                raise ValueError("rejection_counts reason is invalid")
            if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
                raise ValueError("rejection_counts must contain positive integer counts")
            normalized[reason] = int(count)
        if sum(normalized.values()) != self.attempts:
            raise ValueError("rejection_counts must account for every attempt")
        object.__setattr__(
            self,
            "rejection_counts",
            MappingProxyType(dict(sorted(normalized.items()))),
        )


def _canonical_sha256(value: object, *, label: str) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError(f"{label} must be finite canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_mapping(
    value: object, *, name: str, keys: set[str]
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} keys are invalid")
    return value


def _require_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_exact_string(value: object, *, name: str, expected: str) -> str:
    result = _require_string(value, name=name)
    if result != expected:
        raise ValueError(f"{name} must be {expected!r}")
    return result


def _require_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _require_exact_integer(value: object, *, name: str, expected: int) -> int:
    result = _require_integer(value, name=name)
    if result != expected:
        raise ValueError(f"{name} must be {expected}")
    return result


def _require_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _require_exact_real(value: object, *, name: str, expected: float) -> float:
    result = _require_finite_real(value, name=name)
    if result != expected:
        raise ValueError(f"{name} must be {expected!r}")
    return result


def _load_seen_prior_payload(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.load(handle, Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid seen-prior config: {path}") from exc
    return _require_mapping(
        payload,
        name="seen-prior config",
        keys={
            "schema_version",
            "required_history_regime",
            "trajectory",
            "angle_prior",
            "sampling",
        },
    )


def load_seen_prior_config(path: str | Path) -> SeenPriorConfig:
    """Load the fixed Long40 continuous-prior configuration from ``path``."""

    raw = _load_seen_prior_payload(Path(path))
    schema_version = _require_exact_string(
        raw["schema_version"], name="schema_version", expected=_SCHEMA_VERSION
    )
    required_history_regime = _require_exact_string(
        raw["required_history_regime"],
        name="required_history_regime",
        expected=_HISTORY_REGIME,
    )
    trajectory = _require_mapping(
        raw["trajectory"],
        name="trajectory",
        keys={"history_steps", "current_index", "future_steps", "dt_s"},
    )
    history_steps = _require_exact_integer(
        trajectory["history_steps"],
        name="trajectory.history_steps",
        expected=_HISTORY_STEPS,
    )
    current_index = _require_exact_integer(
        trajectory["current_index"],
        name="trajectory.current_index",
        expected=_CURRENT_INDEX,
    )
    future_steps = _require_exact_integer(
        trajectory["future_steps"],
        name="trajectory.future_steps",
        expected=_FUTURE_STEPS,
    )
    dt_s = _require_exact_real(trajectory["dt_s"], name="trajectory.dt_s", expected=_DT_S)

    angle_prior = _require_mapping(
        raw["angle_prior"],
        name="angle_prior",
        keys={
            "kind",
            "mean_rad",
            "sigma_rad",
            "min_rad_inclusive",
            "max_rad_exclusive",
        },
    )
    angle_prior_kind = _require_exact_string(
        angle_prior["kind"], name="angle_prior.kind", expected=_ANGLE_PRIOR_KIND
    )
    mean_rad = _require_exact_real(
        angle_prior["mean_rad"], name="angle_prior.mean_rad", expected=_ANGLE_MEAN_RAD
    )
    sigma_rad = _require_exact_real(
        angle_prior["sigma_rad"],
        name="angle_prior.sigma_rad",
        expected=_ANGLE_SIGMA_RAD,
    )
    min_rad_inclusive = _require_exact_real(
        angle_prior["min_rad_inclusive"],
        name="angle_prior.min_rad_inclusive",
        expected=_MIN_ANGLE_RAD,
    )
    max_rad_exclusive = _require_exact_real(
        angle_prior["max_rad_exclusive"],
        name="angle_prior.max_rad_exclusive",
        expected=_MAX_ANGLE_RAD,
    )

    sampling = _require_mapping(
        raw["sampling"],
        name="sampling",
        keys={"seed_namespace", "max_attempts_per_mother", "max_variants_per_mother"},
    )
    seed_namespace = _require_exact_string(
        sampling["seed_namespace"],
        name="sampling.seed_namespace",
        expected=_SEED_NAMESPACE,
    )
    max_attempts_per_mother = _require_exact_integer(
        sampling["max_attempts_per_mother"],
        name="sampling.max_attempts_per_mother",
        expected=_MAX_ATTEMPTS_PER_MOTHER,
    )
    max_variants_per_mother = _require_exact_integer(
        sampling["max_variants_per_mother"],
        name="sampling.max_variants_per_mother",
        expected=_MAX_VARIANTS_PER_MOTHER,
    )
    return SeenPriorConfig(
        schema_version=schema_version,
        required_history_regime=required_history_regime,
        history_steps=history_steps,
        current_index=current_index,
        future_steps=future_steps,
        dt_s=dt_s,
        angle_prior_kind=angle_prior_kind,
        mean_rad=mean_rad,
        sigma_rad=sigma_rad,
        min_rad_inclusive=min_rad_inclusive,
        max_rad_exclusive=max_rad_exclusive,
        seed_namespace=seed_namespace,
        max_attempts_per_mother=max_attempts_per_mother,
        max_variants_per_mother=max_variants_per_mother,
        digest=_canonical_sha256(raw, label="seen-prior config"),
    )


def _require_identifier(value: object, *, name: str) -> str:
    identifier = _require_string(value, name=name)
    try:
        identifier.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be UTF-8 encodable") from exc
    return identifier


def _require_pose_array(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite floating-point array")
    return value


def validate_seen_prior_source(source: SeenPriorSource, config: SeenPriorConfig) -> None:
    """Validate M1 eligibility and the fixed Long40 source-array boundary."""

    if not isinstance(source, SeenPriorSource):
        raise TypeError("source must be a SeenPriorSource")
    if not isinstance(config, SeenPriorConfig):
        raise TypeError("config must be a SeenPriorConfig")
    _require_identifier(source.mother_id, name="source.mother_id")
    _require_identifier(source.split, name="source.split")
    _require_identifier(
        source.source_collection_identity,
        name="source.source_collection_identity",
    )
    if source.history_regime != config.required_history_regime:
        raise ValueError("source.history_regime is not authenticated seen_then_occluded")
    history = _require_pose_array(
        source.target_history_poses,
        name="source.target_history_poses",
        shape=(config.history_steps, 3),
    )
    current = _require_pose_array(
        source.target_current_pose,
        name="source.target_current_pose",
        shape=(3,),
    )
    future = _require_pose_array(
        source.target_future_poses,
        name="source.target_future_poses",
        shape=(config.future_steps, 3),
    )
    visibility = source.target_visibility_history
    if (
        not isinstance(visibility, np.ndarray)
        or visibility.shape != (config.history_steps,)
        or visibility.dtype != np.bool_
    ):
        raise ValueError("source.target_visibility_history must be boolean [8]")
    if not bool(visibility[0]):
        raise ValueError("source target must be visible at history index 0")
    if current.dtype != history.dtype or current.tobytes(order="C") != history[
        config.current_index
    ].tobytes(order="C"):
        raise ValueError("source.target_current_pose must be byte-equal to history index 7")
    if future.dtype != history.dtype:
        raise ValueError("source target pose arrays must share one dtype")


def _seed_component(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(4, byteorder="big", signed=False) + encoded


def seen_prior_mother_seed(
    config: SeenPriorConfig,
    *,
    dataset_seed: int,
    split: str,
    source_collection_identity: str,
    mother_id: str,
) -> int:
    """Return the prescribed stable 64-bit NumPy seed for one mother event."""

    if not isinstance(config, SeenPriorConfig):
        raise TypeError("config must be a SeenPriorConfig")
    if isinstance(dataset_seed, bool) or not isinstance(dataset_seed, Integral):
        raise ValueError("dataset_seed must be an integer")
    components = (
        config.seed_namespace,
        str(int(dataset_seed)),
        _require_identifier(split, name="split"),
        _require_identifier(source_collection_identity, name="source_collection_identity"),
        _require_identifier(mother_id, name="mother_id"),
    )
    payload = b"".join(_seed_component(component) for component in components)
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def make_seen_prior_rng(
    config: SeenPriorConfig,
    *,
    dataset_seed: int,
    split: str,
    source_collection_identity: str,
    mother_id: str,
) -> np.random.Generator:
    """Create the per-mother generator independent of order or worker count."""

    return np.random.default_rng(
        seen_prior_mother_seed(
            config,
            dataset_seed=dataset_seed,
            split=split,
            source_collection_identity=source_collection_identity,
            mother_id=mother_id,
        )
    )


def draw_seen_prior_angle_attempts(
    config: SeenPriorConfig,
    *,
    dataset_seed: int,
    split: str,
    source_collection_identity: str,
    mother_id: str,
    attempts: int | None = None,
) -> tuple[float | None, ...]:
    """Draw bounded attempt slots; ``None`` is an out-of-range consumed draw."""

    if not isinstance(config, SeenPriorConfig):
        raise TypeError("config must be a SeenPriorConfig")
    attempt_count = config.max_attempts_per_mother if attempts is None else attempts
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, Integral)
        or not 0 <= int(attempt_count) <= config.max_attempts_per_mother
    ):
        raise ValueError("attempts must be in [0, max_attempts_per_mother]")
    rng = make_seen_prior_rng(
        config,
        dataset_seed=dataset_seed,
        split=split,
        source_collection_identity=source_collection_identity,
        mother_id=mother_id,
    )
    values: list[float | None] = []
    for _ in range(int(attempt_count)):
        draw = float(rng.normal(loc=config.mean_rad, scale=config.sigma_rad))
        values.append(
            draw if config.min_rad_inclusive <= draw < config.max_rad_exclusive else None
        )
    return tuple(values)


def _output_dtype_with_half_open_yaw(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        upper = np.nextafter(dtype.type(math.pi), dtype.type(-math.inf))
        lower = dtype.type(-math.pi)
        np.clip(result, lower, upper, out=result)
    return result


def transform_seen_prior_future(
    source: SeenPriorSource,
    config: SeenPriorConfig,
    *,
    theta_rad: float,
) -> SeenPriorFuture:
    """Rotate only source future poses about history index 7 without smoothing."""

    validate_seen_prior_source(source, config)
    theta = _require_finite_real(theta_rad, name="theta_rad")
    if not config.min_rad_inclusive <= theta < config.max_rad_exclusive:
        raise ValueError("theta_rad must be in [-pi, pi)")
    history = source.target_history_poses
    future = source.target_future_poses
    pivot = history[config.current_index, :2].astype(np.float64, copy=False)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    transformed = np.empty(future.shape, dtype=future.dtype, order="C")
    transformed[:, :2] = (
        (future[:, :2].astype(np.float64, copy=False) - pivot) @ rotation.T + pivot
    ).astype(future.dtype)
    transformed[:, 2] = _output_dtype_with_half_open_yaw(
        np.asarray(wrap_angle(future[:, 2].astype(np.float64, copy=False) + theta)),
        dtype=future.dtype,
    )
    return SeenPriorFuture(
        history_poses=np.array(history, dtype=history.dtype, order="C", copy=True),
        current_pose=np.array(
            source.target_current_pose,
            dtype=source.target_current_pose.dtype,
            order="C",
            copy=True,
        ),
        future_poses=transformed,
        theta_rad=theta,
    )


def _require_supported_footprint(value: object, *, name: str) -> Footprint:
    if not isinstance(value, (CircleFootprint, RectangleFootprint)):
        raise TypeError(f"{name} must be a CircleFootprint or RectangleFootprint")
    return value


def _require_floating_pose_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    finite: bool,
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype.kind != "f":
        raise ValueError(f"{name} must be a floating-point array")
    if finite and not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _require_occupancy(
    value: object,
    *,
    name: str,
    grid: GridSpec,
) -> np.ndarray:
    occupancy = np.asarray(value)
    if (
        occupancy.shape != (grid.height, grid.width)
        or occupancy.dtype.kind not in "biuf"
    ):
        raise ValueError(f"{name} must be a numeric grid-shaped array")
    if occupancy.dtype.kind in "iuf" and not np.isfinite(occupancy).all():
        raise ValueError(f"{name} must be finite")
    return occupancy


def _validate_seen_prior_environment(environment: SeenPriorEnvironment) -> None:
    if not isinstance(environment, SeenPriorEnvironment):
        raise TypeError("environment must be a SeenPriorEnvironment")
    if not isinstance(environment.grid, GridSpec):
        raise TypeError("environment.grid must be a GridSpec")
    grid_bounds(environment.grid)
    if (
        environment.grid.history_steps != _HISTORY_STEPS
        or environment.grid.future_steps != _FUTURE_STEPS
    ):
        raise ValueError("environment.grid must use the Long40 layout")
    _require_supported_footprint(
        environment.target_footprint,
        name="environment.target_footprint",
    )
    _require_occupancy(
        environment.static_occupancy,
        name="environment.static_occupancy",
        grid=environment.grid,
    )
    _require_occupancy(
        environment.occluder_occupancy,
        name="environment.occluder_occupancy",
        grid=environment.grid,
    )
    if not isinstance(environment.context_sweeps, tuple):
        raise ValueError("environment.context_sweeps must be a tuple")
    for context in environment.context_sweeps:
        if not isinstance(context, SeenPriorContextSweep):
            raise TypeError("context_sweeps must contain SeenPriorContextSweep")
        _require_identifier(context.context_object_id, name="context.context_object_id")
        _require_supported_footprint(context.footprint, name="context.footprint")
        _require_floating_pose_array(
            context.poses,
            name="context.poses",
            shape=(_FUTURE_STEPS + 1, 3),
            finite=True,
        )


def _future_is_in_grid(
    future_poses: np.ndarray,
    *,
    footprint: Footprint,
    grid: GridSpec,
) -> bool:
    x_min, x_max, y_min, y_max = grid_bounds(grid)
    for pose in future_poses:
        pose_x_min, pose_x_max, pose_y_min, pose_y_max = footprint_aabb(
            footprint,
            pose,
        )
        if (
            pose_x_min < x_min
            or pose_x_max >= x_max
            or pose_y_min < y_min
            or pose_y_max >= y_max
        ):
            return False
    return True


def validate_seen_prior_future_environment(
    candidate: SeenPriorFuture,
    *,
    environment: SeenPriorEnvironment,
) -> SeenPriorEnvironmentValidation:
    """Accept only a finite, in-bounds future free of non-robot geometry contact."""

    if not isinstance(candidate, SeenPriorFuture):
        raise TypeError("candidate must be a SeenPriorFuture")
    _validate_seen_prior_environment(environment)
    current = _require_floating_pose_array(
        candidate.current_pose,
        name="candidate.current_pose",
        shape=(3,),
        finite=True,
    )
    future = _require_floating_pose_array(
        candidate.future_poses,
        name="candidate.future_poses",
        shape=(_FUTURE_STEPS, 3),
        finite=False,
    )
    if not np.isfinite(future).all():
        return SeenPriorEnvironmentValidation(False, "future_nonfinite")
    if not _future_is_in_grid(
        future,
        footprint=environment.target_footprint,
        grid=environment.grid,
    ):
        return SeenPriorEnvironmentValidation(False, "future_out_of_bounds")

    target_sweep = np.vstack((current, future))
    if swept_footprint_intersects_occupancy(
        environment.target_footprint,
        target_sweep,
        environment.static_occupancy,
        grid=environment.grid,
    ):
        return SeenPriorEnvironmentValidation(False, "future_static_collision")
    if swept_footprint_intersects_occupancy(
        environment.target_footprint,
        target_sweep,
        environment.occluder_occupancy,
        grid=environment.grid,
    ):
        return SeenPriorEnvironmentValidation(False, "future_occluder_collision")
    for context in environment.context_sweeps:
        if synchronized_sweeps_intersect(
            environment.target_footprint,
            target_sweep,
            context.footprint,
            context.poses,
            grid=environment.grid,
        ):
            return SeenPriorEnvironmentValidation(False, "future_context_collision")
    return SeenPriorEnvironmentValidation(True, None)


def generate_seen_prior(
    source: SeenPriorSource,
    environment: SeenPriorEnvironment,
    config: SeenPriorConfig,
    dataset_seed: int,
) -> SeenPriorResult | SeenPriorFailure:
    """Select the first environment-legal future for the SOP05 scenario handoff."""

    validate_seen_prior_source(source, config)
    _validate_seen_prior_environment(environment)
    angles = draw_seen_prior_angle_attempts(
        config,
        dataset_seed=dataset_seed,
        split=source.split,
        source_collection_identity=source.source_collection_identity,
        mother_id=source.mother_id,
    )
    if len(angles) != config.max_attempts_per_mother:
        raise RuntimeError("M1 angle stream must contain exactly 32 attempts")
    rejection_counts: dict[str, int] = {}
    for attempt, theta in enumerate(angles, start=1):
        if theta is None:
            rejection_counts[_ANGLE_OUT_OF_RANGE_REASON] = (
                rejection_counts.get(_ANGLE_OUT_OF_RANGE_REASON, 0) + 1
            )
            continue
        candidate = transform_seen_prior_future(source, config, theta_rad=theta)
        gate = validate_seen_prior_future_environment(
            candidate,
            environment=environment,
        )
        if not gate.accepted:
            if gate.reason not in SEEN_PRIOR_M2_REJECTION_REASONS:
                raise RuntimeError("M2 gate returned an invalid rejection reason")
            rejection_counts[gate.reason] = rejection_counts.get(gate.reason, 0) + 1
            continue
        return SeenPriorResult(
            mother_id=source.mother_id,
            history_poses=candidate.history_poses,
            current_pose=candidate.current_pose,
            future_poses=candidate.future_poses,
            theta_rad=theta,
            accepted_attempt=attempt,
        )
    return SeenPriorFailure(
        mother_id=source.mother_id,
        reason=SEEN_PRIOR_M3_FAILURE_REASON,
        attempts=config.max_attempts_per_mother,
        rejection_counts=rejection_counts,
    )


__all__ = (
    "SEEN_PRIOR_ANGLE_PRIOR_VERSION",
    "SEEN_PRIOR_CONFIG_VERSION",
    "SEEN_PRIOR_GENERATOR_VERSION",
    "SEEN_PRIOR_M2_ENVIRONMENT_GATE_VERSION",
    "SEEN_PRIOR_M2_REJECTION_REASONS",
    "SEEN_PRIOR_M3_FAILURE_REASON",
    "SEEN_PRIOR_M3_SELECTION_VERSION",
    "SEEN_PRIOR_SAMPLING_VERSION",
    "SEEN_PRIOR_TRANSFORM_VERSION",
    "SeenPriorConfig",
    "SeenPriorContextSweep",
    "SeenPriorEnvironment",
    "SeenPriorEnvironmentValidation",
    "SeenPriorFailure",
    "SeenPriorResult",
    "SeenPriorFuture",
    "SeenPriorSource",
    "draw_seen_prior_angle_attempts",
    "generate_seen_prior",
    "load_seen_prior_config",
    "make_seen_prior_rng",
    "seen_prior_mother_seed",
    "transform_seen_prior_future",
    "validate_seen_prior_future_environment",
    "validate_seen_prior_source",
)
