"""Frozen data contracts for the event-centered blind-spot risk project.

This module is owned exclusively by SOP-00. Every other workflow imports from
here and must not redefine schemas, channel ordering, tensor dimensions, or the
input/oracle isolation rules.

Key guarantees enforced here:

- A single ``SCHEMA_VERSION`` string stamps every serialized artifact.
- Observed state, oracle context, and oracle world are three distinct types so
  that future/hidden information can never leak into a model input object.
- ``RiskSample`` and ``VerificationSample`` carry only deployment-available
  inputs, supervision labels, and provenance metadata.
- Serialization uses ``.npz`` with numeric arrays plus an embedded JSON metadata
  string. ``allow_pickle`` is never used, so no Python object arrays are stored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "3.0.0"

# Every serialized future pose uses endpoint semantics.  The current pose q0 is
# an integration seed only; array index k stores q_(k+1) at (k+1) * future_dt.
POSE_TIME_LAYOUT_VERSION = "future_endpoints_dt_to_horizon_v1"

# --- Channel layout (order is a frozen contract; see spec §11.3 and §2.5) ------
# Two per-timestep history channels, stacked over K history steps.
HISTORY_CHANNELS: tuple[str, ...] = (
    "past_dynamic_occupancy",
    "past_visible_mask",
)
# Nine single-frame current-state channels.
STATE_CHANNELS: tuple[str, ...] = (
    "current_visible_free",
    "current_visible_occupied",
    "current_unobservable_mask",
    "last_seen_occupancy",
    "occlusion_age_map",
    "static_obstacle_map",
    "robot_footprint",
    "robot_velocity_channel",
    "robot_yaw_rate_channel",
)
# Four trajectory-query channels.
TRAJECTORY_CHANNELS: tuple[str, ...] = (
    "swept_volume_mask",
    "time_to_arrival_map",
    "braking_margin_map",
    "centerline_map",
)
# Full ordered listing used for documentation and checkpoint stamping.
INPUT_CHANNELS: tuple[str, ...] = HISTORY_CHANNELS + STATE_CHANNELS + TRAJECTORY_CHANNELS

N_HISTORY_CHANNELS = len(HISTORY_CHANNELS)  # 2
N_STATE_CHANNELS = len(STATE_CHANNELS)  # 9
N_TRAJECTORY_CHANNELS = len(TRAJECTORY_CHANNELS)  # 4

# Fixed vector dimensions (frozen; changing them is a schema change).
ROBOT_STATE_DIM = 2  # (v, omega)
ACTION_VECTOR_DIM = 3  # (duration_s, delta_forward_m, delta_yaw_rad)
QUANTILE_LEVELS: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)
DYNAMIC_OBJECT_TYPES: tuple[str, ...] = (
    "human",
    "carried_object",
    "unknown_dynamic",
)

# Tokens that must never appear in a model-input dataclass field name.
FORBIDDEN_INPUT_TOKENS: tuple[str, ...] = (
    "oracle",
    "hidden_future",
    "pedestrian_future",
    "ped_future",
    "dynamic_object_future",
    "object_future",
    "future_occupancy",
    "post_verification_occupancy",
    "post_verify_occupancy",
    "world",
    "ground_truth",
)

ARRAY_DTYPE = np.float32


class ContractError(ValueError):
    """Raised when an object violates a frozen contract."""


# --- Grid specification --------------------------------------------------------
@dataclass(frozen=True)
class GridSpec:
    """Spatial/temporal grid derived from config; used by shape validators."""

    height: int
    width: int
    history_steps: int
    future_steps: int
    resolution_m: float
    n_history_channels: int = N_HISTORY_CHANNELS
    n_state_channels: int = N_STATE_CHANNELS
    n_trajectory_channels: int = N_TRAJECTORY_CHANNELS


def build_grid_spec(config: dict) -> GridSpec:
    """Build a :class:`GridSpec` from a validated config dict."""
    bev = config["bev"]
    return GridSpec(
        height=int(bev["size"]),
        width=int(bev["size"]),
        history_steps=int(bev["history_steps"]),
        future_steps=int(bev["future_steps"]),
        resolution_m=float(bev["resolution_m"]),
    )


# --- Core dataclasses ----------------------------------------------------------
@dataclass(frozen=True)
class BaseState:
    """Deployment-observable robot-centric state. No future/oracle fields."""

    state_id: str
    split: str
    recording_id: str
    dynamic_object_ids: tuple[str, ...]
    timestamp: float
    robot_history: np.ndarray  # [K, 3] -> x, y, yaw
    robot_state: np.ndarray  # [ROBOT_STATE_DIM]
    visible_dynamic_object_history: dict  # object_id -> [K, 3]
    visible_dynamic_object_specs: dict  # object_id -> JSON-safe footprint spec
    static_map_local: np.ndarray | None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OracleContext:
    """Full dynamic-object history/future for labels and analysis only."""

    base_state_id: str
    dynamic_object_history: dict  # object_id -> [K, 3]
    dynamic_object_future: dict  # object_id -> [T, 3]
    dynamic_object_specs: dict  # object_id -> JSON-safe footprint spec
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OracleWorld:
    """One counterfactual hidden world used to compute labels/targets."""

    world_id: str
    base_state_id: str
    static_occupancy: np.ndarray  # [H, W]
    dynamic_object_trajectories: dict  # object_id -> [T, 3]
    dynamic_object_specs: dict  # object_id -> JSON-safe footprint spec
    occluders: tuple[dict, ...]
    blind_spot_config: dict
    random_seed: int
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LocalTrajectory:
    """Candidate local plan and its precomputed query maps."""

    trajectory_id: str
    poses: np.ndarray  # [T, 3]; poses[k] == q_(k+1) at (k+1) * future_dt
    controls: np.ndarray  # [T, 2]; controls[k] acts on [k*dt, (k+1)*dt]
    swept_mask: np.ndarray  # [H, W]
    tta_map: np.ndarray  # [H, W], not-traversed cells == -1
    braking_map: np.ndarray  # [H, W]
    centerline_map: np.ndarray  # [H, W]
    task_cost: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RiskSample:
    """Trajectory-conditioned hidden-risk training sample (inputs + labels)."""

    sample_id: str
    split: str
    base_state_id: str
    pair_group_id: str
    event_type: str
    bev_history: np.ndarray  # [K, N_HISTORY_CHANNELS, H, W]
    state_channels: np.ndarray  # [N_STATE_CHANNELS, H, W]
    trajectory_channels: np.ndarray  # [N_TRAJECTORY_CHANNELS, H, W]
    robot_state: np.ndarray  # [ROBOT_STATE_DIM]
    collision_label: int
    risk_severity: float
    min_clearance: float
    near_miss: int
    first_collision_time: float | None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationSample:
    """Verification-value training sample (inputs + net-value targets)."""

    sample_id: str
    split: str
    base_state_id: str
    nominal_trajectory_id: str
    verification_action_id: str
    bev_history: np.ndarray  # [K, N_HISTORY_CHANNELS, H, W]
    state_channels: np.ndarray  # [N_STATE_CHANNELS, H, W]
    trajectory_channels: np.ndarray  # [N_TRAJECTORY_CHANNELS, H, W]
    verification_fov_mask: np.ndarray  # [1, H, W] expected-visible geometry only
    verification_action_vector: np.ndarray  # [ACTION_VECTOR_DIM]
    value_target: float
    useful_target: int
    br_before: float
    post_risk: float
    metadata: dict = field(default_factory=dict)


_CLASS_REGISTRY: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        BaseState,
        OracleContext,
        OracleWorld,
        LocalTrajectory,
        RiskSample,
        VerificationSample,
    )
}

# Dataclasses that represent model inputs and must pass the oracle-leakage guard.
MODEL_INPUT_CLASSES: tuple[type, ...] = (BaseState, RiskSample, VerificationSample)


# --- Oracle-leakage guard ------------------------------------------------------
def assert_no_oracle_leakage(cls: type) -> None:
    """Fail if a model-input dataclass exposes a forbidden field name.

    Only structural field names are checked. ``metadata`` is permitted because it
    is provenance, but no top-level field may reference oracle/future/world data.
    """
    if not is_dataclass(cls):
        raise ContractError(f"{cls!r} is not a dataclass")
    for f in fields(cls):
        name = f.name.lower()
        if name == "metadata":
            continue
        for token in FORBIDDEN_INPUT_TOKENS:
            if token in name:
                raise ContractError(
                    f"{cls.__name__}.{f.name} contains forbidden token '{token}'"
                )


# --- Shape/dtype validators ----------------------------------------------------
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _check_float_array(arr: Any, shape: tuple[int, ...], name: str) -> None:
    _require(isinstance(arr, np.ndarray), f"{name} must be np.ndarray")
    _require(arr.dtype == ARRAY_DTYPE, f"{name} dtype must be float32, got {arr.dtype}")
    _require(arr.shape == shape, f"{name} shape must be {shape}, got {arr.shape}")
    _require(np.isfinite(arr).all(), f"{name} contains NaN/Inf")


def validate_dynamic_object_spec(spec: dict) -> None:
    """Validate the JSON-safe type and footprint fields for one dynamic object."""
    _require(isinstance(spec, dict), "dynamic object spec must be a dict")
    _require(
        set(spec) == {"object_type", "footprint"},
        "dynamic object spec keys must be object_type and footprint",
    )
    object_type = spec["object_type"]
    _require(
        object_type in DYNAMIC_OBJECT_TYPES,
        f"object_type must be one of {DYNAMIC_OBJECT_TYPES}",
    )
    footprint = spec["footprint"]
    _require(isinstance(footprint, dict), "footprint must be a dict")
    kind = footprint.get("kind")
    _require(kind in {"circle", "rectangle"}, "unsupported footprint kind")
    if object_type == "human":
        _require(kind == "circle", "human footprint must be circle")
    if object_type == "carried_object":
        _require(kind == "rectangle", "carried_object footprint must be rectangle")
    expected_keys = (
        {"kind", "radius_m"}
        if kind == "circle"
        else {"kind", "length_m", "width_m"}
    )
    _require(set(footprint) == expected_keys, "footprint keys do not match kind")
    dimensions = (
        (footprint["radius_m"],)
        if kind == "circle"
        else (footprint["length_m"], footprint["width_m"])
    )
    _require(
        all(
            not isinstance(value, bool)
            and isinstance(value, (int, float, np.integer, np.floating))
            and np.isfinite(value)
            and float(value) > 0.0
            for value in dimensions
        ),
        "footprint dimensions must be finite and positive",
    )


def validate_base_state(state: BaseState, grid: GridSpec) -> None:
    """Validate one deployment-observable dynamic-object base state."""
    _check_float_array(
        state.robot_history, (grid.history_steps, 3), "robot_history"
    )
    _check_float_array(state.robot_state, (ROBOT_STATE_DIM,), "robot_state")
    _require(
        isinstance(state.dynamic_object_ids, tuple),
        "dynamic_object_ids must be a tuple",
    )
    _require(
        all(isinstance(object_id, str) and object_id for object_id in state.dynamic_object_ids),
        "dynamic_object_ids must contain non-empty strings",
    )
    _require(
        state.dynamic_object_ids == tuple(sorted(set(state.dynamic_object_ids))),
        "dynamic_object_ids must be sorted and unique",
    )
    object_ids = set(state.dynamic_object_ids)
    _require(
        object_ids == set(state.visible_dynamic_object_history),
        "dynamic_object_ids and visible history keys must match",
    )
    _require(
        object_ids == set(state.visible_dynamic_object_specs),
        "dynamic_object_ids and visible spec keys must match",
    )
    for object_id in state.dynamic_object_ids:
        _check_float_array(
            state.visible_dynamic_object_history[object_id],
            (grid.history_steps, 3),
            f"visible_dynamic_object_history[{object_id!r}]",
        )
        validate_dynamic_object_spec(state.visible_dynamic_object_specs[object_id])
    if state.static_map_local is not None:
        _check_float_array(
            state.static_map_local,
            (grid.height, grid.width),
            "static_map_local",
        )


def validate_oracle_context(context: OracleContext, grid: GridSpec) -> None:
    """Validate full dynamic-object label context and key alignment."""
    object_ids = set(context.dynamic_object_history)
    _require(
        object_ids == set(context.dynamic_object_future),
        "oracle history and future keys must match",
    )
    _require(
        object_ids == set(context.dynamic_object_specs),
        "oracle trajectory and spec keys must match",
    )
    for object_id in sorted(object_ids):
        _check_float_array(
            context.dynamic_object_history[object_id],
            (grid.history_steps, 3),
            f"dynamic_object_history[{object_id!r}]",
        )
        _check_float_array(
            context.dynamic_object_future[object_id],
            (grid.future_steps, 3),
            f"dynamic_object_future[{object_id!r}]",
        )
        validate_dynamic_object_spec(context.dynamic_object_specs[object_id])


def validate_oracle_world(world: OracleWorld, grid: GridSpec) -> None:
    """Validate one counterfactual world under the dynamic-object schema."""
    _check_float_array(
        world.static_occupancy,
        (grid.height, grid.width),
        "static_occupancy",
    )
    object_ids = set(world.dynamic_object_trajectories)
    _require(
        object_ids == set(world.dynamic_object_specs),
        "oracle world trajectory and spec keys must match",
    )
    for object_id in sorted(object_ids):
        _check_float_array(
            world.dynamic_object_trajectories[object_id],
            (grid.future_steps, 3),
            f"dynamic_object_trajectories[{object_id!r}]",
        )
        validate_dynamic_object_spec(world.dynamic_object_specs[object_id])
    _require(
        isinstance(world.random_seed, int) and not isinstance(world.random_seed, bool),
        "random_seed must be an int",
    )


def validate_risk_sample(sample: RiskSample, grid: GridSpec) -> None:
    """Raise :class:`ContractError` if a :class:`RiskSample` is malformed."""
    h, w, k = grid.height, grid.width, grid.history_steps
    _check_float_array(
        sample.bev_history, (k, grid.n_history_channels, h, w), "bev_history"
    )
    _check_float_array(
        sample.state_channels, (grid.n_state_channels, h, w), "state_channels"
    )
    _check_float_array(
        sample.trajectory_channels,
        (grid.n_trajectory_channels, h, w),
        "trajectory_channels",
    )
    _check_float_array(sample.robot_state, (ROBOT_STATE_DIM,), "robot_state")
    _require(sample.collision_label in (0, 1), "collision_label must be 0/1")
    _require(sample.near_miss in (0, 1), "near_miss must be 0/1")
    _require(0.0 <= sample.risk_severity <= 1.0, "risk_severity must be in [0, 1]")
    _require(
        sample.first_collision_time is None
        or isinstance(sample.first_collision_time, float),
        "first_collision_time must be None or float",
    )
    if sample.collision_label == 1:
        _require(sample.near_miss == 0, "collision implies near_miss == 0")


def validate_verification_sample(sample: VerificationSample, grid: GridSpec) -> None:
    """Raise :class:`ContractError` if a :class:`VerificationSample` is malformed."""
    h, w, k = grid.height, grid.width, grid.history_steps
    _check_float_array(
        sample.bev_history, (k, grid.n_history_channels, h, w), "bev_history"
    )
    _check_float_array(
        sample.state_channels, (grid.n_state_channels, h, w), "state_channels"
    )
    _check_float_array(
        sample.trajectory_channels,
        (grid.n_trajectory_channels, h, w),
        "trajectory_channels",
    )
    _check_float_array(
        sample.verification_fov_mask, (1, h, w), "verification_fov_mask"
    )
    _check_float_array(
        sample.verification_action_vector,
        (ACTION_VECTOR_DIM,),
        "verification_action_vector",
    )
    _require(sample.useful_target in (0, 1), "useful_target must be 0/1")
    _require(
        (sample.value_target > 0.0) == (sample.useful_target == 1),
        "useful_target must equal int(value_target > 0)",
    )


# --- Serialization (npz + embedded JSON metadata; no object arrays) ------------
_META_KEY = "meta_json"


def _to_jsonable(value: Any) -> Any:
    """Convert config/metadata leaves to JSON-safe python primitives."""
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    raise ContractError(f"value of type {type(value)!r} is not JSON serializable")


def encode_dataclass(obj: Any) -> tuple[dict[str, np.ndarray], dict]:
    """Split a dataclass into an ``arr_N`` array map and a JSON metadata dict."""
    if not is_dataclass(obj):
        raise ContractError(f"{obj!r} is not a dataclass instance")
    arrays: dict[str, np.ndarray] = {}
    field_meta: dict[str, dict] = {}
    order: list[np.ndarray] = []

    def _add(arr: np.ndarray) -> int:
        idx = len(order)
        order.append(np.ascontiguousarray(arr))
        return idx

    for f in fields(obj):
        val = getattr(obj, f.name)
        if isinstance(val, np.ndarray):
            field_meta[f.name] = {"kind": "ndarray", "idx": _add(val)}
        elif (
            isinstance(val, dict)
            and len(val) > 0
            and all(isinstance(v, np.ndarray) for v in val.values())
        ):
            keys = [str(k) for k in val.keys()]
            idxs = [_add(val[k]) for k in val.keys()]
            field_meta[f.name] = {"kind": "ndarray_dict", "keys": keys, "idx": idxs}
        elif isinstance(val, tuple):
            field_meta[f.name] = {"kind": "json_tuple", "value": _to_jsonable(list(val))}
        else:
            field_meta[f.name] = {"kind": "json", "value": _to_jsonable(val)}

    for i, arr in enumerate(order):
        arrays[f"arr_{i}"] = arr
    meta = {
        "schema_version": SCHEMA_VERSION,
        "class": type(obj).__name__,
        "fields": field_meta,
    }
    return arrays, meta


def decode_dataclass(arrays: dict[str, np.ndarray], meta: dict) -> Any:
    """Rebuild a dataclass instance from an array map and JSON metadata."""
    _require(
        meta.get("schema_version") == SCHEMA_VERSION,
        (
            f"schema_version must be {SCHEMA_VERSION}, "
            f"got {meta.get('schema_version')!r}"
        ),
    )
    cls = _CLASS_REGISTRY.get(meta["class"])
    if cls is None:
        raise ContractError(f"unknown class in metadata: {meta['class']!r}")
    kwargs: dict[str, Any] = {}
    for name, spec in meta["fields"].items():
        kind = spec["kind"]
        if kind == "ndarray":
            kwargs[name] = arrays[f"arr_{spec['idx']}"]
        elif kind == "ndarray_dict":
            kwargs[name] = {
                key: arrays[f"arr_{idx}"]
                for key, idx in zip(spec["keys"], spec["idx"])
            }
        elif kind == "json_tuple":
            kwargs[name] = tuple(spec["value"])
        elif kind == "json":
            kwargs[name] = spec["value"]
        else:  # pragma: no cover - guarded by encoder
            raise ContractError(f"unknown field kind: {kind!r}")
    return cls(**kwargs)


def save_dataclass(obj: Any, path: str | Path) -> Path:
    """Serialize a dataclass to a single ``.npz`` file (arrays + JSON metadata)."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, meta = encode_dataclass(obj)
    payload = dict(arrays)
    payload[_META_KEY] = np.asarray(json.dumps(meta, sort_keys=True))
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **payload)
    tmp.replace(path)  # atomic rename so partial writes are never observed
    return path


def load_dataclass(path: str | Path) -> Any:
    """Load a dataclass previously written by :func:`save_dataclass`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data[_META_KEY]))
        arrays = {key: data[key] for key in data.files if key != _META_KEY}
    return decode_dataclass(arrays, meta)
