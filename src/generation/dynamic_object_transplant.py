"""Typed MotionSnippet sampling and deterministic event-centred transplant."""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

import numpy as np

from src.contracts import DYNAMIC_OBJECT_TYPES, validate_dynamic_object_spec
from src.datasets.snippet_library import MotionSnippet, SnippetLibrary
from src.geometry import CircleFootprint, Footprint, RectangleFootprint, wrap_angle
from src.utils.seeding import stable_digest


class TargetPolicyError(ValueError):
    """Raised when a target-type policy is incomplete or nonphysical."""


class TransplantError(ValueError):
    """A finite, auditable rejection from typed snippet transplantation."""

    def __init__(self, reason: str, message: str | None = None):
        super().__init__(message or reason)
        self.reason = reason


@dataclass(frozen=True)
class TargetTypePolicy:
    """Canonical whitelist, frozen-order normalized weights, and digest."""

    whitelist: tuple[str, ...]
    weights: dict[str, float]
    digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "whitelist": list(self.whitelist),
            "weights": dict(self.weights),
        }


@dataclass(frozen=True)
class TransplantedDynamicObject:
    """One target object's current pose, future trajectory, and provenance."""

    target_dynamic_object_id: str
    source_object_id: str
    snippet_id: str
    object_type: str
    footprint_spec: dict[str, object]
    footprint_spec_digest: str
    current_pose: np.ndarray
    future_poses: np.ndarray
    provenance: dict[str, object]


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: Any, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric vector with shape ({size},)")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def normalize_target_type_policy(policy: Mapping[str, Any]) -> TargetTypePolicy:
    """Strictly validate and canonicalize a complete target-type policy."""

    if not isinstance(policy, Mapping) or set(policy) != {"whitelist", "weights"}:
        raise TargetPolicyError("policy must contain whitelist and weights")
    whitelist = policy["whitelist"]
    if not isinstance(whitelist, (list, tuple)) or not whitelist:
        raise TargetPolicyError("whitelist must be a non-empty sequence")
    if len(set(whitelist)) != len(whitelist) or any(
        item not in DYNAMIC_OBJECT_TYPES for item in whitelist
    ):
        raise TargetPolicyError("whitelist must contain unique frozen object types")
    canonical_whitelist = tuple(
        object_type
        for object_type in DYNAMIC_OBJECT_TYPES
        if object_type in whitelist
    )
    weights = policy["weights"]
    if not isinstance(weights, Mapping) or set(weights) != set(DYNAMIC_OBJECT_TYPES):
        raise TargetPolicyError(
            "weights must contain exactly the three frozen dynamic-object types"
        )
    parsed_weights: dict[str, float] = {}
    for object_type in DYNAMIC_OBJECT_TYPES:
        try:
            value = _finite_real(weights[object_type], name=f"weights.{object_type}")
        except (TypeError, ValueError) as exc:
            raise TargetPolicyError(str(exc)) from exc
        if value < 0.0:
            raise TargetPolicyError("target-type weights must be non-negative")
        parsed_weights[object_type] = (
            value if object_type in canonical_whitelist else 0.0
        )
    total = sum(parsed_weights[object_type] for object_type in canonical_whitelist)
    if not np.isfinite(total) or total <= 0.0:
        raise TargetPolicyError("at least one whitelisted weight must be positive")
    normalized_weights = {
        object_type: (
            parsed_weights[object_type] / total
            if object_type in canonical_whitelist
            else 0.0
        )
        for object_type in DYNAMIC_OBJECT_TYPES
    }
    canonical = {
        "whitelist": list(canonical_whitelist),
        "weights": normalized_weights,
    }
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return TargetTypePolicy(
        whitelist=canonical_whitelist,
        weights=normalized_weights,
        digest=stable_digest(payload, size=16),
    )


def footprint_from_spec(spec: Mapping[str, Any]) -> Footprint:
    """Materialize a frozen contract footprint without reclassification."""

    if not isinstance(spec, dict):
        spec = dict(spec)
    validate_dynamic_object_spec(spec)
    footprint = spec["footprint"]
    if footprint["kind"] == "circle":
        return CircleFootprint(float(footprint["radius_m"]))
    return RectangleFootprint(
        float(footprint["length_m"]), float(footprint["width_m"])
    )


def _validate_snippet(snippet: MotionSnippet) -> None:
    if not isinstance(snippet, MotionSnippet):
        raise TypeError("snippet must be a MotionSnippet")
    spec = {"object_type": snippet.object_type, "footprint": snippet.footprint}
    try:
        validate_dynamic_object_spec(spec)
    except ValueError as exc:
        raise TransplantError("snippet_contract_invalid", str(exc)) from exc
    positions = np.asarray(snippet.positions)
    velocities = np.asarray(snippet.velocities)
    headings = np.asarray(snippet.headings)
    if (
        positions.ndim != 2
        or positions.shape[1:] != (2,)
        or positions.shape[0] < 2
        or velocities.shape != positions.shape
        or headings.shape != (positions.shape[0],)
    ):
        raise TransplantError("snippet_shape_invalid")
    if (
        positions.dtype != np.float32
        or velocities.dtype != np.float32
        or headings.dtype != np.float32
    ):
        raise TransplantError("snippet_dtype_invalid")
    if not (
        np.isfinite(positions).all()
        and np.isfinite(velocities).all()
        and np.isfinite(headings).all()
    ):
        raise TransplantError("snippet_nonfinite")
    if not np.isfinite(snippet.duration_s) or snippet.duration_s <= 0.0:
        raise TransplantError("snippet_duration_invalid")


def sample_motion_snippet(
    libraries: Mapping[str, SnippetLibrary],
    *,
    split: str,
    policy: TargetTypePolicy,
    rng: np.random.Generator,
) -> MotionSnippet:
    """Sample from split/type-isolated libraries without weight fallback."""

    if not isinstance(libraries, Mapping):
        raise TypeError("libraries must be a mapping")
    if not isinstance(policy, TargetTypePolicy):
        raise TypeError("policy must be a TargetTypePolicy")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    for object_type in policy.whitelist:
        if policy.weights[object_type] <= 0.0:
            continue
        library = libraries.get(object_type)
        if not isinstance(library, SnippetLibrary) or not library.snippets:
            raise TransplantError("snippet_library_missing")
        if library.object_type != object_type:
            raise TransplantError("snippet_library_type_mismatch")
        for snippet in library.snippets:
            _validate_snippet(snippet)
            if snippet.object_type != object_type:
                raise TransplantError("snippet_library_type_mismatch")
            if snippet.split != split:
                raise TransplantError("snippet_split_mismatch")
    probabilities = np.asarray(
        [policy.weights[object_type] for object_type in DYNAMIC_OBJECT_TYPES],
        dtype=np.float64,
    )
    object_type = str(rng.choice(DYNAMIC_OBJECT_TYPES, p=probabilities))
    library = libraries[object_type]
    index = int(rng.integers(0, len(library.snippets)))
    return library.snippets[index]


def _stable_target_id(
    *,
    snippet: MotionSnippet,
    base_state_id: str,
    trajectory_id: str,
    target_type_policy_digest: str,
    seed: int,
    conflict_time_s: float,
    conflict_point: np.ndarray,
    crossing_direction: np.ndarray,
    time_scale: float,
    context_object_ids: set[str],
) -> str:
    salt = 0
    while True:
        digest = stable_digest(
            base_state_id,
            trajectory_id,
            snippet.snippet_id,
            snippet.source_object_id,
            target_type_policy_digest,
            int(seed),
            f"{conflict_time_s:.9f}",
            *(f"{value:.9f}" for value in conflict_point),
            *(f"{value:.9f}" for value in crossing_direction),
            f"{time_scale:.9f}",
            salt,
            size=12,
        )
        candidate = f"generated::{snippet.object_type}::{digest}"
        if candidate not in context_object_ids:
            return candidate
        salt += 1


def transplant_snippet(
    snippet: MotionSnippet,
    *,
    conflict_point: Any,
    conflict_time_s: float,
    crossing_direction: Any,
    time_scale: float,
    future_dt_s: float,
    future_steps: int,
    base_state_id: str,
    trajectory_id: str,
    target_type_policy_digest: str,
    seed: int,
    context_object_ids: tuple[str, ...] | list[str] | set[str],
) -> TransplantedDynamicObject:
    """Anchor one frozen typed snippet at a trajectory conflict event.

    The snippet's temporal samples remain on the canonical output grid. Time
    scaling changes displacement about the conflict anchor, preserving the
    complete real motion shape and per-frame heading without extrapolation.
    """

    _validate_snippet(snippet)
    point = _vector(conflict_point, name="conflict_point", size=2)
    direction = _vector(crossing_direction, name="crossing_direction", size=2)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-9:
        raise ValueError("crossing_direction must be non-zero")
    direction /= direction_norm
    conflict_time = _finite_real(conflict_time_s, name="conflict_time_s")
    scale = _finite_real(time_scale, name="time_scale")
    dt_s = _finite_real(future_dt_s, name="future_dt_s")
    if not 0.8 <= scale <= 1.2:
        raise ValueError("time_scale must be in [0.8, 1.2]")
    if dt_s <= 0.0:
        raise ValueError("future_dt_s must be positive")
    if isinstance(future_steps, (bool, np.bool_)) or not isinstance(
        future_steps, (int, np.integer)
    ):
        raise TypeError("future_steps must be an integer")
    future_steps = int(future_steps)
    if future_steps <= 0:
        raise ValueError("future_steps must be positive")
    horizon_s = future_steps * dt_s
    if not 0.0 < conflict_time <= horizon_s:
        raise ValueError("conflict_time_s must lie in the future horizon")
    if snippet.duration_s + 1e-9 < horizon_s:
        raise TransplantError("snippet_too_short")
    if not isinstance(base_state_id, str) or not base_state_id:
        raise ValueError("base_state_id must be a non-empty string")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("trajectory_id must be a non-empty string")
    if not isinstance(target_type_policy_digest, str) or not target_type_policy_digest:
        raise ValueError("target_type_policy_digest must be a non-empty string")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    context_ids = set(context_object_ids)
    if any(not isinstance(object_id, str) or not object_id for object_id in context_ids):
        raise ValueError("context_object_ids must contain non-empty strings")

    output_times = np.arange(future_steps + 1, dtype=np.float64) * dt_s
    source_times = np.linspace(
        0.0, float(snippet.duration_s), snippet.positions.shape[0], dtype=np.float64
    )
    positions = np.column_stack(
        (
            np.interp(output_times, source_times, snippet.positions[:, 0]),
            np.interp(output_times, source_times, snippet.positions[:, 1]),
        )
    )
    unwrapped_headings = np.unwrap(snippet.headings.astype(np.float64))
    headings = np.interp(output_times, source_times, unwrapped_headings)
    anchor = np.asarray(
        [
            np.interp(conflict_time, source_times, snippet.positions[:, 0]),
            np.interp(conflict_time, source_times, snippet.positions[:, 1]),
        ],
        dtype=np.float64,
    )
    source_velocity = np.asarray(
        [
            np.interp(conflict_time, source_times, snippet.velocities[:, 0]),
            np.interp(conflict_time, source_times, snippet.velocities[:, 1]),
        ],
        dtype=np.float64,
    )
    if float(np.linalg.norm(source_velocity)) <= 1e-9:
        before = max(0.0, conflict_time - dt_s)
        after = min(float(snippet.duration_s), conflict_time + dt_s)
        source_velocity = np.asarray(
            [
                np.interp(after, source_times, snippet.positions[:, 0])
                - np.interp(before, source_times, snippet.positions[:, 0]),
                np.interp(after, source_times, snippet.positions[:, 1])
                - np.interp(before, source_times, snippet.positions[:, 1]),
            ]
        )
    if float(np.linalg.norm(source_velocity)) <= 1e-9:
        raise TransplantError("snippet_stationary_at_conflict")
    source_angle = float(np.arctan2(source_velocity[1], source_velocity[0]))
    target_angle = float(np.arctan2(direction[1], direction[0]))
    rotation_angle = float(wrap_angle(target_angle - source_angle))
    cosine = np.cos(rotation_angle)
    sine = np.sin(rotation_angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    transformed_positions = ((positions - anchor) / scale) @ rotation.T + point
    transformed_headings = wrap_angle(headings + rotation_angle)
    poses = np.column_stack((transformed_positions, transformed_headings)).astype(
        np.float32
    )
    if not np.isfinite(poses).all():
        raise TransplantError("transplant_nonfinite")

    footprint_spec = {
        "object_type": snippet.object_type,
        "footprint": dict(snippet.footprint),
    }
    footprint_payload = json.dumps(
        footprint_spec,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    footprint_digest = stable_digest(footprint_payload, size=16)
    target_id = _stable_target_id(
        snippet=snippet,
        base_state_id=base_state_id,
        trajectory_id=trajectory_id,
        target_type_policy_digest=target_type_policy_digest,
        seed=int(seed),
        conflict_time_s=conflict_time,
        conflict_point=point,
        crossing_direction=direction,
        time_scale=scale,
        context_object_ids=context_ids,
    )
    track_provenance = snippet.provenance.get("track_provenance", {})
    provenance = {
        "snippet_id": snippet.snippet_id,
        "source_recording_id": snippet.source_recording_id,
        "source_object_id": snippet.source_object_id,
        "source_body_name": snippet.provenance.get("source_body_name"),
        "raw_role": snippet.provenance.get("raw_role"),
        "geometry_source": track_provenance.get("geometry_source", "unknown"),
        "orientation_source": track_provenance.get(
            "orientation_source", "unknown"
        ),
        "target_type_policy_digest": target_type_policy_digest,
        "footprint_spec_digest": footprint_digest,
        "conflict_time_s": conflict_time,
        "conflict_point": [float(value) for value in point],
        "crossing_direction": [float(value) for value in direction],
        "rotation_rad": rotation_angle,
        "time_scale": scale,
        "seed": int(seed),
    }
    return TransplantedDynamicObject(
        target_dynamic_object_id=target_id,
        source_object_id=snippet.source_object_id,
        snippet_id=snippet.snippet_id,
        object_type=snippet.object_type,
        footprint_spec=footprint_spec,
        footprint_spec_digest=footprint_digest,
        current_pose=poses[0],
        future_poses=poses[1:],
        provenance=provenance,
    )
