"""Typed MotionSnippet sampling and deterministic event-centred transplant."""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

import numpy as np

from src.contracts import DYNAMIC_OBJECT_TYPES, validate_dynamic_object_spec
from src.datasets.snippet_library import MotionSnippet, SnippetLibrary
from src.generation.blind_reachability import (
    BLIND_REACHABILITY_ALGORITHM_VERSION,
    REACHABLE_ARC_SCHEDULE_VERSION,
    ReachabilityCandidate,
)
from src.geometry import CircleFootprint, Footprint, RectangleFootprint, wrap_angle
from src.utils.seeding import stable_digest


MOTION_SNIPPET_LAYOUT_VERSION = "history8_current7_future15_v1"
MOTION_SNIPPET_SAMPLE_COUNT = 23
MOTION_SNIPPET_HISTORY_STEPS = 8
MOTION_SNIPPET_CURRENT_INDEX = 7
MOTION_SNIPPET_FUTURE_STEPS = 15
MOTION_SNIPPET_SAMPLE_DT_S = 0.2
MOTION_SNIPPET_DURATION_S = 4.4
MOTION_SNIPPET_CURRENT_TIME_S = 1.4
TRANSFORM_ALGORITHM_VERSION = "reachability_candidate_se2_v2"


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
    """One target object's measured history, current pose, and future."""

    target_dynamic_object_id: str
    source_object_id: str
    snippet_id: str
    object_type: str
    footprint_spec: dict[str, object]
    footprint_spec_digest: str
    history_poses: np.ndarray
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
    for name, value in (
        ("source_recording_id", snippet.source_recording_id),
        ("source_session_id", snippet.source_session_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise TransplantError(
                "snippet_contract_invalid", f"snippet {name} must be non-empty"
            )
    positions = np.asarray(snippet.positions)
    velocities = np.asarray(snippet.velocities)
    headings = np.asarray(snippet.headings)
    if (
        positions.shape != (MOTION_SNIPPET_SAMPLE_COUNT, 2)
        or velocities.shape != (MOTION_SNIPPET_SAMPLE_COUNT, 2)
        or headings.shape != (MOTION_SNIPPET_SAMPLE_COUNT,)
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
    if not np.isfinite(snippet.duration_s) or not np.isclose(
        float(snippet.duration_s),
        MOTION_SNIPPET_DURATION_S,
        rtol=0.0,
        atol=1e-6,
    ):
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
            snippet.source_recording_id,
            snippet.source_session_id,
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

    The original 23 samples receive one shared rigid SE(2) transform. The
    measured history and future are split around source current index 7; no
    temporal scaling, resampling, duplication, or extrapolation is allowed.
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
    if scale != 1.0:
        raise ValueError("time_scale must equal 1.0")
    if not np.isclose(
        dt_s,
        MOTION_SNIPPET_SAMPLE_DT_S,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("future_dt_s must equal 0.2")
    if isinstance(future_steps, (bool, np.bool_)) or not isinstance(
        future_steps, (int, np.integer)
    ):
        raise TypeError("future_steps must be an integer")
    future_steps = int(future_steps)
    if future_steps != MOTION_SNIPPET_FUTURE_STEPS:
        raise ValueError("future_steps must equal 15")
    horizon_s = future_steps * dt_s
    if not 0.0 < conflict_time <= horizon_s:
        raise ValueError("conflict_time_s must lie in the future horizon")
    source_anchor_time_s = MOTION_SNIPPET_CURRENT_TIME_S + conflict_time
    if source_anchor_time_s > MOTION_SNIPPET_DURATION_S + 1e-9:
        raise TransplantError("snippet_anchor_out_of_bounds")
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

    source_times = (
        np.arange(MOTION_SNIPPET_SAMPLE_COUNT, dtype=np.float64)
        * MOTION_SNIPPET_SAMPLE_DT_S
    )
    positions = snippet.positions.astype(np.float64)
    headings = snippet.headings.astype(np.float64)
    anchor = np.asarray(
        [
            np.interp(source_anchor_time_s, source_times, positions[:, 0]),
            np.interp(source_anchor_time_s, source_times, positions[:, 1]),
        ],
        dtype=np.float64,
    )
    source_velocity = np.asarray(
        [
            np.interp(
                source_anchor_time_s, source_times, snippet.velocities[:, 0]
            ),
            np.interp(
                source_anchor_time_s, source_times, snippet.velocities[:, 1]
            ),
        ],
        dtype=np.float64,
    )
    if float(np.linalg.norm(source_velocity)) <= 1e-9:
        before = max(0.0, source_anchor_time_s - dt_s)
        after = min(MOTION_SNIPPET_DURATION_S, source_anchor_time_s + dt_s)
        source_velocity = np.asarray(
            [
                np.interp(after, source_times, positions[:, 0])
                - np.interp(before, source_times, positions[:, 0]),
                np.interp(after, source_times, positions[:, 1])
                - np.interp(before, source_times, positions[:, 1]),
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
    transformed_positions = (positions - anchor) @ rotation.T + point
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
        "source_session_id": snippet.source_session_id,
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
        "motion_snippet_layout_version": MOTION_SNIPPET_LAYOUT_VERSION,
        "source_current_index": MOTION_SNIPPET_CURRENT_INDEX,
        "source_current_time_s": MOTION_SNIPPET_CURRENT_TIME_S,
        "source_conflict_anchor_time_s": source_anchor_time_s,
        "seed": int(seed),
    }
    history_poses = np.array(
        poses[:MOTION_SNIPPET_HISTORY_STEPS],
        dtype=np.float32,
        order="C",
        copy=True,
    )
    current_pose = np.array(
        history_poses[-1], dtype=np.float32, order="C", copy=True
    )
    future_poses = np.array(
        poses[MOTION_SNIPPET_HISTORY_STEPS:],
        dtype=np.float32,
        order="C",
        copy=True,
    )
    return TransplantedDynamicObject(
        target_dynamic_object_id=target_id,
        source_object_id=snippet.source_object_id,
        snippet_id=snippet.snippet_id,
        object_type=snippet.object_type,
        footprint_spec=footprint_spec,
        footprint_spec_digest=footprint_digest,
        history_poses=history_poses,
        current_pose=current_pose,
        future_poses=future_poses,
        provenance=provenance,
    )


def _canonical_context_object_ids(
    value: tuple[str, ...] | list[str] | set[str],
) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list, set)):
        raise TypeError("context_object_ids must be a tuple, list, or set")
    if any(not isinstance(object_id, str) or not object_id for object_id in value):
        raise ValueError("context_object_ids must contain non-empty strings")
    return tuple(sorted(set(value)))


def _reachability_transform_id(
    *,
    snippet: MotionSnippet,
    candidate: ReachabilityCandidate,
    footprint_spec_digest: str,
    target_type_policy_digest: str,
    seed: int,
    context_object_ids: tuple[str, ...],
    transformed_poses: np.ndarray,
) -> str:
    canonical_poses = np.ascontiguousarray(transformed_poses, dtype=np.dtype("<f4"))
    payload = {
        "domain": "reachability-candidate-se2-transform-identity-v2",
        "transform_algorithm_version": TRANSFORM_ALGORITHM_VERSION,
        "reachability_algorithm_version": BLIND_REACHABILITY_ALGORITHM_VERSION,
        "reachable_arc_schedule_version": REACHABLE_ARC_SCHEDULE_VERSION,
        "motion_snippet_layout_version": MOTION_SNIPPET_LAYOUT_VERSION,
        "reachability_candidate_id": candidate.candidate_id,
        "snippet_identity": {
            "snippet_id": snippet.snippet_id,
            "split": snippet.split,
            "source_recording_id": snippet.source_recording_id,
            "source_session_id": snippet.source_session_id,
            "source_object_id": snippet.source_object_id,
            "object_type": snippet.object_type,
        },
        "footprint_spec_digest": footprint_spec_digest,
        "target_type_policy_digest": target_type_policy_digest,
        "seed": seed,
        "context_object_ids": list(context_object_ids),
        "transformed_poses": {
            "shape": list(canonical_poses.shape),
            "dtype": "<f4",
            "order": "C",
            "bytes_hex": canonical_poses.tobytes(order="C").hex(),
        },
    }
    canonical_payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"transform-{stable_digest(canonical_payload, size=16)}"


def _stable_reachability_target_id(
    *,
    object_type: str,
    transform_id: str,
    context_object_ids: tuple[str, ...],
) -> str:
    context_ids = set(context_object_ids)
    salt = 0
    while True:
        digest = stable_digest(
            "reachability-candidate-target-identity-v2",
            transform_id,
            salt,
            size=12,
        )
        target_id = f"generated::{object_type}::{digest}"
        if target_id not in context_ids:
            return target_id
        salt += 1


def transplant_reachability_candidate(
    snippet: MotionSnippet,
    *,
    candidate: ReachabilityCandidate,
    future_dt_s: float,
    future_steps: int,
    target_type_policy_digest: str,
    seed: int,
    context_object_ids: tuple[str, ...] | list[str] | set[str],
) -> TransplantedDynamicObject:
    """Apply one frozen reachability candidate to all 23 source samples.

    This path consumes the candidate's already-validated SE(2) transform.  It
    never estimates a direction from instantaneous velocity and performs no
    interpolation, temporal scaling, resampling, or extrapolation.
    """

    _validate_snippet(snippet)
    if not isinstance(candidate, ReachabilityCandidate):
        raise TypeError("candidate must be a ReachabilityCandidate")
    if snippet.snippet_id != candidate.identity.source_snippet_id:
        raise ValueError("snippet_id must match candidate source_snippet_id")
    if snippet.source_session_id != candidate.identity.source_session_id:
        raise ValueError(
            "source_session_id must match candidate source_session_id"
        )

    dt_s = _finite_real(future_dt_s, name="future_dt_s")
    if dt_s != MOTION_SNIPPET_SAMPLE_DT_S:
        raise ValueError("future_dt_s must equal 0.2")
    if isinstance(future_steps, (bool, np.bool_)) or not isinstance(
        future_steps, (int, np.integer)
    ):
        raise TypeError("future_steps must be an integer")
    if int(future_steps) != MOTION_SNIPPET_FUTURE_STEPS:
        raise ValueError("future_steps must equal 15")
    if (
        not isinstance(target_type_policy_digest, str)
        or not target_type_policy_digest
    ):
        raise ValueError("target_type_policy_digest must be a non-empty string")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")
    canonical_seed = int(seed)
    canonical_context_ids = _canonical_context_object_ids(context_object_ids)

    conflict_index = candidate.identity.conflict_index
    expected_conflict_time_s = (
        float(conflict_index + 1) * MOTION_SNIPPET_SAMPLE_DT_S
    )
    time_tolerance = (
        8.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(expected_conflict_time_s))
    )
    if (
        abs(candidate.identity.conflict_time_s - expected_conflict_time_s)
        > time_tolerance
    ):
        raise ValueError(
            "candidate conflict_time_s must equal (conflict_index + 1) * 0.2"
        )
    source_anchor_index = (
        MOTION_SNIPPET_CURRENT_INDEX + conflict_index + 1
    )
    if not 0 <= source_anchor_index < MOTION_SNIPPET_SAMPLE_COUNT:
        raise ValueError("candidate source_anchor_index must lie in [0, 22]")

    positions = snippet.positions.astype(np.float64)
    headings = snippet.headings.astype(np.float64)
    expected_source_delta = (
        positions[source_anchor_index] - positions[MOTION_SNIPPET_CURRENT_INDEX]
    )
    candidate_delta_bytes = np.ascontiguousarray(
        candidate.source_delta_xy, dtype=np.dtype(">f8")
    ).tobytes(order="C")
    expected_delta_bytes = np.ascontiguousarray(
        expected_source_delta, dtype=np.dtype(">f8")
    ).tobytes(order="C")
    if candidate_delta_bytes != expected_delta_bytes:
        raise ValueError(
            "candidate source_delta_xy must exactly match the source snippet anchor"
        )

    transformed_xy = (
        (positions - positions[MOTION_SNIPPET_CURRENT_INDEX])
        @ candidate.rotation_matrix.T
        + candidate.current_xy
    )
    transformed_headings = headings + candidate.rotation_rad
    transformed_poses = np.column_stack(
        (transformed_xy, transformed_headings)
    ).astype(np.float32)
    if not np.isfinite(transformed_poses).all():
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
    transform_id = _reachability_transform_id(
        snippet=snippet,
        candidate=candidate,
        footprint_spec_digest=footprint_digest,
        target_type_policy_digest=target_type_policy_digest,
        seed=canonical_seed,
        context_object_ids=canonical_context_ids,
        transformed_poses=transformed_poses,
    )
    target_id = _stable_reachability_target_id(
        object_type=snippet.object_type,
        transform_id=transform_id,
        context_object_ids=canonical_context_ids,
    )

    track_provenance = snippet.provenance.get("track_provenance", {})
    provenance = {
        "transform_algorithm_version": TRANSFORM_ALGORITHM_VERSION,
        "transform_id": transform_id,
        "reachability_candidate_id": candidate.candidate_id,
        "reachability_algorithm_version": BLIND_REACHABILITY_ALGORITHM_VERSION,
        "reachable_arc_schedule_version": REACHABLE_ARC_SCHEDULE_VERSION,
        "motion_snippet_layout_version": MOTION_SNIPPET_LAYOUT_VERSION,
        "snippet_id": snippet.snippet_id,
        "source_recording_id": snippet.source_recording_id,
        "source_session_id": snippet.source_session_id,
        "source_object_id": snippet.source_object_id,
        "source_body_name": snippet.provenance.get("source_body_name"),
        "raw_role": snippet.provenance.get("raw_role"),
        "geometry_source": track_provenance.get("geometry_source", "unknown"),
        "orientation_source": track_provenance.get(
            "orientation_source", "unknown"
        ),
        "source_start_timestamp": float(snippet.start_timestamp),
        "source_current_index": MOTION_SNIPPET_CURRENT_INDEX,
        "source_current_time_s": MOTION_SNIPPET_CURRENT_TIME_S,
        "source_anchor_index": source_anchor_index,
        "source_anchor_time_s": source_anchor_index * MOTION_SNIPPET_SAMPLE_DT_S,
        "source_delta_xy": candidate.source_delta_xy.tolist(),
        "candidate_current_xy": candidate.current_xy.tolist(),
        "conflict_point": candidate.conflict_point.tolist(),
        "rotation_rad": candidate.rotation_rad,
        "desired_crossing_direction": (
            candidate.desired_crossing_direction.tolist()
        ),
        "crossing_side": candidate.identity.crossing_side,
        "angle_offset_deg": candidate.identity.angle_offset_deg,
        "conflict_index": conflict_index,
        "conflict_time_s": candidate.identity.conflict_time_s,
        "time_scale": 1.0,
        "future_dt_s": MOTION_SNIPPET_SAMPLE_DT_S,
        "future_steps": MOTION_SNIPPET_FUTURE_STEPS,
        "base_state_id": candidate.identity.base_state_id,
        "trajectory_id": candidate.identity.trajectory_id,
        "target_type_policy_digest": target_type_policy_digest,
        "footprint_spec_digest": footprint_digest,
        "seed": canonical_seed,
        "context_object_ids": list(canonical_context_ids),
    }
    try:
        json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TransplantError("provenance_not_json_safe", str(exc)) from exc

    history_poses = np.array(
        transformed_poses[:MOTION_SNIPPET_HISTORY_STEPS],
        dtype=np.float32,
        order="C",
        copy=True,
    )
    current_pose = np.array(
        history_poses[-1], dtype=np.float32, order="C", copy=True
    )
    future_poses = np.array(
        transformed_poses[MOTION_SNIPPET_HISTORY_STEPS:],
        dtype=np.float32,
        order="C",
        copy=True,
    )
    return TransplantedDynamicObject(
        target_dynamic_object_id=target_id,
        source_object_id=snippet.source_object_id,
        snippet_id=snippet.snippet_id,
        object_type=snippet.object_type,
        footprint_spec=footprint_spec,
        footprint_spec_digest=footprint_digest,
        history_poses=history_poses,
        current_pose=current_pose,
        future_poses=future_poses,
        provenance=provenance,
    )
