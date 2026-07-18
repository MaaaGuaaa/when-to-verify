"""Deterministic reachable-arc proposals and conservative chord triage.

This module only constructs exact SE(2) start positions and cheap broad-phase
queries.  It deliberately does not transform full snippets or decide physical
collisions; those decisions belong to the continuous validator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np

from src.contracts import GridSpec
from src.geometry import (
    grid_bounds,
    grid_cell_centers,
    points_in_grid,
    segments_intersect,
    world_to_grid,
)
from src.utils.seeding import stable_digest

BLIND_REACHABILITY_ALGORITHM_VERSION = "blind_reachability_first_v1"
REACHABLE_ARC_SCHEDULE_VERSION = "reachable_arc_schedule_v1"

_TRIAGE_OUTCOMES = frozenset(("certified_clear", "unresolved"))


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric vector") from exc
    if source.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")
    if source.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers")
    result = np.asarray(source, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _unit_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    scale = float(np.max(np.abs(value)))
    if scale == 0.0:
        raise ValueError(f"{name} must be non-degenerate")
    scaled = value / scale
    norm = float(np.hypot(scaled[0], scaled[1]))
    result = scaled / norm
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must produce a finite direction")
    return np.asarray(result, dtype=np.float64)


def _immutable_float64(value: np.ndarray) -> np.ndarray:
    """Return float64 values backed by immutable bytes rather than caller memory."""

    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64).reshape(
        contiguous.shape
    )


def _float64_bytes_hex(value: Any) -> str:
    """Encode float64 values in canonical big-endian bytes for stable IDs."""

    array = np.ascontiguousarray(value, dtype=np.float64).astype(">f8", copy=False)
    return array.tobytes(order="C").hex()


def _text_identity_token(name: str, value: str) -> str:
    encoded = value.encode("utf-8")
    return f"{name}:{len(encoded)}:{encoded.hex()}"


@dataclass(frozen=True)
class ReachabilityIdentity:
    """Persisted inputs that identify one scheduled snippet/anchor proposal."""

    base_state_id: str
    trajectory_id: str
    source_snippet_id: str
    conflict_index: int
    conflict_time_s: float
    crossing_side: int
    angle_offset_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_state_id",
            _nonempty_string(self.base_state_id, name="base_state_id"),
        )
        object.__setattr__(
            self,
            "trajectory_id",
            _nonempty_string(self.trajectory_id, name="trajectory_id"),
        )
        object.__setattr__(
            self,
            "source_snippet_id",
            _nonempty_string(self.source_snippet_id, name="source_snippet_id"),
        )
        if isinstance(self.conflict_index, (bool, np.bool_)) or not isinstance(
            self.conflict_index, (Integral, np.integer)
        ):
            raise TypeError("conflict_index must be an integer")
        conflict_index = int(self.conflict_index)
        if conflict_index < 0:
            raise ValueError("conflict_index must be non-negative")
        object.__setattr__(self, "conflict_index", conflict_index)

        conflict_time = _finite_real(self.conflict_time_s, name="conflict_time_s")
        if conflict_time < 0.0:
            raise ValueError("conflict_time_s must be non-negative")
        object.__setattr__(self, "conflict_time_s", conflict_time)

        if isinstance(self.crossing_side, (bool, np.bool_)) or not isinstance(
            self.crossing_side, (Integral, np.integer)
        ):
            raise TypeError("crossing_side must be either -1 or 1")
        crossing_side = int(self.crossing_side)
        if crossing_side not in (-1, 1):
            raise ValueError("crossing_side must be either -1 or 1")
        object.__setattr__(self, "crossing_side", crossing_side)
        object.__setattr__(
            self,
            "angle_offset_deg",
            _finite_real(self.angle_offset_deg, name="angle_offset_deg"),
        )


@dataclass(frozen=True)
class ReachabilityCandidate:
    """One exact continuous start and rotation for a real source snippet."""

    candidate_id: str
    identity: ReachabilityIdentity
    rotation_rad: float
    rotation_matrix: np.ndarray = field(compare=False, hash=False)
    current_xy: np.ndarray = field(compare=False, hash=False)
    conflict_point: np.ndarray = field(compare=False, hash=False)
    source_delta_xy: np.ndarray = field(compare=False, hash=False)
    desired_crossing_direction: np.ndarray = field(compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _nonempty_string(self.candidate_id, name="candidate_id"),
        )
        if not isinstance(self.identity, ReachabilityIdentity):
            raise TypeError("identity must be a ReachabilityIdentity")
        rotation_rad = _finite_real(self.rotation_rad, name="rotation_rad")
        object.__setattr__(self, "rotation_rad", rotation_rad)

        rotation_matrix = np.asarray(self.rotation_matrix)
        if rotation_matrix.shape != (2, 2):
            raise ValueError("rotation_matrix must have shape (2, 2)")
        if rotation_matrix.dtype.kind not in "iuf":
            raise TypeError("rotation_matrix must contain real numbers")
        rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
        if not np.all(np.isfinite(rotation_matrix)):
            raise ValueError("rotation_matrix must contain only finite values")

        current = _finite_vector(self.current_xy, name="current_xy")
        conflict = _finite_vector(self.conflict_point, name="conflict_point")
        source_delta = _finite_vector(self.source_delta_xy, name="source_delta_xy")
        source_direction = _unit_vector(source_delta, name="source_delta_xy")
        desired = _finite_vector(
            self.desired_crossing_direction,
            name="desired_crossing_direction",
        )
        desired_norm = float(np.hypot(desired[0], desired[1]))
        if not np.isclose(
            desired_norm,
            1.0,
            rtol=0.0,
            atol=8.0 * np.finfo(np.float64).eps,
        ):
            raise ValueError("desired_crossing_direction must have unit norm")

        cosine = np.cos(rotation_rad)
        sine = np.sin(rotation_rad)
        expected_rotation = np.array(
            [[cosine, -sine], [sine, cosine]], dtype=np.float64
        )
        if not np.allclose(
            rotation_matrix, expected_rotation, rtol=1e-13, atol=1e-15
        ):
            raise ValueError("rotation_matrix must match rotation_rad")
        if not np.allclose(
            current + rotation_matrix @ source_delta,
            conflict,
            rtol=1e-13,
            atol=1e-15,
        ):
            raise ValueError("current_xy must map source anchor to conflict_point")
        if not np.allclose(
            rotation_matrix @ source_direction,
            desired,
            rtol=1e-13,
            atol=1e-15,
        ):
            raise ValueError(
                "rotation must map source_delta_xy to desired_crossing_direction"
            )

        expected_candidate_id = _candidate_identifier(
            identity=self.identity,
            rotation_rad=rotation_rad,
            rotation_matrix=rotation_matrix,
            current_xy=current,
            conflict_point=conflict,
            source_delta_xy=source_delta,
            desired_crossing_direction=desired,
        )
        if self.candidate_id != expected_candidate_id:
            raise ValueError(
                "candidate_id must match the complete versioned identity and geometry"
            )

        object.__setattr__(
            self, "rotation_matrix", _immutable_float64(rotation_matrix)
        )
        object.__setattr__(self, "current_xy", _immutable_float64(current))
        object.__setattr__(self, "conflict_point", _immutable_float64(conflict))
        object.__setattr__(
            self, "source_delta_xy", _immutable_float64(source_delta)
        )
        object.__setattr__(
            self,
            "desired_crossing_direction",
            _immutable_float64(desired),
        )


@dataclass(frozen=True)
class ChordTriage:
    """Conservative broad-phase result; never a physical collision verdict."""

    outcome: Literal["certified_clear", "unresolved"]
    chord_deviation_bound_m: float
    candidate_id: str
    identity: ReachabilityIdentity

    def __post_init__(self) -> None:
        if self.outcome not in _TRIAGE_OUTCOMES:
            raise ValueError(
                "outcome must be either 'certified_clear' or 'unresolved'"
            )
        deviation = _finite_real(
            self.chord_deviation_bound_m,
            name="chord_deviation_bound_m",
        )
        if deviation < 0.0:
            raise ValueError("chord_deviation_bound_m must be non-negative")
        object.__setattr__(self, "chord_deviation_bound_m", deviation)
        object.__setattr__(
            self,
            "candidate_id",
            _nonempty_string(self.candidate_id, name="candidate_id"),
        )
        if not isinstance(self.identity, ReachabilityIdentity):
            raise TypeError("identity must be a ReachabilityIdentity")


def scheduled_crossing_directions(
    normal_xy: Any,
    *,
    maximum_angle_deg: float,
    angle_step_deg: float,
) -> np.ndarray:
    """Return unit directions at offsets ``-maximum`` through ``+maximum``.

    Offsets are constructed from integer step indices, so order and values do
    not depend on a random seed, Python hashing, or worker completion order.
    """

    normal = _unit_vector(
        _finite_vector(normal_xy, name="normal_xy"), name="normal_xy"
    )
    maximum = _finite_real(maximum_angle_deg, name="maximum_angle_deg")
    step = _finite_real(angle_step_deg, name="angle_step_deg")
    if maximum < 0.0:
        raise ValueError("maximum_angle_deg must be non-negative")
    if step <= 0.0:
        raise ValueError("angle_step_deg must be positive")

    ratio = maximum / step
    nearest_steps = int(round(ratio))
    reconstructed = float(nearest_steps) * step
    tolerance = 8.0 * np.finfo(np.float64).eps * max(
        1.0, abs(maximum), abs(reconstructed)
    )
    if abs(maximum - reconstructed) > tolerance:
        raise ValueError("maximum_angle_deg must be an integer multiple of angle_step_deg")

    step_indices = np.arange(-nearest_steps, nearest_steps + 1, dtype=np.int64)
    angles = np.deg2rad(step_indices.astype(np.float64) * step)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    directions = np.column_stack(
        (
            cosine * normal[0] - sine * normal[1],
            sine * normal[0] + cosine * normal[1],
        )
    )
    lengths = np.hypot(directions[:, 0], directions[:, 1])
    directions = directions / lengths[:, None]
    return _immutable_float64(directions)


def _candidate_identifier(
    *,
    identity: ReachabilityIdentity,
    rotation_rad: float,
    rotation_matrix: np.ndarray,
    current_xy: np.ndarray,
    conflict_point: np.ndarray,
    source_delta_xy: np.ndarray,
    desired_crossing_direction: np.ndarray,
) -> str:
    digest = stable_digest(
        f"algorithm_version={BLIND_REACHABILITY_ALGORITHM_VERSION}",
        f"schedule_version={REACHABLE_ARC_SCHEDULE_VERSION}",
        _text_identity_token("base_state_id", identity.base_state_id),
        _text_identity_token("trajectory_id", identity.trajectory_id),
        _text_identity_token("source_snippet_id", identity.source_snippet_id),
        f"conflict_index={identity.conflict_index}",
        f"conflict_time_s_f64={_float64_bytes_hex(identity.conflict_time_s)}",
        f"crossing_side={identity.crossing_side}",
        f"angle_offset_deg_f64={_float64_bytes_hex(identity.angle_offset_deg)}",
        f"rotation_rad_f64={_float64_bytes_hex(rotation_rad)}",
        f"rotation_matrix_f64={_float64_bytes_hex(rotation_matrix)}",
        f"current_xy_f64={_float64_bytes_hex(current_xy)}",
        f"conflict_point_f64={_float64_bytes_hex(conflict_point)}",
        f"source_delta_xy_f64={_float64_bytes_hex(source_delta_xy)}",
        "desired_crossing_direction_f64="
        f"{_float64_bytes_hex(desired_crossing_direction)}",
        size=16,
    )
    return f"reachability-{digest}"


def build_reachability_candidate(
    *,
    conflict_point: Any,
    source_current_xy: Any,
    source_anchor_xy: Any,
    desired_crossing_direction: Any,
    identity: ReachabilityIdentity,
) -> ReachabilityCandidate:
    """Construct the unique SE(2) start that maps one source anchor to conflict."""

    if not isinstance(identity, ReachabilityIdentity):
        raise TypeError("identity must be a ReachabilityIdentity")
    conflict = _finite_vector(conflict_point, name="conflict_point")
    source_current = _finite_vector(source_current_xy, name="source_current_xy")
    source_anchor = _finite_vector(source_anchor_xy, name="source_anchor_xy")
    source_delta = source_anchor - source_current
    if not np.all(np.isfinite(source_delta)):
        raise ValueError("source delta must contain only finite values")
    _unit_vector(source_delta, name="source anchor-current delta")
    desired = _unit_vector(
        _finite_vector(
            desired_crossing_direction,
            name="desired_crossing_direction",
        ),
        name="desired_crossing_direction",
    )

    rotation_rad = float(
        np.arctan2(desired[1], desired[0])
        - np.arctan2(source_delta[1], source_delta[0])
    )
    cosine = np.cos(rotation_rad)
    sine = np.sin(rotation_rad)
    rotation_matrix = np.array(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    transformed_delta = rotation_matrix @ source_delta
    current = conflict - transformed_delta
    if not np.all(np.isfinite(current)):
        raise ValueError("exact current_xy must contain only finite values")

    candidate_id = _candidate_identifier(
        identity=identity,
        rotation_rad=rotation_rad,
        rotation_matrix=rotation_matrix,
        current_xy=current,
        conflict_point=conflict,
        source_delta_xy=source_delta,
        desired_crossing_direction=desired,
    )
    return ReachabilityCandidate(
        candidate_id=candidate_id,
        identity=identity,
        rotation_rad=rotation_rad,
        rotation_matrix=rotation_matrix,
        current_xy=current,
        conflict_point=conflict,
        source_delta_xy=source_delta,
        desired_crossing_direction=desired,
    )


def candidate_queries_mask(
    candidate: ReachabilityCandidate,
    mask: Any,
    grid: GridSpec,
) -> bool:
    """Query only the cell containing the candidate's exact continuous start."""

    if not isinstance(candidate, ReachabilityCandidate):
        raise TypeError("candidate must be a ReachabilityCandidate")
    grid_bounds(grid)
    mask_array = np.asarray(mask)
    if mask_array.shape != (grid.height, grid.width):
        raise ValueError("mask shape must match grid")
    if mask_array.dtype != np.bool_:
        raise TypeError("mask must have boolean dtype")
    row, column = world_to_grid(candidate.current_xy, grid)
    return bool(mask_array[row, column])


def _obstacle_mask(value: Any, grid: GridSpec) -> np.ndarray:
    occupancy = np.asarray(value)
    if occupancy.shape != (grid.height, grid.width):
        raise ValueError("obstacle_occupancy shape must match grid")
    if occupancy.dtype.kind not in "biuf":
        raise TypeError("obstacle_occupancy must contain real values")
    if not np.all(np.isfinite(occupancy)):
        raise ValueError("obstacle_occupancy must contain only finite values")
    return np.asarray(occupancy != 0, dtype=bool)


def _point_to_cell_distance(
    point: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> float:
    dx = max(x_min - float(point[0]), 0.0, float(point[0]) - x_max)
    dy = max(y_min - float(point[1]), 0.0, float(point[1]) - y_max)
    return float(np.hypot(dx, dy))


def _segment_to_cell_distance(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    *,
    half_cell: float,
    unit_segment: np.ndarray,
    segment_length: float,
) -> float:
    """Return exact distance from a finite segment to one closed square cell."""

    x_min = float(center[0]) - half_cell
    x_max = float(center[0]) + half_cell
    y_min = float(center[1]) - half_cell
    y_max = float(center[1]) + half_cell
    corners = np.array(
        (
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ),
        dtype=np.float64,
    )
    for corner, next_corner in zip(corners, np.roll(corners, -1, axis=0)):
        if segments_intersect(start, end, corner, next_corner):
            return 0.0

    endpoint_distance = min(
        _point_to_cell_distance(
            start,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        ),
        _point_to_cell_distance(
            end,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        ),
    )
    projections = (corners - start) @ unit_segment
    projections = np.clip(projections, 0.0, segment_length)
    nearest = start + projections[:, None] * unit_segment
    corner_distances = np.hypot(
        corners[:, 0] - nearest[:, 0],
        corners[:, 1] - nearest[:, 1],
    )
    return min(endpoint_distance, float(np.min(corner_distances)))


def triage_chord(
    candidate: ReachabilityCandidate,
    *,
    obstacle_occupancy: Any,
    grid: GridSpec,
    footprint_radius_m: float,
    chord_deviation_bound_m: float,
) -> ChordTriage:
    """Conservatively triage the current-to-conflict chord against occupancy.

    The test uses the exact distance between the chord segment and each closed
    occupied grid square.  It is exact for this coarse tube representation, but
    it does not inspect or replace the real transformed snippet trajectory.
    """

    if not isinstance(candidate, ReachabilityCandidate):
        raise TypeError("candidate must be a ReachabilityCandidate")
    grid_bounds(grid)
    footprint_radius = _finite_real(
        footprint_radius_m, name="footprint_radius_m"
    )
    if footprint_radius <= 0.0:
        raise ValueError("footprint_radius_m must be positive")
    deviation = _finite_real(
        chord_deviation_bound_m, name="chord_deviation_bound_m"
    )
    if deviation < 0.0:
        raise ValueError("chord_deviation_bound_m must be non-negative")
    occupied = _obstacle_mask(obstacle_occupancy, grid)

    endpoints = np.stack((candidate.current_xy, candidate.conflict_point))
    if not np.all(points_in_grid(endpoints, grid)):
        raise ValueError("chord endpoints must lie inside the grid bounds")

    outcome: Literal["certified_clear", "unresolved"] = "certified_clear"
    if np.any(occupied):
        occupied_centers = grid_cell_centers(grid)[occupied]
        segment = candidate.conflict_point - candidate.current_xy
        segment_length = float(np.hypot(segment[0], segment[1]))
        if not np.isfinite(segment_length) or segment_length <= 0.0:
            raise ValueError("candidate chord must be non-degenerate and finite")
        unit_segment = segment / segment_length
        tube_radius = footprint_radius + deviation
        half_cell = 0.5 * float(grid.resolution_m)
        possible = (
            (occupied_centers[:, 0] + half_cell >= np.min(endpoints[:, 0]) - tube_radius)
            & (
                occupied_centers[:, 0] - half_cell
                <= np.max(endpoints[:, 0]) + tube_radius
            )
            & (
                occupied_centers[:, 1] + half_cell
                >= np.min(endpoints[:, 1]) - tube_radius
            )
            & (
                occupied_centers[:, 1] - half_cell
                <= np.max(endpoints[:, 1]) + tube_radius
            )
        )
        for center in occupied_centers[possible]:
            distance = _segment_to_cell_distance(
                candidate.current_xy,
                candidate.conflict_point,
                center,
                half_cell=half_cell,
                unit_segment=unit_segment,
                segment_length=segment_length,
            )
            if distance <= tube_radius:
                outcome = "unresolved"
                break

    return ChordTriage(
        outcome=outcome,
        chord_deviation_bound_m=deviation,
        candidate_id=candidate.candidate_id,
        identity=candidate.identity,
    )


__all__ = (
    "BLIND_REACHABILITY_ALGORITHM_VERSION",
    "REACHABLE_ARC_SCHEDULE_VERSION",
    "ReachabilityIdentity",
    "ReachabilityCandidate",
    "ChordTriage",
    "scheduled_crossing_directions",
    "build_reachability_candidate",
    "candidate_queries_mask",
    "triage_chord",
)
