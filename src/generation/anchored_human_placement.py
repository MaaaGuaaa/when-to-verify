"""M5 rigid human anchoring and synchronized centerline geometry."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from src.contracts import ARRAY_DTYPE, BaseState, OracleContext, build_grid_spec
from src.datasets.long_snippet_library import (
    LONG_MOTION_SNIPPET_LAYOUT,
    LongMotionSnippet,
)
from src.geometry import (
    StaticOccluder,
    footprint_aabb,
    rasterize_occluder,
    segment_intersects_occluder,
    wrap_angle,
)
from src.utils.seeding import derive_seed

from .event_contracts import footprint_from_spec
from .history_visibility import (
    SeenThenOccludedHistoryAssessment,
    classify_sop05r_seen_then_occluded_history,
)
from .occluder_sampler import (
    swept_footprint_intersects_occupancy,
    synchronized_sweeps_intersect,
)
from .sop05r_contracts import (
    SOP05R_TEB_OCCLUSION_VERSION,
    Sop05rTebConfig,
)
from .sop05r_teb_templates import Sop05rTebTaskTemplate


PLACEMENT_SELECTION_MODES = ("seen_first", "h0_hidden")
_CANDIDATE_SEARCH_BY_SELECTION_MODE = {
    "seen_first": "synchronized_half_plane_step_seen_then_occlude_v5",
    "h0_hidden": "synchronized_half_plane_step_h0_hidden_v1",
}


def _readonly_array(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"expected finite array with shape {shape}")
    result = np.array(array, dtype=dtype, order="C", copy=True)
    return np.frombuffer(result.tobytes(), dtype=dtype).reshape(shape)


@dataclass(frozen=True)
class CollisionAnchor:
    """One exact equality between a route endpoint and snippet sample time."""

    route_sample_index: int
    route_time_s: float
    world_position_xy: np.ndarray
    snippet_anchor_index: int
    snippet_time_s: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.route_sample_index, bool)
            or not isinstance(self.route_sample_index, int)
            or self.route_sample_index < 0
        ):
            raise ValueError("route_sample_index must be a nonnegative integer")
        if (
            isinstance(self.snippet_anchor_index, bool)
            or not isinstance(self.snippet_anchor_index, int)
            or self.snippet_anchor_index < 0
        ):
            raise ValueError("snippet_anchor_index must be a nonnegative integer")
        if not np.isfinite((self.route_time_s, self.snippet_time_s)).all():
            raise ValueError("anchor times must be finite")
        if not np.isclose(
            self.route_time_s, self.snippet_time_s, rtol=0.0, atol=1e-6
        ):
            raise ValueError("route and snippet anchor times must match")
        object.__setattr__(
            self,
            "world_position_xy",
            _readonly_array(
                self.world_position_xy,
                shape=(2,),
                dtype=np.dtype(np.float64),
            ),
        )


@dataclass(frozen=True)
class VisibilityGuidedRotationAngles:
    """Deterministic half-plane margins beyond one occluder guide ray."""

    theta_align_rad: float
    signed_ab_rad: float
    half_plane_side: int
    angles_rad: tuple[float, ...]
    angular_margins_deg: tuple[float, ...]
    rejection_reason: str | None


@dataclass(frozen=True)
class _VisibilityGuidedRotationBatch:
    sweep: VisibilityGuidedRotationAngles
    angles_rad: np.ndarray
    transformed_poses: np.ndarray
    translations_xy_m: np.ndarray
    blocked: np.ndarray
    blocker_indices: np.ndarray
    history_assessments: tuple[SeenThenOccludedHistoryAssessment, ...]
    preferred_indices: tuple[int, ...]
    fallback_indices: tuple[int, ...]


@dataclass(frozen=True)
class CenterlineOcclusionWitness:
    """One synchronized pre-anchor segment blocked by a represented occluder."""

    version: str
    time_s: float
    sample_index: int
    robot_position_xy: np.ndarray
    target_position_xy: np.ndarray
    blocking_occluder_id: str

    def __post_init__(self) -> None:
        if self.version != SOP05R_TEB_OCCLUSION_VERSION:
            raise ValueError("centerline witness version mismatch")
        if (
            isinstance(self.sample_index, bool)
            or not isinstance(self.sample_index, int)
            or self.sample_index < 0
            or not np.isfinite(self.time_s)
        ):
            raise ValueError("centerline witness time or index is invalid")
        if not isinstance(self.blocking_occluder_id, str) or not self.blocking_occluder_id:
            raise ValueError("centerline witness must identify a blocking occluder")
        for name in ("robot_position_xy", "target_position_xy"):
            object.__setattr__(
                self,
                name,
                _readonly_array(
                    getattr(self, name),
                    shape=(2,),
                    dtype=np.dtype(np.float64),
                ),
            )


@dataclass(frozen=True)
class AnchoredHumanPlacement:
    """A full long40 human motion rigidly joined to one route anchor."""

    source_snippet_id: str
    anchor: CollisionAnchor
    rotation_rad: float
    translation_xy_m: np.ndarray
    spatial_scale: float
    temporal_scale: float
    history_poses: np.ndarray
    current_pose: np.ndarray
    future_poses: np.ndarray
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.spatial_scale != 1.0:
            raise ValueError("M5 spatial_scale must equal 1.0")
        if not np.isfinite((self.rotation_rad, self.temporal_scale)).all():
            raise ValueError("placement rotation and temporal scale must be finite")
        object.__setattr__(
            self,
            "translation_xy_m",
            _readonly_array(
                self.translation_xy_m,
                shape=(2,),
                dtype=np.dtype(np.float64),
            ),
        )
        object.__setattr__(
            self,
            "history_poses",
            _readonly_array(self.history_poses, shape=(8, 3)),
        )
        object.__setattr__(
            self,
            "current_pose",
            _readonly_array(self.current_pose, shape=(3,)),
        )
        object.__setattr__(
            self,
            "future_poses",
            _readonly_array(self.future_poses, shape=(32, 3)),
        )
        if not np.array_equal(self.current_pose, self.history_poses[-1]):
            raise ValueError("current_pose must equal the final history pose")


@dataclass(frozen=True)
class AnchoredPlacementResult:
    """Accepted M5 first-fit output with immutable visibility evidence."""

    placement: AnchoredHumanPlacement
    witness: CenterlineOcclusionWitness
    visibility: SeenThenOccludedHistoryAssessment
    attempted_candidates: int
    rejection_counts: Mapping[str, int]
    candidate_counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AnchoredPlacementEvaluation:
    """M5 first-fit result or its bounded-search rejection evidence."""

    result: AnchoredPlacementResult | None
    rejection_reason: str | None
    attempted_candidates: int
    rejection_counts: Mapping[str, int]
    candidate_counts: Mapping[str, int] = field(default_factory=dict)


def apply_anchored_rigid_transform(
    *,
    source_poses: object,
    source_velocities: object,
    anchor: CollisionAnchor,
    rotation_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate a full snippet around an anchor and translate it to route contact."""

    if not isinstance(anchor, CollisionAnchor):
        raise TypeError("anchor must be a CollisionAnchor")
    poses = np.asarray(source_poses, dtype=np.float64)
    velocities = np.asarray(source_velocities, dtype=np.float64)
    if (
        poses.ndim != 2
        or poses.shape[1] != 3
        or velocities.shape != (poses.shape[0], 2)
        or not np.isfinite(poses).all()
        or not np.isfinite(velocities).all()
    ):
        raise ValueError("source poses and velocities must be finite aligned arrays")
    if not 0 <= anchor.snippet_anchor_index < poses.shape[0]:
        raise ValueError("snippet_anchor_index is outside source poses")
    if not np.isfinite(rotation_rad):
        raise ValueError("rotation_rad must be finite")
    cosine, sine = np.cos(rotation_rad), np.sin(rotation_rad)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    source_anchor_xy = poses[anchor.snippet_anchor_index, :2]
    translation = anchor.world_position_xy - rotation @ source_anchor_xy
    transformed = np.empty_like(poses)
    transformed[:, :2] = poses[:, :2] @ rotation.T + translation
    transformed[:, 2] = wrap_angle(poses[:, 2] + rotation_rad)
    transformed_velocities = velocities @ rotation.T
    return (
        np.asarray(transformed, dtype=np.float32),
        np.asarray(transformed_velocities, dtype=np.float32),
        np.asarray(translation, dtype=np.float64),
    )


def resample_long_motion_snippet(
    snippet: LongMotionSnippet,
    *,
    time_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample one long40 snippet about its decision/current sample."""

    layout = LONG_MOTION_SNIPPET_LAYOUT
    sample_count = int(layout["sample_count"])
    current_index = int(layout["current_index"])
    dt_s = float(layout["sample_dt_s"])
    if not isinstance(snippet, LongMotionSnippet):
        raise TypeError("snippet must be a LongMotionSnippet")
    if (
        snippet.positions.shape != (sample_count, 2)
        or snippet.velocities.shape != (sample_count, 2)
        or snippet.headings.shape != (sample_count,)
        or snippet.positions.dtype != ARRAY_DTYPE
        or snippet.velocities.dtype != ARRAY_DTYPE
        or snippet.headings.dtype != ARRAY_DTYPE
        or not np.isfinite(snippet.positions).all()
        or not np.isfinite(snippet.velocities).all()
        or not np.isfinite(snippet.headings).all()
    ):
        raise ValueError("long40 snippet arrays violate the frozen layout")
    if isinstance(time_scale, (bool, np.bool_)) or not np.isfinite(time_scale):
        raise TypeError("time_scale must be a finite real number")
    scale = float(time_scale)
    if scale <= 0.0:
        raise ValueError("time_scale must be positive")
    source_times = np.arange(sample_count, dtype=np.float64) * dt_s
    current_time_s = current_index * dt_s
    query_times = current_time_s + (source_times - current_time_s) * scale
    if (
        float(np.min(query_times)) < float(source_times[0]) - 1e-9
        or float(np.max(query_times)) > float(source_times[-1]) + 1e-9
    ):
        raise ValueError("source_extrapolation_required")
    positions = np.column_stack(
        (
            np.interp(query_times, source_times, snippet.positions[:, 0]),
            np.interp(query_times, source_times, snippet.positions[:, 1]),
        )
    )
    headings = wrap_angle(
        np.interp(
            query_times,
            source_times,
            np.unwrap(snippet.headings.astype(np.float64)),
        )
    )
    velocities = np.column_stack(
        (
            np.interp(query_times, source_times, snippet.velocities[:, 0]),
            np.interp(query_times, source_times, snippet.velocities[:, 1]),
        )
    )
    velocities *= scale
    return (
        np.column_stack((positions, headings)).astype(ARRAY_DTYPE),
        velocities.astype(ARRAY_DTYPE),
    )


def synchronized_centerline_blocking(
    robot_positions_xy: object,
    target_positions_xy: object,
    occluders: tuple[StaticOccluder, ...],
    *,
    epsilon_m: float,
) -> tuple[np.ndarray, tuple[str | None, ...]]:
    """Return synchronized centerline blockage and first blocker per sample."""

    robot_positions = np.asarray(robot_positions_xy, dtype=np.float64)
    target_positions = np.asarray(target_positions_xy, dtype=np.float64)
    if (
        robot_positions.ndim != 2
        or robot_positions.shape[1] != 2
        or target_positions.shape != robot_positions.shape
        or not np.isfinite(robot_positions).all()
        or not np.isfinite(target_positions).all()
    ):
        raise ValueError("robot and target positions must be finite [N,2] arrays")
    if not occluders:
        raise ValueError("occluders must not be empty")
    blocked = np.zeros(robot_positions.shape[0], dtype=np.bool_)
    blocker_ids: list[str | None] = [None] * robot_positions.shape[0]
    for occluder in occluders:
        intersects = segment_intersects_occluder(
            occluder,
            robot_positions,
            target_positions,
            epsilon_m=epsilon_m,
        )
        newly_blocked = intersects & ~blocked
        for index in np.flatnonzero(newly_blocked):
            blocker_ids[int(index)] = occluder.occluder_id
        blocked |= intersects
    return blocked, tuple(blocker_ids)


def _batched_centerline_blocking(
    robot_positions_xy: np.ndarray,
    target_positions_by_angle_xy: np.ndarray,
    occluders: tuple[StaticOccluder, ...],
    *,
    epsilon_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return blockage and first-blocker indices for an `[A,T,2]` batch."""

    robot_positions = np.asarray(robot_positions_xy, dtype=np.float64)
    targets = np.asarray(target_positions_by_angle_xy, dtype=np.float64)
    if (
        robot_positions.ndim != 2
        or robot_positions.shape[1] != 2
        or targets.ndim != 3
        or targets.shape[1:] != robot_positions.shape
        or not np.isfinite(robot_positions).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("batched centerlines must be finite [A,T,2] arrays")
    if not occluders:
        raise ValueError("occluders must not be empty")
    angle_count, sample_count = targets.shape[:2]
    starts = np.broadcast_to(
        robot_positions[None, :, :],
        targets.shape,
    ).reshape(angle_count * sample_count, 2)
    ends = targets.reshape(angle_count * sample_count, 2)
    blocked = np.zeros((angle_count, sample_count), dtype=np.bool_)
    blocker_indices = np.full((angle_count, sample_count), -1, dtype=np.int16)
    for occluder_index, occluder in enumerate(occluders):
        intersects = segment_intersects_occluder(
            occluder,
            starts,
            ends,
            epsilon_m=epsilon_m,
        ).reshape(angle_count, sample_count)
        newly_blocked = intersects & ~blocked
        blocker_indices[newly_blocked] = occluder_index
        blocked |= intersects
    return blocked, blocker_indices


def _construct_visibility_guided_rotation_batch(
    *,
    source_poses: object,
    anchor: CollisionAnchor,
    robot_positions_xy: object,
    occluders: tuple[StaticOccluder, ...],
    decision_index: int,
    guide_history_index: int,
    minimum_visible_history_frames: int = 4,
    minimum_occluded_history_frames: int = 1,
    occluder_angular_margin_step_deg: float,
    search_stage: str = "primary",
    epsilon_m: float = 0.01,
) -> _VisibilityGuidedRotationBatch:
    """Evaluate half-plane margin angles against exact centerlines."""

    poses = np.asarray(source_poses, dtype=np.float64)
    robot_positions = np.asarray(robot_positions_xy, dtype=np.float64)
    if (
        poses.ndim != 2
        or poses.shape[1] != 3
        or robot_positions.shape != (poses.shape[0], 2)
        or not np.isfinite(poses).all()
        or not np.isfinite(robot_positions).all()
    ):
        raise ValueError("source poses and robot positions must be finite aligned arrays")
    if isinstance(decision_index, bool) or not isinstance(decision_index, int):
        raise ValueError("decision index must lie within source poses")
    sweep = construct_visibility_guided_rotation_angles(
        source_poses=poses,
        anchor=anchor,
        robot_positions_xy=robot_positions,
        occluders=occluders,
        guide_history_index=guide_history_index,
        occluder_angular_margin_step_deg=occluder_angular_margin_step_deg,
        search_stage=search_stage,
        epsilon_m=epsilon_m,
    )
    angles = np.asarray(sweep.angles_rad, dtype=np.float64)
    if not angles.size:
        return _VisibilityGuidedRotationBatch(
            sweep=sweep,
            angles_rad=angles,
            transformed_poses=np.empty((0, poses.shape[0], 3), dtype=np.float64),
            translations_xy_m=np.empty((0, 2), dtype=np.float64),
            blocked=np.empty((0, poses.shape[0]), dtype=np.bool_),
            blocker_indices=np.empty((0, poses.shape[0]), dtype=np.int16),
            history_assessments=(),
            preferred_indices=(),
            fallback_indices=(),
        )
    zero_velocities = np.zeros((poses.shape[0], 2), dtype=np.float64)
    transformed_poses: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for angle in angles.tolist():
        transformed, _, translation = apply_anchored_rigid_transform(
            source_poses=poses,
            source_velocities=zero_velocities,
            anchor=anchor,
            rotation_rad=float(angle),
        )
        transformed_poses.append(transformed)
        translations.append(translation)
    transformed_pose_batch = np.stack(transformed_poses, axis=0)
    blocked_by_angle, blocker_indices = _batched_centerline_blocking(
        robot_positions,
        transformed_pose_batch[:, :, :2],
        occluders,
        epsilon_m=epsilon_m,
    )
    assessments: list[SeenThenOccludedHistoryAssessment] = []
    preferred: list[int] = []
    fallback: list[int] = []
    for angle_index, blocked in enumerate(blocked_by_angle):
        history_assessment = classify_sop05r_seen_then_occluded_history(
            blocked,
            decision_index=decision_index,
            minimum_visible_frames=minimum_visible_history_frames,
            minimum_occluded_frames=minimum_occluded_history_frames,
        )
        assessments.append(history_assessment)
        if not history_assessment.eligible:
            continue
        if history_assessment.preferred:
            preferred.append(angle_index)
        else:
            fallback.append(angle_index)
    return _VisibilityGuidedRotationBatch(
        sweep=sweep,
        angles_rad=angles,
        transformed_poses=transformed_pose_batch,
        translations_xy_m=np.stack(translations, axis=0),
        blocked=blocked_by_angle,
        blocker_indices=blocker_indices,
        history_assessments=tuple(assessments),
        preferred_indices=tuple(preferred),
        fallback_indices=tuple(fallback),
    )


def _occluder_center_xy(occluder: StaticOccluder) -> np.ndarray:
    """Return the world center of one represented primitive."""

    if hasattr(occluder, "center_xy"):
        return np.asarray(occluder.center_xy, dtype=np.float64)
    return np.asarray(occluder.pose[:2], dtype=np.float64)


def _representative_occluder_center_xy(
    *,
    collision_xy: np.ndarray,
    guide_robot_xy: np.ndarray,
    occluders: tuple[StaticOccluder, ...],
) -> np.ndarray:
    """Choose the primitive center nearest the robot guide segment."""

    guide_vector = guide_robot_xy - collision_xy
    guide_length_squared = float(np.dot(guide_vector, guide_vector))
    distances: list[tuple[float, str, np.ndarray]] = []
    for occluder in occluders:
        center = _occluder_center_xy(occluder)
        fraction = float(
            np.clip(
                np.dot(center - collision_xy, guide_vector) / guide_length_squared,
                0.0,
                1.0,
            )
        )
        nearest = collision_xy + fraction * guide_vector
        distances.append(
            (
                float(np.sum((center - nearest) ** 2)),
                occluder.occluder_id,
                center,
            )
        )
    nearest_distance = min(item[0] for item in distances)
    _, _, center = min(
        (
            item
            for item in distances
            if np.isclose(item[0], nearest_distance, rtol=0.0, atol=1e-12)
        ),
        key=lambda item: item[1],
    )
    return center


def construct_visibility_guided_rotation_angles(
    *,
    source_poses: object,
    anchor: CollisionAnchor,
    robot_positions_xy: object,
    occluders: tuple[StaticOccluder, ...],
    guide_history_index: int,
    occluder_angular_margin_step_deg: object,
    search_stage: str = "primary",
    epsilon_m: float = 0.01,
) -> VisibilityGuidedRotationAngles:
    """Construct the outward primary sweep or its complementary fallback."""

    poses = np.asarray(source_poses, dtype=np.float64)
    robot_positions = np.asarray(robot_positions_xy, dtype=np.float64)
    if (
        poses.ndim != 2
        or poses.shape[1] != 3
        or robot_positions.shape != (poses.shape[0], 2)
        or not np.isfinite(poses).all()
        or not np.isfinite(robot_positions).all()
    ):
        raise ValueError("source poses and robot positions must be finite aligned arrays")
    if (
        isinstance(guide_history_index, bool)
        or not isinstance(guide_history_index, int)
        or not 0 <= guide_history_index < poses.shape[0]
        or not 0 <= anchor.snippet_anchor_index < poses.shape[0]
    ):
        raise ValueError("guide and anchor indices must lie within source poses")
    if not occluders:
        raise ValueError("half-plane margins require a represented occluder")
    if search_stage not in {"primary", "secondary"}:
        raise ValueError("search stage must be primary or secondary")
    if isinstance(epsilon_m, (bool, np.bool_)) or not isinstance(
        epsilon_m, (float, int, np.floating, np.integer)
    ):
        raise TypeError("epsilon_m must be a real value")
    epsilon = float(epsilon_m)
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon_m must be finite and nonnegative")
    if (
        isinstance(occluder_angular_margin_step_deg, (bool, np.bool_))
        or not isinstance(
            occluder_angular_margin_step_deg,
            (float, int, np.floating, np.integer),
        )
    ):
        raise TypeError("occluder angular margin step must be a real value")
    margin_step_deg = float(occluder_angular_margin_step_deg)
    if (
        not np.isfinite(margin_step_deg)
        or margin_step_deg <= 0.0
        or margin_step_deg >= 180.0
        or not np.isclose(
            180.0 / margin_step_deg,
            round(180.0 / margin_step_deg),
        )
    ):
        raise ValueError("occluder angular margin step must evenly divide 180 degrees")

    source_vector = (
        poses[guide_history_index, :2]
        - poses[anchor.snippet_anchor_index, :2]
    )
    guide_robot_xy = robot_positions[guide_history_index]
    robot_guide_vector = guide_robot_xy - anchor.world_position_xy
    if (
        np.linalg.norm(source_vector) <= epsilon
        or np.linalg.norm(robot_guide_vector) <= epsilon
    ):
        return VisibilityGuidedRotationAngles(
            theta_align_rad=0.0,
            signed_ab_rad=0.0,
            half_plane_side=1,
            angles_rad=(),
            angular_margins_deg=(),
            rejection_reason="guide_ray_degenerate",
        )
    theta_align = float(
        wrap_angle(
            np.arctan2(robot_guide_vector[1], robot_guide_vector[0])
            - np.arctan2(source_vector[1], source_vector[0])
        )
    )
    representative_center = _representative_occluder_center_xy(
        collision_xy=anchor.world_position_xy,
        guide_robot_xy=guide_robot_xy,
        occluders=occluders,
    )
    obstacle_vector = representative_center - anchor.world_position_xy
    if np.linalg.norm(obstacle_vector) <= epsilon:
        return VisibilityGuidedRotationAngles(
            theta_align_rad=theta_align,
            signed_ab_rad=0.0,
            half_plane_side=1,
            angles_rad=(),
            angular_margins_deg=(),
            rejection_reason="guide_ray_degenerate",
        )
    signed_ab = float(
        wrap_angle(
            np.arctan2(obstacle_vector[1], obstacle_vector[0])
            - np.arctan2(robot_guide_vector[1], robot_guide_vector[0])
        )
    )
    half_plane_side = 1 if signed_ab >= 0.0 else -1
    signed_ab_deg = float(np.rad2deg(signed_ab))
    relative_degrees = np.arange(
        -180.0,
        180.0,
        margin_step_deg,
        dtype=np.float64,
    )
    primary_degrees = [
        float(relative_deg)
        for relative_deg in relative_degrees.tolist()
        if half_plane_side * relative_deg > abs(signed_ab_deg)
    ]
    primary_degrees.sort(key=lambda relative_deg: half_plane_side * relative_deg)
    if search_stage == "primary":
        selected_degrees = primary_degrees
    else:
        primary_degree_set = set(primary_degrees)

        def circular_distance_to_b(relative_deg: float) -> tuple[float, float]:
            distance = abs(
                (relative_deg - signed_ab_deg + 180.0) % 360.0 - 180.0
            )
            return distance, relative_deg

        selected_degrees = sorted(
            (
                float(relative_deg)
                for relative_deg in relative_degrees.tolist()
                if float(relative_deg) not in primary_degree_set
            ),
            key=circular_distance_to_b,
        )
    candidate_angles = [
        float(wrap_angle(theta_align + np.deg2rad(relative_deg)))
        for relative_deg in selected_degrees
    ]
    retained_margins = [
        half_plane_side * relative_deg - abs(signed_ab_deg)
        for relative_deg in selected_degrees
    ]
    return VisibilityGuidedRotationAngles(
        theta_align_rad=theta_align,
        signed_ab_rad=signed_ab,
        half_plane_side=half_plane_side,
        angles_rad=tuple(candidate_angles),
        angular_margins_deg=tuple(retained_margins),
        rejection_reason=None if candidate_angles else "half_plane_margin_missing",
    )


def _sample_route_poses(
    task_template: Sop05rTebTaskTemplate,
    sample_times_s: np.ndarray,
) -> np.ndarray:
    """Sample the full route and hold its final validated pose past 5 seconds."""

    route = task_template.route
    times = np.asarray(sample_times_s, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < -1e-6):
        raise ValueError("sample_times_s must be finite nonnegative [T]")
    times = np.maximum(times, 0.0)
    anchor_times = np.r_[0.0, route.sample_times_s.astype(np.float64)]
    anchor_poses = np.vstack(
        (route.band_poses_world[0], route.sampled_poses_world)
    ).astype(np.float64)
    poses = np.empty((times.size, 3), dtype=np.float64)
    poses[:, 0] = np.interp(times, anchor_times, anchor_poses[:, 0])
    poses[:, 1] = np.interp(times, anchor_times, anchor_poses[:, 1])
    poses[:, 2] = wrap_angle(
        np.interp(times, anchor_times, np.unwrap(anchor_poses[:, 2]))
    )
    return poses


def _sample_robot_poses(
    task_template: Sop05rTebTaskTemplate,
    base_state: BaseState,
    sample_times_s: np.ndarray,
    *,
    dt_s: float,
) -> np.ndarray:
    """Join source history before t=0 with the fixed M4 route after t=0."""

    times = np.asarray(sample_times_s, dtype=np.float64)
    earliest_time_s = -(base_state.robot_history.shape[0] - 1) * dt_s
    if np.any(times < earliest_time_s - 1e-6):
        raise ValueError("requested robot history predates source BaseState support")
    result = np.empty((times.size, 3), dtype=np.float64)
    source_mask = times <= 0.0
    source_times = (
        np.arange(base_state.robot_history.shape[0], dtype=np.float64)
        - (base_state.robot_history.shape[0] - 1)
    ) * dt_s
    source_poses = np.asarray(base_state.robot_history, dtype=np.float64)
    result[source_mask, 0] = np.interp(
        times[source_mask], source_times, source_poses[:, 0]
    )
    result[source_mask, 1] = np.interp(
        times[source_mask], source_times, source_poses[:, 1]
    )
    result[source_mask, 2] = wrap_angle(
        np.interp(
            times[source_mask],
            source_times,
            np.unwrap(source_poses[:, 2]),
        )
    )
    if np.any(~source_mask):
        result[~source_mask] = _sample_route_poses(
            task_template,
            times[~source_mask],
        )
    return result


def _route_anchor_indices(
    task_template: Sop05rTebTaskTemplate,
    config: Sop05rTebConfig,
) -> tuple[int, ...]:
    route = task_template.route
    positions = np.vstack(
        (route.band_poses_world[0, :2], route.sampled_poses_world[:, :2])
    ).astype(np.float64)
    lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    total_length = float(cumulative[-1])
    lower_fraction, upper_fraction = (
        config.generation.collision_route_path_fraction_range
    )
    candidates = [
        index
        for index in range(route.sample_times_s.size)
        if (
            index > 0
            and total_length > 0.0
            and lower_fraction <= cumulative[index + 1] / total_length <= upper_fraction
            and route.sample_times_s[index]
            <= config.trajectory.future_horizon_s + 1e-6
            and route.sample_times_s[index]
            < route.goal_arrival_time_s - 1e-6
        )
    ]
    return tuple(
        reversed(candidates[-config.generation.max_route_anchor_candidates :])
    )


def synchronized_long40_anchor_index(
    *,
    route_time_s: float,
    current_index: int,
    future_dt_s: float,
    sample_count: int,
) -> int:
    """Map one route future endpoint to its same-time Long40 target index."""

    if not np.isfinite(route_time_s) or route_time_s <= 0.0:
        raise ValueError("route_time_s must be a positive finite future time")
    if (
        isinstance(current_index, bool)
        or not isinstance(current_index, int)
        or current_index < 0
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= current_index + 1
        or not np.isfinite(future_dt_s)
        or future_dt_s <= 0.0
    ):
        raise ValueError("Long40 synchronization parameters are invalid")
    future_step = int(round(route_time_s / future_dt_s))
    if not np.isclose(
        route_time_s,
        future_step * future_dt_s,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("route_time_s must lie on the Long40 future grid")
    anchor_index = current_index + future_step
    if not current_index < anchor_index < sample_count:
        raise ValueError("route_time_s lies outside Long40 future support")
    return anchor_index


def _within_grid(
    footprint: object,
    poses: np.ndarray,
    *,
    base_config: Mapping[str, object],
) -> bool:
    grid = build_grid_spec(dict(base_config))
    x_min = -0.5 * grid.width * grid.resolution_m
    x_max = 0.5 * grid.width * grid.resolution_m
    y_min = -0.5 * grid.height * grid.resolution_m
    y_max = 0.5 * grid.height * grid.resolution_m
    return all(
        x_min < bounds[0] and bounds[2] < x_max and y_min < bounds[1] and bounds[3] < y_max
        for bounds in (footprint_aabb(footprint, pose) for pose in poses)
    )


def _physics_rejection(
    *,
    task_template: Sop05rTebTaskTemplate,
    transformed_poses: np.ndarray,
    snippet: LongMotionSnippet,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, object],
    anchor_index: int,
) -> str | None:
    grid = build_grid_spec(dict(base_config))
    footprint = footprint_from_spec(
        {"object_type": snippet.object_type, "footprint": snippet.footprint}
    )
    if not _within_grid(footprint, transformed_poses, base_config=base_config):
        return "target_out_of_bounds"
    source_static = (
        np.zeros((grid.height, grid.width), dtype=np.bool_)
        if base_state.static_map_local is None
        else np.asarray(base_state.static_map_local != 0, dtype=np.bool_)
    )
    occluder_mask = np.logical_or.reduce(
        tuple(rasterize_occluder(component, grid) for component in task_template.occluders)
    )
    if swept_footprint_intersects_occupancy(
        footprint, transformed_poses, occluder_mask, grid=grid
    ):
        return "target_occluder_collision"
    if np.any(source_static) and swept_footprint_intersects_occupancy(
        footprint, transformed_poses, source_static, grid=grid
    ):
        return "target_source_static_collision"
    dt_s = float(base_config["bev"]["future_dt_s"])
    velocities = np.diff(transformed_poses[:, :2], axis=0) / dt_s
    dynamic_config = base_config["dynamic_objects"][snippet.object_type]
    if np.any(np.linalg.norm(velocities, axis=1) > float(dynamic_config["max_speed_mps"]) + 1e-6):
        return "target_speed_limit"
    accelerations = np.diff(velocities, axis=0) / dt_s
    if accelerations.size and np.any(
        np.linalg.norm(accelerations, axis=1)
        > float(dynamic_config["max_acceleration_mps2"]) + 1e-5
    ):
        return "target_acceleration_limit"
    for object_id in sorted(oracle_context.dynamic_object_history):
        context_footprint = footprint_from_spec(oracle_context.dynamic_object_specs[object_id])
        context_poses = np.vstack(
            (
                oracle_context.dynamic_object_history[object_id],
                oracle_context.dynamic_object_future[object_id],
            )
        )
        if synchronized_sweeps_intersect(
            footprint,
            transformed_poses,
            context_footprint,
            context_poses,
            grid=grid,
        ):
            return "target_context_collision"
    return None


def solve_anchored_human_placement(
    *,
    task_template: Sop05rTebTaskTemplate,
    snippet: LongMotionSnippet,
    base_state: BaseState,
    oracle_context: OracleContext,
    base_config: Mapping[str, object],
    teb_config: Sop05rTebConfig,
    seed: int,
    selection_mode: str = "seen_first",
) -> AnchoredPlacementEvaluation:
    """Return the first physical M5 candidate from half-plane margin samples."""

    if snippet.object_type != "human" or snippet.split != base_state.split:
        return AnchoredPlacementEvaluation(
            result=None,
            rejection_reason="snippet_split_mismatch",
            attempted_candidates=0,
            rejection_counts={"snippet_split_mismatch": 1},
        )
    if oracle_context.base_state_id != base_state.state_id:
        raise ValueError("oracle context must belong to the supplied base state")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if selection_mode not in PLACEMENT_SELECTION_MODES:
        raise ValueError(
            "selection_mode must be one of " + ", ".join(PLACEMENT_SELECTION_MODES)
        )

    route_indices = _route_anchor_indices(task_template, teb_config)
    rng = np.random.default_rng(
        derive_seed(seed, task_template.template_id, snippet.snippet_id, teb_config.digest)
    )
    current_index = teb_config.trajectory.current_index
    sample_count = teb_config.trajectory.history_steps + teb_config.trajectory.future_steps
    sample_dt_s = teb_config.trajectory.future_dt_s
    temporal_scales = tuple(
        np.asarray(teb_config.placement.temporal_scales)[
            rng.permutation(len(teb_config.placement.temporal_scales))
        ]
    )
    rejection_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    attempted_candidates = 0
    for route_index in route_indices:
        route_index = int(route_index)
        route_time_s = float(task_template.route.sample_times_s[route_index])
        try:
            snippet_anchor_index = synchronized_long40_anchor_index(
                route_time_s=route_time_s,
                current_index=current_index,
                future_dt_s=sample_dt_s,
                sample_count=sample_count,
            )
        except ValueError:
            rejection_counts["collision_anchor_outside_long40_future"] += 1
            continue
        anchor = CollisionAnchor(
            route_sample_index=route_index,
            route_time_s=route_time_s,
            world_position_xy=task_template.route.sampled_poses_world[
                route_index, :2
            ],
            snippet_anchor_index=snippet_anchor_index,
            snippet_time_s=route_time_s,
        )
        snippet_time_offsets_s = (
            np.arange(sample_count, dtype=np.float64) - current_index
        ) * sample_dt_s
        route_poses = _sample_robot_poses(
            task_template,
            base_state,
            snippet_time_offsets_s,
            dt_s=sample_dt_s,
        )
        for temporal_scale in temporal_scales:
            try:
                source_poses, _ = resample_long_motion_snippet(
                    snippet,
                    time_scale=float(temporal_scale),
                )
            except ValueError:
                rejection_counts["source_extrapolation_required"] += 1
                continue

            for search_stage in ("primary", "secondary"):
                visibility_batch = _construct_visibility_guided_rotation_batch(
                    source_poses=source_poses,
                    anchor=anchor,
                    robot_positions_xy=route_poses[:, :2],
                    occluders=task_template.occluders,
                    decision_index=current_index,
                    guide_history_index=current_index - 4,
                    minimum_visible_history_frames=(
                        teb_config.occlusion.minimum_visible_history_frames
                    ),
                    minimum_occluded_history_frames=(
                        teb_config.occlusion.minimum_occluded_history_frames
                    ),
                    occluder_angular_margin_step_deg=(
                        teb_config.placement.occluder_angular_margin_step_deg
                    ),
                    search_stage=search_stage,
                    epsilon_m=teb_config.occlusion.centerline_intersection_epsilon_m,
                )
                candidate_counts["half_plane_batches"] += 1
                candidate_counts["half_plane_margin_candidates"] += len(
                    visibility_batch.angles_rad
                )
                if visibility_batch.sweep.rejection_reason is not None:
                    rejection_counts[visibility_batch.sweep.rejection_reason] += 1
                    if visibility_batch.sweep.rejection_reason == "guide_ray_degenerate":
                        break
                    continue
                candidate_counts["prefix4_visible_then_occluded"] += len(
                    visibility_batch.preferred_indices
                )
                candidate_counts["fallback_seen_then_occluded"] += len(
                    visibility_batch.fallback_indices
                )
                for angle_index, history_assessment in enumerate(
                    visibility_batch.history_assessments
                ):
                    if history_assessment.eligible or (
                        selection_mode == "h0_hidden"
                        and bool(visibility_batch.blocked[angle_index, 0])
                    ):
                        continue
                    rejection_counts[
                        (
                            "visible_history_insufficient"
                            if (
                                history_assessment.visible_frames
                                < teb_config.occlusion.minimum_visible_history_frames
                            )
                            else "window_occlusion_missing"
                        )
                    ] += 1
                if selection_mode == "h0_hidden":
                    eligible_indices = tuple(
                        sorted(
                            (
                                angle_index
                                for angle_index in range(
                                    len(visibility_batch.history_assessments)
                                )
                                if bool(visibility_batch.blocked[angle_index, 0])
                            ),
                            key=lambda angle_index: (
                                -visibility_batch.history_assessments[
                                    angle_index
                                ].occluded_frames,
                                angle_index,
                            ),
                        )
                    )
                    candidate_counts["h0_hidden_candidates"] += len(eligible_indices)
                else:
                    eligible_indices = (
                        *visibility_batch.preferred_indices,
                        *visibility_batch.fallback_indices,
                    )
                for angle_index in eligible_indices:
                    history_assessment = visibility_batch.history_assessments[angle_index]
                    attempted_candidates += 1
                    candidate_counts["tested_candidates"] += 1
                    angle = float(visibility_batch.angles_rad[angle_index])
                    transformed = visibility_batch.transformed_poses[angle_index]
                    translation = visibility_batch.translations_xy_m[angle_index]
                    blocked = visibility_batch.blocked[angle_index]
                    rejection = _physics_rejection(
                        task_template=task_template,
                        transformed_poses=transformed,
                        snippet=snippet,
                        base_state=base_state,
                        oracle_context=oracle_context,
                        base_config=base_config,
                        anchor_index=snippet_anchor_index,
                    )
                    if rejection is not None:
                        rejection_counts[rejection] += 1
                        continue
                    witness_index = history_assessment.blocked_indices[-1]
                    blocker_index = int(
                        visibility_batch.blocker_indices[angle_index, witness_index]
                    )
                    if blocker_index < 0:
                        rejection_counts["window_occlusion_missing"] += 1
                        continue
                    blocker_id = task_template.occluders[blocker_index].occluder_id
                    placement = AnchoredHumanPlacement(
                        source_snippet_id=snippet.snippet_id,
                        anchor=anchor,
                        rotation_rad=angle,
                        translation_xy_m=translation,
                        spatial_scale=1.0,
                        temporal_scale=float(temporal_scale),
                        history_poses=transformed[:8],
                        current_pose=transformed[7],
                        future_poses=transformed[8:],
                        provenance={
                            "placement_version": teb_config.placement.version,
                            "seed": int(seed),
                            "candidate_search": _CANDIDATE_SEARCH_BY_SELECTION_MODE[
                                selection_mode
                            ],
                            "placement_selection_mode": selection_mode,
                            "decision_time_s": 0.0,
                            "visible_history_frames": history_assessment.visible_frames,
                            "occluded_history_frames": history_assessment.occluded_frames,
                            "observed_history_class": history_assessment.observed_class,
                            "decision_visible": bool(~blocked[current_index]),
                            "theta_align_rad": visibility_batch.sweep.theta_align_rad,
                            "signed_ab_rad": visibility_batch.sweep.signed_ab_rad,
                            "half_plane_side": visibility_batch.sweep.half_plane_side,
                            "angular_margin_deg": (
                                visibility_batch.sweep.angular_margins_deg[angle_index]
                            ),
                        },
                    )
                    witness = CenterlineOcclusionWitness(
                        version=teb_config.occlusion.version,
                        time_s=(witness_index - current_index) * sample_dt_s,
                        sample_index=witness_index,
                        robot_position_xy=route_poses[witness_index, :2],
                        target_position_xy=transformed[witness_index, :2],
                        blocking_occluder_id=blocker_id,
                    )
                    return AnchoredPlacementEvaluation(
                        result=AnchoredPlacementResult(
                            placement=placement,
                            witness=witness,
                            visibility=history_assessment,
                            attempted_candidates=attempted_candidates,
                            rejection_counts=dict(sorted(rejection_counts.items())),
                            candidate_counts=dict(sorted(candidate_counts.items())),
                        ),
                        rejection_reason=None,
                        attempted_candidates=attempted_candidates,
                        rejection_counts=dict(sorted(rejection_counts.items())),
                        candidate_counts=dict(sorted(candidate_counts.items())),
                    )
    return AnchoredPlacementEvaluation(
        result=None,
        rejection_reason="occlusion_witness_missing",
        attempted_candidates=attempted_candidates,
        rejection_counts=dict(sorted(rejection_counts.items())),
        candidate_counts=dict(sorted(candidate_counts.items())),
    )
