"""Typed transplant, policy, event orchestration, and determinism tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.contracts import (
    BaseState,
    LocalTrajectory,
    OracleContext,
    build_grid_spec,
    validate_oracle_world,
)
from src.datasets.snippet_library import MotionSnippet, SnippetLibrary
from src.generation import event_sampler as event_sampler_module
from src.generation import event_target_motion_shard as target_motion_shard_module
from src.generation.dynamic_object_transplant import (
    TRANSFORM_ALGORITHM_VERSION,
    TargetPolicyError,
    TransplantError,
    footprint_from_spec,
    normalize_target_type_policy,
    transplant_reachability_candidate,
    transplant_snippet,
)
from src.generation.blind_reachability import (
    BLIND_REACHABILITY_ALGORITHM_VERSION,
    REACHABLE_ARC_SCHEDULE_VERSION,
    ReachabilityCandidate,
    ReachabilityIdentity,
    build_reachability_candidate,
)
from src.generation.event_sampler import (
    GeneratorConfigError,
    build_event_type_schedule,
    generate_events,
    load_generator_config,
    normalize_generator_config,
)
from src.generation.structural_blindspot import StructuralBlindSpot
from src.geometry import (
    CircleFootprint,
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint,
    rasterize_footprint_sweep,
    trajectory_signed_clearances,
    wrap_angle,
)
from src.planning.differential_drive import rollout_constant_control
from src.planning.query_maps import build_local_trajectory
from src.planning.trajectory_sampler import CandidateRollout, sample_candidate_rollouts
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _policy() -> dict:
    return {
        "whitelist": ["human"],
        "weights": {
            "human": 1.0,
            "carried_object": 0.0,
            "unknown_dynamic": 0.0,
        },
    }


def _snippet(
    *, object_type: str = "human", footprint: dict | None = None
) -> MotionSnippet:
    times = np.arange(23, dtype=np.float32) * np.float32(0.2)
    positions = np.column_stack((times, 0.08 * times**2)).astype(np.float32)
    velocities = np.column_stack(
        (np.ones_like(times), 0.16 * times)
    ).astype(np.float32)
    headings = np.arctan2(velocities[:, 1], velocities[:, 0]).astype(np.float32)
    footprint = footprint or {"kind": "circle", "radius_m": 0.30}
    return MotionSnippet(
        snippet_id=f"train-{object_type}-snippet-fixed",
        split="train",
        source_recording_id="source-recording",
        source_object_id="source-recording::misleading-Helmet-name",
        object_type=object_type,
        footprint=footprint,
        start_timestamp=2.0,
        positions=positions,
        velocities=velocities,
        headings=headings,
        duration_s=4.4,
        mean_speed_mps=1.02,
        max_acceleration_mps2=0.16,
        mean_abs_curvature_per_m=0.10,
        provenance={
            "source_body_name": "misleading-Helmet-name",
            "raw_role": "Visitors-Alone",
            "track_provenance": {
                "geometry_source": "marker_extent_p95",
                "orientation_source": "qtm_rotation",
            },
        },
    )


def _reachability_candidate(
    snippet: MotionSnippet,
    *,
    conflict_index: int = 7,
    conflict_time_s: float | None = None,
    conflict_point: tuple[float, float] = (1.25, -0.35),
    desired_crossing_direction: tuple[float, float] = (0.0, 1.0),
    base_state_id: str = "train-base-reachability",
    trajectory_id: str = "trajectory-reachability",
) -> ReachabilityCandidate:
    anchor_index = min(22, 7 + conflict_index + 1)
    identity = ReachabilityIdentity(
        base_state_id=base_state_id,
        trajectory_id=trajectory_id,
        source_snippet_id=snippet.snippet_id,
        conflict_index=conflict_index,
        conflict_time_s=(
            (conflict_index + 1) * 0.2
            if conflict_time_s is None
            else conflict_time_s
        ),
        crossing_side=-1,
        angle_offset_deg=15.0,
    )
    return build_reachability_candidate(
        conflict_point=np.asarray(conflict_point, dtype=np.float64),
        source_current_xy=snippet.positions[7].astype(np.float64),
        source_anchor_xy=snippet.positions[anchor_index].astype(np.float64),
        desired_crossing_direction=np.asarray(
            desired_crossing_direction, dtype=np.float64
        ),
        identity=identity,
    )


def _reachability_transplant_kwargs(
    base_candidate: ReachabilityCandidate,
    **changes: object,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "candidate": base_candidate,
        "future_dt_s": 0.2,
        "future_steps": 15,
        "target_type_policy_digest": "reachability-policy-digest",
        "seed": 23,
        "context_object_ids": ("context-b", "context-a"),
    }
    kwargs.update(changes)
    return kwargs


def test_reachability_transplant_uses_candidate_se2_without_resampling() -> None:
    snippet = _snippet()
    candidate = _reachability_candidate(snippet)
    positions_before = snippet.positions.copy()
    headings_before = snippet.headings.copy()
    candidate_arrays_before = tuple(
        array.copy()
        for array in (
            candidate.rotation_matrix,
            candidate.current_xy,
            candidate.conflict_point,
            candidate.source_delta_xy,
            candidate.desired_crossing_direction,
        )
    )

    result = transplant_reachability_candidate(
        snippet,
        **_reachability_transplant_kwargs(candidate),
    )

    full_poses = np.vstack((result.history_poses, result.future_poses))
    expected_positions = (
        (snippet.positions.astype(np.float64) - snippet.positions[7].astype(np.float64))
        @ candidate.rotation_matrix.T
        + candidate.current_xy
    )
    expected_headings = snippet.headings.astype(np.float64) + candidate.rotation_rad
    old_velocity_rotation = float(
        np.pi / 2.0
        - np.arctan2(
            float(snippet.velocities[15, 1]),
            float(snippet.velocities[15, 0]),
        )
    )

    assert abs(candidate.rotation_rad - old_velocity_rotation) > 0.01
    np.testing.assert_array_equal(
        result.current_pose[:2], candidate.current_xy.astype(np.float32)
    )
    np.testing.assert_array_equal(
        result.future_poses[candidate.identity.conflict_index, :2],
        candidate.conflict_point.astype(np.float32),
    )
    np.testing.assert_allclose(full_poses[:, :2], expected_positions, atol=1e-6)
    np.testing.assert_allclose(full_poses[:, 2], expected_headings, atol=1e-6)
    np.testing.assert_allclose(
        np.diff(full_poses[:, :2], axis=0),
        np.diff(snippet.positions.astype(np.float64), axis=0)
        @ candidate.rotation_matrix.T,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.linalg.norm(
            full_poses[:, None, :2] - full_poses[None, :, :2], axis=-1
        ),
        np.linalg.norm(
            snippet.positions[:, None, :].astype(np.float64)
            - snippet.positions[None, :, :].astype(np.float64),
            axis=-1,
        ),
        atol=2e-6,
    )
    np.testing.assert_array_equal(snippet.positions, positions_before)
    np.testing.assert_array_equal(snippet.headings, headings_before)
    for actual, expected in zip(
        (
            candidate.rotation_matrix,
            candidate.current_xy,
            candidate.conflict_point,
            candidate.source_delta_xy,
            candidate.desired_crossing_direction,
        ),
        candidate_arrays_before,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_reachability_transplant_has_frozen_output_and_json_provenance() -> None:
    snippet = _snippet()
    candidate = _reachability_candidate(snippet, conflict_index=4)

    result = transplant_reachability_candidate(
        snippet,
        **_reachability_transplant_kwargs(candidate),
    )

    assert TRANSFORM_ALGORITHM_VERSION == "reachability_candidate_se2_v1"
    assert result.history_poses.shape == (8, 3)
    assert result.current_pose.shape == (3,)
    assert result.future_poses.shape == (15, 3)
    assert result.history_poses.dtype == np.float32
    assert result.current_pose.dtype == np.float32
    assert result.future_poses.dtype == np.float32
    assert np.isfinite(result.history_poses).all()
    assert np.isfinite(result.current_pose).all()
    assert np.isfinite(result.future_poses).all()
    np.testing.assert_array_equal(result.current_pose, result.history_poses[-1])
    assert result.target_dynamic_object_id.startswith("generated::human::")
    assert result.provenance["transform_id"].startswith("transform-")
    assert result.provenance["transform_id"] != result.target_dynamic_object_id
    assert result.provenance["transform_algorithm_version"] == (
        TRANSFORM_ALGORITHM_VERSION
    )
    assert result.provenance["reachability_candidate_id"] == candidate.candidate_id
    assert result.provenance["reachability_algorithm_version"] == (
        BLIND_REACHABILITY_ALGORITHM_VERSION
    )
    assert result.provenance["reachable_arc_schedule_version"] == (
        REACHABLE_ARC_SCHEDULE_VERSION
    )
    assert result.provenance["source_current_index"] == 7
    assert result.provenance["source_anchor_index"] == 12
    assert result.provenance["source_delta_xy"] == candidate.source_delta_xy.tolist()
    assert result.provenance["candidate_current_xy"] == candidate.current_xy.tolist()
    assert result.provenance["conflict_point"] == candidate.conflict_point.tolist()
    assert result.provenance["rotation_rad"] == candidate.rotation_rad
    assert result.provenance["desired_crossing_direction"] == (
        candidate.desired_crossing_direction.tolist()
    )
    assert result.provenance["crossing_side"] == -1
    assert result.provenance["angle_offset_deg"] == 15.0
    assert result.provenance["conflict_index"] == 4
    assert result.provenance["conflict_time_s"] == pytest.approx(1.0)
    assert result.provenance["time_scale"] == 1.0
    assert result.provenance["motion_snippet_layout_version"] == (
        "history8_current7_future15_v1"
    )
    assert result.provenance["source_recording_id"] == snippet.source_recording_id
    assert result.provenance["source_object_id"] == snippet.source_object_id
    json.dumps(result.provenance, sort_keys=True, allow_nan=False)


def test_reachability_transplant_ids_bind_candidate_seed_context_and_arrays() -> None:
    snippet = _snippet()
    candidate = _reachability_candidate(snippet)
    base = transplant_reachability_candidate(
        snippet,
        **_reachability_transplant_kwargs(candidate),
    )
    repeated = transplant_reachability_candidate(
        snippet,
        **_reachability_transplant_kwargs(
            candidate, context_object_ids=("context-a", "context-b")
        ),
    )
    changed_seed = transplant_reachability_candidate(
        snippet,
        **_reachability_transplant_kwargs(candidate, seed=24),
    )
    changed_context = transplant_reachability_candidate(
        snippet,
        **_reachability_transplant_kwargs(
            candidate, context_object_ids=("context-a", "context-b", "context-c")
        ),
    )
    changed_candidate_value = _reachability_candidate(
        snippet, conflict_point=(1.35, -0.35)
    )
    changed_candidate = transplant_reachability_candidate(
        snippet,
        **_reachability_transplant_kwargs(changed_candidate_value),
    )
    changed_positions = snippet.positions.copy()
    changed_positions[0, 1] += np.float32(0.125)
    changed_array_snippet = replace(snippet, positions=changed_positions)
    changed_array = transplant_reachability_candidate(
        changed_array_snippet,
        **_reachability_transplant_kwargs(candidate),
    )
    collision_resolved = transplant_reachability_candidate(
        snippet,
        **_reachability_transplant_kwargs(
            candidate,
            context_object_ids=(
                "context-a",
                "context-b",
                base.target_dynamic_object_id,
            ),
        ),
    )

    assert repeated.target_dynamic_object_id == base.target_dynamic_object_id
    assert repeated.provenance["transform_id"] == base.provenance["transform_id"]
    for changed in (
        changed_seed,
        changed_context,
        changed_candidate,
        changed_array,
        collision_resolved,
    ):
        assert changed.target_dynamic_object_id != base.target_dynamic_object_id
        assert changed.provenance["transform_id"] != base.provenance["transform_id"]
    assert collision_resolved.target_dynamic_object_id != base.target_dynamic_object_id


def test_reachability_transplant_preserves_unwrapped_headings() -> None:
    snippet = _snippet()
    headings = snippet.headings.copy()
    headings[:] = np.linspace(2.9, 3.5, 23, dtype=np.float32)
    snippet = replace(snippet, headings=headings)
    candidate = _reachability_candidate(
        snippet, desired_crossing_direction=(-1.0, 0.0)
    )

    result = transplant_reachability_candidate(
        snippet,
        **_reachability_transplant_kwargs(candidate),
    )
    full_poses = np.vstack((result.history_poses, result.future_poses))

    np.testing.assert_allclose(
        full_poses[:, 2],
        headings.astype(np.float64) + candidate.rotation_rad,
        atol=1e-6,
    )
    assert np.max(np.abs(full_poses[:, 2])) > np.pi


def test_reachability_transplant_rejects_candidate_identity_mismatches() -> None:
    snippet = _snippet()
    candidate = _reachability_candidate(snippet)

    with pytest.raises(ValueError, match="snippet_id"):
        transplant_reachability_candidate(
            replace(snippet, snippet_id="different-snippet"),
            **_reachability_transplant_kwargs(candidate),
        )

    bad_time = _reachability_candidate(
        snippet, conflict_time_s=candidate.identity.conflict_time_s + 1e-4
    )
    with pytest.raises(ValueError, match="conflict_time_s"):
        transplant_reachability_candidate(
            snippet,
            **_reachability_transplant_kwargs(bad_time),
        )

    bad_index = _reachability_candidate(snippet, conflict_index=15)
    with pytest.raises(ValueError, match="source_anchor_index"):
        transplant_reachability_candidate(
            snippet,
            **_reachability_transplant_kwargs(bad_index),
        )

    changed_positions = snippet.positions.copy()
    changed_positions[15, 0] += np.float32(1e-4)
    with pytest.raises(ValueError, match="source_delta_xy"):
        transplant_reachability_candidate(
            replace(snippet, positions=changed_positions),
            **_reachability_transplant_kwargs(candidate),
        )


def test_reachability_transplant_requires_byte_exact_source_delta() -> None:
    snippet = _snippet()
    straight_positions = snippet.positions.copy()
    straight_positions[:, 1] = np.float32(0.0)
    straight_velocities = snippet.velocities.copy()
    straight_velocities[:, 1] = np.float32(0.0)
    straight_headings = np.zeros(23, dtype=np.float32)
    snippet = replace(
        snippet,
        positions=straight_positions,
        velocities=straight_velocities,
        headings=straight_headings,
    )
    candidate = _reachability_candidate(snippet)
    signed_zero_positions = straight_positions.copy()
    signed_zero_positions[15, 1] = np.float32(-0.0)

    assert np.array_equal(
        candidate.source_delta_xy,
        signed_zero_positions[15].astype(np.float64)
        - signed_zero_positions[7].astype(np.float64),
    )
    with pytest.raises(ValueError, match="source_delta_xy"):
        transplant_reachability_candidate(
            replace(snippet, positions=signed_zero_positions),
            **_reachability_transplant_kwargs(candidate),
        )


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"candidate": object()}, TypeError, "ReachabilityCandidate"),
        ({"future_dt_s": 0.1}, ValueError, "future_dt_s"),
        ({"future_dt_s": np.nan}, ValueError, "future_dt_s"),
        ({"future_steps": 14}, ValueError, "future_steps"),
        ({"future_steps": 15.0}, TypeError, "future_steps"),
        ({"seed": np.nan}, TypeError, "seed"),
        ({"context_object_ids": "context-a"}, TypeError, "context_object_ids"),
    ],
)
def test_reachability_transplant_rejects_noncanonical_inputs(
    changes: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    snippet = _snippet()
    candidate = _reachability_candidate(snippet)

    with pytest.raises(error, match=message):
        transplant_reachability_candidate(
            snippet,
            **_reachability_transplant_kwargs(candidate, **changes),
        )


def test_reachability_transplant_rejects_nonfinite_snippet() -> None:
    snippet = _snippet()
    candidate = _reachability_candidate(snippet)
    positions = snippet.positions.copy()
    positions[0, 0] = np.nan

    with pytest.raises(TransplantError) as exc_info:
        transplant_reachability_candidate(
            replace(snippet, positions=positions),
            **_reachability_transplant_kwargs(candidate),
        )

    assert exc_info.value.reason == "snippet_nonfinite"


def _generator_config(event_kind: str = "structural") -> dict:
    return {
        "schema_version": "2.0.0",
        "target_type_policy": _policy(),
        "event_type_weights": {
            "environment": float(event_kind == "environment"),
            "structural": float(event_kind == "structural"),
            "mixed": float(event_kind == "mixed"),
        },
        "conflict_time_range_s": [1.8, 1.8],
        "max_local_curvature_per_m": 1.0,
        "crossing_angle_max_deg": 35.0,
        "time_scale_range": [1.0, 1.0],
        "max_resample_attempts": 32,
        "min_contiguous_visible_frames": 2,
        "occluders": {
            "types": ["pillar"],
            "normal_offset_range_m": [0.8, 1.2],
            "wall": {
                "length_range_m": [1.0, 1.4],
                "width_range_m": [0.2, 0.3],
            },
            "shelf": {
                "length_range_m": [1.0, 1.4],
                "width_range_m": [0.4, 0.5],
            },
            "pillar": {
                "length_range_m": [0.4, 0.5],
                "width_range_m": [0.4, 0.5],
            },
        },
        "structural_fov": {
            "forward_fov_deg": [160.0],
            "range_m": [6.0],
            "optional_blind_sectors": [
                {"center_deg": -90.0, "width_deg": 110.0},
                {"center_deg": 90.0, "width_deg": 110.0},
            ],
        },
    }


def _base_inputs():
    config = load_config()
    grid = build_grid_spec(config)
    context_id = "context-recording::LO1"
    context_history = np.tile(
        np.asarray([-2.0, 2.0, np.pi / 4.0], dtype=np.float32),
        (grid.history_steps, 1),
    )
    context_future = np.tile(
        np.asarray([-2.0, 2.0, np.pi / 4.0], dtype=np.float32),
        (grid.future_steps, 1),
    )
    context_spec = {
        "object_type": "carried_object",
        "footprint": {"kind": "rectangle", "length_m": 0.8, "width_m": 0.2},
    }
    base_state = BaseState(
        state_id="train-base-event-fixture",
        split="train",
        recording_id="context-recording",
        dynamic_object_ids=(context_id,),
        timestamp=10.0,
        robot_history=np.zeros((grid.history_steps, 3), dtype=np.float32),
        robot_state=np.asarray([0.4, 0.0], dtype=np.float32),
        visible_dynamic_object_history={context_id: context_history.copy()},
        visible_dynamic_object_specs={context_id: context_spec},
        static_map_local=np.zeros((grid.height, grid.width), dtype=np.float32),
        metadata={"fixture": "sop05"},
    )
    oracle_context = OracleContext(
        base_state_id=base_state.state_id,
        dynamic_object_history={context_id: context_history},
        dynamic_object_future={context_id: context_future},
        dynamic_object_specs={context_id: context_spec},
        metadata={"future_dt_s": 0.2},
    )
    candidate = sample_candidate_rollouts(config)[7]  # v=0.4, omega=0
    trajectory = build_local_trajectory(
        candidate, config, braking_deceleration_mps2=1.0
    )
    snippet = _snippet()
    libraries = {
        "human": SnippetLibrary(
            object_type="human",
            snippets=(snippet,),
            summary={"split": "train", "accepted_count": 1},
        )
    }
    return config, grid, base_state, oracle_context, trajectory, libraries


def _constant_curvature_trajectory(config: dict, *, radius_m: float):
    dt_s = float(config["bev"]["future_dt_s"])
    steps = int(config["bev"]["future_steps"])
    angular_speed = -0.8
    poses, controls = rollout_constant_control(
        v=float(radius_m) * abs(angular_speed),
        omega=angular_speed,
        dt_s=dt_s,
        steps=steps,
    )
    assert poses.dtype == np.float32
    assert controls.dtype == np.float32
    return build_local_trajectory(
        CandidateRollout(
            trajectory_id=f"constant-curvature-radius-{radius_m}",
            poses=poses,
            controls=controls,
            is_stop=False,
            is_reverse=False,
        ),
        config,
        braking_deceleration_mps2=1.0,
    )


@pytest.fixture(scope="module")
def constant_curvature_inputs() -> tuple[
    dict, LocalTrajectory, LocalTrajectory
]:
    config = load_config()
    return (
        config,
        _constant_curvature_trajectory(config, radius_m=1.0),
        _constant_curvature_trajectory(config, radius_m=1.0 / 1.01),
    )


def _valid_physics_target():
    target = transplant_snippet(
        _snippet(),
        conflict_point=(1.2, 0.0),
        conflict_time_s=1.8,
        crossing_direction=(0.0, 1.0),
        time_scale=1.0,
        future_dt_s=0.2,
        future_steps=15,
        base_state_id="train-base-event-fixture",
        trajectory_id="physics-fixture",
        target_type_policy_digest="policy-digest",
        seed=71,
        context_object_ids=(),
    )
    times = np.arange(23, dtype=np.float32) * np.float32(0.2)
    full_poses = np.column_stack(
        (
            np.float32(2.0) + np.float32(0.5) * times,
            np.full(23, 3.0, dtype=np.float32),
            np.zeros(23, dtype=np.float32),
        )
    ).astype(np.float32)
    return replace(
        target,
        history_poses=full_poses[:8].copy(),
        current_pose=full_poses[7].copy(),
        future_poses=full_poses[8:].copy(),
    )


def test_target_policy_is_complete_normalized_and_order_stable() -> None:
    first = normalize_target_type_policy(
        {
            "whitelist": ["human", "unknown_dynamic"],
            "weights": {
                "human": 3.0,
                "carried_object": 99.0,
                "unknown_dynamic": 1.0,
            },
        }
    )
    second = normalize_target_type_policy(
        {
            "weights": {
                "unknown_dynamic": 1.0,
                "human": 3.0,
                "carried_object": 0.0,
            },
            "whitelist": ["unknown_dynamic", "human"],
        }
    )

    assert first.whitelist == ("human", "unknown_dynamic")
    assert first.weights == {
        "human": 0.75,
        "carried_object": 0.0,
        "unknown_dynamic": 0.25,
    }
    assert first.digest == second.digest


@pytest.mark.parametrize(
    "policy",
    [
        {"whitelist": ["human"], "weights": {"human": 1.0}},
        {
            "whitelist": ["human"],
            "weights": {
                "human": -1.0,
                "carried_object": 0.0,
                "unknown_dynamic": 0.0,
            },
        },
        {
            "whitelist": ["human"],
            "weights": {
                "human": np.nan,
                "carried_object": 0.0,
                "unknown_dynamic": 0.0,
            },
        },
        {
            "whitelist": ["human"],
            "weights": {
                "human": 0.0,
                "carried_object": 1.0,
                "unknown_dynamic": 0.0,
            },
        },
    ],
)
def test_target_policy_rejects_incomplete_or_nonphysical_weights(policy: dict) -> None:
    with pytest.raises(TargetPolicyError):
        normalize_target_type_policy(policy)


@pytest.mark.parametrize(
    ("object_type", "footprint"),
    [
        ("human", {"kind": "circle", "radius_m": 0.30}),
        (
            "carried_object",
            {"kind": "rectangle", "length_m": 0.8, "width_m": 0.2},
        ),
    ],
)
def test_transplant_preserves_frozen_type_footprint_and_full_rigid_motion(
    object_type: str, footprint: dict
) -> None:
    snippet = _snippet(object_type=object_type, footprint=footprint)
    policy = normalize_target_type_policy(
        {
            "whitelist": [object_type],
            "weights": {
                "human": float(object_type == "human"),
                "carried_object": float(object_type == "carried_object"),
                "unknown_dynamic": 0.0,
            },
        }
    )
    result = transplant_snippet(
        snippet,
        conflict_point=np.asarray([1.2, 0.0], dtype=np.float32),
        conflict_time_s=1.6,
        crossing_direction=np.asarray([0.0, 1.0], dtype=np.float32),
        time_scale=1.0,
        future_dt_s=0.2,
        future_steps=15,
        base_state_id="train-base-1",
        trajectory_id="traj-1",
        target_type_policy_digest=policy.digest,
        seed=13,
        context_object_ids=("context-object",),
    )
    full_poses = np.vstack((result.history_poses, result.future_poses))
    source_anchor_index = 7 + 8  # source current index + 1.6 / 0.2
    source_velocity = snippet.velocities[source_anchor_index].astype(np.float64)
    source_angle = float(np.arctan2(source_velocity[1], source_velocity[0]))
    rotation_angle = float(np.pi / 2.0 - source_angle)
    rotation = np.asarray(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ],
        dtype=np.float64,
    )
    expected_positions = (
        (
            snippet.positions.astype(np.float64)
            - snippet.positions[source_anchor_index].astype(np.float64)
        )
        @ rotation.T
        + np.asarray([1.2, 0.0], dtype=np.float64)
    )

    assert result.object_type == object_type
    assert result.footprint_spec == {"object_type": object_type, "footprint": footprint}
    assert result.source_object_id == snippet.source_object_id
    assert result.target_dynamic_object_id != snippet.source_object_id
    assert result.target_dynamic_object_id not in {"context-object"}
    assert result.history_poses.dtype == np.float32
    assert result.current_pose.dtype == np.float32
    assert result.future_poses.dtype == np.float32
    assert result.history_poses.shape == (8, 3)
    assert result.current_pose.shape == (3,)
    assert result.future_poses.shape == (15, 3)
    assert np.isfinite(full_poses).all()
    np.testing.assert_array_equal(result.current_pose, result.history_poses[-1])
    np.testing.assert_allclose(full_poses[:, :2], expected_positions, atol=1e-6)
    np.testing.assert_allclose(result.future_poses[7, :2], [1.2, 0.0], atol=1e-6)
    expected_headings = wrap_angle(
        snippet.headings.astype(np.float64) + rotation_angle
    )
    np.testing.assert_allclose(full_poses[:, 2], expected_headings, atol=1e-6)
    source_yaw_delta = np.unwrap(snippet.headings) - float(snippet.headings[0])
    result_yaw_delta = np.unwrap(full_poses[:, 2]) - float(full_poses[0, 2])
    np.testing.assert_allclose(result_yaw_delta, source_yaw_delta, atol=1e-6)
    assert result.provenance["motion_snippet_layout_version"] == (
        "history8_current7_future15_v1"
    )
    assert result.provenance["source_current_index"] == 7
    assert result.provenance["source_current_time_s"] == pytest.approx(1.4)
    assert result.provenance["source_conflict_anchor_time_s"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("snippet", "reason"),
    [
        (
            replace(
                _snippet(),
                positions=_snippet().positions[:16].copy(),
                velocities=_snippet().velocities[:16].copy(),
                headings=_snippet().headings[:16].copy(),
                duration_s=3.0,
            ),
            "snippet_shape_invalid",
        ),
        (replace(_snippet(), duration_s=3.0), "snippet_duration_invalid"),
    ],
)
def test_transplant_rejects_noncanonical_snippet_layout(
    snippet: MotionSnippet, reason: str
) -> None:
    with pytest.raises(TransplantError) as exc_info:
        transplant_snippet(
            snippet,
            conflict_point=(1.2, 0.0),
            conflict_time_s=1.6,
            crossing_direction=(0.0, 1.0),
            time_scale=1.0,
            future_dt_s=0.2,
            future_steps=15,
            base_state_id="train-base-1",
            trajectory_id="traj-1",
            target_type_policy_digest="policy-digest",
            seed=13,
            context_object_ids=(),
        )

    assert exc_info.value.reason == reason


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"time_scale": 0.8}, "time_scale must equal 1.0"),
        ({"time_scale": 1.2}, "time_scale must equal 1.0"),
        ({"future_dt_s": 0.1}, "future_dt_s must equal 0.2"),
        ({"future_steps": 14}, "future_steps must equal 15"),
        ({"future_steps": 16}, "future_steps must equal 15"),
    ],
)
def test_transplant_rejects_noncanonical_time_or_output_grid(
    overrides: dict[str, object], message: str
) -> None:
    kwargs: dict[str, object] = {
        "conflict_point": (1.2, 0.0),
        "conflict_time_s": 1.6,
        "crossing_direction": (0.0, 1.0),
        "time_scale": 1.0,
        "future_dt_s": 0.2,
        "future_steps": 15,
        "base_state_id": "train-base-1",
        "trajectory_id": "traj-1",
        "target_type_policy_digest": "policy-digest",
        "seed": 13,
        "context_object_ids": (),
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        transplant_snippet(_snippet(), **kwargs)


def test_transplant_accepts_float32_canonical_future_dt() -> None:
    result = transplant_snippet(
        _snippet(),
        conflict_point=(1.2, 0.0),
        conflict_time_s=1.6,
        crossing_direction=(0.0, 1.0),
        time_scale=1.0,
        future_dt_s=np.float32(0.2),
        future_steps=15,
        base_state_id="train-base-1",
        trajectory_id="traj-1",
        target_type_policy_digest="policy-digest",
        seed=13,
        context_object_ids=(),
    )

    assert result.future_poses.shape == (15, 3)


def test_transplant_outputs_are_independently_owned_c_contiguous_arrays() -> None:
    result = transplant_snippet(
        _snippet(),
        conflict_point=(1.2, 0.0),
        conflict_time_s=1.6,
        crossing_direction=(0.0, 1.0),
        time_scale=1.0,
        future_dt_s=0.2,
        future_steps=15,
        base_state_id="train-base-1",
        trajectory_id="traj-1",
        target_type_policy_digest="policy-digest",
        seed=13,
        context_object_ids=(),
    )

    arrays = (result.history_poses, result.current_pose, result.future_poses)
    assert all(array.flags.c_contiguous for array in arrays)
    assert all(array.flags.owndata for array in arrays)
    assert not np.shares_memory(result.history_poses, result.current_pose)
    assert not np.shares_memory(result.history_poses, result.future_poses)
    assert not np.shares_memory(result.current_pose, result.future_poses)

    current_before = result.current_pose.copy()
    future_before = result.future_poses.copy()
    result.history_poses[-1, 0] += np.float32(1.0)
    np.testing.assert_array_equal(result.current_pose, current_before)
    np.testing.assert_array_equal(result.future_poses, future_before)


def test_target_id_is_deterministic_and_resolves_context_collision() -> None:
    snippet = _snippet()
    policy = normalize_target_type_policy(_policy())
    kwargs = dict(
        conflict_point=(1.2, 0.0),
        conflict_time_s=1.6,
        crossing_direction=(0.0, 1.0),
        time_scale=1.0,
        future_dt_s=0.2,
        future_steps=15,
        base_state_id="train-base-1",
        trajectory_id="traj-1",
        target_type_policy_digest=policy.digest,
        seed=13,
    )
    first = transplant_snippet(snippet, context_object_ids=(), **kwargs)
    repeated = transplant_snippet(snippet, context_object_ids=(), **kwargs)
    collision_resolved = transplant_snippet(
        snippet,
        context_object_ids=(first.target_dynamic_object_id,),
        **kwargs,
    )

    assert first.target_dynamic_object_id == repeated.target_dynamic_object_id
    np.testing.assert_array_equal(first.history_poses, repeated.history_poses)
    np.testing.assert_array_equal(first.future_poses, repeated.future_poses)
    assert collision_resolved.target_dynamic_object_id != first.target_dynamic_object_id


def _canonical_event_identity_inputs() -> dict[str, object]:
    spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.3},
    }
    return {
        "generator_algorithm_version": "joint_occluder_first_v4",
        "generator_config_digest": "1" * 32,
        "base_state_id": "base-known",
        "trajectory_id": "trajectory-known",
        "event_index": 3,
        "attempt_index": 5,
        "attempt_seed": 17,
        "event_kind": "mixed",
        "conflict_index": 8,
        "conflict_time_s": 1.8,
        "target_dynamic_object_id": "generated::human::known",
        "source_snippet_id": "snippet-known",
        "source_object_id": "recording::human",
        "object_type": "human",
        "footprint_spec": spec,
        "footprint_spec_digest": (
            target_motion_shard_module.compute_footprint_spec_digest(spec)
        ),
        "target_type_policy_digest": "3" * 32,
        "layout_version": "event_target_motion_history8_future15_v1",
    }


def _canonical_world_identity_inputs(
    lineage: dict[str, object], generated_event_id: str
) -> dict[str, object]:
    return {
        "generator_algorithm_version": lineage["generator_algorithm_version"],
        "generator_config_digest": lineage["generator_config_digest"],
        "generated_event_id": generated_event_id,
        "base_state_id": lineage["base_state_id"],
        "trajectory_id": lineage["trajectory_id"],
        "event_kind": lineage["event_kind"],
        "target_dynamic_object_id": lineage["target_dynamic_object_id"],
        "source_snippet_id": lineage["source_snippet_id"],
        "source_object_id": lineage["source_object_id"],
        "object_type": lineage["object_type"],
        "footprint_spec": lineage["footprint_spec"],
        "footprint_spec_digest": lineage["footprint_spec_digest"],
        "target_type_policy_digest": lineage["target_type_policy_digest"],
        "layout_version": lineage["layout_version"],
        "history_array_digest": "4" * 32,
        "current_pose": np.asarray([1.25, -2.5, 0.75], dtype=np.float32),
        "future_array_digest": "5" * 32,
    }


def test_event_and_world_identity_known_vectors_are_mapping_order_stable() -> None:
    lineage = _canonical_event_identity_inputs()
    spec = lineage["footprint_spec"]
    reordered_spec = {
        "footprint": {"radius_m": 0.3, "kind": "circle"},
        "object_type": "human",
    }

    generated_event_id = event_sampler_module.compute_generated_event_id(**lineage)
    assert generated_event_id == "event-8a376f991cfa77c88dda2c4d74724b05"
    assert generated_event_id == event_sampler_module._build_generated_event_id(
        **lineage
    )
    assert generated_event_id == event_sampler_module._build_generated_event_id(
        **{**lineage, "footprint_spec": reordered_spec}
    )

    world_identity = _canonical_world_identity_inputs(lineage, generated_event_id)
    world_id = event_sampler_module.compute_generated_world_id(**world_identity)
    assert world_id == "world-1c7e0b43bcc58f287741f306c0ee1ef4"
    assert world_id == event_sampler_module._build_world_id(**world_identity)
    assert world_id == event_sampler_module._build_world_id(
        **{**world_identity, "footprint_spec": reordered_spec}
    )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("event_index", True),
        ("event_index", "3"),
        ("event_index", 3.5),
        ("attempt_seed", -1),
        ("conflict_time_s", True),
        ("base_state_id", ""),
        ("generator_config_digest", "not-a-digest"),
        ("footprint_spec_digest", "2" * 32),
        ("object_type", "carried_object"),
    ],
)
def test_generated_event_identity_rejects_noncanonical_inputs(
    field_name: str, bad_value: object
) -> None:
    lineage = _canonical_event_identity_inputs()

    with pytest.raises(ValueError):
        event_sampler_module._build_generated_event_id(
            **{**lineage, field_name: bad_value}
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("generated_event_id", ""),
        ("base_state_id", ""),
        ("event_kind", ""),
        ("history_array_digest", "not-a-digest"),
        ("future_array_digest", "A" * 32),
        ("target_type_policy_digest", "3" * 31),
        ("footprint_spec_digest", "2" * 32),
        ("object_type", "carried_object"),
    ],
)
def test_world_identity_rejects_noncanonical_inputs(
    field_name: str, bad_value: object
) -> None:
    lineage = _canonical_event_identity_inputs()
    generated_event_id = event_sampler_module._build_generated_event_id(**lineage)
    world_identity = _canonical_world_identity_inputs(
        lineage, generated_event_id
    )

    with pytest.raises(ValueError):
        event_sampler_module._build_world_id(
            **{**world_identity, field_name: bad_value}
        )


def test_mother_event_lineage_is_motion_invariant_while_world_and_record_identity_are_motion_bound() -> None:
    spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.3},
    }
    spec_digest = target_motion_shard_module.compute_footprint_spec_digest(spec)
    policy_digest = "3" * 32
    lineage = {
        "generator_algorithm_version": "joint_occluder_first_v4",
        "generator_config_digest": "1" * 32,
        "base_state_id": "base-paired",
        "trajectory_id": "trajectory-paired",
        "event_index": 2,
        "attempt_index": 4,
        "attempt_seed": 29,
        "event_kind": "structural",
        "conflict_index": 7,
        "conflict_time_s": 1.6,
        "target_dynamic_object_id": "generated::human::paired",
        "source_snippet_id": "snippet-paired",
        "source_object_id": "recording::human-paired",
        "object_type": "human",
        "footprint_spec": spec,
        "footprint_spec_digest": spec_digest,
        "target_type_policy_digest": policy_digest,
        "layout_version": target_motion_shard_module.EVENT_TARGET_MOTION_LAYOUT_VERSION,
    }
    generated_event_id = event_sampler_module._build_generated_event_id(**lineage)
    assert generated_event_id == event_sampler_module._build_generated_event_id(**lineage)

    history = np.zeros((8, 3), dtype=np.float32)
    current = history[7].copy()
    future = np.zeros((15, 3), dtype=np.float32)
    variant_history = history.copy()
    variant_history[0, 0] = np.float32(0.25)
    variant_future = future.copy()
    variant_future[-1, 1] = np.float32(0.5)

    def world_id_for(history_poses: np.ndarray, future_poses: np.ndarray) -> str:
        return event_sampler_module._build_world_id(
            generator_algorithm_version=lineage["generator_algorithm_version"],
            generator_config_digest=lineage["generator_config_digest"],
            generated_event_id=generated_event_id,
            base_state_id=lineage["base_state_id"],
            trajectory_id=lineage["trajectory_id"],
            event_kind=lineage["event_kind"],
            target_dynamic_object_id=lineage["target_dynamic_object_id"],
            source_snippet_id=lineage["source_snippet_id"],
            source_object_id=lineage["source_object_id"],
            object_type=lineage["object_type"],
            footprint_spec=spec,
            footprint_spec_digest=spec_digest,
            target_type_policy_digest=policy_digest,
            layout_version=lineage["layout_version"],
            history_array_digest=target_motion_shard_module.compute_motion_array_digest(
                history_poses, field_name="target_history_poses"
            ),
            current_pose=current,
            future_array_digest=target_motion_shard_module.compute_motion_array_digest(
                future_poses, field_name="target_future_poses"
            ),
        )

    mother_world_id = world_id_for(history, future)
    variant_world_id = world_id_for(variant_history, variant_future)
    assert mother_world_id != variant_world_id

    def record_for(
        world_id: str, history_poses: np.ndarray, future_poses: np.ndarray
    ):
        return target_motion_shard_module.create_event_target_motion_record(
            generated_event_id=generated_event_id,
            world_id=world_id,
            base_state_id=lineage["base_state_id"],
            trajectory_id=lineage["trajectory_id"],
            target_dynamic_object_id=lineage["target_dynamic_object_id"],
            source_snippet_id=lineage["source_snippet_id"],
            source_object_id=lineage["source_object_id"],
            object_type=lineage["object_type"],
            footprint_spec=spec,
            footprint_spec_digest=spec_digest,
            target_type_policy_digest=policy_digest,
            history_poses=history_poses,
            current_pose=current,
            future_poses=future_poses,
        )

    mother_record = record_for(mother_world_id, history, future)
    variant_record = record_for(
        variant_world_id, variant_history, variant_future
    )
    assert mother_record.generated_event_id == variant_record.generated_event_id
    assert mother_record.record_digest != variant_record.record_digest


def test_generator_configs_freeze_human_only_policy_and_603010_mix() -> None:
    for filename in ("generator_train.yaml", "generator_test.yaml"):
        config = load_generator_config(ROOT / "configs" / filename)
        assert config["target_type_policy"].whitelist == ("human",)
        assert config["target_type_policy"].weights == {
            "human": 1.0,
            "carried_object": 0.0,
            "unknown_dynamic": 0.0,
        }
        assert config["event_type_weights"] == {
            "environment": 0.6,
            "structural": 0.3,
            "mixed": 0.1,
        }
        assert config["time_scale_range"] == (1.0, 1.0)
        assert len(config["target_type_policy"].digest) == 32


@pytest.mark.parametrize(
    "time_scale_range",
    ([0.8, 1.2], [0.8, 1.0], [1.0, 1.2]),
)
def test_generator_config_rejects_non_unit_time_scale_range(
    time_scale_range: list[float],
) -> None:
    config = _generator_config()
    config["time_scale_range"] = time_scale_range

    with pytest.raises(
        GeneratorConfigError,
        match="time_scale_range must equal \\[1.0, 1.0\\]",
    ):
        normalize_generator_config(config)


def test_event_type_schedule_preserves_603010_accepted_batch_target() -> None:
    weights = {"environment": 0.6, "structural": 0.3, "mixed": 0.1}

    first = build_event_type_schedule(
        weights, event_count=10, rng=np.random.default_rng(41)
    )
    repeated = build_event_type_schedule(
        weights, event_count=10, rng=np.random.default_rng(41)
    )

    assert first == repeated
    assert len(first) == 10
    assert {kind: first.count(kind) for kind in weights} == {
        "environment": 6,
        "structural": 3,
        "mixed": 1,
    }


@pytest.mark.parametrize("conflict_index", range(4, 11))
def test_float32_unit_circle_at_curvature_limit_is_allowed(
    conflict_index: int,
    constant_curvature_inputs: tuple[dict, LocalTrajectory, LocalTrajectory],
) -> None:
    config, trajectory, _ = constant_curvature_inputs
    generator_config = _generator_config("structural")
    conflict_time_s = (conflict_index + 1) * float(
        config["bev"]["future_dt_s"]
    )
    generator_config["conflict_time_range_s"] = [
        conflict_time_s,
        conflict_time_s,
    ]
    generator_config["max_local_curvature_per_m"] = 1.0
    normalized = normalize_generator_config(generator_config)

    index, actual_conflict_time_s, conflict_point, tangent, normal = (
        event_sampler_module._trajectory_geometry(
            trajectory=trajectory,
            dt_s=float(config["bev"]["future_dt_s"]),
            conflict_range=normalized["conflict_time_range_s"],
            rng=np.random.default_rng(23),
            max_curvature=normalized["max_local_curvature_per_m"],
        )
    )

    assert abs(
        float(
            trajectory.controls[conflict_index, 1]
            / trajectory.controls[conflict_index, 0]
        )
    ) == pytest.approx(1.0, abs=1e-7)
    assert index == conflict_index
    assert actual_conflict_time_s == pytest.approx(conflict_time_s)
    assert np.isfinite(conflict_point).all()
    assert np.isfinite(tangent).all()
    assert np.isfinite(normal).all()


@pytest.mark.parametrize("conflict_index", range(4, 11))
def test_float32_circle_with_true_curvature_1_01_is_rejected(
    conflict_index: int,
    constant_curvature_inputs: tuple[dict, LocalTrajectory, LocalTrajectory],
) -> None:
    config, _, trajectory = constant_curvature_inputs
    generator_config = _generator_config("structural")
    conflict_time_s = (conflict_index + 1) * float(
        config["bev"]["future_dt_s"]
    )
    generator_config["conflict_time_range_s"] = [
        conflict_time_s,
        conflict_time_s,
    ]
    generator_config["max_local_curvature_per_m"] = 1.0
    normalized = normalize_generator_config(generator_config)

    assert abs(
        float(
            trajectory.controls[conflict_index, 1]
            / trajectory.controls[conflict_index, 0]
        )
    ) == pytest.approx(1.01, abs=1e-7)
    with pytest.raises(event_sampler_module._EventRejection) as exc_info:
        event_sampler_module._trajectory_geometry(
            trajectory=trajectory,
            dt_s=float(config["bev"]["future_dt_s"]),
            conflict_range=normalized["conflict_time_range_s"],
            rng=np.random.default_rng(23),
            max_curvature=normalized["max_local_curvature_per_m"],
        )

    assert exc_info.value.reason == "conflict_curvature"


def test_float32_curvature_check_preserves_degenerate_tangent_rejection(
    constant_curvature_inputs: tuple[dict, LocalTrajectory, LocalTrajectory],
) -> None:
    config, trajectory, _ = constant_curvature_inputs
    conflict_index = 5
    poses = trajectory.poses.copy()
    poses[conflict_index] = poses[conflict_index - 1]
    degenerate = replace(trajectory, poses=poses)
    conflict_time_s = (conflict_index + 1) * float(
        config["bev"]["future_dt_s"]
    )

    with pytest.raises(event_sampler_module._EventRejection) as exc_info:
        event_sampler_module._trajectory_geometry(
            trajectory=degenerate,
            dt_s=float(config["bev"]["future_dt_s"]),
            conflict_range=(conflict_time_s, conflict_time_s),
            rng=np.random.default_rng(23),
            max_curvature=1.0,
        )

    assert exc_info.value.reason == "conflict_tangent_degenerate"


def test_generated_event_carries_canonical_target_motion_record_and_world_join() -> None:
    config, grid, base, oracle, trajectory, libraries = _base_inputs()
    report = generate_events(
        base_state=base,
        oracle_context=oracle,
        trajectory=trajectory,
        snippet_libraries=libraries,
        base_config=config,
        generator_config=normalize_generator_config(
            _generator_config("structural")
        ),
        seed=23,
        event_count=1,
    )

    assert len(report.events) == 1
    event = report.events[0]
    record = event.target_motion_record
    metadata = event.world.metadata
    target_id = event.target.target_dynamic_object_id

    assert event.generated_event_id.startswith("event-")
    assert len(event.generated_event_id) == len("event-") + 32
    assert event.generated_event_id == record.generated_event_id
    assert event.world.world_id == record.world_id == metadata["world_id"]
    assert record.base_state_id == base.state_id == metadata["base_state_id"]
    assert record.trajectory_id == trajectory.trajectory_id
    assert record.trajectory_id == metadata["trajectory_id"]
    assert record.target_dynamic_object_id == target_id
    assert metadata["target_dynamic_object_id"] == target_id
    assert record.source_snippet_id == event.target.snippet_id
    assert metadata["source_snippet_id"] == record.source_snippet_id
    assert metadata["dynamic_object_snippet_id"] == record.source_snippet_id
    assert record.source_object_id == event.target.source_object_id
    assert metadata["source_object_id"] == record.source_object_id
    assert record.object_type == event.target.object_type
    assert metadata["target_object_type"] == record.object_type
    assert record.footprint_spec == event.target.footprint_spec
    assert metadata["target_footprint_spec"] == record.footprint_spec
    assert record.footprint_spec_digest == event.target.footprint_spec_digest
    assert metadata["target_footprint_spec_digest"] == record.footprint_spec_digest
    assert record.target_type_policy_digest == report.summary[
        "target_type_policy_digest"
    ]
    assert metadata["target_type_policy_digest"] == (
        record.target_type_policy_digest
    )
    assert metadata["event_target_motion_layout_version"] == record.layout_version
    assert metadata["target_history_array_digest"] == record.history_array_digest
    assert metadata["target_future_array_digest"] == record.future_array_digest
    assert metadata["target_motion_record_digest"] == record.record_digest
    assert metadata["event_slot_index"] == 0
    assert metadata["attempt_index"] == 0
    assert metadata["target_current_pose"] == [
        float(value) for value in record.current_pose
    ]
    np.testing.assert_array_equal(record.history_poses, event.target.history_poses)
    np.testing.assert_array_equal(record.current_pose, event.target.current_pose)
    np.testing.assert_array_equal(record.future_poses, event.target.future_poses)
    np.testing.assert_array_equal(
        event.world.dynamic_object_trajectories[target_id], record.future_poses
    )
    assert event.world.dynamic_object_specs[target_id] == record.footprint_spec
    target_motion_shard_module.validate_event_target_motion_world_join(
        record, event.world, grid
    )


def test_real_generated_event_batch_round_trips_through_strict_motion_shard(
    tmp_path: Path,
) -> None:
    config, grid, base, oracle, trajectory, libraries = _base_inputs()
    report = generate_events(
        base_state=base,
        oracle_context=oracle,
        trajectory=trajectory,
        snippet_libraries=libraries,
        base_config=config,
        generator_config=normalize_generator_config(
            _generator_config("structural")
        ),
        seed=23,
        event_count=2,
    )
    assert len(report.events) == 2
    records = tuple(event.target_motion_record for event in report.events)
    worlds = tuple(event.world for event in report.events)

    output = tmp_path / "generated-events"
    target_motion_shard_module.write_event_target_motion_shard(
        records, worlds, output, grid=grid
    )
    loaded = target_motion_shard_module.load_event_target_motion_shard(
        output,
        grid=grid,
        expected_generated_event_ids={
            event.generated_event_id for event in report.events
        },
        expected_base_state_ids={base.state_id},
        expected_trajectory_ids={trajectory.trajectory_id},
    )

    assert [record.generated_event_id for record in loaded.records] == sorted(
        event.generated_event_id for event in report.events
    )
    source_by_event_id = {
        event.generated_event_id: event for event in report.events
    }
    for record in loaded.records:
        source = source_by_event_id[record.generated_event_id]
        assert record.record_digest == source.target_motion_record.record_digest
        np.testing.assert_array_equal(
            record.history_poses, source.target.history_poses
        )
        np.testing.assert_array_equal(
            record.future_poses, source.target.future_poses
        )
        target_motion_shard_module.validate_event_target_motion_world_join(
            record, loaded.worlds[record.world_id], grid
        )


def test_same_seed_generated_event_shards_are_byte_identical(
    tmp_path: Path,
) -> None:
    config, grid, base, oracle, trajectory, libraries = _base_inputs()
    generator_config = normalize_generator_config(_generator_config("structural"))

    def generate_and_write(output: Path):
        report = generate_events(
            base_state=base,
            oracle_context=oracle,
            trajectory=trajectory,
            snippet_libraries=libraries,
            base_config=config,
            generator_config=generator_config,
            seed=23,
            event_count=2,
        )
        assert len(report.events) == 2
        target_motion_shard_module.write_event_target_motion_shard(
            tuple(event.target_motion_record for event in report.events),
            tuple(event.world for event in report.events),
            output,
            grid=grid,
        )
        return report

    left_report = generate_and_write(tmp_path / "left")
    right_report = generate_and_write(tmp_path / "right")
    left_files = {
        path.relative_to(tmp_path / "left"): path.read_bytes()
        for path in sorted((tmp_path / "left").rglob("*"))
        if path.is_file()
    }
    right_files = {
        path.relative_to(tmp_path / "right"): path.read_bytes()
        for path in sorted((tmp_path / "right").rglob("*"))
        if path.is_file()
    }
    assert left_files == right_files
    assert [event.generated_event_id for event in left_report.events] == [
        event.generated_event_id for event in right_report.events
    ]

    left_loaded = target_motion_shard_module.load_event_target_motion_shard(
        tmp_path / "left", grid=grid
    )
    right_loaded = target_motion_shard_module.load_event_target_motion_shard(
        tmp_path / "right", grid=grid
    )
    assert left_loaded.manifest_digest == right_loaded.manifest_digest
    assert left_loaded.payload_semantic_digest == (
        right_loaded.payload_semantic_digest
    )


def test_generate_event_preserves_context_and_is_elementwise_deterministic() -> None:
    config, grid, base, oracle, trajectory, libraries = _base_inputs()
    generator_config = normalize_generator_config(_generator_config("structural"))

    first = generate_events(
        base_state=base,
        oracle_context=oracle,
        trajectory=trajectory,
        snippet_libraries=libraries,
        base_config=config,
        generator_config=generator_config,
        seed=23,
        event_count=1,
    )
    second = generate_events(
        base_state=base,
        oracle_context=oracle,
        trajectory=trajectory,
        snippet_libraries=libraries,
        base_config=config,
        generator_config=generator_config,
        seed=23,
        event_count=1,
    )

    assert len(first.events) == 1
    assert len(second.events) == 1
    event = first.events[0]
    repeated = second.events[0]
    assert event.generated_event_id == repeated.generated_event_id
    assert event.target_motion_record.generated_event_id == event.generated_event_id
    assert repeated.target_motion_record.generated_event_id == repeated.generated_event_id
    assert event.target_motion_record.world_id == repeated.target_motion_record.world_id
    assert event.target_motion_record.record_digest == (
        repeated.target_motion_record.record_digest
    )
    for field_name in ("history_poses", "current_pose", "future_poses"):
        assert getattr(event.target_motion_record, field_name).tobytes(order="C") == (
            getattr(repeated.target_motion_record, field_name).tobytes(order="C")
        )
    validate_oracle_world(event.world, grid)
    context_id = next(iter(oracle.dynamic_object_future))
    assert context_id in event.world.dynamic_object_trajectories
    np.testing.assert_array_equal(
        event.world.dynamic_object_trajectories[context_id],
        oracle.dynamic_object_future[context_id],
    )
    assert event.world.dynamic_object_specs[context_id] == oracle.dynamic_object_specs[context_id]
    assert event.target.target_dynamic_object_id in event.world.dynamic_object_trajectories
    assert event.target_visibility_history.shape == (8,)
    assert event.target_visibility_history.dtype == np.bool_
    assert repeated.target_visibility_history.shape == (8,)
    assert repeated.target_visibility_history.dtype == np.bool_
    assert event.visibility_sequence.shape == (16,)
    assert event.visibility_sequence.dtype == np.bool_
    assert repeated.visibility_sequence.shape == (16,)
    assert repeated.visibility_sequence.dtype == np.bool_
    assert not bool(event.target_visibility_history[7])
    assert not bool(event.visibility_sequence[0])
    assert event.target_visibility_history[7] == event.visibility_sequence[0]
    assert bool(event.visibility_sequence[-1])
    assert event.world.world_id == repeated.world.world_id
    assert event.world.metadata == repeated.world.metadata
    np.testing.assert_array_equal(event.visibility_sequence, repeated.visibility_sequence)
    np.testing.assert_array_equal(
        event.target_visibility_history,
        repeated.target_visibility_history,
    )
    assert event.world.metadata["target_visibility_history"] == [
        bool(value) for value in event.target_visibility_history
    ]
    assert all(
        type(value) is bool
        for value in event.world.metadata["target_visibility_history"]
    )
    assert event.world.metadata["target_visibility_history_layout"] == (
        "target_visibility_history8_current7_v1"
    )
    for object_id in event.world.dynamic_object_trajectories:
        np.testing.assert_array_equal(
            event.world.dynamic_object_trajectories[object_id],
            repeated.world.dynamic_object_trajectories[object_id],
        )
    assert first.summary == second.summary
    assert first.summary["by_object_type"]["human"]["accepted"] == 1
    assert first.summary["by_footprint_kind"]["circle"]["accepted"] == 1
    assert first.summary["by_geometry_source"]["marker_extent_p95"]["accepted"] == 1


def test_target_visibility_history_recomputes_moving_sensor_and_context_per_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, grid, base, oracle, _, _ = _base_inputs()
    context_id = next(iter(oracle.dynamic_object_history))
    robot_history = np.column_stack(
        (
            np.linspace(-0.7, 0.0, grid.history_steps, dtype=np.float32),
            np.zeros(grid.history_steps, dtype=np.float32),
            np.zeros(grid.history_steps, dtype=np.float32),
        )
    )
    context_history = np.column_stack(
        (
            np.linspace(-3.0, -1.6, grid.history_steps, dtype=np.float32),
            np.full(grid.history_steps, 2.0, dtype=np.float32),
            np.zeros(grid.history_steps, dtype=np.float32),
        )
    )
    moving_base = replace(base, robot_history=robot_history)
    moving_oracle = replace(
        oracle,
        dynamic_object_history={context_id: context_history},
    )
    target = _valid_physics_target()
    target_history = np.column_stack(
        (
            np.linspace(1.0, 1.6, grid.history_steps, dtype=np.float32),
            np.full(grid.history_steps, -2.0, dtype=np.float32),
            np.zeros(grid.history_steps, dtype=np.float32),
        )
    )
    target = replace(
        target,
        history_poses=target_history,
        current_pose=target_history[-1].copy(),
    )
    target_footprint = footprint_from_spec(target.footprint_spec)
    context_footprint = RectangleFootprint(0.8, 0.2)
    static = rasterize_footprint(
        CircleFootprint(0.05),
        np.asarray([4.0, 4.0, 0.0], dtype=np.float32),
        grid,
    ).astype(np.float32)
    structural = StructuralBlindSpot(
        forward_fov_deg=160.0,
        range_m=6.0,
        blind_sectors=(),
    )
    observed: list[tuple[np.ndarray, np.ndarray, StructuralBlindSpot]] = []

    def controlled_structural_visibility(
        occupancy: np.ndarray,
        _grid,
        *,
        sensor_pose: np.ndarray,
        blind_spot: StructuralBlindSpot,
    ) -> np.ndarray:
        history_index = len(observed)
        observed.append((occupancy.copy(), np.asarray(sensor_pose).copy(), blind_spot))
        if history_index == grid.history_steps - 1:
            return np.zeros((grid.height, grid.width), dtype=bool)
        return rasterize_footprint(
            target_footprint,
            target.history_poses[history_index],
            grid,
        )

    monkeypatch.setattr(
        event_sampler_module,
        "build_structural_visibility",
        controlled_structural_visibility,
    )

    history_visibility = event_sampler_module._target_visibility_history(
        event_kind="structural",
        static_occupancy=static,
        placement=None,
        grid=grid,
        base_state=moving_base,
        oracle_context=moving_oracle,
        context_footprints={context_id: context_footprint},
        target=target,
        target_footprint=target_footprint,
        structural=structural,
    )

    np.testing.assert_array_equal(
        history_visibility,
        np.asarray([True] * 7 + [False], dtype=bool),
    )
    assert history_visibility.shape == (8,)
    assert history_visibility.dtype == np.bool_
    assert len(observed) == grid.history_steps
    for index, (occupied, sensor_pose, used_structural) in enumerate(observed):
        expected_context = rasterize_footprint(
            context_footprint,
            context_history[index],
            grid,
        )
        np.testing.assert_array_equal(
            occupied,
            static.astype(bool) | expected_context,
        )
        np.testing.assert_array_equal(sensor_pose, robot_history[index])
        assert used_structural is structural


@pytest.mark.parametrize("event_kind", ["structural", "mixed"])
def test_structural_visibility_attempt_evaluates_only_one_preselected_candidate(
    event_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, grid, base, oracle, trajectory, _ = _base_inputs()
    target = _valid_physics_target()
    target_footprint = footprint_from_spec(target.footprint_spec)
    candidate = StructuralBlindSpot(
        forward_fov_deg=160.0,
        range_m=6.0,
        blind_sectors=(),
    )
    evaluated: list[StructuralBlindSpot] = []

    def always_invalid(
        occupancy: np.ndarray,
        _grid,
        *,
        sensor_pose: np.ndarray,
        blind_spot: StructuralBlindSpot,
    ) -> np.ndarray:
        evaluated.append(blind_spot)
        return np.zeros_like(occupancy, dtype=bool)

    monkeypatch.setattr(
        event_sampler_module,
        "build_structural_visibility",
        always_invalid,
    )
    placement = None
    if event_kind == "mixed":
        placement = type(
            "PlacementFixture",
            (),
            {
                "mask": np.zeros((grid.height, grid.width), dtype=bool),
                "rejection_reasons": {},
            },
        )()

    with pytest.raises(event_sampler_module._EventRejection) as exc_info:
        event_sampler_module._visibility_for_event(
            event_kind=event_kind,
            static_occupancy=np.zeros((grid.height, grid.width), dtype=np.float32),
            context_current_occupancy=np.zeros(
                (grid.height, grid.width), dtype=bool
            ),
            grid=grid,
            sensor_pose=base.robot_history[-1],
            target=target,
            target_footprint=target_footprint,
            trajectory=trajectory,
            robot_footprint=RectangleFootprint(0.8, 0.6),
            conflict_point=np.asarray([1.2, 0.0], dtype=np.float32),
            normal=np.asarray([0.0, 1.0], dtype=np.float32),
            oracle_context=oracle,
            generator_config=normalize_generator_config(_generator_config(event_kind)),
            rng=np.random.default_rng(29),
            structural_candidate=candidate,
            precomputed_placement=placement,
        )

    assert exc_info.value.reason == "structural_visibility_invalid"
    assert evaluated == [candidate]


def test_environment_occluder_receives_complete_robot_and_context_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, base, oracle, trajectory, libraries = _base_inputs()
    raw_generator_config = _generator_config("environment")
    raw_generator_config["max_resample_attempts"] = 1
    captured: list[tuple[object, ...]] = []
    real_propose = event_sampler_module.propose_environment_occluder_geometry

    def capture_sweeps(**kwargs: object):
        captured.append(tuple(kwargs.get("collision_sweeps", ())))
        return real_propose(**kwargs)

    monkeypatch.setattr(
        event_sampler_module,
        "propose_environment_occluder_geometry",
        capture_sweeps,
    )

    generate_events(
        base_state=base,
        oracle_context=oracle,
        trajectory=trajectory,
        snippet_libraries=libraries,
        base_config=config,
        generator_config=normalize_generator_config(raw_generator_config),
        seed=20260716,
        event_count=1,
    )

    assert len(captured) == 1
    assert [sweep.rejection_reason for sweep in captured[0]] == [
        "occluder_robot_swept_overlap",
        "occluder_context_collision",
    ]
    np.testing.assert_array_equal(
        captured[0][0].poses,
        np.vstack((base.robot_history, trajectory.poses)),
    )
    context_id = next(iter(oracle.dynamic_object_specs))
    np.testing.assert_array_equal(
        captured[0][1].poses,
        np.vstack(
            (
                oracle.dynamic_object_history[context_id],
                oracle.dynamic_object_future[context_id],
            )
        ),
    )
    assert all(sweep.poses.shape == (23, 3) for sweep in captured[0])


def test_target_physics_rejects_static_collision_only_between_early_history_frames(
) -> None:
    config, grid, base, oracle, _, _ = _base_inputs()
    target = _valid_physics_target()
    history = target.history_poses.copy()
    history[0] = np.asarray([-4.0, 3.0, 0.0], dtype=np.float32)
    history[1] = np.asarray([-3.0, 3.0, 0.0], dtype=np.float32)
    target = replace(target, history_poses=history)
    static = rasterize_footprint(
        CircleFootprint(0.01),
        np.asarray([-3.5, 3.0, 0.0], dtype=np.float32),
        grid,
    ).astype(np.float32)
    discrete_sweep = rasterize_footprint_sweep(
        footprint_from_spec(target.footprint_spec),
        np.vstack((target.history_poses, target.future_poses)),
        grid,
    )
    assert not np.any(discrete_sweep & static.astype(bool))

    with pytest.raises(event_sampler_module._EventRejection) as exc_info:
        event_sampler_module._validate_target_physics(
            target,
            base_state=replace(base, static_map_local=static),
            oracle_context=oracle,
            base_config=config,
        )

    assert exc_info.value.reason == "target_static_collision"


def test_target_physics_rejects_collision_only_in_context_history() -> None:
    config, _, base, oracle, _, _ = _base_inputs()
    target = _valid_physics_target()
    context_id = next(iter(oracle.dynamic_object_history))
    context_history = np.tile(
        np.asarray([-6.0, -6.0, 0.0], dtype=np.float32),
        (8, 1),
    )
    context_history[3] = target.history_poses[3]
    context_future = np.tile(
        np.asarray([-6.0, -6.0, 0.0], dtype=np.float32),
        (15, 1),
    )
    history_only_collision = replace(
        oracle,
        dynamic_object_history={context_id: context_history},
        dynamic_object_future={context_id: context_future},
    )

    with pytest.raises(event_sampler_module._EventRejection) as exc_info:
        event_sampler_module._validate_target_physics(
            target,
            base_state=base,
            oracle_context=history_only_collision,
            base_config=config,
        )

    assert exc_info.value.reason == "target_context_collision"


def test_target_physics_rejects_speed_jump_only_in_history() -> None:
    config, _, base, oracle, _, _ = _base_inputs()
    target = _valid_physics_target()
    history = target.history_poses.copy()
    history[0, 0] = np.float32(-4.0)
    target = replace(target, history_poses=history)

    with pytest.raises(event_sampler_module._EventRejection) as exc_info:
        event_sampler_module._validate_target_physics(
            target,
            base_state=base,
            oracle_context=oracle,
            base_config=config,
        )

    assert exc_info.value.reason == "target_speed_limit"


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("history-shape", "target_history_shape_invalid"),
        ("future-shape", "target_future_shape_invalid"),
        ("dtype", "target_motion_dtype_invalid"),
        ("nonfinite", "target_motion_nonfinite"),
        ("seam", "target_current_history_mismatch"),
    ],
)
def test_target_physics_rejects_invalid_complete_motion_contract(
    case: str,
    reason: str,
) -> None:
    config, _, base, oracle, _, _ = _base_inputs()
    target = _valid_physics_target()
    if case == "history-shape":
        target = replace(target, history_poses=target.history_poses[:-1].copy())
    elif case == "future-shape":
        target = replace(target, future_poses=target.future_poses[:-1].copy())
    elif case == "dtype":
        target = replace(
            target,
            history_poses=target.history_poses.astype(np.float64),
        )
    elif case == "nonfinite":
        history = target.history_poses.copy()
        history[0, 0] = np.nan
        target = replace(target, history_poses=history)
    else:
        current = target.current_pose.copy()
        current[0] += np.float32(0.1)
        target = replace(target, current_pose=current)

    with pytest.raises(event_sampler_module._EventRejection) as exc_info:
        event_sampler_module._validate_target_physics(
            target,
            base_state=base,
            oracle_context=oracle,
            base_config=config,
        )

    assert exc_info.value.reason == reason


def test_impossible_world_exhausts_finite_retries_with_explicit_reason() -> None:
    config, _, base, oracle, trajectory, libraries = _base_inputs()
    impossible = replace(
        base,
        static_map_local=np.ones_like(base.static_map_local, dtype=np.float32),
    )
    raw_generator_config = _generator_config("structural")
    raw_generator_config["max_resample_attempts"] = 3
    report = generate_events(
        base_state=impossible,
        oracle_context=oracle,
        trajectory=trajectory,
        snippet_libraries=libraries,
        base_config=config,
        generator_config=normalize_generator_config(raw_generator_config),
        seed=31,
        event_count=1,
    )

    assert report.events == ()
    assert report.summary["attempted_count"] == 3
    assert report.summary["accepted_count"] == 0
    assert report.summary["rejected_count"] == 3
    assert report.summary["rejection_reasons"] == {"target_static_collision": 3}
    assert report.summary["by_object_type"]["human"]["rejected"] == 3
    structural_summary = report.summary["by_event_kind"]["structural"]
    assert structural_summary == {
        "requested": 1,
        "attempted": 3,
        "accepted": 0,
        "rejected": 3,
        "request_acceptance_rate": 0.0,
        "attempt_acceptance_rate": 0.0,
        "rejection_reasons": {"target_static_collision": 3},
        "rejection_stage_counts": {
            "occluder_geometry": 0,
            "target_conditioning": 3,
            "visibility": 0,
        },
    }


def test_summary_uses_complete_joint_attempt_denominator_and_reconciles_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, base, oracle, trajectory, libraries = _base_inputs()
    raw_generator_config = _generator_config("structural")
    raw_generator_config["max_resample_attempts"] = 3
    real_trajectory_geometry = event_sampler_module._trajectory_geometry
    geometry_call_count = 0

    def reject_first_two_candidates(*args: object, **kwargs: object):
        nonlocal geometry_call_count
        geometry_call_count += 1
        if geometry_call_count <= 2:
            raise event_sampler_module._EventRejection("forced_before_target")
        return real_trajectory_geometry(*args, **kwargs)

    def controlled_footprint_visibility(
        footprint,
        poses: np.ndarray,
        visibility_mask: np.ndarray,
        grid,
    ) -> np.ndarray:
        del footprint, visibility_mask, grid
        pose_count = np.asarray(poses).shape[0]
        if pose_count == 16:
            return np.asarray([False] + [True] * 15, dtype=bool)
        assert pose_count == 1
        return np.asarray([False], dtype=bool)

    monkeypatch.setattr(
        event_sampler_module,
        "_trajectory_geometry",
        reject_first_two_candidates,
    )
    monkeypatch.setattr(
        event_sampler_module,
        "footprint_visibility_sequence",
        controlled_footprint_visibility,
    )

    report = generate_events(
        base_state=base,
        oracle_context=oracle,
        trajectory=trajectory,
        snippet_libraries=libraries,
        base_config=config,
        generator_config=normalize_generator_config(raw_generator_config),
        seed=43,
        event_count=1,
    )

    assert len(report.events) == 1
    summary = report.summary
    assert summary["complete_joint_candidates_attempted"] == 3
    assert summary["attempted_count"] == 3
    assert summary["joint_candidate_attempted_count"] == 3
    assert summary["accepted_count"] == 1
    assert summary["rejected_count"] == 2
    assert summary["attempt_acceptance_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["request_acceptance_rate"] == 1.0
    assert summary["acceptance_rate"] == summary["request_acceptance_rate"]
    assert summary["rejection_reasons"] == {"forced_before_target": 2}

    assigned_keys = {
        "by_object_type": "human",
        "by_footprint_kind": "circle",
        "by_geometry_source": "marker_extent_p95",
    }
    for dimension, assigned_key in assigned_keys.items():
        buckets = summary[dimension]
        assert sum(bucket["attempted"] for bucket in buckets.values()) == 3
        assert buckets["unassigned"] == {
            "attempted": 2,
            "accepted": 0,
            "rejected": 2,
            "attempt_acceptance_rate": 0.0,
            "rejection_reasons": {"forced_before_target": 2},
        }
        assert buckets[assigned_key] == {
            "attempted": 1,
            "accepted": 1,
            "rejected": 0,
            "attempt_acceptance_rate": 1.0,
            "rejection_reasons": {},
        }


def test_generate_events_resumes_absolute_candidate_index_without_reseeding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, base, oracle, trajectory, libraries = _base_inputs()
    raw_generator_config = _generator_config("structural")
    raw_generator_config["max_resample_attempts"] = 3
    normalized = normalize_generator_config(raw_generator_config)
    real_trajectory_geometry = event_sampler_module._trajectory_geometry
    geometry_call_count = 0

    def reject_first_two_candidates(*args: object, **kwargs: object):
        nonlocal geometry_call_count
        geometry_call_count += 1
        if geometry_call_count <= 2:
            raise event_sampler_module._EventRejection("forced_before_target")
        return real_trajectory_geometry(*args, **kwargs)

    def controlled_footprint_visibility(
        footprint,
        poses: np.ndarray,
        visibility_mask: np.ndarray,
        grid,
    ) -> np.ndarray:
        del footprint, visibility_mask, grid
        pose_count = np.asarray(poses).shape[0]
        if pose_count == 16:
            return np.asarray([False] + [True] * 15, dtype=bool)
        assert pose_count == 1
        return np.asarray([False], dtype=bool)

    monkeypatch.setattr(
        event_sampler_module,
        "_trajectory_geometry",
        reject_first_two_candidates,
    )
    monkeypatch.setattr(
        event_sampler_module,
        "footprint_visibility_sequence",
        controlled_footprint_visibility,
    )
    uninterrupted = generate_events(
        base_state=base,
        oracle_context=oracle,
        trajectory=trajectory,
        snippet_libraries=libraries,
        base_config=config,
        generator_config=normalized,
        seed=43,
        event_count=1,
    )
    assert uninterrupted.summary["attempted_count"] == 3
    assert len(uninterrupted.events) == 1

    monkeypatch.setattr(
        event_sampler_module,
        "_trajectory_geometry",
        real_trajectory_geometry,
    )
    resumed = generate_events(
        base_state=base,
        oracle_context=oracle,
        trajectory=trajectory,
        snippet_libraries=libraries,
        base_config=config,
        generator_config=normalized,
        seed=43,
        event_count=1,
        attempt_index_start=2,
    )

    assert resumed.summary["attempt_index_start"] == 2
    assert resumed.summary["attempt_index_stop_exclusive"] == 3
    assert resumed.summary["attempted_count"] == 1
    assert len(resumed.events) == 1
    expected = uninterrupted.events[0]
    actual = resumed.events[0]
    assert actual.generated_event_id == expected.generated_event_id
    assert actual.world.world_id == expected.world.world_id
    assert actual.world.metadata["attempt_index"] == 2
    np.testing.assert_array_equal(
        actual.target.history_poses,
        expected.target.history_poses,
    )
    np.testing.assert_array_equal(
        actual.target.future_poses,
        expected.target.future_poses,
    )


def test_rectangle_target_uses_yaw_when_checking_static_collision() -> None:
    config, grid, base, oracle, trajectory, _ = _base_inputs()
    rectangle = _snippet(
        object_type="carried_object",
        footprint={"kind": "rectangle", "length_m": 1.2, "width_m": 0.2},
    )
    policy_config = _generator_config("structural")
    policy_config["target_type_policy"] = {
        "whitelist": ["carried_object"],
        "weights": {
            "human": 0.0,
            "carried_object": 1.0,
            "unknown_dynamic": 0.0,
        },
    }
    libraries = {
        "carried_object": SnippetLibrary(
            object_type="carried_object",
            snippets=(rectangle,),
            summary={"split": "train", "accepted_count": 1},
        )
    }
    report = generate_events(
        base_state=base,
        oracle_context=oracle,
        trajectory=trajectory,
        snippet_libraries=libraries,
        base_config=config,
        generator_config=normalize_generator_config(policy_config),
        seed=37,
        event_count=1,
    )

    assert len(report.events) == 1
    event = report.events[0]
    target_mask = rasterize_footprint_sweep(
        RectangleFootprint(1.2, 0.2),
        np.vstack((event.target.current_pose, event.target.future_poses)),
        grid,
    )
    assert target_mask.any()
    assert np.unique(np.round(event.target.future_poses[:, 2], decimals=5)).size > 1


def test_environment_generation_uses_single_layer_joint_attempts() -> None:
    config, _, base, oracle, trajectory, libraries = _base_inputs()
    report = generate_events(
        base_state=base,
        oracle_context=oracle,
        trajectory=trajectory,
        snippet_libraries=libraries,
        base_config=config,
        generator_config=load_generator_config(
            ROOT / "configs" / "generator_train.yaml"
        ),
        seed=20260716,
        event_count=1,
    )

    assert len(report.events) == 1
    assert report.summary["generator_algorithm_version"] == "joint_occluder_first_v4"
    assert report.summary["joint_candidate_attempted_count"] == report.summary[
        "attempted_count"
    ]
    assert report.summary["attempt_acceptance_rate"] >= 0.5
    assert report.summary["rejection_stage_counts"] == {
        "occluder_geometry": 0,
        "target_conditioning": 0,
        "visibility": 0,
    }
    environment_summary = report.summary["by_event_kind"]["environment"]
    assert environment_summary["requested"] == 1
    assert environment_summary["attempted"] == report.summary["attempted_count"]
    assert environment_summary["accepted"] == 1
    assert environment_summary["rejected"] == environment_summary["attempted"] - 1
    assert environment_summary["request_acceptance_rate"] == 1.0
    assert environment_summary["attempt_acceptance_rate"] == pytest.approx(
        1.0 / environment_summary["attempted"]
    )
    assert sum(environment_summary["rejection_stage_counts"].values()) == (
        environment_summary["rejected"]
    )
    event = report.events[0]
    assert event.world.metadata["generator_algorithm_version"] == (
        "joint_occluder_first_v4"
    )
    assert event.world.occluders[0]["placement_strategy"] == (
        "joint_occluder_first_v2"
    )
    robot_footprint = inflate_footprint(
        RectangleFootprint(
            config["robot"]["length_m"], config["robot"]["width_m"]
        ),
        config["robot"]["inflation_m"],
    )
    clearances = trajectory_signed_clearances(
        robot_footprint,
        trajectory.poses,
        footprint_from_spec(event.target.footprint_spec),
        event.target.future_poses,
    )
    assert float(np.min(clearances)) <= 0.0


def test_environment_physics_prefix_accepts_first_candidate_for_fixture_batch() -> None:
    config, _, base, oracle, trajectory, libraries = _base_inputs()
    generator_config = load_generator_config(
        ROOT / "configs" / "generator_train.yaml"
    )
    generator_config["event_type_weights"] = {
        "environment": 1.0,
        "structural": 0.0,
        "mixed": 0.0,
    }
    generator_config["max_resample_attempts"] = 1

    report = generate_events(
        base_state=base,
        oracle_context=oracle,
        trajectory=trajectory,
        snippet_libraries=libraries,
        base_config=config,
        generator_config=generator_config,
        seed=20260716,
        event_count=4,
    )

    assert len(report.events) == 4
    assert report.summary["attempted_count"] == 4
    assert report.summary["attempt_acceptance_rate"] == 1.0
    assert report.summary["rejection_reasons"] == {}
    assert all(event.conflict_time_s == 2.2 for event in report.events)
    occluder_poses = {
        tuple(np.round(event.world.occluders[0]["pose"], decimals=4))
        for event in report.events
    }
    assert len(occluder_poses) >= 3
