"""Typed transplant, policy, event orchestration, and determinism tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.contracts import (
    BaseState,
    OracleContext,
    build_grid_spec,
    validate_oracle_world,
)
from src.datasets.snippet_library import MotionSnippet, SnippetLibrary
from src.generation.dynamic_object_transplant import (
    TargetPolicyError,
    footprint_from_spec,
    normalize_target_type_policy,
    transplant_snippet,
)
from src.generation.event_sampler import (
    build_event_type_schedule,
    generate_events,
    load_generator_config,
    normalize_generator_config,
)
from src.geometry import (
    CircleFootprint,
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint_sweep,
    trajectory_signed_clearances,
)
from src.planning.query_maps import build_local_trajectory
from src.planning.trajectory_sampler import sample_candidate_rollouts
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
    times = np.linspace(0.0, 3.0, 16, dtype=np.float32)
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
        duration_s=3.0,
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
def test_transplant_preserves_frozen_type_footprint_and_frame_yaw(
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
    all_poses = np.vstack((result.current_pose, result.future_poses))

    assert result.object_type == object_type
    assert result.footprint_spec == {"object_type": object_type, "footprint": footprint}
    assert result.source_object_id == snippet.source_object_id
    assert result.target_dynamic_object_id != snippet.source_object_id
    assert result.target_dynamic_object_id not in {"context-object"}
    assert result.current_pose.dtype == np.float32
    assert result.future_poses.dtype == np.float32
    assert result.future_poses.shape == (15, 3)
    np.testing.assert_allclose(all_poses[8, :2], [1.2, 0.0], atol=1e-6)
    source_yaw_delta = np.unwrap(snippet.headings) - float(snippet.headings[0])
    result_yaw_delta = np.unwrap(all_poses[:, 2]) - float(all_poses[0, 2])
    np.testing.assert_allclose(result_yaw_delta, source_yaw_delta, atol=1e-6)


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
    np.testing.assert_array_equal(first.future_poses, repeated.future_poses)
    assert collision_resolved.target_dynamic_object_id != first.target_dynamic_object_id


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
        assert len(config["target_type_policy"].digest) == 32


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
    validate_oracle_world(event.world, grid)
    context_id = next(iter(oracle.dynamic_object_future))
    assert context_id in event.world.dynamic_object_trajectories
    np.testing.assert_array_equal(
        event.world.dynamic_object_trajectories[context_id],
        oracle.dynamic_object_future[context_id],
    )
    assert event.world.dynamic_object_specs[context_id] == oracle.dynamic_object_specs[context_id]
    assert event.target.target_dynamic_object_id in event.world.dynamic_object_trajectories
    assert not bool(event.visibility_sequence[0])
    assert bool(event.visibility_sequence[-1])
    assert event.world.world_id == repeated.world.world_id
    assert event.world.metadata == repeated.world.metadata
    np.testing.assert_array_equal(event.visibility_sequence, repeated.visibility_sequence)
    for object_id in event.world.dynamic_object_trajectories:
        np.testing.assert_array_equal(
            event.world.dynamic_object_trajectories[object_id],
            repeated.world.dynamic_object_trajectories[object_id],
        )
    assert first.summary == second.summary
    assert first.summary["by_object_type"]["human"]["accepted"] == 1
    assert first.summary["by_footprint_kind"]["circle"]["accepted"] == 1
    assert first.summary["by_geometry_source"]["marker_extent_p95"]["accepted"] == 1


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
    assert report.summary["generator_algorithm_version"] == "joint_occluder_first_v2"
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
        "joint_occluder_first_v2"
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
