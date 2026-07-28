from dataclasses import replace
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.generation.verification_pipeline as verification_pipeline
from src.contracts import BaseState, STATE_CHANNELS, validate_verification_sample
from src.generation.sop06_single import (
    Sop06SinglePublication,
    Sop06SingleRendererInput,
)
from src.generation.verification_gt import load_verification_gt_config
from src.generation.verification_pipeline import (
    VerificationSourceIneligibleError,
    build_verification_toy_input,
    generate_verification_group,
)
from src.generation.verification_response import (
    VERIFICATION_RESPONSE_POLICY_VERSION,
    VerificationResponseDecision,
    VerificationResponseResolution,
)
from src.planning.replanning import (
    POST_PLAN_STATUS_SAFE_STOP_NO_FEASIBLE_PLAN,
)
from src.planning.verification_actions import (
    load_verification_actions,
    sample_state_aware_action_trace,
)
from src.planning.verification_responses import build_reactive_braking_branch
from src.utils.config import load_config
from src.utils.seeding import stable_digest


ROOT = Path(__file__).resolve().parents[1]


def test_group_generation_has_no_downstream_scenario_bank_or_posterior_inputs():
    parameters = set(inspect.signature(generate_verification_group).parameters)
    assert {
        "scenario_config",
        "bank_size",
        "posterior_mode",
        "posterior_temperature",
        "signature_normalizer",
        "seed",
    }.isdisjoint(parameters)


def _finalized_publication_fixture():
    action_library = load_verification_actions(
        ROOT / "configs/verification_actions.yaml"
    )
    source, config = build_verification_toy_input(
        load_config(ROOT / "configs/base.yaml"),
        action_library=action_library,
        group_index=0,
    )
    target_id = source.target_object_id
    assert target_id is not None
    histories = {
        object_id: np.repeat(
            np.asarray(pose, dtype=np.float32)[None, :],
            source.grid.history_steps,
            axis=0,
        )
        for object_id, pose in source.current_dynamic_poses.items()
    }
    target_fill_forward_pose = np.asarray(
        [1.25, -0.75, 0.0],
        dtype=np.float32,
    )
    histories[target_id][:] = target_fill_forward_pose
    observed = {
        object_id: np.ones(source.grid.history_steps, dtype=np.bool_)
        for object_id in histories
    }
    observed[target_id][2:] = False
    base_state = BaseState(
        state_id=source.base_state_id,
        split=source.split,
        recording_id="finalized-verification-recording",
        dynamic_object_ids=(),
        timestamp=1.4,
        robot_history=np.repeat(
            source.robot_pose[None, :],
            source.grid.history_steps,
            axis=0,
        ),
        robot_state=np.array(source.robot_state, copy=True),
        visible_dynamic_object_history={},
        visible_dynamic_object_specs={},
        static_map_local=np.array(source.current_world.static_occupancy, copy=True),
        metadata={"source_mode": "sop05-final"},
    )
    renderer_input = Sop06SingleRendererInput(
        sample_id="finalized-verification-sample",
        mother_id="finalized-verification-mother",
        split=source.split,
        base_state=base_state,
        observed_static_occupancy=np.array(
            source.current_world.static_occupancy,
            copy=True,
        ),
        scene_dynamic_history=histories,
        scene_dynamic_specs={
            object_id: dict(spec)
            for object_id, spec in source.current_world.dynamic_object_specs.items()
        },
        scene_dynamic_history_observed=observed,
        sensor_config=None,
    )
    publication = Sop06SinglePublication(
        sample_id=renderer_input.sample_id,
        mother_id=renderer_input.mother_id,
        split=renderer_input.split,
        regime="seen_then_occluded",
        renderer_input=renderer_input,
        trajectory=source.nominal_trajectory,
        oracle_world=source.current_world,
        hidden_object_ids=(target_id,),
        provenance={"source_kind": "finalized-verification-fixture"},
    )
    return publication, config, action_library, target_fill_forward_pose


def _stub_finalized_render(monkeypatch, *, grid):
    bev_history = np.full(
        (
            grid.history_steps,
            grid.n_history_channels,
            grid.height,
            grid.width,
        ),
        np.float32(0.25),
        dtype=np.float32,
    )
    state_channels = np.zeros(
        (grid.n_state_channels, grid.height, grid.width),
        dtype=np.float32,
    )
    expected_visible = np.zeros((grid.height, grid.width), dtype=np.bool_)
    expected_visible[1, 2] = True
    expected_visible[3, 4] = True
    state_channels[
        STATE_CHANNELS.index("current_visible_free"),
        1,
        2,
    ] = 1.0
    state_channels[
        STATE_CHANNELS.index("current_visible_occupied"),
        3,
        4,
    ] = 1.0
    expected_age = np.full(
        (grid.height, grid.width),
        np.float32(0.625),
        dtype=np.float32,
    )
    state_channels[STATE_CHANNELS.index("occlusion_age_map")] = expected_age
    calls = []

    def render(base_state, **kwargs):
        calls.append((base_state, kwargs))
        return SimpleNamespace(
            bev_history=bev_history,
            state_channels=state_channels,
        )

    monkeypatch.setattr(verification_pipeline, "render_observation", render)
    return bev_history, state_channels, expected_visible, expected_age, calls


def test_finalized_publication_binds_exact_current_target_and_source_contract(
    monkeypatch,
):
    publication, config, action_library, fill_forward_pose = (
        _finalized_publication_fixture()
    )
    grid = publication.trajectory.swept_mask.shape
    rendered_bev, rendered_state, expected_visible, expected_age, calls = (
        _stub_finalized_render(
            monkeypatch,
            grid=verification_pipeline.build_grid_spec(config),
        )
    )
    target_current_pose = np.asarray(
        [3.75, -0.25, 0.4],
        dtype=np.float32,
    )
    target_id = publication.hidden_object_ids[0]
    assert not publication.renderer_input.scene_dynamic_history_observed[
        target_id
    ][-1]
    assert target_id in publication.oracle_world.dynamic_object_trajectories
    np.testing.assert_array_equal(
        publication.renderer_input.scene_dynamic_history[target_id][-1],
        fill_forward_pose,
    )
    source_digest = "source-publication-semantic-digest"
    release_identity = "final-release-identity"

    result = verification_pipeline.build_finalized_verification_input(
        publication,
        base_config=config,
        action_library=action_library,
        target_current_pose=target_current_pose,
        source_publication_semantic_digest=source_digest,
        final_release_identity=release_identity,
    )

    assert grid == (result.grid.height, result.grid.width)
    assert result.split == publication.split
    assert result.base_state_id == publication.renderer_input.base_state.state_id
    assert result.source_namespace == (
        f"sop05-final/{publication.split}/{publication.sample_id}"
    )
    assert result.nominal_trajectory is publication.trajectory
    assert result.current_world is publication.oracle_world
    assert result.target_object_id == target_id
    assert not np.array_equal(target_current_pose, fill_forward_pose)
    np.testing.assert_array_equal(
        result.current_dynamic_poses[result.target_object_id],
        target_current_pose,
    )
    np.testing.assert_array_equal(
        result.robot_pose,
        publication.renderer_input.base_state.robot_history[-1],
    )
    np.testing.assert_array_equal(
        result.robot_state,
        publication.renderer_input.base_state.robot_state,
    )
    np.testing.assert_array_equal(result.bev_history, rendered_bev)
    np.testing.assert_array_equal(result.state_channels, rendered_state)
    np.testing.assert_array_equal(result.current_visible_mask, expected_visible)
    np.testing.assert_array_equal(result.current_age_map, expected_age)
    assert len(calls) == 1
    rendered_base_state, render_kwargs = calls[0]
    assert rendered_base_state is publication.renderer_input.base_state
    assert render_kwargs["scene_dynamic_history"] is (
        publication.renderer_input.scene_dynamic_history
    )
    assert render_kwargs["scene_dynamic_specs"] is (
        publication.renderer_input.scene_dynamic_specs
    )
    assert render_kwargs["scene_dynamic_history_observed"] is (
        publication.renderer_input.scene_dynamic_history_observed
    )
    assert render_kwargs["static_occupancy"] is (
        publication.renderer_input.observed_static_occupancy
    )
    assert result.provenance == {
        "source_mode": "sop05-final",
        "source_sample_id": publication.sample_id,
        "source_mother_id": publication.mother_id,
        "blind_type": "seen_then_occluded",
        "target_object_type": "carried_object",
        "target_footprint_kind": "rectangle",
        "source_artifact_digest": stable_digest(
            source_digest,
            release_identity,
            publication.sample_id,
            size=16,
        ),
        "verification_sensor_fov_deg": 360.0,
        "verification_sensor_range_m": result.sensor_range_m,
    }
    parameters = set(
        inspect.signature(
            verification_pipeline.build_finalized_verification_input
        ).parameters
    )
    assert {
        "scenario_config",
        "bank_size",
        "posterior_mode",
        "posterior_temperature",
        "signature_normalizer",
        "seed",
    }.isdisjoint(parameters)
    assert {
        "scenario_bank_digest",
        "bank_size",
        "posterior_mode",
    }.isdisjoint(vars(result))


def test_finalized_target_absent_with_zero_hidden_ids_is_typed_ineligible(
    monkeypatch,
):
    publication, config, action_library, _ = _finalized_publication_fixture()
    target_id = publication.hidden_object_ids[0]
    absent_renderer = replace(
        publication.renderer_input,
        scene_dynamic_history={
            object_id: history
            for object_id, history in (
                publication.renderer_input.scene_dynamic_history.items()
            )
            if object_id != target_id
        },
        scene_dynamic_specs={
            object_id: spec
            for object_id, spec in publication.renderer_input.scene_dynamic_specs.items()
            if object_id != target_id
        },
        scene_dynamic_history_observed={
            object_id: observed
            for object_id, observed in (
                publication.renderer_input.scene_dynamic_history_observed.items()
            )
            if object_id != target_id
        },
    )
    absent_world = replace(
        publication.oracle_world,
        dynamic_object_trajectories={
            object_id: trajectory
            for object_id, trajectory in (
                publication.oracle_world.dynamic_object_trajectories.items()
            )
            if object_id != target_id
        },
        dynamic_object_specs={
            object_id: spec
            for object_id, spec in publication.oracle_world.dynamic_object_specs.items()
            if object_id != target_id
        },
    )
    absent_publication = replace(
        publication,
        regime="unseen_in_history_window",
        renderer_input=absent_renderer,
        oracle_world=absent_world,
        hidden_object_ids=(),
    )
    _stub_finalized_render(
        monkeypatch,
        grid=verification_pipeline.build_grid_spec(config),
    )

    with pytest.raises(VerificationSourceIneligibleError) as exc_info:
        verification_pipeline.build_finalized_verification_input(
            absent_publication,
            base_config=config,
            action_library=action_library,
            target_current_pose=None,
            source_publication_semantic_digest="source-digest",
            final_release_identity="release-identity",
        )

    assert exc_info.value.reason == "hidden_target_count"


def test_visible_target_is_ineligible_under_configured_verification_sensor():
    config = load_config(ROOT / "configs/base.yaml")
    action_library = load_verification_actions(
        ROOT / "configs/verification_actions.yaml"
    )
    source, _ = build_verification_toy_input(
        config,
        action_library=action_library,
        group_index=0,
    )
    visible_source = replace(
        source,
        current_visible_mask=np.ones_like(source.current_visible_mask),
    )

    with pytest.raises(VerificationSourceIneligibleError) as exc_info:
        verification_pipeline._hidden_object_ids(visible_source)

    assert exc_info.value.reason == "target_visible_under_verification_sensor"


def test_toy_group_runs_same_six_action_geometry_value_and_sample_path(
    monkeypatch,
):
    config = load_config(ROOT / "configs/base.yaml")
    action_library = load_verification_actions(
        ROOT / "configs/verification_actions.yaml"
    )
    source, toy_config = build_verification_toy_input(
        config,
        action_library=action_library,
        group_index=3,
    )
    gt_config = load_verification_gt_config(ROOT / "configs/verification_gt.yaml")
    original_replanning = verification_pipeline.generate_replanned_candidates
    original_response_resolution = (
        verification_pipeline.resolve_verification_response
    )
    replanning_starts = {}
    response_branch_kinds = []

    def record_replanning_start(**kwargs):
        replanning_starts.setdefault(kwargs["action_id"], []).append(
            np.array(kwargs["post_action_pose"], copy=True)
        )
        return original_replanning(**kwargs)

    monkeypatch.setattr(
        verification_pipeline,
        "generate_replanned_candidates",
        record_replanning_start,
    )

    def record_response_resolution(**kwargs):
        resolution = original_response_resolution(**kwargs)
        response_branch_kinds.append(resolution.decision.branch_kind)
        return resolution

    monkeypatch.setattr(
        verification_pipeline,
        "resolve_verification_response",
        record_response_resolution,
    )

    result = generate_verification_group(
        source,
        base_config=toy_config,
        action_library=action_library,
        gt_config=gt_config,
        max_replan_candidates=4,
    )

    assert len(result.samples) == 6
    assert result.sampled_child_world_id == source.current_world.world_id
    assert result.infeasible_action_ids == ()
    assert source.sensor_fov_rad == 2.0 * np.pi
    assert action_library.sensor_fov_rad == 2.0 * np.pi
    assert all(
        action_library.by_id[action_id].delta_forward_m > 0.0
        for action_id in (
            "arc_left_30",
            "arc_right_30",
            "arc_left_45",
            "arc_right_45",
        )
    )
    np.testing.assert_array_equal(
        source.robot_state, np.zeros(2, dtype=np.float32)
    )
    assert len({item.metadata["ranking_group_id"] for item in result.samples}) == 1
    assert all(np.isfinite(item.value_target) for item in result.samples)
    assert all(
        item.value_target == item.br_before - item.post_risk
        for item in result.samples
    )
    for sample in result.samples:
        validate_verification_sample(sample, source.grid)
        assert sample.metadata["provenance"]["source_mode"] == "toy"

    assert len(response_branch_kinds) == len(result.samples)
    assert set(response_branch_kinds) <= {
        "complete",
        "observe_and_replan",
        "emergency_brake",
    }
    response_id = result.values[
        "arc_left_45"
    ].policy_response_trajectory_id
    assert (
        response_id is None
        or response_id.startswith("policy::brake@")
    )
    assert all(
        value.sampled_child_world_id == source.current_world.world_id
        for value in result.values.values()
    )
    arc_action = action_library.by_id["arc_left_45"]
    full_trace = sample_state_aware_action_trace(
        source.robot_pose,
        arc_action,
        robot_state=source.robot_state,
        braking_deceleration_mps2=gt_config.braking_deceleration_mps2,
    )
    arc_starts = replanning_starts["arc_left_45"]
    assert any(
        np.allclose(start, full_trace.poses[-1], rtol=0.0, atol=1e-6)
        for start in arc_starts
    )


def _run_forced_emergency_no_plan(monkeypatch, *, collision_free):
    config = load_config(ROOT / "configs/base.yaml")
    action_library = load_verification_actions(
        ROOT / "configs/verification_actions.yaml"
    )
    source, toy_config = build_verification_toy_input(
        config,
        action_library=action_library,
        group_index=3,
    )
    gt_config = load_verification_gt_config(ROOT / "configs/verification_gt.yaml")
    original_replanning = verification_pipeline.generate_replanned_candidates

    def return_no_plan(**kwargs):
        return replace(
            original_replanning(**kwargs),
            candidates=(),
            plan_status=POST_PLAN_STATUS_SAFE_STOP_NO_FEASIBLE_PLAN,
        )

    def force_emergency_stop(**kwargs):
        resolved_collision_free = (
            collision_free() if callable(collision_free) else collision_free
        )
        action_trace = kwargs["action_trace"]
        trigger_time_s = float(action_trace.times_s[1])
        branch = build_reactive_braking_branch(
            action_trace=action_trace,
            response_time_s=trigger_time_s,
            braking_deceleration_mps2=kwargs["braking_deceleration_mps2"],
            angular_deceleration_radps2=kwargs[
                "angular_deceleration_radps2"
            ],
            future_horizon_s=kwargs["future_horizon_s"],
        )
        return VerificationResponseResolution(
            version=VERIFICATION_RESPONSE_POLICY_VERSION,
            decision=VerificationResponseDecision(
                branch_kind="emergency_brake",
                observation_time_s=trigger_time_s,
                predicted_ttc_s=0.0,
                stopping_time_s=0.0,
                braking_threshold_s=kwargs["braking_margin_s"],
                brake_trace_collision_free=resolved_collision_free,
                brake_trace_failure_reason=(
                    None if resolved_collision_free else "dynamic_collision"
                ),
            ),
            branch=branch,
        )

    monkeypatch.setattr(
        verification_pipeline,
        "generate_replanned_candidates",
        return_no_plan,
    )
    monkeypatch.setattr(
        verification_pipeline,
        "resolve_verification_response",
        force_emergency_stop,
    )

    return generate_verification_group(
        source,
        base_config=toy_config,
        action_library=action_library,
        gt_config=gt_config,
        max_replan_candidates=4,
    )


def test_safe_emergency_stop_without_plan_uses_reject_bypass(monkeypatch):
    gt_config = load_verification_gt_config(ROOT / "configs/verification_gt.yaml")
    result = _run_forced_emergency_no_plan(monkeypatch, collision_free=True)

    for value in result.values.values():
        assert (
            value.post_plan_status
            == POST_PLAN_STATUS_SAFE_STOP_NO_FEASIBLE_PLAN
        )
        assert value.best_decision_id == "reject"
        assert value.policy_response_trajectory_id is None
        assert value.realized_post_decision_risk_before_action_cost == (
            gt_config.reject_cost
        )
    for sample in result.samples:
        assert sample.metadata["label_audit"]["post_plan_status"] == (
            POST_PLAN_STATUS_SAFE_STOP_NO_FEASIBLE_PLAN
        )


def test_unsafe_emergency_stop_cannot_use_no_plan_bypass(monkeypatch):
    with pytest.raises(
        VerificationSourceIneligibleError,
        match="emergency brake",
    ) as exc_info:
        _run_forced_emergency_no_plan(monkeypatch, collision_free=False)

    assert exc_info.value.reason == "unsafe_emergency_brake"
