from dataclasses import replace
import inspect
from pathlib import Path

import numpy as np
import pytest

from src.contracts import LocalTrajectory, OracleWorld, SCHEMA_VERSION
from src.geometry import CircleFootprint
from src.generation.counterfactual_verify import (
    CounterfactualObservation,
    fit_signature_normalizer,
)
from src.generation.scenario_bank import (
    build_scenario_bank,
    load_scenario_bank_config,
)
from src.generation.verification_gt import (
    SAMPLED_REALIZATION_GT_VERSION,
    TypedFootprintRiskLoss,
    VERIFICATION_GT_VERSION,
    evaluate_sampled_realization_value,
    evaluate_verification_value,
    load_verification_gt_config,
    relative_task_regret,
)
from src.planning.replanning import (
    POST_PLAN_STATUS_SAFE_STOP_NO_FEASIBLE_PLAN,
    ReplannedCandidate,
    ReplanningResult,
)
from src.planning.verification_actions import ActionTrace, load_verification_actions
from src.planning.verification_responses import (
    build_completed_action_branch,
    build_reactive_braking_branch,
    compose_time_aligned_policy_trajectory,
)
from tests.fixtures.verification_world import build_verification_toy_world


ROOT = Path(__file__).resolve().parents[1]
GT_CONFIG = ROOT / "configs/verification_gt.yaml"
ACTION_CONFIG = ROOT / "configs/verification_actions.yaml"


def test_sampled_realization_value_api_has_no_bank_or_posterior():
    import src.generation.verification_gt as verification_gt

    evaluator = getattr(
        verification_gt,
        "evaluate_sampled_realization_value",
        None,
    )
    assert evaluator is not None
    parameters = set(inspect.signature(evaluator).parameters)
    assert {
        "bank",
        "observations",
        "signatures",
        "posterior_mode",
        "signature_normalizer",
        "posterior_temperature",
    }.isdisjoint(parameters)
    assert {
        "realized_world",
        "replanning_result",
        "time_aligned_policy_trajectories",
    }.issubset(parameters)


def test_relative_task_regret_is_scale_invariant_and_nonnegative():
    assert relative_task_regret(8.0, nominal_task_cost=8.0) == 0.0
    assert relative_task_regret(6.0, nominal_task_cost=8.0) == 0.0
    assert relative_task_regret(10.0, nominal_task_cost=8.0) == pytest.approx(0.25)
    assert relative_task_regret(
        1000.0, nominal_task_cost=800.0
    ) == pytest.approx(0.25)


def test_relative_task_regret_rejects_zero_nominal_cost():
    with pytest.raises(ValueError, match="nominal_task_cost must be positive"):
        relative_task_regret(1.0, nominal_task_cost=0.0)


def _sampled_world(toy) -> OracleWorld:
    return OracleWorld(
        world_id="gt-toy-current",
        base_state_id="gt-toy-base",
        static_occupancy=toy.static_occupancy.copy(),
        dynamic_object_trajectories={
            key: value.copy() for key, value in toy.dynamic_future_poses.items()
        },
        dynamic_object_specs={key: dict(value) for key, value in toy.dynamic_specs.items()},
        occluders=(),
        blind_spot_config={"kind": "structural", "occluder_ids": []},
        random_seed=7,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "split": "train",
            "source_namespace": "toy/train/gt-source",
        },
    )


def _bank():
    toy = build_verification_toy_world()
    current = _sampled_world(toy)
    bank = build_scenario_bank(
        current_world=current,
        target_object_id="critical_cart",
        current_dynamic_poses=toy.dynamic_current_poses,
        current_visible_mask=toy.current_visible_mask,
        grid=toy.grid,
        split="train",
        source_namespace="toy/train/gt-source",
        seed=7,
        size=8,
        config=load_scenario_bank_config(GT_CONFIG),
    )
    return toy, bank


def _trajectory(
    trajectory_id: str, task_cost: float, *, x_step: float, grid
) -> LocalTrajectory:
    times = np.arange(1, grid.future_steps + 1, dtype=np.float32)
    poses = np.column_stack(
        (
            times * np.float32(x_step),
            np.zeros(grid.future_steps, dtype=np.float32),
            np.zeros(grid.future_steps, dtype=np.float32),
        )
    ).astype(np.float32)
    controls = np.tile(
        np.asarray([x_step / 0.2, 0.0], dtype=np.float32),
        (grid.future_steps, 1),
    )
    zeros = np.zeros((grid.height, grid.width), dtype=np.float32)
    return LocalTrajectory(
        trajectory_id=trajectory_id,
        poses=poses,
        controls=controls,
        swept_mask=zeros.copy(),
        tta_map=np.full_like(zeros, -1.0),
        braking_map=zeros.copy(),
        centerline_map=zeros.copy(),
        task_cost=task_cost,
        metadata={
            "pose_time_layout_version": "future_endpoints_dt_to_horizon_v1",
            "nominal_suffix_used": False,
        },
    )


def _replanning(nominal: LocalTrajectory, grid) -> ReplanningResult:
    candidates = []
    for trajectory_id, task_cost, x_step in (
        ("direct", 0.02, 0.12),
        ("avoid", 0.10, 0.04),
    ):
        trajectory = _trajectory(trajectory_id, task_cost, x_step=x_step, grid=grid)
        trajectory.metadata.update(
            {
                "nominal_trajectory_id": nominal.trajectory_id,
                "action_id": "arc_left_30",
                "sampling_origin": "post_action_pose",
                "nominal_suffix_used": False,
                "is_stop": False,
            }
        )
        candidates.append(
            ReplannedCandidate(
                trajectory=trajectory,
                implicit_start_pose=np.zeros(3, dtype=np.float32),
                poses_in_parent_frame=trajectory.poses.copy(),
                swept_mask_in_parent_frame=np.zeros(
                    (grid.height, grid.width), dtype=np.float32
                ),
                intent_error=task_cost,
            )
        )
    return ReplanningResult(
        version="post_action_anchored_sampler_v1",
        post_action_pose=np.zeros(3, dtype=np.float32),
        task_anchor_pose=nominal.poses[-1].copy(),
        candidates=tuple(candidates),
        reject_available=True,
        rejection_counts={},
    )


def _completed_policy_group(
    nominal: LocalTrajectory,
    replanning: ReplanningResult,
    action,
) -> tuple[LocalTrajectory, ...]:
    branch = build_completed_action_branch(
        ActionTrace(
            poses=np.stack(
                (
                    np.zeros(3, dtype=np.float32),
                    replanning.post_action_pose,
                )
            ).astype(np.float32),
            times_s=np.asarray([0.0, action.duration_s], dtype=np.float64),
            linear_velocities_mps=np.zeros(2, dtype=np.float64),
            angular_velocities_radps=np.zeros(2, dtype=np.float64),
        )
    )
    return tuple(
        compose_time_aligned_policy_trajectory(
            template_trajectory=candidate.trajectory,
            branch=branch,
            future_dt_s=0.2,
            trajectory_id=(
                f"policy::complete::{candidate.trajectory.trajectory_id}"
            ),
            source_action_id=action.action_id,
            source_nominal_trajectory_id=nominal.trajectory_id,
            suffix_trajectory=candidate.trajectory,
            suffix_poses_in_parent_frame=candidate.poses_in_parent_frame,
        )
        for candidate in replanning.candidates
    )


def test_safe_stop_without_future_plan_bypasses_policy_risk_evaluation():
    toy = build_verification_toy_world()
    world = _sampled_world(toy)
    nominal = _trajectory("nominal", 0.05, x_step=0.10, grid=toy.grid)
    action = load_verification_actions(ACTION_CONFIG).by_id["arc_left_30"]
    no_plan = replace(
        _replanning(nominal, toy.grid),
        candidates=(),
        plan_status=POST_PLAN_STATUS_SAFE_STOP_NO_FEASIBLE_PLAN,
    )
    calls = []

    def realized_risk(trajectory, poses_in_parent_frame, realized_world):
        calls.append(
            (
                trajectory.trajectory_id,
                np.array(poses_in_parent_frame, copy=True),
                realized_world.world_id,
            )
        )
        return 0.40

    result = evaluate_sampled_realization_value(
        realized_world=world,
        nominal_trajectory=nominal,
        action=action,
        replanning_result=no_plan,
        risk_loss=realized_risk,
        reject_cost=0.60,
        risk_weight=1.0,
        action_cost_config={
            "lambda_time": 0.04,
            "lambda_distance": 0.05,
            "lambda_yaw_per_deg": 0.0015,
        },
        time_aligned_policy_trajectories=(),
    )

    assert result.version == SAMPLED_REALIZATION_GT_VERSION
    assert result.sampled_child_world_id == world.world_id
    assert result.post_plan_status == POST_PLAN_STATUS_SAFE_STOP_NO_FEASIBLE_PLAN
    assert result.best_decision_id == "reject"
    assert result.unclipped_best_policy_loss is None
    assert result.policy_response_trajectory_id is None
    assert result.realized_post_decision_risk_before_action_cost == pytest.approx(
        0.60
    )
    assert [call[0] for call in calls] == [nominal.trajectory_id]


def _observation(occupied: bool, shape: tuple[int, int]):
    visible = np.zeros(shape, dtype=bool)
    visible[0, 0] = True
    occupancy = np.zeros(shape, dtype=bool)
    if occupied:
        occupancy[0, 0] = True
    return CounterfactualObservation(
        visible_mask=visible,
        visible_occupied_mask=occupancy,
        visible_dynamic_occupancy=occupancy.copy(),
        newly_visible_mask=visible.copy(),
        updated_age_map=np.zeros(shape, dtype=np.float32),
    )


class _HandRisk:
    def __init__(self):
        self.calls = []

    def __call__(self, trajectory, poses_in_parent_frame, hypothesis):
        self.calls.append(
            (
                trajectory.trajectory_id,
                np.array(poses_in_parent_frame, copy=True),
                hypothesis.variant_kind,
            )
        )
        dangerous = hypothesis.variant_kind in {"current", "temporal", "speed"}
        decision_id = trajectory.metadata.get(
            "suffix_trajectory_id", trajectory.trajectory_id
        )
        if decision_id == "nominal":
            return 0.9 if dangerous else 0.1
        if decision_id == "direct":
            return 1.0 if dangerous else 0.0
        if decision_id == "avoid":
            return 0.25
        if trajectory.trajectory_id.startswith("policy::brake@"):
            return 0.0
        raise AssertionError("unexpected trajectory")


def _evaluate(observations, *, time_weight: float = 0.04, posterior_mode="exact"):
    toy, bank = _bank()
    nominal = _trajectory("nominal", 0.05, x_step=0.10, grid=toy.grid)
    replanning = _replanning(nominal, toy.grid)
    action = load_verification_actions(ACTION_CONFIG).by_id["arc_left_30"]
    risk = _HandRisk()
    signatures = np.asarray(
        [[float(index)] * 7 for index in range(bank.size)], dtype=np.float32
    )
    normalizer = fit_signature_normalizer(signatures, split="train")
    policies = _completed_policy_group(nominal, replanning, action)
    result = evaluate_verification_value(
        bank=bank,
        nominal_trajectory=nominal,
        action=action,
        observations=observations,
        signatures=signatures,
        replanning_results=(replanning,) * bank.size,
        risk_loss=risk,
        posterior_mode=posterior_mode,
        signature_normalizer=normalizer if posterior_mode == "soft" else None,
        posterior_temperature=0.2 if posterior_mode == "soft" else None,
        reject_cost=0.60,
        risk_weight=1.0,
        action_cost_config={
            "lambda_time": time_weight,
            "lambda_distance": 0.05,
            "lambda_yaw_per_deg": 0.0015,
        },
        time_aligned_policy_trajectories=(policies,) * bank.size,
    )
    return toy, bank, nominal, replanning, result, risk


def test_exact_g_star_matches_hand_enumerated_mixed_footprint_bank():
    toy, bank = _bank()
    shape = (toy.grid.height, toy.grid.width)
    observations = tuple(
        _observation(
            hypothesis.variant_kind in {"current", "temporal", "speed"}, shape
        )
        for hypothesis in bank.hypotheses
    )
    _, bank, nominal, replanning, result, risk = _evaluate(observations)

    assert result.version == VERIFICATION_GT_VERSION
    assert result.mean_execute_loss == pytest.approx(0.50)
    assert result.br_before == pytest.approx(0.50)
    np.testing.assert_allclose(
        result.post_decision_risks,
        np.asarray(
            [
                0.60
                if item.variant_kind in {"current", "temporal", "speed"}
                else 0.0
                for item in bank.hypotheses
            ]
        ),
        atol=1e-12,
    )
    assert result.mean_post_decision_risk_before_action_cost == pytest.approx(0.30)
    assert result.action_cost == pytest.approx(0.0995)
    assert result.post_risk == pytest.approx(0.3995)
    assert result.value_target == pytest.approx(0.1005)
    assert result.useful_target == 1
    assert result.unclipped_best_policy_losses == tuple(
        1.0 if item.variant_kind in {"current", "temporal", "speed"} else 0.0
        for item in bank.hypotheses
    )
    assert result.posterior_mode == "exact"
    assert bank.hypotheses[0].world.dynamic_object_specs["critical_cart"][
        "footprint"
    ]["kind"] == "rectangle"
    assert bank.hypotheses[0].world.dynamic_object_specs["irrelevant_person"][
        "footprint"
    ]["kind"] == "circle"

    post_calls = [call for call in risk.calls if call[0] != nominal.trajectory_id]
    assert post_calls
    assert len(post_calls) == bank.size * len(replanning.candidates)
    action = load_verification_actions(ACTION_CONFIG).by_id["arc_left_30"]
    aligned_by_id = {
        item.trajectory_id: item.poses
        for item in _completed_policy_group(nominal, replanning, action)
    }
    for trajectory_id, poses, _ in post_calls:
        np.testing.assert_array_equal(poses, aligned_by_id[trajectory_id])
    assert all(
        trajectory_id.startswith("policy::complete::")
        for trajectory_id, _, _ in post_calls
    )
    assert all(
        item.trajectory.metadata["nominal_suffix_used"] is False
        for item in replanning.candidates
    )


def test_uninformative_observation_stays_negative_after_relative_task_cost():
    toy, bank = _bank()
    shape = (toy.grid.height, toy.grid.width)
    observations = tuple(_observation(False, shape) for _ in bank.hypotheses)
    _, _, _, _, result, _ = _evaluate(observations)

    assert result.br_before == pytest.approx(0.50)
    np.testing.assert_allclose(result.post_decision_risks, 0.50, atol=1e-12)
    assert result.value_target == pytest.approx(-result.action_cost)
    assert result.useful_target == 0


def test_action_cost_is_added_once_and_critical_observation_beats_irrelevant():
    toy, bank = _bank()
    shape = (toy.grid.height, toy.grid.width)
    critical = tuple(
        _observation(
            item.variant_kind in {"current", "temporal", "speed"}, shape
        )
        for item in bank.hypotheses
    )
    irrelevant = tuple(_observation(False, shape) for _ in bank.hypotheses)
    *_, low_cost, _ = _evaluate(critical, time_weight=0.04)
    *_, high_cost, _ = _evaluate(critical, time_weight=0.14)
    *_, irrelevant_result, _ = _evaluate(irrelevant, time_weight=0.04)

    expected_increment = 0.10 * 0.80
    assert high_cost.post_risk - low_cost.post_risk == pytest.approx(
        expected_increment
    )
    assert low_cost.value_target - high_cost.value_target == pytest.approx(
        expected_increment
    )
    assert low_cost.value_target > irrelevant_result.value_target
    assert low_cost.mean_post_decision_risk_before_action_cost < (
        irrelevant_result.mean_post_decision_risk_before_action_cost
    )


def test_soft_mode_and_gt_config_are_finite():
    toy, bank = _bank()
    observations = tuple(
        _observation(index % 2 == 0, (toy.grid.height, toy.grid.width))
        for index in range(bank.size)
    )
    *_, result, _ = _evaluate(observations, posterior_mode="soft")
    assert result.posterior_mode == "soft"
    assert np.isfinite(result.posterior).all()
    np.testing.assert_allclose(result.posterior.sum(axis=1), 1.0, atol=1e-12)

    config = load_verification_gt_config(GT_CONFIG)
    assert config.reject_cost == 0.20
    assert config.risk_weight == 1.0
    assert config.braking_deceleration_mps2 == 1.0
    assert config.angular_deceleration_radps2 == 1.6
    assert config.braking_margin_s == 0.4


def test_negative_or_nonfinite_risk_loss_is_rejected():
    toy, bank = _bank()
    shape = (toy.grid.height, toy.grid.width)
    observations = tuple(_observation(False, shape) for _ in bank.hypotheses)
    nominal = _trajectory("nominal", 0.05, x_step=0.1, grid=toy.grid)
    replanning = _replanning(nominal, toy.grid)
    action = load_verification_actions(ACTION_CONFIG).by_id["arc_left_30"]
    policies = _completed_policy_group(nominal, replanning, action)

    for invalid in (-0.1, float("nan")):
        with pytest.raises(ValueError, match="risk loss"):
            evaluate_verification_value(
                bank=bank,
                nominal_trajectory=nominal,
                action=action,
                observations=observations,
                signatures=None,
                replanning_results=(replanning,) * bank.size,
                risk_loss=lambda *_: invalid,
                posterior_mode="exact",
                signature_normalizer=None,
                posterior_temperature=None,
                reject_cost=0.2,
                risk_weight=1.0,
                action_cost_config={
                    "lambda_time": 0.04,
                    "lambda_distance": 0.05,
                    "lambda_yaw_per_deg": 0.0015,
                },
                time_aligned_policy_trajectories=(policies,) * bank.size,
            )


def test_evaluator_rejects_zero_nominal_task_cost():
    toy, bank = _bank()
    shape = (toy.grid.height, toy.grid.width)
    observations = tuple(_observation(False, shape) for _ in bank.hypotheses)
    nominal = _trajectory("nominal", 0.0, x_step=0.1, grid=toy.grid)
    replanning = _replanning(nominal, toy.grid)
    action = load_verification_actions(ACTION_CONFIG).by_id["arc_left_30"]
    policies = _completed_policy_group(nominal, replanning, action)

    with pytest.raises(
        ValueError, match="nominal trajectory task_cost must be positive"
    ):
        evaluate_verification_value(
            bank=bank,
            nominal_trajectory=nominal,
            action=action,
            observations=observations,
            signatures=None,
            replanning_results=(replanning,) * bank.size,
            risk_loss=lambda *_: 0.0,
            posterior_mode="exact",
            signature_normalizer=None,
            posterior_temperature=None,
            reject_cost=0.2,
            risk_weight=1.0,
            action_cost_config={
                "lambda_time": 0.04,
                "lambda_distance": 0.05,
                "lambda_yaw_per_deg": 0.0015,
            },
            time_aligned_policy_trajectories=(policies,) * bank.size,
        )


def test_long40_gt_requires_absolute_time_aligned_policy_groups():
    toy, bank = _bank()
    shape = (toy.grid.height, toy.grid.width)
    observations = tuple(_observation(False, shape) for _ in bank.hypotheses)
    nominal = _trajectory("nominal", 0.05, x_step=0.1, grid=toy.grid)
    replanning = _replanning(nominal, toy.grid)
    action = load_verification_actions(ACTION_CONFIG).by_id["arc_left_30"]

    with pytest.raises(ValueError, match="time-aligned policy"):
        evaluate_verification_value(
            bank=bank,
            nominal_trajectory=nominal,
            action=action,
            observations=observations,
            signatures=None,
            replanning_results=(replanning,) * bank.size,
            risk_loss=lambda *_: 0.0,
            posterior_mode="exact",
            signature_normalizer=None,
            posterior_temperature=None,
            reject_cost=0.2,
            risk_weight=1.0,
            action_cost_config={
                "lambda_time": 0.04,
                "lambda_distance": 0.05,
                "lambda_yaw_per_deg": 0.0015,
            },
        )


def test_reject_fallback_is_available_after_every_observation():
    toy, bank = _bank()
    shape = (toy.grid.height, toy.grid.width)
    observations = tuple(_observation(False, shape) for _ in bank.hypotheses)
    nominal = _trajectory("nominal", 0.05, x_step=0.1, grid=toy.grid)
    replanning = _replanning(nominal, toy.grid)
    action = load_verification_actions(ACTION_CONFIG).by_id["arc_left_30"]
    policies = _completed_policy_group(nominal, replanning, action)

    result = evaluate_verification_value(
        bank=bank,
        nominal_trajectory=nominal,
        action=action,
        observations=observations,
        signatures=None,
        replanning_results=(replanning,) * bank.size,
        risk_loss=lambda *_: 10.0,
        posterior_mode="exact",
        signature_normalizer=None,
        posterior_temperature=None,
        reject_cost=0.2,
        risk_weight=1.0,
        action_cost_config={
            "lambda_time": 0.04,
            "lambda_distance": 0.05,
            "lambda_yaw_per_deg": 0.0015,
        },
        time_aligned_policy_trajectories=(policies,) * bank.size,
    )

    assert result.br_before == pytest.approx(0.2)
    np.testing.assert_allclose(result.post_decision_risks, 0.2, atol=1e-12)
    assert result.best_decision_ids == ("reject",) * bank.size
    assert result.value_target == pytest.approx(-result.action_cost)
    assert result.useful_target == 0


def test_empty_zero_risk_bank_has_nonpositive_verification_value():
    toy, bank = _bank()
    shape = (toy.grid.height, toy.grid.width)
    observations = tuple(_observation(False, shape) for _ in bank.hypotheses)
    nominal = _trajectory("nominal", 0.05, x_step=0.1, grid=toy.grid)
    replanning = _replanning(nominal, toy.grid)
    action = load_verification_actions(ACTION_CONFIG).by_id["arc_left_30"]
    policies = _completed_policy_group(nominal, replanning, action)

    result = evaluate_verification_value(
        bank=bank,
        nominal_trajectory=nominal,
        action=action,
        observations=observations,
        signatures=None,
        replanning_results=(replanning,) * bank.size,
        risk_loss=lambda *_: 0.0,
        posterior_mode="exact",
        signature_normalizer=None,
        posterior_temperature=None,
        reject_cost=0.2,
        risk_weight=1.0,
        action_cost_config={
            "lambda_time": 0.04,
            "lambda_distance": 0.05,
            "lambda_yaw_per_deg": 0.0015,
        },
        time_aligned_policy_trajectories=(policies,) * bank.size,
    )

    assert result.br_before == 0.0
    assert result.value_target < 0.0
    assert result.useful_target == 0


def test_label_side_braking_response_enters_post_verification_risk_candidates():
    toy, bank = _bank()
    nominal = _trajectory("nominal", 0.05, x_step=0.1, grid=toy.grid)
    replanning = _replanning(nominal, toy.grid)
    action = load_verification_actions(ACTION_CONFIG).by_id["arc_left_30"]
    observations = tuple(
        _observation(False, (toy.grid.height, toy.grid.width))
        for _ in bank.hypotheses
    )
    branch = build_reactive_braking_branch(
        action_trace=ActionTrace(
            poses=np.zeros((3, 3), dtype=np.float32),
            times_s=np.asarray([0.0, 0.2, 0.8], dtype=np.float64),
            linear_velocities_mps=np.zeros(3, dtype=np.float64),
            angular_velocities_radps=np.zeros(3, dtype=np.float64),
        ),
        response_time_s=0.2,
        braking_deceleration_mps2=1.0,
        future_horizon_s=6.4,
    )
    emergency_policies = tuple(
        compose_time_aligned_policy_trajectory(
            template_trajectory=candidate.trajectory,
            branch=branch,
            future_dt_s=0.2,
            trajectory_id=(
                f"policy::emergency::{candidate.trajectory.trajectory_id}"
            ),
            source_action_id=action.action_id,
            source_nominal_trajectory_id=nominal.trajectory_id,
            suffix_trajectory=candidate.trajectory,
            suffix_poses_in_parent_frame=candidate.poses_in_parent_frame,
        )
        for candidate in replanning.candidates
    )
    response = compose_time_aligned_policy_trajectory(
        template_trajectory=nominal,
        branch=branch,
        future_dt_s=0.2,
        trajectory_id="policy::brake@0.200000::arc_left_30",
        source_action_id=action.action_id,
        source_nominal_trajectory_id=nominal.trajectory_id,
    )
    emergency_policies = (*emergency_policies, response)
    completed_policies = _completed_policy_group(nominal, replanning, action)
    policy_groups = (
        emergency_policies,
        *((completed_policies,) * (bank.size - 1)),
    )
    risk = _HandRisk()

    result = evaluate_verification_value(
        bank=bank,
        nominal_trajectory=nominal,
        action=action,
        observations=observations,
        signatures=None,
        replanning_results=(replanning,) * bank.size,
        risk_loss=risk,
        posterior_mode="exact",
        signature_normalizer=None,
        posterior_temperature=None,
        reject_cost=0.60,
        risk_weight=1.0,
        action_cost_config={
            "lambda_time": 0.04,
            "lambda_distance": 0.05,
            "lambda_yaw_per_deg": 0.0015,
        },
        time_aligned_policy_trajectories=policy_groups,
    )

    assert result.policy_response_trajectory_ids == (
        response.trajectory_id,
        *((None,) * (bank.size - 1)),
    )
    assert result.best_decision_ids[0] == response.trajectory_id
    response_calls = [
        call for call in risk.calls if call[0] == response.trajectory_id
    ]
    assert len(response_calls) == bank.size
    for _, poses, _ in response_calls:
        np.testing.assert_array_equal(poses, response.poses)
    assert response.metadata["label_side_policy_trajectory"] is True


def test_safe_stop_without_feasible_plan_bypasses_post_trajectory_risk():
    toy, bank = _bank()
    nominal = _trajectory("nominal", 0.05, x_step=0.1, grid=toy.grid)
    no_plan = replace(
        _replanning(nominal, toy.grid),
        candidates=(),
        plan_status="safe_stop_no_feasible_plan",
    )
    action = load_verification_actions(ACTION_CONFIG).by_id["arc_left_30"]
    observations = tuple(
        _observation(False, (toy.grid.height, toy.grid.width))
        for _ in bank.hypotheses
    )
    risk_calls = []

    def risk(trajectory, poses_in_parent_frame, hypothesis):
        risk_calls.append(trajectory.trajectory_id)
        assert trajectory.trajectory_id == nominal.trajectory_id
        return 0.9

    result = evaluate_verification_value(
        bank=bank,
        nominal_trajectory=nominal,
        action=action,
        observations=observations,
        signatures=None,
        replanning_results=(no_plan,) * bank.size,
        risk_loss=risk,
        posterior_mode="exact",
        signature_normalizer=None,
        posterior_temperature=None,
        reject_cost=0.60,
        risk_weight=1.0,
        action_cost_config={
            "lambda_time": 0.04,
            "lambda_distance": 0.05,
            "lambda_yaw_per_deg": 0.0015,
        },
        time_aligned_policy_trajectories=((),) * bank.size,
    )

    assert risk_calls == [nominal.trajectory_id] * bank.size
    np.testing.assert_array_equal(
        result.post_decision_risks,
        np.full(bank.size, 0.60, dtype=np.float64),
    )
    assert result.best_decision_ids == ("reject",) * bank.size
    assert result.post_plan_statuses == (
        "safe_stop_no_feasible_plan",
    ) * bank.size
    assert result.policy_response_trajectory_ids == (None,) * bank.size
    assert result.post_risk == pytest.approx(0.60 + result.action_cost)


def test_verification_result_requires_a_plan_status_for_every_world():
    toy, bank = _bank()
    observations = tuple(
        _observation(False, (toy.grid.height, toy.grid.width))
        for _ in bank.hypotheses
    )
    *_, result, _ = _evaluate(observations)

    with pytest.raises(ValueError, match="align with the scenario bank"):
        replace(result, post_plan_statuses=())


def test_time_aligned_policy_candidates_replace_local_replans_in_gt():
    toy, bank = _bank()
    nominal = _trajectory("nominal", 0.05, x_step=0.1, grid=toy.grid)
    action = load_verification_actions(ACTION_CONFIG).by_id["arc_left_30"]
    local_replanning = _replanning(nominal, toy.grid)
    action_trace = ActionTrace(
        poses=np.asarray([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float32),
        times_s=np.asarray([0.0, action.duration_s], dtype=np.float64),
        linear_velocities_mps=np.zeros(2, dtype=np.float64),
        angular_velocities_radps=np.zeros(2, dtype=np.float64),
    )
    branch = build_completed_action_branch(action_trace)
    shifted_candidates = tuple(
        replace(
            candidate,
            implicit_start_pose=branch.end_pose,
            poses_in_parent_frame=(
                candidate.poses_in_parent_frame
                + np.asarray([0.05, 0.0, 0.0], dtype=np.float32)
            ).astype(np.float32),
        )
        for candidate in local_replanning.candidates
    )
    replanning = replace(
        local_replanning,
        post_action_pose=branch.end_pose,
        candidates=shifted_candidates,
    )
    aligned = tuple(
        compose_time_aligned_policy_trajectory(
            template_trajectory=candidate.trajectory,
            branch=branch,
            future_dt_s=0.2,
            trajectory_id=f"policy::complete::{candidate.trajectory.trajectory_id}",
            source_action_id=action.action_id,
            source_nominal_trajectory_id=nominal.trajectory_id,
            suffix_trajectory=candidate.trajectory,
            suffix_poses_in_parent_frame=candidate.poses_in_parent_frame,
        )
        for candidate in replanning.candidates
    )
    observations = tuple(
        _observation(False, (toy.grid.height, toy.grid.width))
        for _ in bank.hypotheses
    )
    calls = []

    def risk(trajectory, poses_in_parent_frame, hypothesis):
        calls.append(
            (
                trajectory.trajectory_id,
                np.array(poses_in_parent_frame, copy=True),
                hypothesis.hypothesis_id,
            )
        )
        return 0.9 if trajectory.trajectory_id == nominal.trajectory_id else 0.0

    result = evaluate_verification_value(
        bank=bank,
        nominal_trajectory=nominal,
        action=action,
        observations=observations,
        signatures=None,
        replanning_results=(replanning,) * bank.size,
        risk_loss=risk,
        posterior_mode="exact",
        signature_normalizer=None,
        posterior_temperature=None,
        reject_cost=0.60,
        risk_weight=1.0,
        action_cost_config={
            "lambda_time": 0.04,
            "lambda_distance": 0.05,
            "lambda_yaw_per_deg": 0.0015,
        },
        time_aligned_policy_trajectories=(aligned,) * bank.size,
    )

    aligned_by_id = {item.trajectory_id: item.poses for item in aligned}
    policy_calls = [call for call in calls if call[0] != nominal.trajectory_id]
    assert len(policy_calls) == len(aligned) * bank.size
    assert {trajectory_id for trajectory_id, _, _ in policy_calls} == set(
        aligned_by_id
    )
    for trajectory_id, poses, _ in policy_calls:
        np.testing.assert_array_equal(poses, aligned_by_id[trajectory_id])
    assert all(
        not trajectory_id.startswith(("direct", "avoid"))
        for trajectory_id, _, _ in policy_calls
    )
    assert result.policy_response_trajectory_ids == (None,) * bank.size
    assert all(
        decision_id.startswith("policy::complete::")
        for decision_id in result.best_decision_ids
    )

    for invalid_end_time_s, message in (
        (action.duration_s - 0.1, "duration"),
        (6.5, "horizon"),
    ):
        invalid = tuple(
            replace(
                item,
                metadata={
                    **item.metadata,
                    "branch_end_time_s": invalid_end_time_s,
                },
            )
            for item in aligned
        )
        with pytest.raises(ValueError, match=message):
            evaluate_verification_value(
                bank=bank,
                nominal_trajectory=nominal,
                action=action,
                observations=observations,
                signatures=None,
                replanning_results=(replanning,) * bank.size,
                risk_loss=risk,
                posterior_mode="exact",
                signature_normalizer=None,
                posterior_temperature=None,
                reject_cost=0.60,
                risk_weight=1.0,
                action_cost_config={
                    "lambda_time": 0.04,
                    "lambda_distance": 0.05,
                    "lambda_yaw_per_deg": 0.0015,
                },
                time_aligned_policy_trajectories=(invalid,) * bank.size,
            )


def test_typed_risk_adapter_reuses_circle_and_rectangle_geometry():
    toy, bank = _bank()
    nominal = _trajectory("nominal", 0.05, x_step=0.1, grid=toy.grid)
    adapter = TypedFootprintRiskLoss(
        hidden_object_ids=("critical_cart", "irrelevant_person"),
        robot_footprint=CircleFootprint(0.25),
        grid=toy.grid,
        future_dt_s=0.2,
        sigma_distance_m=0.5,
        sigma_time_s=2.0,
        near_miss_distance_m=0.35,
    )
    current = next(item for item in bank.hypotheses if item.variant_kind == "current")
    empty = next(item for item in bank.hypotheses if item.variant_kind == "empty")

    current_loss = adapter(nominal, nominal.poses, current)
    empty_loss = adapter(nominal, nominal.poses, empty)

    assert np.isfinite(current_loss)
    assert np.isfinite(empty_loss)
    assert current_loss >= empty_loss
    assert current.world.dynamic_object_specs["critical_cart"]["footprint"][
        "kind"
    ] == "rectangle"
    assert empty.world.dynamic_object_specs["irrelevant_person"]["footprint"][
        "kind"
    ] == "circle"
