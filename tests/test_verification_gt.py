from pathlib import Path

import numpy as np
import pytest

from src.contracts import LocalTrajectory, OracleWorld
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
    TypedFootprintRiskLoss,
    VERIFICATION_GT_VERSION,
    evaluate_verification_value,
    load_verification_gt_config,
)
from src.planning.replanning import ReplannedCandidate, ReplanningResult
from src.planning.verification_actions import load_verification_actions
from tests.fixtures.verification_world import build_verification_toy_world


ROOT = Path(__file__).resolve().parents[1]
GT_CONFIG = ROOT / "configs/verification_gt.yaml"
ACTION_CONFIG = ROOT / "configs/verification_actions.yaml"


def _bank():
    toy = build_verification_toy_world()
    current = OracleWorld(
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
            "schema_version": "3.0.0",
            "split": "train",
            "source_namespace": "toy/train/gt-source",
        },
    )
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
                "action_id": "yaw_left_10",
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
        if trajectory.trajectory_id == "nominal":
            return 0.9 if dangerous else 0.1
        if trajectory.trajectory_id == "direct":
            return 1.0 if dangerous else 0.0
        if trajectory.trajectory_id == "avoid":
            return 0.25
        raise AssertionError("unexpected trajectory")


def _evaluate(observations, *, time_weight: float = 0.04, posterior_mode="exact"):
    toy, bank = _bank()
    nominal = _trajectory("nominal", 0.05, x_step=0.10, grid=toy.grid)
    replanning = _replanning(nominal, toy.grid)
    action = load_verification_actions(ACTION_CONFIG).by_id["yaw_left_10"]
    risk = _HandRisk()
    signatures = np.asarray(
        [[float(index)] * 7 for index in range(bank.size)], dtype=np.float32
    )
    normalizer = fit_signature_normalizer(signatures, split="train")
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
    assert result.br_before == pytest.approx(0.55)
    np.testing.assert_allclose(
        result.post_decision_risks,
        np.asarray(
            [
                0.35
                if item.variant_kind in {"current", "temporal", "speed"}
                else 0.02
                for item in bank.hypotheses
            ]
        ),
        atol=1e-12,
    )
    assert result.mean_post_decision_risk_before_action_cost == pytest.approx(0.185)
    assert result.action_cost == pytest.approx(0.035)
    assert result.post_risk == pytest.approx(0.220)
    assert result.value_target == pytest.approx(0.330)
    assert result.useful_target == 1
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
    parent_by_id = {
        item.trajectory.trajectory_id: item.poses_in_parent_frame
        for item in replanning.candidates
    }
    for trajectory_id, poses, _ in post_calls:
        np.testing.assert_array_equal(poses, parent_by_id[trajectory_id])
    assert all(
        item.trajectory.metadata["nominal_suffix_used"] is False
        for item in replanning.candidates
    )


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

    expected_increment = 0.10 * 0.50
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


def test_negative_or_nonfinite_risk_loss_is_rejected():
    toy, bank = _bank()
    shape = (toy.grid.height, toy.grid.width)
    observations = tuple(_observation(False, shape) for _ in bank.hypotheses)
    nominal = _trajectory("nominal", 0.0, x_step=0.1, grid=toy.grid)
    replanning = _replanning(nominal, toy.grid)
    action = load_verification_actions(ACTION_CONFIG).by_id["yaw_left_10"]

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
            )


def test_reject_fallback_is_available_after_every_observation():
    toy, bank = _bank()
    shape = (toy.grid.height, toy.grid.width)
    observations = tuple(_observation(False, shape) for _ in bank.hypotheses)
    nominal = _trajectory("nominal", 0.0, x_step=0.1, grid=toy.grid)
    replanning = _replanning(nominal, toy.grid)
    action = load_verification_actions(ACTION_CONFIG).by_id["yaw_left_10"]

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
    )

    assert result.br_before == pytest.approx(0.2)
    np.testing.assert_allclose(result.post_decision_risks, 0.2, atol=1e-12)
    assert result.best_decision_ids == ("reject",) * bank.size
    assert result.value_target == pytest.approx(-result.action_cost)
    assert result.useful_target == 0


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
