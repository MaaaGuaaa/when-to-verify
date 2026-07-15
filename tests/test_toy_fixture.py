"""Toy-fixture tests for SOP-00: hand-verifiable answers and determinism."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_FIX = Path(__file__).resolve().parent / "fixtures"
if str(_FIX) not in sys.path:
    sys.path.insert(0, str(_FIX))

import toy_world  # noqa: E402

from src.contracts import (  # noqa: E402
    ACTION_VECTOR_DIM,
    ROBOT_STATE_DIM,
    validate_risk_sample,
    validate_verification_sample,
)

TOL = 1e-6


# --- Structure -----------------------------------------------------------------
def test_toy_world_structure_counts():
    world = toy_world.build_toy_world()
    assert len(world["base_states"]) == 4
    for trajs in world["trajectories"].values():
        assert len(trajs) == 6
    assert len(world["oracle_worlds"]) == 4
    assert len(world["verification_actions"]) == 4


# --- Determinism ---------------------------------------------------------------
def test_same_seed_is_bit_identical():
    a = toy_world.build_toy_world(42)["seed_probe"]
    b = toy_world.build_toy_world(42)["seed_probe"]
    assert np.array_equal(a, b)


def test_different_seed_changes_stochastic_field():
    a = toy_world.build_toy_world(42)["seed_probe"]
    c = toy_world.build_toy_world(7)["seed_probe"]
    assert not np.array_equal(a, c)


def test_scene_geometry_is_seed_independent():
    a = toy_world.build_toy_world(42)["trajectories"]["toy_bs_0"][0].poses
    c = toy_world.build_toy_world(7)["trajectories"]["toy_bs_0"][0].poses
    assert np.array_equal(a, c)


# --- Risk answers match hand derivation ---------------------------------------
def test_collision_case_matches_hand_answer():
    world = toy_world.build_toy_world()
    got = world["risk_cases"]["collision"]
    exp = toy_world.toy_hand_answers()["risk"]["collision"]
    assert got["collision"] == exp["collision"]
    assert got["near_miss"] == exp["near_miss"]
    assert got["risk_severity"] == pytest.approx(exp["risk_severity"])
    assert got["min_clearance"] == pytest.approx(exp["min_clearance"], abs=1e-6)
    assert got["first_collision_time"] == pytest.approx(exp["first_collision_time"])


def test_near_miss_case_matches_hand_answer():
    world = toy_world.build_toy_world()
    got = world["risk_cases"]["near_miss"]
    exp = toy_world.toy_hand_answers()["risk"]["near_miss"]
    assert got["collision"] == exp["collision"]
    assert got["near_miss"] == exp["near_miss"]
    assert got["min_clearance"] == pytest.approx(exp["min_clearance"], abs=1e-6)
    assert 0.0 < got["risk_severity"] < 1.0


def test_temporal_safe_case_is_spatially_crossing_but_safe():
    world = toy_world.build_toy_world()
    got = world["risk_cases"]["temporal_safe"]
    exp = toy_world.toy_hand_answers()["risk"]["temporal_safe"]
    assert got["collision"] == exp["collision"]
    assert got["near_miss"] == exp["near_miss"]
    assert got["min_clearance"] > exp["min_clearance_gt"]


def test_empty_case_has_no_hidden_risk():
    world = toy_world.build_toy_world()
    got = world["risk_cases"]["empty"]
    assert got["collision"] == 0
    assert got["risk_severity"] == pytest.approx(0.0)
    assert got["first_collision_time"] is None


def test_clearance_monotonic_closer_is_not_less_risky():
    # Reference severity for a non-collision standing pedestrian must increase
    # as the pedestrian gets closer (holding time fixed via the min-distance k).
    robot_xy = toy_world.rollout(toy_world.NOMINAL_V, 0.0)[:, :2]

    def severity_for(offset_y: float) -> float:
        ped = np.tile(np.array([0.6, offset_y]), (robot_xy.shape[0], 1))
        return toy_world.risk_gt_reference(
            robot_xy, ped, sigma_d=0.5, sigma_t=2.0, near_miss_distance=0.35
        )["risk_severity"]

    far = severity_for(1.6)
    near = severity_for(0.9)
    assert near > far


# --- Verification G* answers match hand derivation ----------------------------
def test_verification_values_match_hand_answer():
    world = toy_world.build_toy_world()
    ver = world["verification_example"]
    exp = toy_world.toy_hand_answers()["verification"]
    assert ver["br_before"] == pytest.approx(exp["br_before"])
    peek = ver["actions"]["forward_peek"]
    yaw = ver["actions"]["yaw_left_10"]
    assert peek["value"] == pytest.approx(exp["forward_peek"]["value"], abs=1e-6)
    assert peek["useful"] == exp["forward_peek"]["useful"]
    assert yaw["value"] == pytest.approx(exp["yaw_left_10"]["value"], abs=1e-6)
    assert yaw["useful"] == exp["yaw_left_10"]["useful"]


def test_exactly_two_actions_are_useful():
    world = toy_world.build_toy_world()
    actions = world["verification_example"]["actions"]
    useful = sum(a["useful"] for a in actions.values())
    assert useful == toy_world.toy_hand_answers()["verification"]["useful_count"]


def test_higher_action_cost_lowers_value():
    # G* is monotonically non-increasing in verification cost (same information).
    base = toy_world.verification_value_reference(
        toy_world.TOY_WORLD_EXECUTE_COSTS, toy_world.TOY_REJECT_COST, 0.05, True
    )["value"]
    costlier = toy_world.verification_value_reference(
        toy_world.TOY_WORLD_EXECUTE_COSTS, toy_world.TOY_REJECT_COST, 0.15, True
    )["value"]
    assert costlier < base
    assert costlier == pytest.approx(base - 0.10, abs=1e-9)


# --- Batches for downstream model forward -------------------------------------
def test_risk_batch_shapes_and_finite():
    world = toy_world.build_toy_world()
    grid = world["grid"]
    batch = toy_world.make_risk_batch(grid, batch=2)
    assert batch["bev_history"].shape == (2, grid.history_steps, 2, 160, 160)
    assert batch["state_channels"].shape == (2, 9, 160, 160)
    assert batch["trajectory_channels"].shape == (2, 4, 160, 160)
    assert batch["robot_state"].shape == (2, ROBOT_STATE_DIM)
    for key, arr in batch.items():
        assert np.isfinite(arr).all(), key


def test_verification_batch_shapes_and_finite():
    world = toy_world.build_toy_world()
    grid = world["grid"]
    batch = toy_world.make_verification_batch(grid, batch=2)
    assert batch["verification_fov_mask"].shape == (2, 1, 160, 160)
    assert batch["verification_action_vector"].shape == (2, ACTION_VECTOR_DIM)
    for key, arr in batch.items():
        assert np.isfinite(arr).all(), key


def test_toy_samples_satisfy_contract():
    world = toy_world.build_toy_world()
    grid = world["grid"]
    validate_risk_sample(toy_world.make_risk_sample(grid), grid)
    validate_verification_sample(toy_world.make_verification_sample(grid), grid)
