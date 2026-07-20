from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from src.contracts import STATE_CHANNELS, TRAJECTORY_CHANNELS
from src.evaluation.verification_baselines import (
    critical_swept_coverage_score,
    occupancy_entropy_reduction_score,
    visible_area_score,
)
from src.evaluation.verification_metrics import (
    VERIFICATION_CHECKPOINT_MANIFEST_VERSION,
    build_verification_checkpoint_manifest,
    evaluate_verification_predictions,
    kendall_tau_b,
    pairwise_ranking_accuracy,
    spearman_correlation,
    validate_verification_checkpoint_manifest,
)
from src.models.verification_model import load_verify_model_config
from src.planning.verification_actions import CANONICAL_ACTION_IDS


ROOT = Path(__file__).resolve().parents[1]


def _baseline_inputs():
    state = np.zeros((len(STATE_CHANNELS), 2, 2), dtype=np.float32)
    trajectory = np.zeros((len(TRAJECTORY_CHANNELS), 2, 2), dtype=np.float32)
    fov = np.asarray([[[1.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
    state[STATE_CHANNELS.index("current_unobservable_mask")] = np.asarray(
        [[1.0, 1.0], [1.0, 0.0]], dtype=np.float32
    )
    state[STATE_CHANNELS.index("last_seen_occupancy")] = np.asarray(
        [[0.0, 1.0], [0.0, 0.0]], dtype=np.float32
    )
    state[STATE_CHANNELS.index("occlusion_age_map")] = np.asarray(
        [[0.0, 1.0], [0.5, 0.0]], dtype=np.float32
    )
    trajectory[TRAJECTORY_CHANNELS.index("swept_volume_mask")] = np.asarray(
        [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    return state, trajectory, fov


def test_three_legal_input_baselines_match_hand_calculation():
    state, trajectory, fov = _baseline_inputs()

    assert visible_area_score(
        state_channels=state, verification_fov_mask=fov
    ) == pytest.approx(3.0)
    assert critical_swept_coverage_score(
        state_channels=state,
        trajectory_channels=trajectory,
        verification_fov_mask=fov,
    ) == pytest.approx(0.5)
    expected_entropy = np.log(2.0) + (
        -0.25 * np.log(0.25) - 0.75 * np.log(0.75)
    )
    assert occupancy_entropy_reduction_score(
        state_channels=state, verification_fov_mask=fov
    ) == pytest.approx(expected_entropy)


def test_baselines_reject_nonfinite_or_wrong_legal_input_shapes():
    state, trajectory, fov = _baseline_inputs()
    bad = state.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        visible_area_score(state_channels=bad, verification_fov_mask=fov)
    with pytest.raises(ValueError, match="trajectory_channels"):
        critical_swept_coverage_score(
            state_channels=state,
            trajectory_channels=trajectory[:2],
            verification_fov_mask=fov,
        )


def test_metrics_match_group_local_hand_calculation_and_slice_counts():
    actions = CANONICAL_ACTION_IDS[:3] * 2
    report = evaluate_verification_predictions(
        value_prediction=np.asarray([2.0, 3.0, 1.0, 0.0, 1.0, 2.0]),
        useful_probability=np.asarray([0.9, 0.8, 0.2, 0.1, 0.4, 0.9]),
        value_target=np.asarray([3.0, 2.0, 1.0, 0.0, 1.0, 2.0]),
        useful_target=np.asarray([1, 1, 0, 0, 1, 1]),
        group_ids=("g0", "g0", "g0", "g1", "g1", "g1"),
        action_ids=actions,
        huber_delta=1.0,
        slice_fields={
            "target_object_type": (
                "human",
                "human",
                "human",
                "cart",
                "cart",
                "cart",
            )
        },
    )

    assert report["sample_count"] == 6
    assert report["group_count"] == 2
    assert report["useful_f1"] == pytest.approx(6.0 / 7.0)
    assert report["value_mse"] == pytest.approx(1.0 / 3.0)
    assert report["value_huber"] == pytest.approx(1.0 / 6.0)
    assert report["pairwise_accuracy"] == pytest.approx(5.0 / 6.0)
    assert report["pair_count"] == 6
    assert report["top1_regret_mean"] == pytest.approx(0.5)
    assert report["top_two_selection_rate"] == pytest.approx(1.0)
    assert report["selected_action_counts"] == {
        CANONICAL_ACTION_IDS[0]: 0,
        CANONICAL_ACTION_IDS[1]: 1,
        CANONICAL_ACTION_IDS[2]: 1,
    }
    assert report["oracle_best_action_counts"] == {
        CANONICAL_ACTION_IDS[0]: 1,
        CANONICAL_ACTION_IDS[1]: 0,
        CANONICAL_ACTION_IDS[2]: 1,
    }
    assert report["oracle_second_action_counts"] == {
        CANONICAL_ACTION_IDS[0]: 0,
        CANONICAL_ACTION_IDS[1]: 2,
        CANONICAL_ACTION_IDS[2]: 0,
    }
    assert report["slices"]["target_object_type"]["human"]["sample_count"] == 3
    assert report["slices"]["target_object_type"]["cart"]["sample_count"] == 3


def test_rank_correlations_and_pairwise_tie_policy_are_explicit():
    increasing = np.asarray([1.0, 2.0, 3.0, 4.0])
    decreasing = increasing[::-1].copy()
    assert spearman_correlation(increasing, increasing) == pytest.approx(1.0)
    assert spearman_correlation(increasing, decreasing) == pytest.approx(-1.0)
    assert kendall_tau_b(increasing, increasing) == pytest.approx(1.0)
    assert kendall_tau_b(increasing, decreasing) == pytest.approx(-1.0)
    accuracy, count = pairwise_ranking_accuracy(
        np.asarray([0.0, 0.0]),
        np.asarray([0.0, 1.0]),
        group_ids=("group", "group"),
        action_ids=("left", "right"),
    )
    assert count == 1
    assert accuracy == pytest.approx(0.5)
    assert spearman_correlation(np.ones(3), np.ones(3)) == pytest.approx(0.0)
    assert kendall_tau_b(np.ones(3), np.ones(3)) == pytest.approx(0.0)


def test_checkpoint_manifest_v2_binds_all_frozen_inputs_and_rejects_legacy():
    config = load_verify_model_config(ROOT / "configs/verify_model.yaml")
    model_config = asdict(config)
    manifest = build_verification_checkpoint_manifest(
        input_manifest_digest="a" * 64,
        split_digests={"train": "b" * 64},
        model_config=model_config,
        seed=42,
        code_version="c" * 40,
    )

    assert manifest["manifest_version"] == VERIFICATION_CHECKPOINT_MANIFEST_VERSION
    validated = validate_verification_checkpoint_manifest(
        manifest,
        expected_input_manifest_digest="a" * 64,
        expected_split_digests={"train": "b" * 64},
        expected_model_config=model_config,
        expected_seed=42,
        expected_code_version="c" * 40,
    )
    assert validated == manifest

    for field, invalid, match in (
        ("manifest_version", "verification_checkpoint_manifest_v1", "legacy"),
        ("schema_version", "2.0.0", "schema"),
        ("history_channels", ["wrong"], "channel"),
        ("action_order", list(reversed(CANONICAL_ACTION_IDS)), "action"),
        ("input_manifest_digest", "d" * 64, "input manifest"),
    ):
        changed = {**manifest, field: invalid}
        with pytest.raises(ValueError, match=match):
            validate_verification_checkpoint_manifest(
                changed,
                expected_input_manifest_digest="a" * 64,
                expected_split_digests={"train": "b" * 64},
                expected_model_config=model_config,
                expected_seed=42,
                expected_code_version="c" * 40,
            )
