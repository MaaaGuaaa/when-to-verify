from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.contracts import LocalTrajectory, validate_verification_sample
from src.datasets.verification_dataset import (
    VERIFICATION_DATASET_VERSION,
    VerificationGroupInput,
    build_verification_samples,
)
from src.generation.verification_gt import (
    VERIFICATION_GT_VERSION,
    VerificationValueResult,
)
from src.planning.verification_actions import (
    CANONICAL_ACTION_IDS,
    load_verification_actions,
)
from tests.fixtures.verification_world import build_verification_toy_world


ROOT = Path(__file__).resolve().parents[1]
ACTION_CONFIG = ROOT / "configs/verification_actions.yaml"


def _nominal(grid) -> LocalTrajectory:
    zeros = np.zeros((grid.height, grid.width), dtype=np.float32)
    poses = np.zeros((grid.future_steps, 3), dtype=np.float32)
    poses[:, 0] = np.arange(1, grid.future_steps + 1, dtype=np.float32) * 0.1
    return LocalTrajectory(
        trajectory_id="nominal-toy",
        poses=poses,
        controls=np.zeros((grid.future_steps, 2), dtype=np.float32),
        swept_mask=zeros.copy(),
        tta_map=np.full_like(zeros, -1.0),
        braking_map=zeros.copy(),
        centerline_map=zeros.copy(),
        task_cost=0.05,
        metadata={
            "pose_time_layout_version": "future_endpoints_dt_to_horizon_v1"
        },
    )


def _value(action_id: str, index: int) -> VerificationValueResult:
    action_cost = 0.01 + 0.001 * index
    post_risk = 0.20 + action_cost
    value = 0.50 - post_risk
    return VerificationValueResult(
        version=VERIFICATION_GT_VERSION,
        bank_size=1,
        scenario_bank_digest="scenario-digest-toy",
        nominal_trajectory_id="nominal-toy",
        verification_action_id=action_id,
        posterior_mode="exact",
        posterior_temperature=None,
        posterior=np.ones((1, 1), dtype=np.float64),
        nominal_execute_losses=np.asarray([0.50], dtype=np.float64),
        mean_execute_loss=0.50,
        br_before=0.50,
        post_decision_risks=np.asarray([0.20], dtype=np.float64),
        best_decision_ids=("replan-toy",),
        mean_post_decision_risk_before_action_cost=0.20,
        action_cost=action_cost,
        post_risk=post_risk,
        value_target=value,
        useful_target=int(value > 0.0),
    )


def _source_and_library(*, split: str = "train"):
    toy = build_verification_toy_world()
    grid = toy.grid
    library = load_verification_actions(ACTION_CONFIG)
    bev = np.zeros(
        (
            grid.history_steps,
            grid.n_history_channels,
            grid.height,
            grid.width,
        ),
        dtype=np.float32,
    )
    state = np.zeros(
        (grid.n_state_channels, grid.height, grid.width), dtype=np.float32
    )
    masks = {}
    values = {}
    for index, action in enumerate(library.actions):
        mask = np.zeros((1, grid.height, grid.width), dtype=np.float32)
        mask[0, index, index] = 1.0
        masks[action.action_id] = mask
        values[action.action_id] = _value(action.action_id, index)
    source = VerificationGroupInput(
        split=split,
        base_state_id="base-state-toy",
        nominal_trajectory=_nominal(grid),
        bev_history=bev,
        state_channels=state,
        expected_fov_masks=masks,
        value_results=values,
        provenance={
            "source_mode": "toy",
            "source_artifact_digest": "source-digest-toy",
        },
    )
    return grid, library, source


def test_builds_canonical_six_action_group_and_validates_contract():
    grid, library, source = _source_and_library()

    samples = build_verification_samples(source, library=library, grid=grid)
    repeated = build_verification_samples(source, library=library, grid=grid)

    assert tuple(item.verification_action_id for item in samples) == (
        CANONICAL_ACTION_IDS
    )
    assert len(samples) == 6
    assert tuple(item.sample_id for item in samples) == tuple(
        item.sample_id for item in repeated
    )
    assert len({item.sample_id for item in samples}) == 6
    assert len({item.metadata["ranking_group_id"] for item in samples}) == 1
    for index, (sample, action) in enumerate(zip(samples, library.actions, strict=True)):
        validate_verification_sample(sample, grid)
        np.testing.assert_array_equal(sample.verification_action_vector, action.vector)
        np.testing.assert_array_equal(
            sample.verification_fov_mask,
            source.expected_fov_masks[action.action_id],
        )
        assert sample.metadata == {
            "schema_version": "3.0.0",
            "verification_dataset_version": VERIFICATION_DATASET_VERSION,
            "ranking_group_id": samples[0].metadata["ranking_group_id"],
            "action_index": index,
            "action_order": list(CANONICAL_ACTION_IDS),
            "provenance": {
                "source_artifact_digest": "source-digest-toy",
                "source_mode": "toy",
            },
            "label_audit": {
                "verification_gt_version": VERIFICATION_GT_VERSION,
                "scenario_bank_digest": "scenario-digest-toy",
                "posterior_mode": "exact",
                "posterior_temperature": None,
                "bank_size": 1,
            },
        }
        assert sample.bev_history.dtype == np.float32
        assert sample.state_channels.dtype == np.float32
        assert sample.trajectory_channels.dtype == np.float32
        assert not sample.bev_history.flags.writeable
        assert not sample.state_channels.flags.writeable
        assert not sample.trajectory_channels.flags.writeable
        assert not sample.verification_fov_mask.flags.writeable
        assert not sample.verification_action_vector.flags.writeable

    assert not np.shares_memory(samples[0].bev_history, source.bev_history)
    assert not np.shares_memory(samples[0].state_channels, source.state_channels)


def test_action_result_mismatch_and_non_static_mask_values_fail_closed():
    grid, library, source = _source_and_library()
    values = dict(source.value_results)
    values["yaw_left_10"] = replace(
        values["yaw_left_10"], verification_action_id="yaw_right_10"
    )
    with pytest.raises(ValueError, match="action ID"):
        build_verification_samples(
            replace(source, value_results=values), library=library, grid=grid
        )

    masks = dict(source.expected_fov_masks)
    masks["yaw_left_10"] = masks["yaw_left_10"].copy()
    masks["yaw_left_10"][0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        build_verification_samples(
            replace(source, expected_fov_masks=masks), library=library, grid=grid
        )


def test_split_is_part_of_group_and_sample_identity():
    grid, library, train = _source_and_library(split="train")
    _, _, validation = _source_and_library(split="val")

    train_samples = build_verification_samples(train, library=library, grid=grid)
    validation_samples = build_verification_samples(
        validation, library=library, grid=grid
    )

    assert {item.split for item in train_samples} == {"train"}
    assert {item.split for item in validation_samples} == {"val"}
    assert train_samples[0].metadata["ranking_group_id"] != (
        validation_samples[0].metadata["ranking_group_id"]
    )
    assert {item.sample_id for item in train_samples}.isdisjoint(
        item.sample_id for item in validation_samples
    )
