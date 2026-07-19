from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.contracts import LocalTrajectory, validate_verification_sample
from src.datasets.verification_dataset import (
    VERIFICATION_DATASET_VERSION,
    VerificationGroupInput,
    build_verification_samples,
)
from src.datasets.verification_dataloader import (
    load_verification_collection,
    load_verification_shard,
    write_verification_shard,
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


def test_group_identity_exposes_cross_split_reuse_while_sample_id_stays_split_scoped():
    grid, library, train = _source_and_library(split="train")
    _, _, validation = _source_and_library(split="val")

    train_samples = build_verification_samples(train, library=library, grid=grid)
    validation_samples = build_verification_samples(
        validation, library=library, grid=grid
    )

    assert {item.split for item in train_samples} == {"train"}
    assert {item.split for item in validation_samples} == {"val"}
    assert train_samples[0].metadata["ranking_group_id"] == (
        validation_samples[0].metadata["ranking_group_id"]
    )
    assert {item.sample_id for item in train_samples}.isdisjoint(
        item.sample_id for item in validation_samples
    )


def test_verification_shard_round_trip_is_deterministic_and_pickle_free(tmp_path):
    grid, library, source = _source_and_library()
    samples = build_verification_samples(source, library=library, grid=grid)
    first_path = tmp_path / "shard-a"
    second_path = tmp_path / "shard-b"

    first_files = write_verification_shard(
        samples,
        first_path,
        grid=grid,
        library=library,
        shard_index=0,
        expected_sample_count=6,
    )
    write_verification_shard(
        tuple(reversed(samples)),
        second_path,
        grid=grid,
        library=library,
        shard_index=0,
        expected_sample_count=6,
    )
    loaded = load_verification_shard(
        first_path,
        grid=grid,
        library=library,
        recompute_value=lambda sample: sample.br_before - sample.post_risk,
    )
    repeated = load_verification_shard(second_path, grid=grid, library=library)

    assert set(path.name for path in first_path.iterdir()) == {
        "samples.npz",
        "metadata.jsonl",
        "summary.json",
    }
    assert loaded.semantic_digest == repeated.semantic_digest
    assert loaded.manifest_digest == repeated.manifest_digest
    assert loaded.summary["checksums"] == repeated.summary["checksums"]
    assert loaded.action_counts == {action_id: 1 for action_id in CANONICAL_ACTION_IDS}
    assert tuple(item.sample_id for item in loaded.samples) == tuple(
        sorted(item.sample_id for item in samples)
    )
    with np.load(first_files["payload"], allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    for expected, actual in zip(
        sorted(samples, key=lambda item: item.sample_id),
        loaded.samples,
        strict=True,
    ):
        assert actual.sample_id == expected.sample_id
        np.testing.assert_array_equal(actual.bev_history, expected.bev_history)
        np.testing.assert_array_equal(
            actual.verification_action_vector,
            expected.verification_action_vector,
        )

    with pytest.raises(FileExistsError, match="overwrite"):
        write_verification_shard(
            samples,
            first_path,
            grid=grid,
            library=library,
            shard_index=0,
            expected_sample_count=6,
        )


def test_writer_and_loader_reject_corrupt_numeric_or_metadata_payloads(tmp_path):
    grid, library, source = _source_and_library()
    samples = build_verification_samples(source, library=library, grid=grid)

    bad = samples[0].bev_history.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(Exception, match="NaN/Inf|finite"):
        write_verification_shard(
            (replace(samples[0], bev_history=bad), *samples[1:]),
            tmp_path / "nan",
            grid=grid,
            library=library,
            expected_sample_count=6,
        )

    write_verification_shard(
        samples,
        tmp_path / "payload-corrupt",
        grid=grid,
        library=library,
        expected_sample_count=6,
    )
    payload_path = tmp_path / "payload-corrupt" / "samples.npz"
    payload = bytearray(payload_path.read_bytes())
    payload[len(payload) // 2] ^= 1
    payload_path.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum|payload"):
        load_verification_shard(
            tmp_path / "payload-corrupt", grid=grid, library=library
        )

    write_verification_shard(
        samples,
        tmp_path / "metadata-corrupt",
        grid=grid,
        library=library,
        expected_sample_count=6,
    )
    manifest_path = tmp_path / "metadata-corrupt" / "metadata.jsonl"
    manifest_path.write_bytes(manifest_path.read_bytes().replace(b"train", b"test", 1))
    with pytest.raises(ValueError, match="checksum|manifest"):
        load_verification_shard(
            tmp_path / "metadata-corrupt", grid=grid, library=library
        )

@pytest.mark.parametrize("corruption", ["dtype", "shape", "nan"])
def test_loader_rejects_numeric_contract_corruption_even_with_updated_checksum(
    tmp_path, corruption
):
    grid, library, source = _source_and_library()
    samples = build_verification_samples(source, library=library, grid=grid)
    root = tmp_path / corruption
    write_verification_shard(
        samples,
        root,
        grid=grid,
        library=library,
        expected_sample_count=6,
    )
    payload_path = root / "samples.npz"
    with np.load(payload_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    bev = arrays["bev_history"].copy()
    if corruption == "dtype":
        bev = bev.astype(np.float64)
    elif corruption == "shape":
        bev = bev[:, :, :, :-1, :]
    else:
        bev[0, 0, 0, 0, 0] = np.nan
    arrays["bev_history"] = bev
    with payload_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checksums"]["payload_sha256"] = hashlib.sha256(
        payload_path.read_bytes()
    ).hexdigest()
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises((TypeError, ValueError), match=f"{corruption}|dtype|shape|finite"):
        load_verification_shard(root, grid=grid, library=library)


def test_incomplete_action_group_and_recomputation_mismatch_are_rejected(tmp_path):
    grid, library, source = _source_and_library()
    samples = build_verification_samples(source, library=library, grid=grid)
    with pytest.raises(ValueError, match="action imbalance"):
        write_verification_shard(
            samples[:-1],
            tmp_path / "incomplete",
            grid=grid,
            library=library,
            expected_sample_count=5,
        )

    root = tmp_path / "complete"
    write_verification_shard(
        samples,
        root,
        grid=grid,
        library=library,
        expected_sample_count=6,
    )
    with pytest.raises(ValueError, match=r"recomputed G\*"):
        load_verification_shard(
            root,
            grid=grid,
            library=library,
            recompute_value=lambda sample: sample.value_target + 1.0,
        )


def test_collection_rejects_cross_split_groups_and_duplicate_sample_ids(tmp_path):
    grid, library, train_source = _source_and_library(split="train")
    _, _, val_source = _source_and_library(split="val")
    train = build_verification_samples(train_source, library=library, grid=grid)
    val = build_verification_samples(val_source, library=library, grid=grid)
    write_verification_shard(
        train,
        tmp_path / "train",
        grid=grid,
        library=library,
        expected_sample_count=6,
    )
    write_verification_shard(
        val,
        tmp_path / "val",
        grid=grid,
        library=library,
        expected_sample_count=6,
    )
    with pytest.raises(ValueError, match="cross-split ranking group"):
        load_verification_collection(
            (tmp_path / "train", tmp_path / "val"),
            grid=grid,
            library=library,
        )

    write_verification_shard(
        train,
        tmp_path / "train-copy",
        grid=grid,
        library=library,
        expected_sample_count=6,
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_verification_collection(
            (tmp_path / "train", tmp_path / "train-copy"),
            grid=grid,
            library=library,
        )


def test_collection_batch_order_is_seeded_and_group_complete(tmp_path):
    grid, library, source = _source_and_library()
    samples = build_verification_samples(source, library=library, grid=grid)
    write_verification_shard(
        samples,
        tmp_path / "train",
        grid=grid,
        library=library,
        expected_sample_count=6,
    )
    dataset = load_verification_collection(
        (tmp_path / "train",), grid=grid, library=library
    )

    first = dataset.ordered_indices(seed=17, epoch=2, shuffle=True)
    repeated = dataset.ordered_indices(seed=17, epoch=2, shuffle=True)
    assert first == repeated
    assert tuple(sorted(first)) == tuple(range(6))
    batches = tuple(dataset.iter_batches(4, seed=17, epoch=2, shuffle=True))
    assert tuple(len(batch) for batch in batches) == (4, 2)
    assert {item.sample_id for batch in batches for item in batch} == {
        item.sample_id for item in samples
    }
