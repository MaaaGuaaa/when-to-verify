from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.contracts import VerificationSample
from src.datasets.verification_release_collection import (
    load_calibrated_verification_release,
)
from src.evaluation.verification_value_calibration import (
    LoadedRejectCostCalibration,
)
from src.generation.verification_release import VerificationRevaluationRecord
from src.planning.verification_actions import CANONICAL_ACTION_IDS


def _samples(*, split: str = "train"):
    rows = []
    shared = np.zeros((1, 1, 1), dtype=np.float32)
    vector = np.zeros((4,), dtype=np.float32)
    for index, action_id in enumerate(CANONICAL_ACTION_IDS):
        rows.append(
            VerificationSample(
                sample_id=f"sample-{index}",
                split=split,
                base_state_id="base",
                nominal_trajectory_id="nominal",
                verification_action_id=action_id,
                bev_history=shared,
                state_channels=shared,
                trajectory_channels=shared,
                verification_fov_mask=shared,
                verification_action_vector=vector,
                value_target=-0.05,
                useful_target=0,
                br_before=0.2,
                post_risk=0.25,
                metadata={"ranking_group_id": "group"},
            )
        )
    return tuple(rows)


def _records(*, split: str = "train"):
    return tuple(
        VerificationRevaluationRecord(
            release_request_identity="release-a",
            split=split,
            task_id="task",
            mother_id="mother",
            sample_id=f"sample-{index}",
            ranking_group_id="group",
            action_id=action_id,
            realized_execute_loss=1.0,
            unclipped_best_policy_loss=0.1 if index % 2 == 0 else 0.4,
            action_cost=0.05,
            original_reject_cost=0.2,
        )
        for index, action_id in enumerate(CANONICAL_ACTION_IDS)
    )


def _calibration(*, source_digest: str = "a" * 64):
    return LoadedRejectCostCalibration(
        root=None,
        status="selected",
        selected_reject_cost=0.3,
        calibration_digest="c" * 64,
        source_release_manifest_digests=(source_digest,),
        source_release_request_identities=("release-a",),
        calibration={},
        manifest={},
    )


def _patch_release(
    monkeypatch,
    tmp_path,
    *,
    split: str = "train",
    records=None,
):
    import src.datasets.verification_release_collection as module

    data_root = tmp_path / "release/shards/shard-00000/data"
    data_root.mkdir(parents=True)
    samples = _samples(split=split)
    release = SimpleNamespace(
        root=tmp_path / "release",
        request_identity="release-a",
        split=split,
        sample_count=6,
        accepted_group_count=1,
        shard_count=1,
        manifest_digest="a" * 64,
        manifest={
            "shards": [
                {
                    "relative_root": "shards/shard-00000",
                    "accepted_group_count": 1,
                    "sample_count": 6,
                }
            ]
        },
    )
    shard = SimpleNamespace(
        samples=samples,
        manifest_digest="d" * 64,
        semantic_digest="e" * 64,
        summary={"split": split},
    )
    monkeypatch.setattr(module, "load_verification_release", lambda _: release)
    monkeypatch.setattr(
        module,
        "load_verification_revaluation_records",
        lambda _: _records(split=split) if records is None else records,
    )
    monkeypatch.setattr(
        module,
        "load_verification_shard",
        lambda *args, **kwargs: shard,
    )
    monkeypatch.setattr(
        module,
        "validate_verification_sample_for_publication",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "verification_input_digests",
        lambda _: ("d" * 64, {split: "e" * 64}),
    )
    return samples


def test_release_collection_revalues_only_labels_and_binds_calibration(
    tmp_path,
    monkeypatch,
):
    original = _patch_release(monkeypatch, tmp_path)

    loaded = load_calibrated_verification_release(
        tmp_path / "release",
        grid=object(),
        library=object(),
        expected_split="train",
        calibration=_calibration(),
    )

    assert loaded.release_manifest_digest == "a" * 64
    assert loaded.calibration_digest == "c" * 64
    assert loaded.reject_cost == pytest.approx(0.3)
    assert loaded.split == "train"
    assert loaded.split_digests == {"train": loaded.split_digest}
    assert loaded.samples[0].br_before == pytest.approx(0.3)
    assert loaded.samples[0].post_risk == pytest.approx(0.15)
    assert loaded.samples[0].value_target == pytest.approx(0.15)
    assert loaded.samples[0].useful_target == 1
    assert loaded.samples[1].value_target == pytest.approx(-0.05)
    for old, new in zip(original, loaded.samples, strict=True):
        assert new.bev_history is old.bev_history
        assert new.state_channels is old.state_channels
        assert new.trajectory_channels is old.trajectory_channels
        assert new.verification_fov_mask is old.verification_fov_mask
        assert new.verification_action_vector is old.verification_action_vector


def test_release_collection_requires_exact_record_alignment(
    tmp_path,
    monkeypatch,
):
    rows = _records()[:-1]
    _patch_release(monkeypatch, tmp_path, records=rows)

    with pytest.raises(ValueError, match="sample IDs"):
        load_calibrated_verification_release(
            tmp_path / "release",
            grid=object(),
            library=object(),
            expected_split="train",
            calibration=_calibration(),
        )


def test_release_collection_enforces_split_and_train_source_binding(
    tmp_path,
    monkeypatch,
):
    _patch_release(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="split"):
        load_calibrated_verification_release(
            tmp_path / "release",
            grid=object(),
            library=object(),
            expected_split="val",
            calibration=_calibration(),
        )

    with pytest.raises(ValueError, match="source"):
        load_calibrated_verification_release(
            tmp_path / "release",
            grid=object(),
            library=object(),
            expected_split="train",
            calibration=_calibration(source_digest="f" * 64),
        )


def test_heldout_release_must_not_be_a_calibration_source(
    tmp_path,
    monkeypatch,
):
    _patch_release(monkeypatch, tmp_path, split="val")

    with pytest.raises(ValueError, match="held-out"):
        load_calibrated_verification_release(
            tmp_path / "release",
            grid=object(),
            library=object(),
            expected_split="val",
            calibration=_calibration(),
        )
