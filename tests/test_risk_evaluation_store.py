"""Authenticated sibling storage for production evaluation records."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.datasets.risk_dataset_seal import load_risk_dataset_seal, publish_risk_dataset_seal
from src.datasets.risk_evaluation_metadata import (
    CONTACT_POLICY_RULE_VERSION,
    OOD_ROUTING_RULE_VERSION,
    PAIR_ELIGIBILITY_RULE_VERSION,
    ROBOT_FOOTPRINT_PROVENANCE_RULE_VERSION,
)
from src.datasets.risk_evaluation_store import (
    EvaluationCollectionError,
    load_risk_evaluation_replay_shard,
    load_risk_evaluation_collection,
    publish_risk_evaluation_replay_shard,
    publish_risk_evaluation_collection,
)
from src.datasets.shard_writer import load_risk_shard
from tests.fixtures.formal_risk_publication import (
    create_formal_risk_publication,
)


def _dataset(root: Path):
    publication = create_formal_risk_publication(
        root / "upstream",
        history_steps=8,
        future_steps=15,
    )
    seal = publish_risk_dataset_seal(
        root / "seal",
        collection_root=publication.collection_root,
        base_config_path=publication.base_config_path,
        split_provenance_path=publication.split_provenance_path,
        expected_split="train",
        expected_collection_handoff_sha256=publication.handoff_sha256,
    )
    return load_risk_dataset_seal(
        seal,
        collection_root=publication.collection_root,
        expected_split="train",
    )


def _record(sample) -> dict[str, object]:
    provenance = sample.metadata["provenance"]
    audit = sample.metadata["label_audit"]
    critical_object_id = audit["critical_object_id"]
    target_object_type = audit["critical_object_type"]
    target_footprint_spec = (
        None
        if critical_object_id is None
        else {"kind": "circle", "radius_m": 0.3}
    )
    return {
        "risk_evaluation_record_layout_version": "risk_evaluation_record_v1",
        "sample_id": sample.sample_id,
        "split": sample.split,
        "base_state_id": sample.base_state_id,
        "pair_group_id": sample.pair_group_id,
        "event_type": sample.event_type,
        "trajectory_id": provenance["trajectory_id"] if "trajectory_id" in provenance else sample.metadata["trajectory_id"],
        "base_recording_id": provenance["base_recording_id"],
        "base_session_id": provenance["base_session_id"],
        "source_recording_id": provenance["source_recording_id"],
        "source_session_id": provenance["source_session_id"],
        "source_object_id": f"source-object-{sample.sample_id}",
        "source_snippet_id": provenance["source_snippet_id"],
        "seed_namespace": provenance["seed_namespace"],
        "collision_label": sample.collision_label,
        "risk_severity": sample.risk_severity,
        "min_clearance": sample.min_clearance,
        "near_miss": sample.near_miss,
        "first_collision_time": sample.first_collision_time,
        "blind_type": "corner",
        "critical_area_fraction": 0.1,
        "age_s": 0.2,
        "critical_region_empty": False,
        "density_fraction": 0.1,
        "critical_object_id": critical_object_id,
        "target_object_type": target_object_type,
        "target_footprint_spec": target_footprint_spec,
        "footprint_kind": "none" if critical_object_id is None else "circle",
        "pair_eligible": False,
        "pair_eligibility_rule_version": PAIR_ELIGIBILITY_RULE_VERSION,
        "ood_tag": "in_distribution",
        "ood_evidence": {
            "rule_version": OOD_ROUTING_RULE_VERSION,
            "source": "default_in_distribution",
            "reason": "fixture",
        },
        "robot_footprint_spec": {
            "kind": "rectangle",
            "length_m": 1.0,
            "width_m": 0.8500000000000001,
        },
        "robot_footprint_provenance": {
            "rule_version": ROBOT_FOOTPRINT_PROVENANCE_RULE_VERSION,
            "base_footprint_spec": {
                "kind": "rectangle",
                "length_m": 0.7,
                "width_m": 0.55,
            },
            "inflation_m": 0.15,
            "effective_footprint_spec": {
                "kind": "rectangle",
                "length_m": 1.0,
                "width_m": 0.8500000000000001,
            },
            "base_config_digest": provenance["base_config_digest"],
        },
        "contact_policy_rule_version": CONTACT_POLICY_RULE_VERSION,
    }


def _records(dataset):
    return {
        descriptor.shard_index: tuple(
            _record(sample)
            for sample in load_risk_shard(
                dataset.collection_root / descriptor.relative_root,
                grid=dataset.grid,
            ).samples
        )
        for descriptor in dataset.shards
    }


def test_evaluation_collection_round_trips_against_sealed_shards(tmp_path: Path):
    dataset = _dataset(tmp_path)
    output = publish_risk_evaluation_collection(
        tmp_path / "evaluation",
        dataset=dataset,
        records_by_shard=_records(dataset),
    )

    loaded = load_risk_evaluation_collection(output, dataset=dataset)
    assert loaded.sample_ids == tuple(
        sample.sample_id
        for descriptor in dataset.shards
        for sample in load_risk_shard(
            dataset.collection_root / descriptor.relative_root,
            grid=dataset.grid,
        ).samples
    )
    assert loaded.sample_count == dataset.sample_count
    assert loaded.risk_dataset_manifest_digest == dataset.risk_dataset_manifest_digest
    assert loaded.collection_semantic_digest_sha256


def test_evaluation_collection_rejects_label_or_sample_id_drift(tmp_path: Path):
    dataset = _dataset(tmp_path)
    records = _records(dataset)
    records[0] = list(records[0])
    records[0][0] = dict(records[0][0])
    records[0][0]["min_clearance"] = -0.1
    with pytest.raises(EvaluationCollectionError, match="label mismatch"):
        publish_risk_evaluation_collection(
            tmp_path / "label-drift",
            dataset=dataset,
            records_by_shard=records,
        )
    assert not (tmp_path / "label-drift").exists()

    records = _records(dataset)
    records[0] = list(records[0])
    records[0][0] = dict(records[0][0])
    records[0][0]["sample_id"] = "wrong-sample-id"
    with pytest.raises(EvaluationCollectionError, match="sample_id"):
        publish_risk_evaluation_collection(
            tmp_path / "id-drift",
            dataset=dataset,
            records_by_shard=records,
        )
    assert not (tmp_path / "id-drift").exists()


def test_evaluation_collection_rejects_unknown_files_and_partial_publish(tmp_path: Path):
    dataset = _dataset(tmp_path)
    output = publish_risk_evaluation_collection(
        tmp_path / "evaluation",
        dataset=dataset,
        records_by_shard=_records(dataset),
    )
    (output / "unexpected.txt").write_text("tamper", encoding="utf-8")
    with pytest.raises(EvaluationCollectionError, match="layout|unexpected"):
        load_risk_evaluation_collection(output, dataset=dataset)

    (output / "unexpected.txt").unlink()
    (output / ".producer-complete").unlink()
    with pytest.raises(EvaluationCollectionError, match="complete|marker"):
        load_risk_evaluation_collection(output, dataset=dataset)


def test_replay_publisher_refuses_risk_semantic_digest_mismatch(tmp_path: Path):
    dataset = _dataset(tmp_path)
    records = _records(dataset)
    records[0] = list(records[0])
    records[0][0] = dict(records[0][0])
    records[0][0]["min_clearance"] = -0.1
    with pytest.raises(EvaluationCollectionError, match="label mismatch"):
        publish_risk_evaluation_collection(
            tmp_path / "mismatch",
            dataset=dataset,
            records_by_shard=records,
        )
    assert not (tmp_path / "mismatch").exists()


def test_evaluation_replay_shard_round_trips_against_exact_risk_shard(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    descriptor = dataset.shards[0]
    risk_shard = load_risk_shard(
        dataset.collection_root / descriptor.relative_root,
        grid=dataset.grid,
    )
    records = tuple(_record(sample) for sample in risk_shard.samples)

    output = publish_risk_evaluation_replay_shard(
        tmp_path / "evaluation-replay-shard",
        risk_shard=risk_shard,
        records=records,
    )
    loaded = load_risk_evaluation_replay_shard(
        output,
        risk_shard=risk_shard,
    )

    assert loaded.records == records
    assert loaded.sample_ids == tuple(sample.sample_id for sample in risk_shard.samples)
    assert loaded.source_risk_shard_semantic_digest == risk_shard.semantic_digest


def test_evaluation_replay_shard_rejects_risk_digest_or_row_drift(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    first_descriptor, second_descriptor = dataset.shards
    first = load_risk_shard(
        dataset.collection_root / first_descriptor.relative_root,
        grid=dataset.grid,
    )
    second = load_risk_shard(
        dataset.collection_root / second_descriptor.relative_root,
        grid=dataset.grid,
    )
    records = tuple(_record(sample) for sample in first.samples)
    output = publish_risk_evaluation_replay_shard(
        tmp_path / "evaluation-replay-shard",
        risk_shard=first,
        records=records,
    )

    with pytest.raises(
        EvaluationCollectionError,
        match="risk shard|digest|order|sample_id",
    ):
        load_risk_evaluation_replay_shard(output, risk_shard=second)

    drifted = list(records)
    drifted[0] = {**drifted[0], "collision_label": 1 - drifted[0]["collision_label"]}
    with pytest.raises(EvaluationCollectionError, match="label|record"):
        publish_risk_evaluation_replay_shard(
            tmp_path / "drifted-replay-shard",
            risk_shard=first,
            records=drifted,
        )
    assert not (tmp_path / "drifted-replay-shard").exists()
