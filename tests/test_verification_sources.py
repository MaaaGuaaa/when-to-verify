from copy import deepcopy
import json

import numpy as np

import pytest

from src.datasets.verification_sources import (
    VERIFICATION_SOURCE_INDEX_VERSION,
    load_verification_source_index,
    select_source_shards,
    validate_source_snippet_record_identity,
)
from src.datasets.snippet_library import MotionSnippet
from src.generation.event_target_motion_shard import EventTargetMotionRecord


BATCH_DIGEST = "a" * 64
COLLECTION_DIGEST = "b" * 64


def _batch_handoff():
    return {
        "artifact_role": "sop05_train_batch_complete_index",
        "batch_semantic_digest_sha256": BATCH_DIGEST,
        "batch_state": "complete",
        "code_commit": "1" * 40,
        "common_contracts": {
            "input_lock": {
                "version": "sop05_input_lock_v2",
                "split": "train",
                "sop03": {
                    "audit_sha256": "2" * 64,
                    "checksum_manifest_sha256": "3" * 64,
                    "code_commit": "4" * 40,
                    "completion_policy": "sop03_complete_marker_v1",
                },
                "sop04": {
                    "audit_sha256": "5" * 64,
                    "bank_semantic_digest_sha256": "6" * 64,
                    "checksum_manifest_sha256": "7" * 64,
                    "code_commit": "8" * 40,
                    "completion_policy": "sop04_audited_bank_v2",
                    "dt_s": 0.2,
                    "external_handoff_digest_sha256": "9" * 64,
                    "first_pose_time_s": 0.2,
                    "last_pose_time_s": 3.0,
                    "pose_time_layout_version": "future_endpoints_dt_to_horizon_v1",
                    "pose_time_offsets_sha256": "c" * 64,
                    "trajectory_bank_version": "sop04_audited_bank_v2",
                    "trajectory_steps": 15,
                },
            }
        },
        "counts": {"events": 2, "planned_events": 2, "shards": 2},
        "handoff_version": "sop05_batch_index_handoff_v1",
        "schema_version": "3.0.0",
        "shards": [
            {
                "event_count": 1,
                "publication_semantic_digest": "d" * 64,
                "relative_root": "shard-00000",
                "run_id": "run-zero",
                "shard_index": 0,
                "trajectory_id": "forward_v00_w02",
            },
            {
                "event_count": 1,
                "publication_semantic_digest": "e" * 64,
                "relative_root": "shard-00001",
                "run_id": "run-one",
                "shard_index": 1,
                "trajectory_id": "forward_v01_w01",
            },
        ],
    }


def _collection_handoff():
    return {
        "artifact_role": "sop07_train_collection_complete_handoff",
        "code_commit": "f" * 40,
        "collection_semantic_digest_sha256": COLLECTION_DIGEST,
        "collection_state": "complete",
        "downstream_contract": {
            "generation_evidence_join": "PROVEN",
            "global_cross_split_leakage": "NOT_PROVEN",
            "global_sample_id_uniqueness": "PROVEN",
        },
        "event_type_counts": {
            "collision": 2,
            "empty_blind_spot": 2,
            "irrelevant_hidden": 2,
            "near_miss": 2,
            "spatial_safe": 2,
            "temporal_safe": 1,
        },
        "handoff_version": "sop07_collection_complete_handoff_v1",
        "launch_evidence": {
            "source_sop05_batch": {
                "batch_semantic_digest_sha256": BATCH_DIGEST,
                "event_count": 2,
                "shard_count": 2,
            }
        },
        "sample_count": 11,
        "schema_version": "3.0.0",
        "shard_count": 2,
        "shards": [
            {"relative_root": "shard-00000", "shard_index": 0},
            {"relative_root": "shard-00001", "shard_index": 1},
        ],
        "split": "train",
    }


def _write_handoffs(tmp_path, *, batch=None, collection=None):
    batch_path = tmp_path / "sop05" / "batch_complete_handoff.json"
    collection_path = tmp_path / "sop07" / "collection_complete_handoff.json"
    batch_path.parent.mkdir(parents=True)
    collection_path.parent.mkdir(parents=True)
    batch_path.write_text(json.dumps(batch or _batch_handoff()), encoding="utf-8")
    collection_path.write_text(
        json.dumps(collection or _collection_handoff()), encoding="utf-8"
    )
    return batch_path, collection_path


def test_loads_train_only_index_and_deterministic_shard_selection(tmp_path):
    batch_path, collection_path = _write_handoffs(tmp_path)

    index = load_verification_source_index(
        batch_path,
        collection_path,
        expected_sop05_batch_digest=BATCH_DIGEST,
        expected_sop07_collection_digest=COLLECTION_DIGEST,
    )
    first = select_source_shards(index, count=1, seed=17)
    repeated = select_source_shards(index, count=1, seed=17)

    assert index.version == VERIFICATION_SOURCE_INDEX_VERSION
    assert index.split == "train"
    assert index.scientific_status == "train_smoke_only"
    assert index.global_cross_split_leakage == "NOT_PROVEN"
    assert index.temporal_safe_count == 1
    assert index.event_count == 2
    assert index.sop07_sample_count == 11
    assert first == repeated
    assert len(first) == 1
    assert first[0].root == batch_path.parent / first[0].relative_root


@pytest.mark.parametrize(
    ("target", "path", "value", "message"),
    [
        ("batch", ("schema_version",), "2.0.0", "schema"),
        ("batch", ("batch_state",), "partial", "complete"),
        ("batch", ("common_contracts", "input_lock", "split"), "val", "split"),
        ("collection", ("split",), "val", "train"),
        (
            "collection",
            ("downstream_contract", "global_cross_split_leakage"),
            "PROVEN",
            "NOT_PROVEN",
        ),
        ("batch", ("shards", 0, "relative_root"), "../escape", "relative"),
    ],
)
def test_rejects_contract_mismatch_and_unsafe_relative_paths(
    tmp_path, target, path, value, message
):
    batch = _batch_handoff()
    collection = _collection_handoff()
    selected = batch if target == "batch" else collection
    cursor = selected
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    batch_path, collection_path = _write_handoffs(
        tmp_path, batch=batch, collection=collection
    )

    with pytest.raises(ValueError, match=message):
        load_verification_source_index(
            batch_path,
            collection_path,
            expected_sop05_batch_digest=BATCH_DIGEST,
            expected_sop07_collection_digest=COLLECTION_DIGEST,
        )


def test_requires_external_digest_anchors_and_sop05_input_lock(tmp_path):
    batch_path, collection_path = _write_handoffs(tmp_path)
    with pytest.raises(ValueError, match="external trust anchor"):
        load_verification_source_index(
            batch_path,
            collection_path,
            expected_sop05_batch_digest="",
            expected_sop07_collection_digest=COLLECTION_DIGEST,
        )

    batch = deepcopy(_batch_handoff())
    del batch["common_contracts"]["input_lock"]["sop04"]
    batch_path, collection_path = _write_handoffs(
        tmp_path / "missing-lock", batch=batch
    )
    with pytest.raises(ValueError, match="input lock"):
        load_verification_source_index(
            batch_path,
            collection_path,
            expected_sop05_batch_digest=BATCH_DIGEST,
            expected_sop07_collection_digest=COLLECTION_DIGEST,
        )


def test_snippet_geometry_is_compared_to_nested_record_footprint():
    positions = np.zeros((23, 2), dtype=np.float32)
    snippet = MotionSnippet(
        snippet_id="snippet-a",
        split="train",
        source_recording_id="recording-a",
        source_session_id="session-a",
        source_object_id="object-a",
        object_type="human",
        footprint={"kind": "circle", "radius_m": 0.3},
        start_timestamp=0.0,
        positions=positions,
        velocities=positions.copy(),
        headings=np.zeros(23, dtype=np.float32),
        duration_s=4.4,
        mean_speed_mps=0.5,
        max_acceleration_mps2=0.1,
        mean_abs_curvature_per_m=0.0,
        provenance={},
    )
    record = EventTargetMotionRecord(
        schema_version="3.0.0",
        layout_version="history8_current7_future15_v1",
        generated_event_id="event-a",
        world_id="world-a",
        base_state_id="base-a",
        trajectory_id="trajectory-a",
        target_dynamic_object_id="target-a",
        source_snippet_id="snippet-a",
        source_object_id="object-a",
        object_type="human",
        footprint_spec={
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": 0.3},
        },
        footprint_spec_digest="digest-a",
        target_type_policy_digest="policy-a",
        history_poses=np.zeros((8, 3), dtype=np.float32),
        current_pose=np.zeros(3, dtype=np.float32),
        future_poses=np.zeros((15, 3), dtype=np.float32),
        history_array_digest="history-a",
        future_array_digest="future-a",
        record_digest="record-a",
    )

    validate_source_snippet_record_identity(snippet, record, split="train")

    bad = deepcopy(record)
    bad.footprint_spec["footprint"]["radius_m"] = 0.4
    with pytest.raises(ValueError, match="footprint"):
        validate_source_snippet_record_identity(snippet, bad, split="train")
