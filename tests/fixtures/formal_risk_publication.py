"""Compact formal SOP03/SOP07 publication fixture for dataset-seal tests.

The fixture deliberately exercises the real ``write_risk_shard`` API.  It
publishes two contiguous immutable schema-3 shards containing twelve samples,
then writes a compact handoff with the same identity fields as the accepted
SOP07 collection handoff.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from src.contracts import (
    POSE_TIME_LAYOUT_VERSION,
    SCHEMA_VERSION,
    GridSpec,
    RiskSample,
    build_grid_spec,
)
from src.datasets.shard_writer import (
    RISK_SHARD_LAYOUT_VERSION,
    load_risk_shard,
    write_risk_shard,
)
from src.datasets.sop03_publication import publish_checksum_envelope
from src.generation.observation_renderer import RENDERER_LAYOUT_VERSION
from src.generation.risk_gt import RISK_GT_VERSION
from src.utils.config import DEFAULT_CONFIG, load_config


G1_SPLIT_MANIFEST_DIGEST = "1" * 32
TARGET_TYPE_POLICY_DIGEST = "2" * 32
SOP07_CODE_COMMIT = "3" * 40
SOP03_CODE_COMMIT = "4" * 40


@dataclass(frozen=True)
class FormalRiskPublication:
    """Paths and frozen identities for one compact formal publication."""

    root: Path
    collection_root: Path
    base_config_path: Path
    split_provenance_path: Path
    handoff_path: Path
    handoff_sha256: str
    grid: GridSpec
    g1_split_manifest_digest: str
    target_type_policy_digest: str


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_canonical_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_bytes((canonical_json(dict(value)) + "\n").encode("utf-8"))


def rewrite_collection_handoff(
    publication: FormalRiskPublication,
    handoff: Mapping[str, object],
) -> str:
    payload = canonical_json(dict(handoff)).encode("utf-8")
    publication.handoff_path.write_bytes(payload)
    return sha256_bytes(payload)


def resign_dataset_seal(seal_root: Path) -> None:
    """Refresh only the outer seal checksum envelope after a test mutation."""

    marker = seal_root / ".producer-complete"
    lines = [
        f"{sha256_file(marker)}  .producer-complete\n",
        f"{sha256_file(seal_root / 'dataset_manifest.json')}  dataset_manifest.json\n",
    ]
    (seal_root / "checksums.sha256").write_text(
        "".join(sorted(lines, key=lambda line: line.split("  ", 1)[1])),
        encoding="utf-8",
    )


def _base_config() -> dict[str, object]:
    config = deepcopy(DEFAULT_CONFIG)
    config["bev"].update(  # type: ignore[union-attr]
        {
            "range_m": 2.0,
            "resolution_m": 0.5,
            "size": 4,
            "history_steps": 2,
            "history_dt_s": 0.2,
            "future_steps": 3,
            "future_dt_s": 0.2,
        }
    )
    return config


def _sample(
    index: int,
    *,
    grid: GridSpec,
    split: str,
    target_type_policy_digest: str,
) -> RiskSample:
    event_types = (
        "collision",
        "near_miss",
        "irrelevant_hidden",
        "empty_blind_spot",
        "spatial_safe",
        "temporal_safe",
    )
    event_type = event_types[index % len(event_types)]
    collision = event_type == "collision"
    # The formal temporal-safe row is intentionally a hard negative: its
    # published labels remain collision=0 and near_miss=1.
    near_miss = event_type in {"near_miss", "temporal_safe"}
    value = float(index + 1) / 20.0
    sample_id = f"{split}-formal-{index:03d}"
    base_state_id = f"base-state-{index:03d}"
    return RiskSample(
        sample_id=sample_id,
        split=split,
        base_state_id=base_state_id,
        pair_group_id=f"pair-group-{index:03d}",
        event_type=event_type,
        bev_history=np.full(
            (
                grid.history_steps,
                grid.n_history_channels,
                grid.height,
                grid.width,
            ),
            value,
            dtype=np.float32,
        ),
        state_channels=np.full(
            (grid.n_state_channels, grid.height, grid.width),
            value + 0.1,
            dtype=np.float32,
        ),
        trajectory_channels=np.full(
            (grid.n_trajectory_channels, grid.height, grid.width),
            value + 0.2,
            dtype=np.float32,
        ),
        robot_state=np.asarray([value, -value], dtype=np.float32),
        collision_label=int(collision),
        risk_severity=1.0 if collision else (0.4 if near_miss else 0.1),
        min_clearance=-0.05 if collision else (0.2 if near_miss else 0.8),
        near_miss=int(near_miss),
        first_collision_time=0.2 if collision else None,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "renderer": {
                "renderer_layout_version": RENDERER_LAYOUT_VERSION,
                "base_state_id": base_state_id,
                "sensor_config_digest": f"sensor-{index:03d}",
                "static_occupancy_digest": f"static-{index:03d}",
            },
            "trajectory_id": f"trajectory-{index % 3}",
            "provenance": {
                "base_recording_id": f"base-recording-{index:03d}",
                "base_session_id": "formal-base-session",
                "source_recording_id": f"source-recording-{index:03d}",
                "source_session_id": "formal-source-session",
                "source_snippet_id": f"source-snippet-{index:03d}",
                "seed_namespace": f"sop07/{split}/formal-{index:03d}",
                "target_type_policy_digest": target_type_policy_digest,
            },
            "label_audit": {
                "risk_gt_version": RISK_GT_VERSION,
                "pose_time_layout_version": POSE_TIME_LAYOUT_VERSION,
                "critical_object_id": f"hidden-human-{index:03d}",
                "critical_object_type": "human",
                "time_to_min_clearance_s": 0.2,
                "has_hidden_target": True,
            },
        },
    )


def create_formal_risk_publication(
    root: Path,
    *,
    split: str = "train",
    g1_split_manifest_digest: str = G1_SPLIT_MANIFEST_DIGEST,
    target_type_policy_digest: str = TARGET_TYPE_POLICY_DIGEST,
    runtime_metadata: Mapping[str, object] | None = None,
) -> FormalRiskPublication:
    """Publish one compact formal SOP03 provenance root and SOP07 collection."""

    root.mkdir(parents=True, exist_ok=False)
    config_dir = root / "config"
    config_dir.mkdir()
    base_config_path = config_dir / "base.yaml"
    base_config_path.write_text(
        yaml.safe_dump(_base_config(), sort_keys=True), encoding="utf-8"
    )
    base_config = load_config(base_config_path)
    grid = build_grid_spec(base_config)

    sop03_root = root / "sop03"
    sop03_root.mkdir()
    split_provenance_path = sop03_root / "run_manifest.json"
    run_manifest = {
        "run_id": "formal-sop03-fixture",
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "repository": {
            "code_commit": SOP03_CODE_COMMIT,
            "finalizer_commit": SOP03_CODE_COMMIT,
        },
        "inputs": {
            "dataset": "FORMAL-RISK-FIXTURE",
            "split_manifest_digest": g1_split_manifest_digest,
        },
        "producer_protocol": {
            "config_snapshots": {
                "base": {
                    "path": "configs/base.yaml",
                    "sha256": sha256_file(base_config_path),
                    "value": base_config,
                }
            }
        },
        "environment": dict(runtime_metadata or {}),
        "validation": {"status": "passed"},
    }
    write_canonical_json(split_provenance_path, run_manifest)
    publish_checksum_envelope(sop03_root, workers=1)

    collection_root = root / "collection"
    collection_root.mkdir()
    samples = tuple(
        _sample(
            index,
            grid=grid,
            split=split,
            target_type_policy_digest=target_type_policy_digest,
        )
        for index in range(12)
    )
    descriptors: list[dict[str, object]] = []
    for shard_index in range(2):
        shard_root = collection_root / f"shard-{shard_index:05d}"
        shard_samples = samples[shard_index * 6 : (shard_index + 1) * 6]
        paths = write_risk_shard(
            shard_samples,
            shard_root,
            grid=grid,
            shard_index=shard_index,
            expected_sample_count=6,
        )
        loaded = load_risk_shard(shard_root, grid=grid)
        descriptors.append(
            {
                "array_layout_digest_sha256": sha256_bytes(
                    canonical_json(loaded.summary["array_layout"]).encode("utf-8")
                ),
                "audit_context_digest": loaded.summary["audit_context_digest"],
                "boundary": loaded.summary["boundary"],
                "formal_loader_verified": True,
                "manifest_digest": loaded.manifest_digest,
                "metadata_sha256": sha256_file(paths["manifest"]),
                "payload_sha256": sha256_file(paths["payload"]),
                "relative_root": shard_root.name,
                "sample_count": len(loaded.samples),
                "semantic_digest": loaded.semantic_digest,
                "shard_index": shard_index,
                "summary_sha256": sha256_file(paths["summary"]),
            }
        )

    collection_semantics = {
        "schema_version": SCHEMA_VERSION,
        "layout_version": RISK_SHARD_LAYOUT_VERSION,
        "split": split,
        "sample_count": 12,
        "shards": [
            {
                key: descriptor[key]
                for key in (
                    "shard_index",
                    "relative_root",
                    "sample_count",
                    "manifest_digest",
                    "semantic_digest",
                )
            }
            for descriptor in descriptors
        ],
    }
    semantic_digest = sha256_bytes(
        canonical_json(collection_semantics).encode("utf-8")
    )
    handoff = {
        "artifact_role": f"sop07_{split}_collection_complete_handoff",
        "builder_version": "sop07_collection_handoff_builder_v1",
        "code_commit": SOP07_CODE_COMMIT,
        "collection_instance_digest_sha256": sha256_bytes(
            canonical_json(
                {"semantics": collection_semantics, "runtime": runtime_metadata or {}}
            ).encode("utf-8")
        ),
        "collection_semantic_digest_sha256": semantic_digest,
        "collection_state": "complete",
        "downstream_contract": {
            "consumption": "fan_out_or_stream_shards_in_shard_index_order",
            "global_sample_id_uniqueness": "PROVEN",
            "physical_npz_merge_performed": False,
        },
        "handoff_version": "sop07_collection_complete_handoff_v1",
        "layout_version": RISK_SHARD_LAYOUT_VERSION,
        "producer_version": "sop07_risk_dataset_cli_v3",
        "runtime_metadata": dict(runtime_metadata or {}),
        "sample_count": 12,
        "schema_version": SCHEMA_VERSION,
        "shard_count": 2,
        "shards": descriptors,
        "split": split,
    }
    handoff_path = collection_root / "collection_complete_handoff.json"
    handoff_payload = canonical_json(handoff).encode("utf-8")
    handoff_path.write_bytes(handoff_payload)
    return FormalRiskPublication(
        root=root,
        collection_root=collection_root,
        base_config_path=base_config_path,
        split_provenance_path=split_provenance_path,
        handoff_path=handoff_path,
        handoff_sha256=sha256_bytes(handoff_payload),
        grid=grid,
        g1_split_manifest_digest=g1_split_manifest_digest,
        target_type_policy_digest=target_type_policy_digest,
    )


__all__ = [
    "FormalRiskPublication",
    "G1_SPLIT_MANIFEST_DIGEST",
    "TARGET_TYPE_POLICY_DIGEST",
    "canonical_json",
    "create_formal_risk_publication",
    "resign_dataset_seal",
    "rewrite_collection_handoff",
    "sha256_file",
    "write_canonical_json",
]
