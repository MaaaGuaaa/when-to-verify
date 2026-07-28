"""Tests for SOP07 single-release to SOP08 collection adaptation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from src.datasets.risk_dataloader import RiskDataContractError
from src.datasets.risk_dataset_seal import (
    load_risk_dataset_seal,
    publish_risk_dataset_seal,
)
from src.datasets.shard_writer import load_risk_shard, write_risk_shard
from src.datasets.sop07_single_risk_collection import (
    SOP07_SINGLE_RISK_RELEASE_VERSION,
    publish_sop07_single_risk_collection,
)
from src.utils.config import load_config
from tests.fixtures.formal_risk_publication import (
    FormalRiskPublication,
    canonical_json,
    create_formal_risk_publication,
    sha256_file,
    write_canonical_json,
)


def _single_release(
    root: Path,
    publication: FormalRiskPublication,
) -> Path:
    release = root / "release"
    shard_root = release / "shards" / "shard-00000"
    source = load_risk_shard(
        publication.collection_root / "shard-00000",
        grid=publication.grid,
    )
    samples = []
    for sample in source.samples:
        metadata = deepcopy(sample.metadata)
        provenance = metadata["provenance"]
        assert isinstance(provenance, dict)
        provenance.pop("target_type_policy_digest")
        samples.append(replace(sample, metadata=metadata))
    write_risk_shard(
        tuple(samples),
        shard_root,
        grid=publication.grid,
        shard_index=0,
        expected_sample_count=len(samples),
    )
    loaded = load_risk_shard(shard_root, grid=publication.grid)
    request_payload = {
        "split": loaded.samples[0].split,
        "version": SOP07_SINGLE_RISK_RELEASE_VERSION,
    }
    request_identity = hashlib.sha256(
        b"sop07_single_risk_release_request_v1\0"
        + canonical_json(request_payload).encode("ascii")
    ).hexdigest()
    request = {
        "version": SOP07_SINGLE_RISK_RELEASE_VERSION,
        "request_identity": request_identity,
        "request": request_payload,
    }
    manifest = {
        "version": SOP07_SINGLE_RISK_RELEASE_VERSION,
        "request_identity": request_identity,
        "split": loaded.samples[0].split,
        "sample_count": len(samples),
        "shard_count": 1,
        "resolved_base_config": load_config(publication.base_config_path),
        "shards": [
            {
                "shard_index": 0,
                "relative_root": "shards/shard-00000",
                "sample_count": len(samples),
                "first_sample_id": loaded.samples[0].sample_id,
                "last_sample_id": loaded.samples[-1].sample_id,
                "risk_manifest_digest": loaded.manifest_digest,
                "risk_semantic_digest": loaded.semantic_digest,
            }
        ],
    }
    write_canonical_json(release / "request.json", request)
    write_canonical_json(release / "manifest.json", manifest)
    write_canonical_json(
        release / "checksums.json",
        {
            "request.json": sha256_file(release / "request.json"),
            "manifest.json": sha256_file(release / "manifest.json"),
        },
    )
    manifest_digest = hashlib.sha256(
        b"sop07_single_risk_release_manifest_v1\0"
        + canonical_json(manifest).encode("ascii")
    ).hexdigest()
    write_canonical_json(
        release / "COMPLETE.json",
        {
            "version": SOP07_SINGLE_RISK_RELEASE_VERSION,
            "request_identity": request_identity,
            "sample_count": len(samples),
            "shard_count": 1,
            "manifest_digest": manifest_digest,
        },
    )
    return release


@pytest.mark.parametrize("split", ("train", "val"))
def test_adapter_preserves_shards_and_supports_dataset_seal(
    tmp_path: Path,
    split: str,
) -> None:
    publication = create_formal_risk_publication(
        tmp_path / "upstream",
        split=split,
    )
    release = _single_release(tmp_path, publication)
    collection = tmp_path / "collection"
    result = publish_sop07_single_risk_collection(
        release,
        collection,
        target_type_policy_digest=publication.target_type_policy_digest,
        code_commit="a" * 40,
    )

    assert result.split == split
    assert result.shard_count == 1
    for name in ("samples.npz", "metadata.jsonl", "summary.json"):
        assert (release / "shards" / "shard-00000" / name).read_bytes() == (
            collection / "shard-00000" / name
        ).read_bytes()

    seal = publish_risk_dataset_seal(
        tmp_path / "seal",
        collection_root=collection,
        base_config_path=publication.base_config_path,
        split_provenance_path=publication.split_provenance_path,
        expected_split=split,
        expected_collection_handoff_sha256=result.handoff_sha256,
    )
    loaded = load_risk_dataset_seal(
        seal,
        collection_root=collection,
        expected_split=split,
    )
    assert loaded.sample_count == result.sample_count
    assert loaded.provenance["target_type_policy_digest"] == (
        publication.target_type_policy_digest
    )

    with pytest.raises(FileExistsError):
        publish_sop07_single_risk_collection(
            release,
            collection,
            target_type_policy_digest=publication.target_type_policy_digest,
            code_commit="a" * 40,
        )


def test_adapter_rejects_release_manifest_tampering(tmp_path: Path) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    release = _single_release(tmp_path, publication)
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["sample_count"] += 1
    write_canonical_json(manifest_path, manifest)

    with pytest.raises(RiskDataContractError, match="checksum mismatch"):
        publish_sop07_single_risk_collection(
            release,
            tmp_path / "collection",
            target_type_policy_digest=publication.target_type_policy_digest,
            code_commit="a" * 40,
        )
