"""Formal publication and fail-closed loading tests for ``risk_dataset_v2``."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.contracts import (
    HISTORY_CHANNELS,
    INPUT_CHANNELS,
    STATE_CHANNELS,
    TRAJECTORY_CHANNELS,
    GridSpec,
)
from src.datasets.risk_dataloader import (
    RiskDataContractError,
    load_production_risk_dataset,
)
import src.datasets.risk_dataset_seal as seal_module
from src.datasets.risk_dataset_seal import (
    RISK_DATASET_FAMILY_LAYOUT_VERSION,
    RISK_DATASET_LAYOUT_VERSION,
    LoadedRiskDataset,
    RiskShardDescriptor,
    load_occupancy_sidecar_collection,
    canonical_dynamic_objects_digest,
    load_risk_dataset_seal,
    publish_risk_dataset_seal,
    validate_risk_dataset_manifest,
)
from src.datasets.shard_writer import load_risk_shard
from src.datasets.sop03_publication import publish_checksum_envelope
from tests.fixtures.formal_risk_publication import (
    FormalRiskPublication,
    canonical_json,
    create_formal_risk_publication,
    create_formal_risk_sidecar_publication,
    heldout_collection_semantic_digest,
    resign_dataset_seal,
    rewrite_collection_handoff,
    sha256_file,
    write_canonical_json,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "04_seal_risk_dataset.py"


def _publish(
    publication: FormalRiskPublication,
    output_dir: Path,
    *,
    expected_split: str = "train",
    expected_handoff_sha256: str | None = None,
    sidecar_root: Path | None = None,
) -> Path:
    return publish_risk_dataset_seal(
        output_dir,
        collection_root=publication.collection_root,
        base_config_path=publication.base_config_path,
        split_provenance_path=publication.split_provenance_path,
        expected_split=expected_split,
        expected_collection_handoff_sha256=(
            expected_handoff_sha256 or publication.handoff_sha256
        ),
        sidecar_root=sidecar_root,
    )


def _manifest(seal_root: Path) -> dict[str, object]:
    value = json.loads(
        (seal_root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def _mutate_manifest(
    seal_root: Path, path: tuple[str, ...], value: object
) -> dict[str, object]:
    manifest = _manifest(seal_root)
    node: dict[str, object] = manifest
    for key in path[:-1]:
        child = node[key]
        assert isinstance(child, dict)
        node = child
    node[path[-1]] = value
    write_canonical_json(seal_root / "dataset_manifest.json", manifest)
    resign_dataset_seal(seal_root)
    return manifest


def _republish_sop03_envelope(publication: FormalRiskPublication) -> None:
    root = publication.split_provenance_path.parent
    for name in (
        ".producer-complete",
        "artifact_checksums.sha256",
        "artifact_checksum_summary.json",
    ):
        (root / name).unlink()
    publish_checksum_envelope(root, workers=1)


def _rewrite_as_heldout_collection(
    publication: FormalRiskPublication,
    *,
    report_updates: dict[str, object] | None = None,
    evidence_updates: dict[str, object] | None = None,
    handoff_updates: dict[str, object] | None = None,
) -> str:
    """Rewrite the compact fixture using the accepted heldout handoff dialect."""

    handoff = json.loads(publication.handoff_path.read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "artifact_role": "sop07_heldout_batch_generation_report",
        "batch_generation_instance_digest_sha256": "5" * 64,
        "batch_generation_semantic_digest_sha256": "6" * 64,
        "code_commit": handoff["code_commit"],
        "conservation": {"status": "PROVEN"},
        "event_count": 12,
        "generation_state": "complete",
        "layout_version": handoff["layout_version"],
        "producer_version": "sop07_risk_dataset_cli_v3",
        "report_version": "sop07_heldout_batch_generation_report_v1",
        "sample_count": handoff["sample_count"],
        "schema_version": handoff["schema_version"],
        "shard_count": handoff["shard_count"],
        "split": handoff["split"],
    }
    report.update(report_updates or {})
    report_path = publication.collection_root / "batch_generation_report.json"
    write_canonical_json(report_path, report)

    evidence: dict[str, object] = {
        "conservation_status": "PROVEN",
        "event_count": 12,
        "instance_digest_sha256": "5" * 64,
        "relative_path": "batch_generation_report.json",
        "sample_count": handoff["sample_count"],
        "semantic_digest_sha256": "6" * 64,
        "sha256": sha256_file(report_path),
        "shard_count": handoff["shard_count"],
    }
    evidence.update(evidence_updates or {})
    handoff.update(
        {
            "artifact_role": "sop07_heldout_collection_complete_handoff",
            "generation_report_evidence": evidence,
            "handoff_version": (
                "sop07_heldout_collection_complete_handoff_v1"
            ),
        }
    )
    handoff.pop("producer_version", None)
    handoff.update(handoff_updates or {})
    handoff["collection_semantic_digest_sha256"] = (
        heldout_collection_semantic_digest(handoff)
    )
    return rewrite_collection_handoff(publication, handoff)


def test_public_contract_is_frozen_and_dynamic_digest_is_canonical() -> None:
    assert RISK_DATASET_LAYOUT_VERSION == "risk_dataset_v2"
    assert RISK_DATASET_FAMILY_LAYOUT_VERSION == "risk_dataset_family_v1"
    assert [field.name for field in fields(RiskShardDescriptor)] == [
        "shard_index",
        "relative_root",
        "sample_count",
        "manifest_digest",
        "semantic_digest",
        "payload_sha256",
        "metadata_sha256",
        "summary_sha256",
    ]
    assert [field.name for field in fields(LoadedRiskDataset)] == [
        "seal_root",
        "collection_root",
        "manifest",
        "grid",
        "shards",
        "split",
        "sample_count",
        "risk_dataset_manifest_digest",
        "provenance",
    ]
    descriptor = RiskShardDescriptor(0, "shard-00000", 1, *("a" * 64,) * 5)
    with pytest.raises(FrozenInstanceError):
        descriptor.sample_count = 2  # type: ignore[misc]

    dynamic_objects = {"human": {"radius_m": 0.3}, "ü": [2, 1]}
    expected = hashlib.sha256(
        json.dumps(
            dynamic_objects,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert canonical_dynamic_objects_digest(dynamic_objects) == expected
    assert canonical_dynamic_objects_digest({"ü": [2, 1], "human": {"radius_m": 0.3}}) == expected
    with pytest.raises((TypeError, ValueError)):
        canonical_dynamic_objects_digest({"bad": float("nan")})


def test_public_manifest_validator_authenticates_complete_semantics(
    tmp_path: Path,
) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    loaded = load_risk_dataset_seal(
        _publish(publication, tmp_path / "seal"),
        collection_root=publication.collection_root,
        expected_split="train",
    )
    assert (
        validate_risk_dataset_manifest(loaded.manifest)
        == loaded.risk_dataset_manifest_digest
    )

    tampered = dict(loaded.manifest)
    grid = dict(loaded.manifest["grid"])
    grid["sample_dt_s"] = float(grid["sample_dt_s"]) + 0.1
    tampered["grid"] = grid
    with pytest.raises(RiskDataContractError, match="manifest digest"):
        validate_risk_dataset_manifest(tampered)

    with pytest.raises(RiskDataContractError, match="mapping"):
        validate_risk_dataset_manifest([])  # type: ignore[arg-type]
    with pytest.raises(RiskDataContractError, match="keys"):
        validate_risk_dataset_manifest({**dict(loaded.manifest), "extra": 1})
    with pytest.raises(RiskDataContractError, match="risk_dataset_manifest_digest"):
        validate_risk_dataset_manifest(
            {**dict(loaded.manifest), "risk_dataset_manifest_digest": "A" * 64}
        )

    nonfinite = dict(loaded.manifest)
    nonfinite_grid = dict(loaded.manifest["grid"])
    nonfinite_grid["sample_dt_s"] = float("nan")
    nonfinite["grid"] = nonfinite_grid
    with pytest.raises(RiskDataContractError, match="finite canonical JSON"):
        validate_risk_dataset_manifest(nonfinite)


def test_publish_load_round_trip_uses_every_real_shard_and_preserves_temporal_safe(
    tmp_path: Path,
) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    seal_root = _publish(publication, tmp_path / "seal")
    loaded = load_risk_dataset_seal(
        seal_root,
        collection_root=publication.collection_root,
        expected_split="train",
    )

    assert isinstance(loaded, LoadedRiskDataset)
    assert loaded.seal_root == seal_root
    assert loaded.collection_root == publication.collection_root
    assert loaded.grid == publication.grid
    assert loaded.sample_count == 12
    assert [shard.shard_index for shard in loaded.shards] == [0, 1]
    assert [shard.relative_root for shard in loaded.shards] == [
        "shard-00000",
        "shard-00001",
    ]
    assert set(path.name for path in seal_root.iterdir()) == {
        "dataset_manifest.json",
        "checksums.sha256",
        ".producer-complete",
    }
    assert len(loaded.provenance["g1_split_manifest_digest"]) == 32
    assert len(loaded.provenance["target_type_policy_digest"]) == 32
    assert len(loaded.provenance["dynamic_objects_config_digest"]) == 64
    assert len(loaded.provenance["risk_dataset_manifest_digest"]) == 64
    assert loaded.risk_dataset_manifest_digest == loaded.provenance[
        "risk_dataset_manifest_digest"
    ]

    samples = tuple(
        sample
        for shard in loaded.shards
        for sample in load_risk_shard(
            publication.collection_root / shard.relative_root,
            grid=loaded.grid,
        ).samples
    )
    temporal_safe = [sample for sample in samples if sample.event_type == "temporal_safe"]
    assert temporal_safe
    assert all(sample.collision_label == 0 for sample in temporal_safe)
    assert all(sample.near_miss == 1 for sample in temporal_safe)


def test_manifest_digest_ignores_absolute_roots_and_runtime_metadata(
    tmp_path: Path,
) -> None:
    source = create_formal_risk_publication(
        tmp_path / "source",
        runtime_metadata={"created_at": "2026-07-19T01:02:03Z", "host": "node-a"},
    )
    clone_root = tmp_path / "different" / "absolute" / "source"
    clone_root.parent.mkdir(parents=True)
    shutil.copytree(source.root, clone_root)
    clone = replace(
        source,
        root=clone_root,
        collection_root=clone_root / "collection",
        base_config_path=clone_root / "config" / "base.yaml",
        split_provenance_path=clone_root / "sop03" / "run_manifest.json",
        handoff_path=clone_root / "collection" / "collection_complete_handoff.json",
    )
    handoff = json.loads(clone.handoff_path.read_text(encoding="utf-8"))
    handoff["runtime_metadata"] = {
        "created_at": "2037-01-01T00:00:00Z",
        "host": "node-z",
        "slurm_job_id": "999999",
    }
    handoff["collection_instance_digest_sha256"] = "f" * 64
    clone = replace(
        clone,
        handoff_sha256=rewrite_collection_handoff(clone, handoff),
    )

    first = load_risk_dataset_seal(
        _publish(source, tmp_path / "seal-a"),
        collection_root=source.collection_root,
        expected_split="train",
    )
    second = load_risk_dataset_seal(
        _publish(clone, tmp_path / "elsewhere" / "seal-b"),
        collection_root=clone.collection_root,
        expected_split="train",
    )

    assert first.manifest["collection_handoff_sha256"] != second.manifest[
        "collection_handoff_sha256"
    ]
    assert first.risk_dataset_manifest_digest == second.risk_dataset_manifest_digest
    assert first.manifest["shards"] == second.manifest["shards"]
    serialized = canonical_json(first.manifest)
    assert str(source.root) not in serialized
    assert str(first.seal_root) not in serialized


def test_publisher_authenticates_handoff_before_parsing_collection(
    tmp_path: Path,
) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    publication.handoff_path.write_bytes(publication.handoff_path.read_bytes() + b" ")

    with pytest.raises(RiskDataContractError, match="handoff SHA-256"):
        _publish(publication, tmp_path / "seal")
    assert not (tmp_path / "seal").exists()


@pytest.mark.parametrize("split", ["calibration", "val", "test"])
def test_heldout_publisher_authenticates_report_and_normalizes_producer(
    tmp_path: Path,
    split: str,
) -> None:
    publication = create_formal_risk_publication(
        tmp_path / "upstream",
        split=split,
    )
    handoff_sha256 = _rewrite_as_heldout_collection(publication)

    seal_root = _publish(
        publication,
        tmp_path / "seal",
        expected_split=split,
        expected_handoff_sha256=handoff_sha256,
    )
    manifest = _manifest(seal_root)
    assert manifest["collection_handoff_version"] == (
        "sop07_heldout_collection_complete_handoff_v1"
    )
    assert manifest["collection_artifact_role"] == (
        "sop07_heldout_collection_complete_handoff"
    )
    assert manifest["collection_producer_version"] == "sop07_risk_dataset_cli_v3"
    loaded = load_risk_dataset_seal(
        seal_root,
        collection_root=publication.collection_root,
        expected_split=split,
    )
    assert loaded.manifest["collection_producer_version"] == (
        "sop07_risk_dataset_cli_v3"
    )


def test_heldout_publisher_requires_framed_producer_collection_semantics(
    tmp_path: Path,
) -> None:
    publication = create_formal_risk_publication(
        tmp_path / "upstream",
        split="val",
        handoff_dialect="heldout",
    )
    handoff = json.loads(publication.handoff_path.read_text(encoding="utf-8"))
    assert handoff["collection_semantic_digest_sha256"] == (
        heldout_collection_semantic_digest(handoff)
    )

    seal_root = _publish(
        publication,
        tmp_path / "seal",
        expected_split="val",
    )
    assert _manifest(seal_root)["collection_semantic_digest_sha256"] == (
        handoff["collection_semantic_digest_sha256"]
    )


def test_train_collection_semantic_digest_is_authenticated_opaque_identity(
    tmp_path: Path,
) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    handoff = json.loads(publication.handoff_path.read_text(encoding="utf-8"))
    handoff["collection_semantic_digest_sha256"] = "a" * 64
    handoff_sha256 = rewrite_collection_handoff(publication, handoff)

    seal_root = _publish(
        publication,
        tmp_path / "seal",
        expected_handoff_sha256=handoff_sha256,
    )
    loaded = load_risk_dataset_seal(
        seal_root,
        collection_root=publication.collection_root,
        expected_split="train",
    )
    assert loaded.manifest["collection_semantic_digest_sha256"] == "a" * 64


def test_train_opaque_digest_does_not_weaken_handoff_or_shard_authentication(
    tmp_path: Path,
) -> None:
    stale_sha = create_formal_risk_publication(tmp_path / "stale-sha")
    handoff = json.loads(stale_sha.handoff_path.read_text(encoding="utf-8"))
    handoff["collection_semantic_digest_sha256"] = "a" * 64
    rewrite_collection_handoff(stale_sha, handoff)
    with pytest.raises(RiskDataContractError, match="handoff SHA-256"):
        _publish(stale_sha, tmp_path / "stale-sha-seal")

    bad_shard = create_formal_risk_publication(tmp_path / "bad-shard")
    handoff = json.loads(bad_shard.handoff_path.read_text(encoding="utf-8"))
    handoff["collection_semantic_digest_sha256"] = "a" * 64
    handoff["shards"][0]["semantic_digest"] = "f" * 64
    handoff_sha256 = rewrite_collection_handoff(bad_shard, handoff)
    with pytest.raises(RiskDataContractError, match="shard semantic digest"):
        _publish(
            bad_shard,
            tmp_path / "bad-shard-seal",
            expected_handoff_sha256=handoff_sha256,
        )


def test_collection_handoff_dialects_are_split_specific(tmp_path: Path) -> None:
    legacy_val = create_formal_risk_publication(
        tmp_path / "legacy-val",
        split="val",
    )
    with pytest.raises(RiskDataContractError, match="handoff dialect"):
        _publish(
            legacy_val,
            tmp_path / "legacy-val-seal",
            expected_split="val",
        )

    heldout_train = create_formal_risk_publication(tmp_path / "heldout-train")
    heldout_sha256 = _rewrite_as_heldout_collection(heldout_train)
    with pytest.raises(RiskDataContractError, match="handoff dialect"):
        _publish(
            heldout_train,
            tmp_path / "heldout-train-seal",
            expected_handoff_sha256=heldout_sha256,
        )


@pytest.mark.parametrize("failure", ["missing", "tampered", "unsafe_path"])
def test_heldout_generation_report_publication_fails_closed(
    tmp_path: Path,
    failure: str,
) -> None:
    publication = create_formal_risk_publication(
        tmp_path / "upstream",
        split="val",
    )
    evidence_updates = (
        {"relative_path": "nested/batch_generation_report.json"}
        if failure == "unsafe_path"
        else None
    )
    handoff_sha256 = _rewrite_as_heldout_collection(
        publication,
        evidence_updates=evidence_updates,
    )
    report_path = publication.collection_root / "batch_generation_report.json"
    if failure == "missing":
        report_path.unlink()
    elif failure == "tampered":
        report_path.write_bytes(report_path.read_bytes() + b" ")

    with pytest.raises(RiskDataContractError, match="heldout generation report"):
        _publish(
            publication,
            tmp_path / "seal",
            expected_split="val",
            expected_handoff_sha256=handoff_sha256,
        )
    assert not (tmp_path / "seal").exists()


@pytest.mark.parametrize(
    ("report_updates", "evidence_updates"),
    [
        ({"generation_state": "partial"}, None),
        ({"schema_version": "2.0.0"}, None),
        ({"layout_version": "risk_shard_npz_jsonl_v1"}, None),
        ({"split": "test"}, None),
        ({"code_commit": "4" * 40}, None),
        ({"sample_count": 13}, None),
        ({"shard_count": 3}, None),
        ({"event_count": 13}, None),
        ({"batch_generation_semantic_digest_sha256": "7" * 64}, None),
        ({"batch_generation_instance_digest_sha256": "8" * 64}, None),
        ({"conservation": {"status": "FAILED"}}, None),
        (None, {"sample_count": 13}),
    ],
)
def test_heldout_generation_report_identity_mismatch_fails_closed(
    tmp_path: Path,
    report_updates: dict[str, object] | None,
    evidence_updates: dict[str, object] | None,
) -> None:
    publication = create_formal_risk_publication(
        tmp_path / "upstream",
        split="calibration",
    )
    handoff_sha256 = _rewrite_as_heldout_collection(
        publication,
        report_updates=report_updates,
        evidence_updates=evidence_updates,
    )
    with pytest.raises(
        RiskDataContractError,
        match="heldout generation report",
    ):
        _publish(
            publication,
            tmp_path / "seal",
            expected_split="calibration",
            expected_handoff_sha256=handoff_sha256,
        )
    assert not (tmp_path / "seal").exists()


@pytest.mark.parametrize(
    ("document", "old", "new"),
    [
        (
            "handoff",
            b'"global_sample_id_uniqueness":"PROVEN"',
            (
                b'"global_sample_id_uniqueness":"PROVEN",'
                b'"global_sample_id_uniqueness":"PROVEN"'
            ),
        ),
        (
            "handoff",
            b'"sample_count":12,"schema_version"',
            b'"sample_count":1e400,"schema_version"',
        ),
        (
            "report",
            b'"conservation":{"status":"PROVEN"}',
            (
                b'"conservation":{"status":"PROVEN",'
                b'"status":"PROVEN"}'
            ),
        ),
        ("report", b'"event_count":12', b'"event_count":1e400'),
    ],
)
def test_handoff_and_report_reject_recursive_duplicates_and_overflow_numbers(
    tmp_path: Path,
    document: str,
    old: bytes,
    new: bytes,
) -> None:
    publication = create_formal_risk_publication(
        tmp_path / "upstream",
        split="test",
    )
    handoff_sha256 = _rewrite_as_heldout_collection(publication)
    if document == "handoff":
        raw = publication.handoff_path.read_bytes()
        assert raw.count(old) == 1
        mutated = raw.replace(old, new, 1)
        publication.handoff_path.write_bytes(mutated)
        handoff_sha256 = hashlib.sha256(mutated).hexdigest()
    else:
        report_path = publication.collection_root / "batch_generation_report.json"
        raw = report_path.read_bytes()
        assert raw.count(old) == 1
        mutated = raw.replace(old, new, 1)
        report_path.write_bytes(mutated)
        handoff = json.loads(
            publication.handoff_path.read_text(encoding="utf-8")
        )
        handoff["generation_report_evidence"]["sha256"] = hashlib.sha256(
            mutated
        ).hexdigest()
        handoff_sha256 = rewrite_collection_handoff(publication, handoff)

    with pytest.raises(RiskDataContractError, match="strict finite JSON"):
        _publish(
            publication,
            tmp_path / "seal",
            expected_split="test",
            expected_handoff_sha256=handoff_sha256,
        )
    assert not (tmp_path / "seal").exists()


@pytest.mark.parametrize("mutation", ["semantic_digest", "ordered_indices"])
def test_publisher_rejects_handoff_shard_descriptor_mutation(
    tmp_path: Path, mutation: str
) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    handoff = json.loads(publication.handoff_path.read_text(encoding="utf-8"))
    shards = handoff["shards"]
    assert isinstance(shards, list)
    if mutation == "semantic_digest":
        shards[0]["semantic_digest"] = "f" * 64
    else:
        shards.reverse()
    handoff_sha = rewrite_collection_handoff(publication, handoff)

    with pytest.raises(RiskDataContractError, match="shard|ordered|semantic"):
        _publish(
            publication,
            tmp_path / "seal",
            expected_handoff_sha256=handoff_sha,
        )
    assert not (tmp_path / "seal").exists()


def test_publisher_rejects_wrong_split_and_inconsistent_dynamic_snapshot(
    tmp_path: Path,
) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    with pytest.raises(RiskDataContractError, match="split"):
        _publish(publication, tmp_path / "wrong-split", expected_split="validation")

    manifest = json.loads(
        publication.split_provenance_path.read_text(encoding="utf-8")
    )
    manifest["producer_protocol"]["config_snapshots"]["base"]["value"][
        "dynamic_objects"
    ]["human"]["radius_m"] = 99.0
    write_canonical_json(publication.split_provenance_path, manifest)
    _republish_sop03_envelope(publication)
    with pytest.raises(RiskDataContractError, match="base config snapshot"):
        _publish(publication, tmp_path / "wrong-dynamic")


@pytest.mark.parametrize(
    ("source_field", "bad_value", "message"),
    [
        ("g1", "a" * 64, "g1_split_manifest_digest"),
        ("target", "b" * 64, "target_type_policy_digest"),
    ],
)
def test_publisher_applies_field_specific_upstream_digest_rules(
    tmp_path: Path,
    source_field: str,
    bad_value: str,
    message: str,
) -> None:
    publication = create_formal_risk_publication(
        tmp_path / "upstream",
        g1_split_manifest_digest=(bad_value if source_field == "g1" else "1" * 32),
        target_type_policy_digest=(
            bad_value if source_field == "target" else "2" * 32
        ),
    )
    with pytest.raises(RiskDataContractError, match=message):
        _publish(publication, tmp_path / "seal")


@pytest.mark.parametrize(
    ("path", "bad_value", "message"),
    [
        (("split",), "validation", "split"),
        (("grid", "height"), 5, "grid|shape"),
        (
            ("channel_spec", "history"),
            list(reversed(HISTORY_CHANNELS)),
            "channel",
        ),
        (("g1_split_manifest_digest",), "a" * 64, "g1_split_manifest_digest"),
        (
            ("target_type_policy_digest",),
            "b" * 64,
            "target_type_policy_digest",
        ),
        (
            ("dynamic_objects_config_digest",),
            "c" * 32,
            "dynamic_objects_config_digest",
        ),
    ],
)
def test_loader_rejects_manifest_semantic_and_provenance_mutations(
    tmp_path: Path,
    path: tuple[str, ...],
    bad_value: object,
    message: str,
) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    seal_root = _publish(publication, tmp_path / "seal")
    _mutate_manifest(seal_root, path, bad_value)

    with pytest.raises(RiskDataContractError, match=message):
        load_risk_dataset_seal(
            seal_root,
            collection_root=publication.collection_root,
            expected_split="train",
        )


def test_loader_rejects_channel_order_even_when_lengths_are_unchanged(
    tmp_path: Path,
) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    seal_root = _publish(publication, tmp_path / "seal")
    manifest = _manifest(seal_root)
    channel_spec = manifest["channel_spec"]
    assert isinstance(channel_spec, dict)
    assert channel_spec == {
        "history": list(HISTORY_CHANNELS),
        "state": list(STATE_CHANNELS),
        "trajectory": list(TRAJECTORY_CHANNELS),
        "flat": list(INPUT_CHANNELS),
        "targets": [
            "collision_label",
            "risk_severity",
            "min_clearance",
            "near_miss",
            "first_collision_time",
        ],
    }
    state = channel_spec["state"]
    assert isinstance(state, list)
    state[0], state[1] = state[1], state[0]
    write_canonical_json(seal_root / "dataset_manifest.json", manifest)
    resign_dataset_seal(seal_root)

    with pytest.raises(RiskDataContractError, match="channel"):
        load_risk_dataset_seal(
            seal_root,
            collection_root=publication.collection_root,
            expected_split="train",
        )


def test_load_rejects_expected_manifest_digest_mismatch(tmp_path: Path) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    seal_root = _publish(publication, tmp_path / "seal")
    with pytest.raises(RiskDataContractError, match="expected manifest digest"):
        load_risk_dataset_seal(
            seal_root,
            collection_root=publication.collection_root,
            expected_split="train",
            expected_manifest_digest="f" * 64,
        )


def test_publisher_rejects_overwrite_and_collection_symlinks(tmp_path: Path) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    existing = tmp_path / "existing-seal"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        _publish(publication, existing)

    collection_link = tmp_path / "collection-link"
    collection_link.symlink_to(publication.collection_root, target_is_directory=True)
    linked = replace(publication, collection_root=collection_link)
    with pytest.raises(RiskDataContractError, match="symlink"):
        _publish(linked, tmp_path / "linked-seal")


def test_publisher_rejects_symlinked_shard_before_formal_loading(
    tmp_path: Path,
) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    shard = publication.collection_root / "shard-00001"
    real_shard = publication.root / "detached-shard-00001"
    shard.rename(real_shard)
    shard.symlink_to(real_shard, target_is_directory=True)

    with pytest.raises(RiskDataContractError, match="symlink"):
        _publish(publication, tmp_path / "seal")


@pytest.mark.parametrize("mutation", ["missing_marker", "unexpected_file", "symlink"])
def test_loader_rejects_partial_unexpected_and_symlinked_seals(
    tmp_path: Path, mutation: str
) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    seal_root = _publish(publication, tmp_path / "seal")
    if mutation == "missing_marker":
        (seal_root / ".producer-complete").unlink()
    elif mutation == "unexpected_file":
        (seal_root / "orphan.tmp").write_text("partial", encoding="utf-8")
    else:
        manifest = seal_root / "dataset_manifest.json"
        detached = tmp_path / "detached-manifest.json"
        manifest.rename(detached)
        manifest.symlink_to(detached)

    with pytest.raises(RiskDataContractError, match="incomplete|unexpected|symlink"):
        load_risk_dataset_seal(
            seal_root,
            collection_root=publication.collection_root,
            expected_split="train",
        )


def test_loader_rejects_v1_layout_and_shard_root_as_dataset_seal(
    tmp_path: Path,
) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    seal_root = _publish(publication, tmp_path / "seal")
    _mutate_manifest(
        seal_root,
        ("dataset_layout_version",),
        "risk_shard_npz_jsonl_v1",
    )
    with pytest.raises(RiskDataContractError, match="risk_dataset_v2|unsupported"):
        load_risk_dataset_seal(
            seal_root,
            collection_root=publication.collection_root,
            expected_split="train",
        )
    with pytest.raises(RiskDataContractError, match="dataset|seal|incomplete"):
        load_risk_dataset_seal(
            publication.collection_root / "shard-00000",
            collection_root=publication.collection_root,
            expected_split="train",
        )


def test_publisher_formally_loads_every_shard_before_atomic_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    real_loader = seal_module.load_risk_shard
    calls: list[str] = []

    def fail_on_second_shard(root: str | Path, **kwargs: object) -> object:
        name = Path(root).name
        calls.append(name)
        if name == "shard-00001":
            raise ValueError("formal second-shard sentinel")
        return real_loader(root, **kwargs)

    monkeypatch.setattr(seal_module, "load_risk_shard", fail_on_second_shard)
    output = tmp_path / "seal"
    with pytest.raises(RiskDataContractError, match="formal second-shard sentinel"):
        _publish(publication, output)

    assert calls == ["shard-00000", "shard-00001"]
    assert not output.exists()
    assert not list(tmp_path.glob(".seal.staging-*"))


def test_loader_rejects_post_publication_shard_file_tamper(tmp_path: Path) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    seal_root = _publish(publication, tmp_path / "seal")
    summary = publication.collection_root / "shard-00001" / "summary.json"
    summary.write_bytes(summary.read_bytes() + b" ")

    with pytest.raises(RiskDataContractError, match="checksum|summary"):
        load_risk_dataset_seal(
            seal_root,
            collection_root=publication.collection_root,
            expected_split="train",
        )


def test_production_loader_delegates_same_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = object()
    observed: dict[str, object] = {}

    def fake_loader(
        seal_root: str | Path,
        *,
        collection_root: str | Path,
        expected_split: str,
        expected_manifest_digest: str | None = None,
    ) -> object:
        observed.update(
            {
                "seal_root": seal_root,
                "collection_root": collection_root,
                "expected_split": expected_split,
                "expected_manifest_digest": expected_manifest_digest,
            }
        )
        return sentinel

    monkeypatch.setattr(seal_module, "load_risk_dataset_seal", fake_loader)
    result = load_production_risk_dataset(
        tmp_path / "seal",
        collection_root=tmp_path / "collection",
        expected_split="train",
        expected_manifest_digest="a" * 64,
    )

    assert result is sentinel
    assert observed == {
        "seal_root": tmp_path / "seal",
        "collection_root": tmp_path / "collection",
        "expected_split": "train",
        "expected_manifest_digest": "a" * 64,
    }


def test_cli_success_argument_failure_and_overwrite_failure(tmp_path: Path) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    seal_root = tmp_path / "cli-seal"
    command = [
        sys.executable,
        str(SCRIPT),
        "--collection-root",
        str(publication.collection_root),
        "--output-dir",
        str(seal_root),
        "--base-config",
        str(publication.base_config_path),
        "--split-provenance",
        str(publication.split_provenance_path),
        "--split",
        "train",
        "--expected-collection-handoff-sha256",
        publication.handoff_sha256,
    ]
    success = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert success.returncode == 0, success.stderr
    payload = json.loads(success.stdout)
    assert payload["status"] == "complete"
    assert payload["sample_count"] == 12
    assert payload["risk_dataset_manifest_digest"] == _manifest(seal_root)[
        "risk_dataset_manifest_digest"
    ]

    overwrite = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert overwrite.returncode == 2
    assert "overwrite" in overwrite.stderr

    bad_argument = subprocess.run(
        [*command[:-1], "not-a-sha256"], cwd=ROOT, text=True, capture_output=True
    )
    assert bad_argument.returncode == 2
    assert "64 lowercase" in bad_argument.stderr


def test_cli_can_seal_an_authenticated_occupancy_sidecar_collection(
    tmp_path: Path,
) -> None:
    publication = create_formal_risk_publication(
        tmp_path / "upstream",
        history_steps=8,
        future_steps=15,
    )
    sidecars = create_formal_risk_sidecar_publication(
        publication,
        tmp_path / "sidecars",
    )
    seal_root = tmp_path / "occupancy-seal"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--collection-root",
            str(publication.collection_root),
            "--sidecar-root",
            str(sidecars.sidecar_root),
            "--output-dir",
            str(seal_root),
            "--base-config",
            str(publication.base_config_path),
            "--split-provenance",
            str(publication.split_provenance_path),
            "--split",
            "train",
            "--expected-collection-handoff-sha256",
            publication.handoff_sha256,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    manifest = _manifest(seal_root)
    assert payload["occupancy_sidecar_collection_digest_sha256"] == (
        manifest["occupancy_sidecars"]["collection_digest_sha256"]
    )


def test_sidecar_seal_rejects_missing_extra_and_reordered_evidence(
    tmp_path: Path,
) -> None:
    publication = create_formal_risk_publication(
        tmp_path / "upstream",
        history_steps=8,
        future_steps=15,
    )
    sidecars = create_formal_risk_sidecar_publication(
        publication,
        tmp_path / "sidecars",
    )
    seal_root = _publish(
        publication,
        tmp_path / "seal",
        sidecar_root=sidecars.sidecar_root,
    )
    loaded = load_risk_dataset_seal(
        seal_root,
        collection_root=publication.collection_root,
        expected_split="train",
        sidecar_root=sidecars.sidecar_root,
    )

    extra = sidecars.sidecar_root / "unexpected.txt"
    extra.write_text("not admitted", encoding="utf-8")
    with pytest.raises(RiskDataContractError, match="missing/extra"):
        load_occupancy_sidecar_collection(
            loaded,
            sidecar_root=sidecars.sidecar_root,
        )
    extra.unlink()

    marker = sidecars.sidecar_root / (
        "shard-00001.risk-sidecar-pair-complete.json"
    )
    marker.rename(marker.with_suffix(".missing"))
    with pytest.raises(RiskDataContractError, match="missing/extra"):
        load_occupancy_sidecar_collection(
            loaded,
            sidecar_root=sidecars.sidecar_root,
        )
    marker.with_suffix(".missing").rename(marker)

    manifest = _manifest(seal_root)
    sidecar_section = manifest["occupancy_sidecars"]
    sidecar_section["shards"] = list(reversed(sidecar_section["shards"]))
    write_canonical_json(seal_root / "dataset_manifest.json", manifest)
    resign_dataset_seal(seal_root)
    with pytest.raises(RiskDataContractError, match="descriptor|digest|contiguous"):
        load_risk_dataset_seal(
            seal_root,
            collection_root=publication.collection_root,
            expected_split="train",
            sidecar_root=sidecars.sidecar_root,
        )


def test_fixture_is_two_contiguous_formal_shards_with_twelve_samples(
    tmp_path: Path,
) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    handoff = json.loads(publication.handoff_path.read_text(encoding="utf-8"))
    assert handoff["handoff_version"] == "sop07_collection_complete_handoff_v1"
    assert handoff["layout_version"] == "risk_shard_npz_jsonl_v2"
    assert handoff["sample_count"] == 12
    assert handoff["shard_count"] == 2
    assert [item["shard_index"] for item in handoff["shards"]] == [0, 1]
    assert [item["relative_root"] for item in handoff["shards"]] == [
        "shard-00000",
        "shard-00001",
    ]
    assert sha256_file(publication.handoff_path) == publication.handoff_sha256


def test_grid_type_remains_the_existing_contract(tmp_path: Path) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    loaded = load_risk_dataset_seal(
        _publish(publication, tmp_path / "seal"),
        collection_root=publication.collection_root,
        expected_split="train",
    )
    assert type(loaded.grid) is GridSpec


def test_publisher_recomputes_collection_semantic_digest_after_formal_load(
    tmp_path: Path,
) -> None:
    publication = create_formal_risk_publication(
        tmp_path / "upstream",
        split="val",
        handoff_dialect="heldout",
    )
    handoff = json.loads(publication.handoff_path.read_text(encoding="utf-8"))
    handoff["collection_semantic_digest_sha256"] = "0" * 64
    resigned_handoff_sha256 = rewrite_collection_handoff(publication, handoff)

    with pytest.raises(RiskDataContractError, match="collection semantic digest"):
        _publish(
            publication,
            tmp_path / "seal",
            expected_split="val",
            expected_handoff_sha256=resigned_handoff_sha256,
        )
    assert not (tmp_path / "seal").exists()


def test_loaded_dataset_manifest_and_provenance_are_deeply_immutable(
    tmp_path: Path,
) -> None:
    publication = create_formal_risk_publication(tmp_path / "upstream")
    loaded = load_risk_dataset_seal(
        _publish(publication, tmp_path / "seal"),
        collection_root=publication.collection_root,
        expected_split="train",
    )
    scalar_digest = loaded.risk_dataset_manifest_digest
    provenance_digest = loaded.provenance["risk_dataset_manifest_digest"]
    manifest_digest = loaded.manifest["risk_dataset_manifest_digest"]
    assert isinstance(loaded.manifest, dict)
    assert isinstance(loaded.provenance, dict)

    with pytest.raises(TypeError):
        loaded.provenance["risk_dataset_manifest_digest"] = "f" * 64
    grid = loaded.manifest["grid"]
    assert isinstance(grid, dict)
    with pytest.raises(TypeError):
        grid["height"] = 999
    shards = loaded.manifest["shards"]
    assert isinstance(shards, tuple)
    first_shard = shards[0]
    assert isinstance(first_shard, dict)
    with pytest.raises(TypeError):
        first_shard["semantic_digest"] = "f" * 64

    for mutate in (
        lambda value: value.__delitem__("g1_split_manifest_digest"),
        lambda value: value.clear(),
        lambda value: value.pop("g1_split_manifest_digest"),
        lambda value: value.popitem(),
        lambda value: value.setdefault("new", "value"),
        lambda value: value.update({"new": "value"}),
        lambda value: value.__ior__({"new": "value"}),
    ):
        with pytest.raises(TypeError):
            mutate(loaded.provenance)

    assert loaded.risk_dataset_manifest_digest == scalar_digest
    assert loaded.provenance["risk_dataset_manifest_digest"] == provenance_digest
    assert loaded.manifest["risk_dataset_manifest_digest"] == manifest_digest


def test_atomic_directory_rename_noreplace_preserves_both_existing_roots(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    destination.mkdir()
    (staging / "payload").write_text("staged", encoding="utf-8")

    with pytest.raises(FileExistsError):
        seal_module._atomic_rename_directory_noreplace(staging, destination)

    assert staging.is_dir()
    assert (staging / "payload").read_text(encoding="utf-8") == "staged"
    assert destination.is_dir()
    assert not list(destination.iterdir())


def test_relative_external_paths_are_anchored_before_cwd_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    publication = create_formal_risk_publication(workspace / "upstream")
    monkeypatch.chdir(workspace)
    published = publish_risk_dataset_seal(
        "seal",
        collection_root="upstream/collection",
        base_config_path="upstream/config/base.yaml",
        split_provenance_path="upstream/sop03/run_manifest.json",
        expected_split="train",
        expected_collection_handoff_sha256=publication.handoff_sha256,
    )
    loaded = load_risk_dataset_seal(
        "seal",
        collection_root="upstream/collection",
        expected_split="train",
    )
    expected_seal_root = workspace / "seal"
    expected_collection_root = workspace / "upstream" / "collection"

    monkeypatch.chdir(tmp_path)
    assert published == expected_seal_root
    assert published.is_absolute()
    assert loaded.seal_root == expected_seal_root
    assert loaded.seal_root.is_absolute()
    assert loaded.collection_root == expected_collection_root
    assert loaded.collection_root.is_absolute()
    for descriptor in loaded.shards:
        shard_root = loaded.collection_root / descriptor.relative_root
        assert shard_root.is_dir()
        assert (shard_root / "summary.json").is_file()
