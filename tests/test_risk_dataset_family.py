"""Authenticated four-split publication tests for ``risk_dataset_family_v1``."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import pytest

from src.datasets.risk_dataloader import RiskDataContractError
import src.datasets.risk_dataset_seal as seal_module
from src.datasets.risk_dataset_seal import (
    RISK_DATASET_FAMILY_LAYOUT_VERSION,
    LoadedRiskDataset,
    LoadedRiskDatasetFamily,
    load_risk_dataset_family,
    load_risk_dataset_seal,
    publish_risk_dataset_family,
    publish_risk_dataset_seal,
)
from tests.fixtures.formal_risk_publication import (
    FormalRiskPublication,
    canonical_json,
    create_formal_risk_publication,
    sha256_file,
    write_canonical_json,
)


MEMBER_ORDER = ("train", "calibration", "val", "test")
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "04_seal_risk_dataset.py"


def _publish_members(
    root: Path,
    *,
    member_options: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[dict[str, LoadedRiskDataset], dict[str, FormalRiskPublication]]:
    members: dict[str, LoadedRiskDataset] = {}
    publications: dict[str, FormalRiskPublication] = {}
    options = member_options or {}
    for split in MEMBER_ORDER:
        publication = create_formal_risk_publication(
            root / f"upstream-{split}",
            split=split,
            handoff_dialect="legacy" if split == "train" else "heldout",
            **dict(options.get(split, {})),
        )
        seal_root = publish_risk_dataset_seal(
            root / f"seal-{split}",
            collection_root=publication.collection_root,
            base_config_path=publication.base_config_path,
            split_provenance_path=publication.split_provenance_path,
            expected_split=split,
            expected_collection_handoff_sha256=publication.handoff_sha256,
        )
        publications[split] = publication
        members[split] = load_risk_dataset_seal(
            seal_root,
            collection_root=publication.collection_root,
            expected_split=split,
        )
    return members, publications


def _member_digests(
    members: Mapping[str, LoadedRiskDataset],
) -> dict[str, str]:
    return {
        split: members[split].risk_dataset_manifest_digest
        for split in MEMBER_ORDER
    }


def _family_manifest(root: Path) -> dict[str, object]:
    value = json.loads(
        (root / "family_manifest.json").read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def _family_digest(manifest: Mapping[str, object]) -> str:
    projection = {
        key: value
        for key, value in manifest.items()
        if key != "risk_dataset_family_digest"
    }
    return hashlib.sha256(canonical_json(projection).encode("utf-8")).hexdigest()


def _resign_family(root: Path) -> None:
    entries = {
        ".producer-complete": sha256_file(root / ".producer-complete"),
        "family_manifest.json": sha256_file(root / "family_manifest.json"),
    }
    (root / "checksums.sha256").write_text(
        "".join(
            f"{entries[relative]}  {relative}\n" for relative in sorted(entries)
        ),
        encoding="utf-8",
    )


def _load_seal_cli_module() -> object:
    spec = importlib.util.spec_from_file_location("risk_dataset_seal_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_family_public_contract_round_trip_and_deep_immutability(
    tmp_path: Path,
) -> None:
    assert RISK_DATASET_FAMILY_LAYOUT_VERSION == "risk_dataset_family_v1"
    assert [field.name for field in fields(LoadedRiskDatasetFamily)] == [
        "root",
        "manifest",
        "members",
        "cross_split_audit",
        "risk_dataset_family_digest",
    ]
    members, _ = _publish_members(tmp_path / "source")

    root = publish_risk_dataset_family(tmp_path / "family", members=members)
    loaded = load_risk_dataset_family(
        root,
        expected_member_digests=_member_digests(members),
    )

    assert loaded.root == root
    assert tuple(loaded.manifest["member_order"]) == MEMBER_ORDER
    assert set(loaded.manifest) == {
        "dataset_family_layout_version",
        "schema_version",
        "member_order",
        "members",
        "common_contract",
        "cross_split_audit",
        "risk_dataset_family_digest",
    }
    assert tuple(loaded.members) == MEMBER_ORDER
    assert set(path.name for path in root.iterdir()) == {
        "family_manifest.json",
        "checksums.sha256",
        ".producer-complete",
    }
    assert len(loaded.risk_dataset_family_digest) == 64
    for split in MEMBER_ORDER:
        descriptor = loaded.members[split]
        assert descriptor == {
            "split": split,
            "risk_dataset_manifest_digest": (
                members[split].risk_dataset_manifest_digest
            ),
            "sample_count": 12,
            "shard_count": 2,
        }

    with pytest.raises(FrozenInstanceError):
        loaded.root = tmp_path / "forged"  # type: ignore[misc]
    with pytest.raises(TypeError, match="frozen"):
        loaded.manifest["schema_version"] = "forged"
    with pytest.raises(TypeError, match="frozen"):
        loaded.members["train"]["sample_count"] = 999


def test_family_allows_and_reports_base_and_source_session_overlap(
    tmp_path: Path,
) -> None:
    members, _ = _publish_members(tmp_path / "source")
    family = load_risk_dataset_family(
        publish_risk_dataset_family(tmp_path / "family", members=members)
    )

    audit = family.cross_split_audit
    assert audit["evidence_completeness"] == "PROVEN"
    assert audit["global_sample_id_uniqueness"] == "PROVEN"
    assert audit["global_cross_split_leakage"] == "PROVEN"
    allowed = audit["allowed_overlaps"]
    assert allowed["base_session_id"]["overlap_values"] == (
        "formal-base-session",
    )
    assert allowed["base_session_id"]["overlap_count"] == 1
    assert allowed["source_session_id"]["overlap_values"] == (
        "formal-source-session",
    )
    assert allowed["source_session_id"]["overlap_count"] == 1


@pytest.mark.parametrize(
    ("identity_prefixes", "overlap_field"),
    [
        ({"sample_id": "train-formal"}, "sample_id"),
        (
            {"base_recording_id": "train-base-recording"},
            "base_recording_id",
        ),
        (
            {"source_recording_id": "train-source-recording"},
            "source_recording_id",
        ),
        (
            {"source_recording_id": "train-base-recording"},
            "base_source_cross_role_recording_id",
        ),
        (
            {"source_snippet_id": "train-source-snippet"},
            "source_snippet_id",
        ),
        ({"pair_group_id": "train-pair-group"}, "pair_group_id"),
        ({"seed_namespace": "sop07/train/formal"}, "seed_namespace"),
    ],
)
def test_family_rejects_each_forbidden_cross_split_identity_overlap(
    tmp_path: Path,
    identity_prefixes: Mapping[str, str],
    overlap_field: str,
) -> None:
    members, _ = _publish_members(
        tmp_path / "source",
        member_options={
            "calibration": {"identity_prefixes": identity_prefixes}
        },
    )

    with pytest.raises(RiskDataContractError, match=overlap_field):
        publish_risk_dataset_family(tmp_path / "family", members=members)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_family_requires_exact_member_splits(
    tmp_path: Path,
    mutation: str,
) -> None:
    members, _ = _publish_members(tmp_path / "source")
    if mutation == "missing":
        members.pop("test")
    else:
        members["validation"] = members["val"]

    with pytest.raises(RiskDataContractError, match="exactly"):
        publish_risk_dataset_family(tmp_path / "family", members=members)


@pytest.mark.parametrize(
    ("member_options", "mismatch_field"),
    [
        ({"g1_split_manifest_digest": "a" * 32}, "g1_split_manifest_digest"),
        ({"history_steps": 3}, "grid"),
        ({"dynamic_human_radius_m": 0.31}, "dynamic_objects_config_digest"),
        ({"target_type_policy_digest": "b" * 32}, "target_type_policy_digest"),
    ],
)
def test_family_rejects_common_contract_mismatch(
    tmp_path: Path,
    member_options: Mapping[str, object],
    mismatch_field: str,
) -> None:
    members, _ = _publish_members(
        tmp_path / "source",
        member_options={"test": member_options},
    )

    with pytest.raises(RiskDataContractError, match=mismatch_field):
        publish_risk_dataset_family(tmp_path / "family", members=members)


def test_cli_has_explicit_family_publication_and_validation_paths(
    tmp_path: Path,
) -> None:
    members, _ = _publish_members(tmp_path / "source")
    family_root = tmp_path / "cli-family"
    publish_command = [
        sys.executable,
        str(SCRIPT),
        "family-publish",
        "--output-dir",
        str(family_root),
    ]
    for split in MEMBER_ORDER:
        publish_command.extend(
            [
                "--member",
                split,
                str(members[split].seal_root),
                str(members[split].collection_root),
                members[split].risk_dataset_manifest_digest,
            ]
        )
    published = subprocess.run(
        publish_command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert published.returncode == 0, published.stderr
    publication_payload = json.loads(published.stdout)
    assert publication_payload["status"] == "complete"
    assert publication_payload["member_order"] == list(MEMBER_ORDER)
    assert len(publication_payload["risk_dataset_family_digest"]) == 64

    validate_command = [
        sys.executable,
        str(SCRIPT),
        "family-validate",
        "--family-root",
        str(family_root),
    ]
    for split, digest in _member_digests(members).items():
        validate_command.extend(
            ["--expected-member-digest", split, digest]
        )
    validated = subprocess.run(
        validate_command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert validated.returncode == 0, validated.stderr
    validation_payload = json.loads(validated.stdout)
    assert validation_payload == publication_payload


@pytest.mark.parametrize(
    "mutation",
    ["manifest", "checksum", "self_digest", "noncanonical"],
)
def test_family_loader_rejects_manifest_checksum_and_digest_tamper(
    tmp_path: Path,
    mutation: str,
) -> None:
    members, _ = _publish_members(tmp_path / "source")
    root = publish_risk_dataset_family(tmp_path / "family", members=members)
    manifest_path = root / "family_manifest.json"
    if mutation == "manifest":
        manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
        expected = "checksum"
    elif mutation == "checksum":
        checksum_path = root / "checksums.sha256"
        payload = checksum_path.read_text(encoding="utf-8")
        checksum_path.write_text("0" * 64 + payload[64:], encoding="utf-8")
        expected = "checksum"
    elif mutation == "self_digest":
        manifest = _family_manifest(root)
        manifest["risk_dataset_family_digest"] = "f" * 64
        write_canonical_json(manifest_path, manifest)
        _resign_family(root)
        expected = "family digest"
    else:
        manifest = _family_manifest(root)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _resign_family(root)
        expected = "canonical"

    with pytest.raises(RiskDataContractError, match=expected):
        load_risk_dataset_family(root)


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "symlink"])
def test_family_loader_rejects_incomplete_unexpected_and_symlinked_roots(
    tmp_path: Path,
    mutation: str,
) -> None:
    members, _ = _publish_members(tmp_path / "source")
    root = publish_risk_dataset_family(tmp_path / "family", members=members)
    if mutation == "missing":
        (root / ".producer-complete").unlink()
    elif mutation == "unexpected":
        (root / "orphan.tmp").write_text("partial", encoding="utf-8")
    else:
        manifest_path = root / "family_manifest.json"
        detached = tmp_path / "detached-family-manifest.json"
        manifest_path.rename(detached)
        manifest_path.symlink_to(detached)

    with pytest.raises(RiskDataContractError, match="missing|unexpected|symlink"):
        load_risk_dataset_family(root)


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong"])
def test_family_loader_requires_exact_expected_member_digest_mapping(
    tmp_path: Path,
    mutation: str,
) -> None:
    members, _ = _publish_members(tmp_path / "source")
    root = publish_risk_dataset_family(tmp_path / "family", members=members)
    expected = _member_digests(members)
    if mutation == "missing":
        expected.pop("test")
        message = "exactly cover"
    elif mutation == "extra":
        expected["validation"] = expected["val"]
        message = "exactly cover"
    else:
        expected["test"] = "f" * 64
        message = "test"

    with pytest.raises(RiskDataContractError, match=message):
        load_risk_dataset_family(root, expected_member_digests=expected)


def test_family_publication_refuses_overwrite(tmp_path: Path) -> None:
    members, _ = _publish_members(tmp_path / "source")
    root = publish_risk_dataset_family(tmp_path / "family", members=members)

    with pytest.raises(FileExistsError, match="overwrite"):
        publish_risk_dataset_family(root, members=members)


def test_family_manifest_and_digest_are_deterministic_across_absolute_roots(
    tmp_path: Path,
) -> None:
    first_members, _ = _publish_members(tmp_path / "first" / "source")
    second_members, _ = _publish_members(tmp_path / "second" / "source")
    first_root = publish_risk_dataset_family(
        tmp_path / "first" / "family", members=first_members
    )
    second_root = publish_risk_dataset_family(
        tmp_path / "second" / "family", members=second_members
    )

    first = load_risk_dataset_family(first_root)
    second = load_risk_dataset_family(second_root)
    assert first.risk_dataset_family_digest == second.risk_dataset_family_digest
    assert (first_root / "family_manifest.json").read_bytes() == (
        second_root / "family_manifest.json"
    ).read_bytes()


def test_family_loader_rejects_recomputed_non_proven_audit(
    tmp_path: Path,
) -> None:
    members, _ = _publish_members(tmp_path / "source")
    root = publish_risk_dataset_family(tmp_path / "family", members=members)
    manifest = _family_manifest(root)
    audit = manifest["cross_split_audit"]
    assert isinstance(audit, dict)
    audit["global_cross_split_leakage"] = "NOT_PROVEN"
    manifest["risk_dataset_family_digest"] = _family_digest(manifest)
    write_canonical_json(root / "family_manifest.json", manifest)
    _resign_family(root)

    with pytest.raises(RiskDataContractError, match="not proven"):
        load_risk_dataset_family(root)


def test_family_loader_pins_one_manifest_snapshot_across_hash_and_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members, _ = _publish_members(tmp_path / "source")
    root = publish_risk_dataset_family(tmp_path / "family", members=members)
    original = _family_manifest(root)
    original_digest = original["risk_dataset_family_digest"]
    replacement = json.loads(canonical_json(original))
    replacement_members = replacement["members"]
    replacement_audit = replacement["cross_split_audit"]
    assert isinstance(replacement_members, dict)
    assert isinstance(replacement_members["train"], dict)
    assert isinstance(replacement_audit, dict)
    replacement_counts = replacement_audit["authenticated_metadata_row_counts"]
    assert isinstance(replacement_counts, dict)
    replacement_members["train"]["sample_count"] = 13
    replacement_counts["train"] = 13
    replacement["risk_dataset_family_digest"] = _family_digest(replacement)
    replacement_path = tmp_path / "replacement-family-manifest.json"
    write_canonical_json(replacement_path, replacement)
    manifest_path = root / "family_manifest.json"
    real_hash = seal_module._sha256_file
    real_open = seal_module.os.open
    swapped = False

    def replace_manifest_once() -> None:
        nonlocal swapped
        if not swapped:
            os.replace(replacement_path, manifest_path)
            swapped = True

    def swap_after_path_hash(path: Path) -> str:
        digest = real_hash(path)
        if Path(path) == manifest_path:
            replace_manifest_once()
        return digest

    def swap_after_pinned_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd is not None and os.fspath(path) == "family_manifest.json":
            replace_manifest_once()
        return descriptor

    monkeypatch.setattr(seal_module, "_sha256_file", swap_after_path_hash)
    monkeypatch.setattr(seal_module.os, "open", swap_after_pinned_open)

    loaded = load_risk_dataset_family(root)

    assert swapped is True
    assert loaded.risk_dataset_family_digest == original_digest
    assert loaded.members["train"]["sample_count"] == 12


def test_publisher_reauthenticates_members_and_formally_reloads_metadata_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members, _ = _publish_members(tmp_path / "source")
    members["train"] = replace(
        members["train"],
        manifest={"forged": True},
        split="test",
        sample_count=999,
        provenance={"forged": "true"},
    )
    real_shard_loader = seal_module.load_risk_shard
    shard_calls: list[str] = []

    def tracking_shard_loader(
        root: str | Path,
        **kwargs: object,
    ) -> object:
        shard_calls.append(Path(root).name)
        return real_shard_loader(root, **kwargs)

    monkeypatch.setattr(seal_module, "load_risk_shard", tracking_shard_loader)
    family = load_risk_dataset_family(
        publish_risk_dataset_family(tmp_path / "family", members=members)
    )

    assert shard_calls == [
        shard
        for _split in MEMBER_ORDER
        for shard in (
            "shard-00000",
            "shard-00001",
        )
    ]
    assert family.members["train"]["sample_count"] == 12


def test_family_publish_cli_formally_decodes_each_member_shard_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members, _ = _publish_members(tmp_path / "source")
    cli = _load_seal_cli_module()
    real_shard_loader = seal_module.load_risk_shard
    shard_calls: list[str] = []

    def tracking_shard_loader(
        root: str | Path,
        **kwargs: object,
    ) -> object:
        shard_calls.append(Path(root).name)
        return real_shard_loader(root, **kwargs)

    monkeypatch.setattr(seal_module, "load_risk_shard", tracking_shard_loader)
    argv = ["--output-dir", str(tmp_path / "cli-count-family")]
    for split in MEMBER_ORDER:
        argv.extend(
            [
                "--member",
                split,
                str(members[split].seal_root),
                str(members[split].collection_root),
                members[split].risk_dataset_manifest_digest,
            ]
        )

    assert cli._main_family_publish(argv) == 0
    assert shard_calls == [
        shard
        for _split in MEMBER_ORDER
        for shard in ("shard-00000", "shard-00001")
    ]
