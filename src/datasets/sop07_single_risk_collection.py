"""Adapt an immutable SOP07 single-risk release into an SOP08 collection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile

from src.contracts import SCHEMA_VERSION, GridSpec, build_grid_spec
from src.datasets.risk_dataloader import RiskDataContractError
from src.datasets.risk_dataset_seal import (
    SOP07_SINGLE_RISK_COLLECTION_HANDOFF_ROLE,
    SOP07_SINGLE_RISK_COLLECTION_HANDOFF_VERSION,
    SOP07_SINGLE_RISK_COLLECTION_PRODUCER_VERSION,
    RiskShardDescriptor,
    _formally_validate_collection,
    _load_authenticated_handoff,
    _single_release_collection_semantic_digest,
)
from src.datasets.shard_writer import RISK_SHARD_LAYOUT_VERSION, load_risk_shard
from src.utils.atomic_publish import atomic_rename_noreplace


SOP07_SINGLE_RISK_RELEASE_VERSION = "sop07_single_risk_release_v1"
_REQUEST_NAME = "request.json"
_MANIFEST_NAME = "manifest.json"
_CHECKSUMS_NAME = "checksums.json"
_COMPLETE_NAME = "COMPLETE.json"
_SHARDS_NAME = "shards"
_RELEASE_ENTRIES = frozenset(
    {_REQUEST_NAME, _MANIFEST_NAME, _CHECKSUMS_NAME, _COMPLETE_NAME, _SHARDS_NAME}
)
_SHARD_FILES = frozenset({"samples.npz", "metadata.jsonl", "summary.json"})


@dataclass(frozen=True)
class LoadedSingleRiskRelease:
    root: Path
    request_identity: str
    manifest_digest: str
    split: str
    sample_count: int
    shard_count: int
    grid: GridSpec
    descriptors: tuple[RiskShardDescriptor, ...]


@dataclass(frozen=True)
class PublishedSingleRiskCollection:
    root: Path
    split: str
    sample_count: int
    shard_count: int
    source_release_manifest_digest: str
    collection_semantic_digest: str
    handoff_sha256: str


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RiskDataContractError("single-risk metadata is not canonical JSON") from exc


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RiskDataContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise RiskDataContractError(f"{label} not found: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RiskDataContractError(f"{label} must be a direct regular file")
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RiskDataContractError(f"{label} is not strict finite JSON") from exc
    if not isinstance(value, dict):
        raise RiskDataContractError(f"{label} must contain a JSON object")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RiskDataContractError(f"failed to checksum {path}") from exc
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RiskDataContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_blake2b128(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RiskDataContractError(f"{label} must be a lowercase BLAKE2b-128")
    return value


def _require_commit(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RiskDataContractError("code_commit must be a lowercase 40-hex commit")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 1:
        raise RiskDataContractError(f"{label} must be a positive integer")
    return value


def _load_single_risk_release(root: str | Path) -> LoadedSingleRiskRelease:
    release_root = Path(root)
    if release_root.is_symlink() or not release_root.is_dir():
        raise RiskDataContractError("single-risk release root must be a direct directory")
    actual_entries = {path.name for path in release_root.iterdir()}
    if actual_entries != _RELEASE_ENTRIES:
        raise RiskDataContractError(
            "single-risk release entries differ: "
            f"missing={sorted(_RELEASE_ENTRIES - actual_entries)}, "
            f"unexpected={sorted(actual_entries - _RELEASE_ENTRIES)}"
        )

    request_document = _read_json(
        release_root / _REQUEST_NAME, label="single-risk request"
    )
    manifest = _read_json(
        release_root / _MANIFEST_NAME, label="single-risk manifest"
    )
    checksums = _read_json(
        release_root / _CHECKSUMS_NAME, label="single-risk checksums"
    )
    complete = _read_json(
        release_root / _COMPLETE_NAME, label="single-risk completion marker"
    )
    if (
        request_document.get("version") != SOP07_SINGLE_RISK_RELEASE_VERSION
        or manifest.get("version") != SOP07_SINGLE_RISK_RELEASE_VERSION
        or complete.get("version") != SOP07_SINGLE_RISK_RELEASE_VERSION
    ):
        raise RiskDataContractError("single-risk release version mismatch")
    request = request_document.get("request")
    if not isinstance(request, Mapping):
        raise RiskDataContractError("single-risk request payload is missing")
    request_identity = _sha256_bytes(
        b"sop07_single_risk_release_request_v1\0" + _canonical_json(dict(request))
    )
    if (
        request_document.get("request_identity") != request_identity
        or manifest.get("request_identity") != request_identity
    ):
        raise RiskDataContractError("single-risk request identity mismatch")
    if set(checksums) != {_REQUEST_NAME, _MANIFEST_NAME}:
        raise RiskDataContractError("single-risk checksum coverage mismatch")
    for name in (_REQUEST_NAME, _MANIFEST_NAME):
        expected = _require_sha256(
            checksums.get(name), label=f"single-risk checksum {name}"
        )
        if _sha256_file(release_root / name) != expected:
            raise RiskDataContractError(f"single-risk checksum mismatch: {name}")

    raw_config = manifest.get("resolved_base_config")
    if not isinstance(raw_config, Mapping):
        raise RiskDataContractError("single-risk resolved_base_config is missing")
    try:
        grid = build_grid_spec(dict(raw_config))
    except (TypeError, ValueError) as exc:
        raise RiskDataContractError("single-risk grid config is invalid") from exc
    split = manifest.get("split")
    if split not in {"train", "calibration", "val", "test"}:
        raise RiskDataContractError("single-risk split is invalid")
    sample_count = _positive_int(
        manifest.get("sample_count"), label="single-risk sample_count"
    )
    shard_count = _positive_int(
        manifest.get("shard_count"), label="single-risk shard_count"
    )
    raw_descriptors = manifest.get("shards")
    if not isinstance(raw_descriptors, list) or len(raw_descriptors) != shard_count:
        raise RiskDataContractError("single-risk shard descriptors are incomplete")
    shard_parent = release_root / _SHARDS_NAME
    if shard_parent.is_symlink() or not shard_parent.is_dir():
        raise RiskDataContractError("single-risk shards root must be a direct directory")
    expected_names = {f"shard-{index:05d}" for index in range(shard_count)}
    if {path.name for path in shard_parent.iterdir()} != expected_names:
        raise RiskDataContractError("single-risk child shard set mismatch")

    descriptors: list[RiskShardDescriptor] = []
    ordered_ids: list[str] = []
    for index, raw_descriptor in enumerate(raw_descriptors):
        if not isinstance(raw_descriptor, Mapping):
            raise RiskDataContractError("single-risk shard descriptor must be a mapping")
        source_relative = f"shards/shard-{index:05d}"
        if (
            raw_descriptor.get("shard_index") != index
            or raw_descriptor.get("relative_root") != source_relative
        ):
            raise RiskDataContractError("single-risk shard ordering mismatch")
        shard_root = release_root / source_relative
        try:
            loaded = load_risk_shard(shard_root, grid=grid)
        except (OSError, TypeError, ValueError) as exc:
            raise RiskDataContractError(
                f"single-risk shard {index} failed formal load"
            ) from exc
        sample_ids = tuple(sample.sample_id for sample in loaded.samples)
        expected_count = _positive_int(
            raw_descriptor.get("sample_count"),
            label=f"single-risk shard {index} sample_count",
        )
        if (
            loaded.summary.get("split") != split
            or loaded.summary.get("shard_index") != index
            or len(sample_ids) != expected_count
            or loaded.manifest_digest != raw_descriptor.get("risk_manifest_digest")
            or loaded.semantic_digest != raw_descriptor.get("risk_semantic_digest")
            or sample_ids[0] != raw_descriptor.get("first_sample_id")
            or sample_ids[-1] != raw_descriptor.get("last_sample_id")
        ):
            raise RiskDataContractError(
                f"single-risk shard {index} identity differs from manifest"
            )
        descriptors.append(
            RiskShardDescriptor(
                shard_index=index,
                relative_root=f"shard-{index:05d}",
                sample_count=len(sample_ids),
                manifest_digest=loaded.manifest_digest,
                semantic_digest=loaded.semantic_digest,
                payload_sha256=_sha256_file(shard_root / "samples.npz"),
                metadata_sha256=_sha256_file(shard_root / "metadata.jsonl"),
                summary_sha256=_sha256_file(shard_root / "summary.json"),
            )
        )
        ordered_ids.extend(sample_ids)

    if (
        len(ordered_ids) != sample_count
        or ordered_ids != sorted(ordered_ids)
        or len(set(ordered_ids)) != sample_count
    ):
        raise RiskDataContractError("single-risk global sample identities are invalid")
    manifest_digest = _sha256_bytes(
        b"sop07_single_risk_release_manifest_v1\0" + _canonical_json(manifest)
    )
    if complete != {
        "version": SOP07_SINGLE_RISK_RELEASE_VERSION,
        "request_identity": request_identity,
        "sample_count": sample_count,
        "shard_count": shard_count,
        "manifest_digest": manifest_digest,
    }:
        raise RiskDataContractError("single-risk completion identity mismatch")
    return LoadedSingleRiskRelease(
        root=release_root,
        request_identity=request_identity,
        manifest_digest=manifest_digest,
        split=str(split),
        sample_count=sample_count,
        shard_count=shard_count,
        grid=grid,
        descriptors=tuple(descriptors),
    )


def _copy_shard(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise RiskDataContractError("single-risk source shard must be a direct directory")
    entries = {path.name for path in source.iterdir()}
    if entries != _SHARD_FILES:
        raise RiskDataContractError("single-risk source shard file set mismatch")
    destination.mkdir()
    for name in sorted(_SHARD_FILES):
        source_file = source / name
        if source_file.is_symlink() or not source_file.is_file():
            raise RiskDataContractError("single-risk shard member must be regular")
        shutil.copy2(source_file, destination / name)


def publish_sop07_single_risk_collection(
    release_root: str | Path,
    output_dir: str | Path,
    *,
    target_type_policy_digest: str,
    code_commit: str,
) -> PublishedSingleRiskCollection:
    """Copy one verified single release into an immutable flat SOP08 collection."""

    loaded = _load_single_risk_release(release_root)
    target_digest = _require_blake2b128(
        target_type_policy_digest, label="target_type_policy_digest"
    )
    commit = _require_commit(code_commit)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable collection: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        for descriptor in loaded.descriptors:
            source = loaded.root / "shards" / descriptor.relative_root
            _copy_shard(source, staging / descriptor.relative_root)
        source_identity = {
            "version": SOP07_SINGLE_RISK_RELEASE_VERSION,
            "request_identity": loaded.request_identity,
            "manifest_digest": loaded.manifest_digest,
            "sample_count": loaded.sample_count,
            "shard_count": loaded.shard_count,
        }
        handoff: dict[str, object] = {
            "handoff_version": SOP07_SINGLE_RISK_COLLECTION_HANDOFF_VERSION,
            "artifact_role": SOP07_SINGLE_RISK_COLLECTION_HANDOFF_ROLE,
            "collection_state": "complete",
            "schema_version": SCHEMA_VERSION,
            "layout_version": RISK_SHARD_LAYOUT_VERSION,
            "split": loaded.split,
            "sample_count": loaded.sample_count,
            "shard_count": loaded.shard_count,
            "collection_instance_digest_sha256": "0" * 64,
            "collection_semantic_digest_sha256": "0" * 64,
            "code_commit": commit,
            "producer_version": SOP07_SINGLE_RISK_COLLECTION_PRODUCER_VERSION,
            "target_type_policy_digest": target_digest,
            "source_release": source_identity,
            "downstream_contract": {
                "global_sample_id_uniqueness": "PROVEN",
                "physical_npz_merge_performed": False,
            },
            "shards": [asdict(descriptor) for descriptor in loaded.descriptors],
        }
        semantic_digest = _single_release_collection_semantic_digest(
            handoff,
            descriptors=loaded.descriptors,
            expected_split=loaded.split,
            sample_count=loaded.sample_count,
        )
        handoff["collection_semantic_digest_sha256"] = semantic_digest
        handoff["collection_instance_digest_sha256"] = _sha256_bytes(
            b"sop07-single-risk-collection-instance-v1\0"
            + _canonical_json(
                {
                    "source_release": source_identity,
                    "collection_semantic_digest_sha256": semantic_digest,
                    "code_commit": commit,
                }
            )
        )
        handoff_path = staging / "collection_complete_handoff.json"
        handoff_path.write_bytes(_canonical_json(handoff) + b"\n")
        handoff_sha256 = _sha256_file(handoff_path)
        authenticated_handoff = _load_authenticated_handoff(
            staging,
            expected_split=loaded.split,
            expected_sha256=handoff_sha256,
        )
        descriptors, observed_target_digest = _formally_validate_collection(
            staging,
            grid=loaded.grid,
            expected_split=loaded.split,
            handoff=authenticated_handoff,
        )
        if (
            descriptors != loaded.descriptors
            or observed_target_digest != target_digest
        ):
            raise RiskDataContractError("single-risk collection staging reload mismatch")
        atomic_rename_noreplace(staging, output)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return PublishedSingleRiskCollection(
        root=output,
        split=loaded.split,
        sample_count=loaded.sample_count,
        shard_count=loaded.shard_count,
        source_release_manifest_digest=loaded.manifest_digest,
        collection_semantic_digest=semantic_digest,
        handoff_sha256=handoff_sha256,
    )


__all__ = [
    "LoadedSingleRiskRelease",
    "PublishedSingleRiskCollection",
    "SOP07_SINGLE_RISK_RELEASE_VERSION",
    "publish_sop07_single_risk_collection",
]
