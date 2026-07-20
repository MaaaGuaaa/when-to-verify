"""Immutable dataset-level seal for accepted schema-3 SOP07 risk shards.

The shard writer remains the only parser for shard payloads.  This module
authenticates one complete SOP07 collection, binds its ordered shards and
frozen provenance into ``risk_dataset_v2``, and fully reloads both publication
levels before returning a dataset descriptor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Mapping, Sequence

from src.contracts import (
    HISTORY_CHANNELS,
    INPUT_CHANNELS,
    SCHEMA_VERSION,
    STATE_CHANNELS,
    TRAJECTORY_CHANNELS,
    GridSpec,
    build_grid_spec,
)
from src.datasets.risk_dataloader import RiskDataContractError
from src.datasets.shard_writer import (
    RISK_SHARD_LAYOUT_VERSION,
    LoadedRiskShard,
    load_risk_shard,
)
from src.datasets.sidecar_writer import (
    RISK_SIDECAR_PAIR_COMPLETION_MARKER_VERSION,
    RISK_SIDECAR_SHARD_LAYOUT_VERSION,
    load_risk_sidecar_pair_completion_marker,
    load_risk_sidecar_shard,
    risk_sidecar_pair_completion_marker_path,
)
from src.utils.config import ConfigError, config_digest, load_config
from src.utils.atomic_publish import atomic_rename_noreplace


RISK_DATASET_LAYOUT_VERSION = "risk_dataset_v2"
RISK_DATASET_FAMILY_LAYOUT_VERSION = "risk_dataset_family_v1"
PRODUCTION_RISK_EVALUATION_METADATA_LAYOUT_VERSION = (
    "production_risk_evaluation_metadata_v1"
)
RISK_SIDECAR_COLLECTION_LAYOUT_VERSION = "risk_label_sidecar_collection_v1"

_COLLECTION_HANDOFF_VERSION = "sop07_collection_complete_handoff_v1"
_HELDOUT_COLLECTION_HANDOFF_VERSION = (
    "sop07_heldout_collection_complete_handoff_v1"
)
_HELDOUT_COLLECTION_HANDOFF_ROLE = "sop07_heldout_collection_complete_handoff"
_HELDOUT_GENERATION_REPORT_NAME = "batch_generation_report.json"
_HELDOUT_GENERATION_REPORT_VERSION = "sop07_heldout_batch_generation_report_v1"
_HELDOUT_GENERATION_REPORT_ROLE = "sop07_heldout_batch_generation_report"
_HELDOUT_COLLECTION_SEMANTIC_DOMAIN = (
    b"sop07-heldout-collection-semantic-v1\0"
)
_HELDOUT_SPLITS = frozenset({"calibration", "val", "test"})
_COLLECTION_HANDOFF_NAME = "collection_complete_handoff.json"
_DATASET_MANIFEST_NAME = "dataset_manifest.json"
_FAMILY_MANIFEST_NAME = "family_manifest.json"
_CHECKSUMS_NAME = "checksums.sha256"
_COMPLETE_MARKER_NAME = ".producer-complete"
_REQUIRED_SEAL_FILES = frozenset(
    {_DATASET_MANIFEST_NAME, _CHECKSUMS_NAME, _COMPLETE_MARKER_NAME}
)
_REQUIRED_FAMILY_FILES = frozenset(
    {_FAMILY_MANIFEST_NAME, _CHECKSUMS_NAME, _COMPLETE_MARKER_NAME}
)
_FAMILY_MEMBER_ORDER = ("train", "calibration", "val", "test")
_FAMILY_MANIFEST_KEYS = frozenset(
    {
        "dataset_family_layout_version",
        "schema_version",
        "member_order",
        "members",
        "common_contract",
        "cross_split_audit",
        "risk_dataset_family_digest",
    }
)
_FAMILY_MEMBER_KEYS = frozenset(
    {
        "split",
        "risk_dataset_manifest_digest",
        "sample_count",
        "shard_count",
    }
)
_FAMILY_COMMON_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "grid",
        "channel_spec",
        "g1_split_manifest_digest",
        "dynamic_objects_config_digest",
        "target_type_policy_digest",
    }
)
_FAMILY_AUDIT_KEYS = frozenset(
    {
        "policy_version",
        "member_order",
        "authenticated_metadata_row_counts",
        "forbidden_overlaps",
        "allowed_overlaps",
        "evidence_completeness",
        "global_sample_id_uniqueness",
        "global_cross_split_leakage",
    }
)
_FAMILY_OVERLAP_ENTRY_KEYS = frozenset(
    {"overlap_count", "overlap_values", "split_pair_overlaps"}
)
_FAMILY_SPLIT_PAIR_OVERLAP_KEYS = frozenset(
    {"left_split", "right_split", "overlap_count", "overlap_values"}
)
_FAMILY_FORBIDDEN_IDENTITY_FIELDS = (
    "sample_id",
    "base_recording_id",
    "source_recording_id",
    "base_source_cross_role_recording_id",
    "source_snippet_id",
    "pair_group_id",
    "seed_namespace",
)
_FAMILY_ALLOWED_IDENTITY_FIELDS = (
    "base_session_id",
    "source_session_id",
    "base_source_cross_role_session_id",
)
_FAMILY_AUDIT_POLICY_VERSION = "thor_recording_generalization_v1"
_SHARD_NAME = re.compile(r"^shard-([0-9]{5})$")
_LOWER_HEX_32 = re.compile(r"^[0-9a-f]{32}$")
_LOWER_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_LOWER_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_TARGET_CHANNELS = (
    "collision_label",
    "risk_severity",
    "min_clearance",
    "near_miss",
    "first_collision_time",
)
_GRID_KEYS = frozenset(
    {
        "height",
        "width",
        "history_steps",
        "future_steps",
        "resolution_m",
        "sample_dt_s",
    }
)
_CHANNEL_SPEC_KEYS = frozenset(
    {"history", "state", "trajectory", "flat", "targets"}
)
_SHARD_DESCRIPTOR_KEYS = frozenset(
    {
        "shard_index",
        "relative_root",
        "sample_count",
        "manifest_digest",
        "semantic_digest",
        "payload_sha256",
        "metadata_sha256",
        "summary_sha256",
    }
)
_DATASET_MANIFEST_KEYS = frozenset(
    {
        "dataset_layout_version",
        "schema_version",
        "split",
        "collection_handoff_version",
        "collection_handoff_sha256",
        "collection_artifact_role",
        "collection_instance_digest_sha256",
        "collection_semantic_digest_sha256",
        "collection_code_commit",
        "collection_producer_version",
        "sop03_code_commit",
        "sop03_finalizer_commit",
        "sample_count",
        "shard_count",
        "grid",
        "channel_spec",
        "shards",
        "g1_split_manifest_digest",
        "dynamic_objects_config_digest",
        "target_type_policy_digest",
        "risk_dataset_manifest_digest",
    }
)
_DATASET_MANIFEST_WITH_SIDECARS_KEYS = frozenset(
    {*_DATASET_MANIFEST_KEYS, "occupancy_sidecars"}
)
_SIDECAR_DESCRIPTOR_KEYS = frozenset(
    {
        "shard_index",
        "relative_root",
        "marker_relative_path",
        "sample_count",
        "sidecar_semantic_digest",
        "source_risk_shard_semantic_digest",
        "pair_marker_digest_sha256",
        "ordered_sample_ids_digest_sha256",
    }
)
_SIDECAR_COLLECTION_KEYS = frozenset(
    {
        "collection_layout_version",
        "sidecar_shard_layout_version",
        "pair_completion_marker_version",
        "base_risk_dataset_manifest_digest",
        "base_config_digest",
        "query_geometry",
        "sample_count",
        "shard_count",
        "shards",
        "collection_digest_sha256",
    }
)
_RUNTIME_IDENTITY_FIELDS = frozenset(
    {"collection_handoff_sha256", "collection_instance_digest_sha256"}
)


@dataclass(frozen=True)
class RiskShardDescriptor:
    """One ordered, authenticated SOP07 risk shard."""

    shard_index: int
    relative_root: str
    sample_count: int
    manifest_digest: str
    semantic_digest: str
    payload_sha256: str
    metadata_sha256: str
    summary_sha256: str


@dataclass(frozen=True)
class LoadedRiskDataset:
    """A fully validated dataset seal and its immutable shard collection."""

    seal_root: Path
    collection_root: Path
    manifest: dict[str, object]
    grid: GridSpec
    shards: tuple[RiskShardDescriptor, ...]
    split: str
    sample_count: int
    risk_dataset_manifest_digest: str
    provenance: dict[str, str]


@dataclass(frozen=True)
class LoadedRiskDatasetFamily:
    """A complete immutable four-split risk-dataset family publication."""

    root: Path
    manifest: dict[str, object]
    members: dict[str, dict[str, object]]
    cross_split_audit: dict[str, object]
    risk_dataset_family_digest: str


@dataclass(frozen=True)
class SidecarShardDescriptor:
    """One ordered sidecar shard and its exact risk-pair completion proof."""

    shard_index: int
    relative_root: str
    marker_relative_path: str
    sample_count: int
    sidecar_semantic_digest: str
    source_risk_shard_semantic_digest: str
    pair_marker_digest_sha256: str
    ordered_sample_ids_digest_sha256: str


@dataclass(frozen=True)
class LoadedRiskSidecarCollection:
    """Fully verified SOP08 sidecars bound to one base risk dataset."""

    root: Path
    shards: tuple[SidecarShardDescriptor, ...]
    sample_count: int
    collection_digest_sha256: str
    query_geometry: dict[str, object]


class _FrozenDict(dict):
    """Recursively frozen ``dict`` preserving JSON-like downstream behavior."""

    def __init__(self, value: Mapping[object, object]) -> None:
        dict.__init__(
            self,
            ((key, _deep_freeze_json(child)) for key, child in value.items()),
        )

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("frozen dataset metadata cannot be mutated")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _deep_freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_json(child) for child in value)
    return value


def _frozen_string_object_dict(
    value: Mapping[str, object],
) -> dict[str, object]:
    return _FrozenDict(value)


def _frozen_string_string_dict(value: Mapping[str, str]) -> dict[str, str]:
    return _FrozenDict(value)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_dynamic_objects_digest(value: object) -> str:
    """Return SHA-256 over the exact canonical-JSON dynamic-object subtree."""

    try:
        payload = _canonical_json(value).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RiskDataContractError(
            "dynamic_objects must be finite canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _reject_duplicate_json_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def _strict_json_loads(payload: str, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
            object_pairs_hook=_reject_duplicate_json_pairs,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise RiskDataContractError(f"{label} is not strict finite JSON") from exc
    if not isinstance(value, dict):
        raise RiskDataContractError(f"{label} must contain a JSON object")
    return value


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise RiskDataContractError(f"failed to read {label}: {path}") from exc
    return _strict_json_loads(text, label=label)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _framed_domain_digest(domain: bytes, value: object, *, label: str) -> str:
    try:
        payload = _canonical_json(value).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RiskDataContractError(
            f"{label} must be finite canonical JSON"
        ) from exc
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RiskDataContractError(f"failed to hash file: {path}") from exc
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _LOWER_HEX_64.fullmatch(value) is None:
        raise RiskDataContractError(
            f"{field} must be exactly 64 lowercase hexadecimal characters (SHA-256)"
        )
    return value


def _require_blake2b128(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _LOWER_HEX_32.fullmatch(value) is None:
        raise RiskDataContractError(
            f"{field} must be exactly 32 lowercase hexadecimal characters "
            "(BLAKE2b-128)"
        )
    return value


def _require_commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _LOWER_HEX_40.fullmatch(value) is None:
        raise RiskDataContractError(
            f"{field} must be exactly 40 lowercase hexadecimal characters"
        )
    return value


def _require_nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RiskDataContractError(f"{field} must be a non-empty string")
    return value


def _require_positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise RiskDataContractError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise RiskDataContractError(f"{field} must be a non-negative integer")
    return value


def _require_positive_finite_float(value: object, *, field: str) -> float:
    if type(value) not in {int, float}:
        raise RiskDataContractError(f"{field} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RiskDataContractError(f"{field} must be a positive finite number")
    return result


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _atomic_rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Publish through the repository's portable immutable-path primitive."""

    atomic_rename_noreplace(source, destination)


def _assert_no_symlink_components(
    path: Path,
    *,
    label: str,
    allow_missing: bool = False,
) -> None:
    absolute = _absolute_without_symlink_resolution(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if allow_missing:
                return
            raise RiskDataContractError(f"{label} does not exist: {path}")
        except OSError as exc:
            raise RiskDataContractError(f"failed to inspect {label}: {path}") from exc
        if stat.S_ISLNK(mode):
            raise RiskDataContractError(
                f"{label} contains a forbidden symlink component: {current}"
            )


def _require_regular_file(path: Path, *, label: str) -> None:
    _assert_no_symlink_components(path, label=label)
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise RiskDataContractError(f"missing {label}: {path}") from exc
    if not stat.S_ISREG(mode):
        raise RiskDataContractError(f"{label} must be a regular file: {path}")


def _require_directory(path: Path, *, label: str) -> None:
    _assert_no_symlink_components(path, label=label)
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise RiskDataContractError(f"missing {label}: {path}") from exc
    if not stat.S_ISDIR(mode):
        raise RiskDataContractError(f"{label} must be a directory: {path}")


def _expected_channel_spec() -> dict[str, object]:
    return {
        "history": list(HISTORY_CHANNELS),
        "state": list(STATE_CHANNELS),
        "trajectory": list(TRAJECTORY_CHANNELS),
        "flat": list(INPUT_CHANNELS),
        "targets": list(_TARGET_CHANNELS),
    }


def _validate_channel_spec(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _CHANNEL_SPEC_KEYS:
        raise RiskDataContractError("dataset channel_spec keys violate the frozen order")
    expected = _expected_channel_spec()
    if value != expected:
        raise RiskDataContractError(
            "dataset channel_spec does not match the frozen ordered channels"
        )
    return dict(value)


def _grid_manifest(config: Mapping[str, object], grid: GridSpec) -> dict[str, object]:
    bev = config.get("bev")
    if not isinstance(bev, Mapping):
        raise RiskDataContractError("base config bev snapshot must be a mapping")
    history_dt = bev.get("history_dt_s")
    future_dt = bev.get("future_dt_s")
    if (
        isinstance(history_dt, bool)
        or not isinstance(history_dt, (int, float))
        or isinstance(future_dt, bool)
        or not isinstance(future_dt, (int, float))
        or not math.isfinite(float(history_dt))
        or not math.isfinite(float(future_dt))
        or float(history_dt) <= 0.0
        or float(future_dt) <= 0.0
        or float(history_dt) != float(future_dt)
    ):
        raise RiskDataContractError(
            "base config history_dt_s and future_dt_s must be one equal positive grid"
        )
    return {
        "height": grid.height,
        "width": grid.width,
        "history_steps": grid.history_steps,
        "future_steps": grid.future_steps,
        "resolution_m": grid.resolution_m,
        "sample_dt_s": float(future_dt),
    }


def _grid_from_manifest(value: object) -> GridSpec:
    if not isinstance(value, dict) or set(value) != _GRID_KEYS:
        raise RiskDataContractError("dataset grid keys violate risk_dataset_v2")
    integer_values = {}
    for field in ("height", "width", "history_steps", "future_steps"):
        integer_values[field] = _require_positive_int(
            value.get(field), field=f"grid.{field}"
        )
    resolution = value.get("resolution_m")
    sample_dt = value.get("sample_dt_s")
    for field, raw in (("resolution_m", resolution), ("sample_dt_s", sample_dt)):
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) <= 0.0
        ):
            raise RiskDataContractError(f"grid.{field} must be finite and positive")
    if integer_values["height"] != integer_values["width"]:
        raise RiskDataContractError(
            "dataset grid height and width must match the frozen square BEV"
        )
    return GridSpec(
        height=integer_values["height"],
        width=integer_values["width"],
        history_steps=integer_values["history_steps"],
        future_steps=integer_values["future_steps"],
        resolution_m=float(resolution),
    )


def _load_base_config(path: Path) -> tuple[dict[str, object], GridSpec, dict[str, object]]:
    _require_regular_file(path, label="base config")
    try:
        config = load_config(path)
        grid = build_grid_spec(config)
    except (ConfigError, KeyError, OSError, TypeError, ValueError) as exc:
        raise RiskDataContractError(f"invalid base config: {path}") from exc
    if not isinstance(config, dict):  # defensive: load_config contract
        raise RiskDataContractError("base config must contain a mapping")
    grid_value = _grid_manifest(config, grid)
    if _grid_from_manifest(grid_value) != grid:
        raise RiskDataContractError("base config grid is internally inconsistent")
    return config, grid, grid_value


def _parse_checksum_manifest_bytes(
    raw: bytes,
    *,
    label: str,
) -> dict[str, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise RiskDataContractError(f"failed to read {label}") from exc
    if not raw or not raw.endswith(b"\n"):
        raise RiskDataContractError(f"{label} must be non-empty and newline-terminated")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise RiskDataContractError(f"{label} line {line_number} is malformed")
        digest = _require_sha256(parts[0], field=f"{label} digest")
        relative = parts[1]
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or relative_path.as_posix() != relative
        ):
            raise RiskDataContractError(f"{label} contains an unsafe path")
        if relative in entries:
            raise RiskDataContractError(f"{label} contains a duplicate path")
        entries[relative] = digest
    expected_text = "".join(
        f"{entries[relative]}  {relative}\n" for relative in sorted(entries)
    )
    if text != expected_text:
        raise RiskDataContractError(f"{label} entries must be in canonical path order")
    return entries


def _parse_checksum_manifest(path: Path, *, label: str) -> dict[str, str]:
    _require_regular_file(path, label=label)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RiskDataContractError(f"failed to read {label}") from exc
    return _parse_checksum_manifest_bytes(raw, label=label)


def _load_authenticated_sop03_manifest(
    run_manifest_path: Path,
    *,
    base_config_path: Path,
    base_config: Mapping[str, object],
) -> tuple[str, str, str, str]:
    _require_regular_file(run_manifest_path, label="SOP03 run_manifest.json")
    if run_manifest_path.name != "run_manifest.json":
        raise RiskDataContractError(
            "split_provenance_path must identify authenticated run_manifest.json"
        )
    root = run_manifest_path.parent
    marker = root / _COMPLETE_MARKER_NAME
    checksums = root / "artifact_checksums.sha256"
    summary_path = root / "artifact_checksum_summary.json"
    _require_regular_file(marker, label="SOP03 completion marker")
    if marker.read_bytes() != b"":
        raise RiskDataContractError("SOP03 completion marker must be empty")
    entries = _parse_checksum_manifest(checksums, label="SOP03 checksum manifest")
    required_entries = {"run_manifest.json", _COMPLETE_MARKER_NAME}
    if not required_entries.issubset(entries):
        raise RiskDataContractError(
            "SOP03 checksum manifest does not authenticate run_manifest.json and marker"
        )
    if entries["run_manifest.json"] != _sha256_file(run_manifest_path):
        raise RiskDataContractError("SOP03 run_manifest.json checksum mismatch")
    if entries[_COMPLETE_MARKER_NAME] != _sha256_file(marker):
        raise RiskDataContractError("SOP03 completion marker checksum mismatch")
    _require_regular_file(summary_path, label="SOP03 checksum summary")
    checksum_summary = _read_json(summary_path, label="SOP03 checksum summary")
    if (
        checksum_summary.get("status") != "complete"
        or checksum_summary.get("checksum_algorithm") != "sha256"
        or checksum_summary.get("checksum_manifest") != "artifact_checksums.sha256"
        or checksum_summary.get("checksum_manifest_sha256") != _sha256_file(checksums)
        or checksum_summary.get("covered_file_count") != len(entries)
    ):
        raise RiskDataContractError("SOP03 checksum summary is inconsistent")

    manifest = _read_json(run_manifest_path, label="SOP03 run_manifest.json")
    if manifest.get("status") != "complete":
        raise RiskDataContractError("SOP03 run_manifest status must be complete")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RiskDataContractError("SOP03 run_manifest schema_version mismatch")
    validation = manifest.get("validation")
    if not isinstance(validation, Mapping) or validation.get("status") != "passed":
        raise RiskDataContractError("SOP03 run_manifest validation is not passed")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise RiskDataContractError("SOP03 run_manifest inputs are missing")
    g1_digest = _require_blake2b128(
        inputs.get("split_manifest_digest"), field="g1_split_manifest_digest"
    )
    protocol = manifest.get("producer_protocol")
    snapshots = protocol.get("config_snapshots") if isinstance(protocol, Mapping) else None
    snapshot = snapshots.get("base") if isinstance(snapshots, Mapping) else None
    if not isinstance(snapshot, Mapping):
        raise RiskDataContractError(
            "SOP03 run_manifest frozen base config snapshot is absent"
        )
    expected_config_sha = _sha256_file(base_config_path)
    if snapshot.get("sha256") != expected_config_sha:
        raise RiskDataContractError("SOP03 frozen base config snapshot SHA-256 mismatch")
    snapshot_value = snapshot.get("value")
    if not isinstance(snapshot_value, Mapping) or dict(snapshot_value) != dict(base_config):
        raise RiskDataContractError(
            "SOP03 frozen base config snapshot is inconsistent with base_config_path"
        )
    dynamic_objects = snapshot_value.get("dynamic_objects")
    if not isinstance(dynamic_objects, Mapping) or not dynamic_objects:
        raise RiskDataContractError(
            "SOP03 frozen base config snapshot dynamic_objects subtree is absent"
        )
    dynamic_digest = canonical_dynamic_objects_digest(dynamic_objects)
    _require_sha256(dynamic_digest, field="dynamic_objects_config_digest")

    repository = manifest.get("repository")
    if not isinstance(repository, Mapping):
        raise RiskDataContractError("SOP03 repository provenance is missing")
    code_commit = _require_commit(
        repository.get("code_commit"), field="SOP03 code_commit"
    )
    finalizer_commit = _require_commit(
        repository.get("finalizer_commit"), field="SOP03 finalizer_commit"
    )
    return g1_digest, dynamic_digest, code_commit, finalizer_commit


def _load_authenticated_heldout_generation_report(
    collection_root: Path,
    *,
    handoff: Mapping[str, object],
) -> str:
    evidence = handoff.get("generation_report_evidence")
    if not isinstance(evidence, Mapping):
        raise RiskDataContractError(
            "heldout generation report evidence is missing"
        )
    if evidence.get("relative_path") != _HELDOUT_GENERATION_REPORT_NAME:
        raise RiskDataContractError(
            "heldout generation report must use the safe batch_generation_report.json "
            "basename"
        )
    report_path = collection_root / _HELDOUT_GENERATION_REPORT_NAME
    _require_regular_file(report_path, label="heldout generation report")
    expected_sha256 = _require_sha256(
        evidence.get("sha256"),
        field="heldout generation report sha256",
    )
    try:
        raw = report_path.read_bytes()
    except OSError as exc:
        raise RiskDataContractError(
            "failed to read heldout generation report"
        ) from exc
    if _sha256_bytes(raw) != expected_sha256:
        raise RiskDataContractError("heldout generation report SHA-256 mismatch")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise RiskDataContractError(
            "heldout generation report must be UTF-8"
        ) from exc
    report = _strict_json_loads(text, label="heldout generation report")
    if (
        report.get("report_version") != _HELDOUT_GENERATION_REPORT_VERSION
        or report.get("artifact_role") != _HELDOUT_GENERATION_REPORT_ROLE
    ):
        raise RiskDataContractError("heldout generation report dialect mismatch")
    if report.get("generation_state") != "complete":
        raise RiskDataContractError("heldout generation report is not complete")
    for field in ("schema_version", "layout_version", "split", "code_commit"):
        if report.get(field) != handoff.get(field):
            raise RiskDataContractError(
                f"heldout generation report {field} mismatch"
            )
    _require_commit(report.get("code_commit"), field="heldout report code_commit")

    report_sample_count = _require_positive_int(
        report.get("sample_count"), field="heldout report sample_count"
    )
    report_shard_count = _require_positive_int(
        report.get("shard_count"), field="heldout report shard_count"
    )
    report_event_count = _require_positive_int(
        report.get("event_count"), field="heldout report event_count"
    )
    evidence_sample_count = _require_positive_int(
        evidence.get("sample_count"), field="heldout report evidence sample_count"
    )
    evidence_shard_count = _require_positive_int(
        evidence.get("shard_count"), field="heldout report evidence shard_count"
    )
    evidence_event_count = _require_positive_int(
        evidence.get("event_count"), field="heldout report evidence event_count"
    )
    if (
        report_sample_count != handoff.get("sample_count")
        or report_sample_count != evidence_sample_count
        or report_shard_count != handoff.get("shard_count")
        or report_shard_count != evidence_shard_count
        or report_event_count != evidence_event_count
    ):
        raise RiskDataContractError("heldout generation report count mismatch")

    report_semantic_digest = _require_sha256(
        report.get("batch_generation_semantic_digest_sha256"),
        field="heldout report semantic digest",
    )
    report_instance_digest = _require_sha256(
        report.get("batch_generation_instance_digest_sha256"),
        field="heldout report instance digest",
    )
    evidence_semantic_digest = _require_sha256(
        evidence.get("semantic_digest_sha256"),
        field="heldout report evidence semantic digest",
    )
    evidence_instance_digest = _require_sha256(
        evidence.get("instance_digest_sha256"),
        field="heldout report evidence instance digest",
    )
    if (
        report_semantic_digest != evidence_semantic_digest
        or report_instance_digest != evidence_instance_digest
    ):
        raise RiskDataContractError("heldout generation report digest mismatch")
    conservation = report.get("conservation")
    if (
        evidence.get("conservation_status") != "PROVEN"
        or not isinstance(conservation, Mapping)
        or conservation.get("status") != "PROVEN"
    ):
        raise RiskDataContractError(
            "heldout generation report conservation is not proven"
        )
    return _require_nonempty_string(
        report.get("producer_version"),
        field="heldout generation report producer_version",
    )


def _load_authenticated_handoff(
    collection_root: Path,
    *,
    expected_split: str,
    expected_sha256: str,
) -> dict[str, object]:
    _require_directory(collection_root, label="SOP07 collection root")
    handoff_path = collection_root / _COLLECTION_HANDOFF_NAME
    _require_regular_file(handoff_path, label="SOP07 collection handoff")
    expected_digest = _require_sha256(
        expected_sha256, field="expected_collection_handoff_sha256"
    )
    try:
        raw = handoff_path.read_bytes()
    except OSError as exc:
        raise RiskDataContractError("failed to read SOP07 collection handoff") from exc
    if _sha256_bytes(raw) != expected_digest:
        raise RiskDataContractError("SOP07 collection handoff SHA-256 mismatch")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise RiskDataContractError("SOP07 collection handoff must be UTF-8") from exc
    handoff = _strict_json_loads(text, label="SOP07 collection handoff")
    if expected_split == "train":
        expected_version = _COLLECTION_HANDOFF_VERSION
        expected_role = "sop07_train_collection_complete_handoff"
        is_heldout = False
    elif expected_split in _HELDOUT_SPLITS:
        expected_version = _HELDOUT_COLLECTION_HANDOFF_VERSION
        expected_role = _HELDOUT_COLLECTION_HANDOFF_ROLE
        is_heldout = True
    else:
        raise RiskDataContractError(
            "unsupported SOP07 collection split/handoff dialect"
        )
    if (
        handoff.get("handoff_version") != expected_version
        or handoff.get("artifact_role") != expected_role
    ):
        raise RiskDataContractError("SOP07 collection handoff dialect mismatch")
    if handoff.get("collection_state") != "complete":
        raise RiskDataContractError("SOP07 collection handoff is not complete")
    if handoff.get("schema_version") != SCHEMA_VERSION:
        raise RiskDataContractError("SOP07 collection schema_version mismatch")
    if handoff.get("layout_version") != RISK_SHARD_LAYOUT_VERSION:
        raise RiskDataContractError("SOP07 collection uses an unsupported shard layout")
    if handoff.get("split") != expected_split:
        raise RiskDataContractError("SOP07 collection split mismatch")
    _require_positive_int(handoff.get("sample_count"), field="collection sample_count")
    _require_positive_int(handoff.get("shard_count"), field="collection shard_count")
    _require_sha256(
        handoff.get("collection_instance_digest_sha256"),
        field="collection_instance_digest_sha256",
    )
    _require_sha256(
        handoff.get("collection_semantic_digest_sha256"),
        field="collection_semantic_digest_sha256",
    )
    _require_commit(handoff.get("code_commit"), field="SOP07 code_commit")
    if is_heldout:
        producer_version = _load_authenticated_heldout_generation_report(
            collection_root,
            handoff=handoff,
        )
        declared_producer = handoff.get("producer_version")
        if declared_producer is not None and (
            _require_nonempty_string(
                declared_producer,
                field="SOP07 heldout producer_version",
            )
            != producer_version
        ):
            raise RiskDataContractError(
                "heldout generation report producer_version mismatch"
            )
        handoff = dict(handoff)
        handoff["producer_version"] = producer_version
    else:
        _require_nonempty_string(
            handoff.get("producer_version"), field="SOP07 producer_version"
        )
    downstream = handoff.get("downstream_contract")
    if not isinstance(downstream, Mapping) or (
        downstream.get("global_sample_id_uniqueness") != "PROVEN"
        or downstream.get("physical_npz_merge_performed") is not False
    ):
        raise RiskDataContractError(
            "SOP07 collection completeness/uniqueness evidence is missing"
        )
    shards = handoff.get("shards")
    if not isinstance(shards, list) or len(shards) != handoff["shard_count"]:
        raise RiskDataContractError("SOP07 collection shard descriptors are incomplete")
    return handoff


def _discover_shards(collection_root: Path) -> tuple[Path, ...]:
    discovered: list[tuple[int, Path]] = []
    try:
        entries = list(os.scandir(collection_root))
    except OSError as exc:
        raise RiskDataContractError("failed to enumerate SOP07 collection") from exc
    for entry in entries:
        if entry.is_symlink():
            raise RiskDataContractError(
                f"SOP07 collection contains a forbidden symlink: {entry.name}"
            )
        if not entry.name.startswith("shard-"):
            continue
        match = _SHARD_NAME.fullmatch(entry.name)
        if match is None:
            raise RiskDataContractError(
                f"malformed SOP07 shard directory name: {entry.name}"
            )
        if not entry.is_dir(follow_symlinks=False):
            raise RiskDataContractError(f"SOP07 shard root is not a directory: {entry.name}")
        discovered.append((int(match.group(1)), Path(entry.path)))
    discovered.sort(key=lambda item: item[0])
    if not discovered:
        raise RiskDataContractError("SOP07 collection contains no shards")
    indices = [index for index, _ in discovered]
    if indices != list(range(len(discovered))):
        raise RiskDataContractError(
            "SOP07 shard indices must be unique and contiguous from shard-00000"
        )
    return tuple(path for _, path in discovered)


def _safe_shard_root(path: Path) -> None:
    _require_directory(path, label="SOP07 shard root")
    try:
        entries = list(os.scandir(path))
    except OSError as exc:
        raise RiskDataContractError(f"failed to enumerate SOP07 shard: {path}") from exc
    for entry in entries:
        if entry.is_symlink():
            raise RiskDataContractError(
                f"SOP07 shard contains a forbidden symlink: {path / entry.name}"
            )


def _descriptor_from_mapping(
    value: object,
    *,
    position: int,
    exact_keys: bool,
) -> RiskShardDescriptor:
    if not isinstance(value, Mapping):
        raise RiskDataContractError(f"shards[{position}] must be a mapping")
    if exact_keys and set(value) != _SHARD_DESCRIPTOR_KEYS:
        raise RiskDataContractError(
            f"dataset shard descriptor {position} keys violate risk_dataset_v2"
        )
    missing = _SHARD_DESCRIPTOR_KEYS - set(value)
    if missing:
        raise RiskDataContractError(
            f"shards[{position}] missing descriptor fields: {sorted(missing)}"
        )
    shard_index = _require_nonnegative_int(
        value.get("shard_index"), field=f"shards[{position}].shard_index"
    )
    if shard_index != position:
        raise RiskDataContractError(
            "ordered shard indices must be unique and contiguous from zero"
        )
    relative_root = _require_nonempty_string(
        value.get("relative_root"), field=f"shards[{position}].relative_root"
    )
    if relative_root != f"shard-{position:05d}":
        raise RiskDataContractError(
            "ordered shard relative roots must equal shard-00000.. sequence"
        )
    if Path(relative_root).name != relative_root or Path(relative_root).is_absolute():
        raise RiskDataContractError("shard relative_root must be one safe path component")
    sample_count = _require_positive_int(
        value.get("sample_count"), field=f"shards[{position}].sample_count"
    )
    return RiskShardDescriptor(
        shard_index=shard_index,
        relative_root=relative_root,
        sample_count=sample_count,
        manifest_digest=_require_sha256(
            value.get("manifest_digest"),
            field=f"shards[{position}].manifest_digest",
        ),
        semantic_digest=_require_sha256(
            value.get("semantic_digest"),
            field=f"shards[{position}].semantic_digest",
        ),
        payload_sha256=_require_sha256(
            value.get("payload_sha256"),
            field=f"shards[{position}].payload_sha256",
        ),
        metadata_sha256=_require_sha256(
            value.get("metadata_sha256"),
            field=f"shards[{position}].metadata_sha256",
        ),
        summary_sha256=_require_sha256(
            value.get("summary_sha256"),
            field=f"shards[{position}].summary_sha256",
        ),
    )


def _query_geometry_from_config(
    config: Mapping[str, object], *, base_config_digest: str, grid: GridSpec
) -> dict[str, object]:
    robot = config.get("robot")
    bev = config.get("bev")
    if not isinstance(robot, Mapping) or not isinstance(bev, Mapping):
        raise RiskDataContractError("base config robot/bev query geometry is missing")
    if robot.get("model") != "differential_drive":
        raise RiskDataContractError(
            "SOP08 production query reconstruction requires differential_drive"
        )
    dt_s = _require_positive_finite_float(
        bev.get("future_dt_s"), field="base config bev.future_dt_s"
    )
    if not math.isclose(dt_s, 0.2, rel_tol=0.0, abs_tol=1e-12):
        raise RiskDataContractError("SOP08 future_dt_s must equal 0.2")
    if grid.future_steps != 15:
        raise RiskDataContractError("SOP08 future_steps must equal 15")
    length = _require_positive_finite_float(
        robot.get("length_m"), field="base config robot.length_m"
    )
    width = _require_positive_finite_float(
        robot.get("width_m"), field="base config robot.width_m"
    )
    inflation_value = robot.get("inflation_m")
    if type(inflation_value) not in {int, float}:
        raise RiskDataContractError("base config robot.inflation_m must be finite")
    inflation = float(inflation_value)
    if not math.isfinite(inflation) or inflation < 0.0:
        raise RiskDataContractError(
            "base config robot.inflation_m must be finite and non-negative"
        )
    if not (
        math.isclose(length, 0.70, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(width, 0.55, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(inflation, 0.15, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise RiskDataContractError(
            "SOP08 frozen robot query footprint must be 0.70x0.55m + 0.15m inflation"
        )
    return {
        "query_layout_version": "constant_control_robot_endpoint_query_v1",
        "base_config_digest": _require_blake2b128(
            base_config_digest, field="base_config_digest"
        ),
        "robot_model": "differential_drive",
        "robot_length_m": length,
        "robot_width_m": width,
        "robot_inflation_m": inflation,
        "future_steps": grid.future_steps,
        "future_dt_s": dt_s,
    }


def _validate_query_geometry(value: object, *, grid: GridSpec) -> dict[str, object]:
    keys = {
        "query_layout_version",
        "base_config_digest",
        "robot_model",
        "robot_length_m",
        "robot_width_m",
        "robot_inflation_m",
        "future_steps",
        "future_dt_s",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RiskDataContractError("occupancy query_geometry fields mismatch")
    if value.get("query_layout_version") != (
        "constant_control_robot_endpoint_query_v1"
    ):
        raise RiskDataContractError("occupancy query geometry layout mismatch")
    if value.get("robot_model") != "differential_drive":
        raise RiskDataContractError("occupancy query robot model mismatch")
    digest = _require_blake2b128(
        value.get("base_config_digest"), field="query_geometry.base_config_digest"
    )
    length = _require_positive_finite_float(
        value.get("robot_length_m"), field="query_geometry.robot_length_m"
    )
    width = _require_positive_finite_float(
        value.get("robot_width_m"), field="query_geometry.robot_width_m"
    )
    inflation_value = value.get("robot_inflation_m")
    if type(inflation_value) not in {int, float}:
        raise RiskDataContractError(
            "query_geometry.robot_inflation_m must be finite and non-negative"
        )
    inflation = float(inflation_value)
    if not math.isfinite(inflation) or inflation < 0.0:
        raise RiskDataContractError(
            "query_geometry.robot_inflation_m must be finite and non-negative"
        )
    if not (
        math.isclose(length, 0.70, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(width, 0.55, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(inflation, 0.15, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise RiskDataContractError("occupancy frozen robot footprint mismatch")
    steps = _require_positive_int(
        value.get("future_steps"), field="query_geometry.future_steps"
    )
    dt_s = _require_positive_finite_float(
        value.get("future_dt_s"), field="query_geometry.future_dt_s"
    )
    if steps != grid.future_steps or steps != 15 or not math.isclose(
        dt_s, 0.2, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RiskDataContractError("occupancy query time grid mismatch")
    return {
        "query_layout_version": "constant_control_robot_endpoint_query_v1",
        "base_config_digest": digest,
        "robot_model": "differential_drive",
        "robot_length_m": length,
        "robot_width_m": width,
        "robot_inflation_m": inflation,
        "future_steps": steps,
        "future_dt_s": dt_s,
    }


def _sidecar_descriptor_from_mapping(
    value: object, *, position: int
) -> SidecarShardDescriptor:
    if not isinstance(value, Mapping) or set(value) != _SIDECAR_DESCRIPTOR_KEYS:
        raise RiskDataContractError(
            f"occupancy sidecar descriptor {position} fields mismatch"
        )
    shard_index = _require_nonnegative_int(
        value.get("shard_index"), field=f"occupancy shards[{position}].shard_index"
    )
    if shard_index != position:
        raise RiskDataContractError(
            "occupancy sidecar shard indices must be contiguous from zero"
        )
    relative_root = _require_nonempty_string(
        value.get("relative_root"), field=f"occupancy shards[{position}].relative_root"
    )
    expected_root = f"shard-{position:05d}"
    if relative_root != expected_root:
        raise RiskDataContractError("occupancy sidecar relative_root ordering mismatch")
    marker_relative_path = _require_nonempty_string(
        value.get("marker_relative_path"),
        field=f"occupancy shards[{position}].marker_relative_path",
    )
    expected_marker = f"{expected_root}.risk-sidecar-pair-complete.json"
    if marker_relative_path != expected_marker:
        raise RiskDataContractError("occupancy sidecar marker path mismatch")
    return SidecarShardDescriptor(
        shard_index=shard_index,
        relative_root=relative_root,
        marker_relative_path=marker_relative_path,
        sample_count=_require_positive_int(
            value.get("sample_count"),
            field=f"occupancy shards[{position}].sample_count",
        ),
        sidecar_semantic_digest=_require_sha256(
            value.get("sidecar_semantic_digest"),
            field=f"occupancy shards[{position}].sidecar_semantic_digest",
        ),
        source_risk_shard_semantic_digest=_require_sha256(
            value.get("source_risk_shard_semantic_digest"),
            field=(
                f"occupancy shards[{position}].source_risk_shard_semantic_digest"
            ),
        ),
        pair_marker_digest_sha256=_require_sha256(
            value.get("pair_marker_digest_sha256"),
            field=f"occupancy shards[{position}].pair_marker_digest_sha256",
        ),
        ordered_sample_ids_digest_sha256=_require_sha256(
            value.get("ordered_sample_ids_digest_sha256"),
            field=(
                f"occupancy shards[{position}].ordered_sample_ids_digest_sha256"
            ),
        ),
    )


def _sidecar_collection_digest(payload: Mapping[str, object]) -> str:
    projection = {
        key: value
        for key, value in payload.items()
        if key != "collection_digest_sha256"
    }
    return _sha256_bytes(_canonical_json(projection).encode("utf-8"))


def _validate_sidecar_collection_section(
    value: object,
    *,
    grid: GridSpec,
    base_risk_dataset_manifest_digest: str,
) -> tuple[tuple[SidecarShardDescriptor, ...], dict[str, object], str]:
    if not isinstance(value, Mapping) or set(value) != _SIDECAR_COLLECTION_KEYS:
        raise RiskDataContractError("occupancy_sidecars section fields mismatch")
    if value.get("collection_layout_version") != (
        RISK_SIDECAR_COLLECTION_LAYOUT_VERSION
    ):
        raise RiskDataContractError("occupancy sidecar collection layout mismatch")
    if value.get("sidecar_shard_layout_version") != (
        RISK_SIDECAR_SHARD_LAYOUT_VERSION
    ):
        raise RiskDataContractError("occupancy sidecar shard layout mismatch")
    if value.get("pair_completion_marker_version") != (
        RISK_SIDECAR_PAIR_COMPLETION_MARKER_VERSION
    ):
        raise RiskDataContractError("occupancy pair marker version mismatch")
    if _require_sha256(
        value.get("base_risk_dataset_manifest_digest"),
        field="occupancy base risk dataset digest",
    ) != base_risk_dataset_manifest_digest:
        raise RiskDataContractError("occupancy sidecars base risk digest mismatch")
    base_config_digest = _require_blake2b128(
        value.get("base_config_digest"), field="occupancy base_config_digest"
    )
    query_geometry = _validate_query_geometry(value.get("query_geometry"), grid=grid)
    if query_geometry["base_config_digest"] != base_config_digest:
        raise RiskDataContractError("occupancy query/base config digest mismatch")
    sample_count = _require_positive_int(
        value.get("sample_count"), field="occupancy sidecar sample_count"
    )
    shard_count = _require_positive_int(
        value.get("shard_count"), field="occupancy sidecar shard_count"
    )
    raw_shards = value.get("shards")
    if not isinstance(raw_shards, (list, tuple)) or len(raw_shards) != shard_count:
        raise RiskDataContractError("occupancy sidecar shard descriptors are incomplete")
    descriptors = tuple(
        _sidecar_descriptor_from_mapping(item, position=index)
        for index, item in enumerate(raw_shards)
    )
    if sum(item.sample_count for item in descriptors) != sample_count:
        raise RiskDataContractError("occupancy sidecar sample count mismatch")
    declared_digest = _require_sha256(
        value.get("collection_digest_sha256"),
        field="occupancy sidecar collection digest",
    )
    if declared_digest != _sidecar_collection_digest(value):
        raise RiskDataContractError("occupancy sidecar collection digest mismatch")
    return descriptors, query_geometry, declared_digest


def _formally_validate_sidecar_collection(
    sidecar_root: Path,
    *,
    collection_root: Path,
    risk_descriptors: tuple[RiskShardDescriptor, ...],
    grid: GridSpec,
    expected_split: str,
    base_risk_dataset_manifest_digest: str,
    base_config_digest: str,
    query_geometry: Mapping[str, object],
    expected_section: Mapping[str, object] | None = None,
) -> LoadedRiskSidecarCollection:
    _require_directory(sidecar_root, label="occupancy sidecar collection root")
    expected_names = {
        name
        for descriptor in risk_descriptors
        for name in (
            descriptor.relative_root,
            f"{descriptor.relative_root}.risk-sidecar-pair-complete.json",
        )
    }
    try:
        entries = list(os.scandir(sidecar_root))
    except OSError as exc:
        raise RiskDataContractError("failed to enumerate occupancy sidecars") from exc
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        raise RiskDataContractError(
            "occupancy sidecar collection has missing/extra entries: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    if any(entry.is_symlink() for entry in entries):
        raise RiskDataContractError("occupancy sidecar collection forbids symlinks")

    sidecar_descriptors: list[SidecarShardDescriptor] = []
    for risk_descriptor in risk_descriptors:
        risk_root = collection_root / risk_descriptor.relative_root
        try:
            loaded_risk = load_risk_shard(
                risk_root, grid=grid, split_audit_records=()
            )
        except (OSError, TypeError, ValueError) as exc:
            raise RiskDataContractError(
                f"formal risk reload failed for occupancy join: {exc}"
            ) from exc
        if loaded_risk.semantic_digest != risk_descriptor.semantic_digest:
            raise RiskDataContractError("occupancy join risk shard digest mismatch")
        sample_ids = tuple(sample.sample_id for sample in loaded_risk.samples)
        sidecar_shard_root = sidecar_root / risk_descriptor.relative_root
        marker_path = risk_sidecar_pair_completion_marker_path(sidecar_shard_root)
        try:
            loaded_sidecar = load_risk_sidecar_shard(
                sidecar_shard_root,
                grid=grid,
                expected_sample_ids=sample_ids,
                expected_source_risk_shard_semantic_digest=(
                    risk_descriptor.semantic_digest
                ),
            )
            marker = load_risk_sidecar_pair_completion_marker(
                marker_path,
                expected_risk_root=risk_root,
                expected_sidecar_root=sidecar_shard_root,
                expected_split=expected_split,
                expected_shard_index=risk_descriptor.shard_index,
                expected_sample_ids=sample_ids,
                expected_risk_shard_semantic_digest=risk_descriptor.semantic_digest,
                expected_sidecar_shard_semantic_digest=(
                    loaded_sidecar.semantic_digest
                ),
            )
        except (OSError, TypeError, ValueError) as exc:
            raise RiskDataContractError(
                f"formal sidecar/marker load failed for {risk_descriptor.relative_root}: {exc}"
            ) from exc
        if (
            loaded_sidecar.split != expected_split
            or loaded_sidecar.shard_index != risk_descriptor.shard_index
            or loaded_sidecar.sample_ids != sample_ids
        ):
            raise RiskDataContractError("occupancy sidecar/risk ordered join mismatch")
        sidecar_descriptors.append(
            SidecarShardDescriptor(
                shard_index=risk_descriptor.shard_index,
                relative_root=risk_descriptor.relative_root,
                marker_relative_path=marker_path.name,
                sample_count=len(sample_ids),
                sidecar_semantic_digest=loaded_sidecar.semantic_digest,
                source_risk_shard_semantic_digest=(
                    loaded_sidecar.source_risk_shard_semantic_digest
                ),
                pair_marker_digest_sha256=marker.marker_digest_sha256,
                ordered_sample_ids_digest_sha256=(
                    marker.ordered_sample_ids_digest_sha256
                ),
            )
        )

    section: dict[str, object] = {
        "collection_layout_version": RISK_SIDECAR_COLLECTION_LAYOUT_VERSION,
        "sidecar_shard_layout_version": RISK_SIDECAR_SHARD_LAYOUT_VERSION,
        "pair_completion_marker_version": (
            RISK_SIDECAR_PAIR_COMPLETION_MARKER_VERSION
        ),
        "base_risk_dataset_manifest_digest": base_risk_dataset_manifest_digest,
        "base_config_digest": base_config_digest,
        "query_geometry": dict(query_geometry),
        "sample_count": sum(item.sample_count for item in sidecar_descriptors),
        "shard_count": len(sidecar_descriptors),
        "shards": [asdict(item) for item in sidecar_descriptors],
    }
    section["collection_digest_sha256"] = _sidecar_collection_digest(section)
    if expected_section is not None and _canonical_json(expected_section) != (
        _canonical_json(section)
    ):
        raise RiskDataContractError(
            "occupancy sidecar collection differs from authenticated seal"
        )
    return LoadedRiskSidecarCollection(
        root=sidecar_root,
        shards=tuple(sidecar_descriptors),
        sample_count=int(section["sample_count"]),
        collection_digest_sha256=str(section["collection_digest_sha256"]),
        query_geometry=_frozen_string_object_dict(dict(query_geometry)),
    )


def _target_policy_from_loaded_shard(
    loaded: LoadedRiskShard,
    *,
    shard_index: int,
) -> set[str]:
    values: set[str] = set()
    for row_index, row in enumerate(loaded.manifest):
        metadata = row.get("metadata")
        provenance = metadata.get("provenance") if isinstance(metadata, Mapping) else None
        if not isinstance(provenance, Mapping):
            raise RiskDataContractError(
                f"shard {shard_index} row {row_index} provenance is missing"
            )
        values.add(
            _require_blake2b128(
                provenance.get("target_type_policy_digest"),
                field="target_type_policy_digest",
            )
        )
    return values


def _heldout_collection_semantic_digest(
    handoff: Mapping[str, object],
    *,
    descriptors: list[RiskShardDescriptor],
    expected_split: str,
    sample_count: int,
) -> str:
    """Recompute the frozen heldout-finalizer framed semantic identity."""

    generation_evidence = handoff.get("generation_report_evidence")
    if not isinstance(generation_evidence, Mapping):
        raise RiskDataContractError(
            "heldout collection generation report evidence is missing"
        )
    rich_scope: dict[str, Mapping[str, object]] = {}
    for field in (
        "class_prior",
        "event_type_counts",
        "provenance_coverage",
        "per_sample_array_layout",
    ):
        value = handoff.get(field)
        if not isinstance(value, Mapping):
            raise RiskDataContractError(
                f"heldout collection semantic scope {field} must be a mapping"
            )
        rich_scope[field] = value
    scope = {
        "handoff_version": _HELDOUT_COLLECTION_HANDOFF_VERSION,
        "schema_version": SCHEMA_VERSION,
        "layout_version": RISK_SHARD_LAYOUT_VERSION,
        "split": expected_split,
        "shard_count": len(descriptors),
        "sample_count": sample_count,
        "generation_semantic_digest": _require_sha256(
            generation_evidence.get("semantic_digest_sha256"),
            field="heldout generation semantic digest",
        ),
        "class_prior": rich_scope["class_prior"],
        "event_type_counts": rich_scope["event_type_counts"],
        "provenance_coverage": rich_scope["provenance_coverage"],
        "per_sample_array_layout": rich_scope["per_sample_array_layout"],
        "shards": [asdict(descriptor) for descriptor in descriptors],
    }
    return _framed_domain_digest(
        _HELDOUT_COLLECTION_SEMANTIC_DOMAIN,
        scope,
        label="heldout collection semantic scope",
    )


def _formally_validate_collection(
    collection_root: Path,
    *,
    grid: GridSpec,
    expected_split: str,
    handoff: Mapping[str, object],
    authenticated_metadata_rows: list[dict[str, object]] | None = None,
) -> tuple[tuple[RiskShardDescriptor, ...], str]:
    shard_roots = _discover_shards(collection_root)
    handoff_values = handoff.get("shards")
    if not isinstance(handoff_values, list):
        raise RiskDataContractError("SOP07 handoff shards must be a list")
    if len(shard_roots) != len(handoff_values):
        raise RiskDataContractError("discovered shard count differs from handoff")

    descriptors: list[RiskShardDescriptor] = []
    target_policy_values: set[str] = set()
    sample_ids: set[str] = set()
    for position, (shard_root, handoff_value) in enumerate(
        zip(shard_roots, handoff_values)
    ):
        descriptor = _descriptor_from_mapping(
            handoff_value, position=position, exact_keys=False
        )
        if shard_root.name != descriptor.relative_root:
            raise RiskDataContractError("discovered shard order differs from handoff")
        _safe_shard_root(shard_root)
        try:
            loaded = load_risk_shard(
                shard_root,
                grid=grid,
                split_audit_records=(),
            )
        except (OSError, TypeError, ValueError) as exc:
            raise RiskDataContractError(
                f"formal load_risk_shard failed for {descriptor.relative_root}: {exc}"
            ) from exc
        summary = loaded.summary
        if (
            summary.get("schema_version") != SCHEMA_VERSION
            or summary.get("layout_version") != RISK_SHARD_LAYOUT_VERSION
            or summary.get("shard_index") != position
            or summary.get("split") != expected_split
            or len(loaded.samples) != descriptor.sample_count
            or summary.get("expected_sample_count") != descriptor.sample_count
        ):
            raise RiskDataContractError(
                f"formal shard summary mismatch for {descriptor.relative_root}"
            )
        if loaded.manifest_digest != descriptor.manifest_digest:
            raise RiskDataContractError(
                f"shard manifest digest mismatch for {descriptor.relative_root}"
            )
        if loaded.semantic_digest != descriptor.semantic_digest:
            raise RiskDataContractError(
                f"shard semantic digest mismatch for {descriptor.relative_root}"
            )
        file_digests = {
            "payload_sha256": _sha256_file(shard_root / "samples.npz"),
            "metadata_sha256": _sha256_file(shard_root / "metadata.jsonl"),
            "summary_sha256": _sha256_file(shard_root / "summary.json"),
        }
        for field, observed in file_digests.items():
            if getattr(descriptor, field) != observed:
                raise RiskDataContractError(
                    f"shard file checksum mismatch for {descriptor.relative_root}: {field}"
                )
        target_policy_values.update(
            _target_policy_from_loaded_shard(loaded, shard_index=position)
        )
        for sample in loaded.samples:
            if sample.sample_id in sample_ids:
                raise RiskDataContractError(
                    f"duplicate sample_id across SOP07 shards: {sample.sample_id}"
                )
            sample_ids.add(sample.sample_id)
        if authenticated_metadata_rows is not None:
            authenticated_metadata_rows.extend(
                dict(row) for row in loaded.manifest
            )
        descriptors.append(descriptor)

    expected_count = _require_positive_int(
        handoff.get("sample_count"), field="collection sample_count"
    )
    if sum(item.sample_count for item in descriptors) != expected_count:
        raise RiskDataContractError("collection sample_count differs from shard totals")
    if len(sample_ids) != expected_count:
        raise RiskDataContractError("collection global sample identity count mismatch")
    if (
        authenticated_metadata_rows is not None
        and len(authenticated_metadata_rows) != expected_count
    ):
        raise RiskDataContractError(
            "collection authenticated metadata row count mismatch"
        )
    if len(target_policy_values) != 1:
        raise RiskDataContractError(
            "collection must contain exactly one consistent target_type_policy_digest"
        )
    declared_collection_digest = _require_sha256(
        handoff.get("collection_semantic_digest_sha256"),
        field="collection_semantic_digest_sha256",
    )
    if expected_split in _HELDOUT_SPLITS:
        recomputed_collection_digest = _heldout_collection_semantic_digest(
            handoff,
            descriptors=descriptors,
            expected_split=expected_split,
            sample_count=len(sample_ids),
        )
        if recomputed_collection_digest != declared_collection_digest:
            raise RiskDataContractError("collection semantic digest mismatch")
    elif expected_split == "train":
        # The accepted train producer did not publish its digest construction or
        # builder evidence.  Treat this value as opaque producer provenance
        # rather than inventing a formula: publication authenticity comes from
        # the caller-pinned expected handoff SHA-256, which the immutable seal
        # persists and rechecks.  Collection integrity remains independently
        # enforced above by full formal shard reload, ordered descriptors, file
        # checksums, semantic digests, sample counts, and global sample IDs.
        pass
    else:  # pragma: no cover - handoff dialect validation rejects this earlier
        raise RiskDataContractError("unsupported SOP07 collection split")
    return tuple(descriptors), next(iter(target_policy_values))


def _semantic_manifest_projection(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"risk_dataset_manifest_digest", "occupancy_sidecars"}
        and key not in _RUNTIME_IDENTITY_FIELDS
    }


def _risk_dataset_manifest_digest(manifest: Mapping[str, object]) -> str:
    try:
        payload = _canonical_json(_semantic_manifest_projection(manifest)).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise RiskDataContractError(
            "dataset manifest semantics must be finite canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def validate_risk_dataset_manifest(manifest: Mapping[str, object]) -> str:
    """Authenticate one complete in-memory ``risk_dataset_v2`` manifest."""

    if not isinstance(manifest, Mapping):
        raise RiskDataContractError("dataset manifest must be a mapping")
    actual_keys = frozenset(manifest)
    if actual_keys not in {
        _DATASET_MANIFEST_KEYS,
        _DATASET_MANIFEST_WITH_SIDECARS_KEYS,
    }:
        missing_keys = sorted(_DATASET_MANIFEST_KEYS - actual_keys)
        extra_keys = sorted(actual_keys - _DATASET_MANIFEST_WITH_SIDECARS_KEYS)
        raise RiskDataContractError(
            "dataset manifest keys mismatch: "
            f"missing={missing_keys}, unexpected={extra_keys}"
        )
    declared_digest = _require_sha256(
        manifest.get("risk_dataset_manifest_digest"),
        field="risk_dataset_manifest_digest",
    )
    recomputed_digest = _risk_dataset_manifest_digest(manifest)
    if recomputed_digest != declared_digest:
        raise RiskDataContractError("risk dataset manifest digest mismatch")
    if "occupancy_sidecars" in manifest:
        _validate_sidecar_collection_section(
            manifest["occupancy_sidecars"],
            grid=_grid_from_manifest(manifest.get("grid")),
            base_risk_dataset_manifest_digest=declared_digest,
        )
    return declared_digest


def _write_dataset_seal(staging: Path, manifest: Mapping[str, object]) -> None:
    manifest_path = staging / _DATASET_MANIFEST_NAME
    marker_path = staging / _COMPLETE_MARKER_NAME
    checksums_path = staging / _CHECKSUMS_NAME
    manifest_path.write_bytes((_canonical_json(dict(manifest)) + "\n").encode("utf-8"))
    marker_path.write_bytes(b"")
    entries = {
        _COMPLETE_MARKER_NAME: _sha256_file(marker_path),
        _DATASET_MANIFEST_NAME: _sha256_file(manifest_path),
    }
    checksums_path.write_text(
        "".join(
            f"{entries[relative]}  {relative}\n" for relative in sorted(entries)
        ),
        encoding="utf-8",
    )


def publish_risk_dataset_seal(
    output_dir: str | Path,
    *,
    collection_root: str | Path,
    base_config_path: str | Path,
    split_provenance_path: str | Path,
    expected_split: str,
    expected_collection_handoff_sha256: str,
    sidecar_root: str | Path | None = None,
) -> Path:
    """Authenticate an accepted SOP07 collection and atomically publish its seal."""

    output_path = _absolute_without_symlink_resolution(Path(output_dir))
    collection_path = _absolute_without_symlink_resolution(Path(collection_root))
    config_path = _absolute_without_symlink_resolution(Path(base_config_path))
    provenance_path = _absolute_without_symlink_resolution(
        Path(split_provenance_path)
    )
    split = _require_nonempty_string(expected_split, field="expected_split")
    _assert_no_symlink_components(
        output_path, label="dataset seal output", allow_missing=True
    )
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite immutable risk dataset seal: {output_path}"
        )
    config, grid, grid_value = _load_base_config(config_path)
    g1_digest, dynamic_digest, sop03_commit, sop03_finalizer = (
        _load_authenticated_sop03_manifest(
            provenance_path,
            base_config_path=config_path,
            base_config=config,
        )
    )
    handoff_digest = _require_sha256(
        expected_collection_handoff_sha256,
        field="expected_collection_handoff_sha256",
    )
    handoff = _load_authenticated_handoff(
        collection_path,
        expected_split=split,
        expected_sha256=handoff_digest,
    )
    formal_evidence = handoff.get("formal_loader_evidence")
    if isinstance(formal_evidence, Mapping) and (
        formal_evidence.get("config_sha256") != _sha256_file(config_path)
    ):
        raise RiskDataContractError(
            "SOP07 handoff formal-loader base config SHA-256 mismatch"
        )
    descriptors, target_digest = _formally_validate_collection(
        collection_path,
        grid=grid,
        expected_split=split,
        handoff=handoff,
    )
    _require_blake2b128(g1_digest, field="g1_split_manifest_digest")
    _require_blake2b128(target_digest, field="target_type_policy_digest")
    _require_sha256(dynamic_digest, field="dynamic_objects_config_digest")
    manifest: dict[str, object] = {
        "dataset_layout_version": RISK_DATASET_LAYOUT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "collection_handoff_version": handoff["handoff_version"],
        "collection_handoff_sha256": handoff_digest,
        "collection_artifact_role": handoff["artifact_role"],
        "collection_instance_digest_sha256": handoff[
            "collection_instance_digest_sha256"
        ],
        "collection_semantic_digest_sha256": handoff[
            "collection_semantic_digest_sha256"
        ],
        "collection_code_commit": handoff["code_commit"],
        "collection_producer_version": handoff["producer_version"],
        "sop03_code_commit": sop03_commit,
        "sop03_finalizer_commit": sop03_finalizer,
        "sample_count": handoff["sample_count"],
        "shard_count": len(descriptors),
        "grid": grid_value,
        "channel_spec": _expected_channel_spec(),
        "shards": [asdict(descriptor) for descriptor in descriptors],
        "g1_split_manifest_digest": g1_digest,
        "dynamic_objects_config_digest": dynamic_digest,
        "target_type_policy_digest": target_digest,
    }
    dataset_digest = _risk_dataset_manifest_digest(manifest)
    manifest["risk_dataset_manifest_digest"] = dataset_digest
    sidecar_path: Path | None = None
    if sidecar_root is not None:
        sidecar_path = _absolute_without_symlink_resolution(Path(sidecar_root))
        base_config_digest = config_digest(config)
        query_geometry = _query_geometry_from_config(
            config,
            base_config_digest=base_config_digest,
            grid=grid,
        )
        loaded_sidecars = _formally_validate_sidecar_collection(
            sidecar_path,
            collection_root=collection_path,
            risk_descriptors=descriptors,
            grid=grid,
            expected_split=split,
            base_risk_dataset_manifest_digest=dataset_digest,
            base_config_digest=base_config_digest,
            query_geometry=query_geometry,
        )
        manifest["occupancy_sidecars"] = {
            "collection_layout_version": RISK_SIDECAR_COLLECTION_LAYOUT_VERSION,
            "sidecar_shard_layout_version": RISK_SIDECAR_SHARD_LAYOUT_VERSION,
            "pair_completion_marker_version": (
                RISK_SIDECAR_PAIR_COMPLETION_MARKER_VERSION
            ),
            "base_risk_dataset_manifest_digest": dataset_digest,
            "base_config_digest": base_config_digest,
            "query_geometry": query_geometry,
            "sample_count": loaded_sidecars.sample_count,
            "shard_count": len(loaded_sidecars.shards),
            "shards": [asdict(item) for item in loaded_sidecars.shards],
            "collection_digest_sha256": (
                loaded_sidecars.collection_digest_sha256
            ),
        }
        # Sidecars are an independent label publication.  They must not alter
        # the already-authenticated base risk dataset semantic identity.
        if _risk_dataset_manifest_digest(manifest) != dataset_digest:
            raise RuntimeError("occupancy sidecars changed base risk dataset digest")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(output_path.parent, label="dataset seal output parent")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.staging-",
            dir=output_path.parent,
        )
    )
    try:
        _write_dataset_seal(staging, manifest)
        reloaded = load_risk_dataset_seal(
            staging,
            collection_root=collection_path,
            expected_split=split,
            expected_manifest_digest=dataset_digest,
            sidecar_root=sidecar_path,
        )
        if (
            reloaded.sample_count != handoff["sample_count"]
            or reloaded.shards != descriptors
            or reloaded.risk_dataset_manifest_digest != dataset_digest
        ):
            raise RiskDataContractError("formal staging seal reload mismatch")
        _atomic_rename_directory_noreplace(staging, output_path)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return output_path


def _load_seal_manifest(seal_root: Path) -> dict[str, object]:
    _require_directory(seal_root, label="dataset-level v2 seal root")
    try:
        entries = list(os.scandir(seal_root))
    except OSError as exc:
        raise RiskDataContractError("failed to enumerate dataset-level v2 seal") from exc
    actual_names = {entry.name for entry in entries}
    missing = _REQUIRED_SEAL_FILES - actual_names
    unexpected = actual_names - _REQUIRED_SEAL_FILES
    if missing:
        raise RiskDataContractError(
            "incomplete dataset-level v2 seal: missing " + ", ".join(sorted(missing))
        )
    if unexpected:
        raise RiskDataContractError(
            "unexpected dataset seal files: " + ", ".join(sorted(unexpected))
        )
    for entry in entries:
        if entry.is_symlink():
            raise RiskDataContractError(
                f"dataset seal contains a forbidden symlink: {entry.name}"
            )
        if not entry.is_file(follow_symlinks=False):
            raise RiskDataContractError(
                f"dataset seal entry must be a regular file: {entry.name}"
            )
    marker = seal_root / _COMPLETE_MARKER_NAME
    if marker.read_bytes() != b"":
        raise RiskDataContractError("dataset seal completion marker must be empty")
    checksum_entries = _parse_checksum_manifest(
        seal_root / _CHECKSUMS_NAME, label="dataset seal checksum manifest"
    )
    expected_checksum_entries = {_COMPLETE_MARKER_NAME, _DATASET_MANIFEST_NAME}
    if set(checksum_entries) != expected_checksum_entries:
        raise RiskDataContractError("dataset seal checksum coverage is not exact")
    for relative in sorted(expected_checksum_entries):
        if checksum_entries[relative] != _sha256_file(seal_root / relative):
            raise RiskDataContractError(
                f"dataset seal checksum mismatch for {relative}"
            )
    manifest_path = seal_root / _DATASET_MANIFEST_NAME
    try:
        raw = manifest_path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise RiskDataContractError("failed to read dataset manifest") from exc
    manifest = _strict_json_loads(text, label="dataset manifest")
    expected_raw = (_canonical_json(manifest) + "\n").encode("utf-8")
    if raw != expected_raw:
        raise RiskDataContractError("dataset manifest is not canonical compact JSON")
    actual_keys = frozenset(manifest)
    if actual_keys not in {
        _DATASET_MANIFEST_KEYS,
        _DATASET_MANIFEST_WITH_SIDECARS_KEYS,
    }:
        missing_keys = sorted(_DATASET_MANIFEST_KEYS - actual_keys)
        extra_keys = sorted(actual_keys - _DATASET_MANIFEST_WITH_SIDECARS_KEYS)
        raise RiskDataContractError(
            "dataset manifest keys mismatch: "
            f"missing={missing_keys}, unexpected={extra_keys}"
        )
    return manifest


def _load_risk_dataset_seal_with_authenticated_rows(
    seal_root: str | Path,
    *,
    collection_root: str | Path,
    expected_split: str,
    expected_manifest_digest: str | None = None,
    sidecar_root: str | Path | None = None,
    retain_authenticated_metadata_rows: bool,
) -> tuple[LoadedRiskDataset, tuple[dict[str, object], ...]]:
    """Strictly load one dataset while retaining its authenticated metadata rows."""

    seal_path = _absolute_without_symlink_resolution(Path(seal_root))
    collection_path = _absolute_without_symlink_resolution(Path(collection_root))
    split = _require_nonempty_string(expected_split, field="expected_split")
    manifest = _load_seal_manifest(seal_path)
    if manifest.get("dataset_layout_version") != RISK_DATASET_LAYOUT_VERSION:
        raise RiskDataContractError(
            f"unsupported dataset layout; require {RISK_DATASET_LAYOUT_VERSION}"
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RiskDataContractError("dataset manifest schema_version mismatch")
    if manifest.get("split") != split:
        raise RiskDataContractError("dataset manifest split mismatch")
    if split == "train":
        expected_handoff_version = _COLLECTION_HANDOFF_VERSION
    elif split in _HELDOUT_SPLITS:
        expected_handoff_version = _HELDOUT_COLLECTION_HANDOFF_VERSION
    else:
        raise RiskDataContractError(
            "unsupported SOP07 collection split/handoff dialect"
        )
    if manifest.get("collection_handoff_version") != expected_handoff_version:
        raise RiskDataContractError("dataset collection handoff version mismatch")
    g1_digest = _require_blake2b128(
        manifest.get("g1_split_manifest_digest"), field="g1_split_manifest_digest"
    )
    target_digest = _require_blake2b128(
        manifest.get("target_type_policy_digest"),
        field="target_type_policy_digest",
    )
    dynamic_digest = _require_sha256(
        manifest.get("dynamic_objects_config_digest"),
        field="dynamic_objects_config_digest",
    )
    _validate_channel_spec(manifest.get("channel_spec"))
    grid = _grid_from_manifest(manifest.get("grid"))
    sample_count = _require_positive_int(
        manifest.get("sample_count"), field="dataset sample_count"
    )
    shard_count = _require_positive_int(
        manifest.get("shard_count"), field="dataset shard_count"
    )
    raw_descriptors = manifest.get("shards")
    if not isinstance(raw_descriptors, list) or len(raw_descriptors) != shard_count:
        raise RiskDataContractError("dataset shard_count differs from descriptors")
    descriptors = tuple(
        _descriptor_from_mapping(value, position=index, exact_keys=True)
        for index, value in enumerate(raw_descriptors)
    )
    if sum(item.sample_count for item in descriptors) != sample_count:
        raise RiskDataContractError("dataset sample_count differs from shard totals")
    dataset_digest = validate_risk_dataset_manifest(manifest)
    if expected_manifest_digest is not None:
        expected_digest = _require_sha256(
            expected_manifest_digest, field="expected_manifest_digest"
        )
        if dataset_digest != expected_digest:
            raise RiskDataContractError("expected manifest digest does not match seal")

    handoff_digest = _require_sha256(
        manifest.get("collection_handoff_sha256"),
        field="collection_handoff_sha256",
    )
    handoff = _load_authenticated_handoff(
        collection_path,
        expected_split=split,
        expected_sha256=handoff_digest,
    )
    if (
        handoff.get("handoff_version") != manifest["collection_handoff_version"]
        or handoff.get("artifact_role") != manifest["collection_artifact_role"]
        or handoff.get("collection_instance_digest_sha256")
        != manifest["collection_instance_digest_sha256"]
        or handoff.get("collection_semantic_digest_sha256")
        != manifest["collection_semantic_digest_sha256"]
        or handoff.get("code_commit") != manifest["collection_code_commit"]
        or handoff.get("producer_version") != manifest["collection_producer_version"]
        or handoff.get("sample_count") != sample_count
        or handoff.get("shard_count") != shard_count
    ):
        raise RiskDataContractError("dataset manifest source collection identity mismatch")
    authenticated_metadata_rows: list[dict[str, object]] | None = (
        [] if retain_authenticated_metadata_rows else None
    )
    loaded_descriptors, loaded_target_digest = _formally_validate_collection(
        collection_path,
        grid=grid,
        expected_split=split,
        handoff=handoff,
        authenticated_metadata_rows=authenticated_metadata_rows,
    )
    if loaded_descriptors != descriptors:
        raise RiskDataContractError("dataset shard descriptors differ from source handoff")
    if loaded_target_digest != target_digest:
        raise RiskDataContractError(
            "target_type_policy_digest differs from formally loaded shard provenance"
        )

    sidecar_section = manifest.get("occupancy_sidecars")
    if sidecar_section is not None:
        sidecar_descriptors, query_geometry, _ = (
            _validate_sidecar_collection_section(
                sidecar_section,
                grid=grid,
                base_risk_dataset_manifest_digest=dataset_digest,
            )
        )
        if (
            len(sidecar_descriptors) != shard_count
            or sum(item.sample_count for item in sidecar_descriptors) != sample_count
        ):
            raise RiskDataContractError(
                "occupancy sidecar collection differs from base risk shard counts"
            )
        if sidecar_root is not None:
            _formally_validate_sidecar_collection(
                _absolute_without_symlink_resolution(Path(sidecar_root)),
                collection_root=collection_path,
                risk_descriptors=descriptors,
                grid=grid,
                expected_split=split,
                base_risk_dataset_manifest_digest=dataset_digest,
                base_config_digest=str(sidecar_section["base_config_digest"]),
                query_geometry=query_geometry,
                expected_section=sidecar_section,
            )
    elif sidecar_root is not None:
        raise RiskDataContractError(
            "dataset seal does not contain an occupancy_sidecars publication"
        )

    provenance = {
        "g1_split_manifest_digest": g1_digest,
        "risk_dataset_manifest_digest": dataset_digest,
        "dynamic_objects_config_digest": dynamic_digest,
        "target_type_policy_digest": target_digest,
    }
    loaded_dataset = LoadedRiskDataset(
        seal_root=seal_path,
        collection_root=collection_path,
        manifest=_frozen_string_object_dict(manifest),
        grid=grid,
        shards=descriptors,
        split=split,
        sample_count=sample_count,
        risk_dataset_manifest_digest=dataset_digest,
        provenance=_frozen_string_string_dict(provenance),
    )
    return loaded_dataset, tuple(authenticated_metadata_rows or ())


def load_risk_dataset_seal(
    seal_root: str | Path,
    *,
    collection_root: str | Path,
    expected_split: str,
    expected_manifest_digest: str | None = None,
    sidecar_root: str | Path | None = None,
) -> LoadedRiskDataset:
    """Load only a complete v2 seal whose entire source collection still verifies."""

    loaded, _ = _load_risk_dataset_seal_with_authenticated_rows(
        seal_root,
        collection_root=collection_root,
        expected_split=expected_split,
        expected_manifest_digest=expected_manifest_digest,
        sidecar_root=sidecar_root,
        retain_authenticated_metadata_rows=False,
    )
    return loaded


def _canonical_mapping_copy(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RiskDataContractError(f"{label} must be a mapping")
    return _strict_json_loads(_canonical_json(value), label=label)


def _family_split_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(
        (left, right)
        for left_index, left in enumerate(_FAMILY_MEMBER_ORDER)
        for right in _FAMILY_MEMBER_ORDER[left_index + 1 :]
    )


def _family_common_contract(dataset: LoadedRiskDataset) -> dict[str, object]:
    return {
        "schema_version": dataset.manifest["schema_version"],
        "grid": _canonical_mapping_copy(
            dataset.manifest["grid"], label="family member grid"
        ),
        "channel_spec": _canonical_mapping_copy(
            dataset.manifest["channel_spec"],
            label="family member channel_spec",
        ),
        "g1_split_manifest_digest": dataset.provenance[
            "g1_split_manifest_digest"
        ],
        "dynamic_objects_config_digest": dataset.provenance[
            "dynamic_objects_config_digest"
        ],
        "target_type_policy_digest": dataset.provenance[
            "target_type_policy_digest"
        ],
    }


def _family_identity_sets(
    rows: tuple[dict[str, object], ...],
    *,
    split: str,
) -> dict[str, set[str]]:
    result = {
        field: set()
        for field in (
            "sample_id",
            "base_recording_id",
            "source_recording_id",
            "source_snippet_id",
            "pair_group_id",
            "seed_namespace",
            "base_session_id",
            "source_session_id",
        )
    }
    for row in rows:
        if row.get("split") != split:
            raise RiskDataContractError(
                f"family authenticated metadata row split mismatch for {split}"
            )
        for field in result:
            result[field].add(
                _require_nonempty_string(
                    row.get(field), field=f"family metadata {split}.{field}"
                )
            )
    if len(result["sample_id"]) != len(rows):
        raise RiskDataContractError(
            f"duplicate sample_id within authenticated family member {split}"
        )
    return result


def _family_pair_overlap_values(
    identities: Mapping[str, Mapping[str, set[str]]],
    *,
    field: str,
    left: str,
    right: str,
) -> list[str]:
    if field == "base_source_cross_role_recording_id":
        values = (
            identities[left]["base_recording_id"]
            & identities[right]["source_recording_id"]
        ) | (
            identities[left]["source_recording_id"]
            & identities[right]["base_recording_id"]
        )
    elif field == "base_source_cross_role_session_id":
        values = (
            identities[left]["base_session_id"]
            & identities[right]["source_session_id"]
        ) | (
            identities[left]["source_session_id"]
            & identities[right]["base_session_id"]
        )
    else:
        values = identities[left][field] & identities[right][field]
    return sorted(values)


def _family_overlap_entry(
    identities: Mapping[str, Mapping[str, set[str]]],
    *,
    field: str,
) -> dict[str, object]:
    pair_entries: list[dict[str, object]] = []
    all_values: set[str] = set()
    for left, right in _family_split_pairs():
        values = _family_pair_overlap_values(
            identities,
            field=field,
            left=left,
            right=right,
        )
        all_values.update(values)
        pair_entries.append(
            {
                "left_split": left,
                "right_split": right,
                "overlap_count": len(values),
                "overlap_values": values,
            }
        )
    ordered_values = sorted(all_values)
    return {
        "overlap_count": len(ordered_values),
        "overlap_values": ordered_values,
        "split_pair_overlaps": pair_entries,
    }


def _build_family_cross_split_audit(
    identities: Mapping[str, Mapping[str, set[str]]],
    *,
    row_counts: Mapping[str, int],
) -> dict[str, object]:
    forbidden = {
        field: _family_overlap_entry(identities, field=field)
        for field in _FAMILY_FORBIDDEN_IDENTITY_FIELDS
    }
    for field in _FAMILY_FORBIDDEN_IDENTITY_FIELDS:
        entry = forbidden[field]
        if entry["overlap_count"]:
            values = entry["overlap_values"]
            assert isinstance(values, list)
            preview = ", ".join(values[:3])
            raise RiskDataContractError(
                f"forbidden cross-split {field} overlap: {preview}"
            )
    allowed = {
        field: _family_overlap_entry(identities, field=field)
        for field in _FAMILY_ALLOWED_IDENTITY_FIELDS
    }
    return {
        "policy_version": _FAMILY_AUDIT_POLICY_VERSION,
        "member_order": list(_FAMILY_MEMBER_ORDER),
        "authenticated_metadata_row_counts": {
            split: row_counts[split] for split in _FAMILY_MEMBER_ORDER
        },
        "forbidden_overlaps": forbidden,
        "allowed_overlaps": allowed,
        "evidence_completeness": "PROVEN",
        "global_sample_id_uniqueness": "PROVEN",
        "global_cross_split_leakage": "PROVEN",
    }


def _risk_dataset_family_digest(manifest: Mapping[str, object]) -> str:
    projection = {
        key: value
        for key, value in manifest.items()
        if key != "risk_dataset_family_digest"
    }
    try:
        payload = _canonical_json(projection).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RiskDataContractError(
            "dataset family manifest must be finite canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _validate_family_member_descriptor(
    value: object,
    *,
    split: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FAMILY_MEMBER_KEYS:
        raise RiskDataContractError(
            f"dataset family member descriptor keys mismatch for {split}"
        )
    if value.get("split") != split:
        raise RiskDataContractError(
            f"dataset family member descriptor split mismatch for {split}"
        )
    digest = _require_sha256(
        value.get("risk_dataset_manifest_digest"),
        field=f"members.{split}.risk_dataset_manifest_digest",
    )
    sample_count = _require_positive_int(
        value.get("sample_count"), field=f"members.{split}.sample_count"
    )
    shard_count = _require_positive_int(
        value.get("shard_count"), field=f"members.{split}.shard_count"
    )
    return {
        "split": split,
        "risk_dataset_manifest_digest": digest,
        "sample_count": sample_count,
        "shard_count": shard_count,
    }


def _validate_family_common_contract(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FAMILY_COMMON_CONTRACT_KEYS:
        raise RiskDataContractError("dataset family common_contract keys mismatch")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise RiskDataContractError("dataset family common schema_version mismatch")
    grid_value = _canonical_mapping_copy(
        value.get("grid"), label="dataset family common grid"
    )
    _grid_from_manifest(grid_value)
    channel_spec = _canonical_mapping_copy(
        value.get("channel_spec"),
        label="dataset family common channel_spec",
    )
    _validate_channel_spec(channel_spec)
    return {
        "schema_version": SCHEMA_VERSION,
        "grid": grid_value,
        "channel_spec": channel_spec,
        "g1_split_manifest_digest": _require_blake2b128(
            value.get("g1_split_manifest_digest"),
            field="common_contract.g1_split_manifest_digest",
        ),
        "dynamic_objects_config_digest": _require_sha256(
            value.get("dynamic_objects_config_digest"),
            field="common_contract.dynamic_objects_config_digest",
        ),
        "target_type_policy_digest": _require_blake2b128(
            value.get("target_type_policy_digest"),
            field="common_contract.target_type_policy_digest",
        ),
    }


def _validate_family_overlap_entry(
    value: object,
    *,
    field: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FAMILY_OVERLAP_ENTRY_KEYS:
        raise RiskDataContractError(
            f"dataset family overlap audit keys mismatch for {field}"
        )
    overlap_count = _require_nonnegative_int(
        value.get("overlap_count"), field=f"{field}.overlap_count"
    )
    overlap_values = value.get("overlap_values")
    if (
        not isinstance(overlap_values, list)
        or any(not isinstance(item, str) or not item for item in overlap_values)
        or overlap_values != sorted(set(overlap_values))
        or len(overlap_values) != overlap_count
    ):
        raise RiskDataContractError(
            f"dataset family overlap values are non-canonical for {field}"
        )
    raw_pairs = value.get("split_pair_overlaps")
    expected_pairs = _family_split_pairs()
    if not isinstance(raw_pairs, list) or len(raw_pairs) != len(expected_pairs):
        raise RiskDataContractError(
            f"dataset family split-pair audit coverage mismatch for {field}"
        )
    normalized_pairs: list[dict[str, object]] = []
    union: set[str] = set()
    for raw, (expected_left, expected_right) in zip(raw_pairs, expected_pairs):
        if (
            not isinstance(raw, Mapping)
            or set(raw) != _FAMILY_SPLIT_PAIR_OVERLAP_KEYS
            or raw.get("left_split") != expected_left
            or raw.get("right_split") != expected_right
        ):
            raise RiskDataContractError(
                f"dataset family split-pair audit order mismatch for {field}"
            )
        pair_count = _require_nonnegative_int(
            raw.get("overlap_count"),
            field=f"{field}.{expected_left}.{expected_right}.overlap_count",
        )
        pair_values = raw.get("overlap_values")
        if (
            not isinstance(pair_values, list)
            or any(not isinstance(item, str) or not item for item in pair_values)
            or pair_values != sorted(set(pair_values))
            or len(pair_values) != pair_count
        ):
            raise RiskDataContractError(
                f"dataset family split-pair values are non-canonical for {field}"
            )
        union.update(pair_values)
        normalized_pairs.append(
            {
                "left_split": expected_left,
                "right_split": expected_right,
                "overlap_count": pair_count,
                "overlap_values": list(pair_values),
            }
        )
    if sorted(union) != list(overlap_values):
        raise RiskDataContractError(
            f"dataset family aggregate overlap values mismatch for {field}"
        )
    return {
        "overlap_count": overlap_count,
        "overlap_values": list(overlap_values),
        "split_pair_overlaps": normalized_pairs,
    }


def _validate_family_cross_split_audit(
    value: object,
    *,
    members: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FAMILY_AUDIT_KEYS:
        raise RiskDataContractError("dataset family cross_split_audit keys mismatch")
    if value.get("policy_version") != _FAMILY_AUDIT_POLICY_VERSION:
        raise RiskDataContractError("dataset family audit policy_version mismatch")
    member_order = value.get("member_order")
    if not isinstance(member_order, list) or tuple(member_order) != _FAMILY_MEMBER_ORDER:
        raise RiskDataContractError("dataset family audit member_order mismatch")
    raw_counts = value.get("authenticated_metadata_row_counts")
    if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(
        _FAMILY_MEMBER_ORDER
    ):
        raise RiskDataContractError(
            "dataset family authenticated metadata row-count coverage mismatch"
        )
    row_counts: dict[str, int] = {}
    for split in _FAMILY_MEMBER_ORDER:
        count = _require_positive_int(
            raw_counts.get(split),
            field=f"authenticated_metadata_row_counts.{split}",
        )
        if count != members[split]["sample_count"]:
            raise RiskDataContractError(
                f"dataset family authenticated row count differs for {split}"
            )
        row_counts[split] = count

    raw_forbidden = value.get("forbidden_overlaps")
    if not isinstance(raw_forbidden, Mapping) or set(raw_forbidden) != set(
        _FAMILY_FORBIDDEN_IDENTITY_FIELDS
    ):
        raise RiskDataContractError("dataset family forbidden overlap coverage mismatch")
    forbidden = {
        field: _validate_family_overlap_entry(raw_forbidden[field], field=field)
        for field in _FAMILY_FORBIDDEN_IDENTITY_FIELDS
    }
    for field, entry in forbidden.items():
        if entry["overlap_count"] != 0:
            raise RiskDataContractError(
                f"dataset family forbidden overlap is nonzero for {field}"
            )

    raw_allowed = value.get("allowed_overlaps")
    if not isinstance(raw_allowed, Mapping) or set(raw_allowed) != set(
        _FAMILY_ALLOWED_IDENTITY_FIELDS
    ):
        raise RiskDataContractError("dataset family allowed overlap coverage mismatch")
    allowed = {
        field: _validate_family_overlap_entry(raw_allowed[field], field=field)
        for field in _FAMILY_ALLOWED_IDENTITY_FIELDS
    }
    if value.get("evidence_completeness") != "PROVEN":
        raise RiskDataContractError(
            "dataset family cross-split evidence is not complete"
        )
    if value.get("global_sample_id_uniqueness") != "PROVEN":
        raise RiskDataContractError(
            "dataset family global sample_id uniqueness is not proven"
        )
    if value.get("global_cross_split_leakage") != "PROVEN":
        raise RiskDataContractError(
            "dataset family global cross-split leakage is not proven"
        )
    return {
        "policy_version": _FAMILY_AUDIT_POLICY_VERSION,
        "member_order": list(_FAMILY_MEMBER_ORDER),
        "authenticated_metadata_row_counts": row_counts,
        "forbidden_overlaps": forbidden,
        "allowed_overlaps": allowed,
        "evidence_completeness": "PROVEN",
        "global_sample_id_uniqueness": "PROVEN",
        "global_cross_split_leakage": "PROVEN",
    }


def _validate_risk_dataset_family_manifest(
    manifest: Mapping[str, object],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, object],
    str,
]:
    if not isinstance(manifest, Mapping) or set(manifest) != _FAMILY_MANIFEST_KEYS:
        raise RiskDataContractError("dataset family manifest keys mismatch")
    if manifest.get("dataset_family_layout_version") != (
        RISK_DATASET_FAMILY_LAYOUT_VERSION
    ):
        raise RiskDataContractError(
            f"unsupported dataset family layout; require {RISK_DATASET_FAMILY_LAYOUT_VERSION}"
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RiskDataContractError("dataset family schema_version mismatch")
    member_order = manifest.get("member_order")
    if not isinstance(member_order, list) or tuple(member_order) != _FAMILY_MEMBER_ORDER:
        raise RiskDataContractError("dataset family member_order mismatch")
    raw_members = manifest.get("members")
    if not isinstance(raw_members, Mapping) or set(raw_members) != set(
        _FAMILY_MEMBER_ORDER
    ):
        raise RiskDataContractError(
            "dataset family members must be exactly train, calibration, val, test"
        )
    members = {
        split: _validate_family_member_descriptor(raw_members[split], split=split)
        for split in _FAMILY_MEMBER_ORDER
    }
    common_contract = _validate_family_common_contract(
        manifest.get("common_contract")
    )
    if common_contract["schema_version"] != manifest["schema_version"]:
        raise RiskDataContractError(
            "dataset family top-level/common schema_version mismatch"
        )
    cross_split_audit = _validate_family_cross_split_audit(
        manifest.get("cross_split_audit"), members=members
    )
    declared_digest = _require_sha256(
        manifest.get("risk_dataset_family_digest"),
        field="risk_dataset_family_digest",
    )
    if _risk_dataset_family_digest(manifest) != declared_digest:
        raise RiskDataContractError("risk dataset family digest mismatch")
    return members, cross_split_audit, declared_digest


def _write_family_seal(staging: Path, manifest: Mapping[str, object]) -> None:
    manifest_path = staging / _FAMILY_MANIFEST_NAME
    marker_path = staging / _COMPLETE_MARKER_NAME
    checksums_path = staging / _CHECKSUMS_NAME
    manifest_path.write_bytes(
        (_canonical_json(dict(manifest)) + "\n").encode("utf-8")
    )
    marker_path.write_bytes(b"")
    entries = {
        _COMPLETE_MARKER_NAME: _sha256_file(marker_path),
        _FAMILY_MANIFEST_NAME: _sha256_file(manifest_path),
    }
    checksums_path.write_text(
        "".join(
            f"{entries[relative]}  {relative}\n" for relative in sorted(entries)
        ),
        encoding="utf-8",
    )


def _snapshot_family_files(root: Path) -> dict[str, bytes]:
    """Read each exact family member once below one pinned directory FD."""

    _require_directory(root, label="risk dataset family root")
    try:
        root_fd = os.open(
            root,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
    except OSError as exc:
        raise RiskDataContractError(
            "risk dataset family root must be a direct non-symlink directory"
        ) from exc
    try:
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            raise RiskDataContractError(
                "risk dataset family root must be a directory"
            )
        try:
            actual_names = set(os.listdir(root_fd))
        except OSError as exc:
            raise RiskDataContractError(
                "failed to enumerate pinned risk dataset family"
            ) from exc
        missing = _REQUIRED_FAMILY_FILES - actual_names
        unexpected = actual_names - _REQUIRED_FAMILY_FILES
        if missing:
            raise RiskDataContractError(
                "incomplete risk dataset family: missing "
                + ", ".join(sorted(missing))
            )
        if unexpected:
            raise RiskDataContractError(
                "unexpected risk dataset family files: "
                + ", ".join(sorted(unexpected))
            )
        snapshots: dict[str, bytes] = {}
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        for name in sorted(_REQUIRED_FAMILY_FILES):
            try:
                descriptor = os.open(name, flags, dir_fd=root_fd)
            except OSError as exc:
                raise RiskDataContractError(
                    "risk dataset family entry is missing or a forbidden "
                    f"symlink: {name}"
                ) from exc
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise RiskDataContractError(
                        "risk dataset family entry must be a regular file: "
                        f"{name}"
                    )
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 1 << 20)
                    if not chunk:
                        break
                    chunks.append(chunk)
                snapshots[name] = b"".join(chunks)
            finally:
                os.close(descriptor)
        return snapshots
    finally:
        os.close(root_fd)


def _load_family_manifest(root: Path) -> dict[str, object]:
    snapshots = _snapshot_family_files(root)
    marker = snapshots[_COMPLETE_MARKER_NAME]
    if marker != b"":
        raise RiskDataContractError(
            "risk dataset family completion marker must be empty"
        )
    checksum_entries = _parse_checksum_manifest_bytes(
        snapshots[_CHECKSUMS_NAME],
        label="risk dataset family checksum manifest",
    )
    expected_checksum_entries = {_COMPLETE_MARKER_NAME, _FAMILY_MANIFEST_NAME}
    if set(checksum_entries) != expected_checksum_entries:
        raise RiskDataContractError(
            "risk dataset family checksum coverage is not exact"
        )
    for relative in sorted(expected_checksum_entries):
        if checksum_entries[relative] != _sha256_bytes(snapshots[relative]):
            raise RiskDataContractError(
                f"risk dataset family checksum mismatch for {relative}"
            )
    raw = snapshots[_FAMILY_MANIFEST_NAME]
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise RiskDataContractError("failed to read dataset family manifest") from exc
    manifest = _strict_json_loads(text, label="dataset family manifest")
    if raw != (_canonical_json(manifest) + "\n").encode("utf-8"):
        raise RiskDataContractError(
            "dataset family manifest is not canonical compact JSON"
        )
    return manifest


def _publish_risk_dataset_family_from_references(
    output_dir: str | Path,
    *,
    member_references: Mapping[str, tuple[str | Path, str | Path, str]],
) -> Path:
    """Reauthenticate referenced members once and publish their family seal."""

    output_path = _absolute_without_symlink_resolution(Path(output_dir))
    _assert_no_symlink_components(
        output_path, label="risk dataset family output", allow_missing=True
    )
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite immutable risk dataset family: {output_path}"
        )
    if not isinstance(member_references, Mapping) or set(
        member_references
    ) != set(
        _FAMILY_MEMBER_ORDER
    ):
        raise RiskDataContractError(
            "risk dataset family members must be exactly train, calibration, val, test"
        )

    reloaded_members: dict[str, LoadedRiskDataset] = {}
    member_descriptors: dict[str, dict[str, object]] = {}
    identities: dict[str, dict[str, set[str]]] = {}
    row_counts: dict[str, int] = {}
    common_contract: dict[str, object] | None = None
    for split in _FAMILY_MEMBER_ORDER:
        reference = member_references[split]
        if not isinstance(reference, tuple) or len(reference) != 3:
            raise RiskDataContractError(
                f"family member reference {split} must contain seal root, "
                "collection root, and dataset digest"
            )
        seal_root, collection_root, expected_digest = reference
        reloaded, rows = _load_risk_dataset_seal_with_authenticated_rows(
            seal_root,
            collection_root=collection_root,
            expected_split=split,
            expected_manifest_digest=expected_digest,
            retain_authenticated_metadata_rows=True,
        )
        observed_contract = _family_common_contract(reloaded)
        if common_contract is None:
            common_contract = observed_contract
        elif observed_contract != common_contract:
            differing = next(
                key
                for key in _FAMILY_COMMON_CONTRACT_KEYS
                if observed_contract[key] != common_contract[key]
            )
            raise RiskDataContractError(
                f"risk dataset family common contract mismatch: {differing}"
            )
        reloaded_members[split] = reloaded
        row_counts[split] = len(rows)
        identities[split] = _family_identity_sets(rows, split=split)
        member_descriptors[split] = {
            "split": split,
            "risk_dataset_manifest_digest": (
                reloaded.risk_dataset_manifest_digest
            ),
            "sample_count": len(rows),
            "shard_count": len(reloaded.shards),
        }
    assert common_contract is not None
    audit = _build_family_cross_split_audit(identities, row_counts=row_counts)
    manifest: dict[str, object] = {
        "dataset_family_layout_version": RISK_DATASET_FAMILY_LAYOUT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "member_order": list(_FAMILY_MEMBER_ORDER),
        "members": member_descriptors,
        "common_contract": common_contract,
        "cross_split_audit": audit,
    }
    family_digest = _risk_dataset_family_digest(manifest)
    manifest["risk_dataset_family_digest"] = family_digest

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(
        output_path.parent, label="risk dataset family output parent"
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.staging-",
            dir=output_path.parent,
        )
    )
    try:
        _write_family_seal(staging, manifest)
        expected_digests = {
            split: reloaded_members[split].risk_dataset_manifest_digest
            for split in _FAMILY_MEMBER_ORDER
        }
        reloaded_family = load_risk_dataset_family(
            staging,
            expected_member_digests=expected_digests,
        )
        if reloaded_family.risk_dataset_family_digest != family_digest:
            raise RiskDataContractError("formal staging family reload mismatch")
        _atomic_rename_directory_noreplace(staging, output_path)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return output_path


def publish_risk_dataset_family(
    output_dir: str | Path,
    *,
    members: Mapping[str, LoadedRiskDataset],
) -> Path:
    """Reauthenticate and atomically publish exactly four split dataset seals."""

    if not isinstance(members, Mapping) or set(members) != set(
        _FAMILY_MEMBER_ORDER
    ):
        raise RiskDataContractError(
            "risk dataset family members must be exactly train, calibration, val, test"
        )
    references: dict[str, tuple[str | Path, str | Path, str]] = {}
    for split in _FAMILY_MEMBER_ORDER:
        member = members[split]
        if not isinstance(member, LoadedRiskDataset):
            raise RiskDataContractError(
                f"family member {split} must be a LoadedRiskDataset"
            )
        references[split] = (
            member.seal_root,
            member.collection_root,
            member.risk_dataset_manifest_digest,
        )
    return _publish_risk_dataset_family_from_references(
        output_dir,
        member_references=references,
    )


def load_risk_dataset_family(
    root: str | Path,
    *,
    expected_member_digests: Mapping[str, str] | None = None,
) -> LoadedRiskDatasetFamily:
    """Load one self-contained family seal without guessing collection roots."""

    root_path = _absolute_without_symlink_resolution(Path(root))
    manifest = _load_family_manifest(root_path)
    members, cross_split_audit, family_digest = (
        _validate_risk_dataset_family_manifest(manifest)
    )
    if expected_member_digests is not None:
        if not isinstance(expected_member_digests, Mapping) or set(
            expected_member_digests
        ) != set(_FAMILY_MEMBER_ORDER):
            raise RiskDataContractError(
                "expected_member_digests must exactly cover train, calibration, val, test"
            )
        for split in _FAMILY_MEMBER_ORDER:
            expected = _require_sha256(
                expected_member_digests[split],
                field=f"expected_member_digests.{split}",
            )
            if expected != members[split]["risk_dataset_manifest_digest"]:
                raise RiskDataContractError(
                    f"expected member digest mismatch for {split}"
                )
    ordered_members = {
        split: members[split] for split in _FAMILY_MEMBER_ORDER
    }
    return LoadedRiskDatasetFamily(
        root=root_path,
        manifest=_frozen_string_object_dict(manifest),
        members=_FrozenDict(ordered_members),
        cross_split_audit=_frozen_string_object_dict(cross_split_audit),
        risk_dataset_family_digest=family_digest,
    )


def _canonical_mapping_copy(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RiskDataContractError(f"{label} must be a mapping")
    return _strict_json_loads(_canonical_json(value), label=label)


def risk_dataset_family_sample_ids_digest(
    sample_ids: Sequence[str],
) -> str:
    """Hash one split's exact sample-ID membership independent of row order."""

    if isinstance(sample_ids, (str, bytes)) or not isinstance(
        sample_ids, Sequence
    ):
        raise RiskDataContractError("family sample IDs must be a sequence")
    values = list(sample_ids)
    if any(not isinstance(value, str) or not value for value in values):
        raise RiskDataContractError(
            "family sample IDs must be non-empty strings"
        )
    if len(set(values)) != len(values):
        raise RiskDataContractError("family sample IDs must be unique")
    payload = _canonical_json(sorted(values)).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"risk-dataset-family-sample-ids-v1\0")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def _family_split_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(
        (left, right)
        for left_index, left in enumerate(_FAMILY_MEMBER_ORDER)
        for right in _FAMILY_MEMBER_ORDER[left_index + 1 :]
    )


def _family_common_contract(dataset: LoadedRiskDataset) -> dict[str, object]:
    return {
        "schema_version": dataset.manifest["schema_version"],
        "grid": _canonical_mapping_copy(
            dataset.manifest["grid"], label="family member grid"
        ),
        "channel_spec": _canonical_mapping_copy(
            dataset.manifest["channel_spec"],
            label="family member channel_spec",
        ),
        "g1_split_manifest_digest": dataset.provenance[
            "g1_split_manifest_digest"
        ],
        "dynamic_objects_config_digest": dataset.provenance[
            "dynamic_objects_config_digest"
        ],
        "target_type_policy_digest": dataset.provenance[
            "target_type_policy_digest"
        ],
    }


def _authenticate_family_member_seal(
    member: LoadedRiskDataset | RiskDatasetFamilyMemberSource,
    *,
    split: str,
) -> LoadedRiskDataset:
    """Authenticate the immutable seal without reopening numeric shard payloads."""

    if isinstance(member, LoadedRiskDataset):
        seal_root = member.seal_root
        collection_root = member.collection_root
        pinned_digest = member.risk_dataset_manifest_digest
    elif isinstance(member, RiskDatasetFamilyMemberSource):
        seal_root = member.seal_root
        collection_root = member.collection_root
        pinned_digest = member.expected_manifest_digest
    else:
        raise RiskDataContractError(
            f"family member {split} must be a loaded dataset or explicit source"
        )
    seal_path = _absolute_without_symlink_resolution(Path(seal_root))
    collection_path = _absolute_without_symlink_resolution(Path(collection_root))
    _require_directory(collection_path, label=f"family {split} collection root")
    manifest = _load_seal_manifest(seal_path)
    if manifest.get("dataset_layout_version") != RISK_DATASET_LAYOUT_VERSION:
        raise RiskDataContractError("family member dataset layout mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RiskDataContractError("family member schema_version mismatch")
    if manifest.get("split") != split:
        raise RiskDataContractError(f"family member split mismatch for {split}")
    expected_handoff_version = (
        _COLLECTION_HANDOFF_VERSION
        if split == "train"
        else _HELDOUT_COLLECTION_HANDOFF_VERSION
    )
    if manifest.get("collection_handoff_version") != expected_handoff_version:
        raise RiskDataContractError(
            f"family member handoff version mismatch for {split}"
        )

    g1_digest = _require_blake2b128(
        manifest.get("g1_split_manifest_digest"),
        field=f"family {split} g1_split_manifest_digest",
    )
    target_digest = _require_blake2b128(
        manifest.get("target_type_policy_digest"),
        field=f"family {split} target_type_policy_digest",
    )
    dynamic_digest = _require_sha256(
        manifest.get("dynamic_objects_config_digest"),
        field=f"family {split} dynamic_objects_config_digest",
    )
    _validate_channel_spec(manifest.get("channel_spec"))
    grid = _grid_from_manifest(manifest.get("grid"))
    sample_count = _require_positive_int(
        manifest.get("sample_count"), field=f"family {split} sample_count"
    )
    shard_count = _require_positive_int(
        manifest.get("shard_count"), field=f"family {split} shard_count"
    )
    raw_descriptors = manifest.get("shards")
    if not isinstance(raw_descriptors, list) or len(raw_descriptors) != shard_count:
        raise RiskDataContractError(
            f"family {split} shard_count differs from descriptors"
        )
    descriptors = tuple(
        _descriptor_from_mapping(value, position=index, exact_keys=True)
        for index, value in enumerate(raw_descriptors)
    )
    if sum(item.sample_count for item in descriptors) != sample_count:
        raise RiskDataContractError(
            f"family {split} sample_count differs from shard totals"
        )
    dataset_digest = validate_risk_dataset_manifest(manifest)
    expected_digest = _require_sha256(
        pinned_digest,
        field=f"family {split} expected manifest digest",
    )
    if dataset_digest != expected_digest:
        raise RiskDataContractError(
            f"family member manifest digest mismatch for {split}"
        )
    return LoadedRiskDataset(
        seal_root=seal_path,
        collection_root=collection_path,
        manifest=_frozen_string_object_dict(manifest),
        grid=grid,
        shards=descriptors,
        split=split,
        sample_count=sample_count,
        risk_dataset_manifest_digest=dataset_digest,
        provenance=_frozen_string_string_dict(
            {
                "g1_split_manifest_digest": g1_digest,
                "risk_dataset_manifest_digest": dataset_digest,
                "dynamic_objects_config_digest": dynamic_digest,
                "target_type_policy_digest": target_digest,
            }
        ),
    )


def _load_family_member_metadata_rows(
    dataset: LoadedRiskDataset,
) -> tuple[dict[str, object], ...]:
    """Read only seal-authenticated JSONL metadata needed for leakage checks."""

    rows: list[dict[str, object]] = []
    for descriptor in dataset.shards:
        shard_root = dataset.collection_root / descriptor.relative_root
        _require_directory(
            shard_root,
            label=f"family {dataset.split}/{descriptor.relative_root} shard",
        )
        metadata_path = shard_root / "metadata.jsonl"
        _require_regular_file(
            metadata_path,
            label=f"family {dataset.split}/{descriptor.relative_root} metadata",
        )
        if _sha256_file(metadata_path) != descriptor.metadata_sha256:
            raise RiskDataContractError(
                "family member metadata SHA-256 differs from authenticated seal: "
                f"{dataset.split}/{descriptor.relative_root}"
            )
        try:
            raw = metadata_path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise RiskDataContractError(
                "family metadata read failed for "
                f"{dataset.split}/{descriptor.relative_root}: {exc}"
            ) from exc
        if not raw or not raw.endswith(b"\n"):
            raise RiskDataContractError(
                "family metadata must be non-empty newline-terminated JSONL: "
                f"{dataset.split}/{descriptor.relative_root}"
            )
        shard_rows = tuple(
            _strict_json_loads(
                line,
                label=(
                    f"family {dataset.split}/{descriptor.relative_root} "
                    f"metadata row {row_index}"
                ),
            )
            for row_index, line in enumerate(text.splitlines())
        )
        canonical = (
            "\n".join(_canonical_json(row) for row in shard_rows) + "\n"
        ).encode("utf-8")
        if canonical != raw:
            raise RiskDataContractError(
                "family metadata is not canonical compact JSONL: "
                f"{dataset.split}/{descriptor.relative_root}"
            )
        digest = hashlib.sha256()
        digest.update(b"risk-shard-manifest-v2\0")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        if digest.hexdigest() != descriptor.manifest_digest:
            raise RiskDataContractError(
                "family metadata manifest digest differs from authenticated seal: "
                f"{dataset.split}/{descriptor.relative_root}"
            )
        if len(shard_rows) != descriptor.sample_count:
            raise RiskDataContractError(
                "family metadata row count differs from authenticated seal: "
                f"{dataset.split}/{descriptor.relative_root}"
            )
        rows.extend(shard_rows)
    if len(rows) != dataset.sample_count:
        raise RiskDataContractError(
            f"family authenticated metadata row count mismatch for {dataset.split}"
        )
    return tuple(rows)


def _family_identity_sets(
    rows: tuple[dict[str, object], ...],
    *,
    split: str,
) -> dict[str, set[str]]:
    result = {
        field: set()
        for field in (
            "sample_id",
            "base_recording_id",
            "source_recording_id",
            "source_snippet_id",
            "pair_group_id",
            "seed_namespace",
            "base_session_id",
            "source_session_id",
        )
    }
    for row in rows:
        if row.get("split") != split:
            raise RiskDataContractError(
                f"family authenticated metadata row split mismatch for {split}"
            )
        for field in result:
            result[field].add(
                _require_nonempty_string(
                    row.get(field), field=f"family metadata {split}.{field}"
                )
            )
    if len(result["sample_id"]) != len(rows):
        raise RiskDataContractError(
            f"duplicate sample_id within authenticated family member {split}"
        )
    return result


def _family_pair_overlap_values(
    identities: Mapping[str, Mapping[str, set[str]]],
    *,
    field: str,
    left: str,
    right: str,
) -> list[str]:
    if field == "base_source_cross_role_recording_id":
        values = (
            identities[left]["base_recording_id"]
            & identities[right]["source_recording_id"]
        ) | (
            identities[left]["source_recording_id"]
            & identities[right]["base_recording_id"]
        )
    elif field == "base_source_cross_role_session_id":
        values = (
            identities[left]["base_session_id"]
            & identities[right]["source_session_id"]
        ) | (
            identities[left]["source_session_id"]
            & identities[right]["base_session_id"]
        )
    else:
        values = identities[left][field] & identities[right][field]
    return sorted(values)


def _family_overlap_entry(
    identities: Mapping[str, Mapping[str, set[str]]],
    *,
    field: str,
) -> dict[str, object]:
    pair_entries: list[dict[str, object]] = []
    all_values: set[str] = set()
    for left, right in _family_split_pairs():
        values = _family_pair_overlap_values(
            identities,
            field=field,
            left=left,
            right=right,
        )
        all_values.update(values)
        pair_entries.append(
            {
                "left_split": left,
                "right_split": right,
                "overlap_count": len(values),
                "overlap_values": values,
            }
        )
    ordered_values = sorted(all_values)
    return {
        "overlap_count": len(ordered_values),
        "overlap_values": ordered_values,
        "split_pair_overlaps": pair_entries,
    }


def _build_family_cross_split_audit(
    identities: Mapping[str, Mapping[str, set[str]]],
    *,
    row_counts: Mapping[str, int],
) -> dict[str, object]:
    forbidden = {
        field: _family_overlap_entry(identities, field=field)
        for field in _FAMILY_FORBIDDEN_IDENTITY_FIELDS
    }
    for field in _FAMILY_FORBIDDEN_IDENTITY_FIELDS:
        entry = forbidden[field]
        if entry["overlap_count"]:
            values = entry["overlap_values"]
            assert isinstance(values, list)
            preview = ", ".join(values[:3])
            raise RiskDataContractError(
                f"forbidden cross-split {field} overlap: {preview}"
            )
    allowed = {
        field: _family_overlap_entry(identities, field=field)
        for field in _FAMILY_ALLOWED_IDENTITY_FIELDS
    }
    return {
        "policy_version": _FAMILY_AUDIT_POLICY_VERSION,
        "member_order": list(_FAMILY_MEMBER_ORDER),
        "authenticated_metadata_row_counts": {
            split: row_counts[split] for split in _FAMILY_MEMBER_ORDER
        },
        "forbidden_overlaps": forbidden,
        "allowed_overlaps": allowed,
        "evidence_completeness": "PROVEN",
        "global_sample_id_uniqueness": "PROVEN",
        "global_cross_split_leakage": "PROVEN",
    }


def _production_evaluation_metadata() -> dict[str, object]:
    return {
        "layout_version": PRODUCTION_RISK_EVALUATION_METADATA_LAYOUT_VERSION,
        "training_split": "train",
        "selection_split": "val",
        "calibration_fit_split": "calibration",
        "evaluation_split": "test",
        "test_used_for_training_or_selection": False,
        "test_used_for_calibration": False,
        "calibration_statistics_scope": "calibration_only",
    }


def _validate_production_evaluation_metadata(
    value: object,
) -> dict[str, object]:
    expected = _production_evaluation_metadata()
    if not isinstance(value, Mapping) or set(value) != (
        _PRODUCTION_EVALUATION_METADATA_KEYS
    ):
        raise RiskDataContractError(
            "production evaluation metadata keys mismatch"
        )
    normalized = _canonical_mapping_copy(
        value,
        label="production evaluation metadata",
    )
    if normalized != expected:
        raise RiskDataContractError(
            "production evaluation metadata split policy mismatch"
        )
    return normalized


def _risk_dataset_family_digest(manifest: Mapping[str, object]) -> str:
    projection = {
        key: value
        for key, value in manifest.items()
        if key != "risk_dataset_family_digest"
    }
    try:
        payload = _canonical_json(projection).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RiskDataContractError(
            "dataset family manifest must be finite canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _validate_family_member_descriptor(
    value: object,
    *,
    split: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FAMILY_MEMBER_KEYS:
        raise RiskDataContractError(
            f"dataset family member descriptor keys mismatch for {split}"
        )
    if value.get("split") != split:
        raise RiskDataContractError(
            f"dataset family member descriptor split mismatch for {split}"
        )
    digest = _require_sha256(
        value.get("risk_dataset_manifest_digest"),
        field=f"members.{split}.risk_dataset_manifest_digest",
    )
    sample_count = _require_positive_int(
        value.get("sample_count"), field=f"members.{split}.sample_count"
    )
    sample_ids_digest = _require_sha256(
        value.get("sample_ids_digest_sha256"),
        field=f"members.{split}.sample_ids_digest_sha256",
    )
    shard_count = _require_positive_int(
        value.get("shard_count"), field=f"members.{split}.shard_count"
    )
    return {
        "split": split,
        "risk_dataset_manifest_digest": digest,
        "sample_count": sample_count,
        "sample_ids_digest_sha256": sample_ids_digest,
        "shard_count": shard_count,
    }


def _validate_family_common_contract(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FAMILY_COMMON_CONTRACT_KEYS:
        raise RiskDataContractError("dataset family common_contract keys mismatch")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise RiskDataContractError("dataset family common schema_version mismatch")
    grid_value = _canonical_mapping_copy(
        value.get("grid"), label="dataset family common grid"
    )
    _grid_from_manifest(grid_value)
    channel_spec = _canonical_mapping_copy(
        value.get("channel_spec"),
        label="dataset family common channel_spec",
    )
    _validate_channel_spec(channel_spec)
    return {
        "schema_version": SCHEMA_VERSION,
        "grid": grid_value,
        "channel_spec": channel_spec,
        "g1_split_manifest_digest": _require_blake2b128(
            value.get("g1_split_manifest_digest"),
            field="common_contract.g1_split_manifest_digest",
        ),
        "dynamic_objects_config_digest": _require_sha256(
            value.get("dynamic_objects_config_digest"),
            field="common_contract.dynamic_objects_config_digest",
        ),
        "target_type_policy_digest": _require_blake2b128(
            value.get("target_type_policy_digest"),
            field="common_contract.target_type_policy_digest",
        ),
    }


def _validate_family_overlap_entry(
    value: object,
    *,
    field: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FAMILY_OVERLAP_ENTRY_KEYS:
        raise RiskDataContractError(
            f"dataset family overlap audit keys mismatch for {field}"
        )
    overlap_count = _require_nonnegative_int(
        value.get("overlap_count"), field=f"{field}.overlap_count"
    )
    overlap_values = value.get("overlap_values")
    if (
        not isinstance(overlap_values, list)
        or any(not isinstance(item, str) or not item for item in overlap_values)
        or overlap_values != sorted(set(overlap_values))
        or len(overlap_values) != overlap_count
    ):
        raise RiskDataContractError(
            f"dataset family overlap values are non-canonical for {field}"
        )
    raw_pairs = value.get("split_pair_overlaps")
    expected_pairs = _family_split_pairs()
    if not isinstance(raw_pairs, list) or len(raw_pairs) != len(expected_pairs):
        raise RiskDataContractError(
            f"dataset family split-pair audit coverage mismatch for {field}"
        )
    normalized_pairs: list[dict[str, object]] = []
    union: set[str] = set()
    for raw, (expected_left, expected_right) in zip(raw_pairs, expected_pairs):
        if (
            not isinstance(raw, Mapping)
            or set(raw) != _FAMILY_SPLIT_PAIR_OVERLAP_KEYS
            or raw.get("left_split") != expected_left
            or raw.get("right_split") != expected_right
        ):
            raise RiskDataContractError(
                f"dataset family split-pair audit order mismatch for {field}"
            )
        pair_count = _require_nonnegative_int(
            raw.get("overlap_count"),
            field=f"{field}.{expected_left}.{expected_right}.overlap_count",
        )
        pair_values = raw.get("overlap_values")
        if (
            not isinstance(pair_values, list)
            or any(not isinstance(item, str) or not item for item in pair_values)
            or pair_values != sorted(set(pair_values))
            or len(pair_values) != pair_count
        ):
            raise RiskDataContractError(
                f"dataset family split-pair values are non-canonical for {field}"
            )
        union.update(pair_values)
        normalized_pairs.append(
            {
                "left_split": expected_left,
                "right_split": expected_right,
                "overlap_count": pair_count,
                "overlap_values": list(pair_values),
            }
        )
    if sorted(union) != list(overlap_values):
        raise RiskDataContractError(
            f"dataset family aggregate overlap values mismatch for {field}"
        )
    return {
        "overlap_count": overlap_count,
        "overlap_values": list(overlap_values),
        "split_pair_overlaps": normalized_pairs,
    }


def _validate_family_cross_split_audit(
    value: object,
    *,
    members: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FAMILY_AUDIT_KEYS:
        raise RiskDataContractError("dataset family cross_split_audit keys mismatch")
    if value.get("policy_version") != _FAMILY_AUDIT_POLICY_VERSION:
        raise RiskDataContractError("dataset family audit policy_version mismatch")
    member_order = value.get("member_order")
    if not isinstance(member_order, list) or tuple(member_order) != _FAMILY_MEMBER_ORDER:
        raise RiskDataContractError("dataset family audit member_order mismatch")
    raw_counts = value.get("authenticated_metadata_row_counts")
    if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(
        _FAMILY_MEMBER_ORDER
    ):
        raise RiskDataContractError(
            "dataset family authenticated metadata row-count coverage mismatch"
        )
    row_counts: dict[str, int] = {}
    for split in _FAMILY_MEMBER_ORDER:
        count = _require_positive_int(
            raw_counts.get(split),
            field=f"authenticated_metadata_row_counts.{split}",
        )
        if count != members[split]["sample_count"]:
            raise RiskDataContractError(
                f"dataset family authenticated row count differs for {split}"
            )
        row_counts[split] = count

    raw_forbidden = value.get("forbidden_overlaps")
    if not isinstance(raw_forbidden, Mapping) or set(raw_forbidden) != set(
        _FAMILY_FORBIDDEN_IDENTITY_FIELDS
    ):
        raise RiskDataContractError("dataset family forbidden overlap coverage mismatch")
    forbidden = {
        field: _validate_family_overlap_entry(raw_forbidden[field], field=field)
        for field in _FAMILY_FORBIDDEN_IDENTITY_FIELDS
    }
    for field, entry in forbidden.items():
        if entry["overlap_count"] != 0:
            raise RiskDataContractError(
                f"dataset family forbidden overlap is nonzero for {field}"
            )

    raw_allowed = value.get("allowed_overlaps")
    if not isinstance(raw_allowed, Mapping) or set(raw_allowed) != set(
        _FAMILY_ALLOWED_IDENTITY_FIELDS
    ):
        raise RiskDataContractError("dataset family allowed overlap coverage mismatch")
    allowed = {
        field: _validate_family_overlap_entry(raw_allowed[field], field=field)
        for field in _FAMILY_ALLOWED_IDENTITY_FIELDS
    }
    if value.get("evidence_completeness") != "PROVEN":
        raise RiskDataContractError(
            "dataset family cross-split evidence is not complete"
        )
    if value.get("global_sample_id_uniqueness") != "PROVEN":
        raise RiskDataContractError(
            "dataset family global sample_id uniqueness is not proven"
        )
    if value.get("global_cross_split_leakage") != "PROVEN":
        raise RiskDataContractError(
            "dataset family global cross-split leakage is not proven"
        )
    return {
        "policy_version": _FAMILY_AUDIT_POLICY_VERSION,
        "member_order": list(_FAMILY_MEMBER_ORDER),
        "authenticated_metadata_row_counts": row_counts,
        "forbidden_overlaps": forbidden,
        "allowed_overlaps": allowed,
        "evidence_completeness": "PROVEN",
        "global_sample_id_uniqueness": "PROVEN",
        "global_cross_split_leakage": "PROVEN",
    }


def _validate_risk_dataset_family_manifest(
    manifest: Mapping[str, object],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, object],
    str,
]:
    if not isinstance(manifest, Mapping) or set(manifest) != _FAMILY_MANIFEST_KEYS:
        raise RiskDataContractError("dataset family manifest keys mismatch")
    if manifest.get("dataset_family_layout_version") != (
        RISK_DATASET_FAMILY_LAYOUT_VERSION
    ):
        raise RiskDataContractError(
            f"unsupported dataset family layout; require {RISK_DATASET_FAMILY_LAYOUT_VERSION}"
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RiskDataContractError("dataset family schema_version mismatch")
    member_order = manifest.get("member_order")
    if not isinstance(member_order, list) or tuple(member_order) != _FAMILY_MEMBER_ORDER:
        raise RiskDataContractError("dataset family member_order mismatch")
    raw_members = manifest.get("members")
    if not isinstance(raw_members, Mapping) or set(raw_members) != set(
        _FAMILY_MEMBER_ORDER
    ):
        raise RiskDataContractError(
            "dataset family members must be exactly train, calibration, val, test"
        )
    members = {
        split: _validate_family_member_descriptor(raw_members[split], split=split)
        for split in _FAMILY_MEMBER_ORDER
    }
    common_contract = _validate_family_common_contract(
        manifest.get("common_contract")
    )
    if common_contract["schema_version"] != manifest["schema_version"]:
        raise RiskDataContractError(
            "dataset family top-level/common schema_version mismatch"
        )
    cross_split_audit = _validate_family_cross_split_audit(
        manifest.get("cross_split_audit"), members=members
    )
    production_evaluation_metadata = (
        _validate_production_evaluation_metadata(
            manifest.get("production_evaluation_metadata")
        )
    )
    declared_digest = _require_sha256(
        manifest.get("risk_dataset_family_digest"),
        field="risk_dataset_family_digest",
    )
    if _risk_dataset_family_digest(manifest) != declared_digest:
        raise RiskDataContractError("risk dataset family digest mismatch")
    return (
        members,
        cross_split_audit,
        production_evaluation_metadata,
        declared_digest,
    )


def _write_family_seal(staging: Path, manifest: Mapping[str, object]) -> None:
    manifest_path = staging / _FAMILY_MANIFEST_NAME
    marker_path = staging / _COMPLETE_MARKER_NAME
    checksums_path = staging / _CHECKSUMS_NAME
    manifest_path.write_bytes(
        (_canonical_json(dict(manifest)) + "\n").encode("utf-8")
    )
    marker_path.write_bytes(b"")
    entries = {
        _COMPLETE_MARKER_NAME: _sha256_file(marker_path),
        _FAMILY_MANIFEST_NAME: _sha256_file(manifest_path),
    }
    checksums_path.write_text(
        "".join(
            f"{entries[relative]}  {relative}\n" for relative in sorted(entries)
        ),
        encoding="utf-8",
    )


def _load_family_manifest(root: Path) -> dict[str, object]:
    _require_directory(root, label="risk dataset family root")
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise RiskDataContractError("failed to enumerate risk dataset family") from exc
    actual_names = {entry.name for entry in entries}
    missing = _REQUIRED_FAMILY_FILES - actual_names
    unexpected = actual_names - _REQUIRED_FAMILY_FILES
    if missing:
        raise RiskDataContractError(
            "incomplete risk dataset family: missing " + ", ".join(sorted(missing))
        )
    if unexpected:
        raise RiskDataContractError(
            "unexpected risk dataset family files: "
            + ", ".join(sorted(unexpected))
        )
    for entry in entries:
        if entry.is_symlink():
            raise RiskDataContractError(
                f"risk dataset family contains a forbidden symlink: {entry.name}"
            )
        if not entry.is_file(follow_symlinks=False):
            raise RiskDataContractError(
                f"risk dataset family entry must be a regular file: {entry.name}"
            )
    marker = root / _COMPLETE_MARKER_NAME
    if marker.read_bytes() != b"":
        raise RiskDataContractError(
            "risk dataset family completion marker must be empty"
        )
    checksum_entries = _parse_checksum_manifest(
        root / _CHECKSUMS_NAME,
        label="risk dataset family checksum manifest",
    )
    expected_checksum_entries = {_COMPLETE_MARKER_NAME, _FAMILY_MANIFEST_NAME}
    if set(checksum_entries) != expected_checksum_entries:
        raise RiskDataContractError(
            "risk dataset family checksum coverage is not exact"
        )
    for relative in sorted(expected_checksum_entries):
        if checksum_entries[relative] != _sha256_file(root / relative):
            raise RiskDataContractError(
                f"risk dataset family checksum mismatch for {relative}"
            )
    manifest_path = root / _FAMILY_MANIFEST_NAME
    try:
        raw = manifest_path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise RiskDataContractError("failed to read dataset family manifest") from exc
    manifest = _strict_json_loads(text, label="dataset family manifest")
    if raw != (_canonical_json(manifest) + "\n").encode("utf-8"):
        raise RiskDataContractError(
            "dataset family manifest is not canonical compact JSON"
        )
    return manifest


def publish_risk_dataset_family(
    output_dir: str | Path,
    *,
    members: Mapping[
        str,
        LoadedRiskDataset | RiskDatasetFamilyMemberSource,
    ],
) -> Path:
    """Reauthenticate and atomically publish exactly four split dataset seals."""

    output_path = _absolute_without_symlink_resolution(Path(output_dir))
    _assert_no_symlink_components(
        output_path, label="risk dataset family output", allow_missing=True
    )
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite immutable risk dataset family: {output_path}"
        )
    if not isinstance(members, Mapping) or set(members) != set(
        _FAMILY_MEMBER_ORDER
    ):
        raise RiskDataContractError(
            "risk dataset family members must be exactly train, calibration, val, test"
        )

    reloaded_members: dict[str, LoadedRiskDataset] = {}
    member_descriptors: dict[str, dict[str, object]] = {}
    identities: dict[str, dict[str, set[str]]] = {}
    row_counts: dict[str, int] = {}
    common_contract: dict[str, object] | None = None
    for split in _FAMILY_MEMBER_ORDER:
        member = members[split]
        reloaded = _authenticate_family_member_seal(
            member,
            split=split,
        )
        rows = _load_family_member_metadata_rows(reloaded)
        observed_contract = _family_common_contract(reloaded)
        if common_contract is None:
            common_contract = observed_contract
        elif observed_contract != common_contract:
            differing = next(
                key
                for key in _FAMILY_COMMON_CONTRACT_KEYS
                if observed_contract[key] != common_contract[key]
            )
            raise RiskDataContractError(
                f"risk dataset family common contract mismatch: {differing}"
            )
        reloaded_members[split] = reloaded
        row_counts[split] = len(rows)
        identities[split] = _family_identity_sets(rows, split=split)
        member_descriptors[split] = {
            "split": split,
            "risk_dataset_manifest_digest": (
                reloaded.risk_dataset_manifest_digest
            ),
            "sample_count": len(rows),
            "sample_ids_digest_sha256": (
                risk_dataset_family_sample_ids_digest(
                    sorted(identities[split]["sample_id"])
                )
            ),
            "shard_count": len(reloaded.shards),
        }
    assert common_contract is not None
    audit = _build_family_cross_split_audit(identities, row_counts=row_counts)
    manifest: dict[str, object] = {
        "dataset_family_layout_version": RISK_DATASET_FAMILY_LAYOUT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "member_order": list(_FAMILY_MEMBER_ORDER),
        "members": member_descriptors,
        "common_contract": common_contract,
        "cross_split_audit": audit,
        "production_evaluation_metadata": _production_evaluation_metadata(),
    }
    family_digest = _risk_dataset_family_digest(manifest)
    manifest["risk_dataset_family_digest"] = family_digest

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(
        output_path.parent, label="risk dataset family output parent"
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.staging-",
            dir=output_path.parent,
        )
    )
    try:
        _write_family_seal(staging, manifest)
        expected_digests = {
            split: reloaded_members[split].risk_dataset_manifest_digest
            for split in _FAMILY_MEMBER_ORDER
        }
        reloaded_family = load_risk_dataset_family(
            staging,
            expected_member_digests=expected_digests,
        )
        if reloaded_family.risk_dataset_family_digest != family_digest:
            raise RiskDataContractError("formal staging family reload mismatch")
        _atomic_rename_directory_noreplace(staging, output_path)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return output_path


def load_risk_dataset_family(
    root: str | Path,
    *,
    expected_member_digests: Mapping[str, str] | None = None,
) -> LoadedRiskDatasetFamily:
    """Load one self-contained family seal without guessing collection roots."""

    root_path = _absolute_without_symlink_resolution(Path(root))
    manifest = _load_family_manifest(root_path)
    (
        members,
        cross_split_audit,
        production_evaluation_metadata,
        family_digest,
    ) = (
        _validate_risk_dataset_family_manifest(manifest)
    )
    if expected_member_digests is not None:
        if not isinstance(expected_member_digests, Mapping) or set(
            expected_member_digests
        ) != set(_FAMILY_MEMBER_ORDER):
            raise RiskDataContractError(
                "expected_member_digests must exactly cover train, calibration, val, test"
            )
        for split in _FAMILY_MEMBER_ORDER:
            expected = _require_sha256(
                expected_member_digests[split],
                field=f"expected_member_digests.{split}",
            )
            if expected != members[split]["risk_dataset_manifest_digest"]:
                raise RiskDataContractError(
                    f"expected member digest mismatch for {split}"
                )
    ordered_members = {
        split: members[split] for split in _FAMILY_MEMBER_ORDER
    }
    return LoadedRiskDatasetFamily(
        root=root_path,
        manifest=_frozen_string_object_dict(manifest),
        members=_FrozenDict(ordered_members),
        cross_split_audit=_frozen_string_object_dict(cross_split_audit),
        production_evaluation_metadata=_frozen_string_object_dict(
            production_evaluation_metadata
        ),
        risk_dataset_family_digest=family_digest,
    )


def load_occupancy_sidecar_collection(
    dataset: LoadedRiskDataset,
    *,
    sidecar_root: str | Path,
) -> LoadedRiskSidecarCollection:
    """Fully reload sidecars and pair markers bound by ``dataset``."""

    if not isinstance(dataset, LoadedRiskDataset):
        raise RiskDataContractError(
            "occupancy sidecars require an authenticated LoadedRiskDataset"
        )
    section = dataset.manifest.get("occupancy_sidecars")
    if not isinstance(section, Mapping):
        raise RiskDataContractError(
            "dataset seal does not contain an occupancy_sidecars publication"
        )
    descriptors, query_geometry, _ = _validate_sidecar_collection_section(
        section,
        grid=dataset.grid,
        base_risk_dataset_manifest_digest=dataset.risk_dataset_manifest_digest,
    )
    if len(descriptors) != len(dataset.shards):
        raise RiskDataContractError("occupancy sidecar/risk shard count mismatch")
    return _formally_validate_sidecar_collection(
        _absolute_without_symlink_resolution(Path(sidecar_root)),
        collection_root=dataset.collection_root,
        risk_descriptors=dataset.shards,
        grid=dataset.grid,
        expected_split=dataset.split,
        base_risk_dataset_manifest_digest=dataset.risk_dataset_manifest_digest,
        base_config_digest=str(section["base_config_digest"]),
        query_geometry=query_geometry,
        expected_section=section,
    )


__all__ = [
    "PRODUCTION_RISK_EVALUATION_METADATA_LAYOUT_VERSION",
    "RISK_DATASET_FAMILY_LAYOUT_VERSION",
    "RISK_DATASET_LAYOUT_VERSION",
    "RISK_SIDECAR_COLLECTION_LAYOUT_VERSION",
    "LoadedRiskDataset",
    "LoadedRiskDatasetFamily",
    "LoadedRiskSidecarCollection",
    "RiskDatasetFamilyMemberSource",
    "RiskShardDescriptor",
    "SidecarShardDescriptor",
    "canonical_dynamic_objects_digest",
    "load_risk_dataset_family",
    "load_risk_dataset_seal",
    "load_occupancy_sidecar_collection",
    "publish_risk_dataset_family",
    "publish_risk_dataset_seal",
    "risk_dataset_family_sample_ids_digest",
    "validate_risk_dataset_manifest",
]
