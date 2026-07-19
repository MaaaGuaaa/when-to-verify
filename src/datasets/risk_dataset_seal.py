"""Immutable dataset-level seal for accepted schema-3 SOP07 risk shards.

The shard writer remains the only parser for shard payloads.  This module
authenticates one complete SOP07 collection, binds its ordered shards and
frozen provenance into ``risk_dataset_v2``, and fully reloads both publication
levels before returning a dataset descriptor.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Mapping

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
from src.utils.config import ConfigError, load_config


RISK_DATASET_LAYOUT_VERSION = "risk_dataset_v2"
RISK_DATASET_FAMILY_LAYOUT_VERSION = "risk_dataset_family_v1"

_COLLECTION_HANDOFF_VERSION = "sop07_collection_complete_handoff_v1"
_COLLECTION_HANDOFF_NAME = "collection_complete_handoff.json"
_DATASET_MANIFEST_NAME = "dataset_manifest.json"
_CHECKSUMS_NAME = "checksums.sha256"
_COMPLETE_MARKER_NAME = ".producer-complete"
_REQUIRED_SEAL_FILES = frozenset(
    {_DATASET_MANIFEST_NAME, _CHECKSUMS_NAME, _COMPLETE_MARKER_NAME}
)
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


def _strict_json_loads(payload: str, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload, parse_constant=_reject_json_constant)
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


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _atomic_rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing an existing destination."""

    source_path = _absolute_without_symlink_resolution(Path(source))
    destination_path = _absolute_without_symlink_resolution(Path(destination))
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as exc:  # pragma: no cover - Linux libc is always loadable
        raise OSError(errno.ENOSYS, "libc is unavailable for renameat2") from exc
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOSYS,
            "libc renameat2 is unavailable; refusing overwrite-capable fallback",
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source_path),
        -100,  # AT_FDCWD
        os.fsencode(destination_path),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno() or errno.EIO
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination_path,
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination_path,
    )


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


def _parse_checksum_manifest(path: Path, *, label: str) -> dict[str, str]:
    _require_regular_file(path, label=label)
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
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
    if handoff.get("handoff_version") != _COLLECTION_HANDOFF_VERSION:
        raise RiskDataContractError("unsupported SOP07 collection handoff version")
    if handoff.get("collection_state") != "complete":
        raise RiskDataContractError("SOP07 collection handoff is not complete")
    if handoff.get("schema_version") != SCHEMA_VERSION:
        raise RiskDataContractError("SOP07 collection schema_version mismatch")
    if handoff.get("layout_version") != RISK_SHARD_LAYOUT_VERSION:
        raise RiskDataContractError("SOP07 collection uses an unsupported shard layout")
    if handoff.get("split") != expected_split:
        raise RiskDataContractError("SOP07 collection split mismatch")
    expected_role = f"sop07_{expected_split}_collection_complete_handoff"
    if handoff.get("artifact_role") != expected_role:
        raise RiskDataContractError("SOP07 collection source artifact role mismatch")
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


def _formally_validate_collection(
    collection_root: Path,
    *,
    grid: GridSpec,
    expected_split: str,
    handoff: Mapping[str, object],
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
        descriptors.append(descriptor)

    expected_count = _require_positive_int(
        handoff.get("sample_count"), field="collection sample_count"
    )
    if sum(item.sample_count for item in descriptors) != expected_count:
        raise RiskDataContractError("collection sample_count differs from shard totals")
    if len(sample_ids) != expected_count:
        raise RiskDataContractError("collection global sample identity count mismatch")
    if len(target_policy_values) != 1:
        raise RiskDataContractError(
            "collection must contain exactly one consistent target_type_policy_digest"
        )
    collection_semantics = {
        "schema_version": SCHEMA_VERSION,
        "layout_version": RISK_SHARD_LAYOUT_VERSION,
        "split": expected_split,
        "sample_count": len(sample_ids),
        "shards": [
            {
                "shard_index": descriptor.shard_index,
                "relative_root": descriptor.relative_root,
                "sample_count": descriptor.sample_count,
                "manifest_digest": descriptor.manifest_digest,
                "semantic_digest": descriptor.semantic_digest,
            }
            for descriptor in descriptors
        ],
    }
    recomputed_collection_digest = _sha256_bytes(
        _canonical_json(collection_semantics).encode("utf-8")
    )
    declared_collection_digest = _require_sha256(
        handoff.get("collection_semantic_digest_sha256"),
        field="collection_semantic_digest_sha256",
    )
    if recomputed_collection_digest != declared_collection_digest:
        raise RiskDataContractError("collection semantic digest mismatch")
    return tuple(descriptors), next(iter(target_policy_values))


def _semantic_manifest_projection(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in manifest.items()
        if key != "risk_dataset_manifest_digest" and key not in _RUNTIME_IDENTITY_FIELDS
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
    if set(manifest) != _DATASET_MANIFEST_KEYS:
        missing_keys = sorted(_DATASET_MANIFEST_KEYS - set(manifest))
        extra_keys = sorted(set(manifest) - _DATASET_MANIFEST_KEYS)
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
        "collection_handoff_version": _COLLECTION_HANDOFF_VERSION,
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
    if set(manifest) != _DATASET_MANIFEST_KEYS:
        missing_keys = sorted(_DATASET_MANIFEST_KEYS - set(manifest))
        extra_keys = sorted(set(manifest) - _DATASET_MANIFEST_KEYS)
        raise RiskDataContractError(
            "dataset manifest keys mismatch: "
            f"missing={missing_keys}, unexpected={extra_keys}"
        )
    return manifest


def load_risk_dataset_seal(
    seal_root: str | Path,
    *,
    collection_root: str | Path,
    expected_split: str,
    expected_manifest_digest: str | None = None,
) -> LoadedRiskDataset:
    """Load only a complete v2 seal whose entire source collection still verifies."""

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
    if manifest.get("collection_handoff_version") != _COLLECTION_HANDOFF_VERSION:
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
    loaded_descriptors, loaded_target_digest = _formally_validate_collection(
        collection_path,
        grid=grid,
        expected_split=split,
        handoff=handoff,
    )
    if loaded_descriptors != descriptors:
        raise RiskDataContractError("dataset shard descriptors differ from source handoff")
    if loaded_target_digest != target_digest:
        raise RiskDataContractError(
            "target_type_policy_digest differs from formally loaded shard provenance"
        )

    provenance = {
        "g1_split_manifest_digest": g1_digest,
        "risk_dataset_manifest_digest": dataset_digest,
        "dynamic_objects_config_digest": dynamic_digest,
        "target_type_policy_digest": target_digest,
    }
    return LoadedRiskDataset(
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


__all__ = [
    "RISK_DATASET_FAMILY_LAYOUT_VERSION",
    "RISK_DATASET_LAYOUT_VERSION",
    "LoadedRiskDataset",
    "RiskShardDescriptor",
    "canonical_dynamic_objects_digest",
    "load_risk_dataset_seal",
    "publish_risk_dataset_seal",
    "validate_risk_dataset_manifest",
]
