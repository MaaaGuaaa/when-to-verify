"""Deterministic immutable shard I/O for oracle-only SOP-08 labels."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Any, BinaryIO, Mapping, Sequence

import numpy as np

from src.contracts import SCHEMA_VERSION, GridSpec
from src.datasets.split_manager import SPLIT_NAMES
from src.generation.risk_sidecars import RiskLabelSidecar


RISK_SIDECAR_SHARD_LAYOUT_VERSION = "risk_label_sidecar_shard_v1"
RISK_SIDECAR_FUTURE_DT_S = 0.2
RISK_SIDECAR_PAIR_COMPLETION_MARKER_VERSION = (
    "risk_sidecar_pair_completion_v1"
)

_PAYLOAD_NAME = "sidecars.npz"
_SUMMARY_NAME = "summary.json"
_REQUIRED_FILES = frozenset({_PAYLOAD_NAME, _SUMMARY_NAME})
_ARRAY_NAMES = (
    "hidden_risk_occupancy",
    "robot_future_footprints",
    "future_endpoint_times_s",
)
_NPZ_KEYS = frozenset(_ARRAY_NAMES)
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "layout_version",
        "split",
        "shard_index",
        "sample_count",
        "sample_ids",
        "source_risk_shard_semantic_digest",
        "grid",
        "future_endpoint_times_s",
        "files",
        "array_layout",
        "total_array_nbytes",
        "payload_sha256",
        "semantic_digest",
    }
)
_PAIR_MARKER_KEYS = frozenset(
    {
        "schema_version",
        "completion_marker_version",
        "risk_root_basename",
        "sidecar_root_basename",
        "split",
        "shard_index",
        "ordered_sample_ids_digest_sha256",
        "risk_shard_semantic_digest",
        "sidecar_shard_semantic_digest",
        "marker_digest_sha256",
    }
)


@dataclass(frozen=True)
class LoadedRiskSidecarShard:
    """Verified float32 SOP-08 labels joined by ordered sample identity."""

    sample_ids: tuple[str, ...]
    hidden_risk_occupancy: np.ndarray
    robot_future_footprints: np.ndarray
    future_endpoint_times_s: np.ndarray
    split: str
    shard_index: int
    source_risk_shard_semantic_digest: str
    semantic_digest: str
    summary: Mapping[str, object]


@dataclass(frozen=True)
class LoadedRiskSidecarPairCompletionMarker:
    """Verified evidence that one risk/sidecar shard pair is complete."""

    risk_root_basename: str
    sidecar_root_basename: str
    split: str
    shard_index: int
    ordered_sample_ids_digest_sha256: str
    risk_shard_semantic_digest: str
    sidecar_shard_semantic_digest: str
    marker_digest_sha256: str
    summary: Mapping[str, object]


@dataclass(frozen=True)
class _PathIdentity:
    path: Path
    device: int
    inode: int
    file_type: int


def _identity_from_stat(path: Path, metadata: os.stat_result) -> _PathIdentity:
    return _PathIdentity(
        path=path,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        file_type=stat.S_IFMT(metadata.st_mode),
    )


def _same_identity(first: _PathIdentity, second: _PathIdentity) -> bool:
    return (
        first.device == second.device
        and first.inode == second.inode
        and first.file_type == second.file_type
    )


def _open_path_nofollow(
    path: Path, *, expected_file_type: int, name: str
) -> tuple[int, _PathIdentity]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if expected_file_type == stat.S_IFDIR:
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"{name} must not be a symlink: {path}") from exc
        if exc.errno == errno.ENOENT:
            raise ValueError(f"{name} not found: {path}") from exc
        raise ValueError(f"failed to open {name}: {path}") from exc
    try:
        identity = _identity_from_stat(path, os.fstat(descriptor))
        if identity.file_type != expected_file_type:
            raise ValueError(f"{name} has an invalid file type: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, identity


def _open_relative_regular_file_nofollow(
    root_fd: int, name: str
) -> tuple[int, _PathIdentity]:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("sidecar member name must be a basename")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=root_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"sidecar member must not be a symlink: {name}") from exc
        if exc.errno == errno.ENOENT:
            raise ValueError(f"sidecar member not found: {name}") from exc
        raise ValueError(f"failed to open sidecar member: {name}") from exc
    try:
        identity = _identity_from_stat(Path(name), os.fstat(descriptor))
        if identity.file_type != stat.S_IFREG:
            raise ValueError(
                f"sidecar member must be a direct regular file: {name}"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, identity


def _read_open_file(handle: BinaryIO) -> bytes:
    handle.seek(0)
    payload = handle.read()
    handle.seek(0)
    return payload


def _sha256_open_file(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _verify_path_identity(
    expected: _PathIdentity, *, name: str
) -> None:
    descriptor, actual = _open_path_nofollow(
        expected.path,
        expected_file_type=expected.file_type,
        name=name,
    )
    os.close(descriptor)
    if not _same_identity(expected, actual):
        raise ValueError(f"{name} identity changed during verified load")


def _verify_relative_identity(
    root_fd: int, name: str, expected: _PathIdentity
) -> None:
    descriptor, actual = _open_relative_regular_file_nofollow(root_fd, name)
    os.close(descriptor)
    if not _same_identity(expected, actual):
        raise ValueError(
            f"sidecar member identity changed during verified load: {name}"
        )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _strict_json_loads(payload: str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(payload, parse_constant=reject_constant)


def _serialize_summary(value: Mapping[str, object]) -> bytes:
    return (_canonical_json(dict(value)) + "\n").encode("utf-8")


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _LOWER_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_direct_directory(path: Path, *, name: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{name} directory not found: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{name} must not be a symlink: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{name} must be a direct directory: {path}")


def _require_direct_regular_file(path: Path, *, name: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{name} file not found: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{name} must not be a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a direct regular file: {path}")


def _require_basename(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise ValueError(f"{name} must be a non-empty basename")
    return value


def _ordered_sample_ids_digest(sample_ids: Sequence[str]) -> str:
    if isinstance(sample_ids, (str, bytes)) or not isinstance(
        sample_ids, Sequence
    ):
        raise TypeError("sample_ids must be a sequence")
    ordered = tuple(sample_ids)
    if not ordered or not all(
        isinstance(sample_id, str) and sample_id for sample_id in ordered
    ):
        raise ValueError("sample_ids must contain non-empty strings")
    if len(ordered) != len(set(ordered)):
        raise ValueError("sample_ids must be unique")
    digest = hashlib.sha256()
    digest.update(b"risk-sidecar-pair-ordered-sample-ids-v1\0")
    digest.update(_canonical_json(list(ordered)).encode("utf-8"))
    return digest.hexdigest()


def _pair_marker_digest(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(b"risk-sidecar-pair-completion-marker-v1\0")
    digest.update(_canonical_json(dict(payload)).encode("utf-8"))
    return digest.hexdigest()


def risk_sidecar_pair_completion_marker_path(
    sidecar_root: str | Path,
) -> Path:
    """Return the external marker path adjacent to a frozen sidecar root."""

    root = Path(sidecar_root)
    _require_basename(root.name, name="sidecar root basename")
    return root.with_name(f"{root.name}.risk-sidecar-pair-complete.json")


def _validate_grid(grid: Any) -> GridSpec:
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    for name in ("height", "width", "future_steps"):
        _require_positive_int(int(getattr(grid, name)), name=f"grid.{name}")
    resolution = grid.resolution_m
    if isinstance(resolution, (bool, np.bool_)) or not isinstance(
        resolution, (int, float, np.integer, np.floating)
    ):
        raise TypeError("grid.resolution_m must be a positive finite real")
    if not np.isfinite(resolution) or float(resolution) <= 0.0:
        raise ValueError("grid.resolution_m must be a positive finite real")
    if grid.future_steps != 15:
        raise ValueError("SOP08 sidecars require exactly 15 future endpoints")
    return grid


def _grid_manifest(grid: GridSpec) -> dict[str, object]:
    return {
        "future_steps": grid.future_steps,
        "height": grid.height,
        "resolution_m": float(grid.resolution_m),
        "width": grid.width,
    }


def _expected_endpoint_times(grid: GridSpec) -> np.ndarray:
    return (
        np.arange(1, grid.future_steps + 1, dtype=np.float32)
        * np.float32(RISK_SIDECAR_FUTURE_DT_S)
    )


def _endpoint_manifest(grid: GridSpec) -> list[float]:
    return [
        float((index + 1) * RISK_SIDECAR_FUTURE_DT_S)
        for index in range(grid.future_steps)
    ]


def _build_arrays(
    sidecars: Sequence[RiskLabelSidecar], *, grid: GridSpec
) -> dict[str, np.ndarray]:
    expected_times = _expected_endpoint_times(grid)
    for sidecar in sidecars:
        expected_shape = (grid.future_steps, grid.height, grid.width)
        if sidecar.hidden_risk_occupancy.shape != expected_shape:
            raise ValueError("hidden_risk_occupancy shape differs from grid")
        if sidecar.robot_future_footprints.shape != expected_shape:
            raise ValueError("robot_future_footprints shape differs from grid")
        if not np.array_equal(sidecar.future_endpoint_times_s, expected_times):
            raise ValueError("sidecar future endpoint times must be 0.2 ... 3.0 s")
    arrays = {
        "hidden_risk_occupancy": np.ascontiguousarray(
            np.stack(
                [sidecar.hidden_risk_occupancy for sidecar in sidecars], axis=0
            ),
            dtype=np.uint8,
        ),
        "robot_future_footprints": np.ascontiguousarray(
            np.stack(
                [sidecar.robot_future_footprints for sidecar in sidecars], axis=0
            ),
            dtype=np.uint8,
        ),
        "future_endpoint_times_s": np.ascontiguousarray(
            expected_times, dtype=np.dtype("<f4")
        ),
    }
    return arrays


def _array_layout(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        name: {
            "dtype": arrays[name].dtype.str,
            "nbytes": arrays[name].nbytes,
            "order": "C",
            "shape": list(arrays[name].shape),
        }
        for name in _ARRAY_NAMES
    }


def _semantic_digest(
    arrays: Mapping[str, np.ndarray],
    *,
    sample_ids: Sequence[str],
    split: str,
    shard_index: int,
    source_digest: str,
    grid: GridSpec,
) -> str:
    header = {
        "schema_version": SCHEMA_VERSION,
        "layout_version": RISK_SIDECAR_SHARD_LAYOUT_VERSION,
        "split": split,
        "shard_index": shard_index,
        "sample_ids": list(sample_ids),
        "source_risk_shard_semantic_digest": source_digest,
        "grid": _grid_manifest(grid),
        "future_endpoint_times_s": _endpoint_manifest(grid),
        "array_layout": _array_layout(arrays),
    }
    digest = hashlib.sha256()
    digest.update(b"risk-label-sidecar-shard-v1\0")
    header_bytes = _canonical_json(header).encode("utf-8")
    digest.update(len(header_bytes).to_bytes(8, "big"))
    digest.update(header_bytes)
    for name in _ARRAY_NAMES:
        name_bytes = name.encode("utf-8")
        raw = arrays[name].tobytes(order="C")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _atomic_rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Use Linux ``renameat2(RENAME_NOREPLACE)``; never fall back to clobber."""

    source_path = Path(os.path.abspath(os.fspath(source)))
    destination_path = Path(os.path.abspath(os.fspath(destination)))
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as exc:  # pragma: no cover - Linux libc is always available
        raise OSError(errno.ENOSYS, "libc unavailable for renameat2") from exc
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOSYS,
            "renameat2 unavailable; refusing overwrite-capable fallback",
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
        -100,
        os.fsencode(source_path),
        -100,
        os.fsencode(destination_path),
        1,
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
    raise OSError(error_number, os.strerror(error_number), destination_path)


def _capture_owned_path(
    path: Path, *, expected_file_type: int
) -> _PathIdentity:
    descriptor, identity = _open_path_nofollow(
        path,
        expected_file_type=expected_file_type,
        name="owned transaction path",
    )
    os.close(descriptor)
    return identity


_atomic_cleanup_claim_noreplace = _atomic_rename_directory_noreplace


def _remove_owned_path_via_quarantine(owned: _PathIdentity) -> bool:
    """Atomically claim, verify, then delete only this invocation's inode."""

    quarantine = Path(
        tempfile.mkdtemp(
            prefix=f".{owned.path.name}.cleanup-quarantine-",
            dir=owned.path.parent,
        )
    )
    quarantine_identity = _capture_owned_path(
        quarantine, expected_file_type=stat.S_IFDIR
    )
    claimed = quarantine / "claimed"
    try:
        try:
            _atomic_cleanup_claim_noreplace(owned.path, claimed)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                _verify_path_identity(
                    quarantine_identity, name="cleanup quarantine"
                )
                os.rmdir(quarantine)
                return True
            raise

        claimed_identity = _identity_from_stat(claimed, claimed.lstat())
        if not _same_identity(owned, claimed_identity):
            try:
                _atomic_rename_directory_noreplace(claimed, owned.path)
            except OSError as restore_exc:
                raise ValueError(
                    "cleanup incomplete: raced competitor could not be restored; "
                    f"preserved at {claimed}"
                ) from restore_exc
            _verify_path_identity(
                quarantine_identity, name="cleanup quarantine"
            )
            os.rmdir(quarantine)
            return False

        if owned.file_type == stat.S_IFDIR:
            shutil.rmtree(claimed)
        elif owned.file_type == stat.S_IFREG:
            claimed.unlink()
        else:  # pragma: no cover - owned paths are restricted at capture
            raise ValueError("cleanup incomplete: unsupported owned path type")
        _verify_path_identity(quarantine_identity, name="cleanup quarantine")
        os.rmdir(quarantine)
        return True
    except BaseException:
        if quarantine.exists() and not any(quarantine.iterdir()):
            try:
                _verify_path_identity(
                    quarantine_identity, name="cleanup quarantine"
                )
                os.rmdir(quarantine)
            except (OSError, ValueError):
                pass
        raise


def _atomic_publish_owned_noreplace(
    source: Path,
    destination: Path,
    expected_identity: _PathIdentity,
) -> _PathIdentity:
    """No-replace publish with post-rename inode verification and restore."""

    _atomic_rename_directory_noreplace(source, destination)
    try:
        actual = _capture_owned_path(
            destination,
            expected_file_type=expected_identity.file_type,
        )
    except BaseException:
        try:
            _atomic_rename_directory_noreplace(destination, source)
        except OSError as restore_exc:
            raise ValueError(
                "publication cleanup incomplete: destination could not be restored"
            ) from restore_exc
        raise
    if not _same_identity(expected_identity, actual):
        try:
            _atomic_rename_directory_noreplace(destination, source)
        except OSError as restore_exc:
            raise ValueError(
                "publication cleanup incomplete: raced staging path was preserved "
                f"at {destination}"
            ) from restore_exc
        raise ValueError("staging identity changed during no-replace publication")
    return _PathIdentity(
        path=destination,
        device=expected_identity.device,
        inode=expected_identity.inode,
        file_type=expected_identity.file_type,
    )


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _load_summary(raw: bytes) -> dict[str, object]:
    try:
        parsed = _strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("sidecar summary is not strict finite JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("sidecar summary must be a JSON object")
    if _serialize_summary(parsed) != raw:
        raise ValueError("sidecar summary must be canonical compact JSON")
    if set(parsed) != _SUMMARY_KEYS:
        raise ValueError("sidecar summary keys violate the shard layout")
    return parsed


def _load_arrays(handle: BinaryIO) -> dict[str, np.ndarray]:
    try:
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            if set(archive.files) != _NPZ_KEYS:
                raise ValueError("sidecar NPZ keys violate the shard layout")
            arrays = {
                name: np.array(archive[name], order="C", copy=True)
                for name in _ARRAY_NAMES
            }
        handle.seek(0)
    except (OSError, ValueError) as exc:
        raise ValueError("failed to load pickle-free sidecar NPZ") from exc
    return arrays


def _validate_loaded_arrays(
    arrays: Mapping[str, np.ndarray], *, grid: GridSpec, sample_count: int
) -> None:
    expected_shapes = {
        "hidden_risk_occupancy": (
            sample_count,
            grid.future_steps,
            grid.height,
            grid.width,
        ),
        "robot_future_footprints": (
            sample_count,
            grid.future_steps,
            grid.height,
            grid.width,
        ),
        "future_endpoint_times_s": (grid.future_steps,),
    }
    expected_dtypes = {
        "hidden_risk_occupancy": np.dtype(np.uint8),
        "robot_future_footprints": np.dtype(np.uint8),
        "future_endpoint_times_s": np.dtype("<f4"),
    }
    for name in _ARRAY_NAMES:
        array = arrays[name]
        if array.shape != expected_shapes[name]:
            raise ValueError(f"{name} shape differs from the sidecar summary/grid")
        if array.dtype != expected_dtypes[name]:
            raise TypeError(f"{name} dtype differs from the sidecar layout")
        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must use C-contiguous storage")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
    for name in ("hidden_risk_occupancy", "robot_future_footprints"):
        if not np.isin(arrays[name], (0, 1)).all():
            raise ValueError(f"{name} must be binary")
    if not np.array_equal(
        arrays["future_endpoint_times_s"], _expected_endpoint_times(grid)
    ):
        raise ValueError("future endpoint times must be exactly 0.2 ... 3.0 s")


def _load_pair_marker_summary(raw: bytes) -> dict[str, object]:
    try:
        parsed = _strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "pair completion marker is not strict finite JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("pair completion marker must be a JSON object")
    if _serialize_summary(parsed) != raw:
        raise ValueError(
            "pair completion marker must be canonical compact JSON"
        )
    if set(parsed) != _PAIR_MARKER_KEYS:
        raise ValueError("pair completion marker keys violate the layout")
    return parsed


def write_risk_sidecar_pair_completion_marker(
    output_file: str | Path,
    *,
    risk_root: str | Path,
    sidecar_root: str | Path,
    split: str,
    shard_index: int,
    sample_ids: Sequence[str],
    risk_shard_semantic_digest: str,
    sidecar_shard_semantic_digest: str,
) -> Path:
    """Write one immutable canonical pair marker to a caller-owned target."""

    risk_basename = _require_basename(
        Path(risk_root).name, name="risk root basename"
    )
    sidecar_basename = _require_basename(
        Path(sidecar_root).name, name="sidecar root basename"
    )
    if split not in SPLIT_NAMES:
        raise ValueError(f"unsupported split: {split!r}")
    shard_index = _require_nonnegative_int(shard_index, name="shard_index")
    ordered_ids_digest = _ordered_sample_ids_digest(sample_ids)
    risk_digest = _require_sha256(
        risk_shard_semantic_digest,
        name="risk shard semantic digest",
    )
    sidecar_digest = _require_sha256(
        sidecar_shard_semantic_digest,
        name="sidecar shard semantic digest",
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "completion_marker_version": (
            RISK_SIDECAR_PAIR_COMPLETION_MARKER_VERSION
        ),
        "risk_root_basename": risk_basename,
        "sidecar_root_basename": sidecar_basename,
        "split": split,
        "shard_index": shard_index,
        "ordered_sample_ids_digest_sha256": ordered_ids_digest,
        "risk_shard_semantic_digest": risk_digest,
        "sidecar_shard_semantic_digest": sidecar_digest,
    }
    payload["marker_digest_sha256"] = _pair_marker_digest(payload)

    output_path = Path(output_file)
    if os.path.lexists(output_path):
        raise FileExistsError(
            f"refusing to overwrite immutable pair marker: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.staging-",
        dir=output_path.parent,
    )
    staging = Path(temporary_name)
    staging_identity = _identity_from_stat(staging, os.fstat(descriptor))
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as staging_handle:
            staging_handle.write(_serialize_summary(payload))
            staging_handle.flush()
            os.fsync(staging_handle.fileno())
        loaded = load_risk_sidecar_pair_completion_marker(
            staging,
            expected_risk_root=risk_root,
            expected_sidecar_root=sidecar_root,
            expected_split=split,
            expected_shard_index=shard_index,
            expected_sample_ids=sample_ids,
            expected_risk_shard_semantic_digest=risk_digest,
            expected_sidecar_shard_semantic_digest=sidecar_digest,
        )
        if loaded.marker_digest_sha256 != payload["marker_digest_sha256"]:
            raise ValueError("pair marker staging reload digest mismatch")
        _verify_path_identity(
            staging_identity, name="pair marker staging file"
        )
        _atomic_publish_owned_noreplace(
            staging, output_path, staging_identity
        )
    except BaseException as exc:
        try:
            cleanup_complete = _remove_owned_path_via_quarantine(
                staging_identity
            )
        except BaseException as cleanup_exc:
            if isinstance(exc, Exception):
                raise ValueError(
                    f"{exc}; cleanup incomplete: {cleanup_exc}"
                ) from exc
            exc.add_note(f"cleanup incomplete: {cleanup_exc}")
            raise
        if not cleanup_complete:
            if isinstance(exc, Exception):
                raise ValueError(
                    f"{exc}; cleanup incomplete: staging identity changed"
                ) from exc
            exc.add_note("cleanup incomplete: staging identity changed")
        raise
    return output_path


def load_risk_sidecar_pair_completion_marker(
    marker_file: str | Path,
    *,
    expected_risk_root: str | Path,
    expected_sidecar_root: str | Path,
    expected_split: str,
    expected_shard_index: int,
    expected_sample_ids: Sequence[str],
    expected_risk_shard_semantic_digest: str,
    expected_sidecar_shard_semantic_digest: str,
) -> LoadedRiskSidecarPairCompletionMarker:
    """Verify canonical evidence for one exact risk/sidecar shard pair."""

    expected_risk_basename = _require_basename(
        Path(expected_risk_root).name, name="expected risk root basename"
    )
    expected_sidecar_basename = _require_basename(
        Path(expected_sidecar_root).name,
        name="expected sidecar root basename",
    )
    if expected_split not in SPLIT_NAMES:
        raise ValueError(f"unsupported expected split: {expected_split!r}")
    expected_shard_index = _require_nonnegative_int(
        expected_shard_index, name="expected shard_index"
    )
    expected_ids_digest = _ordered_sample_ids_digest(expected_sample_ids)
    expected_risk_digest = _require_sha256(
        expected_risk_shard_semantic_digest,
        name="expected risk shard semantic digest",
    )
    expected_sidecar_digest = _require_sha256(
        expected_sidecar_shard_semantic_digest,
        name="expected sidecar shard semantic digest",
    )

    marker_path = Path(marker_file)
    marker_fd, marker_identity = _open_path_nofollow(
        marker_path,
        expected_file_type=stat.S_IFREG,
        name="pair completion marker",
    )
    try:
        with os.fdopen(marker_fd, "rb", closefd=True) as marker_handle:
            marker_raw = _read_open_file(marker_handle)
    except BaseException:
        try:
            os.close(marker_fd)
        except OSError:
            pass
        raise
    summary = _load_pair_marker_summary(marker_raw)
    if summary["schema_version"] != SCHEMA_VERSION:
        raise ValueError("pair completion marker schema_version mismatch")
    if summary["completion_marker_version"] != (
        RISK_SIDECAR_PAIR_COMPLETION_MARKER_VERSION
    ):
        raise ValueError("unsupported pair completion marker version")
    risk_basename = _require_basename(
        summary["risk_root_basename"], name="marker risk root basename"
    )
    sidecar_basename = _require_basename(
        summary["sidecar_root_basename"],
        name="marker sidecar root basename",
    )
    split = summary["split"]
    if split not in SPLIT_NAMES:
        raise ValueError("pair completion marker split is invalid")
    shard_index = _require_nonnegative_int(
        summary["shard_index"], name="marker shard_index"
    )
    ids_digest = _require_sha256(
        summary["ordered_sample_ids_digest_sha256"],
        name="marker ordered sample IDs digest",
    )
    risk_digest = _require_sha256(
        summary["risk_shard_semantic_digest"],
        name="marker risk shard semantic digest",
    )
    sidecar_digest = _require_sha256(
        summary["sidecar_shard_semantic_digest"],
        name="marker sidecar shard semantic digest",
    )
    if (
        risk_basename != expected_risk_basename
        or sidecar_basename != expected_sidecar_basename
        or split != expected_split
        or shard_index != expected_shard_index
        or ids_digest != expected_ids_digest
        or risk_digest != expected_risk_digest
        or sidecar_digest != expected_sidecar_digest
    ):
        raise ValueError("pair completion marker evidence mismatch")
    digest_payload = {
        key: value
        for key, value in summary.items()
        if key != "marker_digest_sha256"
    }
    marker_digest = _require_sha256(
        summary["marker_digest_sha256"], name="pair marker digest"
    )
    if marker_digest != _pair_marker_digest(digest_payload):
        raise ValueError("pair completion marker digest mismatch")
    _verify_path_identity(
        marker_identity, name="pair completion marker"
    )
    frozen_summary = _freeze_json(summary)
    if not isinstance(frozen_summary, Mapping):  # pragma: no cover
        raise RuntimeError("pair marker summary freezing failed")
    return LoadedRiskSidecarPairCompletionMarker(
        risk_root_basename=risk_basename,
        sidecar_root_basename=sidecar_basename,
        split=str(split),
        shard_index=shard_index,
        ordered_sample_ids_digest_sha256=ids_digest,
        risk_shard_semantic_digest=risk_digest,
        sidecar_shard_semantic_digest=sidecar_digest,
        marker_digest_sha256=marker_digest,
        summary=frozen_summary,
    )


def write_risk_sidecar_shard(
    sidecars: Sequence[RiskLabelSidecar],
    output_dir: str | Path,
    *,
    grid: GridSpec,
    split: str,
    shard_index: int,
    source_risk_shard_semantic_digest: str,
) -> dict[str, Path]:
    """Publish one sidecar shard after a complete staging-directory reload."""

    validated_grid = _validate_grid(grid)
    if split not in SPLIT_NAMES:
        raise ValueError(f"unsupported split: {split!r}")
    shard_index = _require_nonnegative_int(shard_index, name="shard_index")
    source_digest = _require_sha256(
        source_risk_shard_semantic_digest,
        name="source risk shard semantic digest",
    )
    if isinstance(sidecars, (str, bytes)) or not isinstance(sidecars, Sequence):
        raise TypeError("sidecars must be a sequence")
    if not sidecars or any(
        not isinstance(sidecar, RiskLabelSidecar) for sidecar in sidecars
    ):
        raise TypeError("sidecars must contain RiskLabelSidecar instances")
    ordered = tuple(sorted(sidecars, key=lambda item: item.sample_id))
    sample_ids = tuple(item.sample_id for item in ordered)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample_id in sidecar shard")
    arrays = _build_arrays(ordered, grid=validated_grid)
    array_layout = _array_layout(arrays)
    semantic_digest = _semantic_digest(
        arrays,
        sample_ids=sample_ids,
        split=split,
        shard_index=shard_index,
        source_digest=source_digest,
        grid=validated_grid,
    )

    output_path = Path(output_dir)
    if os.path.lexists(output_path):
        raise FileExistsError(
            f"refusing to overwrite immutable sidecar shard: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.staging-", dir=output_path.parent
        )
    )
    staging_identity = _capture_owned_path(
        staging, expected_file_type=stat.S_IFDIR
    )
    try:
        payload_path = staging / _PAYLOAD_NAME
        with payload_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        payload_fd, _ = _open_path_nofollow(
            payload_path,
            expected_file_type=stat.S_IFREG,
            name="sidecar staging payload",
        )
        with os.fdopen(payload_fd, "rb", closefd=True) as payload_handle:
            payload_sha256 = _sha256_open_file(payload_handle)
        summary: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "layout_version": RISK_SIDECAR_SHARD_LAYOUT_VERSION,
            "split": split,
            "shard_index": shard_index,
            "sample_count": len(sample_ids),
            "sample_ids": list(sample_ids),
            "source_risk_shard_semantic_digest": source_digest,
            "grid": _grid_manifest(validated_grid),
            "future_endpoint_times_s": _endpoint_manifest(validated_grid),
            "files": {"payload": _PAYLOAD_NAME, "summary": _SUMMARY_NAME},
            "array_layout": array_layout,
            "total_array_nbytes": sum(
                array.nbytes for array in arrays.values()
            ),
            "payload_sha256": payload_sha256,
            "semantic_digest": semantic_digest,
        }
        with (staging / _SUMMARY_NAME).open("wb") as summary_handle:
            summary_handle.write(_serialize_summary(summary))
            summary_handle.flush()
            os.fsync(summary_handle.fileno())
        reloaded = load_risk_sidecar_shard(
            staging,
            grid=validated_grid,
            expected_sample_ids=sample_ids,
            expected_source_risk_shard_semantic_digest=source_digest,
        )
        if reloaded.semantic_digest != semantic_digest:
            raise ValueError("formal sidecar staging reload digest mismatch")
        _verify_path_identity(
            staging_identity, name="sidecar staging directory"
        )
        _atomic_publish_owned_noreplace(
            staging, output_path, staging_identity
        )
    except BaseException as exc:
        try:
            cleanup_complete = _remove_owned_path_via_quarantine(
                staging_identity
            )
        except BaseException as cleanup_exc:
            if isinstance(exc, Exception):
                raise ValueError(
                    f"{exc}; cleanup incomplete: {cleanup_exc}"
                ) from exc
            exc.add_note(f"cleanup incomplete: {cleanup_exc}")
            raise
        if not cleanup_complete:
            if isinstance(exc, Exception):
                raise ValueError(
                    f"{exc}; cleanup incomplete: staging identity changed"
                ) from exc
            exc.add_note("cleanup incomplete: staging identity changed")
        raise
    return {
        "directory": output_path,
        "payload": output_path / _PAYLOAD_NAME,
        "summary": output_path / _SUMMARY_NAME,
    }


def load_risk_sidecar_shard(
    output_dir: str | Path,
    *,
    grid: GridSpec,
    expected_sample_ids: Sequence[str],
    expected_source_risk_shard_semantic_digest: str,
) -> LoadedRiskSidecarShard:
    """Load a sidecar shard only after identity, layout, and byte checks pass."""

    validated_grid = _validate_grid(grid)
    source_digest = _require_sha256(
        expected_source_risk_shard_semantic_digest,
        name="expected source risk shard semantic digest",
    )
    if isinstance(expected_sample_ids, (str, bytes)) or not isinstance(
        expected_sample_ids, Sequence
    ):
        raise TypeError("expected_sample_ids must be a sequence")
    expected_ids = tuple(expected_sample_ids)
    if not expected_ids or not all(
        isinstance(sample_id, str) and sample_id for sample_id in expected_ids
    ):
        raise ValueError("expected_sample_ids must contain non-empty strings")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("expected_sample_ids must be unique")

    root = Path(output_dir)
    root_fd, root_identity = _open_path_nofollow(
        root,
        expected_file_type=stat.S_IFDIR,
        name="sidecar shard",
    )
    payload_fd: int | None = None
    summary_fd: int | None = None
    try:
        actual_files = set(os.listdir(root_fd))
        missing = _REQUIRED_FILES - actual_files
        unexpected = actual_files - _REQUIRED_FILES
        if missing:
            raise ValueError(
                "incomplete sidecar shard: missing "
                + ", ".join(sorted(missing))
            )
        if unexpected:
            raise ValueError(
                "unexpected sidecar shard files: "
                + ", ".join(sorted(unexpected))
            )
        payload_fd, payload_identity = _open_relative_regular_file_nofollow(
            root_fd, _PAYLOAD_NAME
        )
        summary_fd, summary_identity = _open_relative_regular_file_nofollow(
            root_fd, _SUMMARY_NAME
        )

        with os.fdopen(os.dup(summary_fd), "rb", closefd=True) as handle:
            summary = _load_summary(_read_open_file(handle))
        if summary["schema_version"] != SCHEMA_VERSION:
            raise ValueError("sidecar summary schema_version mismatch")
        if summary["layout_version"] != RISK_SIDECAR_SHARD_LAYOUT_VERSION:
            raise ValueError("unsupported sidecar shard layout")
        split = summary["split"]
        if split not in SPLIT_NAMES:
            raise ValueError("sidecar summary split is invalid")
        shard_index = _require_nonnegative_int(
            summary["shard_index"], name="summary shard_index"
        )
        sample_count = _require_positive_int(
            summary["sample_count"], name="summary sample_count"
        )
        declared_ids = summary["sample_ids"]
        if not isinstance(declared_ids, list) or not all(
            isinstance(sample_id, str) and sample_id
            for sample_id in declared_ids
        ):
            raise ValueError("sidecar summary sample_ids are invalid")
        if declared_ids != sorted(declared_ids) or len(
            set(declared_ids)
        ) != len(declared_ids):
            raise ValueError(
                "sidecar summary sample IDs must be stable and unique"
            )
        if len(declared_ids) != sample_count:
            raise ValueError("sidecar summary sample_count mismatch")
        if tuple(declared_ids) != expected_ids:
            raise ValueError(
                "sidecar ordered sample IDs differ from the risk shard"
            )
        declared_source = _require_sha256(
            summary["source_risk_shard_semantic_digest"],
            name="source risk shard semantic digest",
        )
        if declared_source != source_digest:
            raise ValueError("source risk shard semantic digest mismatch")
        if summary["grid"] != _grid_manifest(validated_grid):
            raise ValueError("sidecar summary grid mismatch")
        if summary["future_endpoint_times_s"] != _endpoint_manifest(
            validated_grid
        ):
            raise ValueError("sidecar future endpoint summary mismatch")
        if summary["files"] != {
            "payload": _PAYLOAD_NAME,
            "summary": _SUMMARY_NAME,
        }:
            raise ValueError("sidecar summary file layout mismatch")

        with os.fdopen(os.dup(payload_fd), "rb", closefd=True) as source:
            payload_snapshot = _read_open_file(source)
        with io.BytesIO(payload_snapshot) as handle:
            payload_digest = _sha256_open_file(handle)
            if _require_sha256(
                summary["payload_sha256"], name="sidecar payload SHA-256"
            ) != payload_digest:
                raise ValueError("sidecar payload SHA-256 mismatch")
            arrays = _load_arrays(handle)
        _validate_loaded_arrays(
            arrays, grid=validated_grid, sample_count=sample_count
        )
        layout = _array_layout(arrays)
        if summary["array_layout"] != layout:
            raise ValueError("sidecar array dtype/shape/byte layout mismatch")
        if summary["total_array_nbytes"] != sum(
            array.nbytes for array in arrays.values()
        ):
            raise ValueError("sidecar total array bytes mismatch")
        semantic_digest = _semantic_digest(
            arrays,
            sample_ids=expected_ids,
            split=split,
            shard_index=shard_index,
            source_digest=source_digest,
            grid=validated_grid,
        )
        if _require_sha256(
            summary["semantic_digest"], name="sidecar semantic digest"
        ) != semantic_digest:
            raise ValueError("sidecar semantic digest mismatch")

        _verify_relative_identity(
            root_fd, _PAYLOAD_NAME, payload_identity
        )
        _verify_relative_identity(
            root_fd, _SUMMARY_NAME, summary_identity
        )
        _verify_path_identity(root_identity, name="sidecar shard")

        hidden = np.array(
            arrays["hidden_risk_occupancy"],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        robot = np.array(
            arrays["robot_future_footprints"],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        endpoint_times = np.array(
            arrays["future_endpoint_times_s"],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        for array in (hidden, robot, endpoint_times):
            array.setflags(write=False)
        frozen_summary = _freeze_json(summary)
        if not isinstance(frozen_summary, Mapping):  # pragma: no cover
            raise RuntimeError("sidecar summary freezing failed")
        return LoadedRiskSidecarShard(
            sample_ids=expected_ids,
            hidden_risk_occupancy=hidden,
            robot_future_footprints=robot,
            future_endpoint_times_s=endpoint_times,
            split=str(split),
            shard_index=shard_index,
            source_risk_shard_semantic_digest=source_digest,
            semantic_digest=semantic_digest,
            summary=frozen_summary,
        )
    finally:
        if summary_fd is not None:
            os.close(summary_fd)
        if payload_fd is not None:
            os.close(payload_fd)
        os.close(root_fd)


__all__ = (
    "RISK_SIDECAR_FUTURE_DT_S",
    "RISK_SIDECAR_PAIR_COMPLETION_MARKER_VERSION",
    "RISK_SIDECAR_SHARD_LAYOUT_VERSION",
    "LoadedRiskSidecarPairCompletionMarker",
    "LoadedRiskSidecarShard",
    "load_risk_sidecar_pair_completion_marker",
    "load_risk_sidecar_shard",
    "risk_sidecar_pair_completion_marker_path",
    "write_risk_sidecar_pair_completion_marker",
    "write_risk_sidecar_shard",
)
