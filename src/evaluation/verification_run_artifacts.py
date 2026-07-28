"""Atomic, checksum-authenticated directories for verification experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType

from src.utils.atomic_publish import atomic_rename_noreplace


VERIFICATION_TRAINING_RUN_LAYOUT_VERSION = "verification_training_run_v2"
VERIFICATION_EVALUATION_RUN_LAYOUT_VERSION = "verification_evaluation_run_v2"
_COMPLETE = "COMPLETE.json"


def canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("run metadata must be finite canonical JSON") from exc


def strict_json_file(path: str | Path, *, label: str) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a real file")

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        result = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return result


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"authenticated payload must be a real file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot checksum authenticated payload: {source}") from exc
    return digest.hexdigest()


def load_authenticated_run_directory(
    input_dir: str | Path,
    *,
    expected_layout_version: str,
    required_payloads: frozenset[str] | None = None,
) -> Mapping[str, object]:
    """Validate exact layout and all payload hashes."""

    root = Path(input_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("authenticated run must be a real directory")
    complete = strict_json_file(root / _COMPLETE, label=_COMPLETE)
    if set(complete) != {"layout_version", "payload_count", "files"}:
        raise ValueError("run completion marker keys are invalid")
    if complete["layout_version"] != expected_layout_version:
        raise ValueError("run layout version differs")
    files = complete["files"]
    if (
        not isinstance(files, dict)
        or not files
        or complete["payload_count"] != len(files)
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name == _COMPLETE
            for name in files
        )
    ):
        raise ValueError("run completion payload map is invalid")
    if required_payloads is not None and set(files) != required_payloads:
        raise ValueError("authenticated run payload set differs")
    if {path.name for path in root.iterdir()} != {*files, _COMPLETE}:
        raise ValueError("authenticated run file layout differs")
    for name, expected in files.items():
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or sha256_file(root / name) != expected
        ):
            raise ValueError(f"authenticated run payload checksum differs: {name}")
    return MappingProxyType(dict(complete))


def seal_authenticated_run_staging(
    staging_dir: str | Path,
    *,
    layout_version: str,
    required_payloads: frozenset[str],
) -> Mapping[str, object]:
    """Checksum an exact staging payload set and write its completion marker."""

    root = Path(staging_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("run staging path must be a real directory")
    if (
        not required_payloads
        or _COMPLETE in required_payloads
        or any(Path(name).name != name for name in required_payloads)
    ):
        raise ValueError("required staging payload names are invalid")
    if {path.name for path in root.iterdir()} != required_payloads:
        raise ValueError("run staging payload set differs")
    files = {
        name: sha256_file(root / name)
        for name in sorted(required_payloads)
    }
    complete = {
        "layout_version": layout_version,
        "payload_count": len(files),
        "files": files,
    }
    (root / _COMPLETE).write_bytes(canonical_json_bytes(complete))
    return load_authenticated_run_directory(
        root,
        expected_layout_version=layout_version,
        required_payloads=required_payloads,
    )


def publish_authenticated_run_directory(
    output_dir: str | Path,
    *,
    layout_version: str,
    payloads: Mapping[str, bytes],
    validate_staging: Callable[[Path], None] | None = None,
) -> Mapping[str, object]:
    """Publish payloads through staging and a no-replace atomic rename."""

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite immutable output: {destination}"
        )
    if not isinstance(layout_version, str) or not layout_version:
        raise ValueError("layout_version must be non-empty")
    if not isinstance(payloads, Mapping) or not payloads:
        raise ValueError("payloads must be a non-empty mapping")
    normalized: dict[str, bytes] = {}
    for name, payload in payloads.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name == _COMPLETE
        ):
            raise ValueError("authenticated payload name is unsafe")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("authenticated payloads must be non-empty bytes")
        normalized[name] = payload
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        for name, payload in sorted(normalized.items()):
            (staging / name).write_bytes(payload)
        seal_authenticated_run_staging(
            staging,
            required_payloads=frozenset(normalized),
            layout_version=layout_version,
        )
        if validate_staging is not None:
            validate_staging(staging)
        atomic_rename_noreplace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return load_authenticated_run_directory(
        destination,
        expected_layout_version=layout_version,
        required_payloads=frozenset(normalized),
    )


__all__ = (
    "VERIFICATION_EVALUATION_RUN_LAYOUT_VERSION",
    "VERIFICATION_TRAINING_RUN_LAYOUT_VERSION",
    "canonical_json_bytes",
    "load_authenticated_run_directory",
    "publish_authenticated_run_directory",
    "seal_authenticated_run_staging",
    "sha256_file",
    "strict_json_file",
)
