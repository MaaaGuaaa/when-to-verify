"""Immutable catalogs over completed SOP06 history-BEV entry releases."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from types import MappingProxyType
from typing import Mapping, Sequence

from src.generation.sop06_history_release import (
    load_sop06_history_release_checkpoint,
)
from src.utils.atomic_publish import atomic_rename_noreplace

from .sop06_history_bev import load_sop06_history_shard_checkpoint


SOP06_HISTORY_CATALOG_VERSION = "sop06_history_bev_catalog_v1"
_MANIFEST = "manifest.json"
_CHECKSUMS = "checksums.json"
_COMPLETE = "COMPLETE.json"
_FILES = frozenset({_MANIFEST, _CHECKSUMS, _COMPLETE})
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Sop06HistoryCatalogEntry:
    entry_id: str
    relative_root: str
    source_family: str
    source_mode: str
    split: str
    source_publication_semantic_digest: str
    final_release_identity: str
    sample_count: int
    shard_count: int
    scenario_ids_digest: str
    release_manifest_digest: str


@dataclass(frozen=True)
class LoadedSop06HistoryCatalog:
    root: Path
    entries: tuple[Sop06HistoryCatalogEntry, ...]
    entry_count: int
    sample_count: int
    catalog_digest: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("SOP06 catalog metadata must be canonical JSON") from exc


def _json_file(value: object) -> bytes:
    return _canonical_json(value) + b"\n"


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read SOP06 catalog JSON: {path.name}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"failed to checksum SOP06 catalog file: {path.name}") from exc
    return digest.hexdigest()


def _relative_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("SOP06 catalog entry root must be inside repository") from exc


def _safe_entry_root(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("SOP06 catalog relative_root is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("SOP06 catalog relative_root is unsafe")
    root = (_REPOSITORY_ROOT / relative).resolve()
    try:
        root.relative_to(_REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("SOP06 catalog relative_root escapes repository") from exc
    return root


def _set_digest(prefix: bytes, values: Sequence[str]) -> str:
    return hashlib.sha256(prefix + _canonical_json(sorted(values))).hexdigest()


def _collect_entries(
    roots: Sequence[Path],
) -> tuple[
    tuple[Sop06HistoryCatalogEntry, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    if not roots:
        raise ValueError("SOP06 catalog requires at least one release")
    if len({root.resolve() for root in roots}) != len(roots):
        raise ValueError("SOP06 catalog release roots must be unique")
    entries: list[Sop06HistoryCatalogEntry] = []
    sample_ids: list[str] = []
    seen_samples: set[str] = set()
    mother_identities: list[str] = []
    seen_mothers: set[tuple[str, str]] = set()
    for root in roots:
        release = load_sop06_history_release_checkpoint(root)
        relative_root = _relative_root(root)
        entry = Sop06HistoryCatalogEntry(
            entry_id=f"{release.source_family}:{root.name}",
            relative_root=relative_root,
            source_family=release.source_family,
            source_mode=release.source_mode,
            split=release.split,
            source_publication_semantic_digest=(
                release.source_publication_semantic_digest
            ),
            final_release_identity=release.final_release_identity,
            sample_count=release.sample_count,
            shard_count=release.shard_count,
            scenario_ids_digest=release.scenario_ids_digest,
            release_manifest_digest=release.manifest_digest,
        )
        entries.append(entry)
        descriptors = release.manifest.get("shards")
        if not isinstance(descriptors, (list, tuple)):
            raise ValueError("SOP06 catalog release descriptors are invalid")
        for descriptor in descriptors:
            if not isinstance(descriptor, Mapping):
                raise ValueError("SOP06 catalog shard descriptor is invalid")
            relative_shard = descriptor.get("relative_root")
            if not isinstance(relative_shard, str):
                raise ValueError("SOP06 catalog shard path is invalid")
            shard = load_sop06_history_shard_checkpoint(root / relative_shard)
            for sample_id, mother_id in zip(shard.sample_ids, shard.mother_ids):
                if sample_id in seen_samples:
                    raise ValueError("duplicate sample_id across SOP06 catalog entries")
                mother_key = (shard.provenance.split, mother_id)
                if mother_key in seen_mothers:
                    raise ValueError("duplicate mother identity across SOP06 catalog entries")
                seen_samples.add(sample_id)
                seen_mothers.add(mother_key)
                sample_ids.append(sample_id)
                mother_identities.append(
                    f"{shard.provenance.split}\0{mother_id}"
                )
    ordered_entries = tuple(
        sorted(entries, key=lambda item: (item.source_family, item.split, item.relative_root))
    )
    if len({entry.entry_id for entry in ordered_entries}) != len(ordered_entries):
        raise ValueError("SOP06 catalog entry IDs must be unique")
    return ordered_entries, tuple(sample_ids), tuple(mother_identities)


def _entry_dict(entry: Sop06HistoryCatalogEntry) -> dict[str, object]:
    return {
        "entry_id": entry.entry_id,
        "relative_root": entry.relative_root,
        "source_family": entry.source_family,
        "source_mode": entry.source_mode,
        "split": entry.split,
        "source_publication_semantic_digest": (
            entry.source_publication_semantic_digest
        ),
        "final_release_identity": entry.final_release_identity,
        "sample_count": entry.sample_count,
        "shard_count": entry.shard_count,
        "scenario_ids_digest": entry.scenario_ids_digest,
        "release_manifest_digest": entry.release_manifest_digest,
    }


def _manifest_for_roots(roots: Sequence[Path]) -> dict[str, object]:
    entries, sample_ids, mother_identities = _collect_entries(roots)
    return {
        "version": SOP06_HISTORY_CATALOG_VERSION,
        "entry_count": len(entries),
        "sample_count": len(sample_ids),
        "source_families": sorted({entry.source_family for entry in entries}),
        "splits": sorted({entry.split for entry in entries}),
        "sample_ids_digest": _set_digest(
            b"sop06_history_catalog_samples_v1\0", sample_ids
        ),
        "mother_identities_digest": _set_digest(
            b"sop06_history_catalog_mothers_v1\0", mother_identities
        ),
        "entries": [_entry_dict(entry) for entry in entries],
    }


def publish_sop06_history_catalog(
    entry_roots: Sequence[str | Path],
    output_dir: str | Path,
) -> LoadedSop06HistoryCatalog:
    if isinstance(entry_roots, (str, bytes)) or not isinstance(entry_roots, Sequence):
        raise TypeError("entry_roots must be a sequence")
    roots = tuple(Path(root) for root in entry_roots)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite SOP06 catalog: {output}")
    if output.resolve() in {root.resolve() for root in roots}:
        raise ValueError("SOP06 catalog cannot overwrite an entry root")
    manifest = _manifest_for_roots(roots)
    catalog_digest = hashlib.sha256(
        b"sop06_history_catalog_manifest_v1\0" + _canonical_json(manifest)
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        (staging / _MANIFEST).write_bytes(_json_file(manifest))
        (staging / _CHECKSUMS).write_bytes(
            _json_file({_MANIFEST: _sha256_file(staging / _MANIFEST)})
        )
        (staging / _COMPLETE).write_bytes(
            _json_file(
                {
                    "version": SOP06_HISTORY_CATALOG_VERSION,
                    "entry_count": manifest["entry_count"],
                    "sample_count": manifest["sample_count"],
                    "catalog_digest": catalog_digest,
                }
            )
        )
        staged = load_sop06_history_catalog(staging)
        atomic_rename_noreplace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    loaded = load_sop06_history_catalog(output)
    if loaded.catalog_digest != staged.catalog_digest:
        raise ValueError("published SOP06 catalog differs from staging")
    return loaded


def load_sop06_history_catalog(
    input_dir: str | Path,
) -> LoadedSop06HistoryCatalog:
    root = Path(input_dir)
    if not root.is_dir() or {path.name for path in root.iterdir()} != _FILES:
        raise ValueError("SOP06 catalog file set mismatch")
    checksums = _read_json(root / _CHECKSUMS)
    if not isinstance(checksums, dict) or set(checksums) != {_MANIFEST}:
        raise ValueError("SOP06 catalog checksum schema mismatch")
    if _sha256_file(root / _MANIFEST) != checksums[_MANIFEST]:
        raise ValueError("SOP06 catalog checksum mismatch")
    manifest = _read_json(root / _MANIFEST)
    complete = _read_json(root / _COMPLETE)
    if not isinstance(manifest, dict) or not isinstance(complete, dict):
        raise ValueError("SOP06 catalog manifest/complete must be objects")
    if manifest.get("version") != SOP06_HISTORY_CATALOG_VERSION:
        raise ValueError("SOP06 catalog version mismatch")
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("SOP06 catalog entries are invalid")
    try:
        declared_entries = tuple(
            Sop06HistoryCatalogEntry(**entry) for entry in raw_entries
        )
    except (TypeError, KeyError) as exc:
        raise ValueError("SOP06 catalog entry schema mismatch") from exc
    roots = tuple(_safe_entry_root(entry.relative_root) for entry in declared_entries)
    rebuilt = _manifest_for_roots(roots)
    if rebuilt != manifest:
        raise ValueError("SOP06 catalog manifest differs from referenced releases")
    catalog_digest = hashlib.sha256(
        b"sop06_history_catalog_manifest_v1\0" + _canonical_json(manifest)
    ).hexdigest()
    if complete != {
        "version": SOP06_HISTORY_CATALOG_VERSION,
        "entry_count": manifest["entry_count"],
        "sample_count": manifest["sample_count"],
        "catalog_digest": catalog_digest,
    }:
        raise ValueError("SOP06 catalog completion identity mismatch")
    return LoadedSop06HistoryCatalog(
        root=root,
        entries=declared_entries,
        entry_count=int(manifest["entry_count"]),
        sample_count=int(manifest["sample_count"]),
        catalog_digest=catalog_digest,
        manifest=manifest,
    )


__all__ = (
    "LoadedSop06HistoryCatalog",
    "SOP06_HISTORY_CATALOG_VERSION",
    "Sop06HistoryCatalogEntry",
    "load_sop06_history_catalog",
    "publish_sop06_history_catalog",
)
