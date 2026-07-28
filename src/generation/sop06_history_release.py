"""Resumable immutable entry releases for SOP06 history-only observations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
from types import MappingProxyType

import numpy as np

from src.datasets.sop06_history_bev import (
    SOP06_HISTORY_SHARD_VERSION,
    Sop06HistoryBevSample,
    Sop06HistoryShardProvenance,
    load_sop06_history_shard,
    load_sop06_history_shard_checkpoint,
    write_sop06_history_shard,
)
from src.generation.sop05r_contracts import load_sop05r_teb_config
from src.utils.atomic_publish import atomic_rename_noreplace
from src.utils.config import load_config

from .sop06_finalized_source import (
    Sop06AcceptedFinalRecord,
    Sop06FinalizedSource,
    load_sop06_finalized_source,
)
from .sop06_pipeline import render_sop06_single_input


SOP06_HISTORY_RELEASE_VERSION = "sop06_history_bev_release_v1"
_REQUEST = "request.json"
_MANIFEST = "manifest.json"
_CHECKSUMS = "checksums.json"
_COMPLETE = "COMPLETE.json"
_SHARDS = "shards"
_ROOT_FILES = frozenset({_REQUEST, _MANIFEST, _CHECKSUMS, _COMPLETE, _SHARDS})
_SOURCE_FAMILIES = frozenset({"natural", "a_supplement"})
_SOURCE_MODES = frozenset({"complete_mother", "partial_m6_reconstruction"})
_SPLITS = frozenset({"train", "calibration", "val", "test"})
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Sop06HistoryReleaseRequest:
    source_family: str
    source_mode: str
    source_root: Path
    final_scenario_root: Path
    split: str
    output_dir: Path
    source_cache_root: Path | None = None
    workers: int = 1
    samples_per_shard: int = 128
    sop03_root: Path | None = None
    long40_human_artifact: Path | None = None
    base_state_start: int | None = None
    max_base_states: int | None = None
    base_config_path: Path | None = None
    generator_config_path: Path | None = None


@dataclass(frozen=True)
class Sop06HistoryReleaseResult:
    output_dir: Path
    source_family: str
    source_mode: str
    source_publication_semantic_digest: str
    split: str
    sample_count: int
    shard_count: int
    reused_shard_count: int
    manifest_digest: str


@dataclass(frozen=True)
class LoadedSop06HistoryRelease:
    root: Path
    request_identity: str
    source_family: str
    source_mode: str
    source_publication_semantic_digest: str
    final_release_identity: str
    split: str
    sample_count: int
    shard_count: int
    scenario_ids_digest: str
    manifest_digest: str
    request: Mapping[str, object]
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "request", MappingProxyType(dict(self.request)))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


@dataclass(frozen=True)
class _ForkRenderedPayload:
    sample_id: str
    mother_id: str
    split: str
    regime: str
    bev_history: np.ndarray
    state_channels: np.ndarray
    renderer_metadata: dict[str, str]


RenderOne = Callable[
    [Sop06FinalizedSource, Sop06AcceptedFinalRecord],
    Sop06HistoryBevSample,
]
ProgressCallback = Callable[[int, int, bool], None]


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
        raise ValueError("SOP06 release metadata must be canonical JSON") from exc


def _json_file(value: object) -> bytes:
    return _canonical_json(value) + b"\n"


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read SOP06 release JSON: {path.name}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"failed to checksum SOP06 release file: {path.name}") from exc
    return digest.hexdigest()


def _repository_relative(path: Path, *, name: str) -> str:
    candidate = path if path.is_absolute() else (_REPOSITORY_ROOT / path)
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(_REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} must be inside the repository") from exc
    return relative.as_posix()


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _request_payload(request: Sop06HistoryReleaseRequest) -> dict[str, object]:
    if not isinstance(request, Sop06HistoryReleaseRequest):
        raise TypeError("request must be a Sop06HistoryReleaseRequest")
    if request.source_family not in _SOURCE_FAMILIES:
        raise ValueError("source_family is invalid")
    if request.source_mode not in _SOURCE_MODES:
        raise ValueError("source_mode is invalid")
    if request.split not in _SPLITS:
        raise ValueError("split is invalid")
    _positive_int(request.workers, name="workers")
    _positive_int(request.samples_per_shard, name="samples_per_shard")
    if request.output_dir.resolve() in {
        request.source_root.resolve(),
        request.final_scenario_root.resolve(),
    }:
        raise ValueError("output_dir must differ from both input roots")
    partial_values = (
        request.sop03_root,
        request.long40_human_artifact,
        request.base_state_start,
        request.max_base_states,
        request.base_config_path,
        request.generator_config_path,
    )
    if request.source_mode == "partial_m6_reconstruction":
        if any(value is None for value in partial_values):
            raise ValueError("partial-M6 release arguments are required")
        assert request.base_state_start is not None
        if (
            isinstance(request.base_state_start, bool)
            or not isinstance(request.base_state_start, int)
            or request.base_state_start < 0
        ):
            raise ValueError("base_state_start must be a nonnegative integer")
        _positive_int(request.max_base_states, name="max_base_states")
    elif any(value is not None for value in partial_values):
        raise ValueError("partial-M6 arguments are forbidden for complete mothers")
    payload: dict[str, object] = {
        "version": SOP06_HISTORY_RELEASE_VERSION,
        "source_family": request.source_family,
        "source_mode": request.source_mode,
        "source_root": _repository_relative(request.source_root, name="source_root"),
        "final_scenario_root": _repository_relative(
            request.final_scenario_root,
            name="final_scenario_root",
        ),
        "split": request.split,
        "samples_per_shard": request.samples_per_shard,
    }
    if request.source_mode == "partial_m6_reconstruction":
        assert request.sop03_root is not None
        assert request.long40_human_artifact is not None
        assert request.base_config_path is not None
        assert request.generator_config_path is not None
        payload["partial_m6"] = {
            "sop03_root": _repository_relative(
                request.sop03_root, name="sop03_root"
            ),
            "long40_human_artifact": _repository_relative(
                request.long40_human_artifact,
                name="long40_human_artifact",
            ),
            "base_state_start": request.base_state_start,
            "max_base_states": request.max_base_states,
            "base_config_path": _repository_relative(
                request.base_config_path, name="base_config_path"
            ),
            "generator_config_path": _repository_relative(
                request.generator_config_path,
                name="generator_config_path",
            ),
        }
    return payload


def _request_document(request: Sop06HistoryReleaseRequest) -> dict[str, object]:
    payload = _request_payload(request)
    identity = hashlib.sha256(
        b"sop06_history_release_request_v1\0" + _canonical_json(payload)
    ).hexdigest()
    return {
        "version": SOP06_HISTORY_RELEASE_VERSION,
        "request_identity": identity,
        "request": payload,
    }


def _load_source(request: Sop06HistoryReleaseRequest) -> Sop06FinalizedSource:
    read_source_root = request.source_cache_root or request.source_root
    if request.source_mode == "complete_mother":
        return load_sop06_finalized_source(
            source_mode=request.source_mode,
            source_root=read_source_root,
            final_scenario_root=request.final_scenario_root,
            split=request.split,
        )
    assert request.sop03_root is not None
    assert request.long40_human_artifact is not None
    assert request.base_state_start is not None
    assert request.max_base_states is not None
    assert request.base_config_path is not None
    assert request.generator_config_path is not None
    base_config = load_config(request.base_config_path)
    generator = load_sop05r_teb_config(request.generator_config_path)
    return load_sop06_finalized_source(
        source_mode=request.source_mode,
        source_root=read_source_root,
        final_scenario_root=request.final_scenario_root,
        split=request.split,
        sop03_root=request.sop03_root,
        long40_human_artifact=request.long40_human_artifact,
        base_state_start=request.base_state_start,
        max_base_states=request.max_base_states,
        base_config=base_config,
        source_config_digest=generator.digest,
        centerline_epsilon_m=(
            generator.occlusion.centerline_intersection_epsilon_m
        ),
    )


def _default_render_one(
    source: Sop06FinalizedSource,
    accepted: Sop06AcceptedFinalRecord,
) -> Sop06HistoryBevSample:
    renderer_input = source.resolve_history_renderer_input(accepted)
    observation = render_sop06_single_input(
        renderer_input,
        config=source.base_config,
    )
    return Sop06HistoryBevSample(
        sample_id=accepted.scenario_id,
        mother_id=accepted.mother_id,
        split=accepted.split,
        regime=accepted.regime,
        bev_history=observation.bev_history,
        state_channels=observation.state_channels,
        renderer_metadata=observation.metadata,
    )


_FORK_SOURCE: Sop06FinalizedSource | None = None


def _fork_render(accepted: Sop06AcceptedFinalRecord) -> _ForkRenderedPayload:
    if _FORK_SOURCE is None:
        raise RuntimeError("SOP06 fork source is unavailable")
    sample = _default_render_one(_FORK_SOURCE, accepted)
    return _ForkRenderedPayload(
        sample_id=sample.sample_id,
        mother_id=sample.mother_id,
        split=sample.split,
        regime=sample.regime,
        bev_history=sample.bev_history,
        state_channels=sample.state_channels,
        renderer_metadata=dict(sample.renderer_metadata),
    )


def _render_boundary(
    source: Sop06FinalizedSource,
    boundary: tuple[Sop06AcceptedFinalRecord, ...],
    *,
    workers: int,
    render_one: RenderOne,
) -> tuple[Sop06HistoryBevSample, ...]:
    prepare_name = (
        "prepare_history_boundary"
        if render_one is _default_render_one
        else "prepare_boundary"
    )
    prepare_boundary = getattr(source, prepare_name, None)
    if callable(prepare_boundary):
        source = prepare_boundary(boundary)
    if workers == 1 or len(boundary) == 1:
        return tuple(render_one(source, accepted) for accepted in boundary)
    if render_one is not _default_render_one:
        raise ValueError("custom render_one requires workers=1")
    if multiprocessing.get_start_method(allow_none=True) not in (None, "fork"):
        raise ValueError("parallel SOP06 rendering requires fork")
    global _FORK_SOURCE
    previous = _FORK_SOURCE
    _FORK_SOURCE = source
    try:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(boundary)),
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            payloads = executor.map(_fork_render, boundary, chunksize=1)
            return tuple(
                Sop06HistoryBevSample(
                    sample_id=payload.sample_id,
                    mother_id=payload.mother_id,
                    split=payload.split,
                    regime=payload.regime,
                    bev_history=payload.bev_history,
                    state_channels=payload.state_channels,
                    renderer_metadata=payload.renderer_metadata,
                )
                for payload in payloads
            )
    finally:
        _FORK_SOURCE = previous


def _scenario_ids_digest(sample_ids: tuple[str, ...]) -> str:
    return hashlib.sha256(
        b"sop06_history_scenario_ids_v1\0" + _canonical_json(list(sample_ids))
    ).hexdigest()


def _load_sop06_history_release(
    input_dir: str | Path,
    *,
    strict_shards: bool,
) -> LoadedSop06HistoryRelease:
    root = Path(input_dir)
    if not root.is_dir() or {path.name for path in root.iterdir()} != _ROOT_FILES:
        raise ValueError("SOP06 history release file set mismatch")
    checksums = _read_json(root / _CHECKSUMS)
    if not isinstance(checksums, dict) or set(checksums) != {_REQUEST, _MANIFEST}:
        raise ValueError("SOP06 release checksum schema mismatch")
    for name, expected in checksums.items():
        if not isinstance(expected, str) or _sha256_file(root / name) != expected:
            raise ValueError(f"SOP06 release checksum mismatch: {name}")
    request_document = _read_json(root / _REQUEST)
    manifest = _read_json(root / _MANIFEST)
    complete = _read_json(root / _COMPLETE)
    if not all(isinstance(value, dict) for value in (request_document, manifest, complete)):
        raise ValueError("SOP06 release root metadata must be objects")
    if (
        request_document.get("version") != SOP06_HISTORY_RELEASE_VERSION
        or manifest.get("version") != SOP06_HISTORY_RELEASE_VERSION
        or complete.get("version") != SOP06_HISTORY_RELEASE_VERSION
    ):
        raise ValueError("SOP06 history release version mismatch")
    request_payload = request_document.get("request")
    if not isinstance(request_payload, dict):
        raise ValueError("SOP06 release request payload is invalid")
    request_identity = hashlib.sha256(
        b"sop06_history_release_request_v1\0"
        + _canonical_json(request_payload)
    ).hexdigest()
    if request_document.get("request_identity") != request_identity:
        raise ValueError("SOP06 release request identity mismatch")
    descriptors = manifest.get("shards")
    if not isinstance(descriptors, list) or not descriptors:
        raise ValueError("SOP06 release shard descriptors are invalid")
    shard_root = root / _SHARDS
    expected_names = {f"shard-{index:05d}" for index in range(len(descriptors))}
    if {path.name for path in shard_root.iterdir()} != expected_names:
        raise ValueError("SOP06 release child shard set mismatch")
    sample_ids: list[str] = []
    seen_sample_ids: set[str] = set()
    mother_keys: set[tuple[str, str]] = set()
    source_digest = manifest.get("source_publication_semantic_digest")
    final_identity = manifest.get("final_release_identity")
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict):
            raise ValueError("SOP06 release shard descriptor must be an object")
        relative = f"shards/shard-{index:05d}"
        if descriptor.get("shard_index") != index or descriptor.get("relative_root") != relative:
            raise ValueError("SOP06 release shard descriptor ordering differs")
        if strict_shards:
            shard = load_sop06_history_shard(root / relative)
            shard_sample_ids = tuple(sample.sample_id for sample in shard.samples)
            shard_mother_ids = tuple(sample.mother_id for sample in shard.samples)
        else:
            shard = load_sop06_history_shard_checkpoint(root / relative)
            shard_sample_ids = shard.sample_ids
            shard_mother_ids = shard.mother_ids
        if (
            shard.semantic_digest != descriptor.get("semantic_digest")
            or len(shard_sample_ids) != descriptor.get("sample_count")
            or shard.provenance.source_family != manifest.get("source_family")
            or shard.provenance.source_mode != manifest.get("source_mode")
            or shard.provenance.split != manifest.get("split")
            or shard.provenance.source_publication_semantic_digest != source_digest
            or shard.provenance.final_release_identity != final_identity
        ):
            raise ValueError("SOP06 release child shard identity differs")
        for sample_id, mother_id in zip(shard_sample_ids, shard_mother_ids):
            if sample_id in seen_sample_ids:
                raise ValueError("duplicate sample_id across SOP06 release shards")
            key = (shard.provenance.split, mother_id)
            if key in mother_keys:
                raise ValueError("duplicate mother identity across SOP06 release shards")
            sample_ids.append(sample_id)
            seen_sample_ids.add(sample_id)
            mother_keys.add(key)
    ordered_ids = tuple(sample_ids)
    if ordered_ids != tuple(sorted(ordered_ids)):
        raise ValueError("SOP06 release sample IDs are not globally sorted")
    scenario_digest = _scenario_ids_digest(ordered_ids)
    manifest_digest = hashlib.sha256(
        b"sop06_history_release_manifest_v1\0" + _canonical_json(manifest)
    ).hexdigest()
    if (
        manifest.get("sample_count") != len(ordered_ids)
        or manifest.get("shard_count") != len(descriptors)
        or manifest.get("scenario_ids_digest") != scenario_digest
        or complete
        != {
            "version": SOP06_HISTORY_RELEASE_VERSION,
            "request_identity": request_identity,
            "sample_count": len(ordered_ids),
            "shard_count": len(descriptors),
            "scenario_ids_digest": scenario_digest,
            "manifest_digest": manifest_digest,
        }
    ):
        raise ValueError("SOP06 release completion identity mismatch")
    return LoadedSop06HistoryRelease(
        root=root,
        request_identity=request_identity,
        source_family=str(manifest["source_family"]),
        source_mode=str(manifest["source_mode"]),
        source_publication_semantic_digest=str(source_digest),
        final_release_identity=str(final_identity),
        split=str(manifest["split"]),
        sample_count=len(ordered_ids),
        shard_count=len(descriptors),
        scenario_ids_digest=scenario_digest,
        manifest_digest=manifest_digest,
        request=request_payload,
        manifest=manifest,
    )


def load_sop06_history_release(
    input_dir: str | Path,
) -> LoadedSop06HistoryRelease:
    """Strictly audit release metadata and every observation array."""

    return _load_sop06_history_release(input_dir, strict_shards=True)


def load_sop06_history_release_checkpoint(
    input_dir: str | Path,
) -> LoadedSop06HistoryRelease:
    """Validate release checkpoints without opening observation arrays."""

    return _load_sop06_history_release(input_dir, strict_shards=False)


def _result(
    loaded: LoadedSop06HistoryRelease,
    *,
    reused_shard_count: int,
) -> Sop06HistoryReleaseResult:
    return Sop06HistoryReleaseResult(
        output_dir=loaded.root,
        source_family=loaded.source_family,
        source_mode=loaded.source_mode,
        source_publication_semantic_digest=(
            loaded.source_publication_semantic_digest
        ),
        split=loaded.split,
        sample_count=loaded.sample_count,
        shard_count=loaded.shard_count,
        reused_shard_count=reused_shard_count,
        manifest_digest=loaded.manifest_digest,
    )


def publish_sop06_history_release(
    request: Sop06HistoryReleaseRequest,
    *,
    render_one: RenderOne = _default_render_one,
    progress_callback: ProgressCallback | None = None,
) -> Sop06HistoryReleaseResult:
    request_document = _request_document(request)
    request_identity = str(request_document["request_identity"])
    output = request.output_dir
    if output.exists() or output.is_symlink():
        loaded = load_sop06_history_release_checkpoint(output)
        if loaded.request_identity != request_identity:
            raise ValueError("completed SOP06 release request differs")
        return _result(loaded, reused_shard_count=loaded.shard_count)

    in_progress = output.parent / f".{output.name}.inprogress"
    shards_root = in_progress / _SHARDS
    if in_progress.exists():
        if (in_progress / _COMPLETE).is_file():
            staged = load_sop06_history_release_checkpoint(in_progress)
            if staged.request_identity != request_identity:
                raise ValueError("completed in-progress SOP06 request differs")
            atomic_rename_noreplace(in_progress, output)
            published = replace(staged, root=output)
            return _result(published, reused_shard_count=published.shard_count)
        if {path.name for path in in_progress.iterdir()} != {_REQUEST, _SHARDS}:
            raise ValueError("in-progress SOP06 release file set mismatch")
        if _read_json(in_progress / _REQUEST) != request_document:
            raise ValueError("in-progress SOP06 release request differs")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        in_progress.mkdir()
        shards_root.mkdir()
        (in_progress / _REQUEST).write_bytes(_json_file(request_document))

    source = _load_source(request)
    accepted = tuple(sorted(source.accepted, key=lambda item: item.scenario_id))
    if not accepted:
        raise ValueError("SOP06 release contains no accepted final scenarios")
    if len({item.scenario_id for item in accepted}) != len(accepted):
        raise ValueError("SOP06 accepted scenario IDs are not unique")
    if len({(item.split, item.mother_id) for item in accepted}) != len(accepted):
        raise ValueError("SOP06 accepted mother identities are not unique")
    if any(item.split != request.split for item in accepted):
        raise ValueError("SOP06 accepted scenario split differs")

    boundaries = tuple(
        accepted[start : start + request.samples_per_shard]
        for start in range(0, len(accepted), request.samples_per_shard)
    )
    expected_names = {f"shard-{index:05d}" for index in range(len(boundaries))}
    actual_names = {path.name for path in shards_root.iterdir()}
    if not actual_names.issubset(expected_names):
        raise ValueError("in-progress SOP06 release contains unexpected shard")
    provenance = Sop06HistoryShardProvenance(
        source_family=request.source_family,
        source_mode=request.source_mode,
        split=request.split,
        source_publication_semantic_digest=(
            source.source_publication_semantic_digest
        ),
        final_release_identity=source.final_release_identity,
        final_scenario_root=_repository_relative(
            request.final_scenario_root,
            name="final_scenario_root",
        ),
    )
    descriptors: list[dict[str, object]] = []
    reused = 0
    for index, boundary in enumerate(boundaries):
        shard_path = shards_root / f"shard-{index:05d}"
        expected_ids = tuple(item.scenario_id for item in boundary)
        was_reused = shard_path.exists()
        if was_reused:
            shard = load_sop06_history_shard_checkpoint(shard_path)
            if (
                shard.provenance != provenance
                or shard.sample_ids != expected_ids
            ):
                raise ValueError("in-progress SOP06 shard differs from fixed boundary")
            reused += 1
        else:
            samples = _render_boundary(
                source,
                boundary,
                workers=request.workers,
                render_one=render_one,
            )
            if tuple(sorted(sample.sample_id for sample in samples)) != expected_ids:
                raise ValueError("rendered SOP06 sample IDs differ from boundary")
            write_sop06_history_shard(
                samples,
                shard_path,
                shard_index=index,
                expected_sample_count=len(boundary),
                provenance=provenance,
            )
            shard = load_sop06_history_shard_checkpoint(shard_path)
        descriptors.append(
            {
                "shard_index": index,
                "relative_root": f"shards/shard-{index:05d}",
                "sample_count": len(shard.sample_ids),
                "first_sample_id": shard.sample_ids[0],
                "last_sample_id": shard.sample_ids[-1],
                "semantic_digest": shard.semantic_digest,
            }
        )
        if progress_callback is not None:
            progress_callback(index + 1, len(boundaries), was_reused)

    sample_ids = tuple(item.scenario_id for item in accepted)
    scenario_digest = _scenario_ids_digest(sample_ids)
    manifest = {
        "version": SOP06_HISTORY_RELEASE_VERSION,
        "request_identity": request_identity,
        "source_family": request.source_family,
        "source_mode": request.source_mode,
        "source_publication_semantic_digest": (
            source.source_publication_semantic_digest
        ),
        "final_release_identity": source.final_release_identity,
        "final_scenario_root": provenance.final_scenario_root,
        "split": request.split,
        "sample_count": len(sample_ids),
        "shard_count": len(descriptors),
        "samples_per_shard": request.samples_per_shard,
        "scenario_ids_digest": scenario_digest,
        "renderer_layout_version": "bev_history2_state9_v1",
        "child_shard_version": SOP06_HISTORY_SHARD_VERSION,
        "shards": descriptors,
    }
    manifest_digest = hashlib.sha256(
        b"sop06_history_release_manifest_v1\0" + _canonical_json(manifest)
    ).hexdigest()
    (in_progress / _MANIFEST).write_bytes(_json_file(manifest))
    checksums = {
        name: _sha256_file(in_progress / name)
        for name in (_REQUEST, _MANIFEST)
    }
    (in_progress / _CHECKSUMS).write_bytes(_json_file(checksums))
    (in_progress / _COMPLETE).write_bytes(
        _json_file(
            {
                "version": SOP06_HISTORY_RELEASE_VERSION,
                "request_identity": request_identity,
                "sample_count": len(sample_ids),
                "shard_count": len(descriptors),
                "scenario_ids_digest": scenario_digest,
                "manifest_digest": manifest_digest,
            }
        )
    )
    staged = load_sop06_history_release_checkpoint(in_progress)
    atomic_rename_noreplace(in_progress, output)
    loaded = replace(staged, root=output)
    return _result(loaded, reused_shard_count=reused)


__all__ = (
    "LoadedSop06HistoryRelease",
    "SOP06_HISTORY_RELEASE_VERSION",
    "Sop06HistoryReleaseRequest",
    "Sop06HistoryReleaseResult",
    "load_sop06_history_release",
    "load_sop06_history_release_checkpoint",
    "publish_sop06_history_release",
)
