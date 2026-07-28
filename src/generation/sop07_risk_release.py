"""Resumable SOP07 risk releases built from persisted SOP06 observations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from src.contracts import RiskSample, build_grid_spec
from src.datasets.risk_dataset import build_risk_sample
from src.datasets.shard_writer import (
    RISK_SHARD_LAYOUT_VERSION,
    LoadedRiskShard,
    load_risk_shard,
    write_risk_shard,
)
from src.datasets.sop06_history_bev import (
    Sop06HistoryBevSample,
    Sop06HistoryShardCheckpoint,
    load_sop06_history_shard,
    load_sop06_history_shard_checkpoint,
)
from src.generation.observation_renderer import RenderedObservation
from src.generation.sop05r_contracts import load_sop05r_teb_config
from src.generation.sop06_finalized_source import (
    Sop06AcceptedFinalRecord,
    Sop06FinalizedSource,
    load_sop06_finalized_source,
)
from src.generation.sop06_history_release import (
    LoadedSop06HistoryRelease,
    load_sop06_history_release_checkpoint,
)
from src.generation.sop06_pipeline import build_sop06_single_risk_input
from src.utils.atomic_publish import atomic_rename_noreplace
from src.utils.config import config_digest, load_config


SOP07_RISK_RELEASE_VERSION = "sop07_single_risk_release_v1"
_REQUEST = "request.json"
_MANIFEST = "manifest.json"
_CHECKSUMS = "checksums.json"
_COMPLETE = "COMPLETE.json"
_SHARDS = "shards"
_ROOT_FILES = frozenset({_REQUEST, _MANIFEST, _CHECKSUMS, _COMPLETE, _SHARDS})
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_PROVENANCE = (
    "base_recording_id",
    "base_session_id",
    "source_recording_id",
    "source_session_id",
    "source_object_id",
    "source_snippet_id",
    "seed_namespace",
    "base_config_digest",
)


@dataclass(frozen=True)
class Sop07RiskReleaseRequest:
    sop06_release_root: Path
    output_dir: Path
    sop03_root: Path | None = None
    long40_human_artifact: Path | None = None


@dataclass(frozen=True)
class Sop07CompleteMotherLineage:
    """Explicit, repository-local sources for a legacy complete mother."""

    sop03_root: Path
    long40_human_artifact: Path


@dataclass(frozen=True)
class Sop07ShardBinding:
    sop06_release_manifest_digest: str
    sop06_shard_index: int
    sop06_shard_semantic_digest: str
    source_family: str
    source_mode: str


@dataclass(frozen=True)
class Sop07RiskReleaseResult:
    output_dir: Path
    split: str
    sample_count: int
    shard_count: int
    reused_shard_count: int
    manifest_digest: str


@dataclass(frozen=True)
class LoadedSop07RiskRelease:
    root: Path
    request_identity: str
    split: str
    sample_count: int
    shard_count: int
    sop06_release_manifest_digest: str
    manifest_digest: str
    request: Mapping[str, object]
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "request", MappingProxyType(dict(self.request)))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


BuildOne = Callable[
    [
        Sop06FinalizedSource,
        Sop06AcceptedFinalRecord,
        Sop06HistoryBevSample,
        Sop07ShardBinding,
    ],
    RiskSample,
]
ProgressCallback = Callable[[int, int, bool], None]
SourceLoader = Callable[
    [LoadedSop06HistoryRelease],
    Sop06FinalizedSource,
]


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
        raise ValueError("SOP07 release metadata must be canonical JSON") from exc


def _json_file(value: object) -> bytes:
    return _canonical_json(value) + b"\n"


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read SOP07 release JSON: {path.name}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"failed to checksum SOP07 release file: {path.name}") from exc
    return digest.hexdigest()


def _repository_relative(path: Path, *, name: str) -> str:
    candidate = path if path.is_absolute() else (_REPOSITORY_ROOT / path)
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(_REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} must be inside the repository") from exc
    return relative.as_posix()


def _repository_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a repository-relative path")
    resolved = (_REPOSITORY_ROOT / value).resolve()
    try:
        resolved.relative_to(_REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} escapes the repository") from exc
    return resolved


def _complete_mother_lineage_from_request(
    request: Sop07RiskReleaseRequest,
    upstream: LoadedSop06HistoryRelease,
) -> Sop07CompleteMotherLineage | None:
    supplied = (request.sop03_root, request.long40_human_artifact)
    if supplied == (None, None):
        return None
    if any(value is None for value in supplied):
        raise ValueError(
            "complete-mother lineage requires both SOP3 and Long40 roots"
        )
    if upstream.source_mode != "complete_mother":
        raise ValueError(
            "complete-mother lineage is forbidden for partial-M6 SOP06 releases"
        )
    if request.sop03_root is None or request.long40_human_artifact is None:
        raise RuntimeError("complete-mother lineage roots are unavailable")
    return Sop07CompleteMotherLineage(
        sop03_root=request.sop03_root,
        long40_human_artifact=request.long40_human_artifact,
    )


def _complete_mother_lineage_from_payload(
    payload: Mapping[str, object],
) -> Sop07CompleteMotherLineage | None:
    raw = payload.get("complete_mother_lineage")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != {
        "sop03_root",
        "long40_human_artifact",
    }:
        raise ValueError("SOP07 complete-mother lineage payload is invalid")
    return Sop07CompleteMotherLineage(
        sop03_root=_repository_path(raw["sop03_root"], name="SOP07 SOP3 root"),
        long40_human_artifact=_repository_path(
            raw["long40_human_artifact"],
            name="SOP07 Long40 human artifact",
        ),
    )


def _join_digest(pairs: Sequence[tuple[str, str]]) -> str:
    return hashlib.sha256(
        b"sop07_sop06_sample_mother_join_v1\0"
        + _canonical_json([[sample_id, mother_id] for sample_id, mother_id in pairs])
    ).hexdigest()


def _request_payload(
    request: Sop07RiskReleaseRequest,
    upstream: LoadedSop06HistoryRelease,
    *,
    base_config_digest: str,
    risk_config_digest: str,
) -> dict[str, object]:
    if not isinstance(request, Sop07RiskReleaseRequest):
        raise TypeError("request must be a Sop07RiskReleaseRequest")
    if request.output_dir.resolve() == request.sop06_release_root.resolve():
        raise ValueError("SOP07 output_dir must differ from the SOP06 release root")
    lineage = _complete_mother_lineage_from_request(request, upstream)
    payload: dict[str, object] = {
        "version": SOP07_RISK_RELEASE_VERSION,
        "sop06_release_root": _repository_relative(
            request.sop06_release_root,
            name="sop06_release_root",
        ),
        "sop06_release_request_identity": upstream.request_identity,
        "sop06_release_manifest_digest": upstream.manifest_digest,
        "source_family": upstream.source_family,
        "source_mode": upstream.source_mode,
        "split": upstream.split,
        "sample_count": upstream.sample_count,
        "shard_count": upstream.shard_count,
        "base_config_digest": base_config_digest,
        "risk_config_digest": risk_config_digest,
    }
    if lineage is not None:
        payload["complete_mother_lineage"] = {
            "sop03_root": _repository_relative(
                lineage.sop03_root,
                name="SOP07 SOP3 root",
            ),
            "long40_human_artifact": _repository_relative(
                lineage.long40_human_artifact,
                name="SOP07 Long40 human artifact",
            ),
        }
    return payload


def _request_document(payload: Mapping[str, object]) -> dict[str, object]:
    identity = hashlib.sha256(
        b"sop07_single_risk_release_request_v1\0" + _canonical_json(payload)
    ).hexdigest()
    return {
        "version": SOP07_RISK_RELEASE_VERSION,
        "request_identity": identity,
        "request": dict(payload),
    }


def _default_source_loader(
    upstream: LoadedSop06HistoryRelease,
    *,
    complete_mother_lineage: Sop07CompleteMotherLineage | None = None,
) -> Sop06FinalizedSource:
    request = upstream.request
    source_mode = request.get("source_mode")
    kwargs: dict[str, object] = {
        "source_mode": source_mode,
        "source_root": _repository_path(
            request.get("source_root"),
            name="SOP06 source_root",
        ),
        "final_scenario_root": _repository_path(
            request.get("final_scenario_root"),
            name="SOP06 final_scenario_root",
        ),
        "split": request.get("split"),
    }
    if source_mode == "complete_mother":
        if complete_mother_lineage is not None:
            kwargs.update(
                {
                    "sop03_root": complete_mother_lineage.sop03_root,
                    "long40_human_artifact": (
                        complete_mother_lineage.long40_human_artifact
                    ),
                }
            )
    elif source_mode == "partial_m6_reconstruction":
        if complete_mother_lineage is not None:
            raise ValueError(
                "complete-mother lineage is invalid for partial-M6 SOP06 releases"
            )
        partial = request.get("partial_m6")
        if not isinstance(partial, Mapping):
            raise ValueError("SOP06 partial-M6 request metadata is missing")
        base_config = load_config(
            _repository_path(
                partial.get("base_config_path"),
                name="SOP06 partial base_config_path",
            )
        )
        generator = load_sop05r_teb_config(
            _repository_path(
                partial.get("generator_config_path"),
                name="SOP06 partial generator_config_path",
            )
        )
        kwargs.update(
            {
                "sop03_root": _repository_path(
                    partial.get("sop03_root"),
                    name="SOP06 partial sop03_root",
                ),
                "long40_human_artifact": _repository_path(
                    partial.get("long40_human_artifact"),
                    name="SOP06 partial long40_human_artifact",
                ),
                "base_state_start": partial.get("base_state_start"),
                "max_base_states": partial.get("max_base_states"),
                "base_config": base_config,
                "source_config_digest": generator.digest,
                "centerline_epsilon_m": (
                    generator.occlusion.centerline_intersection_epsilon_m
                ),
            }
        )
    else:
        raise ValueError("SOP06 source_mode is invalid")
    source = load_sop06_finalized_source(**kwargs)
    if (
        source.source_mode != upstream.source_mode
        or source.source_publication_semantic_digest
        != upstream.source_publication_semantic_digest
        or source.final_release_identity != upstream.final_release_identity
        or len(source.accepted) != upstream.sample_count
    ):
        raise ValueError("SOP05 oracle source differs from the SOP06 release")
    return source


def _validate_observation_join(
    observation: Sop06HistoryBevSample,
    accepted: Sop06AcceptedFinalRecord,
) -> None:
    if (
        observation.sample_id != accepted.scenario_id
        or observation.mother_id != accepted.mother_id
        or observation.split != accepted.split
        or observation.regime != accepted.regime
    ):
        raise ValueError(
            "SOP06 observation and SOP05 oracle sample_id/mother_id join failed"
        )


def _default_build_one(
    source: Sop06FinalizedSource,
    accepted: Sop06AcceptedFinalRecord,
    observation: Sop06HistoryBevSample,
    binding: Sop07ShardBinding,
) -> RiskSample:
    _validate_observation_join(observation, accepted)
    resolved = source.resolve(accepted)
    publication = resolved.publication
    if (
        publication.sample_id != observation.sample_id
        or publication.mother_id != observation.mother_id
        or publication.split != observation.split
        or publication.regime != observation.regime
    ):
        raise ValueError("resolved SOP05 oracle differs from the SOP06 observation")
    base_state_id = observation.renderer_metadata.get("base_state_id")
    if (
        not isinstance(base_state_id, str)
        or base_state_id != publication.renderer_input.base_state.state_id
    ):
        raise ValueError("SOP06 observation base_state_id differs from SOP05 oracle")

    risk_input = build_sop06_single_risk_input(publication)
    provenance = dict(risk_input.provenance)
    missing = [
        key
        for key in _REQUIRED_PROVENANCE
        if not isinstance(provenance.get(key), str) or not provenance[key]
    ]
    if missing:
        raise ValueError(
            f"SOP05 oracle provenance is incomplete for SOP07: {sorted(missing)}"
        )
    observed_config_digest = config_digest(source.base_config)
    if provenance["base_config_digest"] != observed_config_digest:
        raise ValueError("SOP05 oracle base_config_digest differs")
    risk_input = replace(
        risk_input,
        provenance={
            **provenance,
            "sop06_history_release_manifest_digest": (
                binding.sop06_release_manifest_digest
            ),
            "sop06_history_shard_index": binding.sop06_shard_index,
            "sop06_history_shard_semantic_digest": (
                binding.sop06_shard_semantic_digest
            ),
            "sop06_source_family": binding.source_family,
            "sop06_source_mode": binding.source_mode,
        },
    )
    rendered = RenderedObservation(
        bev_history=observation.bev_history,
        state_channels=observation.state_channels,
        metadata=dict(observation.renderer_metadata),
    )
    risk_config = source.base_config.get("risk_gt")
    if not isinstance(risk_config, Mapping):
        raise ValueError("SOP05 base config has no risk_gt section")
    return build_risk_sample(
        risk_input,
        base_config=source.base_config,
        risk_config=risk_config,
        rendered_observation=rendered,
    )


def _validate_risk_binding(
    loaded: LoadedRiskShard,
    checkpoint: Sop06HistoryShardCheckpoint,
    binding: Sop07ShardBinding,
) -> None:
    sample_ids = tuple(sample.sample_id for sample in loaded.samples)
    if (
        loaded.summary.get("shard_index") != checkpoint.shard_index
        or sample_ids != checkpoint.sample_ids
    ):
        raise ValueError("SOP07 risk shard boundary differs from SOP06")
    pairs: list[tuple[str, str]] = []
    for sample, mother_id in zip(
        loaded.samples,
        checkpoint.mother_ids,
        strict=True,
    ):
        provenance = sample.metadata.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("SOP07 risk sample provenance is missing")
        if (
            provenance.get("mother_id") != mother_id
            or provenance.get("sop06_history_release_manifest_digest")
            != binding.sop06_release_manifest_digest
            or provenance.get("sop06_history_shard_index")
            != binding.sop06_shard_index
            or provenance.get("sop06_history_shard_semantic_digest")
            != binding.sop06_shard_semantic_digest
            or provenance.get("sop06_source_family") != binding.source_family
            or provenance.get("sop06_source_mode") != binding.source_mode
        ):
            raise ValueError("SOP07 risk sample SOP06 binding differs")
        pairs.append((sample.sample_id, mother_id))
    if len(set(pairs)) != len(pairs):
        raise ValueError("SOP07 risk sample/mother joins are not unique")


def _descriptor(
    loaded: LoadedRiskShard,
    checkpoint: Sop06HistoryShardCheckpoint,
) -> dict[str, object]:
    pairs = tuple(zip(checkpoint.sample_ids, checkpoint.mother_ids, strict=True))
    return {
        "shard_index": checkpoint.shard_index,
        "relative_root": f"shards/shard-{checkpoint.shard_index:05d}",
        "sample_count": len(checkpoint.sample_ids),
        "first_sample_id": checkpoint.sample_ids[0],
        "last_sample_id": checkpoint.sample_ids[-1],
        "sample_mother_join_digest": _join_digest(pairs),
        "sop06_history_shard_semantic_digest": checkpoint.semantic_digest,
        "risk_manifest_digest": loaded.manifest_digest,
        "risk_semantic_digest": loaded.semantic_digest,
    }


def _label_population(samples: Sequence[RiskSample]) -> dict[str, int]:
    population = {
        "hidden_target_count": 0,
        "current_visible_target_exclusion_count": 0,
        "empty_target_count": 0,
        "sample_count": 0,
    }
    for sample in samples:
        provenance = sample.metadata.get("provenance")
        audit = sample.metadata.get("label_audit")
        if not isinstance(provenance, Mapping) or not isinstance(audit, Mapping):
            raise ValueError("SOP07 sample label population metadata is missing")
        target_present = provenance.get("target_present")
        target_observed = provenance.get("target_currently_observed")
        has_hidden = audit.get("has_hidden_target")
        if not all(
            isinstance(value, bool)
            for value in (target_present, target_observed, has_hidden)
        ):
            raise ValueError("SOP07 target eligibility flags must be booleans")
        if target_observed and not target_present:
            raise ValueError("an absent SOP07 target cannot be currently observed")
        if has_hidden != (target_present and not target_observed):
            raise ValueError("SOP07 hidden-target eligibility flags disagree")
        if has_hidden:
            population["hidden_target_count"] += 1
        elif target_observed:
            population["current_visible_target_exclusion_count"] += 1
        else:
            population["empty_target_count"] += 1
        population["sample_count"] += 1
    return population


def _add_population(
    total: dict[str, int],
    shard: Mapping[str, int],
) -> None:
    if set(total) != set(shard):
        raise ValueError("SOP07 label population schemas differ")
    for key, value in shard.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("SOP07 label population counts must be nonnegative")
        total[key] += value


def _load_sop07_risk_release(input_dir: str | Path) -> LoadedSop07RiskRelease:
    root = Path(input_dir)
    if not root.is_dir() or {path.name for path in root.iterdir()} != _ROOT_FILES:
        raise ValueError("SOP07 risk release file set mismatch")
    checksums = _read_json(root / _CHECKSUMS)
    if not isinstance(checksums, dict) or set(checksums) != {_REQUEST, _MANIFEST}:
        raise ValueError("SOP07 release checksum schema mismatch")
    for name, expected in checksums.items():
        if not isinstance(expected, str) or _sha256_file(root / name) != expected:
            raise ValueError(f"SOP07 release checksum mismatch: {name}")
    request_document = _read_json(root / _REQUEST)
    manifest = _read_json(root / _MANIFEST)
    complete = _read_json(root / _COMPLETE)
    if not all(isinstance(value, dict) for value in (request_document, manifest, complete)):
        raise ValueError("SOP07 release root metadata must be objects")
    if (
        request_document.get("version") != SOP07_RISK_RELEASE_VERSION
        or manifest.get("version") != SOP07_RISK_RELEASE_VERSION
        or complete.get("version") != SOP07_RISK_RELEASE_VERSION
    ):
        raise ValueError("SOP07 risk release version mismatch")
    request_payload = request_document.get("request")
    if not isinstance(request_payload, dict):
        raise ValueError("SOP07 release request payload is invalid")
    request_identity = hashlib.sha256(
        b"sop07_single_risk_release_request_v1\0"
        + _canonical_json(request_payload)
    ).hexdigest()
    if (
        request_document.get("request_identity") != request_identity
        or manifest.get("request_identity") != request_identity
    ):
        raise ValueError("SOP07 release request identity mismatch")
    descriptors = manifest.get("shards")
    if not isinstance(descriptors, list) or not descriptors:
        raise ValueError("SOP07 release shard descriptors are invalid")
    expected_names = {
        f"shard-{index:05d}" for index in range(len(descriptors))
    }
    shard_root = root / _SHARDS
    if {path.name for path in shard_root.iterdir()} != expected_names:
        raise ValueError("SOP07 release child shard set mismatch")
    grid_config = manifest.get("resolved_base_config")
    if not isinstance(grid_config, Mapping):
        raise ValueError("SOP07 resolved base config is unavailable")
    grid = build_grid_spec(dict(grid_config))
    sample_ids: list[str] = []
    label_population = {
        "hidden_target_count": 0,
        "current_visible_target_exclusion_count": 0,
        "empty_target_count": 0,
        "sample_count": 0,
    }
    for index, descriptor in enumerate(descriptors):
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("shard_index") != index
            or descriptor.get("relative_root") != f"shards/shard-{index:05d}"
        ):
            raise ValueError("SOP07 release shard descriptor ordering differs")
        loaded = load_risk_shard(root / descriptor["relative_root"], grid=grid)
        ids = tuple(sample.sample_id for sample in loaded.samples)
        pairs = tuple(
            (
                sample.sample_id,
                str(sample.metadata.get("provenance", {}).get("mother_id")),
            )
            for sample in loaded.samples
        )
        if (
            loaded.manifest_digest != descriptor.get("risk_manifest_digest")
            or loaded.semantic_digest != descriptor.get("risk_semantic_digest")
            or len(ids) != descriptor.get("sample_count")
            or ids[0] != descriptor.get("first_sample_id")
            or ids[-1] != descriptor.get("last_sample_id")
            or _join_digest(pairs)
            != descriptor.get("sample_mother_join_digest")
        ):
            raise ValueError("SOP07 release child shard identity differs")
        for sample in loaded.samples:
            provenance = sample.metadata.get("provenance")
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("sop06_history_release_manifest_digest")
                != manifest.get("sop06_release_manifest_digest")
                or provenance.get("sop06_history_shard_index") != index
                or provenance.get("sop06_history_shard_semantic_digest")
                != descriptor.get("sop06_history_shard_semantic_digest")
            ):
                raise ValueError("SOP07 release child SOP06 binding differs")
        _add_population(label_population, _label_population(loaded.samples))
        sample_ids.extend(ids)
    ordered_ids = tuple(sample_ids)
    if (
        ordered_ids != tuple(sorted(ordered_ids))
        or len(set(ordered_ids)) != len(ordered_ids)
    ):
        raise ValueError("SOP07 release sample IDs are not globally unique and sorted")
    manifest_digest = hashlib.sha256(
        b"sop07_single_risk_release_manifest_v1\0" + _canonical_json(manifest)
    ).hexdigest()
    if (
        manifest.get("sample_count") != len(ordered_ids)
        or manifest.get("shard_count") != len(descriptors)
        or manifest.get("label_population") != label_population
        or complete
        != {
            "version": SOP07_RISK_RELEASE_VERSION,
            "request_identity": request_identity,
            "sample_count": len(ordered_ids),
            "shard_count": len(descriptors),
            "manifest_digest": manifest_digest,
        }
    ):
        raise ValueError("SOP07 release completion identity mismatch")
    return LoadedSop07RiskRelease(
        root=root,
        request_identity=request_identity,
        split=str(manifest["split"]),
        sample_count=len(ordered_ids),
        shard_count=len(descriptors),
        sop06_release_manifest_digest=str(
            manifest["sop06_release_manifest_digest"]
        ),
        manifest_digest=manifest_digest,
        request=request_payload,
        manifest=manifest,
    )


def load_sop07_risk_release(
    input_dir: str | Path,
) -> LoadedSop07RiskRelease:
    """Strictly validate a complete SOP07 release and all child risk arrays."""

    return _load_sop07_risk_release(input_dir)


def _result(
    loaded: LoadedSop07RiskRelease,
    *,
    reused_shard_count: int,
) -> Sop07RiskReleaseResult:
    return Sop07RiskReleaseResult(
        output_dir=loaded.root,
        split=loaded.split,
        sample_count=loaded.sample_count,
        shard_count=loaded.shard_count,
        reused_shard_count=reused_shard_count,
        manifest_digest=loaded.manifest_digest,
    )


def publish_sop07_risk_release(
    request: Sop07RiskReleaseRequest,
    *,
    build_one: BuildOne = _default_build_one,
    source_loader: SourceLoader = _default_source_loader,
    progress_callback: ProgressCallback | None = None,
) -> Sop07RiskReleaseResult:
    """Publish one risk shard per SOP06 shard, reusing exact completed boundaries."""

    upstream = load_sop06_history_release_checkpoint(request.sop06_release_root)
    complete_mother_lineage = _complete_mother_lineage_from_request(
        request,
        upstream,
    )
    if source_loader is _default_source_loader:
        source = source_loader(
            upstream,
            complete_mother_lineage=complete_mother_lineage,
        )
    else:
        if complete_mother_lineage is not None:
            raise ValueError(
                "custom SOP07 source_loader cannot bypass complete-mother lineage"
            )
        source = source_loader(upstream)
    for attribute in ("base_config", "accepted", "prepare_boundary"):
        if not hasattr(source, attribute):
            raise TypeError(
                f"source_loader result is missing required attribute: {attribute}"
            )
    risk_config = source.base_config.get("risk_gt")
    if not isinstance(risk_config, Mapping):
        raise ValueError("SOP05 base config has no risk_gt section")
    base_digest = config_digest(source.base_config)
    risk_digest = config_digest(risk_config)
    payload = _request_payload(
        request,
        upstream,
        base_config_digest=base_digest,
        risk_config_digest=risk_digest,
    )
    request_document = _request_document(payload)
    request_identity = str(request_document["request_identity"])
    in_progress = (
        request.output_dir.parent / f".{request.output_dir.name}.inprogress"
    )
    if request.output_dir.exists() or request.output_dir.is_symlink():
        if in_progress.exists() or in_progress.is_symlink():
            raise ValueError("complete and in-progress SOP07 releases both exist")
        loaded = load_sop07_risk_release(request.output_dir)
        if loaded.request_identity != request_identity:
            raise ValueError("existing SOP07 release differs from request")
        return _result(loaded, reused_shard_count=loaded.shard_count)

    if in_progress.exists() or in_progress.is_symlink():
        actual_files = (
            {path.name for path in in_progress.iterdir()}
            if in_progress.is_dir()
            else set()
        )
        if actual_files == _ROOT_FILES:
            staged = load_sop07_risk_release(in_progress)
            if staged.request_identity != request_identity:
                raise ValueError("completed in-progress SOP07 request differs")
            atomic_rename_noreplace(in_progress, request.output_dir)
            published = replace(staged, root=request.output_dir)
            return _result(
                published,
                reused_shard_count=published.shard_count,
            )
        if (
            not in_progress.is_dir()
            or actual_files != {_REQUEST, _SHARDS}
            or _read_json(in_progress / _REQUEST) != request_document
        ):
            raise ValueError("in-progress SOP07 release differs from request")
    else:
        in_progress.mkdir(parents=True)
        (in_progress / _SHARDS).mkdir()
        (in_progress / _REQUEST).write_bytes(_json_file(request_document))

    accepted_by_sample = {
        accepted.scenario_id: accepted for accepted in source.accepted
    }
    if (
        len(accepted_by_sample) != len(source.accepted)
        or len(accepted_by_sample) != upstream.sample_count
    ):
        raise ValueError("SOP05 accepted oracle identities differ from SOP06")
    descriptors = upstream.manifest.get("shards")
    if not isinstance(descriptors, list) or len(descriptors) != upstream.shard_count:
        raise ValueError("SOP06 release shard manifest is invalid")
    expected_child_names = {
        f"shard-{index:05d}" for index in range(upstream.shard_count)
    }
    shard_root = in_progress / _SHARDS
    if {path.name for path in shard_root.iterdir()} - expected_child_names:
        raise ValueError("in-progress SOP07 release contains unexpected shard")

    grid = build_grid_spec(dict(source.base_config))
    output_descriptors: list[dict[str, object]] = []
    reused = 0
    observed_ids: list[str] = []
    label_population = {
        "hidden_target_count": 0,
        "current_visible_target_exclusion_count": 0,
        "empty_target_count": 0,
        "sample_count": 0,
    }
    for index, upstream_descriptor in enumerate(descriptors):
        if (
            not isinstance(upstream_descriptor, Mapping)
            or upstream_descriptor.get("shard_index") != index
            or upstream_descriptor.get("relative_root")
            != f"shards/shard-{index:05d}"
        ):
            raise ValueError("SOP06 shard descriptor ordering differs")
        sop06_path = upstream.root / str(upstream_descriptor["relative_root"])
        checkpoint = load_sop06_history_shard_checkpoint(sop06_path)
        binding = Sop07ShardBinding(
            sop06_release_manifest_digest=upstream.manifest_digest,
            sop06_shard_index=index,
            sop06_shard_semantic_digest=checkpoint.semantic_digest,
            source_family=upstream.source_family,
            source_mode=upstream.source_mode,
        )
        output_path = shard_root / f"shard-{index:05d}"
        was_reused = output_path.exists()
        if was_reused:
            loaded = load_risk_shard(output_path, grid=grid)
            _validate_risk_binding(loaded, checkpoint, binding)
            reused += 1
        else:
            observation_shard = load_sop06_history_shard(sop06_path)
            if (
                observation_shard.semantic_digest != checkpoint.semantic_digest
                or tuple(
                    sample.sample_id for sample in observation_shard.samples
                )
                != checkpoint.sample_ids
                or tuple(
                    sample.mother_id for sample in observation_shard.samples
                )
                != checkpoint.mother_ids
            ):
                raise ValueError("strict SOP06 shard differs from its checkpoint")
            boundary: list[Sop06AcceptedFinalRecord] = []
            for observation in observation_shard.samples:
                accepted = accepted_by_sample.get(observation.sample_id)
                if accepted is None:
                    raise ValueError("SOP06 sample_id has no SOP05 oracle")
                _validate_observation_join(observation, accepted)
                boundary.append(accepted)
            prepared = source.prepare_boundary(tuple(boundary))
            built: list[RiskSample] = []
            for accepted, observation in zip(
                boundary,
                observation_shard.samples,
                strict=True,
            ):
                try:
                    built.append(
                        build_one(prepared, accepted, observation, binding)
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "SOP07 sample build failed for "
                        f"sample_id={observation.sample_id!r}, "
                        f"mother_id={observation.mother_id!r}: {exc}"
                    ) from exc
            samples = tuple(built)
            if tuple(sample.sample_id for sample in samples) != checkpoint.sample_ids:
                raise ValueError("built SOP07 samples differ from SOP06 boundary")
            write_risk_shard(
                samples,
                output_path,
                grid=grid,
                shard_index=index,
                expected_sample_count=len(checkpoint.sample_ids),
            )
            loaded = load_risk_shard(output_path, grid=grid)
            _validate_risk_binding(loaded, checkpoint, binding)
        output_descriptors.append(_descriptor(loaded, checkpoint))
        _add_population(label_population, _label_population(loaded.samples))
        observed_ids.extend(checkpoint.sample_ids)
        if progress_callback is not None:
            progress_callback(index + 1, upstream.shard_count, was_reused)

    if tuple(observed_ids) != tuple(sorted(accepted_by_sample)):
        raise ValueError("SOP07 processed IDs differ from the complete SOP05/SOP06 join")
    manifest = {
        "version": SOP07_RISK_RELEASE_VERSION,
        "request_identity": request_identity,
        "source_family": upstream.source_family,
        "source_mode": upstream.source_mode,
        "split": upstream.split,
        "sample_count": len(observed_ids),
        "shard_count": len(output_descriptors),
        "sop06_release_request_identity": upstream.request_identity,
        "sop06_release_manifest_digest": upstream.manifest_digest,
        "sop06_scenario_ids_digest": upstream.scenario_ids_digest,
        "base_config_digest": base_digest,
        "risk_config_digest": risk_digest,
        "risk_shard_layout_version": RISK_SHARD_LAYOUT_VERSION,
        "label_population": label_population,
        "resolved_base_config": dict(source.base_config),
        "shards": output_descriptors,
    }
    manifest_digest = hashlib.sha256(
        b"sop07_single_risk_release_manifest_v1\0" + _canonical_json(manifest)
    ).hexdigest()
    (in_progress / _MANIFEST).write_bytes(_json_file(manifest))
    (in_progress / _CHECKSUMS).write_bytes(
        _json_file(
            {
                _REQUEST: _sha256_file(in_progress / _REQUEST),
                _MANIFEST: _sha256_file(in_progress / _MANIFEST),
            }
        )
    )
    (in_progress / _COMPLETE).write_bytes(
        _json_file(
            {
                "version": SOP07_RISK_RELEASE_VERSION,
                "request_identity": request_identity,
                "sample_count": len(observed_ids),
                "shard_count": len(output_descriptors),
                "manifest_digest": manifest_digest,
            }
        )
    )
    loaded = load_sop07_risk_release(in_progress)
    if loaded.manifest_digest != manifest_digest:
        raise ValueError("staged SOP07 release manifest digest differs")
    atomic_rename_noreplace(in_progress, request.output_dir)
    published = load_sop07_risk_release(request.output_dir)
    if published.manifest_digest != manifest_digest:
        raise ValueError("published SOP07 release differs from staging")
    return _result(published, reused_shard_count=reused)


__all__ = (
    "LoadedSop07RiskRelease",
    "SOP07_RISK_RELEASE_VERSION",
    "Sop07CompleteMotherLineage",
    "Sop07RiskReleaseRequest",
    "Sop07RiskReleaseResult",
    "Sop07ShardBinding",
    "load_sop07_risk_release",
    "publish_sop07_risk_release",
)
