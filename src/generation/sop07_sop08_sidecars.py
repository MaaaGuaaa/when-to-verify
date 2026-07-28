"""Publish SOP08 label sidecars for an immutable single-scene SOP07 release."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from src.contracts import build_grid_spec
from src.datasets.risk_dataset import _robot_footprint
from src.datasets.shard_writer import load_risk_shard
from src.datasets.sidecar_writer import (
    load_risk_sidecar_pair_completion_marker,
    load_risk_sidecar_shard,
    risk_sidecar_pair_completion_marker_path,
    write_risk_sidecar_pair_completion_marker,
    write_risk_sidecar_shard,
)
from src.datasets.sop06_history_bev import load_sop06_history_shard
from src.generation.observation_renderer import RenderedObservation
from src.generation.risk_sidecars import build_risk_label_sidecar
from src.generation.sop06_pipeline import build_sop06_single_risk_input
from src.generation.sop07_risk_release import (
    Sop07ShardBinding,
    _complete_mother_lineage_from_payload,
    _default_source_loader,
    _repository_path,
    _validate_observation_join,
    load_sop07_risk_release,
)
from src.generation.sop06_history_release import load_sop06_history_release_checkpoint


SOP07_SOP08_SIDECAR_RELEASE_VERSION = "sop07_single_scene_sop08_sidecars_v1"


@dataclass(frozen=True)
class Sop07Sop08SidecarRequest:
    sop07_release_root: Path
    sidecar_root: Path
    shard_modulus: int = 1
    shard_remainder: int = 0


@dataclass(frozen=True)
class Sop07Sop08SidecarResult:
    sidecar_root: Path
    sample_count: int
    shard_count: int
    reused_shard_count: int


ProgressCallback = Callable[[int, int, bool], None]


def _build_sidecar_from_oracle(
    *,
    source: object,
    accepted: object,
    observation: object,
    binding: Sop07ShardBinding,
) -> object:
    _validate_observation_join(observation, accepted)
    resolved = source.resolve(accepted)
    publication = resolved.publication
    if (
        publication.sample_id != observation.sample_id
        or publication.mother_id != observation.mother_id
    ):
        raise ValueError("resolved SOP05 oracle differs from the SOP06 observation")
    if observation.renderer_metadata.get("base_state_id") != (
        publication.renderer_input.base_state.state_id
    ):
        raise ValueError("SOP06 observation base_state_id differs from SOP05 oracle")
    risk_input = build_sop06_single_risk_input(publication)
    risk_input = replace(
        risk_input,
        provenance={
            **dict(risk_input.provenance),
            "sop06_history_release_manifest_digest": binding.sop06_release_manifest_digest,
            "sop06_history_shard_index": binding.sop06_shard_index,
            "sop06_history_shard_semantic_digest": binding.sop06_shard_semantic_digest,
            "sop06_source_family": binding.source_family,
            "sop06_source_mode": binding.source_mode,
        },
    )
    grid = build_grid_spec(dict(source.base_config))
    sidecar = build_risk_label_sidecar(
        sample_id=observation.sample_id,
        trajectory=risk_input.trajectory,
        world=risk_input.oracle_world,
        hidden_object_ids=risk_input.hidden_object_ids,
        robot_footprint=_robot_footprint(source.base_config),
        grid=grid,
        future_dt_s=source.base_config["bev"]["future_dt_s"],
    )
    return sidecar


def _validate_existing_sidecar(
    *,
    sidecar_root: Path,
    risk_root: Path,
    split: str,
    shard_index: int,
    risk_shard: object,
    grid: object,
) -> None:
    sample_ids = tuple(sample.sample_id for sample in risk_shard.samples)
    sidecar = load_risk_sidecar_shard(
        sidecar_root,
        grid=grid,
        expected_sample_ids=sample_ids,
        expected_source_risk_shard_semantic_digest=risk_shard.semantic_digest,
    )
    load_risk_sidecar_pair_completion_marker(
        risk_sidecar_pair_completion_marker_path(sidecar_root),
        expected_risk_root=risk_root,
        expected_sidecar_root=sidecar_root,
        expected_split=split,
        expected_shard_index=shard_index,
        expected_sample_ids=sample_ids,
        expected_risk_shard_semantic_digest=risk_shard.semantic_digest,
        expected_sidecar_shard_semantic_digest=sidecar.semantic_digest,
    )


def publish_sop07_sop08_sidecars(
    request: Sop07Sop08SidecarRequest,
    *,
    progress_callback: ProgressCallback | None = None,
) -> Sop07Sop08SidecarResult:
    """Rebuild oracle-only labels and bind each one to an immutable SOP07 shard."""

    if (
        type(request.shard_modulus) is not int
        or request.shard_modulus < 1
        or type(request.shard_remainder) is not int
        or not 0 <= request.shard_remainder < request.shard_modulus
    ):
        raise ValueError("shard partition must satisfy 0 <= remainder < modulus")
    release = load_sop07_risk_release(request.sop07_release_root)
    if request.sidecar_root.exists() and not request.sidecar_root.is_dir():
        raise ValueError("sidecar_root must be a directory when it already exists")
    request.sidecar_root.mkdir(parents=True, exist_ok=True)
    upstream = load_sop06_history_release_checkpoint(
        _repository_path(release.request["sop06_release_root"], name="sop06_release_root")
    )
    source = _default_source_loader(
        upstream,
        complete_mother_lineage=_complete_mother_lineage_from_payload(
            release.request
        ),
    )
    grid = build_grid_spec(dict(source.base_config))
    accepted_by_id = {item.scenario_id: item for item in source.accepted}
    if len(accepted_by_id) != release.sample_count:
        raise ValueError("SOP05 accepted oracle identities differ from SOP07 release")
    descriptors = release.manifest["shards"]
    reused = 0
    for index, descriptor in enumerate(descriptors):
        if index % request.shard_modulus != request.shard_remainder:
            continue
        risk_root = release.root / descriptor["relative_root"]
        risk_shard = load_risk_shard(risk_root, grid=grid)
        sidecar_root = request.sidecar_root / f"shard-{index:05d}"
        marker = risk_sidecar_pair_completion_marker_path(sidecar_root)
        if sidecar_root.exists() or marker.exists():
            if not (sidecar_root.exists() and marker.exists()):
                raise ValueError("incomplete existing risk/sidecar pair")
            _validate_existing_sidecar(
                sidecar_root=sidecar_root,
                risk_root=risk_root,
                split=release.split,
                shard_index=index,
                risk_shard=risk_shard,
                grid=grid,
            )
            reused += 1
            if progress_callback is not None:
                progress_callback(index + 1, release.shard_count, True)
            continue
        observation_shard = load_sop06_history_shard(
            upstream.root / f"shards/shard-{index:05d}"
        )
        if tuple(sample.sample_id for sample in observation_shard.samples) != tuple(
            sample.sample_id for sample in risk_shard.samples
        ):
            raise ValueError("SOP06/SOP07 shard sample boundary differs")
        boundary = tuple(accepted_by_id[observation.sample_id] for observation in observation_shard.samples)
        prepared = source.prepare_boundary(boundary)
        binding = Sop07ShardBinding(
            sop06_release_manifest_digest=upstream.manifest_digest,
            sop06_shard_index=index,
            sop06_shard_semantic_digest=observation_shard.semantic_digest,
            source_family=upstream.source_family,
            source_mode=upstream.source_mode,
        )
        sidecars = []
        for expected, accepted, observation in zip(
            risk_shard.samples, boundary, observation_shard.samples, strict=True
        ):
            sidecar = _build_sidecar_from_oracle(
                source=prepared,
                accepted=accepted,
                observation=observation,
                binding=binding,
            )
            if sidecar.sample_id != expected.sample_id:
                raise ValueError("rebuilt sidecar sample_id differs from SOP07 shard")
            sidecars.append(sidecar)
        write_risk_sidecar_shard(
            tuple(sidecars),
            sidecar_root,
            grid=grid,
            split=release.split,
            shard_index=index,
            source_risk_shard_semantic_digest=risk_shard.semantic_digest,
        )
        written = load_risk_sidecar_shard(
            sidecar_root,
            grid=grid,
            expected_sample_ids=tuple(sample.sample_id for sample in risk_shard.samples),
            expected_source_risk_shard_semantic_digest=risk_shard.semantic_digest,
        )
        write_risk_sidecar_pair_completion_marker(
            marker,
            risk_root=risk_root,
            sidecar_root=sidecar_root,
            split=release.split,
            shard_index=index,
            sample_ids=tuple(sample.sample_id for sample in risk_shard.samples),
            risk_shard_semantic_digest=risk_shard.semantic_digest,
            sidecar_shard_semantic_digest=written.semantic_digest,
        )
        _validate_existing_sidecar(
            sidecar_root=sidecar_root,
            risk_root=risk_root,
            split=release.split,
            shard_index=index,
            risk_shard=risk_shard,
            grid=grid,
        )
        if progress_callback is not None:
            progress_callback(index + 1, release.shard_count, False)
    return Sop07Sop08SidecarResult(
        sidecar_root=request.sidecar_root,
        sample_count=release.sample_count,
        shard_count=release.shard_count,
        reused_shard_count=reused,
    )


__all__ = (
    "SOP07_SOP08_SIDECAR_RELEASE_VERSION",
    "Sop07Sop08SidecarRequest",
    "Sop07Sop08SidecarResult",
    "publish_sop07_sop08_sidecars",
)
