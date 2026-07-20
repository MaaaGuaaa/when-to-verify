"""Trust-anchored, read-only SOP03/04/05/07 inputs for verification smoke runs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping

from src.contracts import BaseState, GridSpec, LocalTrajectory, OracleContext, SCHEMA_VERSION
from src.datasets.snippet_library import MotionSnippet
from src.generation.event_sampler import GeneratedEvent
from src.generation.event_target_motion_shard import EventTargetMotionRecord
from src.generation.sop05_input_adapter import (
    Sop03SplitInputs,
    Sop04TrajectoryBank,
    load_sop03_split_inputs,
    load_sop04_trajectory_bank,
)
from src.generation.sop05_output_loader import load_complete_sop05_events
from src.utils.seeding import stable_digest


VERIFICATION_SOURCE_INDEX_VERSION = "verification_source_index_v2"
_SPLITS = ("train", "calibration", "val", "test")
_SCIENTIFIC_STATUS = {
    split: f"{split}_smoke_only" for split in _SPLITS
}
_BATCH_CONTRACTS = {
    "train": (
        "sop05_batch_index_handoff_v1",
        "sop05_train_batch_complete_index",
    ),
    "heldout": (
        "sop05_heldout_batch_complete_handoff_v1",
        "sop05_heldout_batch_complete_index",
    ),
}
_COLLECTION_CONTRACTS = {
    "train": (
        "sop07_collection_complete_handoff_v1",
        "sop07_train_collection_complete_handoff",
    ),
    "heldout": (
        "sop07_heldout_collection_complete_handoff_v1",
        "sop07_heldout_collection_complete_handoff",
    ),
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class Sop05ShardSource:
    shard_index: int
    relative_root: str
    root: Path
    event_count: int
    publication_semantic_digest: str
    run_id: str
    trajectory_id: str


@dataclass(frozen=True)
class VerificationSourceIndex:
    version: str
    schema_version: str
    split: str
    scientific_status: str
    global_cross_split_leakage: str
    sop05_batch_root: Path
    sop05_batch_digest: str
    sop05_code_commit: str
    sop07_collection_root: Path
    sop07_collection_digest: str
    sop07_code_commit: str
    event_count: int
    sop07_sample_count: int
    temporal_safe_count: int
    sop03_input_lock: Mapping[str, object]
    sop04_input_lock: Mapping[str, object]
    shards: tuple[Sop05ShardSource, ...]


@dataclass(frozen=True)
class VerificationSourceEvent:
    event: GeneratedEvent
    base_state: BaseState
    oracle_context: OracleContext
    nominal_trajectory: LocalTrajectory
    source_snippet: MotionSnippet
    shard: Sop05ShardSource


@dataclass(frozen=True)
class VerificationSourceBundle:
    index: VerificationSourceIndex
    sop03: Sop03SplitInputs
    sop04: Sop04TrajectoryBank
    events: tuple[VerificationSourceEvent, ...]
    loaded_sop05_publication_digests: tuple[str, ...]


def _strict_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a real file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _digest(value: object, *, name: str, anchor: bool = False) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        suffix = " external trust anchor" if anchor else ""
        raise ValueError(f"{name}{suffix} must be a 64-character lowercase digest")
    return value


def _commit(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 40-character lowercase commit")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _safe_relative_root(value: object, *, name: str) -> str:
    text = _string(value, name=name)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{name} must be a safe relative path")
    if len(path.parts) != 1 or not re.fullmatch(r"shard-[0-9]{5}", text):
        raise ValueError(f"{name} must be a canonical relative shard root")
    return text


def _safe_launch_manifest(value: object) -> str:
    text = _string(value, name="SOP07 launch manifest relative_path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or text != "batch_launch_manifest.json"
    ):
        raise ValueError("SOP07 launch manifest must use its canonical relative path")
    return text


def _sha256_file(path: Path, *, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a real file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot read {label}") from exc
    return digest.hexdigest()


def _source_batch_binding(
    collection: Mapping[str, object],
    *,
    collection_path: Path,
    split: str,
    collection_commit: str,
) -> Mapping[str, object]:
    launch_evidence = _mapping(
        collection.get("launch_evidence"), name="SOP07 launch evidence"
    )
    if split == "train":
        return _mapping(
            launch_evidence.get("source_sop05_batch"),
            name="SOP07 source SOP05 batch",
        )

    relative = _safe_launch_manifest(launch_evidence.get("relative_path"))
    expected_digest = _digest(
        launch_evidence.get("sha256"), name="SOP07 launch manifest digest"
    )
    launch_path = collection_path.parent / relative
    observed_digest = _sha256_file(launch_path, label="SOP07 launch manifest")
    if observed_digest != expected_digest:
        raise ValueError("SOP07 launch manifest digest mismatch")
    launch = _strict_json(launch_path, label="SOP07 launch manifest")
    if launch.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("SOP07 launch manifest schema mismatch")
    if launch.get("split") != split:
        raise ValueError("SOP07 launch manifest split mismatch")
    if launch.get("code_commit") != collection_commit:
        raise ValueError("SOP07 launch manifest code commit mismatch")
    if launch.get("artifact_role") != f"sop07_{split}_batch_launch_manifest":
        raise ValueError("SOP07 launch manifest artifact role mismatch")
    return _mapping(
        launch.get("source_sop05_batch"), name="SOP07 source SOP05 batch"
    )


def _validate_sop03_lock(value: object) -> Mapping[str, object]:
    lock = _mapping(value, name="SOP03 input lock")
    required = {
        "audit_sha256",
        "checksum_manifest_sha256",
        "code_commit",
        "completion_policy",
    }
    if not required.issubset(lock):
        raise ValueError("SOP03 input lock is incomplete")
    _digest(lock["audit_sha256"], name="SOP03 audit digest")
    _digest(lock["checksum_manifest_sha256"], name="SOP03 checksum digest")
    _commit(lock["code_commit"], name="SOP03 code commit")
    if lock["completion_policy"] != "sop03_complete_marker_v1":
        raise ValueError("unsupported SOP03 completion policy")
    return MappingProxyType(dict(lock))


def _validate_sop04_lock(value: object) -> Mapping[str, object]:
    lock = _mapping(value, name="SOP04 input lock")
    required = {
        "audit_sha256",
        "bank_semantic_digest_sha256",
        "checksum_manifest_sha256",
        "code_commit",
        "completion_policy",
        "dt_s",
        "external_handoff_digest_sha256",
        "first_pose_time_s",
        "last_pose_time_s",
        "pose_time_layout_version",
        "pose_time_offsets_sha256",
        "trajectory_bank_version",
        "trajectory_steps",
    }
    if not required.issubset(lock):
        raise ValueError("SOP04 input lock is incomplete")
    for key in (
        "audit_sha256",
        "bank_semantic_digest_sha256",
        "checksum_manifest_sha256",
        "external_handoff_digest_sha256",
        "pose_time_offsets_sha256",
    ):
        _digest(lock[key], name=f"SOP04 {key}")
    _commit(lock["code_commit"], name="SOP04 code commit")
    if (
        lock["completion_policy"] != "sop04_audited_bank_v2"
        or lock["trajectory_bank_version"] != "sop04_audited_bank_v2"
        or lock["pose_time_layout_version"]
        != "future_endpoints_dt_to_horizon_v1"
        or lock["trajectory_steps"] != 15
        or lock["dt_s"] != 0.2
        or lock["first_pose_time_s"] != 0.2
        or lock["last_pose_time_s"] != 3.0
    ):
        raise ValueError("SOP04 input lock time/layout contract mismatch")
    return MappingProxyType(dict(lock))


def load_verification_source_index(
    sop05_batch_handoff: str | Path,
    sop07_collection_handoff: str | Path,
    *,
    expected_sop05_batch_digest: str,
    expected_sop07_collection_digest: str,
    expected_split: str = "train",
) -> VerificationSourceIndex:
    """Bind two retained handoffs to caller-supplied external trust anchors."""

    expected_batch = _digest(
        expected_sop05_batch_digest,
        name="SOP05 batch",
        anchor=True,
    )
    expected_collection = _digest(
        expected_sop07_collection_digest,
        name="SOP07 collection",
        anchor=True,
    )
    split = _string(expected_split, name="expected split")
    if split not in _SPLITS:
        raise ValueError(f"expected split must be one of {_SPLITS}")
    contract_kind = "train" if split == "train" else "heldout"
    batch_version, batch_role = _BATCH_CONTRACTS[contract_kind]
    collection_version, collection_role = _COLLECTION_CONTRACTS[contract_kind]
    batch_path = Path(sop05_batch_handoff)
    collection_path = Path(sop07_collection_handoff)
    batch = _strict_json(batch_path, label="SOP05 batch handoff")
    collection = _strict_json(collection_path, label="SOP07 collection handoff")

    if batch.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("SOP05 batch schema version mismatch")
    if batch.get("handoff_version") != batch_version:
        raise ValueError("unsupported SOP05 batch handoff version")
    if batch.get("artifact_role") != batch_role:
        raise ValueError("SOP05 batch artifact role mismatch")
    if batch.get("batch_state") != "complete":
        raise ValueError("SOP05 batch is not complete")
    if batch.get("split") != split:
        raise ValueError(f"SOP05 batch split differs from expected split {split}")
    observed_batch = _digest(
        batch.get("batch_semantic_digest_sha256"), name="SOP05 batch digest"
    )
    if observed_batch != expected_batch:
        raise ValueError("SOP05 batch digest differs from external trust anchor")
    batch_commit = _commit(batch.get("code_commit"), name="SOP05 code commit")
    common = _mapping(batch.get("common_contracts"), name="SOP05 common contracts")
    input_lock = _mapping(common.get("input_lock"), name="SOP05 input lock")
    if input_lock.get("version") != "sop05_input_lock_v2":
        raise ValueError("unsupported SOP05 input lock version")
    if input_lock.get("split") != split:
        raise ValueError(f"SOP05 input lock split differs from expected split {split}")
    if "sop03" not in input_lock or "sop04" not in input_lock:
        raise ValueError("SOP05 input lock must contain SOP03 and SOP04")
    sop03_lock = _validate_sop03_lock(input_lock["sop03"])
    sop04_lock = _validate_sop04_lock(input_lock["sop04"])
    counts = _mapping(batch.get("counts"), name="SOP05 counts")
    event_count = _integer(counts.get("events"), name="SOP05 event count", minimum=1)
    if split == "train" and counts.get("planned_events") != event_count:
        raise ValueError("SOP05 planned/event count mismatch")
    if split != "train" and "planned_events" in counts:
        if counts.get("planned_events") != event_count:
            raise ValueError("SOP05 held-out planned/event count mismatch")
    shard_count = _integer(counts.get("shards"), name="SOP05 shard count", minimum=1)
    raw_shards = batch.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != shard_count:
        raise ValueError("SOP05 shard index count mismatch")
    shards: list[Sop05ShardSource] = []
    for position, raw in enumerate(raw_shards):
        row = _mapping(raw, name=f"SOP05 shard {position}")
        index = _integer(row.get("shard_index"), name="SOP05 shard_index")
        if index != position:
            raise ValueError("SOP05 shard indices must be contiguous and ordered")
        relative = _safe_relative_root(
            row.get("relative_root"), name="SOP05 relative_root"
        )
        shards.append(
            Sop05ShardSource(
                shard_index=index,
                relative_root=relative,
                root=batch_path.parent / relative,
                event_count=_integer(
                    row.get("event_count"), name="SOP05 shard event_count", minimum=1
                ),
                publication_semantic_digest=_digest(
                    row.get("publication_semantic_digest"),
                    name="SOP05 publication digest",
                ),
                run_id=_string(row.get("run_id"), name="SOP05 run_id"),
                trajectory_id=_string(
                    row.get("trajectory_id"), name="SOP05 trajectory_id"
                ),
            )
        )
    if sum(item.event_count for item in shards) != event_count:
        raise ValueError("SOP05 shard events do not conserve the batch event count")

    if collection.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("SOP07 collection schema version mismatch")
    if collection.get("handoff_version") != collection_version:
        raise ValueError("unsupported SOP07 collection handoff version")
    if collection.get("artifact_role") != collection_role:
        raise ValueError("SOP07 collection artifact role mismatch")
    if collection.get("collection_state") != "complete":
        raise ValueError("SOP07 collection is not complete")
    if collection.get("split") != split:
        raise ValueError(f"SOP07 handoff split differs from expected split {split}")
    observed_collection = _digest(
        collection.get("collection_semantic_digest_sha256"),
        name="SOP07 collection digest",
    )
    if observed_collection != expected_collection:
        raise ValueError("SOP07 collection digest differs from external trust anchor")
    collection_commit = _commit(
        collection.get("code_commit"), name="SOP07 code commit"
    )
    downstream = _mapping(
        collection.get("downstream_contract"), name="SOP07 downstream contract"
    )
    if downstream.get("generation_evidence_join") != "PROVEN":
        raise ValueError("SOP07 generation evidence join is not proven")
    if downstream.get("global_sample_id_uniqueness") != "PROVEN":
        raise ValueError("SOP07 global sample ID uniqueness is not proven")
    cross_split_status = downstream.get("global_cross_split_leakage")
    if cross_split_status != "NOT_PROVEN":
        raise ValueError("per-split SOP07 cross-split status must remain NOT_PROVEN")
    source_batch = _source_batch_binding(
        collection,
        collection_path=collection_path,
        split=split,
        collection_commit=collection_commit,
    )
    if (
        source_batch.get("batch_semantic_digest_sha256") != observed_batch
        or source_batch.get("event_count") != event_count
        or source_batch.get("shard_count") != shard_count
    ):
        raise ValueError("SOP07 source SOP05 batch binding mismatch")
    if collection.get("shard_count") != shard_count:
        raise ValueError("SOP07/SOP05 shard count mismatch")
    sop07_sample_count = _integer(
        collection.get("sample_count"), name="SOP07 sample count", minimum=1
    )
    event_types = _mapping(
        collection.get("event_type_counts"), name="SOP07 event type counts"
    )
    temporal_safe_count = _integer(
        event_types.get("temporal_safe"), name="SOP07 temporal_safe count"
    )
    risk_shards = collection.get("shards")
    if not isinstance(risk_shards, list) or len(risk_shards) != shard_count:
        raise ValueError("SOP07 shard index count mismatch")
    for position, raw in enumerate(risk_shards):
        row = _mapping(raw, name=f"SOP07 shard {position}")
        if row.get("shard_index") != position:
            raise ValueError("SOP07 shard indices must be contiguous and ordered")
        _safe_relative_root(row.get("relative_root"), name="SOP07 relative_root")

    return VerificationSourceIndex(
        version=VERIFICATION_SOURCE_INDEX_VERSION,
        schema_version=SCHEMA_VERSION,
        split=split,
        scientific_status=_SCIENTIFIC_STATUS[split],
        global_cross_split_leakage=str(cross_split_status),
        sop05_batch_root=batch_path.parent,
        sop05_batch_digest=observed_batch,
        sop05_code_commit=batch_commit,
        sop07_collection_root=collection_path.parent,
        sop07_collection_digest=observed_collection,
        sop07_code_commit=collection_commit,
        event_count=event_count,
        sop07_sample_count=sop07_sample_count,
        temporal_safe_count=temporal_safe_count,
        sop03_input_lock=sop03_lock,
        sop04_input_lock=sop04_lock,
        shards=tuple(shards),
    )


def select_source_shards(
    index: VerificationSourceIndex,
    *,
    count: int,
    seed: int,
) -> tuple[Sop05ShardSource, ...]:
    """Select a stable shard subset without depending on process hash order."""

    if not isinstance(index, VerificationSourceIndex):
        raise TypeError("index must be a VerificationSourceIndex")
    requested = _integer(count, name="count", minimum=1)
    if requested > len(index.shards):
        raise ValueError("count exceeds available SOP05 shards")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    ranked = sorted(
        index.shards,
        key=lambda shard: (
            stable_digest(
                VERIFICATION_SOURCE_INDEX_VERSION,
                index.sop05_batch_digest,
                seed,
                shard.shard_index,
                size=16,
            ),
            shard.shard_index,
        ),
    )
    return tuple(ranked[:requested])


def _validate_producer_evidence(
    *,
    name: str,
    observed,
    expected: Mapping[str, object],
) -> None:
    if observed.code_commit != expected["code_commit"]:
        raise ValueError(f"{name} producer commit differs from SOP05 input lock")
    if observed.checksum_manifest_sha256 != expected["checksum_manifest_sha256"]:
        raise ValueError(f"{name} checksum digest differs from SOP05 input lock")
    if observed.audit_sha256 != expected["audit_sha256"]:
        raise ValueError(f"{name} audit digest differs from SOP05 input lock")
    if observed.completion_policy != expected["completion_policy"]:
        raise ValueError(f"{name} completion policy differs from SOP05 input lock")


def validate_source_snippet_record_identity(
    snippet: MotionSnippet,
    record: EventTargetMotionRecord,
    *,
    split: str,
) -> None:
    """Validate the SOP03 geometry-only footprint against the SOP05 full spec."""

    if not isinstance(snippet, MotionSnippet):
        raise TypeError("snippet must be a MotionSnippet")
    if not isinstance(record, EventTargetMotionRecord):
        raise TypeError("record must be an EventTargetMotionRecord")
    expected_split = _string(split, name="split")
    identities = (
        (snippet.split, expected_split, "split"),
        (snippet.snippet_id, record.source_snippet_id, "snippet"),
        (snippet.object_type, record.object_type, "object type"),
        (snippet.source_object_id, record.source_object_id, "source object"),
    )
    for observed, expected, label in identities:
        if observed != expected:
            raise ValueError(f"SOP05 event/SOP03 snippet {label} identity mismatch")
    spec = _mapping(record.footprint_spec, name="SOP05 footprint spec")
    if spec.get("object_type") != record.object_type:
        raise ValueError("SOP05 record footprint object type mismatch")
    if snippet.footprint != spec.get("footprint"):
        raise ValueError("SOP05 event/SOP03 snippet footprint mismatch")


def load_joined_source_events(
    index: VerificationSourceIndex,
    *,
    sop03_root: str | Path,
    sop04_root: str | Path,
    grid: GridSpec,
    event_count: int,
    seed: int,
    checksum_workers: int = 1,
) -> VerificationSourceBundle:
    """Use formal upstream loaders, then join a deterministic small event set."""

    if not isinstance(index, VerificationSourceIndex):
        raise TypeError("index must be a VerificationSourceIndex")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    requested = _integer(event_count, name="event_count", minimum=1)
    if requested > index.event_count:
        raise ValueError("event_count exceeds the trusted SOP05 batch")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    workers = _integer(checksum_workers, name="checksum_workers", minimum=1)

    sop03 = load_sop03_split_inputs(
        sop03_root, index.split, grid, checksum_workers=workers
    )
    _validate_producer_evidence(
        name="SOP03",
        observed=sop03.producer_evidence,
        expected=index.sop03_input_lock,
    )
    sop04 = load_sop04_trajectory_bank(
        sop04_root,
        grid,
        expected_external_handoff_digest_sha256=str(
            index.sop04_input_lock["external_handoff_digest_sha256"]
        ),
        checksum_workers=workers,
    )
    _validate_producer_evidence(
        name="SOP04",
        observed=sop04.producer_evidence,
        expected=index.sop04_input_lock,
    )
    if (
        sop04.bank_semantic_digest_sha256
        != index.sop04_input_lock["bank_semantic_digest_sha256"]
        or sop04.pose_time_offsets_sha256
        != index.sop04_input_lock["pose_time_offsets_sha256"]
    ):
        raise ValueError("SOP04 semantic/time digest differs from SOP05 input lock")

    snippets: dict[str, MotionSnippet] = {}
    for library in sop03.typed_libraries.values():
        for snippet in library.snippets:
            if snippet.snippet_id in snippets:
                raise ValueError("duplicate snippet ID across typed SOP03 libraries")
            snippets[snippet.snippet_id] = snippet

    loaded_events: list[tuple[GeneratedEvent, Sop05ShardSource]] = []
    loaded_digests: list[str] = []
    for shard in select_source_shards(index, count=len(index.shards), seed=seed):
        loaded = load_complete_sop05_events(
            shard.root,
            grid=grid,
            expected_publication_semantic_digest=shard.publication_semantic_digest,
            expected_run_id=shard.run_id,
        )
        if loaded.split != index.split or len(loaded.events) != shard.event_count:
            raise ValueError("SOP05 formal loader result differs from batch handoff")
        loaded_digests.append(loaded.publication_semantic_digest)
        loaded_events.extend((event, shard) for event in loaded.events)
        if len(loaded_events) >= requested:
            break
    ranked = sorted(
        loaded_events,
        key=lambda item: (
            stable_digest(
                VERIFICATION_SOURCE_INDEX_VERSION,
                index.sop05_batch_digest,
                seed,
                item[0].generated_event_id,
                size=16,
            ),
            item[0].generated_event_id,
        ),
    )[:requested]
    joined: list[VerificationSourceEvent] = []
    for event, shard in ranked:
        record = event.target_motion_record
        if record.trajectory_id != shard.trajectory_id:
            raise ValueError("SOP05 event trajectory differs from shard handoff")
        if record.trajectory_id not in sop04.by_id:
            raise ValueError("SOP05 event trajectory is absent from SOP04 bank")
        if record.source_snippet_id not in snippets:
            raise ValueError("SOP05 event snippet is absent from SOP03 libraries")
        snippet = snippets[record.source_snippet_id]
        validate_source_snippet_record_identity(snippet, record, split=index.split)
        base_state, oracle_context = sop03.load_pair(record.base_state_id, grid)
        if (
            base_state.split != index.split
            or event.world.base_state_id != base_state.state_id
            or oracle_context.base_state_id != base_state.state_id
        ):
            raise ValueError("SOP05 event/SOP03 base-state join mismatch")
        joined.append(
            VerificationSourceEvent(
                event=event,
                base_state=base_state,
                oracle_context=oracle_context,
                nominal_trajectory=sop04.by_id[record.trajectory_id],
                source_snippet=snippet,
                shard=shard,
            )
        )
    if len(joined) != requested:
        raise ValueError("formal SOP05 loading produced too few selected events")
    return VerificationSourceBundle(
        index=index,
        sop03=sop03,
        sop04=sop04,
        events=tuple(joined),
        loaded_sop05_publication_digests=tuple(loaded_digests),
    )


__all__ = (
    "Sop05ShardSource",
    "VERIFICATION_SOURCE_INDEX_VERSION",
    "VerificationSourceBundle",
    "VerificationSourceEvent",
    "VerificationSourceIndex",
    "load_joined_source_events",
    "load_verification_source_index",
    "select_source_shards",
    "validate_source_snippet_record_identity",
)
