"""Strict loader for published SOP05R lightweight-TEB v2 collections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from src.contracts import BaseState, build_grid_spec, load_dataclass, validate_base_state

from .dynamic_object_transplant import TransplantedDynamicObject
from .event_sampler import GeneratedEvent
from .event_target_motion_shard import (
    LoadedEventTargetMotionShard,
    load_event_target_motion_shard,
)
from .sop05r_contracts import (
    SOP05R_TEB_COMPLETION_MARKER_VERSION,
    SOP05R_TEB_GENERATOR_VERSION,
    SOP05R_TEB_MANIFEST_VERSION,
    SOP05R_TEB_RUN_VERSION,
    SOP05R_TEB_SUMMARY_VERSION,
    SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION,
)
from .sop05r_teb_templates import canonical_sop05r_teb_base_state_digest
from .sop05r_teb_trajectory_store import (
    Sop05rTebTrajectoryStore,
    load_sop05r_teb_trajectory_store,
)


SOP05R_TEB_EVENT_ROW_VERSION = "sop05r_teb_event_row_v1"
SOP05R_TEB_EMPTY_TARGET_MOTION_VERSION = "sop05r_teb_empty_target_motion_v1"
_MANIFEST = "manifest.json"
_SUMMARY = "generation_summary.json"
_EVENTS = "events.json"
_CHECKSUMS = "checksums.json"
_COMPLETE = "COMPLETE.json"
_DECISION_STATES = "decision_states"
_TARGET_MOTION = "target_motion"
_TRAJECTORIES = "trajectory_store"


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
        raise ValueError("SOP05R TEB output must contain canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"failed to checksum SOP05R TEB artifact: {path}") from exc


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read SOP05R TEB JSON: {path.name}") from exc


def compute_sop05r_teb_publication_semantic_digest(
    *,
    event_rows: list[Mapping[str, object]],
    trajectory_collection_digest: str,
    target_motion_payload_digest: str,
    decision_state_digests: list[str],
    config_digest: str,
) -> str:
    """Bind every independently authenticated collection into one run identity."""

    payload = {
        "event_record_digests": [
            row["record_semantic_digest"] for row in event_rows
        ],
        "trajectory_collection_digest": trajectory_collection_digest,
        "target_motion_payload_digest": target_motion_payload_digest,
        "decision_state_digests": decision_state_digests,
        "config_digest": config_digest,
    }
    return _sha256(b"sop05r_teb_publication_v1\0" + _canonical_json(payload))


@dataclass(frozen=True)
class LoadedSop05rTebOutput:
    events: tuple[GeneratedEvent, ...]
    decision_states: Mapping[str, BaseState]
    trajectories: Sop05rTebTrajectoryStore
    target_motion: LoadedEventTargetMotionShard
    manifest: Mapping[str, object]
    summary: Mapping[str, object]
    publication_semantic_digest: str
    complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision_states",
            MappingProxyType(dict(self.decision_states)),
        )
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary)))


def _validate_checksums(root: Path, expected_root_entries: set[str]) -> None:
    checksums = _read_json(root / _CHECKSUMS)
    if not isinstance(checksums, dict) or set(checksums) != {
        "manifest_version",
        "files",
    }:
        raise ValueError("SOP05R TEB checksum manifest schema mismatch")
    if checksums["manifest_version"] != SOP05R_TEB_MANIFEST_VERSION:
        raise ValueError("SOP05R TEB checksum manifest version mismatch")
    files = checksums["files"]
    if not isinstance(files, dict):
        raise ValueError("SOP05R TEB checksum files must be a mapping")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != _CHECKSUMS
    }
    if set(files) != actual_files:
        raise ValueError("SOP05R TEB checksum file set mismatch")
    for relative, digest in files.items():
        if _sha256_file(root / relative) != digest:
            raise ValueError(f"SOP05R TEB checksum mismatch: {relative}")
    if {path.name for path in root.iterdir()} != expected_root_entries:
        raise ValueError("SOP05R TEB output root entry set mismatch")


def _event_record_digest(row: Mapping[str, object]) -> str:
    payload = dict(row)
    payload.pop("record_semantic_digest", None)
    return _sha256(b"sop05r_teb_event_row_v1\0" + _canonical_json(payload))


def load_sop05r_teb_output(
    input_dir: str | Path,
    *,
    require_complete: bool = False,
) -> LoadedSop05rTebOutput:
    """Load a v2 collection while rejecting v1, mixed, extra, or tampered files."""

    root = Path(input_dir)
    if not root.is_dir():
        raise ValueError("SOP05R TEB output directory does not exist")
    complete = (root / _COMPLETE).is_file()
    expected_root = {
        _MANIFEST,
        _SUMMARY,
        _EVENTS,
        _CHECKSUMS,
        _DECISION_STATES,
        _TARGET_MOTION,
        _TRAJECTORIES,
    }
    if complete:
        expected_root.add(_COMPLETE)
    if require_complete and not complete:
        raise ValueError("SOP05R TEB completion marker is missing")
    _validate_checksums(root, expected_root)

    manifest = _read_json(root / _MANIFEST)
    summary = _read_json(root / _SUMMARY)
    rows = _read_json(root / _EVENTS)
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        raise ValueError("SOP05R TEB manifest and summary must be mappings")
    if not isinstance(rows, list):
        raise ValueError("SOP05R TEB event rows must be a list")
    if manifest.get("manifest_version") != SOP05R_TEB_MANIFEST_VERSION:
        raise ValueError("v1 or unknown SOP05R manifest is not accepted by v2 loader")
    if manifest.get("run_version") != SOP05R_TEB_RUN_VERSION:
        raise ValueError("SOP05R TEB run version mismatch")
    if manifest.get("generator_algorithm_version") != SOP05R_TEB_GENERATOR_VERSION:
        raise ValueError("SOP05R TEB generator version mismatch")
    if summary.get("summary_version") != SOP05R_TEB_SUMMARY_VERSION:
        raise ValueError("SOP05R TEB summary version mismatch")
    base_config = manifest.get("base_config")
    if not isinstance(base_config, dict):
        raise ValueError("SOP05R TEB base_config snapshot is missing")
    grid = build_grid_spec(base_config)

    event_ids: list[str] = []
    decision_state_digests: list[str] = []
    decision_states: dict[str, BaseState] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("SOP05R TEB event row must be a mapping")
        if row.get("row_version") != SOP05R_TEB_EVENT_ROW_VERSION:
            raise ValueError("SOP05R TEB event row version mismatch")
        if row.get("row_index") != row_index:
            raise ValueError("SOP05R TEB event row index mismatch")
        if _event_record_digest(row) != row.get("record_semantic_digest"):
            raise ValueError("SOP05R TEB event row semantic digest mismatch")
        event_id = row.get("event_id")
        decision_id = row.get("decision_state_id")
        filename = row.get("decision_state_file")
        decision_digest = row.get("decision_state_digest")
        if (
            not isinstance(event_id, str)
            or not isinstance(decision_id, str)
            or not isinstance(filename, str)
            or not isinstance(decision_digest, str)
        ):
            raise ValueError("SOP05R TEB event identity fields are invalid")
        expected_filename = f"{decision_id}.npz"
        if filename != expected_filename:
            raise ValueError("SOP05R TEB decision-state filename mismatch")
        state = load_dataclass(root / _DECISION_STATES / filename)
        if not isinstance(state, BaseState):
            raise ValueError("SOP05R TEB decision-state payload must be BaseState")
        validate_base_state(state, grid)
        if state.state_id != decision_id:
            raise ValueError("SOP05R TEB decision-state identity mismatch")
        if canonical_sop05r_teb_base_state_digest(state) != decision_digest:
            raise ValueError("SOP05R TEB decision-state semantic digest mismatch")
        event_ids.append(event_id)
        decision_state_digests.append(decision_digest)
        decision_states[decision_id] = state
    if event_ids != sorted(event_ids) or len(event_ids) != len(set(event_ids)):
        raise ValueError("SOP05R TEB event IDs must be unique and sorted")
    actual_decision_files = {
        path.name for path in (root / _DECISION_STATES).iterdir() if path.is_file()
    }
    if actual_decision_files != {
        str(row["decision_state_file"]) for row in rows
    }:
        raise ValueError("SOP05R TEB decision-state file set mismatch")

    trajectories = load_sop05r_teb_trajectory_store(
        root / _TRAJECTORIES,
        require_complete=complete,
    )
    if event_ids:
        target_motion = load_event_target_motion_shard(
            root / _TARGET_MOTION,
            grid=grid,
            expected_generated_event_ids=set(event_ids),
            expected_base_state_ids=set(decision_states),
            expected_trajectory_ids={
                record.nominal_trajectory.trajectory_id
                for record in trajectories.records
            },
        )
    else:
        empty_dir = root / _TARGET_MOTION
        if {path.name for path in empty_dir.iterdir()} != {"empty.json"}:
            raise ValueError("empty SOP05R TEB target-motion file set mismatch")
        empty_summary = _read_json(empty_dir / "empty.json")
        if (
            not isinstance(empty_summary, dict)
            or empty_summary.get("version")
            != SOP05R_TEB_EMPTY_TARGET_MOTION_VERSION
            or empty_summary.get("record_count") != 0
            or not isinstance(empty_summary.get("manifest_digest"), str)
            or not isinstance(empty_summary.get("payload_semantic_digest"), str)
        ):
            raise ValueError("empty SOP05R TEB target-motion sentinel mismatch")
        target_motion = LoadedEventTargetMotionShard(
            records=(),
            worlds={},
            manifest_digest=str(empty_summary["manifest_digest"]),
            payload_semantic_digest=str(
                empty_summary["payload_semantic_digest"]
            ),
            summary=empty_summary,
        )
    trajectory_by_event = {
        record.event_id: record for record in trajectories.records
    }
    motion_by_event = {
        record.generated_event_id: record for record in target_motion.records
    }
    if set(trajectory_by_event) != set(event_ids) or set(motion_by_event) != set(
        event_ids
    ):
        raise ValueError("SOP05R TEB nested event identity sets disagree")

    events: list[GeneratedEvent] = []
    for row in rows:
        event_id = str(row["event_id"])
        motion = motion_by_event[event_id]
        trajectory = trajectory_by_event[event_id]
        if (
            trajectory.decision_state_id != row["decision_state_id"]
            or motion.base_state_id != row["decision_state_id"]
            or motion.trajectory_id != trajectory.nominal_trajectory.trajectory_id
            or motion.world_id != row["world_id"]
        ):
            raise ValueError("SOP05R TEB event nested identity join mismatch")
        provenance = row.get("target_provenance")
        if not isinstance(provenance, dict):
            raise ValueError("SOP05R TEB target provenance must be a mapping")
        target = TransplantedDynamicObject(
            target_dynamic_object_id=motion.target_dynamic_object_id,
            source_object_id=motion.source_object_id,
            snippet_id=motion.source_snippet_id,
            object_type=motion.object_type,
            footprint_spec=dict(motion.footprint_spec),
            footprint_spec_digest=motion.footprint_spec_digest,
            history_poses=motion.history_poses,
            current_pose=motion.current_pose,
            future_poses=motion.future_poses,
            provenance=provenance,
        )
        visibility = np.asarray(row["visibility_sequence"], dtype=np.bool_)
        history_visibility = np.asarray(
            row["target_visibility_history"],
            dtype=np.bool_,
        )
        if (
            visibility.shape != (grid.future_steps,)
            or history_visibility.shape != (grid.history_steps,)
        ):
            raise ValueError("SOP05R TEB visibility arrays have invalid shape")
        world = target_motion.worlds[motion.world_id]
        events.append(
            GeneratedEvent(
                generated_event_id=event_id,
                event_kind=str(row["event_kind"]),
                world=world,
                target=target,
                target_motion_record=motion,
                visibility_sequence=visibility,
                target_visibility_history=history_visibility,
                conflict_time_s=float(row["conflict_time_s"]),
                conflict_index=int(row["conflict_index"]),
            )
        )

    publication_digest = compute_sop05r_teb_publication_semantic_digest(
        event_rows=rows,
        trajectory_collection_digest=trajectories.collection_semantic_digest,
        target_motion_payload_digest=target_motion.payload_semantic_digest,
        decision_state_digests=decision_state_digests,
        config_digest=str(manifest["config_digest"]),
    )
    if (
        publication_digest != manifest.get("publication_semantic_digest")
        or publication_digest != summary.get("publication_semantic_digest")
    ):
        raise ValueError("SOP05R TEB publication semantic digest mismatch")
    if (
        manifest.get("trajectory_collection_version")
        != SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION
        or manifest.get("trajectory_collection_digest")
        != trajectories.collection_semantic_digest
        or manifest.get("target_motion_payload_digest")
        != target_motion.payload_semantic_digest
        or manifest.get("event_ids") != event_ids
    ):
        raise ValueError("SOP05R TEB manifest nested collection mismatch")
    if (
        summary.get("accepted_count") != len(events)
        or summary.get("requested_count") != manifest.get("requested_count")
        or bool(summary.get("quota_met")) != complete
    ):
        raise ValueError("SOP05R TEB generation summary mismatch")
    if complete:
        marker = _read_json(root / _COMPLETE)
        if (
            not isinstance(marker, dict)
            or marker.get("completion_marker_version")
            != SOP05R_TEB_COMPLETION_MARKER_VERSION
            or marker.get("publication_semantic_digest") != publication_digest
            or marker.get("accepted_count") != len(events)
            or marker.get("requested_count") != manifest.get("requested_count")
        ):
            raise ValueError("SOP05R TEB completion marker mismatch")
    return LoadedSop05rTebOutput(
        events=tuple(events),
        decision_states=decision_states,
        trajectories=trajectories,
        target_motion=target_motion,
        manifest=manifest,
        summary=summary,
        publication_semantic_digest=publication_digest,
        complete=complete,
    )


load_sop05r_teb_events = load_sop05r_teb_output
