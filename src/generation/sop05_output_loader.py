"""Strict consumer for complete SOP-05 publications.

This module is the restart-safe bridge between the on-disk SOP-05 artifact and
SOP-06.  It verifies the outer publication evidence, loads the versioned target
motion shard, and reconstructs ``GeneratedEvent`` instances solely from data
whose semantics are bound by the shard digests.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import numpy as np

from src.contracts import SCHEMA_VERSION, GridSpec, OracleWorld, build_grid_spec
from src.generation.dynamic_object_transplant import (
    MOTION_SNIPPET_CURRENT_INDEX,
    MOTION_SNIPPET_CURRENT_TIME_S,
    MOTION_SNIPPET_LAYOUT_VERSION,
    TransplantedDynamicObject,
    normalize_target_type_policy,
)
from src.generation.event_sampler import (
    GeneratedEvent,
    SOP05_GENERATOR_ALGORITHM_VERSION,
    _generator_digest,
    compute_generated_event_id,
    compute_generated_world_id,
    load_generator_config,
)
from src.generation.event_target_motion_shard import (
    EventTargetMotionRecord,
    LoadedEventTargetMotionShard,
    load_event_target_motion_shard,
    validate_event_target_motion_world_join,
)
from src.generation.sop05_selection import (
    SOP05_EVENT_KIND_ORDER,
    SOP05_PAIR_REPORT_VERSION,
    SOP05_RUN_PRODUCER_VERSION,
    SOP05_TOTAL_QUOTA_SELECTION_VERSION,
    select_sop05_event_ids,
)
from src.generation.sop05_publication_identity import (
    SOP05_PUBLICATION_IDENTITY_VERSION,
    compute_sop05_publication_semantic_digest,
)
from src.generation.structural_blindspot import (
    StructuralBlindSpot,
    has_continuous_emergence,
)
from src.utils.config import load_config


SOP05_RUN_MANIFEST_VERSION = "sop05_run_manifest_v2"
SOP05_GENERATION_SUMMARY_VERSION = "sop05_generation_summary_v2"
SOP05_COMPLETION_MARKER_VERSION = "sop05_producer_complete_v2"

_EVENT_KINDS = SOP05_EVENT_KIND_ORDER
_VISIBILITY_HISTORY_LAYOUT = "target_visibility_history8_current7_v1"
_RUN_MANIFEST_KEYS = frozenset(
    {
        "manifest_version",
        "producer_version",
        "producer_source_identity",
        "run_id",
        "run_state",
        "split",
        "input_lock",
        "scientific_request",
        "runtime",
        "artifacts",
    }
)
_ARTIFACT_KEYS = frozenset(
    {
        "base_config_snapshot",
        "generator_config_snapshot",
        "generation_summary",
        "pair_generation_reports",
        "checksums",
        "target_motion_shard",
        "producer_complete",
    }
)
_EXPECTED_ARTIFACTS = {
    "base_config_snapshot": "configs/base.yaml",
    "generator_config_snapshot": "configs/generator.yaml",
    "generation_summary": "generation_summary.json",
    "pair_generation_reports": "pair_generation_reports.jsonl",
    "checksums": "checksums.sha256",
    "target_motion_shard": "target_motions",
    "producer_complete": ".producer-complete",
}
_COMPLETION_MARKER_KEYS = frozenset(
    {
        "marker_version",
        "publication_identity_version",
        "publication_semantic_digest",
        "run_id",
        "run_manifest_sha256",
        "checksums_sha256",
        "target_motion_manifest_digest",
        "target_motion_payload_semantic_digest",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "snippet_id",
        "source_recording_id",
        "source_object_id",
        "source_body_name",
        "raw_role",
        "geometry_source",
        "orientation_source",
        "target_type_policy_digest",
        "footprint_spec_digest",
        "conflict_time_s",
        "conflict_point",
        "crossing_direction",
        "rotation_rad",
        "time_scale",
        "motion_snippet_layout_version",
        "source_current_index",
        "source_current_time_s",
        "source_conflict_anchor_time_s",
        "seed",
    }
)
_SOURCE_IDENTITY_KEYS = frozenset(
    {"version", "git_commit", "worktree_state", "dirty_tree_sha256"}
)
_INPUT_LOCK_KEYS = frozenset({"version", "split", "sop03", "sop04", "selection"})
_UPSTREAM_EVIDENCE_KEYS = frozenset(
    {
        "code_commit",
        "checksum_manifest_sha256",
        "audit_sha256",
        "completion_policy",
    }
)
_SOP04_EVIDENCE_KEYS = _UPSTREAM_EVIDENCE_KEYS | frozenset(
    {
        "trajectory_bank_version",
        "pose_time_layout_version",
        "trajectory_steps",
        "dt_s",
        "first_pose_time_s",
        "last_pose_time_s",
        "pose_time_offsets_sha256",
        "bank_semantic_digest_sha256",
        "external_handoff_digest_sha256",
    }
)
_SELECTION_KEYS = frozenset(
    {
        "seed",
        "max_base_states",
        "trajectory_count",
        "max_pairs",
        "pair_count",
        "pair_schedule_sha256",
    }
)
_SCIENTIFIC_REQUEST_KEYS = frozenset(
    {
        "seed",
        "accepted_quota",
        "events_per_pair",
        "max_base_states",
        "trajectory_count",
        "max_pairs",
        "selection_version",
        "base_config_sha256",
        "generator_config_sha256",
        "generator_config_semantic_digest",
        "target_type_policy",
        "target_type_policy_digest",
        "pair_schedule",
    }
)
_RUNTIME_KEYS = frozenset(
    {"workers", "checksum_workers", "git_executable", "resolved_input_roots"}
)
_RESOLVED_ROOT_KEYS = frozenset({"sop03", "sop04"})
_PAIR_SCHEDULE_ROW_KEYS = frozenset(
    {"rank", "state_id", "trajectory_id", "pair_seed"}
)
_PAIR_REPORT_KEYS = frozenset(
    {
        "report_version",
        "selection_version",
        "rank",
        "state_id",
        "trajectory_id",
        "seed",
        "summary",
        "accepted_events",
    }
)
_ACCEPTED_EVENT_KEYS = frozenset({"generated_event_id", "event_kind"})
_PAIR_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "seed",
        "requested_event_count",
        "complete_joint_candidates_attempted",
        "attempted_count",
        "joint_candidate_attempted_count",
        "attempt_index_start",
        "attempt_index_stop_exclusive",
        "accepted_count",
        "rejected_count",
        "acceptance_rate",
        "attempt_acceptance_rate",
        "request_acceptance_rate",
        "unaccepted_event_count",
        "rejection_reasons",
        "rejection_stage_counts",
        "occluder_candidate_rejection_reasons",
        "requested_event_kind_counts",
        "event_kind_counts",
        "by_event_kind",
        "by_object_type",
        "by_footprint_kind",
        "by_geometry_source",
        "target_type_policy",
        "target_type_policy_digest",
        "generator_config_digest",
        "generator_algorithm_version",
    }
)
_GLOBAL_SUMMARY_KEYS = frozenset(
    {
        "summary_version",
        "run_id",
        "run_state",
        "processed_pair_count",
        "requested_event_count",
        "attempted_count",
        "generator_accepted_count",
        "selected_count",
        "quota_trimmed_count",
        "generated_event_kind_counts",
        "selected_event_kind_counts",
        "quota_met",
        "rejection_reasons",
        "rejection_stage_counts",
        "generator_invariants",
    }
)
_GENERATOR_INVARIANT_KEYS = frozenset(
    {
        "schema_version",
        "target_type_policy_digest",
        "generator_config_digest",
        "generator_algorithm_version",
    }
)
_REJECTION_STAGE_KEYS = frozenset(
    {"occluder_geometry", "target_conditioning", "visibility"}
)
_EVENT_KIND_BUCKET_KEYS = frozenset(
    {
        "requested",
        "attempted",
        "accepted",
        "rejected",
        "request_acceptance_rate",
        "attempt_acceptance_rate",
        "rejection_reasons",
        "rejection_stage_counts",
    }
)
_BUCKET_KEYS = frozenset(
    {
        "attempted",
        "accepted",
        "rejected",
        "attempt_acceptance_rate",
        "rejection_reasons",
    }
)
_BLIND_SPOT_KEYS = frozenset({"kind", "structural", "occluder_ids"})
_STRUCTURAL_KEYS = frozenset({"forward_fov_deg", "range_m", "blind_sectors"})
_ALLOWED_SPLITS = frozenset({"train", "calibration", "val", "test"})


@dataclass(frozen=True)
class _ValidatedRunContract:
    run_id: str
    split: str
    schedule: tuple[dict[str, object], ...]
    selection_seed: int
    accepted_quota: int
    events_per_pair: int
    target_type_policy: dict[str, object]
    target_type_policy_digest: str
    generator_config_semantic_digest: str


@dataclass(frozen=True)
class _ValidatedPairReports:
    generated_event_ids: frozenset[str]
    event_pair_identity: Mapping[str, tuple[str, str]]
    event_kind_by_id: Mapping[str, str]
    selected_event_ids: tuple[str, ...]
    requested_event_count: int
    attempted_count: int
    generated_event_kind_counts: dict[str, int]
    rejection_reasons: dict[str, int]
    rejection_stage_counts: dict[str, int]
    generator_invariants: dict[str, object]


@dataclass(frozen=True)
class LoadedSop05Events:
    """A complete, evidence-checked SOP-05 run ready for SOP-06."""

    run_id: str
    publication_semantic_digest: str
    split: str
    events: tuple[GeneratedEvent, ...]
    events_by_id: Mapping[str, GeneratedEvent]
    shard: LoadedEventTargetMotionShard
    run_manifest: dict[str, object]
    generation_summary: dict[str, object]


def _load_json(path: Path, *, label: str) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite JSON value {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}") from exc


def _canonical_json_copy(value: object, *, label: str) -> object:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite canonical JSON") from exc


def _canonical_json_bytes(value: object, *, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite canonical JSON") from exc


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: frozenset[str], *, label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys do not match the frozen contract")


def _require_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validate_source_identity(value: object) -> None:
    identity = _require_mapping(value, label="SOP05 producer source identity")
    _require_exact_keys(
        identity,
        _SOURCE_IDENTITY_KEYS,
        label="SOP05 producer source identity",
    )
    if identity.get("version") != "sop05_producer_source_identity_v1":
        raise ValueError("unsupported SOP05 producer source identity")
    commit = identity.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(value not in "0123456789abcdef" for value in commit)
    ):
        raise ValueError("SOP05 producer git_commit is invalid")
    state = identity.get("worktree_state")
    digest = identity.get("dirty_tree_sha256")
    if state == "clean" and digest is None:
        return
    if state != "dirty" or (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(value not in "0123456789abcdef" for value in digest)
    ):
        raise ValueError("SOP05 producer dirty-tree identity is invalid")


def _finite_real(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _positive_int(value: object, *, label: str) -> int:
    result = _nonnegative_int(value, label=label)
    if result == 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _require_hex_digest(value: object, *, size: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != size
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase hexadecimal digest")
    return value


def _require_rate(value: object, *, label: str) -> float:
    result = _finite_real(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return result


def _require_count_map(
    value: object,
    *,
    label: str,
    exact_keys: frozenset[str] | None = None,
) -> dict[str, int]:
    mapping = _require_mapping(value, label=label)
    if exact_keys is not None:
        _require_exact_keys(mapping, exact_keys, label=label)
    result: dict[str, int] = {}
    for key, count in mapping.items():
        if not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        result[key] = _nonnegative_int(count, label=f"{label}[{key!r}]")
    return result


def _require_rate_equal(actual: float, expected: float, *, label: str) -> None:
    if abs(actual - expected) > 1e-12:
        raise ValueError(f"{label} mismatch")


def _bool_vector(value: object, *, size: int, label: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must be a boolean vector with shape ({size},)")
    if any(type(item) is not bool for item in value):
        raise ValueError(f"{label} must contain only boolean values")
    return np.array(value, dtype=np.bool_, order="C", copy=True)


def _validate_numeric_vector(
    value: object, *, size: int, label: str
) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must have length {size}")
    result = np.empty(size, dtype=np.float64)
    for index, item in enumerate(value):
        result[index] = _finite_real(item, label=f"{label}[{index}]")
    return result


def _validate_event_skeleton(world: OracleWorld, event_kind: str) -> None:
    blind_spot = _require_mapping(
        world.blind_spot_config, label="world blind_spot_config"
    )
    _require_exact_keys(
        blind_spot, _BLIND_SPOT_KEYS, label="world blind_spot_config"
    )
    if blind_spot.get("kind") != event_kind:
        raise ValueError("world event skeleton kind mismatch")
    raw_occluder_ids = blind_spot.get("occluder_ids")
    if not isinstance(raw_occluder_ids, list) or any(
        not isinstance(value, str) or not value for value in raw_occluder_ids
    ):
        raise ValueError("world event skeleton occluder_ids are invalid")
    if len(raw_occluder_ids) != len(set(raw_occluder_ids)):
        raise ValueError("world event skeleton occluder_ids are not unique")

    world_occluder_ids: list[str] = []
    for index, raw_occluder in enumerate(world.occluders):
        occluder = _require_mapping(
            raw_occluder, label=f"world occluders[{index}]"
        )
        occluder_id = _require_nonempty_string(
            occluder.get("occluder_id"),
            label=f"world occluders[{index}] occluder_id",
        )
        world_occluder_ids.append(occluder_id)
    if len(world_occluder_ids) != len(set(world_occluder_ids)):
        raise ValueError("world event skeleton occluder IDs are not unique")
    if raw_occluder_ids != world_occluder_ids:
        raise ValueError("world event skeleton occluder ID join mismatch")

    structural = blind_spot.get("structural")
    if event_kind == "environment":
        if not world_occluder_ids:
            raise ValueError("environment event skeleton requires an occluder")
        if structural is not None:
            raise ValueError(
                "environment event skeleton cannot contain a structural sensor"
            )
        return
    if event_kind == "structural" and world_occluder_ids:
        raise ValueError("structural event skeleton cannot contain an occluder")
    if event_kind == "mixed" and not world_occluder_ids:
        raise ValueError("mixed event skeleton requires an occluder")
    structural_mapping = _require_mapping(
        structural, label=f"{event_kind} event skeleton structural sensor"
    )
    _require_exact_keys(
        structural_mapping,
        _STRUCTURAL_KEYS,
        label=f"{event_kind} event skeleton structural sensor",
    )
    sectors = structural_mapping.get("blind_sectors")
    if not isinstance(sectors, list):
        raise ValueError("event skeleton blind_sectors must be a list")
    try:
        StructuralBlindSpot(
            forward_fov_deg=structural_mapping.get("forward_fov_deg"),
            range_m=structural_mapping.get("range_m"),
            blind_sectors=tuple(dict(sector) for sector in sectors),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("event skeleton structural sensor is invalid") from exc


def _validate_target_provenance(
    record: EventTargetMotionRecord,
    raw: object,
    *,
    conflict_time_s: float,
    conflict_index: int,
    attempt_seed: int,
) -> dict[str, object]:
    mapping = _require_mapping(raw, label="target_provenance")
    _require_exact_keys(mapping, _PROVENANCE_KEYS, label="target_provenance")
    copied = _canonical_json_copy(mapping, label="target_provenance")
    if not isinstance(copied, dict):
        raise ValueError("target_provenance must be an object")
    identity = {
        "snippet_id": record.source_snippet_id,
        "source_object_id": record.source_object_id,
        "target_type_policy_digest": record.target_type_policy_digest,
        "footprint_spec_digest": record.footprint_spec_digest,
    }
    for key, expected in identity.items():
        if copied.get(key) != expected:
            raise ValueError(f"target_provenance {key} mismatch")
    for key in ("source_recording_id", "geometry_source", "orientation_source"):
        _require_nonempty_string(copied.get(key), label=f"target_provenance {key}")
    for key in ("source_body_name", "raw_role"):
        value = copied.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(
                f"target_provenance {key} must be None or a non-empty string"
            )
    stored_conflict_time = _finite_real(
        copied.get("conflict_time_s"),
        label="target_provenance conflict_time_s",
    )
    if abs(stored_conflict_time - conflict_time_s) > 1e-9:
        raise ValueError("target_provenance conflict_time_s mismatch")
    time_scale = _finite_real(
        copied.get("time_scale"), label="target_provenance time_scale"
    )
    if time_scale != 1.0:
        raise ValueError("target_provenance time_scale must equal 1.0")
    if copied.get("motion_snippet_layout_version") != (
        MOTION_SNIPPET_LAYOUT_VERSION
    ):
        raise ValueError("target_provenance motion snippet layout mismatch")
    if copied.get("source_current_index") != MOTION_SNIPPET_CURRENT_INDEX:
        raise ValueError("target_provenance source current index mismatch")
    source_current_time = _finite_real(
        copied.get("source_current_time_s"),
        label="target_provenance source_current_time_s",
    )
    if abs(source_current_time - MOTION_SNIPPET_CURRENT_TIME_S) > 1e-9:
        raise ValueError("target_provenance source current time mismatch")
    source_anchor_time = _finite_real(
        copied.get("source_conflict_anchor_time_s"),
        label="target_provenance source_conflict_anchor_time_s",
    )
    expected_anchor_time = MOTION_SNIPPET_CURRENT_TIME_S + conflict_time_s
    if abs(source_anchor_time - expected_anchor_time) > 1e-9:
        raise ValueError("target_provenance source conflict anchor mismatch")
    conflict_point = _validate_numeric_vector(
        copied.get("conflict_point"), size=2, label="target_provenance conflict_point"
    )
    _validate_numeric_vector(
        copied.get("crossing_direction"),
        size=2,
        label="target_provenance crossing_direction",
    )
    _finite_real(copied.get("rotation_rad"), label="target_provenance rotation_rad")
    provenance_seed = _nonnegative_int(
        copied.get("seed"), label="target_provenance seed"
    )
    if provenance_seed != attempt_seed:
        raise ValueError(
            "target_provenance seed does not match world random_seed"
        )
    expected_conflict_point = np.asarray(
        record.future_poses[conflict_index, :2], dtype=np.float64
    )
    if not np.allclose(
        conflict_point,
        expected_conflict_point,
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError(
            "target_provenance conflict_point does not match target future"
        )
    return copied


def _validate_generated_identity(
    record: EventTargetMotionRecord,
    world: OracleWorld,
    metadata: Mapping[str, object],
    *,
    event_kind: str,
    conflict_index: int,
    conflict_time_s: float,
) -> int:
    generator_algorithm_version = _require_nonempty_string(
        metadata.get("generator_algorithm_version"),
        label="generator_algorithm_version",
    )
    if generator_algorithm_version != SOP05_GENERATOR_ALGORITHM_VERSION:
        raise ValueError(
            "generator_algorithm_version does not match the frozen "
            f"{SOP05_GENERATOR_ALGORITHM_VERSION!r} contract"
        )
    generator_config_digest = _require_hex_digest(
        metadata.get("generator_config_digest"),
        size=32,
        label="generator_config_digest",
    )
    event_slot_index = _nonnegative_int(
        metadata.get("event_slot_index"), label="event_slot_index"
    )
    attempt_index = _nonnegative_int(
        metadata.get("attempt_index"), label="attempt_index"
    )
    attempt_seed = _nonnegative_int(world.random_seed, label="random_seed")

    expected_event_id = compute_generated_event_id(
        generator_algorithm_version=generator_algorithm_version,
        generator_config_digest=generator_config_digest,
        base_state_id=record.base_state_id,
        trajectory_id=record.trajectory_id,
        event_index=event_slot_index,
        attempt_index=attempt_index,
        attempt_seed=attempt_seed,
        event_kind=event_kind,
        conflict_index=conflict_index,
        conflict_time_s=conflict_time_s,
        target_dynamic_object_id=record.target_dynamic_object_id,
        source_snippet_id=record.source_snippet_id,
        source_object_id=record.source_object_id,
        object_type=record.object_type,
        footprint_spec=record.footprint_spec,
        footprint_spec_digest=record.footprint_spec_digest,
        target_type_policy_digest=record.target_type_policy_digest,
        layout_version=record.layout_version,
    )
    if expected_event_id != record.generated_event_id:
        raise ValueError(
            "generated_event_id does not match canonical event identity"
        )

    expected_world_id = compute_generated_world_id(
        generator_algorithm_version=generator_algorithm_version,
        generator_config_digest=generator_config_digest,
        generated_event_id=record.generated_event_id,
        base_state_id=record.base_state_id,
        trajectory_id=record.trajectory_id,
        event_kind=event_kind,
        target_dynamic_object_id=record.target_dynamic_object_id,
        source_snippet_id=record.source_snippet_id,
        source_object_id=record.source_object_id,
        object_type=record.object_type,
        footprint_spec=record.footprint_spec,
        footprint_spec_digest=record.footprint_spec_digest,
        target_type_policy_digest=record.target_type_policy_digest,
        layout_version=record.layout_version,
        history_array_digest=record.history_array_digest,
        current_pose=record.current_pose,
        future_array_digest=record.future_array_digest,
    )
    if expected_world_id != record.world_id:
        raise ValueError("world_id does not match canonical world identity")
    return attempt_seed


def restore_generated_event(
    record: EventTargetMotionRecord,
    world: OracleWorld,
    *,
    grid: GridSpec,
) -> GeneratedEvent:
    """Reconstruct one mother event from its digest-bound shard row/world."""

    if not isinstance(record, EventTargetMotionRecord):
        raise TypeError("record must be an EventTargetMotionRecord")
    if not isinstance(world, OracleWorld):
        raise TypeError("world must be an OracleWorld")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    validate_event_target_motion_world_join(record, world, grid)
    metadata = _require_mapping(world.metadata, label="world metadata")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("world metadata schema_version mismatch")
    event_kind = metadata.get("event_kind")
    if event_kind not in _EVENT_KINDS:
        raise ValueError("world metadata event_kind is invalid")
    blind_spot_config = _require_mapping(
        world.blind_spot_config, label="world blind_spot_config"
    )
    if blind_spot_config.get("kind") != event_kind:
        raise ValueError("world metadata event_kind/blind spot mismatch")
    _validate_event_skeleton(world, event_kind)
    if metadata.get("dynamic_object_snippet_id") != record.source_snippet_id:
        raise ValueError("world metadata dynamic_object_snippet_id mismatch")

    conflict_index = _nonnegative_int(
        metadata.get("conflict_index"), label="conflict_index"
    )
    conflict_time_s = _finite_real(
        metadata.get("conflict_time_s"), label="conflict_time_s"
    )
    if conflict_index >= grid.future_steps or abs(
        conflict_time_s - (conflict_index + 1) * 0.2
    ) > 1e-9:
        raise ValueError("conflict index/time mismatch")

    attempt_seed = _validate_generated_identity(
        record,
        world,
        metadata,
        event_kind=event_kind,
        conflict_index=conflict_index,
        conflict_time_s=conflict_time_s,
    )

    target_policy = _require_mapping(
        metadata.get("target_type_policy"), label="target_type_policy"
    )
    normalized_policy = normalize_target_type_policy(target_policy)
    if normalized_policy.digest != record.target_type_policy_digest:
        raise ValueError("target_type_policy digest mismatch")
    if record.object_type not in normalized_policy.whitelist:
        raise ValueError("target object type is outside target_type_policy")

    target_visibility_history = _bool_vector(
        metadata.get("target_visibility_history"),
        size=8,
        label="target_visibility_history",
    )
    visibility_sequence = _bool_vector(
        metadata.get("visibility_sequence"),
        size=16,
        label="visibility_sequence",
    )
    if metadata.get("target_visibility_history_layout") != (
        _VISIBILITY_HISTORY_LAYOUT
    ):
        raise ValueError("target visibility history layout mismatch")
    if (
        bool(target_visibility_history[-1])
        or bool(visibility_sequence[0])
        or bool(target_visibility_history[-1]) != bool(visibility_sequence[0])
    ):
        raise ValueError("target visibility seam/current-hidden invariant failed")
    if not has_continuous_emergence(
        visibility_sequence, min_visible_frames=2
    ) or not bool(visibility_sequence[-1]):
        raise ValueError("target visibility continuous-emergence invariant failed")

    context_ids = metadata.get("context_dynamic_object_ids")
    if not isinstance(context_ids, list) or any(
        not isinstance(value, str) or not value for value in context_ids
    ):
        raise ValueError("context_dynamic_object_ids must be a string list")
    if context_ids != sorted(set(context_ids)):
        raise ValueError("context_dynamic_object_ids must be sorted and unique")
    expected_context_ids = sorted(
        set(world.dynamic_object_trajectories)
        - {record.target_dynamic_object_id}
    )
    if context_ids != expected_context_ids:
        raise ValueError("context_dynamic_object_ids mismatch world")

    provenance = _validate_target_provenance(
        record,
        metadata.get("target_provenance"),
        conflict_time_s=conflict_time_s,
        conflict_index=conflict_index,
        attempt_seed=attempt_seed,
    )
    footprint_spec = _canonical_json_copy(
        record.footprint_spec, label="target footprint spec"
    )
    if not isinstance(footprint_spec, dict):
        raise RuntimeError("validated target footprint spec changed type")
    target = TransplantedDynamicObject(
        target_dynamic_object_id=record.target_dynamic_object_id,
        source_object_id=record.source_object_id,
        snippet_id=record.source_snippet_id,
        object_type=record.object_type,
        footprint_spec=footprint_spec,
        footprint_spec_digest=record.footprint_spec_digest,
        history_poses=np.array(
            record.history_poses, dtype=np.float32, order="C", copy=True
        ),
        current_pose=np.array(
            record.current_pose, dtype=np.float32, order="C", copy=True
        ),
        future_poses=np.array(
            record.future_poses, dtype=np.float32, order="C", copy=True
        ),
        provenance=provenance,
    )
    return GeneratedEvent(
        generated_event_id=record.generated_event_id,
        event_kind=event_kind,
        world=world,
        target=target,
        target_motion_record=record,
        visibility_sequence=visibility_sequence,
        target_visibility_history=target_visibility_history,
        conflict_time_s=conflict_time_s,
        conflict_index=conflict_index,
    )


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"failed to read publication file {path.name!r}") from exc


def _validate_upstream_evidence(
    value: object,
    *,
    label: str,
    completion_policy: str,
) -> dict[str, str]:
    evidence = _require_mapping(value, label=label)
    _require_exact_keys(evidence, _UPSTREAM_EVIDENCE_KEYS, label=label)
    commit = _require_hex_digest(
        evidence.get("code_commit"), size=40, label=f"{label} code_commit"
    )
    checksum = _require_hex_digest(
        evidence.get("checksum_manifest_sha256"),
        size=64,
        label=f"{label} checksum_manifest_sha256",
    )
    audit = _require_hex_digest(
        evidence.get("audit_sha256"), size=64, label=f"{label} audit_sha256"
    )
    if evidence.get("completion_policy") != completion_policy:
        raise ValueError(f"{label} completion_policy mismatch")
    return {
        "code_commit": commit,
        "checksum_manifest_sha256": checksum,
        "audit_sha256": audit,
    }


def _validate_sop04_evidence(value: object, *, label: str) -> dict[str, object]:
    evidence = _require_mapping(value, label=label)
    _require_exact_keys(evidence, _SOP04_EVIDENCE_KEYS, label=label)
    identity: dict[str, object] = _validate_upstream_evidence(
        {key: evidence[key] for key in _UPSTREAM_EVIDENCE_KEYS},
        label=label,
        completion_policy="sop04_audited_bank_v2",
    )
    if evidence.get("trajectory_bank_version") != "sop04_audited_bank_v2":
        raise ValueError(f"{label} trajectory bank version mismatch")
    if (
        evidence.get("pose_time_layout_version")
        != "future_endpoints_dt_to_horizon_v1"
    ):
        raise ValueError(f"{label} pose-time layout version mismatch")
    if evidence.get("trajectory_steps") != 15:
        raise ValueError(f"{label} trajectory_steps mismatch")
    for name, expected in (
        ("dt_s", 0.2),
        ("first_pose_time_s", 0.2),
        ("last_pose_time_s", 3.0),
    ):
        observed = _finite_real(evidence.get(name), label=f"{label} {name}")
        if abs(observed - expected) > 1e-12:
            raise ValueError(f"{label} {name} mismatch")
        identity[name] = observed
    expected_offsets = tuple(
        float(value)
        for value in (np.arange(15, dtype=np.float64) + 1.0) * 0.2
    )
    offsets_payload = json.dumps(
        list(expected_offsets),
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    expected_offsets_digest = hashlib.sha256(
        b"sop04_pose_time_offsets_v1\0" + offsets_payload
    ).hexdigest()
    offsets_digest = _require_hex_digest(
        evidence.get("pose_time_offsets_sha256"),
        size=64,
        label=f"{label} pose_time_offsets_sha256",
    )
    if offsets_digest != expected_offsets_digest:
        raise ValueError(f"{label} pose time offsets digest mismatch")
    identity.update(
        {
            "trajectory_bank_version": "sop04_audited_bank_v2",
            "pose_time_layout_version": "future_endpoints_dt_to_horizon_v1",
            "trajectory_steps": 15,
            "pose_time_offsets_sha256": offsets_digest,
            "bank_semantic_digest_sha256": _require_hex_digest(
                evidence.get("bank_semantic_digest_sha256"),
                size=64,
                label=f"{label} bank semantic digest",
            ),
            "external_handoff_digest_sha256": _require_hex_digest(
                evidence.get("external_handoff_digest_sha256"),
                size=64,
                label=f"{label} external handoff digest",
            ),
        }
    )
    return identity


def _validate_pair_schedule(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("SOP05 scientific_request pair_schedule must be non-empty")
    schedule: list[dict[str, object]] = []
    pair_keys: set[tuple[str, str]] = set()
    for index, raw_row in enumerate(value):
        row = _require_mapping(
            raw_row, label=f"SOP05 pair_schedule[{index}]"
        )
        _require_exact_keys(
            row,
            _PAIR_SCHEDULE_ROW_KEYS,
            label=f"SOP05 pair_schedule[{index}]",
        )
        rank = _nonnegative_int(
            row.get("rank"), label=f"SOP05 pair_schedule[{index}] rank"
        )
        if rank != index:
            raise ValueError("SOP05 pair_schedule ranks must equal 0..N-1")
        state_id = _require_nonempty_string(
            row.get("state_id"),
            label=f"SOP05 pair_schedule[{index}] state_id",
        )
        trajectory_id = _require_nonempty_string(
            row.get("trajectory_id"),
            label=f"SOP05 pair_schedule[{index}] trajectory_id",
        )
        pair_seed = _nonnegative_int(
            row.get("pair_seed"),
            label=f"SOP05 pair_schedule[{index}] pair_seed",
        )
        pair_key = (state_id, trajectory_id)
        if pair_key in pair_keys:
            raise ValueError("SOP05 pair_schedule contains duplicate pairs")
        pair_keys.add(pair_key)
        schedule.append(
            {
                "rank": rank,
                "state_id": state_id,
                "trajectory_id": trajectory_id,
                "pair_seed": pair_seed,
            }
        )
    return tuple(schedule)


def _validate_runtime(value: object) -> None:
    runtime = _require_mapping(value, label="SOP05 runtime")
    _require_exact_keys(runtime, _RUNTIME_KEYS, label="SOP05 runtime")
    _positive_int(runtime.get("workers"), label="SOP05 runtime workers")
    _positive_int(
        runtime.get("checksum_workers"),
        label="SOP05 runtime checksum_workers",
    )
    git_executable = _require_nonempty_string(
        runtime.get("git_executable"), label="SOP05 runtime git_executable"
    )
    if not Path(git_executable).is_absolute():
        raise ValueError("SOP05 runtime git_executable must be absolute")
    roots = _require_mapping(
        runtime.get("resolved_input_roots"),
        label="SOP05 runtime resolved_input_roots",
    )
    _require_exact_keys(
        roots, _RESOLVED_ROOT_KEYS, label="SOP05 runtime resolved_input_roots"
    )
    for name in sorted(_RESOLVED_ROOT_KEYS):
        path = _require_nonempty_string(
            roots.get(name), label=f"SOP05 runtime resolved_input_roots {name}"
        )
        if not Path(path).is_absolute():
            raise ValueError(f"SOP05 runtime resolved_input_roots {name} is not absolute")


def _validate_run_contract(
    root: Path,
    run_manifest: Mapping[str, object],
    *,
    grid: GridSpec,
) -> _ValidatedRunContract:
    split = _require_nonempty_string(
        run_manifest.get("split"), label="SOP05 split"
    )
    if split not in _ALLOWED_SPLITS:
        raise ValueError("SOP05 split is outside the frozen split set")

    input_lock = _require_mapping(
        run_manifest.get("input_lock"), label="SOP05 input_lock"
    )
    _require_exact_keys(input_lock, _INPUT_LOCK_KEYS, label="SOP05 input_lock")
    if input_lock.get("version") != "sop05_input_lock_v2":
        raise ValueError("unsupported SOP05 input_lock version")
    if input_lock.get("split") != split:
        raise ValueError("SOP05 input_lock split mismatch")
    sop03_identity = _validate_upstream_evidence(
        input_lock.get("sop03"),
        label="SOP05 input_lock sop03",
        completion_policy="sop03_complete_marker_v1",
    )
    sop04_identity = _validate_sop04_evidence(
        input_lock.get("sop04"),
        label="SOP05 input_lock sop04",
    )
    selection = _require_mapping(
        input_lock.get("selection"), label="SOP05 input_lock selection"
    )
    _require_exact_keys(
        selection, _SELECTION_KEYS, label="SOP05 input_lock selection"
    )
    selection_values = {
        "seed": _nonnegative_int(
            selection.get("seed"), label="SOP05 input_lock selection seed"
        ),
        "max_base_states": _positive_int(
            selection.get("max_base_states"),
            label="SOP05 input_lock selection max_base_states",
        ),
        "trajectory_count": _positive_int(
            selection.get("trajectory_count"),
            label="SOP05 input_lock selection trajectory_count",
        ),
        "max_pairs": _positive_int(
            selection.get("max_pairs"),
            label="SOP05 input_lock selection max_pairs",
        ),
        "pair_count": _positive_int(
            selection.get("pair_count"),
            label="SOP05 input_lock selection pair_count",
        ),
    }
    schedule_digest = _require_hex_digest(
        selection.get("pair_schedule_sha256"),
        size=64,
        label="SOP05 input_lock selection pair_schedule_sha256",
    )

    scientific = _require_mapping(
        run_manifest.get("scientific_request"),
        label="SOP05 scientific_request",
    )
    _require_exact_keys(
        scientific,
        _SCIENTIFIC_REQUEST_KEYS,
        label="SOP05 scientific_request",
    )
    scientific_ints = {
        "seed": _nonnegative_int(
            scientific.get("seed"), label="SOP05 scientific_request seed"
        ),
        "accepted_quota": _positive_int(
            scientific.get("accepted_quota"),
            label="SOP05 scientific_request accepted_quota",
        ),
        "events_per_pair": _positive_int(
            scientific.get("events_per_pair"),
            label="SOP05 scientific_request events_per_pair",
        ),
        "max_base_states": _positive_int(
            scientific.get("max_base_states"),
            label="SOP05 scientific_request max_base_states",
        ),
        "trajectory_count": _positive_int(
            scientific.get("trajectory_count"),
            label="SOP05 scientific_request trajectory_count",
        ),
        "max_pairs": _positive_int(
            scientific.get("max_pairs"),
            label="SOP05 scientific_request max_pairs",
        ),
    }
    if scientific_ints["events_per_pair"] % 10:
        raise ValueError("SOP05 scientific_request events_per_pair must be a multiple of 10")
    for name in ("seed", "max_base_states", "trajectory_count", "max_pairs"):
        if scientific_ints[name] != selection_values[name]:
            raise ValueError(f"SOP05 scientific_request {name} differs from input_lock")

    schedule = _validate_pair_schedule(scientific.get("pair_schedule"))
    if selection_values["pair_count"] != len(schedule):
        raise ValueError("SOP05 input_lock pair_count differs from pair_schedule")
    if len(schedule) > selection_values["max_pairs"]:
        raise ValueError("SOP05 pair_schedule exceeds max_pairs")
    computed_schedule_digest = hashlib.sha256(
        _canonical_json_bytes(list(schedule), label="SOP05 pair_schedule")
    ).hexdigest()
    if schedule_digest != computed_schedule_digest:
        raise ValueError("SOP05 input_lock pair_schedule_sha256 mismatch")
    if len(schedule) * scientific_ints["events_per_pair"] < scientific_ints[
        "accepted_quota"
    ]:
        raise ValueError("SOP05 scientific_request has insufficient schedule capacity")

    if scientific.get("selection_version") != (
        SOP05_TOTAL_QUOTA_SELECTION_VERSION
    ):
        raise ValueError("unsupported SOP05 scientific_request selection version")

    base_digest = _require_hex_digest(
        scientific.get("base_config_sha256"),
        size=64,
        label="SOP05 scientific_request base_config_sha256",
    )
    generator_file_digest = _require_hex_digest(
        scientific.get("generator_config_sha256"),
        size=64,
        label="SOP05 scientific_request generator_config_sha256",
    )
    generator_semantic_digest = _require_hex_digest(
        scientific.get("generator_config_semantic_digest"),
        size=32,
        label="SOP05 scientific_request generator_config_semantic_digest",
    )
    policy_digest = _require_hex_digest(
        scientific.get("target_type_policy_digest"),
        size=32,
        label="SOP05 scientific_request target_type_policy_digest",
    )
    base_config_path = root / "configs/base.yaml"
    generator_config_path = root / "configs/generator.yaml"
    if _sha256(base_config_path) != base_digest:
        raise ValueError("SOP05 scientific_request base_config_sha256 mismatch")
    if _sha256(generator_config_path) != generator_file_digest:
        raise ValueError("SOP05 scientific_request generator_config_sha256 mismatch")
    try:
        base_config = load_config(base_config_path)
        generator_config = load_generator_config(generator_config_path)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("SOP05 config snapshot is invalid") from exc
    if build_grid_spec(base_config) != grid:
        raise ValueError("SOP05 base config grid differs from requested grid")
    for name in ("history_dt_s", "future_dt_s"):
        if abs(
            _finite_real(base_config["bev"].get(name), label=f"SOP05 base config {name}")
            - 0.2
        ) > 1e-9:
            raise ValueError(f"SOP05 base config {name} must equal 0.2")
    observed_generator_digest = _generator_digest(generator_config)
    if observed_generator_digest != generator_semantic_digest:
        raise ValueError(
            "SOP05 scientific_request generator_config_semantic_digest mismatch"
        )
    if generator_config.get("event_type_weights") != {
        "environment": 0.6,
        "structural": 0.3,
        "mixed": 0.1,
    }:
        raise ValueError("SOP05 generator event_type_weights differ from 60/30/10")
    raw_policy = _require_mapping(
        scientific.get("target_type_policy"),
        label="SOP05 scientific_request target_type_policy",
    )
    policy_copy = _canonical_json_copy(
        raw_policy, label="SOP05 scientific_request target_type_policy"
    )
    normalized_policy = normalize_target_type_policy(raw_policy)
    if policy_copy != normalized_policy.as_dict():
        raise ValueError("SOP05 scientific_request target_type_policy is not normalized")
    if policy_digest != normalized_policy.digest:
        raise ValueError("SOP05 scientific_request target_type_policy_digest mismatch")
    generator_policy = generator_config.get("target_type_policy")
    if (
        getattr(generator_policy, "as_dict", lambda: None)() != policy_copy
        or getattr(generator_policy, "digest", None) != policy_digest
    ):
        raise ValueError("SOP05 target_type_policy differs from generator snapshot")

    _validate_runtime(run_manifest.get("runtime"))
    identity_payload = {
        "version": SOP05_RUN_PRODUCER_VERSION,
        "producer_source_identity": run_manifest.get("producer_source_identity"),
        "split": split,
        "sop03": sop03_identity,
        "sop04": sop04_identity,
        "selection": dict(selection),
        "base_config_sha256": base_digest,
        "generator_config_sha256": generator_file_digest,
        "generator_config_semantic_digest": generator_semantic_digest,
        "target_type_policy": policy_copy,
        "target_type_policy_digest": policy_digest,
        "accepted_quota": scientific_ints["accepted_quota"],
        "events_per_pair": scientific_ints["events_per_pair"],
        "selection_version": SOP05_TOTAL_QUOTA_SELECTION_VERSION,
    }
    expected_run_id = "sop05-run-" + hashlib.blake2b(
        _canonical_json_bytes(identity_payload, label="SOP05 run identity"),
        digest_size=16,
    ).hexdigest()
    if run_manifest.get("run_id") != expected_run_id:
        raise ValueError("SOP05 run_id does not match the scientific identity")
    return _ValidatedRunContract(
        run_id=expected_run_id,
        split=split,
        schedule=schedule,
        selection_seed=scientific_ints["seed"],
        accepted_quota=scientific_ints["accepted_quota"],
        events_per_pair=scientific_ints["events_per_pair"],
        target_type_policy=policy_copy,
        target_type_policy_digest=policy_digest,
        generator_config_semantic_digest=generator_semantic_digest,
    )


def _validate_relative_file_name(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("checksum manifest contains an unsafe path")


def _publication_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("SOP05 publication must not contain symlinks")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"checksums.sha256", ".producer-complete"}:
            continue
        files[relative] = path
    return files


def _validate_publication_layout(root: Path) -> None:
    expected_root_entries = {
        ".producer-complete",
        "checksums.sha256",
        "configs",
        "generation_summary.json",
        "pair_generation_reports.jsonl",
        "run_manifest.json",
        "target_motions",
    }
    entries = {path.name for path in root.iterdir()}
    if entries != expected_root_entries:
        raise ValueError("SOP05 publication root layout mismatch")
    configs = root / "configs"
    if configs.is_symlink() or not configs.is_dir():
        raise ValueError("SOP05 publication configs layout is invalid")
    if {path.name for path in configs.iterdir()} != {
        "base.yaml",
        "generator.yaml",
    }:
        raise ValueError("SOP05 publication configs layout mismatch")


def _validate_checksum_manifest(root: Path) -> None:
    checksum_path = root / "checksums.sha256"
    try:
        raw_lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("invalid SOP05 checksum manifest") from exc
    if not raw_lines:
        raise ValueError("SOP05 checksum manifest must not be empty")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(raw_lines, start=1):
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError(
                f"invalid SOP05 checksum line {line_number}"
            )
        digest, relative = parts
        if len(digest) != 64 or any(
            value not in "0123456789abcdef" for value in digest
        ):
            raise ValueError("SOP05 checksum digest is invalid")
        _validate_relative_file_name(relative)
        if relative in entries:
            raise ValueError("SOP05 checksum manifest contains duplicate paths")
        entries[relative] = digest
    if list(entries) != sorted(entries):
        raise ValueError("SOP05 checksum manifest paths must be sorted")
    files = _publication_files(root)
    if set(entries) != set(files):
        raise ValueError("SOP05 checksum manifest file set mismatch")
    for relative, expected in entries.items():
        if _sha256(files[relative]) != expected:
            raise ValueError(f"SOP05 checksum mismatch for {relative}")


def _validate_bucket_map(
    value: object,
    *,
    label: str,
    expected_attempted: int,
    expected_accepted: int,
    expected_rejected: int,
    expected_rejection_reasons: Mapping[str, int],
) -> None:
    buckets = _require_mapping(value, label=label)
    if not buckets:
        raise ValueError(f"{label} must not be empty")
    attempted_total = 0
    accepted_total = 0
    rejected_total = 0
    rejection_totals: Counter[str] = Counter()
    for key, raw_bucket in buckets.items():
        if not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        bucket = _require_mapping(raw_bucket, label=f"{label}[{key!r}]")
        _require_exact_keys(bucket, _BUCKET_KEYS, label=f"{label}[{key!r}]")
        attempted = _nonnegative_int(
            bucket.get("attempted"), label=f"{label}[{key!r}] attempted"
        )
        accepted = _nonnegative_int(
            bucket.get("accepted"), label=f"{label}[{key!r}] accepted"
        )
        rejected = _nonnegative_int(
            bucket.get("rejected"), label=f"{label}[{key!r}] rejected"
        )
        if accepted + rejected != attempted:
            raise ValueError(f"{label}[{key!r}] count mismatch")
        rate = _require_rate(
            bucket.get("attempt_acceptance_rate"),
            label=f"{label}[{key!r}] attempt_acceptance_rate",
        )
        _require_rate_equal(
            rate,
            accepted / attempted if attempted else 0.0,
            label=f"{label}[{key!r}] attempt_acceptance_rate",
        )
        reasons = _require_count_map(
            bucket.get("rejection_reasons"),
            label=f"{label}[{key!r}] rejection_reasons",
        )
        if sum(reasons.values()) != rejected:
            raise ValueError(f"{label}[{key!r}] rejection reason total mismatch")
        attempted_total += attempted
        accepted_total += accepted
        rejected_total += rejected
        rejection_totals.update(reasons)
    if (
        attempted_total != expected_attempted
        or accepted_total != expected_accepted
        or rejected_total != expected_rejected
        or dict(rejection_totals) != dict(expected_rejection_reasons)
    ):
        raise ValueError(f"{label} aggregate mismatch")


def _validate_pair_summary(
    value: object,
    *,
    pair_seed: int,
    accepted_id_count: int,
    contract: _ValidatedRunContract,
) -> dict[str, object]:
    summary = _require_mapping(value, label="pair generation report summary")
    _require_exact_keys(
        summary, _PAIR_SUMMARY_KEYS, label="pair generation report summary"
    )
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("pair generation report summary schema_version mismatch")
    if summary.get("seed") != pair_seed:
        raise ValueError("pair generation report summary seed mismatch")
    requested = _nonnegative_int(
        summary.get("requested_event_count"),
        label="pair generation report requested_event_count",
    )
    if requested != contract.events_per_pair:
        raise ValueError("pair generation report requested_event_count mismatch")
    attempted = _nonnegative_int(
        summary.get("attempted_count"),
        label="pair generation report attempted_count",
    )
    for alias in (
        "complete_joint_candidates_attempted",
        "joint_candidate_attempted_count",
    ):
        if _nonnegative_int(
            summary.get(alias), label=f"pair generation report {alias}"
        ) != attempted:
            raise ValueError("pair generation report attempted_count aliases mismatch")
    _nonnegative_int(
        summary.get("attempt_index_start"),
        label="pair generation report attempt_index_start",
    )
    if summary.get("attempt_index_stop_exclusive") is not None:
        raise ValueError(
            "pair generation report attempt_index_stop_exclusive must be None"
        )
    accepted = _nonnegative_int(
        summary.get("accepted_count"),
        label="pair generation report accepted_count",
    )
    if accepted != accepted_id_count or accepted > requested:
        raise ValueError("pair generation report accepted_count mismatch")
    rejected = _nonnegative_int(
        summary.get("rejected_count"),
        label="pair generation report rejected_count",
    )
    if attempted < accepted or rejected != attempted - accepted:
        raise ValueError("pair generation report rejected_count mismatch")
    unaccepted = _nonnegative_int(
        summary.get("unaccepted_event_count"),
        label="pair generation report unaccepted_event_count",
    )
    if unaccepted != requested - accepted:
        raise ValueError("pair generation report unaccepted_event_count mismatch")
    expected_attempt_rate = accepted / attempted if attempted else 0.0
    expected_request_rate = accepted / requested
    for name, expected in (
        ("attempt_acceptance_rate", expected_attempt_rate),
        ("acceptance_rate", expected_request_rate),
        ("request_acceptance_rate", expected_request_rate),
    ):
        _require_rate_equal(
            _require_rate(summary.get(name), label=f"pair generation report {name}"),
            expected,
            label=f"pair generation report {name}",
        )
    rejection_reasons = _require_count_map(
        summary.get("rejection_reasons"),
        label="pair generation report rejection_reasons",
    )
    if sum(rejection_reasons.values()) != rejected:
        raise ValueError("pair generation report rejection reason total mismatch")
    rejection_stages = _require_count_map(
        summary.get("rejection_stage_counts"),
        label="pair generation report rejection_stage_counts",
        exact_keys=_REJECTION_STAGE_KEYS,
    )
    if sum(rejection_stages.values()) != rejected:
        raise ValueError("pair generation report rejection stage total mismatch")
    _require_count_map(
        summary.get("occluder_candidate_rejection_reasons"),
        label="pair generation report occluder_candidate_rejection_reasons",
    )
    expected_requested_counts = {
        "environment": requested * 6 // 10,
        "structural": requested * 3 // 10,
        "mixed": requested // 10,
    }
    requested_counts = _require_count_map(
        summary.get("requested_event_kind_counts"),
        label="pair generation report requested_event_kind_counts",
        exact_keys=frozenset(_EVENT_KINDS),
    )
    if requested_counts != expected_requested_counts:
        raise ValueError("pair generation report requested event kind counts mismatch")
    event_kind_counts = _require_count_map(
        summary.get("event_kind_counts"),
        label="pair generation report event_kind_counts",
        exact_keys=frozenset(_EVENT_KINDS),
    )
    if sum(event_kind_counts.values()) != accepted:
        raise ValueError("pair generation report event kind count total mismatch")

    by_event_kind = _require_mapping(
        summary.get("by_event_kind"), label="pair generation report by_event_kind"
    )
    _require_exact_keys(
        by_event_kind,
        frozenset(_EVENT_KINDS),
        label="pair generation report by_event_kind",
    )
    kind_attempted_total = 0
    kind_reasons: Counter[str] = Counter()
    kind_stages: Counter[str] = Counter()
    for kind in _EVENT_KINDS:
        bucket = _require_mapping(
            by_event_kind.get(kind),
            label=f"pair generation report by_event_kind[{kind!r}]",
        )
        _require_exact_keys(
            bucket,
            _EVENT_KIND_BUCKET_KEYS,
            label=f"pair generation report by_event_kind[{kind!r}]",
        )
        kind_requested = _nonnegative_int(
            bucket.get("requested"),
            label=f"pair generation report {kind} requested",
        )
        kind_attempted = _nonnegative_int(
            bucket.get("attempted"),
            label=f"pair generation report {kind} attempted",
        )
        kind_accepted = _nonnegative_int(
            bucket.get("accepted"),
            label=f"pair generation report {kind} accepted",
        )
        kind_rejected = _nonnegative_int(
            bucket.get("rejected"),
            label=f"pair generation report {kind} rejected",
        )
        if (
            kind_requested != requested_counts[kind]
            or kind_accepted != event_kind_counts[kind]
            or kind_accepted + kind_rejected != kind_attempted
        ):
            raise ValueError(f"pair generation report {kind} bucket count mismatch")
        for name, expected in (
            (
                "request_acceptance_rate",
                kind_accepted / kind_requested if kind_requested else 0.0,
            ),
            (
                "attempt_acceptance_rate",
                kind_accepted / kind_attempted if kind_attempted else 0.0,
            ),
        ):
            _require_rate_equal(
                _require_rate(
                    bucket.get(name),
                    label=f"pair generation report {kind} {name}",
                ),
                expected,
                label=f"pair generation report {kind} {name}",
            )
        reasons = _require_count_map(
            bucket.get("rejection_reasons"),
            label=f"pair generation report {kind} rejection_reasons",
        )
        stages = _require_count_map(
            bucket.get("rejection_stage_counts"),
            label=f"pair generation report {kind} rejection_stage_counts",
            exact_keys=_REJECTION_STAGE_KEYS,
        )
        if sum(reasons.values()) != kind_rejected or sum(stages.values()) != kind_rejected:
            raise ValueError(f"pair generation report {kind} rejection total mismatch")
        kind_attempted_total += kind_attempted
        kind_reasons.update(reasons)
        kind_stages.update(stages)
    if (
        kind_attempted_total != attempted
        or dict(kind_reasons) != rejection_reasons
        or {key: kind_stages[key] for key in _REJECTION_STAGE_KEYS}
        != rejection_stages
    ):
        raise ValueError("pair generation report event-kind aggregates mismatch")

    for name in ("by_object_type", "by_footprint_kind", "by_geometry_source"):
        _validate_bucket_map(
            summary.get(name),
            label=f"pair generation report {name}",
            expected_attempted=attempted,
            expected_accepted=accepted,
            expected_rejected=rejected,
            expected_rejection_reasons=rejection_reasons,
        )
    if summary.get("target_type_policy") != contract.target_type_policy:
        raise ValueError("pair generation report target_type_policy mismatch")
    if summary.get("target_type_policy_digest") != contract.target_type_policy_digest:
        raise ValueError("pair generation report target_type_policy_digest mismatch")
    if summary.get("generator_config_digest") != (
        contract.generator_config_semantic_digest
    ):
        raise ValueError("pair generation report generator_config_digest mismatch")
    algorithm_version = _require_nonempty_string(
        summary.get("generator_algorithm_version"),
        label="pair generation report generator_algorithm_version",
    )
    if algorithm_version != SOP05_GENERATOR_ALGORITHM_VERSION:
        raise ValueError(
            "pair generation report generator_algorithm_version does not "
            f"match the frozen {SOP05_GENERATOR_ALGORITHM_VERSION!r} contract"
        )
    return {
        "requested": requested,
        "attempted": attempted,
        "event_kind_counts": event_kind_counts,
        "rejection_reasons": rejection_reasons,
        "rejection_stage_counts": rejection_stages,
        "invariants": {
            "schema_version": SCHEMA_VERSION,
            "target_type_policy_digest": contract.target_type_policy_digest,
            "generator_config_digest": contract.generator_config_semantic_digest,
            "generator_algorithm_version": algorithm_version,
        },
    }


def _load_pair_reports(
    path: Path, *, contract: _ValidatedRunContract
) -> _ValidatedPairReports:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("invalid pair generation reports") from exc
    if not lines or len(lines) != len(contract.schedule):
        raise ValueError("pair generation reports must not be empty")
    event_ids: list[str] = []
    accepted_events: list[tuple[str, str]] = []
    event_pair_identity: dict[str, tuple[str, str]] = {}
    event_kind_by_id: dict[str, str] = {}
    requested_total = 0
    attempted_total = 0
    kind_totals: Counter[str] = Counter()
    rejection_totals: Counter[str] = Counter()
    rejection_stage_totals: Counter[str] = Counter()
    generator_invariants: dict[str, object] | None = None
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON value {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"invalid pair generation report row {line_number}"
            ) from exc
        mapping = _require_mapping(
            row, label=f"pair generation report row {line_number}"
        )
        _require_exact_keys(
            mapping,
            _PAIR_REPORT_KEYS,
            label=f"pair generation report row {line_number}",
        )
        canonical_line = _canonical_json_bytes(
            mapping, label=f"pair generation report row {line_number}"
        ).decode("utf-8")
        if line != canonical_line:
            raise ValueError("pair generation report rows must be canonical JSON")
        if mapping.get("report_version") != SOP05_PAIR_REPORT_VERSION:
            raise ValueError("unsupported pair generation report version")
        if (
            mapping.get("selection_version")
            != SOP05_TOTAL_QUOTA_SELECTION_VERSION
        ):
            raise ValueError("unsupported SOP05 selection version")
        scheduled = contract.schedule[line_number - 1]
        if (
            mapping.get("rank") != scheduled["rank"]
            or mapping.get("state_id") != scheduled["state_id"]
            or mapping.get("trajectory_id") != scheduled["trajectory_id"]
            or mapping.get("seed") != scheduled["pair_seed"]
        ):
            raise ValueError("pair generation report order/schedule mismatch")
        raw_accepted_events = mapping.get("accepted_events")
        if not isinstance(raw_accepted_events, list):
            raise ValueError("pair report accepted_events must be a list")
        ids: list[str] = []
        event_kinds: list[str] = []
        for event_index, raw_event in enumerate(raw_accepted_events):
            accepted_event = _require_mapping(
                raw_event,
                label=(
                    "pair generation report accepted event "
                    f"{line_number}:{event_index}"
                ),
            )
            _require_exact_keys(
                accepted_event,
                _ACCEPTED_EVENT_KEYS,
                label="pair generation report accepted event",
            )
            event_id = _require_nonempty_string(
                accepted_event.get("generated_event_id"),
                label="pair generation report accepted event ID",
            )
            event_kind = _require_nonempty_string(
                accepted_event.get("event_kind"),
                label="pair generation report accepted event kind",
            )
            if event_kind not in _EVENT_KINDS:
                raise ValueError(
                    "pair generation report accepted event kind is unsupported"
                )
            ids.append(event_id)
            event_kinds.append(event_kind)
        if len(ids) != len(set(ids)):
            raise ValueError("pair generation report contains duplicate event IDs")
        validated_summary = _validate_pair_summary(
            mapping.get("summary"),
            pair_seed=scheduled["pair_seed"],
            accepted_id_count=len(ids),
            contract=contract,
        )
        if Counter(event_kinds) != Counter(
            validated_summary["event_kind_counts"]
        ):
            raise ValueError(
                "pair report accepted event kinds differ from summary"
            )
        requested_total += validated_summary["requested"]
        attempted_total += validated_summary["attempted"]
        kind_totals.update(validated_summary["event_kind_counts"])
        rejection_totals.update(validated_summary["rejection_reasons"])
        rejection_stage_totals.update(validated_summary["rejection_stage_counts"])
        invariants = validated_summary["invariants"]
        if generator_invariants is None:
            generator_invariants = dict(invariants)
        elif generator_invariants != invariants:
            raise ValueError("pair generation report generator invariants differ")
        for event_id, event_kind in zip(ids, event_kinds, strict=True):
            event_pair_identity[event_id] = (
                scheduled["state_id"],
                scheduled["trajectory_id"],
            )
            event_kind_by_id[event_id] = event_kind
            accepted_events.append((event_id, event_kind))
        event_ids.extend(ids)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("pair generation reports contain duplicate event IDs")
    if generator_invariants is None:
        raise RuntimeError("validated non-empty pair reports lost invariants")
    selected_event_ids = select_sop05_event_ids(
        accepted_events,
        seed=contract.selection_seed,
        accepted_quota=contract.accepted_quota,
    )
    return _ValidatedPairReports(
        generated_event_ids=frozenset(event_ids),
        event_pair_identity=MappingProxyType(event_pair_identity),
        event_kind_by_id=MappingProxyType(event_kind_by_id),
        selected_event_ids=selected_event_ids,
        requested_event_count=requested_total,
        attempted_count=attempted_total,
        generated_event_kind_counts={kind: kind_totals[kind] for kind in _EVENT_KINDS},
        rejection_reasons=dict(sorted(rejection_totals.items())),
        rejection_stage_counts={
            kind: rejection_stage_totals[kind]
            for kind in sorted(_REJECTION_STAGE_KEYS)
        },
        generator_invariants=generator_invariants,
    )


def _validate_generation_summary(
    value: object,
    *,
    contract: _ValidatedRunContract,
    reports: _ValidatedPairReports,
    events: tuple[GeneratedEvent, ...],
) -> dict[str, object]:
    summary = _require_mapping(value, label="SOP05 generation summary")
    _require_exact_keys(
        summary, _GLOBAL_SUMMARY_KEYS, label="SOP05 generation summary"
    )
    if (
        summary.get("summary_version") != SOP05_GENERATION_SUMMARY_VERSION
        or summary.get("run_id") != contract.run_id
        or summary.get("run_state") != "complete"
        or summary.get("quota_met") is not True
        or len(events) != contract.accepted_quota
        or len(reports.generated_event_ids) < contract.accepted_quota
    ):
        raise ValueError("SOP05 generation summary completion mismatch")
    scalar_expectations = {
        "processed_pair_count": len(contract.schedule),
        "requested_event_count": reports.requested_event_count,
        "attempted_count": reports.attempted_count,
        "generator_accepted_count": len(reports.generated_event_ids),
        "selected_count": len(events),
        "quota_trimmed_count": len(reports.generated_event_ids) - len(events),
    }
    for name, expected in scalar_expectations.items():
        observed = _nonnegative_int(
            summary.get(name), label=f"SOP05 generation summary {name}"
        )
        if observed != expected:
            raise ValueError(f"SOP05 generation summary {name} mismatch")
    generated_counts = _require_count_map(
        summary.get("generated_event_kind_counts"),
        label="SOP05 generation summary generated_event_kind_counts",
        exact_keys=frozenset(_EVENT_KINDS),
    )
    selected_counts = _require_count_map(
        summary.get("selected_event_kind_counts"),
        label="SOP05 generation summary selected_event_kind_counts",
        exact_keys=frozenset(_EVENT_KINDS),
    )
    observed_selected = Counter(event.event_kind for event in events)
    expected_selected = {
        kind: observed_selected[kind]
        for kind in _EVENT_KINDS
    }
    if (
        generated_counts != reports.generated_event_kind_counts
        or selected_counts != expected_selected
    ):
        raise ValueError("SOP05 generation summary event-kind counts mismatch")
    rejection_reasons = _require_count_map(
        summary.get("rejection_reasons"),
        label="SOP05 generation summary rejection_reasons",
    )
    rejection_stages = _require_count_map(
        summary.get("rejection_stage_counts"),
        label="SOP05 generation summary rejection_stage_counts",
        exact_keys=_REJECTION_STAGE_KEYS,
    )
    if (
        rejection_reasons != reports.rejection_reasons
        or rejection_stages != reports.rejection_stage_counts
    ):
        raise ValueError("SOP05 generation summary rejection aggregates mismatch")
    invariants = _require_mapping(
        summary.get("generator_invariants"),
        label="SOP05 generation summary generator_invariants",
    )
    _require_exact_keys(
        invariants,
        _GENERATOR_INVARIANT_KEYS,
        label="SOP05 generation summary generator_invariants",
    )
    if dict(invariants) != reports.generator_invariants:
        raise ValueError("SOP05 generation summary generator invariants mismatch")
    copy = _canonical_json_copy(summary, label="SOP05 generation summary")
    if not isinstance(copy, dict):
        raise RuntimeError("validated SOP05 generation summary changed type")
    return copy


def load_complete_sop05_events(
    root: str | Path,
    *,
    grid: GridSpec,
    expected_publication_semantic_digest: str,
    expected_run_id: str | None = None,
) -> LoadedSop05Events:
    """Verify one complete SOP-05 publication against an external trust anchor.

    ``expected_publication_semantic_digest`` must come from the producer result
    or another retained handoff record, never from the directory being loaded.
    """

    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    trusted_publication_digest = _require_hex_digest(
        expected_publication_semantic_digest,
        size=64,
        label="expected_publication_semantic_digest",
    )
    root_path = Path(root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise ValueError("SOP05 publication root must be a real directory")
    marker_path = root_path / ".producer-complete"
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError("SOP05 publication is missing .producer-complete")

    run_manifest_raw = _load_json(
        root_path / "run_manifest.json", label="SOP05 run manifest"
    )
    run_manifest = _require_mapping(
        run_manifest_raw, label="SOP05 run manifest"
    )
    _require_exact_keys(
        run_manifest, _RUN_MANIFEST_KEYS, label="SOP05 run manifest"
    )
    if run_manifest.get("manifest_version") != SOP05_RUN_MANIFEST_VERSION:
        raise ValueError("unsupported SOP05 run manifest version")
    if run_manifest.get("producer_version") != SOP05_RUN_PRODUCER_VERSION:
        raise ValueError("unsupported SOP05 producer version")
    _validate_source_identity(run_manifest.get("producer_source_identity"))
    run_id = _require_nonempty_string(
        run_manifest.get("run_id"), label="SOP05 run_id"
    )
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError("SOP05 run_id differs from expected_run_id")
    if run_manifest.get("run_state") != "complete":
        raise ValueError("SOP05 publication run_state is not complete")
    artifacts = _require_mapping(
        run_manifest.get("artifacts"), label="SOP05 artifacts"
    )
    _require_exact_keys(artifacts, _ARTIFACT_KEYS, label="SOP05 artifacts")
    if dict(artifacts) != _EXPECTED_ARTIFACTS:
        raise ValueError("SOP05 artifact layout differs from frozen contract")

    marker_raw = _load_json(marker_path, label="SOP05 producer-complete marker")
    marker = _require_mapping(
        marker_raw, label="SOP05 producer-complete marker"
    )
    _require_exact_keys(
        marker,
        _COMPLETION_MARKER_KEYS,
        label="SOP05 producer-complete marker",
    )
    if marker.get("marker_version") != SOP05_COMPLETION_MARKER_VERSION:
        raise ValueError("unsupported SOP05 producer-complete marker version")
    if marker.get("publication_identity_version") != (
        SOP05_PUBLICATION_IDENTITY_VERSION
    ):
        raise ValueError("unsupported SOP05 publication identity version")
    if marker.get("run_id") != run_id:
        raise ValueError("SOP05 producer-complete run_id mismatch")
    if marker.get("run_manifest_sha256") != _sha256(
        root_path / "run_manifest.json"
    ):
        raise ValueError("SOP05 producer-complete run manifest digest mismatch")
    if marker.get("checksums_sha256") != _sha256(
        root_path / "checksums.sha256"
    ):
        raise ValueError("SOP05 producer-complete checksum digest mismatch")
    _validate_publication_layout(root_path)
    _validate_checksum_manifest(root_path)
    run_contract = _validate_run_contract(root_path, run_manifest, grid=grid)

    shard = load_event_target_motion_shard(
        root_path / "target_motions", grid=grid
    )
    if marker.get("target_motion_manifest_digest") != shard.manifest_digest:
        raise ValueError("SOP05 target-motion manifest digest mismatch")
    if marker.get("target_motion_payload_semantic_digest") != (
        shard.payload_semantic_digest
    ):
        raise ValueError("SOP05 target-motion semantic digest mismatch")
    computed_publication_digest = compute_sop05_publication_semantic_digest(
        run_id=run_id,
        run_manifest_sha256=_sha256(root_path / "run_manifest.json"),
        checksums_sha256=_sha256(root_path / "checksums.sha256"),
        target_motion_manifest_digest=shard.manifest_digest,
        target_motion_payload_semantic_digest=shard.payload_semantic_digest,
    )
    stored_publication_digest = _require_hex_digest(
        marker.get("publication_semantic_digest"),
        size=64,
        label="SOP05 publication semantic digest",
    )
    if stored_publication_digest != computed_publication_digest:
        raise ValueError("SOP05 publication semantic digest mismatch")
    if trusted_publication_digest != computed_publication_digest:
        raise ValueError(
            "SOP05 publication semantic digest differs from trusted handoff"
        )
    events = tuple(
        restore_generated_event(record, shard.worlds[record.world_id], grid=grid)
        for record in shard.records
    )
    event_ids = [event.generated_event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("SOP05 restored generated_event_id values are not unique")
    for event in events:
        if event.target_motion_record.target_type_policy_digest != (
            run_contract.target_type_policy_digest
        ):
            raise ValueError("SOP05 event target_type_policy_digest differs from run")
        metadata = _require_mapping(
            event.world.metadata, label="SOP05 event world metadata"
        )
        if metadata.get("generator_config_digest") != (
            run_contract.generator_config_semantic_digest
        ):
            raise ValueError("SOP05 event generator_config_digest differs from run")

    reports = _load_pair_reports(
        root_path / "pair_generation_reports.jsonl", contract=run_contract
    )
    # The shard contract canonicalizes records lexicographically by event ID
    # after global total-quota selection; require that exact persisted sequence.
    expected_shard_event_ids = tuple(sorted(reports.selected_event_ids))
    if tuple(event_ids) != expected_shard_event_ids:
        raise ValueError(
            "selected SOP05 shard order differs from frozen selection"
        )
    for event in events:
        expected_pair = reports.event_pair_identity.get(event.generated_event_id)
        observed_pair = (
            event.target_motion_record.base_state_id,
            event.target_motion_record.trajectory_id,
        )
        if expected_pair is None or expected_pair != observed_pair:
            raise ValueError(
                "selected SOP05 event pair identity differs from pair reports"
            )
        if reports.event_kind_by_id.get(event.generated_event_id) != (
            event.event_kind
        ):
            raise ValueError(
                "selected SOP05 event kind differs from pair reports"
            )
        if event.world.metadata.get("generator_algorithm_version") != (
            reports.generator_invariants["generator_algorithm_version"]
        ):
            raise ValueError(
                "SOP05 event generator_algorithm_version differs from pair reports"
            )

    summary_raw = _load_json(
        root_path / "generation_summary.json",
        label="SOP05 generation summary",
    )
    summary_copy = _validate_generation_summary(
        summary_raw,
        contract=run_contract,
        reports=reports,
        events=events,
    )

    manifest_copy = _canonical_json_copy(
        run_manifest, label="SOP05 run manifest"
    )
    if not isinstance(manifest_copy, dict) or not isinstance(summary_copy, dict):
        raise RuntimeError("validated SOP05 JSON unexpectedly changed type")
    by_id = MappingProxyType(
        {event.generated_event_id: event for event in events}
    )
    return LoadedSop05Events(
        run_id=run_id,
        publication_semantic_digest=computed_publication_digest,
        split=run_contract.split,
        events=events,
        events_by_id=by_id,
        shard=shard,
        run_manifest=manifest_copy,
        generation_summary=summary_copy,
    )
