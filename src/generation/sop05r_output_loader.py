"""Strict loader for independently published SOP05R event collections."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from src.contracts import (
    GridSpec,
    build_grid_spec,
    load_dataclass,
    validate_oracle_world,
)
from src.datasets.split_manager import SPLIT_NAMES
from src.generation.dynamic_object_transplant import TransplantedDynamicObject
from src.generation.event_sampler import GeneratedEvent
from src.generation.event_target_motion_shard import (
    EventTargetMotionRecord,
    LoadedEventTargetMotionShard,
    compute_motion_array_digest,
    compute_oracle_world_semantic_digest,
    load_event_target_motion_shard,
    validate_event_target_motion_world_join,
)
from src.generation.history_visibility import (
    HISTORY_VISIBILITY_REGIMES,
    classify_sop05r_history,
)
from src.planning.verification_actions import (
    ACTION_LIBRARY_VERSION,
    CANONICAL_ACTION_IDS,
)
from src.utils.config import validate_config

from .sop05r_contracts import (
    SOP05R_ACTIVE_REVEALABILITY_VERSION,
    SOP05R_COMPLETION_MARKER_VERSION,
    SOP05R_GENERATOR_VERSION,
    SOP05R_MANIFEST_VERSION,
    SOP05R_REPORT_VERSION,
    SOP05R_RUN_VERSION,
    SOP05R_SELECTION_VERSION,
    SOP05R_SUMMARY_VERSION,
    Sop05rConfig,
    normalize_sop05r_config,
)
from .sop05r_run import (
    SOP05R_EMPTY_TARGET_MOTION_VERSION,
    SOP05R_EMPTY_TRAJECTORY_STORE_VERSION,
    SOP05R_INPUT_LOCK_VERSION,
    Sop05rPublicationContext,
    Sop05rScheduleEntry,
    Sop05rSelectionCandidate,
    _publication_digest,
    select_sop05r_event_ids,
)
from .sop05r_trajectory_store import (
    Sop05rTrajectoryStore,
    load_sop05r_trajectory_store,
)


_ROOT_ENTRIES = {
    "manifest.json",
    "generation_summary.json",
    "template_reports.jsonl",
    "events.jsonl",
    "worlds",
    "target_motion",
    "planner_trajectories",
    "artifact_checksums.sha256",
}
_MANIFEST_KEYS = {
    "schema_version",
    "manifest_version",
    "producer_version",
    "generator_algorithm_version",
    "selection_version",
    "report_version",
    "run_id",
    "run_state",
    "split",
    "seed",
    "accepted_quota",
    "input_lock",
    "producer_source_identity",
    "base_config",
    "sop05r_config",
    "verification_action_config",
    "schedule",
    "selected_event_ids",
    "publication_semantic_digest",
    "artifacts",
}
_ARTIFACT_KEYS = {
    "generation_summary",
    "template_reports",
    "events",
    "worlds",
    "target_motion",
    "planner_trajectories",
    "checksums",
    "completion_marker",
    "trajectory_collection_semantic_digest",
    "target_motion_manifest_digest",
    "target_motion_payload_semantic_digest",
}
_EVENT_ROW_KEYS = {
    "generated_event_id",
    "event_kind",
    "world_id",
    "base_state_id",
    "trajectory_id",
    "template_id",
    "target_motion_record_digest",
    "visibility_sequence",
    "target_visibility_history",
    "conflict_time_s",
    "conflict_index",
    "history_visibility_regime",
    "active_revealable",
    "active_revealable_action_ids",
}
_REPORT_KEYS = {
    "report_version",
    "base_rank",
    "state_id",
    "template_id",
    "template_schedule_rank",
    "attempt_index",
    "geometry_eligible",
    "planner_feasible",
    "exact_history_qualified",
    "time_aligned_collision",
    "active_revealable",
    "generated_event_id",
    "history_visibility_regime",
    "rejection_reason",
}
_MARKER_KEYS = {
    "marker_version",
    "run_id",
    "publication_semantic_digest",
    "manifest_sha256",
    "artifact_checksums_sha256",
    "trajectory_collection_semantic_digest",
    "target_motion_manifest_digest",
    "target_motion_payload_semantic_digest",
}
_SOURCE_IDENTITY_KEYS = {
    "version",
    "git_commit",
    "worktree_state",
    "dirty_tree_sha256",
}
_ACTION_TOP_LEVEL_KEYS = {
    "schema_version",
    "library_version",
    "sensor_fov_deg",
    "actions",
}


@dataclass(frozen=True)
class LoadedSop05rEvents:
    events: tuple[GeneratedEvent, ...]
    target_motion: LoadedEventTargetMotionShard
    trajectory_store: Sop05rTrajectoryStore
    template_reports: tuple[Mapping[str, object], ...]
    event_rows: tuple[Mapping[str, object], ...]
    manifest: Mapping[str, object]
    summary: Mapping[str, object]
    publication_semantic_digest: str
    complete: bool


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON must not contain {value}")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("SOP05R artifact is not canonical JSON") from exc


def _json_file_bytes(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"failed to hash SOP05R artifact: {path}") from exc


def _hex_digest(value: object, *, name: str, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return value


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("ascii"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid SOP05R {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"SOP05R {label} must be a mapping")
    if payload != _json_file_bytes(value):
        raise ValueError(f"SOP05R {label} is not canonical")
    return value


def _load_jsonl(
    path: Path,
    *,
    label: str,
    allow_empty: bool = False,
) -> list[dict[str, object]]:
    try:
        payload = path.read_bytes()
        lines = payload.splitlines(keepends=True)
    except OSError as exc:
        raise ValueError(f"failed to read SOP05R {label}") from exc
    if not lines and allow_empty:
        return []
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise ValueError(f"SOP05R {label} must be nonempty newline-terminated JSONL")
    rows = []
    for line in lines:
        try:
            row = json.loads(
                line.decode("ascii"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid SOP05R {label} row") from exc
        if not isinstance(row, dict) or line != _canonical_json_bytes(row) + b"\n":
            raise ValueError(f"SOP05R {label} row is not canonical")
        rows.append(row)
    return rows


def _checksum_manifest_bytes(root: Path) -> bytes:
    rows = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"SOP05R artifact must not be a symlink: {relative}")
        if not path.is_file() or relative in {
            "artifact_checksums.sha256",
            ".sop05r-complete",
        }:
            continue
        rows.append(f"{_sha256_file(path)}  {relative}\n")
    return "".join(sorted(rows)).encode("ascii")


def _validate_root(root: Path, *, require_complete: bool) -> bool:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("SOP05R publication root must be a real directory")
    observed = {path.name for path in root.iterdir()}
    complete = ".sop05r-complete" in observed
    expected = set(_ROOT_ENTRIES)
    if complete:
        expected.add(".sop05r-complete")
    if observed != expected:
        raise ValueError("SOP05R publication root entries mismatch")
    if require_complete and not complete:
        raise ValueError("SOP05R completion marker is required")
    for directory in ("worlds", "target_motion", "planner_trajectories"):
        path = root / directory
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"SOP05R {directory} must be a real directory")
    expected_checksums = _checksum_manifest_bytes(root)
    try:
        observed_checksums = (root / "artifact_checksums.sha256").read_bytes()
    except OSError as exc:
        raise ValueError("SOP05R checksum manifest is missing") from exc
    if observed_checksums != expected_checksums:
        raise ValueError("SOP05R checksum manifest mismatch")
    return complete


def _validate_source_identity(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _SOURCE_IDENTITY_KEYS:
        raise ValueError("SOP05R producer source identity keys mismatch")
    _nonempty_text(value["version"], name="producer source identity version")
    _hex_digest(value["git_commit"], name="producer git commit", length=40)
    if value["worktree_state"] not in {"clean", "dirty"}:
        raise ValueError("SOP05R producer worktree state is invalid")
    dirty = value["dirty_tree_sha256"]
    if value["worktree_state"] == "clean":
        if dirty is not None:
            raise ValueError("clean source identity must not have a dirty digest")
    else:
        _hex_digest(dirty, name="producer dirty tree digest")
    return value


def _validate_action_config(value: object, *, schema_version: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _ACTION_TOP_LEVEL_KEYS:
        raise ValueError("SOP05R verification action config keys mismatch")
    if value["schema_version"] != schema_version:
        raise ValueError("SOP05R verification action schema mismatch")
    if value["library_version"] != ACTION_LIBRARY_VERSION:
        raise ValueError("SOP05R verification action version mismatch")
    if not np.isclose(
        _finite_real(value["sensor_fov_deg"], name="sensor_fov_deg"),
        360.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("SOP05R verification sensor FOV must equal 360 degrees")
    actions = value["actions"]
    if not isinstance(actions, list) or tuple(
        row.get("action_id") if isinstance(row, dict) else None for row in actions
    ) != CANONICAL_ACTION_IDS:
        raise ValueError("SOP05R verification action order mismatch")
    expected_keys = {"action_id", "duration_s", "delta_forward_m", "delta_yaw_deg"}
    for row in actions:
        if set(row) != expected_keys:
            raise ValueError("SOP05R verification action row keys mismatch")
        if _finite_real(row["duration_s"], name="action duration") <= 0.0:
            raise ValueError("SOP05R verification action duration must be positive")
        _finite_real(row["delta_forward_m"], name="action forward delta")
        _finite_real(row["delta_yaw_deg"], name="action yaw delta")
    return value


def _load_manifest_contract(
    root: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    Sop05rConfig,
    tuple[Sop05rScheduleEntry, ...],
]:
    manifest = _load_json(root / "manifest.json", label="manifest")
    if set(manifest) != _MANIFEST_KEYS:
        raise ValueError("SOP05R manifest keys mismatch")
    expected_versions = {
        "manifest_version": SOP05R_MANIFEST_VERSION,
        "producer_version": SOP05R_RUN_VERSION,
        "generator_algorithm_version": SOP05R_GENERATOR_VERSION,
        "selection_version": SOP05R_SELECTION_VERSION,
        "report_version": SOP05R_REPORT_VERSION,
    }
    for name, expected in expected_versions.items():
        if manifest[name] != expected:
            raise ValueError(f"SOP05R {name} mismatch")
    run_id = _nonempty_text(manifest["run_id"], name="run_id")
    if manifest["run_state"] not in {"complete", "quota_unmet"}:
        raise ValueError("SOP05R run_state is invalid")
    if manifest["split"] not in SPLIT_NAMES:
        raise ValueError("SOP05R split is invalid")
    _nonnegative_int(manifest["seed"], name="seed")
    _positive_int(manifest["accepted_quota"], name="accepted_quota")
    source_identity = _validate_source_identity(manifest["producer_source_identity"])
    base_config = manifest["base_config"]
    if not isinstance(base_config, dict):
        raise ValueError("SOP05R base_config must be a mapping")
    validate_config(base_config)
    raw_config = manifest["sop05r_config"]
    if not isinstance(raw_config, dict) or "digest" not in raw_config:
        raise ValueError("SOP05R normalized config is invalid")
    config_digest = _hex_digest(raw_config["digest"], name="SOP05R config digest")
    config_payload = {key: nested for key, nested in raw_config.items() if key != "digest"}
    config = normalize_sop05r_config(config_payload)
    if config.digest != config_digest:
        raise ValueError("SOP05R config semantic digest mismatch")
    action_config = _validate_action_config(
        manifest["verification_action_config"],
        schema_version=config.schema_version,
    )
    schedule_rows = manifest["schedule"]
    if not isinstance(schedule_rows, list) or not schedule_rows:
        raise ValueError("SOP05R schedule must be nonempty")
    schedule = []
    for expected_rank, row in enumerate(schedule_rows):
        if not isinstance(row, dict) or set(row) != {"rank", "state_id", "base_seed"}:
            raise ValueError("SOP05R schedule row keys mismatch")
        if row["rank"] != expected_rank:
            raise ValueError("SOP05R schedule ranks are not contiguous")
        schedule.append(
            Sop05rScheduleEntry(
                rank=expected_rank,
                state_id=_nonempty_text(row["state_id"], name="schedule state_id"),
                base_seed=_nonnegative_int(row["base_seed"], name="base_seed"),
            )
        )
    if len({row.state_id for row in schedule}) != len(schedule):
        raise ValueError("SOP05R schedule state IDs are not unique")
    input_lock = manifest["input_lock"]
    if not isinstance(input_lock, dict):
        raise ValueError("SOP05R input_lock must be a mapping")
    required_input_keys = {
        "version",
        "split",
        "sop03",
        "sop04_trajectory_bank_is_input",
        "base_config_sha256",
        "sop05r_config_digest",
        "verification_action_config_sha256",
        "schedule_sha256",
        "versions",
    }
    if set(input_lock) != required_input_keys:
        raise ValueError("SOP05R input_lock keys mismatch")
    if (
        input_lock["version"] != SOP05R_INPUT_LOCK_VERSION
        or input_lock["split"] != manifest["split"]
        or input_lock["sop04_trajectory_bank_is_input"] is not False
    ):
        raise ValueError("SOP05R input_lock identity mismatch")
    sop03 = input_lock["sop03"]
    if not isinstance(sop03, dict) or set(sop03) != {
        "code_commit",
        "checksum_manifest_sha256",
        "audit_sha256",
        "completion_policy",
    }:
        raise ValueError("SOP05R SOP03 evidence keys mismatch")
    _hex_digest(sop03["code_commit"], name="SOP03 code commit", length=40)
    _hex_digest(
        sop03["checksum_manifest_sha256"], name="SOP03 checksum manifest"
    )
    _hex_digest(sop03["audit_sha256"], name="SOP03 audit digest")
    _nonempty_text(sop03["completion_policy"], name="SOP03 completion policy")
    expected_lock = {
        "base_config_sha256": _sha256_bytes(_canonical_json_bytes(base_config)),
        "sop05r_config_digest": config.digest,
        "verification_action_config_sha256": _sha256_bytes(
            _canonical_json_bytes(action_config)
        ),
        "schedule_sha256": _sha256_bytes(
            _canonical_json_bytes([row.as_dict() for row in schedule])
        ),
    }
    for name, expected in expected_lock.items():
        if input_lock[name] != expected:
            raise ValueError(f"SOP05R input_lock {name} mismatch")
    versions = input_lock["versions"]
    if versions != {
        "generator_algorithm_version": SOP05R_GENERATOR_VERSION,
        "run_producer_version": SOP05R_RUN_VERSION,
        "selection_version": SOP05R_SELECTION_VERSION,
        "report_version": SOP05R_REPORT_VERSION,
    }:
        raise ValueError("SOP05R input_lock versions mismatch")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != _ARTIFACT_KEYS:
        raise ValueError("SOP05R manifest artifact keys mismatch")
    expected_paths = {
        "generation_summary": "generation_summary.json",
        "template_reports": "template_reports.jsonl",
        "events": "events.jsonl",
        "worlds": "worlds",
        "target_motion": "target_motion",
        "planner_trajectories": "planner_trajectories",
        "checksums": "artifact_checksums.sha256",
        "completion_marker": ".sop05r-complete",
    }
    for name, expected in expected_paths.items():
        if artifacts[name] != expected:
            raise ValueError(f"SOP05R artifact path {name} mismatch")
    selected_ids = manifest["selected_event_ids"]
    if (
        not isinstance(selected_ids, list)
        or len(selected_ids) != len(set(selected_ids))
        or any(not isinstance(value, str) or not value for value in selected_ids)
    ):
        raise ValueError("SOP05R selected event IDs are invalid")
    if not selected_ids and manifest["run_state"] != "quota_unmet":
        raise ValueError("complete SOP05R run must select at least one event")
    _hex_digest(
        manifest["publication_semantic_digest"],
        name="publication semantic digest",
    )
    return manifest, source_identity, config, tuple(schedule)


def _validate_marker(
    root: Path,
    manifest: Mapping[str, object],
    *,
    complete: bool,
) -> None:
    if manifest["run_state"] == "complete" and not complete:
        raise ValueError("complete SOP05R run is missing its completion marker")
    if manifest["run_state"] == "quota_unmet" and complete:
        raise ValueError("quota-unmet SOP05R run must not have a completion marker")
    if not complete:
        return
    marker = _load_json(root / ".sop05r-complete", label="completion marker")
    if set(marker) != _MARKER_KEYS:
        raise ValueError("SOP05R completion marker keys mismatch")
    artifacts = manifest["artifacts"]
    expected = {
        "marker_version": SOP05R_COMPLETION_MARKER_VERSION,
        "run_id": manifest["run_id"],
        "publication_semantic_digest": manifest["publication_semantic_digest"],
        "manifest_sha256": _sha256_file(root / "manifest.json"),
        "artifact_checksums_sha256": _sha256_file(
            root / "artifact_checksums.sha256"
        ),
        "trajectory_collection_semantic_digest": artifacts[
            "trajectory_collection_semantic_digest"
        ],
        "target_motion_manifest_digest": artifacts[
            "target_motion_manifest_digest"
        ],
        "target_motion_payload_semantic_digest": artifacts[
            "target_motion_payload_semantic_digest"
        ],
    }
    if marker != expected:
        raise ValueError("SOP05R completion marker evidence mismatch")


def _validate_reports(rows: list[dict[str, object]]) -> None:
    previous = None
    event_ids = []
    for row in rows:
        if set(row) != _REPORT_KEYS:
            raise ValueError("SOP05R template report keys mismatch")
        if row["report_version"] != SOP05R_REPORT_VERSION:
            raise ValueError("SOP05R template report version mismatch")
        base_rank = _nonnegative_int(row["base_rank"], name="report base_rank")
        attempt = _nonnegative_int(row["attempt_index"], name="report attempt_index")
        state_id = _nonempty_text(row["state_id"], name="report state_id")
        template_id = _nonempty_text(row["template_id"], name="report template_id")
        schedule_rank = row["template_schedule_rank"]
        if (
            not isinstance(schedule_rank, list)
            or not schedule_rank
            or any(type(value) is not int or value < 0 for value in schedule_rank)
        ):
            raise ValueError("SOP05R template schedule rank is invalid")
        key = (base_rank, attempt, template_id)
        if previous is not None and key <= previous:
            raise ValueError("SOP05R template reports are not strictly ordered")
        previous = key
        for name in (
            "geometry_eligible",
            "planner_feasible",
            "exact_history_qualified",
            "time_aligned_collision",
            "active_revealable",
        ):
            if type(row[name]) is not bool:
                raise ValueError(f"SOP05R template report {name} must be boolean")
        event_id = row["generated_event_id"]
        regime = row["history_visibility_regime"]
        reason = row["rejection_reason"]
        if event_id is None:
            if regime is not None or reason is None:
                raise ValueError("rejected SOP05R report identity is inconsistent")
            _nonempty_text(reason, name="report rejection_reason")
            if row["exact_history_qualified"] or row["time_aligned_collision"]:
                raise ValueError("rejected SOP05R report has accepted-stage flags")
        else:
            _nonempty_text(event_id, name="report generated_event_id")
            if regime not in HISTORY_VISIBILITY_REGIMES or reason is not None:
                raise ValueError("accepted SOP05R report evidence is inconsistent")
            if not all(
                row[name]
                for name in (
                    "geometry_eligible",
                    "planner_feasible",
                    "exact_history_qualified",
                    "time_aligned_collision",
                )
            ):
                raise ValueError("accepted SOP05R report is missing stage evidence")
            event_ids.append(event_id)
        if not state_id or not template_id:  # pragma: no cover - validators above
            raise RuntimeError("validated report identity changed")
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("SOP05R accepted report event IDs are not unique")


def _validate_active_metadata(
    metadata: Mapping[str, object],
    row: Mapping[str, object],
) -> None:
    if metadata.get("active_revealability_version") != (
        SOP05R_ACTIVE_REVEALABILITY_VERSION
    ):
        raise ValueError("SOP05R active revealability version mismatch")
    active_ids = metadata.get("active_revealable_action_ids")
    if (
        not isinstance(active_ids, list)
        or any(action_id not in CANONICAL_ACTION_IDS for action_id in active_ids)
        or len(active_ids) != len(set(active_ids))
        or "stop_scan" in active_ids
        or active_ids != [action_id for action_id in CANONICAL_ACTION_IDS if action_id in active_ids]
    ):
        raise ValueError("SOP05R active revealable action IDs are invalid")
    expected_active = bool(active_ids)
    if row["active_revealable"] is not expected_active:
        raise ValueError("SOP05R event active revealability flag mismatch")
    expected_status = "active_revealable" if expected_active else "natural_difficult"
    if metadata.get("active_revealability_status") != expected_status:
        raise ValueError("SOP05R active revealability status mismatch")
    if row["active_revealable_action_ids"] != active_ids:
        raise ValueError("SOP05R event action IDs differ from world metadata")
    for field in (
        "first_visible_time_by_verification_action",
        "matched_wait_visible_time",
        "active_revealability_actions",
    ):
        mapping = metadata.get(field)
        if not isinstance(mapping, dict) or set(mapping) != set(CANONICAL_ACTION_IDS):
            raise ValueError(f"SOP05R {field} action identity mismatch")
    actions = metadata["active_revealability_actions"]
    for action_id in CANONICAL_ACTION_IDS:
        action = actions[action_id]
        if not isinstance(action, dict) or action.get("active_revealable") is not (
            action_id in active_ids
        ):
            raise ValueError("SOP05R action revealability evidence mismatch")


def _restore_event(
    row: Mapping[str, object],
    *,
    record: EventTargetMotionRecord,
    world: object,
    grid: GridSpec,
    manifest: Mapping[str, object],
    config: Sop05rConfig,
    trajectory_store: Sop05rTrajectoryStore,
) -> GeneratedEvent:
    if set(row) != _EVENT_ROW_KEYS:
        raise ValueError("SOP05R event row keys mismatch")
    validate_event_target_motion_world_join(record, world, grid)
    validate_oracle_world(world, grid)
    if row["generated_event_id"] != record.generated_event_id:
        raise ValueError("SOP05R event row/target record ID mismatch")
    expected_scalars = {
        "event_kind": "environment",
        "world_id": record.world_id,
        "base_state_id": record.base_state_id,
        "trajectory_id": record.trajectory_id,
        "target_motion_record_digest": record.record_digest,
    }
    for name, expected in expected_scalars.items():
        if row[name] != expected:
            raise ValueError(f"SOP05R event {name} mismatch")
    metadata = world.metadata
    if not isinstance(metadata, dict):
        raise ValueError("SOP05R world metadata must be a mapping")
    expected_metadata = {
        "schema_version": config.schema_version,
        "generator_algorithm_version": SOP05R_GENERATOR_VERSION,
        "event_kind": "environment",
        "scene_template_id": row["template_id"],
        "base_state_id": record.base_state_id,
        "target_snippet_id": record.source_snippet_id,
        "source_object_id": record.source_object_id,
        "nominal_trajectory_id": record.trajectory_id,
        "config_digest": config.digest,
        "run_id": manifest["run_id"],
        "producer_source_identity": manifest["producer_source_identity"],
    }
    for name, expected in expected_metadata.items():
        if metadata.get(name) != expected:
            raise ValueError(f"SOP05R world metadata {name} mismatch")
    history = np.asarray(row["target_visibility_history"])
    visibility = np.asarray(row["visibility_sequence"])
    if history.shape != (8,) or history.dtype != np.bool_:
        raise ValueError("SOP05R target visibility history is invalid")
    if visibility.shape != (grid.future_steps,) or visibility.dtype != np.bool_:
        raise ValueError("SOP05R future visibility sequence is invalid")
    assessment = classify_sop05r_history(history)
    if row["history_visibility_regime"] != assessment.regime:
        raise ValueError("SOP05R event history regime mismatch")
    requested_history_regime = metadata.get(
        "requested_history_visibility_regime"
    )
    if requested_history_regime not in HISTORY_VISIBILITY_REGIMES:
        raise ValueError("SOP05R requested history regime is invalid")
    fallback_used = metadata.get("history_regime_fallback_used")
    if not isinstance(fallback_used, bool) or fallback_used != (
        assessment.regime != requested_history_regime
    ):
        raise ValueError("SOP05R history regime fallback metadata mismatch")
    expected_history_metadata = {
        "target_history_visibility_vector": [bool(value) for value in history],
        "target_history_visibility_regime": assessment.regime,
        "target_history_last_visible_index": assessment.last_visible_index,
        "target_history_trailing_hidden_frames": assessment.trailing_hidden_frames,
        "target_history_visibility_policy_version": config.history_policy.version,
        "target_history_visibility_policy": config.history_policy.as_dict(),
        "visibility_sequence": [bool(value) for value in visibility],
    }
    policy_digest = _sha256_bytes(_canonical_json_bytes(config.history_policy.as_dict()))
    expected_history_metadata["target_history_visibility_policy_digest"] = policy_digest
    for name, expected in expected_history_metadata.items():
        if metadata.get(name) != expected:
            raise ValueError(f"SOP05R history metadata {name} mismatch")
    conflict_time = _finite_real(row["conflict_time_s"], name="conflict_time_s")
    conflict_index = _nonnegative_int(row["conflict_index"], name="conflict_index")
    if conflict_index >= grid.future_steps or conflict_time <= 0.0:
        raise ValueError("SOP05R conflict index/time is invalid")
    if (
        metadata.get("conflict_time_s") != conflict_time
        or metadata.get("conflict_index") != conflict_index
    ):
        raise ValueError("SOP05R conflict metadata mismatch")
    _validate_active_metadata(metadata, row)
    trajectory_by_event = {
        item.event_id: item for item in trajectory_store.records
    }
    trajectory = trajectory_by_event.get(record.generated_event_id)
    if trajectory is None:
        raise ValueError("SOP05R event trajectory record is missing")
    if (
        trajectory.base_state_id != record.base_state_id
        or trajectory.template_id != row["template_id"]
        or trajectory.nominal_trajectory_id != record.trajectory_id
        or trajectory.config_digest != config.digest
        or list(trajectory.candidate_trajectory_ids)
        != metadata.get("candidate_trajectory_ids")
        or list(trajectory.alternative_trajectory_ids)
        != metadata.get("alternative_trajectory_ids")
    ):
        raise ValueError("SOP05R event/trajectory store join mismatch")
    provenance = metadata.get("target_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("SOP05R target provenance must be a mapping")
    footprint_spec = json.loads(_canonical_json_bytes(record.footprint_spec).decode("ascii"))
    target = TransplantedDynamicObject(
        target_dynamic_object_id=record.target_dynamic_object_id,
        source_object_id=record.source_object_id,
        snippet_id=record.source_snippet_id,
        object_type=record.object_type,
        footprint_spec=footprint_spec,
        footprint_spec_digest=record.footprint_spec_digest,
        history_poses=np.array(record.history_poses, dtype=np.float32, order="C", copy=True),
        current_pose=np.array(record.current_pose, dtype=np.float32, order="C", copy=True),
        future_poses=np.array(record.future_poses, dtype=np.float32, order="C", copy=True),
        provenance=json.loads(_canonical_json_bytes(provenance).decode("ascii")),
    )
    identity = {
        "generator_algorithm_version": SOP05R_GENERATOR_VERSION,
        "template_id": row["template_id"],
        "base_state_id": record.base_state_id,
        "split": manifest["split"],
        "seed": world.random_seed,
        "config_digest": config.digest,
        "planner_candidate_set_digest": metadata.get("planner_candidate_set_digest"),
        "nominal_trajectory_id": record.trajectory_id,
        "alternative_trajectory_ids": metadata.get("alternative_trajectory_ids"),
        "target_dynamic_object_id": record.target_dynamic_object_id,
        "target_history_visibility": [bool(value) for value in history],
        "requested_history_visibility_regime": requested_history_regime,
        "conflict_time_s": conflict_time,
        "conflict_point": metadata.get("conflict_point"),
    }
    expected_event_id = "event-" + hashlib.sha256(
        b"sop05r_event_identity_v1\0" + _canonical_json_bytes(identity)
    ).hexdigest()[:24]
    if expected_event_id != record.generated_event_id:
        raise ValueError("SOP05R generated event identity mismatch")
    expected_world_id = "world-" + hashlib.sha256(
        b"sop05r_world_identity_v1\0"
        + expected_event_id.encode("ascii")
        + compute_motion_array_digest(
            record.history_poses, field_name="target_history_poses"
        ).encode("ascii")
        + compute_motion_array_digest(
            record.future_poses, field_name="target_future_poses"
        ).encode("ascii")
    ).hexdigest()[:24]
    if expected_world_id != record.world_id:
        raise ValueError("SOP05R generated world identity mismatch")
    return GeneratedEvent(
        generated_event_id=record.generated_event_id,
        event_kind="environment",
        world=world,
        target=target,
        target_motion_record=record,
        visibility_sequence=visibility,
        target_visibility_history=history,
        conflict_time_s=conflict_time,
        conflict_index=conflict_index,
    )


def _validate_world_copies(root: Path, target_motion: LoadedEventTargetMotionShard) -> None:
    directory = root / "worlds"
    expected = {f"{world_id}.npz" for world_id in target_motion.worlds}
    observed = {path.name for path in directory.iterdir()}
    if observed != expected:
        raise ValueError("SOP05R independent world file set mismatch")
    for world_id, expected_world in target_motion.worlds.items():
        path = directory / f"{world_id}.npz"
        if path.is_symlink() or not path.is_file():
            raise ValueError("SOP05R independent world artifact is invalid")
        try:
            world = load_dataclass(path)
        except Exception as exc:
            raise ValueError("failed to load SOP05R independent world") from exc
        if compute_oracle_world_semantic_digest(world) != (
            compute_oracle_world_semantic_digest(expected_world)
        ):
            raise ValueError("SOP05R independent world semantic digest mismatch")


def _load_empty_nested_collections(
    root: Path,
) -> tuple[LoadedEventTargetMotionShard, Sop05rTrajectoryStore]:
    target_dir = root / "target_motion"
    trajectory_dir = root / "planner_trajectories"
    if {path.name for path in target_dir.iterdir()} != {"empty.json"}:
        raise ValueError("SOP05R empty target-motion layout mismatch")
    if {path.name for path in trajectory_dir.iterdir()} != {"empty.json"}:
        raise ValueError("SOP05R empty trajectory-store layout mismatch")
    target = _load_json(target_dir / "empty.json", label="empty target motion")
    if set(target) != {
        "version",
        "record_count",
        "manifest_digest",
        "payload_semantic_digest",
    } or target["version"] != SOP05R_EMPTY_TARGET_MOTION_VERSION or target[
        "record_count"
    ] != 0:
        raise ValueError("SOP05R empty target-motion evidence mismatch")
    target_manifest_digest = _hex_digest(
        target["manifest_digest"], name="empty target manifest digest"
    )
    target_payload_digest = _hex_digest(
        target["payload_semantic_digest"], name="empty target payload digest"
    )
    trajectory = _load_json(
        trajectory_dir / "empty.json", label="empty trajectory store"
    )
    if set(trajectory) != {
        "version",
        "record_count",
        "collection_semantic_digest",
    } or trajectory["version"] != SOP05R_EMPTY_TRAJECTORY_STORE_VERSION or trajectory[
        "record_count"
    ] != 0:
        raise ValueError("SOP05R empty trajectory-store evidence mismatch")
    trajectory_digest = _hex_digest(
        trajectory["collection_semantic_digest"],
        name="empty trajectory collection digest",
    )
    return (
        LoadedEventTargetMotionShard(
            records=(),
            worlds={},
            manifest_digest=target_manifest_digest,
            payload_semantic_digest=target_payload_digest,
            summary=target,
        ),
        Sop05rTrajectoryStore(
            records=(),
            manifest=trajectory,
            collection_semantic_digest=trajectory_digest,
        ),
    )


def _validate_summary_and_selection(
    summary: dict[str, object],
    reports: list[dict[str, object]],
    event_rows: list[dict[str, object]],
    *,
    manifest: Mapping[str, object],
    config: Sop05rConfig,
) -> None:
    required_keys = {
        "summary_version",
        "selection_version",
        "input_base_count",
        "geometry_eligible_base_count",
        "template_count",
        "geometry_eligible_template_count",
        "planner_feasible_template_count",
        "exact_history_qualified_count",
        "time_aligned_collision_count",
        "active_revealable_count",
        "accepted_count",
        "selected_count",
        "accepted_quota",
        "quota_met",
        "history_visibility",
        "revealability",
        "selected_event_ids",
        "attempts_per_accepted_event",
        "rejection_reasons",
        "run_id",
        "run_state",
        "split",
        "publication_semantic_digest",
    }
    if set(summary) != required_keys:
        raise ValueError("SOP05R generation summary keys mismatch")
    if (
        summary["summary_version"] != SOP05R_SUMMARY_VERSION
        or summary["selection_version"] != SOP05R_SELECTION_VERSION
        or summary["run_id"] != manifest["run_id"]
        or summary["run_state"] != manifest["run_state"]
        or summary["split"] != manifest["split"]
        or summary["accepted_quota"] != manifest["accepted_quota"]
        or summary["publication_semantic_digest"]
        != manifest["publication_semantic_digest"]
    ):
        raise ValueError("SOP05R generation summary identity mismatch")
    accepted_reports = [row for row in reports if row["generated_event_id"] is not None]
    candidates = tuple(
        Sop05rSelectionCandidate(
            generated_event_id=row["generated_event_id"],
            base_state_id=row["state_id"],
            template_id=row["template_id"],
            history_visibility_regime=row["history_visibility_regime"],
            active_revealable=row["active_revealable"],
            schedule_rank=row["base_rank"],
        )
        for row in accepted_reports
    )
    selection = select_sop05r_event_ids(
        candidates,
        accepted_quota=manifest["accepted_quota"],
        seed=manifest["seed"],
        config=config,
    )
    selected_ids = [row["generated_event_id"] for row in event_rows]
    if (
        selected_ids != list(selection.event_ids)
        or selected_ids != manifest["selected_event_ids"]
        or selected_ids != summary["selected_event_ids"]
    ):
        raise ValueError("SOP05R selected event sequence mismatch")
    expected_counts = {
        "input_base_count": len(manifest["schedule"]),
        "geometry_eligible_base_count": len(
            {
                row["base_rank"]
                for row in reports
                if row["geometry_eligible"]
            }
        ),
        "template_count": len(reports),
        "geometry_eligible_template_count": sum(row["geometry_eligible"] for row in reports),
        "planner_feasible_template_count": sum(row["planner_feasible"] for row in reports),
        "exact_history_qualified_count": sum(row["exact_history_qualified"] for row in reports),
        "time_aligned_collision_count": sum(row["time_aligned_collision"] for row in reports),
        "active_revealable_count": sum(row["active_revealable"] for row in accepted_reports),
        "accepted_count": len(accepted_reports),
        "selected_count": len(event_rows),
    }
    for name, expected in expected_counts.items():
        if summary[name] != expected:
            raise ValueError(f"SOP05R generation summary {name} mismatch")
    expected_history = {
        "requested": dict(selection.requested_history_counts),
        "exact_qualified": dict(selection.exact_qualified_history_counts),
        "accepted": dict(selection.accepted_history_counts),
        "selected": dict(selection.selected_history_counts),
        "deficits": {
            regime: selection.deficits[regime]
            for regime in HISTORY_VISIBILITY_REGIMES
        },
    }
    if summary["history_visibility"] != expected_history:
        raise ValueError("SOP05R history selection summary mismatch")
    expected_revealability = {
        "selection_filtering": config.revealability.selection_filtering,
        "requested_active_count": selection.active_revealable_requested_count,
        "accepted_active_count": selection.active_revealable_accepted_count,
        "selected_active_count": selection.active_revealable_selected_count,
        "selected_natural_difficult_count": selection.natural_difficult_selected_count,
        "active_deficit": selection.deficits["active_revealable"],
    }
    if summary["revealability"] != expected_revealability:
        raise ValueError("SOP05R revealability selection summary mismatch")
    if summary["quota_met"] is not selection.quota_met:
        raise ValueError("SOP05R quota status mismatch")
    attempts = [row["attempt_index"] + 1 for row in accepted_reports]
    if summary["attempts_per_accepted_event"] != attempts:
        raise ValueError("SOP05R attempts-per-accepted summary mismatch")
    reasons = Counter(
        row["rejection_reason"] for row in reports if row["rejection_reason"] is not None
    )
    if summary["rejection_reasons"] != dict(sorted(reasons.items())):
        raise ValueError("SOP05R rejection reason summary mismatch")
    expected_state = "complete" if selection.quota_met else "quota_unmet"
    if manifest["run_state"] != expected_state:
        raise ValueError("SOP05R run_state differs from recomputed quota")


def load_sop05r_events(
    root: str | Path,
    *,
    require_complete: bool = True,
    expected_publication_semantic_digest: str | None = None,
    expected_run_id: str | None = None,
) -> LoadedSop05rEvents:
    directory = Path(root)
    complete = _validate_root(directory, require_complete=require_complete)
    manifest, source_identity, config, schedule = _load_manifest_contract(directory)
    _validate_marker(directory, manifest, complete=complete)
    if expected_run_id is not None and manifest["run_id"] != expected_run_id:
        raise ValueError("SOP05R run_id differs from expected_run_id")
    if expected_publication_semantic_digest is not None:
        _hex_digest(
            expected_publication_semantic_digest,
            name="expected publication semantic digest",
        )
        if manifest["publication_semantic_digest"] != (
            expected_publication_semantic_digest
        ):
            raise ValueError("SOP05R publication semantic digest differs from expected")
    grid = build_grid_spec(manifest["base_config"])
    selected_ids = tuple(manifest["selected_event_ids"])
    if selected_ids:
        target_motion = load_event_target_motion_shard(
            directory / "target_motion",
            grid=grid,
            expected_generated_event_ids=set(selected_ids),
        )
        trajectory_store = load_sop05r_trajectory_store(
            directory / "planner_trajectories"
        )
    else:
        target_motion, trajectory_store = _load_empty_nested_collections(
            directory
        )
    if tuple(record.event_id for record in trajectory_store.records) != tuple(
        sorted(selected_ids)
    ):
        raise ValueError("SOP05R trajectory store event set mismatch")
    artifacts = manifest["artifacts"]
    if (
        trajectory_store.collection_semantic_digest
        != artifacts["trajectory_collection_semantic_digest"]
        or target_motion.manifest_digest != artifacts["target_motion_manifest_digest"]
        or target_motion.payload_semantic_digest
        != artifacts["target_motion_payload_semantic_digest"]
    ):
        raise ValueError("SOP05R nested collection digest mismatch")
    _validate_world_copies(directory, target_motion)
    reports = _load_jsonl(
        directory / "template_reports.jsonl", label="template reports"
    )
    _validate_reports(reports)
    event_rows = _load_jsonl(
        directory / "events.jsonl",
        label="events",
        allow_empty=True,
    )
    if [row.get("generated_event_id") for row in event_rows] != list(selected_ids):
        raise ValueError("SOP05R event row order mismatch")
    records = {
        record.generated_event_id: record for record in target_motion.records
    }
    events = tuple(
        _restore_event(
            row,
            record=records[row["generated_event_id"]],
            world=target_motion.worlds[records[row["generated_event_id"]].world_id],
            grid=grid,
            manifest=manifest,
            config=config,
            trajectory_store=trajectory_store,
        )
        for row in event_rows
    )
    summary = _load_json(
        directory / "generation_summary.json", label="generation summary"
    )
    _validate_summary_and_selection(
        summary,
        reports,
        event_rows,
        manifest=manifest,
        config=config,
    )
    context = Sop05rPublicationContext(
        run_id=manifest["run_id"],
        split=manifest["split"],
        seed=manifest["seed"],
        accepted_quota=manifest["accepted_quota"],
        base_config=manifest["base_config"],
        config=config,
        verification_action_config=manifest["verification_action_config"],
        input_lock=manifest["input_lock"],
        producer_source_identity=source_identity,
        schedule=schedule,
    )
    publication_digest = _publication_digest(
        context=context,
        event_rows=event_rows,
        reports_sha256=_sha256_file(directory / "template_reports.jsonl"),
        trajectory_digest=trajectory_store.collection_semantic_digest,
        target_manifest_digest=target_motion.manifest_digest,
        target_payload_digest=target_motion.payload_semantic_digest,
    )
    if publication_digest != manifest["publication_semantic_digest"]:
        raise ValueError("SOP05R publication semantic digest mismatch")
    return LoadedSop05rEvents(
        events=events,
        target_motion=target_motion,
        trajectory_store=trajectory_store,
        template_reports=tuple(reports),
        event_rows=tuple(event_rows),
        manifest=manifest,
        summary=summary,
        publication_semantic_digest=publication_digest,
        complete=complete,
    )
