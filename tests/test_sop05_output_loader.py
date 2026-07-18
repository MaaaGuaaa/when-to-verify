"""Strict SOP-05 publication consumer and GeneratedEvent restoration tests."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.contracts import GridSpec, OracleWorld
from src.generation.event_target_motion_shard import (
    build_event_target_motion_world_metadata,
    compute_footprint_spec_digest,
    compute_motion_array_digest,
    create_event_target_motion_record,
    write_event_target_motion_shard,
)


_FIXTURE_PAIRS_BY_EVENT_ID: dict[str, tuple[object, OracleWorld]] = {}


def _grid() -> GridSpec:
    return GridSpec(
        height=32,
        width=32,
        history_steps=8,
        future_steps=15,
        resolution_m=0.1,
    )


def _generator_contract() -> tuple[dict[str, object], str, str]:
    from src.generation.event_sampler import _generator_digest, load_generator_config

    config_path = Path(__file__).resolve().parents[1] / "configs/generator_test.yaml"
    config = load_generator_config(config_path)
    policy = config["target_type_policy"]
    return policy.as_dict(), policy.digest, _generator_digest(config)


def _record_and_world(
    event_index: int = 0,
    event_kind: str = "structural",
    *,
    generator_algorithm_version: str = "joint_occluder_first_v4",
):
    from src.generation.dynamic_object_transplant import (
        MOTION_SNIPPET_LAYOUT_VERSION,
        normalize_target_type_policy,
    )
    from src.generation.event_target_motion_shard import (
        EVENT_TARGET_MOTION_LAYOUT_VERSION,
    )

    history = np.zeros((8, 3), dtype=np.float32)
    history[:, 0] = np.linspace(-0.7, 0.0, 8, dtype=np.float32)
    future = np.zeros((15, 3), dtype=np.float32)
    future[:, 0] = np.linspace(0.1, 1.5, 15, dtype=np.float32)
    current = history[-1].copy()
    spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.3},
    }
    target_type_policy, target_type_policy_digest, generator_config_digest = (
        _generator_contract()
    )
    assert (
        normalize_target_type_policy(target_type_policy).digest
        == target_type_policy_digest
    )
    suffix = f"{event_kind}-{event_index}"
    from src.generation.event_sampler import (
        compute_generated_event_id,
        compute_generated_world_id,
    )

    target_dynamic_object_id = f"target-loader-fixture-{suffix}"
    source_snippet_id = f"snippet-loader-fixture-{suffix}"
    source_object_id = "recording::person"
    footprint_spec_digest = compute_footprint_spec_digest(spec)
    generated_event_id = compute_generated_event_id(
        generator_algorithm_version=generator_algorithm_version,
        generator_config_digest=generator_config_digest,
        base_state_id="base-loader-fixture",
        trajectory_id="trajectory-loader-fixture",
        event_index=event_index,
        attempt_index=event_index,
        attempt_seed=17 + event_index,
        event_kind=event_kind,
        conflict_index=4,
        conflict_time_s=1.0,
        target_dynamic_object_id=target_dynamic_object_id,
        source_snippet_id=source_snippet_id,
        source_object_id=source_object_id,
        object_type="human",
        footprint_spec=spec,
        footprint_spec_digest=footprint_spec_digest,
        target_type_policy_digest=target_type_policy_digest,
        layout_version=EVENT_TARGET_MOTION_LAYOUT_VERSION,
    )
    world_id = compute_generated_world_id(
        generator_algorithm_version=generator_algorithm_version,
        generator_config_digest=generator_config_digest,
        generated_event_id=generated_event_id,
        base_state_id="base-loader-fixture",
        trajectory_id="trajectory-loader-fixture",
        event_kind=event_kind,
        target_dynamic_object_id=target_dynamic_object_id,
        source_snippet_id=source_snippet_id,
        source_object_id=source_object_id,
        object_type="human",
        footprint_spec=spec,
        footprint_spec_digest=footprint_spec_digest,
        target_type_policy_digest=target_type_policy_digest,
        layout_version=EVENT_TARGET_MOTION_LAYOUT_VERSION,
        history_array_digest=compute_motion_array_digest(
            history, field_name="target_history_poses"
        ),
        current_pose=current,
        future_array_digest=compute_motion_array_digest(
            future, field_name="target_future_poses"
        ),
    )
    record = create_event_target_motion_record(
        generated_event_id=generated_event_id,
        world_id=world_id,
        base_state_id="base-loader-fixture",
        trajectory_id="trajectory-loader-fixture",
        target_dynamic_object_id=target_dynamic_object_id,
        source_snippet_id=source_snippet_id,
        source_object_id=source_object_id,
        object_type="human",
        footprint_spec=spec,
        footprint_spec_digest=footprint_spec_digest,
        target_type_policy_digest=target_type_policy_digest,
        history_poses=history,
        current_pose=current,
        future_poses=future,
    )
    provenance = {
        "snippet_id": record.source_snippet_id,
        "source_recording_id": "recording",
        "source_object_id": record.source_object_id,
        "source_body_name": "person",
        "raw_role": "person",
        "geometry_source": "thor_marker_extent",
        "orientation_source": "velocity",
        "target_type_policy_digest": record.target_type_policy_digest,
        "footprint_spec_digest": record.footprint_spec_digest,
        "conflict_time_s": 1.0,
        "conflict_point": [0.5, 0.0],
        "crossing_direction": [1.0, 0.0],
        "rotation_rad": 0.0,
        "time_scale": 1.0,
        "motion_snippet_layout_version": MOTION_SNIPPET_LAYOUT_VERSION,
        "source_current_index": 7,
        "source_current_time_s": 1.4,
        "source_conflict_anchor_time_s": 2.4,
        "seed": 17 + event_index,
    }
    visibility = [False, False, False, True, True] + [True] * 11
    structural = (
        {
            "forward_fov_deg": 180.0,
            "range_m": 6.0,
            "blind_sectors": [],
        }
        if event_kind in {"structural", "mixed"}
        else None
    )
    occluders = (
        (
            {
                "occluder_id": f"occluder-loader-fixture-{suffix}",
                "kind": "pillar",
                "polygon_xy": [
                    [0.1, -0.2],
                    [0.3, -0.2],
                    [0.3, 0.2],
                    [0.1, 0.2],
                ],
                "height_m": 2.0,
            },
        )
        if event_kind in {"environment", "mixed"}
        else ()
    )
    metadata = {
        **build_event_target_motion_world_metadata(record),
        "schema_version": "2.0.0",
        "generator_algorithm_version": generator_algorithm_version,
        "event_kind": event_kind,
        "dynamic_object_snippet_id": record.source_snippet_id,
        "target_type_policy": target_type_policy,
        "generator_config_digest": generator_config_digest,
        "conflict_time_s": 1.0,
        "conflict_index": 4,
        "event_slot_index": event_index,
        "attempt_index": event_index,
        "target_provenance": provenance,
        "visibility_sequence": visibility,
        "target_visibility_history": [False] * 8,
        "target_visibility_history_layout": (
            "target_visibility_history8_current7_v1"
        ),
        "context_dynamic_object_ids": [],
        "occluder_candidate_rejection_reasons": {},
    }
    world = OracleWorld(
        world_id=record.world_id,
        base_state_id=record.base_state_id,
        static_occupancy=np.zeros((32, 32), dtype=np.float32),
        dynamic_object_trajectories={
            record.target_dynamic_object_id: future.copy()
        },
        dynamic_object_specs={record.target_dynamic_object_id: spec},
        occluders=occluders,
        blind_spot_config={
            "kind": event_kind,
            "structural": structural,
            "occluder_ids": [
                occluder["occluder_id"] for occluder in occluders
            ],
        },
        random_seed=17 + event_index,
        metadata=metadata,
    )
    assert record.layout_version == EVENT_TARGET_MOTION_LAYOUT_VERSION
    _FIXTURE_PAIRS_BY_EVENT_ID[record.generated_event_id] = (record, world)
    return record, world


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _run_id(manifest: dict[str, object]) -> str:
    input_lock = manifest["input_lock"]
    scientific = manifest["scientific_request"]
    identity = {
        "version": manifest["producer_version"],
        "producer_source_identity": manifest["producer_source_identity"],
        "split": manifest["split"],
        "sop03": {
            key: input_lock["sop03"][key]
            for key in (
                "code_commit",
                "checksum_manifest_sha256",
                "audit_sha256",
            )
        },
        "sop04": {
            key: input_lock["sop04"][key]
            for key in (
                "code_commit",
                "checksum_manifest_sha256",
                "audit_sha256",
                "trajectory_bank_version",
                "pose_time_layout_version",
                "trajectory_steps",
                "dt_s",
                "first_pose_time_s",
                "last_pose_time_s",
                "pose_time_offsets_sha256",
                "bank_semantic_digest_sha256",
                "external_handoff_digest_sha256",
            )
        },
        "selection": input_lock["selection"],
        "base_config_sha256": scientific["base_config_sha256"],
        "generator_config_sha256": scientific["generator_config_sha256"],
        "generator_config_semantic_digest": scientific[
            "generator_config_semantic_digest"
        ],
        "target_type_policy": scientific["target_type_policy"],
        "target_type_policy_digest": scientific[
            "target_type_policy_digest"
        ],
        "accepted_quota": scientific["accepted_quota"],
        "events_per_pair": scientific["events_per_pair"],
        "selection_version": scientific["selection_version"],
    }
    digest = hashlib.blake2b(_canonical_json_bytes(identity), digest_size=16)
    return f"sop05-run-{digest.hexdigest()}"


def _write_outer_checksums(root: Path) -> None:
    checksum_lines = []
    for path in sorted(
        root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative not in {
            "checksums.sha256",
            ".producer-complete",
        }:
            checksum_lines.append(f"{_sha256(path)}  {relative}\n")
    (root / "checksums.sha256").write_text(
        "".join(checksum_lines), encoding="utf-8"
    )


def _reseal_publication(root: Path) -> None:
    from src.generation.sop05_publication_identity import (
        SOP05_PUBLICATION_IDENTITY_VERSION,
        compute_sop05_publication_semantic_digest,
    )

    _write_outer_checksums(root)
    marker_path = root / ".producer-complete"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    marker["marker_version"] = "sop05_producer_complete_v2"
    marker["publication_identity_version"] = (
        SOP05_PUBLICATION_IDENTITY_VERSION
    )
    marker["run_id"] = manifest["run_id"]
    marker["run_manifest_sha256"] = _sha256(root / "run_manifest.json")
    marker["checksums_sha256"] = _sha256(root / "checksums.sha256")
    marker["publication_semantic_digest"] = (
        compute_sop05_publication_semantic_digest(
            run_id=marker["run_id"],
            run_manifest_sha256=marker["run_manifest_sha256"],
            checksums_sha256=marker["checksums_sha256"],
            target_motion_manifest_digest=marker[
                "target_motion_manifest_digest"
            ],
            target_motion_payload_semantic_digest=marker[
                "target_motion_payload_semantic_digest"
            ],
        )
    )
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _fixture_publication_digest(root: Path) -> str:
    marker = json.loads(
        (root / ".producer-complete").read_text(encoding="utf-8")
    )
    value = marker["publication_semantic_digest"]
    assert isinstance(value, str)
    return value


def _bucket_summary(
    *, attempted: int, accepted: int, rejection_reason: str
) -> dict[str, object]:
    rejected = attempted - accepted
    return {
        "attempted": attempted,
        "accepted": accepted,
        "rejected": rejected,
        "attempt_acceptance_rate": accepted / attempted,
        "rejection_reasons": (
            {rejection_reason: rejected} if rejected else {}
        ),
    }


def _event_kind_bucket(
    *, requested: int, accepted: int, rejection_reason: str
) -> dict[str, object]:
    rejected = requested - accepted
    return {
        "requested": requested,
        "attempted": requested,
        "accepted": accepted,
        "rejected": rejected,
        "request_acceptance_rate": accepted / requested,
        "attempt_acceptance_rate": accepted / requested,
        "rejection_reasons": (
            {rejection_reason: rejected} if rejected else {}
        ),
        "rejection_stage_counts": {
            "occluder_geometry": 0,
            "target_conditioning": 0,
            "visibility": rejected,
        },
    }


def _selection_key_for_id(seed: int, event_id: str) -> tuple[str, str]:
    payload = _canonical_json_bytes(
        ["sop05_total_quota_selection_v1", seed, event_id]
    )
    return hashlib.blake2b(payload, digest_size=16).hexdigest(), event_id


def _write_complete_publication(
    root: Path,
    *,
    accepted_quota: int = 10,
    selected_kind: str = "structural",
) -> tuple[object, OracleWorld, str]:
    if selected_kind not in {"environment", "structural", "mixed"}:
        raise ValueError("unsupported selected_kind fixture")
    requested_count = 40
    requested_counts = {"environment": 24, "structural": 12, "mixed": 4}
    if accepted_quota + 1 > requested_counts[selected_kind]:
        raise ValueError("fixture quota exceeds selected-kind request capacity")

    primary_pairs = [
        _record_and_world(event_index, selected_kind)
        for event_index in range(accepted_quota + 1)
    ]
    selected_pairs = sorted(
        primary_pairs,
        key=lambda pair: _selection_key_for_id(
            23, pair[0].generated_event_id
        ),
    )[:accepted_quota]
    worst_selected_key = max(
        _selection_key_for_id(23, pair[0].generated_event_id)
        for pair in selected_pairs
    )

    # Keep accepted events of the other kinds in the signed pair report, but
    # deliberately choose IDs ranked below the primary-kind top-N boundary.
    # This proves that event kind neither grants a quota nor changes ranking.
    other_pairs = []
    candidate_index = 100
    for event_kind in (
        kind
        for kind in ("environment", "structural", "mixed")
        if kind != selected_kind
    ):
        while True:
            candidate = _record_and_world(candidate_index, event_kind)
            candidate_index += 1
            if (
                _selection_key_for_id(
                    23, candidate[0].generated_event_id
                )
                > worst_selected_key
            ):
                other_pairs.append(candidate)
                break
    pairs = primary_pairs + other_pairs
    records = [record for record, _ in pairs]
    event_kinds = [world.metadata["event_kind"] for _, world in pairs]
    generated_counts = {
        kind: event_kinds.count(kind)
        for kind in ("environment", "structural", "mixed")
    }
    selected_records = [record for record, _ in selected_pairs]
    selected_worlds = [world for _, world in selected_pairs]
    record, world = selected_pairs[0]
    generated_count = len(pairs)
    root.mkdir()
    shard = root / "target_motions"
    write_event_target_motion_shard(
        selected_records, selected_worlds, shard, grid=_grid()
    )
    configs = root / "configs"
    configs.mkdir()
    (configs / "base.yaml").write_text(
        "schema_version: '2.0.0'\n"
        "bev:\n"
        "  range_m: 3.2\n"
        "  resolution_m: 0.1\n"
        "  size: 32\n"
        "  history_steps: 8\n"
        "  history_dt_s: 0.2\n"
        "  future_steps: 15\n"
        "  future_dt_s: 0.2\n",
        encoding="utf-8",
    )
    generator_source = (
        Path(__file__).resolve().parents[1] / "configs/generator_test.yaml"
    )
    (configs / "generator.yaml").write_bytes(generator_source.read_bytes())
    target_type_policy, target_type_policy_digest, generator_config_digest = (
        _generator_contract()
    )
    rejected_count = requested_count - generated_count
    rejection_reason = "fixture_visibility_rejection"
    rejection_reasons = {rejection_reason: rejected_count}
    rejection_stages = {
        "occluder_geometry": 0,
        "target_conditioning": 0,
        "visibility": rejected_count,
    }
    pair_summary = {
        "schema_version": "2.0.0",
        "seed": 17,
        "requested_event_count": requested_count,
        "complete_joint_candidates_attempted": requested_count,
        "attempted_count": requested_count,
        "joint_candidate_attempted_count": requested_count,
        "attempt_index_start": 0,
        "attempt_index_stop_exclusive": None,
        "accepted_count": generated_count,
        "rejected_count": rejected_count,
        "acceptance_rate": generated_count / requested_count,
        "attempt_acceptance_rate": generated_count / requested_count,
        "request_acceptance_rate": generated_count / requested_count,
        "unaccepted_event_count": rejected_count,
        "rejection_reasons": rejection_reasons,
        "rejection_stage_counts": rejection_stages,
        "occluder_candidate_rejection_reasons": {},
        "requested_event_kind_counts": requested_counts,
        "event_kind_counts": generated_counts,
        "by_event_kind": {
            kind: _event_kind_bucket(
                requested=requested_counts[kind],
                accepted=generated_counts[kind],
                rejection_reason=rejection_reason,
            )
            for kind in requested_counts
        },
        "by_object_type": {
            "human": _bucket_summary(
                attempted=requested_count,
                accepted=generated_count,
                rejection_reason=rejection_reason,
            )
        },
        "by_footprint_kind": {
            "circle": _bucket_summary(
                attempted=requested_count,
                accepted=generated_count,
                rejection_reason=rejection_reason,
            )
        },
        "by_geometry_source": {
            "thor_marker_extent": _bucket_summary(
                attempted=requested_count,
                accepted=generated_count,
                rejection_reason=rejection_reason,
            )
        },
        "target_type_policy": target_type_policy,
        "target_type_policy_digest": target_type_policy_digest,
        "generator_config_digest": generator_config_digest,
        "generator_algorithm_version": "joint_occluder_first_v4",
    }
    (root / "pair_generation_reports.jsonl").write_text(
        json.dumps(
            {
                "report_version": "sop05_pair_generation_report_v2",
                "selection_version": "sop05_total_quota_selection_v1",
                "rank": 0,
                "state_id": record.base_state_id,
                "trajectory_id": record.trajectory_id,
                "seed": 17,
                "summary": pair_summary,
                "accepted_events": [
                    {
                        "generated_event_id": item.generated_event_id,
                        "event_kind": event_kind,
                    }
                    for item, event_kind in zip(
                        records, event_kinds, strict=True
                    )
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    selected_counts = {
        kind: sum(
            world.metadata["event_kind"] == kind
            for _, world in selected_pairs
        )
        for kind in ("environment", "structural", "mixed")
    }
    summary = {
        "summary_version": "sop05_generation_summary_v2",
        "run_id": None,
        "run_state": "complete",
        "processed_pair_count": 1,
        "requested_event_count": requested_count,
        "attempted_count": requested_count,
        "generator_accepted_count": generated_count,
        "selected_count": accepted_quota,
        "quota_trimmed_count": generated_count - accepted_quota,
        "generated_event_kind_counts": generated_counts,
        "selected_event_kind_counts": selected_counts,
        "quota_met": True,
        "rejection_reasons": rejection_reasons,
        "rejection_stage_counts": rejection_stages,
        "generator_invariants": {
            "schema_version": "2.0.0",
            "target_type_policy_digest": target_type_policy_digest,
            "generator_config_digest": generator_config_digest,
            "generator_algorithm_version": "joint_occluder_first_v4",
        },
    }
    schedule = [
        {
            "rank": 0,
            "state_id": record.base_state_id,
            "trajectory_id": record.trajectory_id,
            "pair_seed": 17,
        }
    ]
    selection = {
        "seed": 23,
        "max_base_states": 1,
        "trajectory_count": 1,
        "max_pairs": 1,
        "pair_count": 1,
        "pair_schedule_sha256": hashlib.sha256(
            _canonical_json_bytes(schedule)
        ).hexdigest(),
    }
    input_lock = {
        "version": "sop05_input_lock_v2",
        "split": "train",
        "sop03": {
            "code_commit": "1" * 40,
            "checksum_manifest_sha256": "a" * 64,
            "audit_sha256": "b" * 64,
            "completion_policy": "sop03_complete_marker_v1",
        },
        "sop04": {
            "code_commit": "2" * 40,
            "checksum_manifest_sha256": "c" * 64,
            "audit_sha256": "d" * 64,
            "completion_policy": "sop04_audited_bank_v2",
            "trajectory_bank_version": "sop04_audited_bank_v2",
            "pose_time_layout_version": (
                "future_endpoints_dt_to_horizon_v1"
            ),
            "trajectory_steps": 15,
            "dt_s": 0.2,
            "first_pose_time_s": 0.2,
            "last_pose_time_s": 3.0,
            "pose_time_offsets_sha256": (
                "1287220ad6fb1eec96bc41662ab61e2e443b1927f9fed0a4f6d3325aed2027db"
            ),
            "bank_semantic_digest_sha256": "f" * 64,
            "external_handoff_digest_sha256": "e" * 64,
        },
        "selection": selection,
    }
    scientific_request = {
        "seed": 23,
        "accepted_quota": accepted_quota,
        "events_per_pair": requested_count,
        "max_base_states": 1,
        "trajectory_count": 1,
        "max_pairs": 1,
        "selection_version": "sop05_total_quota_selection_v1",
        "base_config_sha256": _sha256(configs / "base.yaml"),
        "generator_config_sha256": _sha256(configs / "generator.yaml"),
        "generator_config_semantic_digest": generator_config_digest,
        "target_type_policy": target_type_policy,
        "target_type_policy_digest": target_type_policy_digest,
        "pair_schedule": schedule,
    }
    manifest = {
        "manifest_version": "sop05_run_manifest_v2",
        "producer_version": "sop05_generation_run_v4",
        "producer_source_identity": {
            "version": "sop05_producer_source_identity_v1",
            "git_commit": "3" * 40,
            "worktree_state": "clean",
            "dirty_tree_sha256": None,
        },
        "run_id": None,
        "run_state": "complete",
        "split": "train",
        "input_lock": input_lock,
        "scientific_request": scientific_request,
        "runtime": {
            "workers": 8,
            "checksum_workers": 8,
            "git_executable": (
                "/home/home/ccnt_zq/zq_zhouyiqun/.local/git/bin/git"
            ),
            "resolved_input_roots": {
                "sop03": "/fixture/sop03",
                "sop04": "/fixture/sop04",
            },
        },
        "artifacts": {
            "base_config_snapshot": "configs/base.yaml",
            "generator_config_snapshot": "configs/generator.yaml",
            "generation_summary": "generation_summary.json",
            "pair_generation_reports": "pair_generation_reports.jsonl",
            "checksums": "checksums.sha256",
            "target_motion_shard": "target_motions",
            "producer_complete": ".producer-complete",
        },
    }
    manifest["run_id"] = _run_id(manifest)
    summary["run_id"] = manifest["run_id"]
    (root / "generation_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n"
    )
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    _write_outer_checksums(root)

    from src.generation.event_target_motion_shard import (
        load_event_target_motion_shard,
    )

    loaded_shard = load_event_target_motion_shard(shard, grid=_grid())
    marker = {
        "marker_version": "sop05_producer_complete_v2",
        "run_id": manifest["run_id"],
        "run_manifest_sha256": _sha256(root / "run_manifest.json"),
        "checksums_sha256": _sha256(root / "checksums.sha256"),
        "target_motion_manifest_digest": loaded_shard.manifest_digest,
        "target_motion_payload_semantic_digest": (
            loaded_shard.payload_semantic_digest
        ),
    }
    from src.generation.sop05_publication_identity import (
        SOP05_PUBLICATION_IDENTITY_VERSION,
        compute_sop05_publication_semantic_digest,
    )

    marker["publication_identity_version"] = (
        SOP05_PUBLICATION_IDENTITY_VERSION
    )
    marker["publication_semantic_digest"] = (
        compute_sop05_publication_semantic_digest(
            run_id=marker["run_id"],
            run_manifest_sha256=marker["run_manifest_sha256"],
            checksums_sha256=marker["checksums_sha256"],
            target_motion_manifest_digest=marker[
                "target_motion_manifest_digest"
            ],
            target_motion_payload_semantic_digest=marker[
                "target_motion_payload_semantic_digest"
            ],
        )
    )
    (root / ".producer-complete").write_text(
        json.dumps(marker, sort_keys=True, indent=2) + "\n"
    )
    return record, world, str(marker["publication_semantic_digest"])


def _fixture_pair_from_event_id(event_id: str):
    try:
        return _FIXTURE_PAIRS_BY_EVENT_ID[event_id]
    except KeyError as exc:
        raise ValueError("unknown loader fixture event ID") from exc


def _rewrite_selected_shard(root: Path, selected_ids: list[str]) -> None:
    pairs = [_fixture_pair_from_event_id(event_id) for event_id in selected_ids]
    shard_path = root / "target_motions"
    shutil.rmtree(shard_path)
    write_event_target_motion_shard(
        [record for record, _ in pairs],
        [world for _, world in pairs],
        shard_path,
        grid=_grid(),
    )
    from src.generation.event_target_motion_shard import (
        load_event_target_motion_shard,
    )

    loaded_shard = load_event_target_motion_shard(shard_path, grid=_grid())
    marker_path = root / ".producer-complete"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["target_motion_manifest_digest"] = loaded_shard.manifest_digest
    marker["target_motion_payload_semantic_digest"] = (
        loaded_shard.payload_semantic_digest
    )
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _reseal_publication(root)


def test_restore_generated_event_recovers_full_audited_event() -> None:
    from src.generation.sop05_output_loader import restore_generated_event

    record, world = _record_and_world()
    event = restore_generated_event(record, world, grid=_grid())

    assert event.generated_event_id == record.generated_event_id
    assert event.event_kind == "structural"
    assert event.world is world
    assert event.target.target_dynamic_object_id == (
        record.target_dynamic_object_id
    )
    assert event.target.footprint_spec is not record.footprint_spec
    assert event.target.footprint_spec["footprint"] is not (
        record.footprint_spec["footprint"]
    )
    event.target.footprint_spec["footprint"]["radius_m"] = 0.31
    assert record.footprint_spec["footprint"]["radius_m"] == pytest.approx(0.3)
    assert event.target.provenance == world.metadata["target_provenance"]
    np.testing.assert_array_equal(event.target.history_poses, record.history_poses)
    np.testing.assert_array_equal(event.target.current_pose, record.current_pose)
    np.testing.assert_array_equal(event.target.future_poses, record.future_poses)
    assert event.visibility_sequence.shape == (16,)
    assert event.visibility_sequence.dtype == np.bool_
    assert event.target_visibility_history.shape == (8,)
    assert event.target_visibility_history.dtype == np.bool_
    assert event.conflict_time_s == pytest.approx(1.0)
    assert event.conflict_index == 4
    for array in (
        event.target.history_poses,
        event.target.current_pose,
        event.target.future_poses,
        event.visibility_sequence,
        event.target_visibility_history,
    ):
        assert array.flags.c_contiguous
        assert array.flags.owndata


def test_restore_generated_event_rejects_old_generator_algorithm() -> None:
    from src.generation.sop05_output_loader import restore_generated_event

    record, world = _record_and_world(
        generator_algorithm_version="joint_occluder_first_v3"
    )

    with pytest.raises(ValueError, match="generator_algorithm_version"):
        restore_generated_event(record, world, grid=_grid())


@pytest.mark.parametrize(
    ("source", "value", "match"),
    [
        ("event_slot_index", None, "event_slot_index"),
        ("event_slot_index", True, "event_slot_index"),
        ("event_slot_index", -1, "event_slot_index"),
        ("attempt_index", None, "attempt_index"),
        ("attempt_index", True, "attempt_index"),
        ("attempt_index", -1, "attempt_index"),
        ("random_seed", True, "random_seed"),
        ("random_seed", -1, "random_seed"),
    ],
)
def test_restore_generated_event_rejects_invalid_identity_indices(
    source: str, value: object, match: str
) -> None:
    from src.generation.sop05_output_loader import restore_generated_event

    record, world = _record_and_world()
    if source == "random_seed":
        tampered = replace(world, random_seed=value)
    else:
        metadata = dict(world.metadata)
        if value is None:
            metadata.pop(source)
        else:
            metadata[source] = value
        tampered = replace(world, metadata=metadata)

    with pytest.raises(ValueError, match=match):
        restore_generated_event(record, tampered, grid=_grid())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_slot_index", 1),
        ("attempt_index", 1),
        ("conflict_index", 5),
        ("conflict_time_s", 1.2),
    ],
)
def test_restore_generated_event_rejects_recomputed_event_identity_drift(
    field: str, value: object
) -> None:
    from src.generation.sop05_output_loader import restore_generated_event

    record, world = _record_and_world()
    metadata = dict(world.metadata)
    metadata[field] = value
    if field in {"conflict_index", "conflict_time_s"}:
        if field == "conflict_index":
            metadata["conflict_time_s"] = (int(value) + 1) * 0.2
        else:
            metadata["conflict_index"] = int(round(float(value) / 0.2)) - 1
        provenance = dict(metadata["target_provenance"])
        provenance["conflict_time_s"] = float(metadata["conflict_time_s"])
        provenance["source_conflict_anchor_time_s"] = (
            1.4 + float(metadata["conflict_time_s"])
        )
        provenance["conflict_point"] = [
            float(record.future_poses[int(metadata["conflict_index"]), 0]),
            float(record.future_poses[int(metadata["conflict_index"]), 1]),
        ]
        metadata["target_provenance"] = provenance

    with pytest.raises(ValueError, match="generated_event_id"):
        restore_generated_event(
            record, replace(world, metadata=metadata), grid=_grid()
        )


def test_restore_generated_event_rejects_resealed_target_future_with_stale_world_id(
) -> None:
    from src.generation.sop05_output_loader import restore_generated_event

    record, world = _record_and_world()
    changed_future = record.future_poses.copy()
    changed_future[-1, 1] = np.float32(0.25)
    changed_record = create_event_target_motion_record(
        generated_event_id=record.generated_event_id,
        world_id=record.world_id,
        base_state_id=record.base_state_id,
        trajectory_id=record.trajectory_id,
        target_dynamic_object_id=record.target_dynamic_object_id,
        source_snippet_id=record.source_snippet_id,
        source_object_id=record.source_object_id,
        object_type=record.object_type,
        footprint_spec=record.footprint_spec,
        footprint_spec_digest=record.footprint_spec_digest,
        target_type_policy_digest=record.target_type_policy_digest,
        history_poses=record.history_poses,
        current_pose=record.current_pose,
        future_poses=changed_future,
    )
    changed_metadata = {
        **world.metadata,
        **build_event_target_motion_world_metadata(changed_record),
    }
    changed_world = replace(
        world,
        dynamic_object_trajectories={
            **world.dynamic_object_trajectories,
            record.target_dynamic_object_id: changed_future,
        },
        metadata=changed_metadata,
    )

    with pytest.raises(ValueError, match="world_id"):
        restore_generated_event(changed_record, changed_world, grid=_grid())


def test_restore_generated_event_rejects_resealed_fake_generated_identity() -> None:
    from src.generation.sop05_output_loader import restore_generated_event

    record, world = _record_and_world()
    fake_event_id = "event-" + "f" * 32
    fake_world_id = "world-" + "e" * 32
    fake_record = create_event_target_motion_record(
        generated_event_id=fake_event_id,
        world_id=fake_world_id,
        base_state_id=record.base_state_id,
        trajectory_id=record.trajectory_id,
        target_dynamic_object_id=record.target_dynamic_object_id,
        source_snippet_id=record.source_snippet_id,
        source_object_id=record.source_object_id,
        object_type=record.object_type,
        footprint_spec=record.footprint_spec,
        footprint_spec_digest=record.footprint_spec_digest,
        target_type_policy_digest=record.target_type_policy_digest,
        history_poses=record.history_poses,
        current_pose=record.current_pose,
        future_poses=record.future_poses,
    )
    fake_metadata = {
        **world.metadata,
        **build_event_target_motion_world_metadata(fake_record),
    }
    fake_world = replace(
        world,
        world_id=fake_world_id,
        metadata=fake_metadata,
    )

    with pytest.raises(ValueError, match="generated_event_id"):
        restore_generated_event(fake_record, fake_world, grid=_grid())


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("event_kind", "environment", "event_kind"),
        ("conflict_index", 5, "conflict index/time"),
        ("visibility_sequence", [0] * 16, "boolean"),
        (
            "target_visibility_history_layout",
            "wrong-layout",
            "visibility history layout",
        ),
    ],
)
def test_restore_generated_event_rejects_event_metadata_drift(
    key: str, value: object, match: str
) -> None:
    from src.generation.sop05_output_loader import restore_generated_event

    record, world = _record_and_world()
    metadata = dict(world.metadata)
    metadata[key] = value
    tampered = replace(world, metadata=metadata)

    with pytest.raises(ValueError, match=match):
        restore_generated_event(record, tampered, grid=_grid())


def test_restore_generated_event_rejects_target_provenance_drift() -> None:
    from src.generation.sop05_output_loader import restore_generated_event

    record, world = _record_and_world()
    metadata = dict(world.metadata)
    provenance = dict(metadata["target_provenance"])
    provenance["time_scale"] = 1.2
    metadata["target_provenance"] = provenance

    with pytest.raises(ValueError, match="time_scale"):
        restore_generated_event(
            record, replace(world, metadata=metadata), grid=_grid()
        )


def test_restore_generated_event_rejects_target_provenance_seed_drift() -> None:
    from src.generation.sop05_output_loader import restore_generated_event

    record, world = _record_and_world()
    metadata = dict(world.metadata)
    provenance = dict(metadata["target_provenance"])
    provenance["seed"] = world.random_seed + 1
    metadata["target_provenance"] = provenance

    with pytest.raises(ValueError, match="random_seed"):
        restore_generated_event(
            record, replace(world, metadata=metadata), grid=_grid()
        )


@pytest.mark.parametrize(
    "case",
    [
        "environment_without_occluder",
        "structural_with_occluder",
        "mixed_without_structural_sensor",
        "occluder_id_mismatch",
    ],
)
def test_restore_generated_event_rejects_invalid_event_skeleton(case: str) -> None:
    from src.generation.sop05_output_loader import restore_generated_event

    if case == "environment_without_occluder":
        record, world = _record_and_world(0, "environment")
        world = replace(
            world,
            occluders=(),
            blind_spot_config={
                "kind": "environment",
                "structural": None,
                "occluder_ids": [],
            },
        )
    elif case == "structural_with_occluder":
        record, world = _record_and_world(0, "structural")
        _, environment = _record_and_world(1, "environment")
        occluders = environment.occluders
        world = replace(
            world,
            occluders=occluders,
            blind_spot_config={
                **world.blind_spot_config,
                "occluder_ids": [occluders[0]["occluder_id"]],
            },
        )
    elif case == "mixed_without_structural_sensor":
        record, world = _record_and_world(0, "mixed")
        world = replace(
            world,
            blind_spot_config={
                **world.blind_spot_config,
                "structural": None,
            },
        )
    else:
        record, world = _record_and_world(0, "environment")
        world = replace(
            world,
            blind_spot_config={
                **world.blind_spot_config,
                "occluder_ids": ["different-occluder-id"],
            },
        )

    with pytest.raises(ValueError, match="occluder|structural|event skeleton"):
        restore_generated_event(record, world, grid=_grid())


def test_restore_generated_event_rejects_conflict_point_drift() -> None:
    from src.generation.sop05_output_loader import restore_generated_event

    record, world = _record_and_world()
    metadata = dict(world.metadata)
    provenance = dict(metadata["target_provenance"])
    provenance["conflict_point"] = [0.75, 0.0]
    metadata["target_provenance"] = provenance

    with pytest.raises(ValueError, match="conflict_point"):
        restore_generated_event(
            record, replace(world, metadata=metadata), grid=_grid()
        )


def test_load_complete_sop05_events_validates_evidence_and_restores(
    tmp_path: Path,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    record, _, publication_digest = _write_complete_publication(root)
    loaded = load_complete_sop05_events(
        root,
        grid=_grid(),
        expected_publication_semantic_digest=publication_digest,
    )

    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert loaded.run_id == manifest["run_id"]
    assert loaded.split == "train"
    assert loaded.publication_semantic_digest == publication_digest
    assert len(loaded.events) == 10
    assert loaded.events_by_id[record.generated_event_id].generated_event_id == (
        record.generated_event_id
    )
    assert loaded.shard.summary["record_count"] == 10
    assert {event.event_kind for event in loaded.events} == {"structural"}

    report = json.loads(
        (root / "pair_generation_reports.jsonl").read_text(encoding="utf-8")
    )
    expected_ids = tuple(
        sorted(
            (
                item["generated_event_id"]
                for item in report["accepted_events"]
            ),
            key=lambda event_id: _selection_key_for_id(23, event_id),
        )[:10]
    )
    assert tuple(
        sorted(event.generated_event_id for event in loaded.events)
    ) == tuple(sorted(expected_ids))


def test_load_complete_sop05_events_requires_external_publication_digest(
    tmp_path: Path,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _write_complete_publication(root)

    with pytest.raises(TypeError, match="expected_publication_semantic_digest"):
        load_complete_sop05_events(root, grid=_grid())


@pytest.mark.parametrize(
    "invalid_digest",
    [None, True, "a" * 63, "A" * 64, "g" * 64],
)
def test_load_complete_sop05_events_rejects_invalid_external_publication_digest(
    tmp_path: Path, invalid_digest: object
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _write_complete_publication(root)

    with pytest.raises(ValueError, match="expected_publication_semantic_digest"):
        load_complete_sop05_events(
            root,
            grid=_grid(),
            expected_publication_semantic_digest=invalid_digest,
        )


def test_load_complete_sop05_events_rejects_fully_resealed_content_with_original_anchor(
    tmp_path: Path,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _, _, trusted_digest = _write_complete_publication(root)
    summary_path = root / "generation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["attempted_count"] += 1
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _reseal_publication(root)
    assert _fixture_publication_digest(root) != trusted_digest

    with pytest.raises(ValueError, match="publication semantic digest"):
        load_complete_sop05_events(
            root,
            grid=_grid(),
            expected_publication_semantic_digest=trusted_digest,
        )


@pytest.mark.parametrize(
    ("selected_kind", "accepted_quota"),
    [("environment", 7), ("mixed", 3)],
)
def test_load_complete_sop05_events_allows_any_kind_to_fill_total_quota(
    tmp_path: Path,
    selected_kind: str,
    accepted_quota: int,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _, _, publication_digest = _write_complete_publication(
        root,
        accepted_quota=accepted_quota,
        selected_kind=selected_kind,
    )

    loaded = load_complete_sop05_events(
        root,
        grid=_grid(),
        expected_publication_semantic_digest=publication_digest,
    )

    assert len(loaded.events) == accepted_quota
    assert {event.event_kind for event in loaded.events} == {selected_kind}
    assert loaded.generation_summary["quota_met"] is True
    assert "required_event_kind_counts" not in loaded.generation_summary
    assert "quota_deficits" not in loaded.generation_summary


@pytest.mark.parametrize(
    ("drift", "match"),
    [
        ("input_lock", "input_lock"),
        ("runtime", "runtime"),
        ("scientific_request", "scientific_request|accepted_quota|run_id"),
        ("base_config_digest", "base_config_sha256|run_id"),
        ("generator_semantic_digest", "generator_config_semantic_digest|run_id"),
        ("target_policy_digest", "target_type_policy_digest|run_id"),
        ("run_id", "run_id"),
    ],
)
def test_load_complete_sop05_events_rejects_resealed_manifest_identity_drift(
    tmp_path: Path,
    drift: str,
    match: str,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _write_complete_publication(root)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if drift == "input_lock":
        manifest["input_lock"] = 7
    elif drift == "runtime":
        manifest["runtime"] = "invalid-runtime"
    elif drift == "scientific_request":
        manifest["scientific_request"]["accepted_quota"] = 9
    elif drift == "base_config_digest":
        manifest["scientific_request"]["base_config_sha256"] = "e" * 64
    elif drift == "generator_semantic_digest":
        manifest["scientific_request"][
            "generator_config_semantic_digest"
        ] = "e" * 32
    elif drift == "target_policy_digest":
        manifest["scientific_request"]["target_type_policy_digest"] = "e" * 32
    else:
        manifest["run_id"] = f"sop05-run-{'f' * 32}"
        summary_path = root / "generation_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["run_id"] = manifest["run_id"]
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _reseal_publication(root)

    with pytest.raises(ValueError, match=match):
        load_complete_sop05_events(
            root,
            grid=_grid(),
            expected_publication_semantic_digest=(
                _fixture_publication_digest(root)
            ),
        )


@pytest.mark.parametrize(
    "drift",
    [
        "missing_schema",
        "extra_event_id",
        "accepted_event_schema",
        "accepted_event_kind",
        "selection_version",
        "schedule_rank",
        "pair_summary",
        "global_summary",
    ],
)
def test_load_complete_sop05_events_rejects_resealed_pair_report_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _write_complete_publication(root)
    report_path = root / "pair_generation_reports.jsonl"
    row = json.loads(report_path.read_text(encoding="utf-8"))
    if drift == "missing_schema":
        row = {"accepted_events": row["accepted_events"]}
    elif drift == "extra_event_id":
        row["accepted_events"].append(
            {
                "generated_event_id": "event-not-generated",
                "event_kind": "environment",
            }
        )
    elif drift == "accepted_event_schema":
        del row["accepted_events"][0]["event_kind"]
    elif drift == "accepted_event_kind":
        row["accepted_events"][0]["event_kind"] = "environment"
    elif drift == "selection_version":
        row["selection_version"] = "sop05_kind_quota_selection_v1"
    elif drift == "schedule_rank":
        row["rank"] = 1
    elif drift == "pair_summary":
        row["summary"]["accepted_count"] = 9
    else:
        summary_path = root / "generation_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["processed_pair_count"] = 2
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    report_path.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _reseal_publication(root)

    with pytest.raises(
        ValueError,
        match=(
            "pair generation report|pair report|accepted_count|"
            "generation summary|selection"
        ),
    ):
        load_complete_sop05_events(
            root,
            grid=_grid(),
            expected_publication_semantic_digest=(
                _fixture_publication_digest(root)
            ),
        )


def test_load_complete_sop05_events_rejects_resealed_better_fake_accepted_id(
    tmp_path: Path,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _, _, publication_digest = _write_complete_publication(root)
    loaded = load_complete_sop05_events(
        root,
        grid=_grid(),
        expected_publication_semantic_digest=publication_digest,
    )
    selected_ids = {event.generated_event_id for event in loaded.events}

    report_path = root / "pair_generation_reports.jsonl"
    row = json.loads(report_path.read_text(encoding="utf-8"))
    accepted_events = row["accepted_events"]
    accepted_ids = [
        event["generated_event_id"] for event in accepted_events
    ]
    replaced_event = next(
        event
        for event in accepted_events
        if event["generated_event_id"] not in selected_ids
    )
    worst_selected_key = max(
        _selection_key_for_id(23, event_id)
        for event_id in selected_ids
    )
    injected_id = next(
        candidate
        for index in range(10_000)
        if (candidate := f"event-injected-global-{index}")
        not in accepted_ids
        and _selection_key_for_id(23, candidate) < worst_selected_key
    )
    replaced_event["generated_event_id"] = injected_id
    report_path.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _reseal_publication(root)

    with pytest.raises(ValueError, match="selected.*selection|selection.*selected"):
        load_complete_sop05_events(
            root,
            grid=_grid(),
            expected_publication_semantic_digest=(
                _fixture_publication_digest(root)
            ),
        )


def test_load_complete_sop05_events_rejects_resealed_worse_valid_shard_id(
    tmp_path: Path,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _write_complete_publication(root)
    report = json.loads(
        (root / "pair_generation_reports.jsonl").read_text(encoding="utf-8")
    )
    accepted = report["accepted_events"]
    ranked = sorted(
        (item["generated_event_id"] for item in accepted),
        key=lambda event_id: _selection_key_for_id(23, event_id),
    )
    selected_ids = ranked[:10]
    selected_kind = next(
        item["event_kind"]
        for item in accepted
        if item["generated_event_id"] == selected_ids[0]
    )
    worse_valid_id = next(
        item["generated_event_id"]
        for item in accepted
        if item["generated_event_id"] not in selected_ids
        and item["event_kind"] == selected_kind
    )
    tampered_ids = selected_ids[:-1] + [worse_valid_id]
    _rewrite_selected_shard(root, tampered_ids)

    with pytest.raises(ValueError, match="selected.*selection|selection.*selected"):
        load_complete_sop05_events(
            root,
            grid=_grid(),
            expected_publication_semantic_digest=(
                _fixture_publication_digest(root)
            ),
        )


@pytest.mark.parametrize(
    ("artifact", "field", "value"),
    [
        (
            "run_manifest.json",
            "required_event_kind_counts",
            {"environment": 6, "structural": 3, "mixed": 1},
        ),
        (
            "generation_summary.json",
            "required_event_kind_counts",
            {"environment": 6, "structural": 3, "mixed": 1},
        ),
        (
            "generation_summary.json",
            "quota_deficits",
            {"environment": 0, "structural": 0, "mixed": 0},
        ),
    ],
)
def test_load_complete_sop05_events_rejects_resealed_hard_quota_fields(
    tmp_path: Path,
    artifact: str,
    field: str,
    value: object,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _write_complete_publication(root)
    path = root / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    if artifact == "run_manifest.json":
        payload["scientific_request"][field] = value
    else:
        payload[field] = value
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _reseal_publication(root)

    with pytest.raises(ValueError, match="keys do not match|frozen contract"):
        load_complete_sop05_events(
            root,
            grid=_grid(),
            expected_publication_semantic_digest=(
                _fixture_publication_digest(root)
            ),
        )


@pytest.mark.parametrize(
    ("artifact", "field", "old_value"),
    [
        ("run_manifest.json", "manifest_version", "sop05_run_manifest_v1"),
        ("run_manifest.json", "producer_version", "sop05_generation_run_v2"),
        (
            "run_manifest.json",
            "selection_version",
            "sop05_kind_quota_selection_v1",
        ),
        (
            "generation_summary.json",
            "summary_version",
            "sop05_generation_summary_v1",
        ),
    ],
)
def test_load_complete_sop05_events_rejects_resealed_old_contract_versions(
    tmp_path: Path,
    artifact: str,
    field: str,
    old_value: str,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _write_complete_publication(root)
    path = root / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    if field == "selection_version":
        payload["scientific_request"][field] = old_value
        payload["run_id"] = _run_id(payload)
        summary_path = root / "generation_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["run_id"] = payload["run_id"]
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        payload[field] = old_value
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _reseal_publication(root)

    with pytest.raises(ValueError, match="unsupported|completion mismatch"):
        load_complete_sop05_events(
            root,
            grid=_grid(),
            expected_publication_semantic_digest=(
                _fixture_publication_digest(root)
            ),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("completion_policy", "sop04_audited_bank_v1", "completion_policy"),
        ("trajectory_bank_version", "sop04_audited_bank_v1", "bank version"),
        (
            "pose_time_layout_version",
            "legacy_t0_to_horizon_minus_dt_v0",
            "layout",
        ),
        ("trajectory_steps", 14, "trajectory_steps"),
        ("dt_s", 0.25, "dt_s"),
        ("first_pose_time_s", 0.0, "first_pose_time_s"),
        ("last_pose_time_s", 2.8, "last_pose_time_s"),
        ("pose_time_offsets_sha256", "0" * 64, "offsets"),
        ("bank_semantic_digest_sha256", "bad", "semantic"),
        ("external_handoff_digest_sha256", "bad", "handoff"),
    ],
)
def test_load_complete_sop05_events_rejects_resealed_stale_sop04_contract(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _write_complete_publication(root)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_lock"]["sop04"][field] = value
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _reseal_publication(root)

    with pytest.raises(ValueError, match=match):
        load_complete_sop05_events(
            root,
            grid=_grid(),
            expected_publication_semantic_digest=(
                _fixture_publication_digest(root)
            ),
        )


def test_load_complete_sop05_events_rejects_payload_checksum_tamper(
    tmp_path: Path,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _, _, publication_digest = _write_complete_publication(root)
    with (root / "pair_generation_reports.jsonl").open("a") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="checksum"):
        load_complete_sop05_events(
            root,
            grid=_grid(),
            expected_publication_semantic_digest=publication_digest,
        )


def test_load_complete_sop05_events_rejects_missing_completion_marker(
    tmp_path: Path,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _, _, publication_digest = _write_complete_publication(root)
    (root / ".producer-complete").unlink()

    with pytest.raises(ValueError, match="producer-complete"):
        load_complete_sop05_events(
            root,
            grid=_grid(),
            expected_publication_semantic_digest=publication_digest,
        )


def test_load_complete_sop05_events_rejects_signed_extra_root_artifact(
    tmp_path: Path,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _write_complete_publication(root)
    (root / "unexpected.json").write_text("{}\n", encoding="utf-8")
    _reseal_publication(root)

    with pytest.raises(ValueError, match="root layout"):
        load_complete_sop05_events(
            root,
            grid=_grid(),
            expected_publication_semantic_digest=(
                _fixture_publication_digest(root)
            ),
        )
