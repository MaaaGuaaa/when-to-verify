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
    event_kind: str = "environment",
    *,
    generator_algorithm_version: str = "blind_reachability_quota_first_v3",
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
    source_recording_id = "recording"
    source_session_id = "session-loader-fixture"
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
        source_recording_id=source_recording_id,
        source_session_id=source_session_id,
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
        source_recording_id=source_recording_id,
        source_session_id=source_session_id,
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
        "transform_algorithm_version": "reachability_candidate_se2_v2",
        "transform_id": f"transform-loader-fixture-{suffix}",
        "reachability_candidate_id": f"candidate-loader-fixture-{suffix}",
        "reachability_algorithm_version": "blind_reachability_quota_first_v3",
        "reachable_arc_schedule_version": "reachable_arc_schedule_v1",
        "motion_snippet_layout_version": MOTION_SNIPPET_LAYOUT_VERSION,
        "snippet_id": record.source_snippet_id,
        "source_recording_id": source_recording_id,
        "source_session_id": source_session_id,
        "source_object_id": record.source_object_id,
        "source_body_name": "person",
        "raw_role": "person",
        "geometry_source": "thor_marker_extent",
        "orientation_source": "velocity",
        "source_start_timestamp": 0.0,
        "source_current_index": 7,
        "source_current_time_s": 1.4,
        "source_anchor_index": 12,
        "source_anchor_time_s": 2.4,
        "source_delta_xy": [0.5, 0.0],
        "candidate_current_xy": [0.0, 0.0],
        "conflict_point": [0.5, 0.0],
        "rotation_rad": 0.0,
        "desired_crossing_direction": [1.0, 0.0],
        "crossing_side": -1,
        "angle_offset_deg": 0.0,
        "conflict_index": 4,
        "conflict_time_s": 1.0,
        "time_scale": 1.0,
        "future_dt_s": 0.2,
        "future_steps": 15,
        "base_state_id": record.base_state_id,
        "trajectory_id": record.trajectory_id,
        "target_type_policy_digest": record.target_type_policy_digest,
        "footprint_spec_digest": record.footprint_spec_digest,
        "seed": 17 + event_index,
        "context_object_ids": [],
    }
    visibility = [False, False, False, True, True] + [True] * 11
    proposal_id = f"proposal-{event_index:08d}"
    blind_region_id = f"blind-loader-fixture-{suffix}"
    occluders = (
        {
            "occluder_id": proposal_id,
            "proposal_id": proposal_id,
            "type": "pillar",
            "polygon_xy": [
                [0.1, -0.2],
                [0.3, -0.2],
                [0.3, 0.2],
                [0.1, 0.2],
            ],
            "height_m": 2.0,
        },
    )
    metadata = {
        **build_event_target_motion_world_metadata(record),
        "schema_version": "3.0.0",
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
        "causal_occluder_proposal_id": proposal_id,
        "blind_region_id": blind_region_id,
        "reachability_candidate_id": provenance[
            "reachability_candidate_id"
        ],
        "reachability_transform_id": provenance["transform_id"],
        "exact_validation_id": f"exact-loader-fixture-{suffix}",
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
            "kind": "environment",
            "occluder_ids": [proposal_id],
            "blind_region_digest": blind_region_id,
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
    marker["marker_version"] = "sop05_producer_complete_v3"
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


def _selection_key_for_id(seed: int, event_id: str) -> tuple[str, ...]:
    payload = _canonical_json_bytes(
        [
            "sop05_diversity_total_selection_v1",
            seed,
            "base-loader-fixture",
            "trajectory-loader-fixture",
            event_id,
        ]
    )
    return (
        hashlib.blake2b(payload, digest_size=16).hexdigest(),
        "base-loader-fixture",
        "trajectory-loader-fixture",
        event_id,
    )


def _accepted_event_row(record: object, world: OracleWorld) -> dict[str, object]:
    metadata = world.metadata
    provenance = metadata["target_provenance"]
    return {
        "generated_event_id": record.generated_event_id,
        "event_kind": "environment",
        "object_type": record.object_type,
        "occluder_type": world.occluders[0]["type"],
        "crossing_side": provenance["crossing_side"],
        "conflict_index": metadata["conflict_index"],
        "causal_occluder_proposal_id": metadata[
            "causal_occluder_proposal_id"
        ],
        "blind_region_id": metadata["blind_region_id"],
        "reachability_candidate_id": metadata["reachability_candidate_id"],
        "reachability_transform_id": metadata["reachability_transform_id"],
        "exact_validation_id": metadata["exact_validation_id"],
    }


def _write_complete_publication(
    root: Path,
    *,
    accepted_quota: int = 10,
    selected_kind: str = "environment",
) -> tuple[object, OracleWorld, str]:
    if selected_kind != "environment":
        raise ValueError("formal v5 fixture is environment-only")
    requested_count = 40
    if accepted_quota + 1 > requested_count:
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
    pairs = primary_pairs
    records = [record for record, _ in pairs]
    event_kinds = [world.metadata["event_kind"] for _, world in pairs]
    generated_counts = {"environment": len(event_kinds)}
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
        "schema_version: '3.0.0'\n"
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
    accepted_rows = sorted(
        (
            _accepted_event_row(item, item_world)
            for item, item_world in pairs
        ),
        key=lambda row: (
            row["causal_occluder_proposal_id"],
            row["reachability_candidate_id"],
            row["reachability_transform_id"],
        ),
    )
    proposal_ids = [
        str(row["causal_occluder_proposal_id"]) for row in accepted_rows
    ] + [
        f"proposal-rejected-{index:08d}"
        for index in range(requested_count - generated_count)
    ]
    pair_summary = {
        "schema_version": "3.0.0",
        "seed": 17,
        "requested_event_count": requested_count,
        "accepted_count": generated_count,
        "rejected_count": rejected_count,
        "unaccepted_event_count": rejected_count,
        "attempt_index_start": 0,
        "attempt_index_stop_exclusive": requested_count,
        "rejection_reasons": rejection_reasons,
        "obstacle_proposal_count": requested_count,
        "obstacle_proposal_rejected_count": 0,
        "obstacle_proposal_passed_count": requested_count,
        "transform_candidate_count": requested_count,
        "transform_rejected_count": 0,
        "chord_certified_count": requested_count,
        "chord_unresolved_count": 0,
        "exact_validation_count": requested_count,
        "exact_validation_accepted_count": generated_count,
        "exact_validation_rejected_count": rejected_count,
        "proposal_ids": proposal_ids,
        "reachability_candidate_ids": [
            row["reachability_candidate_id"] for row in accepted_rows
        ],
        "reachability_transform_ids": [
            row["reachability_transform_id"] for row in accepted_rows
        ],
        "exact_validation_ids": [
            row["exact_validation_id"] for row in accepted_rows
        ],
        "robot_sweep_cache": {
            "size": 1,
            "hits": 0,
            "misses": 1,
            "builds": 1,
        },
        "target_type_policy": target_type_policy,
        "target_type_policy_digest": target_type_policy_digest,
        "generator_config_digest": generator_config_digest,
        "generator_algorithm_version": "blind_reachability_quota_first_v3",
        "production_event_kind": "environment",
    }
    (root / "pair_generation_reports.jsonl").write_text(
        json.dumps(
            {
                "report_version": "sop05_pair_generation_report_v4",
                "selection_version": "sop05_diversity_total_selection_v1",
                "rank": 0,
                "state_id": record.base_state_id,
                "trajectory_id": record.trajectory_id,
                "seed": 17,
                "allocated_cpu_seconds": 12.5,
                "summary": pair_summary,
                "accepted_events": accepted_rows,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    selected_counts = {"environment": len(selected_pairs)}
    canonical_candidate_order = [
        {
            "base_state_id": item.base_state_id,
            "trajectory_id": item.trajectory_id,
            "generated_event_id": item.generated_event_id,
        }
        for item in sorted(
            records,
            key=lambda value: (
                value.base_state_id,
                value.trajectory_id,
                value.generated_event_id,
            ),
        )
    ]
    selected_event_ids = [item.generated_event_id for item in selected_records]
    summary = {
        "summary_version": "sop05_generation_summary_v3",
        "run_id": None,
        "run_state": "complete",
        "processed_pair_count": 1,
        "requested_event_count": requested_count,
        "generator_accepted_count": generated_count,
        "candidate_count": generated_count,
        "selected_count": accepted_quota,
        "quota_trimmed_count": generated_count - accepted_quota,
        "generated_event_kind_counts": generated_counts,
        "selected_event_kind_counts": selected_counts,
        "quota_met": True,
        "production_event_kind": "environment",
        "canonical_candidate_order": canonical_candidate_order,
        "selected_event_ids": selected_event_ids,
        "rejection_reasons": rejection_reasons,
        "stage_counts": {
            name: pair_summary[name]
            for name in (
                "obstacle_proposal_count",
                "obstacle_proposal_rejected_count",
                "obstacle_proposal_passed_count",
                "transform_candidate_count",
                "transform_rejected_count",
                "chord_certified_count",
                "chord_unresolved_count",
                "exact_validation_count",
                "exact_validation_accepted_count",
                "exact_validation_rejected_count",
            )
        },
        "stage_ids": {
            name: pair_summary[name]
            for name in (
                "proposal_ids",
                "reachability_candidate_ids",
                "reachability_transform_ids",
                "exact_validation_ids",
            )
        },
        "allocated_cpu_seconds": 12.5,
        "generator_invariants": {
            "schema_version": "3.0.0",
            "target_type_policy_digest": target_type_policy_digest,
            "generator_config_digest": generator_config_digest,
            "generator_algorithm_version": "blind_reachability_quota_first_v3",
            "production_event_kind": "environment",
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
        "selection_version": "sop05_diversity_total_selection_v1",
        "base_config_sha256": _sha256(configs / "base.yaml"),
        "generator_config_sha256": _sha256(configs / "generator.yaml"),
        "generator_config_semantic_digest": generator_config_digest,
        "target_type_policy": target_type_policy,
        "target_type_policy_digest": target_type_policy_digest,
        "pair_schedule": schedule,
    }
    manifest = {
        "manifest_version": "sop05_run_manifest_v3",
        "producer_version": "sop05_generation_run_v6",
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
        "marker_version": "sop05_producer_complete_v3",
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
    assert event.event_kind == "environment"
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


@pytest.mark.parametrize(
    "old_version",
    ("joint_occluder_first_v4", "blind_reachability_first_v1"),
)
def test_restore_generated_event_rejects_old_generator_algorithm(
    old_version: str,
) -> None:
    from src.generation.sop05_output_loader import restore_generated_event

    record, world = _record_and_world(
        generator_algorithm_version=old_version
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
        provenance["conflict_index"] = int(metadata["conflict_index"])
        provenance["source_anchor_index"] = 8 + int(
            metadata["conflict_index"]
        )
        provenance["source_anchor_time_s"] = 0.2 * provenance[
            "source_anchor_index"
        ]
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
        ("event_kind", "structural", "event_kind"),
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

    with pytest.raises(ValueError, match="seed"):
        restore_generated_event(
            record, replace(world, metadata=metadata), grid=_grid()
        )


@pytest.mark.parametrize(
    "tamper", ["missing", "changed", "blank", "whitespace"]
)
def test_restore_generated_event_rejects_source_session_provenance_drift(
    tamper: str,
) -> None:
    from src.generation.sop05_output_loader import restore_generated_event

    record, world = _record_and_world()
    metadata = dict(world.metadata)
    provenance = dict(metadata["target_provenance"])
    if tamper == "missing":
        provenance.pop("source_session_id")
    elif tamper == "changed":
        provenance["source_session_id"] = "session-tampered"
    elif tamper == "blank":
        provenance["source_session_id"] = ""
    else:
        provenance["source_session_id"] = "   "
    metadata["target_provenance"] = provenance

    with pytest.raises(ValueError, match="source_session_id|canonical event identity"):
        restore_generated_event(
            record, replace(world, metadata=metadata), grid=_grid()
        )


@pytest.mark.parametrize(
    "case",
    [
        "environment_without_occluder",
        "multiple_occluders",
        "proposal_id_mismatch",
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
                "occluder_ids": [],
                "blind_region_digest": world.blind_spot_config[
                    "blind_region_digest"
                ],
            },
        )
    elif case == "multiple_occluders":
        record, world = _record_and_world(0, "environment")
        extra = {**world.occluders[0], "occluder_id": "extra", "proposal_id": "extra"}
        world = replace(
            world,
            occluders=(*world.occluders, extra),
            blind_spot_config={
                **world.blind_spot_config,
                "occluder_ids": [world.occluders[0]["occluder_id"], "extra"],
            },
        )
    elif case == "proposal_id_mismatch":
        record, world = _record_and_world(0, "environment")
        changed = {**world.occluders[0], "proposal_id": "different-proposal"}
        world = replace(
            world,
            occluders=(changed,),
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

    with pytest.raises(ValueError, match="occluder|event skeleton"):
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
    assert {event.event_kind for event in loaded.events} == {"environment"}

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


def test_loader_formal_versions_are_exact() -> None:
    from src.generation import sop05_output_loader as loader

    assert loader.SOP05_RUN_MANIFEST_VERSION == "sop05_run_manifest_v3"
    assert loader.SOP05_GENERATION_SUMMARY_VERSION == (
        "sop05_generation_summary_v3"
    )
    assert loader.SOP05_COMPLETION_MARKER_VERSION == (
        "sop05_producer_complete_v3"
    )
    assert loader.SOP05_RUN_PRODUCER_VERSION == "sop05_generation_run_v6"
    assert loader.SOP05_PAIR_REPORT_VERSION == (
        "sop05_pair_generation_report_v4"
    )
    assert loader.SOP05_TOTAL_QUOTA_SELECTION_VERSION == (
        "sop05_diversity_total_selection_v1"
    )


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("pair_stage_count", "transform conservation"),
        ("global_stage_count", "stage counts"),
        ("global_stage_id", "stage IDs"),
        ("candidate_order", "candidate order"),
        ("causal_identity", "causal_occluder_proposal_id|causal"),
        ("cpu_accounting", "CPU accounting"),
    ],
)
def test_load_complete_rejects_resealed_v5_evidence_tamper(
    tmp_path: Path,
    tamper: str,
    match: str,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _write_complete_publication(root)
    report_path = root / "pair_generation_reports.jsonl"
    summary_path = root / "generation_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if tamper == "pair_stage_count":
        report["summary"]["chord_certified_count"] -= 1
    elif tamper == "global_stage_count":
        summary["stage_counts"]["obstacle_proposal_count"] += 1
    elif tamper == "global_stage_id":
        summary["stage_ids"]["proposal_ids"][0] += "-tampered"
    elif tamper == "candidate_order":
        report["accepted_events"] = list(
            reversed(report["accepted_events"])
        )
    elif tamper == "causal_identity":
        selected_id = summary["selected_event_ids"][0]
        selected_row = next(
            row
            for row in report["accepted_events"]
            if row["generated_event_id"] == selected_id
        )
        old_id = selected_row["causal_occluder_proposal_id"]
        new_id = old_id + "-tampered"
        selected_row["causal_occluder_proposal_id"] = new_id
        report["summary"]["proposal_ids"] = [
            new_id if value == old_id else value
            for value in report["summary"]["proposal_ids"]
        ]
        summary["stage_ids"]["proposal_ids"] = [
            new_id if value == old_id else value
            for value in summary["stage_ids"]["proposal_ids"]
        ]
        report["accepted_events"].sort(
            key=lambda row: (
                row["causal_occluder_proposal_id"],
                row["reachability_candidate_id"],
                row["reachability_transform_id"],
            )
        )
    else:
        summary["allocated_cpu_seconds"] += 1.0
    report_path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
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


def test_load_complete_rejects_resealed_v2_completion_marker(
    tmp_path: Path,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _write_complete_publication(root)
    marker_path = root / ".producer-complete"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["marker_version"] = "sop05_producer_complete_v2"
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="producer-complete marker version"):
        load_complete_sop05_events(
            root,
            grid=_grid(),
            expected_publication_semantic_digest=marker[
                "publication_semantic_digest"
            ],
        )


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
    summary["candidate_count"] += 1
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


@pytest.mark.parametrize("accepted_quota", [7, 3])
def test_load_complete_sop05_events_fills_total_environment_quota(
    tmp_path: Path,
    accepted_quota: int,
) -> None:
    from src.generation.sop05_output_loader import load_complete_sop05_events

    root = tmp_path / "run"
    _, _, publication_digest = _write_complete_publication(
        root,
        accepted_quota=accepted_quota,
    )

    loaded = load_complete_sop05_events(
        root,
        grid=_grid(),
        expected_publication_semantic_digest=publication_digest,
    )

    assert len(loaded.events) == accepted_quota
    assert {event.event_kind for event in loaded.events} == {"environment"}
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
        row["accepted_events"][0]["event_kind"] = "structural"
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
                "generation summary|selection|formal v5|environment"
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
        ("run_manifest.json", "manifest_version", "sop05_run_manifest_v2"),
        ("run_manifest.json", "producer_version", "sop05_generation_run_v5"),
        (
            "pair_generation_reports.jsonl",
            "report_version",
            "sop05_pair_generation_report_v3",
        ),
        (
            "run_manifest.json",
            "selection_version",
            "sop05_total_quota_selection_v1",
        ),
        (
            "generation_summary.json",
            "summary_version",
            "sop05_generation_summary_v2",
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
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if path.suffix == ".jsonl"
        else json.dumps(payload, sort_keys=True, indent=2)
    )
    path.write_text(encoded + "\n", encoding="utf-8")
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
