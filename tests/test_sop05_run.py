"""SOP-05 run orchestration and atomic publication tests."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import pickle
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.contracts import OracleWorld
from src.generation.dynamic_object_transplant import (
    MOTION_SNIPPET_CURRENT_INDEX,
    MOTION_SNIPPET_CURRENT_TIME_S,
    MOTION_SNIPPET_LAYOUT_VERSION,
    TransplantedDynamicObject,
)
from src.generation.event_sampler import (
    EventGenerationReport,
    GeneratedEvent,
    SOP05_GENERATOR_ALGORITHM_VERSION,
    _generator_digest,
    compute_generated_event_id,
    compute_generated_world_id,
)
from src.generation.event_target_motion_shard import (
    EVENT_TARGET_MOTION_LAYOUT_VERSION,
    build_event_target_motion_world_metadata,
    compute_footprint_spec_digest,
    compute_motion_array_digest,
    create_event_target_motion_record,
    load_event_target_motion_shard,
)
from src.generation.sop05_input_adapter import ProducerEvidence, StablePair


ROOT = Path(__file__).resolve().parents[1]
_REAL_GIT_EXECUTABLE = Path(
    "/home/home/ccnt_zq/zq_zhouyiqun/.local/git/bin/git"
)
_CLEAN_SOURCE_IDENTITY = {
    "version": "sop05_producer_source_identity_v1",
    "git_commit": "3" * 40,
    "worktree_state": "clean",
    "dirty_tree_sha256": None,
}


def _sut():
    return importlib.import_module("src.generation.sop05_run")


def _publication_identity_sut():
    return importlib.import_module(
        "src.generation.sop05_publication_identity"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_outer_checksums(root: Path) -> dict[str, str]:
    lines = (root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    entries = {
        name: digest
        for line in lines
        for digest, name in (line.split("  ", 1),)
    }
    expected_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"checksums.sha256", ".producer-complete"}
    }
    assert set(entries) == expected_paths
    assert entries == {name: _sha256(root / name) for name in sorted(entries)}
    return entries


def _evidence(root: Path, digit: str) -> ProducerEvidence:
    completion_policies = {
        "1": "sop03_complete_marker_v1",
        "2": "sop04_audited_bank_v1",
    }
    return ProducerEvidence(
        root=root,
        code_commit=digit * 40,
        checksum_manifest_sha256=digit * 64,
        audit_sha256=("a" if digit != "a" else "b") * 64,
        completion_policy=completion_policies[digit],
        payload_checksums={f"payload-{digit}.npz": digit * 64},
    )


def _fixture_git_executable(tmp_path: Path, name: str = "fixture-git") -> Path:
    path = tmp_path / name
    path.write_bytes(b"#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def _publication_identity_fields() -> dict[str, str]:
    return {
        "run_id": "sop05-run-" + "1" * 32,
        "run_manifest_sha256": "2" * 64,
        "checksums_sha256": "3" * 64,
        "target_motion_manifest_digest": "4" * 32,
        "target_motion_payload_semantic_digest": "5" * 32,
    }


def test_publication_semantic_digest_uses_frozen_domain_and_canonical_payload(
) -> None:
    identity = _publication_identity_sut()
    fields = _publication_identity_fields()

    observed = identity.compute_sop05_publication_semantic_digest(**fields)
    canonical_payload = json.dumps(
        fields,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    expected = hashlib.blake2b(
        b"sop05_publication_semantic_digest_v1\0" + canonical_payload,
        digest_size=32,
    ).hexdigest()

    assert identity.SOP05_PUBLICATION_IDENTITY_VERSION == (
        "sop05_publication_semantic_digest_v1"
    )
    assert observed == expected
    assert len(observed) == 64
    assert observed == identity.compute_sop05_publication_semantic_digest(
        **dict(reversed(tuple(fields.items())))
    )
    for field_name, value in fields.items():
        changed = dict(fields)
        replacement = "a" if value[-1] != "a" else "b"
        changed[field_name] = value[:-1] + replacement
        assert (
            identity.compute_sop05_publication_semantic_digest(**changed)
            != observed
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_type"),
    [
        ("run_id", 23, TypeError),
        ("run_id", "sop05-run-" + "1" * 31, ValueError),
        ("run_id", "other-run-" + "1" * 32, ValueError),
        ("run_id", "sop05-run-" + "A" * 32, ValueError),
        ("run_manifest_sha256", None, TypeError),
        ("run_manifest_sha256", "2" * 63, ValueError),
        ("run_manifest_sha256", "G" * 64, ValueError),
        ("checksums_sha256", b"3" * 64, TypeError),
        ("checksums_sha256", "3" * 65, ValueError),
        ("target_motion_manifest_digest", "4" * 31, ValueError),
        ("target_motion_manifest_digest", "Z" * 32, ValueError),
        ("target_motion_payload_semantic_digest", "5" * 33, ValueError),
        ("target_motion_payload_semantic_digest", "F" * 32, ValueError),
    ],
)
def test_publication_semantic_digest_rejects_noncanonical_identity_fields(
    field_name: str,
    invalid_value: object,
    error_type: type[Exception],
) -> None:
    identity = _publication_identity_sut()
    fields: dict[str, object] = _publication_identity_fields()
    fields[field_name] = invalid_value

    with pytest.raises(error_type, match=field_name):
        identity.compute_sop05_publication_semantic_digest(**fields)


def _request(module, tmp_path: Path, **overrides):
    values = {
        "sop03_root": tmp_path / "sop03",
        "sop04_root": tmp_path / "sop04",
        "split": "train",
        "base_config_path": ROOT / "configs/base.yaml",
        "generator_config_path": ROOT / "configs/generator_train.yaml",
        "output_dir": tmp_path / "run",
        "seed": 23,
        "accepted_quota": 10,
        "events_per_pair": 10,
        "max_base_states": 2,
        "trajectory_count": 1,
        "max_pairs": 2,
        "checksum_workers": 2,
        "workers": 2,
        "git_executable": _fixture_git_executable(tmp_path),
    }
    values.update(overrides)
    return module.Sop05RunRequest(**values)


def _install_preflight_inputs(
    monkeypatch,
    module,
    tmp_path: Path,
    *,
    sop03_root: Path | None = None,
    sop04_root: Path | None = None,
    producer_identity: dict[str, object] | None = None,
):
    resolved_sop03_root = sop03_root or tmp_path / "sop03"
    resolved_sop04_root = sop04_root or tmp_path / "sop04"
    sop03 = SimpleNamespace(
        split="train",
        manifest_index={"base-a": object(), "base-b": object()},
        typed_libraries={"human": object()},
        producer_evidence=_evidence(resolved_sop03_root, "1"),
        load_pair=lambda state_id, grid: (
            SimpleNamespace(state_id=state_id),
            SimpleNamespace(base_state_id=state_id),
        ),
    )
    sop04 = SimpleNamespace(
        trajectories=(object(),),
        by_id={"trajectory-a": object()},
        producer_evidence=_evidence(resolved_sop04_root, "2"),
    )
    calls = {"sop03": 0, "sop04": 0, "schedule": 0}

    def load_sop03(root, split, grid, *, checksum_workers):
        calls["sop03"] += 1
        assert Path(root) == resolved_sop03_root
        assert split == "train"
        assert grid.history_steps == 8 and grid.future_steps == 15
        assert checksum_workers == 2
        return sop03

    def load_sop04(root, grid, *, checksum_workers):
        calls["sop04"] += 1
        assert Path(root) == resolved_sop04_root
        assert grid.history_steps == 8 and grid.future_steps == 15
        assert checksum_workers == 2
        return sop04

    def schedule(loaded_sop03, loaded_sop04, **kwargs):
        calls["schedule"] += 1
        assert loaded_sop03 is sop03 and loaded_sop04 is sop04
        assert kwargs == {
            "seed": 23,
            "max_base_states": 2,
            "trajectory_count": 1,
        }
        return (
            StablePair("base-b", "trajectory-a", 102),
            StablePair("base-a", "trajectory-a", 101),
        )

    monkeypatch.setattr(module, "load_sop03_split_inputs", load_sop03)
    monkeypatch.setattr(module, "load_sop04_trajectory_bank", load_sop04)
    monkeypatch.setattr(module, "build_stable_pair_schedule", schedule)
    identity = producer_identity or _CLEAN_SOURCE_IDENTITY
    monkeypatch.setattr(
        module,
        "_load_producer_source_identity",
        lambda git_executable: dict(identity),
        raising=False,
    )
    return calls


def _generated_event(
    event_id: str,
    event_kind: str,
    grid,
    offset: float,
    base_state_id: str,
    *,
    target_type_policy_digest: str,
    target_type_policy: dict[str, object] | None = None,
    generator_config_digest: str | None = None,
    event_index: int = 0,
    attempt_index: int | None = None,
    attempt_seed: int = 17,
    canonical_identity: bool = False,
) -> GeneratedEvent:
    history = np.zeros((8, 3), dtype=np.float32)
    history[:, 0] = np.float32(offset)
    current = history[7].copy()
    future = np.zeros((15, 3), dtype=np.float32)
    future[:, 0] = np.float32(offset + 0.25)
    spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.3},
    }
    spec_digest = compute_footprint_spec_digest(spec)
    target_id = f"target-{event_id}"
    source_snippet_id = f"snippet-{event_id}"
    source_object_id = "recording::human"
    policy = target_type_policy or {
        "whitelist": ["human"],
        "weights": {
            "human": 1.0,
            "carried_object": 0.0,
            "unknown_dynamic": 0.0,
        },
    }
    config_digest = generator_config_digest or "0" * 32
    resolved_attempt_index = (
        event_index if attempt_index is None else attempt_index
    )
    if canonical_identity:
        event_id = compute_generated_event_id(
            generator_algorithm_version=SOP05_GENERATOR_ALGORITHM_VERSION,
            generator_config_digest=config_digest,
            base_state_id=base_state_id,
            trajectory_id="trajectory-a",
            event_index=event_index,
            attempt_index=resolved_attempt_index,
            attempt_seed=attempt_seed,
            event_kind=event_kind,
            conflict_index=4,
            conflict_time_s=1.0,
            target_dynamic_object_id=target_id,
            source_snippet_id=source_snippet_id,
            source_object_id=source_object_id,
            object_type="human",
            footprint_spec=spec,
            footprint_spec_digest=spec_digest,
            target_type_policy_digest=target_type_policy_digest,
            layout_version=EVENT_TARGET_MOTION_LAYOUT_VERSION,
        )
        world_id = compute_generated_world_id(
            generator_algorithm_version=SOP05_GENERATOR_ALGORITHM_VERSION,
            generator_config_digest=config_digest,
            generated_event_id=event_id,
            base_state_id=base_state_id,
            trajectory_id="trajectory-a",
            event_kind=event_kind,
            target_dynamic_object_id=target_id,
            source_snippet_id=source_snippet_id,
            source_object_id=source_object_id,
            object_type="human",
            footprint_spec=spec,
            footprint_spec_digest=spec_digest,
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
    else:
        world_id = f"world-{event_id}"
    record = create_event_target_motion_record(
        generated_event_id=event_id,
        world_id=world_id,
        base_state_id=base_state_id,
        trajectory_id="trajectory-a",
        target_dynamic_object_id=target_id,
        source_snippet_id=source_snippet_id,
        source_object_id=source_object_id,
        object_type="human",
        footprint_spec=spec,
        footprint_spec_digest=spec_digest,
        target_type_policy_digest=target_type_policy_digest,
        history_poses=history,
        current_pose=current,
        future_poses=future,
    )
    target = TransplantedDynamicObject(
        target_dynamic_object_id=target_id,
        source_object_id=source_object_id,
        snippet_id=source_snippet_id,
        object_type="human",
        footprint_spec=spec,
        footprint_spec_digest=spec_digest,
        history_poses=history,
        current_pose=current,
        future_poses=future,
        provenance={},
    )
    provenance = {
        "snippet_id": record.source_snippet_id,
        "source_recording_id": "recording",
        "source_object_id": record.source_object_id,
        "source_body_name": "human",
        "raw_role": "human",
        "geometry_source": "thor_marker_extent",
        "orientation_source": "velocity",
        "target_type_policy_digest": record.target_type_policy_digest,
        "footprint_spec_digest": record.footprint_spec_digest,
        "conflict_time_s": 1.0,
        "conflict_point": [
            float(record.future_poses[4, 0]),
            float(record.future_poses[4, 1]),
        ],
        "crossing_direction": [1.0, 0.0],
        "rotation_rad": 0.0,
        "time_scale": 1.0,
        "motion_snippet_layout_version": MOTION_SNIPPET_LAYOUT_VERSION,
        "source_current_index": MOTION_SNIPPET_CURRENT_INDEX,
        "source_current_time_s": MOTION_SNIPPET_CURRENT_TIME_S,
        "source_conflict_anchor_time_s": (
            MOTION_SNIPPET_CURRENT_TIME_S + 1.0
        ),
        "seed": attempt_seed,
    }
    target = replace(target, provenance=provenance)
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
                "occluder_id": f"occluder-{target_id}",
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
    visibility = [False] * 4 + [True] * 12
    metadata = {
        **build_event_target_motion_world_metadata(record),
        "schema_version": "2.0.0",
        "generator_algorithm_version": SOP05_GENERATOR_ALGORITHM_VERSION,
        "generator_config_digest": config_digest,
        "event_kind": event_kind,
        "dynamic_object_snippet_id": record.source_snippet_id,
        "target_type_policy": policy,
        "conflict_time_s": 1.0,
        "conflict_index": 4,
        "event_slot_index": event_index,
        "attempt_index": resolved_attempt_index,
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
        static_occupancy=np.zeros((grid.height, grid.width), dtype=np.float32),
        dynamic_object_trajectories={target_id: future.copy()},
        dynamic_object_specs={target_id: spec},
        occluders=occluders,
        blind_spot_config={
            "kind": event_kind,
            "structural": structural,
            "occluder_ids": [
                occluder["occluder_id"] for occluder in occluders
            ],
        },
        random_seed=attempt_seed,
        metadata=metadata,
    )
    return GeneratedEvent(
        generated_event_id=event_id,
        event_kind=event_kind,
        world=world,
        target=target,
        target_motion_record=record,
        visibility_sequence=np.array(visibility, dtype=np.bool_),
        target_visibility_history=np.zeros(8, dtype=np.bool_),
        conflict_time_s=1.0,
        conflict_index=4,
    )


def _report(
    prepared,
    seed: int,
    events: tuple[GeneratedEvent, ...],
) -> EventGenerationReport:
    counts = Counter(event.event_kind for event in events)
    accepted = len(events)
    rejected = 10 - accepted
    reasons = {} if not rejected else {"fixture_rejection": rejected}
    stages = {
        "occluder_geometry": 0,
        "target_conditioning": rejected,
        "visibility": 0,
    }
    requested_counts = {
        "environment": 6,
        "structural": 3,
        "mixed": 1,
    }

    def event_kind_bucket(kind: str) -> dict[str, object]:
        requested = requested_counts[kind]
        kind_accepted = counts[kind]
        kind_rejected = requested - kind_accepted
        return {
            "requested": requested,
            "attempted": requested,
            "accepted": kind_accepted,
            "rejected": kind_rejected,
            "request_acceptance_rate": kind_accepted / requested,
            "attempt_acceptance_rate": kind_accepted / requested,
            "rejection_reasons": (
                {"fixture_rejection": kind_rejected}
                if kind_rejected
                else {}
            ),
            "rejection_stage_counts": {
                "occluder_geometry": 0,
                "target_conditioning": kind_rejected,
                "visibility": 0,
            },
        }

    aggregate_bucket = {
        "attempted": 10,
        "accepted": accepted,
        "rejected": rejected,
        "attempt_acceptance_rate": accepted / 10,
        "rejection_reasons": reasons,
    }
    policy = prepared.generator_config["target_type_policy"]
    return EventGenerationReport(
        events=events,
        summary={
            "schema_version": "2.0.0",
            "seed": seed,
            "requested_event_count": 10,
            "complete_joint_candidates_attempted": 10,
            "attempted_count": 10,
            "joint_candidate_attempted_count": 10,
            "attempt_index_start": 0,
            "attempt_index_stop_exclusive": None,
            "accepted_count": accepted,
            "rejected_count": rejected,
            "acceptance_rate": accepted / 10,
            "attempt_acceptance_rate": accepted / 10,
            "request_acceptance_rate": accepted / 10,
            "unaccepted_event_count": rejected,
            "rejection_reasons": reasons,
            "rejection_stage_counts": stages,
            "occluder_candidate_rejection_reasons": {},
            "requested_event_kind_counts": requested_counts,
            "event_kind_counts": {
                "environment": counts["environment"],
                "structural": counts["structural"],
                "mixed": counts["mixed"],
            },
            "by_event_kind": {
                kind: event_kind_bucket(kind) for kind in requested_counts
            },
            "by_object_type": {"human": dict(aggregate_bucket)},
            "by_footprint_kind": {"circle": dict(aggregate_bucket)},
            "by_geometry_source": {
                "thor_marker_extent": dict(aggregate_bucket)
            },
            "target_type_policy": policy.as_dict(),
            "target_type_policy_digest": policy.digest,
            "generator_config_digest": _generator_digest(
                prepared.generator_config
            ),
            "generator_algorithm_version": "joint_occluder_first_v4",
        },
    )


def _prepared_with_reports(
    module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_preflight_inputs(monkeypatch, module, tmp_path)
    prepared = module.prepare_sop05_run(_request(module, tmp_path))
    policy_digest = prepared.generator_config["target_type_policy"].digest
    pair_zero = tuple(
        _generated_event(
            f"fixture-{kind}-{index}",
            kind,
            prepared.grid,
            index,
            "base-b",
            target_type_policy_digest=policy_digest,
            target_type_policy=prepared.target_type_policy,
            generator_config_digest=(
                prepared.generator_config_semantic_digest
            ),
            event_index=index,
            attempt_seed=10200 + index,
            canonical_identity=True,
        )
        for index, kind in enumerate(
            ("environment",) * 6 + ("structural",) * 3
        )
    )
    pair_one = (
        _generated_event(
            "fixture-mixed-9",
            "mixed",
            prepared.grid,
            20,
            "base-a",
            target_type_policy_digest=policy_digest,
            target_type_policy=prepared.target_type_policy,
            generator_config_digest=(
                prepared.generator_config_semantic_digest
            ),
            event_index=0,
            attempt_seed=10100,
            canonical_identity=True,
        ),
    )
    return prepared, {
        102: _report(prepared, 102, pair_zero),
        101: _report(prepared, 101, pair_one),
    }


def _install_thread_pair_pool(monkeypatch, module) -> None:
    def make_pool(prepared):
        module._initialize_pair_worker(prepared)
        return ThreadPoolExecutor(max_workers=prepared.request.workers)

    monkeypatch.setattr(module, "_make_pair_process_pool", make_pool)


def test_prepare_is_read_only_and_freezes_ranked_pair_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    calls = _install_preflight_inputs(monkeypatch, module, tmp_path)
    request = _request(module, tmp_path)

    prepared = module.prepare_sop05_run(request)
    summary = module.preflight_summary(prepared)

    assert calls == {"sop03": 1, "sop04": 1, "schedule": 1}
    assert [
        (item.rank, item.state_id, item.trajectory_id, item.pair_seed)
        for item in prepared.schedule
    ] == [
        (0, "base-b", "trajectory-a", 102),
        (1, "base-a", "trajectory-a", 101),
    ]
    assert summary["status"] == "preflight_ok"
    assert summary["pair_count"] == 2
    assert summary["theoretical_capacity"] == 20
    assert "required_event_kind_counts" not in summary
    assert not hasattr(prepared, "required_event_kind_counts")
    assert summary["run_id"].startswith("sop05-run-")
    assert prepared.input_lock["sop03"]["code_commit"] == "1" * 40
    assert prepared.input_lock["sop04"]["code_commit"] == "2" * 40
    assert "payload_checksums" not in prepared.input_lock["sop03"]
    assert "payload_checksums" not in prepared.input_lock["sop04"]
    assert not request.output_dir.exists()
    assert not list(tmp_path.glob(".run.staging-*"))


def test_scientific_run_identity_ignores_workers_and_resolved_input_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    _install_preflight_inputs(monkeypatch, module, tmp_path)
    baseline = module.prepare_sop05_run(_request(module, tmp_path, workers=1))

    relocated_sop03 = tmp_path / "relocated" / "sop03"
    relocated_sop04 = tmp_path / "relocated" / "sop04"
    _install_preflight_inputs(
        monkeypatch,
        module,
        tmp_path,
        sop03_root=relocated_sop03,
        sop04_root=relocated_sop04,
    )
    relocated = module.prepare_sop05_run(
        _request(
            module,
            tmp_path,
            sop03_root=relocated_sop03,
            sop04_root=relocated_sop04,
            output_dir=tmp_path / "relocated-run",
            workers=2,
        )
    )

    assert baseline.run_id == relocated.run_id
    assert baseline.runtime_provenance["resolved_input_roots"] != (
        relocated.runtime_provenance["resolved_input_roots"]
    )


def test_run_identity_binds_only_the_total_selection_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    _install_preflight_inputs(monkeypatch, module, tmp_path)

    prepared = module.prepare_sop05_run(_request(module, tmp_path))

    identity_payload = {
        "version": "sop05_generation_run_v4",
        "selection_version": "sop05_total_quota_selection_v1",
        "producer_source_identity": prepared.producer_source_identity,
        "split": prepared.request.split,
        "sop03": {
            "code_commit": prepared.sop03.producer_evidence.code_commit,
            "checksum_manifest_sha256": (
                prepared.sop03.producer_evidence.checksum_manifest_sha256
            ),
            "audit_sha256": prepared.sop03.producer_evidence.audit_sha256,
        },
        "sop04": {
            "code_commit": prepared.sop04.producer_evidence.code_commit,
            "checksum_manifest_sha256": (
                prepared.sop04.producer_evidence.checksum_manifest_sha256
            ),
            "audit_sha256": prepared.sop04.producer_evidence.audit_sha256,
        },
        "selection": prepared.input_lock["selection"],
        "base_config_sha256": prepared.base_config_sha256,
        "generator_config_sha256": prepared.generator_config_sha256,
        "generator_config_semantic_digest": (
            prepared.generator_config_semantic_digest
        ),
        "target_type_policy": prepared.target_type_policy,
        "target_type_policy_digest": prepared.target_type_policy_digest,
        "accepted_quota": prepared.request.accepted_quota,
        "events_per_pair": prepared.request.events_per_pair,
    }
    expected_digest = hashlib.blake2b(
        module._canonical_json_bytes(identity_payload), digest_size=16
    ).hexdigest()

    assert prepared.run_id == f"sop05-run-{expected_digest}"
    assert "required_event_kind_counts" not in identity_payload
    assert "quota_deficits" not in identity_payload


def test_git_executable_is_runtime_only_and_forwarded_to_identity_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    first_git = _fixture_git_executable(tmp_path, "first-git")
    second_git = _fixture_git_executable(tmp_path, "second-git")
    observed: list[Path] = []

    def load_identity(git_executable: Path) -> dict[str, object]:
        observed.append(git_executable)
        return dict(_CLEAN_SOURCE_IDENTITY)

    _install_preflight_inputs(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_load_producer_source_identity", load_identity)
    first = module.prepare_sop05_run(
        _request(module, tmp_path, git_executable=first_git)
    )
    _install_preflight_inputs(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_load_producer_source_identity", load_identity)
    second = module.prepare_sop05_run(
        _request(
            module,
            tmp_path,
            git_executable=second_git,
            output_dir=tmp_path / "second-run",
        )
    )

    assert observed == [first_git, first_git, second_git, second_git]
    assert first.run_id == second.run_id
    assert first.runtime_provenance["git_executable"] == str(first_git)
    assert second.runtime_provenance["git_executable"] == str(second_git)


def test_prepare_rejects_untrusted_git_executable_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    _install_preflight_inputs(monkeypatch, module, tmp_path)
    executable = _fixture_git_executable(tmp_path, "real-git")
    symlink = tmp_path / "symlink-git"
    symlink.symlink_to(executable)
    nonexecutable = tmp_path / "nonexecutable-git"
    nonexecutable.write_bytes(b"not executable")
    nonexecutable.chmod(0o644)
    invalid_paths = (
        Path("relative-git"),
        tmp_path / "missing-git",
        symlink,
        nonexecutable,
    )

    for index, git_executable in enumerate(invalid_paths):
        request = _request(
            module,
            tmp_path,
            git_executable=git_executable,
            output_dir=tmp_path / f"invalid-run-{index}",
        )
        with pytest.raises(module.Sop05RunError, match="git_executable"):
            module.prepare_sop05_run(request)
        assert not request.output_dir.exists()


def test_run_id_and_manifest_bind_current_producer_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    first_identity = dict(_CLEAN_SOURCE_IDENTITY)
    second_identity = {**first_identity, "git_commit": "4" * 40}
    _install_preflight_inputs(
        monkeypatch,
        module,
        tmp_path,
        producer_identity=first_identity,
    )
    first = module.prepare_sop05_run(
        _request(module, tmp_path, output_dir=tmp_path / "first")
    )
    _install_preflight_inputs(
        monkeypatch,
        module,
        tmp_path,
        producer_identity=second_identity,
    )
    second = module.prepare_sop05_run(
        _request(module, tmp_path, output_dir=tmp_path / "second")
    )

    assert first.run_id != second.run_id
    assert first.producer_source_identity == first_identity
    manifest = module._run_manifest(
        first, run_state="complete", shard_directory_name="target_motions"
    )
    assert manifest["producer_source_identity"] == first_identity
    assert manifest["runtime"]["git_executable"] == str(
        first.request.git_executable
    )


def test_dirty_source_identity_is_explicit_and_changes_scientific_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    dirty_a = {
        **_CLEAN_SOURCE_IDENTITY,
        "worktree_state": "dirty",
        "dirty_tree_sha256": "a" * 64,
    }
    dirty_b = {**dirty_a, "dirty_tree_sha256": "b" * 64}
    prepared = []
    for index, identity in enumerate(
        (_CLEAN_SOURCE_IDENTITY, dirty_a, dirty_b)
    ):
        _install_preflight_inputs(
            monkeypatch,
            module,
            tmp_path,
            producer_identity=identity,
        )
        prepared.append(
            module.prepare_sop05_run(
                _request(
                    module,
                    tmp_path,
                    output_dir=tmp_path / f"run-{index}",
                )
            )
        )

    assert len({item.run_id for item in prepared}) == 3
    assert prepared[1].producer_source_identity["worktree_state"] == "dirty"


def test_prepare_rejects_dirty_source_identity_without_content_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    invalid = {
        **_CLEAN_SOURCE_IDENTITY,
        "worktree_state": "dirty",
        "dirty_tree_sha256": None,
    }
    _install_preflight_inputs(
        monkeypatch,
        module,
        tmp_path,
        producer_identity=invalid,
    )

    with pytest.raises(module.Sop05RunError, match="dirty_tree_sha256"):
        module.prepare_sop05_run(_request(module, tmp_path))


def test_source_identity_loader_uses_explicit_git_and_reports_dirty_digest() -> None:
    module = _sut()

    identity = module._load_producer_source_identity(_REAL_GIT_EXECUTABLE)

    assert set(identity) == {
        "version",
        "git_commit",
        "worktree_state",
        "dirty_tree_sha256",
    }
    assert len(identity["git_commit"]) == 40
    assert identity["worktree_state"] in {"clean", "dirty"}
    if identity["worktree_state"] == "dirty":
        assert len(identity["dirty_tree_sha256"]) == 64
    else:
        assert identity["dirty_tree_sha256"] is None


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("accepted_quota", 0, "accepted_quota"),
        ("events_per_pair", 5, "multiple of 10"),
        ("max_base_states", 0, "max_base_states"),
        ("trajectory_count", 0, "trajectory_count"),
        ("max_pairs", 0, "max_pairs"),
        ("checksum_workers", 0, "checksum_workers"),
        ("workers", 0, "workers"),
    ],
)
def test_prepare_rejects_invalid_finite_run_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    bad_value: object,
    message: str,
) -> None:
    module = _sut()
    _install_preflight_inputs(monkeypatch, module, tmp_path)
    request = replace(_request(module, tmp_path), **{field_name: bad_value})

    with pytest.raises(module.Sop05RunError, match=message):
        module.prepare_sop05_run(request)
    assert not request.output_dir.exists()


def test_prepare_accepts_nondecadal_total_quota_but_not_partial_pair_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    _install_preflight_inputs(monkeypatch, module, tmp_path)

    prepared = module.prepare_sop05_run(
        _request(module, tmp_path, accepted_quota=7)
    )

    assert prepared.request.accepted_quota == 7
    assert not hasattr(prepared, "required_event_kind_counts")
    with pytest.raises(module.Sop05RunError, match="events_per_pair.*multiple of 10"):
        module.prepare_sop05_run(
            _request(
                module,
                tmp_path,
                accepted_quota=7,
                events_per_pair=7,
                output_dir=tmp_path / "partial-pair-run",
            )
        )


def test_prepare_rejects_existing_output_and_impossible_capacity_before_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    _install_preflight_inputs(monkeypatch, module, tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.prepare_sop05_run(
            _request(module, tmp_path, output_dir=existing)
        )

    with pytest.raises(module.Sop05RunError, match="theoretical capacity"):
        module.prepare_sop05_run(
            _request(module, tmp_path, accepted_quota=30, max_pairs=2)
        )


def test_parallel_pair_collection_completes_with_full_mixed_kind_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    _install_thread_pair_pool(monkeypatch, module)
    barrier = threading.Barrier(2)
    calls: list[int] = []
    finish_order: list[int] = []

    def generate(**kwargs):
        pair_seed = kwargs["seed"]
        calls.append(pair_seed)
        assert kwargs["event_count"] == 10
        barrier.wait(timeout=2.0)
        if pair_seed == 102:
            time.sleep(0.03)
        finish_order.append(pair_seed)
        return reports[pair_seed]

    monkeypatch.setattr(module, "generate_events", generate)
    collection = module.collect_sop05_generation(prepared)

    assert sorted(calls) == [101, 102]
    assert Counter(calls) == {101: 1, 102: 1}
    assert finish_order == [101, 102]
    assert [item.rank for item in collection.pair_reports] == [0, 1]
    assert Counter(event.event_kind for event in collection.selected_events) == {
        "environment": 6,
        "structural": 3,
        "mixed": 1,
    }
    assert collection.generation_summary["processed_pair_count"] == 2
    assert collection.generation_summary["generator_accepted_count"] == 10
    assert collection.generation_summary["selected_count"] == 10
    assert collection.generation_summary["quota_met"] is True
    assert collection.generation_summary["selected_event_kind_counts"] == {
        "environment": 6,
        "structural": 3,
        "mixed": 1,
    }
    assert "required_event_kind_counts" not in collection.generation_summary
    assert "quota_deficits" not in collection.generation_summary


def test_collection_rejects_report_count_mismatch_and_global_identity_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    _install_thread_pair_pool(monkeypatch, module)
    invalid = replace(
        reports[102],
        summary={**reports[102].summary, "accepted_count": 8},
    )
    monkeypatch.setattr(
        module,
        "generate_events",
        lambda **kwargs: invalid if kwargs["seed"] == 102 else reports[101],
    )
    with pytest.raises(module.Sop05RunError, match="accepted_count"):
        module.collect_sop05_generation(prepared)

    duplicate = _generated_event(
        reports[102].events[0].generated_event_id,
        "environment",
        prepared.grid,
        99,
        "base-a",
        target_type_policy_digest=(
            prepared.generator_config["target_type_policy"].digest
        ),
    )
    replay = replace(reports[101], events=(duplicate,))
    replay.summary["event_kind_counts"] = {
        "environment": 1,
        "structural": 0,
        "mixed": 0,
    }
    monkeypatch.setattr(
        module,
        "generate_events",
        lambda **kwargs: reports[102] if kwargs["seed"] == 102 else replay,
    )
    with pytest.raises(module.Sop05RunError, match="duplicate generated_event_id"):
        module.collect_sop05_generation(prepared)


@pytest.mark.parametrize(
    ("summary_update", "message"),
    [
        ({"rejection_reasons": {}}, "rejection_reasons total"),
        (
            {
                "rejection_stage_counts": {
                    "occluder_geometry": 0,
                    "target_conditioning": 0,
                    "visibility": 0,
                }
            },
            "rejection_stage_counts total",
        ),
        ({"attempt_acceptance_rate": 0.0}, "attempt_acceptance_rate"),
        ({"generator_config_digest": None}, "generator_config_digest"),
    ],
)
def test_collection_rejects_inconsistent_strict_report_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    summary_update: dict[str, object],
    message: str,
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    _install_thread_pair_pool(monkeypatch, module)
    invalid = replace(
        reports[102],
        summary={**reports[102].summary, **summary_update},
    )
    monkeypatch.setattr(
        module,
        "generate_events",
        lambda **kwargs: invalid if kwargs["seed"] == 102 else reports[101],
    )

    with pytest.raises(module.Sop05RunError, match=message):
        module.collect_sop05_generation(prepared)


@pytest.mark.parametrize(
    ("wrong_field", "message"),
    [
        ("generator_config_digest", "generator_config_digest"),
        ("target_type_policy", "target_type_policy"),
        ("target_type_policy_digest", "target_type_policy_digest"),
    ],
)
def test_collection_rejects_all_reports_that_agree_on_wrong_config_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_field: str,
    message: str,
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    _install_thread_pair_pool(monkeypatch, module)
    wrong_value: object = "0" * 32
    if wrong_field == "target_type_policy":
        wrong_value = {
            "whitelist": ["carried_object"],
            "weights": {
                "human": 0.0,
                "carried_object": 1.0,
                "unknown_dynamic": 0.0,
            },
        }
    wrong_reports = {
        seed: replace(
            report,
            summary={**report.summary, wrong_field: wrong_value},
        )
        for seed, report in reports.items()
    }
    monkeypatch.setattr(
        module,
        "generate_events",
        lambda **kwargs: wrong_reports[kwargs["seed"]],
    )

    with pytest.raises(module.Sop05RunError, match=message):
        module.collect_sop05_generation(prepared)


def test_collection_rejects_old_generator_algorithm_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    _install_thread_pair_pool(monkeypatch, module)
    old_reports = {
        seed: replace(
            report,
            summary={
                **report.summary,
                "generator_algorithm_version": "joint_occluder_first_v3",
            },
        )
        for seed, report in reports.items()
    }
    monkeypatch.setattr(
        module,
        "generate_events",
        lambda **kwargs: old_reports[kwargs["seed"]],
    )

    with pytest.raises(module.Sop05RunError, match="generator_algorithm_version"):
        module.collect_sop05_generation(prepared)


def test_collection_rejects_record_policy_digest_not_bound_to_prepared_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    _install_thread_pair_pool(monkeypatch, module)
    original = reports[102].events[0]
    wrong_event = _generated_event(
        original.generated_event_id,
        original.event_kind,
        prepared.grid,
        0,
        "base-b",
        target_type_policy_digest="0" * 32,
    )
    wrong_first = replace(
        reports[102], events=(wrong_event, *reports[102].events[1:])
    )
    monkeypatch.setattr(
        module,
        "generate_events",
        lambda **kwargs: wrong_first
        if kwargs["seed"] == 102
        else reports[101],
    )

    with pytest.raises(module.Sop05RunError, match="record target_type_policy_digest"):
        module.collect_sop05_generation(prepared)


def test_process_result_transport_round_trips_without_frozen_mapping_pickle_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    original = module.PairGenerationReport(
        rank=0,
        state_id="base-b",
        trajectory_id="trajectory-a",
        pair_seed=102,
        report=reports[102],
    )

    transported = pickle.loads(pickle.dumps(module._transport_pair_report(original)))
    restored = module._restore_pair_report(transported)

    assert [event.generated_event_id for event in restored.report.events] == [
        event.generated_event_id for event in original.report.events
    ]
    assert [
        event.target_motion_record.record_digest for event in restored.report.events
    ] == [
        event.target_motion_record.record_digest for event in original.report.events
    ]


def test_default_parallel_backend_runs_generation_in_child_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    parent_pid = os.getpid()

    def generate(**kwargs):
        report = reports[kwargs["seed"]]
        return replace(
            report,
            summary={**report.summary, "fixture_worker_pid": os.getpid()},
        )

    monkeypatch.setattr(module, "generate_events", generate)

    collection = module.collect_sop05_generation(prepared)

    child_pids = {
        item.report.summary["fixture_worker_pid"]
        for item in collection.pair_reports
    }
    assert child_pids
    assert parent_pid not in child_pids


def test_worker_count_does_not_change_selected_scientific_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    prepared_two, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    prepared_one = replace(
        prepared_two,
        request=replace(
            prepared_two.request,
            output_dir=tmp_path / "single-worker-run",
            workers=1,
        ),
        runtime_provenance={
            **prepared_two.runtime_provenance,
            "workers": 1,
        },
    )
    _install_thread_pair_pool(monkeypatch, module)
    monkeypatch.setattr(
        module, "generate_events", lambda **kwargs: reports[kwargs["seed"]]
    )

    single = module.collect_sop05_generation(prepared_one)
    parallel = module.collect_sop05_generation(prepared_two)

    assert [event.generated_event_id for event in single.selected_events] == [
        event.generated_event_id for event in parallel.selected_events
    ]
    assert [
        event.target_motion_record.record_digest for event in single.selected_events
    ] == [
        event.target_motion_record.record_digest
        for event in parallel.selected_events
    ]
    single_result = module.publish_sop05_generation(prepared_one, single)
    parallel_result = module.publish_sop05_generation(prepared_two, parallel)
    single_shard = load_event_target_motion_shard(
        single_result.output_dir / "target_motions", grid=prepared_one.grid
    )
    parallel_shard = load_event_target_motion_shard(
        parallel_result.output_dir / "target_motions", grid=prepared_two.grid
    )
    assert single_shard.manifest_digest == parallel_shard.manifest_digest
    assert (
        single_shard.payload_semantic_digest
        == parallel_shard.payload_semantic_digest
    )
    assert [record.record_digest for record in single_shard.records] == [
        record.record_digest for record in parallel_shard.records
    ]


def test_workers_only_bound_execution_and_never_change_processed_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    _install_preflight_inputs(monkeypatch, module, tmp_path)
    prepared_two = module.prepare_sop05_run(_request(module, tmp_path, workers=2))
    prepared_one = replace(
        prepared_two, request=replace(prepared_two.request, workers=1)
    )
    _install_thread_pair_pool(monkeypatch, module)

    def complete_report(pair_seed: int, base_state_id: str) -> EventGenerationReport:
        events = tuple(
            _generated_event(
                f"event-{pair_seed}-{kind}-{index}",
                kind,
                prepared_two.grid,
                pair_seed + index,
                base_state_id,
                target_type_policy_digest=(
                    prepared_two.generator_config["target_type_policy"].digest
                ),
            )
            for kind, count in (("environment", 6), ("structural", 3), ("mixed", 1))
            for index in range(count)
        )
        return _report(prepared_two, pair_seed, events)

    reports = {
        102: complete_report(102, "base-b"),
        101: complete_report(101, "base-a"),
    }
    monkeypatch.setattr(
        module, "generate_events", lambda **kwargs: reports[kwargs["seed"]]
    )

    single = module.collect_sop05_generation(prepared_one)
    parallel = module.collect_sop05_generation(prepared_two)

    assert [item.rank for item in single.pair_reports] == [0, 1]
    assert [item.rank for item in parallel.pair_reports] == [0, 1]
    assert [event.generated_event_id for event in single.selected_events] == [
        event.generated_event_id for event in parallel.selected_events
    ]


def test_success_publication_writes_one_strict_shard_and_complete_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    _install_thread_pair_pool(monkeypatch, module)
    monkeypatch.setattr(
        module, "generate_events", lambda **kwargs: reports[kwargs["seed"]]
    )
    collection = module.collect_sop05_generation(prepared)

    result = module.publish_sop05_generation(prepared, collection)

    assert result.run_state == "complete"
    assert isinstance(result.publication_semantic_digest, str)
    assert len(result.publication_semantic_digest) == 64
    assert result.output_dir == prepared.request.output_dir
    assert {path.name for path in result.output_dir.iterdir()} == {
        ".producer-complete",
        "checksums.sha256",
        "configs",
        "generation_summary.json",
        "pair_generation_reports.jsonl",
        "run_manifest.json",
        "target_motions",
    }
    marker = result.output_dir / ".producer-complete"
    assert marker.is_file() and marker.stat().st_size > 0
    manifest = json.loads(
        (result.output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_state"] == "complete"
    assert manifest["run_id"] == prepared.run_id
    assert manifest["manifest_version"] == "sop05_run_manifest_v2"
    assert manifest["producer_version"] == "sop05_generation_run_v4"
    assert manifest["runtime"] == {
        "checksum_workers": 2,
        "git_executable": str(prepared.request.git_executable),
        "resolved_input_roots": {
            "sop03": str(tmp_path / "sop03"),
            "sop04": str(tmp_path / "sop04"),
        },
        "workers": 2,
    }
    assert "payload_checksums" not in manifest["input_lock"]["sop03"]
    assert (result.output_dir / "configs/base.yaml").read_bytes() == (
        prepared.request.base_config_path.read_bytes()
    )
    assert (result.output_dir / "configs/generator.yaml").read_bytes() == (
        prepared.request.generator_config_path.read_bytes()
    )
    assert manifest["scientific_request"]["base_config_sha256"] == _sha256(
        result.output_dir / "configs/base.yaml"
    )
    assert manifest["scientific_request"]["generator_config_sha256"] == _sha256(
        result.output_dir / "configs/generator.yaml"
    )
    assert manifest["scientific_request"]["selection_version"] == (
        "sop05_total_quota_selection_v1"
    )
    assert "required_event_kind_counts" not in manifest["scientific_request"]
    generation_summary = json.loads(
        (result.output_dir / "generation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert generation_summary["summary_version"] == (
        "sop05_generation_summary_v2"
    )
    assert generation_summary["quota_met"] is True
    assert generation_summary["selected_event_kind_counts"] == {
        "environment": 6,
        "mixed": 1,
        "structural": 3,
    }
    assert "required_event_kind_counts" not in generation_summary
    assert "quota_deficits" not in generation_summary
    report_rows = [
        json.loads(line)
        for line in (result.output_dir / "pair_generation_reports.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["rank"] for row in report_rows] == [0, 1]
    assert report_rows[0]["report_version"] == (
        "sop05_pair_generation_report_v2"
    )
    assert report_rows[0]["selection_version"] == (
        "sop05_total_quota_selection_v1"
    )
    assert report_rows[0]["accepted_events"] == [
        {
            "generated_event_id": event.generated_event_id,
            "event_kind": event.event_kind,
        }
        for event in reports[102].events
    ]
    assert report_rows[0]["summary"] == reports[102].summary
    _assert_outer_checksums(result.output_dir)
    loaded = load_event_target_motion_shard(
        result.output_dir / "target_motions",
        grid=prepared.grid,
        expected_generated_event_ids={
            event.generated_event_id for event in collection.selected_events
        },
    )
    assert len(loaded.records) == 10
    from src.generation.sop05_output_loader import load_complete_sop05_events

    consumer_loaded = load_complete_sop05_events(
        result.output_dir,
        grid=prepared.grid,
        expected_publication_semantic_digest=(
            result.publication_semantic_digest
        ),
        expected_run_id=result.run_id,
    )
    assert len(consumer_loaded.events) == len(collection.selected_events) == 10
    assert consumer_loaded.publication_semantic_digest == (
        result.publication_semantic_digest
    )
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_payload["marker_version"] == "sop05_producer_complete_v2"
    assert marker_payload["publication_identity_version"] == (
        "sop05_publication_semantic_digest_v1"
    )
    assert marker_payload["run_manifest_sha256"] == _sha256(
        result.output_dir / "run_manifest.json"
    )
    assert marker_payload["checksums_sha256"] == _sha256(
        result.output_dir / "checksums.sha256"
    )
    assert marker_payload["target_motion_manifest_digest"] == (
        loaded.manifest_digest
    )
    assert marker_payload["target_motion_payload_semantic_digest"] == (
        loaded.payload_semantic_digest
    )
    assert marker_payload["publication_semantic_digest"] == (
        result.publication_semantic_digest
    )
    identity = _publication_identity_sut()
    assert result.publication_semantic_digest == (
        identity.compute_sop05_publication_semantic_digest(
            run_id=prepared.run_id,
            run_manifest_sha256=marker_payload["run_manifest_sha256"],
            checksums_sha256=marker_payload["checksums_sha256"],
            target_motion_manifest_digest=loaded.manifest_digest,
            target_motion_payload_semantic_digest=(
                loaded.payload_semantic_digest
            ),
        )
    )
    assert not list(tmp_path.glob(".run.staging-*"))


def test_complete_publication_requires_formal_consumer_round_trip_before_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    _install_thread_pair_pool(monkeypatch, module)
    original = reports[102].events[0]
    fake_identity_event = _generated_event(
        "event-coherent-fake",
        original.event_kind,
        prepared.grid,
        99.0,
        "base-b",
        target_type_policy_digest=prepared.target_type_policy_digest,
        target_type_policy=prepared.target_type_policy,
        generator_config_digest=prepared.generator_config_semantic_digest,
        event_index=0,
        attempt_seed=10200,
    )
    tampered_reports = {
        **reports,
        102: replace(
            reports[102],
            events=(fake_identity_event, *reports[102].events[1:]),
        ),
    }
    monkeypatch.setattr(
        module,
        "generate_events",
        lambda **kwargs: tampered_reports[kwargs["seed"]],
    )
    collection = module.collect_sop05_generation(prepared)

    with pytest.raises(
        ValueError, match="generated_event_id does not match canonical"
    ):
        module.publish_sop05_generation(prepared, collection)

    assert not prepared.request.output_dir.exists()
    assert not list(tmp_path.glob(".run.staging-*"))


def test_publication_rejects_source_change_after_generation_before_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    _install_thread_pair_pool(monkeypatch, module)
    monkeypatch.setattr(
        module, "generate_events", lambda **kwargs: reports[kwargs["seed"]]
    )
    collection = module.collect_sop05_generation(prepared)
    observed: list[Path] = []

    def changed_identity(git_executable: Path) -> dict[str, object]:
        observed.append(git_executable)
        return {
            **_CLEAN_SOURCE_IDENTITY,
            "worktree_state": "dirty",
            "dirty_tree_sha256": "f" * 64,
        }

    monkeypatch.setattr(
        module, "_load_producer_source_identity", changed_identity
    )

    with pytest.raises(
        module.Sop05RunError, match="producer source changed after generation"
    ):
        module.publish_sop05_generation(prepared, collection)

    assert observed == [prepared.request.git_executable]
    assert not prepared.request.output_dir.exists()
    assert not list(tmp_path.glob(".run.staging-*"))


def test_quota_shortfall_publishes_explicit_partial_failure_without_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    _install_thread_pair_pool(monkeypatch, module)
    sparse_reports = {
        102: _report(prepared, 102, reports[102].events[:6]),
        101: _report(prepared, 101, ()),
    }
    monkeypatch.setattr(
        module,
        "generate_events",
        lambda **kwargs: sparse_reports[kwargs["seed"]],
    )
    collection = module.collect_sop05_generation(prepared)
    assert collection.generation_summary["quota_met"] is False

    result = module.publish_sop05_generation(prepared, collection)

    assert result.run_state == "quota_unmet"
    assert result.publication_semantic_digest is None
    assert {path.name for path in result.output_dir.iterdir()} == {
        "checksums.sha256",
        "configs",
        "generation_summary.json",
        "pair_generation_reports.jsonl",
        "partial_target_motions",
        "run_manifest.json",
    }
    assert not (result.output_dir / ".producer-complete").exists()
    assert not (result.output_dir / "target_motions").exists()
    manifest = json.loads(
        (result.output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (result.output_dir / "generation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["run_state"] == "quota_unmet"
    assert summary["run_state"] == "quota_unmet"
    assert summary["summary_version"] == "sop05_generation_summary_v2"
    assert summary["selected_count"] == 6
    assert summary["selected_event_kind_counts"] == {
        "environment": 6,
        "mixed": 0,
        "structural": 0,
    }
    assert "required_event_kind_counts" not in summary
    assert "quota_deficits" not in summary
    assert "required_event_kind_counts" not in manifest["scientific_request"]
    partial = load_event_target_motion_shard(
        result.output_dir / "partial_target_motions", grid=prepared.grid
    )
    assert len(partial.records) == 6
    _assert_outer_checksums(result.output_dir)
    assert (result.output_dir / "configs/base.yaml").read_bytes() == (
        prepared.request.base_config_path.read_bytes()
    )
    assert len(
        (result.output_dir / "pair_generation_reports.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 2


def test_zero_acceptance_failure_has_audit_evidence_but_no_empty_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    prepared, _ = _prepared_with_reports(module, tmp_path, monkeypatch)
    _install_thread_pair_pool(monkeypatch, module)
    monkeypatch.setattr(
        module,
        "generate_events",
        lambda **kwargs: _report(prepared, kwargs["seed"], ()),
    )
    collection = module.collect_sop05_generation(prepared)

    result = module.publish_sop05_generation(prepared, collection)

    assert result.run_state == "quota_unmet"
    assert result.publication_semantic_digest is None
    assert {path.name for path in result.output_dir.iterdir()} == {
        "checksums.sha256",
        "configs",
        "generation_summary.json",
        "pair_generation_reports.jsonl",
        "run_manifest.json",
    }
    assert not (result.output_dir / ".producer-complete").exists()
    assert not (result.output_dir / "partial_target_motions").exists()
    _assert_outer_checksums(result.output_dir)


def test_publication_rejects_selection_order_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    _install_thread_pair_pool(monkeypatch, module)
    monkeypatch.setattr(
        module, "generate_events", lambda **kwargs: reports[kwargs["seed"]]
    )
    collection = module.collect_sop05_generation(prepared)
    tampered = replace(
        collection,
        selected_events=tuple(reversed(collection.selected_events)),
    )

    with pytest.raises(
        module.Sop05RunError, match="frozen total-quota selection"
    ):
        module.publish_sop05_generation(prepared, tampered)

    assert not prepared.request.output_dir.exists()


def test_publication_rejects_selected_payload_replaced_behind_same_id_and_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    _install_thread_pair_pool(monkeypatch, module)
    monkeypatch.setattr(
        module, "generate_events", lambda **kwargs: reports[kwargs["seed"]]
    )
    collection = module.collect_sop05_generation(prepared)
    original = collection.selected_events[0]
    replacement = _generated_event(
        original.generated_event_id,
        original.event_kind,
        prepared.grid,
        999.0,
        original.target_motion_record.base_state_id,
        target_type_policy_digest=prepared.target_type_policy_digest,
    )
    assert replacement.target_motion_record.record_digest != (
        original.target_motion_record.record_digest
    )
    tampered = replace(
        collection,
        selected_events=(replacement, *collection.selected_events[1:]),
    )

    with pytest.raises(
        module.Sop05RunError,
        match="selected event payload differs from pair reports",
    ):
        module.publish_sop05_generation(prepared, tampered)

    assert not prepared.request.output_dir.exists()


@pytest.mark.parametrize(
    ("summary_update", "message"),
    [
        ({"processed_pair_count": 999}, "processed_pair_count"),
        ({"requested_event_count": 999}, "requested_event_count"),
        ({"attempted_count": 999}, "attempted_count"),
        ({"generator_accepted_count": 999}, "generator_accepted_count"),
        ({"quota_trimmed_count": 999}, "quota_trimmed_count"),
        (
            {
                "generated_event_kind_counts": {
                    "environment": 10,
                    "structural": 0,
                    "mixed": 0,
                }
            },
            "generated_event_kind_counts",
        ),
        ({"rejection_reasons": {}}, "rejection_reasons"),
        ({"rejection_stage_counts": {}}, "rejection_stage_counts"),
        ({"generator_invariants": {}}, "generator_invariants"),
        ({"obsolete_kind_quota_field": {}}, "schema"),
    ],
)
def test_publication_recomputes_every_generation_summary_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    summary_update: dict[str, object],
    message: str,
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    _install_thread_pair_pool(monkeypatch, module)
    monkeypatch.setattr(
        module, "generate_events", lambda **kwargs: reports[kwargs["seed"]]
    )
    collection = module.collect_sop05_generation(prepared)
    tampered = replace(
        collection,
        generation_summary={
            **collection.generation_summary,
            **summary_update,
        },
    )

    with pytest.raises(module.Sop05RunError, match=message):
        module.publish_sop05_generation(prepared, tampered)

    assert not prepared.request.output_dir.exists()


def test_outer_publication_failure_cleans_staging_and_never_exposes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _sut()
    prepared, reports = _prepared_with_reports(module, tmp_path, monkeypatch)
    _install_thread_pair_pool(monkeypatch, module)
    monkeypatch.setattr(
        module, "generate_events", lambda **kwargs: reports[kwargs["seed"]]
    )
    collection = module.collect_sop05_generation(prepared)

    def fail_writer(records, worlds, output_dir, *, grid):
        Path(output_dir).mkdir()
        raise RuntimeError("fixture write failure")

    monkeypatch.setattr(module, "write_event_target_motion_shard", fail_writer)
    with pytest.raises(RuntimeError, match="fixture write failure"):
        module.publish_sop05_generation(prepared, collection)

    assert not prepared.request.output_dir.exists()
    assert not list(tmp_path.glob(".run.staging-*"))
