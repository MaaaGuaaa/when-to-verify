"""Strict manifest-driven SOP-03/SOP-04 input adapter tests."""

from __future__ import annotations

import builtins
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.contracts import (
    BaseState,
    GridSpec,
    OracleContext,
    save_dataclass,
)
from src.generation.sop05_input_adapter import (
    Sop05InputError,
    build_stable_pair_schedule,
    load_sop03_split_inputs,
    load_sop04_trajectory_bank,
)


SCHEMA_VERSION = "2.0.0"
SPLIT_DIGEST = "0123456789abcdef0123456789abcdef"
SOP03_COMMIT = "1" * 40
SOP04_COMMIT = "2" * 40
LAYOUT_VERSION = "history8_current7_future15_v1"
SOP04_BANK_VERSION = "sop04_audited_bank_v2"
SOP04_POSE_TIME_LAYOUT_VERSION = "future_endpoints_dt_to_horizon_v1"


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_provenance() -> dict[str, object]:
    return {
        "evaluation_scope": "fixture_unseen_recording",
        "grouping_unit": "recording_id",
        "split_manifest_digest": SPLIT_DIGEST,
    }


def _snippet_summary(
    *,
    object_type: str,
    count: int,
    sample_count: int,
    duration_s: float,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "split": "train",
        "object_type": object_type,
        "accepted_count": count,
        "candidate_count": count,
        "rejected_count": 0,
        "sample_count": sample_count,
        "duration_s": duration_s,
        "sample_dt_s": 0.2,
        "history_steps": 8,
        "current_index": 7,
        "future_steps": 15,
        "motion_snippet_layout_version": (
            LAYOUT_VERSION if sample_count == 23 else "legacy16_current7_v0"
        ),
        "split_manifest_digest": SPLIT_DIGEST,
        "split_provenance": _split_provenance(),
        "array_sha256": "a" * 64,
    }


def _write_snippet_library(
    root: Path,
    *,
    object_type: str,
    sample_count: int,
    declared_sample_count: int | None = None,
) -> None:
    directory = root / "snippets" / "train" / object_type
    directory.mkdir(parents=True, exist_ok=True)
    declared = sample_count if declared_sample_count is None else declared_sample_count
    duration_s = 4.4 if declared == 23 else 3.0
    count = 0 if object_type == "unknown_dynamic" else 1
    summary = _snippet_summary(
        object_type=object_type,
        count=count,
        sample_count=declared,
        duration_s=duration_s,
    )
    if declared_sample_count == 23:
        summary["motion_snippet_layout_version"] = LAYOUT_VERSION
        summary["duration_s"] = 4.4

    if count:
        positions = np.zeros((1, sample_count, 2), dtype=np.float32)
        positions[0, :, 0] = np.arange(sample_count, dtype=np.float32) * 0.1
        velocities = np.zeros_like(positions)
        velocities[0, :, 0] = np.float32(0.5)
        headings = np.zeros((1, sample_count), dtype=np.float32)
        footprint = (
            {"kind": "circle", "radius_m": 0.3}
            if object_type == "human"
            else {"kind": "rectangle", "length_m": 0.6, "width_m": 0.3}
        )
        snippet_id = f"train-{object_type}-snippet-fixture"
        metadata_rows = [
            {
                "snippet_id": snippet_id,
                "split": "train",
                "source_recording_id": "recording-a",
                "source_object_id": f"recording-a::{object_type}",
                "object_type": object_type,
                "footprint": footprint,
                "start_timestamp": 1.0,
                "duration_s": duration_s,
                "mean_speed_mps": 0.5,
                "max_acceleration_mps2": 0.0,
                "mean_abs_curvature_per_m": 0.0,
                "provenance": {
                    "track_provenance": {
                        "geometry_source": "fixture",
                        "orientation_source": "fixture",
                    }
                },
            }
        ]
        source_rows = [
            {
                "schema_version": SCHEMA_VERSION,
                "split": "train",
                "object_type": object_type,
                "snippet_id": snippet_id,
                "source_recording_id": "recording-a",
                "source_object_id": f"recording-a::{object_type}",
                "footprint": footprint,
                "sample_count": declared,
                "duration_s": duration_s,
                "sample_dt_s": 0.2,
                "history_steps": 8,
                "current_index": 7,
                "future_steps": 15,
                "motion_snippet_layout_version": summary[
                    "motion_snippet_layout_version"
                ],
                "split_manifest_digest": SPLIT_DIGEST,
                "split_provenance": _split_provenance(),
            }
        ]
    else:
        positions = np.empty((0, sample_count, 2), dtype=np.float32)
        velocities = np.empty((0, sample_count, 2), dtype=np.float32)
        headings = np.empty((0, sample_count), dtype=np.float32)
        metadata_rows = []
        source_rows = []

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "object_type": object_type,
        "summary": summary,
        "snippets": metadata_rows,
    }
    np.savez(
        directory / "snippet_library.npz",
        positions=positions,
        velocities=velocities,
        headings=headings,
        meta_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    _json_write(directory / "summary.json", summary)
    _jsonl_write(directory / "source_manifest.jsonl", source_rows)


def _base_state(state_id: str, grid: GridSpec) -> BaseState:
    object_id = "recording-a::human"
    history = np.tile(
        np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        (grid.history_steps, 1),
    )
    spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.3},
    }
    return BaseState(
        state_id=state_id,
        split="train",
        recording_id="recording-a",
        dynamic_object_ids=(object_id,),
        timestamp=10.0,
        robot_history=np.zeros((grid.history_steps, 3), dtype=np.float32),
        robot_state=np.zeros(2, dtype=np.float32),
        visible_dynamic_object_history={object_id: history.copy()},
        visible_dynamic_object_specs={object_id: spec},
        static_map_local=np.zeros((grid.height, grid.width), dtype=np.float32),
        metadata={"source_recording_id": "recording-a"},
    )


def _oracle_context(state_id: str, grid: GridSpec) -> OracleContext:
    object_id = "recording-a::human"
    history = np.tile(
        np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        (grid.history_steps, 1),
    )
    future = np.tile(
        np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        (grid.future_steps, 1),
    )
    return OracleContext(
        base_state_id=state_id,
        dynamic_object_history={object_id: history},
        dynamic_object_future={object_id: future},
        dynamic_object_specs={
            object_id: {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.3},
            }
        },
        metadata={"source_recording_id": "recording-a", "future_dt_s": 0.2},
    )


def _refresh_sop03_checksums(root: Path) -> None:
    excluded = {"artifact_checksum_summary.json", "artifact_checksums.sha256"}
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    )
    lines = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files
    ]
    manifest = root / "artifact_checksums.sha256"
    manifest.write_text("".join(lines), encoding="utf-8")
    _json_write(
        root / "artifact_checksum_summary.json",
        {
            "checksum_algorithm": "sha256",
            "checksum_manifest": "artifact_checksums.sha256",
            "checksum_manifest_sha256": _sha256(manifest),
            "covered_file_count": len(files),
            "covered_total_bytes": sum(path.stat().st_size for path in files),
            "excluded_paths": sorted(excluded),
            "status": "complete",
        },
    )


def _write_sop03_bundle(
    root: Path,
    grid: GridSpec,
    *,
    snippet_steps: int = 23,
    declared_snippet_steps: int | None = None,
) -> Path:
    root.mkdir(parents=True)
    (root / ".producer-complete").write_bytes(b"")
    state_ids = ("train-base-a", "train-base-b", "train-base-c")
    base_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    split_dir = root / "base_states" / "train"
    for state_id in state_ids:
        base_rel = f"base_states/{state_id}.npz"
        oracle_rel = f"oracle_contexts/{state_id}.npz"
        save_dataclass(_base_state(state_id, grid), split_dir / base_rel)
        save_dataclass(_oracle_context(state_id, grid), split_dir / oracle_rel)
        base_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "split": "train",
                "state_id": state_id,
                "recording_id": "recording-a",
                "timestamp": 10.0,
                "dynamic_object_ids": ["recording-a::human"],
                "base_state_file": base_rel,
                "split_provenance": _split_provenance(),
            }
        )
        oracle_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "base_state_id": state_id,
                "source_recording_id": "recording-a",
                "source_dynamic_object_ids": ["recording-a::human"],
                "oracle_context_file": oracle_rel,
                "split_provenance": _split_provenance(),
            }
        )
    _jsonl_write(split_dir / "base_state_manifest.jsonl", base_rows)
    _jsonl_write(split_dir / "oracle_context_manifest.jsonl", oracle_rows)
    _json_write(
        split_dir / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "split": "train",
            "accepted_count": len(state_ids),
            "candidate_count": len(state_ids),
            "rejected_count": 0,
            "history_steps": 8,
            "future_steps": 15,
            "dt_s": 0.2,
            "split_provenance": _split_provenance(),
        },
    )
    for object_type in ("human", "carried_object", "unknown_dynamic"):
        _write_snippet_library(
            root,
            object_type=object_type,
            sample_count=snippet_steps,
            declared_sample_count=declared_snippet_steps,
        )
    snippet_counts = {"human": 1, "carried_object": 1, "unknown_dynamic": 0}
    _json_write(
        root / "run_manifest.json",
        {
            "status": "complete",
            "repository": {"code_commit": SOP03_COMMIT},
            "inputs": {"split_manifest_digest": SPLIT_DIGEST},
            "counts": {
                "base_states": len(state_ids),
                "oracle_contexts": len(state_ids),
                "snippets": sum(snippet_counts.values()),
            },
            "validation": {"status": "passed", "audit_report": "audit_report.json"},
        },
    )
    _json_write(
        root / "audit_report.json",
        {
            "status": "ok",
            "code_commit": SOP03_COMMIT,
            "base_states": {
                "accepted_counts": {"train": len(state_ids)},
                "total_accepted_count": len(state_ids),
                "shape_dtype_finite_contract_validation": "passed_all",
            },
            "snippets": {
                "accepted_counts": {"train": snippet_counts},
                "total_accepted_count": sum(snippet_counts.values()),
                "strict_source_window_and_array_validation": "passed_all",
            },
            "split": {
                "manifest_digest": SPLIT_DIGEST,
                "disallowed_overlap_count": 0,
            },
            "old_artifact_integrity": {
                "checksum_manifest_sha256": "f" * 64,
                "status": "legacy_only",
            },
        },
    )
    _refresh_sop03_checksums(root)
    return root


def _trajectory_summary(grid: GridSpec) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_bank_version": SOP04_BANK_VERSION,
        "pose_time_layout_version": SOP04_POSE_TIME_LAYOUT_VERSION,
        "accepted_count": 21,
        "candidate_count": 21,
        "rejected_count": 0,
        "acceptance_rate": 1.0,
        "array_dtype": "float32",
        "trajectory_steps": 15,
        "dt_s": 0.2,
        "first_pose_time_s": 0.2,
        "last_pose_time_s": 3.0,
        "grid_height": grid.height,
        "grid_width": grid.width,
        "grid_resolution_m": grid.resolution_m,
        "meets_minimum_acceptance_rate": True,
        "minimum_acceptance_rate": 0.7,
        "braking_deceleration_mps2": 1.0,
        "task_cost_semantics": "fixture_zero",
        "provenance": {
            "canonical_shared_bank": True,
            "code_commit": SOP04_COMMIT,
            "config": "configs/base.yaml",
        },
    }


def _refresh_sop04_checksums(root: Path) -> None:
    excluded = {
        "audit_report.json",
        "artifact_checksums.sha256",
        "external_handoff_digest.sha256",
    }
    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.name not in excluded
    )
    manifest = root / "artifact_checksums.sha256"
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    audit_path = root / "audit_report.json"
    audit = _json_read(audit_path)
    assert isinstance(audit, dict)
    audit["checksum_manifest_sha256"] = _sha256(manifest)
    audit["checksummed_payload_file_count"] = len(files)
    _json_write(audit_path, audit)
    envelope = hashlib.sha256()
    envelope.update(b"sop04_audited_bank_v2_external_handoff\0")
    envelope.update(manifest.read_bytes())
    envelope.update(b"\0")
    envelope.update(audit_path.read_bytes())
    (root / "external_handoff_digest.sha256").write_text(
        f"{envelope.hexdigest()}  sop04_audited_bank_v2_envelope\n",
        encoding="utf-8",
    )


def _sop04_handoff_digest(root: Path) -> str:
    return (root / "external_handoff_digest.sha256").read_text(
        encoding="utf-8"
    ).split()[0]


def _load_sop04(root: Path, grid: GridSpec):
    return load_sop04_trajectory_bank(
        root,
        grid,
        expected_external_handoff_digest_sha256=_sop04_handoff_digest(root),
    )


def _write_sop04_bundle(root: Path, grid: GridSpec) -> Path:
    root.mkdir(parents=True)
    count = 21
    ids = [f"trajectory-{index:02d}" for index in range(count - 1)] + ["stop"]
    poses = np.zeros((count, 15, 3), dtype=np.float32)
    controls = np.zeros((count, 15, 2), dtype=np.float32)
    swept = np.zeros((count, grid.height, grid.width), dtype=np.float32)
    swept[:, 0, 0] = 1.0
    tta = np.full_like(swept, -1.0)
    tta[:, 0, 0] = 0.2
    braking = np.zeros_like(swept)
    braking[:, 1, 1] = np.float32(-0.25)
    centerline = swept.copy()
    task_costs = np.zeros(count, dtype=np.float32)
    trajectory_metadata: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for index, trajectory_id in enumerate(ids):
        is_stop = trajectory_id == "stop"
        velocity = 0.0 if is_stop else 0.2 + 0.01 * index
        omega = 0.4 if index == 0 else 0.0
        controls[index, :, 0] = np.float32(velocity)
        controls[index, :, 1] = np.float32(omega)
        times = np.arange(1, 16, dtype=np.float64) * 0.2
        if omega == 0.0:
            poses[index, :, 0] = (velocity * times).astype(np.float32)
        else:
            yaw = omega * times
            poses[index, :, 0] = ((velocity / omega) * np.sin(yaw)).astype(
                np.float32
            )
            poses[index, :, 1] = (
                (velocity / omega) * (1.0 - np.cos(yaw))
            ).astype(np.float32)
            poses[index, :, 2] = yaw.astype(np.float32)
        metadata = {
            "dt_s": 0.2,
            "first_pose_time_s": 0.2,
            "last_pose_time_s": 3.0,
            "trajectory_steps": 15,
            "pose_time_layout_version": SOP04_POSE_TIME_LAYOUT_VERSION,
            "v": velocity,
            "omega": omega,
            "is_stop": is_stop,
            "is_reverse": False,
            "braking_deceleration_mps2": 1.0,
        }
        trajectory_metadata.append(metadata)
        manifest_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "array_index": index,
                "trajectory_id": trajectory_id,
                "trajectory_steps": 15,
                "trajectory_bank_version": SOP04_BANK_VERSION,
                "pose_time_layout_version": SOP04_POSE_TIME_LAYOUT_VERSION,
                "dt_s": 0.2,
                "first_pose_time_s": 0.2,
                "last_pose_time_s": 3.0,
                "query_map_shape": [grid.height, grid.width],
                "v_mps": velocity,
                "omega_radps": omega,
                "is_stop": is_stop,
                "is_reverse": False,
                "task_cost": 0.0,
            }
        )
    external_summary = _trajectory_summary(grid)
    embedded_summary = {
        key: value
        for key, value in external_summary.items()
        if key not in {"schema_version", "provenance"}
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_bank_version": SOP04_BANK_VERSION,
        "pose_time_layout_version": SOP04_POSE_TIME_LAYOUT_VERSION,
        "summary": embedded_summary,
        "trajectory_ids": ids,
        "trajectory_metadata": trajectory_metadata,
    }
    np.savez(
        root / "trajectory_bank.npz",
        poses=poses,
        controls=controls,
        swept_masks=swept,
        tta_maps=tta,
        braking_maps=braking,
        centerline_maps=centerline,
        task_costs=task_costs,
        meta_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    _jsonl_write(root / "trajectory_manifest.jsonl", manifest_rows)
    _json_write(root / "summary.json", external_summary)
    _json_write(
        root / "audit_report.json",
        {
            "status": "ok",
            "schema_version": SCHEMA_VERSION,
            "trajectory_bank_version": SOP04_BANK_VERSION,
            "pose_time_layout_version": SOP04_POSE_TIME_LAYOUT_VERSION,
            "trajectory_count": count,
            "trajectory_steps": 15,
            "dt_s": 0.2,
            "first_pose_time_s": 0.2,
            "last_pose_time_s": 3.0,
            "artifact_reload_validation": "passed",
            "shape_dtype_finite_validation": "passed_all",
            "future_endpoint_kinematics": "passed_all",
            "query_map_invariants": "passed_all",
            "manifest_array_alignment": "passed_all",
            "summary_npz_alignment": "passed_all",
            "checksum_verification": "passed_all",
            "determinism_reference_exact_match": True,
            "serial_parallel_exact_match": True,
            "checksum_file": "artifact_checksums.sha256",
            "bank_semantic_digest_sha256": "b" * 64,
            "provenance": {
                "canonical_shared_bank": True,
                "code_commit": SOP04_COMMIT,
                "config": "configs/base.yaml",
            },
        },
    )
    _refresh_sop04_checksums(root)
    return root


@pytest.fixture
def grid() -> GridSpec:
    return GridSpec(
        height=4,
        width=5,
        history_steps=8,
        future_steps=15,
        resolution_m=0.2,
    )


@pytest.fixture
def producer_roots(tmp_path: Path, grid: GridSpec) -> tuple[Path, Path]:
    return (
        _write_sop03_bundle(tmp_path / "sop03", grid),
        _write_sop04_bundle(tmp_path / "sop04", grid),
    )


def test_sop03_adapter_accepts_history8_current7_future15_bundle(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    sop03_root, _ = producer_roots

    inputs = load_sop03_split_inputs(sop03_root, "train", grid)

    assert inputs.split == "train"
    assert tuple(sorted(inputs.manifest_index)) == (
        "train-base-a",
        "train-base-b",
        "train-base-c",
    )
    assert tuple(sorted(inputs.typed_libraries)) == (
        "carried_object",
        "human",
        "unknown_dynamic",
    )
    assert len(inputs.typed_libraries["unknown_dynamic"].snippets) == 0
    assert inputs.producer_evidence.code_commit == SOP03_COMMIT
    assert inputs.producer_evidence.completion_policy == "sop03_complete_marker_v1"


def test_sop03_adapter_rejects_legacy_16_sample_three_second_library(
    tmp_path: Path, grid: GridSpec
) -> None:
    root = _write_sop03_bundle(tmp_path / "legacy", grid, snippet_steps=16)

    with pytest.raises(Sop05InputError, match="layout"):
        load_sop03_split_inputs(root, "train", grid)


def test_sop03_adapter_rejects_array_layout_when_summary_claims_v1(
    tmp_path: Path, grid: GridSpec
) -> None:
    root = _write_sop03_bundle(
        tmp_path / "forged-layout",
        grid,
        snippet_steps=16,
        declared_snippet_steps=23,
    )

    with pytest.raises(Sop05InputError, match="array shape"):
        load_sop03_split_inputs(root, "train", grid)


@pytest.mark.parametrize("marker_payload", [None, b"not-empty"])
def test_sop03_adapter_requires_empty_complete_marker(
    producer_roots: tuple[Path, Path], grid: GridSpec, marker_payload: bytes | None
) -> None:
    root, _ = producer_roots
    marker = root / ".producer-complete"
    if marker_payload is None:
        marker.unlink()
    else:
        marker.write_bytes(marker_payload)
    _refresh_sop03_checksums(root)

    with pytest.raises(Sop05InputError, match="producer-complete"):
        load_sop03_split_inputs(root, "train", grid)


@pytest.mark.parametrize(
    ("file_name", "path", "value", "message"),
    [
        ("run_manifest.json", ("status",), "failed", "run status"),
        ("run_manifest.json", ("validation", "status"), "failed", "validation"),
        ("audit_report.json", ("status",), "failed", "audit status"),
        ("audit_report.json", ("code_commit",), "3" * 40, "commit"),
    ],
)
def test_sop03_adapter_rejects_incomplete_audit_or_commit_mismatch(
    producer_roots: tuple[Path, Path],
    grid: GridSpec,
    file_name: str,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    root, _ = producer_roots
    payload = _json_read(root / file_name)
    assert isinstance(payload, dict)
    cursor = payload
    for key in path[:-1]:
        nested = cursor[key]
        assert isinstance(nested, dict)
        cursor = nested
    cursor[path[-1]] = value
    _json_write(root / file_name, payload)
    _refresh_sop03_checksums(root)

    with pytest.raises(Sop05InputError, match=message):
        load_sop03_split_inputs(root, "train", grid)


def test_sop03_adapter_rejects_payload_checksum_mismatch(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    root, _ = producer_roots
    path = root / "base_states/train/summary.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(Sop05InputError, match="checksum mismatch"):
        load_sop03_split_inputs(root, "train", grid)


def test_sop03_adapter_rejects_unsafe_checksum_path(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    root, _ = producer_roots
    manifest = root / "artifact_checksums.sha256"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + f"{'0' * 64}  ../escape\n",
        encoding="utf-8",
    )
    summary = _json_read(root / "artifact_checksum_summary.json")
    assert isinstance(summary, dict)
    summary["checksum_manifest_sha256"] = _sha256(manifest)
    summary["covered_file_count"] = int(summary["covered_file_count"]) + 1
    _json_write(root / "artifact_checksum_summary.json", summary)

    with pytest.raises(Sop05InputError, match="unsafe checksum path"):
        load_sop03_split_inputs(root, "train", grid)


def test_sop03_adapter_rejects_unlisted_payload(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    root, _ = producer_roots
    (root / "unexpected.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(Sop05InputError, match="payload set"):
        load_sop03_split_inputs(root, "train", grid)


def test_sop03_adapter_rejects_split_digest_mismatch(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    root, _ = producer_roots
    summary_path = root / "base_states/train/summary.json"
    summary = _json_read(summary_path)
    assert isinstance(summary, dict)
    provenance = summary["split_provenance"]
    assert isinstance(provenance, dict)
    provenance["split_manifest_digest"] = "bad-digest"
    _json_write(summary_path, summary)
    _refresh_sop03_checksums(root)

    with pytest.raises(Sop05InputError, match="split digest"):
        load_sop03_split_inputs(root, "train", grid)


def test_sop03_adapter_rejects_manifest_count_mismatch(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    root, _ = producer_roots
    summary_path = root / "base_states/train/summary.json"
    summary = _json_read(summary_path)
    assert isinstance(summary, dict)
    summary["accepted_count"] = 4
    _json_write(summary_path, summary)
    _refresh_sop03_checksums(root)

    with pytest.raises(Sop05InputError, match="base state count"):
        load_sop03_split_inputs(root, "train", grid)


def test_sop03_adapter_rejects_unpaired_or_duplicate_manifest_ids(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    root, _ = producer_roots
    manifest_path = root / "base_states/train/oracle_context_manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    rows[1]["base_state_id"] = rows[0]["base_state_id"]
    _jsonl_write(manifest_path, rows)
    _refresh_sop03_checksums(root)

    with pytest.raises(Sop05InputError, match="duplicate.*base_state_id"):
        load_sop03_split_inputs(root, "train", grid)


def test_sop03_pair_is_lazy_loaded_and_runs_frozen_validators(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    root, _ = producer_roots
    invalid = replace(
        _base_state("train-base-a", grid),
        robot_history=np.zeros((7, 3), dtype=np.float32),
    )
    save_dataclass(
        invalid,
        root / "base_states/train/base_states/train-base-a.npz",
    )
    _refresh_sop03_checksums(root)
    inputs = load_sop03_split_inputs(root, "train", grid)

    with pytest.raises(Sop05InputError, match="BaseState contract"):
        inputs.load_pair("train-base-a", grid)


def test_sop03_adapter_reconciles_snippet_source_ids_and_counts(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    root, _ = producer_roots
    source_path = root / "snippets/train/human/source_manifest.jsonl"
    row = json.loads(source_path.read_text(encoding="utf-8"))
    row["snippet_id"] = "different-id"
    _jsonl_write(source_path, [row])
    _refresh_sop03_checksums(root)

    with pytest.raises(Sop05InputError, match="snippet IDs"):
        load_sop03_split_inputs(root, "train", grid)


def test_sop04_adapter_builds_local_trajectories_by_manifest_index(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    _, root = producer_roots

    bank = _load_sop04(root, grid)

    assert len(bank.trajectories) == 21
    assert bank.trajectories[0].trajectory_id == "trajectory-00"
    assert bank.trajectories[-1].trajectory_id == "stop"
    assert tuple(bank.by_id) == tuple(item.trajectory_id for item in bank.trajectories)
    assert bank.producer_evidence.code_commit == SOP04_COMMIT
    assert bank.producer_evidence.completion_policy == SOP04_BANK_VERSION
    assert bank.trajectory_bank_version == SOP04_BANK_VERSION
    assert bank.pose_time_layout_version == SOP04_POSE_TIME_LAYOUT_VERSION
    np.testing.assert_array_equal(
        bank.pose_time_offsets_s,
        (np.arange(15, dtype=np.float64) + 1.0) * 0.2,
    )
    for trajectory in bank.trajectories:
        assert trajectory.poses.flags.owndata
        assert trajectory.swept_mask.flags.owndata


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("trajectory_bank_version", "sop04_audited_bank_v1"),
        ("pose_time_layout_version", "legacy_t0_to_horizon_minus_dt_v0"),
        ("first_pose_time_s", 0.0),
        ("last_pose_time_s", 2.8),
        ("dt_s", None),
    ],
)
def test_sop04_adapter_rejects_stale_or_incomplete_time_contract(
    producer_roots: tuple[Path, Path],
    grid: GridSpec,
    key: str,
    value: object,
) -> None:
    _, root = producer_roots
    audit_path = root / "audit_report.json"
    audit = _json_read(audit_path)
    assert isinstance(audit, dict)
    audit[key] = value
    _json_write(audit_path, audit)
    _refresh_sop04_checksums(root)

    with pytest.raises(Sop05InputError, match="version|time|layout|dt"):
        _load_sop04(root, grid)


def test_sop04_adapter_rejects_t0_poses_resealed_under_v2_labels(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    _, root = producer_roots
    path = root / "trajectory_bank.npz"
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays["poses"][:, :, 0] -= arrays["controls"][:, :, 0] * np.float32(0.2)
    np.savez(path, **arrays)
    _refresh_sop04_checksums(root)

    with pytest.raises(Sop05InputError, match="future endpoint"):
        _load_sop04(root, grid)


def test_sop04_adapter_rejects_external_handoff_digest_tamper(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    _, root = producer_roots
    expected = _sop04_handoff_digest(root)
    (root / "external_handoff_digest.sha256").write_text(
        f"{'0' * 64}  sop04_audited_bank_v2_envelope\n",
        encoding="utf-8",
    )

    with pytest.raises(Sop05InputError, match="external handoff"):
        load_sop04_trajectory_bank(
            root,
            grid,
            expected_external_handoff_digest_sha256=expected,
        )


def test_sop04_adapter_rejects_coherently_resealed_bundle_against_trusted_handoff(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    _, root = producer_roots
    trusted = _sop04_handoff_digest(root)
    audit_path = root / "audit_report.json"
    audit = _json_read(audit_path)
    assert isinstance(audit, dict)
    audit["bank_semantic_digest_sha256"] = "c" * 64
    _json_write(audit_path, audit)
    _refresh_sop04_checksums(root)

    with pytest.raises(Sop05InputError, match="trusted handoff"):
        load_sop04_trajectory_bank(
            root,
            grid,
            expected_external_handoff_digest_sha256=trusted,
        )


def test_sop04_adapter_rejects_extra_checksummed_payload(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    _, root = producer_roots
    (root / "extra.json").write_text("{}\n", encoding="utf-8")
    _refresh_sop04_checksums(root)

    with pytest.raises(Sop05InputError, match="exactly the v2 core"):
        _load_sop04(root, grid)


@pytest.mark.parametrize("layer", ["summary", "npz", "manifest"])
def test_sop04_adapter_rejects_old_layout_token_at_every_artifact_layer(
    producer_roots: tuple[Path, Path],
    grid: GridSpec,
    layer: str,
) -> None:
    _, root = producer_roots
    old = "legacy_t0_to_horizon_minus_dt_v0"
    if layer == "summary":
        summary_path = root / "summary.json"
        summary = _json_read(summary_path)
        assert isinstance(summary, dict)
        summary["pose_time_layout_version"] = old
        _json_write(summary_path, summary)
    elif layer == "npz":
        path = root / "trajectory_bank.npz"
        with np.load(path, allow_pickle=False) as payload:
            arrays = {key: payload[key].copy() for key in payload.files}
        metadata = json.loads(str(arrays["meta_json"]))
        metadata["pose_time_layout_version"] = old
        arrays["meta_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        np.savez(path, **arrays)
    else:
        path = root / "trajectory_manifest.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["pose_time_layout_version"] = old
        _jsonl_write(path, rows)
    _refresh_sop04_checksums(root)

    with pytest.raises(Sop05InputError, match="layout"):
        _load_sop04(root, grid)


@pytest.mark.parametrize("case", ["duplicate", "gap", "out-of-range"])
def test_sop04_adapter_rejects_invalid_manifest_array_index(
    producer_roots: tuple[Path, Path], grid: GridSpec, case: str
) -> None:
    _, root = producer_roots
    manifest_path = root / "trajectory_manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    if case == "duplicate":
        rows[1]["array_index"] = 0
    elif case == "gap":
        rows.pop(1)
    else:
        rows[-1]["array_index"] = 21
    _jsonl_write(manifest_path, rows)
    _refresh_sop04_checksums(root)

    with pytest.raises(Sop05InputError, match="array_index"):
        _load_sop04(root, grid)


def test_sop04_adapter_rejects_query_shape_or_tta_invariant(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    _, root = producer_roots
    path = root / "trajectory_bank.npz"
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays["tta_maps"][0, 1, 1] = np.float32(0.0)
    np.savez(path, **arrays)
    _refresh_sop04_checksums(root)

    with pytest.raises(Sop05InputError, match="TTA"):
        _load_sop04(root, grid)


def test_sop04_adapter_rejects_external_summary_drift(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    _, root = producer_roots
    path = root / "summary.json"
    summary = _json_read(path)
    assert isinstance(summary, dict)
    summary["unexpected"] = "drift"
    _json_write(path, summary)
    _refresh_sop04_checksums(root)

    with pytest.raises(Sop05InputError, match="embedded summary"):
        _load_sop04(root, grid)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("status", "failed", "audit status"),
        ("shape_dtype_finite_validation", "failed", "shape"),
        ("query_map_invariants", "failed", "query"),
        ("manifest_array_alignment", "failed", "manifest"),
        ("serial_parallel_exact_match", False, "serial_parallel"),
    ],
)
def test_sop04_adapter_requires_audited_bank_readiness(
    producer_roots: tuple[Path, Path],
    grid: GridSpec,
    key: str,
    value: object,
    message: str,
) -> None:
    _, root = producer_roots
    path = root / "audit_report.json"
    audit = _json_read(path)
    assert isinstance(audit, dict)
    audit[key] = value
    _json_write(path, audit)

    with pytest.raises(Sop05InputError, match=message):
        _load_sop04(root, grid)


def test_sop04_adapter_rejects_commit_mismatch(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    _, root = producer_roots
    path = root / "summary.json"
    summary = _json_read(path)
    assert isinstance(summary, dict)
    provenance = summary["provenance"]
    assert isinstance(provenance, dict)
    provenance["code_commit"] = "4" * 40
    _json_write(path, summary)
    _refresh_sop04_checksums(root)

    with pytest.raises(Sop05InputError, match="commit"):
        _load_sop04(root, grid)


def test_stable_pair_schedule_ignores_manifest_and_bank_order_and_builtin_hash(
    producer_roots: tuple[Path, Path],
    grid: GridSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sop03_root, sop04_root = producer_roots
    sop03 = load_sop03_split_inputs(sop03_root, "train", grid)
    sop04 = _load_sop04(sop04_root, grid)
    first = build_stable_pair_schedule(
        sop03,
        sop04,
        seed=42,
        max_base_states=2,
        trajectory_count=3,
    )
    records = list(sop03.manifest_index.items())[::-1]
    sop03.manifest_index.clear()
    sop03.manifest_index.update(records)
    trajectories = tuple(reversed(sop04.trajectories))
    reordered_bank = replace(
        sop04,
        trajectories=trajectories,
        by_id={item.trajectory_id: item for item in trajectories},
    )

    def forbidden_hash(value: object) -> int:
        raise AssertionError(f"built-in hash called for {value!r}")

    monkeypatch.setattr(builtins, "hash", forbidden_hash)
    second = build_stable_pair_schedule(
        sop03,
        reordered_bank,
        seed=42,
        max_base_states=2,
        trajectory_count=3,
    )

    assert first == second
    assert len(first) == 6
    assert all(type(item.seed) is int for item in first)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed": True, "max_base_states": 1, "trajectory_count": 1}, "seed"),
        ({"seed": 1, "max_base_states": 0, "trajectory_count": 1}, "max_base_states"),
        ({"seed": 1, "max_base_states": 1, "trajectory_count": False}, "trajectory_count"),
    ],
)
def test_stable_pair_schedule_rejects_nonphysical_arguments(
    producer_roots: tuple[Path, Path],
    grid: GridSpec,
    kwargs: dict[str, object],
    message: str,
) -> None:
    sop03_root, sop04_root = producer_roots
    sop03 = load_sop03_split_inputs(sop03_root, "train", grid)
    sop04 = _load_sop04(sop04_root, grid)

    with pytest.raises((TypeError, ValueError), match=message):
        build_stable_pair_schedule(sop03, sop04, **kwargs)


def test_stable_pair_schedule_truncates_limits_to_available_inputs(
    producer_roots: tuple[Path, Path], grid: GridSpec
) -> None:
    sop03_root, sop04_root = producer_roots
    sop03 = load_sop03_split_inputs(sop03_root, "train", grid)
    sop04 = _load_sop04(sop04_root, grid)

    schedule = build_stable_pair_schedule(
        sop03,
        sop04,
        seed=9,
        max_base_states=100,
        trajectory_count=100,
    )

    assert len(schedule) == 3 * 21
