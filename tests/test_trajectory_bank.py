"""Production trajectory-bank behavior for SOP-04."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.utils.config import load_config


@pytest.fixture(scope="module")
def canonical_banks():
    from src.planning.trajectory_bank import build_trajectory_bank

    config = load_config()
    return (
        build_trajectory_bank(
            config,
            braking_deceleration_mps2=1.0,
            workers=1,
        ),
        build_trajectory_bank(
            config,
            braking_deceleration_mps2=1.0,
            workers=8,
        ),
    )


def test_parallel_bank_matches_serial_and_materializes_main_distribution(
    canonical_banks,
):
    from src.planning.trajectory_bank import trajectory_bank_semantic_digest

    serial, parallel = canonical_banks

    assert len(serial.trajectories) == len(parallel.trajectories) == 21
    assert serial.summary["candidate_count"] == 21
    assert serial.summary["accepted_count"] == 21
    assert serial.summary["rejected_count"] == 0
    assert serial.summary["acceptance_rate"] == 1.0
    assert serial.summary["meets_minimum_acceptance_rate"] is True
    assert serial.summary["braking_deceleration_mps2"] == 1.0
    assert serial.summary["state_specific_filtering"] is False
    assert parallel.summary["workers_requested"] == 8
    assert parallel.summary["workers_used"] == 8
    assert (
        serial.summary["trajectory_bank_version"]
        == "sop04_audited_bank_v2"
    )
    assert (
        serial.summary["pose_time_layout_version"]
        == "future_endpoints_dt_to_horizon_v1"
    )
    assert serial.summary["first_pose_time_s"] == pytest.approx(0.2)
    assert serial.summary["last_pose_time_s"] == pytest.approx(3.0)
    assert serial.summary["dt_s"] == pytest.approx(0.2)
    assert trajectory_bank_semantic_digest(
        serial
    ) == trajectory_bank_semantic_digest(parallel)
    assert [item.trajectory_id for item in serial.trajectories] == [
        item.trajectory_id for item in parallel.trajectories
    ]
    assert serial.trajectories[-1].trajectory_id == "stop"
    assert all(not item.metadata["is_reverse"] for item in serial.trajectories)
    for expected, actual in zip(serial.trajectories, parallel.trajectories):
        assert expected.metadata == actual.metadata
        assert expected.task_cost == actual.task_cost == 0.0
        for name in (
            "poses",
            "controls",
            "swept_mask",
            "tta_map",
            "braking_map",
            "centerline_map",
        ):
            np.testing.assert_array_equal(
                getattr(actual, name), getattr(expected, name)
            )


def test_bank_rejects_invalid_worker_count():
    from src.planning.trajectory_bank import build_trajectory_bank

    with pytest.raises(ValueError, match="workers must be a positive integer"):
        build_trajectory_bank(
            load_config(),
            braking_deceleration_mps2=1.0,
            workers=0,
        )


def test_bank_round_trip_is_numeric_and_refuses_overwrite(
    tmp_path, canonical_banks
):
    from src.planning.trajectory_bank import (
        audit_trajectory_bank_artifact,
        load_trajectory_bank,
        write_trajectory_bank,
    )

    serial_reference, bank = canonical_banks
    output_dir = tmp_path / "canonical-bank"
    paths = write_trajectory_bank(
        bank,
        output_dir,
        provenance={"code_commit": "test-commit", "config": "base.yaml"},
        determinism_reference=serial_reference,
    )
    restored = load_trajectory_bank(paths["bank"])

    assert set(paths) == {
        "directory",
        "bank",
        "manifest",
        "summary",
        "checksums",
        "audit",
        "handoff_digest",
    }
    assert len(restored.trajectories) == 21
    assert restored.summary == bank.summary
    assert len(
        (output_dir / "trajectory_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 21
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["schema_version"] == "2.0.0"
    assert summary["provenance"]["code_commit"] == "test-commit"
    assert summary["trajectory_bank_version"] == "sop04_audited_bank_v2"
    assert (
        summary["pose_time_layout_version"]
        == "future_endpoints_dt_to_horizon_v1"
    )
    with np.load(paths["bank"], allow_pickle=False) as payload:
        assert payload["poses"].dtype == np.float32
        assert payload["controls"].dtype == np.float32
        assert payload["swept_masks"].dtype == np.float32
        assert payload["tta_maps"].dtype == np.float32
        assert payload["braking_maps"].dtype == np.float32
        assert payload["centerline_maps"].dtype == np.float32
        assert payload["task_costs"].dtype == np.float32
        metadata = json.loads(str(payload["meta_json"]))
        legacy_payload = {
            name: payload[name].copy()
            for name in payload.files
            if name != "meta_json"
        }
    metadata["trajectory_bank_version"] = "sop04_audited_bank_v1"
    metadata["summary"]["trajectory_bank_version"] = "sop04_audited_bank_v1"
    legacy_path = tmp_path / "legacy-bank.npz"
    np.savez(
        legacy_path,
        **legacy_payload,
        meta_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="trajectory bank version"):
        load_trajectory_bank(legacy_path)

    checksums = paths["checksums"].read_text(encoding="utf-8").splitlines()
    assert len(checksums) == 3
    checksum_manifest_sha256 = hashlib.sha256(
        paths["checksums"].read_bytes()
    ).hexdigest()
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["status"] == "ok"
    assert audit["artifact_reload_validation"] == "passed"
    assert audit["determinism_reference_exact_match"] is True
    assert audit["serial_parallel_exact_match"] is True
    assert audit["checksum_manifest_sha256"] == checksum_manifest_sha256
    assert audit["trajectory_bank_version"] == "sop04_audited_bank_v2"
    assert audit["first_pose_time_s"] == pytest.approx(0.2)
    assert audit["last_pose_time_s"] == pytest.approx(3.0)
    handoff_digest = paths["handoff_digest"].read_text(
        encoding="utf-8"
    ).split()[0]
    expected_handoff_hasher = hashlib.sha256()
    expected_handoff_hasher.update(
        b"sop04_audited_bank_v2_external_handoff\0"
    )
    expected_handoff_hasher.update(paths["checksums"].read_bytes())
    expected_handoff_hasher.update(b"\0")
    expected_handoff_hasher.update(paths["audit"].read_bytes())
    assert handoff_digest == expected_handoff_hasher.hexdigest()

    repeated_dir = tmp_path / "canonical-bank-repeat"
    repeated_paths = write_trajectory_bank(
        bank,
        repeated_dir,
        provenance={"code_commit": "test-commit", "config": "base.yaml"},
        determinism_reference=serial_reference,
    )
    for key in (
        "bank",
        "manifest",
        "summary",
        "checksums",
        "audit",
        "handoff_digest",
    ):
        assert hashlib.sha256(paths[key].read_bytes()).digest() == hashlib.sha256(
            repeated_paths[key].read_bytes()
        ).digest()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_trajectory_bank(
            bank,
            output_dir,
            provenance={},
            determinism_reference=serial_reference,
        )
    paths["manifest"].write_text(
        paths["manifest"].read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest count"):
        audit_trajectory_bank_artifact(
            output_dir,
            expected_bank=bank,
            provenance={"code_commit": "test-commit", "config": "base.yaml"},
            determinism_reference=serial_reference,
        )


def test_cli_builds_bank_with_explicit_deceleration_and_workers(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "cli-bank"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/04_build_trajectory_bank.py"),
            "--config",
            str(root / "configs/base.yaml"),
            "--braking-deceleration-mps2",
            "1.0",
            "--workers",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "candidate_count=21" in completed.stdout
    assert "accepted_count=21" in completed.stdout
    assert "workers_requested=2" in completed.stdout
    assert "workers_used=2" in completed.stdout
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["braking_deceleration_mps2"] == 1.0
    assert summary["workers_requested"] == 2
    assert summary["trajectory_bank_version"] == "sop04_audited_bank_v2"
    assert (
        summary["pose_time_layout_version"]
        == "future_endpoints_dt_to_horizon_v1"
    )
    audit = json.loads(
        (output_dir / "audit_report.json").read_text(encoding="utf-8")
    )
    assert audit["serial_parallel_exact_match"] is True
