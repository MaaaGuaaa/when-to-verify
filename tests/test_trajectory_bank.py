"""Production trajectory-bank behavior for SOP-04."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.utils.config import load_config


def test_parallel_bank_matches_serial_and_materializes_main_distribution():
    from src.planning.trajectory_bank import build_trajectory_bank

    config = load_config()
    serial = build_trajectory_bank(
        config,
        braking_deceleration_mps2=1.0,
        workers=1,
    )
    parallel = build_trajectory_bank(
        config,
        braking_deceleration_mps2=1.0,
        workers=8,
    )

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


def test_bank_round_trip_is_numeric_and_refuses_overwrite(tmp_path):
    from src.planning.trajectory_bank import (
        build_trajectory_bank,
        load_trajectory_bank,
        write_trajectory_bank,
    )

    bank = build_trajectory_bank(
        load_config(), braking_deceleration_mps2=1.0, workers=2
    )
    output_dir = tmp_path / "canonical-bank"
    paths = write_trajectory_bank(
        bank,
        output_dir,
        provenance={"code_commit": "test-commit", "config": "base.yaml"},
    )
    restored = load_trajectory_bank(paths["bank"])

    assert set(paths) == {"directory", "bank", "manifest", "summary"}
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
    with np.load(paths["bank"], allow_pickle=False) as payload:
        assert payload["poses"].dtype == np.float32
        assert payload["controls"].dtype == np.float32
        assert payload["swept_masks"].dtype == np.float32
        assert payload["tta_maps"].dtype == np.float32
        assert payload["braking_maps"].dtype == np.float32
        assert payload["centerline_maps"].dtype == np.float32
        assert payload["task_costs"].dtype == np.float32
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_trajectory_bank(bank, output_dir, provenance={})


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
