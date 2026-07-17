"""THÖR recording-generalization metadata and frozen-split tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _write_header_csv(raw_root: Path, recording_id: str) -> Path:
    scenario = raw_root / "Scenario_1"
    scenario.mkdir(parents=True, exist_ok=True)
    path = scenario / f"THOR-Magni_{recording_id}.csv"
    path.write_text(
        f"FILE_ID,{recording_id},\nFrame,Time\n1,0.01\n",
        encoding="utf-8",
    )
    return path


def _toy_recordings(raw_root: Path) -> tuple[str, ...]:
    recording_ids = (
        "120522_SC1A_R1",
        "120522_SC1A_R2",
        "120522_SC1B_R1",
        "120522_SC1B_R2",
    )
    for recording_id in recording_ids:
        _write_header_csv(raw_root, recording_id)
    return recording_ids


def _write_assignment(path: Path, recording_ids: tuple[str, ...]) -> Path:
    payload = {
        "train": [f"thor_magni::{recording_ids[0]}"],
        "calibration": [f"thor_magni::{recording_ids[1]}"],
        "validation": [f"thor_magni::{recording_ids[2]}"],
        "test": [f"thor_magni::{recording_ids[3]}"],
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _write_config(
    path: Path, *, raw_root: Path, assignment_manifest: Path
) -> Path:
    path.write_text(
        "\n".join(
            [
                "dataset: thor",
                "evaluation_scope: unseen_recording_within_known_sessions",
                "grouping_unit: recording_id",
                "recording_overlap_policy: forbidden",
                "session_overlap_policy: allowed_reported",
                "participant_overlap_policy: unavailable",
                "seed: 42",
                f"raw_root: {json.dumps(str(raw_root))}",
                "assignment_manifest: "
                f"{json.dumps(str(assignment_manifest))}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_metadata_index_uses_official_file_id_and_recording_day(tmp_path):
    from src.datasets.thor_split import index_thor_recording_metadata

    raw_root = tmp_path / "raw"
    recording_id = "120522_SC1A_R1"
    _write_header_csv(raw_root, recording_id)

    rows = index_thor_recording_metadata(raw_root)

    assert rows == (
        {
            "recording_id": recording_id,
            "session_id": "120522",
            "source_path": f"Scenario_1/THOR-Magni_{recording_id}.csv",
        },
    )


@pytest.mark.parametrize(
    ("filename_id", "header_id", "message"),
    [
        (
            "120522_SC1A_R1",
            "130522_SC1A_R1",
            "FILE_ID does not match filename",
        ),
        (
            "bad-session_SC1A_R1",
            "bad-session_SC1A_R1",
            "six-digit recording day",
        ),
    ],
)
def test_metadata_index_rejects_bad_identity(
    tmp_path, filename_id, header_id, message
):
    from src.datasets.split_manager import SplitIndexError
    from src.datasets.thor_split import index_thor_recording_metadata

    path = _write_header_csv(tmp_path / "raw", filename_id)
    path.write_text(
        f"FILE_ID,{header_id},\nFrame,Time\n1,0.01\n",
        encoding="utf-8",
    )

    with pytest.raises(SplitIndexError, match=message):
        index_thor_recording_metadata(tmp_path / "raw")


def test_frozen_assignment_normalizes_prefix_and_validation(tmp_path):
    from src.datasets.thor_split import load_frozen_recording_assignment

    recording_ids = (
        "120522_SC1A_R1",
        "120522_SC1A_R2",
        "120522_SC1B_R1",
        "120522_SC1B_R2",
    )
    assignment_path = _write_assignment(
        tmp_path / "assignment.json", recording_ids
    )

    assignments = load_frozen_recording_assignment(assignment_path)

    assert assignments == {
        recording_ids[0]: "train",
        recording_ids[1]: "calibration",
        recording_ids[2]: "val",
        recording_ids[3]: "test",
    }


def test_thor_split_artifacts_are_complete_and_deterministic(tmp_path):
    from src.datasets.thor_split import (
        build_thor_recording_split,
        write_thor_recording_split_artifacts,
    )

    raw_root = tmp_path / "raw"
    recording_ids = _toy_recordings(raw_root)
    assignment_path = _write_assignment(
        tmp_path / "assignment.json", recording_ids
    )
    build = build_thor_recording_split(
        raw_root=raw_root,
        assignment_manifest=assignment_path,
        seed=17,
    )

    first = write_thor_recording_split_artifacts(build, tmp_path / "first")
    second = write_thor_recording_split_artifacts(build, tmp_path / "second")

    assert set(first) == {
        "metadata",
        "manifest",
        "summary",
        "overlap_report",
    }
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()
    rows = [
        json.loads(line)
        for line in first["manifest"].read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads(first["summary"].read_text(encoding="utf-8"))
    report = json.loads(first["overlap_report"].read_text(encoding="utf-8"))
    assert len(rows) == 4
    assert {row["session_id"] for row in rows} == {"120522"}
    assert {row["evaluation_scope"] for row in rows} == {
        "unseen_recording_within_known_sessions"
    }
    assert summary["source_assignment_sha256"]
    assert summary["metadata_digest"]
    assert summary["split_statistics"] == {
        split: {"record_count": 1, "actual_record_ratio": 0.25}
        for split in ("train", "calibration", "val", "test")
    }
    assert report["status"] == "ok"
    assert report["allowed_overlap_count"] == 1
    assert report["disallowed_overlap_count"] == 0
    assert report["fields"]["session"]["overlaps"] == [
        {
            "value": "120522",
            "splits": ["calibration", "test", "train", "val"],
        }
    ]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_thor_recording_split_artifacts(build, tmp_path / "first")


def test_thor_split_rejects_assignment_metadata_mismatch(tmp_path):
    from src.datasets.split_manager import SplitIndexError
    from src.datasets.thor_split import build_thor_recording_split

    raw_root = tmp_path / "raw"
    recording_ids = _toy_recordings(raw_root)
    assignment_path = _write_assignment(
        tmp_path / "assignment.json", recording_ids[:-1] + ("120522_SC2_R1",)
    )

    with pytest.raises(SplitIndexError, match="assignment id mismatch"):
        build_thor_recording_split(
            raw_root=raw_root,
            assignment_manifest=assignment_path,
            seed=42,
        )


def test_thor_split_cli_is_cross_process_deterministic(tmp_path):
    raw_root = tmp_path / "raw"
    recording_ids = _toy_recordings(raw_root)
    assignment_path = _write_assignment(
        tmp_path / "assignment.json", recording_ids
    )
    config_path = _write_config(
        tmp_path / "config.yaml",
        raw_root=raw_root,
        assignment_manifest=assignment_path,
    )
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "00_freeze_thor_recording_split.py"
    output_dirs = (tmp_path / "run-a", tmp_path / "run-b")

    for output_dir, hash_seed in zip(output_dirs, ("1", "987654")):
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--config",
                str(config_path),
                "--seed",
                "29",
                "--output-dir",
                str(output_dir),
            ],
            cwd=root,
            env=dict(os.environ, PYTHONHASHSEED=hash_seed),
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "recording_count=4" in completed.stdout
        assert "allowed_session_overlap_count=1" in completed.stdout

    for name in (
        "recording_metadata.jsonl",
        "split_manifest.jsonl",
        "split_summary.json",
        "overlap_report.json",
    ):
        assert (output_dirs[0] / name).read_bytes() == (
            output_dirs[1] / name
        ).read_bytes()
