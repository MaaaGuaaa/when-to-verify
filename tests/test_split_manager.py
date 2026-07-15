"""Behavioral tests for deterministic connected-group dataset splits."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_shared_recording_and_participant_form_one_connected_group():
    from src.datasets.split_manager import make_split_manifest

    records = [
        {
            "frame_id": "a-000",
            "recording_id": "rec-a",
            "session_id": "session-a",
            "participant_id": "ped-shared",
        },
        {
            "frame_id": "a-001",
            "recording_id": "rec-a",
            "session_id": "session-a",
            "participant_id": "ped-a",
        },
        {
            "frame_id": "b-000",
            "recording_id": "rec-b",
            "session_id": "session-b",
            "participant_id": "ped-shared",
        },
        {
            "frame_id": "c-000",
            "recording_id": "rec-c",
            "session_id": "session-c",
            "participant_id": "ped-c",
        },
    ]

    result = make_split_manifest(records, seed=42)
    by_frame = {row["frame_id"]: row for row in result.manifest}

    connected = [by_frame[frame] for frame in ("a-000", "a-001", "b-000")]
    assert len({row["group_id"] for row in connected}) == 1
    assert len({row["split"] for row in connected}) == 1
    assert by_frame["c-000"]["group_id"] != connected[0]["group_id"]


def test_default_ratios_allocate_ten_groups_as_seven_one_one_one():
    from src.datasets.split_manager import make_split_manifest

    records = [
        {
            "recording_id": f"rec-{index:02d}",
            "session_id": f"session-{index:02d}",
            "participant_id": f"ped-{index:02d}",
        }
        for index in range(10)
    ]

    result = make_split_manifest(records, seed=42)
    group_assignments = {
        row["group_id"]: row["split"] for row in result.manifest
    }

    assert Counter(group_assignments.values()) == {
        "train": 7,
        "calibration": 1,
        "val": 1,
        "test": 1,
    }


def test_manifest_bytes_are_independent_of_input_order_for_same_seed():
    from src.datasets.split_manager import make_split_manifest, serialize_manifest

    records = [
        {
            "frame_id": f"frame-{index:02d}",
            "recording_id": f"rec-{index // 2:02d}",
            "session_id": f"session-{index // 2:02d}",
            "participant_id": f"ped-{index // 2:02d}",
        }
        for index in range(20)
    ]

    forward = make_split_manifest(records, seed=17)
    reverse = make_split_manifest(list(reversed(records)), seed=17)

    assert serialize_manifest(forward.manifest) == serialize_manifest(reverse.manifest)


def test_each_split_has_an_independent_seed_namespace():
    from src.datasets.split_manager import make_split_manifest

    records = [
        {
            "recording_id": f"rec-{index:02d}",
            "participant_id": f"ped-{index:02d}",
        }
        for index in range(10)
    ]

    result = make_split_manifest(records, seed=42)
    namespace_by_split: dict[str, set[tuple[str, int]]] = {}
    for row in result.manifest:
        namespace_by_split.setdefault(row["split"], set()).add(
            (row["seed_namespace"], row["generator_seed"])
        )

    assert set(namespace_by_split) == {"train", "calibration", "val", "test"}
    assert all(len(values) == 1 for values in namespace_by_split.values())
    assert len({next(iter(values)) for values in namespace_by_split.values()}) == 4


def test_missing_participant_falls_back_to_recording_and_is_marked_unavailable():
    from src.datasets.split_manager import make_split_manifest

    records = [
        {"frame_id": "000", "recording_id": "rec-missing-participant"},
        {"frame_id": "001", "recording_id": "rec-missing-participant"},
    ]

    result = make_split_manifest(records, seed=42)

    assert len({row["group_id"] for row in result.manifest}) == 1
    assert len({row["split"] for row in result.manifest}) == 1
    assert {row["participant_check"] for row in result.manifest} == {"unavailable"}


def test_empty_source_index_is_rejected():
    from src.datasets.split_manager import SplitIndexError, make_split_manifest

    with pytest.raises(SplitIndexError, match="must not be empty"):
        make_split_manifest([], seed=42)


@pytest.mark.parametrize(
    "records, expected_message",
    [
        (["not-a-mapping"], "row 0 must be a mapping"),
        ([{"recording_id": 7}], "recording_id must be a non-empty string"),
        (
            [{"recording_id": "rec-a", "participant_ids": "ped-a"}],
            "participant_ids must be a list or tuple",
        ),
        (
            [{"recording_id": "rec-a", "timestamp": float("nan")}],
            "timestamp must not contain NaN/Inf",
        ),
        (
            [{"recording_id": "rec-a", "metadata": {"score": float("inf")}}],
            "metadata.score must not contain NaN/Inf",
        ),
    ],
)
def test_source_index_schema_rejects_bad_shape_dtype_and_nonfinite_values(
    records, expected_message
):
    from src.datasets.split_manager import SplitIndexError, make_split_manifest

    with pytest.raises(SplitIndexError, match=expected_message):
        make_split_manifest(records, seed=42)


@pytest.mark.parametrize("reserved_field", ["split", "group_id", "seed_namespace"])
def test_source_index_rejects_preassigned_split_fields(reserved_field):
    from src.datasets.split_manager import SplitIndexError, make_split_manifest

    records = [{"recording_id": "rec-a", reserved_field: "preassigned"}]

    with pytest.raises(SplitIndexError, match="reserved output field"):
        make_split_manifest(records, seed=42)


def test_source_index_rejects_incompatible_schema_version():
    from src.datasets.split_manager import SplitIndexError, make_split_manifest

    records = [{"recording_id": "rec-a", "schema_version": "0.9.0"}]

    with pytest.raises(SplitIndexError, match="schema_version"):
        make_split_manifest(records, seed=42)


def test_written_artifacts_are_complete_and_byte_identical(tmp_path):
    from src.contracts import SCHEMA_VERSION
    from src.datasets.split_manager import make_split_manifest, write_split_artifacts

    records = [
        {
            "frame_id": f"frame-{index:02d}",
            "recording_id": f"rec-{index // 2:02d}",
            "participant_id": f"ped-{index // 2:02d}",
        }
        for index in range(20)
    ]
    first = make_split_manifest(records, seed=42)
    second = make_split_manifest(list(reversed(records)), seed=42)

    first_paths = write_split_artifacts(first, tmp_path / "first")
    second_paths = write_split_artifacts(second, tmp_path / "second")

    assert set(first_paths) == {"manifest", "summary", "overlap_report"}
    for artifact in first_paths:
        assert first_paths[artifact].read_bytes() == second_paths[artifact].read_bytes()

    summary = json.loads(first_paths["summary"].read_text(encoding="utf-8"))
    report = json.loads(first_paths["overlap_report"].read_text(encoding="utf-8"))
    manifest_rows = [
        json.loads(line)
        for line in first_paths["manifest"].read_text(encoding="utf-8").splitlines()
    ]
    assert summary["schema_version"] == SCHEMA_VERSION
    assert report["schema_version"] == SCHEMA_VERSION
    assert {row["schema_version"] for row in manifest_rows} == {SCHEMA_VERSION}
    assert summary["seed"] == 42
    assert summary["source_record_count"] == 20
    assert summary["connected_group_count"] == 10
    assert summary["requested_ratios"] == {
        "train": 0.7,
        "calibration": 0.1,
        "val": 0.1,
        "test": 0.1,
    }
    assert {
        split: stats["group_count"]
        for split, stats in summary["split_statistics"].items()
    } == {"train": 7, "calibration": 1, "val": 1, "test": 1}
    assert report["status"] == "ok"
    assert report["total_overlap_count"] == 0
    assert report["manifest_digest"] == summary["manifest_digest"]


def test_cli_uses_config_ratios_and_is_cross_process_deterministic(tmp_path):
    records = [
        {
            "recording_id": f"rec-{index:02d}",
            "session_id": f"session-{index:02d}",
            "participant_id": f"ped-{index:02d}",
        }
        for index in range(10)
    ]
    index_path = tmp_path / "toy-index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    config_path = tmp_path / "toy-data.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset: thor",
                f"input_manifest: {json.dumps(str(index_path))}",
                "input_format: jsonl",
                "split_ratios:",
                "  train: 0.4",
                "  calibration: 0.3",
                "  val: 0.2",
                "  test: 0.1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "00_make_splits.py"
    output_dirs = [tmp_path / "run-a", tmp_path / "run-b"]

    for output_dir, hash_seed in zip(output_dirs, ("1", "987654")):
        environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
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
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    for filename in (
        "split_manifest.jsonl",
        "split_summary.json",
        "overlap_report.json",
    ):
        assert (output_dirs[0] / filename).read_bytes() == (
            output_dirs[1] / filename
        ).read_bytes()

    summary = json.loads(
        (output_dirs[0] / "split_summary.json").read_text(encoding="utf-8")
    )
    assert summary["seed"] == 29
    assert summary["requested_ratios"] == {
        "train": 0.4,
        "calibration": 0.3,
        "val": 0.2,
        "test": 0.1,
    }
    assert {
        split: stats["group_count"]
        for split, stats in summary["split_statistics"].items()
    } == {"train": 4, "calibration": 3, "val": 2, "test": 1}
