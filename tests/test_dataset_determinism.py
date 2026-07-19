"""Determinism, integrity, and leakage tests for schema-v3 risk shards."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.contracts import GridSpec, RiskSample, SCHEMA_VERSION
from src.datasets import shard_writer
from src.datasets.shard_writer import (
    RISK_SHARD_LAYOUT_VERSION,
    LoadedRiskShard,
    load_risk_shard,
    write_risk_shard,
)
from src.datasets.split_manager import SplitLeakageError
from src.generation.observation_renderer import RENDERER_LAYOUT_VERSION
from src.generation.risk_gt import RISK_GT_VERSION
from src.contracts import POSE_TIME_LAYOUT_VERSION


def _grid() -> GridSpec:
    return GridSpec(
        height=3,
        width=4,
        history_steps=2,
        future_steps=3,
        resolution_m=0.5,
    )


def _sample(
    sample_id: str,
    *,
    split: str = "train",
    value: float = 1.0,
    collision: bool = False,
    base_recording_id: str | None = None,
    base_session_id: str = "base-session-shared",
    source_recording_id: str | None = None,
    source_session_id: str = "source-session-shared",
    snippet_id: str | None = None,
    pair_group_id: str | None = None,
    seed_namespace: str | None = None,
) -> RiskSample:
    grid = _grid()
    provenance = {
        "base_recording_id": base_recording_id or f"base-recording-{sample_id}",
        "base_session_id": base_session_id,
        "source_recording_id": (
            source_recording_id or f"source-recording-{sample_id}"
        ),
        "source_session_id": source_session_id,
        "dynamic_object_snippet_id": snippet_id or f"snippet-{sample_id}",
        "seed_namespace": seed_namespace or f"seed/{split}/{sample_id}",
    }
    return RiskSample(
        sample_id=sample_id,
        split=split,
        base_state_id=f"base-{sample_id}",
        pair_group_id=pair_group_id or f"pair-{sample_id}",
        event_type="collision" if collision else "near_miss",
        bev_history=np.full(
            (grid.history_steps, grid.n_history_channels, grid.height, grid.width),
            value,
            dtype=np.float32,
        ),
        state_channels=np.full(
            (grid.n_state_channels, grid.height, grid.width),
            value + 1.0,
            dtype=np.float32,
        ),
        trajectory_channels=np.full(
            (grid.n_trajectory_channels, grid.height, grid.width),
            value + 2.0,
            dtype=np.float32,
        ),
        robot_state=np.asarray([value, -value], dtype=np.float32),
        collision_label=int(collision),
        risk_severity=1.0 if collision else 0.4,
        min_clearance=-0.1 if collision else 0.2,
        near_miss=0 if collision else 1,
        first_collision_time=0.2 if collision else None,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "renderer": {
                "renderer_layout_version": RENDERER_LAYOUT_VERSION,
                "base_state_id": f"base-{sample_id}",
                "sensor_config_digest": f"sensor-{sample_id}",
                "static_occupancy_digest": f"static-{sample_id}",
            },
            "trajectory_id": f"trajectory-{sample_id}",
            "provenance": provenance,
            "label_audit": {
                "risk_gt_version": RISK_GT_VERSION,
                "pose_time_layout_version": POSE_TIME_LAYOUT_VERSION,
                "critical_object_id": f"object-{sample_id}",
                "critical_object_type": "human",
                "time_to_min_clearance_s": 0.2,
                "has_hidden_target": True,
            },
        },
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _external_audit_row(**updates: str) -> dict[str, str]:
    row = {
        "split": "test",
        "base_recording_id": "base-recording-external",
        "base_session_id": "base-session-shared",
        "source_recording_id": "source-recording-external",
        "source_session_id": "source-session-shared",
        "source_snippet_id": "snippet-external",
        "pair_group_id": "pair-external",
        "seed_namespace": "seed/test/external",
    }
    row.update(updates)
    return row


def _rewrite_npz(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    mutate(arrays)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def test_reordered_input_has_identical_manifest_and_semantic_digest(
    tmp_path: Path,
) -> None:
    samples = (
        _sample("sample-b", value=2.0, collision=True),
        _sample("sample-a", value=1.0),
    )
    left = tmp_path / "left"
    right = tmp_path / "right"

    left_paths = write_risk_shard(
        samples,
        left,
        grid=_grid(),
        shard_index=3,
        expected_sample_count=2,
    )
    right_paths = write_risk_shard(
        tuple(reversed(samples)),
        right,
        grid=_grid(),
        shard_index=3,
        expected_sample_count=2,
    )
    loaded_left = load_risk_shard(left, grid=_grid())
    loaded_right = load_risk_shard(right, grid=_grid())

    assert isinstance(loaded_left, LoadedRiskShard)
    assert [sample.sample_id for sample in loaded_left.samples] == [
        "sample-a",
        "sample-b",
    ]
    assert left_paths["manifest"].read_bytes() == right_paths["manifest"].read_bytes()
    assert left_paths["summary"].read_bytes() == right_paths["summary"].read_bytes()
    assert loaded_left.manifest_digest == loaded_right.manifest_digest
    assert loaded_left.semantic_digest == loaded_right.semantic_digest
    assert loaded_left.summary["boundary"] == {
        "first_sample_id": "sample-a",
        "last_sample_id": "sample-b",
        "sample_count": 2,
    }


def test_npz_is_pickle_free_and_optional_time_uses_value_and_validity_mask(
    tmp_path: Path,
) -> None:
    paths = write_risk_shard(
        (_sample("safe"), _sample("collision", collision=True)),
        tmp_path / "shard",
        grid=_grid(),
        expected_sample_count=2,
    )

    with np.load(paths["payload"], allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
        assert np.isfinite(archive["first_collision_time_value"]).all()
        by_id = {
            sample_id: index
            for index, sample_id in enumerate(
                json.loads(str(archive["meta_json"]))["sample_ids"]
            )
        }
        assert archive["first_collision_time_valid"][by_id["safe"]] == 0
        assert archive["first_collision_time_value"][by_id["safe"]] == 0.0
        assert archive["first_collision_time_valid"][by_id["collision"]] == 1
        assert archive["first_collision_time_value"][by_id["collision"]] == 0.2


def test_manifest_jsonl_is_canonical_compact_and_finite(tmp_path: Path) -> None:
    paths = write_risk_shard(
        (_sample("sample-b"), _sample("sample-a")),
        tmp_path / "shard",
        grid=_grid(),
        expected_sample_count=2,
    )

    payload = paths["manifest"].read_bytes()
    assert payload.endswith(b"\n")
    lines = payload.decode("utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert lines == [_canonical_json(row) for row in rows]
    assert [row["sample_id"] for row in rows] == ["sample-a", "sample-b"]
    assert b"NaN" not in payload and b"Infinity" not in payload


@pytest.mark.parametrize("old_schema", ["1.0.0", "2.0.0"])
def test_loader_rejects_old_schema_shards(tmp_path: Path, old_schema: str) -> None:
    paths = write_risk_shard(
        (_sample("sample-a"),),
        tmp_path / "shard",
        grid=_grid(),
        expected_sample_count=1,
    )
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["schema_version"] = old_schema
    paths["summary"].write_text(_canonical_json(summary) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_risk_shard(tmp_path / "shard", grid=_grid())


def test_layout_v2_is_frozen_and_loader_rejects_v1(tmp_path: Path) -> None:
    assert RISK_SHARD_LAYOUT_VERSION == "risk_shard_npz_jsonl_v2"
    paths = write_risk_shard(
        (_sample("sample-a"),),
        tmp_path / "shard",
        grid=_grid(),
        expected_sample_count=1,
    )
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["layout_version"] = "risk_shard_npz_jsonl_v1"
    paths["summary"].write_text(_canonical_json(summary) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported risk shard layout"):
        load_risk_shard(tmp_path / "shard", grid=_grid())


@pytest.mark.parametrize("tamper", ["content", "dtype", "shape", "metadata"])
def test_loader_fails_closed_on_tampering(tmp_path: Path, tamper: str) -> None:
    paths = write_risk_shard(
        (_sample("sample-a"),),
        tmp_path / "shard",
        grid=_grid(),
        expected_sample_count=1,
    )
    if tamper == "metadata":
        row = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        row["event_type"] = "tampered"
        paths["manifest"].write_text(_canonical_json(row) + "\n", encoding="utf-8")
    elif tamper == "content":
        _rewrite_npz(
            paths["payload"],
            lambda arrays: arrays["bev_history"].__setitem__((0, 0, 0, 0, 0), 9.0),
        )
    elif tamper == "dtype":
        _rewrite_npz(
            paths["payload"],
            lambda arrays: arrays.__setitem__(
                "risk_severity", arrays["risk_severity"].astype(np.float32)
            ),
        )
    else:
        _rewrite_npz(
            paths["payload"],
            lambda arrays: arrays.__setitem__(
                "robot_state", arrays["robot_state"].reshape(2)
            ),
        )

    with pytest.raises((TypeError, ValueError), match="digest|dtype|shape|manifest"):
        load_risk_shard(tmp_path / "shard", grid=_grid())


@pytest.mark.parametrize("missing", ["payload", "manifest", "summary"])
def test_loader_rejects_partial_publication(tmp_path: Path, missing: str) -> None:
    paths = write_risk_shard(
        (_sample("sample-a"),),
        tmp_path / "shard",
        grid=_grid(),
        expected_sample_count=1,
    )
    paths[missing].unlink()

    with pytest.raises(ValueError, match="incomplete shard"):
        load_risk_shard(tmp_path / "shard", grid=_grid())


def test_loader_rejects_unexpected_partial_file(tmp_path: Path) -> None:
    output = tmp_path / "shard"
    write_risk_shard(
        (_sample("sample-a"),),
        output,
        grid=_grid(),
        expected_sample_count=1,
    )
    (output / "orphan.tmp").write_text("partial", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected shard files"):
        load_risk_shard(output, grid=_grid())


@pytest.mark.parametrize(
    "missing_key",
    [
        "base_recording_id",
        "base_session_id",
        "source_recording_id",
        "source_session_id",
        "dynamic_object_snippet_id",
        "seed_namespace",
    ],
)
def test_writer_rejects_missing_required_provenance(
    tmp_path: Path, missing_key: str
) -> None:
    sample = _sample("sample-a")
    metadata = json.loads(json.dumps(sample.metadata))
    del metadata["provenance"][missing_key]

    with pytest.raises((ValueError, SplitLeakageError), match="provenance"):
        write_risk_shard(
            (replace(sample, metadata=metadata),),
            tmp_path / "shard",
            grid=_grid(),
            expected_sample_count=1,
        )


def test_session_overlap_is_allowed_and_reported(tmp_path: Path) -> None:
    audit_records = (_external_audit_row(),)
    output = tmp_path / "shard"

    write_risk_shard(
        (_sample("sample-a"),),
        output,
        grid=_grid(),
        expected_sample_count=1,
        split_audit_records=audit_records,
    )
    loaded = load_risk_shard(
        output,
        grid=_grid(),
        split_audit_records=audit_records,
    )

    assert loaded.leakage_report["status"] == "ok"
    for identity in ("base_identity", "source_identity"):
        report = loaded.leakage_report[identity]
        assert report["field_policies"]["session"] == "allowed_reported"
        assert report["fields"]["session"]["overlap_count"] == 1
    assert loaded.leakage_report["combined_identity"]["fields"]["session"][
        "overlap_count"
    ] == 2
    assert loaded.leakage_report["allowed_overlap_count"] == 2
    assert loaded.leakage_report["disallowed_overlap_count"] == 0


@pytest.mark.parametrize(
    ("field", "external_key", "external_value"),
    [
        ("base recording", "base_recording_id", "base-recording-sample-a"),
        (
            "source recording",
            "source_recording_id",
            "source-recording-sample-a",
        ),
        (
            "recording",
            "source_recording_id",
            "base-recording-sample-a",
        ),
        (
            "recording",
            "base_recording_id",
            "source-recording-sample-a",
        ),
        ("snippet", "source_snippet_id", "snippet-sample-a"),
        ("pair_group", "pair_group_id", "pair-sample-a"),
        ("seed_namespace", "seed_namespace", "seed/train/sample-a"),
    ],
)
def test_dual_recording_snippet_and_pair_overlap_are_forbidden(
    tmp_path: Path,
    field: str,
    external_key: str,
    external_value: str,
) -> None:
    audit_row = _external_audit_row(**{external_key: external_value})

    with pytest.raises(SplitLeakageError, match=field):
        write_risk_shard(
            (_sample("sample-a"),),
            tmp_path / "shard",
            grid=_grid(),
            expected_sample_count=1,
            split_audit_records=(audit_row,),
        )
    assert not (tmp_path / "shard").exists()


def test_writer_rejects_duplicates_mixed_splits_bad_boundary_and_overwrite(
    tmp_path: Path,
) -> None:
    sample = _sample("sample-a")
    with pytest.raises(ValueError, match="duplicate sample_id"):
        write_risk_shard(
            (sample, sample),
            tmp_path / "duplicate",
            grid=_grid(),
            expected_sample_count=2,
        )
    with pytest.raises(ValueError, match="mixed split"):
        write_risk_shard(
            (sample, _sample("sample-b", split="test")),
            tmp_path / "mixed",
            grid=_grid(),
            expected_sample_count=2,
        )
    with pytest.raises(ValueError, match="expected_sample_count"):
        write_risk_shard(
            (sample,),
            tmp_path / "boundary",
            grid=_grid(),
            expected_sample_count=2,
        )

    output = tmp_path / "immutable"
    write_risk_shard(
        (sample,), output, grid=_grid(), expected_sample_count=1
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        write_risk_shard(
            (sample,), output, grid=_grid(), expected_sample_count=1
        )


def test_failed_formal_reload_cleans_only_its_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "shard"
    foreign = tmp_path / ".shard.staging-foreign"
    foreign.mkdir()
    marker = foreign / "keep.txt"
    marker.write_text("owned elsewhere", encoding="utf-8")

    def _fail_reload(*args, **kwargs):
        raise ValueError("forced formal reload failure")

    monkeypatch.setattr(shard_writer, "load_risk_shard", _fail_reload)
    with pytest.raises(ValueError, match="formal reload"):
        write_risk_shard(
            (_sample("sample-a"),),
            output,
            grid=_grid(),
            expected_sample_count=1,
        )

    assert not output.exists()
    assert marker.read_text(encoding="utf-8") == "owned elsewhere"
    staging = sorted(tmp_path.glob(".shard.staging-*"))
    assert staging == [foreign]
