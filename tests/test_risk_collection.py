"""Global split-leakage and integrity tests for SOP07 risk collections."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.contracts import (
    POSE_TIME_LAYOUT_VERSION,
    GridSpec,
    RiskSample,
    SCHEMA_VERSION,
)
from src.datasets.shard_writer import load_risk_shard, write_risk_shard
from src.datasets.split_manager import freeze_preassigned_split, write_split_artifacts
from src.datasets.thor_split import THOR_RECORDING_GENERALIZATION_POLICY
from src.generation.observation_renderer import RENDERER_LAYOUT_VERSION
from src.generation.risk_gt import RISK_GT_VERSION


def _grid() -> GridSpec:
    return GridSpec(
        height=3,
        width=3,
        history_steps=2,
        future_steps=3,
        resolution_m=0.5,
    )


def _sample(
    split: str,
    *,
    sample_id: str | None = None,
    base_recording_id: str | None = None,
    base_session_id: str = "shared-session",
    source_recording_id: str | None = None,
    source_session_id: str = "shared-session",
    snippet_id: str | None = None,
    pair_group_id: str | None = None,
    seed_namespace: str | None = None,
) -> RiskSample:
    sample_id = sample_id or f"sample-{split}"
    base_state_id = f"base-{split}"
    return RiskSample(
        sample_id=sample_id,
        split=split,
        base_state_id=base_state_id,
        pair_group_id=pair_group_id or f"pair-{split}",
        event_type="near_miss",
        bev_history=np.zeros(
            (
                _grid().history_steps,
                _grid().n_history_channels,
                _grid().height,
                _grid().width,
            ),
            dtype=np.float32,
        ),
        state_channels=np.zeros(
            (_grid().n_state_channels, _grid().height, _grid().width),
            dtype=np.float32,
        ),
        trajectory_channels=np.zeros(
            (_grid().n_trajectory_channels, _grid().height, _grid().width),
            dtype=np.float32,
        ),
        robot_state=np.asarray([0.0, 0.0], dtype=np.float32),
        collision_label=0,
        risk_severity=0.4,
        min_clearance=0.2,
        near_miss=1,
        first_collision_time=None,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "renderer": {
                "renderer_layout_version": RENDERER_LAYOUT_VERSION,
                "base_state_id": base_state_id,
                "sensor_config_digest": f"sensor-{split}",
                "static_occupancy_digest": f"static-{split}",
            },
            "trajectory_id": f"trajectory-{split}",
            "provenance": {
                "base_recording_id": (
                    base_recording_id or f"base-recording-{split}"
                ),
                "base_session_id": base_session_id,
                "source_recording_id": (
                    source_recording_id or f"source-recording-{split}"
                ),
                "source_session_id": source_session_id,
                "source_snippet_id": snippet_id or f"snippet-{split}",
                "seed_namespace": (
                    seed_namespace or f"sop07/{split}/event-{split}"
                ),
                # Reusing the root seed across split is intentional.  The namespace,
                # rather than this bare integer, is the leakage boundary.
                "sop05_transplant_seed": 42,
                "sop06_paired_seed": 42,
                "sop07_dataset_seed": 42,
            },
            "label_audit": {
                "risk_gt_version": RISK_GT_VERSION,
                "pose_time_layout_version": POSE_TIME_LAYOUT_VERSION,
                "critical_object_id": f"object-{split}",
                "critical_object_type": "human",
                "time_to_min_clearance_s": 0.2,
                "has_hidden_target": True,
            },
        },
    )


def _write_split_artifacts(root: Path) -> Path:
    rows = []
    assignments = {}
    for split in ("train", "calibration", "val", "test"):
        for role in ("base", "source"):
            recording_id = f"{role}-recording-{split}"
            rows.append(
                {
                    "recording_id": recording_id,
                    "session_id": "shared-session",
                    "source_path": f"{recording_id}.csv",
                }
            )
            assignments[recording_id] = split
    result = freeze_preassigned_split(
        rows,
        assignments,
        seed=42,
        policy=THOR_RECORDING_GENERALIZATION_POLICY,
    )
    output = root / "splits"
    write_split_artifacts(result, output)
    return output


def _split_manifest_digest(split_artifacts: Path) -> str:
    summary = json.loads(
        (split_artifacts / "split_summary.json").read_text(encoding="utf-8")
    )
    return str(summary["manifest_digest"])


def _write_shards(
    root: Path, *, samples: dict[str, RiskSample] | None = None
):
    from src.datasets.risk_collection import RiskCollectionMemberRequest

    requests = []
    for split in ("train", "calibration", "val", "test"):
        relative_path = Path(split) / "shard-000000"
        output = root / relative_path
        write_risk_shard(
            ((samples or {}).get(split, _sample(split)),),
            output,
            grid=_grid(),
            shard_index=0,
            expected_sample_count=1,
        )
        loaded = load_risk_shard(output, grid=_grid())
        requests.append(
            RiskCollectionMemberRequest(
                relative_path=relative_path.as_posix(),
                split=split,
                shard_index=0,
                expected_sample_count=1,
                expected_manifest_digest=loaded.manifest_digest,
                expected_semantic_digest=loaded.semantic_digest,
            )
        )
    return tuple(requests)


def test_four_split_collection_round_trips_without_absolute_paths(tmp_path: Path):
    from src.datasets.risk_collection import (
        RISK_COLLECTION_LAYOUT_VERSION,
        load_risk_collection,
        write_risk_collection,
    )

    split_artifacts = _write_split_artifacts(tmp_path)
    shard_root = tmp_path / "risk-shards"
    requests = _write_shards(shard_root)
    output = tmp_path / "collection"

    paths = write_risk_collection(
        requests,
        output,
        shard_root=shard_root,
        split_artifact_dir=split_artifacts,
        expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
        grid=_grid(),
    )
    loaded = load_risk_collection(
        output,
        shard_root=shard_root,
        split_artifact_dir=split_artifacts,
        expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
        grid=_grid(),
    )

    assert set(paths) == {"members", "leakage_report", "summary"}
    assert loaded.summary["layout_version"] == RISK_COLLECTION_LAYOUT_VERSION
    assert loaded.summary["required_splits"] == [
        "train",
        "calibration",
        "val",
        "test",
    ]
    assert loaded.summary["sample_count"] == 4
    split_summary = json.loads(
        (split_artifacts / "split_summary.json").read_text(encoding="utf-8")
    )
    assert loaded.summary["split_manifest_digest"] == (
        split_summary["manifest_digest"]
    )
    assert loaded.leakage_report["status"] == "ok"
    assert loaded.leakage_report["disallowed_overlap_count"] == 0
    assert loaded.leakage_report["fields"]["session"]["overlap_count"] == 1
    assert loaded.leakage_report["fields"]["session"]["policy"] == (
        "allowed_reported"
    )
    assert loaded.summary["collection_semantic_digest"] == (
        loaded.collection_semantic_digest
    )

    forbidden = str(tmp_path).encode("utf-8")
    for path in paths.values():
        assert forbidden not in path.read_bytes()
    json.dumps(loaded.summary, sort_keys=True, allow_nan=False)


def _prepared_inputs(
    tmp_path: Path, *, samples: dict[str, RiskSample] | None = None
):
    split_artifacts = _write_split_artifacts(tmp_path)
    shard_root = tmp_path / "risk-shards"
    requests = _write_shards(shard_root, samples=samples)
    return split_artifacts, shard_root, requests


def _write_cli_inputs(tmp_path: Path, requests) -> tuple[Path, Path]:
    request_path = tmp_path / "member-requests.jsonl"
    request_path.write_text(
        "\n".join(
            json.dumps(request.__dict__, sort_keys=True, separators=(",", ":"))
            for request in requests
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                'schema_version: "3.0.0"',
                "bev:",
                "  size: 3",
                "  history_steps: 2",
                "  future_steps: 3",
                "  resolution_m: 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return request_path, config_path


def test_collection_requires_all_four_splits(tmp_path: Path):
    from src.datasets.risk_collection import RiskCollectionError, write_risk_collection

    split_artifacts, shard_root, requests = _prepared_inputs(tmp_path)

    with pytest.raises(RiskCollectionError, match="requires all four splits"):
        write_risk_collection(
            requests[:-1],
            tmp_path / "collection",
            shard_root=shard_root,
            split_artifact_dir=split_artifacts,
            expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
            grid=_grid(),
        )


def test_collection_requires_the_expected_frozen_split_manifest(tmp_path: Path):
    from src.datasets.risk_collection import RiskCollectionError, write_risk_collection

    split_artifacts, shard_root, requests = _prepared_inputs(tmp_path)

    with pytest.raises(
        RiskCollectionError, match="expected split manifest digest mismatch"
    ):
        write_risk_collection(
            requests,
            tmp_path / "collection",
            shard_root=shard_root,
            split_artifact_dir=split_artifacts,
            expected_split_manifest_digest="0" * 32,
            grid=_grid(),
        )
    assert not (tmp_path / "collection").exists()


def test_collection_rejects_recording_that_disagrees_with_authority(tmp_path: Path):
    from src.datasets.risk_collection import RiskCollectionError, write_risk_collection

    samples = {
        "test": _sample(
            "test", source_recording_id="base-recording-train"
        )
    }
    split_artifacts, shard_root, requests = _prepared_inputs(
        tmp_path, samples=samples
    )

    with pytest.raises(RiskCollectionError, match="authoritative split mismatch"):
        write_risk_collection(
            requests,
            tmp_path / "collection",
            shard_root=shard_root,
            split_artifact_dir=split_artifacts,
            expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
            grid=_grid(),
        )


def test_collection_rejects_session_that_disagrees_with_authority(tmp_path: Path):
    from src.datasets.risk_collection import RiskCollectionError, write_risk_collection

    samples = {
        "test": _sample("test", source_session_id="drifted-session")
    }
    split_artifacts, shard_root, requests = _prepared_inputs(
        tmp_path, samples=samples
    )

    with pytest.raises(RiskCollectionError, match="authoritative session mismatch"):
        write_risk_collection(
            requests,
            tmp_path / "collection",
            shard_root=shard_root,
            split_artifact_dir=split_artifacts,
            expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
            grid=_grid(),
        )


@pytest.mark.parametrize(
    ("field", "train_sample", "test_sample"),
    [
        (
            "source_snippet",
            _sample("train", snippet_id="shared-snippet"),
            _sample("test", snippet_id="shared-snippet"),
        ),
        (
            "pair_group",
            _sample("train", pair_group_id="shared-pair"),
            _sample("test", pair_group_id="shared-pair"),
        ),
        (
            "sample",
            _sample("train", sample_id="shared-sample"),
            _sample("test", sample_id="shared-sample"),
        ),
        (
            "seed_namespace",
            _sample("train", seed_namespace="shared-seed-namespace"),
            _sample("test", seed_namespace="shared-seed-namespace"),
        ),
    ],
)
def test_collection_rejects_cross_split_identity_overlap(
    tmp_path: Path,
    field: str,
    train_sample: RiskSample,
    test_sample: RiskSample,
):
    from src.datasets.risk_collection import RiskCollectionError, write_risk_collection

    split_artifacts, shard_root, requests = _prepared_inputs(
        tmp_path,
        samples={"train": train_sample, "test": test_sample},
    )

    with pytest.raises(RiskCollectionError, match=field):
        write_risk_collection(
            requests,
            tmp_path / "collection",
            shard_root=shard_root,
            split_artifact_dir=split_artifacts,
            expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
            grid=_grid(),
        )


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("expected_manifest_digest", "expected manifest digest mismatch"),
        ("expected_semantic_digest", "expected semantic digest mismatch"),
    ],
)
def test_collection_requires_expected_shard_digests(
    tmp_path: Path, attribute: str, message: str
):
    from src.datasets.risk_collection import RiskCollectionError, write_risk_collection

    split_artifacts, shard_root, requests = _prepared_inputs(tmp_path)
    requests = list(requests)
    requests[0] = replace(requests[0], **{attribute: "0" * 64})

    with pytest.raises(RiskCollectionError, match=message):
        write_risk_collection(
            requests,
            tmp_path / "collection",
            shard_root=shard_root,
            split_artifact_dir=split_artifacts,
            expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
            grid=_grid(),
        )


def test_collection_bytes_are_independent_of_member_request_order(tmp_path: Path):
    from src.datasets.risk_collection import write_risk_collection

    split_artifacts, shard_root, requests = _prepared_inputs(tmp_path)
    outputs = (tmp_path / "collection-a", tmp_path / "collection-b")
    write_risk_collection(
        requests,
        outputs[0],
        shard_root=shard_root,
        split_artifact_dir=split_artifacts,
        expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
        grid=_grid(),
    )
    write_risk_collection(
        tuple(reversed(requests)),
        outputs[1],
        shard_root=shard_root,
        split_artifact_dir=split_artifacts,
        expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
        grid=_grid(),
    )

    for name in ("members.jsonl", "leakage_report.json", "summary.json"):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()


def test_collection_supports_multiple_shards_per_split(tmp_path: Path):
    from src.datasets.risk_collection import (
        RiskCollectionMemberRequest,
        load_risk_collection,
        write_risk_collection,
    )

    split_artifacts, shard_root, requests = _prepared_inputs(tmp_path)
    relative_path = Path("train/shard-000001")
    extra_sample = _sample(
        "train",
        sample_id="sample-train-extra",
        snippet_id="snippet-train-extra",
        pair_group_id="pair-train-extra",
        seed_namespace="sop07/train/event-train-extra",
    )
    extra_path = shard_root / relative_path
    write_risk_shard(
        (extra_sample,),
        extra_path,
        grid=_grid(),
        shard_index=1,
        expected_sample_count=1,
    )
    extra = load_risk_shard(extra_path, grid=_grid())
    members = (
        *requests,
        RiskCollectionMemberRequest(
            relative_path=relative_path.as_posix(),
            split="train",
            shard_index=1,
            expected_sample_count=1,
            expected_manifest_digest=extra.manifest_digest,
            expected_semantic_digest=extra.semantic_digest,
        ),
    )
    output = tmp_path / "collection"
    write_risk_collection(
        members,
        output,
        shard_root=shard_root,
        split_artifact_dir=split_artifacts,
        expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
        grid=_grid(),
    )
    loaded = load_risk_collection(
        output,
        shard_root=shard_root,
        split_artifact_dir=split_artifacts,
        expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
        grid=_grid(),
    )

    assert loaded.summary["member_count"] == 5
    assert loaded.summary["sample_count"] == 5
    assert loaded.summary["split_counts"]["train"] == {
        "member_count": 2,
        "sample_count": 2,
    }


def test_collection_rejects_duplicate_sample_id_across_same_split_shards(
    tmp_path: Path,
):
    from src.datasets.risk_collection import (
        RiskCollectionError,
        RiskCollectionMemberRequest,
        write_risk_collection,
    )

    split_artifacts, shard_root, requests = _prepared_inputs(tmp_path)
    relative_path = Path("train/shard-000001")
    duplicate_path = shard_root / relative_path
    write_risk_shard(
        (_sample("train"),),
        duplicate_path,
        grid=_grid(),
        shard_index=1,
        expected_sample_count=1,
    )
    duplicate = load_risk_shard(duplicate_path, grid=_grid())
    members = (
        *requests,
        RiskCollectionMemberRequest(
            relative_path=relative_path.as_posix(),
            split="train",
            shard_index=1,
            expected_sample_count=1,
            expected_manifest_digest=duplicate.manifest_digest,
            expected_semantic_digest=duplicate.semantic_digest,
        ),
    )

    with pytest.raises(RiskCollectionError, match="duplicate sample_id"):
        write_risk_collection(
            members,
            tmp_path / "collection",
            shard_root=shard_root,
            split_artifact_dir=split_artifacts,
            expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
            grid=_grid(),
        )


def test_formal_collection_loader_rejects_unexpected_files(tmp_path: Path):
    from src.datasets.risk_collection import (
        RiskCollectionError,
        load_risk_collection,
        write_risk_collection,
    )

    split_artifacts, shard_root, requests = _prepared_inputs(tmp_path)
    output = tmp_path / "collection"
    write_risk_collection(
        requests,
        output,
        shard_root=shard_root,
        split_artifact_dir=split_artifacts,
        expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
        grid=_grid(),
    )
    (output / "partial.tmp").write_text("unsafe", encoding="utf-8")

    with pytest.raises(RiskCollectionError, match="file layout mismatch"):
        load_risk_collection(
            output,
            shard_root=shard_root,
            split_artifact_dir=split_artifacts,
            expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
            grid=_grid(),
        )


def test_collection_rejects_a_member_that_fails_formal_shard_load(tmp_path: Path):
    from src.datasets.risk_collection import RiskCollectionError, write_risk_collection

    split_artifacts, shard_root, requests = _prepared_inputs(tmp_path)
    broken = shard_root / requests[0].relative_path / "samples.npz"
    broken.unlink()

    with pytest.raises(RiskCollectionError, match="formal risk shard load failed"):
        write_risk_collection(
            requests,
            tmp_path / "collection",
            shard_root=shard_root,
            split_artifact_dir=split_artifacts,
            expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
            grid=_grid(),
        )
    assert not (tmp_path / "collection").exists()


def test_collection_cli_publishes_from_explicit_member_requests(tmp_path: Path):
    from src.datasets.risk_collection import load_risk_collection

    split_artifacts, shard_root, requests = _prepared_inputs(tmp_path)
    request_path, config_path = _write_cli_inputs(tmp_path, requests)
    output = tmp_path / "collection"
    script = Path(__file__).resolve().parents[1] / "scripts/05_audit_risk_collection.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--split-artifact-dir",
            str(split_artifacts),
            "--expected-split-manifest-digest",
            _split_manifest_digest(split_artifacts),
            "--shard-root",
            str(shard_root),
            "--member-requests",
            str(request_path),
            "--output-dir",
            str(output),
        ],
        cwd=script.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "ok"
    assert report["sample_count"] == 4
    loaded = load_risk_collection(
        output,
        shard_root=shard_root,
        split_artifact_dir=split_artifacts,
        expected_split_manifest_digest=_split_manifest_digest(split_artifacts),
        grid=_grid(),
    )
    assert report["collection_semantic_digest"] == (
        loaded.collection_semantic_digest
    )


def test_collection_cli_rejects_wrong_split_manifest_digest(tmp_path: Path):
    split_artifacts, shard_root, requests = _prepared_inputs(tmp_path)
    request_path, config_path = _write_cli_inputs(tmp_path, requests)
    output = tmp_path / "collection"
    script = Path(__file__).resolve().parents[1] / "scripts/05_audit_risk_collection.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--split-artifact-dir",
            str(split_artifacts),
            "--expected-split-manifest-digest",
            "0" * 32,
            "--shard-root",
            str(shard_root),
            "--member-requests",
            str(request_path),
            "--output-dir",
            str(output),
        ],
        cwd=script.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "expected split manifest digest mismatch" in completed.stderr
    assert not output.exists()
