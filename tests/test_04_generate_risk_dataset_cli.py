"""Behavioral tests for the formal SOP-07 risk-dataset CLI."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

import src.datasets.risk_dataset as risk_dataset_module
from src.generation.paired_variants import PairGenerationError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "04_generate_risk_dataset.py"


@pytest.fixture
def cli_module():
    assert SCRIPT.is_file(), "the formal SOP-07 CLI does not exist"
    module_name = "_test_04_generate_risk_dataset"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def _evidence(*, policy: str) -> SimpleNamespace:
    return SimpleNamespace(
        code_commit="1" * 40,
        checksum_manifest_sha256="2" * 64,
        audit_sha256="3" * 64,
        completion_policy=policy,
    )


def _sop03_lock(evidence: SimpleNamespace) -> dict[str, object]:
    return {
        "code_commit": evidence.code_commit,
        "checksum_manifest_sha256": evidence.checksum_manifest_sha256,
        "audit_sha256": evidence.audit_sha256,
        "completion_policy": evidence.completion_policy,
    }


def _sop04_lock(bank: SimpleNamespace) -> dict[str, object]:
    evidence = bank.producer_evidence
    return {
        **_sop03_lock(evidence),
        "trajectory_bank_version": bank.trajectory_bank_version,
        "pose_time_layout_version": bank.pose_time_layout_version,
        "trajectory_steps": 15,
        "dt_s": 0.2,
        "first_pose_time_s": 0.2,
        "last_pose_time_s": 3.0,
        "pose_time_offsets_sha256": bank.pose_time_offsets_sha256,
        "bank_semantic_digest_sha256": bank.bank_semantic_digest_sha256,
        "external_handoff_digest_sha256": bank.external_handoff_digest_sha256,
    }


def _event(event_id: str, *, state_id: str, trajectory_id: str) -> SimpleNamespace:
    record = SimpleNamespace(
        generated_event_id=event_id,
        base_state_id=state_id,
        trajectory_id=trajectory_id,
        source_snippet_id=f"snippet-{event_id}",
        source_object_id=f"object-{event_id}",
        object_type="human",
        footprint_spec={
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": 0.3},
        },
    )
    return SimpleNamespace(
        generated_event_id=event_id,
        target_motion_record=record,
        world=SimpleNamespace(world_id=f"world-{event_id}"),
    )


def _snippet(event: SimpleNamespace, *, recording_id: str) -> SimpleNamespace:
    record = event.target_motion_record
    event.target = SimpleNamespace(
        provenance={"source_recording_id": recording_id}
    )
    return SimpleNamespace(
        snippet_id=record.source_snippet_id,
        source_object_id=record.source_object_id,
        source_recording_id=recording_id,
        source_session_id=f"session-{recording_id}",
        object_type=record.object_type,
    )


def _variant(kind: str, event_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        variant_kind=kind,
        world=SimpleNamespace(world_id=f"paired-{event_id}-{kind}"),
    )


def _group(event_id: str, kinds: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        pair_group_id=f"pair-{event_id}",
        variants=tuple(_variant(kind, event_id) for kind in kinds),
        coverage_mask=(True, True, False, False, False, True),
        missing_variant_reasons={
            "temporal_safe": "temporal_variant_still_collides",
            "spatial_safe": "spatial_clearance_unavailable",
            "irrelevant_hidden": "irrelevant_clearance_unavailable",
        },
        is_complete=False,
        eligible_for_strict_evaluation=False,
        paired_config_digest="paired-digest",
    )


def _request(module, tmp_path: Path, *, event_count: int, sample_count: int):
    return module.RiskDatasetRunRequest(
        sop03_root=tmp_path / "sop03",
        sop04_root=tmp_path / "sop04",
        sop04_handoff_digest="4" * 64,
        sop05_root=tmp_path / "sop05",
        sop05_publication_digest="5" * 64,
        split="train",
        config_path=tmp_path / "base.yaml",
        paired_config_path=tmp_path / "paired.yaml",
        seed=17,
        output_dir=tmp_path / "risk-shard",
        shard_index=7,
        expected_event_count=event_count,
        expected_sample_count=sample_count,
        checksum_workers=3,
    )


def test_cli_version_marks_explicit_shard_index_contract(cli_module) -> None:
    assert cli_module.SOP07_RISK_DATASET_CLI_VERSION == "sop07_risk_dataset_cli_v4"


def _install_success_dependencies(
    module,
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: tuple[SimpleNamespace, ...],
    output_dir: Path,
):
    base_config = {
        "schema_version": "3.0.0",
        "risk_gt": {
            "sigma_distance_m": 0.5,
            "sigma_time_s": 2.0,
            "near_miss_distance_m": 0.35,
        },
    }
    grid = SimpleNamespace(name="grid")
    sop03_evidence = _evidence(policy="sop03_complete_marker_v1")
    sop04_evidence = _evidence(policy="sop04_audited_bank_v2")
    trajectories = {
        event.target_motion_record.trajectory_id: SimpleNamespace(
            trajectory_id=event.target_motion_record.trajectory_id
        )
        for event in events
    }
    snippets = tuple(
        _snippet(event, recording_id=f"recording-{index % 2}")
        for index, event in enumerate(events)
    )

    class FakeSop03:
        split = "train"
        producer_evidence = sop03_evidence
        manifest_index = {
            event.target_motion_record.base_state_id: SimpleNamespace()
            for event in events
        }
        typed_libraries = {
            "human": SimpleNamespace(snippets=tuple(reversed(snippets)))
        }

        def load_pair(self, state_id, loaded_grid):
            assert loaded_grid is grid
            return (
                SimpleNamespace(state_id=state_id, split="train"),
                SimpleNamespace(base_state_id=state_id),
            )

    sop03 = FakeSop03()
    sop04 = SimpleNamespace(
        producer_evidence=sop04_evidence,
        trajectory_bank_version="sop04_audited_bank_v2",
        pose_time_layout_version="future_endpoints_dt_to_horizon_v1",
        pose_time_offsets_sha256="6" * 64,
        bank_semantic_digest_sha256="7" * 64,
        external_handoff_digest_sha256="4" * 64,
        by_id=trajectories,
    )
    run_manifest = {
        "input_lock": {
            "version": "sop05_input_lock_v2",
            "split": "train",
            "sop03": _sop03_lock(sop03_evidence),
            "sop04": _sop04_lock(sop04),
            "selection": {"pair_count": len(events)},
        }
    }
    loaded_sop05 = SimpleNamespace(
        split="train",
        events=events,
        run_manifest=run_manifest,
    )
    paired_config = SimpleNamespace(digest="paired-digest")

    monkeypatch.setattr(module, "load_config", lambda path: base_config)
    monkeypatch.setattr(module, "build_grid_spec", lambda config: grid)
    monkeypatch.setattr(
        module, "load_paired_variant_config", lambda path: paired_config
    )
    monkeypatch.setattr(
        module,
        "load_sop03_split_inputs",
        lambda root, split, loaded_grid, checksum_workers: sop03,
    )
    monkeypatch.setattr(
        module,
        "load_sop04_trajectory_bank",
        lambda root, loaded_grid, expected_external_handoff_digest_sha256,
        checksum_workers: sop04,
    )
    monkeypatch.setattr(
        module,
        "load_complete_sop05_events",
        lambda root, grid, expected_publication_semantic_digest: loaded_sop05,
    )
    monkeypatch.setattr(module, "derive_seed", lambda seed, *parts: seed + 100)

    calls: dict[str, list] = {
        "pair_events": [],
        "adapter": [],
        "samples": [],
        "writer": [],
        "risk_load": [],
        "pinned_risk_open": [],
        "pinned_risk_verify": [],
        "pinned_risk_close": [],
    }

    def fake_generate(**kwargs):
        event = kwargs["mother_event"]
        calls["pair_events"].append(event.generated_event_id)
        assert kwargs["source_snippet"].snippet_id == (
            event.target_motion_record.source_snippet_id
        )
        return _group(
            event.generated_event_id,
            ("collision", "near_miss", "empty_blind_spot"),
        )

    def fake_adapter(**kwargs):
        calls["adapter"].append(kwargs)
        event = kwargs["mother_event"]
        built = tuple(
            SimpleNamespace(
                sample_id=(
                    f"sample-{event.generated_event_id}-{variant.variant_kind}"
                ),
                collision_label=int(variant.variant_kind == "collision"),
                near_miss=int(variant.variant_kind == "near_miss"),
            )
            for variant in kwargs["group"].variants
        )
        calls["samples"].extend(built)
        return built

    def fake_write(
        samples, path, *, grid, shard_index, expected_sample_count
    ):
        calls["writer"].append(
            (tuple(samples), path, grid, expected_sample_count, shard_index)
        )
        path.mkdir(parents=True)
        return {"directory": path}

    def fake_load(path, *, grid):
        calls["risk_load"].append((path, grid))
        return SimpleNamespace(
            samples=tuple(calls["samples"]),
            semantic_digest="8" * 64,
            manifest_digest="9" * 64,
            summary={"split": "train", "shard_index": 7},
        )

    class FakePinnedRiskShard:
        def __init__(self, path, *, grid):
            self.path = path
            self.grid = grid
            self.loaded_shard = None

        def __enter__(self):
            calls["pinned_risk_open"].append((self.path, self.grid))
            self.loaded_shard = fake_load(self.path, grid=self.grid)
            return self

        def verify_unchanged(self):
            calls["pinned_risk_verify"].append(self.path)

        def __exit__(self, exc_type, exc, traceback):
            calls["pinned_risk_close"].append(self.path)
            return False

    def fake_pin(path, *, grid):
        return FakePinnedRiskShard(path, grid=grid)

    monkeypatch.setattr(module, "generate_paired_variants", fake_generate)
    monkeypatch.setattr(
        module, "build_risk_samples_from_sop06_group", fake_adapter
    )
    monkeypatch.setattr(module, "write_risk_shard", fake_write)
    monkeypatch.setattr(module, "load_risk_shard", fake_load)
    monkeypatch.setattr(
        module, "pin_risk_shard_snapshot", fake_pin
    )
    monkeypatch.setattr(
        module,
        "summarize_paired_groups",
        lambda groups: {
            "group_count": len(tuple(groups)),
            "coverage_counts": {
                "collision": len(events),
                "near_miss": len(events),
                "empty_blind_spot": len(events),
            },
        },
    )
    return calls, loaded_sop05, sop03, sop04


def _install_success_sidecar_dependencies(
    module,
    monkeypatch: pytest.MonkeyPatch,
    calls: dict[str, list],
):
    sidecar_write_calls: list[tuple[object, ...]] = []
    sidecar_load_calls: list[tuple[object, ...]] = []

    def fake_combined_adapter(**kwargs):
        samples = module.build_risk_samples_from_sop06_group(**kwargs)
        sidecars = tuple(
            SimpleNamespace(sample_id=sample.sample_id) for sample in samples
        )
        return samples, sidecars

    def fake_sidecar_write(
        sidecars,
        path,
        *,
        grid,
        split,
        shard_index,
        source_risk_shard_semantic_digest,
    ):
        sidecar_write_calls.append(
            (
                tuple(sidecars),
                path,
                grid,
                split,
                shard_index,
                source_risk_shard_semantic_digest,
            )
        )
        path.mkdir(parents=True)
        return {"directory": path}

    def fake_sidecar_load(
        path,
        *,
        grid,
        expected_sample_ids,
        expected_source_risk_shard_semantic_digest,
    ):
        sidecar_load_calls.append(
            (
                path,
                grid,
                tuple(expected_sample_ids),
                expected_source_risk_shard_semantic_digest,
            )
        )
        assert tuple(expected_sample_ids) == tuple(
            sample.sample_id for sample in calls["samples"]
        )
        assert expected_source_risk_shard_semantic_digest == "8" * 64
        return SimpleNamespace(
            sample_ids=tuple(expected_sample_ids),
            semantic_digest="a" * 64,
            split="train",
            shard_index=7,
        )

    monkeypatch.setattr(
        module,
        "build_risk_samples_and_sidecars_from_sop06_group",
        fake_combined_adapter,
    )
    monkeypatch.setattr(module, "write_risk_sidecar_shard", fake_sidecar_write)
    monkeypatch.setattr(module, "load_risk_sidecar_shard", fake_sidecar_load)
    return sidecar_write_calls, sidecar_load_calls


def _pair_marker_path(module, request) -> Path:
    assert request.sidecar_output_dir is not None
    return module.risk_sidecar_pair_completion_marker_path(
        request.sidecar_output_dir
    )


def _assert_pair_staging_clean(request) -> None:
    assert request.sidecar_output_dir is not None
    for parent in {
        request.output_dir.parent,
        request.sidecar_output_dir.parent,
    }:
        if parent.exists():
            assert not tuple(parent.glob(".*.pair-staging-*"))


def _assert_no_published_pair(module, request) -> None:
    assert request.sidecar_output_dir is not None
    assert not request.output_dir.exists()
    assert not request.sidecar_output_dir.exists()
    assert not _pair_marker_path(module, request).exists()
    _assert_pair_staging_clean(request)


def test_run_stably_rebuilds_partial_groups_and_writes_verified_shard(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-b", state_id="base-b", trajectory_id="trajectory-b"),
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = _request(cli_module, tmp_path, event_count=2, sample_count=6)
    calls, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )

    report = cli_module.run_risk_dataset(request)

    assert calls["pair_events"] == ["event-a", "event-b"]
    assert [call["group"].pair_group_id for call in calls["adapter"]] == [
        "pair-event-a",
        "pair-event-b",
    ]
    assert all(
        call["paired_config"].digest == "paired-digest"
        for call in calls["adapter"]
    )
    assert all(
        call["source_snippet"].source_session_id.startswith("session-")
        for call in calls["adapter"]
    )
    assert all(call["dataset_seed"] == 17 for call in calls["adapter"])
    assert calls["writer"][0][3] == 6
    assert calls["writer"][0][4] == 7
    assert report == {
        "schema_version": "3.0.0",
        "producer_version": cli_module.SOP07_RISK_DATASET_CLI_VERSION,
        "split": "train",
        "seed": 17,
        "shard_index": 7,
        "output_dir": str(request.output_dir),
        "event_count": 2,
        "sample_count": 6,
        "rejection_report": {
            "attempted_event_count": 2,
            "accepted_group_count": 2,
            "rejected_event_count": 0,
            "reason_counts": {},
        },
        "class_prior": {
            "collision": {"count": 2, "rate": pytest.approx(2 / 6)},
            "near_miss": {"count": 2, "rate": pytest.approx(2 / 6)},
            "safe": {"count": 2, "rate": pytest.approx(2 / 6)},
        },
        "pair_coverage": {
            "group_count": 2,
            "coverage_counts": {
                "collision": 2,
                "near_miss": 2,
                "empty_blind_spot": 2,
            },
        },
        "source_coverage": {
            "accepted_event_count": 2,
            "unique_source_recording_count": 2,
            "unique_source_session_count": 2,
            "unique_source_snippet_count": 2,
            "object_type_counts": {"human": 2},
            "footprint_kind_counts": {"circle": 2},
        },
        "manifest_digest": "9" * 64,
        "semantic_digest": "8" * 64,
    }


def test_optional_sidecar_root_publishes_separate_id_bound_shard(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = replace(
        _request(cli_module, tmp_path, event_count=1, sample_count=3),
        output_dir=tmp_path / "risk-parent" / "risk-shard",
        sidecar_output_dir=tmp_path / "sidecar-parent" / "sidecar-shard",
    )
    calls, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )
    sidecar_calls, sidecar_load_calls = _install_success_sidecar_dependencies(
        cli_module,
        monkeypatch,
        calls,
    )

    report = cli_module.run_risk_dataset(request)

    assert len(sidecar_calls) == 1
    assert len(calls["risk_load"]) == 3
    assert len(sidecar_load_calls) == 3
    risk_staging_root = calls["writer"][0][1]
    sidecar_staging_root = sidecar_calls[0][1]
    assert risk_staging_root != request.output_dir
    assert sidecar_staging_root != request.sidecar_output_dir
    assert risk_staging_root.parent.parent == request.output_dir.parent
    assert sidecar_staging_root.parent.parent == request.sidecar_output_dir.parent
    assert calls["risk_load"][0][0] == risk_staging_root
    assert calls["risk_load"][1][0] == request.output_dir
    assert sidecar_load_calls[0][0] == sidecar_staging_root
    assert sidecar_load_calls[1][0] == request.sidecar_output_dir
    assert sidecar_calls[0][3:] == ("train", 7, "8" * 64)
    assert report["publication_status"] == "complete"
    assert report["sidecar_output_dir"] == str(request.sidecar_output_dir)
    assert report["risk_shard_semantic_digest"] == "8" * 64
    assert report["sidecar_shard_semantic_digest"] == "a" * 64
    marker_path = _pair_marker_path(cli_module, request)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker_path.parent == request.sidecar_output_dir.parent
    assert marker["risk_root_basename"] == request.output_dir.name
    assert marker["sidecar_root_basename"] == request.sidecar_output_dir.name
    assert marker["split"] == "train"
    assert marker["shard_index"] == 7
    assert marker["risk_shard_semantic_digest"] == "8" * 64
    assert marker["sidecar_shard_semantic_digest"] == "a" * 64
    assert len(marker["ordered_sample_ids_digest_sha256"]) == 64
    assert report["pair_completion_marker_path"] == str(marker_path)
    assert report["pair_completion_marker_digest"] == marker[
        "marker_digest_sha256"
    ]
    _assert_pair_staging_clean(request)


def test_complete_pair_uses_pinned_risk_snapshot_guard(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = replace(
        _request(cli_module, tmp_path, event_count=1, sample_count=3),
        sidecar_output_dir=tmp_path / "sidecar-shard",
    )
    calls, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )
    _install_success_sidecar_dependencies(cli_module, monkeypatch, calls)

    report = cli_module.run_risk_dataset(request)

    assert report["publication_status"] == "complete"
    assert len(calls["pinned_risk_open"]) == 1
    assert calls["pinned_risk_open"][0][0] == request.output_dir
    assert calls["pinned_risk_verify"] == [request.output_dir]
    assert calls["pinned_risk_close"] == [request.output_dir]
    assert all(
        not str(path).startswith("/proc/self/fd/")
        for path, _ in calls["risk_load"]
    )


@pytest.mark.parametrize("mutation_stage", ("sidecar", "marker"))
@pytest.mark.parametrize(
    ("mutation_kind", "error_pattern"),
    (
        ("same-inode-content", "content changed"),
        ("replace-member", "identity changed"),
        ("add-member", "membership changed"),
        ("delete-member", "membership changed"),
    ),
)
def test_complete_pair_pins_risk_members_through_downstream_gate(
    cli_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation_stage: str,
    mutation_kind: str,
    error_pattern: str,
) -> None:
    risk_root = tmp_path / "risk-shard"
    sidecar_root = tmp_path / "sidecar-shard"
    risk_root.mkdir()
    sidecar_root.mkdir()
    original_members = {
        "samples.npz": b"payload-original",
        "metadata.jsonl": b"manifest-original\n",
        "summary.json": b"summary-original\n",
    }
    for name, payload in original_members.items():
        (risk_root / name).write_bytes(payload)
    marker_path = cli_module.risk_sidecar_pair_completion_marker_path(
        sidecar_root
    )
    marker_path.write_bytes(b"marker-original\n")
    request = replace(
        _request(cli_module, tmp_path, event_count=1, sample_count=1),
        output_dir=risk_root,
        sidecar_output_dir=sidecar_root,
    )
    risk_identity = cli_module._capture_owned_path(
        risk_root, expected_file_type=cli_module.stat.S_IFDIR
    )
    sidecar_identity = cli_module._capture_owned_path(
        sidecar_root, expected_file_type=cli_module.stat.S_IFDIR
    )
    marker_identity = cli_module._capture_owned_path(
        marker_path, expected_file_type=cli_module.stat.S_IFREG
    )
    root_inode = os.lstat(risk_root).st_ino
    risk_shard = SimpleNamespace(
        samples=(SimpleNamespace(sample_id="sample-a"),),
        semantic_digest="8" * 64,
        summary={"split": "train", "shard_index": 7},
    )
    mutation_count = 0

    def mutate_risk_root() -> None:
        nonlocal mutation_count
        mutation_count += 1
        if mutation_kind == "same-inode-content":
            member = risk_root / "summary.json"
            inode = os.lstat(member).st_ino
            with member.open("r+b") as handle:
                handle.seek(0)
                handle.write(b"BORK")
                handle.flush()
                os.fsync(handle.fileno())
            assert os.lstat(member).st_ino == inode
        elif mutation_kind == "replace-member":
            member = risk_root / "metadata.jsonl"
            member.rename(tmp_path / "displaced-metadata.jsonl")
            member.write_bytes(b"replacement-manifest\n")
        elif mutation_kind == "add-member":
            (risk_root / "unexpected-member").write_bytes(b"unexpected\n")
        else:
            assert mutation_kind == "delete-member"
            (risk_root / "samples.npz").unlink()
        assert os.lstat(risk_root).st_ino == root_inode

    def fake_risk_load(path, *, grid):
        return risk_shard

    def fake_sidecar_load(path, **kwargs):
        if mutation_stage == "sidecar":
            mutate_risk_root()
        return SimpleNamespace(
            sample_ids=("sample-a",),
            semantic_digest="a" * 64,
            split="train",
            shard_index=7,
        )

    def fake_marker_load(path, **kwargs):
        if mutation_stage == "marker":
            mutate_risk_root()
        return SimpleNamespace(marker_digest_sha256="b" * 64)

    monkeypatch.setattr(
        risk_dataset_module,
        "_load_risk_shard_from_snapshot_directory",
        fake_risk_load,
    )
    monkeypatch.setattr(
        cli_module, "load_risk_sidecar_shard", fake_sidecar_load
    )
    monkeypatch.setattr(
        cli_module,
        "load_risk_sidecar_pair_completion_marker",
        fake_marker_load,
    )

    with pytest.raises(ValueError, match=error_pattern):
        cli_module._load_complete_risk_sidecar_pair(
            request=request,
            grid=SimpleNamespace(name="grid"),
            risk_identity=risk_identity,
            sidecar_identity=sidecar_identity,
            marker_identity=marker_identity,
        )

    assert mutation_count == 1
    assert os.lstat(risk_root).st_ino == root_inode


@pytest.mark.parametrize("nested_direction", ["sidecar-inside-risk", "risk-inside-sidecar"])
def test_nested_risk_and_sidecar_roots_are_rejected_before_any_write(
    cli_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    nested_direction: str,
) -> None:
    request = _request(cli_module, tmp_path, event_count=1, sample_count=3)
    if nested_direction == "sidecar-inside-risk":
        request = replace(
            request,
            sidecar_output_dir=request.output_dir / "sidecars",
        )
    else:
        parent = tmp_path / "combined-root"
        request = replace(
            request,
            output_dir=parent / "risk-shard",
            sidecar_output_dir=parent,
        )
    monkeypatch.setattr(
        cli_module,
        "_load_inputs",
        lambda value: pytest.fail("nested roots must fail before loading inputs"),
    )

    with pytest.raises(
        cli_module.RiskDatasetRunError, match="must not be nested"
    ):
        cli_module.run_risk_dataset(request)

    assert not request.output_dir.exists()
    assert request.sidecar_output_dir is not None
    assert not request.sidecar_output_dir.exists()


def test_sidecar_write_failure_cleans_transaction_and_is_retryable(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = replace(
        _request(cli_module, tmp_path, event_count=1, sample_count=3),
        sidecar_output_dir=tmp_path / "sidecar-shard",
    )
    calls, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )
    _, _ = _install_success_sidecar_dependencies(
        cli_module, monkeypatch, calls
    )
    successful_write = cli_module.write_risk_sidecar_shard
    fail_once = True

    def flaky_sidecar_write(*args, **kwargs):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise ValueError("forced sidecar write failure")
        return successful_write(*args, **kwargs)

    monkeypatch.setattr(
        cli_module, "write_risk_sidecar_shard", flaky_sidecar_write
    )

    with pytest.raises(
        cli_module.RiskDatasetRunError, match="publication failed"
    ):
        cli_module.run_risk_dataset(request)

    _assert_no_published_pair(cli_module, request)
    calls["samples"].clear()

    report = cli_module.run_risk_dataset(request)

    assert report["publication_status"] == "complete"
    assert request.output_dir.is_dir()
    assert request.sidecar_output_dir is not None
    assert request.sidecar_output_dir.is_dir()
    assert _pair_marker_path(cli_module, request).is_file()


def test_sidecar_staging_reload_failure_cleans_transaction_and_is_retryable(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = replace(
        _request(cli_module, tmp_path, event_count=1, sample_count=3),
        sidecar_output_dir=tmp_path / "sidecar-shard",
    )
    calls, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )
    _, _ = _install_success_sidecar_dependencies(
        cli_module, monkeypatch, calls
    )
    successful_load = cli_module.load_risk_sidecar_shard
    fail_once = True

    def flaky_sidecar_load(*args, **kwargs):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise ValueError("forced sidecar staging reload failure")
        return successful_load(*args, **kwargs)

    monkeypatch.setattr(
        cli_module, "load_risk_sidecar_shard", flaky_sidecar_load
    )

    with pytest.raises(
        cli_module.RiskDatasetRunError, match="publication failed"
    ):
        cli_module.run_risk_dataset(request)

    _assert_no_published_pair(cli_module, request)
    calls["samples"].clear()

    report = cli_module.run_risk_dataset(request)

    assert report["publication_status"] == "complete"


def test_final_reload_failure_removes_only_this_invocations_pair(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = replace(
        _request(cli_module, tmp_path, event_count=1, sample_count=3),
        sidecar_output_dir=tmp_path / "sidecar-shard",
    )
    calls, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )
    _, _ = _install_success_sidecar_dependencies(
        cli_module, monkeypatch, calls
    )
    successful_load = cli_module.load_risk_shard
    load_count = 0

    def fail_final_risk_reload(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        if load_count == 2:
            raise ValueError("forced final risk reload failure")
        return successful_load(*args, **kwargs)

    monkeypatch.setattr(cli_module, "load_risk_shard", fail_final_risk_reload)

    with pytest.raises(
        cli_module.RiskDatasetRunError, match="publication failed"
    ):
        cli_module.run_risk_dataset(request)

    assert load_count == 2
    _assert_no_published_pair(cli_module, request)


def test_second_directory_commit_race_preserves_competitor_and_rolls_back_first(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = replace(
        _request(cli_module, tmp_path, event_count=1, sample_count=3),
        sidecar_output_dir=tmp_path / "sidecar-shard",
    )
    calls, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )
    _, _ = _install_success_sidecar_dependencies(
        cli_module, monkeypatch, calls
    )
    real_commit = cli_module._atomic_pair_commit_noreplace
    commit_count = 0
    competitor_inode: int | None = None

    def race_second_commit(source: Path, destination: Path) -> None:
        nonlocal commit_count, competitor_inode
        commit_count += 1
        if commit_count == 2:
            destination.mkdir()
            competitor_inode = os.lstat(destination).st_ino
        real_commit(source, destination)

    monkeypatch.setattr(
        cli_module, "_atomic_pair_commit_noreplace", race_second_commit
    )

    with pytest.raises(
        cli_module.RiskDatasetRunError, match="publication failed"
    ):
        cli_module.run_risk_dataset(request)

    assert commit_count == 2
    assert not request.output_dir.exists()
    assert request.sidecar_output_dir is not None
    assert request.sidecar_output_dir.is_dir()
    assert os.lstat(request.sidecar_output_dir).st_ino == competitor_inode
    assert not tuple(request.sidecar_output_dir.iterdir())
    assert not _pair_marker_path(cli_module, request).exists()
    _assert_pair_staging_clean(request)


def test_marker_commit_race_preserves_competitor_and_removes_owned_pair(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = replace(
        _request(cli_module, tmp_path, event_count=1, sample_count=3),
        sidecar_output_dir=tmp_path / "sidecar-shard",
    )
    calls, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )
    _, _ = _install_success_sidecar_dependencies(
        cli_module, monkeypatch, calls
    )
    real_commit = cli_module._atomic_pair_commit_noreplace
    commit_count = 0
    competitor_inode: int | None = None

    def race_marker_commit(source: Path, destination: Path) -> None:
        nonlocal commit_count, competitor_inode
        commit_count += 1
        if commit_count == 3:
            destination.write_text("competitor\n", encoding="utf-8")
            competitor_inode = os.lstat(destination).st_ino
        real_commit(source, destination)

    monkeypatch.setattr(
        cli_module, "_atomic_pair_commit_noreplace", race_marker_commit
    )

    with pytest.raises(
        cli_module.RiskDatasetRunError, match="publication failed"
    ):
        cli_module.run_risk_dataset(request)

    marker_path = _pair_marker_path(cli_module, request)
    assert commit_count == 3
    assert not request.output_dir.exists()
    assert request.sidecar_output_dir is not None
    assert not request.sidecar_output_dir.exists()
    assert marker_path.read_text(encoding="utf-8") == "competitor\n"
    assert os.lstat(marker_path).st_ino == competitor_inode
    _assert_pair_staging_clean(request)


def test_final_marker_reload_failure_removes_owned_marker_and_pair(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = replace(
        _request(cli_module, tmp_path, event_count=1, sample_count=3),
        sidecar_output_dir=tmp_path / "sidecar-shard",
    )
    calls, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )
    _, _ = _install_success_sidecar_dependencies(
        cli_module, monkeypatch, calls
    )
    successful_load = cli_module.load_risk_sidecar_pair_completion_marker
    load_count = 0

    def fail_final_marker_reload(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        if load_count == 2:
            raise ValueError("forced final marker reload failure")
        return successful_load(*args, **kwargs)

    monkeypatch.setattr(
        cli_module,
        "load_risk_sidecar_pair_completion_marker",
        fail_final_marker_reload,
    )

    with pytest.raises(
        cli_module.RiskDatasetRunError, match="publication failed"
    ):
        cli_module.run_risk_dataset(request)

    assert load_count == 2
    _assert_no_published_pair(cli_module, request)


def test_cleanup_claim_restores_competitor_when_owned_path_is_replaced(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    owned_path = tmp_path / "owned-directory"
    displaced_owned = tmp_path / "displaced-owned-directory"
    owned_path.mkdir()
    owned = cli_module._capture_owned_path(
        owned_path, expected_file_type=cli_module.stat.S_IFDIR
    )
    competitor_inode: int | None = None
    real_claim = cli_module._atomic_pair_cleanup_claim_noreplace

    def swap_before_claim(source: Path, destination: Path) -> None:
        nonlocal competitor_inode
        source.rename(displaced_owned)
        source.mkdir()
        competitor_inode = os.lstat(source).st_ino
        real_claim(source, destination)

    monkeypatch.setattr(
        cli_module,
        "_atomic_pair_cleanup_claim_noreplace",
        swap_before_claim,
    )

    assert cli_module._remove_owned_path(owned) is False

    assert displaced_owned.is_dir()
    assert owned_path.is_dir()
    assert os.lstat(owned_path).st_ino == competitor_inode
    assert not tuple(tmp_path.glob(".*.cleanup-quarantine-*"))


def test_final_root_replacement_before_marker_commit_prevents_complete_report(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = replace(
        _request(cli_module, tmp_path, event_count=1, sample_count=3),
        sidecar_output_dir=tmp_path / "sidecar-shard",
    )
    calls, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )
    _install_success_sidecar_dependencies(cli_module, monkeypatch, calls)
    real_commit = cli_module._atomic_pair_commit_noreplace
    displaced_risk = tmp_path / "displaced-owned-risk"
    competitor_inode: int | None = None
    commit_count = 0

    def replace_risk_before_marker(source: Path, destination: Path) -> None:
        nonlocal commit_count, competitor_inode
        commit_count += 1
        if commit_count == 3:
            request.output_dir.rename(displaced_risk)
            request.output_dir.mkdir()
            competitor_inode = os.lstat(request.output_dir).st_ino
        real_commit(source, destination)

    monkeypatch.setattr(
        cli_module,
        "_atomic_pair_commit_noreplace",
        replace_risk_before_marker,
    )

    with pytest.raises(
        cli_module.RiskDatasetRunError, match="cleanup incomplete"
    ):
        cli_module.run_risk_dataset(request)

    assert commit_count == 3
    assert displaced_risk.is_dir()
    assert request.output_dir.is_dir()
    assert os.lstat(request.output_dir).st_ino == competitor_inode
    assert request.sidecar_output_dir is not None
    assert not request.sidecar_output_dir.exists()
    assert not _pair_marker_path(cli_module, request).exists()
    _assert_pair_staging_clean(request)


def test_final_root_replacement_after_marker_commit_prevents_complete_report(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = replace(
        _request(cli_module, tmp_path, event_count=1, sample_count=3),
        sidecar_output_dir=tmp_path / "sidecar-shard",
    )
    calls, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )
    _install_success_sidecar_dependencies(cli_module, monkeypatch, calls)
    real_complete_load = cli_module._load_complete_risk_sidecar_pair
    displaced_sidecar = tmp_path / "displaced-owned-sidecar"
    competitor_inode: int | None = None

    def replace_sidecar_then_load(*args, **kwargs):
        nonlocal competitor_inode
        assert request.sidecar_output_dir is not None
        request.sidecar_output_dir.rename(displaced_sidecar)
        request.sidecar_output_dir.mkdir()
        competitor_inode = os.lstat(request.sidecar_output_dir).st_ino
        return real_complete_load(*args, **kwargs)

    monkeypatch.setattr(
        cli_module,
        "_load_complete_risk_sidecar_pair",
        replace_sidecar_then_load,
    )

    with pytest.raises(
        cli_module.RiskDatasetRunError, match="cleanup incomplete"
    ):
        cli_module.run_risk_dataset(request)

    assert displaced_sidecar.is_dir()
    assert request.sidecar_output_dir is not None
    assert request.sidecar_output_dir.is_dir()
    assert os.lstat(request.sidecar_output_dir).st_ino == competitor_inode
    assert not request.output_dir.exists()
    assert not _pair_marker_path(cli_module, request).exists()
    _assert_pair_staging_clean(request)


def test_base_exception_still_cleans_all_pair_staging(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = replace(
        _request(cli_module, tmp_path, event_count=1, sample_count=3),
        sidecar_output_dir=tmp_path / "sidecar-shard",
    )
    calls, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )
    _install_success_sidecar_dependencies(cli_module, monkeypatch, calls)

    def interrupt_sidecar_write(*args, **kwargs):
        raise KeyboardInterrupt("forced interrupt")

    monkeypatch.setattr(
        cli_module, "write_risk_sidecar_shard", interrupt_sidecar_write
    )

    with pytest.raises(KeyboardInterrupt, match="forced interrupt"):
        cli_module.run_risk_dataset(request)

    _assert_no_published_pair(cli_module, request)


def test_run_rejects_sop05_input_lock_that_differs_from_reloaded_sop03(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = _request(cli_module, tmp_path, event_count=1, sample_count=3)
    _, loaded_sop05, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )
    loaded_sop05.run_manifest["input_lock"]["sop03"]["audit_sha256"] = "f" * 64

    with pytest.raises(cli_module.RiskDatasetRunError, match="SOP03 evidence"):
        cli_module.run_risk_dataset(request)

    assert not request.output_dir.exists()


def test_run_rejects_source_recording_mismatch_between_event_and_snippet(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = _request(cli_module, tmp_path, event_count=1, sample_count=3)
    _, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )
    events[0].target.provenance["source_recording_id"] = "wrong-recording"

    with pytest.raises(cli_module.RiskDatasetRunError, match="source recording"):
        cli_module.run_risk_dataset(request)

    assert not request.output_dir.exists()


def test_run_requires_exact_event_count_before_pair_generation(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = _request(cli_module, tmp_path, event_count=2, sample_count=3)
    calls, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )

    with pytest.raises(cli_module.RiskDatasetRunError, match="expected_event_count"):
        cli_module.run_risk_dataset(request)

    assert calls["pair_events"] == []
    assert not request.output_dir.exists()


def test_pair_rejections_are_reported_but_exact_sample_count_still_gates_publish(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
        _event("event-b", state_id="base-b", trajectory_id="trajectory-b"),
    )
    request = _request(cli_module, tmp_path, event_count=2, sample_count=6)
    calls, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )
    original_generate = cli_module.generate_paired_variants

    def rejecting_generate(**kwargs):
        if kwargs["mother_event"].generated_event_id == "event-b":
            raise PairGenerationError("collision_mother_invalid")
        return original_generate(**kwargs)

    monkeypatch.setattr(cli_module, "generate_paired_variants", rejecting_generate)

    with pytest.raises(
        cli_module.RiskDatasetRunError, match="expected_sample_count"
    ) as caught:
        cli_module.run_risk_dataset(request)

    assert "collision_mother_invalid" in str(caught.value)
    assert "expected=6" in str(caught.value)
    assert "actual=3" in str(caught.value)
    assert calls["writer"] == []
    assert not request.output_dir.exists()


def test_unexpected_adapter_exception_is_not_swallowed(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = (
        _event("event-a", state_id="base-a", trajectory_id="trajectory-a"),
    )
    request = _request(cli_module, tmp_path, event_count=1, sample_count=3)
    calls, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )

    def crash(**kwargs):
        raise RuntimeError("adapter programming defect")

    monkeypatch.setattr(cli_module, "build_risk_samples_from_sop06_group", crash)

    with pytest.raises(RuntimeError, match="programming defect"):
        cli_module.run_risk_dataset(request)

    assert calls["writer"] == []
    assert not request.output_dir.exists()


def test_ten_event_smoke_keeps_every_partial_variant(
    cli_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = tuple(
        _event(
            f"event-{index:02d}",
            state_id=f"base-{index:02d}",
            trajectory_id=f"trajectory-{index:02d}",
        )
        for index in reversed(range(10))
    )
    request = _request(cli_module, tmp_path, event_count=10, sample_count=30)
    _, _, _, _ = _install_success_dependencies(
        cli_module,
        monkeypatch,
        events=events,
        output_dir=request.output_dir,
    )

    report = cli_module.run_risk_dataset(request)

    assert report["event_count"] == 10
    assert report["sample_count"] == 30
    assert report["rejection_report"]["accepted_group_count"] == 10


def test_main_prints_one_canonical_json_line(
    cli_module,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    report = {
        "schema_version": "3.0.0",
        "producer_version": cli_module.SOP07_RISK_DATASET_CLI_VERSION,
        "semantic_digest": "8" * 64,
    }
    monkeypatch.setattr(cli_module, "run_risk_dataset", lambda request: report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--sop03-root", str(tmp_path / "sop03"),
            "--sop04-root", str(tmp_path / "sop04"),
            "--sop04-handoff-digest", "4" * 64,
            "--sop05-root", str(tmp_path / "sop05"),
            "--sop05-publication-digest", "5" * 64,
            "--split", "train",
            "--config", str(tmp_path / "base.yaml"),
            "--paired-config", str(tmp_path / "paired.yaml"),
            "--seed", "17",
            "--output-dir", str(tmp_path / "risk"),
            "--shard-index", "7",
            "--expected-event-count", "10",
            "--expected-sample-count", "30",
            "--checksum-workers", "3",
        ],
    )

    assert cli_module.main() == 0

    captured = capsys.readouterr()
    expected = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert captured.out == expected + "\n"
    assert captured.err == ""


@pytest.mark.parametrize(
    "option,value",
    [
        ("--seed", "-1"),
        ("--shard-index", "-1"),
        ("--expected-event-count", "0"),
        ("--expected-sample-count", "0"),
        ("--checksum-workers", "0"),
    ],
)
def test_parser_rejects_invalid_integer_contracts(
    cli_module, option: str, value: str
) -> None:
    parser = cli_module._parser()
    required = [
        "--sop03-root", "sop03",
        "--sop04-root", "sop04",
        "--sop04-handoff-digest", "4" * 64,
        "--sop05-root", "sop05",
        "--sop05-publication-digest", "5" * 64,
        "--split", "train",
        "--config", "base.yaml",
        "--paired-config", "paired.yaml",
        "--seed", "17",
        "--output-dir", "risk",
        "--shard-index", "7",
        "--expected-event-count", "10",
        "--expected-sample-count", "30",
        "--checksum-workers", "3",
    ]
    index = required.index(option)
    required[index + 1] = value

    with pytest.raises(SystemExit):
        parser.parse_args(required)
