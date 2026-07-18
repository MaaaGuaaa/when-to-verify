"""Behavioral tests for the formal SOP-07 risk-dataset CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

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
        expected_event_count=event_count,
        expected_sample_count=sample_count,
        checksum_workers=3,
    )


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
        return SimpleNamespace(
            event_id=kwargs["mother_record"].generated_event_id,
            variant_kind=kwargs["variant"].variant_kind,
            pair_group_id=f"pair-{kwargs['mother_record'].generated_event_id}",
        )

    def fake_build(source, *, base_config, risk_config):
        index = len(calls["samples"])
        collision = int(source.variant_kind == "collision")
        near_miss = int(source.variant_kind == "near_miss")
        sample = SimpleNamespace(
            sample_id=f"sample-{source.event_id}-{source.variant_kind}",
            collision_label=collision,
            near_miss=near_miss,
        )
        calls["samples"].append(sample)
        return sample

    def fake_write(samples, path, *, grid, expected_sample_count):
        calls["writer"].append((tuple(samples), path, grid, expected_sample_count))
        path.mkdir(parents=True)
        return {"directory": path}

    def fake_load(path, *, grid):
        return SimpleNamespace(
            samples=tuple(calls["samples"]),
            semantic_digest="8" * 64,
            manifest_digest="9" * 64,
        )

    monkeypatch.setattr(module, "generate_paired_variants", fake_generate)
    monkeypatch.setattr(
        module, "build_risk_input_from_sop06_variant", fake_adapter
    )
    monkeypatch.setattr(module, "build_risk_sample", fake_build)
    monkeypatch.setattr(module, "write_risk_shard", fake_write)
    monkeypatch.setattr(module, "load_risk_shard", fake_load)
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
    assert [call["variant"].variant_kind for call in calls["adapter"]] == [
        "collision",
        "near_miss",
        "empty_blind_spot",
        "collision",
        "near_miss",
        "empty_blind_spot",
    ]
    assert all(
        call["expected_paired_config_digest"] == "paired-digest"
        for call in calls["adapter"]
    )
    assert all(call["source_session_id"].startswith("session-") for call in calls["adapter"])
    assert all(
        call["seed_namespace"].startswith("sop07/train/seed-17/")
        for call in calls["adapter"]
    )
    assert calls["writer"][0][3] == 6
    assert report == {
        "schema_version": "3.0.0",
        "producer_version": cli_module.SOP07_RISK_DATASET_CLI_VERSION,
        "split": "train",
        "seed": 17,
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

    monkeypatch.setattr(cli_module, "build_risk_input_from_sop06_variant", crash)

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
        "--expected-event-count", "10",
        "--expected-sample-count", "30",
        "--checksum-workers", "3",
    ]
    index = required.index(option)
    required[index + 1] = value

    with pytest.raises(SystemExit):
        parser.parse_args(required)
