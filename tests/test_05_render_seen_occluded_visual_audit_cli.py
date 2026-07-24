"""CLI contracts for the real seen-then-occluded visual audit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.seen_occluded_visual_audit import SeenOccludedAuditError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/05_render_seen_occluded_visual_audit.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("seen_occluded_audit_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--sop03-root",
        str(tmp_path / "sop03"),
        "--sop04-root",
        str(tmp_path / "sop04"),
        "--sop04-handoff-digest",
        "a" * 64,
        "--split",
        "train",
        "--base-config",
        str(ROOT / "configs/base.yaml"),
        "--generator-config",
        str(ROOT / "configs/generator_test.yaml"),
        "--paired-config",
        str(ROOT / "configs/paired_variants.yaml"),
        "--joint-config",
        str(ROOT / "configs/seen_occluded_joint_visual_audit.yaml"),
        "--output-dir",
        str(tmp_path / "audit"),
        "--seed",
        "42",
        "--sample-count",
        "3",
        "--events-per-pair",
        "10",
        "--max-base-states",
        "512",
        "--trajectory-count",
        "21",
        "--max-pairs",
        "512",
        "--max-seen-mothers",
        "4096",
        "--checksum-workers",
        "8",
        "--workers",
        "8",
        "--git-executable",
        "/usr/bin/git",
        *extra,
    ]


def test_cli_forwards_exact_request_and_reports_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    requests = []

    def run(request):
        requests.append(request)
        return SimpleNamespace(
            status="complete",
            output_dir=request.output_dir,
            manifest_sha256="b" * 64,
            checksum_manifest_sha256="c" * 64,
            exit_code=0,
        )

    monkeypatch.setattr(cli, "run_real_audit", run)

    assert cli.main(_argv(tmp_path)) == 0
    assert len(requests) == 1
    request = requests[0]
    assert request.sample_count == 3
    assert request.max_seen_mothers == 4096
    assert request.workers == 8
    assert request.sop03_root == tmp_path / "sop03"
    assert request.paired_config_path == ROOT / "configs/paired_variants.yaml"
    assert request.joint_config_path == (
        ROOT / "configs/seen_occluded_joint_visual_audit.yaml"
    )
    assert json.loads(capsys.readouterr().out) == {
        "checksum_manifest_sha256": "c" * 64,
        "manifest_sha256": "b" * 64,
        "output_dir": str(tmp_path / "audit"),
        "status": "complete",
    }


def test_cli_preflight_never_runs_search_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    prepared = object()
    monkeypatch.setattr(cli, "prepare_real_audit", lambda request: prepared)
    monkeypatch.setattr(
        cli,
        "real_audit_preflight_summary",
        lambda value: {"status": "preflight_ok", "schedule_count": 512}
        if value is prepared
        else pytest.fail("wrong prepared input"),
    )
    monkeypatch.setattr(
        cli,
        "run_real_audit",
        lambda request: pytest.fail("preflight must not run search"),
    )

    assert cli.main(_argv(tmp_path, "--preflight-only")) == 0
    assert not (tmp_path / "audit").exists()
    assert json.loads(capsys.readouterr().out) == {
        "schedule_count": 512,
        "status": "preflight_ok",
    }


def test_cli_returns_structured_error_for_expected_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(
        cli,
        "run_real_audit",
        lambda request: (_ for _ in ()).throw(
            SeenOccludedAuditError("schema mismatch")
        ),
    )

    assert cli.main(_argv(tmp_path)) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err) == {
        "error": "schema mismatch",
        "error_type": "SeenOccludedAuditError",
        "status": "error",
    }
