"""CLI contract tests for the bounded SOP-05 runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.contracts import ContractError
from src.generation import sop05_run
from src.generation.event_sampler import GeneratorConfigError
from src.generation.sop05_input_adapter import Sop05InputError
from src.generation.sop05_run import Sop05RunError
from src.utils.config import ConfigError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/05_generate_events.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("sop05_generate_events_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        str(SCRIPT),
        "--sop03-root",
        str(tmp_path / "sop03"),
        "--sop04-root",
        str(tmp_path / "sop04"),
        "--sop04-handoff-digest",
        "e" * 64,
        "--split",
        "train",
        "--base-config",
        str(ROOT / "configs/base.yaml"),
        "--generator-config",
        str(ROOT / "configs/generator_train.yaml"),
        "--output-dir",
        str(tmp_path / "run"),
        "--seed",
        "23",
        "--accepted-quota",
        "20",
        "--events-per-pair",
        "10",
        "--max-base-states",
        "2",
        "--trajectory-count",
        "1",
        "--max-pairs",
        "2",
        "--checksum-workers",
        "4",
        "--workers",
        "3",
        "--git-executable",
        str(tmp_path / "git"),
        *extra,
    ]


def test_cli_executes_once_and_forwards_positive_worker_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    requests = []

    def execute(request):
        requests.append(request)
        return SimpleNamespace(
            run_state="complete",
            run_id="sop05-run-fixture",
            output_dir=request.output_dir,
            generation_summary={
                "selected_count": 20,
                "allocated_cpu_seconds": 12.5,
            },
            publication_semantic_digest="a" * 64,
            exit_code=0,
        )

    monkeypatch.setattr(cli, "execute_sop05_run", execute)
    monkeypatch.setattr(
        cli,
        "prepare_sop05_run",
        lambda request: pytest.fail("normal CLI must not pre-load inputs"),
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    assert cli.main() == 0
    assert len(requests) == 1
    request = requests[0]
    assert request.workers == 3
    assert request.checksum_workers == 4
    assert request.events_per_pair == 10
    assert request.sop03_root == tmp_path / "sop03"
    assert request.sop04_external_handoff_digest_sha256 == "e" * 64
    assert request.git_executable == tmp_path / "git"
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "allocated_cpu_seconds": 12.5,
        "output_dir": str(tmp_path / "run"),
        "producer_version": "sop05_generation_run_v6",
        "publication_semantic_digest": "a" * 64,
        "run_id": "sop05-run-fixture",
        "run_state": "complete",
        "selection_version": "sop05_diversity_total_selection_v1",
        "selected_count": 20,
    }


def test_cli_preflight_loads_once_and_never_executes_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    calls = []
    prepared = object()

    def prepare(request):
        calls.append(request)
        assert not request.output_dir.exists()
        return prepared

    monkeypatch.setattr(cli, "prepare_sop05_run", prepare)
    monkeypatch.setattr(
        cli,
        "execute_sop05_run",
        lambda request: pytest.fail("preflight must not execute generation"),
    )
    monkeypatch.setattr(
        cli,
        "preflight_summary",
        lambda value: {
            "status": "preflight_ok",
            "run_id": "sop05-run-fixture",
            "output_dir": str(tmp_path / "run"),
        }
        if value is prepared
        else pytest.fail("wrong prepared run"),
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--preflight-only"))

    assert cli.main() == 0
    assert len(calls) == 1
    assert not (tmp_path / "run").exists()
    assert json.loads(capsys.readouterr().out) == {
        "output_dir": str(tmp_path / "run"),
        "publication_semantic_digest": None,
        "run_id": "sop05-run-fixture",
        "status": "preflight_ok",
    }


def test_cli_returns_four_for_atomically_published_quota_shortfall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()

    def execute(request):
        return SimpleNamespace(
            run_state="quota_unmet",
            run_id="sop05-run-fixture",
            output_dir=request.output_dir,
            generation_summary={
                "selected_count": 6,
                "allocated_cpu_seconds": 3.25,
            },
            publication_semantic_digest=None,
            exit_code=4,
        )

    monkeypatch.setattr(cli, "execute_sop05_run", execute)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    assert cli.main() == 4
    assert json.loads(capsys.readouterr().out) == {
        "allocated_cpu_seconds": 3.25,
        "output_dir": str(tmp_path / "run"),
        "producer_version": "sop05_generation_run_v6",
        "publication_semantic_digest": None,
        "run_id": "sop05-run-fixture",
        "run_state": "quota_unmet",
        "selection_version": "sop05_diversity_total_selection_v1",
        "selected_count": 6,
    }


def test_cli_rejects_nonpositive_workers_before_loading_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(
        cli,
        "execute_sop05_run",
        lambda request: pytest.fail("invalid CLI must not load inputs"),
    )
    argv = _argv(tmp_path)
    argv[argv.index("--workers") + 1] = "0"
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 2


@pytest.mark.parametrize("digest", ["e" * 63, "E" * 64, "g" * 64])
def test_cli_rejects_noncanonical_sop04_handoff_digest_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    digest: str,
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(
        cli,
        "execute_sop05_run",
        lambda request: pytest.fail("invalid CLI must not load inputs"),
    )
    argv = _argv(tmp_path)
    argv[argv.index("--sop04-handoff-digest") + 1] = digest
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 2


@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize(
    "error_type",
    [
        Sop05InputError,
        Sop05RunError,
        ConfigError,
        GeneratorConfigError,
        ContractError,
        FileExistsError,
    ],
)
def test_cli_reports_expected_input_errors_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    preflight: bool,
    error_type: type[Exception],
) -> None:
    cli = _load_cli()
    error = error_type("fixture input failure")

    def fail(_request):
        raise error

    monkeypatch.setattr(cli, "prepare_sop05_run", fail)
    monkeypatch.setattr(cli, "execute_sop05_run", fail)
    extra = ("--preflight-only",) if preflight else ()
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, *extra))

    assert cli.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: fixture input failure\n"
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize(
    "config_option", ["--base-config", "--generator-config"]
)
def test_cli_reports_yaml_syntax_errors_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    preflight: bool,
    config_option: str,
) -> None:
    cli = _load_cli()
    malformed_config = tmp_path / f"malformed-{config_option[2:]}.yaml"
    malformed_config.write_text("broken: [1,\n", encoding="utf-8")
    git_executable = tmp_path / "git"
    git_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    git_executable.chmod(0o755)
    monkeypatch.setattr(
        sop05_run,
        "_load_producer_source_identity",
        lambda _git_executable: {
            "version": "sop05_producer_source_identity_v1",
            "git_commit": "3" * 40,
            "worktree_state": "clean",
            "dirty_tree_sha256": None,
        },
    )
    extra = ("--preflight-only",) if preflight else ()
    argv = _argv(tmp_path, *extra)
    argv[argv.index(config_option) + 1] = str(malformed_config)
    monkeypatch.setattr(sys, "argv", argv)

    assert cli.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert captured.err != "error: \n"
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_cli_does_not_swallow_unexpected_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preflight: bool,
    error_type: type[Exception],
) -> None:
    cli = _load_cli()

    def fail(_request):
        raise error_type("unexpected internal failure")

    monkeypatch.setattr(cli, "prepare_sop05_run", fail)
    monkeypatch.setattr(cli, "execute_sop05_run", fail)
    extra = ("--preflight-only",) if preflight else ()
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, *extra))

    with pytest.raises(error_type, match="unexpected internal failure"):
        cli.main()
