"""CLI contract tests for the current SOP05 lightweight-TEB runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.contracts import ContractError
from src.generation.sop05r_contracts import Sop05rConfigError
from src.generation.sop05r_teb_run import Sop05rTebRunError
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
        "--generator-mode",
        "obstacle_first_teb",
        "--sop03-root",
        str(tmp_path / "sop03"),
        "--long40-human-artifact",
        str(tmp_path / "long40"),
        "--split",
        "train",
        "--base-config",
        str(ROOT / "configs/base.yaml"),
        "--generator-config",
        str(ROOT / "configs/generator_obstacle_first_teb_train.yaml"),
        "--verification-action-config",
        str(ROOT / "configs/verification_actions.yaml"),
        "--output-dir",
        str(tmp_path / "run"),
        "--seed",
        "23",
        "--accepted-quota",
        "20",
        "--max-base-states",
        "2",
        "--checksum-workers",
        "4",
        "--workers",
        "3",
        "--git-executable",
        str(tmp_path / "git"),
        *extra,
    ]


def _config_variant(tmp_path: Path, name: str, old: str, new: str) -> Path:
    path = tmp_path / name
    source = ROOT / "configs/generator_obstacle_first_teb_train.yaml"
    payload = source.read_text(encoding="utf-8")
    assert old in payload
    path.write_text(payload.replace(old, new), encoding="utf-8")
    return path


def test_teb_preflight_validates_current_config_without_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(
        cli,
        "execute_sop05r_teb_run",
        lambda *args, **kwargs: pytest.fail("preflight must not generate"),
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--preflight-only"))

    assert cli.main() == 0
    assert not (tmp_path / "run").exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "preflight_ok"
    assert payload["generator_algorithm_version"] == (
        "obstacle_first_lightweight_teb_v8"
    )
    assert len(payload["config_digest"]) == 64
    assert payload["publication_semantic_digest"] is None


@pytest.mark.parametrize(
    "option", ["--generator-mode", "--long40-human-artifact", "--verification-action-config"]
)
def test_teb_cli_requires_current_mode_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
) -> None:
    argv = _argv(tmp_path)
    index = argv.index(option)
    del argv[index : index + 2]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        _load_cli().main()

    assert exc_info.value.code == 2


@pytest.mark.parametrize("retired_mode", ["legacy", "obstacle_first"])
def test_cli_rejects_retired_generator_modes_at_parse_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    retired_mode: str,
) -> None:
    argv = _argv(tmp_path, "--preflight-only")
    argv[argv.index("--generator-mode") + 1] = retired_mode
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        _load_cli().main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err
    assert retired_mode in captured.err


def test_teb_cli_rejects_wrong_typed_current_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = _config_variant(
        tmp_path,
        "wrong-typed.yaml",
        "  future_steps: 32",
        "  future_steps: true",
    )
    argv = _argv(tmp_path, "--preflight-only")
    argv[argv.index("--generator-config") + 1] = str(invalid)
    monkeypatch.setattr(sys, "argv", argv)

    assert _load_cli().main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid obstacle_first_teb generator config" in captured.err
    assert "trajectory.future_steps" in captured.err
    assert "Traceback" not in captured.err


def test_teb_cli_rejects_unknown_generator_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = _config_variant(
        tmp_path,
        "unknown-version.yaml",
        "generator_algorithm_version: obstacle_first_lightweight_teb_v8",
        "generator_algorithm_version: unsupported_generator",
    )
    argv = _argv(tmp_path, "--preflight-only")
    argv[argv.index("--generator-config") + 1] = str(invalid)
    monkeypatch.setattr(sys, "argv", argv)

    assert _load_cli().main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "generator_algorithm_version" in captured.err
    assert "Traceback" not in captured.err


def test_teb_preflight_rejects_invalid_verification_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = tmp_path / "invalid-actions.yaml"
    invalid.write_text("actions: []\n", encoding="utf-8")
    argv = _argv(tmp_path, "--preflight-only")
    argv[argv.index("--verification-action-config") + 1] = str(invalid)
    monkeypatch.setattr(sys, "argv", argv)

    assert _load_cli().main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "verification action config" in captured.err
    assert "Traceback" not in captured.err


def test_teb_cli_dispatches_once_and_forwards_runtime_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    requests = []

    def execute(request, *, progress_callback):
        assert callable(progress_callback)
        requests.append(request)
        return SimpleNamespace(
            run_state="quota_unmet",
            output_dir=request.output_dir,
            accepted_count=3,
            requested_count=20,
            publication_semantic_digest="c" * 64,
            exit_code=4,
        )

    monkeypatch.setattr(cli, "execute_sop05r_teb_run", execute)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    assert cli.main() == 4
    assert len(requests) == 1
    request = requests[0]
    assert request.accepted_quota == 20
    assert request.max_base_states == 2
    assert request.checksum_workers == 4
    assert request.workers == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["producer_version"] == "sop05r_lightweight_teb_generation_run_v7"
    assert payload["accepted_count"] == 3
    assert payload["requested_count"] == 20


def test_teb_cli_dispatches_all_accepted_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    requests = []

    def execute(request, *, progress_callback):
        assert callable(progress_callback)
        requests.append(request)
        return SimpleNamespace(
            run_state="complete",
            output_dir=request.output_dir,
            accepted_count=3,
            requested_count=3,
            publication_semantic_digest="d" * 64,
            exit_code=0,
        )

    argv = _obstacle_first_teb_argv(tmp_path)
    quota_index = argv.index("--accepted-quota")
    del argv[quota_index : quota_index + 2]
    argv.append("--all-accepted")
    monkeypatch.setattr(cli, "execute_sop05r_teb_run", execute)
    monkeypatch.setattr(sys, "argv", argv)

    assert cli.main() == 0
    assert len(requests) == 1
    assert requests[0].accepted_quota is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["accepted_count"] == payload["requested_count"] == 3


def test_teb_cli_propagates_h0_hidden_placement_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli()
    requests = []

    def execute(request, *, progress_callback):
        assert callable(progress_callback)
        requests.append(request)
        return SimpleNamespace(
            run_state="complete",
            output_dir=request.output_dir,
            accepted_count=20,
            requested_count=20,
            publication_semantic_digest="d" * 64,
            exit_code=0,
        )

    monkeypatch.setattr(cli, "execute_sop05r_teb_run", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        _obstacle_first_teb_argv(
            tmp_path,
            "--placement-selection-mode",
            "h0_hidden",
        ),
    )

    assert cli.main() == 0
    assert len(requests) == 1
    assert requests[0].placement_selection_mode == "h0_hidden"


def test_teb_cli_writes_parent_progress_to_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()

    def execute(request, *, progress_callback):
        progress_callback(
            {
                "progress_version": "sop05r_teb_progress_v1",
                "accepted_count": 3,
                "requested_count": 20,
            }
        )
        return SimpleNamespace(
            run_state="quota_unmet",
            output_dir=request.output_dir,
            accepted_count=3,
            requested_count=20,
            publication_semantic_digest="c" * 64,
            exit_code=4,
        )

    monkeypatch.setattr(cli, "execute_sop05r_teb_run", execute)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    assert cli.main() == 4
    stderr_lines = capsys.readouterr().err.splitlines()
    assert json.loads(stderr_lines[-1]) == {
        "accepted_count": 3,
        "progress_version": "sop05r_teb_progress_v1",
        "requested_count": 20,
    }


def test_cli_rejects_nonpositive_workers_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _argv(tmp_path)
    argv[argv.index("--workers") + 1] = "0"
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        _load_cli().main()

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "error_type",
    [Sop05rTebRunError, Sop05rConfigError, ConfigError, ContractError, FileExistsError],
)
def test_cli_reports_expected_errors_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error_type: type[Exception],
) -> None:
    cli = _load_cli()

    def fail(*args, **kwargs):
        raise error_type("fixture input failure")

    monkeypatch.setattr(cli, "execute_sop05r_teb_run", fail)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    assert cli.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: fixture input failure\n"
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_cli_does_not_swallow_unexpected_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    cli = _load_cli()

    def fail(*args, **kwargs):
        raise error_type("unexpected internal failure")

    monkeypatch.setattr(cli, "execute_sop05r_teb_run", fail)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    with pytest.raises(error_type, match="unexpected internal failure"):
        cli.main()
