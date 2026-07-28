from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/05_render_sop05r_audit.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("sop05r_audit_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        str(SCRIPT),
        "--sop05r-root",
        str(tmp_path / "sop05r"),
        "--sop03-root",
        str(tmp_path / "sop03"),
        "--paired-config",
        str(ROOT / "configs/paired_variants_visual_audit.yaml"),
        "--output-dir",
        str(tmp_path / "audit"),
        "--sample-count",
        "100",
        "--seed",
        "20260724",
        "--checksum-workers",
        "4",
        *extra,
    ]


def test_cli_runs_one_strict_sop05r_audit(
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
            selected_event_ids=tuple(f"event-{index}" for index in range(100)),
            manifest_sha256="a" * 64,
            checksum_manifest_sha256="b" * 64,
            exit_code=0,
        )

    monkeypatch.setattr(cli, "run_sop05r_audit", run)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    assert cli.main() == 0
    assert len(requests) == 1
    assert requests[0].sample_count == 100
    assert requests[0].checksum_workers == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "complete"
    assert payload["selected_count"] == 100
    assert payload["manifest_sha256"] == "a" * 64


def test_cli_rejects_nonpositive_sample_count_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli()
    argv = _argv(tmp_path)
    argv[argv.index("--sample-count") + 1] = "0"
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        cli,
        "run_sop05r_audit",
        lambda request: pytest.fail("invalid CLI must not load inputs"),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 2


def test_cli_runs_current_teb_visual_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    verification_config = ROOT / "configs/verification_actions.yaml"
    source = SimpleNamespace(
        manifest={
            "config_digest": "teb-config-digest",
            "verification_action_digest": cli._sha256_file(verification_config),
        }
    )
    captured = []
    monkeypatch.setattr(cli, "load_sop05r_teb_output", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(
        cli,
        "load_sop05r_teb_config",
        lambda _path: SimpleNamespace(digest="teb-config-digest"),
    )
    monkeypatch.setattr(cli, "load_verification_actions", lambda _path: "actions")
    monkeypatch.setattr(
        cli,
        "publish_sop05r_teb_visual_audit",
        lambda *args, **kwargs: (
            captured.append((args, kwargs))
            or SimpleNamespace(
                output_dir=kwargs["output_dir"],
                selected_event_ids=("event-1",),
                publication_semantic_digest="c" * 64,
            )
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--generator-mode",
            "obstacle_first_teb",
            "--source-root",
            str(tmp_path / "source"),
            "--output-dir",
            str(tmp_path / "audit"),
            "--sample-count",
            "1",
            "--seed",
            "31",
        ],
    )

    assert cli.main() == 0
    assert captured[0][1]["sample_count"] == 1
    assert captured[0][1]["action_library"] == "actions"
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_event_ids"] == ["event-1"]
