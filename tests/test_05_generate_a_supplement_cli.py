from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/05_generate_a_supplement.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("sop05_a_supplement_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv(tmp_path: Path) -> list[str]:
    return [
        str(SCRIPT),
        "--source-root",
        str(tmp_path / "source"),
        "--output-dir",
        str(tmp_path / "output"),
        "--split",
        "train",
        "--config",
        str(ROOT / "configs/sop05_a_supplement.yaml"),
        "--workers",
        "4",
    ]


def test_cli_publishes_one_split_and_reports_exact_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    source = SimpleNamespace(publication_semantic_digest="source-digest")
    config = SimpleNamespace(digest="config-digest")
    observed = []
    monkeypatch.setattr(cli, "load_sop05r_teb_output", lambda *args, **kwargs: source)
    monkeypatch.setattr(cli, "load_sop05_a_supplement_config", lambda _: config)

    def publish(loaded, **kwargs):
        observed.append((loaded, kwargs))
        return SimpleNamespace(
            publication=SimpleNamespace(
                output_dir=kwargs["output_dir"],
                accepted_count=3,
                source_publication_semantic_digest="source-digest",
            ),
            present_count=1,
            empty_count=2,
            config_digest="config-digest",
        )

    monkeypatch.setattr(cli, "publish_sop05_a_supplement", publish)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    assert cli.main() == 0
    assert observed[0][1]["split"] == "train"
    assert observed[0][1]["workers"] == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "accepted_count": 3,
        "config_digest": "config-digest",
        "empty_count": 2,
        "output_dir": str(tmp_path / "output"),
        "present_count": 1,
        "source_publication_semantic_digest": "source-digest",
        "split": "train",
        "status": "complete",
    }


def test_cli_returns_two_for_an_invalid_or_insufficient_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(
        cli,
        "load_sop05r_teb_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("quota unmet")),
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    assert cli.main() == 2
    assert "quota unmet" in capsys.readouterr().err
