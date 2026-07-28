"""CLI contract for immutable SOP06 history catalog publication."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/06_publish_history_bev_catalog.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("sop06_catalog_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_cli_passes_all_entry_roots(tmp_path: Path, monkeypatch, capsys) -> None:
    cli = _load_cli()
    captured: dict[str, object] = {}

    def publish(roots, output):
        captured["roots"] = tuple(roots)
        captured["output"] = output
        return SimpleNamespace(
            root=output,
            entry_count=2,
            sample_count=5,
            catalog_digest="a" * 64,
        )

    monkeypatch.setattr(cli, "publish_sop06_history_catalog", publish)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--entry-root",
            "outputs/natural",
            "--entry-root",
            "outputs/supplement",
            "--output-dir",
            str(tmp_path / "catalog"),
        ],
    )

    assert cli.main() == 0
    assert captured["roots"] == (
        Path("outputs/natural"),
        Path("outputs/supplement"),
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["entry_count"] == 2
    assert payload["sample_count"] == 5
