"""CLI contract for persisted SOP06 to resumable SOP07 production."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/07_generate_single_scene_risk.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("sop07_risk_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_dispatches_sop06_release_request(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cli = _load_cli()
    captured: dict[str, object] = {}

    def publish(request, **kwargs):
        captured["request"] = request
        return SimpleNamespace(
            output_dir=request.output_dir,
            split="train",
            sample_count=6790,
            shard_count=54,
            reused_shard_count=3,
            manifest_digest="a" * 64,
        )

    monkeypatch.setattr(cli, "publish_sop07_risk_release", publish)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--sop06-release",
            "outputs/sop06/train-first10k",
            "--output-dir",
            str(tmp_path / "sop07"),
            "--sop03-root",
            "inputs/sop03",
            "--long40-human-artifact",
            "inputs/long40",
        ],
    )

    assert cli.main() == 0
    request = captured["request"]
    assert request.sop06_release_root == Path(
        "outputs/sop06/train-first10k"
    )
    assert request.output_dir == tmp_path / "sop07"
    assert request.sop03_root == Path("inputs/sop03")
    assert request.long40_human_artifact == Path("inputs/long40")
    payload = json.loads(capsys.readouterr().out)
    assert payload["sample_count"] == 6790
    assert payload["shard_count"] == 54
    assert payload["reused_shard_count"] == 3
