"""CLI contract for finalized SOP05 to SOP06 history-BEV production."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/06_generate_single_scene_bev.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("sop06_history_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_dispatches_complete_mother_request(
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
            source_family=request.source_family,
            source_mode=request.source_mode,
            source_publication_semantic_digest="a" * 64,
            split=request.split,
            sample_count=7,
            shard_count=2,
            reused_shard_count=0,
            manifest_digest="b" * 64,
        )

    monkeypatch.setattr(cli, "publish_sop06_history_release", publish)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--source-family",
            "natural",
            "--source-mode",
            "complete_mother",
            "--source-root",
            "outputs/mother",
            "--final-scenario-root",
            "outputs/final",
            "--split",
            "val",
            "--output-dir",
            str(tmp_path / "output"),
            "--workers",
            "2",
            "--source-cache-root",
            str(tmp_path / "mother-cache"),
        ],
    )

    assert cli.main() == 0
    request = captured["request"]
    assert request.source_family == "natural"
    assert request.source_mode == "complete_mother"
    assert request.split == "val"
    assert request.workers == 2
    assert request.samples_per_shard == 128
    assert request.source_cache_root == tmp_path / "mother-cache"
    assert request.sop03_root is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["sample_count"] == 7
    assert payload["shard_count"] == 2


def test_cli_requires_partial_reconstruction_arguments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--source-family",
            "natural",
            "--source-mode",
            "partial_m6_reconstruction",
            "--source-root",
            "outputs/partial",
            "--final-scenario-root",
            "outputs/final",
            "--split",
            "train",
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("partial-M6 CLI unexpectedly accepted missing inputs")
