from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/05_render_sop05_ab_visual_audit.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("sop05_ab_visual_audit_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_conditions_regime_a_visuals_on_pedestrian_presence(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cli = _load_cli()
    source = SimpleNamespace(
        manifest={"config_digest": "teb-digest", "base_config": {}},
        publication_semantic_digest="source-digest",
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_sop05r_teb_output", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        cli,
        "load_sop05r_teb_config",
        lambda *args, **kwargs: SimpleNamespace(digest="teb-digest"),
    )
    monkeypatch.setattr(cli, "_sha256_file", lambda path: "actions-digest")
    monkeypatch.setattr(cli, "load_verification_actions", lambda path: "actions")
    monkeypatch.setattr(cli, "_load_unseen_config", lambda *args, **kwargs: "unseen")
    monkeypatch.setattr(cli, "load_seen_prior_config", lambda path: "seen")

    def publish(source_arg, **kwargs):
        calls["source"] = source_arg
        calls.update(kwargs)
        return SimpleNamespace(
            output_dir=tmp_path / "audit",
            regime_a_event_ids=("a0",) * 10,
            regime_b_event_ids=("b0",) * 10,
            source_publication_semantic_digest="source-digest",
        )

    monkeypatch.setattr(cli, "publish_sop05_ab_visual_audit", publish)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--source-root",
            str(tmp_path / "source"),
            "--output-dir",
            str(tmp_path / "audit"),
            "--regime-a-present-only",
        ],
    )

    assert cli.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert calls["source"] is source
    assert calls["sample_count_per_regime"] == 10
    assert calls["selection_seed"] == 20260727
    assert calls["regime_a_present_only"] is True
    assert payload["regime_a_count"] == payload["regime_b_count"] == 10
