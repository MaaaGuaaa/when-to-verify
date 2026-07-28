"""CLI contract for the SOP06 offline visual audit renderer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tests.test_sop06_visual_audit import _write_packet


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/06_render_sop06_audit.py"


def _module():
    spec = importlib.util.spec_from_file_location("render_sop06_audit_cli", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery
        raise RuntimeError("cannot load SOP06 audit CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sop06_audit_cli_publishes_required_fixed_names(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    packet = tmp_path / "packet.npz"
    _write_packet(packet)
    output = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--audit-packet",
            str(packet),
            "--output-dir",
            str(output),
            "--frame-duration-ms",
            "250",
        ],
    )

    assert _module().main() == 0
    assert (output / "bev_pair.png").is_file()
    assert (output / "bev_toggle.gif").is_file()
    assert '"candidate_blind_endpoint_reveal_fraction"' in capsys.readouterr().out
