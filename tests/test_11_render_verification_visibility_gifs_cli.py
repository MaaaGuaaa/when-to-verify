from __future__ import annotations

import importlib.util
from pathlib import Path

from src.planning.verification_actions import load_verification_actions


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "11_render_verification_visibility_gifs.py"
ACTIONS = ROOT / "configs" / "verification_actions.yaml"


def _module():
    spec = importlib.util.spec_from_file_location(
        "render_verification_visibility_gifs_cli",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_selects_the_five_motion_actions_in_audited_order() -> None:
    module = _module()
    actions = module._motion_actions(load_verification_actions(ACTIONS))

    assert tuple(action.action_id for action in actions) == (
        "arc_left_30",
        "arc_right_30",
        "arc_left_45",
        "arc_right_45",
        "forward_peek",
    )
