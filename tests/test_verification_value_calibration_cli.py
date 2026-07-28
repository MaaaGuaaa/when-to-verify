from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/08_calibrate_verification_value.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "calibrate_verification_value_cli",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_calibration_cli_passes_all_inputs_and_returns_selection_status(
    tmp_path,
    monkeypatch,
):
    module = _module()
    calls = []

    def publish(output_dir, **kwargs):
        calls.append((output_dir, kwargs))
        return SimpleNamespace(
            status="selected",
            selected_reject_cost=0.3,
            calibration_digest="a" * 64,
        )

    monkeypatch.setattr(module, "publish_reject_cost_calibration", publish)
    output = tmp_path / "calibration"
    result = module.main(
        [
            "--release-dir",
            str(tmp_path / "release-a"),
            "--release-dir",
            str(tmp_path / "release-b"),
            "--config",
            str(tmp_path / "calibration.yaml"),
            "--gt-config",
            str(tmp_path / "verification-gt.yaml"),
            "--output-dir",
            str(output),
        ]
    )

    assert result == 0
    assert calls == [
        (
            output,
            {
                "release_dirs": (
                    tmp_path / "release-a",
                    tmp_path / "release-b",
                ),
                "config_path": tmp_path / "calibration.yaml",
                "gt_config_path": tmp_path / "verification-gt.yaml",
            },
        )
    ]


def test_calibration_cli_returns_two_for_complete_failed_selection(
    tmp_path,
    monkeypatch,
):
    module = _module()
    monkeypatch.setattr(
        module,
        "publish_reject_cost_calibration",
        lambda *args, **kwargs: SimpleNamespace(
            status="no_candidate_passed",
            selected_reject_cost=None,
            calibration_digest="b" * 64,
        ),
    )

    result = module.main(
        [
            "--release-dir",
            str(tmp_path / "release"),
            "--config",
            str(tmp_path / "calibration.yaml"),
            "--gt-config",
            str(tmp_path / "verification-gt.yaml"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert result == 2
