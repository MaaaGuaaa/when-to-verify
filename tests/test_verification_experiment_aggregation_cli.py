from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/10_aggregate_verification_value.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "aggregate_verification_value_cli",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aggregation_cli_passes_repeated_evaluations(tmp_path, monkeypatch):
    module = _module()
    calls = []

    def aggregate(evaluation_dirs, **kwargs):
        calls.append((evaluation_dirs, kwargs))
        return SimpleNamespace(
            experiment_id="v0",
            run_count=2,
            seeds=(1, 2),
            aggregate_digest="a" * 64,
            root=kwargs["output_dir"],
        )

    monkeypatch.setattr(module, "aggregate_verification_evaluations", aggregate)
    output = tmp_path / "aggregate"
    assert module.main(
        [
            "--evaluation-dir",
            str(tmp_path / "eval-1"),
            "--evaluation-dir",
            str(tmp_path / "eval-2"),
            "--experiment-id",
            "v0",
            "--output-dir",
            str(output),
        ]
    ) == 0
    assert calls == [
        (
            (tmp_path / "eval-1", tmp_path / "eval-2"),
            {"experiment_id": "v0", "output_dir": output},
        )
    ]
