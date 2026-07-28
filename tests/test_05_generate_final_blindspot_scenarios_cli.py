import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/05_generate_final_blindspot_scenarios.py"
    spec = importlib.util.spec_from_file_location("generate_final_blindspot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_m6_cli_requires_explicit_source_reconstruction_inputs() -> None:
    parser = _module()._parser()
    args = parser.parse_args(
        [
            "--source-root",
            "partial",
            "--output-dir",
            "final",
            "--partial-m6-staging",
            "--sop03-root",
            "sop03",
            "--long40-human-artifact",
            "long40",
            "--base-state-start",
            "1026",
            "--max-base-states",
            "11755",
        ]
    )

    assert args.partial_m6_staging is True
    assert args.sop03_root == Path("sop03")
    assert args.base_state_start == 1026
    assert args.max_base_states == 11755
