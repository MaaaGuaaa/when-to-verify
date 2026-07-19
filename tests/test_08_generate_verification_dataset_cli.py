import importlib.util
import json
from pathlib import Path

import pytest

from src.contracts import build_grid_spec
from src.datasets.verification_dataloader import load_verification_shard
from src.planning.verification_actions import load_verification_actions
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/08_generate_verification_dataset.py"


def _module():
    spec = importlib.util.spec_from_file_location("generate_verification_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _toy_args(output: Path):
    return [
        "--mode",
        "toy",
        "--output-dir",
        str(output),
        "--sample-count",
        "12",
        "--config",
        str(ROOT / "configs/base.yaml"),
        "--actions-config",
        str(ROOT / "configs/verification_actions.yaml"),
        "--gt-config",
        str(ROOT / "configs/verification_gt.yaml"),
        "--bank-size",
        "8",
        "--max-replan-candidates",
        "3",
        "--seed",
        "17",
    ]


def test_toy_cli_is_deterministic_immutable_and_explicitly_smoke_only(tmp_path):
    module = _module()
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert module.main(_toy_args(first)) == 0
    assert module.main(_toy_args(second)) == 0

    first_report = json.loads((first / "generation_report.json").read_text())
    second_report = json.loads((second / "generation_report.json").read_text())
    assert first_report["scientific_status"] == "toy_smoke_only"
    assert first_report["sample_count"] == 12
    assert first_report["group_count"] == 2
    assert first_report["collection_semantic_digest"] == (
        second_report["collection_semantic_digest"]
    )
    assert first_report["limitations"] == [
        "toy data are not paper-scale evidence",
        "validation/test performance and cross-split leakage are not proven",
    ]
    toy_config = load_config(ROOT / "configs/base.yaml")
    toy_config["bev"].update({"range_m": 8.0, "size": 80})
    loaded = load_verification_shard(
        first / "shard-00000",
        grid=build_grid_spec(toy_config),
        library=load_verification_actions(
            ROOT / "configs/verification_actions.yaml"
        ),
    )
    assert len(loaded.samples) == 12

    with pytest.raises(FileExistsError, match="overwrite"):
        module.main(_toy_args(first))


@pytest.mark.parametrize("count", [9, 11, 102])
def test_cli_rejects_out_of_bounds_or_incomplete_six_action_counts(tmp_path, count):
    module = _module()
    args = _toy_args(tmp_path / f"bad-{count}")
    args[args.index("12")] = str(count)
    with pytest.raises(ValueError, match="sample_count"):
        module.main(args)


def test_real_mode_never_falls_back_when_trust_paths_are_missing(tmp_path):
    module = _module()
    args = _toy_args(tmp_path / "real")
    args[args.index("toy")] = "sop05-train"
    with pytest.raises(ValueError, match="required for sop05-train"):
        module.main(args)
