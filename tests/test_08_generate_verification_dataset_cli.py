import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.contracts import build_grid_spec
from src.datasets.verification_dataloader import load_verification_shard
from src.planning.verification_actions import load_verification_actions
from src.generation.verification_pipeline import VerificationSourceIneligibleError
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
    assert first_report["value_semantics"] == (
        "one_sop05_sampled_child_per_group"
    )
    assert len(first_report["sampled_child_world_ids"]) == 2
    assert {
        "bank_size",
        "posterior_mode",
        "posterior_temperature",
        "scenario_bank_digests",
    }.isdisjoint(first_report)
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


def test_heldout_mode_requires_explicit_split_and_never_falls_back(tmp_path):
    module = _module()
    args = _toy_args(tmp_path / "heldout")
    args[args.index("toy")] = "sop05-heldout"
    with pytest.raises(ValueError, match="--split"):
        module.main(args)

    args.extend(["--split", "val"])
    with pytest.raises(ValueError, match="required for sop05-heldout"):
        module.main(args)


def test_cli_exposes_no_downstream_bank_or_posterior_controls():
    module = _module()
    destinations = {action.dest for action in module._parser()._actions}
    assert {
        "bank_size",
        "posterior_mode",
        "posterior_temperature",
    }.isdisjoint(destinations)


def test_finalized_release_routes_one_task_per_mother_without_sample_count(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _module()
    captured = {}

    def publish(request, *, progress_callback):
        captured["request"] = request
        progress_callback(1, 1, False)
        return SimpleNamespace(
            output_dir=request.output_dir,
            split=request.split,
            task_count=3,
            accepted_group_count=2,
            rejected_task_count=1,
            sample_count=12,
            shard_count=1,
            reused_shard_count=0,
            manifest_digest="a" * 64,
        )

    monkeypatch.setattr(module, "publish_verification_release", publish)
    args = [
        "--mode",
        "sop05-final",
        "--split",
        "train",
        "--source-family",
        "natural",
        "--source-mode",
        "complete_mother",
        "--source-root",
        str(tmp_path / "source"),
        "--final-scenario-root",
        str(tmp_path / "final"),
        "--output-dir",
        str(tmp_path / "release"),
        "--actions-config",
        str(ROOT / "configs/verification_actions.yaml"),
        "--gt-config",
        str(ROOT / "configs/verification_gt.yaml"),
        "--workers",
        "2",
        "--groups-per-shard",
        "16",
        "--max-tasks",
        "3",
    ]

    assert module.main(args) == 0
    request = captured["request"]
    assert request.max_tasks == 3
    assert request.groups_per_shard == 16
    assert request.workers == 2
    assert request.max_replan_candidates == 4
    assert json.loads(capsys.readouterr().out)["sample_count"] == 12

    with pytest.raises(ValueError, match="one fixed task per mother"):
        module.main([*args, "--sample-count", "12"])


def test_real_source_selection_retries_only_typed_ineligibility():
    module = _module()

    def build(value):
        if value in {"bad-current", "bad-variant"}:
            reason = (
                "scenario_current_static_overlap"
                if value == "bad-current"
                else "scenario_variant_static_overlap"
            )
            raise VerificationSourceIneligibleError(reason, value)
        return f"group::{value}"

    selection = module._generate_exact_eligible_groups(
        ("bad-current", "good-1", "bad-variant", "good-2", "unused"),
        required_group_count=2,
        build_group=build,
        event_id=lambda value: value,
    )

    assert selection.groups == ("group::good-1", "group::good-2")
    assert selection.accepted_event_ids == ("good-1", "good-2")
    assert selection.attempted_event_count == 4
    assert selection.rejection_counts == {
        "scenario_current_static_overlap": 1,
        "scenario_variant_static_overlap": 1,
    }
    assert selection.rejected_event_ids == {
        "scenario_current_static_overlap": ("bad-current",),
        "scenario_variant_static_overlap": ("bad-variant",),
    }


def test_real_source_selection_fails_closed_when_pool_is_exhausted():
    module = _module()

    def reject(value):
        raise VerificationSourceIneligibleError("infeasible_actions", value)

    with pytest.raises(RuntimeError, match="eligible groups"):
        module._generate_exact_eligible_groups(
            ("bad-1", "bad-2"),
            required_group_count=1,
            build_group=reject,
            event_id=lambda value: value,
        )


def test_real_source_selection_does_not_swallow_unexpected_errors():
    module = _module()

    with pytest.raises(KeyError, match="unexpected"):
        module._generate_exact_eligible_groups(
            ("event",),
            required_group_count=1,
            build_group=lambda _: (_ for _ in ()).throw(KeyError("unexpected")),
            event_id=lambda value: value,
        )
