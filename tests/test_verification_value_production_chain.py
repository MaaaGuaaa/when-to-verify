from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from src.evaluation.verification_experiment_aggregation import (
    aggregate_verification_evaluations,
)
from src.evaluation.verification_value_calibration import (
    publish_reject_cost_calibration,
)
from src.generation.verification_release import publish_verification_release
from tests.test_verification_release import (
    _build_one_factory,
    _request,
    _source,
    release_harness,
)


ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _revaluation_build(template):
    provenance = {
        **template.provenance,
        "blind_type": "seen_then_occluded",
        "target_object_type": "human",
        "target_footprint_kind": "circle",
    }

    def build(source, accepted, library, gt_config, max_replan_candidates):
        values = {}
        for index, action in enumerate(library.actions):
            action_id = action.action_id
            policy_loss = (
                0.19 if (index + accepted.source_index) % 2 == 0 else 0.35
            )
            action_cost = 0.05
            post_before_cost = min(policy_loss, 0.2)
            post_risk = post_before_cost + action_cost
            value_target = 0.2 - post_risk
            values[action_id] = replace(
                template.value_results[action_id],
                realized_execute_loss=1.0,
                reject_cost=0.2,
                br_before=0.2,
                realized_post_decision_risk_before_action_cost=post_before_cost,
                unclipped_best_policy_loss=policy_loss,
                action_cost=action_cost,
                post_risk=post_risk,
                value_target=value_target,
                useful_target=int(value_target > 0.0),
            )
        dynamic_template = replace(
            template,
            provenance=provenance,
            value_results=values,
        )
        return _build_one_factory(dynamic_template)(
            source,
            accepted,
            library,
            gt_config,
            max_replan_candidates,
        )

    return build


def _fast_config(source: Path, destination: Path) -> Path:
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["training"].update({"epochs": 1, "batch_size": 6})
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return destination


@pytest.mark.usefixtures("release_harness")
def test_release_calibration_train_evaluate_and_aggregate_chain(
    release_harness,
):
    harness = release_harness
    harness.actions_path.write_bytes(
        (ROOT / "configs/verification_actions.yaml").read_bytes()
    )
    harness.gt_path.write_bytes(
        (ROOT / "configs/verification_gt.yaml").read_bytes()
    )
    build_one = _revaluation_build(harness.template)

    train_source = _source(2, harness.grid)
    train_request = _request(harness, groups_per_shard=2)
    publish_verification_release(
        train_request,
        source_loader=lambda _: train_source,
        build_one=build_one,
    )
    val_source = _source(1, harness.grid)
    val_source.accepted = tuple(
        replace(row, split="val") for row in val_source.accepted
    )
    val_request = replace(
        train_request,
        split="val",
        output_dir=harness.root / "outputs/verification-val",
    )
    publish_verification_release(
        val_request,
        source_loader=lambda _: val_source,
        build_one=build_one,
    )

    calibration_config = harness.root / "configs/value-calibration.yaml"
    calibration_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": "4.0.0",
                "calibration_version": "verification_reject_cost_calibration_v1",
                "candidates": [0.2, 0.3, 0.5],
                "criteria": {
                    "minimum_group_count": 2,
                    "minimum_positive_fraction": 0.25,
                    "maximum_positive_fraction": 0.75,
                    "minimum_mixed_action_count": 2,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    calibration_dir = harness.root / "outputs/value-calibration"
    calibrated = publish_reject_cost_calibration(
        calibration_dir,
        release_dirs=(train_request.output_dir,),
        config_path=calibration_config,
        gt_config_path=harness.gt_path,
    )
    assert calibrated.selected_reject_cost == pytest.approx(0.3)

    base = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    base["bev"].update({"range_m": 8.0, "size": 80})
    base_path = harness.root / "configs/base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))
    v0_config = _fast_config(
        ROOT / "configs/verify_model.yaml",
        harness.root / "configs/v0.yaml",
    )
    no_ranking_config = _fast_config(
        ROOT / "configs/verify_model_no_ranking.yaml",
        harness.root / "configs/no-ranking.yaml",
    )
    train_cli = _script("09_train_verification_model.py")
    evaluate_cli = _script("10_evaluate_verification_model.py")
    evaluation_dirs = []
    for seed in (1, 2):
        trained = harness.root / f"outputs/v0-seed-{seed}"
        assert train_cli.main(
            [
                "--release-dir",
                str(train_request.output_dir),
                "--value-calibration",
                str(calibration_dir),
                "--output-dir",
                str(trained),
                "--base-config",
                str(base_path),
                "--actions-config",
                str(ROOT / "configs/verification_actions.yaml"),
                "--model-config",
                str(v0_config),
                "--code-version",
                "9" * 40,
                "--seed",
                str(seed),
            ]
        ) == 0
        evaluated = harness.root / f"outputs/v0-eval-seed-{seed}"
        assert evaluate_cli.main(
            [
                "--split",
                "val",
                "--release-dir",
                str(val_request.output_dir),
                "--value-calibration",
                str(calibration_dir),
                "--checkpoint",
                str(trained / "checkpoint.pt"),
                "--checkpoint-manifest",
                str(trained / "manifest.json"),
                "--output-dir",
                str(evaluated),
                "--base-config",
                str(base_path),
                "--actions-config",
                str(ROOT / "configs/verification_actions.yaml"),
                "--model-config",
                str(v0_config),
                "--expected-code-version",
                "9" * 40,
            ]
        ) == 0
        evaluation_dirs.append(evaluated)

    aggregate = aggregate_verification_evaluations(
        tuple(evaluation_dirs),
        experiment_id="v0",
        output_dir=harness.root / "outputs/v0-aggregate",
    )
    assert aggregate.seeds == (1, 2)

    no_ranking = harness.root / "outputs/no-ranking-seed-1"
    assert train_cli.main(
        [
            "--release-dir",
            str(train_request.output_dir),
            "--value-calibration",
            str(calibration_dir),
            "--output-dir",
            str(no_ranking),
            "--base-config",
            str(base_path),
            "--actions-config",
            str(ROOT / "configs/verification_actions.yaml"),
            "--model-config",
            str(no_ranking_config),
            "--code-version",
            "9" * 40,
            "--seed",
            "1",
        ]
    ) == 0
    no_ranking_manifest = json.loads(
        (no_ranking / "manifest.json").read_text(encoding="utf-8")
    )
    assert no_ranking_manifest["model_config"]["loss"]["ranking_weight"] == 0.0
    assert (no_ranking / "COMPLETE.json").is_file()
