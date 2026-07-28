import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from src.evaluation.result_registry import load_result
from src.evaluation.toy_experiment_matrix import (
    build_toy_scenarios,
    load_toy_matrix_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repository_toy_matrix_config_resolves_its_shared_closed_loop_config():
    matrix = load_toy_matrix_config(ROOT / "configs/experiments/toy.yaml")

    assert matrix.closed_loop_config_path == ROOT / "configs/closed_loop.yaml"


def test_toy_scenarios_have_policy_specific_values_and_seed_specific_content():
    first = build_toy_scenarios(7, 3)
    second = build_toy_scenarios(11, 3)
    revealable = next(scenario for scenario in first if scenario.action_values)

    assert revealable.action_values_for("learned") != revealable.action_values_for(
        "visible"
    )
    assert [scenario.required_execute_time_s for scenario in first] != [
        scenario.required_execute_time_s for scenario in second
    ]


def _script_module(script_name: str):
    path = ROOT / "scripts" / script_name
    assert path.is_file(), f"{script_name} is missing"
    spec = importlib.util.spec_from_file_location(script_name.replace(".", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _closed_loop_config(path: Path) -> Path:
    payload = {
        "schema_version": "4.0.0",
        "closed_loop_version": "sop15_toy_closed_loop_v1",
        "future_horizon_s": 6.4,
        "execute_step_s": 0.2,
        "verify_step_s": 0.6,
        "risk_weight": 1.0,
        "verify_margin": 0.01,
        "collision_risk_threshold": 0.7,
        "near_miss_risk_threshold": 0.4,
        "max_decisions": 32,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_closed_loop_cli_publishes_one_structured_toy_run(tmp_path):
    module = _script_module("11_eval_closed_loop.py")
    config = _closed_loop_config(tmp_path / "closed_loop.yaml")
    output = tmp_path / "one-run"

    assert (
        module.main(
            [
                "--config",
                str(config),
                "--strategy",
                "learned",
                "--seed",
                "7",
                "--episode-count",
                "3",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    loaded = load_result(output)

    assert loaded.config["strategy"] == "learned"
    assert loaded.provenance.seed == 7
    assert loaded.metrics["episode_count"] == 3.0
    assert len(loaded.episodes) == 3


def test_matrix_cli_publishes_seed_aggregates_without_overwriting(tmp_path):
    module = _script_module("12_run_experiment_matrix.py")
    closed_loop = _closed_loop_config(tmp_path / "closed_loop.yaml")
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "schema_version": "4.0.0",
                "matrix_version": "sop16_toy_matrix_v1",
                "closed_loop_config": closed_loop.name,
                "seeds": [7, 11],
                "strategies": ["never", "learned"],
                "episode_count": 3,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "matrix-output"
    args = ["--config", str(matrix), "--output-dir", str(output)]

    assert module.main(args) == 0
    summary = json.loads((output / "matrix_summary.json").read_text(encoding="utf-8"))
    assert summary["run_count"] == 4
    assert summary["aggregates"]["learned"]["seed_count"] == 2
    assert (output / "learned-seed-7").is_dir()
    assert (output / "never-seed-11").is_dir()
    matrix_module = importlib.import_module("src.evaluation.toy_experiment_matrix")
    loader = getattr(matrix_module, "load_toy_matrix_result", None)
    assert loader is not None, "matrix output loader is missing"
    (output / "matrix_summary.json").write_text(
        json.dumps({"tampered": True}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="summary"):
        loader(output)
    with pytest.raises(FileExistsError, match="overwrite"):
        module.main(args)


def _runtime_config(path: Path) -> Path:
    payload = {
        "schema_version": "4.0.0",
        "runtime_version": "sop15_closed_loop_runtime_v1",
        "future_horizon_s": 6.4,
        "execute_step_s": 0.2,
        "risk_weight": 1.0,
        "verify_margin": 0.01,
        "minimum_verify_duration_s": 0.5,
        "maximum_verify_duration_s": 1.0,
        "max_decisions": 32,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _publish_one_step_replay(path: Path) -> Path:
    from src.evaluation import closed_loop_replay as replay

    def artifact(artifact_id: str, character: str):
        return replay.EvidenceArtifact(
            artifact_id=artifact_id,
            sha256=character * 64,
            schema_version="4.0.0",
            long40_layout_version="history8_current7_future32_v1",
        )

    evidence = replay.ReplayEvidence(
        schema_version="4.0.0",
        seed=17,
        split="test",
        scientific_status="framework_fixture_only",
        input_manifest=artifact("input", "a"),
        risk_checkpoint=artifact("risk", "b"),
        calibration=artifact("calibration", "c"),
        value_checkpoint=artifact("value", "d"),
        world=artifact("world", "e"),
        dynamic_objects_config_digest="f" * 64,
        target_type_policy="human_target_with_contextual_dynamic_objects",
        target_type_policy_digest="1" * 64,
        object_type_counts={
            "human": 1,
            "carried_object": 0,
            "unknown_dynamic": 0,
        },
        geometry_source="typed_continuous_swept_footprint",
        geometry_fallback_fraction=0.0,
    )
    return replay.publish_replay_suite(
        path,
        evidence=evidence,
        episodes=[
            {
                "episode_id": "episode-cli",
                "initial_state_id": "s0",
                "hazard_object_type": "human",
                "dynamic_object_counts": {
                    "human": 1,
                    "carried_object": 0,
                    "unknown_dynamic": 0,
                },
                "nominal_task_time_s": 0.2,
                "nominal_path_length_m": 0.1,
                "frames": {
                    "s0": {
                        "plan_id": "plan-0",
                        "task_cost": 0.1,
                        "calibrated_risk": 0.1,
                        "reject_cost": 1.0,
                        "action_values_by_strategy": {},
                        "action_durations_s": {},
                    }
                },
                "execute_transitions": {
                    "s0": {
                        "duration_s": 0.2,
                        "path_length_m": 0.1,
                        "task_complete": True,
                        "termination_reason": "goal_reached",
                    }
                },
                "verify_transitions": [],
            }
        ],
    )


def test_repository_production_closed_loop_config_is_long40():
    from src.evaluation.closed_loop_replay import load_runtime_config

    config = load_runtime_config(ROOT / "configs/closed_loop_production.yaml")

    assert config.future_horizon_s == pytest.approx(6.4)
    assert config.execute_step_s == pytest.approx(0.2)


def test_closed_loop_cli_runs_authenticated_replay_without_toy_arguments(tmp_path):
    module = _script_module("11_eval_closed_loop.py")
    suite = _publish_one_step_replay(tmp_path / "suite")
    config = _runtime_config(tmp_path / "runtime.yaml")
    output = tmp_path / "result"

    assert (
        module.main(
            [
                "--mode",
                "replay",
                "--config",
                str(config),
                "--suite",
                str(suite),
                "--strategy",
                "never",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    result = load_result(output)

    assert result.config["evaluation_mode"] == "authenticated_replay"
    assert result.run_id == "never-seed-17"
    assert result.provenance.seed == 17
    assert result.metrics["success_rate"] == pytest.approx(1.0)
