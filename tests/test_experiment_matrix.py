from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _artifact(replay, artifact_id: str, character: str):
    return replay.EvidenceArtifact(
        artifact_id=artifact_id,
        sha256=character * 64,
        schema_version="4.0.0",
        long40_layout_version="history8_current7_future32_v1",
    )


def _publish_suite(
    path: Path,
    *,
    seed: int,
    episode_count: int = 3,
    experiment_binding=None,
) -> Path:
    from src.evaluation import closed_loop_replay as replay

    evidence = replay.ReplayEvidence(
        schema_version="4.0.0",
        seed=seed,
        split="test",
        scientific_status="framework_fixture_only",
        input_manifest=_artifact(replay, f"input-{seed}", "a"),
        risk_checkpoint=_artifact(replay, "risk-main", "b"),
        calibration=_artifact(replay, "calibration-main", "c"),
        value_checkpoint=_artifact(replay, "value-main", "d"),
        world=_artifact(replay, "world-main", "e"),
        dynamic_objects_config_digest="f" * 64,
        target_type_policy="human_target_with_contextual_dynamic_objects",
        target_type_policy_digest="1" * 64,
        object_type_counts={
            "human": episode_count,
            "carried_object": 0,
            "unknown_dynamic": 0,
        },
        geometry_source="typed_continuous_swept_footprint",
        geometry_fallback_fraction=0.0,
        experiment_binding=(
            {
                "risk_method": "risk_calibration",
                "value_method": "learned_value",
                "parameters": {
                    "calibrated": True,
                    "target_type": "human",
                },
            }
            if experiment_binding is None
            else experiment_binding
        ),
    )
    episodes = []
    for index in range(episode_count):
        episodes.append(
            {
                "episode_id": f"seed-{seed}-episode-{index}",
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
                        "calibrated_risk": 0.9,
                        "reject_cost": 1.2,
                        "action_values_by_strategy": {
                            "always": {"peek": 0.0},
                            "visible": {"peek": -0.1},
                            "swept": {"peek": 0.2},
                            "entropy": {"peek": 0.1},
                            "learned": {"peek": 0.8},
                            "oracle": {"peek": 0.9},
                        },
                        "action_durations_s": {"peek": 0.6},
                    },
                    "s1": {
                        "plan_id": "plan-1",
                        "task_cost": 0.1,
                        "calibrated_risk": 0.1,
                        "reject_cost": 1.2,
                        "action_values_by_strategy": {},
                        "action_durations_s": {},
                    },
                },
                "execute_transitions": {
                    "s0": {
                        "duration_s": 0.2,
                        "path_length_m": 0.1,
                        "collision": True,
                        "termination_reason": "continuous_collision",
                    },
                    "s1": {
                        "duration_s": 0.2,
                        "path_length_m": 0.1,
                        "task_complete": True,
                        "termination_reason": "goal_reached",
                    },
                },
                "verify_transitions": [
                    {
                        "state_id": "s0",
                        "action_id": "peek",
                        "duration_s": 0.6,
                        "path_length_m": 0.02,
                        "next_state_id": "s1",
                        "next_plan_id": "plan-1",
                        "replanned": True,
                        "critical_actor_revealed": True,
                        "verification_useful": True,
                    }
                ],
            }
        )
    return replay.publish_replay_suite(
        path,
        evidence=evidence,
        episodes=episodes,
        offline_metrics={
            "calibration_ece": 0.05 + seed / 1000.0,
            "coverage_90": 0.9,
            "top1_regret": 0.04,
        },
    )


def _runtime_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "4.0.0",
                "runtime_version": "sop15_closed_loop_runtime_v1",
                "future_horizon_s": 6.4,
                "execute_step_s": 0.2,
                "risk_weight": 1.0,
                "verify_margin": 0.01,
                "minimum_verify_duration_s": 0.5,
                "maximum_verify_duration_s": 1.0,
                "max_decisions": 32,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _matrix_config(path: Path, runtime_path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "4.0.0",
                "matrix_version": "sop16_experiment_matrix_v1",
                "matrix_name": "fixture-main",
                "scientific_status": "framework_fixture_only",
                "closed_loop_config": runtime_path.name,
                "required_target_type_policy": (
                    "human_target_with_contextual_dynamic_objects"
                ),
                "seeds": [7, 11],
                "strategies": ["never", "learned"],
                "experiments": [
                    {
                        "experiment_id": "risk-calibrated-main",
                        "category": "main",
                        "risk_method": "risk_calibration",
                        "value_method": "learned_value",
                        "suite_pattern": "main/seed-{seed}",
                        "parameters": {
                            "calibrated": True,
                            "target_type": "human",
                        },
                        "runtime_overrides": {},
                        "claims": [
                            "learned verification exposes the safety-efficiency tradeoff"
                        ],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_matrix_runs_seeds_and_builds_pareto_cases_and_claim_index(tmp_path):
    from src.evaluation.experiment_matrix import (
        load_experiment_matrix_result,
        run_experiment_matrix,
    )

    runtime = _runtime_config(tmp_path / "runtime.yaml")
    config = _matrix_config(tmp_path / "matrix.yaml", runtime)
    suite_root = tmp_path / "suites"
    _publish_suite(suite_root / "main/seed-7", seed=7)
    _publish_suite(suite_root / "main/seed-11", seed=11)

    result = run_experiment_matrix(
        config,
        suite_root=suite_root,
        output_dir=tmp_path / "matrix-output",
    )
    loaded = load_experiment_matrix_result(result.output_dir)
    experiment = loaded.summary["experiments"]["risk-calibrated-main"]

    assert loaded.summary["expected_run_count"] == 4
    assert loaded.summary["completed_run_count"] == 4
    assert loaded.summary["failed_run_count"] == 0
    assert loaded.summary["scientifically_complete"] is True
    assert loaded.summary["acceptance_gates"]["pareto_available"]["status"] == "pass"
    assert loaded.summary["acceptance_gates"]["case_trace_count"]["status"] == "pass"
    assert (
        loaded.summary["acceptance_gates"]["learned_vs_visible_fixed_budget"][
            "status"
        ]
        == "not_evaluable"
    )
    assert experiment["aggregates"]["never"]["seed_count"] == 2
    assert (
        experiment["aggregates"]["never"]["metrics"]["collision_rate"]["mean"]
        == pytest.approx(1.0)
    )
    assert (
        experiment["aggregates"]["learned"]["metrics"]["collision_rate"]["mean"]
        == pytest.approx(0.0)
    )
    assert {row["strategy"] for row in loaded.pareto_rows} == {
        "never",
        "learned",
    }
    assert all(row["pareto_optimal"] for row in loaded.pareto_rows)
    assert 5 <= len(loaded.case_index["cases"]) <= 10
    claim = loaded.claim_index["claims"][0]
    assert claim["evidence"]["summary_json_pointer"].startswith(
        "/experiments/risk-calibrated-main"
    )
    assert all(Path(path).is_dir() for path in loaded.run_paths)

    pareto_path = result.output_dir / "pareto.json"
    pareto_path.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        load_experiment_matrix_result(result.output_dir)


def test_matrix_publishes_failed_runs_instead_of_silently_dropping_them(tmp_path):
    from src.evaluation.experiment_matrix import run_experiment_matrix

    runtime = _runtime_config(tmp_path / "runtime.yaml")
    config = _matrix_config(tmp_path / "matrix.yaml", runtime)
    suite_root = tmp_path / "suites"
    _publish_suite(suite_root / "main/seed-7", seed=7)

    result = run_experiment_matrix(
        config,
        suite_root=suite_root,
        output_dir=tmp_path / "matrix-output",
    )

    assert result.summary["expected_run_count"] == 4
    assert result.summary["completed_run_count"] == 2
    assert result.summary["failed_run_count"] == 2
    assert result.summary["scientifically_complete"] is False
    assert len(result.failures["failures"]) == 2
    assert {failure["seed"] for failure in result.failures["failures"]} == {11}
    assert all(failure["error_type"] for failure in result.failures["failures"])


def test_threshold_sensitivity_matrix_also_emits_pareto_rows(tmp_path):
    from src.evaluation.experiment_matrix import run_experiment_matrix

    runtime = _runtime_config(tmp_path / "runtime.yaml")
    config = _matrix_config(tmp_path / "matrix.yaml", runtime)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    experiment = document["experiments"][0]
    experiment["category"] = "sensitivity"
    experiment["parameters"] = {"axis": "verify_margin", "value": 0.01}
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    suite_root = tmp_path / "suites"
    binding = {
        "risk_method": "risk_calibration",
        "value_method": "learned_value",
        "parameters": {"axis": "verify_margin", "value": 0.01},
    }
    _publish_suite(
        suite_root / "main/seed-7",
        seed=7,
        experiment_binding=binding,
    )
    _publish_suite(
        suite_root / "main/seed-11",
        seed=11,
        experiment_binding=binding,
    )

    result = run_experiment_matrix(
        config,
        suite_root=suite_root,
        output_dir=tmp_path / "matrix-output",
    )

    assert result.pareto_rows
    assert {row["strategy"] for row in result.pareto_rows} == {
        "never",
        "learned",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("risk_method", "risk_only"),
        ("value_method", "value_without_ranking"),
        (
            "parameters",
            {
                "calibrated": True,
                "target_type": "human",
                "ablation": "ranking",
            },
        ),
    ),
)
def test_matrix_records_suite_method_binding_mismatch_before_evaluation(
    tmp_path,
    field,
    value,
):
    from src.evaluation.experiment_matrix import run_experiment_matrix

    runtime = _runtime_config(tmp_path / "runtime.yaml")
    config = _matrix_config(tmp_path / "matrix.yaml", runtime)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["seeds"] = [7]
    document["strategies"] = ["learned"]
    document["experiments"][0][field] = value
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    suite_root = tmp_path / "suites"
    _publish_suite(suite_root / "main/seed-7", seed=7)

    result = run_experiment_matrix(
        config,
        suite_root=suite_root,
        output_dir=tmp_path / "matrix-output",
    )

    assert result.summary["completed_run_count"] == 0
    assert result.summary["failed_run_count"] == 1
    assert "binding" in result.failures["failures"][0]["error_message"]


def test_matrix_excludes_runtime_overrides_from_suite_identity(tmp_path):
    from src.evaluation.experiment_matrix import run_experiment_matrix

    runtime = _runtime_config(tmp_path / "runtime.yaml")
    config = _matrix_config(tmp_path / "matrix.yaml", runtime)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["seeds"] = [7]
    document["strategies"] = ["learned"]
    experiment = document["experiments"][0]
    experiment["parameters"] = {"axis": "verify_margin", "value": 0.02}
    experiment["runtime_overrides"] = {"verify_margin": 0.02}
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    suite_root = tmp_path / "suites"
    _publish_suite(
        suite_root / "main/seed-7",
        seed=7,
        experiment_binding={
            "risk_method": "risk_calibration",
            "value_method": "learned_value",
            "parameters": {},
        },
    )

    result = run_experiment_matrix(
        config,
        suite_root=suite_root,
        output_dir=tmp_path / "matrix-output",
    )

    assert result.summary["completed_run_count"] == 1
    assert result.summary["failed_run_count"] == 0


def test_matrix_config_rejects_suite_path_escape(tmp_path):
    from src.evaluation.experiment_matrix import load_experiment_matrix_config

    runtime = _runtime_config(tmp_path / "runtime.yaml")
    config = _matrix_config(tmp_path / "matrix.yaml", runtime)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["experiments"][0]["suite_pattern"] = "../outside/seed-{seed}"
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="suite_pattern"):
        load_experiment_matrix_config(config)


def test_repository_matrix_configs_register_required_sop16_axes():
    from src.evaluation.experiment_matrix import load_experiment_matrix_config

    main = load_experiment_matrix_config(ROOT / "configs/experiments/main.yaml")
    ablations = load_experiment_matrix_config(
        ROOT / "configs/experiments/ablations.yaml"
    )
    sensitivity = load_experiment_matrix_config(
        ROOT / "configs/experiments/sensitivity.yaml"
    )

    assert set(main.strategies) == {
        "never",
        "always",
        "visible",
        "swept",
        "entropy",
        "learned",
        "oracle",
    }
    assert {experiment.risk_method for experiment in main.experiments} >= {
        "last",
        "age",
        "occupancy_aggregation",
        "risk_only",
        "risk_calibration",
        "risk_aux_optional",
    }
    assert {
        experiment.parameters.get("controlled_test")
        for experiment in main.experiments
        if experiment.category == "controlled"
    } == {"same_area", "temporal_safe", "irrelevant_hidden", "empty"}
    assert {experiment.experiment_id for experiment in ablations.experiments} == {
        "without-age",
        "without-history",
        "without-trajectory-query",
        "without-calibration",
        "without-ranking",
    }
    sensitivity_values = {
        (experiment.parameters["axis"], str(experiment.parameters["value"]))
        for experiment in sensitivity.experiments
    }
    assert {value for axis, value in sensitivity_values if axis == "scenario_bank_m"} == {
        "8",
        "16",
        "32",
    }
    assert {value for axis, value in sensitivity_values if axis == "posterior_tau"} == {
        "0.1",
        "0.2",
        "0.5",
    }
    assert {
        "scenario_composition",
        "signature_feature",
        "world_prior",
        "verification_cost_scale",
        "verify_margin",
        "risk_weight",
    } <= {axis for axis, _ in sensitivity_values}


def _script_module(script_name: str):
    path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_cli_runs_authenticated_replay_mode(tmp_path):
    from src.evaluation.experiment_matrix import load_experiment_matrix_result

    runtime = _runtime_config(tmp_path / "runtime.yaml")
    config = _matrix_config(tmp_path / "matrix.yaml", runtime)
    suite_root = tmp_path / "suites"
    _publish_suite(suite_root / "main/seed-7", seed=7)
    _publish_suite(suite_root / "main/seed-11", seed=11)
    output = tmp_path / "matrix-output"
    module = _script_module("12_run_experiment_matrix.py")

    assert (
        module.main(
            [
                "--mode",
                "replay",
                "--config",
                str(config),
                "--suite-root",
                str(suite_root),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    loaded = load_experiment_matrix_result(output)
    assert loaded.summary["completed_run_count"] == 4


def test_report_is_rebuilt_from_authenticated_matrix_artifacts(tmp_path):
    pytest.importorskip("matplotlib")
    from src.evaluation.experiment_matrix import run_experiment_matrix
    from src.evaluation.plots import (
        build_evaluation_report,
        load_evaluation_report,
    )

    runtime = _runtime_config(tmp_path / "runtime.yaml")
    config = _matrix_config(tmp_path / "matrix.yaml", runtime)
    suite_root = tmp_path / "suites"
    _publish_suite(suite_root / "main/seed-7", seed=7)
    _publish_suite(suite_root / "main/seed-11", seed=11)
    matrix = run_experiment_matrix(
        config,
        suite_root=suite_root,
        output_dir=tmp_path / "matrix-output",
    )

    report = build_evaluation_report(
        [matrix.output_dir],
        output_dir=tmp_path / "report",
    )
    loaded = load_evaluation_report(report.output_dir)

    assert {
        "pareto.pdf",
        "pareto.png",
        "cases.pdf",
        "cases.png",
        "offline_metrics.pdf",
        "offline_metrics.png",
    } <= set(loaded.generated_files)
    assert (report.output_dir / "pareto.pdf").read_bytes().startswith(b"%PDF")
    assert (report.output_dir / "pareto.png").read_bytes().startswith(b"\x89PNG")
    metrics_header = (
        report.output_dir / "metrics_long.csv"
    ).read_text(encoding="utf-8").splitlines()[0]
    assert "seed_values_json" in metrics_header
    assert loaded.summary["source_matrix_count"] == 1
    assert loaded.summary["all_sources_scientifically_complete"] is True
    assert loaded.summary["case_count"] == 10
    assert loaded.summary["offline_metric_names"] == [
        "calibration_ece",
        "coverage_90",
        "top1_regret",
    ]

    metrics_path = report.output_dir / "metrics_long.csv"
    metrics_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        load_evaluation_report(report.output_dir)


def test_report_cli_builds_publication_outputs(tmp_path):
    pytest.importorskip("matplotlib")
    from src.evaluation.experiment_matrix import run_experiment_matrix

    runtime = _runtime_config(tmp_path / "runtime.yaml")
    config = _matrix_config(tmp_path / "matrix.yaml", runtime)
    suite_root = tmp_path / "suites"
    _publish_suite(suite_root / "main/seed-7", seed=7)
    _publish_suite(suite_root / "main/seed-11", seed=11)
    matrix = run_experiment_matrix(
        config,
        suite_root=suite_root,
        output_dir=tmp_path / "matrix-output",
    )
    output = tmp_path / "report"
    module = _script_module("13_build_report.py")

    assert (
        module.main(
            [
                "--matrix-dir",
                str(matrix.output_dir),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "report_manifest.json").is_file()
