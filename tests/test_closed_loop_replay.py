from __future__ import annotations

import json
import hashlib

import pytest


def _artifact(replay, artifact_id: str, digest_character: str):
    return replay.EvidenceArtifact(
        artifact_id=artifact_id,
        sha256=digest_character * 64,
        schema_version="4.0.0",
        long40_layout_version="history8_current7_future32_v1",
    )


def _evidence(
    replay,
    *,
    schema_version: str = "4.0.0",
    scientific_status: str = "framework_fixture_only",
):
    return replay.ReplayEvidence(
        schema_version=schema_version,
        seed=7,
        split="test",
        scientific_status=scientific_status,
        input_manifest=_artifact(replay, "test-input", "a"),
        risk_checkpoint=_artifact(replay, "risk-r2", "b"),
        calibration=_artifact(replay, "split-conformal", "c"),
        value_checkpoint=_artifact(replay, "value-r1", "d"),
        world=_artifact(replay, "world-replay", "e"),
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
        experiment_binding={
            "risk_method": "risk_calibration",
            "value_method": "learned_value",
            "parameters": {"calibrated": True, "target_type": "human"},
        },
    )


def _episode():
    return {
        "episode_id": "episode-0001",
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


def test_replay_suite_is_immutable_authenticated_and_runnable(tmp_path):
    from src.evaluation import closed_loop_replay as replay
    from src.evaluation.closed_loop_runtime import ClosedLoopRuntimeConfig
    from src.evaluation.result_registry import load_result

    suite_path = replay.publish_replay_suite(
        tmp_path / "suite",
        evidence=_evidence(replay),
        episodes=[_episode()],
        offline_metrics={
            "calibration_ece": 0.08,
            "coverage_90": 0.91,
            "top1_regret": 0.04,
        },
    )
    suite = replay.load_replay_suite(suite_path)

    assert suite.evidence.schema_version == "4.0.0"
    assert suite.evidence.experiment_binding["risk_method"] == "risk_calibration"
    assert suite.evidence.experiment_binding["parameters"] == {
        "calibrated": True,
        "target_type": "human",
    }
    assert suite.suite_digest_sha256
    assert len(suite.environments) == 1
    assert suite.environments[0].dynamic_object_counts["human"] == 1
    with pytest.raises(FileExistsError, match="overwrite"):
        replay.publish_replay_suite(
            suite_path,
            evidence=_evidence(replay),
            episodes=[_episode()],
        )

    output = replay.run_replay_evaluation(
        suite_path=suite_path,
        strategy="learned",
        config=ClosedLoopRuntimeConfig(),
        output_dir=tmp_path / "result",
    )
    result = load_result(output)

    assert result.config["evaluation_mode"] == "authenticated_replay"
    assert result.config["suite_digest_sha256"] == suite.suite_digest_sha256
    assert result.provenance.scientific_status == "framework_fixture_only"
    assert result.provenance.risk_checkpoint_digest == "b" * 64
    assert result.provenance.calibration_digest == "c" * 64
    assert result.provenance.value_checkpoint_digest == "d" * 64
    assert result.provenance.world_digest == "e" * 64
    assert result.provenance.target_type_policy_digest == "1" * 64
    assert result.provenance.object_type_counts == {
        "human": 1,
        "carried_object": 0,
        "unknown_dynamic": 0,
    }
    assert result.metrics["collision_rate"] == pytest.approx(0.0)
    assert result.metrics["success_rate"] == pytest.approx(1.0)
    assert result.metrics["calibration_ece"] == pytest.approx(0.08)
    assert result.metrics["coverage_90"] == pytest.approx(0.91)
    assert result.metrics["top1_regret"] == pytest.approx(0.04)
    assert result.episodes[0]["steps"][0]["decision"] == "verify"
    assert result.episodes[0]["steps"][0]["next_plan_id"] == "plan-1"


def test_replay_loader_rejects_episode_tampering(tmp_path):
    from src.evaluation import closed_loop_replay as replay

    suite_path = replay.publish_replay_suite(
        tmp_path / "suite",
        evidence=_evidence(replay),
        episodes=[_episode()],
    )
    episodes_path = suite_path / "episodes.json"
    document = json.loads(episodes_path.read_text(encoding="utf-8"))
    document["episodes"][0]["frames"]["s0"]["calibrated_risk"] = 0.0
    episodes_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="episodes digest"):
        replay.load_replay_suite(suite_path)


def test_replay_evidence_rejects_non_long40_artifact_or_schema():
    from src.evaluation import closed_loop_replay as replay

    with pytest.raises(ValueError, match="schema_version"):
        _evidence(replay, schema_version="3.0.0")
    with pytest.raises(ValueError, match="Long40"):
        replay.EvidenceArtifact(
            artifact_id="legacy-risk",
            sha256="a" * 64,
            schema_version="4.0.0",
            long40_layout_version="history8_current7_future15_v1",
        )


def test_replay_loader_rejects_transition_plan_identity_mismatch(tmp_path):
    from src.evaluation import closed_loop_replay as replay

    episode = _episode()
    episode["verify_transitions"][0]["next_plan_id"] = "stale-plan"

    with pytest.raises(ValueError, match="next plan"):
        replay.publish_replay_suite(
            tmp_path / "suite",
            evidence=_evidence(replay),
            episodes=[episode],
        )


def test_production_replay_publish_requires_verified_upstream_files(tmp_path):
    from src.evaluation import closed_loop_replay as replay

    with pytest.raises(ValueError, match="artifact_paths"):
        replay.publish_replay_suite(
            tmp_path / "suite",
            evidence=_evidence(
                replay,
                scientific_status="production_evaluation",
            ),
            episodes=[_episode()],
        )


def test_production_replay_publish_verifies_each_upstream_digest(tmp_path):
    from src.evaluation import closed_loop_replay as replay

    paths = {}
    artifacts = {}
    for field_name in (
        "input_manifest",
        "risk_checkpoint",
        "calibration",
        "value_checkpoint",
        "world",
    ):
        path = tmp_path / f"{field_name}.json"
        payload = f"{field_name}-schema4-long40\n".encode()
        path.write_bytes(payload)
        paths[field_name] = path
        artifacts[field_name] = replay.EvidenceArtifact(
            artifact_id=field_name,
            sha256=hashlib.sha256(payload).hexdigest(),
            schema_version="4.0.0",
            long40_layout_version="history8_current7_future32_v1",
        )
    evidence = replay.ReplayEvidence(
        schema_version="4.0.0",
        seed=7,
        split="test",
        scientific_status="production_evaluation",
        **artifacts,
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

    suite_path = replay.publish_replay_suite(
        tmp_path / "suite",
        evidence=evidence,
        episodes=[_episode()],
        artifact_paths=paths,
    )

    assert replay.load_replay_suite(suite_path).upstream_files_verified is True
