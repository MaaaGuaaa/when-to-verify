import importlib
import importlib.util

import pytest


def _registry_module():
    spec = importlib.util.find_spec("src.evaluation.result_registry")
    assert spec is not None, "SOP-16 result registry module is missing"
    return importlib.import_module("src.evaluation.result_registry")


def _provenance(registry, seed):
    return registry.ResultProvenance(
        schema_version="4.0.0",
        seed=seed,
        input_manifest_digest="a" * 64,
        checkpoint_id="toy-no-checkpoint",
        scientific_status="toy_framework_only",
        dynamic_objects_config_digest="b" * 64,
        target_type_policy="human_target_with_contextual_dynamic_objects",
        geometry_source="toy_scalar_risk",
        geometry_fallback_fraction=1.0,
    )


def test_registry_publishes_an_immutable_result_with_provenance_and_episode_trace(
    tmp_path,
):
    registry = _registry_module()
    output = tmp_path / "learned-seed-7"

    published = registry.publish_result(
        output,
        run_id="learned-seed-7",
        config={"strategy": "learned", "seed": 7},
        provenance=_provenance(registry, 7),
        metrics={"collision_rate": 0.0, "success_rate": 1.0},
        episodes=[{"episode_id": "toy-0", "termination_reason": "success"}],
    )
    loaded = registry.load_result(output)

    assert published == output
    assert loaded.run_id == "learned-seed-7"
    assert loaded.manifest["config_digest_sha256"]
    assert loaded.provenance.seed == 7
    assert loaded.metrics["success_rate"] == 1.0
    assert loaded.episodes[0]["episode_id"] == "toy-0"
    with pytest.raises(FileExistsError, match="overwrite"):
        registry.publish_result(
            output,
            run_id="learned-seed-7",
            config={"strategy": "learned", "seed": 7},
            provenance=_provenance(registry, 7),
            metrics={"collision_rate": 0.0, "success_rate": 1.0},
            episodes=[],
        )


def test_registry_aggregates_raw_seed_values_with_mean_and_standard_deviation(tmp_path):
    registry = _registry_module()
    first = registry.publish_result(
        tmp_path / "learned-seed-7",
        run_id="learned-seed-7",
        config={"strategy": "learned", "seed": 7},
        provenance=_provenance(registry, 7),
        metrics={"collision_rate": 0.0, "success_rate": 1.0},
        episodes=[],
    )
    second = registry.publish_result(
        tmp_path / "learned-seed-11",
        run_id="learned-seed-11",
        config={"strategy": "learned", "seed": 11},
        provenance=_provenance(registry, 11),
        metrics={"collision_rate": 0.5, "success_rate": 0.5},
        episodes=[],
    )

    aggregate = registry.aggregate_seed_metrics(
        [registry.load_result(first), registry.load_result(second)]
    )

    learned = aggregate["learned"]
    assert learned["seed_count"] == 2
    assert learned["metrics"]["collision_rate"] == {
        "mean": 0.25,
        "std": 0.25,
        "seed_values": {"7": 0.0, "11": 0.5},
    }


def test_registry_refuses_to_pool_seeds_with_different_experiment_identity(tmp_path):
    registry = _registry_module()
    first = registry.publish_result(
        tmp_path / "learned-seed-7",
        run_id="learned-seed-7",
        config={
            "strategy": "learned",
            "seed": 7,
            "closed_loop": {"execute_step_s": 0.2},
        },
        provenance=_provenance(registry, 7),
        metrics={"collision_rate": 0.0, "success_rate": 1.0},
        episodes=[],
    )
    second = registry.publish_result(
        tmp_path / "learned-seed-11",
        run_id="learned-seed-11",
        config={
            "strategy": "learned",
            "seed": 11,
            "closed_loop": {"execute_step_s": 0.4},
        },
        provenance=_provenance(registry, 11),
        metrics={"collision_rate": 0.5, "success_rate": 0.5},
        episodes=[],
    )

    with pytest.raises(ValueError, match="experiment identity"):
        registry.aggregate_seed_metrics(
            [registry.load_result(first), registry.load_result(second)]
        )
