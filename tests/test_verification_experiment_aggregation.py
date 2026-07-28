from __future__ import annotations

import hashlib
import json

import pytest

from src.evaluation.verification_experiment_aggregation import (
    aggregate_verification_evaluations,
    load_verification_aggregate,
)
from src.evaluation.verification_run_artifacts import (
    VERIFICATION_EVALUATION_RUN_LAYOUT_VERSION,
    canonical_json_bytes,
    publish_authenticated_run_directory,
)


def _digest_json(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _evaluation(
    root,
    *,
    seed,
    value,
    split="test",
    split_digest="a" * 64,
    calibration_digest="b" * 64,
    ranking_weight=0.5,
):
    model_config = {
        "model": {"version": "verification_concat_cnn_v0"},
        "loss": {"ranking_weight": ranking_weight},
        "training": {"seed": seed, "epochs": 2},
    }
    report = {
        "schema_version": "4.0.0",
        "run_layout_version": VERIFICATION_EVALUATION_RUN_LAYOUT_VERSION,
        "data_mode": "release",
        "scientific_status": "production_release",
        "split": split,
        "sample_count": 2,
        "group_count": 1,
        "evaluation_split_digests": {split: split_digest},
        "value_calibration_digest": calibration_digest,
        "reject_cost": 0.3,
        "seed": seed,
        "model_config": model_config,
        "model_config_digest": _digest_json(model_config),
        "checkpoint_sha256": str(seed) * 64,
        "checkpoint_code_version": "c" * 40,
        "run_identity": hashlib.sha256(f"run-{root}".encode()).hexdigest(),
    }
    metrics = {
        "schema_version": "4.0.0",
        "split": split,
        "seed": seed,
        "value_calibration_digest": calibration_digest,
        "reject_cost": 0.3,
        "losses": {"total": value + 1.0},
        "learned": {
            "value_mse": value,
            "useful_f1": value / 10.0,
            "slices": {
                "blind_type": {
                    "seen_then_occluded": {
                        "sample_count": 2,
                        "value_mse": value,
                    }
                }
            },
        },
        "baselines": {
            "visible_area": {
                "top1_regret_mean": value + 2.0,
                "slices": {
                    "blind_type": {
                        "seen_then_occluded": {
                            "sample_count": 2,
                            "value_mse": value + 3.0,
                        }
                    }
                },
            }
        },
    }
    predictions = b"".join(
        canonical_json_bytes(
            {
                "sample_id": f"sample-{index}",
                "value_prediction": value,
                "useful_probability": 0.5,
            }
        )
        for index in range(2)
    )
    publish_authenticated_run_directory(
        root,
        layout_version=VERIFICATION_EVALUATION_RUN_LAYOUT_VERSION,
        payloads={
            "evaluation_report.json": canonical_json_bytes(report),
            "metrics.json": canonical_json_bytes(metrics),
            "predictions.jsonl": predictions,
        },
    )
    return root


def test_aggregate_reports_mean_population_std_and_exact_seed_values(tmp_path):
    inputs = tuple(
        _evaluation(
            tmp_path / f"eval-{seed}",
            seed=seed,
            value=float(seed),
        )
        for seed in (1, 2, 3)
    )

    loaded = aggregate_verification_evaluations(
        inputs,
        experiment_id="v0",
        output_dir=tmp_path / "aggregate",
    )

    metric = loaded.summary["metrics"]["learned"]["value_mse"]
    assert loaded.seeds == (1, 2, 3)
    assert metric["per_seed"] == {"1": 1.0, "2": 2.0, "3": 3.0}
    assert metric["mean"] == pytest.approx(2.0)
    assert metric["population_std"] == pytest.approx((2.0 / 3.0) ** 0.5)
    assert loaded.summary["slice_summaries"]["1"]["learned"][
        "blind_type"
    ]["seen_then_occluded"]["sample_count"] == 2
    assert load_verification_aggregate(tmp_path / "aggregate").aggregate_digest == (
        loaded.aggregate_digest
    )
    assert {path.name for path in (tmp_path / "aggregate").iterdir()} == {
        "summary.json",
        "metrics_long.csv",
        "manifest.json",
        "COMPLETE.json",
    }


@pytest.mark.parametrize(
    ("change", "match"),
    (
        ({"seed": 1}, "duplicate seed"),
        ({"split": "val"}, "split"),
        ({"split_digest": "d" * 64}, "split digest"),
        ({"calibration_digest": "e" * 64}, "calibration"),
        ({"ranking_weight": 0.0}, "model config"),
    ),
)
def test_aggregate_rejects_mixed_experiment_identity(tmp_path, change, match):
    first = _evaluation(tmp_path / "first", seed=1, value=1.0)
    values = {
        "seed": 2,
        "value": 2.0,
        "split": "test",
        "split_digest": "a" * 64,
        "calibration_digest": "b" * 64,
        "ranking_weight": 0.5,
    }
    values.update(change)
    second = _evaluation(tmp_path / "second", **values)

    with pytest.raises(ValueError, match=match):
        aggregate_verification_evaluations(
            (first, second),
            experiment_id="v0",
            output_dir=tmp_path / "aggregate",
        )


def test_aggregate_rejects_tampered_authenticated_evaluation(tmp_path):
    source = _evaluation(tmp_path / "evaluation", seed=1, value=1.0)
    (source / "metrics.json").write_text("{}\n")

    with pytest.raises(ValueError, match="checksum"):
        aggregate_verification_evaluations(
            (source,),
            experiment_id="v0",
            output_dir=tmp_path / "aggregate",
        )
