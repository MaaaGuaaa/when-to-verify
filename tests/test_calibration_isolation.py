from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.calibration.split_conformal import (
    CALIBRATION_ARTIFACT_LAYOUT_VERSION,
    PREDICTION_TABLE_LAYOUT_VERSION,
    CalibrationContractError,
    assert_calibration_test_isolation,
    calibration_artifact_semantic_digest,
    fit_calibration_artifact,
    prediction_table_semantic_digest,
    validate_calibration_artifact,
    validate_prediction_table,
)
from src.datasets.toy_risk_learning import (
    TOY_MANIFEST_ROW_KEYS,
    frozen_channel_spec,
)
from src.calibration.grouped_calibration import (
    GROUP_DIMENSIONS,
    fit_grouped_calibration,
)


def _row(sample_id: str, split: str, **overrides: object) -> dict:
    row = {
        "sample_id": sample_id,
        "split": split,
        "pair_group_id": f"pair-{sample_id}",
        "event_type": "collision",
        "recording_id": f"recording-{sample_id}",
        "session_id": f"session-{sample_id}",
        "source_object_id": f"object-{sample_id}",
        "snippet_id": f"snippet-{sample_id}",
        "base_state_id": f"base-{sample_id}",
        "seed_namespace": f"seed-{sample_id}",
        "trajectory_id": f"trajectory-{sample_id}",
        "occluder_id": f"occluder-{sample_id}",
        "background_id": f"background-{sample_id}",
        "collision_label": 1,
        "risk_severity": 0.8,
        "min_clearance": 0.0,
        "near_miss": 0,
        "first_collision_time": 0.2,
        "critical_object_id": f"critical-{sample_id}",
        "p_collision": 0.8,
        "q50": 0.2,
        "q80": 0.4,
        "q90": 0.7,
        "q95": 0.9,
        "blind_type": "corner",
        "critical_area_fraction": 0.04,
        "age_s": 0.8,
        "density_fraction": 0.02,
        "target_object_type": "human",
        "footprint_kind": "single_grid_cell_square",
        "footprint_dimensions_m": [0.2, 0.2],
        "robot_footprint_kind": "single_grid_cell_square",
        "robot_footprint_dimensions_m": [0.2, 0.2],
        "footprint_contact_policy": "positive_area_overlap",
        "ood_tag": "in_distribution",
        "pair_eligible": True,
    }
    row.update(overrides)
    return row


def _test_cohort_digest(table: dict) -> str:
    dataset_digest = table.get(
        "toy_dataset_manifest_digest", table.get("risk_dataset_manifest_digest")
    )
    payload = {
        "cohort_layout_version": "risk_prediction_cohort_v1",
        "mode": table["mode"],
        "schema_version": table["schema_version"],
        "split": table["split"],
        "dataset_manifest_digest": dataset_digest,
        "rows": [
            {field: row[field] for field in sorted(TOY_MANIFEST_ROW_KEYS)}
            for row in table["rows"]
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _toy_table(split: str, rows: list[dict]) -> dict:
    table = {
        "prediction_table_layout_version": PREDICTION_TABLE_LAYOUT_VERSION,
        "mode": "toy",
        "schema_version": "3.0.0",
        "split": split,
        "method_id": "r0",
        "checkpoint_layout_version": "risk_model_checkpoint_v2",
        "checkpoint_digest": "d" * 64,
        "checkpoint_digest_kind": "risk_checkpoint_semantic_sha256",
        "toy_dataset_manifest_digest": "a" * 32,
        "seed": 42,
        "channel_spec": frozen_channel_spec(),
        "config_digest_sha256": "c" * 64,
        "rows": rows,
    }
    table["cohort_digest_sha256"] = _test_cohort_digest(table)
    table["semantic_digest"] = prediction_table_semantic_digest(table)
    return table


def _as_occupancy_table(table: dict) -> dict:
    result = copy.deepcopy(table)
    result["method_id"] = "occupancy_b3_hand_aggregation"
    result["checkpoint_layout_version"] = "occupancy_baseline_checkpoint_v2"
    result["checkpoint_digest"] = "e" * 64
    result["checkpoint_digest_kind"] = "occupancy_checkpoint_semantic_sha256"
    result["prediction_semantics"] = (
        "scalar_baseline_score_repeated_for_common_calibration"
    )
    result["semantic_digest"] = prediction_table_semantic_digest(result)
    return result


def _with_toy_dataset_digest(table: dict, digest: str) -> dict:
    result = copy.deepcopy(table)
    result["toy_dataset_manifest_digest"] = digest
    result["cohort_digest_sha256"] = _test_cohort_digest(result)
    result["semantic_digest"] = prediction_table_semantic_digest(result)
    return result


def _with_grouped_calibration(
    artifact: dict,
    table: dict,
    *,
    alpha: float,
    min_group_size: int = 3,
) -> dict:
    result = copy.deepcopy(artifact)
    result["grouped"] = fit_grouped_calibration(
        table["rows"],
        alpha=alpha,
        prediction_key=result["prediction_key"],
        dimensions=GROUP_DIMENSIONS,
        min_group_size=min_group_size,
    )
    result["semantic_digest"] = calibration_artifact_semantic_digest(result)
    return result


def test_semantic_digests_ignore_all_runtime_metadata_names() -> None:
    table = _toy_table("calibration", [_row("c0", "calibration")])
    table_digest = table["semantic_digest"]
    table.update(
        {
            "generated_at": "2026-01-01T00:00:00Z",
            "generated_at_utc": "2026-01-02T00:00:00Z",
            "job_id": "old-job-name",
            "slurm_job_id": "123456",
        }
    )
    assert prediction_table_semantic_digest(table) == table_digest

    artifact = fit_calibration_artifact(
        _toy_table("calibration", [_row("c0", "calibration")]),
        alpha=0.5,
        prediction_key="q90",
    )
    artifact_digest = artifact["semantic_digest"]
    artifact.update(
        {
            "generated_at": "2026-01-01T00:00:00Z",
            "generated_at_utc": "2026-01-02T00:00:00Z",
            "job_id": "old-job-name",
            "slurm_job_id": "123456",
        }
    )
    assert calibration_artifact_semantic_digest(artifact) == artifact_digest


def test_prediction_table_validates_schema_rows_and_semantic_digest() -> None:
    table = _toy_table("calibration", [_row("c0", "calibration")])

    validated = validate_prediction_table(
        table, expected_mode="toy", expected_split="calibration"
    )

    assert validated["semantic_digest"] == table["semantic_digest"]


def test_prediction_table_accepts_finite_negative_signed_collision_clearance() -> None:
    table = _toy_table(
        "calibration",
        [_row("c0", "calibration", collision_label=1, min_clearance=-0.125)],
    )

    validated = validate_prediction_table(table, expected_mode="toy")

    assert validated["rows"][0]["min_clearance"] == pytest.approx(-0.125)


def test_prediction_table_accepts_v2_occupancy_baseline_checkpoint() -> None:
    table = _toy_table("calibration", [_row("c0", "calibration")])
    table["method_id"] = "occupancy_b3_hand_aggregation"
    table["checkpoint_layout_version"] = "occupancy_baseline_checkpoint_v2"
    table["checkpoint_digest"] = "e" * 64
    table["checkpoint_digest_kind"] = "occupancy_checkpoint_semantic_sha256"
    table["prediction_semantics"] = (
        "scalar_baseline_score_repeated_for_common_calibration"
    )
    table["semantic_digest"] = prediction_table_semantic_digest(table)

    validated = validate_prediction_table(table, expected_mode="toy")

    assert validated["checkpoint_layout_version"] == (
        "occupancy_baseline_checkpoint_v2"
    )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("checkpoint_digest", "not-a-sha256", "checkpoint_digest"),
        ("toy_dataset_manifest_digest", "not-a-blake2b128", "toy_dataset"),
        ("config_digest_sha256", "short", "config_digest"),
        ("cohort_digest_sha256", "short", "cohort_digest"),
        ("seed", -1, "seed"),
        ("channel_spec", {"flat": ["drifted"]}, "channel_spec"),
    ],
)
def test_prediction_table_requires_canonical_direct_provenance(
    field: str, bad_value: object, message: str
) -> None:
    table = _toy_table("calibration", [_row("c0", "calibration")])
    table[field] = bad_value
    table["semantic_digest"] = prediction_table_semantic_digest(table)

    with pytest.raises(CalibrationContractError, match=message):
        validate_prediction_table(table, expected_mode="toy")


def test_prediction_table_cohort_digest_binds_labels_groups_and_identities() -> None:
    table = _toy_table("calibration", [_row("c0", "calibration")])
    table["rows"][0]["age_s"] = 2.5
    table["semantic_digest"] = prediction_table_semantic_digest(table)

    with pytest.raises(CalibrationContractError, match="cohort digest"):
        validate_prediction_table(table, expected_mode="toy")


def test_prediction_table_public_api_fails_closed_on_production() -> None:
    table = _toy_table("calibration", [_row("c0", "calibration")])
    table["mode"] = "production"
    table.pop("toy_dataset_manifest_digest")
    table.update(
        {
            "g1_split_manifest_digest": "1" * 64,
            "risk_dataset_manifest_digest": "2" * 64,
            "dynamic_objects_config_digest": "3" * 64,
            "target_type_policy_digest": "4" * 64,
        }
    )
    table["semantic_digest"] = prediction_table_semantic_digest(table)

    with pytest.raises(CalibrationContractError, match="production.*fail-closed"):
        validate_prediction_table(table)


@pytest.mark.parametrize(
    ("layout", "field", "bad_value", "message"),
    [
        (
            "risk_model_checkpoint_v2",
            "checkpoint_digest_kind",
            None,
            "risk_checkpoint_semantic_sha256",
        ),
        (
            "occupancy_baseline_checkpoint_v2",
            "checkpoint_digest_kind",
            "model_state_semantic_sha256",
            "occupancy_checkpoint_semantic_sha256",
        ),
        (
            "occupancy_baseline_checkpoint_v2",
            "prediction_semantics",
            "quantile",
            "scalar_baseline_score_repeated_for_common_calibration",
        ),
    ],
)
def test_prediction_table_requires_layout_specific_semantic_provenance(
    layout: str,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    table = _toy_table("calibration", [_row("c0", "calibration")])
    if layout == "occupancy_baseline_checkpoint_v2":
        table = _as_occupancy_table(table)
    if bad_value is None:
        table.pop(field)
    else:
        table[field] = bad_value
    table["semantic_digest"] = prediction_table_semantic_digest(table)

    with pytest.raises(CalibrationContractError, match=message):
        validate_prediction_table(table, expected_mode="toy")


def test_prediction_table_rejects_tampering_and_cross_mode_provenance() -> None:
    table = _toy_table("calibration", [_row("c0", "calibration")])
    table["rows"][0]["risk_severity"] = 0.1
    with pytest.raises(CalibrationContractError, match="(?:cohort|semantic) digest"):
        validate_prediction_table(table, expected_mode="toy")

    production = copy.deepcopy(table)
    production["mode"] = "production"
    production.pop("toy_dataset_manifest_digest")
    production.update(
        {
            "g1_split_manifest_digest": "g1",
            "risk_dataset_manifest_digest": "risk-v2",
            "dynamic_objects_config_digest": "dynamic",
            "target_type_policy_digest": "target-policy",
        }
    )
    production["semantic_digest"] = prediction_table_semantic_digest(production)
    with pytest.raises(CalibrationContractError, match="mode"):
        validate_prediction_table(production, expected_mode="toy")


def test_calibration_fit_rejects_non_calibration_rows() -> None:
    table = _toy_table("test", [_row("t0", "test")])
    with pytest.raises(CalibrationContractError, match="calibration split"):
        fit_calibration_artifact(table, alpha=0.1, prediction_key="q90")


def test_test_label_perturbation_cannot_change_fitted_correction() -> None:
    calibration = _toy_table(
        "calibration",
        [
            _row("c0", "calibration", risk_severity=0.8, q90=0.5),
            _row("c1", "calibration", risk_severity=0.6, q90=0.5),
        ],
    )
    test_a = [_row("t0", "test", risk_severity=0.0)]
    test_b = [_row("t0", "test", risk_severity=1.0)]

    artifact = fit_calibration_artifact(
        calibration, alpha=0.5, prediction_key="q90"
    )

    # Test labels are accepted only by the application/evaluation side, not fit.
    assert test_a != test_b
    assert artifact["global"]["correction"] == pytest.approx(0.3)
    assert artifact["fit_split"] == "calibration"


@pytest.mark.parametrize(
    "identity_field",
    [
        "sample_id",
        "pair_group_id",
        "recording_id",
        "session_id",
        "source_object_id",
        "snippet_id",
        "base_state_id",
        "seed_namespace",
    ],
)
def test_calibration_test_identity_overlap_is_rejected(identity_field: str) -> None:
    calibration = [_row("c0", "calibration")]
    test = [_row("t0", "test")]
    test[0][identity_field] = calibration[0][identity_field]

    with pytest.raises(CalibrationContractError, match=identity_field):
        assert_calibration_test_isolation(calibration, test)


def test_calibration_artifact_rejects_old_version_and_tampering() -> None:
    table = _toy_table(
        "calibration", [_row("c0", "calibration", risk_severity=0.8, q90=0.5)]
    )
    artifact = fit_calibration_artifact(table, alpha=0.5, prediction_key="q90")
    assert artifact["calibration_artifact_layout_version"] == (
        CALIBRATION_ARTIFACT_LAYOUT_VERSION
    )

    old = copy.deepcopy(artifact)
    old["calibration_artifact_layout_version"] = "risk_calibration_v1"
    with pytest.raises(CalibrationContractError, match="layout version"):
        validate_calibration_artifact(old, expected_mode="toy")

    tampered = copy.deepcopy(artifact)
    tampered["global"]["correction"] = 0.0
    with pytest.raises(
        CalibrationContractError, match="(?:semantic digest|residual bounds)"
    ):
        validate_calibration_artifact(tampered, expected_mode="toy")


@pytest.mark.parametrize("checkpoint_layout", ["risk", "occupancy"])
def test_calibration_artifact_roundtrip_binds_layout_specific_semantics(
    checkpoint_layout: str,
) -> None:
    table = _toy_table("calibration", [_row("c0", "calibration")])
    if checkpoint_layout == "occupancy":
        table = _as_occupancy_table(table)

    artifact = fit_calibration_artifact(table, alpha=0.5, prediction_key="q90")
    validated = validate_calibration_artifact(artifact, expected_mode="toy")

    assert validated["checkpoint_digest_kind"] == table["checkpoint_digest_kind"]
    assert validated["seed"] == table["seed"]
    assert validated["channel_spec"] == table["channel_spec"]
    assert validated["config_digest_sha256"] == table["config_digest_sha256"]
    assert validated["calibration_cohort_digest_sha256"] == table[
        "cohort_digest_sha256"
    ]
    if checkpoint_layout == "occupancy":
        assert validated["prediction_semantics"] == table["prediction_semantics"]
    else:
        assert "prediction_semantics" not in validated


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("sample_id", [], "sample_id"),
        ("sample_id", ["c0", "c0"], "sorted unique"),
        ("recording_id", ["recording-z", "recording-a"], "sorted unique"),
    ],
)
def test_calibration_artifact_rejects_incomplete_or_noncanonical_identities(
    field: str, replacement: list[str], message: str
) -> None:
    table = _toy_table(
        "calibration", [_row("c0", "calibration"), _row("c1", "calibration")]
    )
    artifact = fit_calibration_artifact(table, alpha=0.5, prediction_key="q90")
    artifact["fitted_identities"][field] = replacement
    artifact["semantic_digest"] = calibration_artifact_semantic_digest(artifact)

    with pytest.raises(CalibrationContractError, match=message):
        validate_calibration_artifact(artifact, expected_mode="toy")


def test_calibration_artifact_rejects_correction_outside_residual_evidence() -> None:
    table = _toy_table(
        "calibration", [_row("c0", "calibration", risk_severity=0.8, q90=0.5)]
    )
    artifact = fit_calibration_artifact(table, alpha=0.5, prediction_key="q90")
    artifact["global"]["correction"] = 0.9
    artifact["semantic_digest"] = calibration_artifact_semantic_digest(artifact)

    with pytest.raises(CalibrationContractError, match="residual bounds"):
        validate_calibration_artifact(artifact, expected_mode="toy")


def test_calibration_artifact_public_api_fails_closed_on_production() -> None:
    table = _toy_table("calibration", [_row("c0", "calibration")])
    artifact = fit_calibration_artifact(table, alpha=0.5, prediction_key="q90")
    artifact["mode"] = "production"
    artifact.pop("toy_dataset_manifest_digest")
    artifact.update(
        {
            "g1_split_manifest_digest": "1" * 64,
            "risk_dataset_manifest_digest": "2" * 64,
            "dynamic_objects_config_digest": "3" * 64,
            "target_type_policy_digest": "4" * 64,
        }
    )
    artifact["semantic_digest"] = calibration_artifact_semantic_digest(artifact)

    with pytest.raises(CalibrationContractError, match="production.*fail-closed"):
        validate_calibration_artifact(artifact)


@pytest.mark.parametrize(
    ("checkpoint_layout", "mutation", "message"),
    [
        ("risk", "wrong_digest_kind", "risk_checkpoint_semantic_sha256"),
        ("risk", "occupancy_semantics", "must not contain prediction_semantics"),
        ("occupancy", "wrong_digest_kind", "occupancy_checkpoint_semantic_sha256"),
        (
            "occupancy",
            "missing_semantics",
            "scalar_baseline_score_repeated_for_common_calibration",
        ),
    ],
)
def test_calibration_artifact_rejects_self_digested_semantic_provenance_tampering(
    checkpoint_layout: str,
    mutation: str,
    message: str,
) -> None:
    table = _toy_table("calibration", [_row("c0", "calibration")])
    if checkpoint_layout == "occupancy":
        table = _as_occupancy_table(table)
    artifact = fit_calibration_artifact(table, alpha=0.5, prediction_key="q90")
    if mutation == "wrong_digest_kind":
        artifact["checkpoint_digest_kind"] = (
            "occupancy_checkpoint_semantic_sha256"
            if checkpoint_layout == "risk"
            else "risk_checkpoint_semantic_sha256"
        )
    elif mutation == "occupancy_semantics":
        artifact["prediction_semantics"] = (
            "scalar_baseline_score_repeated_for_common_calibration"
        )
    else:
        artifact.pop("prediction_semantics", None)
    artifact["semantic_digest"] = calibration_artifact_semantic_digest(artifact)

    with pytest.raises(CalibrationContractError, match=message):
        validate_calibration_artifact(artifact, expected_mode="toy")


def test_calibration_artifact_rejects_invalid_grouped_payload_after_redigest() -> None:
    table = _toy_table(
        "calibration",
        [
            _row("c0", "calibration", risk_severity=0.8, q90=0.5),
            _row("c1", "calibration", risk_severity=0.7, q90=0.5),
            _row("c2", "calibration", risk_severity=0.6, q90=0.5),
            _row("c3", "calibration", risk_severity=0.5, q90=0.5),
        ],
    )
    artifact = fit_calibration_artifact(table, alpha=0.2, prediction_key="q90")
    artifact = _with_grouped_calibration(
        artifact,
        table,
        alpha=0.2,
        min_group_size=5,
    )
    blind_group = artifact["grouped"]["dimensions"]["blind_type"]["corner"]
    blind_group["count"] = 3
    blind_group["fallback_reason"] = "group_count_below_minimum:3<5"
    artifact["semantic_digest"] = calibration_artifact_semantic_digest(artifact)

    with pytest.raises(CalibrationContractError, match="grouped.*counts must sum"):
        validate_calibration_artifact(artifact, expected_mode="toy")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("alpha",), "alpha"),
        (("prediction_key",), "prediction_key"),
        (("global", "rank_one_based"), "rank"),
    ],
)
def test_calibration_artifact_rejects_semantically_invalid_self_digested_values(
    mutation: tuple[str, ...], message: str
) -> None:
    table = _toy_table(
        "calibration", [_row("c0", "calibration", risk_severity=0.8, q90=0.5)]
    )
    artifact = fit_calibration_artifact(table, alpha=0.5, prediction_key="q90")
    if mutation == ("alpha",):
        artifact["alpha"] = 0.0
    elif mutation == ("prediction_key",):
        artifact["prediction_key"] = "q99"
    else:
        artifact["global"]["rank_one_based"] = 0
    artifact["semantic_digest"] = calibration_artifact_semantic_digest(artifact)

    with pytest.raises(CalibrationContractError, match=message):
        validate_calibration_artifact(artifact, expected_mode="toy")


def test_calibration_cli_accepts_toy_table_and_production_fails_closed(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "calibration-table.json"
    table_path.write_text(
        json.dumps(
            _toy_table(
                "calibration",
                [_row("c0", "calibration"), _row("c1", "calibration")],
            )
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "calibration-output"
    script = Path(__file__).parents[1] / "scripts" / "07_calibrate_risk.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "toy",
            "--prediction-table",
            str(table_path),
            "--split",
            "calibration",
            "--output-dir",
            str(output_dir),
            "--min-group-size",
            "1",
            "--alpha",
            "0.5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "calibration.json").is_file()
    assert (output_dir / "manifest.json").is_file()

    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "production",
            "--prediction-table",
            str(table_path),
            "--split",
            "calibration",
            "--output-dir",
            str(tmp_path / "production-output"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "production" in rejected.stderr.lower()
    assert "v2" in rejected.stderr.lower()


def test_offline_eval_cli_applies_calibration_and_marks_toy_gate_unresolved(
    tmp_path: Path,
) -> None:
    calibration_table = _with_toy_dataset_digest(
        _toy_table(
            "calibration",
            [
                _row("c0", "calibration", risk_severity=0.8, q90=0.5),
                _row("c1", "calibration", risk_severity=0.6, q90=0.5),
            ],
        ),
        "b" * 32,
    )
    artifact = fit_calibration_artifact(
        calibration_table, alpha=0.5, prediction_key="q90"
    )
    artifact = _with_grouped_calibration(
        artifact,
        calibration_table,
        alpha=0.5,
    )
    artifact_path = tmp_path / "calibration.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    test_table = _with_toy_dataset_digest(
        _toy_table(
            "test",
            [
                _row(
                    "t0", "test", collision_label=1, risk_severity=0.9, q90=0.6
                ),
                _row(
                    "t1",
                    "test",
                    collision_label=0,
                    risk_severity=0.0,
                    p_collision=0.1,
                    q50=0.05,
                    q80=0.1,
                    q90=0.2,
                    q95=0.4,
                    critical_object_id=None,
                    min_clearance=22.627,
                    event_type="empty",
                ),
            ],
        ),
        "c" * 32,
    )
    table_path = tmp_path / "test-table.json"
    table_path.write_text(json.dumps(test_table), encoding="utf-8")
    output_dir = tmp_path / "evaluation-output"
    script = Path(__file__).parents[1] / "scripts" / "10_eval_offline.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "toy",
            "--task",
            "risk",
            "--prediction-table",
            str(table_path),
            "--calibration-artifact",
            str(artifact_path),
            "--split",
            "test",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert metrics["sample_count"] == 2
    assert metrics["classification"]["auroc"]["value"] == pytest.approx(1.0)
    application = metrics["calibration_application"]
    assert application["primary_overall"] == "global_split_conformal"
    assert set(application["grouped_by_dimension"]) == set(GROUP_DIMENSIONS)
    for dimension in GROUP_DIMENSIONS:
        grouped = application["grouped_by_dimension"][dimension]
        assert grouped["coverage"]["count"] == 2
        assert grouped["tightness"]["mean_upper_bound"]["count"] == 2
        assert grouped["fallback"]["sample_count"] == 2
        assert grouped["fallback"]["fallback_count"] == 2
    assert metrics["definitions"]["false_safe_scope"] == (
        "raw_collision_head_probability_not_conformal"
    )
    assert manifest["false_safe_scope"] == (
        "raw_collision_head_probability_not_conformal"
    )
    assert manifest["scientific_gate_status"] == "not_evaluated_real_data"
    assert manifest["calibration_test_identity_overlap"] == {
        field: 0
        for field in (
            "sample_id",
            "pair_group_id",
            "recording_id",
            "session_id",
            "source_object_id",
            "snippet_id",
            "base_state_id",
            "seed_namespace",
        )
    }

    mismatched_artifact = copy.deepcopy(artifact)
    mismatched_artifact["checkpoint_layout_version"] = (
        "occupancy_baseline_checkpoint_v2"
    )
    mismatched_artifact["checkpoint_digest_kind"] = (
        "occupancy_checkpoint_semantic_sha256"
    )
    mismatched_artifact["prediction_semantics"] = (
        "scalar_baseline_score_repeated_for_common_calibration"
    )
    mismatched_artifact["semantic_digest"] = calibration_artifact_semantic_digest(
        mismatched_artifact
    )
    mismatched_artifact_path = tmp_path / "mismatched-calibration.json"
    mismatched_artifact_path.write_text(
        json.dumps(mismatched_artifact), encoding="utf-8"
    )
    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "toy",
            "--task",
            "risk",
            "--prediction-table",
            str(table_path),
            "--calibration-artifact",
            str(mismatched_artifact_path),
            "--split",
            "test",
            "--output-dir",
            str(tmp_path / "mismatched-evaluation-output"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "checkpoint_layout_version" in rejected.stderr


def test_offline_eval_production_mode_fails_closed(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "10_eval_offline.py"
    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "production",
            "--task",
            "risk",
            "--prediction-table",
            str(tmp_path / "missing-table.json"),
            "--calibration-artifact",
            str(tmp_path / "missing-calibration.json"),
            "--split",
            "test",
            "--output-dir",
            str(tmp_path / "production-output"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "production" in rejected.stderr.lower()
    assert "v2" in rejected.stderr.lower()


def test_offline_eval_compares_main_and_occupancy_under_same_protocol(
    tmp_path: Path,
) -> None:
    calibration_rows = [
        _row("c0", "calibration", risk_severity=0.8, q90=0.5),
        _row("c1", "calibration", risk_severity=0.6, q90=0.5),
    ]
    main_calibration_table = _toy_table("calibration", calibration_rows)
    occupancy_calibration_table = _as_occupancy_table(main_calibration_table)
    main_artifact = fit_calibration_artifact(
        main_calibration_table, alpha=0.5, prediction_key="q90"
    )
    occupancy_artifact = fit_calibration_artifact(
        occupancy_calibration_table, alpha=0.5, prediction_key="q90"
    )
    main_artifact = _with_grouped_calibration(
        main_artifact,
        main_calibration_table,
        alpha=0.5,
    )
    occupancy_artifact = _with_grouped_calibration(
        occupancy_artifact,
        occupancy_calibration_table,
        alpha=0.5,
    )
    main_artifact_path = tmp_path / "main-calibration.json"
    occupancy_artifact_path = tmp_path / "occupancy-calibration.json"
    main_artifact_path.write_text(json.dumps(main_artifact), encoding="utf-8")
    occupancy_artifact_path.write_text(
        json.dumps(occupancy_artifact), encoding="utf-8"
    )

    main_rows = [
        _row("t0", "test", collision_label=1, p_collision=0.8),
        _row(
            "t1",
            "test",
            collision_label=0,
            risk_severity=0.0,
            p_collision=0.1,
            event_type="temporal_safe",
        ),
    ]
    occupancy_rows = copy.deepcopy(main_rows)
    occupancy_rows[0]["p_collision"] = 0.2
    occupancy_rows[1]["p_collision"] = 0.6
    main_test_table = _toy_table("test", main_rows)
    occupancy_test_table = _as_occupancy_table(
        _toy_table("test", occupancy_rows)
    )
    main_table_path = tmp_path / "main-test.json"
    occupancy_table_path = tmp_path / "occupancy-test.json"
    main_table_path.write_text(json.dumps(main_test_table), encoding="utf-8")
    occupancy_table_path.write_text(
        json.dumps(occupancy_test_table), encoding="utf-8"
    )
    output_dir = tmp_path / "comparison-output"
    script = Path(__file__).parents[1] / "scripts" / "10_eval_offline.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "toy",
            "--task",
            "risk",
            "--prediction-table",
            str(main_table_path),
            "--calibration-artifact",
            str(main_artifact_path),
            "--baseline-prediction-table",
            str(occupancy_table_path),
            "--baseline-calibration-artifact",
            str(occupancy_artifact_path),
            "--split",
            "test",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["comparison"]["sample_identity_match"] is True
    assert metrics["same_calibration_protocol"] is True
    assert set(
        metrics["occupancy_baseline"]["calibration_application"][
            "grouped_by_dimension"
        ]
    ) == set(GROUP_DIMENSIONS)
    assert metrics["comparison"]["hard_negative_better_count"] >= 1

    mismatched_test_cohort = copy.deepcopy(occupancy_test_table)
    mismatched_test_cohort["rows"][0]["age_s"] = 2.5
    mismatched_test_cohort["cohort_digest_sha256"] = _test_cohort_digest(
        mismatched_test_cohort
    )
    mismatched_test_cohort["semantic_digest"] = prediction_table_semantic_digest(
        mismatched_test_cohort
    )
    mismatched_test_path = tmp_path / "occupancy-test-mismatched-cohort.json"
    mismatched_test_path.write_text(
        json.dumps(mismatched_test_cohort), encoding="utf-8"
    )
    rejected_test_cohort = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "toy",
            "--task",
            "risk",
            "--prediction-table",
            str(main_table_path),
            "--calibration-artifact",
            str(main_artifact_path),
            "--baseline-prediction-table",
            str(mismatched_test_path),
            "--baseline-calibration-artifact",
            str(occupancy_artifact_path),
            "--split",
            "test",
            "--output-dir",
            str(tmp_path / "mismatched-test-cohort-output"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected_test_cohort.returncode != 0
    assert "test cohort digest mismatch" in rejected_test_cohort.stderr

    altered_calibration_rows = copy.deepcopy(calibration_rows)
    altered_calibration_rows[0]["age_s"] = 2.5
    altered_calibration_table = _as_occupancy_table(
        _toy_table("calibration", altered_calibration_rows)
    )
    altered_calibration_artifact = fit_calibration_artifact(
        altered_calibration_table, alpha=0.5, prediction_key="q90"
    )
    altered_calibration_artifact = _with_grouped_calibration(
        altered_calibration_artifact,
        altered_calibration_table,
        alpha=0.5,
    )
    altered_calibration_path = tmp_path / "occupancy-calibration-mismatched-cohort.json"
    altered_calibration_path.write_text(
        json.dumps(altered_calibration_artifact), encoding="utf-8"
    )
    rejected_calibration_cohort = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "toy",
            "--task",
            "risk",
            "--prediction-table",
            str(main_table_path),
            "--calibration-artifact",
            str(main_artifact_path),
            "--baseline-prediction-table",
            str(occupancy_table_path),
            "--baseline-calibration-artifact",
            str(altered_calibration_path),
            "--split",
            "test",
            "--output-dir",
            str(tmp_path / "mismatched-calibration-cohort-output"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected_calibration_cohort.returncode != 0
    assert "calibration protocol mismatch" in rejected_calibration_cohort.stderr
    assert "calibration_cohort_digest_sha256" in rejected_calibration_cohort.stderr

    occupancy_only = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "toy",
            "--task",
            "risk",
            "--prediction-table",
            str(occupancy_table_path),
            "--calibration-artifact",
            str(occupancy_artifact_path),
            "--split",
            "test",
            "--output-dir",
            str(tmp_path / "occupancy-only-output"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert occupancy_only.returncode == 0, occupancy_only.stderr

    invalid_comparison = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "toy",
            "--task",
            "risk",
            "--prediction-table",
            str(occupancy_table_path),
            "--calibration-artifact",
            str(occupancy_artifact_path),
            "--baseline-prediction-table",
            str(occupancy_table_path),
            "--baseline-calibration-artifact",
            str(occupancy_artifact_path),
            "--split",
            "test",
            "--output-dir",
            str(tmp_path / "invalid-comparison-output"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid_comparison.returncode != 0
    assert "main prediction table" in invalid_comparison.stderr
    assert "risk_model_checkpoint_v2" in invalid_comparison.stderr
