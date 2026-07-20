#!/usr/bin/env python3
"""Evaluate a calibrated risk prediction table without test-time fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.calibration.split_conformal import (  # noqa: E402
    OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
    RISK_CHECKPOINT_LAYOUT_VERSION,
    apply_split_conformal,
    assert_calibration_artifact_test_isolation,
    calibration_artifact_semantic_digest,
    validate_calibration_artifact,
    validate_prediction_table,
)
from src.calibration.grouped_calibration import (  # noqa: E402
    GROUP_DIMENSIONS,
    apply_grouped_calibration,
)
from src.evaluation.risk_metrics import (  # noqa: E402
    compare_risk_methods,
    evaluate_risk_rows,
    quantile_coverage,
    upper_bound_tightness,
)
from src.datasets.risk_dataset_seal import (  # noqa: E402
    RISK_DATASET_FAMILY_LAYOUT_VERSION,
    LoadedRiskDatasetFamily,
    load_risk_dataset_family,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _publish(output_dir: Path, files: dict[str, dict[str, Any]]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        for name, value in files.items():
            _write_json(staging / name, value)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _json_sha256(value: dict[str, Any]) -> str:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _calibrated_rows(
    table: dict[str, Any], artifact: dict[str, Any]
) -> list[dict[str, Any]]:
    prediction_key = artifact["prediction_key"]
    predictions = np.asarray(
        [row[prediction_key] for row in table["rows"]], dtype=np.float64
    )
    calibrated = apply_split_conformal(
        predictions, correction=float(artifact["global"]["correction"])
    )
    rows: list[dict[str, Any]] = []
    for source, upper in zip(table["rows"], calibrated):
        row = dict(source)
        row["calibrated_upper"] = float(upper)
        rows.append(row)
    return rows


def _calibration_application_report(
    table: dict[str, Any], artifact: dict[str, Any]
) -> dict[str, Any]:
    grouped_artifact = artifact.get("grouped")
    if not isinstance(grouped_artifact, dict):
        raise ValueError("offline evaluation requires a validated grouped artifact")
    if tuple(grouped_artifact.get("group_dimensions", ())) != GROUP_DIMENSIONS:
        raise ValueError(
            "offline evaluation requires every frozen grouped calibration dimension"
        )
    severity = np.asarray(
        [row["risk_severity"] for row in table["rows"]], dtype=np.float64
    )
    dimension_reports: dict[str, Any] = {}
    for dimension in GROUP_DIMENSIONS:
        calibrated, decisions = apply_grouped_calibration(
            table["rows"],
            grouped_artifact,
            prediction_key=artifact["prediction_key"],
            dimension=dimension,
        )
        fallback_count = sum(bool(decision["fallback"]) for decision in decisions)
        reason_counts: dict[str, int] = {}
        group_counts: dict[str, int] = {}
        for decision in decisions:
            group = str(decision["group"])
            group_counts[group] = group_counts.get(group, 0) + 1
            if decision["fallback"]:
                reason = str(decision["fallback_reason"])
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        sample_count = len(decisions)
        dimension_reports[dimension] = {
            "coverage": quantile_coverage(severity, calibrated),
            "tightness": upper_bound_tightness(severity, calibrated),
            "fallback": {
                "sample_count": sample_count,
                "fallback_count": fallback_count,
                "fallback_fraction": (
                    float(fallback_count / sample_count) if sample_count else None
                ),
                "reason_counts": dict(sorted(reason_counts.items())),
                "group_counts": dict(sorted(group_counts.items())),
            },
        }
    return {
        "primary_overall": "global_split_conformal",
        "grouped_policy": "one_dimension_at_a_time_diagnostic",
        "grouped_by_dimension": dimension_reports,
    }


def _assert_same_calibration_protocol(
    main_artifact: dict[str, Any], baseline_artifact: dict[str, Any]
) -> None:
    for field in (
        "alpha",
        "prediction_key",
        "fit_split",
        "fitted_identities",
        "toy_dataset_manifest_digest",
        "risk_dataset_family_layout_version",
        "risk_dataset_family_digest",
        "risk_dataset_manifest_digest",
        "calibration_cohort_digest_sha256",
        "seed",
        "channel_spec",
        "config_digest_sha256",
    ):
        if main_artifact.get(field) != baseline_artifact.get(field):
            raise ValueError(
                f"main and occupancy baseline calibration protocol mismatch: {field}"
            )
    main_grouped = main_artifact.get("grouped")
    baseline_grouped = baseline_artifact.get("grouped")
    if (main_grouped is None) != (baseline_grouped is None):
        raise ValueError(
            "main and occupancy baseline must both use grouped calibration or neither"
        )
    if main_grouped is not None:
        for field in (
            "alpha",
            "prediction_key",
            "min_group_size",
            "group_dimensions",
            "continuous_group_bins",
            "combination_policy",
        ):
            if main_grouped.get(field) != baseline_grouped.get(field):
                raise ValueError(
                    "main and occupancy baseline grouped calibration protocol "
                    f"mismatch: {field}"
                )


def _expected_artifact_provenance(table: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "method_id": table["method_id"],
        "checkpoint_layout_version": table["checkpoint_layout_version"],
        "checkpoint_digest": table["checkpoint_digest"],
        "checkpoint_digest_kind": table["checkpoint_digest_kind"],
        "seed": table["seed"],
        "channel_spec": table["channel_spec"],
        "config_digest_sha256": table["config_digest_sha256"],
    }
    if table["checkpoint_layout_version"] == OCCUPANCY_CHECKPOINT_LAYOUT_VERSION:
        expected["prediction_semantics"] = table["prediction_semantics"]
    if table["mode"] == "production":
        expected["risk_dataset_family_layout_version"] = table[
            "risk_dataset_family_layout_version"
        ]
        expected["risk_dataset_family_digest"] = table[
            "risk_dataset_family_digest"
        ]
    return expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("toy", "production"), required=True)
    parser.add_argument("--task", choices=("risk",), default="risk")
    parser.add_argument("--prediction-table", type=Path, required=True)
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    parser.add_argument("--dataset-family-root", type=Path)
    parser.add_argument("--baseline-prediction-table", type=Path)
    parser.add_argument("--baseline-calibration-artifact", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_family: LoadedRiskDatasetFamily | None = None
    if args.mode == "production" and args.dataset_family_root is None:
        raise SystemExit(
            "production risk_dataset_v2 evaluation is fail-closed without "
            "--dataset-family-root"
        )
    if args.mode == "toy" and args.dataset_family_root is not None:
        raise SystemExit("toy evaluation must not use --dataset-family-root")
    if args.mode == "production":
        dataset_family = load_risk_dataset_family(args.dataset_family_root)
    if args.split != "test":
        raise SystemExit("offline evaluation requires --split test")
    if (args.baseline_prediction_table is None) != (
        args.baseline_calibration_artifact is None
    ):
        raise SystemExit(
            "baseline comparison requires both --baseline-prediction-table and "
            "--baseline-calibration-artifact"
        )
    table = validate_prediction_table(
        _read_json(args.prediction_table),
        expected_mode=args.mode,
        expected_split="test",
        dataset_family=dataset_family,
    )
    if (
        args.baseline_prediction_table is not None
        and table["checkpoint_layout_version"] != RISK_CHECKPOINT_LAYOUT_VERSION
    ):
        raise ValueError(
            "main prediction table in a baseline comparison must bind a "
            f"{RISK_CHECKPOINT_LAYOUT_VERSION} checkpoint"
        )
    artifact = validate_calibration_artifact(
        _read_json(args.calibration_artifact),
        expected_mode=args.mode,
        expected_provenance=_expected_artifact_provenance(table),
        dataset_family=dataset_family,
    )
    isolation = assert_calibration_artifact_test_isolation(
        artifact,
        table["rows"],
        dataset_family=dataset_family,
    )
    rows = _calibrated_rows(table, artifact)
    metrics = evaluate_risk_rows(rows)
    metrics["calibration_application"] = _calibration_application_report(
        table, artifact
    )
    metrics["mode"] = args.mode
    metrics["method_id"] = table["method_id"]
    metrics["scientific_gate_status"] = (
        "production_evaluation_published"
        if args.mode == "production"
        else "not_evaluated_real_data"
    )
    baseline_manifest_fields: dict[str, Any] = {}
    if args.baseline_prediction_table is not None:
        baseline_table = validate_prediction_table(
            _read_json(args.baseline_prediction_table),
            expected_mode=args.mode,
            expected_split="test",
            dataset_family=dataset_family,
        )
        if baseline_table["checkpoint_layout_version"] != (
            OCCUPANCY_CHECKPOINT_LAYOUT_VERSION
        ):
            raise ValueError(
                "baseline prediction table must bind an "
                f"{OCCUPANCY_CHECKPOINT_LAYOUT_VERSION} checkpoint"
            )
        baseline_artifact = validate_calibration_artifact(
            _read_json(args.baseline_calibration_artifact),
            expected_mode=args.mode,
            expected_provenance=_expected_artifact_provenance(baseline_table),
            dataset_family=dataset_family,
        )
        if args.mode == "toy":
            if baseline_table["toy_dataset_manifest_digest"] != table[
                "toy_dataset_manifest_digest"
            ]:
                raise ValueError(
                    "main and occupancy baseline must use the same toy dataset "
                    "manifest"
                )
        elif baseline_table["risk_dataset_family_digest"] != table[
            "risk_dataset_family_digest"
        ]:
            raise ValueError(
                "main and occupancy baseline risk_dataset_family_digest mismatch"
            )
        if baseline_table["cohort_digest_sha256"] != table[
            "cohort_digest_sha256"
        ]:
            raise ValueError(
                "main and occupancy baseline test cohort digest mismatch"
            )
        _assert_same_calibration_protocol(artifact, baseline_artifact)
        baseline_isolation = assert_calibration_artifact_test_isolation(
            baseline_artifact,
            baseline_table["rows"],
            dataset_family=dataset_family,
        )
        baseline_rows = _calibrated_rows(baseline_table, baseline_artifact)
        metrics["occupancy_baseline"] = evaluate_risk_rows(baseline_rows)
        metrics["occupancy_baseline"][
            "calibration_application"
        ] = _calibration_application_report(baseline_table, baseline_artifact)
        metrics["comparison"] = compare_risk_methods(rows, baseline_rows)
        metrics["same_calibration_protocol"] = True
        baseline_manifest_fields = {
            "baseline_method_id": baseline_table["method_id"],
            "baseline_checkpoint_layout_version": baseline_table[
                "checkpoint_layout_version"
            ],
            "baseline_checkpoint_digest": baseline_table[
                "checkpoint_digest"
            ],
            "baseline_checkpoint_digest_kind": baseline_table[
                "checkpoint_digest_kind"
            ],
            "baseline_prediction_table_semantic_digest": baseline_table[
                "semantic_digest"
            ],
            "baseline_calibration_semantic_digest": baseline_artifact[
                "semantic_digest"
            ],
            "baseline_calibration_test_identity_overlap": baseline_isolation,
            "baseline_test_cohort_digest_sha256": baseline_table[
                "cohort_digest_sha256"
            ],
            "same_calibration_protocol": True,
        }
    else:
        metrics["same_calibration_protocol"] = "not_compared"
    metrics["semantic_digest"] = calibration_artifact_semantic_digest(metrics)
    manifest = {
        "artifact_kind": (
            "sop10_production_offline_evaluation"
            if args.mode == "production"
            else "sop10_toy_offline_evaluation"
        ),
        "mode": args.mode,
        "schema_version": "3.0.0",
        "evaluated_split": "test",
        "method_id": table["method_id"],
        "checkpoint_layout_version": table["checkpoint_layout_version"],
        "checkpoint_digest": table["checkpoint_digest"],
        "checkpoint_digest_kind": table["checkpoint_digest_kind"],
        "prediction_table_semantic_digest": table["semantic_digest"],
        "test_cohort_digest_sha256": table["cohort_digest_sha256"],
        "calibration_cohort_digest_sha256": artifact[
            "calibration_cohort_digest_sha256"
        ],
        "seed": table["seed"],
        "channel_spec": table["channel_spec"],
        "config_digest_sha256": table["config_digest_sha256"],
        "calibration_semantic_digest": artifact["semantic_digest"],
        "metrics_semantic_digest": metrics["semantic_digest"],
        "metrics_file_sha256": _json_sha256(metrics),
        "sample_count": len(rows),
        "calibration_test_identity_overlap": isolation,
        "scientific_gate_status": (
            "production_evaluation_published"
            if args.mode == "production"
            else "not_evaluated_real_data"
        ),
        "primary_calibration_scope": "global_split_conformal",
        "grouped_calibration_scope": "one_dimension_at_a_time_diagnostic",
        "false_safe_scope": "raw_collision_head_probability_not_conformal",
        "false_safe_calibration_note": (
            "false-safe compares raw p_collision and is not a conformal "
            "improvement claim"
        ),
        "production_thor_session_policy": (
            "bound_by_risk_dataset_family_cross_split_audit"
            if args.mode == "production"
            else (
                "not_applied_in_toy; future production evaluation must explicitly "
                "allow-and-report the frozen THOR recording-generalization policy"
            )
        ),
        **baseline_manifest_fields,
    }
    if dataset_family is None:
        manifest.update(
            {
                "test_toy_dataset_manifest_digest": table[
                    "toy_dataset_manifest_digest"
                ],
                "calibration_toy_dataset_manifest_digest": artifact[
                    "toy_dataset_manifest_digest"
                ],
            }
        )
    else:
        manifest.update(
            {
                "risk_dataset_family_layout_version": (
                    RISK_DATASET_FAMILY_LAYOUT_VERSION
                ),
                "risk_dataset_family_digest": (
                    dataset_family.risk_dataset_family_digest
                ),
                "calibration_risk_dataset_manifest_digest": (
                    dataset_family.members["calibration"][
                        "risk_dataset_manifest_digest"
                    ]
                ),
                "calibration_sample_ids_digest_sha256": (
                    dataset_family.members["calibration"][
                        "sample_ids_digest_sha256"
                    ]
                ),
                "test_risk_dataset_manifest_digest": (
                    dataset_family.members["test"][
                        "risk_dataset_manifest_digest"
                    ]
                ),
                "test_sample_ids_digest_sha256": (
                    dataset_family.members["test"][
                        "sample_ids_digest_sha256"
                    ]
                ),
                "production_evaluation_metadata": dict(
                    dataset_family.production_evaluation_metadata
                ),
            }
        )
    _publish(args.output_dir, {"metrics.json": metrics, "manifest.json": manifest})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
