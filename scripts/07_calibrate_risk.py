#!/usr/bin/env python3
"""Fit a split-conformal artifact from a validated prediction table."""

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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.calibration.grouped_calibration import (  # noqa: E402
    GROUP_DIMENSIONS,
    fit_grouped_calibration,
)
from src.calibration.split_conformal import (  # noqa: E402
    CALIBRATION_ARTIFACT_LAYOUT_VERSION,
    calibration_artifact_semantic_digest,
    fit_calibration_artifact,
    validate_calibration_artifact,
    validate_prediction_table,
)
from src.datasets.risk_dataset_seal import (  # noqa: E402
    RISK_DATASET_FAMILY_LAYOUT_VERSION,
    LoadedRiskDatasetFamily,
    load_risk_dataset_family,
)
from src.evaluation.prediction_tables import (  # noqa: E402
    validate_prediction_protocol,
)
from src.utils.atomic_publish import atomic_rename_noreplace  # noqa: E402


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(output_dir: Path, files: dict[str, dict[str, Any]]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        for name, value in files.items():
            _write_json(staging / name, value)
        checksums = {
            name: _sha256(staging / name) for name in sorted(files)
        }
        checksums_bytes = "".join(
            f"{digest}  {name}\n" for name, digest in checksums.items()
        ).encode("ascii")
        (staging / "checksums.sha256").write_bytes(checksums_bytes)
        marker = {
            "calibration_artifact_layout_version": (
                CALIBRATION_ARTIFACT_LAYOUT_VERSION
            ),
            "calibration_semantic_digest": files["calibration.json"][
                "semantic_digest"
            ],
            "manifest_sha256": _sha256(staging / "manifest.json"),
            "checksums_sha256": hashlib.sha256(checksums_bytes).hexdigest(),
        }
        _write_json(staging / ".producer-complete", marker)
        atomic_rename_noreplace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("toy", "production"), required=True)
    parser.add_argument("--prediction-table", type=Path, required=True)
    parser.add_argument("--prediction-protocol", type=Path)
    parser.add_argument("--dataset-family-root", type=Path)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--prediction-key", default="q90")
    parser.add_argument("--min-group-size", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_family: LoadedRiskDatasetFamily | None = None
    if args.mode == "production" and args.dataset_family_root is None:
        raise SystemExit(
            "production risk_dataset_v2 calibration is fail-closed without "
            "--dataset-family-root"
        )
    if args.mode == "toy" and args.dataset_family_root is not None:
        raise SystemExit("toy calibration must not use --dataset-family-root")
    if args.mode == "production":
        dataset_family = load_risk_dataset_family(args.dataset_family_root)
    if args.split != "calibration":
        raise SystemExit("calibration fitting requires --split calibration")
    table = validate_prediction_table(
        _read_json(args.prediction_table),
        expected_mode=args.mode,
        expected_split="calibration",
        dataset_family=dataset_family,
    )
    protocol_path = getattr(args, "prediction_protocol", None)
    protocol = None
    if "prediction_protocol_digest_sha256" in table:
        if protocol_path is None:
            raise ValueError(
                "protocol-bound production table requires --prediction-protocol"
            )
        protocol = validate_prediction_protocol(_read_json(protocol_path))
        if table["prediction_protocol_digest_sha256"] != protocol[
            "protocol_digest_sha256"
        ]:
            raise ValueError("prediction table/protocol digest mismatch")
        grouped_protocol = protocol["grouped_calibration"]
        if not isinstance(grouped_protocol, dict):
            raise ValueError("prediction protocol grouped calibration is invalid")
        if float(args.alpha) != float(protocol["alpha"]):
            raise ValueError("--alpha differs from the shared prediction protocol")
        if args.prediction_key != protocol["prediction_key"]:
            raise ValueError(
                "--prediction-key differs from the shared prediction protocol"
            )
        if args.min_group_size != grouped_protocol["min_group_size"]:
            raise ValueError(
                "--min-group-size differs from the shared prediction protocol"
            )
        grouped_dimensions = tuple(grouped_protocol["group_dimensions"])
    else:
        if protocol_path is not None:
            raise ValueError(
                "--prediction-protocol requires a protocol-bound prediction table"
            )
        grouped_dimensions = GROUP_DIMENSIONS
    artifact = fit_calibration_artifact(
        table,
        alpha=args.alpha,
        prediction_key=args.prediction_key,
        dataset_family=dataset_family,
    )
    artifact["grouped"] = fit_grouped_calibration(
        table["rows"],
        alpha=args.alpha,
        prediction_key=args.prediction_key,
        dimensions=grouped_dimensions,
        min_group_size=args.min_group_size,
    )
    artifact["semantic_digest"] = calibration_artifact_semantic_digest(artifact)
    validate_calibration_artifact(
        artifact,
        expected_mode=args.mode,
        dataset_family=dataset_family,
    )

    # The manifest is semantic metadata; calibration.json has a separate full-file
    # checksum because formatting bytes are deliberately outside semantic digest.
    calibration_bytes = (
        json.dumps(
            artifact,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    manifest = {
        "artifact_kind": (
            "sop10_production_calibration"
            if args.mode == "production"
            else "sop10_toy_calibration"
        ),
        "mode": args.mode,
        "schema_version": "3.0.0",
        "fit_split": "calibration",
        "method_id": table["method_id"],
        "checkpoint_layout_version": table["checkpoint_layout_version"],
        "checkpoint_digest": table["checkpoint_digest"],
        "checkpoint_digest_kind": table["checkpoint_digest_kind"],
        "seed": table["seed"],
        "channel_spec": table["channel_spec"],
        "config_digest_sha256": table["config_digest_sha256"],
        "calibration_cohort_digest_sha256": table["cohort_digest_sha256"],
        "prediction_table_semantic_digest": table["semantic_digest"],
        "calibration_semantic_digest": artifact["semantic_digest"],
        "calibration_file_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
        "calibration_count": len(table["rows"]),
        "scientific_gate_status": (
            "production_calibration_published"
            if args.mode == "production"
            else "not_evaluated_real_data"
        ),
    }
    if dataset_family is None:
        manifest["toy_dataset_manifest_digest"] = table[
            "toy_dataset_manifest_digest"
        ]
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
                "production_evaluation_metadata": dict(
                    dataset_family.production_evaluation_metadata
                ),
            }
        )
        if protocol is not None:
            manifest.update(
                {
                    "prediction_protocol_layout_version": protocol[
                        "protocol_layout_version"
                    ],
                    "prediction_protocol_digest_sha256": protocol[
                        "protocol_digest_sha256"
                    ],
                    "evaluation_record_collection_digest_sha256": table[
                        "evaluation_record_collection_digest_sha256"
                    ],
                    "occupancy_sidecar_collection_digest_sha256": table[
                        "occupancy_sidecar_collection_digest_sha256"
                    ],
                    "cohort_binding_digest_sha256": table[
                        "cohort_binding_digest_sha256"
                    ],
                }
            )
    _publish(
        args.output_dir,
        {"calibration.json": artifact, "manifest.json": manifest},
    )
    actual_sha = _sha256(args.output_dir / "calibration.json")
    if actual_sha != manifest["calibration_file_sha256"]:
        raise RuntimeError("published calibration checksum differs from manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
