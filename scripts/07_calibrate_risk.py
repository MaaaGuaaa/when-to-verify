#!/usr/bin/env python3
"""Fit a toy split-conformal artifact from a validated prediction table."""

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
    calibration_artifact_semantic_digest,
    fit_calibration_artifact,
    validate_calibration_artifact,
    validate_prediction_table,
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
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("toy", "production"), required=True)
    parser.add_argument("--prediction-table", type=Path, required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--prediction-key", default="q90")
    parser.add_argument("--min-group-size", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "production":
        raise SystemExit(
            "production mode is fail-closed until the risk dataset v2 manifest "
            "and required provenance are published"
        )
    if args.split != "calibration":
        raise SystemExit("calibration fitting requires --split calibration")
    table = validate_prediction_table(
        _read_json(args.prediction_table),
        expected_mode="toy",
        expected_split="calibration",
    )
    artifact = fit_calibration_artifact(
        table,
        alpha=args.alpha,
        prediction_key=args.prediction_key,
    )
    artifact["grouped"] = fit_grouped_calibration(
        table["rows"],
        alpha=args.alpha,
        prediction_key=args.prediction_key,
        dimensions=GROUP_DIMENSIONS,
        min_group_size=args.min_group_size,
    )
    artifact["semantic_digest"] = calibration_artifact_semantic_digest(artifact)
    validate_calibration_artifact(artifact, expected_mode="toy")

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
        "artifact_kind": "sop10_toy_calibration",
        "mode": "toy",
        "schema_version": "3.0.0",
        "fit_split": "calibration",
        "method_id": table["method_id"],
        "toy_dataset_manifest_digest": table["toy_dataset_manifest_digest"],
        "seed": table["seed"],
        "channel_spec": table["channel_spec"],
        "config_digest_sha256": table["config_digest_sha256"],
        "calibration_cohort_digest_sha256": table["cohort_digest_sha256"],
        "prediction_table_semantic_digest": table["semantic_digest"],
        "calibration_semantic_digest": artifact["semantic_digest"],
        "calibration_file_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
        "calibration_count": len(table["rows"]),
        "scientific_gate_status": "not_evaluated_real_data",
    }
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
