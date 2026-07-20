#!/usr/bin/env python
"""Seal an SOP07 collection and optional authenticated SOP08 sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets.risk_dataloader import RiskDataContractError  # noqa: E402
from src.datasets.risk_dataset_seal import (  # noqa: E402
    load_risk_dataset_seal,
    publish_risk_dataset_seal,
)


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError(
            "must be exactly 64 lowercase hexadecimal characters (SHA-256)"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate every immutable SOP07 shard and atomically publish "
            "a dataset-level risk_dataset_v2 seal."
        )
    )
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        help=(
            "Optional complete Task4 sidecar collection; when supplied, every "
            "risk/sidecar/marker triple is sealed for SOP08."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--split-provenance", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--expected-collection-handoff-sha256",
        type=_lower_sha256,
        required=True,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        seal_root = publish_risk_dataset_seal(
            args.output_dir,
            collection_root=args.collection_root,
            base_config_path=args.base_config,
            split_provenance_path=args.split_provenance,
            expected_split=args.split,
            expected_collection_handoff_sha256=(
                args.expected_collection_handoff_sha256
            ),
            sidecar_root=args.sidecar_root,
        )
        loaded = load_risk_dataset_seal(
            seal_root,
            collection_root=args.collection_root,
            expected_split=args.split,
            sidecar_root=args.sidecar_root,
        )
    except (RiskDataContractError, FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = {
        "risk_dataset_manifest_digest": loaded.risk_dataset_manifest_digest,
        "sample_count": loaded.sample_count,
        "seal_root": str(seal_root),
        "shard_count": len(loaded.shards),
        "split": loaded.split,
        "status": "complete",
    }
    occupancy_sidecars = loaded.manifest.get("occupancy_sidecars")
    if isinstance(occupancy_sidecars, dict):
        result["occupancy_sidecar_collection_digest_sha256"] = (
            occupancy_sidecars["collection_digest_sha256"]
        )
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
