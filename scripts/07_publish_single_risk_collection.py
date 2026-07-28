#!/usr/bin/env python
"""Publish an SOP07 single-risk release as an SOP08 risk collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets.risk_dataloader import RiskDataContractError  # noqa: E402
from src.datasets.sop07_single_risk_collection import (  # noqa: E402
    publish_sop07_single_risk_collection,
)


def _lower_hex(length: int):
    def parse(value: str) -> str:
        if len(value) != length or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise argparse.ArgumentTypeError(
                f"must be exactly {length} lowercase hexadecimal characters"
            )
        return value

    return parse


def _repository_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly adapt sop07_single_risk_release_v1 into a flat immutable "
            "risk collection accepted by the SOP08 dataset-seal loader."
        )
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-type-policy-digest",
        type=_lower_hex(32),
        required=True,
    )
    parser.add_argument(
        "--code-commit",
        type=_lower_hex(40),
        help="Defaults to the current repository HEAD.",
    )
    args = parser.parse_args()
    try:
        result = publish_sop07_single_risk_collection(
            args.release_root,
            args.output_dir,
            target_type_policy_digest=args.target_type_policy_digest,
            code_commit=args.code_commit or _repository_commit(),
        )
    except (
        FileExistsError,
        OSError,
        subprocess.CalledProcessError,
        RiskDataContractError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "collection_root": str(result.root),
                "collection_semantic_digest_sha256": (
                    result.collection_semantic_digest
                ),
                "handoff_sha256": result.handoff_sha256,
                "sample_count": result.sample_count,
                "shard_count": result.shard_count,
                "source_release_manifest_digest": (
                    result.source_release_manifest_digest
                ),
                "split": result.split,
                "status": "complete",
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
