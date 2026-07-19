#!/usr/bin/env python
"""Audit and atomically publish one complete SOP-03 producer directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets.sop03_publication import (  # noqa: E402
    Sop03PublicationError,
    finalize_sop03_artifact,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate every SOP-03 payload and publish its checksum envelope."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--base-config", type=Path, default=_ROOT / "configs/base.yaml"
    )
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--finalizer-commit", required=True)
    parser.add_argument("--git-executable", type=Path, required=True)
    parser.add_argument("--producer-job-id")
    parser.add_argument("--workers", type=_positive_int, default=8)
    args = parser.parse_args()
    try:
        git_path = args.git_executable
        if (
            not git_path.is_absolute()
            or not git_path.is_file()
            or git_path.is_symlink()
        ):
            raise Sop03PublicationError(
                "git executable must be an absolute, regular, non-symlink file"
            )
        observed_commit = subprocess.run(
            [str(git_path), "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if observed_commit != args.finalizer_commit:
            raise Sop03PublicationError(
                "finalizer commit does not match the current checkout"
            )
        result = finalize_sop03_artifact(
            args.root,
            base_config_path=args.base_config,
            producer_commit=args.producer_commit,
            finalizer_commit=args.finalizer_commit,
            producer_job_id=args.producer_job_id,
            workers=args.workers,
        )
    except (
        Sop03PublicationError,
        OSError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_root": str(args.root.resolve()),
                "checksum_manifest_sha256": result["checksum_summary"][
                    "checksum_manifest_sha256"
                ],
                "counts": result["counts"],
                "status": "complete",
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
