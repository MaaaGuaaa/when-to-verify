#!/usr/bin/env python
"""Build the reusable SOP-04 canonical local-trajectory bank."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.planning.trajectory_bank import (  # noqa: E402
    TRAJECTORY_BANK_VERSION,
    build_trajectory_bank,
    trajectory_bank_semantic_digest,
    write_trajectory_bank,
)
from src.planning.differential_drive import (  # noqa: E402
    POSE_TIME_LAYOUT_VERSION,
)
from src.utils.config import load_config  # noqa: E402


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a positive integer"
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a finite positive number"
        ) from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the canonical SOP-04 trajectory/query-map bank."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/base.yaml",
    )
    parser.add_argument(
        "--braking-deceleration-mps2",
        type=_positive_finite_float,
        required=True,
    )
    parser.add_argument("--workers", type=_positive_integer, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-commit", default="unversioned")
    args = parser.parse_args()

    generation_started_at_utc = datetime.now(timezone.utc).isoformat()
    config = load_config(args.config)
    bank = build_trajectory_bank(
        config,
        braking_deceleration_mps2=args.braking_deceleration_mps2,
        workers=args.workers,
    )
    determinism_reference = (
        build_trajectory_bank(
            config,
            braking_deceleration_mps2=args.braking_deceleration_mps2,
            workers=1,
        )
        if args.workers > 1
        else build_trajectory_bank(
            config,
            braking_deceleration_mps2=args.braking_deceleration_mps2,
            workers=2,
        )
    )
    paths = write_trajectory_bank(
        bank,
        args.output_dir,
        provenance={
            "code_commit": args.code_commit,
            "config": str(args.config),
            "config_snapshot": config,
            "canonical_shared_bank": True,
            "generation_started_at_utc": generation_started_at_utc,
            "generation_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        determinism_reference=determinism_reference,
    )
    print(f"bank={paths['bank']}")
    print(f"manifest={paths['manifest']}")
    print(f"checksums={paths['checksums']}")
    print(f"audit={paths['audit']}")
    print(f"handoff_digest={paths['handoff_digest']}")
    print(f"trajectory_bank_version={TRAJECTORY_BANK_VERSION}")
    print(f"pose_time_layout_version={POSE_TIME_LAYOUT_VERSION}")
    print(
        "bank_semantic_digest_sha256="
        f"{trajectory_bank_semantic_digest(bank)}"
    )
    print(
        "external_handoff_digest_sha256="
        f"{paths['handoff_digest'].read_text(encoding='utf-8').split()[0]}"
    )
    print(f"candidate_count={bank.summary['candidate_count']}")
    print(f"accepted_count={bank.summary['accepted_count']}")
    print(f"rejected_count={bank.summary['rejected_count']}")
    print(f"acceptance_rate={bank.summary['acceptance_rate']:.6f}")
    print(f"workers_requested={bank.summary['workers_requested']}")
    print(f"workers_used={bank.summary['workers_used']}")
    print(
        "braking_deceleration_mps2="
        f"{bank.summary['braking_deceleration_mps2']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
