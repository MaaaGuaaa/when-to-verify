#!/usr/bin/env python3
"""Aggregate authenticated verification-value evaluation seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.verification_experiment_aggregation import (  # noqa: E402
    aggregate_verification_evaluations,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir",
        action="append",
        dest="evaluation_dirs",
        type=Path,
        required=True,
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    loaded = aggregate_verification_evaluations(
        tuple(args.evaluation_dirs),
        experiment_id=args.experiment_id,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "experiment_id": loaded.experiment_id,
                "run_count": loaded.run_count,
                "seeds": list(loaded.seeds),
                "aggregate_digest": loaded.aggregate_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
