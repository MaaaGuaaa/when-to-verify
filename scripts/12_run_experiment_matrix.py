#!/usr/bin/env python3
"""Run a SOP-16 toy or authenticated Long40 replay experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.toy_experiment_matrix import run_toy_experiment_matrix  # noqa: E402
from src.evaluation.experiment_matrix import run_experiment_matrix  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("toy", "replay"), default="toy")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.mode == "toy":
        if args.suite_root is not None:
            parser.error("toy mode does not accept --suite-root")
        result = run_toy_experiment_matrix(args.config, output_dir=args.output_dir)
        completed_run_count = len(result.run_paths)
        failed_run_count = 0
    else:
        if args.suite_root is None:
            parser.error("replay mode requires --suite-root")
        result = run_experiment_matrix(
            args.config,
            suite_root=args.suite_root,
            output_dir=args.output_dir,
        )
        completed_run_count = int(result.summary["completed_run_count"])
        failed_run_count = int(result.summary["failed_run_count"])
    print(
        json.dumps(
            {
                "mode": args.mode,
                "output_dir": str(result.output_dir),
                "completed_run_count": completed_run_count,
                "failed_run_count": failed_run_count,
            },
            sort_keys=True,
        )
    )
    return 0 if failed_run_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
