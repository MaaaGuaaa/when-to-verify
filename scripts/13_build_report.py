#!/usr/bin/env python3
"""Build publication tables and figures from authenticated SOP-16 matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.plots import build_evaluation_report  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        action="append",
        required=True,
        help="Authenticated matrix directory; repeat to combine main/ablation/sensitivity.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_evaluation_report(
        args.matrix_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(report.output_dir),
                "generated_file_count": len(report.generated_files),
                "all_sources_scientifically_complete": report.summary[
                    "all_sources_scientifically_complete"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if report.summary["all_sources_scientifically_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
