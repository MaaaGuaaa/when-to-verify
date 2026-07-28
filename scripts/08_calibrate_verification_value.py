#!/usr/bin/env python3
"""Select and seal the train-only verification reject cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.verification_value_calibration import (  # noqa: E402
    publish_reject_cost_calibration,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-dir",
        action="append",
        dest="release_dirs",
        type=Path,
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gt-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    loaded = publish_reject_cost_calibration(
        args.output_dir,
        release_dirs=tuple(args.release_dirs),
        config_path=args.config,
        gt_config_path=args.gt_config,
    )
    print(
        json.dumps(
                {
                    "status": loaded.status,
                    "output_dir": str(args.output_dir),
                "selected_reject_cost": loaded.selected_reject_cost,
                "calibration_digest": loaded.calibration_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0 if loaded.status == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
