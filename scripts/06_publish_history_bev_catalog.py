#!/usr/bin/env python
"""Publish an immutable catalog over completed SOP06 history-BEV releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets.sop06_history_catalog import (  # noqa: E402
    publish_sop06_history_catalog,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish one immutable catalog over completed SOP06 releases."
    )
    parser.add_argument(
        "--entry-root",
        action="append",
        type=Path,
        required=True,
        help="completed SOP06 entry root; repeat for every entry",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        loaded = publish_sop06_history_catalog(
            tuple(args.entry_root),
            args.output_dir,
        )
    except (OSError, TypeError, ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(loaded.root),
                "entry_count": loaded.entry_count,
                "sample_count": loaded.sample_count,
                "catalog_digest": loaded.catalog_digest,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
