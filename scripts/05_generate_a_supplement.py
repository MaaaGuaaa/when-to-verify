#!/usr/bin/env python
"""Publish one split-local targeted SOP05 regime-A supplement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.generation.sop05_a_supplement import (  # noqa: E402
    Sop05ASupplementError,
    load_sop05_a_supplement_config,
    publish_sop05_a_supplement,
)
from src.generation.sop05r_teb_output_loader import (  # noqa: E402
    load_sop05r_teb_output,
)


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish one deterministic split-local SOP05 A supplement."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "calibration", "val", "test"),
        required=True,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/sop05_a_supplement.yaml",
    )
    parser.add_argument("--workers", type=_positive_int, default=1)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        source = load_sop05r_teb_output(args.source_root, require_complete=True)
        config = load_sop05_a_supplement_config(args.config)
        result = publish_sop05_a_supplement(
            source,
            split=args.split,
            config=config,
            output_dir=args.output_dir,
            workers=args.workers,
        )
    except (OSError, ValueError, FileExistsError, Sop05ASupplementError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(result.publication.output_dir),
                "split": args.split,
                "accepted_count": result.publication.accepted_count,
                "present_count": result.present_count,
                "empty_count": result.empty_count,
                "source_publication_semantic_digest": (
                    result.publication.source_publication_semantic_digest
                ),
                "config_digest": result.config_digest,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
