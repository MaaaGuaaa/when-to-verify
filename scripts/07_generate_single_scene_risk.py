#!/usr/bin/env python
"""Build one resumable SOP07 risk release from a completed SOP06 release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.generation.sop07_risk_release import (  # noqa: E402
    Sop07RiskReleaseRequest,
    publish_sop07_risk_release,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join persisted SOP06 observations to same-scene SOP05 oracle "
            "data and publish one resumable SOP07 risk shard per SOP06 shard."
        )
    )
    parser.add_argument("--sop06-release", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sop03-root", type=Path)
    parser.add_argument("--long40-human-artifact", type=Path)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if (args.sop03_root is None) != (args.long40_human_artifact is None):
        parser.error(
            "legacy complete-mother recovery requires both --sop03-root and "
            "--long40-human-artifact"
        )
    request = Sop07RiskReleaseRequest(
        sop06_release_root=args.sop06_release,
        output_dir=args.output_dir,
        sop03_root=args.sop03_root,
        long40_human_artifact=args.long40_human_artifact,
    )

    def progress(completed: int, total: int, reused: bool) -> None:
        print(
            json.dumps(
                {
                    "status": "running",
                    "completed_shards": completed,
                    "total_shards": total,
                    "last_shard_reused": reused,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    try:
        result = publish_sop07_risk_release(
            request,
            progress_callback=progress,
        )
    except (OSError, TypeError, ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(result.output_dir),
                "split": result.split,
                "sample_count": result.sample_count,
                "shard_count": result.shard_count,
                "reused_shard_count": result.reused_shard_count,
                "manifest_digest": result.manifest_digest,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
