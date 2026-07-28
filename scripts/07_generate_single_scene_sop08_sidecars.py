#!/usr/bin/env python
"""Build SOP08 supervision sidecars for one completed SOP07 release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.generation.sop07_sop08_sidecars import (  # noqa: E402
    Sop07Sop08SidecarRequest,
    publish_sop07_sop08_sidecars,
)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sop07-release", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--shard-modulus", type=_positive_int, default=1)
    parser.add_argument("--shard-remainder", type=_nonnegative_int, default=0)
    args = parser.parse_args()
    if args.shard_remainder >= args.shard_modulus:
        parser.error("--shard-remainder must be smaller than --shard-modulus")
    try:
        result = publish_sop07_sop08_sidecars(
            Sop07Sop08SidecarRequest(
                sop07_release_root=args.sop07_release,
                sidecar_root=args.sidecar_root,
                shard_modulus=args.shard_modulus,
                shard_remainder=args.shard_remainder,
            ),
            progress_callback=lambda completed, total, reused: print(
                json.dumps(
                    {
                        "completed_shards": completed,
                        "last_shard_reused": reused,
                        "status": "running",
                        "total_shards": total,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            ),
        )
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "sample_count": result.sample_count,
                "shard_count": result.shard_count,
                "sidecar_root": str(result.sidecar_root),
                "status": "complete",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
