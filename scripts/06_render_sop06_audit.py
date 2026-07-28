#!/usr/bin/env python
"""Render the required SOP06 offline audit pair and visibility toggle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.sop06_visual_audit import (  # noqa: E402
    load_sop06_visual_audit_packet,
    render_sop06_visual_audit,
)


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render offline-only SOP06 current/oracle/post-verification audit artifacts."
    )
    parser.add_argument(
        "--audit-packet",
        type=Path,
        required=True,
        help="pickle-free NPZ packet with schema sop06_visual_audit_packet_v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-duration-ms", type=_positive_int, default=750)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        bundle = load_sop06_visual_audit_packet(args.audit_packet)
        artifact = render_sop06_visual_audit(
            bundle,
            args.output_dir,
            frame_duration_ms=args.frame_duration_ms,
        )
    except (OSError, ValueError, TypeError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(artifact.output_dir),
                "bev_pair": str(artifact.bev_pair_path),
                "bev_toggle": str(artifact.bev_toggle_path),
                "manifest": str(artifact.manifest_path),
                "candidate_blind_endpoint_reveal_fraction": artifact.metadata[
                    "candidate_blind_endpoint_reveal_fraction"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
