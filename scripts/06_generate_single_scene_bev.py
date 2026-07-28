#!/usr/bin/env python
"""Render one immutable SOP06 history-BEV release from finalized SOP05 scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.generation.sop06_history_release import (  # noqa: E402
    Sop06HistoryReleaseRequest,
    publish_sop06_history_release,
)


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render finalized SOP05 scenarios into one SOP06 history-BEV release."
    )
    parser.add_argument(
        "--source-family",
        required=True,
        choices=("natural", "a_supplement"),
    )
    parser.add_argument(
        "--source-mode",
        required=True,
        choices=("complete_mother", "partial_m6_reconstruction"),
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-cache-root", type=Path)
    parser.add_argument("--final-scenario-root", required=True, type=Path)
    parser.add_argument(
        "--split",
        required=True,
        choices=("train", "calibration", "val", "test"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--samples-per-shard", type=_positive_int, default=128)
    parser.add_argument("--sop03-root", type=Path)
    parser.add_argument("--long40-human-artifact", type=Path)
    parser.add_argument("--base-state-start", type=_nonnegative_int)
    parser.add_argument("--max-base-states", type=_positive_int)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--generator-config", type=Path)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    reconstruction_values = (
        args.sop03_root,
        args.long40_human_artifact,
        args.base_state_start,
        args.max_base_states,
    )
    if args.source_mode == "partial_m6_reconstruction":
        if any(value is None for value in reconstruction_values):
            parser.error(
                "partial_m6_reconstruction requires --sop03-root, "
                "--long40-human-artifact, --base-state-start, and "
                "--max-base-states"
            )
        base_config = args.base_config or (_ROOT / "configs/base.yaml")
        generator_config = args.generator_config or (
            _ROOT / "configs/generator_obstacle_first_teb_train.yaml"
        )
    else:
        if any(value is not None for value in reconstruction_values) or (
            args.base_config is not None or args.generator_config is not None
        ):
            parser.error(
                "partial-M6 reconstruction arguments require "
                "--source-mode partial_m6_reconstruction"
            )
        base_config = None
        generator_config = None
    request = Sop06HistoryReleaseRequest(
        source_family=args.source_family,
        source_mode=args.source_mode,
        source_root=args.source_root,
        source_cache_root=args.source_cache_root,
        final_scenario_root=args.final_scenario_root,
        split=args.split,
        output_dir=args.output_dir,
        workers=args.workers,
        samples_per_shard=args.samples_per_shard,
        sop03_root=args.sop03_root,
        long40_human_artifact=args.long40_human_artifact,
        base_state_start=args.base_state_start,
        max_base_states=args.max_base_states,
        base_config_path=base_config,
        generator_config_path=generator_config,
    )

    def progress(completed: int, total: int, reused: bool) -> None:
        if completed == total or completed % 10 == 0:
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
        result = publish_sop06_history_release(
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
                "source_family": result.source_family,
                "source_mode": result.source_mode,
                "source_publication_semantic_digest": (
                    result.source_publication_semantic_digest
                ),
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
