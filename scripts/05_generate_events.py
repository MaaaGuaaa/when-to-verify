#!/usr/bin/env python
"""Run bounded, audited SOP-05 event generation for one accepted split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets.split_manager import SPLIT_NAMES  # noqa: E402
from src.contracts import ContractError  # noqa: E402
from src.generation.event_sampler import GeneratorConfigError  # noqa: E402
from src.generation.sop05_input_adapter import Sop05InputError  # noqa: E402
from src.generation.sop05_run import (  # noqa: E402
    Sop05RunError,
    Sop05RunRequest,
    execute_sop05_run,
    preflight_summary,
    prepare_sop05_run,
)
from src.utils.config import ConfigError  # noqa: E402


_EXPECTED_INPUT_ERRORS = (
    Sop05InputError,
    Sop05RunError,
    ConfigError,
    GeneratorConfigError,
    ContractError,
    FileExistsError,
    yaml.YAMLError,
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
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one deterministic SOP-05 shard from accepted SOP-03 "
            "and SOP-04 producer artifacts."
        )
    )
    parser.add_argument("--sop03-root", type=Path, required=True)
    parser.add_argument("--sop04-root", type=Path, required=True)
    parser.add_argument("--split", choices=SPLIT_NAMES, required=True)
    parser.add_argument(
        "--base-config", type=Path, default=_ROOT / "configs/base.yaml"
    )
    parser.add_argument("--generator-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=_nonnegative_int, required=True)
    parser.add_argument("--accepted-quota", type=_positive_int, required=True)
    parser.add_argument("--events-per-pair", type=_positive_int, default=10)
    parser.add_argument("--max-base-states", type=_positive_int, required=True)
    parser.add_argument("--trajectory-count", type=_positive_int, required=True)
    parser.add_argument("--max-pairs", type=_positive_int, required=True)
    parser.add_argument("--checksum-workers", type=_positive_int, default=8)
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="CPU process workers; does not affect scientific identity.",
    )
    parser.add_argument(
        "--git-executable",
        type=Path,
        required=True,
        help=(
            "Absolute non-symlink Git executable used only for read-only "
            "source identity checks; does not affect scientific identity."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    request = Sop05RunRequest(
        sop03_root=args.sop03_root,
        sop04_root=args.sop04_root,
        split=args.split,
        base_config_path=args.base_config,
        generator_config_path=args.generator_config,
        output_dir=args.output_dir,
        seed=args.seed,
        accepted_quota=args.accepted_quota,
        events_per_pair=args.events_per_pair,
        max_base_states=args.max_base_states,
        trajectory_count=args.trajectory_count,
        max_pairs=args.max_pairs,
        checksum_workers=args.checksum_workers,
        workers=args.workers,
        git_executable=args.git_executable,
    )
    try:
        if args.preflight_only:
            payload = preflight_summary(prepare_sop05_run(request))
            payload["publication_semantic_digest"] = None
            print(json.dumps(payload, sort_keys=True, allow_nan=False))
            return 0

        result = execute_sop05_run(request)
    except _EXPECTED_INPUT_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    payload = {
        "run_state": result.run_state,
        "run_id": result.run_id,
        "output_dir": str(result.output_dir),
        "selected_count": result.generation_summary["selected_count"],
        "publication_semantic_digest": result.publication_semantic_digest,
    }
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
