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
from src.generation.sop05r_contracts import (  # noqa: E402
    SOP05R_TEB_GENERATOR_VERSION,
    SOP05R_TEB_RUN_VERSION,
    Sop05rConfigError,
    load_sop05r_teb_config,
)
from src.generation.sop05r_teb_run import (  # noqa: E402
    Sop05rTebRunError,
    Sop05rTebRunRequest,
    execute_sop05r_teb_run,
    preflight_summary as sop05r_teb_preflight_summary,
)
from src.planning.verification_actions import load_verification_actions  # noqa: E402
from src.utils.config import ConfigError  # noqa: E402


_EXPECTED_INPUT_ERRORS = (
    Sop05rTebRunError,
    Sop05rConfigError,
    ConfigError,
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
        description="Generate one deterministic lightweight-TEB SOP05R collection."
    )
    parser.add_argument(
        "--generator-mode",
        choices=("obstacle_first_teb",),
        required=True,
    )
    parser.add_argument("--sop03-root", type=Path, required=True)
    parser.add_argument("--long40-human-artifact", type=Path, required=True)
    parser.add_argument("--split", choices=SPLIT_NAMES, required=True)
    parser.add_argument(
        "--base-config", type=Path, default=_ROOT / "configs/base.yaml"
    )
    parser.add_argument("--generator-config", type=Path, required=True)
    parser.add_argument("--verification-action-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=_nonnegative_int, required=True)
    parser.add_argument("--accepted-quota", type=_positive_int, required=True)
    parser.add_argument("--max-base-states", type=_positive_int, required=True)
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


def _run_obstacle_first_teb(
    args: argparse.Namespace,
) -> tuple[dict[str, object], int]:
    """Run or preflight the independent lightweight-TEB v3 producer."""

    try:
        config = load_sop05r_teb_config(args.generator_config)
    except TypeError as exc:
        raise Sop05rConfigError(
            f"invalid obstacle_first_teb generator config: {exc}"
        ) from exc
    try:
        load_verification_actions(args.verification_action_config)
    except ValueError as exc:
        raise Sop05rConfigError(
            f"invalid verification action config: {exc}"
        ) from exc
    request = Sop05rTebRunRequest(
        sop03_root=args.sop03_root,
        long40_human_artifact=args.long40_human_artifact,
        split=args.split,
        base_config_path=args.base_config,
        generator_config_path=args.generator_config,
        verification_action_config_path=args.verification_action_config,
        output_dir=args.output_dir,
        seed=args.seed,
        accepted_quota=args.accepted_quota,
        max_base_states=args.max_base_states,
        checksum_workers=args.checksum_workers,
        workers=args.workers,
        git_executable=args.git_executable,
    )
    if args.preflight_only:
        payload = sop05r_teb_preflight_summary(request)
        payload["config_digest"] = config.digest
        return payload, 0
    result = execute_sop05r_teb_run(
        request,
        progress_callback=lambda progress: print(
            json.dumps(progress, sort_keys=True),
            file=sys.stderr,
            flush=True,
        ),
    )
    return (
        {
            "producer_version": SOP05R_TEB_RUN_VERSION,
            "generator_algorithm_version": SOP05R_TEB_GENERATOR_VERSION,
            "run_state": result.run_state,
            "output_dir": str(result.output_dir),
            "accepted_count": result.accepted_count,
            "requested_count": result.requested_count,
            "publication_semantic_digest": result.publication_semantic_digest,
        },
        result.exit_code,
    )


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        payload, exit_code = _run_obstacle_first_teb(args)
    except _EXPECTED_INPUT_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
