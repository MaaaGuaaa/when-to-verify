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
from src.generation.anchored_human_placement import (  # noqa: E402
    PLACEMENT_SELECTION_MODES,
)
from src.generation.sop05_input_adapter import Sop05InputError  # noqa: E402
from src.generation.sop05_run import (  # noqa: E402
    SOP05_RUN_VERSION,
    Sop05RunError,
    Sop05RunRequest,
    execute_sop05_run,
    preflight_summary,
    prepare_sop05_run,
)
from src.generation.sop05_selection import (  # noqa: E402
    SOP05_TOTAL_QUOTA_SELECTION_VERSION,
)
from src.generation.sop05r_contracts import (  # noqa: E402
    SOP05R_RUN_VERSION,
    SOP05R_SELECTION_VERSION,
    SOP05R_TEB_GENERATOR_VERSION,
    SOP05R_TEB_RUN_VERSION,
    Sop05rConfigError,
    load_sop05r_teb_config,
)
from src.generation.sop05r_run import (  # noqa: E402
    Sop05rRunError,
    Sop05rRunRequest,
    execute_sop05r_run,
    preflight_summary as sop05r_preflight_summary,
    prepare_sop05r_run,
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
    Sop05InputError,
    Sop05RunError,
    Sop05rRunError,
    Sop05rTebRunError,
    Sop05rConfigError,
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


def _lower_sha256(text: str) -> str:
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise argparse.ArgumentTypeError(
            "must be 64 lowercase hexadecimal characters"
        )
    return text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one deterministic legacy, obstacle-first, or "
            "lightweight-TEB SOP05R event collection."
        )
    )
    parser.add_argument(
        "--generator-mode",
        choices=("legacy", "obstacle_first", "obstacle_first_teb"),
        required=True,
    )
    parser.add_argument("--sop03-root", type=Path, required=True)
    parser.add_argument("--long40-human-artifact", type=Path)
    parser.add_argument("--sop04-root", type=Path)
    parser.add_argument(
        "--sop04-handoff-digest",
        type=_lower_sha256,
        required=False,
        help=(
            "Trusted external SOP-04 v2 handoff SHA-256 supplied outside "
            "the artifact directory."
        ),
    )
    parser.add_argument("--split", choices=SPLIT_NAMES, required=True)
    parser.add_argument(
        "--base-config", type=Path, default=_ROOT / "configs/base.yaml"
    )
    parser.add_argument("--generator-config", type=Path, required=True)
    parser.add_argument("--verification-action-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=_nonnegative_int, required=True)
    parser.add_argument("--accepted-quota", type=_positive_int)
    parser.add_argument(
        "--all-accepted",
        action="store_true",
        help=(
            "obstacle_first_teb only: process every requested BaseState and "
            "publish every accepted M6 mother"
        ),
    )
    parser.add_argument("--events-per-pair", type=_positive_int)
    parser.add_argument("--max-base-states", type=_positive_int, required=True)
    parser.add_argument("--base-state-start", type=_nonnegative_int, default=0)
    parser.add_argument("--exclude-existing-output", type=Path)
    parser.add_argument("--resume-publish-staging", type=Path)
    parser.add_argument(
        "--placement-selection-mode",
        choices=PLACEMENT_SELECTION_MODES,
        default="seen_first",
    )
    parser.add_argument("--trajectory-count", type=_positive_int)
    parser.add_argument("--max-pairs", type=_positive_int)
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


def _validate_mode_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.generator_mode == "obstacle_first_teb":
        if args.all_accepted == (args.accepted_quota is not None):
            parser.error(
                "obstacle_first_teb mode requires exactly one of "
                "--accepted-quota or --all-accepted"
            )
    elif args.all_accepted:
        parser.error("--all-accepted is only supported by obstacle_first_teb mode")
    elif args.accepted_quota is None:
        parser.error("--accepted-quota is required for this generator mode")
    if args.generator_mode != "obstacle_first_teb" and (
        args.base_state_start != 0
        or args.exclude_existing_output is not None
        or args.resume_publish_staging is not None
        or args.placement_selection_mode != "seen_first"
    ):
        parser.error(
            "--base-state-start, --exclude-existing-output, and "
            "--resume-publish-staging, and --placement-selection-mode h0_hidden "
            "are only supported by obstacle_first_teb mode"
        )
    if args.generator_mode == "legacy":
        missing = [
            option
            for option, value in (
                ("--sop04-root", args.sop04_root),
                ("--sop04-handoff-digest", args.sop04_handoff_digest),
                ("--trajectory-count", args.trajectory_count),
                ("--max-pairs", args.max_pairs),
            )
            if value is None
        ]
        if missing:
            parser.error(
                "legacy mode requires " + ", ".join(missing)
            )
        if args.verification_action_config is not None:
            parser.error(
                "legacy mode does not accept --verification-action-config"
            )
        if args.long40_human_artifact is not None:
            parser.error("legacy mode does not accept --long40-human-artifact")
        return
    if args.verification_action_config is None:
        parser.error(
            f"{args.generator_mode} mode requires --verification-action-config"
        )
    forbidden = [
        option
        for option, value in (
            ("--sop04-root", args.sop04_root),
            ("--sop04-handoff-digest", args.sop04_handoff_digest),
            ("--events-per-pair", args.events_per_pair),
            ("--trajectory-count", args.trajectory_count),
            ("--max-pairs", args.max_pairs),
        )
        if value is not None
    ]
    if forbidden:
        parser.error(
            f"{args.generator_mode} mode does not accept " + ", ".join(forbidden)
        )
    if (
        args.generator_mode == "obstacle_first_teb"
        and args.long40_human_artifact is None
    ):
        parser.error(
            "obstacle_first_teb mode requires --long40-human-artifact"
        )
    if (
        args.generator_mode == "obstacle_first"
        and args.long40_human_artifact is not None
    ):
        parser.error("obstacle_first mode does not accept --long40-human-artifact")


def _run_legacy(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    assert args.accepted_quota is not None
    request = Sop05RunRequest(
        sop03_root=args.sop03_root,
        sop04_root=args.sop04_root,
        sop04_external_handoff_digest_sha256=args.sop04_handoff_digest,
        split=args.split,
        base_config_path=args.base_config,
        generator_config_path=args.generator_config,
        output_dir=args.output_dir,
        seed=args.seed,
        accepted_quota=args.accepted_quota,
        events_per_pair=(
            10 if args.events_per_pair is None else args.events_per_pair
        ),
        max_base_states=args.max_base_states,
        trajectory_count=args.trajectory_count,
        max_pairs=args.max_pairs,
        checksum_workers=args.checksum_workers,
        workers=args.workers,
        git_executable=args.git_executable,
    )
    if args.preflight_only:
        payload = preflight_summary(prepare_sop05_run(request))
        payload["publication_semantic_digest"] = None
        return payload, 0
    result = execute_sop05_run(request)
    payload = {
        "producer_version": SOP05_RUN_VERSION,
        "selection_version": SOP05_TOTAL_QUOTA_SELECTION_VERSION,
        "run_state": result.run_state,
        "run_id": result.run_id,
        "output_dir": str(result.output_dir),
        "selected_count": result.generation_summary["selected_count"],
        "allocated_cpu_seconds": result.generation_summary[
            "allocated_cpu_seconds"
        ],
        "publication_semantic_digest": result.publication_semantic_digest,
    }
    return payload, result.exit_code


def _run_obstacle_first(
    args: argparse.Namespace,
) -> tuple[dict[str, object], int]:
    assert args.accepted_quota is not None
    request = Sop05rRunRequest(
        sop03_root=args.sop03_root,
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
        payload = sop05r_preflight_summary(prepare_sop05r_run(request))
        payload["publication_semantic_digest"] = None
        return payload, 0
    result = execute_sop05r_run(request)
    return (
        {
            "producer_version": SOP05R_RUN_VERSION,
            "selection_version": SOP05R_SELECTION_VERSION,
            "run_state": result.run_state,
            "run_id": result.run_id,
            "output_dir": str(result.output_dir),
            "selected_count": result.generation_summary["selected_count"],
            "publication_semantic_digest": result.publication_semantic_digest,
        },
        result.exit_code,
    )


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
        base_state_start=args.base_state_start,
        exclude_existing_output=args.exclude_existing_output,
        resume_staging_root=args.resume_publish_staging,
        placement_selection_mode=args.placement_selection_mode,
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
    _validate_mode_arguments(parser, args)
    try:
        if args.generator_mode == "legacy":
            payload, exit_code = _run_legacy(args)
        elif args.generator_mode == "obstacle_first":
            payload, exit_code = _run_obstacle_first(args)
        elif args.generator_mode == "obstacle_first_teb":
            payload, exit_code = _run_obstacle_first_teb(args)
        else:  # pragma: no cover - argparse freezes the choices
            raise AssertionError("argparse accepted an unknown generator mode")
    except _EXPECTED_INPUT_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
