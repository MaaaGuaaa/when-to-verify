#!/usr/bin/env python
"""Render an immutable three-sample real seen-then-occluded visual audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.contracts import ContractError  # noqa: E402
from src.generation.event_sampler import GeneratorConfigError  # noqa: E402
from src.generation.paired_variants import (  # noqa: E402
    PairedVariantConfigError,
)
from src.generation.sop05_input_adapter import Sop05InputError  # noqa: E402
from src.generation.sop05_run import Sop05RunError  # noqa: E402
from src.evaluation.seen_occluded_visual_audit import (  # noqa: E402
    SeenOccludedAuditError,
    SeenOccludedAuditRequest,
    prepare_real_audit,
    real_audit_preflight_summary,
    run_real_audit,
)
from src.evaluation.seen_occluded_joint_search import (  # noqa: E402
    JointAuditSearchError,
)
from src.utils.config import ConfigError  # noqa: E402


_EXPECTED_ERRORS = (
    SeenOccludedAuditError,
    Sop05InputError,
    Sop05RunError,
    GeneratorConfigError,
    PairedVariantConfigError,
    ConfigError,
    ContractError,
    FileExistsError,
    OSError,
    RuntimeError,
    yaml.YAMLError,
    JointAuditSearchError,
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
            "Search authenticated schema-3 THOR inputs for three complete "
            "seen-then-occluded visual-audit groups."
        )
    )
    parser.add_argument("--sop03-root", type=Path, required=True)
    parser.add_argument("--sop04-root", type=Path, required=True)
    parser.add_argument(
        "--sop04-handoff-digest",
        type=_lower_sha256,
        required=True,
    )
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument(
        "--base-config", type=Path, default=_ROOT / "configs/base.yaml"
    )
    parser.add_argument(
        "--generator-config",
        type=Path,
        default=_ROOT / "configs/generator_seen_occluded_visual_audit.yaml",
    )
    parser.add_argument(
        "--paired-config",
        type=Path,
        default=_ROOT / "configs/paired_variants_visual_audit.yaml",
    )
    parser.add_argument(
        "--joint-config",
        type=Path,
        default=(
            _ROOT / "configs/seen_occluded_joint_visual_audit.yaml"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=_nonnegative_int, default=42)
    parser.add_argument("--sample-count", type=_positive_int, default=3)
    parser.add_argument("--events-per-pair", type=_positive_int, default=1)
    parser.add_argument("--max-base-states", type=_positive_int, default=512)
    parser.add_argument("--trajectory-count", type=_positive_int, default=21)
    parser.add_argument("--max-pairs", type=_positive_int, default=512)
    parser.add_argument("--max-seen-mothers", type=_positive_int, default=512)
    parser.add_argument("--checksum-workers", type=_positive_int, default=8)
    parser.add_argument("--workers", type=_positive_int, default=8)
    parser.add_argument("--git-executable", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _json_line(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = SeenOccludedAuditRequest(
        sop03_root=args.sop03_root,
        sop04_root=args.sop04_root,
        sop04_handoff_digest=args.sop04_handoff_digest,
        split=args.split,
        base_config_path=args.base_config,
        generator_config_path=args.generator_config,
        paired_config_path=args.paired_config,
        output_dir=args.output_dir,
        seed=args.seed,
        sample_count=args.sample_count,
        events_per_pair=args.events_per_pair,
        max_base_states=args.max_base_states,
        trajectory_count=args.trajectory_count,
        max_pairs=args.max_pairs,
        max_seen_mothers=args.max_seen_mothers,
        checksum_workers=args.checksum_workers,
        workers=args.workers,
        git_executable=args.git_executable,
        joint_config_path=args.joint_config,
    )
    try:
        if args.preflight_only:
            payload = real_audit_preflight_summary(prepare_real_audit(request))
            print(_json_line(payload))
            return 0
        result = run_real_audit(request)
    except _EXPECTED_ERRORS as exc:
        print(
            _json_line(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 2
    print(
        _json_line(
            {
                "status": result.status,
                "output_dir": str(result.output_dir),
                "manifest_sha256": result.manifest_sha256,
                "checksum_manifest_sha256": (
                    result.checksum_manifest_sha256
                ),
            }
        )
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
