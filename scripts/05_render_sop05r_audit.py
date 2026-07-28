#!/usr/bin/env python
"""Render a strict SOP05R statistical and visual audit collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.sop05r_audit import (  # noqa: E402
    Sop05rAuditError,
    Sop05rAuditRequest,
    run_sop05r_audit,
)
from src.evaluation.sop05r_teb_audit import (  # noqa: E402
    publish_sop05r_teb_visual_audit,
)
from src.generation.sop05r_contracts import load_sop05r_teb_config  # noqa: E402
from src.generation.sop05r_teb_output_loader import load_sop05r_teb_output  # noqa: E402
from src.planning.verification_actions import load_verification_actions  # noqa: E402


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
        description="Render one authenticated SOP05R visual audit collection."
    )
    parser.add_argument(
        "--generator-mode",
        choices=("obstacle_first_teb",),
        help="render the current authenticated Long40 TEB collection",
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--generator-config",
        type=Path,
        default=_ROOT / "configs/generator_obstacle_first_teb_train.yaml",
    )
    parser.add_argument(
        "--verification-action-config",
        type=Path,
        default=_ROOT / "configs/verification_actions.yaml",
    )
    parser.add_argument("--sop05r-root", type=Path)
    parser.add_argument("--sop03-root", type=Path)
    parser.add_argument("--paired-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=_positive_int, required=True)
    parser.add_argument("--seed", type=_nonnegative_int, required=True)
    parser.add_argument("--checksum-workers", type=_positive_int, default=8)
    return parser


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_teb_audit(args: argparse.Namespace) -> int:
    assert args.source_root is not None
    source = load_sop05r_teb_output(args.source_root, require_complete=True)
    teb_config = load_sop05r_teb_config(args.generator_config)
    if source.manifest.get("config_digest") != teb_config.digest:
        raise ValueError("generator config digest differs from the strict source")
    action_digest = _sha256_file(args.verification_action_config)
    if source.manifest.get("verification_action_digest") != action_digest:
        raise ValueError("verification-action config differs from the strict source")
    result = publish_sop05r_teb_visual_audit(
        source,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        teb_config=teb_config,
        action_library=load_verification_actions(args.verification_action_config),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(result.output_dir),
                "selected_count": len(result.selected_event_ids),
                "selected_event_ids": list(result.selected_event_ids),
                "publication_semantic_digest": result.publication_semantic_digest,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.generator_mode == "obstacle_first_teb":
        if args.source_root is None:
            parser.error("--source-root is required with --generator-mode")
        try:
            return _run_teb_audit(args)
        except (ValueError, FileExistsError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if (
        args.sop05r_root is None
        or args.sop03_root is None
        or args.paired_config is None
    ):
        parser.error(
            "legacy audit requires --sop05r-root, --sop03-root, and --paired-config"
        )
    request = Sop05rAuditRequest(
        sop05r_root=args.sop05r_root,
        sop03_root=args.sop03_root,
        paired_config_path=args.paired_config,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        checksum_workers=args.checksum_workers,
    )
    try:
        result = run_sop05r_audit(request)
    except (Sop05rAuditError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    payload = {
        "status": result.status,
        "output_dir": str(result.output_dir),
        "selected_count": len(result.selected_event_ids),
        "selected_event_ids": list(result.selected_event_ids),
        "manifest_sha256": result.manifest_sha256,
        "checksum_manifest_sha256": result.checksum_manifest_sha256,
    }
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
