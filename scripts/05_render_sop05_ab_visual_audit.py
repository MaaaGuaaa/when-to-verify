#!/usr/bin/env python
"""Generate a small SOP05 A/B scenario audit and render the selected scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.sop05_ab_visual_audit import (  # noqa: E402
    publish_sop05_ab_visual_audit,
)
from src.generation.sop05_seen_prior import load_seen_prior_config  # noqa: E402
from src.generation.sop05_unseen_prior import (  # noqa: E402
    normalize_unseen_prior_config,
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_unseen_config(path: Path, *, base_config: dict) -> object:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid unseen-prior config: {path}") from exc
    return normalize_unseen_prior_config(raw, base_config=base_config)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and render a ten-per-regime SOP05 A/B audit."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--generator-config",
        type=Path,
        default=_ROOT / "configs/generator_obstacle_first_teb_train.yaml",
    )
    parser.add_argument(
        "--unseen-config",
        type=Path,
        default=_ROOT / "configs/sop05_unseen_prior.yaml",
    )
    parser.add_argument(
        "--seen-config",
        type=Path,
        default=_ROOT / "configs/sop05_seen_prior.yaml",
    )
    parser.add_argument(
        "--verification-action-config",
        type=Path,
        default=_ROOT / "configs/verification_actions.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count-per-regime", type=_positive_int, default=10)
    parser.add_argument("--selection-seed", type=_nonnegative_int, default=20260727)
    parser.add_argument(
        "--regime-a-present-only",
        action="store_true",
        help="condition the regime-A visual audit on the sampled pedestrian being present",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        source = load_sop05r_teb_output(args.source_root, require_complete=True)
        teb_config = load_sop05r_teb_config(args.generator_config)
        if source.manifest.get("config_digest") != teb_config.digest:
            raise ValueError("generator config digest differs from the mother collection")
        expected_action_digest = source.manifest.get("verification_action_digest")
        action_digest = _sha256_file(args.verification_action_config)
        if expected_action_digest is not None and expected_action_digest != action_digest:
            raise ValueError("verification-action config differs from the mother collection")
        result = publish_sop05_ab_visual_audit(
            source,
            output_dir=args.output_dir,
            sample_count_per_regime=args.sample_count_per_regime,
            selection_seed=args.selection_seed,
            regime_a_present_only=args.regime_a_present_only,
            unseen_config=_load_unseen_config(
                args.unseen_config,
                base_config=dict(source.manifest["base_config"]),
            ),
            seen_config=load_seen_prior_config(args.seen_config),
            teb_config=teb_config,
            action_library=load_verification_actions(args.verification_action_config),
        )
    except (OSError, ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(result.output_dir),
                "regime_a_count": len(result.regime_a_event_ids),
                "regime_b_count": len(result.regime_b_event_ids),
                "regime_a_present_only": args.regime_a_present_only,
                "source_publication_semantic_digest": (
                    result.source_publication_semantic_digest
                ),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
