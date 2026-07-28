#!/usr/bin/env python
"""Publish one finalized SOP05 A/B scenario for each mother event."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.generation.sop05_final_scenarios import (  # noqa: E402
    Sop05FinalScenarioError,
    publish_sop05_final_scenarios,
)
from src.generation.sop05_partial_m6_final import (  # noqa: E402
    PartialM6Error,
    publish_partial_m6_final_scenarios,
)
from src.generation.sop05_seen_prior import load_seen_prior_config  # noqa: E402
from src.generation.sop05_unseen_prior import (  # noqa: E402
    normalize_unseen_prior_config,
)
from src.generation.sop05r_contracts import load_sop05r_teb_config  # noqa: E402
from src.generation.sop05r_teb_output_loader import load_sop05r_teb_output  # noqa: E402
from src.generation.sop05r_teb_long40_inputs import (  # noqa: E402
    load_sop05r_teb_long40_inputs,
)
from src.contracts import build_grid_spec  # noqa: E402
from src.utils.config import load_config  # noqa: E402


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
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def _load_unseen_config(path: Path, *, base_config: dict) -> object:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid unseen-prior config: {path}") from exc
    return normalize_unseen_prior_config(raw, base_config=base_config)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish one finalized SOP05 blind-spot scenario per mother."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--partial-m6-staging",
        action="store_true",
        help="read completed M6 stores directly without the 47GB trajectory payload",
    )
    parser.add_argument("--sop03-root", type=Path)
    parser.add_argument("--long40-human-artifact", type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "calibration", "val", "test"),
        default="train",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=_ROOT / "configs/base.yaml",
    )
    parser.add_argument("--base-state-start", type=_nonnegative_int)
    parser.add_argument("--max-base-states", type=_positive_int)
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
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--max-mothers", type=_positive_int)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    partial_inputs = (
        args.sop03_root,
        args.long40_human_artifact,
        args.base_state_start,
        args.max_base_states,
    )
    if args.partial_m6_staging and any(value is None for value in partial_inputs):
        parser.error(
            "--partial-m6-staging requires --sop03-root, "
            "--long40-human-artifact, --base-state-start, and --max-base-states"
        )
    if not args.partial_m6_staging and any(value is not None for value in partial_inputs):
        parser.error("partial M6 source arguments require --partial-m6-staging")
    try:
        teb_config = load_sop05r_teb_config(args.generator_config)
        if args.partial_m6_staging:
            assert args.sop03_root is not None
            assert args.long40_human_artifact is not None
            assert args.base_state_start is not None
            assert args.max_base_states is not None
            base_config = load_config(args.base_config)
            inputs = load_sop05r_teb_long40_inputs(
                recording_root=args.sop03_root,
                long40_human_artifact=args.long40_human_artifact,
                split=args.split,
                grid=build_grid_spec(base_config),
                max_base_states=args.max_base_states,
                base_state_start=args.base_state_start,
            )
            source_states = {
                state.state_id: state for state, _ in inputs.state_pairs
            }
            snippet_sources = {
                snippet.snippet_id: (
                    snippet.source_recording_id,
                    snippet.source_session_id,
                )
                for snippet in inputs.snippets
            }

            def progress(completed: int, total: int) -> None:
                if completed == total or completed % 1000 == 0:
                    print(
                        json.dumps(
                            {
                                "status": "running",
                                "processed_mother_count": completed,
                                "source_mother_count": total,
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

            result = publish_partial_m6_final_scenarios(
                args.source_root,
                source_states=source_states,
                snippet_sources=snippet_sources,
                base_config=base_config,
                source_config_digest=teb_config.digest,
                centerline_epsilon_m=(
                    teb_config.occlusion.centerline_intersection_epsilon_m
                ),
                output_dir=args.output_dir,
                unseen_config=_load_unseen_config(
                    args.unseen_config,
                    base_config=base_config,
                ),
                seen_config=load_seen_prior_config(args.seen_config),
                workers=args.workers,
                max_mothers=args.max_mothers,
                progress_callback=progress,
            )
        else:
            source = load_sop05r_teb_output(args.source_root, require_complete=True)
            result = publish_sop05_final_scenarios(
                source,
                output_dir=args.output_dir,
                unseen_config=_load_unseen_config(
                    args.unseen_config,
                    base_config=dict(source.manifest["base_config"]),
                ),
                seen_config=load_seen_prior_config(args.seen_config),
                expected_source_config_digest=teb_config.digest,
                workers=args.workers,
                max_mothers=args.max_mothers,
            )
    except (
        OSError,
        ValueError,
        FileExistsError,
        PartialM6Error,
        Sop05FinalScenarioError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(result.output_dir),
                "processed_mother_count": result.processed_mother_count,
                "accepted_count": result.accepted_count,
                "deficit_count": result.deficit_count,
                "full_source_coverage": result.full_source_coverage,
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
