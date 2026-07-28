#!/usr/bin/env python3
"""Run one SOP-15 toy or authenticated Long40 replay evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.toy_experiment_matrix import (  # noqa: E402
    load_closed_loop_config,
    run_toy_evaluation,
)
from src.evaluation.closed_loop_replay import (  # noqa: E402
    load_runtime_config,
    run_replay_evaluation,
)
from src.evaluation.result_registry import load_result  # noqa: E402
from src.planning.decision_policy import DECISION_STRATEGIES  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("toy", "replay"), default="toy")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--strategy", choices=DECISION_STRATEGIES, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episode-count", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.mode == "toy":
        if args.seed is None or args.episode_count is None:
            parser.error("toy mode requires --seed and --episode-count")
        if args.suite is not None:
            parser.error("toy mode does not accept --suite")
        config = load_closed_loop_config(args.config)
        output = run_toy_evaluation(
            config=config,
            strategy=args.strategy,
            seed=args.seed,
            episode_count=args.episode_count,
            output_dir=args.output_dir,
        )
        seed = args.seed
    else:
        if args.suite is None:
            parser.error("replay mode requires --suite")
        if args.seed is not None or args.episode_count is not None:
            parser.error("replay mode derives seed and episode count from --suite")
        config = load_runtime_config(args.config)
        output = run_replay_evaluation(
            suite_path=args.suite,
            strategy=args.strategy,
            config=config,
            output_dir=args.output_dir,
        )
        seed = load_result(output).provenance.seed
    print(
        json.dumps(
            {
                "mode": args.mode,
                "output_dir": str(output),
                "strategy": args.strategy,
                "seed": seed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
