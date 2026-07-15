#!/usr/bin/env python
"""Extract robot-centric BaseState and separate OracleContext artifacts.

Usage:
    python scripts/03_extract_base_states.py \
      --config configs/data_thor.yaml --all-splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.contracts import build_grid_spec  # noqa: E402
from src.datasets.base_state_index import (  # noqa: E402
    extract_base_state_index,
    write_base_state_extraction,
)
from src.datasets.split_manager import SPLIT_NAMES  # noqa: E402
from src.datasets.thor_adapter import (  # noqa: E402
    ThorDataError,
    load_recording_indexes_from_dir,
)
from src.utils.config import load_config  # noqa: E402


def _validate_data_config(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict) or config.get("dataset") != "thor":
        raise ThorDataError("data config dataset must be 'thor'")
    if config.get("input_format") != "jsonl":
        raise ThorDataError("data config input_format must be 'jsonl'")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract split-isolated BaseState and OracleContext indexes."
    )
    parser.add_argument("--config", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--split", choices=SPLIT_NAMES)
    selection.add_argument("--all-splits", action="store_true")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=_ROOT / "configs/base.yaml",
    )
    parser.add_argument(
        "--recording-dir",
        type=Path,
        default=_ROOT / "outputs/recording_indexes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "outputs/base_state_indexes",
    )
    parser.add_argument("--stride-s", type=float, default=0.6)
    args = parser.parse_args()

    _validate_data_config(args.config)
    base_config = load_config(args.base_config)
    grid = build_grid_spec(base_config)
    expected_dt = float(base_config["bev"]["history_dt_s"])
    if float(base_config["bev"]["future_dt_s"]) != expected_dt:
        raise ThorDataError("history and future dt must match for SOP-03")
    splits = SPLIT_NAMES if args.all_splits else (args.split,)
    loaded = {
        split: load_recording_indexes_from_dir(
            args.recording_dir / split, expected_split=split
        )
        for split in splits
    }
    if any(
        abs(recording.dt_s - expected_dt) > 1e-9
        for recordings in loaded.values()
        for recording in recordings
    ):
        raise ThorDataError("recording dt does not match the frozen base config")

    total = 0
    for split in splits:
        extraction = extract_base_state_index(
            loaded[split],
            split=split,
            grid=grid,
            stride_s=args.stride_s,
        )
        paths = write_base_state_extraction(
            extraction, args.output_dir / split
        )
        accepted = int(extraction.summary["accepted_count"])
        total += accepted
        print(f"manifest={paths['manifest']}")
        print(f"split={split}")
        print(f"accepted_count={accepted}")
        print(f"rejected_count={extraction.summary['rejected_count']}")
    print(f"total_accepted_count={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
