#!/usr/bin/env python
"""Build one split-isolated THÖR pedestrian snippet library.

Usage:
    python scripts/02_build_snippet_library.py \
      --config configs/data_thor.yaml --split train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets.snippet_library import (  # noqa: E402
    audit_snippet_source_overlap,
    build_snippet_library,
    load_snippet_library,
    write_snippet_artifacts,
)
from src.datasets.split_manager import SPLIT_NAMES, SplitLeakageError  # noqa: E402
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
        description="Build a normalized pedestrian snippet library for one split."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=SPLIT_NAMES, required=True)
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
        default=_ROOT / "outputs/snippets",
    )
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--stride-s", type=float, default=1.0)
    args = parser.parse_args()

    _validate_data_config(args.config)
    base_config = load_config(args.base_config)
    recordings = load_recording_indexes_from_dir(
        args.recording_dir / args.split,
        expected_split=args.split,
    )
    expected_dt = float(base_config["bev"]["history_dt_s"])
    if float(base_config["bev"]["future_dt_s"]) != expected_dt:
        raise ThorDataError("history and future dt must match for SOP-03")
    if any(abs(recording.dt_s - expected_dt) > 1e-9 for recording in recordings):
        raise ThorDataError("recording dt does not match the frozen base config")
    pedestrian = base_config["pedestrian"]
    library = build_snippet_library(
        recordings,
        split=args.split,
        duration_s=args.duration_s,
        stride_s=args.stride_s,
        min_mean_speed_mps=float(pedestrian["min_speed_mps"]),
        max_mean_speed_mps=float(pedestrian["max_speed_mps"]),
        max_acceleration_mps2=float(pedestrian["max_acceleration_mps2"]),
    )

    existing = []
    for split in SPLIT_NAMES:
        path = args.output_dir / split / "snippet_library.npz"
        if path.is_file():
            existing.append(load_snippet_library(path))
    report = audit_snippet_source_overlap([*existing, library])
    if report["total_overlap_count"]:
        raise SplitLeakageError("snippet source overlap detected")
    paths = write_snippet_artifacts(
        library,
        args.output_dir / args.split,
        overlap_report=report,
    )
    print(f"library={paths['library']}")
    print(f"split={args.split}")
    print(f"candidate_count={library.summary['candidate_count']}")
    print(f"accepted_count={library.summary['accepted_count']}")
    print(f"rejected_count={library.summary['rejected_count']}")
    print(f"source_overlap_count={report['total_overlap_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
