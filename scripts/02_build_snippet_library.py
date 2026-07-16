#!/usr/bin/env python
"""Build split- and type-isolated THÖR dynamic-object snippet libraries.

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

from src.contracts import DYNAMIC_OBJECT_TYPES  # noqa: E402
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


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a positive integer"
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _validate_data_config(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict) or config.get("dataset") != "thor":
        raise ThorDataError("data config dataset must be 'thor'")
    if config.get("input_format") != "jsonl":
        raise ThorDataError("data config input_format must be 'jsonl'")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build normalized dynamic-object snippet libraries for one split."
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
    parser.add_argument("--workers", type=_positive_integer, default=8)
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
    dynamic_config = base_config["dynamic_objects"]
    libraries = []
    for object_type in DYNAMIC_OBJECT_TYPES:
        thresholds = dynamic_config[object_type]
        libraries.append(
            build_snippet_library(
                recordings,
                split=args.split,
                object_type=object_type,
                duration_s=args.duration_s,
                stride_s=args.stride_s,
                min_mean_speed_mps=float(thresholds["min_speed_mps"]),
                max_mean_speed_mps=float(thresholds["max_speed_mps"]),
                max_acceleration_mps2=float(
                    thresholds["max_acceleration_mps2"]
                ),
                workers=args.workers,
            )
        )

    existing = []
    for split in SPLIT_NAMES:
        for object_type in DYNAMIC_OBJECT_TYPES:
            path = args.output_dir / split / object_type / "snippet_library.npz"
            if path.is_file():
                existing.append(load_snippet_library(path))
    report = audit_snippet_source_overlap([*existing, *libraries])
    if report["total_overlap_count"]:
        raise SplitLeakageError("snippet source overlap detected")
    paths = [
        write_snippet_artifacts(
            library,
            args.output_dir / args.split / library.object_type,
            overlap_report=report,
        )
        for library in libraries
    ]
    for library, library_paths in zip(libraries, paths):
        print(f"library[{library.object_type}]={library_paths['library']}")
        print(
            f"accepted_count[{library.object_type}]="
            f"{library.summary['accepted_count']}"
        )
    print(f"split={args.split}")
    print(f"workers_requested={args.workers}")
    print(f"workers_used={min(args.workers, len(recordings))}")
    print(
        "total_candidate_count="
        f"{sum(int(item.summary['candidate_count']) for item in libraries)}"
    )
    print(
        "total_accepted_count="
        f"{sum(int(item.summary['accepted_count']) for item in libraries)}"
    )
    print(
        "total_rejected_count="
        f"{sum(int(item.summary['rejected_count']) for item in libraries)}"
    )
    print(f"source_overlap_count={report['total_overlap_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
