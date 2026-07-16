#!/usr/bin/env python
"""Parse one split of official THÖR-MAGNI CSVs into recording indexes.

Usage:
    python scripts/01_index_recordings.py \
      --config configs/data_thor.yaml --split train
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets.split_manager import SPLIT_NAMES  # noqa: E402
from src.datasets.thor_adapter import (  # noqa: E402
    ThorDataError,
    load_thor_recording,
    parse_recording_id,
    write_recording_indexes,
)
from src.utils.config import load_config  # noqa: E402


def _reject_json_constant(value: str) -> None:
    raise ThorDataError(f"split manifest must not contain {value}")


def _load_data_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ThorDataError("data config must be a mapping")
    if config.get("dataset") != "thor":
        raise ThorDataError("data config dataset must be 'thor'")
    if config.get("input_format") != "jsonl":
        raise ThorDataError("data config input_format must be 'jsonl'")
    return config


def _read_split_rows(path: Path, split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
            except json.JSONDecodeError as error:
                raise ThorDataError(
                    f"invalid split manifest line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ThorDataError("split manifest rows must be objects")
            if row.get("split") == split:
                rows.append(row)
    if not rows:
        raise ThorDataError(f"split manifest has no rows for {split!r}")
    return sorted(rows, key=lambda row: str(row.get("recording_id", "")))


def _discover_csvs(raw_root: Path) -> dict[str, Path]:
    by_recording: dict[str, Path] = {}
    for path in sorted(raw_root.rglob("THOR-Magni_*.csv")):
        recording_id = parse_recording_id(path)
        if recording_id in by_recording:
            raise ThorDataError(f"duplicate raw recording id: {recording_id}")
        by_recording[recording_id] = path
    return by_recording


def _resolve_sources(
    rows: list[dict[str, object]], raw_root: Path
) -> list[Path]:
    discovered: dict[str, Path] | None = None
    sources: list[Path] = []
    seen: set[str] = set()
    for row in rows:
        recording_id = row.get("recording_id")
        if not isinstance(recording_id, str) or not recording_id:
            raise ThorDataError("every split row needs a recording_id")
        if recording_id in seen:
            continue
        seen.add(recording_id)
        declared = next(
            (
                row[key]
                for key in ("source_path", "csv_path", "path")
                if isinstance(row.get(key), str) and row[key]
            ),
            None,
        )
        if isinstance(declared, str):
            source = Path(declared)
            if not source.is_absolute():
                source = raw_root / source
        else:
            if discovered is None:
                discovered = _discover_csvs(raw_root)
            source = discovered.get(recording_id, Path())
        if not source.is_file():
            raise FileNotFoundError(
                f"raw CSV for recording {recording_id!r} was not found"
            )
        if parse_recording_id(source) != recording_id:
            raise ThorDataError(
                f"split row id {recording_id!r} does not match {source.name!r}"
            )
        sources.append(source)
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build split-isolated THÖR recording indexes."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=_ROOT / "configs/base.yaml",
    )
    parser.add_argument("--split", choices=SPLIT_NAMES, required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=_ROOT / "outputs/splits/split_manifest.jsonl",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=(
            _ROOT
            / "data/raw/thor_magni/THOR_MAGNI/CSVs_Scenarios"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "outputs/recording_indexes",
    )
    parser.add_argument("--dt-s", type=float, default=0.2)
    parser.add_argument("--max-gap-s", type=float, default=0.3)
    parser.add_argument(
        "--limit",
        type=int,
        help="Explicit smoke-test limit after deterministic source ordering.",
    )
    args = parser.parse_args()

    _load_data_config(args.config)
    base_config = load_config(args.base_config)
    if not math.isfinite(args.dt_s) or args.dt_s <= 0.0:
        raise ThorDataError("--dt-s must be finite and positive")
    if not math.isfinite(args.max_gap_s) or args.max_gap_s < args.dt_s:
        raise ThorDataError("--max-gap-s must be finite and at least --dt-s")
    if args.limit is not None and args.limit < 1:
        raise ThorDataError("--limit must be positive")
    rows = _read_split_rows(args.split_manifest, args.split)
    sources = _resolve_sources(rows, args.raw_root)
    if args.limit is not None:
        sources = sources[: args.limit]
    recordings = [
        load_thor_recording(
            source,
            dt_s=args.dt_s,
            max_gap_s=args.max_gap_s,
            dynamic_object_config=base_config["dynamic_objects"],
        )
        for source in sources
    ]
    paths = write_recording_indexes(
        recordings,
        split=args.split,
        output_dir=args.output_dir / args.split,
    )
    print(f"manifest={paths['manifest']}")
    print(f"split={args.split}")
    print(f"recording_count={len(recordings)}")
    print(f"robot_sample_count={sum(r.timestamps.size for r in recordings)}")
    print(
        "dynamic_object_track_count="
        f"{sum(len(r.dynamic_objects) for r in recordings)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
