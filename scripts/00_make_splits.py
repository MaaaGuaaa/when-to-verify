#!/usr/bin/env python
"""Create deterministic connected-group splits from a source index (SOP-01).

The input is an index produced by a dataset adapter, not raw THÖR data.  SOP-03
owns the THÖR adapter; this command only consumes its JSONL index.

Usage:
    python scripts/00_make_splits.py \
      --config configs/data_thor.yaml \
      --seed 42 \
      --output-dir outputs/splits
"""

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

from src.datasets.split_manager import (  # noqa: E402
    SPLIT_NAMES,
    SplitIndexError,
    assert_no_split_leakage,
    make_split_manifest,
    write_split_artifacts,
)

_CONFIG_KEYS = {"dataset", "input_manifest", "input_format", "split_ratios"}


def _load_data_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise SplitIndexError("data config must be a mapping")
    unknown = set(config) - _CONFIG_KEYS
    missing = _CONFIG_KEYS - set(config)
    if unknown:
        raise SplitIndexError(
            "unknown data config keys: " + ", ".join(sorted(unknown))
        )
    if missing:
        raise SplitIndexError(
            "missing data config keys: " + ", ".join(sorted(missing))
        )
    if not isinstance(config["dataset"], str) or not config["dataset"]:
        raise SplitIndexError("dataset must be a non-empty string")
    if (
        not isinstance(config["input_manifest"], str)
        or not config["input_manifest"]
    ):
        raise SplitIndexError("input_manifest must be a non-empty string")
    if config["input_format"] != "jsonl":
        raise SplitIndexError("input_format must be 'jsonl'")
    if not isinstance(config["split_ratios"], dict):
        raise SplitIndexError("split_ratios must be a mapping")
    if set(config["split_ratios"]) != set(SPLIT_NAMES):
        raise SplitIndexError(
            "split_ratios must contain exactly: " + ", ".join(SPLIT_NAMES)
        )
    return config


def _reject_json_constant(value: str) -> None:
    raise SplitIndexError(f"input manifest must not contain {value}")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, SplitIndexError) as error:
                raise SplitIndexError(
                    f"invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise SplitIndexError(
                    f"JSONL row {line_number} must be an object"
                )
            records.append(row)
    return records


def _resolve_input_path(config_path: Path, declared_path: str) -> Path:
    path = Path(declared_path)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic connected-group "
            "train/calibration/val/test splits."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = _load_data_config(args.config)
    input_path = _resolve_input_path(args.config, config["input_manifest"])
    input_payload = input_path.read_bytes()
    records = _read_jsonl(input_path)
    result = make_split_manifest(
        records,
        seed=args.seed,
        ratios=config["split_ratios"],
    )
    assert_no_split_leakage(result.manifest)

    config_payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    result.summary.update(
        {
            "dataset": config["dataset"],
            "config_digest": hashlib.blake2b(
                config_payload, digest_size=16
            ).hexdigest(),
            "input_digest": hashlib.blake2b(
                input_payload, digest_size=16
            ).hexdigest(),
        }
    )
    result.overlap_report["config_digest"] = result.summary["config_digest"]
    paths = write_split_artifacts(result, args.output_dir)
    print(f"manifest={paths['manifest']}")
    print(f"manifest_digest={result.manifest_digest}")
    print(f"overlap_count={result.overlap_report['total_overlap_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
