#!/usr/bin/env python
"""Freeze the approved THÖR unseen-recording evaluation split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets.split_manager import SplitIndexError  # noqa: E402
from src.datasets.thor_split import (  # noqa: E402
    build_thor_recording_split,
    write_thor_recording_split_artifacts,
)


_CONFIG_KEYS = {
    "dataset",
    "evaluation_scope",
    "grouping_unit",
    "recording_overlap_policy",
    "session_overlap_policy",
    "participant_overlap_policy",
    "seed",
    "raw_root",
    "assignment_manifest",
}
_FROZEN_VALUES = {
    "dataset": "thor",
    "evaluation_scope": "unseen_recording_within_known_sessions",
    "grouping_unit": "recording_id",
    "recording_overlap_policy": "forbidden",
    "session_overlap_policy": "allowed_reported",
    "participant_overlap_policy": "unavailable",
}


def _load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise SplitIndexError("THÖR split config must be a mapping")
    unknown = set(config) - _CONFIG_KEYS
    missing = _CONFIG_KEYS - set(config)
    if unknown:
        raise SplitIndexError(
            "unknown THÖR split config keys: " + ", ".join(sorted(unknown))
        )
    if missing:
        raise SplitIndexError(
            "missing THÖR split config keys: " + ", ".join(sorted(missing))
        )
    for key, expected in _FROZEN_VALUES.items():
        if config[key] != expected:
            raise SplitIndexError(
                f"{key} must be frozen as {expected!r}, got {config[key]!r}"
            )
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise SplitIndexError("seed must be an integer")
    for key in ("raw_root", "assignment_manifest"):
        if not isinstance(config[key], str) or not config[key]:
            raise SplitIndexError(f"{key} must be a non-empty path string")
    return config


def _resolve_config_path(config_path: Path, declared: str) -> Path:
    path = Path(declared)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the audited THÖR unseen-recording split."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--assignment-manifest", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = _load_config(args.config)
    raw_root = args.raw_root or _resolve_config_path(
        args.config, str(config["raw_root"])
    )
    assignment_manifest = args.assignment_manifest or _resolve_config_path(
        args.config, str(config["assignment_manifest"])
    )
    seed = int(config["seed"]) if args.seed is None else args.seed
    build = build_thor_recording_split(
        raw_root=raw_root,
        assignment_manifest=assignment_manifest,
        seed=seed,
    )
    paths = write_thor_recording_split_artifacts(build, args.output_dir)
    report = build.result.overlap_report
    print(f"metadata={paths['metadata']}")
    print(f"manifest={paths['manifest']}")
    print(f"manifest_digest={build.result.manifest_digest}")
    print(f"recording_count={len(build.metadata)}")
    print(
        "allowed_session_overlap_count="
        f"{report['fields']['session']['overlap_count']}"
    )
    print(f"disallowed_overlap_count={report['disallowed_overlap_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
