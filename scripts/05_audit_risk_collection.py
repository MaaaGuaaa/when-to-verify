#!/usr/bin/env python
"""Publish a global four-split SOP07 risk collection from trusted shard IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.contracts import build_grid_spec  # noqa: E402
from src.datasets.risk_collection import (  # noqa: E402
    RiskCollectionError,
    RiskCollectionMemberRequest,
    write_risk_collection,
)
from src.utils.config import ConfigError, load_config  # noqa: E402


_REQUEST_KEYS = frozenset(
    {
        "relative_path",
        "split",
        "shard_index",
        "expected_sample_count",
        "expected_manifest_digest",
        "expected_semantic_digest",
    }
)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Formally reload SOP07 shards and publish one immutable global "
            "train/calibration/val/test leakage proof."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-artifact-dir", type=Path, required=True)
    parser.add_argument("--expected-split-manifest-digest", required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--member-requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_member_requests(
    path: Path,
) -> tuple[RiskCollectionMemberRequest, ...]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise RiskCollectionError("failed to read member request JSONL") from exc
    if not raw or not raw.endswith(b"\n"):
        raise RiskCollectionError(
            "member request JSONL must be non-empty and newline-terminated"
        )
    requests: list[RiskCollectionMemberRequest] = []
    for index, line in enumerate(text.splitlines()):
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RiskCollectionError(
                f"member request row {index} is invalid JSON"
            ) from exc
        if not isinstance(value, dict) or set(value) != _REQUEST_KEYS:
            raise RiskCollectionError(f"member request row {index} keys mismatch")
        requests.append(RiskCollectionMemberRequest(**value))
    return tuple(requests)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        grid = build_grid_spec(load_config(args.config))
        members = _load_member_requests(args.member_requests)
        paths = write_risk_collection(
            members,
            args.output_dir,
            shard_root=args.shard_root,
            split_artifact_dir=args.split_artifact_dir,
            expected_split_manifest_digest=(
                args.expected_split_manifest_digest
            ),
            grid=grid,
        )
        summary = json.loads(
            paths["summary"].read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (ConfigError, FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = {
        "status": "ok",
        "member_count": summary["member_count"],
        "sample_count": summary["sample_count"],
        "split_manifest_digest": summary["split_manifest_digest"],
        "collection_semantic_digest": summary["collection_semantic_digest"],
    }
    print(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
