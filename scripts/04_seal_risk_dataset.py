#!/usr/bin/env python
"""Seal an SOP07 collection and optional authenticated SOP08 sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets.risk_dataloader import RiskDataContractError  # noqa: E402
from src.datasets.risk_dataset_seal import (  # noqa: E402
    LoadedRiskDatasetFamily,
    load_risk_dataset_family,
    load_risk_dataset_seal,
    publish_risk_dataset_family,
    publish_risk_dataset_seal,
)


_FAMILY_MEMBER_ORDER = ("train", "calibration", "val", "test")


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError(
            "must be exactly 64 lowercase hexadecimal characters (SHA-256)"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate every immutable SOP07 shard and atomically publish "
            "a dataset-level risk_dataset_v2 seal."
        )
    )
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        help=(
            "Optional complete Task4 sidecar collection; when supplied, every "
            "risk/sidecar/marker triple is sealed for SOP08."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--split-provenance", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--expected-collection-handoff-sha256",
        type=_lower_sha256,
        required=True,
    )
    return parser


def _family_publish_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} family-publish",
        description=(
            "Explicitly reauthenticate four dataset seals and atomically "
            "publish risk_dataset_family_v1."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--member",
        action="append",
        nargs=4,
        required=True,
        metavar=("SPLIT", "SEAL_ROOT", "COLLECTION_ROOT", "DATASET_SHA256"),
        help=(
            "Repeat exactly once for train, calibration, val, and test; "
            "paths are supplied explicitly and are never discovered."
        ),
    )
    return parser


def _family_validate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} family-validate",
        description="Explicitly validate one risk_dataset_family_v1 seal.",
    )
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument(
        "--expected-member-digest",
        action="append",
        nargs=2,
        metavar=("SPLIT", "DATASET_SHA256"),
        help=(
            "Optional pinned member digest; when supplied, repeat exactly "
            "once for all four canonical splits."
        ),
    )
    return parser


def _family_result(loaded: LoadedRiskDatasetFamily) -> dict[str, object]:
    return {
        "family_root": str(loaded.root),
        "global_cross_split_leakage": loaded.cross_split_audit[
            "global_cross_split_leakage"
        ],
        "member_order": list(_FAMILY_MEMBER_ORDER),
        "members": {
            split: dict(loaded.members[split]) for split in _FAMILY_MEMBER_ORDER
        },
        "risk_dataset_family_digest": loaded.risk_dataset_family_digest,
        "status": "complete",
    }


def _print_result(value: dict[str, object]) -> None:
    print(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _main_family_publish(argv: list[str]) -> int:
    args = _family_publish_parser().parse_args(argv)
    try:
        raw_members = args.member
        by_split: dict[str, tuple[Path, Path, str]] = {}
        for split, seal_root, collection_root, digest in raw_members:
            if split in by_split:
                raise RiskDataContractError(
                    f"duplicate family member declaration for {split}"
                )
            by_split[split] = (
                Path(seal_root),
                Path(collection_root),
                digest,
            )
        if set(by_split) != set(_FAMILY_MEMBER_ORDER):
            raise RiskDataContractError(
                "family members must exactly cover train, calibration, val, test"
            )
        expected_digests = {
            split: by_split[split][2] for split in _FAMILY_MEMBER_ORDER
        }
        members = {
            split: load_risk_dataset_seal(
                by_split[split][0],
                collection_root=by_split[split][1],
                expected_split=split,
                expected_manifest_digest=expected_digests[split],
            )
            for split in _FAMILY_MEMBER_ORDER
        }
        family_root = publish_risk_dataset_family(
            args.output_dir,
            members=members,
        )
        loaded = load_risk_dataset_family(
            family_root,
            expected_member_digests=expected_digests,
        )
    except (RiskDataContractError, FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_result(_family_result(loaded))
    return 0


def _main_family_validate(argv: list[str]) -> int:
    args = _family_validate_parser().parse_args(argv)
    try:
        expected: dict[str, str] | None = None
        if args.expected_member_digest is not None:
            expected = {}
            for split, digest in args.expected_member_digest:
                if split in expected:
                    raise RiskDataContractError(
                        f"duplicate expected member digest for {split}"
                    )
                expected[split] = digest
        loaded = load_risk_dataset_family(
            args.family_root,
            expected_member_digests=expected,
        )
    except (RiskDataContractError, FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_result(_family_result(loaded))
    return 0


def _main_dataset(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    try:
        seal_root = publish_risk_dataset_seal(
            args.output_dir,
            collection_root=args.collection_root,
            base_config_path=args.base_config,
            split_provenance_path=args.split_provenance,
            expected_split=args.split,
            expected_collection_handoff_sha256=(
                args.expected_collection_handoff_sha256
            ),
            sidecar_root=args.sidecar_root,
        )
        loaded = load_risk_dataset_seal(
            seal_root,
            collection_root=args.collection_root,
            expected_split=args.split,
            sidecar_root=args.sidecar_root,
        )
    except (RiskDataContractError, FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = {
        "risk_dataset_manifest_digest": loaded.risk_dataset_manifest_digest,
        "sample_count": loaded.sample_count,
        "seal_root": str(seal_root),
        "shard_count": len(loaded.shards),
        "split": loaded.split,
        "status": "complete",
    }
    occupancy_sidecars = loaded.manifest.get("occupancy_sidecars")
    if isinstance(occupancy_sidecars, dict):
        result["occupancy_sidecar_collection_digest_sha256"] = (
            occupancy_sidecars["collection_digest_sha256"]
        )
    _print_result(result)
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "family-publish":
        return _main_family_publish(argv[1:])
    if argv and argv[0] == "family-validate":
        return _main_family_validate(argv[1:])
    return _main_dataset(argv)


if __name__ == "__main__":
    raise SystemExit(main())
