#!/usr/bin/env python3
"""Generate bounded SOP13 toy or audited-train verification smoke data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import SCHEMA_VERSION, build_grid_spec
from src.datasets.verification_dataloader import write_verification_shard
from src.datasets.verification_sources import (
    load_joined_source_events,
    load_verification_source_index,
)
from src.generation.scenario_bank import load_scenario_bank_config
from src.generation.verification_gt import load_verification_gt_config
from src.generation.verification_pipeline import (
    VERIFICATION_PIPELINE_VERSION,
    build_real_verification_input,
    build_verification_toy_input,
    generate_verification_group,
)
from src.planning.verification_actions import load_verification_actions
from src.utils.config import load_config
from src.utils.seeding import derive_seed


GENERATION_VERSION = "verification_dataset_cli_v1"
_REAL_REQUIRED_ARGS = (
    "sop03_root",
    "sop04_root",
    "sop05_batch_handoff",
    "sop07_collection_handoff",
    "expected_sop05_batch_digest",
    "expected_sop07_collection_digest",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("toy", "sop05-train"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--actions-config", type=Path, required=True)
    parser.add_argument("--gt-config", type=Path, required=True)
    parser.add_argument("--bank-size", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument(
        "--posterior-mode", choices=("exact", "soft"), default="exact"
    )
    parser.add_argument("--posterior-temperature", type=float)
    parser.add_argument("--max-replan-candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checksum-workers", type=int, default=2)
    parser.add_argument("--sop03-root", type=Path)
    parser.add_argument("--sop04-root", type=Path)
    parser.add_argument("--sop05-batch-handoff", type=Path)
    parser.add_argument("--sop07-collection-handoff", type=Path)
    parser.add_argument("--expected-sop05-batch-digest")
    parser.add_argument("--expected-sop07-collection-digest")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if (
        isinstance(args.sample_count, bool)
        or not 10 <= args.sample_count <= 100
        or args.sample_count % 6 != 0
    ):
        raise ValueError(
            "sample_count must be in [10,100] and divisible by the six-action group size"
        )
    if args.max_replan_candidates <= 0:
        raise ValueError("max_replan_candidates must be positive")
    if args.checksum_workers <= 0:
        raise ValueError("checksum_workers must be positive")
    if args.posterior_mode == "exact" and args.posterior_temperature is not None:
        raise ValueError("exact posterior does not accept --posterior-temperature")
    if args.mode == "sop05-train":
        missing = [name for name in _REAL_REQUIRED_ARGS if getattr(args, name) is None]
        if missing:
            raise ValueError(
                "required for sop05-train: " + ", ".join(name.replace("_", "-") for name in missing)
            )


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _implementation_digest(root: Path, config_paths: Sequence[Path]) -> str:
    relative_files = (
        Path("scripts/08_generate_verification_dataset.py"),
        Path("src/generation/verification_pipeline.py"),
        Path("src/generation/verification_gt.py"),
        Path("src/datasets/verification_dataset.py"),
        Path("src/datasets/verification_dataloader.py"),
    )
    digest = hashlib.sha256()
    digest.update(b"verification-smoke-implementation-v1\0")
    for path in (*[root / value for value in relative_files], *config_paths):
        payload = path.read_bytes()
        label = str(path.name).encode("utf-8")
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _collection_digest(
    *,
    mode: str,
    scientific_status: str,
    sample_count: int,
    group_count: int,
    shard_semantic_digest: str,
    scenario_bank_digests: Sequence[str],
    seed: int,
    bank_size: int,
    posterior_mode: str,
    implementation_digest: str,
) -> str:
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "generation_version": GENERATION_VERSION,
        "pipeline_version": VERIFICATION_PIPELINE_VERSION,
        "mode": mode,
        "scientific_status": scientific_status,
        "sample_count": sample_count,
        "group_count": group_count,
        "shard_semantic_digest": shard_semantic_digest,
        "scenario_bank_digests": list(scenario_bank_digests),
        "seed": seed,
        "bank_size": bank_size,
        "posterior_mode": posterior_mode,
        "implementation_digest_sha256": implementation_digest,
    }
    digest = hashlib.sha256()
    digest.update(b"verification-collection-semantic-v1\0")
    digest.update(_canonical_bytes(semantic))
    return digest.hexdigest()


def _write_collection(
    args: argparse.Namespace,
    *,
    samples,
    grid,
    library,
    scientific_status: str,
    source_summary: dict[str, object],
    scenario_bank_digests: Sequence[str],
    posterior_temperature: float | None,
    elapsed_seconds: float,
) -> dict[str, object]:
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        write_verification_shard(
            tuple(samples),
            staging / "shard-00000",
            grid=grid,
            library=library,
            shard_index=0,
            expected_sample_count=args.sample_count,
        )
        shard_summary = json.loads(
            (staging / "shard-00000" / "summary.json").read_text(encoding="utf-8")
        )
        implementation_digest = _implementation_digest(
            ROOT, (args.config, args.actions_config, args.gt_config)
        )
        collection_digest = _collection_digest(
            mode=args.mode,
            scientific_status=scientific_status,
            sample_count=args.sample_count,
            group_count=args.sample_count // 6,
            shard_semantic_digest=str(shard_summary["semantic_digest"]),
            scenario_bank_digests=scenario_bank_digests,
            seed=args.seed,
            bank_size=args.bank_size,
            posterior_mode=args.posterior_mode,
            implementation_digest=implementation_digest,
        )
        limitations = [
            (
                "toy data are not paper-scale evidence"
                if args.mode == "toy"
                else "train-only smoke data are not paper-scale evidence"
            ),
            "validation/test performance and cross-split leakage are not proven",
        ]
        report: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "generation_version": GENERATION_VERSION,
            "pipeline_version": VERIFICATION_PIPELINE_VERSION,
            "mode": args.mode,
            "scientific_status": scientific_status,
            "split": "train",
            "sample_count": args.sample_count,
            "group_count": args.sample_count // 6,
            "bank_size": args.bank_size,
            "posterior_mode": args.posterior_mode,
            "posterior_temperature": posterior_temperature,
            "max_replan_candidates": args.max_replan_candidates,
            "seed": args.seed,
            "grid": {
                "height": grid.height,
                "width": grid.width,
                "history_steps": grid.history_steps,
                "future_steps": grid.future_steps,
                "resolution_m": grid.resolution_m,
            },
            "implementation_digest_sha256": implementation_digest,
            "shard_semantic_digest": shard_summary["semantic_digest"],
            "collection_semantic_digest": collection_digest,
            "scenario_bank_digests": list(scenario_bank_digests),
            "source": source_summary,
            "elapsed_seconds": elapsed_seconds,
            "limitations": limitations,
        }
        report_bytes = _canonical_bytes(report)
        (staging / "generation_report.json").write_bytes(report_bytes)
        handoff = {
            "schema_version": SCHEMA_VERSION,
            "handoff_version": "verification_collection_handoff_v1",
            "collection_state": "complete",
            "scientific_status": scientific_status,
            "split": "train",
            "sample_count": args.sample_count,
            "group_count": args.sample_count // 6,
            "collection_semantic_digest": collection_digest,
            "generation_report_sha256": _sha256(report_bytes),
            "shards": [
                {
                    "shard_index": 0,
                    "relative_root": "shard-00000",
                    "sample_count": args.sample_count,
                    "semantic_digest": shard_summary["semantic_digest"],
                }
            ],
            "limitations": limitations,
        }
        (staging / "collection_complete_handoff.json").write_bytes(
            _canonical_bytes(handoff)
        )
        os.rename(staging, output)
        return report
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable output: {args.output_dir}"
        )
    started = time.perf_counter()
    config = load_config(args.config)
    action_library = load_verification_actions(args.actions_config)
    gt_config = load_verification_gt_config(args.gt_config)
    scenario_config = load_scenario_bank_config(args.gt_config)
    posterior_temperature = args.posterior_temperature
    if args.posterior_mode == "soft" and posterior_temperature is None:
        from src.generation.observation_posterior import (
            load_observation_posterior_config,
        )

        posterior_temperature = load_observation_posterior_config(
            args.gt_config
        ).default_temperature

    group_count = args.sample_count // 6
    groups = []
    scenario_digests: list[str] = []
    if args.mode == "toy":
        toy_config = None
        for group_index in range(group_count):
            source, candidate_config = build_verification_toy_input(
                config, group_index=group_index
            )
            if toy_config is None:
                toy_config = candidate_config
            result = generate_verification_group(
                source,
                base_config=candidate_config,
                action_library=action_library,
                gt_config=gt_config,
                scenario_config=scenario_config,
                bank_size=args.bank_size,
                posterior_mode=args.posterior_mode,
                posterior_temperature=posterior_temperature,
                seed=derive_seed(args.seed, "toy-verification-group", group_index),
                max_replan_candidates=args.max_replan_candidates,
            )
            groups.append(result.samples)
            scenario_digests.append(result.scenario_bank_digest)
        assert toy_config is not None
        grid = build_grid_spec(toy_config)
        scientific_status = "toy_smoke_only"
        source_summary = {"mode": "toy", "cross_split_status": "NOT_PROVEN"}
    else:
        index = load_verification_source_index(
            args.sop05_batch_handoff,
            args.sop07_collection_handoff,
            expected_sop05_batch_digest=args.expected_sop05_batch_digest,
            expected_sop07_collection_digest=args.expected_sop07_collection_digest,
        )
        grid = build_grid_spec(config)
        bundle = load_joined_source_events(
            index,
            sop03_root=args.sop03_root,
            sop04_root=args.sop04_root,
            grid=grid,
            event_count=group_count,
            seed=args.seed,
            checksum_workers=args.checksum_workers,
        )
        for group_index, event in enumerate(bundle.events):
            source = build_real_verification_input(
                event,
                base_config=config,
                sop05_batch_digest=index.sop05_batch_digest,
                sop07_collection_digest=index.sop07_collection_digest,
            )
            result = generate_verification_group(
                source,
                base_config=config,
                action_library=action_library,
                gt_config=gt_config,
                scenario_config=scenario_config,
                bank_size=args.bank_size,
                posterior_mode=args.posterior_mode,
                posterior_temperature=posterior_temperature,
                seed=derive_seed(args.seed, "real-verification-group", group_index),
                max_replan_candidates=args.max_replan_candidates,
            )
            groups.append(result.samples)
            scenario_digests.append(result.scenario_bank_digest)
        scientific_status = index.scientific_status
        source_summary = {
            "mode": "sop05-train",
            "sop05_batch_digest": index.sop05_batch_digest,
            "sop07_collection_digest": index.sop07_collection_digest,
            "cross_split_status": index.global_cross_split_leakage,
            "loaded_sop05_publication_digests": list(
                bundle.loaded_sop05_publication_digests
            ),
            "selected_event_ids": [
                item.event.generated_event_id for item in bundle.events
            ],
        }

    samples = tuple(sample for group in groups for sample in group)
    if len(samples) != args.sample_count:
        raise RuntimeError("generated sample count differs from the exact request")
    report = _write_collection(
        args,
        samples=samples,
        grid=grid,
        library=action_library,
        scientific_status=scientific_status,
        source_summary=source_summary,
        scenario_bank_digests=scenario_digests,
        posterior_temperature=posterior_temperature,
        elapsed_seconds=time.perf_counter() - started,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "sample_count": report["sample_count"],
                "scientific_status": report["scientific_status"],
                "collection_semantic_digest": report[
                    "collection_semantic_digest"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
