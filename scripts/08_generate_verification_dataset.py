#!/usr/bin/env python3
"""Generate SOP13 smoke data or a resumable finalized-SOP5 release."""

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

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import SCHEMA_VERSION, build_grid_spec
from src.datasets.verification_dataloader import write_verification_shard
from src.generation.verification_gt import load_verification_gt_config
from src.generation.verification_pipeline import (
    VERIFICATION_PIPELINE_VERSION,
    build_verification_toy_input,
    generate_verification_group,
)
from src.generation.verification_release import (
    VerificationReleaseRequest,
    publish_verification_release,
)
from src.planning.verification_actions import load_verification_actions
from src.utils.config import load_config


GENERATION_VERSION = "verification_dataset_cli_v5"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("toy", "sop05-final"),
        required=True,
    )
    parser.add_argument("--split", choices=("train", "calibration", "val", "test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--actions-config", type=Path, required=True)
    parser.add_argument("--gt-config", type=Path, required=True)
    parser.add_argument("--max-replan-candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-family",
        choices=("natural", "a_supplement"),
    )
    parser.add_argument(
        "--source-mode",
        choices=("complete_mother", "partial_m6_reconstruction"),
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-cache-root", type=Path)
    parser.add_argument("--final-scenario-root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--groups-per-shard", type=int, default=16)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--sop03-lineage-root", type=Path)
    parser.add_argument("--long40-human-artifact", type=Path)
    parser.add_argument("--base-state-start", type=int)
    parser.add_argument("--max-base-states", type=int)
    parser.add_argument("--generator-config", type=Path)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_replan_candidates <= 0:
        raise ValueError("max_replan_candidates must be positive")
    if args.mode == "sop05-final":
        required = (
            "split",
            "source_family",
            "source_mode",
            "source_root",
            "final_scenario_root",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ValueError(
                "required for sop05-final: "
                + ", ".join(name.replace("_", "-") for name in missing)
            )
        for name in ("workers", "groups_per_shard"):
            value = getattr(args, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if args.max_tasks is not None and args.max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        allocated = os.environ.get("SLURM_CPUS_PER_TASK")
        if allocated is not None and args.workers > int(allocated):
            raise ValueError("workers must not exceed SLURM_CPUS_PER_TASK")
        partial_values = (
            args.sop03_lineage_root,
            args.long40_human_artifact,
            args.base_state_start,
            args.max_base_states,
            args.config,
            args.generator_config,
        )
        if args.source_mode == "partial_m6_reconstruction":
            if any(value is None for value in partial_values):
                raise ValueError(
                    "partial_m6_reconstruction requires lineage, Long40, "
                    "base-state bounds, base config, and generator config"
                )
        elif any(value is not None for value in partial_values):
            raise ValueError(
                "partial reconstruction arguments require "
                "--source-mode partial_m6_reconstruction"
            )
        if args.sample_count is not None:
            raise ValueError(
                "sop05-final uses one fixed task per mother; use --max-tasks"
            )
        return
    if args.config is None or args.sample_count is None:
        raise ValueError("config and sample_count are required for smoke modes")
    if (
        isinstance(args.sample_count, bool)
        or not 10 <= args.sample_count <= 100
        or args.sample_count % 6 != 0
    ):
        raise ValueError(
            "sample_count must be in [10,100] and divisible by the six-action group size"
        )
    if args.mode == "toy" and args.split is not None:
        raise ValueError("toy mode does not accept --split")


def _resolved_split(args: argparse.Namespace) -> str:
    if args.mode == "toy":
        return "train"
    assert args.split in {"train", "calibration", "val", "test"}
    return str(args.split)


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
    digest.update(b"verification-smoke-implementation-v4\0")
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
    split: str,
    scientific_status: str,
    sample_count: int,
    group_count: int,
    shard_semantic_digest: str,
    sampled_child_world_ids: Sequence[str],
    seed: int,
    implementation_digest: str,
) -> str:
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "generation_version": GENERATION_VERSION,
        "pipeline_version": VERIFICATION_PIPELINE_VERSION,
        "mode": mode,
        "split": split,
        "scientific_status": scientific_status,
        "sample_count": sample_count,
        "group_count": group_count,
        "shard_semantic_digest": shard_semantic_digest,
        "sampled_child_world_ids": list(sampled_child_world_ids),
        "seed": seed,
        "implementation_digest_sha256": implementation_digest,
    }
    digest = hashlib.sha256()
    digest.update(b"verification-collection-semantic-v3\0")
    digest.update(_canonical_bytes(semantic))
    return digest.hexdigest()


def _write_collection(
    args: argparse.Namespace,
    *,
    samples,
    grid,
    library,
    scientific_status: str,
    split: str,
    source_summary: dict[str, object],
    sampled_child_world_ids: Sequence[str],
    elapsed_seconds: float,
) -> dict[str, object]:
    observed_splits = {sample.split for sample in samples}
    if observed_splits != {split}:
        raise ValueError("generated samples differ from the declared collection split")
    if (
        len(sampled_child_world_ids) != args.sample_count // 6
        or any(
            not isinstance(world_id, str) or not world_id
            for world_id in sampled_child_world_ids
        )
    ):
        raise ValueError(
            "sampled_child_world_ids must contain one ID per ranking group"
        )
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
            split=split,
            scientific_status=scientific_status,
            sample_count=args.sample_count,
            group_count=args.sample_count // 6,
            shard_semantic_digest=str(shard_summary["semantic_digest"]),
            sampled_child_world_ids=sampled_child_world_ids,
            seed=args.seed,
            implementation_digest=implementation_digest,
        )
        if args.mode == "toy":
            limitations = [
                "toy data are not paper-scale evidence",
                "validation/test performance and cross-split leakage are not proven",
            ]
        elif split == "train":
            limitations = [
                "train-only smoke data are not paper-scale evidence",
                "validation/test performance and cross-split leakage are not proven",
            ]
        else:
            limitations = [
                "held-out smoke data are not paper-scale evidence",
                "cross-split leakage is not proven by this per-split output",
            ]
        report: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "generation_version": GENERATION_VERSION,
            "pipeline_version": VERIFICATION_PIPELINE_VERSION,
            "mode": args.mode,
            "scientific_status": scientific_status,
            "split": split,
            "sample_count": args.sample_count,
            "group_count": args.sample_count // 6,
            "value_semantics": "one_sop05_sampled_child_per_group",
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
            "sampled_child_world_ids": list(sampled_child_world_ids),
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
            "split": split,
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
    split = _resolved_split(args)
    if args.mode == "sop05-final":
        assert args.source_family is not None
        assert args.source_mode is not None
        assert args.source_root is not None
        assert args.final_scenario_root is not None
        request = VerificationReleaseRequest(
            source_family=args.source_family,
            source_mode=args.source_mode,
            source_root=args.source_root,
            source_cache_root=args.source_cache_root,
            final_scenario_root=args.final_scenario_root,
            split=split,
            output_dir=args.output_dir,
            actions_config_path=args.actions_config,
            gt_config_path=args.gt_config,
            workers=args.workers,
            groups_per_shard=args.groups_per_shard,
            max_replan_candidates=args.max_replan_candidates,
            max_tasks=args.max_tasks,
            sop03_root=args.sop03_lineage_root,
            long40_human_artifact=args.long40_human_artifact,
            base_state_start=args.base_state_start,
            max_base_states=args.max_base_states,
            base_config_path=args.config,
            generator_config_path=args.generator_config,
        )

        def progress(completed: int, total: int, reused: bool) -> None:
            if completed == total or completed % 10 == 0:
                print(
                    json.dumps(
                        {
                            "status": "running",
                            "completed_shards": completed,
                            "total_shards": total,
                            "last_shard_reused": reused,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )

        result = publish_verification_release(
            request,
            progress_callback=progress,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output_dir": str(result.output_dir),
                    "split": result.split,
                    "task_count": result.task_count,
                    "accepted_group_count": result.accepted_group_count,
                    "rejected_task_count": result.rejected_task_count,
                    "sample_count": result.sample_count,
                    "shard_count": result.shard_count,
                    "reused_shard_count": result.reused_shard_count,
                    "manifest_digest": result.manifest_digest,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable output: {args.output_dir}"
        )
    started = time.perf_counter()
    config = load_config(args.config)
    action_library = load_verification_actions(args.actions_config)
    gt_config = load_verification_gt_config(args.gt_config)

    group_count = args.sample_count // 6
    groups = []
    sampled_child_world_ids: list[str] = []
    if args.mode == "toy":
        toy_config = None
        for group_index in range(group_count):
            source, candidate_config = build_verification_toy_input(
                config,
                action_library=action_library,
                group_index=group_index,
            )
            if toy_config is None:
                toy_config = candidate_config
            result = generate_verification_group(
                source,
                base_config=candidate_config,
                action_library=action_library,
                gt_config=gt_config,
                max_replan_candidates=args.max_replan_candidates,
            )
            groups.append(result.samples)
            sampled_child_world_ids.append(result.sampled_child_world_id)
        assert toy_config is not None
        grid = build_grid_spec(toy_config)
        scientific_status = "toy_smoke_only"
        source_summary = {"mode": "toy", "cross_split_status": "NOT_PROVEN"}
    samples = tuple(sample for group in groups for sample in group)
    if len(samples) != args.sample_count:
        raise RuntimeError("generated sample count differs from the exact request")
    report = _write_collection(
        args,
        samples=samples,
        grid=grid,
        library=action_library,
        scientific_status=scientific_status,
        split=split,
        source_summary=source_summary,
        sampled_child_world_ids=sampled_child_world_ids,
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
