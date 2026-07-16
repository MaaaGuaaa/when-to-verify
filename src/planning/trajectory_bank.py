"""Build and serialize the reusable canonical SOP-04 trajectory bank."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from src.contracts import ARRAY_DTYPE, LocalTrajectory, SCHEMA_VERSION

from .query_maps import build_local_trajectory
from .trajectory_filters import filter_trajectory_candidates
from .trajectory_sampler import CandidateRollout, sample_candidate_rollouts


@dataclass(frozen=True)
class TrajectoryBank:
    """Canonical main-distribution trajectories and audit statistics."""

    trajectories: tuple[LocalTrajectory, ...]
    summary: dict[str, object]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"trajectory metadata must not contain {value}")


def _materialize_candidate(
    candidate: CandidateRollout,
    *,
    config: dict,
    braking_deceleration_mps2: float,
) -> LocalTrajectory:
    return build_local_trajectory(
        candidate,
        config,
        braking_deceleration_mps2=braking_deceleration_mps2,
        task_cost=0.0,
    )


def _validate_trajectory(
    trajectory: LocalTrajectory,
    *,
    steps: int,
    height: int,
    width: int,
    braking_deceleration_mps2: float,
) -> None:
    if not isinstance(trajectory.trajectory_id, str) or not trajectory.trajectory_id:
        raise ValueError("trajectory_id must be a non-empty string")
    arrays = {
        "poses": (trajectory.poses, (steps, 3)),
        "controls": (trajectory.controls, (steps, 2)),
        "swept_mask": (trajectory.swept_mask, (height, width)),
        "tta_map": (trajectory.tta_map, (height, width)),
        "braking_map": (trajectory.braking_map, (height, width)),
        "centerline_map": (trajectory.centerline_map, (height, width)),
    }
    for name, (array, expected_shape) in arrays.items():
        if array.shape != expected_shape:
            raise ValueError(f"{name} shape mismatch")
        if array.dtype != ARRAY_DTYPE:
            raise TypeError(f"{name} dtype must be {ARRAY_DTYPE}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN/Inf")
    swept = trajectory.swept_mask.astype(bool)
    if not np.all(np.isin(trajectory.swept_mask, (0.0, 1.0))):
        raise ValueError("swept_mask must be binary")
    if not np.all(np.isin(trajectory.centerline_map, (0.0, 1.0))):
        raise ValueError("centerline_map must be binary")
    if not np.all(trajectory.tta_map[~swept] == -1.0):
        raise ValueError("tta_map must be -1 outside the swept volume")
    if not np.all(trajectory.tta_map[swept] >= 0.0):
        raise ValueError("tta_map must be non-negative inside the swept volume")
    if not np.all(trajectory.braking_map[~swept] == 0.0):
        raise ValueError("braking_map must be zero outside the swept volume")
    if not math.isfinite(float(trajectory.task_cost)):
        raise ValueError("task_cost must be finite")
    if (
        float(trajectory.metadata.get("braking_deceleration_mps2", math.nan))
        != braking_deceleration_mps2
    ):
        raise ValueError("trajectory braking deceleration metadata mismatch")


def _validate_bank(bank: TrajectoryBank) -> None:
    if not bank.trajectories:
        raise ValueError("trajectory bank must not be empty")
    ids = [trajectory.trajectory_id for trajectory in bank.trajectories]
    if len(ids) != len(set(ids)):
        raise ValueError("trajectory ids must be unique")
    steps = int(bank.summary["trajectory_steps"])
    height = int(bank.summary["grid_height"])
    width = int(bank.summary["grid_width"])
    deceleration = float(bank.summary["braking_deceleration_mps2"])
    for trajectory in bank.trajectories:
        _validate_trajectory(
            trajectory,
            steps=steps,
            height=height,
            width=width,
            braking_deceleration_mps2=deceleration,
        )
    if int(bank.summary["accepted_count"]) != len(bank.trajectories):
        raise ValueError("trajectory count does not match summary")


def build_trajectory_bank(
    config: dict,
    *,
    braking_deceleration_mps2: float,
    workers: int = 1,
) -> TrajectoryBank:
    """Build the 20 forward primitives plus stop, parallelizing query maps."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    braking_deceleration_mps2 = float(braking_deceleration_mps2)
    if (
        not math.isfinite(braking_deceleration_mps2)
        or braking_deceleration_mps2 <= 0.0
    ):
        raise ValueError("braking deceleration must be finite and positive")

    candidates = sample_candidate_rollouts(config, reverse_stress=False)
    report = filter_trajectory_candidates(candidates, config)
    if not report.accepted:
        raise ValueError("no trajectory candidate passed the frozen filters")
    worker_count = min(workers, len(report.accepted))
    build_one = partial(
        _materialize_candidate,
        config=config,
        braking_deceleration_mps2=braking_deceleration_mps2,
    )
    if worker_count == 1:
        trajectories = tuple(build_one(candidate) for candidate in report.accepted)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            trajectories = tuple(executor.map(build_one, report.accepted))

    config_payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    summary: dict[str, object] = {
        "candidate_count": len(candidates),
        "accepted_count": len(trajectories),
        "rejected_count": len(report.rejected),
        "acceptance_rate": report.acceptance_rate,
        "minimum_acceptance_rate": 0.70,
        "meets_minimum_acceptance_rate": report.acceptance_rate >= 0.70,
        "rejection_counts": report.rejection_counts,
        "workers_requested": workers,
        "workers_used": worker_count,
        "braking_deceleration_mps2": braking_deceleration_mps2,
        "state_specific_filtering": False,
        "state_specific_filtering_reason": (
            "canonical bank has no BaseState-specific static occupancy or "
            "acceleration contract"
        ),
        "reverse_stress": False,
        "task_cost_semantics": "unassigned_canonical_bank_zero",
        "trajectory_steps": int(config["bev"]["future_steps"]),
        "grid_height": int(config["bev"]["size"]),
        "grid_width": int(config["bev"]["size"]),
        "array_dtype": str(np.dtype(ARRAY_DTYPE)),
        "config_blake2b_128": hashlib.blake2b(
            config_payload, digest_size=16
        ).hexdigest(),
    }
    bank = TrajectoryBank(trajectories=trajectories, summary=summary)
    _validate_bank(bank)
    return bank


def _save_trajectory_bank(bank: TrajectoryBank, path: Path) -> Path:
    _validate_bank(bank)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "summary": bank.summary,
        "trajectory_ids": [
            trajectory.trajectory_id for trajectory in bank.trajectories
        ],
        "trajectory_metadata": [
            trajectory.metadata for trajectory in bank.trajectories
        ],
    }
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            poses=np.stack(
                [trajectory.poses for trajectory in bank.trajectories]
            ),
            controls=np.stack(
                [trajectory.controls for trajectory in bank.trajectories]
            ),
            swept_masks=np.stack(
                [trajectory.swept_mask for trajectory in bank.trajectories]
            ),
            tta_maps=np.stack(
                [trajectory.tta_map for trajectory in bank.trajectories]
            ),
            braking_maps=np.stack(
                [trajectory.braking_map for trajectory in bank.trajectories]
            ),
            centerline_maps=np.stack(
                [trajectory.centerline_map for trajectory in bank.trajectories]
            ),
            task_costs=np.asarray(
                [trajectory.task_cost for trajectory in bank.trajectories],
                dtype=ARRAY_DTYPE,
            ),
            meta_json=np.asarray(
                json.dumps(metadata, sort_keys=True, allow_nan=False)
            ),
        )
    temporary.replace(path)
    return path


def load_trajectory_bank(path: str | Path) -> TrajectoryBank:
    """Load and validate a numeric trajectory-bank NPZ without pickle."""
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(
            str(payload["meta_json"]),
            parse_constant=_reject_json_constant,
        )
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("trajectory bank schema_version mismatch")
        poses = payload["poses"].copy()
        controls = payload["controls"].copy()
        swept_masks = payload["swept_masks"].copy()
        tta_maps = payload["tta_maps"].copy()
        braking_maps = payload["braking_maps"].copy()
        centerline_maps = payload["centerline_maps"].copy()
        task_costs = payload["task_costs"].copy()
    trajectory_ids = metadata["trajectory_ids"]
    trajectory_metadata = metadata["trajectory_metadata"]
    count = len(trajectory_ids)
    if not all(
        array.shape[0] == count
        for array in (
            poses,
            controls,
            swept_masks,
            tta_maps,
            braking_maps,
            centerline_maps,
            task_costs,
        )
    ) or len(trajectory_metadata) != count:
        raise ValueError("trajectory bank arrays and metadata do not align")
    trajectories = tuple(
        LocalTrajectory(
            trajectory_id=str(trajectory_ids[index]),
            poses=poses[index],
            controls=controls[index],
            swept_mask=swept_masks[index],
            tta_map=tta_maps[index],
            braking_map=braking_maps[index],
            centerline_map=centerline_maps[index],
            task_cost=float(task_costs[index]),
            metadata=trajectory_metadata[index],
        )
        for index in range(count)
    )
    bank = TrajectoryBank(
        trajectories=trajectories,
        summary=metadata["summary"],
    )
    _validate_bank(bank)
    return bank


def write_trajectory_bank(
    bank: TrajectoryBank,
    output_dir: str | Path,
    *,
    provenance: dict[str, object],
) -> dict[str, Path]:
    """Atomically write the bank, manifest, summary, and provenance."""
    _validate_bank(bank)
    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_name(f".{output_path.name}.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        bank_path = _save_trajectory_bank(
            bank, staging / "trajectory_bank.npz"
        )
        rows = [
            {
                "schema_version": SCHEMA_VERSION,
                "array_index": index,
                "trajectory_id": trajectory.trajectory_id,
                "is_stop": bool(trajectory.metadata["is_stop"]),
                "is_reverse": bool(trajectory.metadata["is_reverse"]),
                "v_mps": float(trajectory.metadata["v"]),
                "omega_radps": float(trajectory.metadata["omega"]),
                "trajectory_steps": int(trajectory.poses.shape[0]),
                "query_map_shape": list(trajectory.swept_mask.shape),
                "task_cost": float(trajectory.task_cost),
            }
            for index, trajectory in enumerate(bank.trajectories)
        ]
        manifest = "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            **bank.summary,
            "provenance": provenance,
        }
        (staging / "trajectory_manifest.jsonl").write_text(
            manifest, encoding="utf-8"
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        staging.replace(output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "directory": output_path,
        "bank": output_path / bank_path.name,
        "manifest": output_path / "trajectory_manifest.jsonl",
        "summary": output_path / "summary.json",
    }
