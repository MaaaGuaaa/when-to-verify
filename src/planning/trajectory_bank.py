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

from src.contracts import (
    ARRAY_DTYPE,
    POSE_TIME_LAYOUT_VERSION,
    LocalTrajectory,
    SCHEMA_VERSION,
)

from .differential_drive import (
    rollout_constant_control,
)
from .query_maps import build_local_trajectory
from .trajectory_filters import filter_trajectory_candidates
from .trajectory_sampler import CandidateRollout, sample_candidate_rollouts

TRAJECTORY_BANK_VERSION = "sop04_audited_bank_v2"


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
    dt_s: float,
    grid_resolution_m: float,
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
    if not np.all(trajectory.tta_map[swept] <= steps * dt_s + 1e-6):
        raise ValueError("tta_map exceeds the future trajectory horizon")
    if not math.isfinite(float(trajectory.task_cost)):
        raise ValueError("task_cost must be finite")
    if (
        float(trajectory.metadata.get("braking_deceleration_mps2", math.nan))
        != braking_deceleration_mps2
    ):
        raise ValueError("trajectory braking deceleration metadata mismatch")
    metadata = trajectory.metadata
    if metadata.get("pose_time_layout_version") != POSE_TIME_LAYOUT_VERSION:
        raise ValueError("trajectory pose time layout version mismatch")
    expected_time_fields = {
        "first_pose_time_s": dt_s,
        "last_pose_time_s": steps * dt_s,
        "dt_s": dt_s,
    }
    for name, expected in expected_time_fields.items():
        try:
            actual = float(metadata[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"trajectory {name} metadata mismatch") from error
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"trajectory {name} metadata mismatch")
    if int(metadata.get("trajectory_steps", -1)) != steps:
        raise ValueError("trajectory trajectory_steps metadata mismatch")
    for name in ("is_stop", "is_reverse"):
        if not isinstance(metadata.get(name), bool):
            raise ValueError(f"trajectory {name} metadata must be boolean")
    expected_controls = np.repeat(trajectory.controls[:1], steps, axis=0)
    if not np.array_equal(trajectory.controls, expected_controls):
        raise ValueError("canonical trajectory controls must be constant")
    expected_poses, _ = rollout_constant_control(
        v=float(trajectory.controls[0, 0]),
        omega=float(trajectory.controls[0, 1]),
        dt_s=dt_s,
        steps=steps,
    )
    if not np.allclose(
        trajectory.poses, expected_poses, rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            "trajectory poses do not match future endpoint control semantics"
        )
    v = float(trajectory.controls[0, 0])
    omega = float(trajectory.controls[0, 1])
    if not math.isclose(
        float(metadata.get("v", math.nan)), v, rel_tol=0.0, abs_tol=1e-12
    ) or not math.isclose(
        float(metadata.get("omega", math.nan)),
        omega,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("trajectory control metadata mismatch")
    is_stationary = bool(np.all(trajectory.controls == 0.0))
    if metadata["is_stop"] != is_stationary:
        raise ValueError("trajectory stop metadata mismatch")
    if metadata["is_reverse"] != (v < 0.0):
        raise ValueError("trajectory reverse metadata mismatch")
    origin_row = height // 2
    origin_column = width // 2
    if not swept[origin_row, origin_column]:
        raise ValueError("swept_mask must include the current origin footprint")
    if trajectory.tta_map[origin_row, origin_column] != 0.0:
        raise ValueError("current origin footprint must have zero arrival time")
    if trajectory.centerline_map[origin_row, origin_column] != 1.0:
        raise ValueError("centerline_map must include the current origin")
    if not np.all(trajectory.centerline_map <= trajectory.swept_mask):
        raise ValueError("centerline_map must be contained in swept_mask")
    x_min = -0.5 * width * grid_resolution_m
    y_min = -0.5 * height * grid_resolution_m
    pose_columns = np.floor(
        (trajectory.poses[:, 0].astype(np.float64) - x_min)
        / grid_resolution_m
    ).astype(np.int64)
    pose_rows = np.floor(
        (trajectory.poses[:, 1].astype(np.float64) - y_min)
        / grid_resolution_m
    ).astype(np.int64)
    if (
        np.any(pose_rows < 0)
        or np.any(pose_rows >= height)
        or np.any(pose_columns < 0)
        or np.any(pose_columns >= width)
    ):
        raise ValueError("trajectory pose center lies outside the query-map grid")
    if not np.all(
        trajectory.centerline_map[pose_rows, pose_columns] == 1.0
    ):
        raise ValueError("centerline_map must cover every future pose center")
    if not np.all(trajectory.swept_mask[pose_rows, pose_columns] == 1.0):
        raise ValueError("swept_mask must cover every future pose center")
    pose_center_tta = trajectory.tta_map[pose_rows, pose_columns]
    if not np.all(pose_center_tta >= 0.0) or np.any(
        np.diff(pose_center_tta) < -1e-6
    ):
        raise ValueError("future pose-center arrival times must be nondecreasing")
    pose_endpoint_times = (
        np.arange(steps, dtype=np.float64) + 1.0
    ) * dt_s
    if np.any(pose_center_tta > pose_endpoint_times + 1e-6):
        raise ValueError("pose-center arrival cannot follow its future endpoint")
    stopping_distance = v * v / (2.0 * braking_deceleration_mps2)
    expected_braking = abs(v) * trajectory.tta_map[swept] - stopping_distance
    if not np.allclose(
        trajectory.braking_map[swept],
        expected_braking,
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError("braking_map conflicts with control interval path distance")


def _validate_bank(bank: TrajectoryBank) -> None:
    if not bank.trajectories:
        raise ValueError("trajectory bank must not be empty")
    ids = [trajectory.trajectory_id for trajectory in bank.trajectories]
    if len(ids) != len(set(ids)):
        raise ValueError("trajectory ids must be unique")
    steps = int(bank.summary["trajectory_steps"])
    if bank.summary.get("trajectory_bank_version") != TRAJECTORY_BANK_VERSION:
        raise ValueError("trajectory bank version mismatch")
    if bank.summary.get("pose_time_layout_version") != POSE_TIME_LAYOUT_VERSION:
        raise ValueError("trajectory bank pose time layout version mismatch")
    dt_s = float(bank.summary["dt_s"])
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("trajectory bank dt_s must be finite and positive")
    expected_summary_times = {
        "first_pose_time_s": dt_s,
        "last_pose_time_s": steps * dt_s,
    }
    for name, expected in expected_summary_times.items():
        if not math.isclose(
            float(bank.summary[name]), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"trajectory bank {name} mismatch")
    height = int(bank.summary["grid_height"])
    width = int(bank.summary["grid_width"])
    grid_resolution_m = float(bank.summary["grid_resolution_m"])
    if not math.isfinite(grid_resolution_m) or grid_resolution_m <= 0.0:
        raise ValueError("grid_resolution_m must be finite and positive")
    deceleration = float(bank.summary["braking_deceleration_mps2"])
    for trajectory in bank.trajectories:
        _validate_trajectory(
            trajectory,
            steps=steps,
            height=height,
            width=width,
            braking_deceleration_mps2=deceleration,
            dt_s=dt_s,
            grid_resolution_m=grid_resolution_m,
        )
    if int(bank.summary["accepted_count"]) != len(bank.trajectories):
        raise ValueError("trajectory count does not match summary")


def _array_semantic_digest(array: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(array.dtype.str.encode("ascii"))
    hasher.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    hasher.update(np.ascontiguousarray(array).tobytes(order="C"))
    return hasher.hexdigest()


def trajectory_bank_semantic_digest(bank: TrajectoryBank) -> str:
    """Hash all scientific bank content while ignoring worker allocation."""
    _validate_bank(bank)
    scientific_summary = {
        key: value
        for key, value in bank.summary.items()
        if key not in {"workers_requested", "workers_used"}
    }
    payload = {
        "trajectory_bank_version": TRAJECTORY_BANK_VERSION,
        "pose_time_layout_version": POSE_TIME_LAYOUT_VERSION,
        "summary": scientific_summary,
        "trajectories": [
            {
                "trajectory_id": trajectory.trajectory_id,
                "metadata": trajectory.metadata,
                "task_cost": float(trajectory.task_cost),
                "arrays": {
                    name: _array_semantic_digest(getattr(trajectory, name))
                    for name in (
                        "poses",
                        "controls",
                        "swept_mask",
                        "tta_map",
                        "braking_map",
                        "centerline_map",
                    )
                },
            }
            for trajectory in bank.trajectories
        ],
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


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
        "trajectory_bank_version": TRAJECTORY_BANK_VERSION,
        "pose_time_layout_version": POSE_TIME_LAYOUT_VERSION,
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
        "first_pose_time_s": float(config["trajectories"]["dt_s"]),
        "last_pose_time_s": float(config["trajectories"]["horizon_s"]),
        "dt_s": float(config["trajectories"]["dt_s"]),
        "grid_height": int(config["bev"]["size"]),
        "grid_width": int(config["bev"]["size"]),
        "grid_resolution_m": float(config["bev"]["resolution_m"]),
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
        "trajectory_bank_version": TRAJECTORY_BANK_VERSION,
        "pose_time_layout_version": POSE_TIME_LAYOUT_VERSION,
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
        if metadata.get("trajectory_bank_version") != TRAJECTORY_BANK_VERSION:
            raise ValueError("trajectory bank version mismatch")
        if metadata.get("pose_time_layout_version") != POSE_TIME_LAYOUT_VERSION:
            raise ValueError("trajectory bank pose time layout version mismatch")
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_trajectory_bank_artifact(
    artifact_dir: str | Path,
    *,
    expected_bank: TrajectoryBank,
    provenance: dict[str, object],
    determinism_reference: TrajectoryBank,
) -> dict[str, Path]:
    """Reload and audit a persisted bank before it is handed downstream."""
    root = Path(artifact_dir)
    bank_path = root / "trajectory_bank.npz"
    manifest_path = root / "trajectory_manifest.jsonl"
    summary_path = root / "summary.json"
    reloaded = load_trajectory_bank(bank_path)
    expected_digest = trajectory_bank_semantic_digest(expected_bank)
    reloaded_digest = trajectory_bank_semantic_digest(reloaded)
    if reloaded_digest != expected_digest:
        raise ValueError("persisted trajectory bank differs from producer output")

    manifest_rows = [
        json.loads(line, parse_constant=_reject_json_constant)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(manifest_rows) != len(reloaded.trajectories):
        raise ValueError("trajectory manifest count mismatch")
    for index, (row, trajectory) in enumerate(
        zip(manifest_rows, reloaded.trajectories)
    ):
        expected_row = {
            "schema_version": SCHEMA_VERSION,
            "array_index": index,
            "trajectory_id": trajectory.trajectory_id,
            "is_stop": bool(trajectory.metadata["is_stop"]),
            "is_reverse": bool(trajectory.metadata["is_reverse"]),
            "v_mps": float(trajectory.metadata["v"]),
            "omega_radps": float(trajectory.metadata["omega"]),
            "trajectory_steps": int(trajectory.poses.shape[0]),
            "trajectory_bank_version": TRAJECTORY_BANK_VERSION,
            "pose_time_layout_version": POSE_TIME_LAYOUT_VERSION,
            "first_pose_time_s": float(
                trajectory.metadata["first_pose_time_s"]
            ),
            "last_pose_time_s": float(
                trajectory.metadata["last_pose_time_s"]
            ),
            "dt_s": float(trajectory.metadata["dt_s"]),
            "query_map_shape": list(trajectory.swept_mask.shape),
            "task_cost": float(trajectory.task_cost),
        }
        if row != expected_row:
            raise ValueError("trajectory manifest array alignment mismatch")

    persisted_summary = json.loads(
        summary_path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    if persisted_summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("trajectory summary schema_version mismatch")
    if persisted_summary.get("provenance") != provenance:
        raise ValueError("trajectory summary provenance mismatch")
    scientific_summary = {
        key: value
        for key, value in persisted_summary.items()
        if key not in {"schema_version", "provenance"}
    }
    if scientific_summary != reloaded.summary:
        raise ValueError("trajectory summary differs from bank metadata")

    core_paths = (bank_path, manifest_path, summary_path)
    checksums_path = root / "artifact_checksums.sha256"
    checksums_path.write_text(
        "".join(
            f"{_sha256_file(item)}  {item.name}\n" for item in core_paths
        ),
        encoding="utf-8",
    )
    checksum_rows = [
        line.split("  ", maxsplit=1)
        for line in checksums_path.read_text(encoding="utf-8").splitlines()
    ]
    expected_names = [item.name for item in core_paths]
    if [row[1] for row in checksum_rows] != expected_names:
        raise ValueError("artifact checksum manifest file list mismatch")
    for digest, name in checksum_rows:
        if _sha256_file(root / name) != digest:
            raise ValueError(f"artifact checksum mismatch: {name}")

    if determinism_reference is expected_bank:
        raise ValueError("audit requires an independently built trajectory bank")
    reference_digest = trajectory_bank_semantic_digest(determinism_reference)
    determinism_reference_exact_match = reference_digest == expected_digest
    if not determinism_reference_exact_match:
        raise ValueError("determinism reference does not match trajectory bank")
    worker_counts = {
        int(expected_bank.summary["workers_used"]),
        int(determinism_reference.summary["workers_used"]),
    }
    if len(worker_counts) != 2 or 1 not in worker_counts:
        raise ValueError("audit requires distinct serial and parallel builds")
    serial_parallel_exact_match = True

    checksum_manifest_sha256 = _sha256_file(checksums_path)
    audit = {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "trajectory_bank_version": TRAJECTORY_BANK_VERSION,
        "pose_time_layout_version": POSE_TIME_LAYOUT_VERSION,
        "trajectory_count": len(reloaded.trajectories),
        "trajectory_steps": int(reloaded.summary["trajectory_steps"]),
        "first_pose_time_s": float(reloaded.summary["first_pose_time_s"]),
        "last_pose_time_s": float(reloaded.summary["last_pose_time_s"]),
        "dt_s": float(reloaded.summary["dt_s"]),
        "artifact_reload_validation": "passed",
        "shape_dtype_finite_validation": "passed_all",
        "future_endpoint_kinematics": "passed_all",
        "query_map_invariants": "passed_all",
        "manifest_array_alignment": "passed_all",
        "summary_npz_alignment": "passed_all",
        "checksum_verification": "passed_all",
        "determinism_reference_exact_match": (
            determinism_reference_exact_match
        ),
        "serial_parallel_exact_match": serial_parallel_exact_match,
        "checksummed_payload_file_count": len(core_paths),
        "checksum_file": checksums_path.name,
        "checksum_manifest_sha256": checksum_manifest_sha256,
        "bank_semantic_digest_sha256": expected_digest,
        "provenance": provenance,
    }
    audit_path = root / "audit_report.json"
    audit_path.write_text(
        json.dumps(audit, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    envelope_hasher = hashlib.sha256()
    envelope_hasher.update(b"sop04_audited_bank_v2_external_handoff\0")
    envelope_hasher.update(checksums_path.read_bytes())
    envelope_hasher.update(b"\0")
    envelope_hasher.update(audit_path.read_bytes())
    external_handoff_digest = envelope_hasher.hexdigest()
    handoff_path = root / "external_handoff_digest.sha256"
    handoff_path.write_text(
        f"{external_handoff_digest}  sop04_audited_bank_v2_envelope\n",
        encoding="utf-8",
    )
    return {
        "checksums": checksums_path,
        "audit": audit_path,
        "handoff_digest": handoff_path,
    }


def write_trajectory_bank(
    bank: TrajectoryBank,
    output_dir: str | Path,
    *,
    provenance: dict[str, object],
    determinism_reference: TrajectoryBank,
) -> dict[str, Path]:
    """Atomically write the bank, manifest, summary, and provenance."""
    _validate_bank(bank)
    _validate_bank(determinism_reference)
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
                "trajectory_bank_version": TRAJECTORY_BANK_VERSION,
                "pose_time_layout_version": POSE_TIME_LAYOUT_VERSION,
                "first_pose_time_s": float(
                    trajectory.metadata["first_pose_time_s"]
                ),
                "last_pose_time_s": float(
                    trajectory.metadata["last_pose_time_s"]
                ),
                "dt_s": float(trajectory.metadata["dt_s"]),
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
        audit_paths = audit_trajectory_bank_artifact(
            staging,
            expected_bank=bank,
            provenance=provenance,
            determinism_reference=determinism_reference,
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
        **{
            name: output_path / path.name
            for name, path in audit_paths.items()
        },
    }
