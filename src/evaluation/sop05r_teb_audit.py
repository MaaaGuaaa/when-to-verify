"""Recomputed release evidence for complete Long40 SOP05R TEB collections."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping
import tempfile

import numpy as np

from src.geometry import CircleOccluder, RectangleOccluder
from src.generation.anchored_human_placement import synchronized_centerline_blocking
from src.generation.sop05r_contracts import SOP05R_TEB_GENERATOR_VERSION
from src.generation.sop05r_teb_output_loader import (
    LoadedSop05rTebOutput,
    load_sop05r_teb_output,
)
from src.generation.sop05r_contracts import Sop05rTebConfig
from src.planning.verification_actions import VerificationActionLibrary
from src.utils.atomic_publish import atomic_rename_noreplace

from .sop05r_teb_visuals import (
    build_sop05r_teb_visual_bundle,
    render_sop05r_teb_visual_bundle,
)


SOP05R_TEB_AUDIT_VERSION = "sop05r_teb_audit_v1"
_CENTERLINE_EPSILON_M = 0.01


@dataclass(frozen=True)
class Sop05rTebAuditMetrics:
    version: str
    publication_semantic_digest: str
    event_count: int
    history_visible_frames: int
    history_occluded_frames: int
    events_with_visible_and_occluded_history: int
    collision_time_min_s: float
    collision_time_max_s: float
    collision_time_mean_s: float
    occlusion_witness_count: int
    recomputed_witness_count: int
    shape_counts: Mapping[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "publication_semantic_digest": self.publication_semantic_digest,
            "event_count": self.event_count,
            "history_visible_frames": self.history_visible_frames,
            "history_occluded_frames": self.history_occluded_frames,
            "events_with_visible_and_occluded_history": (
                self.events_with_visible_and_occluded_history
            ),
            "collision_time_min_s": self.collision_time_min_s,
            "collision_time_max_s": self.collision_time_max_s,
            "collision_time_mean_s": self.collision_time_mean_s,
            "occlusion_witness_count": self.occlusion_witness_count,
            "recomputed_witness_count": self.recomputed_witness_count,
            "shape_counts": dict(self.shape_counts),
        }


def _typed_occluders(metadata: tuple[dict, ...]):
    result = []
    for item in metadata:
        shape = item.get("shape")
        if shape == "circle":
            result.append(
                CircleOccluder(
                    occluder_id=str(item["occluder_id"]),
                    semantic_type=str(item["semantic_type"]),
                    center_xy=np.asarray(item["center_xy"], dtype=np.float64),
                    radius_m=float(item["radius_m"]),
                )
            )
        elif shape == "rectangle":
            result.append(
                RectangleOccluder(
                    occluder_id=str(item["occluder_id"]),
                    semantic_type=str(item["semantic_type"]),
                    pose=np.asarray(item["pose"], dtype=np.float64),
                    length_m=float(item["length_m"]),
                    width_m=float(item["width_m"]),
                )
            )
        else:
            raise ValueError("TEB audit encountered an unsupported occluder shape")
    if not result:
        raise ValueError("TEB audit requires at least one represented occluder")
    return tuple(result)


def audit_sop05r_teb_collection(
    source: LoadedSop05rTebOutput,
) -> Sop05rTebAuditMetrics:
    """Recompute event-level release statistics from strict M7 objects only."""

    if not isinstance(source, LoadedSop05rTebOutput):
        raise TypeError("source must be a LoadedSop05rTebOutput")
    if not source.complete:
        raise ValueError("TEB audit requires a complete M7 collection")
    if source.manifest.get("generator_algorithm_version") != SOP05R_TEB_GENERATOR_VERSION:
        raise ValueError("TEB audit rejects a non-current generator collection")
    if not source.events:
        raise ValueError("TEB audit requires at least one event")
    if source.summary.get("accepted_count") != len(source.events):
        raise ValueError("TEB audit accepted_count does not match loaded events")

    visible_frames = 0
    occluded_frames = 0
    mixed_history_events = 0
    collision_times: list[float] = []
    shape_counts: dict[str, int] = {}
    witness_count = 0
    recomputed_witness_count = 0
    trajectories = {record.event_id: record for record in source.trajectories.records}
    if set(trajectories) != {event.generated_event_id for event in source.events}:
        raise ValueError("TEB audit event/trajectory identity mismatch")
    for event in source.events:
        history = np.asarray(event.target_visibility_history, dtype=np.bool_)
        record = trajectories[event.generated_event_id]
        state = source.decision_states.get(record.decision_state_id)
        evidence = source.event_evidence.get(event.generated_event_id)
        if state is None or evidence is None:
            raise ValueError("TEB audit event decision or evidence is missing")
        if (
            history.shape != (8,)
            or event.target.history_poses.shape != (8, 3)
            or event.target.future_poses.shape != (32, 3)
            or record.full_route.band_poses_world.shape != (21, 3)
            or record.full_route.sampled_poses_world.shape != (40, 3)
            or record.nominal_trajectory.poses.shape != (32, 3)
        ):
            raise ValueError("TEB audit long40 shape contract mismatch")
        if not (
            np.isfinite(event.target.history_poses).all()
            and np.isfinite(event.target.future_poses).all()
            and np.isfinite(record.full_route.sampled_poses_world).all()
            and np.isfinite(record.nominal_trajectory.poses).all()
        ):
            raise ValueError("TEB audit found non-finite trajectory evidence")
        visible = int(history.sum())
        visible_frames += visible
        occluded_frames += int(history.size - visible)
        mixed_history_events += int(0 < visible < history.size)
        collision_time = float(event.conflict_time_s)
        if not 1.2 <= collision_time <= 6.4 or not np.isfinite(collision_time):
            raise ValueError("TEB audit collision time violates the M6 window")
        collision_times.append(collision_time)
        witness = evidence.get("occlusion_witness")
        if not isinstance(witness, Mapping):
            raise ValueError("TEB audit occlusion witness is invalid")
        witness_index = int(witness["sample_index"])
        robot_long40 = np.vstack((state.robot_history, record.nominal_trajectory.poses))
        target_long40 = np.vstack((event.target.history_poses, event.target.future_poses))
        blocked, blocker_ids = synchronized_centerline_blocking(
            robot_long40[:, :2],
            target_long40[:, :2],
            _typed_occluders(event.world.occluders),
            epsilon_m=_CENTERLINE_EPSILON_M,
        )
        witness_count += 1
        if (
            not bool(blocked[witness_index])
            or blocker_ids[witness_index] != witness.get("occluder_id")
        ):
            raise ValueError("TEB audit persisted occlusion witness does not replay")
        recomputed_witness_count += 1
        for occluder in event.world.occluders:
            shape = str(dict(occluder).get("shape"))
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
    values = np.asarray(collision_times, dtype=np.float64)
    return Sop05rTebAuditMetrics(
        version=SOP05R_TEB_AUDIT_VERSION,
        publication_semantic_digest=source.publication_semantic_digest,
        event_count=len(source.events),
        history_visible_frames=visible_frames,
        history_occluded_frames=occluded_frames,
        events_with_visible_and_occluded_history=mixed_history_events,
        collision_time_min_s=float(values.min()),
        collision_time_max_s=float(values.max()),
        collision_time_mean_s=float(values.mean()),
        occlusion_witness_count=witness_count,
        recomputed_witness_count=recomputed_witness_count,
        shape_counts={key: shape_counts[key] for key in sorted(shape_counts)},
    )


def audit_sop05r_teb_root(root: str | Path) -> Sop05rTebAuditMetrics:
    """Strictly reload then audit a published collection; tampering fails closed."""

    return audit_sop05r_teb_collection(
        load_sop05r_teb_output(root, require_complete=True)
    )


@dataclass(frozen=True)
class Sop05rTebVisualAuditResult:
    output_dir: Path
    selected_event_ids: tuple[str, ...]
    publication_semantic_digest: str
    metrics: Sop05rTebAuditMetrics


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("TEB visual audit metadata must be canonical JSON") from exc


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_event_ids(source: LoadedSop05rTebOutput, *, count: int, seed: int) -> tuple[str, ...]:
    if type(count) is not int or count <= 0:
        raise ValueError("visual audit sample_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("visual audit seed must be a nonnegative integer")
    event_ids = tuple(event.generated_event_id for event in source.events)
    if count > len(event_ids):
        raise ValueError("visual audit sample_count exceeds the strict source count")
    selected_indices = np.random.default_rng(seed).choice(
        len(event_ids), size=count, replace=False
    )
    return tuple(sorted(event_ids[int(index)] for index in selected_indices))


def publish_sop05r_teb_visual_audit(
    source: LoadedSop05rTebOutput,
    *,
    output_dir: str | Path,
    sample_count: int,
    seed: int,
    teb_config: Sop05rTebConfig,
    action_library: VerificationActionLibrary,
) -> Sop05rTebVisualAuditResult:
    """Atomically render every selected strict event; failed samples are never skipped."""

    if not isinstance(source, LoadedSop05rTebOutput):
        raise TypeError("source must be a LoadedSop05rTebOutput")
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite visual audit: {destination}")
    metrics = audit_sop05r_teb_collection(source)
    selected_ids = _selected_event_ids(source, count=sample_count, seed=seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    staging = staging_root / destination.name
    staging.mkdir()
    try:
        samples = []
        samples_dir = staging / "samples"
        samples_dir.mkdir()
        for index, event_id in enumerate(selected_ids):
            bundle = build_sop05r_teb_visual_bundle(
                source,
                event_id=event_id,
                teb_config=teb_config,
                action_library=action_library,
            )
            filename = f"sample-{index:04d}-{event_id[-12:]}.png"
            artifact = render_sop05r_teb_visual_bundle(
                bundle,
                samples_dir / filename,
            )
            samples.append(
                {
                    "index": index,
                    "event_id": event_id,
                    "file": f"samples/{filename}",
                    "metadata": dict(artifact.metadata),
                }
            )
        (staging / "audit_metrics.json").write_bytes(
            _canonical_json(metrics.as_dict()) + b"\n"
        )
        manifest = {
            "version": "sop05r_teb_visual_audit_v1",
            "source_publication_semantic_digest": source.publication_semantic_digest,
            "source_event_count": len(source.events),
            "sample_count": sample_count,
            "seed": seed,
            "selected_event_ids": list(selected_ids),
            "samples": samples,
        }
        (staging / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        checksums = {
            path.relative_to(staging).as_posix(): _sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        (staging / "checksums.json").write_bytes(_canonical_json(checksums) + b"\n")
        (staging / "COMPLETE.json").write_bytes(
            _canonical_json(
                {
                    "version": "sop05r_teb_visual_audit_complete_v1",
                    "source_publication_semantic_digest": (
                        source.publication_semantic_digest
                    ),
                    "selected_event_ids": list(selected_ids),
                }
            )
            + b"\n"
        )
        atomic_rename_noreplace(staging, destination)
    except Exception:
        import shutil

        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    else:
        staging_root.rmdir()
    return Sop05rTebVisualAuditResult(
        output_dir=destination,
        selected_event_ids=selected_ids,
        publication_semantic_digest=source.publication_semantic_digest,
        metrics=metrics,
    )
