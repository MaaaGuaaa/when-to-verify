"""Small authenticated visual audit for SOP05 regime-A and regime-B synthesis."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from src.contracts import build_grid_spec
from src.geometry import (
    CircleOccluder,
    RectangleFootprint,
    RectangleOccluder,
    inflate_footprint,
    rasterize_occluder,
)
from src.generation.dynamic_object_transplant import footprint_from_spec
from src.generation.sop05_seen_prior import (
    SeenPriorContextSweep,
    SeenPriorEnvironment,
    SeenPriorFailure,
    SeenPriorSource,
    generate_seen_prior,
)
from src.generation.sop05_unseen_prior import (
    LONG40_LAYOUT_VERSION,
    Long40TargetMotion,
    UnseenPriorConfig,
    UnseenPriorContextObstacle,
    UnseenPriorMother,
    generate_unseen_prior_mother,
)
from src.generation.sop05r_contracts import Sop05rTebConfig
from src.generation.sop05r_teb_output_loader import LoadedSop05rTebOutput
from src.planning.verification_actions import VerificationActionLibrary
from src.utils.atomic_publish import atomic_rename_noreplace

from .sop05r_teb_visuals import (
    Sop05rTebVisualBundle,
    build_sop05r_teb_visual_bundle,
    render_sop05r_teb_visual_bundle,
)


SOP05_AB_VISUAL_AUDIT_VERSION = "sop05_ab_visual_audit_v1"


class IncompleteContextError(ValueError):
    """A mother lacks the context history required by the selected regime."""


@dataclass(frozen=True)
class Sop05AbVisualAuditResult:
    output_dir: Path
    regime_a_event_ids: tuple[str, ...]
    regime_b_event_ids: tuple[str, ...]
    source_publication_semantic_digest: str


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
        raise ValueError("SOP05 A/B visual audit metadata must be canonical JSON") from exc


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_long40_poses(target: Long40TargetMotion) -> np.ndarray:
    return np.column_stack((target.positions, target.headings)).astype(
        np.float64, copy=False
    )


def _typed_occluder(item: Mapping[str, object]):
    shape = item.get("shape")
    if shape == "circle":
        return CircleOccluder(
            occluder_id=str(item["occluder_id"]),
            semantic_type=str(item["semantic_type"]),
            center_xy=np.asarray(item["center_xy"], dtype=np.float64),
            radius_m=float(item["radius_m"]),
        )
    if shape == "rectangle":
        return RectangleOccluder(
            occluder_id=str(item["occluder_id"]),
            semantic_type=str(item["semantic_type"]),
            pose=np.asarray(item["pose"], dtype=np.float64),
            length_m=float(item["length_m"]),
            width_m=float(item["width_m"]),
        )
    raise ValueError("SOP05 A/B visual audit encountered an unsupported occluder")


def _environment_masks(event, *, grid) -> tuple[np.ndarray, np.ndarray]:
    static = np.asarray(event.world.static_occupancy != 0, dtype=np.bool_)
    if static.shape != (grid.height, grid.width):
        raise ValueError("mother static occupancy does not match its grid")
    masks = tuple(
        rasterize_occluder(_typed_occluder(dict(item)), grid)
        for item in event.world.occluders
    )
    if not masks:
        raise ValueError("SOP05 A/B visual audit requires a represented occluder")
    occluder = np.logical_or.reduce(masks)
    return static & ~occluder, occluder


def _record_for_event(source: LoadedSop05rTebOutput, event_id: str):
    records = [record for record in source.trajectories.records if record.event_id == event_id]
    if len(records) != 1:
        raise ValueError("SOP05 A/B visual audit requires one source trajectory")
    return records[0]


def _state_for_event(source: LoadedSop05rTebOutput, event_id: str):
    record = _record_for_event(source, event_id)
    state = source.decision_states.get(record.decision_state_id)
    if state is None:
        raise ValueError("SOP05 A/B visual audit requires one decision state")
    return state


def _context_long40(event, state) -> tuple[UnseenPriorContextObstacle, ...]:
    target_id = event.target.target_dynamic_object_id
    result: list[UnseenPriorContextObstacle] = []
    for object_id in sorted(event.world.dynamic_object_trajectories):
        if object_id == target_id:
            continue
        history = state.visible_dynamic_object_history.get(object_id)
        future = event.world.dynamic_object_trajectories[object_id]
        spec = event.world.dynamic_object_specs.get(object_id)
        if history is None or spec is None:
            raise IncompleteContextError(
                "represented context lacks complete Long40 source data"
            )
        poses = np.vstack((history, future))
        result.append(
            UnseenPriorContextObstacle(
                object_id=object_id,
                footprint_spec=dict(spec),
                poses=np.asarray(poses, dtype=np.float32),
            )
        )
    return tuple(result)


def _context_future(event, state) -> tuple[SeenPriorContextSweep, ...]:
    target_id = event.target.target_dynamic_object_id
    result: list[SeenPriorContextSweep] = []
    for object_id in sorted(event.world.dynamic_object_trajectories):
        if object_id == target_id:
            continue
        history = state.visible_dynamic_object_history.get(object_id)
        future = event.world.dynamic_object_trajectories[object_id]
        spec = event.world.dynamic_object_specs.get(object_id)
        if history is None or spec is None:
            raise IncompleteContextError(
                "represented context lacks current-plus-future source data"
            )
        result.append(
            SeenPriorContextSweep(
                context_object_id=object_id,
                footprint=footprint_from_spec(dict(spec)),
                poses=np.asarray(np.vstack((history[-1], future)), dtype=np.float32),
            )
        )
    return tuple(result)


def _target_long40_from_event(event) -> Long40TargetMotion:
    poses = np.vstack((event.target.history_poses, event.target.future_poses))
    if poses.shape != (40, 3):
        raise ValueError("mother target must use Long40 poses")
    dt_s = 0.2
    positions = np.asarray(poses[:, :2], dtype=np.float32)
    velocities = np.gradient(positions.astype(np.float64), dt_s, axis=0).astype(
        np.float32
    )
    provenance = dict(event.target.provenance)
    return Long40TargetMotion(
        target_dynamic_object_id=event.target.target_dynamic_object_id,
        source_recording_id=str(provenance.get("source_recording_id", "source-recording")),
        source_session_id=str(provenance.get("source_session_id", "source-session")),
        source_snippet_id=event.target.snippet_id,
        source_object_id=event.target.source_object_id,
        object_type=event.target.object_type,
        footprint_spec=dict(event.target.footprint_spec),
        layout_version=LONG40_LAYOUT_VERSION,
        positions=positions,
        velocities=velocities,
        headings=np.asarray(poses[:, 2], dtype=np.float32),
    )


def _unseen_mother(source: LoadedSop05rTebOutput, event, *, base_config: Mapping[str, object]) -> UnseenPriorMother:
    state = _state_for_event(source, event.generated_event_id)
    grid = build_grid_spec(dict(base_config))
    static, occluder = _environment_masks(event, grid=grid)
    robot = dict(base_config["robot"])
    robot_footprint = inflate_footprint(
        RectangleFootprint(float(robot["length_m"]), float(robot["width_m"])),
        float(robot["inflation_m"]),
    )
    return UnseenPriorMother(
        mother_id=event.generated_event_id,
        split=state.split,
        target_motion=_target_long40_from_event(event),
        grid=grid,
        robot_footprint=robot_footprint,
        robot_history=np.asarray(state.robot_history, dtype=np.float32),
        static_occupancy=static,
        occluder_occupancy=occluder,
        context_obstacles=_context_long40(event, state),
    )


def _seen_source(event, state, *, source_identity: str) -> SeenPriorSource:
    return SeenPriorSource(
        mother_id=event.generated_event_id,
        split=state.split,
        source_collection_identity=source_identity,
        history_regime="seen_then_occluded",
        target_history_poses=np.asarray(event.target.history_poses, dtype=np.float32),
        target_current_pose=np.asarray(event.target.current_pose, dtype=np.float32),
        target_future_poses=np.asarray(event.target.future_poses, dtype=np.float32),
        target_visibility_history=np.asarray(
            event.target_visibility_history, dtype=np.bool_
        ),
    )


def _seen_environment(source: LoadedSop05rTebOutput, event, *, base_config: Mapping[str, object]) -> SeenPriorEnvironment:
    state = _state_for_event(source, event.generated_event_id)
    grid = build_grid_spec(dict(base_config))
    static, occluder = _environment_masks(event, grid=grid)
    return SeenPriorEnvironment(
        grid=grid,
        target_footprint=footprint_from_spec(dict(event.target.footprint_spec)),
        static_occupancy=static,
        occluder_occupancy=occluder,
        context_sweeps=_context_future(event, state),
    )


def build_sop05_ab_visual_bundle(
    source: LoadedSop05rTebOutput,
    *,
    event_id: str,
    target_long40: np.ndarray | None,
    target_visibility_history: np.ndarray | None,
    teb_config: Sop05rTebConfig,
    action_library: VerificationActionLibrary,
) -> Sop05rTebVisualBundle:
    """Reuse the mother BEV layers while withholding invalid mother-risk evidence."""

    if target_long40 is None:
        if target_visibility_history is not None:
            raise ValueError("empty SOP05 A/B scenario cannot carry target visibility")
    else:
        target = np.asarray(target_long40, dtype=np.float64)
        visibility = np.asarray(target_visibility_history, dtype=np.bool_)
        if target.shape != (40, 3) or not np.isfinite(target).all():
            raise ValueError("SOP05 A/B target must be finite Long40 poses")
        if visibility.shape != (8,):
            raise ValueError("SOP05 A/B target visibility must be boolean [8]")
        target_long40 = target
        target_visibility_history = visibility
    mother_bundle = build_sop05r_teb_visual_bundle(
        source,
        event_id=event_id,
        teb_config=teb_config,
        action_library=action_library,
    )
    return replace(
        mother_bundle,
        target_long40=target_long40,
        target_visibility_history=target_visibility_history,
        collision_point_xy=None,
        collision_time_s=None,
        witness_sample_index=None,
        witness_occluder_id=None,
        verification_traces=(),
    )


def _event_order(source: LoadedSop05rTebOutput, *, seed: int) -> tuple[object, ...]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("visual audit seed must be a nonnegative integer")
    indices = np.random.default_rng(seed).permutation(len(source.events))
    return tuple(source.events[int(index)] for index in indices)


def _render_scenario(
    *,
    staging: Path,
    regime: str,
    index: int,
    source: LoadedSop05rTebOutput,
    event_id: str,
    target_long40: np.ndarray | None,
    target_visibility_history: np.ndarray | None,
    teb_config: Sop05rTebConfig,
    action_library: VerificationActionLibrary,
) -> dict[str, object]:
    bundle = build_sop05_ab_visual_bundle(
        source,
        event_id=event_id,
        target_long40=target_long40,
        target_visibility_history=target_visibility_history,
        teb_config=teb_config,
        action_library=action_library,
    )
    filename = f"{regime}-{index:04d}-{event_id[-12:]}.png"
    relative_path = Path(f"regime-{regime}") / filename
    artifact = render_sop05r_teb_visual_bundle(bundle, staging / relative_path)
    return {
        "index": index,
        "source_event_id": event_id,
        "file": relative_path.as_posix(),
        "target_present": target_long40 is not None,
        "metadata": dict(artifact.metadata),
    }


def publish_sop05_ab_visual_audit(
    source: LoadedSop05rTebOutput,
    *,
    output_dir: str | Path,
    sample_count_per_regime: int,
    selection_seed: int,
    regime_a_present_only: bool = False,
    unseen_config: UnseenPriorConfig,
    seen_config,
    teb_config: Sop05rTebConfig,
    action_library: VerificationActionLibrary,
) -> Sop05AbVisualAuditResult:
    """Publish exactly the requested A/B visual audit samples or nothing."""

    if not isinstance(source, LoadedSop05rTebOutput) or not source.complete:
        raise ValueError("SOP05 A/B visual audit requires a complete mother collection")
    if not isinstance(sample_count_per_regime, int) or sample_count_per_regime <= 0:
        raise ValueError("sample_count_per_regime must be a positive integer")
    if not isinstance(regime_a_present_only, bool):
        raise TypeError("regime_a_present_only must be a bool")
    if source.manifest.get("config_digest") != teb_config.digest:
        raise ValueError("TEB config digest differs from the mother collection")
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite visual audit: {destination}")
    base_config = dict(source.manifest["base_config"])
    ordered_events = _event_order(source, seed=selection_seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    staging = staging_root / destination.name
    staging.mkdir()
    try:
        (staging / "regime-a").mkdir()
        (staging / "regime-b").mkdir()
        regime_a: list[dict[str, object]] = []
        regime_b: list[dict[str, object]] = []
        a_deficits: dict[str, int] = {}
        b_deficits: dict[str, int] = {}
        for event in ordered_events:
            if len(regime_a) >= sample_count_per_regime:
                break
            try:
                mother = _unseen_mother(source, event, base_config=base_config)
            except IncompleteContextError:
                a_deficits["incomplete_context"] = (
                    a_deficits.get("incomplete_context", 0) + 1
                )
                continue
            result = generate_unseen_prior_mother(
                mother,
                config=unseen_config,
                seed=unseen_config.seed,
            )
            if result.realization is None:
                reason = result.provenance.outcome
                a_deficits[reason] = a_deficits.get(reason, 0) + 1
                continue
            target = result.realization.target_motion
            if regime_a_present_only and target is None:
                a_deficits["empty_branch_excluded"] = (
                    a_deficits.get("empty_branch_excluded", 0) + 1
                )
                continue
            sample = _render_scenario(
                staging=staging,
                regime="a",
                index=len(regime_a),
                source=source,
                event_id=event.generated_event_id,
                target_long40=None if target is None else _as_long40_poses(target),
                target_visibility_history=(
                    None
                    if target is None
                    else np.zeros(8, dtype=np.bool_)
                ),
                teb_config=teb_config,
                action_library=action_library,
            )
            sample["sampling"] = {
                "branch": result.provenance.presence_branch,
                "outcome": result.provenance.outcome,
                "attempted_angle_count": result.provenance.attempted_angle_count,
                "selected_angle_rad": result.provenance.selected_angle_rad,
                "audit_condition": (
                    "present_only" if regime_a_present_only else "unconditioned"
                ),
            }
            regime_a.append(sample)
        for event in ordered_events:
            if len(regime_b) >= sample_count_per_regime:
                break
            history = np.asarray(event.target_visibility_history, dtype=np.bool_)
            if history.shape != (8,) or not bool(history[:7].any()) or bool(history[7]):
                continue
            try:
                state = _state_for_event(source, event.generated_event_id)
                seen_source = _seen_source(
                    event,
                    state,
                    source_identity=source.publication_semantic_digest,
                )
                seen_environment = _seen_environment(
                    source,
                    event,
                    base_config=base_config,
                )
            except IncompleteContextError:
                b_deficits["incomplete_context"] = (
                    b_deficits.get("incomplete_context", 0) + 1
                )
                continue
            result = generate_seen_prior(
                seen_source,
                seen_environment,
                seen_config,
                int(base_config["seed"]),
            )
            if isinstance(result, SeenPriorFailure):
                b_deficits[result.reason] = b_deficits.get(result.reason, 0) + 1
                continue
            target = np.vstack((result.history_poses, result.future_poses))
            sample = _render_scenario(
                staging=staging,
                regime="b",
                index=len(regime_b),
                source=source,
                event_id=event.generated_event_id,
                target_long40=target,
                target_visibility_history=history,
                teb_config=teb_config,
                action_library=action_library,
            )
            sample["sampling"] = {
                "accepted_attempt": result.accepted_attempt,
                "selected_angle_rad": result.theta_rad,
            }
            regime_b.append(sample)
        if len(regime_a) != sample_count_per_regime:
            raise ValueError("insufficient legal regime-A visual audit samples")
        if len(regime_b) != sample_count_per_regime:
            raise ValueError("insufficient legal regime-B visual audit samples")
        manifest = {
            "version": SOP05_AB_VISUAL_AUDIT_VERSION,
            "source_publication_semantic_digest": source.publication_semantic_digest,
            "sample_count_per_regime": sample_count_per_regime,
            "selection_seed": selection_seed,
            "regime_a_selection": (
                "present_only" if regime_a_present_only else "unconditioned"
            ),
            "regime_a": regime_a,
            "regime_b": regime_b,
            "deficits_before_quota": {"regime_a": a_deficits, "regime_b": b_deficits},
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
                    "version": SOP05_AB_VISUAL_AUDIT_VERSION,
                    "source_publication_semantic_digest": source.publication_semantic_digest,
                    "regime_a_count": len(regime_a),
                    "regime_b_count": len(regime_b),
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
    return Sop05AbVisualAuditResult(
        output_dir=destination,
        regime_a_event_ids=tuple(str(item["source_event_id"]) for item in regime_a),
        regime_b_event_ids=tuple(str(item["source_event_id"]) for item in regime_b),
        source_publication_semantic_digest=source.publication_semantic_digest,
    )


__all__ = (
    "IncompleteContextError",
    "SOP05_AB_VISUAL_AUDIT_VERSION",
    "Sop05AbVisualAuditResult",
    "build_sop05_ab_visual_bundle",
    "publish_sop05_ab_visual_audit",
)
