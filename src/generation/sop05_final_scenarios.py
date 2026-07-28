"""Immutable full-release publisher for the finalized SOP05 A/B scenarios."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any
import zipfile

import numpy as np

from src.contracts import build_grid_spec
from src.geometry import (
    CircleOccluder,
    RectangleFootprint,
    RectangleOccluder,
    inflate_footprint,
    rasterize_occluder,
)
from src.utils.atomic_publish import atomic_rename_noreplace

from .event_contracts import footprint_from_spec
from .sop05_seen_prior import (
    SeenPriorConfig,
    SeenPriorContextSweep,
    SeenPriorEnvironment,
    SeenPriorFailure,
    SeenPriorSource,
    generate_seen_prior,
)
from .sop05_unseen_prior import (
    LONG40_LAYOUT_VERSION,
    Long40TargetMotion,
    UnseenPriorConfig,
    UnseenPriorContextObstacle,
    UnseenPriorMother,
    generate_unseen_prior_mother,
)
from .sop05r_teb_output_loader import LoadedSop05rTebOutput


SOP05_FINAL_SCENARIO_VERSION = "sop05_final_single_scenario_v1"
_MANIFEST = "manifest.json"
_RECORDS = "records.jsonl"
_ORACLE_TARGETS = "oracle_targets.npz"
_PROVENANCE = "provenance.jsonl"
_CHECKSUMS = "checksums.json"
_COMPLETE = "COMPLETE.json"
_FILE_SET = frozenset(
    {_MANIFEST, _RECORDS, _ORACLE_TARGETS, _PROVENANCE, _CHECKSUMS, _COMPLETE}
)
_ORACLE_ARRAYS = frozenset(
    {
        "future_poses",
        "history_poses",
        "source_record_indices",
        "target_present",
    }
)


class Sop05FinalScenarioError(ValueError):
    """Raised when a finalized SOP05 scenario release is invalid."""


class IncompleteContextError(Sop05FinalScenarioError):
    """Raised when a mother lacks the represented context needed by a regime."""


@dataclass(frozen=True)
class Sop05FinalScenarioPublishResult:
    output_dir: Path
    source_publication_semantic_digest: str
    processed_mother_count: int
    accepted_count: int
    deficit_count: int
    full_source_coverage: bool


@dataclass(frozen=True)
class LoadedSop05FinalScenarios:
    root: Path
    manifest: Mapping[str, object]
    records: tuple[dict[str, object], ...]
    history_poses: np.ndarray
    future_poses: np.ndarray
    target_present: np.ndarray
    source_record_indices: np.ndarray
    accepted_count: int
    deficit_count: int
    full_source_coverage: bool


@dataclass(frozen=True)
class Sop05FinalScenarioSelection:
    """One already-decided final scenario ready for immutable publication."""

    mother_id: str
    split: str
    target_present: bool
    history_poses: np.ndarray
    future_poses: np.ndarray
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.mother_id, str) or not self.mother_id:
            raise Sop05FinalScenarioError("selected scenario mother_id is invalid")
        if self.split not in {"train", "calibration", "val", "test"}:
            raise Sop05FinalScenarioError("selected scenario split is invalid")
        if not isinstance(self.target_present, bool):
            raise Sop05FinalScenarioError("selected scenario presence must be boolean")
        for name, shape in (("history_poses", (8, 3)), ("future_poses", (32, 3))):
            array = np.asarray(getattr(self, name))
            if array.shape != shape or not np.issubdtype(array.dtype, np.number):
                raise Sop05FinalScenarioError(
                    f"selected scenario {name} must be a numeric {shape} array"
                )
            result = np.array(array, dtype=np.float32, order="C", copy=True)
            if not np.isfinite(result).all():
                raise Sop05FinalScenarioError(
                    f"selected scenario {name} must contain finite values"
                )
            result.setflags(write=False)
            object.__setattr__(self, name, result)
        if not self.target_present and (
            np.any(self.history_poses) or np.any(self.future_poses)
        ):
            raise Sop05FinalScenarioError(
                "target-empty selected scenarios must use zero oracle target arrays"
            )
        if not isinstance(self.provenance, Mapping):
            raise Sop05FinalScenarioError("selected scenario provenance must be a mapping")
        snapshot = json.loads(_canonical_json(dict(self.provenance)).decode("ascii"))
        object.__setattr__(self, "provenance", MappingProxyType(snapshot))


@dataclass(frozen=True)
class _WorkResult:
    record: dict[str, object]
    provenance: dict[str, object]
    history_poses: np.ndarray | None
    future_poses: np.ndarray | None
    target_present: bool


_WORKER_SOURCE: LoadedSop05rTebOutput | None = None
_WORKER_UNSEEN_CONFIG: UnseenPriorConfig | None = None
_WORKER_SEEN_CONFIG: SeenPriorConfig | None = None


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
        raise Sop05FinalScenarioError("scenario metadata must be canonical JSON") from exc


def _json_file(value: object) -> bytes:
    return _canonical_json(value) + b"\n"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(_json_file(row) for row in rows))


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Sop05FinalScenarioError(f"failed to read {path.name}") from exc


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Sop05FinalScenarioError(f"failed to read {path.name}") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Sop05FinalScenarioError(
                f"{path.name} line {line_number} is not valid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise Sop05FinalScenarioError(
                f"{path.name} line {line_number} must be an object"
            )
        rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise Sop05FinalScenarioError(f"failed to checksum {path.name}") from exc


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.ascontiguousarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


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
    raise Sop05FinalScenarioError("mother contains an unsupported occluder")


def _environment_masks(event: object, *, grid) -> tuple[np.ndarray, np.ndarray]:
    world = getattr(event, "world")
    static = np.asarray(world.static_occupancy != 0, dtype=np.bool_)
    if static.shape != (grid.height, grid.width):
        raise Sop05FinalScenarioError("mother static occupancy does not match its grid")
    masks = tuple(
        rasterize_occluder(_typed_occluder(dict(item)), grid)
        for item in world.occluders
    )
    if not masks:
        raise Sop05FinalScenarioError("mother has no represented occluder")
    occluder = np.logical_or.reduce(masks)
    return static & ~occluder, occluder


def _state_for_event(source: LoadedSop05rTebOutput, event_id: str):
    records = [
        record for record in source.trajectories.records if record.event_id == event_id
    ]
    if len(records) != 1:
        raise Sop05FinalScenarioError("mother must have exactly one trajectory")
    state = source.decision_states.get(records[0].decision_state_id)
    if state is None:
        raise Sop05FinalScenarioError("mother decision state is missing")
    return state


def _context_long40(event: object, state: object) -> tuple[UnseenPriorContextObstacle, ...]:
    target_id = event.target.target_dynamic_object_id
    result: list[UnseenPriorContextObstacle] = []
    for object_id in sorted(event.world.dynamic_object_trajectories):
        if object_id == target_id:
            continue
        history = state.visible_dynamic_object_history.get(object_id)
        future = event.world.dynamic_object_trajectories[object_id]
        spec = event.world.dynamic_object_specs.get(object_id)
        if history is None or spec is None:
            raise IncompleteContextError("represented context lacks complete Long40 data")
        result.append(
            UnseenPriorContextObstacle(
                object_id=object_id,
                footprint_spec=dict(spec),
                poses=np.asarray(np.vstack((history, future)), dtype=np.float32),
            )
        )
    return tuple(result)


def _context_future(event: object, state: object) -> tuple[SeenPriorContextSweep, ...]:
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
                "represented context lacks current-plus-future data"
            )
        result.append(
            SeenPriorContextSweep(
                context_object_id=object_id,
                footprint=footprint_from_spec(dict(spec)),
                poses=np.asarray(np.vstack((history[-1], future)), dtype=np.float32),
            )
        )
    return tuple(result)


def _target_long40_from_event(event: object) -> Long40TargetMotion:
    poses = np.vstack((event.target.history_poses, event.target.future_poses))
    if poses.shape != (40, 3):
        raise Sop05FinalScenarioError("mother target must use Long40 poses")
    positions = np.asarray(poses[:, :2], dtype=np.float32)
    velocities = np.gradient(positions.astype(np.float64), 0.2, axis=0).astype(
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


def _unseen_mother(
    source: LoadedSop05rTebOutput,
    event: object,
    *,
    base_config: Mapping[str, object],
) -> UnseenPriorMother:
    state = _state_for_event(source, event.generated_event_id)
    return _unseen_mother_from_state(event, state, base_config=base_config)


def _unseen_mother_from_state(
    event: object,
    state: object,
    *,
    base_config: Mapping[str, object],
) -> UnseenPriorMother:
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


def _seen_source(
    event: object,
    state: object,
    *,
    source_identity: str,
) -> SeenPriorSource:
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


def _seen_environment(
    source: LoadedSop05rTebOutput,
    event: object,
    *,
    base_config: Mapping[str, object],
) -> SeenPriorEnvironment:
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


def _scenario_id(source_digest: str, mother_id: str) -> str:
    digest = hashlib.sha256(
        f"{SOP05_FINAL_SCENARIO_VERSION}\0{source_digest}\0{mother_id}".encode(
            "ascii"
        )
    ).hexdigest()
    return f"sop05-final-{digest[:24]}"


def _base_record(
    *,
    source_index: int,
    event: object,
    state: object,
    regime: str,
) -> dict[str, object]:
    return {
        "source_index": source_index,
        "mother_id": event.generated_event_id,
        "split": state.split,
        "regime": regime,
    }


def _deficit(
    base: dict[str, object], *, reason: str, provenance: dict[str, object]
) -> _WorkResult:
    return _WorkResult(
        record={**base, "status": "deficit", "reason": reason},
        provenance={**base, "status": "deficit", "reason": reason, **provenance},
        history_poses=None,
        future_poses=None,
        target_present=False,
    )


def _accepted(
    base: dict[str, object],
    *,
    source_digest: str,
    history_poses: np.ndarray,
    future_poses: np.ndarray,
    target_present: bool,
    provenance: dict[str, object],
) -> _WorkResult:
    history = np.asarray(history_poses, dtype=np.float32)
    future = np.asarray(future_poses, dtype=np.float32)
    if (
        history.shape != (8, 3)
        or future.shape != (32, 3)
        or not np.isfinite(history).all()
        or not np.isfinite(future).all()
    ):
        raise Sop05FinalScenarioError("finalized target must be finite Long40 poses")
    return _WorkResult(
        record={
            **base,
            "status": "accepted",
            "scenario_id": _scenario_id(source_digest, str(base["mother_id"])),
            "target_present": target_present,
        },
        provenance={**base, "status": "accepted", **provenance},
        history_poses=np.array(history, dtype=np.float32, order="C", copy=True),
        future_poses=np.array(future, dtype=np.float32, order="C", copy=True),
        target_present=target_present,
    )


def _history_regime(event: object) -> str | None:
    history = np.asarray(event.target_visibility_history, dtype=np.bool_)
    if history.shape != (8,):
        raise Sop05FinalScenarioError("mother target visibility history must be bool[8]")
    if not bool(history[0]):
        return "unseen_in_history_window"
    return "seen_then_occluded"


def _process_event(
    source: LoadedSop05rTebOutput,
    event: object,
    source_index: int,
    *,
    unseen_config: UnseenPriorConfig,
    seen_config: SeenPriorConfig,
) -> _WorkResult:
    base_config = dict(source.manifest["base_config"])
    state = _state_for_event(source, event.generated_event_id)
    regime = _history_regime(event)
    if regime is None:
        base = _base_record(
            source_index=source_index,
            event=event,
            state=state,
            regime="unsupported_history",
        )
        return _deficit(base, reason="unsupported_history_regime", provenance={})
    base = _base_record(
        source_index=source_index,
        event=event,
        state=state,
        regime=regime,
    )
    try:
        if regime == "unseen_in_history_window":
            result = generate_unseen_prior_mother(
                _unseen_mother(source, event, base_config=base_config),
                config=unseen_config,
                seed=unseen_config.seed,
            )
            audit = {
                "presence_branch": result.provenance.presence_branch,
                "outcome": result.provenance.outcome,
                "attempted_angle_count": result.provenance.attempted_angle_count,
                "selected_angle_rad": result.provenance.selected_angle_rad,
                "rejection_reason_counts": dict(
                    result.provenance.rejection_reason_counts
                ),
            }
            if result.realization is None:
                return _deficit(
                    base,
                    reason=result.provenance.outcome,
                    provenance=audit,
                )
            target = result.realization.target_motion
            if target is None:
                return _accepted(
                    base,
                    source_digest=source.publication_semantic_digest,
                    history_poses=np.zeros((8, 3), dtype=np.float32),
                    future_poses=np.zeros((32, 3), dtype=np.float32),
                    target_present=False,
                    provenance=audit,
                )
            history = np.column_stack((target.positions[:8], target.headings[:8]))
            future = np.column_stack((target.positions[8:], target.headings[8:]))
            return _accepted(
                base,
                source_digest=source.publication_semantic_digest,
                history_poses=history,
                future_poses=future,
                target_present=True,
                provenance=audit,
            )

        result = generate_seen_prior(
            _seen_source(
                event,
                state,
                source_identity=source.publication_semantic_digest,
            ),
            _seen_environment(source, event, base_config=base_config),
            seen_config,
            int(base_config["seed"]),
        )
        if isinstance(result, SeenPriorFailure):
            return _deficit(
                base,
                reason=result.reason,
                provenance={
                    "attempted_angle_count": result.attempts,
                    "rejection_reason_counts": dict(result.rejection_counts),
                },
            )
        return _accepted(
            base,
            source_digest=source.publication_semantic_digest,
            history_poses=result.history_poses,
            future_poses=result.future_poses,
            target_present=True,
            provenance={
                "attempted_angle_count": result.accepted_attempt,
                "selected_angle_rad": result.theta_rad,
            },
        )
    except IncompleteContextError:
        return _deficit(base, reason="incomplete_context", provenance={})


def _initialize_worker(
    source: LoadedSop05rTebOutput,
    unseen_config: UnseenPriorConfig,
    seen_config: SeenPriorConfig,
) -> None:
    global _WORKER_SOURCE, _WORKER_UNSEEN_CONFIG, _WORKER_SEEN_CONFIG
    _WORKER_SOURCE = source
    _WORKER_UNSEEN_CONFIG = unseen_config
    _WORKER_SEEN_CONFIG = seen_config


def _worker_process(task: tuple[int, str]) -> _WorkResult:
    source = _WORKER_SOURCE
    unseen_config = _WORKER_UNSEEN_CONFIG
    seen_config = _WORKER_SEEN_CONFIG
    if source is None or unseen_config is None or seen_config is None:
        raise RuntimeError("SOP05 final scenario worker was not initialized")
    source_index, event_id = task
    matches = [event for event in source.events if event.generated_event_id == event_id]
    if len(matches) != 1:
        raise Sop05FinalScenarioError("worker could not resolve its mother event")
    return _process_event(
        source,
        matches[0],
        source_index,
        unseen_config=unseen_config,
        seen_config=seen_config,
    )


def _process_all(
    source: LoadedSop05rTebOutput,
    events: tuple[object, ...],
    *,
    unseen_config: UnseenPriorConfig,
    seen_config: SeenPriorConfig,
    workers: int,
    progress_callback: Callable[[int, int], None] | None,
) -> list[_WorkResult]:
    if workers == 1:
        results: list[_WorkResult] = []
        for source_index, event in enumerate(events):
            results.append(
                _process_event(
                    source,
                    event,
                    source_index,
                    unseen_config=unseen_config,
                    seen_config=seen_config,
                )
            )
            if progress_callback is not None:
                progress_callback(source_index + 1, len(events))
        return results
    if multiprocessing.get_start_method(allow_none=True) not in (None, "fork"):
        raise Sop05FinalScenarioError("parallel SOP05 final release requires fork")
    tasks = tuple((index, event.generated_event_id) for index, event in enumerate(events))
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(tasks)),
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(source, unseen_config, seen_config),
    ) as executor:
        results = []
        for completed, result in enumerate(
            executor.map(_worker_process, tasks, chunksize=8), start=1
        ):
            results.append(result)
            if progress_callback is not None:
                progress_callback(completed, len(events))
    return results


def _validate_publish_inputs(
    source: LoadedSop05rTebOutput,
    *,
    output_dir: str | Path,
    unseen_config: UnseenPriorConfig,
    seen_config: SeenPriorConfig,
    expected_source_config_digest: str,
    workers: int,
    max_mothers: int | None,
) -> tuple[Path, tuple[object, ...]]:
    if not isinstance(source, LoadedSop05rTebOutput) or not source.complete:
        raise Sop05FinalScenarioError("SOP05 final release requires a complete mother collection")
    if not isinstance(unseen_config, UnseenPriorConfig):
        raise TypeError("unseen_config must be an UnseenPriorConfig")
    if not isinstance(seen_config, SeenPriorConfig):
        raise TypeError("seen_config must be a SeenPriorConfig")
    if not isinstance(expected_source_config_digest, str) or not expected_source_config_digest:
        raise TypeError("expected_source_config_digest must be a non-empty string")
    if source.manifest.get("config_digest") != expected_source_config_digest:
        raise Sop05FinalScenarioError("mother collection config digest differs")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if max_mothers is not None and (
        isinstance(max_mothers, bool) or not isinstance(max_mothers, int) or max_mothers <= 0
    ):
        raise ValueError("max_mothers must be a positive integer or None")
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite final scenario release: {destination}")
    events = tuple(source.events if max_mothers is None else source.events[:max_mothers])
    if len(source.events) > unseen_config.hard_total_sample_cap:
        raise Sop05FinalScenarioError("mother collection exceeds the configured release cap")
    if len({event.generated_event_id for event in events}) != len(events):
        raise Sop05FinalScenarioError("mother collection has duplicate event IDs")
    return destination, events


def publish_sop05_final_scenarios(
    source: LoadedSop05rTebOutput,
    *,
    output_dir: str | Path,
    unseen_config: UnseenPriorConfig,
    seen_config: SeenPriorConfig,
    expected_source_config_digest: str,
    workers: int = 1,
    max_mothers: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Sop05FinalScenarioPublishResult:
    """Synthesize at most one final A/B scenario for every selected mother."""

    destination, events = _validate_publish_inputs(
        source,
        output_dir=output_dir,
        unseen_config=unseen_config,
        seen_config=seen_config,
        expected_source_config_digest=expected_source_config_digest,
        workers=workers,
        max_mothers=max_mothers,
    )
    results = _process_all(
        source,
        events,
        unseen_config=unseen_config,
        seen_config=seen_config,
        workers=workers,
        progress_callback=progress_callback,
    )
    if [result.record["source_index"] for result in results] != list(range(len(events))):
        raise Sop05FinalScenarioError("worker results do not preserve source order")

    return _publish_work_results(
        source_publication_semantic_digest=source.publication_semantic_digest,
        source_config_digest=expected_source_config_digest,
        source_mother_count=len(source.events),
        output_dir=destination,
        results=results,
        unseen_prior_config_digest=unseen_config.config_digest,
        seen_prior_config_digest=seen_config.digest,
    )


def publish_selected_sop05_final_scenarios(
    source: LoadedSop05rTebOutput,
    *,
    selections: tuple[Sop05FinalScenarioSelection, ...],
    output_dir: str | Path,
    unseen_prior_config_digest: str,
    seen_prior_config_digest: str,
) -> Sop05FinalScenarioPublishResult:
    """Publish selected present/empty scenarios without running either prior."""

    if not isinstance(source, LoadedSop05rTebOutput) or not source.complete:
        raise Sop05FinalScenarioError(
            "selected final scenarios require a complete mother collection"
        )
    if not isinstance(selections, tuple) or not all(
        isinstance(item, Sop05FinalScenarioSelection) for item in selections
    ):
        raise TypeError(
            "selections must be a tuple of Sop05FinalScenarioSelection values"
        )
    for name, value in (
        ("unseen_prior_config_digest", unseen_prior_config_digest),
        ("seen_prior_config_digest", seen_prior_config_digest),
    ):
        if not isinstance(value, str) or not value:
            raise TypeError(f"{name} must be a non-empty string")
    source_config_digest = source.manifest.get("config_digest")
    if not isinstance(source_config_digest, str) or not source_config_digest:
        raise Sop05FinalScenarioError("mother collection config digest is missing")
    source_ids = tuple(event.generated_event_id for event in source.events)
    if len(set(source_ids)) != len(source_ids):
        raise Sop05FinalScenarioError("mother collection has duplicate event IDs")
    selected_ids = tuple(item.mother_id for item in selections)
    if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids).issubset(
        source_ids
    ):
        raise Sop05FinalScenarioError(
            "selected scenario mothers must be unique members of the source"
        )
    results = [
        _accepted(
            {
                "source_index": source_index,
                "mother_id": selection.mother_id,
                "split": selection.split,
                "regime": "unseen_in_history_window",
            },
            source_digest=source.publication_semantic_digest,
            history_poses=selection.history_poses,
            future_poses=selection.future_poses,
            target_present=selection.target_present,
            provenance=dict(selection.provenance),
        )
        for source_index, selection in enumerate(selections)
    ]
    return _publish_work_results(
        source_publication_semantic_digest=source.publication_semantic_digest,
        source_config_digest=source_config_digest,
        source_mother_count=len(source.events),
        output_dir=output_dir,
        results=results,
        unseen_prior_config_digest=unseen_prior_config_digest,
        seen_prior_config_digest=seen_prior_config_digest,
    )


def _publish_work_results(
    *,
    source_publication_semantic_digest: str,
    source_config_digest: str,
    source_mother_count: int,
    output_dir: str | Path,
    results: list[_WorkResult],
    unseen_prior_config_digest: str,
    seen_prior_config_digest: str,
) -> Sop05FinalScenarioPublishResult:
    """Write one validated sequence of final work results atomically."""

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite final scenario release: {destination}")
    if [result.record["source_index"] for result in results] != list(range(len(results))):
        raise Sop05FinalScenarioError("final scenario results do not preserve row order")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    staging = staging_root / destination.name
    staging.mkdir()
    try:
        records: list[dict[str, object]] = []
        provenance: list[dict[str, object]] = []
        histories: list[np.ndarray] = []
        futures: list[np.ndarray] = []
        present: list[bool] = []
        source_indices: list[int] = []
        for result in results:
            record = dict(result.record)
            if record["status"] == "accepted":
                if result.history_poses is None or result.future_poses is None:
                    raise Sop05FinalScenarioError("accepted scenario lacks oracle target arrays")
                record["target_row"] = len(histories)
                histories.append(result.history_poses)
                futures.append(result.future_poses)
                present.append(result.target_present)
                source_indices.append(int(record["source_index"]))
            records.append(record)
            provenance.append(dict(result.provenance))
        accepted_count = len(histories)
        deficit_count = len(records) - accepted_count
        full_source_coverage = len(results) == source_mother_count
        manifest = {
            "version": SOP05_FINAL_SCENARIO_VERSION,
            "source_publication_semantic_digest": source_publication_semantic_digest,
            "source_config_digest": source_config_digest,
            "unseen_prior_config_digest": unseen_prior_config_digest,
            "seen_prior_config_digest": seen_prior_config_digest,
            "source_mother_count": source_mother_count,
            "processed_mother_count": len(records),
            "accepted_count": accepted_count,
            "deficit_count": deficit_count,
            "full_source_coverage": full_source_coverage,
            "oracle_payload": _ORACLE_TARGETS,
            "records": _RECORDS,
            "audit_provenance": _PROVENANCE,
        }
        arrays = {
            "history_poses": (
                np.stack(histories).astype(np.float32, copy=False)
                if histories
                else np.empty((0, 8, 3), dtype=np.float32)
            ),
            "future_poses": (
                np.stack(futures).astype(np.float32, copy=False)
                if futures
                else np.empty((0, 32, 3), dtype=np.float32)
            ),
            "target_present": np.asarray(present, dtype=np.bool_),
            "source_record_indices": np.asarray(source_indices, dtype=np.int64),
        }
        (staging / _MANIFEST).write_bytes(_json_file(manifest))
        _write_jsonl(staging / _RECORDS, records)
        _write_deterministic_npz(staging / _ORACLE_TARGETS, arrays)
        _write_jsonl(staging / _PROVENANCE, provenance)
        checksums = {
            name: _sha256_file(staging / name)
            for name in (_MANIFEST, _RECORDS, _ORACLE_TARGETS, _PROVENANCE)
        }
        (staging / _CHECKSUMS).write_bytes(_json_file(checksums))
        (staging / _COMPLETE).write_bytes(
            _json_file(
                {
                    "version": SOP05_FINAL_SCENARIO_VERSION,
                    "source_publication_semantic_digest": source_publication_semantic_digest,
                    "processed_mother_count": len(records),
                    "accepted_count": accepted_count,
                    "deficit_count": deficit_count,
                    "full_source_coverage": full_source_coverage,
                }
            )
        )
        load_sop05_final_scenarios(
            staging,
            expected_source_publication_semantic_digest=(
                source_publication_semantic_digest
            ),
        )
        atomic_rename_noreplace(staging, destination)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    else:
        staging_root.rmdir()
    return Sop05FinalScenarioPublishResult(
        output_dir=destination,
        source_publication_semantic_digest=source_publication_semantic_digest,
        processed_mother_count=len(records),
        accepted_count=accepted_count,
        deficit_count=deficit_count,
        full_source_coverage=full_source_coverage,
    )


def load_sop05_final_scenarios(
    input_dir: str | Path,
    *,
    expected_source_publication_semantic_digest: str | None = None,
) -> LoadedSop05FinalScenarios:
    """Strictly validate a complete finalized SOP05 scenario sidecar."""

    root = Path(input_dir)
    if not root.is_dir() or {path.name for path in root.iterdir()} != _FILE_SET:
        raise Sop05FinalScenarioError("final scenario release file set mismatch")
    manifest = _read_json(root / _MANIFEST)
    complete = _read_json(root / _COMPLETE)
    checksums = _read_json(root / _CHECKSUMS)
    if not isinstance(manifest, dict) or not isinstance(complete, dict):
        raise Sop05FinalScenarioError("final scenario release metadata must be objects")
    if not isinstance(checksums, dict) or set(checksums) != {
        _MANIFEST,
        _RECORDS,
        _ORACLE_TARGETS,
        _PROVENANCE,
    }:
        raise Sop05FinalScenarioError("final scenario release checksum schema mismatch")
    for name, expected in checksums.items():
        if not isinstance(expected, str) or _sha256_file(root / name) != expected:
            raise Sop05FinalScenarioError(f"final scenario release checksum mismatch: {name}")
    required_manifest = {
        "version",
        "source_publication_semantic_digest",
        "source_config_digest",
        "unseen_prior_config_digest",
        "seen_prior_config_digest",
        "source_mother_count",
        "processed_mother_count",
        "accepted_count",
        "deficit_count",
        "full_source_coverage",
        "oracle_payload",
        "records",
        "audit_provenance",
    }
    if set(manifest) != required_manifest or manifest["version"] != SOP05_FINAL_SCENARIO_VERSION:
        raise Sop05FinalScenarioError("final scenario manifest schema mismatch")
    if manifest["oracle_payload"] != _ORACLE_TARGETS or manifest["records"] != _RECORDS:
        raise Sop05FinalScenarioError("final scenario manifest layout mismatch")
    source_digest = manifest["source_publication_semantic_digest"]
    if not isinstance(source_digest, str) or not source_digest:
        raise Sop05FinalScenarioError("final scenario source digest is invalid")
    if (
        expected_source_publication_semantic_digest is not None
        and source_digest != expected_source_publication_semantic_digest
    ):
        raise Sop05FinalScenarioError("final scenario source digest differs")
    count_keys = (
        "source_mother_count",
        "processed_mother_count",
        "accepted_count",
        "deficit_count",
    )
    if any(
        isinstance(manifest[key], bool)
        or not isinstance(manifest[key], int)
        or manifest[key] < 0
        for key in count_keys
    ):
        raise Sop05FinalScenarioError("final scenario counts are invalid")
    if manifest["accepted_count"] + manifest["deficit_count"] != manifest["processed_mother_count"]:
        raise Sop05FinalScenarioError("final scenario counts do not balance")
    if not isinstance(manifest["full_source_coverage"], bool) or (
        manifest["full_source_coverage"]
        != (manifest["processed_mother_count"] == manifest["source_mother_count"])
    ):
        raise Sop05FinalScenarioError("final scenario coverage is invalid")
    records = _read_jsonl(root / _RECORDS)
    provenance = _read_jsonl(root / _PROVENANCE)
    if len(records) != manifest["processed_mother_count"] or len(provenance) != len(records):
        raise Sop05FinalScenarioError("final scenario row counts are invalid")
    mother_ids: set[str] = set()
    accepted_rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        required = {"source_index", "mother_id", "split", "regime", "status"}
        if not required.issubset(record) or record.get("source_index") != index:
            raise Sop05FinalScenarioError("final scenario record index is invalid")
        mother_id = record["mother_id"]
        if not isinstance(mother_id, str) or not mother_id or mother_id in mother_ids:
            raise Sop05FinalScenarioError("final scenario mother identities are invalid")
        mother_ids.add(mother_id)
        if record["status"] == "accepted":
            if set(record) != required | {"scenario_id", "target_present", "target_row"}:
                raise Sop05FinalScenarioError("accepted final scenario record schema mismatch")
            if record["target_row"] != len(accepted_rows) or not isinstance(
                record["target_present"], bool
            ):
                raise Sop05FinalScenarioError("accepted final scenario target row is invalid")
            accepted_rows.append(record)
        elif record["status"] == "deficit":
            if set(record) != required | {"reason"}:
                raise Sop05FinalScenarioError("deficit final scenario record schema mismatch")
        else:
            raise Sop05FinalScenarioError("final scenario status is invalid")
    if len(accepted_rows) != manifest["accepted_count"]:
        raise Sop05FinalScenarioError("accepted final scenario count is invalid")
    for index, row in enumerate(provenance):
        if row.get("source_index") != index or row.get("mother_id") != records[index]["mother_id"]:
            raise Sop05FinalScenarioError("final scenario provenance is misaligned")
    try:
        with np.load(root / _ORACLE_TARGETS, allow_pickle=False) as payload:
            if set(payload.files) != _ORACLE_ARRAYS:
                raise Sop05FinalScenarioError("final scenario oracle array schema mismatch")
            history = np.asarray(payload["history_poses"])
            future = np.asarray(payload["future_poses"])
            present = np.asarray(payload["target_present"])
            source_indices = np.asarray(payload["source_record_indices"])
    except (OSError, ValueError) as exc:
        raise Sop05FinalScenarioError("failed to read final scenario oracle arrays") from exc
    count = len(accepted_rows)
    if (
        history.shape != (count, 8, 3)
        or future.shape != (count, 32, 3)
        or history.dtype != np.dtype(np.float32)
        or future.dtype != np.dtype(np.float32)
        or present.shape != (count,)
        or present.dtype != np.dtype(np.bool_)
        or source_indices.shape != (count,)
        or source_indices.dtype != np.dtype(np.int64)
        or not np.isfinite(history).all()
        or not np.isfinite(future).all()
    ):
        raise Sop05FinalScenarioError("final scenario oracle array values are invalid")
    if source_indices.tolist() != [row["source_index"] for row in accepted_rows]:
        raise Sop05FinalScenarioError("final scenario oracle source indices are invalid")
    if present.tolist() != [row["target_present"] for row in accepted_rows]:
        raise Sop05FinalScenarioError("final scenario oracle target presence is invalid")
    required_complete = {
        "version",
        "source_publication_semantic_digest",
        "processed_mother_count",
        "accepted_count",
        "deficit_count",
        "full_source_coverage",
    }
    if set(complete) != required_complete or complete["version"] != SOP05_FINAL_SCENARIO_VERSION:
        raise Sop05FinalScenarioError("final scenario completion marker schema mismatch")
    for key in required_complete - {"version"}:
        if complete[key] != manifest[key]:
            raise Sop05FinalScenarioError("final scenario completion marker differs")
    return LoadedSop05FinalScenarios(
        root=root,
        manifest=dict(manifest),
        records=tuple(dict(record) for record in records),
        history_poses=np.array(
            history, dtype=np.float32, order="C", copy=True
        ),
        future_poses=np.array(
            future, dtype=np.float32, order="C", copy=True
        ),
        target_present=np.array(
            present, dtype=np.bool_, order="C", copy=True
        ),
        source_record_indices=np.array(
            source_indices, dtype=np.int64, order="C", copy=True
        ),
        accepted_count=len(accepted_rows),
        deficit_count=len(records) - len(accepted_rows),
        full_source_coverage=bool(manifest["full_source_coverage"]),
    )


__all__ = (
    "IncompleteContextError",
    "LoadedSop05FinalScenarios",
    "SOP05_FINAL_SCENARIO_VERSION",
    "Sop05FinalScenarioError",
    "Sop05FinalScenarioPublishResult",
    "Sop05FinalScenarioSelection",
    "load_sop05_final_scenarios",
    "publish_selected_sop05_final_scenarios",
    "publish_sop05_final_scenarios",
)
