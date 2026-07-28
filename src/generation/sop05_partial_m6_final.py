"""Direct final-scenario views over an interrupted M6 staging release."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from src.contracts import BaseState, OracleWorld, load_dataclass
from src.geometry import CircleOccluder, RectangleOccluder

from .anchored_human_placement import synchronized_centerline_blocking
from .sop05_final_scenarios import (
    _CHECKSUMS,
    _COMPLETE,
    _MANIFEST,
    _ORACLE_TARGETS,
    _PROVENANCE,
    _RECORDS,
    SOP05_FINAL_SCENARIO_VERSION,
    Sop05FinalScenarioPublishResult,
    _WorkResult,
    _json_file,
    _sha256_file,
    _write_deterministic_npz,
    _write_jsonl,
    load_sop05_final_scenarios,
    _process_event,
)
from .sop05_seen_prior import SeenPriorConfig
from .sop05_unseen_prior import UnseenPriorConfig
from src.utils.atomic_publish import atomic_rename_noreplace


class PartialM6Error(ValueError):
    """Raised when interrupted M6 artifacts cannot form a final-scenario view."""


@dataclass(frozen=True)
class PartialTarget:
    target_dynamic_object_id: str
    source_recording_id: str
    source_session_id: str
    source_object_id: str
    snippet_id: str
    object_type: str
    footprint_spec: Mapping[str, object]
    history_poses: np.ndarray
    current_pose: np.ndarray
    future_poses: np.ndarray
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class PartialEvent:
    generated_event_id: str
    world: OracleWorld
    target: PartialTarget
    target_visibility_history: np.ndarray


@dataclass(frozen=True)
class PartialMotherView:
    event: PartialEvent
    state: BaseState


@dataclass(frozen=True)
class _PartialSource:
    root: Path
    trajectory_rows: tuple[Mapping[str, object], ...]
    target_rows: tuple[Mapping[str, object], ...]
    history_poses: np.ndarray
    current_poses: np.ndarray
    future_poses: np.ndarray
    source_identity: str


@dataclass(frozen=True)
class _PartialWorkContext:
    partial: _PartialSource
    source_states: Mapping[str, BaseState]
    snippet_sources: Mapping[str, tuple[str, str]]
    base_config: Mapping[str, object]
    centerline_epsilon_m: float
    unseen_config: UnseenPriorConfig
    seen_config: SeenPriorConfig


_PARTIAL_WORK_CONTEXT: _PartialWorkContext | None = None


def _occluder(item: Mapping[str, object]) -> CircleOccluder | RectangleOccluder:
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
    raise PartialM6Error("partial M6 world contains an unsupported occluder")


def _require_same(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise PartialM6Error(f"partial M6 {label} mismatch")


def build_partial_mother_view(
    *,
    trajectory_row: Mapping[str, object],
    target_row: Mapping[str, object],
    target_history: np.ndarray,
    target_current: np.ndarray,
    target_future: np.ndarray,
    world: OracleWorld,
    source_state: BaseState,
    source_recording_id: str,
    source_session_id: str,
    centerline_epsilon_m: float,
) -> PartialMotherView:
    """Reconstruct only the mother fields consumed by final A/B generation."""

    event_id = str(target_row["generated_event_id"])
    decision_id = str(target_row["base_state_id"])
    trajectory_id = str(target_row["trajectory_id"])
    target_id = str(target_row["target_dynamic_object_id"])
    _require_same(trajectory_row.get("event_id"), event_id, "event ID")
    _require_same(trajectory_row.get("source_base_state_id"), source_state.state_id, "source state ID")
    _require_same(trajectory_row.get("decision_state_id"), decision_id, "decision state ID")
    _require_same(trajectory_row.get("nominal_trajectory_id"), trajectory_id, "trajectory ID")
    _require_same(target_row.get("world_id"), world.world_id, "world ID")
    _require_same(world.base_state_id, decision_id, "world decision state ID")

    history = np.asarray(target_history, dtype=np.float32)
    current = np.asarray(target_current, dtype=np.float32)
    future = np.asarray(target_future, dtype=np.float32)
    if history.shape != (8, 3) or current.shape != (3,) or future.shape != (32, 3):
        raise PartialM6Error("partial M6 target motion shape mismatch")
    if not np.array_equal(current, history[7]):
        raise PartialM6Error("partial M6 current pose differs from history index 7")

    represented_ids = set(world.dynamic_object_trajectories) & set(
        world.dynamic_object_specs
    )
    context_ids = tuple(
        object_id
        for object_id in source_state.dynamic_object_ids
        if object_id != target_id
        and object_id in represented_ids
        and object_id in source_state.visible_dynamic_object_history
        and object_id in source_state.visible_dynamic_object_specs
    )
    state = BaseState(
        state_id=decision_id,
        split=source_state.split,
        recording_id=source_state.recording_id,
        dynamic_object_ids=context_ids,
        timestamp=source_state.timestamp,
        robot_history=np.asarray(source_state.robot_history, dtype=np.float32),
        robot_state=np.asarray(source_state.robot_state, dtype=np.float32),
        visible_dynamic_object_history={
            object_id: np.asarray(
                source_state.visible_dynamic_object_history[object_id],
                dtype=np.float32,
            )
            for object_id in context_ids
        },
        visible_dynamic_object_specs={
            object_id: dict(source_state.visible_dynamic_object_specs[object_id])
            for object_id in context_ids
        },
        static_map_local=np.asarray(world.static_occupancy, dtype=np.float32),
        metadata={
            "source_base_state_id": source_state.state_id,
            "partial_m6_direct_view": True,
            **(
                {"session_id": source_state.metadata["session_id"]}
                if isinstance(source_state.metadata.get("session_id"), str)
                and source_state.metadata["session_id"]
                else {}
            ),
        },
    )
    blocked, _ = synchronized_centerline_blocking(
        state.robot_history[:, :2],
        history[:, :2],
        tuple(_occluder(dict(item)) for item in world.occluders),
        epsilon_m=float(centerline_epsilon_m),
    )
    footprint_spec = dict(target_row["footprint_spec"])
    provenance = MappingProxyType(
        {
            "source_recording_id": source_recording_id,
            "source_session_id": source_session_id,
            "source_snippet_id": str(target_row["source_snippet_id"]),
            "source_object_id": str(target_row["source_object_id"]),
        }
    )
    target = PartialTarget(
        target_dynamic_object_id=target_id,
        source_recording_id=source_recording_id,
        source_session_id=source_session_id,
        source_object_id=str(target_row["source_object_id"]),
        snippet_id=str(target_row["source_snippet_id"]),
        object_type=str(target_row["object_type"]),
        footprint_spec=MappingProxyType(footprint_spec),
        history_poses=np.array(history, copy=True),
        current_pose=np.array(current, copy=True),
        future_poses=np.array(future, copy=True),
        provenance=provenance,
    )
    return PartialMotherView(
        event=PartialEvent(
            generated_event_id=event_id,
            world=world,
            target=target,
            target_visibility_history=np.asarray(~blocked, dtype=np.bool_),
        ),
        state=state,
    )


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PartialM6Error(f"failed to read partial M6 {path.name}") from exc


def _load_partial_source(
    root: Path,
    *,
    source_config_digest: str,
    max_mothers: int | None,
) -> _PartialSource:
    trajectory_root = root / "trajectory_store"
    target_root = root / "target_motion"
    trajectory_manifest = _read_json(trajectory_root / "manifest.json")
    target_summary = _read_json(target_root / "shard_summary.json")
    if not isinstance(trajectory_manifest, dict) or not isinstance(target_summary, dict):
        raise PartialM6Error("partial M6 manifests must be JSON objects")
    raw_trajectory_rows = trajectory_manifest.get("records")
    if not isinstance(raw_trajectory_rows, list) or any(
        not isinstance(row, dict) for row in raw_trajectory_rows
    ):
        raise PartialM6Error("partial M6 trajectory records are invalid")
    try:
        target_lines = (
            target_root / "generated_event_manifest.jsonl"
        ).read_text(encoding="ascii").splitlines()
        raw_target_rows = [json.loads(line) for line in target_lines]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PartialM6Error("partial M6 target manifest is invalid") from exc
    if any(not isinstance(row, dict) for row in raw_target_rows):
        raise PartialM6Error("partial M6 target records are invalid")
    source_count = len(raw_target_rows)
    if (
        len(raw_trajectory_rows) != source_count
        or trajectory_manifest.get("record_count") != source_count
        or target_summary.get("record_count") != source_count
    ):
        raise PartialM6Error("partial M6 record counts differ")
    for index, (trajectory_row, target_row) in enumerate(
        zip(raw_trajectory_rows, raw_target_rows)
    ):
        if (
            target_row.get("row_index") != index
            or trajectory_row.get("event_id") != target_row.get("generated_event_id")
            or trajectory_row.get("decision_state_id") != target_row.get("base_state_id")
            or trajectory_row.get("nominal_trajectory_id")
            != target_row.get("trajectory_id")
        ):
            raise PartialM6Error("partial M6 trajectory/target join differs")
    selected_count = source_count if max_mothers is None else min(max_mothers, source_count)
    payload_path = target_root / "event_target_motion_history8_future32_v2.npz"
    try:
        with np.load(payload_path, allow_pickle=False) as payload:
            history = payload["history_poses"][:selected_count].copy()
            current = payload["current_poses"][:selected_count].copy()
            future = payload["future_poses"][:selected_count].copy()
    except (OSError, KeyError, ValueError) as exc:
        raise PartialM6Error("partial M6 target payload is invalid") from exc
    if (
        history.shape != (selected_count, 8, 3)
        or current.shape != (selected_count, 3)
        or future.shape != (selected_count, 32, 3)
        or not np.array_equal(current, history[:, 7])
    ):
        raise PartialM6Error("partial M6 target payload layout differs")
    identity_payload = {
        "version": "sop05_partial_m6_direct_source_v1",
        "source_config_digest": source_config_digest,
        "source_mother_count": source_count,
        "trajectory_collection_digest": trajectory_manifest.get(
            "collection_semantic_digest"
        ),
        "target_manifest_digest": target_summary.get("manifest_digest"),
        "target_payload_semantic_digest": target_summary.get(
            "payload_semantic_digest"
        ),
    }
    source_identity = hashlib.sha256(
        json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return _PartialSource(
        root=root,
        trajectory_rows=tuple(raw_trajectory_rows[:selected_count]),
        target_rows=tuple(raw_target_rows[:selected_count]),
        history_poses=history,
        current_poses=current,
        future_poses=future,
        source_identity=source_identity,
    )


def load_partial_m6_source(
    input_root: str | Path,
    *,
    source_config_digest: str,
    max_mothers: int | None = None,
) -> _PartialSource:
    """Strictly load the authenticated partial-M6 inputs without finalizing them."""

    return _load_partial_source(
        Path(input_root),
        source_config_digest=source_config_digest,
        max_mothers=max_mothers,
    )


def _partial_worker(index: int) -> _WorkResult:
    context = _PARTIAL_WORK_CONTEXT
    if context is None:
        raise RuntimeError("partial M6 worker was not initialized")
    partial = context.partial
    trajectory_row = partial.trajectory_rows[index]
    target_row = partial.target_rows[index]
    source_id = str(trajectory_row["source_base_state_id"])
    source_state = context.source_states.get(source_id)
    if source_state is None:
        raise PartialM6Error(f"partial M6 source state is missing: {source_id}")
    snippet_id = str(target_row["source_snippet_id"])
    snippet_source = context.snippet_sources.get(snippet_id)
    if snippet_source is None:
        raise PartialM6Error(f"partial M6 snippet source is missing: {snippet_id}")
    target_root = partial.root / "target_motion"
    world_path = (target_root / str(target_row["world_file"])).resolve()
    try:
        world_path.relative_to(target_root.resolve())
    except ValueError as exc:
        raise PartialM6Error("partial M6 world path escapes target root") from exc
    world = load_dataclass(world_path)
    if not isinstance(world, OracleWorld):
        raise PartialM6Error("partial M6 world payload is not an OracleWorld")
    view = build_partial_mother_view(
        trajectory_row=trajectory_row,
        target_row=target_row,
        target_history=partial.history_poses[index],
        target_current=partial.current_poses[index],
        target_future=partial.future_poses[index],
        world=world,
        source_state=source_state,
        source_recording_id=snippet_source[0],
        source_session_id=snippet_source[1],
        centerline_epsilon_m=context.centerline_epsilon_m,
    )
    fake_source = SimpleNamespace(
        trajectories=SimpleNamespace(
            records=(
                SimpleNamespace(
                    event_id=view.event.generated_event_id,
                    decision_state_id=view.state.state_id,
                ),
            )
        ),
        decision_states={view.state.state_id: view.state},
        manifest={"base_config": dict(context.base_config)},
        publication_semantic_digest=partial.source_identity,
    )
    return _process_event(
        fake_source,
        view.event,
        index,
        unseen_config=context.unseen_config,
        seen_config=context.seen_config,
    )


def _process_partial_all(
    partial: _PartialSource,
    *,
    source_states: Mapping[str, BaseState],
    snippet_sources: Mapping[str, tuple[str, str]],
    base_config: Mapping[str, object],
    centerline_epsilon_m: float,
    unseen_config: UnseenPriorConfig,
    seen_config: SeenPriorConfig,
    workers: int,
    progress_callback: Callable[[int, int], None] | None,
) -> list[_WorkResult]:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    context = _PartialWorkContext(
        partial=partial,
        source_states=source_states,
        snippet_sources=snippet_sources,
        base_config=base_config,
        centerline_epsilon_m=float(centerline_epsilon_m),
        unseen_config=unseen_config,
        seen_config=seen_config,
    )
    total = len(partial.target_rows)
    global _PARTIAL_WORK_CONTEXT
    previous = _PARTIAL_WORK_CONTEXT
    _PARTIAL_WORK_CONTEXT = context
    try:
        if workers == 1:
            results: list[_WorkResult] = []
            for index in range(total):
                results.append(_partial_worker(index))
                if progress_callback is not None:
                    progress_callback(index + 1, total)
            return results
        if multiprocessing.get_start_method(allow_none=True) not in (None, "fork"):
            raise PartialM6Error("parallel partial M6 release requires fork")
        mp_context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=min(workers, total),
            mp_context=mp_context,
        ) as executor:
            results = []
            for completed, result in enumerate(
                executor.map(_partial_worker, range(total), chunksize=8), start=1
            ):
                results.append(result)
                if progress_callback is not None:
                    progress_callback(completed, total)
            return results
    finally:
        _PARTIAL_WORK_CONTEXT = previous


def _publish_results(
    *,
    results: Sequence[_WorkResult],
    destination: Path,
    source_identity: str,
    source_config_digest: str,
    unseen_config: UnseenPriorConfig,
    seen_config: SeenPriorConfig,
    source_mother_count: int,
    full_source_coverage: bool,
) -> Sop05FinalScenarioPublishResult:
    records: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    histories: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    present: list[bool] = []
    source_indices: list[int] = []
    for result in results:
        record = dict(result.record)
        if record.get("status") == "accepted":
            if result.history_poses is None or result.future_poses is None:
                raise PartialM6Error("accepted partial M6 scenario lacks target arrays")
            record["target_row"] = len(histories)
            histories.append(result.history_poses)
            futures.append(result.future_poses)
            present.append(result.target_present)
            source_indices.append(int(record["source_index"]))
        records.append(record)
        provenance.append(dict(result.provenance))
    accepted_count = len(histories)
    deficit_count = len(records) - accepted_count
    manifest = {
        "version": SOP05_FINAL_SCENARIO_VERSION,
        "source_publication_semantic_digest": source_identity,
        "source_config_digest": source_config_digest,
        "unseen_prior_config_digest": unseen_config.config_digest,
        "seen_prior_config_digest": seen_config.digest,
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    staging = staging_root / destination.name
    staging.mkdir()
    try:
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
                    "source_publication_semantic_digest": source_identity,
                    "processed_mother_count": len(records),
                    "accepted_count": accepted_count,
                    "deficit_count": deficit_count,
                    "full_source_coverage": full_source_coverage,
                }
            )
        )
        load_sop05_final_scenarios(
            staging,
            expected_source_publication_semantic_digest=source_identity,
        )
        atomic_rename_noreplace(staging, destination)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    else:
        staging_root.rmdir()
    return Sop05FinalScenarioPublishResult(
        output_dir=destination,
        source_publication_semantic_digest=source_identity,
        processed_mother_count=len(records),
        accepted_count=accepted_count,
        deficit_count=deficit_count,
        full_source_coverage=full_source_coverage,
    )


def publish_partial_m6_final_scenarios(
    input_root: str | Path,
    *,
    source_states: Mapping[str, BaseState],
    snippet_sources: Mapping[str, tuple[str, str]],
    base_config: Mapping[str, object],
    source_config_digest: str,
    centerline_epsilon_m: float,
    output_dir: str | Path,
    unseen_config: UnseenPriorConfig,
    seen_config: SeenPriorConfig,
    workers: int = 1,
    max_mothers: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Sop05FinalScenarioPublishResult:
    """Publish final A/B scenarios directly from completed partial M6 stores."""

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite final scenario release: {destination}")
    partial = _load_partial_source(
        Path(input_root),
        source_config_digest=source_config_digest,
        max_mothers=max_mothers,
    )
    results = _process_partial_all(
        partial,
        source_states=source_states,
        snippet_sources=snippet_sources,
        base_config=base_config,
        centerline_epsilon_m=centerline_epsilon_m,
        unseen_config=unseen_config,
        seen_config=seen_config,
        workers=workers,
        progress_callback=progress_callback,
    )
    if [result.record.get("source_index") for result in results] != list(
        range(len(partial.target_rows))
    ):
        raise PartialM6Error("partial M6 worker results do not preserve source order")
    source_count = int(
        _read_json(Path(input_root) / "target_motion" / "shard_summary.json")[
            "record_count"
        ]
    )
    return _publish_results(
        results=results,
        destination=destination,
        source_identity=partial.source_identity,
        source_config_digest=source_config_digest,
        unseen_config=unseen_config,
        seen_config=seen_config,
        source_mother_count=source_count,
        full_source_coverage=len(partial.target_rows) == source_count,
    )


__all__ = [
    "PartialEvent",
    "PartialM6Error",
    "PartialMotherView",
    "PartialTarget",
    "build_partial_mother_view",
    "load_partial_m6_source",
    "publish_partial_m6_final_scenarios",
]
