"""Extract deployment-observable BaseState objects and separate oracle data."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.contracts import (
    SCHEMA_VERSION,
    BaseState,
    GridSpec,
    OracleContext,
    save_dataclass,
    validate_base_state,
    validate_oracle_context,
)
from src.geometry.transforms import transform_poses_global_to_local
from src.utils.seeding import stable_digest

from .thor_adapter import RecordingIndex, ThorDataError, validate_recording_index


@dataclass(frozen=True)
class BaseStateExtraction:
    """Accepted BaseState/OracleContext pairs and deterministic counts."""

    base_states: tuple[BaseState, ...]
    oracle_contexts: tuple[OracleContext, ...]
    summary: dict[str, object]


def _exact_indices(source: np.ndarray, query: np.ndarray, tolerance: float) -> np.ndarray | None:
    indices = np.searchsorted(source, query)
    if np.any(indices >= source.size):
        return None
    if not np.allclose(source[indices], query, rtol=0.0, atol=tolerance):
        return None
    return indices.astype(np.int64, copy=False)


def _same_segment(segment_ids: np.ndarray, indices: np.ndarray) -> bool:
    return bool(indices.size and np.all(segment_ids[indices] == segment_ids[indices[0]]))


def _object_spec(object_type: str, footprint: dict) -> dict:
    return {"object_type": object_type, "footprint": footprint}


def extract_base_states(
    recording: RecordingIndex,
    *,
    split: str,
    grid: GridSpec,
    stride_s: float = 0.6,
) -> BaseStateExtraction:
    """Extract complete robot-centric windows without oracle/input mixing."""
    validate_recording_index(recording)
    if split not in {"train", "calibration", "val", "test"}:
        raise ThorDataError("split must be train, calibration, val, or test")
    if not math.isfinite(stride_s) or not 0.5 <= stride_s <= 1.0:
        raise ThorDataError("stride_s must be in the SOP range [0.5, 1.0]")
    stride_steps = int(round(stride_s / recording.dt_s))
    if not math.isclose(
        stride_steps * recording.dt_s, stride_s, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ThorDataError("stride_s must be an integer multiple of recording.dt_s")

    first_current = grid.history_steps - 1
    last_current = recording.timestamps.size - grid.future_steps - 1
    candidates = list(range(first_current, last_current + 1, stride_steps))
    base_states: list[BaseState] = []
    oracle_contexts: list[OracleContext] = []
    rejected = {"robot_gap": 0}
    empty_dynamic_count = 0
    half_extent_m = 0.5 * grid.width * grid.resolution_m
    tolerance = max(1e-9, recording.dt_s * 1e-6)

    for current_index in candidates:
        history_indices = np.arange(
            current_index - grid.history_steps + 1,
            current_index + 1,
            dtype=np.int64,
        )
        future_indices = np.arange(
            current_index + 1,
            current_index + 1 + grid.future_steps,
            dtype=np.int64,
        )
        combined_robot_indices = np.concatenate((history_indices, future_indices))
        if not _same_segment(recording.robot_segment_ids, combined_robot_indices):
            rejected["robot_gap"] += 1
            continue

        current_pose = recording.robot_pose[current_index].astype(np.float64)
        robot_history = transform_poses_global_to_local(
            recording.robot_pose[history_indices], current_pose
        ).astype(np.float32)
        history_times = recording.timestamps[history_indices]
        future_times = recording.timestamps[future_indices]
        visible_history: dict[str, np.ndarray] = {}
        visible_specs: dict[str, dict] = {}
        oracle_history: dict[str, np.ndarray] = {}
        oracle_future: dict[str, np.ndarray] = {}
        oracle_specs: dict[str, dict] = {}

        for object_id in sorted(recording.dynamic_objects):
            track = recording.dynamic_objects[object_id]
            object_history_indices = _exact_indices(
                track.timestamps, history_times, tolerance
            )
            if object_history_indices is None or not _same_segment(
                track.segment_ids, object_history_indices
            ):
                continue
            local_history = transform_poses_global_to_local(
                track.poses[object_history_indices], current_pose
            ).astype(np.float32)
            spec = _object_spec(track.object_type, track.footprint)
            current_position = local_history[-1, :2]
            if np.all(np.abs(current_position) <= half_extent_m):
                visible_history[object_id] = local_history
                visible_specs[object_id] = spec
            object_future_indices = _exact_indices(
                track.timestamps, future_times, tolerance
            )
            if object_future_indices is None:
                continue
            combined_object_indices = np.concatenate(
                (object_history_indices, object_future_indices)
            )
            if not _same_segment(track.segment_ids, combined_object_indices):
                continue
            oracle_history[object_id] = local_history
            oracle_future[object_id] = transform_poses_global_to_local(
                track.poses[object_future_indices], current_pose
            ).astype(np.float32)
            oracle_specs[object_id] = spec

        if not visible_history:
            empty_dynamic_count += 1
        timestamp = float(recording.timestamps[current_index])
        state_id = (
            f"{split}-base-"
            f"{stable_digest(recording.recording_id, f'{timestamp:.9f}', size=12)}"
        )
        dynamic_object_ids = tuple(sorted(visible_history))
        metadata = {
            "session_id": recording.session_id,
            "source_file": recording.source_file,
            "coordinate_frame": "robot current pose; x forward; y left; metres",
            "history_dt_s": recording.dt_s,
        }
        static_map = recording.static_map
        if static_map is not None:
            if static_map.shape != (grid.height, grid.width):
                raise ThorDataError("static_map shape must match GridSpec")
            static_map = np.asarray(static_map, dtype=np.float32)
        state = BaseState(
            state_id=state_id,
            split=split,
            recording_id=recording.recording_id,
            dynamic_object_ids=dynamic_object_ids,
            timestamp=timestamp,
            robot_history=robot_history,
            robot_state=recording.robot_twist[current_index].astype(np.float32),
            visible_dynamic_object_history=visible_history,
            visible_dynamic_object_specs=visible_specs,
            static_map_local=static_map,
            metadata=metadata,
        )
        oracle = OracleContext(
            base_state_id=state_id,
            dynamic_object_history=oracle_history,
            dynamic_object_future=oracle_future,
            dynamic_object_specs=oracle_specs,
            metadata={
                "source_recording_id": recording.recording_id,
                "source_dynamic_object_ids": sorted(oracle_history),
                "future_dt_s": recording.dt_s,
                "coordinate_frame": metadata["coordinate_frame"],
            },
        )
        validate_base_state(state, grid)
        validate_oracle_context(oracle, grid)
        base_states.append(state)
        oracle_contexts.append(oracle)

    summary: dict[str, object] = {
        "recording_id": recording.recording_id,
        "split": split,
        "candidate_count": len(candidates),
        "accepted_count": len(base_states),
        "rejected_count": sum(rejected.values()),
        "rejection_reasons": rejected,
        "empty_dynamic_count": empty_dynamic_count,
        "history_steps": grid.history_steps,
        "future_steps": grid.future_steps,
        "dt_s": recording.dt_s,
        "dynamic_object_type_counts": {
            object_type: sum(
                track.object_type == object_type
                for track in recording.dynamic_objects.values()
            )
            for object_type in ("human", "carried_object", "unknown_dynamic")
        },
    }
    return BaseStateExtraction(
        base_states=tuple(base_states),
        oracle_contexts=tuple(oracle_contexts),
        summary=summary,
    )


def extract_base_state_index(
    recordings: list[RecordingIndex] | tuple[RecordingIndex, ...],
    *,
    split: str,
    grid: GridSpec,
    stride_s: float = 0.6,
) -> BaseStateExtraction:
    """Extract and combine one split without losing per-recording provenance."""
    if not recordings:
        raise ThorDataError("recordings must not be empty")
    results = [
        extract_base_states(
            recording, split=split, grid=grid, stride_s=stride_s
        )
        for recording in sorted(recordings, key=lambda item: item.recording_id)
    ]
    base_states = tuple(
        sorted(
            (state for result in results for state in result.base_states),
            key=lambda state: state.state_id,
        )
    )
    oracle_by_id = {
        oracle.base_state_id: oracle
        for result in results
        for oracle in result.oracle_contexts
    }
    if len(oracle_by_id) != len(base_states):
        raise ThorDataError("base state ids must be unique across recordings")
    oracle_contexts = tuple(oracle_by_id[state.state_id] for state in base_states)
    rejection_keys = sorted(
        {
            key
            for result in results
            for key in result.summary["rejection_reasons"]
        }
    )
    rejection_reasons = {
        key: sum(
            int(result.summary["rejection_reasons"].get(key, 0))
            for result in results
        )
        for key in rejection_keys
    }
    summary: dict[str, object] = {
        "split": split,
        "recording_count": len(recordings),
        "candidate_count": sum(
            int(result.summary["candidate_count"]) for result in results
        ),
        "accepted_count": len(base_states),
        "rejected_count": sum(rejection_reasons.values()),
        "rejection_reasons": rejection_reasons,
        "empty_dynamic_count": sum(
            int(result.summary["empty_dynamic_count"]) for result in results
        ),
        "history_steps": grid.history_steps,
        "future_steps": grid.future_steps,
        "dt_s": results[0].summary["dt_s"],
        "stride_s": stride_s,
        "dynamic_object_type_counts": {
            object_type: sum(
                int(result.summary["dynamic_object_type_counts"][object_type])
                for result in results
            )
            for object_type in ("human", "carried_object", "unknown_dynamic")
        },
    }
    return BaseStateExtraction(
        base_states=base_states,
        oracle_contexts=oracle_contexts,
        summary=summary,
    )


def _json_bytes(payload: object, *, lines: bool = False) -> bytes:
    if lines:
        rows = payload if isinstance(payload, list) else []
        text = "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        )
    else:
        text = json.dumps(
            payload, sort_keys=True, indent=2, allow_nan=False
        ) + "\n"
    return text.encode("utf-8")


def write_base_state_extraction(
    extraction: BaseStateExtraction, output_dir: str | Path
) -> dict[str, Path]:
    """Atomically write observable and oracle artifacts to separate trees."""
    if len(extraction.base_states) != len(extraction.oracle_contexts):
        raise ThorDataError("base state and oracle context counts must match")
    pairs = sorted(
        zip(extraction.base_states, extraction.oracle_contexts),
        key=lambda pair: pair[0].state_id,
    )
    if any(state.state_id != oracle.base_state_id for state, oracle in pairs):
        raise ThorDataError("base state and oracle context ids must match")

    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_name(f".{output_path.name}.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    base_dir = staging / "base_states"
    oracle_dir = staging / "oracle_contexts"
    base_dir.mkdir(parents=True)
    oracle_dir.mkdir(parents=True)
    base_manifest: list[dict[str, object]] = []
    oracle_manifest: list[dict[str, object]] = []
    try:
        for state, oracle in pairs:
            save_dataclass(state, base_dir / state.state_id)
            save_dataclass(oracle, oracle_dir / oracle.base_state_id)
            base_manifest.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "state_id": state.state_id,
                    "split": state.split,
                    "recording_id": state.recording_id,
                    "dynamic_object_ids": list(state.dynamic_object_ids),
                    "timestamp": state.timestamp,
                    "base_state_file": f"base_states/{state.state_id}.npz",
                }
            )
            oracle_manifest.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "base_state_id": oracle.base_state_id,
                    "source_recording_id": oracle.metadata.get(
                        "source_recording_id"
                    ),
                    "source_dynamic_object_ids": oracle.metadata.get(
                        "source_dynamic_object_ids", []
                    ),
                    "oracle_context_file": (
                        f"oracle_contexts/{oracle.base_state_id}.npz"
                    ),
                }
            )
        summary = {"schema_version": SCHEMA_VERSION, **extraction.summary}
        (staging / "base_state_manifest.jsonl").write_bytes(
            _json_bytes(base_manifest, lines=True)
        )
        (staging / "oracle_context_manifest.jsonl").write_bytes(
            _json_bytes(oracle_manifest, lines=True)
        )
        (staging / "summary.json").write_bytes(_json_bytes(summary))
        staging.replace(output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "base_states": output_path / "base_states",
        "oracle_contexts": output_path / "oracle_contexts",
        "manifest": output_path / "base_state_manifest.jsonl",
        "oracle_manifest": output_path / "oracle_context_manifest.jsonl",
        "summary": output_path / "summary.json",
    }
