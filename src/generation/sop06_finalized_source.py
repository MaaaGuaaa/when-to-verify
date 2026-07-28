"""Strict persisted SOP05 final-release sources for SOP06 history rendering."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from src.contracts import (
    SCHEMA_VERSION,
    BaseState,
    OracleWorld,
    build_grid_spec,
    load_dataclass,
    validate_base_state,
)
from src.utils.config import config_digest

from .sop05_final_scenarios import (
    LoadedSop05FinalScenarios,
    load_sop05_final_scenarios,
)
from .sop05_partial_m6_final import (
    build_partial_mother_view,
    load_partial_m6_source,
)
from .sop05r_teb_long40_inputs import load_sop05r_teb_long40_inputs
from .sop05r_teb_output_loader import (
    SOP05R_TEB_EVENT_ROW_VERSION,
    compute_sop05r_teb_publication_semantic_digest,
)
from .event_target_motion_shard import (
    EventTargetMotionRecord,
    EventTargetMotionSelectionReader,
    load_event_target_motion_selection,
)
from .sop05r_contracts import (
    SOP05R_TEB_COMPLETION_MARKER_VERSION,
    SOP05R_TEB_GENERATOR_VERSION,
    SOP05R_TEB_MANIFEST_VERSION,
    SOP05R_TEB_RUN_VERSION,
    SOP05R_TEB_SUMMARY_VERSION,
    SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION,
)
from .sop05r_teb_templates import canonical_sop05r_teb_base_state_digest
from .sop05r_teb_trajectory_store import (
    open_sop05r_teb_trajectory_selection,
)
from .sop06_pipeline import (
    Sop06SinglePublication,
    Sop06SinglePublicationContext,
    Sop06SingleRendererInput,
    adapt_finalized_sop05_scenario,
    adapt_finalized_sop05_renderer_input,
)


_SOURCE_MODES = frozenset({"complete_mother", "partial_m6_reconstruction"})
_SPLITS = frozenset({"train", "calibration", "val", "test"})
_COMPLETE_LINEAGE_EVIDENCE_FIELDS = (
    "input_kind",
    "recording_index_manifest_sha256",
    "recording_index_summary_sha256",
    "recording_count",
    "base_state_start",
    "long40_checksum_manifest_sha256",
    "long40_semantic_digest",
    "accepted_source_snippet_count",
    "materialized_base_state_count",
)


@dataclass(frozen=True)
class Sop06AcceptedFinalRecord:
    source_index: int
    mother_id: str
    scenario_id: str
    split: str
    regime: str
    target_present: bool
    target_row: int


@dataclass(frozen=True)
class ResolvedSop06Scenario:
    accepted: Sop06AcceptedFinalRecord
    publication: Sop06SinglePublication


@dataclass(frozen=True)
class _CompleteMotherPayload:
    event_row: Mapping[str, object]
    motion_record: EventTargetMotionRecord
    state: BaseState
    world: OracleWorld
    trajectory: object


@dataclass(frozen=True)
class _CompleteMotherLineage:
    """Authenticated scalar provenance recovered from SOP3 and Long40."""

    base_sources: Mapping[str, tuple[str, str]]
    snippet_sources: Mapping[str, tuple[str, str, str]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_sources", MappingProxyType(dict(self.base_sources)))
        object.__setattr__(
            self,
            "snippet_sources",
            MappingProxyType(dict(self.snippet_sources)),
        )


@dataclass(frozen=True)
class _CompleteResolutionState:
    root: Path
    event_rows: tuple[Mapping[str, object], ...]
    motion_reader: EventTargetMotionSelectionReader
    trajectory_reader: object
    payloads: Mapping[str, _CompleteMotherPayload]
    lineage: _CompleteMotherLineage | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_rows",
            tuple(MappingProxyType(dict(row)) for row in self.event_rows),
        )
        object.__setattr__(self, "payloads", MappingProxyType(dict(self.payloads)))


@dataclass(frozen=True)
class _PartialResolutionState:
    partial: object
    source_states: Mapping[str, object]
    snippet_sources: Mapping[str, tuple[str, str]]
    trajectory_reader: object
    trajectories: Mapping[str, object]
    centerline_epsilon_m: float


def _read_json(path: Path, *, name: str) -> object:
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read {name}") from exc


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _complete_mother_lineage(
    *,
    manifest: Mapping[str, object],
    state: _CompleteResolutionState,
    accepted: Sequence[Sop06AcceptedFinalRecord],
    split: str,
    sop03_root: str | Path,
    long40_human_artifact: str | Path,
) -> _CompleteMotherLineage:
    """Recover legacy complete-mother IDs only from authenticated sources."""

    source_evidence = manifest.get("source_evidence")
    if not isinstance(source_evidence, Mapping):
        raise ValueError("complete mother source_evidence is unavailable")
    missing_evidence = [
        field
        for field in _COMPLETE_LINEAGE_EVIDENCE_FIELDS
        if field not in source_evidence
    ]
    if missing_evidence:
        raise ValueError(
            "complete mother source_evidence cannot authenticate lineage: "
            f"{sorted(missing_evidence)}"
        )
    if source_evidence.get("input_kind") != "sop03_schema4_long40_runtime_v1":
        raise ValueError("complete mother source_evidence input_kind is invalid")
    base_state_start = source_evidence["base_state_start"]
    max_base_states = source_evidence["materialized_base_state_count"]
    if (
        isinstance(base_state_start, bool)
        or not isinstance(base_state_start, int)
        or base_state_start < 0
    ):
        raise ValueError("complete mother source_evidence base_state_start is invalid")
    if (
        isinstance(max_base_states, bool)
        or not isinstance(max_base_states, int)
        or max_base_states <= 0
    ):
        raise ValueError(
            "complete mother source_evidence materialized_base_state_count is invalid"
        )

    required_state_ids: set[str] = set()
    required_snippet_ids: set[str] = set()
    motion_rows: dict[str, Mapping[str, object]] = {}
    for record in accepted:
        event_row = state.event_rows[record.source_index]
        if event_row.get("event_id") != record.mother_id:
            raise ValueError("complete mother event/source index differs")
        source_state_id = _nonempty_string(
            event_row.get("source_base_state_id"),
            name="complete mother source_base_state_id",
        )
        motion_row = state.motion_reader.rows_by_event_id.get(record.mother_id)
        if not isinstance(motion_row, Mapping):
            raise ValueError("complete mother target-motion row is missing")
        snippet_id = _nonempty_string(
            motion_row.get("source_snippet_id"),
            name="complete mother source_snippet_id",
        )
        required_state_ids.add(source_state_id)
        required_snippet_ids.add(snippet_id)
        motion_rows[record.mother_id] = motion_row

    base_config = manifest.get("base_config")
    if not isinstance(base_config, Mapping):
        raise ValueError("complete mother base_config is invalid")
    inputs = load_sop05r_teb_long40_inputs(
        recording_root=sop03_root,
        long40_human_artifact=long40_human_artifact,
        split=split,
        grid=build_grid_spec(dict(base_config)),
        base_state_start=base_state_start,
        max_base_states=max_base_states,
        required_state_ids=frozenset(required_state_ids),
    )
    observed_evidence = inputs.source_evidence
    for field in _COMPLETE_LINEAGE_EVIDENCE_FIELDS:
        if field == "materialized_base_state_count":
            observed = observed_evidence.get("selected_base_state_count")
        else:
            observed = observed_evidence.get(field)
        if observed != source_evidence[field]:
            raise ValueError(
                "complete mother lineage source evidence differs for "
                f"{field}"
            )

    base_sources: dict[str, tuple[str, str]] = {}
    for source_state, _ in inputs.state_pairs:
        session_id = _nonempty_string(
            source_state.metadata.get("session_id"),
            name="recovered base_session_id",
        )
        base_sources[source_state.state_id] = (
            _nonempty_string(
                source_state.recording_id,
                name="recovered base_recording_id",
            ),
            session_id,
        )
    if set(base_sources) != required_state_ids:
        raise ValueError("complete mother recovered BaseState IDs differ")

    snippets_by_id = {snippet.snippet_id: snippet for snippet in inputs.snippets}
    if len(snippets_by_id) != len(inputs.snippets):
        raise ValueError("complete mother recovered snippet IDs are not unique")
    snippet_sources: dict[str, tuple[str, str, str]] = {}
    for mother_id, motion_row in motion_rows.items():
        snippet_id = _nonempty_string(
            motion_row.get("source_snippet_id"),
            name="complete mother source_snippet_id",
        )
        snippet = snippets_by_id.get(snippet_id)
        if snippet is None:
            raise ValueError("complete mother source snippet is missing")
        if (
            snippet.source_object_id != motion_row.get("source_object_id")
            or snippet.object_type != motion_row.get("object_type")
        ):
            raise ValueError("complete mother source snippet identity differs")
        snippet_sources[snippet_id] = (
            _nonempty_string(
                snippet.source_recording_id,
                name="recovered source_recording_id",
            ),
            _nonempty_string(
                snippet.source_session_id,
                name="recovered source_session_id",
            ),
            _nonempty_string(
                snippet.source_object_id,
                name="recovered source_object_id",
            ),
        )
    if set(snippet_sources) != required_snippet_ids:
        raise ValueError("complete mother recovered snippet IDs differ")
    return _CompleteMotherLineage(
        base_sources=base_sources,
        snippet_sources=snippet_sources,
    )


def _validate_complete_event_row(
    row: object,
    *,
    row_index: int,
    grid,
) -> Mapping[str, object]:
    if not isinstance(row, dict):
        raise ValueError("SOP05R TEB event row must be a mapping")
    if row.get("row_version") != SOP05R_TEB_EVENT_ROW_VERSION:
        raise ValueError("SOP05R TEB event row version mismatch")
    if row.get("row_index") != row_index:
        raise ValueError("SOP05R TEB event row index mismatch")
    semantic = dict(row)
    stored_digest = semantic.pop("record_semantic_digest", None)
    encoded = json.dumps(
        semantic,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    expected_digest = hashlib.sha256(
        b"sop05r_teb_event_row_v1\0" + encoded
    ).hexdigest()
    if stored_digest != expected_digest:
        raise ValueError("SOP05R TEB event row semantic digest mismatch")
    event_id = row.get("event_id")
    decision_id = row.get("decision_state_id")
    filename = row.get("decision_state_file")
    decision_digest = row.get("decision_state_digest")
    world_id = row.get("world_id")
    if not all(
        isinstance(value, str) and value
        for value in (event_id, decision_id, filename, decision_digest, world_id)
    ):
        raise ValueError("SOP05R TEB event identity fields are invalid")
    if filename != f"{decision_id}.npz":
        raise ValueError("SOP05R TEB decision-state filename mismatch")
    if not isinstance(row.get("target_provenance"), dict):
        raise ValueError("SOP05R TEB target provenance must be a mapping")
    visibility = np.asarray(row.get("target_visibility_history"), dtype=np.bool_)
    if visibility.shape != (grid.history_steps,):
        raise ValueError("SOP05R TEB history visibility shape is invalid")
    return row


def _load_complete_resolution_state(
    source_root: Path,
    *,
    accepted: Sequence[Sop06AcceptedFinalRecord],
) -> tuple[Mapping[str, object], _CompleteResolutionState, str]:
    if not source_root.is_dir() or not (source_root / "COMPLETE.json").is_file():
        raise ValueError("complete SOP05R mother release is unavailable")
    expected_root = {
        "manifest.json",
        "generation_summary.json",
        "events.json",
        "checksums.json",
        "COMPLETE.json",
        "decision_states",
        "target_motion",
        "trajectory_store",
    }
    if {path.name for path in source_root.iterdir()} != expected_root:
        raise ValueError("complete SOP05R mother root file set differs")
    manifest = _read_json(source_root / "manifest.json", name="SOP05R manifest")
    summary = _read_json(
        source_root / "generation_summary.json", name="SOP05R summary"
    )
    rows = _read_json(source_root / "events.json", name="SOP05R events")
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        raise ValueError("SOP05R manifest or summary is invalid")
    if not isinstance(rows, list):
        raise ValueError("SOP05R events are invalid")
    if manifest.get("manifest_version") != SOP05R_TEB_MANIFEST_VERSION:
        raise ValueError("SOP05R manifest version mismatch")
    if manifest.get("run_version") != SOP05R_TEB_RUN_VERSION:
        raise ValueError("SOP05R run version mismatch")
    if manifest.get("generator_algorithm_version") != SOP05R_TEB_GENERATOR_VERSION:
        raise ValueError("SOP05R generator version mismatch")
    if summary.get("summary_version") != SOP05R_TEB_SUMMARY_VERSION:
        raise ValueError("SOP05R summary version mismatch")
    base_config = manifest.get("base_config")
    if not isinstance(base_config, dict):
        raise ValueError("SOP05R base config is invalid")
    grid = build_grid_spec(base_config)
    event_rows = tuple(
        _validate_complete_event_row(row, row_index=index, grid=grid)
        for index, row in enumerate(rows)
    )
    event_ids = tuple(str(row["event_id"]) for row in event_rows)
    if event_ids != tuple(sorted(event_ids)) or len(set(event_ids)) != len(event_ids):
        raise ValueError("SOP05R event IDs are invalid")
    if manifest.get("event_ids") != list(event_ids):
        raise ValueError("SOP05R manifest event IDs differ")
    decision_digests = [str(row["decision_state_digest"]) for row in event_rows]
    motion_reader = load_event_target_motion_selection(
        source_root / "target_motion",
        expected_payload_semantic_digest=manifest.get("target_motion_payload_digest"),
    )
    if set(motion_reader.rows_by_event_id) != set(event_ids):
        raise ValueError("SOP05R target-motion event IDs differ")
    for record in accepted:
        if not 0 <= record.source_index < len(event_rows):
            raise ValueError("final source index exceeds mother collection")
        if event_rows[record.source_index]["event_id"] != record.mother_id:
            raise ValueError("final record mother differs from source order")
    requested_mothers = tuple(
        dict.fromkeys(record.mother_id for record in accepted)
    )
    trajectory_reader = open_sop05r_teb_trajectory_selection(
        source_root / "trajectory_store",
        event_ids=requested_mothers,
    )
    if trajectory_reader.collection_semantic_digest != manifest.get(
        "trajectory_collection_digest"
    ):
        raise ValueError("SOP05R trajectory collection digest differs")
    publication_digest = compute_sop05r_teb_publication_semantic_digest(
        event_rows=list(event_rows),
        trajectory_collection_digest=trajectory_reader.collection_semantic_digest,
        target_motion_payload_digest=motion_reader.payload_semantic_digest,
        decision_state_digests=decision_digests,
        config_digest=str(manifest.get("config_digest")),
    )
    if (
        publication_digest != manifest.get("publication_semantic_digest")
        or publication_digest != summary.get("publication_semantic_digest")
    ):
        raise ValueError("SOP05R publication semantic digest differs")
    if (
        manifest.get("trajectory_collection_version")
        != SOP05R_TEB_TRAJECTORY_COLLECTION_VERSION
        or manifest.get("target_motion_payload_digest")
        != motion_reader.payload_semantic_digest
        or summary.get("accepted_count") != len(event_rows)
        or summary.get("requested_count") != manifest.get("requested_count")
        or not bool(summary.get("quota_met"))
    ):
        raise ValueError("SOP05R nested metadata differs")
    marker = _read_json(source_root / "COMPLETE.json", name="SOP05R completion marker")
    if (
        not isinstance(marker, dict)
        or marker.get("completion_marker_version")
        != SOP05R_TEB_COMPLETION_MARKER_VERSION
        or marker.get("publication_semantic_digest") != publication_digest
        or marker.get("accepted_count") != len(event_rows)
        or marker.get("requested_count") != manifest.get("requested_count")
    ):
        raise ValueError("SOP05R completion marker differs")
    return (
        manifest,
        _CompleteResolutionState(
            root=source_root,
            event_rows=event_rows,
            motion_reader=motion_reader,
            trajectory_reader=trajectory_reader,
            payloads={},
        ),
        publication_digest,
    )


def _prepare_complete_boundary(
    state: _CompleteResolutionState,
    *,
    boundary: Sequence[Sop06AcceptedFinalRecord],
    accepted_set: frozenset[Sop06AcceptedFinalRecord],
    grid,
) -> _CompleteResolutionState:
    if any(record not in accepted_set for record in boundary):
        raise ValueError("boundary contains a record not owned by this source")
    event_ids = tuple(dict.fromkeys(record.mother_id for record in boundary))
    motion_pairs = state.motion_reader.load_records_and_worlds(event_ids, grid=grid)
    motion_by_event = {record.generated_event_id: (record, world) for record, world in motion_pairs}
    trajectories = state.trajectory_reader.load_records(event_ids)
    trajectory_by_event = {record.event_id: record for record in trajectories}
    if set(motion_by_event) != set(event_ids) or set(trajectory_by_event) != set(event_ids):
        raise ValueError("prepared complete-mother records differ from boundary")
    payloads: dict[str, _CompleteMotherPayload] = {}
    for accepted in boundary:
        event_row = state.event_rows[accepted.source_index]
        if event_row["event_id"] != accepted.mother_id:
            raise ValueError("final record mother differs from source order")
        motion, world = motion_by_event[accepted.mother_id]
        trajectory = trajectory_by_event[accepted.mother_id]
        if (
            motion.world_id != event_row["world_id"]
            or motion.base_state_id != event_row["decision_state_id"]
            or motion.trajectory_id != trajectory.nominal_trajectory.trajectory_id
            or trajectory.decision_state_id != event_row["decision_state_id"]
        ):
            raise ValueError("complete-mother nested identity join differs")
        state_path = state.root / "decision_states" / str(event_row["decision_state_file"])
        loaded_state = load_dataclass(state_path)
        if not isinstance(loaded_state, BaseState):
            raise ValueError("complete-mother decision state is invalid")
        validate_base_state(loaded_state, grid)
        if (
            loaded_state.state_id != event_row["decision_state_id"]
            or canonical_sop05r_teb_base_state_digest(loaded_state)
            != event_row["decision_state_digest"]
        ):
            raise ValueError("complete-mother decision state digest differs")
        payloads[accepted.mother_id] = _CompleteMotherPayload(
            event_row=event_row,
            motion_record=motion,
            state=loaded_state,
            world=world,
            trajectory=trajectory,
        )
    return replace(state, payloads=payloads)


def _renderer_base_state_without_target(
    state: BaseState,
    target_id: str,
) -> BaseState:
    if target_id not in state.dynamic_object_ids:
        return state
    context_ids = tuple(
        object_id
        for object_id in state.dynamic_object_ids
        if object_id != target_id
    )
    return replace(
        state,
        dynamic_object_ids=context_ids,
        visible_dynamic_object_history={
            object_id: state.visible_dynamic_object_history[object_id]
            for object_id in context_ids
        },
        visible_dynamic_object_specs={
            object_id: state.visible_dynamic_object_specs[object_id]
            for object_id in context_ids
        },
    )


def _partial_view_for_accepted(
    partial_state: _PartialResolutionState,
    accepted: Sop06AcceptedFinalRecord,
):
    partial = partial_state.partial
    if accepted.source_index >= len(partial.target_rows):
        raise ValueError("final source index exceeds partial-M6 source")
    trajectory_row = partial.trajectory_rows[accepted.source_index]
    target_row = partial.target_rows[accepted.source_index]
    if (
        trajectory_row.get("event_id") != accepted.mother_id
        or target_row.get("generated_event_id") != accepted.mother_id
    ):
        raise ValueError("final record mother differs from partial source")
    source_state = partial_state.source_states.get(
        str(trajectory_row["source_base_state_id"])
    )
    if source_state is None:
        raise ValueError("partial-M6 source BaseState is missing")
    snippet_source = partial_state.snippet_sources.get(
        str(target_row["source_snippet_id"])
    )
    if snippet_source is None:
        raise ValueError("partial-M6 snippet source is missing")
    target_root = Path(partial.root) / "target_motion"
    world_path = (target_root / str(target_row["world_file"])).resolve()
    try:
        world_path.relative_to(target_root.resolve())
    except ValueError as exc:
        raise ValueError("partial-M6 world path escapes source root") from exc
    world = load_dataclass(world_path)
    if not isinstance(world, OracleWorld):
        raise ValueError("partial-M6 world payload is invalid")
    return build_partial_mother_view(
        trajectory_row=trajectory_row,
        target_row=target_row,
        target_history=partial.history_poses[accepted.source_index],
        target_current=partial.current_poses[accepted.source_index],
        target_future=partial.future_poses[accepted.source_index],
        world=world,
        source_state=source_state,
        source_recording_id=snippet_source[0],
        source_session_id=snippet_source[1],
        centerline_epsilon_m=partial_state.centerline_epsilon_m,
    )


@dataclass(frozen=True)
class Sop06FinalizedSource:
    source_mode: str
    source_publication_semantic_digest: str
    final_release_identity: str
    base_config: Mapping[str, object]
    finalized: LoadedSop05FinalScenarios
    accepted: tuple[Sop06AcceptedFinalRecord, ...]
    _accepted_set: frozenset[Sop06AcceptedFinalRecord]
    _complete_state: _CompleteResolutionState | None
    _partial_state: _PartialResolutionState | None

    def prepare_boundary(
        self,
        boundary: tuple[Sop06AcceptedFinalRecord, ...],
    ) -> Sop06FinalizedSource:
        """Materialize only source payloads needed by one output shard."""

        if self.source_mode == "complete_mother":
            complete_state = self._complete_state
            if complete_state is None:
                raise RuntimeError("complete mother resolution state is unavailable")
            prepared = _prepare_complete_boundary(
                complete_state,
                boundary=boundary,
                accepted_set=self._accepted_set,
                grid=build_grid_spec(dict(self.base_config)),
            )
            return replace(self, _complete_state=prepared)
        partial_state = self._partial_state
        if partial_state is None:
            raise RuntimeError("partial-M6 resolution state is unavailable")
        if any(record not in self._accepted_set for record in boundary):
            raise ValueError("boundary contains a record not owned by this source")
        event_ids = tuple(record.mother_id for record in boundary)
        records = partial_state.trajectory_reader.load_records(event_ids)
        if tuple(record.event_id for record in records) != event_ids:
            raise ValueError("prepared partial-M6 trajectories differ from boundary")
        trajectories = {record.event_id: record for record in records}
        if len(trajectories) != len(records):
            raise ValueError("partial-M6 boundary trajectory IDs are not unique")
        return replace(
            self,
            _partial_state=replace(
                partial_state,
                trajectories=trajectories,
            ),
        )

    def prepare_history_boundary(
        self,
        boundary: tuple[Sop06AcceptedFinalRecord, ...],
    ) -> Sop06FinalizedSource:
        if self.source_mode == "complete_mother":
            return self.prepare_boundary(boundary)
        if any(record not in self._accepted_set for record in boundary):
            raise ValueError("boundary contains a record not owned by this source")
        return self

    def resolve_history_renderer_input(
        self,
        accepted: Sop06AcceptedFinalRecord,
    ) -> Sop06SingleRendererInput:
        if accepted not in self._accepted_set:
            raise ValueError("accepted final record is not owned by this source")
        if self.source_mode == "complete_mother":
            return self.resolve(accepted).publication.renderer_input
        partial_state = self._partial_state
        if partial_state is None:
            raise RuntimeError("partial-M6 resolution state is unavailable")
        view = _partial_view_for_accepted(partial_state, accepted)
        event = view.event
        state = view.state
        if state.split != accepted.split:
            raise ValueError("SOP06 source decision split differs")
        target_id = event.target.target_dynamic_object_id
        renderer_state = _renderer_base_state_without_target(state, target_id)
        histories = {
            object_id: np.array(history, dtype=np.float32, order="C", copy=True)
            for object_id, history in (
                renderer_state.visible_dynamic_object_history.items()
            )
        }
        base_input = Sop06SingleRendererInput(
            sample_id=accepted.scenario_id,
            mother_id=accepted.mother_id,
            split=accepted.split,
            base_state=renderer_state,
            observed_static_occupancy=np.array(
                event.world.static_occupancy,
                dtype=np.float32,
                order="C",
                copy=True,
            ),
            scene_dynamic_history=histories,
            scene_dynamic_specs={
                object_id: dict(spec)
                for object_id, spec in (
                    renderer_state.visible_dynamic_object_specs.items()
                )
            },
            scene_dynamic_history_observed={
                object_id: np.ones(8, dtype=np.bool_) for object_id in histories
            },
            sensor_config=None,
        )
        return adapt_finalized_sop05_renderer_input(
            renderer_input=base_input,
            regime=accepted.regime,
            target_present=accepted.target_present,
            target_dynamic_object_id=target_id,
            target_footprint_spec=event.target.footprint_spec,
            target_history_observed=np.asarray(
                event.target_visibility_history,
                dtype=np.bool_,
            ),
            history_poses=self.finalized.history_poses[accepted.target_row],
        )

    def resolve(
        self,
        accepted: Sop06AcceptedFinalRecord,
    ) -> ResolvedSop06Scenario:
        if accepted not in self._accepted_set:
            raise ValueError("accepted final record is not owned by this source")
        complete_lineage: _CompleteMotherLineage | None = None
        source_base_state_id: str | None = None
        if self.source_mode == "complete_mother":
            complete_state = self._complete_state
            if complete_state is None:
                raise RuntimeError("complete mother resolution state is unavailable")
            payload = complete_state.payloads.get(accepted.mother_id)
            if payload is None:
                complete_state = _prepare_complete_boundary(
                    complete_state,
                    boundary=(accepted,),
                    accepted_set=self._accepted_set,
                    grid=build_grid_spec(dict(self.base_config)),
                )
                payload = complete_state.payloads.get(accepted.mother_id)
            if payload is None:
                raise RuntimeError("complete mother payload is unavailable")
            state = payload.state
            trajectory_record = payload.trajectory
            world = payload.world
            target_id = payload.motion_record.target_dynamic_object_id
            target_footprint_spec = dict(payload.motion_record.footprint_spec)
            target_provenance = dict(payload.event_row["target_provenance"])
            target_history_observed = np.asarray(
                payload.event_row["target_visibility_history"], dtype=np.bool_
            )
            source_object_id = payload.motion_record.source_object_id
            source_snippet_id = payload.motion_record.source_snippet_id
            complete_lineage = complete_state.lineage
            source_base_state_id = _nonempty_string(
                payload.event_row.get("source_base_state_id"),
                name="complete mother source_base_state_id",
            )
        else:
            partial_state = self._partial_state
            if partial_state is None:
                raise RuntimeError("partial-M6 resolution state is unavailable")
            view = _partial_view_for_accepted(partial_state, accepted)
            event = view.event
            state = view.state
            world = event.world
            target_id = event.target.target_dynamic_object_id
            target_footprint_spec = dict(event.target.footprint_spec)
            target_provenance = dict(event.target.provenance)
            target_history_observed = np.asarray(
                event.target_visibility_history, dtype=np.bool_
            )
            source_object_id = event.target.source_object_id
            source_snippet_id = event.target.snippet_id
            trajectory_record = partial_state.trajectories.get(
                accepted.mother_id
            )
            if trajectory_record is None:
                loaded = partial_state.trajectory_reader.load_records(
                    (accepted.mother_id,)
                )
                if (
                    len(loaded) != 1
                    or loaded[0].event_id != accepted.mother_id
                ):
                    raise ValueError("partial-M6 mother trajectory is missing")
                trajectory_record = loaded[0]
        if state.split != accepted.split:
            raise ValueError("SOP06 source decision split differs")

        source_recording_id = target_provenance.get("source_recording_id")
        source_session_id = target_provenance.get("source_session_id")
        base_session_id = state.metadata.get("session_id")
        if complete_lineage is not None:
            if source_base_state_id is None:
                raise RuntimeError("complete mother source BaseState ID is unavailable")
            base_source = complete_lineage.base_sources.get(source_base_state_id)
            snippet_source = complete_lineage.snippet_sources.get(source_snippet_id)
            if base_source is None or snippet_source is None:
                raise ValueError("complete mother recovered lineage is missing")
            recovered_base_recording_id, recovered_base_session_id = base_source
            (
                recovered_source_recording_id,
                recovered_source_session_id,
                recovered_source_object_id,
            ) = snippet_source
            if recovered_base_recording_id != state.recording_id:
                raise ValueError(
                    "complete mother recovered base recording differs from decision state"
                )
            if recovered_source_object_id != source_object_id:
                raise ValueError(
                    "complete mother recovered source object differs from target motion"
                )
            for value, expected, label in (
                (base_session_id, recovered_base_session_id, "base session"),
                (
                    source_recording_id,
                    recovered_source_recording_id,
                    "source recording",
                ),
                (source_session_id, recovered_source_session_id, "source session"),
            ):
                if isinstance(value, str) and value.strip() and value != expected:
                    raise ValueError(
                        f"complete mother persisted {label} differs from recovered lineage"
                    )
            base_session_id = recovered_base_session_id
            source_recording_id = recovered_source_recording_id
            source_session_id = recovered_source_session_id
        if (
            (not isinstance(base_session_id, str) or not base_session_id)
            and isinstance(source_recording_id, str)
            and source_recording_id == state.recording_id
            and isinstance(source_session_id, str)
        ):
            base_session_id = source_session_id
        renderer_state = _renderer_base_state_without_target(state, target_id)
        histories = {
            object_id: np.array(history, dtype=np.float32, order="C", copy=True)
            for object_id, history in (
                renderer_state.visible_dynamic_object_history.items()
            )
        }
        specs = {
            object_id: dict(spec)
            for object_id, spec in renderer_state.visible_dynamic_object_specs.items()
        }
        target_currently_observed = (
            accepted.target_present
            and accepted.regime == "seen_then_occluded"
            and bool(target_history_observed[-1])
        )
        hidden_ids = (
            (target_id,)
            if accepted.target_present and not target_currently_observed
            else ()
        )
        oracle_metadata = dict(world.metadata)
        declared_schema = oracle_metadata.get("schema_version")
        if declared_schema not in (None, SCHEMA_VERSION):
            raise ValueError("SOP06 oracle world schema_version differs")
        oracle_world = replace(
            world,
            metadata={**oracle_metadata, "schema_version": SCHEMA_VERSION},
        )
        context = Sop06SinglePublicationContext(
            sample_id=accepted.scenario_id,
            mother_id=accepted.mother_id,
            split=accepted.split,
            base_state=renderer_state,
            trajectory=trajectory_record.nominal_trajectory,
            oracle_world=oracle_world,
            observed_static_occupancy=np.array(
                world.static_occupancy,
                dtype=np.float32,
                order="C",
                copy=True,
            ),
            scene_dynamic_history=histories,
            scene_dynamic_specs=specs,
            hidden_object_ids=hidden_ids,
            sensor_config=None,
            target_dynamic_object_id=target_id,
            target_footprint_spec=target_footprint_spec,
            target_history_observed=np.array(
                target_history_observed,
                dtype=np.bool_,
                order="C",
                copy=True,
            ),
            provenance={
                "source_kind": "persisted_final_scenario",
                "source_digest": self.source_publication_semantic_digest,
                "base_recording_id": state.recording_id,
                **(
                    {"base_session_id": base_session_id}
                    if isinstance(base_session_id, str) and base_session_id
                    else {}
                ),
                **(
                    {"source_recording_id": source_recording_id}
                    if isinstance(source_recording_id, str) and source_recording_id
                    else {}
                ),
                **(
                    {"source_session_id": source_session_id}
                    if isinstance(source_session_id, str) and source_session_id
                    else {}
                ),
                "source_object_id": source_object_id,
                "source_snippet_id": source_snippet_id,
                "seed_namespace": (
                    f"sop07/{accepted.split}/sop06-single/"
                    f"{accepted.scenario_id}"
                ),
                "base_config_digest": config_digest(self.base_config),
                "target_present": accepted.target_present,
                "target_currently_observed": target_currently_observed,
            },
        )
        row = accepted.target_row
        publication = adapt_finalized_sop05_scenario(
            context=context,
            regime=accepted.regime,
            target_present=accepted.target_present,
            history_poses=self.finalized.history_poses[row],
            future_poses=self.finalized.future_poses[row],
        )
        return ResolvedSop06Scenario(
            accepted=accepted,
            publication=publication,
        )


def _accepted_records(
    finalized: LoadedSop05FinalScenarios,
    *,
    split: str,
) -> tuple[Sop06AcceptedFinalRecord, ...]:
    result: list[Sop06AcceptedFinalRecord] = []
    scenario_ids: set[str] = set()
    for record in finalized.records:
        if record.get("split") != split:
            raise ValueError("final scenario split differs from requested split")
        if record.get("status") != "accepted":
            continue
        scenario_id = str(record["scenario_id"])
        if scenario_id in scenario_ids:
            raise ValueError("final scenario IDs must be unique")
        scenario_ids.add(scenario_id)
        result.append(
            Sop06AcceptedFinalRecord(
                source_index=int(record["source_index"]),
                mother_id=str(record["mother_id"]),
                scenario_id=scenario_id,
                split=split,
                regime=str(record["regime"]),
                target_present=bool(record["target_present"]),
                target_row=int(record["target_row"]),
            )
        )
    if len(result) != finalized.accepted_count:
        raise ValueError("accepted final scenario count differs")
    return tuple(result)


def _validate_partial_accepted_joins(
    *,
    partial: object,
    source_states: Mapping[str, object],
    snippet_sources: Mapping[str, tuple[str, str]],
    trajectory_event_ids: frozenset[str],
    accepted: tuple[Sop06AcceptedFinalRecord, ...],
) -> None:
    for record in accepted:
        if not 0 <= record.source_index < len(partial.target_rows):
            raise ValueError("final source index exceeds partial-M6 source")
        trajectory_row = partial.trajectory_rows[record.source_index]
        target_row = partial.target_rows[record.source_index]
        if (
            trajectory_row.get("event_id") != record.mother_id
            or target_row.get("generated_event_id") != record.mother_id
        ):
            raise ValueError("final record mother differs from partial source")
        source_state = source_states.get(
            str(trajectory_row["source_base_state_id"])
        )
        if source_state is None:
            raise ValueError("partial-M6 source BaseState is missing")
        if source_state.split != record.split:
            raise ValueError("SOP06 source decision split differs")
        if str(target_row["source_snippet_id"]) not in snippet_sources:
            raise ValueError("partial-M6 snippet source is missing")
        if record.mother_id not in trajectory_event_ids:
            raise ValueError("partial-M6 mother trajectory is missing")


def compute_sop05_final_release_identity(root: str | Path) -> str:
    """Bind the strict six-file release after its loader has validated it."""

    path = Path(root)
    digest = hashlib.sha256()
    digest.update(b"sop05_final_release_identity_v1\0")
    for name in ("checksums.json", "COMPLETE.json"):
        digest.update(name.encode("ascii") + b"\0")
        try:
            digest.update((path / name).read_bytes())
        except OSError as exc:
            raise ValueError(f"failed to read final release identity file: {name}") from exc
        digest.update(b"\0")
    return digest.hexdigest()


def load_sop06_finalized_source(
    *,
    source_mode: str,
    source_root: str | Path,
    final_scenario_root: str | Path,
    split: str,
    sop03_root: str | Path | None = None,
    long40_human_artifact: str | Path | None = None,
    base_state_start: int | None = None,
    max_base_states: int | None = None,
    base_config: Mapping[str, object] | None = None,
    source_config_digest: str | None = None,
    centerline_epsilon_m: float | None = None,
) -> Sop06FinalizedSource:
    if source_mode not in _SOURCE_MODES:
        raise ValueError("source_mode is invalid")
    if split not in _SPLITS:
        raise ValueError("split is invalid")
    if source_mode == "complete_mother":
        if (sop03_root is None) != (long40_human_artifact is None):
            raise ValueError(
                "complete-mother lineage requires both SOP3 and Long40 roots"
            )
        if any(
            value is not None
            for value in (
                base_state_start,
                max_base_states,
                base_config,
                source_config_digest,
                centerline_epsilon_m,
            )
        ):
            raise ValueError(
                "complete-mother lineage derives source bounds from source evidence"
            )
        finalized = load_sop05_final_scenarios(
            final_scenario_root,
        )
        accepted = _accepted_records(finalized, split=split)
        manifest, complete_state, publication_digest = _load_complete_resolution_state(
            Path(source_root),
            accepted=accepted,
        )
        if (
            finalized.manifest.get("source_publication_semantic_digest")
            != publication_digest
        ):
            raise ValueError("final scenario source digest differs from mother")
        if sop03_root is not None:
            if long40_human_artifact is None:
                raise RuntimeError("complete-mother Long40 artifact is unavailable")
            complete_state = replace(
                complete_state,
                lineage=_complete_mother_lineage(
                    manifest=manifest,
                    state=complete_state,
                    accepted=accepted,
                    split=split,
                    sop03_root=sop03_root,
                    long40_human_artifact=long40_human_artifact,
                ),
            )
        return Sop06FinalizedSource(
            source_mode=source_mode,
            source_publication_semantic_digest=publication_digest,
            final_release_identity=compute_sop05_final_release_identity(
                final_scenario_root
            ),
            base_config=dict(manifest["base_config"]),
            finalized=finalized,
            accepted=accepted,
            _accepted_set=frozenset(accepted),
            _complete_state=complete_state,
            _partial_state=None,
        )

    if (
        sop03_root is None
        or long40_human_artifact is None
        or base_state_start is None
        or max_base_states is None
        or base_config is None
        or source_config_digest is None
        or centerline_epsilon_m is None
    ):
        raise ValueError("partial-M6 source arguments are required")
    config = dict(base_config)
    partial = load_partial_m6_source(
        source_root,
        source_config_digest=source_config_digest,
    )
    finalized = load_sop05_final_scenarios(
        final_scenario_root,
        expected_source_publication_semantic_digest=partial.source_identity,
    )
    accepted = _accepted_records(finalized, split=split)
    selected_trajectory_rows: list[Mapping[str, object]] = []
    required_state_ids: set[str] = set()
    for record in accepted:
        if not (
            0 <= record.source_index < len(partial.trajectory_rows)
            and record.source_index < len(partial.target_rows)
        ):
            raise ValueError("final source index exceeds partial-M6 source")
        trajectory_row = partial.trajectory_rows[record.source_index]
        target_row = partial.target_rows[record.source_index]
        if (
            trajectory_row.get("event_id") != record.mother_id
            or target_row.get("generated_event_id") != record.mother_id
        ):
            raise ValueError("final record mother differs from partial source")
        selected_trajectory_rows.append(trajectory_row)
        required_state_ids.add(str(trajectory_row["source_base_state_id"]))
    inputs = load_sop05r_teb_long40_inputs(
        recording_root=sop03_root,
        long40_human_artifact=long40_human_artifact,
        split=split,
        grid=build_grid_spec(config),
        max_base_states=max_base_states,
        base_state_start=base_state_start,
        required_state_ids=frozenset(required_state_ids),
    )
    source_states = {
        state.state_id: state for state, _ in inputs.state_pairs
    }
    if len(source_states) != len(inputs.state_pairs):
        raise ValueError("partial-M6 source BaseState IDs are not unique")
    snippet_sources = {
        snippet.snippet_id: (
            snippet.source_recording_id,
            snippet.source_session_id,
        )
        for snippet in inputs.snippets
    }
    if len(snippet_sources) != len(inputs.snippets):
        raise ValueError("partial-M6 snippet IDs are not unique")
    trajectory_event_ids = frozenset(
        str(row["event_id"]) for row in selected_trajectory_rows
    )
    if len(trajectory_event_ids) != len(selected_trajectory_rows):
        raise ValueError("partial-M6 selected trajectory event IDs are not unique")
    _validate_partial_accepted_joins(
        partial=partial,
        source_states=source_states,
        snippet_sources=snippet_sources,
        trajectory_event_ids=trajectory_event_ids,
        accepted=accepted,
    )
    trajectory_reader = open_sop05r_teb_trajectory_selection(
        Path(source_root) / "trajectory_store",
        rows=tuple(selected_trajectory_rows),
    )
    return Sop06FinalizedSource(
        source_mode=source_mode,
        source_publication_semantic_digest=partial.source_identity,
        final_release_identity=compute_sop05_final_release_identity(
            final_scenario_root
        ),
        base_config=config,
        finalized=finalized,
        accepted=accepted,
        _accepted_set=frozenset(accepted),
        _complete_state=None,
        _partial_state=_PartialResolutionState(
            partial=partial,
            source_states=source_states,
            snippet_sources=snippet_sources,
            trajectory_reader=trajectory_reader,
            trajectories={},
            centerline_epsilon_m=float(centerline_epsilon_m),
        ),
    )


__all__ = (
    "ResolvedSop06Scenario",
    "Sop06AcceptedFinalRecord",
    "Sop06FinalizedSource",
    "compute_sop05_final_release_identity",
    "load_sop06_finalized_source",
)
