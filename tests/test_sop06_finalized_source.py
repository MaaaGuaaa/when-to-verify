"""Persisted SOP05 final-release to SOP06 source-boundary tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

import src.generation.sop05_final_scenarios as final_scenarios
import src.generation.sop05_partial_m6_final as partial_final
import src.generation.sop06_finalized_source as finalized_source_module
from src.generation.sop05_final_scenarios import publish_sop05_final_scenarios
from src.generation.sop05_partial_m6_final import (
    publish_partial_m6_final_scenarios,
)
from src.generation.sop05_seen_prior import load_seen_prior_config
from src.generation.sop05_unseen_prior import normalize_unseen_prior_config
from src.generation.sop06_finalized_source import load_sop06_finalized_source
from src.generation.sop06_single import (
    render_sop06_single_input,
    render_sop06_single_publication,
)
from tests.test_anchored_human_placement import _m4_inputs, _snippet
from tests.test_sop05r_teb_sop06_handoff import _strict_collection


ROOT = Path(__file__).resolve().parents[1]


def _complete_lineage_evidence() -> dict[str, object]:
    return {
        "input_kind": "sop03_schema4_long40_runtime_v1",
        "recording_index_manifest_sha256": "a" * 64,
        "recording_index_summary_sha256": "b" * 64,
        "recording_count": 1,
        "base_state_start": 0,
        "long40_checksum_manifest_sha256": "c" * 64,
        "long40_semantic_digest": "d" * 64,
        "accepted_source_snippet_count": 1,
        "materialized_base_state_count": 1,
    }


def _publish_complete_final(
    collection,
    output: Path,
    monkeypatch,
    *,
    record_mother_id: str | None = None,
) -> str:
    event = collection.events[0]
    mother_id = record_mother_id or event.generated_event_id
    history = np.asarray(event.target.history_poses, dtype=np.float32)
    future = np.asarray(event.target.future_poses, dtype=np.float32)
    regime = (
        "seen_then_occluded"
        if bool(np.asarray(event.target_visibility_history, dtype=np.bool_)[0])
        else "unseen_in_history_window"
    )
    scenario_id = "sop05-final-persisted-fixture"
    monkeypatch.setattr(
        final_scenarios,
        "_process_all",
        lambda *args, **kwargs: [
            final_scenarios._WorkResult(
                record={
                    "source_index": 0,
                    "mother_id": mother_id,
                    "split": "train",
                    "regime": regime,
                    "status": "accepted",
                    "scenario_id": scenario_id,
                    "target_present": True,
                },
                provenance={
                    "source_index": 0,
                    "mother_id": mother_id,
                    "status": "accepted",
                },
                history_poses=history,
                future_poses=future,
                target_present=True,
            )
        ],
    )
    base_config = dict(collection.manifest["base_config"])
    unseen = normalize_unseen_prior_config(
        yaml.safe_load((ROOT / "configs/sop05_unseen_prior.yaml").read_text()),
        base_config=base_config,
    )
    publish_sop05_final_scenarios(
        collection,
        output_dir=output,
        unseen_config=unseen,
        seen_config=load_seen_prior_config(ROOT / "configs/sop05_seen_prior.yaml"),
        expected_source_config_digest=str(collection.manifest["config_digest"]),
    )
    return scenario_id


def test_complete_source_joins_final_record_by_mother_and_scenario_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mother_root = tmp_path / "m7"
    collection = _strict_collection(tmp_path)
    final_root = tmp_path / "final"
    scenario_id = _publish_complete_final(collection, final_root, monkeypatch)

    source = load_sop06_finalized_source(
        source_mode="complete_mother",
        source_root=mother_root,
        final_scenario_root=final_root,
        split="train",
    )
    assert source.source_publication_semantic_digest == (
        collection.publication_semantic_digest
    )
    assert len(source.final_release_identity) == 64
    assert len(source.accepted) == 1
    accepted = source.accepted[0]
    resolved = source.resolve(accepted)

    assert accepted.scenario_id == scenario_id
    assert resolved.publication.sample_id == scenario_id
    assert resolved.publication.mother_id == collection.events[0].generated_event_id
    assert resolved.publication.split == "train"
    rendered = render_sop06_single_publication(
        resolved.publication,
        config=source.base_config,
    )
    assert rendered.bev_history.shape == (8, 2, 160, 160)
    assert rendered.state_channels.shape == (9, 160, 160)


def test_complete_source_recovers_legacy_identity_from_authenticated_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evidence = _complete_lineage_evidence()
    collection = _strict_collection(tmp_path, source_evidence=evidence)
    mother_root = tmp_path / "m7"
    final_root = tmp_path / "final"
    _publish_complete_final(collection, final_root, monkeypatch)
    event = collection.events[0]
    decision_state = collection.decision_states[
        collection.trajectories.records[0].decision_state_id
    ]
    source_state = replace(
        decision_state,
        state_id=str(event.world.metadata["source_base_state_id"]),
        metadata={**decision_state.metadata, "session_id": "base-session"},
    )
    motion = event.target_motion_record
    source_snippet = SimpleNamespace(
        snippet_id=motion.source_snippet_id,
        source_recording_id="source-recording",
        source_session_id="source-session",
        source_object_id=motion.source_object_id,
        object_type=motion.object_type,
    )
    monkeypatch.setattr(
        finalized_source_module,
        "load_sop05r_teb_long40_inputs",
        lambda **kwargs: SimpleNamespace(
            state_pairs=((source_state, object()),),
            snippets=(source_snippet,),
            source_evidence={
                **evidence,
                "selected_base_state_count": 1,
            },
        ),
    )

    source = load_sop06_finalized_source(
        source_mode="complete_mother",
        source_root=mother_root,
        final_scenario_root=final_root,
        split="train",
        sop03_root=tmp_path / "sop03",
        long40_human_artifact=tmp_path / "long40",
    )
    resolved = source.resolve(source.accepted[0])
    provenance = resolved.publication.provenance

    assert provenance["base_session_id"] == "base-session"
    assert provenance["source_recording_id"] == "source-recording"
    assert provenance["source_session_id"] == "source-session"


def test_complete_source_uses_selected_loader_not_full_collection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    collection = _strict_collection(tmp_path)
    final_root = tmp_path / "final"
    _publish_complete_final(collection, final_root, monkeypatch)
    monkeypatch.setattr(
        finalized_source_module,
        "load_sop05r_teb_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("complete source eagerly loaded the full SOP5 collection")
        ),
        raising=False,
    )

    source = load_sop06_finalized_source(
        source_mode="complete_mother",
        source_root=tmp_path / "m7",
        final_scenario_root=final_root,
        split="train",
    )

    resolved = source.resolve(source.accepted[0])

    assert resolved.publication.sample_id == source.accepted[0].scenario_id


def test_complete_source_defers_decision_state_loading_until_resolution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    collection = _strict_collection(tmp_path)
    final_root = tmp_path / "final"
    _publish_complete_final(collection, final_root, monkeypatch)
    calls: list[Path] = []
    original_load = finalized_source_module.load_dataclass

    def tracked_load(path):
        calls.append(Path(path))
        return original_load(path)

    monkeypatch.setattr(finalized_source_module, "load_dataclass", tracked_load)
    source = load_sop06_finalized_source(
        source_mode="complete_mother",
        source_root=tmp_path / "m7",
        final_scenario_root=final_root,
        split="train",
    )

    assert calls == []
    source.resolve(source.accepted[0])

    decision_state_id = collection.trajectories.records[0].decision_state_id
    assert calls == [tmp_path / "m7" / "decision_states" / f"{decision_state_id}.npz"]


def test_complete_source_resolve_does_not_scan_all_accepted_records(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class NoContainsTuple(tuple):
        def __contains__(self, value: object) -> bool:
            raise AssertionError("resolve scanned the accepted-record tuple")

    mother_root = tmp_path / "m7"
    collection = _strict_collection(tmp_path)
    final_root = tmp_path / "final"
    _publish_complete_final(collection, final_root, monkeypatch)
    source = load_sop06_finalized_source(
        source_mode="complete_mother",
        source_root=mother_root,
        final_scenario_root=final_root,
        split="train",
    )
    object.__setattr__(source, "accepted", NoContainsTuple(source.accepted))

    resolved = source.resolve(source.accepted[0])

    assert resolved.accepted.scenario_id == source.accepted[0].scenario_id


def test_complete_source_removes_target_from_renderer_base_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mother_root = tmp_path / "m7"
    collection = _strict_collection(tmp_path)
    event = collection.events[0]
    target_id = event.target.target_dynamic_object_id
    trajectory = collection.trajectories.records[0]
    state = collection.decision_states[trajectory.decision_state_id]
    state_with_target = replace(
        state,
        dynamic_object_ids=(
            *(
                object_id
                for object_id in state.dynamic_object_ids
                if object_id != target_id
            ),
            target_id,
        ),
        visible_dynamic_object_history={
            **state.visible_dynamic_object_history,
            target_id: np.asarray(event.target.history_poses, dtype=np.float32),
        },
        visible_dynamic_object_specs={
            **state.visible_dynamic_object_specs,
            target_id: dict(event.target.footprint_spec),
        },
    )
    final_root = tmp_path / "final"
    _publish_complete_final(collection, final_root, monkeypatch)

    source = load_sop06_finalized_source(
        source_mode="complete_mother",
        source_root=mother_root,
        final_scenario_root=final_root,
        split="train",
    )
    original_prepare = finalized_source_module._prepare_complete_boundary

    def prepare_with_target(*args, **kwargs):
        prepared = original_prepare(*args, **kwargs)
        mother_id = source.accepted[0].mother_id
        payload = prepared.payloads[mother_id]
        return replace(
            prepared,
            payloads={
                **prepared.payloads,
                mother_id: replace(payload, state=state_with_target),
            },
        )

    monkeypatch.setattr(
        finalized_source_module,
        "_prepare_complete_boundary",
        prepare_with_target,
    )
    resolved = source.resolve(source.accepted[0])

    assert target_id not in resolved.publication.renderer_input.base_state.dynamic_object_ids
    rendered = render_sop06_single_publication(
        resolved.publication,
        config=source.base_config,
    )
    assert rendered.bev_history.shape == (8, 2, 160, 160)


def test_complete_source_rejects_final_mother_join_during_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mother_root = tmp_path / "m7"
    collection = _strict_collection(tmp_path)
    final_root = tmp_path / "final"
    _publish_complete_final(
        collection,
        final_root,
        monkeypatch,
        record_mother_id="missing-mother",
    )

    try:
        load_sop06_finalized_source(
            source_mode="complete_mother",
            source_root=mother_root,
            final_scenario_root=final_root,
            split="train",
        )
    except ValueError as exc:
        assert "mother" in str(exc)
    else:
        raise AssertionError("source loader accepted a mismatched final mother")


def test_partial_source_rejects_final_mother_join_during_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_config, source_state, oracle_context, teb_config = _m4_inputs()
    snippet = _snippet()
    mother_id = "partial-mother"
    finalized = SimpleNamespace(
        records=(
            {
                "source_index": 0,
                "mother_id": "missing-mother",
                "scenario_id": "partial-scenario",
                "split": "train",
                "regime": "unseen_in_history_window",
                "status": "accepted",
                "target_present": True,
                "target_row": 0,
            },
        ),
        accepted_count=1,
    )
    partial = SimpleNamespace(
        source_identity="a" * 64,
        trajectory_rows=(
            {
                "event_id": mother_id,
                "source_base_state_id": source_state.state_id,
            },
        ),
        target_rows=(
            {
                "generated_event_id": mother_id,
                "source_snippet_id": snippet.snippet_id,
            },
        ),
    )
    monkeypatch.setattr(
        finalized_source_module,
        "load_partial_m6_source",
        lambda *args, **kwargs: partial,
    )
    monkeypatch.setattr(
        finalized_source_module,
        "load_sop05_final_scenarios",
        lambda *args, **kwargs: finalized,
    )
    monkeypatch.setattr(
        finalized_source_module,
        "load_sop05r_teb_long40_inputs",
        lambda **kwargs: SimpleNamespace(
            state_pairs=((source_state, oracle_context),),
            snippets=(snippet,),
        ),
    )
    monkeypatch.setattr(
        finalized_source_module,
        "compute_sop05_final_release_identity",
        lambda *args, **kwargs: "b" * 64,
    )

    try:
        load_sop06_finalized_source(
            source_mode="partial_m6_reconstruction",
            source_root=tmp_path / "partial",
            final_scenario_root=tmp_path / "final",
            split="train",
            sop03_root=tmp_path / "sop03",
            long40_human_artifact=tmp_path / "long40",
            base_state_start=0,
            max_base_states=1,
            base_config=base_config,
            source_config_digest=teb_config.digest,
            centerline_epsilon_m=(
                teb_config.occlusion.centerline_intersection_epsilon_m
            ),
        )
    except ValueError as exc:
        assert "mother" in str(exc)
    else:
        raise AssertionError("source loader accepted a mismatched partial mother")


def test_partial_source_defers_trajectory_arrays_until_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_config, source_state, oracle_context, teb_config = _m4_inputs()
    snippet = _snippet()
    mother_id = "partial-mother"
    finalized = SimpleNamespace(
        records=(
            {
                "source_index": 0,
                "mother_id": mother_id,
                "scenario_id": "partial-scenario",
                "split": "train",
                "regime": "unseen_in_history_window",
                "status": "accepted",
                "target_present": True,
                "target_row": 0,
            },
        ),
        accepted_count=1,
    )
    partial = SimpleNamespace(
        source_identity="a" * 64,
        trajectory_rows=(
            {
                "event_id": mother_id,
                "source_base_state_id": source_state.state_id,
            },
        ),
        target_rows=(
            {
                "generated_event_id": mother_id,
                "source_snippet_id": snippet.snippet_id,
            },
        ),
    )
    loaded_boundaries: list[tuple[str, ...]] = []

    class Reader:
        def load_records(self, event_ids):
            event_ids = tuple(event_ids)
            loaded_boundaries.append(event_ids)
            return tuple(
                SimpleNamespace(event_id=event_id)
                for event_id in event_ids
            )

        def close(self) -> None:
            return None

    assert hasattr(
        finalized_source_module,
        "open_sop05r_teb_trajectory_selection",
    )
    assert hasattr(finalized_source_module.Sop06FinalizedSource, "prepare_boundary")

    monkeypatch.setattr(
        finalized_source_module,
        "load_partial_m6_source",
        lambda *args, **kwargs: partial,
    )
    monkeypatch.setattr(
        finalized_source_module,
        "load_sop05_final_scenarios",
        lambda *args, **kwargs: finalized,
    )
    monkeypatch.setattr(
        finalized_source_module,
        "load_sop05r_teb_long40_inputs",
        lambda **kwargs: SimpleNamespace(
            state_pairs=((source_state, oracle_context),),
            snippets=(snippet,),
        ),
    )
    monkeypatch.setattr(
        finalized_source_module,
        "open_sop05r_teb_trajectory_selection",
        lambda *args, **kwargs: Reader(),
    )
    assert "load_sop05r_teb_trajectory_store" not in vars(
        finalized_source_module
    )
    monkeypatch.setattr(
        finalized_source_module,
        "compute_sop05_final_release_identity",
        lambda *args, **kwargs: "b" * 64,
    )

    source = load_sop06_finalized_source(
        source_mode="partial_m6_reconstruction",
        source_root=tmp_path / "partial",
        final_scenario_root=tmp_path / "final",
        split="train",
        sop03_root=tmp_path / "sop03",
        long40_human_artifact=tmp_path / "long40",
        base_state_start=0,
        max_base_states=1,
        base_config=base_config,
        source_config_digest=teb_config.digest,
        centerline_epsilon_m=(
            teb_config.occlusion.centerline_intersection_epsilon_m
        ),
    )

    assert loaded_boundaries == []
    source.prepare_history_boundary(source.accepted)
    assert loaded_boundaries == []
    prepared = source.prepare_boundary(source.accepted)
    assert loaded_boundaries == [(mother_id,)]
    assert prepared._partial_state is not None
    assert tuple(prepared._partial_state.trajectories) == (mother_id,)


def test_partial_source_recomputes_final_release_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mother_root = tmp_path / "m7"
    collection = _strict_collection(tmp_path)
    event = collection.events[0]
    base_config, source_state, oracle_context, teb_config = _m4_inputs()
    snippet = _snippet()
    scenario_id = "sop05-final-partial-fixture"
    monkeypatch.setattr(
        partial_final,
        "_process_partial_all",
        lambda *args, **kwargs: [
            final_scenarios._WorkResult(
                record={
                    "source_index": 0,
                    "mother_id": event.generated_event_id,
                    "split": "train",
                    "regime": "unseen_in_history_window",
                    "status": "accepted",
                    "scenario_id": scenario_id,
                    "target_present": True,
                },
                provenance={
                    "source_index": 0,
                    "mother_id": event.generated_event_id,
                    "status": "accepted",
                },
                history_poses=np.asarray(
                    event.target.history_poses, dtype=np.float32
                ),
                future_poses=np.asarray(
                    event.target.future_poses, dtype=np.float32
                ),
                target_present=True,
            )
        ],
    )
    final_root = tmp_path / "partial-final"
    published = publish_partial_m6_final_scenarios(
        mother_root,
        source_states={source_state.state_id: source_state},
        snippet_sources={
            snippet.snippet_id: (
                snippet.source_recording_id,
                snippet.source_session_id,
            )
        },
        base_config=base_config,
        source_config_digest=teb_config.digest,
        centerline_epsilon_m=(
            teb_config.occlusion.centerline_intersection_epsilon_m
        ),
        output_dir=final_root,
        unseen_config=normalize_unseen_prior_config(
            yaml.safe_load(
                (ROOT / "configs/sop05_unseen_prior.yaml").read_text()
            ),
            base_config=base_config,
        ),
        seen_config=load_seen_prior_config(
            ROOT / "configs/sop05_seen_prior.yaml"
        ),
    )
    monkeypatch.setattr(
        finalized_source_module,
        "load_sop05r_teb_long40_inputs",
        lambda **kwargs: SimpleNamespace(
            state_pairs=((source_state, oracle_context),),
            snippets=(snippet,),
        ),
    )

    source = load_sop06_finalized_source(
        source_mode="partial_m6_reconstruction",
        source_root=mother_root,
        final_scenario_root=final_root,
        split="train",
        sop03_root=tmp_path / "sop03",
        long40_human_artifact=tmp_path / "long40",
        base_state_start=0,
        max_base_states=1,
        base_config=base_config,
        source_config_digest=teb_config.digest,
        centerline_epsilon_m=(
            teb_config.occlusion.centerline_intersection_epsilon_m
        ),
    )
    history_input = source.resolve_history_renderer_input(source.accepted[0])
    resolved = source.resolve(source.accepted[0])

    assert source.source_publication_semantic_digest == (
        published.source_publication_semantic_digest
    )
    assert len(source.final_release_identity) == 64
    assert resolved.publication.sample_id == scenario_id
    assert resolved.publication.mother_id == event.generated_event_id
    target_id = event.target.target_dynamic_object_id
    observed = resolved.publication.renderer_input.scene_dynamic_history_observed
    expected_hidden = (
        ()
        if target_id in observed and bool(observed[target_id][-1])
        else (target_id,)
    )
    assert resolved.publication.hidden_object_ids == expected_hidden
    rendered = render_sop06_single_publication(
        resolved.publication,
        config=source.base_config,
    )
    history_rendered = render_sop06_single_input(
        history_input,
        config=source.base_config,
    )
    np.testing.assert_array_equal(history_rendered.bev_history, rendered.bev_history)
    np.testing.assert_array_equal(
        history_rendered.state_channels,
        rendered.state_channels,
    )
    assert rendered.bev_history.shape == (8, 2, 160, 160)
    assert rendered.state_channels.shape == (9, 160, 160)
