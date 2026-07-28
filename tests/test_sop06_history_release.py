"""Resumable immutable SOP06 entry-release tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import src.generation.sop06_history_release as release_module
from src.datasets.sop06_history_bev import Sop06HistoryBevSample
from src.generation.sop06_finalized_source import Sop06AcceptedFinalRecord
from src.generation.sop06_history_release import (
    Sop06HistoryReleaseRequest,
    load_sop06_history_release,
    publish_sop06_history_release,
)
from tests.test_sop06_finalized_source import (
    _publish_complete_final,
    _strict_collection,
)


def _accepted(index: int) -> Sop06AcceptedFinalRecord:
    return Sop06AcceptedFinalRecord(
        source_index=index,
        mother_id=f"mother-{index}",
        scenario_id=f"scenario-{index}",
        split="train",
        regime="unseen_in_history_window",
        target_present=True,
        target_row=index,
    )


def _source(count: int):
    return SimpleNamespace(
        source_mode="complete_mother",
        source_publication_semantic_digest="a" * 64,
        final_release_identity="b" * 64,
        base_config={},
        accepted=tuple(_accepted(index) for index in range(count)),
    )


def _render_one(source, accepted) -> Sop06HistoryBevSample:
    history = np.zeros((8, 2, 5, 5), dtype=np.float32)
    history[:, 1] = 1.0
    state = np.zeros((9, 5, 5), dtype=np.float32)
    state[0] = 1.0
    return Sop06HistoryBevSample(
        sample_id=accepted.scenario_id,
        mother_id=accepted.mother_id,
        split=accepted.split,
        regime=accepted.regime,
        bev_history=history,
        state_channels=state,
        renderer_metadata={
            "renderer_layout_version": "bev_history2_state9_v1",
            "base_state_id": f"state-{accepted.source_index}",
            "sensor_config_digest": "c" * 32,
            "static_occupancy_digest": "d" * 32,
        },
    )


def _request(tmp_path: Path) -> Sop06HistoryReleaseRequest:
    return Sop06HistoryReleaseRequest(
        source_family="natural",
        source_mode="complete_mother",
        source_root=Path("outputs/mother"),
        final_scenario_root=Path("outputs/final"),
        split="train",
        output_dir=tmp_path / "entry",
        workers=1,
        samples_per_shard=2,
    )


def test_entry_release_resumes_only_matching_completed_subshards(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _source(5)
    monkeypatch.setattr(
        release_module,
        "load_sop06_finalized_source",
        lambda **kwargs: source,
    )
    request = _request(tmp_path)
    calls = 0

    def interrupt(source_value, accepted):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("intentional interruption")
        return _render_one(source_value, accepted)

    try:
        publish_sop06_history_release(request, render_one=interrupt)
    except RuntimeError as exc:
        assert str(exc) == "intentional interruption"
    else:
        raise AssertionError("release unexpectedly ignored interruption")

    in_progress = request.output_dir.parent / f".{request.output_dir.name}.inprogress"
    assert (in_progress / "shards" / "shard-00000" / "COMPLETE.json").is_file()

    with monkeypatch.context() as fast_path:
        fast_path.setattr(
            release_module,
            "load_sop06_history_shard",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("resume used strict shard loader")
            ),
        )
        fast_path.setattr(
            release_module,
            "load_sop06_history_release",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("publish used strict release loader")
            ),
        )
        result = publish_sop06_history_release(
            request,
            render_one=_render_one,
        )
        replay = publish_sop06_history_release(request, render_one=_render_one)
    loaded = load_sop06_history_release(request.output_dir)

    assert result.sample_count == 5
    assert result.shard_count == 3
    assert result.reused_shard_count == 1
    assert loaded.sample_count == 5
    assert loaded.shard_count == 3
    assert not in_progress.exists()

    assert replay.reused_shard_count == 3


def test_complete_mother_fixture_runs_end_to_end(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mother_root = tmp_path / "m7"
    collection = _strict_collection(tmp_path)
    final_root = tmp_path / "final"
    scenario_id = _publish_complete_final(collection, final_root, monkeypatch)
    monkeypatch.setattr(release_module, "_REPOSITORY_ROOT", tmp_path)
    output = tmp_path / "sop06"

    result = publish_sop06_history_release(
        Sop06HistoryReleaseRequest(
            source_family="natural",
            source_mode="complete_mother",
            source_root=mother_root,
            final_scenario_root=final_root,
            split="train",
            output_dir=output,
            workers=1,
            samples_per_shard=1,
        )
    )
    loaded = load_sop06_history_release(output)

    assert result.sample_count == 1
    assert result.shard_count == 1
    assert loaded.sample_count == 1
    shard = release_module.load_sop06_history_shard(
        output / "shards" / "shard-00000"
    )
    assert shard.samples[0].sample_id == scenario_id
    assert shard.samples[0].bev_history.shape == (8, 2, 160, 160)
    assert shard.samples[0].state_channels.shape == (9, 160, 160)


def test_parallel_boundary_returns_history_samples_to_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mother_root = tmp_path / "m7"
    collection = _strict_collection(tmp_path)
    final_root = tmp_path / "final"
    _publish_complete_final(collection, final_root, monkeypatch)
    source = release_module.load_sop06_finalized_source(
        source_mode="complete_mother",
        source_root=mother_root,
        final_scenario_root=final_root,
        split="train",
    )
    first = source.accepted[0]
    second = replace(first, scenario_id=f"{first.scenario_id}-second")
    accepted = (first, second)
    object.__setattr__(source, "accepted", accepted)
    object.__setattr__(source, "_accepted_set", frozenset(accepted))

    samples = release_module._render_boundary(
        source,
        accepted,
        workers=2,
        render_one=release_module._default_render_one,
    )

    assert tuple(sample.sample_id for sample in samples) == tuple(
        record.scenario_id for record in accepted
    )
    assert all(isinstance(sample.renderer_metadata, dict) is False for sample in samples)


def test_source_cache_changes_read_location_without_changing_request_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(release_module, "_REPOSITORY_ROOT", tmp_path)
    source_root = tmp_path / "outputs" / "source"
    cache_root = tmp_path / "cache" / "source"
    request = Sop06HistoryReleaseRequest(
        source_family="natural",
        source_mode="partial_m6_reconstruction",
        source_root=source_root,
        source_cache_root=cache_root,
        final_scenario_root=tmp_path / "outputs" / "final",
        split="train",
        output_dir=tmp_path / "outputs" / "sop06",
        sop03_root=tmp_path / "outputs" / "sop03",
        long40_human_artifact=tmp_path / "outputs" / "long40",
        base_state_start=0,
        max_base_states=1,
        base_config_path=tmp_path / "configs" / "base.yaml",
        generator_config_path=tmp_path / "configs" / "generator.yaml",
    )
    uncached = replace(request, source_cache_root=None)
    assert release_module._request_document(request) == release_module._request_document(
        uncached
    )

    loaded = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(release_module, "load_config", lambda path: {})
    monkeypatch.setattr(
        release_module,
        "load_sop05r_teb_config",
        lambda path: SimpleNamespace(
            digest="c" * 64,
            occlusion=SimpleNamespace(centerline_intersection_epsilon_m=0.01),
        ),
    )

    def load_source(**kwargs):
        captured.update(kwargs)
        return loaded

    monkeypatch.setattr(release_module, "load_sop06_finalized_source", load_source)

    assert release_module._load_source(request) is loaded
    assert captured["source_root"] == cache_root


def test_default_renderer_uses_history_only_source(monkeypatch) -> None:
    accepted = _accepted(0)
    renderer_input = object()

    class Source:
        base_config = {"grid": "fixture"}

        def resolve_history_renderer_input(self, record):
            assert record == accepted
            return renderer_input

        def resolve(self, record):
            raise AssertionError("history rendering loaded oracle trajectory")

    monkeypatch.setattr(
        release_module,
        "render_sop06_single_input",
        lambda value, *, config: SimpleNamespace(
            bev_history=np.zeros((8, 2, 5, 5), dtype=np.float32),
            state_channels=np.zeros((9, 5, 5), dtype=np.float32),
            metadata={
                "renderer_layout_version": "bev_history2_state9_v1",
                "base_state_id": "state-0",
                "sensor_config_digest": "c" * 32,
                "static_occupancy_digest": "d" * 32,
            },
        ),
    )

    sample = release_module._default_render_one(Source(), accepted)

    assert sample.sample_id == accepted.scenario_id
