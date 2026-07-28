"""Persisted-observation, dual-ID, and resume tests for SOP07 releases."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.datasets.risk_dataset as risk_dataset
import src.generation.sop06_history_release as sop06_release
import src.generation.sop07_risk_release as sop07_release
from src.contracts import (
    POSE_TIME_LAYOUT_VERSION,
    SCHEMA_VERSION,
    RiskSample,
    build_grid_spec,
)
from src.datasets.sop06_history_bev import Sop06HistoryBevSample
from src.generation.observation_renderer import RENDERER_LAYOUT_VERSION
from src.generation.risk_gt import RISK_GT_VERSION
from src.generation.risk_gt import resolve_no_object_clearance_sentinel
from src.generation.sop06_finalized_source import Sop06AcceptedFinalRecord
from src.generation.sop06_history_release import (
    Sop06HistoryReleaseRequest,
    publish_sop06_history_release,
)
from src.generation.sop07_risk_release import (
    Sop07RiskReleaseRequest,
    load_sop07_risk_release,
    publish_sop07_risk_release,
)
from tests.test_sop06_finalized_source import _publish_complete_final
from tests.test_sop05r_teb_sop06_handoff import _strict_collection


def _base_config() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "bev": {
            "range_m": 4.0,
            "resolution_m": 1.0,
            "size": 4,
            "history_steps": 8,
            "history_dt_s": 0.2,
            "future_steps": 32,
            "future_dt_s": 0.2,
        },
        "risk_gt": {
            "sigma_distance_m": 0.5,
            "sigma_time_s": 2.0,
            "near_miss_distance_m": 0.35,
        },
    }


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


def _mock_source(count: int):
    source = SimpleNamespace(
        source_mode="complete_mother",
        source_publication_semantic_digest="a" * 64,
        final_release_identity="b" * 64,
        base_config=_base_config(),
        accepted=tuple(_accepted(index) for index in range(count)),
    )
    source.prepare_boundary = lambda boundary: source
    return source


def _render_one(source, accepted) -> Sop06HistoryBevSample:
    grid = build_grid_spec(source.base_config)
    history = np.zeros(
        (grid.history_steps, grid.n_history_channels, grid.height, grid.width),
        dtype=np.float32,
    )
    history[:, 1] = 1.0
    state = np.zeros(
        (grid.n_state_channels, grid.height, grid.width),
        dtype=np.float32,
    )
    return Sop06HistoryBevSample(
        sample_id=accepted.scenario_id,
        mother_id=accepted.mother_id,
        split=accepted.split,
        regime=accepted.regime,
        bev_history=history,
        state_channels=state,
        renderer_metadata={
            "renderer_layout_version": RENDERER_LAYOUT_VERSION,
            "base_state_id": f"base-{accepted.scenario_id}",
            "sensor_config_digest": f"sensor-{accepted.scenario_id}",
            "static_occupancy_digest": f"static-{accepted.scenario_id}",
        },
    )


def _build_one(source, accepted, observation, binding) -> RiskSample:
    grid = build_grid_spec(source.base_config)
    provenance = {
        "mother_id": accepted.mother_id,
        "base_recording_id": f"base-recording-{accepted.scenario_id}",
        "base_session_id": "base-session",
        "source_recording_id": f"source-recording-{accepted.scenario_id}",
        "source_session_id": "source-session",
        "source_object_id": f"source-object-{accepted.scenario_id}",
        "source_snippet_id": f"snippet-{accepted.scenario_id}",
        "seed_namespace": f"sop07/train/{accepted.scenario_id}",
        "base_config_digest": "c" * 32,
        "target_present": True,
        "target_currently_observed": True,
        "sop06_history_release_manifest_digest": (
            binding.sop06_release_manifest_digest
        ),
        "sop06_history_shard_index": binding.sop06_shard_index,
        "sop06_history_shard_semantic_digest": (
            binding.sop06_shard_semantic_digest
        ),
        "sop06_source_family": binding.source_family,
        "sop06_source_mode": binding.source_mode,
    }
    return RiskSample(
        sample_id=observation.sample_id,
        split=observation.split,
        base_state_id=observation.renderer_metadata["base_state_id"],
        pair_group_id=f"sop06-single/{observation.mother_id}",
        event_type=observation.regime,
        bev_history=np.array(observation.bev_history, copy=True),
        state_channels=np.array(observation.state_channels, copy=True),
        trajectory_channels=np.zeros(
            (
                grid.n_trajectory_channels,
                grid.height,
                grid.width,
            ),
            dtype=np.float32,
        ),
        robot_state=np.zeros(2, dtype=np.float32),
        collision_label=0,
        risk_severity=0.0,
        min_clearance=resolve_no_object_clearance_sentinel(grid),
        near_miss=0,
        first_collision_time=None,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "renderer": dict(observation.renderer_metadata),
            "trajectory_id": f"trajectory-{accepted.scenario_id}",
            "provenance": provenance,
            "label_audit": {
                "risk_gt_version": RISK_GT_VERSION,
                "pose_time_layout_version": POSE_TIME_LAYOUT_VERSION,
                "critical_object_id": None,
                "critical_object_type": None,
                "time_to_min_clearance_s": None,
                "has_hidden_target": False,
            },
        },
    )


def _mock_sop06_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    count: int,
) -> tuple[object, Path]:
    source = _mock_source(count)
    monkeypatch.setattr(sop06_release, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(sop06_release, "_load_source", lambda request: source)
    output = tmp_path / "sop06"
    publish_sop06_history_release(
        Sop06HistoryReleaseRequest(
            source_family="natural",
            source_mode="complete_mother",
            source_root=Path("outputs/mother"),
            final_scenario_root=Path("outputs/final"),
            split="train",
            output_dir=output,
            workers=1,
            samples_per_shard=2,
        ),
        render_one=_render_one,
    )
    return source, output


def test_release_resumes_matching_sop06_shards_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, sop06_root = _mock_sop06_release(
        tmp_path,
        monkeypatch,
        count=5,
    )
    monkeypatch.setattr(sop07_release, "_REPOSITORY_ROOT", tmp_path)
    request = Sop07RiskReleaseRequest(
        sop06_release_root=sop06_root,
        output_dir=tmp_path / "sop07",
    )
    calls = 0

    def interrupt(source_value, accepted, observation, binding):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("intentional interruption")
        return _build_one(source_value, accepted, observation, binding)

    with pytest.raises(RuntimeError, match="intentional interruption"):
        publish_sop07_risk_release(
            request,
            source_loader=lambda upstream: source,
            build_one=interrupt,
        )

    in_progress = tmp_path / ".sop07.inprogress"
    assert (in_progress / "shards" / "shard-00000" / "summary.json").is_file()
    result = publish_sop07_risk_release(
        request,
        source_loader=lambda upstream: source,
        build_one=_build_one,
    )
    replay = publish_sop07_risk_release(
        request,
        source_loader=lambda upstream: source,
        build_one=_build_one,
    )
    loaded = load_sop07_risk_release(request.output_dir)

    assert result.sample_count == 5
    assert result.shard_count == 3
    assert result.reused_shard_count == 1
    assert replay.reused_shard_count == 3
    assert loaded.sample_count == 5
    assert not in_progress.exists()


def test_observation_oracle_join_requires_both_ids() -> None:
    observation = _render_one(_mock_source(1), _accepted(0))
    wrong = replace(_accepted(0), mother_id="different-mother")

    with pytest.raises(ValueError, match="sample_id/mother_id join failed"):
        sop07_release._validate_observation_join(observation, wrong)


def test_complete_mother_lineage_is_bound_to_sop07_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, sop06_root = _mock_sop06_release(tmp_path, monkeypatch, count=1)
    monkeypatch.setattr(sop07_release, "_REPOSITORY_ROOT", tmp_path)
    upstream = sop07_release.load_sop06_history_release_checkpoint(sop06_root)

    payload = sop07_release._request_payload(
        Sop07RiskReleaseRequest(
            sop06_release_root=sop06_root,
            output_dir=tmp_path / "sop07",
            sop03_root=tmp_path / "inputs" / "sop03",
            long40_human_artifact=tmp_path / "inputs" / "long40",
        ),
        upstream,
        base_config_digest="a" * 64,
        risk_config_digest="b" * 64,
    )

    assert payload["complete_mother_lineage"] == {
        "sop03_root": "inputs/sop03",
        "long40_human_artifact": "inputs/long40",
    }


def test_legacy_complete_fixture_without_identity_provenance_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = _strict_collection(tmp_path)
    mother_root = tmp_path / "m7"
    final_root = tmp_path / "final"
    _publish_complete_final(collection, final_root, monkeypatch)
    monkeypatch.setattr(sop06_release, "_REPOSITORY_ROOT", tmp_path)
    sop06_root = tmp_path / "sop06-real"
    publish_sop06_history_release(
        Sop06HistoryReleaseRequest(
            source_family="natural",
            source_mode="complete_mother",
            source_root=mother_root,
            final_scenario_root=final_root,
            split="train",
            output_dir=sop06_root,
            workers=1,
            samples_per_shard=1,
        )
    )
    monkeypatch.setattr(sop07_release, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        risk_dataset,
        "render_observation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("SOP07 rerendered a persisted SOP06 observation")
        ),
    )

    with pytest.raises(ValueError, match="provenance is incomplete"):
        publish_sop07_risk_release(
            Sop07RiskReleaseRequest(
                sop06_release_root=sop06_root,
                output_dir=tmp_path / "sop07-real",
            )
        )
