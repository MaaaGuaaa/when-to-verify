"""Immutable SOP06 history-only shard tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.datasets.sop06_history_bev as history_bev_module
from src.datasets.sop06_history_bev import (
    Sop06HistoryBevSample,
    Sop06HistoryShardProvenance,
    load_sop06_history_shard,
    write_sop06_history_shard,
)


def _sample(sample_id: str, mother_id: str) -> Sop06HistoryBevSample:
    history = np.zeros((8, 2, 5, 5), dtype=np.float32)
    history[:, 1] = 1.0
    state = np.zeros((9, 5, 5), dtype=np.float32)
    state[0] = 1.0
    return Sop06HistoryBevSample(
        sample_id=sample_id,
        mother_id=mother_id,
        split="val",
        regime="seen_then_occluded",
        bev_history=history,
        state_channels=state,
        renderer_metadata={
            "renderer_layout_version": "bev_history2_state9_v1",
            "base_state_id": f"state-{mother_id}",
            "sensor_config_digest": "a" * 32,
            "static_occupancy_digest": "b" * 32,
        },
    )


def _provenance() -> Sop06HistoryShardProvenance:
    return Sop06HistoryShardProvenance(
        source_family="natural",
        source_mode="complete_mother",
        split="val",
        source_publication_semantic_digest="c" * 64,
        final_release_identity="d" * 64,
        final_scenario_root="outputs/final-val",
    )


def test_history_bev_shard_round_trips_without_oracle_fields(
    tmp_path: Path,
) -> None:
    output = tmp_path / "shard-00000"
    write_sop06_history_shard(
        (_sample("scenario-a", "mother-a"), _sample("scenario-b", "mother-b")),
        output,
        shard_index=0,
        expected_sample_count=2,
        provenance=_provenance(),
    )

    loaded = load_sop06_history_shard(output)

    assert loaded.provenance == _provenance()
    assert tuple(item.sample_id for item in loaded.samples) == (
        "scenario-a",
        "scenario-b",
    )
    assert {path.name for path in output.iterdir()} == {
        "observations.npz",
        "metadata.jsonl",
        "summary.json",
        "checksums.json",
        "COMPLETE.json",
    }
    assert all(item.bev_history.shape == (8, 2, 5, 5) for item in loaded.samples)
    assert all(item.state_channels.shape == (9, 5, 5) for item in loaded.samples)
    assert all(
        not any(
            token in key.lower()
            for token in (
                "future",
                "oracle",
                "angle",
                "attempt",
                "collision",
                "clearance",
                "risk",
            )
        )
        for item in loaded.samples
        for key in item.renderer_metadata
    )


def test_history_bev_shard_rejects_tampering_and_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "shard-00000"
    samples = (_sample("scenario-a", "mother-a"),)
    write_sop06_history_shard(
        samples,
        output,
        shard_index=0,
        expected_sample_count=1,
        provenance=_provenance(),
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        write_sop06_history_shard(
            samples,
            output,
            shard_index=0,
            expected_sample_count=1,
            provenance=_provenance(),
        )

    summary = output / "summary.json"
    summary.write_bytes(summary.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        load_sop06_history_shard(output)


def test_history_shard_write_and_checkpoint_do_not_reload_arrays(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "shard-00000"
    monkeypatch.setattr(
        history_bev_module,
        "load_sop06_history_shard",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("writer used strict shard reload")
        ),
    )
    monkeypatch.setattr(
        history_bev_module.np,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint opened observation arrays")
        ),
    )

    write_sop06_history_shard(
        (_sample("scenario-a", "mother-a"),),
        output,
        shard_index=0,
        expected_sample_count=1,
        provenance=_provenance(),
    )
    load_checkpoint = getattr(
        history_bev_module,
        "load_sop06_history_shard_checkpoint",
        None,
    )
    assert callable(load_checkpoint)
    checkpoint = load_checkpoint(output)

    assert checkpoint.sample_ids == ("scenario-a",)
    assert checkpoint.mother_ids == ("mother-a",)
    assert checkpoint.provenance == _provenance()
