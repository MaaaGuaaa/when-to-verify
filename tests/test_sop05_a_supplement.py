from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from src.generation.sop05r_teb_output_loader import LoadedSop05rTebOutput


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sop05_a_supplement.yaml"


def _module():
    return importlib.import_module("src.generation.sop05_a_supplement")


def _source(*, selection_mode: str = "h0_hidden", count: int = 6):
    source = object.__new__(LoadedSop05rTebOutput)
    event_ids = tuple(f"mother-{index}" for index in range(count))
    object.__setattr__(
        source,
        "events",
        tuple(
            SimpleNamespace(
                generated_event_id=event_id,
                target_visibility_history=np.zeros(8, dtype=np.bool_),
            )
            for event_id in event_ids
        ),
    )
    object.__setattr__(
        source,
        "manifest",
        {
            "config_digest": "source-config",
            "source_evidence": {"placement_selection_mode": selection_mode},
        },
    )
    object.__setattr__(source, "publication_semantic_digest", "source-digest")
    object.__setattr__(source, "complete", True)
    object.__setattr__(
        source,
        "trajectories",
        SimpleNamespace(
            records=tuple(
                SimpleNamespace(event_id=event_id, decision_state_id=f"decision-{index}")
                for index, event_id in enumerate(event_ids)
            )
        ),
    )
    object.__setattr__(
        source,
        "decision_states",
        {
            f"decision-{index}": SimpleNamespace(split="train")
            for index in range(count)
        },
    )
    return source


def test_config_freezes_the_approved_additional_a_quotas() -> None:
    supplement = _module()
    config = supplement.load_sop05_a_supplement_config(CONFIG)

    assert config.version == "sop05_a_supplement_v1"
    assert config.present_max_attempts_per_mother == 256
    assert {
        split: (quota.accepted, quota.present, quota.empty)
        for split, quota in config.quotas.items()
    } == {
        "train": (16531, 2859, 13672),
        "calibration": (2221, 383, 1838),
        "val": (2049, 382, 1667),
        "test": (2073, 384, 1689),
    }
    assert set(config.source_generation_seeds) == {
        "train",
        "calibration",
        "val",
        "test",
    }
    assert dict(config.source_mother_quotas) == {
        "train": 23000,
        "calibration": 3000,
        "val": 3000,
        "test": 3000,
    }


def test_present_angles_are_deterministic_uniform_half_open_samples() -> None:
    supplement = _module()
    config = supplement.load_sop05_a_supplement_config(CONFIG)

    first = supplement.sample_present_angles(
        config=config,
        split="train",
        source_publication_semantic_digest="source-digest",
        mother_id="mother-0",
    )
    repeated = supplement.sample_present_angles(
        config=config,
        split="train",
        source_publication_semantic_digest="source-digest",
        mother_id="mother-0",
    )
    other = supplement.sample_present_angles(
        config=config,
        split="train",
        source_publication_semantic_digest="source-digest",
        mother_id="mother-1",
    )

    assert first.shape == (256,)
    assert first.dtype == np.float64
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, other)
    assert np.all(first >= -np.pi)
    assert np.all(first < np.pi)


def test_selection_meets_present_and_empty_quotas_without_reusing_mothers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplement = _module()
    config = supplement.load_sop05_a_supplement_config(CONFIG)
    config = replace(
        config,
        quotas=MappingProxyType(
            {"train": supplement.Sop05ASupplementQuota(accepted=3, present=1)}
        ),
    )
    observed_states = []

    def accept_present(source, event, *, split, config, state=None):
        del source, split, config
        observed_states.append(state)
        return supplement._PresentEvaluation(
            mother_id=event.generated_event_id,
            history_poses=np.zeros((8, 3), dtype=np.float32),
            future_poses=np.ones((32, 3), dtype=np.float32),
            attempted_angle_count=2,
            selected_angle_rad=0.25,
            rejection_reason_counts={"history_visible": 1},
        )

    monkeypatch.setattr(supplement, "_evaluate_present_event", accept_present)
    selections = supplement.select_a_supplement_scenarios(
        _source(),
        split="train",
        config=config,
        workers=1,
    )

    assert len(selections) == 3
    assert observed_states and all(state is not None for state in observed_states)
    assert sum(item.target_present for item in selections) == 1
    assert len({item.mother_id for item in selections}) == 3
    assert {item.provenance["stratum"] for item in selections} == {
        "a_present",
        "a_empty",
    }
    empty = [item for item in selections if not item.target_present]
    assert all(not np.any(item.history_poses) for item in empty)
    assert all(not np.any(item.future_poses) for item in empty)


def test_selection_rejects_a_source_not_generated_in_h0_hidden_mode() -> None:
    supplement = _module()
    config = supplement.load_sop05_a_supplement_config(CONFIG)
    config = replace(
        config,
        quotas=MappingProxyType(
            {"train": supplement.Sop05ASupplementQuota(accepted=1, present=0)}
        ),
    )

    with pytest.raises(supplement.Sop05ASupplementError, match="h0_hidden"):
        supplement.select_a_supplement_scenarios(
            _source(selection_mode="seen_first", count=1),
            split="train",
            config=config,
        )


def test_source_validation_builds_the_event_state_index_in_one_pass() -> None:
    supplement = _module()
    source = _source(count=4)
    records = tuple(source.trajectories.records)

    class SinglePassRecords:
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("trajectory records were rescanned")
            return iter(records)

    single_pass = SinglePassRecords()
    object.__setattr__(source, "trajectories", SimpleNamespace(records=single_pass))

    states = supplement._validate_source(
        source,
        split="train",
        quota=supplement.Sop05ASupplementQuota(accepted=1, present=0),
    )

    assert set(states) == {event.generated_event_id for event in source.events}
    assert single_pass.iterations == 1
