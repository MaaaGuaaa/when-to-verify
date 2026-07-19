from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.contracts import OracleWorld
from src.generation.scenario_bank import (
    SCENARIO_BANK_VERSION,
    ScenarioBankGeometryError,
    build_scenario_bank,
    load_scenario_bank_config,
    validate_scenario_bank,
)
from tests.fixtures.verification_world import build_verification_toy_world


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "verification_gt.yaml"


def _current_inputs():
    toy = build_verification_toy_world()
    world = OracleWorld(
        world_id="toy-current-world",
        base_state_id="toy-base",
        static_occupancy=toy.static_occupancy.copy(),
        dynamic_object_trajectories={
            key: value.copy() for key, value in toy.dynamic_future_poses.items()
        },
        dynamic_object_specs={key: dict(value) for key, value in toy.dynamic_specs.items()},
        occluders=(),
        blind_spot_config={"kind": "structural", "occluder_ids": []},
        random_seed=42,
        metadata={"split": "train", "source_namespace": "toy/train/source-0"},
    )
    return toy, world


def _build(size: int = 16):
    toy, world = _current_inputs()
    config = load_scenario_bank_config(CONFIG)
    bank = build_scenario_bank(
        current_world=world,
        target_object_id="critical_cart",
        current_dynamic_poses=toy.dynamic_current_poses,
        current_visible_mask=toy.current_visible_mask,
        grid=toy.grid,
        split="train",
        source_namespace="toy/train/source-0",
        seed=42,
        size=size,
        config=config,
    )
    return toy, bank


def test_m16_frozen_composition_and_m8_m32_presets():
    config = load_scenario_bank_config(CONFIG)
    assert config.version == SCENARIO_BANK_VERSION
    assert config.composition(8) == {
        "current": 1,
        "empty": 1,
        "temporal": 2,
        "spatial": 2,
        "speed": 1,
        "irrelevant": 1,
    }
    assert config.composition(16) == {
        "current": 1,
        "empty": 2,
        "temporal": 5,
        "spatial": 4,
        "speed": 2,
        "irrelevant": 2,
    }
    assert config.composition(32) == {
        "current": 1,
        "empty": 4,
        "temporal": 10,
        "spatial": 8,
        "speed": 4,
        "irrelevant": 5,
    }
    for size in (8, 16, 32):
        toy, bank = _build(size)
        assert len(bank.hypotheses) == size
        assert Counter(item.variant_kind for item in bank.hypotheses) == Counter(
            config.composition(size)
        )
        validate_scenario_bank(bank, grid=toy.grid)


def test_bank_is_deterministic_and_preserves_non_target_circle_exactly():
    toy, first = _build(16)
    _, second = _build(16)
    assert first.semantic_digest == second.semantic_digest
    assert [item.hypothesis_id for item in first.hypotheses] == [
        item.hypothesis_id for item in second.hypotheses
    ]
    for left, right in zip(first.hypotheses, second.hypotheses, strict=True):
        assert left.variant_kind == right.variant_kind
        for object_id in left.world.dynamic_object_trajectories:
            np.testing.assert_array_equal(
                left.world.dynamic_object_trajectories[object_id],
                right.world.dynamic_object_trajectories[object_id],
            )

    reference = first.hypotheses[0]
    assert reference.variant_kind == "current"
    non_target_future = reference.world.dynamic_object_trajectories[
        "irrelevant_person"
    ]
    non_target_spec = reference.world.dynamic_object_specs["irrelevant_person"]
    assert non_target_spec["footprint"]["kind"] == "circle"
    assert reference.world.dynamic_object_specs["critical_cart"]["footprint"][
        "kind"
    ] == "rectangle"
    for hypothesis in first.hypotheses:
        np.testing.assert_array_equal(
            hypothesis.world.dynamic_object_trajectories["irrelevant_person"],
            non_target_future,
        )
        assert (
            hypothesis.world.dynamic_object_specs["irrelevant_person"]
            == non_target_spec
        )
        assert np.isfinite(hypothesis.world.static_occupancy).all()
        assert hypothesis.seed_namespace.startswith("scenario/train/")
    assert len({item.seed_namespace for item in first.hypotheses}) == 16
    assert first.current_visible_occupancy_digest
    validate_scenario_bank(first, grid=toy.grid)


def test_validator_rejects_static_visible_and_non_target_mutations():
    toy, bank = _build(16)
    hypothesis = bank.hypotheses[1]

    changed_static = hypothesis.world.static_occupancy.copy()
    changed_static[0, 0] = 1.0
    static_world = replace(hypothesis.world, static_occupancy=changed_static)
    static_bank = replace(
        bank,
        hypotheses=(
            bank.hypotheses[0],
            replace(hypothesis, world=static_world),
            *bank.hypotheses[2:],
        ),
    )
    with pytest.raises(ValueError, match="static occupancy"):
        validate_scenario_bank(static_bank, grid=toy.grid)

    temporal_index = next(
        index
        for index, item in enumerate(bank.hypotheses)
        if item.variant_kind == "temporal"
    )
    temporal = bank.hypotheses[temporal_index]
    visible_current = {
        key: value.copy() for key, value in temporal.current_dynamic_poses.items()
    }
    visible_current["critical_cart"] = np.asarray(
        [1.5, 0.0, 0.0], dtype=np.float32
    )
    visible_items = list(bank.hypotheses)
    visible_items[temporal_index] = replace(
        temporal, current_dynamic_poses=visible_current
    )
    with pytest.raises(ValueError, match="visible occupancy"):
        validate_scenario_bank(
            replace(bank, hypotheses=tuple(visible_items)), grid=toy.grid
        )

    non_target_world = replace(
        temporal.world,
        dynamic_object_trajectories={
            **temporal.world.dynamic_object_trajectories,
            "irrelevant_person": temporal.world.dynamic_object_trajectories[
                "irrelevant_person"
            ]
            + np.asarray([0.0, 0.1, 0.0], dtype=np.float32),
        },
    )
    non_target_items = list(bank.hypotheses)
    non_target_items[temporal_index] = replace(temporal, world=non_target_world)
    with pytest.raises(ValueError, match="non-target"):
        validate_scenario_bank(
            replace(bank, hypotheses=tuple(non_target_items)), grid=toy.grid
        )


def test_builder_rejects_unsupported_size_and_cross_split_namespace():
    toy, world = _current_inputs()
    config = load_scenario_bank_config(CONFIG)
    kwargs = dict(
        current_world=world,
        target_object_id="critical_cart",
        current_dynamic_poses=toy.dynamic_current_poses,
        current_visible_mask=toy.current_visible_mask,
        grid=toy.grid,
        split="train",
        source_namespace="toy/train/source-0",
        seed=42,
        config=config,
    )
    with pytest.raises(ValueError, match="8, 16, or 32"):
        build_scenario_bank(size=12, **kwargs)
    with pytest.raises(ValueError, match="split"):
        build_scenario_bank(
            size=8,
            **{
                **kwargs,
                "split": "val",
                "source_namespace": "toy/train/source-0",
            },
        )


def test_builder_reports_typed_static_geometry_ineligibility():
    toy, world = _current_inputs()
    colliding_static = world.static_occupancy.copy()
    colliding_static[toy.critical_mask] = 1.0
    colliding_world = replace(world, static_occupancy=colliding_static)

    with pytest.raises(ScenarioBankGeometryError) as captured:
        build_scenario_bank(
            current_world=colliding_world,
            target_object_id="critical_cart",
            current_dynamic_poses=toy.dynamic_current_poses,
            current_visible_mask=toy.current_visible_mask,
            grid=toy.grid,
            split="train",
            source_namespace="toy/train/source-0",
            seed=42,
            size=8,
            config=load_scenario_bank_config(CONFIG),
        )

    assert captured.value.variant_kind == "current"
    assert captured.value.object_id == "critical_cart"
    assert captured.value.future_index == 0
