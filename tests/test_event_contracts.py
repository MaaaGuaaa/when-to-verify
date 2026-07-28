from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import numpy as np

from src.contracts import OracleWorld
from src.generation.event_target_motion_shard import (
    compute_footprint_spec_digest,
    create_event_target_motion_record,
)


def _event_contracts():
    spec = importlib.util.find_spec("src.generation.event_contracts")
    assert spec is not None, "Long40 event contracts module is missing"
    return importlib.import_module("src.generation.event_contracts")


def test_long40_event_contracts_construct_an_independent_target_event():
    contracts = _event_contracts()
    footprint_spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.25},
    }
    history = np.zeros((8, 3), dtype=np.float32)
    future = np.zeros((32, 3), dtype=np.float32)
    record = create_event_target_motion_record(
        generated_event_id="event-1",
        world_id="world-1",
        base_state_id="state-1",
        trajectory_id="trajectory-1",
        target_dynamic_object_id="target-1",
        source_snippet_id="snippet-1",
        source_object_id="source-1",
        object_type="human",
        footprint_spec=footprint_spec,
        footprint_spec_digest=compute_footprint_spec_digest(footprint_spec),
        target_type_policy_digest="a" * 32,
        history_poses=history,
        current_pose=history[-1],
        future_poses=future,
    )
    target = contracts.TransplantedDynamicObject(
        target_dynamic_object_id="target-1",
        source_object_id="source-1",
        snippet_id="snippet-1",
        object_type="human",
        footprint_spec=footprint_spec,
        footprint_spec_digest=record.footprint_spec_digest,
        history_poses=history,
        current_pose=history[-1],
        future_poses=future,
        provenance={},
    )
    world = OracleWorld(
        world_id="world-1",
        base_state_id="state-1",
        static_occupancy=np.zeros((4, 4), dtype=np.float32),
        dynamic_object_trajectories={"target-1": future},
        dynamic_object_specs={"target-1": footprint_spec},
        occluders=(),
        blind_spot_config={},
        random_seed=1,
        metadata={"schema_version": "4.0.0"},
    )
    event = contracts.GeneratedEvent(
        generated_event_id="event-1",
        event_kind="long40",
        world=world,
        target=target,
        target_motion_record=record,
        visibility_sequence=np.zeros(32, dtype=np.bool_),
        target_visibility_history=np.zeros(8, dtype=np.bool_),
        conflict_time_s=1.0,
        conflict_index=4,
    )

    assert event.target.future_poses.shape == (32, 3)
    assert contracts.footprint_from_spec(footprint_spec).radius_m == 0.25


def test_teb_production_entrypoints_do_not_import_retired_contract_modules():
    root = Path(__file__).resolve().parents[1]
    modules = (
        root / "src/generation/sop05r_teb_event_sampler.py",
        root / "src/generation/sop05r_teb_output_loader.py",
    )

    for module in modules:
        source = module.read_text(encoding="utf-8")
        assert "dynamic_object_transplant" not in source
        assert "event_sampler" not in source


def test_long40_footprint_consumers_do_not_import_retired_transplant_module():
    root = Path(__file__).resolve().parents[1]
    modules = (
        "src/generation/anchored_human_placement.py",
        "src/generation/counterfactual_verify.py",
        "src/generation/observation_renderer.py",
        "src/generation/risk_gt.py",
        "src/generation/scenario_bank.py",
        "src/generation/sop05_final_scenarios.py",
        "src/generation/sop05_unseen_prior.py",
        "src/generation/sop05r_teb_revealability.py",
        "src/generation/sop05r_teb_templates.py",
        "src/generation/verification_pipeline.py",
        "src/generation/verification_response.py",
    )

    for relative in modules:
        source = (root / relative).read_text(encoding="utf-8")
        assert "dynamic_object_transplant" not in source
