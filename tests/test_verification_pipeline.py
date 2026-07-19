from pathlib import Path

import numpy as np

from src.contracts import validate_verification_sample
from src.generation.scenario_bank import load_scenario_bank_config
from src.generation.verification_gt import load_verification_gt_config
from src.generation.verification_pipeline import (
    build_verification_toy_input,
    generate_verification_group,
)
from src.planning.verification_actions import load_verification_actions
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_toy_group_runs_same_six_action_geometry_value_and_sample_path():
    config = load_config(ROOT / "configs/base.yaml")
    source, toy_config = build_verification_toy_input(config, group_index=3)
    action_library = load_verification_actions(
        ROOT / "configs/verification_actions.yaml"
    )
    gt_config = load_verification_gt_config(ROOT / "configs/verification_gt.yaml")
    scenario_config = load_scenario_bank_config(
        ROOT / "configs/verification_gt.yaml"
    )

    result = generate_verification_group(
        source,
        base_config=toy_config,
        action_library=action_library,
        gt_config=gt_config,
        scenario_config=scenario_config,
        bank_size=8,
        posterior_mode="exact",
        posterior_temperature=None,
        seed=41,
        max_replan_candidates=4,
    )

    assert len(result.samples) == 6
    assert result.bank_size == 8
    assert result.posterior_mode == "exact"
    assert result.infeasible_action_ids == ()
    assert len({item.metadata["ranking_group_id"] for item in result.samples}) == 1
    assert all(np.isfinite(item.value_target) for item in result.samples)
    assert all(
        item.value_target == item.br_before - item.post_risk
        for item in result.samples
    )
    for sample in result.samples:
        validate_verification_sample(sample, source.grid)
        assert sample.metadata["provenance"]["source_mode"] == "toy"
