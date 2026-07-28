from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.evaluation.sop05_ab_visual_audit import (
    IncompleteContextError,
    _context_long40,
    build_sop05_ab_visual_bundle,
    publish_sop05_ab_visual_audit,
)
from src.evaluation.sop05r_teb_visuals import render_sop05r_teb_visual_bundle
from src.generation.sop05_seen_prior import load_seen_prior_config
from src.generation.sop05_unseen_prior import normalize_unseen_prior_config
from src.planning.verification_actions import load_verification_actions
from tests.test_sop05r_teb_event_sampler import _mother_fixture
from tests.test_sop05r_teb_sop06_handoff import _strict_collection


ROOT = Path(__file__).resolve().parents[1]


def test_scenario_visual_bundle_uses_synthesized_target_without_mother_collision(
    tmp_path: Path,
) -> None:
    collection = _strict_collection(tmp_path)
    event = collection.events[0]
    target_long40 = np.vstack((event.target.history_poses, event.target.future_poses))

    bundle = build_sop05_ab_visual_bundle(
        collection,
        event_id=event.generated_event_id,
        target_long40=target_long40,
        target_visibility_history=event.target_visibility_history,
        teb_config=_mother_fixture()[3],
        action_library=load_verification_actions(
            ROOT / "configs/verification_actions.yaml"
        ),
    )

    artifact = render_sop05r_teb_visual_bundle(bundle, tmp_path / "scenario.png")

    assert bundle.collision_point_xy is None
    assert bundle.collision_time_s is None
    assert bundle.witness_sample_index is None
    assert bundle.witness_occluder_id is None
    assert bundle.verification_traces == ()
    assert artifact.metadata["direction_annotations"] == ["robot", "pedestrian"]


def test_ab_visual_audit_publishes_one_scenario_for_each_regime(tmp_path: Path) -> None:
    collection = _strict_collection(tmp_path)
    teb_config = _mother_fixture()[3]
    unseen_config = normalize_unseen_prior_config(
        yaml.safe_load((ROOT / "configs/sop05_unseen_prior.yaml").read_text()),
        base_config=dict(collection.manifest["base_config"]),
    )

    result = publish_sop05_ab_visual_audit(
        collection,
        output_dir=tmp_path / "audit",
        sample_count_per_regime=1,
        selection_seed=20260727,
        unseen_config=unseen_config,
        seen_config=load_seen_prior_config(ROOT / "configs/sop05_seen_prior.yaml"),
        teb_config=teb_config,
        action_library=load_verification_actions(
            ROOT / "configs/verification_actions.yaml"
        ),
    )

    assert len(result.regime_a_event_ids) == len(result.regime_b_event_ids) == 1
    assert (result.output_dir / "COMPLETE.json").is_file()
    assert len(list((result.output_dir / "regime-a").glob("*.png"))) == 1
    assert len(list((result.output_dir / "regime-b").glob("*.png"))) == 1


def test_context_adapter_marks_missing_history_as_a_skippable_mother(
    tmp_path: Path,
) -> None:
    collection = _strict_collection(tmp_path)
    event = collection.events[0]
    context_id = "missing-history-context"
    incomplete_event = replace(
        event,
        world=replace(
            event.world,
            dynamic_object_trajectories={
                **event.world.dynamic_object_trajectories,
                context_id: np.zeros((32, 3), dtype=np.float32),
            },
            dynamic_object_specs={
                **event.world.dynamic_object_specs,
                context_id: dict(event.target.footprint_spec),
            },
        ),
    )
    state = next(iter(collection.decision_states.values()))

    with pytest.raises(IncompleteContextError, match="complete Long40"):
        _context_long40(incomplete_event, state)
