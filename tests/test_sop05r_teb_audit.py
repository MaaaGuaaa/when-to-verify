from dataclasses import replace
import json

import pytest

from src.evaluation.sop05r_teb_audit import (
    audit_sop05r_teb_collection,
    publish_sop05r_teb_visual_audit,
)
from src.planning.verification_actions import load_verification_actions
from tests.test_anchored_human_placement import _m4_inputs
from tests.test_sop05r_teb_sop06_handoff import _strict_collection


def test_teb_audit_recomputes_long40_metrics_from_strict_collection(tmp_path) -> None:
    collection = _strict_collection(tmp_path)

    metrics = audit_sop05r_teb_collection(collection)

    assert metrics.event_count == 1
    assert metrics.history_visible_frames + metrics.history_occluded_frames == 8
    assert 0 <= metrics.events_with_visible_and_occluded_history <= 1
    assert 1.2 <= metrics.collision_time_min_s <= metrics.collision_time_max_s <= 6.4
    assert metrics.occlusion_witness_count == metrics.recomputed_witness_count == 1
    assert sum(metrics.shape_counts.values()) >= 1


def test_teb_audit_rejects_incomplete_collection_before_metrics(tmp_path) -> None:
    collection = _strict_collection(tmp_path)

    with pytest.raises(ValueError, match="complete M7"):
        audit_sop05r_teb_collection(replace(collection, complete=False))


def test_teb_visual_audit_publishes_every_selected_strict_event(tmp_path) -> None:
    collection = _strict_collection(tmp_path)
    teb_config = _m4_inputs()[3]
    output = tmp_path / "visual-audit"

    result = publish_sop05r_teb_visual_audit(
        collection,
        output_dir=output,
        sample_count=1,
        seed=31,
        teb_config=teb_config,
        action_library=load_verification_actions("configs/verification_actions.yaml"),
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="ascii"))
    assert result.selected_event_ids == tuple(manifest["selected_event_ids"])
    assert (output / "COMPLETE.json").is_file()
    assert len(list((output / "samples").glob("*.png"))) == 1
    with pytest.raises(FileExistsError):
        publish_sop05r_teb_visual_audit(
            collection,
            output_dir=output,
            sample_count=1,
            seed=31,
            teb_config=teb_config,
            action_library=load_verification_actions(
                "configs/verification_actions.yaml"
            ),
        )
