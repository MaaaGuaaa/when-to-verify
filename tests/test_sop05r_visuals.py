from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation.seen_occluded_visuals import PAIRED_PANEL_ORDER
from src.evaluation.sop05r_visuals import (
    Sop05rVisualRequest,
    build_sop05r_visual_bundle,
    render_sop05r_visual_artifacts,
)
from src.generation.paired_variants import PairedEventGroup, PairedVariant
from src.generation.sop05r_event_sampler import evaluate_obstacle_first_template
from src.planning.verification_actions import load_verification_actions
from tests.test_sop05r_event_sampler import _fixture as event_fixture


ROOT = Path(__file__).resolve().parents[1]


def _visual_request() -> Sop05rVisualRequest:
    base_config, config, base_state, context, template, _ = event_fixture()
    evaluation = evaluate_obstacle_first_template(
        template=template,
        base_state=base_state,
        oracle_context=context,
        base_config=base_config,
        config=config,
        seed=31,
    )
    assert evaluation.mother is not None
    mother = evaluation.mother
    variants = []
    clearances = {
        "collision": -0.1,
        "near_miss": 0.2,
        "temporal_safe": 0.4,
        "spatial_safe": 0.7,
        "irrelevant_hidden": 1.7,
    }
    for index, kind in enumerate(PAIRED_PANEL_ORDER):
        if kind == "empty_blind_spot":
            variants.append(
                PairedVariant(
                    variant_kind=kind,
                    world=mother.event.world,
                    target=None,
                    target_visibility_history=None,
                    visibility_sequence=None,
                    clearance_sequence_m=None,
                    min_clearance_m=None,
                    time_to_min_clearance_s=None,
                )
            )
            continue
        history = mother.event.target.history_poses.copy()
        future = mother.event.target.future_poses.copy()
        if kind in {"near_miss", "spatial_safe", "irrelevant_hidden"}:
            history[:, 1] += np.float32(0.12 * index)
            future[:, 1] += np.float32(0.12 * index)
        target = replace(
            mother.event.target,
            history_poses=history,
            current_pose=history[-1].copy(),
            future_poses=future,
        )
        variants.append(
            PairedVariant(
                variant_kind=kind,
                world=mother.event.world,
                target=target,
                target_visibility_history=mother.event.target_visibility_history.copy(),
                visibility_sequence=mother.event.visibility_sequence.copy(),
                clearance_sequence_m=np.full(15, clearances[kind], dtype=np.float64),
                min_clearance_m=clearances[kind],
                time_to_min_clearance_s=1.2,
                temporal_offset_s=0.8 if kind == "temporal_safe" else None,
            )
        )
    group = PairedEventGroup(
        pair_group_id="pair-sop05r-visual",
        variants=tuple(variants),
        coverage_mask=(True,) * 6,
        missing_variant_reasons={},
        is_complete=True,
        eligible_for_strict_evaluation=True,
        paired_config_digest="a" * 64,
    )
    return Sop05rVisualRequest(
        event=mother.event,
        trajectory_record=mother.trajectory_record,
        base_state=base_state,
        oracle_context=context,
        pair_group=group,
        base_config=base_config,
        action_library=load_verification_actions(
            ROOT / "configs/verification_actions.yaml"
        ),
        verification_action_id="arc_left_45",
    )


def test_visual_bundle_contains_every_required_sop05r_layer() -> None:
    request = _visual_request()

    bundle = build_sop05r_visual_bundle(request)

    assert bundle.source_static_occupancy is not None
    assert len(bundle.planner_routes_world) == len(
        request.trajectory_record.routes
    )
    assert bundle.planner_slot_ids == tuple(
        route.slot_id for route in request.trajectory_record.routes
    )
    assert bundle.nominal_trajectory_id == (
        request.trajectory_record.nominal_trajectory_id
    )
    assert bundle.alternative_trajectory_ids == (
        request.trajectory_record.alternative_trajectory_ids
    )
    np.testing.assert_array_equal(
        bundle.shared_goal_world_pose,
        request.trajectory_record.shared_goal_world_pose,
    )
    np.testing.assert_allclose(
        bundle.conflict_point,
        request.event.world.metadata["conflict_point"],
    )
    assert bundle.inflated_obstacle_margin_m > 0.0
    assert bundle.verification_action_id == "arc_left_45"
    assert bundle.verification_trace.shape[0] > 2


def test_sop05r_visuals_are_fixed_nonblank_and_deterministic(tmp_path: Path) -> None:
    request = _visual_request()

    first = render_sop05r_visual_artifacts(request, tmp_path / "first")
    second = render_sop05r_visual_artifacts(request, tmp_path / "second")

    with Image.open(first.event_replay_path) as replay:
        assert replay.size == (1200, 900)
        assert replay.n_frames == 23
        replay.seek(10)
        assert np.asarray(replay.convert("RGB")).std() > 5.0
    with Image.open(first.paired_events_path) as paired:
        assert paired.size == (2100, 1200)
        assert np.asarray(paired.convert("RGB")).std() > 5.0
    for metadata in (
        first.event_replay_metadata,
        first.paired_events_metadata,
    ):
        assert metadata["source_static_map_rendered"] is True
        assert metadata["generated_obstacle_rendered"] is True
        assert metadata["inflated_obstacle_boundary_rendered"] is True
        assert metadata["shared_goal_rendered"] is True
        assert metadata["conflict_point_rendered"] is True
        assert metadata["planner_route_count"] == len(
            request.trajectory_record.routes
        )
        assert metadata["verification_action_id"] == "arc_left_45"
    assert first.event_replay_metadata["sha256"] == (
        second.event_replay_metadata["sha256"]
    )
    assert first.paired_events_metadata["sha256"] == (
        second.paired_events_metadata["sha256"]
    )
