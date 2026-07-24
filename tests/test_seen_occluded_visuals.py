"""Deterministic visual contracts for real seen-then-occluded audits."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageSequence

from src.contracts import (
    BaseState,
    LocalTrajectory,
    OracleContext,
    build_grid_spec,
)
from src.evaluation.seen_occluded_visuals import (
    PAIRED_PANEL_ORDER,
    VisualAuditBundle,
    VisualVariant,
    build_replay_frames,
    render_visual_artifacts,
)
from src.geometry import RectangleFootprint, rasterize_footprint


def _visual_bundle() -> VisualAuditBundle:
    config = {
        "bev": {
            "size": 41,
            "resolution_m": 0.2,
            "history_steps": 8,
            "future_steps": 15,
        }
    }
    grid = build_grid_spec(config)
    robot_history = np.zeros((8, 3), dtype=np.float32)
    context_id = "recording::visible-context"
    context_history = np.column_stack(
        (
            np.linspace(-0.8, -0.45, 8, dtype=np.float32),
            np.linspace(1.0, 1.2, 8, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
        )
    ).astype(np.float32)
    context_future = np.column_stack(
        (
            np.linspace(-0.35, 0.7, 15, dtype=np.float32),
            np.linspace(1.25, 1.6, 15, dtype=np.float32),
            np.zeros(15, dtype=np.float32),
        )
    ).astype(np.float32)
    context_spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.25},
    }
    base_state = BaseState(
        state_id="train-base-visual",
        split="train",
        recording_id="recording",
        dynamic_object_ids=(context_id,),
        timestamp=12.0,
        robot_history=robot_history,
        robot_state=np.asarray([0.5, 0.0], dtype=np.float32),
        visible_dynamic_object_history={context_id: context_history.copy()},
        visible_dynamic_object_specs={context_id: dict(context_spec)},
        static_map_local=np.zeros((41, 41), dtype=np.float32),
        metadata={"coordinate_frame": "robot_current"},
    )
    oracle_context = OracleContext(
        base_state_id=base_state.state_id,
        dynamic_object_history={context_id: context_history},
        dynamic_object_future={context_id: context_future},
        dynamic_object_specs={context_id: context_spec},
        metadata={"future_dt_s": 0.2},
    )
    trajectory_poses = np.column_stack(
        (
            np.arange(1, 16, dtype=np.float32) * np.float32(0.2),
            np.zeros(15, dtype=np.float32),
            np.zeros(15, dtype=np.float32),
        )
    ).astype(np.float32)
    maps = np.zeros((41, 41), dtype=np.float32)
    trajectory = LocalTrajectory(
        trajectory_id="forward-straight",
        poses=trajectory_poses,
        controls=np.tile(
            np.asarray([0.5, 0.0], dtype=np.float32), (15, 1)
        ),
        swept_mask=maps.copy(),
        tta_map=np.full((41, 41), -1.0, dtype=np.float32),
        braking_map=maps.copy(),
        centerline_map=maps.copy(),
        task_cost=0.0,
        metadata={"dt_s": 0.2},
    )
    target_history = np.column_stack(
        (
            np.asarray(
                [0.7, 0.8, 0.9, 1.8, 1.9, 2.0, 2.1, 2.2],
                dtype=np.float32,
            ),
            np.asarray(
                [1.2, 1.0, 0.8, 0.1, 0.08, 0.05, 0.02, 0.0],
                dtype=np.float32,
            ),
            np.zeros(8, dtype=np.float32),
        )
    ).astype(np.float32)
    target_future = np.column_stack(
        (
            np.linspace(2.0, 0.5, 15, dtype=np.float32),
            np.zeros(15, dtype=np.float32),
            np.full(15, np.pi, dtype=np.float32),
        )
    ).astype(np.float32)
    visibility_history = np.asarray(
        [True, True, True, False, False, False, False, False], dtype=np.bool_
    )
    variants = []
    clearances = {
        "collision": -0.2,
        "near_miss": 0.2,
        "temporal_safe": 0.1,
        "spatial_safe": 0.7,
        "irrelevant_hidden": 1.6,
    }
    for index, kind in enumerate(PAIRED_PANEL_ORDER):
        if kind == "empty_blind_spot":
            variants.append(
                VisualVariant(
                    kind=kind,
                    target_history=None,
                    target_future=None,
                    visibility_history=None,
                    min_clearance_m=None,
                    time_to_min_clearance_s=None,
                    temporal_offset_s=None,
                )
            )
            continue
        shifted_history = target_history.copy()
        shifted_future = target_future.copy()
        if kind in {"near_miss", "spatial_safe", "irrelevant_hidden"}:
            shifted_history[:, 1] += np.float32(index * 0.15)
            shifted_future[:, 1] += np.float32(index * 0.15)
        variants.append(
            VisualVariant(
                kind=kind,
                target_history=shifted_history,
                target_future=shifted_future,
                visibility_history=visibility_history.copy(),
                min_clearance_m=clearances[kind],
                time_to_min_clearance_s=1.4,
                temporal_offset_s=0.8 if kind == "temporal_safe" else None,
            )
        )
    occluder_pose = np.asarray([1.2, 0.0, 0.0], dtype=np.float32)
    static_occupancy = rasterize_footprint(
        RectangleFootprint(length_m=0.6, width_m=1.0),
        occluder_pose,
        grid,
    )
    return VisualAuditBundle(
        event_id="event-visual",
        base_state=base_state,
        oracle_context=oracle_context,
        trajectory=trajectory,
        static_occupancy=static_occupancy,
        occluders=(
            {
                "occluder_id": "occluder-visual",
                "type": "shelf",
                "pose": occluder_pose.tolist(),
                "length_m": 0.6,
                "width_m": 1.0,
            },
        ),
        variants=tuple(variants),
        grid=grid,
    )


def test_build_replay_frames_uses_real_history_and_frozen_current_future() -> None:
    bundle = _visual_bundle()
    frames = build_replay_frames(bundle)
    context_id = "recording::visible-context"

    assert [frame.time_s for frame in frames[:8]] == pytest.approx(
        [-1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0]
    )
    assert [frame.time_s for frame in frames[8:]] == pytest.approx(
        [0.2 * index for index in range(1, 16)]
    )
    assert len(frames) == 23
    assert [frame.target_visible for frame in frames[:8]] == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert all(frame.phase == "observed_history" for frame in frames[:8])
    for index, frame in enumerate(frames[:8]):
        assert frame.context_object_ids == (context_id,)
        np.testing.assert_array_equal(
            frame.context_poses[0],
            bundle.oracle_context.dynamic_object_history[context_id][index],
        )
        assert frame.context_poses.dtype == np.float32
        assert frame.context_visible.dtype == np.bool_
        assert bool(frame.context_visible[0]) is True
    for index, frame in enumerate(frames[8:]):
        np.testing.assert_array_equal(
            frame.visibility_mask, frames[7].visibility_mask
        )
        assert frame.phase == "oracle_replay"
        assert frame.target_visible is None
        assert frame.context_object_ids == (context_id,)
        np.testing.assert_array_equal(
            frame.context_poses[0],
            bundle.oracle_context.dynamic_object_future[context_id][index],
        )
        assert frame.context_poses.dtype == np.float32
        assert frame.context_visible is None
    assert not np.array_equal(frames[8].context_poses, frames[-1].context_poses)


def test_visual_bundle_rejects_noncanonical_panel_order() -> None:
    bundle = _visual_bundle()
    changed = replace(bundle, variants=tuple(reversed(bundle.variants)))

    with pytest.raises(ValueError, match="panel order"):
        build_replay_frames(changed)


def test_visual_bundle_rejects_history_without_seen_then_occluded_pattern() -> None:
    bundle = _visual_bundle()
    collision = replace(
        bundle.variants[0],
        visibility_history=np.zeros(8, dtype=np.bool_),
    )
    changed = replace(bundle, variants=(collision, *bundle.variants[1:]))

    with pytest.raises(ValueError, match="seen_then_occluded"):
        build_replay_frames(changed)


def test_render_visual_artifacts_have_fixed_layout_and_nonblank_pixels(
    tmp_path,
) -> None:
    result = render_visual_artifacts(_visual_bundle(), tmp_path)

    with Image.open(result.event_replay_path) as gif:
        assert gif.n_frames == 23
        assert gif.size == (1200, 900)
        extrema = [
            frame.convert("RGB").getextrema()
            for frame in ImageSequence.Iterator(gif)
        ]
        assert any(
            any(low != high for low, high in channels)
            for channels in extrema
        )
    with Image.open(result.paired_events_path) as png:
        assert png.size == (2100, 1200)
        assert any(
            low != high for low, high in png.convert("RGB").getextrema()
        )
    assert result.event_replay_metadata["frame_count"] == 23
    assert result.event_replay_metadata["context_object_count"] == 1
    assert result.event_replay_metadata["moving_context_object_count"] == 1
    assert result.event_replay_metadata["context_future_rendering"] == (
        "oracle_only_animated"
    )
    assert result.paired_events_metadata["panel_order"] == list(
        PAIRED_PANEL_ORDER
    )
    assert result.paired_events_metadata["empty_target_removed"] is True
