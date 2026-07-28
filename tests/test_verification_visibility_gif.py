from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageSequence

from src.contracts import GridSpec
from src.evaluation.verification_visibility_gif import (
    VERIFICATION_VISIBILITY_GIF_VERSION,
    VerificationVisibilityGifCase,
    build_verification_visibility_case,
    render_verification_visibility_gif,
)
from src.geometry import CircleFootprint
from src.generation.counterfactual_verify import (
    CounterfactualObservation,
    CounterfactualObservationTrace,
)
from src.planning.verification_actions import (
    ActionTrace,
    VerificationAction,
)


def _case() -> VerificationVisibilityGifCase:
    grid = GridSpec(
        height=24,
        width=24,
        history_steps=8,
        future_steps=32,
        resolution_m=0.1,
    )
    static = np.zeros((grid.height, grid.width), dtype=bool)
    static[12, 8:16] = True
    poses = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.2],
            [0.2, 0.03, 0.4],
        ],
        dtype=np.float32,
    )
    times = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    action_trace = ActionTrace(
        poses=poses,
        times_s=times,
        linear_velocities_mps=np.asarray([0.5, 0.5, 0.0], dtype=np.float64),
        angular_velocities_radps=np.asarray([1.0, 1.0, 0.0], dtype=np.float64),
    )
    baseline = np.zeros_like(static)
    baseline[5:19, 4:12] = True
    visible_masks = (
        baseline,
        baseline | np.pad(
            np.ones((6, 4), dtype=bool),
            ((8, 10), (12, 8)),
        ),
        baseline | np.pad(
            np.ones((8, 7), dtype=bool),
            ((7, 9), (12, 5)),
        ),
    )
    frames = []
    for index, visible in enumerate(visible_masks):
        dynamic = np.zeros_like(static)
        dynamic[10 + index, 17] = True
        frames.append(
            CounterfactualObservation(
                visible_mask=visible,
                visible_occupied_mask=visible & (static | dynamic),
                visible_dynamic_occupancy=visible & dynamic,
                newly_visible_mask=visible & ~baseline,
                updated_age_map=np.zeros_like(static, dtype=np.float32),
            )
        )
    aggregate = CounterfactualObservation(
        visible_mask=np.logical_or.reduce(visible_masks),
        visible_occupied_mask=np.logical_or.reduce(
            [frame.visible_occupied_mask for frame in frames]
        ),
        visible_dynamic_occupancy=np.logical_or.reduce(
            [frame.visible_dynamic_occupancy for frame in frames]
        ),
        newly_visible_mask=np.logical_or.reduce(
            [frame.newly_visible_mask for frame in frames]
        ),
        updated_age_map=np.zeros_like(static, dtype=np.float32),
    )
    observation_trace = CounterfactualObservationTrace(
        aggregate=aggregate,
        frames=tuple(frames),
        times_s=times,
    )
    return VerificationVisibilityGifCase(
        sample_id="sample-001",
        action_id="arc_left_30",
        grid=grid,
        static_occupancy=static,
        action_trace=action_trace,
        observation_trace=observation_trace,
    )


def test_renders_time_aligned_visibility_progression(tmp_path: Path) -> None:
    artifact = render_verification_visibility_gif(
        _case(),
        tmp_path / "audit",
        frame_duration_ms=120,
    )

    assert artifact.gif_path == tmp_path / "audit" / "visibility_progress.gif"
    assert artifact.manifest_path == tmp_path / "audit" / "manifest.json"
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == VERIFICATION_VISIBILITY_GIF_VERSION
    assert manifest["sample_id"] == "sample-001"
    assert manifest["action_id"] == "arc_left_30"
    assert manifest["frame_count"] == 3
    assert manifest["times_s"] == [0.0, 0.1, 0.2]
    assert [
        item["newly_visible_cell_count"] for item in manifest["frame_metrics"]
    ] == [0, 24, 56]

    with Image.open(artifact.gif_path) as image:
        assert image.format == "GIF"
        assert image.size == (800, 800)
        assert image.n_frames == 3
        frames = [
            np.asarray(frame.convert("RGB"))
            for frame in ImageSequence.Iterator(image)
        ]
    assert any(not np.array_equal(frames[0], frame) for frame in frames[1:])


def test_refuses_to_overwrite_visibility_audit(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    render_verification_visibility_gif(_case(), output)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        render_verification_visibility_gif(_case(), output)


def test_collapses_sub_centisecond_display_frames(tmp_path: Path) -> None:
    original = _case()
    trace = ActionTrace(
        poses=np.concatenate(
            (original.action_trace.poses[:1], original.action_trace.poses),
            axis=0,
        ).astype(np.float32),
        times_s=np.asarray([0.0, 1e-5, 0.1, 0.2], dtype=np.float64),
        linear_velocities_mps=np.asarray(
            [0.5, 0.5, 0.5, 0.0],
            dtype=np.float64,
        ),
        angular_velocities_radps=np.asarray(
            [1.0, 1.0, 1.0, 0.0],
            dtype=np.float64,
        ),
    )
    frames = (
        original.observation_trace.frames[0],
        *original.observation_trace.frames,
    )
    observation = CounterfactualObservationTrace(
        aggregate=original.observation_trace.aggregate,
        frames=frames,
        times_s=trace.times_s,
    )
    case = VerificationVisibilityGifCase(
        sample_id=original.sample_id,
        action_id=original.action_id,
        grid=original.grid,
        static_occupancy=original.static_occupancy,
        action_trace=trace,
        observation_trace=observation,
    )

    artifact = render_verification_visibility_gif(case, tmp_path / "audit")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert manifest["source_trace_frame_count"] == 4
    assert manifest["rendered_frame_indices"] == [0, 2, 3]
    assert manifest["frame_count"] == 3
    with Image.open(artifact.gif_path) as image:
        assert image.n_frames == 3


def test_builds_time_aligned_case_from_action_and_world_state() -> None:
    grid = GridSpec(
        height=80,
        width=80,
        history_steps=8,
        future_steps=32,
        resolution_m=0.1,
    )
    shape = (grid.height, grid.width)
    action = VerificationAction(
        action_id="arc_left_30",
        duration_s=0.8,
        delta_forward_m=0.45,
        delta_yaw_rad=float(np.deg2rad(30.0)),
    )

    case = build_verification_visibility_case(
        sample_id="sample-real-001",
        action=action,
        grid=grid,
        robot_pose=np.zeros(3, dtype=np.float32),
        robot_state=np.zeros(2, dtype=np.float32),
        robot_footprint=CircleFootprint(0.2),
        static_occupancy=np.zeros(shape, dtype=np.float32),
        dynamic_current_poses={},
        dynamic_future_poses={},
        dynamic_specs={},
        current_visible_mask=np.zeros(shape, dtype=bool),
        current_age_map=np.ones(shape, dtype=np.float32),
        future_dt_s=0.2,
        age_max_s=3.0,
        fov_rad=float(2.0 * np.pi),
        max_range_m=8.0,
        braking_deceleration_mps2=1.0,
        angular_deceleration_radps2=1.6,
    )

    assert case.sample_id == "sample-real-001"
    assert case.action_id == "arc_left_30"
    assert len(case.observation_trace.frames) == case.action_trace.poses.shape[0]
    np.testing.assert_array_equal(
        case.observation_trace.times_s,
        case.action_trace.times_s,
    )
    assert case.action_trace.times_s[-1] == pytest.approx(0.8)
    assert case.action_trace.poses[-1, 2] == pytest.approx(
        np.deg2rad(30.0),
        abs=1e-6,
    )
