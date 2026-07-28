from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import src.generation.obstacle_first_templates as template_module
from src.contracts import BaseState, OracleContext, build_grid_spec
from src.datasets.snippet_library import MotionSnippet, SnippetLibrary
from src.generation.obstacle_first_templates import (
    Sop05rTemplateError,
    canonical_base_state_digest,
    iter_obstacle_target_templates,
    resample_sop05r_snippet,
)
from src.generation.sop05r_contracts import load_sop05r_config
from src.geometry import wrap_angle
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _snippet(*, split: str = "train") -> MotionSnippet:
    times = np.arange(23, dtype=np.float32) * np.float32(0.2)
    positions = np.column_stack(
        (0.8 * times, 0.04 * np.sin(1.5 * times))
    ).astype(np.float32)
    velocities = np.gradient(positions, 0.2, axis=0).astype(np.float32)
    headings = np.arctan2(velocities[:, 1], velocities[:, 0]).astype(np.float32)
    return MotionSnippet(
        snippet_id=f"{split}-human-obstacle-first",
        split=split,
        source_recording_id=f"{split}-recording",
        source_session_id=f"{split}-session",
        source_object_id=f"{split}-recording::human-1",
        object_type="human",
        footprint={"kind": "circle", "radius_m": 0.3},
        start_timestamp=1.0,
        positions=positions,
        velocities=velocities,
        headings=headings,
        duration_s=4.4,
        mean_speed_mps=float(np.linalg.norm(velocities, axis=1).mean()),
        max_acceleration_mps2=0.1,
        mean_abs_curvature_per_m=0.05,
        provenance={"fixture": "obstacle-first"},
    )


def _library(*, split: str = "train") -> dict[str, SnippetLibrary]:
    snippet = _snippet(split=split)
    return {
        "human": SnippetLibrary(
            object_type="human",
            snippets=(snippet,),
            summary={"split": split, "accepted_count": 1},
            split_provenance={"split": split},
        )
    }


def _many_snippet_library(
    *, split: str = "train", count: int = 16
) -> dict[str, SnippetLibrary]:
    snippets = tuple(
        replace(
            _snippet(split=split),
            snippet_id=f"{split}-human-obstacle-first-{index:02d}",
            source_object_id=f"{split}-recording::human-{index:02d}",
        )
        for index in range(count)
    )
    return {
        "human": SnippetLibrary(
            object_type="human",
            snippets=snippets,
            summary={"split": split, "accepted_count": count},
            split_provenance={"split": split},
        )
    }


def _reversal_snippet(index: int, *, split: str = "train") -> MotionSnippet:
    current_index = 7
    positions = np.zeros((23, 2), dtype=np.float32)
    positions[: current_index + 1, 0] = np.linspace(
        0.0, 0.4 + 0.01 * index, current_index + 1, dtype=np.float32
    )
    positions[current_index:, 0] = np.linspace(
        positions[current_index, 0],
        positions[current_index, 0] - 1.6 - 0.01 * index,
        23 - current_index,
        dtype=np.float32,
    )
    velocities = np.gradient(positions, 0.2, axis=0).astype(np.float32)
    headings = np.arctan2(velocities[:, 1], velocities[:, 0]).astype(np.float32)
    return replace(
        _snippet(split=split),
        snippet_id=f"{split}-reversal-{index:02d}",
        source_recording_id=f"{split}-turn-recording-{index:02d}",
        source_object_id=f"{split}-turn-recording-{index:02d}::human",
        positions=positions,
        velocities=velocities,
        headings=headings,
        mean_speed_mps=float(np.linalg.norm(velocities, axis=1).mean()),
        mean_abs_curvature_per_m=1.0,
    )


def _inputs():
    base_config = load_config()
    grid = build_grid_spec(base_config)
    base_state = BaseState(
        state_id="train-obstacle-first-base",
        split="train",
        recording_id="train-base-recording",
        dynamic_object_ids=(),
        timestamp=12.0,
        robot_history=np.zeros((grid.history_steps, 3), dtype=np.float32),
        robot_state=np.asarray([0.4, 0.0], dtype=np.float32),
        visible_dynamic_object_history={},
        visible_dynamic_object_specs={},
        static_map_local=np.zeros((grid.height, grid.width), dtype=np.float32),
        metadata={"fixture": "obstacle-first"},
    )
    context = OracleContext(
        base_state_id=base_state.state_id,
        dynamic_object_history={},
        dynamic_object_future={},
        dynamic_object_specs={},
        metadata={"future_dt_s": 0.2},
    )
    config = load_sop05r_config(
        ROOT / "configs" / "generator_obstacle_first_train.yaml"
    )
    return base_config, grid, base_state, context, config


def _schedule(*, base_state=None, context=None, base_config=None):
    defaults = _inputs()
    return tuple(
        iter_obstacle_target_templates(
            base_state=defaults[2] if base_state is None else base_state,
            oracle_context=defaults[3] if context is None else context,
            snippet_libraries=_library(),
            base_config=defaults[0] if base_config is None else base_config,
            config=defaults[4],
            seed=17,
        )
    )


def test_template_schedule_is_stable_and_does_not_mutate_base_state() -> None:
    _, _, base_state, _, config = _inputs()
    before = canonical_base_state_digest(base_state)

    first = _schedule(base_state=base_state)
    second = _schedule(base_state=base_state)

    assert len(first) == config.generation.max_templates_per_base
    assert [row.template_id for row in first] == [row.template_id for row in second]
    assert [row.rejection_reason for row in first] == [
        row.rejection_reason for row in second
    ]
    assert canonical_base_state_digest(base_state) == before
    assert base_state.dynamic_object_ids == ()
    assert np.count_nonzero(base_state.static_map_local) == 0


def test_accepted_template_uses_joint_se2_and_frozen_real_snippet() -> None:
    _, grid, base_state, _, config = _inputs()
    accepted = [row.template for row in _schedule() if row.template is not None]

    assert accepted
    template = accepted[0]
    assert template.obstacle.obstacle_type in {"wall", "shelf"}
    assert 1.0 <= template.obstacle.length_m <= 2.5
    assert 0.2 <= template.obstacle.width_m <= 0.6
    assert template.target.provenance["source_split"] == base_state.split
    assert template.target_time_scale in config.template.target_time_scales
    assert template.obstacle_mask.shape == (grid.height, grid.width)
    assert template.obstacle_mask.dtype == np.bool_
    assert np.any(template.obstacle_mask)
    assert not np.any(template.obstacle_mask & (base_state.static_map_local != 0))

    resampled = resample_sop05r_snippet(
        _snippet(), time_scale=template.target_time_scale
    )
    transform = template.provenance["joint_se2_transform"]
    rotation_angle = float(transform["rotation_rad"])
    rotation = np.asarray(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )
    translation = np.asarray(transform["translation_xy_m"], dtype=np.float64)
    expected_positions = resampled[:, :2].astype(np.float64) @ rotation.T + translation
    expected_headings = wrap_angle(resampled[:, 2].astype(np.float64) + rotation_angle)
    actual = np.vstack((template.target.history_poses, template.target.future_poses))
    np.testing.assert_allclose(actual[:, :2], expected_positions, atol=1e-6)
    np.testing.assert_allclose(actual[:, 2], expected_headings, atol=1e-6)

    canonical_obstacle_pose = np.asarray(
        transform["canonical_obstacle_pose"], dtype=np.float64
    )
    expected_obstacle_xy = canonical_obstacle_pose[:2] @ rotation.T + translation
    np.testing.assert_allclose(template.obstacle.pose[:2], expected_obstacle_xy)
    assert template.obstacle.pose[2] == pytest.approx(
        float(wrap_angle(canonical_obstacle_pose[2] + rotation_angle))
    )


def test_relative_layout_offsets_obstacle_to_opposite_task_sides() -> None:
    accepted = [row.template for row in _schedule() if row.template is not None]

    assert accepted
    observed_layouts = set()
    for template in accepted:
        task_direction = template.local_goal_world_pose[:2].astype(np.float64)
        task_direction /= np.linalg.norm(task_direction)
        task_normal = np.asarray(
            [-task_direction[1], task_direction[0]], dtype=np.float64
        )
        lateral_offset = float(np.dot(template.obstacle.pose[:2], task_normal))
        layout = template.provenance["relative_layout"]
        observed_layouts.add(layout)
        if layout == "target_side":
            assert lateral_offset > 0.0
        else:
            assert layout == "opposite_side"
            assert lateral_offset < 0.0
        assert abs(lateral_offset) == pytest.approx(
            0.5 * template.obstacle.length_m + 0.42
        )
    assert observed_layouts == {"target_side", "opposite_side"}


def test_template_snippet_window_is_base_seeded_across_the_full_library() -> None:
    base_config, _, base_state, context, config = _inputs()
    libraries = _many_snippet_library()

    def scheduled_source_ids(state_id: str, seed: int) -> set[str]:
        state = replace(base_state, state_id=state_id)
        aligned_context = replace(context, base_state_id=state_id)
        return {
            str(row.provenance["source_snippet_id"])
            for row in iter_obstacle_target_templates(
                base_state=state,
                oracle_context=aligned_context,
                snippet_libraries=libraries,
                base_config=base_config,
                config=config,
                seed=seed,
            )
        }

    first = scheduled_source_ids("train-obstacle-first-base-a", 17)
    second = scheduled_source_ids("train-obstacle-first-base-b", 23)
    lexicographic_prefix = {
        snippet.snippet_id
        for snippet in sorted(
            libraries["human"].snippets,
            key=lambda snippet: snippet.snippet_id,
        )[: config.generation.max_target_snippets_per_template]
    }

    assert first != second
    assert (first | second) - lexicographic_prefix


def test_snippet_window_reserves_eight_of_ten_for_seen_motion_potential() -> None:
    _, _, base_state, _, config = _inputs()
    reversal = tuple(_reversal_snippet(index) for index in range(12))
    regular = _many_snippet_library(count=12)["human"].snippets
    library = SnippetLibrary(
        object_type="human",
        snippets=(*reversal, *regular),
        summary={"split": "train", "accepted_count": 24},
        split_provenance={"split": "train"},
    )

    selected = template_module._human_snippets(
        {"human": library},
        split="train",
        limit=config.generation.max_target_snippets_per_template,
        base_state_id=base_state.state_id,
        seed=17,
    )

    assert len(selected) == 10
    assert sum(snippet.snippet_id.startswith("train-reversal-") for snippet in selected) == 8
    assert len({snippet.source_object_id for snippet in selected}) == 10


def test_snippet_resampling_is_bounded_and_never_extrapolates() -> None:
    resampled = resample_sop05r_snippet(_snippet(), time_scale=0.9)

    assert resampled.shape == (23, 3)
    assert resampled.dtype == np.float32
    assert np.isfinite(resampled).all()
    with pytest.raises(Sop05rTemplateError, match="source_extrapolation_required"):
        resample_sop05r_snippet(_snippet(), time_scale=1.1)


def test_template_schedule_rejects_cross_split_snippets() -> None:
    base_config, _, base_state, context, config = _inputs()

    with pytest.raises(Sop05rTemplateError, match="split"):
        tuple(
            iter_obstacle_target_templates(
                base_state=base_state,
                oracle_context=context,
                snippet_libraries=_library(split="test"),
                base_config=base_config,
                config=config,
                seed=17,
            )
        )


def test_exact_overlap_gates_reject_source_static_robot_and_context() -> None:
    base_config, grid, base_state, context, _ = _inputs()
    baseline = _schedule()
    accepted_row = next(row for row in baseline if row.template is not None)
    accepted = accepted_row.template

    blocked_static = accepted.obstacle_mask.astype(np.float32)
    static_state = replace(base_state, static_map_local=blocked_static)
    static_match = next(
        row for row in _schedule(base_state=static_state) if row.template_id == accepted_row.template_id
    )
    assert static_match.rejection_reason == "source_static_overlap"

    robot_history = np.tile(accepted.obstacle.pose, (grid.history_steps, 1)).astype(
        np.float32
    )
    robot_state = replace(base_state, robot_history=robot_history)
    robot_match = next(
        row for row in _schedule(base_state=robot_state) if row.template_id == accepted_row.template_id
    )
    assert robot_match.rejection_reason == "robot_history_overlap"

    context_id = "train-base-recording::context-1"
    context_history = np.tile(
        accepted.obstacle.pose, (grid.history_steps, 1)
    ).astype(np.float32)
    context_future = np.tile(
        accepted.obstacle.pose, (grid.future_steps, 1)
    ).astype(np.float32)
    context_spec = {
        "object_type": "carried_object",
        "footprint": {"kind": "rectangle", "length_m": 0.8, "width_m": 0.4},
    }
    context_state = replace(
        base_state,
        dynamic_object_ids=(context_id,),
        visible_dynamic_object_history={context_id: context_history.copy()},
        visible_dynamic_object_specs={context_id: deepcopy(context_spec)},
    )
    occupied_context = OracleContext(
        base_state_id=base_state.state_id,
        dynamic_object_history={context_id: context_history},
        dynamic_object_future={context_id: context_future},
        dynamic_object_specs={context_id: context_spec},
        metadata=context.metadata,
    )
    context_match = next(
        row
        for row in _schedule(base_state=context_state, context=occupied_context)
        if row.template_id == accepted_row.template_id
    )
    assert context_match.rejection_reason == "context_overlap"


def test_small_bev_rejects_templates_instead_of_clipping_geometry() -> None:
    base_config, _, base_state, context, config = _inputs()
    small_config = deepcopy(base_config)
    small_config["bev"]["size"] = 30
    small_grid = build_grid_spec(small_config)
    small_state = replace(
        base_state,
        static_map_local=np.zeros(
            (small_grid.height, small_grid.width), dtype=np.float32
        ),
    )

    evaluations = tuple(
        iter_obstacle_target_templates(
            base_state=small_state,
            oracle_context=context,
            snippet_libraries=_library(),
            base_config=small_config,
            config=config,
            seed=17,
        )
    )

    assert evaluations
    assert all(row.template is None for row in evaluations)
    assert {row.rejection_reason for row in evaluations} <= {
        "obstacle_out_of_bounds",
        "target_out_of_bounds",
        "source_extrapolation_required",
    }
