"""Tests for deterministic snippet reachability and conservative chord triage."""

from __future__ import annotations

import builtins
from dataclasses import replace

import numpy as np
import pytest

import src.generation.blind_reachability as blind_reachability
from src.contracts import GridSpec
from src.generation.blind_reachability import (
    BLIND_REACHABILITY_ALGORITHM_VERSION,
    REACHABLE_ARC_SCHEDULE_VERSION,
    ChordTriage,
    ReachabilityCandidate,
    ReachabilityIdentity,
    build_reachability_candidate,
    candidate_queries_mask,
    scheduled_crossing_directions,
    triage_chord,
)
from src.geometry import grid_to_world, world_to_grid


@pytest.fixture
def grid() -> GridSpec:
    return GridSpec(
        height=9,
        width=9,
        history_steps=8,
        future_steps=15,
        resolution_m=1.0,
    )


def _identity(**changes: object) -> ReachabilityIdentity:
    values: dict[str, object] = {
        "base_state_id": "base-001",
        "trajectory_id": "trajectory-002",
        "source_snippet_id": "snippet-003",
        "source_session_id": "session-004",
        "conflict_index": 5,
        "conflict_time_s": 1.4,
        "crossing_side": 1,
        "angle_offset_deg": 30.0,
    }
    values.update(changes)
    return ReachabilityIdentity(**values)


def _candidate(**changes: object) -> ReachabilityCandidate:
    values: dict[str, object] = {
        "conflict_point": np.array([2.0, 0.0], dtype=np.float64),
        "source_current_xy": np.array([0.0, 0.0], dtype=np.float64),
        "source_anchor_xy": np.array([4.0, 0.0], dtype=np.float64),
        "desired_crossing_direction": np.array([1.0, 0.0], dtype=np.float64),
        "identity": _identity(),
    }
    values.update(changes)
    return build_reachability_candidate(**values)


def test_versions_are_frozen() -> None:
    assert BLIND_REACHABILITY_ALGORITHM_VERSION == (
        "blind_reachability_quota_first_v3"
    )
    assert REACHABLE_ARC_SCHEDULE_VERSION == "reachable_arc_schedule_v1"


@pytest.mark.parametrize(
    ("desired_direction", "expected_rotation"),
    [
        (np.array([0.0, 1.0]), np.pi / 2.0),
        (np.array([0.0, -1.0]), -np.pi / 2.0),
        (np.array([1.0, 1.0]), np.pi / 4.0),
        (np.array([-1.0, 1.0]), 3.0 * np.pi / 4.0),
    ],
)
def test_candidate_uses_exact_se2_start_and_maps_anchor_to_conflict(
    desired_direction: np.ndarray, expected_rotation: float
) -> None:
    conflict = np.array([3.25, -0.75])
    source_current = np.array([-2.0, 4.0])
    source_anchor = np.array([0.0, 4.0])

    candidate = build_reachability_candidate(
        conflict_point=conflict,
        source_current_xy=source_current,
        source_anchor_xy=source_anchor,
        desired_crossing_direction=desired_direction,
        identity=_identity(crossing_side=-1 if desired_direction[1] < 0 else 1),
    )

    expected_delta = source_anchor - source_current
    expected_current = conflict - candidate.rotation_matrix @ expected_delta
    np.testing.assert_allclose(candidate.current_xy, expected_current, atol=1e-14)
    np.testing.assert_allclose(
        candidate.current_xy + candidate.rotation_matrix @ expected_delta,
        conflict,
        atol=1e-14,
    )
    assert candidate.rotation_rad == pytest.approx(expected_rotation)
    np.testing.assert_allclose(
        candidate.rotation_matrix @ (expected_delta / np.linalg.norm(expected_delta)),
        desired_direction / np.linalg.norm(desired_direction),
        atol=1e-14,
    )


def test_candidate_preserves_unwrapped_heading_difference_across_branch_cut() -> None:
    source_delta = np.array([-1.0, 0.1], dtype=np.float64)
    desired_direction = np.array([-1.0, -0.1], dtype=np.float64)
    expected_rotation = float(
        np.arctan2(desired_direction[1], desired_direction[0])
        - np.arctan2(source_delta[1], source_delta[0])
    )

    candidate = build_reachability_candidate(
        conflict_point=[0.0, 0.0],
        source_current_xy=[0.0, 0.0],
        source_anchor_xy=source_delta,
        desired_crossing_direction=desired_direction,
        identity=_identity(crossing_side=-1, angle_offset_deg=-10.0),
    )

    assert expected_rotation < -np.pi
    assert candidate.rotation_rad == expected_rotation
    np.testing.assert_allclose(
        candidate.current_xy + candidate.rotation_matrix @ source_delta,
        [0.0, 0.0],
        atol=1e-14,
    )


def test_candidate_builds_ordinary_oblique_direction_with_stable_bytes() -> None:
    desired_direction = np.random.default_rng(1).normal(size=(8, 2))[7]
    kwargs = {
        "conflict_point": [1.0, 1.0],
        "source_current_xy": [0.0, 0.0],
        "source_anchor_xy": [1.0, 0.0],
        "desired_crossing_direction": desired_direction,
        "identity": _identity(),
    }

    first = build_reachability_candidate(**kwargs)
    second = build_reachability_candidate(**kwargs)

    assert first.candidate_id == second.candidate_id
    np.testing.assert_array_equal(
        first.desired_crossing_direction, second.desired_crossing_direction
    )
    assert np.linalg.norm(first.desired_crossing_direction) == pytest.approx(1.0)


def test_crossing_schedule_has_frozen_order_count_and_unit_directions() -> None:
    directions = scheduled_crossing_directions(
        np.array([2.0, 0.0], dtype=np.float32),
        maximum_angle_deg=60.0,
        angle_step_deg=30.0,
    )
    expected_angles = np.deg2rad(np.array([-60.0, -30.0, 0.0, 30.0, 60.0]))
    expected = np.column_stack((np.cos(expected_angles), np.sin(expected_angles)))

    assert directions.shape == (5, 2)
    assert directions.dtype == np.float64
    np.testing.assert_allclose(directions, expected, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-15)
    assert directions[0, 1] < 0.0
    assert directions[-1, 1] > 0.0


def test_crossing_schedule_rotates_an_oblique_normal_on_both_sides() -> None:
    normal = np.array([1.0, 1.0], dtype=np.float64)
    directions = scheduled_crossing_directions(
        normal, maximum_angle_deg=45.0, angle_step_deg=45.0
    )

    np.testing.assert_allclose(directions[0], [1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(
        directions[1], normal / np.linalg.norm(normal), atol=1e-15
    )
    np.testing.assert_allclose(directions[2], [0.0, 1.0], atol=1e-15)


@pytest.mark.parametrize(
    ("maximum_angle_deg", "angle_step_deg"),
    [
        (50.0, 30.0),
        (-30.0, 10.0),
        (30.0, 0.0),
        (30.0, -10.0),
        (np.nan, 10.0),
        (30.0, np.inf),
        (True, 10.0),
    ],
)
def test_crossing_schedule_rejects_invalid_angle_parameters(
    maximum_angle_deg: object, angle_step_deg: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        scheduled_crossing_directions(
            [1.0, 0.0],
            maximum_angle_deg=maximum_angle_deg,
            angle_step_deg=angle_step_deg,
        )


@pytest.mark.parametrize(
    "normal",
    [
        [0.0, 0.0],
        [np.nan, 0.0],
        [0.0, np.inf],
        [1.0],
        [1.0, 0.0, 0.0],
        ["1", "0"],
        [True, False],
    ],
)
def test_crossing_schedule_rejects_bad_normals(normal: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        scheduled_crossing_directions(
            normal, maximum_angle_deg=30.0, angle_step_deg=10.0
        )


def test_two_exact_starts_in_one_cell_are_not_snapped_and_keep_distinct_identity(
    grid: GridSpec,
) -> None:
    first = build_reachability_candidate(
        conflict_point=[1.1, 0.1],
        source_current_xy=[0.0, 0.0],
        source_anchor_xy=[1.0, 0.0],
        desired_crossing_direction=[1.0, 0.0],
        identity=_identity(conflict_index=1, conflict_time_s=1.0),
    )
    second = build_reachability_candidate(
        conflict_point=[1.2, 0.1],
        source_current_xy=[0.0, 0.0],
        source_anchor_xy=[1.0, 0.0],
        desired_crossing_direction=[1.0, 0.0],
        identity=_identity(conflict_index=2, conflict_time_s=1.2),
    )
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    first_cell, second_cell = world_to_grid(
        np.stack((first.current_xy, second.current_xy)), grid
    )
    mask[tuple(first_cell)] = True

    np.testing.assert_allclose(first.current_xy, [0.1, 0.1], atol=1e-15)
    np.testing.assert_allclose(second.current_xy, [0.2, 0.1], atol=1e-15)
    np.testing.assert_array_equal(first_cell, second_cell)
    assert not np.array_equal(first.current_xy, grid_to_world(first_cell, grid))
    assert not np.array_equal(second.current_xy, grid_to_world(second_cell, grid))
    assert first.identity != second.identity
    assert first.candidate_id != second.candidate_id
    assert candidate_queries_mask(first, mask, grid)
    assert candidate_queries_mask(second, mask, grid)


def test_candidate_id_is_repeatable_and_does_not_call_builtin_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _candidate()

    def forbidden_hash(_value: object) -> int:
        raise AssertionError("built-in hash() must not construct candidate IDs")

    monkeypatch.setattr(builtins, "hash", forbidden_hash)
    second = _candidate()

    assert second.candidate_id == first.candidate_id


@pytest.mark.parametrize(
    "changed_identity",
    [
        {"base_state_id": "base-other"},
        {"trajectory_id": "trajectory-other"},
        {"source_snippet_id": "snippet-other"},
        {"source_session_id": "session-other"},
        {"conflict_index": 6},
        {"conflict_time_s": 1.6},
        {"crossing_side": -1},
        {"angle_offset_deg": -30.0},
    ],
)
def test_candidate_id_binds_every_identity_field(
    changed_identity: dict[str, object],
) -> None:
    baseline = _candidate()
    changed = _candidate(identity=_identity(**changed_identity))

    assert changed.candidate_id != baseline.candidate_id


@pytest.mark.parametrize("source_session_id", ["", "   ", None, 23])
def test_reachability_identity_rejects_invalid_source_session_id(
    source_session_id: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="source_session_id"):
        _identity(source_session_id=source_session_id)


def test_candidate_id_binds_exact_float64_geometry() -> None:
    baseline = _candidate()
    changed_conflict = np.array([2.0, np.nextafter(0.0, 1.0)], dtype=np.float64)
    changed = _candidate(conflict_point=changed_conflict)

    assert changed.candidate_id != baseline.candidate_id


@pytest.mark.parametrize(
    "version_name",
    ["BLIND_REACHABILITY_ALGORITHM_VERSION", "REACHABLE_ARC_SCHEDULE_VERSION"],
)
def test_candidate_id_binds_each_frozen_version(
    version_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _candidate()
    monkeypatch.setattr(blind_reachability, version_name, "changed-for-test")

    changed = _candidate()

    assert changed.candidate_id != baseline.candidate_id


def test_candidate_dataclass_rejects_forged_or_stale_candidate_id() -> None:
    candidate = _candidate()

    with pytest.raises(ValueError, match="candidate_id"):
        replace(candidate, candidate_id="reachability-forged")
    with pytest.raises(ValueError, match="candidate_id"):
        replace(candidate, identity=_identity(base_state_id="base-other"))


def test_mask_query_uses_boolean_cell_value_and_rejects_out_of_bounds(
    grid: GridSpec,
) -> None:
    candidate = _candidate()
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    cell = world_to_grid(candidate.current_xy, grid)

    assert candidate_queries_mask(candidate, mask, grid) is False
    mask[tuple(cell)] = True
    assert candidate_queries_mask(candidate, mask, grid) is True

    outside = _candidate(conflict_point=[20.0, 0.0])
    with pytest.raises(ValueError, match="outside"):
        candidate_queries_mask(outside, mask, grid)


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((8, 9), dtype=bool),
        np.zeros((9, 9, 1), dtype=bool),
        np.zeros((9, 9), dtype=np.float32),
        np.full((9, 9), np.nan, dtype=np.float64),
        np.full((9, 9), "false", dtype="U5"),
    ],
)
def test_mask_query_rejects_bad_shape_or_non_boolean_mask(
    mask: np.ndarray, grid: GridSpec
) -> None:
    with pytest.raises((TypeError, ValueError)):
        candidate_queries_mask(_candidate(), mask, grid)


def test_empty_chord_tube_is_certified_clear_and_keeps_stable_identity(
    grid: GridSpec,
) -> None:
    candidate = _candidate()
    occupancy = np.zeros((grid.height, grid.width), dtype=bool)

    result = triage_chord(
        candidate,
        obstacle_occupancy=occupancy,
        grid=grid,
        footprint_radius_m=0.2,
        chord_deviation_bound_m=0.1,
    )

    assert result == ChordTriage(
        outcome="certified_clear",
        chord_deviation_bound_m=0.1,
        candidate_id=candidate.candidate_id,
        identity=candidate.identity,
    )


def test_occupied_cell_overlapping_chord_tube_is_only_unresolved(
    grid: GridSpec,
) -> None:
    candidate = _candidate()
    occupancy = np.zeros((grid.height, grid.width), dtype=np.float32)
    chord_midpoint_cell = world_to_grid([0.0, 0.0], grid)
    occupancy[tuple(chord_midpoint_cell)] = 1.0

    result = triage_chord(
        candidate,
        obstacle_occupancy=occupancy,
        grid=grid,
        footprint_radius_m=0.2,
        chord_deviation_bound_m=0.1,
    )

    assert result.outcome == "unresolved"
    assert result.outcome != "collision"
    assert result.candidate_id == candidate.candidate_id
    assert result.identity == candidate.identity


def test_chord_tube_accounts_for_footprint_and_deviation_at_cell_boundary(
    grid: GridSpec,
) -> None:
    candidate = _candidate()
    occupancy = np.zeros((grid.height, grid.width), dtype=bool)
    occupancy[5, 4] = True  # Square spans y=[0.5, 1.5] above the x-axis chord.

    unresolved = triage_chord(
        candidate,
        obstacle_occupancy=occupancy,
        grid=grid,
        footprint_radius_m=0.25,
        chord_deviation_bound_m=0.25,
    )
    clear = triage_chord(
        candidate,
        obstacle_occupancy=occupancy,
        grid=grid,
        footprint_radius_m=0.2,
        chord_deviation_bound_m=0.2,
    )

    assert unresolved.outcome == "unresolved"
    # The occupied square starts at y=0.5, outside the radius-0.4 tube.
    assert clear.outcome == "certified_clear"


@pytest.mark.parametrize(
    "occupancy",
    [
        np.zeros((8, 9), dtype=bool),
        np.zeros((9, 9, 1), dtype=bool),
        np.full((9, 9), np.nan, dtype=np.float64),
        np.full((9, 9), "free", dtype="U4"),
        np.zeros((9, 9), dtype=np.complex128),
    ],
)
def test_chord_triage_rejects_bad_occupancy(
    occupancy: np.ndarray, grid: GridSpec
) -> None:
    with pytest.raises((TypeError, ValueError)):
        triage_chord(
            _candidate(),
            obstacle_occupancy=occupancy,
            grid=grid,
            footprint_radius_m=0.2,
            chord_deviation_bound_m=0.1,
        )


@pytest.mark.parametrize(
    ("footprint_radius_m", "chord_deviation_bound_m"),
    [
        (0.0, 0.1),
        (-0.1, 0.1),
        (np.nan, 0.1),
        (0.2, -0.1),
        (0.2, np.inf),
        (True, 0.1),
    ],
)
def test_chord_triage_rejects_invalid_bounds(
    footprint_radius_m: object,
    chord_deviation_bound_m: object,
    grid: GridSpec,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        triage_chord(
            _candidate(),
            obstacle_occupancy=np.zeros((grid.height, grid.width), dtype=bool),
            grid=grid,
            footprint_radius_m=footprint_radius_m,
            chord_deviation_bound_m=chord_deviation_bound_m,
        )


def test_float32_and_float64_inputs_produce_canonical_float64_arrays() -> None:
    candidate32 = build_reachability_candidate(
        conflict_point=np.array([2.0, 1.0], dtype=np.float32),
        source_current_xy=np.array([0.0, 0.0], dtype=np.float32),
        source_anchor_xy=np.array([1.0, 0.0], dtype=np.float32),
        desired_crossing_direction=np.array([0.0, 2.0], dtype=np.float32),
        identity=_identity(),
    )
    candidate64 = build_reachability_candidate(
        conflict_point=np.array([2.0, 1.0], dtype=np.float64),
        source_current_xy=np.array([0.0, 0.0], dtype=np.float64),
        source_anchor_xy=np.array([1.0, 0.0], dtype=np.float64),
        desired_crossing_direction=np.array([0.0, 2.0], dtype=np.float64),
        identity=_identity(),
    )

    for value in (
        candidate32.rotation_matrix,
        candidate32.current_xy,
        candidate32.conflict_point,
        candidate32.source_delta_xy,
        candidate32.desired_crossing_direction,
    ):
        assert value.dtype == np.float64
    np.testing.assert_array_equal(
        candidate32.desired_crossing_direction, [0.0, 1.0]
    )
    assert candidate32.candidate_id == candidate64.candidate_id


@pytest.mark.parametrize(
    "changes",
    [
        {"conflict_point": [1.0]},
        {"conflict_point": [np.nan, 0.0]},
        {"source_current_xy": [0.0, np.inf]},
        {"source_anchor_xy": [0.0, 0.0, 0.0]},
        {"desired_crossing_direction": [0.0, 0.0]},
        {"desired_crossing_direction": ["east", "north"]},
        {"source_current_xy": [1.0, 1.0], "source_anchor_xy": [1.0, 1.0]},
    ],
)
def test_candidate_builder_rejects_bad_vectors_or_degenerate_geometry(
    changes: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _candidate(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_id": ""},
        {"rotation_rad": np.inf},
        {"rotation_matrix": np.ones((2, 3))},
        {"rotation_matrix": np.array([[1.0, 0.0], [0.0, np.nan]])},
        {"current_xy": np.ones(3)},
        {"source_delta_xy": np.zeros(2)},
        {"desired_crossing_direction": np.zeros(2)},
        {"desired_crossing_direction": np.array([2.0, 0.0])},
    ],
)
def test_candidate_dataclass_rejects_invalid_fields(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_candidate(), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"base_state_id": ""},
        {"trajectory_id": "   "},
        {"source_snippet_id": 3},
        {"conflict_index": -1},
        {"conflict_index": 1.5},
        {"conflict_time_s": np.nan},
        {"conflict_time_s": -0.1},
        {"crossing_side": 0},
        {"crossing_side": 2},
        {"crossing_side": True},
        {"angle_offset_deg": np.inf},
    ],
)
def test_reachability_identity_strictly_validates_fields(
    changes: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _identity(**changes)


def test_candidate_arrays_are_isolated_from_inputs_and_immutable() -> None:
    conflict = np.array([2.0, 1.0], dtype=np.float64)
    source_current = np.array([0.0, 0.0], dtype=np.float64)
    source_anchor = np.array([1.0, 0.0], dtype=np.float64)
    desired = np.array([0.0, 1.0], dtype=np.float64)
    candidate = build_reachability_candidate(
        conflict_point=conflict,
        source_current_xy=source_current,
        source_anchor_xy=source_anchor,
        desired_crossing_direction=desired,
        identity=_identity(),
    )
    expected_conflict = candidate.conflict_point.copy()
    expected_delta = candidate.source_delta_xy.copy()
    expected_direction = candidate.desired_crossing_direction.copy()

    conflict[:] = 99.0
    source_current[:] = 99.0
    source_anchor[:] = 99.0
    desired[:] = 99.0

    np.testing.assert_array_equal(candidate.conflict_point, expected_conflict)
    np.testing.assert_array_equal(candidate.source_delta_xy, expected_delta)
    np.testing.assert_array_equal(
        candidate.desired_crossing_direction, expected_direction
    )
    for value in (
        candidate.rotation_matrix,
        candidate.current_xy,
        candidate.conflict_point,
        candidate.source_delta_xy,
        candidate.desired_crossing_direction,
    ):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flat[0] = 123.0


def test_schedule_array_is_immutable() -> None:
    directions = scheduled_crossing_directions(
        [1.0, 0.0], maximum_angle_deg=30.0, angle_step_deg=10.0
    )

    assert not directions.flags.writeable
    with pytest.raises(ValueError):
        directions[0, 0] = 0.0


def test_chord_triage_dataclass_rejects_non_frozen_outcome() -> None:
    with pytest.raises(ValueError, match="outcome"):
        ChordTriage(
            outcome="collision",
            chord_deviation_bound_m=0.1,
            candidate_id="candidate-id",
            identity=_identity(),
        )
