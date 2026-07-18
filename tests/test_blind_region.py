"""Tests for current-causal blind-region construction."""

from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from src.contracts import BaseState, GridSpec
from src.generation import blind_region
from src.generation.causal_occluder import (
    CAUSAL_OCCLUDER_PROPOSAL_VERSION,
    CAUSAL_OCCLUDER_SCHEDULE_VERSION,
)
from src.generation.dynamic_object_transplant import footprint_from_spec
from src.generation.occluder_sampler import OccluderGeometryCandidate
from src.geometry import RectangleFootprint, rasterize_footprint, raycast_visibility


def _grid(*, size: int = 21) -> GridSpec:
    return GridSpec(
        height=size,
        width=size,
        history_steps=3,
        future_steps=15,
        resolution_m=0.5,
    )


def _circle_spec(radius_m: float = 0.3) -> dict[str, object]:
    return {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": radius_m},
    }


def _base_state(
    grid: GridSpec,
    *,
    state_id: str = "base-1",
    static: np.ndarray | None = None,
    with_context: bool = True,
) -> BaseState:
    robot_history = np.asarray(
        [[-0.4, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    if with_context:
        object_ids = ("context-1",)
        context_history = {
            "context-1": np.asarray(
                [[-3.0, -3.0, 0.0], [-2.5, -2.5, 0.0], [-2.0, 2.0, 0.0]],
                dtype=np.float32,
            )
        }
        context_specs = {"context-1": _circle_spec()}
    else:
        object_ids = ()
        context_history = {}
        context_specs = {}
    return BaseState(
        state_id=state_id,
        split="train",
        recording_id="recording-1",
        dynamic_object_ids=object_ids,
        timestamp=1.0,
        robot_history=robot_history,
        robot_state=np.asarray([0.5, 0.0], dtype=np.float32),
        visible_dynamic_object_history=context_history,
        visible_dynamic_object_specs=context_specs,
        static_map_local=(
            np.zeros((grid.height, grid.width), dtype=np.float32)
            if static is None
            else static
        ),
    )


def _candidate(
    grid: GridSpec,
    *,
    base_state_id: str = "base-1",
    pose: np.ndarray | None = None,
    footprint: RectangleFootprint | None = None,
) -> OccluderGeometryCandidate:
    candidate_pose = (
        np.asarray([2.0, 0.0, 0.0], dtype=np.float64)
        if pose is None
        else pose
    )
    candidate_footprint = footprint or RectangleFootprint(1.0, 0.5)
    mask = rasterize_footprint(candidate_footprint, candidate_pose, grid)
    proposal_id = "causal-occluder-test"
    metadata = {
        "occluder_id": proposal_id,
        "proposal_id": proposal_id,
        "type": "wall",
        "pose": tuple(float(value) for value in candidate_pose),
        "length_m": float(candidate_footprint.length_m),
        "width_m": float(candidate_footprint.width_m),
        "geometry_source": "generator_config",
        "placement_strategy": "causal_free_space_schedule_v1",
        "schedule_version": CAUSAL_OCCLUDER_SCHEDULE_VERSION,
        "proposal_version": CAUSAL_OCCLUDER_PROPOSAL_VERSION,
        "base_state_id": base_state_id,
        "proposal_index": 0,
    }
    return OccluderGeometryCandidate(
        occluder=metadata,
        footprint=candidate_footprint,
        pose=candidate_pose,
        mask=mask,
        proposal_index=0,
    )


def test_public_builder_exposes_only_current_causal_inputs() -> None:
    assert blind_region.BLIND_REGION_VERSION == "blind_region_v1"
    signature = inspect.signature(blind_region.build_blind_region)
    assert tuple(signature.parameters) == ("base_state", "causal_occluder", "grid")
    forbidden = ("oracle", "future", "target", "world", "scene")
    assert not any(token in name for name in signature.parameters for token in forbidden)


def test_builder_uses_one_formal_environment_raycast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _grid()
    base = _base_state(grid)
    candidate = _candidate(grid)
    calls: list[tuple[np.ndarray, np.ndarray, float, float | None]] = []

    def capture(
        occupancy: np.ndarray,
        call_grid: GridSpec,
        *,
        sensor_pose: np.ndarray,
        fov_rad: float,
        max_range_m: float | None,
    ) -> np.ndarray:
        assert call_grid == grid
        calls.append(
            (
                np.array(occupancy, copy=True),
                np.array(sensor_pose, copy=True),
                float(fov_rad),
                max_range_m,
            )
        )
        return raycast_visibility(
            occupancy,
            call_grid,
            sensor_pose=sensor_pose,
            fov_rad=fov_rad,
            max_range_m=max_range_m,
        )

    monkeypatch.setattr(blind_region, "raycast_visibility", capture)
    region = blind_region.build_blind_region(base, candidate, grid=grid)

    assert len(calls) == 1
    occupancy, sensor_pose, fov_rad, max_range_m = calls[0]
    assert np.array_equal(occupancy, region.total_current_occupancy)
    assert np.array_equal(sensor_pose, base.robot_history[-1])
    assert fov_rad == pytest.approx(2.0 * np.pi, rel=0.0, abs=0.0)
    assert max_range_m is None


def test_builder_matches_renderer_kernel_and_mask_equations() -> None:
    grid = _grid()
    static = np.zeros((grid.height, grid.width), dtype=np.float32)
    static[:, 16] = 1.0
    base = _base_state(grid, static=static)
    candidate = _candidate(grid)

    region = blind_region.build_blind_region(base, candidate, grid=grid)
    context = rasterize_footprint(
        footprint_from_spec(base.visible_dynamic_object_specs["context-1"]),
        base.visible_dynamic_object_history["context-1"][-1],
        grid,
    )
    expected_static = static != 0.0
    expected_total = expected_static | context | candidate.mask
    direct_visibility = raycast_visibility(
        expected_total,
        grid,
        sensor_pose=base.robot_history[-1],
        fov_rad=2.0 * np.pi,
        max_range_m=None,
    )

    assert np.array_equal(region.static_occupancy, expected_static)
    assert np.array_equal(region.current_context_occupancy, context)
    assert np.array_equal(region.causal_occluder_mask, candidate.mask)
    assert np.array_equal(region.total_current_occupancy, expected_total)
    assert region.visibility_mask.tobytes() == direct_visibility.tobytes()
    assert np.array_equal(region.raw_unobservable_mask, ~region.visibility_mask)
    assert np.array_equal(
        region.blind_free_mask,
        region.raw_unobservable_mask & ~region.total_current_occupancy,
    )
    assert not np.any(region.blind_free_mask & region.total_current_occupancy)


def test_builder_uses_only_last_visible_context_pose() -> None:
    grid = _grid()
    base = _base_state(grid)
    candidate = _candidate(grid)

    region = blind_region.build_blind_region(base, candidate, grid=grid)
    footprint = footprint_from_spec(base.visible_dynamic_object_specs["context-1"])
    old_mask = rasterize_footprint(
        footprint,
        base.visible_dynamic_object_history["context-1"][0],
        grid,
    )
    current_mask = rasterize_footprint(
        footprint,
        base.visible_dynamic_object_history["context-1"][-1],
        grid,
    )

    assert np.array_equal(region.current_context_occupancy, current_mask)
    assert np.any(old_mask & ~current_mask)
    assert not np.any((old_mask & ~current_mask) & region.current_context_occupancy)


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    (
        ("base_id", "base_state_id"),
        ("proposal_id", "proposal_id"),
        ("pose_metadata", "metadata pose"),
        ("dimension_metadata", "metadata dimensions"),
        ("version", "proposal_version"),
        ("type", "occluder type"),
        ("proposal_index", "proposal_index"),
        ("mask", "rasterized"),
    ),
)
def test_builder_rejects_spliced_causal_candidates(
    mutation: str,
    error_match: str,
) -> None:
    grid = _grid()
    base = _base_state(grid)
    candidate = _candidate(grid)
    metadata = dict(candidate.occluder)
    changed = candidate
    if mutation == "base_id":
        metadata["base_state_id"] = "base-other"
    elif mutation == "proposal_id":
        metadata["proposal_id"] = "causal-occluder-other"
    elif mutation == "pose_metadata":
        metadata["pose"] = (1.5, 0.0, 0.0)
    elif mutation == "dimension_metadata":
        metadata["length_m"] = 2.0
    elif mutation == "version":
        metadata["proposal_version"] = "causal_occluder_proposal_v2"
    elif mutation == "type":
        metadata["type"] = "decorative"
    elif mutation == "proposal_index":
        changed = replace(candidate, proposal_index=1)
    else:
        changed_mask = np.array(candidate.mask, copy=True)
        changed_mask[0, 0] = ~changed_mask[0, 0]
        changed = replace(candidate, mask=changed_mask)
    if metadata != candidate.occluder:
        changed = replace(changed, occluder=metadata)

    with pytest.raises((TypeError, ValueError), match=error_match):
        blind_region.build_blind_region(base, changed, grid=grid)


@pytest.mark.parametrize("mutation", ("pose_dtype", "pose_nan", "mask_dtype", "bounds"))
def test_builder_rejects_noncanonical_or_out_of_bounds_geometry(
    mutation: str,
) -> None:
    grid = _grid()
    base = _base_state(grid)
    candidate = _candidate(grid)
    if mutation == "pose_dtype":
        changed = replace(candidate, pose=candidate.pose.astype(np.float32))
    elif mutation == "pose_nan":
        pose = np.array(candidate.pose, copy=True)
        pose[0] = np.nan
        changed = replace(candidate, pose=pose)
    elif mutation == "mask_dtype":
        changed = replace(candidate, mask=candidate.mask.astype(np.uint8))
    else:
        changed = _candidate(
            grid,
            pose=np.asarray([5.0, 0.0, 0.0], dtype=np.float64),
        )

    with pytest.raises((TypeError, ValueError)):
        blind_region.build_blind_region(base, changed, grid=grid)


@pytest.mark.parametrize("mutation", ("empty_id", "robot_origin", "static_nonbinary"))
def test_builder_rejects_renderer_incompatible_base_state(mutation: str) -> None:
    grid = _grid()
    base = _base_state(grid)
    if mutation == "empty_id":
        base = replace(base, state_id="")
    elif mutation == "robot_origin":
        robot_history = np.array(base.robot_history, copy=True)
        robot_history[-1, 0] = 0.1
        base = replace(base, robot_history=robot_history)
    else:
        static = np.array(base.static_map_local, copy=True)
        static[0, 0] = 0.5
        base = replace(base, static_map_local=static)

    with pytest.raises((TypeError, ValueError)):
        blind_region.build_blind_region(base, _candidate(grid), grid=grid)


def test_region_arrays_are_bytes_backed_read_only_and_deterministic() -> None:
    grid = _grid()
    base = _base_state(grid)
    candidate = _candidate(grid)

    first = blind_region.build_blind_region(base, candidate, grid=grid)
    second = blind_region.build_blind_region(base, candidate, grid=grid)

    assert first.region_digest == second.region_digest
    assert first.raw_unobservable_digest == second.raw_unobservable_digest
    assert first.blind_free_digest == second.blind_free_digest
    assert first.blind_free_count == int(np.count_nonzero(first.blind_free_mask))
    for name in (
        "sensor_pose",
        "static_occupancy",
        "current_context_occupancy",
        "causal_occluder_mask",
        "total_current_occupancy",
        "visibility_mask",
        "raw_unobservable_mask",
        "blind_free_mask",
    ):
        values = getattr(first, name)
        assert values.flags.c_contiguous
        assert not values.flags.writeable
        assert isinstance(values.base, (bytes, np.ndarray))
        with pytest.raises(ValueError):
            values.flat[0] = values.flat[0]


def test_none_static_map_is_treated_as_empty_renderer_occupancy() -> None:
    grid = _grid()
    base = replace(_base_state(grid, with_context=False), static_map_local=None)
    region = blind_region.build_blind_region(base, _candidate(grid), grid=grid)

    assert not np.any(region.static_occupancy)
    assert not np.any(region.current_context_occupancy)
    assert np.array_equal(region.total_current_occupancy, region.causal_occluder_mask)
