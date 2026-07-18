"""Tests for current-causal blind-region construction."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect

import numpy as np
import pytest

import src.generation.causal_occluder as causal_occluder_module
from src.contracts import BaseState, GridSpec
from src.generation import blind_region
from src.generation.causal_occluder import (
    CAUSAL_OCCLUDER_PROPOSAL_VERSION,
    CAUSAL_OCCLUDER_SCHEDULE_VERSION,
)
from src.generation.dynamic_object_transplant import footprint_from_spec
from src.generation.event_target_motion_shard import compute_footprint_spec_digest
from src.generation.occluder_sampler import (
    OccluderCollisionSweep,
    OccluderGeometryCandidate,
)
from src.geometry import (
    CircleFootprint,
    RectangleFootprint,
    footprint_aabb,
    grid_bounds,
    grid_cell_centers,
    rasterize_footprint,
    raycast_visibility,
)


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


def _causal_config() -> dict[str, object]:
    return {
        "types": ["wall", "shelf", "pillar"],
        "interaction_range_m": [1.0, 4.0],
        "bearing_bin_count": 8,
        "yaw_step_deg": 15.0,
        "minimum_shadow_center_cells": 1,
        "wall": {
            "length_range_m": [1.0, 1.0],
            "width_range_m": [0.5, 0.5],
        },
        "shelf": {
            "length_range_m": [0.8, 0.8],
            "width_range_m": [0.4, 0.4],
        },
        "pillar": {
            "length_range_m": [0.5, 0.5],
            "width_range_m": [0.5, 0.5],
        },
    }


def _decision(base: BaseState, grid: GridSpec):
    static = (
        np.zeros((grid.height, grid.width), dtype=np.float32)
        if base.static_map_local is None
        else base.static_map_local
    )
    current = np.zeros((grid.height, grid.width), dtype=np.bool_)
    for object_id in base.dynamic_object_ids:
        current |= rasterize_footprint(
            footprint_from_spec(base.visible_dynamic_object_specs[object_id]),
            base.visible_dynamic_object_history[object_id][-1],
            grid,
        )
    config = _causal_config()
    context = causal_occluder_module.build_causal_occluder_context(
        static_occupancy=static,
        current_context_occupancy=current,
        interaction_poses=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        sensor_pose=np.asarray(base.robot_history[-1], dtype=np.float64),
        grid=grid,
        config=config,
    )
    parameters = causal_occluder_module.CausalOccluderParameters(
        proposal_index=0,
        anchor_index=2,
        anchor_quantile=0.5,
        bearing_bin=0,
        range_quantile=1.0 / 3.0,
        yaw_index=0,
        yaw_offset_rad=0.0,
        occluder_type="wall",
        dimension_quantile=0.0,
    )
    decision = causal_occluder_module.propose_causal_occluder(
        context,
        collision_sweeps=(
            OccluderCollisionSweep(
                footprint=CircleFootprint(0.05),
                poses=np.asarray([[-4.0, -4.0, 0.0]], dtype=np.float64),
                rejection_reason="occluder_robot_swept_overlap",
            ),
        ),
        config=config,
        parameters=parameters,
        seed=17,
        base_state_id=base.state_id,
        trajectory_id="trajectory-1",
    )
    assert decision.accepted is not None
    return decision


def test_public_builder_exposes_only_current_causal_inputs() -> None:
    assert blind_region.BLIND_REGION_VERSION == "blind_region_v1"
    assert (
        blind_region.VISIBILITY_ALGORITHM_VERSION
        == "raycast_visibility_environment_v1"
    )
    signature = inspect.signature(blind_region.build_blind_region)
    assert tuple(signature.parameters) == ("base_state", "causal_occluder", "grid")
    forbidden = ("oracle", "future", "target", "world", "scene")
    assert not any(token in name for name in signature.parameters for token in forbidden)


def test_builder_rejects_a_bare_surface_valid_candidate() -> None:
    grid = _grid()
    base = _base_state(grid)

    with pytest.raises(TypeError, match="CausalOccluderDecision"):
        blind_region.build_blind_region(base, _candidate(grid), grid=grid)


def test_region_derives_causal_identity_from_the_verified_decision() -> None:
    grid = _grid()
    base = _base_state(grid)
    decision = _decision(base, grid)
    region = blind_region.build_blind_region(base, decision, grid=grid)

    assert region.causal_occluder_id == decision.proposal_id
    assert region.causal_context_digest == decision.context_digest
    assert region.causal_proposal_binding_digest == hashlib.sha256(
        decision._proposal_binding
    ).hexdigest()
    assert region.visibility_algorithm_version == (
        blind_region.VISIBILITY_ALGORITHM_VERSION
    )
    assert region.renderer_layout_version == "bev_history2_state9_v1"
    with pytest.raises(ValueError):
        replace(region, causal_occluder_id="causal-occluder-forged")
    with pytest.raises(ValueError):
        replace(region, causal_context_digest="0" * 64)


def test_builder_uses_one_formal_environment_raycast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _grid()
    base = _base_state(grid)
    decision = _decision(base, grid)
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
    region = blind_region.build_blind_region(base, decision, grid=grid)

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
    decision = _decision(base, grid)
    candidate = decision.accepted
    assert candidate is not None

    region = blind_region.build_blind_region(base, decision, grid=grid)
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
    decision = _decision(base, grid)

    region = blind_region.build_blind_region(base, decision, grid=grid)
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
        ("pose_metadata", "metadata (pose|geometry)"),
        ("dimension_metadata", "metadata (dimensions|geometry)"),
        ("version", "proposal_version"),
        ("type", "metadata type|occluder type"),
        ("proposal_index", "proposal_index"),
        ("mask", "mask.*(proposal geometry|rasterized)"),
    ),
)
def test_builder_rejects_spliced_causal_candidates(
    mutation: str,
    error_match: str,
) -> None:
    grid = _grid()
    base = _base_state(grid)
    decision = _decision(base, grid)
    candidate = decision.accepted
    assert candidate is not None
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
        changed_decision = replace(decision, accepted=changed)
        blind_region.build_blind_region(base, changed_decision, grid=grid)


@pytest.mark.parametrize("mutation", ("pose_dtype", "pose_nan", "mask_dtype", "bounds"))
def test_builder_rejects_noncanonical_or_out_of_bounds_geometry(
    mutation: str,
) -> None:
    grid = _grid()
    base = _base_state(grid)
    decision = _decision(base, grid)
    candidate = decision.accepted
    assert candidate is not None
    if mutation == "pose_dtype":
        changed = replace(candidate, pose=candidate.pose.astype(np.float32))
    elif mutation == "pose_nan":
        pose = np.array(candidate.pose, copy=True)
        pose[0] = np.nan
        changed = replace(candidate, pose=pose)
    elif mutation == "mask_dtype":
        changed = replace(candidate, mask=candidate.mask.astype(np.uint8))
    else:
        pose = np.asarray([5.0, 0.0, 0.0], dtype=np.float64)
        footprint = candidate.footprint
        metadata = dict(candidate.occluder)
        metadata["pose"] = tuple(float(value) for value in pose)
        changed = replace(
            candidate,
            pose=pose,
            mask=rasterize_footprint(footprint, pose, grid),
            occluder=metadata,
        )

    with pytest.raises((TypeError, ValueError)):
        changed_decision = replace(decision, accepted=changed)
        blind_region.build_blind_region(base, changed_decision, grid=grid)


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
        blind_region.build_blind_region(
            base,
            _decision(_base_state(grid), grid),
            grid=grid,
        )


def test_region_arrays_are_bytes_backed_read_only_and_deterministic() -> None:
    grid = _grid()
    base = _base_state(grid)
    decision = _decision(base, grid)

    first = blind_region.build_blind_region(base, decision, grid=grid)
    second = blind_region.build_blind_region(base, decision, grid=grid)

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
    region = blind_region.build_blind_region(base, _decision(base, grid), grid=grid)

    assert not np.any(region.static_occupancy)
    assert not np.any(region.current_context_occupancy)
    assert np.array_equal(region.total_current_occupancy, region.causal_occluder_mask)


def _rectangle_spec(
    *,
    length_m: float = 1.5,
    width_m: float = 0.5,
) -> dict[str, object]:
    return {
        "object_type": "carried_object",
        "footprint": {
            "kind": "rectangle",
            "length_m": length_m,
            "width_m": width_m,
        },
    }


def _region(*, size: int = 21, with_context: bool = False):
    grid = _grid(size=size)
    base = _base_state(grid, with_context=with_context)
    return grid, blind_region.build_blind_region(
        base,
        _decision(base, grid),
        grid=grid,
    )


def _inside(footprint, pose: np.ndarray, grid: GridSpec) -> bool:
    x_min, x_max, y_min, y_max = footprint_aabb(footprint, pose)
    gx_min, gx_max, gy_min, gy_max = grid_bounds(grid)
    return bool(
        x_min >= gx_min
        and x_max < gx_max
        and y_min >= gy_min
        and y_max < gy_max
    )


def _brute_center_mask(
    region,
    *,
    footprint_spec: dict[str, object],
    yaw_rad: float,
) -> np.ndarray:
    footprint = footprint_from_spec(footprint_spec)
    result = np.zeros(
        (region.grid.height, region.grid.width),
        dtype=np.bool_,
    )
    centers = grid_cell_centers(region.grid)
    for row in range(region.grid.height):
        for column in range(region.grid.width):
            pose = np.asarray(
                [centers[row, column, 0], centers[row, column, 1], yaw_rad],
                dtype=np.float64,
            )
            footprint_mask = rasterize_footprint(footprint, pose, region.grid)
            result[row, column] = bool(
                _inside(footprint, pose, region.grid)
                and np.all(region.blind_free_mask[footprint_mask])
            )
    return result


def test_center_and_exact_versions_and_public_apis_are_frozen() -> None:
    assert blind_region.CENTER_MASK_VERSION == "footprint_center_mask_v1"
    assert blind_region.EXACT_HIDDEN_POSE_VERSION == "exact_hidden_pose_v1"
    assert tuple(
        inspect.signature(blind_region.build_footprint_center_mask).parameters
    ) == (
        "region",
        "footprint_spec",
        "footprint_spec_digest",
        "yaw_bin_rad",
    )
    assert tuple(
        inspect.signature(blind_region.check_exact_hidden_pose).parameters
    ) == ("region", "footprint_spec", "footprint_spec_digest", "pose")


def test_circle_center_mask_is_yaw_invariant_and_matches_brute_force() -> None:
    _, region = _region(size=17)
    spec = _circle_spec(radius_m=0.35)
    digest = compute_footprint_spec_digest(spec)

    zero = blind_region.build_footprint_center_mask(
        region,
        footprint_spec=spec,
        footprint_spec_digest=digest,
        yaw_bin_rad=0.0,
    )
    rotated = blind_region.build_footprint_center_mask(
        region,
        footprint_spec=spec,
        footprint_spec_digest=digest,
        yaw_bin_rad=1.234,
    )
    brute = _brute_center_mask(region, footprint_spec=spec, yaw_rad=0.0)

    assert zero.yaw_bin_rad == 0.0
    assert rotated.yaw_bin_rad == 0.0
    assert np.array_equal(zero.center_mask, rotated.center_mask)
    assert np.array_equal(zero.center_mask, brute)
    assert zero.valid_cell_count == int(np.count_nonzero(brute))


def test_rectangle_center_masks_are_yaw_specific_and_match_brute_force() -> None:
    _, region = _region(size=21)
    spec = _rectangle_spec(length_m=1.5, width_m=0.5)
    digest = compute_footprint_spec_digest(spec)
    zero = blind_region.build_footprint_center_mask(
        region,
        footprint_spec=spec,
        footprint_spec_digest=digest,
        yaw_bin_rad=0.0,
    )
    quarter = blind_region.build_footprint_center_mask(
        region,
        footprint_spec=spec,
        footprint_spec_digest=digest,
        yaw_bin_rad=np.pi / 2.0,
    )

    assert np.array_equal(
        zero.center_mask,
        _brute_center_mask(region, footprint_spec=spec, yaw_rad=0.0),
    )
    assert np.array_equal(
        quarter.center_mask,
        _brute_center_mask(region, footprint_spec=spec, yaw_rad=np.pi / 2.0),
    )
    assert not np.array_equal(zero.center_mask, quarter.center_mask)


def test_center_mask_even_grid_and_boundaries_match_exact_brute_force() -> None:
    grid, region = _region(size=20)
    spec = _rectangle_spec(length_m=1.1, width_m=0.7)
    digest = compute_footprint_spec_digest(spec)
    result = blind_region.build_footprint_center_mask(
        region,
        footprint_spec=spec,
        footprint_spec_digest=digest,
        yaw_bin_rad=np.pi / 6.0,
    )
    brute = _brute_center_mask(
        region,
        footprint_spec=spec,
        yaw_rad=np.pi / 6.0,
    )

    assert not np.any(np.all(grid_cell_centers(grid) == 0.0, axis=-1))
    assert np.array_equal(result.center_mask, brute)
    assert not np.any(result.center_mask[[0, -1], :])
    assert not np.any(result.center_mask[:, [0, -1]])


def test_center_mask_uses_constant_number_of_footprint_rasterizations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, region = _region(size=160)
    spec = _rectangle_spec()
    digest = compute_footprint_spec_digest(spec)
    real_rasterize = blind_region.rasterize_footprint
    calls = 0

    def capture(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return real_rasterize(*args, **kwargs)

    monkeypatch.setattr(blind_region, "rasterize_footprint", capture)
    result = blind_region.build_footprint_center_mask(
        region,
        footprint_spec=spec,
        footprint_spec_digest=digest,
        yaw_bin_rad=np.pi / 3.0,
    )

    assert result.center_mask.shape == (160, 160)
    assert calls == 1


@pytest.mark.parametrize("api", ("center", "exact"))
def test_footprint_queries_reject_digest_splices(api: str) -> None:
    _, region = _region()
    spec = _circle_spec()
    kwargs = {
        "region": region,
        "footprint_spec": spec,
        "footprint_spec_digest": "0" * 32,
    }

    with pytest.raises(ValueError, match="footprint_spec_digest"):
        if api == "center":
            blind_region.build_footprint_center_mask(**kwargs, yaw_bin_rad=0.0)
        else:
            blind_region.check_exact_hidden_pose(
                **kwargs,
                pose=np.asarray([3.0, 0.0, 0.0], dtype=np.float64),
            )


def test_exact_hidden_pose_accepts_a_center_mask_pose_and_matches_direct_truth() -> None:
    grid, region = _region(size=21)
    spec = _circle_spec(radius_m=0.3)
    digest = compute_footprint_spec_digest(spec)
    center_mask = blind_region.build_footprint_center_mask(
        region,
        footprint_spec=spec,
        footprint_spec_digest=digest,
        yaw_bin_rad=0.0,
    )
    valid_indices = np.argwhere(center_mask.center_mask)
    assert valid_indices.size > 0
    row, column = valid_indices[len(valid_indices) // 2]
    xy = grid_cell_centers(grid)[row, column]
    pose = np.asarray([xy[0], xy[1], 0.37], dtype=np.float64)

    result = blind_region.check_exact_hidden_pose(
        region,
        footprint_spec=spec,
        footprint_spec_digest=digest,
        pose=pose,
    )
    footprint = CircleFootprint(0.3)
    direct_mask = rasterize_footprint(footprint, pose, grid)

    assert result.in_bounds
    assert result.collision_free == (
        not np.any(direct_mask & region.total_current_occupancy)
    )
    assert result.fully_hidden == (
        not np.any(direct_mask & region.visibility_mask)
    )
    assert result.accepted
    assert result.rejection_reason is None
    assert np.array_equal(result.footprint_mask, direct_mask)


@pytest.mark.parametrize(
    ("pose", "reason"),
    (
        (np.asarray([5.1, 0.0, 0.0], dtype=np.float64), "out_of_bounds"),
        (np.asarray([2.0, 0.0, 0.0], dtype=np.float64), "current_collision"),
        (np.asarray([-2.0, 0.0, 0.0], dtype=np.float64), "partially_visible"),
    ),
)
def test_exact_hidden_pose_reports_stable_rejection_reasons(
    pose: np.ndarray,
    reason: str,
) -> None:
    _, region = _region()
    spec = _circle_spec(radius_m=0.25)
    result = blind_region.check_exact_hidden_pose(
        region,
        footprint_spec=spec,
        footprint_spec_digest=compute_footprint_spec_digest(spec),
        pose=pose,
    )

    assert not result.accepted
    assert reason in result.rejection_reason


def test_continuous_yaw_exact_check_overrides_yaw_bin_broad_phase() -> None:
    grid, region = _region(size=25)
    spec = _rectangle_spec(length_m=1.8, width_m=0.3)
    digest = compute_footprint_spec_digest(spec)
    broad = blind_region.build_footprint_center_mask(
        region,
        footprint_spec=spec,
        footprint_spec_digest=digest,
        yaw_bin_rad=0.0,
    )
    centers = grid_cell_centers(grid)
    mismatched = None
    for row, column in np.argwhere(broad.center_mask):
        for yaw in (np.pi / 6.0, np.pi / 4.0, np.pi / 2.0):
            pose = np.asarray(
                [centers[row, column, 0], centers[row, column, 1], yaw],
                dtype=np.float64,
            )
            exact = blind_region.check_exact_hidden_pose(
                region,
                footprint_spec=spec,
                footprint_spec_digest=digest,
                pose=pose,
            )
            if not exact.accepted:
                mismatched = exact
                break
        if mismatched is not None:
            break

    assert mismatched is not None
    assert not mismatched.accepted
    assert mismatched.rejection_reason in {
        "hidden_pose_current_collision",
        "hidden_pose_partially_visible",
    }


def test_center_and_exact_outputs_are_immutable_and_deterministic() -> None:
    grid, region = _region()
    spec = _circle_spec()
    digest = compute_footprint_spec_digest(spec)
    center_a = blind_region.build_footprint_center_mask(
        region,
        footprint_spec=spec,
        footprint_spec_digest=digest,
        yaw_bin_rad=2.0,
    )
    center_b = blind_region.build_footprint_center_mask(
        region,
        footprint_spec=spec,
        footprint_spec_digest=digest,
        yaw_bin_rad=-1.0,
    )
    row, column = np.argwhere(center_a.center_mask)[0]
    xy = grid_cell_centers(grid)[row, column]
    pose = np.asarray([xy[0], xy[1], 0.0], dtype=np.float64)
    exact_a = blind_region.check_exact_hidden_pose(
        region,
        footprint_spec=spec,
        footprint_spec_digest=digest,
        pose=pose,
    )
    exact_b = blind_region.check_exact_hidden_pose(
        region,
        footprint_spec=spec,
        footprint_spec_digest=digest,
        pose=pose,
    )

    assert center_a.center_mask_digest == center_b.center_mask_digest
    assert exact_a.result_digest == exact_b.result_digest
    for values in (center_a.center_mask, exact_a.pose, exact_a.footprint_mask):
        assert values.flags.c_contiguous
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.flat[0] = values.flat[0]


@pytest.mark.parametrize("yaw", (True, np.nan, np.inf, "0"))
def test_center_mask_rejects_noncanonical_yaw(yaw: object) -> None:
    _, region = _region()
    spec = _circle_spec()

    with pytest.raises((TypeError, ValueError), match="yaw_bin_rad"):
        blind_region.build_footprint_center_mask(
            region,
            footprint_spec=spec,
            footprint_spec_digest=compute_footprint_spec_digest(spec),
            yaw_bin_rad=yaw,
        )


@pytest.mark.parametrize(
    "pose",
    (
        np.asarray([3.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([3.0, 0.0], dtype=np.float64),
        np.asarray([3.0, np.nan, 0.0], dtype=np.float64),
        [3.0, 0.0, 0.0],
    ),
)
def test_exact_hidden_pose_rejects_noncanonical_pose(pose: object) -> None:
    _, region = _region()
    spec = _circle_spec()

    with pytest.raises((TypeError, ValueError), match="pose"):
        blind_region.check_exact_hidden_pose(
            region,
            footprint_spec=spec,
            footprint_spec_digest=compute_footprint_spec_digest(spec),
            pose=pose,
        )


def test_center_and_exact_derived_evidence_cannot_be_replaced() -> None:
    grid, region = _region()
    spec = _circle_spec()
    digest = compute_footprint_spec_digest(spec)
    center = blind_region.build_footprint_center_mask(
        region,
        footprint_spec=spec,
        footprint_spec_digest=digest,
        yaw_bin_rad=0.0,
    )
    row, column = np.argwhere(center.center_mask)[0]
    xy = grid_cell_centers(grid)[row, column]
    exact = blind_region.check_exact_hidden_pose(
        region,
        footprint_spec=spec,
        footprint_spec_digest=digest,
        pose=np.asarray([xy[0], xy[1], 0.0], dtype=np.float64),
    )

    with pytest.raises(ValueError):
        replace(center, center_mask=np.zeros_like(center.center_mask))
    with pytest.raises(ValueError):
        replace(center, valid_cell_count=0)
    with pytest.raises(ValueError):
        replace(exact, accepted=not exact.accepted)
    with pytest.raises(ValueError):
        replace(exact, footprint_mask=np.zeros_like(exact.footprint_mask))
    with pytest.raises(ValueError, match="footprint_spec_digest"):
        replace(center, footprint_spec_digest="0" * 32)


def test_too_large_footprint_has_no_center_without_clipped_false_pass() -> None:
    _, region = _region(size=17)
    spec = _rectangle_spec(length_m=20.0, width_m=20.0)
    result = blind_region.build_footprint_center_mask(
        region,
        footprint_spec=spec,
        footprint_spec_digest=compute_footprint_spec_digest(spec),
        yaw_bin_rad=0.0,
    )

    assert result.valid_cell_count == 0
    assert not np.any(result.center_mask)
