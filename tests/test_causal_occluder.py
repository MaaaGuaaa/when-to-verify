"""Behavioral tests for target-independent causal occluder proposals."""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

import src.generation.causal_occluder as causal_occluder
from src.contracts import GridSpec
from src.generation.occluder_sampler import (
    OccluderCollisionSweep,
    prepare_occluder_collision_sweep,
)
from src.geometry import (
    CircleFootprint,
    RectangleFootprint,
    grid_cell_centers,
    rasterize_footprint,
    raycast_visibility,
    signed_clearance,
)


def _config(**changes) -> dict:
    config = {
        "types": ["wall", "shelf", "pillar"],
        "interaction_range_m": [1.0, 4.0],
        "bearing_bin_count": 8,
        "yaw_step_deg": 15.0,
        "minimum_shadow_center_cells": 1,
        "wall": {
            "length_range_m": [0.8, 1.6],
            "width_range_m": [0.10, 0.20],
        },
        "shelf": {
            "length_range_m": [0.6, 1.2],
            "width_range_m": [0.20, 0.40],
        },
        "pillar": {
            "length_range_m": [0.2, 0.5],
            "width_range_m": [0.2, 0.5],
        },
    }
    config.update(changes)
    return config


def _grid() -> GridSpec:
    return GridSpec(
        height=40,
        width=40,
        history_steps=3,
        future_steps=4,
        resolution_m=0.25,
    )


def _context_inputs() -> dict:
    grid = _grid()
    static = np.zeros((grid.height, grid.width), dtype=np.float32)
    static[10:30, 31] = 1.0
    current = np.zeros_like(static)
    current[23, 24] = 1.0
    return {
        "static_occupancy": static,
        "current_context_occupancy": current,
        "interaction_poses": np.asarray(
            [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.25]], dtype=np.float64
        ),
        "sensor_pose": np.asarray([0.0, 0.0, 0.4], dtype=np.float64),
        "grid": grid,
        "config": _config(),
    }


def _proposal_grid() -> GridSpec:
    return GridSpec(
        height=160,
        width=160,
        history_steps=3,
        future_steps=4,
        resolution_m=0.125,
    )


def _proposal_config(**changes) -> dict:
    config = _config(
        pillar={
            "length_range_m": [0.25, 0.5],
            "width_range_m": [0.25, 0.5],
        }
    )
    config.update(changes)
    return config


def _proposal_context(
    *,
    config: dict | None = None,
    static: np.ndarray | None = None,
    current: np.ndarray | None = None,
    grid: GridSpec | None = None,
):
    grid = grid or _proposal_grid()
    config = config or _proposal_config()
    static = (
        np.zeros((grid.height, grid.width), dtype=np.float32)
        if static is None
        else static
    )
    current = (
        np.zeros((grid.height, grid.width), dtype=np.float32)
        if current is None
        else current
    )
    return causal_occluder.build_causal_occluder_context(
        static_occupancy=static,
        current_context_occupancy=current,
        interaction_poses=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        sensor_pose=np.asarray([0.0, 0.0, 1.2], dtype=np.float64),
        grid=grid,
        config=config,
    )


def _parameters(
    *,
    proposal_index: int = 0,
    bearing_bin: int = 0,
    range_quantile: float = 1.0 / 3.0,
    dimension_quantile: float = 0.0,
    occluder_type: str = "pillar",
) -> causal_occluder.CausalOccluderParameters:
    return causal_occluder.CausalOccluderParameters(
        proposal_index=proposal_index,
        anchor_index=2,
        anchor_quantile=0.5,
        bearing_bin=bearing_bin,
        range_quantile=range_quantile,
        yaw_index=0,
        yaw_offset_rad=0.0,
        occluder_type=occluder_type,
        dimension_quantile=dimension_quantile,
    )


def _far_sweeps() -> tuple[OccluderCollisionSweep, ...]:
    return (
        OccluderCollisionSweep(
            footprint=CircleFootprint(0.05),
            poses=np.asarray([[-7.0, -7.0, 0.0]], dtype=np.float64),
            rejection_reason="occluder_robot_swept_overlap",
        ),
    )


def _propose(
    context,
    *,
    config: dict | None = None,
    parameters=None,
    collision_sweeps=None,
    seed: int = 17,
    base_state_id: str = "base-1",
    trajectory_id: str = "trajectory-1",
):
    return causal_occluder.propose_causal_occluder(
        context,
        collision_sweeps=(
            _far_sweeps() if collision_sweeps is None else collision_sweeps
        ),
        config=config or _proposal_config(),
        parameters=parameters or _parameters(),
        seed=seed,
        base_state_id=base_state_id,
        trajectory_id=trajectory_id,
    )


def test_causal_occluder_module_is_available() -> None:
    module = importlib.import_module("src.generation.causal_occluder")

    assert module.CAUSAL_OCCLUDER_SCHEDULE_VERSION == "causal_occluder_schedule_v1"
    assert module.CAUSAL_OCCLUDER_PROPOSAL_VERSION == "causal_occluder_proposal_v1"


def test_public_apis_expose_only_current_causal_inputs() -> None:
    assert tuple(
        inspect.signature(causal_occluder.build_causal_occluder_schedule).parameters
    ) == (
        "config",
        "max_candidates",
        "seed",
        "base_state_id",
        "trajectory_id",
    )
    assert tuple(
        inspect.signature(causal_occluder.build_causal_occluder_context).parameters
    ) == (
        "static_occupancy",
        "current_context_occupancy",
        "interaction_poses",
        "sensor_pose",
        "grid",
        "config",
    )
    assert tuple(
        inspect.signature(causal_occluder.propose_causal_occluder).parameters
    ) == (
        "context",
        "collision_sweeps",
        "config",
        "parameters",
        "seed",
        "base_state_id",
        "trajectory_id",
    )
    forbidden = {
        "conflict_point",
        "trajectory_normal",
        "target",
        "oracle",
        "future",
        "world",
        "scene",
    }
    for function in (
        causal_occluder.build_causal_occluder_schedule,
        causal_occluder.build_causal_occluder_context,
        causal_occluder.propose_causal_occluder,
    ):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)


def test_normalizer_enforces_the_exact_causal_schema() -> None:
    normalized = causal_occluder.normalize_causal_occluder_config(_config())

    assert normalized == {
        "types": ("wall", "shelf", "pillar"),
        "interaction_range_m": (1.0, 4.0),
        "bearing_bin_count": 8,
        "yaw_step_deg": 15.0,
        "minimum_shadow_center_cells": 1,
        "wall": {
            "length_range_m": (0.8, 1.6),
            "width_range_m": (0.1, 0.2),
        },
        "shelf": {
            "length_range_m": (0.6, 1.2),
            "width_range_m": (0.2, 0.4),
        },
        "pillar": {
            "length_range_m": (0.2, 0.5),
            "width_range_m": (0.2, 0.5),
        },
    }
    with pytest.raises(ValueError, match="keys.*frozen causal"):
        causal_occluder.normalize_causal_occluder_config(
            {**_config(), "normal_offset_range_m": [1.0, 2.0]}
        )
    with pytest.raises(ValueError, match="keys.*frozen causal"):
        causal_occluder.normalize_causal_occluder_config(
            {**_config(), "oracle_hint": 1}
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"types": []}, "types"),
        ({"types": ["wall", "wall"]}, "types"),
        ({"interaction_range_m": [2.0, 1.0]}, "interaction_range_m"),
        ({"bearing_bin_count": 3}, "bearing_bin_count"),
        ({"bearing_bin_count": True}, "bearing_bin_count"),
        ({"yaw_step_deg": np.nan}, "yaw_step_deg"),
        ({"minimum_shadow_center_cells": 0}, "minimum_shadow_center_cells"),
        (
            {"pillar": {"length_range_m": [0.0, 0.2], "width_range_m": [0.2, 0.3]}},
            "pillar.length_range_m",
        ),
    ],
)
def test_normalizer_rejects_invalid_causal_values(
    changes: dict, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        causal_occluder.normalize_causal_occluder_config(_config(**changes))


def test_parameters_are_frozen_and_self_validating() -> None:
    parameters = causal_occluder.CausalOccluderParameters(
        proposal_index=3,
        anchor_index=2,
        anchor_quantile=0.5,
        bearing_bin=5,
        range_quantile=0.75,
        yaw_index=-1,
        yaw_offset_rad=-np.deg2rad(15.0),
        occluder_type="shelf",
        dimension_quantile=0.25,
    )

    assert tuple(field.name for field in fields(parameters)) == (
        "proposal_index",
        "anchor_index",
        "anchor_quantile",
        "bearing_bin",
        "range_quantile",
        "yaw_index",
        "yaw_offset_rad",
        "occluder_type",
        "dimension_quantile",
    )
    with pytest.raises(FrozenInstanceError):
        parameters.proposal_index = 4  # type: ignore[misc]
    with pytest.raises(ValueError, match="proposal_index"):
        replace(parameters, proposal_index=-1)
    with pytest.raises(ValueError, match="quantile"):
        replace(parameters, range_quantile=1.1)
    with pytest.raises(ValueError, match="occluder_type"):
        replace(parameters, occluder_type="box")


def test_schedule_is_deterministic_finite_and_stratified_at_160() -> None:
    arguments = {
        "config": _config(),
        "max_candidates": 160,
        "seed": 1729,
        "base_state_id": "base-001",
        "trajectory_id": "trajectory-002",
    }
    first = causal_occluder.build_causal_occluder_schedule(**arguments)
    repeated = causal_occluder.build_causal_occluder_schedule(**arguments)

    assert first == repeated
    assert isinstance(first, tuple)
    assert len(first) == 160
    assert tuple(item.proposal_index for item in first) == tuple(range(160))
    assert {item.bearing_bin * 4 // 8 for item in first[:4]} == {0, 1, 2, 3}
    assert {item.bearing_bin for item in first} == set(range(8))
    assert {item.range_quantile for item in first} == {
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    }
    assert {item.yaw_index for item in first} == {-2, -1, 0, 1, 2}
    assert {item.occluder_type for item in first} == {
        "wall",
        "shelf",
        "pillar",
    }
    assert {item.dimension_quantile for item in first} == {
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    }
    assert {item.anchor_index for item in first} == set(range(5))
    assert {
        (item.anchor_index, item.anchor_quantile) for item in first
    } == set(enumerate((0.0, 0.25, 0.5, 0.75, 1.0)))
    for item in first:
        assert item.yaw_offset_rad == pytest.approx(
            item.yaw_index * np.deg2rad(15.0), abs=0.0
        )


def test_schedule_identity_inputs_change_the_stratified_order() -> None:
    common = {
        "config": _config(),
        "max_candidates": 32,
        "seed": 8,
        "base_state_id": "base-a",
        "trajectory_id": "trajectory-a",
    }
    original = causal_occluder.build_causal_occluder_schedule(**common)

    for changes in (
        {"seed": 9},
        {"base_state_id": "base-b"},
        {"trajectory_id": "trajectory-b"},
    ):
        changed = causal_occluder.build_causal_occluder_schedule(
            **{**common, **changes}
        )
        assert changed != original
        assert len(changed) == len(original)
        assert {item.bearing_bin * 4 // 8 for item in changed[:4]} == {
            0,
            1,
            2,
            3,
        }


def test_schedule_respects_every_requested_finite_budget() -> None:
    for budget in (1, 3, 4, 17):
        schedule = causal_occluder.build_causal_occluder_schedule(
            config=_config(),
            max_candidates=budget,
            seed=1,
            base_state_id="base",
            trajectory_id="trajectory",
        )
        assert len(schedule) == budget
        if budget >= 4:
            assert {item.bearing_bin * 4 // 8 for item in schedule[:4]} == {
                0,
                1,
                2,
                3,
            }


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_candidates": 0}, "max_candidates"),
        ({"max_candidates": True}, "max_candidates"),
        ({"seed": True}, "seed"),
        ({"base_state_id": ""}, "base_state_id"),
        ({"trajectory_id": ""}, "trajectory_id"),
    ],
)
def test_schedule_rejects_invalid_budget_or_identity(
    changes: dict, message: str
) -> None:
    arguments = {
        "config": _config(),
        "max_candidates": 4,
        "seed": 1,
        "base_state_id": "base",
        "trajectory_id": "trajectory",
    }
    with pytest.raises((TypeError, ValueError), match=message):
        causal_occluder.build_causal_occluder_schedule(
            **{**arguments, **changes}
        )


def test_context_stores_renderer_exact_360_infinite_baseline_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def recording_raycast(occupancy, grid, **kwargs):
        calls.append(dict(kwargs))
        return raycast_visibility(occupancy, grid, **kwargs)

    monkeypatch.setattr(
        causal_occluder, "raycast_visibility", recording_raycast
    )
    inputs = _context_inputs()
    context = causal_occluder.build_causal_occluder_context(**inputs)
    expected_occupancy = np.asarray(
        (inputs["static_occupancy"] != 0)
        | (inputs["current_context_occupancy"] != 0),
        dtype=bool,
    )
    expected_visibility = raycast_visibility(
        expected_occupancy,
        inputs["grid"],
        sensor_pose=inputs["sensor_pose"],
        fov_rad=2.0 * np.pi,
        max_range_m=None,
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["sensor_pose"], context.sensor_pose)
    assert calls[0]["fov_rad"] == 2.0 * np.pi
    assert calls[0]["max_range_m"] is None
    assert context.baseline_occupancy.tobytes() == expected_occupancy.tobytes()
    assert context.baseline_visibility.tobytes() == expected_visibility.tobytes()


def test_context_interaction_region_uses_true_grid_centres_and_pose_band() -> None:
    inputs = _context_inputs()
    context = causal_occluder.build_causal_occluder_context(**inputs)
    centers = grid_cell_centers(inputs["grid"])
    expected = np.zeros((inputs["grid"].height, inputs["grid"].width), dtype=bool)
    for pose in inputs["interaction_poses"]:
        distances = np.linalg.norm(centers - pose[:2], axis=-1)
        expected |= (distances >= 1.0) & (distances <= 4.0)

    assert context.interaction_region.tobytes() == expected.tobytes()
    assert context.interaction_range_m == (1.0, 4.0)


def test_context_owns_immutable_bytes_backed_canonical_arrays() -> None:
    inputs = _context_inputs()
    context = causal_occluder.build_causal_occluder_context(**inputs)
    arrays = (
        context.static_occupancy,
        context.current_context_occupancy,
        context.baseline_occupancy,
        context.baseline_visibility,
        context.interaction_region,
        context.interaction_poses,
        context.sensor_pose,
    )
    snapshots = tuple(array.tobytes(order="C") for array in arrays)

    inputs["static_occupancy"][:] = 0.0
    inputs["current_context_occupancy"][:] = 1.0
    inputs["interaction_poses"][:] = 2.0
    inputs["sensor_pose"][:] = 1.0

    assert tuple(array.tobytes(order="C") for array in arrays) == snapshots
    for array in arrays:
        assert array.flags.c_contiguous
        assert not array.flags.writeable
        assert not array.flags.owndata
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)


def test_context_dataclass_rejects_spliced_arrays_and_forged_digests() -> None:
    context = causal_occluder.build_causal_occluder_context(**_context_inputs())
    changed_static = np.array(context.static_occupancy, copy=True)
    changed_static[0, 0] = ~changed_static[0, 0]

    with pytest.raises(ValueError, match="static_occupancy_digest"):
        replace(context, static_occupancy=changed_static)
    with pytest.raises(ValueError, match="context_digest"):
        replace(context, context_digest="0" * 64)
    with pytest.raises(FrozenInstanceError):
        context.context_digest = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("static_occupancy", np.zeros((39, 40), dtype=np.float32), "shape"),
        ("static_occupancy", np.zeros((40, 40), dtype=np.float64), "dtype"),
        (
            "static_occupancy",
            np.full((40, 40), 0.5, dtype=np.float32),
            "binary",
        ),
        (
            "current_context_occupancy",
            np.full((40, 40), np.nan, dtype=np.float32),
            "finite",
        ),
        (
            "current_context_occupancy",
            np.full((40, 40), np.inf, dtype=np.float32),
            "finite",
        ),
        ("interaction_poses", np.zeros((3, 2), dtype=np.float64), "shape"),
        ("interaction_poses", np.zeros((2, 3), dtype=np.int64), "dtype"),
        (
            "interaction_poses",
            np.full((2, 3), np.nan, dtype=np.float64),
            "finite",
        ),
        ("sensor_pose", np.zeros(2, dtype=np.float64), "shape"),
        ("sensor_pose", np.zeros(3, dtype=np.int64), "dtype"),
        ("sensor_pose", np.full(3, np.inf, dtype=np.float64), "finite"),
    ],
)
def test_context_rejects_wrong_dtype_shape_nonfinite_or_nonbinary_arrays(
    field: str, value: np.ndarray, message: str
) -> None:
    inputs = _context_inputs()
    inputs[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        causal_occluder.build_causal_occluder_context(**inputs)


def test_empty_160_grid_accepts_robot_relative_front_left_back_and_right() -> None:
    context = _proposal_context()
    decisions = tuple(
        _propose(context, parameters=_parameters(proposal_index=index, bearing_bin=bin_))
        for index, bin_ in enumerate((0, 2, 4, 6))
    )

    assert all(decision.accepted is not None for decision in decisions)
    centers = np.asarray([decision.accepted.pose[:2] for decision in decisions])
    assert centers[0, 0] > 0.0
    assert centers[1, 1] > 0.0
    assert centers[2, 0] < 0.0
    assert centers[3, 1] < 0.0
    assert len({tuple(np.sign(center)) for center in centers}) == 4


def test_accepted_candidate_uses_direct_anchor_bearing_range_yaw_and_dimensions() -> None:
    config = _proposal_config()
    context = _proposal_context(config=config)
    parameters = _parameters(dimension_quantile=0.5)
    decision = _propose(context, config=config, parameters=parameters)

    assert decision.accepted is not None
    candidate = decision.accepted
    np.testing.assert_array_equal(
        candidate.pose,
        np.asarray([2.0, 0.0, 0.5 * np.pi], dtype=np.float64),
    )
    assert candidate.footprint.length_m == 0.375
    assert candidate.footprint.width_m == 0.375
    assert candidate.proposal_index == parameters.proposal_index
    assert candidate.occluder["placement_strategy"] == "causal_free_space_schedule_v1"
    assert candidate.occluder["proposal_parameters"] == tuple(
        (field.name, getattr(parameters, field.name)) for field in fields(parameters)
    )
    assert candidate.occluder["proposal_id"] == decision.proposal_id
    assert candidate.occluder["occluder_id"] == decision.proposal_id
    assert candidate.occluder["schedule_version"] == (
        causal_occluder.CAUSAL_OCCLUDER_SCHEDULE_VERSION
    )
    assert candidate.occluder["proposal_version"] == (
        causal_occluder.CAUSAL_OCCLUDER_PROPOSAL_VERSION
    )


def test_accepted_metadata_is_recursively_immutable() -> None:
    decision = _propose(_proposal_context())
    assert decision.accepted is not None
    metadata = decision.accepted.occluder

    with pytest.raises(TypeError):
        metadata["pose"][0] = 99.0  # type: ignore[index]
    with pytest.raises(TypeError):
        metadata["proposal_parameters"][0] = ("proposal_index", 99)  # type: ignore[index]

    def assert_immutable_nested(value) -> None:
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        assert isinstance(value, tuple)
        for nested in value:
            assert_immutable_nested(nested)

    for value in metadata.values():
        assert_immutable_nested(value)


def test_accepted_metadata_is_standard_json_encodable_after_top_level_copy() -> None:
    parameters = _parameters(dimension_quantile=0.5)
    decision = _propose(
        _proposal_context(),
        parameters=parameters,
    )
    assert decision.accepted is not None
    payload = dict(decision.accepted.occluder)

    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded["pose"] == [2.0, 0.0, 0.5 * np.pi]
    assert decoded["proposal_parameters"] == [
        [field.name, getattr(parameters, field.name)] for field in fields(parameters)
    ]


def test_proposal_id_is_stable_and_binds_every_scientific_identity_input() -> None:
    config = _proposal_config()
    context = _proposal_context(config=config)
    original = _propose(context, config=config)
    repeated = _propose(context, config=dict(config))
    assert original.proposal_id == repeated.proposal_id

    changed_ids = {
        _propose(context, config=config, seed=18).proposal_id,
        _propose(context, config=config, base_state_id="base-2").proposal_id,
        _propose(context, config=config, trajectory_id="trajectory-2").proposal_id,
        _propose(
            context,
            config=config,
            parameters=_parameters(bearing_bin=1),
        ).proposal_id,
        _propose(
            context,
            config=config,
            parameters=_parameters(dimension_quantile=1.0),
        ).proposal_id,
    }
    changed_config = _proposal_config(minimum_shadow_center_cells=2)
    changed_ids.add(
        _propose(
            _proposal_context(config=changed_config),
            config=changed_config,
        ).proposal_id
    )
    current = np.zeros((160, 160), dtype=np.float32)
    current[0, 0] = 1.0
    changed_ids.add(
        _propose(
            _proposal_context(config=config, current=current),
            config=config,
        ).proposal_id
    )

    assert original.proposal_id not in changed_ids
    assert len(changed_ids) == 7


def test_useful_shadow_is_byte_exact_direct_renderer_delta_without_target() -> None:
    context = _proposal_context()
    decision = _propose(context)

    assert decision.accepted is not None
    visibility_with_obstacle = raycast_visibility(
        context.baseline_occupancy | decision.accepted.mask,
        context.grid,
        sensor_pose=context.sensor_pose,
        fov_rad=2.0 * np.pi,
        max_range_m=None,
    )
    expected = (
        context.baseline_visibility
        & ~visibility_with_obstacle
        & ~context.baseline_occupancy
        & ~decision.accepted.mask
        & context.interaction_region
    )

    assert decision.useful_shadow_mask.tobytes() == expected.tobytes()
    assert decision.useful_shadow_count == int(np.count_nonzero(expected))
    assert not np.any(decision.useful_shadow_mask & context.baseline_occupancy)
    assert not np.any(decision.useful_shadow_mask & decision.accepted.mask)
    assert not np.any(decision.useful_shadow_mask & ~context.interaction_region)


def test_accepted_decision_and_candidate_arrays_are_frozen_bytes_backed() -> None:
    decision = _propose(_proposal_context())

    assert decision.rejection_stage is None
    assert decision.rejection_reason is None
    assert decision.accepted is not None
    arrays = (
        decision.useful_shadow_mask,
        decision.accepted.pose,
        decision.accepted.mask,
    )
    for array in arrays:
        assert array.flags.c_contiguous
        assert not array.flags.writeable
        assert not array.flags.owndata
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        decision.proposal_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="useful_shadow_count"):
        replace(decision, useful_shadow_count=decision.useful_shadow_count + 1)
    with pytest.raises(ValueError, match="proposal_id"):
        replace(decision, proposal_id="forged-proposal")


def test_rejection_stages_follow_bounds_static_clearance_shadow_order() -> None:
    bounds_config = _proposal_config(interaction_range_m=[20.0, 20.0])
    bounds = _propose(
        _proposal_context(config=bounds_config),
        config=bounds_config,
        collision_sweeps=(object(),),
    )

    grid = _proposal_grid()
    footprint = RectangleFootprint(0.25, 0.25)
    pose = np.asarray([2.0, 0.0, 0.5 * np.pi], dtype=np.float64)
    static = rasterize_footprint(footprint, pose, grid).astype(np.float32)
    colliding = OccluderCollisionSweep(
        footprint=CircleFootprint(0.1),
        poses=pose[None, :],
        rejection_reason="occluder_robot_swept_overlap",
    )
    static_overlap = _propose(
        _proposal_context(static=static),
        collision_sweeps=(colliding,),
    )
    clearance = _propose(
        _proposal_context(),
        collision_sweeps=(colliding,),
    )

    shadow_config = _proposal_config(
        minimum_shadow_center_cells=grid.height * grid.width
    )
    shadow = _propose(
        _proposal_context(config=shadow_config),
        config=shadow_config,
    )

    assert (bounds.rejection_stage, bounds.rejection_reason) == (
        "bounds",
        "occluder_out_of_bounds",
    )
    assert (static_overlap.rejection_stage, static_overlap.rejection_reason) == (
        "static",
        "occluder_static_overlap",
    )
    assert (clearance.rejection_stage, clearance.rejection_reason) == (
        "continuous_clearance",
        "occluder_robot_swept_overlap",
    )
    assert (shadow.rejection_stage, shadow.rejection_reason) == (
        "shadow",
        "occluder_no_useful_shadow",
    )
    for rejected in (bounds, static_overlap, clearance, shadow):
        assert rejected.accepted is None
        assert rejected.proposal_id.startswith("causal-occluder-")
        assert rejected.proposal_index == 0
    with pytest.raises(ValueError, match="proposal_id"):
        replace(bounds, proposal_id="forged-rejected-proposal")
    with pytest.raises(ValueError, match="rejection_reason"):
        replace(bounds, rejection_reason="occluder_static_overlap")


@pytest.mark.parametrize(
    "reason",
    ("occluder_robot_swept_overlap", "occluder_context_collision"),
)
def test_continuous_clearance_preserves_the_first_authoritative_reason(
    reason: str,
) -> None:
    first = OccluderCollisionSweep(
        footprint=CircleFootprint(0.1),
        poses=np.asarray([[2.0, 0.0, 0.0]], dtype=np.float64),
        rejection_reason=reason,
    )
    second = OccluderCollisionSweep(
        footprint=CircleFootprint(0.1),
        poses=np.asarray([[2.0, 0.0, 0.0]], dtype=np.float64),
        rejection_reason="must-not-win",
    )

    decision = _propose(
        _proposal_context(), collision_sweeps=(first, second)
    )

    assert decision.rejection_stage == "continuous_clearance"
    assert decision.rejection_reason == reason


def test_raw_prepared_and_mixed_collision_sweeps_have_identical_verdicts() -> None:
    context = _proposal_context()
    raw = OccluderCollisionSweep(
        footprint=CircleFootprint(0.1),
        poses=np.asarray([[2.0, 0.0, 0.0]], dtype=np.float64),
        rejection_reason="occluder_context_collision",
    )
    prepared = prepare_occluder_collision_sweep(raw, grid=context.grid)
    far_prepared = prepare_occluder_collision_sweep(
        _far_sweeps()[0], grid=context.grid
    )

    raw_decision = _propose(context, collision_sweeps=(raw,))
    prepared_decision = _propose(context, collision_sweeps=(prepared,))
    mixed_decision = _propose(
        context, collision_sweeps=(far_prepared, raw)
    )

    assert raw_decision.proposal_id == prepared_decision.proposal_id
    assert (
        raw_decision.rejection_stage,
        raw_decision.rejection_reason,
    ) == (
        prepared_decision.rejection_stage,
        prepared_decision.rejection_reason,
    ) == (
        mixed_decision.rejection_stage,
        mixed_decision.rejection_reason,
    ) == ("continuous_clearance", "occluder_context_collision")


def test_raster_sweep_overlap_cannot_override_positive_signed_clearance() -> None:
    grid = GridSpec(
        height=40,
        width=40,
        history_steps=3,
        future_steps=4,
        resolution_m=0.5,
    )
    context = _proposal_context(grid=grid)
    occluder = RectangleFootprint(0.25, 0.25)
    occluder_pose = np.asarray([2.0, 0.0, 0.5 * np.pi], dtype=np.float64)
    robot = CircleFootprint(0.02)
    robot_pose = np.asarray([2.0, 0.4, 0.0], dtype=np.float64)
    occluder_mask = rasterize_footprint(occluder, occluder_pose, grid)
    robot_mask = rasterize_footprint(robot, robot_pose, grid)

    assert signed_clearance(occluder, occluder_pose, robot, robot_pose) > 0.0
    assert np.any(occluder_mask & robot_mask)

    decision = _propose(
        context,
        collision_sweeps=(
            OccluderCollisionSweep(
                footprint=robot,
                poses=robot_pose[None, :],
                rejection_reason="occluder_robot_swept_overlap",
            ),
        ),
    )

    assert decision.accepted is not None
    assert decision.rejection_stage is None


def test_unexpected_clearance_errors_propagate_and_parameter_binding_is_strict() -> None:
    context = _proposal_context()
    with pytest.raises(TypeError, match="collision_sweeps"):
        _propose(context, collision_sweeps=(object(),))
    with pytest.raises(ValueError, match="yaw index/offset"):
        _propose(
            context,
            parameters=replace(_parameters(), yaw_offset_rad=0.1),
        )
    with pytest.raises(ValueError, match="config.*context"):
        _propose(
            context,
            config=_proposal_config(minimum_shadow_center_cells=2),
        )
