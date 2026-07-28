"""M1 tests for the SOP05 unseen-history prior input and rotation contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import yaml

import src.generation.sop05_unseen_prior as unseen_prior_module
from src.contracts import (
    BaseState,
    GridSpec,
    LocalTrajectory,
    OracleWorld,
    POSE_TIME_LAYOUT_VERSION,
)
from src.geometry import CircleFootprint
from src.generation.sop06_single import (
    Sop06SinglePublication,
    Sop06SinglePublicationContext,
    Sop06SingleRendererInput,
    adapt_unseen_prior_realization,
    build_sop06_single_risk_input,
    coordinate_sop06_single_release,
    render_sop06_single_publication,
    write_sop06_single_risk_shard,
)
from src.generation.sop05_unseen_prior import (
    LONG40_LAYOUT_VERSION,
    LONG40_SCHEMA_VERSION,
    UNSEEN_PRIOR_CONTRACT_VERSION,
    UNSEEN_PRIOR_GENERATOR_VERSION,
    UNSEEN_PRIOR_REGIME,
    Long40TargetMotion,
    UnseenPriorCandidateDecision,
    UnseenPriorConfig,
    UnseenPriorConfigError,
    UnseenPriorContextObstacle,
    UnseenPriorInputError,
    UnseenPriorMother,
    UnseenPriorMotherResult,
    UnseenPriorRun,
    evaluate_candidate,
    generate_unseen_prior_mother,
    normalize_unseen_prior_config,
    run_unseen_prior,
    transform_long40_target,
    validate_long40_base_config,
)
from src.generation.risk_gt import compute_hidden_risk_gt
from src.generation.structural_blindspot import StructuralBlindSpot
from src.utils.config import config_digest, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "sop05_unseen_prior.yaml"
BASE_CONFIG_PATH = ROOT / "configs" / "base.yaml"


def _raw_config() -> dict[str, object]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _motion_kwargs() -> dict[str, object]:
    positions = np.zeros((40, 2), dtype=np.float32)
    positions[:, 0] = np.linspace(-2.0, 5.8, 40, dtype=np.float32)
    positions[:, 1] = np.linspace(1.5, -2.4, 40, dtype=np.float32)
    velocities = np.zeros((40, 2), dtype=np.float32)
    velocities[:, 0] = np.linspace(0.2, 1.0, 40, dtype=np.float32)
    velocities[:, 1] = np.linspace(-0.8, 0.4, 40, dtype=np.float32)
    return {
        "target_dynamic_object_id": "target-1",
        "source_recording_id": "recording-1",
        "source_session_id": "session-1",
        "source_snippet_id": "snippet-1",
        "source_object_id": "object-1",
        "object_type": "human",
        "footprint_spec": {
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": 0.3},
        },
        "layout_version": LONG40_LAYOUT_VERSION,
        "positions": positions,
        "velocities": velocities,
        "headings": np.linspace(-1.0, 1.0, 40, dtype=np.float32),
    }


def _float32_wrapped_headings(headings: np.ndarray, angle_rad: float) -> np.ndarray:
    wrapped64 = np.remainder(
        headings.astype(np.float64) + angle_rad + np.pi, 2.0 * np.pi
    ) - np.pi
    wrapped64 = np.where(wrapped64 >= np.pi, wrapped64 - 2.0 * np.pi, wrapped64)
    wrapped = np.asarray(wrapped64, dtype=np.float32)
    return np.clip(
        wrapped,
        np.nextafter(np.float32(-np.pi), np.float32(np.inf)),
        np.nextafter(np.float32(np.pi), np.float32(-np.inf)),
    )


def test_config_is_minimal_and_normalizes_to_a_stable_digest() -> None:
    raw = _raw_config()
    assert raw == {
        "schema_version": "4.0.0",
        "generator_version": "sop05_unseen_history_prior_v1",
        "contract_version": "sop05_unseen_history_prior_contract_v1",
        "p_hidden_human": 0.30,
        "max_attempts_per_mother": 32,
        "max_variants_per_mother": 1,
        "hard_total_sample_cap": 125000,
        "manifest_targets": [50000, 100000, 125000],
        "seed": 42,
    }

    first = normalize_unseen_prior_config(raw, base_config=load_config(BASE_CONFIG_PATH))
    second = normalize_unseen_prior_config(deepcopy(raw))

    assert first == second
    assert first.config_digest == config_digest(raw)
    assert first.max_attempts_per_mother == 32
    assert first.max_variants_per_mother == 1
    assert first.manifest_targets == (50000, 100000, 125000)


def test_config_removes_grid_audit_and_training_weight_interfaces() -> None:
    raw = _raw_config()
    forbidden_keys = {
        "production_angle_count",
        "production_angle_step_deg",
        "audit_max_mothers",
        "audit_angle_count",
        "audit_angle_step_deg",
        "near_collision_clearance_threshold_m",
    }
    assert forbidden_keys.isdisjoint(raw)
    for name in (
        "UNSEEN_PRIOR_ANGLE_SCHEDULE_VERSION",
        "UNSEEN_PRIOR_AUDIT_VERSION",
        "UNSEEN_PRIOR_TRAINING_WEIGHT_VERSION",
        "UnseenPriorTrainingWeightHandoff",
    ):
        assert not hasattr(unseen_prior_module, name)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda raw: raw.__setitem__("p_hidden_human", 0.20),
        lambda raw: raw.__setitem__("max_attempts_per_mother", 8),
        lambda raw: raw.__setitem__("max_variants_per_mother", 2),
        lambda raw: raw.__setitem__("hard_total_sample_cap", 125001),
        lambda raw: raw.__setitem__("manifest_targets", [50000, 100000, 124999]),
        lambda raw: raw.__setitem__("seed", -1),
        lambda raw: raw.__setitem__("production_angle_count", 360),
        lambda raw: raw.pop("seed"),
    ),
)
def test_config_rejects_contract_drift(mutate: object) -> None:
    raw = _raw_config()
    assert callable(mutate)
    mutate(raw)
    with pytest.raises(UnseenPriorConfigError):
        normalize_unseen_prior_config(raw)


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("schema_version", "3.0.0"),
        ("generator_version", "different-generator"),
        ("contract_version", "different-contract"),
        ("p_hidden_human", float("nan")),
        ("max_attempts_per_mother", 32.0),
        ("seed", True),
    ),
)
def test_config_rejects_invalid_value_types(field: str, invalid: object) -> None:
    raw = _raw_config()
    raw[field] = invalid
    with pytest.raises(UnseenPriorConfigError):
        normalize_unseen_prior_config(raw)


@pytest.mark.parametrize(
    ("path", "invalid"),
    (
        (("schema_version",), "3.0.0"),
        (("bev", "history_steps"), 7),
        (("bev", "future_steps"), 15),
        (("bev", "history_dt_s"), 0.1),
        (("bev", "future_dt_s"), 0.1),
    ),
)
def test_config_validates_the_base_long40_layout(
    path: tuple[str, ...], invalid: object
) -> None:
    base = load_config(BASE_CONFIG_PATH)
    node = base
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = invalid
    with pytest.raises(UnseenPriorConfigError):
        validate_long40_base_config(base)


def test_long40_target_requires_the_authenticated_fixed_layout() -> None:
    motion = Long40TargetMotion(**_motion_kwargs())
    assert tuple(field.name for field in fields(Long40TargetMotion)) == (
        "target_dynamic_object_id",
        "source_recording_id",
        "source_session_id",
        "source_snippet_id",
        "source_object_id",
        "object_type",
        "footprint_spec",
        "layout_version",
        "positions",
        "velocities",
        "headings",
    )
    assert LONG40_SCHEMA_VERSION == "4.0.0"
    assert LONG40_LAYOUT_VERSION == "history8_current7_future32_v1"
    assert UNSEEN_PRIOR_GENERATOR_VERSION == "sop05_unseen_history_prior_v1"
    assert UNSEEN_PRIOR_CONTRACT_VERSION == "sop05_unseen_history_prior_contract_v1"
    assert UNSEEN_PRIOR_REGIME == "unseen_in_history_window"
    assert motion.positions.shape == (40, 2)
    assert motion.velocities.shape == (40, 2)
    assert motion.headings.shape == (40,)
    assert all(array.dtype == np.dtype(np.float32) for array in (
        motion.positions,
        motion.velocities,
        motion.headings,
    ))
    assert all(not array.flags.writeable for array in (
        motion.positions,
        motion.velocities,
        motion.headings,
    ))


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("layout_version", "legacy23_current7_future15_v0"),
        ("positions", np.zeros((39, 2), dtype=np.float32)),
        ("positions", np.zeros((40, 2), dtype=np.float64)),
        ("velocities", np.full((40, 2), np.inf, dtype=np.float32)),
        ("headings", np.zeros((40, 1), dtype=np.float32)),
        ("headings", np.full(40, np.nan, dtype=np.float32)),
        ("target_dynamic_object_id", ""),
        ("footprint_spec", {}),
    ),
)
def test_long40_target_rejects_missing_or_inconsistent_input(
    field: str, invalid: object
) -> None:
    kwargs = _motion_kwargs()
    kwargs[field] = invalid
    with pytest.raises(UnseenPriorInputError):
        Long40TargetMotion(**kwargs)


@pytest.mark.parametrize("angle_rad", (0.0, np.pi / 2.0, -0.75))
def test_transform_long40_rotates_positions_velocity_and_heading_synchronously(
    angle_rad: float,
) -> None:
    source = Long40TargetMotion(**_motion_kwargs())
    source_bytes = tuple(
        array.tobytes() for array in (
            source.positions,
            source.velocities,
            source.headings,
        )
    )
    pivot = source.positions[7].astype(np.float64)
    rotation = np.asarray(
        (
            (np.cos(angle_rad), -np.sin(angle_rad)),
            (np.sin(angle_rad), np.cos(angle_rad)),
        ),
        dtype=np.float64,
    )
    expected_positions = (
        pivot + (source.positions.astype(np.float64) - pivot) @ rotation.T
    ).astype(np.float32)
    expected_velocities = (source.velocities.astype(np.float64) @ rotation.T).astype(
        np.float32
    )
    expected_headings = _float32_wrapped_headings(source.headings, angle_rad)

    transformed = transform_long40_target(source, angle_rad=angle_rad)

    np.testing.assert_allclose(transformed.positions, expected_positions, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(transformed.velocities, expected_velocities, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(transformed.headings, expected_headings, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(transformed.positions[7], source.positions[7], atol=1e-5, rtol=0.0)
    assert tuple(array.tobytes() for array in (
        source.positions,
        source.velocities,
        source.headings,
    )) == source_bytes
    assert all(not np.shares_memory(source_array, transformed_array) for source_array, transformed_array in (
        (source.positions, transformed.positions),
        (source.velocities, transformed.velocities),
        (source.headings, transformed.headings),
    ))
    assert transformed.layout_version == source.layout_version


def test_transform_long40_wraps_heading_to_the_shared_float32_half_open_interval() -> None:
    kwargs = _motion_kwargs()
    kwargs["headings"] = np.full(40, np.float32(3.0), dtype=np.float32)
    transformed = transform_long40_target(Long40TargetMotion(**kwargs), angle_rad=0.5)

    lower = np.nextafter(np.float32(-np.pi), np.float32(np.inf))
    upper = np.nextafter(np.float32(np.pi), np.float32(-np.inf))
    assert np.all(transformed.headings >= lower)
    assert np.all(transformed.headings <= upper)
    assert transformed.headings[0] < 0.0


@pytest.mark.parametrize("angle_rad", (float("nan"), float("inf"), True, "0.0"))
def test_transform_long40_rejects_nonfinite_or_nonreal_angle(angle_rad: object) -> None:
    with pytest.raises(UnseenPriorInputError):
        transform_long40_target(Long40TargetMotion(**_motion_kwargs()), angle_rad=angle_rad)


def test_normalized_config_dataclass_is_limited_to_m1_fields() -> None:
    assert tuple(field.name for field in fields(UnseenPriorConfig)) == (
        "schema_version",
        "generator_version",
        "contract_version",
        "p_hidden_human",
        "max_attempts_per_mother",
        "max_variants_per_mother",
        "hard_total_sample_cap",
        "manifest_targets",
        "seed",
        "config_digest",
    )
    assert UnseenPriorConfig.__dataclass_params__.frozen


def _m2_grid() -> GridSpec:
    return GridSpec(
        height=20,
        width=20,
        history_steps=8,
        future_steps=32,
        resolution_m=1.0,
    )


def _m2_target(*, positions: np.ndarray | None = None) -> Long40TargetMotion:
    kwargs = _motion_kwargs()
    location = np.tile(np.asarray((2.0, 0.0), dtype=np.float32), (40, 1))
    kwargs["positions"] = location if positions is None else positions
    kwargs["velocities"] = np.zeros((40, 2), dtype=np.float32)
    kwargs["headings"] = np.zeros(40, dtype=np.float32)
    return Long40TargetMotion(**kwargs)


def _fully_blind_sensor() -> StructuralBlindSpot:
    return StructuralBlindSpot(
        forward_fov_deg=360.0,
        range_m=20.0,
        blind_sectors=({"center_deg": 0.0, "width_deg": 360.0},),
    )


def _m2_mother(
    target: Long40TargetMotion,
    *,
    mother_id: str = "mother-m2",
    split: str = "train",
    static_occupancy: np.ndarray | None = None,
    occluder_occupancy: np.ndarray | None = None,
    context_obstacles: tuple[UnseenPriorContextObstacle, ...] = (),
    robot_history: np.ndarray | None = None,
    sensor_config: StructuralBlindSpot | None = _fully_blind_sensor(),
) -> UnseenPriorMother:
    grid = _m2_grid()
    assert sensor_config is None or isinstance(sensor_config, StructuralBlindSpot)
    return UnseenPriorMother(
        mother_id=mother_id,
        split=split,
        target_motion=target,
        grid=grid,
        robot_footprint=CircleFootprint(0.25),
        robot_history=(
            np.zeros((8, 3), dtype=np.float32)
            if robot_history is None
            else robot_history
        ),
        static_occupancy=(
            np.zeros((grid.height, grid.width), dtype=bool)
            if static_occupancy is None
            else static_occupancy
        ),
        occluder_occupancy=(
            np.zeros((grid.height, grid.width), dtype=bool)
            if occluder_occupancy is None
            else occluder_occupancy
        ),
        context_obstacles=context_obstacles,
        sensor_config=sensor_config,
    )
def _zero_based_cell_for_x2() -> tuple[int, int]:
    return (10, 12)


def test_legality_accepts_one_valid_transformed_target() -> None:
    source = _m2_target()
    candidate = transform_long40_target(source, angle_rad=0.0)

    decision = evaluate_candidate(_m2_mother(source), transformed_target=candidate)

    assert decision == UnseenPriorCandidateDecision(
        legal=True,
        rejection_reason=None,
        accepted_target=candidate,
    )


def test_prepared_candidate_context_reuses_history_visibility_raycast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _m2_target()
    mother = _m2_mother(source, sensor_config=None)
    calls = 0

    def never_visible(occupancy, grid, *, sensor_pose):
        nonlocal calls
        del occupancy, sensor_pose
        calls += 1
        return np.zeros((grid.height, grid.width), dtype=np.bool_)

    monkeypatch.setattr(unseen_prior_module, "raycast_visibility", never_visible)
    prepared = unseen_prior_module.prepare_unseen_candidate_context(mother)
    first = evaluate_candidate(
        mother,
        transformed_target=transform_long40_target(source, angle_rad=0.0),
        prepared_context=prepared,
    )
    second = evaluate_candidate(
        mother,
        transformed_target=transform_long40_target(source, angle_rad=0.2),
        prepared_context=prepared,
    )

    assert first.legal and second.legal
    assert calls == mother.grid.history_steps


def test_legality_rejects_nonfinite_transformed_motion() -> None:
    source = _m2_target()
    candidate = transform_long40_target(source, angle_rad=0.0)
    corrupted_positions = np.array(candidate.positions, copy=True)
    corrupted_positions[0, 0] = np.nan
    object.__setattr__(candidate, "positions", corrupted_positions)

    decision = evaluate_candidate(_m2_mother(source), transformed_target=candidate)

    assert not decision.legal
    assert decision.rejection_reason == "nonfinite_or_out_of_bounds"


def test_legality_rejects_a_target_footprint_outside_the_grid() -> None:
    source = _m2_target()
    positions = np.tile(np.asarray((9.8, 0.0), dtype=np.float32), (40, 1))
    candidate = transform_long40_target(
        _m2_target(positions=positions), angle_rad=0.0
    )

    decision = evaluate_candidate(
        _m2_mother(_m2_target(positions=positions)), transformed_target=candidate
    )

    assert not decision.legal
    assert decision.rejection_reason == "nonfinite_or_out_of_bounds"


def test_obstacle_collision_rejects_static_occupancy_over_the_full_sweep() -> None:
    source = _m2_target()
    occupancy = np.zeros((20, 20), dtype=bool)
    occupancy[_zero_based_cell_for_x2()] = True

    decision = evaluate_candidate(
        _m2_mother(source, static_occupancy=occupancy),
        transformed_target=transform_long40_target(source, angle_rad=0.0),
    )

    assert not decision.legal
    assert decision.rejection_reason == "obstacle_collision"


def test_obstacle_collision_rejects_represented_occluder_over_the_full_sweep() -> None:
    source = _m2_target()
    occupancy = np.zeros((20, 20), dtype=bool)
    occupancy[_zero_based_cell_for_x2()] = True

    decision = evaluate_candidate(
        _m2_mother(source, occluder_occupancy=occupancy),
        transformed_target=transform_long40_target(source, angle_rad=0.0),
    )

    assert not decision.legal
    assert decision.rejection_reason == "obstacle_collision"


def test_obstacle_collision_rejects_represented_context_over_the_full_sweep() -> None:
    source = _m2_target()
    context = UnseenPriorContextObstacle(
        object_id="context-1",
        footprint_spec={
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": 0.3},
        },
        poses=np.column_stack(
            (
                source.positions,
                np.zeros(40, dtype=np.float32),
            )
        ),
    )

    decision = evaluate_candidate(
        _m2_mother(source, context_obstacles=(context,)),
        transformed_target=transform_long40_target(source, angle_rad=0.0),
    )

    assert not decision.legal
    assert decision.rejection_reason == "obstacle_collision"


def test_robot_history_collision_rejects_only_the_observed_eight_frames() -> None:
    source = _m2_target(
        positions=np.zeros((40, 2), dtype=np.float32)
    )

    decision = evaluate_candidate(
        _m2_mother(source),
        transformed_target=transform_long40_target(source, angle_rad=0.0),
    )

    assert not decision.legal
    assert decision.rejection_reason == "robot_history_collision"


def test_visibility_rejects_any_target_visible_in_history() -> None:
    source = _m2_target()

    decision = evaluate_candidate(
        _m2_mother(source, sensor_config=None),
        transformed_target=transform_long40_target(source, angle_rad=0.0),
    )

    assert not decision.legal
    assert decision.rejection_reason == "history_visible"


def test_future_collision_does_not_reject_and_reaches_existing_label_path() -> None:
    positions = np.tile(np.asarray((2.0, 0.0), dtype=np.float32), (40, 1))
    positions[8:] = 0.0
    source = _m2_target(positions=positions)
    candidate = transform_long40_target(source, angle_rad=0.0)
    mother = _m2_mother(source)

    decision = evaluate_candidate(mother, transformed_target=candidate)

    assert decision.legal
    assert decision.accepted_target is candidate
    robot_future = np.zeros((32, 3), dtype=np.float32)
    trajectory = LocalTrajectory(
        trajectory_id="future-collision",
        poses=robot_future,
        controls=np.zeros((32, 2), dtype=np.float32),
        swept_mask=np.zeros((20, 20), dtype=bool),
        tta_map=np.zeros((20, 20), dtype=np.float32),
        braking_map=np.zeros((20, 20), dtype=np.float32),
        centerline_map=np.zeros((20, 20), dtype=np.float32),
        task_cost=0.0,
        metadata={"pose_time_layout_version": POSE_TIME_LAYOUT_VERSION},
    )
    target_future = np.column_stack(
        (candidate.positions[8:], candidate.headings[8:])
    ).astype(np.float32)
    world = OracleWorld(
        world_id="future-collision-world",
        base_state_id="base-m2",
        static_occupancy=np.zeros((20, 20), dtype=np.float32),
        dynamic_object_trajectories={candidate.target_dynamic_object_id: target_future},
        dynamic_object_specs={candidate.target_dynamic_object_id: candidate.footprint_spec},
        occluders=(),
        blind_spot_config={},
        random_seed=0,
        metadata={"schema_version": "4.0.0"},
    )
    labels = compute_hidden_risk_gt(
        trajectory,
        world,
        hidden_object_ids=(candidate.target_dynamic_object_id,),
        robot_footprint=mother.robot_footprint,
        grid=mother.grid,
        future_dt_s=0.2,
        sigma_distance_m=0.5,
        sigma_time_s=2.0,
        near_miss_distance_m=0.35,
    )

    assert labels.collision_label == 1


class _ScriptedRng:
    def __init__(self, *, presence: float, angles: list[float]) -> None:
        self.presence = presence
        self.angles = list(angles)
        self.uniform_bounds: list[tuple[float, float]] = []

    def random(self) -> float:
        return self.presence

    def uniform(self, low: float, high: float) -> float:
        self.uniform_bounds.append((low, high))
        return self.angles.pop(0)


def _m3_config() -> UnseenPriorConfig:
    return normalize_unseen_prior_config(_raw_config())


def _result_signature(result: UnseenPriorMotherResult) -> tuple[object, ...]:
    target = None if result.realization is None else result.realization.target_motion
    return (
        result.provenance.presence_branch,
        result.provenance.outcome,
        result.provenance.attempted_angle_count,
        result.provenance.selected_angle_rad,
        dict(result.provenance.rejection_reason_counts),
        None if target is None else target.positions.tobytes(),
        None if target is None else target.headings.tobytes(),
    )


def test_presence_empty_removes_only_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _m2_target()
    mother = _m2_mother(source)
    rng = _ScriptedRng(presence=0.30, angles=[])
    monkeypatch.setattr(
        unseen_prior_module, "_stable_mother_rng", lambda mother, seed: rng
    )

    result = generate_unseen_prior_mother(mother, config=_m3_config(), seed=42)

    assert result.provenance.presence_branch == "empty"
    assert result.provenance.outcome == "empty"
    assert result.provenance.attempted_angle_count == 0
    assert result.realization is not None
    assert result.realization.target_motion is None
    assert result.realization.robot_history is mother.robot_history
    assert result.realization.static_occupancy is mother.static_occupancy
    assert result.realization.occluder_occupancy is mother.occluder_occupancy
    assert result.realization.context_obstacles is mother.context_obstacles


def test_sampler_accepts_the_first_legal_continuous_angle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _m2_target()
    rng = _ScriptedRng(presence=0.0, angles=[-np.pi])
    monkeypatch.setattr(
        unseen_prior_module, "_stable_mother_rng", lambda mother, seed: rng
    )

    result = generate_unseen_prior_mother(
        _m2_mother(source), config=_m3_config(), seed=42
    )

    assert result.provenance.outcome == "present"
    assert result.provenance.attempted_angle_count == 1
    assert result.provenance.selected_angle_rad == -np.pi
    assert rng.uniform_bounds == [(-np.pi, np.pi)]
    assert result.realization is not None
    assert result.realization.target_motion is not None


def test_sampler_retries_after_rejection_then_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positions = np.tile(np.asarray((2.0, 0.0), dtype=np.float32), (40, 1))
    positions[8:, 0] = 3.0
    source = _m2_target(positions=positions)
    occupancy = np.zeros((20, 20), dtype=bool)
    occupancy[10, 13] = True
    rng = _ScriptedRng(presence=0.0, angles=[0.0, np.pi - 0.1])
    monkeypatch.setattr(
        unseen_prior_module, "_stable_mother_rng", lambda mother, seed: rng
    )

    result = generate_unseen_prior_mother(
        _m2_mother(source, static_occupancy=occupancy),
        config=_m3_config(),
        seed=42,
    )

    assert result.provenance.outcome == "present"
    assert result.provenance.attempted_angle_count == 2
    assert result.provenance.selected_angle_rad == pytest.approx(np.pi - 0.1)
    assert dict(result.provenance.rejection_reason_counts) == {
        "obstacle_collision": 1
    }


def test_attempts_record_a_deficit_after_exactly_thirty_two_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _m2_target()
    occupancy = np.zeros((20, 20), dtype=bool)
    occupancy[_zero_based_cell_for_x2()] = True
    rng = _ScriptedRng(presence=0.0, angles=[0.0] * 32)
    monkeypatch.setattr(
        unseen_prior_module, "_stable_mother_rng", lambda mother, seed: rng
    )

    result = generate_unseen_prior_mother(
        _m2_mother(source, static_occupancy=occupancy),
        config=_m3_config(),
        seed=42,
    )

    assert result.realization is None
    assert result.provenance.presence_branch == "present"
    assert result.provenance.outcome == "no_legal_angle"
    assert result.provenance.attempted_angle_count == 32
    assert dict(result.provenance.rejection_reason_counts) == {
        "obstacle_collision": 32
    }


def test_reproducible_sampler_is_stable_across_batching_and_emits_zero_or_one() -> None:
    config = _m3_config()
    first_mother = _m2_mother(_m2_target(), mother_id="mother-a")
    second_mother = _m2_mother(_m2_target(), mother_id="mother-b")

    first = generate_unseen_prior_mother(first_mother, config=config, seed=42)
    repeated = generate_unseen_prior_mother(first_mother, config=config, seed=42)
    full_run = run_unseen_prior(
        (first_mother, second_mother), config=config, seed=42
    )
    second_only = run_unseen_prior((second_mother,), config=config, seed=42)

    assert _result_signature(first) == _result_signature(repeated)
    assert _result_signature(full_run.results[1]) == _result_signature(
        second_only.results[0]
    )
    assert isinstance(full_run, UnseenPriorRun)
    assert len(full_run.results) == 2
    assert len(full_run.realizations) <= len(full_run.results)
    assert full_run.deficit_count <= len(full_run.results)


def _m4_base_config() -> dict[str, object]:
    return {
        "schema_version": LONG40_SCHEMA_VERSION,
        "bev": {
            "range_m": 20.0,
            "resolution_m": 1.0,
            "size": 20,
            "history_steps": 8,
            "history_dt_s": 0.2,
            "future_steps": 32,
            "future_dt_s": 0.2,
        },
        "robot": {
            "length_m": 0.70,
            "width_m": 0.55,
            "inflation_m": 0.15,
        },
        "age_map": {
            "a_max_s": 5.0,
            "never_seen_value": 1.0,
            "visible_value": 0.0,
        },
    }


def _m4_risk_config() -> dict[str, float]:
    return {
        "sigma_distance_m": 0.5,
        "sigma_time_s": 2.0,
        "near_miss_distance_m": 0.35,
    }


def _m4_context(
    mother: UnseenPriorMother,
    result: UnseenPriorMotherResult,
    *,
    config: UnseenPriorConfig,
) -> Sop06SinglePublicationContext:
    realization = result.realization
    assert realization is not None
    static = np.asarray(
        realization.static_occupancy | realization.occluder_occupancy,
        dtype=np.float32,
    )
    base_state = BaseState(
        state_id=f"base-state-{mother.mother_id}",
        split=mother.split,
        recording_id=f"base-recording-{mother.mother_id}",
        dynamic_object_ids=(),
        timestamp=1.4,
        robot_history=np.array(realization.robot_history, copy=True),
        robot_state=np.asarray((0.4, 0.0), dtype=np.float32),
        visible_dynamic_object_history={},
        visible_dynamic_object_specs={},
        static_map_local=static.copy(),
        metadata={"session_id": f"base-session-{mother.mother_id}"},
    )
    grid = realization.grid
    trajectory = LocalTrajectory(
        trajectory_id=f"trajectory-{mother.mother_id}",
        poses=np.zeros((grid.future_steps, 3), dtype=np.float32),
        controls=np.zeros((grid.future_steps, 2), dtype=np.float32),
        swept_mask=np.zeros((grid.height, grid.width), dtype=np.float32),
        tta_map=np.full((grid.height, grid.width), -1.0, dtype=np.float32),
        braking_map=np.zeros((grid.height, grid.width), dtype=np.float32),
        centerline_map=np.zeros((grid.height, grid.width), dtype=np.float32),
        task_cost=0.0,
        metadata={"pose_time_layout_version": POSE_TIME_LAYOUT_VERSION},
    )
    world = OracleWorld(
        world_id=f"world-{mother.mother_id}",
        base_state_id=base_state.state_id,
        static_occupancy=static.copy(),
        dynamic_object_trajectories={},
        dynamic_object_specs={},
        occluders=(),
        blind_spot_config=(
            {} if realization.sensor_config is None else realization.sensor_config.as_dict()
        ),
        random_seed=config.seed,
        metadata={"schema_version": LONG40_SCHEMA_VERSION},
    )
    return Sop06SinglePublicationContext(
        sample_id=f"sop06-unseen-{mother.mother_id}",
        mother_id=mother.mother_id,
        split=mother.split,
        base_state=base_state,
        trajectory=trajectory,
        oracle_world=world,
        observed_static_occupancy=static.copy(),
        scene_dynamic_history={},
        scene_dynamic_specs={},
        hidden_object_ids=(),
        sensor_config=realization.sensor_config,
        target_dynamic_object_id=mother.target_motion.target_dynamic_object_id,
        target_footprint_spec=mother.target_motion.footprint_spec,
        target_history_observed=np.zeros(8, dtype=np.bool_),
        provenance={
            "base_recording_id": base_state.recording_id,
            "base_session_id": base_state.metadata["session_id"],
            "source_recording_id": mother.target_motion.source_recording_id,
            "source_session_id": mother.target_motion.source_session_id,
            "source_snippet_id": mother.target_motion.source_snippet_id,
            "seed_namespace": (
                "sop05_unseen_history_prior/" + config.config_digest
            ),
        },
    )


def _m4_generation_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[UnseenPriorConfig, dict[str, UnseenPriorMother], UnseenPriorRun]:
    moving_positions = np.tile(
        np.asarray((2.0, 0.0), dtype=np.float32), (40, 1)
    )
    moving_positions[8:, 0] = 3.0
    present_source = _m2_target(positions=moving_positions)
    rejection_occupancy = np.zeros((20, 20), dtype=bool)
    rejection_occupancy[10, 13] = True
    mothers = {
        "present": _m2_mother(
            present_source,
            mother_id="m4-present",
            static_occupancy=rejection_occupancy,
        ),
        "empty": _m2_mother(_m2_target(), mother_id="m4-empty"),
    }
    deficit_occupancy = np.zeros((20, 20), dtype=bool)
    deficit_occupancy[_zero_based_cell_for_x2()] = True
    mothers["deficit"] = _m2_mother(
        _m2_target(),
        mother_id="m4-deficit",
        static_occupancy=deficit_occupancy,
    )
    schedules = {
        "m4-present": (0.0, [0.0, np.pi - 0.1]),
        "m4-empty": (0.30, []),
        "m4-deficit": (0.0, [0.0] * 32),
    }

    def fixture_rng(mother: UnseenPriorMother, seed: int) -> _ScriptedRng:
        assert seed == 42
        presence, angles = schedules[mother.mother_id]
        return _ScriptedRng(presence=presence, angles=angles)

    monkeypatch.setattr(unseen_prior_module, "_stable_mother_rng", fixture_rng)
    config = _m3_config()
    run = run_unseen_prior(
        tuple(mothers.values()), config=config, seed=config.seed
    )
    return config, mothers, run


def test_small_end_to_end_unseen_prior_replays_and_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src.datasets.risk_dataset import build_risk_sample

    config, mothers, first_run = _m4_generation_fixture(monkeypatch)
    second_run = run_unseen_prior(
        tuple(mothers.values()), config=config, seed=config.seed
    )
    assert [_result_signature(result) for result in first_run.results] == [
        _result_signature(result) for result in second_run.results
    ]
    assert first_run.deficit_count == 1
    assert dict(first_run.rejection_reason_counts) == {"obstacle_collision": 33}

    by_mother = {result.provenance.mother_id: result for result in first_run.results}
    mothers_by_id = {mother.mother_id: mother for mother in mothers.values()}
    present = by_mother["m4-present"]
    empty = by_mother["m4-empty"]
    deficit = by_mother["m4-deficit"]
    assert present.provenance.selected_angle_rad == pytest.approx(np.pi - 0.1)
    assert present.provenance.attempted_angle_count == 2
    assert empty.realization is not None and empty.realization.target_motion is None
    assert deficit.realization is None

    publications = tuple(
        adapt_unseen_prior_realization(
            result.realization,
            context=_m4_context(
                mothers_by_id[result.provenance.mother_id], result, config=config
            ),
        )
        for result in (present, empty)
        if result.realization is not None
    )
    samples = tuple(
        build_risk_sample(
            build_sop06_single_risk_input(publication),
            base_config=_m4_base_config(),
            risk_config=_m4_risk_config(),
        )
        for publication in publications
    )
    assert len(samples) == 2
    assert {sample.metadata["label_audit"]["has_hidden_target"] for sample in samples} == {
        False,
        True,
    }

    output = tmp_path / "unseen-prior-shard"
    first_paths = write_sop06_single_risk_shard(
        publications,
        output,
        base_config=_m4_base_config(),
        risk_config=_m4_risk_config(),
    )
    resumed_paths = write_sop06_single_risk_shard(
        publications,
        output,
        base_config=_m4_base_config(),
        risk_config=_m4_risk_config(),
    )
    assert first_paths == resumed_paths
    assert all(path.is_file() for name, path in first_paths.items() if name != "directory")


def test_causal_boundary_excludes_unseen_sampling_and_future_from_model_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, mothers, run = _m4_generation_fixture(monkeypatch)
    present = next(
        result for result in run.results if result.provenance.outcome == "present"
    )
    assert present.realization is not None
    publication = adapt_unseen_prior_realization(
        present.realization,
        context=_m4_context(mothers["present"], present, config=config),
    )
    target_id = mothers["present"].target_motion.target_dynamic_object_id
    assert target_id not in publication.renderer_input.scene_dynamic_history
    assert target_id not in publication.renderer_input.scene_dynamic_specs
    assert target_id not in publication.renderer_input.scene_dynamic_history_observed
    assert all(
        token not in field.name.lower()
        for field in fields(Sop06SingleRendererInput)
        for token in ("future", "oracle", "angle", "attempt", "rejection", "collision")
    )
    assert all(
        token not in key.lower()
        for key in publication.provenance
        for token in ("future", "oracle", "angle", "attempt", "rejection", "collision")
    )

    rendered = render_sop06_single_publication(
        publication, config=_m4_base_config()
    )
    changed_world = replace(
        publication.oracle_world,
        dynamic_object_trajectories={
            target_id: publication.oracle_world.dynamic_object_trajectories[target_id]
            + np.float32(5.0)
        },
    )
    changed_publication = replace(publication, oracle_world=changed_world)
    replayed = render_sop06_single_publication(
        changed_publication, config=_m4_base_config()
    )
    np.testing.assert_array_equal(rendered.bev_history, replayed.bev_history)
    np.testing.assert_array_equal(rendered.state_channels, replayed.state_channels)


def test_combined_cap_is_shared_and_prefix_stable() -> None:
    seed = Sop06SinglePublication(
        sample_id="seed",
        mother_id="seed",
        split="train",
        regime="unseen_in_history_window",
        renderer_input=None,
        trajectory=None,
        oracle_world=None,
        hidden_object_ids=(),
        provenance={},
    )
    entries = tuple(
        replace(
            seed,
            sample_id=f"combined-sample-{index:06d}",
            mother_id=f"combined-mother-{index:06d}",
            regime=(
                "unseen_in_history_window"
                if index % 2
                else "seen_then_occluded"
            ),
        )
        for index in range(125001)
    )

    prefix_50k = coordinate_sop06_single_release(
        entries, requested_prefix_size=50000
    )
    prefix_100k = coordinate_sop06_single_release(
        tuple(reversed(entries)), requested_prefix_size=100000
    )
    prefix_125k = coordinate_sop06_single_release(
        entries, requested_prefix_size=125000
    )
    assert prefix_50k.entries == prefix_100k.entries[:50000]
    assert prefix_100k.entries == prefix_125k.entries[:100000]
    assert len(prefix_125k.entries) == 125000
    assert sum(prefix_125k.accepted_counts.values()) == 125000
    assert set(prefix_125k.accepted_counts) == {
        "seen_then_occluded",
        "unseen_in_history_window",
    }
    with pytest.raises(ValueError, match="exceeds the 125000 hard cap"):
        coordinate_sop06_single_release(entries)
