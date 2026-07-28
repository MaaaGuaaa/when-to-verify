"""Focused M1 contract tests for the SOP05 seen-then-occluded prior."""

from __future__ import annotations

from dataclasses import fields, replace
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from src.contracts import GridSpec
from src.geometry import CircleFootprint, world_to_grid, wrap_angle
from src.generation.sop05_seen_prior import (
    SEEN_PRIOR_ANGLE_PRIOR_VERSION,
    SEEN_PRIOR_CONFIG_VERSION,
    SEEN_PRIOR_GENERATOR_VERSION,
    SEEN_PRIOR_M2_ENVIRONMENT_GATE_VERSION,
    SEEN_PRIOR_M2_REJECTION_REASONS,
    SEEN_PRIOR_M3_FAILURE_REASON,
    SEEN_PRIOR_M3_SELECTION_VERSION,
    SEEN_PRIOR_SAMPLING_VERSION,
    SEEN_PRIOR_TRANSFORM_VERSION,
    SeenPriorConfig,
    SeenPriorContextSweep,
    SeenPriorEnvironment,
    SeenPriorFailure,
    SeenPriorResult,
    SeenPriorSource,
    draw_seen_prior_angle_attempts,
    generate_seen_prior,
    load_seen_prior_config,
    make_seen_prior_rng,
    seen_prior_mother_seed,
    transform_seen_prior_future,
    validate_seen_prior_future_environment,
    validate_seen_prior_source,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "sop05_seen_prior.yaml"


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _config_payload() -> dict[str, object]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "seen_prior.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _source() -> SeenPriorSource:
    history = np.zeros((8, 3), dtype=np.float32)
    history[:, 0] = np.linspace(-0.7, 0.0, 8, dtype=np.float32)
    history[:, 1] = np.linspace(0.3, -0.1, 8, dtype=np.float32)
    history[:, 2] = np.linspace(-0.2, 0.15, 8, dtype=np.float32)
    future = np.zeros((32, 3), dtype=np.float32)
    future[:, 0] = np.float32(0.1) + np.arange(32, dtype=np.float32) * np.float32(0.08)
    future[:, 1] = np.float32(-0.1) + np.arange(32, dtype=np.float32) * np.float32(0.01)
    future[:, 2] = np.float32(0.2) + np.arange(32, dtype=np.float32) * np.float32(0.02)
    return SeenPriorSource(
        mother_id="mother-001",
        split="train",
        source_collection_identity="long40-publication-digest",
        history_regime="seen_then_occluded",
        target_history_poses=history,
        target_current_pose=np.array(history[7], dtype=np.float32, copy=True),
        target_future_poses=future,
        target_visibility_history=np.array(
            (True, True, False, False, False, False, False, False), dtype=np.bool_
        ),
    )


def _config() -> SeenPriorConfig:
    return load_seen_prior_config(CONFIG_PATH)


def _candidate(*, theta_rad: float = 0.0):
    return transform_seen_prior_future(_source(), _config(), theta_rad=theta_rad)


def _environment(
    *,
    static_occupancy: np.ndarray | None = None,
    occluder_occupancy: np.ndarray | None = None,
    context_sweeps: tuple[SeenPriorContextSweep, ...] = (),
) -> SeenPriorEnvironment:
    grid = GridSpec(
        height=100,
        width=100,
        history_steps=8,
        future_steps=32,
        resolution_m=0.1,
    )
    return SeenPriorEnvironment(
        grid=grid,
        target_footprint=CircleFootprint(0.05),
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
        context_sweeps=context_sweeps,
    )


def _occupied_at(pose: np.ndarray, *, grid: GridSpec) -> np.ndarray:
    occupancy = np.zeros((grid.height, grid.width), dtype=bool)
    index = world_to_grid(np.asarray(pose[:2], dtype=np.float64), grid)
    occupancy[tuple(index)] = True
    return occupancy


def test_minimal_continuous_config_contract_is_exact() -> None:
    raw = _config_payload()
    config = _config()

    assert SEEN_PRIOR_CONFIG_VERSION == "sop05_seen_prior_config_v1"
    assert SEEN_PRIOR_GENERATOR_VERSION == "sop05_seen_occluded_scenario_v1"
    assert SEEN_PRIOR_ANGLE_PRIOR_VERSION == "truncated_normal_zero_mean_v1"
    assert SEEN_PRIOR_SAMPLING_VERSION == "sha256_pcg64_per_mother_v1"
    assert SEEN_PRIOR_TRANSFORM_VERSION == "future32_rigid_rotation_about_history7_v1"
    assert [field.name for field in fields(SeenPriorConfig)] == [
        "schema_version",
        "required_history_regime",
        "history_steps",
        "current_index",
        "future_steps",
        "dt_s",
        "angle_prior_kind",
        "mean_rad",
        "sigma_rad",
        "min_rad_inclusive",
        "max_rad_exclusive",
        "seed_namespace",
        "max_attempts_per_mother",
        "max_variants_per_mother",
        "digest",
    ]
    assert SeenPriorConfig.__dataclass_params__.frozen
    assert set(raw) == {
        "schema_version",
        "required_history_regime",
        "trajectory",
        "angle_prior",
        "sampling",
    }
    assert set(raw["angle_prior"]) == {
        "kind",
        "mean_rad",
        "sigma_rad",
        "min_rad_inclusive",
        "max_rad_exclusive",
    }
    assert config.schema_version == "4.0.0"
    assert config.required_history_regime == "seen_then_occluded"
    assert (config.history_steps, config.current_index, config.future_steps, config.dt_s) == (
        8,
        7,
        32,
        0.2,
    )
    assert config.angle_prior_kind == "truncated_normal"
    assert config.mean_rad == 0.0
    assert config.sigma_rad == pytest.approx(math.pi / 12.0, abs=0.0)
    assert (config.min_rad_inclusive, config.max_rad_exclusive) == (-math.pi, math.pi)
    assert config.max_attempts_per_mother == 32
    assert config.max_variants_per_mother == 1
    assert config.digest == _canonical_digest(raw)
    assert "angle_count" not in raw["angle_prior"]
    assert "classification" not in raw
    assert "publication" not in raw


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.__setitem__("legacy", "73-angle-grid"),
        lambda payload: payload["trajectory"].__setitem__("history_steps", 7),
        lambda payload: payload["trajectory"].__setitem__("dt_s", 0.1),
        lambda payload: payload["angle_prior"].__setitem__("kind", "categorical"),
        lambda payload: payload["angle_prior"].__setitem__("mean_rad", 0.01),
        lambda payload: payload["angle_prior"].__setitem__("sigma_rad", 0.0),
        lambda payload: payload["angle_prior"].__setitem__("sigma_rad", 0.5),
        lambda payload: payload["angle_prior"].__setitem__("sigma_rad", float("nan")),
        lambda payload: payload["angle_prior"].__setitem__("angle_count", 73),
        lambda payload: payload["sampling"].__setitem__("seed_namespace", "other"),
        lambda payload: payload["sampling"].__setitem__("max_attempts_per_mother", 31),
        lambda payload: payload["sampling"].__setitem__("max_variants_per_mother", 2),
    ),
)
def test_config_rejects_nonminimal_or_drifted_contract(
    tmp_path: Path, mutate: object
) -> None:
    payload = copy.deepcopy(_config_payload())
    assert callable(mutate)
    mutate(payload)

    with pytest.raises(ValueError):
        load_seen_prior_config(_write_config(tmp_path, payload))


def test_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text('schema_version: "4.0.0"\nschema_version: "4.0.0"\n', encoding="utf-8")

    with pytest.raises(ValueError):
        load_seen_prior_config(path)


def test_source_eligibility_requires_authenticated_seen_then_occluded_long40() -> None:
    source = _source()

    validate_seen_prior_source(source, _config())

    current_visible = replace(
        source,
        target_visibility_history=np.array(
            (True, False, False, False, False, False, False, True), dtype=np.bool_
        ),
    )
    validate_seen_prior_source(current_visible, _config())


@pytest.mark.parametrize(
    "mutate",
    (
        lambda source: replace(source, history_regime="unseen_in_history_window"),
        lambda source: replace(
            source, target_visibility_history=np.zeros(8, dtype=np.bool_)
        ),
        lambda source: replace(
            source,
            target_visibility_history=np.array(
                (False, True, False, False, False, False, False, False), dtype=np.bool_
            ),
        ),
        lambda source: replace(source, target_future_poses=np.zeros((31, 3), dtype=np.float32)),
        lambda source: replace(source, target_visibility_history=np.zeros(8, dtype=np.uint8)),
        lambda source: replace(
            source,
            target_current_pose=np.array((1.0, 0.0, 0.0), dtype=np.float32),
        ),
        lambda source: replace(
            source,
            target_history_poses=np.full((8, 3), np.nan, dtype=np.float32),
        ),
    ),
)
def test_source_validation_rejects_invalid_eligibility_or_long40_arrays(
    mutate: object,
) -> None:
    assert callable(mutate)
    source = mutate(_source())

    with pytest.raises(ValueError):
        validate_seen_prior_source(source, _config())


def test_per_mother_rng_is_sha256_seeded_reproducible_and_order_independent() -> None:
    config = _config()
    kwargs = {
        "dataset_seed": 23,
        "split": "train",
        "source_collection_identity": "long40-publication-digest",
        "mother_id": "mother-001",
    }
    first = draw_seen_prior_angle_attempts(config, **kwargs)
    repeated = draw_seen_prior_angle_attempts(config, **kwargs)
    later_kwargs = dict(kwargs)
    later_kwargs["mother_id"] = "mother-002"
    later_mother = draw_seen_prior_angle_attempts(config, **later_kwargs)
    seed = seen_prior_mother_seed(config, **kwargs)
    expected_rng = np.random.default_rng(seed)
    expected_raw = tuple(
        float(expected_rng.normal(loc=0.0, scale=math.pi / 12.0)) for _ in range(32)
    )

    assert first == repeated
    assert first != later_mother
    assert len(first) == config.max_attempts_per_mother
    assert all(
        value is None or -math.pi <= value < math.pi for value in first
    )
    assert first == tuple(
        value if -math.pi <= value < math.pi else None for value in expected_raw
    )
    np.testing.assert_array_equal(
        make_seen_prior_rng(config, **kwargs).normal(size=5),
        make_seen_prior_rng(config, **kwargs).normal(size=5),
    )


def test_out_of_range_draw_consumes_its_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGenerator:
        def __init__(self) -> None:
            self.values = iter((4.0, 0.25, -4.0))

        def normal(self, *, loc: float, scale: float) -> float:
            assert (loc, scale) == (0.0, _config().sigma_rad)
            return next(self.values)

    monkeypatch.setattr(
        "src.generation.sop05_seen_prior.make_seen_prior_rng",
        lambda *args, **kwargs: FakeGenerator(),
    )

    values = draw_seen_prior_angle_attempts(
        _config(),
        dataset_seed=1,
        split="train",
        source_collection_identity="collection",
        mother_id="mother",
        attempts=3,
    )

    assert values == (None, 0.25, None)


def test_future_rotation_changes_only_all_32_future_poses_about_index_7() -> None:
    config = _config()
    source = _source()
    source_history = source.target_history_poses.tobytes(order="C")
    source_current = source.target_current_pose.tobytes(order="C")
    source_future = source.target_future_poses.tobytes(order="C")
    theta = math.pi / 2.0

    transformed = transform_seen_prior_future(source, config, theta_rad=theta)
    pivot = source.target_history_poses[7, :2].astype(np.float64)
    rotation = np.asarray(((0.0, -1.0), (1.0, 0.0)), dtype=np.float64)
    expected_xy = (
        (source.target_future_poses[:, :2].astype(np.float64) - pivot) @ rotation.T
        + pivot
    )
    expected_yaw = wrap_angle(source.target_future_poses[:, 2].astype(np.float64) + theta)

    assert source.target_history_poses.tobytes(order="C") == source_history
    assert source.target_current_pose.tobytes(order="C") == source_current
    assert source.target_future_poses.tobytes(order="C") == source_future
    assert transformed.history_poses.tobytes(order="C") == source_history
    assert transformed.current_pose.tobytes(order="C") == source_current
    assert transformed.future_poses.shape == (32, 3)
    assert transformed.future_poses.dtype == source.target_future_poses.dtype
    assert transformed.future_poses.flags.c_contiguous
    assert not np.shares_memory(transformed.history_poses, source.target_history_poses)
    assert not np.shares_memory(transformed.current_pose, source.target_current_pose)
    assert not np.shares_memory(transformed.future_poses, source.target_future_poses)
    np.testing.assert_allclose(transformed.future_poses[:, :2], expected_xy, rtol=0.0, atol=2e-7)
    np.testing.assert_allclose(transformed.future_poses[:, 2], expected_yaw, rtol=0.0, atol=2e-7)
    assert transformed.theta_rad == theta


def test_abrupt_boundary_turn_is_accepted_without_smoothing_or_dynamics_checks() -> None:
    source = _source()
    config = _config()
    transformed = transform_seen_prior_future(source, config, theta_rad=-2.9)
    version_text = " ".join(
        (
            SEEN_PRIOR_CONFIG_VERSION,
            SEEN_PRIOR_GENERATOR_VERSION,
            SEEN_PRIOR_TRANSFORM_VERSION,
        )
    )

    assert transformed.future_poses.shape == (32, 3)
    assert np.isfinite(transformed.future_poses).all()
    assert all(
        token not in version_text.lower()
        for token in ("speed", "acceleration", "smoothing")
    )


@pytest.mark.parametrize("theta_rad", (math.pi, -math.pi - 1e-9, float("nan")))
def test_transform_rejects_angles_outside_the_continuous_prior_support(
    theta_rad: float,
) -> None:
    with pytest.raises(ValueError):
        transform_seen_prior_future(_source(), _config(), theta_rad=theta_rad)


def test_m2_environment_gate_has_only_the_five_focused_rejection_reasons() -> None:
    assert SEEN_PRIOR_M2_ENVIRONMENT_GATE_VERSION == "future32_environment_legality_v1"
    assert SEEN_PRIOR_M2_REJECTION_REASONS == (
        "future_nonfinite",
        "future_out_of_bounds",
        "future_static_collision",
        "future_occluder_collision",
        "future_context_collision",
    )

    candidate = _candidate()
    environment = _environment()
    static = _occupied_at(candidate.future_poses[-1], grid=environment.grid)
    occluders = _occupied_at(candidate.future_poses[-1], grid=environment.grid)
    context = SeenPriorContextSweep(
        context_object_id="protected-context",
        footprint=CircleFootprint(0.05),
        poses=np.vstack((candidate.current_pose, candidate.future_poses)),
    )

    nonfinite_future = candidate.future_poses.copy()
    nonfinite_future[0, 0] = np.nan
    assert validate_seen_prior_future_environment(
        replace(candidate, future_poses=nonfinite_future),
        environment=_environment(static_occupancy=static),
    ).reason == "future_nonfinite"

    out_of_bounds_future = candidate.future_poses.copy()
    out_of_bounds_future[-1, 0] = np.float32(5.0)
    assert validate_seen_prior_future_environment(
        replace(candidate, future_poses=out_of_bounds_future),
        environment=_environment(static_occupancy=static),
    ).reason == "future_out_of_bounds"

    assert validate_seen_prior_future_environment(
        candidate,
        environment=_environment(
            static_occupancy=static,
            occluder_occupancy=occluders,
            context_sweeps=(context,),
        ),
    ).reason == "future_static_collision"
    assert validate_seen_prior_future_environment(
        candidate,
        environment=_environment(
            occluder_occupancy=occluders,
            context_sweeps=(context,),
        ),
    ).reason == "future_occluder_collision"
    context_only = SeenPriorContextSweep(
        context_object_id="protected-context",
        footprint=CircleFootprint(0.05),
        poses=np.vstack((candidate.current_pose, candidate.future_poses)),
    )
    context_result = validate_seen_prior_future_environment(
        candidate,
        environment=_environment(context_sweeps=(context_only,)),
    )
    assert not context_result.accepted
    assert context_result.reason == "future_context_collision"


def test_m2_checks_all_32_future_steps_and_continuous_target_sweep() -> None:
    candidate = _candidate()
    environment = _environment()
    static_at_final_step = _occupied_at(
        candidate.future_poses[31],
        grid=environment.grid,
    )

    result = validate_seen_prior_future_environment(
        candidate,
        environment=_environment(static_occupancy=static_at_final_step),
    )

    assert not result.accepted
    assert result.reason == "future_static_collision"


def test_m2_does_not_gate_on_robot_collision_future_visibility_or_hard_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.generation.continuous_collision as continuous_collision

    calls = 0

    def fail_if_called(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("M2 must not call the robot collision authority")

    monkeypatch.setattr(
        continuous_collision,
        "compute_continuous_collision_evidence",
        fail_if_called,
    )
    candidate = _candidate(theta_rad=-2.9)
    result = validate_seen_prior_future_environment(
        candidate,
        environment=_environment(),
    )

    assert result.accepted
    assert result.reason is None
    assert calls == 0
    environment_fields = {field.name for field in fields(SeenPriorEnvironment)}
    assert not {
        "robot_footprint",
        "robot_poses",
        "future_visibility",
        "dynamics",
    }.intersection(environment_fields)


def test_m2_rejects_invalid_non_robot_environment_inputs() -> None:
    candidate = _candidate()
    invalid_context = SeenPriorContextSweep(
        context_object_id="context",
        footprint=CircleFootprint(0.1),
        poses=np.zeros((32, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError):
        validate_seen_prior_future_environment(
            candidate,
            environment=_environment(context_sweeps=(invalid_context,)),
        )


def test_m3_selects_first_legal_attempt_without_computing_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.generation.sop05_seen_prior as seen_prior

    gate_calls: list[float] = []
    angles = (0.1, -0.2) + (0.0,) * 30

    def gate(
        candidate: object,
        *,
        environment: object,
    ) -> object:
        assert hasattr(candidate, "theta_rad")
        gate_calls.append(float(candidate.theta_rad))
        if len(gate_calls) == 1:
            return seen_prior.SeenPriorEnvironmentValidation(
                False,
                "future_static_collision",
            )
        return seen_prior.SeenPriorEnvironmentValidation(True, None)

    monkeypatch.setattr(
        seen_prior,
        "draw_seen_prior_angle_attempts",
        lambda *args, **kwargs: angles,
    )
    monkeypatch.setattr(seen_prior, "validate_seen_prior_future_environment", gate)
    source = _source()

    result = generate_seen_prior(source, _environment(), _config(), dataset_seed=11)

    assert isinstance(result, SeenPriorResult)
    assert result.accepted_attempt == 2
    assert result.theta_rad == -0.2
    assert gate_calls == [0.1, -0.2]
    assert result.history_poses.tobytes(order="C") == source.target_history_poses.tobytes(order="C")
    assert result.current_pose.tobytes(order="C") == source.target_current_pose.tobytes(order="C")
    assert [field.name for field in fields(SeenPriorResult)] == [
        "mother_id",
        "history_poses",
        "current_pose",
        "future_poses",
        "theta_rad",
        "accepted_attempt",
    ]


def test_m3_full_block_stops_after_exactly_32_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.generation.sop05_seen_prior as seen_prior

    gate_calls = 0

    def always_blocked(*args: object, **kwargs: object) -> object:
        nonlocal gate_calls
        gate_calls += 1
        return seen_prior.SeenPriorEnvironmentValidation(
            False,
            "future_occluder_collision",
        )

    monkeypatch.setattr(
        seen_prior,
        "draw_seen_prior_angle_attempts",
        lambda *args, **kwargs: (0.0,) * 32,
    )
    monkeypatch.setattr(seen_prior, "validate_seen_prior_future_environment", always_blocked)

    result = generate_seen_prior(_source(), _environment(), _config(), dataset_seed=11)

    assert isinstance(result, SeenPriorFailure)
    assert result.reason == SEEN_PRIOR_M3_FAILURE_REASON
    assert result.attempts == 32
    assert dict(result.rejection_counts) == {"future_occluder_collision": 32}
    assert gate_calls == 32


def test_m3_result_is_reproducible_for_the_same_source_and_seed() -> None:
    source = _source()
    first = generate_seen_prior(source, _environment(), _config(), dataset_seed=29)
    repeated = generate_seen_prior(source, _environment(), _config(), dataset_seed=29)

    assert isinstance(first, SeenPriorResult)
    assert isinstance(repeated, SeenPriorResult)
    assert first.mother_id == repeated.mother_id == source.mother_id
    assert first.accepted_attempt == repeated.accepted_attempt
    assert first.theta_rad == repeated.theta_rad
    assert first.future_poses.tobytes(order="C") == repeated.future_poses.tobytes(order="C")


def test_seen_prior_small_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.generation.sop05_seen_prior as seen_prior

    base = _source()
    retry = replace(base, mother_id="retry")
    blocked = replace(base, mother_id="blocked")
    safe = replace(base, mother_id="safe")
    retry_zero = transform_seen_prior_future(retry, _config(), theta_rad=0.0)
    retry_environment = _environment(
        static_occupancy=_occupied_at(
            retry_zero.future_poses[-1],
            grid=_environment().grid,
        )
    )
    blocked_environment = _environment(
        static_occupancy=np.ones((100, 100), dtype=bool)
    )
    schedules = {
        "retry": (0.0, 0.7) + (0.0,) * 30,
        "blocked": (0.0,) * 32,
        "safe": (0.0,) * 32,
    }

    monkeypatch.setattr(
        seen_prior,
        "draw_seen_prior_angle_attempts",
        lambda config, **kwargs: schedules[kwargs["mother_id"]],
    )

    inputs = (
        (retry, retry_environment),
        (blocked, blocked_environment),
        (safe, _environment()),
    )
    first = tuple(
        generate_seen_prior(source, environment, _config(), dataset_seed=7)
        for source, environment in inputs
    )
    repeated = tuple(
        generate_seen_prior(source, environment, _config(), dataset_seed=7)
        for source, environment in inputs
    )
    accepted = tuple(result for result in first if isinstance(result, SeenPriorResult))
    failures = tuple(result for result in first if isinstance(result, SeenPriorFailure))
    source_by_mother = {source.mother_id: source for source, _ in inputs}

    assert len(accepted) + len(failures) == len(inputs)
    assert len(accepted) == 2
    assert len(failures) == 1
    assert failures[0].mother_id == "blocked"
    assert failures[0].attempts == 32
    assert next(result for result in accepted if result.mother_id == "retry").accepted_attempt == 2
    assert len({result.mother_id for result in accepted}) == len(accepted)
    for result in accepted:
        source = source_by_mother[result.mother_id]
        assert result.history_poses.tobytes(order="C") == source.target_history_poses.tobytes(order="C")
        assert result.current_pose.tobytes(order="C") == source.target_current_pose.tobytes(order="C")
        assert result.future_poses.shape == (32, 3)
        assert -math.pi <= result.theta_rad < math.pi
    assert len(first) == len(repeated)
    for initial, rerun in zip(first, repeated):
        assert type(initial) is type(rerun)
        if isinstance(initial, SeenPriorResult):
            assert isinstance(rerun, SeenPriorResult)
            assert initial.mother_id == rerun.mother_id
            assert initial.accepted_attempt == rerun.accepted_attempt
            assert initial.theta_rad == rerun.theta_rad
            assert initial.future_poses.tobytes(order="C") == rerun.future_poses.tobytes(order="C")
        else:
            assert isinstance(rerun, SeenPriorFailure)
            assert initial.mother_id == rerun.mother_id
            assert dict(initial.rejection_counts) == dict(rerun.rejection_counts)


def test_import_has_no_config_or_rng_side_effect() -> None:
    script = """
from pathlib import Path
import numpy as np

def fail(*args, **kwargs):
    raise AssertionError('module import attempted an impure operation')

Path.read_text = fail
Path.open = fail
np.random.default_rng = fail
import src.generation.sop05_seen_prior
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
