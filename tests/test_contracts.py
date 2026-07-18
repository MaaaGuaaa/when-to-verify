"""Contract-layer tests for SOP-00: schema, serialization, seeding, config."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import contracts  # noqa: E402
from src.contracts import (  # noqa: E402
    HISTORY_CHANNELS,
    INPUT_CHANNELS,
    MODEL_INPUT_CLASSES,
    POSE_TIME_LAYOUT_VERSION,
    STATE_CHANNELS,
    TRAJECTORY_CHANNELS,
    ContractError,
    assert_no_oracle_leakage,
    build_grid_spec,
    load_dataclass,
    save_dataclass,
    validate_risk_sample,
    validate_verification_sample,
)
from src.utils import seeding  # noqa: E402
from src.utils.config import (  # noqa: E402
    ConfigError,
    config_digest,
    load_config,
    validate_config,
)

_FIX = Path(__file__).resolve().parent / "fixtures"
if str(_FIX) not in sys.path:
    sys.path.insert(0, str(_FIX))

import toy_world  # noqa: E402


# --- Channels and grid ---------------------------------------------------------
def test_channel_counts_and_order():
    assert len(HISTORY_CHANNELS) == 2
    assert len(STATE_CHANNELS) == 9
    assert len(TRAJECTORY_CHANNELS) == 4
    assert len(INPUT_CHANNELS) == 15
    # No duplicate channel names anywhere.
    assert len(set(INPUT_CHANNELS)) == len(INPUT_CHANNELS)
    # Ordering contract: history block, then state block, then trajectory block.
    assert INPUT_CHANNELS[:2] == HISTORY_CHANNELS
    assert INPUT_CHANNELS[2:11] == STATE_CHANNELS
    assert INPUT_CHANNELS[11:] == TRAJECTORY_CHANNELS


def test_grid_spec_from_default_config():
    grid = build_grid_spec(load_config())
    assert (grid.height, grid.width) == (160, 160)
    assert grid.history_steps == 8
    assert grid.future_steps == 15
    assert grid.n_history_channels == 2
    assert grid.n_state_channels == 9
    assert grid.n_trajectory_channels == 4


def test_future_endpoint_time_contract_is_central_and_breaking():
    assert contracts.SCHEMA_VERSION == "3.0.0"
    assert POSE_TIME_LAYOUT_VERSION == "future_endpoints_dt_to_horizon_v1"


# --- Oracle-leakage guard ------------------------------------------------------
def test_model_input_classes_have_no_oracle_fields():
    for cls in MODEL_INPUT_CLASSES:
        assert_no_oracle_leakage(cls)


def test_dynamic_object_contract_replaces_pedestrian_only_fields():
    assert tuple(field.name for field in fields(contracts.BaseState)) == (
        "state_id",
        "split",
        "recording_id",
        "dynamic_object_ids",
        "timestamp",
        "robot_history",
        "robot_state",
        "visible_dynamic_object_history",
        "visible_dynamic_object_specs",
        "static_map_local",
        "metadata",
    )
    assert tuple(field.name for field in fields(contracts.OracleContext)) == (
        "base_state_id",
        "dynamic_object_history",
        "dynamic_object_future",
        "dynamic_object_specs",
        "metadata",
    )
    assert tuple(field.name for field in fields(contracts.OracleWorld)) == (
        "world_id",
        "base_state_id",
        "static_occupancy",
        "dynamic_object_trajectories",
        "dynamic_object_specs",
        "occluders",
        "blind_spot_config",
        "random_seed",
        "metadata",
    )


def test_oracle_leakage_guard_detects_forbidden_field():
    from dataclasses import make_dataclass

    bad = make_dataclass("BadInput", [("oracle_future", np.ndarray)])
    with pytest.raises(ContractError):
        assert_no_oracle_leakage(bad)


def test_dynamic_object_specs_accept_frozen_circle_and_rectangle_shapes():
    from src.contracts import validate_dynamic_object_spec

    validate_dynamic_object_spec(
        {
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": 0.30},
        }
    )
    validate_dynamic_object_spec(
        {
            "object_type": "carried_object",
            "footprint": {
                "kind": "rectangle",
                "length_m": 0.80,
                "width_m": 0.20,
            },
        }
    )


@pytest.mark.parametrize(
    "spec",
    [
        {
            "object_type": "vehicle",
            "footprint": {"kind": "circle", "radius_m": 0.30},
        },
        {
            "object_type": "human",
            "footprint": {"kind": "capsule", "radius_m": 0.30},
        },
        {
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": -0.30},
        },
        {
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": np.nan},
        },
        {
            "object_type": "carried_object",
            "footprint": {
                "kind": "rectangle",
                "length_m": 0.80,
                "width_m": 0.0,
            },
        },
        {
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": 0.30},
            "raw_role": "must-be-provenance-only",
        },
    ],
)
def test_dynamic_object_specs_reject_unknown_or_nonphysical_values(spec):
    from src.contracts import validate_dynamic_object_spec

    with pytest.raises(ContractError):
        validate_dynamic_object_spec(spec)


def _make_dynamic_base_state(grid):
    object_id = "recording::Helmet_1"
    return contracts.BaseState(
        state_id="base-1",
        split="train",
        recording_id="recording",
        dynamic_object_ids=(object_id,),
        timestamp=1.4,
        robot_history=np.zeros((grid.history_steps, 3), dtype=np.float32),
        robot_state=np.zeros((2,), dtype=np.float32),
        visible_dynamic_object_history={
            object_id: np.zeros((grid.history_steps, 3), dtype=np.float32)
        },
        visible_dynamic_object_specs={
            object_id: {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.30},
            }
        },
        static_map_local=None,
    )


def test_valid_dynamic_object_base_state_passes_contract_validation():
    from src.contracts import validate_base_state

    grid = build_grid_spec(load_config())
    state = _make_dynamic_base_state(grid)

    validate_base_state(state, grid)


@pytest.mark.parametrize(
    "mutation",
    [
        {"dynamic_object_ids": ("recording::Helmet_1", "recording::Helmet_1")},
        {"dynamic_object_ids": ("",)},
        {"visible_dynamic_object_specs": {}},
        {
            "visible_dynamic_object_history": {
                "recording::Helmet_1": np.zeros((8, 2), dtype=np.float32)
            }
        },
        {
            "visible_dynamic_object_history": {
                "recording::Helmet_1": np.zeros((8, 3), dtype=np.float64)
            }
        },
        {
            "visible_dynamic_object_history": {
                "recording::Helmet_1": np.full(
                    (8, 3), np.nan, dtype=np.float32
                )
            }
        },
    ],
)
def test_dynamic_object_base_state_rejects_id_spec_or_shape_drift(mutation):
    from src.contracts import validate_base_state

    grid = build_grid_spec(load_config())
    state = replace(_make_dynamic_base_state(grid), **mutation)

    with pytest.raises(ContractError):
        validate_base_state(state, grid)


def _make_dynamic_oracle_context(grid):
    object_id = "recording::LO1"
    return contracts.OracleContext(
        base_state_id="base-1",
        dynamic_object_history={
            object_id: np.zeros((grid.history_steps, 3), dtype=np.float32)
        },
        dynamic_object_future={
            object_id: np.zeros((grid.future_steps, 3), dtype=np.float32)
        },
        dynamic_object_specs={
            object_id: {
                "object_type": "carried_object",
                "footprint": {
                    "kind": "rectangle",
                    "length_m": 0.80,
                    "width_m": 0.20,
                },
            }
        },
    )


def test_valid_dynamic_object_oracle_context_passes_contract_validation():
    from src.contracts import validate_oracle_context

    grid = build_grid_spec(load_config())
    context = _make_dynamic_oracle_context(grid)

    validate_oracle_context(context, grid)


def test_dynamic_object_oracle_context_rejects_key_and_finite_value_drift():
    from src.contracts import validate_oracle_context

    grid = build_grid_spec(load_config())
    context = _make_dynamic_oracle_context(grid)
    with pytest.raises(ContractError):
        validate_oracle_context(replace(context, dynamic_object_future={}), grid)

    bad_future = {
        "recording::LO1": np.full(
            (grid.future_steps, 3), np.inf, dtype=np.float32
        )
    }
    with pytest.raises(ContractError):
        validate_oracle_context(
            replace(context, dynamic_object_future=bad_future),
            grid,
        )


def test_valid_dynamic_object_oracle_world_passes_contract_validation():
    from src.contracts import validate_oracle_world

    grid = build_grid_spec(load_config())
    world = toy_world.make_oracle_world("world-1", "base-1", 7, grid)

    validate_oracle_world(world, grid)


def test_dynamic_object_oracle_world_rejects_dtype_and_nonfinite_occupancy():
    from src.contracts import validate_oracle_world

    grid = build_grid_spec(load_config())
    world = toy_world.make_oracle_world("world-1", "base-1", 7, grid)
    invalid_occupancies = (
        np.zeros((grid.height, grid.width), dtype=np.float64),
        np.full((grid.height, grid.width), np.nan, dtype=np.float32),
    )
    for occupancy in invalid_occupancies:
        with pytest.raises(ContractError):
            validate_oracle_world(
                replace(world, static_occupancy=occupancy),
                grid,
            )


# --- Validators ----------------------------------------------------------------
def test_valid_samples_pass_validation():
    grid = build_grid_spec(load_config())
    validate_risk_sample(toy_world.make_risk_sample(grid), grid)
    validate_verification_sample(toy_world.make_verification_sample(grid), grid)


def test_risk_sample_rejects_wrong_shape():
    grid = build_grid_spec(load_config())
    sample = toy_world.make_risk_sample(grid)
    broken = sample.__class__(
        **{**sample.__dict__, "robot_state": np.zeros((3,), dtype=np.float32)}
    )
    with pytest.raises(ContractError):
        validate_risk_sample(broken, grid)


def test_verification_useful_must_match_value_sign():
    grid = build_grid_spec(load_config())
    sample = toy_world.make_verification_sample(grid)
    inconsistent = sample.__class__(
        **{**sample.__dict__, "value_target": -0.1, "useful_target": 1}
    )
    with pytest.raises(ContractError):
        validate_verification_sample(inconsistent, grid)


# --- Serialization round trips -------------------------------------------------
def _assert_array_roundtrip(original, restored):
    for f in original.__dataclass_fields__:
        a = getattr(original, f)
        b = getattr(restored, f)
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype
            assert a.shape == b.shape
            assert np.array_equal(a, b)
        elif isinstance(a, dict) and a and all(isinstance(v, np.ndarray) for v in a.values()):
            assert a.keys() == b.keys()
            for key in a:
                assert np.array_equal(a[key], b[key])
                assert a[key].dtype == b[key].dtype
        else:
            assert a == b


@pytest.mark.parametrize(
    "builder",
    [
        toy_world.make_risk_sample,
        toy_world.make_verification_sample,
        lambda grid: toy_world.make_base_state("toy_bs_0", grid),
        lambda grid: toy_world.make_local_trajectory("toy_traj_0", 0.6, 0.0, grid),
        lambda grid: toy_world.make_oracle_world("toy_world_0", "toy_bs_0", 7, grid),
    ],
)
def test_dataclass_roundtrip(tmp_path, builder):
    grid = build_grid_spec(load_config())
    obj = builder(grid)
    path = save_dataclass(obj, tmp_path / "obj.npz")
    restored = load_dataclass(path)
    assert type(restored) is type(obj)
    _assert_array_roundtrip(obj, restored)


def test_serialization_uses_no_object_arrays(tmp_path):
    grid = build_grid_spec(load_config())
    path = save_dataclass(toy_world.make_oracle_world("w", "bs", 3, grid), tmp_path / "w.npz")
    # allow_pickle=False must succeed; object arrays would raise here.
    with np.load(path, allow_pickle=False) as data:
        assert "meta_json" in data.files


@pytest.mark.parametrize("old_schema_version", ["1.0.0", "2.0.0"])
def test_load_dataclass_rejects_pre_v3_schema_artifacts(
    tmp_path, old_schema_version
):
    grid = build_grid_spec(load_config())
    path = save_dataclass(_make_dynamic_base_state(grid), tmp_path / "base.npz")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    metadata = json.loads(str(arrays["meta_json"]))
    metadata["schema_version"] = old_schema_version
    arrays["meta_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    with path.open("wb") as handle:
        np.savez(handle, **arrays)

    with pytest.raises(ContractError, match="schema_version"):
        load_dataclass(path)


def test_contract_validation_cli_checks_dynamic_object_artifacts():
    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "00_validate_contracts.py"),
            "--config",
            str(_ROOT / "configs" / "base.yaml"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[ok] dynamic-object base/oracle artifacts round-trip and validate" in result.stdout


# --- Seeding -------------------------------------------------------------------
def test_derive_seed_is_deterministic_and_order_sensitive():
    assert seeding.derive_seed(42, "a", "b") == seeding.derive_seed(42, "a", "b")
    assert seeding.derive_seed(42, "a", "b") != seeding.derive_seed(42, "b", "a")
    assert seeding.derive_seed(1, "a") != seeding.derive_seed(2, "a")


def test_make_rng_reproducible():
    a = seeding.make_rng(42, "risk", "train").standard_normal(8)
    b = seeding.make_rng(42, "risk", "train").standard_normal(8)
    c = seeding.make_rng(7, "risk", "train").standard_normal(8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sample_id_stable_regardless_of_call_order():
    first = seeding.sample_id("train", "rec1", "bs1", "traj1", "collision", 42)
    second = seeding.sample_id("train", "rec1", "bs1", "traj1", "collision", 42)
    assert first == second
    other = seeding.sample_id("test", "rec1", "bs1", "traj1", "collision", 42)
    assert first != other


# --- Config --------------------------------------------------------------------
def test_base_config_loads_and_validates():
    cfg = load_config(_ROOT / "configs" / "base.yaml")
    assert cfg["schema_version"] == contracts.SCHEMA_VERSION
    assert contracts.SCHEMA_VERSION == "3.0.0"
    assert cfg["bev"]["size"] == 160
    assert cfg["scenario_bank"]["reject_cost"] == 0.20
    assert "pedestrian" not in cfg
    assert cfg["dynamic_objects"] == {
        "human": {
            "radius_m": 0.30,
            "carrier_radius_m": 0.45,
            "min_speed_mps": 0.30,
            "max_speed_mps": 2.00,
            "max_acceleration_mps2": 2.50,
        },
        "carried_object": {
            "fallback_length_m": 0.80,
            "fallback_width_m": 0.20,
            "min_speed_mps": 0.05,
            "max_speed_mps": 2.00,
            "max_acceleration_mps2": 2.50,
        },
        "unknown_dynamic": {
            "fallback_radius_m": 0.50,
            "min_speed_mps": 0.05,
            "max_speed_mps": 2.00,
            "max_acceleration_mps2": 2.50,
        },
        "marker_geometry": {
            "extent_quantile": 0.95,
            "minimum_valid_frames": 20,
            "min_extent_m": 0.05,
            "max_extent_m": 3.00,
        },
    }


def test_unknown_config_key_rejected():
    with pytest.raises(ConfigError):
        validate_config({"bev": {"unexpected_key": 1}})
    with pytest.raises(ConfigError):
        validate_config({"totally_unknown": True})


@pytest.mark.parametrize("old_schema_version", ["1.0.0", "2.0.0"])
def test_load_config_rejects_schema_version_drift(
    tmp_path, old_schema_version
):
    config_path = tmp_path / "old-schema.yaml"
    config_path.write_text(
        f'schema_version: "{old_schema_version}"\n', encoding="utf-8"
    )

    with pytest.raises(ConfigError, match="schema_version"):
        load_config(config_path)


def test_config_digest_is_stable_and_sensitive():
    cfg = load_config(_ROOT / "configs" / "base.yaml")
    assert config_digest(cfg) == config_digest(dict(cfg))
    mutated = load_config(_ROOT / "configs" / "base.yaml")
    mutated["seed"] = 999
    assert config_digest(cfg) != config_digest(mutated)
