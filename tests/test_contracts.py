"""Contract-layer tests for SOP-00: schema, serialization, seeding, config."""

from __future__ import annotations

import sys
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


# --- Oracle-leakage guard ------------------------------------------------------
def test_model_input_classes_have_no_oracle_fields():
    for cls in MODEL_INPUT_CLASSES:
        assert_no_oracle_leakage(cls)


def test_oracle_leakage_guard_detects_forbidden_field():
    from dataclasses import make_dataclass

    bad = make_dataclass("BadInput", [("oracle_future", np.ndarray)])
    with pytest.raises(ContractError):
        assert_no_oracle_leakage(bad)


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
    assert cfg["bev"]["size"] == 160
    assert cfg["scenario_bank"]["reject_cost"] == 0.20


def test_unknown_config_key_rejected():
    with pytest.raises(ConfigError):
        validate_config({"bev": {"unexpected_key": 1}})
    with pytest.raises(ConfigError):
        validate_config({"totally_unknown": True})


def test_config_digest_is_stable_and_sensitive():
    cfg = load_config(_ROOT / "configs" / "base.yaml")
    assert config_digest(cfg) == config_digest(dict(cfg))
    mutated = load_config(_ROOT / "configs" / "base.yaml")
    mutated["seed"] = 999
    assert config_digest(cfg) != config_digest(mutated)
