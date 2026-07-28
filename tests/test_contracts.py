"""Contract tests for the active Schema 4 SOP-01 through SOP-06 surface."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields, make_dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from src import contracts
from src.contracts import (
    ARRAY_DTYPE,
    BaseState,
    ContractError,
    GridSpec,
    LocalTrajectory,
    OracleContext,
    OracleWorld,
    POSE_TIME_LAYOUT_VERSION,
    assert_no_oracle_leakage,
    build_grid_spec,
    load_dataclass,
    save_dataclass,
    validate_base_state,
    validate_dynamic_object_spec,
    validate_oracle_context,
    validate_oracle_world,
)
from src.utils import seeding
from src.utils.config import (
    ConfigError,
    config_digest,
    load_config,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _object_spec() -> dict[str, object]:
    return {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.30},
    }


def _base_state(grid: GridSpec) -> BaseState:
    object_id = "recording::human"
    return BaseState(
        state_id="base-1",
        split="train",
        recording_id="recording",
        dynamic_object_ids=(object_id,),
        timestamp=1.4,
        robot_history=np.zeros((grid.history_steps, 3), dtype=ARRAY_DTYPE),
        robot_state=np.zeros((2,), dtype=ARRAY_DTYPE),
        visible_dynamic_object_history={
            object_id: np.zeros((grid.history_steps, 3), dtype=ARRAY_DTYPE)
        },
        visible_dynamic_object_specs={object_id: _object_spec()},
        static_map_local=np.zeros((grid.height, grid.width), dtype=ARRAY_DTYPE),
    )


def _oracle_context(grid: GridSpec) -> OracleContext:
    object_id = "recording::human"
    return OracleContext(
        base_state_id="base-1",
        dynamic_object_history={
            object_id: np.zeros((grid.history_steps, 3), dtype=ARRAY_DTYPE)
        },
        dynamic_object_future={
            object_id: np.zeros((grid.future_steps, 3), dtype=ARRAY_DTYPE)
        },
        dynamic_object_specs={object_id: _object_spec()},
    )


def _oracle_world(grid: GridSpec) -> OracleWorld:
    object_id = "recording::human"
    return OracleWorld(
        world_id="world-1",
        base_state_id="base-1",
        static_occupancy=np.zeros((grid.height, grid.width), dtype=ARRAY_DTYPE),
        dynamic_object_trajectories={
            object_id: np.zeros((grid.future_steps, 3), dtype=ARRAY_DTYPE)
        },
        dynamic_object_specs={object_id: _object_spec()},
        occluders=(),
        blind_spot_config={},
        random_seed=7,
    )


def _local_trajectory(grid: GridSpec) -> LocalTrajectory:
    return LocalTrajectory(
        trajectory_id="trajectory-1",
        poses=np.zeros((grid.future_steps, 3), dtype=ARRAY_DTYPE),
        controls=np.zeros((grid.future_steps, 2), dtype=ARRAY_DTYPE),
        swept_mask=np.zeros((grid.height, grid.width), dtype=ARRAY_DTYPE),
        tta_map=np.full((grid.height, grid.width), -1.0, dtype=ARRAY_DTYPE),
        braking_map=np.zeros((grid.height, grid.width), dtype=ARRAY_DTYPE),
        centerline_map=np.zeros((grid.height, grid.width), dtype=ARRAY_DTYPE),
        task_cost=0.0,
    )


def test_grid_spec_and_time_layout_are_current_long40_contract() -> None:
    grid = build_grid_spec(load_config(ROOT / "configs/base.yaml"))

    assert tuple(field.name for field in fields(GridSpec)) == (
        "height",
        "width",
        "history_steps",
        "future_steps",
        "resolution_m",
        "n_history_channels",
        "n_state_channels",
        "n_trajectory_channels",
    )
    assert (grid.height, grid.width) == (160, 160)
    assert grid.history_steps == 8
    assert grid.future_steps == 32
    assert contracts.SCHEMA_VERSION == "4.0.0"
    assert POSE_TIME_LAYOUT_VERSION == "future_endpoints_dt_to_horizon_v1"


def test_current_long40_document_freezes_the_same_layout() -> None:
    contents = (ROOT / "docs/long40_system_contract.md").read_text(encoding="utf-8")

    assert "history8_current7_future32_v1" in contents
    assert "future_steps                      = 32" in contents
    assert "sample_count                      = 40" in contents


def test_current_dataclass_surface_includes_active_risk_sample() -> None:
    assert set(contracts._CLASS_REGISTRY) == {
        "BaseState",
        "OracleContext",
        "OracleWorld",
        "LocalTrajectory",
        "RiskSample",
    }
    assert tuple(field.name for field in fields(BaseState)) == (
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


def test_base_state_has_no_oracle_fields() -> None:
    assert_no_oracle_leakage(BaseState)
    bad = make_dataclass("BadInput", [("oracle_future", np.ndarray)])
    with pytest.raises(ContractError):
        assert_no_oracle_leakage(bad)


def test_dynamic_object_specs_accept_current_circle_and_rectangle_shapes() -> None:
    validate_dynamic_object_spec(_object_spec())
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
            "footprint": {"kind": "rectangle", "length_m": 0.8, "width_m": 0.2},
        },
        {
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": np.nan},
        },
        {
            "object_type": "human",
            "footprint": {"kind": "circle", "radius_m": 0.30},
            "extra": True,
        },
    ],
)
def test_dynamic_object_specs_reject_unknown_or_nonphysical_values(
    spec: dict[str, object],
) -> None:
    with pytest.raises(ContractError):
        validate_dynamic_object_spec(spec)


def test_current_base_and_oracle_contracts_validate() -> None:
    grid = build_grid_spec(load_config())

    validate_base_state(_base_state(grid), grid)
    validate_oracle_context(_oracle_context(grid), grid)
    validate_oracle_world(_oracle_world(grid), grid)


def test_current_contracts_reject_shape_dtype_and_key_drift() -> None:
    grid = build_grid_spec(load_config())
    base = _base_state(grid)
    context = _oracle_context(grid)
    world = _oracle_world(grid)

    with pytest.raises(ContractError):
        validate_base_state(
            replace(
                base,
                robot_history=np.zeros(
                    (grid.history_steps, 3), dtype=np.float64
                ),
            ),
            grid,
        )
    with pytest.raises(ContractError):
        validate_oracle_context(replace(context, dynamic_object_future={}), grid)
    with pytest.raises(ContractError):
        validate_oracle_world(
            replace(
                world,
                static_occupancy=np.full(
                    (grid.height, grid.width), np.nan, dtype=ARRAY_DTYPE
                ),
            ),
            grid,
        )


def _assert_roundtrip(original: object, restored: object) -> None:
    assert type(restored) is type(original)
    for field in original.__dataclass_fields__:
        left = getattr(original, field)
        right = getattr(restored, field)
        if isinstance(left, np.ndarray):
            assert left.dtype == right.dtype
            assert left.shape == right.shape
            assert np.array_equal(left, right)
        elif (
            isinstance(left, dict)
            and left
            and all(isinstance(value, np.ndarray) for value in left.values())
        ):
            assert left.keys() == right.keys()
            for key in left:
                assert np.array_equal(left[key], right[key])
                assert left[key].dtype == right[key].dtype
        else:
            assert left == right


@pytest.mark.parametrize(
    "builder",
    [_base_state, _oracle_context, _oracle_world, _local_trajectory],
)
def test_current_dataclasses_roundtrip_without_pickle(
    tmp_path: Path,
    builder,
) -> None:
    grid = build_grid_spec(load_config())
    original = builder(grid)
    path = save_dataclass(original, tmp_path / "object.npz")
    restored = load_dataclass(path)

    _assert_roundtrip(original, restored)
    with np.load(path, allow_pickle=False) as payload:
        assert "meta_json" in payload.files


def test_load_dataclass_rejects_incompatible_schema(tmp_path: Path) -> None:
    grid = build_grid_spec(load_config())
    path = save_dataclass(_base_state(grid), tmp_path / "base.npz")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    metadata = json.loads(str(arrays["meta_json"]))
    metadata["schema_version"] = "unsupported"
    arrays["meta_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    with path.open("wb") as handle:
        np.savez(handle, **arrays)

    with pytest.raises(ContractError, match="schema_version"):
        load_dataclass(path)


def test_contract_validation_cli_checks_current_artifacts() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/00_validate_contracts.py"),
            "--config",
            str(ROOT / "configs/base.yaml"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[ok] SOP01-06 contract artifacts round-trip and validate" in result.stdout


def test_seeding_is_deterministic_and_order_sensitive() -> None:
    assert seeding.derive_seed(42, "a", "b") == seeding.derive_seed(42, "a", "b")
    assert seeding.derive_seed(42, "a", "b") != seeding.derive_seed(42, "b", "a")
    first = seeding.make_rng(42, "generation", "train").standard_normal(8)
    second = seeding.make_rng(42, "generation", "train").standard_normal(8)
    other = seeding.make_rng(7, "generation", "train").standard_normal(8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_base_config_contains_only_current_sections() -> None:
    config = load_config(ROOT / "configs/base.yaml")

    assert set(config) == {
        "seed",
        "schema_version",
        "bev",
        "robot",
        "dynamic_objects",
    }
    assert config["schema_version"] == contracts.SCHEMA_VERSION
    assert config["bev"]["future_steps"] == 32
    assert config["bev"]["future_dt_s"] == 0.2


def test_unknown_config_key_and_schema_drift_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        validate_config({"bev": {"unexpected_key": 1}})
    with pytest.raises(ConfigError):
        validate_config({"totally_unknown": True})

    config_path = tmp_path / "unsupported-schema.yaml"
    config_path.write_text('schema_version: "unsupported"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="schema_version"):
        load_config(config_path)


def test_config_digest_is_stable_sensitive_and_finite() -> None:
    config = load_config(ROOT / "configs/base.yaml")
    assert config_digest(config) == config_digest(dict(config))
    mutated = load_config(ROOT / "configs/base.yaml")
    mutated["seed"] = 999
    assert config_digest(config) != config_digest(mutated)

    canonical = json.dumps(
        {"说明": "机器人", "seed": 42},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert config_digest({"说明": "机器人", "seed": 42}) == seeding.stable_digest(
        canonical,
        size=16,
    )
    with pytest.raises(ValueError, match="finite canonical JSON"):
        config_digest({"value": float("nan")})
