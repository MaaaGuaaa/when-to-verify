#!/usr/bin/env python
"""Validate the current Schema 4 SOP-01 through SOP-06 contract layer."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import contracts  # noqa: E402
from src.contracts import (  # noqa: E402
    BaseState,
    LocalTrajectory,
    OracleContext,
    OracleWorld,
    assert_no_oracle_leakage,
    build_grid_spec,
    load_dataclass,
    save_dataclass,
    validate_base_state,
    validate_oracle_context,
    validate_oracle_world,
)
from src.utils.config import config_digest, load_config  # noqa: E402


def _object_spec() -> dict[str, object]:
    return {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.30},
    }


def _base_state(grid: contracts.GridSpec) -> BaseState:
    object_id = "validation::human"
    return BaseState(
        state_id="validation-base",
        split="train",
        recording_id="validation-recording",
        dynamic_object_ids=(object_id,),
        timestamp=1.4,
        robot_history=np.zeros((grid.history_steps, 3), dtype=np.float32),
        robot_state=np.zeros((2,), dtype=np.float32),
        visible_dynamic_object_history={
            object_id: np.zeros((grid.history_steps, 3), dtype=np.float32)
        },
        visible_dynamic_object_specs={object_id: _object_spec()},
        static_map_local=np.zeros((grid.height, grid.width), dtype=np.float32),
    )


def _oracle_context(grid: contracts.GridSpec) -> OracleContext:
    object_id = "validation::human"
    return OracleContext(
        base_state_id="validation-base",
        dynamic_object_history={
            object_id: np.zeros((grid.history_steps, 3), dtype=np.float32)
        },
        dynamic_object_future={
            object_id: np.zeros((grid.future_steps, 3), dtype=np.float32)
        },
        dynamic_object_specs={object_id: _object_spec()},
    )


def _oracle_world(grid: contracts.GridSpec) -> OracleWorld:
    object_id = "validation::human"
    return OracleWorld(
        world_id="validation-world",
        base_state_id="validation-base",
        static_occupancy=np.zeros((grid.height, grid.width), dtype=np.float32),
        dynamic_object_trajectories={
            object_id: np.zeros((grid.future_steps, 3), dtype=np.float32)
        },
        dynamic_object_specs={object_id: _object_spec()},
        occluders=(),
        blind_spot_config={},
        random_seed=42,
    )


def _local_trajectory(grid: contracts.GridSpec) -> LocalTrajectory:
    return LocalTrajectory(
        trajectory_id="validation-trajectory",
        poses=np.zeros((grid.future_steps, 3), dtype=np.float32),
        controls=np.zeros((grid.future_steps, 2), dtype=np.float32),
        swept_mask=np.zeros((grid.height, grid.width), dtype=np.float32),
        tta_map=np.full((grid.height, grid.width), -1.0, dtype=np.float32),
        braking_map=np.zeros((grid.height, grid.width), dtype=np.float32),
        centerline_map=np.zeros((grid.height, grid.width), dtype=np.float32),
        task_cost=0.0,
    )


def _print_schema_summary(cfg: dict) -> None:
    grid = build_grid_spec(cfg)
    print("=== contract summary ===")
    print(f"schema_version : {contracts.SCHEMA_VERSION}")
    print(f"config_digest  : {config_digest(cfg)}")
    print(
        "grid           : "
        f"H={grid.height} W={grid.width} "
        f"K={grid.history_steps} T={grid.future_steps} "
        f"res={grid.resolution_m}"
    )
    print(f"dataclasses    : {', '.join(contracts._CLASS_REGISTRY)}")


def _assert_roundtrip(original: object, restored: object) -> None:
    if type(restored) is not type(original):
        raise SystemExit("[fail] serialized class changed")
    for name in original.__dataclass_fields__:
        left = getattr(original, name)
        right = getattr(restored, name)
        if isinstance(left, np.ndarray):
            if not (
                left.shape == right.shape
                and left.dtype == right.dtype
                and np.array_equal(left, right)
            ):
                raise SystemExit(f"[fail] round trip mismatch on field {name}")


def _check_contracts(grid: contracts.GridSpec) -> None:
    base_state = _base_state(grid)
    oracle_context = _oracle_context(grid)
    oracle_world = _oracle_world(grid)
    assert_no_oracle_leakage(BaseState)
    validate_base_state(base_state, grid)
    validate_oracle_context(oracle_context, grid)
    validate_oracle_world(oracle_world, grid)
    with tempfile.TemporaryDirectory() as tmp:
        for value in (
            base_state,
            oracle_context,
            oracle_world,
            _local_trajectory(grid),
        ):
            restored = load_dataclass(
                save_dataclass(value, Path(tmp) / f"{type(value).__name__}.npz")
            )
            _assert_roundtrip(value, restored)
    print("[ok] BaseState has no oracle fields")
    print("[ok] SOP01-06 contract artifacts round-trip and validate")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate current Long40 contracts.")
    parser.add_argument("--config", type=Path, default=_ROOT / "configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _print_schema_summary(cfg)
    _check_contracts(build_grid_spec(cfg))
    print("=== all contract checks passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
