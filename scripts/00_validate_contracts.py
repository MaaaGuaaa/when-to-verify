#!/usr/bin/env python
"""Validate the frozen contract layer and print a schema summary (SOP-00).

Exit code 0 means: config loads and validates, the toy world builds
deterministically, model-input classes carry no oracle fields, and a sample
survives a serialization round trip. Any failure exits non-zero.

Usage:
    python scripts/00_validate_contracts.py --config configs/base.yaml
"""

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
    INPUT_CHANNELS,
    MODEL_INPUT_CLASSES,
    assert_no_oracle_leakage,
    build_grid_spec,
    load_dataclass,
    save_dataclass,
    validate_base_state,
    validate_oracle_world,
    validate_risk_sample,
    validate_verification_sample,
)
from src.utils.config import config_digest, load_config  # noqa: E402

_FIX = _ROOT / "tests" / "fixtures"
if str(_FIX) not in sys.path:
    sys.path.insert(0, str(_FIX))

import toy_world  # noqa: E402


def _print_schema_summary(cfg: dict) -> None:
    grid = build_grid_spec(cfg)
    print("=== contract summary ===")
    print(f"schema_version : {contracts.SCHEMA_VERSION}")
    print(f"config_digest  : {config_digest(cfg)}")
    print(f"grid           : H={grid.height} W={grid.width} "
          f"K={grid.history_steps} T={grid.future_steps} res={grid.resolution_m}")
    print(f"channels ({len(INPUT_CHANNELS)}) : {', '.join(INPUT_CHANNELS)}")
    print(f"dataclasses    : {', '.join(contracts._CLASS_REGISTRY)}")


def _check_no_oracle_leakage() -> None:
    for cls in MODEL_INPUT_CLASSES:
        assert_no_oracle_leakage(cls)
    print("[ok] model-input classes carry no oracle fields")


def _check_roundtrip(grid) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        builders = (
            toy_world.make_risk_sample,
            toy_world.make_verification_sample,
            lambda spec: toy_world.make_base_state("toy_bs_0", spec),
            lambda spec: toy_world.make_oracle_world("toy_world_0", "toy_bs_0", 7, spec),
        )
        for builder in builders:
            obj = builder(grid)
            path = save_dataclass(obj, Path(tmp) / "obj.npz")
            restored = load_dataclass(path)
            for f in obj.__dataclass_fields__:
                a, b = getattr(obj, f), getattr(restored, f)
                if isinstance(a, np.ndarray):
                    if not (a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)):
                        raise SystemExit(f"[fail] round trip mismatch on field {f}")
        validate_risk_sample(toy_world.make_risk_sample(grid), grid)
        validate_verification_sample(toy_world.make_verification_sample(grid), grid)
        validate_base_state(toy_world.make_base_state("toy_bs_0", grid), grid)
        validate_oracle_world(
            toy_world.make_oracle_world("toy_world_0", "toy_bs_0", 7, grid),
            grid,
        )
    print("[ok] risk/verification samples round-trip and validate")
    print("[ok] dynamic-object base/oracle artifacts round-trip and validate")


def _check_determinism() -> None:
    a = toy_world.build_toy_world(42)["seed_probe"]
    b = toy_world.build_toy_world(42)["seed_probe"]
    c = toy_world.build_toy_world(7)["seed_probe"]
    if not np.array_equal(a, b):
        raise SystemExit("[fail] same seed produced different toy world")
    if np.array_equal(a, c):
        raise SystemExit("[fail] different seed produced identical toy world")
    print("[ok] toy world deterministic per seed, varies across seeds")


def _check_toy_answers() -> None:
    world = toy_world.build_toy_world()
    hand = toy_world.toy_hand_answers()
    coll = world["risk_cases"]["collision"]
    if coll["collision"] != 1 or coll["risk_severity"] != 1.0:
        raise SystemExit("[fail] toy collision answer mismatch")
    peek = world["verification_example"]["actions"]["forward_peek"]
    if abs(peek["value"] - hand["verification"]["forward_peek"]["value"]) > 1e-6:
        raise SystemExit("[fail] toy G* answer mismatch")
    print("[ok] toy risk and G* match hand-derived answers")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate frozen contracts (SOP-00).")
    parser.add_argument("--config", type=Path, default=_ROOT / "configs" / "base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _print_schema_summary(cfg)
    _check_no_oracle_leakage()
    _check_determinism()
    _check_toy_answers()
    _check_roundtrip(build_grid_spec(cfg))
    print("=== all contract checks passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
