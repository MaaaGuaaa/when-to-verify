"""Config loading with default merge, unknown-key rejection, and stable digest.

The schema below is the single source of truth for allowed configuration keys.
Loading rejects any unknown key (typo protection) and then fills defaults, so a
partial config file remains valid while silent drift is impossible.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ..contracts import SCHEMA_VERSION


class ConfigError(ValueError):
    """Raised when a config file contains unknown keys or an invalid shape."""


# Frozen default configuration (spec §29 plus SOP-00 frozen contracts).
DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "schema_version": SCHEMA_VERSION,
    "bev": {
        "range_m": 16.0,
        "resolution_m": 0.1,
        "size": 160,
        "history_steps": 8,
        "history_dt_s": 0.2,
        "future_steps": 15,
        "future_dt_s": 0.2,
    },
    "robot": {
        "model": "differential_drive",
        "length_m": 0.70,
        "width_m": 0.55,
        "inflation_m": 0.15,
        "max_linear_speed_mps": 0.9,
        "max_angular_speed_radps": 0.8,
    },
    "dynamic_objects": {
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
    },
    "age_map": {
        "a_max_s": 5.0,
        "never_seen_value": 1.0,
        "visible_value": 0.0,
    },
    "trajectories": {
        "linear_velocities": [0.2, 0.4, 0.6, 0.8],
        "angular_velocities": [-0.8, -0.4, 0.0, 0.4, 0.8],
        "reverse_velocities": [-0.2, -0.4],
        "reverse_probability": 0.2,
        "horizon_s": 3.0,
        "dt_s": 0.2,
    },
    "risk_gt": {
        "sigma_distance_m": 0.5,
        "sigma_time_s": 2.0,
        "near_miss_distance_m": 0.35,
    },
    "scenario_bank": {
        "size": 16,
        "posterior_temperature": 0.2,
        "reject_cost": 0.20,
    },
    "verification": {
        "useful_margin": 0.0,
        "decision_margin": 0.01,
    },
    "verification_cost": {
        "lambda_time": 0.04,
        "lambda_distance": 0.05,
        "lambda_yaw_per_deg": 0.0015,
    },
}


def _schema_from_defaults(node: Any) -> Any:
    """Derive the allowed-key schema tree from the default config structure."""
    if isinstance(node, dict):
        return {key: _schema_from_defaults(value) for key, value in node.items()}
    return None  # leaf marker


CONFIG_SCHEMA = _schema_from_defaults(DEFAULT_CONFIG)


def validate_config(cfg: dict, schema: Any = CONFIG_SCHEMA, path: str = "") -> None:
    """Reject any key not present in the schema (recursively)."""
    if not isinstance(schema, dict):
        return
    if not isinstance(cfg, dict):
        raise ConfigError(f"config node at '{path or '/'}' must be a mapping")
    for key, value in cfg.items():
        child_path = f"{path}/{key}" if path else key
        if key not in schema:
            raise ConfigError(f"unknown config key: '{child_path}'")
        validate_config(value, schema[key], child_path)


def apply_defaults(cfg: dict) -> dict:
    """Deep-merge ``cfg`` onto a copy of :data:`DEFAULT_CONFIG`."""
    merged = copy.deepcopy(DEFAULT_CONFIG)

    def _merge(dst: dict, src: dict) -> None:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                _merge(dst[key], value)
            else:
                dst[key] = value

    _merge(merged, cfg)
    return merged


def load_config(path: str | Path | None = None) -> dict:
    """Load, validate against the schema, and default-fill a config file."""
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError("top-level config must be a mapping")
    validate_config(raw)
    merged = apply_defaults(raw)
    if merged["schema_version"] != SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version must be {SCHEMA_VERSION}, "
            f"got {merged['schema_version']!r}"
        )
    return merged


def config_digest(cfg: dict, size: int = 16) -> str:
    """Return a stable digest of a config dict (order-independent)."""
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=size).hexdigest()
