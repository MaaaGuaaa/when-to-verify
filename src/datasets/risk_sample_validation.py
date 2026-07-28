"""Publication validation for current-schema ``RiskSample`` payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np

from src.contracts import (
    DYNAMIC_OBJECT_TYPES,
    POSE_TIME_LAYOUT_VERSION,
    SCHEMA_VERSION,
    GridSpec,
    RiskSample,
    assert_no_oracle_leakage,
    validate_risk_sample,
)
from src.generation.risk_gt import (
    RISK_GT_VERSION,
    resolve_no_object_clearance_sentinel,
)


RISK_SAMPLE_RENDERER_LAYOUT_VERSION = "bev_history2_state9_v1"

_METADATA_KEYS = frozenset(
    {"schema_version", "renderer", "trajectory_id", "provenance", "label_audit"}
)
_RENDERER_METADATA_KEYS = frozenset(
    {
        "renderer_layout_version",
        "base_state_id",
        "sensor_config_digest",
        "static_occupancy_digest",
    }
)
_LABEL_AUDIT_KEYS = frozenset(
    {
        "risk_gt_version",
        "pose_time_layout_version",
        "critical_object_id",
        "critical_object_type",
        "time_to_min_clearance_s",
        "has_hidden_target",
    }
)
_FORBIDDEN_METADATA_KEY_TOKENS = (
    "future",
    "oracle",
    "clearance_sequence",
    "dynamic_object_trajectories",
    "hidden_object_ids",
)


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_metadata_value(value: object, *, path: str) -> None:
    if isinstance(value, np.ndarray):
        raise TypeError(f"metadata {path} must not contain ndarray payloads")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"metadata {path} keys must be non-empty strings")
            lowered = key.lower()
            if any(token in lowered for token in _FORBIDDEN_METADATA_KEY_TOKENS):
                raise ValueError(
                    f"metadata {path}.{key} contains a forbidden payload key"
                )
            _validate_metadata_value(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_metadata_value(child, path=f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"metadata {path} must contain only finite values")
        return
    if isinstance(value, (np.generic, Real)):
        raise TypeError(f"metadata {path} must contain JSON-native scalar values")
    raise TypeError(f"metadata {path} contains a non-JSON value")


def validate_risk_sample_for_publication(
    sample: RiskSample,
    grid: GridSpec,
) -> None:
    """Validate model arrays, finite labels, and recursive metadata isolation."""

    if not isinstance(sample, RiskSample):
        raise TypeError("sample must be a RiskSample")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    validate_risk_sample(sample, grid)
    assert_no_oracle_leakage(RiskSample)
    for name in ("sample_id", "split", "base_state_id", "pair_group_id", "event_type"):
        _require_nonempty_string(getattr(sample, name), name=name)
    if isinstance(sample.collision_label, (bool, np.bool_)) or not isinstance(
        sample.collision_label, (int, np.integer)
    ):
        raise TypeError("collision_label must be an integer")
    if isinstance(sample.near_miss, (bool, np.bool_)) or not isinstance(
        sample.near_miss, (int, np.integer)
    ):
        raise TypeError("near_miss must be an integer")
    severity = _finite_float(sample.risk_severity, name="risk_severity")
    minimum = _finite_float(sample.min_clearance, name="min_clearance")
    first_collision = sample.first_collision_time
    if first_collision is not None:
        first_collision = _finite_float(
            first_collision, name="first_collision_time"
        )
        if first_collision <= 0.0:
            raise ValueError("first_collision_time must be positive")
    if sample.collision_label == 1:
        if first_collision is None:
            raise ValueError("collision requires first_collision_time")
        if severity != 1.0:
            raise ValueError("collision requires risk_severity == 1")
        if minimum > 0.0:
            raise ValueError("collision requires min_clearance <= 0")
    elif first_collision is not None:
        raise ValueError("noncollision requires first_collision_time=None")
    elif minimum <= 0.0:
        raise ValueError("noncollision requires positive min_clearance")

    if not isinstance(sample.metadata, dict):
        raise TypeError("metadata must be a dict")
    if set(sample.metadata) != _METADATA_KEYS:
        raise ValueError(f"metadata keys must be exactly {sorted(_METADATA_KEYS)}")
    _validate_metadata_value(sample.metadata, path="metadata")
    if sample.metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"metadata schema_version must be {SCHEMA_VERSION}")
    if sample.metadata["trajectory_id"] == "" or not isinstance(
        sample.metadata["trajectory_id"], str
    ):
        raise ValueError("metadata trajectory_id must be a non-empty string")

    renderer = sample.metadata["renderer"]
    if not isinstance(renderer, dict) or set(renderer) != _RENDERER_METADATA_KEYS:
        raise ValueError("renderer metadata keys violate the frozen contract")
    if (
        renderer["renderer_layout_version"]
        != RISK_SAMPLE_RENDERER_LAYOUT_VERSION
    ):
        raise ValueError("renderer layout version mismatch")
    if renderer["base_state_id"] != sample.base_state_id:
        raise ValueError("renderer base_state_id mismatch")
    if not isinstance(sample.metadata["provenance"], dict):
        raise TypeError("provenance metadata must be a dict")

    audit = sample.metadata["label_audit"]
    if not isinstance(audit, dict) or set(audit) != _LABEL_AUDIT_KEYS:
        raise ValueError("label_audit keys violate the frozen contract")
    if audit["risk_gt_version"] != RISK_GT_VERSION:
        raise ValueError("risk_gt_version mismatch")
    if audit["pose_time_layout_version"] != POSE_TIME_LAYOUT_VERSION:
        raise ValueError("pose_time_layout_version mismatch")
    has_hidden = audit["has_hidden_target"]
    if not isinstance(has_hidden, bool):
        raise TypeError("has_hidden_target must be bool")
    critical_id = audit["critical_object_id"]
    critical_type = audit["critical_object_type"]
    time_to_minimum = audit["time_to_min_clearance_s"]
    if has_hidden:
        _require_nonempty_string(critical_id, name="critical_object_id")
        if critical_type not in DYNAMIC_OBJECT_TYPES:
            raise ValueError("critical_object_type is invalid")
        time_to_minimum = _finite_float(
            time_to_minimum, name="time_to_min_clearance_s"
        )
        if time_to_minimum <= 0.0:
            raise ValueError("time_to_min_clearance_s must be positive")
    else:
        if any(
            value is not None
            for value in (critical_id, critical_type, time_to_minimum)
        ):
            raise ValueError("empty hidden set requires empty label_audit identity")
        if sample.collision_label != 0 or sample.near_miss != 0 or severity != 0.0:
            raise ValueError("empty hidden set requires zero risk labels")
        sentinel = resolve_no_object_clearance_sentinel(grid)
        if minimum != sentinel:
            raise ValueError("empty hidden set requires the grid-diagonal sentinel")


__all__ = [
    "RISK_SAMPLE_RENDERER_LAYOUT_VERSION",
    "validate_risk_sample_for_publication",
]
