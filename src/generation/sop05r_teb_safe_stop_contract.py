"""Versioned M8 safe-stop contract, independent of immutable M6 inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Real
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml


SOP05R_TEB_SAFE_STOP_CONFIG_VERSION = "sop05r_teb_safe_stop_config_v1"
SOP05R_TEB_SAFE_STOP_LABEL_VERSION = "sop05r_teb_safe_stop_v2"


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("M8 safe-stop contract must be canonical JSON") from exc


def _positive_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class Sop05rTebSafeStopConfig:
    """Frozen M8 v2 policy semantics for one independently versioned audit."""

    version: str
    label_definition_version: str
    require_hidden_at_decision: bool
    allow_stop_scan: bool
    braking_margin_s: float
    hold_until_collision: bool
    digest: str

    def __post_init__(self) -> None:
        if self.version != SOP05R_TEB_SAFE_STOP_CONFIG_VERSION:
            raise ValueError("unsupported M8 safe-stop config version")
        if self.label_definition_version != SOP05R_TEB_SAFE_STOP_LABEL_VERSION:
            raise ValueError("unsupported M8 safe-stop label definition")
        for name in (
            "require_hidden_at_decision",
            "allow_stop_scan",
            "hold_until_collision",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"M8 safe-stop {name} must be boolean")
        if self.hold_until_collision is not True:
            raise ValueError("M8 v2 safe-stop must hold until the collision horizon")
        braking_margin_s = _positive_real(
            self.braking_margin_s, name="braking_margin_s"
        )
        object.__setattr__(self, "braking_margin_s", braking_margin_s)
        expected_digest = hashlib.sha256(_canonical_json(self.as_dict())).hexdigest()
        if self.digest != expected_digest:
            raise ValueError("M8 safe-stop config digest mismatch")

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "label_definition_version": self.label_definition_version,
            "require_hidden_at_decision": self.require_hidden_at_decision,
            "allow_stop_scan": self.allow_stop_scan,
            "braking_margin_s": self.braking_margin_s,
            "hold_until_collision": self.hold_until_collision,
        }


def load_sop05r_teb_safe_stop_config(
    path: str | Path,
) -> Sop05rTebSafeStopConfig:
    """Load only the frozen v2 M8 safe-stop policy layout."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid M8 safe-stop config: {config_path}") from exc
    expected_keys = {
        "version",
        "label_definition_version",
        "require_hidden_at_decision",
        "allow_stop_scan",
        "braking_margin_s",
        "hold_until_collision",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise ValueError("M8 safe-stop config keys are invalid")
    if raw["version"] != SOP05R_TEB_SAFE_STOP_CONFIG_VERSION:
        raise ValueError("unsupported M8 safe-stop config version")
    if raw["label_definition_version"] != SOP05R_TEB_SAFE_STOP_LABEL_VERSION:
        raise ValueError("unsupported M8 safe-stop label definition")
    for name in (
        "require_hidden_at_decision",
        "allow_stop_scan",
        "hold_until_collision",
    ):
        if not isinstance(raw[name], bool):
            raise TypeError(f"M8 safe-stop {name} must be boolean")
    if raw["hold_until_collision"] is not True:
        raise ValueError("M8 v2 safe-stop must hold until the collision horizon")
    normalized = {
        "version": raw["version"],
        "label_definition_version": raw["label_definition_version"],
        "require_hidden_at_decision": raw["require_hidden_at_decision"],
        "allow_stop_scan": raw["allow_stop_scan"],
        "braking_margin_s": _positive_real(
            raw["braking_margin_s"], name="braking_margin_s"
        ),
        "hold_until_collision": raw["hold_until_collision"],
    }
    return Sop05rTebSafeStopConfig(
        **normalized,
        digest=hashlib.sha256(_canonical_json(normalized)).hexdigest(),
    )


__all__ = (
    "SOP05R_TEB_SAFE_STOP_CONFIG_VERSION",
    "SOP05R_TEB_SAFE_STOP_LABEL_VERSION",
    "Sop05rTebSafeStopConfig",
    "load_sop05r_teb_safe_stop_config",
)
