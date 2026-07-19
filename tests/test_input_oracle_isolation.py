"""Structural guardrails keeping observation rendering oracle-free."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.contracts import BaseState, assert_no_oracle_leakage  # noqa: E402
from src.generation.observation_renderer import (  # noqa: E402
    RenderedObservation,
    render_observation,
)
from src.generation.structural_blindspot import StructuralBlindSpot  # noqa: E402


def test_renderer_public_api_and_output_dataclass_exclude_oracle_inputs() -> None:
    forbidden_tokens = ("oracle", "world", "future", "trajectory")
    signature = inspect.signature(render_observation)
    parameters = tuple(signature.parameters.values())

    assert tuple(parameter.name for parameter in parameters) == (
        "base_state",
        "scene_dynamic_history",
        "scene_dynamic_specs",
        "static_occupancy",
        "sensor_config",
        "config",
    )
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters[1:]
    )
    assert all(
        parameter.default is inspect.Parameter.empty for parameter in parameters
    )
    assert not any(
        token in parameter.name.lower()
        for parameter in parameters
        for token in forbidden_tokens
    )
    assert get_type_hints(render_observation) == {
        "base_state": BaseState,
        "scene_dynamic_history": Mapping[str, np.ndarray],
        "scene_dynamic_specs": Mapping[str, dict[str, object]],
        "static_occupancy": np.ndarray,
        "sensor_config": StructuralBlindSpot | None,
        "config": Mapping[str, Any],
        "return": RenderedObservation,
    }
    assert is_dataclass(RenderedObservation)
    assert RenderedObservation.__dataclass_params__.frozen
    assert tuple(field.name for field in fields(RenderedObservation)) == (
        "bev_history",
        "state_channels",
        "metadata",
    )
    assert get_type_hints(RenderedObservation) == {
        "bev_history": np.ndarray,
        "state_channels": np.ndarray,
        "metadata": dict[str, str],
    }
    assert_no_oracle_leakage(RenderedObservation)


def test_renderer_rejects_unexpected_dynamic_object_future_keyword() -> None:
    with pytest.raises(TypeError, match="dynamic_object_future"):
        render_observation(
            None,
            scene_dynamic_history={},
            scene_dynamic_specs={},
            static_occupancy=None,
            sensor_config=None,
            config={},
            dynamic_object_future=np.zeros((1, 3), dtype=np.float32),
        )
