"""Exact observable grouping and train-normalized soft scenario posterior."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.contracts import SCHEMA_VERSION
from src.generation.counterfactual_verify import (
    CounterfactualObservation,
    SignatureNormalizer,
)


_TOP_KEYS = frozenset({"schema_version", "scenario_bank", "posterior", "decision"})
_POSTERIOR_KEYS = frozenset({"default_temperature", "supported_temperatures"})


@dataclass(frozen=True)
class ObservationPosteriorConfig:
    default_temperature: float
    supported_temperatures: tuple[float, ...]


def _positive_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def load_observation_posterior_config(
    path: str | Path,
) -> ObservationPosteriorConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid verification GT config: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != _TOP_KEYS:
        raise ValueError("verification GT config keys are invalid")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"verification GT schema must be {SCHEMA_VERSION}")
    section = raw["posterior"]
    if not isinstance(section, dict) or set(section) != _POSTERIOR_KEYS:
        raise ValueError("posterior config keys are invalid")
    values = section["supported_temperatures"]
    if not isinstance(values, list) or not values:
        raise ValueError("supported_temperatures must be a non-empty list")
    temperatures = tuple(
        _positive_real(value, name=f"supported_temperatures[{index}]")
        for index, value in enumerate(values)
    )
    if temperatures != (0.1, 0.2, 0.5):
        raise ValueError("supported temperatures must be exactly 0.1, 0.2, 0.5")
    default = _positive_real(
        section["default_temperature"], name="default_temperature"
    )
    if default not in temperatures:
        raise ValueError("default temperature must be supported")
    return ObservationPosteriorConfig(
        default_temperature=default,
        supported_temperatures=temperatures,
    )


def observable_observation_digest(
    observation: CounterfactualObservation,
) -> str:
    """Digest only visible geometry and occupancy, never scenario identity."""

    if not isinstance(observation, CounterfactualObservation):
        raise TypeError("observation must be a CounterfactualObservation")
    digest = hashlib.sha256()
    digest.update(b"verification-observable-visible-occupancy-v1\0")
    for name in ("visible_mask", "visible_occupied_mask"):
        value = getattr(observation, name)
        digest.update(name.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(value, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def exact_observation_posterior(
    observation_digests: Sequence[str],
) -> np.ndarray:
    """Uniformly weight worlds that produce the exact same visible observation."""

    if isinstance(observation_digests, (str, bytes)) or not isinstance(
        observation_digests, Sequence
    ):
        raise TypeError("observation_digests must be a sequence")
    digests = tuple(observation_digests)
    if not digests or any(not isinstance(value, str) or not value for value in digests):
        raise ValueError("observation digests must be non-empty strings")
    size = len(digests)
    posterior = np.zeros((size, size), dtype=np.float64)
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(digests):
        groups.setdefault(value, []).append(index)
    for indices in groups.values():
        weight = 1.0 / len(indices)
        posterior[np.ix_(indices, indices)] = weight
    validate_posterior_matrix(posterior, size=size)
    return posterior


def soft_observation_posterior(
    signatures: np.ndarray,
    *,
    normalizer: SignatureNormalizer,
    temperature: float,
) -> np.ndarray:
    """Return row-wise `p(world_j | observation_from_world_m)` weights."""

    if not isinstance(normalizer, SignatureNormalizer):
        raise TypeError("normalizer must be a train-fitted SignatureNormalizer")
    tau = _positive_real(temperature, name="temperature")
    normalized = normalizer.transform(signatures)
    if normalized.ndim != 2 or normalized.shape[0] == 0:
        raise ValueError("signatures must be a non-empty matrix")
    values = normalized.astype(np.float64)
    differences = values[:, None, :] - values[None, :, :]
    squared_distances = np.einsum(
        "ijk,ijk->ij", differences, differences, optimize=True
    )
    logits = -squared_distances / tau
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    normalizing = np.sum(weights, axis=1, keepdims=True)
    if not np.isfinite(weights).all() or np.any(normalizing <= 0.0):
        raise ValueError("soft posterior normalization is non-finite")
    posterior = weights / normalizing
    validate_posterior_matrix(posterior, size=values.shape[0])
    return posterior


def validate_posterior_matrix(posterior: np.ndarray, *, size: int) -> None:
    if isinstance(size, (bool, np.bool_)) or not isinstance(
        size, (Integral, np.integer)
    ):
        raise TypeError("size must be an integer")
    expected = int(size)
    if expected <= 0:
        raise ValueError("size must be positive")
    if not isinstance(posterior, np.ndarray) or posterior.dtype != np.float64:
        raise TypeError("posterior must be a float64 ndarray")
    if posterior.shape != (expected, expected):
        raise ValueError("posterior shape mismatch")
    if not np.isfinite(posterior).all():
        raise ValueError("posterior must be finite")
    if np.any(posterior < 0.0) or np.any(posterior > 1.0):
        raise ValueError("posterior entries must be in [0,1]")
    if not np.allclose(
        np.sum(posterior, axis=1), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("posterior rows must sum to one")


__all__ = (
    "ObservationPosteriorConfig",
    "exact_observation_posterior",
    "load_observation_posterior_config",
    "observable_observation_digest",
    "soft_observation_posterior",
    "validate_posterior_matrix",
)
