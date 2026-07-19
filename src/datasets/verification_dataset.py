"""Schema-3 verification samples with an explicit model-input allowlist."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from src.contracts import (
    ARRAY_DTYPE,
    SCHEMA_VERSION,
    GridSpec,
    LocalTrajectory,
    VerificationSample,
    validate_verification_sample,
)
from src.datasets.risk_dataset import build_trajectory_channels
from src.datasets.split_manager import SPLIT_NAMES
from src.generation.verification_gt import (
    VERIFICATION_GT_VERSION,
    VerificationValueResult,
)
from src.planning.verification_actions import (
    CANONICAL_ACTION_IDS,
    VerificationActionLibrary,
)
from src.utils.seeding import stable_digest


VERIFICATION_DATASET_VERSION = "verification_dataset_schema3_v1"
MODEL_INPUT_KEYS = (
    "bev_history",
    "state_channels",
    "trajectory_channels",
    "verification_fov_mask",
    "verification_action_vector",
)
_FORBIDDEN_PROVENANCE_TOKENS = (
    "oracle",
    "future",
    "world",
    "post_observation",
    "visible_occupied",
    "dynamic_object_trajectories",
)


@dataclass(frozen=True)
class VerificationGroupInput:
    """Deployment tensors and label results for one six-action ranking group."""

    split: str
    base_state_id: str
    nominal_trajectory: LocalTrajectory
    bev_history: np.ndarray
    state_channels: np.ndarray
    expected_fov_masks: Mapping[str, np.ndarray]
    value_results: Mapping[str, VerificationValueResult]
    provenance: Mapping[str, object]


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _owned_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    binary: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an np.ndarray")
    if value.dtype != ARRAY_DTYPE:
        raise TypeError(f"{name} dtype must be float32")
    if value.shape != shape:
        raise ValueError(f"{name} shape must be {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    if binary and not np.isin(value, (0.0, 1.0)).all():
        raise ValueError(f"{name} must be binary")
    result = np.array(value, dtype=ARRAY_DTYPE, order="C", copy=True)
    result.setflags(write=False)
    return result


def _canonical_provenance(value: Any, *, path: str = "provenance") -> dict:
    if not isinstance(value, Mapping):
        raise TypeError("provenance must be a mapping")

    def validate(item: object, *, item_path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str) or not key:
                    raise TypeError(f"{item_path} keys must be non-empty strings")
                lowered = key.lower()
                if any(token in lowered for token in _FORBIDDEN_PROVENANCE_TOKENS):
                    raise ValueError(
                        f"{item_path}.{key} contains a forbidden label-side key"
                    )
                validate(child, item_path=f"{item_path}.{key}")
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                validate(child, item_path=f"{item_path}[{index}]")
            return
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float) and np.isfinite(item):
            return
        raise TypeError(f"{item_path} must contain only finite JSON-native values")

    copied = dict(value)
    validate(copied, item_path=path)
    return json.loads(
        json.dumps(copied, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def _validate_result_group(
    source: VerificationGroupInput,
    library: VerificationActionLibrary,
) -> dict[str, VerificationValueResult]:
    if not isinstance(source.value_results, Mapping):
        raise TypeError("value_results must be a mapping")
    results = dict(source.value_results)
    if set(results) != set(CANONICAL_ACTION_IDS):
        raise ValueError("value_results must contain exactly the six canonical actions")
    reference: VerificationValueResult | None = None
    for action in library.actions:
        result = results[action.action_id]
        if not isinstance(result, VerificationValueResult):
            raise TypeError("value_results must contain VerificationValueResult values")
        if result.verification_action_id != action.action_id:
            raise ValueError("verification result action ID mismatch")
        if result.nominal_trajectory_id != source.nominal_trajectory.trajectory_id:
            raise ValueError("verification result nominal trajectory ID mismatch")
        if reference is None:
            reference = result
            continue
        if result.scenario_bank_digest != reference.scenario_bank_digest:
            raise ValueError("six-action group must use one scenario bank")
        if result.bank_size != reference.bank_size:
            raise ValueError("six-action group bank sizes differ")
        if result.posterior_mode != reference.posterior_mode or (
            result.posterior_temperature != reference.posterior_temperature
        ):
            raise ValueError("six-action group posterior configuration differs")
        if not np.isclose(result.br_before, reference.br_before, rtol=0.0, atol=1e-12):
            raise ValueError("six-action group br_before values differ")
    return results


def _validate_masks(
    source: VerificationGroupInput,
    *,
    grid: GridSpec,
) -> dict[str, np.ndarray]:
    if not isinstance(source.expected_fov_masks, Mapping):
        raise TypeError("expected_fov_masks must be a mapping")
    values = dict(source.expected_fov_masks)
    if set(values) != set(CANONICAL_ACTION_IDS):
        raise ValueError(
            "expected_fov_masks must contain exactly the six canonical actions"
        )
    return {
        action_id: _owned_array(
            values[action_id],
            name=f"expected_fov_masks[{action_id!r}]",
            shape=(1, grid.height, grid.width),
            binary=True,
        )
        for action_id in CANONICAL_ACTION_IDS
    }


def build_verification_samples(
    source: VerificationGroupInput,
    *,
    library: VerificationActionLibrary,
    grid: GridSpec,
) -> tuple[VerificationSample, ...]:
    """Build the canonical six samples without copying label data into inputs."""

    if not isinstance(source, VerificationGroupInput):
        raise TypeError("source must be a VerificationGroupInput")
    if not isinstance(library, VerificationActionLibrary):
        raise TypeError("library must be a VerificationActionLibrary")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    action_ids = tuple(action.action_id for action in library.actions)
    if action_ids != CANONICAL_ACTION_IDS:
        raise ValueError("library action order differs from the canonical order")
    split = _nonempty_string(source.split, name="split")
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    base_state_id = _nonempty_string(source.base_state_id, name="base_state_id")
    if not isinstance(source.nominal_trajectory, LocalTrajectory):
        raise TypeError("nominal_trajectory must be a LocalTrajectory")
    nominal_id = _nonempty_string(
        source.nominal_trajectory.trajectory_id, name="nominal_trajectory_id"
    )

    bev = _owned_array(
        source.bev_history,
        name="bev_history",
        shape=(
            grid.history_steps,
            grid.n_history_channels,
            grid.height,
            grid.width,
        ),
    )
    state = _owned_array(
        source.state_channels,
        name="state_channels",
        shape=(grid.n_state_channels, grid.height, grid.width),
    )
    trajectory_channels = build_trajectory_channels(source.nominal_trajectory, grid)
    trajectory_channels.setflags(write=False)
    masks = _validate_masks(source, grid=grid)
    results = _validate_result_group(source, library)
    provenance = _canonical_provenance(source.provenance)
    ranking_group_id = "verification-group-" + stable_digest(
        VERIFICATION_DATASET_VERSION,
        split,
        base_state_id,
        nominal_id,
        size=16,
    )

    samples: list[VerificationSample] = []
    for action_index, action in enumerate(library.actions):
        result = results[action.action_id]
        vector = np.array(action.vector, dtype=ARRAY_DTYPE, order="C", copy=True)
        vector.setflags(write=False)
        sample_id = "verification-" + stable_digest(
            VERIFICATION_DATASET_VERSION,
            split,
            ranking_group_id,
            action.action_id,
            size=16,
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "verification_dataset_version": VERIFICATION_DATASET_VERSION,
            "ranking_group_id": ranking_group_id,
            "action_index": action_index,
            "action_order": list(CANONICAL_ACTION_IDS),
            "provenance": _canonical_provenance(provenance),
            "label_audit": {
                "verification_gt_version": result.version,
                "scenario_bank_digest": result.scenario_bank_digest,
                "posterior_mode": result.posterior_mode,
                "posterior_temperature": result.posterior_temperature,
                "bank_size": result.bank_size,
            },
        }
        sample = VerificationSample(
            sample_id=sample_id,
            split=split,
            base_state_id=base_state_id,
            nominal_trajectory_id=nominal_id,
            verification_action_id=action.action_id,
            bev_history=_owned_array(
                bev,
                name="bev_history",
                shape=bev.shape,
            ),
            state_channels=_owned_array(
                state,
                name="state_channels",
                shape=state.shape,
            ),
            trajectory_channels=_owned_array(
                trajectory_channels,
                name="trajectory_channels",
                shape=trajectory_channels.shape,
            ),
            verification_fov_mask=masks[action.action_id],
            verification_action_vector=vector,
            value_target=float(result.value_target),
            useful_target=int(result.useful_target),
            br_before=float(result.br_before),
            post_risk=float(result.post_risk),
            metadata=metadata,
        )
        validate_verification_sample(sample, grid)
        samples.append(sample)
    return tuple(samples)


def verification_model_inputs(
    sample: VerificationSample,
) -> Mapping[str, np.ndarray]:
    """Project a sample to the only tensors the value model may consume."""

    if not isinstance(sample, VerificationSample):
        raise TypeError("sample must be a VerificationSample")
    return MappingProxyType(
        {name: getattr(sample, name) for name in MODEL_INPUT_KEYS}
    )


__all__ = (
    "MODEL_INPUT_KEYS",
    "VERIFICATION_DATASET_VERSION",
    "VerificationGroupInput",
    "build_verification_samples",
    "verification_model_inputs",
)
