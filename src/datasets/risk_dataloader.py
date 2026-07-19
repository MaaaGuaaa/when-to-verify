"""Strict SOP09 collation and production-boundary validation.

Only schema-valid ``RiskSample`` inputs are converted to tensors.  The current
repository does not publish a dataset-level production v2 manifest, so the
formal production loader intentionally fails closed instead of interpreting
the ambiguous ``risk_shard_npz_jsonl_v1`` layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.contracts import (
    FORBIDDEN_INPUT_TOKENS,
    HISTORY_CHANNELS,
    INPUT_CHANNELS,
    SCHEMA_VERSION,
    STATE_CHANNELS,
    TRAJECTORY_CHANNELS,
    ContractError,
    GridSpec,
    RiskSample,
    validate_risk_sample,
)
from src.datasets.toy_risk_learning import (
    TOY_DATASET_LAYOUT_VERSION,
    TOY_DATASET_MANIFEST_KEYS,
    TOY_DT_S,
    TOY_FUTURE_ENDPOINT_TIMES_S,
    TOY_FUTURE_STEPS,
    TOY_HISTORY_STEPS,
    frozen_channel_spec,
    toy_dataset_manifest_digest,
    toy_grid_manifest,
    toy_label_digest,
    toy_model_input_digest,
    toy_ordered_sample_digest,
    toy_ordered_sample_ids_digest,
    toy_sample_id_sequence_digest,
)

MODEL_INPUT_KEYS: tuple[str, ...] = (
    "bev_history",
    "state_channels",
    "trajectory_channels",
    "robot_state",
)
TARGET_KEYS: tuple[str, ...] = (
    "collision_label",
    "risk_severity",
    "min_clearance",
    "near_miss",
)
PRODUCTION_DATASET_LAYOUT_VERSION = "risk_dataset_v2"
_PRODUCTION_PROVENANCE_KEYS = frozenset(
    {
        "g1_split_manifest_digest",
        "risk_dataset_manifest_digest",
        "dynamic_objects_config_digest",
        "target_type_policy_digest",
    }
)
_TOY_GRID_KEYS = frozenset(
    {
        "height",
        "width",
        "history_steps",
        "future_steps",
        "resolution_m",
        "sample_dt_s",
        "future_time_layout",
    }
)


class RiskDataContractError(ValueError):
    """Raised before training when risk data/provenance is ambiguous or invalid."""


@dataclass(frozen=True)
class RiskBatch:
    """Model-only inputs, supervised targets, identities, and provenance."""

    model_inputs: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    sample_ids: tuple[str, ...]
    split: str
    provenance: dict[str, object]


def _key_is_forbidden(key: object) -> bool:
    lowered = str(key).lower()
    return any(token in lowered for token in FORBIDDEN_INPUT_TOKENS)


def validate_model_input_mapping(mapping: Mapping[str, Any]) -> None:
    """Recursively reject oracle/future/ground-truth keys and bad tensors."""

    if not isinstance(mapping, Mapping):
        raise RiskDataContractError("model inputs must be a mapping")

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                child = f"{path}.{key}" if path else str(key)
                if _key_is_forbidden(key):
                    raise RiskDataContractError(
                        f"forbidden oracle/future model-input key: {child}"
                    )
                _walk(value, child)
            return
        if isinstance(node, torch.Tensor):
            if node.dtype != torch.float32:
                raise RiskDataContractError(
                    f"model input {path} dtype must be float32, got {node.dtype}"
                )
            if not torch.isfinite(node).all().item():
                raise RiskDataContractError(f"model input {path} contains NaN/Inf")
            return
        if isinstance(node, np.ndarray):
            if node.dtype != np.float32:
                raise RiskDataContractError(
                    f"model input {path} dtype must be float32, got {node.dtype}"
                )
            if not np.isfinite(node).all():
                raise RiskDataContractError(f"model input {path} contains NaN/Inf")

    _walk(mapping, "")


def _validate_channel_spec(value: object) -> None:
    expected = frozen_channel_spec()
    if value != expected:
        raise RiskDataContractError(
            "channel_spec does not match frozen ordered history/state/trajectory channels"
        )
    if expected["history"] != list(HISTORY_CHANNELS):  # defensive drift guard
        raise RiskDataContractError("internal frozen history channel_spec drift")
    if expected["state"] != list(STATE_CHANNELS):
        raise RiskDataContractError("internal frozen state channel_spec drift")
    if expected["trajectory"] != list(TRAJECTORY_CHANNELS):
        raise RiskDataContractError("internal frozen trajectory channel_spec drift")
    if expected["flat"] != list(INPUT_CHANNELS):
        raise RiskDataContractError("internal frozen flat channel_spec drift")


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RiskDataContractError(
            f"{name} must be a lowercase SHA-256 hex digest"
        )
    return value


def _production_provenance_fields(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in _PRODUCTION_PROVENANCE_KEYS:
                found.add(key_text)
            found.update(_production_provenance_fields(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_production_provenance_fields(child))
    return found


def validate_toy_dataset_manifest(
    manifest: Mapping[str, object], *, expected_split: str
) -> None:
    """Validate all provenance needed to interpret a toy publication."""

    if not isinstance(manifest, Mapping):
        raise RiskDataContractError("dataset_manifest must be a mapping")
    production_fields = sorted(_production_provenance_fields(manifest))
    if production_fields:
        raise RiskDataContractError(
            "toy manifest must not contain production provenance fields: "
            f"{production_fields}"
        )
    actual_keys = set(manifest)
    if actual_keys != set(TOY_DATASET_MANIFEST_KEYS):
        missing = sorted(set(TOY_DATASET_MANIFEST_KEYS) - actual_keys)
        unexpected = sorted(actual_keys - set(TOY_DATASET_MANIFEST_KEYS))
        raise RiskDataContractError(
            "dataset manifest top-level keys mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if manifest.get("mode") != "toy":
        raise RiskDataContractError("dataset manifest mode must be 'toy'")
    if manifest.get("dataset_layout_version") != TOY_DATASET_LAYOUT_VERSION:
        raise RiskDataContractError(
            f"dataset_layout_version must be {TOY_DATASET_LAYOUT_VERSION}"
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RiskDataContractError(
            f"schema_version must be {SCHEMA_VERSION}, got {manifest.get('schema_version')!r}"
        )
    _validate_channel_spec(manifest.get("channel_spec"))
    if manifest.get("split") != expected_split:
        raise RiskDataContractError(
            f"dataset manifest split must be {expected_split!r}, got {manifest.get('split')!r}"
        )
    seed = manifest.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise RiskDataContractError("dataset manifest seed must be an integer")
    digest = manifest.get("toy_dataset_manifest_digest")
    if (
        not isinstance(digest, str)
        or len(digest) != 32
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RiskDataContractError(
            "toy_dataset_manifest_digest must be a lowercase BLAKE2b-128 hex digest"
        )
    sample_count = manifest.get("sample_count")
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 1
    ):
        raise RiskDataContractError("dataset manifest sample_count must be positive")
    grid = manifest.get("grid")
    if not isinstance(grid, Mapping):
        raise RiskDataContractError("dataset manifest grid must be a mapping")
    if set(grid) != set(_TOY_GRID_KEYS):
        raise RiskDataContractError("dataset manifest grid keys mismatch")
    for field in ("height", "width"):
        value = grid.get(field)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 8
        ):
            raise RiskDataContractError(f"dataset manifest grid.{field} is invalid")
    if grid.get("history_steps") != TOY_HISTORY_STEPS:
        raise RiskDataContractError("dataset manifest grid.history_steps mismatch")
    if grid.get("future_steps") != TOY_FUTURE_STEPS:
        raise RiskDataContractError("dataset manifest grid.future_steps mismatch")
    resolution = grid.get("resolution_m")
    if (
        not isinstance(resolution, (int, float))
        or isinstance(resolution, bool)
        or not math.isfinite(float(resolution))
        or float(resolution) <= 0.0
    ):
        raise RiskDataContractError("dataset manifest grid.resolution_m is invalid")
    if not math.isclose(
        float(grid.get("sample_dt_s", -1.0)), TOY_DT_S, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RiskDataContractError("dataset manifest grid.sample_dt_s mismatch")
    if grid.get("future_time_layout") != "endpoint_dt_to_horizon":
        raise RiskDataContractError("dataset manifest grid.future_time_layout mismatch")
    expected_times = [
        round(float(value), 7) for value in TOY_FUTURE_ENDPOINT_TIMES_S
    ]
    if manifest.get("future_endpoint_times_s") != expected_times:
        raise RiskDataContractError("future_endpoint_times_s mismatch")
    ordered_ids = manifest.get("ordered_sample_ids")
    if (
        not isinstance(ordered_ids, list)
        or len(ordered_ids) != sample_count
        or any(not isinstance(value, str) or not value for value in ordered_ids)
        or len(set(ordered_ids)) != len(ordered_ids)
    ):
        raise RiskDataContractError(
            "dataset manifest ordered_sample_ids must contain sample_count unique IDs"
        )
    declared_ids_digest = _require_sha256(
        manifest.get("ordered_sample_ids_digest_sha256"),
        "ordered_sample_ids_digest_sha256",
    )
    if declared_ids_digest != toy_sample_id_sequence_digest(ordered_ids):
        raise RiskDataContractError("ordered_sample_ids_digest_sha256 mismatch")
    for field in (
        "model_input_digest_sha256",
        "label_digest_sha256",
        "ordered_sample_digest_sha256",
        "manifest_rows_digest_sha256",
        "label_sidecars_digest_sha256",
    ):
        _require_sha256(manifest.get(field), field)
    try:
        recomputed_digest = toy_dataset_manifest_digest(manifest)
    except (TypeError, ValueError) as error:
        raise RiskDataContractError(
            "dataset manifest header must be finite canonical JSON"
        ) from error
    if digest != recomputed_digest:
        raise RiskDataContractError(
            "toy_dataset_manifest_digest mismatch; authenticated header includes "
            "sample_count, grid, ordered IDs, samples, rows, and sidecars"
        )


def collate_risk_samples(
    samples: Sequence[RiskSample],
    *,
    grid: GridSpec,
    dataset_manifest: Mapping[str, object],
    expected_split: str,
) -> RiskBatch:
    """Validate and collate real ``RiskSample`` field names into CPU tensors."""

    if not samples:
        raise RiskDataContractError("risk sample batch must not be empty")
    validate_toy_dataset_manifest(dataset_manifest, expected_split=expected_split)
    if dataset_manifest["sample_count"] != len(samples):
        raise RiskDataContractError(
            "dataset manifest sample_count does not match supplied samples"
        )
    if dataset_manifest["grid"] != toy_grid_manifest(grid):
        raise RiskDataContractError(
            "dataset manifest grid does not match the supplied GridSpec"
        )
    sample_ids: list[str] = []
    for sample in samples:
        if not isinstance(sample, RiskSample):
            raise RiskDataContractError("every item must be a RiskSample")
        if sample.split != expected_split:
            raise RiskDataContractError(
                f"sample split must be {expected_split!r}, got {sample.split!r}"
            )
        try:
            validate_risk_sample(sample, grid)
        except ContractError as error:
            raise RiskDataContractError(str(error)) from error
        if not math.isfinite(float(sample.min_clearance)):
            raise RiskDataContractError("min_clearance must use a finite sentinel")
        if sample.first_collision_time is not None:
            collision_time = float(sample.first_collision_time)
            if not math.isfinite(collision_time) or collision_time <= 0.0:
                raise RiskDataContractError("first_collision_time must be finite and positive")
        sample_ids.append(sample.sample_id)
    if len(set(sample_ids)) != len(sample_ids):
        raise RiskDataContractError("sample_id values must be unique within a batch")
    if sample_ids != dataset_manifest["ordered_sample_ids"]:
        raise RiskDataContractError(
            "dataset manifest ordered_sample_ids do not match supplied sample order"
        )
    if toy_ordered_sample_ids_digest(samples) != dataset_manifest[
        "ordered_sample_ids_digest_sha256"
    ]:
        raise RiskDataContractError("ordered_sample_ids_digest_sha256 mismatch")
    if toy_model_input_digest(samples) != dataset_manifest[
        "model_input_digest_sha256"
    ]:
        raise RiskDataContractError("model_input_digest_sha256 mismatch")
    if toy_label_digest(samples) != dataset_manifest["label_digest_sha256"]:
        raise RiskDataContractError("label_digest_sha256 mismatch")
    if toy_ordered_sample_digest(samples) != dataset_manifest[
        "ordered_sample_digest_sha256"
    ]:
        raise RiskDataContractError("ordered_sample_digest_sha256 mismatch")

    model_inputs = {
        "bev_history": torch.from_numpy(
            np.stack([sample.bev_history for sample in samples])
        ),
        "state_channels": torch.from_numpy(
            np.stack([sample.state_channels for sample in samples])
        ),
        "trajectory_channels": torch.from_numpy(
            np.stack([sample.trajectory_channels for sample in samples])
        ),
        "robot_state": torch.from_numpy(
            np.stack([sample.robot_state for sample in samples])
        ),
    }
    validate_model_input_mapping(model_inputs)
    targets = {
        "collision_label": torch.tensor(
            [sample.collision_label for sample in samples], dtype=torch.float32
        ),
        "risk_severity": torch.tensor(
            [sample.risk_severity for sample in samples], dtype=torch.float32
        ),
        "min_clearance": torch.tensor(
            [sample.min_clearance for sample in samples], dtype=torch.float32
        ),
        "near_miss": torch.tensor(
            [sample.near_miss for sample in samples], dtype=torch.float32
        ),
    }
    if not all(torch.isfinite(value).all().item() for value in targets.values()):
        raise RiskDataContractError("risk targets contain NaN/Inf")
    return RiskBatch(
        model_inputs=model_inputs,
        targets=targets,
        sample_ids=tuple(sample_ids),
        split=expected_split,
        provenance={
            "mode": "toy",
            "schema_version": SCHEMA_VERSION,
            "channel_spec": frozen_channel_spec(),
            "toy_dataset_manifest_digest": dataset_manifest[
                "toy_dataset_manifest_digest"
            ],
            "ordered_sample_ids_digest_sha256": dataset_manifest[
                "ordered_sample_ids_digest_sha256"
            ],
            "model_input_digest_sha256": dataset_manifest[
                "model_input_digest_sha256"
            ],
            "label_digest_sha256": dataset_manifest["label_digest_sha256"],
            "ordered_sample_digest_sha256": dataset_manifest[
                "ordered_sample_digest_sha256"
            ],
            "manifest_rows_digest_sha256": dataset_manifest[
                "manifest_rows_digest_sha256"
            ],
            "label_sidecars_digest_sha256": dataset_manifest[
                "label_sidecars_digest_sha256"
            ],
        },
    )


def load_production_risk_dataset(root: str | Path) -> None:
    """Fail closed until a dataset-level v2 production publication exists."""

    root_path = Path(root)
    manifest_path = root_path / "dataset_manifest.json"
    observed_layout: object = None
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RiskDataContractError(
                "production risk data requires a valid dataset-level v2 manifest"
            ) from error
        if isinstance(payload, Mapping):
            observed_layout = payload.get(
                "dataset_layout_version", payload.get("layout_version")
            )
    raise RiskDataContractError(
        "production risk dataset is unavailable: require a dataset-level v2 "
        f"manifest ({PRODUCTION_DATASET_LAYOUT_VERSION}); observed {observed_layout!r}"
    )


__all__ = [
    "MODEL_INPUT_KEYS",
    "PRODUCTION_DATASET_LAYOUT_VERSION",
    "RiskBatch",
    "RiskDataContractError",
    "collate_risk_samples",
    "load_production_risk_dataset",
    "validate_model_input_mapping",
    "validate_toy_dataset_manifest",
]
