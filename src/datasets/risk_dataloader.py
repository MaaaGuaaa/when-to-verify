"""Strict SOP09 collation and production-boundary validation.

Only schema-valid ``RiskSample`` inputs are converted to tensors.  Production
data is admitted exclusively through the formally authenticated
``risk_dataset_v2`` seal loader; shard roots and legacy v1 layouts never fall
through to an alternate parser.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

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
from src.datasets.shard_writer import (
    RISK_SHARD_LAYOUT_VERSION,
    LoadedRiskShard,
    load_risk_shard,
)

if TYPE_CHECKING:
    from src.datasets.risk_dataset_seal import LoadedRiskDataset

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
_SUBSET_MEMBERSHIP_DOMAIN = "risk-production-subset-membership-v1"
_SUBSET_DIGEST_DOMAIN = "risk-production-subset-v1"
_SHARD_ORDER_DOMAIN = "risk-production-shard-order-v1"
_ROW_ORDER_DOMAIN = "risk-production-row-order-v1"
_PRODUCTION_TARGET_CHANNELS = (*TARGET_KEYS, "first_collision_time")


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


@dataclass(frozen=True)
class ProductionRiskSubset:
    """Authenticated deterministic sample membership for production training."""

    sample_ids: tuple[str, ...]
    sample_ids_digest_sha256: str
    dataset_manifest_digest: str
    seed: int
    max_samples: int


@dataclass(frozen=True)
class RiskStreamCursor:
    """Authenticated position of the next selected row in one epoch."""

    epoch: int
    shard_order_position: int
    shard_index: int
    row_order_position: int
    samples_yielded: int
    dataset_manifest_digest: str
    subset_digest_sha256: str


@dataclass(frozen=True)
class _VerifiedProductionSubsetPlan:
    """Immutable sample-ID-only result of formal membership verification."""

    dataset_manifest_digest: str
    subset_digest_sha256: str
    sample_ids: tuple[str, ...]
    seed: int
    max_samples: int
    selected_by_shard: tuple[tuple[int, frozenset[str]], ...]


_PRODUCTION_SUBSET_PLAN_CACHE_MAX_ENTRIES = 4
_PRODUCTION_SUBSET_PLAN_CACHE: OrderedDict[
    tuple[str, str], _VerifiedProductionSubsetPlan
] = OrderedDict()


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


def _require_blake2b128(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RiskDataContractError(
            f"{name} must be a lowercase BLAKE2b-128 hex digest"
        )
    return value


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RiskDataContractError(f"{name} must be a non-empty string")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RiskDataContractError(f"{name} must be a nonnegative integer")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RiskDataContractError(f"{name} must be a positive integer")
    return value


def _canonical_json_bytes(value: object, *, name: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RiskDataContractError(f"{name} must be finite canonical JSON") from error


def _stable_score(domain: str, **components: object) -> bytes:
    """Hash canonical JSON containing the exact caller-supplied domain string."""

    return hashlib.sha256(
        _canonical_json_bytes(
            {"domain": domain, **components},
            name=f"{domain} score components",
        )
    ).digest()


def _subset_digest(
    sample_ids: tuple[str, ...],
    *,
    dataset_manifest_digest: str,
    seed: int,
    max_samples: int,
) -> str:
    """Bind ordered membership using domain ``risk-production-subset-v1``."""

    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "dataset_manifest_digest": dataset_manifest_digest,
                "domain": _SUBSET_DIGEST_DOMAIN,
                "max_samples": max_samples,
                "sample_ids": list(sample_ids),
                "seed": seed,
            },
            name="production subset",
        )
    ).hexdigest()


def _validated_production_provenance(
    dataset_provenance: object,
) -> dict[str, str]:
    if not isinstance(dataset_provenance, Mapping):
        raise RiskDataContractError("dataset_provenance must be a mapping")
    if set(dataset_provenance) != _PRODUCTION_PROVENANCE_KEYS:
        raise RiskDataContractError(
            "dataset_provenance keys must exactly match authenticated production fields"
        )
    values = {
        "g1_split_manifest_digest": _require_blake2b128(
            dataset_provenance.get("g1_split_manifest_digest"),
            "g1_split_manifest_digest",
        ),
        "risk_dataset_manifest_digest": _require_sha256(
            dataset_provenance.get("risk_dataset_manifest_digest"),
            "risk_dataset_manifest_digest",
        ),
        "dynamic_objects_config_digest": _require_sha256(
            dataset_provenance.get("dynamic_objects_config_digest"),
            "dynamic_objects_config_digest",
        ),
        "target_type_policy_digest": _require_blake2b128(
            dataset_provenance.get("target_type_policy_digest"),
            "target_type_policy_digest",
        ),
    }
    return values


def _validate_production_channel_spec(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "history",
        "state",
        "trajectory",
        "flat",
        "targets",
    }:
        raise RiskDataContractError("production channel_spec keys mismatch")
    expected = {
        "history": HISTORY_CHANNELS,
        "state": STATE_CHANNELS,
        "trajectory": TRAJECTORY_CHANNELS,
        "flat": INPUT_CHANNELS,
        "targets": _PRODUCTION_TARGET_CHANNELS,
    }
    for field, channels in expected.items():
        supplied = value.get(field)
        if not isinstance(supplied, (list, tuple)) or tuple(supplied) != channels:
            raise RiskDataContractError(
                f"production channel_spec.{field} ordering mismatch"
            )


def _validate_loaded_risk_dataset(dataset: object) -> "LoadedRiskDataset":
    from src.datasets.risk_dataset_seal import (
        LoadedRiskDataset,
        RiskShardDescriptor,
        validate_risk_dataset_manifest,
    )

    if not isinstance(dataset, LoadedRiskDataset):
        raise RiskDataContractError(
            "production streaming requires a LoadedRiskDataset from risk_dataset_v2"
        )
    digest = _require_sha256(
        dataset.risk_dataset_manifest_digest,
        "dataset risk_dataset_manifest_digest",
    )
    if not isinstance(dataset.manifest, Mapping):
        raise RiskDataContractError("loaded dataset manifest must be a mapping")
    validated_manifest_digest = validate_risk_dataset_manifest(dataset.manifest)
    if validated_manifest_digest != digest:
        raise RiskDataContractError(
            "validated manifest digest does not match loaded dataset digest"
        )
    if dataset.manifest.get("dataset_layout_version") != PRODUCTION_DATASET_LAYOUT_VERSION:
        raise RiskDataContractError(
            f"loaded dataset must use {PRODUCTION_DATASET_LAYOUT_VERSION}"
        )
    if dataset.manifest.get("schema_version") != SCHEMA_VERSION:
        raise RiskDataContractError("loaded dataset schema_version mismatch")
    split = _require_nonempty_string(dataset.split, "dataset split")
    if dataset.manifest.get("split") != split:
        raise RiskDataContractError("loaded dataset split differs from manifest")
    if dataset.manifest.get("risk_dataset_manifest_digest") != digest:
        raise RiskDataContractError("dataset manifest digest does not match loaded dataset")
    sample_count = _require_positive_int(dataset.sample_count, "dataset sample_count")
    if dataset.manifest.get("sample_count") != sample_count:
        raise RiskDataContractError("dataset sample_count differs from manifest")
    if not isinstance(dataset.grid, GridSpec):
        raise RiskDataContractError("loaded dataset grid must be a GridSpec")
    manifest_grid = dataset.manifest.get("grid")
    if not isinstance(manifest_grid, Mapping):
        raise RiskDataContractError("dataset manifest grid must be a mapping")
    runtime_grid = {
        "height": dataset.grid.height,
        "width": dataset.grid.width,
        "history_steps": dataset.grid.history_steps,
        "future_steps": dataset.grid.future_steps,
        "resolution_m": dataset.grid.resolution_m,
    }
    if any(manifest_grid.get(field) != value for field, value in runtime_grid.items()):
        raise RiskDataContractError("loaded dataset grid differs from manifest grid")
    if (
        dataset.grid.n_history_channels != len(HISTORY_CHANNELS)
        or dataset.grid.n_state_channels != len(STATE_CHANNELS)
        or dataset.grid.n_trajectory_channels != len(TRAJECTORY_CHANNELS)
    ):
        raise RiskDataContractError("loaded dataset grid channel counts are invalid")
    _validate_production_channel_spec(dataset.manifest.get("channel_spec"))
    if not isinstance(dataset.shards, tuple) or not dataset.shards:
        raise RiskDataContractError("loaded dataset shards must be a non-empty tuple")
    if dataset.manifest.get("shard_count") != len(dataset.shards):
        raise RiskDataContractError("dataset shard count differs from manifest")
    manifest_shards = dataset.manifest.get("shards")
    if not isinstance(manifest_shards, (list, tuple)) or len(
        manifest_shards
    ) != len(dataset.shards):
        raise RiskDataContractError("dataset manifest shards differ from loaded shards")

    descriptor_total = 0
    for position, descriptor in enumerate(dataset.shards):
        if not isinstance(descriptor, RiskShardDescriptor):
            raise RiskDataContractError("dataset shards must be RiskShardDescriptor values")
        if descriptor.shard_index != position:
            raise RiskDataContractError(
                "dataset shard indices must be unique and contiguous from zero"
            )
        if descriptor.relative_root != f"shard-{position:05d}":
            raise RiskDataContractError("dataset shard relative_root is invalid")
        descriptor_total += _require_positive_int(
            descriptor.sample_count, f"shards[{position}].sample_count"
        )
        for field in (
            "manifest_digest",
            "semantic_digest",
            "payload_sha256",
            "metadata_sha256",
            "summary_sha256",
        ):
            _require_sha256(
                getattr(descriptor, field), f"shards[{position}].{field}"
            )
        manifest_descriptor = manifest_shards[position]
        expected_descriptor = {
            "shard_index": descriptor.shard_index,
            "relative_root": descriptor.relative_root,
            "sample_count": descriptor.sample_count,
            "manifest_digest": descriptor.manifest_digest,
            "semantic_digest": descriptor.semantic_digest,
            "payload_sha256": descriptor.payload_sha256,
            "metadata_sha256": descriptor.metadata_sha256,
            "summary_sha256": descriptor.summary_sha256,
        }
        if (
            not isinstance(manifest_descriptor, Mapping)
            or dict(manifest_descriptor) != expected_descriptor
        ):
            raise RiskDataContractError(
                f"dataset manifest shard {position} differs from loaded descriptor"
            )
    if descriptor_total != sample_count:
        raise RiskDataContractError("dataset sample_count differs from shard totals")

    provenance = _validated_production_provenance(dataset.provenance)
    if provenance["risk_dataset_manifest_digest"] != digest:
        raise RiskDataContractError(
            "dataset provenance digest does not match loaded dataset digest"
        )
    if any(dataset.manifest.get(field) != value for field, value in provenance.items()):
        raise RiskDataContractError(
            "loaded dataset provenance differs from authenticated manifest"
        )
    return dataset


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise RiskDataContractError(f"failed to hash production shard file: {path}") from error
    return digest.hexdigest()


def _validate_production_sample(
    sample: object,
    *,
    grid: GridSpec,
    expected_split: str,
) -> RiskSample:
    if not isinstance(sample, RiskSample):
        raise RiskDataContractError("every production item must be a RiskSample")
    _require_nonempty_string(sample.sample_id, "sample_id")
    if sample.split != expected_split:
        raise RiskDataContractError(
            f"sample split must be {expected_split!r}, got {sample.split!r}"
        )
    if not isinstance(sample.metadata, Mapping) or sample.metadata.get(
        "schema_version"
    ) != SCHEMA_VERSION:
        raise RiskDataContractError("sample schema_version mismatch")
    try:
        validate_risk_sample(sample, grid)
    except ContractError as error:
        raise RiskDataContractError(str(error)) from error
    if not math.isfinite(float(sample.min_clearance)):
        raise RiskDataContractError("min_clearance must use a finite sentinel")
    if sample.first_collision_time is not None:
        collision_time = float(sample.first_collision_time)
        if not math.isfinite(collision_time) or collision_time <= 0.0:
            raise RiskDataContractError(
                "first_collision_time must be finite and positive"
            )
    return sample


def _load_validated_production_shard(
    dataset: "LoadedRiskDataset", descriptor: object
) -> LoadedRiskShard:
    from src.datasets.risk_dataset_seal import RiskShardDescriptor

    if not isinstance(descriptor, RiskShardDescriptor):
        raise RiskDataContractError("dataset shard descriptor type mismatch")
    shard_root = dataset.collection_root / descriptor.relative_root
    try:
        loaded = load_risk_shard(
            shard_root,
            grid=dataset.grid,
            split_audit_records=(),
        )
    except (OSError, TypeError, ValueError) as error:
        raise RiskDataContractError(
            f"formal load_risk_shard failed for {descriptor.relative_root}: {error}"
        ) from error
    if not isinstance(loaded, LoadedRiskShard):
        raise RiskDataContractError("formal shard loader returned an invalid type")
    summary = loaded.summary
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("layout_version") != RISK_SHARD_LAYOUT_VERSION
        or summary.get("shard_index") != descriptor.shard_index
        or summary.get("split") != dataset.split
        or summary.get("expected_sample_count") != descriptor.sample_count
        or len(loaded.samples) != descriptor.sample_count
        or len(loaded.manifest) != descriptor.sample_count
    ):
        raise RiskDataContractError(
            f"formal shard summary mismatch for {descriptor.relative_root}"
        )
    if loaded.manifest_digest != descriptor.manifest_digest:
        raise RiskDataContractError(
            f"shard manifest digest mismatch for {descriptor.relative_root}"
        )
    if loaded.semantic_digest != descriptor.semantic_digest:
        raise RiskDataContractError(
            f"shard semantic digest mismatch for {descriptor.relative_root}"
        )
    observed_files = {
        "payload_sha256": _sha256_file(shard_root / "samples.npz"),
        "metadata_sha256": _sha256_file(shard_root / "metadata.jsonl"),
        "summary_sha256": _sha256_file(shard_root / "summary.json"),
    }
    for field, observed in observed_files.items():
        if getattr(descriptor, field) != observed:
            raise RiskDataContractError(
                f"shard file checksum mismatch for {descriptor.relative_root}: {field}"
            )

    sample_ids: set[str] = set()
    for sample in loaded.samples:
        _validate_production_sample(
            sample,
            grid=dataset.grid,
            expected_split=dataset.split,
        )
        if sample.sample_id in sample_ids:
            raise RiskDataContractError(
                f"duplicate sample_id within production shard: {sample.sample_id}"
            )
        sample_ids.add(sample.sample_id)
    return loaded


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


def _rank_production_sample_ids(
    dataset: "LoadedRiskDataset",
    *,
    max_samples: int,
    seed: int,
) -> tuple[tuple[str, ...], dict[int, frozenset[str]], frozenset[str]]:
    ranked: list[tuple[bytes, str, int]] = []
    all_sample_ids: set[str] = set()
    for descriptor in dataset.shards:
        loaded = _load_validated_production_shard(dataset, descriptor)
        for sample in loaded.samples:
            sample_id = sample.sample_id
            if sample_id in all_sample_ids:
                raise RiskDataContractError(
                    f"duplicate sample_id across production shards: {sample_id}"
                )
            all_sample_ids.add(sample_id)
            ranked.append(
                (
                    _stable_score(
                        _SUBSET_MEMBERSHIP_DOMAIN,
                        dataset_manifest_digest=dataset.risk_dataset_manifest_digest,
                        seed=seed,
                        sample_id=sample_id,
                    ),
                    sample_id,
                    descriptor.shard_index,
                )
            )
        del sample
        del loaded
    if len(ranked) != dataset.sample_count:
        raise RiskDataContractError(
            "formally loaded sample count differs from loaded dataset"
        )

    ranked.sort(key=lambda item: (item[0], item[1]))
    selected = ranked[: min(max_samples, len(ranked))]
    sample_ids = tuple(item[1] for item in selected)
    by_shard: dict[int, set[str]] = {}
    for _, sample_id, shard_index in selected:
        by_shard.setdefault(shard_index, set()).add(sample_id)
    return (
        sample_ids,
        {index: frozenset(values) for index, values in by_shard.items()},
        frozenset(all_sample_ids),
    )


def _production_subset_plan_key(
    dataset_manifest_digest: str, subset_digest_sha256: str
) -> tuple[str, str]:
    return dataset_manifest_digest, subset_digest_sha256


def _store_verified_production_subset_plan(
    subset: ProductionRiskSubset,
    selected_by_shard: Mapping[int, frozenset[str]],
) -> None:
    frozen_mapping = tuple(
        (shard_index, frozenset(sample_ids))
        for shard_index, sample_ids in sorted(selected_by_shard.items())
    )
    if frozenset(
        sample_id
        for _, sample_ids in frozen_mapping
        for sample_id in sample_ids
    ) != frozenset(subset.sample_ids):
        raise RiskDataContractError(
            "internal verified subset plan does not bind exact sample membership"
        )
    plan = _VerifiedProductionSubsetPlan(
        dataset_manifest_digest=subset.dataset_manifest_digest,
        subset_digest_sha256=subset.sample_ids_digest_sha256,
        sample_ids=subset.sample_ids,
        seed=subset.seed,
        max_samples=subset.max_samples,
        selected_by_shard=frozen_mapping,
    )
    key = _production_subset_plan_key(
        subset.dataset_manifest_digest,
        subset.sample_ids_digest_sha256,
    )
    _PRODUCTION_SUBSET_PLAN_CACHE[key] = plan
    _PRODUCTION_SUBSET_PLAN_CACHE.move_to_end(key)
    while (
        len(_PRODUCTION_SUBSET_PLAN_CACHE)
        > _PRODUCTION_SUBSET_PLAN_CACHE_MAX_ENTRIES
    ):
        _PRODUCTION_SUBSET_PLAN_CACHE.popitem(last=False)


def _get_verified_production_subset_plan(
    subset: ProductionRiskSubset,
) -> dict[int, frozenset[str]] | None:
    key = _production_subset_plan_key(
        subset.dataset_manifest_digest,
        subset.sample_ids_digest_sha256,
    )
    plan = _PRODUCTION_SUBSET_PLAN_CACHE.get(key)
    if plan is None:
        return None
    if (
        plan.dataset_manifest_digest != subset.dataset_manifest_digest
        or plan.subset_digest_sha256 != subset.sample_ids_digest_sha256
        or plan.sample_ids != subset.sample_ids
        or plan.seed != subset.seed
        or plan.max_samples != subset.max_samples
    ):
        _PRODUCTION_SUBSET_PLAN_CACHE.pop(key, None)
        return None
    _PRODUCTION_SUBSET_PLAN_CACHE.move_to_end(key)
    return {
        shard_index: sample_ids
        for shard_index, sample_ids in plan.selected_by_shard
    }


def select_production_risk_subset(
    dataset: "LoadedRiskDataset",
    *,
    max_samples: int,
    seed: int,
) -> ProductionRiskSubset:
    """Select fixed membership with ``risk-production-subset-membership-v1``.

    The ordered tuple is separately bound by canonical finite JSON using the
    exact domain string ``risk-production-subset-v1``.
    """

    loaded_dataset = _validate_loaded_risk_dataset(dataset)
    limit = _require_positive_int(max_samples, "max_samples")
    selection_seed = _require_nonnegative_int(seed, "seed")
    sample_ids, selected_by_shard, _ = _rank_production_sample_ids(
        loaded_dataset,
        max_samples=limit,
        seed=selection_seed,
    )
    subset = ProductionRiskSubset(
        sample_ids=sample_ids,
        sample_ids_digest_sha256=_subset_digest(
            sample_ids,
            dataset_manifest_digest=loaded_dataset.risk_dataset_manifest_digest,
            seed=selection_seed,
            max_samples=limit,
        ),
        dataset_manifest_digest=loaded_dataset.risk_dataset_manifest_digest,
        seed=selection_seed,
        max_samples=limit,
    )
    _store_verified_production_subset_plan(subset, selected_by_shard)
    return subset


def _validate_production_subset(
    dataset: "LoadedRiskDataset",
    subset: object,
) -> dict[int, frozenset[str]]:
    if not isinstance(subset, ProductionRiskSubset):
        raise RiskDataContractError("subset must be a ProductionRiskSubset")
    subset_seed = _require_nonnegative_int(subset.seed, "subset.seed")
    max_samples = _require_positive_int(subset.max_samples, "subset.max_samples")
    dataset_digest = _require_sha256(
        subset.dataset_manifest_digest,
        "subset.dataset_manifest_digest",
    )
    if dataset_digest != dataset.risk_dataset_manifest_digest:
        raise RiskDataContractError(
            "subset dataset digest does not match loaded dataset digest"
        )
    subset_digest = _require_sha256(
        subset.sample_ids_digest_sha256,
        "subset.sample_ids_digest_sha256",
    )
    if not isinstance(subset.sample_ids, tuple):
        raise RiskDataContractError("subset.sample_ids must be a tuple")
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in subset.sample_ids):
        raise RiskDataContractError(
            "subset.sample_ids must contain non-empty strings"
        )
    if len(set(subset.sample_ids)) != len(subset.sample_ids):
        raise RiskDataContractError("subset.sample_ids must be unique")
    expected_count = min(max_samples, dataset.sample_count)
    if len(subset.sample_ids) != expected_count:
        raise RiskDataContractError(
            "subset sample count is inconsistent with max_samples and dataset size"
        )
    recomputed_digest = _subset_digest(
        subset.sample_ids,
        dataset_manifest_digest=dataset_digest,
        seed=subset_seed,
        max_samples=max_samples,
    )
    if subset_digest != recomputed_digest:
        raise RiskDataContractError("subset sample_ids_digest_sha256 mismatch")

    cached_plan = _get_verified_production_subset_plan(subset)
    if cached_plan is not None:
        return cached_plan

    expected_ids, selected_by_shard, all_sample_ids = _rank_production_sample_ids(
        dataset,
        max_samples=max_samples,
        seed=subset_seed,
    )
    unknown = sorted(set(subset.sample_ids) - set(all_sample_ids))
    if unknown:
        raise RiskDataContractError(f"subset contains unknown sample IDs: {unknown}")
    if subset.sample_ids != expected_ids:
        raise RiskDataContractError(
            "subset.sample_ids do not match deterministic production selection"
        )
    _store_verified_production_subset_plan(subset, selected_by_shard)
    return selected_by_shard


def collate_production_risk_samples(
    samples: Sequence[RiskSample],
    *,
    grid: GridSpec,
    expected_split: str,
    dataset_provenance: Mapping[str, object],
) -> RiskBatch:
    """Validate and collate authenticated production samples without oracle data."""

    if not isinstance(grid, GridSpec):
        raise RiskDataContractError("grid must be a GridSpec")
    split = _require_nonempty_string(expected_split, "expected_split")
    provenance = _validated_production_provenance(dataset_provenance)
    if not samples:
        raise RiskDataContractError("production risk sample batch must not be empty")

    sample_ids: list[str] = []
    validated_samples: list[RiskSample] = []
    for sample in samples:
        validated = _validate_production_sample(
            sample,
            grid=grid,
            expected_split=split,
        )
        if validated.sample_id in sample_ids:
            raise RiskDataContractError(
                "sample_id values must be unique within a production batch"
            )
        sample_ids.append(validated.sample_id)
        validated_samples.append(validated)

    model_inputs = {
        "bev_history": torch.from_numpy(
            np.stack([sample.bev_history for sample in validated_samples])
        ).to(dtype=torch.float32),
        "state_channels": torch.from_numpy(
            np.stack([sample.state_channels for sample in validated_samples])
        ).to(dtype=torch.float32),
        "trajectory_channels": torch.from_numpy(
            np.stack([sample.trajectory_channels for sample in validated_samples])
        ).to(dtype=torch.float32),
        "robot_state": torch.from_numpy(
            np.stack([sample.robot_state for sample in validated_samples])
        ).to(dtype=torch.float32),
    }
    if tuple(model_inputs) != MODEL_INPUT_KEYS:
        raise RiskDataContractError("internal production model input key drift")
    validate_model_input_mapping(model_inputs)
    targets = {
        "collision_label": torch.tensor(
            [sample.collision_label for sample in validated_samples],
            dtype=torch.float32,
        ),
        "risk_severity": torch.tensor(
            [sample.risk_severity for sample in validated_samples],
            dtype=torch.float32,
        ),
        "min_clearance": torch.tensor(
            [sample.min_clearance for sample in validated_samples],
            dtype=torch.float32,
        ),
        "near_miss": torch.tensor(
            [sample.near_miss for sample in validated_samples],
            dtype=torch.float32,
        ),
    }
    if tuple(targets) != TARGET_KEYS:
        raise RiskDataContractError("internal production target key drift")
    if not all(
        tensor.dtype == torch.float32 and torch.isfinite(tensor).all().item()
        for tensor in targets.values()
    ):
        raise RiskDataContractError("production risk targets contain NaN/Inf")
    return RiskBatch(
        model_inputs=model_inputs,
        targets=targets,
        sample_ids=tuple(sample_ids),
        split=split,
        provenance={
            "mode": "production",
            "schema_version": SCHEMA_VERSION,
            "channel_spec": frozen_channel_spec(),
            **provenance,
        },
    )


def _ordered_shard_indices(
    dataset: "LoadedRiskDataset", *, seed: int, epoch: int
) -> tuple[int, ...]:
    scored = [
        (
            _stable_score(
                _SHARD_ORDER_DOMAIN,
                dataset_manifest_digest=dataset.risk_dataset_manifest_digest,
                seed=seed,
                epoch=epoch,
                shard_index=descriptor.shard_index,
            ),
            descriptor.shard_index,
        )
        for descriptor in dataset.shards
    ]
    scored.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[1] for item in scored)


def _selected_row_score(
    dataset: "LoadedRiskDataset",
    *,
    seed: int,
    epoch: int,
    sample_id: str,
) -> bytes:
    return _stable_score(
        _ROW_ORDER_DOMAIN,
        dataset_manifest_digest=dataset.risk_dataset_manifest_digest,
        seed=seed,
        epoch=epoch,
        sample_id=sample_id,
    )


def _cursor_int(value: object, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RiskDataContractError(
            f"cursor.{field} must be an integer greater than or equal to {minimum}"
        )
    return value


def _validate_start_cursor(
    cursor: object,
    *,
    epoch: int,
    shard_order: tuple[int, ...],
    selected_counts: Mapping[int, int],
    dataset_digest: str,
    subset_digest: str,
    total_samples: int,
) -> tuple[int, int, int]:
    if not isinstance(cursor, RiskStreamCursor):
        raise RiskDataContractError("start_cursor must be a RiskStreamCursor")
    cursor_epoch = _cursor_int(cursor.epoch, "epoch")
    order_position = _cursor_int(
        cursor.shard_order_position, "shard_order_position"
    )
    shard_index = _cursor_int(cursor.shard_index, "shard_index", minimum=-1)
    row_position = _cursor_int(cursor.row_order_position, "row_order_position")
    samples_yielded = _cursor_int(cursor.samples_yielded, "samples_yielded")
    cursor_dataset_digest = _require_sha256(
        cursor.dataset_manifest_digest,
        "cursor.dataset_manifest_digest",
    )
    cursor_subset_digest = _require_sha256(
        cursor.subset_digest_sha256,
        "cursor.subset_digest_sha256",
    )
    if cursor_epoch != epoch:
        raise RiskDataContractError("cursor epoch does not match requested epoch")
    if cursor_dataset_digest != dataset_digest:
        raise RiskDataContractError("cursor dataset digest mismatch")
    if cursor_subset_digest != subset_digest:
        raise RiskDataContractError("cursor subset digest mismatch")
    if order_position > len(shard_order):
        raise RiskDataContractError("cursor shard_order_position is out of bounds")

    if order_position == len(shard_order):
        if shard_index != -1 or row_position != 0 or samples_yielded != total_samples:
            raise RiskDataContractError("cursor terminal fields are inconsistent")
        return order_position, row_position, samples_yielded

    expected_shard_index = shard_order[order_position]
    selected_count = selected_counts.get(expected_shard_index, 0)
    if shard_index != expected_shard_index:
        raise RiskDataContractError("cursor shard_index does not match epoch shard order")
    if selected_count == 0:
        raise RiskDataContractError("cursor points at a shard with no selected rows")
    if row_position >= selected_count:
        raise RiskDataContractError(
            "cursor row_order_position is outside selected-row ordering"
        )
    expected_yielded = sum(
        selected_counts.get(shard_order[position], 0)
        for position in range(order_position)
    ) + row_position
    if samples_yielded != expected_yielded:
        raise RiskDataContractError("cursor samples_yielded is inconsistent with position")
    return order_position, row_position, samples_yielded


def _initial_stream_position(
    shard_order: tuple[int, ...], selected_counts: Mapping[int, int]
) -> tuple[int, int, int]:
    for position, shard_index in enumerate(shard_order):
        if selected_counts.get(shard_index, 0) > 0:
            return position, 0, 0
    raise RiskDataContractError("production subset contains no streamable samples")


def _next_stream_cursor(
    *,
    epoch: int,
    shard_order: tuple[int, ...],
    selected_counts: Mapping[int, int],
    current_order_position: int,
    next_row_position: int,
    samples_yielded: int,
    dataset_digest: str,
    subset_digest: str,
) -> RiskStreamCursor:
    current_shard = shard_order[current_order_position]
    if next_row_position < selected_counts[current_shard]:
        return RiskStreamCursor(
            epoch=epoch,
            shard_order_position=current_order_position,
            shard_index=current_shard,
            row_order_position=next_row_position,
            samples_yielded=samples_yielded,
            dataset_manifest_digest=dataset_digest,
            subset_digest_sha256=subset_digest,
        )
    for position in range(current_order_position + 1, len(shard_order)):
        shard_index = shard_order[position]
        if selected_counts.get(shard_index, 0) > 0:
            return RiskStreamCursor(
                epoch=epoch,
                shard_order_position=position,
                shard_index=shard_index,
                row_order_position=0,
                samples_yielded=samples_yielded,
                dataset_manifest_digest=dataset_digest,
                subset_digest_sha256=subset_digest,
            )
    return RiskStreamCursor(
        epoch=epoch,
        shard_order_position=len(shard_order),
        shard_index=-1,
        row_order_position=0,
        samples_yielded=samples_yielded,
        dataset_manifest_digest=dataset_digest,
        subset_digest_sha256=subset_digest,
    )


def iter_production_risk_batches(
    dataset: "LoadedRiskDataset",
    *,
    subset: ProductionRiskSubset,
    batch_size: int,
    seed: int,
    epoch: int,
    start_cursor: RiskStreamCursor | None = None,
) -> Iterator[tuple[RiskBatch, RiskStreamCursor]]:
    """Stream one formal shard at a time in deterministic epoch order.

    Shards use domain ``risk-production-shard-order-v1``; selected rows use
    ``risk-production-row-order-v1``. Both scores bind dataset digest, seed,
    and epoch, while fixed subset membership never includes epoch.
    """

    loaded_dataset = _validate_loaded_risk_dataset(dataset)
    size = _require_positive_int(batch_size, "batch_size")
    stream_seed = _require_nonnegative_int(seed, "seed")
    stream_epoch = _require_nonnegative_int(epoch, "epoch")
    if not isinstance(subset, ProductionRiskSubset):
        raise RiskDataContractError("subset must be a ProductionRiskSubset")
    if stream_seed != subset.seed:
        raise RiskDataContractError("stream seed must match subset.seed")
    selected_by_shard = _validate_production_subset(loaded_dataset, subset)
    selected_counts = {
        shard_index: len(sample_ids)
        for shard_index, sample_ids in selected_by_shard.items()
    }
    shard_order = _ordered_shard_indices(
        loaded_dataset,
        seed=stream_seed,
        epoch=stream_epoch,
    )
    if start_cursor is None:
        order_position, row_position, _ = _initial_stream_position(
            shard_order, selected_counts
        )
    else:
        order_position, row_position, _ = _validate_start_cursor(
            start_cursor,
            epoch=stream_epoch,
            shard_order=shard_order,
            selected_counts=selected_counts,
            dataset_digest=loaded_dataset.risk_dataset_manifest_digest,
            subset_digest=subset.sample_ids_digest_sha256,
            total_samples=len(subset.sample_ids),
        )
        if order_position == len(shard_order):
            return

    for current_position in range(order_position, len(shard_order)):
        shard_index = shard_order[current_position]
        expected_sample_ids = selected_by_shard.get(shard_index)
        if not expected_sample_ids:
            continue
        descriptor = loaded_dataset.shards[shard_index]
        loaded = _load_validated_production_shard(loaded_dataset, descriptor)
        selected_rows: list[tuple[bytes, str, int, RiskSample]] = []
        found_ids: set[str] = set()
        for original_index, sample in enumerate(loaded.samples):
            if sample.sample_id not in expected_sample_ids:
                continue
            if sample.sample_id in found_ids:
                raise RiskDataContractError(
                    f"selected sample appears more than once: {sample.sample_id}"
                )
            found_ids.add(sample.sample_id)
            selected_rows.append(
                (
                    _selected_row_score(
                        loaded_dataset,
                        seed=stream_seed,
                        epoch=stream_epoch,
                        sample_id=sample.sample_id,
                    ),
                    sample.sample_id,
                    original_index,
                    sample,
                )
            )
        del sample
        if found_ids != set(expected_sample_ids):
            missing = sorted(set(expected_sample_ids) - found_ids)
            raise RiskDataContractError(
                f"selected sample IDs were not found exactly once: {missing}"
            )
        selected_rows.sort(key=lambda item: (item[0], item[1], item[2]))
        current_row = row_position if current_position == order_position else 0
        samples_before_shard = sum(
            selected_counts.get(shard_order[position], 0)
            for position in range(current_position)
        )
        while current_row < len(selected_rows):
            end = min(current_row + size, len(selected_rows))
            batch_samples = tuple(
                selected_rows[position][3] for position in range(current_row, end)
            )
            batch = collate_production_risk_samples(
                batch_samples,
                grid=loaded_dataset.grid,
                expected_split=loaded_dataset.split,
                dataset_provenance=loaded_dataset.provenance,
            )
            samples_yielded = samples_before_shard + end
            cursor = _next_stream_cursor(
                epoch=stream_epoch,
                shard_order=shard_order,
                selected_counts=selected_counts,
                current_order_position=current_position,
                next_row_position=end,
                samples_yielded=samples_yielded,
                dataset_digest=loaded_dataset.risk_dataset_manifest_digest,
                subset_digest=subset.sample_ids_digest_sha256,
            )
            yield batch, cursor
            del batch
            del cursor
            del batch_samples
            current_row = end
        del selected_rows
        del loaded


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


def load_production_risk_dataset(
    seal_root: str | Path,
    *,
    collection_root: str | Path | None = None,
    expected_split: str | None = None,
    expected_manifest_digest: str | None = None,
) -> "LoadedRiskDataset":
    """Delegate the production boundary to the strict dataset-seal loader.

    The optional defaults preserve the old one-argument fail-closed call: it is
    still routed through the v2 loader and cannot interpret a shard or legacy
    manifest.  Production callers provide all arguments explicitly.
    """

    from src.datasets.risk_dataset_seal import load_risk_dataset_seal

    return load_risk_dataset_seal(
        seal_root,
        collection_root=seal_root if collection_root is None else collection_root,
        expected_split=(
            "__unspecified_production_split__"
            if expected_split is None
            else expected_split
        ),
        expected_manifest_digest=expected_manifest_digest,
    )


__all__ = [
    "MODEL_INPUT_KEYS",
    "PRODUCTION_DATASET_LAYOUT_VERSION",
    "TARGET_KEYS",
    "ProductionRiskSubset",
    "RiskBatch",
    "RiskDataContractError",
    "RiskStreamCursor",
    "collate_production_risk_samples",
    "collate_risk_samples",
    "iter_production_risk_batches",
    "load_production_risk_dataset",
    "select_production_risk_subset",
    "validate_model_input_mapping",
    "validate_toy_dataset_manifest",
]
