"""Minimal authenticated SOP08 production smoke trainer.

Only ``one_shard_smoke`` is executable.  Larger stages and resume are explicit
gates so this module cannot accidentally be used for a scientific training run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import uuid
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from src.contracts import SCHEMA_VERSION
from src.datasets.risk_dataloader import (
    ProductionOccupancyBatch,
    ProductionRiskSubset,
    iter_production_occupancy_batches,
)
from src.datasets.risk_dataset_seal import LoadedRiskDataset
from src.datasets.risk_training_store import AuthenticatedOccupancySnapshot
from src.evaluation.risk_baselines import score_production_occupancy_baseline
from src.models.occupancy_baseline import (
    ConvGRUOccupancyPredictor,
    LearnedOccupancyRiskAggregator,
)


PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION = "sop08_production_training_v1"
PRODUCTION_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION = (
    "sop08_production_occupancy_checkpoint_v1"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_BLAKE2B128_RE = re.compile(r"[0-9a-f]{32}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_ARTIFACT_NAMES = frozenset(
    {"config_snapshot.json", "metrics.json", "final_checkpoint.pt"}
)
_PUBLICATION_NAMES = frozenset(
    {
        *_ARTIFACT_NAMES,
        "training_manifest.json",
        "checksums.sha256",
        ".producer-complete",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "training_layout_version",
        "mode",
        "stage",
        "config_digest_sha256",
        "provenance",
        "runtime_environment",
        "validation_status",
        "best_checkpoint",
        "train_sample_count",
        "artifact_sha256",
        "artifact_semantic_bindings",
        "semantic_digest_sha256",
    }
)
_CONFIG_SNAPSHOT_KEYS = frozenset(
    {
        "training_layout_version",
        "mode",
        "optimizer",
        "stage",
        "seed",
        "device",
        "hidden_channels",
        "convgru_kernel_size",
        "learned_aggregator_hidden_dim",
        "batch_size",
        "occupancy_epochs",
        "aggregator_epochs",
        "gradient_accumulation_steps",
        "occupancy_learning_rate",
        "aggregator_learning_rate",
        "weight_decay",
        "b2_tau_s",
        "b2_a_max_s",
        "sigma_time_s",
        "checkpoint_interval_steps",
        "scientific_claim_eligible",
    }
)
_METRICS_KEYS = frozenset(
    {
        "training_layout_version",
        "stage",
        "training_data_scale",
        "scientific_claim_eligible",
        "selected_sample_count",
        "global_statistics_sample_count",
        "train_sample_count",
        "occupancy_global_positive_count",
        "occupancy_global_negative_count",
        "collision_global_positive_count",
        "collision_global_negative_count",
        "occupancy_global_pos_weight",
        "collision_global_pos_weight",
        "occupancy_final_loss",
        "aggregator_final_loss",
        "optimizer_steps",
        "b3_state_digest_before_b4_sha256",
        "b3_state_digest_after_b4_sha256",
        "b3_frozen_during_b4",
        "baseline_score_summary",
        "test_samples_used_for_training_or_selection",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "schema_version",
        "g1_split_manifest_digest",
        "risk_dataset_manifest_digest",
        "dynamic_objects_config_digest",
        "target_type_policy_digest",
        "occupancy_sidecar_collection_digest_sha256",
        "subset_digest_sha256",
        "selected_sample_count",
        "consumed_sample_ids",
        "code_commit",
        "scientific_claim_eligible",
        "test_samples_used_for_training_or_selection",
    }
)
_MARKER_KEYS = frozenset(
    {
        "training_layout_version",
        "semantic_digest_sha256",
        "publication_instance_digest_sha256",
        "training_manifest_sha256",
        "checksums_sha256",
    }
)
_CHECKPOINT_KEYS = frozenset(
    {
        "checkpoint_layout_version",
        "config_digest_sha256",
        "provenance",
        "model_spec",
        "b3_model_state_dict",
        "b4_aggregator_state_dict",
        "b3_state_digest_sha256",
        "b4_state_digest_sha256",
        "checkpoint_semantic_digest_sha256",
    }
)


@dataclass(frozen=True)
class ProductionOccupancyTrainingConfig:
    """Frozen engineering-smoke hyperparameters."""

    stage: str
    seed: int
    device: str
    hidden_channels: int
    convgru_kernel_size: int
    learned_aggregator_hidden_dim: int
    batch_size: int
    occupancy_epochs: int
    aggregator_epochs: int
    gradient_accumulation_steps: int
    occupancy_learning_rate: float
    aggregator_learning_rate: float
    weight_decay: float
    b2_tau_s: float
    b2_a_max_s: float
    sigma_time_s: float
    checkpoint_interval_steps: int = 1

    def __post_init__(self) -> None:
        if self.stage not in {
            "one_shard_smoke",
            "real_1k_overfit",
            "formal_50k",
        }:
            raise ValueError("stage must be one_shard_smoke, real_1k_overfit, or formal_50k")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.device not in {"cpu", "cuda"} and not self.device.startswith("cuda:"):
            raise ValueError("device must be cpu or an allocated cuda device")
        for name in (
            "hidden_channels",
            "convgru_kernel_size",
            "learned_aggregator_hidden_dim",
            "batch_size",
            "occupancy_epochs",
            "aggregator_epochs",
            "gradient_accumulation_steps",
            "checkpoint_interval_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.convgru_kernel_size % 2 == 0:
            raise ValueError("convgru_kernel_size must be odd")
        if self.stage == "one_shard_smoke":
            for name in (
                "occupancy_epochs",
                "aggregator_epochs",
                "gradient_accumulation_steps",
                "checkpoint_interval_steps",
            ):
                if getattr(self, name) != 1:
                    raise ValueError(
                        f"{name} must equal 1 for one_shard_smoke"
                    )
        for name in (
            "occupancy_learning_rate",
            "aggregator_learning_rate",
            "b2_tau_s",
            "b2_a_max_s",
            "sigma_time_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(float(self.weight_decay)) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative and finite")


@dataclass(frozen=True)
class ProductionOccupancyTrainingResult:
    output_dir: Path
    manifest_path: Path
    metrics_path: Path
    final_checkpoint: Path
    semantic_digest_sha256: str
    publication_instance_digest_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_code_commit(value: object) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError("code_commit must be a 40-character lowercase Git commit")
    return value


def _require_blake2b128(value: object, name: str) -> str:
    if not isinstance(value, str) or _BLAKE2B128_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase BLAKE2b-128 digest")
    return value


def compute_global_binary_pos_weight(
    *, positive_count: int, negative_count: int, name: str
) -> float:
    """Return one dataset-global negative/positive class weight."""
    if isinstance(positive_count, bool) or not isinstance(positive_count, int):
        raise ValueError(f"{name} positive count must be an integer")
    if isinstance(negative_count, bool) or not isinstance(negative_count, int):
        raise ValueError(f"{name} negative count must be an integer")
    if positive_count < 1:
        raise ValueError(f"{name} requires at least one positive example")
    if negative_count < 1:
        raise ValueError(f"{name} requires at least one negative example")
    return float(negative_count) / float(positive_count)


def compute_weighted_binary_loss_sum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: float,
    name: str,
) -> tuple[torch.Tensor, int]:
    """Return weighted BCE-with-logits sum and its exact element count."""
    if not torch.is_tensor(logits) or not torch.is_tensor(targets):
        raise TypeError(f"{name} logits and targets must be tensors")
    if logits.dtype != torch.float32 or targets.dtype != torch.float32:
        raise ValueError(f"{name} logits and targets must be float32")
    if logits.shape != targets.shape or logits.numel() < 1:
        raise ValueError(f"{name} logits and targets must have equal non-empty shape")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError(f"{name} logits and targets must be finite")
    if bool(((targets != 0.0) & (targets != 1.0)).any()):
        raise ValueError(f"{name} targets must be binary")
    if not math.isfinite(float(pos_weight)) or float(pos_weight) <= 0.0:
        raise ValueError(f"{name} pos_weight must be positive and finite")
    weight = torch.as_tensor(float(pos_weight), dtype=torch.float32, device=logits.device)
    return (
        F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=weight,
            reduction="sum",
        ),
        int(targets.numel()),
    )


def _state_dict_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _config_snapshot(config: ProductionOccupancyTrainingConfig) -> dict[str, object]:
    return {
        "training_layout_version": PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION,
        "mode": "production",
        "optimizer": "AdamW",
        "scientific_claim_eligible": False,
        **asdict(config),
    }


def _runtime_environment(device: torch.device) -> dict[str, str]:
    if device.type == "cuda":
        actual = f"cuda:{torch.cuda.current_device()}"
    else:
        actual = "cpu"
    return {
        "runtime_environment_layout_version": "sop08_runtime_environment_v1",
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "actual_device": actual,
    }


def _move_batch(
    batch: ProductionOccupancyBatch, device: torch.device
) -> ProductionOccupancyBatch:
    return ProductionOccupancyBatch(
        model_inputs={key: value.to(device=device) for key, value in batch.model_inputs.items()},
        targets={key: value.to(device=device) for key, value in batch.targets.items()},
        query_inputs={key: value.to(device=device) for key, value in batch.query_inputs.items()},
        occupancy_targets={
            key: value.to(device=device) for key, value in batch.occupancy_targets.items()
        },
        sample_ids=batch.sample_ids,
        split=batch.split,
        provenance=batch.provenance,
    )


def _checkpoint_semantic_digest(checkpoint: Mapping[str, object]) -> str:
    projection = {
        key: checkpoint[key]
        for key in (
            "checkpoint_layout_version",
            "config_digest_sha256",
            "provenance",
            "model_spec",
            "b3_state_digest_sha256",
            "b4_state_digest_sha256",
        )
    }
    return _sha256_bytes(_canonical_json_bytes(projection))


def _manifest_semantic_digest(manifest: Mapping[str, object]) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {key: value for key, value in manifest.items() if key != "semantic_digest_sha256"}
        )
    )


def production_occupancy_publication_instance_digest(
    *, manifest_bytes: bytes, checksums_bytes: bytes
) -> str:
    """Bind exact manifest and checksum bytes without a circular manifest field."""
    if not isinstance(manifest_bytes, bytes) or not isinstance(checksums_bytes, bytes):
        raise TypeError("manifest_bytes and checksums_bytes must be bytes")
    digest = hashlib.sha256()
    digest.update(b"sop08-production-publication-instance-v1\0")
    digest.update(len(manifest_bytes).to_bytes(8, "big"))
    digest.update(manifest_bytes)
    digest.update(len(checksums_bytes).to_bytes(8, "big"))
    digest.update(checksums_bytes)
    return digest.hexdigest()


def _read_regular(path: Path) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"publication path must be a regular file: {path.name}")
    return path.read_bytes()


def _read_canonical_json(raw: bytes, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be canonical JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    try:
        canonical = _canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite canonical JSON") from exc
    if raw != canonical:
        raise ValueError(f"{name} is not canonical JSON")
    return value


def _require_exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_exact_float(value: object, name: str, *, positive: bool) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")
    if not positive and value < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite float")
    return value


def _validate_config_snapshot(value: Mapping[str, object]) -> None:
    if set(value) != _CONFIG_SNAPSHOT_KEYS:
        raise ValueError("production occupancy config snapshot keys mismatch")
    if (
        value["training_layout_version"]
        != PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION
        or value["mode"] != "production"
        or value["optimizer"] != "AdamW"
        or value["stage"] != "one_shard_smoke"
    ):
        raise ValueError("production occupancy config snapshot identity mismatch")
    if value["scientific_claim_eligible"] is not False:
        raise ValueError("config scientific_claim_eligible must be false")
    _require_exact_int(value["seed"], "config seed")
    for name in (
        "hidden_channels",
        "convgru_kernel_size",
        "learned_aggregator_hidden_dim",
        "batch_size",
    ):
        _require_exact_int(value[name], f"config {name}", minimum=1)
    if value["convgru_kernel_size"] % 2 == 0:
        raise ValueError("config convgru_kernel_size must be odd")
    for name in (
        "occupancy_epochs",
        "aggregator_epochs",
        "gradient_accumulation_steps",
        "checkpoint_interval_steps",
    ):
        if _require_exact_int(value[name], f"config {name}", minimum=1) != 1:
            raise ValueError(f"config {name} must equal 1 for one_shard_smoke")
    for name in (
        "occupancy_learning_rate",
        "aggregator_learning_rate",
        "b2_tau_s",
        "b2_a_max_s",
        "sigma_time_s",
    ):
        _require_exact_float(value[name], f"config {name}", positive=True)
    _require_exact_float(value["weight_decay"], "config weight_decay", positive=False)
    if not isinstance(value["device"], str) or (
        value["device"] not in {"cpu", "cuda"}
        and not value["device"].startswith("cuda:")
    ):
        raise ValueError("config device must be cpu or cuda")


def _validate_provenance(value: Mapping[str, object]) -> None:
    if set(value) != _PROVENANCE_KEYS:
        raise ValueError("production occupancy provenance keys mismatch")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("production occupancy provenance schema mismatch")
    for name in ("g1_split_manifest_digest", "target_type_policy_digest"):
        _require_blake2b128(value[name], f"provenance {name}")
    for name in (
        "risk_dataset_manifest_digest",
        "dynamic_objects_config_digest",
        "occupancy_sidecar_collection_digest_sha256",
        "subset_digest_sha256",
    ):
        _require_sha256(value[name], f"provenance {name}")
    _require_code_commit(value["code_commit"])
    selected = _require_exact_int(
        value["selected_sample_count"], "provenance selected_sample_count", minimum=1
    )
    consumed = value["consumed_sample_ids"]
    if (
        not isinstance(consumed, list)
        or not consumed
        or len(consumed) > selected
        or len(set(consumed)) != len(consumed)
        or any(not isinstance(item, str) or not item for item in consumed)
    ):
        raise ValueError("provenance consumed_sample_ids mismatch")
    if value["scientific_claim_eligible"] is not False:
        raise ValueError("provenance scientific_claim_eligible must be false")
    if value["test_samples_used_for_training_or_selection"] != 0 or type(
        value["test_samples_used_for_training_or_selection"]
    ) is not int:
        raise ValueError("provenance test samples must equal integer zero")


def _validate_metrics(value: Mapping[str, object]) -> None:
    if set(value) != _METRICS_KEYS:
        raise ValueError("production occupancy metrics keys mismatch")
    if (
        value["training_layout_version"]
        != PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION
        or value["stage"] != "one_shard_smoke"
        or value["training_data_scale"] != "one_shard_smoke"
    ):
        raise ValueError("production occupancy metrics identity mismatch")
    if value["scientific_claim_eligible"] is not False:
        raise ValueError("metrics scientific_claim_eligible must be false")
    if value["b3_frozen_during_b4"] is not True:
        raise ValueError("metrics must prove B3 was frozen during B4")
    if value["test_samples_used_for_training_or_selection"] != 0 or type(
        value["test_samples_used_for_training_or_selection"]
    ) is not int:
        raise ValueError("metrics test samples must equal integer zero")
    for name in (
        "selected_sample_count",
        "global_statistics_sample_count",
        "train_sample_count",
        "occupancy_global_positive_count",
        "occupancy_global_negative_count",
        "collision_global_positive_count",
        "collision_global_negative_count",
    ):
        _require_exact_int(value[name], f"metrics {name}", minimum=1)
    if value["optimizer_steps"] != 2 or type(value["optimizer_steps"]) is not int:
        raise ValueError("metrics optimizer_steps must equal integer two")
    for name in (
        "occupancy_global_pos_weight",
        "collision_global_pos_weight",
        "occupancy_final_loss",
        "aggregator_final_loss",
    ):
        _require_exact_float(value[name], f"metrics {name}", positive=True)
    for name in (
        "b3_state_digest_before_b4_sha256",
        "b3_state_digest_after_b4_sha256",
    ):
        _require_sha256(value[name], f"metrics {name}")
    summaries = value["baseline_score_summary"]
    if not isinstance(summaries, dict) or set(summaries) != {"B1", "B2", "B3", "B4"}:
        raise ValueError("metrics baseline_score_summary keys mismatch")
    for method, summary in summaries.items():
        if not isinstance(summary, dict) or set(summary) != {
            "sample_count",
            "minimum",
            "maximum",
            "mean",
        }:
            raise ValueError(f"metrics {method} summary keys mismatch")
        if _require_exact_int(
            summary["sample_count"], f"metrics {method} sample_count", minimum=1
        ) != value["train_sample_count"]:
            raise ValueError(f"metrics {method} sample count mismatch")
        minimum = _require_exact_float(
            summary["minimum"], f"metrics {method} minimum", positive=False
        )
        maximum = _require_exact_float(
            summary["maximum"], f"metrics {method} maximum", positive=False
        )
        mean = _require_exact_float(
            summary["mean"], f"metrics {method} mean", positive=False
        )
        if not 0.0 <= minimum <= mean <= maximum <= 1.0:
            raise ValueError(f"metrics {method} score bounds mismatch")


def load_production_occupancy_checkpoint(
    path: str | Path | io.BytesIO,
    *,
    expected_checkpoint_semantic_digest_sha256: str,
    expected_provenance: Mapping[str, object],
    expected_config_digest_sha256: str,
) -> dict[str, object]:
    """Safely load and fully authenticate one final SOP08 smoke checkpoint."""
    expected_semantic = _require_sha256(
        expected_checkpoint_semantic_digest_sha256,
        "expected_checkpoint_semantic_digest_sha256",
    )
    expected_config = _require_sha256(
        expected_config_digest_sha256, "expected_config_digest_sha256"
    )
    if isinstance(path, io.BytesIO):
        source: object = path
    else:
        source = io.BytesIO(_read_regular(Path(path)))
    try:
        value = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as exc:
        raise ValueError(f"unable to load production occupancy checkpoint: {exc}") from exc
    if not isinstance(value, dict) or set(value) != _CHECKPOINT_KEYS:
        raise ValueError("production occupancy checkpoint keys mismatch")
    if value["checkpoint_layout_version"] != PRODUCTION_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION:
        raise ValueError("production occupancy checkpoint layout mismatch")
    if value["config_digest_sha256"] != expected_config:
        raise ValueError("production occupancy checkpoint config digest mismatch")
    if not isinstance(expected_provenance, Mapping):
        raise ValueError("expected production occupancy provenance must be a mapping")
    _validate_provenance(expected_provenance)
    if value["provenance"] != dict(expected_provenance):
        raise ValueError("production occupancy checkpoint provenance mismatch")
    declared = _require_sha256(
        value["checkpoint_semantic_digest_sha256"],
        "checkpoint_semantic_digest_sha256",
    )
    if declared != expected_semantic or declared != _checkpoint_semantic_digest(value):
        raise ValueError("production occupancy checkpoint semantic digest mismatch")
    model_spec = value["model_spec"]
    if not isinstance(model_spec, dict) or set(model_spec) != {
        "hidden_channels",
        "convgru_kernel_size",
        "learned_aggregator_hidden_dim",
        "future_steps",
    }:
        raise ValueError("production occupancy checkpoint model_spec mismatch")
    for name in (
        "hidden_channels",
        "convgru_kernel_size",
        "learned_aggregator_hidden_dim",
        "future_steps",
    ):
        if type(model_spec[name]) is not int or model_spec[name] < 1:
            raise ValueError(
                f"production occupancy checkpoint model_spec.{name} must be a positive integer"
            )
    if model_spec["convgru_kernel_size"] % 2 == 0:
        raise ValueError(
            "production occupancy checkpoint model_spec.convgru_kernel_size must be odd"
        )
    if model_spec["future_steps"] != 15:
        raise ValueError("production occupancy checkpoint future_steps mismatch")
    states: list[Mapping[str, torch.Tensor]] = []
    for key in ("b3_model_state_dict", "b4_aggregator_state_dict"):
        state_value = value[key]
        if not isinstance(state_value, Mapping) or not state_value:
            raise ValueError(f"{key} must be a non-empty state dictionary")
        for tensor in state_value.values():
            if not torch.is_tensor(tensor) or tensor.dtype != torch.float32:
                raise ValueError(f"{key} tensors must be float32")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{key} tensors must be finite")
        states.append(state_value)
    if _state_dict_digest(states[0]) != value["b3_state_digest_sha256"]:
        raise ValueError("B3 model state digest mismatch")
    if _state_dict_digest(states[1]) != value["b4_state_digest_sha256"]:
        raise ValueError("B4 aggregator state digest mismatch")
    try:
        model = ConvGRUOccupancyPredictor(
            hidden_channels=model_spec["hidden_channels"],
            future_steps=15,
            kernel_size=model_spec["convgru_kernel_size"],
        )
        aggregator = LearnedOccupancyRiskAggregator(
            future_steps=15,
            hidden_dim=model_spec["learned_aggregator_hidden_dim"],
        )
        model.load_state_dict(states[0], strict=True)
        aggregator.load_state_dict(states[1], strict=True)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("production occupancy checkpoint state shape mismatch") from exc
    return value


def validate_production_occupancy_training_publication(
    root: str | Path,
    *,
    expected_publication_instance_digest_sha256: str,
) -> dict[str, object]:
    """Validate the exact closed SOP08 smoke publication."""
    expected_instance = _require_sha256(
        expected_publication_instance_digest_sha256,
        "expected_publication_instance_digest_sha256",
    )
    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("production occupancy publication root must be a directory")
    names = {path.name for path in directory.iterdir()}
    if names != _PUBLICATION_NAMES:
        raise ValueError("production occupancy publication file set mismatch")
    snapshots = {name: _read_regular(directory / name) for name in names}
    manifest = _read_canonical_json(
        snapshots["training_manifest.json"], name="training manifest"
    )
    if set(manifest) != _MANIFEST_KEYS:
        raise ValueError("production occupancy training manifest keys mismatch")
    if manifest["training_layout_version"] != PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION:
        raise ValueError("production occupancy training layout mismatch")
    if manifest["mode"] != "production" or manifest["stage"] != "one_shard_smoke":
        raise ValueError("production occupancy manifest mode/stage mismatch")
    semantic = _require_sha256(manifest["semantic_digest_sha256"], "semantic digest")
    if semantic != _manifest_semantic_digest(manifest):
        raise ValueError("production occupancy manifest semantic digest mismatch")
    checksum_lines = snapshots["checksums.sha256"].decode("ascii").splitlines()
    expected_checksum_names = sorted({*_ARTIFACT_NAMES, "training_manifest.json"})
    if len(checksum_lines) != len(expected_checksum_names):
        raise ValueError("production occupancy checksum file set mismatch")
    for line, name in zip(checksum_lines, expected_checksum_names):
        expected_line = f"{_sha256_bytes(snapshots[name])}  {name}"
        if line != expected_line:
            raise ValueError(f"production occupancy checksum mismatch: {name}")
    instance = production_occupancy_publication_instance_digest(
        manifest_bytes=snapshots["training_manifest.json"],
        checksums_bytes=snapshots["checksums.sha256"],
    )
    if instance != expected_instance:
        raise ValueError("production occupancy publication instance digest mismatch")
    marker = _read_canonical_json(
        snapshots[".producer-complete"], name="producer-complete marker"
    )
    if set(marker) != _MARKER_KEYS or marker != {
        "training_layout_version": PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION,
        "semantic_digest_sha256": semantic,
        "publication_instance_digest_sha256": instance,
        "training_manifest_sha256": _sha256_bytes(snapshots["training_manifest.json"]),
        "checksums_sha256": _sha256_bytes(snapshots["checksums.sha256"]),
    }:
        raise ValueError("production occupancy producer-complete marker mismatch")
    artifact_hashes = manifest["artifact_sha256"]
    bindings = manifest["artifact_semantic_bindings"]
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != _ARTIFACT_NAMES:
        raise ValueError("production occupancy artifact hashes mismatch")
    if not isinstance(bindings, dict) or set(bindings) != _ARTIFACT_NAMES:
        raise ValueError("production occupancy artifact semantic bindings mismatch")
    for name in _ARTIFACT_NAMES:
        if artifact_hashes[name] != _sha256_bytes(snapshots[name]):
            raise ValueError(f"production occupancy artifact hash mismatch: {name}")
    config = _read_canonical_json(snapshots["config_snapshot.json"], name="config snapshot")
    metrics = _read_canonical_json(snapshots["metrics.json"], name="metrics")
    _validate_config_snapshot(config)
    _validate_metrics(metrics)
    provenance = manifest["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("production occupancy provenance must be a mapping")
    _validate_provenance(provenance)
    config_digest = _sha256_bytes(snapshots["config_snapshot.json"])
    if manifest["config_digest_sha256"] != config_digest:
        raise ValueError("production occupancy config digest mismatch")
    if bindings["config_snapshot.json"] != config_digest:
        raise ValueError("production occupancy config semantic binding mismatch")
    if bindings["metrics.json"] != _sha256_bytes(snapshots["metrics.json"]):
        raise ValueError("production occupancy metrics semantic binding mismatch")
    checkpoint = load_production_occupancy_checkpoint(
        io.BytesIO(snapshots["final_checkpoint.pt"]),
        expected_checkpoint_semantic_digest_sha256=str(bindings["final_checkpoint.pt"]),
        expected_provenance=provenance,
        expected_config_digest_sha256=config_digest,
    )
    if (
        type(manifest["train_sample_count"]) is not int
        or manifest["train_sample_count"] != metrics["train_sample_count"]
        or manifest["train_sample_count"] != len(provenance["consumed_sample_ids"])
    ):
        raise ValueError("production occupancy train sample count mismatch")
    if checkpoint["b3_state_digest_sha256"] != metrics.get(
        "b3_state_digest_after_b4_sha256"
    ):
        raise ValueError("production occupancy B3 freeze evidence mismatch")
    return manifest


def _score_summary(scores: torch.Tensor) -> dict[str, float | int]:
    values = scores.detach().cpu()
    return {
        "sample_count": int(values.numel()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
        "mean": float(values.mean().item()),
    }


def train_production_occupancy_baselines(
    *,
    train_dataset: LoadedRiskDataset,
    train_subset: ProductionRiskSubset,
    sidecar_root: str | Path,
    config: ProductionOccupancyTrainingConfig,
    output_dir: str | Path,
    code_commit: str,
    resume_from: str | Path | None = None,
    resume_expected_publication_instance_digest_sha256: str | None = None,
    training_snapshot: AuthenticatedOccupancySnapshot | None = None,
) -> ProductionOccupancyTrainingResult:
    """Run exactly one B3 step and one frozen-B3/B4 step, then publish."""
    if not isinstance(config, ProductionOccupancyTrainingConfig):
        raise TypeError("config must be ProductionOccupancyTrainingConfig")
    destination = Path(os.path.abspath(os.fspath(output_dir)))
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    if resume_from is not None or resume_expected_publication_instance_digest_sha256 is not None:
        raise ValueError("resume is not implemented for the one_shard_smoke trainer")
    if config.stage != "one_shard_smoke":
        raise ValueError(f"{config.stage} is not implemented; only one_shard_smoke is available")
    checked_commit = _require_code_commit(code_commit)
    if not isinstance(train_dataset, LoadedRiskDataset):
        raise TypeError("train_dataset must be an authenticated LoadedRiskDataset")
    if train_dataset.split != "train" or train_dataset.grid.history_steps != 8:
        raise ValueError("SOP08 smoke requires authenticated train history_steps=8")
    if train_dataset.grid.future_steps != 15:
        raise ValueError("SOP08 smoke requires future_steps=15")
    grid = train_dataset.manifest.get("grid")
    if not isinstance(grid, Mapping) or not math.isclose(
        float(grid.get("sample_dt_s", math.nan)), 0.2, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("SOP08 smoke requires sample_dt_s=0.2")
    if not isinstance(train_subset, ProductionRiskSubset):
        raise TypeError("train_subset must be ProductionRiskSubset")
    if train_subset.dataset_manifest_digest != train_dataset.risk_dataset_manifest_digest:
        raise ValueError("training subset dataset digest mismatch")
    if training_snapshot is not None:
        if not isinstance(training_snapshot, AuthenticatedOccupancySnapshot):
            raise TypeError(
                "training_snapshot must be an AuthenticatedOccupancySnapshot"
            )
        if (
            training_snapshot.split != "train"
            or training_snapshot.source_identity.get(
                "risk_dataset_manifest_digest"
            )
            != train_dataset.risk_dataset_manifest_digest
            or training_snapshot.select_subset(
                max_samples=train_subset.max_samples,
                seed=train_subset.seed,
            )
            != train_subset
        ):
            raise ValueError(
                "authenticated occupancy snapshot does not match the training subset"
            )
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA smoke requires an allocated CUDA GPU")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(config.seed)
    first_batch: ProductionOccupancyBatch | None = None
    occupancy_positive = occupancy_negative = 0
    collision_positive = collision_negative = 0
    statistics_samples = 0
    stream = (
        iter_production_occupancy_batches(
            train_dataset,
            sidecar_root=sidecar_root,
            subset=train_subset,
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=0,
        )
        if training_snapshot is None
        else training_snapshot.iter_batches(
            subset=train_subset,
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=0,
        )
    )
    for batch, _ in stream:
        if first_batch is None:
            first_batch = batch
        hidden = batch.occupancy_targets["hidden_risk_occupancy"]
        collision = batch.targets["collision_label"]
        occupancy_positive += int(torch.count_nonzero(hidden).item())
        occupancy_negative += int(hidden.numel()) - int(torch.count_nonzero(hidden).item())
        collision_positive += int(torch.count_nonzero(collision).item())
        collision_negative += int(collision.numel()) - int(torch.count_nonzero(collision).item())
        statistics_samples += len(batch.sample_ids)
    if first_batch is None:
        raise ValueError("one_shard_smoke occupancy stream is empty")
    if statistics_samples != len(train_subset.sample_ids):
        raise ValueError("global statistics did not consume the authenticated subset")
    occupancy_pos_weight = compute_global_binary_pos_weight(
        positive_count=occupancy_positive,
        negative_count=occupancy_negative,
        name="occupancy",
    )
    collision_pos_weight = compute_global_binary_pos_weight(
        positive_count=collision_positive,
        negative_count=collision_negative,
        name="collision",
    )

    batch = _move_batch(first_batch, device)
    model = ConvGRUOccupancyPredictor(
        hidden_channels=config.hidden_channels,
        future_steps=15,
        kernel_size=config.convgru_kernel_size,
    ).to(device)
    aggregator = LearnedOccupancyRiskAggregator(
        future_steps=15,
        hidden_dim=config.learned_aggregator_hidden_dim,
    ).to(device)
    occupancy_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.occupancy_learning_rate,
        weight_decay=config.weight_decay,
    )
    model.train()
    occupancy_optimizer.zero_grad(set_to_none=True)
    occupancy_logits = model.predict_logits(batch.model_inputs["bev_history"])
    occupancy_loss_sum, occupancy_count = compute_weighted_binary_loss_sum(
        occupancy_logits,
        batch.occupancy_targets["hidden_risk_occupancy"],
        pos_weight=occupancy_pos_weight,
        name="occupancy",
    )
    occupancy_loss = occupancy_loss_sum / occupancy_count
    occupancy_loss.backward()
    occupancy_optimizer.step()
    if not math.isfinite(float(occupancy_loss.detach().cpu().item())):
        raise ValueError("occupancy smoke loss contains NaN/Inf")

    b3_before = _state_dict_digest(model.state_dict())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    with torch.no_grad():
        frozen_occupancy = model(batch.model_inputs["bev_history"]).detach()
    aggregator_optimizer = torch.optim.AdamW(
        aggregator.parameters(),
        lr=config.aggregator_learning_rate,
        weight_decay=config.weight_decay,
    )
    aggregator.train()
    aggregator_optimizer.zero_grad(set_to_none=True)
    collision_probability = aggregator(
        frozen_occupancy, batch.query_inputs["robot_endpoint_footprints"]
    )
    collision_weights = torch.where(
        batch.targets["collision_label"] > 0.5,
        torch.full_like(collision_probability, collision_pos_weight),
        torch.ones_like(collision_probability),
    )
    aggregator_loss = F.binary_cross_entropy(
        collision_probability,
        batch.targets["collision_label"],
        weight=collision_weights,
        reduction="sum",
    ) / int(collision_probability.numel())
    aggregator_loss.backward()
    aggregator_optimizer.step()
    if not math.isfinite(float(aggregator_loss.detach().cpu().item())):
        raise ValueError("aggregator smoke loss contains NaN/Inf")
    b3_after = _state_dict_digest(model.state_dict())
    if b3_before != b3_after:
        raise ValueError("B3 changed while fitting B4")

    summaries = {
        method: _score_summary(
            score_production_occupancy_baseline(
                method=method,
                model_inputs=batch.model_inputs,
                query_inputs=batch.query_inputs,
                occupancy_model=model,
                learned_aggregator=aggregator,
                b2_tau_s=config.b2_tau_s,
                b2_a_max_s=config.b2_a_max_s,
                sigma_time_s=config.sigma_time_s,
            )
        )
        for method in ("B1", "B2", "B3", "B4")
    }
    config_snapshot = _config_snapshot(config)
    config_bytes = _canonical_json_bytes(config_snapshot)
    config_digest = _sha256_bytes(config_bytes)
    sidecar_section = train_dataset.manifest.get("occupancy_sidecars")
    if not isinstance(sidecar_section, Mapping):
        raise ValueError("authenticated dataset lacks occupancy sidecars")
    sidecar_digest = _require_sha256(
        sidecar_section.get("collection_digest_sha256"),
        "occupancy sidecar collection digest",
    )
    provenance: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        **dict(train_dataset.provenance),
        "occupancy_sidecar_collection_digest_sha256": sidecar_digest,
        "subset_digest_sha256": train_subset.sample_ids_digest_sha256,
        "selected_sample_count": len(train_subset.sample_ids),
        "consumed_sample_ids": list(batch.sample_ids),
        "code_commit": checked_commit,
        "scientific_claim_eligible": False,
        "test_samples_used_for_training_or_selection": 0,
    }
    b3_state = _cpu_state(model)
    b4_state = _cpu_state(aggregator)
    checkpoint: dict[str, object] = {
        "checkpoint_layout_version": PRODUCTION_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
        "config_digest_sha256": config_digest,
        "provenance": provenance,
        "model_spec": {
            "hidden_channels": config.hidden_channels,
            "convgru_kernel_size": config.convgru_kernel_size,
            "learned_aggregator_hidden_dim": config.learned_aggregator_hidden_dim,
            "future_steps": 15,
        },
        "b3_model_state_dict": b3_state,
        "b4_aggregator_state_dict": b4_state,
        "b3_state_digest_sha256": b3_after,
        "b4_state_digest_sha256": _state_dict_digest(b4_state),
    }
    checkpoint["checkpoint_semantic_digest_sha256"] = _checkpoint_semantic_digest(
        checkpoint
    )
    metrics = {
        "training_layout_version": PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION,
        "stage": "one_shard_smoke",
        "training_data_scale": "one_shard_smoke",
        "scientific_claim_eligible": False,
        "selected_sample_count": len(train_subset.sample_ids),
        "global_statistics_sample_count": statistics_samples,
        "train_sample_count": len(batch.sample_ids),
        "occupancy_global_positive_count": occupancy_positive,
        "occupancy_global_negative_count": occupancy_negative,
        "collision_global_positive_count": collision_positive,
        "collision_global_negative_count": collision_negative,
        "occupancy_global_pos_weight": occupancy_pos_weight,
        "collision_global_pos_weight": collision_pos_weight,
        "occupancy_final_loss": float(occupancy_loss.detach().cpu().item()),
        "aggregator_final_loss": float(aggregator_loss.detach().cpu().item()),
        "optimizer_steps": 2,
        "b3_state_digest_before_b4_sha256": b3_before,
        "b3_state_digest_after_b4_sha256": b3_after,
        "b3_frozen_during_b4": True,
        "baseline_score_summary": summaries,
        "test_samples_used_for_training_or_selection": 0,
    }
    metrics_bytes = _canonical_json_bytes(metrics)
    runtime = _runtime_environment(device)
    staging = destination.with_name(
        f".{destination.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        (staging / "config_snapshot.json").write_bytes(config_bytes)
        (staging / "metrics.json").write_bytes(metrics_bytes)
        torch.save(checkpoint, staging / "final_checkpoint.pt")
        artifact_bytes = {
            name: (staging / name).read_bytes() for name in _ARTIFACT_NAMES
        }
        artifact_hashes = {
            name: _sha256_bytes(value) for name, value in artifact_bytes.items()
        }
        bindings = {
            "config_snapshot.json": artifact_hashes["config_snapshot.json"],
            "metrics.json": artifact_hashes["metrics.json"],
            "final_checkpoint.pt": checkpoint["checkpoint_semantic_digest_sha256"],
        }
        manifest: dict[str, object] = {
            "training_layout_version": PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION,
            "mode": "production",
            "stage": "one_shard_smoke",
            "config_digest_sha256": config_digest,
            "provenance": provenance,
            "runtime_environment": runtime,
            "validation_status": "unavailable_engineering_final_only",
            "best_checkpoint": None,
            "train_sample_count": len(batch.sample_ids),
            "artifact_sha256": artifact_hashes,
            "artifact_semantic_bindings": bindings,
        }
        manifest["semantic_digest_sha256"] = _manifest_semantic_digest(manifest)
        manifest_bytes = _canonical_json_bytes(manifest)
        (staging / "training_manifest.json").write_bytes(manifest_bytes)
        checksum_names = sorted({*_ARTIFACT_NAMES, "training_manifest.json"})
        checksums_bytes = "".join(
            f"{_sha256_bytes((staging / name).read_bytes())}  {name}\n"
            for name in checksum_names
        ).encode("ascii")
        (staging / "checksums.sha256").write_bytes(checksums_bytes)
        instance = production_occupancy_publication_instance_digest(
            manifest_bytes=manifest_bytes,
            checksums_bytes=checksums_bytes,
        )
        marker = {
            "training_layout_version": PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION,
            "semantic_digest_sha256": manifest["semantic_digest_sha256"],
            "publication_instance_digest_sha256": instance,
            "training_manifest_sha256": _sha256_bytes(manifest_bytes),
            "checksums_sha256": _sha256_bytes(checksums_bytes),
        }
        (staging / ".producer-complete").write_bytes(_canonical_json_bytes(marker))
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    validated = validate_production_occupancy_training_publication(
        destination,
        expected_publication_instance_digest_sha256=instance,
    )
    return ProductionOccupancyTrainingResult(
        output_dir=destination,
        manifest_path=destination / "training_manifest.json",
        metrics_path=destination / "metrics.json",
        final_checkpoint=destination / "final_checkpoint.pt",
        semantic_digest_sha256=str(validated["semantic_digest_sha256"]),
        publication_instance_digest_sha256=instance,
    )


__all__ = [
    "PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION",
    "ProductionOccupancyTrainingConfig",
    "ProductionOccupancyTrainingResult",
    "compute_global_binary_pos_weight",
    "compute_weighted_binary_loss_sum",
    "load_production_occupancy_checkpoint",
    "production_occupancy_publication_instance_digest",
    "train_production_occupancy_baselines",
    "validate_production_occupancy_training_publication",
]
