"""Minimal authenticated SOP08 production smoke trainer.

Only ``one_shard_smoke`` is executable.  Larger stages and resume are explicit
gates so this module cannot accidentally be used for a scientific training run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
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
    select_production_risk_subset,
)
from src.datasets.risk_dataset_seal import (
    LoadedRiskDataset,
    LoadedRiskDatasetFamily,
    load_risk_dataset_family,
)
from src.datasets.risk_training_store import AuthenticatedOccupancySnapshot
from src.evaluation.risk_baselines import score_production_occupancy_baseline
from src.models.occupancy_baseline import (
    ConvGRUOccupancyPredictor,
    LearnedOccupancyRiskAggregator,
)
from src.utils.atomic_publish import atomic_rename_noreplace


PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION = "sop08_production_training_v1"
PRODUCTION_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION = (
    "sop08_production_occupancy_checkpoint_v1"
)
FORMAL_PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION = (
    "sop08_formal_occupancy_training_v1"
)
FORMAL_PRODUCTION_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION = (
    "sop08_formal_occupancy_checkpoint_v1"
)
_FORMAL_TRAIN_SAMPLE_COUNT = 50_000

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
_FORMAL_CHECKPOINT_KEYS = frozenset(
    {
        "checkpoint_layout_version",
        "checkpoint_role",
        "config_digest_sha256",
        "provenance",
        "model_spec",
        "b3_model_state_dict",
        "b4_aggregator_state_dict",
        "b3_training_final_state_dict",
        "b3_state_digest_sha256",
        "b4_state_digest_sha256",
        "b3_training_final_state_digest_sha256",
        "selection",
        "checkpoint_semantic_digest_sha256",
    }
)
_FORMAL_ARTIFACT_NAMES = frozenset(
    {
        "config_snapshot.json",
        "metrics.json",
        "selection.json",
        "best_checkpoint.pt",
        "final_checkpoint.pt",
        "training_state.pt",
    }
)
_FORMAL_PUBLICATION_NAMES = frozenset(
    {
        *_FORMAL_ARTIFACT_NAMES,
        "training_manifest.json",
        "checksums.sha256",
        ".producer-complete",
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
    best_checkpoint: Path | None
    final_checkpoint: Path
    training_state_checkpoint: Path | None
    semantic_digest_sha256: str
    publication_instance_digest_sha256: str


@dataclass(frozen=True)
class FormalValidationSelection:
    """One immutable minimum-validation checkpoint selection."""

    phase: str
    epoch: int
    optimizer_step: int
    validation_loss: float
    state_dict: Mapping[str, torch.Tensor]


def update_formal_validation_selection(
    current: FormalValidationSelection | None,
    *,
    phase: str,
    epoch: int,
    optimizer_step: int,
    validation_loss: float,
    state_dict: Mapping[str, torch.Tensor],
) -> FormalValidationSelection:
    """Keep the lowest finite validation loss, resolving ties to the earliest."""

    if phase not in {"b3", "b4"}:
        raise ValueError("formal selection phase must be b3 or b4")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("formal selection epoch must be positive")
    if type(optimizer_step) is not int or optimizer_step < 0:
        raise ValueError("formal selection optimizer_step must be nonnegative")
    loss = float(validation_loss)
    if not math.isfinite(loss):
        raise ValueError("formal validation loss must be finite")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("formal selection state_dict must be non-empty")
    owned: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not name or not torch.is_tensor(tensor):
            raise ValueError("formal selection state_dict is invalid")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("formal selection state_dict must be finite")
        owned[name] = tensor.detach().cpu().clone()
    if current is not None:
        if not isinstance(current, FormalValidationSelection):
            raise TypeError("current selection must be FormalValidationSelection")
        if current.phase != phase:
            raise ValueError("formal selection phase mismatch")
        if loss >= current.validation_loss:
            return current
    return FormalValidationSelection(
        phase=phase,
        epoch=epoch,
        optimizer_step=optimizer_step,
        validation_loss=loss,
        state_dict=copy.deepcopy(owned),
    )


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


def _occupancy_sidecar_digest(dataset: LoadedRiskDataset, *, split: str) -> str:
    section = dataset.manifest.get("occupancy_sidecars")
    if not isinstance(section, Mapping):
        raise ValueError(f"formal_50k {split} dataset lacks occupancy sidecars")
    return _require_sha256(
        section.get("collection_digest_sha256"),
        f"formal_50k {split} occupancy sidecar digest",
    )


def _validate_formal_stage_inputs(
    *,
    train_dataset: LoadedRiskDataset,
    train_subset: ProductionRiskSubset,
    validation_dataset: LoadedRiskDataset | None,
    validation_sidecar_root: str | Path | None,
    dataset_family: LoadedRiskDatasetFamily | None,
    training_snapshot: AuthenticatedOccupancySnapshot | None,
    validation_snapshot: AuthenticatedOccupancySnapshot | None,
) -> LoadedRiskDatasetFamily:
    if validation_dataset is None or validation_sidecar_root is None:
        raise ValueError(
            "formal_50k requires an authenticated validation dataset and sidecar root"
        )
    if not isinstance(dataset_family, LoadedRiskDatasetFamily):
        raise ValueError("formal_50k requires an authenticated dataset family")
    if not isinstance(train_dataset, LoadedRiskDataset) or train_dataset.split != "train":
        raise ValueError("formal_50k train dataset must be the authenticated train split")
    if not isinstance(validation_dataset, LoadedRiskDataset) or validation_dataset.split != "val":
        raise ValueError("formal_50k validation dataset must be the authenticated val split")
    if not isinstance(train_subset, ProductionRiskSubset):
        raise ValueError("formal_50k train subset must be authenticated")
    if (
        len(train_subset.sample_ids) != _FORMAL_TRAIN_SAMPLE_COUNT
        or train_subset.max_samples != _FORMAL_TRAIN_SAMPLE_COUNT
    ):
        raise ValueError("formal_50k requires exactly 50,000 selected train samples")
    if train_subset.dataset_manifest_digest != train_dataset.risk_dataset_manifest_digest:
        raise ValueError("formal_50k train subset dataset digest mismatch")
    authenticated_family = load_risk_dataset_family(dataset_family.root)
    if authenticated_family != dataset_family:
        raise ValueError("formal_50k dataset family object is stale or forged")
    if authenticated_family.cross_split_audit.get(
        "global_cross_split_leakage"
    ) != "PROVEN":
        raise ValueError("formal_50k family leakage gate must be PROVEN")
    for split, dataset in (("train", train_dataset), ("val", validation_dataset)):
        member = authenticated_family.members.get(split)
        if not isinstance(member, Mapping) or (
            member.get("risk_dataset_manifest_digest")
            != dataset.risk_dataset_manifest_digest
            or member.get("sample_count") != dataset.sample_count
            or member.get("shard_count") != len(dataset.shards)
        ):
            raise ValueError(f"formal_50k family {split} member mismatch")
    for field in (
        "g1_split_manifest_digest",
        "dynamic_objects_config_digest",
        "target_type_policy_digest",
    ):
        if train_dataset.provenance[field] != validation_dataset.provenance[field]:
            raise ValueError(f"formal_50k train/val common contract mismatch: {field}")
    train_sidecar_digest = _occupancy_sidecar_digest(train_dataset, split="train")
    validation_sidecar_digest = _occupancy_sidecar_digest(
        validation_dataset,
        split="val",
    )
    for split, snapshot, dataset, sidecar_digest in (
        ("train", training_snapshot, train_dataset, train_sidecar_digest),
        ("val", validation_snapshot, validation_dataset, validation_sidecar_digest),
    ):
        if snapshot is None:
            continue
        if not isinstance(snapshot, AuthenticatedOccupancySnapshot):
            raise ValueError(f"formal_50k {split} snapshot has the wrong type")
        if (
            snapshot.split != split
            or snapshot.source_identity.get("risk_dataset_manifest_digest")
            != dataset.risk_dataset_manifest_digest
            or snapshot.source_identity.get(
                "occupancy_sidecar_collection_digest_sha256"
            )
            != sidecar_digest
        ):
            raise ValueError(f"formal_50k {split} snapshot identity mismatch")
    return authenticated_family


def _formal_sample_ids_digest(sample_ids: tuple[str, ...]) -> str:
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("formal sample IDs must be unique and non-empty")
    return _sha256_bytes(
        b"sop08-formal-sample-ids-v1\0" + _canonical_json_bytes(list(sample_ids))
    )


def _formal_batches(
    *,
    dataset: LoadedRiskDataset,
    subset: ProductionRiskSubset,
    sidecar_root: str | Path,
    config: ProductionOccupancyTrainingConfig,
    epoch: int,
    snapshot: AuthenticatedOccupancySnapshot | None,
):
    if snapshot is not None:
        return snapshot.iter_batches(
            subset=subset,
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=epoch,
        )
    return iter_production_occupancy_batches(
        dataset,
        sidecar_root=sidecar_root,
        subset=subset,
        batch_size=config.batch_size,
        seed=config.seed,
        epoch=epoch,
    )


def _formal_class_statistics(
    *,
    dataset: LoadedRiskDataset,
    subset: ProductionRiskSubset,
    sidecar_root: str | Path,
    config: ProductionOccupancyTrainingConfig,
    snapshot: AuthenticatedOccupancySnapshot | None,
) -> dict[str, int | float]:
    occupancy_positive = occupancy_negative = 0
    collision_positive = collision_negative = 0
    sample_count = 0
    for batch, _ in _formal_batches(
        dataset=dataset,
        subset=subset,
        sidecar_root=sidecar_root,
        config=config,
        epoch=0,
        snapshot=snapshot,
    ):
        hidden = batch.occupancy_targets["hidden_risk_occupancy"]
        collision = batch.targets["collision_label"]
        hidden_positive = int(torch.count_nonzero(hidden).item())
        collision_count = int(torch.count_nonzero(collision).item())
        occupancy_positive += hidden_positive
        occupancy_negative += int(hidden.numel()) - hidden_positive
        collision_positive += collision_count
        collision_negative += int(collision.numel()) - collision_count
        sample_count += len(batch.sample_ids)
    if sample_count != len(subset.sample_ids):
        raise ValueError("formal train statistics did not cover the exact subset")
    return {
        "sample_count": sample_count,
        "occupancy_positive": occupancy_positive,
        "occupancy_negative": occupancy_negative,
        "collision_positive": collision_positive,
        "collision_negative": collision_negative,
        "occupancy_pos_weight": compute_global_binary_pos_weight(
            positive_count=occupancy_positive,
            negative_count=occupancy_negative,
            name="occupancy",
        ),
        "collision_pos_weight": compute_global_binary_pos_weight(
            positive_count=collision_positive,
            negative_count=collision_negative,
            name="collision",
        ),
    }


def _finish_accumulated_step(
    optimizer: torch.optim.Optimizer,
    module: torch.nn.Module,
    *,
    normalization_count: int,
    phase: str,
) -> None:
    if normalization_count < 1:
        raise ValueError(f"formal {phase} accumulation window is empty")
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        parameter.grad.div_(float(normalization_count))
        if not bool(torch.isfinite(parameter.grad).all()):
            raise ValueError(f"formal {phase} gradients contain NaN/Inf")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _evaluate_formal_b3_validation_loss(
    model: ConvGRUOccupancyPredictor,
    *,
    dataset: LoadedRiskDataset,
    subset: ProductionRiskSubset,
    sidecar_root: str | Path,
    config: ProductionOccupancyTrainingConfig,
    snapshot: AuthenticatedOccupancySnapshot | None,
    device: torch.device,
    pos_weight: float,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for raw_batch, _ in _formal_batches(
            dataset=dataset,
            subset=subset,
            sidecar_root=sidecar_root,
            config=config,
            epoch=0,
            snapshot=snapshot,
        ):
            batch = _move_batch(raw_batch, device)
            logits = model.predict_logits(batch.model_inputs["bev_history"])
            loss_sum, element_count = compute_weighted_binary_loss_sum(
                logits,
                batch.occupancy_targets["hidden_risk_occupancy"],
                pos_weight=pos_weight,
                name="formal validation occupancy",
            )
            total += float(loss_sum.detach().cpu().item())
            count += element_count
    if count < 1:
        raise ValueError("formal B3 validation stream is empty")
    result = total / count
    if not math.isfinite(result):
        raise ValueError("formal B3 validation loss contains NaN/Inf")
    return result


def _weighted_collision_loss_sum(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: float,
    name: str,
) -> tuple[torch.Tensor, int]:
    if probability.shape != target.shape or probability.ndim != 1:
        raise ValueError(f"{name} probability/target must share [B]")
    if not bool(torch.isfinite(probability).all()) or bool(
        ((probability < 0.0) | (probability > 1.0)).any()
    ):
        raise ValueError(f"{name} probability must be finite in [0,1]")
    weights = torch.where(
        target > 0.5,
        torch.full_like(probability, float(pos_weight)),
        torch.ones_like(probability),
    )
    return (
        F.binary_cross_entropy(
            probability,
            target,
            weight=weights,
            reduction="sum",
        ),
        int(target.numel()),
    )


def _evaluate_formal_b4_validation_loss(
    model: ConvGRUOccupancyPredictor,
    aggregator: LearnedOccupancyRiskAggregator,
    *,
    dataset: LoadedRiskDataset,
    subset: ProductionRiskSubset,
    sidecar_root: str | Path,
    config: ProductionOccupancyTrainingConfig,
    snapshot: AuthenticatedOccupancySnapshot | None,
    device: torch.device,
    pos_weight: float,
) -> float:
    model.eval()
    aggregator.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for raw_batch, _ in _formal_batches(
            dataset=dataset,
            subset=subset,
            sidecar_root=sidecar_root,
            config=config,
            epoch=0,
            snapshot=snapshot,
        ):
            batch = _move_batch(raw_batch, device)
            occupancy = model(batch.model_inputs["bev_history"])
            probability = aggregator(
                occupancy,
                batch.query_inputs["robot_endpoint_footprints"],
            )
            loss_sum, sample_count = _weighted_collision_loss_sum(
                probability,
                batch.targets["collision_label"],
                pos_weight=pos_weight,
                name="formal validation collision",
            )
            total += float(loss_sum.detach().cpu().item())
            count += sample_count
    if count < 1:
        raise ValueError("formal B4 validation stream is empty")
    result = total / count
    if not math.isfinite(result):
        raise ValueError("formal B4 validation loss contains NaN/Inf")
    return result


def _selection_metadata(selection: FormalValidationSelection) -> dict[str, object]:
    return {
        "phase": selection.phase,
        "epoch": selection.epoch,
        "optimizer_step": selection.optimizer_step,
        "validation_loss": selection.validation_loss,
        "state_digest_sha256": _state_dict_digest(selection.state_dict),
        "tie_break_rule": "minimum_finite_loss_then_earliest_epoch_step",
    }


def _formal_checkpoint_semantic_digest(checkpoint: Mapping[str, object]) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                key: checkpoint[key]
                for key in (
                    "checkpoint_layout_version",
                    "checkpoint_role",
                    "config_digest_sha256",
                    "provenance",
                    "model_spec",
                    "b3_state_digest_sha256",
                    "b4_state_digest_sha256",
                    "b3_training_final_state_digest_sha256",
                    "selection",
                )
            }
        )
    )


def _build_formal_checkpoint(
    *,
    role: str,
    config: ProductionOccupancyTrainingConfig,
    config_digest: str,
    provenance: Mapping[str, object],
    b3_state: Mapping[str, torch.Tensor],
    b4_state: Mapping[str, torch.Tensor],
    b3_training_final_state: Mapping[str, torch.Tensor],
    b3_selection: FormalValidationSelection,
    b4_selection: FormalValidationSelection,
) -> dict[str, object]:
    checkpoint: dict[str, object] = {
        "checkpoint_layout_version": FORMAL_PRODUCTION_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
        "checkpoint_role": role,
        "config_digest_sha256": config_digest,
        "provenance": dict(provenance),
        "model_spec": {
            "hidden_channels": config.hidden_channels,
            "convgru_kernel_size": config.convgru_kernel_size,
            "learned_aggregator_hidden_dim": config.learned_aggregator_hidden_dim,
            "future_steps": 15,
        },
        "b3_model_state_dict": copy.deepcopy(dict(b3_state)),
        "b4_aggregator_state_dict": copy.deepcopy(dict(b4_state)),
        "b3_training_final_state_dict": copy.deepcopy(dict(b3_training_final_state)),
        "b3_state_digest_sha256": _state_dict_digest(b3_state),
        "b4_state_digest_sha256": _state_dict_digest(b4_state),
        "b3_training_final_state_digest_sha256": _state_dict_digest(
            b3_training_final_state
        ),
        "selection": {
            "b3": _selection_metadata(b3_selection),
            "b4": _selection_metadata(b4_selection),
        },
    }
    checkpoint["checkpoint_semantic_digest_sha256"] = (
        _formal_checkpoint_semantic_digest(checkpoint)
    )
    return checkpoint


def load_formal_production_occupancy_checkpoint(
    path: str | Path | io.BytesIO,
    *,
    expected_checkpoint_semantic_digest_sha256: str | None = None,
    expected_provenance: Mapping[str, object] | None = None,
    expected_config_digest_sha256: str | None = None,
) -> dict[str, object]:
    """Load a selected or final formal B3/B4 checkpoint."""

    source: object = path if isinstance(path, io.BytesIO) else io.BytesIO(_read_regular(Path(path)))
    try:
        value = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as exc:
        raise ValueError(f"unable to load formal occupancy checkpoint: {exc}") from exc
    if not isinstance(value, dict) or set(value) != _FORMAL_CHECKPOINT_KEYS:
        raise ValueError("formal occupancy checkpoint keys mismatch")
    if value["checkpoint_layout_version"] != FORMAL_PRODUCTION_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION:
        raise ValueError("formal occupancy checkpoint layout mismatch")
    if value["checkpoint_role"] not in {"best", "final"}:
        raise ValueError("formal occupancy checkpoint role mismatch")
    semantic = _require_sha256(
        value["checkpoint_semantic_digest_sha256"],
        "formal checkpoint semantic digest",
    )
    if semantic != _formal_checkpoint_semantic_digest(value):
        raise ValueError("formal occupancy checkpoint semantic digest mismatch")
    if expected_checkpoint_semantic_digest_sha256 is not None and semantic != _require_sha256(
        expected_checkpoint_semantic_digest_sha256,
        "expected formal checkpoint semantic digest",
    ):
        raise ValueError("expected formal occupancy checkpoint digest mismatch")
    if expected_config_digest_sha256 is not None and value["config_digest_sha256"] != _require_sha256(
        expected_config_digest_sha256,
        "expected formal config digest",
    ):
        raise ValueError("formal occupancy checkpoint config digest mismatch")
    if expected_provenance is not None and value["provenance"] != dict(expected_provenance):
        raise ValueError("formal occupancy checkpoint provenance mismatch")
    model_spec = value["model_spec"]
    if not isinstance(model_spec, Mapping) or set(model_spec) != {
        "hidden_channels",
        "convgru_kernel_size",
        "learned_aggregator_hidden_dim",
        "future_steps",
    }:
        raise ValueError("formal occupancy checkpoint model spec mismatch")
    if model_spec["future_steps"] != 15:
        raise ValueError("formal occupancy checkpoint future_steps mismatch")
    states = (
        value["b3_model_state_dict"],
        value["b4_aggregator_state_dict"],
        value["b3_training_final_state_dict"],
    )
    for state in states:
        if not isinstance(state, Mapping) or not state:
            raise ValueError("formal occupancy checkpoint state dictionary is invalid")
        if any(
            not torch.is_tensor(tensor)
            or tensor.dtype != torch.float32
            or not bool(torch.isfinite(tensor).all())
            for tensor in state.values()
        ):
            raise ValueError("formal occupancy checkpoint tensors must be finite float32")
    for state, digest_key in zip(
        states,
        (
            "b3_state_digest_sha256",
            "b4_state_digest_sha256",
            "b3_training_final_state_digest_sha256",
        ),
        strict=True,
    ):
        if _state_dict_digest(state) != value[digest_key]:
            raise ValueError(f"formal occupancy checkpoint {digest_key} mismatch")
    try:
        model = ConvGRUOccupancyPredictor(
            hidden_channels=int(model_spec["hidden_channels"]),
            future_steps=15,
            kernel_size=int(model_spec["convgru_kernel_size"]),
        )
        aggregator = LearnedOccupancyRiskAggregator(
            future_steps=15,
            hidden_dim=int(model_spec["learned_aggregator_hidden_dim"]),
        )
        model.load_state_dict(states[0], strict=True)
        aggregator.load_state_dict(states[1], strict=True)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("formal occupancy checkpoint state shape mismatch") from exc
    selection = value["selection"]
    if not isinstance(selection, Mapping) or set(selection) != {"b3", "b4"}:
        raise ValueError("formal occupancy checkpoint selection mismatch")
    for phase in ("b3", "b4"):
        record = selection[phase]
        if not isinstance(record, Mapping) or set(record) != {
            "phase",
            "epoch",
            "optimizer_step",
            "validation_loss",
            "state_digest_sha256",
            "tie_break_rule",
        }:
            raise ValueError(f"formal occupancy {phase} selection fields mismatch")
        if record["phase"] != phase or not math.isfinite(float(record["validation_loss"])):
            raise ValueError(f"formal occupancy {phase} selection is invalid")
    return value


def _formal_config_snapshot(config: ProductionOccupancyTrainingConfig) -> dict[str, object]:
    return {
        "training_layout_version": FORMAL_PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION,
        "mode": "production",
        "optimizer": "AdamW",
        "scientific_claim_eligible": True,
        **asdict(config),
    }


def _formal_training_manifest_semantic_digest(manifest: Mapping[str, object]) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {key: value for key, value in manifest.items() if key != "semantic_digest_sha256"}
        )
    )


def validate_formal_occupancy_training_publication(
    root: str | Path,
    *,
    expected_publication_instance_digest_sha256: str | None = None,
) -> dict[str, object]:
    """Validate the closed formal occupancy-training publication."""

    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("formal occupancy publication root must be a real directory")
    names = {path.name for path in directory.iterdir()}
    if names != _FORMAL_PUBLICATION_NAMES:
        raise ValueError("formal occupancy publication file set mismatch")
    snapshots = {name: _read_regular(directory / name) for name in names}
    manifest = _read_canonical_json(
        snapshots["training_manifest.json"],
        name="formal training manifest",
    )
    if manifest.get("training_layout_version") != FORMAL_PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION:
        raise ValueError("formal occupancy training layout mismatch")
    if manifest.get("stage") != "formal_50k" or manifest.get("mode") != "production":
        raise ValueError("formal occupancy manifest mode/stage mismatch")
    semantic = _require_sha256(manifest.get("semantic_digest_sha256"), "formal manifest semantic digest")
    if semantic != _formal_training_manifest_semantic_digest(manifest):
        raise ValueError("formal occupancy manifest semantic digest mismatch")
    checksum_lines = snapshots["checksums.sha256"].decode("ascii").splitlines()
    checksum_names = sorted({*_FORMAL_ARTIFACT_NAMES, "training_manifest.json"})
    if checksum_lines != [
        f"{_sha256_bytes(snapshots[name])}  {name}" for name in checksum_names
    ]:
        raise ValueError("formal occupancy checksum manifest mismatch")
    instance = production_occupancy_publication_instance_digest(
        manifest_bytes=snapshots["training_manifest.json"],
        checksums_bytes=snapshots["checksums.sha256"],
    )
    if expected_publication_instance_digest_sha256 is not None and instance != _require_sha256(
        expected_publication_instance_digest_sha256,
        "expected formal publication instance digest",
    ):
        raise ValueError("formal occupancy publication instance digest mismatch")
    marker = _read_canonical_json(
        snapshots[".producer-complete"],
        name="formal producer-complete marker",
    )
    if marker != {
        "training_layout_version": FORMAL_PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION,
        "semantic_digest_sha256": semantic,
        "publication_instance_digest_sha256": instance,
        "training_manifest_sha256": _sha256_bytes(snapshots["training_manifest.json"]),
        "checksums_sha256": _sha256_bytes(snapshots["checksums.sha256"]),
    }:
        raise ValueError("formal occupancy completion marker mismatch")
    artifact_hashes = manifest.get("artifact_sha256")
    bindings = manifest.get("artifact_semantic_bindings")
    if not isinstance(artifact_hashes, Mapping) or set(artifact_hashes) != _FORMAL_ARTIFACT_NAMES:
        raise ValueError("formal occupancy artifact hashes mismatch")
    if not isinstance(bindings, Mapping) or set(bindings) != _FORMAL_ARTIFACT_NAMES:
        raise ValueError("formal occupancy artifact bindings mismatch")
    for name in _FORMAL_ARTIFACT_NAMES:
        if artifact_hashes[name] != _sha256_bytes(snapshots[name]):
            raise ValueError(f"formal occupancy artifact hash mismatch: {name}")
    for name in ("best_checkpoint.pt", "final_checkpoint.pt"):
        load_formal_production_occupancy_checkpoint(
            io.BytesIO(snapshots[name]),
            expected_checkpoint_semantic_digest_sha256=str(bindings[name]),
            expected_provenance=manifest["provenance"],
            expected_config_digest_sha256=str(manifest["config_digest_sha256"]),
        )
    return manifest


def _train_formal_occupancy_baselines(
    *,
    train_dataset: LoadedRiskDataset,
    train_subset: ProductionRiskSubset,
    sidecar_root: str | Path,
    config: ProductionOccupancyTrainingConfig,
    output_dir: str | Path,
    code_commit: str,
    validation_dataset: LoadedRiskDataset,
    validation_sidecar_root: str | Path,
    dataset_family: LoadedRiskDatasetFamily,
    training_snapshot: AuthenticatedOccupancySnapshot | None,
    validation_snapshot: AuthenticatedOccupancySnapshot | None,
) -> ProductionOccupancyTrainingResult:
    destination = Path(os.path.abspath(os.fspath(output_dir)))
    checked_commit = _require_code_commit(code_commit)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("formal_50k requires an allocated CUDA device")
    validation_subset = (
        validation_snapshot.select_subset(
            max_samples=validation_dataset.sample_count,
            seed=config.seed,
        )
        if validation_snapshot is not None
        else select_production_risk_subset(
            validation_dataset,
            max_samples=validation_dataset.sample_count,
            seed=config.seed,
        )
    )
    statistics = _formal_class_statistics(
        dataset=train_dataset,
        subset=train_subset,
        sidecar_root=sidecar_root,
        config=config,
        snapshot=training_snapshot,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
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
    b3_selection: FormalValidationSelection | None = None
    b3_history: list[dict[str, object]] = []
    b3_train_history: list[float] = []
    b3_optimizer_steps = 0
    for epoch in range(config.occupancy_epochs):
        model.train()
        occupancy_optimizer.zero_grad(set_to_none=True)
        epoch_sum = 0.0
        epoch_count = 0
        window_count = 0
        window_batches = 0
        saw_batch = False
        for raw_batch, _ in _formal_batches(
            dataset=train_dataset,
            subset=train_subset,
            sidecar_root=sidecar_root,
            config=config,
            epoch=epoch,
            snapshot=training_snapshot,
        ):
            saw_batch = True
            batch = _move_batch(raw_batch, device)
            logits = model.predict_logits(batch.model_inputs["bev_history"])
            loss_sum, count = compute_weighted_binary_loss_sum(
                logits,
                batch.occupancy_targets["hidden_risk_occupancy"],
                pos_weight=float(statistics["occupancy_pos_weight"]),
                name="formal train occupancy",
            )
            loss_sum.backward()
            scalar_sum = float(loss_sum.detach().cpu().item())
            if not math.isfinite(scalar_sum):
                raise ValueError("formal B3 train loss contains NaN/Inf")
            epoch_sum += scalar_sum
            epoch_count += count
            window_count += count
            window_batches += 1
            if window_batches == config.gradient_accumulation_steps:
                _finish_accumulated_step(
                    occupancy_optimizer,
                    model,
                    normalization_count=window_count,
                    phase="B3",
                )
                b3_optimizer_steps += 1
                window_count = window_batches = 0
        if not saw_batch:
            raise ValueError("formal B3 train stream is empty")
        if window_batches:
            _finish_accumulated_step(
                occupancy_optimizer,
                model,
                normalization_count=window_count,
                phase="B3",
            )
            b3_optimizer_steps += 1
        train_loss = epoch_sum / epoch_count
        b3_train_history.append(train_loss)
        validation_loss = _evaluate_formal_b3_validation_loss(
            model,
            dataset=validation_dataset,
            subset=validation_subset,
            sidecar_root=validation_sidecar_root,
            config=config,
            snapshot=validation_snapshot,
            device=device,
            pos_weight=float(statistics["occupancy_pos_weight"]),
        )
        b3_history.append(
            {
                "epoch": epoch + 1,
                "optimizer_step": b3_optimizer_steps,
                "validation_loss": validation_loss,
            }
        )
        b3_selection = update_formal_validation_selection(
            b3_selection,
            phase="b3",
            epoch=epoch + 1,
            optimizer_step=b3_optimizer_steps,
            validation_loss=validation_loss,
            state_dict=model.state_dict(),
        )
    if b3_selection is None:
        raise ValueError("formal B3 completed without a validation selection")
    b3_training_final_state = _cpu_state(model)
    model.load_state_dict(b3_selection.state_dict, strict=True)
    b3_before = _state_dict_digest(model.state_dict())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    aggregator_optimizer = torch.optim.AdamW(
        aggregator.parameters(),
        lr=config.aggregator_learning_rate,
        weight_decay=config.weight_decay,
    )
    b4_selection: FormalValidationSelection | None = None
    b4_history: list[dict[str, object]] = []
    b4_train_history: list[float] = []
    b4_optimizer_steps = 0
    for epoch in range(config.aggregator_epochs):
        aggregator.train()
        aggregator_optimizer.zero_grad(set_to_none=True)
        epoch_sum = 0.0
        epoch_count = 0
        window_count = 0
        window_batches = 0
        saw_batch = False
        for raw_batch, _ in _formal_batches(
            dataset=train_dataset,
            subset=train_subset,
            sidecar_root=sidecar_root,
            config=config,
            epoch=epoch,
            snapshot=training_snapshot,
        ):
            saw_batch = True
            batch = _move_batch(raw_batch, device)
            with torch.no_grad():
                occupancy = model(batch.model_inputs["bev_history"]).detach()
            probability = aggregator(
                occupancy,
                batch.query_inputs["robot_endpoint_footprints"],
            )
            loss_sum, count = _weighted_collision_loss_sum(
                probability,
                batch.targets["collision_label"],
                pos_weight=float(statistics["collision_pos_weight"]),
                name="formal train collision",
            )
            loss_sum.backward()
            scalar_sum = float(loss_sum.detach().cpu().item())
            if not math.isfinite(scalar_sum):
                raise ValueError("formal B4 train loss contains NaN/Inf")
            epoch_sum += scalar_sum
            epoch_count += count
            window_count += count
            window_batches += 1
            if window_batches == config.gradient_accumulation_steps:
                _finish_accumulated_step(
                    aggregator_optimizer,
                    aggregator,
                    normalization_count=window_count,
                    phase="B4",
                )
                b4_optimizer_steps += 1
                window_count = window_batches = 0
        if not saw_batch:
            raise ValueError("formal B4 train stream is empty")
        if window_batches:
            _finish_accumulated_step(
                aggregator_optimizer,
                aggregator,
                normalization_count=window_count,
                phase="B4",
            )
            b4_optimizer_steps += 1
        b4_train_history.append(epoch_sum / epoch_count)
        validation_loss = _evaluate_formal_b4_validation_loss(
            model,
            aggregator,
            dataset=validation_dataset,
            subset=validation_subset,
            sidecar_root=validation_sidecar_root,
            config=config,
            snapshot=validation_snapshot,
            device=device,
            pos_weight=float(statistics["collision_pos_weight"]),
        )
        b4_history.append(
            {
                "epoch": epoch + 1,
                "optimizer_step": b4_optimizer_steps,
                "validation_loss": validation_loss,
            }
        )
        b4_selection = update_formal_validation_selection(
            b4_selection,
            phase="b4",
            epoch=epoch + 1,
            optimizer_step=b4_optimizer_steps,
            validation_loss=validation_loss,
            state_dict=aggregator.state_dict(),
        )
        if _state_dict_digest(model.state_dict()) != b3_before:
            raise ValueError("B3 changed while fitting formal B4")
    if b4_selection is None:
        raise ValueError("formal B4 completed without a validation selection")
    b3_after = _state_dict_digest(model.state_dict())
    if b3_after != b3_before:
        raise ValueError("B3 changed during formal B4 training")

    config_snapshot = _formal_config_snapshot(config)
    config_bytes = _canonical_json_bytes(config_snapshot)
    config_digest = _sha256_bytes(config_bytes)
    train_sidecar_digest = _occupancy_sidecar_digest(train_dataset, split="train")
    val_sidecar_digest = _occupancy_sidecar_digest(validation_dataset, split="val")
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "g1_split_manifest_digest": train_dataset.provenance["g1_split_manifest_digest"],
        "dynamic_objects_config_digest": train_dataset.provenance[
            "dynamic_objects_config_digest"
        ],
        "target_type_policy_digest": train_dataset.provenance[
            "target_type_policy_digest"
        ],
        "risk_dataset_family_digest": dataset_family.risk_dataset_family_digest,
        "global_cross_split_leakage": "PROVEN",
        "train_risk_dataset_manifest_digest": train_dataset.risk_dataset_manifest_digest,
        "validation_risk_dataset_manifest_digest": validation_dataset.risk_dataset_manifest_digest,
        "train_occupancy_sidecar_collection_digest_sha256": train_sidecar_digest,
        "validation_occupancy_sidecar_collection_digest_sha256": val_sidecar_digest,
        "train_subset_digest_sha256": train_subset.sample_ids_digest_sha256,
        "train_sample_ids_digest_sha256": _formal_sample_ids_digest(
            tuple(train_subset.sample_ids)
        ),
        "validation_sample_ids_digest_sha256": _formal_sample_ids_digest(
            tuple(validation_snapshot.sample_ids)
            if validation_snapshot is not None
            else tuple(validation_subset.sample_ids)
        ),
        "selected_train_sample_count": len(train_subset.sample_ids),
        "validation_sample_count": validation_dataset.sample_count,
        "code_commit": checked_commit,
        "scientific_claim_eligible": True,
        "test_samples_used_for_training_or_selection": 0,
    }
    best_checkpoint = _build_formal_checkpoint(
        role="best",
        config=config,
        config_digest=config_digest,
        provenance=provenance,
        b3_state=b3_selection.state_dict,
        b4_state=b4_selection.state_dict,
        b3_training_final_state=b3_training_final_state,
        b3_selection=b3_selection,
        b4_selection=b4_selection,
    )
    final_checkpoint = _build_formal_checkpoint(
        role="final",
        config=config,
        config_digest=config_digest,
        provenance=provenance,
        b3_state=b3_selection.state_dict,
        b4_state=_cpu_state(aggregator),
        b3_training_final_state=b3_training_final_state,
        b3_selection=b3_selection,
        b4_selection=b4_selection,
    )
    selection_payload = {
        "selection_layout_version": "sop08_formal_validation_selection_v1",
        "b3": _selection_metadata(b3_selection),
        "b4": _selection_metadata(b4_selection),
    }
    metrics = {
        "training_layout_version": FORMAL_PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION,
        "stage": "formal_50k",
        "train_sample_count": len(train_subset.sample_ids),
        "validation_sample_count": validation_dataset.sample_count,
        "class_statistics": statistics,
        "b3_train_loss_history": b3_train_history,
        "b4_train_loss_history": b4_train_history,
        "validation_history": {"b3": b3_history, "b4": b4_history},
        "selection": selection_payload,
        "b3_state_digest_before_b4_sha256": b3_before,
        "b3_state_digest_after_b4_sha256": b3_after,
        "b3_frozen_during_b4": True,
        "test_samples_used_for_training_or_selection": 0,
        "scientific_claim_eligible": True,
    }
    training_state = {
        "training_state_layout_version": "sop08_formal_training_state_v1",
        "phase": "complete",
        "config": asdict(config),
        "config_digest_sha256": config_digest,
        "provenance": provenance,
        "b3_model_state_dict": copy.deepcopy(dict(b3_selection.state_dict)),
        "b4_aggregator_state_dict": _cpu_state(aggregator),
        "b3_optimizer_state_dict": occupancy_optimizer.state_dict(),
        "b4_optimizer_state_dict": aggregator_optimizer.state_dict(),
        "b3_selection": {
            **_selection_metadata(b3_selection),
            "state_dict": copy.deepcopy(dict(b3_selection.state_dict)),
        },
        "b4_selection": {
            **_selection_metadata(b4_selection),
            "state_dict": copy.deepcopy(dict(b4_selection.state_dict)),
        },
        "torch_rng_state": torch.get_rng_state(),
    }

    staging = destination.with_name(
        f".{destination.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        (staging / "config_snapshot.json").write_bytes(config_bytes)
        (staging / "metrics.json").write_bytes(_canonical_json_bytes(metrics))
        (staging / "selection.json").write_bytes(
            _canonical_json_bytes(selection_payload)
        )
        torch.save(best_checkpoint, staging / "best_checkpoint.pt")
        torch.save(final_checkpoint, staging / "final_checkpoint.pt")
        torch.save(training_state, staging / "training_state.pt")
        artifact_bytes = {
            name: (staging / name).read_bytes() for name in _FORMAL_ARTIFACT_NAMES
        }
        artifact_hashes = {
            name: _sha256_bytes(value) for name, value in artifact_bytes.items()
        }
        bindings = dict(artifact_hashes)
        bindings["best_checkpoint.pt"] = best_checkpoint[
            "checkpoint_semantic_digest_sha256"
        ]
        bindings["final_checkpoint.pt"] = final_checkpoint[
            "checkpoint_semantic_digest_sha256"
        ]
        manifest: dict[str, object] = {
            "training_layout_version": FORMAL_PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION,
            "mode": "production",
            "stage": "formal_50k",
            "config_digest_sha256": config_digest,
            "provenance": provenance,
            "runtime_environment": _runtime_environment(device),
            "validation_status": "selected_on_authenticated_val",
            "validation_history": {"b3": b3_history, "b4": b4_history},
            "best_checkpoint": "best_checkpoint.pt",
            "final_checkpoint": "final_checkpoint.pt",
            "train_sample_count": len(train_subset.sample_ids),
            "validation_sample_count": validation_dataset.sample_count,
            "artifact_sha256": artifact_hashes,
            "artifact_semantic_bindings": bindings,
            "b3_state_digest_before_b4_sha256": b3_before,
            "b3_state_digest_after_b4_sha256": b3_after,
            "b3_frozen_during_b4": True,
            "test_samples_used_for_training_or_selection": 0,
        }
        manifest["semantic_digest_sha256"] = _formal_training_manifest_semantic_digest(
            manifest
        )
        manifest_bytes = _canonical_json_bytes(manifest)
        (staging / "training_manifest.json").write_bytes(manifest_bytes)
        checksum_names = sorted({*_FORMAL_ARTIFACT_NAMES, "training_manifest.json"})
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
            "training_layout_version": FORMAL_PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION,
            "semantic_digest_sha256": manifest["semantic_digest_sha256"],
            "publication_instance_digest_sha256": instance,
            "training_manifest_sha256": _sha256_bytes(manifest_bytes),
            "checksums_sha256": _sha256_bytes(checksums_bytes),
        }
        (staging / ".producer-complete").write_bytes(_canonical_json_bytes(marker))
        validate_formal_occupancy_training_publication(
            staging,
            expected_publication_instance_digest_sha256=instance,
        )
        atomic_rename_noreplace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    validated = validate_formal_occupancy_training_publication(
        destination,
        expected_publication_instance_digest_sha256=instance,
    )
    return ProductionOccupancyTrainingResult(
        output_dir=destination,
        manifest_path=destination / "training_manifest.json",
        metrics_path=destination / "metrics.json",
        best_checkpoint=destination / "best_checkpoint.pt",
        final_checkpoint=destination / "final_checkpoint.pt",
        training_state_checkpoint=destination / "training_state.pt",
        semantic_digest_sha256=str(validated["semantic_digest_sha256"]),
        publication_instance_digest_sha256=instance,
    )


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
    validation_dataset: LoadedRiskDataset | None = None,
    validation_sidecar_root: str | Path | None = None,
    dataset_family: LoadedRiskDatasetFamily | None = None,
    validation_snapshot: AuthenticatedOccupancySnapshot | None = None,
) -> ProductionOccupancyTrainingResult:
    """Run exactly one B3 step and one frozen-B3/B4 step, then publish."""
    if not isinstance(config, ProductionOccupancyTrainingConfig):
        raise TypeError("config must be ProductionOccupancyTrainingConfig")
    destination = Path(os.path.abspath(os.fspath(output_dir)))
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    if resume_from is not None or resume_expected_publication_instance_digest_sha256 is not None:
        raise ValueError("resume is not implemented for production occupancy training")
    if config.stage == "formal_50k":
        authenticated_family = _validate_formal_stage_inputs(
            train_dataset=train_dataset,
            train_subset=train_subset,
            validation_dataset=validation_dataset,
            validation_sidecar_root=validation_sidecar_root,
            dataset_family=dataset_family,
            training_snapshot=training_snapshot,
            validation_snapshot=validation_snapshot,
        )
        assert validation_dataset is not None
        assert validation_sidecar_root is not None
        return _train_formal_occupancy_baselines(
            train_dataset=train_dataset,
            train_subset=train_subset,
            sidecar_root=sidecar_root,
            config=config,
            output_dir=destination,
            code_commit=code_commit,
            validation_dataset=validation_dataset,
            validation_sidecar_root=validation_sidecar_root,
            dataset_family=authenticated_family,
            training_snapshot=training_snapshot,
            validation_snapshot=validation_snapshot,
        )
    if config.stage != "one_shard_smoke":
        raise ValueError(f"{config.stage} is not implemented; only one_shard_smoke is available")
    if any(
        value is not None
        for value in (
            validation_dataset,
            validation_sidecar_root,
            dataset_family,
            validation_snapshot,
        )
    ):
        raise ValueError("one_shard_smoke rejects validation and dataset family inputs")
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
        best_checkpoint=None,
        final_checkpoint=destination / "final_checkpoint.pt",
        training_state_checkpoint=None,
        semantic_digest_sha256=str(validated["semantic_digest_sha256"]),
        publication_instance_digest_sha256=instance,
    )


__all__ = [
    "PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION",
    "FORMAL_PRODUCTION_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION",
    "FORMAL_PRODUCTION_OCCUPANCY_TRAINING_LAYOUT_VERSION",
    "FormalValidationSelection",
    "ProductionOccupancyTrainingConfig",
    "ProductionOccupancyTrainingResult",
    "compute_global_binary_pos_weight",
    "compute_weighted_binary_loss_sum",
    "load_production_occupancy_checkpoint",
    "load_formal_production_occupancy_checkpoint",
    "production_occupancy_publication_instance_digest",
    "train_production_occupancy_baselines",
    "update_formal_validation_selection",
    "validate_production_occupancy_training_publication",
    "validate_formal_occupancy_training_publication",
]
