"""Deterministic, resumable SOP09 training over authenticated schema-3 shards."""

from __future__ import annotations

import copy
import ctypes
from dataclasses import asdict, dataclass
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import platform
import random
import re
import shutil
import stat
from typing import Literal, Mapping
import uuid

import numpy as np
import torch

from src.contracts import SCHEMA_VERSION
from src.datasets.risk_dataloader import (
    ProductionRiskSubset,
    RiskBatch,
    RiskDataContractError,
    RiskStreamCursor,
    iter_production_risk_batches,
    select_production_risk_subset,
)
from src.datasets.risk_dataset_seal import LoadedRiskDataset
from src.datasets.toy_risk_learning import frozen_channel_spec
from src.models.risk_model import (
    RiskModel,
    compute_risk_batch_loss,
    load_risk_checkpoint,
    save_risk_checkpoint,
)


PRODUCTION_RISK_TRAINING_LAYOUT_VERSION = "sop09_production_training_v1"
PRODUCTION_RISK_TRAINING_STATE_LAYOUT_VERSION = "sop09_training_state_v2"
_TRAINING_STAGES = frozenset(
    {"one_shard_smoke", "real_1k_overfit", "formal_50k"}
)
_MODEL_VARIANTS = frozenset({"r0", "r1"})
_TRAINING_MANIFEST_KEYS = frozenset(
    {
        "training_layout_version",
        "schema_version",
        "mode",
        "stage",
        "variant",
        "config_digest_sha256",
        "provenance",
        "train_sample_count",
        "optimizer_steps",
        "artifact_sha256",
        "artifact_semantic_bindings",
        "runtime_environment",
        "runtime_environment_digest_sha256",
        "resume_lineage",
        "semantic_digest_sha256",
        "publication_instance_digest_sha256",
    }
)
_PRODUCER_COMPLETE_KEYS = frozenset(
    {
        "training_layout_version",
        "semantic_digest_sha256",
        "publication_instance_digest_sha256",
    }
)
_INTERVAL_STATE_NAME = re.compile(r"^training_state_step_([0-9]{8})\.pt$")
_MANDATORY_RESULT_ARTIFACTS = frozenset(
    {
        "config_snapshot.json",
        "metrics.json",
        "final_checkpoint.pt",
        "training_state.pt",
    }
)
_TRAINING_STATE_KEYS = frozenset(
    {
        "training_state_layout_version",
        "model_config",
        "model_state_dict",
        "optimizer_state_dict",
        "config",
        "config_digest_sha256",
        "provenance",
        "optimizer_steps",
        "completed_epochs",
        "next_epoch",
        "next_cursor",
        "loss_history",
        "optimizer_step_loss_history",
        "validation_loss_history",
        "epoch_running_loss_sum",
        "epoch_running_sample_count",
        "initial_train_loss",
        "best_validation_loss",
        "best_validation_step",
        "best_model_state_dict",
        "rng_state",
        "training_state_semantic_digest_sha256",
    }
)


@dataclass(frozen=True)
class ProductionRiskTrainingConfig:
    """Frozen optimizer and publication configuration for one production run."""

    stage: Literal["one_shard_smoke", "real_1k_overfit", "formal_50k"]
    variant: Literal["r0", "r1"]
    seed: int
    device: str
    hidden_channels: int
    batch_size: int
    epochs: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    lambda_collision: float
    checkpoint_interval_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or self.stage not in _TRAINING_STAGES:
            raise RiskDataContractError(
                f"stage must be one of {sorted(_TRAINING_STAGES)}"
            )
        if not isinstance(self.variant, str) or self.variant not in _MODEL_VARIANTS:
            raise RiskDataContractError(
                f"variant must be one of {sorted(_MODEL_VARIANTS)}"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise RiskDataContractError("seed must be a non-negative integer")
        if not isinstance(self.device, str) or (
            self.device != "cpu"
            and self.device != "cuda"
            and not self.device.startswith("cuda:")
        ):
            raise RiskDataContractError("device must be cpu, cuda, or cuda:<index>")
        if self.device.startswith("cuda:") and not self.device[5:].isdigit():
            raise RiskDataContractError("CUDA device index must be a non-negative integer")
        for field in (
            "hidden_channels",
            "batch_size",
            "epochs",
            "gradient_accumulation_steps",
            "checkpoint_interval_steps",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise RiskDataContractError(f"{field} must be a positive integer")
        for field in ("learning_rate", "weight_decay", "lambda_collision"):
            value = getattr(self, field)
            if (
                type(value) not in {int, float}
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise RiskDataContractError(f"{field} must be finite and non-negative")
        if float(self.learning_rate) <= 0.0:
            raise RiskDataContractError("learning_rate must be positive")


@dataclass(frozen=True)
class ProductionRiskTrainingResult:
    """Absolute paths and semantic identity of one complete publication."""

    output_dir: Path
    best_checkpoint: Path | None
    final_checkpoint: Path
    training_state_checkpoint: Path
    metrics_path: Path
    manifest_path: Path
    semantic_digest_sha256: str


@dataclass(frozen=True)
class _ValidatedTrainingPublication:
    root: Path
    manifest: dict[str, object]
    target_state: dict[str, object]
    target_state_filename: str
    target_state_sha256: str
    interval_states: tuple[tuple[str, dict[str, object]], ...]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RiskDataContractError("training artifact must be finite canonical JSON") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RiskDataContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_code_commit(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RiskDataContractError(
            "code_commit must be a lowercase 40-character Git commit"
        )
    return value


def _runtime_environment(device: torch.device) -> dict[str, object]:
    if device.type == "cpu":
        actual_device = "cpu"
    else:
        index = torch.cuda.current_device() if device.index is None else device.index
        actual_device = f"cuda:{index}"
    return {
        "runtime_environment_layout_version": "sop09_runtime_environment_v1",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "actual_device": actual_device,
    }


def _runtime_environment_digest(value: Mapping[str, object]) -> str:
    return _sha256_bytes(_canonical_json_bytes(dict(value)))


def _validate_runtime_environment_snapshot(
    value: object, *, configured_device: str
) -> Mapping[str, object]:
    expected_keys = {
        "runtime_environment_layout_version",
        "python_version",
        "torch_version",
        "numpy_version",
        "actual_device",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RiskDataContractError("training runtime environment fields mismatch")
    if value.get("runtime_environment_layout_version") != (
        "sop09_runtime_environment_v1"
    ):
        raise RiskDataContractError("training runtime environment layout mismatch")
    for field in ("python_version", "torch_version", "numpy_version"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise RiskDataContractError(
                f"training runtime environment {field} must be non-empty"
            )
    actual_device = value.get("actual_device")
    if configured_device == "cpu":
        if actual_device != "cpu":
            raise RiskDataContractError("CPU runtime environment device mismatch")
    elif not isinstance(actual_device, str) or re.fullmatch(
        r"cuda:[0-9]+", actual_device
    ) is None:
        raise RiskDataContractError("CUDA runtime environment device mismatch")
    return value


def _sample_id_membership_digest(sample_ids: object) -> str:
    if not isinstance(sample_ids, (list, tuple, set, frozenset)):
        raise RiskDataContractError("consumed sample IDs must be a finite collection")
    values = sorted(sample_ids)
    if (
        not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise RiskDataContractError(
            "consumed sample IDs must be unique non-empty strings"
        )
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "domain": "sop09-consumed-sample-membership-v1",
                "sample_ids": values,
            }
        )
    )


def _training_data_scale(
    config: ProductionRiskTrainingConfig, selected_sample_count: int
) -> tuple[str, bool]:
    if config.stage == "one_shard_smoke":
        return "one_shard_smoke", False
    if config.stage == "real_1k_overfit":
        if selected_sample_count < 1000:
            return "fixture_standin", False
        return "real_1k", True
    return "formal_50k", True


def _config_snapshot(config: ProductionRiskTrainingConfig) -> dict[str, object]:
    return {
        "artifact_layout_version": PRODUCTION_RISK_TRAINING_LAYOUT_VERSION,
        "mode": "production",
        "optimizer": "AdamW",
        **asdict(config),
    }


def _config_digest(config: ProductionRiskTrainingConfig) -> str:
    return _sha256_bytes(_canonical_json_bytes(_config_snapshot(config)))


def _cursor_to_mapping(cursor: RiskStreamCursor | None) -> dict[str, object] | None:
    return None if cursor is None else asdict(cursor)


def _cursor_from_mapping(value: object) -> RiskStreamCursor | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RiskDataContractError("training state next_cursor must be a mapping or None")
    try:
        return RiskStreamCursor(**dict(value))
    except TypeError as exc:
        raise RiskDataContractError("training state next_cursor fields mismatch") from exc


def _tree_digest(value: object) -> str:
    """Deterministically bind nested tensor/array optimizer and RNG state."""

    digest = hashlib.sha256()

    def update(node: object) -> None:
        if isinstance(node, torch.Tensor):
            tensor = node.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(",".join(str(size) for size in tensor.shape).encode("ascii"))
            digest.update(b"\0")
            digest.update(tensor.numpy().tobytes(order="C"))
            return
        if isinstance(node, np.ndarray):
            array = np.ascontiguousarray(node)
            digest.update(b"ndarray\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(",".join(str(size) for size in array.shape).encode("ascii"))
            digest.update(b"\0")
            digest.update(array.tobytes(order="C"))
            return
        if isinstance(node, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(node, key=lambda item: repr(item)):
                update(key)
                update(node[key])
            return
        if isinstance(node, tuple):
            digest.update(b"tuple\0")
            for child in node:
                update(child)
            return
        if isinstance(node, list):
            digest.update(b"list\0")
            for child in node:
                update(child)
            return
        if node is None:
            digest.update(b"none\0")
            return
        if type(node) in {str, int, float, bool}:
            digest.update(type(node).__name__.encode("ascii"))
            digest.update(b"\0")
            digest.update(repr(node).encode("utf-8"))
            digest.update(b"\0")
            return
        raise RiskDataContractError(
            f"unsupported training-state value for semantic digest: {type(node).__name__}"
        )

    update(value)
    return digest.hexdigest()


def _training_state_semantic_digest(payload: Mapping[str, object]) -> str:
    projection = {
        "training_state_layout_version": payload.get("training_state_layout_version"),
        "model_config": payload.get("model_config"),
        "model_state_digest_sha256": _tree_digest(payload.get("model_state_dict")),
        "optimizer_state_digest_sha256": _tree_digest(
            payload.get("optimizer_state_dict")
        ),
        "config": payload.get("config"),
        "config_digest_sha256": payload.get("config_digest_sha256"),
        "provenance": payload.get("provenance"),
        "optimizer_steps": payload.get("optimizer_steps"),
        "completed_epochs": payload.get("completed_epochs"),
        "next_epoch": payload.get("next_epoch"),
        "next_cursor": payload.get("next_cursor"),
        "loss_history": payload.get("loss_history"),
        "optimizer_step_loss_history": payload.get(
            "optimizer_step_loss_history"
        ),
        "validation_loss_history": payload.get("validation_loss_history"),
        "epoch_running_loss_sum": payload.get("epoch_running_loss_sum"),
        "epoch_running_sample_count": payload.get("epoch_running_sample_count"),
        "initial_train_loss": payload.get("initial_train_loss"),
        "best_validation_loss": payload.get("best_validation_loss"),
        "best_validation_step": payload.get("best_validation_step"),
        "best_model_state_digest_sha256": (
            None
            if payload.get("best_model_state_dict") is None
            else _tree_digest(payload.get("best_model_state_dict"))
        ),
        "rng_state_digest_sha256": _tree_digest(payload.get("rng_state")),
    }
    return _sha256_bytes(_canonical_json_bytes(projection))


def _capture_rng_state(device: torch.device) -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(device=device).clone()
            if device.type == "cuda"
            else None
        ),
    }


def _restore_rng_state(value: object, *, device: torch.device) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise RiskDataContractError("training state RNG fields mismatch")
    try:
        random.setstate(value["python"])  # type: ignore[arg-type]
        numpy_state = value["numpy"]
        if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
            "bit_generator",
            "keys",
            "position",
            "has_gauss",
            "cached_gaussian",
        }:
            raise TypeError("NumPy RNG state fields mismatch")
        numpy_keys = numpy_state["keys"]
        if not isinstance(numpy_keys, torch.Tensor):
            raise TypeError("NumPy RNG keys must be a tensor")
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                numpy_keys.detach().cpu().numpy().astype(np.uint32, copy=False),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
        torch.set_rng_state(value["torch_cpu"])  # type: ignore[arg-type]
        cuda_state = value["torch_cuda"]
        if device.type == "cpu":
            if cuda_state is not None:
                raise TypeError("CPU training state must not contain CUDA RNG state")
        else:
            if not isinstance(cuda_state, torch.Tensor):
                raise TypeError("CUDA training state must contain one device RNG tensor")
            torch.cuda.set_rng_state(cuda_state, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RiskDataContractError("invalid training state RNG payload") from exc


def _write_training_state(path: Path, payload: Mapping[str, object]) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite training state: {destination}")
    frozen = copy.deepcopy(dict(payload))
    frozen["training_state_semantic_digest_sha256"] = (
        _training_state_semantic_digest(frozen)
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(frozen, temporary)
    temporary.replace(destination)
    return destination


def _load_training_state(path: Path | bytes | bytearray | memoryview) -> dict[str, object]:
    source: object = (
        io.BytesIO(bytes(path))
        if isinstance(path, (bytes, bytearray, memoryview))
        else Path(path)
    )
    try:
        value = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise RiskDataContractError(f"unable to load production training state: {exc}") from exc
    if not isinstance(value, dict) or set(value) != _TRAINING_STATE_KEYS:
        raise RiskDataContractError("production training-state keys mismatch")
    if value.get("training_state_layout_version") != (
        PRODUCTION_RISK_TRAINING_STATE_LAYOUT_VERSION
    ):
        raise RiskDataContractError("production training-state layout mismatch")
    declared = _require_sha256(
        value.get("training_state_semantic_digest_sha256"),
        "training_state_semantic_digest_sha256",
    )
    actual = _training_state_semantic_digest(value)
    if declared != actual:
        raise RiskDataContractError("training_state_semantic_digest_sha256 mismatch")
    return value


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _atomic_rename_directory_noreplace(source: Path, destination: Path) -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (OSError, AttributeError) as exc:
        raise OSError(
            errno.ENOSYS,
            "renameat2 is unavailable; refusing overwrite-capable fallback",
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    code = ctypes.get_errno()
    if code == errno.EEXIST:
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    raise OSError(code, os.strerror(code), os.fspath(destination))


def _move_batch(batch: RiskBatch, device: torch.device) -> RiskBatch:
    return RiskBatch(
        model_inputs={key: value.to(device=device) for key, value in batch.model_inputs.items()},
        targets={key: value.to(device=device) for key, value in batch.targets.items()},
        sample_ids=batch.sample_ids,
        split=batch.split,
        provenance=batch.provenance,
    )


def _dataset_provenance(dataset: LoadedRiskDataset) -> dict[str, str]:
    if not isinstance(dataset, LoadedRiskDataset):
        raise RiskDataContractError(
            "production training requires an authenticated LoadedRiskDataset"
        )
    expected = {
        "g1_split_manifest_digest",
        "risk_dataset_manifest_digest",
        "dynamic_objects_config_digest",
        "target_type_policy_digest",
    }
    if set(dataset.provenance) != expected:
        raise RiskDataContractError("authenticated train provenance fields mismatch")
    if dataset.risk_dataset_manifest_digest != dataset.provenance.get(
        "risk_dataset_manifest_digest"
    ):
        raise RiskDataContractError("train dataset manifest/provenance digest mismatch")
    grid = dataset.manifest.get("grid")
    if not isinstance(grid, Mapping):
        raise RiskDataContractError("production dataset grid manifest is missing")
    if dataset.grid.history_steps != 8 or grid.get("history_steps") != 8:
        raise RiskDataContractError(
            "SOP09 production training requires frozen history_steps=8"
        )
    if dataset.grid.future_steps != 15 or grid.get("future_steps") != 15:
        raise RiskDataContractError(
            "SOP09 production training requires frozen future_steps=15"
        )
    sample_dt_s = grid.get("sample_dt_s")
    if (
        type(sample_dt_s) not in {int, float}
        or not math.isclose(float(sample_dt_s), 0.2, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise RiskDataContractError(
            "SOP09 production training requires frozen sample_dt_s=0.2"
        )
    return dict(dataset.provenance)


def _validate_stage_gates(
    *,
    train_dataset: LoadedRiskDataset,
    train_subset: ProductionRiskSubset,
    config: ProductionRiskTrainingConfig,
    validation_dataset: LoadedRiskDataset | None,
    cross_split_audit: Mapping[str, object] | None,
) -> tuple[str | None, str | None, str]:
    _dataset_provenance(train_dataset)
    if train_dataset.split != "train":
        raise RiskDataContractError(
            "production trainer accepts only the authenticated train split"
        )
    if train_dataset.split in {"test", "calibration"}:
        raise RiskDataContractError("test/calibration data cannot enter training")
    if train_subset.dataset_manifest_digest != train_dataset.risk_dataset_manifest_digest:
        raise RiskDataContractError("training subset dataset digest mismatch")

    selected_count = len(train_subset.sample_ids)
    if config.stage == "real_1k_overfit":
        expected = min(1000, train_dataset.sample_count)
        if selected_count != expected:
            if train_dataset.sample_count < 1000:
                raise RiskDataContractError(
                    f"real_1k_overfit requires the complete {train_dataset.sample_count}-sample fixture"
                )
            raise RiskDataContractError(
                "real_1k_overfit requires exactly 1000 authenticated samples"
            )
    if config.stage in {"one_shard_smoke", "real_1k_overfit"}:
        if validation_dataset is not None:
            raise RiskDataContractError(
                "one_shard_smoke/real_1k_overfit rejects validation data"
            )
        if cross_split_audit is not None:
            raise RiskDataContractError(
                "one_shard_smoke/real_1k_overfit rejects cross-split audit claims"
            )
        return None, None, "NOT_PROVEN"
    raise RiskDataContractError(
        "formal_50k requires the authenticated risk_dataset_family_v1 loader; "
        "that strong typed boundary is not available yet, so raw audit mappings "
        "and JSON claims are rejected"
    )


def _checkpoint_provenance(
    *,
    dataset: LoadedRiskDataset,
    subset: ProductionRiskSubset,
    config: ProductionRiskTrainingConfig,
    config_digest: str,
    validation_digest: str | None,
    family_digest: str | None,
    leakage_status: str,
    code_commit: str,
    runtime_environment_digest: str,
    training_data_scale: str,
    scientific_claim_eligible: bool,
    consumed_sample_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "channel_spec": frozen_channel_spec(),
        "model_variant": config.variant,
        "config_digest": config_digest,
        **_dataset_provenance(dataset),
        "training_stage": config.stage,
        "training_subset_digest_sha256": subset.sample_ids_digest_sha256,
        "validation_risk_dataset_manifest_digest": validation_digest,
        "risk_dataset_family_digest": family_digest,
        "global_cross_split_leakage": leakage_status,
        "seed": config.seed,
        "code_commit": code_commit,
        "runtime_environment_digest_sha256": runtime_environment_digest,
        "training_data_scale": training_data_scale,
        "scientific_claim_eligible": scientific_claim_eligible,
        "selected_sample_count": len(subset.sample_ids),
        "consumed_sample_count": len(consumed_sample_ids),
        "consumed_sample_ids_digest_sha256": _sample_id_membership_digest(
            consumed_sample_ids
        ),
    }


def _evaluate_loss(
    model: RiskModel,
    *,
    dataset: LoadedRiskDataset,
    subset: ProductionRiskSubset,
    config: ProductionRiskTrainingConfig,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.eval()
    weighted_loss = 0.0
    sample_count = 0
    crossings = 0
    quantile_comparisons = 0
    with torch.no_grad():
        for raw_batch, _ in iter_production_risk_batches(
            dataset,
            subset=subset,
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=epoch,
        ):
            batch = _move_batch(raw_batch, device)
            output, losses = compute_risk_batch_loss(
                model, batch, lambda_collision=config.lambda_collision
            )
            batch_size = len(batch.sample_ids)
            value = losses["total"]
            if not torch.isfinite(value).item():
                raise RiskDataContractError("production evaluation loss contains NaN/Inf")
            weighted_loss += float(value.detach().cpu().item()) * batch_size
            sample_count += batch_size
            comparison = output["quantiles"][:, 1:] < output["quantiles"][:, :-1]
            crossings += int(torch.count_nonzero(comparison).item())
            quantile_comparisons += comparison.numel()
    if sample_count != len(subset.sample_ids) or sample_count < 1:
        raise RiskDataContractError("production evaluation sample count mismatch")
    return weighted_loss / sample_count, crossings / max(1, quantile_comparisons)


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _make_training_state(
    *,
    model: RiskModel,
    optimizer: torch.optim.Optimizer,
    config: ProductionRiskTrainingConfig,
    config_digest: str,
    provenance: Mapping[str, object],
    optimizer_steps: int,
    completed_epochs: int,
    next_epoch: int,
    next_cursor: RiskStreamCursor | None,
    loss_history: list[float],
    optimizer_step_loss_history: list[float],
    validation_loss_history: list[float],
    epoch_running_loss_sum: float,
    epoch_running_sample_count: int,
    initial_train_loss: float,
    best_validation_loss: float | None,
    best_validation_step: int | None,
    best_model_state_dict: Mapping[str, torch.Tensor] | None,
    device: torch.device,
) -> dict[str, object]:
    return {
        "training_state_layout_version": (
            PRODUCTION_RISK_TRAINING_STATE_LAYOUT_VERSION
        ),
        "model_config": model.export_config(),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "config": asdict(config),
        "config_digest_sha256": config_digest,
        "provenance": copy.deepcopy(dict(provenance)),
        "optimizer_steps": optimizer_steps,
        "completed_epochs": completed_epochs,
        "next_epoch": next_epoch,
        "next_cursor": _cursor_to_mapping(next_cursor),
        "loss_history": list(loss_history),
        "optimizer_step_loss_history": list(optimizer_step_loss_history),
        "validation_loss_history": list(validation_loss_history),
        "epoch_running_loss_sum": float(epoch_running_loss_sum),
        "epoch_running_sample_count": epoch_running_sample_count,
        "initial_train_loss": float(initial_train_loss),
        "best_validation_loss": best_validation_loss,
        "best_validation_step": best_validation_step,
        "best_model_state_dict": (
            None
            if best_model_state_dict is None
            else copy.deepcopy(dict(best_model_state_dict))
        ),
        "rng_state": _capture_rng_state(device),
    }


def _validate_resume_config(
    saved: object, current: ProductionRiskTrainingConfig
) -> ProductionRiskTrainingConfig:
    if not isinstance(saved, Mapping):
        raise RiskDataContractError("resume training config must be a mapping")
    try:
        prior = ProductionRiskTrainingConfig(**dict(saved))
    except (TypeError, ValueError) as exc:
        raise RiskDataContractError("resume training config is invalid") from exc
    prior_values = asdict(prior)
    current_values = asdict(current)
    for field in prior_values:
        if field == "epochs":
            continue
        if prior_values[field] != current_values[field]:
            raise RiskDataContractError(
                f"resume config mismatch for {field}; only epochs may increase"
            )
    if current.epochs < prior.epochs:
        raise RiskDataContractError("resume config epochs may only increase")
    return prior


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json_bytes(value))


def _write_checksum_manifest(root: Path) -> None:
    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{_sha256_file(path)}  {path.name}\n" for path in files]
    (root / "checksums.sha256").write_text("".join(lines), encoding="utf-8")


def _read_canonical_json_bytes(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RiskDataContractError(f"{label} must be strict finite JSON") from exc
    if not isinstance(value, dict):
        raise RiskDataContractError(f"{label} must be a JSON mapping")
    if raw != _canonical_json_bytes(value):
        raise RiskDataContractError(f"{label} must use exact canonical JSON encoding")
    return value


def _snapshot_direct_regular_files(root: Path) -> dict[str, bytes]:
    """Read each direct file once through an O_NOFOLLOW descriptor."""

    try:
        names = sorted(os.listdir(root))
    except OSError as exc:
        raise RiskDataContractError(
            f"unable to enumerate published training artifact: {exc}"
        ) from exc
    snapshots: dict[str, bytes] = {}
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    for name in names:
        if not name or name in {".", ".."} or "/" in name:
            raise RiskDataContractError("published training artifact filename is invalid")
        try:
            descriptor = os.open(root / name, flags)
        except OSError as exc:
            raise RiskDataContractError(
                f"published artifact is not a direct regular file: {name}"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise RiskDataContractError(
                    f"published artifact is not a direct regular file: {name}"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            snapshots[name] = b"".join(chunks)
        finally:
            os.close(descriptor)
    return snapshots


def _validate_checksum_snapshot(snapshots: Mapping[str, bytes]) -> None:
    raw = snapshots.get("checksums.sha256")
    if raw is None:
        raise RiskDataContractError(
            "complete published training artifact requires checksums.sha256"
        )
    try:
        raw_lines = raw.decode("utf-8").splitlines(keepends=True)
    except UnicodeError as exc:
        raise RiskDataContractError("training checksum manifest is not UTF-8") from exc
    entries: dict[str, str] = {}
    for raw_line in raw_lines:
        if not raw_line.endswith("\n"):
            raise RiskDataContractError("training checksum manifest is not canonical")
        digest, separator, name = raw_line[:-1].partition("  ")
        if (
            not separator
            or not name
            or "/" in name
            or name in entries
            or name == "checksums.sha256"
        ):
            raise RiskDataContractError("training checksum manifest is malformed")
        entries[name] = _require_sha256(digest, f"checksum for {name}")
    expected = set(snapshots) - {"checksums.sha256"}
    if set(entries) != expected:
        raise RiskDataContractError("training checksum manifest file set mismatch")
    canonical_lines = [f"{entries[name]}  {name}\n" for name in sorted(entries)]
    if raw_lines != canonical_lines:
        raise RiskDataContractError("training checksum manifest is not canonical")
    for name, digest in entries.items():
        if _sha256_bytes(snapshots[name]) != digest:
            raise RiskDataContractError(f"training artifact checksum mismatch: {name}")


def _training_manifest_semantic_digest(manifest: Mapping[str, object]) -> str:
    bindings = manifest.get("artifact_semantic_bindings")
    if not isinstance(bindings, Mapping):
        raise RiskDataContractError(
            "training manifest artifact_semantic_bindings must be a mapping"
        )
    result_names = set(_MANDATORY_RESULT_ARTIFACTS)
    if "best_checkpoint.pt" in bindings:
        result_names.add("best_checkpoint.pt")
    if not result_names.issubset(bindings):
        raise RiskDataContractError(
            "training manifest lacks mandatory final-result semantic bindings"
        )
    result_bindings = {
        name: bindings[name]
        for name in sorted(result_names)
    }
    projection = {
        "training_layout_version": manifest.get("training_layout_version"),
        "schema_version": manifest.get("schema_version"),
        "mode": manifest.get("mode"),
        "stage": manifest.get("stage"),
        "variant": manifest.get("variant"),
        "config_digest_sha256": manifest.get("config_digest_sha256"),
        "provenance": manifest.get("provenance"),
        "train_sample_count": manifest.get("train_sample_count"),
        "optimizer_steps": manifest.get("optimizer_steps"),
        "runtime_environment": manifest.get("runtime_environment"),
        "runtime_environment_digest_sha256": manifest.get(
            "runtime_environment_digest_sha256"
        ),
        "result_artifact_semantic_bindings": result_bindings,
    }
    return _sha256_bytes(_canonical_json_bytes(projection))


def _training_publication_instance_digest(manifest: Mapping[str, object]) -> str:
    projection = {
        key: value
        for key, value in manifest.items()
        if key != "publication_instance_digest_sha256"
    }
    return _sha256_bytes(_canonical_json_bytes(projection))


def _validate_resume_lineage(value: object) -> None:
    if value is None:
        return
    expected_keys = {
        "parent_scientific_digest_sha256",
        "parent_publication_instance_digest_sha256",
        "resume_state_filename",
        "resume_state_file_sha256",
        "resume_state_semantic_digest_sha256",
        "resume_optimizer_step",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RiskDataContractError("training manifest resume_lineage fields mismatch")
    for field in (
        "parent_scientific_digest_sha256",
        "parent_publication_instance_digest_sha256",
        "resume_state_file_sha256",
        "resume_state_semantic_digest_sha256",
    ):
        _require_sha256(value.get(field), f"resume_lineage.{field}")
    filename = value.get("resume_state_filename")
    match = _INTERVAL_STATE_NAME.fullmatch(filename) if isinstance(filename, str) else None
    if filename != "training_state.pt" and match is None:
        raise RiskDataContractError("resume_lineage resume_state_filename is invalid")
    optimizer_step = value.get("resume_optimizer_step")
    if type(optimizer_step) is not int or optimizer_step < 1:
        raise RiskDataContractError("resume_lineage resume_optimizer_step is invalid")
    if match is not None and int(match.group(1)) != optimizer_step:
        raise RiskDataContractError(
            "resume_lineage filename/optimizer step mismatch"
        )


def _finite_float_list(value: object, *, field: str) -> list[float]:
    if not isinstance(value, list):
        raise RiskDataContractError(f"training state {field} must be a list")
    result: list[float] = []
    for item in value:
        if type(item) not in {int, float} or not math.isfinite(float(item)):
            raise RiskDataContractError(
                f"training state {field} must contain finite numbers"
            )
        result.append(float(item))
    return result


def _validate_state_against_publication(
    state: Mapping[str, object],
    *,
    config: ProductionRiskTrainingConfig,
    config_digest: str,
    provenance: Mapping[str, object],
    label: str,
) -> None:
    if state.get("config") != asdict(config):
        raise RiskDataContractError(f"{label} config does not match published snapshot")
    if state.get("config_digest_sha256") != config_digest:
        raise RiskDataContractError(f"{label} config digest mismatch")
    if state.get("provenance") != provenance:
        raise RiskDataContractError(f"{label} provenance mismatch")
    expected_model_config = {
        "variant": config.variant,
        "hidden_channels": config.hidden_channels,
        "history_steps": 8,
    }
    if state.get("model_config") != expected_model_config:
        raise RiskDataContractError(f"{label} model configuration mismatch")
    optimizer_steps = state.get("optimizer_steps")
    if type(optimizer_steps) is not int or optimizer_steps < 1:
        raise RiskDataContractError(f"{label} optimizer_steps must be positive")
    completed_epochs = state.get("completed_epochs")
    next_epoch = state.get("next_epoch")
    epoch_sample_count = state.get("epoch_running_sample_count")
    if (
        type(completed_epochs) is not int
        or completed_epochs < 0
        or type(next_epoch) is not int
        or next_epoch < completed_epochs
        or type(epoch_sample_count) is not int
        or epoch_sample_count < 0
    ):
        raise RiskDataContractError(f"{label} epoch counters are invalid")
    step_history = _finite_float_list(
        state.get("optimizer_step_loss_history"),
        field="optimizer_step_loss_history",
    )
    if len(step_history) != optimizer_steps:
        raise RiskDataContractError(
            f"{label} optimizer history length does not match optimizer_steps"
        )
    _finite_float_list(state.get("loss_history"), field="loss_history")
    _finite_float_list(
        state.get("validation_loss_history"), field="validation_loss_history"
    )
    for field in ("epoch_running_loss_sum", "initial_train_loss"):
        value = state.get(field)
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise RiskDataContractError(f"{label} {field} must be finite")
    rng_state = state.get("rng_state")
    if not isinstance(rng_state, Mapping):
        raise RiskDataContractError(f"{label} RNG state must be a mapping")
    cuda_rng = rng_state.get("torch_cuda")
    if config.device == "cpu" and cuda_rng is not None:
        raise RiskDataContractError(f"{label} CPU state contains CUDA RNG data")
    if config.device.startswith("cuda") and not isinstance(cuda_rng, torch.Tensor):
        raise RiskDataContractError(f"{label} CUDA state lacks selected-device RNG data")


def _require_history_prefix(
    prefix_state: Mapping[str, object],
    final_state: Mapping[str, object],
    *,
    field: str,
    label: str,
) -> None:
    prefix = prefix_state.get(field)
    complete = final_state.get(field)
    if not isinstance(prefix, list) or not isinstance(complete, list):
        raise RiskDataContractError(f"{label} lineage field {field} is not a list")
    if len(prefix) > len(complete) or prefix != complete[: len(prefix)]:
        raise RiskDataContractError(f"{label} does not form a valid {field} lineage")


def _validate_published_resume_state(
    resume_path: Path,
    *,
    expected_publication_instance_digest_sha256: str,
) -> _ValidatedTrainingPublication:
    """Validate one immutable publication from single-read file snapshots."""

    path = _absolute(resume_path)
    root = path.parent
    if path.name != "training_state.pt" and _INTERVAL_STATE_NAME.fullmatch(
        path.name
    ) is None:
        raise RiskDataContractError(
            "resume_from must name a published final or interval training state"
        )
    expected_instance = _require_sha256(
        expected_publication_instance_digest_sha256,
        "expected publication_instance_digest_sha256",
    )
    snapshots = _snapshot_direct_regular_files(root)
    required_outer = {
        "training_manifest.json",
        ".producer-complete",
        "checksums.sha256",
    }
    if path.name not in snapshots or not required_outer.issubset(snapshots):
        raise RiskDataContractError(
            "resume_from must name a state inside a complete published training artifact"
        )
    _validate_checksum_snapshot(snapshots)
    manifest = _read_canonical_json_bytes(
        snapshots["training_manifest.json"], label="training manifest"
    )
    if set(manifest) != _TRAINING_MANIFEST_KEYS:
        raise RiskDataContractError("training manifest keys mismatch")
    if (
        manifest.get("training_layout_version")
        != PRODUCTION_RISK_TRAINING_LAYOUT_VERSION
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("mode") != "production"
    ):
        raise RiskDataContractError("training manifest layout/schema/mode mismatch")
    _validate_resume_lineage(manifest.get("resume_lineage"))
    scientific_digest = _require_sha256(
        manifest.get("semantic_digest_sha256"), "manifest semantic_digest_sha256"
    )
    if scientific_digest != _training_manifest_semantic_digest(manifest):
        raise RiskDataContractError("training manifest scientific semantic digest mismatch")
    instance_digest = _require_sha256(
        manifest.get("publication_instance_digest_sha256"),
        "manifest publication_instance_digest_sha256",
    )
    if instance_digest != _training_publication_instance_digest(manifest):
        raise RiskDataContractError("training publication instance digest mismatch")
    if instance_digest != expected_instance:
        raise RiskDataContractError(
            "training publication instance does not match the trusted expected digest"
        )
    marker = _read_canonical_json_bytes(
        snapshots[".producer-complete"], label="producer-complete marker"
    )
    if set(marker) != _PRODUCER_COMPLETE_KEYS or marker != {
        "training_layout_version": PRODUCTION_RISK_TRAINING_LAYOUT_VERSION,
        "semantic_digest_sha256": scientific_digest,
        "publication_instance_digest_sha256": instance_digest,
    }:
        raise RiskDataContractError(
            "producer-complete marker does not bind scientific and publication digests"
        )

    artifact_hashes = manifest.get("artifact_sha256")
    bindings = manifest.get("artifact_semantic_bindings")
    if not isinstance(artifact_hashes, dict) or not isinstance(bindings, dict):
        raise RiskDataContractError("training manifest artifact bindings must be mappings")
    actual_artifacts = set(snapshots) - required_outer
    if set(artifact_hashes) != actual_artifacts or set(bindings) != actual_artifacts:
        raise RiskDataContractError("training manifest artifact file set mismatch")
    if not _MANDATORY_RESULT_ARTIFACTS.issubset(actual_artifacts):
        raise RiskDataContractError("training publication lacks final result artifacts")
    for name in sorted(actual_artifacts):
        declared_hash = _require_sha256(
            artifact_hashes.get(name), f"artifact_sha256.{name}"
        )
        _require_sha256(bindings.get(name), f"artifact_semantic_bindings.{name}")
        if _sha256_bytes(snapshots[name]) != declared_hash:
            raise RiskDataContractError(f"artifact_sha256 mismatch: {name}")

    config_snapshot = _read_canonical_json_bytes(
        snapshots["config_snapshot.json"], label="config snapshot"
    )
    config_fields = {
        key: value
        for key, value in config_snapshot.items()
        if key not in {"artifact_layout_version", "mode", "optimizer"}
    }
    try:
        published_config = ProductionRiskTrainingConfig(**config_fields)
    except (TypeError, ValueError) as exc:
        raise RiskDataContractError("published training config is invalid") from exc
    if config_snapshot != _config_snapshot(published_config):
        raise RiskDataContractError("published config snapshot fields mismatch")
    published_config_digest = _config_digest(published_config)
    if manifest.get("config_digest_sha256") != published_config_digest:
        raise RiskDataContractError("training manifest config digest mismatch")
    if manifest.get("stage") != published_config.stage or manifest.get(
        "variant"
    ) != published_config.variant:
        raise RiskDataContractError("training manifest stage/variant mismatch")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RiskDataContractError("training manifest provenance must be a mapping")
    runtime_environment = _validate_runtime_environment_snapshot(
        manifest.get("runtime_environment"),
        configured_device=published_config.device,
    )
    runtime_digest = _require_sha256(
        manifest.get("runtime_environment_digest_sha256"),
        "manifest runtime_environment_digest_sha256",
    )
    if runtime_digest != _runtime_environment_digest(runtime_environment):
        raise RiskDataContractError("training manifest runtime environment digest mismatch")
    if provenance.get("runtime_environment_digest_sha256") != runtime_digest:
        raise RiskDataContractError("training provenance runtime environment digest mismatch")
    metrics = _read_canonical_json_bytes(
        snapshots["metrics.json"], label="training metrics"
    )
    if (
        metrics.get("training_layout_version")
        != PRODUCTION_RISK_TRAINING_LAYOUT_VERSION
        or metrics.get("stage") != published_config.stage
        or metrics.get("variant") != published_config.variant
        or metrics.get("seed") != published_config.seed
        or metrics.get("train_sample_count")
        != provenance.get("consumed_sample_count")
        or manifest.get("train_sample_count")
        != provenance.get("consumed_sample_count")
    ):
        raise RiskDataContractError("training metrics/provenance identity mismatch")
    if bindings["config_snapshot.json"] != artifact_hashes["config_snapshot.json"]:
        raise RiskDataContractError("config snapshot semantic binding mismatch")
    if bindings["metrics.json"] != artifact_hashes["metrics.json"]:
        raise RiskDataContractError("training metrics semantic binding mismatch")

    states: dict[str, dict[str, object]] = {}
    intervals: list[tuple[str, dict[str, object]]] = []
    final_checkpoint_payload: dict[str, object] | None = None
    for name in sorted(actual_artifacts):
        if name in {"config_snapshot.json", "metrics.json"}:
            continue
        if name in {"final_checkpoint.pt", "best_checkpoint.pt"}:
            _, checkpoint = load_risk_checkpoint(
                io.BytesIO(snapshots[name]),
                expected_mode="production",
                expected_provenance=provenance,
            )
            if bindings[name] != checkpoint["checkpoint_semantic_digest_sha256"]:
                raise RiskDataContractError(
                    f"checkpoint semantic binding mismatch: {name}"
                )
            if name == "final_checkpoint.pt":
                final_checkpoint_payload = checkpoint
            continue
        if name == "training_state.pt" or _INTERVAL_STATE_NAME.fullmatch(name):
            state_value = _load_training_state(snapshots[name])
            _validate_state_against_publication(
                state_value,
                config=published_config,
                config_digest=published_config_digest,
                provenance=provenance,
                label=name,
            )
            if bindings[name] != state_value["training_state_semantic_digest_sha256"]:
                raise RiskDataContractError(
                    f"training-state semantic binding mismatch: {name}"
                )
            states[name] = state_value
            match = _INTERVAL_STATE_NAME.fullmatch(name)
            if match is not None:
                if state_value["optimizer_steps"] != int(match.group(1)):
                    raise RiskDataContractError(
                        f"interval filename/optimizer step mismatch: {name}"
                    )
                intervals.append((name, state_value))
            continue
        raise RiskDataContractError(f"unexpected published training artifact: {name}")

    final_state = states.get("training_state.pt")
    if final_state is None or final_checkpoint_payload is None:
        raise RiskDataContractError("published final checkpoint/state is missing")
    if manifest.get("optimizer_steps") != final_state.get("optimizer_steps"):
        raise RiskDataContractError("manifest/final-state optimizer step mismatch")
    checkpoint_state = final_checkpoint_payload["model_state_dict"]
    training_model_state = final_state.get("model_state_dict")
    if not isinstance(training_model_state, Mapping) or set(checkpoint_state) != set(
        training_model_state
    ):
        raise RiskDataContractError("final checkpoint/training-state model keys mismatch")
    if not all(
        torch.equal(checkpoint_state[name], training_model_state[name])
        for name in checkpoint_state
    ):
        raise RiskDataContractError("final checkpoint/training-state model mismatch")
    for interval_name, interval_state in intervals:
        if int(interval_state["optimizer_steps"]) > int(final_state["optimizer_steps"]):
            raise RiskDataContractError("interval optimizer step exceeds final state")
        if interval_state["initial_train_loss"] != final_state["initial_train_loss"]:
            raise RiskDataContractError("interval initial-loss lineage mismatch")
        if int(interval_state["completed_epochs"]) > int(final_state["completed_epochs"]):
            raise RiskDataContractError("interval completed-epoch lineage mismatch")
        for field in (
            "optimizer_step_loss_history",
            "loss_history",
            "validation_loss_history",
        ):
            _require_history_prefix(
                interval_state,
                final_state,
                field=field,
                label=interval_name,
            )
    if path.name not in states:
        raise RiskDataContractError("resume state lacks an external manifest binding")
    return _ValidatedTrainingPublication(
        root=root,
        manifest=manifest,
        target_state=states[path.name],
        target_state_filename=path.name,
        target_state_sha256=str(artifact_hashes[path.name]),
        interval_states=tuple(
            sorted(intervals, key=lambda item: int(item[1]["optimizer_steps"]))
        ),
    )


def train_production_risk_model(
    *,
    train_dataset: LoadedRiskDataset,
    train_subset: ProductionRiskSubset,
    config: ProductionRiskTrainingConfig,
    output_dir: str | Path,
    code_commit: str,
    validation_dataset: LoadedRiskDataset | None = None,
    resume_from: str | Path | None = None,
    resume_expected_publication_instance_digest_sha256: str | None = None,
    cross_split_audit: Mapping[str, object] | None = None,
) -> ProductionRiskTrainingResult:
    """Train R0/R1 while preserving split gates, stream cursors, and provenance."""

    if not isinstance(config, ProductionRiskTrainingConfig):
        raise RiskDataContractError("config must be ProductionRiskTrainingConfig")
    checked_code_commit = _require_code_commit(code_commit)
    if (resume_from is None) != (
        resume_expected_publication_instance_digest_sha256 is None
    ):
        raise RiskDataContractError(
            "resume_from and resume expected publication instance digest must be supplied together"
        )
    expected_resume_instance = (
        None
        if resume_expected_publication_instance_digest_sha256 is None
        else _require_sha256(
            resume_expected_publication_instance_digest_sha256,
            "resume_expected_publication_instance_digest_sha256",
        )
    )
    destination = _absolute(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    validation_digest, family_digest, leakage_status = _validate_stage_gates(
        train_dataset=train_dataset,
        train_subset=train_subset,
        config=config,
        validation_dataset=validation_dataset,
        cross_split_audit=cross_split_audit,
    )
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RiskDataContractError(
            "CUDA production training requires an allocated CUDA device"
        )

    runtime_environment = _runtime_environment(device)
    runtime_environment_digest = _runtime_environment_digest(runtime_environment)
    selected_count = len(train_subset.sample_ids)
    data_scale, scientific_claim_eligible = _training_data_scale(
        config, selected_count
    )
    if config.stage == "one_shard_smoke":
        try:
            smoke_first_batch, _ = next(
                iter(
                    iter_production_risk_batches(
                        train_dataset,
                        subset=train_subset,
                        batch_size=config.batch_size,
                        seed=config.seed,
                        epoch=0,
                    )
                )
            )
        except StopIteration as exc:
            raise RiskDataContractError("one_shard_smoke stream is empty") from exc
        consumed_sample_ids = tuple(smoke_first_batch.sample_ids)
    else:
        consumed_sample_ids = tuple(train_subset.sample_ids)

    config_digest = _config_digest(config)
    provenance = _checkpoint_provenance(
        dataset=train_dataset,
        subset=train_subset,
        config=config,
        config_digest=config_digest,
        validation_digest=validation_digest,
        family_digest=family_digest,
        leakage_status=leakage_status,
        code_commit=checked_code_commit,
        runtime_environment_digest=runtime_environment_digest,
        training_data_scale=data_scale,
        scientific_claim_eligible=scientific_claim_eligible,
        consumed_sample_ids=consumed_sample_ids,
    )
    staging = destination.with_name(
        f".{destination.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    try:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        model = RiskModel(
            variant=config.variant,
            hidden_channels=config.hidden_channels,
        ).to(device=device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )

        optimizer_steps = 0
        completed_epochs = 0
        next_epoch = 0
        next_cursor: RiskStreamCursor | None = None
        loss_history: list[float] = []
        optimizer_step_loss_history: list[float] = []
        validation_loss_history: list[float] = []
        epoch_running_loss_sum = 0.0
        epoch_running_sample_count = 0
        initial_train_loss = float("nan")
        best_validation_loss: float | None = None
        best_validation_step: int | None = None
        best_model_state_dict: Mapping[str, torch.Tensor] | None = None
        resume_lineage: dict[str, object] | None = None

        if resume_from is not None:
            resume_path = _absolute(resume_from)
            assert expected_resume_instance is not None
            resume_publication = _validate_published_resume_state(
                resume_path,
                expected_publication_instance_digest_sha256=expected_resume_instance,
            )
            state = resume_publication.target_state
            resume_lineage = {
                "parent_scientific_digest_sha256": resume_publication.manifest[
                    "semantic_digest_sha256"
                ],
                "parent_publication_instance_digest_sha256": (
                    resume_publication.manifest[
                        "publication_instance_digest_sha256"
                    ]
                ),
                "resume_state_filename": resume_publication.target_state_filename,
                "resume_state_file_sha256": resume_publication.target_state_sha256,
                "resume_state_semantic_digest_sha256": state[
                    "training_state_semantic_digest_sha256"
                ],
                "resume_optimizer_step": state["optimizer_steps"],
            }
            prior_config = _validate_resume_config(state["config"], config)
            expected_prior_digest = _sha256_bytes(
                _canonical_json_bytes(
                    {
                        "artifact_layout_version": PRODUCTION_RISK_TRAINING_LAYOUT_VERSION,
                        "mode": "production",
                        "optimizer": "AdamW",
                        **asdict(prior_config),
                    }
                )
            )
            if state.get("config_digest_sha256") != expected_prior_digest:
                raise RiskDataContractError("resume config_digest_sha256 mismatch")
            saved_provenance = state.get("provenance")
            if not isinstance(saved_provenance, Mapping):
                raise RiskDataContractError("resume provenance must be a mapping")
            for field, expected in provenance.items():
                if field == "config_digest":
                    continue
                if saved_provenance.get(field) != expected:
                    raise RiskDataContractError(f"resume provenance mismatch for {field}")
            if saved_provenance.get("config_digest") != expected_prior_digest:
                raise RiskDataContractError("resume provenance config digest mismatch")
            if state.get("model_config") != model.export_config():
                raise RiskDataContractError("resume model configuration mismatch")
            try:
                model.load_state_dict(state["model_state_dict"], strict=True)
                optimizer.load_state_dict(state["optimizer_state_dict"])
            except (TypeError, ValueError, RuntimeError) as exc:
                raise RiskDataContractError("resume model/optimizer state is invalid") from exc
            _optimizer_to_device(optimizer, device)
            optimizer_steps = int(state["optimizer_steps"])
            completed_epochs = int(state["completed_epochs"])
            next_epoch = int(state["next_epoch"])
            next_cursor = _cursor_from_mapping(state["next_cursor"])
            loss_history = [float(value) for value in state["loss_history"]]
            optimizer_step_loss_history = [
                float(value) for value in state["optimizer_step_loss_history"]
            ]
            validation_loss_history = [
                float(value) for value in state["validation_loss_history"]
            ]
            epoch_running_loss_sum = float(state["epoch_running_loss_sum"])
            epoch_running_sample_count = int(state["epoch_running_sample_count"])
            initial_train_loss = float(state["initial_train_loss"])
            best_validation_loss = (
                None
                if state["best_validation_loss"] is None
                else float(state["best_validation_loss"])
            )
            best_validation_step = (
                None
                if state["best_validation_step"] is None
                else int(state["best_validation_step"])
            )
            best_model_state_dict = state["best_model_state_dict"]
            _restore_rng_state(state["rng_state"], device=device)

        elif config.stage != "one_shard_smoke":
            initial_train_loss, _ = _evaluate_loss(
                model,
                dataset=train_dataset,
                subset=train_subset,
                config=config,
                device=device,
                epoch=0,
            )
            loss_history.append(initial_train_loss)

        validation_subset: ProductionRiskSubset | None = None
        if validation_dataset is not None:
            validation_subset = select_production_risk_subset(
                validation_dataset,
                max_samples=validation_dataset.sample_count,
                seed=config.seed,
            )

        all_finite = True
        training_finished = False
        smoke_final_loss: float | None = None
        observed_consumed_sample_ids: set[str] = set()
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(next_epoch, config.epochs):
            cursor_for_epoch = next_cursor if epoch == next_epoch else None
            accumulated_weighted_loss_sum = 0.0
            accumulated_sample_count = 0
            accumulated_microbatch_count = 0
            saw_batch = False
            for raw_batch, stream_cursor in iter_production_risk_batches(
                train_dataset,
                subset=train_subset,
                batch_size=config.batch_size,
                seed=config.seed,
                epoch=epoch,
                start_cursor=cursor_for_epoch,
            ):
                saw_batch = True
                observed_consumed_sample_ids.update(raw_batch.sample_ids)
                batch = _move_batch(raw_batch, device)
                model.train()
                output, losses = compute_risk_batch_loss(
                    model, batch, lambda_collision=config.lambda_collision
                )
                tensors = tuple(output.values()) + tuple(losses.values())
                if not all(torch.isfinite(value).all().item() for value in tensors):
                    all_finite = False
                    raise RiskDataContractError(
                        "production forward/loss contains NaN/Inf"
                    )
                crossing = output["quantiles"][:, 1:] < output["quantiles"][:, :-1]
                if torch.any(crossing).item():
                    raise RiskDataContractError("production quantile outputs cross")
                raw_loss = float(losses["total"].detach().cpu().item())
                if config.stage == "one_shard_smoke" and not loss_history:
                    initial_train_loss = raw_loss
                    loss_history.append(raw_loss)
                batch_sample_count = len(batch.sample_ids)
                (losses["total"] * batch_sample_count).backward()
                accumulated_weighted_loss_sum += raw_loss * batch_sample_count
                accumulated_sample_count += batch_sample_count
                accumulated_microbatch_count += 1
                epoch_running_loss_sum += raw_loss * batch_sample_count
                epoch_running_sample_count += batch_sample_count
                terminal_cursor = stream_cursor.shard_index == -1
                must_step = (
                    accumulated_microbatch_count
                    == config.gradient_accumulation_steps
                    or terminal_cursor
                    or config.stage == "one_shard_smoke"
                )
                if not must_step:
                    continue
                if accumulated_sample_count < 1:
                    raise RiskDataContractError("empty gradient accumulation window")
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(float(accumulated_sample_count))
                        if not torch.isfinite(parameter.grad).all().item():
                            all_finite = False
                            raise RiskDataContractError(
                                "production gradients contain NaN/Inf"
                            )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                optimizer_step_loss_history.append(
                    accumulated_weighted_loss_sum / accumulated_sample_count
                )
                accumulated_weighted_loss_sum = 0.0
                accumulated_sample_count = 0
                accumulated_microbatch_count = 0

                if terminal_cursor:
                    if epoch_running_sample_count < 1:
                        raise RiskDataContractError("empty production training epoch")
                    loss_history.append(
                        epoch_running_loss_sum / epoch_running_sample_count
                    )
                    completed_epochs = epoch + 1
                    next_epoch = epoch + 1
                    next_cursor = None
                    epoch_running_loss_sum = 0.0
                    epoch_running_sample_count = 0
                    if validation_dataset is not None and validation_subset is not None:
                        validation_loss, _ = _evaluate_loss(
                            model,
                            dataset=validation_dataset,
                            subset=validation_subset,
                            config=config,
                            device=device,
                            epoch=epoch,
                        )
                        validation_loss_history.append(validation_loss)
                        if (
                            best_validation_loss is None
                            or validation_loss < best_validation_loss
                        ):
                            best_validation_loss = validation_loss
                            best_validation_step = optimizer_steps
                            best_model_state_dict = copy.deepcopy(model.state_dict())
                else:
                    next_epoch = epoch
                    next_cursor = stream_cursor

                state_payload = _make_training_state(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    config_digest=config_digest,
                    provenance=provenance,
                    optimizer_steps=optimizer_steps,
                    completed_epochs=completed_epochs,
                    next_epoch=next_epoch,
                    next_cursor=next_cursor,
                    loss_history=loss_history,
                    optimizer_step_loss_history=optimizer_step_loss_history,
                    validation_loss_history=validation_loss_history,
                    epoch_running_loss_sum=epoch_running_loss_sum,
                    epoch_running_sample_count=epoch_running_sample_count,
                    initial_train_loss=initial_train_loss,
                    best_validation_loss=best_validation_loss,
                    best_validation_step=best_validation_step,
                    best_model_state_dict=best_model_state_dict,
                    device=device,
                )
                if optimizer_steps % config.checkpoint_interval_steps == 0:
                    _write_training_state(
                        staging
                        / f"training_state_step_{optimizer_steps:08d}.pt",
                        state_payload,
                    )
                if config.stage == "one_shard_smoke":
                    model.eval()
                    with torch.no_grad():
                        _, final_smoke_losses = compute_risk_batch_loss(
                            model,
                            batch,
                            lambda_collision=config.lambda_collision,
                        )
                    smoke_final_loss = float(
                        final_smoke_losses["total"].detach().cpu().item()
                    )
                    if not math.isfinite(smoke_final_loss):
                        raise RiskDataContractError(
                            "production smoke final loss contains NaN/Inf"
                        )
                    loss_history.append(smoke_final_loss)
                    training_finished = True
                    break
            if config.stage != "one_shard_smoke" and not saw_batch:
                raise RiskDataContractError("production stream yielded no training batches")
            if training_finished:
                break
            next_cursor = None

        if optimizer_steps < 1:
            raise RiskDataContractError("production training performed no optimizer steps")
        expected_consumed_membership = set(consumed_sample_ids)
        if resume_from is None and observed_consumed_sample_ids != (
            expected_consumed_membership
        ):
            raise RiskDataContractError(
                "actual training consumption does not match the bound sample membership"
            )
        if config.stage == "one_shard_smoke" and observed_consumed_sample_ids != (
            expected_consumed_membership
        ):
            raise RiskDataContractError(
                "one_shard_smoke must consume exactly its first streamed batch"
            )
        if config.stage == "one_shard_smoke":
            if optimizer_steps != 1 or smoke_final_loss is None:
                raise RiskDataContractError(
                    "one_shard_smoke must perform exactly one optimizer step"
                )
            final_train_loss = smoke_final_loss
            final_crossing_rate = 0.0
        else:
            final_train_loss, final_crossing_rate = _evaluate_loss(
                model,
                dataset=train_dataset,
                subset=train_subset,
                config=config,
                device=device,
                epoch=max(0, config.epochs - 1),
            )
        if not all(
            math.isfinite(value)
            for value in (
                initial_train_loss,
                final_train_loss,
                *loss_history,
                *optimizer_step_loss_history,
                *validation_loss_history,
            )
        ):
            raise RiskDataContractError("production loss history contains NaN/Inf")

        final_state = _make_training_state(
            model=model,
            optimizer=optimizer,
            config=config,
            config_digest=config_digest,
            provenance=provenance,
            optimizer_steps=optimizer_steps,
            completed_epochs=completed_epochs,
            next_epoch=next_epoch,
            next_cursor=next_cursor,
            loss_history=loss_history,
            optimizer_step_loss_history=optimizer_step_loss_history,
            validation_loss_history=validation_loss_history,
            epoch_running_loss_sum=epoch_running_loss_sum,
            epoch_running_sample_count=epoch_running_sample_count,
            initial_train_loss=initial_train_loss,
            best_validation_loss=best_validation_loss,
            best_validation_step=best_validation_step,
            best_model_state_dict=best_model_state_dict,
            device=device,
        )
        _write_training_state(staging / "training_state.pt", final_state)
        save_risk_checkpoint(
            staging / "final_checkpoint.pt",
            model=model,
            mode="production",
            provenance=provenance,
        )
        if config.stage == "formal_50k":
            if best_model_state_dict is None:
                raise RiskDataContractError(
                    "formal_50k completed without a best validation state"
                )
            final_model_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_model_state_dict, strict=True)
            save_risk_checkpoint(
                staging / "best_checkpoint.pt",
                model=model,
                mode="production",
                provenance=provenance,
            )
            model.load_state_dict(final_model_state, strict=True)

        metrics = {
            "training_layout_version": PRODUCTION_RISK_TRAINING_LAYOUT_VERSION,
            "mode": "production",
            "stage": config.stage,
            "variant": config.variant,
            "seed": config.seed,
            "device": config.device,
            "train_split": "train",
            "validation_split": None if validation_dataset is None else "val",
            "selected_sample_count": len(train_subset.sample_ids),
            "train_sample_count": len(consumed_sample_ids),
            "unique_consumed_sample_count": len(consumed_sample_ids),
            "consumed_sample_ids_digest_sha256": provenance[
                "consumed_sample_ids_digest_sha256"
            ],
            "training_data_scale": data_scale,
            "scientific_claim_eligible": scientific_claim_eligible,
            "validation_sample_count": (
                0 if validation_dataset is None else validation_dataset.sample_count
            ),
            "optimizer_steps": optimizer_steps,
            "completed_epochs": completed_epochs,
            "initial_train_loss": initial_train_loss,
            "final_train_loss": final_train_loss,
            "loss_history": loss_history,
            "optimizer_step_loss_history": optimizer_step_loss_history,
            "validation_loss_history": validation_loss_history,
            "best_validation_loss": best_validation_loss,
            "best_validation_step": best_validation_step,
            "quantile_crossing_rate": final_crossing_rate,
            "all_forward_loss_gradient_values_finite": all_finite,
            "test_samples_used_for_training_or_selection": 0,
        }
        _write_json(staging / "metrics.json", metrics)
        _write_json(staging / "config_snapshot.json", _config_snapshot(config))

        artifact_hashes = {
            path.name: _sha256_file(path)
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
            if path.is_file()
        }
        semantic_bindings: dict[str, str] = {
            "config_snapshot.json": artifact_hashes["config_snapshot.json"],
            "metrics.json": artifact_hashes["metrics.json"],
        }
        for state_path in sorted(staging.glob("training_state*.pt")):
            loaded_state = _load_training_state(state_path)
            semantic_bindings[state_path.name] = str(
                loaded_state["training_state_semantic_digest_sha256"]
            )
        for checkpoint_name in ("final_checkpoint.pt", "best_checkpoint.pt"):
            checkpoint_path = staging / checkpoint_name
            if not checkpoint_path.exists():
                continue
            _, checkpoint_payload = load_risk_checkpoint(
                checkpoint_path,
                expected_mode="production",
                expected_provenance=provenance,
            )
            semantic_bindings[checkpoint_name] = str(
                checkpoint_payload["checkpoint_semantic_digest_sha256"]
            )
        manifest: dict[str, object] = {
            "training_layout_version": PRODUCTION_RISK_TRAINING_LAYOUT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "mode": "production",
            "stage": config.stage,
            "variant": config.variant,
            "config_digest_sha256": config_digest,
            "provenance": provenance,
            "train_sample_count": len(consumed_sample_ids),
            "optimizer_steps": optimizer_steps,
            "artifact_sha256": artifact_hashes,
            "artifact_semantic_bindings": semantic_bindings,
            "runtime_environment": runtime_environment,
            "runtime_environment_digest_sha256": runtime_environment_digest,
            "resume_lineage": resume_lineage,
        }
        manifest["semantic_digest_sha256"] = _training_manifest_semantic_digest(
            manifest
        )
        semantic_digest = str(manifest["semantic_digest_sha256"])
        manifest["publication_instance_digest_sha256"] = (
            _training_publication_instance_digest(manifest)
        )
        publication_instance_digest = str(
            manifest["publication_instance_digest_sha256"]
        )
        _write_json(staging / "training_manifest.json", manifest)
        _write_json(
            staging / ".producer-complete",
            {
                "training_layout_version": PRODUCTION_RISK_TRAINING_LAYOUT_VERSION,
                "semantic_digest_sha256": semantic_digest,
                "publication_instance_digest_sha256": publication_instance_digest,
            },
        )
        _write_checksum_manifest(staging)
        _validate_published_resume_state(
            staging / "training_state.pt",
            expected_publication_instance_digest_sha256=(
                publication_instance_digest
            ),
        )
        _atomic_rename_directory_noreplace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    absolute_output = _absolute(destination)
    _validate_published_resume_state(
        absolute_output / "training_state.pt",
        expected_publication_instance_digest_sha256=publication_instance_digest,
    )
    return ProductionRiskTrainingResult(
        output_dir=absolute_output,
        best_checkpoint=(
            absolute_output / "best_checkpoint.pt"
            if config.stage == "formal_50k"
            else None
        ),
        final_checkpoint=absolute_output / "final_checkpoint.pt",
        training_state_checkpoint=absolute_output / "training_state.pt",
        metrics_path=absolute_output / "metrics.json",
        manifest_path=absolute_output / "training_manifest.json",
        semantic_digest_sha256=semantic_digest,
    )


__all__ = [
    "PRODUCTION_RISK_TRAINING_LAYOUT_VERSION",
    "ProductionRiskTrainingConfig",
    "ProductionRiskTrainingResult",
    "train_production_risk_model",
]
