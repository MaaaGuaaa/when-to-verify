"""Authenticated single-node DDP training for SOP09 risk models."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import asdict
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
from typing import Mapping, Sequence
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.contracts import SCHEMA_VERSION
from src.datasets.risk_dataloader import RiskDataContractError
from src.datasets.risk_training_store import AuthenticatedRiskTrainingView
from src.datasets.toy_risk_learning import frozen_channel_spec
from src.models.risk_model import (
    RiskModel,
    compute_risk_batch_loss,
    load_risk_checkpoint,
    save_risk_checkpoint,
)
from src.training.distributed import (
    DistributedRuntime,
    all_reduce_sample_count,
    broadcast_rank_zero_setup,
    scale_distributed_batch_mean_loss,
)
from src.training.risk_trainer import (
    PRODUCTION_RISK_TRAINING_LAYOUT_VERSION,
    ProductionRiskTrainingConfig,
    ProductionRiskTrainingResult,
    _absolute,
    _atomic_rename_directory_noreplace,
    _capture_rng_state,
    _canonical_json_bytes,
    _config_digest,
    _config_snapshot,
    _move_batch,
    _optimizer_to_device,
    _require_code_commit,
    _require_sha256,
    _restore_rng_state,
    _sample_id_membership_digest,
    _sha256_bytes,
    _sha256_file,
    _snapshot_direct_regular_files,
    _training_data_scale,
    _tree_digest,
    _validate_checksum_snapshot,
    _validate_resume_config,
    _write_checksum_manifest,
)


DISTRIBUTED_RISK_TRAINING_LAYOUT_VERSION = "sop09_distributed_training_v1"
DISTRIBUTED_RISK_TRAINING_STATE_LAYOUT_VERSION = (
    "sop09_distributed_training_state_v1"
)
_DISTRIBUTED_IDENTITY_LAYOUT_VERSION = "sop09_distributed_identity_v1"
_RUNTIME_LAYOUT_VERSION = "sop09_distributed_runtime_environment_v1"
_STATE_KEYS = frozenset(
    {
        "training_state_layout_version",
        "model_config",
        "model_state_dict",
        "optimizer_state_dict",
        "optimizer_state_digest_sha256",
        "config",
        "config_digest_sha256",
        "checkpoint_provenance",
        "distributed_identity",
        "optimizer_steps",
        "completed_epochs",
        "next_epoch",
        "next_microbatch_index",
        "loss_history",
        "optimizer_step_loss_history",
        "validation_loss_history",
        "epoch_running_loss_sum",
        "epoch_running_sample_count",
        "initial_train_loss",
        "best_validation_loss",
        "best_validation_step",
        "best_model_state_dict",
        "epoch_records",
        "per_rank_state",
        "training_state_semantic_digest_sha256",
    }
)
_DISTRIBUTED_IDENTITY_KEYS = frozenset(
    {
        "layout_version",
        "world_size",
        "backend",
        "train_snapshot_digest_sha256",
        "train_view_digest_sha256",
        "validation_snapshot_digest_sha256",
        "validation_view_digest_sha256",
        "partition_spec_digest_sha256",
        "current_epoch_plan_digest_sha256",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "training_layout_version",
        "schema_version",
        "mode",
        "stage",
        "variant",
        "config_digest_sha256",
        "checkpoint_provenance",
        "distributed_identity",
        "train_sample_count",
        "optimizer_steps",
        "artifact_sha256",
        "artifact_semantic_bindings",
        "runtime_environment",
        "resume_lineage",
        "writer_rank",
        "semantic_digest_sha256",
        "publication_instance_digest_sha256",
    }
)
_COMPLETE_KEYS = frozenset(
    {
        "training_layout_version",
        "semantic_digest_sha256",
        "publication_instance_digest_sha256",
    }
)
_INTERVAL_STATE = re.compile(r"^training_state_step_[0-9]{8}\.pt$")


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json_bytes(value))


def _strict_json(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RiskDataContractError(f"{label} must be valid JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise RiskDataContractError(f"{label} must be canonical JSON mapping")
    return value


def _state_semantic_digest(payload: Mapping[str, object]) -> str:
    projection = {
        "training_state_layout_version": payload.get(
            "training_state_layout_version"
        ),
        "model_config": payload.get("model_config"),
        "model_state_digest_sha256": _tree_digest(
            payload.get("model_state_dict")
        ),
        "optimizer_state_digest_sha256": _tree_digest(
            payload.get("optimizer_state_dict")
        ),
        "config": payload.get("config"),
        "config_digest_sha256": payload.get("config_digest_sha256"),
        "checkpoint_provenance": payload.get("checkpoint_provenance"),
        "distributed_identity": payload.get("distributed_identity"),
        "optimizer_steps": payload.get("optimizer_steps"),
        "completed_epochs": payload.get("completed_epochs"),
        "next_epoch": payload.get("next_epoch"),
        "next_microbatch_index": payload.get("next_microbatch_index"),
        "loss_history": payload.get("loss_history"),
        "optimizer_step_loss_history": payload.get(
            "optimizer_step_loss_history"
        ),
        "validation_loss_history": payload.get("validation_loss_history"),
        "epoch_running_loss_sum": payload.get("epoch_running_loss_sum"),
        "epoch_running_sample_count": payload.get(
            "epoch_running_sample_count"
        ),
        "initial_train_loss": payload.get("initial_train_loss"),
        "best_validation_loss": payload.get("best_validation_loss"),
        "best_validation_step": payload.get("best_validation_step"),
        "best_model_state_digest_sha256": (
            None
            if payload.get("best_model_state_dict") is None
            else _tree_digest(payload.get("best_model_state_dict"))
        ),
        "epoch_records": payload.get("epoch_records"),
        "per_rank_state_digest_sha256": _tree_digest(
            payload.get("per_rank_state")
        ),
    }
    return _sha256_bytes(_canonical_json_bytes(projection))


def _validate_state_payload(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _STATE_KEYS:
        raise RiskDataContractError("distributed training-state keys mismatch")
    if value.get("training_state_layout_version") != (
        DISTRIBUTED_RISK_TRAINING_STATE_LAYOUT_VERSION
    ):
        raise RiskDataContractError("distributed training-state layout mismatch")
    optimizer_digest = _require_sha256(
        value.get("optimizer_state_digest_sha256"),
        "optimizer_state_digest_sha256",
    )
    if optimizer_digest != _tree_digest(value.get("optimizer_state_dict")):
        raise RiskDataContractError("optimizer_state_digest_sha256 mismatch")
    declared = _require_sha256(
        value.get("training_state_semantic_digest_sha256"),
        "training_state_semantic_digest_sha256",
    )
    if declared != _state_semantic_digest(value):
        raise RiskDataContractError(
            "distributed training_state_semantic_digest_sha256 mismatch"
        )
    identity = value.get("distributed_identity")
    if not isinstance(identity, Mapping) or set(identity) != _DISTRIBUTED_IDENTITY_KEYS:
        raise RiskDataContractError("distributed training-state identity mismatch")
    per_rank = value.get("per_rank_state")
    world_size = identity.get("world_size")
    if (
        type(world_size) is not int
        or world_size < 2
        or not isinstance(per_rank, list)
        or len(per_rank) != world_size
    ):
        raise RiskDataContractError("distributed per-rank state count mismatch")
    expected_ranks = list(range(world_size))
    if [entry.get("rank") for entry in per_rank if isinstance(entry, Mapping)] != (
        expected_ranks
    ):
        raise RiskDataContractError("distributed per-rank state ordering mismatch")
    return value


def _load_state_source(source: Path | bytes) -> dict[str, object]:
    stream: object = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        value = torch.load(stream, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise RiskDataContractError(
            f"unable to load distributed training state: {exc}"
        ) from exc
    return _validate_state_payload(value)


def load_distributed_risk_training_state(path: str | Path) -> dict[str, object]:
    """Load and authenticate one standalone distributed training-state file."""

    return _load_state_source(_absolute(path))


def _write_training_state(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite training state: {path}")
    frozen = copy.deepcopy(dict(payload))
    frozen["optimizer_state_digest_sha256"] = _tree_digest(
        frozen["optimizer_state_dict"]
    )
    frozen["training_state_semantic_digest_sha256"] = _state_semantic_digest(
        frozen
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(frozen, temporary)
    temporary.replace(path)


def _runtime_environment(
    config: ProductionRiskTrainingConfig,
    runtime: DistributedRuntime,
) -> dict[str, object]:
    return {
        "runtime_environment_layout_version": _RUNTIME_LAYOUT_VERSION,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "configured_device": config.device,
        "world_size": runtime.world_size,
        "backend": runtime.backend,
    }


def _distributed_identity(
    *,
    train_view: AuthenticatedRiskTrainingView,
    validation_view: AuthenticatedRiskTrainingView | None,
    runtime: DistributedRuntime,
    current_epoch: int,
) -> dict[str, object]:
    plan = train_view.partition(epoch=current_epoch)
    return {
        "layout_version": _DISTRIBUTED_IDENTITY_LAYOUT_VERSION,
        "world_size": runtime.world_size,
        "backend": runtime.backend,
        "train_snapshot_digest_sha256": (
            train_view.snapshot.snapshot_digest_sha256
        ),
        "train_view_digest_sha256": train_view.training_view_digest_sha256,
        "validation_snapshot_digest_sha256": (
            None
            if validation_view is None
            else validation_view.snapshot.snapshot_digest_sha256
        ),
        "validation_view_digest_sha256": (
            None
            if validation_view is None
            else validation_view.training_view_digest_sha256
        ),
        "partition_spec_digest_sha256": plan.partition_spec_digest_sha256,
        "current_epoch_plan_digest_sha256": plan.epoch_plan_digest_sha256,
    }


def validate_distributed_resume_bindings(
    state: Mapping[str, object],
    *,
    expected_world_size: int,
    expected_backend: str,
    expected_train_snapshot_digest_sha256: str,
    expected_train_view_digest_sha256: str,
    expected_partition_spec_digest_sha256: str,
    expected_epoch_plan_digest_sha256: str,
) -> None:
    """Reject resume state whose topology or current data plan changed."""

    identity = state.get("distributed_identity")
    if not isinstance(identity, Mapping) or set(identity) != _DISTRIBUTED_IDENTITY_KEYS:
        raise RiskDataContractError("resume distributed identity fields mismatch")
    expected = {
        "world_size": expected_world_size,
        "backend": expected_backend,
        "train_snapshot_digest_sha256": _require_sha256(
            expected_train_snapshot_digest_sha256,
            "expected_train_snapshot_digest_sha256",
        ),
        "train_view_digest_sha256": _require_sha256(
            expected_train_view_digest_sha256,
            "expected_train_view_digest_sha256",
        ),
        "partition_spec_digest_sha256": _require_sha256(
            expected_partition_spec_digest_sha256,
            "expected_partition_spec_digest_sha256",
        ),
        "current_epoch_plan_digest_sha256": _require_sha256(
            expected_epoch_plan_digest_sha256,
            "expected_epoch_plan_digest_sha256",
        ),
    }
    for field, value in expected.items():
        if identity.get(field) != value:
            raise RiskDataContractError(
                f"distributed resume {field} mismatch"
            )


def _validate_training_inputs(
    *,
    train_view: AuthenticatedRiskTrainingView,
    validation_view: AuthenticatedRiskTrainingView | None,
    config: ProductionRiskTrainingConfig,
    runtime: DistributedRuntime,
    dataset_family_digest_sha256: str | None,
    global_cross_split_leakage: str,
) -> tuple[str | None, str | None, str]:
    if not isinstance(train_view, AuthenticatedRiskTrainingView):
        raise RiskDataContractError(
            "distributed training requires an authenticated training view"
        )
    if not isinstance(config, ProductionRiskTrainingConfig):
        raise RiskDataContractError("config must be ProductionRiskTrainingConfig")
    if not isinstance(runtime, DistributedRuntime) or not runtime.is_distributed:
        raise RiskDataContractError("DDP trainer requires WORLD_SIZE greater than one")
    if not dist.is_available() or not dist.is_initialized():
        raise RiskDataContractError("DDP trainer requires an initialized process group")
    if runtime.backend not in {"gloo", "nccl"}:
        raise RiskDataContractError("DDP trainer backend must be gloo or nccl")
    if train_view.split_role != "train" or train_view.snapshot.split != "train":
        raise RiskDataContractError("DDP train view must bind the train split")
    if (
        train_view.world_size != runtime.world_size
        or train_view.batch_size != config.batch_size
        or train_view.gradient_accumulation_steps
        != config.gradient_accumulation_steps
        or train_view.subset.seed != config.seed
    ):
        raise RiskDataContractError("DDP train view/config partition mismatch")
    if runtime.backend == "gloo" and config.device != "cpu":
        raise RiskDataContractError("Gloo DDP training requires device=cpu")
    if runtime.backend == "nccl" and config.device != "cuda":
        raise RiskDataContractError("NCCL DDP training requires device=cuda")
    source = train_view.snapshot.source_identity
    if source.get("schema_version") != SCHEMA_VERSION:
        raise RiskDataContractError("DDP train snapshot schema mismatch")
    source_channels = source.get("channel_spec")
    model_channels = frozen_channel_spec()
    if not isinstance(source_channels, Mapping) or {
        key: source_channels.get(key) for key in model_channels
    } != model_channels:
        raise RiskDataContractError("DDP train snapshot channel contract mismatch")
    selected_count = len(train_view.subset.sample_ids)
    if config.stage == "real_1k_overfit":
        expected = min(1000, len(train_view.snapshot.sample_ids))
        if selected_count != expected:
            raise RiskDataContractError(
                "real_1k_overfit requires exactly the frozen 1k/fixture subset"
            )
    if config.stage != "formal_50k":
        if validation_view is not None or dataset_family_digest_sha256 is not None:
            raise RiskDataContractError(
                "smoke/overfit DDP training rejects validation/family claims"
            )
        if global_cross_split_leakage != "NOT_PROVEN":
            raise RiskDataContractError(
                "smoke/overfit DDP leakage status must be NOT_PROVEN"
            )
        return None, None, "NOT_PROVEN"
    if validation_view is None:
        raise RiskDataContractError("formal_50k DDP requires a validation view")
    if (
        validation_view.split_role != "validation"
        or validation_view.snapshot.split != "val"
        or validation_view.world_size != runtime.world_size
        or validation_view.batch_size != config.batch_size
        or validation_view.gradient_accumulation_steps
        != config.gradient_accumulation_steps
    ):
        raise RiskDataContractError("formal_50k validation view is incompatible")
    family_digest = _require_sha256(
        dataset_family_digest_sha256,
        "dataset_family_digest_sha256",
    )
    if global_cross_split_leakage != "PROVEN":
        raise RiskDataContractError(
            "formal_50k DDP requires typed-family leakage status PROVEN"
        )
    validation_source = validation_view.snapshot.source_identity
    for field in (
        "g1_split_manifest_digest",
        "dynamic_objects_config_digest",
        "target_type_policy_digest",
    ):
        if source.get(field) != validation_source.get(field):
            raise RiskDataContractError(
                f"DDP train/validation common contract mismatch: {field}"
            )
    return (
        str(validation_source["risk_dataset_manifest_digest"]),
        family_digest,
        "PROVEN",
    )


def _checkpoint_provenance(
    *,
    train_view: AuthenticatedRiskTrainingView,
    config: ProductionRiskTrainingConfig,
    config_digest: str,
    validation_digest: str | None,
    family_digest: str | None,
    leakage_status: str,
    code_commit: str,
    runtime_environment_digest: str,
    consumed_sample_ids: Sequence[str],
) -> dict[str, object]:
    selected_count = len(train_view.subset.sample_ids)
    data_scale, eligible = _training_data_scale(config, selected_count)
    source = train_view.snapshot.source_identity
    consumed = tuple(consumed_sample_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "channel_spec": frozen_channel_spec(),
        "model_variant": config.variant,
        "config_digest": config_digest,
        "g1_split_manifest_digest": source["g1_split_manifest_digest"],
        "risk_dataset_manifest_digest": source[
            "risk_dataset_manifest_digest"
        ],
        "dynamic_objects_config_digest": source[
            "dynamic_objects_config_digest"
        ],
        "target_type_policy_digest": source["target_type_policy_digest"],
        "training_stage": config.stage,
        "training_subset_digest_sha256": (
            train_view.subset.sample_ids_digest_sha256
        ),
        "validation_risk_dataset_manifest_digest": validation_digest,
        "risk_dataset_family_digest": family_digest,
        "global_cross_split_leakage": leakage_status,
        "seed": config.seed,
        "code_commit": code_commit,
        "runtime_environment_digest_sha256": runtime_environment_digest,
        "training_data_scale": data_scale,
        "scientific_claim_eligible": eligible,
        "selected_sample_count": selected_count,
        "consumed_sample_count": len(consumed),
        "consumed_sample_ids_digest_sha256": _sample_id_membership_digest(
            consumed
        ),
    }


def _all_reduce_values(
    values: Sequence[float], runtime: DistributedRuntime
) -> tuple[float, ...]:
    tensor = torch.tensor(
        list(values),
        dtype=torch.float64,
        device=torch.device(runtime.device),
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tuple(float(value) for value in tensor.cpu().tolist())


def _all_ranks_true(value: bool, runtime: DistributedRuntime) -> bool:
    tensor = torch.tensor(
        1 if value else 0,
        dtype=torch.int32,
        device=torch.device(runtime.device),
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(tensor.item())


def _all_gather_objects(value: object, runtime: DistributedRuntime) -> list[object]:
    gathered: list[object] = [None] * runtime.world_size
    dist.all_gather_object(gathered, value)
    return gathered


def _evaluate_batches(
    model: DistributedDataParallel,
    *,
    view: AuthenticatedRiskTrainingView,
    runtime: DistributedRuntime,
    config: ProductionRiskTrainingConfig,
    epoch: int,
    first_microbatch_only: bool = False,
) -> tuple[float, float]:
    plan = view.partition(epoch=epoch)
    batches = plan.rank_microbatches[runtime.rank]
    if first_microbatch_only:
        batches = batches[:1]
    model.eval()
    local_loss_sum = 0.0
    local_sample_count = 0
    local_crossings = 0
    local_comparisons = 0
    with torch.no_grad():
        for sample_ids in batches:
            batch = _move_batch(view.snapshot.batch(sample_ids), torch.device(runtime.device))
            output, losses = compute_risk_batch_loss(
                model,
                batch,
                lambda_collision=config.lambda_collision,
            )
            finite = all(
                bool(torch.isfinite(value).all().item())
                for value in (*output.values(), *losses.values())
            )
            if not finite:
                raise RiskDataContractError("distributed evaluation contains NaN/Inf")
            count = len(sample_ids)
            local_loss_sum += float(losses["total"].item()) * count
            local_sample_count += count
            crossing = output["quantiles"][:, 1:] < output["quantiles"][:, :-1]
            local_crossings += int(torch.count_nonzero(crossing).item())
            local_comparisons += crossing.numel()
    loss_sum, sample_count, crossings, comparisons = _all_reduce_values(
        (
            local_loss_sum,
            local_sample_count,
            local_crossings,
            local_comparisons,
        ),
        runtime,
    )
    expected_count = (
        sum(len(rank_batches[0]) for rank_batches in plan.rank_microbatches)
        if first_microbatch_only
        else len(view.subset.sample_ids)
    )
    if int(sample_count) != expected_count or expected_count < 1:
        raise RiskDataContractError("distributed evaluation sample count mismatch")
    return loss_sum / sample_count, crossings / max(1.0, comparisons)


def _rank_membership_record(
    plan,
    *,
    epoch: int,
) -> dict[str, object]:
    rank_ids = [
        tuple(sample_id for batch in rank_batches for sample_id in batch)
        for rank_batches in plan.rank_microbatches
    ]
    return {
        "epoch": epoch,
        "epoch_plan_digest_sha256": plan.epoch_plan_digest_sha256,
        "rank_membership_digest_sha256": [
            _sample_id_membership_digest(ids) for ids in rank_ids
        ],
        "rank_sample_counts": [len(ids) for ids in rank_ids],
        "rank_local_microbatch_sizes": [
            list(sizes) for sizes in plan.local_microbatch_sizes
        ],
    }


def _validate_actual_rank_membership(
    gathered: Sequence[object],
    *,
    expected_ids: set[str],
) -> list[tuple[str, ...]]:
    normalized: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for rank, value in enumerate(gathered):
        if not isinstance(value, (list, tuple)):
            raise RiskDataContractError(
                f"rank {rank} consumed membership is invalid"
            )
        ids = tuple(value)
        if (
            not ids
            or any(not isinstance(sample_id, str) or not sample_id for sample_id in ids)
            or len(ids) != len(set(ids))
            or seen.intersection(ids)
        ):
            raise RiskDataContractError(
                "distributed rank consumption overlaps or is invalid"
            )
        seen.update(ids)
        normalized.append(ids)
    if seen != expected_ids:
        raise RiskDataContractError(
            "distributed rank consumption has missing or unexpected samples"
        )
    return normalized


def _make_state(
    *,
    model: RiskModel,
    optimizer: torch.optim.Optimizer,
    config: ProductionRiskTrainingConfig,
    config_digest: str,
    checkpoint_provenance: Mapping[str, object],
    distributed_identity: Mapping[str, object],
    optimizer_steps: int,
    completed_epochs: int,
    next_epoch: int,
    next_microbatch_index: int,
    loss_history: Sequence[float],
    optimizer_step_loss_history: Sequence[float],
    validation_loss_history: Sequence[float],
    epoch_running_loss_sum: float,
    epoch_running_sample_count: int,
    initial_train_loss: float,
    best_validation_loss: float | None,
    best_validation_step: int | None,
    best_model_state_dict: Mapping[str, torch.Tensor] | None,
    epoch_records: Sequence[Mapping[str, object]],
    per_rank_state: Sequence[object],
) -> dict[str, object]:
    return {
        "training_state_layout_version": (
            DISTRIBUTED_RISK_TRAINING_STATE_LAYOUT_VERSION
        ),
        "model_config": model.export_config(),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "optimizer_state_digest_sha256": "0" * 64,
        "config": asdict(config),
        "config_digest_sha256": config_digest,
        "checkpoint_provenance": copy.deepcopy(dict(checkpoint_provenance)),
        "distributed_identity": copy.deepcopy(dict(distributed_identity)),
        "optimizer_steps": optimizer_steps,
        "completed_epochs": completed_epochs,
        "next_epoch": next_epoch,
        "next_microbatch_index": next_microbatch_index,
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
        "epoch_records": copy.deepcopy(list(epoch_records)),
        "per_rank_state": copy.deepcopy(list(per_rank_state)),
        "training_state_semantic_digest_sha256": "0" * 64,
    }


def _capture_per_rank_state(
    *,
    runtime: DistributedRuntime,
    device: torch.device,
    next_epoch: int,
    next_microbatch_index: int,
    epoch_consumed_sample_ids: set[str],
) -> list[object]:
    local = {
        "rank": runtime.rank,
        "next_epoch": next_epoch,
        "next_microbatch_index": next_microbatch_index,
        "epoch_consumed_sample_ids": sorted(epoch_consumed_sample_ids),
        "rng_state": _capture_rng_state(device),
    }
    return _all_gather_objects(local, runtime)


def _broadcast_resume_state(
    *,
    runtime: DistributedRuntime,
    resume_from: Path,
    expected_publication_digest: str,
) -> tuple[dict[str, object], dict[str, object]]:
    objects: list[object | None] = [None]
    if runtime.is_rank_zero:
        try:
            state, manifest = _load_resume_publication(
                resume_from,
                expected_publication_digest=expected_publication_digest,
            )
            objects[0] = {"ok": True, "state": state, "manifest": manifest}
        except Exception as exc:
            objects[0] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "message": str(exc)[:4096],
            }
    dist.broadcast_object_list(objects, src=0)
    envelope = objects[0]
    if not isinstance(envelope, Mapping) or envelope.get("ok") not in {True, False}:
        raise RiskDataContractError("distributed resume envelope is invalid")
    if envelope["ok"] is False:
        raise RiskDataContractError(
            "rank-zero distributed resume failed "
            f"({envelope.get('error_type', 'Error')}): "
            f"{envelope.get('message', 'unknown error')}"
        )
    state = envelope.get("state")
    manifest = envelope.get("manifest")
    return _validate_state_payload(state), dict(manifest)  # type: ignore[arg-type]


def _manifest_semantic_digest(manifest: Mapping[str, object]) -> str:
    bindings = manifest.get("artifact_semantic_bindings")
    if not isinstance(bindings, Mapping):
        raise RiskDataContractError("distributed artifact bindings are invalid")
    result_names = {
        "config_snapshot.json",
        "metrics.json",
        "final_checkpoint.pt",
        "training_state.pt",
    }
    if "best_checkpoint.pt" in bindings:
        result_names.add("best_checkpoint.pt")
    if not result_names.issubset(bindings):
        raise RiskDataContractError("distributed final result bindings are incomplete")
    projection = {
        "training_layout_version": manifest.get("training_layout_version"),
        "schema_version": manifest.get("schema_version"),
        "mode": manifest.get("mode"),
        "stage": manifest.get("stage"),
        "variant": manifest.get("variant"),
        "config_digest_sha256": manifest.get("config_digest_sha256"),
        "checkpoint_provenance": manifest.get("checkpoint_provenance"),
        "distributed_identity": manifest.get("distributed_identity"),
        "train_sample_count": manifest.get("train_sample_count"),
        "optimizer_steps": manifest.get("optimizer_steps"),
        "runtime_environment": manifest.get("runtime_environment"),
        "result_artifact_semantic_bindings": {
            name: bindings[name] for name in sorted(result_names)
        },
    }
    return _sha256_bytes(_canonical_json_bytes(projection))


def _manifest_instance_digest(manifest: Mapping[str, object]) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                key: value
                for key, value in manifest.items()
                if key != "publication_instance_digest_sha256"
            }
        )
    )


def _load_resume_publication(
    state_path: Path,
    *,
    expected_publication_digest: str,
) -> tuple[dict[str, object], dict[str, object]]:
    path = _absolute(state_path)
    if path.name != "training_state.pt" and _INTERVAL_STATE.fullmatch(
        path.name
    ) is None:
        raise RiskDataContractError("distributed resume state filename is invalid")
    snapshots = _snapshot_direct_regular_files(path.parent)
    _validate_checksum_snapshot(snapshots)
    required = {
        "config_snapshot.json",
        "metrics.json",
        "final_checkpoint.pt",
        "training_state.pt",
        "training_manifest.json",
        ".producer-complete",
        "checksums.sha256",
    }
    unknown = {
        name
        for name in snapshots
        if name not in required
        and name != "best_checkpoint.pt"
        and _INTERVAL_STATE.fullmatch(name) is None
    }
    if not required.issubset(snapshots) or unknown:
        raise RiskDataContractError(
            "distributed training publication file set mismatch"
        )
    manifest = _strict_json(
        snapshots["training_manifest.json"],
        label="distributed training manifest",
    )
    marker = _strict_json(
        snapshots[".producer-complete"],
        label="distributed training completion marker",
    )
    if set(manifest) != _MANIFEST_KEYS:
        raise RiskDataContractError("distributed training manifest keys mismatch")
    if manifest.get("training_layout_version") != (
        DISTRIBUTED_RISK_TRAINING_LAYOUT_VERSION
    ):
        raise RiskDataContractError("distributed training manifest layout mismatch")
    semantic_digest = _require_sha256(
        manifest.get("semantic_digest_sha256"),
        "distributed semantic_digest_sha256",
    )
    if semantic_digest != _manifest_semantic_digest(manifest):
        raise RiskDataContractError("distributed manifest semantic digest mismatch")
    instance_digest = _require_sha256(
        manifest.get("publication_instance_digest_sha256"),
        "distributed publication_instance_digest_sha256",
    )
    if instance_digest != _manifest_instance_digest(manifest):
        raise RiskDataContractError("distributed publication instance digest mismatch")
    if instance_digest != _require_sha256(
        expected_publication_digest,
        "expected distributed publication digest",
    ):
        raise RiskDataContractError("distributed publication trusted digest mismatch")
    if set(marker) != _COMPLETE_KEYS or marker != {
        "training_layout_version": DISTRIBUTED_RISK_TRAINING_LAYOUT_VERSION,
        "semantic_digest_sha256": semantic_digest,
        "publication_instance_digest_sha256": instance_digest,
    }:
        raise RiskDataContractError("distributed completion marker mismatch")
    artifact_hashes = manifest.get("artifact_sha256")
    bindings = manifest.get("artifact_semantic_bindings")
    if not isinstance(artifact_hashes, Mapping) or not isinstance(bindings, Mapping):
        raise RiskDataContractError("distributed artifact manifests are invalid")
    expected_artifacts = set(snapshots) - {
        "training_manifest.json",
        ".producer-complete",
        "checksums.sha256",
    }
    if set(artifact_hashes) != expected_artifacts or set(bindings) != expected_artifacts:
        raise RiskDataContractError("distributed artifact manifest file set mismatch")
    for name in expected_artifacts:
        if artifact_hashes.get(name) != _sha256_bytes(snapshots[name]):
            raise RiskDataContractError(
                f"distributed artifact hash mismatch: {name}"
            )
    if path.name not in snapshots:
        raise RiskDataContractError("distributed resume state is not published")
    state = _load_state_source(snapshots[path.name])
    if bindings.get(path.name) != state[
        "training_state_semantic_digest_sha256"
    ]:
        raise RiskDataContractError("distributed state semantic binding mismatch")
    if state.get("checkpoint_provenance") != manifest.get(
        "checkpoint_provenance"
    ):
        raise RiskDataContractError("distributed state provenance mismatch")
    return state, manifest


def _publish_interval_state(
    *,
    runtime: DistributedRuntime,
    staging: Path | None,
    optimizer_steps: int,
    state: Mapping[str, object] | None,
) -> None:
    def publish() -> Mapping[str, object]:
        assert staging is not None
        assert state is not None
        path = staging / f"training_state_step_{optimizer_steps:08d}.pt"
        _write_training_state(path, state)
        return {
            "optimizer_steps": optimizer_steps,
            "training_state_semantic_digest_sha256": (
                load_distributed_risk_training_state(path)[
                    "training_state_semantic_digest_sha256"
                ]
            ),
        }

    broadcast_rank_zero_setup(runtime, publish)


def train_distributed_production_risk_model(
    *,
    train_view: AuthenticatedRiskTrainingView,
    config: ProductionRiskTrainingConfig,
    output_dir: str | Path,
    code_commit: str,
    runtime: DistributedRuntime,
    validation_view: AuthenticatedRiskTrainingView | None = None,
    dataset_family_digest_sha256: str | None = None,
    global_cross_split_leakage: str = "NOT_PROVEN",
    resume_from: str | Path | None = None,
    resume_expected_publication_instance_digest_sha256: str | None = None,
) -> ProductionRiskTrainingResult:
    """Train one risk model with exact ragged DDP sample weighting."""

    checked_code_commit = _require_code_commit(code_commit)
    if (resume_from is None) != (
        resume_expected_publication_instance_digest_sha256 is None
    ):
        raise RiskDataContractError(
            "distributed resume path and publication digest must be supplied together"
        )
    validation_digest, family_digest, leakage_status = _validate_training_inputs(
        train_view=train_view,
        validation_view=validation_view,
        config=config,
        runtime=runtime,
        dataset_family_digest_sha256=dataset_family_digest_sha256,
        global_cross_split_leakage=global_cross_split_leakage,
    )
    destination = _absolute(output_dir)
    staging_payload = broadcast_rank_zero_setup(
        runtime,
        lambda: _prepare_output_staging(destination),
    )
    staging = (
        Path(str(staging_payload["staging"])) if runtime.is_rank_zero else None
    )
    device = torch.device(runtime.device)
    runtime_environment = _runtime_environment(config, runtime)
    runtime_environment_digest = _sha256_bytes(
        _canonical_json_bytes(runtime_environment)
    )
    epoch_zero_plan = train_view.partition(epoch=0)
    if config.stage == "one_shard_smoke":
        consumed_sample_ids = tuple(
            sample_id
            for rank_batches in epoch_zero_plan.rank_microbatches
            for sample_id in rank_batches[0]
        )
    else:
        consumed_sample_ids = tuple(train_view.subset.sample_ids)
    config_digest = _config_digest(config)
    checkpoint_provenance = _checkpoint_provenance(
        train_view=train_view,
        config=config,
        config_digest=config_digest,
        validation_digest=validation_digest,
        family_digest=family_digest,
        leakage_status=leakage_status,
        code_commit=checked_code_commit,
        runtime_environment_digest=runtime_environment_digest,
        consumed_sample_ids=consumed_sample_ids,
    )

    try:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        base_model = RiskModel(
            variant=config.variant,
            hidden_channels=config.hidden_channels,
        ).to(device=device)
        optimizer = torch.optim.AdamW(
            base_model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )

        optimizer_steps = 0
        completed_epochs = 0
        next_epoch = 0
        next_microbatch_index = 0
        loss_history: list[float] = []
        optimizer_step_loss_history: list[float] = []
        validation_loss_history: list[float] = []
        epoch_running_loss_sum = 0.0
        epoch_running_sample_count = 0
        initial_train_loss = float("nan")
        best_validation_loss: float | None = None
        best_validation_step: int | None = None
        best_model_state_dict: Mapping[str, torch.Tensor] | None = None
        epoch_records: list[dict[str, object]] = []
        epoch_consumed_sample_ids: set[str] = set()
        resume_lineage: dict[str, object] | None = None
        resume_state: dict[str, object] | None = None

        if resume_from is not None:
            assert resume_expected_publication_instance_digest_sha256 is not None
            resume_state, resume_manifest = _broadcast_resume_state(
                runtime=runtime,
                resume_from=_absolute(resume_from),
                expected_publication_digest=_require_sha256(
                    resume_expected_publication_instance_digest_sha256,
                    "resume expected publication digest",
                ),
            )
            prior_config = _validate_resume_config(resume_state["config"], config)
            if resume_state.get("config_digest_sha256") != _config_digest(prior_config):
                raise RiskDataContractError("distributed resume config digest mismatch")
            next_epoch = int(resume_state["next_epoch"])
            next_microbatch_index = int(resume_state["next_microbatch_index"])
            expected_plan = train_view.partition(epoch=next_epoch)
            validate_distributed_resume_bindings(
                resume_state,
                expected_world_size=runtime.world_size,
                expected_backend=str(runtime.backend),
                expected_train_snapshot_digest_sha256=(
                    train_view.snapshot.snapshot_digest_sha256
                ),
                expected_train_view_digest_sha256=(
                    train_view.training_view_digest_sha256
                ),
                expected_partition_spec_digest_sha256=(
                    expected_plan.partition_spec_digest_sha256
                ),
                expected_epoch_plan_digest_sha256=(
                    expected_plan.epoch_plan_digest_sha256
                ),
            )
            saved_provenance = resume_state.get("checkpoint_provenance")
            if not isinstance(saved_provenance, Mapping):
                raise RiskDataContractError("distributed resume provenance is invalid")
            for field, expected in checkpoint_provenance.items():
                if field == "config_digest":
                    continue
                if saved_provenance.get(field) != expected:
                    raise RiskDataContractError(
                        f"distributed resume provenance mismatch: {field}"
                    )
            if saved_provenance.get("config_digest") != _config_digest(prior_config):
                raise RiskDataContractError(
                    "distributed resume provenance config digest mismatch"
                )
            if resume_state.get("model_config") != base_model.export_config():
                raise RiskDataContractError("distributed resume model config mismatch")
            try:
                base_model.load_state_dict(
                    resume_state["model_state_dict"], strict=True
                )
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            except (TypeError, ValueError, RuntimeError) as exc:
                raise RiskDataContractError(
                    "distributed resume model/optimizer state is invalid"
                ) from exc
            _optimizer_to_device(optimizer, device)
            optimizer_steps = int(resume_state["optimizer_steps"])
            completed_epochs = int(resume_state["completed_epochs"])
            loss_history = [float(value) for value in resume_state["loss_history"]]
            optimizer_step_loss_history = [
                float(value)
                for value in resume_state["optimizer_step_loss_history"]
            ]
            validation_loss_history = [
                float(value) for value in resume_state["validation_loss_history"]
            ]
            epoch_running_loss_sum = float(
                resume_state["epoch_running_loss_sum"]
            )
            epoch_running_sample_count = int(
                resume_state["epoch_running_sample_count"]
            )
            initial_train_loss = float(resume_state["initial_train_loss"])
            best_validation_loss = (
                None
                if resume_state["best_validation_loss"] is None
                else float(resume_state["best_validation_loss"])
            )
            best_validation_step = (
                None
                if resume_state["best_validation_step"] is None
                else int(resume_state["best_validation_step"])
            )
            best_model_state_dict = resume_state["best_model_state_dict"]
            epoch_records = copy.deepcopy(resume_state["epoch_records"])
            rank_state = resume_state["per_rank_state"][runtime.rank]
            if (
                not isinstance(rank_state, Mapping)
                or rank_state.get("rank") != runtime.rank
                or rank_state.get("next_epoch") != next_epoch
                or rank_state.get("next_microbatch_index")
                != next_microbatch_index
            ):
                raise RiskDataContractError("distributed resume per-rank cursor mismatch")
            epoch_consumed_sample_ids = set(
                rank_state.get("epoch_consumed_sample_ids", ())
            )
            resume_lineage = {
                "parent_semantic_digest_sha256": resume_manifest[
                    "semantic_digest_sha256"
                ],
                "parent_publication_instance_digest_sha256": resume_manifest[
                    "publication_instance_digest_sha256"
                ],
                "resume_state_filename": Path(resume_from).name,
                "resume_state_semantic_digest_sha256": resume_state[
                    "training_state_semantic_digest_sha256"
                ],
                "resume_optimizer_step": optimizer_steps,
            }

        if device.type == "cuda":
            model = DistributedDataParallel(
                base_model,
                device_ids=[runtime.local_rank],
                output_device=runtime.local_rank,
            )
        else:
            model = DistributedDataParallel(base_model)

        if resume_state is not None:
            rank_state = resume_state["per_rank_state"][runtime.rank]
            _restore_rng_state(rank_state["rng_state"], device=device)
        else:
            rank_seed = config.seed + runtime.rank
            random.seed(rank_seed)
            np.random.seed(rank_seed)
            torch.manual_seed(rank_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(rank_seed)

        if resume_state is None:
            initial_train_loss, _ = _evaluate_batches(
                model,
                view=train_view,
                runtime=runtime,
                config=config,
                epoch=0,
                first_microbatch_only=config.stage == "one_shard_smoke",
            )
            loss_history.append(initial_train_loss)

        all_finite = True
        smoke_actual_ids: list[str] | None = None
        training_finished = False
        for epoch in range(next_epoch, config.epochs):
            plan = train_view.partition(epoch=epoch)
            rank_batches = plan.rank_microbatches[runtime.rank]
            start_index = next_microbatch_index if epoch == next_epoch else 0
            if start_index < 0 or start_index > len(rank_batches):
                raise RiskDataContractError("distributed resume microbatch cursor invalid")
            if start_index and start_index % config.gradient_accumulation_steps != 0:
                raise RiskDataContractError(
                    "distributed resume cursor is not an optimizer-window boundary"
                )
            limit = 1 if config.stage == "one_shard_smoke" else len(rank_batches)
            if start_index >= limit:
                raise RiskDataContractError("distributed epoch has no remaining batches")
            window_start = start_index
            while window_start < limit:
                window_end = min(
                    window_start + config.gradient_accumulation_steps,
                    limit,
                )
                window = rank_batches[window_start:window_end]
                local_window_count = sum(len(batch) for batch in window)
                global_window_count = all_reduce_sample_count(
                    local_window_count,
                    runtime,
                )
                optimizer.zero_grad(set_to_none=True)
                local_loss_sum = 0.0
                local_crossings = 0
                local_comparisons = 0
                local_outputs_finite = True
                for position, sample_ids in enumerate(window):
                    terminal = position == len(window) - 1
                    synchronization = (
                        nullcontext() if terminal else model.no_sync()
                    )
                    with synchronization:
                        batch = _move_batch(
                            train_view.snapshot.batch(sample_ids),
                            device,
                        )
                        model.train()
                        output, losses = compute_risk_batch_loss(
                            model,
                            batch,
                            lambda_collision=config.lambda_collision,
                        )
                        local_outputs_finite = local_outputs_finite and all(
                            bool(torch.isfinite(value).all().item())
                            for value in (*output.values(), *losses.values())
                        )
                        crossing = (
                            output["quantiles"][:, 1:]
                            < output["quantiles"][:, :-1]
                        )
                        local_crossings += int(
                            torch.count_nonzero(crossing).item()
                        )
                        local_comparisons += crossing.numel()
                        local_count = len(sample_ids)
                        local_loss_sum += (
                            float(losses["total"].detach().item()) * local_count
                        )
                        scaled_loss = scale_distributed_batch_mean_loss(
                            losses["total"],
                            local_sample_count=local_count,
                            global_window_sample_count=global_window_count,
                            runtime=runtime,
                        )
                        scaled_loss.backward()
                    epoch_consumed_sample_ids.update(sample_ids)
                local_gradients_finite = all(
                    parameter.grad is None
                    or bool(torch.isfinite(parameter.grad).all().item())
                    for parameter in base_model.parameters()
                )
                if not _all_ranks_true(
                    local_outputs_finite and local_gradients_finite,
                    runtime,
                ):
                    all_finite = False
                    raise RiskDataContractError(
                        "distributed forward/loss/gradient contains NaN/Inf"
                    )
                if not _all_ranks_true(local_crossings == 0, runtime):
                    raise RiskDataContractError(
                        "distributed quantile outputs cross"
                    )
                optimizer.step()
                optimizer_steps += 1
                (
                    global_loss_sum,
                    reduced_sample_count,
                    _,
                    _,
                ) = _all_reduce_values(
                    (
                        local_loss_sum,
                        local_window_count,
                        local_crossings,
                        local_comparisons,
                    ),
                    runtime,
                )
                if int(reduced_sample_count) != global_window_count:
                    raise RiskDataContractError(
                        "distributed optimizer window sample count mismatch"
                    )
                optimizer_step_loss_history.append(
                    global_loss_sum / reduced_sample_count
                )
                epoch_running_loss_sum += global_loss_sum
                epoch_running_sample_count += int(reduced_sample_count)
                next_microbatch_index = window_end
                epoch_complete = (
                    config.stage != "one_shard_smoke"
                    and next_microbatch_index == len(rank_batches)
                )
                if epoch_complete:
                    gathered_ids = _all_gather_objects(
                        tuple(sorted(epoch_consumed_sample_ids)),
                        runtime,
                    )
                    _validate_actual_rank_membership(
                        gathered_ids,
                        expected_ids=set(train_view.subset.sample_ids),
                    )
                    if epoch_running_sample_count != len(
                        train_view.subset.sample_ids
                    ):
                        raise RiskDataContractError(
                            "distributed epoch sample count mismatch"
                        )
                    loss_history.append(
                        epoch_running_loss_sum / epoch_running_sample_count
                    )
                    completed_epochs = epoch + 1
                    next_epoch = epoch + 1
                    next_microbatch_index = 0
                    epoch_running_loss_sum = 0.0
                    epoch_running_sample_count = 0
                    epoch_records.append(
                        _rank_membership_record(plan, epoch=epoch)
                    )
                    epoch_consumed_sample_ids = set()
                    if validation_view is not None:
                        validation_loss, _ = _evaluate_batches(
                            model,
                            view=validation_view,
                            runtime=runtime,
                            config=config,
                            epoch=epoch,
                        )
                        validation_loss_history.append(validation_loss)
                        if (
                            best_validation_loss is None
                            or validation_loss < best_validation_loss
                        ):
                            best_validation_loss = validation_loss
                            best_validation_step = optimizer_steps
                            best_model_state_dict = copy.deepcopy(
                                base_model.state_dict()
                            )
                else:
                    next_epoch = epoch

                per_rank_state = _capture_per_rank_state(
                    runtime=runtime,
                    device=device,
                    next_epoch=next_epoch,
                    next_microbatch_index=next_microbatch_index,
                    epoch_consumed_sample_ids=epoch_consumed_sample_ids,
                )
                current_identity = _distributed_identity(
                    train_view=train_view,
                    validation_view=validation_view,
                    runtime=runtime,
                    current_epoch=next_epoch,
                )
                state_payload = (
                    _make_state(
                        model=base_model,
                        optimizer=optimizer,
                        config=config,
                        config_digest=config_digest,
                        checkpoint_provenance=checkpoint_provenance,
                        distributed_identity=current_identity,
                        optimizer_steps=optimizer_steps,
                        completed_epochs=completed_epochs,
                        next_epoch=next_epoch,
                        next_microbatch_index=next_microbatch_index,
                        loss_history=loss_history,
                        optimizer_step_loss_history=optimizer_step_loss_history,
                        validation_loss_history=validation_loss_history,
                        epoch_running_loss_sum=epoch_running_loss_sum,
                        epoch_running_sample_count=epoch_running_sample_count,
                        initial_train_loss=initial_train_loss,
                        best_validation_loss=best_validation_loss,
                        best_validation_step=best_validation_step,
                        best_model_state_dict=best_model_state_dict,
                        epoch_records=epoch_records,
                        per_rank_state=per_rank_state,
                    )
                    if runtime.is_rank_zero
                    else None
                )
                if optimizer_steps % config.checkpoint_interval_steps == 0:
                    _publish_interval_state(
                        runtime=runtime,
                        staging=staging,
                        optimizer_steps=optimizer_steps,
                        state=state_payload,
                    )
                if config.stage == "one_shard_smoke":
                    gathered_ids = _all_gather_objects(
                        tuple(sorted(epoch_consumed_sample_ids)),
                        runtime,
                    )
                    normalized = _validate_actual_rank_membership(
                        gathered_ids,
                        expected_ids=set(consumed_sample_ids),
                    )
                    smoke_actual_ids = sorted(
                        sample_id for ids in normalized for sample_id in ids
                    )
                    training_finished = True
                    break
                window_start = window_end
            if training_finished:
                break
            next_microbatch_index = 0

        if optimizer_steps < 1:
            raise RiskDataContractError("distributed training performed no optimizer steps")
        if config.stage == "one_shard_smoke":
            if optimizer_steps != 1 or smoke_actual_ids is None:
                raise RiskDataContractError(
                    "distributed one_shard_smoke must perform one global step"
                )
            final_train_loss, final_crossing_rate = _evaluate_batches(
                model,
                view=train_view,
                runtime=runtime,
                config=config,
                epoch=0,
                first_microbatch_only=True,
            )
            loss_history.append(final_train_loss)
        else:
            final_train_loss, final_crossing_rate = _evaluate_batches(
                model,
                view=train_view,
                runtime=runtime,
                config=config,
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
            raise RiskDataContractError("distributed loss history contains NaN/Inf")

        per_rank_state = _capture_per_rank_state(
            runtime=runtime,
            device=device,
            next_epoch=next_epoch,
            next_microbatch_index=next_microbatch_index,
            epoch_consumed_sample_ids=epoch_consumed_sample_ids,
        )
        final_identity = _distributed_identity(
            train_view=train_view,
            validation_view=validation_view,
            runtime=runtime,
            current_epoch=next_epoch,
        )
        final_state = (
            _make_state(
                model=base_model,
                optimizer=optimizer,
                config=config,
                config_digest=config_digest,
                checkpoint_provenance=checkpoint_provenance,
                distributed_identity=final_identity,
                optimizer_steps=optimizer_steps,
                completed_epochs=completed_epochs,
                next_epoch=next_epoch,
                next_microbatch_index=next_microbatch_index,
                loss_history=loss_history,
                optimizer_step_loss_history=optimizer_step_loss_history,
                validation_loss_history=validation_loss_history,
                epoch_running_loss_sum=epoch_running_loss_sum,
                epoch_running_sample_count=epoch_running_sample_count,
                initial_train_loss=initial_train_loss,
                best_validation_loss=best_validation_loss,
                best_validation_step=best_validation_step,
                best_model_state_dict=best_model_state_dict,
                epoch_records=epoch_records,
                per_rank_state=per_rank_state,
            )
            if runtime.is_rank_zero
            else None
        )
        metrics = {
            "training_layout_version": DISTRIBUTED_RISK_TRAINING_LAYOUT_VERSION,
            "mode": "production",
            "stage": config.stage,
            "variant": config.variant,
            "seed": config.seed,
            "device": config.device,
            "world_size": runtime.world_size,
            "backend": runtime.backend,
            "nominal_per_rank_batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "selected_sample_count": len(train_view.subset.sample_ids),
            "consumed_sample_count": len(consumed_sample_ids),
            "smoke_consumed_sample_ids": smoke_actual_ids,
            "rank_local_microbatch_sizes": [
                list(sizes) for sizes in epoch_zero_plan.local_microbatch_sizes
            ],
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
            "epoch_records": epoch_records,
            "test_samples_used_for_training_or_selection": 0,
        }

        def publish_final() -> Mapping[str, object]:
            assert staging is not None
            assert final_state is not None
            _write_training_state(staging / "training_state.pt", final_state)
            save_risk_checkpoint(
                staging / "final_checkpoint.pt",
                model=base_model,
                mode="production",
                provenance=checkpoint_provenance,
            )
            if config.stage == "formal_50k":
                if best_model_state_dict is None:
                    raise RiskDataContractError(
                        "formal_50k DDP completed without best validation state"
                    )
                final_model_state = copy.deepcopy(base_model.state_dict())
                base_model.load_state_dict(best_model_state_dict, strict=True)
                save_risk_checkpoint(
                    staging / "best_checkpoint.pt",
                    model=base_model,
                    mode="production",
                    provenance=checkpoint_provenance,
                )
                base_model.load_state_dict(final_model_state, strict=True)
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
                state_value = load_distributed_risk_training_state(state_path)
                semantic_bindings[state_path.name] = str(
                    state_value["training_state_semantic_digest_sha256"]
                )
            for checkpoint_name in ("final_checkpoint.pt", "best_checkpoint.pt"):
                checkpoint_path = staging / checkpoint_name
                if not checkpoint_path.exists():
                    continue
                _, checkpoint = load_risk_checkpoint(
                    checkpoint_path,
                    expected_mode="production",
                    expected_provenance=checkpoint_provenance,
                )
                semantic_bindings[checkpoint_name] = str(
                    checkpoint["checkpoint_semantic_digest_sha256"]
                )
            manifest: dict[str, object] = {
                "training_layout_version": (
                    DISTRIBUTED_RISK_TRAINING_LAYOUT_VERSION
                ),
                "schema_version": SCHEMA_VERSION,
                "mode": "production",
                "stage": config.stage,
                "variant": config.variant,
                "config_digest_sha256": config_digest,
                "checkpoint_provenance": checkpoint_provenance,
                "distributed_identity": final_identity,
                "train_sample_count": len(consumed_sample_ids),
                "optimizer_steps": optimizer_steps,
                "artifact_sha256": artifact_hashes,
                "artifact_semantic_bindings": semantic_bindings,
                "runtime_environment": runtime_environment,
                "resume_lineage": resume_lineage,
                "writer_rank": 0,
            }
            manifest["semantic_digest_sha256"] = _manifest_semantic_digest(
                manifest
            )
            semantic_digest = str(manifest["semantic_digest_sha256"])
            manifest["publication_instance_digest_sha256"] = (
                _manifest_instance_digest(manifest)
            )
            instance_digest = str(
                manifest["publication_instance_digest_sha256"]
            )
            _write_json(staging / "training_manifest.json", manifest)
            _write_json(
                staging / ".producer-complete",
                {
                    "training_layout_version": (
                        DISTRIBUTED_RISK_TRAINING_LAYOUT_VERSION
                    ),
                    "semantic_digest_sha256": semantic_digest,
                    "publication_instance_digest_sha256": instance_digest,
                },
            )
            _write_checksum_manifest(staging)
            _load_resume_publication(
                staging / "training_state.pt",
                expected_publication_digest=instance_digest,
            )
            _atomic_rename_directory_noreplace(staging, destination)
            return {
                "output_dir": str(destination),
                "semantic_digest_sha256": semantic_digest,
                "publication_instance_digest_sha256": instance_digest,
            }

        result_payload = broadcast_rank_zero_setup(runtime, publish_final)
        semantic_digest = _require_sha256(
            result_payload.get("semantic_digest_sha256"),
            "distributed result semantic digest",
        )
        output = _absolute(str(result_payload["output_dir"]))
        return ProductionRiskTrainingResult(
            output_dir=output,
            best_checkpoint=(
                output / "best_checkpoint.pt"
                if config.stage == "formal_50k"
                else None
            ),
            final_checkpoint=output / "final_checkpoint.pt",
            training_state_checkpoint=output / "training_state.pt",
            metrics_path=output / "metrics.json",
            manifest_path=output / "training_manifest.json",
            semantic_digest_sha256=semantic_digest,
        )
    except BaseException:
        if runtime.is_rank_zero and staging is not None and staging.exists():
            shutil.rmtree(staging)
        raise


def _prepare_output_staging(destination: Path) -> dict[str, object]:
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        f".{destination.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    return {"staging": str(staging), "destination": str(destination)}


__all__ = [
    "DISTRIBUTED_RISK_TRAINING_LAYOUT_VERSION",
    "DISTRIBUTED_RISK_TRAINING_STATE_LAYOUT_VERSION",
    "load_distributed_risk_training_state",
    "train_distributed_production_risk_model",
    "validate_distributed_resume_bindings",
]
