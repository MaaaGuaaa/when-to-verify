"""Single-node distributed runtime and deterministic risk-batch partitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import os
from typing import Callable, Mapping, Sequence

import torch
import torch.distributed as dist

from src.datasets.risk_dataloader import RiskDataContractError


_PARTITION_LAYOUT_VERSION = "sop09_synchronous_partition_v1"
_RUNTIME_ENVIRONMENT_NAMES = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
_MAX_SETUP_ENVELOPE_BYTES = 64 * 1024


@dataclass(frozen=True)
class DistributedRuntime:
    """Validated torchrun topology without initializing a process group."""

    rank: int
    world_size: int
    local_rank: int
    backend: str | None
    device: str

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class SynchronousPartitionPlan:
    """Exact, non-padding assignment with equal collective counts per rank."""

    layout_version: str
    subset_digest_sha256: str
    seed: int
    epoch: int
    world_size: int
    batch_size: int
    gradient_accumulation_steps: int
    rank_microbatches: tuple[tuple[tuple[str, ...], ...], ...]
    partition_spec_digest_sha256: str
    epoch_plan_digest_sha256: str

    @property
    def local_microbatch_sizes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(len(batch) for batch in rank_batches)
            for rank_batches in self.rank_microbatches
        )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RiskDataContractError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RiskDataContractError(f"{name} must be a non-negative integer")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RiskDataContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def discover_distributed_runtime(configured_device: str) -> DistributedRuntime:
    """Validate torchrun variables before source datasets can be opened."""

    present = [name in os.environ for name in _RUNTIME_ENVIRONMENT_NAMES]
    if any(present) and not all(present):
        raise RiskDataContractError(
            "RANK, WORLD_SIZE, and LOCAL_RANK must all be present or all absent"
        )
    if not all(present):
        return DistributedRuntime(0, 1, 0, None, configured_device)

    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except ValueError as exc:
        raise RiskDataContractError("distributed environment variables must be integers") from exc
    _nonnegative_int(rank, "RANK")
    _positive_int(world_size, "WORLD_SIZE")
    _nonnegative_int(local_rank, "LOCAL_RANK")
    if rank >= world_size or local_rank >= world_size:
        raise RiskDataContractError("distributed rank is outside WORLD_SIZE")
    if "LOCAL_WORLD_SIZE" in os.environ:
        try:
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        except ValueError as exc:
            raise RiskDataContractError("LOCAL_WORLD_SIZE must be an integer") from exc
        _positive_int(local_world_size, "LOCAL_WORLD_SIZE")
        if local_world_size != world_size or local_rank >= local_world_size:
            raise RiskDataContractError("distributed training supports single-node topology only")
    if "GROUP_RANK" in os.environ:
        try:
            group_rank = int(os.environ["GROUP_RANK"])
        except ValueError as exc:
            raise RiskDataContractError("GROUP_RANK must be an integer") from exc
        if group_rank != 0:
            raise RiskDataContractError("distributed training supports single-node topology only")
    if world_size == 1:
        if rank != 0 or local_rank != 0:
            raise RiskDataContractError("WORLD_SIZE=1 requires rank zero")
        return DistributedRuntime(0, 1, 0, None, configured_device)
    if configured_device == "cpu":
        return DistributedRuntime(rank, world_size, local_rank, "gloo", "cpu")
    if configured_device != "cuda":
        raise RiskDataContractError(
            "distributed CUDA training rejects a hard-coded CUDA device; use cuda"
        )
    return DistributedRuntime(rank, world_size, local_rank, "nccl", f"cuda:{local_rank}")


def initialize_distributed_process_group(
    runtime: DistributedRuntime,
    *,
    init_method: str | None = None,
    timeout_seconds: int = 120,
) -> None:
    """Initialize one validated single-node process group."""

    if not isinstance(runtime, DistributedRuntime):
        raise RiskDataContractError("runtime must be a DistributedRuntime")
    timeout = _positive_int(timeout_seconds, "timeout_seconds")
    if not runtime.is_distributed:
        return
    if runtime.backend not in {"gloo", "nccl"}:
        raise RiskDataContractError("distributed backend must be gloo or nccl")
    if not dist.is_available():
        raise RiskDataContractError("PyTorch distributed support is unavailable")
    if dist.is_initialized():
        raise RiskDataContractError("distributed process group is already initialized")
    if runtime.backend == "nccl":
        if not dist.is_nccl_available() or not torch.cuda.is_available():
            raise RiskDataContractError("NCCL distributed training requires allocated CUDA GPUs")
        if runtime.local_rank >= torch.cuda.device_count():
            raise RiskDataContractError("LOCAL_RANK does not identify an allocated CUDA GPU")
        torch.cuda.set_device(runtime.local_rank)
    arguments: dict[str, object] = {
        "backend": runtime.backend,
        "rank": runtime.rank,
        "world_size": runtime.world_size,
        "timeout": timedelta(seconds=timeout),
    }
    if init_method is not None:
        if not isinstance(init_method, str) or not init_method:
            raise RiskDataContractError("init_method must be a non-empty string")
        arguments["init_method"] = init_method
    try:
        dist.init_process_group(**arguments)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RiskDataContractError(f"failed to initialize distributed process group: {exc}") from exc


def destroy_distributed_process_group() -> None:
    """Destroy an initialized process group on normal and error paths."""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_rank_zero_setup(
    runtime: DistributedRuntime,
    setup: Callable[[], Mapping[str, object]],
) -> dict[str, object]:
    """Run setup only on rank zero and broadcast one bounded JSON envelope."""

    if not isinstance(runtime, DistributedRuntime):
        raise RiskDataContractError("runtime must be a DistributedRuntime")
    if not callable(setup):
        raise RiskDataContractError("setup must be callable")
    if not runtime.is_distributed:
        value = setup()
        if not isinstance(value, Mapping):
            raise RiskDataContractError("rank-zero setup result must be a mapping")
        return dict(value)
    if not dist.is_available() or not dist.is_initialized():
        raise RiskDataContractError("distributed process group must be initialized before setup")

    object_list: list[object | None] = [None]
    if runtime.is_rank_zero:
        try:
            value = setup()
            if not isinstance(value, Mapping):
                raise RiskDataContractError("rank-zero setup result must be a mapping")
            payload = dict(value)
            envelope: dict[str, object] = {"ok": True, "payload": payload}
            encoded = _canonical_json_bytes(envelope)
            if len(encoded) > _MAX_SETUP_ENVELOPE_BYTES:
                raise RiskDataContractError("rank-zero setup envelope exceeds size limit")
        except Exception as exc:
            envelope = {
                "ok": False,
                "error_type": type(exc).__name__,
                "message": str(exc)[:4096],
            }
        object_list[0] = envelope
    dist.broadcast_object_list(object_list, src=0)
    received = object_list[0]
    if not isinstance(received, Mapping) or received.get("ok") not in {True, False}:
        raise RiskDataContractError("distributed setup envelope is invalid")
    if received["ok"] is False:
        error_type = str(received.get("error_type", "Error"))
        message = str(received.get("message", "rank-zero setup failed"))
        raise RiskDataContractError(
            f"rank-zero distributed setup failed ({error_type}): {message}"
        )
    payload = received.get("payload")
    if not isinstance(payload, Mapping):
        raise RiskDataContractError("distributed setup payload is invalid")
    return dict(payload)


def all_reduce_sample_count(
    local_sample_count: int,
    runtime: DistributedRuntime,
) -> int:
    """Return the exact real-sample count for one synchronized window."""

    local_count = _positive_int(local_sample_count, "local_sample_count")
    if not isinstance(runtime, DistributedRuntime):
        raise RiskDataContractError("runtime must be a DistributedRuntime")
    if not runtime.is_distributed:
        return local_count
    if not dist.is_available() or not dist.is_initialized():
        raise RiskDataContractError("distributed process group is not initialized")
    count = torch.tensor(local_count, dtype=torch.int64, device=torch.device(runtime.device))
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    global_count = int(count.item())
    if global_count < local_count or global_count < runtime.world_size:
        raise RiskDataContractError("distributed global sample count is invalid")
    return global_count


def scale_distributed_batch_mean_loss(
    local_mean_loss: torch.Tensor,
    *,
    local_sample_count: int,
    global_window_sample_count: int,
    runtime: DistributedRuntime,
) -> torch.Tensor:
    """Scale a local batch mean so DDP averaging yields a global sample mean."""

    if not torch.is_tensor(local_mean_loss) or local_mean_loss.numel() != 1:
        raise RiskDataContractError("local_mean_loss must be a scalar tensor")
    local_count = _positive_int(local_sample_count, "local_sample_count")
    global_count = _positive_int(
        global_window_sample_count, "global_window_sample_count"
    )
    if not isinstance(runtime, DistributedRuntime):
        raise RiskDataContractError("runtime must be a DistributedRuntime")
    if global_count < local_count:
        raise RiskDataContractError("global sample count cannot be smaller than local count")
    scale = (local_count * runtime.world_size) / global_count
    return local_mean_loss * float(scale)


def build_synchronous_partition_plan(
    sample_ids: Sequence[str],
    *,
    subset_digest_sha256: str,
    seed: int,
    epoch: int,
    world_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> SynchronousPartitionPlan:
    """Partition a deterministic epoch order without sample duplication or padding."""

    ids = tuple(sample_ids)
    if not ids or any(not isinstance(sample_id, str) or not sample_id for sample_id in ids):
        raise RiskDataContractError("sample_ids must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise RiskDataContractError("sample_ids must be unique")
    digest = _sha256(subset_digest_sha256, "subset_digest_sha256")
    checked_seed = _nonnegative_int(seed, "seed")
    checked_epoch = _nonnegative_int(epoch, "epoch")
    checked_world_size = _positive_int(world_size, "world_size")
    checked_batch_size = _positive_int(batch_size, "batch_size")
    checked_accumulation = _positive_int(
        gradient_accumulation_steps, "gradient_accumulation_steps"
    )
    if checked_world_size > 1:
        if len(ids) < checked_world_size:
            raise RiskDataContractError("distributed training requires at least WORLD_SIZE samples")
        if checked_batch_size < 2:
            raise RiskDataContractError(
                "distributed training requires a per-rank batch_size of at least two"
            )

    ordered = list(ids)
    rank_ranges = tuple(
        tuple(ordered[(rank * len(ordered)) // checked_world_size : ((rank + 1) * len(ordered)) // checked_world_size])
        for rank in range(checked_world_size)
    )
    batch_counts = [
        (len(rows) + checked_batch_size - 1) // checked_batch_size for rows in rank_ranges
    ]
    synchronous_steps = max(batch_counts)
    if synchronous_steps < 1:
        raise RiskDataContractError("partition plan has no synchronous microbatches")
    # Rebalance each contiguous range into exactly synchronous_steps non-empty batches.
    if any(len(rows) < synchronous_steps for rows in rank_ranges):
        raise RiskDataContractError("partition plan cannot form non-empty synchronous microbatches")
    rank_microbatches = tuple(
        tuple(
            rows[(step * len(rows)) // synchronous_steps : ((step + 1) * len(rows)) // synchronous_steps]
            for step in range(synchronous_steps)
        )
        for rows in rank_ranges
    )
    if any(len(batch) > checked_batch_size for rows in rank_microbatches for batch in rows):
        raise RiskDataContractError("partition plan exceeds per-rank batch_size")
    flattened = tuple(sample_id for rows in rank_microbatches for batch in rows for sample_id in batch)
    if len(flattened) != len(ids) or set(flattened) != set(ids) or len(flattened) != len(set(flattened)):
        raise RiskDataContractError("partition plan has overlapping or missing sample IDs")
    partition_spec = {
        "layout_version": _PARTITION_LAYOUT_VERSION,
        "subset_digest_sha256": digest,
        "seed": checked_seed,
        "world_size": checked_world_size,
        "batch_size": checked_batch_size,
        "gradient_accumulation_steps": checked_accumulation,
    }
    return SynchronousPartitionPlan(
        layout_version=_PARTITION_LAYOUT_VERSION,
        subset_digest_sha256=digest,
        seed=checked_seed,
        epoch=checked_epoch,
        world_size=checked_world_size,
        batch_size=checked_batch_size,
        gradient_accumulation_steps=checked_accumulation,
        rank_microbatches=rank_microbatches,
        partition_spec_digest_sha256=_digest(partition_spec),
        epoch_plan_digest_sha256=_digest({**partition_spec, "epoch": checked_epoch, "rank_microbatches": rank_microbatches}),
    )
