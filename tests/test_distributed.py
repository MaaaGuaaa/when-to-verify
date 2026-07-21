from __future__ import annotations

from contextlib import nullcontext
import multiprocessing
from pathlib import Path
import queue

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.datasets.risk_dataloader import RiskDataContractError
from src.training.distributed import (
    DistributedRuntime,
    all_reduce_sample_count,
    broadcast_rank_zero_setup,
    build_synchronous_partition_plan,
    destroy_distributed_process_group,
    discover_distributed_runtime,
    initialize_distributed_process_group,
    scale_distributed_batch_mean_loss,
)


def _gloo_setup_worker(rank: int, init_path: str, fail: bool, result_queue) -> None:
    runtime = DistributedRuntime(
        rank=rank,
        world_size=2,
        local_rank=rank,
        backend="gloo",
        device="cpu",
    )
    status: tuple[object, ...]
    try:
        initialize_distributed_process_group(
            runtime,
            init_method=f"file://{init_path}",
        )

        def setup():
            if fail:
                raise RiskDataContractError("fixture setup failed")
            return {"snapshot_digest_sha256": "a" * 64, "sample_count": 11}

        descriptor = broadcast_rank_zero_setup(runtime, setup)
        status = ("ok", rank, descriptor)
    except Exception as exc:
        status = ("error", rank, type(exc).__name__, str(exc))
    finally:
        destroy_distributed_process_group()
        result_queue.put((*status, dist.is_initialized()))


def _run_gloo_setup(tmp_path: Path, *, fail: bool) -> list[tuple[object, ...]]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    init_path = tmp_path / ("failure-init" if fail else "success-init")
    processes = [
        context.Process(
            target=_gloo_setup_worker,
            args=(rank, str(init_path), fail, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = []
    try:
        for _ in processes:
            results.append(result_queue.get(timeout=30))
    except queue.Empty:
        pytest.fail("Gloo setup workers did not report before timeout")
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert [process.exitcode for process in processes] == [0, 0]
    return sorted(results, key=lambda item: int(item[1]))


def _gloo_gradient_worker(rank: int, init_path: str, result_queue) -> None:
    runtime = DistributedRuntime(rank, 2, rank, "gloo", "cpu")
    sample_ids = tuple(f"sample-{index:02d}" for index in range(19))
    plan = build_synchronous_partition_plan(
        sample_ids,
        subset_digest_sha256="c" * 64,
        seed=3,
        epoch=0,
        world_size=2,
        batch_size=3,
        gradient_accumulation_steps=2,
    )
    try:
        initialize_distributed_process_group(
            runtime,
            init_method=f"file://{init_path}",
        )
        module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.zero_()
        model = DistributedDataParallel(module)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        rank_batches = plan.rank_microbatches[rank]
        optimizer.zero_grad(set_to_none=True)
        for window_start in range(0, len(rank_batches), 2):
            window = rank_batches[window_start : window_start + 2]
            global_count = all_reduce_sample_count(
                sum(len(batch) for batch in window), runtime
            )
            for microstep, batch_ids in enumerate(window):
                indices = torch.tensor(
                    [int(sample_id.split("-")[1]) + 1 for sample_id in batch_ids],
                    dtype=torch.float32,
                ).reshape(-1, 1)
                target = 2.0 * indices
                terminal = microstep == len(window) - 1
                context = model.no_sync() if not terminal else nullcontext()
                with context:
                    local_mean = torch.mean((model(indices) - target) ** 2)
                    scaled = scale_distributed_batch_mean_loss(
                        local_mean,
                        local_sample_count=len(batch_ids),
                        global_window_sample_count=global_count,
                        runtime=runtime,
                    )
                    scaled.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        result_queue.put((rank, float(module.weight.detach().item())))
    finally:
        destroy_distributed_process_group()


def test_discover_distributed_runtime_keeps_absent_environment_single_process(monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)

    runtime = discover_distributed_runtime("cpu")

    assert runtime.rank == 0
    assert runtime.world_size == 1
    assert runtime.local_rank == 0
    assert runtime.backend is None
    assert runtime.device == "cpu"


def test_discover_distributed_runtime_rejects_partial_environment(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    with pytest.raises(RiskDataContractError, match="all be present or all absent"):
        discover_distributed_runtime("cpu")


def test_discover_distributed_runtime_requires_unindexed_cuda_for_ddp(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    with pytest.raises(RiskDataContractError, match="hard-coded CUDA device"):
        discover_distributed_runtime("cuda:1")


def test_discover_distributed_runtime_rejects_multi_node_topology(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")

    with pytest.raises(RiskDataContractError, match="single-node"):
        discover_distributed_runtime("cpu")


def test_synchronous_partition_plan_is_exact_for_non_divisible_input():
    sample_ids = tuple(f"sample-{index:02d}" for index in range(11))

    plan = build_synchronous_partition_plan(
        sample_ids,
        subset_digest_sha256="a" * 64,
        seed=7,
        epoch=3,
        world_size=3,
        batch_size=2,
        gradient_accumulation_steps=1,
    )

    assert plan.world_size == 3
    assert len(plan.rank_microbatches) == 3
    assert len({len(rows) for rows in plan.rank_microbatches}) == 1
    flattened = [
        sample_id
        for rank_rows in plan.rank_microbatches
        for microbatch in rank_rows
        for sample_id in microbatch
    ]
    assert len(flattened) == len(sample_ids)
    assert flattened == list(sample_ids)
    assert len(set(flattened)) == len(flattened)
    assert plan.epoch_plan_digest_sha256 == build_synchronous_partition_plan(
        sample_ids,
        subset_digest_sha256="a" * 64,
        seed=7,
        epoch=3,
        world_size=3,
        batch_size=2,
        gradient_accumulation_steps=1,
    ).epoch_plan_digest_sha256


@pytest.mark.parametrize("world_size,batch_size", [(4, 2), (2, 1)])
def test_synchronous_partition_plan_rejects_unsafe_distributed_shapes(
    world_size: int, batch_size: int
):
    with pytest.raises(RiskDataContractError):
        build_synchronous_partition_plan(
            ("one", "two", "three"),
            subset_digest_sha256="b" * 64,
            seed=1,
            epoch=0,
            world_size=world_size,
            batch_size=batch_size,
            gradient_accumulation_steps=1,
        )


def test_gloo_rank_zero_setup_broadcasts_one_bounded_descriptor(tmp_path):
    results = _run_gloo_setup(tmp_path, fail=False)

    assert [result[0] for result in results] == ["ok", "ok"]
    assert results[0][2] == results[1][2] == {
        "snapshot_digest_sha256": "a" * 64,
        "sample_count": 11,
    }
    assert [result[-1] for result in results] == [False, False]


def test_gloo_rank_zero_setup_failure_reaches_every_rank_and_destroys_group(tmp_path):
    results = _run_gloo_setup(tmp_path, fail=True)

    assert [result[0] for result in results] == ["error", "error"]
    assert [result[2] for result in results] == [
        "RiskDataContractError",
        "RiskDataContractError",
    ]
    assert results[0][3] == results[1][3]
    assert "fixture setup failed" in str(results[0][3])
    assert [result[-1] for result in results] == [False, False]


def test_gloo_ragged_accumulation_matches_synchronous_single_process_schedule(
    tmp_path,
):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    init_path = tmp_path / "gradient-init"
    processes = [
        context.Process(
            target=_gloo_gradient_worker,
            args=(rank, str(init_path), result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        distributed_weights = dict(result_queue.get(timeout=30) for _ in processes)
    except queue.Empty:
        pytest.fail("Gloo gradient workers did not report before timeout")
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert [process.exitcode for process in processes] == [0, 0]

    sample_ids = tuple(f"sample-{index:02d}" for index in range(19))
    plan = build_synchronous_partition_plan(
        sample_ids,
        subset_digest_sha256="c" * 64,
        seed=3,
        epoch=0,
        world_size=2,
        batch_size=3,
        gradient_accumulation_steps=2,
    )
    reference = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        reference.weight.zero_()
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
    for window_start in range(0, len(plan.rank_microbatches[0]), 2):
        window_ids = [
            sample_id
            for rank_batches in plan.rank_microbatches
            for batch_ids in rank_batches[window_start : window_start + 2]
            for sample_id in batch_ids
        ]
        indices = torch.tensor(
            [int(sample_id.split("-")[1]) + 1 for sample_id in window_ids],
            dtype=torch.float32,
        ).reshape(-1, 1)
        target = 2.0 * indices
        optimizer.zero_grad(set_to_none=True)
        torch.mean((reference(indices) - target) ** 2).backward()
        optimizer.step()
    expected = float(reference.weight.detach().item())

    assert distributed_weights[0] == pytest.approx(expected, abs=1e-6)
    assert distributed_weights[1] == pytest.approx(expected, abs=1e-6)
