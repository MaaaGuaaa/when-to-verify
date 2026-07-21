from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import queue

import pytest
import torch

from src.datasets.risk_training_store import (
    build_authenticated_risk_training_view,
    open_authenticated_risk_snapshot,
    open_authenticated_risk_snapshot_descriptor,
)
from src.models.risk_model import load_risk_checkpoint
from src.training.distributed import (
    DistributedRuntime,
    destroy_distributed_process_group,
    initialize_distributed_process_group,
)
from src.training.risk_ddp_trainer import (
    DISTRIBUTED_RISK_TRAINING_STATE_LAYOUT_VERSION,
    load_distributed_risk_training_state,
    train_distributed_production_risk_model,
    validate_distributed_resume_bindings,
)
from src.training.risk_trainer import ProductionRiskTrainingConfig
from tests.test_risk_production_training import _publish_and_load


def _distributed_training_worker(
    rank: int,
    init_path: str,
    descriptor: dict[str, object],
    config_values: dict[str, object],
    max_samples: int,
    output_dir: str,
    resume_from: str | None,
    resume_publication_digest: str | None,
    result_queue,
) -> None:
    torch.set_num_threads(1)
    runtime = DistributedRuntime(rank, 2, rank, "gloo", "cpu")
    status: tuple[object, ...]
    try:
        initialize_distributed_process_group(
            runtime,
            init_method=f"file://{init_path}",
        )
        snapshot = open_authenticated_risk_snapshot_descriptor(descriptor)
        config = ProductionRiskTrainingConfig(**config_values)
        subset = snapshot.select_subset(max_samples=max_samples, seed=config.seed)
        view = build_authenticated_risk_training_view(
            snapshot,
            subset=subset,
            split_role="train",
            world_size=runtime.world_size,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        result = train_distributed_production_risk_model(
            train_view=view,
            config=config,
            output_dir=output_dir,
            code_commit="a" * 40,
            runtime=runtime,
            resume_from=resume_from,
            resume_expected_publication_instance_digest_sha256=(
                resume_publication_digest
            ),
        )
        status = (
            "ok",
            rank,
            str(result.output_dir),
            result.semantic_digest_sha256,
        )
    except Exception as exc:
        status = ("error", rank, type(exc).__name__, str(exc))
    finally:
        destroy_distributed_process_group()
        result_queue.put(status)


def _run_distributed_training(
    tmp_path: Path,
    *,
    descriptor: dict[str, object],
    config: ProductionRiskTrainingConfig,
    max_samples: int,
    output_dir: Path,
    run_name: str,
    resume_from: Path | None = None,
    resume_publication_digest: str | None = None,
) -> list[tuple[object, ...]]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    init_path = tmp_path / f"{run_name}-init"
    processes = [
        context.Process(
            target=_distributed_training_worker,
            args=(
                rank,
                str(init_path),
                descriptor,
                config.__dict__,
                max_samples,
                str(output_dir),
                None if resume_from is None else str(resume_from),
                resume_publication_digest,
                result_queue,
            ),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    results: list[tuple[object, ...]] = []
    try:
        for _ in processes:
            results.append(result_queue.get(timeout=120))
    except queue.Empty:
        pytest.fail("distributed risk trainer workers did not finish before timeout")
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert [process.exitcode for process in processes] == [0, 0]
    return sorted(results, key=lambda item: int(item[1]))


def _config(*, stage: str, epochs: int) -> ProductionRiskTrainingConfig:
    return ProductionRiskTrainingConfig(
        stage=stage,
        variant="r0",
        seed=31,
        device="cpu",
        hidden_channels=2,
        batch_size=3,
        epochs=epochs,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        lambda_collision=1.0,
        checkpoint_interval_steps=1,
    )


def test_gloo_trainer_publishes_exact_ragged_smoke_union(tmp_path: Path) -> None:
    _, dataset = _publish_and_load(tmp_path / "source")
    snapshot = open_authenticated_risk_snapshot(
        dataset,
        cache_root=tmp_path / "cache",
    )
    config = _config(stage="one_shard_smoke", epochs=1)
    subset = snapshot.select_subset(max_samples=11, seed=config.seed)
    view = build_authenticated_risk_training_view(
        snapshot,
        subset=subset,
        split_role="train",
        world_size=2,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    plan = view.partition(epoch=0)
    expected_ids = {
        sample_id
        for rank_batches in plan.rank_microbatches
        for sample_id in rank_batches[0]
    }
    output_dir = tmp_path / "smoke-output"

    results = _run_distributed_training(
        tmp_path,
        descriptor=snapshot.descriptor(),
        config=config,
        max_samples=11,
        output_dir=output_dir,
        run_name="smoke",
    )

    assert [result[0] for result in results] == ["ok", "ok"], results
    assert results[0][2:] == results[1][2:]
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output_dir / "training_manifest.json").read_text(encoding="utf-8")
    )
    assert metrics["world_size"] == 2
    assert metrics["backend"] == "gloo"
    assert metrics["optimizer_steps"] == 1
    assert metrics["selected_sample_count"] == 11
    assert metrics["consumed_sample_count"] == len(expected_ids)
    assert metrics["smoke_consumed_sample_ids"] == sorted(expected_ids)
    assert metrics["rank_local_microbatch_sizes"] == [
        list(sizes) for sizes in plan.local_microbatch_sizes
    ]
    assert manifest["writer_rank"] == 0
    assert manifest["distributed_identity"]["train_snapshot_digest_sha256"] == (
        snapshot.snapshot_digest_sha256
    )
    assert manifest["distributed_identity"]["train_view_digest_sha256"] == (
        view.training_view_digest_sha256
    )
    state = load_distributed_risk_training_state(
        output_dir / "training_state.pt"
    )
    assert state["training_state_layout_version"] == (
        DISTRIBUTED_RISK_TRAINING_STATE_LAYOUT_VERSION
    )
    assert len(state["per_rank_state"]) == 2
    _, checkpoint = load_risk_checkpoint(
        output_dir / "final_checkpoint.pt",
        expected_mode="production",
        expected_provenance=manifest["checkpoint_provenance"],
    )
    assert checkpoint["provenance"]["consumed_sample_count"] == len(expected_ids)


def test_gloo_resume_epoch_extension_matches_uninterrupted_training(
    tmp_path: Path,
) -> None:
    _, dataset = _publish_and_load(tmp_path / "source")
    snapshot = open_authenticated_risk_snapshot(
        dataset,
        cache_root=tmp_path / "cache",
    )
    descriptor = snapshot.descriptor()
    first_output = tmp_path / "first-epoch"
    first_results = _run_distributed_training(
        tmp_path,
        descriptor=descriptor,
        config=_config(stage="real_1k_overfit", epochs=1),
        max_samples=dataset.sample_count,
        output_dir=first_output,
        run_name="first",
    )
    assert [result[0] for result in first_results] == ["ok", "ok"], first_results
    first_manifest = json.loads(
        (first_output / "training_manifest.json").read_text(encoding="utf-8")
    )

    resumed_output = tmp_path / "resumed"
    resumed_results = _run_distributed_training(
        tmp_path,
        descriptor=descriptor,
        config=_config(stage="real_1k_overfit", epochs=2),
        max_samples=dataset.sample_count,
        output_dir=resumed_output,
        run_name="resumed",
        resume_from=first_output / "training_state.pt",
        resume_publication_digest=first_manifest[
            "publication_instance_digest_sha256"
        ],
    )
    assert [result[0] for result in resumed_results] == [
        "ok",
        "ok",
    ], resumed_results

    uninterrupted_output = tmp_path / "uninterrupted"
    uninterrupted_results = _run_distributed_training(
        tmp_path,
        descriptor=descriptor,
        config=_config(stage="real_1k_overfit", epochs=2),
        max_samples=dataset.sample_count,
        output_dir=uninterrupted_output,
        run_name="uninterrupted",
    )
    assert [result[0] for result in uninterrupted_results] == [
        "ok",
        "ok",
    ], uninterrupted_results

    _, resumed_checkpoint = load_risk_checkpoint(
        resumed_output / "final_checkpoint.pt",
        expected_mode="production",
    )
    _, uninterrupted_checkpoint = load_risk_checkpoint(
        uninterrupted_output / "final_checkpoint.pt",
        expected_mode="production",
    )
    assert resumed_checkpoint["model_state_digest_sha256"] == (
        uninterrupted_checkpoint["model_state_digest_sha256"]
    )
    resumed_state = load_distributed_risk_training_state(
        resumed_output / "training_state.pt"
    )
    uninterrupted_state = load_distributed_risk_training_state(
        uninterrupted_output / "training_state.pt"
    )
    assert resumed_state["optimizer_state_digest_sha256"] == (
        uninterrupted_state["optimizer_state_digest_sha256"]
    )
    assert resumed_state["optimizer_step_loss_history"] == pytest.approx(
        uninterrupted_state["optimizer_step_loss_history"],
        rel=0.0,
        abs=1e-10,
    )

    with pytest.raises(ValueError, match="world_size"):
        validate_distributed_resume_bindings(
            resumed_state,
            expected_world_size=3,
            expected_backend="gloo",
            expected_train_snapshot_digest_sha256=(
                snapshot.snapshot_digest_sha256
            ),
            expected_train_view_digest_sha256=(
                resumed_state["distributed_identity"][
                    "train_view_digest_sha256"
                ]
            ),
            expected_partition_spec_digest_sha256=(
                resumed_state["distributed_identity"][
                    "partition_spec_digest_sha256"
                ]
            ),
            expected_epoch_plan_digest_sha256=(
                resumed_state["distributed_identity"][
                    "current_epoch_plan_digest_sha256"
                ]
            ),
        )
