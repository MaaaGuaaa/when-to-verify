from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from src.datasets.risk_dataloader import RiskDataContractError
from src.training.distributed import DistributedRuntime
from src.training.risk_trainer import ProductionRiskTrainingResult
from tests.test_risk_production_training import _publish_and_load


ROOT = Path(__file__).resolve().parents[1]
RISK_SCRIPT = ROOT / "scripts" / "06_train_risk_model.py"
RISK_CONFIG = ROOT / "configs" / "risk_model_production.yaml"
OCCUPANCY_SCRIPT = ROOT / "scripts" / "05_train_occupancy_baseline.py"
OCCUPANCY_CONFIG = ROOT / "configs" / "occupancy_baseline_production.yaml"


def _load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_risk_distributed_cli_initializes_before_snapshot_source_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, dataset = _publish_and_load(tmp_path / "source")
    module = _load_script(RISK_SCRIPT, "risk_distributed_cli_order")
    runtime = DistributedRuntime(0, 2, 0, "gloo", "cpu")
    events: list[str] = []
    original_loader = module.load_authenticated_risk_snapshot

    monkeypatch.setattr(
        module,
        "discover_distributed_runtime",
        lambda configured_device: runtime,
    )
    monkeypatch.setattr(
        module,
        "initialize_distributed_process_group",
        lambda discovered: events.append("initialize"),
    )
    monkeypatch.setattr(
        module,
        "destroy_distributed_process_group",
        lambda: events.append("destroy"),
    )
    monkeypatch.setattr(
        module,
        "broadcast_rank_zero_setup",
        lambda discovered, setup: setup(),
    )

    def observed_snapshot_loader(*args, **kwargs):
        events.append("source_snapshot")
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "load_authenticated_risk_snapshot",
        observed_snapshot_loader,
    )
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        module,
        "train_distributed_production_risk_model",
        lambda **kwargs: ProductionRiskTrainingResult(
            output_dir=output_dir,
            best_checkpoint=None,
            final_checkpoint=output_dir / "final_checkpoint.pt",
            training_state_checkpoint=output_dir / "training_state.pt",
            metrics_path=output_dir / "metrics.json",
            manifest_path=output_dir / "training_manifest.json",
            semantic_digest_sha256="b" * 64,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RISK_SCRIPT),
            "--config",
            str(RISK_CONFIG),
            "--output-dir",
            str(output_dir),
            "--train-seal-root",
            str(dataset.seal_root),
            "--train-collection-root",
            str(publication.collection_root),
            "--stage",
            "one_shard_smoke",
            "--max-samples",
            "11",
            "--batch-size",
            "3",
            "--device",
            "cpu",
            "--code-commit",
            "a" * 40,
            "--distributed",
            "--training-cache-mode",
            "authenticated_snapshot",
            "--training-cache-root",
            str(tmp_path / "cache"),
        ],
    )

    assert module.main() == 0
    assert events[0] == "initialize"
    assert events.index("initialize") < events.index("source_snapshot")
    assert events[-1] == "destroy"


def test_formal_distributed_cli_authenticates_typed_family_before_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(RISK_SCRIPT, "risk_distributed_cli_formal_gate")
    runtime = DistributedRuntime(0, 2, 0, "gloo", "cpu")
    events: list[str] = []
    family = SimpleNamespace(
        members={
            "train": {"risk_dataset_manifest_digest": "1" * 64},
            "val": {"risk_dataset_manifest_digest": "2" * 64},
        },
        cross_split_audit={"global_cross_split_leakage": "PROVEN"},
        risk_dataset_family_digest="3" * 64,
    )

    monkeypatch.setattr(
        module,
        "discover_distributed_runtime",
        lambda configured_device: runtime,
    )
    monkeypatch.setattr(
        module,
        "initialize_distributed_process_group",
        lambda discovered: events.append("initialize"),
    )
    monkeypatch.setattr(module, "destroy_distributed_process_group", lambda: None)
    monkeypatch.setattr(
        module,
        "broadcast_rank_zero_setup",
        lambda discovered, setup: setup(),
    )

    def load_family(path):
        events.append("typed_family")
        return family

    def stop_at_snapshot(*args, **kwargs):
        events.append("source_snapshot")
        raise RiskDataContractError("stop after formal gate")

    monkeypatch.setattr(module, "load_risk_dataset_family", load_family)
    monkeypatch.setattr(
        module,
        "load_authenticated_risk_snapshot",
        stop_at_snapshot,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RISK_SCRIPT),
            "--config",
            str(RISK_CONFIG),
            "--output-dir",
            str(tmp_path / "output"),
            "--train-seal-root",
            str(tmp_path / "train-seal"),
            "--train-collection-root",
            str(tmp_path / "train-collection"),
            "--validation-seal-root",
            str(tmp_path / "val-seal"),
            "--validation-collection-root",
            str(tmp_path / "val-collection"),
            "--dataset-family-root",
            str(tmp_path / "typed-family"),
            "--stage",
            "formal_50k",
            "--device",
            "cpu",
            "--code-commit",
            "a" * 40,
            "--distributed",
            "--training-cache-mode",
            "authenticated_snapshot",
            "--training-cache-root",
            str(tmp_path / "cache"),
        ],
    )

    with pytest.raises(SystemExit):
        module.main()
    assert events[:3] == ["initialize", "typed_family", "source_snapshot"]


def test_occupancy_cli_rejects_world_size_before_source_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(OCCUPANCY_SCRIPT, "occupancy_distributed_rejection")
    source_loaded = False

    monkeypatch.setattr(
        module,
        "discover_distributed_runtime",
        lambda configured_device: DistributedRuntime(0, 2, 0, "gloo", "cpu"),
    )

    def forbidden_source_load(*args, **kwargs):
        nonlocal source_loaded
        source_loaded = True
        raise AssertionError("occupancy WORLD_SIZE gate must precede source load")

    monkeypatch.setattr(module, "load_risk_dataset_seal", forbidden_source_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(OCCUPANCY_SCRIPT),
            "--config",
            str(OCCUPANCY_CONFIG),
            "--mode",
            "production",
            "--output-dir",
            str(tmp_path / "output"),
            "--dataset-seal-root",
            str(tmp_path / "seal"),
            "--risk-collection-root",
            str(tmp_path / "collection"),
            "--sidecar-collection-root",
            str(tmp_path / "sidecars"),
            "--device",
            "cpu",
        ],
    )

    assert module.main() == 2
    assert source_loaded is False
