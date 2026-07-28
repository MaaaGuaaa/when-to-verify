"""Focused R2 trajectory-query Transformer contract tests."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

from src.contracts import SCHEMA_VERSION
from src.datasets.risk_dataloader import RiskDataContractError
from src.datasets.risk_dataloader import ProductionOccupancyBatch
from src.datasets.risk_dataset_seal import (
    load_risk_dataset_seal,
    publish_risk_dataset_seal,
)
from src.datasets.toy_risk_learning import frozen_channel_spec
from src.models.risk_model import (
    R2_D_MODEL,
    R2_MAX_SCENE_TOKENS,
    R2_OCCUPANCY_AUX_CHANNELS,
    R2_QUERY_BINS,
    R2_SCENE_INPUT_CHANNELS,
    RiskModel,
    compute_risk_batch_loss,
    load_risk_checkpoint,
    save_risk_checkpoint,
)
from src.training.risk_trainer import (
    ProductionRiskTrainingConfig,
    train_production_risk_model,
)


ROOT = Path(__file__).resolve().parents[1]


def _inputs(batch_size: int, height: int, width: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260727 + height + width + batch_size)
    return {
        "bev_history": torch.rand(
            (batch_size, 8, 2, height, width), generator=generator
        ),
        "state_channels": torch.rand(
            (batch_size, 9, height, width), generator=generator
        ),
        "trajectory_channels": torch.rand(
            (batch_size, 4, height, width), generator=generator
        ),
        "robot_state": torch.rand((batch_size, 2), generator=generator),
    }


def _toy_provenance(variant: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "channel_spec": frozen_channel_spec(),
        "model_variant": variant,
        "config_digest": "a" * 32,
        "toy_dataset_manifest_digest": "b" * 32,
        "validation_dataset_manifest_digest": "c" * 32,
        "seed": 17,
    }


def _load_training_entry_module():
    path = ROOT / "scripts" / "06_train_risk_model.py"
    specification = importlib.util.spec_from_file_location("r2_training_entry", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_r2_uses_bounded_scene_tokens_and_preserves_risk_output_contract():
    model = RiskModel(variant="r2").eval()
    backbone = model.r2_model
    assert backbone.scene_stem[0].in_channels == R2_SCENE_INPUT_CHANNELS == 25
    downsampling_convs = [
        layer
        for layer in (
            *backbone.scene_down_80,
            *backbone.scene_down_40,
            *backbone.scene_down_20,
        )
        if isinstance(layer, nn.Conv2d)
    ]
    assert len(downsampling_convs) == 3
    assert all(layer.stride == (2, 2) for layer in downsampling_convs)
    assert len(backbone.trajectory_decoder.layers) == 2
    for layer in backbone.trajectory_decoder.layers:
        assert layer.norm_first is True
        assert layer.self_attn.batch_first is True
        assert layer.self_attn.embed_dim == R2_D_MODEL
        assert layer.self_attn.num_heads == 4
        assert layer.linear1.out_features == 256
        assert layer.dropout.p == pytest.approx(0.1)

    observed: dict[str, tuple[int, ...]] = {}

    def capture_decoder_tokens(_module, arguments):
        observed["query"] = tuple(arguments[0].shape)
        observed["scene"] = tuple(arguments[1].shape)

    handle = backbone.trajectory_decoder.register_forward_pre_hook(
        capture_decoder_tokens
    )
    try:
        output = model(_inputs(batch_size=1, height=160, width=160))
    finally:
        handle.remove()

    assert observed == {"query": (1, R2_QUERY_BINS, 128), "scene": (1, 400, 128)}
    assert observed["scene"][1] <= R2_MAX_SCENE_TOKENS
    assert observed["query"][1] == R2_QUERY_BINS
    assert set(output) == {"quantiles", "collision_logits", "p_collision"}
    assert output["quantiles"].shape == (1, 4)
    assert output["collision_logits"].shape == (1,)
    assert output["p_collision"].shape == (1,)
    assert all(value.dtype == torch.float32 for value in output.values())
    assert all(torch.isfinite(value).all() for value in output.values())
    assert torch.all(output["quantiles"][:, 1:] >= output["quantiles"][:, :-1])
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_500_000


def test_r2_backward_has_finite_trajectory_gradients_and_is_trajectory_sensitive():
    model = RiskModel(variant="r2").eval()
    inputs = _inputs(batch_size=2, height=32, width=32)
    trajectory = inputs["trajectory_channels"].clone().requires_grad_(True)
    output = model({**inputs, "trajectory_channels": trajectory})
    (output["quantiles"].sum() + output["collision_logits"].sum()).backward()

    assert trajectory.grad is not None
    assert torch.isfinite(trajectory.grad).all()
    assert float(trajectory.grad.abs().sum()) > 0.0
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )

    with torch.no_grad():
        counterfactual = model(
            {**inputs, "trajectory_channels": torch.zeros_like(trajectory)}
        )
    delta = (
        torch.mean(
            torch.abs(output["quantiles"].detach() - counterfactual["quantiles"])
        )
        + torch.mean(
            torch.abs(
                output["collision_logits"].detach()
                - counterfactual["collision_logits"]
            )
        )
    )
    assert torch.isfinite(delta)
    assert float(delta) > 1e-8


def test_r2_query_bins_cover_the_full_long40_horizon():
    model = RiskModel(
        variant="r2",
        r2_d_model=32,
        r2_nhead=4,
        r2_num_decoder_layers=1,
        r2_dim_feedforward=64,
        r2_query_bins=8,
    )
    backbone = model.r2_model
    trajectory_features = torch.zeros((1, 32, 4, 4), dtype=torch.float32)
    trajectory_channels = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
    for row, column, arrival_s in ((0, 0, 0.2), (1, 1, 3.2), (3, 3, 6.4)):
        trajectory_channels[0, 0, row, column] = 1.0
        trajectory_channels[0, 1, row, column] = arrival_s

    _, valid = backbone._trajectory_queries(
        trajectory_features,
        trajectory_channels,
    )

    assert valid[0, 0]
    assert valid[0, 4]
    assert valid[0, 7]


def test_r2_auxiliary_branch_is_trajectory_invariant_and_uses_existing_risk_loss():
    model = RiskModel(variant="r2", occupancy_aux_enabled=True).eval()
    inputs = _inputs(batch_size=2, height=32, width=32)
    with torch.no_grad():
        reference = model(inputs)
        changed_query = model(
            {
                **inputs,
                "trajectory_channels": torch.zeros_like(
                    inputs["trajectory_channels"]
                ),
            }
        )

    auxiliary = reference["occupancy_aux_logits"]
    assert auxiliary.shape == (2, R2_OCCUPANCY_AUX_CHANNELS, 32, 32)
    assert auxiliary.dtype == torch.float32
    assert torch.isfinite(auxiliary).all()
    assert torch.equal(auxiliary, changed_query["occupancy_aux_logits"])
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_500_000

    batch = SimpleNamespace(
        model_inputs=inputs,
        targets={
            "risk_severity": torch.zeros(2, dtype=torch.float32),
            "collision_label": torch.zeros(2, dtype=torch.float32),
        },
        occupancy_targets={"hidden_risk_occupancy": torch.zeros_like(auxiliary)},
    )
    _, losses = compute_risk_batch_loss(
        model,
        batch,
        lambda_collision=1.0,
        lambda_occupancy_aux=0.2,
    )
    assert all(torch.isfinite(value).all() for value in losses.values())
    assert float(losses["occupancy_aux"]) > 0.0


def test_r2_binds_explicit_capacity_query_and_auxiliary_horizon_configuration():
    model = RiskModel(
        variant="r2",
        r2_d_model=64,
        r2_nhead=4,
        r2_num_decoder_layers=1,
        r2_dim_feedforward=128,
        r2_query_bins=8,
        occupancy_aux_enabled=True,
        occupancy_future_steps=15,
    ).eval()
    backbone = model.r2_model
    assert backbone.d_model == 64
    assert backbone.query_bins == 8
    assert backbone.occupancy_future_steps == 15
    assert backbone.fusion[0].in_features == 64 + 2

    observed: dict[str, tuple[int, ...]] = {}

    def capture_decoder_tokens(_module, arguments):
        observed["query"] = tuple(arguments[0].shape)

    handle = backbone.trajectory_decoder.register_forward_pre_hook(
        capture_decoder_tokens
    )
    try:
        output = model(_inputs(batch_size=2, height=32, width=32))
    finally:
        handle.remove()

    assert observed["query"] == (2, 8, 64)
    assert output["occupancy_aux_logits"].shape == (2, 15, 32, 32)
    assert model.export_config()["occupancy_future_steps"] == 15
    assert model.export_config()["r2_query_bins"] == 8


@pytest.mark.parametrize("variant", ("r0", "r1"))
def test_r0_r1_legacy_configs_and_checkpoints_remain_round_trippable(
    tmp_path, variant
):
    model = RiskModel(variant=variant, hidden_channels=4).eval()
    assert model.export_config() == {
        "variant": variant,
        "hidden_channels": 4,
        "history_steps": 8,
    }
    path = tmp_path / f"{variant}.pt"
    provenance = _toy_provenance(variant)
    save_risk_checkpoint(path, model=model, mode="toy", provenance=provenance)
    loaded, payload = load_risk_checkpoint(
        path, expected_mode="toy", expected_provenance=provenance
    )
    assert payload["model_config"] == model.export_config()
    inputs = _inputs(batch_size=2, height=16, width=16)
    with torch.no_grad():
        expected = model(inputs)
        actual = loaded(inputs)
    assert set(actual) == {"quantiles", "collision_logits", "p_collision"}
    assert all(torch.equal(expected[key], actual[key]) for key in expected)


def test_r2_checkpoint_round_trip_and_production_config_are_bound(tmp_path):
    model = RiskModel(variant="r2").eval()
    path = tmp_path / "r2.pt"
    provenance = _toy_provenance("r2")
    save_risk_checkpoint(path, model=model, mode="toy", provenance=provenance)
    loaded, payload = load_risk_checkpoint(
        path, expected_mode="toy", expected_provenance=provenance
    )
    assert payload["model_config"] == model.export_config()
    inputs = _inputs(batch_size=2, height=32, width=32)
    with torch.no_grad():
        expected = model(inputs)
        actual = loaded(inputs)
    assert all(torch.equal(expected[key], actual[key]) for key in expected)

    config = ProductionRiskTrainingConfig(
        stage="one_shard_smoke",
        variant="r2",
        seed=17,
        device="cpu",
        hidden_channels=None,
        batch_size=2,
        epochs=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        lambda_collision=1.0,
        checkpoint_interval_steps=1,
    )
    assert config.occupancy_aux_enabled is False
    assert config.lambda_occupancy_aux == 0.0
    assert RiskModel(variant="r2", r2_d_model=64, r2_nhead=4).r2_model.d_model == 64
    with pytest.raises(RiskDataContractError, match="occupancy_aux_enabled"):
        replace(
            config,
            variant="r0",
            hidden_channels=4,
            occupancy_aux_enabled=True,
            lambda_occupancy_aux=0.2,
        )


def test_r2_auxiliary_loss_receives_the_train_derived_positive_weight():
    model = RiskModel(
        variant="r2",
        r2_d_model=64,
        r2_nhead=4,
        r2_num_decoder_layers=1,
        r2_dim_feedforward=128,
        occupancy_aux_enabled=True,
        occupancy_future_steps=15,
    ).eval()
    inputs = _inputs(batch_size=2, height=32, width=32)
    target = torch.zeros((2, 15, 32, 32), dtype=torch.float32)
    target[0, 0, 0, 0] = 1.0
    batch = SimpleNamespace(
        model_inputs=inputs,
        targets={
            "risk_severity": torch.zeros(2, dtype=torch.float32),
            "collision_label": torch.zeros(2, dtype=torch.float32),
        },
        occupancy_targets={"hidden_risk_occupancy": target},
    )

    _, losses = compute_risk_batch_loss(
        model,
        batch,
        lambda_collision=1.0,
        lambda_occupancy_aux=0.2,
        occupancy_pos_weight=3.0,
    )

    assert float(losses["occupancy_aux"]) > 0.0


def test_r2_auxiliary_production_checkpoint_requires_bound_sidecar_provenance(
    tmp_path,
):
    from tests.test_risk_production_training import (
        _production_provenance,
        _publish_and_load,
    )

    _, dataset = _publish_and_load(tmp_path / "source")
    provenance = _production_provenance(dataset)
    provenance["model_variant"] = "r2"
    model = RiskModel(
        variant="r2",
        r2_d_model=64,
        r2_nhead=4,
        r2_num_decoder_layers=1,
        r2_dim_feedforward=128,
        occupancy_aux_enabled=True,
        occupancy_future_steps=32,
    )

    with pytest.raises(RiskDataContractError, match="auxiliary provenance"):
        save_risk_checkpoint(
            tmp_path / "missing-auxiliary-provenance.pt",
            model=model,
            mode="production",
            provenance=provenance,
        )


def test_occupancy_batch_uses_its_declared_32_step_time_grid():
    inputs = _inputs(batch_size=2, height=16, width=16)
    horizon = 32
    masks = torch.zeros((2, horizon, 16, 16), dtype=torch.float32)
    batch = ProductionOccupancyBatch(
        model_inputs=inputs,
        targets={
            "collision_label": torch.zeros(2, dtype=torch.float32),
            "risk_severity": torch.zeros(2, dtype=torch.float32),
            "min_clearance": torch.ones(2, dtype=torch.float32),
            "near_miss": torch.zeros(2, dtype=torch.float32),
        },
        query_inputs={
            "robot_endpoint_footprints": masks,
            "endpoint_times_s": torch.arange(1, horizon + 1, dtype=torch.float32)
            * 0.2,
        },
        occupancy_targets={"hidden_risk_occupancy": masks.clone()},
        sample_ids=("r2-occupancy-0", "r2-occupancy-1"),
        split="train",
        provenance={},
    )

    assert batch.occupancy_targets["hidden_risk_occupancy"].shape[1] == horizon


def test_r2_configs_and_cli_variant_are_accepted_by_the_training_entry():
    training_entry = _load_training_entry_module()
    toy_config = training_entry._load_config(ROOT / "configs" / "risk_model_r2.yaml")
    production_config = training_entry._load_production_config(
        ROOT / "configs" / "risk_model_r2_production.yaml"
    )
    assert toy_config["variants"] == ["r0", "r1", "r2"]
    assert production_config["variant"] == "r2"
    assert production_config["r2_fusion_mode"] == "cross_attention"
    assert production_config["occupancy_aux_enabled"] is True
    assert production_config["lambda_occupancy_aux"] == pytest.approx(0.2)
    no_aux_config = training_entry._load_production_config(
        ROOT / "configs" / "risk_model_r2_no_aux_production.yaml"
    )
    concat_config = training_entry._load_production_config(
        ROOT / "configs" / "risk_model_r2_concat_control_production.yaml"
    )
    r0_config = training_entry._load_production_config(
        ROOT / "configs" / "risk_model_r0_production.yaml"
    )
    r1_config = training_entry._load_production_config(
        ROOT / "configs" / "risk_model_r1_production.yaml"
    )
    assert no_aux_config["occupancy_aux_enabled"] is False
    assert no_aux_config["lambda_occupancy_aux"] == 0.0
    assert r0_config["variant"] == "r0"
    assert r1_config["variant"] == "r1"
    assert concat_config["r2_fusion_mode"] == "concat"
    assert concat_config["occupancy_aux_enabled"] is True
    matrix = yaml.safe_load(
        (ROOT / "configs" / "risk_experiment_matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert matrix["layout_version"] == "risk_experiment_matrix_v1"
    assert matrix["schema_version"] == SCHEMA_VERSION
    assert matrix["seeds"] == [42, 43, 44]
    assert matrix["prediction_protocol"] == (
        "configs/prediction_protocol_production.json"
    )
    assert matrix["stage_sample_limits"] == {
        "one_shard_smoke": "one_authenticated_shard",
        "real_1k_overfit": 1000,
        "formal_50k": 50_000,
    }
    assert matrix["formal_methods"] == [
        "risk-r0",
        "risk-r1",
        "risk-r2",
        "B1",
        "B2",
        "B3",
        "B4",
    ]
    expected_risk_experiments = {
        "R0": ("r0", None, False, "risk-r0"),
        "R1": ("r1", None, False, "risk-r1"),
        "R2": ("r2", "cross_attention", True, "risk-r2"),
        "R2-no-aux": ("r2", "cross_attention", False, None),
        "R2-concat": ("r2", "concat", True, None),
    }
    assert [item["name"] for item in matrix["risk_experiments"]] == list(
        expected_risk_experiments
    )
    for experiment in matrix["risk_experiments"]:
        variant, fusion, auxiliary, method_id = expected_risk_experiments[
            experiment["name"]
        ]
        loaded = training_entry._load_production_config(
            ROOT / experiment["config"]
        )
        assert experiment["formal_method_id"] == method_id
        assert loaded["variant"] == variant
        assert loaded["occupancy_aux_enabled"] is auxiliary
        if fusion is not None:
            assert loaded["r2_fusion_mode"] == fusion
    assert matrix["occupancy_experiments"] == [
        {
            "name": "B1",
            "role": "last_observation_hold_hand_aggregation",
            "config": "configs/occupancy_baseline_production.yaml",
            "formal_method_id": "B1",
        },
        {
            "name": "B2",
            "role": "age_decay_hand_aggregation",
            "config": "configs/occupancy_baseline_production.yaml",
            "formal_method_id": "B2",
        },
        {
            "name": "B3",
            "role": "convgru_occupancy_hand_aggregation",
            "config": "configs/occupancy_baseline_production.yaml",
            "formal_method_id": "B3",
        },
        {
            "name": "B4",
            "role": "convgru_occupancy_learned_aggregation",
            "config": "configs/occupancy_baseline_production.yaml",
            "formal_method_id": "B4",
        },
    ]
    assert not (ROOT / "configs" / "risk_model_r2_aux_production.yaml").exists()
    parsed = training_entry._parser().parse_args(
        [
            "--output-dir",
            "r2-training-output",
            "--variant",
            "r2",
            "--train-occupancy-sidecar-root",
            "r2-sidecars",
        ]
    )
    assert parsed.variant == "r2"
    assert parsed.train_occupancy_sidecar_root == Path("r2-sidecars")
    seeded = training_entry._parser().parse_args(
        [
            "--config",
            str(ROOT / "configs" / "risk_model_r2_production.yaml"),
            "--output-dir",
            "r2-seed-43-output",
            "--seed",
            "43",
        ]
    )
    assert training_entry._effective_config(seeded)["seed"] == 43


def test_r2_concat_control_uses_the_same_auxiliary_target_contract():
    model = RiskModel(
        variant="r2",
        r2_d_model=64,
        r2_nhead=4,
        r2_num_decoder_layers=1,
        r2_dim_feedforward=128,
        r2_fusion_mode="concat",
        occupancy_aux_enabled=True,
        occupancy_future_steps=32,
    ).eval()
    backbone = model.r2_model
    assert backbone.trajectory_decoder is None
    assert backbone.trajectory_concat_encoder is not None
    assert backbone.fusion[0].in_features == 2 * 64 + 2
    output = model(_inputs(batch_size=2, height=32, width=32))
    assert output["occupancy_aux_logits"].shape == (2, 32, 32, 32)
    assert model.export_config()["r2_fusion_mode"] == "concat"


def test_r2_one_shard_production_training_writes_a_reloadable_checkpoint(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires the Slurm GPU allocation used by this test suite")
    from src.datasets.risk_dataloader import select_production_risk_subset
    from tests.test_risk_production_training import CODE_COMMIT, _publish_and_load

    _, dataset = _publish_and_load(tmp_path / "source")
    subset = select_production_risk_subset(dataset, max_samples=12, seed=17)
    config = ProductionRiskTrainingConfig(
        stage="one_shard_smoke",
        variant="r2",
        seed=17,
        device="cuda",
        hidden_channels=None,
        batch_size=6,
        epochs=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        lambda_collision=1.0,
        checkpoint_interval_steps=1,
    )
    result = train_production_risk_model(
        train_dataset=dataset,
        train_subset=subset,
        config=config,
        output_dir=tmp_path / "r2-output",
        code_commit=CODE_COMMIT,
    )
    loaded, payload = load_risk_checkpoint(
        result.final_checkpoint, expected_mode="production"
    )
    assert payload["provenance"]["model_variant"] == "r2"
    assert payload["model_config"] == loaded.export_config()


def test_r2_auxiliary_training_requires_sidecars_and_records_train_statistics(
    tmp_path,
):
    if not torch.cuda.is_available():
        pytest.skip("requires the Slurm GPU allocation used by this test suite")
    from tests.fixtures.formal_risk_publication import (
        create_formal_risk_sidecar_publication,
    )
    from tests.test_risk_production_training import (
        CODE_COMMIT,
        _model_compatible_publication,
    )

    publication = _model_compatible_publication(tmp_path / "upstream", split="train")
    sidecars = create_formal_risk_sidecar_publication(
        publication, tmp_path / "sidecars"
    )
    seal_root = publish_risk_dataset_seal(
        tmp_path / "seal",
        collection_root=publication.collection_root,
        base_config_path=publication.base_config_path,
        split_provenance_path=publication.split_provenance_path,
        expected_split="train",
        expected_collection_handoff_sha256=publication.handoff_sha256,
        sidecar_root=sidecars.sidecar_root,
    )
    dataset = load_risk_dataset_seal(
        seal_root,
        collection_root=publication.collection_root,
        expected_split="train",
        sidecar_root=sidecars.sidecar_root,
    )
    from src.datasets.risk_dataloader import select_production_risk_subset

    subset = select_production_risk_subset(dataset, max_samples=12, seed=17)
    config = ProductionRiskTrainingConfig(
        stage="one_shard_smoke",
        variant="r2",
        seed=17,
        device="cuda",
        hidden_channels=None,
        batch_size=6,
        epochs=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        lambda_collision=1.0,
        checkpoint_interval_steps=1,
        occupancy_aux_enabled=True,
        lambda_occupancy_aux=0.2,
        occupancy_future_steps=32,
        r2_d_model=64,
        r2_nhead=4,
        r2_num_decoder_layers=1,
        r2_dim_feedforward=128,
    )
    with pytest.raises(RiskDataContractError, match="occupancy sidecar root"):
        train_production_risk_model(
            train_dataset=dataset,
            train_subset=subset,
            config=config,
            output_dir=tmp_path / "missing-sidecar-output",
            code_commit=CODE_COMMIT,
        )

    result = train_production_risk_model(
        train_dataset=dataset,
        train_subset=subset,
        config=config,
        output_dir=tmp_path / "aux-output",
        code_commit=CODE_COMMIT,
        train_occupancy_sidecar_root=sidecars.sidecar_root,
    )
    _, payload = load_risk_checkpoint(result.final_checkpoint, expected_mode="production")
    metrics = __import__("json").loads(result.metrics_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["occupancy_sidecar_collection_digest_sha256"] == (
        dataset.manifest["occupancy_sidecars"]["collection_digest_sha256"]
    )
    assert metrics["occupancy_global_positive_count"] > 0
    assert metrics["occupancy_global_pos_weight"] > 0.0
    assert payload["model_config"]["occupancy_aux_enabled"] is True
