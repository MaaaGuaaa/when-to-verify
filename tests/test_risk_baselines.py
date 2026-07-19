from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from src.contracts import HISTORY_CHANNELS, INPUT_CHANNELS, SCHEMA_VERSION, STATE_CHANNELS
from src.datasets.toy_risk_learning import make_toy_risk_dataset
from src.evaluation import risk_baselines as risk_baselines_module
from src.evaluation.risk_baselines import (
    BASELINE_SPECS,
    ProductionOccupancyContractUnavailable,
    build_occupancy_checkpoint,
    load_occupancy_checkpoint,
    occupancy_checkpoint_semantic_digest,
    occupancy_binary_metrics,
    save_occupancy_checkpoint,
    validate_occupancy_checkpoint_provenance,
    validate_occupancy_dataset_manifest,
)
from src.models.occupancy_baseline import (
    AgeDecay,
    ConvGRUOccupancyPredictor,
    LastObservationHold,
    LearnedOccupancyRiskAggregator,
)


def test_last_observation_hold_uses_frozen_dynamic_history_channel() -> None:
    history = np.zeros((2, 8, len(HISTORY_CHANNELS), 3, 4), dtype=np.float32)
    dynamic_index = HISTORY_CHANNELS.index("past_dynamic_occupancy")
    visible_index = HISTORY_CHANNELS.index("past_visible_mask")
    history[:, -1, dynamic_index] = 0.25
    history[:, -1, visible_index] = 0.99

    prediction = LastObservationHold(future_steps=15)(history)

    assert prediction.shape == (2, 15, 3, 4)
    assert prediction.dtype == np.float32
    np.testing.assert_array_equal(prediction, np.full_like(prediction, 0.25))


@pytest.mark.parametrize("history_steps", [7, 9])
def test_last_observation_hold_requires_exactly_eight_history_frames(
    history_steps: int,
) -> None:
    history = np.zeros(
        (1, history_steps, len(HISTORY_CHANNELS), 3, 4),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="exactly 8 history frames"):
        LastObservationHold(future_steps=15)(history)


def test_age_decay_uses_last_seen_occupancy_and_endpoint_time() -> None:
    state = np.zeros((1, len(STATE_CHANNELS), 2, 2), dtype=np.float32)
    state[:, STATE_CHANNELS.index("last_seen_occupancy")] = 0.8
    state[:, STATE_CHANNELS.index("occlusion_age_map")] = 0.5

    prediction = AgeDecay(
        future_steps=2,
        dt_s=0.2,
        tau_s=2.0,
        a_max_s=5.0,
    )(state)

    expected = np.array(
        [0.8 * np.exp(-(2.5 + 0.2) / 2.0), 0.8 * np.exp(-(2.5 + 0.4) / 2.0)],
        dtype=np.float32,
    )
    assert prediction.shape == (1, 2, 2, 2)
    np.testing.assert_allclose(prediction[0, :, 0, 0], expected, rtol=1e-6)
    assert np.isfinite(prediction).all()
    assert np.logical_and(prediction >= 0.0, prediction <= 1.0).all()


def test_age_decay_allows_signed_nonprobability_state_channels() -> None:
    state = np.zeros((1, len(STATE_CHANNELS), 2, 2), dtype=np.float32)
    state[:, STATE_CHANNELS.index("last_seen_occupancy")] = 0.5
    state[:, STATE_CHANNELS.index("occlusion_age_map")] = 0.25
    state[:, STATE_CHANNELS.index("robot_velocity_channel")] = -0.4
    state[:, STATE_CHANNELS.index("robot_yaw_rate_channel")] = -0.8

    prediction = AgeDecay(future_steps=2)(state)

    assert prediction.shape == (1, 2, 2, 2)
    assert np.isfinite(prediction).all()


def test_baseline_registry_freezes_b1_through_b4_semantics() -> None:
    assert BASELINE_SPECS == {
        "B1": "last_observation_hold+hand_aggregation",
        "B2": "age_decay+hand_aggregation",
        "B3": "convgru_occupancy+hand_aggregation",
        "B4": "convgru_occupancy+learned_aggregation",
    }


def test_occupancy_metrics_match_hand_values_and_are_json_finite() -> None:
    probability = np.array([[[[0.9, 0.2], [0.7, 0.1]]]], dtype=np.float32)
    target = np.array([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=np.float32)

    metrics = occupancy_binary_metrics(probability, target, threshold=0.5)

    expected_brier = float(np.mean((probability - target) ** 2))
    assert metrics["brier"] == pytest.approx(expected_brier)
    assert metrics["intersection_over_union"] == pytest.approx(1.0)
    assert metrics["binary_accuracy"] == pytest.approx(1.0)
    assert metrics["positive_precision"] == pytest.approx(1.0)
    assert metrics["positive_recall"] == pytest.approx(1.0)
    assert all(np.isfinite(value) for value in metrics.values())


def _toy_manifest() -> dict:
    return copy.deepcopy(
        make_toy_risk_dataset(split="train", count=7, seed=0, grid_size=8).manifest
    )


def _toy_checkpoint() -> dict:
    model = ConvGRUOccupancyPredictor(hidden_channels=4, future_steps=15)
    aggregator = LearnedOccupancyRiskAggregator(future_steps=15, hidden_dim=8)
    return build_occupancy_checkpoint(
        model=model,
        learned_aggregator=aggregator,
        toy_dataset_manifest_digest="a" * 64,
        config_digest="c" * 64,
        seed=0,
    )


def test_toy_manifest_requires_exact_schema_channels_endpoints_and_digest() -> None:
    manifest = _toy_manifest()
    manifest_digest = manifest["toy_dataset_manifest_digest"]

    validated = validate_occupancy_dataset_manifest(
        manifest,
        mode="toy",
        expected_manifest_digest=manifest_digest,
    )

    assert validated["toy_dataset_manifest_digest"] == manifest_digest
    for key, bad_value in (
        ("schema_version", "2.0.0"),
        ("channel_spec", list(reversed(INPUT_CHANNELS))),
        (
            "grid",
            {
                "future_steps": 15,
                "sample_dt_s": 0.0,
                "future_time_layout": "endpoint_dt_to_horizon",
            },
        ),
        ("future_endpoint_times_s", [0.0] + manifest["future_endpoint_times_s"][1:]),
        ("toy_dataset_manifest_digest", "b" * 32),
        ("ordered_sample_ids_digest_sha256", "0" * 64),
        ("model_input_digest_sha256", "0" * 64),
        ("label_digest_sha256", "0" * 64),
        ("ordered_sample_digest_sha256", "0" * 64),
    ):
        bad = copy.deepcopy(manifest)
        bad[key] = bad_value
        with pytest.raises(ValueError):
            validate_occupancy_dataset_manifest(
                bad,
                mode="toy",
                expected_manifest_digest=manifest_digest,
            )


def test_toy_manifest_rejects_unknown_and_all_production_provenance_fields() -> None:
    manifest = _toy_manifest()
    manifest_digest = manifest["toy_dataset_manifest_digest"]
    for field in (
        "unbound_note",
        "g1_split_manifest_digest",
        "risk_dataset_manifest_digest",
        "dynamic_objects_config_digest",
        "target_type_policy_digest",
    ):
        bad = copy.deepcopy(manifest)
        bad[field] = "f" * 64
        with pytest.raises(ValueError):
            validate_occupancy_dataset_manifest(
                bad,
                mode="toy",
                expected_manifest_digest=manifest_digest,
            )


def test_occupancy_collation_binds_strict_samples_and_ordered_sidecar_ids() -> None:
    collate = getattr(
        risk_baselines_module,
        "collate_occupancy_toy_dataset",
        None,
    )
    assert callable(collate), "SOP08 must expose strict toy collation"
    dataset = make_toy_risk_dataset(split="train", count=7, seed=0, grid_size=8)

    batch = collate(dataset)

    expected_ids = tuple(sample.sample_id for sample in dataset.samples)
    assert batch["sample_ids"] == expected_ids
    assert batch["label_sidecars"]["sample_ids"] == expected_ids
    assert batch["strict_provenance"]["ordered_sample_digest_sha256"] == (
        dataset.manifest["ordered_sample_digest_sha256"]
    )

    wrong_sidecar = replace(dataset.sidecars[0], sample_id="wrong-sidecar-id")
    tampered = replace(dataset, sidecars=(wrong_sidecar, *dataset.sidecars[1:]))
    with pytest.raises(ValueError, match="sidecar.*sample IDs"):
        collate(tampered)


@pytest.mark.parametrize(
    "field",
    ("hidden_risk_occupancy", "robot_future_footprints"),
)
def test_occupancy_collation_rejects_tampered_sidecar_content(field: str) -> None:
    dataset = make_toy_risk_dataset(split="train", count=7, seed=1, grid_size=8)
    sidecar = dataset.sidecars[0]
    changed = getattr(sidecar, field).copy()
    changed.flat[0] = np.float32(1.0 - changed.flat[0])
    tampered_sidecar = replace(sidecar, **{field: changed})
    tampered = replace(dataset, sidecars=(tampered_sidecar, *dataset.sidecars[1:]))

    with pytest.raises(ValueError, match="label_sidecars_digest_sha256 mismatch"):
        risk_baselines_module.collate_occupancy_toy_dataset(tampered)


def test_occupancy_collation_rejects_tampered_manifest_row_content() -> None:
    dataset = make_toy_risk_dataset(split="train", count=7, seed=2, grid_size=8)
    changed_row = dict(dataset.manifest_rows[0])
    changed_row["blind_type"] = "tampered-blind-type"
    tampered = replace(
        dataset,
        manifest_rows=(changed_row, *dataset.manifest_rows[1:]),
    )

    with pytest.raises(ValueError, match="manifest_rows_digest_sha256 mismatch"):
        risk_baselines_module.collate_occupancy_toy_dataset(tampered)


def test_production_manifest_is_rejected_until_v2_sidecars_are_published() -> None:
    with pytest.raises(
        ProductionOccupancyContractUnavailable,
        match="dataset-level v2 manifest",
    ):
        validate_occupancy_dataset_manifest(
            {"mode": "production", "schema_version": SCHEMA_VERSION},
            mode="production",
            expected_manifest_digest="real-digest",
        )


def test_checkpoint_provenance_is_toy_only_and_fail_closed() -> None:
    checkpoint = _toy_checkpoint()

    validated = validate_occupancy_checkpoint_provenance(
        checkpoint,
        mode="toy",
        expected_manifest_digest="a" * 64,
        expected_config_digest="c" * 64,
        expected_seed=0,
        expected_model_state_digest=checkpoint["model_state_digest_sha256"],
    )

    assert validated["checkpoint_layout_version"].endswith("_v2")
    for key, value in (
        ("checkpoint_layout_version", "occupancy_baseline_checkpoint_v1"),
        ("mode", "production"),
        ("schema_version", "2.0.0"),
        ("channel_spec", list(reversed(INPUT_CHANNELS))),
        ("toy_dataset_manifest_digest", "b" * 64),
        ("config_digest", "e" * 64),
        ("g1_split_manifest_digest", "fake-g1"),
        ("seed", -1),
        ("model_variant", "tampered_model"),
        ("model_state_digest_sha256", "e" * 64),
    ):
        bad = dict(checkpoint)
        bad[key] = value
        bad["checkpoint_semantic_digest_sha256"] = (
            occupancy_checkpoint_semantic_digest(bad)
        )
        with pytest.raises(ValueError):
            validate_occupancy_checkpoint_provenance(
                bad,
                mode="toy",
                expected_manifest_digest="a" * 64,
                expected_config_digest="c" * 64,
                expected_seed=0,
                expected_model_state_digest=checkpoint["model_state_digest_sha256"],
            )


def test_checkpoint_rejects_unbound_top_level_field_after_semantic_redigest() -> None:
    checkpoint = _toy_checkpoint()
    checkpoint["review_note"] = "not-covered-by-the-frozen-semantic-projection"
    checkpoint["checkpoint_semantic_digest_sha256"] = (
        occupancy_checkpoint_semantic_digest(checkpoint)
    )

    with pytest.raises(ValueError, match="unexpected top-level fields"):
        validate_occupancy_checkpoint_provenance(
            checkpoint,
            mode="toy",
            expected_manifest_digest="a" * 64,
            expected_config_digest="c" * 64,
            expected_seed=0,
        )


@pytest.mark.parametrize(
    "field",
    (
        "g1_split_manifest_digest",
        "risk_dataset_manifest_digest",
        "dynamic_objects_config_digest",
        "target_type_policy_digest",
    ),
)
def test_toy_checkpoint_rejects_every_production_provenance_field_after_redigest(
    field: str,
) -> None:
    checkpoint = _toy_checkpoint()
    checkpoint[field] = "f" * 64
    checkpoint["checkpoint_semantic_digest_sha256"] = (
        occupancy_checkpoint_semantic_digest(checkpoint)
    )

    with pytest.raises(ValueError, match="production provenance"):
        validate_occupancy_checkpoint_provenance(
            checkpoint,
            mode="toy",
            expected_manifest_digest="a" * 64,
            expected_config_digest="c" * 64,
            expected_seed=0,
        )


def test_checkpoint_semantic_digest_covers_all_frozen_provenance_fields() -> None:
    checkpoint = {
        "checkpoint_layout_version": "occupancy_baseline_checkpoint_v2",
        "mode": "toy",
        "schema_version": SCHEMA_VERSION,
        "channel_spec": list(INPUT_CHANNELS),
        "toy_dataset_manifest_digest": "a" * 64,
        "config_digest": "c" * 64,
        "seed": 0,
        "model_variant": "convgru_hidden_occupancy+B4_learned_aggregator",
        "model_state_digest_sha256": "d" * 64,
    }
    reference = occupancy_checkpoint_semantic_digest(checkpoint)

    mutations = {
        "checkpoint_layout_version": "other-layout",
        "mode": "production",
        "schema_version": "2.0.0",
        "channel_spec": list(reversed(INPUT_CHANNELS)),
        "toy_dataset_manifest_digest": "b" * 64,
        "config_digest": "e" * 64,
        "seed": 1,
        "model_variant": "other-model",
        "model_state_digest_sha256": "f" * 64,
    }
    for key, value in mutations.items():
        changed = dict(checkpoint)
        changed[key] = value
        assert occupancy_checkpoint_semantic_digest(changed) != reference


def test_checkpoint_round_trip_reproduces_identical_predictions(tmp_path) -> None:
    import torch

    torch.manual_seed(31)
    model = ConvGRUOccupancyPredictor(hidden_channels=4, future_steps=15)
    aggregator = LearnedOccupancyRiskAggregator(future_steps=15, hidden_dim=8)
    history = torch.rand(2, 8, len(HISTORY_CHANNELS), 5, 5, dtype=torch.float32)
    footprint = torch.ones(2, 15, 5, 5, dtype=torch.float32)
    with torch.no_grad():
        expected_occupancy = model(history)
        expected_risk = aggregator(expected_occupancy, footprint)
    checkpoint = build_occupancy_checkpoint(
        model=model,
        learned_aggregator=aggregator,
        toy_dataset_manifest_digest="a" * 64,
        config_digest="c" * 64,
        seed=31,
    )
    checkpoint_path = tmp_path / "occupancy.pt"
    save_occupancy_checkpoint(checkpoint_path, checkpoint)

    reloaded_model = ConvGRUOccupancyPredictor(hidden_channels=4, future_steps=15)
    reloaded_aggregator = LearnedOccupancyRiskAggregator(future_steps=15, hidden_dim=8)
    load_occupancy_checkpoint(
        checkpoint_path,
        model=reloaded_model,
        learned_aggregator=reloaded_aggregator,
        mode="toy",
        expected_manifest_digest="a" * 64,
        expected_config_digest="c" * 64,
        expected_seed=31,
    )

    with torch.no_grad():
        actual_occupancy = reloaded_model(history)
        actual_risk = reloaded_aggregator(actual_occupancy, footprint)
    torch.testing.assert_close(actual_occupancy, expected_occupancy, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_risk, expected_risk, rtol=0.0, atol=0.0)
