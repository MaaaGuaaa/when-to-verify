"""SOP09 model, dataloader, provenance, and toy-contract tests."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.contracts import (
    HISTORY_CHANNELS,
    INPUT_CHANNELS,
    SCHEMA_VERSION,
    STATE_CHANNELS,
    TRAJECTORY_CHANNELS,
    validate_risk_sample,
)
from src.datasets.risk_dataloader import (
    MODEL_INPUT_KEYS,
    RiskDataContractError,
    collate_risk_samples,
    load_production_risk_dataset,
    validate_toy_dataset_manifest,
    validate_model_input_mapping,
)
from src.datasets import toy_risk_learning
from src.datasets.toy_risk_learning import (
    TOY_CASES,
    TOY_FUTURE_ENDPOINT_TIMES_S,
    assert_toy_split_isolation,
    make_toy_batch,
    make_toy_risk_dataset,
)
from src.models.risk_model import (
    RISK_CHECKPOINT_LAYOUT_VERSION,
    RiskModel,
    load_risk_checkpoint,
    save_risk_checkpoint,
)
from src.models import risk_model as risk_model_module


def _channel_spec() -> dict[str, list[str]]:
    return {
        "history": list(HISTORY_CHANNELS),
        "state": list(STATE_CHANNELS),
        "trajectory": list(TRAJECTORY_CHANNELS),
        "flat": list(INPUT_CHANNELS),
    }


def _input_and_label_content_digest(sample) -> str:
    """Hash numerical content only so renamed IDs cannot fake split isolation."""

    digest = hashlib.sha256()
    for value in (
        sample.bev_history,
        sample.state_channels,
        sample.trajectory_channels,
        sample.robot_state,
    ):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.view(np.uint8))
    labels = np.asarray(
        [
            sample.collision_label,
            sample.risk_severity,
            sample.min_clearance,
            sample.near_miss,
            -1.0
            if sample.first_collision_time is None
            else sample.first_collision_time,
        ],
        dtype=np.float64,
    )
    digest.update(labels.tobytes())
    return digest.hexdigest()


def _rewrite_checkpoint_semantic_digest(payload: dict[str, object]) -> None:
    """Re-sign a deliberately modified semantic payload for contract tests."""

    semantic = {
        "checkpoint_layout_version": payload.get("checkpoint_layout_version"),
        "mode": payload.get("mode"),
        "model_config": payload.get("model_config"),
        "model_state_digest_sha256": payload.get("model_state_digest_sha256"),
        "provenance": payload.get("provenance"),
        "inference_parameters": payload.get("inference_parameters"),
    }
    encoded = json.dumps(
        semantic,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    payload["checkpoint_semantic_digest_sha256"] = hashlib.sha256(encoded).hexdigest()


def _save_contract_test_checkpoint(tmp_path: Path) -> Path:
    dataset = make_toy_risk_dataset(split="train", count=2, seed=44, grid_size=12)
    validation = make_toy_risk_dataset(split="val", count=2, seed=44, grid_size=12)
    model = RiskModel(variant="r0", hidden_channels=8)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "channel_spec": _channel_spec(),
        "model_variant": "r0",
        "config_digest": "c" * 32,
        "toy_dataset_manifest_digest": dataset.manifest_digest,
        "validation_dataset_manifest_digest": validation.manifest_digest,
        "seed": 44,
    }
    path = tmp_path / "contract-test.pt"
    save_risk_checkpoint(path, model=model, mode="toy", provenance=provenance)
    return path


def test_toy_publication_is_schema_valid_and_keeps_oracle_sidecars_out_of_inputs():
    dataset = make_toy_risk_dataset(split="train", count=14, seed=17, grid_size=16)
    batch = make_toy_batch(dataset)

    assert len(dataset.samples) == 14
    for sample in dataset.samples:
        validate_risk_sample(sample, dataset.grid)
    assert set(batch["model_inputs"]) == set(MODEL_INPUT_KEYS)
    assert batch["model_inputs"]["bev_history"].shape == (14, 8, 2, 16, 16)
    assert batch["model_inputs"]["state_channels"].shape == (14, 9, 16, 16)
    assert batch["model_inputs"]["trajectory_channels"].shape == (14, 4, 16, 16)
    assert batch["model_inputs"]["robot_state"].shape == (14, 2)
    assert all(value.dtype == np.float32 for value in batch["model_inputs"].values())
    assert all(np.isfinite(value).all() for value in batch["model_inputs"].values())

    sidecars = batch["label_sidecars"]
    assert sidecars["hidden_risk_occupancy"].shape == (14, 15, 16, 16)
    assert sidecars["robot_future_footprints"].shape == (14, 15, 16, 16)
    assert np.array_equal(
        sidecars["future_endpoint_times_s"], TOY_FUTURE_ENDPOINT_TIMES_S
    )
    assert sidecars["future_endpoint_times_s"].dtype == np.float32
    assert sidecars["future_endpoint_times_s"][0] == pytest.approx(0.2)
    assert sidecars["future_endpoint_times_s"][-1] == pytest.approx(3.0)
    assert not any("future" in key or "oracle" in key for key in batch["model_inputs"])
    assert all(
        "hidden_risk_occupancy" not in sample.metadata
        and "robot_future_footprints" not in sample.metadata
        for sample in dataset.samples
    )

    manifest = dataset.manifest
    assert manifest["mode"] == "toy"
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["channel_spec"] == _channel_spec()
    assert manifest["toy_dataset_manifest_digest"] == dataset.manifest_digest
    assert manifest["ordered_sample_ids"] == [
        sample.sample_id for sample in dataset.samples
    ]
    for field in (
        "ordered_sample_ids_digest_sha256",
        "model_input_digest_sha256",
        "label_digest_sha256",
        "ordered_sample_digest_sha256",
    ):
        assert len(manifest[field]) == 64
    assert manifest["future_endpoint_times_s"] == pytest.approx(
        [0.2 * step for step in range(1, 16)]
    )
    assert "g1_split_manifest_digest" not in manifest


def test_frozen_endpoint_constant_cannot_be_mutated_by_a_caller():
    with pytest.raises(ValueError, match="read-only"):
        TOY_FUTURE_ENDPOINT_TIMES_S[0] = np.float32(9.9)


def test_toy_hidden_risk_observations_are_visibility_consistent_for_every_case_type():
    publications = tuple(
        make_toy_risk_dataset(
            split="train", count=21, seed=seed, grid_size=16
        )
        for seed in range(17, 23)
    )
    state_index = {name: index for index, name in enumerate(STATE_CHANNELS)}
    observed_case_types: set[tuple[str, str]] = set()

    for dataset in publications:
        for sample in dataset.samples:
            observed_case_types.add(
                (sample.event_type, str(sample.metadata["target_object_type"]))
            )
            history_occupied = sample.bev_history[:, 0] > 0.5
            history_visible = sample.bev_history[:, 1] > 0.5
            visible_free = (
                sample.state_channels[state_index["current_visible_free"]] > 0.5
            )
            visible_occupied = (
                sample.state_channels[state_index["current_visible_occupied"]] > 0.5
            )
            unobservable = (
                sample.state_channels[state_index["current_unobservable_mask"]] > 0.5
            )
            last_seen = (
                sample.state_channels[state_index["last_seen_occupancy"]] > 0.5
            )

            assert np.all(~history_occupied | history_visible), sample.sample_id
            assert not np.any(visible_free & visible_occupied), sample.sample_id
            assert not np.any(visible_free & unobservable), sample.sample_id
            assert not np.any(visible_occupied & unobservable), sample.sample_id
            assert np.all(
                visible_free | visible_occupied | unobservable
            ), sample.sample_id
            if sample.event_type == "empty":
                assert not np.any(last_seen)
            else:
                assert np.any(last_seen)
                assert np.all(~last_seen | unobservable), sample.sample_id
                assert not np.any(last_seen & visible_free), sample.sample_id
                assert not np.any(last_seen & visible_occupied), sample.sample_id

            assert "hidden_risk_occupancy" not in sample.metadata
            assert "robot_future_footprints" not in sample.metadata

    expected_case_types = {
        (case, object_type)
        for case in TOY_CASES
        for object_type in ("human", "carried_object", "unknown_dynamic")
    }
    assert observed_case_types == expected_case_types


def test_toy_labels_use_declared_single_cell_target_and_robot_footprint_geometry():
    dataset = make_toy_risk_dataset(split="train", count=14, seed=29, grid_size=16)
    sidecars = dataset.sidecar_by_sample_id()
    rows = {str(row["sample_id"]): row for row in dataset.manifest_rows}
    resolution_m = dataset.grid.resolution_m

    for sample in dataset.samples:
        row = rows[sample.sample_id]
        sidecar = sidecars[sample.sample_id]
        for source in (sample.metadata, row):
            assert source["footprint_kind"] == "single_grid_cell_square"
            assert source["footprint_dimensions_m"] == pytest.approx(
                [resolution_m, resolution_m]
            )
            assert source["robot_footprint_kind"] == "single_grid_cell_square"
            assert source["robot_footprint_dimensions_m"] == pytest.approx(
                [resolution_m, resolution_m]
            )
            assert source["footprint_contact_policy"] == "positive_area_overlap"

        signed_clearances: list[float] = []
        collision_steps: list[int] = []
        for step in range(dataset.grid.future_steps):
            target_cells = np.argwhere(
                sidecar.hidden_risk_occupancy[step] > 0.5
            )
            robot_cells = np.argwhere(sidecar.robot_future_footprints[step] > 0.5)
            if target_cells.size == 0:
                continue
            delta_m = (
                np.abs(target_cells[:, None, :] - robot_cells[None, :, :])
                * resolution_m
            )
            axis_gaps = delta_m - resolution_m
            overlap = np.all(axis_gaps < 0.0, axis=-1)
            if np.any(overlap):
                collision_steps.append(step)
                signed_clearances.append(
                    -float(np.min(-axis_gaps[overlap], axis=-1).max())
                )
            else:
                separation = np.maximum(axis_gaps, 0.0)
                signed_clearances.append(
                    float(np.sqrt(np.sum(separation**2, axis=-1)).min())
                )

        expected_collision = int(bool(collision_steps))
        expected_clearance = 99.0 if not signed_clearances else min(signed_clearances)
        expected_near_miss = int(
            not expected_collision and expected_clearance <= 0.25
        )
        assert sample.collision_label == expected_collision, sample.sample_id
        assert sample.min_clearance == pytest.approx(expected_clearance)
        assert sample.near_miss == expected_near_miss
        assert np.isfinite(sample.risk_severity)
        if collision_steps:
            assert sample.first_collision_time == pytest.approx(
                (collision_steps[0] + 1) * 0.2
            )
        else:
            assert sample.first_collision_time is None

    collision = next(
        sample for sample in dataset.samples if sample.event_type == "collision"
    )
    near_miss = next(
        sample for sample in dataset.samples if sample.event_type == "near_miss"
    )
    safe = next(
        sample for sample in dataset.samples if sample.event_type == "same_area_safe"
    )
    assert (collision.collision_label, collision.near_miss) == (1, 0)
    assert collision.min_clearance == pytest.approx(-resolution_m)
    assert (near_miss.collision_label, near_miss.near_miss) == (0, 1)
    assert near_miss.min_clearance == pytest.approx(0.0)
    assert (safe.collision_label, safe.near_miss) == (0, 0)
    assert safe.min_clearance >= 2.0 * resolution_m - 1e-7


def test_toy_publication_digest_is_canonical_and_binds_rows_and_sidecars():
    dataset = make_toy_risk_dataset(split="train", count=7, seed=30, grid_size=12)
    manifest = dataset.manifest
    assert len(manifest["manifest_rows_digest_sha256"]) == 64
    assert len(manifest["label_sidecars_digest_sha256"]) == 64
    header = dict(manifest)
    declared = str(header.pop("toy_dataset_manifest_digest"))
    canonical_header = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert declared == hashlib.blake2b(
        canonical_header, digest_size=16
    ).hexdigest()
    assert toy_risk_learning.validate_toy_risk_dataset_publication(dataset)[
        "toy_dataset_manifest_digest"
    ] == declared


def test_toy_publication_validation_rejects_row_sidecar_input_and_label_tampering():
    dataset = make_toy_risk_dataset(split="train", count=7, seed=32, grid_size=12)
    validate = toy_risk_learning.validate_toy_risk_dataset_publication

    changed_rows = [dict(row) for row in dataset.manifest_rows]
    changed_rows[0]["blind_type"] = (
        "dynamic" if changed_rows[0]["blind_type"] == "structural" else "structural"
    )
    with pytest.raises(ValueError, match="manifest_rows_digest_sha256"):
        validate(replace(dataset, manifest_rows=tuple(changed_rows)))

    changed_sidecar = dataset.sidecars[0].hidden_risk_occupancy.copy()
    changed_sidecar[0, 0, 0] = np.float32(
        1.0 - changed_sidecar[0, 0, 0]
    )
    sidecars = (
        replace(dataset.sidecars[0], hidden_risk_occupancy=changed_sidecar),
        *dataset.sidecars[1:],
    )
    with pytest.raises(ValueError, match="label_sidecars_digest_sha256"):
        validate(replace(dataset, sidecars=sidecars))

    robot_state = dataset.samples[0].robot_state.copy()
    robot_state[0] += np.float32(0.01)
    changed_input = replace(dataset.samples[0], robot_state=robot_state)
    with pytest.raises(ValueError, match="model_input_digest_sha256"):
        validate(replace(dataset, samples=(changed_input, *dataset.samples[1:])))

    changed_severity = (
        0.99
        if dataset.samples[0].risk_severity == 1.0
        else float(dataset.samples[0].risk_severity + 0.01)
    )
    changed_label = replace(
        dataset.samples[0], risk_severity=changed_severity
    )
    with pytest.raises(ValueError, match="label_digest_sha256"):
        validate(replace(dataset, samples=(changed_label, *dataset.samples[1:])))


def test_toy_manifest_and_collate_recompute_full_publication_digest():
    dataset = make_toy_risk_dataset(split="train", count=7, seed=34, grid_size=12)
    tampered = copy.deepcopy(dataset.manifest)
    tampered["toy_dataset_manifest_digest"] = "a" * 32
    with pytest.raises(RiskDataContractError, match="toy_dataset_manifest_digest"):
        validate_toy_dataset_manifest(tampered, expected_split="train")
    with pytest.raises(RiskDataContractError, match="toy_dataset_manifest_digest"):
        collate_risk_samples(
            dataset.samples,
            grid=dataset.grid,
            dataset_manifest=tampered,
            expected_split="train",
        )

    production_tampered = copy.deepcopy(dataset.manifest)
    production_tampered["g1_split_manifest_digest"] = "legal-looking-production-id"
    with pytest.raises(RiskDataContractError, match="production provenance"):
        validate_toy_dataset_manifest(production_tampered, expected_split="train")

    unexpected = copy.deepcopy(dataset.manifest)
    unexpected["unbound_but_legal_json"] = "value"
    with pytest.raises(RiskDataContractError, match="top-level keys"):
        validate_toy_dataset_manifest(unexpected, expected_split="train")


def test_toy_publications_are_deterministic_and_strictly_split_isolated():
    first = make_toy_risk_dataset(split="train", count=14, seed=23, grid_size=12)
    repeat = make_toy_risk_dataset(split="train", count=14, seed=23, grid_size=12)
    calibration = make_toy_risk_dataset(
        split="calibration", count=14, seed=23, grid_size=12
    )
    validation = make_toy_risk_dataset(
        split="val", count=14, seed=23, grid_size=12
    )
    test = make_toy_risk_dataset(split="test", count=14, seed=23, grid_size=12)
    different_seed = make_toy_risk_dataset(
        split="train", count=14, seed=24, grid_size=12
    )

    assert first.manifest_digest == repeat.manifest_digest
    assert [sample.sample_id for sample in first.samples] == [
        sample.sample_id for sample in repeat.samples
    ]
    for left, right in zip(first.samples, repeat.samples):
        assert np.array_equal(left.bev_history, right.bev_history)
        assert np.array_equal(left.state_channels, right.state_channels)
        assert np.array_equal(left.trajectory_channels, right.trajectory_channels)
    report = assert_toy_split_isolation((first, calibration, validation, test))
    assert report["passed"] is True
    assert all(value == 0 for value in report["overlap_counts"].values())

    numeric_digest_sets = [
        {_input_and_label_content_digest(sample) for sample in dataset.samples}
        for dataset in (first, calibration, validation, test)
    ]
    for left_index, left in enumerate(numeric_digest_sets):
        for right in numeric_digest_sets[left_index + 1 :]:
            assert left.isdisjoint(right)
    assert first.manifest["model_input_digest_sha256"] != (
        different_seed.manifest["model_input_digest_sha256"]
    )
    assert first.manifest["label_digest_sha256"] != (
        different_seed.manifest["label_digest_sha256"]
    )
    assert any(
        not np.array_equal(left.robot_state, right.robot_state)
        for left, right in zip(first.samples, different_seed.samples)
    )


def test_toy_matched_groups_share_context_and_ood_samples_are_singletons():
    dataset = make_toy_risk_dataset(split="train", count=28, seed=25, grid_size=16)
    sample_by_id = {sample.sample_id: sample for sample in dataset.samples}
    grouped_rows: dict[str, list[dict[str, object]]] = {}
    for row in dataset.manifest_rows:
        grouped_rows.setdefault(str(row["pair_group_id"]), []).append(row)

    expected_cases = set(TOY_CASES) - {"ood"}
    normal_group_count = 0
    singleton_count = 0
    for rows in grouped_rows.values():
        if bool(rows[0]["pair_eligible"]):
            normal_group_count += 1
            assert len(rows) == 6
            assert {str(row["event_type"]) for row in rows} == expected_cases
            for field in (
                "pair_group_id",
                "base_state_id",
                "trajectory_id",
                "snippet_id",
                "occluder_id",
                "background_id",
                "source_object_id",
                "target_object_type",
                "footprint_kind",
                "footprint_dimensions_m",
            ):
                values = {
                    json.dumps(row[field], sort_keys=True) for row in rows
                }
                assert len(values) == 1, field
            samples = [sample_by_id[str(row["sample_id"])] for row in rows]
            assert all(
                np.array_equal(samples[0].trajectory_channels, sample.trajectory_channels)
                for sample in samples[1:]
            )
        else:
            singleton_count += 1
            assert len(rows) == 1
            assert rows[0]["event_type"] == "ood"
            assert rows[0]["ood_tag"] == "heldout_motion"
    assert normal_group_count == 4
    assert singleton_count == 4


def test_toy_static_obstacles_never_overlap_robot_swept_volume():
    dataset = make_toy_risk_dataset(split="train", count=28, seed=26, grid_size=16)
    static_index = STATE_CHANNELS.index("static_obstacle_map")
    swept_index = TRAJECTORY_CHANNELS.index("swept_volume_mask")

    for sample in dataset.samples:
        static_obstacles = sample.state_channels[static_index] > 0.5
        robot_swept_volume = sample.trajectory_channels[swept_index] > 0.5
        assert np.any(static_obstacles[1:-1, 1:-1])
        assert not np.any(static_obstacles & robot_swept_volume), sample.sample_id


def test_toy_static_obstacles_are_current_visible_occupied_not_free_or_hidden():
    dataset = make_toy_risk_dataset(split="train", count=70, seed=26, grid_size=16)
    state_index = {name: index for index, name in enumerate(STATE_CHANNELS)}

    for sample in dataset.samples:
        static_obstacles = (
            sample.state_channels[state_index["static_obstacle_map"]] > 0.5
        )
        visible_free = (
            sample.state_channels[state_index["current_visible_free"]] > 0.5
        )
        visible_occupied = (
            sample.state_channels[state_index["current_visible_occupied"]] > 0.5
        )
        unobservable = (
            sample.state_channels[state_index["current_unobservable_mask"]] > 0.5
        )

        assert np.any(static_obstacles), sample.sample_id
        assert not np.any(static_obstacles & visible_free), sample.sample_id
        assert not np.any(static_obstacles & unobservable), sample.sample_id
        assert np.all(~static_obstacles | visible_occupied), sample.sample_id
        assert np.all(
            visible_free | visible_occupied | unobservable
        ), sample.sample_id


def test_toy_history_is_contiguous_motion_not_an_arbitrary_event_code():
    dataset = make_toy_risk_dataset(split="train", count=14, seed=27, grid_size=16)
    sidecars = dataset.sidecar_by_sample_id()
    state_index = {name: index for index, name in enumerate(STATE_CHANNELS)}
    last_seen_index = state_index["last_seen_occupancy"]
    groups: dict[str, list] = {}
    for sample in dataset.samples:
        if sample.metadata["pair_eligible"]:
            groups.setdefault(sample.pair_group_id, []).append(sample)

    assert groups
    for samples in groups.values():
        reference_without_last_seen = np.delete(
            samples[0].state_channels, last_seen_index, axis=0
        )
        for sample in samples:
            assert np.array_equal(
                reference_without_last_seen,
                np.delete(sample.state_channels, last_seen_index, axis=0),
            )
            history_cells = [
                np.argwhere(sample.bev_history[step, 0] > 0.5)
                for step in range(sample.bev_history.shape[0])
            ]
            if sample.event_type == "empty":
                assert all(cells.size == 0 for cells in history_cells)
                continue
            assert all(cells.shape == (1, 2) for cells in history_cells)
            positions = np.stack([cells[0] for cells in history_cells])
            assert np.all(np.abs(np.diff(positions, axis=0)) <= 1)
            future_cells = np.argwhere(
                sidecars[sample.sample_id].hidden_risk_occupancy[0] > 0.5
            )
            assert future_cells.shape == (1, 2)
            assert np.all(np.abs(future_cells[0] - positions[-1]) <= 1)


def test_toy_case_labels_come_from_time_aligned_future_geometry():
    dataset = make_toy_risk_dataset(split="train", count=14, seed=29, grid_size=16)
    batch = make_toy_batch(dataset)
    occupancy = batch["label_sidecars"]["hidden_risk_occupancy"]
    footprints = batch["label_sidecars"]["robot_future_footprints"]
    intersections = np.any((occupancy > 0.5) & (footprints > 0.5), axis=(1, 2, 3))

    assert np.array_equal(
        intersections.astype(np.float32), batch["labels"]["collision_label"]
    )
    for index, sample in enumerate(dataset.samples):
        if sample.first_collision_time is None:
            continue
        first = np.flatnonzero(
            np.any(
                (occupancy[index] > 0.5) & (footprints[index] > 0.5),
                axis=(1, 2),
            )
        )[0]
        assert sample.first_collision_time == pytest.approx(float((first + 1) * 0.2))


def test_toy_future_occupancy_is_a_function_of_all_deployable_inputs():
    dataset = make_toy_risk_dataset(split="train", count=128, seed=30, grid_size=12)
    sidecars = dataset.sidecar_by_sample_id()
    future_by_history: dict[bytes, np.ndarray] = {}
    for sample in dataset.samples:
        key = b"".join(
            value.tobytes(order="C")
            for value in (
                sample.bev_history,
                sample.state_channels,
                sample.trajectory_channels,
                sample.robot_state,
            )
        )
        future = sidecars[sample.sample_id].hidden_risk_occupancy
        if key in future_by_history:
            assert np.array_equal(future_by_history[key], future)
        else:
            future_by_history[key] = future.copy()


def test_dataloader_validates_channels_split_dtype_finite_and_oracle_isolation():
    dataset = make_toy_risk_dataset(split="train", count=7, seed=31, grid_size=12)
    batch = collate_risk_samples(
        dataset.samples,
        grid=dataset.grid,
        dataset_manifest=dataset.manifest,
        expected_split="train",
    )
    assert tuple(batch.model_inputs) == MODEL_INPUT_KEYS
    assert batch.targets["risk_severity"].dtype == torch.float32
    assert batch.provenance["toy_dataset_manifest_digest"] == dataset.manifest_digest

    wrong_order = copy.deepcopy(dataset.manifest)
    wrong_order["channel_spec"]["state"] = list(reversed(STATE_CHANNELS))
    with pytest.raises(RiskDataContractError, match="channel_spec"):
        collate_risk_samples(
            dataset.samples,
            grid=dataset.grid,
            dataset_manifest=wrong_order,
            expected_split="train",
        )

    wrong_schema = copy.deepcopy(dataset.manifest)
    wrong_schema["schema_version"] = "2.0.0"
    with pytest.raises(RiskDataContractError, match="schema_version"):
        collate_risk_samples(
            dataset.samples,
            grid=dataset.grid,
            dataset_manifest=wrong_schema,
            expected_split="train",
        )

    with pytest.raises(RiskDataContractError, match="split"):
        collate_risk_samples(
            dataset.samples,
            grid=dataset.grid,
            dataset_manifest=dataset.manifest,
            expected_split="test",
        )

    another_publication = make_toy_risk_dataset(
        split="train", count=7, seed=32, grid_size=12
    )
    with pytest.raises(RiskDataContractError, match="ordered_sample_ids"):
        collate_risk_samples(
            dataset.samples,
            grid=dataset.grid,
            dataset_manifest=another_publication.manifest,
            expected_split="train",
        )

    wrong_count = copy.deepcopy(dataset.manifest)
    wrong_count["sample_count"] += 1
    with pytest.raises(RiskDataContractError, match="sample_count"):
        collate_risk_samples(
            dataset.samples,
            grid=dataset.grid,
            dataset_manifest=wrong_count,
            expected_split="train",
        )

    wrong_grid = copy.deepcopy(dataset.manifest)
    wrong_grid["grid"]["width"] += 1
    with pytest.raises(RiskDataContractError, match="grid"):
        collate_risk_samples(
            dataset.samples,
            grid=dataset.grid,
            dataset_manifest=wrong_grid,
            expected_split="train",
        )

    changed_robot_state = dataset.samples[0].robot_state.copy()
    changed_robot_state[0] += np.float32(0.01)
    changed_input = replace(dataset.samples[0], robot_state=changed_robot_state)
    with pytest.raises(RiskDataContractError, match="model_input_digest_sha256"):
        collate_risk_samples(
            (changed_input, *dataset.samples[1:]),
            grid=dataset.grid,
            dataset_manifest=dataset.manifest,
            expected_split="train",
        )

    bad_dtype = replace(
        dataset.samples[0], bev_history=dataset.samples[0].bev_history.astype(np.float64)
    )
    with pytest.raises(RiskDataContractError, match="float32"):
        collate_risk_samples(
            (bad_dtype, *dataset.samples[1:]),
            grid=dataset.grid,
            dataset_manifest=dataset.manifest,
            expected_split="train",
        )

    bad_values_array = dataset.samples[0].bev_history.copy()
    bad_values_array[0, 0, 0, 0] = np.nan
    bad_values = replace(dataset.samples[0], bev_history=bad_values_array)
    with pytest.raises(RiskDataContractError, match="NaN/Inf"):
        collate_risk_samples(
            (bad_values, *dataset.samples[1:]),
            grid=dataset.grid,
            dataset_manifest=dataset.manifest,
            expected_split="train",
        )

    with pytest.raises(RiskDataContractError, match="forbidden"):
        validate_model_input_mapping(
            {**batch.model_inputs, "nested": {"oracle_future": torch.zeros(1)}}
        )


def test_toy_label_digest_binds_near_miss_and_first_collision_time():
    dataset = make_toy_risk_dataset(split="train", count=7, seed=33, grid_size=12)
    samples = list(dataset.samples)
    near_miss_index = next(
        index for index, sample in enumerate(samples) if sample.near_miss == 1
    )
    samples[near_miss_index] = replace(samples[near_miss_index], near_miss=0)
    with pytest.raises(RiskDataContractError, match="label_digest_sha256"):
        collate_risk_samples(
            samples,
            grid=dataset.grid,
            dataset_manifest=dataset.manifest,
            expected_split="train",
        )

    samples = list(dataset.samples)
    collision_index = next(
        index for index, sample in enumerate(samples) if sample.collision_label == 1
    )
    original_time = samples[collision_index].first_collision_time
    assert original_time is not None
    samples[collision_index] = replace(
        samples[collision_index], first_collision_time=float(original_time + 0.2)
    )
    with pytest.raises(RiskDataContractError, match="label_digest_sha256"):
        collate_risk_samples(
            samples,
            grid=dataset.grid,
            dataset_manifest=dataset.manifest,
            expected_split="train",
        )


def test_production_v1_dataset_is_rejected_without_interpretation(tmp_path):
    root = tmp_path / "ambiguous-v1"
    root.mkdir()
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "layout_version": "risk_shard_npz_jsonl_v1",
                "schema_version": SCHEMA_VERSION,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RiskDataContractError, match="dataset-level v2"):
        load_production_risk_dataset(root)


@pytest.mark.parametrize("variant", ["r0", "r1"])
def test_risk_models_have_unified_finite_noncrossing_output_and_gradients(variant):
    dataset = make_toy_risk_dataset(split="train", count=4, seed=37, grid_size=12)
    batch = collate_risk_samples(
        dataset.samples,
        grid=dataset.grid,
        dataset_manifest=dataset.manifest,
        expected_split="train",
    )
    model = RiskModel(variant=variant, hidden_channels=8)
    output = model(batch.model_inputs)

    assert set(output) == {"quantiles", "collision_logits", "p_collision"}
    assert output["quantiles"].shape == (4, 4)
    assert output["collision_logits"].shape == (4,)
    assert output["p_collision"].shape == (4,)
    assert torch.isfinite(output["quantiles"]).all()
    assert torch.isfinite(output["collision_logits"]).all()
    assert torch.all(output["quantiles"][:, 1:] >= output["quantiles"][:, :-1])
    assert torch.all((output["p_collision"] >= 0) & (output["p_collision"] <= 1))
    assert torch.any(output["quantiles"][:, -1] < 0.9999)

    (output["quantiles"].sum() + output["collision_logits"].sum()).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("variant", ["r0", "r1"])
def test_r0_and_r1_outputs_depend_on_legal_permuted_validation_queries(variant):
    dataset = make_toy_risk_dataset(split="val", count=7, seed=39, grid_size=12)
    batch = collate_risk_samples(
        dataset.samples,
        grid=dataset.grid,
        dataset_manifest=dataset.manifest,
        expected_split="val",
    )
    torch.manual_seed(390)
    model = RiskModel(variant=variant, hidden_channels=8).eval()
    trajectory = batch.model_inputs["trajectory_channels"].clone().requires_grad_(True)
    inputs = {**batch.model_inputs, "trajectory_channels": trajectory}
    output = model(inputs)
    scalar = output["quantiles"].sum() + output["collision_logits"].sum()
    scalar.backward()

    assert trajectory.grad is not None
    assert torch.isfinite(trajectory.grad).all()
    assert float(trajectory.grad.abs().sum().item()) > 0.0

    permutation = torch.roll(torch.arange(len(batch.sample_ids)), shifts=-1)
    permuted_trajectory = batch.model_inputs["trajectory_channels"][permutation]
    permuted_robot_state = batch.model_inputs["robot_state"][permutation]
    trajectory_changed = torch.any(
        (
            permuted_trajectory
            != batch.model_inputs["trajectory_channels"]
        ).reshape(len(batch.sample_ids), -1),
        dim=1,
    )
    robot_state_changed = torch.any(
        permuted_robot_state != batch.model_inputs["robot_state"], dim=1
    )
    changed_count = int(torch.count_nonzero(trajectory_changed | robot_state_changed))
    assert changed_count > 0
    with torch.no_grad():
        counterfactual = model(
            {
                **batch.model_inputs,
                "trajectory_channels": permuted_trajectory,
                "robot_state": permuted_robot_state,
            }
        )
    output_delta = (
        torch.mean(
            torch.abs(output["quantiles"].detach() - counterfactual["quantiles"])
        )
        + torch.mean(
            torch.abs(
                output["p_collision"].detach() - counterfactual["p_collision"]
            )
        )
    )
    assert torch.isfinite(output_delta)
    assert float(output_delta.item()) > (
        risk_model_module.TRAJECTORY_SENSITIVITY_EPSILON
    )


def test_model_rejects_wrong_input_keys_dtype_and_shape():
    dataset = make_toy_risk_dataset(split="train", count=2, seed=41, grid_size=12)
    batch = collate_risk_samples(
        dataset.samples,
        grid=dataset.grid,
        dataset_manifest=dataset.manifest,
        expected_split="train",
    )
    model = RiskModel(variant="r0", hidden_channels=8)

    with pytest.raises(RiskDataContractError, match="keys"):
        model({**batch.model_inputs, "future_occupancy": torch.zeros(1)})
    with pytest.raises(RiskDataContractError, match="float32"):
        model(
            {
                **batch.model_inputs,
                "robot_state": batch.model_inputs["robot_state"].to(torch.float64),
            }
        )
    with pytest.raises(RiskDataContractError, match="history"):
        model(
            {
                **batch.model_inputs,
                "bev_history": batch.model_inputs["bev_history"][:, :-1],
            }
        )


def test_toy_checkpoint_round_trip_and_mode_provenance_are_fail_closed(tmp_path):
    dataset = make_toy_risk_dataset(split="train", count=4, seed=43, grid_size=12)
    validation = make_toy_risk_dataset(split="val", count=4, seed=43, grid_size=12)
    batch = collate_risk_samples(
        dataset.samples,
        grid=dataset.grid,
        dataset_manifest=dataset.manifest,
        expected_split="train",
    )
    model = RiskModel(variant="r0", hidden_channels=8).eval()
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "channel_spec": _channel_spec(),
        "model_variant": "r0",
        "config_digest": "a" * 32,
        "toy_dataset_manifest_digest": dataset.manifest_digest,
        "validation_dataset_manifest_digest": validation.manifest_digest,
        "seed": 43,
    }
    path = tmp_path / "risk-r0.pt"
    with torch.no_grad():
        expected = model(batch.model_inputs)
    save_risk_checkpoint(path, model=model, mode="toy", provenance=provenance)
    loaded, payload = load_risk_checkpoint(
        path,
        expected_mode="toy",
        expected_provenance=provenance,
    )
    with torch.no_grad():
        actual = loaded.eval()(batch.model_inputs)

    assert payload["checkpoint_layout_version"] == RISK_CHECKPOINT_LAYOUT_VERSION
    assert len(payload["model_state_digest_sha256"]) == 64
    assert len(payload["checkpoint_semantic_digest_sha256"]) == 64
    assert "g1_split_manifest_digest" not in payload["provenance"]
    assert payload["provenance"]["toy_dataset_manifest_digest"] == (
        dataset.manifest_digest
    )
    assert payload["provenance"]["validation_dataset_manifest_digest"] == (
        validation.manifest_digest
    )
    assert torch.equal(expected["quantiles"], actual["quantiles"])
    assert torch.equal(expected["collision_logits"], actual["collision_logits"])
    with pytest.raises(RiskDataContractError, match="mode"):
        load_risk_checkpoint(path, expected_mode="production")

    legacy = tmp_path / "legacy.pt"
    torch.save({"checkpoint_layout_version": "risk_model_checkpoint_v1"}, legacy)
    with pytest.raises(RiskDataContractError, match="checkpoint_layout_version"):
        load_risk_checkpoint(legacy, expected_mode="toy")

    tampered_payload = torch.load(path, map_location="cpu")
    first_name = sorted(tampered_payload["model_state_dict"])[0]
    tampered_payload["model_state_dict"][first_name].reshape(-1)[0] += 1.0
    tampered = tmp_path / "tampered.pt"
    torch.save(tampered_payload, tampered)
    with pytest.raises(RiskDataContractError, match="model_state_digest_sha256"):
        load_risk_checkpoint(tampered, expected_mode="toy")

    semantic_payload = torch.load(path, map_location="cpu")
    semantic_payload["inference_parameters"]["quantile_levels"][0] = 0.4
    semantic_tampered = tmp_path / "semantic-tampered.pt"
    torch.save(semantic_payload, semantic_tampered)
    with pytest.raises(RiskDataContractError, match="checkpoint_semantic_digest_sha256"):
        load_risk_checkpoint(semantic_tampered, expected_mode="toy")


def test_toy_checkpoint_requires_exact_distinct_train_and_validation_provenance(
    tmp_path,
):
    train = make_toy_risk_dataset(split="train", count=4, seed=45, grid_size=12)
    validation = make_toy_risk_dataset(split="val", count=4, seed=45, grid_size=12)
    model = RiskModel(variant="r0", hidden_channels=8)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "channel_spec": _channel_spec(),
        "model_variant": "r0",
        "config_digest": "d" * 32,
        "toy_dataset_manifest_digest": train.manifest_digest,
        "validation_dataset_manifest_digest": validation.manifest_digest,
        "seed": 45,
    }

    missing_validation = dict(provenance)
    del missing_validation["validation_dataset_manifest_digest"]
    with pytest.raises(
        RiskDataContractError, match="validation_dataset_manifest_digest"
    ):
        save_risk_checkpoint(
            tmp_path / "missing-validation.pt",
            model=model,
            mode="toy",
            provenance=missing_validation,
        )

    with pytest.raises(RiskDataContractError, match="provenance keys"):
        save_risk_checkpoint(
            tmp_path / "extra-provenance.pt",
            model=model,
            mode="toy",
            provenance={**provenance, "unbound_extra": "value"},
        )

    with pytest.raises(RiskDataContractError, match="must be distinct"):
        save_risk_checkpoint(
            tmp_path / "same-publication.pt",
            model=model,
            mode="toy",
            provenance={
                **provenance,
                "validation_dataset_manifest_digest": train.manifest_digest,
            },
        )

    valid_path = tmp_path / "valid-provenance.pt"
    save_risk_checkpoint(
        valid_path, model=model, mode="toy", provenance=provenance
    )
    extra_payload = torch.load(valid_path, map_location="cpu")
    extra_payload["provenance"]["unbound_extra"] = "value"
    _rewrite_checkpoint_semantic_digest(extra_payload)
    extra_path = tmp_path / "loaded-extra-provenance.pt"
    torch.save(extra_payload, extra_path)
    with pytest.raises(RiskDataContractError, match="provenance keys"):
        load_risk_checkpoint(extra_path, expected_mode="toy")

    tampered_payload = torch.load(valid_path, map_location="cpu")
    tampered_payload["provenance"]["validation_dataset_manifest_digest"] = "f" * 32
    _rewrite_checkpoint_semantic_digest(tampered_payload)
    tampered_path = tmp_path / "tampered-validation-provenance.pt"
    torch.save(tampered_payload, tampered_path)
    with pytest.raises(
        RiskDataContractError, match="validation_dataset_manifest_digest"
    ):
        load_risk_checkpoint(
            tampered_path,
            expected_mode="toy",
            expected_provenance=provenance,
        )


def test_risk_checkpoint_loader_rejects_extra_top_level_fields(tmp_path):
    path = _save_contract_test_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu")
    payload["unbound_extra_field"] = "must-not-be-ignored"
    tampered_path = tmp_path / "extra-top-level-field.pt"
    torch.save(payload, tampered_path)

    with pytest.raises(RiskDataContractError, match="top-level keys"):
        load_risk_checkpoint(tampered_path, expected_mode="toy")


def test_risk_checkpoint_loader_binds_model_variant_across_sections(tmp_path):
    path = _save_contract_test_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu")
    payload["provenance"]["model_variant"] = "r1"
    _rewrite_checkpoint_semantic_digest(payload)
    tampered_path = tmp_path / "variant-mismatch.pt"
    torch.save(payload, tampered_path)

    with pytest.raises(RiskDataContractError, match="model_config.variant"):
        load_risk_checkpoint(tampered_path, expected_mode="toy")


@pytest.mark.parametrize(
    ("inference_parameters", "error_match"),
    (
        (
            {
                "quantile_levels": [0.5, 0.8, 0.9, 0.95],
                "collision_probability": "sigmoid_logit",
                "temperature": 1.0,
            },
            "inference_parameters keys",
        ),
        (
            {
                "quantile_levels": [0.4, 0.8, 0.9, 0.95],
                "collision_probability": "sigmoid_logit",
            },
            "quantile_levels",
        ),
        (
            {
                "quantile_levels": [0.5, 0.8, 0.9, 0.95],
                "collision_probability": "softmax",
            },
            "collision_probability",
        ),
    ),
)
def test_risk_checkpoint_loader_freezes_inference_parameters(
    tmp_path, inference_parameters, error_match
):
    path = _save_contract_test_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu")
    payload["inference_parameters"] = inference_parameters
    _rewrite_checkpoint_semantic_digest(payload)
    tampered_path = tmp_path / "invalid-inference.pt"
    torch.save(payload, tampered_path)

    with pytest.raises(RiskDataContractError, match=error_match):
        load_risk_checkpoint(tampered_path, expected_mode="toy")


def test_production_checkpoint_requires_all_real_provenance(tmp_path):
    model = RiskModel(variant="r0", hidden_channels=8)
    missing = {
        "schema_version": SCHEMA_VERSION,
        "channel_spec": _channel_spec(),
        "model_variant": "r0",
        "config_digest": "b" * 32,
    }
    with pytest.raises(RiskDataContractError, match="g1_split_manifest_digest"):
        save_risk_checkpoint(
            tmp_path / "bad-production.pt",
            model=model,
            mode="production",
            provenance=missing,
        )

    no_seed = {
        **missing,
        "g1_split_manifest_digest": "g1-real-digest",
        "risk_dataset_manifest_digest": "risk-v2-real-digest",
        "dynamic_objects_config_digest": "dynamic-real-digest",
        "target_type_policy_digest": "target-real-digest",
    }
    with pytest.raises(RiskDataContractError, match="seed"):
        save_risk_checkpoint(
            tmp_path / "bad-production-seed.pt",
            model=model,
            mode="production",
            provenance=no_seed,
        )
