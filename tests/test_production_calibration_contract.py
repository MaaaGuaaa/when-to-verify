"""Typed dataset-family contracts for production calibration and evaluation."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Callable

import pytest

from src.calibration.split_conformal import (
    CalibrationContractError,
    assert_calibration_artifact_test_isolation,
    calibration_artifact_semantic_digest,
    fit_calibration_artifact,
    prediction_table_cohort_digest,
    prediction_table_semantic_digest,
    validate_calibration_artifact,
    validate_prediction_table,
)
from src.datasets.risk_dataset_seal import (
    RISK_DATASET_FAMILY_LAYOUT_VERSION,
    LoadedRiskDataset,
    LoadedRiskDatasetFamily,
    load_risk_dataset_family,
    load_risk_dataset_seal,
    publish_risk_dataset_family,
    publish_risk_dataset_seal,
)
from src.datasets.toy_risk_learning import frozen_channel_spec
from tests.fixtures.formal_risk_publication import (
    create_formal_risk_publication,
)


MEMBER_ORDER = ("train", "calibration", "val", "test")


def _row(sample_id: str, split: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "split": split,
        "pair_group_id": f"pair-{sample_id}",
        "event_type": "collision",
        "recording_id": f"recording-{sample_id}",
        "session_id": f"session-{sample_id}",
        "source_object_id": f"object-{sample_id}",
        "snippet_id": f"snippet-{sample_id}",
        "base_state_id": f"base-{sample_id}",
        "seed_namespace": f"seed-{sample_id}",
        "trajectory_id": f"trajectory-{sample_id}",
        "occluder_id": f"occluder-{sample_id}",
        "background_id": f"background-{sample_id}",
        "collision_label": 1,
        "risk_severity": 0.8,
        "min_clearance": -0.05,
        "near_miss": 0,
        "first_collision_time": 0.2,
        "critical_object_id": f"critical-{sample_id}",
        "p_collision": 0.8,
        "q50": 0.2,
        "q80": 0.4,
        "q90": 0.7,
        "q95": 0.9,
        "blind_type": "corner",
        "critical_area_fraction": 0.04,
        "age_s": 0.8,
        "density_fraction": 0.02,
        "target_object_type": "human",
        "footprint_kind": "single_grid_cell_square",
        "footprint_dimensions_m": [0.2, 0.2],
        "robot_footprint_kind": "single_grid_cell_square",
        "robot_footprint_dimensions_m": [0.2, 0.2],
        "footprint_contact_policy": "positive_area_overlap",
        "ood_tag": "in_distribution",
        "pair_eligible": True,
    }


def _seal_member(root: Path, *, split: str, suffix: str = "a") -> LoadedRiskDataset:
    identity_prefixes = None
    if suffix != "a":
        identity_prefixes = {
            "sample_id": f"{split}-{suffix}-formal",
            "base_recording_id": f"{split}-{suffix}-base-recording",
            "source_recording_id": f"{split}-{suffix}-source-recording",
            "source_snippet_id": f"{split}-{suffix}-source-snippet",
            "pair_group_id": f"{split}-{suffix}-pair-group",
            "seed_namespace": f"sop07/{split}/{suffix}/formal",
        }
    publication = create_formal_risk_publication(
        root / f"upstream-{split}-{suffix}",
        split=split,
        handoff_dialect="legacy" if split == "train" else "heldout",
        identity_prefixes=identity_prefixes,
    )
    seal_root = publish_risk_dataset_seal(
        root / f"seal-{split}-{suffix}",
        collection_root=publication.collection_root,
        base_config_path=publication.base_config_path,
        split_provenance_path=publication.split_provenance_path,
        expected_split=split,
        expected_collection_handoff_sha256=publication.handoff_sha256,
    )
    return load_risk_dataset_seal(
        seal_root,
        collection_root=publication.collection_root,
        expected_split=split,
    )


@pytest.fixture(scope="module")
def production_families(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[LoadedRiskDatasetFamily, LoadedRiskDatasetFamily]:
    root = tmp_path_factory.mktemp("production-calibration-family")
    members_a = {split: _seal_member(root, split=split) for split in MEMBER_ORDER}
    family_a = load_risk_dataset_family(
        publish_risk_dataset_family(root / "family-a", members=members_a)
    )
    members_b = dict(members_a)
    members_b["calibration"] = _seal_member(
        root, split="calibration", suffix="b"
    )
    family_b = load_risk_dataset_family(
        publish_risk_dataset_family(root / "family-b", members=members_b)
    )
    assert family_a.risk_dataset_family_digest != (
        family_b.risk_dataset_family_digest
    )
    return family_a, family_b


def _production_table(
    family: LoadedRiskDatasetFamily,
    *,
    split: str,
    method_id: str = "risk-r0",
) -> dict[str, object]:
    common = family.manifest["common_contract"]
    assert isinstance(common, dict)
    family_channels = common["channel_spec"]
    assert isinstance(family_channels, dict)
    input_channels = {
        key: family_channels[key]
        for key in ("history", "state", "trajectory", "flat")
    }
    table: dict[str, object] = {
        "prediction_table_layout_version": "risk_prediction_table_v2",
        "mode": "production",
        "schema_version": common["schema_version"],
        "split": split,
        "method_id": method_id,
        "checkpoint_layout_version": "risk_model_checkpoint_v2",
        "checkpoint_digest": "d" * 64,
        "checkpoint_digest_kind": "risk_checkpoint_semantic_sha256",
        "risk_dataset_family_layout_version": RISK_DATASET_FAMILY_LAYOUT_VERSION,
        "risk_dataset_family_digest": family.risk_dataset_family_digest,
        "g1_split_manifest_digest": common["g1_split_manifest_digest"],
        "risk_dataset_manifest_digest": family.members[split][
            "risk_dataset_manifest_digest"
        ],
        "dynamic_objects_config_digest": common[
            "dynamic_objects_config_digest"
        ],
        "target_type_policy_digest": common["target_type_policy_digest"],
        "seed": 42,
        "channel_spec": json.loads(json.dumps(input_channels)),
        "config_digest_sha256": "c" * 64,
        "rows": [
            _row(f"{split}-formal-{index:03d}", split)
            for index in range(12)
        ],
    }
    table["cohort_digest_sha256"] = prediction_table_cohort_digest(table)
    table["semantic_digest"] = prediction_table_semantic_digest(table)
    return table


def _toy_table(*, split: str) -> dict[str, object]:
    table: dict[str, object] = {
        "prediction_table_layout_version": "risk_prediction_table_v2",
        "mode": "toy",
        "schema_version": "3.0.0",
        "split": split,
        "method_id": "toy-r0",
        "checkpoint_layout_version": "risk_model_checkpoint_v2",
        "checkpoint_digest": "d" * 64,
        "checkpoint_digest_kind": "risk_checkpoint_semantic_sha256",
        "toy_dataset_manifest_digest": "a" * 32,
        "seed": 42,
        "channel_spec": frozen_channel_spec(),
        "config_digest_sha256": "c" * 64,
        "rows": [_row(f"{split}-toy", split)],
    }
    table["cohort_digest_sha256"] = prediction_table_cohort_digest(table)
    table["semantic_digest"] = prediction_table_semantic_digest(table)
    return table


def _redigest_table(table: dict[str, object]) -> None:
    table["cohort_digest_sha256"] = prediction_table_cohort_digest(table)
    table["semantic_digest"] = prediction_table_semantic_digest(table)


def _production_artifact(
    family: LoadedRiskDatasetFamily,
) -> dict[str, object]:
    return fit_calibration_artifact(
        _production_table(family, split="calibration"),
        alpha=0.5,
        prediction_key="q90",
        dataset_family=family,
    )


def test_production_round_trip_binds_authenticated_family_and_split_members(
    production_families: tuple[LoadedRiskDatasetFamily, LoadedRiskDatasetFamily],
) -> None:
    family, _ = production_families
    calibration_table = _production_table(family, split="calibration")

    validated_table = validate_prediction_table(
        calibration_table,
        expected_mode="production",
        expected_split="calibration",
        dataset_family=family,
    )
    artifact = fit_calibration_artifact(
        validated_table,
        alpha=0.5,
        prediction_key="q90",
        dataset_family=family,
    )
    validated_artifact = validate_calibration_artifact(
        artifact,
        expected_mode="production",
        dataset_family=family,
    )
    test_table = validate_prediction_table(
        _production_table(family, split="test"),
        expected_mode="production",
        expected_split="test",
        dataset_family=family,
    )
    isolation = assert_calibration_artifact_test_isolation(
        validated_artifact,
        test_table["rows"],
        dataset_family=family,
    )

    assert validated_table["risk_dataset_manifest_digest"] == family.members[
        "calibration"
    ]["risk_dataset_manifest_digest"]
    assert validated_artifact["risk_dataset_family_layout_version"] == (
        "risk_dataset_family_v1"
    )
    assert validated_artifact["risk_dataset_family_digest"] == (
        family.risk_dataset_family_digest
    )
    assert validated_artifact["risk_dataset_manifest_digest"] == family.members[
        "calibration"
    ]["risk_dataset_manifest_digest"]
    assert test_table["risk_dataset_manifest_digest"] == family.members["test"][
        "risk_dataset_manifest_digest"
    ]
    assert isolation["base_session_id"] == 1
    assert isolation["source_session_id"] == 1
    assert all(
        isolation[field] == 0
        for field in (
            "sample_id",
            "base_recording_id",
            "source_recording_id",
            "base_source_cross_role_recording_id",
            "source_snippet_id",
            "pair_group_id",
            "seed_namespace",
        )
    )


def test_production_apis_require_reauthenticated_typed_family(
    production_families: tuple[LoadedRiskDatasetFamily, LoadedRiskDatasetFamily],
) -> None:
    family, _ = production_families
    table = _production_table(family, split="calibration")
    artifact = _production_artifact(family)
    test_rows = _production_table(family, split="test")["rows"]
    assert isinstance(test_rows, list)
    forged = replace(family, risk_dataset_family_digest="f" * 64)
    calls: tuple[Callable[[object], object], ...] = (
        lambda supplied: validate_prediction_table(
            table, dataset_family=supplied  # type: ignore[arg-type]
        ),
        lambda supplied: fit_calibration_artifact(
            table,
            alpha=0.5,
            prediction_key="q90",
            dataset_family=supplied,  # type: ignore[arg-type]
        ),
        lambda supplied: validate_calibration_artifact(
            artifact, dataset_family=supplied  # type: ignore[arg-type]
        ),
        lambda supplied: assert_calibration_artifact_test_isolation(
            artifact,
            test_rows,
            dataset_family=supplied,  # type: ignore[arg-type]
        ),
    )

    for call in calls:
        with pytest.raises(CalibrationContractError, match="production.*fail-closed"):
            call(None)
        with pytest.raises(
            CalibrationContractError, match="authenticated LoadedRiskDatasetFamily"
        ):
            call(dict(family.manifest))
        with pytest.raises(
            CalibrationContractError, match="forged|stale|reauthentication"
        ):
            call(forged)


def test_production_rejects_cross_family_and_split_member_mismatch(
    production_families: tuple[LoadedRiskDatasetFamily, LoadedRiskDatasetFamily],
) -> None:
    family_a, family_b = production_families
    table = _production_table(family_a, split="calibration")
    with pytest.raises(CalibrationContractError, match="family digest mismatch"):
        validate_prediction_table(table, dataset_family=family_b)

    mismatched_table = _production_table(family_a, split="test")
    mismatched_table["risk_dataset_manifest_digest"] = family_a.members[
        "calibration"
    ]["risk_dataset_manifest_digest"]
    _redigest_table(mismatched_table)
    with pytest.raises(CalibrationContractError, match="member digest mismatch.*test"):
        validate_prediction_table(mismatched_table, dataset_family=family_a)

    wrong_membership = _production_table(family_a, split="test")
    wrong_membership["rows"][0]["sample_id"] = "test-not-a-family-member"
    _redigest_table(wrong_membership)
    with pytest.raises(CalibrationContractError, match="sample ID membership"):
        validate_prediction_table(wrong_membership, dataset_family=family_a)

    artifact = _production_artifact(family_a)
    with pytest.raises(CalibrationContractError, match="family digest mismatch"):
        validate_calibration_artifact(artifact, dataset_family=family_b)

    artifact["risk_dataset_manifest_digest"] = family_a.members["test"][
        "risk_dataset_manifest_digest"
    ]
    artifact["semantic_digest"] = calibration_artifact_semantic_digest(artifact)
    with pytest.raises(
        CalibrationContractError, match="member digest mismatch.*calibration"
    ):
        validate_calibration_artifact(artifact, dataset_family=family_a)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("g1_split_manifest_digest", "0" * 32, "g1_split_manifest_digest"),
        ("dynamic_objects_config_digest", "0" * 64, "dynamic_objects_config_digest"),
        ("target_type_policy_digest", "0" * 32, "target_type_policy_digest"),
        ("schema_version", "3.0.1", "schema_version"),
        ("channel_spec", {"flat": ["drifted"]}, "channel_spec"),
    ],
)
def test_production_table_rejects_family_common_contract_drift(
    production_families: tuple[LoadedRiskDatasetFamily, LoadedRiskDatasetFamily],
    field: str,
    replacement: object,
    message: str,
) -> None:
    family, _ = production_families
    table = _production_table(family, split="calibration")
    table[field] = replacement
    _redigest_table(table)

    with pytest.raises(CalibrationContractError, match=message):
        validate_prediction_table(table, dataset_family=family)


def test_toy_apis_reject_irrelevant_dataset_family(
    production_families: tuple[LoadedRiskDatasetFamily, LoadedRiskDatasetFamily],
) -> None:
    family, _ = production_families
    table = _toy_table(split="calibration")
    artifact = fit_calibration_artifact(
        table,
        alpha=0.5,
        prediction_key="q90",
    )
    test_rows = _toy_table(split="test")["rows"]
    assert isinstance(test_rows, list)
    calls: tuple[Callable[[], object], ...] = (
        lambda: validate_prediction_table(table, dataset_family=family),
        lambda: fit_calibration_artifact(
            table,
            alpha=0.5,
            prediction_key="q90",
            dataset_family=family,
        ),
        lambda: validate_calibration_artifact(
            artifact, dataset_family=family
        ),
        lambda: assert_calibration_artifact_test_isolation(
            artifact,
            test_rows,
            dataset_family=family,
        ),
    )

    for call in calls:
        with pytest.raises(
            CalibrationContractError, match="toy.*must not.*dataset family"
        ):
            call()
