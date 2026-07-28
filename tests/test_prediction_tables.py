"""Shared production prediction cohort and protocol contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.calibration.grouped_calibration import GROUP_DIMENSIONS
from src.calibration.split_conformal import (
    assert_calibration_artifact_test_isolation,
    fit_calibration_artifact,
    prediction_table_semantic_digest,
    validate_calibration_artifact,
)
from src.datasets.risk_dataset_seal import (
    load_risk_dataset_family,
    load_risk_dataset_seal,
    publish_risk_dataset_family,
    publish_risk_dataset_seal,
)
from src.datasets.risk_evaluation_store import (
    publish_risk_evaluation_collection,
    load_risk_evaluation_collection,
)
from src.datasets.shard_writer import load_risk_shard
from src.evaluation.prediction_tables import (
    BASELINE_SPEC_LAYOUT_VERSION,
    PredictionMethodArtifact,
    PredictionTableContractError,
    baseline_spec_digest,
    build_prediction_protocol,
    build_production_prediction_table,
    load_unified_prediction_collection,
    publish_unified_prediction_collection,
    validate_prediction_protocol,
    validate_shared_prediction_tables,
)
from src.training.occupancy_trainer import (
    FORMAL_PRODUCTION_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
)
from tests.fixtures.formal_risk_publication import create_formal_risk_publication
from tests.test_risk_evaluation_store import _record


def _family(root: Path, *, shared_sessions: bool = False):
    members = {}
    for split in ("train", "calibration", "val", "test"):
        publication = create_formal_risk_publication(
            root / f"upstream-{split}",
            split=split,
            handoff_dialect="legacy" if split == "train" else "heldout",
            identity_prefixes={
                "sample_id": f"{split}-prediction",
                "base_recording_id": f"{split}-base-recording",
                "source_recording_id": f"{split}-source-recording",
                "source_snippet_id": f"{split}-source-snippet",
                "pair_group_id": f"{split}-pair",
                "seed_namespace": f"prediction/{split}",
            },
            base_session_id=(
                "shared-base-session" if shared_sessions else f"{split}-base-session"
            ),
            source_session_id=(
                "shared-source-session"
                if shared_sessions
                else f"{split}-source-session"
            ),
        )
        seal = publish_risk_dataset_seal(
            root / f"seal-{split}",
            collection_root=publication.collection_root,
            base_config_path=publication.base_config_path,
            split_provenance_path=publication.split_provenance_path,
            expected_split=split,
            expected_collection_handoff_sha256=publication.handoff_sha256,
        )
        members[split] = load_risk_dataset_seal(
            seal,
            collection_root=publication.collection_root,
            expected_split=split,
        )
    family = load_risk_dataset_family(
        publish_risk_dataset_family(root / "family", members=members)
    )
    return family, members


def _evaluation(root: Path, dataset):
    records = {
        descriptor.shard_index: tuple(
            _record(sample)
            for sample in load_risk_shard(
                dataset.collection_root / descriptor.relative_root,
                grid=dataset.grid,
            ).samples
        )
        for descriptor in dataset.shards
    }
    output = publish_risk_evaluation_collection(
        root,
        dataset=dataset,
        records_by_shard=records,
    )
    return load_risk_evaluation_collection(output, dataset=dataset)


def _values(count: int, offset: float = 0.0) -> dict[str, np.ndarray]:
    probability = np.linspace(0.1 + offset, 0.7 + offset, count, dtype=np.float64)
    probability = np.clip(probability, 0.0, 1.0)
    return {
        "p_collision": probability,
        "quantiles": np.stack(
            (
                probability * 0.5,
                probability * 0.7,
                probability * 0.9,
                probability,
            ),
            axis=1,
        ),
    }


def _artifacts() -> dict[str, PredictionMethodArtifact]:
    baseline_kwargs = {
        "b2_tau_s": 2.0,
        "b2_a_max_s": 5.0,
        "sigma_time_s": 2.0,
    }
    return {
        "risk-r0": PredictionMethodArtifact(
            method_id="risk-r0",
            layout_version="risk_model_checkpoint_v2",
            digest_sha256="0" * 64,
            digest_kind="risk_checkpoint_semantic_sha256",
            score_definition="risk_model_collision_and_quantile_heads",
        ),
        "risk-r1": PredictionMethodArtifact(
            method_id="risk-r1",
            layout_version="risk_model_checkpoint_v2",
            digest_sha256="1" * 64,
            digest_kind="risk_checkpoint_semantic_sha256",
            score_definition="risk_model_collision_and_quantile_heads",
        ),
        "B1": PredictionMethodArtifact(
            method_id="B1",
            layout_version=BASELINE_SPEC_LAYOUT_VERSION,
            digest_sha256=baseline_spec_digest("B1", **baseline_kwargs),
            digest_kind="baseline_spec_sha256",
            score_definition="normalized_weighted_sum",
        ),
        "B2": PredictionMethodArtifact(
            method_id="B2",
            layout_version=BASELINE_SPEC_LAYOUT_VERSION,
            digest_sha256=baseline_spec_digest("B2", **baseline_kwargs),
            digest_kind="baseline_spec_sha256",
            score_definition="normalized_weighted_sum",
        ),
        "B3": PredictionMethodArtifact(
            method_id="B3",
            layout_version=FORMAL_PRODUCTION_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
            digest_sha256="3" * 64,
            digest_kind="formal_occupancy_checkpoint_semantic_sha256",
            score_definition="normalized_weighted_sum",
        ),
        "B4": PredictionMethodArtifact(
            method_id="B4",
            layout_version=FORMAL_PRODUCTION_OCCUPANCY_CHECKPOINT_LAYOUT_VERSION,
            digest_sha256="3" * 64,
            digest_kind="formal_occupancy_checkpoint_semantic_sha256",
            score_definition="learned_occupancy_aggregator",
        ),
    }


def test_protocol_round_trip_binds_all_calibration_settings() -> None:
    protocol = build_prediction_protocol(
        alpha=0.1,
        prediction_key="q90",
        min_group_size=20,
        group_dimensions=GROUP_DIMENSIONS,
    )

    validated = validate_prediction_protocol(protocol)
    assert validated["alpha"] == 0.1
    assert validated["prediction_key"] == "q90"
    assert validated["grouped_calibration"]["group_dimensions"] == list(
        GROUP_DIMENSIONS
    )
    assert len(validated["protocol_digest_sha256"]) == 64

    drifted = dict(protocol)
    drifted["alpha"] = 0.2
    with pytest.raises(PredictionTableContractError, match="digest"):
        validate_prediction_protocol(drifted)


def test_six_methods_share_exact_evaluation_rows_and_protocol(tmp_path: Path) -> None:
    family, members = _family(tmp_path)
    dataset = members["calibration"]
    evaluation = _evaluation(tmp_path / "evaluation-calibration", dataset)
    protocol = build_prediction_protocol(
        alpha=0.1,
        prediction_key="q90",
        min_group_size=20,
    )
    tables = {
        method: build_production_prediction_table(
            dataset_family=family,
            dataset=dataset,
            evaluation_records=evaluation,
            occupancy_sidecar_collection_digest_sha256="a" * 64,
            method_artifact=artifact,
            predictions=_values(dataset.sample_count, offset=index * 0.01),
            protocol=protocol,
            seed=42,
            config_digest_sha256="c" * 64,
        )
        for index, (method, artifact) in enumerate(_artifacts().items())
    }

    validated = validate_shared_prediction_tables(
        tables,
        dataset_family=family,
        dataset=dataset,
        evaluation_records=evaluation,
        expected_protocol=protocol,
        expected_sidecar_collection_digest_sha256="a" * 64,
    )
    ordered_ids = {
        tuple(row["sample_id"] for row in table["rows"])
        for table in validated.values()
    }
    cohorts = {table["cohort_digest_sha256"] for table in validated.values()}
    protocols = {
        table["prediction_protocol_digest_sha256"]
        for table in validated.values()
    }
    evaluation_rows = {
        tuple(
            tuple(
                sorted(
                    (key, repr(value))
                    for key, value in row.items()
                    if key not in {"p_collision", "q50", "q80", "q90", "q95"}
                )
            )
            for row in table["rows"]
        )
        for table in validated.values()
    }
    assert len(ordered_ids) == 1
    assert len(cohorts) == 1
    assert len(protocols) == 1
    assert len(evaluation_rows) == 1
    assert validated["B1"]["checkpoint_layout_version"] == BASELINE_SPEC_LAYOUT_VERSION
    assert validated["B1"]["checkpoint_digest_kind"] == "baseline_spec_sha256"


def test_shared_table_validation_rejects_cohort_protocol_or_provenance_drift(
    tmp_path: Path,
) -> None:
    family, members = _family(tmp_path)
    dataset = members["test"]
    evaluation = _evaluation(tmp_path / "evaluation-test", dataset)
    protocol = build_prediction_protocol(
        alpha=0.1,
        prediction_key="q90",
        min_group_size=20,
    )
    artifacts = _artifacts()
    tables = {
        method: build_production_prediction_table(
            dataset_family=family,
            dataset=dataset,
            evaluation_records=evaluation,
            occupancy_sidecar_collection_digest_sha256="a" * 64,
            method_artifact=artifact,
            predictions=_values(dataset.sample_count),
            protocol=protocol,
            seed=42,
            config_digest_sha256="c" * 64,
        )
        for method, artifact in artifacts.items()
    }

    tables["B4"] = {**tables["B4"], "prediction_protocol_digest_sha256": "f" * 64}
    with pytest.raises(PredictionTableContractError, match="protocol"):
        validate_shared_prediction_tables(
            tables,
            dataset_family=family,
            dataset=dataset,
            evaluation_records=evaluation,
            expected_protocol=protocol,
            expected_sidecar_collection_digest_sha256="a" * 64,
        )


def test_shared_table_validation_rejects_nonformal_b3_artifact(
    tmp_path: Path,
) -> None:
    family, members = _family(tmp_path)
    dataset = members["test"]
    evaluation = _evaluation(tmp_path / "evaluation-test", dataset)
    protocol = build_prediction_protocol(
        alpha=0.1,
        prediction_key="q90",
        min_group_size=20,
    )
    tables = {
        method: build_production_prediction_table(
            dataset_family=family,
            dataset=dataset,
            evaluation_records=evaluation,
            occupancy_sidecar_collection_digest_sha256="a" * 64,
            method_artifact=artifact,
            predictions=_values(dataset.sample_count),
            protocol=protocol,
            seed=42,
            config_digest_sha256="c" * 64,
        )
        for method, artifact in _artifacts().items()
    }
    tables["B3"]["checkpoint_layout_version"] = "occupancy_baseline_checkpoint_v2"
    tables["B3"]["checkpoint_digest_kind"] = (
        "occupancy_checkpoint_semantic_sha256"
    )
    tables["B3"]["semantic_digest"] = prediction_table_semantic_digest(
        tables["B3"]
    )

    with pytest.raises(PredictionTableContractError, match="artifact contract"):
        validate_shared_prediction_tables(
            tables,
            dataset_family=family,
            dataset=dataset,
            evaluation_records=evaluation,
            expected_protocol=protocol,
            expected_sidecar_collection_digest_sha256="a" * 64,
        )


def test_calibration_artifact_inherits_shared_protocol_and_evaluation_binding(
    tmp_path: Path,
) -> None:
    family, members = _family(tmp_path)
    dataset = members["calibration"]
    evaluation = _evaluation(tmp_path / "evaluation-calibration", dataset)
    protocol = build_prediction_protocol(
        alpha=0.1,
        prediction_key="q90",
        min_group_size=20,
    )
    table = build_production_prediction_table(
        dataset_family=family,
        dataset=dataset,
        evaluation_records=evaluation,
        occupancy_sidecar_collection_digest_sha256="a" * 64,
        method_artifact=_artifacts()["B1"],
        predictions=_values(dataset.sample_count),
        protocol=protocol,
        seed=42,
        config_digest_sha256="c" * 64,
    )

    artifact = fit_calibration_artifact(
        table,
        alpha=0.1,
        prediction_key="q90",
        dataset_family=family,
    )
    validated = validate_calibration_artifact(
        artifact,
        expected_mode="production",
        dataset_family=family,
    )

    assert validated["prediction_protocol_digest_sha256"] == protocol[
        "protocol_digest_sha256"
    ]
    assert validated["evaluation_record_collection_digest_sha256"] == (
        evaluation.collection_semantic_digest_sha256
    )
    assert validated["occupancy_sidecar_collection_digest_sha256"] == "a" * 64
    assert set(validated["fitted_identities"]) == {
        "sample_id",
        "pair_group_id",
        "base_recording_id",
        "base_session_id",
        "source_recording_id",
        "source_session_id",
        "source_object_id",
        "source_snippet_id",
        "base_state_id",
        "seed_namespace",
    }


def test_prediction_collection_round_trips_calibration_then_complete(
    tmp_path: Path,
) -> None:
    family, members = _family(tmp_path)
    protocol = build_prediction_protocol(
        alpha=0.1,
        prediction_key="q90",
        min_group_size=20,
    )
    evaluations = {
        split: _evaluation(tmp_path / f"evaluation-{split}", members[split])
        for split in ("calibration", "test")
    }
    split_sources = {
        split: {
            "dataset": members[split],
            "evaluation_records": evaluations[split],
            "occupancy_sidecar_collection_digest_sha256": "a" * 64,
        }
        for split in ("calibration", "test")
    }
    tables = {
        split: {
            method: build_production_prediction_table(
                dataset_family=family,
                dataset=members[split],
                evaluation_records=evaluations[split],
                occupancy_sidecar_collection_digest_sha256="a" * 64,
                method_artifact=artifact,
                predictions=_values(members[split].sample_count),
                protocol=protocol,
                seed=42,
                config_digest_sha256="c" * 64,
            )
            for method, artifact in _artifacts().items()
        }
        for split in ("calibration", "test")
    }

    calibration_root = publish_unified_prediction_collection(
        tmp_path / "calibration-predictions",
        dataset_family=family,
        protocol=protocol,
        split_sources={"calibration": split_sources["calibration"]},
        tables_by_split={"calibration": tables["calibration"]},
    )
    calibration = load_unified_prediction_collection(
        calibration_root,
        dataset_family=family,
        split_sources={"calibration": split_sources["calibration"]},
    )
    assert calibration.manifest["publication_stage"] == "calibration"

    complete_root = publish_unified_prediction_collection(
        tmp_path / "complete-predictions",
        dataset_family=family,
        protocol=protocol,
        split_sources=split_sources,
        tables_by_split=tables,
    )
    complete = load_unified_prediction_collection(
        complete_root,
        dataset_family=family,
        split_sources=split_sources,
    )
    assert complete.manifest["publication_stage"] == "complete"
    assert set(complete.tables_by_split) == {"calibration", "test"}
    assert all(
        set(methods) == {"risk-r0", "risk-r1", "B1", "B2", "B3", "B4"}
        for methods in complete.tables_by_split.values()
    )


def test_production_isolation_allows_family_approved_session_overlap(
    tmp_path: Path,
) -> None:
    family, members = _family(tmp_path, shared_sessions=True)
    protocol = build_prediction_protocol(
        alpha=0.1,
        prediction_key="q90",
        min_group_size=20,
    )
    evaluations = {
        split: _evaluation(tmp_path / f"evaluation-{split}", members[split])
        for split in ("calibration", "test")
    }
    split_sources = {
        split: {
            "dataset": members[split],
            "evaluation_records": evaluations[split],
            "occupancy_sidecar_collection_digest_sha256": "a" * 64,
        }
        for split in ("calibration", "test")
    }
    tables = {
        split: {
            method: build_production_prediction_table(
                dataset_family=family,
                dataset=members[split],
                evaluation_records=evaluations[split],
                occupancy_sidecar_collection_digest_sha256="a" * 64,
                method_artifact=artifact,
                predictions=_values(members[split].sample_count),
                protocol=protocol,
                seed=42,
                config_digest_sha256="c" * 64,
            )
            for method, artifact in _artifacts().items()
        }
        for split in ("calibration", "test")
    }

    output = publish_unified_prediction_collection(
        tmp_path / "complete-predictions-shared-sessions",
        dataset_family=family,
        protocol=protocol,
        split_sources=split_sources,
        tables_by_split=tables,
    )
    artifact = fit_calibration_artifact(
        tables["calibration"]["risk-r0"],
        alpha=0.1,
        prediction_key="q90",
        dataset_family=family,
    )
    isolation = assert_calibration_artifact_test_isolation(
        artifact,
        tables["test"]["risk-r0"]["rows"],
        dataset_family=family,
    )

    assert output.is_dir()
    assert isolation["base_session_id"] == 1
    assert isolation["source_session_id"] == 1
