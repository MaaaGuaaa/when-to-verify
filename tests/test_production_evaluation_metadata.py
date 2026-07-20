"""Production evaluation-record derivation at the oracle/renderer boundary."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from src.contracts import (
    HISTORY_CHANNELS,
    STATE_CHANNELS,
    TRAJECTORY_CHANNELS,
    BaseState,
    LocalTrajectory,
    OracleWorld,
    RiskSample,
)
from src.datasets.risk_dataset import RiskBuildInput
from src.datasets.risk_evaluation_metadata import (
    ALLOWED_OOD_TAGS,
    CONTACT_POLICY_RULE_VERSION,
    OOD_ROUTING_RULE_VERSION,
    PAIR_ELIGIBILITY_RULE_VERSION,
    RISK_EVALUATION_RECORD_LAYOUT_VERSION,
    ROBOT_FOOTPRINT_PROVENANCE_RULE_VERSION,
    derive_robot_footprint_provenance,
    derive_production_evaluation_record,
    validate_production_evaluation_record,
)
from src.generation.observation_renderer import RenderedObservation
from src.geometry import RectangleFootprint


TARGET_ID = "generated::human::critical"


def _boundary_fixture() -> tuple[RiskSample, RiskBuildInput, RenderedObservation]:
    height, width, history_steps, future_steps = 2, 3, 2, 2
    history = np.zeros(
        (history_steps, len(HISTORY_CHANNELS), height, width),
        dtype=np.float32,
    )
    visible = np.asarray(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32
    )
    dynamic = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    history[-1, HISTORY_CHANNELS.index("past_visible_mask")] = visible
    history[-1, HISTORY_CHANNELS.index("past_dynamic_occupancy")] = dynamic

    state = np.zeros((len(STATE_CHANNELS), height, width), dtype=np.float32)
    state[STATE_CHANNELS.index("current_unobservable_mask")] = 1.0 - visible
    state[STATE_CHANNELS.index("occlusion_age_map")] = np.asarray(
        [[0.0, 0.0, 0.2], [0.0, 0.6, 0.0]], dtype=np.float32
    )
    swept = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32
    )
    trajectory_channels = np.zeros(
        (len(TRAJECTORY_CHANNELS), height, width), dtype=np.float32
    )
    trajectory_channels[TRAJECTORY_CHANNELS.index("swept_volume_mask")] = swept

    base = BaseState(
        state_id="base-evaluation",
        split="test",
        recording_id="base-recording-evaluation",
        dynamic_object_ids=(),
        timestamp=1.0,
        robot_history=np.zeros((history_steps, 3), dtype=np.float32),
        robot_state=np.asarray([0.4, 0.0], dtype=np.float32),
        visible_dynamic_object_history={},
        visible_dynamic_object_specs={},
        static_map_local=np.zeros((height, width), dtype=np.float32),
        metadata={"session_id": "base-session-evaluation"},
    )
    trajectory = LocalTrajectory(
        trajectory_id="trajectory-evaluation",
        poses=np.zeros((future_steps, 3), dtype=np.float32),
        controls=np.zeros((future_steps, 2), dtype=np.float32),
        swept_mask=swept.copy(),
        tta_map=np.full((height, width), -1.0, dtype=np.float32),
        braking_map=np.zeros((height, width), dtype=np.float32),
        centerline_map=np.zeros((height, width), dtype=np.float32),
        task_cost=0.0,
    )
    target_spec = {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": 0.2},
    }
    world = OracleWorld(
        world_id="world-evaluation",
        base_state_id=base.state_id,
        static_occupancy=np.zeros((height, width), dtype=np.float32),
        dynamic_object_trajectories={
            TARGET_ID: np.zeros((future_steps, 3), dtype=np.float32)
        },
        dynamic_object_specs={TARGET_ID: target_spec},
        occluders=(),
        blind_spot_config={"kind": "environment"},
        random_seed=17,
    )
    provenance = {
        "base_recording_id": base.recording_id,
        "base_session_id": base.metadata["session_id"],
        "source_recording_id": "source-recording-evaluation",
        "source_session_id": "source-session-evaluation",
        "source_object_id": "source-recording-evaluation::human-7",
        "source_snippet_id": "snippet-evaluation",
        "seed_namespace": "sop07/test/seed-19/event-evaluation",
        "blind_spot_type": "environment",
    }
    source = RiskBuildInput(
        sample_id="test-evaluation-record",
        pair_group_id="pair-evaluation",
        event_type="collision",
        base_state=base,
        trajectory=trajectory,
        oracle_world=world,
        observed_static_occupancy=world.static_occupancy.copy(),
        scene_dynamic_history={},
        scene_dynamic_specs={},
        hidden_object_ids=(TARGET_ID,),
        sensor_config=None,
        provenance=provenance,
    )
    rendered = RenderedObservation(
        bev_history=history.copy(),
        state_channels=state.copy(),
        metadata={"renderer_layout_version": "test-renderer"},
    )
    sample = RiskSample(
        sample_id=source.sample_id,
        split=base.split,
        base_state_id=base.state_id,
        pair_group_id=source.pair_group_id,
        event_type=source.event_type,
        bev_history=history.copy(),
        state_channels=state.copy(),
        trajectory_channels=trajectory_channels,
        robot_state=base.robot_state.copy(),
        collision_label=1,
        risk_severity=1.0,
        min_clearance=-0.1,
        near_miss=0,
        first_collision_time=0.4,
        metadata={
            "trajectory_id": trajectory.trajectory_id,
            "provenance": dict(provenance),
            "label_audit": {
                "critical_object_id": TARGET_ID,
                "critical_object_type": "human",
                "has_hidden_target": True,
            },
        },
    )
    return sample, source, rendered


def _robot_provenance() -> dict[str, object]:
    return {
        "rule_version": ROBOT_FOOTPRINT_PROVENANCE_RULE_VERSION,
        "base_footprint_spec": {
            "kind": "rectangle",
            "length_m": 0.5,
            "width_m": 0.3,
        },
        "inflation_m": 0.1,
        "effective_footprint_spec": {
            "kind": "rectangle",
            "length_m": 0.7,
            "width_m": 0.5,
        },
        "base_config_digest_sha256": "a" * 64,
    }


def _derive() -> dict[str, object]:
    sample, source, rendered = _boundary_fixture()
    return derive_production_evaluation_record(
        sample=sample,
        source=source,
        rendered=rendered,
        robot_footprint=RectangleFootprint(0.7, 0.5),
        age_max_s=5.0,
        pair_eligible=True,
        ood_tag="heldout_motion",
        robot_footprint_provenance=_robot_provenance(),
        ood_evidence={
            "rule_version": OOD_ROUTING_RULE_VERSION,
            "source": "explicit_source_provenance",
            "reason": "source snippet belongs to the held-out motion cohort",
        },
    )


def test_derivation_hand_checks_critical_age_density_and_identity() -> None:
    record = _derive()

    assert record["risk_evaluation_record_layout_version"] == (
        RISK_EVALUATION_RECORD_LAYOUT_VERSION
    )
    assert record["critical_area_fraction"] == pytest.approx(2.0 / 6.0)
    assert record["age_s"] == pytest.approx(2.0)
    assert record["critical_region_empty"] is False
    assert record["density_fraction"] == pytest.approx(2.0 / 4.0)
    assert record["trajectory_id"] == "trajectory-evaluation"
    assert record["source_object_id"] == (
        "source-recording-evaluation::human-7"
    )
    assert record["critical_object_id"] == TARGET_ID
    assert record["target_object_type"] == "human"
    assert record["target_footprint_spec"] == {
        "kind": "circle",
        "radius_m": 0.2,
    }
    assert record["footprint_kind"] == "circle"
    assert record["pair_eligible"] is True
    assert record["pair_eligibility_rule_version"] == (
        PAIR_ELIGIBILITY_RULE_VERSION
    )
    assert record["contact_policy_rule_version"] == CONTACT_POLICY_RULE_VERSION


def test_validation_returns_a_deeply_frozen_exact_record() -> None:
    record = validate_production_evaluation_record(
        dict(_derive()), expected_sample_id="test-evaluation-record"
    )

    with pytest.raises(TypeError, match="frozen"):
        record["ood_tag"] = "map_stress"
    with pytest.raises(TypeError, match="frozen"):
        record["ood_evidence"]["reason"] = "tampered"
    with pytest.raises(TypeError, match="frozen"):
        record["target_footprint_spec"]["radius_m"] = 9.0

    extra = dict(record)
    extra["unexpected"] = "field"
    with pytest.raises(ValueError, match="keys"):
        validate_production_evaluation_record(extra)


def test_derivation_rejects_renderer_and_sample_shape_or_channel_mismatch() -> None:
    sample, source, rendered = _boundary_fixture()
    bad_rendered = replace(
        rendered, state_channels=rendered.state_channels[:, :, :-1]
    )

    with pytest.raises(ValueError, match="shape|join"):
        derive_production_evaluation_record(
            sample=sample,
            source=source,
            rendered=bad_rendered,
            robot_footprint=RectangleFootprint(0.7, 0.5),
            age_max_s=5.0,
            pair_eligible=True,
            ood_tag="heldout_motion",
            robot_footprint_provenance=_robot_provenance(),
            ood_evidence={
                "rule_version": OOD_ROUTING_RULE_VERSION,
                "source": "explicit_source_provenance",
                "reason": "held-out motion cohort",
            },
        )


def test_robot_footprint_provenance_is_recomputed_from_base_config() -> None:
    base_config = {
        "schema_version": "3.0.0",
        "bev": {"size": 2, "resolution_m": 0.5},
        "robot": {"length_m": 0.5, "width_m": 0.3, "inflation_m": 0.1},
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            base_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    provenance = derive_robot_footprint_provenance(
        base_config=base_config,
        effective_footprint=RectangleFootprint(0.7, 0.5),
    )

    assert provenance == {
        "rule_version": ROBOT_FOOTPRINT_PROVENANCE_RULE_VERSION,
        "base_footprint_spec": {
            "kind": "rectangle",
            "length_m": 0.5,
            "width_m": 0.3,
        },
        "inflation_m": 0.1,
        "effective_footprint_spec": {
            "kind": "rectangle",
            "length_m": 0.7,
            "width_m": 0.5,
        },
        "base_config_digest_sha256": expected_digest,
    }
    with pytest.raises(TypeError, match="frozen"):
        provenance["inflation_m"] = 0.0
    with pytest.raises(ValueError, match="effective footprint"):
        derive_robot_footprint_provenance(
            base_config=base_config,
            effective_footprint=RectangleFootprint(0.71, 0.5),
        )


def test_empty_target_uses_null_nested_fields_and_none_flat_kind() -> None:
    sample, source, rendered = _boundary_fixture()
    empty_world = replace(
        source.oracle_world,
        dynamic_object_trajectories={},
        dynamic_object_specs={},
    )
    empty_source = replace(
        source,
        event_type="empty_blind_spot",
        oracle_world=empty_world,
        hidden_object_ids=(),
    )
    metadata = {
        **sample.metadata,
        "label_audit": {
            "critical_object_id": None,
            "critical_object_type": None,
            "has_hidden_target": False,
        },
    }
    empty_sample = replace(
        sample,
        event_type="empty_blind_spot",
        collision_label=0,
        risk_severity=0.0,
        min_clearance=5.0,
        first_collision_time=None,
        metadata=metadata,
    )

    record = derive_production_evaluation_record(
        sample=empty_sample,
        source=empty_source,
        rendered=rendered,
        robot_footprint=RectangleFootprint(0.7, 0.5),
        age_max_s=5.0,
        pair_eligible=False,
        ood_tag="in_distribution",
        robot_footprint_provenance=_robot_provenance(),
        ood_evidence={
            "rule_version": OOD_ROUTING_RULE_VERSION,
            "source": "default_in_distribution",
            "reason": "no explicit OOD provenance was published",
        },
    )

    assert record["critical_object_id"] is None
    assert record["target_object_type"] is None
    assert record["target_footprint_spec"] is None
    assert record["footprint_kind"] == "none"
    assert record["pair_eligible"] is False


def test_ood_tags_are_frozen_provenance_routing_values() -> None:
    assert ALLOWED_OOD_TAGS == frozenset(
        {
            "in_distribution",
            "heldout_motion",
            "multi_object_context",
            "parameter_ood",
            "map_stress",
            "natural_real",
        }
    )
    record = dict(_derive())
    record["ood_tag"] = "unknown_ood"
    with pytest.raises(ValueError, match="allowed"):
        validate_production_evaluation_record(record)

    record = dict(_derive())
    record["ood_evidence"] = {
        "rule_version": OOD_ROUTING_RULE_VERSION,
        "source": "default_in_distribution",
        "reason": "no explicit provenance",
    }
    with pytest.raises(ValueError, match="only select in_distribution"):
        validate_production_evaluation_record(record)

    record = dict(_derive())
    record["ood_evidence"] = {
        "rule_version": OOD_ROUTING_RULE_VERSION,
        "source": "automatic_parameter_inference",
        "reason": "inferred from a parameter",
    }
    with pytest.raises(ValueError, match="source"):
        validate_production_evaluation_record(record)


def test_pair_eligibility_is_strictly_boolean() -> None:
    sample, source, rendered = _boundary_fixture()
    with pytest.raises(TypeError, match="pair_eligible"):
        derive_production_evaluation_record(
            sample=sample,
            source=source,
            rendered=rendered,
            robot_footprint=RectangleFootprint(0.7, 0.5),
            age_max_s=5.0,
            pair_eligible=1,
            ood_tag="heldout_motion",
            robot_footprint_provenance=_robot_provenance(),
            ood_evidence={
                "rule_version": OOD_ROUTING_RULE_VERSION,
                "source": "explicit_source_provenance",
                "reason": "held-out motion cohort",
            },
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("critical_area_fraction", float("nan")),
        ("age_s", float("inf")),
        ("density_fraction", float("-inf")),
        ("risk_severity", float("nan")),
    ),
)
def test_validation_rejects_nonfinite_record_values(
    field: str, value: float
) -> None:
    record = dict(_derive())
    record[field] = value
    with pytest.raises(ValueError, match="finite"):
        validate_production_evaluation_record(record)


def test_derivation_rejects_identity_tamper() -> None:
    sample, source, rendered = _boundary_fixture()
    with pytest.raises(ValueError, match="sample_id join"):
        derive_production_evaluation_record(
            sample=replace(sample, sample_id="tampered-sample"),
            source=source,
            rendered=rendered,
            robot_footprint=RectangleFootprint(0.7, 0.5),
            age_max_s=5.0,
            pair_eligible=True,
            ood_tag="heldout_motion",
            robot_footprint_provenance=_robot_provenance(),
            ood_evidence={
                "rule_version": OOD_ROUTING_RULE_VERSION,
                "source": "explicit_source_provenance",
                "reason": "held-out motion cohort",
            },
        )


def test_validation_rejects_label_tamper_that_claims_false_noncollision() -> None:
    record = dict(_derive())
    record["collision_label"] = 0
    record["first_collision_time"] = None

    with pytest.raises(ValueError, match="noncollision.*clearance"):
        validate_production_evaluation_record(record)


def test_empty_critical_region_has_zero_fraction_and_age() -> None:
    sample, source, rendered = _boundary_fixture()
    empty_swept = np.zeros_like(source.trajectory.swept_mask)
    trajectory_channels = sample.trajectory_channels.copy()
    trajectory_channels[
        TRAJECTORY_CHANNELS.index("swept_volume_mask")
    ] = empty_swept
    sample = replace(sample, trajectory_channels=trajectory_channels)
    source = replace(
        source,
        trajectory=replace(source.trajectory, swept_mask=empty_swept),
    )

    record = derive_production_evaluation_record(
        sample=sample,
        source=source,
        rendered=rendered,
        robot_footprint=RectangleFootprint(0.7, 0.5),
        age_max_s=5.0,
        pair_eligible=True,
        ood_tag="heldout_motion",
        robot_footprint_provenance=_robot_provenance(),
        ood_evidence={
            "rule_version": OOD_ROUTING_RULE_VERSION,
            "source": "explicit_source_provenance",
            "reason": "held-out motion cohort",
        },
    )

    assert record["critical_area_fraction"] == 0.0
    assert record["age_s"] == 0.0
    assert record["critical_region_empty"] is True


def test_density_derivation_fails_when_current_visible_region_is_empty() -> None:
    sample, source, rendered = _boundary_fixture()
    history = sample.bev_history.copy()
    history[-1, HISTORY_CHANNELS.index("past_visible_mask")].fill(0.0)
    history[-1, HISTORY_CHANNELS.index("past_dynamic_occupancy")].fill(0.0)
    state = sample.state_channels.copy()
    state[STATE_CHANNELS.index("current_unobservable_mask")].fill(1.0)
    sample = replace(sample, bev_history=history, state_channels=state)
    rendered = replace(
        rendered,
        bev_history=history.copy(),
        state_channels=state.copy(),
    )

    with pytest.raises(ValueError, match="at least one visible cell"):
        derive_production_evaluation_record(
            sample=sample,
            source=source,
            rendered=rendered,
            robot_footprint=RectangleFootprint(0.7, 0.5),
            age_max_s=5.0,
            pair_eligible=True,
            ood_tag="heldout_motion",
            robot_footprint_provenance=_robot_provenance(),
            ood_evidence={
                "rule_version": OOD_ROUTING_RULE_VERSION,
                "source": "explicit_source_provenance",
                "reason": "held-out motion cohort",
            },
        )
