"""Immutable production-only evaluation records.

The functions in this module belong at the oracle/renderer boundary.  Their
outputs are authenticated analysis metadata and must never be attached to a
``RiskSample`` or exposed as model inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np

from src.contracts import (
    HISTORY_CHANNELS,
    POSE_TIME_LAYOUT_VERSION,
    SCHEMA_VERSION,
    STATE_CHANNELS,
    TRAJECTORY_CHANNELS,
    RiskSample,
)
from src.generation.observation_renderer import RenderedObservation
from src.generation.risk_gt import RiskGroundTruth
from src.geometry import (
    CircleFootprint,
    Footprint,
    RectangleFootprint,
    inflate_footprint,
)

if TYPE_CHECKING:
    from src.datasets.risk_dataset import RiskBuildInput


RISK_EVALUATION_RECORD_LAYOUT_VERSION = "risk_evaluation_record_v1"
OOD_ROUTING_RULE_VERSION = "risk_evaluation_ood_provenance_routing_v1"
PAIR_ELIGIBILITY_RULE_VERSION = "risk_evaluation_pair_eligibility_v1"
ROBOT_FOOTPRINT_PROVENANCE_RULE_VERSION = (
    "risk_evaluation_robot_footprint_provenance_v1"
)
CONTACT_POLICY_RULE_VERSION = "risk_evaluation_signed_contact_policy_v1"

ALLOWED_OOD_TAGS = frozenset(
    {
        "in_distribution",
        "heldout_motion",
        "multi_object_context",
        "parameter_ood",
        "map_stress",
        "natural_real",
    }
)
_OOD_EVIDENCE_SOURCES = frozenset(
    {"explicit_source_provenance", "default_in_distribution"}
)

_IDENTITY_STRING_FIELDS = (
    "sample_id",
    "split",
    "base_state_id",
    "pair_group_id",
    "event_type",
    "trajectory_id",
    "base_recording_id",
    "base_session_id",
    "source_recording_id",
    "source_session_id",
    "source_object_id",
    "source_snippet_id",
    "seed_namespace",
)
_RECORD_KEYS = frozenset(
    {
        "risk_evaluation_record_layout_version",
        *_IDENTITY_STRING_FIELDS,
        "collision_label",
        "risk_severity",
        "min_clearance",
        "near_miss",
        "first_collision_time",
        "blind_type",
        "critical_area_fraction",
        "age_s",
        "critical_region_empty",
        "density_fraction",
        "critical_object_id",
        "target_object_type",
        "target_footprint_spec",
        "footprint_kind",
        "pair_eligible",
        "pair_eligibility_rule_version",
        "ood_tag",
        "ood_evidence",
        "robot_footprint_spec",
        "robot_footprint_provenance",
        "contact_policy_rule_version",
    }
)
_ROBOT_PROVENANCE_KEYS = frozenset(
    {
        "rule_version",
        "base_footprint_spec",
        "inflation_m",
        "effective_footprint_spec",
        "base_config_digest_sha256",
    }
)
_OOD_EVIDENCE_KEYS = frozenset({"rule_version", "source", "reason"})


class _FrozenDict(dict[str, object]):
    """JSON-shaped ``dict`` whose nested identity cannot be mutated."""

    def __init__(self, value: Mapping[str, object]) -> None:
        dict.__init__(
            self,
            ((key, _deep_freeze(child)) for key, child in value.items()),
        )

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("frozen evaluation record cannot be mutated")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenDict:
        memo[id(self)] = self
        return self


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenDict({str(key): child for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _require_nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _binary_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer in {{0, 1}}")
    result = int(value)
    if result not in (0, 1):
        raise ValueError(f"{name} must be in {{0, 1}}")
    return result


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be bool")
    return bool(value)


def _canonical_footprint_mapping(
    value: object, *, name: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    kind = value.get("kind")
    if kind == "circle":
        expected = {"kind", "radius_m"}
        if set(value) != expected:
            raise ValueError(f"{name} keys must be exactly {sorted(expected)}")
        radius = _finite_float(value["radius_m"], name=f"{name}.radius_m")
        if radius <= 0.0:
            raise ValueError(f"{name}.radius_m must be positive")
        return {"kind": "circle", "radius_m": radius}
    if kind == "rectangle":
        expected = {"kind", "length_m", "width_m"}
        if set(value) != expected:
            raise ValueError(f"{name} keys must be exactly {sorted(expected)}")
        length = _finite_float(value["length_m"], name=f"{name}.length_m")
        width = _finite_float(value["width_m"], name=f"{name}.width_m")
        if length <= 0.0 or width <= 0.0:
            raise ValueError(f"{name} dimensions must be positive")
        return {"kind": "rectangle", "length_m": length, "width_m": width}
    raise ValueError(f"{name}.kind must be circle or rectangle")


def footprint_to_canonical_spec(footprint: Footprint) -> dict[str, object]:
    """Convert a supported geometry footprint to its canonical JSON mapping."""

    if isinstance(footprint, CircleFootprint):
        return {"kind": "circle", "radius_m": float(footprint.radius_m)}
    if isinstance(footprint, RectangleFootprint):
        return {
            "kind": "rectangle",
            "length_m": float(footprint.length_m),
            "width_m": float(footprint.width_m),
        }
    raise TypeError("footprint must be a CircleFootprint or RectangleFootprint")


def _footprint_from_canonical_spec(value: Mapping[str, object]) -> Footprint:
    if value["kind"] == "circle":
        return CircleFootprint(value["radius_m"])
    return RectangleFootprint(value["length_m"], value["width_m"])


def _validate_robot_footprint_provenance(
    value: object, *, expected_effective: Footprint | None = None
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _ROBOT_PROVENANCE_KEYS:
        raise ValueError(
            "robot_footprint_provenance keys violate the frozen contract"
        )
    if value.get("rule_version") != ROBOT_FOOTPRINT_PROVENANCE_RULE_VERSION:
        raise ValueError("robot footprint provenance rule version mismatch")
    base = _canonical_footprint_mapping(
        value.get("base_footprint_spec"), name="base_footprint_spec"
    )
    effective = _canonical_footprint_mapping(
        value.get("effective_footprint_spec"), name="effective_footprint_spec"
    )
    inflation = _finite_float(value.get("inflation_m"), name="inflation_m")
    if inflation < 0.0:
        raise ValueError("inflation_m must be non-negative")
    recomputed = footprint_to_canonical_spec(
        inflate_footprint(_footprint_from_canonical_spec(base), inflation)
    )
    if recomputed != effective:
        raise ValueError("effective footprint does not match base plus inflation")
    if expected_effective is not None and (
        footprint_to_canonical_spec(expected_effective) != effective
    ):
        raise ValueError("effective footprint differs from robot_footprint")
    digest = value.get("base_config_digest_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("base_config_digest_sha256 must be 64 lowercase hex")
    return {
        "rule_version": ROBOT_FOOTPRINT_PROVENANCE_RULE_VERSION,
        "base_footprint_spec": base,
        "inflation_m": inflation,
        "effective_footprint_spec": effective,
        "base_config_digest_sha256": digest,
    }


def derive_robot_footprint_provenance(
    *,
    base_config: Mapping[str, object],
    effective_footprint: Footprint,
) -> dict[str, object]:
    """Bind the effective robot footprint to its complete base configuration."""

    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    robot = base_config.get("robot")
    if not isinstance(robot, Mapping):
        raise TypeError("base_config.robot must be a mapping")
    base_spec = _canonical_footprint_mapping(
        {
            "kind": "rectangle",
            "length_m": robot.get("length_m"),
            "width_m": robot.get("width_m"),
        },
        name="base_footprint_spec",
    )
    inflation = _finite_float(robot.get("inflation_m"), name="inflation_m")
    if inflation < 0.0:
        raise ValueError("inflation_m must be non-negative")
    try:
        canonical_config = json.dumps(
            base_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("base_config must be finite canonical JSON") from exc
    provenance = {
        "rule_version": ROBOT_FOOTPRINT_PROVENANCE_RULE_VERSION,
        "base_footprint_spec": base_spec,
        "inflation_m": inflation,
        "effective_footprint_spec": footprint_to_canonical_spec(
            effective_footprint
        ),
        "base_config_digest_sha256": hashlib.sha256(
            canonical_config.encode("utf-8")
        ).hexdigest(),
    }
    normalized = _validate_robot_footprint_provenance(
        provenance, expected_effective=effective_footprint
    )
    return _FrozenDict(normalized)


def _validate_ood(
    tag: object, evidence: object
) -> tuple[str, dict[str, object]]:
    if not isinstance(tag, str) or tag not in ALLOWED_OOD_TAGS:
        raise ValueError("ood_tag is not in the frozen allowed set")
    if not isinstance(evidence, Mapping) or set(evidence) != _OOD_EVIDENCE_KEYS:
        raise ValueError("ood_evidence keys violate the frozen contract")
    if evidence.get("rule_version") != OOD_ROUTING_RULE_VERSION:
        raise ValueError("OOD routing rule version mismatch")
    source = evidence.get("source")
    if source not in _OOD_EVIDENCE_SOURCES:
        raise ValueError("ood_evidence.source is invalid")
    reason = _require_nonempty_string(
        evidence.get("reason"), name="ood_evidence.reason"
    )
    if source == "default_in_distribution" and tag != "in_distribution":
        raise ValueError(
            "default OOD evidence can only select in_distribution"
        )
    return tag, {
        "rule_version": OOD_ROUTING_RULE_VERSION,
        "source": source,
        "reason": reason,
    }


def _binary_mask(
    value: np.ndarray, *, name: str, shape: tuple[int, int]
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"{name} shape must be {shape}")
    if not np.isfinite(value).all() or not np.isin(value, (0.0, 1.0)).all():
        raise ValueError(f"{name} must be a finite binary mask")
    return value.astype(bool, copy=False)


def _derive_grouping_fields(
    *,
    sample: RiskSample,
    source: RiskBuildInput,
    rendered: RenderedObservation,
    age_max_s: float,
) -> dict[str, object]:
    if sample.bev_history.ndim != 4 or sample.bev_history.shape[1] != len(
        HISTORY_CHANNELS
    ):
        raise ValueError("sample bev_history channel shape is invalid")
    if sample.state_channels.ndim != 3 or sample.state_channels.shape[0] != len(
        STATE_CHANNELS
    ):
        raise ValueError("sample state_channels channel shape is invalid")
    if (
        sample.trajectory_channels.ndim != 3
        or sample.trajectory_channels.shape[0] != len(TRAJECTORY_CHANNELS)
    ):
        raise ValueError("sample trajectory_channels channel shape is invalid")
    height, width = sample.state_channels.shape[1:]
    expected_history_shape = (
        sample.bev_history.shape[0],
        len(HISTORY_CHANNELS),
        height,
        width,
    )
    if sample.bev_history.shape != expected_history_shape:
        raise ValueError("sample history/state spatial shape join failed")
    if sample.trajectory_channels.shape[1:] != (height, width):
        raise ValueError("sample trajectory/state spatial shape join failed")
    if rendered.bev_history.shape != sample.bev_history.shape or not (
        np.array_equal(rendered.bev_history, sample.bev_history)
    ):
        raise ValueError("renderer/sample bev_history join failed")
    if rendered.state_channels.shape != sample.state_channels.shape or not (
        np.array_equal(rendered.state_channels, sample.state_channels)
    ):
        raise ValueError("renderer/sample state_channels join failed")
    if source.trajectory.swept_mask.shape != (height, width):
        raise ValueError("source trajectory swept-mask shape join failed")
    swept_value = sample.trajectory_channels[
        TRAJECTORY_CHANNELS.index("swept_volume_mask")
    ]
    if not np.array_equal(swept_value, source.trajectory.swept_mask):
        raise ValueError("sample/source swept-volume join failed")

    shape = (height, width)
    unobservable = _binary_mask(
        rendered.state_channels[
            STATE_CHANNELS.index("current_unobservable_mask")
        ],
        name="current_unobservable_mask",
        shape=shape,
    )
    swept = _binary_mask(swept_value, name="swept_volume_mask", shape=shape)
    visible = _binary_mask(
        rendered.bev_history[-1, HISTORY_CHANNELS.index("past_visible_mask")],
        name="current visible mask",
        shape=shape,
    )
    if not np.array_equal(unobservable, ~visible):
        raise ValueError("current visibility/unobservable channel join failed")
    dynamic = _binary_mask(
        rendered.bev_history[
            -1, HISTORY_CHANNELS.index("past_dynamic_occupancy")
        ],
        name="current visible dynamic occupancy",
        shape=shape,
    )
    if np.any(dynamic & ~visible):
        raise ValueError("visible dynamic occupancy exists outside visible cells")
    visible_count = int(np.count_nonzero(visible))
    if visible_count == 0:
        raise ValueError("density_fraction requires at least one visible cell")
    age = rendered.state_channels[STATE_CHANNELS.index("occlusion_age_map")]
    if age.shape != shape or not np.isfinite(age).all():
        raise ValueError("occlusion_age_map must be finite with joined shape")
    if np.any(age < 0.0) or np.any(age > 1.0):
        raise ValueError("occlusion_age_map must be normalized to [0, 1]")
    maximum_age = _finite_float(age_max_s, name="age_max_s")
    if maximum_age <= 0.0:
        raise ValueError("age_max_s must be positive")

    critical = unobservable & swept
    critical_count = int(np.count_nonzero(critical))
    empty = critical_count == 0
    return {
        "critical_area_fraction": critical_count / float(height * width),
        "age_s": (
            0.0
            if empty
            else float(np.mean(age[critical], dtype=np.float64)) * maximum_age
        ),
        "critical_region_empty": empty,
        "density_fraction": int(np.count_nonzero(dynamic & visible))
        / float(visible_count),
    }


def _joined_identity(
    sample: RiskSample, source: RiskBuildInput
) -> dict[str, str]:
    if sample.sample_id != source.sample_id:
        raise ValueError("sample/source sample_id join failed")
    if sample.split != source.base_state.split:
        raise ValueError("sample/source split join failed")
    if sample.base_state_id != source.base_state.state_id:
        raise ValueError("sample/source base_state_id join failed")
    if sample.pair_group_id != source.pair_group_id:
        raise ValueError("sample/source pair_group_id join failed")
    if sample.event_type != source.event_type:
        raise ValueError("sample/source event_type join failed")
    if not isinstance(sample.metadata, Mapping):
        raise TypeError("sample.metadata must be a mapping")
    trajectory_id = _require_nonempty_string(
        sample.metadata.get("trajectory_id"), name="trajectory_id"
    )
    if trajectory_id != source.trajectory.trajectory_id:
        raise ValueError("sample/source trajectory_id join failed")
    source_provenance = source.provenance
    sample_provenance = sample.metadata.get("provenance")
    if not isinstance(source_provenance, Mapping) or not isinstance(
        sample_provenance, Mapping
    ):
        raise TypeError("sample/source provenance must be mappings")
    identity = {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "base_state_id": sample.base_state_id,
        "pair_group_id": sample.pair_group_id,
        "event_type": sample.event_type,
        "trajectory_id": trajectory_id,
    }
    for field in (
        "base_recording_id",
        "base_session_id",
        "source_recording_id",
        "source_session_id",
        "source_object_id",
        "source_snippet_id",
        "seed_namespace",
    ):
        source_value = _require_nonempty_string(
            source_provenance.get(field), name=field
        )
        sample_value = _require_nonempty_string(
            sample_provenance.get(field), name=f"sample provenance {field}"
        )
        if source_value != sample_value:
            raise ValueError(f"sample/source {field} join failed")
        identity[field] = source_value
    if identity["base_recording_id"] != source.base_state.recording_id:
        raise ValueError("base_recording_id differs from BaseState")
    if identity["base_session_id"] != source.base_state.metadata.get(
        "session_id"
    ):
        raise ValueError("base_session_id differs from BaseState")
    return identity


def _target_fields(
    sample: RiskSample, source: RiskBuildInput
) -> dict[str, object]:
    audit = sample.metadata.get("label_audit")
    if not isinstance(audit, Mapping):
        raise TypeError("sample label_audit must be a mapping")
    critical_id = audit.get("critical_object_id")
    critical_type = audit.get("critical_object_type")
    hidden_ids = tuple(source.hidden_object_ids)
    if not hidden_ids:
        if critical_id is not None or critical_type is not None:
            raise ValueError("empty target requires null critical identity")
        return {
            "critical_object_id": None,
            "target_object_type": None,
            "target_footprint_spec": None,
            "footprint_kind": "none",
        }
    critical_id = _require_nonempty_string(
        critical_id, name="critical_object_id"
    )
    if critical_id not in hidden_ids:
        raise ValueError("critical_object_id is not a declared hidden target")
    spec = source.oracle_world.dynamic_object_specs.get(critical_id)
    if not isinstance(spec, Mapping):
        raise ValueError("critical target spec is unavailable")
    object_type = _require_nonempty_string(
        spec.get("object_type"), name="target_object_type"
    )
    if critical_type != object_type:
        raise ValueError("sample/source target object type join failed")
    footprint = _canonical_footprint_mapping(
        spec.get("footprint"), name="target_footprint_spec"
    )
    return {
        "critical_object_id": critical_id,
        "target_object_type": object_type,
        "target_footprint_spec": footprint,
        "footprint_kind": footprint["kind"],
    }


def _join_ground_truth(
    sample: RiskSample, ground_truth: RiskGroundTruth
) -> None:
    if not isinstance(ground_truth, RiskGroundTruth):
        raise TypeError("ground_truth must be a RiskGroundTruth")
    if ground_truth.schema_version != SCHEMA_VERSION:
        raise ValueError("ground_truth schema_version mismatch")
    if ground_truth.pose_time_layout_version != POSE_TIME_LAYOUT_VERSION:
        raise ValueError("ground_truth pose_time_layout_version mismatch")
    for field in (
        "collision_label",
        "risk_severity",
        "min_clearance",
        "near_miss",
        "first_collision_time",
    ):
        if getattr(sample, field) != getattr(ground_truth, field):
            raise ValueError(f"ground_truth {field} differs from sample")

    audit = sample.metadata.get("label_audit")
    if not isinstance(audit, Mapping):
        raise TypeError("sample label_audit must be a mapping")
    audit_fields = {
        "critical_object_id": ground_truth.critical_object_id,
        "critical_object_type": ground_truth.critical_object_type,
        "time_to_min_clearance_s": ground_truth.time_to_min_clearance,
        "has_hidden_target": ground_truth.has_hidden_target,
    }
    for field, expected in audit_fields.items():
        if audit.get(field) != expected:
            raise ValueError(f"ground_truth {field} differs from sample label_audit")


def derive_production_evaluation_record(
    *,
    sample: RiskSample,
    source: RiskBuildInput,
    rendered: RenderedObservation,
    ground_truth: RiskGroundTruth,
    robot_footprint: Footprint,
    age_max_s: float,
    pair_eligible: bool,
    ood_tag: str,
    robot_footprint_provenance: Mapping[str, object],
    ood_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Derive one in-memory unpublished record at the oracle boundary.

    The returned record is not a formally authenticated publication.  The
    sibling evaluation-collection publisher must later bind it to an
    authenticated risk shard and verify its sample identity and labels.
    """

    from src.datasets.risk_dataset import RiskBuildInput as RiskBuildInputType

    if not isinstance(sample, RiskSample):
        raise TypeError("sample must be a RiskSample")
    if not isinstance(source, RiskBuildInputType):
        raise TypeError("source must be a RiskBuildInput")
    if not isinstance(rendered, RenderedObservation):
        raise TypeError("rendered must be a RenderedObservation")
    _join_ground_truth(sample, ground_truth)
    identity = _joined_identity(sample, source)
    grouping = _derive_grouping_fields(
        sample=sample,
        source=source,
        rendered=rendered,
        age_max_s=age_max_s,
    )
    target = _target_fields(sample, source)
    provenance = _validate_robot_footprint_provenance(
        robot_footprint_provenance, expected_effective=robot_footprint
    )
    tag, evidence = _validate_ood(ood_tag, ood_evidence)
    record: dict[str, object] = {
        "risk_evaluation_record_layout_version": (
            RISK_EVALUATION_RECORD_LAYOUT_VERSION
        ),
        **identity,
        "collision_label": ground_truth.collision_label,
        "risk_severity": ground_truth.risk_severity,
        "min_clearance": ground_truth.min_clearance,
        "near_miss": ground_truth.near_miss,
        "first_collision_time": ground_truth.first_collision_time,
        "blind_type": _require_nonempty_string(
            source.provenance.get("blind_spot_type"), name="blind_type"
        ),
        **grouping,
        **target,
        "pair_eligible": _require_bool(pair_eligible, name="pair_eligible"),
        "pair_eligibility_rule_version": PAIR_ELIGIBILITY_RULE_VERSION,
        "ood_tag": tag,
        "ood_evidence": evidence,
        "robot_footprint_spec": footprint_to_canonical_spec(robot_footprint),
        "robot_footprint_provenance": provenance,
        "contact_policy_rule_version": CONTACT_POLICY_RULE_VERSION,
    }
    return validate_production_evaluation_record(
        record, expected_sample_id=sample.sample_id
    )


def validate_production_evaluation_record(
    record: Mapping[str, object], *, expected_sample_id: str | None = None
) -> dict[str, object]:
    """Validate exact record semantics and return an owned deep-frozen copy."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    if set(record) != _RECORD_KEYS:
        raise ValueError("evaluation record keys violate the frozen contract")
    if record.get("risk_evaluation_record_layout_version") != (
        RISK_EVALUATION_RECORD_LAYOUT_VERSION
    ):
        raise ValueError("evaluation record layout version mismatch")
    normalized: dict[str, object] = {
        "risk_evaluation_record_layout_version": (
            RISK_EVALUATION_RECORD_LAYOUT_VERSION
        )
    }
    for field in _IDENTITY_STRING_FIELDS:
        normalized[field] = _require_nonempty_string(record.get(field), name=field)
    if expected_sample_id is not None and normalized["sample_id"] != (
        _require_nonempty_string(expected_sample_id, name="expected_sample_id")
    ):
        raise ValueError("evaluation record sample_id mismatch")

    collision = _binary_integer(record.get("collision_label"), name="collision_label")
    near_miss = _binary_integer(record.get("near_miss"), name="near_miss")
    severity = _finite_float(record.get("risk_severity"), name="risk_severity")
    clearance = _finite_float(record.get("min_clearance"), name="min_clearance")
    first_collision_raw = record.get("first_collision_time")
    first_collision = (
        None
        if first_collision_raw is None
        else _finite_float(first_collision_raw, name="first_collision_time")
    )
    if not 0.0 <= severity <= 1.0:
        raise ValueError("risk_severity must be in [0, 1]")
    if collision:
        if first_collision is None or first_collision <= 0.0:
            raise ValueError("collision requires positive first_collision_time")
        if clearance > 0.0 or severity != 1.0:
            raise ValueError("collision label fields are inconsistent")
    elif first_collision is not None:
        raise ValueError("noncollision requires null first_collision_time")
    elif clearance <= 0.0:
        raise ValueError("noncollision requires positive min_clearance")
    if collision and near_miss:
        raise ValueError("collision and near_miss cannot both be one")
    normalized.update(
        {
            "collision_label": collision,
            "risk_severity": severity,
            "min_clearance": clearance,
            "near_miss": near_miss,
            "first_collision_time": first_collision,
            "blind_type": _require_nonempty_string(
                record.get("blind_type"), name="blind_type"
            ),
        }
    )
    critical_fraction = _finite_float(
        record.get("critical_area_fraction"), name="critical_area_fraction"
    )
    age_s = _finite_float(record.get("age_s"), name="age_s")
    density = _finite_float(
        record.get("density_fraction"), name="density_fraction"
    )
    empty = _require_bool(
        record.get("critical_region_empty"), name="critical_region_empty"
    )
    if not 0.0 <= critical_fraction <= 1.0:
        raise ValueError("critical_area_fraction must be in [0, 1]")
    if age_s < 0.0:
        raise ValueError("age_s must be non-negative")
    if not 0.0 <= density <= 1.0:
        raise ValueError("density_fraction must be in [0, 1]")
    if empty != (critical_fraction == 0.0) or (empty and age_s != 0.0):
        raise ValueError("critical-region empty semantics are inconsistent")
    normalized.update(
        {
            "critical_area_fraction": critical_fraction,
            "age_s": age_s,
            "critical_region_empty": empty,
            "density_fraction": density,
        }
    )

    critical_id = record.get("critical_object_id")
    target_type = record.get("target_object_type")
    target_footprint = record.get("target_footprint_spec")
    footprint_kind = record.get("footprint_kind")
    if critical_id is None:
        if (
            target_type is not None
            or target_footprint is not None
            or footprint_kind != "none"
        ):
            raise ValueError("empty target fields are inconsistent")
        normalized.update(
            {
                "critical_object_id": None,
                "target_object_type": None,
                "target_footprint_spec": None,
                "footprint_kind": "none",
            }
        )
    else:
        normalized["critical_object_id"] = _require_nonempty_string(
            critical_id, name="critical_object_id"
        )
        normalized["target_object_type"] = _require_nonempty_string(
            target_type, name="target_object_type"
        )
        canonical_target = _canonical_footprint_mapping(
            target_footprint, name="target_footprint_spec"
        )
        if footprint_kind != canonical_target["kind"]:
            raise ValueError("footprint_kind differs from target_footprint_spec")
        normalized["target_footprint_spec"] = canonical_target
        normalized["footprint_kind"] = footprint_kind

    normalized["pair_eligible"] = _require_bool(
        record.get("pair_eligible"), name="pair_eligible"
    )
    if record.get("pair_eligibility_rule_version") != PAIR_ELIGIBILITY_RULE_VERSION:
        raise ValueError("pair eligibility rule version mismatch")
    normalized["pair_eligibility_rule_version"] = PAIR_ELIGIBILITY_RULE_VERSION
    tag, evidence = _validate_ood(record.get("ood_tag"), record.get("ood_evidence"))
    normalized["ood_tag"] = tag
    normalized["ood_evidence"] = evidence
    robot_spec = _canonical_footprint_mapping(
        record.get("robot_footprint_spec"), name="robot_footprint_spec"
    )
    robot_provenance = _validate_robot_footprint_provenance(
        record.get("robot_footprint_provenance"),
        expected_effective=_footprint_from_canonical_spec(robot_spec),
    )
    normalized["robot_footprint_spec"] = robot_spec
    normalized["robot_footprint_provenance"] = robot_provenance
    if record.get("contact_policy_rule_version") != CONTACT_POLICY_RULE_VERSION:
        raise ValueError("contact policy rule version mismatch")
    normalized["contact_policy_rule_version"] = CONTACT_POLICY_RULE_VERSION

    try:
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation record must be finite canonical JSON") from exc
    return _FrozenDict(normalized)


__all__ = [
    "ALLOWED_OOD_TAGS",
    "CONTACT_POLICY_RULE_VERSION",
    "OOD_ROUTING_RULE_VERSION",
    "PAIR_ELIGIBILITY_RULE_VERSION",
    "RISK_EVALUATION_RECORD_LAYOUT_VERSION",
    "ROBOT_FOOTPRINT_PROVENANCE_RULE_VERSION",
    "derive_production_evaluation_record",
    "derive_robot_footprint_provenance",
    "footprint_to_canonical_spec",
    "validate_production_evaluation_record",
]
