"""Current Long40 event and target-motion contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.contracts import OracleWorld, validate_dynamic_object_spec
from src.geometry import CircleFootprint, Footprint, RectangleFootprint

from .event_target_motion_shard import EventTargetMotionRecord


@dataclass(frozen=True)
class TransplantedDynamicObject:
    """One target object's measured history, current pose, and future."""

    target_dynamic_object_id: str
    source_object_id: str
    snippet_id: str
    object_type: str
    footprint_spec: dict[str, object]
    footprint_spec_digest: str
    history_poses: np.ndarray
    current_pose: np.ndarray
    future_poses: np.ndarray
    provenance: dict[str, object]


@dataclass(frozen=True)
class GeneratedEvent:
    """One accepted oracle world and its authenticated target state."""

    generated_event_id: str
    event_kind: str
    world: OracleWorld
    target: TransplantedDynamicObject
    target_motion_record: EventTargetMotionRecord
    visibility_sequence: np.ndarray
    target_visibility_history: np.ndarray
    conflict_time_s: float
    conflict_index: int


def footprint_from_spec(spec: Mapping[str, Any]) -> Footprint:
    """Materialize a frozen contract footprint without reclassification."""

    canonical = dict(spec)
    validate_dynamic_object_spec(canonical)
    footprint = canonical["footprint"]
    if footprint["kind"] == "circle":
        return CircleFootprint(float(footprint["radius_m"]))
    return RectangleFootprint(
        float(footprint["length_m"]),
        float(footprint["width_m"]),
    )
