"""Versioned SOP05-to-SOP06 target-motion shard with strict joins.

The shard deliberately has one supported layout.  It stores target history for
the renderer next to the exact target future consumed by ``OracleWorld`` while
keeping the two uses separate and independently auditable.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.contracts import (
    DYNAMIC_OBJECT_TYPES,
    SCHEMA_VERSION,
    GridSpec,
    OracleContext,
    OracleWorld,
    load_dataclass,
    save_dataclass,
    validate_dynamic_object_spec,
    validate_oracle_world,
)
from src.utils.seeding import stable_digest


EVENT_TARGET_MOTION_LAYOUT_VERSION = "event_target_motion_history8_future15_v1"
HISTORY_STEPS = 8
CURRENT_INDEX = 7
FUTURE_STEPS = 15
SAMPLE_DT_S = 0.2
CURRENT_TIME_OFFSET_S = 0.0
HISTORY_TIME_OFFSETS_S = tuple(
    round((index - CURRENT_INDEX) * SAMPLE_DT_S, 10)
    for index in range(HISTORY_STEPS)
)
FUTURE_TIME_OFFSETS_S = tuple(
    round((index + 1) * SAMPLE_DT_S, 10) for index in range(FUTURE_STEPS)
)

_ARRAY_DIGEST_DOMAIN = "event-target-motion-array-digest-v1"
_RECORD_DIGEST_DOMAIN = "event-target-motion-record-digest-v1"
_MANIFEST_DIGEST_DOMAIN = "event-target-motion-manifest-digest-v1"
_PAYLOAD_DIGEST_DOMAIN = "event-target-motion-payload-semantic-digest-v1"
_WORLD_SEMANTIC_DIGEST_DOMAIN = "event-target-motion-oracle-world-digest-v1"
_FLOAT32_LE_DTYPE_TOKEN = "<f4"
_ARRAY_ORDER_TOKEN = "C"

_PAYLOAD_FILENAME = f"{EVENT_TARGET_MOTION_LAYOUT_VERSION}.npz"
_MANIFEST_FILENAME = "generated_event_manifest.jsonl"
_SUMMARY_FILENAME = "shard_summary.json"
_WORLD_DIRECTORY = "oracle_worlds"

_NPZ_KEYS = frozenset(
    {"history_poses", "current_poses", "future_poses", "meta_json"}
)
_LAYOUT_KEYS = frozenset(
    {
        "schema_version",
        "layout_version",
        "sample_dt_s",
        "history_steps",
        "current_index",
        "future_steps",
        "history_time_offsets_s",
        "current_time_offset_s",
        "future_time_offsets_s",
    }
)
_NPZ_META_KEYS = _LAYOUT_KEYS | frozenset(
    {
        "array_dtype",
        "array_order",
        "record_count",
        "generated_event_ids",
        "history_array_digests",
        "future_array_digests",
        "record_digests",
        "manifest_digest",
        "payload_semantic_digest",
    }
)
_MANIFEST_ROW_KEYS = _LAYOUT_KEYS | frozenset(
    {
        "row_index",
        "generated_event_id",
        "world_id",
        "base_state_id",
        "trajectory_id",
        "target_dynamic_object_id",
        "source_snippet_id",
        "source_object_id",
        "object_type",
        "footprint_spec",
        "footprint_spec_digest",
        "target_type_policy_digest",
        "history_array_digest",
        "future_array_digest",
        "record_digest",
        "world_semantic_digest",
        "world_file",
        "manifest_digest",
    }
)
_SUMMARY_KEYS = _LAYOUT_KEYS | frozenset(
    {
        "array_dtype",
        "array_order",
        "record_count",
        "world_count",
        "history_array_shape",
        "current_array_shape",
        "future_array_shape",
        "manifest_digest",
        "payload_semantic_digest",
    }
)
_WORLD_METADATA_KEYS = frozenset(
    {
        "generated_event_id",
        "world_id",
        "base_state_id",
        "trajectory_id",
        "target_dynamic_object_id",
        "source_snippet_id",
        "source_object_id",
        "target_object_type",
        "target_footprint_spec",
        "target_footprint_spec_digest",
        "target_type_policy_digest",
        "event_target_motion_layout_version",
        "target_history_array_digest",
        "target_future_array_digest",
        "target_motion_record_digest",
        "target_current_pose",
    }
)


class _FrozenDict(dict[str, object]):
    """JSON-serializable dict whose identity content cannot be mutated."""

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        raise TypeError("frozen identity mapping cannot be modified")

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


@dataclass(frozen=True)
class EventTargetMotionRecord:
    """One immutable-identity target-motion record with owned numeric arrays."""

    schema_version: str
    layout_version: str
    generated_event_id: str
    world_id: str
    base_state_id: str
    trajectory_id: str
    target_dynamic_object_id: str
    source_snippet_id: str
    source_object_id: str
    object_type: str
    footprint_spec: dict[str, object]
    footprint_spec_digest: str
    target_type_policy_digest: str
    history_poses: np.ndarray
    current_pose: np.ndarray
    future_poses: np.ndarray
    history_array_digest: str
    future_array_digest: str
    record_digest: str


@dataclass(frozen=True)
class LoadedEventTargetMotionShard:
    """Fully validated shard contents returned by the formal loader."""

    records: tuple[EventTargetMotionRecord, ...]
    worlds: dict[str, OracleWorld]
    manifest_digest: str
    payload_semantic_digest: str
    summary: dict[str, object]


@dataclass(frozen=True)
class RendererScene:
    """History-only dynamic scene consumed by the observation renderer."""

    dynamic_object_history: dict[str, np.ndarray]
    dynamic_object_specs: dict[str, dict[str, object]]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON must not contain {value}")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON") from exc
    return text.encode("utf-8")


def _canonical_json_copy(value: object) -> Any:
    return json.loads(
        _canonical_json_bytes(value).decode("utf-8"),
        parse_constant=_reject_json_constant,
    )


def _freeze_canonical_json(value: object) -> object:
    if isinstance(value, dict):
        return _FrozenDict(
            {
                key: _freeze_canonical_json(nested)
                for key, nested in value.items()
            }
        )
    if isinstance(value, list):
        return tuple(_freeze_canonical_json(nested) for nested in value)
    return value


def _is_recursively_frozen_json(value: object) -> bool:
    if isinstance(value, _FrozenDict):
        return all(
            isinstance(key, str) and _is_recursively_frozen_json(nested)
            for key, nested in value.items()
        )
    if isinstance(value, tuple):
        return all(_is_recursively_frozen_json(nested) for nested in value)
    return not isinstance(value, (dict, list))


def _semantic_digest(domain: str, value: object) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    domain_bytes = domain.encode("utf-8")
    payload = _canonical_json_bytes(value)
    for part in (domain_bytes, payload):
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return hasher.hexdigest()


def _canonical_world_value(value: object, *, path: str) -> object:
    if value is None:
        return {"kind": "none"}
    if isinstance(value, (bool, np.bool_)):
        return {"kind": "bool", "value": bool(value)}
    if isinstance(value, str):
        return {"kind": "string", "value": value}
    if isinstance(value, (int, np.integer)):
        return {"kind": "int", "value": int(value)}
    if isinstance(value, (float, np.floating)):
        normalized = float(value)
        if not np.isfinite(normalized):
            raise ValueError(f"{path} must contain only finite values")
        return {"kind": "float", "value": normalized}
    if isinstance(value, np.ndarray):
        dtype = value.dtype
        if (
            dtype.hasobject
            or dtype.fields is not None
            or dtype.subdtype is not None
            or dtype.kind not in "biuf"
        ):
            raise ValueError(f"{path} has unsupported ndarray dtype {dtype}")
        if not np.isfinite(value).all():
            raise ValueError(f"{path} must contain only finite values")
        canonical_dtype = dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(value, dtype=canonical_dtype)
        return {
            "kind": "ndarray",
            "dtype": canonical.dtype.str,
            "shape": [int(dimension) for dimension in canonical.shape],
            "order": _ARRAY_ORDER_TOKEN,
            "raw_values_hex": canonical.tobytes(order="C").hex(),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} mappings must have string keys")
        return {
            "kind": "mapping",
            "items": [
                [
                    key,
                    _canonical_world_value(
                        value[key], path=f"{path}[{key!r}]"
                    ),
                ]
                for key in sorted(value)
            ],
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "sequence",
            "items": [
                _canonical_world_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    raise ValueError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def compute_oracle_world_semantic_digest(world: OracleWorld) -> str:
    """Digest every OracleWorld dataclass field without pickle or Python hash."""

    if not isinstance(world, OracleWorld):
        raise TypeError("world must be an OracleWorld")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "class": "OracleWorld",
        "fields": [
            [
                field.name,
                _canonical_world_value(
                    getattr(world, field.name),
                    path=f"OracleWorld.{field.name}",
                ),
            ]
            for field in fields(OracleWorld)
        ],
    }
    return _semantic_digest(_WORLD_SEMANTIC_DIGEST_DOMAIN, payload)


def _layout_metadata() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "layout_version": EVENT_TARGET_MOTION_LAYOUT_VERSION,
        "sample_dt_s": SAMPLE_DT_S,
        "history_steps": HISTORY_STEPS,
        "current_index": CURRENT_INDEX,
        "future_steps": FUTURE_STEPS,
        "history_time_offsets_s": list(HISTORY_TIME_OFFSETS_S),
        "current_time_offset_s": CURRENT_TIME_OFFSET_S,
        "future_time_offsets_s": list(FUTURE_TIME_OFFSETS_S),
    }


def _require_exact_keys(
    payload: Mapping[str, object], expected: frozenset[str], name: str
) -> None:
    actual = set(payload)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ValueError(f"{name} keys mismatch: missing={missing}, extra={extra}")


def _validate_layout_metadata(payload: Mapping[str, object], name: str) -> None:
    expected = _layout_metadata()
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{name} {key} mismatch")


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_digest(value: object, name: str) -> str:
    result = _require_nonempty_string(value, name)
    if len(result) != 32 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase BLAKE2b-128 hex digest")
    return result


def _safe_world_filename(world_id: str) -> str:
    if Path(world_id).name != world_id or world_id in {".", ".."} or "\x00" in world_id:
        raise ValueError("world_id is not safe for an artifact filename")
    return f"{world_id}.npz"


def _validate_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    require_owned_c: bool,
    require_read_only: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if value.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {value.shape}")
    if value.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} dtype must be float32, got {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if require_owned_c and (not value.flags.c_contiguous or not value.flags.owndata):
        raise ValueError(f"{name} must own C-contiguous storage")
    if require_read_only and value.flags.writeable:
        raise ValueError(f"{name} must have writeable=False")
    return value


def _owned_float32_array(
    value: object, *, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    array = _validate_array(
        value, name=name, shape=shape, require_owned_c=False
    )
    owned = np.array(array, dtype=np.float32, order="C", copy=True)
    owned.setflags(write=False)
    return owned


def compute_motion_array_digest(
    array: np.ndarray,
    *,
    field_name: str,
    layout_version: str = EVENT_TARGET_MOTION_LAYOUT_VERSION,
) -> str:
    """Digest one float32 array with explicit domain, layout, shape and order."""

    if layout_version != EVENT_TARGET_MOTION_LAYOUT_VERSION:
        raise ValueError("unsupported event target motion layout")
    _require_nonempty_string(field_name, "field_name")
    if not isinstance(array, np.ndarray):
        raise ValueError("array must be a numpy array")
    if array.dtype != np.dtype(np.float32):
        raise ValueError(f"array dtype must be float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("array must contain only finite values")
    canonical = np.ascontiguousarray(array, dtype=np.dtype("<f4"))
    shape_payload = _canonical_json_bytes(list(canonical.shape))
    parts = (
        _ARRAY_DIGEST_DOMAIN.encode("utf-8"),
        layout_version.encode("utf-8"),
        field_name.encode("utf-8"),
        _FLOAT32_LE_DTYPE_TOKEN.encode("ascii"),
        shape_payload,
        _ARRAY_ORDER_TOKEN.encode("ascii"),
        canonical.tobytes(order="C"),
    )
    hasher = hashlib.blake2b(digest_size=16)
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return hasher.hexdigest()


def compute_footprint_spec_digest(spec: Mapping[str, object]) -> str:
    """Return the existing SOP05 canonical digest for a frozen footprint spec."""

    if not isinstance(spec, Mapping):
        raise ValueError("footprint_spec must be a mapping")
    copied = _canonical_json_copy(dict(spec))
    validate_dynamic_object_spec(copied)
    canonical = _canonical_json_bytes(copied).decode("utf-8")
    return stable_digest(canonical, size=16)


def _record_digest_payload(
    *,
    generated_event_id: str,
    world_id: str,
    base_state_id: str,
    trajectory_id: str,
    target_dynamic_object_id: str,
    source_snippet_id: str,
    source_object_id: str,
    object_type: str,
    footprint_spec: dict[str, object],
    footprint_spec_digest: str,
    target_type_policy_digest: str,
    current_pose: np.ndarray,
    history_array_digest: str,
    future_array_digest: str,
) -> dict[str, object]:
    current_le = np.ascontiguousarray(current_pose, dtype=np.dtype("<f4"))
    return {
        **_layout_metadata(),
        "generated_event_id": generated_event_id,
        "world_id": world_id,
        "base_state_id": base_state_id,
        "trajectory_id": trajectory_id,
        "target_dynamic_object_id": target_dynamic_object_id,
        "source_snippet_id": source_snippet_id,
        "source_object_id": source_object_id,
        "object_type": object_type,
        "footprint_spec": footprint_spec,
        "footprint_spec_digest": footprint_spec_digest,
        "target_type_policy_digest": target_type_policy_digest,
        "current_pose_dtype": _FLOAT32_LE_DTYPE_TOKEN,
        "current_pose_shape": [3],
        "current_pose_order": _ARRAY_ORDER_TOKEN,
        "current_pose_le_bytes_hex": current_le.tobytes(order="C").hex(),
        "history_array_digest": history_array_digest,
        "future_array_digest": future_array_digest,
    }


def create_event_target_motion_record(
    *,
    generated_event_id: str,
    world_id: str,
    base_state_id: str,
    trajectory_id: str,
    target_dynamic_object_id: str,
    source_snippet_id: str,
    source_object_id: str,
    object_type: str,
    footprint_spec: Mapping[str, object],
    footprint_spec_digest: str,
    target_type_policy_digest: str,
    history_poses: np.ndarray,
    current_pose: np.ndarray,
    future_poses: np.ndarray,
) -> EventTargetMotionRecord:
    """Create one record without silently converting a wrong input dtype."""

    generated_event_id = _require_nonempty_string(
        generated_event_id, "generated_event_id"
    )
    world_id = _require_nonempty_string(world_id, "world_id")
    _safe_world_filename(world_id)
    base_state_id = _require_nonempty_string(base_state_id, "base_state_id")
    trajectory_id = _require_nonempty_string(trajectory_id, "trajectory_id")
    target_dynamic_object_id = _require_nonempty_string(
        target_dynamic_object_id, "target_dynamic_object_id"
    )
    source_snippet_id = _require_nonempty_string(
        source_snippet_id, "source_snippet_id"
    )
    source_object_id = _require_nonempty_string(source_object_id, "source_object_id")
    if object_type not in DYNAMIC_OBJECT_TYPES:
        raise ValueError(f"object_type must be one of {DYNAMIC_OBJECT_TYPES}")
    copied_spec = _canonical_json_copy(dict(footprint_spec))
    validate_dynamic_object_spec(copied_spec)
    footprint_spec_digest = _require_digest(
        footprint_spec_digest, "footprint_spec_digest"
    )
    expected_spec_digest = compute_footprint_spec_digest(copied_spec)
    if footprint_spec_digest != expected_spec_digest:
        raise ValueError("footprint_spec_digest mismatch")
    target_type_policy_digest = _require_digest(
        target_type_policy_digest, "target_type_policy_digest"
    )

    owned_history = _owned_float32_array(
        history_poses, name="history_poses", shape=(HISTORY_STEPS, 3)
    )
    owned_current = _owned_float32_array(
        current_pose, name="current_pose", shape=(3,)
    )
    owned_future = _owned_float32_array(
        future_poses, name="future_poses", shape=(FUTURE_STEPS, 3)
    )
    if not np.array_equal(owned_current, owned_history[CURRENT_INDEX]):
        raise ValueError("current_pose must equal history_poses[7] elementwise")

    history_digest = compute_motion_array_digest(
        owned_history, field_name="target_history_poses"
    )
    future_digest = compute_motion_array_digest(
        owned_future, field_name="target_future_poses"
    )
    record_digest = _semantic_digest(
        _RECORD_DIGEST_DOMAIN,
        _record_digest_payload(
            generated_event_id=generated_event_id,
            world_id=world_id,
            base_state_id=base_state_id,
            trajectory_id=trajectory_id,
            target_dynamic_object_id=target_dynamic_object_id,
            source_snippet_id=source_snippet_id,
            source_object_id=source_object_id,
            object_type=object_type,
            footprint_spec=copied_spec,
            footprint_spec_digest=footprint_spec_digest,
            target_type_policy_digest=target_type_policy_digest,
            current_pose=owned_current,
            history_array_digest=history_digest,
            future_array_digest=future_digest,
        ),
    )
    frozen_spec = _freeze_canonical_json(copied_spec)
    if not isinstance(frozen_spec, _FrozenDict):
        raise RuntimeError("canonical footprint spec did not freeze as a mapping")
    return EventTargetMotionRecord(
        schema_version=SCHEMA_VERSION,
        layout_version=EVENT_TARGET_MOTION_LAYOUT_VERSION,
        generated_event_id=generated_event_id,
        world_id=world_id,
        base_state_id=base_state_id,
        trajectory_id=trajectory_id,
        target_dynamic_object_id=target_dynamic_object_id,
        source_snippet_id=source_snippet_id,
        source_object_id=source_object_id,
        object_type=object_type,
        footprint_spec=frozen_spec,
        footprint_spec_digest=footprint_spec_digest,
        target_type_policy_digest=target_type_policy_digest,
        history_poses=owned_history,
        current_pose=owned_current,
        future_poses=owned_future,
        history_array_digest=history_digest,
        future_array_digest=future_digest,
        record_digest=record_digest,
    )


def validate_event_target_motion_record(record: EventTargetMotionRecord) -> None:
    """Validate a record, including all stored semantic digests."""

    if not isinstance(record, EventTargetMotionRecord):
        raise TypeError("record must be an EventTargetMotionRecord")
    if record.schema_version != SCHEMA_VERSION:
        raise ValueError("record schema_version mismatch")
    if record.layout_version != EVENT_TARGET_MOTION_LAYOUT_VERSION:
        raise ValueError("record layout_version mismatch")
    for name in (
        "generated_event_id",
        "world_id",
        "base_state_id",
        "trajectory_id",
        "target_dynamic_object_id",
        "source_snippet_id",
        "source_object_id",
    ):
        _require_nonempty_string(getattr(record, name), name)
    _require_digest(record.target_type_policy_digest, "target_type_policy_digest")
    _safe_world_filename(record.world_id)
    if record.object_type not in DYNAMIC_OBJECT_TYPES:
        raise ValueError("record object_type is invalid")
    if not _is_recursively_frozen_json(record.footprint_spec):
        raise ValueError("record footprint_spec must be recursively immutable")
    validate_dynamic_object_spec(record.footprint_spec)
    if record.footprint_spec.get("object_type") != record.object_type:
        raise ValueError("record object_type and footprint_spec mismatch")
    _require_digest(record.footprint_spec_digest, "footprint_spec_digest")
    if record.footprint_spec_digest != compute_footprint_spec_digest(
        record.footprint_spec
    ):
        raise ValueError("footprint_spec_digest mismatch")
    history = _validate_array(
        record.history_poses,
        name="history_poses",
        shape=(HISTORY_STEPS, 3),
        require_owned_c=True,
        require_read_only=True,
    )
    current = _validate_array(
        record.current_pose,
        name="current_pose",
        shape=(3,),
        require_owned_c=True,
        require_read_only=True,
    )
    future = _validate_array(
        record.future_poses,
        name="future_poses",
        shape=(FUTURE_STEPS, 3),
        require_owned_c=True,
        require_read_only=True,
    )
    if not np.array_equal(current, history[CURRENT_INDEX]):
        raise ValueError("current_pose must equal history_poses[7] elementwise")
    history_digest = compute_motion_array_digest(
        history, field_name="target_history_poses"
    )
    future_digest = compute_motion_array_digest(
        future, field_name="target_future_poses"
    )
    if record.history_array_digest != history_digest:
        raise ValueError("history_array_digest mismatch")
    if record.future_array_digest != future_digest:
        raise ValueError("future_array_digest mismatch")
    expected_record_digest = _semantic_digest(
        _RECORD_DIGEST_DOMAIN,
        _record_digest_payload(
            generated_event_id=record.generated_event_id,
            world_id=record.world_id,
            base_state_id=record.base_state_id,
            trajectory_id=record.trajectory_id,
            target_dynamic_object_id=record.target_dynamic_object_id,
            source_snippet_id=record.source_snippet_id,
            source_object_id=record.source_object_id,
            object_type=record.object_type,
            footprint_spec=record.footprint_spec,
            footprint_spec_digest=record.footprint_spec_digest,
            target_type_policy_digest=record.target_type_policy_digest,
            current_pose=current,
            history_array_digest=history_digest,
            future_array_digest=future_digest,
        ),
    )
    if record.record_digest != expected_record_digest:
        raise ValueError("record_digest mismatch")


def _world_metadata_expectations(
    record: EventTargetMotionRecord,
) -> dict[str, object]:
    return {
        "generated_event_id": record.generated_event_id,
        "world_id": record.world_id,
        "base_state_id": record.base_state_id,
        "trajectory_id": record.trajectory_id,
        "target_dynamic_object_id": record.target_dynamic_object_id,
        "source_snippet_id": record.source_snippet_id,
        "source_object_id": record.source_object_id,
        "target_object_type": record.object_type,
        "target_footprint_spec": record.footprint_spec,
        "target_footprint_spec_digest": record.footprint_spec_digest,
        "target_type_policy_digest": record.target_type_policy_digest,
        "event_target_motion_layout_version": record.layout_version,
        "target_history_array_digest": record.history_array_digest,
        "target_future_array_digest": record.future_array_digest,
        "target_motion_record_digest": record.record_digest,
    }


def build_event_target_motion_world_metadata(
    record: EventTargetMotionRecord,
) -> dict[str, object]:
    """Build an owned copy of the frozen SOP05-to-SOP06 world join metadata."""

    validate_event_target_motion_record(record)
    metadata = _canonical_json_copy(_world_metadata_expectations(record))
    metadata["target_current_pose"] = [
        float(value) for value in record.current_pose
    ]
    return metadata


def validate_event_target_motion_world_join(
    record: EventTargetMotionRecord, world: OracleWorld, grid: GridSpec
) -> None:
    """Strictly validate one record against its serialized OracleWorld contract."""

    validate_event_target_motion_record(record)
    if not isinstance(world, OracleWorld):
        raise TypeError("worlds must contain OracleWorld objects")
    validate_oracle_world(world, grid)
    if world.world_id != record.world_id:
        raise ValueError("world_id join mismatch")
    if world.base_state_id != record.base_state_id:
        raise ValueError("base_state_id join mismatch")
    if record.target_dynamic_object_id not in world.dynamic_object_trajectories:
        raise ValueError("target future is missing from OracleWorld")
    if record.target_dynamic_object_id not in world.dynamic_object_specs:
        raise ValueError("target footprint spec is missing from OracleWorld")
    if not np.array_equal(
        world.dynamic_object_trajectories[record.target_dynamic_object_id],
        record.future_poses,
    ):
        raise ValueError("OracleWorld target future does not match target future")
    if world.dynamic_object_specs[record.target_dynamic_object_id] != record.footprint_spec:
        raise ValueError("OracleWorld target footprint spec mismatch")
    if not isinstance(world.metadata, Mapping):
        raise ValueError("OracleWorld metadata must be a mapping")
    missing = sorted(_WORLD_METADATA_KEYS - set(world.metadata))
    if missing:
        raise ValueError(f"OracleWorld target metadata missing keys: {missing}")
    for key, expected in _world_metadata_expectations(record).items():
        if world.metadata.get(key) != expected:
            raise ValueError(f"OracleWorld metadata {key} mismatch")
    current = world.metadata.get("target_current_pose")
    if (
        not isinstance(current, list)
        or len(current) != 3
        or any(type(value) is not float or not np.isfinite(value) for value in current)
    ):
        raise ValueError("OracleWorld target_current_pose metadata is invalid")
    expected_current = [float(value) for value in record.current_pose]
    if any(
        actual.hex() != expected.hex()
        for actual, expected in zip(current, expected_current, strict=True)
    ):
        raise ValueError("OracleWorld target_current_pose metadata mismatch")


def _prepare_records_and_worlds(
    records: Sequence[EventTargetMotionRecord],
    worlds: Sequence[OracleWorld],
    grid: GridSpec,
) -> tuple[tuple[EventTargetMotionRecord, ...], dict[str, OracleWorld]]:
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    if grid.history_steps != HISTORY_STEPS or grid.future_steps != FUTURE_STEPS:
        raise ValueError("grid history/future steps do not match shard layout")
    ordered_input = tuple(records)
    world_input = tuple(worlds)
    if not ordered_input:
        raise ValueError("records must not be empty")
    event_ids = [getattr(record, "generated_event_id", None) for record in ordered_input]
    world_ids = [getattr(record, "world_id", None) for record in ordered_input]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("duplicate generated_event_id")
    if len(world_ids) != len(set(world_ids)):
        raise ValueError("duplicate world_id")
    supplied_world_ids = [getattr(world, "world_id", None) for world in world_input]
    if len(supplied_world_ids) != len(set(supplied_world_ids)):
        raise ValueError("duplicate OracleWorld world_id")
    world_by_id = {world.world_id: world for world in world_input}
    if set(world_ids) != set(world_by_id):
        raise ValueError("record and OracleWorld world_id sets differ")
    ordered = tuple(sorted(ordered_input, key=lambda record: record.generated_event_id))
    for record in ordered:
        validate_event_target_motion_world_join(
            record, world_by_id[record.world_id], grid
        )
    return ordered, world_by_id


def _manifest_base_row(
    record: EventTargetMotionRecord,
    row_index: int,
    world_semantic_digest: str,
) -> dict[str, object]:
    return {
        **_layout_metadata(),
        "row_index": row_index,
        "generated_event_id": record.generated_event_id,
        "world_id": record.world_id,
        "base_state_id": record.base_state_id,
        "trajectory_id": record.trajectory_id,
        "target_dynamic_object_id": record.target_dynamic_object_id,
        "source_snippet_id": record.source_snippet_id,
        "source_object_id": record.source_object_id,
        "object_type": record.object_type,
        "footprint_spec": record.footprint_spec,
        "footprint_spec_digest": record.footprint_spec_digest,
        "target_type_policy_digest": record.target_type_policy_digest,
        "history_array_digest": record.history_array_digest,
        "future_array_digest": record.future_array_digest,
        "record_digest": record.record_digest,
        "world_semantic_digest": world_semantic_digest,
        "world_file": f"{_WORLD_DIRECTORY}/{_safe_world_filename(record.world_id)}",
    }


def _manifest_digest(base_rows: Sequence[Mapping[str, object]]) -> str:
    return _semantic_digest(_MANIFEST_DIGEST_DOMAIN, list(base_rows))


def _payload_semantic_payload(
    records: Sequence[EventTargetMotionRecord], manifest_digest: str
) -> dict[str, object]:
    return {
        **_layout_metadata(),
        "array_dtype": _FLOAT32_LE_DTYPE_TOKEN,
        "array_order": _ARRAY_ORDER_TOKEN,
        "manifest_digest": manifest_digest,
        "records": [
            {
                "generated_event_id": record.generated_event_id,
                "history_array_digest": record.history_array_digest,
                "future_array_digest": record.future_array_digest,
                "record_digest": record.record_digest,
            }
            for record in records
        ],
    }


def _payload_semantic_digest(
    records: Sequence[EventTargetMotionRecord], manifest_digest: str
) -> str:
    return _semantic_digest(
        _PAYLOAD_DIGEST_DOMAIN,
        _payload_semantic_payload(records, manifest_digest),
    )


def _npz_meta(
    records: Sequence[EventTargetMotionRecord],
    manifest_digest: str,
    payload_semantic_digest: str,
) -> dict[str, object]:
    return {
        **_layout_metadata(),
        "array_dtype": _FLOAT32_LE_DTYPE_TOKEN,
        "array_order": _ARRAY_ORDER_TOKEN,
        "record_count": len(records),
        "generated_event_ids": [record.generated_event_id for record in records],
        "history_array_digests": [record.history_array_digest for record in records],
        "future_array_digests": [record.future_array_digest for record in records],
        "record_digests": [record.record_digest for record in records],
        "manifest_digest": manifest_digest,
        "payload_semantic_digest": payload_semantic_digest,
    }


def _summary(
    record_count: int, manifest_digest: str, payload_semantic_digest: str
) -> dict[str, object]:
    return {
        **_layout_metadata(),
        "array_dtype": _FLOAT32_LE_DTYPE_TOKEN,
        "array_order": _ARRAY_ORDER_TOKEN,
        "record_count": record_count,
        "world_count": record_count,
        "history_array_shape": [record_count, HISTORY_STEPS, 3],
        "current_array_shape": [record_count, 3],
        "future_array_shape": [record_count, FUTURE_STEPS, 3],
        "manifest_digest": manifest_digest,
        "payload_semantic_digest": payload_semantic_digest,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_manifest(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(
            _canonical_json_bytes(row).decode("utf-8") + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_payload(
    path: Path,
    records: Sequence[EventTargetMotionRecord],
    metadata: Mapping[str, object],
) -> None:
    history = np.ascontiguousarray(
        np.stack([record.history_poses for record in records]), dtype=np.float32
    )
    current = np.ascontiguousarray(
        np.stack([record.current_pose for record in records]), dtype=np.float32
    )
    future = np.ascontiguousarray(
        np.stack([record.future_poses for record in records]), dtype=np.float32
    )
    meta_json = np.asarray(_canonical_json_bytes(metadata).decode("utf-8"))
    with path.open("wb") as handle:
        np.savez(
            handle,
            history_poses=history,
            current_poses=current,
            future_poses=future,
            meta_json=meta_json,
        )


def write_event_target_motion_shard(
    records: Sequence[EventTargetMotionRecord],
    worlds: Sequence[OracleWorld],
    output_dir: str | Path,
    *,
    grid: GridSpec,
) -> dict[str, Path]:
    """Validate, stage, formally reload, then atomically publish one shard."""

    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output_path}")
    ordered, world_by_id = _prepare_records_and_worlds(records, worlds, grid)
    world_semantic_digests = {
        world_id: compute_oracle_world_semantic_digest(world)
        for world_id, world in world_by_id.items()
    }
    base_rows = [
        _manifest_base_row(
            record,
            index,
            world_semantic_digests[record.world_id],
        )
        for index, record in enumerate(ordered)
    ]
    manifest_digest = _manifest_digest(base_rows)
    manifest_rows = [
        {**row, "manifest_digest": manifest_digest} for row in base_rows
    ]
    payload_digest = _payload_semantic_digest(ordered, manifest_digest)
    metadata = _npz_meta(ordered, manifest_digest, payload_digest)
    summary = _summary(len(ordered), manifest_digest, payload_digest)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.staging-", dir=output_path.parent
        )
    )
    try:
        world_directory = staging / _WORLD_DIRECTORY
        world_directory.mkdir()
        for record in ordered:
            save_dataclass(
                world_by_id[record.world_id],
                world_directory / _safe_world_filename(record.world_id),
            )
        _write_manifest(staging / _MANIFEST_FILENAME, manifest_rows)
        _write_payload(staging / _PAYLOAD_FILENAME, ordered, metadata)
        _write_json(staging / _SUMMARY_FILENAME, summary)

        load_event_target_motion_shard(
            staging,
            grid=grid,
            expected_generated_event_ids={
                record.generated_event_id for record in ordered
            },
            expected_base_state_ids={record.base_state_id for record in ordered},
            expected_trajectory_ids={record.trajectory_id for record in ordered},
        )
        staging.replace(output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "directory": output_path,
        "manifest": output_path / _MANIFEST_FILENAME,
        "payload": output_path / _PAYLOAD_FILENAME,
        "summary": output_path / _SUMMARY_FILENAME,
        "worlds": output_path / _WORLD_DIRECTORY,
    }


def _load_json_object(path: Path, name: str) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _load_manifest(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError("invalid generated event manifest") from exc
    if not lines or any(not line for line in lines):
        raise ValueError("generated event manifest must be non-empty canonical JSONL")
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid manifest row {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"manifest row {line_number} must be an object")
        _require_exact_keys(row, _MANIFEST_ROW_KEYS, f"manifest row {line_number}")
        _validate_layout_metadata(row, f"manifest row {line_number}")
        if not isinstance(row["row_index"], int) or isinstance(
            row["row_index"], bool
        ):
            raise ValueError("row_index must be an int")
        for name in (
            "generated_event_id",
            "world_id",
            "base_state_id",
            "trajectory_id",
            "target_dynamic_object_id",
            "source_snippet_id",
            "source_object_id",
            "world_file",
        ):
            _require_nonempty_string(row[name], name)
        if row["object_type"] not in DYNAMIC_OBJECT_TYPES:
            raise ValueError(f"object_type must be one of {DYNAMIC_OBJECT_TYPES}")
        if not isinstance(row["footprint_spec"], dict):
            raise ValueError("footprint_spec must be a JSON object")
        validate_dynamic_object_spec(row["footprint_spec"])
        if row["footprint_spec"].get("object_type") != row["object_type"]:
            raise ValueError("object_type and footprint_spec mismatch")
        for name in (
            "footprint_spec_digest",
            "target_type_policy_digest",
            "history_array_digest",
            "future_array_digest",
            "record_digest",
            "world_semantic_digest",
            "manifest_digest",
        ):
            _require_digest(row[name], name)
        rows.append(row)
    return rows


def _validate_expected_id_set(
    actual: set[str], expected: set[str] | frozenset[str] | None, name: str
) -> None:
    if expected is None:
        return
    if not isinstance(expected, (set, frozenset)) or any(
        not isinstance(value, str) or not value for value in expected
    ):
        raise TypeError(f"expected_{name}s must be a set of non-empty strings")
    if actual != set(expected):
        raise ValueError(f"{name} set mismatch")


def load_event_target_motion_shard(
    input_dir: str | Path,
    *,
    grid: GridSpec,
    expected_generated_event_ids: set[str] | frozenset[str] | None = None,
    expected_base_state_ids: set[str] | frozenset[str] | None = None,
    expected_trajectory_ids: set[str] | frozenset[str] | None = None,
) -> LoadedEventTargetMotionShard:
    """Strictly load and cross-check every file, digest and OracleWorld join."""

    root = Path(input_dir)
    if not root.is_dir():
        raise ValueError("shard input directory does not exist")
    expected_root_entries = {
        _MANIFEST_FILENAME,
        _PAYLOAD_FILENAME,
        _SUMMARY_FILENAME,
        _WORLD_DIRECTORY,
    }
    actual_root_entries = {path.name for path in root.iterdir()}
    if actual_root_entries != expected_root_entries:
        raise ValueError("shard root file set mismatch")
    rows = _load_manifest(root / _MANIFEST_FILENAME)
    generated_event_ids = [row["generated_event_id"] for row in rows]
    if generated_event_ids != sorted(generated_event_ids):
        raise ValueError("manifest generated_event_id order mismatch")
    if len(generated_event_ids) != len(set(generated_event_ids)):
        raise ValueError("duplicate generated_event_id in manifest")
    if [row["row_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("manifest row indexes must be 0..N-1")
    row_world_ids = [row["world_id"] for row in rows]
    if len(row_world_ids) != len(set(row_world_ids)):
        raise ValueError("duplicate world_id in manifest")
    manifest_digests = {row["manifest_digest"] for row in rows}
    if len(manifest_digests) != 1:
        raise ValueError("manifest_digest must be identical in every row")
    stored_manifest_digest = manifest_digests.pop()
    _require_digest(stored_manifest_digest, "manifest_digest")
    base_rows = [
        {key: value for key, value in row.items() if key != "manifest_digest"}
        for row in rows
    ]
    computed_manifest_digest = _manifest_digest(base_rows)
    if stored_manifest_digest != computed_manifest_digest:
        raise ValueError("manifest_digest mismatch")

    summary = _load_json_object(root / _SUMMARY_FILENAME, "shard summary")
    _require_exact_keys(summary, _SUMMARY_KEYS, "shard summary")
    _validate_layout_metadata(summary, "shard summary")

    payload_path = root / _PAYLOAD_FILENAME
    try:
        with np.load(payload_path, allow_pickle=False) as payload:
            if set(payload.files) != set(_NPZ_KEYS):
                raise ValueError("NPZ payload keys mismatch")
            history = payload["history_poses"].copy()
            current = payload["current_poses"].copy()
            future = payload["future_poses"].copy()
            meta_raw = payload["meta_json"]
            if meta_raw.shape != () or meta_raw.dtype.kind not in "US":
                raise ValueError("NPZ meta_json must be a scalar string")
            meta = json.loads(
                str(meta_raw), parse_constant=_reject_json_constant
            )
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("invalid NPZ payload") from exc
    if not isinstance(meta, dict):
        raise ValueError("NPZ meta_json must decode to an object")
    _require_exact_keys(meta, _NPZ_META_KEYS, "NPZ metadata")
    _validate_layout_metadata(meta, "NPZ metadata")
    count = len(rows)
    _validate_array(
        history,
        name="history_poses payload",
        shape=(count, HISTORY_STEPS, 3),
        require_owned_c=True,
    )
    _validate_array(
        current,
        name="current_poses payload",
        shape=(count, 3),
        require_owned_c=True,
    )
    _validate_array(
        future,
        name="future_poses payload",
        shape=(count, FUTURE_STEPS, 3),
        require_owned_c=True,
    )
    if not np.array_equal(current, history[:, CURRENT_INDEX, :]):
        raise ValueError("payload current/history seam mismatch")
    if meta["array_dtype"] != _FLOAT32_LE_DTYPE_TOKEN:
        raise ValueError("NPZ metadata array_dtype mismatch")
    if meta["array_order"] != _ARRAY_ORDER_TOKEN:
        raise ValueError("NPZ metadata array_order mismatch")
    if meta["record_count"] != count:
        raise ValueError("NPZ metadata record_count mismatch")
    if meta["generated_event_ids"] != generated_event_ids:
        raise ValueError("NPZ metadata generated_event_id order mismatch")
    if meta["manifest_digest"] != stored_manifest_digest:
        raise ValueError("NPZ metadata manifest_digest mismatch")

    records: list[EventTargetMotionRecord] = []
    for index, row in enumerate(rows):
        record = create_event_target_motion_record(
            generated_event_id=row["generated_event_id"],
            world_id=row["world_id"],
            base_state_id=row["base_state_id"],
            trajectory_id=row["trajectory_id"],
            target_dynamic_object_id=row["target_dynamic_object_id"],
            source_snippet_id=row["source_snippet_id"],
            source_object_id=row["source_object_id"],
            object_type=row["object_type"],
            footprint_spec=row["footprint_spec"],
            footprint_spec_digest=row["footprint_spec_digest"],
            target_type_policy_digest=row["target_type_policy_digest"],
            history_poses=history[index],
            current_pose=current[index],
            future_poses=future[index],
        )
        for name in (
            "history_array_digest",
            "future_array_digest",
            "record_digest",
        ):
            if getattr(record, name) != row[name]:
                raise ValueError(f"manifest {name} mismatch")
        if row["world_file"] != (
            f"{_WORLD_DIRECTORY}/{_safe_world_filename(record.world_id)}"
        ):
            raise ValueError("manifest world_file mismatch")
        records.append(record)
    ordered = tuple(records)
    if meta["history_array_digests"] != [
        record.history_array_digest for record in ordered
    ]:
        raise ValueError("NPZ metadata history_array_digests mismatch")
    if meta["future_array_digests"] != [
        record.future_array_digest for record in ordered
    ]:
        raise ValueError("NPZ metadata future_array_digests mismatch")
    if meta["record_digests"] != [record.record_digest for record in ordered]:
        raise ValueError("NPZ metadata record_digests mismatch")
    computed_payload_digest = _payload_semantic_digest(
        ordered, stored_manifest_digest
    )
    if meta["payload_semantic_digest"] != computed_payload_digest:
        raise ValueError("payload_semantic_digest mismatch")

    expected_summary = _summary(
        count, stored_manifest_digest, computed_payload_digest
    )
    if summary != expected_summary:
        raise ValueError("shard summary mismatch")

    world_directory = root / _WORLD_DIRECTORY
    if not world_directory.is_dir():
        raise ValueError("oracle_worlds must be a directory")
    expected_world_files = {
        _safe_world_filename(record.world_id) for record in ordered
    }
    actual_world_files = {path.name for path in world_directory.iterdir()}
    if actual_world_files != expected_world_files or any(
        not path.is_file() for path in world_directory.iterdir()
    ):
        raise ValueError("world file set mismatch")
    worlds: dict[str, OracleWorld] = {}
    for record, row in zip(ordered, rows):
        try:
            world = load_dataclass(
                world_directory / _safe_world_filename(record.world_id)
            )
        except Exception as exc:
            raise ValueError(f"invalid OracleWorld file for {record.world_id}") from exc
        if not isinstance(world, OracleWorld):
            raise ValueError("world artifact did not decode to OracleWorld")
        computed_world_digest = compute_oracle_world_semantic_digest(world)
        if row["world_semantic_digest"] != computed_world_digest:
            raise ValueError("world_semantic_digest mismatch")
        validate_event_target_motion_world_join(record, world, grid)
        worlds[record.world_id] = world

    _validate_expected_id_set(
        {record.generated_event_id for record in ordered},
        expected_generated_event_ids,
        "generated_event_id",
    )
    _validate_expected_id_set(
        {record.base_state_id for record in ordered},
        expected_base_state_ids,
        "base_state_id",
    )
    _validate_expected_id_set(
        {record.trajectory_id for record in ordered},
        expected_trajectory_ids,
        "trajectory_id",
    )
    return LoadedEventTargetMotionShard(
        records=ordered,
        worlds=worlds,
        manifest_digest=stored_manifest_digest,
        payload_semantic_digest=computed_payload_digest,
        summary=deepcopy(summary),
    )


def build_renderer_scene(
    record: EventTargetMotionRecord, oracle_context: OracleContext
) -> RendererScene:
    """Copy context history/specs and append target history without oracle future."""

    validate_event_target_motion_record(record)
    if not isinstance(oracle_context, OracleContext):
        raise TypeError("oracle_context must be an OracleContext")
    if oracle_context.base_state_id != record.base_state_id:
        raise ValueError("oracle_context base_state_id mismatch")
    history_ids = set(oracle_context.dynamic_object_history)
    future_ids = set(oracle_context.dynamic_object_future)
    spec_ids = set(oracle_context.dynamic_object_specs)
    if history_ids != future_ids or history_ids != spec_ids:
        raise ValueError("oracle_context history/future/spec ids must align")
    if record.target_dynamic_object_id in history_ids:
        raise ValueError("target id collides with oracle context")
    histories: dict[str, np.ndarray] = {}
    specs: dict[str, dict[str, object]] = {}
    for object_id in sorted(history_ids):
        history = _validate_array(
            oracle_context.dynamic_object_history[object_id],
            name=f"oracle_context history[{object_id!r}]",
            shape=(HISTORY_STEPS, 3),
            require_owned_c=False,
        )
        _validate_array(
            oracle_context.dynamic_object_future[object_id],
            name=f"oracle_context future[{object_id!r}]",
            shape=(FUTURE_STEPS, 3),
            require_owned_c=False,
        )
        spec = _canonical_json_copy(oracle_context.dynamic_object_specs[object_id])
        validate_dynamic_object_spec(spec)
        histories[object_id] = np.array(
            history, dtype=np.float32, order="C", copy=True
        )
        specs[object_id] = spec
    histories[record.target_dynamic_object_id] = np.array(
        record.history_poses, dtype=np.float32, order="C", copy=True
    )
    specs[record.target_dynamic_object_id] = _canonical_json_copy(
        record.footprint_spec
    )
    return RendererScene(
        dynamic_object_history=histories,
        dynamic_object_specs=specs,
    )
