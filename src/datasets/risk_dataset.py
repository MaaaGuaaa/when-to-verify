"""Schema-v3 RiskSample assembly with an explicit input/label boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
import errno
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

import numpy as np

from src.contracts import (
    DYNAMIC_OBJECT_TYPES,
    HISTORY_CHANNELS,
    POSE_TIME_LAYOUT_VERSION,
    SCHEMA_VERSION,
    TRAJECTORY_CHANNELS,
    BaseState,
    GridSpec,
    LocalTrajectory,
    OracleContext,
    OracleWorld,
    RiskSample,
    assert_no_oracle_leakage,
    build_grid_spec,
    validate_risk_sample,
)
from src.generation.dynamic_object_transplant import (
    TransplantError,
    TransplantedDynamicObject,
    footprint_from_spec,
    transplant_snippet,
)
from src.datasets.snippet_library import MOTION_SNIPPET_LAYOUT, MotionSnippet
from src.datasets.risk_evaluation_metadata import (
    OOD_ROUTING_RULE_VERSION,
    derive_production_evaluation_record,
    derive_robot_footprint_provenance,
)
from src.generation.event_sampler import (
    SOP05_GENERATOR_ALGORITHM_VERSION,
    GeneratedEvent,
)
from src.generation.observation_renderer import (
    RENDERER_LAYOUT_VERSION,
    RenderedObservation,
    render_observation,
)
from src.generation.paired_variants import (
    JOINT_ENVIRONMENT_PAIR_VERSION,
    PAIRED_GENERATOR_ALGORITHM_VERSION,
    PAIRED_GROUP_CONTRACT_VERSION,
    PairedEventGroup,
    PairedVariant,
    PairedVariantConfig,
    normalize_paired_variant_config,
)
from src.generation.risk_gt import (
    RISK_GT_VERSION,
    RiskGroundTruth,
    compute_hidden_risk_gt,
    resolve_no_object_clearance_sentinel,
)
from src.generation.risk_sidecars import (
    RiskLabelSidecar,
    build_risk_label_sidecar,
)
from src.generation.sop06_pipeline import render_sop06_partial_pair_group
from src.generation.structural_blindspot import StructuralBlindSpot
from src.geometry import (
    Footprint,
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint,
    signed_clearance,
    trajectory_signed_clearances,
    wrap_angle,
)
from src.utils.seeding import stable_digest


_RISK_CONFIG_KEYS = frozenset(
    {"sigma_distance_m", "sigma_time_s", "near_miss_distance_m"}
)
_METADATA_KEYS = frozenset(
    {"schema_version", "renderer", "trajectory_id", "provenance", "label_audit"}
)
_RENDERER_METADATA_KEYS = frozenset(
    {
        "renderer_layout_version",
        "base_state_id",
        "sensor_config_digest",
        "static_occupancy_digest",
    }
)
_LABEL_AUDIT_KEYS = frozenset(
    {
        "risk_gt_version",
        "pose_time_layout_version",
        "critical_object_id",
        "critical_object_type",
        "time_to_min_clearance_s",
        "has_hidden_target",
    }
)
_FORBIDDEN_METADATA_KEY_TOKENS = (
    "future",
    "oracle",
    "clearance_sequence",
    "dynamic_object_trajectories",
    "hidden_object_ids",
)
_RISK_INPUT_ADAPTER_VERSION = "sop06_group_to_risk_samples_v2"
_EVALUATION_ONLY_SOURCE_PROVENANCE_KEYS = frozenset(
    {"ood_tag", "ood_evidence"}
)
_RISK_SHARD_SNAPSHOT_MEMBERS = (
    "samples.npz",
    "metadata.jsonl",
    "summary.json",
)


@dataclass(frozen=True)
class _RiskSnapshotIdentity:
    device: int
    inode: int
    file_type: int


def _risk_snapshot_identity(descriptor: int) -> _RiskSnapshotIdentity:
    metadata = os.fstat(descriptor)
    return _RiskSnapshotIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        file_type=stat.S_IFMT(metadata.st_mode),
    )


def _same_risk_snapshot_identity(
    first: _RiskSnapshotIdentity, second: _RiskSnapshotIdentity
) -> bool:
    return (
        first.device == second.device
        and first.inode == second.inode
        and first.file_type == second.file_type
    )


def _open_risk_snapshot_root_nofollow(
    root: Path,
) -> tuple[int, _RiskSnapshotIdentity]:
    try:
        descriptor = os.open(
            root,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(
                f"risk shard root must not be a symlink: {root}"
            ) from exc
        if exc.errno == errno.ENOENT:
            raise ValueError(f"risk shard root not found: {root}") from exc
        raise ValueError(f"failed to open risk shard root: {root}") from exc
    try:
        identity = _risk_snapshot_identity(descriptor)
        if identity.file_type != stat.S_IFDIR:
            raise ValueError(f"risk shard root must be a directory: {root}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, identity


def _open_risk_snapshot_member_nofollow(
    root_fd: int, name: str
) -> tuple[int, _RiskSnapshotIdentity]:
    if name not in _RISK_SHARD_SNAPSHOT_MEMBERS:
        raise ValueError(f"unexpected risk shard member name: {name}")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(
                f"risk shard member must not be a symlink: {name}"
            ) from exc
        if exc.errno == errno.ENOENT:
            raise ValueError(f"risk shard member not found: {name}") from exc
        raise ValueError(f"failed to open risk shard member: {name}") from exc
    try:
        identity = _risk_snapshot_identity(descriptor)
        if identity.file_type != stat.S_IFREG:
            raise ValueError(
                f"risk shard member must be a direct regular file: {name}"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, identity


def _read_risk_snapshot_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1 << 20)
        if not chunk:
            break
        chunks.append(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return b"".join(chunks)


def _load_risk_shard_from_snapshot_directory(
    snapshot_root: Path, *, grid: GridSpec
) -> Any:
    from src.datasets.shard_writer import load_risk_shard

    return load_risk_shard(snapshot_root, grid=grid)


class PinnedRiskShardSnapshot:
    """Hold a verified risk root and every formal member through a gate."""

    def __init__(self, output_dir: str | Path, *, grid: GridSpec) -> None:
        self._root = Path(output_dir)
        self._grid = grid
        self._root_fd: int | None = None
        self._root_identity: _RiskSnapshotIdentity | None = None
        self._member_fds: dict[str, int] = {}
        self._member_identities: dict[str, _RiskSnapshotIdentity] = {}
        self._snapshots: dict[str, bytes] = {}
        self._loaded_shard: Any | None = None

    @property
    def loaded_shard(self) -> Any:
        if self._loaded_shard is None:
            raise RuntimeError("pinned risk shard has not been entered")
        return self._loaded_shard

    def __enter__(self) -> PinnedRiskShardSnapshot:
        if self._root_fd is not None:
            raise RuntimeError("pinned risk shard cannot be entered twice")
        root_fd, root_identity = _open_risk_snapshot_root_nofollow(
            self._root
        )
        self._root_fd = root_fd
        self._root_identity = root_identity
        try:
            actual_files = set(os.listdir(root_fd))
            required_files = set(_RISK_SHARD_SNAPSHOT_MEMBERS)
            missing = required_files - actual_files
            if missing:
                raise ValueError(
                    "incomplete risk shard: missing "
                    + ", ".join(sorted(missing))
                )
            unexpected = actual_files - required_files
            if unexpected:
                raise ValueError(
                    "unexpected risk shard files: "
                    + ", ".join(sorted(unexpected))
                )

            for name in _RISK_SHARD_SNAPSHOT_MEMBERS:
                descriptor, identity = _open_risk_snapshot_member_nofollow(
                    root_fd, name
                )
                self._member_fds[name] = descriptor
                self._member_identities[name] = identity
                self._snapshots[name] = _read_risk_snapshot_descriptor(
                    descriptor
                )

            with tempfile.TemporaryDirectory(
                prefix="risk-shard-immutable-snapshot-"
            ) as temporary:
                snapshot_root = Path(temporary) / "risk-shard"
                snapshot_root.mkdir(mode=0o700)
                for name in _RISK_SHARD_SNAPSHOT_MEMBERS:
                    snapshot_path = snapshot_root / name
                    with snapshot_path.open("xb") as handle:
                        handle.write(self._snapshots[name])
                        handle.flush()
                        os.fsync(handle.fileno())
                self._loaded_shard = (
                    _load_risk_shard_from_snapshot_directory(
                        snapshot_root, grid=self._grid
                    )
                )
            return self
        except BaseException:
            self.close()
            raise

    def verify_unchanged(self) -> None:
        """Fail unless pinned bytes, membership, and identities still match."""

        root_fd = self._root_fd
        root_identity = self._root_identity
        if (
            root_fd is None
            or root_identity is None
            or self._loaded_shard is None
        ):
            raise RuntimeError("pinned risk shard is not open")

        for name in _RISK_SHARD_SNAPSHOT_MEMBERS:
            if (
                _read_risk_snapshot_descriptor(self._member_fds[name])
                != self._snapshots[name]
            ):
                raise ValueError(
                    "risk shard member content changed during complete load: "
                    f"{name}"
                )

        actual_files = set(os.listdir(root_fd))
        required_files = set(_RISK_SHARD_SNAPSHOT_MEMBERS)
        if actual_files != required_files:
            missing = sorted(required_files - actual_files)
            unexpected = sorted(actual_files - required_files)
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise ValueError(
                "risk shard membership changed during complete load: "
                + "; ".join(details)
            )

        for name in _RISK_SHARD_SNAPSHOT_MEMBERS:
            descriptor, identity = _open_risk_snapshot_member_nofollow(
                root_fd, name
            )
            os.close(descriptor)
            if not _same_risk_snapshot_identity(
                self._member_identities[name], identity
            ):
                raise ValueError(
                    "risk shard member identity changed during complete load: "
                    f"{name}"
                )

        verification_fd, verification_identity = (
            _open_risk_snapshot_root_nofollow(self._root)
        )
        os.close(verification_fd)
        if not _same_risk_snapshot_identity(
            root_identity, verification_identity
        ):
            raise ValueError(
                "risk shard root identity changed during complete load"
            )

    def close(self) -> None:
        for descriptor in self._member_fds.values():
            os.close(descriptor)
        self._member_fds.clear()
        self._member_identities.clear()
        self._snapshots.clear()
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None
        self._root_identity = None
        self._loaded_shard = None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def pin_risk_shard_snapshot(
    output_dir: str | Path, *, grid: GridSpec
) -> PinnedRiskShardSnapshot:
    """Create a guard whose fixed FDs remain open until context exit."""

    return PinnedRiskShardSnapshot(output_dir, grid=grid)


def load_hardened_risk_shard_snapshot(
    output_dir: str | Path, *, grid: GridSpec
) -> Any:
    """Load and immediately verify one pinned formal risk shard."""

    with pin_risk_shard_snapshot(output_dir, grid=grid) as pinned:
        loaded = pinned.loaded_shard
        pinned.verify_unchanged()
        return loaded


@dataclass(frozen=True)
class RiskBuildInput:
    """One sample source with observation history separated from label future."""

    sample_id: str
    pair_group_id: str
    event_type: str
    base_state: BaseState
    trajectory: LocalTrajectory
    oracle_world: OracleWorld
    observed_static_occupancy: np.ndarray
    scene_dynamic_history: Mapping[str, np.ndarray]
    scene_dynamic_specs: Mapping[str, dict[str, object]]
    hidden_object_ids: tuple[str, ...]
    sensor_config: StructuralBlindSpot | None
    provenance: Mapping[str, object]


def _canonical_config_digest(config: Mapping[str, object]) -> str:
    try:
        payload = json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("base_config must be finite canonical JSON") from exc
    return stable_digest(payload, size=16)


def _canonical_paired_config(value: PairedVariantConfig) -> PairedVariantConfig:
    if not isinstance(value, PairedVariantConfig):
        raise TypeError("paired_config must be a PairedVariantConfig")
    try:
        normalized = normalize_paired_variant_config(value.as_dict())
    except (TypeError, ValueError) as exc:
        raise ValueError("canonical paired config validation failed") from exc
    if normalized != value:
        raise ValueError("paired_config differs from its canonical paired config")
    return normalized


def _deep_owned_copy(value: Any) -> Any:
    """Deep-copy a formal input graph while preserving ndarray write flags."""

    memo: dict[int, object] = {}
    seen: set[int] = set()

    def seed_arrays(item: Any) -> None:
        item_id = id(item)
        if item_id in seen:
            return
        seen.add(item_id)
        if isinstance(item, np.ndarray):
            owned = np.array(item, dtype=item.dtype, order="K", copy=True)
            if not item.flags.writeable:
                owned.setflags(write=False)
            memo[item_id] = owned
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                seed_arrays(key)
                seed_arrays(child)
            return
        if isinstance(item, (tuple, list, set, frozenset)):
            for child in item:
                seed_arrays(child)
            return
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                seed_arrays(getattr(item, field.name))

    seed_arrays(value)
    return deepcopy(value, memo)


def _copy_sop06_scene_history(
    *,
    base_state: BaseState,
    oracle_context: OracleContext,
    variant: PairedVariant,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    histories: dict[str, np.ndarray] = {}
    specs: dict[str, dict[str, object]] = {}
    for object_id in sorted(base_state.dynamic_object_ids):
        histories[object_id] = np.array(
            base_state.visible_dynamic_object_history[object_id],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        histories[object_id].setflags(write=False)
        specs[object_id] = deepcopy(
            base_state.visible_dynamic_object_specs[object_id]
        )
    for object_id in sorted(oracle_context.dynamic_object_history):
        history = oracle_context.dynamic_object_history[object_id]
        spec = oracle_context.dynamic_object_specs[object_id]
        if object_id in histories:
            if not np.array_equal(histories[object_id], history):
                raise ValueError(
                    "overlapping BaseState/OracleContext history mismatch"
                )
            if specs[object_id] != spec:
                raise ValueError(
                    "overlapping BaseState/OracleContext spec mismatch"
                )
            continue
        histories[object_id] = np.array(
            history, dtype=np.float32, order="C", copy=True
        )
        histories[object_id].setflags(write=False)
        specs[object_id] = deepcopy(spec)
    if variant.target is not None:
        target_id = variant.target.target_dynamic_object_id
        if target_id in histories:
            raise ValueError("paired target id collides with scene history")
        histories[target_id] = np.array(
            variant.target.history_poses,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        histories[target_id].setflags(write=False)
        specs[target_id] = deepcopy(variant.target.footprint_spec)
    return histories, specs


def _validate_formal_inputs(
    *,
    group: PairedEventGroup,
    mother_event: GeneratedEvent,
    source_snippet: MotionSnippet,
    base_state: BaseState,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    paired_config: PairedVariantConfig,
    dataset_seed: int,
) -> tuple[str, int, int]:
    if not isinstance(group, PairedEventGroup):
        raise TypeError("group must be a PairedEventGroup")
    if not isinstance(mother_event, GeneratedEvent):
        raise TypeError("mother_event must be a GeneratedEvent")
    if not isinstance(source_snippet, MotionSnippet):
        raise TypeError("source_snippet must be a MotionSnippet")
    if not isinstance(base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(trajectory, LocalTrajectory):
        raise TypeError("trajectory must be a LocalTrajectory")
    if not isinstance(oracle_context, OracleContext):
        raise TypeError("oracle_context must be an OracleContext")
    if not isinstance(paired_config, PairedVariantConfig):
        raise TypeError("paired_config must be a PairedVariantConfig")
    if isinstance(dataset_seed, (bool, np.bool_)) or not isinstance(
        dataset_seed, (Integral, np.integer)
    ):
        raise TypeError("dataset_seed must be a non-negative integer")
    dataset_seed = int(dataset_seed)
    if dataset_seed < 0:
        raise ValueError("dataset_seed must be a non-negative integer")

    record = mother_event.target_motion_record
    target = mother_event.target
    if mother_event.generated_event_id != record.generated_event_id:
        raise ValueError("mother event/record generated_event_id mismatch")
    if not isinstance(mother_event.world, OracleWorld):
        raise TypeError("mother event world must be an OracleWorld")
    if not isinstance(target, TransplantedDynamicObject):
        raise TypeError("mother event target must be a TransplantedDynamicObject")
    if trajectory.trajectory_id != record.trajectory_id:
        raise ValueError("trajectory_id does not match mother record")
    if record.base_state_id != base_state.state_id:
        raise ValueError("base_state_id does not match mother record")
    if oracle_context.base_state_id != base_state.state_id:
        raise ValueError("oracle_context/base_state IDs must match")
    if source_snippet.split != base_state.split:
        raise ValueError("source snippet split differs from base split")
    identity_fields = (
        (source_snippet.snippet_id, record.source_snippet_id, "snippet"),
        (source_snippet.source_object_id, record.source_object_id, "source object"),
        (source_snippet.object_type, record.object_type, "object type"),
        (target.snippet_id, record.source_snippet_id, "mother target snippet"),
        (target.source_object_id, record.source_object_id, "mother target source object"),
        (target.object_type, record.object_type, "mother target object type"),
    )
    for actual, expected, label in identity_fields:
        if actual != expected:
            raise ValueError(f"{label} identity mismatch")
    if source_snippet.footprint != record.footprint_spec.get("footprint"):
        raise ValueError("source snippet footprint differs from mother record")
    if target.footprint_spec != record.footprint_spec:
        raise ValueError("mother target footprint differs from mother record")
    target_provenance = target.provenance
    if not isinstance(target_provenance, Mapping):
        raise TypeError("mother target provenance must be a mapping")
    source_recording_id = _require_nonempty_string(
        source_snippet.source_recording_id, name="source recording"
    )
    if not source_recording_id.strip():
        raise ValueError("source recording must be a non-empty string")
    source_session_id = _require_nonempty_string(
        source_snippet.source_session_id, name="source session"
    )
    if not source_session_id.strip():
        raise ValueError("source session must be a non-empty string")
    for field, label in (
        ("source_recording_id", "source recording"),
        ("source_session_id", "source session"),
    ):
        value = target_provenance.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"mother target {label} must be non-empty")
    if target_provenance.get("source_recording_id") != source_recording_id:
        raise ValueError("source recording differs from mother target provenance")
    if target_provenance.get("source_session_id") != source_session_id:
        raise ValueError("source session differs from mother target provenance")
    base_recording_id = _require_nonempty_string(
        base_state.recording_id, name="base recording"
    )
    if not isinstance(oracle_context.metadata, Mapping):
        raise TypeError("oracle_context metadata must be a mapping")
    if oracle_context.metadata.get("source_recording_id") != base_recording_id:
        raise ValueError("oracle_context/base recording identity mismatch")
    base_session_id = _require_nonempty_string(
        base_state.metadata.get("session_id"), name="base session"
    )
    sample_count = int(MOTION_SNIPPET_LAYOUT["sample_count"])
    for name, array, shape in (
        ("positions", source_snippet.positions, (sample_count, 2)),
        ("velocities", source_snippet.velocities, (sample_count, 2)),
        ("headings", source_snippet.headings, (sample_count,)),
    ):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"source snippet {name} must be an np.ndarray")
        if array.dtype != np.float32 or array.shape != shape:
            raise ValueError(f"source snippet {name} shape/dtype mismatch")
        if not np.isfinite(array).all():
            raise ValueError(f"source snippet {name} contains NaN/Inf")
    if source_snippet.duration_s != float(MOTION_SNIPPET_LAYOUT["duration_s"]):
        raise ValueError("source snippet duration violates the frozen layout")
    if not isinstance(source_snippet.provenance, dict):
        raise TypeError("source snippet provenance must be a dict")
    snippet_scalars = (
        source_snippet.start_timestamp,
        source_snippet.duration_s,
        source_snippet.mean_speed_mps,
        source_snippet.max_acceleration_mps2,
        source_snippet.mean_abs_curvature_per_m,
    )
    if not all(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, Real)
        and math.isfinite(float(value))
        for value in snippet_scalars
    ):
        raise ValueError("source snippet statistics must be finite")

    current_index = target_provenance.get("source_current_index")
    expected_current_index = int(MOTION_SNIPPET_LAYOUT["current_index"])
    if type(current_index) is not int or current_index != expected_current_index:
        raise ValueError("mother target source_current_index mismatch")
    current_xy = target_provenance.get("candidate_current_xy")
    if not isinstance(current_xy, (list, tuple)) or len(current_xy) != 2:
        raise ValueError("mother target candidate_current_xy must contain two values")
    current_xy_array = np.asarray(current_xy, dtype=np.float64)
    if not np.isfinite(current_xy_array).all():
        raise ValueError("mother target candidate_current_xy must be finite")
    rotation_rad = _finite_float(
        target_provenance.get("rotation_rad"),
        name="mother target rotation_rad",
    )
    cosine = math.cos(rotation_rad)
    sine = math.sin(rotation_rad)
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    positions64 = source_snippet.positions.astype(np.float64)
    transformed_xy = (
        (positions64 - positions64[current_index]) @ rotation.T
        + current_xy_array
    )
    transformed_headings = (
        source_snippet.headings.astype(np.float64) + rotation_rad
    )
    reconstructed = np.column_stack(
        (transformed_xy, transformed_headings)
    ).astype(np.float32)
    target_poses = np.ascontiguousarray(
        np.vstack((target.history_poses, target.future_poses)), dtype=np.float32
    )
    record_poses = np.ascontiguousarray(
        np.vstack((record.history_poses, record.future_poses)), dtype=np.float32
    )
    if not np.array_equal(reconstructed, target_poses):
        raise ValueError("source snippet motion does not reconstruct mother target")
    if not np.array_equal(reconstructed, record_poses):
        raise ValueError("source snippet motion does not reconstruct mother record")
    if not np.array_equal(reconstructed[current_index], target.current_pose) or not (
        np.array_equal(reconstructed[current_index], record.current_pose)
    ):
        raise ValueError("source snippet current pose differs from mother identity")

    if paired_config.schema_version != SCHEMA_VERSION:
        raise ValueError(f"paired_config schema_version must be {SCHEMA_VERSION}")
    if paired_config.paired_generator_algorithm_version != (
        PAIRED_GENERATOR_ALGORITHM_VERSION
    ):
        raise ValueError("paired_config generator version mismatch")
    if paired_config.group_contract_version != PAIRED_GROUP_CONTRACT_VERSION:
        raise ValueError("paired_config group contract version mismatch")
    if paired_config.digest != group.paired_config_digest:
        raise ValueError("paired_config digest differs from group")
    mother_metadata = mother_event.world.metadata
    if not isinstance(mother_metadata, Mapping):
        raise TypeError("mother world metadata must be a mapping")
    joint_version = mother_metadata.get("joint_pair_generator_algorithm_version")
    if joint_version is not None:
        if joint_version == JOINT_ENVIRONMENT_PAIR_VERSION:
            raise ValueError(f"mother uses retired {JOINT_ENVIRONMENT_PAIR_VERSION}")
        raise ValueError("mother contains unsupported joint-pair identity")
    if mother_metadata.get("generator_algorithm_version") != (
        SOP05_GENERATOR_ALGORITHM_VERSION
    ):
        raise ValueError(
            "mother generator_algorithm_version must equal "
            f"{SOP05_GENERATOR_ALGORITHM_VERSION}"
        )
    transplant_seed = target.provenance.get("seed")
    if type(transplant_seed) is not int or transplant_seed < 0:
        raise ValueError("mother target transplant seed must be a non-negative integer")
    paired_seeds = {variant.world.metadata.get("paired_seed") for variant in group.variants}
    if len(paired_seeds) != 1:
        raise ValueError("formal SOP06 group must have one paired seed")
    paired_seed = next(iter(paired_seeds))
    if type(paired_seed) is not int or paired_seed < 0:
        raise ValueError("paired seed must be a non-negative integer")
    return base_session_id, transplant_seed, paired_seed


def _owned_oracle_world(world: OracleWorld) -> OracleWorld:
    static = np.array(world.static_occupancy, dtype=np.float32, order="C", copy=True)
    static.setflags(write=False)
    trajectories: dict[str, np.ndarray] = {}
    for object_id, value in world.dynamic_object_trajectories.items():
        owned = np.array(value, dtype=np.float32, order="C", copy=True)
        owned.setflags(write=False)
        trajectories[object_id] = owned
    return OracleWorld(
        world_id=world.world_id,
        base_state_id=world.base_state_id,
        static_occupancy=static,
        dynamic_object_trajectories=trajectories,
        dynamic_object_specs=deepcopy(world.dynamic_object_specs),
        occluders=tuple(deepcopy(item) for item in world.occluders),
        blind_spot_config=deepcopy(world.blind_spot_config),
        random_seed=world.random_seed,
        metadata=deepcopy(world.metadata),
    )


def _trajectory_primitive(trajectory: LocalTrajectory) -> dict[str, float]:
    if (
        not isinstance(trajectory.controls, np.ndarray)
        or trajectory.controls.ndim != 2
        or trajectory.controls.shape[1] != 2
        or trajectory.controls.dtype != np.float32
        or not np.isfinite(trajectory.controls).all()
    ):
        raise ValueError("trajectory controls violate the primitive contract")
    v = _finite_float(trajectory.metadata.get("v"), name="trajectory metadata v")
    omega = _finite_float(
        trajectory.metadata.get("omega"), name="trajectory metadata omega"
    )
    expected = np.asarray([v, omega], dtype=np.float32)
    if not np.allclose(trajectory.controls, expected, rtol=0.0, atol=1e-7):
        raise ValueError("trajectory controls differ from metadata primitive")
    return {"v_mps": v, "omega_radps": omega}


def _occluder_audit(world: OracleWorld) -> list[dict[str, object]]:
    audit: list[dict[str, object]] = []
    required = (
        "occluder_id",
        "type",
        "pose",
        "length_m",
        "width_m",
        "placement_strategy",
    )
    for index, raw in enumerate(world.occluders):
        if not isinstance(raw, Mapping):
            raise TypeError(f"occluders[{index}] must be a mapping")
        item = {key: deepcopy(raw.get(key)) for key in required}
        for key in ("occluder_id", "type", "placement_strategy"):
            _require_nonempty_string(item[key], name=f"occluders[{index}].{key}")
        pose = item["pose"]
        if not isinstance(pose, (list, tuple)) or len(pose) != 3:
            raise ValueError(f"occluders[{index}].pose must contain three values")
        item["pose"] = [
            _finite_float(value, name=f"occluders[{index}].pose") for value in pose
        ]
        for key in ("length_m", "width_m"):
            item[key] = _finite_float(item[key], name=f"occluders[{index}].{key}")
            if item[key] <= 0.0:
                raise ValueError(f"occluders[{index}].{key} must be positive")
        audit.append(item)
    return audit


def _build_formal_source(
    *,
    variant: PairedVariant,
    mother_event: GeneratedEvent,
    source_snippet: MotionSnippet,
    base_state: BaseState,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    base_config: Mapping[str, object],
    paired_config: PairedVariantConfig,
    risk_config: Mapping[str, object],
    dataset_seed: int,
    base_session_id: str,
    transplant_seed: int,
    paired_seed: int,
) -> RiskBuildInput:
    record = mother_event.target_motion_record
    blind_spot_config = variant.world.blind_spot_config
    if not isinstance(blind_spot_config, Mapping) or (
        blind_spot_config.get("kind") != "environment"
    ):
        raise ValueError("formal SOP07 adapter requires an environment variant")
    histories, specs = _copy_sop06_scene_history(
        base_state=base_state,
        oracle_context=oracle_context,
        variant=variant,
    )
    world = _owned_oracle_world(variant.world)
    pair_group_id = _require_nonempty_string(
        variant.world.metadata.get("pair_group_id"), name="pair_group_id"
    )
    target_footprint = record.footprint_spec.get("footprint")
    if not isinstance(target_footprint, Mapping):
        raise ValueError("mother target footprint must be a mapping")
    target_footprint_kind = _require_nonempty_string(
        target_footprint.get("kind"), name="target_footprint_kind"
    )
    base_config_digest = _canonical_config_digest(base_config)
    risk_config_digest = _canonical_config_digest(risk_config)
    snippet_motion_digest = stable_digest(
        source_snippet.snippet_id,
        source_snippet.start_timestamp,
        source_snippet.positions.tobytes(order="C"),
        source_snippet.velocities.tobytes(order="C"),
        source_snippet.headings.tobytes(order="C"),
        size=16,
    )
    occluders = _occluder_audit(world)
    trajectory_primitive = _trajectory_primitive(trajectory)
    seed_namespace = (
        f"sop07/{base_state.split}/seed-{dataset_seed}/"
        f"{mother_event.generated_event_id}"
    )
    identity = (
        _RISK_INPUT_ADAPTER_VERSION,
        base_state.split,
        base_state.recording_id,
        base_session_id,
        base_state.state_id,
        source_snippet.source_recording_id,
        source_snippet.source_session_id,
        source_snippet.source_object_id,
        source_snippet.snippet_id,
        snippet_motion_digest,
        mother_event.generated_event_id,
        mother_event.world.world_id,
        record.target_dynamic_object_id,
        record.target_type_policy_digest,
        trajectory.trajectory_id,
        pair_group_id,
        variant.variant_kind,
        world.world_id,
        transplant_seed,
        paired_seed,
        dataset_seed,
        paired_config.digest,
        base_config_digest,
        risk_config_digest,
        json.dumps(
            trajectory_primitive, sort_keys=True, separators=(",", ":")
        ),
        json.dumps(occluders, sort_keys=True, separators=(",", ":")),
    )
    sample_id = f"{base_state.split}-" + stable_digest(*identity, size=12)
    provenance = _canonical_metadata_copy(
        {
            "risk_input_adapter_version": _RISK_INPUT_ADAPTER_VERSION,
            "base_recording_id": base_state.recording_id,
            "base_session_id": base_session_id,
            "source_recording_id": source_snippet.source_recording_id,
            "source_session_id": source_snippet.source_session_id,
            "source_snippet_id": source_snippet.snippet_id,
            "source_object_id": source_snippet.source_object_id,
            "source_snippet_motion_digest": snippet_motion_digest,
            "seed_namespace": seed_namespace,
            "sop05_transplant_seed": transplant_seed,
            "sop06_paired_seed": paired_seed,
            "sop07_dataset_seed": dataset_seed,
            "target_object_type": record.object_type,
            "target_footprint_kind": target_footprint_kind,
            "target_type_policy_digest": record.target_type_policy_digest,
            "blind_spot_type": blind_spot_config["kind"],
            "blind_region_digest": blind_spot_config.get("blind_region_digest"),
            "generator_algorithm_version": SOP05_GENERATOR_ALGORITHM_VERSION,
            "paired_generator_algorithm_version": (
                PAIRED_GENERATOR_ALGORITHM_VERSION
            ),
            "pair_group_contract_version": PAIRED_GROUP_CONTRACT_VERSION,
            "paired_config_digest": paired_config.digest,
            "base_config_digest": base_config_digest,
            "risk_config_digest": risk_config_digest,
            "variant_kind": variant.variant_kind,
            "world_id": world.world_id,
            "trajectory_primitive": trajectory_primitive,
            "occluders": occluders,
            **{
                key: deepcopy(source_snippet.provenance[key])
                for key in _EVALUATION_ONLY_SOURCE_PROVENANCE_KEYS
                if key in source_snippet.provenance
            },
        },
        name="provenance",
    )
    hidden_object_ids = (
        ()
        if variant.target is None
        else (variant.target.target_dynamic_object_id,)
    )
    return RiskBuildInput(
        sample_id=sample_id,
        pair_group_id=pair_group_id,
        event_type=variant.variant_kind,
        base_state=base_state,
        trajectory=trajectory,
        oracle_world=world,
        observed_static_occupancy=world.static_occupancy,
        scene_dynamic_history=histories,
        scene_dynamic_specs=specs,
        hidden_object_ids=hidden_object_ids,
        sensor_config=None,
        provenance=provenance,
    )


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _query_map(
    value: Any,
    *,
    name: str,
    grid: GridSpec,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an np.ndarray")
    if value.dtype != np.float32:
        raise TypeError(f"{name} dtype must be float32")
    if value.shape != (grid.height, grid.width):
        raise ValueError(
            f"{name} shape must be ({grid.height}, {grid.width})"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def build_trajectory_channels(
    trajectory: LocalTrajectory,
    grid: GridSpec,
) -> np.ndarray:
    """Stack the four frozen query maps without conversion or reordering."""

    if not isinstance(trajectory, LocalTrajectory):
        raise TypeError("trajectory must be a LocalTrajectory")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    by_channel = {
        "swept_volume_mask": trajectory.swept_mask,
        "time_to_arrival_map": trajectory.tta_map,
        "braking_margin_map": trajectory.braking_map,
        "centerline_map": trajectory.centerline_map,
    }
    if tuple(by_channel) != TRAJECTORY_CHANNELS:
        raise RuntimeError("trajectory query-map order violates the frozen contract")
    arrays = [
        _query_map(by_channel[name], name=name, grid=grid)
        for name in TRAJECTORY_CHANNELS
    ]
    return np.ascontiguousarray(np.stack(arrays, axis=0), dtype=np.float32)


def _validate_metadata_value(value: object, *, path: str) -> None:
    if isinstance(value, np.ndarray):
        raise TypeError(f"metadata {path} must not contain ndarray payloads")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"metadata {path} keys must be non-empty strings")
            lowered = key.lower()
            if any(token in lowered for token in _FORBIDDEN_METADATA_KEY_TOKENS):
                raise ValueError(f"metadata {path}.{key} contains a forbidden payload key")
            _validate_metadata_value(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_metadata_value(child, path=f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"metadata {path} must contain only finite values")
        return
    if isinstance(value, (np.generic, Real)):
        raise TypeError(f"metadata {path} must contain JSON-native scalar values")
    raise TypeError(f"metadata {path} contains a non-JSON value")


def _canonical_metadata_copy(value: Mapping[str, object], *, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    copied = dict(value)
    _validate_metadata_value(copied, path=name)
    return json.loads(
        json.dumps(
            copied,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _normalized_risk_config(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("risk_config must be a mapping")
    config = dict(value)
    if set(config) != _RISK_CONFIG_KEYS:
        raise ValueError(
            f"risk_config keys must be exactly {sorted(_RISK_CONFIG_KEYS)}"
        )
    return config


def _validate_source_join(source: RiskBuildInput) -> None:
    _require_nonempty_string(source.sample_id, name="sample_id")
    _require_nonempty_string(source.pair_group_id, name="pair_group_id")
    _require_nonempty_string(source.event_type, name="event_type")
    if not isinstance(source.base_state, BaseState):
        raise TypeError("base_state must be a BaseState")
    if not isinstance(source.trajectory, LocalTrajectory):
        raise TypeError("trajectory must be a LocalTrajectory")
    if not isinstance(source.oracle_world, OracleWorld):
        raise TypeError("oracle_world must be an OracleWorld")
    if source.oracle_world.base_state_id != source.base_state.state_id:
        raise ValueError("oracle_world and base_state IDs must match")
    if not isinstance(source.scene_dynamic_history, Mapping) or not isinstance(
        source.scene_dynamic_specs, Mapping
    ):
        raise TypeError("scene history and specs must be mappings")
    history_ids = set(source.scene_dynamic_history)
    spec_ids = set(source.scene_dynamic_specs)
    world_ids = set(source.oracle_world.dynamic_object_trajectories)
    if history_ids != spec_ids or not world_ids.issubset(history_ids):
        raise ValueError(
            "scene history/spec IDs must contain oracle_world object IDs"
        )
    if set(source.oracle_world.dynamic_object_specs) != world_ids:
        raise ValueError("oracle_world trajectory/spec IDs must match")
    for object_id in sorted(world_ids):
        if source.scene_dynamic_specs[object_id] != source.oracle_world.dynamic_object_specs[
            object_id
        ]:
            raise ValueError("scene and oracle_world footprint specs must match")
    if not isinstance(source.hidden_object_ids, tuple):
        raise TypeError("hidden_object_ids must be an explicit tuple")
    if not set(source.hidden_object_ids).issubset(world_ids):
        raise ValueError(
            "hidden_object_ids must have history, specs, and oracle trajectories"
        )


def _validate_declared_hidden_visibility(
    source: RiskBuildInput,
    *,
    rendered_history: np.ndarray,
    grid: GridSpec,
) -> None:
    current_visible = rendered_history[
        -1, HISTORY_CHANNELS.index("past_visible_mask")
    ] > 0.5
    for object_id in sorted(source.hidden_object_ids):
        footprint = footprint_from_spec(source.scene_dynamic_specs[object_id])
        current_pose = source.scene_dynamic_history[object_id][-1]
        footprint_mask = rasterize_footprint(footprint, current_pose, grid)
        if not bool(np.any(footprint_mask)):
            raise ValueError(f"hidden object {object_id!r} has no current grid footprint")
        if bool(np.any(footprint_mask & current_visible)):
            raise ValueError(f"hidden object {object_id!r} is currently visible")


def validate_risk_sample_for_publication(
    sample: RiskSample,
    grid: GridSpec,
) -> None:
    """Validate model arrays, finite labels, and recursive metadata isolation."""

    if not isinstance(sample, RiskSample):
        raise TypeError("sample must be a RiskSample")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    validate_risk_sample(sample, grid)
    assert_no_oracle_leakage(RiskSample)
    for name in ("sample_id", "split", "base_state_id", "pair_group_id", "event_type"):
        _require_nonempty_string(getattr(sample, name), name=name)
    if isinstance(sample.collision_label, (bool, np.bool_)) or not isinstance(
        sample.collision_label, (int, np.integer)
    ):
        raise TypeError("collision_label must be an integer")
    if isinstance(sample.near_miss, (bool, np.bool_)) or not isinstance(
        sample.near_miss, (int, np.integer)
    ):
        raise TypeError("near_miss must be an integer")
    severity = _finite_float(sample.risk_severity, name="risk_severity")
    minimum = _finite_float(sample.min_clearance, name="min_clearance")
    first_collision = sample.first_collision_time
    if first_collision is not None:
        first_collision = _finite_float(
            first_collision, name="first_collision_time"
        )
        if first_collision <= 0.0:
            raise ValueError("first_collision_time must be positive")
    if sample.collision_label == 1:
        if first_collision is None:
            raise ValueError("collision requires first_collision_time")
        if severity != 1.0:
            raise ValueError("collision requires risk_severity == 1")
        if minimum > 0.0:
            raise ValueError("collision requires min_clearance <= 0")
    elif first_collision is not None:
        raise ValueError("noncollision requires first_collision_time=None")
    elif minimum <= 0.0:
        raise ValueError("noncollision requires positive min_clearance")

    if not isinstance(sample.metadata, dict):
        raise TypeError("metadata must be a dict")
    if set(sample.metadata) != _METADATA_KEYS:
        raise ValueError(f"metadata keys must be exactly {sorted(_METADATA_KEYS)}")
    _validate_metadata_value(sample.metadata, path="metadata")
    if sample.metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"metadata schema_version must be {SCHEMA_VERSION}")
    if sample.metadata["trajectory_id"] == "" or not isinstance(
        sample.metadata["trajectory_id"], str
    ):
        raise ValueError("metadata trajectory_id must be a non-empty string")

    renderer = sample.metadata["renderer"]
    if not isinstance(renderer, dict) or set(renderer) != _RENDERER_METADATA_KEYS:
        raise ValueError("renderer metadata keys violate the frozen contract")
    if renderer["renderer_layout_version"] != RENDERER_LAYOUT_VERSION:
        raise ValueError("renderer layout version mismatch")
    if renderer["base_state_id"] != sample.base_state_id:
        raise ValueError("renderer base_state_id mismatch")
    if not isinstance(sample.metadata["provenance"], dict):
        raise TypeError("provenance metadata must be a dict")

    audit = sample.metadata["label_audit"]
    if not isinstance(audit, dict) or set(audit) != _LABEL_AUDIT_KEYS:
        raise ValueError("label_audit keys violate the frozen contract")
    if audit["risk_gt_version"] != RISK_GT_VERSION:
        raise ValueError("risk_gt_version mismatch")
    if audit["pose_time_layout_version"] != POSE_TIME_LAYOUT_VERSION:
        raise ValueError("pose_time_layout_version mismatch")
    has_hidden = audit["has_hidden_target"]
    if not isinstance(has_hidden, bool):
        raise TypeError("has_hidden_target must be bool")
    critical_id = audit["critical_object_id"]
    critical_type = audit["critical_object_type"]
    time_to_minimum = audit["time_to_min_clearance_s"]
    if has_hidden:
        _require_nonempty_string(critical_id, name="critical_object_id")
        if critical_type not in DYNAMIC_OBJECT_TYPES:
            raise ValueError("critical_object_type is invalid")
        time_to_minimum = _finite_float(
            time_to_minimum, name="time_to_min_clearance_s"
        )
        if time_to_minimum <= 0.0:
            raise ValueError("time_to_min_clearance_s must be positive")
    else:
        if any(value is not None for value in (critical_id, critical_type, time_to_minimum)):
            raise ValueError("empty hidden set requires empty label_audit identity")
        if sample.collision_label != 0 or sample.near_miss != 0 or severity != 0.0:
            raise ValueError("empty hidden set requires zero risk labels")
        sentinel = resolve_no_object_clearance_sentinel(grid)
        if minimum != sentinel:
            raise ValueError("empty hidden set requires the grid-diagonal sentinel")


def _robot_footprint(base_config: Mapping[str, object]) -> Footprint:
    robot_config = base_config.get("robot")
    if not isinstance(robot_config, Mapping):
        raise TypeError("base_config.robot must be a mapping")
    return inflate_footprint(
        RectangleFootprint(
            robot_config.get("length_m"),
            robot_config.get("width_m"),
        ),
        robot_config.get("inflation_m"),
    )


def _assert_target_reconstruction(
    actual: TransplantedDynamicObject,
    expected: TransplantedDynamicObject,
    *,
    kind: str,
) -> None:
    scalar_fields = (
        "target_dynamic_object_id",
        "source_object_id",
        "snippet_id",
        "object_type",
        "footprint_spec",
        "footprint_spec_digest",
        "provenance",
    )
    if any(getattr(actual, name) != getattr(expected, name) for name in scalar_fields):
        raise ValueError(f"{kind} target does not reconstruct from the real snippet")
    for name in ("history_poses", "current_pose", "future_poses"):
        if not np.array_equal(getattr(actual, name), getattr(expected, name)):
            raise ValueError(f"{kind} target does not reconstruct from the real snippet")


def _reconstruct_spatial_target(
    variant: PairedVariant,
    *,
    mother_event: GeneratedEvent,
) -> TransplantedDynamicObject:
    metadata = variant.world.metadata.get("paired_transform")
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "kind",
        "radial_shift_m",
        "signed_arc_offset_m",
        "rotation_rad",
    }:
        raise ValueError(
            f"{variant.variant_kind} paired_transform metadata is invalid"
        )
    if metadata.get("kind") != "hidden_pose_pivot_v1":
        raise ValueError(
            f"{variant.variant_kind} paired_transform kind is invalid"
        )
    radial = _finite_float(
        metadata.get("radial_shift_m"),
        name=f"{variant.variant_kind} paired_transform radial_shift_m",
    )
    signed_arc = _finite_float(
        metadata.get("signed_arc_offset_m"),
        name=f"{variant.variant_kind} paired_transform signed_arc_offset_m",
    )
    declared_angle = _finite_float(
        metadata.get("rotation_rad"),
        name=f"{variant.variant_kind} paired_transform rotation_rad",
    )

    mother_target = mother_event.target
    future_path = np.vstack(
        (mother_target.current_pose, mother_target.future_poses)
    ).astype(np.float64)
    pivot = future_path[0, :2].copy()
    conflict_index = min(
        mother_event.conflict_index + 1, future_path.shape[0] - 1
    )
    pivot_radius = float(
        np.linalg.norm(future_path[conflict_index, :2] - pivot)
    )
    if pivot_radius <= 1e-6:
        pivot_radius = float(
            np.max(np.linalg.norm(future_path[:, :2] - pivot, axis=1))
        )
    if pivot_radius <= 1e-6:
        raise ValueError("spatial paired target pivot is degenerate")
    sensor_distance = float(np.linalg.norm(pivot))
    if sensor_distance <= 1e-6:
        raise ValueError("spatial paired target blind ray is degenerate")
    ray_direction = pivot / sensor_distance
    angle = signed_arc / pivot_radius
    if not math.isclose(declared_angle, angle, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{variant.variant_kind} paired transform rotation is inconsistent"
        )
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    poses = np.vstack(
        (mother_target.history_poses, mother_target.future_poses)
    ).astype(np.float64)
    poses[:, :2] = (
        (poses[:, :2] - pivot) @ rotation.T
        + pivot
        + radial * ray_direction
    )
    poses[:, 2] = wrap_angle(poses[:, 2] + angle)
    poses = poses.astype(np.float32)
    transform = {
        "kind": "hidden_pose_pivot_v1",
        "radial_shift_m": radial,
        "signed_arc_offset_m": signed_arc,
        "rotation_rad": angle,
    }
    return replace(
        mother_target,
        history_poses=poses[: mother_target.history_poses.shape[0]],
        current_pose=poses[mother_target.history_poses.shape[0] - 1].copy(),
        future_poses=poses[mother_target.history_poses.shape[0] :],
        provenance={**mother_target.provenance, "paired_transform": transform},
    )


def _reconstruct_temporal_target(
    variant: PairedVariant,
    *,
    mother_event: GeneratedEvent,
    source_snippet: MotionSnippet,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    future_dt_s: float,
) -> TransplantedDynamicObject:
    metadata = variant.world.metadata.get("paired_transform")
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "kind",
        "temporal_offset_s",
        "mother_conflict_time_s",
        "variant_conflict_time_s",
    }:
        raise ValueError("temporal_safe paired_transform metadata is invalid")
    if metadata.get("kind") != "temporal_offset_v1":
        raise ValueError("temporal_safe paired_transform kind is invalid")
    offset = _finite_float(
        metadata.get("temporal_offset_s"),
        name="temporal_safe paired_transform temporal_offset_s",
    )
    mother_time = _finite_float(
        metadata.get("mother_conflict_time_s"),
        name="temporal_safe paired_transform mother_conflict_time_s",
    )
    variant_time = _finite_float(
        metadata.get("variant_conflict_time_s"),
        name="temporal_safe paired_transform variant_conflict_time_s",
    )
    if not math.isclose(
        mother_time,
        mother_event.conflict_time_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        variant_time,
        mother_event.conflict_time_s + offset,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("temporal_safe conflict-time transform is inconsistent")
    provenance = mother_event.target.provenance
    crossing_direction = provenance.get(
        "desired_crossing_direction", provenance.get("crossing_direction")
    )
    try:
        expected = transplant_snippet(
            source_snippet,
            conflict_point=provenance["conflict_point"],
            conflict_time_s=variant_time,
            crossing_direction=crossing_direction,
            time_scale=provenance["time_scale"],
            future_dt_s=future_dt_s,
            future_steps=trajectory.poses.shape[0],
            base_state_id=mother_event.world.base_state_id,
            trajectory_id=trajectory.trajectory_id,
            target_type_policy_digest=provenance["target_type_policy_digest"],
            seed=provenance["seed"],
            context_object_ids=tuple(oracle_context.dynamic_object_future),
        )
    except (KeyError, TypeError, ValueError, TransplantError) as exc:
        raise ValueError(
            "temporal_safe target does not reconstruct from the real snippet"
        ) from exc
    return replace(
        expected,
        target_dynamic_object_id=mother_event.target.target_dynamic_object_id,
        provenance={
            **expected.provenance,
            "paired_transform": {
                "kind": "temporal_offset_v1",
                "temporal_offset_s": offset,
                "mother_conflict_time_s": mother_event.conflict_time_s,
            },
        },
    )


def _validate_variant_semantics(
    variant: PairedVariant,
    *,
    source: RiskBuildInput,
    mother_event: GeneratedEvent,
    source_snippet: MotionSnippet,
    oracle_context: OracleContext,
    labels: RiskGroundTruth,
    robot_footprint: Footprint,
    future_dt_s: float,
    paired_config: PairedVariantConfig,
) -> None:
    kind = variant.variant_kind
    transform_values = (
        variant.temporal_offset_s,
        variant.lateral_offset_m,
        variant.radial_shift_m,
        variant.rotation_rad,
    )
    if variant.target is None:
        if kind != "empty_blind_spot":
            raise ValueError("only empty_blind_spot may omit its target")
        if any(value is not None for value in transform_values):
            raise ValueError("empty_blind_spot transform fields must be empty")
        if any(
            value is not None
            for value in (
                variant.clearance_sequence_m,
                variant.min_clearance_m,
                variant.time_to_min_clearance_s,
            )
        ):
            raise ValueError("empty_blind_spot clearance fields must be empty")
        if labels.has_hidden_target or labels.collision_label or labels.near_miss:
            raise ValueError("empty_blind_spot labels must be empty-safe")
        if variant.world.metadata.get("min_clearance_m") is not None or (
            variant.world.metadata.get("time_to_min_clearance_s") is not None
        ):
            raise ValueError("empty_blind_spot clearance metadata must be empty")
        transform = variant.world.metadata.get("paired_transform")
        if transform != {
            "kind": "target_removal",
            "removed_target_dynamic_object_id": (
                mother_event.target.target_dynamic_object_id
            ),
        }:
            raise ValueError("empty_blind_spot target-removal transform mismatch")
        return

    mother_provenance = mother_event.target.provenance
    for field, label in (
        ("source_recording_id", "source recording"),
        ("source_session_id", "source session"),
    ):
        variant_value = variant.target.provenance.get(field)
        mother_value = mother_provenance.get(field)
        if (
            not isinstance(variant_value, str)
            or not variant_value.strip()
            or not isinstance(mother_value, str)
            or not mother_value.strip()
        ):
            raise ValueError(f"{kind} target {label} must be non-empty")
        if variant_value != mother_value:
            raise ValueError(f"{kind} target {label} differs from mother target")

    if kind == "collision":
        if variant.world.metadata.get("paired_transform") != {
            "kind": "collision_mother"
        }:
            raise ValueError("collision mother transform metadata mismatch")
        _assert_target_reconstruction(
            variant.target, mother_event.target, kind=kind
        )
    elif kind == "temporal_safe":
        _assert_target_reconstruction(
            variant.target,
            _reconstruct_temporal_target(
                variant,
                mother_event=mother_event,
                source_snippet=source_snippet,
                trajectory=source.trajectory,
                oracle_context=oracle_context,
                future_dt_s=future_dt_s,
            ),
            kind=kind,
        )
    elif kind in {"near_miss", "spatial_safe", "irrelevant_hidden"}:
        _assert_target_reconstruction(
            variant.target,
            _reconstruct_spatial_target(variant, mother_event=mother_event),
            kind=kind,
        )

    target_id = variant.target.target_dynamic_object_id
    target_footprint = footprint_from_spec(variant.target.footprint_spec)
    actual = trajectory_signed_clearances(
        robot_footprint,
        source.trajectory.poses,
        target_footprint,
        source.oracle_world.dynamic_object_trajectories[target_id],
    )
    declared = variant.clearance_sequence_m
    if (
        not isinstance(declared, np.ndarray)
        or declared.shape != actual.shape
        or not np.isfinite(declared).all()
        or not np.allclose(declared, actual, rtol=0.0, atol=1e-6)
    ):
        raise ValueError(f"{kind} clearance sequence differs from actual geometry")
    minimum_index = int(np.argmin(actual))
    minimum = float(actual[minimum_index])
    time_to_minimum = float((minimum_index + 1) * future_dt_s)
    if not math.isclose(
        _finite_float(variant.min_clearance_m, name=f"{kind} min_clearance_m"),
        minimum,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{kind} min_clearance_m differs from actual geometry")
    if not math.isclose(
        _finite_float(
            variant.time_to_min_clearance_s,
            name=f"{kind} time_to_min_clearance_s",
        ),
        time_to_minimum,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{kind} time_to_min_clearance_s differs from actual geometry")
    metadata = variant.world.metadata
    if metadata.get("min_clearance_m") != variant.min_clearance_m or metadata.get(
        "time_to_min_clearance_s"
    ) != variant.time_to_min_clearance_s:
        raise ValueError(f"{kind} clearance metadata mismatch")
    if not math.isclose(labels.min_clearance, minimum, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{kind} risk label differs from actual geometry")

    if kind == "collision":
        if labels.collision_label != 1:
            raise ValueError("collision variant must collide")
        if any(value is not None for value in transform_values):
            raise ValueError("collision variant must not carry a paired transform")
    elif kind == "near_miss":
        lower, upper = paired_config.near_miss_clearance_range_m
        if labels.collision_label or labels.near_miss != 1 or not lower <= minimum <= upper:
            raise ValueError("near_miss variant violates its clearance semantics")
        lateral = _finite_float(
            variant.lateral_offset_m, name="near_miss lateral_offset_m"
        )
        _finite_float(variant.radial_shift_m, name="near_miss radial_shift_m")
        _finite_float(variant.rotation_rad, name="near_miss rotation_rad")
        if lateral <= 0.0 or variant.temporal_offset_s is not None:
            raise ValueError("near_miss must carry a nonzero spatial transform")
    elif kind == "temporal_safe":
        offset = _finite_float(
            variant.temporal_offset_s, name="temporal_safe temporal_offset_s"
        )
        if not any(
            math.isclose(offset, candidate, rel_tol=0.0, abs_tol=1e-9)
            for candidate in paired_config.temporal_offset_candidates_s
        ):
            raise ValueError("temporal_safe offset is absent from paired_config")
        if labels.collision_label:
            raise ValueError("temporal_safe variant must not collide")
        if any(value is not None for value in transform_values[1:]):
            raise ValueError("temporal_safe must carry only a temporal transform")
        paths_intersect = any(
            signed_clearance(
                robot_footprint,
                robot_pose,
                target_footprint,
                target_pose,
            )
            <= 0.0
            for robot_pose in source.trajectory.poses
            for target_pose in variant.target.future_poses
        )
        if not paths_intersect:
            raise ValueError("temporal_safe spatial paths must intersect")
    elif kind in {"spatial_safe", "irrelevant_hidden"}:
        lateral = _finite_float(
            variant.lateral_offset_m, name=f"{kind} lateral_offset_m"
        )
        _finite_float(variant.radial_shift_m, name=f"{kind} radial_shift_m")
        _finite_float(variant.rotation_rad, name=f"{kind} rotation_rad")
        if lateral <= 0.0 or variant.temporal_offset_s is not None:
            raise ValueError(f"{kind} must carry a nonzero spatial transform")
        if labels.collision_label:
            raise ValueError(f"{kind} variant must not collide")
        if kind == "spatial_safe":
            lower, upper = paired_config.spatial_safe_clearance_range_m
            if not lower <= minimum <= upper:
                raise ValueError("spatial_safe clearance is outside its configured range")
        elif minimum < paired_config.irrelevant_min_clearance_m:
            raise ValueError("irrelevant_hidden clearance is below its configured minimum")
    else:  # pragma: no cover - formal SOP06 renderer rejects unknown kinds first
        raise ValueError(f"unsupported formal paired variant: {kind}")

    transform_metadata = metadata.get("paired_transform")
    if not isinstance(transform_metadata, Mapping):
        raise ValueError(f"{kind} paired_transform metadata must be a mapping")
    if kind == "temporal_safe":
        if not math.isclose(
            _finite_float(
                transform_metadata.get("temporal_offset_s"),
                name="temporal_safe paired_transform temporal_offset_s",
            ),
            float(variant.temporal_offset_s),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("temporal_safe transform metadata mismatch")
    elif kind in {"near_miss", "spatial_safe", "irrelevant_hidden"}:
        signed_arc = _finite_float(
            transform_metadata.get("signed_arc_offset_m"),
            name=f"{kind} paired_transform signed_arc_offset_m",
        )
        radial = _finite_float(
            transform_metadata.get("radial_shift_m"),
            name=f"{kind} paired_transform radial_shift_m",
        )
        rotation = _finite_float(
            transform_metadata.get("rotation_rad"),
            name=f"{kind} paired_transform rotation_rad",
        )
        if not (
            math.isclose(
                abs(signed_arc),
                float(variant.lateral_offset_m),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                radial,
                float(variant.radial_shift_m),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                rotation,
                float(variant.rotation_rad),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(f"{kind} transform metadata mismatch")


def _build_risk_sample_and_optional_sidecar_from_rendered(
    source: RiskBuildInput,
    rendered: RenderedObservation,
    *,
    base_config: Mapping[str, object],
    normalized_risk: Mapping[str, object],
    formal_variant: PairedVariant | None = None,
    paired_config: PairedVariantConfig | None = None,
    mother_event: GeneratedEvent | None = None,
    source_snippet: MotionSnippet | None = None,
    oracle_context: OracleContext | None = None,
    include_sidecar: bool = False,
) -> tuple[RiskSample, RiskLabelSidecar | None, RiskGroundTruth]:
    grid = build_grid_spec(dict(base_config))
    if not np.array_equal(
        source.observed_static_occupancy,
        source.oracle_world.static_occupancy,
    ):
        raise ValueError("observed and oracle_world static occupancy must match")
    _validate_declared_hidden_visibility(
        source,
        rendered_history=rendered.bev_history,
        grid=grid,
    )

    robot_footprint = _robot_footprint(base_config)
    bev_config = base_config.get("bev")
    if not isinstance(bev_config, Mapping):
        raise TypeError("base_config.bev must be a mapping")
    labels = compute_hidden_risk_gt(
        source.trajectory,
        source.oracle_world,
        hidden_object_ids=source.hidden_object_ids,
        robot_footprint=robot_footprint,
        grid=grid,
        future_dt_s=bev_config.get("future_dt_s"),
        sigma_distance_m=normalized_risk["sigma_distance_m"],
        sigma_time_s=normalized_risk["sigma_time_s"],
        near_miss_distance_m=normalized_risk["near_miss_distance_m"],
    )
    sidecar = (
        build_risk_label_sidecar(
            sample_id=source.sample_id,
            trajectory=source.trajectory,
            world=source.oracle_world,
            hidden_object_ids=source.hidden_object_ids,
            robot_footprint=robot_footprint,
            grid=grid,
            future_dt_s=bev_config.get("future_dt_s"),
        )
        if include_sidecar
        else None
    )
    if formal_variant is not None:
        if any(
            value is None
            for value in (
                paired_config,
                mother_event,
                source_snippet,
                oracle_context,
            )
        ):  # pragma: no cover - private call contract
            raise RuntimeError("formal variant validation requires formal context")
        _validate_variant_semantics(
            formal_variant,
            source=source,
            mother_event=mother_event,
            source_snippet=source_snippet,
            oracle_context=oracle_context,
            labels=labels,
            robot_footprint=robot_footprint,
            future_dt_s=float(bev_config.get("future_dt_s")),
            paired_config=paired_config,
        )
    trajectory_channels = build_trajectory_channels(source.trajectory, grid)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "renderer": _canonical_metadata_copy(
            rendered.metadata, name="renderer"
        ),
        "trajectory_id": _require_nonempty_string(
            source.trajectory.trajectory_id, name="trajectory_id"
        ),
        "provenance": _canonical_metadata_copy(
            {
                key: value
                for key, value in source.provenance.items()
                if key not in _EVALUATION_ONLY_SOURCE_PROVENANCE_KEYS
            },
            name="provenance",
        ),
        "label_audit": {
            "risk_gt_version": RISK_GT_VERSION,
            "pose_time_layout_version": labels.pose_time_layout_version,
            "critical_object_id": labels.critical_object_id,
            "critical_object_type": labels.critical_object_type,
            "time_to_min_clearance_s": labels.time_to_min_clearance,
            "has_hidden_target": labels.has_hidden_target,
        },
    }
    sample = RiskSample(
        sample_id=source.sample_id,
        split=source.base_state.split,
        base_state_id=source.base_state.state_id,
        pair_group_id=source.pair_group_id,
        event_type=source.event_type,
        bev_history=np.array(
            rendered.bev_history, dtype=np.float32, order="C", copy=True
        ),
        state_channels=np.array(
            rendered.state_channels, dtype=np.float32, order="C", copy=True
        ),
        trajectory_channels=trajectory_channels,
        robot_state=np.array(
            source.base_state.robot_state,
            dtype=np.float32,
            order="C",
            copy=True,
        ),
        collision_label=labels.collision_label,
        risk_severity=labels.risk_severity,
        min_clearance=labels.min_clearance,
        near_miss=labels.near_miss,
        first_collision_time=labels.first_collision_time,
        metadata=metadata,
    )
    validate_risk_sample_for_publication(sample, grid)
    return sample, sidecar, labels


def build_risk_sample(
    source: RiskBuildInput,
    *,
    base_config: Mapping[str, object],
    risk_config: Mapping[str, object],
) -> RiskSample:
    """Render a general history-only source and compute its hidden-risk labels."""

    if not isinstance(source, RiskBuildInput):
        raise TypeError("source must be a RiskBuildInput")
    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    base_config_dict = dict(base_config)
    if base_config_dict.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"base_config schema_version must be {SCHEMA_VERSION}")
    normalized_risk = _normalized_risk_config(risk_config)
    _validate_source_join(source)
    rendered = render_observation(
        source.base_state,
        scene_dynamic_history=source.scene_dynamic_history,
        scene_dynamic_specs=source.scene_dynamic_specs,
        static_occupancy=source.observed_static_occupancy,
        sensor_config=source.sensor_config,
        config=base_config_dict,
    )
    sample, _, _ = _build_risk_sample_and_optional_sidecar_from_rendered(
        source,
        rendered,
        base_config=base_config_dict,
        normalized_risk=normalized_risk,
    )
    return sample


def _build_risk_samples_from_sop06_group_impl(
    *,
    group: PairedEventGroup,
    mother_event: GeneratedEvent,
    source_snippet: MotionSnippet,
    base_state: BaseState,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    base_config: Mapping[str, object],
    paired_config: PairedVariantConfig,
    risk_config: Mapping[str, object],
    dataset_seed: int,
    include_sidecars: bool,
    include_evaluation_records: bool = False,
) -> (
    tuple[tuple[RiskSample, ...], tuple[RiskLabelSidecar, ...]]
    | tuple[
        tuple[RiskSample, ...],
        tuple[RiskLabelSidecar, ...],
        tuple[dict[str, object], ...],
    ]
):
    """Atomically assemble one formal group with optional oracle sidecars."""

    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    for value, expected_type, name in (
        (group, PairedEventGroup, "group"),
        (mother_event, GeneratedEvent, "mother_event"),
        (source_snippet, MotionSnippet, "source_snippet"),
        (base_state, BaseState, "base_state"),
        (trajectory, LocalTrajectory, "trajectory"),
        (oracle_context, OracleContext, "oracle_context"),
    ):
        if not isinstance(value, expected_type):
            raise TypeError(f"{name} must be a {expected_type.__name__}")
    paired_config = _canonical_paired_config(paired_config)

    # Own one immutable-in-practice snapshot before the formal renderer sees
    # anything.  External array/dict mutation can therefore affect neither the
    # rendered observation nor the later oracle-label branch.
    (
        group,
        mother_event,
        source_snippet,
        base_state,
        trajectory,
        oracle_context,
        base_config_dict,
        risk_config_snapshot,
    ) = _deep_owned_copy(
        (
            group,
            mother_event,
            source_snippet,
            base_state,
            trajectory,
            oracle_context,
            dict(base_config),
            risk_config,
        )
    )
    if base_config_dict.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"base_config schema_version must be {SCHEMA_VERSION}")
    normalized_risk = _normalized_risk_config(risk_config_snapshot)
    if include_evaluation_records and not include_sidecars:
        raise ValueError("evaluation records require the formal sidecar boundary")

    base_session_id, transplant_seed, paired_seed = _validate_formal_inputs(
        group=group,
        mother_event=mother_event,
        source_snippet=source_snippet,
        base_state=base_state,
        trajectory=trajectory,
        oracle_context=oracle_context,
        paired_config=paired_config,
        dataset_seed=dataset_seed,
    )

    # This is deliberately the sole formal SOP06 validation/render boundary.
    # Its returned observations are consumed below without a per-variant rerender.
    rendered_group = render_sop06_partial_pair_group(
        group=group,
        mother_record=mother_event.target_motion_record,
        mother_world=mother_event.world,
        base_state=base_state,
        oracle_context=oracle_context,
        config=base_config_dict,
        expected_paired_config_digest=paired_config.digest,
    )
    if rendered_group.variant_kinds != tuple(
        variant.variant_kind for variant in group.variants
    ) or len(rendered_group.observations) != len(group.variants):
        raise RuntimeError("formal SOP06 renderer returned misaligned observations")

    samples: list[RiskSample] = []
    sidecars: list[RiskLabelSidecar] = []
    evaluation_records: list[dict[str, object]] = []
    robot_footprint: Footprint | None = None
    robot_footprint_provenance: Mapping[str, object] | None = None
    age_max_s: object = None
    if include_evaluation_records:
        robot_footprint = _robot_footprint(base_config_dict)
        robot_footprint_provenance = derive_robot_footprint_provenance(
            base_config=base_config_dict,
            effective_footprint=robot_footprint,
        )
        age_config = base_config_dict.get("age_map")
        if not isinstance(age_config, Mapping):
            raise TypeError("base_config.age_map must be a mapping")
        age_max_s = age_config.get("a_max_s")
    for variant, rendered in zip(
        group.variants, rendered_group.observations, strict=True
    ):
        source = _build_formal_source(
            variant=variant,
            mother_event=mother_event,
            source_snippet=source_snippet,
            base_state=base_state,
            trajectory=trajectory,
            oracle_context=oracle_context,
            base_config=base_config_dict,
            paired_config=paired_config,
            risk_config=normalized_risk,
            dataset_seed=int(dataset_seed),
            base_session_id=base_session_id,
            transplant_seed=transplant_seed,
            paired_seed=paired_seed,
        )
        _validate_source_join(source)
        sample, sidecar, ground_truth = (
            _build_risk_sample_and_optional_sidecar_from_rendered(
                source,
                rendered,
                base_config=base_config_dict,
                normalized_risk=normalized_risk,
                formal_variant=variant,
                paired_config=paired_config,
                mother_event=mother_event,
                source_snippet=source_snippet,
                oracle_context=oracle_context,
                include_sidecar=include_sidecars,
            )
        )
        samples.append(sample)
        if include_sidecars:
            if sidecar is None:  # pragma: no cover - private call invariant
                raise RuntimeError(
                    "formal sidecar construction unexpectedly returned None"
                )
            sidecars.append(sidecar)
        if include_evaluation_records:
            if robot_footprint is None or robot_footprint_provenance is None:
                raise RuntimeError(
                    "evaluation footprint context was not initialized"
                )
            raw_ood_tag = source.provenance.get("ood_tag")
            raw_ood_evidence = source.provenance.get("ood_evidence")
            if raw_ood_tag is None and raw_ood_evidence is None:
                ood_tag = "in_distribution"
                ood_evidence: Mapping[str, object] = {
                    "rule_version": OOD_ROUTING_RULE_VERSION,
                    "source": "default_in_distribution",
                    "reason": (
                        "source provenance declares no explicit OOD routing tag"
                    ),
                }
            elif raw_ood_tag is None or raw_ood_evidence is None:
                raise ValueError(
                    "source provenance must declare both ood_tag and ood_evidence"
                )
            else:
                ood_tag = raw_ood_tag
                if not isinstance(raw_ood_evidence, Mapping):
                    raise TypeError("source provenance ood_evidence must be a mapping")
                ood_evidence = raw_ood_evidence
            evaluation_records.append(
                derive_production_evaluation_record(
                    sample=sample,
                    source=source,
                    rendered=rendered,
                    ground_truth=ground_truth,
                    robot_footprint=robot_footprint,
                    age_max_s=age_max_s,
                    pair_eligible=group.eligible_for_strict_evaluation,
                    ood_tag=ood_tag,
                    robot_footprint_provenance=robot_footprint_provenance,
                    ood_evidence=ood_evidence,
                )
            )
    if include_evaluation_records:
        return tuple(samples), tuple(sidecars), tuple(evaluation_records)
    return tuple(samples), tuple(sidecars)


def build_risk_samples_sidecars_and_evaluation_records_from_sop06_group(
    *,
    group: PairedEventGroup,
    mother_event: GeneratedEvent,
    source_snippet: MotionSnippet,
    base_state: BaseState,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    base_config: Mapping[str, object],
    paired_config: PairedVariantConfig,
    risk_config: Mapping[str, object],
    dataset_seed: int,
) -> tuple[
    tuple[RiskSample, ...],
    tuple[RiskLabelSidecar, ...],
    tuple[dict[str, object], ...],
]:
    """Build aligned model samples, label sidecars, and evaluation records."""

    result = _build_risk_samples_from_sop06_group_impl(
        group=group,
        mother_event=mother_event,
        source_snippet=source_snippet,
        base_state=base_state,
        trajectory=trajectory,
        oracle_context=oracle_context,
        base_config=base_config,
        paired_config=paired_config,
        risk_config=risk_config,
        dataset_seed=dataset_seed,
        include_sidecars=True,
        include_evaluation_records=True,
    )
    if len(result) != 3:  # pragma: no cover - private return invariant
        raise RuntimeError("formal evaluation assembly returned no records")
    samples, sidecars, records = result
    return samples, sidecars, records


def build_risk_samples_and_sidecars_from_sop06_group(
    *,
    group: PairedEventGroup,
    mother_event: GeneratedEvent,
    source_snippet: MotionSnippet,
    base_state: BaseState,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    base_config: Mapping[str, object],
    paired_config: PairedVariantConfig,
    risk_config: Mapping[str, object],
    dataset_seed: int,
) -> tuple[tuple[RiskSample, ...], tuple[RiskLabelSidecar, ...]]:
    """Atomically assemble model samples and separate oracle-only sidecars."""

    return _build_risk_samples_from_sop06_group_impl(
        group=group,
        mother_event=mother_event,
        source_snippet=source_snippet,
        base_state=base_state,
        trajectory=trajectory,
        oracle_context=oracle_context,
        base_config=base_config,
        paired_config=paired_config,
        risk_config=risk_config,
        dataset_seed=dataset_seed,
        include_sidecars=True,
    )


def build_risk_samples_from_sop06_group(
    *,
    group: PairedEventGroup,
    mother_event: GeneratedEvent,
    source_snippet: MotionSnippet,
    base_state: BaseState,
    trajectory: LocalTrajectory,
    oracle_context: OracleContext,
    base_config: Mapping[str, object],
    paired_config: PairedVariantConfig,
    risk_config: Mapping[str, object],
    dataset_seed: int,
) -> tuple[RiskSample, ...]:
    """Compatibility wrapper that exposes only deployment-safe samples."""

    samples, _ = _build_risk_samples_from_sop06_group_impl(
        group=group,
        mother_event=mother_event,
        source_snippet=source_snippet,
        base_state=base_state,
        trajectory=trajectory,
        oracle_context=oracle_context,
        base_config=base_config,
        paired_config=paired_config,
        risk_config=risk_config,
        dataset_seed=dataset_seed,
        include_sidecars=False,
    )
    return samples
