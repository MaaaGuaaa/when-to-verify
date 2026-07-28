"""Schema-4 Long40 RiskSample assembly with an explicit input/label boundary."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import json
import math
from numbers import Real
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

import numpy as np

from src.contracts import (
    DYNAMIC_OBJECT_TYPES,
    HISTORY_CHANNELS,
    LONG40_FUTURE_STEPS,
    LONG40_HISTORY_STEPS,
    LONG40_SAMPLE_DT_S,
    POSE_TIME_LAYOUT_VERSION,
    SCHEMA_VERSION,
    TRAJECTORY_CHANNELS,
    BaseState,
    GridSpec,
    LocalTrajectory,
    OracleWorld,
    RiskSample,
    assert_no_oracle_leakage,
    build_grid_spec,
    validate_risk_sample,
)
from src.generation.event_contracts import footprint_from_spec
from src.generation.observation_renderer import (
    RENDERER_LAYOUT_VERSION,
    RenderedObservation,
    render_observation,
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
from src.generation.structural_blindspot import StructuralBlindSpot
from src.geometry import (
    Footprint,
    RectangleFootprint,
    inflate_footprint,
    rasterize_footprint,
)


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
    scene_dynamic_history_observed: Mapping[str, np.ndarray] | None = None


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


def _validated_long40_base_config(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("base_config must be a mapping")
    config = dict(value)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"base_config schema_version must be {SCHEMA_VERSION}")
    bev = config.get("bev")
    if not isinstance(bev, Mapping):
        raise TypeError("base_config.bev must be a mapping")
    expected = {
        "history_steps": LONG40_HISTORY_STEPS,
        "history_dt_s": LONG40_SAMPLE_DT_S,
        "future_steps": LONG40_FUTURE_STEPS,
        "future_dt_s": LONG40_SAMPLE_DT_S,
    }
    for field, required in expected.items():
        if bev.get(field) != required:
            raise ValueError(
                f"base_config.bev.{field} must equal {required}"
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
    if history_ids != spec_ids:
        raise ValueError("scene history/spec IDs must match")
    if set(source.oracle_world.dynamic_object_specs) != world_ids:
        raise ValueError("oracle_world trajectory/spec IDs must match")
    if not isinstance(source.hidden_object_ids, tuple):
        raise TypeError("hidden_object_ids must be an explicit tuple")
    if not set(source.hidden_object_ids).issubset(world_ids):
        raise ValueError("hidden_object_ids must have oracle trajectories")
    for object_id in sorted(world_ids & history_ids):
        if source.scene_dynamic_specs[object_id] != source.oracle_world.dynamic_object_specs[
            object_id
        ]:
            raise ValueError("scene and oracle_world footprint specs must match")
    observed = source.scene_dynamic_history_observed
    if observed is not None:
        if not isinstance(observed, Mapping) or set(observed) != history_ids:
            raise ValueError(
                "scene observed-history IDs must match scene history IDs"
            )
        history_steps = source.base_state.robot_history.shape[0]
        for object_id in sorted(history_ids):
            mask = observed[object_id]
            if (
                not isinstance(mask, np.ndarray)
                or mask.dtype != np.bool_
                or mask.shape != (history_steps,)
            ):
                raise ValueError(
                    "scene observed-history values must be boolean history masks"
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
        if object_id not in source.scene_dynamic_history:
            continue
        observed = source.scene_dynamic_history_observed
        if observed is not None and not bool(observed[object_id][-1]):
            continue
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


def _build_risk_sample_and_optional_sidecar_from_rendered(
    source: RiskBuildInput,
    rendered: RenderedObservation,
    *,
    base_config: Mapping[str, object],
    normalized_risk: Mapping[str, object],
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
    rendered_observation: RenderedObservation | None = None,
) -> RiskSample:
    """Build one sample from a fresh or already-persisted SOP06 observation."""

    if not isinstance(source, RiskBuildInput):
        raise TypeError("source must be a RiskBuildInput")
    base_config_dict = _validated_long40_base_config(base_config)
    normalized_risk = _normalized_risk_config(risk_config)
    _validate_source_join(source)
    if rendered_observation is None:
        rendered = render_observation(
            source.base_state,
            scene_dynamic_history=source.scene_dynamic_history,
            scene_dynamic_specs=source.scene_dynamic_specs,
            static_occupancy=source.observed_static_occupancy,
            sensor_config=source.sensor_config,
            config=base_config_dict,
            scene_dynamic_history_observed=source.scene_dynamic_history_observed,
        )
    elif isinstance(rendered_observation, RenderedObservation):
        rendered = rendered_observation
    else:
        raise TypeError("rendered_observation must be a RenderedObservation or None")
    sample, _, _ = _build_risk_sample_and_optional_sidecar_from_rendered(
        source,
        rendered,
        base_config=base_config_dict,
        normalized_risk=normalized_risk,
    )
    return sample
