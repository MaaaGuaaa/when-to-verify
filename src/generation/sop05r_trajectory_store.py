"""Authenticated event-local planner trajectory storage for SOP05R."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from src.contracts import ARRAY_DTYPE, LocalTrajectory, SCHEMA_VERSION, build_grid_spec
from src.planning.obstacle_corner_planner import (
    ObstaclePlannedRoute,
)
from src.planning.query_maps import build_local_trajectory
from src.planning.trajectory_sampler import CandidateRollout

from .sop05r_contracts import (
    SOP05R_PLANNER_SLOT_IDS,
    SOP05R_PLANNER_VERSION,
    SOP05R_TRAJECTORY_COLLECTION_VERSION,
)


SOP05R_TRAJECTORY_RECORD_VERSION = "sop05r_planner_trajectory_record_v1"
SOP05R_TRAJECTORY_MANIFEST_VERSION = "sop05r_trajectory_store_manifest_v1"
SOP05R_TRAJECTORY_COMPLETION_MARKER_VERSION = (
    "sop05r_trajectory_store_complete_v1"
)
_TRAJECTORY_ARRAY_NAMES = (
    "poses",
    "controls",
    "swept_mask",
    "tta_map",
    "braking_map",
    "centerline_map",
    "poses_world",
    "waypoints_world",
)


def _readonly_array(value: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("trajectory store evidence must be finite canonical JSON") from exc


def _json_file_bytes(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"trajectory store JSON must not contain {value}")


def _read_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="ascii"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read trajectory store JSON: {path.name}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"failed to hash trajectory store file: {path}") from exc


def _hex_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


@dataclass(frozen=True)
class Sop05rTrajectoryRecord:
    event_id: str
    base_state_id: str
    template_id: str
    planner_version: str
    config_digest: str
    shared_goal_world_pose: np.ndarray
    nominal_trajectory_id: str
    alternative_trajectory_ids: tuple[str, ...]
    routes: tuple[ObstaclePlannedRoute, ...]

    def __post_init__(self) -> None:
        _nonempty_text(self.event_id, name="event_id")
        _nonempty_text(self.base_state_id, name="base_state_id")
        _nonempty_text(self.template_id, name="template_id")
        if self.planner_version != SOP05R_PLANNER_VERSION:
            raise ValueError("planner_version mismatch")
        _hex_digest(self.config_digest, name="config_digest")
        goal = np.asarray(self.shared_goal_world_pose)
        if goal.shape != (3,) or goal.dtype != ARRAY_DTYPE or not np.isfinite(goal).all():
            raise ValueError("shared_goal_world_pose must be finite float32 [3]")
        if not isinstance(self.routes, tuple) or not self.routes:
            raise TypeError("routes must be a nonempty tuple")
        if any(not isinstance(route, ObstaclePlannedRoute) for route in self.routes):
            raise TypeError("routes must contain ObstaclePlannedRoute values")
        _nonempty_text(self.nominal_trajectory_id, name="nominal_trajectory_id")
        if not isinstance(self.alternative_trajectory_ids, tuple):
            raise TypeError("alternative_trajectory_ids must be a tuple")
        if any(
            not isinstance(trajectory_id, str) or not trajectory_id
            for trajectory_id in self.alternative_trajectory_ids
        ):
            raise ValueError("alternative_trajectory_ids must contain nonempty text")
        object.__setattr__(
            self,
            "shared_goal_world_pose",
            _readonly_array(goal, dtype=np.dtype(np.float32)),
        )

    @property
    def candidate_trajectory_ids(self) -> tuple[str, ...]:
        return tuple(route.trajectory.trajectory_id for route in self.routes)


@dataclass(frozen=True)
class Sop05rTrajectoryStore:
    records: tuple[Sop05rTrajectoryRecord, ...]
    manifest: Mapping[str, object]
    collection_semantic_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple):
            raise TypeError("records must be a tuple")
        digest = _hex_digest(
            self.collection_semantic_digest,
            name="collection_semantic_digest",
        )
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))
        object.__setattr__(self, "collection_semantic_digest", digest)


def _base_config_snapshot(base_config: Mapping[str, Any]) -> dict[str, object]:
    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    try:
        snapshot = json.loads(
            _canonical_json_bytes(dict(base_config)).decode("ascii"),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:  # pragma: no cover - canonical producer guards
        raise ValueError("base_config snapshot is invalid") from exc
    if not isinstance(snapshot, dict):
        raise ValueError("base_config snapshot must be a mapping")
    grid = build_grid_spec(snapshot)
    if grid.future_steps <= 0:
        raise ValueError("base_config future grid is invalid")
    return snapshot


def _array_semantic_digest(name: str, array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(b"sop05r_trajectory_array_v1\0")
    digest.update(name.encode("ascii") + b"\0")
    digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
    digest.update(_canonical_json_bytes(list(contiguous.shape)))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _route_arrays(route: ObstaclePlannedRoute) -> dict[str, np.ndarray]:
    return {
        "poses": route.trajectory.poses,
        "controls": route.trajectory.controls,
        "swept_mask": route.trajectory.swept_mask,
        "tta_map": route.trajectory.tta_map,
        "braking_map": route.trajectory.braking_map,
        "centerline_map": route.trajectory.centerline_map,
        "poses_world": route.poses_world,
        "waypoints_world": route.waypoints_world,
    }


def validate_sop05r_trajectory_record(
    record: Sop05rTrajectoryRecord,
    *,
    base_config: Mapping[str, Any],
) -> None:
    """Validate identities and rebuild every trajectory query map."""

    if not isinstance(record, Sop05rTrajectoryRecord):
        raise TypeError("record must be a Sop05rTrajectoryRecord")
    config = _base_config_snapshot(base_config)
    grid = build_grid_spec(config)
    slots = tuple(route.slot_id for route in record.routes)
    expected_slots = tuple(slot for slot in SOP05R_PLANNER_SLOT_IDS if slot in slots)
    if slots != expected_slots or slots[-1] != "stop":
        raise ValueError("candidate routes must preserve slot order and include stop")
    candidate_ids = record.candidate_trajectory_ids
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate trajectory IDs must be unique")
    if record.nominal_trajectory_id not in candidate_ids:
        raise ValueError("nominal trajectory is not a candidate")
    routes_by_id = {
        route.trajectory.trajectory_id: route for route in record.routes
    }
    if routes_by_id[record.nominal_trajectory_id].slot_id == "stop":
        raise ValueError("nominal trajectory must be a moving candidate")
    alternatives = record.alternative_trajectory_ids
    if (
        not alternatives
        or len(alternatives) != len(set(alternatives))
        or record.nominal_trajectory_id in alternatives
        or any(trajectory_id not in routes_by_id for trajectory_id in alternatives)
        or any(routes_by_id[trajectory_id].slot_id == "stop" for trajectory_id in alternatives)
    ):
        raise ValueError("alternative trajectories violate candidate membership")
    for route in record.routes:
        trajectory = route.trajectory
        if trajectory.metadata.get("planner_version") != SOP05R_PLANNER_VERSION:
            raise ValueError("trajectory planner version metadata mismatch")
        if trajectory.metadata.get("planner_slot_id") != route.slot_id:
            raise ValueError("trajectory planner slot metadata mismatch")
        if trajectory.metadata.get("shared_goal_world_pose") != [
            float(value) for value in record.shared_goal_world_pose
        ]:
            raise ValueError("trajectory shared goal metadata mismatch")
        if not np.array_equal(route.waypoints_world[-1], record.shared_goal_world_pose):
            raise ValueError("trajectory waypoints do not bind the shared goal")
        expected_shapes = {
            "poses": (grid.future_steps, 3),
            "controls": (grid.future_steps, 2),
            "swept_mask": (grid.height, grid.width),
            "tta_map": (grid.height, grid.width),
            "braking_map": (grid.height, grid.width),
            "centerline_map": (grid.height, grid.width),
            "poses_world": (grid.future_steps, 3),
        }
        arrays = _route_arrays(route)
        for name, expected_shape in expected_shapes.items():
            array = arrays[name]
            if (
                not isinstance(array, np.ndarray)
                or array.shape != expected_shape
                or array.dtype != ARRAY_DTYPE
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"trajectory {name} violates the float32 array contract")
        if (
            route.waypoints_world.ndim != 2
            or route.waypoints_world.shape[1] != 3
            or route.waypoints_world.dtype != ARRAY_DTYPE
            or not np.isfinite(route.waypoints_world).all()
        ):
            raise ValueError("trajectory waypoints_world violates its array contract")
        candidate = CandidateRollout(
            trajectory_id=trajectory.trajectory_id,
            poses=np.array(trajectory.poses, dtype=ARRAY_DTYPE, order="C", copy=True),
            controls=np.array(
                trajectory.controls, dtype=ARRAY_DTYPE, order="C", copy=True
            ),
            is_stop=bool(trajectory.metadata.get("is_stop")),
            is_reverse=bool(trajectory.metadata.get("is_reverse")),
        )
        deceleration = trajectory.metadata.get("braking_deceleration_mps2")
        rebuilt = build_local_trajectory(
            candidate,
            config,
            braking_deceleration_mps2=float(deceleration),
            task_cost=trajectory.task_cost,
        )
        for name in ("swept_mask", "tta_map", "braking_map", "centerline_map"):
            if not np.array_equal(getattr(trajectory, name), getattr(rebuilt, name)):
                raise ValueError(f"trajectory query map mismatch: {name}")
        if not np.isfinite(
            [
                trajectory.task_cost,
                route.path_length_m,
                route.represented_obstacle_clearance_m,
                route.task_score,
            ]
        ).all():
            raise ValueError("trajectory scalar metadata must be finite")
        if float(trajectory.task_cost) != float(route.task_score):
            raise ValueError("trajectory task score mismatch")


def _array_key(route_index: int, name: str) -> str:
    return f"route_{route_index:02d}__{name}"


def _record_row(
    record: Sop05rTrajectoryRecord,
    *,
    array_file: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    route_rows = []
    for route_index, route in enumerate(record.routes):
        digests = {}
        for name, array in _route_arrays(route).items():
            key = _array_key(route_index, name)
            arrays[key] = np.asarray(array)
            digests[name] = _array_semantic_digest(key, arrays[key])
        route_rows.append(
            {
                "slot_id": route.slot_id,
                "trajectory_id": route.trajectory.trajectory_id,
                "trajectory_metadata": route.trajectory.metadata,
                "task_cost": float(route.trajectory.task_cost),
                "path_length_m": route.path_length_m,
                "represented_obstacle_clearance_m": (
                    route.represented_obstacle_clearance_m
                ),
                "task_score": route.task_score,
                "array_semantic_digests": digests,
            }
        )
    row_without_digest = {
        "record_version": SOP05R_TRAJECTORY_RECORD_VERSION,
        "event_id": record.event_id,
        "base_state_id": record.base_state_id,
        "template_id": record.template_id,
        "planner_version": record.planner_version,
        "config_digest": record.config_digest,
        "shared_goal_world_pose": [
            float(value) for value in record.shared_goal_world_pose
        ],
        "candidate_trajectory_ids": list(record.candidate_trajectory_ids),
        "nominal_trajectory_id": record.nominal_trajectory_id,
        "alternative_trajectory_ids": list(record.alternative_trajectory_ids),
        "array_file": array_file,
        "routes": route_rows,
    }
    return (
        {
            **row_without_digest,
            "record_semantic_digest": _sha256_bytes(
                b"sop05r_trajectory_record_v1\0"
                + _canonical_json_bytes(row_without_digest)
            ),
        },
        arrays,
    )


def _deterministic_npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer,
        np.ascontiguousarray(array),
        allow_pickle=False,
    )
    return buffer.getvalue()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _deterministic_npy_bytes(arrays[name]))
    temporary.replace(path)


def _checksum_manifest_bytes(root: Path) -> bytes:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"trajectory store artifact must not be a symlink: {relative}")
        if not path.is_file() or relative in {
            "artifact_checksums.sha256",
            ".sop05r-trajectories-complete",
        }:
            continue
        rows.append(f"{_sha256_file(path)}  {relative}\n".encode("ascii"))
    return b"".join(rows)


def _collection_digest(
    *,
    base_config_digest: str,
    record_digests: list[str],
) -> str:
    payload = {
        "collection_version": SOP05R_TRAJECTORY_COLLECTION_VERSION,
        "base_config_digest": base_config_digest,
        "record_semantic_digests": record_digests,
    }
    return _sha256_bytes(
        b"sop05r_trajectory_collection_v1\0" + _canonical_json_bytes(payload)
    )


def publish_sop05r_trajectory_store(
    output_dir: str | Path,
    records: tuple[Sop05rTrajectoryRecord, ...],
    *,
    base_config: Mapping[str, Any],
) -> Path:
    """Atomically publish a deterministic no-overwrite trajectory collection."""

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"trajectory store output already exists: {destination}")
    if not isinstance(records, tuple) or not records:
        raise TypeError("records must be a nonempty tuple")
    if any(not isinstance(record, Sop05rTrajectoryRecord) for record in records):
        raise TypeError("records must contain Sop05rTrajectoryRecord values")
    ordered = tuple(sorted(records, key=lambda record: record.event_id))
    event_ids = tuple(record.event_id for record in ordered)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("trajectory store event IDs must be unique")
    snapshot = _base_config_snapshot(base_config)
    for record in ordered:
        validate_sop05r_trajectory_record(record, base_config=snapshot)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        arrays_dir = staging / "arrays"
        arrays_dir.mkdir()
        rows = []
        for record in ordered:
            file_digest = hashlib.sha256(record.event_id.encode("utf-8")).hexdigest()[:24]
            relative = f"arrays/record-{file_digest}.npz"
            row, arrays = _record_row(record, array_file=relative)
            _write_deterministic_npz(staging / relative, arrays)
            rows.append(row)
        records_bytes = b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)
        (staging / "records.jsonl").write_bytes(records_bytes)
        base_config_digest = _sha256_bytes(_canonical_json_bytes(snapshot))
        record_digests = [str(row["record_semantic_digest"]) for row in rows]
        collection_digest = _collection_digest(
            base_config_digest=base_config_digest,
            record_digests=record_digests,
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "manifest_version": SOP05R_TRAJECTORY_MANIFEST_VERSION,
            "collection_version": SOP05R_TRAJECTORY_COLLECTION_VERSION,
            "record_count": len(rows),
            "event_ids": list(event_ids),
            "record_semantic_digests": record_digests,
            "records_jsonl_sha256": _sha256_bytes(records_bytes),
            "base_config": snapshot,
            "base_config_digest": base_config_digest,
            "collection_semantic_digest": collection_digest,
        }
        manifest_bytes = _json_file_bytes(manifest)
        (staging / "manifest.json").write_bytes(manifest_bytes)
        checksums = _checksum_manifest_bytes(staging)
        (staging / "artifact_checksums.sha256").write_bytes(checksums)
        marker = {
            "marker_version": SOP05R_TRAJECTORY_COMPLETION_MARKER_VERSION,
            "collection_semantic_digest": collection_digest,
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "artifact_checksums_sha256": _sha256_bytes(checksums),
        }
        (staging / ".sop05r-trajectories-complete").write_bytes(
            _json_file_bytes(marker)
        )
        loaded = load_sop05r_trajectory_store(staging)
        if loaded.collection_semantic_digest != collection_digest:
            raise ValueError("trajectory store self-reload digest mismatch")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"trajectory store output already exists: {destination}"
            )
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _verify_checksums(root: Path, expected_bytes: bytes) -> None:
    checksum_path = root / "artifact_checksums.sha256"
    try:
        observed = checksum_path.read_bytes()
    except OSError as exc:
        raise ValueError("trajectory store checksum manifest is missing") from exc
    if observed != expected_bytes:
        raise ValueError("trajectory store checksum manifest mismatch")


def _strict_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("trajectory store root must be a regular directory")
    expected = {
        "manifest.json",
        "records.jsonl",
        "arrays",
        "artifact_checksums.sha256",
        ".sop05r-trajectories-complete",
    }
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("trajectory store root entries mismatch")
    arrays = root / "arrays"
    if arrays.is_symlink() or not arrays.is_dir():
        raise ValueError("trajectory store arrays entry must be a directory")


def _load_rows(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("failed to read trajectory store records") from exc
    rows = []
    for line in lines:
        try:
            row = json.loads(line, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as exc:
            raise ValueError("trajectory store record JSON is invalid") from exc
        if not isinstance(row, dict):
            raise ValueError("trajectory store record row must be a mapping")
        rows.append(row)
    return rows


def _expected_array_keys(route_count: int) -> set[str]:
    return {
        _array_key(route_index, name)
        for route_index in range(route_count)
        for name in _TRAJECTORY_ARRAY_NAMES
    }


def _record_from_row(
    root: Path,
    row: Mapping[str, object],
    *,
    base_config: Mapping[str, Any],
) -> Sop05rTrajectoryRecord:
    expected_row_keys = {
        "record_version",
        "event_id",
        "base_state_id",
        "template_id",
        "planner_version",
        "config_digest",
        "shared_goal_world_pose",
        "candidate_trajectory_ids",
        "nominal_trajectory_id",
        "alternative_trajectory_ids",
        "array_file",
        "routes",
        "record_semantic_digest",
    }
    if set(row) != expected_row_keys:
        raise ValueError("trajectory store record keys mismatch")
    if row["record_version"] != SOP05R_TRAJECTORY_RECORD_VERSION:
        raise ValueError("trajectory store record version mismatch")
    route_rows = row["routes"]
    if not isinstance(route_rows, list) or not route_rows:
        raise ValueError("trajectory store record routes are invalid")
    relative = row["array_file"]
    if (
        not isinstance(relative, str)
        or not relative.startswith("arrays/record-")
        or not relative.endswith(".npz")
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("trajectory store array_file is invalid")
    array_path = root / relative
    if array_path.is_symlink() or not array_path.is_file():
        raise ValueError("trajectory store array file is missing")
    try:
        with np.load(array_path, allow_pickle=False) as payload:
            if set(payload.files) != _expected_array_keys(len(route_rows)):
                raise ValueError("trajectory store NPZ array keys mismatch")
            arrays = {name: payload[name].copy() for name in payload.files}
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("failed to load trajectory store NPZ") from exc
    routes = []
    for route_index, route_row in enumerate(route_rows):
        expected_route_keys = {
            "slot_id",
            "trajectory_id",
            "trajectory_metadata",
            "task_cost",
            "path_length_m",
            "represented_obstacle_clearance_m",
            "task_score",
            "array_semantic_digests",
        }
        if not isinstance(route_row, dict) or set(route_row) != expected_route_keys:
            raise ValueError("trajectory store route metadata keys mismatch")
        digests = route_row["array_semantic_digests"]
        if not isinstance(digests, dict) or set(digests) != set(_TRAJECTORY_ARRAY_NAMES):
            raise ValueError("trajectory store array digest keys mismatch")
        route_arrays = {}
        for name in _TRAJECTORY_ARRAY_NAMES:
            key = _array_key(route_index, name)
            array = arrays[key]
            if digests[name] != _array_semantic_digest(key, array):
                raise ValueError("trajectory store array semantic digest mismatch")
            route_arrays[name] = array
        metadata = route_row["trajectory_metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("trajectory metadata must be a mapping")
        trajectory = LocalTrajectory(
            trajectory_id=str(route_row["trajectory_id"]),
            poses=route_arrays["poses"],
            controls=route_arrays["controls"],
            swept_mask=route_arrays["swept_mask"],
            tta_map=route_arrays["tta_map"],
            braking_map=route_arrays["braking_map"],
            centerline_map=route_arrays["centerline_map"],
            task_cost=float(route_row["task_cost"]),
            metadata=metadata,
        )
        routes.append(
            ObstaclePlannedRoute(
                slot_id=str(route_row["slot_id"]),
                trajectory=trajectory,
                poses_world=route_arrays["poses_world"],
                waypoints_world=route_arrays["waypoints_world"],
                path_length_m=float(route_row["path_length_m"]),
                represented_obstacle_clearance_m=float(
                    route_row["represented_obstacle_clearance_m"]
                ),
                task_score=float(route_row["task_score"]),
            )
        )
    record = Sop05rTrajectoryRecord(
        event_id=str(row["event_id"]),
        base_state_id=str(row["base_state_id"]),
        template_id=str(row["template_id"]),
        planner_version=str(row["planner_version"]),
        config_digest=str(row["config_digest"]),
        shared_goal_world_pose=np.asarray(
            row["shared_goal_world_pose"], dtype=ARRAY_DTYPE
        ),
        nominal_trajectory_id=str(row["nominal_trajectory_id"]),
        alternative_trajectory_ids=tuple(
            str(value) for value in row["alternative_trajectory_ids"]
        ),
        routes=tuple(routes),
    )
    if list(record.candidate_trajectory_ids) != row["candidate_trajectory_ids"]:
        raise ValueError("trajectory store candidate order mismatch")
    validate_sop05r_trajectory_record(record, base_config=base_config)
    row_without_digest = {
        key: value for key, value in row.items() if key != "record_semantic_digest"
    }
    expected_digest = _sha256_bytes(
        b"sop05r_trajectory_record_v1\0"
        + _canonical_json_bytes(row_without_digest)
    )
    if row["record_semantic_digest"] != expected_digest:
        raise ValueError("trajectory store record semantic digest mismatch")
    return record


def load_sop05r_trajectory_store(
    root: str | Path,
) -> Sop05rTrajectoryStore:
    """Strictly load, authenticate, reconstruct, and recompute one store."""

    directory = Path(root)
    _strict_root(directory)
    expected_checksums = _checksum_manifest_bytes(directory)
    _verify_checksums(directory, expected_checksums)
    manifest = _read_json(directory / "manifest.json")
    marker = _read_json(directory / ".sop05r-trajectories-complete")
    if not isinstance(manifest, dict) or not isinstance(marker, dict):
        raise ValueError("trajectory store manifest and marker must be mappings")
    expected_manifest_keys = {
        "schema_version",
        "manifest_version",
        "collection_version",
        "record_count",
        "event_ids",
        "record_semantic_digests",
        "records_jsonl_sha256",
        "base_config",
        "base_config_digest",
        "collection_semantic_digest",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("trajectory store manifest keys mismatch")
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["manifest_version"] != SOP05R_TRAJECTORY_MANIFEST_VERSION
        or manifest["collection_version"] != SOP05R_TRAJECTORY_COLLECTION_VERSION
    ):
        raise ValueError("trajectory store manifest version mismatch")
    expected_marker_keys = {
        "marker_version",
        "collection_semantic_digest",
        "manifest_sha256",
        "artifact_checksums_sha256",
    }
    if set(marker) != expected_marker_keys:
        raise ValueError("trajectory store completion marker keys mismatch")
    if marker["marker_version"] != SOP05R_TRAJECTORY_COMPLETION_MARKER_VERSION:
        raise ValueError("trajectory store completion marker version mismatch")
    if marker["manifest_sha256"] != _sha256_file(directory / "manifest.json"):
        raise ValueError("trajectory store completion marker manifest checksum mismatch")
    if marker["artifact_checksums_sha256"] != _sha256_file(
        directory / "artifact_checksums.sha256"
    ):
        raise ValueError("trajectory store completion marker checksum mismatch")
    records_bytes = (directory / "records.jsonl").read_bytes()
    if manifest["records_jsonl_sha256"] != _sha256_bytes(records_bytes):
        raise ValueError("trajectory store records checksum mismatch")
    base_config = manifest["base_config"]
    if not isinstance(base_config, dict):
        raise ValueError("trajectory store base_config must be a mapping")
    base_digest = _sha256_bytes(_canonical_json_bytes(base_config))
    if manifest["base_config_digest"] != base_digest:
        raise ValueError("trajectory store base_config digest mismatch")
    rows = _load_rows(directory / "records.jsonl")
    if len(rows) != manifest["record_count"]:
        raise ValueError("trajectory store record count mismatch")
    event_ids = [row.get("event_id") for row in rows]
    if event_ids != manifest["event_ids"] or event_ids != sorted(event_ids):
        raise ValueError("trajectory store event order mismatch")
    expected_array_files = {str(row.get("array_file")) for row in rows}
    observed_array_files = {
        path.relative_to(directory).as_posix()
        for path in (directory / "arrays").iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if expected_array_files != observed_array_files:
        raise ValueError("trajectory store array file set mismatch")
    records = tuple(
        _record_from_row(directory, row, base_config=base_config) for row in rows
    )
    record_digests = [str(row["record_semantic_digest"]) for row in rows]
    if record_digests != manifest["record_semantic_digests"]:
        raise ValueError("trajectory store record digest order mismatch")
    collection_digest = _collection_digest(
        base_config_digest=base_digest,
        record_digests=record_digests,
    )
    if manifest["collection_semantic_digest"] != collection_digest:
        raise ValueError("trajectory store collection semantic digest mismatch")
    if marker["collection_semantic_digest"] != collection_digest:
        raise ValueError("trajectory store completion marker collection mismatch")
    return Sop05rTrajectoryStore(
        records=records,
        manifest=manifest,
        collection_semantic_digest=collection_digest,
    )
