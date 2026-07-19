"""Deterministic, schema-shaped toy inputs for SOP08--10.

This module is deliberately production code rather than a test-only fixture so
the command-line training paths never import from ``tests``.  It constructs
real :class:`~src.contracts.RiskSample` objects and validates every object
before publication.  Oracle future occupancy and robot query footprints live
in a separate sample-id-keyed sidecar; they are never inserted into a model
input mapping or ``RiskSample.metadata``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.contracts import (
    HISTORY_CHANNELS,
    INPUT_CHANNELS,
    N_HISTORY_CHANNELS,
    N_STATE_CHANNELS,
    N_TRAJECTORY_CHANNELS,
    ROBOT_STATE_DIM,
    SCHEMA_VERSION,
    STATE_CHANNELS,
    TRAJECTORY_CHANNELS,
    GridSpec,
    RiskSample,
    validate_risk_sample,
)

TOY_DATASET_LAYOUT_VERSION = "toy_risk_dataset_v3"
TOY_HISTORY_STEPS = 8
TOY_FUTURE_STEPS = 15
TOY_DT_S = 0.2
TOY_FOOTPRINT_PROXY_KIND = "single_grid_cell_square"
TOY_FOOTPRINT_CONTACT_POLICY = "positive_area_overlap"
TOY_NEAR_MISS_THRESHOLD_M = 0.25
TOY_EMPTY_CLEARANCE_SENTINEL_M = 99.0
TOY_FUTURE_ENDPOINT_TIMES_S = (
    np.arange(1, TOY_FUTURE_STEPS + 1, dtype=np.float32) * np.float32(TOY_DT_S)
)
TOY_FUTURE_ENDPOINT_TIMES_S.setflags(write=False)
TOY_CASES: tuple[str, ...] = (
    "collision",
    "near_miss",
    "temporal_safe",
    "same_area_safe",
    "irrelevant_hidden",
    "empty",
    "ood",
)
TOY_PAIRED_CASES: tuple[str, ...] = tuple(
    case for case in TOY_CASES if case != "ood"
)
TOY_SPLITS: tuple[str, ...] = ("train", "calibration", "val", "test")
TOY_DATASET_MANIFEST_KEYS = frozenset(
    {
        "mode",
        "dataset_layout_version",
        "schema_version",
        "channel_spec",
        "split",
        "sample_count",
        "seed",
        "ordered_sample_ids",
        "ordered_sample_ids_digest_sha256",
        "model_input_digest_sha256",
        "label_digest_sha256",
        "ordered_sample_digest_sha256",
        "manifest_rows_digest_sha256",
        "label_sidecars_digest_sha256",
        "future_endpoint_times_s",
        "grid",
        "toy_dataset_manifest_digest",
    }
)
TOY_MANIFEST_ROW_KEYS = frozenset(
    {
        "sample_id",
        "split",
        "pair_group_id",
        "event_type",
        "recording_id",
        "session_id",
        "source_object_id",
        "snippet_id",
        "base_state_id",
        "seed_namespace",
        "trajectory_id",
        "occluder_id",
        "background_id",
        "collision_label",
        "risk_severity",
        "min_clearance",
        "near_miss",
        "first_collision_time",
        "critical_object_id",
        "blind_type",
        "critical_area_fraction",
        "age_s",
        "density_fraction",
        "target_object_type",
        "footprint_kind",
        "footprint_dimensions_m",
        "robot_footprint_kind",
        "robot_footprint_dimensions_m",
        "footprint_contact_policy",
        "ood_tag",
        "pair_eligible",
    }
)


def frozen_channel_spec() -> dict[str, list[str]]:
    """Return the JSON-safe ordered channel contract imported from SOP00."""

    return {
        "history": list(HISTORY_CHANNELS),
        "state": list(STATE_CHANNELS),
        "trajectory": list(TRAJECTORY_CHANNELS),
        "flat": list(INPUT_CHANNELS),
    }


@dataclass(frozen=True)
class ToyLabelSidecar:
    """Label/query arrays joined to one sample only through ``sample_id``."""

    sample_id: str
    hidden_risk_occupancy: np.ndarray  # [T,H,W], label only
    robot_future_footprints: np.ndarray  # [T,H,W], query geometry only
    future_endpoint_times_s: np.ndarray  # [T], dt ... horizon


@dataclass(frozen=True)
class ToyRiskDataset:
    """One deterministic split publication and its provenance manifest."""

    split: str
    grid: GridSpec
    samples: tuple[RiskSample, ...]
    sidecars: tuple[ToyLabelSidecar, ...]
    manifest_rows: tuple[dict[str, object], ...]
    manifest: dict[str, object]
    manifest_digest: str

    def sidecar_by_sample_id(self) -> dict[str, ToyLabelSidecar]:
        return {sidecar.sample_id: sidecar for sidecar in self.sidecars}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _stable_uint64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), byteorder="big"
    )


def _update_array_digest(
    digest: "hashlib._Hash", name: str, value: np.ndarray
) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(size) for size in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.view(np.uint8))


def toy_grid_manifest(grid: GridSpec) -> dict[str, object]:
    """Return the exact JSON-safe grid contract used by toy publications."""

    return {
        "height": grid.height,
        "width": grid.width,
        "history_steps": grid.history_steps,
        "future_steps": grid.future_steps,
        "resolution_m": grid.resolution_m,
        "sample_dt_s": TOY_DT_S,
        "future_time_layout": "endpoint_dt_to_horizon",
    }


def toy_sample_id_sequence_digest(sample_ids: Sequence[str]) -> str:
    """Hash an explicit ordered sample-ID sequence."""

    return hashlib.sha256(_canonical_json(list(sample_ids))).hexdigest()


def toy_ordered_sample_ids_digest(samples: Sequence[RiskSample]) -> str:
    """Bind exact ordered IDs independently of numerical sample content."""

    return toy_sample_id_sequence_digest([sample.sample_id for sample in samples])


def toy_model_input_digest(samples: Sequence[RiskSample]) -> str:
    """Hash ordered deployable numerical inputs without identity strings."""

    digest = hashlib.sha256(b"toy-model-inputs-v1\0")
    for index, sample in enumerate(samples):
        digest.update(index.to_bytes(8, byteorder="big", signed=False))
        for name, value in (
            ("bev_history", sample.bev_history),
            ("state_channels", sample.state_channels),
            ("trajectory_channels", sample.trajectory_channels),
            ("robot_state", sample.robot_state),
        ):
            _update_array_digest(digest, name, value)
    return digest.hexdigest()


def toy_label_digest(samples: Sequence[RiskSample]) -> str:
    """Hash ordered scalar labels, including near-miss and collision time."""

    labels = [
        {
            "collision_label": sample.collision_label,
            "risk_severity": sample.risk_severity,
            "min_clearance": sample.min_clearance,
            "near_miss": sample.near_miss,
            "first_collision_time": sample.first_collision_time,
        }
        for sample in samples
    ]
    return hashlib.sha256(
        b"toy-labels-v1\0" + _canonical_json(labels)
    ).hexdigest()


def toy_ordered_sample_digest(samples: Sequence[RiskSample]) -> str:
    """Bind ordered identities, inputs, labels, and non-oracle provenance."""

    identities = [
        {
            "sample_id": sample.sample_id,
            "split": sample.split,
            "base_state_id": sample.base_state_id,
            "pair_group_id": sample.pair_group_id,
            "event_type": sample.event_type,
            "metadata": sample.metadata,
        }
        for sample in samples
    ]
    projection = {
        "identity": identities,
        "ordered_sample_ids_digest_sha256": toy_ordered_sample_ids_digest(samples),
        "model_input_digest_sha256": toy_model_input_digest(samples),
        "label_digest_sha256": toy_label_digest(samples),
    }
    return hashlib.sha256(
        b"toy-ordered-samples-v1\0" + _canonical_json(projection)
    ).hexdigest()


def toy_manifest_rows_digest(rows: Sequence[Mapping[str, object]]) -> str:
    """Hash the exact ordered, finite JSON manifest rows."""

    return hashlib.sha256(
        b"toy-manifest-rows-v1\0" + _canonical_json(list(rows))
    ).hexdigest()


def toy_label_sidecars_digest(sidecars: Sequence[ToyLabelSidecar]) -> str:
    """Hash ordered label-only/query sidecars, including IDs and array schemas."""

    digest = hashlib.sha256(b"toy-label-sidecars-v1\0")
    for index, sidecar in enumerate(sidecars):
        digest.update(index.to_bytes(8, byteorder="big", signed=False))
        digest.update(sidecar.sample_id.encode("utf-8"))
        digest.update(b"\0")
        for name, value in (
            ("hidden_risk_occupancy", sidecar.hidden_risk_occupancy),
            ("robot_future_footprints", sidecar.robot_future_footprints),
            ("future_endpoint_times_s", sidecar.future_endpoint_times_s),
        ):
            _update_array_digest(digest, name, value)
    return digest.hexdigest()


def toy_dataset_manifest_digest(manifest_or_header: Mapping[str, object]) -> str:
    """Return BLAKE2b-128 over the canonical manifest header.

    The digest field itself is excluded.  The header contains the ordered-ID,
    deployable-input, scalar-label, full-sample, row, and label-sidecar SHA-256
    component digests, so this one value authenticates the complete toy
    publication without mixing oracle arrays into model inputs.
    """

    header = {
        key: value
        for key, value in manifest_or_header.items()
        if key != "toy_dataset_manifest_digest"
    }
    return hashlib.blake2b(_canonical_json(header), digest_size=16).hexdigest()


def _grid(grid_size: int) -> GridSpec:
    if not isinstance(grid_size, int) or isinstance(grid_size, bool) or grid_size < 8:
        raise ValueError("grid_size must be an integer >= 8")
    return GridSpec(
        height=grid_size,
        width=grid_size,
        history_steps=TOY_HISTORY_STEPS,
        future_steps=TOY_FUTURE_STEPS,
        resolution_m=0.2,
    )


def _robot_future_footprints(grid: GridSpec, context_token: int) -> np.ndarray:
    masks = np.zeros(
        (grid.future_steps, grid.height, grid.width), dtype=np.float32
    )
    row = grid.height // 2
    start = 1 + int(context_token % 2)
    end = grid.width - 2 - int((context_token >> 1) % 2)
    columns = np.rint(
        np.linspace(start, end, grid.future_steps, dtype=np.float32)
    ).astype(np.int64)
    masks[np.arange(grid.future_steps), row, columns] = np.float32(1.0)
    return masks


def _one_cell(mask: np.ndarray, step: int, row: int, column: int) -> None:
    h, w = mask.shape[-2:]
    mask[step, int(np.clip(row, 0, h - 1)), int(np.clip(column, 0, w - 1))] = 1.0


def _hidden_occupancy(
    case: str, robot_masks: np.ndarray, *, context_token: int
) -> np.ndarray:
    occupancy = np.zeros_like(robot_masks, dtype=np.float32)
    steps, height, width = occupancy.shape
    robot_cells = [
        tuple(np.argwhere(robot_masks[step] > 0.5)[0]) for step in range(steps)
    ]
    center_row = height // 2

    if case == "empty":
        return occupancy
    if case == "collision":
        conflict_step = min(4 + int(context_token % 3), steps - 1)
        conflict_row, conflict_column = robot_cells[conflict_step]
        for step in range(steps):
            delta = step - conflict_step
            if delta == 0:
                row_offset = 0
            else:
                row_offset = int(np.sign(delta)) * min(
                    3, max(1, (abs(delta) + 1) // 2)
                )
            _one_cell(
                occupancy,
                step,
                conflict_row + row_offset,
                conflict_column,
            )
        return occupancy
    if case == "near_miss":
        for step, (row, column) in enumerate(robot_cells):
            _one_cell(occupancy, step, row + 1, column)
        return occupancy
    if case == "temporal_safe":
        # The actor moves ahead of the robot through the same corridor and
        # exits the local grid before the robot arrives at those cells.
        # Three cells leave two full proxy-cell widths of surface clearance.
        lead_cells = 3 + int((context_token >> 2) % 2)
        for step, (row, column) in enumerate(robot_cells):
            target_column = column + lead_cells
            if target_column <= width - 2:
                _one_cell(occupancy, step, row, target_column)
        return occupancy
    if case == "same_area_safe":
        row_offset = 3
        for step, (_, column) in enumerate(robot_cells):
            _one_cell(occupancy, step, center_row + row_offset, column)
        return occupancy
    if case == "irrelevant_hidden":
        phase = int((context_token >> 3) % 3)
        for step in range(steps):
            _one_cell(
                occupancy,
                step,
                1 + int((context_token >> 5) % 2),
                width - 2 - ((step + phase) // 3) % max(1, width - 3),
            )
        return occupancy
    if case == "ood":
        for step in range(steps):
            row = height - 2 - min(height - 3, step // 3)
            column = 1 + min(width - 3, step // 2)
            current_row, current_column = robot_cells[step]
            if (row, column) == (current_row, current_column):
                row = max(1, row - 1)
            _one_cell(occupancy, step, row, column)
        return occupancy
    raise ValueError(f"unknown toy case: {case!r}")


def _footprint_labels(
    occupancy: np.ndarray,
    robot_masks: np.ndarray,
    *,
    resolution_m: float,
    target_footprint_kind: str,
    target_footprint_dimensions_m: Sequence[float],
    robot_footprint_kind: str,
    robot_footprint_dimensions_m: Sequence[float],
) -> tuple[int, float, float, int, float | None]:
    """Label time-aligned square footprint proxies from their surface gap.

    A toy target and robot footprint each occupy the interior of one grid-cell
    square.  Positive-area overlap is collision; boundary contact has zero
    clearance and is a near miss rather than a collision.
    """

    for role, kind, dimensions in (
        ("target", target_footprint_kind, target_footprint_dimensions_m),
        ("robot", robot_footprint_kind, robot_footprint_dimensions_m),
    ):
        if kind != TOY_FOOTPRINT_PROXY_KIND:
            raise ValueError(f"{role} footprint must use {TOY_FOOTPRINT_PROXY_KIND}")
        if len(dimensions) != 2 or any(
            not math.isclose(float(value), resolution_m, rel_tol=0.0, abs_tol=1e-9)
            for value in dimensions
        ):
            raise ValueError(
                f"{role} single-cell footprint dimensions must equal grid resolution"
            )

    combined_half_extents_m = (
        np.asarray(target_footprint_dimensions_m, dtype=np.float64)
        + np.asarray(robot_footprint_dimensions_m, dtype=np.float64)
    ) / 2.0
    per_step_clearance: list[tuple[float, int]] = []
    collision_steps: list[int] = []
    for step in range(occupancy.shape[0]):
        object_cells = np.argwhere(occupancy[step] > 0.5)
        robot_cells = np.argwhere(robot_masks[step] > 0.5)
        if object_cells.size == 0:
            continue
        center_delta_m = (
            np.abs(object_cells[:, None, :] - robot_cells[None, :, :]).astype(
                np.float64
            )
            * resolution_m
        )
        axis_gaps_m = center_delta_m - combined_half_extents_m
        positive_area_overlap = np.all(axis_gaps_m < 0.0, axis=-1)
        if np.any(positive_area_overlap):
            penetrations = np.min(
                -axis_gaps_m[positive_area_overlap], axis=-1
            )
            signed_clearance = -float(np.max(penetrations))
            collision_steps.append(step)
        else:
            separation_m = np.maximum(axis_gaps_m, 0.0)
            signed_clearance = float(
                np.sqrt(np.sum(separation_m**2, axis=-1)).min()
            )
        per_step_clearance.append((signed_clearance, step))

    collision = int(bool(collision_steps))
    first_collision_time = None
    if collision:
        first_step = collision_steps[0]
        first_collision_time = float((first_step + 1) * TOY_DT_S)

    if not per_step_clearance:
        min_clearance = TOY_EMPTY_CLEARANCE_SENTINEL_M
        severity = 0.0
        near_miss = 0
    else:
        min_clearance, closest_step = min(
            per_step_clearance, key=lambda item: item[0]
        )
        if collision:
            severity = 1.0
            near_miss = 0
        else:
            endpoint_time = float((closest_step + 1) * TOY_DT_S)
            severity = math.exp(-min_clearance / 0.5) * math.exp(-endpoint_time / 2.0)
            severity = float(np.clip(severity, 0.0, 1.0))
            near_miss = int(min_clearance <= TOY_NEAR_MISS_THRESHOLD_M)
    return collision, float(severity), float(min_clearance), near_miss, first_collision_time


def _history_and_state(
    case: str,
    grid: GridSpec,
    robot_masks: np.ndarray,
    occupancy: np.ndarray,
    context_token: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history = np.zeros(
        (grid.history_steps, N_HISTORY_CHANNELS, grid.height, grid.width),
        dtype=np.float32,
    )
    state = np.zeros(
        (N_STATE_CHANNELS, grid.height, grid.width), dtype=np.float32
    )
    trajectory = np.zeros(
        (N_TRAJECTORY_CHANNELS, grid.height, grid.width), dtype=np.float32
    )

    history[:, 1, :, : grid.width // 2] = 1.0
    history_cells: list[tuple[int, int]] = []
    if case != "empty":
        future_positions = [
            np.argwhere(occupancy[step] > 0.5) for step in range(grid.future_steps)
        ]
        if future_positions[0].shape != (1, 2):
            raise ValueError("non-empty toy motion must occupy one cell at first endpoint")
        first = future_positions[0][0].astype(np.int64)
        velocity = np.zeros(2, dtype=np.int64)
        for positions in future_positions[1:]:
            if positions.shape == (1, 2) and not np.array_equal(positions[0], first):
                velocity = np.sign(positions[0].astype(np.int64) - first).astype(
                    np.int64
                )
                break
        for history_step in range(grid.history_steps):
            offset = history_step - grid.history_steps
            position = first + offset * velocity
            row = int(np.clip(position[0], 1, grid.height - 2))
            column = int(np.clip(position[1], 1, grid.width - 2))
            history_cells.append((row, column))
            history[history_step, 0, row, column] = 1.0
            history[history_step, 1, row, column] = 1.0
        last_row, last_column = history_cells[-1]

    state_index = {name: index for index, name in enumerate(STATE_CHANNELS)}
    swept = np.any(robot_masks > 0.5, axis=0).astype(np.float32)
    # All historical actors are clipped to the interior.  The current toy
    # blind region is therefore the shared interior, while its one-cell border
    # starts visible free.  Known static cells are reclassified visible occupied
    # below so the final masks remain an exact, non-overlapping partition.
    state[state_index["current_visible_free"]] = 1.0
    state[state_index["current_visible_free"], 1:-1, 1:-1] = 0.0
    state[state_index["current_unobservable_mask"], 1:-1, 1:-1] = 1.0
    if case != "empty":
        state[state_index["last_seen_occupancy"], last_row, last_column] = 1.0
    age_value = np.float32(0.1 + ((context_token >> 11) % 7000) / 10000.0)
    state[state_index["occlusion_age_map"], 1:-1, 1:-1] = age_value
    static_obstacles = state[state_index["static_obstacle_map"]]
    static_obstacles[0, swept[0] == 0.0] = 1.0
    static_obstacles[-1, swept[-1] == 0.0] = 1.0
    # The reserved lower-right interior cell is outside the center-row robot
    # sweep and every designed historical actor path.  Keeping it independent
    # of the counterfactual case preserves matched-group observed context.
    obstacle_row, obstacle_column = grid.height - 2, grid.width - 2
    if swept[obstacle_row, obstacle_column] != 0.0 or (
        case != "empty" and (obstacle_row, obstacle_column) == (last_row, last_column)
    ):
        raise ValueError("reserved toy static obstacle cell invariant drift")
    static_obstacles[obstacle_row, obstacle_column] = 1.0
    static_mask = static_obstacles > 0.5
    state[state_index["current_visible_free"]][static_mask] = 0.0
    state[state_index["current_unobservable_mask"]][static_mask] = 0.0
    state[state_index["current_visible_occupied"]][static_mask] = 1.0
    state[state_index["occlusion_age_map"]][static_mask] = 0.0
    first_robot_cell = tuple(np.argwhere(robot_masks[0] > 0.5)[0])
    state[(state_index["robot_footprint"], *first_robot_cell)] = 1.0

    trajectory_index = {name: index for index, name in enumerate(TRAJECTORY_CHANNELS)}
    trajectory[trajectory_index["swept_volume_mask"]] = swept
    trajectory[trajectory_index["centerline_map"]] = swept
    tta = np.full((grid.height, grid.width), -1.0, dtype=np.float32)
    for step in range(grid.future_steps):
        tta[robot_masks[step] > 0.5] = np.float32((step + 1) * TOY_DT_S)
    trajectory[trajectory_index["time_to_arrival_map"]] = tta
    trajectory[trajectory_index["braking_margin_map"]] = swept * np.float32(
        0.2 + ((context_token >> 23) % 2000) / 10000.0
    )

    speed = np.float32(0.3 + (context_token % 1000003) / 5000015.0)
    yaw_rate = np.float32(-0.12 + ((context_token >> 17) % 240001) / 1000000.0)
    state[state_index["robot_velocity_channel"]] = swept * speed
    state[state_index["robot_yaw_rate_channel"]] = swept * yaw_rate
    robot_state = np.asarray([speed, yaw_rate], dtype=np.float32)
    return history, state, trajectory, robot_state


def _validate_sidecar(sidecar: ToyLabelSidecar, grid: GridSpec) -> None:
    expected_map_shape = (grid.future_steps, grid.height, grid.width)
    for name, value in (
        ("hidden_risk_occupancy", sidecar.hidden_risk_occupancy),
        ("robot_future_footprints", sidecar.robot_future_footprints),
    ):
        if value.shape != expected_map_shape or value.dtype != np.float32:
            raise ValueError(f"{name} must be float32 {expected_map_shape}")
        if not np.isfinite(value).all() or np.any((value < 0.0) | (value > 1.0)):
            raise ValueError(f"{name} must be finite probabilities/masks in [0,1]")
    if sidecar.future_endpoint_times_s.shape != (grid.future_steps,):
        raise ValueError("future_endpoint_times_s shape mismatch")
    if sidecar.future_endpoint_times_s.dtype != np.float32:
        raise ValueError("future_endpoint_times_s must be float32")
    if not np.array_equal(sidecar.future_endpoint_times_s, TOY_FUTURE_ENDPOINT_TIMES_S):
        raise ValueError("future_endpoint_times_s must use dt-to-horizon endpoints")


def _sample_schedule(count: int) -> list[tuple[str, bool, int]]:
    """Plan complete six-case matched groups plus OOD-only singleton groups."""

    schedule: list[tuple[str, bool, int]] = []
    normal_pair_count = count // 7
    ood_index = 0
    for pair_index in range(normal_pair_count):
        schedule.extend(
            (case, True, pair_index) for case in TOY_PAIRED_CASES
        )
        schedule.append(("ood", False, ood_index))
        ood_index += 1
    while len(schedule) < count:
        schedule.append(("ood", False, ood_index))
        ood_index += 1
    return schedule


def make_toy_risk_dataset(
    *,
    split: str = "train",
    count: int = 128,
    seed: int = 42,
    grid_size: int = 16,
) -> ToyRiskDataset:
    """Build and validate one deterministic toy split publication."""

    if split not in TOY_SPLITS:
        raise ValueError(f"split must be one of {TOY_SPLITS}, got {split!r}")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    grid = _grid(grid_size)
    samples: list[RiskSample] = []
    sidecars: list[ToyLabelSidecar] = []
    rows: list[dict[str, object]] = []

    normal_pair_count = count // 7
    for index, (case, pair_eligible, group_index) in enumerate(
        _sample_schedule(count)
    ):
        if pair_eligible:
            context_index = group_index
            identity_stem = f"pair-{group_index:05d}"
            pair_group_id = f"toy-{split}-{identity_stem}"
            snippet_id = f"toy-{split}-snippet-{group_index:05d}"
        else:
            context_index = normal_pair_count + group_index
            identity_stem = f"ood-singleton-{group_index:05d}"
            pair_group_id = f"toy-{split}-{identity_stem}"
            snippet_id = f"toy-{split}-snippet-{identity_stem}"
        context_token = _stable_uint64(
            "toy-risk-context-v2", split, seed, context_index
        )
        recording_id = f"toy-{split}-recording-{context_index // 4:05d}"
        session_id = f"toy-{split}-session-{context_index // 8:05d}"
        source_object_id = f"toy-{split}-object-{identity_stem}"
        base_state_id = f"toy-{split}-base-{identity_stem}"
        trajectory_id = f"toy-{split}-trajectory-{identity_stem}"
        occluder_id = f"toy-{split}-occluder-{identity_stem}"
        background_id = f"toy-{split}-background-{identity_stem}"
        seed_namespace = f"toy:{split}:seed:{seed}"
        sample_id = f"toy-{split}-{seed:08d}-{index:06d}"

        # Freeze semantic footprint declarations before any label geometry is
        # generated.  Both toy masks are explicit one-grid-cell square proxies.
        object_types = ("human", "carried_object", "unknown_dynamic")
        object_type = object_types[context_token % len(object_types)]
        target_footprint_kind = TOY_FOOTPRINT_PROXY_KIND
        target_footprint_dimensions = [grid.resolution_m, grid.resolution_m]
        robot_footprint_kind = TOY_FOOTPRINT_PROXY_KIND
        robot_footprint_dimensions = [grid.resolution_m, grid.resolution_m]
        robot_masks = _robot_future_footprints(grid, context_token)
        occupancy = _hidden_occupancy(
            case, robot_masks, context_token=context_token
        )
        collision, severity, clearance, near_miss, collision_time = _footprint_labels(
            occupancy,
            robot_masks,
            resolution_m=grid.resolution_m,
            target_footprint_kind=target_footprint_kind,
            target_footprint_dimensions_m=target_footprint_dimensions,
            robot_footprint_kind=robot_footprint_kind,
            robot_footprint_dimensions_m=robot_footprint_dimensions,
        )
        history, state, trajectory, robot_state = _history_and_state(
            case, grid, robot_masks, occupancy, context_token
        )
        critical_object_id: str | None = None if case == "empty" else source_object_id
        metadata = {
            "source": "deterministic_toy_risk_learning",
            "recording_id": recording_id,
            "session_id": session_id,
            "source_object_id": source_object_id,
            "snippet_id": snippet_id,
            "seed_namespace": seed_namespace,
            "critical_object_id": critical_object_id,
            "trajectory_id": trajectory_id,
            "occluder_id": occluder_id,
            "background_id": background_id,
            "blind_type": (
                "structural" if (context_token >> 5) % 2 == 0 else "dynamic"
            ),
            "critical_area_fraction": float(
                ((context_token >> 9) % 2001) / 10000.0
            ),
            "age_s": float(((context_token >> 17) % 5001) / 1000.0),
            "density_fraction": float(
                ((context_token >> 29) % 1001) / 10000.0
            ),
            "target_object_type": object_type,
            "footprint_kind": target_footprint_kind,
            "footprint_dimensions_m": list(target_footprint_dimensions),
            "robot_footprint_kind": robot_footprint_kind,
            "robot_footprint_dimensions_m": list(robot_footprint_dimensions),
            "footprint_contact_policy": TOY_FOOTPRINT_CONTACT_POLICY,
            "ood_tag": "heldout_motion" if case == "ood" else "in_distribution",
            "pair_eligible": pair_eligible,
        }
        sample = RiskSample(
            sample_id=sample_id,
            split=split,
            base_state_id=base_state_id,
            pair_group_id=pair_group_id,
            event_type=case,
            bev_history=history,
            state_channels=state,
            trajectory_channels=trajectory,
            robot_state=robot_state,
            collision_label=collision,
            risk_severity=severity,
            min_clearance=clearance,
            near_miss=near_miss,
            first_collision_time=collision_time,
            metadata=metadata,
        )
        validate_risk_sample(sample, grid)
        sidecar = ToyLabelSidecar(
            sample_id=sample_id,
            hidden_risk_occupancy=occupancy,
            robot_future_footprints=robot_masks,
            future_endpoint_times_s=TOY_FUTURE_ENDPOINT_TIMES_S.copy(),
        )
        _validate_sidecar(sidecar, grid)
        samples.append(sample)
        sidecars.append(sidecar)
        rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "pair_group_id": pair_group_id,
                "event_type": case,
                "recording_id": recording_id,
                "session_id": session_id,
                "source_object_id": source_object_id,
                "snippet_id": snippet_id,
                "base_state_id": base_state_id,
                "seed_namespace": seed_namespace,
                "trajectory_id": trajectory_id,
                "occluder_id": occluder_id,
                "background_id": background_id,
                "collision_label": collision,
                "risk_severity": severity,
                "min_clearance": clearance,
                "near_miss": near_miss,
                "first_collision_time": collision_time,
                "critical_object_id": critical_object_id,
                "blind_type": metadata["blind_type"],
                "critical_area_fraction": metadata["critical_area_fraction"],
                "age_s": metadata["age_s"],
                "density_fraction": metadata["density_fraction"],
                "target_object_type": object_type,
                "footprint_kind": target_footprint_kind,
                "footprint_dimensions_m": list(target_footprint_dimensions),
                "robot_footprint_kind": robot_footprint_kind,
                "robot_footprint_dimensions_m": list(robot_footprint_dimensions),
                "footprint_contact_policy": TOY_FOOTPRINT_CONTACT_POLICY,
                "ood_tag": metadata["ood_tag"],
                "pair_eligible": metadata["pair_eligible"],
            }
        )

    frozen_samples = tuple(samples)
    frozen_sidecars = tuple(sidecars)
    frozen_rows = tuple(rows)
    ordered_sample_ids = [sample.sample_id for sample in frozen_samples]
    header: dict[str, object] = {
        "mode": "toy",
        "dataset_layout_version": TOY_DATASET_LAYOUT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "channel_spec": frozen_channel_spec(),
        "split": split,
        "sample_count": count,
        "seed": seed,
        "ordered_sample_ids": ordered_sample_ids,
        "ordered_sample_ids_digest_sha256": toy_ordered_sample_ids_digest(
            frozen_samples
        ),
        "model_input_digest_sha256": toy_model_input_digest(frozen_samples),
        "label_digest_sha256": toy_label_digest(frozen_samples),
        "ordered_sample_digest_sha256": toy_ordered_sample_digest(
            frozen_samples
        ),
        "manifest_rows_digest_sha256": toy_manifest_rows_digest(frozen_rows),
        "label_sidecars_digest_sha256": toy_label_sidecars_digest(
            frozen_sidecars
        ),
        "future_endpoint_times_s": [
            round(float(value), 7) for value in TOY_FUTURE_ENDPOINT_TIMES_S
        ],
        "grid": toy_grid_manifest(grid),
    }
    manifest_digest = toy_dataset_manifest_digest(header)
    manifest = {
        **header,
        "toy_dataset_manifest_digest": manifest_digest,
    }
    dataset = ToyRiskDataset(
        split=split,
        grid=grid,
        samples=frozen_samples,
        sidecars=frozen_sidecars,
        manifest_rows=frozen_rows,
        manifest=manifest,
        manifest_digest=manifest_digest,
    )
    validate_toy_risk_dataset_publication(dataset)
    return dataset


def validate_toy_risk_dataset_publication(
    dataset: ToyRiskDataset,
) -> dict[str, str]:
    """Recompute and validate every component of one toy publication."""

    if not isinstance(dataset, ToyRiskDataset):
        raise TypeError("dataset must be ToyRiskDataset")
    manifest = dataset.manifest
    if not isinstance(manifest, Mapping):
        raise ValueError("toy dataset manifest must be a mapping")
    if set(manifest) != set(TOY_DATASET_MANIFEST_KEYS):
        raise ValueError("toy dataset manifest top-level keys mismatch")
    if dataset.split != manifest.get("split"):
        raise ValueError("toy dataset split does not match manifest split")
    if manifest.get("grid") != toy_grid_manifest(dataset.grid):
        raise ValueError("toy dataset grid does not match manifest grid")
    count = len(dataset.samples)
    if not (
        count
        == len(dataset.sidecars)
        == len(dataset.manifest_rows)
        == manifest.get("sample_count")
    ):
        raise ValueError("toy dataset component counts do not match sample_count")

    sample_ids = tuple(sample.sample_id for sample in dataset.samples)
    sidecar_ids = tuple(sidecar.sample_id for sidecar in dataset.sidecars)
    row_ids: list[str] = []
    for row in dataset.manifest_rows:
        if not isinstance(row, Mapping) or set(row) != set(TOY_MANIFEST_ROW_KEYS):
            raise ValueError("toy manifest row keys mismatch")
        row_ids.append(str(row["sample_id"]))
    if sample_ids != tuple(manifest.get("ordered_sample_ids", ())):
        raise ValueError("ordered sample IDs do not match manifest")
    if sidecar_ids != sample_ids or tuple(row_ids) != sample_ids:
        raise ValueError("rows and sidecars must follow exact ordered sample IDs")
    if len(set(sample_ids)) != count:
        raise ValueError("toy publication sample IDs must be unique")

    for sample, sidecar in zip(dataset.samples, dataset.sidecars):
        validate_risk_sample(sample, dataset.grid)
        _validate_sidecar(sidecar, dataset.grid)

    expected_components = {
        "ordered_sample_ids_digest_sha256": toy_ordered_sample_ids_digest(
            dataset.samples
        ),
        "model_input_digest_sha256": toy_model_input_digest(dataset.samples),
        "label_digest_sha256": toy_label_digest(dataset.samples),
        "ordered_sample_digest_sha256": toy_ordered_sample_digest(
            dataset.samples
        ),
        "manifest_rows_digest_sha256": toy_manifest_rows_digest(
            dataset.manifest_rows
        ),
        "label_sidecars_digest_sha256": toy_label_sidecars_digest(
            dataset.sidecars
        ),
    }
    for name, expected in expected_components.items():
        if manifest.get(name) != expected:
            raise ValueError(f"{name} mismatch")

    expected_manifest_digest = toy_dataset_manifest_digest(manifest)
    if manifest.get("toy_dataset_manifest_digest") != expected_manifest_digest:
        raise ValueError("toy_dataset_manifest_digest mismatch")
    if dataset.manifest_digest != expected_manifest_digest:
        raise ValueError("ToyRiskDataset.manifest_digest mismatch")

    row_by_id = {
        str(row["sample_id"]): row for row in dataset.manifest_rows
    }
    for sample, sidecar in zip(dataset.samples, dataset.sidecars):
        row = row_by_id[sample.sample_id]
        declared_target_kind = str(sample.metadata.get("footprint_kind"))
        declared_target_dimensions = sample.metadata.get(
            "footprint_dimensions_m"
        )
        declared_robot_kind = str(sample.metadata.get("robot_footprint_kind"))
        declared_robot_dimensions = sample.metadata.get(
            "robot_footprint_dimensions_m"
        )
        if not isinstance(declared_target_dimensions, Sequence) or isinstance(
            declared_target_dimensions, (str, bytes)
        ):
            raise ValueError("target footprint dimensions must be a sequence")
        if not isinstance(declared_robot_dimensions, Sequence) or isinstance(
            declared_robot_dimensions, (str, bytes)
        ):
            raise ValueError("robot footprint dimensions must be a sequence")
        if sample.metadata.get("footprint_contact_policy") != (
            TOY_FOOTPRINT_CONTACT_POLICY
        ):
            raise ValueError("toy footprint contact policy mismatch")
        for field in (
            "target_object_type",
            "footprint_kind",
            "footprint_dimensions_m",
            "robot_footprint_kind",
            "robot_footprint_dimensions_m",
            "footprint_contact_policy",
        ):
            if row[field] != sample.metadata[field]:
                raise ValueError(f"manifest row {field} does not match sample metadata")
        expected_labels = _footprint_labels(
            sidecar.hidden_risk_occupancy,
            sidecar.robot_future_footprints,
            resolution_m=dataset.grid.resolution_m,
            target_footprint_kind=declared_target_kind,
            target_footprint_dimensions_m=declared_target_dimensions,
            robot_footprint_kind=declared_robot_kind,
            robot_footprint_dimensions_m=declared_robot_dimensions,
        )
        observed_labels = (
            sample.collision_label,
            sample.risk_severity,
            sample.min_clearance,
            sample.near_miss,
            sample.first_collision_time,
        )
        if observed_labels != expected_labels:
            raise ValueError("toy labels do not match declared footprint geometry")

    return {
        **expected_components,
        "toy_dataset_manifest_digest": expected_manifest_digest,
    }


def make_toy_batch(dataset: ToyRiskDataset) -> dict[str, object]:
    """Collate one toy publication while preserving the input/label boundary."""

    if not isinstance(dataset, ToyRiskDataset):
        raise TypeError("dataset must be ToyRiskDataset")
    sidecars = dataset.sidecar_by_sample_id()
    ordered_sidecars = [sidecars[sample.sample_id] for sample in dataset.samples]
    return {
        "sample_ids": tuple(sample.sample_id for sample in dataset.samples),
        "model_inputs": {
            "bev_history": np.stack(
                [sample.bev_history for sample in dataset.samples]
            ).astype(np.float32, copy=False),
            "state_channels": np.stack(
                [sample.state_channels for sample in dataset.samples]
            ).astype(np.float32, copy=False),
            "trajectory_channels": np.stack(
                [sample.trajectory_channels for sample in dataset.samples]
            ).astype(np.float32, copy=False),
            "robot_state": np.stack(
                [sample.robot_state for sample in dataset.samples]
            ).astype(np.float32, copy=False),
        },
        "labels": {
            "collision_label": np.asarray(
                [sample.collision_label for sample in dataset.samples],
                dtype=np.float32,
            ),
            "risk_severity": np.asarray(
                [sample.risk_severity for sample in dataset.samples], dtype=np.float32
            ),
            "min_clearance": np.asarray(
                [sample.min_clearance for sample in dataset.samples], dtype=np.float32
            ),
            "near_miss": np.asarray(
                [sample.near_miss for sample in dataset.samples], dtype=np.float32
            ),
        },
        "label_sidecars": {
            "hidden_risk_occupancy": np.stack(
                [sidecar.hidden_risk_occupancy for sidecar in ordered_sidecars]
            ).astype(np.float32, copy=False),
            "robot_future_footprints": np.stack(
                [sidecar.robot_future_footprints for sidecar in ordered_sidecars]
            ).astype(np.float32, copy=False),
            "future_endpoint_times_s": TOY_FUTURE_ENDPOINT_TIMES_S.copy(),
            "sample_ids": tuple(sidecar.sample_id for sidecar in ordered_sidecars),
        },
        "manifest": dict(dataset.manifest),
        "manifest_rows": tuple(dict(row) for row in dataset.manifest_rows),
    }


def assert_toy_split_isolation(
    datasets: Iterable[ToyRiskDataset],
) -> dict[str, object]:
    """Require zero cross-split overlap for every frozen toy identity."""

    publications = tuple(datasets)
    if len({dataset.split for dataset in publications}) != len(publications):
        raise ValueError("each supplied toy publication must have a distinct split")
    identity_fields = (
        "recording_id",
        "session_id",
        "source_object_id",
        "snippet_id",
        "base_state_id",
        "pair_group_id",
        "seed_namespace",
    )
    overlap_counts: dict[str, int] = {}
    for field in identity_fields:
        split_sets = {
            dataset.split: {str(row[field]) for row in dataset.manifest_rows}
            for dataset in publications
        }
        overlap: set[str] = set()
        split_names = sorted(split_sets)
        for left_index, left in enumerate(split_names):
            for right in split_names[left_index + 1 :]:
                overlap.update(split_sets[left] & split_sets[right])
        overlap_counts[field] = len(overlap)
    if any(overlap_counts.values()):
        raise ValueError(f"toy split identity leakage detected: {overlap_counts}")
    return {"passed": True, "overlap_counts": overlap_counts}


__all__ = [
    "TOY_CASES",
    "TOY_DATASET_LAYOUT_VERSION",
    "TOY_DATASET_MANIFEST_KEYS",
    "TOY_DT_S",
    "TOY_EMPTY_CLEARANCE_SENTINEL_M",
    "TOY_FOOTPRINT_CONTACT_POLICY",
    "TOY_FOOTPRINT_PROXY_KIND",
    "TOY_FUTURE_ENDPOINT_TIMES_S",
    "TOY_FUTURE_STEPS",
    "TOY_HISTORY_STEPS",
    "TOY_MANIFEST_ROW_KEYS",
    "TOY_NEAR_MISS_THRESHOLD_M",
    "TOY_PAIRED_CASES",
    "ToyLabelSidecar",
    "ToyRiskDataset",
    "assert_toy_split_isolation",
    "frozen_channel_spec",
    "make_toy_batch",
    "make_toy_risk_dataset",
    "toy_dataset_manifest_digest",
    "toy_grid_manifest",
    "toy_label_sidecars_digest",
    "toy_label_digest",
    "toy_manifest_rows_digest",
    "toy_model_input_digest",
    "toy_ordered_sample_digest",
    "toy_ordered_sample_ids_digest",
    "toy_sample_id_sequence_digest",
    "validate_toy_risk_dataset_publication",
]
