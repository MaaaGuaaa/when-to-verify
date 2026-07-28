from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.contracts import build_grid_spec
from src.generation.obstacle_first_templates import RectangleObstacle
from src.generation.sop05r_contracts import (
    SOP05R_PLANNER_VERSION,
    load_sop05r_config,
)
from src.generation.sop05r_trajectory_store import (
    Sop05rTrajectoryRecord,
    load_sop05r_trajectory_store,
    publish_sop05r_trajectory_store,
    validate_sop05r_trajectory_record,
)
from src.geometry import rasterize_footprint
from src.planning.obstacle_corner_planner import (
    ObstaclePlannerRequest,
    plan_obstacle_routes,
)
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def store_inputs():
    base_config = load_config(ROOT / "configs" / "base.yaml")
    sop05r_config = load_sop05r_config(
        ROOT / "configs" / "generator_obstacle_first_train.yaml"
    )
    grid = build_grid_spec(base_config)
    obstacle = RectangleObstacle(
        obstacle_id="store-obstacle",
        obstacle_type="wall",
        pose=np.asarray([2.0, 0.0, 0.0], dtype=np.float64),
        length_m=1.0,
        width_m=0.3,
        source="fixture",
    )
    static = rasterize_footprint(obstacle.footprint, obstacle.pose, grid).astype(
        np.float32
    )
    plan = plan_obstacle_routes(
        ObstaclePlannerRequest(
            start_pose=np.zeros(3, dtype=np.float32),
            initial_control=np.asarray([0.4, 0.0], dtype=np.float32),
            static_occupancy=static,
            obstacle=obstacle,
            local_goal_world_pose=np.asarray([4.0, 0.0, 0.0], dtype=np.float32),
            base_config=base_config,
            planner_config=sop05r_config.planner,
        )
    )
    record = Sop05rTrajectoryRecord(
        event_id="event-store-001",
        base_state_id="base-store-001",
        template_id="template-store-001",
        planner_version=SOP05R_PLANNER_VERSION,
        config_digest=sop05r_config.digest,
        shared_goal_world_pose=plan.shared_goal_world_pose,
        nominal_trajectory_id=plan.by_slot["left_near"].trajectory.trajectory_id,
        alternative_trajectory_ids=(
            plan.by_slot["right_near"].trajectory.trajectory_id,
            plan.by_slot["right_far"].trajectory.trajectory_id,
        ),
        routes=plan.routes,
    )
    return base_config, record


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _rewrite_checksums_and_marker(root: Path) -> None:
    checksum_path = root / "artifact_checksums.sha256"
    marker_path = root / ".sop05r-trajectories-complete"
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if not path.is_file() or relative in {
            "artifact_checksums.sha256",
            ".sop05r-trajectories-complete",
        }:
            continue
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n")
    checksum_path.write_text("".join(rows), encoding="utf-8")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifact_checksums_sha256"] = hashlib.sha256(
        checksum_path.read_bytes()
    ).hexdigest()
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def test_trajectory_store_round_trip_recomputes_query_maps(
    tmp_path: Path,
    store_inputs,
) -> None:
    base_config, record = store_inputs
    root = tmp_path / "store"

    publish_sop05r_trajectory_store(root, (record,), base_config=base_config)
    loaded = load_sop05r_trajectory_store(root)

    assert loaded.manifest["record_count"] == 1
    assert loaded.records[0].nominal_trajectory_id == record.nominal_trajectory_id
    assert loaded.records[0].candidate_trajectory_ids == record.candidate_trajectory_ids
    assert loaded.records[0].alternative_trajectory_ids == (
        record.alternative_trajectory_ids
    )
    for actual, expected in zip(
        loaded.records[0].routes, record.routes, strict=True
    ):
        np.testing.assert_array_equal(actual.trajectory.poses, expected.trajectory.poses)
        np.testing.assert_array_equal(actual.trajectory.controls, expected.trajectory.controls)
        np.testing.assert_array_equal(actual.trajectory.tta_map, expected.trajectory.tta_map)
        np.testing.assert_array_equal(
            actual.trajectory.braking_map, expected.trajectory.braking_map
        )
        np.testing.assert_array_equal(actual.poses_world, expected.poses_world)
        np.testing.assert_array_equal(actual.waypoints_world, expected.waypoints_world)


def test_store_rejects_invalid_nominal_or_alternative_membership(store_inputs) -> None:
    base_config, record = store_inputs

    with pytest.raises(ValueError, match="nominal"):
        validate_sop05r_trajectory_record(
            replace(record, nominal_trajectory_id="missing"),
            base_config=base_config,
        )
    with pytest.raises(ValueError, match="alternative"):
        validate_sop05r_trajectory_record(
            replace(record, alternative_trajectory_ids=(record.nominal_trajectory_id,)),
            base_config=base_config,
        )


def test_store_rejects_query_map_tampering(store_inputs) -> None:
    base_config, record = store_inputs
    route = record.routes[0]
    tampered_tta = route.trajectory.tta_map.copy()
    row, column = np.argwhere(route.trajectory.swept_mask != 0)[-1]
    tampered_tta[row, column] += np.float32(0.2)
    tampered_route = replace(
        route,
        trajectory=replace(route.trajectory, tta_map=tampered_tta),
    )

    with pytest.raises(ValueError, match="query map"):
        validate_sop05r_trajectory_record(
            replace(record, routes=(tampered_route, *record.routes[1:])),
            base_config=base_config,
        )


def test_store_detects_checksum_tampering(
    tmp_path: Path,
    store_inputs,
) -> None:
    base_config, record = store_inputs
    root = tmp_path / "tampered"
    publish_sop05r_trajectory_store(root, (record,), base_config=base_config)
    array_path = next((root / "arrays").glob("*.npz"))
    payload = bytearray(array_path.read_bytes())
    payload[-1] ^= 1
    array_path.write_bytes(payload)

    with pytest.raises(ValueError, match="checksum"):
        load_sop05r_trajectory_store(root)


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_store_rejects_extra_or_missing_npz_arrays(
    tmp_path: Path,
    store_inputs,
    mutation: str,
) -> None:
    base_config, record = store_inputs
    root = tmp_path / mutation
    publish_sop05r_trajectory_store(root, (record,), base_config=base_config)
    array_path = next((root / "arrays").glob("*.npz"))
    with np.load(array_path, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    if mutation == "extra":
        arrays["unexpected"] = np.zeros(1, dtype=np.float32)
    else:
        del arrays[sorted(arrays)[0]]
    with array_path.open("wb") as handle:
        np.savez(handle, **arrays)
    _rewrite_checksums_and_marker(root)

    with pytest.raises(ValueError, match="array keys"):
        load_sop05r_trajectory_store(root)


def test_store_publication_is_deterministic_and_refuses_overwrite(
    tmp_path: Path,
    store_inputs,
) -> None:
    base_config, record = store_inputs
    first = tmp_path / "first"
    second = tmp_path / "second"

    publish_sop05r_trajectory_store(first, (record,), base_config=base_config)
    publish_sop05r_trajectory_store(second, (record,), base_config=base_config)

    assert _artifact_bytes(first) == _artifact_bytes(second)
    with pytest.raises(FileExistsError):
        publish_sop05r_trajectory_store(first, (record,), base_config=base_config)
