import json
from pathlib import Path

import numpy as np
import pytest

import src.generation.sop05r_teb_trajectory_store as trajectory_store_module

from src.contracts import LocalTrajectory
from src.generation.sop05r_contracts import SOP05R_TEB_PLANNER_VERSION
from src.generation.sop05r_teb_event_sampler import Sop05rTebTrajectoryRecord
from src.generation.sop05r_teb_trajectory_store import (
    load_sop05r_teb_trajectory_store,
    publish_sop05r_teb_trajectory_store,
)
from src.planning.lightweight_teb import PlannedTebRoute


def _record(event_id: str = "event-001") -> Sop05rTebTrajectoryRecord:
    sample_times = np.arange(1, 41, dtype=np.float32) * np.float32(0.2)
    sampled_poses = np.zeros((40, 3), dtype=np.float32)
    sampled_poses[:, 0] = sample_times / np.float32(8.0) * np.float32(5.0)
    route = PlannedTebRoute(
        planner_version=SOP05R_TEB_PLANNER_VERSION,
        goal_world_pose=np.asarray([5.0, 0.0, 0.0], dtype=np.float32),
        band_poses_world=np.linspace(
            np.zeros(3, dtype=np.float32),
            np.asarray([5.0, 0.0, 0.0], dtype=np.float32),
            21,
            dtype=np.float32,
        ),
        band_interval_dt_s=np.full(20, 0.4, dtype=np.float32),
        sample_times_s=sample_times,
        sampled_poses_world=sampled_poses,
        sampled_controls=np.tile(
            np.asarray([0.625, 0.0], dtype=np.float32), (40, 1)
        ),
        goal_arrival_time_s=5.0,
        task_cost=1.0,
    )
    trajectory = LocalTrajectory(
        trajectory_id=f"{event_id}-nominal",
        poses=sampled_poses[:32],
        controls=route.sampled_controls[:32],
        swept_mask=np.zeros((4, 4), dtype=np.float32),
        tta_map=np.full((4, 4), -1.0, dtype=np.float32),
        braking_map=np.zeros((4, 4), dtype=np.float32),
        centerline_map=np.zeros((4, 4), dtype=np.float32),
        task_cost=1.0,
        metadata={"planner_version": SOP05R_TEB_PLANNER_VERSION},
    )
    return Sop05rTebTrajectoryRecord(
        event_id=event_id,
        source_base_state_id="source-state",
        decision_state_id="decision-state",
        template_id="template-001",
        planner_version=SOP05R_TEB_PLANNER_VERSION,
        config_digest="a" * 64,
        shared_goal_world_pose=np.asarray([5.0, 0.0, 0.0], dtype=np.float32),
        full_route=route,
        nominal_trajectory=trajectory,
    )


def test_store_round_trips_distinct_band_route_and_suffix_arrays(
    tmp_path: Path,
) -> None:
    output = tmp_path / "trajectory_store"
    published = publish_sop05r_teb_trajectory_store(
        (_record(),),
        output,
        requested_count=1,
        complete=True,
    )

    loaded = load_sop05r_teb_trajectory_store(output, require_complete=True)

    assert loaded.collection_semantic_digest == published.collection_semantic_digest
    assert loaded.complete
    assert len(loaded.records) == 1
    record = loaded.records[0]
    assert record.full_route.band_poses_world.shape == (21, 3)
    assert record.full_route.sampled_poses_world.shape == (40, 3)
    assert record.nominal_trajectory.poses.shape == (32, 3)
    assert (output / "COMPLETE.json").is_file()


def test_store_rejects_tampered_array_payload(tmp_path: Path) -> None:
    output = tmp_path / "trajectory_store"
    publish_sop05r_teb_trajectory_store(
        (_record(),),
        output,
        requested_count=2,
        complete=False,
    )
    payload = output / "trajectories.npz"
    payload.write_bytes(payload.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="checksum"):
        load_sop05r_teb_trajectory_store(output)

    assert not (output / "COMPLETE.json").exists()


def test_selected_reader_loads_only_requested_array_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "trajectory_store"
    publish_sop05r_teb_trajectory_store(
        (_record("event-001"), _record("event-002")),
        output,
        requested_count=2,
        complete=True,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="ascii"))
    rows = tuple(manifest["records"])
    selected = next(row for row in rows if row["event_id"] == "event-001")
    unselected = next(row for row in rows if row["event_id"] == "event-002")
    selected_keys = {item["key"] for item in selected["arrays"].values()}
    unselected_keys = {item["key"] for item in unselected["arrays"].values()}
    real_load = trajectory_store_module.np.load
    accessed: list[str] = []

    class TrackingArchive:
        def __init__(self, path: Path, *, allow_pickle: bool):
            self._archive = real_load(path, allow_pickle=allow_pickle)
            self.files = self._archive.files

        def __getitem__(self, key: str):
            accessed.append(key)
            return self._archive[key]

        def close(self) -> None:
            self._archive.close()

    monkeypatch.setattr(
        trajectory_store_module.np,
        "load",
        lambda path, *, allow_pickle: TrackingArchive(
            path,
            allow_pickle=allow_pickle,
        ),
    )

    reader_factory = getattr(
        trajectory_store_module,
        "open_sop05r_teb_trajectory_selection",
        None,
    )
    assert callable(reader_factory), "selected trajectory reader is missing"
    reader = reader_factory(output, rows=rows)
    loaded = reader.load_records(("event-001",))
    reader.close()

    assert tuple(record.event_id for record in loaded) == ("event-001",)
    assert set(accessed) == selected_keys
    assert not set(accessed).intersection(unselected_keys)


def test_selected_reader_indexes_archive_members_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "trajectory_store"
    publish_sop05r_teb_trajectory_store(
        (_record("event-001"), _record("event-002")),
        output,
        requested_count=2,
        complete=True,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="ascii"))
    rows = tuple(manifest["records"])
    real_load = trajectory_store_module.np.load
    member_indexes = []

    class MemberNames:
        def __init__(self, values) -> None:
            self._values = tuple(values)
            self.iteration_count = 0

        def __iter__(self):
            self.iteration_count += 1
            return iter(self._values)

        def __contains__(self, key: object) -> bool:
            raise AssertionError("archive member lookup scanned the full name list")

    def indexed_archive(path: Path, *, allow_pickle: bool):
        archive = real_load(path, allow_pickle=allow_pickle)
        archive.files = MemberNames(archive.files)
        archive._files = MemberNames(archive._files)
        member_indexes.extend((archive.files, archive._files))
        return archive

    monkeypatch.setattr(
        trajectory_store_module.np,
        "load",
        indexed_archive,
    )

    reader = trajectory_store_module.open_sop05r_teb_trajectory_selection(
        output,
        rows=rows,
    )
    loaded = reader.load_records(("event-001",))
    reader.close()

    assert tuple(record.event_id for record in loaded) == ("event-001",)
    assert len(member_indexes) == 2
    assert member_indexes[0].iteration_count == 1
    assert member_indexes[1].iteration_count == 0
