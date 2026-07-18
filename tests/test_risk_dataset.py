"""Behavioral tests for schema-v3 RiskSample assembly and isolation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pytest

from src.contracts import (
    SCHEMA_VERSION,
    TRAJECTORY_CHANNELS,
    BaseState,
    LocalTrajectory,
    OracleWorld,
    RiskSample,
    build_grid_spec,
)
from src.datasets.risk_dataset import (
    RiskBuildInput,
    build_risk_sample,
    build_trajectory_channels,
    validate_risk_sample_for_publication,
)
from src.generation.structural_blindspot import StructuralBlindSpot


TARGET_ID = "generated::human::hidden-target"
CONTEXT_ID = "recording::visible-context"


def _config() -> dict[str, object]:
    return {
        "schema_version": "3.0.0",
        "bev": {
            "range_m": 9.0,
            "resolution_m": 1.0,
            "size": 9,
            "history_steps": 8,
            "history_dt_s": 0.2,
            "future_steps": 15,
            "future_dt_s": 0.2,
        },
        "robot": {
            "model": "differential_drive",
            "length_m": 0.70,
            "width_m": 0.55,
            "inflation_m": 0.15,
            "max_linear_speed_mps": 0.9,
            "max_angular_speed_radps": 0.8,
        },
        "age_map": {
            "a_max_s": 5.0,
            "never_seen_value": 1.0,
            "visible_value": 0.0,
        },
    }


def _risk_config() -> dict[str, float]:
    return {
        "sigma_distance_m": 0.5,
        "sigma_time_s": 2.0,
        "near_miss_distance_m": 0.35,
    }


def _circle_spec(radius_m: float = 0.2) -> dict[str, object]:
    return {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": radius_m},
    }


def _constant_motion(x: float, y: float = 0.0) -> np.ndarray:
    poses = np.empty((15, 3), dtype=np.float32)
    poses[:] = np.asarray([x, y, 0.0], dtype=np.float32)
    return poses


def _base_state(static: np.ndarray) -> BaseState:
    context_history = np.empty((8, 3), dtype=np.float32)
    context_history[:] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    return BaseState(
        state_id="base-risk-dataset-toy",
        split="train",
        recording_id="recording-risk-dataset-toy",
        dynamic_object_ids=(CONTEXT_ID,),
        timestamp=1.4,
        robot_history=np.zeros((8, 3), dtype=np.float32),
        robot_state=np.asarray([0.4, 0.0], dtype=np.float32),
        visible_dynamic_object_history={CONTEXT_ID: context_history},
        visible_dynamic_object_specs={CONTEXT_ID: _circle_spec()},
        static_map_local=static.copy(),
        metadata={"coordinate_frame": "robot_current_local"},
    )


def _trajectory() -> LocalTrajectory:
    grid = build_grid_spec(_config())
    shape = (grid.height, grid.width)
    return LocalTrajectory(
        trajectory_id="trajectory-risk-dataset-toy",
        poses=np.zeros((grid.future_steps, 3), dtype=np.float32),
        controls=np.zeros((grid.future_steps, 2), dtype=np.float32),
        swept_mask=np.full(shape, 1.0, dtype=np.float32),
        tta_map=np.full(shape, 2.0, dtype=np.float32),
        braking_map=np.full(shape, 3.0, dtype=np.float32),
        centerline_map=np.full(shape, 4.0, dtype=np.float32),
        task_cost=0.0,
        metadata={"pose_time_layout_version": "future_endpoints_dt_to_horizon_v1"},
    )


def _sensor() -> StructuralBlindSpot:
    return StructuralBlindSpot(
        forward_fov_deg=360.0,
        range_m=9.0,
        blind_sectors=({"center_deg": 90.0, "width_deg": 90.0},),
    )


def _source(
    *,
    target_future: np.ndarray | None = None,
    event_type: str = "collision",
) -> RiskBuildInput:
    config = _config()
    grid = build_grid_spec(config)
    static = np.zeros((grid.height, grid.width), dtype=np.float32)
    base = _base_state(static)
    context_history = base.visible_dynamic_object_history[CONTEXT_ID].copy()
    scene_history = {CONTEXT_ID: context_history}
    scene_specs = {CONTEXT_ID: _circle_spec()}
    trajectories = {CONTEXT_ID: _constant_motion(0.0)}
    specs = {CONTEXT_ID: _circle_spec()}
    hidden_ids: tuple[str, ...] = ()
    if target_future is not None:
        target_history = np.empty((grid.history_steps, 3), dtype=np.float32)
        target_history[:] = np.asarray([0.0, 2.0, 0.0], dtype=np.float32)
        scene_history[TARGET_ID] = target_history
        scene_specs[TARGET_ID] = _circle_spec()
        trajectories[TARGET_ID] = target_future
        specs[TARGET_ID] = _circle_spec()
        hidden_ids = (TARGET_ID,)
    world = OracleWorld(
        world_id=f"label-source-{event_type}",
        base_state_id=base.state_id,
        static_occupancy=static.copy(),
        dynamic_object_trajectories=trajectories,
        dynamic_object_specs=specs,
        occluders=(),
        blind_spot_config={"kind": "structural"},
        random_seed=23,
        metadata={"schema_version": SCHEMA_VERSION},
    )
    return RiskBuildInput(
        sample_id=f"risk-sample-{event_type}",
        pair_group_id="pair-risk-dataset-toy",
        event_type=event_type,
        base_state=base,
        trajectory=_trajectory(),
        oracle_world=world,
        observed_static_occupancy=static,
        scene_dynamic_history=scene_history,
        scene_dynamic_specs=scene_specs,
        hidden_object_ids=hidden_ids,
        sensor_config=_sensor(),
        provenance={
            "source_recording_id": base.recording_id,
            "session_id": "session-risk-toy",
            "dynamic_object_snippet_id": "snippet-risk-toy",
            "seed_namespace": "split/train/risk-dataset",
        },
    )


def _build(source: RiskBuildInput) -> RiskSample:
    return build_risk_sample(
        source,
        base_config=_config(),
        risk_config=_risk_config(),
    )


def _metadata_has_forbidden_payload(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(
                token in lowered
                for token in (
                    "future",
                    "oracle",
                    "clearance_sequence",
                    "dynamic_object_trajectories",
                    "hidden_object_ids",
                )
            ):
                return True
            if _metadata_has_forbidden_payload(child):
                return True
    if isinstance(value, (list, tuple)):
        return any(_metadata_has_forbidden_payload(child) for child in value)
    return False


def test_trajectory_channels_use_frozen_order_shape_dtype_and_owned_copy() -> None:
    trajectory = _trajectory()
    grid = build_grid_spec(_config())

    channels = build_trajectory_channels(trajectory, grid)

    assert TRAJECTORY_CHANNELS == (
        "swept_volume_mask",
        "time_to_arrival_map",
        "braking_margin_map",
        "centerline_map",
    )
    assert channels.shape == (4, grid.height, grid.width)
    assert channels.dtype == np.float32
    assert channels.flags.c_contiguous
    for index, expected in enumerate((1.0, 2.0, 3.0, 4.0)):
        np.testing.assert_array_equal(channels[index], expected)
    assert not np.shares_memory(channels, trajectory.swept_mask)


@pytest.mark.parametrize("defect", ["shape", "dtype", "nonfinite"])
def test_trajectory_channels_reject_invalid_query_maps(defect: str) -> None:
    trajectory = _trajectory()
    if defect == "shape":
        bad = np.zeros((8, 9), dtype=np.float32)
    elif defect == "dtype":
        bad = trajectory.tta_map.astype(np.float64)
    else:
        bad = trajectory.tta_map.copy()
        bad[0, 0] = np.nan
    trajectory = replace(trajectory, tta_map=bad)

    with pytest.raises((TypeError, ValueError), match="time_to_arrival_map"):
        build_trajectory_channels(trajectory, build_grid_spec(_config()))


def test_changing_only_hidden_future_changes_labels_not_observation_arrays() -> None:
    collision_source = _source(target_future=_constant_motion(0.0))
    safe_world = replace(
        collision_source.oracle_world,
        dynamic_object_trajectories={
            CONTEXT_ID: _constant_motion(0.0),
            TARGET_ID: _constant_motion(3.0),
        },
    )
    safe_source = replace(collision_source, oracle_world=safe_world)

    collision = _build(collision_source)
    safe = _build(safe_source)

    assert collision.collision_label == 1
    assert collision.risk_severity == 1.0
    assert collision.first_collision_time == pytest.approx(0.2)
    assert safe.collision_label == 0
    assert safe.near_miss == 0
    for field in (
        "bev_history",
        "state_channels",
        "trajectory_channels",
        "robot_state",
    ):
        np.testing.assert_array_equal(getattr(collision, field), getattr(safe, field))


def test_label_audit_is_the_only_location_for_critical_object_metadata() -> None:
    sample = _build(_source(target_future=_constant_motion(0.0)))

    assert sample.metadata["schema_version"] == "3.0.0"
    assert set(sample.metadata) == {
        "schema_version",
        "renderer",
        "trajectory_id",
        "provenance",
        "label_audit",
    }
    audit = sample.metadata["label_audit"]
    assert audit["critical_object_id"] == TARGET_ID
    assert audit["critical_object_type"] == "human"
    assert audit["time_to_min_clearance_s"] == pytest.approx(0.2)
    assert audit["has_hidden_target"] is True
    assert "critical_object_id" not in {
        key for key in sample.metadata if key != "label_audit"
    }
    assert not _metadata_has_forbidden_payload(sample.metadata)
    validate_risk_sample_for_publication(sample, build_grid_spec(_config()))


def test_empty_hidden_set_ignores_colliding_visible_context_and_uses_sentinel() -> None:
    sample = _build(_source(target_future=None, event_type="empty_blind_spot"))
    grid = build_grid_spec(_config())

    assert sample.event_type == "empty_blind_spot"
    assert sample.collision_label == 0
    assert sample.near_miss == 0
    assert sample.risk_severity == 0.0
    assert sample.min_clearance == pytest.approx(
        math.hypot(grid.width * grid.resolution_m, grid.height * grid.resolution_m)
    )
    assert sample.first_collision_time is None
    assert sample.metadata["label_audit"] == {
        "risk_gt_version": "hidden_risk_gt_schema3_v1",
        "pose_time_layout_version": "future_endpoints_dt_to_horizon_v1",
        "critical_object_id": None,
        "critical_object_type": None,
        "time_to_min_clearance_s": None,
        "has_hidden_target": False,
    }


def test_irrelevant_hidden_target_keeps_real_safe_label_and_excludes_context() -> None:
    sample = _build(
        _source(
            target_future=_constant_motion(3.0),
            event_type="irrelevant_hidden",
        )
    )

    assert sample.collision_label == 0
    assert sample.near_miss == 0
    assert sample.min_clearance > 0.35
    assert 0.0 < sample.risk_severity < 1.0
    assert sample.metadata["label_audit"]["critical_object_id"] == TARGET_ID
    assert sample.metadata["label_audit"]["critical_object_id"] != CONTEXT_ID


def test_visible_context_cannot_be_redeclared_as_hidden() -> None:
    source = _source(target_future=_constant_motion(3.0))
    source = replace(source, hidden_object_ids=(CONTEXT_ID,))

    with pytest.raises(ValueError, match="currently visible"):
        _build(source)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda metadata: metadata["provenance"].update(
                {"hidden_future_poses": [[0.0, 0.0, 0.0]]}
            ),
            "forbidden",
        ),
        (
            lambda metadata: metadata["provenance"].update(
                {"oracle_payload": {"x": 1.0}}
            ),
            "forbidden",
        ),
        (
            lambda metadata: metadata["provenance"].update(
                {"debug_array": np.zeros(1, dtype=np.float32)}
            ),
            "ndarray",
        ),
        (
            lambda metadata: metadata["provenance"].update(
                {"debug_score": float("nan")}
            ),
            "finite",
        ),
        (
            lambda metadata: metadata.update(
                {"critical_object_id": TARGET_ID}
            ),
            "metadata keys",
        ),
    ],
)
def test_publication_validator_rejects_metadata_leakage(
    mutation,
    message: str,
) -> None:
    sample = _build(_source(target_future=_constant_motion(0.0)))
    metadata = deepcopy(sample.metadata)
    mutation(metadata)

    with pytest.raises((TypeError, ValueError), match=message):
        validate_risk_sample_for_publication(
            replace(sample, metadata=metadata),
            build_grid_spec(_config()),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"min_clearance": float("inf")}, "min_clearance"),
        ({"risk_severity": float("nan")}, "risk_severity"),
        ({"first_collision_time": None}, "first_collision_time"),
    ],
)
def test_publication_validator_rejects_nonfinite_or_inconsistent_labels(
    changes: dict[str, object],
    message: str,
) -> None:
    sample = _build(_source(target_future=_constant_motion(0.0)))

    with pytest.raises(ValueError, match=message):
        validate_risk_sample_for_publication(
            replace(sample, **changes),
            build_grid_spec(_config()),
        )
