from __future__ import annotations

from dataclasses import replace

import numpy as np

import src.generation.sop05r_teb_event_sampler as sampler_module
from src.generation.continuous_collision import (
    ContinuousCollisionEvidence,
    compute_continuous_collision_evidence,
)
from src.geometry import CircleFootprint


def _late_swept_collision() -> ContinuousCollisionEvidence:
    robot_poses = np.zeros((40, 3), dtype=np.float64)
    target_poses = np.zeros((40, 3), dtype=np.float64)
    target_poses[:, 0] = 1.0
    target_poses[-1, 0] = 0.0
    evidence = compute_continuous_collision_evidence(
        robot_footprint=CircleFootprint(radius_m=0.1),
        robot_poses=robot_poses,
        target_footprint=CircleFootprint(radius_m=0.1),
        target_poses=target_poses,
        dt_s=0.2,
        spatial_resolution_m=0.05,
    )
    assert evidence.first_collision_time_s is not None
    assert 7.6 < evidence.first_collision_time_s < 7.8
    return evidence


def _evaluate_collision(monkeypatch, evidence: ContinuousCollisionEvidence):
    from tests.test_sop05r_teb_event_sampler import _mother_fixture

    inputs = _mother_fixture()
    monkeypatch.setattr(
        sampler_module,
        "compute_continuous_collision_evidence",
        lambda **_: evidence,
    )
    return sampler_module._collision_evidence(
        task_template=inputs[4],
        placement_result=inputs[5],
        source_base_state=inputs[1],
        snippet=inputs[6],
        base_config=inputs[0],
        teb_config=inputs[3],
        decision_time_s=0.0,
    )


def test_m6_accepts_continuous_collision_in_final_swept_interval(monkeypatch) -> None:
    collision, rejection = _evaluate_collision(monkeypatch, _late_swept_collision())

    assert rejection is None
    assert collision is not None
    assert 6.2 < collision.first_collision_time_after_decision_s < 6.4


def test_m6_rejects_exact_horizon_endpoint_only_collision(monkeypatch) -> None:
    evidence = _late_swept_collision()
    endpoint_only = replace(evidence, first_collision_time_s=7.8)

    collision, rejection = _evaluate_collision(monkeypatch, endpoint_only)

    assert collision is None
    assert rejection == "endpoint_only_collision"
