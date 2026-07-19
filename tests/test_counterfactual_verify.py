from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from src.geometry import RectangleFootprint, rasterize_footprint
from src.generation.counterfactual_verify import (
    OBSERVATION_SIGNATURE_DIM,
    CounterfactualObservation,
    expected_verification_fov_mask,
    fit_signature_normalizer,
    make_observation_signature,
    simulate_counterfactual_observation,
)
from src.planning.verification_actions import (
    action_endpoint,
    load_verification_actions,
)
from tests.fixtures.verification_world import build_verification_toy_world


ROOT = Path(__file__).resolve().parents[1]


def _actions():
    return load_verification_actions(ROOT / "configs/verification_actions.yaml").by_id


def _observe(world, action_id: str):
    action = _actions()[action_id]
    return simulate_counterfactual_observation(
        post_action_pose=action_endpoint(np.zeros(3, dtype=np.float32), action),
        action_duration_s=action.duration_s,
        static_occupancy=world.static_occupancy,
        dynamic_current_poses=world.dynamic_current_poses,
        dynamic_future_poses=world.dynamic_future_poses,
        dynamic_specs=world.dynamic_specs,
        current_visible_mask=world.current_visible_mask,
        current_age_map=world.current_age_map,
        grid=world.grid,
        future_dt_s=0.2,
        age_max_s=5.0,
        fov_rad=np.deg2rad(20.0),
        max_range_m=4.0,
    )


def test_expected_fov_is_static_only_while_oracle_observation_changes():
    world = build_verification_toy_world()
    action = _actions()["yaw_left_20"]
    post_pose = action_endpoint(np.zeros(3, dtype=np.float32), action)

    expected_before = expected_verification_fov_mask(
        world.static_occupancy,
        world.grid,
        sensor_pose=post_pose,
        fov_rad=np.deg2rad(20.0),
        max_range_m=4.0,
    )
    moved_future = {
        key: value.copy() for key, value in world.dynamic_future_poses.items()
    }
    moved_future["critical_cart"][:, 1] = -2.5
    expected_after = expected_verification_fov_mask(
        world.static_occupancy,
        world.grid,
        sensor_pose=post_pose,
        fov_rad=np.deg2rad(20.0),
        max_range_m=4.0,
    )
    np.testing.assert_array_equal(expected_before, expected_after)
    assert expected_before.shape == (1, world.grid.height, world.grid.width)
    assert expected_before.dtype == np.float32

    observed = _observe(world, "yaw_left_20")
    moved = simulate_counterfactual_observation(
        post_action_pose=post_pose,
        action_duration_s=action.duration_s,
        static_occupancy=world.static_occupancy,
        dynamic_current_poses=world.dynamic_current_poses,
        dynamic_future_poses=moved_future,
        dynamic_specs=world.dynamic_specs,
        current_visible_mask=world.current_visible_mask,
        current_age_map=world.current_age_map,
        grid=world.grid,
        future_dt_s=0.2,
        age_max_s=5.0,
        fov_rad=np.deg2rad(20.0),
        max_range_m=4.0,
    )
    assert not np.array_equal(
        observed.visible_dynamic_occupancy, moved.visible_dynamic_occupancy
    )


def test_critical_and_irrelevant_actions_reveal_different_typed_objects():
    world = build_verification_toy_world()
    left = _observe(world, "yaw_left_20")
    right = _observe(world, "yaw_right_20")

    assert np.any(left.visible_dynamic_occupancy & world.critical_mask)
    assert not np.any(left.visible_dynamic_occupancy & world.irrelevant_mask)
    assert np.any(right.visible_dynamic_occupancy & world.irrelevant_mask)
    assert not np.any(right.visible_dynamic_occupancy & world.critical_mask)


def test_multi_occluder_raycast_hides_actor_behind_both_occluders():
    world = build_verification_toy_world()
    static = world.static_occupancy.copy()
    for x in (1.0, 1.8):
        np.maximum(
            static,
            rasterize_footprint(
                RectangleFootprint(0.20, 1.20),
                np.asarray([x, 0.0, 0.0], dtype=np.float32),
                world.grid,
            ),
            out=static,
        )
    current = {"critical_cart": np.asarray([2.5, 0.0, np.pi / 4], dtype=np.float32)}
    future = {
        "critical_cart": np.tile(current["critical_cart"], (15, 1)).astype(
            np.float32
        )
    }
    observed = simulate_counterfactual_observation(
        post_action_pose=np.zeros(3, dtype=np.float32),
        action_duration_s=0.5,
        static_occupancy=static,
        dynamic_current_poses=current,
        dynamic_future_poses=future,
        dynamic_specs={"critical_cart": world.dynamic_specs["critical_cart"]},
        current_visible_mask=np.zeros_like(static, dtype=bool),
        current_age_map=np.ones_like(static, dtype=np.float32),
        grid=world.grid,
        future_dt_s=0.2,
        age_max_s=5.0,
        fov_rad=np.deg2rad(90.0),
        max_range_m=4.0,
    )
    assert not observed.visible_dynamic_occupancy.any()


def test_seven_feature_signature_uses_only_observable_masks():
    world = build_verification_toy_world()
    shape = (world.grid.height, world.grid.width)
    visible = np.zeros(shape, dtype=bool)
    visible[10:12, 10:12] = True
    occupied = np.zeros(shape, dtype=bool)
    occupied[10, 10] = True
    age = np.ones(shape, dtype=np.float32)
    age[visible] = 0.0
    observation = CounterfactualObservation(
        visible_mask=visible,
        visible_occupied_mask=occupied,
        visible_dynamic_occupancy=occupied.copy(),
        newly_visible_mask=visible.copy(),
        updated_age_map=age,
    )
    original_swept = np.zeros(shape, dtype=bool)
    original_swept[10, 10:12] = True
    replan = np.zeros(shape, dtype=bool)
    replan[10, 10] = True
    critical = visible.copy()

    signature = make_observation_signature(
        observation,
        grid=world.grid,
        original_swept_mask=original_swept,
        replanned_swept_masks=(replan,),
        local_goal_corridor_mask=replan,
        critical_region_mask=critical,
        previous_age_map=np.ones(shape, dtype=np.float32),
    )
    np.testing.assert_allclose(
        signature,
        np.asarray([0.04, 0.02, 0.01, 1.0, 0.0, 1.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )
    assert signature.shape == (OBSERVATION_SIGNATURE_DIM,)
    assert signature.dtype == np.float32
    field_names = {field.name for field in fields(CounterfactualObservation)}
    assert field_names == {
        "visible_mask",
        "visible_occupied_mask",
        "visible_dynamic_occupancy",
        "newly_visible_mask",
        "updated_age_map",
    }
    assert not any(
        token in name
        for name in field_names
        for token in ("oracle", "object_id", "object_type", "footprint", "world")
    )


def test_signature_normalizer_fits_train_only_and_is_finite():
    signatures = np.asarray(
        [[0, 1, 2, 3, 4, 0, 1], [2, 3, 4, 5, 6, 1, 0]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="train"):
        fit_signature_normalizer(signatures, split="val")

    normalizer = fit_signature_normalizer(signatures, split="train")
    transformed = normalizer.transform(signatures)
    assert normalizer.fit_split == "train"
    assert transformed.dtype == np.float32
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-6)
