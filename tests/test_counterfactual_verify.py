from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from src.geometry import RectangleFootprint, rasterize_footprint
from src.generation.counterfactual_verify import (
    OBSERVATION_SIGNATURE_DIM,
    CounterfactualObservation,
    CounterfactualObservationTrace,
    SignatureNormalizer,
    expected_verification_fov_mask,
    expected_verification_fov_trace_mask,
    fit_signature_normalizer,
    make_observation_signature,
    simulate_counterfactual_observation,
    simulate_counterfactual_observation_trace,
)
from src.planning.verification_actions import (
    ActionTrace,
    VerificationAction,
    action_endpoint,
    load_verification_actions,
    sample_action_trace,
    sample_state_aware_action_trace,
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
        fov_rad=2.0 * np.pi,
        max_range_m=4.0,
    )


def test_expected_fov_is_static_only_while_oracle_observation_changes():
    world = build_verification_toy_world()
    action = _actions()["arc_left_30"]
    post_pose = action_endpoint(np.zeros(3, dtype=np.float32), action)

    expected_before = expected_verification_fov_mask(
        world.static_occupancy,
        world.grid,
        sensor_pose=post_pose,
        fov_rad=2.0 * np.pi,
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
        fov_rad=2.0 * np.pi,
        max_range_m=4.0,
    )
    np.testing.assert_array_equal(expected_before, expected_after)
    assert expected_before.shape == (1, world.grid.height, world.grid.width)
    assert expected_before.dtype == np.float32

    observed = _observe(world, "arc_left_30")
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
        fov_rad=2.0 * np.pi,
        max_range_m=4.0,
    )
    assert not np.array_equal(
        observed.visible_dynamic_occupancy, moved.visible_dynamic_occupancy
    )


def test_expected_fov_trace_is_the_static_per_frame_visibility_union():
    world = build_verification_toy_world()
    action = _actions()["arc_left_30"]
    trace = sample_state_aware_action_trace(
        np.zeros(3, dtype=np.float32),
        action,
        robot_state=np.zeros(2, dtype=np.float32),
        braking_deceleration_mps2=1.0,
    )

    traced = expected_verification_fov_trace_mask(
        world.static_occupancy,
        world.grid,
        action_trace=trace,
        fov_rad=2.0 * np.pi,
        max_range_m=4.0,
    )
    per_frame = tuple(
        expected_verification_fov_mask(
            world.static_occupancy,
            world.grid,
            sensor_pose=pose,
            fov_rad=2.0 * np.pi,
            max_range_m=4.0,
        )
        for pose in trace.poses
    )
    expected = np.maximum.reduce(per_frame)

    np.testing.assert_array_equal(traced, expected)
    assert traced.dtype == np.float32
    assert np.count_nonzero(traced) > np.count_nonzero(per_frame[-1])


def test_left_arc_reveals_right_front_occlusion_under_360_fov():
    world = build_verification_toy_world()
    left = _observe(world, "arc_left_30")
    right = _observe(world, "arc_right_30")
    forward = _observe(world, "forward_peek")

    assert not np.any(world.current_visible_mask & world.critical_mask)
    assert np.any(left.visible_dynamic_occupancy & world.critical_mask)
    assert not np.any(right.visible_dynamic_occupancy & world.critical_mask)
    assert not np.any(forward.visible_dynamic_occupancy & world.critical_mask)
    assert np.any(left.visible_dynamic_occupancy & world.irrelevant_mask)
    assert np.any(right.visible_dynamic_occupancy & world.irrelevant_mask)


def test_yaw_only_trace_has_no_visibility_gain_under_360_fov():
    world = build_verification_toy_world()
    yaw_only = sample_action_trace(
        np.zeros(3, dtype=np.float32),
        VerificationAction(
            action_id="yaw_only_control",
            duration_s=0.6,
            delta_forward_m=0.0,
            delta_yaw_rad=float(np.deg2rad(90.0)),
        ),
    )
    stationary = sample_action_trace(
        np.zeros(3, dtype=np.float32),
        VerificationAction(
            action_id="stationary_control",
            duration_s=0.6,
            delta_forward_m=0.0,
            delta_yaw_rad=0.0,
        ),
    )

    yaw_mask = expected_verification_fov_trace_mask(
        world.static_occupancy,
        world.grid,
        action_trace=yaw_only,
        fov_rad=2.0 * np.pi,
        max_range_m=4.0,
    )
    stationary_mask = expected_verification_fov_trace_mask(
        world.static_occupancy,
        world.grid,
        action_trace=stationary,
        fov_rad=2.0 * np.pi,
        max_range_m=4.0,
    )

    np.testing.assert_array_equal(yaw_mask, stationary_mask)


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
        "critical_cart": np.tile(
            current["critical_cart"], (world.grid.future_steps, 1)
        ).astype(
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


def test_trace_observation_rejects_bad_dtypes_and_future_horizon():
    world = build_verification_toy_world()
    action = _actions()["stop_scan"]
    trace = sample_state_aware_action_trace(
        np.zeros(3, dtype=np.float32),
        action,
        robot_state=np.zeros(2, dtype=np.float32),
        braking_deceleration_mps2=1.0,
    )
    common = {
        "action_trace": trace,
        "static_occupancy": world.static_occupancy,
        "dynamic_current_poses": world.dynamic_current_poses,
        "dynamic_future_poses": world.dynamic_future_poses,
        "dynamic_specs": world.dynamic_specs,
        "current_visible_mask": world.current_visible_mask,
        "current_age_map": world.current_age_map,
        "grid": world.grid,
        "future_dt_s": 0.2,
        "age_max_s": 5.0,
        "fov_rad": 2.0 * np.pi,
        "max_range_m": 4.0,
    }

    with pytest.raises(TypeError, match="current_visible_mask dtype"):
        simulate_counterfactual_observation_trace(
            **{
                **common,
                "current_visible_mask": world.current_visible_mask.astype(
                    np.float32
                ),
            }
        )

    long_trace = ActionTrace(
        poses=np.zeros((2, 3), dtype=np.float32),
        times_s=np.asarray([0.0, 6.6], dtype=np.float64),
        linear_velocities_mps=np.zeros(2, dtype=np.float64),
        angular_velocities_radps=np.zeros(2, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="future horizon"):
        simulate_counterfactual_observation_trace(
            **{**common, "action_trace": long_trace}
        )

    trace_fields = {field.name for field in fields(CounterfactualObservationTrace)}
    assert trace_fields == {"aggregate", "frames", "times_s"}


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


def test_signature_excludes_dynamic_cells_that_were_already_visible():
    world = build_verification_toy_world()
    shape = (world.grid.height, world.grid.width)
    visible = np.zeros(shape, dtype=bool)
    visible[10, 10] = True
    visible[20, 20] = True
    already_visible_dynamic = np.zeros(shape, dtype=bool)
    already_visible_dynamic[10, 10] = True
    newly_visible = np.zeros(shape, dtype=bool)
    newly_visible[20, 20] = True
    observation = CounterfactualObservation(
        visible_mask=visible,
        visible_occupied_mask=already_visible_dynamic,
        visible_dynamic_occupancy=already_visible_dynamic.copy(),
        newly_visible_mask=newly_visible,
        updated_age_map=np.zeros(shape, dtype=np.float32),
    )
    corridor = np.zeros(shape, dtype=bool)
    corridor[10, 10] = True
    empty = np.zeros(shape, dtype=bool)

    signature = make_observation_signature(
        observation,
        grid=world.grid,
        original_swept_mask=empty,
        replanned_swept_masks=(),
        local_goal_corridor_mask=corridor,
        critical_region_mask=empty,
        previous_age_map=np.zeros(shape, dtype=np.float32),
    )

    assert signature[3] == 0.0
    assert signature[4] == pytest.approx(
        np.hypot(
            world.grid.height * world.grid.resolution_m,
            world.grid.width * world.grid.resolution_m,
        )
    )
    assert signature[5] == 0.0


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
    assert normalizer.sample_count == signatures.shape[0]
    assert len(normalizer.statistics_digest) == 64
    restored = SignatureNormalizer.from_dict(normalizer.as_dict())
    assert restored.statistics_digest == normalizer.statistics_digest
    np.testing.assert_array_equal(restored.mean, normalizer.mean)
    np.testing.assert_array_equal(restored.scale, normalizer.scale)
    assert transformed.dtype == np.float32
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-6)

    tampered = normalizer.as_dict()
    tampered["mean"][0] += 1.0
    with pytest.raises(ValueError, match="digest"):
        SignatureNormalizer.from_dict(tampered)
