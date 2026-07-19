"""Tests for the frozen empty-scene observation-renderer contract."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.contracts import (  # noqa: E402
    HISTORY_CHANNELS,
    SCHEMA_VERSION,
    STATE_CHANNELS,
    BaseState,
    build_grid_spec,
)
from src.generation.observation_renderer import (  # noqa: E402
    RENDERER_LAYOUT_VERSION,
    RenderedObservation,
    render_observation,
)
from src.generation import (  # noqa: E402
    observation_renderer as observation_renderer_module,
)
from src.generation.structural_blindspot import StructuralBlindSpot  # noqa: E402
from src.geometry import (  # noqa: E402
    CircleFootprint,
    RectangleFootprint,
    rasterize_footprint,
    world_to_grid,
)


def _toy_config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "bev": {
            "range_m": 9.0,
            "resolution_m": 1.0,
            "size": 9,
            "history_steps": 3,
            "history_dt_s": 0.2,
            "future_steps": 2,
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


def _empty_base_state(static_occupancy: np.ndarray) -> BaseState:
    return BaseState(
        state_id="toy-empty-base",
        split="toy",
        recording_id="toy-recording",
        dynamic_object_ids=(),
        timestamp=0.4,
        robot_history=np.zeros((3, 3), dtype=np.float32),
        robot_state=np.array([0.375, -0.25], dtype=np.float32),
        visible_dynamic_object_history={},
        visible_dynamic_object_specs={},
        static_map_local=static_occupancy.copy(),
        metadata={"coordinate_frame": "robot current pose"},
    )


def _render_empty(
    config: dict,
    base_state: BaseState,
    static_occupancy: np.ndarray,
    *,
    sensor_config: StructuralBlindSpot | None = None,
) -> RenderedObservation:
    return render_observation(
        base_state,
        scene_dynamic_history={},
        scene_dynamic_specs={},
        static_occupancy=static_occupancy,
        sensor_config=sensor_config,
        config=config,
    )


def test_empty_scene_renders_frozen_history_and_state_layout() -> None:
    config = _toy_config()
    grid = build_grid_spec(config)
    static_occupancy = np.zeros((9, 9), dtype=np.float32)
    base_state = _empty_base_state(static_occupancy)

    rendered = render_observation(
        base_state=base_state,
        scene_dynamic_history={},
        scene_dynamic_specs={},
        static_occupancy=static_occupancy,
        sensor_config=None,
        config=config,
    )

    assert isinstance(rendered, RenderedObservation)
    for array, expected_shape in (
        (rendered.bev_history, (3, len(HISTORY_CHANNELS), 9, 9)),
        (rendered.state_channels, (len(STATE_CHANNELS), 9, 9)),
    ):
        assert array.shape == expected_shape
        assert array.dtype == np.float32
        assert array.flags.c_contiguous
        assert array.flags.owndata
        assert np.isfinite(array).all()

    history_dynamic = rendered.bev_history[
        :, HISTORY_CHANNELS.index("past_dynamic_occupancy")
    ]
    history_visible = rendered.bev_history[
        :, HISTORY_CHANNELS.index("past_visible_mask")
    ]
    current_visible = history_visible[-1]
    current_free = rendered.state_channels[
        STATE_CHANNELS.index("current_visible_free")
    ]
    current_occupied = rendered.state_channels[
        STATE_CHANNELS.index("current_visible_occupied")
    ]
    current_unknown = rendered.state_channels[
        STATE_CHANNELS.index("current_unobservable_mask")
    ]

    assert np.array_equal(history_dynamic, np.zeros_like(history_dynamic))
    assert np.array_equal(history_visible, np.ones_like(history_visible))
    assert np.array_equal(current_free + current_occupied, current_visible)
    assert np.array_equal(
        current_free + current_occupied + current_unknown,
        np.ones((9, 9), dtype=np.float32),
    )
    assert not np.logical_and(current_free != 0.0, current_occupied != 0.0).any()

    assert np.array_equal(
        rendered.state_channels[STATE_CHANNELS.index("last_seen_occupancy")],
        np.zeros((9, 9), dtype=np.float32),
    )
    assert np.array_equal(
        rendered.state_channels[STATE_CHANNELS.index("occlusion_age_map")],
        np.zeros((9, 9), dtype=np.float32),
    )
    assert np.array_equal(
        rendered.state_channels[STATE_CHANNELS.index("static_obstacle_map")],
        static_occupancy,
    )
    expected_robot = rasterize_footprint(
        RectangleFootprint(
            config["robot"]["length_m"], config["robot"]["width_m"]
        ),
        base_state.robot_history[-1],
        grid,
    ).astype(np.float32)
    assert np.array_equal(
        rendered.state_channels[STATE_CHANNELS.index("robot_footprint")],
        expected_robot,
    )
    assert np.all(
        rendered.state_channels[STATE_CHANNELS.index("robot_velocity_channel")]
        == np.float32(0.375)
    )
    assert np.all(
        rendered.state_channels[STATE_CHANNELS.index("robot_yaw_rate_channel")]
        == np.float32(-0.25)
    )

    assert set(rendered.metadata) == {
        "renderer_layout_version",
        "base_state_id",
        "sensor_config_digest",
        "static_occupancy_digest",
    }
    assert rendered.metadata["renderer_layout_version"] == RENDERER_LAYOUT_VERSION
    assert rendered.metadata["base_state_id"] == base_state.state_id
    assert isinstance(rendered.metadata["sensor_config_digest"], str)
    assert isinstance(rendered.metadata["static_occupancy_digest"], str)


def test_static_wall_occludes_cells_behind_under_full_360_sensor() -> None:
    config = _toy_config()
    grid = build_grid_spec(config)
    static_occupancy = np.zeros((9, 9), dtype=np.float32)
    static_occupancy[:, 6] = 1.0
    rendered = _render_empty(
        config,
        _empty_base_state(static_occupancy),
        static_occupancy,
    )

    wall, behind = world_to_grid(
        np.array([[2.0, 0.0], [4.0, 0.0]], dtype=np.float32), grid
    )
    visible = rendered.bev_history[
        -1, HISTORY_CHANNELS.index("past_visible_mask")
    ]
    current_free = rendered.state_channels[
        STATE_CHANNELS.index("current_visible_free")
    ]
    current_occupied = rendered.state_channels[
        STATE_CHANNELS.index("current_visible_occupied")
    ]
    current_unknown = rendered.state_channels[
        STATE_CHANNELS.index("current_unobservable_mask")
    ]
    assert visible[tuple(wall)] == 1.0
    assert current_occupied[tuple(wall)] == 1.0
    assert visible[tuple(behind)] == 0.0
    assert current_unknown[tuple(behind)] == 1.0
    assert np.array_equal(current_free + current_occupied, visible)
    assert np.array_equal(
        current_free + current_occupied + current_unknown,
        np.ones((9, 9), dtype=np.float32),
    )


def test_structural_blindspot_changes_history_visibility() -> None:
    config = _toy_config()
    grid = build_grid_spec(config)
    static_occupancy = np.zeros((9, 9), dtype=np.float32)
    base_state = _empty_base_state(static_occupancy)
    common = {
        "scene_dynamic_history": {},
        "scene_dynamic_specs": {},
        "static_occupancy": static_occupancy,
        "config": config,
    }

    full_360 = render_observation(base_state, sensor_config=None, **common)
    forward_only = render_observation(
        base_state,
        sensor_config=StructuralBlindSpot(
            forward_fov_deg=180.0,
            range_m=10.0,
        ),
        **common,
    )

    front, back = world_to_grid(
        np.array([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32), grid
    )
    visible_index = HISTORY_CHANNELS.index("past_visible_mask")
    full_visible = full_360.bev_history[-1, visible_index]
    forward_visible = forward_only.bev_history[-1, visible_index]
    assert full_visible[tuple(front)] == 1.0
    assert full_visible[tuple(back)] == 1.0
    assert forward_visible[tuple(front)] == 1.0
    assert forward_visible[tuple(back)] == 0.0
    assert not np.array_equal(forward_visible, full_visible)
    forward_free = forward_only.state_channels[
        STATE_CHANNELS.index("current_visible_free")
    ]
    forward_occupied = forward_only.state_channels[
        STATE_CHANNELS.index("current_visible_occupied")
    ]
    forward_unknown = forward_only.state_channels[
        STATE_CHANNELS.index("current_unobservable_mask")
    ]
    assert forward_unknown[tuple(back)] == 1.0
    assert np.array_equal(forward_free + forward_occupied, forward_visible)
    assert np.array_equal(
        forward_free + forward_occupied + forward_unknown,
        np.ones((9, 9), dtype=np.float32),
    )
    assert (
        forward_only.metadata["sensor_config_digest"]
        != full_360.metadata["sensor_config_digest"]
    )


def test_history_visibility_uses_each_robot_pose_and_yaw() -> None:
    config = _toy_config()
    grid = build_grid_spec(config)
    static_occupancy = np.zeros((9, 9), dtype=np.float32)
    robot_history = np.array(
        [
            [-1.0, 0.0, np.pi],
            [0.0, 1.0, -np.pi / 2.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(robot_history[-1], np.zeros(3, dtype=np.float32))
    base_state = replace(
        _empty_base_state(static_occupancy),
        robot_history=robot_history,
    )

    rendered = _render_empty(
        config,
        base_state,
        static_occupancy,
        sensor_config=StructuralBlindSpot(
            forward_fov_deg=60.0,
            range_m=10.0,
        ),
    )

    left, down, right = world_to_grid(
        np.array(
            [[-3.0, 0.0], [0.0, -2.0], [2.0, 0.0]],
            dtype=np.float32,
        ),
        grid,
    )
    visible = rendered.bev_history[
        :, HISTORY_CHANNELS.index("past_visible_mask")
    ]
    assert visible[0][tuple(left)] == 1.0
    assert visible[0][tuple(right)] == 0.0
    assert visible[1][tuple(down)] == 1.0
    assert visible[1][tuple(right)] == 0.0
    assert visible[2][tuple(right)] == 1.0
    assert visible[2][tuple(left)] == 0.0

    current_free = rendered.state_channels[
        STATE_CHANNELS.index("current_visible_free")
    ]
    current_occupied = rendered.state_channels[
        STATE_CHANNELS.index("current_visible_occupied")
    ]
    current_unknown = rendered.state_channels[
        STATE_CHANNELS.index("current_unobservable_mask")
    ]
    assert np.array_equal(current_free + current_occupied, visible[-1])
    assert np.array_equal(
        current_free + current_occupied + current_unknown,
        np.ones((9, 9), dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("size", True),
        ("size", 9.5),
        ("size", "9"),
        ("history_steps", True),
        ("history_steps", 3.5),
        ("history_steps", "3"),
        ("future_steps", True),
        ("future_steps", 2.5),
        ("future_steps", "2"),
        ("resolution_m", True),
        ("resolution_m", "1.0"),
    ],
)
def test_bev_config_rejects_values_that_would_be_silently_coerced(
    field: str,
    invalid_value: object,
) -> None:
    config = _toy_config()
    config["bev"][field] = invalid_value
    static_occupancy = np.zeros((9, 9), dtype=np.float32)

    with pytest.raises(
        (TypeError, ValueError), match=rf"config\.bev\.{field}"
    ):
        _render_empty(
            config,
            _empty_base_state(static_occupancy),
            static_occupancy,
        )


def test_bev_config_rejects_range_inconsistent_with_grid_extent() -> None:
    config = _toy_config()
    config["bev"]["range_m"] = 8.0
    static_occupancy = np.zeros((9, 9), dtype=np.float32)

    with pytest.raises(ValueError, match=r"config\.bev\.range_m"):
        _render_empty(
            config,
            _empty_base_state(static_occupancy),
            static_occupancy,
        )


def test_renderer_rejects_non_origin_current_robot_pose() -> None:
    config = _toy_config()
    static_occupancy = np.zeros((9, 9), dtype=np.float32)
    base_state = _empty_base_state(static_occupancy)
    robot_history = base_state.robot_history.copy()
    robot_history[-1, 0] = 0.25

    with pytest.raises(ValueError, match="local origin"):
        _render_empty(
            config,
            replace(base_state, robot_history=robot_history),
            static_occupancy,
        )


@pytest.mark.parametrize(
    "static_occupancy",
    [
        np.zeros((9, 9), dtype=np.float64),
        np.zeros((8, 9), dtype=np.float32),
        np.full((9, 9), np.nan, dtype=np.float32),
        np.full((9, 9), np.inf, dtype=np.float32),
        np.full((9, 9), 0.5, dtype=np.float32),
    ],
    ids=("wrong-dtype", "wrong-shape", "nan", "inf", "nonbinary"),
)
def test_renderer_rejects_invalid_static_occupancy(
    static_occupancy: np.ndarray,
) -> None:
    valid_static = np.zeros((9, 9), dtype=np.float32)

    with pytest.raises((TypeError, ValueError)):
        _render_empty(
            _toy_config(),
            _empty_base_state(valid_static),
            static_occupancy,
        )


def _circle_spec(radius_m: float = 0.35) -> dict[str, object]:
    return {
        "object_type": "human",
        "footprint": {"kind": "circle", "radius_m": radius_m},
    }


def _rectangle_spec(
    length_m: float = 1.8,
    width_m: float = 0.4,
) -> dict[str, object]:
    return {
        "object_type": "carried_object",
        "footprint": {
            "kind": "rectangle",
            "length_m": length_m,
            "width_m": width_m,
        },
    }


def _two_actor_renderer_inputs(
    *, base_state_id: str = "two-actor-base"
) -> dict[str, object]:
    static = np.zeros((9, 9), dtype=np.float32)
    static[1, 1] = 1.0
    context_history = np.array(
        [
            [-2.0, 2.0, 0.0],
            [-2.0, 2.0, np.pi / 4.0],
            [-2.0, 2.0, np.pi / 2.0],
        ],
        dtype=np.float32,
    )
    target_history = np.array(
        [
            [2.0, -2.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    context_id = "context-z"
    target_id = "target-a"
    base_state = replace(
        _empty_base_state(static),
        state_id=base_state_id,
        dynamic_object_ids=(context_id,),
        visible_dynamic_object_history={context_id: context_history.copy()},
        visible_dynamic_object_specs={context_id: _rectangle_spec()},
    )
    return {
        "base_state": base_state,
        "scene_dynamic_history": {
            context_id: context_history.copy(),
            target_id: target_history.copy(),
        },
        "scene_dynamic_specs": {
            context_id: _rectangle_spec(),
            target_id: _circle_spec(),
        },
        "static_occupancy": static,
        "sensor_config": None,
        "config": _toy_config(),
    }


def _canonical_metadata_bytes(metadata: dict[str, str]) -> bytes:
    return json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _spy_on_total_occupancy(
    monkeypatch: pytest.MonkeyPatch,
) -> list[np.ndarray]:
    real_raycast_visibility = observation_renderer_module.raycast_visibility
    captured: list[np.ndarray] = []

    def capture_and_raycast(
        occupancy: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> np.ndarray:
        captured.append(np.array(occupancy, dtype=bool, order="C", copy=True))
        return real_raycast_visibility(occupancy, *args, **kwargs)

    monkeypatch.setattr(
        observation_renderer_module,
        "raycast_visibility",
        capture_and_raycast,
    )
    return captured


def test_scene_history_uses_total_occupancy_for_visibility_and_masks_hidden_dynamic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _toy_config()
    grid = build_grid_spec(config)
    static = np.zeros((9, 9), dtype=np.float32)
    static[:, 5] = 1.0
    context_id = "context-rectangle"
    target_id = "augmented-circle"
    context_history = np.tile(
        np.array([-2.0, 2.0, np.pi / 4.0], dtype=np.float32),
        (3, 1),
    )
    target_history = np.array(
        [
            [-2.0, -2.0, 0.0],
            [2.0, -2.0, 0.7],
            [3.0, 0.0, 1.4],
        ],
        dtype=np.float32,
    )
    base_state = replace(
        _empty_base_state(static),
        dynamic_object_ids=(context_id,),
        visible_dynamic_object_history={context_id: context_history.copy()},
        visible_dynamic_object_specs={context_id: _rectangle_spec()},
    )
    scene_history = {
        context_id: context_history.copy(),
        target_id: target_history.copy(),
    }
    scene_specs = {
        context_id: _rectangle_spec(),
        target_id: _circle_spec(),
    }
    captured = _spy_on_total_occupancy(monkeypatch)

    rendered = render_observation(
        base_state,
        scene_dynamic_history=scene_history,
        scene_dynamic_specs=scene_specs,
        static_occupancy=static,
        sensor_config=None,
        config=config,
    )

    assert len(captured) == config["bev"]["history_steps"]
    dynamic = rendered.bev_history[
        :, HISTORY_CHANNELS.index("past_dynamic_occupancy")
    ]
    visible = rendered.bev_history[:, HISTORY_CHANNELS.index("past_visible_mask")]
    assert np.all(dynamic <= visible)

    target_footprint = CircleFootprint(0.35)
    target_visible_cells = rasterize_footprint(
        target_footprint, target_history[0], grid
    )
    target_hidden_cells = rasterize_footprint(
        target_footprint, target_history[-1], grid
    )
    assert target_visible_cells.any()
    assert dynamic[0, target_visible_cells].any()
    assert target_hidden_cells.any()
    assert captured[-1][target_hidden_cells].all()
    assert not visible[-1, target_hidden_cells].any()
    assert not dynamic[-1, target_hidden_cells].any()

    context_cells = rasterize_footprint(
        RectangleFootprint(1.8, 0.4), context_history[-1], grid
    )
    visible_context = context_cells & visible[-1].astype(bool)
    assert visible_context.any()
    assert dynamic[-1, visible_context].all()

    current_occupied = rendered.state_channels[
        STATE_CHANNELS.index("current_visible_occupied")
    ].astype(bool)
    expected_current_occupied = captured[-1] & visible[-1].astype(bool)
    np.testing.assert_array_equal(current_occupied, expected_current_occupied)
    assert current_occupied[visible_context].all()
    assert current_occupied[static.astype(bool) & visible[-1].astype(bool)].all()


def test_renderer_circle_footprint_mask_is_yaw_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _toy_config()
    static = np.zeros((9, 9), dtype=np.float32)
    history = np.array(
        [
            [-2.0, -2.0, 0.0],
            [-2.0, -2.0, np.pi / 3.0],
            [-2.0, -2.0, -np.pi / 2.0],
        ],
        dtype=np.float32,
    )
    captured = _spy_on_total_occupancy(monkeypatch)

    render_observation(
        _empty_base_state(static),
        scene_dynamic_history={"circle": history},
        scene_dynamic_specs={"circle": _circle_spec(0.75)},
        static_occupancy=static,
        sensor_config=None,
        config=config,
    )

    assert len(captured) == 3
    expected = rasterize_footprint(
        CircleFootprint(0.75), history[0], build_grid_spec(config)
    )
    for occupancy in captured:
        np.testing.assert_array_equal(occupancy, expected)


def test_renderer_rectangle_footprint_rotates_its_long_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _toy_config()
    grid = build_grid_spec(config)
    static = np.zeros((9, 9), dtype=np.float32)
    history = np.array(
        [
            [0.0, 2.0, 0.0],
            [0.0, 2.0, np.pi / 2.0],
            [0.0, 2.0, np.pi / 2.0],
        ],
        dtype=np.float32,
    )
    footprint = RectangleFootprint(3.0, 0.4)
    captured = _spy_on_total_occupancy(monkeypatch)

    render_observation(
        _empty_base_state(static),
        scene_dynamic_history={"rectangle": history},
        scene_dynamic_specs={"rectangle": _rectangle_spec(3.0, 0.4)},
        static_occupancy=static,
        sensor_config=None,
        config=config,
    )

    assert len(captured) == 3
    horizontal = rasterize_footprint(footprint, history[0], grid)
    vertical = rasterize_footprint(footprint, history[1], grid)
    np.testing.assert_array_equal(captured[0], horizontal)
    np.testing.assert_array_equal(captured[1], vertical)
    assert not np.array_equal(horizontal, vertical)
    horizontal_rows, horizontal_columns = np.where(horizontal)
    vertical_rows, vertical_columns = np.where(vertical)
    assert np.ptp(horizontal_columns) > np.ptp(horizontal_rows)
    assert np.ptp(vertical_rows) > np.ptp(vertical_columns)


def test_renderer_rejects_scene_history_and_spec_key_mismatch() -> None:
    static = np.zeros((9, 9), dtype=np.float32)

    with pytest.raises(ValueError, match="keys must match"):
        render_observation(
            _empty_base_state(static),
            scene_dynamic_history={
                "actor": np.zeros((3, 3), dtype=np.float32)
            },
            scene_dynamic_specs={},
            static_occupancy=static,
            sensor_config=None,
            config=_toy_config(),
        )


@pytest.mark.parametrize("object_id", ["", 7], ids=("empty", "nonstring"))
def test_renderer_rejects_invalid_scene_object_ids(object_id: object) -> None:
    static = np.zeros((9, 9), dtype=np.float32)

    with pytest.raises((TypeError, ValueError), match="non-empty strings"):
        render_observation(
            _empty_base_state(static),
            scene_dynamic_history={
                object_id: np.zeros((3, 3), dtype=np.float32)
            },
            scene_dynamic_specs={object_id: _circle_spec()},
            static_occupancy=static,
            sensor_config=None,
            config=_toy_config(),
        )


@pytest.mark.parametrize(
    "history",
    [
        [[0.0, 0.0, 0.0]] * 3,
        np.zeros((3, 3), dtype=np.float64),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
        np.full((3, 3), np.nan, dtype=np.float32),
        np.full((3, 3), np.inf, dtype=np.float32),
        np.full((3, 3), -np.inf, dtype=np.float32),
    ],
    ids=(
        "not-array",
        "wrong-dtype",
        "wrong-k",
        "wrong-pose",
        "nan",
        "positive-inf",
        "negative-inf",
    ),
)
def test_renderer_requires_exact_scene_history_contract(history: object) -> None:
    static = np.zeros((9, 9), dtype=np.float32)

    with pytest.raises((TypeError, ValueError), match="scene_dynamic_history"):
        render_observation(
            _empty_base_state(static),
            scene_dynamic_history={"actor": history},
            scene_dynamic_specs={"actor": _circle_spec()},
            static_occupancy=static,
            sensor_config=None,
            config=_toy_config(),
        )


def test_renderer_rejects_malformed_scene_spec() -> None:
    static = np.zeros((9, 9), dtype=np.float32)

    with pytest.raises(ValueError):
        render_observation(
            _empty_base_state(static),
            scene_dynamic_history={
                "actor": np.zeros((3, 3), dtype=np.float32)
            },
            scene_dynamic_specs={"actor": _circle_spec(-0.1)},
            static_occupancy=static,
            sensor_config=None,
            config=_toy_config(),
        )


@pytest.mark.parametrize(
    "alteration",
    ["missing", "history", "spec"],
)
def test_renderer_rejects_missing_or_altered_base_context(
    alteration: str,
) -> None:
    static = np.zeros((9, 9), dtype=np.float32)
    context_id = "context"
    context_history = np.tile(
        np.array([-2.0, 2.0, np.pi / 4.0], dtype=np.float32),
        (3, 1),
    )
    base_state = replace(
        _empty_base_state(static),
        dynamic_object_ids=(context_id,),
        visible_dynamic_object_history={context_id: context_history.copy()},
        visible_dynamic_object_specs={context_id: _rectangle_spec()},
    )
    scene_history: dict[str, np.ndarray] = {context_id: context_history.copy()}
    scene_specs = {context_id: _rectangle_spec()}
    if alteration == "missing":
        scene_history.clear()
        scene_specs.clear()
    elif alteration == "history":
        scene_history[context_id][0, 0] += np.float32(0.25)
    else:
        scene_specs[context_id] = _rectangle_spec(2.0, 0.4)

    with pytest.raises(ValueError, match="BaseState context"):
        render_observation(
            base_state,
            scene_dynamic_history=scene_history,
            scene_dynamic_specs=scene_specs,
            static_occupancy=static,
            sensor_config=None,
            config=_toy_config(),
        )


def test_renderer_rejects_static_map_that_removes_base_static_cells() -> None:
    base_static = np.zeros((9, 9), dtype=np.float32)
    base_static[2, 3] = 1.0
    passed_static = np.zeros((9, 9), dtype=np.float32)

    with pytest.raises(ValueError, match="BaseState static"):
        _render_empty(
            _toy_config(),
            _empty_base_state(base_static),
            passed_static,
        )


@pytest.fixture
def hand_computed_belief_scene() -> dict[str, object]:
    config = _toy_config()
    grid = build_grid_spec(config)
    static = np.zeros((9, 9), dtype=np.float32)
    static_cell, = world_to_grid(
        np.array([[2.0, 1.0]], dtype=np.float32), grid
    )
    static[tuple(static_cell)] = 1.0

    robot_history = np.array(
        [
            [0.0, 0.0, np.pi / 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    target_history = np.tile(
        np.array([0.0, 2.0, 0.0], dtype=np.float32),
        (3, 1),
    )
    background_history = np.tile(
        np.array([2.0, 0.0, 0.0], dtype=np.float32),
        (3, 1),
    )
    target_id = "target"
    background_id = "background"
    target_spec = _circle_spec(0.2)
    background_spec = _circle_spec(0.2)
    base_state = replace(
        _empty_base_state(static),
        dynamic_object_ids=(background_id,),
        robot_history=robot_history,
        visible_dynamic_object_history={
            background_id: background_history.copy()
        },
        visible_dynamic_object_specs={background_id: background_spec.copy()},
    )

    cells = world_to_grid(
        np.array(
            [
                [0.0, 2.0],
                [2.0, 0.0],
                [1.0, 0.0],
                [2.0, 1.0],
                [-2.0, -2.0],
                [0.0, 3.0],
            ],
            dtype=np.float32,
        ),
        grid,
    )
    np.testing.assert_array_equal(
        cells,
        np.array(
            [
                [6, 4],
                [4, 6],
                [4, 5],
                [5, 6],
                [2, 2],
                [7, 4],
            ]
        ),
    )
    return {
        "base_state": base_state,
        "scene_dynamic_history": {
            background_id: background_history,
            target_id: target_history,
        },
        "scene_dynamic_specs": {
            background_id: background_spec,
            target_id: target_spec,
        },
        "static_occupancy": static,
        "sensor_config": StructuralBlindSpot(
            forward_fov_deg=60.0,
            range_m=10.0,
        ),
        "config": config,
        "target_id": target_id,
        "cells": cells,
    }


def test_last_seen_dynamic_and_age_are_hand_computed_from_visibility_history(
    hand_computed_belief_scene: dict[str, object],
) -> None:
    rendered = render_observation(
        hand_computed_belief_scene["base_state"],
        scene_dynamic_history=hand_computed_belief_scene[
            "scene_dynamic_history"
        ],
        scene_dynamic_specs=hand_computed_belief_scene["scene_dynamic_specs"],
        static_occupancy=hand_computed_belief_scene["static_occupancy"],
        sensor_config=hand_computed_belief_scene["sensor_config"],
        config=hand_computed_belief_scene["config"],
    )

    target, background, free, static, never_visible, _ = (
        hand_computed_belief_scene["cells"]
    )
    dynamic = rendered.bev_history[
        :, HISTORY_CHANNELS.index("past_dynamic_occupancy")
    ]
    visible = rendered.bev_history[
        :, HISTORY_CHANNELS.index("past_visible_mask")
    ]
    last_seen = rendered.state_channels[
        STATE_CHANNELS.index("last_seen_occupancy")
    ]
    age = rendered.state_channels[STATE_CHANNELS.index("occlusion_age_map")]

    np.testing.assert_array_equal(
        visible[:, int(target[0]), int(target[1])],
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        dynamic[:, int(target[0]), int(target[1])],
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        visible[:, int(background[0]), int(background[1])],
        np.array([0.0, 1.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        dynamic[:, int(background[0]), int(background[1])],
        np.array([0.0, 1.0, 1.0], dtype=np.float32),
    )
    assert last_seen.dtype == np.float32
    assert last_seen[tuple(target)] == 1.0
    assert age[tuple(target)] == pytest.approx(0.08)
    assert last_seen[tuple(background)] == 1.0
    assert age[tuple(background)] == 0.0
    assert visible[-1][tuple(free)] == 1.0
    assert age[tuple(free)] == 0.0
    assert visible[-1][tuple(static)] == 1.0
    assert last_seen[tuple(static)] == 0.0
    assert age[tuple(static)] == 0.0
    assert not visible[:, int(never_visible[0]), int(never_visible[1])].any()
    assert last_seen[tuple(never_visible)] == 0.0
    assert age[tuple(never_visible)] == 1.0


def test_empty_variant_is_independently_rerendered_after_target_deletion(
    hand_computed_belief_scene: dict[str, object],
) -> None:
    full_history = hand_computed_belief_scene["scene_dynamic_history"]
    full_specs = hand_computed_belief_scene["scene_dynamic_specs"]
    target_id = hand_computed_belief_scene["target_id"]
    common = {
        "static_occupancy": hand_computed_belief_scene["static_occupancy"],
        "sensor_config": hand_computed_belief_scene["sensor_config"],
        "config": hand_computed_belief_scene["config"],
    }

    full = render_observation(
        hand_computed_belief_scene["base_state"],
        scene_dynamic_history=full_history,
        scene_dynamic_specs=full_specs,
        **common,
    )
    full_bev_snapshot = full.bev_history.copy()
    full_state_snapshot = full.state_channels.copy()
    history_snapshot = {
        object_id: history.copy() for object_id, history in full_history.items()
    }
    static_snapshot = common["static_occupancy"].copy()

    empty = render_observation(
        hand_computed_belief_scene["base_state"],
        scene_dynamic_history={
            object_id: history
            for object_id, history in full_history.items()
            if object_id != target_id
        },
        scene_dynamic_specs={
            object_id: spec
            for object_id, spec in full_specs.items()
            if object_id != target_id
        },
        **common,
    )

    np.testing.assert_array_equal(full.bev_history, full_bev_snapshot)
    np.testing.assert_array_equal(full.state_channels, full_state_snapshot)
    for object_id, history in full_history.items():
        np.testing.assert_array_equal(history, history_snapshot[object_id])
    np.testing.assert_array_equal(common["static_occupancy"], static_snapshot)
    assert not np.shares_memory(full.bev_history, empty.bev_history)
    assert not np.shares_memory(full.state_channels, empty.state_channels)

    target, background, _, _, _, behind_target = (
        hand_computed_belief_scene["cells"]
    )
    dynamic_index = HISTORY_CHANNELS.index("past_dynamic_occupancy")
    visible_index = HISTORY_CHANNELS.index("past_visible_mask")
    last_seen_index = STATE_CHANNELS.index("last_seen_occupancy")
    age_index = STATE_CHANNELS.index("occlusion_age_map")
    np.testing.assert_array_equal(
        empty.bev_history[:, dynamic_index, int(target[0]), int(target[1])],
        np.zeros(3, dtype=np.float32),
    )
    assert full.state_channels[last_seen_index][tuple(target)] == 1.0
    assert empty.state_channels[last_seen_index][tuple(target)] == 0.0
    assert empty.state_channels[age_index][tuple(target)] == pytest.approx(0.08)
    np.testing.assert_array_equal(
        full.bev_history[
            :, dynamic_index, int(background[0]), int(background[1])
        ],
        empty.bev_history[
            :, dynamic_index, int(background[0]), int(background[1])
        ],
    )

    assert full.bev_history[0, visible_index][tuple(behind_target)] == 0.0
    assert full.state_channels[age_index][tuple(behind_target)] == 1.0
    assert empty.bev_history[0, visible_index][tuple(behind_target)] == 1.0
    assert empty.state_channels[age_index][tuple(behind_target)] == pytest.approx(
        0.08
    )

    unchanged_state_channels = (
        "current_visible_free",
        "current_visible_occupied",
        "current_unobservable_mask",
        "static_obstacle_map",
        "robot_footprint",
        "robot_velocity_channel",
        "robot_yaw_rate_channel",
    )
    for channel in unchanged_state_channels:
        index = STATE_CHANNELS.index(channel)
        np.testing.assert_array_equal(
            full.state_channels[index], empty.state_channels[index]
        )


def test_last_seen_dynamic_is_overwritten_by_later_visible_free_observation(
) -> None:
    config = _toy_config()
    grid = build_grid_spec(config)
    static = np.zeros((9, 9), dtype=np.float32)
    robot_history = np.array(
        [
            [0.0, 0.0, np.pi / 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    target_history = np.array(
        [
            [2.0, 2.0, 0.0],
            [-2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    blocker_history = np.array(
        [
            [-2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    base_state = replace(
        _empty_base_state(static),
        robot_history=robot_history,
    )

    rendered = render_observation(
        base_state,
        scene_dynamic_history={
            "transient-target": target_history,
            "late-blocker": blocker_history,
        },
        scene_dynamic_specs={
            "transient-target": _circle_spec(0.2),
            "late-blocker": _circle_spec(0.2),
        },
        static_occupancy=static,
        sensor_config=StructuralBlindSpot(
            forward_fov_deg=160.0,
            range_m=10.0,
        ),
        config=config,
    )

    target_cell, = world_to_grid(
        np.array([[2.0, 2.0]], dtype=np.float32), grid
    )
    np.testing.assert_array_equal(target_cell, np.array([6, 6]))
    dynamic = rendered.bev_history[
        :, HISTORY_CHANNELS.index("past_dynamic_occupancy")
    ]
    visible = rendered.bev_history[
        :, HISTORY_CHANNELS.index("past_visible_mask")
    ]
    last_seen = rendered.state_channels[
        STATE_CHANNELS.index("last_seen_occupancy")
    ]
    age = rendered.state_channels[STATE_CHANNELS.index("occlusion_age_map")]

    np.testing.assert_array_equal(
        visible[:, int(target_cell[0]), int(target_cell[1])],
        np.array([1.0, 1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        dynamic[:, int(target_cell[0]), int(target_cell[1])],
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    assert last_seen[tuple(target_cell)] == 0.0
    assert age[tuple(target_cell)] == pytest.approx(0.04)


def test_caller_side_future_target_and_timestamp_cannot_change_rendering() -> None:
    renderer_inputs = _two_actor_renderer_inputs()
    renderer_keys = tuple(renderer_inputs)
    first_world = {
        **renderer_inputs,
        "dynamic_object_future": np.zeros((2, 3), dtype=np.float32),
        "target_sentinel": "first-target",
        "timestamp_sentinel": -10**30,
    }
    second_world = {
        **renderer_inputs,
        "dynamic_object_future": np.full(
            (101, 7), np.inf, dtype=np.float32
        ),
        "target_sentinel": object(),
        "timestamp_sentinel": "far-future-time-that-must-not-be-read",
    }

    first = render_observation(
        **{key: first_world[key] for key in renderer_keys}
    )
    second = render_observation(
        **{key: second_world[key] for key in renderer_keys}
    )

    assert first.bev_history.tobytes(order="C") == second.bev_history.tobytes(
        order="C"
    )
    assert first.state_channels.tobytes(
        order="C"
    ) == second.state_channels.tobytes(order="C")
    assert _canonical_metadata_bytes(
        first.metadata
    ) == _canonical_metadata_bytes(second.metadata)


def test_metadata_is_exact_string_only_safe_whitelist_without_key_leakage(
) -> None:
    arbitrary_id = (
        "oracle world future trajectory target object-list visibility "
        "timestamp time are legitimate state-id text"
    )
    inputs = _two_actor_renderer_inputs(base_state_id=arbitrary_id)
    rendered = render_observation(**inputs)
    whitelist = {
        "renderer_layout_version",
        "base_state_id",
        "sensor_config_digest",
        "static_occupancy_digest",
    }
    forbidden_key_tokens = (
        "oracle",
        "world",
        "future",
        "trajectory",
        "target",
        "object",
        "list",
        "visibility",
        "timestamp",
        "time",
    )

    def audit(value: object, *, root: bool = False) -> None:
        if root:
            assert type(value) is dict
            for key, child in value.items():
                assert type(key) is str
                assert not any(token in key.lower() for token in forbidden_key_tokens)
                audit(child)
            return
        assert type(value) is str

    assert set(rendered.metadata) == whitelist
    audit(rendered.metadata, root=True)
    assert rendered.metadata["base_state_id"] == arbitrary_id
    assert (
        rendered.metadata["renderer_layout_version"]
        == RENDERER_LAYOUT_VERSION
    )

    actor_ids = tuple(inputs["scene_dynamic_history"])
    serialized_payload_markers = ("{", "}", "[", "]", '"', ":", ",")
    for key, value in rendered.metadata.items():
        if key == "base_state_id":
            continue
        lowered = value.lower()
        assert not any(token in lowered for token in forbidden_key_tokens)
        assert not any(actor_id.lower() in lowered for actor_id in actor_ids)
        assert not any(marker in value for marker in serialized_payload_markers)
        assert lowered not in {"true", "false", "null", "none"}

    for key in ("sensor_config_digest", "static_occupancy_digest"):
        digest = rendered.metadata[key]
        assert len(digest) == 32
        assert digest == digest.lower()
        assert all(character in "0123456789abcdef" for character in digest)
        assert int(digest, 16) >= 0


def test_outputs_are_owned_and_do_not_alias_or_mutate_any_array_input() -> None:
    inputs = _two_actor_renderer_inputs()
    base_state = inputs["base_state"]
    assert isinstance(base_state, BaseState)
    ndarray_inputs = [
        inputs["static_occupancy"],
        base_state.robot_history,
        base_state.robot_state,
        base_state.static_map_local,
        *base_state.visible_dynamic_object_history.values(),
        *inputs["scene_dynamic_history"].values(),
    ]
    assert all(isinstance(array, np.ndarray) for array in ndarray_inputs)
    snapshots = [array.copy() for array in ndarray_inputs]

    rendered = render_observation(**inputs)

    for output in (rendered.bev_history, rendered.state_channels):
        assert output.flags.owndata
        assert output.flags.c_contiguous
        for input_array in ndarray_inputs:
            assert not np.shares_memory(output, input_array)
    for input_array, snapshot in zip(ndarray_inputs, snapshots, strict=True):
        np.testing.assert_array_equal(input_array, snapshot)


def test_repeated_calls_and_reversed_actor_insertion_order_are_byte_identical(
) -> None:
    inputs = _two_actor_renderer_inputs()
    repeated = render_observation(**inputs)
    repeated_again = render_observation(**inputs)
    reversed_history_inputs = {
        **inputs,
        "scene_dynamic_history": dict(
            reversed(tuple(inputs["scene_dynamic_history"].items()))
        ),
    }
    reversed_specs_inputs = {
        **inputs,
        "scene_dynamic_specs": dict(
            reversed(tuple(inputs["scene_dynamic_specs"].items()))
        ),
    }
    reversed_history = render_observation(**reversed_history_inputs)
    reversed_specs = render_observation(**reversed_specs_inputs)

    for candidate in (repeated_again, reversed_history, reversed_specs):
        assert repeated.bev_history.tobytes(
            order="C"
        ) == candidate.bev_history.tobytes(order="C")
        assert repeated.state_channels.tobytes(
            order="C"
        ) == candidate.state_channels.tobytes(order="C")
        assert _canonical_metadata_bytes(
            repeated.metadata
        ) == _canonical_metadata_bytes(candidate.metadata)


def test_renderer_rejects_mapping_as_sensor_config() -> None:
    inputs = _two_actor_renderer_inputs()
    inputs["sensor_config"] = {"forward_fov_deg": 180.0, "range_m": 10.0}

    with pytest.raises(TypeError, match="sensor_config"):
        render_observation(**inputs)


def test_renderer_rejects_earlier_robot_pose_outside_grid() -> None:
    inputs = _two_actor_renderer_inputs()
    base_state = inputs["base_state"]
    assert isinstance(base_state, BaseState)
    robot_history = base_state.robot_history.copy()
    robot_history[0] = np.array([100.0, 0.0, 0.0], dtype=np.float32)
    assert np.array_equal(robot_history[-1], np.zeros(3, dtype=np.float32))
    inputs["base_state"] = replace(base_state, robot_history=robot_history)

    with pytest.raises(ValueError):
        render_observation(**inputs)
