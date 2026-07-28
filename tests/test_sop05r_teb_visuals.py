from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation.sop05r_teb_visuals import (
    build_sop05r_teb_visual_bundle,
    render_sop05r_teb_visual_bundle,
)
from src.planning.verification_actions import load_verification_actions
from tests.test_sop05r_teb_event_sampler import _mother_fixture
from tests.test_sop05r_teb_sop06_handoff import _strict_collection


ROOT = Path(__file__).resolve().parents[1]


def _bundle(tmp_path: Path):
    collection = _strict_collection(tmp_path)
    teb_config = _mother_fixture()[3]
    return build_sop05r_teb_visual_bundle(
        collection,
        event_id=collection.events[0].generated_event_id,
        teb_config=teb_config,
        action_library=load_verification_actions(
            ROOT / "configs/verification_actions.yaml"
        ),
    )


def test_teb_visual_bundle_contains_authenticated_long40_layers(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    assert bundle.full_route.shape == (40, 3)
    assert bundle.nominal_suffix.shape == (32, 3)
    assert bundle.target_long40.shape == (40, 3)
    assert bundle.target_visibility_history.shape == (8,)
    assert 0 <= bundle.witness_sample_index < 8
    assert len(bundle.verification_traces) == 6
    np.testing.assert_allclose(
        bundle.full_route[:32],
        bundle.nominal_suffix,
        rtol=0.0,
        atol=1e-5,
    )


def test_teb_visual_is_fixed_nonblank_and_deterministic(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    first = render_sop05r_teb_visual_bundle(bundle, tmp_path / "first.png")
    second = render_sop05r_teb_visual_bundle(bundle, tmp_path / "second.png")

    with Image.open(first.path) as image:
        assert image.size == (1440, 1200)
        assert np.asarray(image.convert("RGB")).std() > 5.0
    assert first.metadata["sha256"] == second.metadata["sha256"]
    assert first.metadata["human_sample_count"] == 40
    assert first.metadata["history_frame_definition"].startswith("H0=-1.4s")
    assert first.metadata["verification_action_line_style"] == "dashed"
    assert first.metadata["verification_action_zoom"] == (
        "decision_frame_-0.30_to_0.75m_x_-0.45_to_0.45m_y"
    )
    assert first.metadata["verification_action_ids"] == [
        "arc_left_30",
        "arc_right_30",
        "arc_left_45",
        "arc_right_45",
        "forward_peek",
        "stop_scan",
    ]


def test_teb_visual_marks_robot_and_pedestrian_travel_directions(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    artifact = render_sop05r_teb_visual_bundle(bundle, tmp_path / "directions.png")

    assert artifact.metadata["direction_annotations"] == ["robot", "pedestrian"]


def test_teb_visual_renders_an_empty_pedestrian_branch(tmp_path: Path) -> None:
    bundle = replace(
        _bundle(tmp_path),
        target_long40=None,
        target_visibility_history=None,
        collision_point_xy=None,
        collision_time_s=None,
        witness_sample_index=None,
        witness_occluder_id=None,
    )

    artifact = render_sop05r_teb_visual_bundle(bundle, tmp_path / "empty.png")

    assert artifact.path.is_file()
    assert artifact.metadata["human_sample_count"] == 0
    assert artifact.metadata["direction_annotations"] == ["robot"]
    assert artifact.metadata["pedestrian_present"] is False
