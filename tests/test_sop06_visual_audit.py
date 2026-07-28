"""Focused acceptance coverage for the offline SOP06 visual review artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.contracts import GridSpec
from src.evaluation.sop06_visual_audit import (
    SOP06_AUDIT_PACKET_VERSION,
    Sop06VisualAuditBundle,
    load_sop06_visual_audit_packet,
    render_sop06_visual_audit,
)


def _bundle() -> Sop06VisualAuditBundle:
    grid = GridSpec(
        height=9,
        width=9,
        history_steps=8,
        future_steps=32,
        resolution_m=1.0,
    )
    static = np.zeros((9, 9), dtype=np.uint8)
    static[3, 4] = 1
    current_visible = np.zeros((9, 9), dtype=np.uint8)
    current_visible[4, 3:5] = 1
    current_unobservable = (~current_visible.astype(bool)).astype(np.uint8)
    post_visible = np.zeros((9, 9), dtype=np.uint8)
    post_visible[4, 4:8] = 1
    post_unobservable = (~post_visible.astype(bool)).astype(np.uint8)
    robot_history = np.column_stack(
        (
            np.linspace(-0.7, 0.0, 8, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
        )
    )
    observed_history = np.asarray(
        [True, True, True, True, False, False, False, False],
        dtype=np.bool_,
    )
    pedestrian_history = np.column_stack(
        (
            np.asarray(
                [-1.4, -1.2, -1.0, -0.8, -0.8, -0.8, -0.8, -0.8],
                dtype=np.float32,
            ),
            np.full(8, 0.8, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
        )
    )
    x = np.linspace(0.1, 3.2, 32, dtype=np.float32)
    candidate = np.column_stack(
        (x, np.zeros(32, dtype=np.float32), np.zeros(32, dtype=np.float32))
    )
    hidden = np.column_stack(
        (x, np.full(32, 0.4, dtype=np.float32), np.zeros(32, dtype=np.float32))
    )
    return Sop06VisualAuditBundle(
        sample_id="sop06-audit-fixture",
        grid=grid,
        static_occupancy=static,
        current_visible_mask=current_visible,
        current_unobservable_mask=current_unobservable,
        post_visible_mask=post_visible,
        post_unobservable_mask=post_unobservable,
        robot_history=robot_history,
        dynamic_history_paths={"hidden-human": pedestrian_history},
        dynamic_history_observed={"hidden-human": observed_history},
        candidate_trajectory=candidate,
        oracle_future_paths={"hidden-human": hidden},
        hidden_object_ids=("hidden-human",),
        post_robot_pose=np.asarray([0.3, 0.0, 0.0], dtype=np.float32),
        verification_trace=np.asarray(
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32
        ),
    )


def _write_packet(path: Path) -> None:
    bundle = _bundle()
    np.savez_compressed(
        path,
        audit_packet_version=np.asarray(SOP06_AUDIT_PACKET_VERSION),
        sample_id=np.asarray(bundle.sample_id),
        resolution_m=np.asarray(bundle.grid.resolution_m, dtype=np.float64),
        static_occupancy=bundle.static_occupancy.astype(np.uint8),
        current_visible_mask=bundle.current_visible_mask.astype(np.uint8),
        current_unobservable_mask=bundle.current_unobservable_mask.astype(np.uint8),
        post_visible_mask=bundle.post_visible_mask.astype(np.uint8),
        post_unobservable_mask=bundle.post_unobservable_mask.astype(np.uint8),
        robot_history=bundle.robot_history.astype(np.float32),
        dynamic_history_object_ids=np.asarray(["hidden-human"]),
        dynamic_history_paths=np.asarray(
            [bundle.dynamic_history_paths["hidden-human"]],
            dtype=np.float32,
        ),
        dynamic_history_observed=np.asarray(
            [bundle.dynamic_history_observed["hidden-human"]],
            dtype=np.bool_,
        ),
        candidate_trajectory=bundle.candidate_trajectory.astype(np.float32),
        oracle_object_ids=np.asarray(["hidden-human"]),
        oracle_future_paths=np.asarray(
            [bundle.oracle_future_paths["hidden-human"]], dtype=np.float32
        ),
        hidden_object_ids=np.asarray(["hidden-human"]),
        post_robot_pose=bundle.post_robot_pose.astype(np.float32),
        verification_trace=bundle.verification_trace.astype(np.float32),
    )


def test_sop06_audit_renders_required_pair_and_two_frame_toggle(tmp_path: Path) -> None:
    artifact = render_sop06_visual_audit(_bundle(), tmp_path / "audit")

    assert artifact.bev_pair_path.name == "bev_pair.png"
    assert artifact.bev_toggle_path.name == "bev_toggle.gif"
    assert artifact.bev_pair_path.exists()
    assert artifact.bev_toggle_path.exists()
    with Image.open(artifact.bev_pair_path) as image:
        assert image.size == (1800, 600)
        assert np.asarray(image.convert("RGB")).std() > 5.0
    with Image.open(artifact.bev_toggle_path) as image:
        assert image.size == (800, 800)
        assert image.n_frames == 2
        image.seek(0)
        current = np.asarray(image.convert("RGB"))
        image.seek(1)
        post = np.asarray(image.convert("RGB"))
        assert not np.array_equal(current, post)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["oracle_boundary"] == "offline_audit_only"
    assert manifest["hidden_object_ids"] == ["hidden-human"]
    assert manifest["post_visibility_semantics"] == (
        "action_endpoint_frame_not_trace_union"
    )
    assert manifest["panel_semantics"]["left"]["oracle_future_rendered"] is False
    assert manifest["panel_semantics"]["right"]["oracle_future_rendered"] is True
    assert np.isclose(manifest["robot_endpoint_displacement_m"], 0.3)
    assert manifest["no_longer_visible_at_endpoint_cell_count"] > 0
    assert manifest["candidate_endpoint_count_initially_unobservable"] > 0
    assert manifest["candidate_endpoint_count_revealed_after_verification"] > 0


def test_sop06_audit_packet_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    packet = tmp_path / "packet.npz"
    _write_packet(packet)

    bundle = load_sop06_visual_audit_packet(packet)
    assert bundle.sample_id == "sop06-audit-fixture"
    assert bundle.hidden_object_ids == ("hidden-human",)
    assert bundle.candidate_trajectory.shape == (32, 3)
    assert bundle.dynamic_history_observed["hidden-human"].sum() == 4
    np.testing.assert_allclose(bundle.post_robot_pose, [0.3, 0.0, 0.0])
    render_sop06_visual_audit(bundle, tmp_path / "audit")
    try:
        render_sop06_visual_audit(bundle, tmp_path / "audit")
    except FileExistsError:
        pass
    else:  # pragma: no cover - guards an immutable-artifact regression
        raise AssertionError("SOP06 audit output unexpectedly overwrote an artifact")
