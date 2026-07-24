"""Deterministic visual layers for seen-then-occluded event audits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np

from src.contracts import (
    BaseState,
    GridSpec,
    LocalTrajectory,
    OracleContext,
    validate_base_state,
    validate_oracle_context,
)
from src.generation.dynamic_object_transplant import footprint_from_spec
from src.geometry import (
    grid_bounds,
    rasterize_footprint,
    raycast_candidate_visibility,
    raycast_visibility,
)


REPLAY_HISTORY_TIMES_S = tuple((index - 7) * 0.2 for index in range(8))
REPLAY_FUTURE_TIMES_S = tuple((index + 1) * 0.2 for index in range(15))
PAIRED_PANEL_ORDER = (
    "collision",
    "near_miss",
    "temporal_safe",
    "spatial_safe",
    "irrelevant_hidden",
    "empty_blind_spot",
)


@dataclass(frozen=True)
class VisualVariant:
    kind: str
    target_history: np.ndarray | None
    target_future: np.ndarray | None
    visibility_history: np.ndarray | None
    min_clearance_m: float | None
    time_to_min_clearance_s: float | None
    temporal_offset_s: float | None


@dataclass(frozen=True)
class VisualAuditBundle:
    event_id: str
    base_state: BaseState
    oracle_context: OracleContext
    trajectory: LocalTrajectory
    static_occupancy: np.ndarray
    occluders: tuple[dict[str, object], ...]
    variants: tuple[VisualVariant, ...]
    grid: GridSpec
    robot_length_m: float = 0.7
    robot_width_m: float = 0.55


@dataclass(frozen=True)
class ReplayFrame:
    frame_index: int
    time_s: float
    phase: str
    visibility_mask: np.ndarray
    robot_pose: np.ndarray
    target_pose: np.ndarray
    target_visible: bool | None
    context_object_ids: tuple[str, ...]
    context_poses: np.ndarray
    context_visible: np.ndarray | None


@dataclass(frozen=True)
class VisualArtifactResult:
    event_replay_path: Path
    paired_events_path: Path
    event_replay_metadata: dict[str, object]
    paired_events_metadata: dict[str, object]


def _pose_array(
    value: Any,
    *,
    name: str,
    expected_steps: int,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.shape != (expected_steps, 3):
        raise ValueError(f"{name} must have shape ({expected_steps}, 3)")
    if value.dtype != np.float32:
        raise TypeError(f"{name} must have float32 dtype")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return value


def _validate_variant(variant: VisualVariant, *, index: int) -> None:
    expected_kind = PAIRED_PANEL_ORDER[index]
    if variant.kind != expected_kind:
        raise ValueError("visual variants must follow the frozen panel order")
    if variant.kind == "empty_blind_spot":
        if any(
            value is not None
            for value in (
                variant.target_history,
                variant.target_future,
                variant.visibility_history,
                variant.min_clearance_m,
                variant.time_to_min_clearance_s,
                variant.temporal_offset_s,
            )
        ):
            raise ValueError("empty_blind_spot must remove the target")
        return
    _pose_array(
        variant.target_history,
        name=f"variants[{index}].target_history",
        expected_steps=8,
    )
    _pose_array(
        variant.target_future,
        name=f"variants[{index}].target_future",
        expected_steps=15,
    )
    if not isinstance(variant.visibility_history, np.ndarray):
        raise TypeError("variant visibility_history must be a numpy array")
    if variant.visibility_history.shape != (8,):
        raise ValueError("variant visibility_history must have shape (8,)")
    if variant.visibility_history.dtype != np.bool_:
        raise TypeError("variant visibility_history must have boolean dtype")
    for name, value in (
        ("min_clearance_m", variant.min_clearance_m),
        ("time_to_min_clearance_s", variant.time_to_min_clearance_s),
    ):
        if value is None or not np.isfinite(value):
            raise ValueError(f"non-empty variant {name} must be finite")
    if variant.temporal_offset_s is not None and not np.isfinite(
        variant.temporal_offset_s
    ):
        raise ValueError("temporal_offset_s must be finite or None")


def _validate_bundle(bundle: VisualAuditBundle) -> None:
    if not isinstance(bundle, VisualAuditBundle):
        raise TypeError("bundle must be a VisualAuditBundle")
    if not bundle.event_id:
        raise ValueError("event_id must be non-empty")
    if not isinstance(bundle.grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    if bundle.grid.history_steps != 8 or bundle.grid.future_steps != 15:
        raise ValueError("visual audit requires history=8 and future=15")
    validate_base_state(bundle.base_state, bundle.grid)
    validate_oracle_context(bundle.oracle_context, bundle.grid)
    if bundle.oracle_context.base_state_id != bundle.base_state.state_id:
        raise ValueError("base_state and oracle_context IDs differ")
    _pose_array(
        bundle.trajectory.poses,
        name="trajectory.poses",
        expected_steps=15,
    )
    static = np.asarray(bundle.static_occupancy)
    if static.shape != (bundle.grid.height, bundle.grid.width):
        raise ValueError("static_occupancy shape does not match grid")
    if static.dtype.kind not in "biuf" or not np.isfinite(static).all():
        raise TypeError("static_occupancy must be finite numeric data")
    if len(bundle.variants) != len(PAIRED_PANEL_ORDER):
        raise ValueError("visual audit requires six variants")
    if (
        not np.isfinite(bundle.robot_length_m)
        or not np.isfinite(bundle.robot_width_m)
        or bundle.robot_length_m <= 0.0
        or bundle.robot_width_m <= 0.0
    ):
        raise ValueError("robot dimensions must be positive finite values")
    for index, variant in enumerate(bundle.variants):
        _validate_variant(variant, index=index)
    collision_visibility = bundle.variants[0].visibility_history
    assert collision_visibility is not None
    if not bool(collision_visibility.any()) or bool(
        collision_visibility[-2:].any()
    ):
        raise ValueError("collision history must be seen_then_occluded")


def _history_visibility_masks(bundle: VisualAuditBundle) -> tuple[np.ndarray, ...]:
    static = np.asarray(bundle.static_occupancy != 0, dtype=np.bool_)
    masks = []
    for history_index in range(bundle.grid.history_steps):
        occupied = static.copy()
        for object_id in sorted(bundle.oracle_context.dynamic_object_history):
            footprint = footprint_from_spec(
                bundle.oracle_context.dynamic_object_specs[object_id]
            )
            occupied |= rasterize_footprint(
                footprint,
                bundle.oracle_context.dynamic_object_history[object_id][
                    history_index
                ],
                bundle.grid,
            )
        masks.append(
            raycast_visibility(
                occupied,
                bundle.grid,
                sensor_pose=bundle.base_state.robot_history[history_index],
            )
        )
    return tuple(masks)


def _context_poses_at(
    trajectories: dict[str, np.ndarray],
    object_ids: tuple[str, ...],
    index: int,
) -> np.ndarray:
    if not object_ids:
        return np.empty((0, 3), dtype=np.float32)
    return np.stack(
        [trajectories[object_id][index] for object_id in object_ids],
        axis=0,
    ).astype(np.float32, copy=True)


def _context_visibility_at(
    object_ids: tuple[str, ...],
    history_index: int,
    bundle: VisualAuditBundle,
) -> np.ndarray:
    occupied = np.asarray(bundle.static_occupancy != 0, dtype=np.bool_).copy()
    object_masks: dict[str, np.ndarray] = {}
    for object_id in object_ids:
        footprint = footprint_from_spec(
            bundle.oracle_context.dynamic_object_specs[object_id]
        )
        object_masks[object_id] = rasterize_footprint(
            footprint,
            bundle.oracle_context.dynamic_object_history[object_id][history_index],
            bundle.grid,
        )
        occupied |= object_masks[object_id]
    values = np.zeros(len(object_ids), dtype=np.bool_)
    sensor_pose = bundle.base_state.robot_history[history_index]
    for index, object_id in enumerate(object_ids):
        candidate_mask = object_masks[object_id]
        values[index] = bool(
            raycast_candidate_visibility(
                occupied & ~candidate_mask,
                candidate_mask,
                bundle.grid,
                sensor_pose=sensor_pose,
            ).any()
        )
    return values


def build_replay_frames(bundle: VisualAuditBundle) -> tuple[ReplayFrame, ...]:
    """Build eight observed-history and fifteen oracle-replay visual frames."""

    _validate_bundle(bundle)
    collision = bundle.variants[0]
    assert collision.target_history is not None
    assert collision.target_future is not None
    assert collision.visibility_history is not None
    history_masks = _history_visibility_masks(bundle)
    context_object_ids = tuple(
        sorted(bundle.oracle_context.dynamic_object_history)
    )
    frames = []
    for index in range(8):
        context_poses = _context_poses_at(
            bundle.oracle_context.dynamic_object_history,
            context_object_ids,
            index,
        )
        frames.append(
            ReplayFrame(
                frame_index=index,
                time_s=REPLAY_HISTORY_TIMES_S[index],
                phase="observed_history",
                visibility_mask=history_masks[index].copy(),
                robot_pose=bundle.base_state.robot_history[index].copy(),
                target_pose=collision.target_history[index].copy(),
                target_visible=bool(collision.visibility_history[index]),
                context_object_ids=context_object_ids,
                context_poses=context_poses,
                context_visible=_context_visibility_at(
                    context_object_ids,
                    index,
                    bundle,
                ),
            )
        )
    current_visibility = history_masks[-1]
    for index in range(15):
        frames.append(
            ReplayFrame(
                frame_index=8 + index,
                time_s=REPLAY_FUTURE_TIMES_S[index],
                phase="oracle_replay",
                visibility_mask=current_visibility.copy(),
                robot_pose=bundle.trajectory.poses[index].copy(),
                target_pose=collision.target_future[index].copy(),
                target_visible=None,
                context_object_ids=context_object_ids,
                context_poses=_context_poses_at(
                    bundle.oracle_context.dynamic_object_future,
                    context_object_ids,
                    index,
                ),
                context_visible=None,
            )
        )
    return tuple(frames)


def _plot_modules():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.lines import Line2D
        from matplotlib.patches import Polygon
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError(
            "visual audit requires the optional visual-audit dependencies"
        ) from exc
    return plt, FigureCanvasAgg, Line2D, Polygon, Image


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rectangle_corners(
    pose: np.ndarray,
    *,
    length_m: float,
    width_m: float,
) -> np.ndarray:
    half_length = 0.5 * float(length_m)
    half_width = 0.5 * float(width_m)
    local = np.asarray(
        [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )
    cosine = float(np.cos(pose[2]))
    sine = float(np.sin(pose[2]))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + np.asarray(pose[:2], dtype=np.float64)


def _axis_limits(bundle: VisualAuditBundle) -> tuple[float, float, float, float]:
    points = [np.zeros((1, 2), dtype=np.float32), bundle.trajectory.poses[:, :2]]
    for variant in bundle.variants:
        if variant.target_history is not None:
            points.extend(
                [variant.target_history[:, :2], variant.target_future[:, :2]]
            )
    for occluder in bundle.occluders:
        pose = np.asarray(occluder["pose"], dtype=np.float64)
        points.append(
            _rectangle_corners(
                pose,
                length_m=float(occluder["length_m"]),
                width_m=float(occluder["width_m"]),
            )
        )
    joined = np.concatenate(points, axis=0).astype(np.float64)
    x_min, x_max, y_min, y_max = grid_bounds(bundle.grid)
    lower = joined.min(axis=0)
    upper = joined.max(axis=0)
    center = 0.5 * (lower + upper)
    span = np.maximum(upper - lower + 1.6, 5.5)
    left = max(x_min, float(center[0] - 0.5 * span[0]))
    right = min(x_max, float(center[0] + 0.5 * span[0]))
    bottom = max(y_min, float(center[1] - 0.5 * span[1]))
    top = min(y_max, float(center[1] + 0.5 * span[1]))
    return left, right, bottom, top


def _mask_rgba(
    mask: np.ndarray,
    color: tuple[float, float, float],
    alpha: float,
) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., :3] = color
    rgba[..., 3] = np.asarray(mask, dtype=np.float32) * np.float32(alpha)
    return rgba


def _draw_background(ax, bundle: VisualAuditBundle, visibility: np.ndarray) -> None:
    x_min, x_max, y_min, y_max = grid_bounds(bundle.grid)
    extent = (x_min, x_max, y_min, y_max)
    ax.imshow(
        _mask_rgba(~visibility, (0.86, 0.18, 0.20), 0.22),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=0,
    )
    ax.imshow(
        _mask_rgba(visibility, (0.18, 0.62, 0.78), 0.11),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=1,
    )
    ax.imshow(
        _mask_rgba(
            np.asarray(bundle.static_occupancy != 0, dtype=np.bool_),
            (0.20, 0.22, 0.24),
            0.88,
        ),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=2,
    )


def _draw_occluders(ax, bundle: VisualAuditBundle, Polygon) -> None:
    for occluder in bundle.occluders:
        pose = np.asarray(occluder["pose"], dtype=np.float64)
        corners = _rectangle_corners(
            pose,
            length_m=float(occluder["length_m"]),
            width_m=float(occluder["width_m"]),
        )
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                facecolor="#34383d",
                edgecolor="#111315",
                linewidth=1.8,
                zorder=5,
            )
        )


def _draw_robot(ax, bundle: VisualAuditBundle, pose: np.ndarray, Polygon) -> None:
    corners = _rectangle_corners(
        pose,
        length_m=bundle.robot_length_m,
        width_m=bundle.robot_width_m,
    )
    ax.add_patch(
        Polygon(
            corners,
            closed=True,
            facecolor="#111820",
            edgecolor="white",
            linewidth=1.3,
            zorder=12,
        )
    )
    heading = np.asarray(
        [np.cos(pose[2]), np.sin(pose[2])], dtype=np.float64
    )
    ax.plot(
        [pose[0], pose[0] + 0.45 * heading[0]],
        [pose[1], pose[1] + 0.45 * heading[1]],
        color="white",
        linewidth=1.5,
        zorder=13,
    )


def _draw_common(ax, bundle: VisualAuditBundle, visibility: np.ndarray, Polygon) -> None:
    _draw_background(ax, bundle, visibility)
    _draw_occluders(ax, bundle, Polygon)
    candidate = np.vstack(
        (np.zeros((1, 3), dtype=np.float32), bundle.trajectory.poses)
    )
    ax.plot(
        candidate[:, 0],
        candidate[:, 1],
        color="#008a8a",
        linewidth=4.2,
        solid_capstyle="round",
        zorder=8,
    )


def _draw_context_history(
    ax,
    frames: tuple[ReplayFrame, ...],
    *,
    history_index: int,
) -> None:
    frame = frames[history_index]
    if frame.context_visible is None:
        raise RuntimeError("observed context frame is missing visibility")
    for object_index, _ in enumerate(frame.context_object_ids):
        poses = np.stack(
            [
                frames[index].context_poses[object_index]
                for index in range(history_index + 1)
            ],
            axis=0,
        )
        visible = np.asarray(
            [
                bool(frames[index].context_visible[object_index])
                for index in range(history_index + 1)
            ],
            dtype=np.bool_,
        )
        for index in range(poses.shape[0] - 1):
            if visible[index] and visible[index + 1]:
                ax.plot(
                    poses[index : index + 2, 0],
                    poses[index : index + 2, 1],
                    color="#2d6a4f",
                    linewidth=1.7,
                    zorder=9,
                )
        previous = poses[:-1][visible[:-1]]
        if previous.size:
            ax.scatter(
                previous[:, 0],
                previous[:, 1],
                s=24,
                color="#2d6a4f",
                edgecolor="white",
                linewidth=0.5,
                alpha=0.72,
                zorder=9,
            )
        if visible[-1]:
            current = poses[-1]
            ax.scatter(
                [current[0]],
                [current[1]],
                s=48,
                color="#2d6a4f",
                edgecolor="white",
                linewidth=0.8,
                zorder=10,
            )


def _draw_context_oracle(
    ax,
    bundle: VisualAuditBundle,
    frame: ReplayFrame,
) -> None:
    for object_index, object_id in enumerate(frame.context_object_ids):
        path = np.vstack(
            (
                bundle.oracle_context.dynamic_object_history[object_id][-1],
                bundle.oracle_context.dynamic_object_future[object_id],
            )
        )
        ax.plot(
            path[:, 0],
            path[:, 1],
            color="#4f6d8a",
            linestyle="--",
            linewidth=1.7,
            alpha=0.9,
            zorder=9,
        )
        pose = frame.context_poses[object_index]
        ax.scatter(
            [pose[0]],
            [pose[1]],
            s=52,
            facecolor="none",
            edgecolor="#4f6d8a",
            linewidth=1.7,
            zorder=10,
        )


def _configure_axis(ax, limits: tuple[float, float, float, float]) -> None:
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(color="#9aa1a8", alpha=0.18, linewidth=0.6)


def _figure_image(fig, FigureCanvasAgg, Image):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba()).copy()
    return Image.fromarray(rgba, mode="RGBA").convert("RGB")


def _render_replay(
    bundle: VisualAuditBundle,
    path: Path,
) -> dict[str, object]:
    plt, FigureCanvasAgg, Line2D, Polygon, Image = _plot_modules()
    frames = build_replay_frames(bundle)
    collision = bundle.variants[0]
    assert collision.target_history is not None
    assert collision.target_future is not None
    assert collision.visibility_history is not None
    limits = _axis_limits(bundle)
    images = []
    for frame in frames:
        fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.90)
        _draw_common(ax, bundle, frame.visibility_mask, Polygon)
        if frame.frame_index < 8:
            history_index = frame.frame_index
            _draw_context_history(
                ax,
                frames,
                history_index=history_index,
            )
            past = collision.target_history[: history_index + 1]
            past_visibility = collision.visibility_history[: history_index + 1]
            if bool(past_visibility.any()):
                visible = past[past_visibility]
                ax.plot(
                    visible[:, 0],
                    visible[:, 1],
                    color="#b41583",
                    linewidth=2.0,
                    zorder=10,
                )
                ax.scatter(
                    visible[:, 0],
                    visible[:, 1],
                    s=42,
                    color="#d21b96",
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=11,
                )
            if bool((~past_visibility).any()):
                hidden = past[~past_visibility]
                ax.plot(
                    hidden[:, 0],
                    hidden[:, 1],
                    color="#d21b96",
                    linestyle="--",
                    linewidth=1.8,
                    zorder=10,
                )
                ax.scatter(
                    hidden[:, 0],
                    hidden[:, 1],
                    s=42,
                    facecolor="none",
                    edgecolor="#d21b96",
                    linewidth=1.4,
                    zorder=11,
                )
            phase_title = "observed history"
        else:
            _draw_context_oracle(ax, bundle, frame)
            ax.plot(
                collision.target_future[:, 0],
                collision.target_future[:, 1],
                color="#d21b96",
                linestyle="--",
                linewidth=2.5,
                zorder=10,
            )
            ax.scatter(
                [frame.target_pose[0]],
                [frame.target_pose[1]],
                s=64,
                facecolor="none",
                edgecolor="#d21b96",
                linewidth=1.8,
                zorder=11,
            )
            phase_title = "oracle replay (t=0 visibility frozen)"
        if frame.frame_index < 8:
            ax.plot(
                collision.target_future[:, 0],
                collision.target_future[:, 1],
                color="#d21b96",
                linestyle="--",
                linewidth=2.2,
                alpha=0.85,
                zorder=9,
            )
        _draw_robot(ax, bundle, frame.robot_pose, Polygon)
        _configure_axis(ax, limits)
        ax.set_title(
            f"{bundle.event_id} | {phase_title} | t={frame.time_s:+.1f} s",
            fontsize=14,
            pad=12,
        )
        legend = [
            Line2D([0], [0], color="#008a8a", linewidth=4.2, label="candidate trajectory"),
            Line2D([0], [0], color="#d21b96", linestyle="--", linewidth=2.2, label="target oracle future"),
            Line2D([0], [0], marker="s", color="#34383d", linestyle="none", label="represented occluder"),
            Line2D([0], [0], marker="o", markerfacecolor="#d21b96", markeredgecolor="white", color="none", label="visible target history"),
            Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="#d21b96", color="none", label="hidden target state"),
        ]
        if frame.context_object_ids:
            legend.extend(
                [
                    Line2D(
                        [0],
                        [0],
                        color="#2d6a4f",
                        marker="o",
                        linewidth=1.7,
                        label="visible context history",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="#4f6d8a",
                        marker="o",
                        markerfacecolor="none",
                        linestyle="--",
                        linewidth=1.7,
                        label="context oracle future",
                    ),
                ]
            )
        ax.legend(
            handles=legend,
            loc="upper right",
            fontsize=9,
            framealpha=0.92,
        )
        images.append(_figure_image(fig, FigureCanvasAgg, Image))
        plt.close(fig)
    images[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=250,
        loop=0,
        optimize=False,
        disposal=2,
    )
    with Image.open(path) as loaded:
        if loaded.n_frames != 23 or loaded.size != (1200, 900):
            raise RuntimeError("written replay GIF violates fixed image contract")
    context_object_ids = frames[0].context_object_ids
    moving_context_object_count = sum(
        not np.array_equal(
            bundle.oracle_context.dynamic_object_future[object_id],
            np.broadcast_to(
                bundle.oracle_context.dynamic_object_history[object_id][-1],
                bundle.oracle_context.dynamic_object_future[object_id].shape,
            ),
        )
        for object_id in context_object_ids
    )
    return {
        "format": "GIF",
        "width": 1200,
        "height": 900,
        "frame_count": 23,
        "frame_duration_ms": 250,
        "loop": 0,
        "context_object_count": len(context_object_ids),
        "moving_context_object_count": moving_context_object_count,
        "context_future_rendering": "oracle_only_animated",
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _variant_title(variant: VisualVariant) -> str:
    labels = {
        "collision": "Collision",
        "near_miss": "Near miss",
        "temporal_safe": "Temporal safe",
        "spatial_safe": "Spatial safe",
        "irrelevant_hidden": "Irrelevant hidden object",
        "empty_blind_spot": "Empty blind spot",
    }
    if variant.kind == "empty_blind_spot":
        return f"{labels[variant.kind]}\nTarget removed"
    title = f"{labels[variant.kind]}\nmin clearance {variant.min_clearance_m:+.3f} m"
    if variant.kind == "temporal_safe":
        title += f" | offset {variant.temporal_offset_s:+.1f} s"
    return title


def _render_paired(
    bundle: VisualAuditBundle,
    path: Path,
) -> dict[str, object]:
    plt, _, Line2D, Polygon, _ = _plot_modules()
    frames = build_replay_frames(bundle)
    current_frame = frames[7]
    current_visibility = current_frame.visibility_mask
    limits = _axis_limits(bundle)
    fig, axes = plt.subplots(2, 3, figsize=(21, 12), dpi=100)
    fig.subplots_adjust(
        left=0.045,
        right=0.985,
        bottom=0.095,
        top=0.89,
        wspace=0.16,
        hspace=0.24,
    )
    for ax, variant in zip(axes.flat, bundle.variants, strict=True):
        _draw_common(ax, bundle, current_visibility, Polygon)
        _draw_context_history(ax, frames, history_index=7)
        _draw_robot(
            ax,
            bundle,
            np.zeros(3, dtype=np.float32),
            Polygon,
        )
        if variant.target_future is not None:
            assert variant.target_history is not None
            ax.plot(
                variant.target_future[:, 0],
                variant.target_future[:, 1],
                color="#d21b96",
                linestyle="--",
                linewidth=2.4,
                zorder=10,
            )
            current = variant.target_history[-1]
            ax.scatter(
                [current[0]],
                [current[1]],
                s=72,
                facecolor="none",
                edgecolor="#d21b96",
                linewidth=1.8,
                zorder=11,
            )
        _configure_axis(ax, limits)
        ax.set_title(_variant_title(variant), fontsize=12, pad=8)
    handles = [
        Line2D([0], [0], color="#008a8a", linewidth=4.2, label="candidate trajectory"),
        Line2D([0], [0], color="#d21b96", linestyle="--", linewidth=2.4, label="target oracle future"),
        Line2D([0], [0], marker="s", color="#34383d", linestyle="none", label="represented occluder"),
        Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="#d21b96", color="none", label="hidden target at t=0"),
    ]
    if current_frame.context_object_ids:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#2d6a4f",
                marker="o",
                linewidth=1.7,
                label="visible context history",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.suptitle(
        f"{bundle.event_id}: shared base state, candidate and occluder skeleton",
        fontsize=17,
        y=0.992,
    )
    fig.text(
        0.5,
        0.035,
        "Controlled difference: target timing, placement, relevance or removal. "
        "Dashed target motion is oracle-only.",
        ha="center",
        fontsize=11,
    )
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    from PIL import Image

    with Image.open(path) as loaded:
        if loaded.size != (2100, 1200):
            raise RuntimeError("written paired PNG violates fixed image contract")
    return {
        "format": "PNG",
        "width": 2100,
        "height": 1200,
        "panel_order": list(PAIRED_PANEL_ORDER),
        "empty_target_removed": True,
        "context_object_count": len(current_frame.context_object_ids),
        "shared_axes": [float(value) for value in limits],
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def render_visual_artifacts(
    bundle: VisualAuditBundle,
    output_dir: str | Path,
) -> VisualArtifactResult:
    """Render one fixed replay GIF and one fixed paired-event PNG."""

    _validate_bundle(bundle)
    directory = Path(output_dir)
    if directory.exists() and not directory.is_dir():
        raise ValueError("visual output path exists and is not a directory")
    directory.mkdir(parents=True, exist_ok=True)
    replay_path = directory / "event_replay.gif"
    paired_path = directory / "paired_events.png"
    if replay_path.exists() or paired_path.exists():
        raise FileExistsError("refusing to overwrite visual audit artifacts")
    replay_metadata = _render_replay(bundle, replay_path)
    paired_metadata = _render_paired(bundle, paired_path)
    return VisualArtifactResult(
        event_replay_path=replay_path,
        paired_events_path=paired_path,
        event_replay_metadata=replay_metadata,
        paired_events_metadata=paired_metadata,
    )
