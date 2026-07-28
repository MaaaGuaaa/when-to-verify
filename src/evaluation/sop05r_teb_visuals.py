"""Deterministic, authenticated Long40 visual evidence for SOP05R TEB."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from src.contracts import GridSpec, build_grid_spec
from src.geometry.rasterization import grid_bounds
from src.generation.sop05r_contracts import (
    SOP05R_TEB_GENERATOR_VERSION,
    Sop05rTebConfig,
)
from src.generation.sop05r_teb_output_loader import LoadedSop05rTebOutput
from src.planning.verification_actions import (
    VerificationActionLibrary,
    sample_state_aware_action_trace,
)


SOP05R_TEB_VISUAL_VERSION = "sop05r_teb_visual_v2"


@dataclass(frozen=True)
class Sop05rTebVerificationTrace:
    action_id: str
    poses: np.ndarray


@dataclass(frozen=True)
class Sop05rTebVisualBundle:
    version: str
    event_id: str
    publication_semantic_digest: str
    grid: GridSpec
    static_occupancy: np.ndarray
    occluders: tuple[Mapping[str, object], ...]
    robot_history: np.ndarray
    full_route: np.ndarray
    nominal_suffix: np.ndarray
    shared_goal_pose: np.ndarray
    target_long40: np.ndarray | None
    target_visibility_history: np.ndarray | None
    collision_point_xy: np.ndarray | None
    collision_time_s: float | None
    witness_sample_index: int | None
    witness_occluder_id: str | None
    verification_traces: tuple[Sop05rTebVerificationTrace, ...]


@dataclass(frozen=True)
class Sop05rTebVisualArtifact:
    path: Path
    metadata: Mapping[str, object]


def _as_pose_array(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return np.asarray(array, dtype=np.float64)


def _to_decision_frame(poses: np.ndarray, decision_pose: np.ndarray) -> np.ndarray:
    cosine = float(np.cos(decision_pose[2]))
    sine = float(np.sin(decision_pose[2]))
    inverse_rotation = np.asarray(((cosine, sine), (-sine, cosine)))
    result = np.empty_like(poses, dtype=np.float64)
    result[:, :2] = (poses[:, :2] - decision_pose[:2]) @ inverse_rotation.T
    result[:, 2] = poses[:, 2] - decision_pose[2]
    return result


def _find_event(source: LoadedSop05rTebOutput, event_id: str):
    if not isinstance(source, LoadedSop05rTebOutput):
        raise TypeError("source must be a LoadedSop05rTebOutput")
    if not source.complete:
        raise ValueError("TEB visuals require a complete M7 collection")
    if source.manifest.get("generator_algorithm_version") != SOP05R_TEB_GENERATOR_VERSION:
        raise ValueError("TEB visuals reject a non-current generator collection")
    matches = [event for event in source.events if event.generated_event_id == event_id]
    if len(matches) != 1:
        raise ValueError("TEB visuals require exactly one source event")
    records = [record for record in source.trajectories.records if record.event_id == event_id]
    if len(records) != 1:
        raise ValueError("TEB visuals require exactly one source trajectory")
    return matches[0], records[0]


def build_sop05r_teb_visual_bundle(
    source: LoadedSop05rTebOutput,
    *,
    event_id: str,
    teb_config: Sop05rTebConfig,
    action_library: VerificationActionLibrary,
) -> Sop05rTebVisualBundle:
    """Build one fixed-coordinate visual bundle using strict M7 evidence only."""

    if not isinstance(teb_config, Sop05rTebConfig):
        raise TypeError("teb_config must be a Sop05rTebConfig")
    if not isinstance(action_library, VerificationActionLibrary):
        raise TypeError("action_library must be a VerificationActionLibrary")
    event, record = _find_event(source, event_id)
    if source.manifest.get("config_digest") != teb_config.digest:
        raise ValueError("visual TEB config digest differs from the source collection")
    state = source.decision_states.get(record.decision_state_id)
    if state is None or state.state_id != record.decision_state_id:
        raise ValueError("visual decision state is missing or mismatched")
    grid = build_grid_spec(dict(source.manifest["base_config"]))
    evidence = source.event_evidence.get(event_id)
    if evidence is None:
        raise ValueError("visual event evidence is missing")
    route_world = _as_pose_array(
        record.full_route.sampled_poses_world,
        name="full route",
        shape=(40, 3),
    )
    route_start = _as_pose_array(
        record.full_route.band_poses_world[0],
        name="full route start",
        shape=(3,),
    )
    full_route = _to_decision_frame(route_world, route_start)
    nominal_suffix = _as_pose_array(
        record.nominal_trajectory.poses,
        name="nominal suffix",
        shape=(32, 3),
    )
    if not np.allclose(full_route[:32], nominal_suffix, rtol=0.0, atol=1e-5):
        raise ValueError("full route does not reproduce the decision suffix")
    goal = _to_decision_frame(
        _as_pose_array(
            record.shared_goal_world_pose,
            name="shared goal",
            shape=(3,),
        )[None, :],
        route_start,
    )[0]
    target_long40 = np.vstack((event.target.history_poses, event.target.future_poses))
    target_long40 = _as_pose_array(
        target_long40,
        name="Long40 target",
        shape=(40, 3),
    )
    visibility = np.asarray(event.target_visibility_history, dtype=np.bool_)
    if visibility.shape != (8,):
        raise ValueError("visual target history visibility must have eight samples")
    collision_point = _as_pose_array(
        evidence["collision_point_xy"],
        name="collision point",
        shape=(2,),
    )
    collision_time = float(evidence["first_collision_time_after_decision_s"])
    witness = evidence["occlusion_witness"]
    if not isinstance(witness, Mapping):
        raise ValueError("visual occlusion witness is invalid")
    witness_index = int(witness["sample_index"])
    if not 0 <= witness_index < 8:
        raise ValueError("visual witness sample index is invalid")
    traces = tuple(
        Sop05rTebVerificationTrace(
            action_id=action.action_id,
            poses=np.asarray(
                sample_state_aware_action_trace(
                    state.robot_history[-1],
                    action,
                    robot_state=state.robot_state,
                    braking_deceleration_mps2=(
                        teb_config.planner.max_linear_acceleration_mps2
                    ),
                    angular_deceleration_radps2=(
                        teb_config.planner.max_angular_acceleration_radps2
                    ),
                ).poses,
                dtype=np.float64,
            ),
        )
        for action in action_library.actions
    )
    return Sop05rTebVisualBundle(
        version=SOP05R_TEB_VISUAL_VERSION,
        event_id=event_id,
        publication_semantic_digest=source.publication_semantic_digest,
        grid=grid,
        static_occupancy=np.asarray(event.world.static_occupancy, dtype=np.float64),
        occluders=tuple(dict(item) for item in event.world.occluders),
        robot_history=_as_pose_array(
            state.robot_history,
            name="robot history",
            shape=(8, 3),
        ),
        full_route=full_route,
        nominal_suffix=nominal_suffix,
        shared_goal_pose=goal,
        target_long40=target_long40,
        target_visibility_history=visibility,
        collision_point_xy=collision_point,
        collision_time_s=collision_time,
        witness_sample_index=witness_index,
        witness_occluder_id=str(witness["occluder_id"]),
        verification_traces=traces,
    )


def _rectangle_corners(pose: np.ndarray, *, length_m: float, width_m: float) -> np.ndarray:
    half_length = 0.5 * length_m
    half_width = 0.5 * width_m
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
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    return local @ rotation.T + pose[:2]


def _draw_occluders(ax, bundle: Sop05rTebVisualBundle, Polygon, Circle) -> None:
    for index, raw in enumerate(bundle.occluders, start=1):
        item = dict(raw)
        shape = item.get("shape")
        if shape == "circle":
            center = np.asarray(item["center_xy"], dtype=np.float64)
            patch = Circle(
                center,
                radius=float(item["radius_m"]),
                facecolor="#34383d",
                edgecolor="#101216",
                linewidth=1.4,
                zorder=5,
            )
            label_at = center
        elif shape == "rectangle":
            corners = _rectangle_corners(
                np.asarray(item["pose"], dtype=np.float64),
                length_m=float(item["length_m"]),
                width_m=float(item["width_m"]),
            )
            patch = Polygon(
                corners,
                closed=True,
                facecolor="#34383d",
                edgecolor="#101216",
                linewidth=1.4,
                zorder=5,
            )
            label_at = corners.mean(axis=0)
        else:
            raise ValueError("visual encountered an unsupported occluder shape")
        ax.add_patch(patch)
        label = f"O{index}"
        if str(item.get("occluder_id")) == bundle.witness_occluder_id:
            label += " (W)"
        ax.text(
            float(label_at[0]),
            float(label_at[1]),
            label,
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            zorder=6,
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _draw_travel_direction(
    axis: object,
    poses: np.ndarray,
    *,
    color: str,
    label: str,
    zorder: int,
) -> None:
    points = np.asarray(poses, dtype=np.float64)
    displacements = np.diff(points[:, :2], axis=0)
    lengths = np.linalg.norm(displacements, axis=1)
    nonstationary = np.flatnonzero(lengths > 1e-6)
    if nonstationary.size == 0:
        return
    segment_index = int(nonstationary[nonstationary.size // 2])
    start = points[segment_index, :2]
    end = points[segment_index + 1, :2]
    axis.annotate(
        "",
        xy=(float(end[0]), float(end[1])),
        xytext=(float(start[0]), float(start[1])),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 2.4,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
            "mutation_scale": 14.0,
        },
        zorder=zorder,
    )
    axis.annotate(
        label,
        xy=(float(end[0]), float(end[1])),
        xytext=(5, 5),
        textcoords="offset points",
        color=color,
        fontsize=7,
        fontweight="bold",
        zorder=zorder + 1,
    )


def render_sop05r_teb_visual_bundle(
    bundle: Sop05rTebVisualBundle,
    output_path: str | Path,
) -> Sop05rTebVisualArtifact:
    """Render a fixed-scale PNG without overwriting a prior audit artifact."""

    if not isinstance(bundle, Sop05rTebVisualBundle):
        raise TypeError("bundle must be a Sop05rTebVisualBundle")
    path = Path(output_path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite visual artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch, Polygon
    from PIL import Image

    x_min, x_max, y_min, y_max = grid_bounds(bundle.grid)
    figure, axis = plt.subplots(figsize=(12, 10), dpi=120)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.89)
    axis.imshow(
        np.asarray(bundle.static_occupancy != 0, dtype=np.float32),
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        interpolation="nearest",
        cmap="Greys",
        alpha=0.45,
        zorder=0,
    )
    _draw_occluders(axis, bundle, Polygon, Circle)
    axis.plot(
        bundle.full_route[:, 0],
        bundle.full_route[:, 1],
        color="#406b8e",
        linewidth=2.0,
        zorder=7,
        label="8.0 s full TEB route (40 endpoints)",
    )
    axis.plot(
        bundle.nominal_suffix[:, 0],
        bundle.nominal_suffix[:, 1],
        color="#008a8a",
        linewidth=3.2,
        zorder=8,
        label="6.4 s decision suffix (32 endpoints)",
    )
    axis.plot(
        bundle.robot_history[:, 0],
        bundle.robot_history[:, 1],
        color="#406b8e",
        linestyle="--",
        linewidth=1.5,
        marker=".",
        zorder=8,
        label="robot history",
    )
    _draw_travel_direction(
        axis,
        bundle.nominal_suffix,
        color="#008a8a",
        label="robot direction",
        zorder=11,
    )
    pedestrian_present = bundle.target_long40 is not None
    if pedestrian_present:
        if bundle.target_visibility_history is None:
            raise ValueError("present pedestrian requires history visibility")
        target_history = _as_pose_array(
            bundle.target_long40[:8], name="target history", shape=(8, 3)
        )
        target_future = _as_pose_array(
            bundle.target_long40[8:], name="target future", shape=(32, 3)
        )
        visibility = np.asarray(bundle.target_visibility_history, dtype=np.bool_)
        if visibility.shape != (8,):
            raise ValueError("target history visibility must have eight samples")
        axis.plot(
            bundle.target_long40[:, 0],
            bundle.target_long40[:, 1],
            color="#d81b60",
            linestyle="--",
            linewidth=1.7,
            zorder=9,
            label="all 40 human samples",
        )
        axis.scatter(
            target_future[:, 0],
            target_future[:, 1],
            color="#d81b60",
            s=10,
            zorder=10,
        )
        _draw_travel_direction(
            axis,
            bundle.target_long40,
            color="#d81b60",
            label="pedestrian direction",
            zorder=14,
        )
        history_offsets = (
            (-30, -20),
            (-38, -2),
            (-29, 20),
            (-2, 30),
            (25, 20),
            (35, 2),
            (25, -20),
            (0, -31),
        )
        for index, pose in enumerate(target_history):
            visible = bool(visibility[index])
            color = "#d81b60" if visible else "#9aa0a6"
            axis.scatter(
                [pose[0]],
                [pose[1]],
                s=44 if index == 7 else 34,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=12,
            )
            axis.annotate(
                f"H{index}" + (" (D)" if index == 7 else ""),
                (pose[0], pose[1]),
                xytext=history_offsets[index],
                textcoords="offset points",
                fontsize=7,
                color="#17191c",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.7},
                zorder=13,
            )
    else:
        target_history = None
        axis.text(
            0.02,
            0.98,
            "no pedestrian in generated branch",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#6b1b1b",
            bbox={"facecolor": "white", "edgecolor": "#6b1b1b", "alpha": 0.9, "pad": 2.5},
            zorder=16,
        )
    has_witness = (
        pedestrian_present
        and target_history is not None
        and bundle.witness_sample_index is not None
    )
    if has_witness:
        witness_robot = bundle.robot_history[bundle.witness_sample_index, :2]
        witness_target = target_history[bundle.witness_sample_index, :2]
        axis.plot(
            [witness_robot[0], witness_target[0]],
            [witness_robot[1], witness_target[1]],
            color="#673ab7",
            linestyle=":",
            linewidth=2.1,
            zorder=11,
            label="persisted centerline occlusion witness",
        )
        axis.scatter(
            [witness_robot[0]],
            [witness_robot[1]],
            marker="P",
            s=55,
            color="#673ab7",
            edgecolor="white",
            linewidth=0.7,
            zorder=13,
        )
    axis.scatter(
        [bundle.shared_goal_pose[0]],
        [bundle.shared_goal_pose[1]],
        marker="*",
        s=165,
        color="#1b5e20",
        edgecolor="white",
        linewidth=0.8,
        zorder=14,
        label="shared goal",
    )
    has_collision = (
        bundle.collision_point_xy is not None and bundle.collision_time_s is not None
    )
    if has_collision:
        axis.scatter(
            [bundle.collision_point_xy[0]],
            [bundle.collision_point_xy[1]],
            marker="X",
            s=100,
            color="#e17c00",
            edgecolor="white",
            linewidth=0.9,
            zorder=15,
            label="continuous collision anchor",
        )
        axis.annotate(
            f"collision {bundle.collision_time_s:.2f} s",
            bundle.collision_point_xy,
            xytext=(7, -13),
            textcoords="offset points",
            fontsize=8,
            color="#8f4d00",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
            zorder=15,
        )
    trace_colors = ("#ef6c00", "#6a1b9a", "#1565c0", "#2e7d32", "#ad1457", "#455a64")
    action_label_offsets = (
        (-30, 16),
        (8, -18),
        (-36, 28),
        (10, -30),
        (10, 18),
        (-36, -14),
    )
    if len(bundle.verification_traces) not in (0, len(trace_colors)):
        raise ValueError("verification traces must be absent or match the action palette")
    if bundle.verification_traces:
        for trace, color in zip(bundle.verification_traces, trace_colors, strict=True):
            axis.plot(
                trace.poses[:, 0],
                trace.poses[:, 1],
                color=color,
                linestyle="--",
                linewidth=2.4,
                alpha=0.95,
                zorder=16,
            )
            axis.scatter(
                [trace.poses[-1, 0]],
                [trace.poses[-1, 1]],
                s=34,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=17,
            )
    if bundle.verification_traces:
        action_inset = axis.inset_axes((0.035, 0.035, 0.42, 0.275))
        action_inset.imshow(
            np.asarray(bundle.static_occupancy != 0, dtype=np.float32),
            origin="lower",
            extent=(x_min, x_max, y_min, y_max),
            interpolation="nearest",
            cmap="Greys",
            alpha=0.4,
            zorder=0,
        )
        _draw_occluders(action_inset, bundle, Polygon, Circle)
        action_inset.scatter(
            [0.0], [0.0], marker="+", s=90, color="#101216", linewidths=1.8, zorder=18
        )
        for index, (trace, color, offset) in enumerate(
            zip(
                bundle.verification_traces,
                trace_colors,
                action_label_offsets,
                strict=True,
            ),
            start=1,
        ):
            action_inset.plot(
                trace.poses[:, 0],
                trace.poses[:, 1],
                color=color,
                linestyle="--",
                linewidth=3.0,
                alpha=1.0,
                zorder=16,
            )
            endpoint = trace.poses[-1, :2]
            action_inset.scatter(
                [endpoint[0]],
                [endpoint[1]],
                s=46,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=17,
            )
            action_inset.annotate(
                f"A{index}",
                endpoint,
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color=color,
                bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.96, "pad": 0.7},
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.0},
                zorder=19,
            )
        action_inset.set_xlim(-0.30, 0.75)
        action_inset.set_ylim(-0.45, 0.45)
        action_inset.set_aspect("equal", adjustable="box")
        action_inset.grid(alpha=0.25, linewidth=0.4)
        action_inset.tick_params(labelsize=6)
        action_inset.set_title("verification actions — decision zoom", fontsize=8, fontweight="bold")
        axis.indicate_inset_zoom(action_inset, edgecolor="#101216", alpha=0.65)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.16, linewidth=0.45)
    axis.set_xlabel("decision-frame x (m)")
    axis.set_ylabel("decision-frame y (m)")
    axis.set_title(
        f"{bundle.event_id}\nH0 = −1.4 s … H7 = 0.0 s decision time; fixed {x_max - x_min:.0f} m BEV",
        fontsize=12,
    )
    handles = [
        Line2D([0], [0], color="#406b8e", linewidth=2.0, label="full TEB route"),
        Line2D([0], [0], color="#008a8a", linewidth=3.2, label="nominal suffix"),
        Patch(facecolor="#34383d", edgecolor="#101216", label="represented occluder"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#1b5e20", label="goal"),
    ]
    if pedestrian_present:
        handles.extend(
            (
                Line2D([0], [0], color="#d81b60", linestyle="--", label="40-sample human motion"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="#d81b60", markeredgecolor="white", label="visible H0-H7"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="#9aa0a6", markeredgecolor="white", label="occluded H0-H7"),
            )
        )
    if has_witness:
        handles.append(
            Line2D([0], [0], color="#673ab7", linestyle=":", label="occlusion witness")
        )
    if has_collision:
        handles.append(
            Line2D([0], [0], marker="X", color="none", markerfacecolor="#e17c00", label="collision")
        )
    if bundle.verification_traces:
        for trace, color in zip(bundle.verification_traces, trace_colors, strict=True):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle="--",
                    label=f"verification action: {trace.action_id}",
                )
            )
    axis.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.92)
    figure.savefig(path, dpi=120, facecolor="white")
    plt.close(figure)
    with Image.open(path) as image:
        if image.size != (1440, 1200):
            raise RuntimeError("written TEB audit image violates the fixed image contract")
        if np.asarray(image.convert("RGB")).std() <= 5.0:
            raise RuntimeError("written TEB audit image is blank")
    metadata = {
        "version": SOP05R_TEB_VISUAL_VERSION,
        "event_id": bundle.event_id,
        "publication_semantic_digest": bundle.publication_semantic_digest,
        "width": 1440,
        "height": 1200,
        "full_route_endpoint_count": int(bundle.full_route.shape[0]),
        "nominal_suffix_endpoint_count": int(bundle.nominal_suffix.shape[0]),
        "human_sample_count": 40 if pedestrian_present else 0,
        "pedestrian_present": pedestrian_present,
        "direction_annotations": ["robot"] + (["pedestrian"] if pedestrian_present else []),
        "history_frame_definition": "H0=-1.4s through H7=0.0s decision time",
        "occluder_count": len(bundle.occluders),
        "verification_action_ids": [
            trace.action_id for trace in bundle.verification_traces
        ],
        "verification_action_line_style": "dashed",
        "verification_action_zoom": "decision_frame_-0.30_to_0.75m_x_-0.45_to_0.45m_y",
        "sha256": _sha256_file(path),
    }
    return Sop05rTebVisualArtifact(path=path, metadata=metadata)
