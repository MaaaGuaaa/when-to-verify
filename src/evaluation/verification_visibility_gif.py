"""Offline GIF audit for time-aligned verification-action visibility."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from src.contracts import GridSpec
from src.geometry import Footprint, grid_bounds, rasterize_footprint
from src.generation.counterfactual_verify import (
    CounterfactualObservationTrace,
    simulate_counterfactual_observation_trace,
)
from src.generation.dynamic_object_transplant import footprint_from_spec
from src.planning.verification_actions import (
    ActionTrace,
    VerificationAction,
    check_action_trace_feasibility,
    sample_state_aware_action_trace,
)


VERIFICATION_VISIBILITY_GIF_VERSION = "verification_visibility_gif_v1"
_IMAGE_SIZE = (800, 800)
_MIN_DISPLAY_FRAME_INTERVAL_S = 0.01


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VerificationVisibilityIneligibleError(ValueError):
    """One scene-action pair cannot be rendered as a valid audit."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = _identifier(reason, name="reason")
        self.detail = _identifier(detail, name="detail")
        super().__init__(f"{self.reason}: {self.detail}")


@dataclass(frozen=True)
class VerificationVisibilityGifCase:
    sample_id: str
    action_id: str
    grid: GridSpec
    static_occupancy: np.ndarray
    action_trace: ActionTrace
    observation_trace: CounterfactualObservationTrace

    def __post_init__(self) -> None:
        _identifier(self.sample_id, name="sample_id")
        _identifier(self.action_id, name="action_id")
        if not isinstance(self.grid, GridSpec):
            raise TypeError("grid must be a GridSpec")
        if not isinstance(self.action_trace, ActionTrace):
            raise TypeError("action_trace must be an ActionTrace")
        if not isinstance(
            self.observation_trace,
            CounterfactualObservationTrace,
        ):
            raise TypeError(
                "observation_trace must be a CounterfactualObservationTrace"
            )
        static = np.asarray(self.static_occupancy)
        if (
            static.dtype != np.bool_
            or static.shape != (self.grid.height, self.grid.width)
        ):
            raise ValueError("static_occupancy must be a bool grid")
        if (
            len(self.observation_trace.frames)
            != self.action_trace.times_s.size
            or not np.array_equal(
                self.observation_trace.times_s,
                self.action_trace.times_s,
            )
        ):
            raise ValueError(
                "observation frames must align with the action trace"
            )
        shape = (self.grid.height, self.grid.width)
        if any(
            frame.visible_mask.shape != shape
            for frame in self.observation_trace.frames
        ):
            raise ValueError("observation frame shape differs from the grid")
        copied = np.array(static, dtype=np.bool_, order="C", copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "static_occupancy", copied)


@dataclass(frozen=True)
class VerificationVisibilityGifArtifact:
    output_dir: Path
    gif_path: Path
    manifest_path: Path
    metadata: Mapping[str, object]


def build_verification_visibility_case(
    *,
    sample_id: str,
    action: VerificationAction,
    grid: GridSpec,
    robot_pose: np.ndarray,
    robot_state: np.ndarray,
    robot_footprint: Footprint,
    static_occupancy: np.ndarray,
    dynamic_current_poses: Mapping[str, np.ndarray],
    dynamic_future_poses: Mapping[str, np.ndarray],
    dynamic_specs: Mapping[str, dict[str, object]],
    current_visible_mask: np.ndarray,
    current_age_map: np.ndarray,
    future_dt_s: float,
    age_max_s: float,
    fov_rad: float,
    max_range_m: float,
    braking_deceleration_mps2: float,
    angular_deceleration_radps2: float,
) -> VerificationVisibilityGifCase:
    """Build one feasible trace and its per-frame counterfactual visibility."""

    if not isinstance(action, VerificationAction):
        raise TypeError("action must be a VerificationAction")
    trace = sample_state_aware_action_trace(
        robot_pose,
        action,
        robot_state=robot_state,
        braking_deceleration_mps2=braking_deceleration_mps2,
        angular_deceleration_radps2=angular_deceleration_radps2,
    )
    footprints = {
        object_id: footprint_from_spec(dynamic_specs[object_id])
        for object_id in sorted(dynamic_specs)
    }
    visible_ids = []
    for object_id, footprint in footprints.items():
        occupied = rasterize_footprint(
            footprint,
            dynamic_current_poses[object_id],
            grid,
        )
        if np.any(occupied & current_visible_mask):
            visible_ids.append(object_id)
    dynamic_poses = {
        object_id: np.vstack(
            (
                dynamic_current_poses[object_id][None, :],
                dynamic_future_poses[object_id],
            )
        ).astype(np.float32)
        for object_id in visible_ids
    }
    feasibility = check_action_trace_feasibility(
        trace,
        robot_footprint=robot_footprint,
        static_occupancy=static_occupancy,
        grid=grid,
        dynamic_object_poses=dynamic_poses,
        dynamic_object_footprints={
            object_id: footprints[object_id] for object_id in visible_ids
        },
        dynamic_dt_s=future_dt_s,
    )
    if not feasibility.feasible:
        detail = feasibility.reason or "unknown collision"
        if feasibility.critical_object_id is not None:
            detail = f"{detail} with {feasibility.critical_object_id}"
        raise VerificationVisibilityIneligibleError(
            "action_infeasible",
            detail,
        )
    observation_trace = simulate_counterfactual_observation_trace(
        action_trace=trace,
        static_occupancy=static_occupancy,
        dynamic_current_poses=dynamic_current_poses,
        dynamic_future_poses=dynamic_future_poses,
        dynamic_specs=dynamic_specs,
        current_visible_mask=current_visible_mask,
        current_age_map=current_age_map,
        grid=grid,
        future_dt_s=future_dt_s,
        age_max_s=age_max_s,
        fov_rad=fov_rad,
        max_range_m=max_range_m,
    )
    return VerificationVisibilityGifCase(
        sample_id=sample_id,
        action_id=action.action_id,
        grid=grid,
        static_occupancy=np.asarray(static_occupancy != 0.0),
        action_trace=trace,
        observation_trace=observation_trace,
    )


def _plot_modules():
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from PIL import Image

    return plt, FigureCanvasAgg, Line2D, Patch, Image


def _mask_rgba(
    mask: np.ndarray,
    color: tuple[float, float, float],
    alpha: float,
) -> np.ndarray:
    image = np.zeros((*mask.shape, 4), dtype=np.float32)
    image[..., :3] = color
    image[..., 3] = np.asarray(mask, dtype=np.float32) * np.float32(alpha)
    return image


def _figure_image(figure, FigureCanvasAgg, Image):
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    width, height = canvas.get_width_height()
    return Image.frombytes("RGB", (width, height), canvas.tostring_rgb())


def _draw_frame(
    axis,
    case: VerificationVisibilityGifCase,
    *,
    frame_index: int,
) -> None:
    frame = case.observation_trace.frames[frame_index]
    x_min, x_max, y_min, y_max = grid_bounds(case.grid)
    extent = (x_min, x_max, y_min, y_max)
    axis.set_facecolor("#f3f3f3")
    axis.imshow(
        _mask_rgba(frame.visible_mask, (0.00, 0.45, 0.70), 0.28),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=0,
    )
    axis.imshow(
        _mask_rgba(frame.newly_visible_mask, (0.90, 0.40, 0.00), 0.62),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=1,
    )
    axis.imshow(
        _mask_rgba(case.static_occupancy, (0.16, 0.16, 0.16), 0.96),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=2,
    )
    axis.imshow(
        _mask_rgba(
            frame.visible_dynamic_occupancy,
            (0.80, 0.10, 0.55),
            0.92,
        ),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=3,
    )

    executed = case.action_trace.poses[: frame_index + 1]
    axis.plot(
        executed[:, 0],
        executed[:, 1],
        color="#101010",
        linewidth=4.4,
        solid_capstyle="round",
        zorder=5,
    )
    axis.plot(
        executed[:, 0],
        executed[:, 1],
        color="white",
        linewidth=2.2,
        solid_capstyle="round",
        zorder=6,
    )
    pose = executed[-1]
    axis.scatter(
        [pose[0]],
        [pose[1]],
        marker="o",
        s=66,
        facecolor="white",
        edgecolor="#101010",
        linewidth=1.5,
        zorder=8,
    )
    heading_length = max(0.35, 3.5 * case.grid.resolution_m)
    dx = heading_length * np.cos(pose[2])
    dy = heading_length * np.sin(pose[2])
    axis.arrow(
        pose[0],
        pose[1],
        dx,
        dy,
        color="#101010",
        width=0.035,
        head_width=0.20,
        length_includes_head=True,
        zorder=7,
    )
    axis.arrow(
        pose[0],
        pose[1],
        dx,
        dy,
        color="white",
        width=0.018,
        head_width=0.13,
        length_includes_head=True,
        zorder=8,
    )

    time_s = float(case.action_trace.times_s[frame_index])
    visible_count = int(frame.visible_mask.sum())
    new_count = int(frame.newly_visible_mask.sum())
    axis.set_title(
        f"{case.action_id}  |  t = {time_s:.2f} s\n"
        f"visible cells = {visible_count:,}  |  newly visible = {new_count:,}",
        fontsize=11,
        fontweight="bold",
    )
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.16, linewidth=0.4)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")


def _frame_metrics(
    case: VerificationVisibilityGifCase,
    frame_indices: tuple[int, ...],
) -> list[dict[str, object]]:
    initial_visible = case.observation_trace.frames[0].visible_mask
    cumulative_new = np.zeros_like(initial_visible)
    result: list[dict[str, object]] = []
    for frame_index in frame_indices:
        time_s = case.observation_trace.times_s[frame_index]
        frame = case.observation_trace.frames[frame_index]
        cumulative_new |= frame.newly_visible_mask
        result.append(
            {
                "time_s": float(time_s),
                "visible_cell_count": int(frame.visible_mask.sum()),
                "newly_visible_cell_count": int(
                    frame.newly_visible_mask.sum()
                ),
                "cumulative_newly_visible_cell_count": int(
                    cumulative_new.sum()
                ),
                "no_longer_visible_cell_count": int(
                    (initial_visible & ~frame.visible_mask).sum()
                ),
                "visible_dynamic_cell_count": int(
                    frame.visible_dynamic_occupancy.sum()
                ),
            }
        )
    return result


def _display_frame_indices(times_s: np.ndarray) -> tuple[int, ...]:
    indices = [0]
    for index in range(1, times_s.size - 1):
        if (
            float(times_s[index] - times_s[indices[-1]])
            >= _MIN_DISPLAY_FRAME_INTERVAL_S
        ):
            indices.append(index)
    final_index = int(times_s.size - 1)
    if indices[-1] != final_index:
        indices.append(final_index)
    return tuple(indices)


def render_verification_visibility_gif(
    case: VerificationVisibilityGifCase,
    output_dir: str | Path,
    *,
    frame_duration_ms: int = 120,
) -> VerificationVisibilityGifArtifact:
    """Atomically render one immutable visibility-progression audit."""

    if not isinstance(case, VerificationVisibilityGifCase):
        raise TypeError("case must be a VerificationVisibilityGifCase")
    if (
        isinstance(frame_duration_ms, bool)
        or not isinstance(frame_duration_ms, int)
        or frame_duration_ms <= 0
    ):
        raise ValueError("frame_duration_ms must be a positive integer")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite visibility audit: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    gif_path = staging / "visibility_progress.gif"
    manifest_path = staging / "manifest.json"
    plt, FigureCanvasAgg, Line2D, Patch, Image = _plot_modules()
    try:
        frame_indices = _display_frame_indices(case.action_trace.times_s)
        frames = []
        for frame_index in frame_indices:
            figure, axis = plt.subplots(figsize=(8, 8), dpi=100)
            figure.subplots_adjust(
                left=0.10,
                right=0.98,
                bottom=0.14,
                top=0.86,
            )
            _draw_frame(axis, case, frame_index=frame_index)
            figure.suptitle(
                f"Verification visibility audit: {case.sample_id}",
                fontsize=13,
                fontweight="bold",
            )
            figure.legend(
                handles=(
                    Patch(
                        facecolor="#0072B2",
                        alpha=0.45,
                        label="Currently visible",
                    ),
                    Patch(
                        facecolor="#E66B00",
                        alpha=0.75,
                        label="New since action start",
                    ),
                    Patch(facecolor="#292929", label="Static obstacle"),
                    Patch(facecolor="#CC197F", label="Visible dynamic"),
                    Line2D(
                        [0],
                        [0],
                        color="white",
                        marker="o",
                        markeredgecolor="#101010",
                        linewidth=2.2,
                        label="Executed robot path",
                    ),
                ),
                loc="lower center",
                ncol=3,
                fontsize=8,
                framealpha=0.96,
            )
            frames.append(_figure_image(figure, FigureCanvasAgg, Image))
            plt.close(figure)
        frames[0].save(
            gif_path,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        with Image.open(gif_path) as image:
            if (
                image.size != _IMAGE_SIZE
                or image.n_frames != len(frames)
                or image.format != "GIF"
            ):
                raise RuntimeError(
                    "visibility GIF violates the fixed image contract"
                )
        metrics = _frame_metrics(case, frame_indices)
        metadata: dict[str, object] = {
            "version": VERIFICATION_VISIBILITY_GIF_VERSION,
            "sample_id": case.sample_id,
            "action_id": case.action_id,
            "frame_count": len(frames),
            "source_trace_frame_count": len(
                case.observation_trace.frames
            ),
            "rendered_frame_indices": list(frame_indices),
            "frame_duration_ms": frame_duration_ms,
            "width": _IMAGE_SIZE[0],
            "height": _IMAGE_SIZE[1],
            "times_s": [
                float(case.action_trace.times_s[index])
                for index in frame_indices
            ],
            "frame_metrics": metrics,
            "gif_sha256": _sha256(gif_path),
        }
        manifest_path.write_text(
            json.dumps(
                metadata,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        os.rename(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return VerificationVisibilityGifArtifact(
        output_dir=destination,
        gif_path=destination / gif_path.name,
        manifest_path=destination / manifest_path.name,
        metadata=MappingProxyType(metadata),
    )


__all__ = (
    "VERIFICATION_VISIBILITY_GIF_VERSION",
    "VerificationVisibilityGifArtifact",
    "VerificationVisibilityGifCase",
    "VerificationVisibilityIneligibleError",
    "build_verification_visibility_case",
    "render_verification_visibility_gif",
)
