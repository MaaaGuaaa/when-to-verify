"""Offline SOP06 audit visualizations with an explicit model/oracle boundary.

The renderer-facing SOP06 API remains history-only.  This module is a separate
review tool that joins a completed observation with oracle paths and a
post-verification observation only after rendering, so reviewers can inspect
whether an action revealed trajectory-relevant blind cells.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Mapping

import numpy as np

from src.contracts import GridSpec, STATE_CHANNELS
from src.geometry import grid_bounds
from src.generation.counterfactual_verify import (
    CounterfactualObservation,
    CounterfactualObservationTrace,
)
from src.generation.observation_renderer import RenderedObservation
from src.generation.sop06_single import Sop06SinglePublication
from src.planning.verification_actions import ActionTrace
from src.utils.atomic_publish import atomic_rename_noreplace


SOP06_VISUAL_AUDIT_VERSION = "sop06_visual_audit_v2"
SOP06_AUDIT_PACKET_VERSION = "sop06_visual_audit_packet_v2"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _require_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _mask(value: object, grid: GridSpec, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (grid.height, grid.width):
        raise ValueError(f"{name} must match the audit grid")
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite numeric or bool")
    if not np.isin(array, (0, 1, False, True)).all():
        raise ValueError(f"{name} must be binary")
    result = np.array(array != 0, dtype=np.bool_, order="C", copy=True)
    result.setflags(write=False)
    return result


def _path(
    value: object,
    grid: GridSpec,
    *,
    name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    expected = (grid.future_steps, 3)
    if allow_empty and array.shape == (0, 3):
        result = np.array(array, dtype=np.float64, order="C", copy=True)
        result.setflags(write=False)
        return result
    if array.shape != expected or array.dtype.kind not in "f" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite float [{expected[0]},3]")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _trace(value: object | None, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.shape[1:] != (3,)
        or array.shape[0] < 2
        or array.dtype.kind not in "f"
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be finite float [N,3] with N >= 2")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _history_path(value: object, grid: GridSpec, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (grid.history_steps, 3)
        or array.dtype.kind not in "f"
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            f"{name} must be finite float [{grid.history_steps},3]"
        )
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _history_observed(value: object, grid: GridSpec, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (grid.history_steps,) or array.dtype != np.bool_:
        raise ValueError(f"{name} must be boolean [{grid.history_steps}]")
    result = np.array(array, dtype=np.bool_, order="C", copy=True)
    result.setflags(write=False)
    return result


def _pose(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (3,)
        or array.dtype.kind not in "f"
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite float pose [3]")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Sop06VisualAuditBundle:
    """Read-only review payload; it is never a SOP06 model input."""

    sample_id: str
    grid: GridSpec
    static_occupancy: np.ndarray
    current_visible_mask: np.ndarray
    current_unobservable_mask: np.ndarray
    post_visible_mask: np.ndarray
    post_unobservable_mask: np.ndarray
    robot_history: np.ndarray
    dynamic_history_paths: Mapping[str, np.ndarray]
    dynamic_history_observed: Mapping[str, np.ndarray]
    candidate_trajectory: np.ndarray
    oracle_future_paths: Mapping[str, np.ndarray]
    hidden_object_ids: tuple[str, ...]
    post_robot_pose: np.ndarray
    verification_trace: np.ndarray | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.sample_id, name="sample_id")
        if not isinstance(self.grid, GridSpec):
            raise TypeError("grid must be a GridSpec")
        if self.grid.future_steps != 32:
            raise ValueError("SOP06 audit requires the frozen 32-step future layout")
        static = _mask(self.static_occupancy, self.grid, name="static_occupancy")
        current_visible = _mask(
            self.current_visible_mask, self.grid, name="current_visible_mask"
        )
        current_unobservable = _mask(
            self.current_unobservable_mask,
            self.grid,
            name="current_unobservable_mask",
        )
        post_visible = _mask(
            self.post_visible_mask, self.grid, name="post_visible_mask"
        )
        post_unobservable = _mask(
            self.post_unobservable_mask,
            self.grid,
            name="post_unobservable_mask",
        )
        if (
            np.any(current_visible & current_unobservable)
            or not np.all(current_visible | current_unobservable)
        ):
            raise ValueError(
                "current visible and unobservable masks must be complements"
            )
        if (
            np.any(post_visible & post_unobservable)
            or not np.all(post_visible | post_unobservable)
        ):
            raise ValueError(
                "post visible and unobservable masks must be endpoint complements"
            )
        robot_history = _history_path(
            self.robot_history,
            self.grid,
            name="robot_history",
        )
        if not isinstance(self.dynamic_history_paths, Mapping) or not isinstance(
            self.dynamic_history_observed, Mapping
        ):
            raise TypeError("dynamic history paths and observed masks must be mappings")
        if set(self.dynamic_history_paths) != set(self.dynamic_history_observed):
            raise ValueError("dynamic history path/observed IDs must match")
        history_paths: dict[str, np.ndarray] = {}
        history_observed: dict[str, np.ndarray] = {}
        for object_id in sorted(self.dynamic_history_paths):
            safe_id = _require_identifier(object_id, name="dynamic history object ID")
            history_paths[safe_id] = _history_path(
                self.dynamic_history_paths[object_id],
                self.grid,
                name=f"dynamic_history_paths[{object_id!r}]",
            )
            history_observed[safe_id] = _history_observed(
                self.dynamic_history_observed[object_id],
                self.grid,
                name=f"dynamic_history_observed[{object_id!r}]",
            )
        candidate = _path(
            self.candidate_trajectory, self.grid, name="candidate_trajectory"
        )
        if not isinstance(self.oracle_future_paths, Mapping):
            raise TypeError("oracle_future_paths must be a mapping")
        paths: dict[str, np.ndarray] = {}
        for object_id, future in sorted(self.oracle_future_paths.items()):
            paths[_require_identifier(object_id, name="oracle object ID")] = _path(
                future,
                self.grid,
                name=f"oracle_future_paths[{object_id!r}]",
            )
        if not isinstance(self.hidden_object_ids, tuple):
            raise TypeError("hidden_object_ids must be a tuple")
        hidden_ids = tuple(
            _require_identifier(value, name="hidden object ID")
            for value in self.hidden_object_ids
        )
        if len(set(hidden_ids)) != len(hidden_ids):
            raise ValueError("hidden_object_ids must be unique")
        if not set(hidden_ids).issubset(paths):
            raise ValueError("hidden_object_ids must be represented in oracle paths")
        object.__setattr__(self, "static_occupancy", static)
        object.__setattr__(self, "current_visible_mask", current_visible)
        object.__setattr__(self, "current_unobservable_mask", current_unobservable)
        object.__setattr__(self, "post_visible_mask", post_visible)
        object.__setattr__(self, "post_unobservable_mask", post_unobservable)
        object.__setattr__(self, "robot_history", robot_history)
        object.__setattr__(
            self,
            "dynamic_history_paths",
            MappingProxyType(history_paths),
        )
        object.__setattr__(
            self,
            "dynamic_history_observed",
            MappingProxyType(history_observed),
        )
        object.__setattr__(self, "candidate_trajectory", candidate)
        object.__setattr__(self, "oracle_future_paths", MappingProxyType(paths))
        object.__setattr__(self, "hidden_object_ids", hidden_ids)
        post_robot_pose = _pose(self.post_robot_pose, name="post_robot_pose")
        object.__setattr__(self, "post_robot_pose", post_robot_pose)
        trace = _trace(self.verification_trace, name="verification_trace")
        if trace is not None:
            if not np.allclose(
                trace[0],
                robot_history[-1],
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError("verification trace must start at current robot pose")
            if not np.allclose(
                trace[-1],
                post_robot_pose,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError("verification trace must end at post robot pose")
        object.__setattr__(
            self,
            "verification_trace",
            trace,
        )


@dataclass(frozen=True)
class Sop06VisualAuditArtifact:
    """Immutable output locations and a compact, machine-readable audit result."""

    output_dir: Path
    bev_pair_path: Path
    bev_toggle_path: Path
    manifest_path: Path
    metadata: Mapping[str, object]


def build_sop06_visual_audit_bundle(
    publication: Sop06SinglePublication,
    rendered_observation: RenderedObservation,
    post_verification: CounterfactualObservation | CounterfactualObservationTrace,
    *,
    grid: GridSpec,
    verification_trace: ActionTrace,
) -> Sop06VisualAuditBundle:
    """Join SOP06 output with oracle/post-action evidence exclusively for review."""

    if not isinstance(publication, Sop06SinglePublication):
        raise TypeError("publication must be a Sop06SinglePublication")
    if not isinstance(rendered_observation, RenderedObservation):
        raise TypeError("rendered_observation must be a RenderedObservation")
    if not isinstance(
        post_verification, (CounterfactualObservation, CounterfactualObservationTrace)
    ):
        raise TypeError("post_verification must be a counterfactual observation or trace")
    if not isinstance(grid, GridSpec):
        raise TypeError("grid must be a GridSpec")
    if not isinstance(verification_trace, ActionTrace):
        raise TypeError("verification_trace must be an ActionTrace")
    if isinstance(post_verification, CounterfactualObservationTrace) and (
        len(post_verification.frames) != verification_trace.poses.shape[0]
        or not np.array_equal(
            post_verification.times_s,
            verification_trace.times_s,
        )
    ):
        raise ValueError(
            "post-verification frames must align with the verification trace"
        )
    if (
        publication.oracle_world.static_occupancy.shape
        != (grid.height, grid.width)
        or publication.trajectory.poses.shape != (grid.future_steps, 3)
    ):
        raise ValueError("publication arrays do not match the supplied SOP06 grid")
    state = np.asarray(rendered_observation.state_channels)
    if state.shape != (len(STATE_CHANNELS), grid.height, grid.width):
        raise ValueError("rendered SOP06 state channels do not match the publication grid")
    current_visible = (
        state[STATE_CHANNELS.index("current_visible_free")] != 0
    ) | (state[STATE_CHANNELS.index("current_visible_occupied")] != 0)
    current_unobservable = (
        state[STATE_CHANNELS.index("current_unobservable_mask")] != 0
    )
    rendered_static = state[STATE_CHANNELS.index("static_obstacle_map")] != 0
    oracle_static = np.asarray(publication.oracle_world.static_occupancy != 0)
    if not np.array_equal(rendered_static, oracle_static):
        raise ValueError("SOP06 static layer differs from the oracle world")
    endpoint = (
        post_verification.frames[-1]
        if isinstance(post_verification, CounterfactualObservationTrace)
        else post_verification
    )
    if endpoint.visible_mask.shape != (grid.height, grid.width):
        raise ValueError("post-verification observation does not match the SOP06 grid")
    post_visible = np.asarray(endpoint.visible_mask, dtype=np.bool_)
    post_unobservable = ~post_visible
    return Sop06VisualAuditBundle(
        sample_id=publication.sample_id,
        grid=grid,
        static_occupancy=oracle_static,
        current_visible_mask=current_visible,
        current_unobservable_mask=current_unobservable,
        post_visible_mask=post_visible,
        post_unobservable_mask=post_unobservable,
        robot_history=publication.renderer_input.base_state.robot_history,
        dynamic_history_paths=publication.renderer_input.scene_dynamic_history,
        dynamic_history_observed=(
            publication.renderer_input.scene_dynamic_history_observed
        ),
        candidate_trajectory=publication.trajectory.poses,
        oracle_future_paths=publication.oracle_world.dynamic_object_trajectories,
        hidden_object_ids=publication.hidden_object_ids,
        post_robot_pose=verification_trace.poses[-1],
        verification_trace=verification_trace.poses,
    )


def _scalar_string(value: np.ndarray, *, name: str) -> str:
    if value.shape not in {(), (1,)}:
        raise ValueError(f"{name} must be a scalar string")
    item = value.item() if value.shape == () else value[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return _require_identifier(item, name=name)


def _scalar_float(value: np.ndarray, *, name: str) -> float:
    if value.shape not in {(), (1,)}:
        raise ValueError(f"{name} must be a scalar")
    item = value.item() if value.shape == () else value[0]
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite float")
    result = float(item)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")
    return result


def load_sop06_visual_audit_packet(path: str | Path) -> Sop06VisualAuditBundle:
    """Load a portable, pickle-free review packet produced after SOP06/SOP11."""

    packet_path = Path(path)
    required = {
        "audit_packet_version",
        "sample_id",
        "resolution_m",
        "static_occupancy",
        "current_visible_mask",
        "current_unobservable_mask",
        "post_visible_mask",
        "post_unobservable_mask",
        "robot_history",
        "dynamic_history_object_ids",
        "dynamic_history_paths",
        "dynamic_history_observed",
        "candidate_trajectory",
        "oracle_object_ids",
        "oracle_future_paths",
        "hidden_object_ids",
        "post_robot_pose",
        "verification_trace",
    }
    with np.load(packet_path, allow_pickle=False) as loaded:
        names = set(loaded.files)
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"SOP06 audit packet missing fields: {missing}")
        if (
            _scalar_string(
                loaded["audit_packet_version"], name="audit_packet_version"
            )
            != SOP06_AUDIT_PACKET_VERSION
        ):
            raise ValueError("unsupported SOP06 audit packet version")
        static = np.array(loaded["static_occupancy"], copy=True)
        if static.ndim != 2 or static.shape[0] <= 0 or static.shape[1] <= 0:
            raise ValueError("static_occupancy must be a non-empty [H,W] array")
        future = np.array(loaded["candidate_trajectory"], copy=True)
        if future.ndim != 2 or future.shape[1:] != (3,):
            raise ValueError("candidate_trajectory must have shape [T,3]")
        robot_history = np.array(loaded["robot_history"], copy=True)
        if robot_history.shape != (8, 3):
            raise ValueError("robot_history must have the frozen shape [8,3]")
        grid = GridSpec(
            height=static.shape[0],
            width=static.shape[1],
            history_steps=robot_history.shape[0],
            future_steps=future.shape[0],
            resolution_m=_scalar_float(loaded["resolution_m"], name="resolution_m"),
        )
        history_ids = np.asarray(loaded["dynamic_history_object_ids"])
        history_paths = np.asarray(loaded["dynamic_history_paths"])
        history_observed = np.asarray(loaded["dynamic_history_observed"])
        if (
            history_ids.ndim != 1
            or history_paths.shape
            != (len(history_ids), grid.history_steps, 3)
            or history_observed.shape
            != (len(history_ids), grid.history_steps)
        ):
            raise ValueError("dynamic history IDs, paths, and observed masks do not align")
        if (
            history_observed.dtype.kind not in "biuf"
            or not np.isfinite(history_observed).all()
            or not np.isin(history_observed, (0, 1, False, True)).all()
        ):
            raise ValueError("dynamic_history_observed must be binary")
        dynamic_paths = {
            _require_identifier(str(object_id), name="dynamic history object ID"):
            history_paths[index]
            for index, object_id in enumerate(history_ids)
        }
        dynamic_observed = {
            _require_identifier(str(object_id), name="dynamic history object ID"):
            np.asarray(history_observed[index] != 0, dtype=np.bool_)
            for index, object_id in enumerate(history_ids)
        }
        if len(dynamic_paths) != len(history_ids):
            raise ValueError("dynamic_history_object_ids must be unique")
        object_ids = np.asarray(loaded["oracle_object_ids"])
        paths = np.asarray(loaded["oracle_future_paths"])
        if object_ids.ndim != 1 or paths.shape != (
            len(object_ids),
            grid.future_steps,
            3,
        ):
            raise ValueError("oracle object IDs and future paths do not align")
        oracle_paths = {
            _require_identifier(str(object_id), name="oracle object ID"): paths[index]
            for index, object_id in enumerate(object_ids)
        }
        if len(oracle_paths) != len(object_ids):
            raise ValueError("oracle_object_ids must be unique")
        hidden_raw = np.asarray(loaded["hidden_object_ids"])
        if hidden_raw.ndim != 1:
            raise ValueError("hidden_object_ids must be one-dimensional")
        trace = np.asarray(loaded["verification_trace"])
        if trace.shape == (0, 3):
            trace_value: np.ndarray | None = None
        else:
            trace_value = trace
        return Sop06VisualAuditBundle(
            sample_id=_scalar_string(loaded["sample_id"], name="sample_id"),
            grid=grid,
            static_occupancy=static,
            current_visible_mask=np.array(loaded["current_visible_mask"], copy=True),
            current_unobservable_mask=np.array(
                loaded["current_unobservable_mask"], copy=True
            ),
            post_visible_mask=np.array(loaded["post_visible_mask"], copy=True),
            post_unobservable_mask=np.array(
                loaded["post_unobservable_mask"],
                copy=True,
            ),
            robot_history=robot_history,
            dynamic_history_paths=dynamic_paths,
            dynamic_history_observed=dynamic_observed,
            candidate_trajectory=future,
            oracle_future_paths=oracle_paths,
            hidden_object_ids=tuple(
                _require_identifier(str(object_id), name="hidden object ID")
                for object_id in hidden_raw
            ),
            post_robot_pose=np.array(loaded["post_robot_pose"], copy=True),
            verification_trace=trace_value,
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


def _candidate_blind_metrics(bundle: Sop06VisualAuditBundle) -> dict[str, object]:
    x_min, _, y_min, _ = grid_bounds(bundle.grid)
    positions = bundle.candidate_trajectory[:, :2]
    columns = np.floor((positions[:, 0] - x_min) / bundle.grid.resolution_m).astype(int)
    rows = np.floor((positions[:, 1] - y_min) / bundle.grid.resolution_m).astype(int)
    valid = (
        (rows >= 0)
        & (rows < bundle.grid.height)
        & (columns >= 0)
        & (columns < bundle.grid.width)
    )
    blind = bundle.current_unobservable_mask[rows[valid], columns[valid]]
    revealed = bundle.post_visible_mask[rows[valid], columns[valid]]
    blind_count = int(blind.sum())
    revealed_count = int(np.logical_and(blind, revealed).sum())
    displacement = float(
        np.linalg.norm(
            bundle.post_robot_pose[:2] - bundle.robot_history[-1, :2]
        )
    )
    return {
        "candidate_endpoint_count_in_grid": int(valid.sum()),
        "candidate_endpoint_count_initially_unobservable": blind_count,
        "candidate_endpoint_count_revealed_after_verification": revealed_count,
        "candidate_blind_endpoint_reveal_fraction": (
            0.0 if blind_count == 0 else revealed_count / blind_count
        ),
        "current_visible_cell_count": int(bundle.current_visible_mask.sum()),
        "post_endpoint_visible_cell_count": int(bundle.post_visible_mask.sum()),
        "newly_visible_at_endpoint_cell_count": int(
            (bundle.post_visible_mask & bundle.current_unobservable_mask).sum()
        ),
        "no_longer_visible_at_endpoint_cell_count": int(
            (bundle.current_visible_mask & bundle.post_unobservable_mask).sum()
        ),
        "robot_endpoint_displacement_m": displacement,
        "hidden_object_observed_history_counts": {
            object_id: int(bundle.dynamic_history_observed[object_id].sum())
            for object_id in bundle.hidden_object_ids
            if object_id in bundle.dynamic_history_observed
        },
    }


def _draw_panel(
    axis,
    bundle: Sop06VisualAuditBundle,
    *,
    title: str,
    visible_mask: np.ndarray,
    unobservable_mask: np.ndarray,
    panel_kind: str,
) -> None:
    if panel_kind not in {"current", "oracle", "post"}:
        raise ValueError("panel_kind must be current, oracle, or post")
    x_min, x_max, y_min, y_max = grid_bounds(bundle.grid)
    extent = (x_min, x_max, y_min, y_max)
    axis.imshow(
        _mask_rgba(unobservable_mask, (0.84, 0.20, 0.24), 0.22),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=0,
    )
    axis.imshow(
        _mask_rgba(visible_mask, (0.10, 0.48, 0.78), 0.14),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=1,
    )
    if panel_kind == "post":
        newly_visible = visible_mask & bundle.current_unobservable_mask
        axis.imshow(
            _mask_rgba(newly_visible, (0.10, 0.68, 0.31), 0.34),
            origin="lower",
            extent=extent,
            interpolation="nearest",
            zorder=2,
        )
    axis.imshow(
        _mask_rgba(bundle.static_occupancy, (0.08, 0.09, 0.10), 0.96),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=3,
    )

    robot_history = bundle.robot_history
    axis.plot(
        robot_history[:, 0],
        robot_history[:, 1],
        color="#111820",
        linewidth=1.8,
        marker="o",
        markersize=3.2,
        alpha=0.88,
        zorder=6,
    )
    for object_id, history in bundle.dynamic_history_paths.items():
        observed = bundle.dynamic_history_observed[object_id]
        color = "#f0a202" if object_id in bundle.hidden_object_ids else "#6d7890"
        indices = np.flatnonzero(observed)
        for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
            if run.size == 0:
                continue
            axis.plot(
                history[run, 0],
                history[run, 1],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                alpha=0.94,
                zorder=7,
            )
        if indices.size:
            last = int(indices[-1])
            axis.scatter(
                [history[last, 0]],
                [history[last, 1]],
                s=44,
                facecolor="white",
                edgecolor=color,
                linewidth=1.6,
                zorder=8,
            )

    candidate = np.vstack(
        (bundle.robot_history[-1:], bundle.candidate_trajectory)
    )
    axis.plot(
        candidate[:, 0],
        candidate[:, 1],
        color="#008a8a",
        linewidth=3.2,
        solid_capstyle="round",
        zorder=9,
    )
    if panel_kind == "post" and bundle.verification_trace is not None:
        axis.plot(
            bundle.verification_trace[:, 0],
            bundle.verification_trace[:, 1],
            color="#e17c00",
            linestyle="-.",
            linewidth=2.4,
            zorder=10,
        )
    if panel_kind in {"oracle", "post"}:
        for object_id, future in bundle.oracle_future_paths.items():
            is_hidden = object_id in bundle.hidden_object_ids
            axis.plot(
                future[:, 0],
                future[:, 1],
                color="#d21b96" if is_hidden else "#8b94a6",
                linestyle="--" if is_hidden else ":",
                linewidth=2.8 if is_hidden else 1.2,
                alpha=1.0 if is_hidden else 0.55,
                zorder=12 if is_hidden else 5,
            )
            if is_hidden:
                axis.scatter(
                    [future[0, 0]],
                    [future[0, 1]],
                    s=52,
                    facecolor="none",
                    edgecolor="#d21b96",
                    linewidth=1.7,
                    zorder=13,
                )

    robot_pose = (
        bundle.post_robot_pose
        if panel_kind == "post"
        else bundle.robot_history[-1]
    )
    if panel_kind == "post":
        current_pose = bundle.robot_history[-1]
        axis.scatter(
            [current_pose[0]],
            [current_pose[1]],
            marker="^",
            s=82,
            facecolor="none",
            edgecolor="#111820",
            linewidth=1.5,
            zorder=14,
        )
    robot_color = "#e17c00" if panel_kind == "post" else "#111820"
    axis.scatter(
        [robot_pose[0]],
        [robot_pose[1]],
        marker="^",
        s=82,
        color=robot_color,
        edgecolor="white",
        linewidth=0.8,
        zorder=15,
    )
    heading_length = max(0.35, 3.5 * bundle.grid.resolution_m)
    axis.arrow(
        robot_pose[0],
        robot_pose[1],
        heading_length * np.cos(robot_pose[2]),
        heading_length * np.sin(robot_pose[2]),
        color=robot_color,
        width=0.015,
        head_width=0.15,
        length_includes_head=True,
        zorder=14,
    )
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18, linewidth=0.4)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(title, fontsize=12, fontweight="bold")


def _legend_handles(
    Line2D,
    Patch,
    *,
    has_trace: bool,
    show_oracle_future: bool,
    show_newly_visible: bool,
):
    handles = [
        Patch(facecolor="#151719", edgecolor="#111315", label="static obstacle"),
        Patch(facecolor="#1f7abf", alpha=0.35, label="visible mask"),
        Patch(facecolor="#d92833", alpha=0.35, label="unobservable mask"),
        Line2D(
            [0],
            [0],
            color="#111820",
            marker="o",
            markersize=3,
            linewidth=1.8,
            label="robot history (8)",
        ),
        Line2D(
            [0],
            [0],
            color="#f0a202",
            marker="o",
            markersize=3,
            linewidth=2.0,
            label="observed pedestrian history",
        ),
        Line2D(
            [0], [0], color="#008a8a", linewidth=3.2, label="candidate trajectory"
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="#111820",
            linestyle="none",
            label="current robot pose",
        ),
    ]
    if show_oracle_future:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#d21b96",
                linestyle="--",
                linewidth=2.8,
                label="pedestrian future 32 (oracle-only)",
            )
        )
    if show_newly_visible:
        handles.append(
            Patch(
                facecolor="#1aad4f",
                alpha=0.45,
                label="newly visible at action endpoint",
            )
        )
    if has_trace:
        handles.extend(
            (
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    markerfacecolor="#e17c00",
                    markeredgecolor="white",
                    color="#e17c00",
                    linestyle="none",
                    label="post-action robot pose",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#e17c00",
                    linestyle="-.",
                    label="verification action trace",
                ),
            )
        )
    return handles


def _figure_image(figure, FigureCanvasAgg, Image):
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return Image.fromarray(rgba, mode="RGBA").convert("RGB")


def _render_pair_png(bundle: Sop06VisualAuditBundle, path: Path) -> dict[str, object]:
    plt, _, Line2D, Patch, Image = _plot_modules()
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
    figure.subplots_adjust(
        left=0.04, right=0.995, bottom=0.16, top=0.72, wspace=0.18
    )
    _draw_panel(
        axes[0],
        bundle,
        title="Current deployment BEV\n(no oracle future)",
        visible_mask=bundle.current_visible_mask,
        unobservable_mask=bundle.current_unobservable_mask,
        panel_kind="current",
    )
    _draw_panel(
        axes[1],
        bundle,
        title="Oracle complete world\n(candidate + pedestrian future 32)",
        visible_mask=bundle.current_visible_mask,
        unobservable_mask=bundle.current_unobservable_mask,
        panel_kind="oracle",
    )
    _draw_panel(
        axes[2],
        bundle,
        title="Post-verification endpoint BEV\n(robot pose + visibility rerendered)",
        visible_mask=bundle.post_visible_mask,
        unobservable_mask=bundle.post_unobservable_mask,
        panel_kind="post",
    )
    metrics = _candidate_blind_metrics(bundle)
    hidden_history_count = next(
        iter(metrics["hidden_object_observed_history_counts"].values()),
        0,
    )
    figure.suptitle(
        f"SOP06 audit: {bundle.sample_id}", fontsize=16, fontweight="bold", y=0.98
    )
    figure.text(
        0.5,
        0.06,
        f"Robot history: {bundle.grid.history_steps}/"
        f"{bundle.grid.history_steps} (overlap when stationary); "
        f"target observed history: {hidden_history_count}/"
        f"{bundle.grid.history_steps}; robot moved "
        f"{metrics['robot_endpoint_displacement_m']:.2f} m; "
        f"newly visible at endpoint: {metrics['newly_visible_at_endpoint_cell_count']} cells; "
        "candidate endpoints initially blind: "
        f"{metrics['candidate_endpoint_count_initially_unobservable']}; "
        "visible at endpoint: "
        f"{metrics['candidate_endpoint_count_revealed_after_verification']} "
        f"({metrics['candidate_blind_endpoint_reveal_fraction']:.0%}). "
        "Oracle future is intentionally absent from the left panel.",
        ha="center",
        fontsize=9,
    )
    figure.legend(
        handles=_legend_handles(
            Line2D,
            Patch,
            has_trace=bundle.verification_trace is not None,
            show_oracle_future=True,
            show_newly_visible=True,
        ),
        loc="upper center",
        ncol=6,
        framealpha=0.94,
        bbox_to_anchor=(0.5, 0.90),
        fontsize=8,
    )
    figure.savefig(path, dpi=100, facecolor="white")
    plt.close(figure)
    with Image.open(path) as image:
        if image.size != (1800, 600):
            raise RuntimeError("bev_pair.png violates the fixed SOP06 audit size")
        if np.asarray(image.convert("RGB")).std() <= 5.0:
            raise RuntimeError("bev_pair.png is blank")
    return {"width": 1800, "height": 600, "sha256": _sha256_file(path)}


def _render_toggle_gif(
    bundle: Sop06VisualAuditBundle,
    path: Path,
    *,
    frame_duration_ms: int,
) -> dict[str, object]:
    plt, FigureCanvasAgg, Line2D, Patch, Image = _plot_modules()
    frames = []
    panels = (
        (
            "Current deployment BEV (no oracle future)",
            bundle.current_visible_mask,
            bundle.current_unobservable_mask,
            "current",
        ),
        (
            "Post-verification endpoint BEV",
            bundle.post_visible_mask,
            bundle.post_unobservable_mask,
            "post",
        ),
    )
    for title, visible, unobservable, panel_kind in panels:
        figure, axis = plt.subplots(figsize=(8, 8), dpi=100)
        figure.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.84)
        _draw_panel(
            axis,
            bundle,
            title=title,
            visible_mask=visible,
            unobservable_mask=unobservable,
            panel_kind=panel_kind,
        )
        figure.suptitle(
            f"SOP06 audit toggle: {bundle.sample_id}",
            fontsize=14,
            fontweight="bold",
        )
        figure.legend(
            handles=_legend_handles(
                Line2D,
                Patch,
                has_trace=(
                    panel_kind == "post"
                    and bundle.verification_trace is not None
                ),
                show_oracle_future=panel_kind == "post",
                show_newly_visible=panel_kind == "post",
            ),
            loc="lower center",
            ncol=4,
            fontsize=7,
            framealpha=0.94,
        )
        frames.append(_figure_image(figure, FigureCanvasAgg, Image))
        plt.close(figure)
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    with Image.open(path) as image:
        if image.size != (800, 800) or image.n_frames != 2:
            raise RuntimeError("bev_toggle.gif violates the SOP06 two-frame contract")
    return {
        "width": 800,
        "height": 800,
        "frame_count": 2,
        "frame_duration_ms": frame_duration_ms,
        "sha256": _sha256_file(path),
    }


def render_sop06_visual_audit(
    bundle: Sop06VisualAuditBundle,
    output_dir: str | Path,
    *,
    frame_duration_ms: int = 750,
) -> Sop06VisualAuditArtifact:
    """Atomically publish the required PNG, two-frame GIF and audit manifest."""

    if not isinstance(bundle, Sop06VisualAuditBundle):
        raise TypeError("bundle must be a Sop06VisualAuditBundle")
    if (
        not isinstance(frame_duration_ms, int)
        or isinstance(frame_duration_ms, bool)
        or frame_duration_ms <= 0
    ):
        raise ValueError("frame_duration_ms must be a positive integer")
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite SOP06 audit: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    staging = staging_root / destination.name
    staging.mkdir()
    try:
        pair_metadata = _render_pair_png(bundle, staging / "bev_pair.png")
        toggle_metadata = _render_toggle_gif(
            bundle,
            staging / "bev_toggle.gif",
            frame_duration_ms=frame_duration_ms,
        )
        metrics = _candidate_blind_metrics(bundle)
        metadata: dict[str, object] = {
            "version": SOP06_VISUAL_AUDIT_VERSION,
            "sample_id": bundle.sample_id,
            "oracle_boundary": "offline_audit_only",
            "oracle_path_object_ids": list(bundle.oracle_future_paths),
            "hidden_object_ids": list(bundle.hidden_object_ids),
            "verification_trace_rendered": bundle.verification_trace is not None,
            "post_visibility_semantics": "action_endpoint_frame_not_trace_union",
            "current_robot_pose": bundle.robot_history[-1].tolist(),
            "post_robot_pose": bundle.post_robot_pose.tolist(),
            "panel_semantics": {
                "left": {
                    "kind": "deployment_current",
                    "oracle_future_rendered": False,
                },
                "middle": {
                    "kind": "oracle_complete_world",
                    "oracle_future_rendered": True,
                },
                "right": {
                    "kind": "post_verification_endpoint",
                    "oracle_future_rendered": True,
                },
            },
            "bev_pair": pair_metadata,
            "bev_toggle": toggle_metadata,
            **metrics,
        }
        (staging / "manifest.json").write_text(
            _canonical_json(metadata) + "\n", encoding="utf-8"
        )
        atomic_rename_noreplace(staging, destination)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    else:
        staging_root.rmdir()
    return Sop06VisualAuditArtifact(
        output_dir=destination,
        bev_pair_path=destination / "bev_pair.png",
        bev_toggle_path=destination / "bev_toggle.gif",
        manifest_path=destination / "manifest.json",
        metadata=MappingProxyType(metadata),
    )


__all__ = [
    "SOP06_AUDIT_PACKET_VERSION",
    "SOP06_VISUAL_AUDIT_VERSION",
    "Sop06VisualAuditArtifact",
    "Sop06VisualAuditBundle",
    "build_sop06_visual_audit_bundle",
    "load_sop06_visual_audit_packet",
    "render_sop06_visual_audit",
]
