#!/usr/bin/env python3
"""Render five real-scene verification visibility progression GIFs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import STATE_CHANNELS, build_grid_spec  # noqa: E402
from src.evaluation.verification_visibility_gif import (  # noqa: E402
    VerificationVisibilityIneligibleError,
    build_verification_visibility_case,
    render_verification_visibility_gif,
)
from src.geometry import RectangleFootprint, inflate_footprint  # noqa: E402
from src.generation.observation_renderer import render_observation  # noqa: E402
from src.generation.sop06_finalized_source import (  # noqa: E402
    load_sop06_finalized_source,
)
from src.generation.structural_blindspot import StructuralBlindSpot  # noqa: E402
from src.generation.verification_gt import load_verification_gt_config  # noqa: E402
from src.planning.verification_actions import (  # noqa: E402
    VerificationAction,
    VerificationActionLibrary,
    load_verification_actions,
)


OUTPUT_VERSION = "verification_visibility_real_smoke5_v1"
_MOTION_ACTION_IDS = (
    "arc_left_30",
    "arc_right_30",
    "arc_left_45",
    "arc_right_45",
    "forward_peek",
)


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--final-scenario-root", type=Path, required=True)
    parser.add_argument("--actions-config", type=Path, required=True)
    parser.add_argument("--gt-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "calibration", "val", "test"))
    parser.add_argument("--frame-duration-ms", type=_positive_int, default=120)
    parser.add_argument("--max-source-attempts", type=_positive_int, default=100)
    return parser


def _motion_actions(
    library: VerificationActionLibrary,
) -> tuple[VerificationAction, ...]:
    by_id = library.by_id
    if any(action_id not in by_id for action_id in _MOTION_ACTION_IDS):
        raise ValueError("verification action library lacks a required motion action")
    return tuple(by_id[action_id] for action_id in _MOTION_ACTION_IDS)


def _sensor_range_m(world, grid) -> float:
    blind_spot = world.blind_spot_config
    if isinstance(blind_spot, Mapping):
        structural = blind_spot.get("structural")
        if isinstance(structural, Mapping) and "range_m" in structural:
            value = float(structural["range_m"])
            if np.isfinite(value) and value > 0.0:
                return value
    return float(
        np.hypot(
            grid.height * grid.resolution_m,
            grid.width * grid.resolution_m,
        )
    )


def _case_from_publication(
    publication,
    *,
    action: VerificationAction,
    library: VerificationActionLibrary,
    base_config: Mapping[str, object],
    braking_deceleration_mps2: float,
    angular_deceleration_radps2: float,
):
    config = dict(base_config)
    grid = build_grid_spec(config)
    world = publication.oracle_world
    renderer = publication.renderer_input
    sensor_range_m = _sensor_range_m(world, grid)
    sensor = StructuralBlindSpot(
        forward_fov_deg=float(np.rad2deg(library.sensor_fov_rad)),
        range_m=sensor_range_m,
    )
    rendered = render_observation(
        renderer.base_state,
        scene_dynamic_history=renderer.scene_dynamic_history,
        scene_dynamic_specs=renderer.scene_dynamic_specs,
        scene_dynamic_history_observed=(
            renderer.scene_dynamic_history_observed
        ),
        static_occupancy=renderer.observed_static_occupancy,
        sensor_config=sensor,
        config=config,
    )
    current_visible = (
        rendered.state_channels[
            STATE_CHANNELS.index("current_visible_free")
        ]
        != 0.0
    ) | (
        rendered.state_channels[
            STATE_CHANNELS.index("current_visible_occupied")
        ]
        != 0.0
    )
    current_age = rendered.state_channels[
        STATE_CHANNELS.index("occlusion_age_map")
    ]
    future_ids = set(world.dynamic_object_trajectories)
    if (
        future_ids != set(world.dynamic_object_specs)
        or not future_ids.issubset(renderer.scene_dynamic_history)
    ):
        raise ValueError("oracle future IDs lack aligned history/spec fields")
    current_poses = {
        object_id: np.array(
            renderer.scene_dynamic_history[object_id][-1],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        for object_id in sorted(future_ids)
    }
    robot = config["robot"]
    robot_footprint = inflate_footprint(
        RectangleFootprint(
            float(robot["length_m"]),
            float(robot["width_m"]),
        ),
        float(robot["inflation_m"]),
    )
    return build_verification_visibility_case(
        sample_id=publication.sample_id,
        action=action,
        grid=grid,
        robot_pose=np.asarray(
            renderer.base_state.robot_history[-1],
            dtype=np.float32,
        ),
        robot_state=np.asarray(
            renderer.base_state.robot_state,
            dtype=np.float32,
        ),
        robot_footprint=robot_footprint,
        static_occupancy=np.asarray(world.static_occupancy, dtype=np.float32),
        dynamic_current_poses=current_poses,
        dynamic_future_poses=world.dynamic_object_trajectories,
        dynamic_specs=world.dynamic_object_specs,
        current_visible_mask=np.asarray(current_visible, dtype=bool),
        current_age_map=np.asarray(current_age, dtype=np.float32),
        future_dt_s=float(config["bev"]["future_dt_s"]),
        age_max_s=float(config["age_map"]["a_max_s"]),
        fov_rad=library.sensor_fov_rad,
        max_range_m=sensor_range_m,
        braking_deceleration_mps2=braking_deceleration_mps2,
        angular_deceleration_radps2=angular_deceleration_radps2,
    )


def _canonical_json(value: object) -> str:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    )


def _render_collection(args: argparse.Namespace) -> dict[str, object]:
    destination = args.output_dir
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite visibility collection: {destination}"
        )
    library = load_verification_actions(args.actions_config)
    gt_config = load_verification_gt_config(args.gt_config)
    source = load_sop06_finalized_source(
        source_mode="complete_mother",
        source_root=args.source_root,
        final_scenario_root=args.final_scenario_root,
        split=args.split,
    )
    candidates = [
        record
        for record in source.accepted
        if record.regime == "seen_then_occluded" and record.target_present
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    cursor = 0
    attempted = 0
    try:
        for ordinal, action in enumerate(_motion_actions(library), start=1):
            selected = None
            while cursor < len(candidates) and attempted < args.max_source_attempts:
                record = candidates[cursor]
                cursor += 1
                attempted += 1
                publication = source.resolve(record).publication
                try:
                    selected = _case_from_publication(
                        publication,
                        action=action,
                        library=library,
                        base_config=source.base_config,
                        braking_deceleration_mps2=(
                            gt_config.braking_deceleration_mps2
                        ),
                        angular_deceleration_radps2=(
                            gt_config.angular_deceleration_radps2
                        ),
                    )
                except VerificationVisibilityIneligibleError as exc:
                    skipped.append(
                        {
                            "sample_id": record.scenario_id,
                            "action_id": action.action_id,
                            "reason": exc.reason,
                            "detail": exc.detail,
                        }
                    )
                    continue
                break
            if selected is None:
                raise RuntimeError(
                    f"no feasible real sample found for {action.action_id}"
                )
            relative_dir = (
                f"{ordinal:02d}_{action.action_id}_{selected.sample_id}"
            )
            artifact = render_verification_visibility_gif(
                selected,
                staging / relative_dir,
                frame_duration_ms=args.frame_duration_ms,
            )
            metrics = artifact.metadata["frame_metrics"]
            rows.append(
                {
                    "ordinal": ordinal,
                    "sample_id": selected.sample_id,
                    "action_id": action.action_id,
                    "gif": f"{relative_dir}/visibility_progress.gif",
                    "manifest": f"{relative_dir}/manifest.json",
                    "frame_count": artifact.metadata["frame_count"],
                    "duration_s": float(selected.action_trace.times_s[-1]),
                    "initial_visible_cell_count": metrics[0][
                        "visible_cell_count"
                    ],
                    "final_visible_cell_count": metrics[-1][
                        "visible_cell_count"
                    ],
                    "final_newly_visible_cell_count": metrics[-1][
                        "newly_visible_cell_count"
                    ],
                    "cumulative_newly_visible_cell_count": metrics[-1][
                        "cumulative_newly_visible_cell_count"
                    ],
                }
            )
        index = {
            "version": OUTPUT_VERSION,
            "selection": (
                "first feasible distinct seen_then_occluded real sample "
                "for each configured motion action"
            ),
            "source_publication_semantic_digest": (
                source.source_publication_semantic_digest
            ),
            "source_final_release_identity": source.final_release_identity,
            "attempted_source_count": attempted,
            "rows": rows,
            "skipped": skipped,
        }
        (staging / "index.json").write_text(
            _canonical_json(index),
            encoding="utf-8",
        )
        os.rename(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "output_dir": str(destination),
        "index": str(destination / "index.json"),
        "gif_count": len(rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.split is None:
        args.split = "train"
    try:
        result = _render_collection(args)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
