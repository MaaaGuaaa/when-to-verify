"""Configuration-driven differential-drive primitive sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .differential_drive import rollout_constant_control


@dataclass(frozen=True)
class TrajectoryPrimitive:
    """One candidate constant-control command with explicit semantic flags."""

    primitive_id: str
    v: float
    omega: float
    is_stop: bool
    is_reverse: bool


@dataclass(frozen=True)
class CandidateRollout:
    """Rolled-out primitive before geometry query maps are attached."""

    trajectory_id: str
    poses: np.ndarray
    controls: np.ndarray
    is_stop: bool
    is_reverse: bool


def sample_trajectory_primitives(
    config: dict,
    *,
    reverse_stress: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[TrajectoryPrimitive, ...]:
    """Build the configured forward primitive grid plus a marked stop."""
    cfg = config["trajectories"]
    primitives = [
        TrajectoryPrimitive(
            primitive_id=f"forward_v{v_index:02d}_w{omega_index:02d}",
            v=float(v),
            omega=float(omega),
            is_stop=False,
            is_reverse=False,
        )
        for v_index, v in enumerate(cfg["linear_velocities"])
        for omega_index, omega in enumerate(cfg["angular_velocities"])
    ]
    if reverse_stress:
        reverse_probability = float(cfg["reverse_probability"])
        if (
            not np.isfinite(reverse_probability)
            or not 0.0 <= reverse_probability <= 1.0
        ):
            raise ValueError("reverse_probability must be finite and in [0, 1]")
        rng = rng if rng is not None else np.random.default_rng(config["seed"])
        if rng.random() < reverse_probability:
            primitives.extend(
                TrajectoryPrimitive(
                    primitive_id=f"reverse_v{v_index:02d}_w{omega_index:02d}",
                    v=float(v),
                    omega=float(omega),
                    is_stop=False,
                    is_reverse=True,
                )
                for v_index, v in enumerate(cfg["reverse_velocities"])
                for omega_index, omega in enumerate(cfg["angular_velocities"])
            )
    primitives.append(
        TrajectoryPrimitive(
            primitive_id="stop",
            v=0.0,
            omega=0.0,
            is_stop=True,
            is_reverse=False,
        )
    )
    return tuple(primitives)


def sample_candidate_rollouts(
    config: dict,
    *,
    reverse_stress: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[CandidateRollout, ...]:
    """Sample and roll out candidates using the configured horizon and timestep."""
    cfg = config["trajectories"]
    dt_s = float(cfg["dt_s"])
    horizon_s = float(cfg["horizon_s"])
    bev = config["bev"]
    if (
        not np.isfinite([dt_s, horizon_s]).all()
        or dt_s <= 0.0
        or horizon_s <= 0.0
    ):
        raise ValueError("trajectory time grid must be finite and positive")
    steps_float = horizon_s / dt_s
    steps = int(round(steps_float))
    if (
        not np.isclose(steps_float, steps, atol=1e-9, rtol=0.0)
        or steps != int(bev["future_steps"])
        or not np.isclose(dt_s, float(bev["future_dt_s"]), atol=1e-12, rtol=0.0)
    ):
        raise ValueError("trajectory time grid conflicts with frozen BEV contract")
    candidates = []
    for primitive in sample_trajectory_primitives(
        config,
        reverse_stress=reverse_stress,
        rng=rng,
    ):
        poses, controls = rollout_constant_control(
            v=primitive.v,
            omega=primitive.omega,
            dt_s=dt_s,
            steps=steps,
        )
        candidates.append(
            CandidateRollout(
                trajectory_id=primitive.primitive_id,
                poses=poses,
                controls=controls,
                is_stop=primitive.is_stop,
                is_reverse=primitive.is_reverse,
            )
        )
    return tuple(candidates)
