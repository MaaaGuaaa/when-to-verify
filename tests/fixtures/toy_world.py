"""Deterministic toy world with hand-verifiable answers (SOP-00).

The scene here is deliberately tiny so that every risk label and every net
verification value ``G*`` can be derived by hand. The reference math in this
module is an INDEPENDENT oracle: it does not call any production function (the
geometry/risk/verification packages do not exist yet). Downstream SOPs must
reproduce these same answers with their real implementations.

Simplifications used only inside this fixture (documented for reviewers):

- Robot and pedestrian footprints are circles with radius 0.30 m, so the
  center-to-center collision threshold is ``R_SUM = 0.60 m``.
- The nominal trajectory is a straight constant-velocity rollout at 0.6 m/s.
- The verification example uses a 4-world scenario bank with a candidate set of
  exactly ``{execute nominal, reject}`` so the exact posterior is trivial.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.contracts import (  # noqa: E402
    ACTION_VECTOR_DIM,
    ROBOT_STATE_DIM,
    BaseState,
    GridSpec,
    LocalTrajectory,
    OracleWorld,
    RiskSample,
    VerificationSample,
    build_grid_spec,
)
from src.utils import seeding  # noqa: E402
from src.utils.config import load_config  # noqa: E402

# --- Fixture-local geometry constants -----------------------------------------
TOY_ROBOT_RADIUS = 0.30
TOY_PED_RADIUS = 0.30
R_SUM = TOY_ROBOT_RADIUS + TOY_PED_RADIUS  # 0.60
FUTURE_STEPS = 15
FUTURE_DT = 0.2
NOMINAL_V = 0.6
TOY_SEED_DEFAULT = 42

# Six candidate (v, omega) primitives; index 0 is the straight nominal.
TOY_TRAJECTORY_PRIMITIVES: tuple[tuple[float, float], ...] = (
    (0.6, 0.0),
    (0.2, 0.0),
    (0.4, 0.4),
    (0.4, -0.4),
    (0.6, 0.8),
    (0.6, -0.8),
)

# Scenario-bank execute costs for the nominal trajectory (2 collide, 2 safe).
TOY_WORLD_EXECUTE_COSTS: tuple[float, ...] = (1.0, 1.0, 0.0, 0.0)
TOY_REJECT_COST = 0.20

# Four verification actions: (name, duration_s, forward_m, yaw_deg, reveals_exact).
TOY_VERIFICATION_ACTIONS: tuple[tuple[str, float, float, float, bool], ...] = (
    ("forward_peek", 0.8, 0.30, 0.0, True),
    ("stop_scan", 0.6, 0.0, 0.0, True),
    ("yaw_left_10", 0.4, 0.0, 10.0, False),
    ("yaw_right_20", 0.7, 0.0, 20.0, False),
)


# --- Independent reference math (test oracle) ---------------------------------
def rollout(v: float, omega: float, dt: float = FUTURE_DT, steps: int = FUTURE_STEPS) -> np.ndarray:
    """Constant-control differential-drive rollout; poses[k] at time k*dt."""
    poses = np.zeros((steps, 3), dtype=np.float32)
    x = y = th = 0.0
    for k in range(steps):
        poses[k] = (x, y, th)
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        th += omega * dt
    return poses


def _clearance(robot_xy: np.ndarray, ped_xy: np.ndarray) -> float:
    return float(math.hypot(robot_xy[0] - ped_xy[0], robot_xy[1] - ped_xy[1]) - R_SUM)


def risk_gt_reference(
    robot_xy: np.ndarray,
    ped_xy_seq: np.ndarray | None,
    *,
    sigma_d: float,
    sigma_t: float,
    near_miss_distance: float,
    dt: float = FUTURE_DT,
) -> dict:
    """Reference hidden-risk labels for one robot path vs one hidden pedestrian."""
    if ped_xy_seq is None:
        return {
            "collision": 0,
            "risk_severity": 0.0,
            "min_clearance": math.inf,
            "first_collision_time": None,
            "near_miss": 0,
        }
    collision = False
    min_clearance = math.inf
    first_collision_time: float | None = None
    max_severity = 0.0
    steps = robot_xy.shape[0]
    for k in range(steps):
        tau = k * dt
        clearance = _clearance(robot_xy[k], ped_xy_seq[k])
        min_clearance = min(min_clearance, clearance)
        if clearance <= 0.0:
            collision = True
            if first_collision_time is None:
                first_collision_time = tau
            severity = 1.0
        else:
            severity = math.exp(-clearance / sigma_d) * math.exp(-tau / sigma_t)
        max_severity = max(max_severity, severity)
    return {
        "collision": int(collision),
        "risk_severity": 1.0 if collision else float(max_severity),
        "min_clearance": float(min_clearance),
        "first_collision_time": first_collision_time,
        "near_miss": int((not collision) and min_clearance < near_miss_distance),
    }


def action_cost_reference(duration: float, forward: float, yaw_deg: float, cfg: dict) -> float:
    """Verification action cost c(v) = lt*dt + ll*dl + lyaw*|deg|."""
    vc = cfg["verification_cost"]
    return (
        vc["lambda_time"] * duration
        + vc["lambda_distance"] * forward
        + vc["lambda_yaw_per_deg"] * abs(yaw_deg)
    )


def verification_value_reference(
    execute_costs: tuple[float, ...],
    reject_cost: float,
    action_cost: float,
    reveals_exact: bool,
) -> dict:
    """Reference net verification value G* with candidate set {nominal, reject}.

    ``reveals_exact`` collapses the posterior onto the observed world; otherwise
    the posterior stays uniform (the action carries no information).
    """
    costs = np.asarray(execute_costs, dtype=np.float64)
    m = costs.shape[0]
    br_before = min(float(costs.mean()), reject_cost)
    per_world_br = []
    for observed in range(m):
        if reveals_exact:
            expected_execute = costs[observed]
        else:
            expected_execute = costs.mean()
        per_world_br.append(min(reject_cost, float(expected_execute)))
    post_risk = action_cost + float(np.mean(per_world_br))
    value = br_before - post_risk
    return {
        "br_before": float(br_before),
        "post_risk": float(post_risk),
        "value": float(value),
        "useful": int(value > 0.0),
    }


# --- Schema-valid object builders ---------------------------------------------
def _grid() -> GridSpec:
    return build_grid_spec(load_config())


def make_base_state(state_id: str = "toy_bs_0", grid: GridSpec | None = None) -> BaseState:
    grid = grid or _grid()
    k = grid.history_steps
    return BaseState(
        state_id=state_id,
        split="toy",
        recording_id="toy_rec",
        dynamic_object_ids=("toy_rec::toy_p0",),
        timestamp=0.0,
        robot_history=np.zeros((k, 3), dtype=np.float32),
        robot_state=np.zeros((ROBOT_STATE_DIM,), dtype=np.float32),
        visible_dynamic_object_history={
            "toy_rec::toy_p0": np.zeros((k, 3), dtype=np.float32)
        },
        visible_dynamic_object_specs={
            "toy_rec::toy_p0": {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.30},
            }
        },
        static_map_local=None,
        metadata={"source": "toy"},
    )


def make_local_trajectory(
    trajectory_id: str, v: float, omega: float, grid: GridSpec | None = None
) -> LocalTrajectory:
    grid = grid or _grid()
    h, w = grid.height, grid.width
    return LocalTrajectory(
        trajectory_id=trajectory_id,
        poses=rollout(v, omega, steps=grid.future_steps),
        controls=np.tile(
            np.array([v, omega], dtype=np.float32), (grid.future_steps, 1)
        ),
        swept_mask=np.zeros((h, w), dtype=np.float32),
        tta_map=np.full((h, w), -1.0, dtype=np.float32),
        braking_map=np.zeros((h, w), dtype=np.float32),
        centerline_map=np.zeros((h, w), dtype=np.float32),
        task_cost=0.0,
        metadata={"v": v, "omega": omega},
    )


def make_oracle_world(
    world_id: str, base_state_id: str, seed: int, grid: GridSpec | None = None
) -> OracleWorld:
    grid = grid or _grid()
    h, w = grid.height, grid.width
    return OracleWorld(
        world_id=world_id,
        base_state_id=base_state_id,
        static_occupancy=np.zeros((h, w), dtype=np.float32),
        dynamic_object_trajectories={
            "toy_rec::toy_ped": rollout(
                0.0, 0.0, steps=grid.future_steps
            )[:, :3]
        },
        dynamic_object_specs={
            "toy_rec::toy_ped": {
                "object_type": "human",
                "footprint": {"kind": "circle", "radius_m": 0.30},
            }
        },
        occluders=({"type": "wall", "length_m": 1.5, "width_m": 0.3},),
        blind_spot_config={"kind": "environment"},
        random_seed=seed,
        metadata={"source": "toy"},
    )


def make_risk_sample(grid: GridSpec | None = None, event_type: str = "collision") -> RiskSample:
    grid = grid or _grid()
    h, w, k = grid.height, grid.width, grid.history_steps
    return RiskSample(
        sample_id="toy-risk-0",
        split="toy",
        base_state_id="toy_bs_0",
        pair_group_id="toy-pair-0",
        event_type=event_type,
        bev_history=np.zeros((k, grid.n_history_channels, h, w), dtype=np.float32),
        state_channels=np.zeros((grid.n_state_channels, h, w), dtype=np.float32),
        trajectory_channels=np.zeros((grid.n_trajectory_channels, h, w), dtype=np.float32),
        robot_state=np.array([NOMINAL_V, 0.0], dtype=np.float32),
        collision_label=1,
        risk_severity=1.0,
        min_clearance=-0.3,
        near_miss=0,
        first_collision_time=0.2,
        metadata={"source": "toy"},
    )


def make_verification_sample(grid: GridSpec | None = None) -> VerificationSample:
    grid = grid or _grid()
    h, w, k = grid.height, grid.width, grid.history_steps
    return VerificationSample(
        sample_id="toy-verify-0",
        split="toy",
        base_state_id="toy_bs_0",
        nominal_trajectory_id="toy_traj_0",
        verification_action_id="forward_peek",
        bev_history=np.zeros((k, grid.n_history_channels, h, w), dtype=np.float32),
        state_channels=np.zeros((grid.n_state_channels, h, w), dtype=np.float32),
        trajectory_channels=np.zeros((grid.n_trajectory_channels, h, w), dtype=np.float32),
        verification_fov_mask=np.zeros((1, h, w), dtype=np.float32),
        verification_action_vector=np.array([0.8, 0.30, 0.0], dtype=np.float32),
        value_target=0.053,
        useful_target=1,
        br_before=0.20,
        post_risk=0.147,
        metadata={"source": "toy"},
    )


def make_risk_batch(grid: GridSpec | None = None, batch: int = 2) -> dict:
    """Minimal shape-correct risk batch for a downstream model forward pass."""
    grid = grid or _grid()
    h, w, k = grid.height, grid.width, grid.history_steps
    return {
        "bev_history": np.zeros(
            (batch, k, grid.n_history_channels, h, w), dtype=np.float32
        ),
        "state_channels": np.zeros(
            (batch, grid.n_state_channels, h, w), dtype=np.float32
        ),
        "trajectory_channels": np.zeros(
            (batch, grid.n_trajectory_channels, h, w), dtype=np.float32
        ),
        "robot_state": np.zeros((batch, ROBOT_STATE_DIM), dtype=np.float32),
        "collision_label": np.array([1, 0], dtype=np.int64)[:batch],
        "risk_severity": np.array([1.0, 0.2], dtype=np.float32)[:batch],
        "min_clearance": np.array([-0.3, 0.25], dtype=np.float32)[:batch],
        "near_miss": np.array([0, 1], dtype=np.int64)[:batch],
    }


def make_verification_batch(grid: GridSpec | None = None, batch: int = 2) -> dict:
    """Minimal shape-correct verification batch for a downstream model forward pass."""
    grid = grid or _grid()
    h, w, k = grid.height, grid.width, grid.history_steps
    return {
        "bev_history": np.zeros(
            (batch, k, grid.n_history_channels, h, w), dtype=np.float32
        ),
        "state_channels": np.zeros(
            (batch, grid.n_state_channels, h, w), dtype=np.float32
        ),
        "trajectory_channels": np.zeros(
            (batch, grid.n_trajectory_channels, h, w), dtype=np.float32
        ),
        "verification_fov_mask": np.zeros((batch, 1, h, w), dtype=np.float32),
        "verification_action_vector": np.zeros(
            (batch, ACTION_VECTOR_DIM), dtype=np.float32
        ),
        "value_target": np.array([0.053, -0.031], dtype=np.float32)[:batch],
        "useful_target": np.array([1, 0], dtype=np.int64)[:batch],
    }


# --- Full toy world assembly + hand answers -----------------------------------
def build_toy_world(seed: int = TOY_SEED_DEFAULT) -> dict:
    """Assemble the toy world. Only ``seed_probe`` depends on ``seed``."""
    cfg = load_config()
    grid = build_grid_spec(cfg)
    rg = cfg["risk_gt"]

    base_states = [make_base_state(f"toy_bs_{i}", grid) for i in range(4)]
    trajectories = {
        bs.state_id: [
            make_local_trajectory(f"{bs.state_id}_traj_{j}", v, omega, grid)
            for j, (v, omega) in enumerate(TOY_TRAJECTORY_PRIMITIVES)
        ]
        for bs in base_states
    }
    oracle_worlds = [
        make_oracle_world(f"toy_world_{i}", base_states[0].state_id, seed + i, grid)
        for i in range(4)
    ]

    robot_xy = rollout(NOMINAL_V, 0.0, steps=grid.future_steps)[:, :2]
    steps = grid.future_steps

    def _standing(x: float, y: float) -> np.ndarray:
        return np.tile(np.array([x, y], dtype=np.float64), (steps, 1))

    def _crossing_at(k_star: int, x: float, speed: float) -> np.ndarray:
        seq = np.zeros((steps, 2), dtype=np.float64)
        for k in range(steps):
            seq[k] = (x, speed * FUTURE_DT * (k - k_star))
        return seq

    risk_scene = {
        "collision": _standing(0.6, 0.3),
        "near_miss": _standing(0.6, 0.85),
        "temporal_safe": _crossing_at(k_star=14, x=0.6, speed=2.0),
        "empty": None,
    }
    risk_cases = {
        name: risk_gt_reference(
            robot_xy,
            ped,
            sigma_d=rg["sigma_distance_m"],
            sigma_t=rg["sigma_time_s"],
            near_miss_distance=rg["near_miss_distance_m"],
        )
        for name, ped in risk_scene.items()
    }

    verification_example = {"br_before": None, "actions": {}}
    for name, duration, forward, yaw_deg, reveal in TOY_VERIFICATION_ACTIONS:
        cost = action_cost_reference(duration, forward, yaw_deg, cfg)
        result = verification_value_reference(
            TOY_WORLD_EXECUTE_COSTS, TOY_REJECT_COST, cost, reveal
        )
        verification_example["br_before"] = result["br_before"]
        verification_example["actions"][name] = {
            "action_cost": cost,
            "reveals_exact": reveal,
            **result,
        }

    seed_probe = seeding.make_rng(seed, "toy_probe").standard_normal(4).astype(np.float32)

    return {
        "config": cfg,
        "grid": grid,
        "base_states": base_states,
        "trajectories": trajectories,
        "oracle_worlds": oracle_worlds,
        "verification_actions": [a[0] for a in TOY_VERIFICATION_ACTIONS],
        "risk_cases": risk_cases,
        "verification_example": verification_example,
        "seed_probe": seed_probe,
    }


def toy_hand_answers() -> dict:
    """Hand-derived expected answers, independent of seed.

    See the module docstring for the derivation. Clearances that stem from
    ``sqrt(0.09)`` etc. are compared with tolerance by the tests.
    """
    return {
        "risk": {
            "collision": {
                "collision": 1,
                "near_miss": 0,
                "risk_severity": 1.0,
                "min_clearance": -0.3,
                "first_collision_time": 0.2,
            },
            "near_miss": {"collision": 0, "near_miss": 1, "min_clearance": 0.25},
            "temporal_safe": {"collision": 0, "near_miss": 0, "min_clearance_gt": 0.40},
            "empty": {"collision": 0, "risk_severity": 0.0},
        },
        "verification": {
            "br_before": 0.20,
            "forward_peek": {"value": 0.053, "useful": 1},
            "yaw_left_10": {"value": -0.031, "useful": 0},
            "useful_count": 2,
        },
    }
