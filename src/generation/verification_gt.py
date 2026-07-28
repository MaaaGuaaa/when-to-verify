"""Verification value for sampled SOP5 realizations and legacy world banks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.contracts import (
    ARRAY_DTYPE,
    GridSpec,
    LONG40_FUTURE_HORIZON_S,
    LocalTrajectory,
    OracleWorld,
    POSE_TIME_LAYOUT_VERSION,
    SCHEMA_VERSION,
)
from src.geometry import CircleFootprint, Footprint, RectangleFootprint
from src.generation.counterfactual_verify import (
    CounterfactualObservation,
    SignatureNormalizer,
)
from src.generation.observation_posterior import (
    exact_observation_posterior,
    observable_observation_digest,
    soft_observation_posterior,
    validate_posterior_matrix,
)
from src.generation.scenario_bank import ScenarioBank, ScenarioHypothesis
from src.generation.risk_gt import compute_hidden_risk_gt
from src.planning.replanning import (
    POST_PLAN_STATUSES,
    POST_PLAN_STATUS_SAFE_STOP_NO_FEASIBLE_PLAN,
    ReplannedCandidate,
    ReplanningResult,
)
from src.planning.verification_actions import (
    VerificationAction,
    action_cost as compute_action_cost,
)
from src.planning.verification_responses import VERIFICATION_RESPONSE_VERSION


VERIFICATION_GT_VERSION = "verification_value_gt_v5"
SAMPLED_REALIZATION_GT_VERSION = "sampled_realization_value_gt_v1"
_TOP_KEYS = frozenset({"schema_version", "scenario_bank", "posterior", "decision"})
_DECISION_KEYS = frozenset(
    {
        "reject_cost",
        "risk_weight",
        "braking_deceleration_mps2",
        "angular_deceleration_radps2",
        "braking_margin_s",
    }
)

RiskLoss = Callable[[LocalTrajectory, np.ndarray, ScenarioHypothesis], float]
RealizedRiskLoss = Callable[[LocalTrajectory, np.ndarray, OracleWorld], float]


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _owned_float64_vector(value: Any, *, name: str, size: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.float64:
        raise TypeError(f"{name} must be a float64 ndarray")
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape ({size},)")
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _owned_posterior(value: Any, *, size: int) -> np.ndarray:
    validate_posterior_matrix(value, size=size)
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class VerificationGTConfig:
    reject_cost: float
    risk_weight: float
    braking_deceleration_mps2: float
    angular_deceleration_radps2: float
    braking_margin_s: float

    def __post_init__(self) -> None:
        reject = _finite_nonnegative(self.reject_cost, name="reject_cost")
        weight = _finite_nonnegative(self.risk_weight, name="risk_weight")
        braking = _finite_nonnegative(
            self.braking_deceleration_mps2,
            name="braking_deceleration_mps2",
        )
        angular = _finite_nonnegative(
            self.angular_deceleration_radps2,
            name="angular_deceleration_radps2",
        )
        margin = _finite_nonnegative(
            self.braking_margin_s,
            name="braking_margin_s",
        )
        if braking <= 0.0 or angular <= 0.0:
            raise ValueError("braking decelerations must be positive")
        object.__setattr__(self, "reject_cost", reject)
        object.__setattr__(self, "risk_weight", weight)
        object.__setattr__(self, "braking_deceleration_mps2", braking)
        object.__setattr__(self, "angular_deceleration_radps2", angular)
        object.__setattr__(self, "braking_margin_s", margin)


@dataclass(frozen=True)
class TypedFootprintRiskLoss:
    """Adapt typed hidden-risk GT to sampled-realization and legacy protocols.

    The configured IDs are explicit label-side hidden objects. An empty tuple
    is valid for an SOP5 child sampled from the target-absent branch.
    """

    hidden_object_ids: tuple[str, ...]
    robot_footprint: Footprint
    grid: GridSpec
    future_dt_s: float
    sigma_distance_m: float
    sigma_time_s: float
    near_miss_distance_m: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.hidden_object_ids, tuple)
            or any(not isinstance(value, str) or not value for value in self.hidden_object_ids)
            or len(set(self.hidden_object_ids)) != len(self.hidden_object_ids)
        ):
            raise ValueError("hidden_object_ids must be a unique tuple")
        if not isinstance(self.robot_footprint, (CircleFootprint, RectangleFootprint)):
            raise TypeError("robot_footprint must be typed circle or rectangle geometry")
        if not isinstance(self.grid, GridSpec):
            raise TypeError("grid must be a GridSpec")
        dt = _finite_nonnegative(self.future_dt_s, name="future_dt_s")
        sigma_distance = _finite_nonnegative(
            self.sigma_distance_m, name="sigma_distance_m"
        )
        sigma_time = _finite_nonnegative(self.sigma_time_s, name="sigma_time_s")
        near_miss = _finite_nonnegative(
            self.near_miss_distance_m, name="near_miss_distance_m"
        )
        if dt <= 0.0 or sigma_distance <= 0.0 or sigma_time <= 0.0:
            raise ValueError("future_dt_s and risk sigmas must be positive")
        object.__setattr__(self, "future_dt_s", dt)
        object.__setattr__(self, "sigma_distance_m", sigma_distance)
        object.__setattr__(self, "sigma_time_s", sigma_time)
        object.__setattr__(self, "near_miss_distance_m", near_miss)

    def __call__(
        self,
        trajectory: LocalTrajectory,
        poses_in_parent_frame: np.ndarray,
        realization: ScenarioHypothesis | OracleWorld,
    ) -> float:
        if isinstance(realization, ScenarioHypothesis):
            world = realization.world
            removable_target = world.metadata.get("scenario_target_object_id")
        elif isinstance(realization, OracleWorld):
            world = realization
            removable_target = None
        else:
            raise TypeError(
                "realization must be an OracleWorld or ScenarioHypothesis"
            )
        if (
            not isinstance(poses_in_parent_frame, np.ndarray)
            or poses_in_parent_frame.dtype != ARRAY_DTYPE
            or poses_in_parent_frame.shape != (self.grid.future_steps, 3)
            or not np.isfinite(poses_in_parent_frame).all()
        ):
            raise ValueError(
                "poses_in_parent_frame must be finite float32 future endpoints"
            )
        available = set(world.dynamic_object_trajectories)
        missing = set(self.hidden_object_ids) - available
        if missing - {removable_target}:
            raise ValueError("non-target hidden object is missing from scenario world")
        hidden_ids = tuple(
            object_id for object_id in self.hidden_object_ids if object_id in available
        )
        parent_trajectory = replace(
            trajectory,
            poses=np.array(
                poses_in_parent_frame,
                dtype=ARRAY_DTYPE,
                order="C",
                copy=True,
            ),
        )
        result = compute_hidden_risk_gt(
            parent_trajectory,
            world,
            hidden_object_ids=hidden_ids,
            robot_footprint=self.robot_footprint,
            grid=self.grid,
            future_dt_s=self.future_dt_s,
            sigma_distance_m=self.sigma_distance_m,
            sigma_time_s=self.sigma_time_s,
            near_miss_distance_m=self.near_miss_distance_m,
        )
        return result.risk_severity


@dataclass(frozen=True)
class VerificationValueResult:
    """Immutable label-side audit record for one verification action."""

    version: str
    bank_size: int
    scenario_bank_digest: str
    nominal_trajectory_id: str
    verification_action_id: str
    posterior_mode: str
    posterior_temperature: float | None
    posterior: np.ndarray
    nominal_execute_losses: np.ndarray
    mean_execute_loss: float
    br_before: float
    post_decision_risks: np.ndarray
    best_decision_ids: tuple[str, ...]
    unclipped_best_policy_losses: tuple[float | None, ...]
    mean_post_decision_risk_before_action_cost: float
    action_cost: float
    post_risk: float
    value_target: float
    useful_target: int
    policy_response_trajectory_ids: tuple[str | None, ...] = ()
    post_plan_statuses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.version != VERIFICATION_GT_VERSION:
            raise ValueError("unsupported verification GT result version")
        if isinstance(self.bank_size, bool) or not isinstance(self.bank_size, int):
            raise TypeError("bank_size must be an integer")
        if self.bank_size <= 0:
            raise ValueError("bank_size must be positive")
        for name in (
            "scenario_bank_digest",
            "nominal_trajectory_id",
            "verification_action_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.posterior_mode not in {"exact", "soft"}:
            raise ValueError("posterior_mode must be exact or soft")
        if self.posterior_mode == "exact":
            if self.posterior_temperature is not None:
                raise ValueError("exact posterior must not record a temperature")
        else:
            temperature = _finite_nonnegative(
                self.posterior_temperature, name="posterior_temperature"
            )
            if temperature <= 0.0:
                raise ValueError("posterior_temperature must be positive")
            object.__setattr__(self, "posterior_temperature", temperature)

        object.__setattr__(
            self,
            "posterior",
            _owned_posterior(self.posterior, size=self.bank_size),
        )
        object.__setattr__(
            self,
            "nominal_execute_losses",
            _owned_float64_vector(
                self.nominal_execute_losses,
                name="nominal_execute_losses",
                size=self.bank_size,
            ),
        )
        post = _owned_float64_vector(
            self.post_decision_risks,
            name="post_decision_risks",
            size=self.bank_size,
        )
        if np.any(post < 0.0):
            raise ValueError("post_decision_risks must be non-negative")
        object.__setattr__(self, "post_decision_risks", post)
        if (
            not isinstance(self.best_decision_ids, tuple)
            or len(self.best_decision_ids) != self.bank_size
            or any(not isinstance(value, str) or not value for value in self.best_decision_ids)
        ):
            raise ValueError("best_decision_ids must align with the scenario bank")
        unclipped = self.unclipped_best_policy_losses
        if not isinstance(unclipped, tuple) or len(unclipped) != self.bank_size:
            raise ValueError(
                "unclipped best policy losses must align with the scenario bank"
            )
        normalized_unclipped: list[float | None] = []
        for index, value in enumerate(unclipped):
            normalized_unclipped.append(
                None
                if value is None
                else _finite_nonnegative(
                    value,
                    name=f"unclipped_best_policy_losses[{index}]",
                )
            )
        object.__setattr__(
            self,
            "unclipped_best_policy_losses",
            tuple(normalized_unclipped),
        )

        for name in (
            "mean_execute_loss",
            "br_before",
            "mean_post_decision_risk_before_action_cost",
            "action_cost",
            "post_risk",
        ):
            object.__setattr__(
                self, name, _finite_nonnegative(getattr(self, name), name=name)
            )
        value = self.value_target
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Real, np.integer, np.floating)
        ):
            raise TypeError("value_target must be a real number")
        value_float = float(value)
        if not np.isfinite(value_float):
            raise ValueError("value_target must be finite")
        object.__setattr__(self, "value_target", value_float)
        if self.useful_target not in (0, 1):
            raise ValueError("useful_target must be binary")
        if self.useful_target != int(value_float > 0.0):
            raise ValueError("useful_target must equal int(value_target > 0)")
        response_ids = self.policy_response_trajectory_ids
        if (
            not isinstance(response_ids, tuple)
            or (response_ids and len(response_ids) != self.bank_size)
            or any(
                value is not None and (not isinstance(value, str) or not value)
                for value in response_ids
            )
        ):
            raise ValueError(
                "policy response trajectory IDs must be empty or align with the bank"
            )
        plan_statuses = self.post_plan_statuses
        if (
            not isinstance(plan_statuses, tuple)
            or len(plan_statuses) != self.bank_size
            or any(value not in POST_PLAN_STATUSES for value in plan_statuses)
        ):
            raise ValueError(
                "post-plan statuses must be empty or align with the scenario bank"
            )
        if not np.isclose(
            self.mean_execute_loss,
            float(np.mean(self.nominal_execute_losses, dtype=np.float64)),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("mean_execute_loss disagrees with per-world losses")
        if not np.isclose(
            self.mean_post_decision_risk_before_action_cost,
            float(np.mean(post, dtype=np.float64)),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("mean post-decision risk disagrees with per-world risks")
        if not np.isclose(
            self.post_risk,
            self.mean_post_decision_risk_before_action_cost + self.action_cost,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("post_risk must add action cost exactly once")
        if not np.isclose(
            self.value_target,
            self.br_before - self.post_risk,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("value_target must equal br_before - post_risk")


@dataclass(frozen=True)
class SampledVerificationValueResult:
    """One Monte Carlo value target for an SOP5-sampled child realization."""

    version: str
    sampled_child_world_id: str
    nominal_trajectory_id: str
    verification_action_id: str
    realized_execute_loss: float
    reject_cost: float
    br_before: float
    realized_post_decision_risk_before_action_cost: float
    best_decision_id: str
    unclipped_best_policy_loss: float | None
    action_cost: float
    post_risk: float
    value_target: float
    useful_target: int
    policy_response_trajectory_id: str | None = None
    post_plan_status: str = "feasible_plan"

    def __post_init__(self) -> None:
        if self.version != SAMPLED_REALIZATION_GT_VERSION:
            raise ValueError("unsupported sampled-realization GT result version")
        for name in (
            "sampled_child_world_id",
            "nominal_trajectory_id",
            "verification_action_id",
            "best_decision_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "realized_execute_loss",
            "reject_cost",
            "br_before",
            "realized_post_decision_risk_before_action_cost",
            "action_cost",
            "post_risk",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        if self.unclipped_best_policy_loss is not None:
            object.__setattr__(
                self,
                "unclipped_best_policy_loss",
                _finite_nonnegative(
                    self.unclipped_best_policy_loss,
                    name="unclipped_best_policy_loss",
                ),
            )
        if self.policy_response_trajectory_id is not None and (
            not isinstance(self.policy_response_trajectory_id, str)
            or not self.policy_response_trajectory_id
        ):
            raise ValueError(
                "policy_response_trajectory_id must be None or non-empty"
            )
        if self.post_plan_status not in POST_PLAN_STATUSES:
            raise ValueError("unsupported post-verification plan status")
        value = self.value_target
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Real, np.integer, np.floating)
        ):
            raise TypeError("value_target must be a real number")
        value_float = float(value)
        if not np.isfinite(value_float):
            raise ValueError("value_target must be finite")
        object.__setattr__(self, "value_target", value_float)
        if self.useful_target not in (0, 1):
            raise ValueError("useful_target must be binary")
        if self.useful_target != int(value_float > 0.0):
            raise ValueError("useful_target must equal int(value_target > 0)")
        if not np.isclose(
            self.br_before,
            min(self.realized_execute_loss, self.reject_cost),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "br_before must be the sampled execute/reject minimum"
            )
        if not np.isclose(
            self.post_risk,
            self.realized_post_decision_risk_before_action_cost
            + self.action_cost,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("post_risk must add action cost exactly once")
        if not np.isclose(
            self.value_target,
            self.br_before - self.post_risk,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("value_target must equal br_before - post_risk")


def load_verification_gt_config(path: str | Path) -> VerificationGTConfig:
    """Load the frozen decision section without accepting alternate keys."""

    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid verification GT config: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != _TOP_KEYS:
        raise ValueError("verification GT config keys are invalid")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"verification GT schema must be {SCHEMA_VERSION}")
    decision = raw["decision"]
    if not isinstance(decision, dict) or set(decision) != _DECISION_KEYS:
        raise ValueError("verification decision config keys are invalid")
    return VerificationGTConfig(
        reject_cost=decision["reject_cost"],
        risk_weight=decision["risk_weight"],
        braking_deceleration_mps2=decision["braking_deceleration_mps2"],
        angular_deceleration_radps2=decision[
            "angular_deceleration_radps2"
        ],
        braking_margin_s=decision["braking_margin_s"],
    )


def _risk_value(
    risk_loss: RiskLoss,
    trajectory: LocalTrajectory,
    poses_in_parent_frame: np.ndarray,
    hypothesis: ScenarioHypothesis,
) -> float:
    try:
        value = risk_loss(trajectory, poses_in_parent_frame, hypothesis)
    except Exception:
        raise
    return _finite_nonnegative(value, name="risk loss")


def _realized_risk_value(
    risk_loss: RealizedRiskLoss,
    trajectory: LocalTrajectory,
    poses_in_parent_frame: np.ndarray,
    realized_world: OracleWorld,
) -> float:
    value = risk_loss(trajectory, poses_in_parent_frame, realized_world)
    return _finite_nonnegative(value, name="realized risk loss")


def _trajectory_task_cost(trajectory: LocalTrajectory) -> float:
    if not isinstance(trajectory, LocalTrajectory):
        raise TypeError("trajectory must be a LocalTrajectory")
    return _finite_nonnegative(trajectory.task_cost, name="trajectory task_cost")


def relative_task_regret(
    policy_task_cost: Any,
    *,
    nominal_task_cost: Any,
) -> float:
    """Return non-negative task degradation relative to the nominal plan."""

    policy = _finite_nonnegative(policy_task_cost, name="policy_task_cost")
    nominal = _finite_nonnegative(
        nominal_task_cost,
        name="nominal_task_cost",
    )
    if nominal <= 0.0:
        raise ValueError("nominal_task_cost must be positive")
    return max(0.0, policy / nominal - 1.0)


def _validate_replanning_result(
    result: ReplanningResult,
    *,
    nominal_trajectory: LocalTrajectory,
    action: VerificationAction,
) -> tuple[ReplannedCandidate, ...]:
    if not isinstance(result, ReplanningResult):
        raise TypeError("replanning_results must contain ReplanningResult values")
    if not np.allclose(
        result.task_anchor_pose,
        nominal_trajectory.poses[-1],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("replanning task anchor must equal the nominal endpoint")
    candidates = tuple(sorted(result.candidates, key=lambda item: item.trajectory.trajectory_id))
    ids = tuple(item.trajectory.trajectory_id for item in candidates)
    if len(set(ids)) != len(ids):
        raise ValueError("replanned candidate IDs must be unique")
    for candidate in candidates:
        metadata = candidate.trajectory.metadata
        if not isinstance(metadata, Mapping):
            raise TypeError("replanned trajectory metadata must be a mapping")
        if metadata.get("nominal_suffix_used") is not False:
            raise ValueError("replanned candidate must not use a nominal suffix")
        if metadata.get("sampling_origin") != "post_action_pose":
            raise ValueError("replanned candidate must be sampled from the post-action pose")
        if metadata.get("nominal_trajectory_id") != nominal_trajectory.trajectory_id:
            raise ValueError("replanned candidate nominal trajectory ID mismatch")
        if metadata.get("action_id") != action.action_id:
            raise ValueError("replanned candidate verification action ID mismatch")
        _trajectory_task_cost(candidate.trajectory)
    return candidates


def _validate_time_aligned_policy_group(
    trajectories: Sequence[LocalTrajectory],
    *,
    replanning_result: ReplanningResult,
    replanned_candidates: tuple[ReplannedCandidate, ...],
    nominal_trajectory: LocalTrajectory,
    action: VerificationAction,
) -> tuple[tuple[LocalTrajectory, ...], str | None]:
    if isinstance(trajectories, (str, bytes)) or not isinstance(
        trajectories, Sequence
    ):
        raise TypeError("time-aligned policy groups must be sequences")
    policies = tuple(trajectories)
    if not policies:
        raise ValueError("time-aligned policy groups must be non-empty")
    if any(not isinstance(value, LocalTrajectory) for value in policies):
        raise TypeError("time-aligned policy groups must contain LocalTrajectory values")
    policies = tuple(sorted(policies, key=lambda value: value.trajectory_id))
    policy_ids = tuple(value.trajectory_id for value in policies)
    if any(not value for value in policy_ids) or len(set(policy_ids)) != len(policy_ids):
        raise ValueError("time-aligned policy trajectory IDs must be non-empty and unique")

    replanned_by_id = {
        candidate.trajectory.trajectory_id: candidate
        for candidate in replanned_candidates
    }
    observed_suffix_ids: set[str] = set()
    response_trajectory_id: str | None = None
    expected_branch_kind: str | None = None
    expected_branch_end_time_s: float | None = None
    expected_branch_trigger_time_s: float | None = None
    expected_planned_action_end_time_s: float | None = None
    expected_branch_end_control: np.ndarray | None = None
    for trajectory in policies:
        for name, value, shape in (
            ("poses", trajectory.poses, nominal_trajectory.poses.shape),
            ("controls", trajectory.controls, nominal_trajectory.controls.shape),
        ):
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != ARRAY_DTYPE
                or value.shape != shape
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    f"time-aligned policy {name} must be finite float32 with shape {shape}"
                )
        metadata = trajectory.metadata
        if not isinstance(metadata, Mapping):
            raise TypeError("time-aligned policy metadata must be a mapping")
        if metadata.get("label_side_policy_trajectory") is not True:
            raise ValueError("time-aligned policy must be marked as label-side")
        if metadata.get("absolute_time_aligned") is not True:
            raise ValueError("policy trajectory must be aligned to absolute future time")
        if metadata.get("pose_time_layout_version") != POSE_TIME_LAYOUT_VERSION:
            raise ValueError("time-aligned policy must use Long40 future endpoints")
        if metadata.get("response_version") != VERIFICATION_RESPONSE_VERSION:
            raise ValueError("time-aligned policy response version is invalid")
        if metadata.get("source_action_id") != action.action_id:
            raise ValueError("time-aligned policy verification action ID mismatch")
        if (
            metadata.get("source_nominal_trajectory_id")
            != nominal_trajectory.trajectory_id
        ):
            raise ValueError("time-aligned policy nominal trajectory ID mismatch")

        branch_kind = metadata.get("branch_kind")
        if branch_kind not in {
            "complete",
            "observe_and_replan",
            "emergency_brake",
        }:
            raise ValueError("time-aligned policy branch kind is invalid")
        branch_end_time_s = _finite_nonnegative(
            metadata.get("branch_end_time_s"), name="branch_end_time_s"
        )
        if branch_end_time_s > LONG40_FUTURE_HORIZON_S + 1e-12:
            raise ValueError("time-aligned policy branch exceeds the Long40 horizon")
        planned_action_end_time_s = _finite_nonnegative(
            metadata.get("planned_action_end_time_s"),
            name="planned_action_end_time_s",
        )
        if planned_action_end_time_s > LONG40_FUTURE_HORIZON_S + 1e-12:
            raise ValueError("planned action exceeds the Long40 horizon")
        if planned_action_end_time_s + 1e-12 < action.duration_s:
            raise ValueError(
                "planned action duration must include the verification primitive"
            )
        trigger_value = metadata.get("branch_trigger_time_s")
        if branch_kind == "complete":
            if trigger_value is not None:
                raise ValueError("complete policy branch must not have a trigger time")
            if not np.isclose(
                branch_end_time_s,
                planned_action_end_time_s,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "complete policy branch must consume the planned action duration"
                )
            branch_trigger_time_s = None
        else:
            branch_trigger_time_s = _finite_nonnegative(
                trigger_value, name="branch_trigger_time_s"
            )
            if branch_trigger_time_s > planned_action_end_time_s + 1e-12:
                raise ValueError(
                    "interrupted policy trigger exceeds the planned action duration"
                )
            if branch_trigger_time_s > branch_end_time_s:
                raise ValueError("policy trigger time must not exceed branch end time")
            if branch_kind == "observe_and_replan" and not np.isclose(
                branch_trigger_time_s,
                branch_end_time_s,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "observe-and-replan policy must end at its trigger time"
                )
        branch_end_pose = np.asarray(metadata.get("branch_end_pose"))
        if (
            branch_end_pose.shape != (3,)
            or branch_end_pose.dtype.kind not in "iuf"
            or not np.isfinite(branch_end_pose).all()
            or not np.allclose(
                branch_end_pose,
                replanning_result.post_action_pose,
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ValueError(
                "time-aligned policy branch endpoint must equal the replanning start"
            )
        branch_end_control = np.asarray(metadata.get("branch_end_control"))
        if (
            branch_end_control.shape != (2,)
            or branch_end_control.dtype.kind not in "iuf"
            or not np.isfinite(branch_end_control).all()
        ):
            raise ValueError("time-aligned policy branch end control is invalid")
        if branch_kind != "observe_and_replan" and not np.allclose(
            branch_end_control,
            0.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "complete and emergency-brake policies must end at rest"
            )
        if expected_branch_kind is None:
            expected_branch_kind = branch_kind
            expected_branch_end_time_s = branch_end_time_s
            expected_branch_trigger_time_s = branch_trigger_time_s
            expected_planned_action_end_time_s = planned_action_end_time_s
            expected_branch_end_control = branch_end_control.copy()
        elif (
            branch_kind != expected_branch_kind
            or not np.isclose(
                branch_end_time_s,
                expected_branch_end_time_s,
                rtol=0.0,
                atol=1e-12,
            )
            or branch_trigger_time_s != expected_branch_trigger_time_s
            or not np.isclose(
                planned_action_end_time_s,
                expected_planned_action_end_time_s,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                branch_end_control,
                expected_branch_end_control,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError("time-aligned policy group mixes branch histories")

        suffix_id = metadata.get("suffix_trajectory_id")
        if suffix_id is None:
            if (
                branch_kind != "emergency_brake"
                or metadata.get("response_id") != "emergency_brake"
                or response_trajectory_id is not None
            ):
                raise ValueError(
                    "only one emergency-brake hold response may omit a replan suffix"
                )
            response_trajectory_id = trajectory.trajectory_id
        else:
            if not isinstance(suffix_id, str) or suffix_id not in replanned_by_id:
                raise ValueError("time-aligned policy suffix is not a replanned candidate")
            if suffix_id in observed_suffix_ids:
                raise ValueError("time-aligned policy contains a duplicate replan suffix")
            observed_suffix_ids.add(suffix_id)
            expected_task = _trajectory_task_cost(
                replanned_by_id[suffix_id].trajectory
            )
            if not np.isclose(
                _trajectory_task_cost(trajectory),
                expected_task,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("time-aligned policy changed the replan task cost")
        _trajectory_task_cost(trajectory)

    if observed_suffix_ids != set(replanned_by_id):
        raise ValueError("time-aligned policy group must cover every replanned candidate")
    if expected_branch_kind == "emergency_brake" and response_trajectory_id is None:
        raise ValueError("emergency-brake policy group must include a hold response")
    if (
        expected_branch_kind in {"complete", "observe_and_replan"}
        and response_trajectory_id is not None
    ):
        raise ValueError(
            "non-emergency policy branch must not include a braking response"
        )
    return policies, response_trajectory_id


def evaluate_sampled_realization_value(
    *,
    realized_world: OracleWorld,
    nominal_trajectory: LocalTrajectory,
    action: VerificationAction,
    replanning_result: ReplanningResult,
    risk_loss: RealizedRiskLoss,
    reject_cost: float,
    risk_weight: float,
    action_cost_config: Mapping[str, Any],
    time_aligned_policy_trajectories: Sequence[LocalTrajectory],
) -> SampledVerificationValueResult:
    """Compute one realized child label without constructing latent variants."""

    if not isinstance(realized_world, OracleWorld):
        raise TypeError("realized_world must be an OracleWorld")
    if not isinstance(realized_world.world_id, str) or not realized_world.world_id:
        raise ValueError("realized_world.world_id must be non-empty")
    if not isinstance(nominal_trajectory, LocalTrajectory):
        raise TypeError("nominal_trajectory must be a LocalTrajectory")
    if not isinstance(action, VerificationAction):
        raise TypeError("action must be a VerificationAction")
    if not isinstance(replanning_result, ReplanningResult):
        raise TypeError("replanning_result must be a ReplanningResult")
    if not callable(risk_loss):
        raise TypeError("risk_loss must be callable")
    if isinstance(time_aligned_policy_trajectories, (str, bytes)) or not isinstance(
        time_aligned_policy_trajectories,
        Sequence,
    ):
        raise TypeError("time_aligned_policy_trajectories must be a sequence")

    reject = _finite_nonnegative(reject_cost, name="reject_cost")
    weight = _finite_nonnegative(risk_weight, name="risk_weight")
    nominal_task = _trajectory_task_cost(nominal_trajectory)
    if nominal_task <= 0.0:
        raise ValueError("nominal trajectory task_cost must be positive")
    realized_execute_loss = weight * _realized_risk_value(
        risk_loss,
        nominal_trajectory,
        nominal_trajectory.poses,
        realized_world,
    )
    br_before = min(realized_execute_loss, reject)

    policy_group = tuple(time_aligned_policy_trajectories)
    plan_status = replanning_result.plan_status
    if plan_status == POST_PLAN_STATUS_SAFE_STOP_NO_FEASIBLE_PLAN:
        replanned_candidates = _validate_replanning_result(
            replanning_result,
            nominal_trajectory=nominal_trajectory,
            action=action,
        )
        if replanned_candidates:
            raise ValueError(
                "safe-stop no-feasible-plan results must not contain candidates"
            )
        if policy_group:
            raise ValueError(
                "safe-stop no-feasible-plan bypass must not expose trajectories"
            )
        realized_post = reject
        best_decision_id = "reject"
        unclipped_best_policy_loss = None
        response_trajectory_id = None
    else:
        replanned_candidates = _validate_replanning_result(
            replanning_result,
            nominal_trajectory=nominal_trajectory,
            action=action,
        )
        policies, response_trajectory_id = _validate_time_aligned_policy_group(
            policy_group,
            replanning_result=replanning_result,
            replanned_candidates=replanned_candidates,
            nominal_trajectory=nominal_trajectory,
            action=action,
        )
        best_policy_loss: float | None = None
        best_policy_id: str | None = None
        for policy in policies:
            policy_loss = relative_task_regret(
                _trajectory_task_cost(policy),
                nominal_task_cost=nominal_task,
            ) + weight * _realized_risk_value(
                risk_loss,
                policy,
                policy.poses,
                realized_world,
            )
            if best_policy_loss is None or policy_loss < best_policy_loss:
                best_policy_loss = policy_loss
                best_policy_id = policy.trajectory_id
        unclipped_best_policy_loss = best_policy_loss
        if best_policy_loss is not None and best_policy_loss < reject:
            realized_post = best_policy_loss
            best_decision_id = str(best_policy_id)
        else:
            realized_post = reject
            best_decision_id = "reject"

    verification_cost = compute_action_cost(action, action_cost_config)
    post_risk = realized_post + verification_cost
    value_target = br_before - post_risk
    return SampledVerificationValueResult(
        version=SAMPLED_REALIZATION_GT_VERSION,
        sampled_child_world_id=realized_world.world_id,
        nominal_trajectory_id=nominal_trajectory.trajectory_id,
        verification_action_id=action.action_id,
        realized_execute_loss=realized_execute_loss,
        reject_cost=reject,
        br_before=br_before,
        realized_post_decision_risk_before_action_cost=realized_post,
        best_decision_id=best_decision_id,
        unclipped_best_policy_loss=unclipped_best_policy_loss,
        action_cost=verification_cost,
        post_risk=post_risk,
        value_target=value_target,
        useful_target=int(value_target > 0.0),
        policy_response_trajectory_id=response_trajectory_id,
        post_plan_status=plan_status,
    )


def _build_posterior(
    *,
    mode: str,
    observations: Sequence[CounterfactualObservation],
    signatures: np.ndarray | None,
    normalizer: SignatureNormalizer | None,
    temperature: float | None,
) -> np.ndarray:
    if mode == "exact":
        if normalizer is not None or temperature is not None:
            raise ValueError("exact posterior does not use a normalizer or temperature")
        digests = tuple(observable_observation_digest(value) for value in observations)
        return exact_observation_posterior(digests)
    if mode != "soft":
        raise ValueError("posterior_mode must be exact or soft")
    if signatures is None:
        raise ValueError("soft posterior requires observation signatures")
    if normalizer is None:
        raise ValueError("soft posterior requires a train-fitted normalizer")
    if temperature is None:
        raise ValueError("soft posterior requires a temperature")
    return soft_observation_posterior(
        signatures,
        normalizer=normalizer,
        temperature=temperature,
    )


def evaluate_verification_value(
    *,
    bank: ScenarioBank,
    nominal_trajectory: LocalTrajectory,
    action: VerificationAction,
    observations: Sequence[CounterfactualObservation],
    signatures: np.ndarray | None,
    replanning_results: Sequence[ReplanningResult],
    risk_loss: RiskLoss,
    posterior_mode: str,
    signature_normalizer: SignatureNormalizer | None,
    posterior_temperature: float | None,
    reject_cost: float,
    risk_weight: float,
    action_cost_config: Mapping[str, Any],
    time_aligned_policy_trajectories: (
        Sequence[Sequence[LocalTrajectory]] | None
    ) = None,
) -> VerificationValueResult:
    """Compute simulator-defined net value without exposing oracle data as input."""

    if not isinstance(bank, ScenarioBank):
        raise TypeError("bank must be a ScenarioBank")
    if bank.size <= 0 or bank.size != len(bank.hypotheses):
        raise ValueError("scenario bank must be non-empty and internally aligned")
    if not isinstance(nominal_trajectory, LocalTrajectory):
        raise TypeError("nominal_trajectory must be a LocalTrajectory")
    if not isinstance(action, VerificationAction):
        raise TypeError("action must be a VerificationAction")
    if not callable(risk_loss):
        raise TypeError("risk_loss must be callable")
    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise TypeError("observations must be a sequence")
    if len(observations) != bank.size or any(
        not isinstance(value, CounterfactualObservation) for value in observations
    ):
        raise ValueError("observations must align one-to-one with the scenario bank")
    if isinstance(replanning_results, (str, bytes)) or not isinstance(
        replanning_results, Sequence
    ):
        raise TypeError("replanning_results must be a sequence")
    if len(replanning_results) != bank.size:
        raise ValueError("replanning_results must align one-to-one with the scenario bank")
    if time_aligned_policy_trajectories is None:
        raise ValueError(
            "Long40 verification GT requires time-aligned policy groups"
        )
    if isinstance(time_aligned_policy_trajectories, (str, bytes)) or not isinstance(
        time_aligned_policy_trajectories, Sequence
    ):
        raise TypeError("time_aligned_policy_trajectories must be a sequence")
    if len(time_aligned_policy_trajectories) != bank.size:
        raise ValueError(
            "time-aligned policy groups must align one-to-one with the scenario bank"
        )
    aligned_policy_groups = tuple(time_aligned_policy_trajectories)

    reject = _finite_nonnegative(reject_cost, name="reject_cost")
    weight = _finite_nonnegative(risk_weight, name="risk_weight")
    nominal_task = _trajectory_task_cost(nominal_trajectory)
    if nominal_task <= 0.0:
        raise ValueError("nominal trajectory task_cost must be positive")
    posterior = _build_posterior(
        mode=posterior_mode,
        observations=observations,
        signatures=signatures,
        normalizer=signature_normalizer,
        temperature=posterior_temperature,
    )
    validate_posterior_matrix(posterior, size=bank.size)

    nominal_losses = np.empty(bank.size, dtype=np.float64)
    for world_index, hypothesis in enumerate(bank.hypotheses):
        risk = _risk_value(
            risk_loss,
            nominal_trajectory,
            nominal_trajectory.poses,
            hypothesis,
        )
        nominal_losses[world_index] = weight * risk
    mean_execute = float(np.mean(nominal_losses, dtype=np.float64))
    br_before = min(mean_execute, reject)

    post_decision_risks = np.empty(bank.size, dtype=np.float64)
    best_ids: list[str] = []
    unclipped_best_policy_losses: list[float | None] = []
    prepared_aligned: dict[
        tuple[int, int], tuple[tuple[LocalTrajectory, ...], np.ndarray, str | None]
    ] = {}
    response_audit_ids: list[str | None] = []
    post_plan_statuses: list[str] = []
    for observed_world_index, result in enumerate(replanning_results):
        group = aligned_policy_groups[observed_world_index]
        plan_status = result.plan_status
        post_plan_statuses.append(plan_status)
        if plan_status == POST_PLAN_STATUS_SAFE_STOP_NO_FEASIBLE_PLAN:
            replanned_candidates = _validate_replanning_result(
                result,
                nominal_trajectory=nominal_trajectory,
                action=action,
            )
            if replanned_candidates:
                raise ValueError(
                    "safe-stop no-feasible-plan results must not contain candidates"
                )
            if isinstance(group, (str, bytes)) or not isinstance(group, Sequence):
                raise TypeError("time-aligned policy groups must be sequences")
            if tuple(group):
                raise ValueError(
                    "safe-stop no-feasible-plan bypass must not expose trajectories"
                )
            post_decision_risks[observed_world_index] = reject
            best_ids.append("reject")
            unclipped_best_policy_losses.append(None)
            response_audit_ids.append(None)
            continue
        prepared_key = (id(result), id(group))
        aligned_prepared = prepared_aligned.get(prepared_key)
        if aligned_prepared is None:
            replanned_candidates = _validate_replanning_result(
                result,
                nominal_trajectory=nominal_trajectory,
                action=action,
            )
            policies, response_id = _validate_time_aligned_policy_group(
                group,
                replanning_result=result,
                replanned_candidates=replanned_candidates,
                nominal_trajectory=nominal_trajectory,
                action=action,
            )
            policy_world_losses = np.empty(
                (len(policies), bank.size), dtype=np.float64
            )
            for policy_index, policy in enumerate(policies):
                task_regret = relative_task_regret(
                    _trajectory_task_cost(policy),
                    nominal_task_cost=nominal_task,
                )
                for world_index, hypothesis in enumerate(bank.hypotheses):
                    policy_world_losses[policy_index, world_index] = (
                        task_regret
                        + weight
                        * _risk_value(
                            risk_loss,
                            policy,
                            policy.poses,
                            hypothesis,
                        )
                    )
            aligned_prepared = (policies, policy_world_losses, response_id)
            prepared_aligned[prepared_key] = aligned_prepared
        policies, policy_world_losses, response_id = aligned_prepared
        best_policy_loss: float | None = None
        best_policy_id: str | None = None
        posterior_row = posterior[observed_world_index]
        for policy_index, policy in enumerate(policies):
            expected = float(
                np.dot(posterior_row, policy_world_losses[policy_index])
            )
            if not np.isfinite(expected) or expected < 0.0:
                raise ValueError(
                    "posterior expected policy loss must be finite and non-negative"
                )
            if best_policy_loss is None or expected < best_policy_loss:
                best_policy_loss = expected
                best_policy_id = policy.trajectory_id
        unclipped_best_policy_losses.append(best_policy_loss)
        if best_policy_loss is not None and best_policy_loss < reject:
            best_loss = best_policy_loss
            best_id = str(best_policy_id)
        else:
            best_loss = reject
            best_id = "reject"
        post_decision_risks[observed_world_index] = best_loss
        best_ids.append(best_id)
        response_audit_ids.append(response_id)

    mean_post = float(np.mean(post_decision_risks, dtype=np.float64))
    verification_cost = compute_action_cost(action, action_cost_config)
    post_risk = mean_post + verification_cost
    value_target = br_before - post_risk
    return VerificationValueResult(
        version=VERIFICATION_GT_VERSION,
        bank_size=bank.size,
        scenario_bank_digest=bank.semantic_digest,
        nominal_trajectory_id=nominal_trajectory.trajectory_id,
        verification_action_id=action.action_id,
        posterior_mode=posterior_mode,
        posterior_temperature=(
            None if posterior_mode == "exact" else float(posterior_temperature)
        ),
        posterior=posterior,
        nominal_execute_losses=nominal_losses,
        mean_execute_loss=mean_execute,
        br_before=br_before,
        post_decision_risks=post_decision_risks,
        best_decision_ids=tuple(best_ids),
        unclipped_best_policy_losses=tuple(unclipped_best_policy_losses),
        mean_post_decision_risk_before_action_cost=mean_post,
        action_cost=verification_cost,
        post_risk=post_risk,
        value_target=value_target,
        useful_target=int(value_target > 0.0),
        policy_response_trajectory_ids=tuple(response_audit_ids),
        post_plan_statuses=tuple(post_plan_statuses),
    )


__all__ = (
    "RealizedRiskLoss",
    "RiskLoss",
    "SAMPLED_REALIZATION_GT_VERSION",
    "SampledVerificationValueResult",
    "TypedFootprintRiskLoss",
    "VERIFICATION_GT_VERSION",
    "VerificationGTConfig",
    "VerificationValueResult",
    "evaluate_sampled_realization_value",
    "evaluate_verification_value",
    "load_verification_gt_config",
    "relative_task_regret",
)
