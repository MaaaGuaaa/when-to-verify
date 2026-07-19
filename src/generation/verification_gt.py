"""Net verification value under a finite, observation-consistent world bank."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.contracts import ARRAY_DTYPE, GridSpec, LocalTrajectory, SCHEMA_VERSION
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
from src.planning.replanning import ReplannedCandidate, ReplanningResult
from src.planning.verification_actions import (
    VerificationAction,
    action_cost as compute_action_cost,
)


VERIFICATION_GT_VERSION = "verification_value_gt_v1"
_TOP_KEYS = frozenset({"schema_version", "scenario_bank", "posterior", "decision"})
_DECISION_KEYS = frozenset(
    {"reject_cost", "risk_weight", "braking_deceleration_mps2"}
)

RiskLoss = Callable[[LocalTrajectory, np.ndarray, ScenarioHypothesis], float]


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

    def __post_init__(self) -> None:
        reject = _finite_nonnegative(self.reject_cost, name="reject_cost")
        weight = _finite_nonnegative(self.risk_weight, name="risk_weight")
        braking = _finite_nonnegative(
            self.braking_deceleration_mps2,
            name="braking_deceleration_mps2",
        )
        if braking <= 0.0:
            raise ValueError("braking_deceleration_mps2 must be positive")
        object.__setattr__(self, "reject_cost", reject)
        object.__setattr__(self, "risk_weight", weight)
        object.__setattr__(self, "braking_deceleration_mps2", braking)


@dataclass(frozen=True)
class TypedFootprintRiskLoss:
    """Adapt the existing typed hidden-risk implementation to the loss protocol.

    The configured IDs are explicit label-side hidden objects. Scenario-bank
    ``empty`` variants may remove only their declared target; any other missing
    ID is rejected instead of being silently ignored.
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
            or not self.hidden_object_ids
            or any(not isinstance(value, str) or not value for value in self.hidden_object_ids)
            or len(set(self.hidden_object_ids)) != len(self.hidden_object_ids)
        ):
            raise ValueError("hidden_object_ids must be a non-empty unique tuple")
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
        hypothesis: ScenarioHypothesis,
    ) -> float:
        if not isinstance(hypothesis, ScenarioHypothesis):
            raise TypeError("hypothesis must be a ScenarioHypothesis")
        if (
            not isinstance(poses_in_parent_frame, np.ndarray)
            or poses_in_parent_frame.dtype != ARRAY_DTYPE
            or poses_in_parent_frame.shape != (self.grid.future_steps, 3)
            or not np.isfinite(poses_in_parent_frame).all()
        ):
            raise ValueError(
                "poses_in_parent_frame must be finite float32 future endpoints"
            )
        available = set(hypothesis.world.dynamic_object_trajectories)
        missing = set(self.hidden_object_ids) - available
        scenario_target = hypothesis.world.metadata.get("scenario_target_object_id")
        if missing - {scenario_target}:
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
            hypothesis.world,
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
    mean_post_decision_risk_before_action_cost: float
    action_cost: float
    post_risk: float
    value_target: float
    useful_target: int

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


def _trajectory_task_cost(trajectory: LocalTrajectory) -> float:
    if not isinstance(trajectory, LocalTrajectory):
        raise TypeError("trajectory must be a LocalTrajectory")
    return _finite_nonnegative(trajectory.task_cost, name="trajectory task_cost")


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

    reject = _finite_nonnegative(reject_cost, name="reject_cost")
    weight = _finite_nonnegative(risk_weight, name="risk_weight")
    nominal_task = _trajectory_task_cost(nominal_trajectory)
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
        nominal_losses[world_index] = nominal_task + weight * risk
    mean_execute = float(np.mean(nominal_losses, dtype=np.float64))
    br_before = min(mean_execute, reject)

    post_decision_risks = np.empty(bank.size, dtype=np.float64)
    best_ids: list[str] = []
    prepared_results: dict[
        int, tuple[tuple[ReplannedCandidate, ...], np.ndarray]
    ] = {}
    for observed_world_index, result in enumerate(replanning_results):
        result_key = id(result)
        prepared = prepared_results.get(result_key)
        if prepared is None:
            candidates = _validate_replanning_result(
                result,
                nominal_trajectory=nominal_trajectory,
                action=action,
            )
            candidate_world_losses = np.empty(
                (len(candidates), bank.size), dtype=np.float64
            )
            for candidate_index, candidate in enumerate(candidates):
                task = _trajectory_task_cost(candidate.trajectory)
                for world_index, hypothesis in enumerate(bank.hypotheses):
                    candidate_world_losses[candidate_index, world_index] = (
                        task
                        + weight
                        * _risk_value(
                            risk_loss,
                            candidate.trajectory,
                            candidate.poses_in_parent_frame,
                            hypothesis,
                        )
                    )
            prepared = (candidates, candidate_world_losses)
            prepared_results[result_key] = prepared
        candidates, candidate_world_losses = prepared
        best_loss = reject
        best_id = "reject"
        posterior_row = posterior[observed_world_index]
        for candidate_index, candidate in enumerate(candidates):
            expected = float(
                np.dot(
                    posterior_row,
                    candidate_world_losses[candidate_index],
                )
            )
            if not np.isfinite(expected) or expected < 0.0:
                raise ValueError("posterior expected decision loss must be finite and non-negative")
            if expected < best_loss:
                best_loss = float(expected)
                best_id = candidate.trajectory.trajectory_id
        post_decision_risks[observed_world_index] = best_loss
        best_ids.append(best_id)

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
        mean_post_decision_risk_before_action_cost=mean_post,
        action_cost=verification_cost,
        post_risk=post_risk,
        value_target=value_target,
        useful_target=int(value_target > 0.0),
    )


__all__ = (
    "RiskLoss",
    "TypedFootprintRiskLoss",
    "VERIFICATION_GT_VERSION",
    "VerificationGTConfig",
    "VerificationValueResult",
    "evaluate_verification_value",
    "load_verification_gt_config",
)
