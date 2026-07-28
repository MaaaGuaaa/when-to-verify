"""Deterministic SOP-15 toy closed loop.

This module is a framework fixture, not a production collision simulator.  It
keeps a single decision-time future budget and exposes explicit termination
reasons so future model/world adapters can replace the toy risk oracle without
changing decision or result interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from src.planning.decision_policy import (
    DECISION_STRATEGIES,
    DecisionRequest,
    DecisionResult,
    select_decision,
)


CLOSED_LOOP_VERSION = "sop15_toy_closed_loop_v1"
TOY_SCIENTIFIC_STATUS = "toy_framework_only"


def _finite_real(value: Any, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _positive_real(value: Any, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _action_values(value: Mapping[str, Real]) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("action_values must be a mapping")
    normalized: dict[str, float] = {}
    for action_id, score in value.items():
        if not isinstance(action_id, str) or not action_id:
            raise ValueError("action_values keys must be non-empty strings")
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError("action_values scores must be real numbers")
        numeric = float(score)
        if not math.isfinite(numeric):
            raise ValueError("action_values scores must be finite")
        normalized[action_id] = numeric
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class ClosedLoopConfig:
    """Fixed timing and safety parameters for a decision-time toy episode."""

    future_horizon_s: float = 6.4
    execute_step_s: float = 0.2
    verify_step_s: float = 0.6
    risk_weight: float = 1.0
    verify_margin: float = 0.01
    collision_risk_threshold: float = 0.7
    near_miss_risk_threshold: float = 0.4
    max_decisions: int = 32

    def __post_init__(self) -> None:
        horizon = _positive_real(self.future_horizon_s, name="future_horizon_s")
        if not math.isclose(horizon, 6.4, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("future_horizon_s must equal the Long40 6.4 s window")
        execute_step = _positive_real(self.execute_step_s, name="execute_step_s")
        verify_step = _positive_real(self.verify_step_s, name="verify_step_s")
        if execute_step > horizon or verify_step > horizon:
            raise ValueError("closed-loop action duration exceeds the future horizon")
        risk_weight = _finite_real(self.risk_weight, name="risk_weight")
        margin = _finite_real(self.verify_margin, name="verify_margin")
        collision_threshold = _finite_real(
            self.collision_risk_threshold,
            name="collision_risk_threshold",
        )
        near_miss_threshold = _finite_real(
            self.near_miss_risk_threshold,
            name="near_miss_risk_threshold",
        )
        if collision_threshold > 1.0 or near_miss_threshold > collision_threshold:
            raise ValueError("risk thresholds must satisfy 0 <= near_miss <= collision <= 1")
        if isinstance(self.max_decisions, bool) or not isinstance(self.max_decisions, Integral):
            raise TypeError("max_decisions must be an integer")
        if self.max_decisions <= 0:
            raise ValueError("max_decisions must be positive")
        object.__setattr__(self, "future_horizon_s", horizon)
        object.__setattr__(self, "execute_step_s", execute_step)
        object.__setattr__(self, "verify_step_s", verify_step)
        object.__setattr__(self, "risk_weight", risk_weight)
        object.__setattr__(self, "verify_margin", margin)
        object.__setattr__(self, "collision_risk_threshold", collision_threshold)
        object.__setattr__(self, "near_miss_risk_threshold", near_miss_threshold)
        object.__setattr__(self, "max_decisions", int(self.max_decisions))


@dataclass(frozen=True)
class ToyScenario:
    """A deterministic scalar-risk fixture with a typed hidden hazard label."""

    episode_id: str
    initial_calibrated_risk: float
    post_verify_calibrated_risk: float
    task_cost: float
    reject_cost: float
    required_execute_time_s: float
    action_values: Mapping[str, Real]
    hazard_object_type: str
    action_values_by_strategy: Mapping[str, Mapping[str, Real]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be a non-empty string")
        if self.hazard_object_type not in {
            "human",
            "carried_object",
            "unknown_dynamic",
        }:
            raise ValueError("hazard_object_type must be a supported dynamic type")
        for name in ("initial_calibrated_risk", "post_verify_calibrated_risk"):
            value = _finite_real(getattr(self, name), name=name)
            if value > 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "task_cost", _finite_real(self.task_cost, name="task_cost"))
        object.__setattr__(self, "reject_cost", _finite_real(self.reject_cost, name="reject_cost"))
        object.__setattr__(
            self,
            "required_execute_time_s",
            _positive_real(self.required_execute_time_s, name="required_execute_time_s"),
        )
        object.__setattr__(self, "action_values", _action_values(self.action_values))
        if self.action_values_by_strategy is None:
            object.__setattr__(self, "action_values_by_strategy", MappingProxyType({}))
            return
        if not isinstance(self.action_values_by_strategy, Mapping):
            raise TypeError("action_values_by_strategy must be a mapping")
        normalized: dict[str, Mapping[str, float]] = {}
        for strategy, values in self.action_values_by_strategy.items():
            if strategy not in DECISION_STRATEGIES:
                raise ValueError("action_values_by_strategy has an unsupported strategy")
            normalized[strategy] = _action_values(values)
        object.__setattr__(
            self,
            "action_values_by_strategy",
            MappingProxyType(dict(sorted(normalized.items()))),
        )

    def action_values_for(self, strategy: str) -> Mapping[str, float]:
        if strategy not in DECISION_STRATEGIES:
            raise ValueError(f"unsupported decision strategy: {strategy!r}")
        return self.action_values_by_strategy.get(strategy, self.action_values)


@dataclass(frozen=True)
class EpisodeStep:
    index: int
    decision: str
    action_id: str | None
    calibrated_risk: float
    elapsed_before_s: float
    elapsed_after_s: float
    remaining_before_s: float
    remaining_after_s: float
    replanned_after: bool
    execute_cost: float
    reject_cost: float
    no_verify_cost: float
    verify_cost: float | None
    predicted_net_value: float | None
    decision_reason: str


@dataclass(frozen=True)
class EpisodeTrace:
    episode_id: str
    strategy: str
    steps: tuple[EpisodeStep, ...]
    termination_reason: str
    elapsed_s: float
    remaining_horizon_s: float
    replan_count: int
    collision: bool
    near_miss: bool
    false_safe_execute: bool
    verification_count: int
    reject_count: int
    success: bool
    hazard_object_type: str
    scientific_status: str = TOY_SCIENTIFIC_STATUS

    def as_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "strategy": self.strategy,
            "termination_reason": self.termination_reason,
            "elapsed_s": self.elapsed_s,
            "remaining_horizon_s": self.remaining_horizon_s,
            "replan_count": self.replan_count,
            "collision": self.collision,
            "near_miss": self.near_miss,
            "false_safe_execute": self.false_safe_execute,
            "verification_count": self.verification_count,
            "reject_count": self.reject_count,
            "success": self.success,
            "hazard_object_type": self.hazard_object_type,
            "scientific_status": self.scientific_status,
            "steps": [
                {
                    "index": step.index,
                    "decision": step.decision,
                    "action_id": step.action_id,
                    "calibrated_risk": step.calibrated_risk,
                    "elapsed_before_s": step.elapsed_before_s,
                    "elapsed_after_s": step.elapsed_after_s,
                    "remaining_before_s": step.remaining_before_s,
                    "remaining_after_s": step.remaining_after_s,
                    "replanned_after": step.replanned_after,
                    "execute_cost": step.execute_cost,
                    "reject_cost": step.reject_cost,
                    "no_verify_cost": step.no_verify_cost,
                    "verify_cost": step.verify_cost,
                    "predicted_net_value": step.predicted_net_value,
                    "decision_reason": step.decision_reason,
                }
                for step in self.steps
            ],
        }


def _step(
    *,
    index: int,
    policy_decision: DecisionResult,
    risk: float,
    elapsed_before: float,
    elapsed_after: float,
    horizon: float,
    replanned_after: bool,
) -> EpisodeStep:
    return EpisodeStep(
        index=index,
        decision=policy_decision.decision,
        action_id=policy_decision.action_id,
        calibrated_risk=risk,
        elapsed_before_s=elapsed_before,
        elapsed_after_s=elapsed_after,
        remaining_before_s=max(0.0, horizon - elapsed_before),
        remaining_after_s=max(0.0, horizon - elapsed_after),
        replanned_after=replanned_after,
        execute_cost=policy_decision.execute_cost,
        reject_cost=policy_decision.reject_cost,
        no_verify_cost=policy_decision.no_verify_cost,
        verify_cost=policy_decision.verify_cost,
        predicted_net_value=policy_decision.predicted_net_value,
        decision_reason=policy_decision.reason,
    )


def _can_replan(*, index: int, elapsed_s: float, config: ClosedLoopConfig) -> bool:
    return (
        index + 1 < config.max_decisions
        and config.future_horizon_s - elapsed_s + 1e-12 >= config.execute_step_s
    )


def run_toy_closed_loop(
    scenario: ToyScenario,
    *,
    strategy: str,
    config: ClosedLoopConfig,
) -> EpisodeTrace:
    """Run one short replan loop within the initial decision-time horizon."""

    if not isinstance(scenario, ToyScenario):
        raise TypeError("scenario must be a ToyScenario")
    if strategy not in DECISION_STRATEGIES:
        raise ValueError(f"unsupported decision strategy: {strategy!r}")
    if not isinstance(config, ClosedLoopConfig):
        raise TypeError("config must be a ClosedLoopConfig")

    elapsed = 0.0
    executed_time = 0.0
    verified = False
    steps: list[EpisodeStep] = []
    replan_count = 0
    verification_count = 0
    reject_count = 0
    collision = False
    near_miss = False
    false_safe_execute = False
    termination_reason = "max_decisions"

    for index in range(config.max_decisions):
        remaining = config.future_horizon_s - elapsed
        if remaining + 1e-12 < config.execute_step_s:
            termination_reason = "future_horizon_exhausted"
            break
        risk = (
            scenario.post_verify_calibrated_risk
            if verified
            else scenario.initial_calibrated_risk
        )
        values = {} if verified else scenario.action_values_for(strategy)
        decision = select_decision(
            DecisionRequest(
                strategy=strategy,
                task_cost=scenario.task_cost,
                calibrated_risk=risk,
                risk_weight=config.risk_weight,
                reject_cost=scenario.reject_cost,
                verify_margin=config.verify_margin,
                action_values=values,
            )
        )
        if decision.decision == "reject":
            steps.append(
                _step(
                    index=index,
                    policy_decision=decision,
                    risk=risk,
                    elapsed_before=elapsed,
                    elapsed_after=elapsed,
                    horizon=config.future_horizon_s,
                    replanned_after=False,
                )
            )
            reject_count += 1
            termination_reason = "rejected_no_candidate"
            break
        if decision.decision == "verify":
            if remaining + 1e-12 < config.verify_step_s:
                termination_reason = "future_horizon_exhausted_before_verify"
                break
            elapsed_before = elapsed
            elapsed += config.verify_step_s
            verified = True
            verification_count += 1
            replanned_after = _can_replan(index=index, elapsed_s=elapsed, config=config)
            replan_count += int(replanned_after)
            steps.append(
                _step(
                    index=index,
                    policy_decision=decision,
                    risk=risk,
                    elapsed_before=elapsed_before,
                    elapsed_after=elapsed,
                    horizon=config.future_horizon_s,
                    replanned_after=replanned_after,
                )
            )
            continue

        elapsed_before = elapsed
        elapsed += config.execute_step_s
        executed_time += config.execute_step_s
        if risk >= config.collision_risk_threshold:
            collision = True
            false_safe_execute = True
            termination_reason = "collision"
            steps.append(
                _step(
                    index=index,
                    policy_decision=decision,
                    risk=risk,
                    elapsed_before=elapsed_before,
                    elapsed_after=elapsed,
                    horizon=config.future_horizon_s,
                    replanned_after=False,
                )
            )
            break
        near_miss = near_miss or risk >= config.near_miss_risk_threshold
        success = executed_time + 1e-12 >= scenario.required_execute_time_s
        if success:
            termination_reason = "success"
        else:
            replanned_after = _can_replan(
                index=index,
                elapsed_s=elapsed,
                config=config,
            )
            replan_count += int(replanned_after)
        steps.append(
            _step(
                index=index,
                policy_decision=decision,
                risk=risk,
                elapsed_before=elapsed_before,
                elapsed_after=elapsed,
                horizon=config.future_horizon_s,
                replanned_after=not success and replanned_after,
            )
        )
        if success:
            break

    success = termination_reason == "success"
    return EpisodeTrace(
        episode_id=scenario.episode_id,
        strategy=strategy,
        steps=tuple(steps),
        termination_reason=termination_reason,
        elapsed_s=elapsed,
        remaining_horizon_s=max(0.0, config.future_horizon_s - elapsed),
        replan_count=replan_count,
        collision=collision,
        near_miss=near_miss,
        false_safe_execute=false_safe_execute,
        verification_count=verification_count,
        reject_count=reject_count,
        success=success,
        hazard_object_type=scenario.hazard_object_type,
    )


def summarize_toy_episodes(traces: Sequence[EpisodeTrace]) -> dict[str, float | int]:
    """Aggregate deterministic toy traces for the SOP-16 registry."""

    rows = tuple(traces)
    if not rows or any(not isinstance(trace, EpisodeTrace) for trace in rows):
        raise ValueError("traces must be a non-empty sequence of EpisodeTrace values")
    count = len(rows)
    successful = tuple(trace for trace in rows if trace.success)
    successful_count = len(successful)
    return {
        "episode_count": count,
        "collision_rate": sum(trace.collision for trace in rows) / count,
        "near_miss_rate": sum(trace.near_miss for trace in rows) / count,
        "false_safe_execution_rate": sum(trace.false_safe_execute for trace in rows)
        / count,
        "verification_count_mean": sum(trace.verification_count for trace in rows)
        / count,
        "reject_rate": sum(trace.reject_count > 0 for trace in rows) / count,
        "success_rate": sum(trace.success for trace in rows) / count,
        "terminal_time_mean_s": sum(trace.elapsed_s for trace in rows) / count,
        "successful_completion_count": successful_count,
        "successful_completion_time_mean_s": (
            sum(trace.elapsed_s for trace in successful) / successful_count
            if successful_count
            else 0.0
        ),
    }


__all__ = (
    "CLOSED_LOOP_VERSION",
    "TOY_SCIENTIFIC_STATUS",
    "ClosedLoopConfig",
    "EpisodeStep",
    "EpisodeTrace",
    "ToyScenario",
    "run_toy_closed_loop",
    "summarize_toy_episodes",
)
