"""Generic Long40 SOP-15 execute/verify/reject closed-loop runtime.

The runtime owns decision sequencing, the single 6.4-second budget, trace
accounting, and policy invariants.  Geometry, model inference, replanning, and
oracle outcome scoring stay behind a small environment interface so the same
runner can consume deterministic fixtures, authenticated replay graphs, or a
future live simulator adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Protocol

from src.contracts import DYNAMIC_OBJECT_TYPES
from src.planning.decision_policy import (
    DECISION_STRATEGIES,
    DecisionRequest,
    DecisionResult,
    select_decision,
)


CLOSED_LOOP_RUNTIME_VERSION = "sop15_closed_loop_runtime_v1"


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return number


def _bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _numeric_mapping(value: object, *, name: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized: dict[str, float] = {}
    for key, raw in value.items():
        item = _string(key, name=f"{name} key")
        normalized[item] = _finite(raw, name=f"{name}[{item!r}]")
    return MappingProxyType(dict(sorted(normalized.items())))


def _strategy_values(
    value: object,
) -> Mapping[str, Mapping[str, float]]:
    if not isinstance(value, Mapping):
        raise TypeError("action_values_by_strategy must be a mapping")
    normalized: dict[str, Mapping[str, float]] = {}
    for strategy, raw_values in value.items():
        if strategy not in DECISION_STRATEGIES:
            raise ValueError(
                "action_values_by_strategy contains an unsupported strategy"
            )
        normalized[str(strategy)] = _numeric_mapping(
            raw_values,
            name=f"action_values_by_strategy[{strategy!r}]",
        )
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class ClosedLoopRuntimeConfig:
    """Frozen runtime timing and decision parameters."""

    future_horizon_s: float = 6.4
    execute_step_s: float = 0.2
    risk_weight: float = 1.0
    verify_margin: float = 0.01
    minimum_verify_duration_s: float = 0.5
    maximum_verify_duration_s: float = 1.0
    max_decisions: int = 32

    def __post_init__(self) -> None:
        horizon = _finite(
            self.future_horizon_s,
            name="future_horizon_s",
            minimum=1e-12,
        )
        if not math.isclose(horizon, 6.4, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("future_horizon_s must equal the Long40 6.4 s window")
        execute_step = _finite(
            self.execute_step_s,
            name="execute_step_s",
            minimum=0.2,
            maximum=0.5,
        )
        minimum_verify = _finite(
            self.minimum_verify_duration_s,
            name="minimum_verify_duration_s",
            minimum=1e-12,
        )
        maximum_verify = _finite(
            self.maximum_verify_duration_s,
            name="maximum_verify_duration_s",
            minimum=minimum_verify,
            maximum=horizon,
        )
        if isinstance(self.max_decisions, bool) or not isinstance(
            self.max_decisions, Integral
        ):
            raise TypeError("max_decisions must be an integer")
        if self.max_decisions <= 0:
            raise ValueError("max_decisions must be positive")
        object.__setattr__(self, "future_horizon_s", horizon)
        object.__setattr__(self, "execute_step_s", execute_step)
        object.__setattr__(
            self,
            "risk_weight",
            _finite(self.risk_weight, name="risk_weight", minimum=0.0),
        )
        object.__setattr__(
            self,
            "verify_margin",
            _finite(self.verify_margin, name="verify_margin", minimum=0.0),
        )
        object.__setattr__(self, "minimum_verify_duration_s", minimum_verify)
        object.__setattr__(self, "maximum_verify_duration_s", maximum_verify)
        object.__setattr__(self, "max_decisions", int(self.max_decisions))


@dataclass(frozen=True)
class DecisionFrame:
    """Deployment-available decision inputs for one environment state."""

    state_id: str
    plan_id: str
    task_cost: float
    calibrated_risk: float
    reject_cost: float
    action_values_by_strategy: Mapping[str, Mapping[str, Real]]
    action_durations_s: Mapping[str, Real]

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", _string(self.state_id, name="state_id"))
        object.__setattr__(self, "plan_id", _string(self.plan_id, name="plan_id"))
        object.__setattr__(
            self,
            "task_cost",
            _finite(self.task_cost, name="task_cost", minimum=0.0),
        )
        object.__setattr__(
            self,
            "calibrated_risk",
            _finite(
                self.calibrated_risk,
                name="calibrated_risk",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "reject_cost",
            _finite(self.reject_cost, name="reject_cost", minimum=0.0),
        )
        values = _strategy_values(self.action_values_by_strategy)
        durations = _numeric_mapping(
            self.action_durations_s,
            name="action_durations_s",
        )
        required_actions = {
            action_id
            for strategy_values in values.values()
            for action_id in strategy_values
        }
        if set(durations) != required_actions:
            raise ValueError(
                "action_durations_s must exactly cover all scored verification actions"
            )
        if any(duration <= 0.0 for duration in durations.values()):
            raise ValueError("verification action durations must be positive")
        object.__setattr__(self, "action_values_by_strategy", values)
        object.__setattr__(self, "action_durations_s", durations)

    def action_values_for(self, strategy: str) -> Mapping[str, float]:
        if strategy not in DECISION_STRATEGIES:
            raise ValueError(f"unsupported decision strategy: {strategy!r}")
        return self.action_values_by_strategy.get(strategy, MappingProxyType({}))


@dataclass(frozen=True)
class EnvironmentTransition:
    """One oracle-scored environment transition after execute or verify."""

    kind: str
    source_state_id: str
    source_plan_id: str
    duration_s: float
    path_length_m: float
    action_id: str | None = None
    next_state_id: str | None = None
    next_plan_id: str | None = None
    replanned: bool = False
    collision: bool = False
    near_miss: bool = False
    task_complete: bool = False
    critical_actor_revealed: bool = False
    verification_useful: bool | None = None
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"execute", "verify"}:
            raise ValueError("transition kind must be execute or verify")
        object.__setattr__(
            self,
            "source_state_id",
            _string(self.source_state_id, name="source_state_id"),
        )
        object.__setattr__(
            self,
            "source_plan_id",
            _string(self.source_plan_id, name="source_plan_id"),
        )
        object.__setattr__(
            self,
            "duration_s",
            _finite(self.duration_s, name="duration_s", minimum=1e-12),
        )
        object.__setattr__(
            self,
            "path_length_m",
            _finite(self.path_length_m, name="path_length_m", minimum=0.0),
        )
        for field in (
            "replanned",
            "collision",
            "near_miss",
            "task_complete",
            "critical_actor_revealed",
        ):
            object.__setattr__(
                self,
                field,
                _bool(getattr(self, field), name=field),
            )
        if self.kind == "verify":
            object.__setattr__(
                self,
                "action_id",
                _string(self.action_id, name="action_id"),
            )
            object.__setattr__(
                self,
                "verification_useful",
                _bool(self.verification_useful, name="verification_useful"),
            )
        elif self.action_id is not None or self.verification_useful is not None:
            raise ValueError(
                "execute transitions cannot carry verification action fields"
            )
        if self.next_state_id is not None:
            object.__setattr__(
                self,
                "next_state_id",
                _string(self.next_state_id, name="next_state_id"),
            )
        if self.next_plan_id is not None:
            object.__setattr__(
                self,
                "next_plan_id",
                _string(self.next_plan_id, name="next_plan_id"),
            )
        if self.termination_reason is not None:
            object.__setattr__(
                self,
                "termination_reason",
                _string(self.termination_reason, name="termination_reason"),
            )
        if self.collision and self.task_complete:
            raise ValueError("a transition cannot be both collision and task-complete")


class ClosedLoopEnvironment(Protocol):
    """Minimal adapter boundary consumed by :func:`run_closed_loop`."""

    episode_id: str
    initial_state_id: str
    hazard_object_type: str
    nominal_task_time_s: float
    nominal_path_length_m: float

    def decision_frame(self, state_id: str) -> DecisionFrame: ...

    def execute(
        self,
        state_id: str,
        duration_s: float,
    ) -> EnvironmentTransition: ...

    def verify(
        self,
        state_id: str,
        action_id: str,
    ) -> EnvironmentTransition: ...


@dataclass(frozen=True)
class ClosedLoopStep:
    index: int
    state_id: str
    plan_id: str
    decision: str
    action_id: str | None
    elapsed_before_s: float
    elapsed_after_s: float
    remaining_before_s: float
    remaining_after_s: float
    path_length_m: float
    next_state_id: str | None
    next_plan_id: str | None
    replanned_after: bool
    collision: bool
    near_miss: bool
    task_complete: bool
    critical_actor_revealed: bool
    verification_useful: bool | None
    execute_cost: float
    reject_cost: float
    no_verify_cost: float
    verify_cost: float | None
    predicted_net_value: float | None
    decision_reason: str
    transition_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class ClosedLoopTrace:
    episode_id: str
    strategy: str
    hazard_object_type: str
    steps: tuple[ClosedLoopStep, ...]
    termination_reason: str
    elapsed_s: float
    remaining_horizon_s: float
    path_length_m: float
    extra_time_s: float
    extra_path_length_m: float
    replan_count: int
    collision: bool
    near_miss: bool
    false_safe_execute: bool
    verification_count: int
    unnecessary_verification_count: int
    reject_count: int
    success: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "strategy": self.strategy,
            "hazard_object_type": self.hazard_object_type,
            "termination_reason": self.termination_reason,
            "elapsed_s": self.elapsed_s,
            "remaining_horizon_s": self.remaining_horizon_s,
            "path_length_m": self.path_length_m,
            "extra_time_s": self.extra_time_s,
            "extra_path_length_m": self.extra_path_length_m,
            "replan_count": self.replan_count,
            "collision": self.collision,
            "near_miss": self.near_miss,
            "false_safe_execute": self.false_safe_execute,
            "verification_count": self.verification_count,
            "unnecessary_verification_count": self.unnecessary_verification_count,
            "reject_count": self.reject_count,
            "success": self.success,
            "steps": [step.as_dict() for step in self.steps],
        }


def _validate_environment(environment: ClosedLoopEnvironment) -> tuple[str, str, float, float]:
    episode_id = _string(
        getattr(environment, "episode_id", None),
        name="environment.episode_id",
    )
    initial_state_id = _string(
        getattr(environment, "initial_state_id", None),
        name="environment.initial_state_id",
    )
    hazard_type = getattr(environment, "hazard_object_type", None)
    if hazard_type not in DYNAMIC_OBJECT_TYPES:
        raise ValueError("environment hazard_object_type is unsupported")
    nominal_time = _finite(
        getattr(environment, "nominal_task_time_s", None),
        name="environment.nominal_task_time_s",
        minimum=0.0,
    )
    nominal_path = _finite(
        getattr(environment, "nominal_path_length_m", None),
        name="environment.nominal_path_length_m",
        minimum=0.0,
    )
    return episode_id, initial_state_id, nominal_time, nominal_path


def _validate_transition(
    transition: object,
    *,
    frame: DecisionFrame,
    expected_kind: str,
    expected_duration_s: float,
    action_id: str | None,
) -> EnvironmentTransition:
    if not isinstance(transition, EnvironmentTransition):
        raise TypeError("environment transition must be an EnvironmentTransition")
    if transition.kind != expected_kind:
        raise ValueError("environment returned the wrong transition kind")
    if transition.source_state_id != frame.state_id:
        raise ValueError("transition source state does not match the decision frame")
    if transition.source_plan_id != frame.plan_id:
        raise ValueError("transition source plan does not match the decision frame")
    if transition.action_id != action_id:
        raise ValueError("transition action does not match the selected action")
    if not math.isclose(
        transition.duration_s,
        expected_duration_s,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("transition duration does not match the selected action")
    terminal = (
        transition.collision
        or transition.task_complete
        or transition.termination_reason is not None
    )
    if terminal:
        if transition.replanned:
            raise ValueError("terminal transitions cannot claim replanning")
        if transition.next_state_id is not None or transition.next_plan_id is not None:
            raise ValueError("terminal transitions cannot carry a next state or plan")
        return transition
    if not transition.replanned:
        raise ValueError("non-terminal execute/verify transitions must replan")
    if transition.next_state_id is None or transition.next_plan_id is None:
        raise ValueError("replanned transitions require a next state and plan")
    if transition.next_state_id == frame.state_id:
        raise ValueError("replanned transitions require a new state")
    if transition.next_plan_id == frame.plan_id:
        raise ValueError("replanned transitions require a new plan")
    return transition


def _step_from_transition(
    *,
    index: int,
    frame: DecisionFrame,
    decision: DecisionResult,
    transition: EnvironmentTransition | None,
    elapsed_before_s: float,
    elapsed_after_s: float,
    horizon_s: float,
) -> ClosedLoopStep:
    return ClosedLoopStep(
        index=index,
        state_id=frame.state_id,
        plan_id=frame.plan_id,
        decision=decision.decision,
        action_id=decision.action_id,
        elapsed_before_s=elapsed_before_s,
        elapsed_after_s=elapsed_after_s,
        remaining_before_s=max(0.0, horizon_s - elapsed_before_s),
        remaining_after_s=max(0.0, horizon_s - elapsed_after_s),
        path_length_m=0.0 if transition is None else transition.path_length_m,
        next_state_id=None if transition is None else transition.next_state_id,
        next_plan_id=None if transition is None else transition.next_plan_id,
        replanned_after=False if transition is None else transition.replanned,
        collision=False if transition is None else transition.collision,
        near_miss=False if transition is None else transition.near_miss,
        task_complete=False if transition is None else transition.task_complete,
        critical_actor_revealed=(
            False if transition is None else transition.critical_actor_revealed
        ),
        verification_useful=(
            None if transition is None else transition.verification_useful
        ),
        execute_cost=decision.execute_cost,
        reject_cost=decision.reject_cost,
        no_verify_cost=decision.no_verify_cost,
        verify_cost=decision.verify_cost,
        predicted_net_value=decision.predicted_net_value,
        decision_reason=decision.reason,
        transition_reason=(
            None if transition is None else transition.termination_reason
        ),
    )


def run_closed_loop(
    environment: ClosedLoopEnvironment,
    *,
    strategy: str,
    config: ClosedLoopRuntimeConfig,
) -> ClosedLoopTrace:
    """Run one bounded episode using environment-authoritative outcomes."""

    if strategy not in DECISION_STRATEGIES:
        raise ValueError(f"unsupported decision strategy: {strategy!r}")
    if not isinstance(config, ClosedLoopRuntimeConfig):
        raise TypeError("config must be a ClosedLoopRuntimeConfig")
    (
        episode_id,
        current_state_id,
        nominal_task_time_s,
        nominal_path_length_m,
    ) = _validate_environment(environment)
    hazard_object_type = str(environment.hazard_object_type)

    elapsed = 0.0
    path_length = 0.0
    replan_count = 0
    verification_count = 0
    unnecessary_verification_count = 0
    reject_count = 0
    collision = False
    near_miss = False
    false_safe_execute = False
    success = False
    steps: list[ClosedLoopStep] = []
    termination_reason = "max_decisions"
    expected_plan_id: str | None = None

    for index in range(config.max_decisions):
        remaining = config.future_horizon_s - elapsed
        if remaining + 1e-12 < config.execute_step_s:
            termination_reason = "future_horizon_exhausted"
            break
        frame = environment.decision_frame(current_state_id)
        if not isinstance(frame, DecisionFrame):
            raise TypeError("environment decision_frame must return DecisionFrame")
        if frame.state_id != current_state_id:
            raise ValueError("environment returned a decision frame for another state")
        if expected_plan_id is not None and frame.plan_id != expected_plan_id:
            raise ValueError(
                "decision frame plan does not match the transition's claimed next plan"
            )
        values = frame.action_values_for(strategy)
        decision = select_decision(
            DecisionRequest(
                strategy=strategy,
                task_cost=frame.task_cost,
                calibrated_risk=frame.calibrated_risk,
                risk_weight=config.risk_weight,
                reject_cost=frame.reject_cost,
                verify_margin=config.verify_margin,
                action_values=values,
            )
        )
        if decision.decision == "reject":
            steps.append(
                _step_from_transition(
                    index=index,
                    frame=frame,
                    decision=decision,
                    transition=None,
                    elapsed_before_s=elapsed,
                    elapsed_after_s=elapsed,
                    horizon_s=config.future_horizon_s,
                )
            )
            reject_count += 1
            termination_reason = "rejected_by_policy"
            break

        if decision.decision == "verify":
            if decision.action_id is None:
                raise RuntimeError("verify decision is missing an action")
            duration = frame.action_durations_s[decision.action_id]
            if not (
                config.minimum_verify_duration_s
                <= duration
                <= config.maximum_verify_duration_s
            ):
                raise ValueError("verification action duration is outside runtime bounds")
            if duration > remaining + 1e-12:
                termination_reason = "future_horizon_exhausted_before_verify"
                break
            transition = _validate_transition(
                environment.verify(current_state_id, decision.action_id),
                frame=frame,
                expected_kind="verify",
                expected_duration_s=duration,
                action_id=decision.action_id,
            )
            verification_count += 1
            unnecessary_verification_count += int(
                transition.verification_useful is False
            )
        else:
            duration = config.execute_step_s
            transition = _validate_transition(
                environment.execute(current_state_id, duration),
                frame=frame,
                expected_kind="execute",
                expected_duration_s=duration,
                action_id=None,
            )

        elapsed_before = elapsed
        elapsed += transition.duration_s
        if elapsed > config.future_horizon_s + 1e-9:
            raise ValueError("environment transition exceeds the Long40 future budget")
        path_length += transition.path_length_m
        collision = collision or transition.collision
        near_miss = near_miss or transition.near_miss
        false_safe_execute = false_safe_execute or (
            decision.decision == "execute" and transition.collision
        )
        replan_count += int(transition.replanned)
        steps.append(
            _step_from_transition(
                index=index,
                frame=frame,
                decision=decision,
                transition=transition,
                elapsed_before_s=elapsed_before,
                elapsed_after_s=elapsed,
                horizon_s=config.future_horizon_s,
            )
        )

        if transition.collision:
            termination_reason = transition.termination_reason or "collision"
            break
        if transition.task_complete:
            success = True
            termination_reason = transition.termination_reason or "goal_reached"
            break
        if transition.termination_reason is not None:
            termination_reason = transition.termination_reason
            break
        if transition.next_state_id is None:
            raise RuntimeError("validated non-terminal transition lost its next state")
        current_state_id = transition.next_state_id
        expected_plan_id = transition.next_plan_id

    return ClosedLoopTrace(
        episode_id=episode_id,
        strategy=strategy,
        hazard_object_type=hazard_object_type,
        steps=tuple(steps),
        termination_reason=termination_reason,
        elapsed_s=elapsed,
        remaining_horizon_s=max(0.0, config.future_horizon_s - elapsed),
        path_length_m=path_length,
        extra_time_s=max(0.0, elapsed - nominal_task_time_s),
        extra_path_length_m=max(0.0, path_length - nominal_path_length_m),
        replan_count=replan_count,
        collision=collision,
        near_miss=near_miss,
        false_safe_execute=false_safe_execute,
        verification_count=verification_count,
        unnecessary_verification_count=unnecessary_verification_count,
        reject_count=reject_count,
        success=success,
    )


def summarize_closed_loop_episodes(
    traces: Sequence[ClosedLoopTrace],
) -> dict[str, float | int]:
    """Aggregate the complete frozen SOP-15 metric set."""

    rows = tuple(traces)
    if not rows or any(not isinstance(trace, ClosedLoopTrace) for trace in rows):
        raise ValueError(
            "traces must be a non-empty sequence of ClosedLoopTrace values"
        )
    count = len(rows)
    total_verifications = sum(trace.verification_count for trace in rows)
    successful = tuple(trace for trace in rows if trace.success)
    return {
        "episode_count": count,
        "collision_rate": math.fsum(trace.collision for trace in rows) / count,
        "near_miss_rate": math.fsum(trace.near_miss for trace in rows) / count,
        "false_safe_execution_rate": (
            math.fsum(trace.false_safe_execute for trace in rows) / count
        ),
        "verification_count_mean": (
            math.fsum(trace.verification_count for trace in rows) / count
        ),
        "verification_episode_rate": (
            math.fsum(trace.verification_count > 0 for trace in rows) / count
        ),
        "unnecessary_verification_rate": (
            math.fsum(trace.unnecessary_verification_count for trace in rows)
            / total_verifications
            if total_verifications
            else 0.0
        ),
        "reject_rate": math.fsum(trace.reject_count > 0 for trace in rows) / count,
        "success_rate": math.fsum(trace.success for trace in rows) / count,
        "terminal_time_mean_s": math.fsum(trace.elapsed_s for trace in rows) / count,
        "successful_completion_count": len(successful),
        "successful_completion_time_mean_s": (
            math.fsum(trace.elapsed_s for trace in successful) / len(successful)
            if successful
            else 0.0
        ),
        "path_length_mean_m": math.fsum(trace.path_length_m for trace in rows)
        / count,
        "extra_path_length_mean_m": (
            math.fsum(trace.extra_path_length_m for trace in rows) / count
        ),
        "extra_time_mean_s": math.fsum(trace.extra_time_s for trace in rows)
        / count,
        "replan_count_mean": math.fsum(trace.replan_count for trace in rows) / count,
        "decision_count_mean": math.fsum(len(trace.steps) for trace in rows) / count,
        "failure_count": sum(not trace.success for trace in rows),
    }


__all__ = (
    "CLOSED_LOOP_RUNTIME_VERSION",
    "ClosedLoopEnvironment",
    "ClosedLoopRuntimeConfig",
    "ClosedLoopStep",
    "ClosedLoopTrace",
    "DecisionFrame",
    "EnvironmentTransition",
    "run_closed_loop",
    "summarize_closed_loop_episodes",
)
