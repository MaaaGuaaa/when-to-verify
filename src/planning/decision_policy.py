"""SOP-15 execute/verify/reject decision policy.

Verification action values are *net* decision values: their action cost was
already accounted for while producing the value.  The policy therefore never
adds an action cost after subtracting a selected value from the no-verify cost.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping


DECISION_POLICY_VERSION = "sop15_decision_policy_v1"
DECISION_STRATEGIES = (
    "never",
    "always",
    "visible",
    "swept",
    "entropy",
    "learned",
    "oracle",
)
_VALUE_DRIVEN_STRATEGIES = frozenset(DECISION_STRATEGIES) - {"never", "always"}


def _finite_real(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _action_values(value: Mapping[str, Real]) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("action_values must be a mapping")
    normalized: dict[str, float] = {}
    for action_id, score in value.items():
        if not isinstance(action_id, str) or not action_id:
            raise ValueError("action_values keys must be non-empty strings")
        normalized[action_id] = _finite_real(
            score, name=f"action_values[{action_id!r}]"
        )
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class DecisionRequest:
    """Inputs available at one closed-loop decision point."""

    strategy: str
    task_cost: float
    calibrated_risk: float
    risk_weight: float
    reject_cost: float
    verify_margin: float
    action_values: Mapping[str, Real]

    def __post_init__(self) -> None:
        if self.strategy not in DECISION_STRATEGIES:
            raise ValueError(f"unsupported decision strategy: {self.strategy!r}")
        object.__setattr__(self, "task_cost", _finite_real(self.task_cost, name="task_cost", minimum=0.0))
        object.__setattr__(
            self,
            "calibrated_risk",
            _finite_real(self.calibrated_risk, name="calibrated_risk", minimum=0.0),
        )
        object.__setattr__(
            self,
            "risk_weight",
            _finite_real(self.risk_weight, name="risk_weight", minimum=0.0),
        )
        object.__setattr__(
            self,
            "reject_cost",
            _finite_real(self.reject_cost, name="reject_cost", minimum=0.0),
        )
        object.__setattr__(
            self,
            "verify_margin",
            _finite_real(self.verify_margin, name="verify_margin", minimum=0.0),
        )
        object.__setattr__(self, "action_values", _action_values(self.action_values))


@dataclass(frozen=True)
class DecisionResult:
    """One auditable policy decision and all costs used to select it."""

    decision: str
    action_id: str | None
    execute_cost: float
    reject_cost: float
    no_verify_cost: float
    verify_cost: float | None
    predicted_net_value: float | None
    reason: str

    def __post_init__(self) -> None:
        if self.decision not in {"execute", "verify", "reject"}:
            raise ValueError("decision must be execute, verify, or reject")
        if self.decision == "verify":
            if self.action_id is None or self.verify_cost is None:
                raise ValueError("verify decisions require an action and verify cost")
        elif self.action_id is not None:
            raise ValueError("only verify decisions may include an action")


def _no_verify_decision(request: DecisionRequest, execute_cost: float) -> tuple[str, str]:
    if execute_cost <= request.reject_cost:
        return "execute", "execute_cost_not_greater_than_reject"
    return "reject", "reject_cost_lower_than_execute"


def _best_action(action_values: Mapping[str, float]) -> tuple[str, float] | None:
    if not action_values:
        return None
    action_id = min(action_values, key=lambda key: (-action_values[key], key))
    return action_id, action_values[action_id]


def select_decision(request: DecisionRequest) -> DecisionResult:
    """Select execute, verify, or reject without double-charging action cost."""

    if not isinstance(request, DecisionRequest):
        raise TypeError("request must be a DecisionRequest")
    execute_cost = request.task_cost + request.risk_weight * request.calibrated_risk
    no_verify_cost = min(execute_cost, request.reject_cost)
    fallback_decision, fallback_reason = _no_verify_decision(request, execute_cost)

    if request.strategy == "never":
        return DecisionResult(
            decision=fallback_decision,
            action_id=None,
            execute_cost=execute_cost,
            reject_cost=request.reject_cost,
            no_verify_cost=no_verify_cost,
            verify_cost=None,
            predicted_net_value=None,
            reason="never_verify:" + fallback_reason,
        )

    selected = _best_action(request.action_values)
    if selected is None:
        return DecisionResult(
            decision=fallback_decision,
            action_id=None,
            execute_cost=execute_cost,
            reject_cost=request.reject_cost,
            no_verify_cost=no_verify_cost,
            verify_cost=None,
            predicted_net_value=None,
            reason="no_verification_actions:" + fallback_reason,
        )
    action_id, net_value = selected
    verify_cost = no_verify_cost - net_value

    if request.strategy == "always":
        return DecisionResult(
            decision="verify",
            action_id=action_id,
            execute_cost=execute_cost,
            reject_cost=request.reject_cost,
            no_verify_cost=no_verify_cost,
            verify_cost=verify_cost,
            predicted_net_value=net_value,
            reason="always_verify",
        )

    if request.strategy not in _VALUE_DRIVEN_STRATEGIES:
        raise RuntimeError("decision strategy classification is incomplete")
    if verify_cost + request.verify_margin < no_verify_cost:
        return DecisionResult(
            decision="verify",
            action_id=action_id,
            execute_cost=execute_cost,
            reject_cost=request.reject_cost,
            no_verify_cost=no_verify_cost,
            verify_cost=verify_cost,
            predicted_net_value=net_value,
            reason="positive_net_value_exceeds_margin",
        )
    return DecisionResult(
        decision=fallback_decision,
        action_id=None,
        execute_cost=execute_cost,
        reject_cost=request.reject_cost,
        no_verify_cost=no_verify_cost,
        verify_cost=verify_cost,
        predicted_net_value=net_value,
        reason="verification_margin_not_met:" + fallback_reason,
    )


__all__ = (
    "DECISION_POLICY_VERSION",
    "DECISION_STRATEGIES",
    "DecisionRequest",
    "DecisionResult",
    "select_decision",
)
