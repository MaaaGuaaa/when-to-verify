"""Formal target history-visibility strata for SOP05 generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Real
from typing import Mapping

import numpy as np

from .sop05r_contracts import SOP05R_HISTORY_POLICY_VERSION


HISTORY_VISIBILITY_POLICY_VERSION = "target_history_visibility_policy_v1"
SEEN_THEN_OCCLUDED = "seen_then_occluded"
UNSEEN_IN_HISTORY_WINDOW = "unseen_in_history_window"
INELIGIBLE_HISTORY_VISIBILITY = "ineligible"
HISTORY_VISIBILITY_REGIMES = (
    SEEN_THEN_OCCLUDED,
    UNSEEN_IN_HISTORY_WINDOW,
)
_HISTORY_STEPS = 8
PREFIX4_VISIBLE_THEN_OCCLUDED = "prefix4_visible_then_occluded"
FALLBACK_SEEN_THEN_OCCLUDED = "fallback_seen_then_occluded"
INELIGIBLE_SEEN_THEN_OCCLUDED = "ineligible"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _finite_weight(value: object, *, regime: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"history visibility weight for {regime} must be real")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(
            f"history visibility weight for {regime} must be finite and nonnegative"
        )
    return result


@dataclass(frozen=True)
class HistoryVisibilityPolicy:
    """Normalized immutable policy for the two formal history strata."""

    min_trailing_hidden_frames: int
    seen_then_occluded_weight: float
    unseen_in_history_window_weight: float

    @property
    def weights(self) -> dict[str, float]:
        return {
            SEEN_THEN_OCCLUDED: self.seen_then_occluded_weight,
            UNSEEN_IN_HISTORY_WINDOW: self.unseen_in_history_window_weight,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_version": HISTORY_VISIBILITY_POLICY_VERSION,
            "min_trailing_hidden_frames": self.min_trailing_hidden_frames,
            "weights": self.weights,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.as_dict())).hexdigest()


@dataclass(frozen=True)
class HistoryVisibilityAssessment:
    """Classification evidence derived from one exact eight-frame vector."""

    regime: str
    last_visible_index: int | None
    trailing_hidden_frames: int


@dataclass(frozen=True)
class SeenThenOccludedHistoryAssessment:
    """One shared M5/M6 classification of a synchronized history window."""

    decision_index: int
    observed_class: str
    blocked_indices: tuple[int, ...]
    visible_frames: int
    occluded_frames: int
    preferred: bool
    eligible: bool


def _strict_history_vector(visibility: object) -> np.ndarray:
    if not isinstance(visibility, np.ndarray):
        raise TypeError("target history visibility must be a numpy array")
    if visibility.shape != (_HISTORY_STEPS,):
        raise ValueError("target history visibility must have shape (8,)")
    if visibility.dtype != np.bool_:
        raise TypeError("target history visibility must have boolean dtype")
    return visibility


def _trailing_hidden_frames(visibility: np.ndarray) -> int:
    trailing_hidden = 0
    for is_visible in visibility[::-1]:
        if bool(is_visible):
            break
        trailing_hidden += 1
    return trailing_hidden


def normalize_history_visibility_policy(
    value: object,
) -> HistoryVisibilityPolicy:
    """Validate and normalize the strict formal history-visibility policy."""

    expected = {
        "policy_version",
        "min_trailing_hidden_frames",
        "weights",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("history visibility policy keys mismatch")
    if value["policy_version"] != HISTORY_VISIBILITY_POLICY_VERSION:
        raise ValueError(
            "history visibility policy_version must equal "
            f"{HISTORY_VISIBILITY_POLICY_VERSION}"
        )
    trailing = value["min_trailing_hidden_frames"]
    if isinstance(trailing, (bool, np.bool_)) or not isinstance(
        trailing, (int, np.integer)
    ):
        raise TypeError("min_trailing_hidden_frames must be an integer")
    trailing = int(trailing)
    if not 1 <= trailing < _HISTORY_STEPS:
        raise ValueError(
            "min_trailing_hidden_frames must lie within [1, 7]"
        )
    raw_weights = value["weights"]
    if not isinstance(raw_weights, Mapping) or set(raw_weights) != set(
        HISTORY_VISIBILITY_REGIMES
    ):
        raise ValueError("history visibility weights keys mismatch")
    weights = {
        regime: _finite_weight(raw_weights[regime], regime=regime)
        for regime in HISTORY_VISIBILITY_REGIMES
    }
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("at least one history visibility weight must be positive")
    normalized = {regime: weights[regime] / total for regime in weights}
    return HistoryVisibilityPolicy(
        min_trailing_hidden_frames=trailing,
        seen_then_occluded_weight=normalized[SEEN_THEN_OCCLUDED],
        unseen_in_history_window_weight=normalized[UNSEEN_IN_HISTORY_WINDOW],
    )


def classify_history_visibility(
    visibility: object,
    policy: HistoryVisibilityPolicy,
) -> HistoryVisibilityAssessment:
    """Classify one canonical model-history visibility vector."""

    if not isinstance(policy, HistoryVisibilityPolicy):
        raise TypeError("history visibility policy has the wrong type")
    visibility = _strict_history_vector(visibility)
    visible_indices = np.flatnonzero(visibility)
    trailing_hidden = _trailing_hidden_frames(visibility)

    if visible_indices.size == 0:
        return HistoryVisibilityAssessment(
            regime=UNSEEN_IN_HISTORY_WINDOW,
            last_visible_index=None,
            trailing_hidden_frames=_HISTORY_STEPS,
        )

    last_visible = int(visible_indices[-1])
    regime = (
        SEEN_THEN_OCCLUDED
        if trailing_hidden >= policy.min_trailing_hidden_frames
        else INELIGIBLE_HISTORY_VISIBILITY
    )
    return HistoryVisibilityAssessment(
        regime=regime,
        last_visible_index=last_visible,
        trailing_hidden_frames=trailing_hidden,
    )


def classify_sop05r_history(visibility: object) -> HistoryVisibilityAssessment:
    """Classify the strict contiguous SOP05R history visibility regimes."""

    checked = _strict_history_vector(visibility)
    visible_indices = np.flatnonzero(checked)
    trailing_hidden = _trailing_hidden_frames(checked)
    if visible_indices.size == 0:
        return HistoryVisibilityAssessment(
            regime=UNSEEN_IN_HISTORY_WINDOW,
            last_visible_index=None,
            trailing_hidden_frames=_HISTORY_STEPS,
        )

    first_visible = int(visible_indices[0])
    last_visible = int(visible_indices[-1])
    visible_run_is_contiguous = bool(
        checked[first_visible : last_visible + 1].all()
    )
    qualifies = (
        first_visible <= 1
        and trailing_hidden >= 2
        and visible_run_is_contiguous
    )
    return HistoryVisibilityAssessment(
        regime=(
            SEEN_THEN_OCCLUDED
            if qualifies
            else INELIGIBLE_HISTORY_VISIBILITY
        ),
        last_visible_index=last_visible,
        trailing_hidden_frames=trailing_hidden,
    )


def classify_sop05r_seen_then_occluded_history(
    blocked: object,
    *,
    decision_index: int,
    minimum_visible_frames: int,
    minimum_occluded_frames: int = 1,
) -> SeenThenOccludedHistoryAssessment:
    """Classify one M5 history as preferred, fallback, or ineligible."""

    if not isinstance(blocked, np.ndarray):
        raise TypeError("centerline blocked sequence must be a numpy array")
    if blocked.ndim != 1 or blocked.dtype != np.bool_:
        raise ValueError("centerline blocked sequence must be a boolean vector")
    if (
        isinstance(decision_index, (bool, np.bool_))
        or not isinstance(decision_index, (int, np.integer))
    ):
        raise TypeError("decision_index must be an integer")
    decision_index = int(decision_index)
    if decision_index < _HISTORY_STEPS - 1 or decision_index >= blocked.size:
        raise ValueError("decision_index must have eight synchronized history frames")
    if (
        isinstance(minimum_visible_frames, (bool, np.bool_))
        or not isinstance(minimum_visible_frames, (int, np.integer))
    ):
        raise TypeError("minimum_visible_frames must be an integer")
    minimum_visible_frames = int(minimum_visible_frames)
    if not 1 <= minimum_visible_frames < _HISTORY_STEPS:
        raise ValueError("minimum_visible_frames must lie within [1, 7]")
    if (
        isinstance(minimum_occluded_frames, (bool, np.bool_))
        or not isinstance(minimum_occluded_frames, (int, np.integer))
    ):
        raise TypeError("minimum_occluded_frames must be an integer")
    minimum_occluded_frames = int(minimum_occluded_frames)
    if not 1 <= minimum_occluded_frames < _HISTORY_STEPS:
        raise ValueError("minimum_occluded_frames must lie within [1, 7]")

    history_start = decision_index - _HISTORY_STEPS + 1
    history = blocked[history_start : decision_index + 1]
    visible_frames = int(np.count_nonzero(~history))
    occluded_frames = int(np.count_nonzero(history))
    blocked_indices = tuple(
        history_start + int(index) for index in np.flatnonzero(history)
    )
    preferred = bool((~history[:4]).all() and history[4:].any())
    last_occluded_index = int(np.flatnonzero(history)[-1]) if occluded_frames else -1
    seen_before_occlusion = bool(
        last_occluded_index > 0 and np.any(~history[:last_occluded_index])
    )
    fallback = bool(
        visible_frames >= minimum_visible_frames
        and occluded_frames >= minimum_occluded_frames
        and seen_before_occlusion
    )
    observed_class = (
        PREFIX4_VISIBLE_THEN_OCCLUDED
        if preferred
        else (
            FALLBACK_SEEN_THEN_OCCLUDED
            if fallback
            else INELIGIBLE_SEEN_THEN_OCCLUDED
        )
    )
    return SeenThenOccludedHistoryAssessment(
        decision_index=decision_index,
        observed_class=observed_class,
        blocked_indices=blocked_indices,
        visible_frames=visible_frames,
        occluded_frames=occluded_frames,
        preferred=preferred,
        eligible=preferred or fallback,
    )


def validate_sop05r_history_metadata(
    visibility: object,
    metadata: object,
) -> HistoryVisibilityAssessment:
    """Recompute SOP05R history evidence and reject published metadata drift."""

    if not isinstance(metadata, Mapping):
        raise TypeError("SOP05R history metadata must be a mapping")
    assessment = classify_sop05r_history(visibility)
    expected = {
        "target_history_visibility_policy_version": (
            SOP05R_HISTORY_POLICY_VERSION
        ),
        "target_history_visibility_regime": assessment.regime,
        "target_history_last_visible_index": assessment.last_visible_index,
        "target_history_trailing_hidden_frames": (
            assessment.trailing_hidden_frames
        ),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("SOP05R history metadata does not match recomputed evidence")
    return assessment


def allocate_history_visibility_counts(
    total_count: object,
    policy: HistoryVisibilityPolicy,
    *,
    seed: object,
    namespace: object,
) -> dict[str, int]:
    """Allocate an integer request with deterministic stratified sampling."""

    if isinstance(total_count, (bool, np.bool_)) or not isinstance(
        total_count, (int, np.integer)
    ):
        raise TypeError("history visibility total_count must be an integer")
    total_count = int(total_count)
    if total_count <= 0:
        raise ValueError("history visibility total_count must be positive")
    if not isinstance(policy, HistoryVisibilityPolicy):
        raise TypeError("history visibility policy has the wrong type")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("history visibility seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("history visibility seed must be nonnegative")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("history visibility namespace must be a nonempty string")

    digest = hashlib.blake2b(
        _canonical_json_bytes(
            [
                HISTORY_VISIBILITY_POLICY_VERSION,
                seed,
                namespace,
                policy.digest,
            ]
        ),
        digest_size=8,
    ).digest()
    offset = int.from_bytes(digest, "big") / float(2**64)
    seen_threshold = policy.seen_then_occluded_weight
    counts = {regime: 0 for regime in HISTORY_VISIBILITY_REGIMES}
    for index in range(total_count):
        position = (float(index) + offset) / float(total_count)
        regime = (
            SEEN_THEN_OCCLUDED
            if position < seen_threshold
            else UNSEEN_IN_HISTORY_WINDOW
        )
        counts[regime] += 1
    return counts
