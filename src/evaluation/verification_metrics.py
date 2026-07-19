"""Dependency-free SOP14 metrics and strict checkpoint provenance."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from src.contracts import (
    HISTORY_CHANNELS,
    SCHEMA_VERSION,
    STATE_CHANNELS,
    TRAJECTORY_CHANNELS,
)
from src.datasets.verification_dataloader import VERIFICATION_SHARD_LAYOUT_VERSION
from src.datasets.verification_dataset import VERIFICATION_DATASET_VERSION
from src.models.verification_model import VERIFICATION_MODEL_VERSION
from src.planning.verification_actions import CANONICAL_ACTION_IDS


VERIFICATION_CHECKPOINT_MANIFEST_VERSION = "verification_checkpoint_manifest_v2"
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = frozenset(
    {
        "manifest_version",
        "schema_version",
        "model_version",
        "verification_dataset_version",
        "verification_shard_layout_version",
        "history_channels",
        "state_channels",
        "trajectory_channels",
        "action_order",
        "input_manifest_digest",
        "split_digests",
        "model_config",
        "model_config_digest",
        "seed",
        "code_version",
    }
)
_SPLITS = frozenset({"train", "calibration", "val", "test"})


def _numeric_vector(value: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _identities(value: object, *, name: str, size: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(value)
    if len(result) != size:
        raise ValueError(f"{name} must align with metric vectors")
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain non-empty strings")
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_correlation(left: object, right: object) -> float:
    """Spearman rho with average ranks; constant input deterministically gives 0."""

    x = _numeric_vector(left, name="left")
    y = _numeric_vector(right, name="right")
    if x.shape != y.shape:
        raise ValueError("rank vectors must align")
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    x_centered = x_rank - np.mean(x_rank)
    y_centered = y_rank - np.mean(y_rank)
    denominator = float(
        np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denominator)


def kendall_tau_b(left: object, right: object) -> float:
    """Kendall tau-b; all-tie or singleton-equivalent inputs return 0."""

    x = _numeric_vector(left, name="left")
    y = _numeric_vector(right, name="right")
    if x.shape != y.shape:
        raise ValueError("rank vectors must align")
    concordant = discordant = tied_x = tied_y = 0
    for first in range(x.size):
        for second in range(first + 1, x.size):
            dx = np.sign(x[first] - x[second])
            dy = np.sign(y[first] - y[second])
            if dx == 0.0 and dy == 0.0:
                continue
            if dx == 0.0:
                tied_x += 1
            elif dy == 0.0:
                tied_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = np.sqrt(
        (concordant + discordant + tied_x)
        * (concordant + discordant + tied_y)
    )
    if denominator == 0.0:
        return 0.0
    return float((concordant - discordant) / denominator)


def _grouped_indices(group_ids: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    groups: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    return {key: tuple(value) for key, value in sorted(groups.items())}


def pairwise_ranking_accuracy(
    value_prediction: object,
    value_target: object,
    *,
    group_ids: Sequence[str],
    action_ids: Sequence[str],
) -> tuple[float, int]:
    """Group-local accuracy; prediction ties receive half credit."""

    prediction = _numeric_vector(value_prediction, name="value_prediction")
    target = _numeric_vector(value_target, name="value_target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must align")
    groups = _identities(group_ids, name="group_ids", size=target.size)
    actions = _identities(action_ids, name="action_ids", size=target.size)
    credits: list[float] = []
    for indices in _grouped_indices(groups).values():
        seen_actions: set[str] = set()
        for index in indices:
            if actions[index] in seen_actions:
                raise ValueError("ranking group contains duplicate action IDs")
            seen_actions.add(actions[index])
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1 :]:
                target_difference = target[left] - target[right]
                if target_difference == 0.0:
                    continue
                predicted_difference = prediction[left] - prediction[right]
                if predicted_difference == 0.0:
                    credits.append(0.5)
                else:
                    credits.append(
                        float(np.sign(predicted_difference) == np.sign(target_difference))
                    )
    if not credits:
        return 0.0, 0
    return float(np.mean(credits, dtype=np.float64)), len(credits)


def _useful_f1(probability: np.ndarray, target: np.ndarray) -> float:
    predicted = probability >= 0.5
    truth = target != 0.0
    true_positive = int(np.count_nonzero(predicted & truth))
    false_positive = int(np.count_nonzero(predicted & ~truth))
    false_negative = int(np.count_nonzero(~predicted & truth))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else float(2 * true_positive / denominator)


def _huber(prediction: np.ndarray, target: np.ndarray, delta: float) -> float:
    error = np.abs(prediction - target)
    losses = np.where(
        error <= delta,
        0.5 * error**2,
        delta * (error - 0.5 * delta),
    )
    return float(np.mean(losses, dtype=np.float64))


def _flat_metrics(
    prediction: np.ndarray,
    probability: np.ndarray,
    target: np.ndarray,
    useful: np.ndarray,
    *,
    huber_delta: float,
) -> dict[str, object]:
    return {
        "sample_count": int(target.size),
        "useful_f1": _useful_f1(probability, useful),
        "value_mse": float(np.mean((prediction - target) ** 2, dtype=np.float64)),
        "value_huber": _huber(prediction, target, huber_delta),
        "spearman": spearman_correlation(prediction, target),
        "kendall_tau_b": kendall_tau_b(prediction, target),
    }


def _action_key(action_id: str) -> tuple[int, str]:
    try:
        return CANONICAL_ACTION_IDS.index(action_id), action_id
    except ValueError:
        return len(CANONICAL_ACTION_IDS), action_id


def evaluate_verification_predictions(
    *,
    value_prediction: object,
    useful_probability: object,
    value_target: object,
    useful_target: object,
    group_ids: Sequence[str],
    action_ids: Sequence[str],
    huber_delta: float,
    slice_fields: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Compute dependency-free sample and group ranking metrics.

    Prediction ties choose the first canonical action. Oracle top-two membership
    uses target descending then the same canonical action order.
    """

    prediction = _numeric_vector(value_prediction, name="value_prediction")
    probability = _numeric_vector(useful_probability, name="useful_probability")
    target = _numeric_vector(value_target, name="value_target")
    useful = _numeric_vector(useful_target, name="useful_target")
    if not (
        prediction.shape == probability.shape == target.shape == useful.shape
    ):
        raise ValueError("verification metric vectors must align")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("useful_probability must lie in [0,1]")
    if not np.isin(useful, (0.0, 1.0)).all():
        raise ValueError("useful_target must be binary")
    if isinstance(huber_delta, bool) or not isinstance(huber_delta, Real):
        raise TypeError("huber_delta must be a real number")
    delta = float(huber_delta)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("huber_delta must be finite and positive")
    groups = _identities(group_ids, name="group_ids", size=target.size)
    actions = _identities(action_ids, name="action_ids", size=target.size)
    ranking_accuracy, pair_count = pairwise_ranking_accuracy(
        prediction,
        target,
        group_ids=groups,
        action_ids=actions,
    )
    grouped = _grouped_indices(groups)
    action_order = tuple(sorted(set(actions), key=_action_key))
    selected_counts = {action_id: 0 for action_id in action_order}
    regrets: list[float] = []
    top_two: list[float] = []
    for indices in grouped.values():
        selected = min(
            indices,
            key=lambda index: (-prediction[index], _action_key(actions[index])),
        )
        selected_counts[actions[selected]] += 1
        oracle_value = max(target[index] for index in indices)
        regrets.append(float(oracle_value - target[selected]))
        oracle_order = sorted(
            indices,
            key=lambda index: (-target[index], _action_key(actions[index])),
        )
        top_two.append(float(selected in set(oracle_order[:2])))
    report = _flat_metrics(
        prediction, probability, target, useful, huber_delta=delta
    )
    report.update(
        {
            "group_count": len(grouped),
            "pairwise_accuracy": ranking_accuracy,
            "pair_count": pair_count,
            "top1_regret_mean": float(np.mean(regrets, dtype=np.float64)),
            "top_two_selection_rate": float(np.mean(top_two, dtype=np.float64)),
            "selected_action_counts": selected_counts,
            "selected_action_proportions": {
                key: float(value / len(grouped))
                for key, value in selected_counts.items()
            },
        }
    )
    slices: dict[str, dict[str, dict[str, object]]] = {}
    if slice_fields is not None:
        if not isinstance(slice_fields, Mapping):
            raise TypeError("slice_fields must be a mapping")
        for field, raw_values in sorted(slice_fields.items()):
            if not isinstance(field, str) or not field:
                raise ValueError("slice field names must be non-empty strings")
            values = _identities(
                raw_values, name=f"slice_fields[{field}]", size=target.size
            )
            field_report: dict[str, dict[str, object]] = {}
            for value in sorted(set(values)):
                mask = np.asarray([item == value for item in values], dtype=bool)
                field_report[value] = _flat_metrics(
                    prediction[mask],
                    probability[mask],
                    target[mask],
                    useful[mask],
                    huber_delta=delta,
                )
            slices[field] = field_report
    report["slices"] = slices
    return report


def _canonical_copy(value: object, *, name: str) -> object:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be finite canonical JSON") from exc


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _split_digest_map(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value or "train" not in value:
        raise ValueError("split_digests must be a non-empty mapping containing train")
    if not set(value).issubset(_SPLITS):
        raise ValueError("split_digests contains an invalid split")
    return {
        key: _digest(value[key], name=f"split_digests[{key}]")
        for key in sorted(value)
    }


def build_verification_checkpoint_manifest(
    *,
    input_manifest_digest: str,
    split_digests: Mapping[str, str],
    model_config: Mapping[str, object],
    seed: int,
    code_version: str,
) -> dict[str, object]:
    """Build deterministic v2 provenance for a schema-3 verification model."""

    input_digest = _digest(
        input_manifest_digest, name="input_manifest_digest"
    )
    splits = _split_digest_map(split_digests)
    config = _canonical_copy(model_config, name="model_config")
    if not isinstance(config, dict) or not config:
        raise ValueError("model_config must be a non-empty mapping")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    if not isinstance(code_version, str) or not code_version:
        raise ValueError("code_version must be a non-empty string")
    return {
        "manifest_version": VERIFICATION_CHECKPOINT_MANIFEST_VERSION,
        "schema_version": SCHEMA_VERSION,
        "model_version": VERIFICATION_MODEL_VERSION,
        "verification_dataset_version": VERIFICATION_DATASET_VERSION,
        "verification_shard_layout_version": VERIFICATION_SHARD_LAYOUT_VERSION,
        "history_channels": list(HISTORY_CHANNELS),
        "state_channels": list(STATE_CHANNELS),
        "trajectory_channels": list(TRAJECTORY_CHANNELS),
        "action_order": list(CANONICAL_ACTION_IDS),
        "input_manifest_digest": input_digest,
        "split_digests": splits,
        "model_config": config,
        "model_config_digest": _sha256_json(config),
        "seed": int(seed),
        "code_version": code_version,
    }


def validate_verification_checkpoint_manifest(
    manifest: object,
    *,
    expected_input_manifest_digest: str,
    expected_split_digests: Mapping[str, str],
    expected_model_config: Mapping[str, object],
    expected_seed: int,
    expected_code_version: str,
) -> dict[str, object]:
    """Reject legacy, incomplete, or context-mismatched checkpoint manifests."""

    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("checkpoint manifest keys are invalid")
    if manifest["manifest_version"] != VERIFICATION_CHECKPOINT_MANIFEST_VERSION:
        if manifest["manifest_version"] == "verification_checkpoint_manifest_v1":
            raise ValueError("legacy v1 verification checkpoint is forbidden")
        raise ValueError("unsupported verification checkpoint manifest version")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    if manifest["model_version"] != VERIFICATION_MODEL_VERSION:
        raise ValueError("checkpoint model version mismatch")
    if manifest["verification_dataset_version"] != VERIFICATION_DATASET_VERSION:
        raise ValueError("checkpoint verification dataset version mismatch")
    if (
        manifest["verification_shard_layout_version"]
        != VERIFICATION_SHARD_LAYOUT_VERSION
    ):
        raise ValueError("checkpoint verification shard layout mismatch")
    if manifest["history_channels"] != list(HISTORY_CHANNELS) or manifest[
        "state_channels"
    ] != list(STATE_CHANNELS) or manifest["trajectory_channels"] != list(
        TRAJECTORY_CHANNELS
    ):
        raise ValueError("checkpoint channel order mismatch")
    if manifest["action_order"] != list(CANONICAL_ACTION_IDS):
        raise ValueError("checkpoint action order mismatch")
    actual_input = _digest(
        manifest["input_manifest_digest"], name="input_manifest_digest"
    )
    expected_input = _digest(
        expected_input_manifest_digest, name="expected_input_manifest_digest"
    )
    if actual_input != expected_input:
        raise ValueError("checkpoint input manifest digest mismatch")
    actual_splits = _split_digest_map(manifest["split_digests"])
    expected_splits = _split_digest_map(expected_split_digests)
    if actual_splits != expected_splits:
        raise ValueError("checkpoint split digests mismatch")
    actual_config = _canonical_copy(manifest["model_config"], name="model_config")
    expected_config = _canonical_copy(
        expected_model_config, name="expected_model_config"
    )
    if actual_config != expected_config:
        raise ValueError("checkpoint model config mismatch")
    if manifest["model_config_digest"] != _sha256_json(actual_config):
        raise ValueError("checkpoint model config digest mismatch")
    if (
        isinstance(manifest["seed"], bool)
        or not isinstance(manifest["seed"], int)
        or manifest["seed"] != expected_seed
    ):
        raise ValueError("checkpoint seed mismatch")
    if (
        not isinstance(manifest["code_version"], str)
        or not manifest["code_version"]
        or manifest["code_version"] != expected_code_version
    ):
        raise ValueError("checkpoint code version mismatch")
    return dict(_canonical_copy(manifest, name="checkpoint manifest"))


__all__ = (
    "VERIFICATION_CHECKPOINT_MANIFEST_VERSION",
    "build_verification_checkpoint_manifest",
    "evaluate_verification_predictions",
    "kendall_tau_b",
    "pairwise_ranking_accuracy",
    "spearman_correlation",
    "validate_verification_checkpoint_manifest",
)
