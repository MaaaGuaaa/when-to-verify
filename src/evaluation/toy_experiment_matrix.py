"""Runnable toy adapters for the SOP-15/16 closed-loop framework."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import shutil
import tempfile
from typing import Any

import yaml

from src.contracts import DYNAMIC_OBJECT_TYPES, SCHEMA_VERSION
from src.evaluation.closed_loop import (
    CLOSED_LOOP_VERSION,
    TOY_SCIENTIFIC_STATUS,
    ClosedLoopConfig,
    EpisodeTrace,
    ToyScenario,
    run_toy_closed_loop,
    summarize_toy_episodes,
)
from src.evaluation.result_registry import (
    RegisteredResult,
    ResultProvenance,
    aggregate_seed_metrics,
    load_result,
    publish_result,
)
from src.planning.decision_policy import DECISION_STRATEGIES
from src.utils.atomic_publish import atomic_rename_noreplace


TOY_MATRIX_VERSION = "sop16_toy_matrix_v1"
_CLOSED_LOOP_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "closed_loop_version",
        "future_horizon_s",
        "execute_step_s",
        "verify_step_s",
        "risk_weight",
        "verify_margin",
        "collision_risk_threshold",
        "near_miss_risk_threshold",
        "max_decisions",
    }
)
_MATRIX_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "matrix_version",
        "closed_loop_config",
        "seeds",
        "strategies",
        "episode_count",
    }
)


def _json_bytes(value: object) -> bytes:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("toy experiment value must be finite JSON") from exc
    return (payload + "\n").encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _yaml_mapping(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a mapping")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def closed_loop_config_payload(config: ClosedLoopConfig) -> dict[str, object]:
    if not isinstance(config, ClosedLoopConfig):
        raise TypeError("config must be a ClosedLoopConfig")
    return {
        "schema_version": SCHEMA_VERSION,
        "closed_loop_version": CLOSED_LOOP_VERSION,
        "future_horizon_s": config.future_horizon_s,
        "execute_step_s": config.execute_step_s,
        "verify_step_s": config.verify_step_s,
        "risk_weight": config.risk_weight,
        "verify_margin": config.verify_margin,
        "collision_risk_threshold": config.collision_risk_threshold,
        "near_miss_risk_threshold": config.near_miss_risk_threshold,
        "max_decisions": config.max_decisions,
    }


def load_closed_loop_config(path: str | Path) -> ClosedLoopConfig:
    """Load the small standalone SOP-15 toy configuration strictly."""

    source = Path(path)
    raw = _yaml_mapping(source, label="closed-loop config")
    if set(raw) != _CLOSED_LOOP_CONFIG_KEYS:
        raise ValueError("closed-loop config keys are invalid")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"closed-loop config schema must be {SCHEMA_VERSION}")
    if raw["closed_loop_version"] != CLOSED_LOOP_VERSION:
        raise ValueError("unsupported closed-loop config version")
    return ClosedLoopConfig(
        future_horizon_s=raw["future_horizon_s"],
        execute_step_s=raw["execute_step_s"],
        verify_step_s=raw["verify_step_s"],
        risk_weight=raw["risk_weight"],
        verify_margin=raw["verify_margin"],
        collision_risk_threshold=raw["collision_risk_threshold"],
        near_miss_risk_threshold=raw["near_miss_risk_threshold"],
        max_decisions=raw["max_decisions"],
    )


@dataclass(frozen=True)
class ToyMatrixConfig:
    config_path: Path
    closed_loop_config_path: Path
    seeds: tuple[int, ...]
    strategies: tuple[str, ...]
    episode_count: int


@dataclass(frozen=True)
class ToyMatrixResult:
    output_dir: Path
    run_paths: tuple[Path, ...]
    summary: dict[str, object]


def _read_json_mapping(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a real file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def load_toy_matrix_config(path: str | Path) -> ToyMatrixConfig:
    """Load a 1--3 seed, finite strategy toy-matrix configuration."""

    source = Path(path)
    raw = _yaml_mapping(source, label="toy matrix config")
    if set(raw) != _MATRIX_CONFIG_KEYS:
        raise ValueError("toy matrix config keys are invalid")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"toy matrix schema must be {SCHEMA_VERSION}")
    if raw["matrix_version"] != TOY_MATRIX_VERSION:
        raise ValueError("unsupported toy matrix version")
    raw_config = raw["closed_loop_config"]
    if not isinstance(raw_config, str) or not raw_config:
        raise ValueError("closed_loop_config must be a non-empty relative path")
    candidate = Path(raw_config)
    if candidate.is_absolute():
        raise ValueError("closed_loop_config must be a relative path")
    candidate_path = source.parent / candidate
    if not candidate_path.is_file() or candidate_path.is_symlink():
        raise ValueError("closed_loop_config must resolve to a real file")
    closed_loop_path = candidate_path.resolve()
    raw_seeds = raw["seeds"]
    if isinstance(raw_seeds, (str, bytes)) or not isinstance(raw_seeds, Sequence):
        raise ValueError("seeds must be a sequence")
    seeds = tuple(_positive_int(seed, name="seed") for seed in raw_seeds)
    if not 1 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds):
        raise ValueError("toy matrix requires 1--3 distinct seeds")
    raw_strategies = raw["strategies"]
    if isinstance(raw_strategies, (str, bytes)) or not isinstance(
        raw_strategies, Sequence
    ):
        raise ValueError("strategies must be a sequence")
    strategies = tuple(raw_strategies)
    if not strategies or len(set(strategies)) != len(strategies):
        raise ValueError("strategies must be non-empty and distinct")
    if any(strategy not in DECISION_STRATEGIES for strategy in strategies):
        raise ValueError("toy matrix contains an unsupported decision strategy")
    return ToyMatrixConfig(
        config_path=source,
        closed_loop_config_path=closed_loop_path,
        seeds=seeds,
        strategies=strategies,
        episode_count=_positive_int(raw["episode_count"], name="episode_count"),
    )


def build_toy_scenarios(seed: int, episode_count: int) -> tuple[ToyScenario, ...]:
    """Build a small deterministic mix of verify, reject, and execute states."""

    normalized_seed = _positive_int(seed, name="seed")
    count = _positive_int(episode_count, name="episode_count")
    seed_variant = normalized_seed % 3
    required_execute_time_s = 0.4 + 0.2 * seed_variant
    templates = (
        {
            "initial_calibrated_risk": 0.9,
            "post_verify_calibrated_risk": 0.1,
            "task_cost": 0.1,
            "reject_cost": 1.2,
            "required_execute_time_s": required_execute_time_s,
            "action_values": {"forward_peek": 0.8},
            "action_values_by_strategy": {
                "always": {"forward_peek": 0.0},
                "visible": {"stop_scan": 0.005},
                "swept": {"forward_peek": 0.008},
                "entropy": {"stop_scan": 0.004},
                "learned": {"forward_peek": 0.8},
                "oracle": {"forward_peek": 0.9},
            },
            "hazard_object_type": "human",
        },
        {
            "initial_calibrated_risk": 0.9,
            "post_verify_calibrated_risk": 0.9,
            "task_cost": 0.2,
            "reject_cost": 0.5,
            "required_execute_time_s": required_execute_time_s,
            "action_values": {},
            "hazard_object_type": "carried_object",
        },
        {
            "initial_calibrated_risk": 0.1,
            "post_verify_calibrated_risk": 0.1,
            "task_cost": 0.1,
            "reject_cost": 1.2,
            "required_execute_time_s": required_execute_time_s,
            "action_values": {},
            "hazard_object_type": "unknown_dynamic",
        },
    )
    offset = normalized_seed % len(templates)
    scenarios: list[ToyScenario] = []
    for index in range(count):
        template = templates[(index + offset) % len(templates)]
        scenarios.append(
            ToyScenario(
                episode_id=f"toy-{normalized_seed}-{index:03d}",
                **template,
            )
        )
    return tuple(scenarios)


def _scenario_payload(scenario: ToyScenario) -> dict[str, object]:
    return {
        "episode_id": scenario.episode_id,
        "initial_calibrated_risk": scenario.initial_calibrated_risk,
        "post_verify_calibrated_risk": scenario.post_verify_calibrated_risk,
        "task_cost": scenario.task_cost,
        "reject_cost": scenario.reject_cost,
        "required_execute_time_s": scenario.required_execute_time_s,
        "action_values": dict(scenario.action_values),
        "action_values_by_strategy": {
            strategy: dict(values)
            for strategy, values in scenario.action_values_by_strategy.items()
        },
        "hazard_object_type": scenario.hazard_object_type,
    }


def _toy_provenance(*, seed: int, scenarios: Sequence[ToyScenario]) -> ResultProvenance:
    input_manifest_digest = _sha256_json(
        {
            "closed_loop_version": CLOSED_LOOP_VERSION,
            "seed": seed,
            "scenarios": [_scenario_payload(scenario) for scenario in scenarios],
        }
    )
    dynamic_config_digest = _sha256_json(
        {"dynamic_object_types": DYNAMIC_OBJECT_TYPES, "geometry": "toy_scalar_risk"}
    )
    return ResultProvenance(
        schema_version=SCHEMA_VERSION,
        seed=seed,
        input_manifest_digest=input_manifest_digest,
        checkpoint_id="toy-no-checkpoint",
        scientific_status=TOY_SCIENTIFIC_STATUS,
        dynamic_objects_config_digest=dynamic_config_digest,
        target_type_policy="toy_mixed_typed_dynamic_objects",
        geometry_source="toy_scalar_risk",
        geometry_fallback_fraction=1.0,
    )


def run_toy_evaluation(
    *,
    config: ClosedLoopConfig,
    strategy: str,
    seed: int,
    episode_count: int,
    output_dir: str | Path,
) -> Path:
    """Run and register one deterministic strategy/seed toy evaluation."""

    if strategy not in DECISION_STRATEGIES:
        raise ValueError(f"unsupported decision strategy: {strategy!r}")
    normalized_seed = _positive_int(seed, name="seed")
    scenarios = build_toy_scenarios(normalized_seed, episode_count)
    traces: tuple[EpisodeTrace, ...] = tuple(
        run_toy_closed_loop(scenario, strategy=strategy, config=config)
        for scenario in scenarios
    )
    metrics = summarize_toy_episodes(traces)
    metrics["failure_count"] = float(sum(not trace.success for trace in traces))
    run_id = f"{strategy}-seed-{normalized_seed}"
    return publish_result(
        output_dir,
        run_id=run_id,
        config={
            "strategy": strategy,
            "seed": normalized_seed,
            "episode_count": _positive_int(episode_count, name="episode_count"),
            "closed_loop": closed_loop_config_payload(config),
        },
        provenance=_toy_provenance(seed=normalized_seed, scenarios=scenarios),
        metrics=metrics,
        episodes=[trace.as_dict() for trace in traces],
    )


def _write_json(path: Path, value: object) -> bytes:
    payload = _json_bytes(value)
    path.write_bytes(payload)
    return payload


def run_toy_experiment_matrix(
    config_path: str | Path,
    *,
    output_dir: str | Path,
) -> ToyMatrixResult:
    """Atomically publish all strategy/seed toy runs and their seed aggregates."""

    matrix = load_toy_matrix_config(config_path)
    closed_loop = load_closed_loop_config(matrix.closed_loop_config_path)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        staged_paths: list[Path] = []
        for strategy in matrix.strategies:
            for seed in matrix.seeds:
                run_id = f"{strategy}-seed-{seed}"
                staged_paths.append(
                    run_toy_evaluation(
                        config=closed_loop,
                        strategy=strategy,
                        seed=seed,
                        episode_count=matrix.episode_count,
                        output_dir=staging / run_id,
                    )
                )
        loaded_results: tuple[RegisteredResult, ...] = tuple(
            load_result(path) for path in staged_paths
        )
        aggregates = aggregate_seed_metrics(loaded_results)
        matrix_config = {
            "schema_version": SCHEMA_VERSION,
            "matrix_version": TOY_MATRIX_VERSION,
            "seeds": list(matrix.seeds),
            "strategies": list(matrix.strategies),
            "episode_count": matrix.episode_count,
            "closed_loop": closed_loop_config_payload(closed_loop),
        }
        summary = {
            "matrix_version": TOY_MATRIX_VERSION,
            "schema_version": SCHEMA_VERSION,
            "scientific_status": TOY_SCIENTIFIC_STATUS,
            "run_count": len(staged_paths),
            "aggregates": aggregates,
        }
        manifest = {
            "matrix_version": TOY_MATRIX_VERSION,
            "schema_version": SCHEMA_VERSION,
            "scientific_status": TOY_SCIENTIFIC_STATUS,
            "config_digest_sha256": _sha256_json(matrix_config),
            "summary_digest_sha256": _sha256_json(summary),
            "run_ids": [path.name for path in staged_paths],
        }
        manifest_payload = _write_json(staging / "matrix_manifest.json", manifest)
        _write_json(staging / "matrix_summary.json", summary)
        _write_json(
            staging / "COMPLETE.json",
            {"matrix_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest()},
        )
        atomic_rename_noreplace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    final_paths = tuple(destination / path.name for path in staged_paths)
    return ToyMatrixResult(output_dir=destination, run_paths=final_paths, summary=summary)


def load_toy_matrix_result(output_dir: str | Path) -> ToyMatrixResult:
    """Load a complete toy matrix and authenticate its summary and sub-runs."""

    root = Path(output_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("matrix output must be a real directory")
    manifest = _read_json_mapping(root / "matrix_manifest.json", label="matrix manifest")
    expected_manifest_keys = {
        "matrix_version",
        "schema_version",
        "scientific_status",
        "config_digest_sha256",
        "summary_digest_sha256",
        "run_ids",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("matrix manifest keys are invalid")
    if manifest["matrix_version"] != TOY_MATRIX_VERSION:
        raise ValueError("unsupported matrix version")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("matrix schema version mismatch")
    if manifest["scientific_status"] != TOY_SCIENTIFIC_STATUS:
        raise ValueError("matrix scientific status mismatch")
    run_ids = manifest["run_ids"]
    if (
        not isinstance(run_ids, list)
        or not run_ids
        or any(not isinstance(run_id, str) or not run_id for run_id in run_ids)
        or len(set(run_ids)) != len(run_ids)
    ):
        raise ValueError("matrix run IDs are invalid")
    summary = _read_json_mapping(root / "matrix_summary.json", label="matrix summary")
    if manifest["summary_digest_sha256"] != _sha256_json(summary):
        raise ValueError("matrix summary digest mismatch")
    complete = _read_json_mapping(root / "COMPLETE.json", label="matrix complete")
    if complete != {"matrix_manifest_sha256": _sha256_json(manifest)}:
        raise ValueError("matrix completion marker does not authenticate the manifest")
    expected_names = {
        "matrix_manifest.json",
        "matrix_summary.json",
        "COMPLETE.json",
        *run_ids,
    }
    if {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("matrix output layout is invalid")
    results = tuple(load_result(root / run_id) for run_id in run_ids)
    expected_summary = {
        "matrix_version": TOY_MATRIX_VERSION,
        "schema_version": SCHEMA_VERSION,
        "scientific_status": TOY_SCIENTIFIC_STATUS,
        "run_count": len(results),
        "aggregates": aggregate_seed_metrics(results),
    }
    if summary != expected_summary:
        raise ValueError("matrix summary does not match registered runs")
    return ToyMatrixResult(
        output_dir=root,
        run_paths=tuple(root / run_id for run_id in run_ids),
        summary=summary,
    )


__all__ = (
    "TOY_MATRIX_VERSION",
    "ToyMatrixConfig",
    "ToyMatrixResult",
    "build_toy_scenarios",
    "closed_loop_config_payload",
    "load_closed_loop_config",
    "load_toy_matrix_result",
    "load_toy_matrix_config",
    "run_toy_evaluation",
    "run_toy_experiment_matrix",
)
