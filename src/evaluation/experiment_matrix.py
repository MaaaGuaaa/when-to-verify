"""SOP-16 replay experiment matrix, aggregation, Pareto, and evidence index."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import csv
import hashlib
import io
import json
import math
from numbers import Integral
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any

import yaml

from src.contracts import SCHEMA_VERSION
from src.evaluation.closed_loop_replay import (
    SCIENTIFIC_STATUSES,
    LoadedReplaySuite,
    load_replay_suite,
    load_runtime_config,
    run_loaded_replay_evaluation,
)
from src.evaluation.closed_loop_runtime import (
    CLOSED_LOOP_RUNTIME_VERSION,
    ClosedLoopRuntimeConfig,
)
from src.evaluation.result_registry import (
    RegisteredResult,
    aggregate_seed_metrics,
    load_result,
)
from src.planning.decision_policy import DECISION_STRATEGIES
from src.utils.atomic_publish import atomic_rename_noreplace


EXPERIMENT_MATRIX_VERSION = "sop16_experiment_matrix_v1"
EXPERIMENT_MATRIX_LAYOUT_VERSION = "sop16_experiment_matrix_layout_v1"
EXPERIMENT_CATEGORIES = ("main", "ablation", "sensitivity", "controlled")
_TOP_LEVEL_FILES = frozenset(
    {
        "matrix_config.json",
        "matrix_summary.json",
        "pareto.json",
        "pareto.csv",
        "case_index.json",
        "claim_to_evidence.json",
        "failures.json",
        "matrix_manifest.json",
        "COMPLETE.json",
        "runs",
    }
)
_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "matrix_version",
        "matrix_name",
        "scientific_status",
        "closed_loop_config",
        "required_target_type_policy",
        "seeds",
        "strategies",
        "experiments",
    }
)
_EXPERIMENT_KEYS = frozenset(
    {
        "experiment_id",
        "category",
        "risk_method",
        "value_method",
        "suite_pattern",
        "parameters",
        "runtime_overrides",
        "claims",
    }
)
_RUNTIME_FIELDS = frozenset(asdict(ClosedLoopRuntimeConfig()))
_EXPERIMENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_PARETO_FIELDS = (
    "point_id",
    "front_id",
    "experiment_id",
    "category",
    "sensitivity_axis",
    "sensitivity_value",
    "strategy",
    "seed_count",
    "collision_rate_mean",
    "false_safe_execution_rate_mean",
    "verification_count_mean",
    "success_rate_mean",
    "extra_time_mean_s",
    "pareto_optimal",
)
_PARETO_SENSITIVITY_AXES = frozenset(
    {"verify_margin", "risk_weight", "verification_cost_scale"}
)


def _json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("matrix value must be finite JSON") from exc
    return (encoded + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _canonical_mapping(value: object, *, name: str) -> dict[str, object]:
    raw = _mapping(value, name=name)
    parsed = json.loads(_json_bytes(raw))
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must encode to an object")
    return parsed


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys are invalid")


def _positive_or_zero_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("seeds must contain non-negative integers")
    return int(value)


def _write_json(path: Path, value: object) -> bytes:
    payload = _json_bytes(value)
    path.write_bytes(payload)
    return payload


def _read_file(path: Path, *, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a real file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"invalid {name}: {exc}") from exc


def _read_json(path: Path, *, name: str) -> dict[str, object]:
    payload = _read_file(path, name=name)
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain an object")
    return value


def _suite_pattern(value: object) -> str:
    pattern = _string(value, name="suite_pattern")
    pure = PurePosixPath(pattern)
    if pure.is_absolute() or ".." in pure.parts or pattern.count("{seed}") != 1:
        raise ValueError(
            "suite_pattern must be a relative in-root path with one {seed}"
        )
    remaining = pattern.replace("{seed}", "")
    if "{" in remaining or "}" in remaining:
        raise ValueError("suite_pattern contains an unsupported placeholder")
    return pattern


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    category: str
    risk_method: str
    value_method: str
    suite_pattern: str
    parameters: Mapping[str, object]
    runtime_overrides: Mapping[str, object]
    claims: tuple[str, ...]

    def __post_init__(self) -> None:
        experiment_id = _string(self.experiment_id, name="experiment_id")
        if _EXPERIMENT_ID_PATTERN.fullmatch(experiment_id) is None:
            raise ValueError("experiment_id must be a lowercase filesystem-safe slug")
        object.__setattr__(self, "experiment_id", experiment_id)
        if self.category not in EXPERIMENT_CATEGORIES:
            raise ValueError("experiment category is unsupported")
        object.__setattr__(
            self, "risk_method", _string(self.risk_method, name="risk_method")
        )
        object.__setattr__(
            self, "value_method", _string(self.value_method, name="value_method")
        )
        object.__setattr__(self, "suite_pattern", _suite_pattern(self.suite_pattern))
        object.__setattr__(
            self,
            "parameters",
            _canonical_mapping(self.parameters, name="parameters"),
        )
        overrides = _canonical_mapping(
            self.runtime_overrides, name="runtime_overrides"
        )
        if not set(overrides) <= _RUNTIME_FIELDS:
            raise ValueError("runtime_overrides contains an unsupported field")
        ClosedLoopRuntimeConfig(**{**asdict(ClosedLoopRuntimeConfig()), **overrides})
        object.__setattr__(self, "runtime_overrides", overrides)
        if isinstance(self.claims, (str, bytes)) or not isinstance(
            self.claims, Sequence
        ):
            raise TypeError("claims must be a sequence")
        claims = tuple(_string(claim, name="claim") for claim in self.claims)
        if len(set(claims)) != len(claims):
            raise ValueError("claims must be distinct within an experiment")
        object.__setattr__(self, "claims", claims)


@dataclass(frozen=True)
class ExperimentMatrixConfig:
    path: Path
    matrix_name: str
    scientific_status: str
    closed_loop_config_path: Path
    required_target_type_policy: str
    seeds: tuple[int, ...]
    strategies: tuple[str, ...]
    experiments: tuple[ExperimentSpec, ...]


def _experiment_from_payload(value: object, *, index: int) -> ExperimentSpec:
    raw = _mapping(value, name=f"experiments[{index}]")
    _exact_keys(raw, _EXPERIMENT_KEYS, name=f"experiments[{index}]")
    return ExperimentSpec(
        experiment_id=raw["experiment_id"],
        category=raw["category"],
        risk_method=raw["risk_method"],
        value_method=raw["value_method"],
        suite_pattern=raw["suite_pattern"],
        parameters=raw["parameters"],
        runtime_overrides=raw["runtime_overrides"],
        claims=raw["claims"],
    )


def load_experiment_matrix_config(path: str | Path) -> ExperimentMatrixConfig:
    """Load the strict, path-portable SOP-16 matrix specification."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("matrix config must be a real file")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid matrix config: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("matrix config must contain a mapping")
    _exact_keys(raw, _CONFIG_KEYS, name="matrix config")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"matrix schema must be {SCHEMA_VERSION}")
    if raw["matrix_version"] != EXPERIMENT_MATRIX_VERSION:
        raise ValueError("unsupported experiment matrix version")
    if raw["scientific_status"] not in SCIENTIFIC_STATUSES:
        raise ValueError("matrix scientific_status is unsupported")
    raw_runtime_path = _string(
        raw["closed_loop_config"], name="closed_loop_config"
    )
    runtime_candidate = Path(raw_runtime_path)
    if runtime_candidate.is_absolute():
        raise ValueError("closed_loop_config must be a relative path")
    runtime_path = source.parent / runtime_candidate
    load_runtime_config(runtime_path)
    raw_seeds = raw["seeds"]
    if isinstance(raw_seeds, (str, bytes)) or not isinstance(raw_seeds, Sequence):
        raise TypeError("seeds must be a sequence")
    seeds = tuple(_positive_or_zero_seed(seed) for seed in raw_seeds)
    if not 1 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds):
        raise ValueError("matrix requires 1--3 distinct seeds")
    raw_strategies = raw["strategies"]
    if isinstance(raw_strategies, (str, bytes)) or not isinstance(
        raw_strategies, Sequence
    ):
        raise TypeError("strategies must be a sequence")
    strategies = tuple(raw_strategies)
    if not strategies or len(set(strategies)) != len(strategies):
        raise ValueError("strategies must be non-empty and distinct")
    if any(strategy not in DECISION_STRATEGIES for strategy in strategies):
        raise ValueError("matrix contains an unsupported strategy")
    raw_experiments = raw["experiments"]
    if isinstance(raw_experiments, (str, bytes)) or not isinstance(
        raw_experiments, Sequence
    ):
        raise TypeError("experiments must be a sequence")
    experiments = tuple(
        _experiment_from_payload(value, index=index)
        for index, value in enumerate(raw_experiments)
    )
    if not experiments:
        raise ValueError("experiments must not be empty")
    experiment_ids = [experiment.experiment_id for experiment in experiments]
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ValueError("experiment IDs must be distinct")
    return ExperimentMatrixConfig(
        path=source,
        matrix_name=_string(raw["matrix_name"], name="matrix_name"),
        scientific_status=str(raw["scientific_status"]),
        closed_loop_config_path=runtime_path,
        required_target_type_policy=_string(
            raw["required_target_type_policy"],
            name="required_target_type_policy",
        ),
        seeds=seeds,
        strategies=tuple(str(strategy) for strategy in strategies),
        experiments=experiments,
    )


def _runtime_payload(config: ClosedLoopRuntimeConfig) -> dict[str, object]:
    return {
        "runtime_version": CLOSED_LOOP_RUNTIME_VERSION,
        **asdict(config),
    }


def _experiment_payload(experiment: ExperimentSpec) -> dict[str, object]:
    return {
        "experiment_id": experiment.experiment_id,
        "category": experiment.category,
        "risk_method": experiment.risk_method,
        "value_method": experiment.value_method,
        "suite_pattern": experiment.suite_pattern,
        "parameters": dict(experiment.parameters),
        "runtime_overrides": dict(experiment.runtime_overrides),
        "claims": list(experiment.claims),
    }


def _experiment_run_identity(experiment: ExperimentSpec) -> dict[str, object]:
    return {
        "experiment_id": experiment.experiment_id,
        "category": experiment.category,
        "risk_method": experiment.risk_method,
        "value_method": experiment.value_method,
        "parameters": dict(experiment.parameters),
        "runtime_overrides": dict(experiment.runtime_overrides),
    }


def _config_payload(
    config: ExperimentMatrixConfig,
    runtime: ClosedLoopRuntimeConfig,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_version": EXPERIMENT_MATRIX_VERSION,
        "matrix_name": config.matrix_name,
        "scientific_status": config.scientific_status,
        "required_target_type_policy": config.required_target_type_policy,
        "seeds": list(config.seeds),
        "strategies": list(config.strategies),
        "closed_loop": _runtime_payload(runtime),
        "experiments": [
            _experiment_payload(experiment) for experiment in config.experiments
        ],
    }


def _runtime_with_overrides(
    base: ClosedLoopRuntimeConfig,
    overrides: Mapping[str, object],
) -> ClosedLoopRuntimeConfig:
    return ClosedLoopRuntimeConfig(**{**asdict(base), **dict(overrides)})


def _resolve_suite_path(
    suite_root: Path,
    experiment: ExperimentSpec,
    seed: int,
) -> tuple[Path, str]:
    relative = experiment.suite_pattern.format(seed=seed)
    candidate = suite_root / relative
    resolved_root = suite_root.resolve()
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("suite_pattern resolves outside suite_root") from exc
    return resolved_candidate, PurePosixPath(relative).as_posix()


def _validate_suite_for_run(
    suite: LoadedReplaySuite,
    *,
    config: ExperimentMatrixConfig,
    experiment: ExperimentSpec,
    runtime: ClosedLoopRuntimeConfig,
    seed: int,
) -> None:
    if suite.evidence.seed != seed:
        raise ValueError("replay suite seed does not match matrix seed")
    if suite.evidence.scientific_status != config.scientific_status:
        raise ValueError("replay suite scientific status does not match matrix")
    if suite.evidence.target_type_policy != config.required_target_type_policy:
        raise ValueError("replay suite target-type policy does not match matrix")
    binding = suite.evidence.experiment_binding
    if not binding:
        raise ValueError("replay suite experiment binding is required by matrix")
    if binding.get("risk_method") != experiment.risk_method:
        raise ValueError("replay suite experiment binding risk method differs")
    if binding.get("value_method") != experiment.value_method:
        raise ValueError("replay suite experiment binding value method differs")
    expected_parameters = dict(experiment.parameters)
    axis = expected_parameters.get("axis")
    if isinstance(axis, str) and axis in experiment.runtime_overrides:
        if expected_parameters.get("value") != experiment.runtime_overrides[axis]:
            raise ValueError(
                "matrix runtime override and sensitivity value differ"
            )
        expected_parameters.pop("axis")
        expected_parameters.pop("value")
    for name, value in experiment.runtime_overrides.items():
        if getattr(runtime, name) != value:
            raise ValueError("constructed runtime differs from runtime_overrides")
        if name in expected_parameters:
            if expected_parameters[name] != value:
                raise ValueError(
                    "matrix parameter and runtime override differ"
                )
            expected_parameters.pop(name)
    if binding.get("parameters") != expected_parameters:
        raise ValueError("replay suite experiment binding parameters differ")


def _group_results_by_experiment(
    results: Sequence[RegisteredResult],
) -> dict[str, list[RegisteredResult]]:
    grouped: dict[str, list[RegisteredResult]] = {}
    for result in results:
        experiment = result.config.get("experiment")
        if not isinstance(experiment, dict):
            raise ValueError("matrix run is missing experiment identity")
        experiment_id = _string(
            experiment.get("experiment_id"),
            name="result experiment_id",
        )
        grouped.setdefault(experiment_id, []).append(result)
    return grouped


def _build_experiment_summaries(
    config: ExperimentMatrixConfig,
    results: Sequence[RegisteredResult],
    failures: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped = _group_results_by_experiment(results)
    summaries: dict[str, object] = {}
    for experiment in config.experiments:
        rows = grouped.get(experiment.experiment_id, [])
        experiment_failures = [
            failure
            for failure in failures
            if failure["experiment_id"] == experiment.experiment_id
        ]
        summaries[experiment.experiment_id] = {
            "category": experiment.category,
            "risk_method": experiment.risk_method,
            "value_method": experiment.value_method,
            "parameters": dict(experiment.parameters),
            "runtime_overrides": dict(experiment.runtime_overrides),
            "expected_run_count": len(config.seeds) * len(config.strategies),
            "completed_run_count": len(rows),
            "failed_run_count": len(experiment_failures),
            "aggregates": aggregate_seed_metrics(rows) if rows else {},
        }
    return summaries


def _metric_mean(
    aggregate: Mapping[str, object],
    metric_name: str,
) -> float:
    metrics = aggregate.get("metrics")
    if not isinstance(metrics, Mapping) or metric_name not in metrics:
        raise ValueError(f"aggregate is missing metric {metric_name!r}")
    metric = metrics[metric_name]
    if not isinstance(metric, Mapping):
        raise ValueError(f"aggregate metric {metric_name!r} is invalid")
    value = metric.get("mean")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"aggregate metric {metric_name!r} mean is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"aggregate metric {metric_name!r} is not finite")
    return result


def _build_pareto_rows(
    config: ExperimentMatrixConfig,
    experiment_summaries: Mapping[str, object],
) -> list[dict[str, object]]:
    fronts: dict[str, list[dict[str, object]]] = {}
    for experiment in config.experiments:
        sensitivity_axis = experiment.parameters.get("axis")
        if experiment.category == "main":
            front_id = experiment.experiment_id
        elif (
            experiment.category == "sensitivity"
            and sensitivity_axis in _PARETO_SENSITIVITY_AXES
        ):
            front_id = f"sensitivity:{sensitivity_axis}"
        else:
            continue
        summary = experiment_summaries[experiment.experiment_id]
        if not isinstance(summary, Mapping):
            raise ValueError("experiment summary is invalid")
        aggregates = summary["aggregates"]
        if not isinstance(aggregates, Mapping):
            raise ValueError("experiment aggregates are invalid")
        for strategy, aggregate in sorted(aggregates.items()):
            if not isinstance(aggregate, Mapping):
                raise ValueError("strategy aggregate is invalid")
            fronts.setdefault(front_id, []).append(
                {
                    "point_id": f"{experiment.experiment_id}:{strategy}",
                    "front_id": front_id,
                    "experiment_id": experiment.experiment_id,
                    "category": experiment.category,
                    "sensitivity_axis": (
                        sensitivity_axis
                        if experiment.category == "sensitivity"
                        else None
                    ),
                    "sensitivity_value": (
                        experiment.parameters.get("value")
                        if experiment.category == "sensitivity"
                        else None
                    ),
                    "strategy": strategy,
                    "seed_count": aggregate["seed_count"],
                    "collision_rate_mean": _metric_mean(
                        aggregate, "collision_rate"
                    ),
                    "false_safe_execution_rate_mean": _metric_mean(
                        aggregate, "false_safe_execution_rate"
                    ),
                    "verification_count_mean": _metric_mean(
                        aggregate, "verification_count_mean"
                    ),
                    "success_rate_mean": _metric_mean(
                        aggregate, "success_rate"
                    ),
                    "extra_time_mean_s": _metric_mean(
                        aggregate, "extra_time_mean_s"
                    ),
                    "pareto_optimal": False,
                }
            )
    rows: list[dict[str, object]] = []
    for front_rows in fronts.values():
        for row in front_rows:
            dominated = any(
                other is not row
                and other["collision_rate_mean"] <= row["collision_rate_mean"]
                and other["verification_count_mean"]
                <= row["verification_count_mean"]
                and (
                    other["collision_rate_mean"] < row["collision_rate_mean"]
                    or other["verification_count_mean"]
                    < row["verification_count_mean"]
                )
                for other in front_rows
            )
            row["pareto_optimal"] = not dominated
        rows.extend(front_rows)
    return rows


def _pareto_csv(rows: Sequence[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=_PARETO_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        rendered = dict(row)
        rendered["pareto_optimal"] = (
            "true" if row["pareto_optimal"] else "false"
        )
        writer.writerow(rendered)
    return stream.getvalue().encode("utf-8")


def _episode_labels(episode: Mapping[str, object]) -> tuple[str, ...]:
    labels: list[str] = []
    if episode.get("collision") is True:
        labels.append("collision")
    if episode.get("false_safe_execute") is True:
        labels.append("false_safe")
    if episode.get("near_miss") is True:
        labels.append("near_miss")
    if (
        isinstance(episode.get("unnecessary_verification_count"), int)
        and episode["unnecessary_verification_count"] > 0
    ):
        labels.append("unnecessary_verification")
    if isinstance(episode.get("reject_count"), int) and episode["reject_count"] > 0:
        labels.append("reject")
    if episode.get("success") is True:
        labels.append("success")
    if not labels:
        labels.append("other_failure")
    return tuple(labels)


def _build_case_index(
    results: Sequence[RegisteredResult],
    relative_run_paths: Mapping[str, str],
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for result in sorted(results, key=lambda row: row.run_id):
        experiment = result.config["experiment"]
        if not isinstance(experiment, Mapping):
            raise ValueError("result experiment metadata is invalid")
        for episode in sorted(
            result.episodes, key=lambda row: str(row.get("episode_id", ""))
        ):
            labels = _episode_labels(episode)
            candidates.append(
                {
                    "case_id": f"{result.run_id}:{episode['episode_id']}",
                    "run_id": result.run_id,
                    "run_path": relative_run_paths[result.run_id],
                    "experiment_id": experiment["experiment_id"],
                    "strategy": result.config["strategy"],
                    "seed": result.provenance.seed,
                    "episode_id": episode["episode_id"],
                    "termination_reason": episode.get("termination_reason"),
                    "labels": list(labels),
                }
            )
    label_order = (
        "collision",
        "false_safe",
        "near_miss",
        "unnecessary_verification",
        "reject",
        "success",
        "other_failure",
    )
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    while len(selected) < min(10, len(candidates)):
        added = False
        for label in label_order:
            candidate = next(
                (
                    row
                    for row in candidates
                    if row["case_id"] not in selected_ids
                    and label in row["labels"]
                ),
                None,
            )
            if candidate is not None:
                selected.append(candidate)
                selected_ids.add(str(candidate["case_id"]))
                added = True
                if len(selected) == 10:
                    break
        if not added:
            break
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_version": EXPERIMENT_MATRIX_VERSION,
        "selection": "deterministic_round_robin_by_outcome_v1",
        "case_count": len(selected),
        "requirement_met": 5 <= len(selected) <= 10,
        "cases": selected,
    }


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _build_claim_index(
    config: ExperimentMatrixConfig,
    results: Sequence[RegisteredResult],
    pareto_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    runs_by_experiment: dict[str, list[str]] = {}
    for result in results:
        experiment = result.config["experiment"]
        if not isinstance(experiment, Mapping):
            raise ValueError("result experiment metadata is invalid")
        runs_by_experiment.setdefault(
            str(experiment["experiment_id"]), []
        ).append(result.run_id)
    claims: list[dict[str, object]] = []
    for experiment in config.experiments:
        run_ids = sorted(runs_by_experiment.get(experiment.experiment_id, []))
        point_ids = sorted(
            str(row["point_id"])
            for row in pareto_rows
            if row["experiment_id"] == experiment.experiment_id
        )
        for index, claim in enumerate(experiment.claims):
            claims.append(
                {
                    "claim_id": f"{experiment.experiment_id}:claim-{index + 1}",
                    "claim": claim,
                    "category": experiment.category,
                    "status": (
                        "supported_by_structured_artifacts"
                        if run_ids
                        else "missing_due_to_recorded_failures"
                    ),
                    "evidence": {
                        "summary_file": "matrix_summary.json",
                        "summary_json_pointer": (
                            "/experiments/"
                            + _json_pointer_token(experiment.experiment_id)
                        ),
                        "run_ids": run_ids,
                        "pareto_point_ids": point_ids,
                    },
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_version": EXPERIMENT_MATRIX_VERSION,
        "claims": claims,
    }


def _build_acceptance_gates(
    config: ExperimentMatrixConfig,
    experiment_summaries: Mapping[str, object],
    pareto_rows: Sequence[Mapping[str, object]],
    case_index: Mapping[str, object],
) -> dict[str, object]:
    matched_budgets: list[dict[str, object]] = []
    for experiment in config.experiments:
        if experiment.category != "main":
            continue
        summary = experiment_summaries[experiment.experiment_id]
        if not isinstance(summary, Mapping):
            continue
        aggregates = summary["aggregates"]
        if not isinstance(aggregates, Mapping) or not {
            "learned",
            "visible",
        } <= set(aggregates):
            continue
        learned = aggregates["learned"]
        visible = aggregates["visible"]
        if not isinstance(learned, Mapping) or not isinstance(visible, Mapping):
            continue
        learned_budget = _metric_mean(learned, "verification_count_mean")
        visible_budget = _metric_mean(visible, "verification_count_mean")
        if math.isclose(
            learned_budget, visible_budget, rel_tol=0.0, abs_tol=1e-9
        ):
            learned_false_safe = _metric_mean(
                learned, "false_safe_execution_rate"
            )
            visible_false_safe = _metric_mean(
                visible, "false_safe_execution_rate"
            )
            matched_budgets.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "verification_budget": learned_budget,
                    "learned_false_safe": learned_false_safe,
                    "visible_false_safe": visible_false_safe,
                    "passed": learned_false_safe < visible_false_safe,
                }
            )
    if matched_budgets:
        learned_gate_status = (
            "pass" if any(row["passed"] for row in matched_budgets) else "fail"
        )
        learned_reason = "compared at exactly matched mean verification budgets"
    else:
        learned_gate_status = "not_evaluable"
        learned_reason = "no main learned/visible pair has an exactly matched budget"

    calibrated: list[tuple[ExperimentSpec, Mapping[str, object]]] = []
    uncalibrated: list[tuple[ExperimentSpec, Mapping[str, object]]] = []
    for experiment in config.experiments:
        if experiment.category != "main":
            continue
        summary = experiment_summaries[experiment.experiment_id]
        if not isinstance(summary, Mapping):
            continue
        aggregates = summary["aggregates"]
        if not isinstance(aggregates, Mapping):
            continue
        learned = aggregates.get("learned")
        if not isinstance(learned, Mapping):
            continue
        if experiment.risk_method == "risk_calibration":
            calibrated.append((experiment, learned))
        elif experiment.risk_method == "risk_only":
            uncalibrated.append((experiment, learned))
    calibration_comparisons: list[dict[str, object]] = []
    for calibrated_experiment, calibrated_aggregate in calibrated:
        for raw_experiment, raw_aggregate in uncalibrated:
            if calibrated_experiment.value_method != raw_experiment.value_method:
                continue
            calibrated_false_safe = _metric_mean(
                calibrated_aggregate, "false_safe_execution_rate"
            )
            raw_false_safe = _metric_mean(
                raw_aggregate, "false_safe_execution_rate"
            )
            calibration_comparisons.append(
                {
                    "calibrated_experiment_id": (
                        calibrated_experiment.experiment_id
                    ),
                    "uncalibrated_experiment_id": raw_experiment.experiment_id,
                    "calibrated_false_safe": calibrated_false_safe,
                    "uncalibrated_false_safe": raw_false_safe,
                    "passed": calibrated_false_safe < raw_false_safe,
                }
            )
    if calibration_comparisons:
        calibration_status = (
            "pass"
            if any(row["passed"] for row in calibration_comparisons)
            else "fail"
        )
        calibration_reason = "compared learned policy with a common value method"
    else:
        calibration_status = "not_evaluable"
        calibration_reason = "risk_only/risk_calibration learned pair is unavailable"

    pareto_status = "pass" if any(
        row.get("pareto_optimal") is True for row in pareto_rows
    ) else "fail"
    case_count = case_index.get("case_count")
    case_status = (
        "pass"
        if isinstance(case_count, int) and 5 <= case_count <= 10
        else "fail"
    )
    return {
        "learned_vs_visible_fixed_budget": {
            "status": learned_gate_status,
            "direction": "lower false-safe at equal mean verification count",
            "reason": learned_reason,
            "comparisons": matched_budgets,
        },
        "calibration_reduces_false_safe": {
            "status": calibration_status,
            "direction": "lower is better",
            "reason": calibration_reason,
            "comparisons": calibration_comparisons,
        },
        "pareto_available": {
            "status": pareto_status,
            "pareto_point_count": len(pareto_rows),
        },
        "case_trace_count": {
            "status": case_status,
            "required_minimum": 5,
            "required_maximum": 10,
            "observed": case_count,
        },
    }


@dataclass(frozen=True)
class ExperimentMatrixResult:
    output_dir: Path
    run_paths: tuple[Path, ...]
    summary: dict[str, object]
    pareto_rows: tuple[dict[str, object], ...]
    case_index: dict[str, object]
    claim_index: dict[str, object]
    failures: dict[str, object]


def _failure_record(
    *,
    run_id: str,
    experiment: ExperimentSpec,
    strategy: str,
    seed: int,
    suite_relative_path: str,
    error: BaseException,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "experiment_id": experiment.experiment_id,
        "category": experiment.category,
        "strategy": strategy,
        "seed": seed,
        "suite_relative_path": suite_relative_path,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def run_experiment_matrix(
    config_path: str | Path,
    *,
    suite_root: str | Path,
    output_dir: str | Path,
) -> ExperimentMatrixResult:
    """Run every configured replay strategy/seed and publish one sealed matrix."""

    config = load_experiment_matrix_config(config_path)
    runtime = load_runtime_config(config.closed_loop_config_path)
    suites = Path(suite_root)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    results: list[RegisteredResult] = []
    relative_run_paths: dict[str, str] = {}
    failures: list[dict[str, object]] = []
    try:
        runs_root = staging / "runs"
        runs_root.mkdir()
        for experiment in config.experiments:
            experiment_runtime = _runtime_with_overrides(
                runtime, experiment.runtime_overrides
            )
            for seed in config.seeds:
                suite_path, suite_relative = _resolve_suite_path(
                    suites, experiment, seed
                )
                try:
                    suite = load_replay_suite(suite_path)
                    _validate_suite_for_run(
                        suite,
                        config=config,
                        experiment=experiment,
                        runtime=experiment_runtime,
                        seed=seed,
                    )
                except Exception as error:
                    for strategy in config.strategies:
                        run_id = (
                            f"{experiment.experiment_id}--{strategy}--seed-{seed}"
                        )
                        failures.append(
                            _failure_record(
                                run_id=run_id,
                                experiment=experiment,
                                strategy=strategy,
                                seed=seed,
                                suite_relative_path=suite_relative,
                                error=error,
                            )
                        )
                    continue
                for strategy in config.strategies:
                    run_id = f"{experiment.experiment_id}--{strategy}--seed-{seed}"
                    relative_path = f"runs/{run_id}"
                    try:
                        path = run_loaded_replay_evaluation(
                            suite=suite,
                            strategy=strategy,
                            config=experiment_runtime,
                            output_dir=staging / relative_path,
                            run_id=run_id,
                            experiment=_experiment_run_identity(experiment),
                        )
                        result = load_result(path)
                        results.append(result)
                        relative_run_paths[run_id] = relative_path
                    except Exception as error:
                        if (staging / relative_path).exists():
                            raise RuntimeError(
                                "a published run failed strict reload"
                            ) from error
                        failures.append(
                            _failure_record(
                                run_id=run_id,
                                experiment=experiment,
                                strategy=strategy,
                                seed=seed,
                                suite_relative_path=suite_relative,
                                error=error,
                            )
                        )
        experiment_summaries = _build_experiment_summaries(
            config, results, failures
        )
        pareto_rows = _build_pareto_rows(config, experiment_summaries)
        case_index = _build_case_index(results, relative_run_paths)
        claim_index = _build_claim_index(config, results, pareto_rows)
        acceptance_gates = _build_acceptance_gates(
            config,
            experiment_summaries,
            pareto_rows,
            case_index,
        )
        expected_run_count = (
            len(config.experiments) * len(config.seeds) * len(config.strategies)
        )
        needs_pareto = any(
            experiment.category == "main"
            or (
                experiment.category == "sensitivity"
                and experiment.parameters.get("axis")
                in _PARETO_SENSITIVITY_AXES
            )
            for experiment in config.experiments
        )
        scientifically_complete = (
            len(results) == expected_run_count
            and not failures
            and bool(case_index["requirement_met"])
            and (bool(pareto_rows) or not needs_pareto)
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "matrix_version": EXPERIMENT_MATRIX_VERSION,
            "matrix_name": config.matrix_name,
            "scientific_status": config.scientific_status,
            "expected_run_count": expected_run_count,
            "completed_run_count": len(results),
            "failed_run_count": len(failures),
            "scientifically_complete": scientifically_complete,
            "case_count": case_index["case_count"],
            "case_requirement_met": case_index["requirement_met"],
            "pareto_point_count": len(pareto_rows),
            "acceptance_gates": acceptance_gates,
            "experiments": experiment_summaries,
        }
        failures_document = {
            "schema_version": SCHEMA_VERSION,
            "matrix_version": EXPERIMENT_MATRIX_VERSION,
            "failure_count": len(failures),
            "failures": failures,
        }
        config_document = _config_payload(config, runtime)
        data_payloads = {
            "matrix_config.json": _write_json(
                staging / "matrix_config.json", config_document
            ),
            "matrix_summary.json": _write_json(
                staging / "matrix_summary.json", summary
            ),
            "pareto.json": _write_json(
                staging / "pareto.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "matrix_version": EXPERIMENT_MATRIX_VERSION,
                    "rows": pareto_rows,
                },
            ),
            "case_index.json": _write_json(
                staging / "case_index.json", case_index
            ),
            "claim_to_evidence.json": _write_json(
                staging / "claim_to_evidence.json", claim_index
            ),
            "failures.json": _write_json(
                staging / "failures.json", failures_document
            ),
        }
        pareto_csv_payload = _pareto_csv(pareto_rows)
        (staging / "pareto.csv").write_bytes(pareto_csv_payload)
        data_payloads["pareto.csv"] = pareto_csv_payload
        run_paths = [
            relative_run_paths[result.run_id]
            for result in sorted(results, key=lambda row: row.run_id)
        ]
        manifest = {
            "matrix_layout_version": EXPERIMENT_MATRIX_LAYOUT_VERSION,
            "matrix_version": EXPERIMENT_MATRIX_VERSION,
            "schema_version": SCHEMA_VERSION,
            "matrix_name": config.matrix_name,
            "scientific_status": config.scientific_status,
            "data_file_digests_sha256": {
                name: _sha256(payload)
                for name, payload in sorted(data_payloads.items())
            },
            "run_paths": run_paths,
            "run_manifest_digests_sha256": {
                relative_path: _sha256(
                    (staging / relative_path / "manifest.json").read_bytes()
                )
                for relative_path in run_paths
            },
        }
        manifest_payload = _write_json(
            staging / "matrix_manifest.json", manifest
        )
        _write_json(
            staging / "COMPLETE.json",
            {
                "matrix_layout_version": EXPERIMENT_MATRIX_LAYOUT_VERSION,
                "matrix_manifest_sha256": _sha256(manifest_payload),
            },
        )
        atomic_rename_noreplace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return ExperimentMatrixResult(
        output_dir=destination,
        run_paths=tuple(destination / path for path in run_paths),
        summary=summary,
        pareto_rows=tuple(pareto_rows),
        case_index=case_index,
        claim_index=claim_index,
        failures=failures_document,
    )


def _safe_run_path(root: Path, value: object) -> tuple[str, Path]:
    relative = _string(value, name="run path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 2:
        raise ValueError("matrix run path is invalid")
    if pure.parts[0] != "runs":
        raise ValueError("matrix run path must be under runs")
    path = root / Path(*pure.parts)
    if path.resolve(strict=False).parent != (root / "runs").resolve(strict=False):
        raise ValueError("matrix run path escapes runs")
    return relative, path


def load_experiment_matrix_result(
    output_dir: str | Path,
) -> ExperimentMatrixResult:
    """Authenticate a complete matrix artifact and every registered sub-run."""

    root = Path(output_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("matrix result must be a real directory")
    if {path.name for path in root.iterdir()} != _TOP_LEVEL_FILES:
        raise ValueError("matrix result has an invalid or incomplete layout")
    manifest_payload = _read_file(
        root / "matrix_manifest.json", name="matrix_manifest.json"
    )
    try:
        manifest_value = json.loads(manifest_payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid matrix manifest: {exc}") from exc
    manifest = _mapping(manifest_value, name="matrix manifest")
    expected_manifest_keys = frozenset(
        {
            "matrix_layout_version",
            "matrix_version",
            "schema_version",
            "matrix_name",
            "scientific_status",
            "data_file_digests_sha256",
            "run_paths",
            "run_manifest_digests_sha256",
        }
    )
    _exact_keys(manifest, expected_manifest_keys, name="matrix manifest")
    if manifest["matrix_layout_version"] != EXPERIMENT_MATRIX_LAYOUT_VERSION:
        raise ValueError("unsupported matrix result layout")
    if manifest["matrix_version"] != EXPERIMENT_MATRIX_VERSION:
        raise ValueError("unsupported matrix result version")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("matrix result schema mismatch")
    complete = _read_json(root / "COMPLETE.json", name="COMPLETE.json")
    if complete != {
        "matrix_layout_version": EXPERIMENT_MATRIX_LAYOUT_VERSION,
        "matrix_manifest_sha256": _sha256(manifest_payload),
    }:
        raise ValueError("matrix completion marker does not authenticate manifest")
    declared_files = _mapping(
        manifest["data_file_digests_sha256"],
        name="data file digests",
    )
    expected_data_files = _TOP_LEVEL_FILES - {
        "matrix_manifest.json",
        "COMPLETE.json",
        "runs",
    }
    if set(declared_files) != expected_data_files:
        raise ValueError("matrix data file digest inventory is invalid")
    for name, digest in declared_files.items():
        if digest != _sha256(_read_file(root / name, name=name)):
            raise ValueError(f"matrix data file digest mismatch: {name}")
    raw_run_paths = manifest["run_paths"]
    if isinstance(raw_run_paths, (str, bytes)) or not isinstance(
        raw_run_paths, Sequence
    ):
        raise ValueError("matrix run paths are invalid")
    if len(set(raw_run_paths)) != len(raw_run_paths):
        raise ValueError("matrix run paths are duplicated")
    declared_run_digests = _mapping(
        manifest["run_manifest_digests_sha256"],
        name="run manifest digests",
    )
    if set(declared_run_digests) != set(raw_run_paths):
        raise ValueError("matrix run manifest inventory is invalid")
    results: list[RegisteredResult] = []
    paths: list[Path] = []
    for raw_path in raw_run_paths:
        relative, path = _safe_run_path(root, raw_path)
        if declared_run_digests[relative] != _sha256(
            _read_file(path / "manifest.json", name="run manifest")
        ):
            raise ValueError("matrix run manifest digest mismatch")
        results.append(load_result(path))
        paths.append(path)
    if {path.name for path in (root / "runs").iterdir()} != {
        path.name for path in paths
    }:
        raise ValueError("matrix runs directory contains undeclared entries")
    summary = _read_json(root / "matrix_summary.json", name="matrix_summary.json")
    if summary.get("completed_run_count") != len(results):
        raise ValueError("matrix completed run count does not match run inventory")
    pareto = _read_json(root / "pareto.json", name="pareto.json")
    if set(pareto) != {"schema_version", "matrix_version", "rows"}:
        raise ValueError("pareto document keys are invalid")
    raw_rows = pareto["rows"]
    if not isinstance(raw_rows, list):
        raise ValueError("pareto rows must be a list")
    case_index = _read_json(root / "case_index.json", name="case_index.json")
    claim_index = _read_json(
        root / "claim_to_evidence.json", name="claim_to_evidence.json"
    )
    failures = _read_json(root / "failures.json", name="failures.json")
    if summary.get("failed_run_count") != failures.get("failure_count"):
        raise ValueError("matrix failure count does not match failure inventory")
    return ExperimentMatrixResult(
        output_dir=root,
        run_paths=tuple(paths),
        summary=summary,
        pareto_rows=tuple(dict(row) for row in raw_rows),
        case_index=case_index,
        claim_index=claim_index,
        failures=failures,
    )


__all__ = (
    "EXPERIMENT_CATEGORIES",
    "EXPERIMENT_MATRIX_LAYOUT_VERSION",
    "EXPERIMENT_MATRIX_VERSION",
    "ExperimentMatrixConfig",
    "ExperimentMatrixResult",
    "ExperimentSpec",
    "load_experiment_matrix_config",
    "load_experiment_matrix_result",
    "run_experiment_matrix",
)
