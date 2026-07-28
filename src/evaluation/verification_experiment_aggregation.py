"""Strict multi-seed aggregation for verification-value evaluations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

from src.contracts import SCHEMA_VERSION
from src.evaluation.verification_run_artifacts import (
    VERIFICATION_EVALUATION_RUN_LAYOUT_VERSION,
    canonical_json_bytes,
    load_authenticated_run_directory,
    publish_authenticated_run_directory,
    sha256_file,
    strict_json_file,
)


VERIFICATION_AGGREGATION_VERSION = "verification_experiment_aggregation_v1"
VERIFICATION_AGGREGATE_LAYOUT_VERSION = "verification_aggregate_run_v1"
_PAYLOADS = frozenset({"summary.json", "metrics_long.csv", "manifest.json"})
_EVALUATION_PAYLOADS = frozenset(
    {"evaluation_report.json", "metrics.json", "predictions.jsonl"}
)
_SPLITS = frozenset({"calibration", "val", "test"})


@dataclass(frozen=True)
class LoadedVerificationAggregate:
    root: Path
    experiment_id: str
    run_count: int
    seeds: tuple[int, ...]
    aggregate_digest: str
    summary: Mapping[str, object]
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "summary",
            MappingProxyType(dict(self.summary)),
        )
        object.__setattr__(
            self,
            "manifest",
            MappingProxyType(dict(self.manifest)),
        )


@dataclass(frozen=True)
class _EvaluationEvidence:
    root: Path
    seed: int
    split: str
    split_digest: str
    calibration_digest: str | None
    reject_cost: float | None
    model_identity: object
    checkpoint_code_version: str
    data_mode: str
    report: dict[str, object]
    metrics: dict[str, object]
    payload_digests: Mapping[str, str]


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_copy(value: object, *, name: str) -> object:
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be finite canonical JSON") from exc


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _model_identity(
    report: Mapping[str, object],
    *,
    seed: int,
) -> object:
    config = _canonical_copy(report.get("model_config"), name="model_config")
    if not isinstance(config, dict) or not config:
        raise ValueError("evaluation model config is invalid")
    if report.get("model_config_digest") != _sha256_json(config):
        raise ValueError("evaluation model config digest differs")
    training = config.get("training")
    if not isinstance(training, dict) or training.get("seed") != seed:
        raise ValueError("evaluation model config seed differs")
    training.pop("seed")
    return config


def _read_predictions(path: Path, *, expected_count: int) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("evaluation predictions must be a real file")

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    sample_ids: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            row = json.loads(line, parse_constant=reject_constant)
            if not isinstance(row, dict):
                raise ValueError("prediction row must be an object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("prediction sample ID is invalid")
            sample_ids.append(sample_id)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid evaluation predictions: {exc}") from exc
    if (
        len(sample_ids) != expected_count
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError("evaluation prediction count or IDs differ")


def _load_evaluation(root: Path) -> _EvaluationEvidence:
    complete = load_authenticated_run_directory(
        root,
        expected_layout_version=VERIFICATION_EVALUATION_RUN_LAYOUT_VERSION,
        required_payloads=_EVALUATION_PAYLOADS,
    )
    report = strict_json_file(
        root / "evaluation_report.json",
        label="evaluation report",
    )
    metrics = strict_json_file(root / "metrics.json", label="evaluation metrics")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or metrics.get("schema_version") != SCHEMA_VERSION
        or report.get("run_layout_version")
        != VERIFICATION_EVALUATION_RUN_LAYOUT_VERSION
    ):
        raise ValueError("evaluation schema or layout differs")
    split = report.get("split")
    if split not in _SPLITS or metrics.get("split") != split:
        raise ValueError("evaluation split is invalid or inconsistent")
    seed = report.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or metrics.get("seed") != seed
    ):
        raise ValueError("evaluation seed is invalid or inconsistent")
    split_digests = report.get("evaluation_split_digests")
    if not isinstance(split_digests, dict) or set(split_digests) != {split}:
        raise ValueError("evaluation split digest map is invalid")
    split_digest = _digest(
        split_digests[split],
        name="evaluation split digest",
    )
    calibration_raw = report.get("value_calibration_digest")
    reject_raw = report.get("reject_cost")
    if calibration_raw is None and reject_raw is None:
        calibration_digest = None
        reject_cost = None
    elif calibration_raw is None or reject_raw is None:
        raise ValueError("evaluation calibration binding is incomplete")
    else:
        calibration_digest = _digest(
            calibration_raw,
            name="evaluation calibration digest",
        )
        if isinstance(reject_raw, bool) or not isinstance(reject_raw, Real):
            raise ValueError("evaluation reject cost is invalid")
        reject_cost = float(reject_raw)
        if not math.isfinite(reject_cost) or reject_cost < 0.0:
            raise ValueError("evaluation reject cost is invalid")
    if (
        metrics.get("value_calibration_digest") != calibration_digest
        or metrics.get("reject_cost") != reject_cost
    ):
        raise ValueError("evaluation calibration metadata differs")
    for section in ("losses", "learned", "baselines"):
        if not isinstance(metrics.get(section), dict):
            raise ValueError(f"evaluation {section} metrics are invalid")
    count = report.get("sample_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("evaluation sample count is invalid")
    _read_predictions(root / "predictions.jsonl", expected_count=count)
    checkpoint_code_version = report.get("checkpoint_code_version")
    data_mode = report.get("data_mode")
    if (
        not isinstance(checkpoint_code_version, str)
        or not checkpoint_code_version
        or data_mode not in {"release", "bounded_fixture"}
    ):
        raise ValueError("evaluation implementation identity is invalid")
    _digest(report.get("checkpoint_sha256"), name="checkpoint digest")
    _digest(report.get("run_identity"), name="evaluation run identity")
    raw_files = complete["files"]
    if not isinstance(raw_files, Mapping):
        raise ValueError("evaluation completion payload map is invalid")
    return _EvaluationEvidence(
        root=root,
        seed=seed,
        split=str(split),
        split_digest=split_digest,
        calibration_digest=calibration_digest,
        reject_cost=reject_cost,
        model_identity=_model_identity(report, seed=seed),
        checkpoint_code_version=checkpoint_code_version,
        data_mode=str(data_mode),
        report=report,
        metrics=metrics,
        payload_digests=MappingProxyType(
            {str(key): str(value) for key, value in raw_files.items()}
        ),
    )


def _flatten_numeric(
    value: object,
    *,
    prefix: str = "",
) -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise ValueError("metric keys must be non-empty strings")
            child = f"{prefix}.{key}" if prefix else key
            result.update(_flatten_numeric(value[key], prefix=child))
    elif not isinstance(value, bool) and isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("aggregate metrics must be finite")
        result[prefix] = number
    return result


def _numeric_sections(
    evidence: _EvaluationEvidence,
) -> dict[str, dict[str, float]]:
    return {
        section: _flatten_numeric(evidence.metrics[section])
        for section in ("losses", "learned", "baselines")
    }


def _slice_summary(evidence: _EvaluationEvidence) -> dict[str, object]:
    learned = evidence.metrics["learned"]
    baselines = evidence.metrics["baselines"]
    assert isinstance(learned, dict)
    assert isinstance(baselines, dict)
    baseline_slices = {
        name: value.get("slices", {})
        for name, value in sorted(baselines.items())
        if isinstance(value, Mapping)
    }
    return {
        "learned": _canonical_copy(
            learned.get("slices", {}),
            name="learned slice summaries",
        ),
        "baselines": _canonical_copy(
            baseline_slices,
            name="baseline slice summaries",
        ),
    }


def _validate_identity(evidence: tuple[_EvaluationEvidence, ...]) -> None:
    seeds = [row.seed for row in evidence]
    if len(set(seeds)) != len(seeds):
        raise ValueError("aggregate evaluation inputs contain duplicate seed")
    first = evidence[0]
    for row in evidence[1:]:
        if row.split != first.split:
            raise ValueError("aggregate evaluation split differs")
        if row.split_digest != first.split_digest:
            raise ValueError("aggregate evaluation split digest differs")
        if (
            row.calibration_digest != first.calibration_digest
            or row.reject_cost != first.reject_cost
        ):
            raise ValueError("aggregate evaluation calibration differs")
        if row.model_identity != first.model_identity:
            raise ValueError("aggregate evaluation model config differs beyond seed")
        if row.checkpoint_code_version != first.checkpoint_code_version:
            raise ValueError("aggregate checkpoint code version differs")
        if row.data_mode != first.data_mode:
            raise ValueError("aggregate evaluation data mode differs")


def _summary_document(
    evidence: tuple[_EvaluationEvidence, ...],
    *,
    experiment_id: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    sections = [_numeric_sections(row) for row in evidence]
    expected_paths = {
        section: set(sections[0][section])
        for section in ("losses", "learned", "baselines")
    }
    for values in sections[1:]:
        for section, paths in expected_paths.items():
            if set(values[section]) != paths:
                raise ValueError(
                    f"aggregate {section} metric paths differ across seeds"
                )
    aggregate_metrics: dict[str, dict[str, object]] = {}
    long_rows: list[dict[str, object]] = []
    for section in ("losses", "learned", "baselines"):
        aggregate_metrics[section] = {}
        for metric in sorted(expected_paths[section]):
            values = [
                sections[index][section][metric]
                for index in range(len(evidence))
            ]
            mean = math.fsum(values) / len(values)
            population_std = math.sqrt(
                math.fsum((value - mean) ** 2 for value in values)
                / len(values)
            )
            per_seed = {
                str(row.seed): values[index]
                for index, row in enumerate(evidence)
            }
            aggregate_metrics[section][metric] = {
                "per_seed": per_seed,
                "mean": mean,
                "population_std": population_std,
            }
            for index, row in enumerate(evidence):
                long_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "section": section,
                        "metric": metric,
                        "statistic": "seed",
                        "seed": row.seed,
                        "value": values[index],
                    }
                )
            for statistic, value in (
                ("mean", mean),
                ("population_std", population_std),
            ):
                long_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "section": section,
                        "metric": metric,
                        "statistic": statistic,
                        "seed": "",
                        "value": value,
                    }
                )
    first = evidence[0]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "aggregation_version": VERIFICATION_AGGREGATION_VERSION,
        "experiment_id": experiment_id,
        "split": first.split,
        "evaluation_split_digest": first.split_digest,
        "value_calibration_digest": first.calibration_digest,
        "reject_cost": first.reject_cost,
        "checkpoint_code_version": first.checkpoint_code_version,
        "data_mode": first.data_mode,
        "run_count": len(evidence),
        "seeds": [row.seed for row in evidence],
        "metrics": aggregate_metrics,
        "slice_summaries": {
            str(row.seed): _slice_summary(row) for row in evidence
        },
        "source_runs": [
            {
                "seed": row.seed,
                "run_identity": row.report["run_identity"],
                "payload_digests": dict(row.payload_digests),
            }
            for row in evidence
        ],
    }
    return summary, long_rows


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    output = io.StringIO(newline="")
    fieldnames = (
        "experiment_id",
        "section",
        "metric",
        "statistic",
        "seed",
        "value",
    )
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        serialized = dict(row)
        serialized["value"] = format(float(serialized["value"]), ".17g")
        writer.writerow(serialized)
    return output.getvalue().encode("utf-8")


def _aggregate_digest(summary: object) -> str:
    return hashlib.sha256(
        b"verification-experiment-aggregation-v1\0"
        + canonical_json_bytes(summary)
    ).hexdigest()


def load_verification_aggregate(
    input_dir: str | Path,
) -> LoadedVerificationAggregate:
    """Strictly load a complete aggregate directory."""

    root = Path(input_dir)
    load_authenticated_run_directory(
        root,
        expected_layout_version=VERIFICATION_AGGREGATE_LAYOUT_VERSION,
        required_payloads=_PAYLOADS,
    )
    summary = strict_json_file(root / "summary.json", label="aggregate summary")
    manifest = strict_json_file(root / "manifest.json", label="aggregate manifest")
    expected_manifest_keys = {
        "layout_version",
        "schema_version",
        "aggregation_version",
        "experiment_id",
        "run_count",
        "seeds",
        "aggregate_digest",
        "source_runs",
        "files",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("aggregate manifest keys are invalid")
    if (
        manifest["layout_version"] != VERIFICATION_AGGREGATE_LAYOUT_VERSION
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["aggregation_version"] != VERIFICATION_AGGREGATION_VERSION
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("aggregation_version") != VERIFICATION_AGGREGATION_VERSION
    ):
        raise ValueError("aggregate schema or layout differs")
    digest = _digest(manifest["aggregate_digest"], name="aggregate digest")
    if digest != _aggregate_digest(summary):
        raise ValueError("aggregate summary digest differs")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != {
        "summary.json",
        "metrics_long.csv",
    }:
        raise ValueError("aggregate payload checksum map is invalid")
    for name, expected in files.items():
        if _digest(expected, name=f"aggregate {name} digest") != sha256_file(
            root / name
        ):
            raise ValueError(f"aggregate payload checksum differs: {name}")
    experiment_id = manifest["experiment_id"]
    seeds = manifest["seeds"]
    run_count = manifest["run_count"]
    if (
        not isinstance(experiment_id, str)
        or not experiment_id
        or not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or run_count != len(seeds)
        or summary.get("experiment_id") != experiment_id
        or summary.get("seeds") != seeds
        or summary.get("run_count") != run_count
        or manifest.get("source_runs") != summary.get("source_runs")
    ):
        raise ValueError("aggregate identity or counts differ")
    try:
        csv_rows = list(
            csv.DictReader(
                (root / "metrics_long.csv").read_text(
                    encoding="utf-8"
                ).splitlines()
            )
        )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"invalid aggregate metrics CSV: {exc}") from exc
    if not csv_rows:
        raise ValueError("aggregate metrics CSV must be non-empty")
    return LoadedVerificationAggregate(
        root=root,
        experiment_id=experiment_id,
        run_count=run_count,
        seeds=tuple(seeds),
        aggregate_digest=digest,
        summary=summary,
        manifest=manifest,
    )


def aggregate_verification_evaluations(
    evaluation_dirs: Sequence[str | Path],
    *,
    experiment_id: str,
    output_dir: str | Path,
) -> LoadedVerificationAggregate:
    """Aggregate authenticated evaluations with identical non-seed identity."""

    if isinstance(evaluation_dirs, (str, bytes)) or not isinstance(
        evaluation_dirs, Sequence
    ):
        raise TypeError("evaluation_dirs must be a sequence")
    roots = tuple(Path(value) for value in evaluation_dirs)
    if not roots:
        raise ValueError("evaluation_dirs must be non-empty")
    if len({root.resolve() for root in roots}) != len(roots):
        raise ValueError("evaluation_dirs contain duplicate paths")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be non-empty")
    loaded = tuple(_load_evaluation(root) for root in roots)
    evidence = tuple(sorted(loaded, key=lambda row: row.seed))
    _validate_identity(evidence)
    summary, long_rows = _summary_document(
        evidence,
        experiment_id=experiment_id,
    )
    summary_payload = canonical_json_bytes(summary)
    csv_payload = _csv_bytes(long_rows)
    aggregate_digest = _aggregate_digest(summary)
    manifest = {
        "layout_version": VERIFICATION_AGGREGATE_LAYOUT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "aggregation_version": VERIFICATION_AGGREGATION_VERSION,
        "experiment_id": experiment_id,
        "run_count": len(evidence),
        "seeds": [row.seed for row in evidence],
        "aggregate_digest": aggregate_digest,
        "source_runs": summary["source_runs"],
        "files": {
            "summary.json": hashlib.sha256(summary_payload).hexdigest(),
            "metrics_long.csv": hashlib.sha256(csv_payload).hexdigest(),
        },
    }
    destination = Path(output_dir)
    publish_authenticated_run_directory(
        destination,
        layout_version=VERIFICATION_AGGREGATE_LAYOUT_VERSION,
        payloads={
            "summary.json": summary_payload,
            "metrics_long.csv": csv_payload,
            "manifest.json": canonical_json_bytes(manifest),
        },
    )
    return load_verification_aggregate(destination)


__all__ = (
    "VERIFICATION_AGGREGATE_LAYOUT_VERSION",
    "VERIFICATION_AGGREGATION_VERSION",
    "LoadedVerificationAggregate",
    "aggregate_verification_evaluations",
    "load_verification_aggregate",
)
