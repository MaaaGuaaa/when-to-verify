"""Immutable SOP-16 experiment-result registry for reproducible evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import shutil
import tempfile
from typing import Any

from src.contracts import DYNAMIC_OBJECT_TYPES, SCHEMA_VERSION
from src.utils.atomic_publish import atomic_rename_noreplace


RESULT_REGISTRY_VERSION = "sop16_result_registry_v1"
RESULT_LAYOUT_VERSION = "sop16_result_layout_v1"
_RESULT_FILES = frozenset(
    {"manifest.json", "metrics.json", "episodes.json", "COMPLETE.json"}
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
        raise ValueError("value must be finite JSON") from exc
    return (encoded + "\n").encode("utf-8")


def _canonical_mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        parsed = json.loads(_json_bytes(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be finite canonical JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must encode to a JSON object")
    return parsed


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _optional_digest(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _digest(value, name=name)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return result


@dataclass(frozen=True)
class ResultProvenance:
    """Required evidence fields retained for every registered experiment run."""

    schema_version: str
    seed: int
    input_manifest_digest: str
    checkpoint_id: str
    scientific_status: str
    dynamic_objects_config_digest: str
    target_type_policy: str
    geometry_source: str
    geometry_fallback_fraction: float
    target_type_policy_digest: str | None = None
    risk_checkpoint_digest: str | None = None
    calibration_digest: str | None = None
    value_checkpoint_digest: str | None = None
    world_digest: str | None = None
    object_type_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(
            self,
            "input_manifest_digest",
            _digest(self.input_manifest_digest, name="input_manifest_digest"),
        )
        object.__setattr__(
            self,
            "dynamic_objects_config_digest",
            _digest(
                self.dynamic_objects_config_digest,
                name="dynamic_objects_config_digest",
            ),
        )
        for field in (
            "checkpoint_id",
            "scientific_status",
            "target_type_policy",
            "geometry_source",
        ):
            object.__setattr__(self, field, _string(getattr(self, field), name=field))
        object.__setattr__(
            self,
            "geometry_fallback_fraction",
            _finite_fraction(
                self.geometry_fallback_fraction,
                name="geometry_fallback_fraction",
            ),
        )
        for field_name in (
            "target_type_policy_digest",
            "risk_checkpoint_digest",
            "calibration_digest",
            "value_checkpoint_digest",
            "world_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_digest(getattr(self, field_name), name=field_name),
            )
        if not isinstance(self.object_type_counts, Mapping):
            raise TypeError("object_type_counts must be a mapping")
        counts: dict[str, int] = {}
        for object_type, raw_count in self.object_type_counts.items():
            if object_type not in DYNAMIC_OBJECT_TYPES:
                raise ValueError("object_type_counts contains an unsupported type")
            if (
                isinstance(raw_count, bool)
                or not isinstance(raw_count, Integral)
                or raw_count < 0
            ):
                raise ValueError("object_type_counts values must be non-negative integers")
            counts[str(object_type)] = int(raw_count)
        object.__setattr__(self, "object_type_counts", dict(sorted(counts.items())))


@dataclass(frozen=True)
class RegisteredResult:
    path: Path
    run_id: str
    config: dict[str, object]
    provenance: ResultProvenance
    manifest: dict[str, object]
    metrics: dict[str, float]
    episodes: tuple[dict[str, object], ...]


def _metrics(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("metrics must be a non-empty mapping")
    normalized: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metric names must be non-empty strings")
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise TypeError(f"metrics[{key!r}] must be a real number")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"metrics[{key!r}] must be finite")
        normalized[key] = number
    return dict(sorted(normalized.items()))


def _episodes(value: object) -> tuple[dict[str, object], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("episodes must be a sequence of JSON objects")
    rows: list[dict[str, object]] = []
    for index, row in enumerate(value):
        rows.append(_canonical_mapping(row, name=f"episodes[{index}]"))
    return tuple(rows)


def _write_json(path: Path, value: object) -> bytes:
    payload = _json_bytes(value)
    path.write_bytes(payload)
    return payload


def publish_result(
    output_dir: str | Path,
    *,
    run_id: str,
    config: Mapping[str, object],
    provenance: ResultProvenance,
    metrics: Mapping[str, Real],
    episodes: Sequence[Mapping[str, object]],
) -> Path:
    """Atomically publish one run, refusing an existing destination."""

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    run = _string(run_id, name="run_id")
    if not isinstance(provenance, ResultProvenance):
        raise TypeError("provenance must be a ResultProvenance")
    canonical_config = _canonical_mapping(config, name="config")
    config_seed = canonical_config.get("seed")
    if config_seed is not None and config_seed != provenance.seed:
        raise ValueError("config seed and provenance seed must match")
    normalized_metrics = _metrics(metrics)
    normalized_episodes = _episodes(episodes)
    config_payload = _json_bytes(canonical_config)
    metrics_payload = _json_bytes(normalized_metrics)
    episodes_payload = _json_bytes({"episodes": normalized_episodes})
    manifest = {
        "result_layout_version": RESULT_LAYOUT_VERSION,
        "result_registry_version": RESULT_REGISTRY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "run_id": run,
        "config": canonical_config,
        "config_digest_sha256": _sha256(config_payload),
        "provenance": asdict(provenance),
        "metrics_digest_sha256": _sha256(metrics_payload),
        "episodes_digest_sha256": _sha256(episodes_payload),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        manifest_payload = _write_json(staging / "manifest.json", manifest)
        (staging / "metrics.json").write_bytes(metrics_payload)
        (staging / "episodes.json").write_bytes(episodes_payload)
        _write_json(
            staging / "COMPLETE.json",
            {
                "result_layout_version": RESULT_LAYOUT_VERSION,
                "manifest_sha256": _sha256(manifest_payload),
            },
        )
        atomic_rename_noreplace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def _read_json(path: Path, *, name: str) -> object:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a real file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {name}: {exc}") from exc


def load_result(output_dir: str | Path) -> RegisteredResult:
    """Load and verify a complete immutable result directory."""

    root = Path(output_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("result output must be a real directory")
    names = {path.name for path in root.iterdir()}
    if names != _RESULT_FILES:
        raise ValueError("result output has an invalid or incomplete file layout")
    manifest = _canonical_mapping(
        _read_json(root / "manifest.json", name="manifest.json"), name="manifest"
    )
    expected_manifest_keys = {
        "result_layout_version",
        "result_registry_version",
        "schema_version",
        "run_id",
        "config",
        "config_digest_sha256",
        "provenance",
        "metrics_digest_sha256",
        "episodes_digest_sha256",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("result manifest keys are invalid")
    if manifest["result_layout_version"] != RESULT_LAYOUT_VERSION:
        raise ValueError("unsupported result layout version")
    if manifest["result_registry_version"] != RESULT_REGISTRY_VERSION:
        raise ValueError("unsupported result registry version")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("result schema version mismatch")
    run_id = _string(manifest["run_id"], name="manifest.run_id")
    config = _canonical_mapping(manifest["config"], name="manifest.config")
    if manifest["config_digest_sha256"] != _sha256(_json_bytes(config)):
        raise ValueError("config digest mismatch")
    provenance_raw = _canonical_mapping(manifest["provenance"], name="provenance")
    try:
        provenance = ResultProvenance(**provenance_raw)
    except TypeError as exc:
        raise ValueError("provenance fields are invalid") from exc
    if config.get("seed") is not None and config["seed"] != provenance.seed:
        raise ValueError("config and provenance seed mismatch")
    metrics = _metrics(_read_json(root / "metrics.json", name="metrics.json"))
    if manifest["metrics_digest_sha256"] != _sha256(_json_bytes(metrics)):
        raise ValueError("metrics digest mismatch")
    episodes_document = _canonical_mapping(
        _read_json(root / "episodes.json", name="episodes.json"), name="episodes"
    )
    if set(episodes_document) != {"episodes"}:
        raise ValueError("episodes document keys are invalid")
    episodes = _episodes(episodes_document["episodes"])
    if manifest["episodes_digest_sha256"] != _sha256(
        _json_bytes({"episodes": episodes})
    ):
        raise ValueError("episodes digest mismatch")
    complete = _canonical_mapping(
        _read_json(root / "COMPLETE.json", name="COMPLETE.json"), name="complete"
    )
    if complete != {
        "result_layout_version": RESULT_LAYOUT_VERSION,
        "manifest_sha256": _sha256(_json_bytes(manifest)),
    }:
        raise ValueError("completion marker does not authenticate the manifest")
    return RegisteredResult(
        path=root,
        run_id=run_id,
        config=config,
        provenance=provenance,
        manifest=manifest,
        metrics=metrics,
        episodes=episodes,
    )


def aggregate_seed_metrics(
    results: Sequence[RegisteredResult],
) -> dict[str, dict[str, object]]:
    """Aggregate exact raw seed values by registered decision strategy."""

    rows = tuple(results)
    if not rows or any(not isinstance(row, RegisteredResult) for row in rows):
        raise ValueError("results must be a non-empty sequence of RegisteredResult")
    grouped: dict[str, list[RegisteredResult]] = {}
    for row in rows:
        strategy = _string(row.config.get("strategy"), name="result config strategy")
        grouped.setdefault(strategy, []).append(row)
    aggregate: dict[str, dict[str, object]] = {}
    for strategy, group in sorted(grouped.items()):
        seen_seeds: set[int] = set()
        metric_names = set(group[0].metrics)
        expected_identity = _experiment_identity(group[0])
        for row in group:
            if row.provenance.seed in seen_seeds:
                raise ValueError("strategy group contains duplicate seed results")
            seen_seeds.add(row.provenance.seed)
            if _experiment_identity(row) != expected_identity:
                raise ValueError("strategy group mixes experiment identity")
            if set(row.metrics) != metric_names:
                raise ValueError("strategy group metrics do not have a common schema")
        metric_report: dict[str, object] = {}
        for metric_name in sorted(metric_names):
            seed_values = {
                str(row.provenance.seed): row.metrics[metric_name]
                for row in sorted(group, key=lambda item: item.provenance.seed)
            }
            values = tuple(seed_values.values())
            mean = math.fsum(values) / len(values)
            variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
            metric_report[metric_name] = {
                "mean": mean,
                "std": math.sqrt(variance),
                "seed_values": seed_values,
            }
        aggregate[strategy] = {
            "seed_count": len(group),
            "metrics": metric_report,
        }
    return aggregate


def _experiment_identity(result: RegisteredResult) -> str:
    config_without_seed = {
        key: value
        for key, value in result.config.items()
        if key not in {"seed", "suite_digest_sha256"}
    }
    provenance = result.provenance
    return _sha256(
        _json_bytes(
            {
                "schema_version": provenance.schema_version,
                "config": config_without_seed,
                "checkpoint_id": provenance.checkpoint_id,
                "scientific_status": provenance.scientific_status,
                "dynamic_objects_config_digest": provenance.dynamic_objects_config_digest,
                "target_type_policy": provenance.target_type_policy,
                "geometry_source": provenance.geometry_source,
                "geometry_fallback_fraction": provenance.geometry_fallback_fraction,
                "target_type_policy_digest": provenance.target_type_policy_digest,
                "risk_checkpoint_digest": provenance.risk_checkpoint_digest,
                "calibration_digest": provenance.calibration_digest,
                "value_checkpoint_digest": provenance.value_checkpoint_digest,
                "world_digest": provenance.world_digest,
                "object_type_counts": provenance.object_type_counts,
            }
        )
    )


__all__ = (
    "RESULT_LAYOUT_VERSION",
    "RESULT_REGISTRY_VERSION",
    "RegisteredResult",
    "ResultProvenance",
    "aggregate_seed_metrics",
    "load_result",
    "publish_result",
)
