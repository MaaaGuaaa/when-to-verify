"""Authenticated Schema-4 Long40 replay adapter for SOP-15 evaluation."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import shutil
import tempfile
from typing import Any

import yaml

from src.contracts import DYNAMIC_OBJECT_TYPES, SCHEMA_VERSION
from src.evaluation.closed_loop_runtime import (
    CLOSED_LOOP_RUNTIME_VERSION,
    ClosedLoopRuntimeConfig,
    DecisionFrame,
    EnvironmentTransition,
    run_closed_loop,
    summarize_closed_loop_episodes,
)
from src.evaluation.result_registry import ResultProvenance, publish_result
from src.planning.decision_policy import DECISION_STRATEGIES
from src.utils.atomic_publish import atomic_rename_noreplace


REPLAY_VERSION = "sop15_long40_replay_v2"
REPLAY_LAYOUT_VERSION = "sop15_replay_layout_v2"
LONG40_LAYOUT_VERSION = "history8_current7_future32_v1"
SCIENTIFIC_STATUSES = ("framework_fixture_only", "production_evaluation")
_REPLAY_FILES = frozenset({"manifest.json", "episodes.json", "COMPLETE.json"})
_RUNTIME_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "runtime_version",
        "future_horizon_s",
        "execute_step_s",
        "risk_weight",
        "verify_margin",
        "minimum_verify_duration_s",
        "maximum_verify_duration_s",
        "max_decisions",
    }
)
_VERIFY_STRATEGIES = frozenset(DECISION_STRATEGIES) - {"never"}
_EVIDENCE_ARTIFACT_FIELDS = frozenset(
    {
        "input_manifest",
        "risk_checkpoint",
        "calibration",
        "value_checkpoint",
        "world",
    }
)
_EXPERIMENT_BINDING_KEYS = frozenset(
    {"risk_method", "value_method", "parameters"}
)
_EPISODE_KEYS = frozenset(
    {
        "episode_id",
        "initial_state_id",
        "hazard_object_type",
        "dynamic_object_counts",
        "nominal_task_time_s",
        "nominal_path_length_m",
        "frames",
        "execute_transitions",
        "verify_transitions",
    }
)
_FRAME_KEYS = frozenset(
    {
        "plan_id",
        "task_cost",
        "calibrated_risk",
        "reject_cost",
        "action_values_by_strategy",
        "action_durations_s",
    }
)
_TRANSITION_REQUIRED_KEYS = frozenset({"duration_s", "path_length_m"})
_TRANSITION_OPTIONAL_KEYS = frozenset(
    {
        "next_state_id",
        "next_plan_id",
        "replanned",
        "collision",
        "near_miss",
        "task_complete",
        "critical_actor_revealed",
        "verification_useful",
        "termination_reason",
    }
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
        raise ValueError("replay value must be finite JSON") from exc
    return (encoded + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


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


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _offline_metrics(value: object) -> dict[str, float]:
    raw = _mapping(value, name="offline_metrics")
    metrics: dict[str, float] = {}
    for metric_name, metric_value in raw.items():
        name = _string(metric_name, name="offline metric name")
        metrics[name] = _finite(metric_value, name=f"offline_metrics[{name!r}]")
    return dict(sorted(metrics.items()))


def _experiment_binding(value: object) -> dict[str, object]:
    raw = _mapping(value, name="experiment_binding")
    if not raw:
        return {}
    _exact_keys(
        raw,
        _EXPERIMENT_BINDING_KEYS,
        name="experiment_binding",
    )
    parameters = _mapping(
        raw["parameters"],
        name="experiment_binding.parameters",
    )
    try:
        canonical = json.loads(
            _json_bytes(
                {
                    "risk_method": _string(
                        raw["risk_method"],
                        name="experiment_binding.risk_method",
                    ),
                    "value_method": _string(
                        raw["value_method"],
                        name="experiment_binding.value_method",
                    ),
                    "parameters": parameters,
                }
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError("experiment_binding must be canonical JSON") from exc
    if not isinstance(canonical, dict) or canonical["parameters"] != parameters:
        raise ValueError("experiment_binding parameters are not canonical")
    return canonical


def _typed_object_counts(value: object, *, name: str) -> dict[str, int]:
    raw_counts = _mapping(value, name=name)
    if set(raw_counts) != set(DYNAMIC_OBJECT_TYPES):
        raise ValueError(f"{name} must exactly cover typed dynamic objects")
    counts: dict[str, int] = {}
    for object_type, raw_count in raw_counts.items():
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, Integral)
            or raw_count < 0
        ):
            raise ValueError(f"{name} values must be non-negative integers")
        counts[object_type] = int(raw_count)
    if sum(counts.values()) <= 0:
        raise ValueError(f"{name} must contain at least one object")
    return dict(sorted(counts.items()))


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys are invalid")


def _write_json(path: Path, value: object) -> bytes:
    payload = _json_bytes(value)
    path.write_bytes(payload)
    return payload


def _read_json_file(path: Path, *, name: str) -> tuple[object, bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a real file")
    try:
        payload = path.read_bytes()
        return json.loads(payload), payload
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {name}: {exc}") from exc


def load_runtime_config(path: str | Path) -> ClosedLoopRuntimeConfig:
    """Strictly load the production SOP-15 runtime configuration."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("runtime config must be a real file")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid runtime config: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("runtime config must contain a mapping")
    _exact_keys(raw, _RUNTIME_CONFIG_KEYS, name="runtime config")
    if raw.pop("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"runtime config schema must be {SCHEMA_VERSION}")
    if raw.pop("runtime_version") != CLOSED_LOOP_RUNTIME_VERSION:
        raise ValueError("unsupported closed-loop runtime version")
    return ClosedLoopRuntimeConfig(**raw)


@dataclass(frozen=True)
class EvidenceArtifact:
    """Identity and digest of one upstream Schema-4 Long40 artifact."""

    artifact_id: str
    sha256: str
    schema_version: str
    long40_layout_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_id", _string(self.artifact_id, name="artifact_id")
        )
        object.__setattr__(self, "sha256", _digest(self.sha256, name="sha256"))
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"artifact schema_version must be {SCHEMA_VERSION}")
        if self.long40_layout_version != LONG40_LAYOUT_VERSION:
            raise ValueError(
                f"artifact must use the Long40 layout {LONG40_LAYOUT_VERSION}"
            )


@dataclass(frozen=True)
class ReplayEvidence:
    """Complete evidence chain shared by every episode in a replay suite."""

    schema_version: str
    seed: int
    split: str
    scientific_status: str
    input_manifest: EvidenceArtifact
    risk_checkpoint: EvidenceArtifact
    calibration: EvidenceArtifact
    value_checkpoint: EvidenceArtifact
    world: EvidenceArtifact
    dynamic_objects_config_digest: str
    target_type_policy: str
    target_type_policy_digest: str
    object_type_counts: Mapping[str, int]
    geometry_source: str
    geometry_fallback_fraction: float
    experiment_binding: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "split", _string(self.split, name="split"))
        if self.scientific_status not in SCIENTIFIC_STATUSES:
            raise ValueError("scientific_status is unsupported")
        for field_name in (
            "input_manifest",
            "risk_checkpoint",
            "calibration",
            "value_checkpoint",
            "world",
        ):
            artifact = getattr(self, field_name)
            if not isinstance(artifact, EvidenceArtifact):
                raise TypeError(f"{field_name} must be an EvidenceArtifact")
        object.__setattr__(
            self,
            "dynamic_objects_config_digest",
            _digest(
                self.dynamic_objects_config_digest,
                name="dynamic_objects_config_digest",
            ),
        )
        object.__setattr__(
            self,
            "target_type_policy",
            _string(self.target_type_policy, name="target_type_policy"),
        )
        object.__setattr__(
            self,
            "target_type_policy_digest",
            _digest(
                self.target_type_policy_digest,
                name="target_type_policy_digest",
            ),
        )
        object.__setattr__(
            self,
            "object_type_counts",
            _typed_object_counts(
                self.object_type_counts,
                name="object_type_counts",
            ),
        )
        object.__setattr__(
            self,
            "geometry_source",
            _string(self.geometry_source, name="geometry_source"),
        )
        object.__setattr__(
            self,
            "geometry_fallback_fraction",
            _finite(
                self.geometry_fallback_fraction,
                name="geometry_fallback_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "experiment_binding",
            _experiment_binding(self.experiment_binding),
        )


def _artifact_payload(artifact: EvidenceArtifact) -> dict[str, object]:
    return {
        "artifact_id": artifact.artifact_id,
        "sha256": artifact.sha256,
        "schema_version": artifact.schema_version,
        "long40_layout_version": artifact.long40_layout_version,
    }


def _artifact_from_payload(value: object, *, name: str) -> EvidenceArtifact:
    raw = _mapping(value, name=name)
    _exact_keys(
        raw,
        frozenset(
            {
                "artifact_id",
                "sha256",
                "schema_version",
                "long40_layout_version",
            }
        ),
        name=name,
    )
    return EvidenceArtifact(**raw)


def _evidence_payload(evidence: ReplayEvidence) -> dict[str, object]:
    return {
        "schema_version": evidence.schema_version,
        "seed": evidence.seed,
        "split": evidence.split,
        "scientific_status": evidence.scientific_status,
        "input_manifest": _artifact_payload(evidence.input_manifest),
        "risk_checkpoint": _artifact_payload(evidence.risk_checkpoint),
        "calibration": _artifact_payload(evidence.calibration),
        "value_checkpoint": _artifact_payload(evidence.value_checkpoint),
        "world": _artifact_payload(evidence.world),
        "dynamic_objects_config_digest": evidence.dynamic_objects_config_digest,
        "target_type_policy": evidence.target_type_policy,
        "target_type_policy_digest": evidence.target_type_policy_digest,
        "object_type_counts": dict(evidence.object_type_counts),
        "geometry_source": evidence.geometry_source,
        "geometry_fallback_fraction": evidence.geometry_fallback_fraction,
        "experiment_binding": dict(evidence.experiment_binding),
    }


def _evidence_from_payload(value: object) -> ReplayEvidence:
    raw = _mapping(value, name="evidence")
    expected = frozenset(
        {
            "schema_version",
            "seed",
            "split",
            "scientific_status",
            "input_manifest",
            "risk_checkpoint",
            "calibration",
            "value_checkpoint",
            "world",
            "dynamic_objects_config_digest",
            "target_type_policy",
            "target_type_policy_digest",
            "object_type_counts",
            "geometry_source",
            "geometry_fallback_fraction",
            "experiment_binding",
        }
    )
    _exact_keys(raw, expected, name="evidence")
    for field_name in (
        "input_manifest",
        "risk_checkpoint",
        "calibration",
        "value_checkpoint",
        "world",
    ):
        raw[field_name] = _artifact_from_payload(raw[field_name], name=field_name)
    return ReplayEvidence(**raw)


@dataclass(frozen=True)
class ReplayEnvironment:
    """A validated deterministic replay graph implementing the runtime protocol."""

    episode_id: str
    initial_state_id: str
    hazard_object_type: str
    dynamic_object_counts: Mapping[str, int]
    nominal_task_time_s: float
    nominal_path_length_m: float
    frames: Mapping[str, DecisionFrame]
    execute_edges: Mapping[str, EnvironmentTransition]
    verify_edges: Mapping[tuple[str, str], EnvironmentTransition]

    def decision_frame(self, state_id: str) -> DecisionFrame:
        try:
            return self.frames[state_id]
        except KeyError as exc:
            raise ValueError(f"unknown replay state: {state_id!r}") from exc

    def execute(self, state_id: str, duration_s: float) -> EnvironmentTransition:
        try:
            transition = self.execute_edges[state_id]
        except KeyError as exc:
            raise ValueError(f"missing execute edge for state: {state_id!r}") from exc
        if not math.isclose(
            transition.duration_s, duration_s, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("runtime execute duration does not match replay edge")
        return transition

    def verify(self, state_id: str, action_id: str) -> EnvironmentTransition:
        try:
            return self.verify_edges[(state_id, action_id)]
        except KeyError as exc:
            raise ValueError(
                f"missing verify edge for state/action: {state_id!r}/{action_id!r}"
            ) from exc


def _frame_from_payload(state_id: str, value: object) -> DecisionFrame:
    raw = _mapping(value, name=f"frame {state_id!r}")
    _exact_keys(raw, _FRAME_KEYS, name=f"frame {state_id!r}")
    frame = DecisionFrame(state_id=state_id, **raw)
    actions = set(frame.action_durations_s)
    if actions:
        if set(frame.action_values_by_strategy) != _VERIFY_STRATEGIES:
            raise ValueError(
                "frames with verification actions must score all six verify strategies"
            )
        if any(
            set(values) != actions
            for values in frame.action_values_by_strategy.values()
        ):
            raise ValueError(
                "each verification strategy must score exactly the declared actions"
            )
    elif frame.action_values_by_strategy:
        raise ValueError("frames without actions cannot contain action values")
    return frame


def _transition_from_payload(
    value: object,
    *,
    kind: str,
    frame: DecisionFrame,
    action_id: str | None,
) -> EnvironmentTransition:
    raw = _mapping(value, name=f"{kind} transition")
    keys = set(raw)
    allowed = _TRANSITION_REQUIRED_KEYS | _TRANSITION_OPTIONAL_KEYS
    if not _TRANSITION_REQUIRED_KEYS <= keys or not keys <= allowed:
        raise ValueError(f"{kind} transition keys are invalid")
    if kind == "execute" and "verification_useful" in raw:
        raise ValueError("execute transitions cannot contain verification_useful")
    if kind == "verify" and "verification_useful" not in raw:
        raise ValueError("verify transitions require verification_useful")
    return EnvironmentTransition(
        kind=kind,
        source_state_id=frame.state_id,
        source_plan_id=frame.plan_id,
        action_id=action_id,
        duration_s=raw["duration_s"],
        path_length_m=raw["path_length_m"],
        next_state_id=raw.get("next_state_id"),
        next_plan_id=raw.get("next_plan_id"),
        replanned=raw.get("replanned", False),
        collision=raw.get("collision", False),
        near_miss=raw.get("near_miss", False),
        task_complete=raw.get("task_complete", False),
        critical_actor_revealed=raw.get("critical_actor_revealed", False),
        verification_useful=raw.get("verification_useful"),
        termination_reason=raw.get("termination_reason"),
    )


def _validate_edge_target(
    transition: EnvironmentTransition,
    *,
    frames: Mapping[str, DecisionFrame],
) -> None:
    terminal = (
        transition.collision
        or transition.task_complete
        or transition.termination_reason is not None
    )
    if terminal:
        if (
            transition.replanned
            or transition.next_state_id is not None
            or transition.next_plan_id is not None
        ):
            raise ValueError("terminal replay transition cannot carry a next plan")
        return
    if (
        not transition.replanned
        or transition.next_state_id is None
        or transition.next_plan_id is None
    ):
        raise ValueError("non-terminal replay transition must provide a replan")
    if transition.next_state_id not in frames:
        raise ValueError("replay transition next state is missing")
    if transition.next_state_id == transition.source_state_id:
        raise ValueError("replay transition must advance to a new state")
    expected_plan = frames[transition.next_state_id].plan_id
    if transition.next_plan_id != expected_plan:
        raise ValueError("replay transition next plan does not match its next frame")
    if transition.next_plan_id == transition.source_plan_id:
        raise ValueError("replay transition must provide a new next plan")


def _environment_from_payload(value: object) -> ReplayEnvironment:
    raw = _mapping(value, name="episode")
    _exact_keys(raw, _EPISODE_KEYS, name="episode")
    episode_id = _string(raw["episode_id"], name="episode_id")
    initial_state_id = _string(raw["initial_state_id"], name="initial_state_id")
    hazard_type = raw["hazard_object_type"]
    if hazard_type not in DYNAMIC_OBJECT_TYPES:
        raise ValueError("episode hazard_object_type is unsupported")
    dynamic_object_counts = _typed_object_counts(
        raw["dynamic_object_counts"],
        name="episode dynamic_object_counts",
    )
    if dynamic_object_counts[str(hazard_type)] <= 0:
        raise ValueError("episode hazard type must have a positive object count")
    raw_frames = _mapping(raw["frames"], name="frames")
    if not raw_frames:
        raise ValueError("episode frames must not be empty")
    frames = {
        _string(state_id, name="state_id"): _frame_from_payload(state_id, frame)
        for state_id, frame in raw_frames.items()
    }
    if initial_state_id not in frames:
        raise ValueError("initial_state_id is missing from frames")

    raw_execute = _mapping(raw["execute_transitions"], name="execute_transitions")
    if set(raw_execute) != set(frames):
        raise ValueError("execute transitions must exactly cover all frames")
    execute_edges = {
        state_id: _transition_from_payload(
            raw_execute[state_id],
            kind="execute",
            frame=frames[state_id],
            action_id=None,
        )
        for state_id in frames
    }

    raw_verify = raw["verify_transitions"]
    if isinstance(raw_verify, (str, bytes)) or not isinstance(raw_verify, Sequence):
        raise TypeError("verify_transitions must be a sequence")
    verify_edges: dict[tuple[str, str], EnvironmentTransition] = {}
    verify_entry_keys = frozenset({"state_id", "action_id"}) | (
        _TRANSITION_REQUIRED_KEYS | _TRANSITION_OPTIONAL_KEYS
    )
    for index, item in enumerate(raw_verify):
        entry = _mapping(item, name=f"verify_transitions[{index}]")
        if set(entry) != verify_entry_keys - {
            "next_state_id",
            "next_plan_id",
            "replanned",
            "collision",
            "near_miss",
            "task_complete",
            "critical_actor_revealed",
            "termination_reason",
        } and (
            not {"state_id", "action_id"} | _TRANSITION_REQUIRED_KEYS
            <= set(entry)
            or not set(entry) <= verify_entry_keys
        ):
            raise ValueError("verify transition keys are invalid")
        state_id = _string(entry.pop("state_id"), name="verify state_id")
        action_id = _string(entry.pop("action_id"), name="verify action_id")
        if state_id not in frames:
            raise ValueError("verify transition references an unknown state")
        if action_id not in frames[state_id].action_durations_s:
            raise ValueError("verify transition references an unknown action")
        key = (state_id, action_id)
        if key in verify_edges:
            raise ValueError("duplicate verify transition")
        transition = _transition_from_payload(
            entry,
            kind="verify",
            frame=frames[state_id],
            action_id=action_id,
        )
        expected_duration = frames[state_id].action_durations_s[action_id]
        if not math.isclose(
            transition.duration_s,
            expected_duration,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("verify transition duration does not match its action")
        verify_edges[key] = transition
    expected_verify = {
        (state_id, action_id)
        for state_id, frame in frames.items()
        for action_id in frame.action_durations_s
    }
    if set(verify_edges) != expected_verify:
        raise ValueError("verify transitions must exactly cover declared actions")

    for transition in (*execute_edges.values(), *verify_edges.values()):
        _validate_edge_target(transition, frames=frames)
    reachable = {initial_state_id}
    queue = deque([initial_state_id])
    all_edges = {
        state_id: (
            execute_edges[state_id],
            *(
                verify_edges[(state_id, action_id)]
                for action_id in frames[state_id].action_durations_s
            ),
        )
        for state_id in frames
    }
    while queue:
        state_id = queue.popleft()
        for transition in all_edges[state_id]:
            next_state = transition.next_state_id
            if next_state is not None and next_state not in reachable:
                reachable.add(next_state)
                queue.append(next_state)
    if reachable != set(frames):
        raise ValueError("episode contains unreachable replay frames")
    return ReplayEnvironment(
        episode_id=episode_id,
        initial_state_id=initial_state_id,
        hazard_object_type=str(hazard_type),
        dynamic_object_counts=dynamic_object_counts,
        nominal_task_time_s=_finite(
            raw["nominal_task_time_s"],
            name="nominal_task_time_s",
            minimum=0.0,
        ),
        nominal_path_length_m=_finite(
            raw["nominal_path_length_m"],
            name="nominal_path_length_m",
            minimum=0.0,
        ),
        frames=dict(sorted(frames.items())),
        execute_edges=dict(sorted(execute_edges.items())),
        verify_edges=dict(sorted(verify_edges.items())),
    )


def _episodes_document(
    episodes: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], tuple[ReplayEnvironment, ...]]:
    if isinstance(episodes, (str, bytes)) or not isinstance(episodes, Sequence):
        raise TypeError("episodes must be a sequence")
    try:
        canonical = json.loads(_json_bytes({"episodes": list(episodes)}))
    except json.JSONDecodeError as exc:
        raise ValueError("episodes must be canonical JSON") from exc
    raw_rows = canonical.get("episodes")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("episodes must be a non-empty sequence")
    environments = tuple(_environment_from_payload(row) for row in raw_rows)
    episode_ids = [environment.episode_id for environment in environments]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("replay episode IDs must be unique")
    return canonical, environments


@dataclass(frozen=True)
class LoadedReplaySuite:
    path: Path
    evidence: ReplayEvidence
    environments: tuple[ReplayEnvironment, ...]
    offline_metrics: Mapping[str, float]
    upstream_files_verified: bool
    suite_digest_sha256: str
    manifest: dict[str, object]


def publish_replay_suite(
    output_dir: str | Path,
    *,
    evidence: ReplayEvidence,
    episodes: Sequence[Mapping[str, object]],
    offline_metrics: Mapping[str, Real] | None = None,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> Path:
    """Validate and atomically publish one immutable replay suite."""

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    if not isinstance(evidence, ReplayEvidence):
        raise TypeError("evidence must be ReplayEvidence")
    if (
        evidence.scientific_status == "production_evaluation"
        and artifact_paths is None
    ):
        raise ValueError(
            "production replay publication requires verified artifact_paths"
        )
    upstream_files_verified = artifact_paths is not None
    if artifact_paths is not None:
        raw_paths = _mapping(artifact_paths, name="artifact_paths")
        if set(raw_paths) != _EVIDENCE_ARTIFACT_FIELDS:
            raise ValueError(
                "artifact_paths must exactly cover the five evidence artifacts"
            )
        for field_name, raw_path in raw_paths.items():
            path = Path(raw_path)
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    f"artifact_paths[{field_name!r}] must be a real file"
                )
            artifact = getattr(evidence, field_name)
            if _sha256_file(path) != artifact.sha256:
                raise ValueError(
                    f"artifact_paths[{field_name!r}] digest mismatch"
                )
    episodes_document, environments = _episodes_document(episodes)
    normalized_offline_metrics = _offline_metrics(
        {} if offline_metrics is None else offline_metrics
    )
    object_type_counts = {
        object_type: sum(
            environment.dynamic_object_counts[object_type]
            for environment in environments
        )
        for object_type in DYNAMIC_OBJECT_TYPES
    }
    if object_type_counts != dict(evidence.object_type_counts):
        raise ValueError(
            "evidence object_type_counts do not match replay episode hazards"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        episodes_payload = _write_json(staging / "episodes.json", episodes_document)
        manifest = {
            "replay_layout_version": REPLAY_LAYOUT_VERSION,
            "replay_version": REPLAY_VERSION,
            "schema_version": SCHEMA_VERSION,
            "scientific_status": evidence.scientific_status,
            "evidence": _evidence_payload(evidence),
            "offline_metrics": normalized_offline_metrics,
            "upstream_files_verified": upstream_files_verified,
            "episode_count": len(environments),
            "episodes_digest_sha256": _sha256(episodes_payload),
        }
        manifest_payload = _write_json(staging / "manifest.json", manifest)
        _write_json(
            staging / "COMPLETE.json",
            {
                "replay_layout_version": REPLAY_LAYOUT_VERSION,
                "manifest_sha256": _sha256(manifest_payload),
            },
        )
        atomic_rename_noreplace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def load_replay_suite(path: str | Path) -> LoadedReplaySuite:
    """Load a complete replay suite and verify all internal digests."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("replay suite must be a real directory")
    if {item.name for item in root.iterdir()} != _REPLAY_FILES:
        raise ValueError("replay suite has an invalid or incomplete layout")
    manifest_value, manifest_payload = _read_json_file(
        root / "manifest.json", name="manifest.json"
    )
    manifest = _mapping(manifest_value, name="manifest")
    expected_manifest_keys = frozenset(
        {
            "replay_layout_version",
            "replay_version",
            "schema_version",
            "scientific_status",
            "evidence",
            "offline_metrics",
            "upstream_files_verified",
            "episode_count",
            "episodes_digest_sha256",
        }
    )
    _exact_keys(manifest, expected_manifest_keys, name="manifest")
    if manifest["replay_layout_version"] != REPLAY_LAYOUT_VERSION:
        raise ValueError("unsupported replay layout version")
    if manifest["replay_version"] != REPLAY_VERSION:
        raise ValueError("unsupported replay version")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("replay schema version mismatch")
    evidence = _evidence_from_payload(manifest["evidence"])
    normalized_offline_metrics = _offline_metrics(manifest["offline_metrics"])
    upstream_files_verified = manifest["upstream_files_verified"]
    if not isinstance(upstream_files_verified, bool):
        raise ValueError("upstream_files_verified must be a bool")
    if (
        evidence.scientific_status == "production_evaluation"
        and not upstream_files_verified
    ):
        raise ValueError("production replay lacks upstream file verification")
    if manifest["scientific_status"] != evidence.scientific_status:
        raise ValueError("replay scientific status mismatch")
    episodes_value, episodes_payload = _read_json_file(
        root / "episodes.json", name="episodes.json"
    )
    if manifest["episodes_digest_sha256"] != _sha256(episodes_payload):
        raise ValueError("episodes digest mismatch")
    episodes_document = _mapping(episodes_value, name="episodes document")
    if set(episodes_document) != {"episodes"}:
        raise ValueError("episodes document keys are invalid")
    _, environments = _episodes_document(episodes_document["episodes"])
    if manifest["episode_count"] != len(environments):
        raise ValueError("replay episode count mismatch")
    observed_counts = {
        object_type: sum(
            environment.dynamic_object_counts[object_type]
            for environment in environments
        )
        for object_type in DYNAMIC_OBJECT_TYPES
    }
    if observed_counts != dict(evidence.object_type_counts):
        raise ValueError("replay object type counts mismatch")
    complete_value, _ = _read_json_file(root / "COMPLETE.json", name="COMPLETE.json")
    complete = _mapping(complete_value, name="complete")
    if complete != {
        "replay_layout_version": REPLAY_LAYOUT_VERSION,
        "manifest_sha256": _sha256(manifest_payload),
    }:
        raise ValueError("completion marker does not authenticate the manifest")
    return LoadedReplaySuite(
        path=root,
        evidence=evidence,
        environments=environments,
        offline_metrics=normalized_offline_metrics,
        upstream_files_verified=upstream_files_verified,
        suite_digest_sha256=_sha256(manifest_payload),
        manifest=manifest,
    )


def _runtime_payload(config: ClosedLoopRuntimeConfig) -> dict[str, object]:
    return {
        "runtime_version": CLOSED_LOOP_RUNTIME_VERSION,
        "future_horizon_s": config.future_horizon_s,
        "execute_step_s": config.execute_step_s,
        "risk_weight": config.risk_weight,
        "verify_margin": config.verify_margin,
        "minimum_verify_duration_s": config.minimum_verify_duration_s,
        "maximum_verify_duration_s": config.maximum_verify_duration_s,
        "max_decisions": config.max_decisions,
    }


def run_replay_evaluation(
    *,
    suite_path: str | Path,
    strategy: str,
    config: ClosedLoopRuntimeConfig,
    output_dir: str | Path,
) -> Path:
    """Run all authenticated replay episodes and publish a registered result."""

    suite = load_replay_suite(suite_path)
    return run_loaded_replay_evaluation(
        suite=suite,
        strategy=strategy,
        config=config,
        output_dir=output_dir,
    )


def run_loaded_replay_evaluation(
    *,
    suite: LoadedReplaySuite,
    strategy: str,
    config: ClosedLoopRuntimeConfig,
    output_dir: str | Path,
    run_id: str | None = None,
    experiment: Mapping[str, object] | None = None,
) -> Path:
    """Evaluate an already authenticated suite without reparsing it per strategy."""

    if strategy not in DECISION_STRATEGIES:
        raise ValueError(f"unsupported decision strategy: {strategy!r}")
    if not isinstance(config, ClosedLoopRuntimeConfig):
        raise TypeError("config must be ClosedLoopRuntimeConfig")
    if not isinstance(suite, LoadedReplaySuite):
        raise TypeError("suite must be LoadedReplaySuite")
    traces = tuple(
        run_closed_loop(environment, strategy=strategy, config=config)
        for environment in suite.environments
    )
    evidence = suite.evidence
    metrics = summarize_closed_loop_episodes(traces)
    if strategy == "learned":
        overlap = set(metrics) & set(suite.offline_metrics)
        if overlap:
            raise ValueError(
                "offline metrics overlap closed-loop metrics: "
                + ", ".join(sorted(overlap))
            )
        metrics.update(suite.offline_metrics)
    result_config: dict[str, object] = {
        "evaluation_mode": "authenticated_replay",
        "strategy": strategy,
        "seed": evidence.seed,
        "split": evidence.split,
        "suite_digest_sha256": suite.suite_digest_sha256,
        "upstream_files_verified": suite.upstream_files_verified,
        "runtime": _runtime_payload(config),
        "risk_checkpoint_id": evidence.risk_checkpoint.artifact_id,
        "calibration_id": evidence.calibration.artifact_id,
        "value_checkpoint_id": evidence.value_checkpoint.artifact_id,
        "world_id": evidence.world.artifact_id,
        "experiment_binding": dict(evidence.experiment_binding),
    }
    if experiment is not None:
        canonical_experiment = json.loads(_json_bytes(dict(experiment)))
        if not isinstance(canonical_experiment, dict):
            raise ValueError("experiment must encode to a JSON object")
        result_config["experiment"] = canonical_experiment
    return publish_result(
        output_dir,
        run_id=(
            f"{strategy}-seed-{evidence.seed}"
            if run_id is None
            else _string(run_id, name="run_id")
        ),
        config=result_config,
        provenance=ResultProvenance(
            schema_version=SCHEMA_VERSION,
            seed=evidence.seed,
            input_manifest_digest=evidence.input_manifest.sha256,
            checkpoint_id=(
                f"{evidence.risk_checkpoint.artifact_id}"
                f"+{evidence.value_checkpoint.artifact_id}"
            ),
            scientific_status=evidence.scientific_status,
            dynamic_objects_config_digest=evidence.dynamic_objects_config_digest,
            target_type_policy=evidence.target_type_policy,
            geometry_source=evidence.geometry_source,
            geometry_fallback_fraction=evidence.geometry_fallback_fraction,
            target_type_policy_digest=evidence.target_type_policy_digest,
            risk_checkpoint_digest=evidence.risk_checkpoint.sha256,
            calibration_digest=evidence.calibration.sha256,
            value_checkpoint_digest=evidence.value_checkpoint.sha256,
            world_digest=evidence.world.sha256,
            object_type_counts=evidence.object_type_counts,
        ),
        metrics=metrics,
        episodes=[trace.as_dict() for trace in traces],
    )


__all__ = (
    "LONG40_LAYOUT_VERSION",
    "REPLAY_LAYOUT_VERSION",
    "REPLAY_VERSION",
    "EvidenceArtifact",
    "LoadedReplaySuite",
    "ReplayEnvironment",
    "ReplayEvidence",
    "load_replay_suite",
    "load_runtime_config",
    "publish_replay_suite",
    "run_loaded_replay_evaluation",
    "run_replay_evaluation",
)
