"""Targeted regime-A supplement selection and immutable publication."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
from pathlib import Path
from types import MappingProxyType

import numpy as np
import yaml

from .sop05_final_scenarios import (
    IncompleteContextError,
    Sop05FinalScenarioPublishResult,
    Sop05FinalScenarioSelection,
    _unseen_mother,
    _unseen_mother_from_state,
    publish_selected_sop05_final_scenarios,
)
from .sop05_unseen_prior import (
    evaluate_candidate,
    prepare_unseen_candidate_context,
    transform_long40_target,
)
from .sop05r_teb_output_loader import LoadedSop05rTebOutput


SOP05_A_SUPPLEMENT_VERSION = "sop05_a_supplement_v1"
SOP05_A_SUPPLEMENT_SAMPLING_ORIGIN = "targeted_a_supplement"
SOP05_A_SUPPLEMENT_ANGLE_DISTRIBUTION = "uniform_half_open_pi_v1"
_SPLITS = ("train", "calibration", "val", "test")
_EXPECTED_QUOTAS = {
    "train": (16531, 2859),
    "calibration": (2221, 383),
    "val": (2049, 382),
    "test": (2073, 384),
}
_EXPECTED_SOURCE_MOTHER_QUOTAS = {
    "train": 23000,
    "calibration": 3000,
    "val": 3000,
    "test": 3000,
}


class Sop05ASupplementError(ValueError):
    """Raised when a targeted A supplement cannot satisfy its frozen contract."""


@dataclass(frozen=True)
class Sop05ASupplementQuota:
    accepted: int
    present: int

    def __post_init__(self) -> None:
        for name, value in (("accepted", self.accepted), ("present", self.present)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Sop05ASupplementError(f"quota {name} must be a nonnegative integer")
        if self.present > self.accepted:
            raise Sop05ASupplementError("present quota cannot exceed accepted quota")

    @property
    def empty(self) -> int:
        return self.accepted - self.present


@dataclass(frozen=True)
class Sop05ASupplementConfig:
    version: str
    sampling_origin: str
    present_angle_distribution: str
    present_max_attempts_per_mother: int
    final_sampling_seed: int
    source_generation_seeds: Mapping[str, int]
    source_mother_quotas: Mapping[str, int]
    quotas: Mapping[str, Sop05ASupplementQuota]
    digest: str


@dataclass(frozen=True)
class _PresentEvaluation:
    mother_id: str
    history_poses: np.ndarray | None
    future_poses: np.ndarray | None
    attempted_angle_count: int
    selected_angle_rad: float | None
    rejection_reason_counts: Mapping[str, int]

    @property
    def accepted(self) -> bool:
        return self.history_poses is not None and self.future_poses is not None


@dataclass(frozen=True)
class Sop05ASupplementPublishResult:
    publication: Sop05FinalScenarioPublishResult
    present_count: int
    empty_count: int
    config_digest: str


@dataclass(frozen=True)
class _PresentWorkerContext:
    source: LoadedSop05rTebOutput
    split: str
    config: Sop05ASupplementConfig
    states_by_event: Mapping[str, object]


_PRESENT_WORKER_CONTEXT: _PresentWorkerContext | None = None


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise Sop05ASupplementError(
            "A-supplement config and provenance must be canonical JSON"
        ) from exc


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Sop05ASupplementError(f"{name} must be a nonnegative integer")
    return value


def normalize_sop05_a_supplement_config(
    raw: Mapping[str, object],
) -> Sop05ASupplementConfig:
    """Validate the exact targeted-supplement configuration."""

    expected_keys = {
        "version",
        "sampling_origin",
        "present_angle_distribution",
        "present_max_attempts_per_mother",
        "final_sampling_seed",
        "source_generation_seeds",
        "source_mother_quotas",
        "quotas",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise Sop05ASupplementError("A-supplement config keys mismatch")
    if raw["version"] != SOP05_A_SUPPLEMENT_VERSION:
        raise Sop05ASupplementError("A-supplement version mismatch")
    if raw["sampling_origin"] != SOP05_A_SUPPLEMENT_SAMPLING_ORIGIN:
        raise Sop05ASupplementError("A-supplement sampling origin mismatch")
    if raw["present_angle_distribution"] != SOP05_A_SUPPLEMENT_ANGLE_DISTRIBUTION:
        raise Sop05ASupplementError("A-supplement angle distribution mismatch")
    attempts = _nonnegative_int(
        raw["present_max_attempts_per_mother"],
        name="present_max_attempts_per_mother",
    )
    if attempts != 256:
        raise Sop05ASupplementError("present_max_attempts_per_mother must equal 256")
    final_seed = _nonnegative_int(raw["final_sampling_seed"], name="final_sampling_seed")
    raw_seeds = raw["source_generation_seeds"]
    if not isinstance(raw_seeds, Mapping) or set(raw_seeds) != set(_SPLITS):
        raise Sop05ASupplementError("source generation seeds must cover all splits")
    seeds = {
        split: _nonnegative_int(raw_seeds[split], name=f"source_generation_seeds.{split}")
        for split in _SPLITS
    }
    if len(set(seeds.values())) != len(seeds):
        raise Sop05ASupplementError("source generation seeds must be unique")
    raw_source_quotas = raw["source_mother_quotas"]
    if not isinstance(raw_source_quotas, Mapping) or set(raw_source_quotas) != set(
        _SPLITS
    ):
        raise Sop05ASupplementError("source mother quotas must cover all splits")
    source_quotas = {
        split: _nonnegative_int(
            raw_source_quotas[split], name=f"source_mother_quotas.{split}"
        )
        for split in _SPLITS
    }
    if source_quotas != _EXPECTED_SOURCE_MOTHER_QUOTAS:
        raise Sop05ASupplementError("source mother quota values differ")
    raw_quotas = raw["quotas"]
    if not isinstance(raw_quotas, Mapping) or set(raw_quotas) != set(_SPLITS):
        raise Sop05ASupplementError("A-supplement quotas must cover all splits")
    quotas: dict[str, Sop05ASupplementQuota] = {}
    for split in _SPLITS:
        node = raw_quotas[split]
        if not isinstance(node, Mapping) or set(node) != {"accepted", "present"}:
            raise Sop05ASupplementError(f"quota schema mismatch for split {split}")
        quota = Sop05ASupplementQuota(
            accepted=_nonnegative_int(node["accepted"], name=f"quotas.{split}.accepted"),
            present=_nonnegative_int(node["present"], name=f"quotas.{split}.present"),
        )
        if (quota.accepted, quota.present) != _EXPECTED_QUOTAS[split]:
            raise Sop05ASupplementError(f"quota values differ for split {split}")
        quotas[split] = quota
    normalized = {
        "version": SOP05_A_SUPPLEMENT_VERSION,
        "sampling_origin": SOP05_A_SUPPLEMENT_SAMPLING_ORIGIN,
        "present_angle_distribution": SOP05_A_SUPPLEMENT_ANGLE_DISTRIBUTION,
        "present_max_attempts_per_mother": attempts,
        "final_sampling_seed": final_seed,
        "source_generation_seeds": seeds,
        "source_mother_quotas": source_quotas,
        "quotas": {
            split: {
                "accepted": quotas[split].accepted,
                "present": quotas[split].present,
            }
            for split in _SPLITS
        },
    }
    digest = hashlib.sha256(_canonical_json(normalized)).hexdigest()
    return Sop05ASupplementConfig(
        version=SOP05_A_SUPPLEMENT_VERSION,
        sampling_origin=SOP05_A_SUPPLEMENT_SAMPLING_ORIGIN,
        present_angle_distribution=SOP05_A_SUPPLEMENT_ANGLE_DISTRIBUTION,
        present_max_attempts_per_mother=attempts,
        final_sampling_seed=final_seed,
        source_generation_seeds=MappingProxyType(seeds),
        source_mother_quotas=MappingProxyType(source_quotas),
        quotas=MappingProxyType(quotas),
        digest=digest,
    )


def load_sop05_a_supplement_config(path: str | Path) -> Sop05ASupplementConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise Sop05ASupplementError(f"failed to read A-supplement config: {path}") from exc
    return normalize_sop05_a_supplement_config(raw)


def _stable_rng(*, namespace: str, payload: Mapping[str, object]) -> np.random.Generator:
    entropy = int.from_bytes(
        hashlib.sha256(
            _canonical_json({"namespace": namespace, **dict(payload)})
        ).digest(),
        byteorder="big",
    )
    return np.random.default_rng(entropy)


def sample_present_angles(
    *,
    config: Sop05ASupplementConfig,
    split: str,
    source_publication_semantic_digest: str,
    mother_id: str,
) -> np.ndarray:
    """Return the deterministic continuous uniform retry sequence for one mother."""

    if not isinstance(config, Sop05ASupplementConfig):
        raise TypeError("config must be a Sop05ASupplementConfig")
    if split not in config.quotas:
        raise Sop05ASupplementError(f"split {split!r} is absent from supplement quotas")
    if not isinstance(source_publication_semantic_digest, str) or not source_publication_semantic_digest:
        raise Sop05ASupplementError("source publication digest is invalid")
    if not isinstance(mother_id, str) or not mother_id:
        raise Sop05ASupplementError("mother_id is invalid")
    rng = _stable_rng(
        namespace="sop05/a-supplement/present-angle/v1",
        payload={
            "seed": config.final_sampling_seed,
            "split": split,
            "source_publication_semantic_digest": source_publication_semantic_digest,
            "mother_id": mother_id,
        },
    )
    angles = np.asarray(
        rng.uniform(
            -np.pi,
            np.pi,
            size=config.present_max_attempts_per_mother,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(angles).all() or np.any(angles < -np.pi) or np.any(angles >= np.pi):
        raise RuntimeError("A-supplement angle sampler escaped [-pi, pi)")
    angles.setflags(write=False)
    return angles


def _validate_source(
    source: LoadedSop05rTebOutput,
    *,
    split: str,
    quota: Sop05ASupplementQuota,
) -> Mapping[str, object]:
    if not isinstance(source, LoadedSop05rTebOutput) or not source.complete:
        raise Sop05ASupplementError("A supplement requires a complete mother collection")
    evidence = source.manifest.get("source_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("placement_selection_mode") != "h0_hidden":
        raise Sop05ASupplementError(
            "A supplement source must be generated with h0_hidden placement selection"
        )
    event_ids = tuple(event.generated_event_id for event in source.events)
    if len(set(event_ids)) != len(event_ids):
        raise Sop05ASupplementError("A supplement source has duplicate mother IDs")
    if len(event_ids) < quota.accepted:
        raise Sop05ASupplementError(
            f"A supplement source has {len(event_ids)} mothers for quota {quota.accepted}"
        )
    record_by_event: dict[str, object] = {}
    for record in source.trajectories.records:
        event_id = record.event_id
        if event_id in record_by_event:
            raise Sop05ASupplementError("mother has duplicate trajectory records")
        record_by_event[event_id] = record
    if set(record_by_event) != set(event_ids):
        raise Sop05ASupplementError("mother and trajectory event identities differ")
    states_by_event: dict[str, object] = {}
    for event in source.events:
        visibility = np.asarray(event.target_visibility_history)
        if visibility.shape != (8,) or visibility.dtype != np.bool_ or bool(visibility[0]):
            raise Sop05ASupplementError("A supplement source contains a non-A H0 history")
        record = record_by_event[event.generated_event_id]
        state = source.decision_states.get(record.decision_state_id)
        if state is None or not isinstance(state.split, str):
            raise Sop05ASupplementError("mother decision split is missing")
        if state.split != split:
            raise Sop05ASupplementError("A supplement source crosses split boundaries")
        states_by_event[event.generated_event_id] = state
    return MappingProxyType(states_by_event)


def _ordered_event_indices(
    source: LoadedSop05rTebOutput,
    *,
    split: str,
    config: Sop05ASupplementConfig,
) -> tuple[int, ...]:
    rng = _stable_rng(
        namespace="sop05/a-supplement/mother-order/v1",
        payload={
            "seed": config.final_sampling_seed,
            "split": split,
            "source_publication_semantic_digest": source.publication_semantic_digest,
        },
    )
    return tuple(int(index) for index in rng.permutation(len(source.events)))


def _evaluate_present_event(
    source: LoadedSop05rTebOutput,
    event: object,
    *,
    split: str,
    config: Sop05ASupplementConfig,
    state: object | None = None,
) -> _PresentEvaluation:
    try:
        base_config = dict(source.manifest["base_config"])
        mother = (
            _unseen_mother(source, event, base_config=base_config)
            if state is None
            else _unseen_mother_from_state(event, state, base_config=base_config)
        )
    except IncompleteContextError:
        return _PresentEvaluation(
            mother_id=event.generated_event_id,
            history_poses=None,
            future_poses=None,
            attempted_angle_count=0,
            selected_angle_rad=None,
            rejection_reason_counts={"incomplete_context": 1},
        )
    rejection_counts: Counter[str] = Counter()
    prepared_context = prepare_unseen_candidate_context(mother)
    angles = sample_present_angles(
        config=config,
        split=split,
        source_publication_semantic_digest=source.publication_semantic_digest,
        mother_id=event.generated_event_id,
    )
    for attempt, angle in enumerate(angles, start=1):
        transformed = transform_long40_target(
            mother.target_motion,
            angle_rad=float(angle),
        )
        decision = evaluate_candidate(
            mother,
            transformed_target=transformed,
            prepared_context=prepared_context,
        )
        if decision.legal:
            target = decision.accepted_target
            assert target is not None
            return _PresentEvaluation(
                mother_id=event.generated_event_id,
                history_poses=np.column_stack(
                    (target.positions[:8], target.headings[:8])
                ).astype(np.float32, copy=False),
                future_poses=np.column_stack(
                    (target.positions[8:], target.headings[8:])
                ).astype(np.float32, copy=False),
                attempted_angle_count=attempt,
                selected_angle_rad=float(angle),
                rejection_reason_counts=dict(sorted(rejection_counts.items())),
            )
        assert decision.rejection_reason is not None
        rejection_counts[decision.rejection_reason] += 1
    return _PresentEvaluation(
        mother_id=event.generated_event_id,
        history_poses=None,
        future_poses=None,
        attempted_angle_count=len(angles),
        selected_angle_rad=None,
        rejection_reason_counts=dict(sorted(rejection_counts.items())),
    )


def _present_worker(index: int) -> _PresentEvaluation:
    context = _PRESENT_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("A-supplement present worker is not initialized")
    return _evaluate_present_event(
        context.source,
        context.source.events[index],
        split=context.split,
        config=context.config,
        state=context.states_by_event[context.source.events[index].generated_event_id],
    )


def _ordered_present_evaluations(
    source: LoadedSop05rTebOutput,
    indices: tuple[int, ...],
    *,
    split: str,
    config: Sop05ASupplementConfig,
    states_by_event: Mapping[str, object],
    workers: int,
) -> Iterator[tuple[int, _PresentEvaluation]]:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise Sop05ASupplementError("workers must be a positive integer")
    if workers == 1:
        for index in indices:
            yield index, _evaluate_present_event(
                source,
                source.events[index],
                split=split,
                config=config,
                state=states_by_event[source.events[index].generated_event_id],
            )
        return
    if multiprocessing.get_start_method(allow_none=True) not in (None, "fork"):
        raise Sop05ASupplementError("parallel A-supplement selection requires fork")
    global _PRESENT_WORKER_CONTEXT
    previous = _PRESENT_WORKER_CONTEXT
    _PRESENT_WORKER_CONTEXT = _PresentWorkerContext(
        source=source,
        split=split,
        config=config,
        states_by_event=states_by_event,
    )
    context = multiprocessing.get_context("fork")
    pool = context.Pool(processes=min(workers, len(indices)))
    try:
        for index, evaluation in zip(
            indices,
            pool.imap(_present_worker, indices, chunksize=1),
        ):
            yield index, evaluation
    finally:
        pool.terminate()
        pool.join()
        _PRESENT_WORKER_CONTEXT = previous


def _selection_provenance(
    *,
    config: Sop05ASupplementConfig,
    split: str,
    quota: Sop05ASupplementQuota,
    stratum: str,
    candidate_rank: int,
) -> dict[str, object]:
    return {
        "supplement_version": config.version,
        "supplement_config_digest": config.digest,
        "sampling_origin": config.sampling_origin,
        "stratum": stratum,
        "stratum_selection": "quota_conditioned_without_replacement_v1",
        "requested_present_fraction": (
            0.0 if quota.accepted == 0 else quota.present / quota.accepted
        ),
        "source_generation_seed": config.source_generation_seeds[split],
        "final_sampling_seed": config.final_sampling_seed,
        "candidate_rank": candidate_rank,
    }


def select_a_supplement_scenarios(
    source: LoadedSop05rTebOutput,
    *,
    split: str,
    config: Sop05ASupplementConfig,
    workers: int = 1,
) -> tuple[Sop05FinalScenarioSelection, ...]:
    """Select exact target-present and target-empty A quotas from one source."""

    if not isinstance(config, Sop05ASupplementConfig):
        raise TypeError("config must be a Sop05ASupplementConfig")
    quota = config.quotas.get(split)
    if quota is None:
        raise Sop05ASupplementError(f"split {split!r} is absent from supplement quotas")
    states_by_event = _validate_source(source, split=split, quota=quota)
    ordered = _ordered_event_indices(source, split=split, config=config)
    rank_by_index = {index: rank for rank, index in enumerate(ordered)}
    attempted_present: set[int] = set()
    present: list[Sop05FinalScenarioSelection] = []
    evaluations = _ordered_present_evaluations(
        source,
        ordered,
        split=split,
        config=config,
        states_by_event=states_by_event,
        workers=workers,
    )
    try:
        for index, evaluation in evaluations:
            if len(present) >= quota.present:
                break
            attempted_present.add(index)
            if not evaluation.accepted:
                continue
            assert evaluation.history_poses is not None
            assert evaluation.future_poses is not None
            present.append(
                Sop05FinalScenarioSelection(
                    mother_id=evaluation.mother_id,
                    split=split,
                    target_present=True,
                    history_poses=evaluation.history_poses,
                    future_poses=evaluation.future_poses,
                    provenance={
                        **_selection_provenance(
                            config=config,
                            split=split,
                            quota=quota,
                            stratum="a_present",
                            candidate_rank=rank_by_index[index],
                        ),
                        "attempted_angle_count": evaluation.attempted_angle_count,
                        "selected_angle_rad": evaluation.selected_angle_rad,
                        "rejection_reason_counts": dict(
                            evaluation.rejection_reason_counts
                        ),
                    },
                )
            )
    finally:
        close = getattr(evaluations, "close", None)
        if callable(close):
            close()
    if len(present) != quota.present:
        raise Sop05ASupplementError(
            "A-present quota unmet: "
            f"accepted {len(present)} of {quota.present} after "
            f"{len(attempted_present)} mothers"
        )

    empty_indices = [index for index in ordered if index not in attempted_present]
    if len(empty_indices) < quota.empty:
        raise Sop05ASupplementError(
            "A-empty quota unmet after reserving present-attempt mothers: "
            f"available {len(empty_indices)} of {quota.empty}"
        )
    zeros_history = np.zeros((8, 3), dtype=np.float32)
    zeros_future = np.zeros((32, 3), dtype=np.float32)
    empty = [
        Sop05FinalScenarioSelection(
            mother_id=source.events[index].generated_event_id,
            split=split,
            target_present=False,
            history_poses=zeros_history,
            future_poses=zeros_future,
            provenance=_selection_provenance(
                config=config,
                split=split,
                quota=quota,
                stratum="a_empty",
                candidate_rank=rank_by_index[index],
            ),
        )
        for index in empty_indices[: quota.empty]
    ]
    selected = tuple((*present, *empty))
    if len(selected) != quota.accepted or len({item.mother_id for item in selected}) != len(
        selected
    ):
        raise RuntimeError("A-supplement selection count or mother uniqueness drifted")
    return selected


def publish_sop05_a_supplement(
    source: LoadedSop05rTebOutput,
    *,
    split: str,
    config: Sop05ASupplementConfig,
    output_dir: str | Path,
    workers: int = 1,
) -> Sop05ASupplementPublishResult:
    selections = select_a_supplement_scenarios(
        source,
        split=split,
        config=config,
        workers=workers,
    )
    publication = publish_selected_sop05_final_scenarios(
        source,
        selections=selections,
        output_dir=output_dir,
        unseen_prior_config_digest=config.digest,
        seen_prior_config_digest="not_applicable_a_supplement_v1",
    )
    present_count = sum(item.target_present for item in selections)
    return Sop05ASupplementPublishResult(
        publication=publication,
        present_count=present_count,
        empty_count=len(selections) - present_count,
        config_digest=config.digest,
    )


__all__ = (
    "SOP05_A_SUPPLEMENT_VERSION",
    "Sop05ASupplementConfig",
    "Sop05ASupplementError",
    "Sop05ASupplementPublishResult",
    "Sop05ASupplementQuota",
    "load_sop05_a_supplement_config",
    "normalize_sop05_a_supplement_config",
    "publish_sop05_a_supplement",
    "sample_present_angles",
    "select_a_supplement_scenarios",
)
