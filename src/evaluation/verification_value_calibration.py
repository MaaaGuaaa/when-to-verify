"""Pure reject-cost revaluation from stored verification decision losses."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from numbers import Real
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from src.contracts import SCHEMA_VERSION
from src.generation.verification_gt import load_verification_gt_config
from src.generation.verification_release import (
    VerificationRevaluationRecord,
    load_verification_release,
    load_verification_revaluation_records,
)
from src.planning.verification_actions import CANONICAL_ACTION_IDS
from src.utils.atomic_publish import atomic_rename_noreplace


REJECT_COST_CALIBRATION_VERSION = "verification_reject_cost_calibration_v1"
REJECT_COST_CALIBRATION_LAYOUT_VERSION = (
    "verification_reject_cost_calibration_layout_v1"
)
_CALIBRATION = "calibration.json"
_FROZEN_GT = "verification_gt_frozen.yaml"
_MANIFEST = "manifest.json"
_COMPLETE = "COMPLETE.json"
_SUCCESS_FILES = frozenset({_CALIBRATION, _FROZEN_GT, _MANIFEST, _COMPLETE})
_FAILURE_FILES = frozenset({_CALIBRATION, _MANIFEST, _COMPLETE})
_CONFIG_KEYS = frozenset(
    {"schema_version", "calibration_version", "candidates", "criteria"}
)
_CRITERIA_KEYS = frozenset(
    {
        "minimum_group_count",
        "minimum_positive_fraction",
        "maximum_positive_fraction",
        "minimum_mixed_action_count",
    }
)


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class RejectCostRevalue:
    """Recomputed value terms for one action and one reject cost."""

    br_before: float
    mean_post_before_action_cost: float
    post_risk: float
    value_target: float
    useful_target: int


@dataclass(frozen=True)
class RejectCostCalibrationCriteria:
    minimum_group_count: int
    minimum_positive_fraction: float
    maximum_positive_fraction: float
    minimum_mixed_action_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_group_count, bool)
            or not isinstance(self.minimum_group_count, int)
            or self.minimum_group_count <= 0
        ):
            raise ValueError("minimum_group_count must be a positive integer")
        minimum = _finite_fraction(
            self.minimum_positive_fraction,
            name="minimum_positive_fraction",
        )
        maximum = _finite_fraction(
            self.maximum_positive_fraction,
            name="maximum_positive_fraction",
        )
        if minimum >= maximum:
            raise ValueError(
                "minimum_positive_fraction must be below maximum_positive_fraction"
            )
        object.__setattr__(self, "minimum_positive_fraction", minimum)
        object.__setattr__(self, "maximum_positive_fraction", maximum)
        if (
            isinstance(self.minimum_mixed_action_count, bool)
            or not isinstance(self.minimum_mixed_action_count, int)
            or not 0
            <= self.minimum_mixed_action_count
            <= len(CANONICAL_ACTION_IDS)
        ):
            raise ValueError(
                "minimum_mixed_action_count must lie within the action library"
            )


@dataclass(frozen=True)
class RejectCostCalibrationResult:
    status: str
    selected_reject_cost: float | None
    group_count: int
    sample_count: int
    release_count: int
    candidate_reports: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        if self.status not in {"selected", "no_candidate_passed"}:
            raise ValueError("calibration status is invalid")
        if (self.status == "selected") != (
            self.selected_reject_cost is not None
        ):
            raise ValueError("calibration status and selected cost disagree")
        if self.selected_reject_cost is not None:
            object.__setattr__(
                self,
                "selected_reject_cost",
                _finite_nonnegative(
                    self.selected_reject_cost,
                    name="selected_reject_cost",
                ),
            )
        for name in ("group_count", "sample_count", "release_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(
            self,
            "candidate_reports",
            MappingProxyType(
                {
                    str(key): MappingProxyType(dict(value))
                    for key, value in self.candidate_reports.items()
                }
            ),
        )


@dataclass(frozen=True)
class LoadedRejectCostCalibration:
    root: Path
    status: str
    selected_reject_cost: float | None
    calibration_digest: str
    source_release_manifest_digests: tuple[str, ...]
    source_release_request_identities: tuple[str, ...]
    calibration: Mapping[str, object]
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "calibration", MappingProxyType(dict(self.calibration))
        )
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


def _finite_fraction(value: Any, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("calibration metadata must be finite canonical JSON") from exc


def _strict_json(path: Path, *, name: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a real file")

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"calibration input must be a real file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def _candidate_key(value: float) -> str:
    return str(float(value))


def _validated_records(
    records: Sequence[VerificationRevaluationRecord],
) -> tuple[VerificationRevaluationRecord, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence")
    rows = tuple(records)
    if not rows or any(
        not isinstance(row, VerificationRevaluationRecord) for row in rows
    ):
        raise ValueError(
            "records must contain VerificationRevaluationRecord values"
        )
    if {row.split for row in rows} != {"train"}:
        raise ValueError("reject-cost calibration accepts train releases only")
    if len({row.sample_id for row in rows}) != len(rows):
        raise ValueError("calibration records contain duplicate sample IDs")
    groups: dict[str, list[VerificationRevaluationRecord]] = {}
    for row in rows:
        groups.setdefault(row.ranking_group_id, []).append(row)
    for group_id, group in groups.items():
        if (
            len(group) != len(CANONICAL_ACTION_IDS)
            or {row.action_id for row in group} != set(CANONICAL_ACTION_IDS)
        ):
            raise ValueError(
                f"calibration group {group_id!r} must contain six canonical actions"
            )
        if len({row.task_id for row in group}) != 1 or len(
            {row.mother_id for row in group}
        ) != 1:
            raise ValueError("calibration ranking group task identity differs")
    return tuple(sorted(rows, key=lambda row: row.sample_id))


def _validated_candidates(candidates: Sequence[Real]) -> tuple[float, ...]:
    if isinstance(candidates, (str, bytes)) or not isinstance(
        candidates, Sequence
    ):
        raise TypeError("candidates must be a sequence")
    normalized = tuple(
        _finite_nonnegative(value, name=f"candidates[{index}]")
        for index, value in enumerate(candidates)
    )
    if not normalized:
        raise ValueError("candidates must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("candidates must be unique")
    return tuple(sorted(normalized))


def _candidate_report(
    records: tuple[VerificationRevaluationRecord, ...],
    *,
    reject_cost: float,
    criteria: RejectCostCalibrationCriteria,
    group_count: int,
) -> dict[str, object]:
    values: list[float] = []
    useful: list[int] = []
    positive_risk_check = True
    before_rejection_count = 0
    post_rejection_count = 0
    per_action_positive: Counter[str] = Counter()
    per_action_total: Counter[str] = Counter()
    for record in records:
        row = revalue_reject_cost(
            nominal_execute_losses=np.asarray(
                [record.realized_execute_loss],
                dtype=np.float64,
            ),
            unclipped_best_policy_losses=(
                record.unclipped_best_policy_loss,
            ),
            action_cost=record.action_cost,
            reject_cost=reject_cost,
        )
        values.append(row.value_target)
        useful.append(row.useful_target)
        per_action_total[record.action_id] += 1
        per_action_positive[record.action_id] += row.useful_target
        before_rejection_count += int(
            record.realized_execute_loss >= reject_cost
        )
        post_rejection_count += int(
            record.unclipped_best_policy_loss is None
            or record.unclipped_best_policy_loss >= reject_cost
        )
        if row.useful_target:
            risk_reduction = (
                row.br_before - row.mean_post_before_action_cost
            )
            positive_risk_check = positive_risk_check and (
                risk_reduction > record.action_cost
            )
    positive_count = sum(useful)
    negative_count = len(records) - positive_count
    positive_fraction = positive_count / len(records)
    mixed_action_count = sum(
        0 < per_action_positive[action_id] < per_action_total[action_id]
        for action_id in CANONICAL_ACTION_IDS
    )
    checks = {
        "minimum_group_count": group_count >= criteria.minimum_group_count,
        "both_value_signs": positive_count > 0 and negative_count > 0,
        "positive_fraction": (
            criteria.minimum_positive_fraction
            <= positive_fraction
            <= criteria.maximum_positive_fraction
        ),
        "mixed_action_count": (
            mixed_action_count >= criteria.minimum_mixed_action_count
        ),
        "positive_risk_reduction_exceeds_action_cost": positive_risk_check,
    }
    value_array = np.asarray(values, dtype=np.float64)
    quantile_levels = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    return {
        "reject_cost": reject_cost,
        "status": "pass" if all(checks.values()) else "fail",
        "failed_checks": [
            name for name, passed in checks.items() if not passed
        ],
        "sample_count": len(records),
        "group_count": group_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_fraction": positive_fraction,
        "mixed_action_count": mixed_action_count,
        "positive_risk_reduction_exceeds_action_cost": positive_risk_check,
        "before_rejection_selection_rate": (
            before_rejection_count / len(records)
        ),
        "post_rejection_selection_rate": post_rejection_count / len(records),
        "per_action": {
            action_id: {
                "sample_count": per_action_total[action_id],
                "positive_count": per_action_positive[action_id],
                "negative_count": (
                    per_action_total[action_id]
                    - per_action_positive[action_id]
                ),
            }
            for action_id in CANONICAL_ACTION_IDS
        },
        "value_quantiles": {
            str(level): float(value)
            for level, value in zip(
                quantile_levels,
                np.quantile(value_array, quantile_levels),
                strict=True,
            )
        },
        "checks": checks,
    }


def revalue_reject_cost(
    *,
    nominal_execute_losses: np.ndarray,
    unclipped_best_policy_losses: tuple[float | None, ...],
    action_cost: Any,
    reject_cost: Any,
) -> RejectCostRevalue:
    """Revalue one action without rerunning posterior or trajectory simulation."""

    if (
        not isinstance(nominal_execute_losses, np.ndarray)
        or nominal_execute_losses.dtype != np.float64
    ):
        raise TypeError("nominal_execute_losses must be a float64 ndarray")
    if nominal_execute_losses.ndim != 1 or nominal_execute_losses.size == 0:
        raise ValueError("nominal_execute_losses must be a non-empty vector")
    if (
        not np.isfinite(nominal_execute_losses).all()
        or np.any(nominal_execute_losses < 0.0)
    ):
        raise ValueError(
            "nominal_execute_losses must be finite and non-negative"
        )
    if not isinstance(unclipped_best_policy_losses, tuple):
        raise TypeError("unclipped_best_policy_losses must be a tuple")
    if len(unclipped_best_policy_losses) != nominal_execute_losses.size:
        raise ValueError(
            "unclipped_best_policy_losses must align with nominal losses"
        )

    policy_losses: list[float | None] = []
    for index, value in enumerate(unclipped_best_policy_losses):
        policy_losses.append(
            None
            if value is None
            else _finite_nonnegative(
                value,
                name=f"unclipped_best_policy_losses[{index}]",
            )
        )
    action = _finite_nonnegative(action_cost, name="action_cost")
    reject = _finite_nonnegative(reject_cost, name="reject_cost")

    mean_execute = float(
        np.mean(nominal_execute_losses, dtype=np.float64)
    )
    br_before = min(mean_execute, reject)
    post = np.asarray(
        [
            reject if value is None else min(value, reject)
            for value in policy_losses
        ],
        dtype=np.float64,
    )
    mean_post = float(np.mean(post, dtype=np.float64))
    post_risk = mean_post + action
    value_target = br_before - post_risk
    return RejectCostRevalue(
        br_before=br_before,
        mean_post_before_action_cost=mean_post,
        post_risk=post_risk,
        value_target=value_target,
        useful_target=int(value_target > 0.0),
    )


def calibrate_reject_cost(
    records: Sequence[VerificationRevaluationRecord],
    *,
    candidates: Sequence[Real],
    criteria: RejectCostCalibrationCriteria,
) -> RejectCostCalibrationResult:
    """Evaluate fixed candidates and choose the smallest scientifically valid cost."""

    if not isinstance(criteria, RejectCostCalibrationCriteria):
        raise TypeError("criteria must be RejectCostCalibrationCriteria")
    rows = _validated_records(records)
    candidate_values = _validated_candidates(candidates)
    group_count = len({row.ranking_group_id for row in rows})
    release_count = len({row.release_request_identity for row in rows})
    reports = {
        _candidate_key(candidate): _candidate_report(
            rows,
            reject_cost=candidate,
            criteria=criteria,
            group_count=group_count,
        )
        for candidate in candidate_values
    }
    passing = [
        candidate
        for candidate in candidate_values
        if reports[_candidate_key(candidate)]["status"] == "pass"
    ]
    selected = passing[0] if passing else None
    return RejectCostCalibrationResult(
        status="selected" if selected is not None else "no_candidate_passed",
        selected_reject_cost=selected,
        group_count=group_count,
        sample_count=len(rows),
        release_count=release_count,
        candidate_reports=reports,
    )


def load_reject_cost_calibration_config(
    path: str | Path,
) -> tuple[tuple[float, ...], RejectCostCalibrationCriteria]:
    """Load the strict candidate set and selection criteria."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("calibration config must be a real file")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid calibration config: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != _CONFIG_KEYS:
        raise ValueError("calibration config keys are invalid")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("calibration config schema mismatch")
    if raw["calibration_version"] != REJECT_COST_CALIBRATION_VERSION:
        raise ValueError("unsupported calibration config version")
    criteria = raw["criteria"]
    if not isinstance(criteria, dict) or set(criteria) != _CRITERIA_KEYS:
        raise ValueError("calibration criteria keys are invalid")
    raw_candidates = raw["candidates"]
    if not isinstance(raw_candidates, list):
        raise ValueError("calibration candidates must be a list")
    return (
        _validated_candidates(raw_candidates),
        RejectCostCalibrationCriteria(**criteria),
    )


def _calibration_digest(value: object) -> str:
    return hashlib.sha256(
        b"verification_reject_cost_calibration_v1\0"
        + _canonical_bytes(value)
    ).hexdigest()


def _calibration_document(
    *,
    result: RejectCostCalibrationResult,
    candidates: tuple[float, ...],
    criteria: RejectCostCalibrationCriteria,
    sources: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "calibration_version": REJECT_COST_CALIBRATION_VERSION,
        "status": result.status,
        "selected_reject_cost": result.selected_reject_cost,
        "sample_count": result.sample_count,
        "group_count": result.group_count,
        "release_count": result.release_count,
        "candidates": list(candidates),
        "criteria": {
            "minimum_group_count": criteria.minimum_group_count,
            "minimum_positive_fraction": (
                criteria.minimum_positive_fraction
            ),
            "maximum_positive_fraction": (
                criteria.maximum_positive_fraction
            ),
            "minimum_mixed_action_count": (
                criteria.minimum_mixed_action_count
            ),
        },
        "source_releases": [dict(source) for source in sources],
        "candidate_reports": {
            key: dict(value)
            for key, value in result.candidate_reports.items()
        },
    }


def _frozen_gt_bytes(
    path: Path,
    *,
    reject_cost: float,
) -> bytes:
    load_verification_gt_config(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid verification GT config: {exc}") from exc
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != SCHEMA_VERSION
        or not isinstance(raw.get("decision"), dict)
    ):
        raise ValueError("verification GT config structure is invalid")
    frozen = json.loads(
        json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    frozen["decision"]["reject_cost"] = reject_cost
    return yaml.safe_dump(
        frozen,
        sort_keys=False,
        allow_unicode=False,
    ).encode("utf-8")


def _load_reject_cost_calibration(
    root: Path,
    *,
    require_selected: bool,
) -> LoadedRejectCostCalibration:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("calibration artifact must be a real directory")
    manifest = _strict_json(root / _MANIFEST, name=_MANIFEST)
    status = manifest.get("status")
    if status not in {"selected", "no_candidate_passed"}:
        raise ValueError("calibration manifest status is invalid")
    expected_files = _SUCCESS_FILES if status == "selected" else _FAILURE_FILES
    if {path.name for path in root.iterdir()} != expected_files:
        raise ValueError("calibration artifact file layout is invalid")
    expected_manifest_keys = {
        "layout_version",
        "schema_version",
        "calibration_version",
        "status",
        "calibration_digest",
        "source_release_manifest_digests",
        "source_release_request_identities",
        "config_sha256",
        "gt_config_sha256",
        "files",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("calibration manifest keys are invalid")
    if (
        manifest["layout_version"] != REJECT_COST_CALIBRATION_LAYOUT_VERSION
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["calibration_version"]
        != REJECT_COST_CALIBRATION_VERSION
    ):
        raise ValueError("calibration manifest version differs")
    calibration = _strict_json(root / _CALIBRATION, name=_CALIBRATION)
    calibration_digest = _digest(
        manifest["calibration_digest"],
        name="calibration_digest",
    )
    if calibration_digest != _calibration_digest(calibration):
        raise ValueError("calibration document digest differs")
    if (
        calibration.get("schema_version") != SCHEMA_VERSION
        or calibration.get("calibration_version")
        != REJECT_COST_CALIBRATION_VERSION
        or calibration.get("status") != status
    ):
        raise ValueError("calibration document identity differs")
    files = manifest["files"]
    expected_payload_names = (
        {_CALIBRATION, _FROZEN_GT}
        if status == "selected"
        else {_CALIBRATION}
    )
    if not isinstance(files, dict) or set(files) != expected_payload_names:
        raise ValueError("calibration payload checksums are invalid")
    for name, expected in files.items():
        if _digest(expected, name=f"files[{name}]") != _sha256_file(
            root / name
        ):
            raise ValueError(f"calibration payload checksum differs: {name}")
    source_digests_raw = manifest["source_release_manifest_digests"]
    source_identities_raw = manifest["source_release_request_identities"]
    if (
        not isinstance(source_digests_raw, list)
        or not source_digests_raw
        or not isinstance(source_identities_raw, list)
        or len(source_identities_raw) != len(source_digests_raw)
        or any(
            not isinstance(value, str) or not value
            for value in source_identities_raw
        )
    ):
        raise ValueError("calibration source release identities are invalid")
    source_digests = tuple(
        _digest(value, name="source release manifest digest")
        for value in source_digests_raw
    )
    source_identities = tuple(str(value) for value in source_identities_raw)
    if len(set(source_digests)) != len(source_digests) or len(
        set(source_identities)
    ) != len(source_identities):
        raise ValueError("calibration source releases are not unique")
    selected_raw = calibration.get("selected_reject_cost")
    if status == "selected":
        selected = _finite_nonnegative(
            selected_raw,
            name="selected_reject_cost",
        )
        frozen = load_verification_gt_config(root / _FROZEN_GT)
        if not np.isclose(
            frozen.reject_cost,
            selected,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("frozen GT reject cost differs from calibration")
    else:
        if selected_raw is not None:
            raise ValueError("failed calibration must not select a reject cost")
        selected = None
        if require_selected:
            raise ValueError("calibration artifact has no passing candidate")
    complete = _strict_json(root / _COMPLETE, name=_COMPLETE)
    expected_complete = {
        "layout_version": REJECT_COST_CALIBRATION_LAYOUT_VERSION,
        "status": status,
        "calibration_digest": calibration_digest,
        "manifest_sha256": _sha256_file(root / _MANIFEST),
    }
    if complete != expected_complete:
        raise ValueError("calibration completion marker differs")
    return LoadedRejectCostCalibration(
        root=root,
        status=str(status),
        selected_reject_cost=selected,
        calibration_digest=calibration_digest,
        source_release_manifest_digests=source_digests,
        source_release_request_identities=source_identities,
        calibration=calibration,
        manifest=manifest,
    )


def load_reject_cost_calibration(
    input_dir: str | Path,
    *,
    require_selected: bool = True,
) -> LoadedRejectCostCalibration:
    """Strictly load a complete calibration or diagnostic artifact."""

    return _load_reject_cost_calibration(
        Path(input_dir),
        require_selected=require_selected,
    )


def publish_reject_cost_calibration(
    output_dir: str | Path,
    *,
    release_dirs: Sequence[str | Path],
    config_path: str | Path,
    gt_config_path: str | Path,
) -> LoadedRejectCostCalibration:
    """Calibrate on train releases and atomically publish success or failure."""

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite calibration artifact: {destination}"
        )
    if isinstance(release_dirs, (str, bytes)) or not isinstance(
        release_dirs, Sequence
    ):
        raise TypeError("release_dirs must be a sequence")
    roots = tuple(Path(value) for value in release_dirs)
    if not roots:
        raise ValueError("release_dirs must be non-empty")
    candidates, criteria = load_reject_cost_calibration_config(config_path)
    gt_path = Path(gt_config_path)
    load_verification_gt_config(gt_path)
    loaded_sources = []
    all_records: list[VerificationRevaluationRecord] = []
    for root in roots:
        release = load_verification_release(root)
        if release.split != "train":
            raise ValueError("reject-cost calibration accepts train releases only")
        records = load_verification_revaluation_records(root)
        loaded_sources.append(
            {
                "request_identity": release.request_identity,
                "manifest_digest": release.manifest_digest,
                "sample_count": release.sample_count,
                "accepted_group_count": release.accepted_group_count,
            }
        )
        all_records.extend(records)
    loaded_sources.sort(key=lambda value: str(value["request_identity"]))
    if len(
        {str(value["request_identity"]) for value in loaded_sources}
    ) != len(loaded_sources):
        raise ValueError("calibration release identities must be unique")
    result = calibrate_reject_cost(
        tuple(all_records),
        candidates=candidates,
        criteria=criteria,
    )
    calibration = _calibration_document(
        result=result,
        candidates=candidates,
        criteria=criteria,
        sources=loaded_sources,
    )
    calibration_digest = _calibration_digest(calibration)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        calibration_payload = _canonical_bytes(calibration)
        (staging / _CALIBRATION).write_bytes(calibration_payload)
        payload_files = {_CALIBRATION: hashlib.sha256(calibration_payload).hexdigest()}
        if result.selected_reject_cost is not None:
            frozen_payload = _frozen_gt_bytes(
                gt_path,
                reject_cost=result.selected_reject_cost,
            )
            (staging / _FROZEN_GT).write_bytes(frozen_payload)
            frozen = load_verification_gt_config(staging / _FROZEN_GT)
            if frozen.reject_cost != result.selected_reject_cost:
                raise ValueError("staged frozen GT reject cost differs")
            payload_files[_FROZEN_GT] = hashlib.sha256(
                frozen_payload
            ).hexdigest()
        manifest = {
            "layout_version": REJECT_COST_CALIBRATION_LAYOUT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "calibration_version": REJECT_COST_CALIBRATION_VERSION,
            "status": result.status,
            "calibration_digest": calibration_digest,
            "source_release_manifest_digests": [
                value["manifest_digest"] for value in loaded_sources
            ],
            "source_release_request_identities": [
                value["request_identity"] for value in loaded_sources
            ],
            "config_sha256": _sha256_file(Path(config_path)),
            "gt_config_sha256": _sha256_file(gt_path),
            "files": payload_files,
        }
        manifest_payload = _canonical_bytes(manifest)
        (staging / _MANIFEST).write_bytes(manifest_payload)
        (staging / _COMPLETE).write_bytes(
            _canonical_bytes(
                {
                    "layout_version": (
                        REJECT_COST_CALIBRATION_LAYOUT_VERSION
                    ),
                    "status": result.status,
                    "calibration_digest": calibration_digest,
                    "manifest_sha256": hashlib.sha256(
                        manifest_payload
                    ).hexdigest(),
                }
            )
        )
        staged = _load_reject_cost_calibration(
            staging,
            require_selected=False,
        )
        atomic_rename_noreplace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    published = _load_reject_cost_calibration(
        destination,
        require_selected=False,
    )
    if published.calibration_digest != staged.calibration_digest:
        raise ValueError("published calibration differs from staging")
    return published


__all__ = (
    "REJECT_COST_CALIBRATION_LAYOUT_VERSION",
    "REJECT_COST_CALIBRATION_VERSION",
    "LoadedRejectCostCalibration",
    "RejectCostCalibrationCriteria",
    "RejectCostCalibrationResult",
    "RejectCostRevalue",
    "calibrate_reject_cost",
    "load_reject_cost_calibration",
    "load_reject_cost_calibration_config",
    "publish_reject_cost_calibration",
    "revalue_reject_cost",
)
