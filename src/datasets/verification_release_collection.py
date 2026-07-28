"""Strict release loading with train-frozen verification-value labels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any

import numpy as np

from src.contracts import GridSpec, VerificationSample
from src.datasets.verification_collection import verification_input_digests
from src.datasets.verification_dataloader import (
    LoadedVerificationShard,
    load_verification_shard,
)
from src.datasets.verification_dataset import (
    audit_verification_groups,
    validate_verification_sample_for_publication,
)
from src.evaluation.verification_value_calibration import (
    LoadedRejectCostCalibration,
    revalue_reject_cost,
)
from src.generation.verification_release import (
    LoadedVerificationRelease,
    VerificationRevaluationRecord,
    load_verification_release,
    load_verification_revaluation_records,
)
from src.planning.verification_actions import VerificationActionLibrary


VERIFICATION_RELEASE_COLLECTION_VERSION = (
    "calibrated_verification_release_collection_v1"
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = frozenset({"train", "calibration", "val", "test"})
_INPUT_ARRAY_FIELDS = (
    "bev_history",
    "state_channels",
    "trajectory_channels",
    "verification_fov_mask",
    "verification_action_vector",
)


@dataclass(frozen=True)
class LoadedCalibratedVerificationRelease:
    """One immutable release with labels revalued at the frozen reject cost."""

    root: Path
    release: LoadedVerificationRelease
    split: str
    samples: tuple[VerificationSample, ...]
    shard_dirs: tuple[Path, ...]
    loaded_shards: tuple[LoadedVerificationShard, ...]
    release_manifest_digest: str
    raw_input_manifest_digest: str
    raw_split_digest: str
    input_manifest_digest: str
    split_digest: str
    split_digests: Mapping[str, str]
    calibration_digest: str
    reject_cost: float
    audit_report: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "split_digests",
            MappingProxyType(dict(self.split_digests)),
        )
        object.__setattr__(
            self,
            "audit_report",
            MappingProxyType(dict(self.audit_report)),
        )


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("release collection identity must be canonical JSON") from exc


def _domain_digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _source_pairs(
    calibration: LoadedRejectCostCalibration,
) -> frozenset[tuple[str, str]]:
    digests = calibration.source_release_manifest_digests
    identities = calibration.source_release_request_identities
    if (
        len(digests) != len(identities)
        or not digests
        or len(set(digests)) != len(digests)
        or len(set(identities)) != len(identities)
    ):
        raise ValueError("calibration source release binding is invalid")
    return frozenset(
        (
            str(identity),
            _digest(digest, name="calibration source release digest"),
        )
        for identity, digest in zip(identities, digests, strict=True)
    )


def _validate_calibration_binding(
    release: LoadedVerificationRelease,
    calibration: LoadedRejectCostCalibration,
) -> float:
    if not isinstance(calibration, LoadedRejectCostCalibration):
        raise TypeError("calibration must be a LoadedRejectCostCalibration")
    if (
        calibration.status != "selected"
        or calibration.selected_reject_cost is None
    ):
        raise ValueError("a selected reject-cost calibration is required")
    _digest(calibration.calibration_digest, name="calibration digest")
    pairs = _source_pairs(calibration)
    current = (release.request_identity, release.manifest_digest)
    source_identities = {identity for identity, _ in pairs}
    source_digests = {digest for _, digest in pairs}
    if release.split == "train":
        if current not in pairs:
            raise ValueError(
                "train release is not an authenticated calibration source"
            )
    elif (
        current in pairs
        or release.request_identity in source_identities
        or release.manifest_digest in source_digests
    ):
        raise ValueError("held-out release must not be a calibration source")
    reject_cost = float(calibration.selected_reject_cost)
    if not np.isfinite(reject_cost) or reject_cost < 0.0:
        raise ValueError("calibrated reject cost must be finite and non-negative")
    return reject_cost


def _release_data_roots(
    release: LoadedVerificationRelease,
) -> tuple[Path, ...]:
    descriptors = release.manifest.get("shards")
    if not isinstance(descriptors, list) or len(descriptors) != release.shard_count:
        raise ValueError("verification release shard descriptors are invalid")
    roots: list[Path] = []
    accepted_total = 0
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, Mapping):
            raise ValueError("verification release shard descriptor is invalid")
        relative = descriptor.get("relative_root")
        path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            path is None
            or path.is_absolute()
            or path.parts != ("shards", f"shard-{index:05d}")
        ):
            raise ValueError("verification release shard path is unsafe")
        accepted = descriptor.get("accepted_group_count")
        sample_count = descriptor.get("sample_count")
        if (
            isinstance(accepted, bool)
            or not isinstance(accepted, int)
            or accepted < 0
            or sample_count != accepted * 6
        ):
            raise ValueError("verification release shard counts are invalid")
        accepted_total += accepted
        if accepted:
            data_root = release.root / str(path) / "data"
            if data_root.is_symlink() or not data_root.is_dir():
                raise ValueError("verification release data shard is incomplete")
            roots.append(data_root)
    if accepted_total != release.accepted_group_count or not roots:
        raise ValueError("verification release contains no usable accepted groups")
    return tuple(roots)


def _aligned_record_map(
    samples: tuple[VerificationSample, ...],
    records: tuple[VerificationRevaluationRecord, ...],
    *,
    release: LoadedVerificationRelease,
) -> dict[str, VerificationRevaluationRecord]:
    sample_ids = {sample.sample_id for sample in samples}
    record_ids = {record.sample_id for record in records}
    if (
        len(sample_ids) != len(samples)
        or len(record_ids) != len(records)
        or sample_ids != record_ids
    ):
        raise ValueError("release sample IDs and revaluation sample IDs differ")
    by_id = {record.sample_id: record for record in records}
    for sample in samples:
        record = by_id[sample.sample_id]
        if (
            record.release_request_identity != release.request_identity
            or record.split != sample.split
            or record.action_id != sample.verification_action_id
            or record.ranking_group_id
            != sample.metadata.get("ranking_group_id")
        ):
            raise ValueError("release sample and revaluation record identity differ")
    return by_id


def _revalue_sample(
    sample: VerificationSample,
    record: VerificationRevaluationRecord,
    *,
    reject_cost: float,
) -> VerificationSample:
    value = revalue_reject_cost(
        nominal_execute_losses=np.asarray(
            [record.realized_execute_loss],
            dtype=np.float64,
        ),
        unclipped_best_policy_losses=(record.unclipped_best_policy_loss,),
        action_cost=record.action_cost,
        reject_cost=reject_cost,
    )
    updated = replace(
        sample,
        value_target=value.value_target,
        useful_target=value.useful_target,
        br_before=value.br_before,
        post_risk=value.post_risk,
    )
    if any(
        getattr(updated, field) is not getattr(sample, field)
        for field in _INPUT_ARRAY_FIELDS
    ):
        raise AssertionError("calibration changed a verification model input")
    return updated


def load_calibrated_verification_release(
    release_dir: str | Path,
    *,
    grid: GridSpec,
    library: VerificationActionLibrary,
    expected_split: str,
    calibration: LoadedRejectCostCalibration,
) -> LoadedCalibratedVerificationRelease:
    """Strictly load one release and deterministically apply frozen labels."""

    if expected_split not in _SPLITS:
        raise ValueError("expected_split is invalid")
    release = load_verification_release(release_dir)
    if release.split != expected_split:
        raise ValueError("verification release split mismatch")
    reject_cost = _validate_calibration_binding(release, calibration)
    shard_dirs = _release_data_roots(release)
    loaded_shards = tuple(
        load_verification_shard(root, grid=grid, library=library)
        for root in shard_dirs
    )
    raw_samples = tuple(
        sorted(
            (
                sample
                for shard in loaded_shards
                for sample in shard.samples
            ),
            key=lambda sample: (sample.split, sample.sample_id),
        )
    )
    if (
        len(raw_samples) != release.sample_count
        or {sample.split for sample in raw_samples} != {expected_split}
    ):
        raise ValueError("loaded release sample count or split differs")
    records = load_verification_revaluation_records(release.root)
    by_id = _aligned_record_map(raw_samples, records, release=release)
    samples = tuple(
        _revalue_sample(
            sample,
            by_id[sample.sample_id],
            reject_cost=reject_cost,
        )
        for sample in raw_samples
    )
    for sample in samples:
        validate_verification_sample_for_publication(
            sample,
            grid=grid,
            library=library,
        )
    audit = audit_verification_groups(list(samples), require_complete=True)
    if (
        audit.get("sample_count") != release.sample_count
        or audit.get("group_count") != release.accepted_group_count
    ):
        raise ValueError("calibrated release audit totals differ")
    raw_input_digest, raw_split_digests = verification_input_digests(
        loaded_shards
    )
    if set(raw_split_digests) != {expected_split}:
        raise ValueError("raw release split digest set differs")
    identity = {
        "version": VERIFICATION_RELEASE_COLLECTION_VERSION,
        "release_manifest_digest": release.manifest_digest,
        "raw_input_manifest_digest": raw_input_digest,
        "calibration_digest": calibration.calibration_digest,
        "reject_cost": reject_cost,
    }
    input_digest = _domain_digest(
        b"calibrated-verification-release-input-v1\0",
        identity,
    )
    raw_split_digest = raw_split_digests[expected_split]
    split_digest = _domain_digest(
        b"calibrated-verification-release-split-v1\0",
        {
            **identity,
            "split": expected_split,
            "raw_split_digest": raw_split_digest,
        },
    )
    return LoadedCalibratedVerificationRelease(
        root=release.root,
        release=release,
        split=expected_split,
        samples=samples,
        shard_dirs=shard_dirs,
        loaded_shards=loaded_shards,
        release_manifest_digest=release.manifest_digest,
        raw_input_manifest_digest=raw_input_digest,
        raw_split_digest=raw_split_digest,
        input_manifest_digest=input_digest,
        split_digest=split_digest,
        split_digests={expected_split: split_digest},
        calibration_digest=calibration.calibration_digest,
        reject_cost=reject_cost,
        audit_report=audit,
    )


__all__ = (
    "VERIFICATION_RELEASE_COLLECTION_VERSION",
    "LoadedCalibratedVerificationRelease",
    "load_calibrated_verification_release",
)
