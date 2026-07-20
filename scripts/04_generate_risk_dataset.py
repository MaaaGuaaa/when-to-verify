#!/usr/bin/env python
"""Build one immutable SOP-07 risk shard from verified SOP03/04/05 inputs."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

import yaml


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.contracts import ContractError, SCHEMA_VERSION, build_grid_spec  # noqa: E402
from src.datasets.risk_dataset import (  # noqa: E402
    build_risk_samples_and_sidecars_from_sop06_group,
    build_risk_samples_from_sop06_group,
    pin_risk_shard_snapshot,
)
from src.datasets.sidecar_writer import (  # noqa: E402
    _atomic_rename_directory_noreplace as _atomic_pair_cleanup_claim_noreplace,
    _atomic_rename_directory_noreplace as _atomic_pair_commit_noreplace,
    _atomic_rename_directory_noreplace as _atomic_pair_restore_noreplace,
    load_risk_sidecar_pair_completion_marker,
    load_risk_sidecar_shard,
    risk_sidecar_pair_completion_marker_path,
    write_risk_sidecar_pair_completion_marker,
    write_risk_sidecar_shard,
)
from src.datasets.shard_writer import (  # noqa: E402
    load_risk_shard,
    write_risk_shard,
)
from src.datasets.split_manager import SPLIT_NAMES  # noqa: E402
from src.generation.paired_variants import (  # noqa: E402
    PairGenerationError,
    PairedVariantConfigError,
    generate_paired_variants,
    load_paired_variant_config,
    summarize_paired_groups,
)
from src.generation.sop05_input_adapter import (  # noqa: E402
    Sop05InputError,
    load_sop03_split_inputs,
    load_sop04_trajectory_bank,
)
from src.generation.sop05_output_loader import (  # noqa: E402
    load_complete_sop05_events,
)
from src.utils.config import ConfigError, load_config  # noqa: E402
from src.utils.seeding import derive_seed  # noqa: E402


SOP07_RISK_DATASET_CLI_VERSION = "sop07_risk_dataset_cli_v4"


class RiskDatasetRunError(ValueError):
    """Raised when a requested SOP-07 publication is unsafe or incomplete."""


@dataclass(frozen=True)
class RiskDatasetRunRequest:
    sop03_root: Path
    sop04_root: Path
    sop04_handoff_digest: str
    sop05_root: Path
    sop05_publication_digest: str
    split: str
    config_path: Path
    paired_config_path: Path
    seed: int
    output_dir: Path
    shard_index: int
    expected_event_count: int
    expected_sample_count: int
    checksum_workers: int
    sidecar_output_dir: Path | None = None


@dataclass(frozen=True)
class _OwnedFilesystemPath:
    path: Path
    device: int
    inode: int
    file_type: int


@dataclass(frozen=True)
class _CompleteRiskSidecarPair:
    risk_shard: object
    sidecar_shard: object
    completion_marker: object


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _lower_sha256(text: str) -> str:
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise argparse.ArgumentTypeError(
            "must be 64 lowercase hexadecimal characters"
        )
    return text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild in-memory SOP06 partial groups from trusted SOP03/04/05 "
            "artifacts and publish one deterministic SOP07 risk shard."
        )
    )
    parser.add_argument("--sop03-root", type=Path, required=True)
    parser.add_argument("--sop04-root", type=Path, required=True)
    parser.add_argument(
        "--sop04-handoff-digest", type=_lower_sha256, required=True
    )
    parser.add_argument("--sop05-root", type=Path, required=True)
    parser.add_argument(
        "--sop05-publication-digest", type=_lower_sha256, required=True
    )
    parser.add_argument("--split", choices=SPLIT_NAMES, required=True)
    parser.add_argument("--config", dest="config_path", type=Path, required=True)
    parser.add_argument(
        "--paired-config",
        dest="paired_config_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=_nonnegative_int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sidecar-output-dir",
        type=Path,
        help=(
            "optional separate immutable root for oracle-only SOP08 labels"
        ),
    )
    parser.add_argument("--shard-index", type=_nonnegative_int, required=True)
    parser.add_argument(
        "--expected-event-count", type=_positive_int, required=True
    )
    parser.add_argument(
        "--expected-sample-count", type=_positive_int, required=True
    )
    parser.add_argument(
        "--checksum-workers", type=_positive_int, default=8
    )
    return parser


def _validate_request(request: RiskDatasetRunRequest) -> None:
    if not isinstance(request, RiskDatasetRunRequest):
        raise TypeError("request must be a RiskDatasetRunRequest")
    if request.split not in SPLIT_NAMES:
        raise RiskDatasetRunError(f"unsupported split: {request.split!r}")
    for name in (
        "seed",
        "shard_index",
        "expected_event_count",
        "expected_sample_count",
        "checksum_workers",
    ):
        value = getattr(request, name)
        minimum = 0 if name in ("seed", "shard_index") else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise RiskDatasetRunError(f"{name} must be an integer >= {minimum}")
    for name in ("sop04_handoff_digest", "sop05_publication_digest"):
        value = getattr(request, name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RiskDatasetRunError(f"{name} must be a lowercase SHA-256")
    if os.path.lexists(request.output_dir):
        raise FileExistsError(
            f"refusing to overwrite immutable shard: {request.output_dir}"
        )
    if request.sidecar_output_dir is not None:
        if not isinstance(request.sidecar_output_dir, Path):
            raise TypeError("sidecar_output_dir must be a Path or None")
        risk_root = request.output_dir.resolve(strict=False)
        sidecar_root = request.sidecar_output_dir.resolve(strict=False)
        if (
            risk_root == sidecar_root
            or risk_root in sidecar_root.parents
            or sidecar_root in risk_root.parents
        ):
            raise RiskDatasetRunError(
                "risk and sidecar output directories must not be nested"
            )
        if os.path.lexists(request.sidecar_output_dir):
            raise FileExistsError(
                "refusing to overwrite immutable sidecar shard: "
                f"{request.sidecar_output_dir}"
            )
        marker_path = risk_sidecar_pair_completion_marker_path(
            request.sidecar_output_dir
        )
        if os.path.lexists(marker_path):
            raise FileExistsError(
                "refusing to overwrite immutable risk/sidecar pair marker: "
                f"{marker_path}"
            )


def _capture_owned_path(
    path: Path, *, expected_file_type: int
) -> _OwnedFilesystemPath:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if expected_file_type == stat.S_IFDIR:
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        file_type = stat.S_IFMT(metadata.st_mode)
        if file_type != expected_file_type:
            raise RiskDatasetRunError(
                f"transaction path has unexpected file type: {path}"
            )
        return _OwnedFilesystemPath(
            path=path,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            file_type=file_type,
        )
    finally:
        os.close(descriptor)


def _same_owned_path(value: _OwnedFilesystemPath) -> bool:
    try:
        actual = _capture_owned_path(
            value.path, expected_file_type=value.file_type
        )
    except (FileNotFoundError, OSError, RiskDatasetRunError):
        return False
    return (
        actual.device == value.device
        and actual.inode == value.inode
        and actual.file_type == value.file_type
    )


def _remove_owned_path(value: _OwnedFilesystemPath) -> bool:
    """Atomically claim and remove only this invocation's inode."""

    quarantine = Path(
        tempfile.mkdtemp(
            prefix=f".{value.path.name}.cleanup-quarantine-",
            dir=value.path.parent,
        )
    )
    quarantine_identity = _capture_owned_path(
        quarantine, expected_file_type=stat.S_IFDIR
    )
    claimed = quarantine / "claimed"
    try:
        try:
            _atomic_pair_cleanup_claim_noreplace(value.path, claimed)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                if not _same_owned_path(quarantine_identity):
                    raise RiskDatasetRunError(
                        "cleanup incomplete: quarantine identity changed"
                    )
                os.rmdir(quarantine)
                return True
            raise
        metadata = claimed.lstat()
        claimed_identity = _OwnedFilesystemPath(
            path=claimed,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            file_type=stat.S_IFMT(metadata.st_mode),
        )
        if (
            claimed_identity.device != value.device
            or claimed_identity.inode != value.inode
            or claimed_identity.file_type != value.file_type
        ):
            try:
                _atomic_pair_restore_noreplace(claimed, value.path)
            except OSError as restore_exc:
                raise RiskDatasetRunError(
                    "cleanup incomplete: competitor could not be restored; "
                    f"preserved at {claimed}"
                ) from restore_exc
            if not _same_owned_path(quarantine_identity):
                raise RiskDatasetRunError(
                    "cleanup incomplete: quarantine identity changed"
                )
            os.rmdir(quarantine)
            return False
        if value.file_type == stat.S_IFDIR:
            shutil.rmtree(claimed)
        elif value.file_type == stat.S_IFREG:
            claimed.unlink()
        else:  # pragma: no cover - capture limits transaction path types
            raise RiskDatasetRunError(
                "cleanup incomplete: unsupported transaction path type"
            )
        if not _same_owned_path(quarantine_identity):
            raise RiskDatasetRunError(
                "cleanup incomplete: quarantine identity changed"
            )
        os.rmdir(quarantine)
        return True
    except BaseException:
        if quarantine.exists() and not any(quarantine.iterdir()):
            try:
                if _same_owned_path(quarantine_identity):
                    os.rmdir(quarantine)
            except OSError:
                pass
        raise


def _make_pair_staging_target(final_path: Path) -> tuple[_OwnedFilesystemPath, Path]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    container = Path(
        tempfile.mkdtemp(
            prefix=f".{final_path.name}.pair-staging-",
            dir=final_path.parent,
        )
    )
    owned_container = _capture_owned_path(
        container, expected_file_type=stat.S_IFDIR
    )
    return owned_container, container / final_path.name


def _commit_staged_path(
    source: Path,
    destination: Path,
    *,
    expected_file_type: int,
) -> _OwnedFilesystemPath:
    source_identity = _capture_owned_path(
        source, expected_file_type=expected_file_type
    )
    _atomic_pair_commit_noreplace(source, destination)
    try:
        destination_identity = _capture_owned_path(
            destination, expected_file_type=expected_file_type
        )
    except BaseException:
        try:
            _atomic_pair_restore_noreplace(destination, source)
        except OSError as restore_exc:
            raise RiskDatasetRunError(
                "publication cleanup incomplete: committed path could not be restored"
            ) from restore_exc
        raise
    if (
        destination_identity.device != source_identity.device
        or destination_identity.inode != source_identity.inode
        or destination_identity.file_type != source_identity.file_type
    ):
        try:
            _atomic_pair_restore_noreplace(destination, source)
        except OSError as restore_exc:
            raise RiskDatasetRunError(
                "publication cleanup incomplete: raced path could not be restored"
            ) from restore_exc
        raise RiskDatasetRunError(
            "staging identity changed during no-replace publication"
        )
    return _OwnedFilesystemPath(
        path=destination,
        device=source_identity.device,
        inode=source_identity.inode,
        file_type=source_identity.file_type,
    )


def _load_complete_risk_sidecar_pair(
    *,
    request: RiskDatasetRunRequest,
    grid: object,
    risk_identity: _OwnedFilesystemPath,
    sidecar_identity: _OwnedFilesystemPath,
    marker_identity: _OwnedFilesystemPath,
) -> _CompleteRiskSidecarPair:
    """Reload and bind one final marker plus both exact final-root inodes."""

    if request.sidecar_output_dir is None:  # pragma: no cover - private contract
        raise RuntimeError("complete pair load requires sidecar_output_dir")
    for owned, name in (
        (risk_identity, "risk root"),
        (sidecar_identity, "sidecar root"),
        (marker_identity, "pair marker"),
    ):
        if not _same_owned_path(owned):
            raise ValueError(f"final {name} identity changed before complete load")

    with pin_risk_shard_snapshot(
        request.output_dir, grid=grid
    ) as pinned_risk:
        risk_shard = pinned_risk.loaded_shard
        sample_ids = tuple(sample.sample_id for sample in risk_shard.samples)
        if len(sample_ids) != request.expected_sample_count:
            raise ValueError("complete pair risk sample count mismatch")
        risk_summary = getattr(risk_shard, "summary", None)
        if not isinstance(risk_summary, Mapping):
            raise ValueError("complete pair risk summary is unavailable")
        if (
            risk_summary.get("split") != request.split
            or risk_summary.get("shard_index") != request.shard_index
        ):
            raise ValueError("complete pair risk split/index mismatch")

        sidecar_shard = load_risk_sidecar_shard(
            request.sidecar_output_dir,
            grid=grid,
            expected_sample_ids=sample_ids,
            expected_source_risk_shard_semantic_digest=(
                risk_shard.semantic_digest
            ),
        )
        if (
            sidecar_shard.sample_ids != sample_ids
            or sidecar_shard.split != request.split
            or sidecar_shard.shard_index != request.shard_index
        ):
            raise ValueError("complete pair sidecar identity mismatch")

        marker_path = risk_sidecar_pair_completion_marker_path(
            request.sidecar_output_dir
        )
        marker = load_risk_sidecar_pair_completion_marker(
            marker_path,
            expected_risk_root=request.output_dir,
            expected_sidecar_root=request.sidecar_output_dir,
            expected_split=request.split,
            expected_shard_index=request.shard_index,
            expected_sample_ids=sample_ids,
            expected_risk_shard_semantic_digest=(
                risk_shard.semantic_digest
            ),
            expected_sidecar_shard_semantic_digest=(
                sidecar_shard.semantic_digest
            ),
        )
        for owned, name in (
            (risk_identity, "risk root"),
            (sidecar_identity, "sidecar root"),
            (marker_identity, "pair marker"),
        ):
            if not _same_owned_path(owned):
                raise ValueError(
                    f"final {name} identity changed during complete load"
                )
        complete_pair = _CompleteRiskSidecarPair(
            risk_shard=risk_shard,
            sidecar_shard=sidecar_shard,
            completion_marker=marker,
        )
        pinned_risk.verify_unchanged()
        return complete_pair


def _publish_risk_sidecar_pair(
    *,
    request: RiskDatasetRunRequest,
    sample_values: tuple[object, ...],
    sidecars: tuple[object, ...],
    grid: object,
):
    """Publish both immutable roots and their external marker as one protocol."""

    if request.sidecar_output_dir is None:  # pragma: no cover - private contract
        raise RuntimeError("paired publication requires sidecar_output_dir")
    marker_final = risk_sidecar_pair_completion_marker_path(
        request.sidecar_output_dir
    )
    staging_containers: list[_OwnedFilesystemPath] = []
    published_paths: list[_OwnedFilesystemPath] = []
    try:
        risk_container, risk_staging = _make_pair_staging_target(
            request.output_dir
        )
        staging_containers.append(risk_container)
        sidecar_container, sidecar_staging = _make_pair_staging_target(
            request.sidecar_output_dir
        )
        staging_containers.append(sidecar_container)

        write_risk_shard(
            sample_values,
            risk_staging,
            grid=grid,
            shard_index=request.shard_index,
            expected_sample_count=request.expected_sample_count,
        )
        staged_risk = load_risk_shard(risk_staging, grid=grid)
        if len(staged_risk.samples) != request.expected_sample_count:
            raise ValueError("formal risk staging reload count mismatch")
        staged_ids = tuple(
            sample.sample_id for sample in staged_risk.samples
        )

        write_risk_sidecar_shard(
            sidecars,
            sidecar_staging,
            grid=grid,
            split=request.split,
            shard_index=request.shard_index,
            source_risk_shard_semantic_digest=staged_risk.semantic_digest,
        )
        staged_sidecars = load_risk_sidecar_shard(
            sidecar_staging,
            grid=grid,
            expected_sample_ids=staged_ids,
            expected_source_risk_shard_semantic_digest=(
                staged_risk.semantic_digest
            ),
        )
        if staged_sidecars.sample_ids != staged_ids:
            raise ValueError("formal sidecar staging reload sample IDs mismatch")

        risk_published = _commit_staged_path(
            risk_staging,
            request.output_dir,
            expected_file_type=stat.S_IFDIR,
        )
        published_paths.append(risk_published)
        sidecar_published = _commit_staged_path(
            sidecar_staging,
            request.sidecar_output_dir,
            expected_file_type=stat.S_IFDIR,
        )
        published_paths.append(sidecar_published)

        final_risk = load_risk_shard(request.output_dir, grid=grid)
        final_ids = tuple(sample.sample_id for sample in final_risk.samples)
        if (
            final_risk.manifest_digest != staged_risk.manifest_digest
            or final_risk.semantic_digest != staged_risk.semantic_digest
            or final_ids != staged_ids
        ):
            raise ValueError("final risk shard differs from verified staging")
        final_sidecars = load_risk_sidecar_shard(
            request.sidecar_output_dir,
            grid=grid,
            expected_sample_ids=final_ids,
            expected_source_risk_shard_semantic_digest=(
                final_risk.semantic_digest
            ),
        )
        if final_sidecars.semantic_digest != staged_sidecars.semantic_digest:
            raise ValueError("final sidecar shard differs from verified staging")

        marker_container, marker_staging = _make_pair_staging_target(
            marker_final
        )
        staging_containers.append(marker_container)
        marker_kwargs = {
            "expected_risk_root": request.output_dir,
            "expected_sidecar_root": request.sidecar_output_dir,
            "expected_split": request.split,
            "expected_shard_index": request.shard_index,
            "expected_sample_ids": final_ids,
            "expected_risk_shard_semantic_digest": (
                final_risk.semantic_digest
            ),
            "expected_sidecar_shard_semantic_digest": (
                final_sidecars.semantic_digest
            ),
        }
        write_risk_sidecar_pair_completion_marker(
            marker_staging,
            risk_root=request.output_dir,
            sidecar_root=request.sidecar_output_dir,
            split=request.split,
            shard_index=request.shard_index,
            sample_ids=final_ids,
            risk_shard_semantic_digest=final_risk.semantic_digest,
            sidecar_shard_semantic_digest=final_sidecars.semantic_digest,
        )
        staged_marker = load_risk_sidecar_pair_completion_marker(
            marker_staging,
            **marker_kwargs,
        )
        marker_published = _commit_staged_path(
            marker_staging,
            marker_final,
            expected_file_type=stat.S_IFREG,
        )
        published_paths.append(marker_published)
        complete_pair = _load_complete_risk_sidecar_pair(
            request=request,
            grid=grid,
            risk_identity=risk_published,
            sidecar_identity=sidecar_published,
            marker_identity=marker_published,
        )
        if (
            complete_pair.completion_marker.marker_digest_sha256
            != staged_marker.marker_digest_sha256
        ):
            raise ValueError("final pair marker differs from verified staging")

        for container in reversed(staging_containers):
            if not _remove_owned_path(container):
                raise RiskDatasetRunError(
                    "transaction staging cleanup incomplete: identity changed"
                )
        return (
            complete_pair.risk_shard,
            complete_pair.sidecar_shard,
            complete_pair.completion_marker,
            marker_final,
        )
    except BaseException as exc:
        cleanup_errors: list[str] = []
        for owned in reversed(published_paths):
            try:
                if not _remove_owned_path(owned):
                    cleanup_errors.append(
                        f"{owned.path}: identity changed before cleanup"
                    )
            except (OSError, RiskDatasetRunError) as cleanup_exc:
                cleanup_errors.append(f"{owned.path}: {cleanup_exc}")
        for container in reversed(staging_containers):
            try:
                if not _remove_owned_path(container):
                    cleanup_errors.append(
                        f"{container.path}: identity changed before cleanup"
                    )
            except (OSError, RiskDatasetRunError) as cleanup_exc:
                cleanup_errors.append(f"{container.path}: {cleanup_exc}")
        if not isinstance(exc, Exception):
            if cleanup_errors:
                exc.add_note(
                    "cleanup incomplete: " + "; ".join(cleanup_errors)
                )
            raise
        detail = f"paired risk/sidecar publication failed: {exc}"
        if cleanup_errors:
            detail += "; cleanup incomplete: " + "; ".join(cleanup_errors)
        raise RiskDatasetRunError(detail) from exc


def _producer_evidence_payload(evidence: object) -> dict[str, object]:
    fields = (
        "code_commit",
        "checksum_manifest_sha256",
        "audit_sha256",
        "completion_policy",
    )
    payload = {name: getattr(evidence, name, None) for name in fields}
    if any(not isinstance(value, str) or not value for value in payload.values()):
        raise RiskDatasetRunError("reloaded upstream producer evidence is invalid")
    return payload


def _sop04_evidence_payload(bank: object) -> dict[str, object]:
    return {
        **_producer_evidence_payload(bank.producer_evidence),
        "trajectory_bank_version": bank.trajectory_bank_version,
        "pose_time_layout_version": bank.pose_time_layout_version,
        "trajectory_steps": 15,
        "dt_s": 0.2,
        "first_pose_time_s": 0.2,
        "last_pose_time_s": 3.0,
        "pose_time_offsets_sha256": bank.pose_time_offsets_sha256,
        "bank_semantic_digest_sha256": bank.bank_semantic_digest_sha256,
        "external_handoff_digest_sha256": bank.external_handoff_digest_sha256,
    }


def _validate_sop05_input_lock(
    loaded_sop05: object,
    *,
    sop03: object,
    sop04: object,
    split: str,
) -> None:
    manifest = getattr(loaded_sop05, "run_manifest", None)
    if not isinstance(manifest, Mapping):
        raise RiskDatasetRunError("SOP05 run manifest is unavailable")
    input_lock = manifest.get("input_lock")
    if not isinstance(input_lock, Mapping):
        raise RiskDatasetRunError("SOP05 input_lock is unavailable")
    if input_lock.get("version") != "sop05_input_lock_v2":
        raise RiskDatasetRunError("SOP05 input_lock version mismatch")
    if input_lock.get("split") != split:
        raise RiskDatasetRunError("SOP05 input_lock split mismatch")
    if input_lock.get("sop03") != _producer_evidence_payload(
        sop03.producer_evidence
    ):
        raise RiskDatasetRunError(
            "SOP05 input_lock SOP03 evidence differs from reloaded SOP03"
        )
    if input_lock.get("sop04") != _sop04_evidence_payload(sop04):
        raise RiskDatasetRunError(
            "SOP05 input_lock SOP04 evidence differs from reloaded SOP04"
        )


def _snippet_index(sop03: object) -> dict[tuple[str, str, str], tuple[object, ...]]:
    libraries = getattr(sop03, "typed_libraries", None)
    if not isinstance(libraries, Mapping):
        raise RiskDatasetRunError("SOP03 typed snippet libraries are unavailable")
    grouped: dict[tuple[str, str, str], list[object]] = {}
    for object_type in sorted(libraries):
        library = libraries[object_type]
        snippets = getattr(library, "snippets", None)
        if not isinstance(snippets, tuple):
            raise RiskDatasetRunError("SOP03 snippet library is malformed")
        for snippet in snippets:
            key = (
                getattr(snippet, "snippet_id", None),
                getattr(snippet, "source_object_id", None),
                getattr(snippet, "object_type", None),
            )
            if not all(isinstance(value, str) and value for value in key):
                raise RiskDatasetRunError("SOP03 snippet identity is malformed")
            grouped.setdefault(key, []).append(snippet)
    return {key: tuple(values) for key, values in grouped.items()}


def _source_snippet(
    event: object,
    index: Mapping[tuple[str, str, str], tuple[object, ...]],
) -> object:
    record = event.target_motion_record
    key = (
        record.source_snippet_id,
        record.source_object_id,
        record.object_type,
    )
    matches = index.get(key, ())
    if len(matches) != 1:
        raise RiskDatasetRunError(
            "SOP05 event source snippet does not uniquely match reloaded SOP03"
        )
    snippet = matches[0]
    for name in ("source_recording_id", "source_session_id"):
        value = getattr(snippet, name, None)
        if not isinstance(value, str) or not value:
            raise RiskDatasetRunError(f"source snippet {name} is invalid")
    target = getattr(event, "target", None)
    provenance = getattr(target, "provenance", None)
    if not isinstance(provenance, Mapping) or provenance.get(
        "source_recording_id"
    ) != snippet.source_recording_id:
        raise RiskDatasetRunError(
            "SOP05 event source recording differs from reloaded source snippet"
        )
    return snippet


def _event_identity_join(event: object, sop03: object, sop04: object) -> None:
    event_id = getattr(event, "generated_event_id", None)
    record = getattr(event, "target_motion_record", None)
    if not isinstance(event_id, str) or not event_id or record is None:
        raise RiskDatasetRunError("SOP05 event identity is malformed")
    if getattr(record, "generated_event_id", None) != event_id:
        raise RiskDatasetRunError("SOP05 event/record generated_event_id mismatch")
    if record.base_state_id not in sop03.manifest_index:
        raise RiskDatasetRunError("SOP05 event base_state_id is absent from SOP03")
    if record.trajectory_id not in sop04.by_id:
        raise RiskDatasetRunError("SOP05 event trajectory_id is absent from SOP04")


def _class_prior(samples: tuple[object, ...]) -> dict[str, object]:
    count = len(samples)
    collision = sum(int(sample.collision_label) for sample in samples)
    near_miss = sum(int(sample.near_miss) for sample in samples)
    if any(
        sample.collision_label not in (0, 1)
        or sample.near_miss not in (0, 1)
        or (sample.collision_label and sample.near_miss)
        for sample in samples
    ):
        raise RiskDatasetRunError("risk samples contain invalid class labels")
    safe = count - collision - near_miss
    return {
        "collision": {"count": collision, "rate": collision / count},
        "near_miss": {"count": near_miss, "rate": near_miss / count},
        "safe": {"count": safe, "rate": safe / count},
    }


def _source_coverage(
    accepted: tuple[tuple[object, object], ...],
) -> dict[str, object]:
    recordings = {snippet.source_recording_id for _, snippet in accepted}
    sessions = {snippet.source_session_id for _, snippet in accepted}
    snippets = {snippet.snippet_id for _, snippet in accepted}
    object_types: Counter[str] = Counter()
    footprint_kinds: Counter[str] = Counter()
    for event, _ in accepted:
        record = event.target_motion_record
        object_types[record.object_type] += 1
        spec = record.footprint_spec
        footprint = spec.get("footprint") if isinstance(spec, Mapping) else None
        kind = footprint.get("kind") if isinstance(footprint, Mapping) else None
        if not isinstance(kind, str) or not kind:
            raise RiskDatasetRunError("SOP05 target footprint kind is invalid")
        footprint_kinds[kind] += 1
    return {
        "accepted_event_count": len(accepted),
        "unique_source_recording_count": len(recordings),
        "unique_source_session_count": len(sessions),
        "unique_source_snippet_count": len(snippets),
        "object_type_counts": dict(sorted(object_types.items())),
        "footprint_kind_counts": dict(sorted(footprint_kinds.items())),
    }


def _load_inputs(request: RiskDatasetRunRequest):
    base_config = load_config(request.config_path)
    grid = build_grid_spec(base_config)
    paired_config = load_paired_variant_config(request.paired_config_path)
    sop03 = load_sop03_split_inputs(
        request.sop03_root,
        request.split,
        grid,
        checksum_workers=request.checksum_workers,
    )
    sop04 = load_sop04_trajectory_bank(
        request.sop04_root,
        grid,
        expected_external_handoff_digest_sha256=(
            request.sop04_handoff_digest
        ),
        checksum_workers=request.checksum_workers,
    )
    try:
        sop05 = load_complete_sop05_events(
            request.sop05_root,
            grid=grid,
            expected_publication_semantic_digest=(
                request.sop05_publication_digest
            ),
        )
    except ValueError as exc:
        raise RiskDatasetRunError(f"failed to load SOP05 publication: {exc}") from exc
    if sop03.split != request.split or sop05.split != request.split:
        raise RiskDatasetRunError("reloaded upstream split mismatch")
    _validate_sop05_input_lock(
        sop05,
        sop03=sop03,
        sop04=sop04,
        split=request.split,
    )
    return base_config, grid, paired_config, sop03, sop04, sop05


def run_risk_dataset(request: RiskDatasetRunRequest) -> dict[str, object]:
    """Execute one deterministic, exact-count SOP-07 publication."""

    _validate_request(request)
    base_config, grid, paired_config, sop03, sop04, sop05 = _load_inputs(request)
    events = tuple(
        sorted(sop05.events, key=lambda event: event.generated_event_id)
    )
    event_ids = tuple(event.generated_event_id for event in events)
    if len(event_ids) != len(set(event_ids)):
        raise RiskDatasetRunError("SOP05 generated_event_id values are not unique")
    if len(events) != request.expected_event_count:
        raise RiskDatasetRunError(
            "expected_event_count does not match verified SOP05 event count"
        )

    snippet_index = _snippet_index(sop03)
    samples: list[object] = []
    sidecars: list[object] = []
    groups: list[object] = []
    accepted_sources: list[tuple[object, object]] = []
    rejection_reasons: Counter[str] = Counter()
    for event in events:
        _event_identity_join(event, sop03, sop04)
        record = event.target_motion_record
        snippet = _source_snippet(event, snippet_index)
        base_state, oracle_context = sop03.load_pair(record.base_state_id, grid)
        trajectory = sop04.by_id[record.trajectory_id]
        pair_seed = derive_seed(
            request.seed,
            "sop07-paired-variants",
            request.split,
            event.generated_event_id,
        )
        try:
            group = generate_paired_variants(
                mother_event=event,
                source_snippet=snippet,
                base_state=base_state,
                trajectory=trajectory,
                oracle_context=oracle_context,
                base_config=base_config,
                paired_config=paired_config,
                seed=pair_seed,
            )
        except PairGenerationError as exc:
            rejection_reasons[exc.reason] += 1
            continue
        groups.append(group)
        accepted_sources.append((event, snippet))
        try:
            adapter_kwargs = {
                "group": group,
                "mother_event": event,
                "source_snippet": snippet,
                "base_state": base_state,
                "trajectory": trajectory,
                "oracle_context": oracle_context,
                "base_config": base_config,
                "paired_config": paired_config,
                "risk_config": base_config["risk_gt"],
                "dataset_seed": request.seed,
            }
            if request.sidecar_output_dir is None:
                group_samples = build_risk_samples_from_sop06_group(
                    **adapter_kwargs
                )
                group_sidecars: tuple[object, ...] = ()
            else:
                group_samples, group_sidecars = (
                    build_risk_samples_and_sidecars_from_sop06_group(
                        **adapter_kwargs
                    )
                )
        except (TypeError, ValueError) as exc:
            raise RiskDatasetRunError(
                "failed to atomically assemble risk samples for "
                f"{event.generated_event_id}: {exc}"
            ) from exc
        samples.extend(group_samples)
        sidecars.extend(group_sidecars)

    sample_values = tuple(samples)
    if len(sample_values) != request.expected_sample_count:
        reason_payload = json.dumps(
            dict(sorted(rejection_reasons.items())),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        raise RiskDatasetRunError(
            "expected_sample_count does not match assembled SOP06 variants; "
            f"expected={request.expected_sample_count}; actual={len(sample_values)}; "
            f"pair rejection reasons={reason_payload}"
        )
    sample_ids = tuple(sample.sample_id for sample in sample_values)
    if len(sample_ids) != len(set(sample_ids)):
        raise RiskDatasetRunError("assembled RiskSample IDs are not unique")
    if request.sidecar_output_dir is not None:
        sidecar_ids = tuple(sidecar.sample_id for sidecar in sidecars)
        if sidecar_ids != sample_ids:
            raise RiskDatasetRunError(
                "assembled sidecars do not align with RiskSample IDs"
            )

    loaded_sidecars = None
    loaded_pair_marker = None
    pair_marker_path = None
    if request.sidecar_output_dir is None:
        try:
            write_risk_shard(
                sample_values,
                request.output_dir,
                grid=grid,
                shard_index=request.shard_index,
                expected_sample_count=request.expected_sample_count,
            )
            loaded = load_risk_shard(request.output_dir, grid=grid)
        except (FileExistsError, TypeError, ValueError) as exc:
            raise RiskDatasetRunError(
                f"risk shard publication failed: {exc}"
            ) from exc
        if len(loaded.samples) != request.expected_sample_count:
            raise RiskDatasetRunError(
                "formal risk shard reload count mismatch"
            )
    else:
        (
            loaded,
            loaded_sidecars,
            loaded_pair_marker,
            pair_marker_path,
        ) = _publish_risk_sidecar_pair(
            request=request,
            sample_values=sample_values,
            sidecars=tuple(sidecars),
            grid=grid,
        )

    rejection_report = {
        "attempted_event_count": len(events),
        "accepted_group_count": len(groups),
        "rejected_event_count": sum(rejection_reasons.values()),
        "reason_counts": dict(sorted(rejection_reasons.items())),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "producer_version": SOP07_RISK_DATASET_CLI_VERSION,
        "split": request.split,
        "seed": request.seed,
        "shard_index": request.shard_index,
        "output_dir": str(request.output_dir),
        "event_count": len(events),
        "sample_count": len(sample_values),
        "rejection_report": rejection_report,
        "class_prior": _class_prior(sample_values),
        "pair_coverage": summarize_paired_groups(tuple(groups)),
        "source_coverage": _source_coverage(tuple(accepted_sources)),
        "manifest_digest": loaded.manifest_digest,
        "semantic_digest": loaded.semantic_digest,
    }
    if loaded_sidecars is not None:
        if loaded_pair_marker is None or pair_marker_path is None:
            raise RuntimeError("complete paired publication lacks its marker")
        report.update(
            {
                "publication_status": "complete",
                "sidecar_output_dir": str(request.sidecar_output_dir),
                "risk_shard_semantic_digest": loaded.semantic_digest,
                "sidecar_shard_semantic_digest": (
                    loaded_sidecars.semantic_digest
                ),
                "pair_completion_marker_path": str(pair_marker_path),
                "pair_completion_marker_digest": (
                    loaded_pair_marker.marker_digest_sha256
                ),
            }
        )
    json.dumps(report, sort_keys=True, allow_nan=False)
    return report


_EXPECTED_INPUT_ERRORS = (
    RiskDatasetRunError,
    Sop05InputError,
    ConfigError,
    PairedVariantConfigError,
    ContractError,
    FileExistsError,
    OSError,
    yaml.YAMLError,
)


def main() -> int:
    args = _parser().parse_args()
    request = RiskDatasetRunRequest(
        sop03_root=args.sop03_root,
        sop04_root=args.sop04_root,
        sop04_handoff_digest=args.sop04_handoff_digest,
        sop05_root=args.sop05_root,
        sop05_publication_digest=args.sop05_publication_digest,
        split=args.split,
        config_path=args.config_path,
        paired_config_path=args.paired_config_path,
        seed=args.seed,
        output_dir=args.output_dir,
        shard_index=args.shard_index,
        expected_event_count=args.expected_event_count,
        expected_sample_count=args.expected_sample_count,
        checksum_workers=args.checksum_workers,
        sidecar_output_dir=args.sidecar_output_dir,
    )
    try:
        report = run_risk_dataset(request)
    except _EXPECTED_INPUT_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
