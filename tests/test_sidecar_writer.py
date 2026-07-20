"""Immutable storage tests for SOP-08 risk-label sidecars."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.contracts import GridSpec
from src.datasets import sidecar_writer as writer_module
from src.datasets.sidecar_writer import (
    RISK_SIDECAR_SHARD_LAYOUT_VERSION,
    load_risk_sidecar_shard,
    write_risk_sidecar_pair_completion_marker,
    write_risk_sidecar_shard,
)
from src.generation.risk_sidecars import RiskLabelSidecar


SOURCE_DIGEST = "a" * 64


def _grid() -> GridSpec:
    return GridSpec(
        height=5,
        width=7,
        history_steps=8,
        future_steps=15,
        resolution_m=0.25,
    )


def _sidecar(
    sample_id: str, *, offset: int, grid: GridSpec | None = None
) -> RiskLabelSidecar:
    grid = _grid() if grid is None else grid
    hidden = np.zeros(
        (grid.future_steps, grid.height, grid.width), dtype=np.uint8
    )
    robot = np.zeros_like(hidden)
    for index in range(grid.future_steps):
        hidden[index, (index + offset) % grid.height, offset % grid.width] = 1
        robot[index, index % grid.height, (index + offset + 1) % grid.width] = 1
    times = (
        np.arange(1, grid.future_steps + 1, dtype=np.float32)
        * np.float32(0.2)
    )
    return RiskLabelSidecar(
        sample_id=sample_id,
        hidden_risk_occupancy=hidden,
        robot_future_footprints=robot,
        future_endpoint_times_s=times,
    )


def _write_custom(
    root: Path,
    *,
    sidecars: tuple[RiskLabelSidecar, ...] | None = None,
    grid: GridSpec | None = None,
    split: str = "train",
    shard_index: int = 7,
    source_digest: str = SOURCE_DIGEST,
):
    grid = _grid() if grid is None else grid
    values = (
        (
            _sidecar("sample-b", offset=2, grid=grid),
            _sidecar("sample-a", offset=1, grid=grid),
        )
        if sidecars is None
        else sidecars
    )
    return write_risk_sidecar_shard(
        values,
        root,
        grid=grid,
        split=split,
        shard_index=shard_index,
        source_risk_shard_semantic_digest=source_digest,
    )


def _write(root: Path):
    return _write_custom(root)


def test_sidecar_shard_round_trip_binds_ids_grid_endpoints_and_source(tmp_path: Path) -> None:
    root = tmp_path / "sidecar-shard"

    paths = _write(root)
    loaded = load_risk_sidecar_shard(
        root,
        grid=_grid(),
        expected_sample_ids=("sample-a", "sample-b"),
        expected_source_risk_shard_semantic_digest=SOURCE_DIGEST,
    )

    assert set(paths) == {"directory", "payload", "summary"}
    assert {path.name for path in root.iterdir()} == {"sidecars.npz", "summary.json"}
    assert loaded.sample_ids == ("sample-a", "sample-b")
    assert loaded.split == "train"
    assert loaded.shard_index == 7
    assert loaded.source_risk_shard_semantic_digest == SOURCE_DIGEST
    assert len(loaded.semantic_digest) == 64
    assert loaded.hidden_risk_occupancy.shape == (2, 15, 5, 7)
    assert loaded.robot_future_footprints.shape == (2, 15, 5, 7)
    assert loaded.hidden_risk_occupancy.dtype == np.float32
    assert loaded.robot_future_footprints.dtype == np.float32
    assert loaded.future_endpoint_times_s.dtype == np.float32
    assert not loaded.hidden_risk_occupancy.flags.writeable
    assert not loaded.robot_future_footprints.flags.writeable
    assert not loaded.future_endpoint_times_s.flags.writeable
    np.testing.assert_array_equal(
        loaded.future_endpoint_times_s,
        np.arange(1, 16, dtype=np.float32) * np.float32(0.2),
    )
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["layout_version"] == RISK_SIDECAR_SHARD_LAYOUT_VERSION
    assert summary["sample_ids"] == ["sample-a", "sample-b"]
    assert summary["source_risk_shard_semantic_digest"] == SOURCE_DIGEST
    assert summary["grid"] == {
        "future_steps": 15,
        "height": 5,
        "resolution_m": 0.25,
        "width": 7,
    }
    assert summary["array_layout"]["hidden_risk_occupancy"] == {
        "dtype": "|u1",
        "nbytes": 1050,
        "order": "C",
        "shape": [2, 15, 5, 7],
    }

    with np.load(root / "sidecars.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "hidden_risk_occupancy",
            "robot_future_footprints",
            "future_endpoint_times_s",
        }
        assert archive["hidden_risk_occupancy"].dtype == np.uint8
        assert archive["robot_future_footprints"].dtype == np.uint8
        assert archive["future_endpoint_times_s"].dtype == np.float32
        assert set(np.unique(archive["hidden_risk_occupancy"])).issubset({0, 1})


def test_sidecar_semantic_digest_is_deterministic(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write(first_root)
    _write(second_root)

    first = load_risk_sidecar_shard(
        first_root,
        grid=_grid(),
        expected_sample_ids=("sample-a", "sample-b"),
        expected_source_risk_shard_semantic_digest=SOURCE_DIGEST,
    )
    second = load_risk_sidecar_shard(
        second_root,
        grid=_grid(),
        expected_sample_ids=("sample-a", "sample-b"),
        expected_source_risk_shard_semantic_digest=SOURCE_DIGEST,
    )

    assert first.semantic_digest == second.semantic_digest
    np.testing.assert_array_equal(
        first.hidden_risk_occupancy, second.hidden_risk_occupancy
    )


def test_sidecar_semantic_digest_binds_identity_source_split_index_grid_and_bytes(
    tmp_path: Path,
) -> None:
    baseline_root = tmp_path / "baseline"
    _write(baseline_root)
    baseline = load_risk_sidecar_shard(
        baseline_root,
        grid=_grid(),
        expected_sample_ids=("sample-a", "sample-b"),
        expected_source_risk_shard_semantic_digest=SOURCE_DIGEST,
    )

    renamed = (
        _sidecar("sample-c", offset=2),
        _sidecar("sample-a", offset=1),
    )
    changed_hidden = _sidecar("sample-b", offset=2)
    changed_mask = changed_hidden.hidden_risk_occupancy.copy()
    changed_mask[0, 0, 0] ^= np.uint8(1)
    single_pixel = replace(
        changed_hidden, hidden_risk_occupancy=changed_mask
    )
    larger_grid = GridSpec(
        height=6,
        width=7,
        history_steps=8,
        future_steps=15,
        resolution_m=0.25,
    )
    cases = (
        {
            "name": "ordered-ids",
            "sidecars": renamed,
            "grid": _grid(),
            "split": "train",
            "shard_index": 7,
            "source_digest": SOURCE_DIGEST,
            "expected_ids": ("sample-a", "sample-c"),
        },
        {
            "name": "source",
            "sidecars": None,
            "grid": _grid(),
            "split": "train",
            "shard_index": 7,
            "source_digest": "b" * 64,
            "expected_ids": ("sample-a", "sample-b"),
        },
        {
            "name": "split",
            "sidecars": None,
            "grid": _grid(),
            "split": "val",
            "shard_index": 7,
            "source_digest": SOURCE_DIGEST,
            "expected_ids": ("sample-a", "sample-b"),
        },
        {
            "name": "index",
            "sidecars": None,
            "grid": _grid(),
            "split": "train",
            "shard_index": 8,
            "source_digest": SOURCE_DIGEST,
            "expected_ids": ("sample-a", "sample-b"),
        },
        {
            "name": "grid",
            "sidecars": None,
            "grid": larger_grid,
            "split": "train",
            "shard_index": 7,
            "source_digest": SOURCE_DIGEST,
            "expected_ids": ("sample-a", "sample-b"),
        },
        {
            "name": "single-pixel",
            "sidecars": (single_pixel, _sidecar("sample-a", offset=1)),
            "grid": _grid(),
            "split": "train",
            "shard_index": 7,
            "source_digest": SOURCE_DIGEST,
            "expected_ids": ("sample-a", "sample-b"),
        },
    )

    for case in cases:
        root = tmp_path / str(case["name"])
        _write_custom(
            root,
            sidecars=case["sidecars"],
            grid=case["grid"],
            split=str(case["split"]),
            shard_index=int(case["shard_index"]),
            source_digest=str(case["source_digest"]),
        )
        loaded = load_risk_sidecar_shard(
            root,
            grid=case["grid"],
            expected_sample_ids=case["expected_ids"],
            expected_source_risk_shard_semantic_digest=str(
                case["source_digest"]
            ),
        )
        assert loaded.semantic_digest != baseline.semantic_digest, case["name"]


def test_sidecar_loader_rejects_layout_version_tampering(tmp_path: Path) -> None:
    root = tmp_path / "layout-tamper"
    _write(root)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["layout_version"] = "risk_label_sidecar_shard_v2"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported sidecar shard layout"):
        load_risk_sidecar_shard(
            root,
            grid=_grid(),
            expected_sample_ids=("sample-a", "sample-b"),
            expected_source_risk_shard_semantic_digest=SOURCE_DIGEST,
        )


def test_sidecar_loader_rejects_reorder_extra_files_and_source_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sidecar-shard"
    _write(root)

    with pytest.raises(ValueError, match="ordered sample IDs"):
        load_risk_sidecar_shard(
            root,
            grid=_grid(),
            expected_sample_ids=("sample-b", "sample-a"),
            expected_source_risk_shard_semantic_digest=SOURCE_DIGEST,
        )
    with pytest.raises(ValueError, match="source risk shard semantic digest"):
        load_risk_sidecar_shard(
            root,
            grid=_grid(),
            expected_sample_ids=("sample-a", "sample-b"),
            expected_source_risk_shard_semantic_digest="b" * 64,
        )
    (root / "unexpected.txt").write_text("not part of the layout", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected sidecar shard files"):
        load_risk_sidecar_shard(
            root,
            grid=_grid(),
            expected_sample_ids=("sample-a", "sample-b"),
            expected_source_risk_shard_semantic_digest=SOURCE_DIGEST,
        )


def test_sidecar_loader_rejects_payload_and_summary_tampering(tmp_path: Path) -> None:
    payload_root = tmp_path / "payload-tamper"
    _write(payload_root)
    with np.load(payload_root / "sidecars.npz", allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["hidden_risk_occupancy"][0, 0, 0, 0] ^= np.uint8(1)
    with (payload_root / "sidecars.npz").open("wb") as handle:
        np.savez_compressed(handle, **payload)
    with pytest.raises(ValueError, match="payload SHA-256"):
        load_risk_sidecar_shard(
            payload_root,
            grid=_grid(),
            expected_sample_ids=("sample-a", "sample-b"),
            expected_source_risk_shard_semantic_digest=SOURCE_DIGEST,
        )

    summary_root = tmp_path / "summary-tamper"
    _write(summary_root)
    summary_path = summary_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["future_endpoint_times_s"][0] = 0.0
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="future endpoint"):
        load_risk_sidecar_shard(
            summary_root,
            grid=_grid(),
            expected_sample_ids=("sample-a", "sample-b"),
            expected_source_risk_shard_semantic_digest=SOURCE_DIGEST,
        )


def test_sidecar_loader_rejects_symlink_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-sidecar-shard"
    linked_root = tmp_path / "linked-sidecar-shard"
    _write(real_root)
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        load_risk_sidecar_shard(
            linked_root,
            grid=_grid(),
            expected_sample_ids=("sample-a", "sample-b"),
            expected_source_risk_shard_semantic_digest=SOURCE_DIGEST,
        )


@pytest.mark.parametrize("member_name", ["sidecars.npz", "summary.json"])
def test_sidecar_loader_rejects_symlink_members(
    tmp_path: Path, member_name: str
) -> None:
    root = tmp_path / f"linked-{member_name}"
    _write(root)
    member = root / member_name
    real_member = tmp_path / f"real-{member_name}"
    member.rename(real_member)
    member.symlink_to(real_member)

    with pytest.raises(ValueError, match="symlink"):
        load_risk_sidecar_shard(
            root,
            grid=_grid(),
            expected_sample_ids=("sample-a", "sample-b"),
            expected_source_risk_shard_semantic_digest=SOURCE_DIGEST,
        )


def test_sidecar_writer_refuses_overwrite_and_cleans_failed_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sidecar-shard"
    _write(root)
    with pytest.raises(FileExistsError, match="overwrite"):
        _write(root)

    failed_root = tmp_path / "failed-sidecar-shard"

    def fail_reload(*args, **kwargs):
        raise ValueError("forced staging reload failure")

    monkeypatch.setattr(writer_module, "load_risk_sidecar_shard", fail_reload)
    with pytest.raises(ValueError, match="forced staging reload failure"):
        _write(failed_root)
    assert not failed_root.exists()
    assert not tuple(tmp_path.glob(".failed-sidecar-shard.staging-*"))


def test_sidecar_writer_no_replace_survives_destination_creation_after_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "raced-sidecar-shard"
    raced_destination_inodes: list[int] = []
    real_rename = writer_module._atomic_rename_directory_noreplace

    def create_destination_then_rename(source: Path, destination: Path) -> None:
        destination.mkdir()
        raced_destination_inodes.append(
            os.stat(destination, follow_symlinks=False).st_ino
        )
        real_rename(source, destination)

    monkeypatch.setattr(
        writer_module,
        "_atomic_rename_directory_noreplace",
        create_destination_then_rename,
    )

    with pytest.raises(FileExistsError):
        _write(root)

    assert len(raced_destination_inodes) == 1
    assert os.stat(root, follow_symlinks=False).st_ino == raced_destination_inodes[0]
    assert not tuple(root.iterdir())
    assert not tuple(tmp_path.glob(".raced-sidecar-shard.staging-*"))


def test_sidecar_writer_cleanup_claim_restores_raced_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "failed-sidecar-shard"
    displaced_owned: list[Path] = []
    competitor_inodes: list[int] = []
    real_claim = writer_module._atomic_cleanup_claim_noreplace

    def fail_reload(*args, **kwargs):
        raise ValueError("forced staging reload failure")

    def swap_before_claim(source: Path, destination: Path) -> None:
        if source.name.startswith(".failed-sidecar-shard.staging-"):
            displaced = tmp_path / "displaced-owned-staging"
            source.rename(displaced)
            source.mkdir()
            displaced_owned.append(displaced)
            competitor_inodes.append(os.lstat(source).st_ino)
        real_claim(source, destination)

    monkeypatch.setattr(writer_module, "load_risk_sidecar_shard", fail_reload)
    monkeypatch.setattr(
        writer_module,
        "_atomic_cleanup_claim_noreplace",
        swap_before_claim,
    )

    with pytest.raises(ValueError, match="cleanup incomplete"):
        _write(root)

    competitors = tuple(tmp_path.glob(".failed-sidecar-shard.staging-*"))
    assert len(displaced_owned) == len(competitor_inodes) == 1
    assert displaced_owned[0].is_dir()
    assert len(competitors) == 1
    assert os.lstat(competitors[0]).st_ino == competitor_inodes[0]
    assert not root.exists()
    assert not tuple(tmp_path.glob(".*.cleanup-quarantine-*"))


def test_sidecar_loader_rejects_member_swapped_to_symlink_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sidecar-shard"
    _write(root)
    summary = root / "summary.json"
    displaced = tmp_path / "real-summary.json"
    real_open = writer_module._open_relative_regular_file_nofollow
    swapped = False

    def swap_summary(root_fd: int, name: str):
        nonlocal swapped
        if name == "summary.json" and not swapped:
            swapped = True
            summary.rename(displaced)
            summary.symlink_to(displaced)
        return real_open(root_fd, name)

    monkeypatch.setattr(
        writer_module,
        "_open_relative_regular_file_nofollow",
        swap_summary,
    )

    with pytest.raises(ValueError, match="symlink"):
        load_risk_sidecar_shard(
            root,
            grid=_grid(),
            expected_sample_ids=("sample-a", "sample-b"),
            expected_source_risk_shard_semantic_digest=SOURCE_DIGEST,
        )

    assert swapped
    assert summary.is_symlink()
    assert displaced.is_file()


def test_sidecar_loader_rejects_payload_path_swap_after_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sidecar-shard"
    _write(root)
    payload = root / "sidecars.npz"
    payload_bytes = payload.read_bytes()
    displaced = tmp_path / "original-sidecars.npz"
    competitor_inode: int | None = None
    real_hash = writer_module._sha256_open_file

    def hash_then_swap(handle):
        nonlocal competitor_inode
        digest = real_hash(handle)
        payload.rename(displaced)
        payload.write_bytes(payload_bytes)
        competitor_inode = os.lstat(payload).st_ino
        return digest

    monkeypatch.setattr(writer_module, "_sha256_open_file", hash_then_swap)

    with pytest.raises(ValueError, match="identity changed"):
        load_risk_sidecar_shard(
            root,
            grid=_grid(),
            expected_sample_ids=("sample-a", "sample-b"),
            expected_source_risk_shard_semantic_digest=SOURCE_DIGEST,
        )

    assert displaced.is_file()
    assert os.lstat(payload).st_ino == competitor_inode


def test_sidecar_loader_parses_the_same_immutable_bytes_that_it_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sidecar-shard"
    _write(root)
    baseline = load_risk_sidecar_shard(
        root,
        grid=_grid(),
        expected_sample_ids=("sample-a", "sample-b"),
        expected_source_risk_shard_semantic_digest=SOURCE_DIGEST,
    )
    payload = root / "sidecars.npz"
    payload_inode = os.lstat(payload).st_ino
    real_hash = writer_module._sha256_open_file
    mutated = False

    def hash_then_mutate_same_inode(handle):
        nonlocal mutated
        digest = real_hash(handle)
        with payload.open("r+b") as mutable:
            mutable.seek(0)
            mutable.write(b"BORK")
            mutable.flush()
            os.fsync(mutable.fileno())
        assert os.lstat(payload).st_ino == payload_inode
        mutated = True
        return digest

    monkeypatch.setattr(
        writer_module, "_sha256_open_file", hash_then_mutate_same_inode
    )

    loaded = load_risk_sidecar_shard(
        root,
        grid=_grid(),
        expected_sample_ids=("sample-a", "sample-b"),
        expected_source_risk_shard_semantic_digest=SOURCE_DIGEST,
    )

    assert mutated
    assert os.lstat(payload).st_ino == payload_inode
    assert payload.read_bytes().startswith(b"BORK")
    assert loaded.semantic_digest == baseline.semantic_digest
    assert np.array_equal(
        loaded.hidden_risk_occupancy, baseline.hidden_risk_occupancy
    )
    assert np.array_equal(
        loaded.robot_future_footprints, baseline.robot_future_footprints
    )


def test_pair_marker_writer_rejects_staging_swap_and_preserves_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "pair-complete.json"
    displaced = tmp_path / "owned-marker-staging"
    competitor_path: Path | None = None
    competitor_inode: int | None = None
    real_publish = writer_module._atomic_publish_owned_noreplace

    def swap_marker_staging(source, destination, expected_identity):
        nonlocal competitor_path, competitor_inode
        original_bytes = source.read_bytes()
        source.rename(displaced)
        source.write_bytes(original_bytes)
        competitor_path = source
        competitor_inode = os.lstat(source).st_ino
        return real_publish(source, destination, expected_identity)

    monkeypatch.setattr(
        writer_module,
        "_atomic_publish_owned_noreplace",
        swap_marker_staging,
    )

    with pytest.raises(ValueError, match="cleanup incomplete"):
        write_risk_sidecar_pair_completion_marker(
            marker,
            risk_root=tmp_path / "risk-shard",
            sidecar_root=tmp_path / "sidecar-shard",
            split="train",
            shard_index=7,
            sample_ids=("sample-a", "sample-b"),
            risk_shard_semantic_digest="b" * 64,
            sidecar_shard_semantic_digest="c" * 64,
        )

    assert not marker.exists()
    assert displaced.is_file()
    assert competitor_path is not None and competitor_path.is_file()
    assert os.lstat(competitor_path).st_ino == competitor_inode
