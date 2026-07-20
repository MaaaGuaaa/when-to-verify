"""Portable immutable-publication tests for repository output directories."""

from __future__ import annotations

import errno
import multiprocessing
import os
from pathlib import Path

import pytest

from src.utils import atomic_publish as publish_module


def _concurrent_publish_worker(
    source: str,
    destination: str,
    start: object,
    results: object,
    force_fallback: bool,
) -> None:
    start.wait()
    try:
        if force_fallback:
            publish_module._flock_rename_noreplace(
                Path(source), Path(destination)
            )
        else:  # pragma: no cover - helper also supports end-to-end use
            publish_module.atomic_rename_noreplace(source, destination)
    except FileExistsError:
        results.put(("exists", Path(source).name))
    except BaseException as exc:  # pragma: no cover - reported to parent
        results.put(("error", f"{type(exc).__name__}: {exc}"))
    else:
        results.put(("success", Path(source).name))


def _force_renameat2_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported(source: Path, destination: Path) -> None:
        raise OSError(errno.EINVAL, os.strerror(errno.EINVAL), destination)

    monkeypatch.setattr(publish_module, "_renameat2_noreplace", unsupported)


def test_lustre_fallback_publishes_directory_without_changing_source_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_renameat2_unsupported(monkeypatch)
    source = tmp_path / "staging"
    destination = tmp_path / "published"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"payload\n")
    source_inode = os.lstat(source).st_ino

    publish_module.atomic_rename_noreplace(source, destination)

    assert not source.exists()
    assert destination.is_dir()
    assert os.lstat(destination).st_ino == source_inode
    assert (destination / "payload.bin").read_bytes() == b"payload\n"


def test_lustre_fallback_publishes_regular_file_without_changing_source_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_renameat2_unsupported(monkeypatch)
    source = tmp_path / "marker.staging"
    destination = tmp_path / "marker.json"
    source.write_bytes(b'{"complete":true}\n')
    source_inode = os.lstat(source).st_ino

    publish_module.atomic_rename_noreplace(source, destination)

    assert not source.exists()
    assert destination.read_bytes() == b'{"complete":true}\n'
    assert os.lstat(destination).st_ino == source_inode


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_lustre_fallback_refuses_and_preserves_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    _force_renameat2_unsupported(monkeypatch)
    source = tmp_path / "staging"
    destination = tmp_path / "published"
    if kind == "directory":
        source.mkdir()
        destination.mkdir()
    elif kind == "file":
        source.write_bytes(b"owned\n")
        destination.write_bytes(b"competitor\n")
    else:
        source.mkdir()
        target = tmp_path / "competitor-target"
        target.mkdir()
        destination.symlink_to(target, target_is_directory=True)
    destination_inode = os.lstat(destination).st_ino

    with pytest.raises(FileExistsError):
        publish_module.atomic_rename_noreplace(source, destination)

    assert os.path.lexists(source)
    assert os.lstat(destination).st_ino == destination_inode
    if kind == "file":
        assert destination.read_bytes() == b"competitor\n"
    elif kind == "symlink":
        assert destination.is_symlink()


def test_lustre_fallback_does_not_create_destination_when_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_renameat2_unsupported(monkeypatch)
    source = tmp_path / "staging"
    destination = tmp_path / "published"
    source.mkdir()

    def fail_rename(source_path: Path, destination_path: Path) -> None:
        raise OSError(errno.EIO, "forced rename failure", destination_path)

    monkeypatch.setattr(publish_module.os, "rename", fail_rename)

    with pytest.raises(OSError, match="forced rename failure"):
        publish_module.atomic_rename_noreplace(source, destination)

    assert source.is_dir()
    assert not destination.exists()


def test_lustre_fallback_does_not_mask_non_capability_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "staging"
    destination = tmp_path / "published"
    source.mkdir()

    def fail_permission(source_path: Path, destination_path: Path) -> None:
        raise PermissionError(errno.EACCES, "denied", destination_path)

    monkeypatch.setattr(
        publish_module, "_renameat2_noreplace", fail_permission
    )

    with pytest.raises(PermissionError, match="denied"):
        publish_module.atomic_rename_noreplace(source, destination)

    assert source.is_dir()
    assert not destination.exists()


def test_concurrent_fallback_publishers_have_exactly_one_winner(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first-staging"
    second = tmp_path / "second-staging"
    destination = tmp_path / "published"
    for source in (first, second):
        source.mkdir()
        (source / "owner.txt").write_text(source.name, encoding="utf-8")

    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    workers = [
        context.Process(
            target=_concurrent_publish_worker,
            args=(str(source), str(destination), start, results, True),
        )
        for source in (first, second)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0

    outcomes = [results.get(timeout=5) for _ in workers]
    assert sorted(status for status, _ in outcomes) == ["exists", "success"]
    winner = next(owner for status, owner in outcomes if status == "success")
    loser = next(owner for status, owner in outcomes if status == "exists")
    assert (destination / "owner.txt").read_text(encoding="utf-8") == winner
    assert not (tmp_path / winner).exists()
    assert (tmp_path / loser).is_dir()
