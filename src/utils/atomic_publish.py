"""Portable no-replace publication for immutable files and directories."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
from pathlib import Path
import stat


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_UNSUPPORTED_NOREPLACE_ERRNOS = frozenset(
    {
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
    }
)


def _absolute(path: str | Path) -> Path:
    """Make a path absolute without resolving symlinks."""

    return Path(os.path.abspath(os.fspath(path)))


def _renameat2_noreplace(source: Path, destination: Path) -> None:
    """Call the Linux no-replace primitive without a compatibility fallback."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as exc:  # pragma: no cover - Linux libc is normally present
        raise OSError(errno.ENOSYS, "libc unavailable for renameat2") from exc
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "libc renameat2 is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno() or errno.EIO
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination,
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def _verify_open_directory(path: Path, descriptor: int) -> None:
    """Require the locked descriptor to remain the named real directory."""

    opened = os.fstat(descriptor)
    named = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(opened.st_mode) or not stat.S_ISDIR(named.st_mode):
        raise ValueError(f"publication parent is not a real directory: {path}")
    if opened.st_dev != named.st_dev or opened.st_ino != named.st_ino:
        raise ValueError(f"publication parent identity changed: {path}")


def _flock_rename_noreplace(source: Path, destination: Path) -> None:
    """Serialize cooperative publishers when renameat2 flags are unsupported.

    Some Lustre deployments reject ``RENAME_NOREPLACE`` with ``EINVAL`` while
    providing cross-node POSIX ``flock``. Locking the destination parent keeps
    the final move atomic and prevents repository writers targeting the same
    name from passing the absence check concurrently. The descriptor-scoped
    lock cannot become a stale lock file after process exit.
    """

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    parent = destination.parent
    descriptor = os.open(parent, flags)
    try:
        _verify_open_directory(parent, descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _verify_open_directory(parent, descriptor)
        if os.path.lexists(destination):
            raise FileExistsError(
                errno.EEXIST,
                os.strerror(errno.EEXIST),
                destination,
            )
        os.rename(source, destination)
        _verify_open_directory(parent, descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def atomic_rename_noreplace(
    source: str | Path, destination: str | Path
) -> None:
    """Atomically move ``source`` while refusing an existing destination.

    The kernel primitive is preferred. A cooperative, descriptor-scoped flock
    fallback is used only when the filesystem reports that no-replace rename
    flags are unsupported. All other failures remain fail-closed.
    """

    source_path = _absolute(source)
    destination_path = _absolute(destination)
    try:
        _renameat2_noreplace(source_path, destination_path)
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_NOREPLACE_ERRNOS:
            raise
        _flock_rename_noreplace(source_path, destination_path)
