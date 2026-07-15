"""Dataset indexing, deterministic splitting, and leakage auditing."""

from .split_manager import (
    SplitIndexError,
    SplitLeakageError,
    SplitResult,
    assert_no_split_leakage,
    audit_split_leakage,
    make_split_manifest,
    serialize_manifest,
    write_split_artifacts,
)

__all__ = [
    "SplitIndexError",
    "SplitLeakageError",
    "SplitResult",
    "assert_no_split_leakage",
    "audit_split_leakage",
    "make_split_manifest",
    "serialize_manifest",
    "write_split_artifacts",
]
