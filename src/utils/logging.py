"""Logging helpers and provenance records for reproducible artifacts.

``code_version`` is ``"unversioned"`` while the workspace is not a git repo (see
``DECISIONS.md``). Once git is initialized, callers should pass the real commit.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..contracts import SCHEMA_VERSION

UNVERSIONED = "unversioned"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a single stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def build_provenance(
    *,
    config_digest: str,
    generator_seed: int,
    split: str | None = None,
    code_version: str = UNVERSIONED,
    extra: dict | None = None,
) -> dict:
    """Assemble the provenance block required on every data artifact."""
    record = {
        "schema_version": SCHEMA_VERSION,
        "config_digest": config_digest,
        "generator_seed": int(generator_seed),
        "split": split,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version,
    }
    if extra:
        record.update(extra)
    return record
