"""Event-centered blind-spot risk and active verification — frozen contract layer.

This package is the fresh, clean reimplementation entry point defined by SOP-00.
It intentionally lives under ``src/`` and does not modify the legacy top-level
packages (``bev/``, ``risk_model/`` ...). See ``DECISIONS.md`` for the rationale.
"""

from . import contracts  # noqa: F401

__all__ = ["contracts"]
