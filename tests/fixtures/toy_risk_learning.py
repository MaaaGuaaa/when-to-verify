"""Thin test wrapper around the production SOP08--10 toy publication."""

from __future__ import annotations

from src.datasets.toy_risk_learning import (
    assert_toy_split_isolation,
    make_toy_batch,
    make_toy_risk_dataset,
)

__all__ = [
    "assert_toy_split_isolation",
    "make_toy_batch",
    "make_toy_risk_dataset",
]
