"""Current trajectory records shared by planning and Long40 generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidateRollout:
    """Rolled-out candidate before geometry query maps are attached."""

    trajectory_id: str
    poses: np.ndarray
    controls: np.ndarray
    is_stop: bool
    is_reverse: bool
