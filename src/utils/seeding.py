"""Deterministic, process-independent seeding and stable identifiers.

Every random process in the project must derive its seed from an explicit base
seed plus a set of naming parts through :func:`derive_seed`. Python's built-in
``hash()`` is never used because it is randomized across processes; all digests
here come from :mod:`hashlib` (BLAKE2b) over a canonical string.
"""

from __future__ import annotations

import hashlib

import numpy as np

_UINT32 = 2**32


def _canonical(base_seed: int, parts: tuple) -> bytes:
    """Build a canonical, order-sensitive byte string from seed and parts."""
    tokens = [f"seed={int(base_seed)}"]
    tokens.extend(str(p) for p in parts)
    return "|".join(tokens).encode("utf-8")


def stable_digest(*parts: object, size: int = 16) -> str:
    """Return a hex digest of the given parts (no base seed)."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=size).hexdigest()


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 32-bit seed from ``base_seed`` and ordered ``parts``."""
    digest = hashlib.blake2b(_canonical(base_seed, parts), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _UINT32


def make_rng(base_seed: int, *parts: object) -> np.random.Generator:
    """Return a NumPy generator seeded deterministically from seed and parts."""
    return np.random.default_rng(derive_seed(base_seed, *parts))


def sample_id(
    split: str,
    recording_id: str,
    base_state_id: str,
    trajectory_id: str,
    variant: str,
    seed: int,
) -> str:
    """Stable id for a risk/verification sample; independent of call order."""
    short = stable_digest(
        split, recording_id, base_state_id, trajectory_id, variant, seed, size=12
    )
    return f"{split}-{short}"


def pair_group_id(
    base_state_id: str, trajectory_id: str, occluder_geometry_id: str, ped_snippet_id: str
) -> str:
    """Stable id shared by all paired counterfactual variants of one event."""
    short = stable_digest(
        base_state_id, trajectory_id, occluder_geometry_id, ped_snippet_id, size=12
    )
    return f"pair-{short}"
