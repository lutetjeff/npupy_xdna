"""Deterministic random number generation for reproducible benchmarks."""

from __future__ import annotations

import numpy as np


def make_rng(salt: str = "") -> np.random.Generator:
    """Return a deterministic :class:`numpy.random.Generator`.

    The seed is derived from ``seed=42`` and ``salt`` so that different salts
    yield independent but reproducible streams.
    """
    seed = 42
    if salt:
        seed = (seed * 31 + hash(salt)) & 0xFFFFFFFF
    return np.random.default_rng(seed)
