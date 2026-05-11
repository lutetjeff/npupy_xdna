from __future__ import annotations

import numpy as np


def gemm_relu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.maximum(np.matmul(A, B), 0)


def make_inputs(M: int = 256, N: int = 256, K: int = 256, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.integers(-10, 11, size=(M, K), dtype=np.int16)
    B = rng.integers(-10, 11, size=(K, N), dtype=np.int16)
    return A, B
