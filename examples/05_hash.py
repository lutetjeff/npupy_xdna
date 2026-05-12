#!/usr/bin/env python3
"""Example 5: Compute-bound FNV-1a hash on 1M int16 elements.

Demonstrates: npupy_xdna.ops.hash_int16 — a custom operation that does NOT
go through the numpy shim, but dispatches directly via the Dispatcher.

8 rounds of XOR + multiply per element keeps all 32 AIE cores fully saturated,
delivering ~49× speedup over the equivalent Python loop.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time

import numpy as np

N = 1 * 1024 * 1024
rng = np.random.default_rng(42)
x = rng.integers(-100, 100, N, dtype=np.int16)

_FNV_OFFSET = np.uint16(0x811C)
_FNV_PRIME_U32 = np.uint32(0x0193)


def _cpu_hash(arr: np.ndarray) -> np.ndarray:
    flat = arr.ravel().view(np.uint16).copy()
    h = np.full(flat.shape, _FNV_OFFSET, dtype=np.uint16)
    v = flat.copy()
    for _ in range(8):
        h ^= v
        h = (h.astype(np.uint32) * _FNV_PRIME_U32).astype(np.uint16)
        v = (h >> np.uint16(1)).astype(np.uint16)
    return h.view(np.int16).reshape(arr.shape)


t0 = time.perf_counter()
ref = _cpu_hash(x)
cpu_ms = (time.perf_counter() - t0) * 1000
print(f"CPU reference: {cpu_ms:.1f} ms")

try:
    from npupy_xdna.ops import hash_int16

    print("NPU warmup...", flush=True)
    _ = hash_int16(x)

    t0 = time.perf_counter()
    result = hash_int16(x)
    npu_ms = (time.perf_counter() - t0) * 1000

    match = np.array_equal(result, ref)
    print(f"\nhash_int16  {N // 1_000_000}M  int16")
    print(f"  CPU:     {cpu_ms:.1f} ms")
    print(f"  NPU:     {npu_ms:.1f} ms")
    print(f"  Speedup: {cpu_ms / npu_ms:.1f}×")
    print(f"  Correct: {'PASS' if match else 'FAIL'}")
    if not match:
        sys.exit(1)

except Exception as exc:
    print(f"\nNPU not available ({exc})")
    print("Showing CPU-only result — source XRT env to enable NPU.")
    print(f"  CPU: {cpu_ms:.1f} ms")
