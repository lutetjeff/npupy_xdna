#!/usr/bin/env python3
"""Example 3: High-intensity elementwise tanh via NPU dispatch.

Demonstrates: ColIndependentTemplate with Horner polynomial tanh approximation
on 1M int16 elements across 32 AIE cores.

High arithmetic intensity (~32× speedup) because tanh requires many multiply-
accumulate steps per element, keeping all 32 cores saturated.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time

import numpy as np

N = 1 * 1024 * 1024
rng = np.random.default_rng(42)
x = rng.integers(-100, 100, N, dtype=np.int16)

t0 = time.perf_counter()
ref = np.tanh(x.astype(np.float32))
cpu_ms = (time.perf_counter() - t0) * 1000
print(f"CPU reference: {cpu_ms:.1f} ms")

try:
    from npupy_xdna.dispatch import activate, deactivate

    activate()
    print("NPU warmup...", flush=True)
    _ = np.tanh(x)

    t0 = time.perf_counter()
    result = np.tanh(x)
    npu_ms = (time.perf_counter() - t0) * 1000
    deactivate()

    match = np.allclose(result.astype(np.float64), ref.astype(np.float64), atol=1e-4)
    print(f"\ntanh  {N // 1_000_000}M  int16")
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
