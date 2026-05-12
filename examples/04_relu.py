#!/usr/bin/env python3
"""Example 4: Simple elementwise ReLU on 1M int16 elements.

Demonstrates: ColIndependentTemplate dispatching np.maximum(0, x) to 32 AIE
cores.  ReLU is memory-bandwidth-bound (one compare per element), so the
speedup is modest (~2×) compared to compute-intensive tanh (~32×).
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
ref = np.maximum(np.int16(0), x)
cpu_ms = (time.perf_counter() - t0) * 1000
print(f"CPU reference: {cpu_ms:.1f} ms")

try:
    from npupy_xdna.dispatch import activate, deactivate

    activate()
    print("NPU warmup...", flush=True)
    _ = np.maximum(np.int16(0), x)

    t0 = time.perf_counter()
    result = np.maximum(np.int16(0), x)
    npu_ms = (time.perf_counter() - t0) * 1000
    deactivate()

    match = np.array_equal(result, ref)
    print(f"\nReLU  {N // 1_000_000}M  int16")
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
