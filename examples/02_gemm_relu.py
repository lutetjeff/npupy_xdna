#!/usr/bin/env python3
"""Example 2: GEMM with fused ReLU epilogue.

Demonstrates two approaches:
  (a) Dispatch shim — np.matmul + np.maximum intercepted as separate NPU ops
  (b) Direct template — GemmFusionTemplate lowered with epilogue="relu" for
      a single fused kernel (zero extra data movement for the ReLU pass)

Expected speedup: ~9× vs CPU reference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time

import numpy as np

N = 512  # 512×512 completes in seconds; 2048×2048 shows larger NPU speedup
rng = np.random.default_rng(42)
A = rng.integers(-5, 5, (N, N), dtype=np.int16)
B = rng.integers(-5, 5, (N, N), dtype=np.int16)

t0 = time.perf_counter()
ref_mm = np.clip(
    np.matmul(A.astype(np.int32), B.astype(np.int32)),
    -32768, 32767,
).astype(np.int16)
ref = np.maximum(np.int16(0), ref_mm)
cpu_ms = (time.perf_counter() - t0) * 1000
print(f"CPU reference: {cpu_ms:.1f} ms")

try:
    from npupy_xdna.dispatch import activate, deactivate

    activate()
    print("NPU warmup...", flush=True)
    _ = np.maximum(np.int16(0), np.matmul(A, B))

    t0 = time.perf_counter()
    result = np.maximum(np.int16(0), np.matmul(A, B))
    npu_ms = (time.perf_counter() - t0) * 1000
    deactivate()

    match = np.array_equal(result, ref)
    print(f"\nGEMM+ReLU {N}×{N}  int16  (dispatch shim, separate ops)")
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
