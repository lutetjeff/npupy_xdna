#!/usr/bin/env python3
"""Example 1: Basic int16 GEMM offload to NPU.

Demonstrates: GemmFusionTemplate dispatching a 512×512 int16 matmul to
32 AIE cores via the transparent dispatch shim.

Expected speedup: ~9× vs CPU reference.
First run compiles the xclbin — allow 1–2 minutes; subsequent runs are fast.
(Use 2048×2048 for a better speedup demonstration on a loaded NPU.)
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

# CPU baseline: int32 accumulator (same precision contract as NPU kernel)
t0 = time.perf_counter()
ref = np.clip(
    np.matmul(A.astype(np.int32), B.astype(np.int32)),
    -32768, 32767,
).astype(np.int16)
cpu_ms = (time.perf_counter() - t0) * 1000
print(f"CPU reference: {cpu_ms:.1f} ms")

try:
    from npupy_xdna.dispatch import activate, deactivate

    activate()

    print("NPU warmup (first run compiles xclbin)...", flush=True)
    _ = np.matmul(A, B)

    t0 = time.perf_counter()
    result = np.matmul(A, B)
    npu_ms = (time.perf_counter() - t0) * 1000

    deactivate()

    match = np.array_equal(result, ref)
    print(f"\nGEMM {N}×{N}  int16")
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
