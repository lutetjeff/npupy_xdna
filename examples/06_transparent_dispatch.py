#!/usr/bin/env python3
"""Example 6: Full transparent dispatch flow.

Demonstrates activate/deactivate and how normal numpy code is automatically
routed to the NPU for supported int16 ops, while unsupported ops (wrong dtype,
no template, reductions) silently fall back to CPU.

The shim intercepts: matmul, add, multiply, maximum, tanh  (int16 only).
Everything else falls through to numpy unchanged.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time

import numpy as np

deactivate = None

try:
    from npupy_xdna.dispatch import activate, deactivate

    activate()
    print("Dispatch active — intercepting int16 numpy ops")
    print()

    rng = np.random.default_rng(42)
    A = rng.integers(-5, 5, (1024, 1024), dtype=np.int16)
    B = rng.integers(-5, 5, (1024, 1024), dtype=np.int16)

    print("Step 1: A @ B  (matmul int16 1024×1024) → NPU gemm_fusion", flush=True)
    t0 = time.perf_counter()
    C = A @ B
    print(f"         done  {(time.perf_counter() - t0) * 1000:.1f} ms  shape={C.shape}")

    print("Step 2: np.tanh(C)  (elementwise_unary int16 1M) → NPU col_independent", flush=True)
    t0 = time.perf_counter()
    D = np.tanh(C)
    print(f"         done  {(time.perf_counter() - t0) * 1000:.1f} ms  shape={D.shape}")

    print("Step 3: np.maximum(0, D)  (elementwise_unary int16 1M) → NPU col_independent", flush=True)
    t0 = time.perf_counter()
    E = np.maximum(np.int16(0), D)
    print(f"         done  {(time.perf_counter() - t0) * 1000:.1f} ms  shape={E.shape}")

    print("Step 4: np.sum(E, axis=0)  (reduction, no NPU template) → CPU fallback", flush=True)
    t0 = time.perf_counter()
    F = np.sum(E, axis=0)
    print(f"         done  {(time.perf_counter() - t0) * 1000:.1f} ms  shape={F.shape}")

    print("Step 5: np.sin(F.astype(float64))  (float64, unsupported dtype) → CPU fallback", flush=True)
    t0 = time.perf_counter()
    G = np.sin(F.astype(np.float64))
    print(f"         done  {(time.perf_counter() - t0) * 1000:.1f} ms  shape={G.shape}")

    deactivate()
    print()
    print("Dispatch deactivated — numpy restored to original state")
    print("All operations completed transparently!")

except Exception as exc:
    if deactivate is not None:
        try:
            deactivate()
        except Exception:
            pass
    print(f"\nNPU not available ({exc})")
    print("Source XRT env to enable NPU dispatch.")
