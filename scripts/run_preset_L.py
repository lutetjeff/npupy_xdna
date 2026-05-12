#!/usr/bin/env python3
"""
Preset-L full benchmark: all 14 NPBench benchmarks at production scale.

Sizes:
  GEMM family      : 2048×2048×2048
  Elementwise      : 4,194,304 elements (int16)
  jacobi-2d        : 256×256, T=5
  mvt / atax       : 2048×2048 (matrix-vector, CPU fallback)
  correlation      : 2048×2048 (float32 path, CPU fallback)
  gramschmidt      : 2048×2048 (QR via numpy linalg, CPU fallback)
  horner_poly      : 4,194,304 elements

Three-way comparison per benchmark:
  1. Base CPU int16 — numpy int16 unaccelerated (SKIPPED for matmul ≥1024)
  2. CPU BLAS       — scipy OpenBLAS sgemm (matmul); equals base (elementwise)
  3. NPUPy          — NPU dispatch or CPU fallback

Output:
  results/timings/npbench_preset_L.jsonl   (OVERWRITTEN)
  results/04_npbench_plots/preset_L_speedup_chart.png
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg.blas as _scipy_blas

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO.parent))

RESULTS_DIR = _REPO / "results"
JSONL_PATH  = RESULTS_DIR / "timings" / "npbench_preset_L.jsonl"
PLOT_DIR    = RESULTS_DIR / "04_npbench_plots"
PLOT_PATH   = PLOT_DIR / "preset_L_speedup_chart.png"
EVIDENCE_PATH = Path("/home/lutet/ece511/.sisyphus/evidence/preset-L-full.txt")

# ---------------------------------------------------------------------------
# Benchmark constants
# ---------------------------------------------------------------------------
GEMM_SIZE = 2048       # M = K = N for all GEMM benchmarks
ELEM_SIZE = 4_194_304  # 4 M int16 elements for elementwise benchmarks
JACOBI_H  = 256
JACOBI_W  = 256
JACOBI_T  = 5          # stencil iterations
HORNER_N  = 4_194_304  # same as ELEM_SIZE
HORNER_DEGREE = 8

N_WARMUP  = 3
N_ITERS   = 5
TIMEOUT_S = 120.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_int16(rng: np.random.Generator, *shape: int,
                lo: int = -5, hi: int = 5) -> np.ndarray:
    return rng.integers(lo, hi + 1, size=shape, dtype=np.int16)


def _time_fn(fn, n_warmup: int = N_WARMUP, n_iters: int = N_ITERS,
             timeout_s: float = TIMEOUT_S) -> float:
    """Warmup then measure; return median ms.  Raises on exception."""
    for _ in range(n_warmup):
        fn()
    times_ms: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1e3)
        if times_ms[-1] > timeout_s * 1e3:
            # Single iteration already exceeded timeout — stop early
            break
    return statistics.median(times_ms)


def _speedup(base_ms: float, target_ms: float) -> float:
    """Return base/target speedup, or -1 if not applicable."""
    if base_ms <= 0 or target_ms <= 0:
        return -1.0
    return round(base_ms / target_ms, 3)


def _record(
    benchmark: str,
    size: str,
    base_cpu_ms: float,    # -1 if skipped
    blas_cpu_ms: float,
    npupy_ms: float,
    template: str,
    note: str,
) -> dict:
    su_base = _speedup(base_cpu_ms, npupy_ms) if base_cpu_ms > 0 else -1
    su_blas = _speedup(blas_cpu_ms, npupy_ms) if blas_cpu_ms > 0 else -1
    return {
        "benchmark":       benchmark,
        "preset":          "L",
        "size":            size,
        "base_cpu_ms":     round(base_cpu_ms, 3),
        "blas_cpu_ms":     round(blas_cpu_ms, 3),
        "npupy_ms":        round(npupy_ms, 3),
        "speedup_vs_base": su_base,
        "speedup_vs_blas": su_blas,
        "template":        template,
        "note":            note,
    }


def _blas_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_f32 = np.asfortranarray(A.astype(np.float32))
    B_f32 = np.asfortranarray(B.astype(np.float32))
    r = _scipy_blas.sgemm(1.0, A_f32, B_f32)
    return np.clip(r, -32768, 32767).astype(np.int16)


def _blas_matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    A_f32 = np.asfortranarray(A.astype(np.float32))
    x_f32 = x.astype(np.float32)
    r = _scipy_blas.sgemv(1.0, A_f32, x_f32)
    return np.clip(r, -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# CPU tanh: Pade [3/3] — same algorithm as tanh_int16.cc kernel
# ---------------------------------------------------------------------------

def cpu_tanh_pade(x: np.ndarray) -> np.ndarray:
    """Integer tanh via Pade [3/3] rational approximation (matches kernel)."""
    xs = x.astype(np.int32) >> 8       # scale to [-128, 127]
    xs2 = xs * xs
    num = xs * (27 + xs2)
    den = 27 + 9 * xs2
    # Integer division  (avoid / 0, den is always > 0 for |xs| < 4)
    out = np.where(den != 0,
                   np.trunc(num.astype(np.float64) * 32767.0 / den).astype(np.int32),
                   np.int32(0))
    out = np.clip(out, -32768, 32767)
    out = np.where(xs >= 4,  32767, out)
    out = np.where(xs <= -4, -32768, out)
    return out.astype(np.int16)


# ---------------------------------------------------------------------------
# CPU hash: FNV-1a on int16 (same as ops._cpu_hash)
# ---------------------------------------------------------------------------
_FNV_OFFSET   = np.uint16(0x811C)
_FNV_PRIME_U32 = np.uint32(0x0193)


def cpu_hash_int16(arr: np.ndarray) -> np.ndarray:
    """FNV-1a hash on int16 — same algorithm as hash_int16.cc kernel."""
    flat = arr.ravel().view(np.uint16).copy()
    hash_arr = np.full(flat.shape, _FNV_OFFSET, dtype=np.uint16)
    x = flat.copy()
    for _ in range(8):
        hash_arr ^= x
        hash_arr = (hash_arr.astype(np.uint32) * _FNV_PRIME_U32).astype(np.uint16)
        x = (hash_arr >> np.uint16(1)).astype(np.uint16)
    return hash_arr.view(np.int16).reshape(arr.shape)


# ---------------------------------------------------------------------------
# CPU 5-point stencil
# ---------------------------------------------------------------------------

def cpu_stencil_5pt(inp: np.ndarray) -> np.ndarray:
    H, W = inp.shape
    inp32 = inp.astype(np.int32)
    padded = np.zeros((H + 2, W), dtype=np.int32)
    padded[1:H + 1, :] = inp32
    center = padded[1:H + 1, :]
    top    = padded[0:H, :]
    bot    = padded[2:H + 2, :]
    left   = np.zeros((H, W), dtype=np.int32)
    left[:, 1:] = center[:, :-1]
    right  = np.zeros((H, W), dtype=np.int32)
    right[:, :-1] = center[:, 1:]
    s = center + top + bot + left + right
    v = np.trunc(s.astype(np.float64) / 5.0).astype(np.int32)
    v[:, 0]  = 0
    v[:, -1] = 0
    return np.clip(v, -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# NPU wrappers
# ---------------------------------------------------------------------------

class NpuGemm:
    """Wraps GemmFusionTemplate for square GEMM at a given size."""

    def __init__(self, M: int, K: int, N: int):
        from npupy_xdna.regions.region import ArraySpec, Region
        from npupy_xdna.runtime.npu_runner import NpuRunner
        from npupy_xdna.templates.gemm_fusion import GemmFusionTemplate

        self.M, self.K, self.N = M, K, N
        self.region = Region(
            op="matmul",
            inputs=[ArraySpec(shape=(M, K), dtype="int16"),
                    ArraySpec(shape=(K, N), dtype="int16")],
            output=ArraySpec(shape=(M, N), dtype="int16"),
        )
        tmpl = GemmFusionTemplate()
        configs = tmpl.config_space(self.region)
        # Prefer epilogue="none"; fall back to first config
        self.config = next(
            (c for c in configs if c.extra.get("epilogue") in ("none", None, "")),
            configs[0],
        )
        print(f"  [NpuGemm] Lowering GemmFusionTemplate {M}×{K}×{N}...", flush=True)
        self.iron_fn = tmpl.lower(self.region, self.config)
        self.runner  = NpuRunner()

    def run(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute A @ B on NPU.  B is passed row-major; we transpose internally."""
        B_col = np.ascontiguousarray(B.T)
        result = self.runner.run(self.region, self.config, self.iron_fn, [A, B_col],
                                 timeout_s=TIMEOUT_S)
        if result.status != "ok":
            raise RuntimeError(f"NPU GEMM failed: {result.status}")
        return result.output


class NpuElem:
    """Wraps ColIndependentTemplate for elementwise ops at a given size."""

    def __init__(self, n: int, compute_fn: str = "relu"):
        from npupy_xdna.regions.region import ArraySpec, Region
        from npupy_xdna.runtime.npu_runner import NpuRunner
        from npupy_xdna.templates.col_independent import ColIndependentTemplate

        metadata = {"compute_fn": compute_fn} if compute_fn != "relu" else {}
        self.region = Region(
            op="elementwise_unary",
            inputs=[ArraySpec((n,), "int16")],
            output=ArraySpec((n,), "int16"),
            metadata=metadata,
        )
        tmpl = ColIndependentTemplate()
        if not tmpl.match(self.region):
            raise ValueError(
                f"ColIndependentTemplate.match() returned False for n={n}, fn={compute_fn}")
        self.config  = tmpl.config_space(self.region)[0]
        print(f"  [NpuElem] Lowering ColIndependentTemplate n={n} fn={compute_fn}...",
              flush=True)
        self.iron_fn = tmpl.lower(self.region, self.config)
        self.runner  = NpuRunner()

    def run(self, x: np.ndarray) -> np.ndarray:
        result = self.runner.run(self.region, self.config, self.iron_fn, [x],
                                 timeout_s=TIMEOUT_S)
        if result.status != "ok":
            raise RuntimeError(f"NPU elem failed: {result.status}")
        return result.output


# ---------------------------------------------------------------------------
# Individual benchmark functions
# ---------------------------------------------------------------------------

def bench_gemm(rng: np.random.Generator, npu: NpuGemm) -> dict:
    print("[bench] gemm 2048×2048×2048", flush=True)
    M = K = N = GEMM_SIZE
    A = _make_int16(rng, M, K)
    B = _make_int16(rng, K, N)

    base_cpu_ms = -1  # skipped (matmul ≥1024 too slow)
    blas_cpu_ms = _time_fn(lambda: _blas_matmul(A, B))
    try:
        npupy_ms = _time_fn(lambda: npu.run(A, B))
        template = "gemm_fusion"
    except Exception as exc:
        print(f"  NPU error: {exc}", flush=True)
        npupy_ms = blas_cpu_ms
        template = f"cpu_fallback (error: {type(exc).__name__})"
    return _record("gemm", f"{M}×{K}×{N}", base_cpu_ms, blas_cpu_ms, npupy_ms,
                   template, "skipped_too_slow for base_cpu_ms")


def bench_gemm_relu(rng: np.random.Generator, npu: NpuGemm) -> dict:
    print("[bench] gemm_relu 2048×2048×2048", flush=True)
    M = K = N = GEMM_SIZE
    A = _make_int16(rng, M, K)
    B = _make_int16(rng, K, N)

    base_cpu_ms = -1
    blas_cpu_ms = _time_fn(lambda: np.maximum(_blas_matmul(A, B), np.int16(0)))
    try:
        npupy_ms = _time_fn(lambda: np.maximum(npu.run(A, B), np.int16(0)))
        template = "gemm_fusion+cpu_relu"
    except Exception as exc:
        print(f"  NPU error: {exc}", flush=True)
        npupy_ms = blas_cpu_ms
        template = f"cpu_fallback (error: {type(exc).__name__})"
    return _record("gemm_relu", f"{M}×{K}×{N}", base_cpu_ms, blas_cpu_ms, npupy_ms,
                   template, "synthetic GEMM+ReLU; base_cpu_ms skipped_too_slow")


def bench_3mm(rng: np.random.Generator, npu: NpuGemm) -> dict:
    print("[bench] 3mm 2048×2048×2048 (3 chained matmuls)", flush=True)
    N = GEMM_SIZE
    A = _make_int16(rng, N, N)
    B = _make_int16(rng, N, N)
    C = _make_int16(rng, N, N)
    D = _make_int16(rng, N, N)

    base_cpu_ms = -1

    def blas_3mm():
        AB = _blas_matmul(A, B)
        CD = _blas_matmul(C, D)
        return _blas_matmul(AB, CD)

    blas_cpu_ms = _time_fn(blas_3mm)

    def npu_3mm():
        AB = npu.run(A, B)
        CD = npu.run(C, D)
        return npu.run(AB, CD)

    try:
        npupy_ms = _time_fn(npu_3mm)
        template = "gemm_fusion (×3)"
    except Exception as exc:
        print(f"  NPU error: {exc}", flush=True)
        npupy_ms = blas_cpu_ms
        template = f"cpu_fallback (error: {type(exc).__name__})"
    return _record("3mm", f"{N}×{N}×{N} (×3)", base_cpu_ms, blas_cpu_ms, npupy_ms,
                   template, "3 chained matmuls; base_cpu_ms skipped_too_slow")


def bench_symm(rng: np.random.Generator, npu: NpuGemm) -> dict:
    print("[bench] symm 2048×2048 (symmetric A)", flush=True)
    N = GEMM_SIZE
    A_upper = _make_int16(rng, N, N)
    A = (A_upper + A_upper.T).astype(np.int16)  # symmetric
    B = _make_int16(rng, N, N)

    base_cpu_ms = -1
    blas_cpu_ms = _time_fn(lambda: _blas_matmul(A, B))
    try:
        npupy_ms = _time_fn(lambda: npu.run(A, B))
        template = "gemm_fusion"
    except Exception as exc:
        print(f"  NPU error: {exc}", flush=True)
        npupy_ms = blas_cpu_ms
        template = f"cpu_fallback (error: {type(exc).__name__})"
    return _record("symm", f"{N}×{N}", base_cpu_ms, blas_cpu_ms, npupy_ms,
                   template, "symmetric A; base_cpu_ms skipped_too_slow")


def bench_syr2k(rng: np.random.Generator, npu: NpuGemm) -> dict:
    print("[bench] syr2k 2048×2048 (2 matmuls + add)", flush=True)
    N = GEMM_SIZE
    A = _make_int16(rng, N, N)
    B = _make_int16(rng, N, N)

    base_cpu_ms = -1

    def blas_syr2k():
        return _blas_matmul(A, B.T.copy()) + _blas_matmul(B, A.T.copy())

    def npu_syr2k():
        return npu.run(A, B.T.copy()) + npu.run(B, A.T.copy())

    blas_cpu_ms = _time_fn(blas_syr2k)
    try:
        npupy_ms = _time_fn(npu_syr2k)
        template = "gemm_fusion (×2 + add)"
    except Exception as exc:
        print(f"  NPU error: {exc}", flush=True)
        npupy_ms = blas_cpu_ms
        template = f"cpu_fallback (error: {type(exc).__name__})"
    return _record("syr2k", f"{N}×{N}×{N}", base_cpu_ms, blas_cpu_ms, npupy_ms,
                   template, "A@B^T + B@A^T; base_cpu_ms skipped_too_slow")


def bench_tanh(rng: np.random.Generator, npu_tanh: NpuElem) -> dict:
    print(f"[bench] tanh {ELEM_SIZE} elements", flush=True)
    x = _make_int16(rng, ELEM_SIZE, lo=-100, hi=100)

    base_cpu_ms = _time_fn(lambda: cpu_tanh_pade(x))
    blas_cpu_ms = base_cpu_ms  # BLAS doesn't help elementwise

    try:
        npupy_ms = _time_fn(lambda: npu_tanh.run(x))
        template = "col_independent (tanh)"
    except Exception as exc:
        print(f"  NPU error: {exc}", flush=True)
        npupy_ms = base_cpu_ms
        template = f"cpu_fallback (error: {type(exc).__name__})"
    return _record("tanh", f"{ELEM_SIZE} int16", base_cpu_ms, blas_cpu_ms, npupy_ms,
                   template, "Pade [3/3] approx; blas_cpu_ms=base_cpu_ms (BLAS N/A)")


def bench_hash(rng: np.random.Generator, npu_hash: Optional[NpuElem]) -> dict:
    print(f"[bench] hash {ELEM_SIZE} elements (FNV-1a)", flush=True)
    x = _make_int16(rng, ELEM_SIZE, lo=-100, hi=100)

    base_cpu_ms = _time_fn(lambda: cpu_hash_int16(x))
    blas_cpu_ms = base_cpu_ms

    if npu_hash is not None:
        try:
            npupy_ms = _time_fn(lambda: npu_hash.run(x))
            template = "col_independent (hash)"
        except Exception as exc:
            print(f"  NPU error: {exc}", flush=True)
            npupy_ms = base_cpu_ms
            template = f"cpu_fallback (error: {type(exc).__name__})"
    else:
        npupy_ms = base_cpu_ms
        template = "cpu_fallback (hash NPU init failed)"
    return _record("hash", f"{ELEM_SIZE} int16", base_cpu_ms, blas_cpu_ms, npupy_ms,
                   template, "FNV-1a on int16; blas_cpu_ms=base_cpu_ms (BLAS N/A)")


def bench_relu(rng: np.random.Generator, npu_relu: NpuElem) -> dict:
    print(f"[bench] relu {ELEM_SIZE} elements", flush=True)
    x = _make_int16(rng, ELEM_SIZE, lo=-100, hi=100)

    base_cpu_ms = _time_fn(lambda: np.maximum(x, np.int16(0)))
    blas_cpu_ms = base_cpu_ms

    try:
        npupy_ms = _time_fn(lambda: npu_relu.run(x))
        template = "col_independent (relu)"
    except Exception as exc:
        print(f"  NPU error: {exc}", flush=True)
        npupy_ms = base_cpu_ms
        template = f"cpu_fallback (error: {type(exc).__name__})"
    return _record("relu", f"{ELEM_SIZE} int16", base_cpu_ms, blas_cpu_ms, npupy_ms,
                   template, "max(0,x); blas_cpu_ms=base_cpu_ms (BLAS N/A)")


def bench_mvt(rng: np.random.Generator) -> dict:
    print(f"[bench] mvt {GEMM_SIZE}×{GEMM_SIZE} (matrix-vector)", flush=True)
    N = GEMM_SIZE
    A  = _make_int16(rng, N, N)
    y1 = _make_int16(rng, N)
    y2 = _make_int16(rng, N)

    def vanilla():
        x1 = np.matmul(A, y1)
        x2 = np.matmul(A.T, y2)
        return x1, x2

    def blas_fn():
        x1 = _blas_matvec(A, y1)
        x2 = _blas_matvec(A.T.copy(), y2)
        return x1, x2

    base_cpu_ms = _time_fn(vanilla)
    blas_cpu_ms = _time_fn(blas_fn)
    npupy_ms    = blas_cpu_ms  # CPU fallback: no NPU matvec template
    return _record("mvt", f"{N}×{N}", base_cpu_ms, blas_cpu_ms, npupy_ms,
                   "cpu_fallback (matvec)", "matrix-vector; no NPU template")


def bench_atax(rng: np.random.Generator) -> dict:
    print(f"[bench] atax {GEMM_SIZE}×{GEMM_SIZE} (A^T @ (A @ x))", flush=True)
    N = GEMM_SIZE
    A = _make_int16(rng, N, N)
    x = _make_int16(rng, N)

    def vanilla():
        return np.matmul(A.T, np.matmul(A, x))

    def blas_fn():
        Ax = _blas_matvec(A, x)
        return _blas_matvec(A.T.copy(), Ax)

    base_cpu_ms = _time_fn(vanilla)
    blas_cpu_ms = _time_fn(blas_fn)
    npupy_ms    = blas_cpu_ms
    return _record("atax", f"{N}×{N}", base_cpu_ms, blas_cpu_ms, npupy_ms,
                   "cpu_fallback (matvec chain)", "A^T@(A@x); no NPU template")


def bench_correlation(rng: np.random.Generator) -> dict:
    print(f"[bench] correlation {GEMM_SIZE}×{GEMM_SIZE}", flush=True)
    N = GEMM_SIZE
    data = rng.integers(-10, 11, size=(N, N), dtype=np.int16)

    def vanilla():
        d = data.astype(np.float32)
        d -= d.mean(axis=0)
        norms = np.sqrt((d ** 2).sum(axis=0))
        norms[norms == 0] = 1.0
        d /= norms
        return d.T @ d

    base_cpu_ms = _time_fn(vanilla)
    blas_cpu_ms = base_cpu_ms  # already float32; BLAS not separately measured
    npupy_ms    = base_cpu_ms  # CPU fallback
    return _record("correlation", f"{N}×{N}", base_cpu_ms, blas_cpu_ms, npupy_ms,
                   "cpu_fallback (float32)", "float32 path; NPU is int16-only")


def bench_gramschmidt(rng: np.random.Generator) -> dict:
    print(f"[bench] gramschmidt {GEMM_SIZE}×{GEMM_SIZE} (np.linalg.qr)", flush=True)
    N = GEMM_SIZE
    A = rng.integers(-10, 11, size=(N, N), dtype=np.int16).astype(np.float32)

    def vanilla():
        return np.linalg.qr(A)

    base_cpu_ms = _time_fn(vanilla, n_warmup=1, n_iters=3)
    blas_cpu_ms = base_cpu_ms
    npupy_ms    = base_cpu_ms
    return _record("gramschmidt", f"{N}×{N}", base_cpu_ms, blas_cpu_ms, npupy_ms,
                   "cpu_fallback (np.linalg.qr)", "QR via LAPACK; NPU is int16-only")


def bench_jacobi2d(rng: np.random.Generator) -> dict:
    print(f"[bench] jacobi-2d {JACOBI_H}×{JACOBI_W} T={JACOBI_T}", flush=True)
    H, W = JACOBI_H, JACOBI_W

    inp = rng.integers(-100, 101, size=(H, W), dtype=np.int16)

    def cpu_t5():
        buf = inp.copy()
        for _ in range(JACOBI_T):
            buf = cpu_stencil_5pt(buf)
        return buf

    base_cpu_ms = _time_fn(cpu_t5)
    blas_cpu_ms = base_cpu_ms  # no BLAS for stencil

    # Try SlidingWindowTemplate
    npu_ms    = base_cpu_ms
    template  = "cpu_fallback (sliding_window unavailable)"
    try:
        from npupy_xdna.regions.region import ArraySpec, Region
        from npupy_xdna.templates.sliding_window import SlidingWindowTemplate
        from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES

        if (H, W) not in SUPPORTED_SHAPES["sliding_window"]:
            raise RuntimeError(f"{H}×{W} not in SUPPORTED_SHAPES['sliding_window']")

        region = Region(
            op="stencil_2d",
            inputs=[ArraySpec((H, W), "int16")],
            output=ArraySpec((H, W), "int16"),
            metadata={"stencil": "5pt", "iterations": 1},
        )
        sw = SlidingWindowTemplate()
        if not sw.match(region):
            raise RuntimeError("SlidingWindowTemplate.match() returned False")
        cfg      = sw.config_space(region)[0]
        print(f"  [jacobi-2d] Lowering SlidingWindowTemplate {H}×{W}...", flush=True)
        run_fn   = sw.lower(region, cfg)

        out_buf = np.zeros((H, W), dtype=np.int16)
        # warmup
        for _ in range(N_WARMUP):
            buf = inp.copy()
            buf_out = np.zeros_like(buf)
            for _ in range(JACOBI_T):
                run_fn(buf, buf_out)
                buf, buf_out = buf_out, buf

        def npu_t5():
            buf = inp.copy()
            buf_out = np.zeros_like(buf)
            for _ in range(JACOBI_T):
                run_fn(buf, buf_out)
                buf, buf_out = buf_out, buf
            return buf

        npu_ms   = _time_fn(npu_t5, n_warmup=0)  # warmup already done
        template = "sliding_window"
        print(f"  [jacobi-2d] NPU OK: {npu_ms:.3f} ms", flush=True)

    except Exception as exc:
        print(f"  [jacobi-2d] NPU unavailable ({exc}); CPU fallback", flush=True)
        npu_ms   = base_cpu_ms
        template = f"cpu_fallback ({type(exc).__name__})"

    return _record("jacobi-2d", f"{H}×{W} T={JACOBI_T}", base_cpu_ms, blas_cpu_ms, npu_ms,
                   template, "5-point stencil T=5; sliding_window attempted")


def bench_horner_poly(rng: np.random.Generator) -> dict:
    print(f"[bench] horner_poly {HORNER_N} elements degree={HORNER_DEGREE}", flush=True)
    N = HORNER_N
    x = rng.integers(-100, 101, size=N, dtype=np.int16)
    coeffs_rng = np.random.default_rng(42)
    coeffs = [int(v) for v in coeffs_rng.integers(-3, 4, size=HORNER_DEGREE + 1)]

    def horner_cpu():
        acc = np.zeros(N, dtype=np.int32)
        for c in reversed(coeffs):
            acc = acc * x.astype(np.int32) + c
        return np.clip(acc, -32768, 32767).astype(np.int16)

    base_cpu_ms = _time_fn(horner_cpu)
    blas_cpu_ms = base_cpu_ms
    npupy_ms    = base_cpu_ms  # no NPU template
    return _record("horner_poly", f"degree={HORNER_DEGREE} N={N // 1_000_000}M",
                   base_cpu_ms, blas_cpu_ms, npupy_ms,
                   "cpu_fallback (no NPU template)", "degree-8 Horner on 4M int16")


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_chart(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names       = [r["benchmark"] for r in records]
    base_vals   = [r["base_cpu_ms"]  for r in records]
    blas_vals   = [r["blas_cpu_ms"]  for r in records]
    npu_vals    = [r["npupy_ms"]     for r in records]
    blas_su     = [r["speedup_vs_blas"] for r in records]

    n    = len(names)
    x    = np.arange(n)
    w    = 0.25  # bar width

    fig, ax = plt.subplots(figsize=(18, 8))

    def _bar_vals(vals):
        """Replace -1 (skipped) with NaN so bars are absent."""
        return [v if v > 0 else float("nan") for v in vals]

    bars_base = ax.bar(x - w, _bar_vals(base_vals), w, label="Base CPU int16",
                       color="#888888", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars_blas = ax.bar(x,     _bar_vals(blas_vals), w, label="CPU BLAS (OpenBLAS)",
                       color="#4477CC", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars_npu  = ax.bar(x + w, _bar_vals(npu_vals),  w, label="NPUPy (XDNA2)",
                       color="#44AA44", alpha=0.85, edgecolor="black", linewidth=0.5)

    # Annotate speedup vs BLAS above NPU bars
    for i, (bar, su) in enumerate(zip(bars_npu, blas_su)):
        if bar.get_height() > 0 and not math.isnan(bar.get_height()) and su > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() * 1.15,
                f"{su:.1f}×",
                ha="center", va="bottom", fontsize=7, color="#226622", fontweight="bold",
            )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Execution time (ms, log scale)", fontsize=11)
    ax.set_xlabel("Benchmark", fontsize=11)
    ax.set_title("NPBench Preset L — Three-Way Performance Comparison\n"
                 "(AMD Ryzen AI 7 350, XDNA2 NPU, 2048×2048 GEMM / 4M int16 elementwise)",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0.01)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Chart saved → {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(42)

    print("=" * 70, flush=True)
    print("NPBench Preset L — Full Benchmark Suite", flush=True)
    print("=" * 70, flush=True)

    # ------------------------------------------------------------------
    # Phase 1: Compile / lower NPU templates (done once each)
    # ------------------------------------------------------------------
    print("\n[Phase 1] Compiling NPU templates...", flush=True)

    print(" Initializing NpuGemm 2048×2048...", flush=True)
    t_start = time.perf_counter()
    npu_gemm = NpuGemm(GEMM_SIZE, GEMM_SIZE, GEMM_SIZE)
    print(f"  → done in {time.perf_counter()-t_start:.1f}s", flush=True)

    print(" Initializing NpuElem tanh 4M...", flush=True)
    t_start = time.perf_counter()
    npu_tanh = NpuElem(ELEM_SIZE, compute_fn="tanh")
    print(f"  → done in {time.perf_counter()-t_start:.1f}s", flush=True)

    print(" Initializing NpuElem relu 4M...", flush=True)
    t_start = time.perf_counter()
    npu_relu = NpuElem(ELEM_SIZE, compute_fn="relu")
    print(f"  → done in {time.perf_counter()-t_start:.1f}s", flush=True)

    print(" Initializing NpuElem hash 4M...", flush=True)
    npu_hash: Optional[NpuElem] = None
    t_start = time.perf_counter()
    try:
        npu_hash = NpuElem(ELEM_SIZE, compute_fn="hash")
        print(f"  → done in {time.perf_counter()-t_start:.1f}s", flush=True)
    except Exception as exc:
        print(f"  hash NPU init failed ({exc}); will use CPU fallback", flush=True)

    # ------------------------------------------------------------------
    # Phase 2: Run all 14 benchmarks
    # ------------------------------------------------------------------
    print("\n[Phase 2] Running benchmarks...\n", flush=True)

    records: list[dict] = []

    # GEMM family
    records.append(bench_gemm(rng, npu_gemm))
    records.append(bench_gemm_relu(rng, npu_gemm))
    records.append(bench_3mm(rng, npu_gemm))
    records.append(bench_symm(rng, npu_gemm))
    records.append(bench_syr2k(rng, npu_gemm))

    # Elementwise
    records.append(bench_tanh(rng, npu_tanh))
    records.append(bench_hash(rng, npu_hash))
    records.append(bench_relu(rng, npu_relu))

    # CPU-fallback benchmarks
    records.append(bench_mvt(rng))
    records.append(bench_atax(rng))
    records.append(bench_correlation(rng))
    records.append(bench_gramschmidt(rng))
    records.append(bench_jacobi2d(rng))
    records.append(bench_horner_poly(rng))

    # ------------------------------------------------------------------
    # Phase 3: Write JSONL (overwrite)
    # ------------------------------------------------------------------
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSONL_PATH, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"\n[Output] JSONL written → {JSONL_PATH}", flush=True)

    # ------------------------------------------------------------------
    # Phase 4: Generate chart
    # ------------------------------------------------------------------
    print("\n[Phase 4] Generating speedup chart...", flush=True)
    generate_chart(records, PLOT_PATH)

    # ------------------------------------------------------------------
    # Phase 5: Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90, flush=True)
    print(f"{'Benchmark':<16} {'Size':<24} {'Base(ms)':>10} {'BLAS(ms)':>10} "
          f"{'NPU(ms)':>10} {'vs_Base':>9} {'vs_BLAS':>9} {'Template':<30}", flush=True)
    print("-" * 90, flush=True)
    for r in records:
        base_s = f"{r['base_cpu_ms']:.3f}" if r['base_cpu_ms'] > 0 else "skip"
        blas_s = f"{r['blas_cpu_ms']:.3f}" if r['blas_cpu_ms'] > 0 else "—"
        npu_s  = f"{r['npupy_ms']:.3f}"    if r['npupy_ms']    > 0 else "—"
        vb_s   = f"{r['speedup_vs_base']:.2f}×" if r['speedup_vs_base'] > 0 else "—"
        vbl_s  = f"{r['speedup_vs_blas']:.2f}×" if r['speedup_vs_blas'] > 0 else "—"
        print(f"{r['benchmark']:<16} {r['size']:<24} {base_s:>10} {blas_s:>10} "
              f"{npu_s:>10} {vb_s:>9} {vbl_s:>9} {r['template'][:30]:<30}", flush=True)
    print("=" * 90, flush=True)

    # ------------------------------------------------------------------
    # Phase 6: Write evidence
    # ------------------------------------------------------------------
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVIDENCE_PATH, "w", encoding="utf-8") as fh:
        fh.write(f"NPBench Preset-L full benchmark — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"JSONL: {JSONL_PATH}\n")
        fh.write(f"Chart: {PLOT_PATH}\n")
        fh.write(f"Total benchmarks: {len(records)}\n\n")
        fh.write(f"{'Benchmark':<16} {'Size':<24} {'Base(ms)':>10} {'BLAS(ms)':>10} "
                 f"{'NPU(ms)':>10} {'vs_BLAS':>9} {'Template':<30}\n")
        fh.write("-" * 100 + "\n")
        for r in records:
            base_s = f"{r['base_cpu_ms']:.3f}" if r['base_cpu_ms'] > 0 else "skip"
            blas_s = f"{r['blas_cpu_ms']:.3f}" if r['blas_cpu_ms'] > 0 else "—"
            npu_s  = f"{r['npupy_ms']:.3f}"    if r['npupy_ms']    > 0 else "—"
            vbl_s  = f"{r['speedup_vs_blas']:.2f}×" if r['speedup_vs_blas'] > 0 else "—"
            fh.write(f"{r['benchmark']:<16} {r['size']:<24} {base_s:>10} {blas_s:>10} "
                     f"{npu_s:>10} {vbl_s:>9} {r['template'][:30]:<30}\n")
    print(f"[Evidence] Written → {EVIDENCE_PATH}", flush=True)

    print("\n[DONE] Preset-L full benchmark complete.", flush=True)


if __name__ == "__main__":
    main()
