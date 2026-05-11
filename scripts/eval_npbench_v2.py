from __future__ import annotations

import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

RESULTS_DIR = Path(__file__).parents[1] / "results"
JSONL_PATH = RESULTS_DIR / "04_npbench_evaluation_v2.jsonl"
MD_PATH = RESULTS_DIR / "04_npbench_evaluation_v2.md"

N_WARMUP = 5
N_ITERS = 10
SIZE = 256
HORNER_N = 256 * 1024


def _make_int16(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.integers(-10, 11, size=shape, dtype=np.int16)


def _time_fn(
    fn: Callable,
    n_warmup: int = N_WARMUP,
    n_iters: int = N_ITERS,
    timeout_s: float = 300.0,
) -> float:
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - t0) * 1e3
        times.append(elapsed)
        if elapsed > timeout_s * 1e3:
            break
    return statistics.median(times)


def _record(
    name: str,
    int16_cpu_ms: float,
    blas_cpu_ms: Optional[float],
    npupy_ms: Optional[float],
    template: str,
    correct: bool,
    preset: str,
    note: str,
) -> dict[str, Any]:
    def _speedup(npu: Optional[float], cpu: Optional[float]) -> Optional[float]:
        if npu is None or cpu is None:
            return None
        if math.isnan(npu) or npu == 0:
            return None
        return round(cpu / npu, 3)

    npu_val = None if (npupy_ms is None or math.isnan(npupy_ms)) else round(npupy_ms, 3)
    blas_val = None if (blas_cpu_ms is None) else round(blas_cpu_ms, 3)

    return {
        "benchmark": name,
        "int16_cpu_ms": round(int16_cpu_ms, 3),
        "blas_cpu_ms": blas_val,
        "npupy_ms": npu_val,
        "speedup_vs_int16": _speedup(npupy_ms, int16_cpu_ms),
        "speedup_vs_blas": _speedup(npupy_ms, blas_cpu_ms),
        "template_used": template,
        "correctness": correct,
        "preset": preset,
        "note": note,
    }


class NpuMatmul:
    def __init__(self):
        from npupy_xdna.regions.region import ArraySpec, Region
        from npupy_xdna.runtime.npu_runner import NpuRunner
        from npupy_xdna.templates.gemm_fusion import GemmFusionTemplate

        M = K = N = SIZE
        self.region = Region(
            op="matmul",
            inputs=[
                ArraySpec(shape=(M, K), dtype="int16"),
                ArraySpec(shape=(K, N), dtype="int16"),
            ],
            output=ArraySpec(shape=(M, N), dtype="int16"),
        )
        tmpl = GemmFusionTemplate()
        configs = tmpl.config_space(self.region)
        self.config = next(c for c in configs if c.extra.get("epilogue") == "none")
        self.iron_fn = tmpl.lower(self.region, self.config)
        self.runner = NpuRunner()

    def run(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        B_T = np.ascontiguousarray(B.T)
        result = self.runner.run(self.region, self.config, self.iron_fn, [A, B_T])
        if result.status != "ok":
            raise RuntimeError(f"NPU run failed: {result.status}")
        return result.output


def _blas_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r = np.matmul(A.astype(np.float32), B.astype(np.float32))
    return np.clip(r, -32768, 32767).astype(np.int16)


def _blas_matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    r = np.matmul(A.astype(np.float32), x.astype(np.float32))
    return np.clip(r, -32768, 32767).astype(np.int16)


def bench_gemm(rng: np.random.Generator, npu: NpuMatmul) -> dict[str, Any]:
    A = _make_int16(rng, SIZE, SIZE)
    B = _make_int16(rng, SIZE, SIZE)

    int16_ms = _time_fn(lambda: np.matmul(A, B))
    blas_ms  = _time_fn(lambda: _blas_matmul(A, B))
    ref = np.matmul(A, B)

    try:
        npu_ms = _time_fn(lambda: npu.run(A, B), timeout_s=60.0)
        npu_result = npu.run(A, B)
        correct = bool(np.allclose(ref.astype(np.float32), npu_result.astype(np.float32), atol=1))
        tmpl = "gemm_fusion"
    except Exception as exc:
        npu_ms = float("nan")
        correct = False
        tmpl = f"error:{exc}"

    return _record("gemm", int16_ms, blas_ms, npu_ms, tmpl, correct, f"M=N=K={SIZE}", "")


def bench_3mm(rng: np.random.Generator, npu: NpuMatmul) -> dict[str, Any]:
    A = _make_int16(rng, SIZE, SIZE)
    B = _make_int16(rng, SIZE, SIZE)
    C = _make_int16(rng, SIZE, SIZE)
    D = _make_int16(rng, SIZE, SIZE)

    def vanilla():
        AB = np.matmul(A, B)
        CD = np.matmul(C, D)
        return np.matmul(AB, CD)

    def blas_fn():
        AB = _blas_matmul(A, B)
        CD = _blas_matmul(C, D)
        return _blas_matmul(AB, CD)

    def npu_fn():
        AB = npu.run(A, B)
        CD = npu.run(C, D)
        return npu.run(AB, CD)

    int16_ms = _time_fn(vanilla)
    blas_ms  = _time_fn(blas_fn)
    ref = vanilla()

    try:
        npu_ms = _time_fn(npu_fn, timeout_s=60.0)
        npu_result = npu_fn()
        correct = bool(np.allclose(ref.astype(np.float64), npu_result.astype(np.float64), atol=64))
        tmpl = "gemm_fusion"
    except Exception as exc:
        npu_ms = float("nan")
        correct = False
        tmpl = f"error:{exc}"

    return _record("3mm", int16_ms, blas_ms, npu_ms, tmpl, correct, f"N={SIZE}", "3 chained matmuls")


def bench_symm(rng: np.random.Generator, npu: NpuMatmul) -> dict[str, Any]:
    A_upper = _make_int16(rng, SIZE, SIZE)
    A = (A_upper + A_upper.T).astype(np.int16)
    B = _make_int16(rng, SIZE, SIZE)

    int16_ms = _time_fn(lambda: np.matmul(A, B))
    blas_ms  = _time_fn(lambda: _blas_matmul(A, B))
    ref = np.matmul(A, B)

    try:
        npu_ms = _time_fn(lambda: npu.run(A, B), timeout_s=60.0)
        npu_result = npu.run(A, B)
        correct = bool(np.allclose(ref.astype(np.float32), npu_result.astype(np.float32), atol=1))
        tmpl = "gemm_fusion"
    except Exception as exc:
        npu_ms = float("nan")
        correct = False
        tmpl = f"error:{exc}"

    return _record("symm", int16_ms, blas_ms, npu_ms, tmpl, correct, f"M=N={SIZE}", "symmetric A")


def bench_syr2k(rng: np.random.Generator, npu: NpuMatmul) -> dict[str, Any]:
    A = _make_int16(rng, SIZE, SIZE)
    B = _make_int16(rng, SIZE, SIZE)

    def vanilla():
        return np.matmul(A, B.T) + np.matmul(B, A.T)

    def blas_fn():
        return _blas_matmul(A, B.T.copy()) + _blas_matmul(B, A.T.copy())

    def npu_fn():
        return npu.run(A, B.T.copy()) + npu.run(B, A.T.copy())

    int16_ms = _time_fn(vanilla)
    blas_ms  = _time_fn(blas_fn)
    ref = vanilla()

    try:
        npu_ms = _time_fn(npu_fn, timeout_s=60.0)
        npu_result = npu_fn()
        correct = bool(np.allclose(ref.astype(np.float32), npu_result.astype(np.float32), atol=2))
        tmpl = "gemm_fusion"
    except Exception as exc:
        npu_ms = float("nan")
        correct = False
        tmpl = f"error:{exc}"

    return _record("syr2k", int16_ms, blas_ms, npu_ms, tmpl, correct, f"N={SIZE}", "2 matmuls + add")


def bench_mvt(rng: np.random.Generator, _npu: NpuMatmul) -> dict[str, Any]:
    A  = _make_int16(rng, SIZE, SIZE)
    y1 = _make_int16(rng, SIZE)
    y2 = _make_int16(rng, SIZE)

    def vanilla():
        x1 = np.matmul(A, y1)
        x2 = np.matmul(A.T, y2)
        return x1, x2

    def blas_fn():
        x1 = _blas_matvec(A, y1)
        x2 = _blas_matvec(A.T, y2)
        return x1, x2

    int16_ms = _time_fn(vanilla)
    blas_ms  = _time_fn(blas_fn)

    return _record(
        "mvt", int16_ms, blas_ms, int16_ms,
        "cpu_fallback (matvec)", True, f"N={SIZE}",
        "matrix-vector; no NPU template",
    )


def bench_atax(rng: np.random.Generator, _npu: NpuMatmul) -> dict[str, Any]:
    A = _make_int16(rng, SIZE, SIZE)
    x = _make_int16(rng, SIZE)

    def vanilla():
        return np.matmul(A.T, np.matmul(A, x))

    def blas_fn():
        Ax  = _blas_matvec(A, x)
        return _blas_matvec(A.T, Ax)

    int16_ms = _time_fn(vanilla)
    blas_ms  = _time_fn(blas_fn)

    return _record(
        "atax", int16_ms, blas_ms, int16_ms,
        "cpu_fallback (matvec)", True, f"N={SIZE}",
        "matrix-vector chain; no NPU template",
    )


def bench_correlation(rng: np.random.Generator, _npu: NpuMatmul) -> dict[str, Any]:
    N = SIZE
    data = rng.integers(-10, 11, size=(N, N), dtype=np.int16)

    def vanilla():
        d = data.astype(np.float32)
        d -= d.mean(axis=0)
        norms = np.sqrt((d ** 2).sum(axis=0))
        norms[norms == 0] = 1.0
        d /= norms
        return np.matmul(d.T, d)

    int16_ms = _time_fn(vanilla)
    blas_ms = int16_ms

    return _record(
        "correlation", int16_ms, blas_ms, int16_ms,
        "cpu_fallback (float32)", True, f"N={SIZE}",
        "float32 path; NPU is int16-only; blas_ms=int16_ms (already f32)",
    )


def bench_gramschmidt(rng: np.random.Generator, _npu: NpuMatmul) -> dict[str, Any]:
    N = SIZE
    A = rng.integers(-10, 11, size=(N, N), dtype=np.int16).astype(np.float32)

    def vanilla():
        Q = np.zeros_like(A)
        R = np.zeros((N, N), dtype=np.float32)
        for k in range(N):
            v = A[:, k].copy()
            for j in range(k):
                R[j, k] = np.dot(Q[:, j], A[:, k])
                v -= R[j, k] * Q[:, j]
            nrm = np.linalg.norm(v)
            if nrm > 1e-10:
                R[k, k] = nrm
                Q[:, k] = v / nrm
        return Q, R

    int16_ms = _time_fn(vanilla, n_warmup=1, n_iters=3)
    blas_ms = int16_ms

    return _record(
        "gramschmidt", int16_ms, blas_ms, int16_ms,
        "cpu_fallback (dot/scalar)", True, f"N={SIZE}",
        "column-wise QR; float32 path; blas_ms=int16_ms (already f32)",
    )


def bench_gemm_relu(rng: np.random.Generator, npu: NpuMatmul) -> dict[str, Any]:
    A = _make_int16(rng, SIZE, SIZE)
    B = _make_int16(rng, SIZE, SIZE)

    def vanilla():
        return np.maximum(np.matmul(A, B), np.int16(0))

    def blas_fn():
        return np.maximum(_blas_matmul(A, B), np.int16(0))

    def npu_fn():
        return np.maximum(npu.run(A, B), np.int16(0))

    int16_ms = _time_fn(vanilla)
    blas_ms  = _time_fn(blas_fn)
    ref = vanilla()

    try:
        npu_ms = _time_fn(npu_fn, timeout_s=60.0)
        npu_result = npu_fn()
        correct = bool(np.allclose(ref.astype(np.float32), npu_result.astype(np.float32), atol=1))
        tmpl = "gemm_fusion+cpu_relu"
    except Exception as exc:
        npu_ms = float("nan")
        correct = False
        tmpl = f"error:{exc}"

    return _record(
        "gemm_relu", int16_ms, blas_ms, npu_ms, tmpl, correct,
        f"M=N=K={SIZE}", "synthetic GEMM+ReLU",
    )


def _cpu_stencil_5pt(inp: np.ndarray) -> np.ndarray:
    H, W = inp.shape
    inp32 = inp.astype(np.int32)
    padded = np.zeros((H + 2, W), dtype=np.int32)
    padded[1 : H + 1, :] = inp32
    center = padded[1 : H + 1, :]
    top    = padded[0:H, :]
    bot    = padded[2 : H + 2, :]
    left   = np.zeros((H, W), dtype=np.int32)
    left[:, 1:] = center[:, :-1]
    right  = np.zeros((H, W), dtype=np.int32)
    right[:, :-1] = center[:, 1:]
    s = center + top + bot + left + right
    v = np.trunc(s.astype(np.float64) / 5.0).astype(np.int32)
    v[:, 0]  = 0
    v[:, -1] = 0
    return np.clip(v, -32768, 32767).astype(np.int16)


def bench_jacobi2d(rng: np.random.Generator, _npu: NpuMatmul) -> dict[str, Any]:
    H = W = SIZE
    inp = rng.integers(-100, 101, size=(H, W), dtype=np.int16)

    int16_ms = _time_fn(lambda: _cpu_stencil_5pt(inp))
    template_name = "sliding_window (attempted)"
    npu_ms: Optional[float] = None
    correct = False

    try:
        from npupy_xdna.regions.region import ArraySpec, Region
        from npupy_xdna.templates.sliding_window import SlidingWindowTemplate

        region = Region(
            op="stencil_2d",
            inputs=[ArraySpec((H, W), "int16")],
            output=ArraySpec((H, W), "int16"),
            metadata={"stencil": "5pt", "iterations": 1},
        )
        tmpl = SlidingWindowTemplate()
        if not tmpl.match(region):
            raise RuntimeError(f"SlidingWindowTemplate.match() returned False for {H}x{W}")

        cfg = tmpl.config_space(region)[0]
        print(f"  [jacobi-2d] Lowering SlidingWindowTemplate {H}x{W} (JIT compile)...", flush=True)
        run_fn = tmpl.lower(region, cfg)

        out_buf = np.zeros((H, W), dtype=np.int16)
        for _ in range(3):
            run_fn(inp, out_buf)

        times = []
        for _ in range(N_ITERS):
            out_buf_i = np.zeros((H, W), dtype=np.int16)
            t0 = time.perf_counter()
            run_fn(inp, out_buf_i)
            times.append((time.perf_counter() - t0) * 1e3)

        npu_ms = statistics.median(times)
        template_name = "sliding_window"

        ref = _cpu_stencil_5pt(inp)
        correct = bool(np.allclose(ref.astype(np.float32), out_buf_i.astype(np.float32), atol=1))

        print(f"  [jacobi-2d] NPU OK: {npu_ms:.3f}ms  correct={correct}", flush=True)

    except Exception as exc:
        print(f"  [jacobi-2d] SlidingWindowTemplate failed ({exc}); CPU fallback", flush=True)
        npu_ms = int16_ms
        correct = True
        template_name = f"cpu_fallback (sliding_window unavailable: {type(exc).__name__})"

    return _record(
        "jacobi-2d", int16_ms, None, npu_ms, template_name, correct,
        f"{H}x{W} int16 T=1",
        "5-point stencil; sliding_window template attempted",
    )


def bench_horner(rng: np.random.Generator, _npu: NpuMatmul) -> dict[str, Any]:
    DEGREE = 8
    N = HORNER_N
    x = rng.integers(-100, 101, size=N, dtype=np.int16)
    coeffs_rng = np.random.default_rng(42)
    coeffs = [int(v) for v in coeffs_rng.integers(-3, 4, size=DEGREE + 1)]

    def horner_cpu():
        acc = np.zeros(N, dtype=np.int32)
        for c in reversed(coeffs):
            acc = acc * x.astype(np.int32) + c
        return np.clip(acc, -32768, 32767).astype(np.int16)

    int16_ms = _time_fn(horner_cpu)

    return _record(
        "horner_poly", int16_ms, None, int16_ms,
        "cpu_fallback (no NPU template)", True,
        f"degree={DEGREE} N={N//1024}K",
        "Horner polynomial eval; no NPU dispatch path",
    )


BENCHMARKS = [
    ("gemm",        bench_gemm),
    ("3mm",         bench_3mm),
    ("symm",        bench_symm),
    ("syr2k",       bench_syr2k),
    ("mvt",         bench_mvt),
    ("atax",        bench_atax),
    ("correlation", bench_correlation),
    ("gramschmidt", bench_gramschmidt),
    ("gemm_relu",   bench_gemm_relu),
    ("jacobi-2d",   bench_jacobi2d),
    ("horner_poly", bench_horner),
]


def run_all() -> list[dict[str, Any]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if JSONL_PATH.exists():
        JSONL_PATH.unlink()

    print("Initializing NpuMatmul (loads GemmFusion iron_fn once)...", flush=True)
    npu = NpuMatmul()
    print("NpuMatmul ready.", flush=True)

    rng = np.random.default_rng(42)
    results = []

    for name, bench_fn in BENCHMARKS:
        print(f"\n[{name}] running...", flush=True)
        try:
            rec = bench_fn(rng, npu)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            rec = {
                "benchmark": name,
                "int16_cpu_ms": None,
                "blas_cpu_ms": None,
                "npupy_ms": None,
                "speedup_vs_int16": None,
                "speedup_vs_blas": None,
                "template_used": "error",
                "correctness": False,
                "preset": f"N={SIZE}",
                "note": str(e),
            }

        sv_int16 = rec.get("speedup_vs_int16")
        sv_blas  = rec.get("speedup_vs_blas")
        s_int16_str = f"{sv_int16:.3f}x" if sv_int16 else "N/A"
        s_blas_str  = f"{sv_blas:.3f}x"  if sv_blas  else "N/A"
        correct_str = "OK" if rec.get("correctness") else "MISMATCH"
        print(
            f"  int16={rec.get('int16_cpu_ms')}ms  "
            f"blas={rec.get('blas_cpu_ms')}ms  "
            f"npupy={rec.get('npupy_ms')}ms  "
            f"speedup_int16={s_int16_str}  speedup_blas={s_blas_str}  {correct_str}",
            flush=True,
        )

        with open(JSONL_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
        results.append(rec)

    return results


def write_markdown(results: list[dict[str, Any]]) -> None:
    lines = [
        "# NPBench V2 Evaluation Results",
        "",
        f"Size preset: N={SIZE} | Warmup={N_WARMUP}, Iterations={N_ITERS}",
        "",
        "| Benchmark | int16 CPU (ms) | BLAS CPU (ms) | NPUPy (ms) | vs int16 | vs BLAS | Template | Correct |",
        "|-----------|---------------|--------------|------------|---------|--------|----------|---------|",
    ]
    for r in results:
        i  = f"{r['int16_cpu_ms']:.3f}" if r.get("int16_cpu_ms") is not None else "N/A"
        b  = f"{r['blas_cpu_ms']:.3f}"  if r.get("blas_cpu_ms")  is not None else "—"
        n  = f"{r['npupy_ms']:.3f}"     if r.get("npupy_ms")     is not None else "N/A"
        si = f"{r['speedup_vs_int16']:.2f}x" if r.get("speedup_vs_int16") else "1.0x"
        sb = f"{r['speedup_vs_blas']:.2f}x"  if r.get("speedup_vs_blas")  else "—"
        t  = r.get("template_used", "N/A")
        c  = "YES" if r.get("correctness") else "NO"
        lines.append(f"| {r['benchmark']} | {i} | {b} | {n} | {si} | {sb} | {t} | {c} |")

    lines += [
        "",
        "## Notes",
        "",
        "- **int16_cpu_ms**: NumPy int16 baseline (no BLAS acceleration – numpy doesn't dispatch int16 to BLAS).",
        "- **blas_cpu_ms**: Honest baseline: int16→f32 cast, BLAS matmul, clip→int16. `—` = no BLAS path.",
        "- **speedup_vs_int16**: NPUPy speedup vs int16 numpy (favorable; int16 numpy is slow).",
        "- **speedup_vs_blas**: NPUPy speedup vs BLAS round-trip (honest; BLAS is fast). `—` = not applicable.",
        "- mvt/atax: matvec; no NPU template → speedup_vs_blas reflects BLAS matvec round-trip.",
        "- correlation/gramschmidt: already float32 path internally → blas_cpu_ms ≈ int16_cpu_ms.",
        "- jacobi-2d: 5-point stencil; SlidingWindowTemplate attempted (256×256 is supported shape).",
        "- horner_poly: degree-8 polynomial, 256K elements; no NPU dispatch path.",
        "",
    ]

    MD_PATH.write_text("\n".join(lines))
    print(f"\nMarkdown written to {MD_PATH}", flush=True)


def write_summary(results: list[dict[str, Any]]) -> None:
    npu_wins = [
        r for r in results
        if r.get("speedup_vs_int16") and r["speedup_vs_int16"] >= 1.2
    ]
    blas_wins = [
        r for r in results
        if r.get("speedup_vs_blas") and r["speedup_vs_blas"] >= 1.2
    ]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total benchmarks : {len(results)}")
    print(f"  ≥1.2x vs int16   : {len(npu_wins)} benchmarks")
    print(f"  ≥1.2x vs BLAS    : {len(blas_wins)} benchmarks")
    if npu_wins:
        print(f"  NPU winners (int16): {[r['benchmark'] for r in npu_wins]}")
    if blas_wins:
        print(f"  NPU winners (BLAS):  {[r['benchmark'] for r in blas_wins]}")
    print(f"\n  Results: {JSONL_PATH}")
    print(f"  Markdown: {MD_PATH}")


if __name__ == "__main__":
    results = run_all()
    write_markdown(results)
    write_summary(results)
