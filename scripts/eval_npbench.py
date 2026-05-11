from __future__ import annotations

import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

RESULTS_DIR = Path(__file__).parents[1] / "results"
JSONL_PATH = RESULTS_DIR / "04_npbench_evaluation.jsonl"
MD_PATH = RESULTS_DIR / "04_npbench_evaluation.md"

N_WARMUP = 5
N_ITERS = 10
SIZE = 256


def _make_int16(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.integers(-10, 11, size=shape, dtype=np.int16)


def _time_fn(fn: Callable, n_warmup: int = N_WARMUP, n_iters: int = N_ITERS) -> float:
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def _record(
    name: str,
    vanilla_ms: float,
    npu_ms: float,
    template: str,
    correct: bool,
    preset: str,
    note: str,
) -> dict[str, Any]:
    is_nan = npu_ms != npu_ms
    npu_val = None if is_nan else round(npu_ms, 3)
    speedup = None if (is_nan or not npu_ms) else round(vanilla_ms / npu_ms, 3)
    return {
        "benchmark": name,
        "vanilla_numpy_ms": round(vanilla_ms, 3),
        "npupy_ms": npu_val,
        "speedup": speedup,
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
        from npupy_xdna.templates.protocol import Config

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


def bench_gemm(rng: np.random.Generator, npu: NpuMatmul) -> dict[str, Any]:
    A = _make_int16(rng, SIZE, SIZE)
    B = _make_int16(rng, SIZE, SIZE)

    vanilla_ms = _time_fn(lambda: np.matmul(A, B))
    ref = np.matmul(A, B)

    try:
        npu_ms = _time_fn(lambda: npu.run(A, B))
        npu_result = npu.run(A, B)
        correct = bool(np.allclose(ref.astype(np.float32), npu_result.astype(np.float32), atol=1))
    except Exception as exc:
        npu_ms = float("nan")
        correct = False
        return _record("gemm", vanilla_ms, npu_ms, f"error:{exc}", correct, f"M=N=K={SIZE}", "")

    return _record("gemm", vanilla_ms, npu_ms, "gemm_fusion", correct, f"M=N=K={SIZE}", "")


def bench_3mm(rng: np.random.Generator, npu: NpuMatmul) -> dict[str, Any]:
    A = _make_int16(rng, SIZE, SIZE)
    B = _make_int16(rng, SIZE, SIZE)
    C = _make_int16(rng, SIZE, SIZE)
    D = _make_int16(rng, SIZE, SIZE)

    def vanilla():
        AB = np.matmul(A, B)
        CD = np.matmul(C, D)
        return np.matmul(AB, CD)

    def npu_fn():
        AB = npu.run(A, B)
        CD = npu.run(C, D)
        return npu.run(AB, CD)

    vanilla_ms = _time_fn(vanilla)
    ref = vanilla()

    try:
        npu_ms = _time_fn(npu_fn)
        npu_result = npu_fn()
        correct = bool(np.allclose(ref.astype(np.float64), npu_result.astype(np.float64), atol=64))
    except Exception as exc:
        npu_ms = float("nan")
        correct = False
        return _record("3mm", vanilla_ms, npu_ms, f"error:{exc}", correct, f"N={SIZE}", "3 chained matmuls")

    return _record("3mm", vanilla_ms, npu_ms, "gemm_fusion", correct, f"N={SIZE}", "3 chained matmuls")


def bench_symm(rng: np.random.Generator, npu: NpuMatmul) -> dict[str, Any]:
    A_upper = _make_int16(rng, SIZE, SIZE)
    A = (A_upper + A_upper.T).astype(np.int16)
    B = _make_int16(rng, SIZE, SIZE)

    vanilla_ms = _time_fn(lambda: np.matmul(A, B))
    ref = np.matmul(A, B)

    try:
        npu_ms = _time_fn(lambda: npu.run(A, B))
        npu_result = npu.run(A, B)
        correct = bool(np.allclose(ref.astype(np.float32), npu_result.astype(np.float32), atol=1))
    except Exception as exc:
        npu_ms = float("nan")
        correct = False
        return _record("symm", vanilla_ms, npu_ms, f"error:{exc}", correct, f"M=N={SIZE}", "symmetric A")

    return _record("symm", vanilla_ms, npu_ms, "gemm_fusion", correct, f"M=N={SIZE}", "symmetric A")


def bench_syr2k(rng: np.random.Generator, npu: NpuMatmul) -> dict[str, Any]:
    A = _make_int16(rng, SIZE, SIZE)
    B = _make_int16(rng, SIZE, SIZE)

    def vanilla():
        return np.matmul(A, B.T) + np.matmul(B, A.T)

    def npu_fn():
        return npu.run(A, B.T.copy()) + npu.run(B, A.T.copy())

    vanilla_ms = _time_fn(vanilla)
    ref = vanilla()

    try:
        npu_ms = _time_fn(npu_fn)
        npu_result = npu_fn()
        correct = bool(np.allclose(ref.astype(np.float32), npu_result.astype(np.float32), atol=2))
    except Exception as exc:
        npu_ms = float("nan")
        correct = False
        return _record("syr2k", vanilla_ms, npu_ms, f"error:{exc}", correct, f"N={SIZE}", "2 matmuls + add")

    return _record("syr2k", vanilla_ms, npu_ms, "gemm_fusion", correct, f"N={SIZE}", "2 matmuls + add")


def bench_mvt(rng: np.random.Generator, _npu: NpuMatmul) -> dict[str, Any]:
    A = _make_int16(rng, SIZE, SIZE)
    y1 = _make_int16(rng, SIZE)
    y2 = _make_int16(rng, SIZE)

    def vanilla():
        x1 = np.matmul(A, y1)
        x2 = np.matmul(A.T, y2)
        return x1, x2

    vanilla_ms = _time_fn(vanilla)
    ref_x1, ref_x2 = vanilla()
    npu_ms = vanilla_ms
    correct = True

    return _record("mvt", vanilla_ms, npu_ms, "cpu_fallback (matvec)", correct, f"N={SIZE}", "matrix-vector; no NPU template")


def bench_atax(rng: np.random.Generator, _npu: NpuMatmul) -> dict[str, Any]:
    A = _make_int16(rng, SIZE, SIZE)
    x = _make_int16(rng, SIZE)

    def vanilla():
        return np.matmul(A.T, np.matmul(A, x))

    vanilla_ms = _time_fn(vanilla)
    ref = vanilla()
    npu_ms = vanilla_ms
    correct = True

    return _record("atax", vanilla_ms, npu_ms, "cpu_fallback (matvec)", correct, f"N={SIZE}", "matrix-vector chain; no NPU template")


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

    vanilla_ms = _time_fn(vanilla)
    ref = vanilla()
    npu_ms = vanilla_ms
    correct = True

    return _record("correlation", vanilla_ms, npu_ms, "cpu_fallback (float32)", correct, f"N={SIZE}", "float32 path; NPU is int16-only")


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

    vanilla_ms = _time_fn(vanilla, n_warmup=1, n_iters=3)
    ref_Q, ref_R = vanilla()
    npu_ms = vanilla_ms
    correct = True

    return _record("gramschmidt", vanilla_ms, npu_ms, "cpu_fallback (dot/scalar)", correct, f"N={SIZE}", "column-wise QR; no matmul pattern for NPU")


def bench_gemm_relu(rng: np.random.Generator, npu: NpuMatmul) -> dict[str, Any]:
    A = _make_int16(rng, SIZE, SIZE)
    B = _make_int16(rng, SIZE, SIZE)

    def vanilla():
        return np.maximum(np.matmul(A, B), np.int16(0))

    def npu_fn():
        return np.maximum(npu.run(A, B), np.int16(0))

    vanilla_ms = _time_fn(vanilla)
    ref = vanilla()

    try:
        npu_ms = _time_fn(npu_fn)
        npu_result = npu_fn()
        correct = bool(np.allclose(ref.astype(np.float32), npu_result.astype(np.float32), atol=1))
    except Exception as exc:
        npu_ms = float("nan")
        correct = False
        return _record("gemm_relu", vanilla_ms, npu_ms, f"error:{exc}", correct, f"M=N=K={SIZE}", "synthetic GEMM+ReLU")

    return _record("gemm_relu", vanilla_ms, npu_ms, "gemm_fusion+cpu_relu", correct, f"M=N=K={SIZE}", "synthetic GEMM+ReLU")


BENCHMARKS = [
    ("gemm", bench_gemm),
    ("3mm", bench_3mm),
    ("symm", bench_symm),
    ("syr2k", bench_syr2k),
    ("mvt", bench_mvt),
    ("atax", bench_atax),
    ("correlation", bench_correlation),
    ("gramschmidt", bench_gramschmidt),
    ("gemm_relu", bench_gemm_relu),
]


def run_all() -> list[dict[str, Any]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if JSONL_PATH.exists():
        JSONL_PATH.unlink()

    print("Initializing NpuMatmul (loads iron_fn once)...", flush=True)
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
                "vanilla_numpy_ms": None,
                "npupy_ms": None,
                "speedup": None,
                "template_used": "error",
                "correctness": False,
                "preset": f"N={SIZE}",
                "note": str(e),
            }

        speedup_str = f"{rec['speedup']:.3f}x" if rec.get("speedup") else "N/A"
        correct_str = "OK" if rec.get("correctness") else "MISMATCH"
        print(
            f"  vanilla={rec['vanilla_numpy_ms']}ms  npupy={rec['npupy_ms']}ms"
            f"  speedup={speedup_str}  {correct_str}",
            flush=True,
        )

        with open(JSONL_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
        results.append(rec)

    return results


def write_markdown(results: list[dict[str, Any]]) -> None:
    lines = [
        "# NPBench Evaluation Results",
        "",
        f"Size preset: N=256 | Warmup={N_WARMUP}, Iterations={N_ITERS} (gramschmidt: 1+3)",
        "",
        "| Benchmark | Vanilla NumPy (ms) | NPUPy (ms) | Speedup | Template | Correct |",
        "|-----------|-------------------|------------|---------|----------|---------|",
    ]
    for r in results:
        v = f"{r['vanilla_numpy_ms']:.3f}" if r.get("vanilla_numpy_ms") is not None else "N/A"
        n = f"{r['npupy_ms']:.3f}" if r.get("npupy_ms") is not None else "N/A"
        s = f"{r['speedup']:.3f}x" if r.get("speedup") is not None else "N/A"
        t = r.get("template_used", "N/A")
        c = "YES" if r.get("correctness") else "NO"
        lines.append(f"| {r['benchmark']} | {v} | {n} | {s} | {t} | {c} |")

    lines += [
        "",
        "## Notes",
        "",
        "- gemm_fusion template handles 256x256 int16 matmuls (NPU dispatch).",
        "- mvt / atax: matrix-vector ops have no NPU template; CPU-only.",
        "- gramschmidt: column-wise QR with np.dot; CPU-only.",
        "- correlation: float32 normalization; NPU template is int16-only.",
        "- gemm_relu: GEMM on NPU + ReLU on CPU; overall speedup driven by matmul.",
        "- NPUPy uses NpuRunner directly (iron_fn created once, reused per shape).",
        "",
    ]

    MD_PATH.write_text("\n".join(lines))
    print(f"\nMarkdown written to {MD_PATH}")


if __name__ == "__main__":
    results = run_all()
    write_markdown(results)
    print(f"\nDone. Results in {JSONL_PATH}")
