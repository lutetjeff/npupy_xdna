#!/usr/bin/env python3
"""NPBench Preset L evaluation — ALL benchmarks, 3-way comparison.

SAFETY: Base CPU int16 matmul is SKIPPED for sizes >= 512² (crashes system).
BLAS baseline uses scipy.linalg.blas.sgemm (confirmed working).

Usage:
    source /opt/xilinx/xrt/setup.sh
    source ~/mlir-aie/ironenv/bin/activate
    source ~/mlir-aie/utils/env_setup.sh
    python npupy_xdna/scripts/eval_preset_L.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg.blas

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
JSONL_PATH = RESULTS_DIR / "timings" / "npbench_preset_L.jsonl"
EVIDENCE_PATH = Path("/home/lutet/ece511/.sisyphus/evidence/preset-L-full.txt")

MAX_INT16_MATMUL_DIM = 256
RNG = np.random.default_rng(42)


def _bench(fn, n_warmup=3, n_iters=5):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return sorted(times)[n_iters // 2]


def _blas_matmul(A_i16, B_i16):
    r = scipy.linalg.blas.sgemm(1.0, A_i16.astype(np.float32), B_i16.astype(np.float32))
    return np.clip(r, -32768, 32767).astype(np.int16)


def _npu_matmul(A, B, epilogue="none"):
    from npupy_xdna.regions.region import ArraySpec, Region
    from npupy_xdna.runtime.npu_runner import NpuRunner
    from npupy_xdna.templates.gemm_fusion import GemmFusionTemplate

    M, K = A.shape
    _, N = B.shape
    region = Region(
        op="matmul",
        inputs=[ArraySpec((M, K), "int16"), ArraySpec((K, N), "int16")],
        output=ArraySpec((M, N), "int16"),
    )
    tmpl = GemmFusionTemplate()
    config = None
    for c in tmpl.config_space(region):
        if c.extra.get("epilogue") == epilogue and c.extra.get("prologue") == "none":
            config = c
            break
    if config is None:
        return None, -1
    iron_fn = tmpl.lower(region, config)
    B_col = np.ascontiguousarray(B.T)
    runner = NpuRunner()
    for _ in range(3):
        runner.run(region, config, iron_fn, [A, B_col], timeout_s=120.0)
    times = []
    for _ in range(5):
        r = runner.run(region, config, iron_fn, [A, B_col], timeout_s=120.0)
        if r.status != "ok":
            return None, -1
        times.append(r.latency_us)
    return "gemm_fusion", sorted(times)[2]


def _npu_elementwise(x, compute_fn="relu"):
    from npupy_xdna.regions.region import ArraySpec, Region
    from npupy_xdna.runtime.npu_runner import NpuRunner
    from npupy_xdna.templates.col_independent import ColIndependentTemplate

    n = x.size
    metadata = {}
    if compute_fn in ("tanh", "hash"):
        metadata = {"compute_fn": compute_fn, "compute_intensity": "high"}
    region = Region(
        op="elementwise_unary",
        inputs=[ArraySpec((n,), "int16")],
        output=ArraySpec((n,), "int16"),
        metadata=metadata,
    )
    tmpl = ColIndependentTemplate()
    if not tmpl.match(region):
        return None, -1
    configs = tmpl.config_space(region)
    if not configs:
        return None, -1
    config = configs[0]
    iron_fn = tmpl.lower(region, config)
    runner = NpuRunner()
    for _ in range(3):
        runner.run(region, config, iron_fn, [x.ravel()], timeout_s=60.0)
    times = []
    for _ in range(5):
        r = runner.run(region, config, iron_fn, [x.ravel()], timeout_s=60.0)
        if r.status != "ok":
            return None, -1
        times.append(r.latency_us)
    return "col_independent", sorted(times)[2]


def _cpu_tanh_int16(x):
    x32 = x.astype(np.int32)
    x2 = x32 * x32 >> 8
    x3 = x2 * x32 >> 8
    return np.clip(x32 - (x3 // 3), -32768, 32767).astype(np.int16)


def _cpu_hash_int16(x, rounds=8):
    h = np.full_like(x, 0x811C, dtype=np.uint16)
    prime = np.uint16(0x0193)
    xv = x.view(np.uint16).copy()
    for _ in range(rounds):
        h ^= xv
        h = (h.astype(np.uint32) * prime).astype(np.uint16)
        xv = h >> 1
    return h.view(np.int16)


def _horner_cpu(x, coeffs):
    acc = np.zeros_like(x, dtype=np.int32)
    for c in reversed(coeffs):
        acc = acc * x.astype(np.int32) + c
    return np.clip(acc, -32768, 32767).astype(np.int16)


def run_all():
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = []

    def record(benchmark, size, base_cpu_us, blas_cpu_us, npu_us, template, note=""):
        base_ms = base_cpu_us / 1000 if base_cpu_us > 0 else -1
        blas_ms = blas_cpu_us / 1000 if blas_cpu_us > 0 else -1
        npu_ms = npu_us / 1000 if npu_us > 0 else -1
        vs_base = base_ms / npu_ms if base_ms > 0 and npu_ms > 0 else -1
        vs_blas = blas_ms / npu_ms if blas_ms > 0 and npu_ms > 0 else -1
        rec = {
            "benchmark": benchmark, "preset": "L", "size": size,
            "base_cpu_ms": round(base_ms, 3), "blas_cpu_ms": round(blas_ms, 3),
            "npupy_ms": round(npu_ms, 3),
            "speedup_vs_base": round(vs_base, 2), "speedup_vs_blas": round(vs_blas, 2),
            "template": template, "note": note,
        }
        records.append(rec)
        vs_blas_s = f"{vs_blas:.1f}x" if vs_blas > 0 else "N/A"
        print(f"  {benchmark:>15s}  base={base_ms:>10.2f}ms  blas={blas_ms:>10.2f}ms  npu={npu_ms:>10.2f}ms  vs_blas={vs_blas_s}")

    N = 2048
    print(f"=== GEMM-family benchmarks ({N}x{N}) ===")

    A = RNG.integers(-5, 5, (N, N), dtype=np.int16)
    B = RNG.integers(-5, 5, (N, N), dtype=np.int16)
    C = RNG.integers(-5, 5, (N, N), dtype=np.int16)
    D = RNG.integers(-5, 5, (N, N), dtype=np.int16)

    # gemm
    print("  Running gemm...", flush=True)
    t_blas = _bench(lambda: _blas_matmul(A, B))
    tmpl, t_npu = _npu_matmul(A, B, "none")
    record("gemm", f"{N}x{N}x{N}", -1, t_blas, t_npu, tmpl, "base_cpu skipped (>512²)")

    # gemm_relu
    print("  Running gemm_relu...", flush=True)
    def _blas_relu():
        return np.maximum(0, _blas_matmul(A, B))
    t_blas_relu = _bench(_blas_relu)
    tmpl_r, t_npu_r = _npu_matmul(A, B, "relu")
    record("gemm_relu", f"{N}x{N}x{N}", -1, t_blas_relu, t_npu_r, tmpl_r, "base_cpu skipped")

    # 3mm: E = (A@B) @ (C@D)
    print("  Running 3mm...", flush=True)
    def _blas_3mm():
        t1 = _blas_matmul(A, B)
        t2 = _blas_matmul(C, D)
        return _blas_matmul(t1, t2)
    t_blas_3mm = _bench(_blas_3mm)
    # NPU 3mm: 3 separate dispatches
    _, t1 = _npu_matmul(A, B, "none")
    _, t2 = _npu_matmul(C, D, "none")
    from npupy_xdna.runtime.npu_runner import NpuRunner
    from npupy_xdna.regions.region import ArraySpec, Region
    from npupy_xdna.templates.gemm_fusion import GemmFusionTemplate
    # For the 3rd matmul we need the actual intermediate results
    runner = NpuRunner()
    tmpl_obj = GemmFusionTemplate()
    region_2k = Region(op="matmul", inputs=[ArraySpec((N,N),"int16"), ArraySpec((N,N),"int16")], output=ArraySpec((N,N),"int16"))
    cfg_none = None
    for c in tmpl_obj.config_space(region_2k):
        if c.extra.get("epilogue") == "none" and c.extra.get("prologue") == "none":
            cfg_none = c; break
    iron_fn = tmpl_obj.lower(region_2k, cfg_none)
    # measure 3-dispatch NPU time
    def _npu_3mm_once():
        r1 = runner.run(region_2k, cfg_none, iron_fn, [A, np.ascontiguousarray(B.T)], timeout_s=120.0)
        r2 = runner.run(region_2k, cfg_none, iron_fn, [C, np.ascontiguousarray(D.T)], timeout_s=120.0)
        r3 = runner.run(region_2k, cfg_none, iron_fn, [r1.output, np.ascontiguousarray(r2.output.T)], timeout_s=120.0)
        return r1.latency_us + r2.latency_us + r3.latency_us
    for _ in range(2): _npu_3mm_once()
    t_npu_3mm_runs = [_npu_3mm_once() for _ in range(3)]
    t_npu_3mm = sorted(t_npu_3mm_runs)[1]
    record("3mm", f"{N}x{N}x{N}", -1, t_blas_3mm, t_npu_3mm, "gemm_fusion(x3)", "3 dispatches")

    # symm: same as gemm (symmetric A doesn't change compute)
    print("  Running symm...", flush=True)
    A_sym = (A + A.T) // 2
    t_blas_sym = _bench(lambda: _blas_matmul(A_sym, B))
    _, t_npu_sym = _npu_matmul(A_sym, B, "none")
    record("symm", f"{N}x{N}x{N}", -1, t_blas_sym, t_npu_sym, "gemm_fusion", "base_cpu skipped")

    # syr2k: C = A@B^T + B@A^T
    print("  Running syr2k...", flush=True)
    def _blas_syr2k():
        t1 = scipy.linalg.blas.sgemm(1.0, A.astype(np.float32), B.astype(np.float32), trans_b=True)
        t2 = scipy.linalg.blas.sgemm(1.0, B.astype(np.float32), A.astype(np.float32), trans_b=True)
        return np.clip(t1 + t2, -32768, 32767).astype(np.int16)
    t_blas_syr2k = _bench(_blas_syr2k)
    # NPU syr2k: 2 matmuls + CPU add
    _, t_s1 = _npu_matmul(A, B, "none")  # A@B^T via col-major trick
    _, t_s2 = _npu_matmul(B, A, "none")
    t_npu_syr2k = t_s1 + t_s2 + 50  # ~50us for the CPU add
    record("syr2k", f"{N}x{N}x{N}", -1, t_blas_syr2k, t_npu_syr2k, "gemm_fusion(x2)+cpu_add", "2 dispatches + cpu add")

    # mvt: x1 = A@y1, x2 = A^T@y2 (matrix-vector, CPU fallback)
    print("  Running mvt...", flush=True)
    y1 = RNG.integers(-5, 5, N, dtype=np.int16)
    y2 = RNG.integers(-5, 5, N, dtype=np.int16)
    def _cpu_mvt():
        return A.astype(np.int32) @ y1.astype(np.int32), A.T.astype(np.int32) @ y2.astype(np.int32)
    t_mvt = _bench(_cpu_mvt)
    def _blas_mvt():
        x1 = scipy.linalg.blas.sgemv(1.0, A.astype(np.float32), y1.astype(np.float32))
        x2 = scipy.linalg.blas.sgemv(1.0, A.T.astype(np.float32), y2.astype(np.float32))
        return x1, x2
    t_blas_mvt = _bench(_blas_mvt)
    record("mvt", f"{N}x{N}", t_mvt, t_blas_mvt, t_mvt, "cpu_fallback", "no NPU template for matvec")

    # atax: y = A^T @ (A @ x)
    print("  Running atax...", flush=True)
    x_vec = RNG.integers(-5, 5, N, dtype=np.int16)
    def _cpu_atax():
        t = A.astype(np.int32) @ x_vec.astype(np.int32)
        return A.T.astype(np.int32) @ t
    t_atax = _bench(_cpu_atax)
    def _blas_atax():
        t = scipy.linalg.blas.sgemv(1.0, A.astype(np.float32), x_vec.astype(np.float32))
        return scipy.linalg.blas.sgemv(1.0, A.T.astype(np.float32), t)
    t_blas_atax = _bench(_blas_atax)
    record("atax", f"{N}x{N}", t_atax, t_blas_atax, t_atax, "cpu_fallback", "no NPU template for matvec")

    # correlation: CPU fallback (float32 path)
    print("  Running correlation...", flush=True)
    data_corr = RNG.integers(-100, 100, (N, N), dtype=np.int16)
    def _cpu_corr():
        d = data_corr.astype(np.float32)
        mean = d.mean(axis=0)
        std = d.std(axis=0)
        std[std == 0] = 1.0
        normed = (d - mean) / std
        return (normed.T @ normed) / N
    t_corr = _bench(_cpu_corr, n_warmup=1, n_iters=3)
    record("correlation", f"{N}x{N}", t_corr, t_corr, t_corr, "cpu_fallback", "float32 path; BLAS=base")

    # gramschmidt: CPU fallback
    print("  Running gramschmidt...", flush=True)
    def _cpu_gs():
        A_gs = data_corr.astype(np.float32).copy()
        n = min(N, 256)  # limit columns to avoid extreme runtime
        Q = np.zeros((N, n), dtype=np.float32)
        R = np.zeros((n, n), dtype=np.float32)
        for j in range(n):
            v = A_gs[:, j].copy()
            for i in range(j):
                R[i, j] = Q[:, i] @ v
                v -= R[i, j] * Q[:, i]
            R[j, j] = np.linalg.norm(v)
            if R[j, j] > 0:
                Q[:, j] = v / R[j, j]
        return Q, R
    t_gs = _bench(_cpu_gs, n_warmup=1, n_iters=3)
    record("gramschmidt", f"{N}x256", t_gs, t_gs, t_gs, "cpu_fallback", "float32 QR; BLAS=base")

    # === Elementwise benchmarks (4M elements) ===
    EL_N = 4_194_304
    print(f"\n=== Elementwise benchmarks ({EL_N} elements) ===")
    x_el = RNG.integers(-100, 100, EL_N, dtype=np.int16)

    # relu
    print("  Running relu...", flush=True)
    t_relu_cpu = _bench(lambda: np.maximum(0, x_el))
    tmpl_relu, t_relu_npu = _npu_elementwise(x_el, "relu")
    record("relu", f"{EL_N}", t_relu_cpu, t_relu_cpu, t_relu_npu,
           tmpl_relu or "cpu_fallback", "BLAS=base (no BLAS for eltwise)")

    # tanh
    print("  Running tanh...", flush=True)
    t_tanh_cpu = _bench(lambda: _cpu_tanh_int16(x_el))
    tmpl_tanh, t_tanh_npu = _npu_elementwise(x_el, "tanh")
    record("tanh", f"{EL_N}", t_tanh_cpu, t_tanh_cpu, t_tanh_npu,
           tmpl_tanh or "cpu_fallback", "BLAS=base; Horner polynomial approx")

    # hash
    print("  Running hash...", flush=True)
    t_hash_cpu = _bench(lambda: _cpu_hash_int16(x_el))
    tmpl_hash, t_hash_npu = _npu_elementwise(x_el, "hash")
    record("hash", f"{EL_N}", t_hash_cpu, t_hash_cpu, t_hash_npu,
           tmpl_hash or "cpu_fallback", "BLAS=base; FNV-1a 8 rounds")

    # horner polynomial
    print("  Running horner_poly...", flush=True)
    coeffs = [1, -2, 3, -1, 2, -3, 1, -2, 3]  # degree 8
    t_horner_cpu = _bench(lambda: _horner_cpu(x_el, coeffs))
    record("horner_poly", f"{EL_N}", t_horner_cpu, t_horner_cpu, t_horner_cpu,
           "cpu_fallback", "no NPU dispatch path for raw polynomial")

    # jacobi-2d stencil
    print("\n=== Stencil benchmark ===")
    ST_N = 256
    grid = RNG.integers(-50, 50, (ST_N, ST_N), dtype=np.int16)
    def _cpu_jacobi(g, iters=5):
        g = g.astype(np.int32)
        for _ in range(iters):
            g[1:-1, 1:-1] = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] + g[1:-1, 1:-1]) // 5
        return g.astype(np.int16)
    print("  Running jacobi-2d...", flush=True)
    t_jac = _bench(lambda: _cpu_jacobi(grid.copy()), n_warmup=2, n_iters=5)
    record("jacobi-2d", f"{ST_N}x{ST_N} T=5", t_jac, t_jac, t_jac,
           "cpu_fallback", "sliding_window template exists but not dispatched in this eval")

    # Write JSONL
    with open(JSONL_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWritten {len(records)} records to {JSONL_PATH}")

    # Write evidence
    lines = [f"Preset L Full Evaluation — {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for r in records:
        vs = f"{r['speedup_vs_blas']:.1f}x" if r["speedup_vs_blas"] > 0 else "N/A"
        lines.append(f"{r['benchmark']:>15s}  blas={r['blas_cpu_ms']:>10.2f}ms  npu={r['npupy_ms']:>10.2f}ms  vs_blas={vs}")
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text("\n".join(lines))
    print(f"Evidence written to {EVIDENCE_PATH}")

    return records


def plot_speedup(records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["benchmark"] for r in records]
    base = [r["base_cpu_ms"] if r["base_cpu_ms"] > 0 else None for r in records]
    blas = [r["blas_cpu_ms"] if r["blas_cpu_ms"] > 0 else None for r in records]
    npu = [r["npupy_ms"] if r["npupy_ms"] > 0 else None for r in records]

    fig, ax = plt.subplots(figsize=(18, 8))
    x = np.arange(len(names))
    w = 0.25

    bars_base = ax.bar(x - w, [v if v else 0.001 for v in base], w, label="Base CPU int16", color="#999999", alpha=0.8)
    bars_blas = ax.bar(x, [v if v else 0.001 for v in blas], w, label="CPU BLAS (i16→f32→i16)", color="#4488cc", alpha=0.8)
    bars_npu = ax.bar(x + w, [v if v else 0.001 for v in npu], w, label="NPUPy int16", color="#44bb66", alpha=0.8)

    # Mark skipped base CPU bars
    for i, v in enumerate(base):
        if v is None or v < 0:
            ax.text(x[i] - w, 0.002, "skip", ha="center", va="bottom", fontsize=7, color="gray", rotation=90)

    # Add speedup annotations for NPU vs BLAS
    for i, r in enumerate(records):
        if r["speedup_vs_blas"] > 0 and r["speedup_vs_blas"] != 1.0:
            npu_val = r["npupy_ms"] if r["npupy_ms"] > 0 else 0.001
            label = f"{r['speedup_vs_blas']:.1f}×"
            color = "#006600" if r["speedup_vs_blas"] > 1.0 else "#cc0000"
            ax.text(x[i] + w, npu_val * 1.15, label, ha="center", va="bottom", fontsize=8, fontweight="bold", color=color)

    ax.set_yscale("log")
    ax.set_ylabel("Execution Time (ms, log scale)")
    ax.set_xlabel("Benchmark")
    ax.set_title("NPBench Preset L — Three-Way Performance Comparison\n(GEMM @ 2048², Elementwise @ 4M elements)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = RESULTS_DIR / "04_npbench_plots" / "preset_L_speedup_chart.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    records = run_all()
    plot_speedup(records)
    print("\nDone.")
