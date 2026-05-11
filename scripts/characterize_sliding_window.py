#!/usr/bin/env python3
"""Characterization sweep for SlidingWindowTemplate.

Usage:
    source /opt/xilinx/xrt/setup.sh
    source ~/mlir-aie/ironenv/bin/activate
    source ~/mlir-aie/utils/env_setup.sh
    python npupy_xdna/scripts/characterize_sliding_window.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES
from npupy_xdna.templates.sliding_window import SlidingWindowTemplate

SHAPES = SUPPORTED_SHAPES["sliding_window"]
JSONL_PATH = Path(__file__).parent.parent.parent / "results" / "timings" / "sliding_window.jsonl"
N_WARMUP = 3
N_ITERS = 5


def cpu_stencil_5pt(inp: np.ndarray) -> np.ndarray:
    """5-point stencil reference matching stencil_5pt_int16.cc.

    out[i,j] = trunc((center+top+bot+left+right) / 5)
    Boundary rows use zero halo; boundary columns are explicitly zeroed.
    Division uses C-style truncation-toward-zero (not Python floor-division).
    """
    H, W = inp.shape
    inp32 = inp.astype(np.int32)

    padded = np.zeros((H + 2, W), dtype=np.int32)
    padded[1 : H + 1, :] = inp32

    center = padded[1 : H + 1, :]
    top    = padded[0 : H, :]
    bot    = padded[2 : H + 2, :]

    left = np.zeros((H, W), dtype=np.int32)
    left[:, 1:] = center[:, :-1]

    right = np.zeros((H, W), dtype=np.int32)
    right[:, :-1] = center[:, 1:]

    s = center + top + bot + left + right

    # np.trunc gives C-style truncation; Python // is floor-div (differs for negatives).
    v = np.trunc(s.astype(np.float64) / 5.0).astype(np.int32)
    v[:, 0]  = 0
    v[:, -1] = 0
    return np.clip(v, -32768, 32767).astype(np.int16)


def benchmark_shape(H: int, W: int) -> dict:
    rng = np.random.default_rng(42)
    inp = rng.integers(-100, 100, size=(H, W), dtype=np.int16)

    region = Region(
        op="stencil_2d",
        inputs=[ArraySpec((H, W), "int16")],
        output=ArraySpec((H, W), "int16"),
        metadata={"stencil": "5pt", "iterations": 1},
    )

    tmpl   = SlidingWindowTemplate()
    cfg    = tmpl.config_space(region)[0]

    print(f"  [{H}x{W}] Lowering (JIT compile on first call)...", flush=True)
    run_fn = tmpl.lower(region, cfg)

    npu_out = np.zeros((H, W), dtype=np.int16)

    print(f"  [{H}x{W}] NPU warmup ({N_WARMUP})...", flush=True)
    for wi in range(N_WARMUP):
        t0 = time.perf_counter()
        run_fn(inp, npu_out)
        print(f"    warmup {wi+1}/{N_WARMUP}  ({(time.perf_counter()-t0)*1e6:.0f} us)", flush=True)

    print(f"  [{H}x{W}] NPU measuring ({N_ITERS})...", flush=True)
    npu_latencies: list[float] = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        run_fn(inp, npu_out)
        t1 = time.perf_counter()
        npu_latencies.append((t1 - t0) * 1e6)

    npu_median = statistics.median(npu_latencies)
    npu_min    = min(npu_latencies)
    npu_max    = max(npu_latencies)

    print(f"  [{H}x{W}] CPU baseline...", flush=True)
    cpu_latencies: list[float] = []
    cpu_out: np.ndarray | None = None
    for i in range(N_WARMUP + N_ITERS):
        t0 = time.perf_counter()
        cpu_out = cpu_stencil_5pt(inp)
        t1 = time.perf_counter()
        if i >= N_WARMUP:
            cpu_latencies.append((t1 - t0) * 1e6)

    cpu_median = statistics.median(cpu_latencies)
    cpu_min    = min(cpu_latencies)
    cpu_max    = max(cpu_latencies)

    assert cpu_out is not None
    correctness = bool(np.array_equal(npu_out, cpu_out))
    if not correctness:
        n_diff = int(np.sum(npu_out != cpu_out))
        max_abs_err = int(np.max(np.abs(npu_out.astype(np.int32) - cpu_out.astype(np.int32))))
        print(
            f"  [{H}x{W}] WARNING: {n_diff}/{H*W} elements differ "
            f"(max |err|={max_abs_err})",
            flush=True,
        )
    else:
        print(f"  [{H}x{W}] Correctness: PASS", flush=True)

    ops    = H * W * 5
    gops   = ops / (npu_median * 1e-6) / 1e9
    speedup = cpu_median / npu_median

    return {
        "shape": [H, W],
        "npu_median_us": round(npu_median, 3),
        "npu_min_us":    round(npu_min,    3),
        "npu_max_us":    round(npu_max,    3),
        "cpu_median_us": round(cpu_median, 3),
        "cpu_min_us":    round(cpu_min,    3),
        "cpu_max_us":    round(cpu_max,    3),
        "gops":          round(gops,       4),
        "speedup":       round(speedup,    4),
        "correctness":   correctness,
        "n_warmup":      N_WARMUP,
        "n_iters":       N_ITERS,
    }


def benchmark_shape_safe(H: int, W: int) -> dict:
    try:
        return benchmark_shape(H, W)
    except Exception as exc:
        error_msg = str(exc)[:200]
        print(f"  [{H}x{W}] FAILED: {error_msg}", flush=True)
        return {
            "shape":         [H, W],
            "status":        "compile_error",
            "error":         error_msg,
            "npu_median_us": None,
            "cpu_median_us": None,
            "gops":          None,
            "speedup":       None,
            "correctness":   None,
            "n_warmup":      N_WARMUP,
            "n_iters":       N_ITERS,
        }


def main() -> None:
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSONL_PATH.write_text("")

    print("=" * 64)
    print("SlidingWindowTemplate Characterization Sweep")
    print(f"Shapes: {SHAPES}  Warmup: {N_WARMUP}  Iters: {N_ITERS}")
    print(f"Output: {JSONL_PATH}")
    print("=" * 64)

    rows: list[dict] = []
    for H, W in SHAPES:
        print(f"\nBenchmarking {H}x{W} ...", flush=True)
        row = benchmark_shape_safe(H, W)
        rows.append(row)
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if row.get("status") == "compile_error":
            print(f"  => COMPILE ERROR (recorded in JSONL)", flush=True)
        else:
            print(
                f"  => npu={row['npu_median_us']:.1f}us  "
                f"cpu={row['cpu_median_us']:.1f}us  "
                f"gops={row['gops']:.4f}  "
                f"speedup={row['speedup']:.2f}x  "
                f"correct={row['correctness']}",
                flush=True,
            )

    print(f"\nDone. {len(rows)} rows written to {JSONL_PATH}")
    hdr = f"{'Shape':>10}  {'NPU(us)':>10}  {'CPU(us)':>10}  {'GOPS':>8}  {'Speedup':>8}  {'Status':>14}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        s = f"{r['shape'][0]}x{r['shape'][1]}"
        if r.get("status") == "compile_error":
            print(f"{s:>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>8}  {'N/A':>8}  {'compile_error':>14}")
        else:
            status = "PASS" if r["correctness"] else "WRONG"
            print(
                f"{s:>10}  {r['npu_median_us']:>10.1f}  "
                f"{r['cpu_median_us']:>10.1f}  {r['gops']:>8.4f}  "
                f"{r['speedup']:>8.2f}x  {status:>14}"
            )

    print(f"\nDone. {len(rows)} rows written to {JSONL_PATH}")
    print()
    hdr = f"{'Shape':>10}  {'NPU(us)':>10}  {'CPU(us)':>10}  {'GOPS':>8}  {'Speedup':>8}  {'Correct':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
        print(
            f"{shape_str:>10}  {r['npu_median_us']:>10.1f}  "
            f"{r['cpu_median_us']:>10.1f}  {r['gops']:>8.4f}  "
            f"{r['speedup']:>8.2f}x  {str(r['correctness']):>8}"
        )


if __name__ == "__main__":
    main()
