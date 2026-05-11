#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.templates.col_independent import ColIndependentTemplate
from npupy_xdna.runtime.npu_runner import NpuRunner
from npupy_xdna.runtime.cpu_runner import CpuRunner
from npupy_xdna.bench.timer import BenchmarkConfig, run_benchmark

NEW_SIZES = [2097152, 4194304]
JSONL_PATH = Path(__file__).parent.parent / "results" / "timings" / "col_indep.jsonl"
N_WARMUP = 5
N_ITERS = 10


def benchmark_size(n: int) -> dict:
    rng = np.random.default_rng(42)
    inp = rng.integers(-100, 100, size=(n,), dtype=np.int16)

    region = Region(
        op="elementwise_unary",
        inputs=[ArraySpec((n,), "int16")],
        output=ArraySpec((n,), "int16"),
    )
    tmpl = ColIndependentTemplate()
    cfg = tmpl.config_space(region)[0]

    print(f"  [N={n:>8d}] Compiling / lowering...", flush=True)
    iron_fn = tmpl.lower(region, cfg)
    runner = NpuRunner()

    print(f"  [N={n:>8d}] Correctness check (bit-exact)...", flush=True)
    result = runner.run(region, cfg, iron_fn, [inp])
    assert result.status == "ok", f"NPU run failed: {result.status}"
    expected = np.maximum(0, inp)
    assert np.array_equal(result.output, expected), (
        f"Correctness FAILED for N={n}: "
        f"first mismatch at {np.where(result.output != expected)[0][0]}"
    )
    print(f"  [N={n:>8d}] NPU warmup ({N_WARMUP})...", flush=True)
    for _ in range(N_WARMUP):
        result = runner.run(region, cfg, iron_fn, [inp])
        assert result.status == "ok", f"warmup failed: {result.status}"

    print(f"  [N={n:>8d}] NPU measuring ({N_ITERS})...", flush=True)
    npu_latencies: list[float] = []
    for _ in range(N_ITERS):
        result = runner.run(region, cfg, iron_fn, [inp])
        assert result.status == "ok", f"NPU run failed: {result.status}"
        npu_latencies.append(result.latency_us)

    npu_median = statistics.median(npu_latencies)
    npu_min = min(npu_latencies)
    npu_max = max(npu_latencies)

    print(f"  [N={n:>8d}] CPU baseline...", flush=True)
    cpu_runner = CpuRunner()
    cpu_cfg = BenchmarkConfig(n_warmup=N_WARMUP, n_iterations=N_ITERS, seed=42)
    cpu_result = run_benchmark(cpu_runner.run, region, [inp], config=cpu_cfg)
    cpu_median = cpu_result.median_us

    # bandwidth: read + write = 2 * n * sizeof(int16) = 2 * n * 2 bytes
    bandwidth_gbps = (2 * n * 2) / (npu_median * 1e-6) / 1e9
    speedup = cpu_median / npu_median

    return {
        "size": n,
        "npu_median_us": round(npu_median, 3),
        "npu_min_us": round(npu_min, 3),
        "npu_max_us": round(npu_max, 3),
        "cpu_median_us": round(cpu_median, 3),
        "bandwidth_gbps": round(bandwidth_gbps, 4),
        "speedup": round(speedup, 4),
    }


def main() -> None:
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ColIndependentTemplate Extension Sweep (2M + 4M)")
    print(f"New sizes: {NEW_SIZES}")
    print(f"Output (append): {JSONL_PATH}")
    print("=" * 60)

    rows: list[dict] = []
    for n in NEW_SIZES:
        print(f"\nBenchmarking N={n}...")
        row = benchmark_size(n)
        rows.append(row)
        # APPEND — never overwrite
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"  -> npu_median={row['npu_median_us']:.1f}us  "
            f"cpu_median={row['cpu_median_us']:.1f}us  "
            f"bw={row['bandwidth_gbps']:.2f}GB/s  "
            f"speedup={row['speedup']:.2f}x"
        )

    print(f"\nDone. {len(rows)} new rows appended to {JSONL_PATH}")
    print("\nNew rows summary:")
    print(f"{'Size':>10}  {'NPU(us)':>10}  {'CPU(us)':>10}  {'BW(GB/s)':>10}  {'Speedup':>8}")
    print("-" * 56)
    for r in rows:
        print(
            f"{r['size']:>10d}  {r['npu_median_us']:>10.1f}  "
            f"{r['cpu_median_us']:>10.1f}  {r['bandwidth_gbps']:>10.2f}  "
            f"{r['speedup']:>8.2f}x"
        )


if __name__ == "__main__":
    main()
