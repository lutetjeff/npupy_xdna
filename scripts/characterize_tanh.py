#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.templates.col_independent import ColIndependentTemplate
from npupy_xdna.runtime.npu_runner import NpuRunner

SIZES = [65536, 262144, 1048576, 4194304]
JSONL_PATH = Path(__file__).parent.parent / "results" / "timings" / "tanh.jsonl"
N_WARMUP = 5
N_ITERS = 10


def cpu_tanh_int16(x: np.ndarray) -> np.ndarray:
    return np.tanh(x.astype(np.float32)).astype(np.int16)


def time_cpu(fn, *args, n_warmup=N_WARMUP, n_iters=N_ITERS):
    for _ in range(n_warmup):
        fn(*args)
    latencies = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn(*args)
        latencies.append((time.perf_counter() - t0) * 1e6)
    return latencies


def benchmark_size(n: int) -> dict:
    rng = np.random.default_rng(42)
    inp = rng.integers(-32768, 32767, size=(n,), dtype=np.int16)

    region = Region(
        op="elementwise_unary",
        inputs=[ArraySpec((n,), "int16")],
        output=ArraySpec((n,), "int16"),
        metadata={"compute_fn": "tanh"},
    )
    tmpl = ColIndependentTemplate()
    cfg = tmpl.config_space(region)[0]

    print(f"  [N={n:>8d}] Compiling / lowering...", flush=True)
    iron_fn = tmpl.lower(region, cfg)
    runner = NpuRunner()

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

    print(f"  [N={n:>8d}] CPU baseline (np.tanh float)...", flush=True)
    cpu_lats = time_cpu(cpu_tanh_int16, inp)
    cpu_median = statistics.median(cpu_lats)

    bytes_moved = 2 * n * 2
    bandwidth_gbps = bytes_moved / (npu_median * 1e-6) / 1e9
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
    JSONL_PATH.write_text("")

    print("=" * 60)
    print("Tanh-Int16 ColIndependent Characterization")
    print(f"Sizes: {SIZES}")
    print(f"Kernel: Pade [3/3] rational approx, 7 ops/element")
    print(f"Output: {JSONL_PATH}")
    print("=" * 60)

    rows: list[dict] = []
    for n in SIZES:
        print(f"\nBenchmarking N={n}...")
        row = benchmark_size(n)
        rows.append(row)
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"  -> npu_median={row['npu_median_us']:.1f}us  "
            f"cpu_median={row['cpu_median_us']:.1f}us  "
            f"bw={row['bandwidth_gbps']:.2f}GB/s  "
            f"speedup={row['speedup']:.2f}x"
        )

    print(f"\nDone. {len(rows)} rows written to {JSONL_PATH}")
    print("\nSummary table:")
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
