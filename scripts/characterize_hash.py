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

from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES

SIZES = SUPPORTED_SHAPES["hash"]
JSONL_PATH = Path(__file__).parent.parent / "results" / "timings" / "hash.jsonl"
N_WARMUP = 3
N_ITERS = 5


def cpu_hash_ref(x: np.ndarray) -> np.ndarray:
    flat = x.ravel().view(np.uint16).copy()
    h = np.full(flat.shape, np.uint16(0x811C), dtype=np.uint16)
    xv = flat.copy()
    for _ in range(8):
        h ^= xv
        h = (h.astype(np.uint32) * np.uint32(0x0193)).astype(np.uint16)
        xv = (h >> np.uint16(1)).astype(np.uint16)
    return h.view(np.int16).reshape(x.shape)


def benchmark_cpu(inp: np.ndarray, n_warmup: int, n_iters: int) -> float:
    for _ in range(n_warmup):
        cpu_hash_ref(inp)
    latencies = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        cpu_hash_ref(inp)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)
    return statistics.median(latencies)


def benchmark_size(n: int) -> dict:
    rng = np.random.default_rng(42)
    inp = rng.integers(-32768, 32767, size=(n,), dtype=np.int16)

    region = Region(
        op="elementwise_unary",
        inputs=[ArraySpec((n,), "int16")],
        output=ArraySpec((n,), "int16"),
        metadata={"compute_fn": "hash", "compute_intensity": "high"},
    )
    tmpl = ColIndependentTemplate()
    cfg = tmpl.config_space(region)[0]

    print(f"  [N={n:>8d}] Lowering / compiling...", flush=True)
    iron_fn = tmpl.lower(region, cfg)
    runner = NpuRunner()

    print(f"  [N={n:>8d}] Correctness check...", flush=True)
    result = runner.run(region, cfg, iron_fn, [inp])
    assert result.status == "ok", f"NPU failed: {result.status}"
    expected = cpu_hash_ref(inp)
    mismatches = np.where(result.output != expected)[0]
    assert len(mismatches) == 0, (
        f"Hash mismatch at N={n}: {len(mismatches)} mismatches, "
        f"first at idx={mismatches[0]} npu={result.output[mismatches[0]]} "
        f"expected={expected[mismatches[0]]}"
    )
    print(f"  [N={n:>8d}] Correctness OK", flush=True)

    print(f"  [N={n:>8d}] NPU warmup ({N_WARMUP})...", flush=True)
    for _ in range(N_WARMUP):
        runner.run(region, cfg, iron_fn, [inp])

    print(f"  [N={n:>8d}] NPU measuring ({N_ITERS})...", flush=True)
    npu_latencies = []
    for _ in range(N_ITERS):
        r = runner.run(region, cfg, iron_fn, [inp])
        assert r.status == "ok"
        npu_latencies.append(r.latency_us)

    npu_median = statistics.median(npu_latencies)
    npu_min = min(npu_latencies)
    npu_max = max(npu_latencies)

    print(f"  [N={n:>8d}] CPU baseline...", flush=True)
    cpu_median = benchmark_cpu(inp, N_WARMUP, N_ITERS)

    ops_per_element = 3 * 8
    gops = (n * ops_per_element) / (npu_median * 1e-6) / 1e9
    speedup = cpu_median / npu_median

    return {
        "kernel": "hash_fnv1a",
        "rounds": 8,
        "size": n,
        "npu_median_us": round(npu_median, 3),
        "npu_min_us": round(npu_min, 3),
        "npu_max_us": round(npu_max, 3),
        "cpu_median_us": round(cpu_median, 3),
        "gops": round(gops, 4),
        "speedup": round(speedup, 4),
    }


def main() -> None:
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSONL_PATH.write_text("")

    print("=" * 60)
    print("FNV-1a Hash Characterization (Col-Indep template)")
    print(f"Sizes: {SIZES}")
    print(f"Rounds: 8  (~24 ops/element = 12 ops/byte)")
    print(f"Output: {JSONL_PATH}")
    print("=" * 60)

    rows: list[dict] = []
    for n in SIZES:
        print(f"\nBenchmarking N={n}...", flush=True)
        row = benchmark_size(n)
        rows.append(row)
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"  -> npu={row['npu_median_us']:.1f}us  "
            f"cpu={row['cpu_median_us']:.1f}us  "
            f"gops={row['gops']:.2f}  "
            f"speedup={row['speedup']:.2f}x"
        )

    print(f"\nDone. {len(rows)} rows written to {JSONL_PATH}")
    print(f"\n{'Size':>10}  {'NPU(us)':>10}  {'CPU(us)':>10}  {'GOPS':>8}  {'Speedup':>8}")
    print("-" * 55)
    for r in rows:
        print(
            f"{r['size']:>10d}  {r['npu_median_us']:>10.1f}  "
            f"{r['cpu_median_us']:>10.1f}  {r['gops']:>8.2f}  "
            f"{r['speedup']:>8.2f}x"
        )


if __name__ == "__main__":
    main()
