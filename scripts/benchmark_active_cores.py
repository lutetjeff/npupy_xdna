#!/usr/bin/env python3
"""GEMM Fusion active-cores scaling benchmark.

Compiles and benchmarks GEMM Fusion at n_aie_cols = 1, 2, 4, 8
at fixed 2048^3 shape, none epilogue only.

Usage:
  source /opt/xilinx/xrt/setup.sh && \\
  source ~/mlir-aie/ironenv/bin/activate && \\
  source ~/mlir-aie/utils/env_setup.sh && \\
  python -m npupy_xdna.scripts.benchmark_active_cores

Outputs:
  /home/lutet/ece511/results/timings/gemm_active_cores.jsonl
  /home/lutet/ece511/.sisyphus/evidence/task-v2-12-active-cores.txt
"""

from __future__ import annotations

import datetime
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.runtime.npu_runner import NpuRunner
from npupy_xdna.templates.gemm_fusion import GemmFusionTemplate
from npupy_xdna.templates.protocol import Config

_N_AIE_ROWS = 4
M, K, N = 2048, 2048, 2048
TILE = (64, 64, 64)
EPILOGUE = "none"
PROLOGUE = "none"
N_WARMUP = 3
N_MEASURED = 5
RNG_SEED = 42
N_AIE_COLS_LIST = [1, 2, 4, 8]

TIMINGS_FILE = Path("/home/lutet/ece511/results/timings/gemm_active_cores.jsonl")
EVIDENCE_FILE = Path("/home/lutet/ece511/.sisyphus/evidence/task-v2-12-active-cores.txt")


def _ref_matmul_partial(A: np.ndarray, B: np.ndarray, rows: int = 4) -> np.ndarray:
    """Reference matmul (first `rows` rows) with int32 accumulation + int16 saturation."""
    raw = np.matmul(A[:rows].astype(np.int32), B.astype(np.int32))
    return np.clip(raw, -32768, 32767).astype(np.int16)


def _derive_gops(m: int, k: int, n: int, median_us: float) -> float:
    return 2.0 * m * k * n / (median_us * 1000.0)


def main() -> int:
    TIMINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TIMINGS_FILE.write_text("")

    rng = np.random.default_rng(RNG_SEED)
    A = rng.integers(-5, 5, size=(M, K), dtype=np.int16)
    B = rng.integers(-5, 5, size=(K, N), dtype=np.int16)
    B_col = np.ascontiguousarray(B.T)  # b_col_maj=True required by kernel

    region = Region(
        op="matmul",
        inputs=[
            ArraySpec(shape=(M, K), dtype="int16"),
            ArraySpec(shape=(K, N), dtype="int16"),
        ],
        output=ArraySpec(shape=(M, N), dtype="int16"),
    )

    tmpl = GemmFusionTemplate()
    npu_runner = NpuRunner()

    records: list[dict[str, Any]] = []
    evidence_lines = [
        "Task: V2 Task 12 -- GEMM Fusion active-cores scaling",
        f"Date: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"Shape: {M}x{K}x{N}  epilogue={EPILOGUE}  prologue={PROLOGUE}",
        f"Tile: {TILE}",
        f"Warmup: {N_WARMUP}  Measured: {N_MEASURED}",
        f"Seed: {RNG_SEED}",
        "",
        "Results:",
    ]

    print("Precomputing reference (first 4 rows of A@B)...", flush=True)
    ref_partial = _ref_matmul_partial(A, B, rows=4)
    print("  Reference done.", flush=True)

    for n_aie_cols in N_AIE_COLS_LIST:
        n_total_cores = n_aie_cols * _N_AIE_ROWS
        config = Config(
            tile=TILE,
            n_cores=n_total_cores,
            extra={"epilogue": EPILOGUE, "prologue": PROLOGUE},
        )
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        print(
            f"\n[{ts}] n_aie_cols={n_aie_cols}  n_cores={n_total_cores}  "
            f"tile={TILE}  epilogue={EPILOGUE!r}",
            flush=True,
        )

        iron_fn = tmpl.lower(region, config)

        print(f"  Warmup ({N_WARMUP} iters, first compiles xclbin)...", flush=True)
        last_result = npu_runner.run(region, config, iron_fn, [A, B_col], timeout_s=600.0)
        if last_result.status != "ok":
            print(f"  ERROR warmup 1: status={last_result.status}", file=sys.stderr)
            return 1
        print(f"    warmup 1: {last_result.latency_us:.0f}µs kernel  (compiled)", flush=True)
        for wi in range(1, N_WARMUP):
            t_start = time.perf_counter()
            last_result = npu_runner.run(region, config, iron_fn, [A, B_col], timeout_s=600.0)
            t_end = time.perf_counter()
            if last_result.status != "ok":
                print(f"  ERROR warmup {wi+1}: status={last_result.status}", file=sys.stderr)
                return 1
            print(
                f"    warmup {wi+1}: {last_result.latency_us:.0f}µs kernel  "
                f"({(t_end-t_start)*1e6:.0f}µs wall)",
                flush=True,
            )

        npu_partial = last_result.output[:4, :]
        correct = bool(np.array_equal(npu_partial, ref_partial))
        correct_str = "pass" if correct else "FAIL"
        if not correct:
            diff = np.abs(npu_partial.astype(np.int32) - ref_partial.astype(np.int32))
            print(
                f"  CORRECTNESS FAIL: max_diff={diff.max()}, "
                f"n_wrong={int((diff > 0).sum())}",
                file=sys.stderr,
            )
        print(f"  Correctness (first 4 rows): {correct_str}", flush=True)

        print(f"  Measuring ({N_MEASURED} iters)...", flush=True)
        wall_times_us: list[float] = []
        kernel_times_us: list[float] = []
        for mi in range(N_MEASURED):
            t_start = time.perf_counter()
            result = npu_runner.run(region, config, iron_fn, [A, B_col], timeout_s=600.0)
            t_end = time.perf_counter()
            if result.status != "ok":
                print(f"  ERROR measure {mi+1}: status={result.status}", file=sys.stderr)
                return 1
            wall_us = (t_end - t_start) * 1e6
            wall_times_us.append(wall_us)
            kernel_times_us.append(result.latency_us)
            print(
                f"    iter {mi+1}: {result.latency_us:.0f}µs kernel  {wall_us:.0f}µs wall",
                flush=True,
            )

        median_kernel_us = statistics.median(kernel_times_us)
        median_wall_us = statistics.median(wall_times_us)
        gops_kernel = _derive_gops(M, K, N, median_kernel_us)
        gops_wall = _derive_gops(M, K, N, median_wall_us)

        npu_median_us = median_kernel_us
        derived_gops = gops_kernel

        print(
            f"  --> kernel: median={median_kernel_us:.0f}µs  GOPS={gops_kernel:.1f}  correct={correct_str}",
            flush=True,
        )
        print(
            f"      wall:   median={median_wall_us:.0f}µs  GOPS={gops_wall:.1f}",
            flush=True,
        )

        record = {
            "n_aie_cols": n_aie_cols,
            "n_total_cores": n_total_cores,
            "shape": [M, K, N],
            "tile": list(TILE),
            "epilogue": EPILOGUE,
            "npu_median_us": round(npu_median_us, 2),
            "npu_min_us": round(min(kernel_times_us), 2),
            "npu_max_us": round(max(kernel_times_us), 2),
            "npu_wall_median_us": round(median_wall_us, 2),
            "derived_gops": round(derived_gops, 1),
            "derived_gops_wall": round(gops_wall, 1),
            "correct": correct_str,
            "seed": RNG_SEED,
        }
        records.append(record)
        with TIMINGS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        evidence_lines.append(
            f"  n_aie_cols={n_aie_cols}  n_cores={n_total_cores:2d}  "
            f"GOPS={derived_gops:7.1f}  median={npu_median_us:.0f}us  correct={correct_str}"
        )

    evidence_lines += ["", "Scaling vs 1-col baseline:"]
    baseline = next(r["derived_gops"] for r in records if r["n_aie_cols"] == 1)
    for r in records:
        factor = r["derived_gops"] / baseline if baseline > 0 else float("nan")
        ideal = float(r["n_aie_cols"])
        eff = factor / ideal * 100.0 if ideal > 0 else 0.0
        evidence_lines.append(
            f"  n_aie_cols={r['n_aie_cols']}: {factor:.2f}x  "
            f"(ideal {ideal:.0f}x, efficiency {eff:.0f}%)"
        )

    evidence_lines += [
        "",
        f"JSONL: {TIMINGS_FILE}",
        f"Total configs: {len(records)}",
    ]

    EVIDENCE_FILE.write_text("\n".join(evidence_lines) + "\n")
    print(f"\nEvidence written to {EVIDENCE_FILE}", flush=True)
    print(f"JSONL written to {TIMINGS_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
