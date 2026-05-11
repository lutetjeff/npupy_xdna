#!/usr/bin/env python3
"""GEMM Fusion tile-size sweep at 2048³ shape, none epilogue.

Usage:
  source /opt/xilinx/xrt/setup.sh && \
  source ~/mlir-aie/ironenv/bin/activate && \
  source ~/mlir-aie/utils/env_setup.sh && \
  python -m npupy_xdna.scripts.sweep_gemm_tiles

Outputs:
  results/timings/gemm_tile_sweep.jsonl
"""

from __future__ import annotations

import datetime
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.runtime.npu_runner import NpuRunner
from npupy_xdna.templates.gemm_fusion import GemmFusionTemplate

_PKG = Path(__file__).resolve().parent.parent
TIMINGS_FILE = _PKG / "results" / "timings" / "gemm_tile_sweep.jsonl"
EVIDENCE_FILE = Path("/home/lutet/ece511/.sisyphus/evidence/task-v2-3-tile-sweep.txt")

M = K = N = 2048
N_WARMUP = 3
N_ITERS = 5
RNG_SEED = 42
EPILOGUE = "none"
PROLOGUE = "none"


def _derive_gops(median_us: float) -> float:
    return 2.0 * M * K * N / (median_us * 1000.0)


def _run_tile(
    tile: tuple[int, int, int],
    npu_runner: NpuRunner,
    A: np.ndarray,
    B_col: np.ndarray,
) -> "dict[str, float | int | str | list[object]]":
    m, k, n = tile
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] tile={tile}  shape={M}³  epilogue={EPILOGUE!r}", flush=True)

    region = Region(
        op="matmul",
        inputs=[
            ArraySpec(shape=(M, K), dtype="int16"),
            ArraySpec(shape=(K, N), dtype="int16"),
        ],
        output=ArraySpec(shape=(M, N), dtype="int16"),
    )

    tmpl = GemmFusionTemplate()
    matching = [
        c for c in tmpl.config_space(region)
        if c.tile == tile
        and c.extra.get("epilogue") == EPILOGUE
        and c.extra.get("prologue") == PROLOGUE
    ]
    if not matching:
        raise ValueError(f"No config found for tile={tile}")
    config = matching[0]

    print(f"  Lowering (triggers xclbin compile)…", flush=True)
    iron_fn = tmpl.lower(region, config)

    def _call():
        return npu_runner.run(region, config, iron_fn, [A, B_col], timeout_s=300.0)

    print(f"  Warmup ({N_WARMUP} iters)…", flush=True)
    for i in range(N_WARMUP):
        result = _call()
        print(f"    warmup {i+1}/{N_WARMUP}: {result.latency_us:.0f} µs", flush=True)
        if result.status != "ok":
            raise RuntimeError(f"Warmup failed: {result.status}")

    print(f"  Measuring ({N_ITERS} iters)…", flush=True)
    latencies: list[float] = []
    last_output = None
    for i in range(N_ITERS):
        result = _call()
        if result.status != "ok":
            raise RuntimeError(f"Measurement failed: {result.status}")
        latencies.append(result.latency_us)
        last_output = result.output
        print(f"    iter {i+1}/{N_ITERS}: {result.latency_us:.0f} µs", flush=True)

    median_us = statistics.median(latencies)
    min_us = min(latencies)
    max_us = max(latencies)
    gops = _derive_gops(median_us)

    print(f"  median={median_us:.0f} µs  GOPS={gops:.1f}", flush=True)

    print(f"  Verifying correctness vs numpy…", flush=True)
    B_row = np.ascontiguousarray(B_col.T)
    raw = A.astype(np.int32) @ B_row.astype(np.int32)
    ref_i16 = np.clip(raw, -32768, 32767).astype(np.int16)
    if last_output is None:
        raise RuntimeError("No output from NPU")
    if not np.array_equal(last_output, ref_i16):
        n_wrong = int(np.sum(last_output != ref_i16))
        raise RuntimeError(f"Correctness FAILED for tile={tile}: {n_wrong} mismatches")
    print(f"  Correctness: PASS (bit-exact vs numpy)", flush=True)

    return {
        "shape": [M, K, N],
        "tile": list(tile),
        "epilogue": EPILOGUE,
        "prologue": PROLOGUE,
        "n_cores": config.n_cores,
        "npu_median_us": round(median_us, 2),
        "npu_min_us": round(min_us, 2),
        "npu_max_us": round(max_us, 2),
        "derived_gops": round(gops, 1),
        "latencies_us": [round(l, 2) for l in latencies],
        "correctness": "pass",
        "seed": RNG_SEED,
        "timestamp": ts,
    }


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    A = rng.integers(-10, 10, size=(M, K), dtype=np.int16)
    B = rng.integers(-10, 10, size=(K, N), dtype=np.int16)
    B_col = np.ascontiguousarray(B.T)

    npu_runner = NpuRunner()
    TIMINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TIMINGS_FILE.write_text("")

    records: list[dict[str, float | int | str | list[object]]] = []
    errors: list[str] = []

    for tile in GemmFusionTemplate.TILE_SIZES:
        try:
            rec = _run_tile(tile, npu_runner, A, B_col)
            records.append(rec)
            with TIMINGS_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as exc:
            msg = f"tile={tile} FAILED: {exc}"
            print(f"  ERROR: {msg}", flush=True)
            errors.append(msg)
            err_rec = {
                "shape": [M, K, N],
                "tile": list(tile),
                "epilogue": EPILOGUE,
                "error": str(exc),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            with TIMINGS_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(err_rec) + "\n")

    if not records:
        print("ERROR: No successful tile configs!", file=sys.stderr)
        sys.exit(1)

    best = max(records, key=lambda r: r["derived_gops"])
    print(f"\n=== Best tile: {best['tile']}  GOPS={best['derived_gops']:.1f} ===", flush=True)

    gops_lines = "\n".join(
        f"  tile={r['tile']}  GOPS={r['derived_gops']:.1f}  median={r['npu_median_us']:.0f}µs  correct={r['correctness']}"
        for r in records
    )
    if errors:
        gops_lines += "\n" + "\n".join(f"  ERROR: {e}" for e in errors)

    evidence = (
        f"Task: V2 Task 3 — GEMM Fusion tile-size sweep at {M}³\n"
        f"Date: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        f"Shape: {M}x{K}x{N}  epilogue={EPILOGUE}  prologue={PROLOGUE}\n"
        f"Warmup: {N_WARMUP}  Measured: {N_ITERS}\n"
        f"Seed: {RNG_SEED}\n"
        f"\nResults ({len(records)}/{len(GemmFusionTemplate.TILE_SIZES)} configs succeeded):\n"
        f"{gops_lines}\n"
        f"\nBest tile: {best['tile']}  GOPS={best['derived_gops']:.1f}  median={best['npu_median_us']:.0f}µs\n"
        f"V1 baseline (64,64,64): 5159 GOPS\n"
        f"\nJSONL: {TIMINGS_FILE}\n"
    )

    EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_FILE.write_text(evidence)
    print(f"\nEvidence written: {EVIDENCE_FILE}")
    print(f"JSONL written:    {TIMINGS_FILE}")


if __name__ == "__main__":
    main()
