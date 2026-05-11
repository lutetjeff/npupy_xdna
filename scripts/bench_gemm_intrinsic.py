#!/usr/bin/env python3
"""GEMM Fusion MMUL intrinsic comparison at 2048³.

Measures the 4×4×8 intrinsic (AIE2P native i16 shape) and documents comparison
with the AIE2 (non-P) 4×4×4 baseline path.

Usage:
  source /opt/xilinx/xrt/setup.sh && \
  source ~/mlir-aie/ironenv/bin/activate && \
  source ~/mlir-aie/utils/env_setup.sh && \
  python -m npupy_xdna.scripts.bench_gemm_intrinsic

Outputs:
  results/timings/gemm_intrinsic.jsonl
  .sisyphus/evidence/task-v2-7-mmul-intrinsic.txt
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
TIMINGS_FILE = _PKG / "results" / "timings" / "gemm_intrinsic.jsonl"
EVIDENCE_FILE = Path("/home/lutet/ece511/.sisyphus/evidence/task-v2-7-mmul-intrinsic.txt")

M = K = N = 2048
TILE = (64, 64, 64)
N_WARMUP = 3
N_ITERS = 5
RNG_SEED = 42
EPILOGUE = "none"
PROLOGUE = "none"
MMUL_VARIANT = "4x4x8"


def _gops(median_us: float) -> float:
    return 2.0 * M * K * N / (median_us * 1000.0)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    A = rng.integers(-10, 10, size=(M, K), dtype=np.int16)
    B = rng.integers(-10, 10, size=(K, N), dtype=np.int16)
    B_col = np.ascontiguousarray(B.T)

    npu_runner = NpuRunner()
    TIMINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)

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
        if c.tile == TILE
        and c.extra.get("epilogue") == EPILOGUE
        and c.extra.get("prologue") == PROLOGUE
        and c.extra.get("mmul_variant") == MMUL_VARIANT
    ]
    if not matching:
        print(f"ERROR: no config for tile={TILE} mmul_variant={MMUL_VARIANT}", file=sys.stderr)
        sys.exit(1)
    config = matching[0]

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] shape={M}³  tile={TILE}  mmul_variant={MMUL_VARIANT}", flush=True)

    print("  Lowering (xclbin compile if not cached)…", flush=True)
    iron_fn = tmpl.lower(region, config)

    def _call():
        return npu_runner.run(region, config, iron_fn, [A, B_col], timeout_s=300.0)

    print(f"  Warmup ({N_WARMUP} iters)…", flush=True)
    for i in range(N_WARMUP):
        result = _call()
        print(f"    warmup {i+1}/{N_WARMUP}: {result.latency_us:.0f} µs", flush=True)
        if result.status != "ok":
            raise RuntimeError(f"Warmup {i+1} failed: {result.status}")

    print(f"  Measuring ({N_ITERS} iters)…", flush=True)
    latencies: list[float] = []
    last_output = None
    for i in range(N_ITERS):
        result = _call()
        if result.status != "ok":
            raise RuntimeError(f"Iteration {i+1} failed: {result.status}")
        latencies.append(result.latency_us)
        last_output = result.output
        print(f"    iter {i+1}/{N_ITERS}: {result.latency_us:.0f} µs", flush=True)

    median_us = statistics.median(latencies)
    min_us = min(latencies)
    max_us = max(latencies)
    gops_4x4x8 = _gops(median_us)
    print(f"  median={median_us:.0f} µs  GOPS={gops_4x4x8:.1f}", flush=True)

    print("  Verifying correctness vs numpy…", flush=True)
    B_row = np.ascontiguousarray(B_col.T)
    raw = A.astype(np.int32) @ B_row.astype(np.int32)
    ref_i16 = np.clip(raw, -32768, 32767).astype(np.int16)
    if last_output is None or not np.array_equal(last_output, ref_i16):
        n_wrong = int(np.sum(last_output != ref_i16)) if last_output is not None else -1
        raise RuntimeError(f"Correctness FAILED: {n_wrong} mismatches")
    print("  Correctness: PASS", flush=True)

    rec_4x4x8 = {
        "shape": [M, K, N],
        "tile": list(TILE),
        "mmul_variant": MMUL_VARIANT,
        "kernel_path": "aie2p/mm.cc → matmul_vectorized_4x4x8_i16_i16",
        "epilogue": EPILOGUE,
        "prologue": PROLOGUE,
        "n_cores": config.n_cores,
        "n_warmup": N_WARMUP,
        "n_iters": N_ITERS,
        "npu_median_us": round(median_us, 2),
        "npu_min_us": round(min_us, 2),
        "npu_max_us": round(max_us, 2),
        "derived_gops": round(gops_4x4x8, 1),
        "latencies_us": [round(l, 2) for l in latencies],
        "correctness": "pass",
        "seed": RNG_SEED,
        "timestamp": ts,
    }

    aie2p_t = 8  # AIE2P native int16 MMUL t-dimension (N-cols per cycle)
    aie2_t = 4   # AIE2 (non-P) int16 MMUL t-dimension: half the N-throughput
    gops_4x4x4_est = round(gops_4x4x8 * (aie2_t / aie2p_t), 1)
    rec_baseline = {
        "shape": [M, K, N],
        "tile": list(TILE),
        "mmul_variant": "4x4x4_aie2_estimate",
        "kernel_path": "aie2/mm.cc → matmul_vectorized_4x4x4_i16_i16 (theoretical)",
        "epilogue": EPILOGUE,
        "prologue": PROLOGUE,
        "n_cores": config.n_cores,
        "n_warmup": "N/A",
        "n_iters": "N/A",
        "npu_median_us": "N/A",
        "npu_min_us": "N/A",
        "npu_max_us": "N/A",
        "derived_gops": gops_4x4x4_est,
        "latencies_us": [],
        "correctness": "N/A",
        "seed": RNG_SEED,
        "timestamp": ts,
        "note": (
            "Theoretical estimate: AIE2P only supports 4x4x8 for int16 natively. "
            "4x4x4 (aie2/mm.cc) would process t=4 vs t=8 N-cols per MMUL cycle, "
            "estimated ~50% throughput. No native 8x2x8 i16 shape exists on AIE2P."
        ),
    }

    TIMINGS_FILE.write_text("")
    with TIMINGS_FILE.open("a") as f:
        f.write(json.dumps(rec_4x4x8) + "\n")
        f.write(json.dumps(rec_baseline) + "\n")

    print(f"\n{'='*60}", flush=True)
    print(f"MMUL Intrinsic Comparison at {M}³:", flush=True)
    print(f"  4×4×8 (AIE2P native, measured): {gops_4x4x8:.1f} GOPS", flush=True)
    print(f"  4×4×4 (AIE2 path, estimated):   {gops_4x4x4_est:.1f} GOPS", flush=True)
    print(f"  V1 reference (4×4×8, prior run): 5159.1 GOPS", flush=True)
    print(f"  Improvement vs 4×4×4 estimate:  {gops_4x4x8 / (gops_4x4x4_est or 1):.2f}×", flush=True)
    print(f"{'='*60}", flush=True)

    evidence = f"""Task: V2 Task 7 — GEMM Fusion MMUL Intrinsic Comparison
Date: {ts}
Shape: {M}x{K}x{N}  tile={TILE}  epilogue={EPILOGUE}
Warmup: {N_WARMUP}  Measured: {N_ITERS}

Kernel investigation (aie2p/mm.cc):
  i16_i16 combo: matmul_vectorized_4x4x8_i16_i16  r=4  s=4  t=8
  IRON program: r, s, t = 4, 4, 8 (matches kernel)
  Tile constraints (with 2×2 expansion): m%(2r)==0, k%s==0, n%(2t)==0 → 8, 4, 16
  At tile=(64,64,64): 64%8=0 ✓, 64%4=0 ✓, 64%16=0 ✓

4×4×8 measured (AIE2P native):
  median={median_us:.2f} µs  GOPS={gops_4x4x8:.1f}
  min={min_us:.2f} µs  max={max_us:.2f} µs
  latencies: {[round(l,1) for l in latencies]}
  correctness: PASS (bit-exact vs numpy)

Comparison:
  4×4×8 (AIE2P, measured):  {gops_4x4x8:.1f} GOPS
  4×4×4 (AIE2, estimated):  {gops_4x4x4_est:.1f} GOPS  [t ratio: 4/8 → ~50% throughput]
  V1 reference (4×4×8):     5159.1 GOPS
  Improvement vs 4×4×4:     ~{gops_4x4x8 / (gops_4x4x4_est or 1):.2f}×

Hardware finding:
  AIE2P (XDNA2 / Krackan) only supports matmul_vectorized_4x4x8_i16_i16 for int16.
  There is no native 8×2×8 int16 shape on AIE2P — the aie2p/mm.cc kernel defines
  only the 4×4×8 combination. The AIE2 (non-P) kernel uses 4×4×4 for i16, providing
  half the N-element throughput per MMUL cycle (t=4 vs t=8).
  The 4×4×8 AIE2P native path IS the optimized variant; 5159 GOPS from V1 was already
  measured using this same kernel.

config_space extension:
  GemmFusionTemplate.MMUL_VARIANTS = ["4x4x8"] added
  Config.extra["mmul_variant"] = "4x4x8" now present in all configs

JSONL: {TIMINGS_FILE}
"""
    EVIDENCE_FILE.write_text(evidence)
    print(f"\nEvidence written to: {EVIDENCE_FILE}", flush=True)
    print(f"JSONL written to:    {TIMINGS_FILE}", flush=True)


if __name__ == "__main__":
    main()
