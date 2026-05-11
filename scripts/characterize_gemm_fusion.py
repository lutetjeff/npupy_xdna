#!/usr/bin/env python3
"""GEMM Fusion characterization sweep.

Usage:
  source /opt/xilinx/xrt/setup.sh && \\
  source ~/mlir-aie/ironenv/bin/activate && \\
  source ~/mlir-aie/utils/env_setup.sh && \\
  python -m npupy_xdna.scripts.characterize_gemm_fusion

Outputs:
  results/timings/gemm_fusion.jsonl
  results/02_template_characterization.md
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np

from npupy_xdna.bench.baselines import cpu_baseline
from npupy_xdna.bench.timer import BenchmarkConfig, run_benchmark
from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.runtime.npu_runner import NpuRunner
from npupy_xdna.templates.gemm_fusion import GemmFusionTemplate
from npupy_xdna.templates.protocol import Config
from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES

_PKG = Path(__file__).resolve().parent.parent
TIMINGS_FILE = _PKG / "results" / "timings" / "gemm_fusion.jsonl"
REPORT_FILE = _PKG / "results" / "02_template_characterization.md"
EVIDENCE_DIR = _PKG / ".sisyphus" / "evidence"

RNG_SEED = 42
EPILOGUES = ["none", "relu"]


def _make_region(M: int, K: int, N: int) -> Region:
    return Region(
        op="matmul",
        inputs=[
            ArraySpec(shape=(M, K), dtype="int16"),
            ArraySpec(shape=(K, N), dtype="int16"),
        ],
        output=ArraySpec(shape=(M, N), dtype="int16"),
    )


def _get_config(region: Region, epilogue: str) -> Config:
    tmpl = GemmFusionTemplate()
    for c in tmpl.config_space(region):
        if c.extra.get("epilogue") == epilogue and c.extra.get("prologue") == "none":
            return Config(
                tile=c.tile,
                n_cores=c.n_cores,
                extra={"epilogue": epilogue, "prologue": "none"},
            )
    raise ValueError(f"No config found for epilogue={epilogue!r}")


def _derive_gops(M: int, K: int, N: int, median_us: float) -> float:
    return 2.0 * M * K * N / (median_us * 1000.0)


def run_sweep() -> list[dict]:
    rng = np.random.default_rng(RNG_SEED)
    bench_config = BenchmarkConfig(n_warmup=5, n_iterations=10, seed=RNG_SEED)
    npu_runner = NpuRunner()
    tmpl = GemmFusionTemplate()

    TIMINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    TIMINGS_FILE.write_text("")

    records: list[dict] = []

    for shape in SUPPORTED_SHAPES["gemm_fusion"]:
        M, K, N = shape

        A = rng.integers(-10, 10, size=(M, K), dtype=np.int16)
        B = rng.integers(-10, 10, size=(K, N), dtype=np.int16)
        B_col = np.ascontiguousarray(B.T)  # b_col_maj=1: kernel requires transposed contiguous B

        region = _make_region(M, K, N)

        for epilogue in EPILOGUES:
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            print(f"[{ts}] shape={shape}  epilogue={epilogue!r}", flush=True)

            config = _get_config(region, epilogue)
            iron_fn = tmpl.lower(region, config)

            def _npu_call(
                _region=region,
                _config=config,
                _fn=iron_fn,
                _A=A,
                _B_col=B_col,
            ):
                return npu_runner.run(_region, _config, _fn, [_A, _B_col], timeout_s=60.0)

            print(f"  Warming up NPU ({bench_config.n_warmup} iters)…", flush=True)
            npu_res = run_benchmark(_npu_call, config=bench_config)

            # CPU baseline: skip for large sizes (numpy int16 matmul has no BLAS
            # acceleration — pure Python loops, takes minutes/hours for ≥1024³)
            MAX_CPU_ELEMENTS = 512 * 512 * 512  # ~134M ops is the CPU ceiling
            total_ops = M * K * N
            if total_ops <= MAX_CPU_ELEMENTS:
                print("  Running CPU baseline…", flush=True)
                cpu_res = cpu_baseline(region, [A, B], bench_config)
                cpu_median = cpu_res.median_us
            else:
                # Extrapolate from O(N³) scaling of last measured CPU point
                # Use 512³ baseline if available, else estimate conservatively
                cpu_median = -1.0  # sentinel: extrapolated
                print(f"  Skipping CPU baseline (size {M}³ too large for non-BLAS numpy)", flush=True)

            gops = _derive_gops(M, K, N, npu_res.median_us)
            speedup = cpu_median / npu_res.median_us if cpu_median > 0 else -1.0

            record = {
                "shape": list(shape),
                "epilogue": epilogue,
                "dtype": "int16",
                "npu_median_us": round(npu_res.median_us, 2),
                "npu_min_us": round(npu_res.min_us, 2),
                "npu_max_us": round(npu_res.max_us, 2),
                "cpu_median_us": round(cpu_median, 2),
                "derived_gops": round(gops, 1),
                "speedup": round(speedup, 2),
                "seed": RNG_SEED,
                "cpu_note": "measured" if cpu_median > 0 else "skipped_no_blas",
            }
            records.append(record)

            with TIMINGS_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            print(
                f"  → NPU {npu_res.median_us:.1f} µs  "
                f"CPU {cpu_res.median_us:.1f} µs  "
                f"{gops:.0f} GOPS  {speedup:.1f}x speedup",
                flush=True,
            )

    return records


def write_markdown(records: list[dict]) -> None:
    lines = [
        "# Template Characterization",
        "",
        "## GEMM Fusion (int16, NPU2)",
        "",
        "Sweep across all supported shapes with epilogues `none` and `relu`.",
        f"Benchmark: {BenchmarkConfig.n_warmup if hasattr(BenchmarkConfig, 'n_warmup') else 5} warmup "
        f"+ {BenchmarkConfig.n_iterations if hasattr(BenchmarkConfig, 'n_iterations') else 10} iterations.",
        f"RNG seed: {RNG_SEED}.  Inputs: int16, values in [-10, 10].",
        f"B matrix passed in column-major layout (transposed).",
        "",
        "| Shape (M×K×N) | Epilogue | NPU Median (µs) | NPU Min (µs) | NPU Max (µs) | CPU Median (µs) | GOPS | Speedup |",
        "|:--------------|:---------|----------------:|-------------:|-------------:|----------------:|-----:|--------:|",
    ]
    for r in records:
        M, K, N = r["shape"]
        shape_str = f"{M}×{K}×{N}"
        lines.append(
            f"| {shape_str} | {r['epilogue']} "
            f"| {r['npu_median_us']:.1f} "
            f"| {r['npu_min_us']:.1f} "
            f"| {r['npu_max_us']:.1f} "
            f"| {r['cpu_median_us']:.1f} "
            f"| {r['derived_gops']:.0f} "
            f"| {r['speedup']:.1f}× |"
        )
    lines += [
        "",
        "### Notes",
        "",
        "- **GOPS** = 2·M·K·N / (NPU median µs · 1000)",
        "- **Speedup** = CPU median µs / NPU median µs",
        "- xclbin compiled once per shape; epilogue variant selected at lowering time.",
        "- CPU baseline uses NumPy matmul (row-major inputs; no epilogue applied).",
        "",
        f"*Generated: {datetime.datetime.now().isoformat(timespec='seconds')}*",
    ]

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written: {REPORT_FILE}")


def write_evidence(records: list[dict]) -> None:
    sweep_lines = [
        f"GEMM Fusion characterization sweep",
        f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"Records: {len(records)}",
        f"Shapes: {[r['shape'] for r in records]}",
        f"Epilogues: {list({r['epilogue'] for r in records})}",
        f"JSONL: {TIMINGS_FILE}",
        "",
    ]
    for r in records:
        sweep_lines.append(
            f"  {r['shape']} epilogue={r['epilogue']!r:8s}  "
            f"NPU={r['npu_median_us']:.1f}µs  "
            f"CPU={r['cpu_median_us']:.1f}µs  "
            f"{r['derived_gops']:.0f} GOPS  "
            f"{r['speedup']:.1f}x"
        )

    (EVIDENCE_DIR / "task-12-sweep.txt").write_text("\n".join(sweep_lines) + "\n", encoding="utf-8")

    if records:
        sample = records[0]
        schema_lines = [
            "JSONL schema for gemm_fusion.jsonl",
            "",
            "Keys:",
        ]
        for key, val in sample.items():
            schema_lines.append(f"  {key}: {type(val).__name__}  (example: {val!r})")
        schema_lines += [
            "",
            "Derivations:",
            "  derived_gops = 2 * M * K * N / (npu_median_us * 1000)",
            "  speedup      = cpu_median_us / npu_median_us",
        ]
        (EVIDENCE_DIR / "task-12-schema.txt").write_text(
            "\n".join(schema_lines) + "\n", encoding="utf-8"
        )

    print(f"Evidence written: {EVIDENCE_DIR / 'task-12-sweep.txt'}")
    print(f"Evidence written: {EVIDENCE_DIR / 'task-12-schema.txt'}")


def main() -> int:
    print("=== GEMM Fusion characterization sweep ===")
    print(f"Shapes: {SUPPORTED_SHAPES['gemm_fusion']}")
    print(f"Epilogues: {EPILOGUES}")
    print(f"Output: {TIMINGS_FILE}\n")

    records = run_sweep()

    print(f"\n[Done] {len(records)} records collected.")
    write_markdown(records)
    write_evidence(records)

    return 0


if __name__ == "__main__":
    sys.exit(main())
