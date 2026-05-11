#!/usr/bin/env python3
"""Measure cold-compile vs warm-cache-hit latency per template.

Usage:
  source ~/mlir-aie/ironenv/bin/activate && \
  python -m npupy_xdna.scripts.measure_compile_cache

Outputs:
  results/timings/compile_cache.jsonl
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.templates.col_independent import ColIndependentTemplate
from npupy_xdna.templates.compute_pool import ComputePoolTemplate
from npupy_xdna.templates.cgra import CgraTemplate
from npupy_xdna.templates.gemm_fusion import GemmFusionTemplate
from npupy_xdna.templates.protocol import Config
from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES
from npupy_xdna.runtime.iron_jit import _cache_key, XCLBIN_CACHE_DIR

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "timings"
RESULTS_FILE = RESULTS_DIR / "compile_cache.jsonl"
EVIDENCE_DIR = Path(__file__).resolve().parent.parent / ".sisyphus" / "evidence"
EVIDENCE_FILE = EVIDENCE_DIR / "task-v2-5-cache-stats.txt"


def _clear_xclbin_cache(template_name: str, shape_tuple: tuple) -> int:
    key = _cache_key(template_name, shape_tuple)
    pattern = f"{template_name}_{key}"
    removed = 0
    if XCLBIN_CACHE_DIR.exists():
        for f in XCLBIN_CACHE_DIR.iterdir():
            if f.is_file() and f.name.startswith(pattern):
                f.unlink()
                removed += 1
    return removed


def _make_region_gemm(shape: tuple[int, int, int]) -> Region:
    M, K, N = shape
    return Region(
        op="matmul",
        inputs=[
            ArraySpec(shape=(M, K), dtype="int16"),
            ArraySpec(shape=(K, N), dtype="int16"),
        ],
        output=ArraySpec(shape=(M, N), dtype="int16"),
    )


def _make_region_elementwise(shape: int | tuple[int, ...]) -> Region:
    if isinstance(shape, int):
        shape = (shape,)
    return Region(
        op="elementwise_unary",
        inputs=[ArraySpec(shape=shape, dtype="int16")],
        output=ArraySpec(shape=shape, dtype="int16"),
    )


def _make_region_cgra(shape: int) -> Region:
    return Region(
        op="chained_elementwise",
        inputs=[
            ArraySpec(shape=(shape,), dtype="int16"),
            ArraySpec(shape=(shape,), dtype="int16"),
            ArraySpec(shape=(shape,), dtype="int16"),
            ArraySpec(shape=(shape,), dtype="int16"),
        ],
        output=ArraySpec(shape=(shape,), dtype="int16"),
    )


def measure_gemm_fusion() -> dict:
    shape = (128, 128, 128)
    region = _make_region_gemm(shape)
    tmpl = GemmFusionTemplate()
    cfg = None
    for c in tmpl.config_space(region):
        if c.extra.get("epilogue") == "none" and c.extra.get("prologue") == "none":
            cfg = c
            break
    assert cfg is not None, "No config found for gemm_fusion"

    _clear_xclbin_cache("gemm_fusion", shape)

    t0 = time.perf_counter()
    iron_fn_cold = tmpl.lower(region, cfg)
    t1 = time.perf_counter()
    cold_ms = (t1 - t0) * 1000.0

    t0 = time.perf_counter()
    iron_fn_warm = tmpl.lower(region, cfg)
    t1 = time.perf_counter()
    warm_ms = (t1 - t0) * 1000.0

    return {
        "template": "gemm_fusion",
        "shape": shape,
        "cold_ms": round(cold_ms, 3),
        "warm_ms": round(warm_ms, 3),
        "ratio": round(cold_ms / max(warm_ms, 0.001), 3),
        "cache_dir": str(XCLBIN_CACHE_DIR),
    }


def measure_col_independent() -> dict:
    shape = 16384
    region = _make_region_elementwise(shape)
    tmpl = ColIndependentTemplate()
    cfg = tmpl.config_space(region)[0]

    _clear_xclbin_cache("col_independent", (shape,))

    t0 = time.perf_counter()
    iron_fn_cold = tmpl.lower(region, cfg)
    t1 = time.perf_counter()
    cold_ms = (t1 - t0) * 1000.0

    t0 = time.perf_counter()
    iron_fn_warm = tmpl.lower(region, cfg)
    t1 = time.perf_counter()
    warm_ms = (t1 - t0) * 1000.0

    return {
        "template": "col_independent",
        "shape": (shape,),
        "cold_ms": round(cold_ms, 3),
        "warm_ms": round(warm_ms, 3),
        "ratio": round(cold_ms / max(warm_ms, 0.001), 3),
        "cache_dir": str(XCLBIN_CACHE_DIR),
    }


def measure_compute_pool() -> dict:
    shape = 32768
    region = _make_region_elementwise(shape)
    tmpl = ComputePoolTemplate()
    cfg = tmpl.config_space(region)[0]

    _clear_xclbin_cache("compute_pool", (shape,))

    t0 = time.perf_counter()
    iron_fn_cold = tmpl.lower(region, cfg)
    t1 = time.perf_counter()
    cold_ms = (t1 - t0) * 1000.0

    t0 = time.perf_counter()
    iron_fn_warm = tmpl.lower(region, cfg)
    t1 = time.perf_counter()
    warm_ms = (t1 - t0) * 1000.0

    return {
        "template": "compute_pool",
        "shape": (shape,),
        "cold_ms": round(cold_ms, 3),
        "warm_ms": round(warm_ms, 3),
        "ratio": round(cold_ms / max(warm_ms, 0.001), 3),
        "cache_dir": str(XCLBIN_CACHE_DIR),
    }


def measure_cgra() -> dict:
    shape = 256
    region = _make_region_cgra(shape)
    tmpl = CgraTemplate()
    cfg = tmpl.config_space(region)[0]

    _clear_xclbin_cache("cgra", (shape,))

    t0 = time.perf_counter()
    iron_fn_cold = tmpl.lower(region, cfg)
    t1 = time.perf_counter()
    cold_ms = (t1 - t0) * 1000.0

    t0 = time.perf_counter()
    iron_fn_warm = tmpl.lower(region, cfg)
    t1 = time.perf_counter()
    warm_ms = (t1 - t0) * 1000.0

    return {
        "template": "cgra",
        "shape": (shape,),
        "cold_ms": round(cold_ms, 3),
        "warm_ms": round(warm_ms, 3),
        "ratio": round(cold_ms / max(warm_ms, 0.001), 3),
        "cache_dir": str(XCLBIN_CACHE_DIR),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    print("[1/4] Measuring gemm_fusion (shape=128x128x128)...")
    records.append(measure_gemm_fusion())
    print(f"  cold={records[-1]['cold_ms']}ms warm={records[-1]['warm_ms']}ms ratio={records[-1]['ratio']}x")

    print("[2/4] Measuring col_independent (shape=16384)...")
    records.append(measure_col_independent())
    print(f"  cold={records[-1]['cold_ms']}ms warm={records[-1]['warm_ms']}ms ratio={records[-1]['ratio']}x")

    print("[3/4] Measuring compute_pool (shape=32768)...")
    records.append(measure_compute_pool())
    print(f"  cold={records[-1]['cold_ms']}ms warm={records[-1]['warm_ms']}ms ratio={records[-1]['ratio']}x")

    print("[4/4] Measuring cgra (shape=256)...")
    records.append(measure_cgra())
    print(f"  cold={records[-1]['cold_ms']}ms warm={records[-1]['warm_ms']}ms ratio={records[-1]['ratio']}x")

    RESULTS_FILE.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"\nWrote {RESULTS_FILE}")

    lines = [
        "Task V2-5: Compile Cache Cold vs Warm Latency",
        "=" * 50,
        "",
        f"Results file: {RESULTS_FILE}",
        f"Cache directory: {XCLBIN_CACHE_DIR}",
        "",
        "Summary:",
        "-" * 50,
    ]
    for r in records:
        lines.append(
            f"  {r['template']:20s}  shape={str(r['shape']):12s}  "
            f"cold={r['cold_ms']:>10.3f}ms  warm={r['warm_ms']:>8.3f}ms  "
            f"ratio={r['ratio']:>8.1f}x"
        )
    lines += ["", "All 4 templates measured successfully.", ""]
    EVIDENCE_FILE.write_text("\n".join(lines) + "\n")
    print(f"Wrote {EVIDENCE_FILE}")


if __name__ == "__main__":
    main()
