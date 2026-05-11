#!/usr/bin/env python3
"""Generate V2 research-grade visualization plots.

Produces PNGs in results/03_heuristic_visualizations/:
  09_cgra_chain_depth.png     – per-op cost vs chain depth
  10_tile_size_sweep.png      – GOPS per tile config
  11_compile_cache_stats.png  – cold vs warm compile time bar chart
  12_arithmetic_intensity.png – bandwidth (GB/s) vs size for col_indep vs tanh
  13_col_indep_extended.png   – bandwidth scaling including 2M/4M points
  14_compute_pool_8core.png   – 8-core vs 32-core dispatch comparison

Usage:
    python npupy_xdna/scripts/generate_v2_plots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

TIMINGS = REPO_ROOT / "npupy_xdna" / "results" / "timings"
OUT_DIR = REPO_ROOT / "npupy_xdna" / "results" / "03_heuristic_visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_DPI = 150
_FIGSIZE = (8, 5)

SKIPPED: list[str] = []
GENERATED: list[str] = []


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Plot 09: CGRA chain depth – per-op latency vs chain depth
# ---------------------------------------------------------------------------
def plot09_cgra_chain_depth() -> None:
    name = "09_cgra_chain_depth.png"
    records = _load_jsonl(TIMINGS / "cgra_depth_sweep.jsonl")
    if not records:
        SKIPPED.append(f"{name} (cgra_depth_sweep.jsonl empty/missing)")
        return

    depths = [r["depth"] for r in records]
    per_op = [r["per_op_latency_us"] for r in records]
    npu_total = [r["npu_median_us"] for r in records]
    cpu_total = [r["cpu_median_us"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(depths, per_op, marker="o", color="steelblue", label="NPU per-op latency")
    ax.set_xlabel("Chain Depth (# ops)")
    ax.set_ylabel("Per-Op Latency (µs)")
    ax.set_title("CGRA: Per-Op Cost vs Chain Depth")
    ax.grid(True, alpha=0.4)
    ax.legend()

    ax2 = axes[1]
    ax2.plot(depths, npu_total, marker="o", color="steelblue", label="NPU total")
    ax2.plot(depths, cpu_total, marker="s", color="darkorange", label="CPU total")
    ax2.set_xlabel("Chain Depth (# ops)")
    ax2.set_ylabel("Total Latency (µs)")
    ax2.set_title("CGRA: NPU vs CPU Total Latency")
    ax2.grid(True, alpha=0.4)
    ax2.legend()

    fig.suptitle("CGRA Chain Depth Sweep (256 elements)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / name
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    GENERATED.append(str(out))
    print(f"  [OK] {name}")


# ---------------------------------------------------------------------------
# Plot 10: Tile size sweep – GOPS per tile config
# ---------------------------------------------------------------------------
def plot10_tile_size_sweep() -> None:
    name = "10_tile_size_sweep.png"
    records = _load_jsonl(TIMINGS / "gemm_tile_sweep.jsonl")
    if not records:
        SKIPPED.append(f"{name} (gemm_tile_sweep.jsonl empty/missing)")
        return

    # Only records with derived_gops (skip error rows)
    good = [r for r in records if "derived_gops" in r and "error" not in r]
    if not good:
        SKIPPED.append(f"{name} (no successful gemm_tile_sweep records)")
        return

    labels = [f"{r['tile'][0]}x{r['tile'][1]}x{r['tile'][2]}" for r in good]
    gops = [r["derived_gops"] for r in good]
    latencies = [r["npu_median_us"] / 1000 for r in good]  # convert to ms

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bars = ax.bar(labels, gops, color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(good)])
    ax.set_xlabel("Tile Config (M×K×N)")
    ax.set_ylabel("Derived GOPS")
    ax.set_title("GEMM Tile Sweep: Throughput")
    for bar, g in zip(bars, gops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{g:.0f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, axis="y", alpha=0.4)

    ax2 = axes[1]
    bars2 = ax2.bar(labels, latencies, color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(good)])
    ax2.set_xlabel("Tile Config (M×K×N)")
    ax2.set_ylabel("NPU Median Latency (ms)")
    ax2.set_title("GEMM Tile Sweep: Latency")
    for bar, l in zip(bars2, latencies):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{l:.2f}", ha="center", va="bottom", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.4)

    shape = good[0]["shape"]
    fig.suptitle(f"Tile Size Sweep: {shape[0]}³ GEMM (2M×2M×2M)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / name
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    GENERATED.append(str(out))
    print(f"  [OK] {name}")


# ---------------------------------------------------------------------------
# Plot 11: Compile cache stats – cold vs warm bar chart
# ---------------------------------------------------------------------------
def plot11_compile_cache() -> None:
    name = "11_compile_cache_stats.png"
    records = _load_jsonl(TIMINGS / "compile_cache.jsonl")
    if not records:
        SKIPPED.append(f"{name} (compile_cache.jsonl empty/missing)")
        return

    templates = [r["template"] for r in records]
    cold = [r["cold_ms"] for r in records]
    warm = [r["warm_ms"] for r in records]
    ratios = [r["ratio"] for r in records]

    x = np.arange(len(templates))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bars1 = ax.bar(x - width / 2, cold, width, label="Cold (first load)", color="#d62728")
    bars2 = ax.bar(x + width / 2, warm, width, label="Warm (cached)", color="#2ca02c")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(templates, rotation=15, ha="right")
    ax.set_ylabel("Compile Time (ms, log scale)")
    ax.set_title("Compile Cache: Cold vs Warm Load Time")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)

    ax2 = axes[1]
    ax2.bar(x, ratios, color="#9467bd")
    ax2.set_xticks(x)
    ax2.set_xticklabels(templates, rotation=15, ha="right")
    ax2.set_ylabel("Cold / Warm Ratio")
    ax2.set_title("Compile Cache: Speedup from Caching")
    for i, r in enumerate(ratios):
        ax2.text(i, r + 20, f"{r:.0f}×", ha="center", va="bottom", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.4)

    fig.suptitle("Compile Cache Statistics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / name
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    GENERATED.append(str(out))
    print(f"  [OK] {name}")


# ---------------------------------------------------------------------------
# Plot 12: Arithmetic intensity – bandwidth vs size for col_indep vs tanh
# ---------------------------------------------------------------------------
def plot12_arithmetic_intensity() -> None:
    name = "12_arithmetic_intensity.png"

    col_records = _load_jsonl(TIMINGS / "col_indep.jsonl")
    tanh_records = _load_jsonl(TIMINGS / "tanh.jsonl")

    if not col_records and not tanh_records:
        SKIPPED.append(f"{name} (col_indep.jsonl and tanh.jsonl both empty/missing)")
        return

    fig, ax = plt.subplots(figsize=_FIGSIZE)

    if col_records:
        c_sorted = sorted(col_records, key=lambda r: r["size"])
        xs = [r["size"] for r in c_sorted]
        ys = [r["bandwidth_gbps"] for r in c_sorted]
        ax.plot(xs, ys, marker="o", color="#1f77b4", label="col_indep (elementwise/relu)")

    if tanh_records:
        t_sorted = sorted(tanh_records, key=lambda r: r["size"])
        xs = [r["size"] for r in t_sorted]
        ys = [r["bandwidth_gbps"] for r in t_sorted]
        ax.plot(xs, ys, marker="s", color="#d62728", label="tanh (via col_indep)")

    # Note about hash
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Element Count")
    ax.set_ylabel("Effective Bandwidth (GB/s)")
    ax.set_title("Arithmetic Intensity: Bandwidth vs Size\n(hash.jsonl missing – skipped)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.annotate("hash.jsonl\nnot collected",
                xy=(0.65, 0.15), xycoords="axes fraction",
                fontsize=8, color="gray", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))

    fig.tight_layout()
    out = OUT_DIR / name
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    GENERATED.append(str(out))
    print(f"  [OK] {name}")


# ---------------------------------------------------------------------------
# Plot 13: col_indep extended – bandwidth scaling with 2M/4M points
# ---------------------------------------------------------------------------
def plot13_col_indep_extended() -> None:
    name = "13_col_indep_extended.png"
    col_records = _load_jsonl(TIMINGS / "col_indep.jsonl")
    if not col_records:
        SKIPPED.append(f"{name} (col_indep.jsonl empty/missing)")
        return

    sorted_recs = sorted(col_records, key=lambda r: r["size"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    xs = [r["size"] for r in sorted_recs]
    bw = [r["bandwidth_gbps"] for r in sorted_recs]
    # highlight 2M and 4M points
    colors = ["#1f77b4" if r["size"] <= 1048576 else "#d62728" for r in sorted_recs]
    for i in range(len(xs) - 1):
        ax.plot([xs[i], xs[i+1]], [bw[i], bw[i+1]], color="steelblue", linewidth=1.5)
    for x, y, c in zip(xs, bw, colors):
        ax.scatter([x], [y], color=c, zorder=5, s=60)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Element Count")
    ax.set_ylabel("Effective Bandwidth (GB/s)")
    ax.set_title("col_indep: Bandwidth Scaling (Extended)")
    ax.grid(True, which="both", alpha=0.3)
    # annotate 2M and 4M
    for r in sorted_recs:
        if r["size"] >= 2097152:
            ax.annotate(f"{r['size']//1048576}M\n{r['bandwidth_gbps']:.1f} GB/s",
                        xy=(r["size"], r["bandwidth_gbps"]),
                        xytext=(10, -15), textcoords="offset points",
                        fontsize=8, color="darkred",
                        arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))
    import matplotlib.patches as mpatches
    v1_patch = mpatches.Patch(color="steelblue", label="V1 data (≤1M elements)")
    v2_patch = mpatches.Patch(color="#d62728", label="V2 extension (2M, 4M)")
    ax.legend(handles=[v1_patch, v2_patch])

    ax2 = axes[1]
    speedups = [r["speedup"] for r in sorted_recs]
    ax2.plot(xs, speedups, marker="^", color="darkgreen")
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="NPU = CPU")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Element Count")
    ax2.set_ylabel("NPU/CPU Speedup")
    ax2.set_title("col_indep: NPU vs CPU Speedup")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    fig.suptitle("col_indep Template: Extended Scaling (up to 4M elements)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / name
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    GENERATED.append(str(out))
    print(f"  [OK] {name}")


# ---------------------------------------------------------------------------
# Plot 14: compute_pool 8-core vs 32-core dispatch comparison
# ---------------------------------------------------------------------------
def plot14_compute_pool_8core() -> None:
    name = "14_compute_pool_8core.png"
    r8 = _load_jsonl(TIMINGS / "compute_pool_8core.jsonl")
    r32 = _load_jsonl(TIMINGS / "compute_pool.jsonl")

    if not r8 and not r32:
        SKIPPED.append(f"{name} (both compute_pool*.jsonl empty/missing)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for label, records, marker, color in [
        ("8-core pool", r8, "^", "#d62728"),
        ("32-core pool", r32, "o", "#1f77b4"),
    ]:
        if records:
            s = sorted(records, key=lambda r: r["size"])
            xs = [r["size"] for r in s]
            bw = [r["bandwidth_gbps"] for r in s]
            ax.plot(xs, bw, marker=marker, color=color, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Element Count")
    ax.set_ylabel("Effective Bandwidth (GB/s)")
    ax.set_title("compute_pool: 8-core vs 32-core Bandwidth")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax2 = axes[1]
    for label, records, marker, color in [
        ("8-core pool", r8, "^", "#d62728"),
        ("32-core pool", r32, "o", "#1f77b4"),
    ]:
        if records:
            s = sorted(records, key=lambda r: r["size"])
            xs = [r["size"] for r in s]
            lat = [r["npu_median_us"] / 1000 for r in s]  # ms
            ax2.plot(xs, lat, marker=marker, color=color, label=label)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Element Count")
    ax2.set_ylabel("NPU Median Latency (ms)")
    ax2.set_title("compute_pool: Dispatch Floor Comparison")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle("Compute Pool: 8-Core vs 32-Core Active-Core Sweep", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / name
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    GENERATED.append(str(out))
    print(f"  [OK] {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating V2 plots...")
    plot09_cgra_chain_depth()
    plot10_tile_size_sweep()
    plot11_compile_cache()
    plot12_arithmetic_intensity()
    plot13_col_indep_extended()
    plot14_compute_pool_8core()

    print(f"\nGenerated ({len(GENERATED)}):")
    for p in GENERATED:
        print(f"  {p}")

    if SKIPPED:
        print(f"\nSkipped ({len(SKIPPED)}):")
        for s in SKIPPED:
            print(f"  {s}")
    else:
        print("\nNo plots skipped.")
