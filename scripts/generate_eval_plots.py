#!/usr/bin/env python3
"""Generate NPBench evaluation plots for the ECE511 final report.

Produces 4 PNGs in results/04_npbench_plots/:
  01_speedup_bar.png          – speedup per benchmark (NPUPy vs vanilla numpy)
  02_latency_comparison.png   – median latency: numpy vs NPUPy (log-scale y)
  03_template_usage.png       – which template was used for each benchmark
  04_gemm_scaling.png         – GOPS vs matrix dimension (from gemm_fusion.jsonl)

Usage:
    python npupy_xdna/scripts/generate_eval_plots.py

Data source (T26/T27):
    npupy_xdna/results/04_npbench_evaluation.jsonl

If that file does not yet exist, the first three plots are generated with
DUMMY data and a prominent "PENDING DATA" watermark so the report skeleton
can be built before T26/T27 complete.  Re-run this script after T26/T27
finish to replace the placeholders with real figures.

Expected JSONL schema (one record per benchmark run):
    {
        "benchmark":     str,          # e.g. "npbench.jacobi_1d"
        "template":      str,          # e.g. "col_indep" | "compute_pool" | "gemm_fusion" | "cgra" | "cpu"
        "numpy_median_us": float,      # vanilla numpy median latency in µs
        "npupy_median_us": float,      # NPUPy median latency in µs  (>0 means NPU ran)
        "speedup":       float,        # numpy_median_us / npupy_median_us
        "correct":       bool          # correctness check passed
    }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

EVAL_JSONL = REPO_ROOT / "npupy_xdna" / "results" / "04_npbench_evaluation.jsonl"
TIMINGS    = REPO_ROOT / "npupy_xdna" / "results" / "timings"
OUT_DIR    = REPO_ROOT / "npupy_xdna" / "results" / "04_npbench_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_DPI     = 150
_FIGSIZE = (10, 5)

TEMPLATE_COLORS = {
    "gemm_fusion":  "#1f77b4",
    "col_indep":    "#ff7f0e",
    "compute_pool": "#2ca02c",
    "cgra":         "#d62728",
    "cpu":          "#9467bd",
    "unknown":      "#8c564b",
}


# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _add_pending_watermark(ax: plt.Axes) -> None:
    ax.text(
        0.5, 0.5, "PENDING DATA",
        transform=ax.transAxes,
        fontsize=28, color="red", alpha=0.25,
        ha="center", va="center", rotation=30,
        fontweight="bold",
        zorder=10,
    )


def _dummy_eval_records() -> list[dict]:
    benchmarks = [
        ("npbench.jacobi_1d",      "col_indep",    120_000,  3_800),
        ("npbench.jacobi_2d",      "col_indep",    980_000, 28_000),
        ("npbench.heat_3d",        "col_indep",  3_200_000, 90_000),
        ("npbench.floyd_warshall", "cgra",          85_000, 14_000),
        ("npbench.nussinov",       "cpu",           42_000, 44_000),
        ("npbench.gemm",           "gemm_fusion",  900_000,  2_200),
        ("npbench.gemver",         "gemm_fusion",  870_000,  2_100),
        ("npbench.syrk",           "compute_pool", 130_000,  8_500),
        ("npbench.atax",           "col_indep",     55_000,  3_200),
        ("npbench.bicg",           "col_indep",     58_000,  3_300),
        ("npbench.doitgen",        "compute_pool",  95_000,  6_000),
        ("npbench.mvt",            "col_indep",     61_000,  3_600),
    ]
    records: list[dict] = []
    for bench, tmpl, np_us, npu_us in benchmarks:
        records.append({
            "benchmark":       bench,
            "template":        tmpl,
            "numpy_median_us": np_us,
            "npupy_median_us": npu_us,
            "speedup":         np_us / npu_us,
            "correct":         True,
        })
    return records


# ---------------------------------------------------------------------------
def plot1_speedup_bar(records: list[dict], pending: bool) -> None:
    names     = [r["benchmark"].split(".")[-1] for r in records]
    speedups  = [r["speedup"] for r in records]
    templates = [r.get("template", "unknown") for r in records]
    colors    = [TEMPLATE_COLORS.get(t, TEMPLATE_COLORS["unknown"]) for t in templates]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    x = np.arange(len(names))
    ax.bar(x, speedups, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="1× (no gain)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Speedup (× over NumPy)")
    ax.set_title("NPUPy Speedup per NPBench Benchmark")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles = [
        mpatches.Patch(color=c, label=t)
        for t, c in TEMPLATE_COLORS.items()
        if t in set(templates)
    ]
    ax.legend(handles=handles, title="Template", fontsize=7, loc="upper right")

    if pending:
        _add_pending_watermark(ax)

    fig.tight_layout()
    out = OUT_DIR / "01_speedup_bar.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    print(f"  [OK] {out.name}{'  (placeholder)' if pending else ''}")


def plot2_latency_comparison(records: list[dict], pending: bool) -> None:
    names    = [r["benchmark"].split(".")[-1] for r in records]
    np_lats  = [r["numpy_median_us"] for r in records]
    npu_lats = [r["npupy_median_us"] for r in records]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    x = np.arange(len(names))
    w = 0.38

    ax.bar(x - w / 2, np_lats,  width=w, label="NumPy (baseline)",
           color="#aec7e8", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, npu_lats, width=w, label="NPUPy",
           color="#1f77b4", edgecolor="black", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Median Latency (µs) — log scale")
    ax.set_title("Latency Comparison: NumPy vs NPUPy per Benchmark")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, which="both")

    if pending:
        _add_pending_watermark(ax)

    fig.tight_layout()
    out = OUT_DIR / "02_latency_comparison.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    print(f"  [OK] {out.name}{'  (placeholder)' if pending else ''}")


def plot3_template_usage(records: list[dict], pending: bool) -> None:
    from collections import Counter

    counter = Counter(r.get("template", "unknown") for r in records)
    templates = list(counter.keys())
    counts    = [counter[t] for t in templates]
    colors    = [TEMPLATE_COLORS.get(t, TEMPLATE_COLORS["unknown"]) for t in templates]

    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(templates))
    hbars = ax.barh(y, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(templates, fontsize=9)
    ax.set_xlabel("Number of Benchmarks Dispatched")
    ax.set_title("Template Usage Breakdown across NPBench Suite")
    ax.bar_label(hbars, padding=3, fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    if pending:
        _add_pending_watermark(ax)

    fig.tight_layout()
    out = OUT_DIR / "03_template_usage.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    print(f"  [OK] {out.name}{'  (placeholder)' if pending else ''}")


def plot4_gemm_scaling() -> None:
    path = TIMINGS / "gemm_fusion.jsonl"
    if not path.exists():
        print(f"  [SKIP] 04_gemm_scaling.png — {path} not found")
        return

    records = _load_jsonl(path)

    groups: dict[str, list[tuple[int, float]]] = {}
    for r in records:
        k = r.get("epilogue", "none")
        groups.setdefault(k, []).append((r["shape"][0], r["derived_gops"]))

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    markers = ["o", "s", "^", "D", "v"]
    for idx, (label, pts) in enumerate(sorted(groups.items())):
        pts_sorted = sorted(pts, key=lambda p: p[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        ax.plot(xs, ys, marker=markers[idx % len(markers)],
                label=f"epilogue={label}", linewidth=1.5, markersize=6)

    all_pts = [(r["shape"][0], r["derived_gops"]) for r in records]
    if all_pts:
        peak = max(all_pts, key=lambda p: p[1])
        ax.annotate(
            f"Peak: {peak[1]:.0f} GOPS\n(M={peak[0]})",
            xy=peak,
            xytext=(peak[0] * 0.6, peak[1] * 0.85),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=8, color="red",
        )

    ax.set_title("GEMM Throughput: GOPS vs Matrix Dimension (int16, NPU)")
    ax.set_xlabel("Matrix Dimension M (= K = N)")
    ax.set_ylabel("Throughput (GOPS)")
    ax.legend(title="Epilogue", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out = OUT_DIR / "04_gemm_scaling.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    print(f"  [OK] {out.name}  (real data)")


# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Output directory: {OUT_DIR}")
    print(f"Evaluation JSONL: {EVAL_JSONL}")

    if EVAL_JSONL.exists():
        raw = _load_jsonl(EVAL_JSONL)
        records = []
        for r in raw:
            norm = dict(r)
            np_ms = r.get("vanilla_numpy_ms", r.get("numpy_ms", 0))
            npu_ms = r.get("npupy_ms", r.get("npu_ms", 0))
            if "numpy_median_us" not in norm:
                norm["numpy_median_us"] = np_ms * 1000
            if "npupy_median_us" not in norm:
                norm["npupy_median_us"] = npu_ms * 1000
            if "speedup" not in norm or norm["speedup"] is None:
                norm["speedup"] = (np_ms / npu_ms) if npu_ms > 0 else 1.0
            if "template" not in norm:
                norm["template"] = r.get("template_used", "cpu")
            if "correct" not in norm:
                norm["correct"] = r.get("correctness", True)
            records.append(norm)
        pending = False
        print(f"  Loaded {len(records)} evaluation records (T26/T27 data present)")
    else:
        records = _dummy_eval_records()
        pending = True
        print(f"  WARNING: {EVAL_JSONL.name} not found — using DUMMY data (re-run after T26/T27)")

    if not records:
        print("  ERROR: No records available — aborting")
        sys.exit(1)

    print("\nGenerating plots …")
    plot1_speedup_bar(records, pending)
    plot2_latency_comparison(records, pending)
    plot3_template_usage(records, pending)
    plot4_gemm_scaling()

    print(f"\nDone — {len(list(OUT_DIR.glob('*.png')))} PNG(s) in {OUT_DIR}")
    if pending:
        print("\n  [!] Plots 01–03 are PLACEHOLDERS. Re-run after T26/T27 complete.")


if __name__ == "__main__":
    main()
