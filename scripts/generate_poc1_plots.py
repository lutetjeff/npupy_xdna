#!/usr/bin/env python3
"""Generate PoC 1 research-grade visualization plots from characterization data.

Produces 6 PNGs in results/03_heuristic_visualizations/:
  01_gemm_throughput.png          – GOPS vs shape (epilogue comparison)
  02_gemm_npu_vs_cpu_latency.png  – NPU vs CPU latency crossover
  03_bandwidth_scaling.png        – GB/s vs size: col_indep vs compute_pool
  04_template_decision_map.png    – 2D heatmap: (op_type x size) -> template
  05_offload_decision_map.png     – 2D heatmap: (op_type x size) -> offload/cpu
  06_all_templates_latency.png    – NPU + CPU latency bar chart across templates

Usage:
    python npupy_xdna/scripts/generate_poc1_plots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: no display server needed
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

from npupy_xdna.heuristic.viz import plot_crossover, _DPI, _FIGSIZE  # noqa: E402
from npupy_xdna.heuristic.classifier import RegionClassifier  # noqa: E402
from npupy_xdna.heuristic.cost_model import CostModel  # noqa: E402
from npupy_xdna.heuristic.offload import OffloadHeuristic  # noqa: E402
from npupy_xdna.regions.region import ArraySpec, Region  # noqa: E402

TIMINGS = REPO_ROOT / "npupy_xdna" / "results" / "timings"
OUT_DIR = REPO_ROOT / "npupy_xdna" / "results" / "03_heuristic_visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _make_matmul_region(M: int) -> Region:
    return Region(
        op="matmul",
        inputs=[ArraySpec((M, M), "int16"), ArraySpec((M, M), "int16")],
        output=ArraySpec((M, M), "int16"),
    )


def _make_elem_region(op: str, n: int) -> Region:
    inputs = (
        [ArraySpec((n,), "int16"), ArraySpec((n,), "int16")]
        if op == "elementwise_binary"
        else [ArraySpec((n,), "int16")]
    )
    return Region(op=op, inputs=inputs, output=ArraySpec((n,), "int16"))


def plot1_gemm_throughput() -> None:
    records = _load_jsonl(TIMINGS / "gemm_fusion.jsonl")

    groups: dict[str, list[tuple[int, float]]] = {}
    for r in records:
        k = r["epilogue"]
        groups.setdefault(k, []).append((r["shape"][0], r["derived_gops"]))

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for label, pts in sorted(groups.items()):
        pts_sorted = sorted(pts, key=lambda p: p[0])
        ax.plot([p[0] for p in pts_sorted], [p[1] for p in pts_sorted],
                marker="o", label=f"epilogue={label}")

    ax.annotate(
        "Peak: 5159 GOPS\n(2048^3, epilogue=none)",
        xy=(2048, 5159.1), xytext=(2500, 4500),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=8, color="red",
    )
    ax.set_title("GEMM Throughput: GOPS vs Matrix Dimension")
    ax.set_xlabel("Matrix Dimension M (=K=N)")
    ax.set_ylabel("Throughput (GOPS)")
    ax.legend(title="Epilogue")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_gemm_throughput.png", dpi=_DPI)
    plt.close(fig)
    print("  [OK] 01_gemm_throughput.png")


def plot2_gemm_crossover() -> None:
    records = _load_jsonl(TIMINGS / "gemm_fusion.jsonl")

    npu_data: list[dict] = []
    cpu_data: list[dict] = []
    for r in records:
        if r["epilogue"] != "none":
            continue
        m = r["shape"][0]
        npu_data.append({"m_size": m, "latency_us": r["npu_median_us"]})
        if r["cpu_median_us"] > 0:
            cpu_data.append({"m_size": m, "latency_us": r["cpu_median_us"]})

    plot_crossover(
        npu_data=npu_data,
        cpu_data=cpu_data,
        output_path=OUT_DIR / "02_gemm_npu_vs_cpu_latency.png",
        title="GEMM: NPU vs CPU Latency (epilogue=none)",
        x_key="m_size",
        y_key="latency_us",
    )
    print("  [OK] 02_gemm_npu_vs_cpu_latency.png")


def plot3_bandwidth_scaling() -> None:
    col_data  = sorted(_load_jsonl(TIMINGS / "col_indep.jsonl"),    key=lambda r: r["size"])
    pool_data = sorted(_load_jsonl(TIMINGS / "compute_pool.jsonl"), key=lambda r: r["size"])

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot([r["size"] for r in col_data],
            [r["bandwidth_gbps"] for r in col_data],
            marker="o", label="col_independent")
    ax.plot([r["size"] for r in pool_data],
            [r["bandwidth_gbps"] for r in pool_data],
            marker="s", linestyle="--", color="darkorange",
            label="compute_pool (dispatch-floored)")

    ax.set_xscale("log", base=2)
    ax.set_title("Elementwise Templates: Bandwidth vs Element Count")
    ax.set_xlabel("Number of Elements (log2 scale)")
    ax.set_ylabel("Effective Bandwidth (GB/s)")
    ax.legend()
    ax.grid(True)

    ax.annotate("Peak: 10.8 GB/s",
                xy=(1_048_576, 10.8145), xytext=(200_000, 9.0),
                arrowprops=dict(arrowstyle="->", color="steelblue"),
                fontsize=8, color="steelblue")
    ax.annotate("~15 ms dispatch floor",
                xy=(32_768, 0.0083), xytext=(65_536, 0.4),
                arrowprops=dict(arrowstyle="->", color="darkorange"),
                fontsize=8, color="darkorange")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_bandwidth_scaling.png", dpi=_DPI)
    plt.close(fig)
    print("  [OK] 03_bandwidth_scaling.png")


def plot4_template_decision_map() -> None:
    classifier = RegionClassifier()

    op_types  = ["matmul", "elementwise_unary", "elementwise_binary", "chained_elementwise"]
    all_sizes = sorted({
        256, 512, 1024, 2048, 4096,
        16_384, 32_768, 65_536, 131_072,
        262_144, 524_288, 1_048_576, 2_097_152,
    })

    table: list[list[str]] = []
    for op in op_types:
        row: list[str] = []
        for sz in all_sizes:
            try:
                region = _make_matmul_region(sz) if op == "matmul" else _make_elem_region(op, sz)
                match = classifier.classify(region)
                row.append(match.template_name if match else "cpu_fallback")
            except Exception:
                row.append("n/a")
        table.append(row)

    all_labels = sorted({cell for row in table for cell in row if cell != "n/a"})
    lbl2int    = {lbl: i for i, lbl in enumerate(all_labels)}
    n_lbls     = len(all_labels)

    grid = np.full((len(op_types), len(all_sizes)), np.nan)
    for oi, row in enumerate(table):
        for si, cell in enumerate(row):
            if cell in lbl2int:
                grid[oi, si] = lbl2int[cell]

    cmap = plt.get_cmap("tab10", n_lbls)
    fig, ax = plt.subplots(figsize=(max(12, len(all_sizes) * 0.9), 4))
    im = ax.imshow(grid, aspect="auto", cmap=cmap,
                   vmin=-0.5, vmax=n_lbls - 0.5, interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, ticks=list(range(n_lbls)))
    cbar.ax.set_yticklabels(all_labels, fontsize=8)
    cbar.set_label("Selected Template")

    ax.set_xticks(range(len(all_sizes)))
    ax.set_xticklabels([str(s) for s in all_sizes], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(op_types)))
    ax.set_yticklabels(op_types, fontsize=9)
    ax.set_title("Template Decision Map: Op Type x Shape Size")
    ax.set_xlabel("Shape Size  (elements for elementwise; M=K=N for matmul)")
    ax.set_ylabel("Operation Type")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_template_decision_map.png", dpi=_DPI)
    plt.close(fig)
    print("  [OK] 04_template_decision_map.png")


def plot5_offload_decision_map() -> None:
    cost_model = CostModel()
    classifier = RegionClassifier()
    heuristic  = OffloadHeuristic(cost_model, classifier)

    op_types  = ["matmul", "elementwise_unary", "elementwise_binary", "chained_elementwise"]
    all_sizes = sorted({
        256, 512, 1024, 2048, 4096,
        16_384, 32_768, 65_536, 131_072,
        262_144, 524_288, 1_048_576, 2_097_152,
    })

    ACTION_MAP = {"offload": 0, "cpu_fallback": 1}  # 0=green, 1=red in RdYlGn_r

    grid = np.full((len(op_types), len(all_sizes)), np.nan)
    for oi, op in enumerate(op_types):
        for si, sz in enumerate(all_sizes):
            try:
                region = _make_matmul_region(sz) if op == "matmul" else _make_elem_region(op, sz)
                decision = heuristic.decide(region)
                if decision.action in ACTION_MAP:
                    grid[oi, si] = ACTION_MAP[decision.action]
            except Exception:
                pass

    cmap = plt.get_cmap("RdYlGn_r", 2)
    fig, ax = plt.subplots(figsize=(max(12, len(all_sizes) * 0.9), 4))
    im = ax.imshow(grid, aspect="auto", cmap=cmap,
                   vmin=-0.5, vmax=1.5, interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["offload -> NPU", "cpu_fallback"], fontsize=8)
    cbar.set_label("Dispatch Decision")

    ax.set_xticks(range(len(all_sizes)))
    ax.set_xticklabels([str(s) for s in all_sizes], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(op_types)))
    ax.set_yticklabels(op_types, fontsize=9)
    ax.set_title("Offload Decision Map: NPU vs CPU Fallback  (only GEMM offloads)")
    ax.set_xlabel("Shape Size")
    ax.set_ylabel("Operation Type")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_offload_decision_map.png", dpi=_DPI)
    plt.close(fig)
    print("  [OK] 05_offload_decision_map.png")


def plot6_all_templates_latency() -> None:
    gemm_data = _load_jsonl(TIMINGS / "gemm_fusion.jsonl")
    col_data  = _load_jsonl(TIMINGS / "col_indep.jsonl")
    pool_data = _load_jsonl(TIMINGS / "compute_pool.jsonl")
    cgra_data = _load_jsonl(TIMINGS / "cgra.jsonl")

    gemm_best = max((r for r in gemm_data if r["epilogue"] == "none"),
                    key=lambda r: r["derived_gops"])
    col_best  = max(col_data,  key=lambda r: r["bandwidth_gbps"])
    pool_best = max(pool_data, key=lambda r: r["bandwidth_gbps"])
    cgra_best = cgra_data[0]

    benchmarks = [
        {"label": "gemm_fusion\n(2048^3)",    "npu_us": gemm_best["npu_median_us"], "cpu_us": None},
        {"label": "col_indep\n(1 M elems)",   "npu_us": col_best["npu_median_us"],  "cpu_us": col_best["cpu_median_us"]},
        {"label": "compute_pool\n(2 M elems)","npu_us": pool_best["npu_median_us"], "cpu_us": pool_best["cpu_median_us"]},
        {"label": "cgra\n(256 elems)",         "npu_us": cgra_best["npu_median_us"], "cpu_us": cgra_best["cpu_median_us"]},
    ]

    x     = np.arange(len(benchmarks))
    width = 0.35

    npu_vals = [b["npu_us"] for b in benchmarks]
    cpu_vals = [b["cpu_us"] if b["cpu_us"] is not None else float("nan") for b in benchmarks]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_npu = ax.bar(x - width / 2, npu_vals, width, label="NPU", color="steelblue", zorder=3)
    bars_cpu = ax.bar(x + width / 2, cpu_vals, width, label="CPU", color="tomato",    zorder=3)

    for bar, val in zip(bars_npu, npu_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                f"{val:.0f}us", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars_cpu, cpu_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                    f"{val:.0f}us", ha="center", va="bottom", fontsize=7)

    ax.text(x[0] + width / 2, 1200,
            "CPU\n>1 s\n(est.)", ha="center", va="bottom",
            fontsize=6, color="gray", style="italic")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([b["label"] for b in benchmarks], fontsize=9)
    ax.set_title("NPU vs CPU Latency: All Templates at Peak-Performance Size")
    ax.set_xlabel("Template (best-throughput size)")
    ax.set_ylabel("Median Latency (us, log scale)")
    ax.legend()
    ax.grid(True, axis="y", zorder=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_all_templates_latency.png", dpi=_DPI)
    plt.close(fig)
    print("  [OK] 06_all_templates_latency.png")


def main() -> None:
    print(f"Output directory: {OUT_DIR}")
    print("Generating plots ...")
    plot1_gemm_throughput()
    plot2_gemm_crossover()
    plot3_bandwidth_scaling()
    plot4_template_decision_map()
    plot5_offload_decision_map()
    plot6_all_templates_latency()
    print(f"\nDone -- {len(list(OUT_DIR.glob('*.png')))} PNGs in {OUT_DIR}")


if __name__ == "__main__":
    main()
