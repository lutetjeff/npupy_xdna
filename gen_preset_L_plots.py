import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from pathlib import Path

DATA = Path("results/timings/npbench_preset_L.jsonl")
OUT  = Path("results/04_npbench_plots")
OUT.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]

data_sorted = sorted(rows, key=lambda r: r["speedup_vs_blas"], reverse=True)

names    = [r["benchmark"] for r in data_sorted]
speedups = [r["speedup_vs_blas"] for r in data_sorted]

def bar_color(s):
    if s > 1.5:
        return "#2ecc71"
    elif s < 0.9:
        return "#e74c3c"
    return "#95a5a6"

colors = [bar_color(s) for s in speedups]

fig, ax = plt.subplots(figsize=(15, 7))

bars = ax.bar(range(len(names)), speedups, color=colors, edgecolor="white", linewidth=0.5)

for i, (bar, s) in enumerate(zip(bars, speedups)):
    label  = f"{s:.1f}×" if s >= 2 else f"{s:.2f}×"
    y_top  = bar.get_height()
    y_pos  = y_top + max(speedups) * 0.01 if y_top < max(speedups) * 0.85 else y_top * 0.90
    c      = "black" if y_top < max(speedups) * 0.85 else "white"
    ax.text(i, y_pos, label, ha="center", va="bottom", fontsize=9, fontweight="bold", color=c)

ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7,
           label="CPU BLAS baseline (1.0×)")

ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=38, ha="right", fontsize=10)
ax.set_ylabel("Speedup vs CPU BLAS (higher = better)", fontsize=12)
ax.set_title(
    "NPUPy Speedup vs BLAS-Accelerated CPU — Preset L\n"
    "(GEMM @ 2048², Elementwise @ 4 M elements)",
    fontsize=13
)

legend_handles = [
    ax.get_legend_handles_labels()[0][0],
    mpatches.Patch(color="#2ecc71", label="NPU wins (>1.5×)"),
    mpatches.Patch(color="#95a5a6", label="Roughly even (0.9–1.5×)"),
    mpatches.Patch(color="#e74c3c", label="BLAS wins (<0.9×)"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

ymax = max(speedups) * 1.15
ax.set_ylim(0, ymax)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()

out1 = OUT / "preset_L_speedup_vs_blas.png"
fig.savefig(out1, dpi=150)
plt.close(fig)
print(f"[OK] {out1}  ({out1.stat().st_size // 1024} KB)")


TMPL_COLOR = {
    "gemm_fusion":              "#27ae60",
    "gemm_fusion(x3)":          "#2ecc71",
    "gemm_fusion(x2)+cpu_add":  "#1abc9c",
    "col_independent":          "#2980b9",
    "cpu_fallback":             "#c0392b",
    "sliding_window":           "#8e44ad",
}
TMPL_ORDER = ["gemm_fusion", "gemm_fusion(x3)", "gemm_fusion(x2)+cpu_add",
              "col_independent", "sliding_window", "cpu_fallback"]

def sort_key(r):
    t = r["template"]
    return (0 if t != "cpu_fallback" else 1, t, r["benchmark"])

data_tmpl = sorted(rows, key=sort_key)

names_t   = [r["benchmark"] for r in data_tmpl]
templates = [r["template"]  for r in data_tmpl]
colors_t  = [TMPL_COLOR.get(t, "#7f8c8d") for t in templates]
speedup_t = [r["speedup_vs_blas"] for r in data_tmpl]

fig, ax = plt.subplots(figsize=(13, 7))

y = np.arange(len(names_t))
ax.barh(y, [1] * len(names_t), color=colors_t, edgecolor="white", linewidth=0.5, height=0.7)

for i, (tmpl, spd) in enumerate(zip(templates, speedup_t)):
    ax.text(0.50, i, tmpl, ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    spd_str = f"{spd:.1f}×" if spd >= 2 else f"{spd:.2f}×"
    ax.text(0.97, i, spd_str, ha="right", va="center", fontsize=8, color="white", alpha=0.9)

ax.set_yticks(y)
ax.set_yticklabels(names_t, fontsize=10)
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title("Template Dispatch Decision — Preset L", fontsize=13)
ax.set_xlabel("Template assigned by NPUPy dispatcher", fontsize=11)
seen = set(templates)
legend_handles = [
    mpatches.Patch(facecolor=TMPL_COLOR.get(t, "#7f8c8d"), label=t)
    for t in TMPL_ORDER if t in seen
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
          framealpha=0.85)

ax.grid(axis="x", alpha=0)
fig.tight_layout()

out2 = OUT / "preset_L_template_usage.png"
fig.savefig(out2, dpi=150)
plt.close(fig)
print(f"[OK] {out2}  ({out2.stat().st_size // 1024} KB)")
