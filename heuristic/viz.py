from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_FIGSIZE = (8, 5)
_DPI = 150


def _ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _load_jsonl(jsonl_path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(jsonl_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_throughput_curve(
    jsonl_path: str | Path,
    output_path: str | Path,
    title: str,
    x_key: str,
    y_key: str,
    group_key: str | None = None,
) -> None:
    records = _load_jsonl(jsonl_path)
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=_FIGSIZE)

    if group_key is not None:
        groups: dict[Any, list[dict]] = {}
        for rec in records:
            key = rec.get(group_key, "unknown")
            groups.setdefault(key, []).append(rec)
        for label, grp in sorted(groups.items(), key=lambda kv: str(kv[0])):
            grp_sorted = sorted(grp, key=lambda r: r[x_key])
            xs = [r[x_key] for r in grp_sorted]
            ys = [r[y_key] for r in grp_sorted]
            ax.plot(xs, ys, marker="o", label=str(label))
        ax.legend(title=str(group_key))
    else:
        records_sorted = sorted(records, key=lambda r: r[x_key])
        xs = [r[x_key] for r in records_sorted]
        ys = [r[y_key] for r in records_sorted]
        ax.plot(xs, ys, marker="o")

    ax.set_title(title)
    ax.set_xlabel(str(x_key))
    ax.set_ylabel(str(y_key))
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)


def plot_crossover(
    npu_data: list[dict],
    cpu_data: list[dict],
    output_path: str | Path,
    title: str,
    x_key: str = "size",
    y_key: str = "throughput_gops",
) -> None:
    _ensure_dir(output_path)

    npu_sorted = sorted(npu_data, key=lambda r: r[x_key])
    cpu_sorted = sorted(cpu_data, key=lambda r: r[x_key])

    npu_xs = np.array([r[x_key] for r in npu_sorted], dtype=float)
    npu_ys = np.array([r[y_key] for r in npu_sorted], dtype=float)
    cpu_xs = np.array([r[x_key] for r in cpu_sorted], dtype=float)
    cpu_ys = np.array([r[y_key] for r in cpu_sorted], dtype=float)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(npu_xs, npu_ys, marker="o", label="NPU")
    ax.plot(cpu_xs, cpu_ys, marker="s", label="CPU")

    if len(npu_xs) >= 2 and len(cpu_xs) >= 2:
        x_common = np.linspace(
            max(npu_xs.min(), cpu_xs.min()),
            min(npu_xs.max(), cpu_xs.max()),
            500,
        )
        npu_interp = np.interp(x_common, npu_xs, npu_ys)
        cpu_interp = np.interp(x_common, cpu_xs, cpu_ys)
        diff = npu_interp - cpu_interp
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        for idx in sign_changes:
            x0, x1 = x_common[idx], x_common[idx + 1]
            d0, d1 = diff[idx], diff[idx + 1]
            if d1 != d0:
                cx = x0 - d0 * (x1 - x0) / (d1 - d0)
                cy = np.interp(cx, npu_xs, npu_ys)
                ax.axvline(cx, color="gray", linestyle="--", alpha=0.7)
                ax.scatter([cx], [cy], color="red", zorder=5, label=f"Crossover ~{cx:.1f}")

    ax.set_title(title)
    ax.set_xlabel(str(x_key))
    ax.set_ylabel(str(y_key))
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)


def plot_decision_map(
    decisions: dict,
    output_path: str | Path,
    title: str,
    x_label: str = "M (rows)",
    y_label: str = "K (cols)",
) -> None:
    _ensure_dir(output_path)

    if not decisions:
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.tight_layout()
        fig.savefig(output_path, dpi=_DPI)
        plt.close(fig)
        return

    keys = list(decisions.keys())
    xs_vals = sorted(set(k[0] for k in keys))
    ys_vals = sorted(set(k[1] for k in keys))

    label_set = sorted(set(decisions.values()))
    label_to_int = {lbl: i for i, lbl in enumerate(label_set)}

    grid = np.full((len(ys_vals), len(xs_vals)), np.nan)
    for xi, xv in enumerate(xs_vals):
        for yi, yv in enumerate(ys_vals):
            lbl = decisions.get((xv, yv))
            if lbl is not None:
                grid[yi, xi] = label_to_int[lbl]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    n_labels = max(len(label_set), 1)
    cmap = plt.get_cmap("tab10", n_labels)
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=n_labels - 0.5,
        extent=[
            xs_vals[0] - 0.5,
            xs_vals[-1] + 0.5,
            ys_vals[0] - 0.5,
            ys_vals[-1] + 0.5,
        ],
    )
    cbar = fig.colorbar(im, ax=ax, ticks=list(range(n_labels)))
    cbar.ax.set_yticklabels(label_set)
    cbar.set_label("Template")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)


def plot_speedup_bars(
    records: list[dict],
    output_path: str | Path,
    title: str,
    name_key: str = "benchmark",
    speedup_key: str = "speedup",
) -> None:
    _ensure_dir(output_path)

    names = [str(r[name_key]) for r in records]
    speedups = [float(r[speedup_key]) for r in records]

    y_pos = np.arange(len(names))
    colors = ["steelblue" if s >= 1.0 else "tomato" for s in speedups]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.6 + 2), 5))
    bars = ax.bar(y_pos, speedups, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, label="Baseline (1x)")

    for bar, val in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02 * max(speedups, default=1),
            f"{val:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(y_pos)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Speedup over CPU (x)")
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)


def plot_bandwidth_scaling(
    jsonl_path: str | Path,
    output_path: str | Path,
    title: str,
    size_key: str = "size_bytes",
    bw_key: str = "bandwidth_gbs",
    group_key: str | None = None,
) -> None:
    records = _load_jsonl(jsonl_path)
    _ensure_dir(output_path)

    fig, ax = plt.subplots(figsize=_FIGSIZE)

    if group_key is not None:
        groups: dict[Any, list[dict]] = {}
        for rec in records:
            key = rec.get(group_key, "unknown")
            groups.setdefault(key, []).append(rec)
        for label, grp in sorted(groups.items(), key=lambda kv: str(kv[0])):
            grp_sorted = sorted(grp, key=lambda r: r[size_key])
            xs = [r[size_key] for r in grp_sorted]
            ys = [r[bw_key] for r in grp_sorted]
            ax.plot(xs, ys, marker="o", label=str(label))
        ax.legend(title=str(group_key))
    else:
        records_sorted = sorted(records, key=lambda r: r[size_key])
        xs = [r[size_key] for r in records_sorted]
        ys = [r[bw_key] for r in records_sorted]
        ax.plot(xs, ys, marker="o")

    ax.set_title(title)
    ax.set_xlabel("Problem Size (bytes)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)
