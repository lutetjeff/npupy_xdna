#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from npupy_xdna.heuristic.cost_model import CostModel
from npupy_xdna.regions.region import ArraySpec, Region

_ROOT = Path(__file__).resolve().parent.parent
_TIMINGS = _ROOT / "results" / "timings"
_OUT_DIR = _ROOT / "results" / "03_heuristic_visualizations"
_PLOT_PATH = _OUT_DIR / "07_cost_model_validation.png"
_EVIDENCE_PATH = _ROOT / ".sisyphus" / "evidence" / "task-v2-17-validation.txt"

_DPI = 150
_FIGSIZE = (8, 7)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _arr(shape: tuple[int, ...], dtype: str = "int16") -> ArraySpec:
    return ArraySpec(shape=shape, dtype=dtype)


def collect_gemm_fusion(model: CostModel) -> list[tuple[float, float, str]]:
    points: list[tuple[float, float, str]] = []
    for rec in _load_jsonl(_TIMINGS / "gemm_fusion.jsonl"):
        M, K, N = rec["shape"]
        epilogue = rec.get("epilogue", "none")
        region = Region(
            op="matmul",
            inputs=[_arr((M, K)), _arr((K, N))],
            output=_arr((M, N)),
        )
        est = model.predict("gemm_fusion", region)
        if est is not None:
            label = f"gemm {M}³ ({epilogue})"
            points.append((est.predicted_latency_us, rec["npu_median_us"], label))
    return points


def collect_col_indep(model: CostModel) -> list[tuple[float, float, str]]:
    points: list[tuple[float, float, str]] = []
    for rec in _load_jsonl(_TIMINGS / "col_indep.jsonl"):
        size = int(rec["size"])
        region = Region(
            op="elementwise_binary",
            inputs=[_arr((size,)), _arr((size,))],
            output=_arr((size,)),
        )
        est = model.predict("col_independent", region)
        if est is not None:
            points.append((est.predicted_latency_us, rec["npu_median_us"], f"col_indep {size:,}"))
    return points


def collect_cgra(model: CostModel) -> list[tuple[float, float, str]]:
    points: list[tuple[float, float, str]] = []
    for rec in _load_jsonl(_TIMINGS / "cgra_depth_sweep.jsonl"):
        depth = int(rec["depth"])
        n_elems = int(rec["n_elements"])
        region = Region(
            op="chained_elementwise",
            # chained_elementwise is used so len(inputs) encodes depth; CostModel reads depth = len(region.inputs)
            inputs=[_arr((n_elems,)) for _ in range(depth)],
            output=_arr((n_elems,)),
        )
        est = model.predict("cgra", region)
        if est is not None:
            points.append(
                (est.predicted_latency_us, rec["npu_median_us"], f"cgra depth={depth}")
            )
    return points


def collect_tanh(model: CostModel) -> list[tuple[float, float, str]]:
    points: list[tuple[float, float, str]] = []
    for rec in _load_jsonl(_TIMINGS / "tanh.jsonl"):
        size = int(rec["size"])
        region = Region(
            op="elementwise_unary",
            inputs=[_arr((size,))],
            output=_arr((size,)),
        )
        est = model.predict("tanh", region)
        if est is not None:
            points.append((est.predicted_latency_us, rec["npu_median_us"], f"tanh {size:,}"))
    return points


def r2_score(predicted: np.ndarray, measured: np.ndarray) -> float:
    ss_res = float(np.sum((measured - predicted) ** 2))
    ss_tot = float(np.sum((measured - np.mean(measured)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


_PALETTE: dict[str, str] = {
    "gemm_fusion": "#1f77b4",
    "col_indep":   "#ff7f0e",
    "cgra":        "#2ca02c",
    "tanh":        "#d62728",
}


def plot_validation(
    all_points: dict[str, list[tuple[float, float, str]]],
    r2: float,
    out: Path,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    all_pred: list[float] = []
    all_meas: list[float] = []
    for pts in all_points.values():
        for pred, meas, _ in pts:
            all_pred.append(pred)
            all_meas.append(meas)

    lo = min(min(all_pred), min(all_meas)) * 0.6
    hi = max(max(all_pred), max(all_meas)) * 1.6

    fig, ax = plt.subplots(figsize=_FIGSIZE)

    for template, pts in all_points.items():
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, label=template, color=_PALETTE.get(template), s=60, alpha=0.85, zorder=3)

    diag = np.array([lo, hi])
    ax.plot(diag, diag, "k--", lw=1.2, label="y = x  (perfect)", zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted latency (µs)", fontsize=12)
    ax.set_ylabel("Measured latency (µs)", fontsize=12)
    ax.set_title(
        f"Cost model validation — predicted vs measured\nR² = {r2:.4f}",
        fontsize=13,
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    print(f"[plot] {out}")


def main() -> None:
    model = CostModel()

    all_points: dict[str, list[tuple[float, float, str]]] = {
        "gemm_fusion": collect_gemm_fusion(model),
        "col_indep":   collect_col_indep(model),
        "cgra":        collect_cgra(model),
        "tanh":        collect_tanh(model),
    }

    all_pred: list[float] = []
    all_meas: list[float] = []
    for pts in all_points.values():
        for pred, meas, _ in pts:
            all_pred.append(pred)
            all_meas.append(meas)

    predicted = np.array(all_pred, dtype=float)
    measured  = np.array(all_meas, dtype=float)
    r2 = r2_score(predicted, measured)

    n_total = len(predicted)
    print(f"Total data points: {n_total}")
    for tmpl, pts in all_points.items():
        print(f"  {tmpl}: {len(pts)}")
    print(f"R² = {r2:.4f}")

    plot_validation(all_points, r2, _PLOT_PATH)

    _EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "Task V2-17: Cost model validation\n",
        "=" * 55 + "\n",
        f"Total data points: {n_total}\n",
    ]
    for tmpl, pts in all_points.items():
        lines.append(f"  {tmpl}: {len(pts)} data points\n")
    lines += [
        "\n",
        f"R² score : {r2:.4f}\n",
        f"Target   : R² >= 0.70\n",
        f"Result   : {'PASS' if r2 >= 0.7 else 'FAIL'}\n",
        "\n",
        f"Plot saved to: {_PLOT_PATH}\n",
        "\n",
        "Per-point breakdown:\n",
        "-" * 55 + "\n",
    ]
    for tmpl, pts in all_points.items():
        lines.append(f"\n[{tmpl}]\n")
        for pred, meas, label in pts:
            err_pct = (pred - meas) / meas * 100.0
            lines.append(
                f"  {label:<30s}  pred={pred:8.1f}µs  meas={meas:8.1f}µs  "
                f"err={err_pct:+6.1f}%\n"
            )

    with open(_EVIDENCE_PATH, "w") as fh:
        fh.writelines(lines)
    print(f"[evidence] {_EVIDENCE_PATH}")


if __name__ == "__main__":
    main()
