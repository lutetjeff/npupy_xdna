from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from npupy_xdna.heuristic.viz import (
    plot_bandwidth_scaling,
    plot_crossover,
    plot_decision_map,
    plot_speedup_bars,
    plot_throughput_curve,
)

_MIN_PNG_BYTES = 10_000


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


class TestPlotThroughputCurve:
    def test_produces_png_no_groups(self, tmp_path):
        data = [{"size": s, "gops": s * 0.5} for s in range(64, 1025, 64)]
        jsonl = tmp_path / "data.jsonl"
        _write_jsonl(jsonl, data)
        out = tmp_path / "throughput.png"
        plot_throughput_curve(jsonl, out, "Throughput", "size", "gops")
        assert out.exists()
        assert out.stat().st_size >= _MIN_PNG_BYTES

    def test_produces_png_with_groups(self, tmp_path):
        data = [
            {"size": s, "gops": s * factor, "template": tmpl}
            for s in [64, 128, 256, 512]
            for tmpl, factor in [("A", 0.5), ("B", 0.8)]
        ]
        jsonl = tmp_path / "data.jsonl"
        _write_jsonl(jsonl, data)
        out = tmp_path / "throughput_grouped.png"
        plot_throughput_curve(jsonl, out, "Grouped Throughput", "size", "gops", group_key="template")
        assert out.exists()
        assert out.stat().st_size >= _MIN_PNG_BYTES


class TestPlotCrossover:
    def test_produces_png_with_crossover(self, tmp_path):
        npu = [{"size": s, "throughput_gops": 0.01 * s} for s in range(64, 1025, 64)]
        cpu = [{"size": s, "throughput_gops": 4.0 + 0.001 * s} for s in range(64, 1025, 64)]
        out = tmp_path / "crossover.png"
        plot_crossover(npu, cpu, out, "NPU vs CPU Crossover")
        assert out.exists()
        assert out.stat().st_size >= _MIN_PNG_BYTES

    def test_produces_png_no_crossover(self, tmp_path):
        npu = [{"size": s, "throughput_gops": 10.0} for s in range(64, 513, 64)]
        cpu = [{"size": s, "throughput_gops": 2.0} for s in range(64, 513, 64)]
        out = tmp_path / "crossover_none.png"
        plot_crossover(npu, cpu, out, "No Crossover")
        assert out.exists()
        assert out.stat().st_size >= _MIN_PNG_BYTES


class TestPlotDecisionMap:
    def test_produces_png_with_data(self, tmp_path):
        templates = ["gemm_4col", "gemm_8col", "conv_pool"]
        decisions = {
            (m, k): templates[(m // 64 + k // 64) % len(templates)]
            for m in range(64, 513, 64)
            for k in range(64, 513, 64)
        }
        out = tmp_path / "decision_map.png"
        plot_decision_map(decisions, out, "Template Decision Map")
        assert out.exists()
        assert out.stat().st_size >= _MIN_PNG_BYTES

    def test_produces_png_empty(self, tmp_path):
        out = tmp_path / "decision_map_empty.png"
        plot_decision_map({}, out, "Empty Decision Map")
        assert out.exists()


class TestPlotSpeedupBars:
    def test_produces_png(self, tmp_path):
        records = [
            {"benchmark": f"bench_{i}", "speedup": 0.5 + i * 0.3}
            for i in range(8)
        ]
        out = tmp_path / "speedup.png"
        plot_speedup_bars(records, out, "Speedup per Benchmark")
        assert out.exists()
        assert out.stat().st_size >= _MIN_PNG_BYTES

    def test_custom_keys(self, tmp_path):
        records = [{"name": "add", "ratio": 2.5}, {"name": "mul", "ratio": 0.9}]
        out = tmp_path / "speedup_custom.png"
        plot_speedup_bars(records, out, "Custom Keys", name_key="name", speedup_key="ratio")
        assert out.exists()
        assert out.stat().st_size >= _MIN_PNG_BYTES


class TestPlotBandwidthScaling:
    def test_produces_png_no_groups(self, tmp_path):
        data = [{"size_bytes": s, "bandwidth_gbs": s / 1e9 * 100} for s in [1024, 4096, 16384, 65536]]
        jsonl = tmp_path / "bw.jsonl"
        _write_jsonl(jsonl, data)
        out = tmp_path / "bandwidth.png"
        plot_bandwidth_scaling(jsonl, out, "Bandwidth Scaling")
        assert out.exists()
        assert out.stat().st_size >= _MIN_PNG_BYTES

    def test_produces_png_with_groups(self, tmp_path):
        sizes = [1024, 4096, 16384, 65536]
        data = [
            {"size_bytes": s, "bandwidth_gbs": s / 1e9 * factor, "tmpl": t}
            for s in sizes
            for t, factor in [("elem_add", 80), ("elem_mul", 60)]
        ]
        jsonl = tmp_path / "bw_grouped.jsonl"
        _write_jsonl(jsonl, data)
        out = tmp_path / "bandwidth_grouped.png"
        plot_bandwidth_scaling(jsonl, out, "Grouped BW", group_key="tmpl")
        assert out.exists()
        assert out.stat().st_size >= _MIN_PNG_BYTES
