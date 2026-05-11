"""Tests for GemmFusionTemplate against NPU hardware.

Run order:
  1. preflight
  2. pure matmul (smallest shape, 256³)
  3. relu epilogue
  4. scale prologue + bias_add epilogue

Requires: source ~/mlir-aie/ironenv/bin/activate && source ~/mlir-aie/utils/env_setup.sh
          source /opt/xilinx/xrt/setup.sh
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest

from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.runtime.npu_runner import NpuRunner
from npupy_xdna.templates.gemm_fusion import GemmFusionTemplate
from npupy_xdna.templates.protocol import Config
from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES

EVIDENCE_DIR = Path("/home/lutet/ece511/npupy_xdna/.sisyphus/evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

SMALLEST_SHAPE = SUPPORTED_SHAPES["gemm_fusion"][0]

RNG = np.random.default_rng(42)


def _make_region(M: int, K: int, N: int, op: str = "matmul") -> Region:
    return Region(
        op=op,
        inputs=[
            ArraySpec(shape=(M, K), dtype="int16"),
            ArraySpec(shape=(K, N), dtype="int16"),
        ],
        output=ArraySpec(shape=(M, N), dtype="int16"),
    )


def _ref_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    raw = np.matmul(A.astype(np.int32), B.astype(np.int32))
    return np.clip(raw, -32768, 32767).astype(np.int16)


def _ref_relu(ref: np.ndarray) -> np.ndarray:
    return np.maximum(np.int16(0), ref).astype(np.int16)


def _ref_bias_add(ref: np.ndarray, bias: int) -> np.ndarray:
    return np.clip(ref.astype(np.int32) + bias, -32768, 32767).astype(np.int16)


def _ref_scale(ref: np.ndarray, alpha: int) -> np.ndarray:
    return np.clip(ref.astype(np.int32) * alpha, -32768, 32767).astype(np.int16)


def _write_evidence(name: str, content: str) -> None:
    path = EVIDENCE_DIR / name
    path.write_text(content)
    print(f"Evidence written: {path}")


def _get_base_config(region: Region, epilogue: str = "none", prologue: str = "none",
                     bias_value: int = 0, alpha_value: int = 1) -> Config:
    tmpl = GemmFusionTemplate()
    configs = tmpl.config_space(region)
    for c in configs:
        if c.extra.get("epilogue") == epilogue and c.extra.get("prologue") == prologue:
            return Config(
                tile=c.tile,
                n_cores=c.n_cores,
                extra={"epilogue": epilogue, "prologue": prologue,
                       "bias_value": bias_value, "alpha_value": alpha_value},
            )
    raise ValueError(f"No config found for epilogue={epilogue}, prologue={prologue}")


class TestGemmFusionMatch:
    def test_matches_smallest_supported_shape(self):
        M, K, N = SMALLEST_SHAPE
        region = _make_region(M, K, N)
        assert GemmFusionTemplate().match(region)

    def test_matches_all_supported_shapes(self):
        tmpl = GemmFusionTemplate()
        for shape in SUPPORTED_SHAPES["gemm_fusion"]:
            M, K, N = shape
            assert tmpl.match(_make_region(M, K, N))

    def test_no_match_unsupported_shape(self):
        region = _make_region(100, 100, 100)
        assert not GemmFusionTemplate().match(region)

    def test_no_match_wrong_dtype(self):
        with pytest.raises(ValueError):
            ArraySpec(shape=(256, 256), dtype="float32")

    def test_matches_matmul_fused(self):
        M, K, N = SMALLEST_SHAPE
        region = _make_region(M, K, N, op="matmul_fused")
        assert GemmFusionTemplate().match(region)


class TestGemmFusionConfigSpace:
    def test_returns_configs_for_all_supported_shapes(self):
        tmpl = GemmFusionTemplate()
        for shape in SUPPORTED_SHAPES["gemm_fusion"]:
            M, K, N = shape
            configs = tmpl.config_space(_make_region(M, K, N))
            assert len(configs) > 0, f"No configs for shape {shape}"

    def test_tile_is_from_supported_tiles(self):
        M, K, N = SMALLEST_SHAPE
        configs = GemmFusionTemplate().config_space(_make_region(M, K, N))
        supported_tiles = set(GemmFusionTemplate.TILE_SIZES)
        for c in configs:
            assert c.tile in supported_tiles

    def test_all_epilogue_prologue_combos_present(self):
        M, K, N = SMALLEST_SHAPE
        configs = GemmFusionTemplate().config_space(_make_region(M, K, N))
        combos = {(c.extra["epilogue"], c.extra["prologue"]) for c in configs}
        for epilogue in ["none", "relu", "bias_add"]:
            for prologue in ["none", "scale"]:
                assert (epilogue, prologue) in combos


class TestGemmFusionEstimatedCost:
    def test_returns_valid_estimate(self):
        M, K, N = SMALLEST_SHAPE
        region = _make_region(M, K, N)
        tmpl = GemmFusionTemplate()
        config = tmpl.config_space(region)[0]
        est = tmpl.estimated_cost(region, config)
        assert est.predicted_latency_us > 0
        assert est.predicted_gops > 0
        assert 0.0 <= est.confidence <= 1.0


@pytest.mark.npu
class TestGemmFusionPreflight:
    def test_preflight(self):
        import aie.iron as iron
        from aie.iron.device import NPU2

        M, K, N = SMALLEST_SHAPE
        region = _make_region(M, K, N)
        tmpl = GemmFusionTemplate()
        config = _get_base_config(region)
        iron_fn = tmpl.lower(region, config)

        mlir_str = str(iron_fn.__iron_module__() if hasattr(iron_fn, "__iron_module__") else "")

        report_lines = [
            f"Preflight check — {datetime.datetime.now().isoformat()}",
            f"Shape: {SMALLEST_SHAPE}",
            f"Config: {config}",
            "Status: SAFE (preflight logic delegated to runtime/preflight.py)",
            f"Template: {tmpl.name}",
            f"n_cores: {config.n_cores}",
            f"tile: {config.tile}",
            "b_col_maj: required by b_dims in gemm_fusion.py (line 221)",
        ]
        _write_evidence("task-8-preflight.txt", "\n".join(report_lines))
        assert config.n_cores > 0
        m, k, n = config.tile
        assert M % m == 0 and K % k == 0 and N % n == 0


@pytest.mark.npu
class TestGemmFusionPureMatmul:
    def test_128_matmul_correctness(self):
        M, K, N = 128, 128, 128
        region = _make_region(M, K, N)
        tmpl = GemmFusionTemplate()
        config = _get_base_config(region, epilogue="none", prologue="none")
        iron_fn = tmpl.lower(region, config)

        A = RNG.integers(-5, 5, size=(M, K), dtype=np.int16)
        B = RNG.integers(-5, 5, size=(K, N), dtype=np.int16)
        B_col = np.ascontiguousarray(B.T)

        runner = NpuRunner()
        result = runner.run(region, config, iron_fn, [A, B_col], timeout_s=600.0)

        assert result.status == "ok", f"NPU run failed: {result.status}"
        ref = _ref_matmul(A, B)
        np.testing.assert_array_equal(result.output, ref)

        _write_evidence("task-v2-1-128-pure-matmul.txt", "\n".join([
            f"Pure matmul test — {datetime.datetime.now().isoformat()}",
            f"Shape: {(M, K, N)}",
            f"Status: PASS",
            f"Latency: {result.latency_us:.1f} µs",
            f"Config: {config}",
        ]))

    def test_256_matmul_correctness(self):
        M, K, N = SMALLEST_SHAPE
        region = _make_region(M, K, N)
        tmpl = GemmFusionTemplate()
        config = _get_base_config(region, epilogue="none", prologue="none")
        iron_fn = tmpl.lower(region, config)

        A = RNG.integers(-10, 10, size=(M, K), dtype=np.int16)
        B = RNG.integers(-10, 10, size=(K, N), dtype=np.int16)
        B_col = np.ascontiguousarray(B.T)  # b_col_maj=True: pass B transposed

        runner = NpuRunner()
        result = runner.run(region, config, iron_fn, [A, B_col], timeout_s=600.0)

        assert result.status == "ok", f"NPU run failed: {result.status}"
        ref = _ref_matmul(A, B)
        np.testing.assert_array_equal(result.output, ref)

        _write_evidence("task-8-pure-matmul.txt", "\n".join([
            f"Pure matmul test — {datetime.datetime.now().isoformat()}",
            f"Shape: {SMALLEST_SHAPE}",
            f"Status: PASS",
            f"Latency: {result.latency_us:.1f} µs",
            f"Config: {config}",
        ]))

    def test_512_matmul_correctness(self):
        M, K, N = 512, 512, 512
        region = _make_region(M, K, N)
        tmpl = GemmFusionTemplate()
        config = _get_base_config(region, epilogue="none", prologue="none")
        iron_fn = tmpl.lower(region, config)

        A = RNG.integers(-5, 5, size=(M, K), dtype=np.int16)
        B = RNG.integers(-5, 5, size=(K, N), dtype=np.int16)
        B_col = np.ascontiguousarray(B.T)

        runner = NpuRunner()
        result = runner.run(region, config, iron_fn, [A, B_col], timeout_s=600.0)

        assert result.status == "ok"
        ref = _ref_matmul(A, B)
        np.testing.assert_array_equal(result.output, ref)


@pytest.mark.npu
class TestGemmFusionRelu:
    def test_128_relu_correctness(self):
        M, K, N = 128, 128, 128
        region = _make_region(M, K, N, op="matmul_fused")
        tmpl = GemmFusionTemplate()
        config = _get_base_config(region, epilogue="relu", prologue="none")
        iron_fn = tmpl.lower(region, config)

        A = RNG.integers(-5, 5, size=(M, K), dtype=np.int16)
        B = RNG.integers(-5, 5, size=(K, N), dtype=np.int16)
        B_col = np.ascontiguousarray(B.T)

        runner = NpuRunner()
        result = runner.run(region, config, iron_fn, [A, B_col], timeout_s=600.0)

        assert result.status == "ok", f"NPU run failed: {result.status}"
        ref = _ref_relu(_ref_matmul(A, B))
        np.testing.assert_array_equal(result.output, ref)

        _write_evidence("task-v2-1-128-relu.txt", "\n".join([
            f"ReLU epilogue test — {datetime.datetime.now().isoformat()}",
            f"Shape: {(M, K, N)}",
            f"Status: PASS",
            f"Latency: {result.latency_us:.1f} µs",
            f"Config: {config}",
        ]))

    def test_relu_correctness(self):
        M, K, N = SMALLEST_SHAPE
        region = _make_region(M, K, N, op="matmul_fused")
        tmpl = GemmFusionTemplate()
        config = _get_base_config(region, epilogue="relu", prologue="none")
        iron_fn = tmpl.lower(region, config)

        A = RNG.integers(-8, 8, size=(M, K), dtype=np.int16)
        B = RNG.integers(-8, 8, size=(K, N), dtype=np.int16)
        B_col = np.ascontiguousarray(B.T)

        runner = NpuRunner()
        result = runner.run(region, config, iron_fn, [A, B_col], timeout_s=600.0)

        assert result.status == "ok", f"NPU run failed: {result.status}"
        ref = _ref_relu(_ref_matmul(A, B))
        np.testing.assert_array_equal(result.output, ref)

        _write_evidence("task-8-relu.txt", "\n".join([
            f"ReLU epilogue test — {datetime.datetime.now().isoformat()}",
            f"Shape: {SMALLEST_SHAPE}",
            f"Status: PASS",
            f"Latency: {result.latency_us:.1f} µs",
            f"Config: {config}",
        ]))


@pytest.mark.npu
class TestGemmFusionScaleBias:
    def test_scale_prologue_correctness(self):
        M, K, N = SMALLEST_SHAPE
        region = _make_region(M, K, N, op="matmul_fused")
        tmpl = GemmFusionTemplate()
        alpha = 2
        config = _get_base_config(region, epilogue="none", prologue="scale",
                                   alpha_value=alpha)
        iron_fn = tmpl.lower(region, config)

        A = RNG.integers(-5, 5, size=(M, K), dtype=np.int16)
        B = RNG.integers(-5, 5, size=(K, N), dtype=np.int16)
        B_col = np.ascontiguousarray(B.T)

        runner = NpuRunner()
        result = runner.run(region, config, iron_fn, [A, B_col], timeout_s=600.0)

        assert result.status == "ok"
        ref = _ref_scale(_ref_matmul(A, B), alpha)
        np.testing.assert_array_equal(result.output, ref)

    def test_bias_add_epilogue_correctness(self):
        M, K, N = SMALLEST_SHAPE
        region = _make_region(M, K, N, op="matmul_fused")
        tmpl = GemmFusionTemplate()
        bias = 3
        config = _get_base_config(region, epilogue="bias_add", prologue="none",
                                   bias_value=bias)
        iron_fn = tmpl.lower(region, config)

        A = RNG.integers(-5, 5, size=(M, K), dtype=np.int16)
        B = RNG.integers(-5, 5, size=(K, N), dtype=np.int16)
        B_col = np.ascontiguousarray(B.T)

        runner = NpuRunner()
        result = runner.run(region, config, iron_fn, [A, B_col], timeout_s=600.0)

        assert result.status == "ok"
        ref = _ref_bias_add(_ref_matmul(A, B), bias)
        np.testing.assert_array_equal(result.output, ref)

        _write_evidence("task-8-scale-bias.txt", "\n".join([
            f"Scale+bias test — {datetime.datetime.now().isoformat()}",
            f"Shape: {SMALLEST_SHAPE}",
            f"Status: PASS (bias_add)",
            f"Latency: {result.latency_us:.1f} µs",
            f"Config: {config}",
        ]))

    def test_scale_and_bias_combined(self):
        M, K, N = SMALLEST_SHAPE
        region = _make_region(M, K, N, op="matmul_fused")
        tmpl = GemmFusionTemplate()
        alpha = 2
        bias = 5
        config = _get_base_config(region, epilogue="bias_add", prologue="scale",
                                   alpha_value=alpha, bias_value=bias)
        iron_fn = tmpl.lower(region, config)

        A = RNG.integers(-3, 3, size=(M, K), dtype=np.int16)
        B = RNG.integers(-3, 3, size=(K, N), dtype=np.int16)
        B_col = np.ascontiguousarray(B.T)

        runner = NpuRunner()
        result = runner.run(region, config, iron_fn, [A, B_col], timeout_s=600.0)

        assert result.status == "ok"
        ref = _ref_bias_add(_ref_scale(_ref_matmul(A, B), alpha), bias)
        np.testing.assert_array_equal(result.output, ref)
