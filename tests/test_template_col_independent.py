from __future__ import annotations

import numpy as np
import pytest

from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.templates.col_independent import ColIndependentTemplate
from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES


@pytest.fixture
def tmpl():
    return ColIndependentTemplate()


class TestMatch:
    def test_unary_int16_supported_size(self, tmpl):
        r = Region(
            op="elementwise_unary",
            inputs=[ArraySpec((16384,), "int16")],
            output=ArraySpec((16384,), "int16"),
        )
        assert tmpl.match(r)

    def test_binary_int16_supported_size(self, tmpl):
        r = Region(
            op="elementwise_binary",
            inputs=[ArraySpec((16384,), "int16"), ArraySpec((16384,), "int16")],
            output=ArraySpec((16384,), "int16"),
        )
        assert tmpl.match(r)

    def test_all_supported_sizes_match(self, tmpl):
        for total in SUPPORTED_SHAPES["col_indep"]:
            r = Region(
                op="elementwise_unary",
                inputs=[ArraySpec((total,), "int16")],
                output=ArraySpec((total,), "int16"),
            )
            assert tmpl.match(r), f"should match total={total}"

    def test_no_match_unsupported_size(self, tmpl):
        r = Region(
            op="elementwise_unary",
            inputs=[ArraySpec((1024,), "int16")],
            output=ArraySpec((1024,), "int16"),
        )
        assert not tmpl.match(r)

    def test_no_match_wrong_op(self, tmpl):
        r = Region(
            op="matmul",
            inputs=[
                ArraySpec((128, 128), "int16"),
                ArraySpec((128, 128), "int16"),
            ],
            output=ArraySpec((128, 128), "int16"),
        )
        assert not tmpl.match(r)

    def test_no_match_chained_elementwise(self, tmpl):
        r = Region(
            op="chained_elementwise",
            inputs=[ArraySpec((16384,), "int16")],
            output=ArraySpec((16384,), "int16"),
        )
        assert not tmpl.match(r)


class TestConfigSpace:
    def test_n_cores_32_for_all_sizes(self, tmpl):
        for total in SUPPORTED_SHAPES["col_indep"]:
            r = Region(
                op="elementwise_unary",
                inputs=[ArraySpec((total,), "int16")],
                output=ArraySpec((total,), "int16"),
            )
            configs = tmpl.config_space(r)
            assert len(configs) == 1
            assert configs[0].n_cores == 32

    def test_chunk_per_pool_is_total_div_8(self, tmpl):
        for total in SUPPORTED_SHAPES["col_indep"]:
            r = Region(
                op="elementwise_unary",
                inputs=[ArraySpec((total,), "int16")],
                output=ArraySpec((total,), "int16"),
            )
            configs = tmpl.config_space(r)
            assert configs[0].extra["chunk_per_pool"] == total // 8

    def test_tile_encodes_total_elements(self, tmpl):
        total = 65536
        r = Region(
            op="elementwise_unary",
            inputs=[ArraySpec((total,), "int16")],
            output=ArraySpec((total,), "int16"),
        )
        configs = tmpl.config_space(r)
        assert configs[0].tile == (total,)


class TestLowerReturnType:
    def test_lower_unary_returns_callable(self, tmpl):
        total = 16384
        r = Region(
            op="elementwise_unary",
            inputs=[ArraySpec((total,), "int16")],
            output=ArraySpec((total,), "int16"),
        )
        cfg = tmpl.config_space(r)[0]
        fn = tmpl.lower(r, cfg)
        assert callable(fn)

    def test_lower_binary_returns_callable(self, tmpl):
        total = 16384
        r = Region(
            op="elementwise_binary",
            inputs=[ArraySpec((total,), "int16"), ArraySpec((total,), "int16")],
            output=ArraySpec((total,), "int16"),
        )
        cfg = tmpl.config_space(r)[0]
        fn = tmpl.lower(r, cfg)
        assert callable(fn)

    def test_lower_unsupported_op_raises(self, tmpl):
        r = Region(
            op="chained_elementwise",
            inputs=[ArraySpec((16384,), "int16")],
            output=ArraySpec((16384,), "int16"),
        )
        cfg = tmpl.config_space(r)[0]
        with pytest.raises(ValueError, match="unsupported op"):
            tmpl.lower(r, cfg)


class TestEstimatedCost:
    def test_latency_positive(self, tmpl):
        r = Region(
            op="elementwise_unary",
            inputs=[ArraySpec((16384,), "int16")],
            output=ArraySpec((16384,), "int16"),
        )
        cfg = tmpl.config_space(r)[0]
        cost = tmpl.estimated_cost(r, cfg)
        assert cost.predicted_latency_us > 0
        assert cost.predicted_gops > 0
        assert 0.0 <= cost.confidence <= 1.0

    def test_binary_costs_more_than_unary(self, tmpl):
        total = 16384
        r_unary = Region(
            op="elementwise_unary",
            inputs=[ArraySpec((total,), "int16")],
            output=ArraySpec((total,), "int16"),
        )
        r_binary = Region(
            op="elementwise_binary",
            inputs=[ArraySpec((total,), "int16"), ArraySpec((total,), "int16")],
            output=ArraySpec((total,), "int16"),
        )
        cfg_u = tmpl.config_space(r_unary)[0]
        cfg_b = tmpl.config_space(r_binary)[0]
        cost_u = tmpl.estimated_cost(r_unary, cfg_u)
        cost_b = tmpl.estimated_cost(r_binary, cfg_b)
        assert cost_b.predicted_latency_us > cost_u.predicted_latency_us


@pytest.mark.npu
class TestNpuCorrectnessRelu:
    def test_relu_smallest_size_bit_exact(self, tmpl):
        from npupy_xdna.runtime.npu_runner import NpuRunner
        from npupy_xdna.runtime.preflight import preflight_check
        from pathlib import Path

        total = 16384
        r = Region(
            op="elementwise_unary",
            inputs=[ArraySpec((total,), "int16")],
            output=ArraySpec((total,), "int16"),
        )
        cfg = tmpl.config_space(r)[0]
        iron_fn = tmpl.lower(r, cfg)

        rng = np.random.default_rng(42)
        inp = rng.integers(-100, 100, size=(total,), dtype=np.int16)

        runner = NpuRunner()
        result = runner.run(r, cfg, iron_fn, [inp])

        assert result.status == "ok", f"NPU run failed: {result.status}"
        expected = np.maximum(0, inp)
        np.testing.assert_array_equal(result.output, expected)

    def test_relu_all_positive_unchanged(self, tmpl):
        from npupy_xdna.runtime.npu_runner import NpuRunner

        total = 16384
        r = Region(
            op="elementwise_unary",
            inputs=[ArraySpec((total,), "int16")],
            output=ArraySpec((total,), "int16"),
        )
        cfg = tmpl.config_space(r)[0]
        iron_fn = tmpl.lower(r, cfg)

        inp = np.ones(total, dtype=np.int16) * 42
        runner = NpuRunner()
        result = runner.run(r, cfg, iron_fn, [inp])

        assert result.status == "ok"
        np.testing.assert_array_equal(result.output, inp)

    def test_relu_all_negative_zeroed(self, tmpl):
        from npupy_xdna.runtime.npu_runner import NpuRunner

        total = 16384
        r = Region(
            op="elementwise_unary",
            inputs=[ArraySpec((total,), "int16")],
            output=ArraySpec((total,), "int16"),
        )
        cfg = tmpl.config_space(r)[0]
        iron_fn = tmpl.lower(r, cfg)

        inp = np.full(total, -7, dtype=np.int16)
        runner = NpuRunner()
        result = runner.run(r, cfg, iron_fn, [inp])

        assert result.status == "ok"
        np.testing.assert_array_equal(result.output, np.zeros(total, dtype=np.int16))
