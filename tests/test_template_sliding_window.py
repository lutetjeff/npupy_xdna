from __future__ import annotations

import math

import numpy as np
import pytest

from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES
from npupy_xdna.templates.sliding_window import SlidingWindowTemplate


@pytest.fixture
def tmpl():
    return SlidingWindowTemplate()


def _stencil_region(H: int, W: int) -> Region:
    return Region(
        op="stencil_2d",
        inputs=[ArraySpec((H, W), "int16")],
        output=ArraySpec((H, W), "int16"),
        metadata={"stencil": "5pt", "iterations": 1},
    )


class TestMatch:
    def test_matches_64x64(self, tmpl):
        assert tmpl.match(_stencil_region(64, 64))

    def test_matches_128x128(self, tmpl):
        assert tmpl.match(_stencil_region(128, 128))

    def test_matches_256x256(self, tmpl):
        assert tmpl.match(_stencil_region(256, 256))

    def test_all_supported_shapes_match(self, tmpl):
        for H, W in SUPPORTED_SHAPES["sliding_window"]:
            assert tmpl.match(_stencil_region(H, W)), f"should match ({H},{W})"

    def test_no_match_wrong_op(self, tmpl):
        r = Region(
            op="elementwise_unary",
            inputs=[ArraySpec((16384,), "int16")],
            output=ArraySpec((16384,), "int16"),
        )
        assert not tmpl.match(r)

    def test_no_match_matmul(self, tmpl):
        r = Region(
            op="matmul",
            inputs=[ArraySpec((64, 64), "int16"), ArraySpec((64, 64), "int16")],
            output=ArraySpec((64, 64), "int16"),
        )
        assert not tmpl.match(r)

    def test_no_match_unsupported_shape(self, tmpl):
        r = Region(
            op="stencil_2d",
            inputs=[ArraySpec((32, 32), "int16")],
            output=ArraySpec((32, 32), "int16"),
        )
        assert not tmpl.match(r)

    def test_no_match_non_square_unsupported(self, tmpl):
        r = Region(
            op="stencil_2d",
            inputs=[ArraySpec((64, 128), "int16")],
            output=ArraySpec((64, 128), "int16"),
        )
        assert not tmpl.match(r)


class TestConfigSpace:
    def test_single_config_for_64x64(self, tmpl):
        configs = tmpl.config_space(_stencil_region(64, 64))
        assert len(configs) == 1

    def test_n_cores_is_8(self, tmpl):
        for H, W in SUPPORTED_SHAPES["sliding_window"]:
            configs = tmpl.config_space(_stencil_region(H, W))
            assert configs[0].n_cores == 8

    def test_strip_h_is_h_div_8(self, tmpl):
        for H, W in SUPPORTED_SHAPES["sliding_window"]:
            configs = tmpl.config_space(_stencil_region(H, W))
            assert configs[0].extra["strip_h"] == H // 8

    def test_tile_encodes_grid_shape(self, tmpl):
        configs = tmpl.config_space(_stencil_region(64, 64))
        assert configs[0].tile == (64, 64)

    def test_halo_is_1(self, tmpl):
        for H, W in SUPPORTED_SHAPES["sliding_window"]:
            configs = tmpl.config_space(_stencil_region(H, W))
            assert configs[0].extra["halo"] == 1


class TestLowerReturnType:
    def test_returns_callable(self, tmpl):
        r = _stencil_region(64, 64)
        cfg = tmpl.config_space(r)[0]
        fn = tmpl.lower(r, cfg)
        assert callable(fn)

    def test_returns_callable_128x128(self, tmpl):
        r = _stencil_region(128, 128)
        cfg = tmpl.config_space(r)[0]
        fn = tmpl.lower(r, cfg)
        assert callable(fn)


class TestEstimatedCost:
    def test_latency_positive(self, tmpl):
        r = _stencil_region(64, 64)
        cfg = tmpl.config_space(r)[0]
        cost = tmpl.estimated_cost(r, cfg)
        assert cost.predicted_latency_us > 0

    def test_gops_positive(self, tmpl):
        r = _stencil_region(64, 64)
        cfg = tmpl.config_space(r)[0]
        cost = tmpl.estimated_cost(r, cfg)
        assert cost.predicted_gops > 0

    def test_confidence_in_range(self, tmpl):
        for H, W in SUPPORTED_SHAPES["sliding_window"]:
            r = _stencil_region(H, W)
            cfg = tmpl.config_space(r)[0]
            cost = tmpl.estimated_cost(r, cfg)
            assert 0.0 <= cost.confidence <= 1.0

    def test_larger_grid_higher_latency(self, tmpl):
        r64 = _stencil_region(64, 64)
        r256 = _stencil_region(256, 256)
        cfg64 = tmpl.config_space(r64)[0]
        cfg256 = tmpl.config_space(r256)[0]
        cost64 = tmpl.estimated_cost(r64, cfg64)
        cost256 = tmpl.estimated_cost(r256, cfg256)
        assert cost256.predicted_latency_us >= cost64.predicted_latency_us


class TestRegionValidation:
    def test_stencil_2d_region_created(self):
        r = _stencil_region(64, 64)
        assert r.op == "stencil_2d"
        assert r.output.shape == (64, 64)

    def test_stencil_2d_requires_1_input(self):
        with pytest.raises(ValueError, match="exactly 1 input"):
            Region(
                op="stencil_2d",
                inputs=[
                    ArraySpec((64, 64), "int16"),
                    ArraySpec((64, 64), "int16"),
                ],
                output=ArraySpec((64, 64), "int16"),
            )

    def test_stencil_2d_requires_2d_input(self):
        with pytest.raises(ValueError, match="2D"):
            Region(
                op="stencil_2d",
                inputs=[ArraySpec((4096,), "int16")],
                output=ArraySpec((4096,), "int16"),
            )

    def test_stencil_2d_output_shape_must_match_input(self):
        with pytest.raises(ValueError, match="must match"):
            Region(
                op="stencil_2d",
                inputs=[ArraySpec((64, 64), "int16")],
                output=ArraySpec((32, 128), "int16"),
            )


def _numpy_stencil_ref(inp: np.ndarray) -> np.ndarray:
    H, W = inp.shape
    out = np.zeros((H, W), dtype=np.int16)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            s = (
                int(inp[i, j])
                + int(inp[i - 1, j])
                + int(inp[i + 1, j])
                + int(inp[i, j - 1])
                + int(inp[i, j + 1])
            )
            v = int(s / 5)
            if v > 32767:
                v = 32767
            if v < -32768:
                v = -32768
            out[i, j] = np.int16(v)
    return out


@pytest.mark.npu
class TestStencilCorrectness:
    def test_stencil_64x64_interior_bit_exact(self, tmpl):
        rng = np.random.default_rng(42)
        inp = rng.integers(-1000, 1000, size=(64, 64), dtype=np.int16)
        ref = _numpy_stencil_ref(inp)

        r = _stencil_region(64, 64)
        cfg = tmpl.config_space(r)[0]
        run = tmpl.lower(r, cfg)

        out = np.zeros((64, 64), dtype=np.int16)
        run(inp, out)

        np.testing.assert_array_equal(
            out[1:-1, 1:-1],
            ref[1:-1, 1:-1],
            err_msg="NPU stencil interior points do not match numpy reference",
        )

    def test_stencil_zeros_input_produces_zeros(self, tmpl):
        inp = np.zeros((64, 64), dtype=np.int16)
        r = _stencil_region(64, 64)
        cfg = tmpl.config_space(r)[0]
        run = tmpl.lower(r, cfg)

        out = np.ones((64, 64), dtype=np.int16)
        run(inp, out)

        np.testing.assert_array_equal(out, np.zeros((64, 64), dtype=np.int16))
