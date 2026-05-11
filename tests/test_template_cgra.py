from __future__ import annotations

import numpy as np
import pytest

from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.templates.cgra import CgraTemplate

_N = 256
_RNG = np.random.default_rng(42)


def _make_region() -> Region:
    spec = ArraySpec(shape=(_N,), dtype="int16")
    return Region(
        op="chained_elementwise",
        inputs=[spec, spec, spec, spec],
        output=spec,
    )


def _make_inputs():
    return [_RNG.integers(-5, 5, size=_N, dtype=np.int16) for _ in range(4)]


def _reference(a, b, c, d) -> np.ndarray:
    a32 = a.astype(np.int32)
    b32 = b.astype(np.int32)
    c32 = c.astype(np.int32)
    d32 = d.astype(np.int32)
    result = (a32 + b32) * c32 - d32
    return np.clip(result, -32768, 32767).astype(np.int16)


class TestCgraTemplateMatch:
    def test_matches_chained_elementwise_4input_256_int16(self):
        tmpl = CgraTemplate()
        region = _make_region()
        assert tmpl.match(region)

    def test_no_match_wrong_op(self):
        tmpl = CgraTemplate()
        region = Region(
            op="elementwise_binary",
            inputs=[ArraySpec(shape=(_N,), dtype="int16"), ArraySpec(shape=(_N,), dtype="int16")],
            output=ArraySpec(shape=(_N,), dtype="int16"),
        )
        assert not tmpl.match(region)

    def test_no_match_wrong_shape(self):
        tmpl = CgraTemplate()
        spec = ArraySpec(shape=(128,), dtype="int16")
        region = Region(
            op="chained_elementwise",
            inputs=[spec, spec, spec, spec],
            output=spec,
        )
        assert not tmpl.match(region)

    def test_no_match_wrong_input_count(self):
        tmpl = CgraTemplate()
        spec = ArraySpec(shape=(_N,), dtype="int16")
        region = Region(
            op="chained_elementwise",
            inputs=[spec, spec, spec],
            output=spec,
        )
        assert not tmpl.match(region)


class TestCgraConfigSpace:
    def test_returns_one_config(self):
        tmpl = CgraTemplate()
        region = _make_region()
        configs = tmpl.config_space(region)
        assert len(configs) == 1
        assert configs[0].n_cores == 3


@pytest.mark.npu
class TestCgraNpuCorrectness:
    def test_add_mul_sub_pipeline_bit_exact(self):
        import aie.iron as iron

        tmpl = CgraTemplate()
        region = _make_region()
        config = tmpl.config_space(region)[0]
        cgra_fn = tmpl.lower(region, config)

        a, b, c, d = _make_inputs()
        expected = _reference(a, b, c, d)

        a_npu = iron.tensor(a, dtype=np.int16, device="npu")
        b_npu = iron.tensor(b, dtype=np.int16, device="npu")
        c_npu = iron.tensor(c, dtype=np.int16, device="npu")
        d_npu = iron.tensor(d, dtype=np.int16, device="npu")
        out_npu = iron.zeros(_N, dtype=np.int16, device="npu")

        cgra_fn(a_npu, b_npu, c_npu, d_npu, out_npu)

        result = np.array(out_npu.numpy(), copy=True)
        np.testing.assert_array_equal(result, expected)
