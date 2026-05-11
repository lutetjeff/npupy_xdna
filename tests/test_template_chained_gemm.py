"""Tests for ChainedGemmTemplate.

The template is kill-switched in V2 (`ENABLED=False`); the NPU correctness
test below is therefore `pytest.mark.skip`'d but kept in the suite so that it
can be flipped on as soon as the underlying IRON limitation (documented in
`npupy_xdna/results/chained_gemm_kill_switch.md`) is removed.
"""

from __future__ import annotations

import numpy as np
import pytest

from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.templates.chained_gemm import (
    SUPPORTED_CHAIN_SHAPES,
    ChainedGemmTemplate,
)
from npupy_xdna.templates.protocol import CostEstimate, Config


def _make_chain_region(M: int, K1: int, N1: int, N2: int) -> Region:
    return Region(
        op="matmul_fused",
        inputs=[
            ArraySpec(shape=(M, K1), dtype="int16"),
            ArraySpec(shape=(K1, N1), dtype="int16"),
        ],
        output=ArraySpec(shape=(M, N1), dtype="int16"),
        metadata={"chain": (M, K1, N1, N2)},
    )


class TestChainedGemmKillSwitch:
    def test_template_is_disabled(self):
        assert ChainedGemmTemplate.ENABLED is False, (
            "ChainedGemmTemplate must remain disabled until the IRON "
            "ObjectFifo single-consume limitation is resolved. See "
            "npupy_xdna/results/chained_gemm_kill_switch.md."
        )

    def test_match_returns_false_for_plain_matmul(self):
        r = Region(
            op="matmul",
            inputs=[
                ArraySpec(shape=(256, 256), dtype="int16"),
                ArraySpec(shape=(256, 256), dtype="int16"),
            ],
            output=ArraySpec(shape=(256, 256), dtype="int16"),
        )
        assert ChainedGemmTemplate().match(r) is False

    def test_match_returns_false_for_chain_marked_region(self):
        r = _make_chain_region(256, 256, 256, 256)
        assert ChainedGemmTemplate().match(r) is False

    def test_match_returns_false_for_every_supported_chain_shape(self):
        tmpl = ChainedGemmTemplate()
        for (M, K1, N1, N2) in SUPPORTED_CHAIN_SHAPES:
            r = _make_chain_region(M, K1, N1, N2)
            assert tmpl.match(r) is False, (
                f"Expected kill-switched match=False for {M=} {K1=} {N1=} {N2=}"
            )

    def test_config_space_is_non_empty_for_reporting(self):
        r = _make_chain_region(256, 256, 256, 256)
        configs = ChainedGemmTemplate().config_space(r)
        assert len(configs) >= 1
        assert all(isinstance(c, Config) for c in configs)
        assert all(c.extra.get("intermediate_in_memtile") is True for c in configs)
        assert all(c.extra.get("enabled") is False for c in configs)

    def test_estimated_cost_has_zero_confidence(self):
        r = _make_chain_region(256, 256, 256, 256)
        tmpl = ChainedGemmTemplate()
        cost = tmpl.estimated_cost(r, tmpl.config_space(r)[0])
        assert isinstance(cost, CostEstimate)
        assert cost.confidence == 0.0, (
            "Cost-model confidence must be 0 while the template is kill-switched"
        )

    def test_lower_raises_not_implemented(self):
        r = _make_chain_region(256, 256, 256, 256)
        tmpl = ChainedGemmTemplate()
        with pytest.raises(NotImplementedError):
            tmpl.lower(r, tmpl.config_space(r)[0])


@pytest.mark.skip(
    reason=(
        "ChainedGemmTemplate is kill-switched in V2 (see "
        "npupy_xdna/results/chained_gemm_kill_switch.md). This NPU "
        "correctness test will be activated when ENABLED flips to True."
    )
)
@pytest.mark.npu
class TestChainedGemmCorrectness:
    """Bit-exact correctness for `D = (A @ B) @ C` at 256³ — currently skipped."""

    def test_chain_256_cubed_bit_exact(self):
        from npupy_xdna.runtime.npu_runner import NpuRunner  # noqa: F401

        M, K1, N1, N2 = 256, 256, 256, 256
        rng = np.random.default_rng(20260511)
        A = rng.integers(-3, 4, size=(M, K1), dtype=np.int16)
        B = rng.integers(-3, 4, size=(K1, N1), dtype=np.int16)
        C = rng.integers(-3, 4, size=(N1, N2), dtype=np.int16)

        T_ref = np.matmul(A.astype(np.int32), B.astype(np.int32))
        D_ref_i32 = np.matmul(T_ref, C.astype(np.int32))
        D_ref = np.clip(D_ref_i32, -32768, 32767).astype(np.int16)

        region = _make_chain_region(M, K1, N1, N2)
        tmpl = ChainedGemmTemplate()
        config = tmpl.config_space(region)[0]
        fn = tmpl.lower(region, config)

        D = np.zeros((M, N2), dtype=np.int16)
        fn(A, B, C, D)

        np.testing.assert_array_equal(D, D_ref)
