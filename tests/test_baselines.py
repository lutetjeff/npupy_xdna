from __future__ import annotations

import numpy as np
import pytest

from npupy_xdna.bench.baselines import cpu_baseline, detect_blas
from npupy_xdna.bench.timer import BenchmarkConfig
from npupy_xdna.regions.region import ArraySpec, Region


class TestDetectBlas:
    def test_returns_non_empty_string(self):
        result = detect_blas()
        assert isinstance(result, str)
        assert result != ""
        assert result in {"openblas", "mkl", "blis", "unknown"}

    def test_memoized(self):
        first = detect_blas()
        second = detect_blas()
        assert first == second


class TestCpuBaseline:
    def test_returns_benchmark_result_with_correct_iteration_count(self):
        region = Region(
            op="matmul",
            inputs=[
                ArraySpec(shape=(8, 8), dtype="int16"),
                ArraySpec(shape=(8, 8), dtype="int16"),
            ],
            output=ArraySpec(shape=(8, 8), dtype="int16"),
        )
        rng = np.random.default_rng(42)
        a = rng.integers(0, 10, size=(8, 8), dtype=np.int16)
        b = rng.integers(0, 10, size=(8, 8), dtype=np.int16)
        inputs = [a, b]

        config = BenchmarkConfig(n_warmup=2, n_iterations=5)
        result = cpu_baseline(region, inputs, config)

        assert len(result.iterations_us) == config.n_iterations
        assert result.median_us >= 0
        assert result.min_us >= 0
        assert result.max_us >= 0
        assert result.min_us <= result.max_us

    def test_elementwise_unary_baseline(self):
        region = Region(
            op="elementwise_unary",
            inputs=[ArraySpec(shape=(16,), dtype="int16")],
            output=ArraySpec(shape=(16,), dtype="int16"),
        )
        rng = np.random.default_rng(42)
        inp = rng.integers(-5, 5, size=(16,), dtype=np.int16)

        config = BenchmarkConfig(n_warmup=1, n_iterations=3)
        result = cpu_baseline(region, [inp], config)

        assert len(result.iterations_us) == config.n_iterations

    def test_matmul_fused_baseline(self):
        region = Region(
            op="matmul_fused",
            inputs=[
                ArraySpec(shape=(4, 4), dtype="int16"),
                ArraySpec(shape=(4, 4), dtype="int16"),
            ],
            output=ArraySpec(shape=(4, 4), dtype="int16"),
        )
        rng = np.random.default_rng(42)
        a = rng.integers(0, 5, size=(4, 4), dtype=np.int16)
        b = rng.integers(0, 5, size=(4, 4), dtype=np.int16)

        config = BenchmarkConfig(n_warmup=1, n_iterations=3)
        result = cpu_baseline(region, [a, b], config)

        assert len(result.iterations_us) == config.n_iterations
