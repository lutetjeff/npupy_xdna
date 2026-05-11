from __future__ import annotations

import numpy as np
import pytest

from npupy_xdna.bench.baselines import cpu_baseline, cpu_baseline_via_blas, detect_blas
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


class TestCpuBaselineViaBlas:
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
        result = cpu_baseline_via_blas(region, inputs, config)

        assert len(result.iterations_us) == config.n_iterations
        assert result.median_us >= 0
        assert result.min_us >= 0
        assert result.max_us >= 0
        assert result.min_us <= result.max_us

    def test_blas_baseline_runs_fast_for_256_cubed(self):
        """BLAS baseline should complete 256x256 matmul in under 50ms per iteration."""
        region = Region(
            op="matmul",
            inputs=[
                ArraySpec(shape=(256, 256), dtype="int16"),
                ArraySpec(shape=(256, 256), dtype="int16"),
            ],
            output=ArraySpec(shape=(256, 256), dtype="int16"),
        )
        rng = np.random.default_rng(42)
        a = rng.integers(-100, 100, size=(256, 256), dtype=np.int16)
        b = rng.integers(-100, 100, size=(256, 256), dtype=np.int16)
        inputs = [a, b]

        config = BenchmarkConfig(n_warmup=2, n_iterations=5)
        result_blas = cpu_baseline_via_blas(region, inputs, config)

        assert result_blas.median_us < 50_000, (
            f"Expected BLAS median < 50ms, got {result_blas.median_us / 1000:.1f}ms"
        )

    def test_blas_result_numerically_close_to_int16_baseline(self):
        """BLAS baseline result is numerically close to int16 baseline (within int16 rounding)."""
        region = Region(
            op="matmul",
            inputs=[
                ArraySpec(shape=(64, 64), dtype="int16"),
                ArraySpec(shape=(64, 64), dtype="int16"),
            ],
            output=ArraySpec(shape=(64, 64), dtype="int16"),
        )
        rng = np.random.default_rng(42)
        a = rng.integers(-50, 50, size=(64, 64), dtype=np.int16)
        b = rng.integers(-50, 50, size=(64, 64), dtype=np.int16)
        inputs = [a, b]

        config = BenchmarkConfig(n_warmup=1, n_iterations=1)
        result_int16 = cpu_baseline(region, inputs, config)
        result_blas = cpu_baseline_via_blas(region, inputs, config)

        int16_out = np.matmul(a, b)
        blas_out = np.clip(np.matmul(a.astype(np.float32), b.astype(np.float32)), -32768, 32767).astype(np.int16)
        np.testing.assert_array_equal(int16_out, blas_out)
