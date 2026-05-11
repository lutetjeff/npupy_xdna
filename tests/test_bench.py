import json
import os
import tempfile
import time

import numpy as np
import pytest

from npupy_xdna.bench.paths import (
    CHECKPOINTS_DIR,
    EVIDENCE_DIR,
    PLOTS_DIR,
    RESULTS_ROOT,
    TIMINGS_DIR,
    XCLBIN_CACHE_DIR,
    ensure_dirs,
)
from npupy_xdna.bench.seed import make_rng
from npupy_xdna.bench.timer import BenchmarkConfig, BenchmarkResult, Timer, run_benchmark


class TestTimer:
    def test_timer_10ms_sleep_elapsed_in_range(self):
        with Timer() as t:
            time.sleep(0.01)
        assert 9000 <= t.elapsed_us <= 30000


class TestRunBenchmark:
    def test_run_benchmark_returns_correct_iteration_count(self):
        config = BenchmarkConfig(n_warmup=2, n_iterations=7)
        result = run_benchmark(lambda: None, config=config)
        assert len(result.iterations_us) == config.n_iterations

    def test_run_benchmark_warmup_does_not_affect_result_length(self):
        config = BenchmarkConfig(n_warmup=5, n_iterations=10)
        call_count = 0

        def counter():
            nonlocal call_count
            call_count += 1

        result = run_benchmark(counter, config=config)
        assert call_count == config.n_warmup + config.n_iterations
        assert len(result.iterations_us) == config.n_iterations

    def test_run_benchmark_stats_consistent(self):
        config = BenchmarkConfig(n_warmup=1, n_iterations=5)
        result = run_benchmark(lambda: time.sleep(0.001), config=config)
        assert result.min_us <= result.median_us <= result.max_us
        assert all(isinstance(v, float) for v in result.iterations_us)


class TestBenchmarkResult:
    def test_jsonl_append_creates_valid_lines(self):
        result = BenchmarkResult(
            median_us=1.0,
            min_us=0.5,
            max_us=2.0,
            iterations_us=[0.5, 1.0, 2.0],
            metadata={"tag": "test"},
        )
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
            path = f.name
        try:
            result.append_to(path)
            result.append_to(path)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 2
            for line in lines:
                record = json.loads(line)
                assert "median_us" in record
                assert "min_us" in record
                assert "max_us" in record
                assert "iterations_us" in record
                assert "metadata" in record
                assert record["metadata"]["tag"] == "test"
        finally:
            os.unlink(path)


class TestMakeRng:
    def test_different_salts_produce_different_first_values(self):
        rng_a = make_rng("a")
        rng_b = make_rng("b")
        assert rng_a.random() != rng_b.random()

    def test_same_salt_produces_same_stream(self):
        rng1 = make_rng("salt")
        rng2 = make_rng("salt")
        assert rng1.random() == rng2.random()


class TestEnsureDirs:
    def test_ensure_dirs_creates_all_canonical_subdirs(self, tmp_path):
        import npupy_xdna.bench.paths as paths_module

        original_root = paths_module.RESULTS_ROOT
        try:
            paths_module.RESULTS_ROOT = tmp_path / "results"
            paths_module.TIMINGS_DIR = paths_module.RESULTS_ROOT / "timings"
            paths_module.PLOTS_DIR = paths_module.RESULTS_ROOT / "plots"
            paths_module.XCLBIN_CACHE_DIR = paths_module.RESULTS_ROOT / "xclbin_cache"
            paths_module.EVIDENCE_DIR = paths_module.RESULTS_ROOT / "evidence"
            paths_module.CHECKPOINTS_DIR = paths_module.RESULTS_ROOT / "checkpoints"
            ensure_dirs()
            assert paths_module.TIMINGS_DIR.is_dir()
            assert paths_module.PLOTS_DIR.is_dir()
            assert paths_module.XCLBIN_CACHE_DIR.is_dir()
            assert paths_module.EVIDENCE_DIR.is_dir()
            assert paths_module.CHECKPOINTS_DIR.is_dir()
        finally:
            paths_module.RESULTS_ROOT = original_root
            paths_module.TIMINGS_DIR = original_root / "timings"
            paths_module.PLOTS_DIR = original_root / "plots"
            paths_module.XCLBIN_CACHE_DIR = original_root / "xclbin_cache"
            paths_module.EVIDENCE_DIR = original_root / "evidence"
            paths_module.CHECKPOINTS_DIR = original_root / "checkpoints"
