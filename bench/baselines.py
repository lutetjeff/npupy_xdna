from __future__ import annotations

import functools

import numpy as np

from npupy_xdna.bench.timer import BenchmarkConfig, BenchmarkResult, run_benchmark
from npupy_xdna.regions.region import Region
from npupy_xdna.runtime.cpu_runner import CpuRunner


@functools.lru_cache(maxsize=1)
def detect_blas() -> str:
    config_info = str(np.show_config())
    text = config_info.lower()
    if "openblas" in text:
        return "openblas"
    if "mkl" in text or "intel" in text:
        return "mkl"
    if "blis" in text:
        return "blis"
    return "unknown"


def cpu_baseline(
    region: Region,
    inputs: list[np.ndarray],
    config: BenchmarkConfig,
) -> BenchmarkResult:
    cpu_runner = CpuRunner()

    def _run():
        return cpu_runner.run(region, inputs)

    return run_benchmark(_run, config=config)
