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


def cpu_baseline_via_blas(
    region: Region,
    inputs: list[np.ndarray],
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Honest CPU baseline: int16 -> f32 -> BLAS matmul -> int16.

    Times the FULL round-trip including conversion overhead.
    This is the fair comparison for int16-targeting users who would
    convert to float32 to leverage OpenBLAS/MKL acceleration.
    """

    def _run():
        f32_inputs = [inp.astype(np.float32) for inp in inputs]
        result_f32 = np.matmul(f32_inputs[0], f32_inputs[1])
        result_int16 = np.clip(result_f32, -32768, 32767).astype(np.int16)
        return result_int16

    return run_benchmark(_run, config=config)
