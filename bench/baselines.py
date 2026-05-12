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
    """Honest CPU baseline: int16 -> f32 -> scipy BLAS sgemm -> f32 -> int16.

    Times the FULL round-trip including conversion overhead.
    This is the fair comparison for int16-targeting users who would
    convert to float32 to leverage OpenBLAS/MKL acceleration.

    IMPORTANT: numpy 1.26.4 in ironenv has NO BLAS linked — f32 matmul via
    numpy is unaccelerated.  scipy IS linked to OpenBLAS; use
    scipy.linalg.blas.sgemm for real BLAS performance.
    """
    import scipy.linalg.blas  # noqa: PLC0415

    def _run():
        if region.op in ("matmul", "matmul_fused"):
            A_f32 = inputs[0].astype(np.float32)
            B_f32 = inputs[1].astype(np.float32)
            # sgemm(alpha, a, b) computes alpha * A @ B
            result_f32 = scipy.linalg.blas.sgemm(1.0, A_f32, B_f32)
            return np.clip(result_f32, -32768, 32767).astype(np.int16)
        else:
            return cpu_baseline(region, inputs, config).output

    return run_benchmark(_run, config=config)
