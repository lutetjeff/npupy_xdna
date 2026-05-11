"""NPUPy XDNA benchmarking utilities."""

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
from npupy_xdna.bench.timer import (
    BenchmarkConfig,
    BenchmarkResult,
    Timer,
    run_benchmark,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "Timer",
    "run_benchmark",
    "make_rng",
    "RESULTS_ROOT",
    "TIMINGS_DIR",
    "PLOTS_DIR",
    "XCLBIN_CACHE_DIR",
    "EVIDENCE_DIR",
    "CHECKPOINTS_DIR",
    "ensure_dirs",
]
