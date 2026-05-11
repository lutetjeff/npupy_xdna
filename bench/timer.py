"""Reusable benchmarking utility for NPUPy XDNA.

Provides:
- BenchmarkConfig: dataclass for warmup/iteration counts, seed, evidence dir.
- Timer: context manager measuring elapsed time via time.perf_counter().
- BenchmarkResult: stores per-iteration timings and metadata.
- run_benchmark: executes n_warmup discarded calls then n_iterations measured calls.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    n_warmup: int = 5
    n_iterations: int = 10
    seed: int = 42
    evidence_dir: str = "/home/lutet/ece511/.sisyphus/evidence"


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    median_us: float
    min_us: float
    max_us: float
    iterations_us: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def append_to(self, path: str) -> None:
        """Append this result as one JSON line to a JSONL file."""
        record = {
            "median_us": self.median_us,
            "min_us": self.min_us,
            "max_us": self.max_us,
            "iterations_us": self.iterations_us,
            "metadata": self.metadata,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class Timer:
    """Context manager that measures elapsed wall time via perf_counter()."""

    def __init__(self) -> None:
        self._start: float | None = None
        self._elapsed_s: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._start is not None:
            self._elapsed_s = time.perf_counter() - self._start
            self._start = None

    @property
    def elapsed_us(self) -> float:
        """Elapsed time in microseconds."""
        return self._elapsed_s * 1e6


def run_benchmark(
    fn: Callable[..., Any],
    *args: Any,
    config: BenchmarkConfig,
    **kwargs: Any,
) -> BenchmarkResult:
    """Run ``fn(*args, **kwargs)`` with warmup then measured iterations.

    - ``config.n_warmup`` calls are executed and their timings discarded.
    - ``config.n_iterations`` calls are measured and stored.
    """
    # Warmup
    for _ in range(config.n_warmup):
        fn(*args, **kwargs)

    # Measured iterations
    iterations_us: list[float] = []
    for _ in range(config.n_iterations):
        with Timer() as t:
            fn(*args, **kwargs)
        iterations_us.append(t.elapsed_us)

    median_us = float(statistics.median(iterations_us))
    min_us = float(min(iterations_us))
    max_us = float(max(iterations_us))

    return BenchmarkResult(
        median_us=median_us,
        min_us=min_us,
        max_us=max_us,
        iterations_us=iterations_us,
        metadata={},
    )
