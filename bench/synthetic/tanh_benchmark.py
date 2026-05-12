from __future__ import annotations

import statistics
import time

import numpy as np


def _cpu_tanh_i16(x: np.ndarray) -> np.ndarray:
    xs = x.astype(np.int32) >> 8
    xs2 = xs * xs
    num = xs * (27 + xs2)
    den = 27 + 9 * xs2
    out = np.where(den == 0, np.int32(0), (num * np.int32(32767)) // den)
    out = np.where(xs >= 4, np.int32(32767), np.where(xs <= -4, np.int32(-32768), out))
    return np.clip(out, -32768, 32767).astype(np.int16)


def run_tanh_benchmark(size: int = 1048576, n_warmup: int = 3, n_iters: int = 10) -> dict:
    rng = np.random.default_rng(42)
    x = rng.integers(-100, 100, size=size, dtype=np.int16)

    cpu_times: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _cpu_tanh_i16(x)
        cpu_times.append((time.perf_counter() - t0) * 1e6)

    from npupy_xdna.dispatch import activate, deactivate

    activate()
    npu_times: list[float] = []
    for _ in range(n_warmup):
        np.tanh(x)
    for _ in range(n_iters):
        t0 = time.perf_counter()
        np.tanh(x)
        npu_times.append((time.perf_counter() - t0) * 1e6)
    deactivate()

    cpu_median = statistics.median(cpu_times)
    npu_median = statistics.median(npu_times)
    bandwidth_gbps = (2 * size * 2) / (npu_median * 1e-6) / 1e9
    speedup = cpu_median / npu_median

    return {
        "size": size,
        "cpu_median_us": round(cpu_median, 3),
        "npu_median_us": round(npu_median, 3),
        "bandwidth_gbps": round(bandwidth_gbps, 4),
        "speedup": round(speedup, 4),
    }


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    result = run_tanh_benchmark()
    print(json.dumps(result, indent=2))
