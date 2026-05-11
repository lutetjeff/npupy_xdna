"""CGRA depth sweep: Horner pipeline at depths 3, 8, 16 on 256-element int16.

Usage:
    source /opt/xilinx/xrt/setup.sh
    source ~/mlir-aie/ironenv/bin/activate
    source ~/mlir-aie/utils/env_setup.sh
    python -m npupy_xdna.scripts.cgra_depth_sweep
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np

_KERNELS_DIR = Path(__file__).parent.parent / "kernels"
_N = 256
_DEPTHS = [3, 8, 16]
_N_WARMUP = 3
_N_ITERS = 5

JSONL_PATH = Path(__file__).parent.parent / "results" / "timings" / "cgra_depth_sweep.jsonl"


def _tile_for_stage(stage_idx: int) -> tuple[int, int]:
    col = stage_idx // 4
    row = 2 + (stage_idx % 4)
    return col, row


def _cpu_horner(x: np.ndarray, depth: int) -> np.ndarray:
    result = x.astype(np.int32)
    for _ in range(depth):
        result = result * 3 + 7
        result = np.clip(result, -32768, 32767)
    return result.astype(np.int16)


def _make_horner_pipeline(depth: int):
    import aie.iron as iron
    from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
    from aie.iron.placers import SequentialPlacer
    from aie.iron.device import NPU2, Tile
    from aie.utils.config import cxx_header_path

    n = _N
    tile_ty = np.ndarray[(n,), np.dtype[np.int16]]
    inc = [cxx_header_path()]
    kernel_src = str(_KERNELS_DIR / "horner_stage_int16.cc")

    def horner_pipeline(x_in, out):
        horner_fn = ExternalFunction(
            "horner_stage",
            source_file=kernel_src,
            arg_types=[tile_ty, tile_ty],
            include_dirs=inc,
        )

        fifos = [
            ObjectFifo(tile_ty, name=f"hof_{depth}d_{i}", depth=2)
            for i in range(depth + 1)
        ]

        def stage_fn(in_fifo, out_fifo, fn):
            elem_in = in_fifo.acquire(1)
            elem_out = out_fifo.acquire(1)
            fn(elem_in, elem_out)
            in_fifo.release(1)
            out_fifo.release(1)

        workers = []
        for i in range(depth):
            col, row = _tile_for_stage(i)
            w = Worker(
                stage_fn,
                fn_args=[fifos[i].cons(), fifos[i + 1].prod(), horner_fn],
                placement=Tile(col, row),
            )
            workers.append(w)

        rt = Runtime()
        with rt.sequence(tile_ty, tile_ty) as (X, OUT):
            rt.start(*workers)
            rt.fill(fifos[0].prod(), X)
            rt.drain(fifos[-1].cons(), OUT, wait=True)

        return Program(NPU2(), rt).resolve_program(SequentialPlacer())

    horner_pipeline.__name__ = f"horner_pipeline_d{depth}"
    horner_pipeline.__qualname__ = f"horner_pipeline_d{depth}"
    return iron.jit(is_placed=False)(horner_pipeline)


def _run_npu(iron_fn, x_int16: np.ndarray) -> tuple[float, list[float]]:
    import aie.iron as iron

    x_buf = iron.tensor(x_int16, dtype=np.int16, device="npu")
    out_buf = iron.zeros(_N, dtype=np.int16, device="npu")

    for i in range(_N_WARMUP):
        t0 = time.perf_counter()
        iron_fn(x_buf, out_buf)
        t1 = time.perf_counter()
        print(f"    warmup[{i}]: {(t1-t0)*1e6:.0f} us")

    timings = []
    for i in range(_N_ITERS):
        t0 = time.perf_counter()
        iron_fn(x_buf, out_buf)
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1e6
        timings.append(elapsed)
        print(f"    iter[{i}]: {elapsed:.0f} us")

    return float(statistics.median(timings)), timings


def _run_cpu(x_int16: np.ndarray, depth: int) -> tuple[float, list[float]]:
    for _ in range(5):
        _cpu_horner(x_int16, depth)

    timings = []
    for _ in range(20):
        t0 = time.perf_counter()
        _cpu_horner(x_int16, depth)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1e6)
    return float(statistics.median(timings)), timings


def sweep() -> None:
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    x = rng.integers(-100, 100, size=_N, dtype=np.int16)

    records = []

    for depth in _DEPTHS:
        print(f"\n=== depth={depth} ===")
        print(f"  tile layout: {[_tile_for_stage(i) for i in range(depth)]}")

        print(f"  Building IRON pipeline depth={depth}...")
        iron_fn = _make_horner_pipeline(depth)

        import aie.iron as iron
        x_buf = iron.tensor(x, dtype=np.int16, device="npu")
        out_buf = iron.zeros(_N, dtype=np.int16, device="npu")

        print(f"  Running NPU benchmark (warmup={_N_WARMUP}, iters={_N_ITERS})...")
        npu_med, npu_timings = _run_npu(iron_fn, x)
        print(f"  NPU median: {npu_med:.1f} us")

        iron_fn(x_buf, out_buf)
        npu_out = np.array(out_buf.numpy(), copy=True)
        cpu_ref = _cpu_horner(x, depth)
        n_mismatches = int(np.sum(npu_out != cpu_ref))
        print(f"  Correctness: {n_mismatches}/{_N} mismatches")

        cpu_med, cpu_timings = _run_cpu(x, depth)
        print(f"  CPU median: {cpu_med:.4f} us")

        total_ops = depth * _N * 2
        per_op_us = npu_med / depth
        speedup = cpu_med / npu_med if npu_med > 0 else 0.0
        outcome = "npu_wins" if speedup > 1.0 else "cpu_wins"

        print(f"  per_op_latency: {per_op_us:.2f} us/stage")
        print(f"  total_ops: {total_ops}")
        print(f"  speedup(CPU/NPU): {speedup:.4f}x -> {outcome}")

        record = {
            "depth": depth,
            "n_elements": _N,
            "total_ops": total_ops,
            "npu_median_us": round(npu_med, 2),
            "per_op_latency_us": round(per_op_us, 4),
            "cpu_median_us": round(cpu_med, 6),
            "speedup_cpu_over_npu": round(speedup, 4),
            "outcome": outcome,
            "n_mismatches": n_mismatches,
            "npu_n_warmup": _N_WARMUP,
            "npu_n_iters": _N_ITERS,
            "npu_timings_us": [round(t, 2) for t in npu_timings],
            "cpu_timings_us": [round(t, 6) for t in cpu_timings],
            "tile_layout": [list(_tile_for_stage(i)) for i in range(depth)],
        }
        records.append(record)
        print(f"  Record: {json.dumps({k: v for k, v in record.items() if k not in ('npu_timings_us', 'cpu_timings_us', 'tile_layout')})}")

    with open(JSONL_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nResults written to {JSONL_PATH}")
    print("\n=== Per-op cost summary ===")
    print(f"{'depth':>6} {'total_us':>10} {'per_op_us':>12} {'total_ops':>12}")
    for rec in records:
        print(f"{rec['depth']:>6} {rec['npu_median_us']:>10.1f} {rec['per_op_latency_us']:>12.4f} {rec['total_ops']:>12}")


if __name__ == "__main__":
    sweep()
