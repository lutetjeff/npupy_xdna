from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np

from npupy_xdna.regions.region import Region
from npupy_xdna.templates.protocol import Config, CostEstimate
from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES

_KERNEL_SRC = Path(__file__).parent.parent / "kernels" / "stencil_5pt_int16.cc"
_KERNEL_GEN_DIR = Path(__file__).parent.parent / "kernels"

_NUM_COLS = 8

_SW_DISPATCH_FLOOR_US: float = 500.0
_SW_PEAK_GOPS: float = 1_000.0


def _kernel_path_for(W: int, strip_h: int) -> Path:
    path = _KERNEL_GEN_DIR / f"stencil_5pt_int16_{W}x{strip_h}.cc"
    if not path.exists():
        src = _KERNEL_SRC.read_text()
        header = f"#define STENCIL_W {W}\n#define STENCIL_STRIP_H {strip_h}\n"
        path.write_text(header + src)
    return path


class SlidingWindowTemplate:
    name = "sliding_window"

    def match(self, region: Region) -> bool:
        if region.op != "stencil_2d":
            return False
        if region.output.dtype != "int16":
            return False
        if len(region.output.shape) != 2:
            return False
        H, W = region.output.shape
        if H % _NUM_COLS != 0:
            return False
        return (H, W) in SUPPORTED_SHAPES["sliding_window"]

    def config_space(self, region: Region) -> list[Config]:
        H, W = region.output.shape
        strip_h = H // _NUM_COLS
        return [
            Config(
                tile=(H, W),
                n_cores=_NUM_COLS,
                extra={
                    "strip_h": strip_h,
                    "halo": 1,
                    "stencil": "5pt",
                },
            )
        ]

    def lower(self, region: Region, config: Config) -> Callable:
        H, W = region.output.shape
        strip_h = config.extra["strip_h"]
        strip_in_size = (strip_h + 2) * W
        strip_out_size = strip_h * W

        kernel_path = str(_kernel_path_for(W, strip_h))

        import aie.iron as iron
        from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
        from aie.iron.device import NPU2
        from aie.iron.placers import SequentialPlacer
        from aie.helpers.taplib.tap import TensorAccessPattern
        from aie.utils.config import cxx_header_path

        strip_in_type = np.ndarray[(strip_in_size,), np.dtype[np.int16]]
        strip_out_type = np.ndarray[(strip_out_size,), np.dtype[np.int16]]
        padded_in_type = np.ndarray[((H + 2) * W,), np.dtype[np.int16]]
        flat_out_type = np.ndarray[(H * W,), np.dtype[np.int16]]

        in_taps = [
            TensorAccessPattern(
                (1, (H + 2) * W),
                col * strip_h * W,
                [1, 1, 1, strip_in_size],
                [0, 0, 0, 1],
            )
            for col in range(_NUM_COLS)
        ]
        out_taps = [
            TensorAccessPattern(
                (1, H * W),
                col * strip_h * W,
                [1, 1, 1, strip_out_size],
                [0, 0, 0, 1],
            )
            for col in range(_NUM_COLS)
        ]

        @iron.jit(is_placed=False)
        def stencil_jit(padded_in, flat_out):
            in_fifos = [
                ObjectFifo(strip_in_type, name=f"sw_in{col}")
                for col in range(_NUM_COLS)
            ]
            out_fifos = [
                ObjectFifo(strip_out_type, name=f"sw_out{col}")
                for col in range(_NUM_COLS)
            ]

            stencil_kernel = ExternalFunction(
                "stencil_5pt_int16",
                source_file=kernel_path,
                arg_types=[strip_in_type, strip_out_type],
                include_dirs=[cxx_header_path()],
            )

            def core_fn(of_in, of_out, k):
                elem_in = of_in.acquire(1)
                elem_out = of_out.acquire(1)
                k(elem_in, elem_out)
                of_in.release(1)
                of_out.release(1)

            workers = [
                Worker(
                    core_fn,
                    [in_fifos[col].cons(), out_fifos[col].prod(), stencil_kernel],
                )
                for col in range(_NUM_COLS)
            ]

            rt = Runtime()
            with rt.sequence(padded_in_type, flat_out_type) as (a_in, b_out):
                rt.start(*workers)
                tg = rt.task_group()
                for col in range(_NUM_COLS):
                    rt.fill(in_fifos[col].prod(), a_in, in_taps[col], task_group=tg)
                for col in range(_NUM_COLS):
                    rt.drain(
                        out_fifos[col].cons(),
                        b_out,
                        out_taps[col],
                        wait=True,
                        task_group=tg,
                    )
                rt.finish_task_group(tg)

            return Program(NPU2(), rt).resolve_program(SequentialPlacer())

        def run(inp: np.ndarray, out: np.ndarray) -> None:
            import aie.iron as _iron

            padded_np = np.zeros((H + 2) * W, dtype=np.int16)
            padded_np[W : (H + 1) * W] = inp.flatten()
            padded_t = _iron.tensor(padded_np, dtype=np.int16, device="npu")
            out_t = _iron.zeros(H * W, dtype=np.int16, device="npu")
            stencil_jit(padded_t, out_t)
            out[:] = np.array(out_t.numpy(), copy=True).reshape(H, W)

        return run

    def estimated_cost(self, region: Region, config: Config) -> CostEstimate:
        H, W = region.output.shape
        ops = H * W * 5
        throughput_us = ops / (_SW_PEAK_GOPS * 1_000.0)
        lat = _SW_DISPATCH_FLOOR_US + throughput_us
        gops = ops / (lat * 1_000.0)
        confidence = min(1.0, throughput_us / _SW_DISPATCH_FLOOR_US)
        return CostEstimate(
            predicted_latency_us=lat,
            predicted_gops=gops,
            confidence=confidence,
        )
