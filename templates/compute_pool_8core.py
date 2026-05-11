from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np

from npupy_xdna.regions.region import Region
from npupy_xdna.templates.protocol import Config, CostEstimate

_N_COLS = 8
_N_CORES = _N_COLS

_KERNEL_SRC = Path(__file__).parent.parent / "kernels" / "relu_int16.cc"
_AIE_INCLUDE = (
    Path(__file__).parent.parent.parent.parent
    / "mlir-aie"
    / "aie_kernels"
    / "aie2"
)

SIZES_8CORE = [32768, 131072, 524288, 2097152]


class ComputePool8CoreTemplate:
    name = "compute_pool_8core"

    def match(self, region: Region) -> bool:
        if region.op not in {"elementwise_unary", "elementwise_binary"}:
            return False
        if region.output.dtype != "int16":
            return False
        total = int(math.prod(region.output.shape))
        return total in SIZES_8CORE

    def config_space(self, region: Region) -> list[Config]:
        total = int(math.prod(region.output.shape))
        chunk = total // _N_CORES  # elements per core
        return [Config(tile=(chunk,), n_cores=_N_CORES, extra={"total": total})]

    def lower(self, region: Region, config: Config) -> Callable:
        total = config.extra["total"]
        chunk = config.tile[0]

        def iron_fn(*args):
            import aie.iron as iron
            from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
            from aie.iron.placers import SequentialPlacer
            from aie.iron.device import NPU2
            from aie.helpers.taplib.tap import TensorAccessPattern

            transfer_ty = np.ndarray[(total,), np.dtype[np.int16]]
            core_ty = np.ndarray[(chunk,), np.dtype[np.int16]]

            in_col_fifos = [
                ObjectFifo(core_ty, name=f"in_col{c}") for c in range(_N_COLS)
            ]
            out_col_fifos = [
                ObjectFifo(core_ty, name=f"out_col{c}") for c in range(_N_COLS)
            ]

            relu_kernel = ExternalFunction(
                "relu_int16",
                source_file=str(_KERNEL_SRC),
                arg_types=[core_ty, core_ty, np.int32],
                include_dirs=[str(_AIE_INCLUDE)],
            )

            def core_fn(of_in, of_out, relu_k):
                elem_out = of_out.acquire(1)
                elem_in = of_in.acquire(1)
                relu_k(elem_in, elem_out, chunk)
                of_in.release(1)
                of_out.release(1)

            workers = [
                Worker(
                    core_fn,
                    fn_args=[
                        in_col_fifos[c].cons(),
                        out_col_fifos[c].prod(),
                        relu_kernel,
                    ],
                )
                for c in range(_N_COLS)
            ]

            rt = Runtime()
            with rt.sequence(transfer_ty, transfer_ty) as (a_in, b_out):
                rt.start(*workers)
                tg = rt.task_group()
                for c in range(_N_COLS):
                    tap = TensorAccessPattern(
                        (1, total),
                        chunk * c,
                        [1, 1, 1, chunk],
                        [0, 0, 0, 1],
                    )
                    rt.fill(in_col_fifos[c].prod(), a_in, tap, task_group=tg)
                    rt.drain(
                        out_col_fifos[c].cons(), b_out, tap, wait=True, task_group=tg
                    )
                rt.finish_task_group(tg)

            return Program(NPU2(), rt).resolve_program(SequentialPlacer())

        return iron_fn

    def estimated_cost(self, region: Region, config: Config) -> CostEstimate:
        total = config.extra["total"]
        bytes_transferred = total * 2 * 2
        bw_gb_s = 20.0
        latency_us = (bytes_transferred / (bw_gb_s * 1e9)) * 1e6 + 100.0
        gops = total / latency_us / 1e3
        return CostEstimate(
            predicted_latency_us=latency_us,
            predicted_gops=gops,
            confidence=0.6,
        )
