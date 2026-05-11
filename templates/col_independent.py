from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from npupy_xdna.regions.region import Region
from npupy_xdna.templates.protocol import Config, CostEstimate
from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES

_KERNEL_CC = Path(__file__).parent.parent / "kernels" / "relu_int16.cc"
_HASH_KERNEL_CC = Path(__file__).parent.parent / "kernels" / "hash_int16.cc"
_TANH_KERNEL_CC = Path(__file__).parent.parent / "kernels" / "tanh_int16.cc"

_NUM_COLS = 8
_NUM_CORES_PER_COL = 4
_N_WORKERS = _NUM_COLS * _NUM_CORES_PER_COL

# LINE_SIZE must match INT16_TILE_SIZE in relu_int16.cc.
# COL_LINE_SIZE = NUM_CORES_PER_COL * LINE_SIZE (col FIFO = 4 core slices merged).
_LINE_SIZE = 512
_COL_LINE_SIZE = _NUM_CORES_PER_COL * _LINE_SIZE  # 2048


class ColIndependentTemplate:
    name = "col_independent"

    def match(self, region: Region) -> bool:
        total = int(np.prod(region.output.shape))
        return (
            region.op in {"elementwise_unary", "elementwise_binary"}
            and region.output.dtype == "int16"
            and total in SUPPORTED_SHAPES["col_indep"]
        )

    def config_space(self, region: Region) -> list[Config]:
        total = int(np.prod(region.output.shape))
        return [
            Config(
                tile=(total,),
                n_cores=_N_WORKERS,
                extra={"chunk_per_pool": total // _NUM_COLS},
            )
        ]

    def lower(self, region: Region, config: Config) -> Callable:
        total = int(np.prod(region.output.shape))
        op = region.op
        compute_fn = region.metadata.get("compute_fn", "relu")
        if compute_fn == "tanh":
            kernel_path = str(_TANH_KERNEL_CC)
        elif compute_fn == "hash":
            kernel_path = str(_HASH_KERNEL_CC)
        else:
            kernel_path = str(_KERNEL_CC)

        import aie.iron as iron
        from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
        from aie.iron.device import NPU2
        from aie.iron.placers import SequentialPlacer
        from aie.helpers.taplib.tap import TensorAccessPattern
        from aie.utils.config import cxx_header_path

        col_type = np.ndarray[(_COL_LINE_SIZE,), np.dtype[np.int16]]
        core_type = np.ndarray[(_LINE_SIZE,), np.dtype[np.int16]]
        transfer_type = np.ndarray[(total,), np.dtype[np.int16]]

        chunk_per_col = total // _NUM_COLS
        split_offsets = [j * _LINE_SIZE for j in range(_NUM_CORES_PER_COL)]

        if op == "elementwise_unary":

            @iron.jit(is_placed=False)
            def relu_fn(input0, output):
                col_ins = [
                    ObjectFifo(col_type, name=f"col_in{col}")
                    for col in range(_NUM_COLS)
                ]
                core_in_fifos = [
                    col_ins[col].cons().split(
                        split_offsets,
                        obj_types=[core_type] * _NUM_CORES_PER_COL,
                        names=[
                            f"core_in{col}_{j}" for j in range(_NUM_CORES_PER_COL)
                        ],
                    )
                    for col in range(_NUM_COLS)
                ]

                col_outs = [
                    ObjectFifo(col_type, name=f"col_out{col}")
                    for col in range(_NUM_COLS)
                ]
                core_out_fifos = [
                    col_outs[col].prod().join(
                        split_offsets,
                        obj_types=[core_type] * _NUM_CORES_PER_COL,
                        names=[
                            f"core_out{col}_{j}" for j in range(_NUM_CORES_PER_COL)
                        ],
                    )
                    for col in range(_NUM_COLS)
                ]

                if compute_fn == "tanh":
                    kernel_sym = "int16_tanh"
                elif compute_fn == "hash":
                    kernel_sym = "int16_hash"
                else:
                    kernel_sym = "int16_relu"
                unary_kernel = ExternalFunction(
                    kernel_sym,
                    source_file=kernel_path,
                    arg_types=[core_type, core_type],
                    include_dirs=[cxx_header_path()],
                )

                def core_fn(of_in, of_out, unary_k):
                    elem_in = of_in.acquire(1)
                    elem_out = of_out.acquire(1)
                    unary_k(elem_in, elem_out)
                    of_in.release(1)
                    of_out.release(1)

                workers = [
                    Worker(
                        core_fn,
                        [
                            core_in_fifos[col][j].cons(),
                            core_out_fifos[col][j].prod(),
                            unary_kernel,
                        ],
                    )
                    for col in range(_NUM_COLS)
                    for j in range(_NUM_CORES_PER_COL)
                ]

                taps = [
                    TensorAccessPattern(
                        (1, total),
                        chunk_per_col * col,
                        [1, 1, 1, chunk_per_col],
                        [0, 0, 0, 1],
                    )
                    for col in range(_NUM_COLS)
                ]

                rt = Runtime()
                with rt.sequence(transfer_type, transfer_type) as (a_in, b_out):
                    rt.start(*workers)
                    tg = rt.task_group()
                    for col in range(_NUM_COLS):
                        rt.fill(col_ins[col].prod(), a_in, taps[col], task_group=tg)
                    for col in range(_NUM_COLS):
                        rt.drain(
                            col_outs[col].cons(),
                            b_out,
                            taps[col],
                            wait=True,
                            task_group=tg,
                        )
                    rt.finish_task_group(tg)

                return Program(NPU2(), rt).resolve_program(SequentialPlacer())

            return relu_fn

        elif op == "elementwise_binary":

            @iron.jit(is_placed=False)
            def add_fn(input0, input1, output):
                col_in1s = [
                    ObjectFifo(col_type, name=f"col_in1_{col}")
                    for col in range(_NUM_COLS)
                ]
                col_in2s = [
                    ObjectFifo(col_type, name=f"col_in2_{col}")
                    for col in range(_NUM_COLS)
                ]
                core_in1_fifos = [
                    col_in1s[col].cons().split(
                        split_offsets,
                        obj_types=[core_type] * _NUM_CORES_PER_COL,
                        names=[
                            f"core_in1_{col}_{j}" for j in range(_NUM_CORES_PER_COL)
                        ],
                    )
                    for col in range(_NUM_COLS)
                ]
                core_in2_fifos = [
                    col_in2s[col].cons().split(
                        split_offsets,
                        obj_types=[core_type] * _NUM_CORES_PER_COL,
                        names=[
                            f"core_in2_{col}_{j}" for j in range(_NUM_CORES_PER_COL)
                        ],
                    )
                    for col in range(_NUM_COLS)
                ]

                col_outs = [
                    ObjectFifo(col_type, name=f"col_out{col}")
                    for col in range(_NUM_COLS)
                ]
                core_out_fifos = [
                    col_outs[col].prod().join(
                        split_offsets,
                        obj_types=[core_type] * _NUM_CORES_PER_COL,
                        names=[
                            f"core_out{col}_{j}" for j in range(_NUM_CORES_PER_COL)
                        ],
                    )
                    for col in range(_NUM_COLS)
                ]

                add_kernel = ExternalFunction(
                    "int16_eltwise_add",
                    source_file=kernel_path,
                    arg_types=[core_type, core_type, core_type],
                    include_dirs=[cxx_header_path()],
                )

                def core_fn_add(of_in1, of_in2, of_out, add_k):
                    elem_in1 = of_in1.acquire(1)
                    elem_in2 = of_in2.acquire(1)
                    elem_out = of_out.acquire(1)
                    add_k(elem_in1, elem_in2, elem_out)
                    of_in1.release(1)
                    of_in2.release(1)
                    of_out.release(1)

                workers = [
                    Worker(
                        core_fn_add,
                        [
                            core_in1_fifos[col][j].cons(),
                            core_in2_fifos[col][j].cons(),
                            core_out_fifos[col][j].prod(),
                            add_kernel,
                        ],
                    )
                    for col in range(_NUM_COLS)
                    for j in range(_NUM_CORES_PER_COL)
                ]

                taps = [
                    TensorAccessPattern(
                        (1, total),
                        chunk_per_col * col,
                        [1, 1, 1, chunk_per_col],
                        [0, 0, 0, 1],
                    )
                    for col in range(_NUM_COLS)
                ]

                rt = Runtime()
                with rt.sequence(
                    transfer_type, transfer_type, transfer_type
                ) as (a_in, b_in, c_out):
                    rt.start(*workers)
                    tg = rt.task_group()
                    for col in range(_NUM_COLS):
                        rt.fill(
                            col_in1s[col].prod(), a_in, taps[col], task_group=tg
                        )
                        rt.fill(
                            col_in2s[col].prod(), b_in, taps[col], task_group=tg
                        )
                    for col in range(_NUM_COLS):
                        rt.drain(
                            col_outs[col].cons(),
                            c_out,
                            taps[col],
                            wait=True,
                            task_group=tg,
                        )
                    rt.finish_task_group(tg)

                return Program(NPU2(), rt).resolve_program(SequentialPlacer())

            return add_fn

        else:
            raise ValueError(f"ColIndependentTemplate.lower: unsupported op={op!r}")

    def estimated_cost(self, region: Region, config: Config) -> CostEstimate:
        total = int(np.prod(region.output.shape))
        n_inputs = len(region.inputs)
        bytes_moved = total * 2 * (n_inputs + 1)
        npu_bw_bytes_per_us = 200_000
        latency_us = bytes_moved / npu_bw_bytes_per_us
        gops = (total / latency_us) / 1_000
        return CostEstimate(
            predicted_latency_us=latency_us,
            predicted_gops=gops,
            confidence=0.5,
        )
