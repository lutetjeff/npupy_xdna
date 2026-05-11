"""
GemmFusionTemplate: whole-array int16 GEMM, 32 AIE cores, npu2.

b_col_maj=True is REQUIRED — omitting it causes silent wrong output.
Dims are baked at xclbin compile time; only SUPPORTED_SHAPES are valid.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from npupy_xdna.regions.region import Region
from npupy_xdna.templates.protocol import CostEstimate, Config
from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES

_COMBINED_KERNEL = "/home/lutet/ece511/npupy_xdna/kernels/gemm_i16_all.cc"
_AIE_KERNELS_DIR = "/home/lutet/mlir-aie/aie_kernels"
_N_AIE_ROWS = 4


def _get_n_aie_cols(M: int, N: int, m: int, n: int) -> int:
    for cols in [8, 4, 2, 1]:
        if N % (n * cols) == 0 and M % (m * _N_AIE_ROWS) == 0:
            return cols
    return 1


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _kernel_bin_name(m: int, k: int, n: int, epilogue: str, prologue: str,
                     bias_value: int, alpha_value: int) -> str:
    tag = f"{m}x{k}x{n}"
    if epilogue == "relu":
        tag += "_relu"
    elif epilogue == "bias_add":
        tag += f"_bias{bias_value}"
    if prologue == "scale":
        tag += f"_scale{alpha_value}"
    return f"gemm_i16_{tag}.o"


def _kernel_flags(m: int, k: int, n: int, epilogue: str, prologue: str,
                  bias_value: int, alpha_value: int) -> list[str]:
    flags = [
        "-Di16_i16_ONLY",
        f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}",
        "-DB_COL_MAJ",
        "-DVECTORIZED_ONLY",
    ]
    if epilogue == "relu":
        flags.append("-DEPILOGUE_RELU")
    elif epilogue == "bias_add":
        flags += ["-DEPILOGUE_BIAS_ADD", f"-DBIAS_VAL={bias_value}"]
    if prologue == "scale":
        flags += ["-DPROLOGUE_SCALE", f"-DALPHA_VAL={alpha_value}"]
    return flags


def _build_iron_fn(
    M: int, K: int, N: int,
    m: int, k: int, n: int,
    n_aie_cols: int,
    epilogue: str,
    prologue: str,
    bias_value: int,
    alpha_value: int,
) -> Callable:
    import aie.iron as iron
    from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
    from aie.iron.placers import SequentialPlacer
    from aie.iron.controlflow import range_
    from aie.iron.device import NPU2, Tile
    from aie.helpers.taplib import TensorTiler2D
    from aie.utils.config import cxx_header_path

    n_aie_rows = _N_AIE_ROWS
    n_aie_cores = n_aie_cols * n_aie_rows
    fifo_depth = 2
    r, s, t = 4, 4, 8  # npu2 i16 MAC intrinsic dims (AIE2P)

    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    if n_aie_cols > n_aie_rows:
        n_shim_mem_A = n_aie_rows
    else:
        n_shim_mem_A = n_aie_cols
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2
    m_row_groups = M // m // n_aie_rows
    c_tb_n_rows = min(tb_n_rows, max(1, m_row_groups))

    dtype = np.int16
    A_ty = np.ndarray[(M * K,), np.dtype[dtype]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype]]

    kernel_includes = [cxx_header_path(), _AIE_KERNELS_DIR]
    kernel_flags = _kernel_flags(m, k, n, epilogue, prologue, bias_value, alpha_value)
    bin_name = _kernel_bin_name(m, k, n, epilogue, prologue, bias_value, alpha_value)

    def _gemm_fn(input0, input1, output):
        def _ef(name, arg_types):
            return ExternalFunction(
                name,
                source_file=_COMBINED_KERNEL,
                object_file_name=bin_name,
                arg_types=arg_types,
                include_dirs=kernel_includes,
                compile_flags=kernel_flags,
            )

        zero_fn = _ef("zero_i16", [C_l1_ty])
        matmul_fn = _ef("matmul_i16_i16", [A_l1_ty, B_l1_ty, C_l1_ty])
        relu_fn = _ef("relu_i16_tile", [C_l1_ty]) if epilogue == "relu" else None
        bias_fn = _ef("add_bias_i16_tile", [C_l1_ty]) if epilogue == "bias_add" else None
        scale_fn = _ef("scale_i16_tile", [C_l1_ty]) if prologue == "scale" else None

        A_l3l2_fifos: list = [None] * n_shim_mem_A
        A_l2l1_fifos: list = [None] * n_aie_rows

        for i in range(n_shim_mem_A):
            A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
            start_row = i * n_A_tiles_per_shim
            stop_row = start_row + n_A_tiles_per_shim
            of_offsets = [m * k * j for j in range(stop_row - start_row)]
            dims_to_stream = [
                [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
            ] * (stop_row - start_row)
            a_tmp = A_l3l2_fifos[i].cons().split(
                offsets=of_offsets,
                obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row_i}" for row_i in range(start_row, stop_row)],
                dims_to_stream=dims_to_stream,
                placement=Tile(2 * i if n_aie_cols == 8 else i, 1),
            )
            for j in range(stop_row - start_row):
                A_l2l1_fifos[j + start_row] = a_tmp[j]

        B_l3l2_fifos: list = [None] * n_aie_cols
        B_l2l1_fifos: list = [None] * n_aie_cols
        C_l1l2_fifos: list = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3_fifos: list = [None] * n_aie_cols

        for col in range(n_aie_cols):
            B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
            b_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]  # b_col_maj=True
            B_l2l1_fifos[col] = B_l3l2_fifos[col].cons().forward(
                obj_type=B_l1_ty, name=f"B_L2L1_{col}",
                dims_to_stream=b_dims, placement=Tile(col, 1),
            )
            C_l2l3_fifos[col] = ObjectFifo(
                C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth,
                dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)],
            )
            c_tmp = C_l2l3_fifos[col].prod().join(
                offsets=[m * n * row_i for row_i in range(n_aie_rows)],
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row_i}" for row_i in range(n_aie_rows)],
                depths=[fifo_depth] * n_aie_rows,
                placement=Tile(col, 1),
            )
            for row_i in range(n_aie_rows):
                C_l1l2_fifos[row_i][col] = c_tmp[row_i]

        if epilogue == "none" and prologue == "none":
            def core_fn(in_a, in_b, out_c, zero, matmul):
                loop = range(1) if n_tiles_per_core <= 1 else range_(n_tiles_per_core)
                for _ in loop:
                    elem_out = out_c.acquire(1)
                    zero(elem_out)
                    for _ in range_(K // k):
                        elem_in_a = in_a.acquire(1)
                        elem_in_b = in_b.acquire(1)
                        matmul(elem_in_a, elem_in_b, elem_out)
                        in_a.release(1)
                        in_b.release(1)
                    out_c.release(1)
        elif epilogue == "relu" and prologue == "none":
            def core_fn(in_a, in_b, out_c, zero, matmul, relu):  # noqa: F811
                loop = range(1) if n_tiles_per_core <= 1 else range_(n_tiles_per_core)
                for _ in loop:
                    elem_out = out_c.acquire(1)
                    zero(elem_out)
                    for _ in range_(K // k):
                        elem_in_a = in_a.acquire(1)
                        elem_in_b = in_b.acquire(1)
                        matmul(elem_in_a, elem_in_b, elem_out)
                        in_a.release(1)
                        in_b.release(1)
                    relu(elem_out)
                    out_c.release(1)
        elif epilogue == "bias_add" and prologue == "none":
            def core_fn(in_a, in_b, out_c, zero, matmul, bias):  # noqa: F811
                loop = range(1) if n_tiles_per_core <= 1 else range_(n_tiles_per_core)
                for _ in loop:
                    elem_out = out_c.acquire(1)
                    zero(elem_out)
                    for _ in range_(K // k):
                        elem_in_a = in_a.acquire(1)
                        elem_in_b = in_b.acquire(1)
                        matmul(elem_in_a, elem_in_b, elem_out)
                        in_a.release(1)
                        in_b.release(1)
                    bias(elem_out)
                    out_c.release(1)
        elif epilogue == "none" and prologue == "scale":
            def core_fn(in_a, in_b, out_c, zero, matmul, scale):  # noqa: F811
                loop = range(1) if n_tiles_per_core <= 1 else range_(n_tiles_per_core)
                for _ in loop:
                    elem_out = out_c.acquire(1)
                    zero(elem_out)
                    for _ in range_(K // k):
                        elem_in_a = in_a.acquire(1)
                        elem_in_b = in_b.acquire(1)
                        matmul(elem_in_a, elem_in_b, elem_out)
                        in_a.release(1)
                        in_b.release(1)
                    scale(elem_out)
                    out_c.release(1)
        elif epilogue == "relu" and prologue == "scale":
            def core_fn(in_a, in_b, out_c, zero, matmul, relu, scale):  # noqa: F811
                loop = range(1) if n_tiles_per_core <= 1 else range_(n_tiles_per_core)
                for _ in loop:
                    elem_out = out_c.acquire(1)
                    zero(elem_out)
                    for _ in range_(K // k):
                        elem_in_a = in_a.acquire(1)
                        elem_in_b = in_b.acquire(1)
                        matmul(elem_in_a, elem_in_b, elem_out)
                        in_a.release(1)
                        in_b.release(1)
                    relu(elem_out)
                    scale(elem_out)
                    out_c.release(1)
        else:
            def core_fn(in_a, in_b, out_c, zero, matmul, bias, scale):  # noqa: F811
                loop = range(1) if n_tiles_per_core <= 1 else range_(n_tiles_per_core)
                for _ in loop:
                    elem_out = out_c.acquire(1)
                    zero(elem_out)
                    for _ in range_(K // k):
                        elem_in_a = in_a.acquire(1)
                        elem_in_b = in_b.acquire(1)
                        matmul(elem_in_a, elem_in_b, elem_out)
                        in_a.release(1)
                        in_b.release(1)
                    scale(elem_out)  # prologue:scale applied before epilogue
                    bias(elem_out)
                    out_c.release(1)

        workers = []
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                w_args = [
                    A_l2l1_fifos[row].cons(),
                    B_l2l1_fifos[col].cons(),
                    C_l1l2_fifos[row][col].prod(),
                    zero_fn,
                    matmul_fn,
                ]
                if relu_fn is not None:
                    w_args.append(relu_fn)
                if bias_fn is not None:
                    w_args.append(bias_fn)
                if scale_fn is not None:
                    w_args.append(scale_fn)
                workers.append(Worker(
                    core_fn, w_args,
                    placement=Tile(col, row + 2),
                    stack_size=0xD00,
                ))

        A_tiles = TensorTiler2D.group_tiler(
            (M, K), (m * n_A_tiles_per_shim, k), (1, K // k),
            pattern_repeat=N // n // n_aie_cols,
            prune_step=False,
        )
        B_tiles = TensorTiler2D.step_tiler(
            (N, K), (n, k),
            tile_group_repeats=(N // n // n_aie_cols, K // k),
            tile_group_steps=(n_aie_cols, 1),
            prune_step=False,
        )
        C_tiles = TensorTiler2D.step_tiler(
            (M, N), (m * n_aie_rows, n),
            tile_group_repeats=(c_tb_n_rows, N // n // n_aie_cols),
            tile_group_steps=(1, n_aie_cols),
            prune_step=False,
        )

        rt = Runtime()
        with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
            rt.start(*workers)
            tg = rt.task_group()
            c_index = 0
            for tb in range(_ceildiv(m_row_groups, tb_max_n_rows)):
                for pingpong in [0, 1]:
                    if c_index >= len(C_tiles):
                        break
                    row_base = tb * tb_max_n_rows + pingpong * c_tb_n_rows
                    current_tb_n_rows = min(c_tb_n_rows, m_row_groups - row_base)
                    for col in range(n_aie_cols):
                        rt.drain(
                            C_l2l3_fifos[col].cons(), C,
                            tap=C_tiles[c_index], wait=True,
                            task_group=tg, placement=Tile(col, 0),
                        )
                        c_index += 1
                        for tile_row in range(current_tb_n_rows):
                            tile_offset = (
                                (row_base + tile_row) * n_shim_mem_A + col
                            ) % len(A_tiles)
                            if col < n_aie_rows:
                                rt.fill(
                                    A_l3l2_fifos[col].prod(), A,
                                    tap=A_tiles[tile_offset], task_group=tg,
                                    placement=Tile(2 * col if n_aie_cols == 8 else col, 0),
                                )
                            rt.fill(
                                B_l3l2_fifos[col].prod(), B,
                                tap=B_tiles[col], task_group=tg,
                                placement=Tile(col, 0),
                            )
                    if tb > 0 or (tb == 0 and pingpong > 0):
                        rt.finish_task_group(tg)
                        tg = rt.task_group()
            rt.finish_task_group(tg)

        my_program = Program(NPU2(), rt)
        return my_program.resolve_program(SequentialPlacer())

    _tag = f"{m}x{k}x{n}_c{n_aie_cols}_{epilogue}_{prologue}_{bias_value}_{alpha_value}"
    _gemm_fn.__name__ = f"gemm_iron_{_tag}"
    gemm_iron = iron.jit(is_placed=False)(_gemm_fn)
    return gemm_iron


class GemmFusionTemplate:
    name = "gemm_fusion"

    def match(self, region: Region) -> bool:
        if region.op not in {"matmul", "matmul_fused"}:
            return False
        if region.inputs[0].dtype != "int16":
            return False
        M = region.inputs[0].shape[0]
        K = region.inputs[0].shape[1]
        N = region.inputs[1].shape[1]
        return (M, K, N) in SUPPORTED_SHAPES["gemm_fusion"]

    # Tile sizes supported by this template.
    # MMUL intrinsic constraint (npu2 i16, r=4, s=4, t=8):
    #   tile_m % r == 0, tile_k % s == 0, tile_n % t == 0
    # All three satisfy: 32%4=0 32%4=0 32%8=0 | 64%4=0 64%4=0 64%8=0 | 128%4=0 64%4=0 128%8=0
    TILE_SIZES: list[tuple[int, int, int]] = [
        (32, 32, 32),
        (64, 64, 64),
        (128, 64, 128),
    ]

    # MMUL intrinsic variants available on AIE2P (XDNA2 / Krackan).
    # "4x4x8": native AIE2P int16 shape via aie2p/mm.cc → matmul_vectorized_4x4x8_i16_i16,
    #          r=4, s=4, t=8; effective tile with 2×2 expansion: (2r)×s×(2t) = 8×4×16.
    #          AIE2 (non-P) kernel path (aie2/mm.cc) uses 4×4×4 for i16, which provides half
    #          the N-throughput per MMUL cycle; no native 8×2×8 i16 shape exists on AIE2P.
    MMUL_VARIANTS: list[str] = ["4x4x8"]

    def config_space(self, region: Region) -> list[Config]:
        M = region.inputs[0].shape[0]
        K = region.inputs[0].shape[1]
        N = region.inputs[1].shape[1]
        configs = []
        for m, k, n in self.TILE_SIZES:
            n_aie_cols = _get_n_aie_cols(M, N, m, n)
            n_cores = n_aie_cols * _N_AIE_ROWS
            for mmul_variant in self.MMUL_VARIANTS:
                for epilogue in ["none", "relu", "bias_add"]:
                    for prologue in ["none", "scale"]:
                        configs.append(Config(
                            tile=(m, k, n),
                            n_cores=n_cores,
                            extra={
                                "epilogue": epilogue,
                                "prologue": prologue,
                                "mmul_variant": mmul_variant,
                            },
                        ))
        return configs

    def lower(self, region: Region, config: Config) -> Callable:
        M = region.inputs[0].shape[0]
        K = region.inputs[0].shape[1]
        N = region.inputs[1].shape[1]
        m, k, n = config.tile
        n_aie_cols = config.n_cores // _N_AIE_ROWS
        epilogue = config.extra.get("epilogue", "none")
        prologue = config.extra.get("prologue", "none")
        bias_value = int(config.extra.get("bias_value", 0))
        alpha_value = int(config.extra.get("alpha_value", 1))
        return _build_iron_fn(
            M, K, N, m, k, n, n_aie_cols,
            epilogue, prologue, bias_value, alpha_value,
        )

    def estimated_cost(self, region: Region, config: Config) -> CostEstimate:
        M = region.inputs[0].shape[0]
        K = region.inputs[0].shape[1]
        N = region.inputs[1].shape[1]
        dispatch_floor = 100.0
        ops_per_us = 8600.0 * 1000.0
        compute_us = 2.0 * M * K * N / ops_per_us
        predicted_latency_us = dispatch_floor + compute_us
        predicted_gops = 2.0 * M * K * N / predicted_latency_us / 1000.0
        return CostEstimate(
            predicted_latency_us=predicted_latency_us,
            predicted_gops=predicted_gops,
            confidence=0.5,
        )
