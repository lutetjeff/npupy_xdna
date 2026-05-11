from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from npupy_xdna.regions.region import Region
from npupy_xdna.templates.protocol import Config, CostEstimate
from npupy_xdna.templates.shape_matrix import assert_shape_supported

_KERNELS_DIR = Path(__file__).parent.parent / "kernels"
_N = 256


class CgraTemplate:
    name = "cgra"

    def match(self, region: Region) -> bool:
        return (
            region.op == "chained_elementwise"
            and len(region.inputs) == 4
            and region.output.shape == (_N,)
            and region.output.dtype == "int16"
            and all(inp.shape == (_N,) and inp.dtype == "int16" for inp in region.inputs)
        )

    def config_space(self, region: Region) -> list[Config]:
        assert_shape_supported("cgra", _N)
        return [Config(tile=(0, _N), n_cores=3)]

    def lower(self, region: Region, config: Config) -> Callable:
        import aie.iron as iron

        @iron.jit(is_placed=False)
        def cgra_pipeline(a, b, c, d, out):
            from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
            from aie.iron.placers import SequentialPlacer
            from aie.iron.device import NPU2, Tile
            from aie.utils.config import cxx_header_path

            n = _N
            tile_ty = np.ndarray[(n,), np.dtype[np.int16]]
            inc = [cxx_header_path()]

            add_fn = ExternalFunction(
                "cgra_add",
                source_file=str(_KERNELS_DIR / "cgra_add_int16.cc"),
                arg_types=[tile_ty, tile_ty, tile_ty],
                include_dirs=inc,
            )
            mul_fn = ExternalFunction(
                "cgra_mul",
                source_file=str(_KERNELS_DIR / "cgra_mul_int16.cc"),
                arg_types=[tile_ty, tile_ty, tile_ty],
                include_dirs=inc,
            )
            sub_fn = ExternalFunction(
                "cgra_sub",
                source_file=str(_KERNELS_DIR / "cgra_sub_int16.cc"),
                arg_types=[tile_ty, tile_ty, tile_ty],
                include_dirs=inc,
            )

            of_a = ObjectFifo(tile_ty, name="cgra_a", depth=2)
            of_b = ObjectFifo(tile_ty, name="cgra_b", depth=2)
            of_c = ObjectFifo(tile_ty, name="cgra_c", depth=2)
            of_d = ObjectFifo(tile_ty, name="cgra_d", depth=2)
            of_ab = ObjectFifo(tile_ty, name="cgra_ab", depth=2)
            of_abc = ObjectFifo(tile_ty, name="cgra_abc", depth=2)
            of_out = ObjectFifo(tile_ty, name="cgra_out", depth=2)

            def stage1(of_a, of_b, of_ab, add_fn):
                elem_a = of_a.acquire(1)
                elem_b = of_b.acquire(1)
                elem_ab = of_ab.acquire(1)
                add_fn(elem_a, elem_b, elem_ab)
                of_a.release(1)
                of_b.release(1)
                of_ab.release(1)

            def stage2(of_ab, of_c, of_abc, mul_fn):
                elem_ab = of_ab.acquire(1)
                elem_c = of_c.acquire(1)
                elem_abc = of_abc.acquire(1)
                mul_fn(elem_ab, elem_c, elem_abc)
                of_ab.release(1)
                of_c.release(1)
                of_abc.release(1)

            def stage3(of_abc, of_d, of_out, sub_fn):
                elem_abc = of_abc.acquire(1)
                elem_d = of_d.acquire(1)
                elem_out = of_out.acquire(1)
                sub_fn(elem_abc, elem_d, elem_out)
                of_abc.release(1)
                of_d.release(1)
                of_out.release(1)

            worker1 = Worker(
                stage1,
                fn_args=[of_a.cons(), of_b.cons(), of_ab.prod(), add_fn],
                placement=Tile(0, 2),
            )
            worker2 = Worker(
                stage2,
                fn_args=[of_ab.cons(), of_c.cons(), of_abc.prod(), mul_fn],
                placement=Tile(0, 3),
            )
            worker3 = Worker(
                stage3,
                fn_args=[of_abc.cons(), of_d.cons(), of_out.prod(), sub_fn],
                placement=Tile(0, 4),
            )

            rt = Runtime()
            with rt.sequence(tile_ty, tile_ty, tile_ty, tile_ty, tile_ty) as (A, B, C, D, OUT):
                rt.start(worker1, worker2, worker3)
                rt.fill(of_a.prod(), A)
                rt.fill(of_b.prod(), B)
                rt.fill(of_c.prod(), C)
                rt.fill(of_d.prod(), D)
                rt.drain(of_out.cons(), OUT, wait=True)

            return Program(NPU2(), rt).resolve_program(SequentialPlacer())

        return cgra_pipeline

    def estimated_cost(self, region: Region, config: Config) -> CostEstimate:
        return CostEstimate(
            predicted_latency_us=50.0,
            predicted_gops=0.001,
            confidence=0.1,
        )
