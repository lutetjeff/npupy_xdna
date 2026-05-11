from __future__ import annotations

import time

import numpy as np

from npupy_xdna.regions.region import Region
from npupy_xdna.runtime.runner import RunResult


class CpuRunner:
    device = "cpu"

    def run(self, region: Region, inputs: list[np.ndarray]) -> RunResult:
        t0 = time.perf_counter()
        if region.op == "matmul":
            if len(inputs) != 2:
                raise ValueError("matmul requires exactly 2 inputs")
            output = np.matmul(inputs[0], inputs[1])
        elif region.op == "matmul_fused":
            if len(inputs) != 2:
                raise ValueError("matmul_fused requires exactly 2 inputs")
            output = np.maximum(0, np.matmul(inputs[0], inputs[1]))
        elif region.op == "elementwise_unary":
            if len(inputs) != 1:
                raise ValueError("elementwise_unary requires exactly 1 input")
            output = np.maximum(0, inputs[0])
        elif region.op == "elementwise_binary":
            if len(inputs) != 2:
                raise ValueError("elementwise_binary requires exactly 2 inputs")
            output = inputs[0] + inputs[1]
        elif region.op == "chained_elementwise":
            if len(inputs) != 1:
                raise ValueError("chained_elementwise requires exactly 1 input")
            output = np.maximum(0, inputs[0] + 1)
        else:
            raise ValueError(f"Unsupported op {region.op!r}")
        t1 = time.perf_counter()
        return RunResult(
            output=output,
            latency_us=(t1 - t0) * 1_000_000,
            status="ok",
            device=self.device,
        )
