from __future__ import annotations

import threading
import time

import numpy as np

from npupy_xdna.regions.region import Region
from npupy_xdna.runtime.npu_lock import npu_exclusive
from npupy_xdna.runtime.runner import RunResult
from npupy_xdna.templates.protocol import Config


class NpuRunner:
    device = "npu"

    def run(
        self,
        region: Region,
        config: Config,
        iron_fn,
        inputs: list[np.ndarray],
        timeout_s: float = 60.0,
    ) -> RunResult:
        import aie.iron as iron

        dtype_map = {"int16": np.int16}
        element_type = dtype_map[region.output.dtype]
        output_size = int(np.prod(region.output.shape))

        result_holder: list[RunResult] = []
        exc_holder: list[BaseException] = []

        def _run_iron():
            try:
                iron_inputs = [
                    iron.tensor(arr, dtype=element_type, device="npu")
                    for arr in inputs
                ]
                output_buf = iron.zeros(output_size, dtype=element_type, device="npu")
                t0 = time.perf_counter()
                iron_fn(*iron_inputs, output_buf)
                t1 = time.perf_counter()
                out_arr = np.array(output_buf.numpy(), copy=True).reshape(region.output.shape)
                result_holder.append(
                    RunResult(
                        output=out_arr,
                        latency_us=(t1 - t0) * 1_000_000,
                        status="ok",
                        device=self.device,
                    )
                )
            except BaseException as exc:
                exc_holder.append(exc)

        with npu_exclusive(timeout_s=timeout_s):
            worker = threading.Thread(target=_run_iron, daemon=True)
            worker.start()
            worker.join(timeout=timeout_s)

            if worker.is_alive():
                return RunResult(
                    output=np.array([]),
                    latency_us=timeout_s * 1_000_000,
                    status="timeout",
                    device=self.device,
                )

        if exc_holder:
            raise exc_holder[0]

        return result_holder[0]
