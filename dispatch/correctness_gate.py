from __future__ import annotations

import numpy as np

from npupy_xdna.regions.region import Region


def verify_correctness(
    npu_output: np.ndarray,
    cpu_output: np.ndarray,
    region: Region,
) -> bool:
    if region.output.dtype == "int16":
        return bool(np.array_equal(npu_output, cpu_output))
    return False
