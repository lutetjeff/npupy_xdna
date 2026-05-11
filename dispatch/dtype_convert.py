from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from npupy_xdna.regions.region import Region

_INT16_MAX = 32767
_INT16_MIN = -32768


@dataclass
class PrecisionLossInfo:
    max_abs_input: float
    did_clip: bool
    would_overflow: bool


def convert_for_template(
    arr: np.ndarray,
    region: Optional[Region] = None,  # noqa: ARG001
) -> tuple[np.ndarray, PrecisionLossInfo]:
    if arr.dtype == np.int16:
        # Fast path: skip expensive max/min scan for already-int16 arrays
        return arr, PrecisionLossInfo(
            max_abs_input=0.0,
            did_clip=False,
            would_overflow=False,
        )

    if arr.dtype in (np.float64, np.float32):
        max_abs = float(max(arr.max(), -arr.min()))
        if max_abs > _INT16_MAX:
            return arr, PrecisionLossInfo(
                max_abs_input=max_abs,
                did_clip=False,
                would_overflow=True,
            )
        converted = np.round(arr).astype(np.int16)
        return converted, PrecisionLossInfo(
            max_abs_input=max_abs,
            did_clip=False,
            would_overflow=False,
        )

    raise TypeError(
        f"convert_for_template: unsupported dtype {arr.dtype!r}; "
        "only int16, float32, float64 are accepted."
    )
