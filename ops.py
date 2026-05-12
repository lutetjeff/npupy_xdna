from __future__ import annotations

from typing import Optional

import numpy as np

from npupy_xdna.regions.region import ArraySpec, Region

_FNV_OFFSET = np.uint16(0x811C)
_FNV_PRIME = np.uint16(0x0193)
_FNV_PRIME_U32 = np.uint32(0x0193)

_dispatcher: Optional[object] = None


def _get_dispatcher():
    global _dispatcher
    if _dispatcher is None:
        from npupy_xdna.dispatch.dispatcher import Dispatcher

        _dispatcher = Dispatcher()
    return _dispatcher


def hash_int16(arr: np.ndarray) -> np.ndarray:
    if not (isinstance(arr, np.ndarray) and arr.dtype == np.int16):
        raise TypeError(
            f"hash_int16 requires int16 ndarray, got {getattr(arr, 'dtype', type(arr))}"
        )
    region = Region(
        op="elementwise_unary",
        inputs=[ArraySpec(arr.shape, "int16")],
        output=ArraySpec(arr.shape, "int16"),
        metadata={"compute_fn": "hash", "compute_intensity": "high"},
    )
    dispatcher = _get_dispatcher()
    info = {"func": "hash_int16", "arg_specs": [{"shape": arr.shape, "dtype": "int16"}]}
    result = dispatcher.dispatch_region(region, [arr], info=info)
    if result is not None:
        return result
    return _cpu_hash(arr)


def _cpu_hash(arr: np.ndarray) -> np.ndarray:
    flat = arr.ravel().view(np.uint16).copy()
    hash_arr = np.full(flat.shape, _FNV_OFFSET, dtype=np.uint16)
    x = flat.copy()
    for _ in range(8):
        hash_arr ^= x
        hash_arr = (hash_arr.astype(np.uint32) * _FNV_PRIME_U32).astype(np.uint16)
        x = (hash_arr >> np.uint16(1)).astype(np.uint16)
    return hash_arr.view(np.int16).reshape(arr.shape)
