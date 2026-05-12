from __future__ import annotations

import operator
from typing import Any, Optional

import numpy as np

from npupy_xdna.regions.region import ArraySpec, Region

_INT16 = "int16"


def numpy_op_to_region(
    numpy_func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Optional[Region]:
    func_name = getattr(numpy_func, "__name__", None)

    if numpy_func is operator.matmul or func_name == "matmul":
        return _build_matmul_region(args)

    if func_name == "add":
        return _build_elementwise_binary_region(args)

    if func_name == "multiply":
        return _build_elementwise_binary_region(args)

    if func_name == "maximum":
        return _build_maximum_region(args)

    if func_name == "tanh":
        return _build_tanh_region(args)

    return None


def _is_int16_array(obj: Any) -> bool:
    return isinstance(obj, np.ndarray) and str(obj.dtype) == _INT16


def _is_scalar_zero(obj: Any) -> bool:
    if isinstance(obj, (int, float)) and obj == 0:
        return True
    if isinstance(obj, (np.integer, np.floating)) and obj == 0:
        return True
    # 0-d numpy array
    if isinstance(obj, np.ndarray) and obj.ndim == 0 and obj.item() == 0:
        return True
    return False


def _spec(arr: np.ndarray) -> ArraySpec:
    return ArraySpec(shape=arr.shape, dtype=_INT16)


def _build_matmul_region(args: tuple[Any, ...]) -> Optional[Region]:
    if len(args) < 2:
        return None

    a, b = args[0], args[1]

    if not (_is_int16_array(a) and _is_int16_array(b)):
        return None

    if a.ndim != 2 or b.ndim != 2:
        return None

    if a.shape[1] != b.shape[0]:
        return None

    return Region(
        op="matmul",
        inputs=[_spec(a), _spec(b)],
        output=ArraySpec(shape=(a.shape[0], b.shape[1]), dtype=_INT16),
    )


def _build_elementwise_binary_region(args: tuple[Any, ...]) -> Optional[Region]:
    if len(args) < 2:
        return None

    a, b = args[0], args[1]

    if not (_is_int16_array(a) and _is_int16_array(b)):
        return None

    return Region(
        op="elementwise_binary",
        inputs=[_spec(a), _spec(b)],
        output=_spec(a),
    )


def _build_maximum_region(args: tuple[Any, ...]) -> Optional[Region]:
    if len(args) < 2:
        return None

    first, second = args[0], args[1]

    if _is_scalar_zero(first) and _is_int16_array(second):
        s = _spec(second)
        return Region(op="elementwise_unary", inputs=[s], output=s)

    if _is_int16_array(first) and _is_scalar_zero(second):
        s = _spec(first)
        return Region(op="elementwise_unary", inputs=[s], output=s)

    if _is_int16_array(first) and _is_int16_array(second):
        return Region(
            op="elementwise_binary",
            inputs=[_spec(first), _spec(second)],
            output=_spec(first),
        )

    return None


def _build_tanh_region(args: tuple[Any, ...]) -> Optional[Region]:
    if len(args) < 1:
        return None
    arr = args[0]
    if not _is_int16_array(arr):
        return None
    s = _spec(arr)
    return Region(
        op="elementwise_unary",
        inputs=[s],
        output=s,
        metadata={"compute_fn": "tanh", "compute_intensity": "high"},
    )
