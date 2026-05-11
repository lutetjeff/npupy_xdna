from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Optional

import numpy as np

_DispatchFn = Callable[..., Any]

SUPPORTED_FUNCS: frozenset[str] = frozenset({"matmul", "add", "maximum", "multiply"})

_active: bool = False
_saved_np_attrs: dict[str, Any] = {}
_dispatch_fn: Optional[_DispatchFn] = None


def _numpy_passthrough(
    func: _DispatchFn,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    info: Optional[dict[str, Any]] = None,
) -> Any:
    return func(*args, **kwargs)


def _extract_info(func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    arg_specs = [
        {"shape": a.shape, "dtype": str(a.dtype)}
        for a in args
        if isinstance(a, np.ndarray)
    ]
    return {"func": func_name, "arg_specs": arg_specs}


def _make_interceptor(func_name: str, orig_fn: _DispatchFn) -> _DispatchFn:
    @functools.wraps(orig_fn)
    def _interceptor(*args: Any, **kwargs: Any) -> Any:
        fn = _dispatch_fn if _dispatch_fn is not None else _numpy_passthrough
        return fn(orig_fn, args, kwargs, info=_extract_info(func_name, args, kwargs))

    _interceptor.__wrapped__ = orig_fn  # type: ignore[attr-defined]
    return _interceptor


def activate(dispatch_fn: Optional[_DispatchFn] = None) -> None:
    """Install numpy overrides for SUPPORTED_FUNCS, routing through dispatch_fn.

    dispatch_fn signature: ``(orig_fn, args, kwargs, *, info) -> result``.
    Defaults to numpy passthrough. T22 supplies the real NPU dispatcher.
    """
    global _active, _dispatch_fn
    if _active:
        return

    _dispatch_fn = dispatch_fn if dispatch_fn is not None else _numpy_passthrough

    for name in SUPPORTED_FUNCS:
        orig = getattr(np, name)
        _saved_np_attrs[name] = orig
        setattr(np, name, _make_interceptor(name, orig))

    _active = True


def deactivate() -> None:
    global _active, _dispatch_fn
    if not _active:
        return

    for name, orig in _saved_np_attrs.items():
        setattr(np, name, orig)

    _saved_np_attrs.clear()
    _dispatch_fn = None
    _active = False


def is_active() -> bool:
    return _active


@contextlib.contextmanager
def dispatch_active(dispatch_fn: Optional[_DispatchFn] = None):
    activate(dispatch_fn)
    try:
        yield
    finally:
        deactivate()


class NPUPyArray(np.ndarray):
    def __new__(cls, input_array: Any) -> "NPUPyArray":
        return np.asarray(input_array).view(cls)

    def __array_finalize__(self, obj: object) -> None:
        pass

    def __array_function__(
        self,
        func: _DispatchFn,
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        fn_name = getattr(func, "__name__", "")
        if fn_name not in SUPPORTED_FUNCS:
            return NotImplemented

        dispatch = _dispatch_fn if _dispatch_fn is not None else _numpy_passthrough
        args_t = tuple(args)
        plain_args = tuple(np.asarray(a) if isinstance(a, NPUPyArray) else a for a in args_t)
        kw = dict(kwargs)
        orig_fn = getattr(func, "__wrapped__", func)
        return dispatch(orig_fn, plain_args, kw, info=_extract_info(fn_name, args_t, kw))

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        fn_name = getattr(ufunc, "__name__", "")
        if fn_name in SUPPORTED_FUNCS and method == "__call__":
            dispatch = _dispatch_fn if _dispatch_fn is not None else _numpy_passthrough
            plain_inputs = tuple(np.asarray(a) if isinstance(a, NPUPyArray) else a for a in inputs)
            return dispatch(ufunc, plain_inputs, kwargs, info=_extract_info(fn_name, inputs, kwargs))
        plain_inputs = tuple(a.view(np.ndarray) if isinstance(a, NPUPyArray) else a for a in inputs)
        return getattr(ufunc, method)(*plain_inputs, **kwargs)
