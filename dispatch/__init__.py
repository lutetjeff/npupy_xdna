from __future__ import annotations

import contextlib
from typing import Optional

from npupy_xdna.dispatch.dispatcher import Dispatcher
from npupy_xdna.dispatch import array_shim as _shim

_dispatcher: Optional[Dispatcher] = None


def _get_dispatcher() -> Dispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = Dispatcher()
    return _dispatcher


def _make_shim_fn(disp: Dispatcher):
    def _shim_dispatch(orig_fn, args, kwargs, *, info=None):
        result = disp.dispatch(orig_fn, args, kwargs, info=info)
        if result is None:
            return orig_fn(*args, **kwargs)
        return result
    return _shim_dispatch


def activate() -> None:
    _shim.activate(_make_shim_fn(_get_dispatcher()))


def deactivate() -> None:
    _shim.deactivate()


@contextlib.contextmanager
def dispatch_active():
    activate()
    try:
        yield
    finally:
        deactivate()


__all__ = ["Dispatcher", "activate", "deactivate", "dispatch_active"]
