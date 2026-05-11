from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Callable

from npupy_xdna.templates.shape_matrix import assert_shape_supported

XCLBIN_CACHE_DIR = Path("/home/lutet/ece511/npupy_xdna/results/xclbin_cache")


def _cache_key(template_name: str, shape_tuple: tuple) -> str:
    raw = f"{template_name}_{shape_tuple}_int16".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def compile_xclbin(
    iron_fn_factory: Callable,
    template_name: str,
    shape_tuple: tuple,
    force: bool = False,
) -> Path:
    assert_shape_supported(template_name, shape_tuple)
    XCLBIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    key = _cache_key(template_name, shape_tuple)
    cache_path = XCLBIN_CACHE_DIR / f"{template_name}_{key}.xclbin"

    if cache_path.exists() and not force:
        return cache_path

    import aie.iron as iron

    iron_fn = iron_fn_factory()

    @iron.jit(is_placed=False)
    def _wrapped(*args):
        return iron_fn(*args)

    tmp_path = cache_path.with_suffix(f".tmp.{os.getpid()}")
    try:
        _wrapped.__iron_compile__(output=str(tmp_path))
        os.replace(tmp_path, cache_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return cache_path
