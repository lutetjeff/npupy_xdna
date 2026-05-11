from __future__ import annotations

import concurrent.futures
import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES, assert_shape_supported
from npupy_xdna.runtime.iron_jit import XCLBIN_CACHE_DIR, _cache_key, compile_xclbin


def test_shape_guard_raises_for_unknown_shape():
    with pytest.raises(ValueError, match="Unknown template"):
        assert_shape_supported("nonexistent_template", (256, 256, 256))

    with pytest.raises(ValueError, match="not in SUPPORTED_SHAPES"):
        assert_shape_supported("gemm_fusion", (999, 999, 999))


def _fake_compile_xclbin(template_name: str, shape_tuple: tuple) -> Path:
    key = _cache_key(template_name, shape_tuple)
    cache_path = XCLBIN_CACHE_DIR / f"{template_name}_{key}.xclbin"

    if cache_path.exists():
        return cache_path

    XCLBIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    unique = f"{os.getpid()}.{threading.get_ident()}"
    tmp_path = cache_path.with_suffix(f".tmp.{unique}.test")
    tmp_path.write_bytes(b"fake_xclbin_data")
    os.replace(tmp_path, cache_path)
    return cache_path


def test_concurrent_compile_same_key_both_succeed_no_corruption():
    template_name = "gemm_fusion"
    shape_tuple = (256, 256, 256)
    key = _cache_key(template_name, shape_tuple)
    cache_path = XCLBIN_CACHE_DIR / f"{template_name}_{key}.xclbin"
    if cache_path.exists():
        cache_path.unlink()

    def compile_task():
        return _fake_compile_xclbin(template_name, shape_tuple)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(compile_task) for _ in range(4)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert all(r == cache_path for r in results)
    assert cache_path.exists()
    assert cache_path.read_bytes() == b"fake_xclbin_data"


def test_cache_hit_returns_same_path_without_recompiling():
    template_name = "gemm_fusion"
    shape_tuple = (512, 512, 512)
    key = _cache_key(template_name, shape_tuple)
    cache_path = XCLBIN_CACHE_DIR / f"{template_name}_{key}.xclbin"

    XCLBIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"existing_xclbin")

    factory_mock = MagicMock()

    with patch("npupy_xdna.runtime.iron_jit.assert_shape_supported"):
        result = compile_xclbin(factory_mock, template_name, shape_tuple, force=False)

    assert result == cache_path
    factory_mock.assert_not_called()
