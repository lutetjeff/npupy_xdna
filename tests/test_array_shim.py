from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from npupy_xdna.dispatch.array_shim import (
    SUPPORTED_FUNCS,
    NPUPyArray,
    activate,
    deactivate,
    dispatch_active,
    is_active,
)

EVIDENCE_DIR = Path("/home/lutet/ece511/npupy_xdna/.sisyphus/evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def ensure_deactivated():
    yield
    deactivate()


def _recording_dispatch(calls: list):
    def dispatch_fn(orig_fn, args, kwargs, *, info=None):
        calls.append(info)
        return orig_fn(*args, **kwargs)
    return dispatch_fn


def test_intercept_matmul():
    calls: list = []
    A = np.ones((3, 4), dtype=np.float64)
    B = np.ones((4, 5), dtype=np.float64)

    with dispatch_active(_recording_dispatch(calls)):
        result = np.matmul(A, B)

    assert len(calls) == 1
    assert calls[0]["func"] == "matmul"
    np.testing.assert_array_equal(result, np.full((3, 5), 4.0))

    intercept_path = EVIDENCE_DIR / "task-20-intercept.txt"
    intercept_path.write_text(
        f"intercept OK: np.matmul intercepted, dispatch called {len(calls)} time(s)\n"
        f"info={calls[0]}\n"
        f"result shape={result.shape}\n"
    )


def test_intercept_add():
    calls: list = []
    A = np.ones((4,), dtype=np.float32)
    B = np.ones((4,), dtype=np.float32)

    with dispatch_active(_recording_dispatch(calls)):
        result = np.add(A, B)

    assert len(calls) == 1
    assert calls[0]["func"] == "add"
    np.testing.assert_array_almost_equal(result, np.full((4,), 2.0))


def test_intercept_multiply():
    calls: list = []
    A = np.array([1.0, 2.0, 3.0])
    B = np.array([4.0, 5.0, 6.0])

    with dispatch_active(_recording_dispatch(calls)):
        result = np.multiply(A, B)

    assert len(calls) == 1
    assert calls[0]["func"] == "multiply"
    np.testing.assert_array_almost_equal(result, np.array([4.0, 10.0, 18.0]))


def test_intercept_maximum():
    calls: list = []
    A = np.array([1.0, 5.0, 3.0])
    B = np.array([4.0, 2.0, 6.0])

    with dispatch_active(_recording_dispatch(calls)):
        result = np.maximum(A, B)

    assert len(calls) == 1
    assert calls[0]["func"] == "maximum"
    np.testing.assert_array_almost_equal(result, np.array([4.0, 5.0, 6.0]))


def test_fallthrough_unsupported_op():
    calls: list = []
    A = np.array([0.0, 1.0, np.pi / 2])

    with dispatch_active(_recording_dispatch(calls)):
        result = np.sin(A)

    assert len(calls) == 0
    np.testing.assert_array_almost_equal(result, np.array([0.0, np.sin(1.0), 1.0]))

    fallthrough_path = EVIDENCE_DIR / "task-20-fallthrough.txt"
    fallthrough_path.write_text(
        "fallthrough OK: np.sin not intercepted (not in SUPPORTED_FUNCS)\n"
        f"SUPPORTED_FUNCS={sorted(SUPPORTED_FUNCS)}\n"
        f"np.sin result={result}\n"
    )


def test_inactive_numpy_unaffected():
    A = np.ones((3, 3))
    B = np.ones((3, 3))
    result_before = np.matmul(A, B).copy()

    assert not is_active()
    result_inactive = np.matmul(A, B)
    np.testing.assert_array_equal(result_inactive, result_before)


def test_activate_deactivate_cycle():
    assert not is_active()

    activate()
    assert is_active()

    deactivate()
    assert not is_active()

    activate()
    assert is_active()

    deactivate()
    assert not is_active()


def test_activate_idempotent():
    calls: list = []
    dispatch = _recording_dispatch(calls)

    activate(dispatch)
    activate(dispatch)
    assert is_active()

    A = np.ones((2, 2))
    B = np.ones((2, 2))
    np.matmul(A, B)
    assert len(calls) == 1

    deactivate()
    assert not is_active()


def test_deactivate_restores_numpy_correctly():
    A = np.ones((4, 4))
    B = np.ones((4, 4))
    expected = np.matmul(A, B).copy()

    activate()
    deactivate()

    result = np.matmul(A, B)
    np.testing.assert_array_equal(result, expected)


def test_passthrough_default_dispatch_correct_results():
    A = np.array([[1, 2], [3, 4]], dtype=np.float64)
    B = np.array([[5, 6], [7, 8]], dtype=np.float64)
    expected = np.matmul(A, B).copy()

    with dispatch_active():
        result = np.matmul(A, B)

    np.testing.assert_array_equal(result, expected)


def test_info_contains_shape_and_dtype():
    calls: list = []
    A = np.ones((2, 3), dtype=np.float32)
    B = np.ones((3, 4), dtype=np.float32)

    with dispatch_active(_recording_dispatch(calls)):
        np.matmul(A, B)

    info = calls[0]
    assert info["func"] == "matmul"
    specs = info["arg_specs"]
    assert len(specs) == 2
    assert specs[0]["shape"] == (2, 3)
    assert specs[0]["dtype"] == "float32"
    assert specs[1]["shape"] == (3, 4)
    assert specs[1]["dtype"] == "float32"


def test_npupyarray_intercepts_ufunc_without_activate():
    calls: list = []
    A = NPUPyArray(np.array([1.0, 2.0, 3.0]))
    B = NPUPyArray(np.array([4.0, 5.0, 6.0]))

    from npupy_xdna.dispatch import array_shim
    original_fn = array_shim._dispatch_fn

    def recording(orig_fn, args, kwargs, *, info=None):
        calls.append(info)
        return orig_fn(*args, **kwargs)

    array_shim._dispatch_fn = recording
    try:
        result = np.add(A, B)
    finally:
        array_shim._dispatch_fn = original_fn

    assert len(calls) == 1
    assert calls[0]["func"] == "add"
    np.testing.assert_array_almost_equal(result, np.array([5.0, 7.0, 9.0]))


def test_npupyarray_fallthrough_unsupported_ufunc():
    A = NPUPyArray(np.array([0.0, 1.0, 2.0]))
    result = np.sqrt(A)
    np.testing.assert_array_almost_equal(result, np.array([0.0, 1.0, np.sqrt(2.0)]))
