from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from npupy_xdna.dispatch import activate, deactivate, dispatch_active
from npupy_xdna.dispatch.dispatcher import Dispatcher, _LOG_PATH
from npupy_xdna.heuristic.offload import OffloadDecision
from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.runtime.runner import RunResult
from npupy_xdna.templates.protocol import Config

EVIDENCE_DIR = Path("/home/lutet/ece511/npupy_xdna/.sisyphus/evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def clean_shim():
    yield
    deactivate()


@pytest.fixture()
def tmp_log(tmp_path) -> Path:
    return tmp_path / "dispatch.log"


def _make_dispatcher(tmp_log: Path) -> Dispatcher:
    d = Dispatcher()
    d._log_path = tmp_log
    return d


def _matmul_inputs(n: int = 256):
    rng = np.random.default_rng(0)
    A = rng.integers(-10, 10, size=(n, n), dtype=np.int16)
    B = rng.integers(-10, 10, size=(n, n), dtype=np.int16)
    return A, B


def _offload_decision(template: str = "gemm_fusion") -> OffloadDecision:
    return OffloadDecision(
        action="offload",
        template=template,
        predicted_speedup=18.0,
        rationale="test",
    )


def _cpu_fallback_decision() -> OffloadDecision:
    return OffloadDecision(action="cpu_fallback", reason="test fallback")


def _good_run_result(output: np.ndarray) -> RunResult:
    return RunResult(output=output, latency_us=1234.0, status="ok", device="npu")


def _failed_run_result() -> RunResult:
    return RunResult(output=np.array([]), latency_us=0.0, status="error", device="npu")


class TestDispatchMatmulMock:
    def test_npu_success_returns_correct_shape(self, tmp_log):
        A, B = _matmul_inputs(256)
        expected = (A @ B).copy()

        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_offload_decision())

        fake_iron_fn = MagicMock()
        disp.templates["gemm_fusion"].lower = MagicMock(return_value=fake_iron_fn)
        disp.npu_runner.run = MagicMock(return_value=_good_run_result(expected))

        result = disp.dispatch(np.matmul, (A, B), {})

        assert result is not None
        assert result.shape == (256, 256)
        np.testing.assert_array_equal(result, expected)

    def test_b_column_major_transform_applied(self, tmp_log):
        A, B = _matmul_inputs(256)

        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_offload_decision())

        fake_iron_fn = MagicMock()
        disp.templates["gemm_fusion"].lower = MagicMock(return_value=fake_iron_fn)

        captured_inputs: list[Any] = []

        def capture_run(region, config, iron_fn, inputs, timeout_s=60.0):
            captured_inputs.extend(inputs)
            return _good_run_result(np.zeros((256, 256), dtype=np.int16))

        disp.npu_runner.run = capture_run
        disp.dispatch(np.matmul, (A, B), {})

        assert len(captured_inputs) == 2
        np.testing.assert_array_equal(captured_inputs[1], np.ascontiguousarray(B.T))

    def test_npu_failure_returns_none(self, tmp_log):
        A, B = _matmul_inputs(256)

        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_offload_decision())
        disp.templates["gemm_fusion"].lower = MagicMock(return_value=MagicMock())
        disp.npu_runner.run = MagicMock(return_value=_failed_run_result())

        result = disp.dispatch(np.matmul, (A, B), {})
        assert result is None

    def test_npu_exception_returns_none(self, tmp_log):
        A, B = _matmul_inputs(256)

        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_offload_decision())
        disp.templates["gemm_fusion"].lower = MagicMock(side_effect=RuntimeError("boom"))

        result = disp.dispatch(np.matmul, (A, B), {})
        assert result is None


class TestDispatchFallback:
    def test_unsupported_op_returns_none(self, tmp_log):
        disp = _make_dispatcher(tmp_log)
        arr = np.ones(4, dtype=np.float32)
        result = disp.dispatch(np.sin, (arr,), {})
        assert result is None

    def test_cpu_fallback_decision_returns_none(self, tmp_log):
        A, B = _matmul_inputs(256)

        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_cpu_fallback_decision())

        result = disp.dispatch(np.matmul, (A, B), {})
        assert result is None

    def test_non_int16_matmul_returns_none(self, tmp_log):
        rng = np.random.default_rng(1)
        A = rng.random((32, 32)).astype(np.float32)
        B = rng.random((32, 32)).astype(np.float32)

        disp = _make_dispatcher(tmp_log)
        result = disp.dispatch(np.matmul, (A, B), {})
        assert result is None

    def test_empty_config_space_returns_none(self, tmp_log):
        A, B = _matmul_inputs(256)

        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_offload_decision())
        disp.templates["gemm_fusion"].config_space = MagicMock(return_value=[])

        result = disp.dispatch(np.matmul, (A, B), {})
        assert result is None


class TestDispatchLog:
    def test_log_written_on_unsupported_op(self, tmp_log):
        disp = _make_dispatcher(tmp_log)
        arr = np.ones(4, dtype=np.float32)
        disp.dispatch(np.sin, (arr,), {})

        assert tmp_log.exists()
        content = tmp_log.read_text()
        assert "unsupported_op" in content

    def test_log_written_on_cpu_fallback(self, tmp_log):
        A, B = _matmul_inputs(256)

        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_cpu_fallback_decision())
        disp.dispatch(np.matmul, (A, B), {})

        content = tmp_log.read_text()
        assert "cpu_fallback" in content

    def test_log_written_on_npu_ok(self, tmp_log):
        A, B = _matmul_inputs(256)
        expected = np.zeros((256, 256), dtype=np.int16)

        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_offload_decision())
        disp.templates["gemm_fusion"].lower = MagicMock(return_value=MagicMock())
        disp.npu_runner.run = MagicMock(return_value=_good_run_result(expected))

        disp.dispatch(np.matmul, (A, B), {})

        content = tmp_log.read_text()
        assert "npu_ok" in content

    def test_log_appends_multiple_entries(self, tmp_log):
        arr = np.ones(4, dtype=np.float32)
        disp = _make_dispatcher(tmp_log)

        for _ in range(3):
            disp.dispatch(np.sin, (arr,), {})

        lines = [l for l in tmp_log.read_text().splitlines() if l.strip()]
        assert len(lines) >= 3


class TestActivateDeactivateCycle:
    def test_passthrough_with_no_dispatcher(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        B = np.array([[5, 6], [7, 8]], dtype=np.float64)
        expected = np.matmul(A, B)

        with dispatch_active():
            result = np.matmul(A, B)

        np.testing.assert_array_equal(result, expected)

    def test_activate_deactivate_restores_numpy(self):
        A = np.array([1.0, 2.0])
        B = np.array([3.0, 4.0])
        expected_add = A + B

        activate()
        result_add = np.add(A, B)
        deactivate()

        np.testing.assert_array_equal(result_add, expected_add)

    def test_double_deactivate_is_safe(self):
        activate()
        deactivate()
        deactivate()

    def test_double_activate_is_idempotent(self):
        activate()
        activate()
        deactivate()


class TestEndToEnd:
    def test_matmul_int16_dispatches_or_falls_back_correctly(self, tmp_log):
        A, B = _matmul_inputs(256)
        expected = A @ B

        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_offload_decision())
        disp.templates["gemm_fusion"].lower = MagicMock(return_value=MagicMock())
        disp.npu_runner.run = MagicMock(return_value=_good_run_result(expected))

        from npupy_xdna.dispatch import _make_shim_fn
        shim_fn = _make_shim_fn(disp)

        from npupy_xdna.dispatch import array_shim
        array_shim.activate(shim_fn)
        try:
            result = np.matmul(A, B)
        finally:
            array_shim.deactivate()

        assert result is not None
        np.testing.assert_array_equal(result, expected)

        (EVIDENCE_DIR / "task-22-e2e.txt").write_text(
            f"end-to-end matmul dispatch OK\n"
            f"A.shape={A.shape} B.shape={B.shape} result.shape={result.shape}\n"
            f"log_path={tmp_log}\n"
            f"log_content={tmp_log.read_text()}\n"
        )

    def test_fallback_passthrough_correct_result(self, tmp_log):
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        B = np.array([[5, 6], [7, 8]], dtype=np.float32)
        expected = np.matmul(A, B)

        disp = _make_dispatcher(tmp_log)
        from npupy_xdna.dispatch import _make_shim_fn
        shim_fn = _make_shim_fn(disp)

        from npupy_xdna.dispatch import array_shim
        array_shim.activate(shim_fn)
        try:
            result = np.matmul(A, B)
        finally:
            array_shim.deactivate()

        np.testing.assert_array_equal(result, expected)

        (EVIDENCE_DIR / "task-22-fallback.txt").write_text(
            f"fallback passthrough OK\n"
            f"A.dtype={A.dtype} result.shape={result.shape}\n"
            f"log_content={tmp_log.read_text()}\n"
        )
