from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from npupy_xdna.dispatch import activate, deactivate, dispatch_active
from npupy_xdna.dispatch.correctness_gate import verify_correctness
from npupy_xdna.dispatch.dispatcher import Dispatcher
from npupy_xdna.heuristic.offload import OffloadDecision
from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.runtime.runner import RunResult

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


class TestVerifyCorrectness:
    def test_int16_exact_match_returns_true(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int16)
        region = Region(
            op="matmul",
            inputs=[ArraySpec(shape=(2, 2), dtype="int16"), ArraySpec(shape=(2, 2), dtype="int16")],
            output=ArraySpec(shape=(2, 2), dtype="int16"),
        )
        assert verify_correctness(a, a.copy(), region) is True

    def test_int16_mismatch_returns_false(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int16)
        b = np.array([[1, 2], [3, 5]], dtype=np.int16)
        region = Region(
            op="matmul",
            inputs=[ArraySpec(shape=(2, 2), dtype="int16"), ArraySpec(shape=(2, 2), dtype="int16")],
            output=ArraySpec(shape=(2, 2), dtype="int16"),
        )
        assert verify_correctness(a, b, region) is False

    def test_non_int16_returns_false(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int16)
        region = Region(
            op="matmul",
            inputs=[ArraySpec(shape=(2, 2), dtype="int16"), ArraySpec(shape=(2, 2), dtype="int16")],
            output=ArraySpec(shape=(2, 2), dtype="int16"),
        )
        region = MagicMock(wraps=region)
        region.output.dtype = "float32"
        assert verify_correctness(a, a.copy(), region) is False


class TestFallbackPaths:
    def test_region_not_classified_returns_none(self, tmp_log):
        A, B = _matmul_inputs(256)
        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_cpu_fallback_decision())

        result = disp.dispatch(np.matmul, (A, B), {})

        assert result is None
        content = tmp_log.read_text()
        assert "cpu_fallback" in content

    def test_cost_model_says_npu_loses_returns_none(self, tmp_log):
        A, B = _matmul_inputs(256)
        disp = _make_dispatcher(tmp_log)

        slow_decision = OffloadDecision(
            action="cpu_fallback",
            reason="predicted speedup 0.50x below threshold",
        )
        disp.offload.decide = MagicMock(return_value=slow_decision)

        result = disp.dispatch(np.matmul, (A, B), {})

        assert result is None
        content = tmp_log.read_text()
        assert "cpu_fallback" in content

    def test_dtype_overflow_f64_returns_none(self, tmp_log):
        rng = np.random.default_rng(2)
        A = rng.random((32, 32)).astype(np.float64) * 40000
        B = rng.random((32, 32)).astype(np.float64) * 40000

        disp = _make_dispatcher(tmp_log)
        result = disp.dispatch(np.matmul, (A, B), {})

        assert result is None
        content = tmp_log.read_text()
        assert "dtype_overflow" in content or "unsupported_op" in content or "extract_error" in content

    def test_shape_not_in_supported_shapes_returns_none(self, tmp_log):
        rng = np.random.default_rng(3)
        A = rng.integers(-10, 10, size=(100, 100), dtype=np.int16)
        B = rng.integers(-10, 10, size=(100, 100), dtype=np.int16)

        disp = _make_dispatcher(tmp_log)
        result = disp.dispatch(np.matmul, (A, B), {})

        assert result is None
        content = tmp_log.read_text()
        assert "cpu_fallback" in content

    def test_npu_runner_exception_returns_none_no_crash(self, tmp_log):
        A, B = _matmul_inputs(256)
        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_offload_decision())
        disp.templates["gemm_fusion"].lower = MagicMock(side_effect=RuntimeError("NPU crashed"))

        result = disp.dispatch(np.matmul, (A, B), {})

        assert result is None
        content = tmp_log.read_text()
        assert "npu_error" in content

    def test_all_fallback_paths_result_is_correct_numpy_reference(self, tmp_log):
        A, B = _matmul_inputs(256)
        expected = np.matmul(A, B)

        with dispatch_active():
            result = np.matmul(A, B)

        np.testing.assert_array_equal(result, expected)

    def test_dispatch_log_records_fallback_reason_for_each_path(self, tmp_log):
        A, B = _matmul_inputs(256)
        disp = _make_dispatcher(tmp_log)

        disp.offload.decide = MagicMock(return_value=_cpu_fallback_decision())
        disp.dispatch(np.matmul, (A, B), {})
        content = tmp_log.read_text()
        assert "cpu_fallback" in content

        tmp_log.unlink(missing_ok=True)

        tmp_log.unlink(missing_ok=True)

        A3, B3 = _matmul_inputs(256)
        disp.offload.decide = MagicMock(return_value=_offload_decision())
        disp.templates["gemm_fusion"].lower = MagicMock(side_effect=RuntimeError("boom"))
        disp.dispatch(np.matmul, (A3, B3), {})
        content3 = tmp_log.read_text()
        assert "npu_error" in content3

    def test_unsupported_op_fallback_logs_reason(self, tmp_log):
        disp = _make_dispatcher(tmp_log)
        arr = np.ones(4, dtype=np.float32)
        disp.dispatch(np.sin, (arr,), {})

        content = tmp_log.read_text()
        assert "unsupported_op" in content

    def test_empty_config_space_fallback_logs_reason(self, tmp_log):
        A, B = _matmul_inputs(256)
        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_offload_decision())
        disp.templates["gemm_fusion"].config_space = MagicMock(return_value=[])

        result = disp.dispatch(np.matmul, (A, B), {})

        assert result is None
        content = tmp_log.read_text()
        assert "empty_config_space" in content

    def test_npu_failed_result_fallback_logs_reason(self, tmp_log):
        A, B = _matmul_inputs(256)
        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_offload_decision())
        disp.templates["gemm_fusion"].lower = MagicMock(return_value=MagicMock())
        disp.npu_runner.run = MagicMock(return_value=_failed_run_result())

        result = disp.dispatch(np.matmul, (A, B), {})

        assert result is None
        content = tmp_log.read_text()
        assert "npu_failed" in content
