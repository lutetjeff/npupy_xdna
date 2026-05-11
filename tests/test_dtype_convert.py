from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from npupy_xdna.dispatch.dtype_convert import PrecisionLossInfo, convert_for_template
from npupy_xdna.regions.region import ArraySpec, Region

EVIDENCE_DIR = Path("/home/lutet/ece511/npupy_xdna/.sisyphus/evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def _dummy_region() -> Region:
    spec = ArraySpec(shape=(4, 4), dtype="int16")
    return Region(op="elementwise_binary", inputs=[spec, spec], output=spec)


class TestF64ModerateValues:
    def test_converts_to_int16_correctly(self):
        arr = np.array([[1.0, -2.5, 100.0], [0.0, 32767.0, -32767.0]], dtype=np.float64)
        region = _dummy_region()

        converted, loss = convert_for_template(arr, region)

        assert converted.dtype == np.int16
        assert not loss.would_overflow
        assert not loss.did_clip
        np.testing.assert_array_equal(converted, np.array([[1, -2, 100], [0, 32767, -32767]], dtype=np.int16))

    def test_rounds_fractional_values(self):
        arr = np.array([0.4, 0.5, 1.4, 1.5, -0.5, -1.5], dtype=np.float64)
        region = _dummy_region()

        converted, loss = convert_for_template(arr, region)

        assert converted.dtype == np.int16
        assert not loss.would_overflow

    def test_max_abs_recorded_correctly(self):
        arr = np.array([10.0, -250.0, 123.0], dtype=np.float64)
        region = _dummy_region()

        _, loss = convert_for_template(arr, region)

        assert loss.max_abs_input == pytest.approx(250.0)
        assert not loss.would_overflow

    def test_f32_moderate_converts(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        region = _dummy_region()

        converted, loss = convert_for_template(arr, region)

        assert converted.dtype == np.int16
        assert not loss.would_overflow
        np.testing.assert_array_equal(converted, np.array([1, 2, 3], dtype=np.int16))


class TestF64Overflow:
    def test_returns_would_overflow_true(self):
        arr = np.array([1.0, 40000.0, -5.0], dtype=np.float64)
        region = _dummy_region()

        returned_arr, loss = convert_for_template(arr, region)

        assert loss.would_overflow
        assert loss.max_abs_input == pytest.approx(40000.0)

    def test_returns_original_array_unchanged_on_overflow(self):
        arr = np.array([50000.0, 1.0], dtype=np.float64)
        region = _dummy_region()

        returned_arr, loss = convert_for_template(arr, region)

        assert returned_arr is arr
        assert returned_arr.dtype == np.float64

    def test_exactly_at_boundary_does_not_overflow(self):
        arr = np.array([32767.0, -32767.0], dtype=np.float64)
        region = _dummy_region()

        converted, loss = convert_for_template(arr, region)

        assert not loss.would_overflow
        assert converted.dtype == np.int16

    def test_one_above_boundary_overflows(self):
        arr = np.array([32768.0], dtype=np.float64)
        region = _dummy_region()

        _, loss = convert_for_template(arr, region)

        assert loss.would_overflow

    def test_f32_overflow(self):
        arr = np.array([99999.0], dtype=np.float32)
        region = _dummy_region()

        returned_arr, loss = convert_for_template(arr, region)

        assert loss.would_overflow
        assert returned_arr is arr


class TestInt16Passthrough:
    def test_int16_returned_unchanged(self):
        arr = np.array([1, 2, 3], dtype=np.int16)
        region = _dummy_region()

        returned_arr, loss = convert_for_template(arr, region)

        assert returned_arr is arr
        assert not loss.would_overflow
        assert not loss.did_clip

    def test_int16_max_abs_recorded(self):
        arr = np.array([100, -500, 32000], dtype=np.int16)
        region = _dummy_region()

        _, loss = convert_for_template(arr, region)

        assert loss.max_abs_input == pytest.approx(32000.0)

    def test_int16_zeros_passthrough(self):
        arr = np.zeros((4, 4), dtype=np.int16)
        region = _dummy_region()

        returned_arr, loss = convert_for_template(arr, region)

        assert returned_arr is arr
        assert loss.max_abs_input == pytest.approx(0.0)


class TestUnsupportedDtype:
    def test_int32_raises_type_error(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        region = _dummy_region()

        with pytest.raises(TypeError, match="unsupported dtype"):
            convert_for_template(arr, region)

    def test_bool_raises_type_error(self):
        arr = np.array([True, False], dtype=bool)
        region = _dummy_region()

        with pytest.raises(TypeError):
            convert_for_template(arr, region)


class TestCacheHitLogging:
    def test_dispatcher_logs_cache_miss(self, tmp_path):
        from npupy_xdna.dispatch.dispatcher import Dispatcher
        from npupy_xdna.heuristic.offload import OffloadDecision
        from npupy_xdna.runtime.runner import RunResult

        A = np.array([[1, 2], [3, 4]], dtype=np.int16)
        B = np.array([[1, 0], [0, 1]], dtype=np.int16)

        disp = Dispatcher()
        disp._log_path = tmp_path / "dispatch.log"

        decision = OffloadDecision(
            action="offload",
            template="gemm_fusion",
            predicted_speedup=5.0,
            rationale="test",
        )
        disp.offload.decide = MagicMock(return_value=decision)
        disp.templates["gemm_fusion"].lower = MagicMock(return_value=MagicMock())
        disp.templates["gemm_fusion"].config_space = MagicMock(return_value=[MagicMock()])
        disp.npu_runner.run = MagicMock(
            return_value=RunResult(output=A @ B, latency_us=100.0, status="ok", device="npu")
        )

        disp.dispatch(np.matmul, (A, B), {})

        log_text = (tmp_path / "dispatch.log").read_text()
        assert "xclbin_cache_" in log_text

    def test_dispatcher_logs_cache_hit_when_file_exists(self, tmp_path):
        from npupy_xdna.dispatch.dispatcher import Dispatcher
        from npupy_xdna.heuristic.offload import OffloadDecision
        from npupy_xdna.runtime.iron_jit import XCLBIN_CACHE_DIR, _cache_key
        from npupy_xdna.runtime.runner import RunResult

        A = np.array([[1, 2], [3, 4]], dtype=np.int16)
        B = np.array([[1, 0], [0, 1]], dtype=np.int16)

        key = _cache_key("gemm_fusion", (2, 2))
        fake_xclbin = XCLBIN_CACHE_DIR / f"gemm_fusion_{key}.xclbin"
        XCLBIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fake_xclbin.write_bytes(b"fake")

        try:
            disp = Dispatcher()
            disp._log_path = tmp_path / "dispatch.log"

            decision = OffloadDecision(
                action="offload",
                template="gemm_fusion",
                predicted_speedup=5.0,
                rationale="test",
            )
            disp.offload.decide = MagicMock(return_value=decision)
            disp.templates["gemm_fusion"].lower = MagicMock(return_value=MagicMock())
            disp.templates["gemm_fusion"].config_space = MagicMock(return_value=[MagicMock()])
            disp.npu_runner.run = MagicMock(
                return_value=RunResult(output=A @ B, latency_us=100.0, status="ok", device="npu")
            )

            disp.dispatch(np.matmul, (A, B), {})

            log_text = (tmp_path / "dispatch.log").read_text()
            assert "xclbin_cache_hit" in log_text
        finally:
            fake_xclbin.unlink(missing_ok=True)


class TestDispatcherOverflowFallback:
    def test_f64_overflow_causes_cpu_fallback(self, tmp_path):
        from npupy_xdna.dispatch.dispatcher import Dispatcher
        from npupy_xdna.heuristic.offload import OffloadDecision

        A = np.array([[40000.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

        disp = Dispatcher()
        disp._log_path = tmp_path / "dispatch.log"

        decision = OffloadDecision(
            action="offload",
            template="gemm_fusion",
            predicted_speedup=5.0,
            rationale="test",
        )
        disp.offload.decide = MagicMock(return_value=decision)
        disp.templates["gemm_fusion"].config_space = MagicMock(return_value=[MagicMock()])

        result = disp.dispatch(np.matmul, (A, B), {})

        assert result is None
        log_text = (tmp_path / "dispatch.log").read_text()
        assert "dtype_overflow" in log_text
