from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from npupy_xdna.dispatch import deactivate
from npupy_xdna.dispatch import array_shim as _shim
from npupy_xdna.dispatch.dispatcher import Dispatcher
from npupy_xdna.dispatch.extract import numpy_op_to_region
from npupy_xdna.heuristic.offload import OffloadDecision
from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.runtime.runner import RunResult
from npupy_xdna.templates.col_independent import ColIndependentTemplate
from npupy_xdna.templates.protocol import Config

_NUM_COLS = 8
_N_WORKERS = 32


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


def _i16(size: int) -> np.ndarray:
    return np.random.default_rng(0).integers(-100, 100, size=size, dtype=np.int16)


def _col_indep_decision() -> OffloadDecision:
    return OffloadDecision(
        action="offload",
        template="col_independent",
        predicted_speedup=4.4,
        rationale="test",
    )


def _cpu_fallback_decision() -> OffloadDecision:
    return OffloadDecision(action="cpu_fallback", reason="test fallback")


def _ok_result(size: int) -> RunResult:
    return RunResult(
        output=np.zeros(size, dtype=np.int16),
        latency_us=865.0,
        status="ok",
        device="npu",
    )


class TestExtractTanh:
    def test_tanh_int16_produces_region_with_metadata(self):
        x = _i16(1048576)
        r = numpy_op_to_region(np.tanh, (x,), {})
        assert r is not None
        assert r.op == "elementwise_unary"
        assert r.inputs[0] == ArraySpec((1048576,), "int16")
        assert r.output == ArraySpec((1048576,), "int16")
        assert r.metadata["compute_fn"] == "tanh"
        assert r.metadata["compute_intensity"] == "high"

    def test_tanh_float64_returns_none(self):
        x = np.random.default_rng(0).random(100).astype(np.float64)
        r = numpy_op_to_region(np.tanh, (x,), {})
        assert r is None

    def test_tanh_float32_returns_none(self):
        x = np.ones(100, dtype=np.float32)
        r = numpy_op_to_region(np.tanh, (x,), {})
        assert r is None

    def test_tanh_empty_args_returns_none(self):
        r = numpy_op_to_region(np.tanh, (), {})
        assert r is None


class TestDispatchTanh:
    def test_tanh_int16_npu_success_returns_correct_shape(self, tmp_log):
        size = 1048576
        x = _i16(size)
        expected = np.zeros(size, dtype=np.int16)

        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_col_indep_decision())
        disp.templates["col_independent"].lower = MagicMock(return_value=MagicMock())
        disp.npu_runner.run = MagicMock(return_value=_ok_result(size))

        result = disp.dispatch(np.tanh, (x,), {})

        assert result is not None
        assert result.shape == (size,)
        assert result.dtype == np.int16
        disp.templates["col_independent"].lower.assert_called_once()

    def test_tanh_int16_log_contains_col_independent(self, tmp_log):
        size = 1048576
        x = _i16(size)

        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_col_indep_decision())
        disp.templates["col_independent"].lower = MagicMock(return_value=MagicMock())
        disp.npu_runner.run = MagicMock(return_value=_ok_result(size))

        disp.dispatch(np.tanh, (x,), {"info": {"func": "tanh"}})

        log = tmp_log.read_text()
        assert "col_independent" in log

    def test_tanh_float64_falls_through_dispatch(self, tmp_log):
        x = np.random.default_rng(0).random(1000).astype(np.float64)

        disp = _make_dispatcher(tmp_log)
        result = disp.dispatch(np.tanh, (x,), {})

        assert result is None

    def test_tanh_small_int16_cpu_fallback(self, tmp_log):
        x = _i16(100)

        disp = _make_dispatcher(tmp_log)
        result = disp.dispatch(np.tanh, (x,), {})

        assert result is None

    def test_tanh_activated_shim_routes_to_npu(self, tmp_log):
        size = 1048576
        x = _i16(size)
        expected = np.zeros(size, dtype=np.int16)

        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_col_indep_decision())
        cfg = Config(tile=(size,), n_cores=_N_WORKERS, extra={"chunk_per_pool": size // _NUM_COLS})
        disp.templates["col_independent"].config_space = MagicMock(return_value=[cfg])
        disp.templates["col_independent"].lower = MagicMock(return_value=MagicMock())
        disp.npu_runner.run = MagicMock(return_value=_ok_result(size))

        def shim_fn(orig_fn, args, kwargs, *, info=None):
            r = disp.dispatch(orig_fn, args, kwargs, info=info)
            if r is None:
                return orig_fn(*args, **kwargs)
            return r

        _shim.activate(shim_fn)
        result = np.tanh(x)

        assert result is not None
        assert result.shape == (size,)
        assert result.dtype == np.int16
        np.testing.assert_array_equal(result, expected)


class TestDispatchRegion:
    def test_dispatch_region_hash_routes_to_col_independent(self, tmp_log):
        size = 1048576
        x = _i16(size)

        region = Region(
            op="elementwise_unary",
            inputs=[ArraySpec((size,), "int16")],
            output=ArraySpec((size,), "int16"),
            metadata={"compute_fn": "hash", "compute_intensity": "high"},
        )

        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_col_indep_decision())
        disp.templates["col_independent"].lower = MagicMock(return_value=MagicMock())
        disp.npu_runner.run = MagicMock(return_value=_ok_result(size))

        result = disp.dispatch_region(region, [x])

        assert result is not None
        assert result.shape == (size,)
        assert result.dtype == np.int16
        disp.templates["col_independent"].lower.assert_called_once()

    def test_dispatch_region_hash_log_contains_col_independent(self, tmp_log):
        size = 1048576
        x = _i16(size)

        region = Region(
            op="elementwise_unary",
            inputs=[ArraySpec((size,), "int16")],
            output=ArraySpec((size,), "int16"),
            metadata={"compute_fn": "hash", "compute_intensity": "high"},
        )

        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_col_indep_decision())
        disp.templates["col_independent"].lower = MagicMock(return_value=MagicMock())
        disp.npu_runner.run = MagicMock(return_value=_ok_result(size))

        disp.dispatch_region(region, [x])

        log = tmp_log.read_text()
        assert "col_independent" in log

    def test_dispatch_region_cpu_fallback_returns_none(self, tmp_log):
        size = 1048576
        x = _i16(size)

        region = Region(
            op="elementwise_unary",
            inputs=[ArraySpec((size,), "int16")],
            output=ArraySpec((size,), "int16"),
            metadata={"compute_fn": "hash", "compute_intensity": "high"},
        )

        disp = _make_dispatcher(tmp_log)
        disp.offload.decide = MagicMock(return_value=_cpu_fallback_decision())

        result = disp.dispatch_region(region, [x])
        assert result is None


class TestHashOps:
    def test_hash_int16_cpu_fallback_on_dispatch_none(self):
        from npupy_xdna.ops import _cpu_hash, hash_int16

        x = _i16(1048576)
        with patch("npupy_xdna.dispatch.dispatcher.Dispatcher") as MockDisp:
            mock_disp = MagicMock()
            mock_disp.dispatch_region.return_value = None
            MockDisp.return_value = mock_disp
            result = hash_int16(x)

        assert result is not None
        assert result.shape == x.shape
        assert result.dtype == np.int16

    def test_hash_int16_wrong_dtype_raises(self):
        from npupy_xdna.ops import hash_int16

        with pytest.raises(TypeError):
            hash_int16(np.ones(100, dtype=np.float32))

    def test_cpu_hash_matches_kernel_reference(self):
        from npupy_xdna.ops import _cpu_hash

        FNV_OFFSET = np.uint16(0x811C)
        FNV_PRIME = np.uint32(0x0193)
        x = np.array([0, 1, -1, 100, -100], dtype=np.int16)
        result = _cpu_hash(x)
        assert result.shape == x.shape
        assert result.dtype == np.int16

        flat = x.view(np.uint16).copy()
        h = np.full(flat.shape, FNV_OFFSET, dtype=np.uint16)
        xv = flat.copy()
        for _ in range(8):
            h ^= xv
            h = (h.astype(np.uint32) * FNV_PRIME).astype(np.uint16)
            xv = (h >> np.uint16(1)).astype(np.uint16)
        expected = h.view(np.int16)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.npu
class TestKernelSelectionDistinct:
    def test_relu_tanh_hash_produce_distinct_outputs(self):
        from npupy_xdna.runtime.npu_runner import NpuRunner

        x = np.array([-100, -50, 50, 100, 1000, -1000, 200, -200] * 8192, dtype=np.int16)
        size = x.size

        outputs = {}
        for fn in ("relu", "tanh", "hash"):
            meta = {"compute_fn": fn} if fn != "relu" else {}
            region = Region(
                op="elementwise_unary",
                inputs=[ArraySpec((size,), "int16")],
                output=ArraySpec((size,), "int16"),
                metadata=meta,
            )
            tmpl = ColIndependentTemplate()
            cfg = tmpl.config_space(region)[0]
            iron_fn = tmpl.lower(region, cfg)
            runner = NpuRunner()
            result = runner.run(region, cfg, iron_fn, [x], timeout_s=60.0)
            assert result.status == "ok", f"{fn} NPU run failed: {result.status}"
            outputs[fn] = result.output[:8].copy()

        assert not np.array_equal(outputs["relu"], outputs["tanh"]), (
            "Kernel selection broken: relu and tanh produce identical output"
        )
        assert not np.array_equal(outputs["tanh"], outputs["hash"]), (
            "Kernel selection broken: tanh and hash produce identical output"
        )
