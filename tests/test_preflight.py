from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from npupy_xdna.runtime.preflight import preflight_check


def _write_mlir(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".mlir", delete=False, mode="w")
    f.write(content)
    f.flush()
    return Path(f.name)


VALID_MLIR = """
module {
  aie.device(npu2) {
    func.func @matmul(%arg0: memref<256x256xi16>, %arg1: memref<256x256xi16>, %arg2: memref<256x256xi16>) {
      return
    }
  }
  b_col_maj
}
"""

WRONG_TARGET_MLIR = """
module {
  aie.device(npu) {
    func.func @matmul(%arg0: memref<256x256xi16>, %arg1: memref<256x256xi16>) {
      return
    }
  }
}
"""

SHAPE_MISMATCH_MLIR = """
module {
  aie.device(npu2) {
    func.func @matmul(%arg0: memref<128x128xi16>) {
      return
    }
  }
  b_col_maj
}
"""


def test_mlir_target_check_catches_npu_vs_npu2():
    mlir_path = _write_mlir(WRONG_TARGET_MLIR)
    shape = (256, 256, 256)
    buf_sizes = [256 * 256 * 2, 256 * 256 * 2, 256 * 256 * 2]
    report = preflight_check("gemm_fusion", shape, "int16", mlir_path, Path("/tmp/test.xclbin"), buf_sizes)
    assert report.status == "FAIL"
    assert "FAIL" in report.checks["mlir_target"]


def test_shape_mismatch_caught():
    mlir_path = _write_mlir(SHAPE_MISMATCH_MLIR)
    shape = (256, 256, 256)
    buf_sizes = [256 * 256 * 2, 256 * 256 * 2, 256 * 256 * 2]
    report = preflight_check("gemm_fusion", shape, "int16", mlir_path, Path("/tmp/test.xclbin"), buf_sizes)
    assert report.status == "FAIL"
    assert "FAIL" in report.checks["shape_binding"]


def test_undersized_buffer_caught():
    mlir_path = _write_mlir(VALID_MLIR)
    shape = (256, 256, 256)
    tiny_buf = [1, 1, 1]
    report = preflight_check("gemm_fusion", shape, "int16", mlir_path, Path("/tmp/test.xclbin"), tiny_buf)
    assert report.status == "FAIL"
    assert "FAIL" in report.checks["host_buf_size"]


def test_safe_for_valid_inputs():
    mlir_path = _write_mlir(VALID_MLIR)
    shape = (256, 256, 256)
    M, K, N = shape
    buf_sizes = [M * K * 2, K * N * 2, M * N * 2]
    report = preflight_check("gemm_fusion", shape, "int16", mlir_path, Path("/tmp/test.xclbin"), buf_sizes)
    assert report.status == "SAFE"
    assert report.failure_reason is None
    assert report.checks["mlir_target"] == "ok"
    assert report.checks["shape_binding"] == "ok"
    assert report.checks["host_buf_size"] == "ok"
    assert report.checks["smallest_size_trial"] == "skipped"
