from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class PreflightReport:
    status: Literal["SAFE", "FAIL"]
    checks: dict[str, str]
    failure_reason: Optional[str] = None


def preflight_check(
    template_name: str,
    shape,
    dtype: str,
    mlir_path: Path,
    xclbin_path: Path,
    host_buf_sizes: list[int],
) -> PreflightReport:
    checks: dict[str, str] = {}
    failure_reason: Optional[str] = None

    mlir_text = Path(mlir_path).read_text() if Path(mlir_path).exists() else ""

    check1_ok = "aie.device(npu2)" in mlir_text
    checks["mlir_target"] = "ok" if check1_ok else "FAIL: aie.device(npu2) not found"
    if not check1_ok:
        failure_reason = checks["mlir_target"]

    shape_dims: list[int] = list(shape) if isinstance(shape, (tuple, list)) else [shape]
    found_dims = set(int(m) for m in re.findall(r"memref<(\d+)", mlir_text))
    missing = [d for d in shape_dims if d not in found_dims]
    check2_ok = len(missing) == 0
    checks["shape_binding"] = "ok" if check2_ok else f"FAIL: dims {missing} missing in memref patterns"
    if not check2_ok and failure_reason is None:
        failure_reason = checks["shape_binding"]

    bytes_per_element = 2 if dtype == "int16" else 4
    if isinstance(shape, (tuple, list)) and len(shape) == 3:
        M, K, N = shape
        expected_bytes = (M * K + K * N + M * N) * bytes_per_element
    else:
        n = shape if isinstance(shape, int) else shape[0]
        expected_bytes = n * bytes_per_element
    total_provided = sum(host_buf_sizes)
    check3_ok = total_provided >= expected_bytes
    checks["host_buf_size"] = "ok" if check3_ok else f"FAIL: provided {total_provided}B < required {expected_bytes}B"
    if not check3_ok and failure_reason is None:
        failure_reason = checks["host_buf_size"]

    if template_name == "gemm_fusion":
        check4_ok = "b_col_maj" in mlir_text
        checks["b_col_maj"] = "ok" if check4_ok else "FAIL: b_col_maj marker absent"
        if not check4_ok and failure_reason is None:
            failure_reason = checks["b_col_maj"]
    else:
        checks["b_col_maj"] = "skipped"

    checks["smallest_size_trial"] = "skipped"

    overall = "SAFE" if failure_reason is None else "FAIL"
    return PreflightReport(status=overall, checks=checks, failure_reason=failure_reason)
