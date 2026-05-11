"""Empirical cost model from Wave-2 characterisation data.

Calibrated parameters:

Template        dispatch_floor  peak_perf
gemm_fusion       500 µs        5 159 GOPS  (at 2048³)
col_independent   300 µs       10.81 GB/s   (at 1 M elements)
compute_pool   15 000 µs        0.55 GB/s   (at 2 M elements; known design issue)
cgra              190 µs        N/A          (single measurement, dispatch-dominated)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from npupy_xdna.regions.region import Region
from npupy_xdna.templates.protocol import CostEstimate
from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES

_GEMM_DISPATCH_FLOOR_US: float = 500.0
_GEMM_PEAK_GOPS: float = 5159.0

_COL_INDEP_DISPATCH_FLOOR_US: float = 300.0
_COL_INDEP_PEAK_BW_GBPS: float = 10.8145

_COMPUTE_POOL_DISPATCH_FLOOR_US: float = 15_000.0
_COMPUTE_POOL_PEAK_BW_GBPS: float = 0.5472

_CGRA_LATENCY_US: float = 190.0

# int16 element size × (read + write) = 2 B × 2 = 4 B per element
_INT16_RW_BYTES: int = 4

_CPU_MATMUL_REF: list[tuple[int, float]] = [
    (256,  9_428.41),
    (512,  79_958.52),
    (1024, 1_267_345.16),
]

_CPU_ELEM_RATE_US_PER_ELEM: float = 2.80e-4

_TMPL_TO_SHAPE_KEY: dict[str, str] = {
    "gemm_fusion": "gemm_fusion",
    "col_independent": "col_indep",
    "compute_pool": "compute_pool",
    "cgra": "cgra",
}

_KNOWN_TEMPLATES: frozenset[str] = frozenset(_TMPL_TO_SHAPE_KEY)


class CostModel:
    """Predict NPU and CPU latency using Wave-2 empirical calibration.

    Usage::

        model = CostModel()
        est = model.predict("gemm_fusion", region)
        if est is not None and est.predicted_latency_us < model.cpu_predict(region):
            dispatch_to_npu(region)
    """

    def __init__(self, timings_dir: Optional[Path] = None) -> None:
        if timings_dir is None:
            timings_dir = Path(__file__).parents[1] / "results" / "timings"
        self._timings_dir = Path(timings_dir)
        self._raw: dict[str, list[dict]] = {}
        for key in ("gemm_fusion", "col_indep", "compute_pool", "cgra"):
            path = self._timings_dir / f"{key}.jsonl"
            if path.exists():
                with open(path) as fh:
                    self._raw[key] = [
                        json.loads(line) for line in fh if line.strip()
                    ]
            else:
                self._raw[key] = []

    def predict(
        self, template_name: str, region: Region
    ) -> Optional[CostEstimate]:
        """Return NPU latency estimate, or None if shape is outside SUPPORTED_SHAPES."""
        if template_name not in _KNOWN_TEMPLATES:
            return None
        if not self._shape_supported(template_name, region):
            return None

        if template_name == "gemm_fusion":
            return self._predict_gemm(region)
        if template_name == "col_independent":
            return self._predict_col_indep(region)
        if template_name == "compute_pool":
            return self._predict_compute_pool(region)
        if template_name == "cgra":
            return self._predict_cgra(region)
        return None

    def cpu_predict(self, region: Region) -> Optional[float]:
        """Estimate CPU latency in µs (scalar int16, no BLAS); None for unknown ops."""
        if region.op in ("matmul", "matmul_fused"):
            M, K = region.inputs[0].shape
            N = region.output.shape[1]
            return self._cpu_matmul_us(M, K, N)
        if region.op in (
            "elementwise_unary", "elementwise_binary", "chained_elementwise"
        ):
            n_elems = math.prod(region.output.shape)
            return n_elems * _CPU_ELEM_RATE_US_PER_ELEM
        return None

    def _predict_gemm(self, region: Region) -> CostEstimate:
        M, K = region.inputs[0].shape
        N = region.output.shape[1]
        ops = 2 * M * K * N
        # 1 GOPS = 10^9 ops/s = 10^3 ops/µs  →  peak_gops × 1000 = ops/µs
        throughput_us = ops / (_GEMM_PEAK_GOPS * 1_000.0)
        lat = max(_GEMM_DISPATCH_FLOOR_US, throughput_us)
        gops = ops / (lat * 1_000.0)
        confidence = min(1.0, throughput_us / _GEMM_DISPATCH_FLOOR_US)
        return CostEstimate(
            predicted_latency_us=lat,
            predicted_gops=gops,
            confidence=confidence,
        )

    def _predict_col_indep(self, region: Region) -> CostEstimate:
        n_elems = math.prod(region.output.shape)
        bw_us = (n_elems * _INT16_RW_BYTES) / (_COL_INDEP_PEAK_BW_GBPS * 1_000.0)
        lat = max(_COL_INDEP_DISPATCH_FLOOR_US, bw_us)
        eff_bw = (n_elems * _INT16_RW_BYTES) / (lat * 1_000.0)
        confidence = min(1.0, bw_us / _COL_INDEP_DISPATCH_FLOOR_US)
        return CostEstimate(
            predicted_latency_us=lat,
            predicted_gops=eff_bw,
            confidence=confidence,
        )

    def _predict_compute_pool(self, region: Region) -> CostEstimate:
        n_elems = math.prod(region.output.shape)
        bw_us = (n_elems * _INT16_RW_BYTES) / (
            _COMPUTE_POOL_PEAK_BW_GBPS * 1_000.0
        )
        lat = max(_COMPUTE_POOL_DISPATCH_FLOOR_US, bw_us)
        eff_bw = (n_elems * _INT16_RW_BYTES) / (lat * 1_000.0)
        confidence = min(1.0, bw_us / _COMPUTE_POOL_DISPATCH_FLOOR_US)
        return CostEstimate(
            predicted_latency_us=lat,
            predicted_gops=eff_bw,
            confidence=confidence,
        )

    def _predict_cgra(self, region: Region) -> CostEstimate:
        lat = _CGRA_LATENCY_US
        n_elems = math.prod(region.output.shape)
        eff_bw = (n_elems * 2) / (lat * 1_000.0)
        return CostEstimate(
            predicted_latency_us=lat,
            predicted_gops=eff_bw,
            confidence=0.6,
        )

    def _shape_supported(self, template_name: str, region: Region) -> bool:
        shape_key = _TMPL_TO_SHAPE_KEY[template_name]
        supported = SUPPORTED_SHAPES[shape_key]
        if template_name == "gemm_fusion":
            if region.op not in ("matmul", "matmul_fused"):
                return False
            M, K = region.inputs[0].shape
            N = region.output.shape[1]
            return (M, K, N) in supported
        else:
            n_elems = math.prod(region.output.shape)
            return n_elems in supported

    def _cpu_matmul_us(self, M: int, K: int, N: int) -> float:
        ops = 2 * M * K * N
        ref_n, ref_us = min(
            _CPU_MATMUL_REF,
            key=lambda r: abs(math.log(ops / (2 * r[0] ** 3))),
        )
        ref_ops = 2 * ref_n ** 3
        return ref_us * (ops / ref_ops)
