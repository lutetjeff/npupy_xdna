"""Empirical cost model from Wave-2 characterisation data (V2).

Calibrated parameters:

Template        dispatch_floor  peak_perf       CI rel-std
gemm_fusion       500 µs        5 159 GOPS      10.1%   (10-pt calibration)
col_independent   300 µs       10.81 GB/s       14.6%   (6-pt calibration)
compute_pool   15 000 µs        0.55 GB/s        2.9%   (floor-dominated, stable)
cgra         depth-interp        N/A             18.0%   (3-depth sweep)
sliding_window    500 µs        1 000 GOPS      15.0%   (placeholder – no jsonl)
tanh              300 µs        5.95 GB/s        9.9%   (4-pt calibration)
hash              300 µs        3.00 GB/s       15.0%   (placeholder – no jsonl)
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Optional

from npupy_xdna.regions.region import Region
from npupy_xdna.templates.protocol import CostEstimate
from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES

_GEMM_DISPATCH_FLOOR_US: float = 500.0
_GEMM_PEAK_GOPS: float = 5159.0
_GEMM_CI_REL_STD: float = 0.101

_COL_INDEP_DISPATCH_FLOOR_US: float = 300.0
_COL_INDEP_PEAK_BW_GBPS: float = 10.8145
_COL_INDEP_CI_REL_STD: float = 0.146

_COMPUTE_POOL_DISPATCH_FLOOR_US: float = 15_000.0
_COMPUTE_POOL_PEAK_BW_GBPS: float = 0.5472
_COMPUTE_POOL_CI_REL_STD: float = 0.029

# CGRA: depth-interpolated from cgra_depth_sweep.jsonl (256 elements, depths 3/8/16).
# key = depth, value = median latency µs.
_CGRA_DEPTH_LATS_US: dict[int, float] = {3: 240.36, 8: 351.19, 16: 364.77}
_CGRA_CI_REL_STD: float = 0.180

_SW_DISPATCH_FLOOR_US: float = 500.0
_SW_PEAK_GOPS: float = 1_000.0
_SW_CI_REL_STD: float = 0.15

_TANH_DISPATCH_FLOOR_US: float = 300.0
_TANH_PEAK_BW_GBPS: float = 5.9516
_TANH_CI_REL_STD: float = 0.099

_HASH_DISPATCH_FLOOR_US: float = 300.0
_HASH_PEAK_BW_GBPS: float = 3.0
_HASH_CI_REL_STD: float = 0.15

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
    "sliding_window": "sliding_window",
    "tanh": "tanh",
    "hash": "hash",
}

_KNOWN_TEMPLATES: frozenset[str] = frozenset(_TMPL_TO_SHAPE_KEY)


def _ci_bounds(lat: float, rel_std: float) -> tuple[float, float]:
    half = 2.0 * rel_std * lat
    return max(0.0, lat - half), lat + half


class CostModel:
    """Predict NPU and CPU latency using Wave-2 empirical calibration (V2).

    Usage::

        model = CostModel()
        est = model.predict("gemm_fusion", region)
        if est is not None and est.predicted_latency_us < model.cpu_predict(region):
            dispatch_to_npu(region)

    CI fields (ci_low, ci_high) on CostEstimate are reporting-only;
    offload decisions always use predicted_latency_us (the median).
    """

    def __init__(self, timings_dir: Optional[Path] = None) -> None:
        if timings_dir is None:
            timings_dir = Path(__file__).parents[1] / "results" / "timings"
        self._timings_dir = Path(timings_dir)
        self._raw: dict[str, list[dict]] = {}
        for key in (
            "gemm_fusion", "col_indep", "compute_pool", "cgra",
            "cgra_depth_sweep", "tanh", "sliding_window", "hash",
        ):
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
        if template_name == "sliding_window":
            return self._predict_sliding_window(region)
        if template_name == "tanh":
            return self._predict_tanh(region)
        if template_name == "hash":
            return self._predict_hash(region)
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
        throughput_us = ops / (_GEMM_PEAK_GOPS * 1_000.0)
        lat = max(_GEMM_DISPATCH_FLOOR_US, throughput_us)
        gops = ops / (lat * 1_000.0)
        confidence = min(1.0, throughput_us / _GEMM_DISPATCH_FLOOR_US)
        ci_low, ci_high = _ci_bounds(lat, _GEMM_CI_REL_STD)
        return CostEstimate(
            predicted_latency_us=lat,
            predicted_gops=gops,
            confidence=confidence,
            ci_low=ci_low,
            ci_high=ci_high,
        )

    def _predict_col_indep(self, region: Region) -> CostEstimate:
        n_elems = math.prod(region.output.shape)
        bw_us = (n_elems * _INT16_RW_BYTES) / (_COL_INDEP_PEAK_BW_GBPS * 1_000.0)
        lat = max(_COL_INDEP_DISPATCH_FLOOR_US, bw_us)
        eff_bw = (n_elems * _INT16_RW_BYTES) / (lat * 1_000.0)
        confidence = min(1.0, bw_us / _COL_INDEP_DISPATCH_FLOOR_US)
        ci_low, ci_high = _ci_bounds(lat, _COL_INDEP_CI_REL_STD)
        return CostEstimate(
            predicted_latency_us=lat,
            predicted_gops=eff_bw,
            confidence=confidence,
            ci_low=ci_low,
            ci_high=ci_high,
        )

    def _predict_compute_pool(self, region: Region) -> CostEstimate:
        n_elems = math.prod(region.output.shape)
        bw_us = (n_elems * _INT16_RW_BYTES) / (
            _COMPUTE_POOL_PEAK_BW_GBPS * 1_000.0
        )
        lat = max(_COMPUTE_POOL_DISPATCH_FLOOR_US, bw_us)
        eff_bw = (n_elems * _INT16_RW_BYTES) / (lat * 1_000.0)
        confidence = min(1.0, bw_us / _COMPUTE_POOL_DISPATCH_FLOOR_US)
        ci_low, ci_high = _ci_bounds(lat, _COMPUTE_POOL_CI_REL_STD)
        return CostEstimate(
            predicted_latency_us=lat,
            predicted_gops=eff_bw,
            confidence=confidence,
            ci_low=ci_low,
            ci_high=ci_high,
        )

    def _predict_cgra(self, region: Region) -> CostEstimate:
        depth = len(region.inputs)
        lat = self._cgra_interp_lat(depth)
        n_elems = math.prod(region.output.shape)
        eff_bw = (n_elems * 2) / (lat * 1_000.0)
        ci_low, ci_high = _ci_bounds(lat, _CGRA_CI_REL_STD)
        return CostEstimate(
            predicted_latency_us=lat,
            predicted_gops=eff_bw,
            confidence=0.6,
            ci_low=ci_low,
            ci_high=ci_high,
        )

    def _cgra_interp_lat(self, depth: int) -> float:
        depths = sorted(_CGRA_DEPTH_LATS_US)
        d_lo = depths[0]
        d_hi = depths[-1]
        depth_clamped = max(d_lo, min(d_hi, depth))
        for i in range(len(depths) - 1):
            a, b = depths[i], depths[i + 1]
            if a <= depth_clamped <= b:
                t = (depth_clamped - a) / (b - a)
                return _CGRA_DEPTH_LATS_US[a] + t * (
                    _CGRA_DEPTH_LATS_US[b] - _CGRA_DEPTH_LATS_US[a]
                )
        return _CGRA_DEPTH_LATS_US[d_hi]

    def _predict_sliding_window(self, region: Region) -> CostEstimate:
        H, W = region.output.shape
        ops = H * W * 5
        throughput_us = ops / (_SW_PEAK_GOPS * 1_000.0)
        lat = _SW_DISPATCH_FLOOR_US + throughput_us
        gops = ops / (lat * 1_000.0)
        confidence = min(1.0, throughput_us / _SW_DISPATCH_FLOOR_US)
        ci_low, ci_high = _ci_bounds(lat, _SW_CI_REL_STD)
        return CostEstimate(
            predicted_latency_us=lat,
            predicted_gops=gops,
            confidence=confidence,
            ci_low=ci_low,
            ci_high=ci_high,
        )

    def _predict_tanh(self, region: Region) -> CostEstimate:
        n_elems = math.prod(region.output.shape)
        bw_us = (n_elems * _INT16_RW_BYTES) / (_TANH_PEAK_BW_GBPS * 1_000.0)
        lat = max(_TANH_DISPATCH_FLOOR_US, bw_us)
        eff_bw = (n_elems * _INT16_RW_BYTES) / (lat * 1_000.0)
        confidence = min(1.0, bw_us / _TANH_DISPATCH_FLOOR_US)
        ci_low, ci_high = _ci_bounds(lat, _TANH_CI_REL_STD)
        return CostEstimate(
            predicted_latency_us=lat,
            predicted_gops=eff_bw,
            confidence=confidence,
            ci_low=ci_low,
            ci_high=ci_high,
        )

    def _predict_hash(self, region: Region) -> CostEstimate:
        n_elems = math.prod(region.output.shape)
        bw_us = (n_elems * _INT16_RW_BYTES) / (_HASH_PEAK_BW_GBPS * 1_000.0)
        lat = max(_HASH_DISPATCH_FLOOR_US, bw_us)
        eff_bw = (n_elems * _INT16_RW_BYTES) / (lat * 1_000.0)
        confidence = min(1.0, bw_us / _HASH_DISPATCH_FLOOR_US)
        ci_low, ci_high = _ci_bounds(lat, _HASH_CI_REL_STD)
        return CostEstimate(
            predicted_latency_us=lat,
            predicted_gops=eff_bw,
            confidence=confidence,
            ci_low=ci_low,
            ci_high=ci_high,
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
        if template_name == "sliding_window":
            if region.op != "stencil_2d":
                return False
            return tuple(region.output.shape) in supported
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
