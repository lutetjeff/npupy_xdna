from __future__ import annotations

import pytest

from npupy_xdna.heuristic.cost_model import CostModel
from npupy_xdna.regions.region import ArraySpec, Region


def _matmul_region(n: int) -> Region:
    spec = ArraySpec(shape=(n, n), dtype="int16")
    return Region(
        op="matmul",
        inputs=[spec, spec],
        output=spec,
    )


def _elementwise_region(n_elems: int) -> Region:
    return Region(
        op="elementwise_unary",
        inputs=[ArraySpec(shape=(n_elems,), dtype="int16")],
        output=ArraySpec(shape=(n_elems,), dtype="int16"),
    )


def _cgra_region() -> Region:
    spec = ArraySpec(shape=(256,), dtype="int16")
    return Region(
        op="chained_elementwise",
        inputs=[spec, spec, spec, spec],
        output=spec,
    )


class TestCostModelInit:
    def test_loads_without_error(self):
        model = CostModel()
        assert model is not None

    def test_raw_data_loaded(self):
        model = CostModel()
        assert len(model._raw["gemm_fusion"]) == 10
        assert len(model._raw["col_indep"]) == 4
        assert len(model._raw["compute_pool"]) == 4
        assert len(model._raw["cgra"]) == 1


class TestPredictGemmFusion:
    def test_supported_shapes_return_estimate(self):
        model = CostModel()
        for n in (256, 512, 1024, 2048, 4096):
            est = model.predict("gemm_fusion", _matmul_region(n))
            assert est is not None
            assert est.predicted_latency_us > 0

    def test_unsupported_shape_returns_none(self):
        model = CostModel()
        est = model.predict("gemm_fusion", _matmul_region(128))
        assert est is None

    def test_wrong_op_returns_none(self):
        model = CostModel()
        region = Region(
            op="elementwise_unary",
            inputs=[ArraySpec(shape=(256,), dtype="int16")],
            output=ArraySpec(shape=(256,), dtype="int16"),
        )
        assert model.predict("gemm_fusion", region) is None

    def test_small_shape_dispatch_floor_dominates(self):
        model = CostModel()
        est = model.predict("gemm_fusion", _matmul_region(256))
        assert est is not None
        assert est.predicted_latency_us == pytest.approx(500.0, rel=1e-6)
        assert est.confidence < 1.0

    def test_peak_shape_throughput_dominates(self):
        model = CostModel()
        est = model.predict("gemm_fusion", _matmul_region(2048))
        assert est is not None
        assert est.predicted_latency_us > 500.0
        assert est.confidence == pytest.approx(1.0, rel=1e-6)

    def test_2048_matches_measured_within_5pct(self):
        model = CostModel()
        est = model.predict("gemm_fusion", _matmul_region(2048))
        assert est is not None
        assert abs(est.predicted_latency_us - 3330.0) / 3330.0 < 0.05

    def test_confidence_increases_with_size(self):
        model = CostModel()
        est_small = model.predict("gemm_fusion", _matmul_region(256))
        est_large = model.predict("gemm_fusion", _matmul_region(2048))
        assert est_small is not None and est_large is not None
        assert est_small.confidence < est_large.confidence

    def test_gops_positive(self):
        model = CostModel()
        for n in (256, 512, 1024, 2048):
            est = model.predict("gemm_fusion", _matmul_region(n))
            assert est is not None
            assert est.predicted_gops > 0


class TestPredictColIndependent:
    def test_supported_sizes_return_estimate(self):
        model = CostModel()
        for n in (16384, 65536, 262144, 1048576):
            est = model.predict("col_independent", _elementwise_region(n))
            assert est is not None
            assert est.predicted_latency_us > 0

    def test_unsupported_size_returns_none(self):
        model = CostModel()
        assert model.predict("col_independent", _elementwise_region(1000)) is None

    def test_small_sizes_dispatch_floor_dominates(self):
        model = CostModel()
        for n in (16384, 65536, 262144):
            est = model.predict("col_independent", _elementwise_region(n))
            assert est is not None
            assert est.predicted_latency_us == pytest.approx(300.0, rel=1e-6)

    def test_1m_elements_matches_measured_within_1pct(self):
        model = CostModel()
        est = model.predict("col_independent", _elementwise_region(1048576))
        assert est is not None
        assert abs(est.predicted_latency_us - 387.842) / 387.842 < 0.01

    def test_bandwidth_reported_as_gops(self):
        model = CostModel()
        est = model.predict("col_independent", _elementwise_region(1048576))
        assert est is not None
        assert est.predicted_gops == pytest.approx(10.8145, rel=0.01)


class TestPredictComputePool:
    def test_supported_sizes_return_estimate(self):
        model = CostModel()
        for n in (32768, 131072, 524288, 2097152):
            est = model.predict("compute_pool", _elementwise_region(n))
            assert est is not None
            assert est.predicted_latency_us >= 15_000.0

    def test_unsupported_size_returns_none(self):
        model = CostModel()
        assert model.predict("compute_pool", _elementwise_region(1000)) is None

    def test_all_sizes_dispatch_floor_dominated(self):
        model = CostModel()
        for n in (32768, 131072, 524288):
            est = model.predict("compute_pool", _elementwise_region(n))
            assert est is not None
            assert est.predicted_latency_us == pytest.approx(15_000.0, rel=1e-6)

    def test_confidence_near_zero_for_small_sizes(self):
        model = CostModel()
        est = model.predict("compute_pool", _elementwise_region(32768))
        assert est is not None
        assert est.confidence < 0.05


class TestPredictCgra:
    def test_returns_fixed_latency(self):
        model = CostModel()
        est = model.predict("cgra", _cgra_region())
        assert est is not None
        assert est.predicted_latency_us == pytest.approx(190.0, rel=1e-6)

    def test_confidence_is_fixed(self):
        model = CostModel()
        est = model.predict("cgra", _cgra_region())
        assert est is not None
        assert est.confidence == pytest.approx(0.6, rel=1e-6)

    def test_unsupported_size_returns_none(self):
        model = CostModel()
        spec = ArraySpec(shape=(512,), dtype="int16")
        region = Region(
            op="chained_elementwise",
            inputs=[spec, spec, spec, spec],
            output=spec,
        )
        assert model.predict("cgra", region) is None


class TestPredictUnknownTemplate:
    def test_unknown_template_returns_none(self):
        model = CostModel()
        est = model.predict("nonexistent_template", _matmul_region(256))
        assert est is None


class TestCpuPredict:
    def test_matmul_256_matches_reference(self):
        model = CostModel()
        region = _matmul_region(256)
        cpu_us = model.cpu_predict(region)
        assert cpu_us is not None
        assert cpu_us == pytest.approx(9_428.41, rel=1e-4)

    def test_matmul_scales_cubic_in_N(self):
        model = CostModel()
        lat_256 = model.cpu_predict(_matmul_region(256))
        lat_512 = model.cpu_predict(_matmul_region(512))
        assert lat_256 is not None and lat_512 is not None
        ratio = lat_512 / lat_256
        assert 6.0 < ratio < 12.0

    def test_elementwise_returns_float(self):
        model = CostModel()
        region = _elementwise_region(1048576)
        cpu_us = model.cpu_predict(region)
        assert cpu_us is not None
        assert cpu_us > 0

    def test_elementwise_scales_linear(self):
        model = CostModel()
        lat_1m = model.cpu_predict(_elementwise_region(1048576))
        lat_2m = model.cpu_predict(_elementwise_region(1048576 * 2))
        assert lat_1m is not None and lat_2m is not None
        assert lat_2m / lat_1m == pytest.approx(2.0, rel=1e-6)

    def test_cgra_region_returns_float(self):
        model = CostModel()
        cpu_us = model.cpu_predict(_cgra_region())
        assert cpu_us is not None
        assert cpu_us > 0


class TestCrossover:
    def test_gemm_512_npu_wins(self):
        model = CostModel()
        region = _matmul_region(512)
        est = model.predict("gemm_fusion", region)
        cpu_us = model.cpu_predict(region)
        assert est is not None and cpu_us is not None
        assert est.predicted_latency_us < cpu_us

    def test_cgra_cpu_wins(self):
        model = CostModel()
        region = _cgra_region()
        est = model.predict("cgra", region)
        cpu_us = model.cpu_predict(region)
        assert est is not None and cpu_us is not None
        assert cpu_us < est.predicted_latency_us

    def test_compute_pool_cpu_always_wins(self):
        model = CostModel()
        for n in (32768, 131072, 524288, 2097152):
            region = _elementwise_region(n)
            est = model.predict("compute_pool", region)
            cpu_us = model.cpu_predict(region)
            assert est is not None and cpu_us is not None
            assert cpu_us < est.predicted_latency_us
