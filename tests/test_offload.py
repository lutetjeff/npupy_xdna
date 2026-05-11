from __future__ import annotations

import math
from pathlib import Path

import pytest

from npupy_xdna.heuristic.classifier import RegionClassifier
from npupy_xdna.heuristic.cost_model import CostModel
from npupy_xdna.heuristic.offload import OffloadDecision, OffloadHeuristic
from npupy_xdna.regions.region import ArraySpec, Region

EVIDENCE_DIR = Path("/home/lutet/ece511/.sisyphus/evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def _matmul_region(n: int) -> Region:
    spec = ArraySpec(shape=(n, n), dtype="int16")
    return Region(op="matmul", inputs=[spec, spec], output=spec)


def _elementwise_region(n_elems: int, op: str = "elementwise_unary") -> Region:
    inputs = [ArraySpec(shape=(n_elems,), dtype="int16")]
    if op == "elementwise_binary":
        inputs.append(ArraySpec(shape=(n_elems,), dtype="int16"))
    return Region(op=op, inputs=inputs, output=ArraySpec(shape=(n_elems,), dtype="int16"))


def _cgra_region(n_elems: int = 256) -> Region:
    spec = ArraySpec(shape=(n_elems,), dtype="int16")
    return Region(op="chained_elementwise", inputs=[spec], output=spec)


@pytest.fixture(scope="module")
def heuristic() -> OffloadHeuristic:
    return OffloadHeuristic(CostModel(), RegionClassifier())


class TestSmallMatmul:
    def test_256_offloads_to_gemm_fusion(self, heuristic):
        region = _matmul_region(256)
        decision = heuristic.decide(region)
        assert decision.action == "offload"
        assert decision.template == "gemm_fusion"
        assert decision.predicted_speedup is not None
        assert decision.predicted_speedup > 1.1

    def test_256_speedup_is_reasonable(self, heuristic):
        region = _matmul_region(256)
        decision = heuristic.decide(region)
        assert decision.predicted_speedup == pytest.approx(9428.41 / 500.0, rel=1e-3)

    def test_256_evidence(self, heuristic):
        region = _matmul_region(256)
        decision = heuristic.decide(region)
        (EVIDENCE_DIR / "task-17-npu.txt").write_text(
            f"op=matmul shape=(256,256,256)\n"
            f"action={decision.action}\n"
            f"template={decision.template}\n"
            f"predicted_speedup={decision.predicted_speedup:.4f}x\n"
            f"rationale={decision.rationale}\n"
        )


class TestLargeMatmul:
    def test_2048_offloads_with_high_speedup(self, heuristic):
        region = _matmul_region(2048)
        decision = heuristic.decide(region)
        assert decision.action == "offload"
        assert decision.template == "gemm_fusion"
        assert decision.predicted_speedup is not None
        assert decision.predicted_speedup > 100.0

    def test_2048_npu_throughput_dominates(self, heuristic):
        model = CostModel()
        region = _matmul_region(2048)
        npu_est = model.predict("gemm_fusion", region)
        assert npu_est is not None
        assert npu_est.predicted_latency_us > 500.0

    def test_2048_cpu_much_slower(self, heuristic):
        model = CostModel()
        region = _matmul_region(2048)
        cpu_lat = model.cpu_predict(region)
        npu_est = model.predict("gemm_fusion", region)
        assert cpu_lat is not None and npu_est is not None
        speedup = cpu_lat / npu_est.predicted_latency_us
        assert speedup > 1000.0


class TestSmallElementwise:
    def test_16384_cpu_fallback_dispatch_floor(self, heuristic):
        region = _elementwise_region(16384)
        decision = heuristic.decide(region)
        assert decision.action == "cpu_fallback"
        assert decision.reason is not None

    def test_16384_npu_dispatch_floor_dominates(self):
        model = CostModel()
        region = _elementwise_region(16384)
        clf = RegionClassifier()
        match = clf.classify(region)
        assert match is not None
        npu_est = model.predict(match.template_name, region)
        assert npu_est is not None
        assert npu_est.predicted_latency_us == pytest.approx(300.0, rel=1e-6)

    def test_16384_evidence(self, heuristic):
        region = _elementwise_region(16384)
        decision = heuristic.decide(region)
        existing = (EVIDENCE_DIR / "task-17-cpu.txt").read_text() if (EVIDENCE_DIR / "task-17-cpu.txt").exists() else ""
        (EVIDENCE_DIR / "task-17-cpu.txt").write_text(
            existing
            + f"op=elementwise_unary shape=(16384,)\n"
            f"action={decision.action}\n"
            f"reason={decision.reason}\n\n"
        )


class TestLargeElementwise:
    def test_1048576_cost_model_result(self, heuristic):
        region = _elementwise_region(1048576)
        decision = heuristic.decide(region)
        model = CostModel()
        clf = RegionClassifier()
        match = clf.classify(region)
        assert match is not None
        assert match.template_name == "col_independent"
        npu_est = model.predict(match.template_name, region)
        cpu_lat = model.cpu_predict(region)
        assert npu_est is not None and cpu_lat is not None
        expected_speedup = cpu_lat / npu_est.predicted_latency_us
        if expected_speedup > 1.1:
            assert decision.action == "offload"
            assert decision.template == "col_independent"
        else:
            assert decision.action == "cpu_fallback"
            assert decision.reason is not None

    def test_1048576_npu_latency_above_dispatch_floor(self):
        model = CostModel()
        region = _elementwise_region(1048576)
        clf = RegionClassifier()
        match = clf.classify(region)
        npu_est = model.predict(match.template_name, region)
        assert npu_est is not None
        assert npu_est.predicted_latency_us > 300.0


class TestCgra:
    def test_256_chained_cpu_fallback(self, heuristic):
        region = _cgra_region(256)
        decision = heuristic.decide(region)
        assert decision.action == "cpu_fallback"
        assert decision.reason is not None

    def test_256_cpu_far_faster_than_npu(self):
        model = CostModel()
        region = _cgra_region(256)
        clf = RegionClassifier()
        match = clf.classify(region)
        assert match is not None
        assert match.template_name == "cgra"
        npu_est = model.predict(match.template_name, region)
        cpu_lat = model.cpu_predict(region)
        assert npu_est is not None and cpu_lat is not None
        speedup = cpu_lat / npu_est.predicted_latency_us
        assert speedup < 1.0

    def test_cgra_evidence(self, heuristic):
        region = _cgra_region(256)
        decision = heuristic.decide(region)
        existing = (EVIDENCE_DIR / "task-17-cpu.txt").read_text() if (EVIDENCE_DIR / "task-17-cpu.txt").exists() else ""
        (EVIDENCE_DIR / "task-17-cpu.txt").write_text(
            existing
            + f"op=chained_elementwise shape=(256,)\n"
            f"action={decision.action}\n"
            f"reason={decision.reason}\n\n"
        )


class TestUnsupportedDtype:
    def test_float32_region_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            ArraySpec(shape=(256,), dtype="float32")

    def test_unsupported_op_region_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported op"):
            Region(
                op="conv2d",
                inputs=[ArraySpec(shape=(256,), dtype="int16")],
                output=ArraySpec(shape=(256,), dtype="int16"),
            )


class TestCpuFallbackPaths:
    def test_unclassifiable_region_fallback(self, heuristic):
        region = _matmul_region(128)
        decision = heuristic.decide(region)
        assert decision.action == "cpu_fallback"
        assert "no matching template" in (decision.reason or "")

    def test_custom_margin_affects_threshold(self):
        tight = OffloadHeuristic(CostModel(), RegionClassifier(), margin=0.0)
        loose = OffloadHeuristic(CostModel(), RegionClassifier(), margin=50.0)
        region = _matmul_region(256)
        assert tight.decide(region).action == "offload"
        assert loose.decide(region).action == "cpu_fallback"

    def test_decision_is_frozen_dataclass(self, heuristic):
        region = _matmul_region(256)
        decision = heuristic.decide(region)
        with pytest.raises((AttributeError, TypeError)):
            decision.action = "cpu_fallback"  # type: ignore[misc]
