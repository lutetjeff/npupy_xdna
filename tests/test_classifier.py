from __future__ import annotations

from pathlib import Path

import pytest

from npupy_xdna.heuristic.classifier import RegionClassifier, TemplateMatch
from npupy_xdna.regions.region import ArraySpec, Region

EVIDENCE_DIR = Path("/home/lutet/ece511/.sisyphus/evidence")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def _matmul_region(M: int, K: int, N: int, op: str = "matmul") -> Region:
    return Region(
        op=op,
        inputs=[ArraySpec(shape=(M, K), dtype="int16"), ArraySpec(shape=(K, N), dtype="int16")],
        output=ArraySpec(shape=(M, N), dtype="int16"),
    )


def _elementwise_region(size: int, op: str = "elementwise_unary") -> Region:
    inputs = [ArraySpec(shape=(size,), dtype="int16")]
    if op == "elementwise_binary":
        inputs.append(ArraySpec(shape=(size,), dtype="int16"))
    return Region(op=op, inputs=inputs, output=ArraySpec(shape=(size,), dtype="int16"))


def _chained_region(size: int) -> Region:
    return Region(
        op="chained_elementwise",
        inputs=[ArraySpec(shape=(size,), dtype="int16")],
        output=ArraySpec(shape=(size,), dtype="int16"),
    )


@pytest.fixture(scope="module")
def clf():
    return RegionClassifier()


class TestMatmulRule:
    def test_matmul_256_maps_to_gemm_fusion(self, clf):
        region = _matmul_region(256, 256, 256)
        match = clf.classify(region)
        assert match is not None
        assert match.template_name == "gemm_fusion"
        assert match.confidence == pytest.approx(0.95)

        path = EVIDENCE_DIR / "task-15-matmul.txt"
        path.write_text(
            f"op={region.op} shape=(256,256,256)\n"
            f"template_name={match.template_name}\n"
            f"confidence={match.confidence}\n"
            f"rationale={match.rationale}\n"
        )

    def test_matmul_fused_512_maps_to_gemm_fusion(self, clf):
        region = _matmul_region(512, 512, 512, op="matmul_fused")
        match = clf.classify(region)
        assert match is not None
        assert match.template_name == "gemm_fusion"
        assert match.confidence == pytest.approx(0.95)

    def test_matmul_unsupported_shape_returns_none(self, clf):
        region = _matmul_region(100, 100, 100)
        assert clf.classify(region) is None


class TestComputePoolRule:
    def test_elementwise_unary_32768_maps_to_compute_pool(self, clf):
        region = _elementwise_region(32768)
        match = clf.classify(region)
        assert match is not None
        assert match.template_name == "compute_pool"
        assert match.confidence == pytest.approx(0.85)

    def test_elementwise_unary_16384_skips_compute_pool(self, clf):
        region = _elementwise_region(16384)
        match = clf.classify(region)
        assert match is not None
        assert match.template_name == "col_independent"
        assert match.confidence == pytest.approx(0.80)


class TestColIndependentRule:
    def test_elementwise_binary_16384_maps_to_col_independent(self, clf):
        region = _elementwise_region(65536, op="elementwise_binary")
        match = clf.classify(region)
        assert match is not None
        assert match.template_name == "col_independent"
        assert match.confidence == pytest.approx(0.80)


class TestCgraRule:
    def test_chained_elementwise_256_maps_to_cgra(self, clf):
        region = _chained_region(256)
        match = clf.classify(region)
        assert match is not None
        assert match.template_name == "cgra"
        assert match.confidence == pytest.approx(0.70)

    def test_chained_elementwise_unsupported_size_returns_none(self, clf):
        region = _chained_region(1024)
        assert clf.classify(region) is None


class TestCpuFallback:
    def test_unsupported_shape_returns_none(self, clf):
        region = _matmul_region(128, 128, 128)
        result = clf.classify(region)
        assert result is None

        path = EVIDENCE_DIR / "task-15-fallback.txt"
        path.write_text(
            f"op={region.op} shape=(128,128,128) — not in SUPPORTED_SHAPES['gemm_fusion']\n"
            f"classify result: {result}\n"
            f"CPU fallback triggered correctly\n"
        )

    def test_returns_template_match_dataclass(self, clf):
        region = _matmul_region(256, 256, 256)
        match = clf.classify(region)
        assert isinstance(match, TemplateMatch)
        assert isinstance(match.template_name, str)
        assert isinstance(match.confidence, float)
        assert isinstance(match.rationale, str)


def _stencil_region(H: int, W: int) -> Region:
    spec = ArraySpec(shape=(H, W), dtype="int16")
    return Region(op="stencil_2d", inputs=[spec], output=spec)


class TestSlidingWindowRule:
    def test_stencil_2d_64x64_maps_to_sliding_window(self, clf):
        region = _stencil_region(64, 64)
        match = clf.classify(region)
        assert match is not None
        assert match.template_name == "sliding_window"
        assert match.confidence == pytest.approx(0.85)

    def test_stencil_2d_128x128_maps_to_sliding_window(self, clf):
        region = _stencil_region(128, 128)
        match = clf.classify(region)
        assert match is not None
        assert match.template_name == "sliding_window"

    def test_stencil_2d_unsupported_shape_returns_none(self, clf):
        region = _stencil_region(32, 32)
        assert clf.classify(region) is None


class TestHighIntensityElementwiseRule:
    def test_high_intensity_elementwise_unary_maps_to_col_independent(self, clf):
        region = Region(
            op="elementwise_unary",
            inputs=[ArraySpec(shape=(65536,), dtype="int16")],
            output=ArraySpec(shape=(65536,), dtype="int16"),
            metadata={"compute_intensity": "high"},
        )
        match = clf.classify(region)
        assert match is not None
        assert match.template_name == "col_independent"
        assert match.confidence == pytest.approx(0.80)

    def test_high_intensity_elementwise_binary_maps_to_col_independent(self, clf):
        region = Region(
            op="elementwise_binary",
            inputs=[
                ArraySpec(shape=(65536,), dtype="int16"),
                ArraySpec(shape=(65536,), dtype="int16"),
            ],
            output=ArraySpec(shape=(65536,), dtype="int16"),
            metadata={"compute_intensity": "high"},
        )
        match = clf.classify(region)
        assert match is not None
        assert match.template_name == "col_independent"

    def test_normal_elementwise_without_metadata_not_affected(self, clf):
        region = _elementwise_region(65536, op="elementwise_binary")
        match = clf.classify(region)
        assert match is not None
        assert match.template_name == "col_independent"


class TestDriftDetection:
    def test_rules_yaml_all_templates_are_reachable(self, clf):
        import yaml
        from pathlib import Path as _Path

        rules_path = _Path(__file__).parent.parent / "heuristic" / "rules.yaml"
        with open(rules_path) as fh:
            rules = yaml.safe_load(fh)

        canonical_regions: dict[str, Region] = {
            "sliding_window": _stencil_region(64, 64),
            "gemm_fusion": _matmul_region(256, 256, 256),
            "col_independent": _elementwise_region(65536, op="elementwise_unary"),
            "compute_pool": _elementwise_region(32768, op="elementwise_unary"),
            "cgra": _chained_region(256),
        }

        seen: set[str] = set()
        for rule in rules:
            template = rule["template"]
            if template in seen or template not in canonical_regions:
                continue
            seen.add(template)
            region = canonical_regions[template]
            match = clf.classify(region)
            assert match is not None, f"No match for template={template}"
            assert match.template_name == template, (
                f"rules.yaml drift: expected {template}, got {match.template_name}"
            )

        assert seen == set(canonical_regions), (
            f"Missing coverage for templates: {set(canonical_regions) - seen}"
        )

    def test_chained_gemm_has_no_route(self, clf):
        from npupy_xdna.regions.region import SUPPORTED_OPS
        assert "chained_gemm" not in SUPPORTED_OPS
