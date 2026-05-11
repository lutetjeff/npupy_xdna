from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from npupy_xdna.regions.region import Region
from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES

_RULES_PATH = Path(__file__).parent / "rules.yaml"

_TEMPLATE_SHAPE_KEY: dict[str, str] = {
    "gemm_fusion": "gemm_fusion",
    "col_independent": "col_indep",
    "compute_pool": "compute_pool",
    "cgra": "cgra",
    "sliding_window": "sliding_window",
}


@dataclass(frozen=True)
class TemplateMatch:
    template_name: str
    confidence: float
    rationale: str


def _total_elements(region: Region) -> int:
    result = 1
    for dim in region.output.shape:
        result *= dim
    return result


def _normalized_shape(region: Region, template_name: str):
    if template_name == "gemm_fusion":
        M = region.inputs[0].shape[0]
        K = region.inputs[0].shape[1]
        N = region.inputs[1].shape[1]
        return (M, K, N)
    if template_name == "sliding_window":
        return tuple(region.output.shape)
    return _total_elements(region)


def _shape_supported(region: Region, template_name: str) -> bool:
    shape_key = _TEMPLATE_SHAPE_KEY.get(template_name)
    if shape_key is None:
        return False
    return _normalized_shape(region, template_name) in SUPPORTED_SHAPES[shape_key]


def _metadata_matches(region: Region, required: dict) -> bool:
    for key, val in required.items():
        if region.metadata.get(key) != val:
            return False
    return True


class RegionClassifier:
    def __init__(self, rules_path: Path | None = None) -> None:
        path = rules_path or _RULES_PATH
        with open(path) as fh:
            self._rules: list[dict] = yaml.safe_load(fh)

    def classify(self, region: Region) -> TemplateMatch | None:
        total = _total_elements(region)
        for rule in self._rules:
            if region.op not in set(rule["ops"]):
                continue
            required_metadata = rule.get("metadata")
            if required_metadata and not _metadata_matches(region, required_metadata):
                continue
            min_elements = rule.get("min_elements")
            if min_elements is not None and total < min_elements:
                continue
            if not _shape_supported(region, rule["template"]):
                continue
            return TemplateMatch(
                template_name=rule["template"],
                confidence=rule["confidence"],
                rationale=rule["rationale"],
            )
        return None
