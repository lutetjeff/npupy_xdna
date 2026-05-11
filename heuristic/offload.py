from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from npupy_xdna.heuristic.classifier import RegionClassifier
from npupy_xdna.heuristic.cost_model import CostModel
from npupy_xdna.regions.region import Region

_NPU_NATIVE_TEMPLATES: frozenset[str] = frozenset({"sliding_window"})


@dataclass(frozen=True)
class OffloadDecision:
    action: Literal["offload", "cpu_fallback"]
    template: Optional[str] = None
    predicted_speedup: Optional[float] = None
    rationale: Optional[str] = None
    reason: Optional[str] = None


class OffloadHeuristic:
    def __init__(
        self,
        cost_model: CostModel,
        classifier: RegionClassifier,
        margin: float = 0.1,
    ) -> None:
        self._cost_model = cost_model
        self._classifier = classifier
        self._margin = margin

    def decide(self, region: Region) -> OffloadDecision:
        match = self._classifier.classify(region)
        if match is None:
            return OffloadDecision(action="cpu_fallback", reason="no matching template")

        npu_est = self._cost_model.predict(match.template_name, region)
        if npu_est is None:
            return OffloadDecision(action="cpu_fallback", reason="shape not supported")

        cpu_latency = self._cost_model.cpu_predict(region)
        if cpu_latency is None:
            if match.template_name in _NPU_NATIVE_TEMPLATES:
                return OffloadDecision(
                    action="offload",
                    template=match.template_name,
                    predicted_speedup=None,
                    rationale=match.rationale,
                )
            return OffloadDecision(
                action="cpu_fallback", reason="cpu_predict returned None for this op"
            )

        predicted_speedup = cpu_latency / npu_est.predicted_latency_us

        if predicted_speedup > 1.0 + self._margin:
            return OffloadDecision(
                action="offload",
                template=match.template_name,
                predicted_speedup=predicted_speedup,
                rationale=match.rationale,
            )

        return OffloadDecision(
            action="cpu_fallback",
            reason=f"predicted speedup {predicted_speedup:.2f}x below threshold",
        )
