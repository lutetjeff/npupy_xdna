from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Protocol, runtime_checkable, Callable

from npupy_xdna.regions.region import Region


@dataclass(frozen=True)
class CostEstimate:
    predicted_latency_us: float
    predicted_gops: float
    confidence: float
    ci_low: float = 0.0   # reporting only; does not affect offload decision
    ci_high: float = 0.0  # reporting only; does not affect offload decision


@dataclass(frozen=True)
class Config:
    tile: tuple[int, ...]
    n_cores: int
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def __str__(self) -> str:
        return f"tile={self.tile} n_cores={self.n_cores}"


@runtime_checkable
class Template(Protocol):
    name: str

    def match(self, region: Region) -> bool: ...

    def config_space(self, region: Region) -> list[Config]: ...

    def lower(self, region: Region, config: Config) -> Callable: ...

    def estimated_cost(self, region: Region, config: Config) -> CostEstimate: ...
