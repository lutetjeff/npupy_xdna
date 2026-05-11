from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Protocol, runtime_checkable

import numpy as np

from npupy_xdna.regions.region import Region
from npupy_xdna.templates.protocol import Config


@dataclass
class RunResult:
    output: np.ndarray
    latency_us: float = 0.0
    status: str = "ok"
    device: str = "unknown"

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("output")
        return json.dumps(d)


@runtime_checkable
class Runner(Protocol):
    def run(self, region: Region, inputs: list[np.ndarray]) -> RunResult: ...
