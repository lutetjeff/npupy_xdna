from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

SUPPORTED_OPS = {
    "matmul",
    "matmul_fused",
    "elementwise_unary",
    "elementwise_binary",
    "chained_elementwise",
}
SUPPORTED_DTYPES = {"int16"}


@dataclass(frozen=True)
class ArraySpec:
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self):
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype {self.dtype!r}, must be one of {SUPPORTED_DTYPES}"
            )
        if not self.shape:
            raise ValueError("shape must not be empty")
        for i, dim in enumerate(self.shape):
            if dim <= 0:
                raise ValueError(f"shape dimension {i} must be positive, got {dim}")


@dataclass(frozen=True)
class Region:
    op: str
    inputs: list[ArraySpec]
    output: ArraySpec
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.op not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported op {self.op!r}, must be one of {SUPPORTED_OPS}")
        if self.output.dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype {self.output.dtype!r}, must be one of {SUPPORTED_DTYPES}"
            )
        for i, inp in enumerate(self.inputs):
            if inp.dtype not in SUPPORTED_DTYPES:
                raise ValueError(
                    f"Unsupported dtype {inp.dtype!r} in inputs[{i}], must be one of {SUPPORTED_DTYPES}"
                )
        if self.op in ("matmul", "matmul_fused"):
            if len(self.inputs) != 2:
                raise ValueError(f"{self.op} requires exactly 2 inputs")
            lhs = self.inputs[0]
            rhs = self.inputs[1]
            if len(lhs.shape) != 2 or len(rhs.shape) != 2:
                raise ValueError(f"{self.op} inputs must be 2D")
            if lhs.shape[1] != rhs.shape[0]:
                raise ValueError(
                    f"Inner dimensions must match: lhs K={lhs.shape[1]} != rhs K={rhs.shape[0]}"
                )
            if self.output.shape != (lhs.shape[0], rhs.shape[1]):
                raise ValueError(
                    f"Output shape mismatch: expected ({lhs.shape[0]}, {rhs.shape[1]}), got {self.output.shape}"
                )

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> Region:
        d = json.loads(s)
        d["inputs"] = [
            ArraySpec(shape=tuple(spec["shape"]), dtype=spec["dtype"])
            for spec in d["inputs"]
        ]
        d["output"] = ArraySpec(
            shape=tuple(d["output"]["shape"]), dtype=d["output"]["dtype"]
        )
        return cls(**d)

    def __str__(self) -> str:
        in_shapes = " x ".join(str(s.shape) for s in self.inputs)
        return f"{self.op} {in_shapes} -> {self.output.shape} {self.output.dtype}"
