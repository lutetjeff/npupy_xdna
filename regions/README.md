# regions/
Region dataclass representing an array operation to accelerate.
- `region.py` — `Region`, `ArraySpec` dataclasses with int16-only validation, JSON serialization
- Supported ops: matmul, matmul_fused, elementwise_unary, elementwise_binary, chained_elementwise, stencil_2d
