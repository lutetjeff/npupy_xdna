# templates/
Spatial templates that lower Regions to NPU-executable IRON programs.
- `protocol.py` — Template Protocol, Config, CostEstimate dataclasses
- `shape_matrix.py` — SUPPORTED_SHAPES registry (compile-time shape enumeration)
- `gemm_fusion.py` — 32-core whole-array matmul with scale/bias/relu epilogues
- `col_independent.py` — 8-pool×4-core elementwise (relu, tanh, hash variants)
- `sliding_window.py` — 2D stencil with strip decomposition
- `compute_pool.py` — 32 independent cores (experimental, high dispatch floor)
- `cgra.py` — spatial pipeline across tiles (working, dispatch-dominated at small sizes)
- `chained_gemm.py` — cross-region fusion attempt (kill-switched)
