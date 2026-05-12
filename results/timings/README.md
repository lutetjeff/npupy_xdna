# results/timings/
Raw JSONL measurement data from characterization sweeps and evaluations.
Each file is append-only JSONL (one JSON object per line).
- `gemm_fusion.jsonl` — GEMM Fusion across sizes 256³–4096³
- `col_indep.jsonl` — Column-Independent across 16K–4M elements
- `compute_pool.jsonl` — Compute Pool 32-core (negative result)
- `compute_pool_8core.jsonl` — 8-core control experiment
- `cgra.jsonl` — CGRA pipeline at 256 elements
- `cgra_depth_sweep.jsonl` — CGRA at depths 3/8/16
- `tanh.jsonl` — tanh template characterization
- `hash.jsonl` — hash template characterization
- `gemm_tile_sweep.jsonl` — tile size optimization
- `gemm_intrinsic.jsonl` — 4×4×8 vs 8×2×8 MMUL comparison
- `compile_cache.jsonl` — cold vs warm compilation statistics
- `npbench_preset_L.jsonl` — full 14-benchmark preset L evaluation (headline results)
