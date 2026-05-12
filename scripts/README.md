# scripts/
Characterization sweeps, evaluation runners, and plot generators.
Run with: `source /opt/xilinx/xrt/setup.sh && source ~/mlir-aie/ironenv/bin/activate && source ~/mlir-aie/utils/env_setup.sh`
- `characterize_*.py` — per-template performance sweeps → results/timings/*.jsonl
- `eval_preset_L.py` — full 14-benchmark preset L evaluation (headline results)
- `eval_npbench_v2.py` — NPBench V2 evaluation with honest BLAS baseline
- `generate_poc1_plots.py` — PoC 1 heuristic visualization plots
- `generate_eval_plots.py` — NPBench evaluation plots
- `sweep_*.py` — parameter sweeps (tile sizes, core counts, CGRA depth)
