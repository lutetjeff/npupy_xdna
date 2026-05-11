# NPUPy XDNA — NumPy-to-NPU Offloading for AMD Ryzen AI

A transparent NumPy backend that offloads array operations to the AMD Ryzen AI NPU (XDNA2 architecture) using heuristic-driven spatial template selection.

**ECE 511 — Advanced Computer Architecture, UIUC**  
Jefferson Zhang (`jyz4@illinois.edu`) & Landon Klecka (`lklecka2@illinois.edu`)

---

## Overview

NPUPy intercepts NumPy operations via Python's `__array_function__` protocol and transparently dispatches profitable workloads to the Ryzen AI NPU while falling back to CPU for operations where the NPU wouldn't provide speedup.

### Architecture

```
numpy.matmul(A, B)
    │
    ▼
__array_function__ shim (dispatch/array_shim.py)
    │
    ▼
Region extraction (dispatch/extract.py) → Region(op="matmul", shape=...)
    │
    ▼
Classifier (heuristic/classifier.py) → "gemm_fusion" template
    │
    ▼
Cost Model (heuristic/cost_model.py) → NPU: 3.3ms vs CPU: 11.4ms
    │
    ▼
Offload Heuristic (heuristic/offload.py) → OFFLOAD (3.4× predicted speedup)
    │
    ▼
Template lower (templates/gemm_fusion.py) → IRON JIT → xclbin
    │
    ▼
NPU Runner (runtime/npu_runner.py) → XRT DMA → 32-core AIE array → result
```

---

## Key Results

### NPBench Evaluation (int16, Preset M)

| Benchmark | NumPy (ms) | NPUPy (ms) | Speedup | Template |
|-----------|-----------|------------|---------|----------|
| gemm | 11.4 | 0.69 | **16.4×** | gemm_fusion |
| 3mm | 28.9 | 1.97 | **14.7×** | gemm_fusion |
| symm | 9.1 | 0.71 | **12.9×** | gemm_fusion |
| gemm_relu | 9.0 | 0.75 | **12.0×** | gemm_fusion |
| syr2k | 11.7 | 1.51 | **7.8×** | gemm_fusion |
| mvt | 0.066 | 0.066 | 1.0× | cpu_fallback (matvec) |
| atax | 0.061 | 0.061 | 1.0× | cpu_fallback (matvec) |
| correlation | 13.6 | 13.6 | 1.0× | cpu_fallback (float32) |
| gramschmidt | 63.1 | 63.1 | 1.0× | cpu_fallback (dot/scalar) |

**5/9 benchmarks show 7.8×–16.4× speedup.** Zero false positives — the heuristic never makes a benchmark slower.

**Peak GEMM throughput:** 5,159 GOPS at 2048³ int16 (10.3% of 50 TOPS theoretical peak).

### CPU Fallback Overhead

| Benchmark | NumPy (ms) | NPUPy (ms) | Overhead | Note |
|-----------|-----------|------------|----------|------|
| jacobi-2d | 6.56 | 7.12 | 1.09× | 150 dispatch calls, all CPU |
| heat-3d | 70.3 | 71.97 | 1.02× | 105 dispatch calls, all CPU |

All CPU-fallback paths add ≤9% overhead — acceptable for a transparent backend.

---

## Four Spatial Templates

| Template | Strategy | Cores | Best Use Case | Peak Perf |
|----------|----------|-------|---------------|-----------|
| **GEMM Fusion** | Whole-array systolic matmul + epilogue | 32 (8×4) | Dense matmul ≥256³ | 5,159 GOPS |
| **Column-Independent** | 8 independent pools of 4 cores | 32 (8×4) | Large elementwise (≥16K elements) | 10.8 GB/s |
| **Compute Pool** | 32 fully independent cores | 32 | Experimental — negative result | 0.55 GB/s |
| **CGRA** | 3-op spatial pipeline across tiles | 3 | Experimental — works but dispatch-dominated | N/A |

---

## Hardware

- **Machine:** ASUS Vivobook S16 M5606KA
- **CPU:** AMD Ryzen AI 7 350 w/ Radeon 860M (16 logical cores)
- **NPU:** Krackan (XDNA2), 8 columns × 4 rows = 32 AIE-ML v2 cores, 50 TOPS theoretical (int8)
- **RAM:** 14,580 MB
- **OS:** Ubuntu Resolute Raccoon (Linux 6.19.0-5-generic, x86_64)
- **Software:** XRT 2.23.0, amdxdna driver 2.23.0_20260218, mlir-aie (IRON interface), Peano llvm-aie

---

## Quick Start

```bash
# Environment setup
source /opt/xilinx/xrt/setup.sh
source ~/mlir-aie/ironenv/bin/activate
source ~/mlir-aie/utils/env_setup.sh

# Install
pip install -e /path/to/npupy_xdna

# Use as transparent NumPy backend
from npupy_xdna.dispatch import activate, deactivate
import numpy as np

activate()
A = np.random.default_rng(42).integers(-5, 5, (256, 256), dtype=np.int16)
B = np.random.default_rng(43).integers(-5, 5, (256, 256), dtype=np.int16)
C = A @ B  # Transparently dispatched to NPU
deactivate()
```

---

## Project Structure

```
npupy_xdna/
├── regions/          # Region dataclass (op, shapes, dtype)
├── templates/        # 4 spatial templates (GEMM, Col-Indep, Compute Pool, CGRA)
├── kernels/          # C++ AIE kernel sources
├── runtime/          # NpuRunner, CpuRunner, preflight checks, NPU lock
├── heuristic/        # Classifier, cost model, offload heuristic, visualizations
├── dispatch/         # __array_function__ shim, region extraction, dispatcher
├── bench/            # Timer, baselines, synthetic benchmarks
├── verify/           # Correctness verification
├── tests/            # 185+ tests across all modules
├── scripts/          # Characterization sweeps, evaluation, plot generation
└── results/          # All measurement data, plots, and final report
    ├── timings/      # JSONL characterization data per template
    ├── 03_heuristic_visualizations/  # PoC 1 plots (6 PNGs)
    ├── 04_npbench_plots/             # NPBench evaluation plots (4 PNGs)
    └── FINAL_REPORT_DATA.md          # Complete results synthesis
```

---

## Research Findings

1. **The NPU is a GEMM engine, not a general-purpose accelerator.** Only matmul operations benefit from offloading; elementwise ops lose to CPU at all tested sizes due to ~100–300 µs dispatch overhead.

2. **Template topology matters.** Column-Independent (8 pools) beats Compute Pool (32 cores) by 50× for elementwise ops — the DMA subsystem is optimized for column-level granularity.

3. **Fusion works within a single kernel.** ReLU epilogue adds <1% overhead when fused into GEMM. Cross-region fusion remains unsolved.

4. **The heuristic correctly partitions workloads.** Zero false positives — the cost model never makes a benchmark slower than vanilla NumPy.

5. **Dispatch overhead floor is ~100–300 µs.** This is the hard barrier for small workloads. Any operation completable in ≤100 µs on CPU should stay on CPU.

---

## Heuristic Visualizations

Six diagnostic plots are generated by `scripts/generate_poc1_plots.py` from Wave-2 characterization data:

| Plot | Description |
|------|-------------|
| `01_gemm_throughput.png` | GOPS vs matrix dimension; peak 5,159 GOPS at 2048³ |
| `02_gemm_npu_vs_cpu_latency.png` | NPU vs CPU latency; NPU 15×–471× faster across all tested sizes |
| `03_bandwidth_scaling.png` | col_independent vs compute_pool bandwidth; 50× gap due to DMA topology |
| `04_template_decision_map.png` | 2D heatmap: (op_type, size) → template assignment |
| `05_offload_decision_map.png` | 2D heatmap: (op_type, size) → offload/fallback decision |
| `06_all_templates_latency.png` | Grouped bar chart: NPU vs CPU at peak-throughput size for each template |

---

## Status

This is a research prototype (ECE 511 course project, May 2026). Results are preliminary.

## License

Research use only. AMD XDNA toolchain components are subject to their respective licenses.
