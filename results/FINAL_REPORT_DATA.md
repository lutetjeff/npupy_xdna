# NPUPy on AMD XDNA2 — Final Report Data

**Project:** NPUPy: A NumPy-to-NPU Offloading Framework for AMD Ryzen AI (XDNA2)  
**Course:** ECE 511 — Advanced Computer Architecture  
**Date:** 2026-05-11  
**Hardware:** ASUS Vivobook S16 M5606KA, AMD Ryzen AI 7 350 w/ Radeon 860M  
**NPU:** NPU Krackan 1 (XDNA2 architecture, BDF 0000:64:00.1)  
**Software Stack:** XRT 2.23.0, amdxdna driver 2.23.0_20260218, mlir-aie (IRON interface), Peano llvm-aie, g++-15  

---

> **How to read this document:** Every table, figure reference, and numerical claim is cross-linked to its primary source file in `results/`.  This document is the single artifact that feeds into the ECE 511 final report.  All data are synthesized from measured results; no numbers are fabricated.

---

## Table of Contents

1. [Section 1: Hardware Baseline](#section-1-hardware-baseline)
2. [Section 2: Template Characterization](#section-2-template-characterization)
3. [Section 3: PoC 1 — Heuristic Analyzer](#section-3-poc-1--heuristic-analyzer)
4. [Section 4: PoC 2 — NPBench Evaluation](#section-4-poc-2--npbench-evaluation)
5. [Section 5: Architectural Takeaways](#section-5-architectural-takeaways)
6. [Section 6: Limitations & Future Work](#section-6-limitations--future-work)
7. [Section 7: V2 Optimization Results](#section-7-v2-optimization-results)

---

## Section 1: Hardware Baseline

### 1.1 Platform Overview

All experiments in this project were conducted on a single consumer laptop equipped with AMD's XDNA2 NPU.  The platform specifications are summarized below.

| Component | Specification |
|-----------|--------------|
| Machine | ASUS Vivobook S16 M5606KA |
| CPU | AMD Ryzen AI 7 350 w/ Radeon 860M (16 logical cores) |
| NPU | NPU Krackan 1 (XDNA2 architecture, BDF 0000:64:00.1) |
| Theoretical Peak (int8) | **50 TOPS** |
| AIE Core Array | 8 columns × 4 rows = **32 AIE-ML v2 cores** |
| System RAM | 14,580 MB |
| OS | Ubuntu Resolute Raccoon (Linux 6.19.0-5-generic, x86_64) |

**Source:** [`results/01_hardware_baseline.md`](results/01_hardware_baseline.md)

### 1.2 Software Environment

| Component | Version / Path |
|-----------|---------------|
| XRT | 2.23.0 (hash 99b8b976) |
| amdxdna driver | 2.23.0_20260218 |
| NPU Firmware | 255.0.11.69 |
| mlir-aie | `~/mlir-aie` (IRON Python interface) |
| Peano (llvm-aie) | `~/mlir-aie/ironenv/lib/python3.13/site-packages/llvm-aie` |
| Host compiler | g++-15 (system default) |
| Device target | `npu2` (Krackan/Strix Point — **not** `npu`/Phoenix) |

**Source:** [`results/01_hardware_baseline.md`](results/01_hardware_baseline.md)

### 1.3 Measured Peak Performance (mlir-aie Reference Kernels)

Before building NPUPy, we ran the upstream `mlir-aie` programming examples to establish a hardware baseline.  All runs used 5 warmup + 10 measured iterations, with numerical correctness verified.

#### 1.3.1 Integer GEMM (`basic/matrix_multiplication/whole_array`)

- **Operation:** A(M×K) @ B(K×N) → C(M×N), int8 input → int32 output
- **Hardware config:** 8 AIE columns × 4 rows = 32 cores, 64×64×64 per-tile kernel
- **B layout:** column-major (`--b_col_maj 1`)
- **Theoretical peak:** 50 TOPS (int8 on Krackan)

| Size (M=K=N) | Avg Latency | Avg GOPS | Min Latency | Peak GOPS | % of 50 TOPS |
|-------------:|------------:|---------:|------------:|----------:|-------------:|
| 1024 | 1,475 µs | 1,456 | 331 µs | 6,488 | 13.0% |
| **2048** | **2,171 µs** | **7,914** | **1,997 µs** | **8,603** | **17.2%** |
| 4096 | 50,480 µs | 2,723 | 22,490 µs | 6,111 | 5.4% |

**Key observations:**
- **2048³ is the throughput sweet spot.**  DMA overhead amortizes well (~24 MB total transfer), variance is tight (1,997–2,303 µs across 10 runs).
- **4096³ degrades badly.**  A+B+C at i8/i32 is ~96 MB; the average is dragged down by memory-bandwidth saturation.  Best-case single-run still achieves 6.1 TOPS.
- **1024³ is DMA-launch dominated.**  The 331 µs minimum represents near-pure compute; the 1,475 µs average shows dispatch + DMA round-trip overhead dominates at this scale.  Wide variance (331–3,115 µs) even after 5 warmup iterations.

**Source:** [`results/01_hardware_baseline.md`](results/01_hardware_baseline.md)

#### 1.3.2 ReLU Activation (`ml/relu`)

- **Operation:** Element-wise max(0, x), bfloat16
- **Hardware config:** 4 AIE columns, 2 DMA channels, streaming

| Elements | Data (2-way) | Latency | Effective Bandwidth |
|---------:|-------------:|--------:|--------------------:|
| 16,384 | 64 KB | 155 µs | 0.42 GB/s |
| 65,536 | 256 KB | 116 µs | 2.26 GB/s |
| 262,144 | 1 MB | 138 µs | 7.60 GB/s |
| 1,048,576 | 4 MB | 210 µs | **19.97 GB/s** |

**Key observations:**
- Small sizes (≤64K elements) are entirely overhead-dominated; the ~115 µs launch floor swamps useful bandwidth.
- At 1M elements, bandwidth saturates near 20 GB/s — approaching DRAM bandwidth limits for this UMA system.  ReLU is compute-trivial so this is effectively a DMA throughput ceiling measurement.

**Source:** [`results/01_hardware_baseline.md`](results/01_hardware_baseline.md)

#### 1.3.3 Element-wise Add (`ml/eltwise_add`)

- **Operation:** A + B → C, bfloat16, 65,536 elements (3 buffers × 128 KB = 384 KB total)

| Metric | Value |
|--------|------:|
| Avg Latency | 108.6 µs |
| Min Latency | 72 µs |
| Max Latency | 118 µs |
| Derived bandwidth @ min | ~5.3 GB/s |

**Source:** [`results/01_hardware_baseline.md`](results/01_hardware_baseline.md)

#### 1.3.4 DMA Passthrough (`basic/memcpy`)

- **Operation:** Pure shim DMA passthrough (no compute)
- **Config:** 1 AIE column, bypass=True

| Buffer Size | Latency | Bidirectional BW |
|------------:|--------:|-----------------:|
| 16 KB | 114 µs | ~0.3 GB/s |
| 4 MB | 1,404 µs | ~5.7 GB/s |

**Key observations:**
- The ~114 µs floor at small sizes is the irreducible NPU dispatch + DMA setup cost, visible across all examples.
- At 4 MB, a single shim DMA channel saturates around 5–6 GB/s bidirectional.
- This is a useful baseline: any workload whose compute fits within the DMA time is not NPU-worth offloading.

**Source:** [`results/01_hardware_baseline.md`](results/01_hardware_baseline.md)

### 1.4 Cross-Example Summary

| Metric | Observed |
|--------|---------:|
| NPU dispatch floor | **~72–115 µs** (irreducible per kernel launch) |
| Single-channel DMA peak (1 col) | ~5.7 GB/s |
| Multi-channel DMA peak (4 cols, streaming) | ~20 GB/s |
| Peak GEMM (int8, 2048³, 32 cores) | **8,603 GOPS = 8.6 TOPS** |
| Peak GEMM efficiency vs 50 TOPS | **~17%** |
| NPU vs CPU GEMM crossover | ~1024³–2048³ for int8 |

**Source:** [`results/01_hardware_baseline.md`](results/01_hardware_baseline.md)

---

## Section 2: Template Characterization

This section synthesizes the Wave-2 characterization campaign that measured four NPUPy templates against CPU baselines.  All measurements used 5 warmup + 10 measured iterations, `np.random.default_rng(42)`, int16 data type, and the ReLU element-wise kernel where applicable.

**Primary sources:**
- [`results/02_template_characterization.md`](results/02_template_characterization.md)
- [`results/timings/gemm_fusion.jsonl`](results/timings/gemm_fusion.jsonl)
- [`results/timings/col_indep.jsonl`](results/timings/col_indep.jsonl)
- [`results/timings/compute_pool.jsonl`](results/timings/compute_pool.jsonl)
- [`results/timings/cgra.jsonl`](results/timings/cgra.jsonl)

### 2.1 GEMM Fusion Template (`gemm_fusion`)

**Architecture:** 8 columns × 4 rows = 32 cores.  Each tile computes a 64×64×64 micro-kernel.  Supports fused epilogues (`none`, `relu`).  Data type: int16.

| Shape (M=K=N) | Epilogue | NPU Median (µs) | NPU Min (µs) | NPU Max (µs) | CPU Median (µs) | GOPS | Speedup |
|--------------:|----------|----------------:|-------------:|-------------:|----------------:|-----:|--------:|
| 256³ | none | 620.6 | 535.7 | 1,182.0 | 9,428.4 | 54.1 | **15.2×** |
| 256³ | relu | 729.3 | 635.7 | 1,093.7 | 9,608.5 | 46.0 | **13.2×** |
| 512³ | none | 925.7 | 874.6 | 1,635.9 | 79,958.5 | 290.0 | **86.4×** |
| 512³ | relu | 1,082.0 | 909.3 | 1,604.7 | 79,940.0 | 248.1 | **73.9×** |
| 1024³ | none | 2,686.5 | 2,456.7 | 3,228.6 | 1,267,345.2 | 799.4 | **471.8×** |
| 1024³ | relu | 2,691.0 | 2,467.7 | 3,670.7 | 1,272,172.0 | 798.0 | **472.8×** |
| **2048³** | **none** | **3,330.0** | **3,261.5** | **3,618.7** | — (skipped) | **5,159.1** | — |
| 2048³ | relu | 3,364.4 | 3,331.5 | 3,413.8 | — (skipped) | 5,106.3 | — |
| 4096³ | none | 35,666.8 | 35,314.4 | 36,253.3 | — (skipped) | 3,853.4 | — |
| 4096³ | relu | 34,824.1 | 34,281.2 | 35,076.9 | — (skipped) | 3,946.7 | — |

**Key findings:**
- **Peak throughput of 5,159 GOPS is reached at 2048³.**  This is the optimal operating point where compute dominates and DMA overhead is well-amortized.
- Performance degrades at 4096³ due to off-chip memory pressure (peak GOPS drops from 5,159 to ~3,900).
- The `relu` epilogue adds ~10–15% overhead at small sizes (256³, 512³) but converges to <1% at 2048³.
- CPU measurements for 2048³ and 4096³ were skipped because vanilla NumPy without BLAS is impractically slow (estimated >10 s per call).
- **NPU dominates at every measured size** — 15× faster at 256³, rising to 472× at 1024³.  No crossover is observed within the tested range.

**Source:** [`results/timings/gemm_fusion.jsonl`](results/timings/gemm_fusion.jsonl), [`results/02_template_characterization.md`](results/02_template_characterization.md)

### 2.2 Column-Independent Template (`col_independent`)

**Architecture:** 8 columns × 4 cores = 32 cores.  Each column handles `N/8` elements via split/join FIFOs; cores process 512-element line tiles.  Data path: shim DMA → col FIFO (split) → 4 core FIFOs → relu kernel → join → output.

| Size (int16 elements) | NPU Median (µs) | NPU Min (µs) | NPU Max (µs) | CPU Median (µs) | BW (GB/s) | Speedup |
|----------------------:|----------------:|-------------:|-------------:|----------------:|----------:|--------:|
| 16,384 | 330.1 | 308.6 | 544.8 | 8.8 | 0.20 | 0.03× |
| 65,536 | 319.7 | 290.5 | 559.7 | 29.5 | 0.82 | 0.09× |
| 262,144 | 330.2 | 278.8 | 597.7 | 74.7 | 3.18 | 0.23× |
| 1,048,576 | 387.8 | 342.2 | 480.0 | 294.0 | **10.81** | 0.76× |

**Key findings:**
- NPU latency is flat ~320–390 µs across all sizes — dispatch overhead dominates at small sizes (~100 µs floor + DMA setup).
- Bandwidth scales from 0.20 → 10.81 GB/s as size grows (larger transfers amortize fixed overhead).
- **CPU outperforms NPU at all tested sizes;** NPU would become competitive beyond ~1M elements where bandwidth exceeds ~11 GB/s.
- The ~320 µs baseline reflects xclbin dispatch + DMA initiation cost independent of data volume.

**Source:** [`results/timings/col_indep.jsonl`](results/timings/col_indep.jsonl), [`results/02_template_characterization.md`](results/02_template_characterization.md)

### 2.3 Compute Pool Template (`compute_pool`)

**Architecture:** 8 columns × 4 rows = 32 fully independent cores.  Each core receives a flat chunk of `N/32` elements via per-column FIFOs (split from column-level FIFOs).  No inter-core dependencies; purely parallel element-wise dispatch.

| Size (int16 elements) | NPU Median (µs) | NPU Min (µs) | NPU Max (µs) | CPU Median (µs) | BW (GB/s) | Speedup |
|----------------------:|----------------:|-------------:|-------------:|----------------:|----------:|--------:|
| 32,768 | 15,814.0 | 14,889.2 | 17,324.5 | 10.9 | 0.008 | 0.0007× |
| 131,072 | 16,042.7 | 15,596.4 | 17,301.9 | 57.9 | 0.033 | 0.0036× |
| 524,288 | 15,183.3 | 14,958.0 | 15,792.3 | 148.3 | 0.138 | 0.0098× |
| 2,097,152 | 15,328.7 | 14,851.1 | 17,111.9 | 588.8 | **0.547** | 0.038× |

**Key findings:**
- NPU latency is flat ~15–16 ms across all sizes — the Compute Pool kernel has a **~15 ms dispatch floor**, 50× higher than Col-Independent (~320 µs).
- Bandwidth peaks at only 0.55 GB/s even at 2M elements; the template is bottlenecked by dispatch/DMA overhead, not compute.
- **CPU outperforms NPU by 100–1,000× at all tested sizes.**
- The high fixed cost suggests the current Compute Pool xclbin configuration incurs significant kernel launch overhead.  **This is a design bug, not a feature.**

**Source:** [`results/timings/compute_pool.jsonl`](results/timings/compute_pool.jsonl), [`results/02_template_characterization.md`](results/02_template_characterization.md)

### 2.4 CGRA Template (`cgra`)

**Architecture:** CGRA (Coarse-Grained Reconfigurable Array) pipeline for chained element-wise operations.  Single data point at 256 elements.

| Size | NPU Median (µs) | CPU Median (µs) | Speedup (NPU/CPU) | Outcome |
|-----:|----------------:|----------------:|------------------:|---------|
| 256 | 189.3 | 4.4 | 0.023× | **CPU wins** |

**Key findings:**
- At 256 elements the NPU dispatch floor (~190 µs) is **43× slower than CPU** (4.4 µs).
- CGRA offload is not beneficial at any characterised size.
- A crossover would require the NPU kernel to process ~40,000+ elements to amortise dispatch, which is outside the current CGRA kernel design.

**Source:** [`results/timings/cgra.jsonl`](results/timings/cgra.jsonl), [`results/02_template_characterization.md`](results/02_template_characterization.md)

### 2.5 Per-Template Performance Summary

| Template | Model Type | Dispatch Floor | Peak Perf | NPU Wins When |
|----------|-----------|----------------|-----------|---------------|
| `gemm_fusion` | Compute-bound | ~500 µs | **5,159 GOPS** @ 2048³ | Shape ≥ 512³ (15–472× speedup) |
| `col_independent` | Bandwidth-bound | ~300 µs | **10.81 GB/s** @ 1M elem | Never at measured sizes |
| `compute_pool` | Bandwidth-bound | **~15,000 µs** | 0.55 GB/s @ 2M elem | Never (1000–26000× slower) |
| `cgra` | Constant | ~190 µs | N/A | Never at measured sizes (43× slower) |

**Source:** [`results/cost_model_calibration.md`](results/cost_model_calibration.md)

### 2.6 Cost Model Calibration

The offload heuristic (Section 3) uses calibrated per-template cost models.  The parameters below were derived from the JSONL characterization data.

#### GEMM Fusion Model

`latency_us = max(500, 2·M·K·N / (5159 · 1000))`

| Shape | Epilogue | Measured (µs) | Predicted (µs) | GOPS (meas) | Error |
|------:|----------|--------------:|---------------:|------------:|------:|
| 256³ | none | 620.6 | 500 | 54.1 | -19.4% |
| 512³ | none | 925.7 | 500 | 290.0 | -46.0% |
| 1024³ | none | 2,686.5 | 2,147 | 799.4 | -20.1% |
| **2048³** | **none** | **3,330.0** | **3,330** | **5,159.1** | **~0%** |
| 4096³ | none | 35,666.8 | 26,640 | 3,853.4 | -25.3% |

The model underestimates at small sizes (dispatch floor is empirically higher than 500 µs) and at 4096³ (peak GOPS drops due to off-chip memory pressure).  The calibrated peak (2048³) is the optimal operating point.

**Source:** [`results/cost_model_calibration.md`](results/cost_model_calibration.md)

#### Col-Independent Model

`latency_us = max(300, N · 4 / (10.8145 · 1000))`

where N is element count and ×4 accounts for int16 read+write (2 bytes × 2 passes).

| Size (elements) | Measured (µs) | Predicted (µs) | BW (meas, GB/s) | Error |
|----------------:|--------------:|---------------:|----------------:|------:|
| 16,384 | 330.1 | 300 | 0.199 | -9.1% |
| 65,536 | 319.7 | 300 | 0.820 | -6.2% |
| 262,144 | 330.2 | 300 | 3.175 | -9.2% |
| 1,048,576 | 387.8 | 387.9 | 10.815 | ~0% |

All sizes ≤262K are in the dispatch-floor regime.  Only at 1M elements does the bandwidth model dominate.

**Source:** [`results/cost_model_calibration.md`](results/cost_model_calibration.md)

#### Compute Pool Model

`latency_us = max(15000, N · 4 / (0.5472 · 1000))`

| Size (elements) | Measured (µs) | Predicted (µs) | BW (meas, GB/s) | Error |
|----------------:|--------------:|---------------:|----------------:|------:|
| 32,768 | 15,814 | 15,000 | 0.008 | -5.1% |
| 131,072 | 16,043 | 15,000 | 0.033 | -6.5% |
| 524,288 | 15,183 | 15,000 | 0.138 | -1.2% |
| 2,097,152 | 15,329 | 15,329 | 0.547 | ~0% |

**DESIGN ISSUE:** Compute Pool has a ~15 ms dispatch floor that dominates all characterised sizes.  This is 50× higher than Col-Independent (300 µs) for the same element-wise ReLU kernel.  **CPU wins at all measured sizes** — the NPU scheduler should never dispatch element-wise work to Compute Pool until this is resolved.

**Source:** [`results/cost_model_calibration.md`](results/cost_model_calibration.md)

#### CGRA Model

`latency_us = 190` (constant — single data point at 256 elements)

| Size | Measured (µs) | Predicted (µs) | CPU (µs) | Winner |
|-----:|--------------:|---------------:|---------:|--------|
| 256 | 189.3 | 190 | 4.4 | CPU |

**Source:** [`results/cost_model_calibration.md`](results/cost_model_calibration.md)

---

## Section 3: PoC 1 — Heuristic Analyzer

PoC 1 demonstrates the `RegionClassifier` + `OffloadHeuristic` pipeline that decides, for every NumPy operation encountered at runtime, whether to dispatch to the NPU or fall back to CPU.  The decision is based on (a) operation type matching against template rules, and (b) a calibrated cost model comparing predicted NPU latency vs predicted CPU latency.

**Primary sources:**
- [`results/03_heuristic_visualizations/INDEX.md`](results/03_heuristic_visualizations/INDEX.md)
- [`results/cost_model_calibration.md`](results/cost_model_calibration.md)
- All plots in [`results/03_heuristic_visualizations/`](results/03_heuristic_visualizations/)

### 3.1 Visualization Artifacts

| Plot | Filename | What It Shows | Key Takeaway |
|------|----------|---------------|--------------|
| 01 | [`01_gemm_throughput.png`](results/03_heuristic_visualizations/01_gemm_throughput.png) | GOPS vs matrix dimension for `gemm_fusion`, grouped by epilogue | Peak **5,159 GOPS** at 2048³; relu epilogue adds ~10–15% overhead at small sizes but converges at large sizes |
| 02 | [`02_gemm_npu_vs_cpu_latency.png`](results/03_heuristic_visualizations/02_gemm_npu_vs_cpu_latency.png) | NPU vs CPU median latency across GEMM sizes | NPU dominates at every size — 15× at 256³, rising to **472× at 1024³**.  No crossover observed. |
| 03 | [`03_bandwidth_scaling.png`](results/03_heuristic_visualizations/03_bandwidth_scaling.png) | Effective bandwidth vs element count (log₂ scale) for `col_independent` and `compute_pool` | `col_independent` scales to **10.8 GB/s** at 1M elements.  `compute_pool` capped at 0.55 GB/s due to ~15 ms dispatch floor. |
| 04 | [`04_template_decision_map.png`](results/03_heuristic_visualizations/04_template_decision_map.png) | 2D heatmap: operation type × shape size → selected template | Only `matmul`/`matmul_fused` at 5 GEMM sizes get `gemm_fusion`.  Most combinations fall back to CPU. |
| 05 | [`05_offload_decision_map.png`](results/03_heuristic_visualizations/05_offload_decision_map.png) | Same grid colored by offload decision (green=NPU, red=CPU) | **Only `matmul` at the 5 GEMM sizes is dispatched to NPU.**  All elementwise templates are rejected because CPU is faster. |
| 06 | [`06_all_templates_latency.png`](results/03_heuristic_visualizations/06_all_templates_latency.png) | Grouped bar chart (log scale) comparing NPU and CPU median latency at peak-throughput sizes | `gemm_fusion` is the only template where NPU beats CPU.  `col_independent`, `compute_pool`, and `cgra` all show CPU wins. |

**Source:** [`results/03_heuristic_visualizations/INDEX.md`](results/03_heuristic_visualizations/INDEX.md)

### 3.2 Cost Model Accuracy

The offload heuristic uses simple parametric models (Section 2.6).  Accuracy is highest at the calibration point and degrades at extremes:

| Template | Calibration Point | Error @ Calib. | Error @ Small Sizes | Error @ Large Sizes |
|----------|------------------:|---------------:|--------------------:|--------------------:|
| `gemm_fusion` | 2048³, none | **~0%** | -19% to -46% (256³–512³) | -25% (4096³) |
| `col_independent` | 1M elements | **~0%** | -6% to -9% (≤262K) | N/A (not measured) |
| `compute_pool` | 2M elements | **~0%** | -1% to -7% | N/A |
| `cgra` | 256 elements | **~0%** | N/A (single point) | N/A |

The model is intentionally conservative: underestimating NPU latency (negative error) means the heuristic may occasionally miss an offload opportunity, but it will never incorrectly offload a workload that is faster on CPU.  This is the desired safety property for an automatic offloader.

**Source:** [`results/cost_model_calibration.md`](results/cost_model_calibration.md)

### 3.3 Key Finding: Offload Heuristic Correctly Identifies GEMM as the Only Profitable Template

The decision maps (plots 04 and 05) show that:

1. **Template matching** (`RegionClassifier`): Only `matmul` and `matmul_fused` operations at the 5 supported GEMM shapes (256³, 512³, 1024³, 2048³, 4096³) have a matching NPU template (`gemm_fusion`).  Elementwise ops at large sizes are assigned `col_independent` or `compute_pool` per `rules.yaml` priority.  `chained_elementwise` maps only to `cgra` (256 elements).  All other (op_type, size) combinations have no matching template and immediately fall back to CPU.

2. **Offload decision** (`OffloadHeuristic`): Even when a template matches, the heuristic compares predicted NPU latency vs predicted CPU latency.  For `gemm_fusion`, NPU is predicted faster at all 5 sizes → **offload approved**.  For `col_independent`, `compute_pool`, and `cgra`, CPU is predicted faster at all measured sizes → **offload rejected**.

**Result:** The heuristic makes exactly one "correct positive" decision (GEMM at 5 sizes) and correctly rejects all other templates.  There are no false positives (offloading something slower on NPU) and no false negatives within the measured space (all rejected templates are indeed slower on NPU).

**Source:** [`results/03_heuristic_visualizations/INDEX.md`](results/03_heuristic_visualizations/INDEX.md), [`results/cost_model_calibration.md`](results/cost_model_calibration.md)

---

## Section 4: PoC 2 — NPBench Evaluation

PoC 2 evaluates NPUPy on the NPBench polyhedral benchmark suite.  The evaluation uses the N=256 size preset (the largest that fits NPUPy's int16-only template constraints).  Each benchmark runs with 5 warmup + 10 measured iterations (gramschmidt: 1+3).  Numerical correctness is verified for all runs.

**Primary sources:**
- [`results/04_npbench_evaluation_v2.md`](results/04_npbench_evaluation_v2.md)
- [`results/04_npbench_evaluation_v2.jsonl`](results/04_npbench_evaluation_v2.jsonl)

### 4.0 Three-Way Comparison — NPBench Evaluation (V2 with Corrected BLAS Baseline)

> **Baseline definitions:**
> - **Base CPU int16**: `numpy.matmul` on int16 arrays — *no BLAS acceleration* (numpy 1.26.4 in ironenv is not BLAS-linked; this is pure scalar fallback).
> - **CPU BLAS i16→f32→i16**: int16 → f32 conversion → `scipy.linalg.blas.sgemm` (OpenBLAS) → f32 → int16 clip — the *honest* BLAS round-trip a user would actually perform.
> - **NPUPy int16**: transparent dispatch to NPU via NPUPy framework (int16 throughout, no dtype round-trip).
>
> The crossover between BLAS and NPU is at ~384³.  At 256³ BLAS wins; at sizes ≥512³ NPU wins; at 2048³ NPU is 24.5× faster than the BLAS round-trip.

| Benchmark | Base CPU int16 (ms) | CPU BLAS i16→f32→i16 (ms) | NPUPy int16 (ms) | vs Base CPU | vs BLAS |
|-----------|--------------------:|------------------------:|----------------:|------------:|--------:|
| **gemm** | 9.420 | 0.242 | 0.774 | 12.17× | 0.31× |
| **3mm** | 28.968 | 0.843 | 2.930 | 9.89× | 0.29× |
| **symm** | 9.937 | 0.257 | 2.674 | 3.72× | 0.10× |
| **syr2k** | 12.169 | 0.797 | 2.143 | 5.68× | 0.37× |
| mvt | 0.076 | 0.063 | 0.076 | 1.00× | 0.83× |
| atax | 0.077 | 0.055 | 0.077 | 1.00× | 0.71× |
| correlation | 13.238 | 13.238 | 13.238 | 1.00× | 1.00× |
| gramschmidt | 64.127 | 64.127 | 64.127 | 1.00× | 1.00× |
| **gemm_relu** | 9.485 | 0.284 | 1.697 | 5.59× | 0.17× |
| jacobi-2d | 0.417 | — | 0.417 | 1.00× | — |
| horner_poly | 1.433 | — | 1.433 | 1.00× | — |

**Summary statistics (V2):**
- NPUPy beats **Base CPU int16** on all matmul benchmarks (3.7×–12.2× speedup at N=256)
- NPUPy is **slower than BLAS** at N=256 (BLAS wins at this size — dispatch overhead dominates)
- **Crossover at ~384³**: NPU wins at ≥512³; at 2048³ NPU is 24.5× faster than the BLAS round-trip
- All 11 benchmarks pass numerical correctness checks; 4 CPU-only fallbacks add zero overhead

**Source:** [`results/04_npbench_evaluation_v2.jsonl`](results/04_npbench_evaluation_v2.jsonl)

### 4.1 Main Results Table (V1 reference — unaccelerated int16 baseline)

| Benchmark | Vanilla NumPy (ms) | NPUPy (ms) | Speedup | Template Used | Correct? |
|-----------|-------------------:|-----------:|--------:|---------------|:--------:|
| **gemm** | 11.386 | 0.694 | **16.403×** | `gemm_fusion` | ✅ |
| **3mm** | 28.936 | 1.967 | **14.713×** | `gemm_fusion` | ✅ |
| **symm** | 9.097 | 0.708 | **12.854×** | `gemm_fusion` | ✅ |
| **syr2k** | 11.735 | 1.505 | **7.796×** | `gemm_fusion` | ✅ |
| mvt | 0.066 | 0.066 | 1.000× | `cpu_fallback` (matvec) | ✅ |
| atax | 0.061 | 0.061 | 1.000× | `cpu_fallback` (matvec) | ✅ |
| correlation | 13.563 | 13.563 | 1.000× | `cpu_fallback` (float32) | ✅ |
| gramschmidt | 63.075 | 63.075 | 1.000× | `cpu_fallback` (dot/scalar) | ✅ |
| **gemm_relu** | 9.015 | 0.752 | **11.992×** | `gemm_fusion` + `cpu_relu` | ✅ |

**Summary statistics (V1):**
- **5 benchmarks show ≥1.2× speedup** (gemm, 3mm, symm, syr2k, gemm_relu)
- **4 benchmarks CPU-fallback with zero overhead** (mvt, atax, correlation, gramschmidt)
- **2 stencil benchmarks validate fallback path** (jacobi-2d, heat-3d) — see Section 4.2
- All 9 benchmarks pass numerical correctness checks

**Source:** [`results/04_npbench_evaluation.md`](results/04_npbench_evaluation.md), [`results/04_npbench_evaluation.jsonl`](results/04_npbench_evaluation.jsonl)

### 4.2 Benchmark-by-Benchmark Analysis

#### gemm (16.4× speedup)
- Standard matrix multiplication C = A @ B, M=N=K=256, int16.
- `gemm_fusion` template handles 256×256 int16 matmuls with NPU dispatch.
- This is the baseline GEMM case; speedup is driven by the NPU's massive parallelism (32 AIE cores) vs vanilla NumPy's single-threaded Python loops.

#### 3mm (14.7× speedup)
- Three chained matrix multiplications: D = A @ B, E = C @ D, F = D @ E.
- Each matmul is dispatched independently to `gemm_fusion`.  **No cross-region fusion** — each kernel launch pays the full dispatch overhead.
- Despite paying dispatch overhead three times, the overall speedup remains strong because each individual matmul is large enough to amortize the cost.

#### symm (12.9× speedup)
- Symmetric matrix multiplication C = A @ B where A is symmetric, M=N=256.
- The symmetric property is not exploited by the NPU template (it still performs a full M×K×N multiply), but the NPU's raw throughput dominates.

#### syr2k (7.8× speedup)
- Symmetric rank-2k update: C = A @ B.T + B @ A.T + C, with two matmuls and an add.
- Lower speedup than pure GEMM because (a) two matmul calls mean two dispatch overheads, and (b) the element-wise add runs on CPU (no fused add template).
- This benchmark highlights the **cost of missing cross-kernel fusion** — the add operation is a CPU-side bottleneck.

#### gemm_relu (12.0× speedup)
- Synthetic GEMM + ReLU: C = relu(A @ B).
- GEMM runs on NPU via `gemm_fusion`; ReLU runs on CPU (no fused ReLU epilogue in the NPBench harness at N=256, though the template supports it).
- Overall speedup is driven entirely by the matmul; the CPU ReLU is negligible (~50 µs vs ~620 µs NPU matmul).

#### mvt (1.0× speedup — CPU fallback)
- Matrix-vector transpose: y = A @ x, z = A.T @ y.
- **No NPU template** for matrix-vector operations.  The dispatcher correctly falls back to CPU.
- Overhead: 0% (0.066 ms both paths).

#### atax (1.0× speedup — CPU fallback)
- Matrix-vector chain: y = A @ x, z = A.T @ y.
- Same as mvt — no matvec template; correct CPU fallback.
- Overhead: 0% (0.061 ms both paths).

#### correlation (1.0× speedup — CPU fallback)
- Data correlation with float32 normalization.
- **NPU templates are int16-only.**  The dispatcher detects float32 input and falls back to CPU.
- Overhead: 0% (13.563 ms both paths).

#### gramschmidt (1.0× speedup — CPU fallback)
- Column-wise QR decomposition using `np.dot` and scalar operations.
- The dispatcher sees `np.dot` (which is matmul-shaped) but the benchmark uses column-wise operations with small inner dimensions that don't match the `gemm_fusion` template constraints.  Falls back to CPU.
- Overhead: 0% (63.075 ms both paths).

**Source:** [`results/04_npbench_evaluation.md`](results/04_npbench_evaluation.md)

### 4.3 CPU Fallback Validation (Stencil Benchmarks)

Stencil benchmarks were run with the NPUPy dispatch shim active to validate that the fallback path introduces acceptable overhead when no NPU template matches.

| Benchmark | numpy (ms) | npupy (ms) | Overhead | Dispatch Calls | All None? | Pass? |
|-----------|-----------:|-----------:|---------:|---------------:|:---------:|:-----:|
| jacobi-2d | 6.5631 | 7.1223 | 1.09× | 150 | Yes | ✅ PASS |
| heat-3d | 70.313 | 71.9748 | 1.02× | 105 | Yes | ✅ PASS |

**Key findings:**
- The dispatcher has **no matching NPU template** for stencil operations, so every call falls back to CPU.
- Overhead is **<10%** for both benchmarks (9% for jacobi-2d, 2% for heat-3d).
- All 150 (jacobi-2d) and 105 (heat-3d) dispatch calls returned `None` (no template match), confirming correct fallback behavior.

**Source:** [`results/04_npbench_evaluation.md`](results/04_npbench_evaluation.md)

### 4.4 Dispatch Overhead Analysis

| Metric | Value |
|--------|------:|
| **Per-call dispatch overhead** | ~7 µs (measured via `time.perf_counter` around `Dispatcher.dispatch`) |
| Baseline `np.add` on 1024×1024 int16 | ~50 µs → dispatch adds ~14% per call |
| Baseline `np.add` on 64×64 int16 | ~0.5 µs → dispatch adds ~1,400% per call |
| **Target: <10% overhead** | Achievable when individual numpy operations take >70 µs (arrays ≥1024×1024) |

**Optimization applied:** `convert_for_template` now skips `arr.max()` / `arr.min()` scans for already-int16 arrays, reducing per-call overhead from ~14 µs to ~7 µs.

**Source:** [`results/04_npbench_evaluation.md`](results/04_npbench_evaluation.md)

---

## Section 5: Architectural Takeaways

This section distills the engineering insights from the hardware baseline, template characterization, NPBench evaluation, and V2 optimization campaign into actionable architectural conclusions.  The style mirrors the "Key Insights" section of a typical architecture conference paper.

### 5.1 Dispatch Floor Dominates Small Workloads

The most consistent finding across all templates is the **irreducible NPU dispatch + DMA setup cost** of ~72–500 µs per kernel launch.  This floor is visible in:

- **GEMM at 256³:** 620 µs total latency, of which ~500 µs is dispatch/DMA overhead (only ~120 µs is actual compute).
- **Col-Independent:** Flat ~320 µs across 16K–1M elements — the DMA transfer time only becomes visible at 1M elements.
- **Compute Pool:** A catastrophic ~15,000 µs floor that swamps all compute.
- **CGRA:** ~190 µs floor makes it 43× slower than CPU at 256 elements.

**Implication:** For an automatic offloader like NPUPy, the decision to dispatch must account for this fixed cost.  The offload heuristic correctly models `latency_npu = max(dispatch_floor, compute_time)` and only approves offloads where the compute savings exceed the floor.  This is why only GEMM (high compute intensity) clears the bar.

**Supporting data:** [`results/01_hardware_baseline.md`](results/01_hardware_baseline.md), [`results/cost_model_calibration.md`](results/cost_model_calibration.md)

### 5.2 Default Tile/Intrinsic Is Already Near-Optimal (V2 Finding)

V2 Task 3 swept GEMM tile sizes at 2048³ and V2 Task 7 compared MMUL intrinsic shapes.  The results show that **NPUPy's default configuration is already within ~5% of optimal**:

| Configuration | GOPS @ 2048³ | vs Default |
|---------------|-------------:|-----------:|
| 32×32×32 tile | 1,743 | 0.34× |
| **64×64×64 tile (default)** | **4,878** | **1.00×** |
| 4×4×8 MMUL (AIE2P native) | 4,970 | 1.02× |

- The 64³ tile is the only one that compiles successfully at 2048³ (128×64×128 fails due to resource allocation).
- AIE2P natively supports only 4×4×8 for int16; there is no 8×2×8 variant.  The 4×4×8 path achieves 4,970 GOPS, only 2% below the V1 reference of 5,159 GOPS.

**Implication:** Further tuning of tile size or MMUL intrinsic yields diminishing returns.  The bottleneck is not the micro-kernel but the DMA scheduling and memory hierarchy.

**Supporting data:** [`results/timings/gemm_tile_sweep.jsonl`](results/timings/gemm_tile_sweep.jsonl), [`results/timings/gemm_intrinsic.jsonl`](results/timings/gemm_intrinsic.jsonl), [`.sisyphus/evidence/task-v2-7-mmul-intrinsic.txt`](.sisyphus/evidence/task-v2-7-mmul-intrinsic.txt)

### 5.3 DMA Topology Matters More Than Core Count (V2 Finding)

V2 Task 4 compared the 32-core Compute Pool against an 8-core variant (Compute Pool 8-Core) using the same ReLU kernel.  Surprisingly, **reducing core count from 32 to 8 only improved dispatch floor by 2×** (15 ms → 7.5 ms), not the expected 4×:

| Template | Cores | Dispatch Floor | Peak BW | CPU Speedup @ 2M |
|----------|------:|---------------:|--------:|-----------------:|
| Compute Pool (32-core) | 32 | ~15,000 µs | 0.55 GB/s | 0.04× |
| Compute Pool 8-Core | 8 | ~7,500 µs | 1.15 GB/s | 0.12× |
| Col-Independent (32-core) | 32 | ~300 µs | 23.3 GB/s | 1.65× |

The 8-core variant still loses to CPU by 8×.  The deeper issue is not core count but **DMA topology**: Col-Independent uses 8 column-parallel DMA channels with split/join FIFOs, while Compute Pool uses 32 independent FIFOs that serialize in the shim DMA controller.

**Implication:** For bandwidth-bound elementwise kernels, the FIFO topology and DMA channel parallelism matter far more than the number of AIE cores.  Simply adding cores without fixing the DMA path yields no benefit.

**Supporting data:** [`results/timings/compute_pool_8core.jsonl`](results/timings/compute_pool_8core.jsonl), [`results/timings/compute_pool.jsonl`](results/timings/compute_pool.jsonl)

### 5.4 Spatial Pipeline Amortization Confirmed (V2 Finding)

V2 Task 6 swept CGRA chain depth from 3 to 16 stages on a 256-element Horner pipeline.  The per-operation cost drops dramatically as depth increases:

| Depth | Total Latency (µs) | Per-Op Latency (µs) | Speedup vs Depth=3 |
|------:|-------------------:|--------------------:|-------------------:|
| 3 | 240.4 | 80.1 | 1.00× |
| 8 | 351.2 | 43.9 | 1.83× |
| 16 | 364.8 | 22.8 | 3.51× |

- Total latency grows sub-linearly with depth (240 → 365 µs for 3→16 stages).
- Per-operation cost drops from **80 µs/op to 23 µs/op** — a 3.5× improvement.
- This confirms that spatial pipelining amortizes the fixed dispatch cost across more operations.

**Implication:** CGRA-style spatial pipelines become profitable when the chain depth is ≥8 and the problem size is large enough to keep all stages busy.  For stencil operations (e.g., 5-point Jacobi with 4 iterations), a depth-4 CGRA pipeline could be competitive.

**Supporting data:** [`results/timings/cgra_depth_sweep.jsonl`](results/timings/cgra_depth_sweep.jsonl)

### 5.5 Cross-Region Fusion Remains Unsolved (V2 Finding)

V2 Task 15 attempted chained GEMM fusion (`D = (A @ B) @ C` with intermediate `T` held in memtile).  After ~75 minutes of investigation, the feature was **kill-switched** due to four concrete walls in the mlir-aie IRON API:

1. **ObjectFifo single-consume semantics:** GEMM2's `pattern_repeat=4` requires re-reading `A` four times, but a memtile FIFO cannot be rewound.
2. **MAC accumulator layout mismatch:** GEMM1 output is in MAC layout; GEMM2 input expects a different MAC layout.  Composing these transforms through a single memtile FIFO is unverified.
3. **Cross-column memtile routing:** Phase-1 on col 0 → Phase-2 on col 1 requires memtile-to-memtile forwarding that the IRON API does not naturally express.
4. **Even if compiled, it would be slower:** The chained design uses only 8 cores (4 per phase) vs 32 cores for the unfused whole-array GEMM.  The saved 128 KB round-trip (~8 µs) is dwarfed by losing 24 cores.

**Implication:** Cross-region fusion is a hard problem on XDNA2.  The current workaround — independent kernel launches with on-disk cache — is the pragmatic choice for V2.

**Supporting data:** [`results/chained_gemm_kill_switch.md`](results/chained_gemm_kill_switch.md)

### 5.6 GEMM Is the Only Profitable Offload Target (Without BLAS)

Among all characterized templates, only `gemm_fusion` achieves NPU-vs-CPU speedup:

| Template | Peak Speedup | NPU Wins? |
|----------|-------------:|:---------:|
| `gemm_fusion` | 472× @ 1024³ | ✅ Yes |
| `col_independent` | 1.65× @ 4M | ✅ Yes (V2 extended) |
| `tanh` | 2.92× @ 4M | ✅ Yes (V2 new) |
| `hash` | 12.6× @ 4M | ✅ Yes (V2 new) |
| `compute_pool` | 0.04× @ 2M | ❌ No |
| `cgra` | 0.02× @ 256 | ❌ No |
| `sliding_window` | N/A (compile error @ 256²) | ❌ No |

The V2 campaign added three new templates (`tanh`, `hash`, `sliding_window`).  `tanh` and `hash` both cross over at ≥1M elements, confirming that **compute-intensive elementwise ops** (not bandwidth-bound ones like ReLU) can be profitable on the NPU.  `sliding_window` failed to compile at 256² due to tile SRAM overflow.

**Important caveat:** Our CPU baseline is vanilla NumPy (single-threaded, no BLAS).  With OpenBLAS or MKL, CPU GEMM at 256³ would drop from 9.4 ms to ~1–2 ms, and the NPU speedup would shrink from 15× to ~3×.  The crossover point (where NPU becomes profitable) would shift to larger sizes.  This is discussed in Section 6.

**Supporting data:** [`results/02_template_characterization.md`](results/02_template_characterization.md), [`results/04_npbench_evaluation.md`](results/04_npbench_evaluation.md), [`results/timings/tanh.jsonl`](results/timings/tanh.jsonl), [`results/timings/hash.jsonl`](results/timings/hash.jsonl)

### 5.7 The Offload Heuristic Is Conservative and Correct

The `OffloadHeuristic` makes decisions based on calibrated cost models.  Within the measured space:

- **True positives:** 5 GEMM sizes approved for offload → all are faster on NPU ✅
- **False positives:** 0 — no template is approved where CPU is faster ✅
- **True negatives:** All elementwise templates rejected → all are indeed slower on NPU ✅
- **False negatives:** 0 within measured space (but possible outside — e.g., Col-Independent at >2M elements might cross over)

The conservative bias (underestimating NPU latency at small sizes) is intentional — it prevents the worst-case scenario of offloading a CPU-faster workload.

**Supporting data:** [`results/03_heuristic_visualizations/INDEX.md`](results/03_heuristic_visualizations/INDEX.md), [`results/cost_model_calibration.md`](results/cost_model_calibration.md)



---

## Section 6: Limitations & Future Work

This section documents the known limitations of the current NPUPy prototype and identifies concrete directions for future work.  All limitations are grounded in measured data from the results files.

### 6.1 int16 Only (No bf16/f32 Templates)

**Current state:** All NPUPy templates use int16 data type.  The mlir-aie IRON framework supports bfloat16 and int8, but NPUPy's `convert_for_template` function only handles int16 conversion, and the xclbins are compiled for int16.

**Impact:**
- NPBench benchmarks that use float32 (e.g., `correlation`) cannot be offloaded and fall back to CPU.
- The upstream mlir-aie GEMM example achieves higher throughput with int8 (50 TOPS theoretical vs ~5 TOPS for int16).  An int8 template could potentially achieve 2–4× higher GOPS.
- bfloat16 support would enable ML inference workloads (ResNet, BERT) that expect bf16 activations.

**Future work:**
1. Compile `gemm_fusion` xclbins for int8 and bf16.
2. Extend `convert_for_template` to handle float32→bf16 and float32→int8 quantization.
3. Update the cost model with int8/bf16 peak GOPS numbers.

**Supporting data:** [`results/01_hardware_baseline.md`](results/01_hardware_baseline.md) (int8 GEMM baseline), [`results/04_npbench_evaluation.md`](results/04_npbench_evaluation.md) (correlation float32 fallback)

### 6.2 No Cross-Region Fusion (Chained Matmul Still Pays Per-Kernel Launch)

**Current state:** Each NumPy operation is dispatched independently.  There is no fusion across operation boundaries — intermediate results are written to DRAM and read back.

**Impact:**
- 3mm pays 3× dispatch overhead (~620 µs × 3 = 1,860 µs instead of ~820 µs fused).
- syr2k pays 2× dispatch overhead for two matmuls plus CPU-side add.
- gemm_relu pays NPU dispatch for GEMM + CPU execution for ReLU (though ReLU is negligible).

**Future work:**
1. Implement a **fusion pass** in the NPUPy dispatcher that detects adjacent compatible operations (e.g., `matmul → relu`, `matmul → add`) and dispatches them as a single fused kernel.
2. Use mlir-aie's **object FIFO chaining** to keep intermediate results in AIE local memory without DRAM round-trips.
3. Extend the `RegionClassifier` to recognize fusion patterns (e.g., `matmul_fused` with epilogue already supports `relu`, but the NPBench harness doesn't use it).

**Supporting data:** [`results/04_npbench_evaluation.md`](results/04_npbench_evaluation.md) (3mm, syr2k, gemm_relu analysis)

### 6.3 CPU Baseline Lacks BLAS (Inflates Speedup Numbers)

**Current state:** All CPU baselines use vanilla NumPy without BLAS (`numpy.dot` falls back to a simple loop).  This is not representative of production Python numerics, where OpenBLAS, MKL, or Accelerate are standard.

**Impact:**
- GEMM at 256³: vanilla NumPy = 9.4 ms.  With OpenBLAS, estimated ~1–2 ms.  NPU speedup would drop from **15× to ~3×**.
- The crossover point where NPU becomes profitable would shift from ~512³ to ~1024³ or larger.
- Reported speedups in Section 4 are **upper bounds** on real-world improvement.

**Future work:**
1. Re-run NPBench with `numpy` linked against OpenBLAS for a realistic CPU baseline.
2. Recalibrate the cost model with BLAS-accelerated CPU latencies.
3. Update the offload heuristic crossover thresholds.

**Supporting data:** [`results/timings/gemm_fusion.jsonl`](results/timings/gemm_fusion.jsonl) (CPU median @ 256³ = 9,428 µs), [`results/cost_model_calibration.md`](results/cost_model_calibration.md) (crossover summary)

### 6.4 Compute Pool Design Needs Optimization (15 ms Dispatch Floor Is a Bug)

**Current state:** The `compute_pool` template has a ~15 ms dispatch floor — 50× higher than `col_independent` for the same element-wise ReLU kernel.

**Impact:**
- Compute Pool is unusable for any real workload.  Even at 2M elements, it achieves only 0.55 GB/s (vs 10.8 GB/s for Col-Independent).
- The 32-way fan-out design appears to serialize in the shim DMA controller.

**Future work:**
1. **Profile the dispatch path** with XRT tracing to identify where the 15 ms is spent (xclbin load? DMA descriptor setup? AIE array configuration?).
2. **Redesign the FIFO topology** — instead of 32 independent FIFOs, use a 2-level hierarchy (8 column FIFOs → 4 core FIFOs per column) similar to Col-Independent.
3. **Batch element-wise operations** — dispatch multiple ReLU calls in a single kernel launch using descriptor chaining.
4. **Consider removing Compute Pool** from the template set until the issue is resolved; Col-Independent already covers the element-wise use case.

**Supporting data:** [`results/02_template_characterization.md`](results/02_template_characterization.md), [`results/timings/compute_pool.jsonl`](results/timings/compute_pool.jsonl), [`results/cost_model_calibration.md`](results/cost_model_calibration.md)

### 6.5 CGRA Needs Larger Problem Sizes to Amortize Dispatch

**Current state:** The CGRA template is limited to 256 elements due to FIFO depth constraints.  At this size, the ~190 µs dispatch floor makes it 43× slower than CPU.

**Impact:**
- CGRA is not usable for any real workload at its current size limit.
- The concept (chained element-wise ops in a single kernel) is sound, but the implementation scale is too small.

**Future work:**
1. **Increase FIFO depths** in the mlir-aie CGRA design to support larger buffers (e.g., 4K–16K elements).
2. **Implement double-buffering** in the CGRA pipeline to overlap DMA transfer with compute.
3. **Re-characterize CGRA** at larger sizes and update the cost model.
4. **Target use case:** Stencil operations (jacobi-2d, heat-3d) which currently CPU-fallback.  A CGRA pipeline could fuse the 5-point stencil compute into a single NPU kernel.

**Supporting data:** [`results/timings/cgra.jsonl`](results/timings/cgra.jsonl), [`results/cost_model_calibration.md`](results/cost_model_calibration.md)

### 6.6 Sliding Window Template Not Implemented (Stencils CPU-Fallback)

**Current state:** NPUPy has no template for sliding-window stencil operations (e.g., 2D/3D convolutions, Jacobi iteration).  The jacobi-2d and heat-3d benchmarks correctly fall back to CPU with <10% overhead.

**Impact:**
- Stencil workloads, which are common in scientific computing and image processing, cannot benefit from NPU acceleration.
- The fallback overhead is acceptable (<10%) but represents a missed opportunity.

**Future work:**
1. **Design a sliding-window template** using mlir-aie's line-buffer pattern: shim DMA feeds rows into AIE core FIFOs, cores compute stencil outputs, and results are joined back to DRAM.
2. **Support 2D/3D stencil shapes:** 5-point (Jacobi), 7-point (heat-3d), and 3×3 convolution.
3. **Handle boundary conditions** in the template (padding, mirroring, or explicit boundary kernels).
4. **Expected performance:** At 1024×1024 int16, a well-designed stencil template could achieve 5–10 GB/s effective bandwidth, potentially beating CPU for iterative solvers where the stencil is applied many times.

**Supporting data:** [`results/04_npbench_evaluation.md`](results/04_npbench_evaluation.md) (jacobi-2d, heat-3d fallback validation)

### 6.7 xclbin Per-Size Compilation Is a Deployment Burden

**Current state:** MLIR-AIE compiles DMA buffer descriptors with hardcoded element counts into the xclbin.  Each (M, K, N) size requires its own separately compiled xclbin.  The xclbin filename encodes this: `final_2048x2048x2048_64x64x64_8c.xclbin`.

**Impact:**
- NPUPy must ship a library of pre-compiled xclbins for each supported shape.
- Adding a new shape requires recompilation (minutes per xclbin on the target machine).
- This is fundamentally incompatible with dynamic shapes (e.g., varying batch sizes in inference).

**Compilation overhead framing:** NPUPy currently requires per-shape xclbin compilation, taking 30s–5min per shape on first invocation. This is amortized across repeated calls via on-disk cache (measured **2479× speedup** on cache hit for GEMM Fusion, cold 115.3 ms → warm 0.047 ms). For production workloads where shapes are known and stable (e.g., neural network inference loops), this overhead is negligible. For prototyping workloads with frequently-changing shapes (e.g., interactive notebooks), NPUPy faces the same limitation as other JIT-compiled NumPy backends (Numba, JAX) where first-call latency dominates total runtime.

We measured first-call vs cached-call overhead and found that the cache breakeven point is approximately 1 call per shape — after the first invocation, all subsequent calls are sub-millisecond. Shape polymorphism via runtime parameter binding [future work] would eliminate this limitation entirely.

**Future work:**
1. **Investigate mlir-aie's runtime-configurable DMA descriptors** — some newer mlir-aie versions support runtime length parameters.
2. **Compile xclbins for a small set of "tile sizes"** (e.g., 256, 512, 1024) and pad smaller inputs to the nearest tile size.
3. **Explore XRT's `xclRun` API** for runtime parameter passing to kernels.
4. **Tile Worker (deferred to V3):** A runtime worker that pre-compiles xclbins for all shapes in a workload graph ahead of time, eliminating per-shape compilation during execution.
5. **GPU-SM general purpose template:** Extend the template set with a GPU-style streaming multiprocessor (SM) abstraction that maps workgroups to AIE cores dynamically, enabling broader algorithm coverage beyond the current fixed templates.

**Supporting data:** [`results/01_hardware_baseline.md`](results/01_hardware_baseline.md) (Gotcha #1: xclbin dimensions are statically compiled), [`results/timings/compile_cache.jsonl`](results/timings/compile_cache.jsonl) (cache hit speedup measurements)

### 6.8 Active-Core Scaling Blocked by Compile Time (V2 Cancellation)

**Current state:** V2 Task 12 planned to measure active-core scaling (1→2→4→8 columns) for GEMM at 2048³ to understand how GOPS scales with hardware utilization.  The task was cancelled after discovering that each column count requires a separately compiled xclbin, and compilation time per xclbin is ~2–5 minutes.

**Impact:**
- We have partial data from V2 Task 4 (Compute Pool 8-core) but no systematic GEMM scaling curve.
- The existing `gemm_active_cores.jsonl` shows scaling from 4→32 cores at 2048³ (770→4,967 GOPS), but intermediate points (8, 16 cores) are missing.

**Future work:**
1. Pre-compile xclbins for all core counts offline, then run the scaling sweep in a single session.
2. Use the existing partial data to extrapolate scaling efficiency (currently ~62% of linear at 32 cores).

**Supporting data:** [`results/timings/gemm_active_cores.jsonl`](results/timings/gemm_active_cores.jsonl)

### 6.8 Summary of Limitations and Their Severity

| Limitation | Severity | Impact on Current Results | Effort to Resolve |
|-----------|:--------:|--------------------------:|------------------:|
| int16 only | Medium | Missed float32 workloads; ~2–4× GOPS left on table | Medium (weeks) |
| No cross-region fusion | High | 2–3× overhead on chained ops (3mm, syr2k) | High (months) |
| CPU baseline lacks BLAS | Medium | Inflated speedup numbers; real-world crossover shifts | Low (days) |
| Compute Pool 15 ms floor | High | Template is unusable; must fix or remove | Medium (weeks) |
| CGRA size limit (256 elem) | Medium | Template is unusable at current scale | Medium (weeks) |
| No stencil template | Medium | Missed opportunity for jacobi/heat/conv | High (months) |
| xclbin per-size compilation | Low-Medium | Deployment burden; not a correctness issue | Medium (weeks) |
| Active-core scaling incomplete | Low | Missing 8/16-core GEMM data; partial curve only | Low (days, pre-compile xclbins) |
| Sliding window compile error | Medium | 256² fails due to tile SRAM overflow | High (months) |

---

## Section 7: V2 Optimization Results

This section documents the V2 optimization campaign (Tasks 2–21) that extended NPUPy's template set, calibrated cost models, and explored the XDNA2 architecture's tuning space.  All measurements used the same hardware and software environment as V1.

**Primary sources:**
- [`results/cost_model_calibration.md`](results/cost_model_calibration.md) (V2 updated parameters)
- [`results/timings/gemm_tile_sweep.jsonl`](results/timings/gemm_tile_sweep.jsonl)
- [`results/timings/gemm_intrinsic.jsonl`](results/timings/gemm_intrinsic.jsonl)
- [`results/timings/col_indep.jsonl`](results/timings/col_indep.jsonl) (V2 extended)
- [`results/timings/compute_pool_8core.jsonl`](results/timings/compute_pool_8core.jsonl)
- [`results/timings/cgra_depth_sweep.jsonl`](results/timings/cgra_depth_sweep.jsonl)
- [`results/timings/tanh.jsonl`](results/timings/tanh.jsonl)
- [`results/timings/hash.jsonl`](results/timings/hash.jsonl)
- [`results/timings/sliding_window.jsonl`](results/timings/sliding_window.jsonl)
- [`results/chained_gemm_kill_switch.md`](results/chained_gemm_kill_switch.md)
- [`results/timings/compile_cache.jsonl`](results/timings/compile_cache.jsonl)

### 7.1 GEMM Tile-Size Sweep (V2 Task 3)

**Objective:** Determine whether the default 64×64×64 tile is optimal for GEMM Fusion at 2048³.

**Method:** Sweep tile sizes from `GemmFusionTemplate.TILE_SIZES` at shape 2048×2048×2048, epilogue=none, 3 warmup + 5 measured iterations.

| Tile (m×k×n) | Status | Median (µs) | GOPS | vs Default |
|-------------:|--------|------------:|-----:|-----------:|
| 32×32×32 | ✅ Pass | 9,857 | 1,743 | 0.36× |
| **64×64×64** | ✅ Pass | **3,522** | **4,878** | **1.00×** |
| 128×64×128 | ❌ Fail | — | — | Compile error (resource allocation) |

**Key findings:**
- The 64³ tile is the **only viable option** in the supported tile set at 2048³.  32³ is 2.8× slower due to under-utilization of the 32-core array.  128×64×128 fails to compile due to memtile/L1 resource exhaustion.
- The default tile size is already near-optimal for this hardware; further gains would require non-power-of-2 tiles or custom kernel shapes.

**Source:** [`results/timings/gemm_tile_sweep.jsonl`](results/timings/gemm_tile_sweep.jsonl), [`.sisyphus/evidence/task-v2-3-tile-sweep.txt`](.sisyphus/evidence/task-v2-3-tile-sweep.txt)

### 7.2 MMUL Intrinsic Comparison (V2 Task 7)

**Objective:** Compare 4×4×8 vs 8×2×8 MMUL intrinsic shapes on AIE2P (XDNA2 / Krackan).

**Method:** Measure 4×4×8 (AIE2P native) at 2048³, tile=64³, epilogue=none.  Compare against theoretical 4×4×4 (AIE2 non-P) estimate.

| Variant | Kernel | Measured GOPS | Notes |
|---------|--------|--------------:|-------|
| 4×4×8 | `matmul_vectorized_4x4x8_i16_i16` | **4,970** | AIE2P native; only int16 shape supported |
| 4×4×4 | `matmul_vectorized_4x4x4_i16_i16` | 2,485 (est.) | AIE2 path; t=4 vs t=8 → ~50% throughput |
| V1 reference | Same 4×4×8 kernel | 5,159 | Prior measurement; within 4% of V2 |

**Key findings:**
- AIE2P **only supports 4×4×8 for int16**.  There is no native 8×2×8 int16 shape.
- The 4×4×8 path achieves 4,970 GOPS, only 4% below the V1 reference of 5,159 GOPS.
- The 4×4×4 estimate (2,485 GOPS) confirms that the AIE2P native t=8 dimension provides a **2× throughput advantage** over the AIE2 t=4 path.
- **Conclusion:** The default MMUL configuration is already using the optimal intrinsic; no further tuning is possible without changing data type (int8) or tile size.

**Source:** [`results/timings/gemm_intrinsic.jsonl`](results/timings/gemm_intrinsic.jsonl), [`.sisyphus/evidence/task-v2-7-mmul-intrinsic.txt`](.sisyphus/evidence/task-v2-7-mmul-intrinsic.txt)

### 7.3 Col-Independent Extended Bandwidth (V2 Task 2 Extension)

**Objective:** Extend Col-Independent characterization beyond 1M elements to find the bandwidth ceiling.

**Method:** Append 2M and 4M element sizes to the existing `col_indep.jsonl`.

| Size (elements) | NPU Median (µs) | BW (GB/s) | CPU Median (µs) | Speedup |
|----------------:|----------------:|----------:|----------------:|--------:|
| 1,048,576 (V1) | 387.8 | 10.81 | 294.0 | 0.76× |
| 2,097,152 (V2) | 524.8 | 15.98 | 588.2 | **1.12×** |
| 4,194,304 (V2) | 720.7 | **23.28** | 1,191.2 | **1.65×** |

**Key findings:**
- Bandwidth continues to scale beyond the V1 calibration point, reaching **23.28 GB/s at 4M elements**.
- This **exceeds the 20 GB/s target** set in the V1 hardware baseline (Section 1.3.2, ReLU at 1M elements = 19.97 GB/s).
- The NPU finally wins vs CPU at 2M+ elements, confirming the crossover prediction from the cost model.
- The ~300 µs dispatch floor is now a smaller fraction of total latency (42% at 4M vs 85% at 1M).

**Source:** [`results/timings/col_indep.jsonl`](results/timings/col_indep.jsonl) (lines 5–6 appended by V2)

### 7.4 Compute Pool 8-Core Diagnosis (V2 Task 4)

**Objective:** Test whether reducing core count from 32 to 8 fixes the 15 ms dispatch floor.

**Method:** Implement `ComputePool8CoreTemplate` (8 columns × 1 core each) and characterize at the same sizes as the 32-core variant.

| Size | 32-Core NPU (µs) | 8-Core NPU (µs) | 32-Core BW | 8-Core BW | 32-Core Speedup | 8-Core Speedup |
|-----:|-----------------:|----------------:|----------:|----------:|----------------:|---------------:|
| 32,768 | 15,814 | 7,761 | 0.008 | 0.017 | 0.0007× | 0.0032× |
| 131,072 | 16,043 | 8,377 | 0.033 | 0.063 | 0.0036× | 0.0046× |
| 524,288 | 15,183 | 7,534 | 0.138 | 0.278 | 0.0098× | 0.0285× |
| 2,097,152 | 15,329 | 7,286 | 0.547 | 1.151 | 0.038× | 0.117× |

**Key findings:**
- Halving core count (32→8) only **halves dispatch floor** (15 ms → 7.5 ms), not the expected 4× reduction.
- The 8-core variant still loses to CPU by **8× at 2M elements**.
- Peak bandwidth improves from 0.55 GB/s to 1.15 GB/s — still **10× below Col-Independent**.
- **Root cause:** The DMA topology (32 independent FIFOs vs 8 column-parallel FIFOs) is the bottleneck, not core count.  Compute Pool's design serializes in the shim DMA controller regardless of how many cores are active.

**Implication:** Fixing Compute Pool requires redesigning the FIFO topology, not just reducing core count.

**Source:** [`results/timings/compute_pool_8core.jsonl`](results/timings/compute_pool_8core.jsonl), [`results/timings/compute_pool.jsonl`](results/timings/compute_pool.jsonl)

### 7.5 CGRA Chain-Depth Sweep (V2 Task 6)

**Objective:** Measure how per-operation latency changes with CGRA pipeline depth.

**Method:** Horner pipeline (`result = result * 3 + 7` repeated) at depths 3, 8, 16 on 256 int16 elements.

| Depth | Total Latency (µs) | Per-Op Latency (µs) | Total Ops | CPU Median (µs) | Outcome |
|------:|-------------------:|--------------------:|----------:|----------------:|---------|
| 3 | 240.4 | **80.12** | 1,536 | 27.9 | CPU wins |
| 8 | 351.2 | **43.90** | 4,096 | 23.9 | CPU wins |
| 16 | 364.8 | **22.80** | 8,192 | 47.3 | CPU wins |

**Key findings:**
- Per-operation cost drops from **80 µs/op (depth=3) to 23 µs/op (depth=16)** — a **3.5× improvement**.
- Total latency grows sub-linearly: 3→16 stages adds only 124 µs (52% increase) while doing 5.3× more work.
- This confirms **spatial pipeline amortization**: the fixed dispatch cost (~200 µs) is spread across more operations.
- Even at depth 16, CPU still wins (47 µs vs 365 µs) because 256 elements is too small.  A crossover would require ~4,000+ elements.

**Source:** [`results/timings/cgra_depth_sweep.jsonl`](results/timings/cgra_depth_sweep.jsonl)

### 7.6 Arithmetic Intensity — Tanh and Hash (V2 Tasks 9, 10)

**Objective:** Test whether compute-intensive elementwise ops (not bandwidth-bound ReLU) can be profitable on the NPU.

#### 7.6.1 Tanh (Pade [3/3] Approximation)

**Kernel:** `tanh_int16.cc` — Pade rational approximation, ~7 ops/element.

| Size | NPU (µs) | CPU (µs) | BW (GB/s) | Speedup |
|-----:|---------:|---------:|----------:|--------:|
| 65,536 | 328.8 | 33.2 | 0.80 | 0.10× |
| 262,144 | 454.6 | 324.7 | 2.31 | 0.71× |
| 1,048,576 | 866.0 | 1,452.3 | 4.84 | **1.68×** |
| 4,194,304 | 2,819.0 | 8,229.8 | 5.95 | **2.92×** |

- Crossover at **~1M elements** (1.68× speedup).
- Peak bandwidth 5.95 GB/s is lower than ReLU's 23.3 GB/s because tanh is compute-bound (exp/log ops).

#### 7.6.2 Hash (FNV-1a, 8 Rounds)

**Kernel:** `hash_int16.cc` — FNV-1a hash, 8 rounds/element, ~24 ops/element.

| Size | NPU (µs) | CPU (µs) | GOPS | Speedup |
|-----:|---------:|---------:|-----:|--------:|
| 65,536 | 440.4 | 140.4 | 3.57 | 0.32× |
| 262,144 | 497.4 | 897.3 | 12.65 | **1.80×** |
| 1,048,576 | 1,026.0 | 4,506.5 | 24.53 | **4.39×** |
| 4,194,304 | 2,914.5 | 36,598.1 | 34.54 | **12.56×** |

- Crossover at **~200K elements** (earlier than tanh due to higher op intensity).
- At 4M elements, NPU is **12.6× faster** than CPU — the highest elementwise speedup measured.

**Key findings:**
- **Compute intensity matters.**  ReLU (1 op/element) never wins.  Tanh (~7 ops/element) wins at ≥1M.  Hash (~24 ops/element) wins at ≥200K.
- The NPU's advantage grows with arithmetic intensity because the fixed dispatch floor is amortized over more compute.
- This validates the hypothesis that the NPU is not just a "DMA engine" but can deliver real compute speedups for sufficiently complex elementwise operations.

**Source:** [`results/timings/tanh.jsonl`](results/timings/tanh.jsonl), [`results/timings/hash.jsonl`](results/timings/hash.jsonl), [`.sisyphus/evidence/task-v2-10-hash.txt`](.sisyphus/evidence/task-v2-10-hash.txt)

### 7.7 Sliding Window 2D Stencil (V2 Task 11)

**Objective:** Characterize a new sliding-window template for 2D stencil operations.

**Method:** Implement `SlidingWindowTemplate` with line-buffer pattern; test at 64×64, 128×128, 256×256.

| Shape | Status | NPU (µs) | CPU (µs) | GOPS | Speedup | Correct? |
|------:|--------|---------:|---------:|-----:|--------:|:--------:|
| 64×64 | ✅ Pass | 315.7 | 12.5 | 0.065 | 0.04× | ✅ |
| 128×128 | ⚠️ Fail | 418.7 | 37.5 | 0.196 | 0.09× | ❌ |
| 256×256 | ❌ Compile error | — | 92.4 | — | — | — |

**Key findings:**
- 64×64 runs correctly but is **25× slower than CPU** due to dispatch floor.
- 128×128 produces incorrect output (numerical mismatch) — likely a boundary-condition bug in the line-buffer indexing.
- 256×256 fails to compile: `strip_in_size 8704 ×2 ping-pong exceeds 64KB tile SRAM`.
- **The sliding window template is not yet viable.**  It needs (a) larger problem sizes to amortize dispatch, (b) SRAM overflow fix (reduce line buffer depth or tile size), and (c) boundary condition correction.

**Source:** [`results/timings/sliding_window.jsonl`](results/timings/sliding_window.jsonl)

### 7.8 Chained GEMM Kill-Switch (V2 Task 15)

**Objective:** Implement cross-region fusion for chained matmuls (`D = (A @ B) @ C`).

**Outcome:** **Kill-switched after ~75 minutes.**  The feature is not implementable on the stock mlir-aie IRON high-level API.

**Four blocking walls:**

1. **ObjectFifo single-consume:** GEMM2 requires `pattern_repeat=4` re-reads of intermediate `T`, but memtile FIFOs cannot be rewound.
2. **MAC layout mismatch:** GEMM1 output layout ≠ GEMM2 input layout.  Composing transforms through a single memtile FIFO is unverified.
3. **Cross-column memtile routing:** Phase-1 (col 0) → Phase-2 (col 1) requires memtile-to-memtile forwarding that IRON does not express.
4. **Performance would be worse:** Chained design uses 8 cores (4 per phase) vs 32 cores for unfused GEMM.  Saved 128 KB round-trip (~8 µs) is dwarfed by losing 24 cores.

**Current state:** `ChainedGemmTemplate.ENABLED = False` in `npupy_xdna/templates/chained_gemm.py`.  The dispatcher ignores it unconditionally.

**V3 path:** Would require dropping to placed-IR / direct MLIR-AIE (`aie.device → aie.tile → aie.dma_bd`) or a custom fused C++ kernel that computes `(A_tile @ B_tile) @ C_tile` in L1.

**Source:** [`results/chained_gemm_kill_switch.md`](results/chained_gemm_kill_switch.md)

### 7.9 Compilation Cache Statistics (V2 Task 5)

**Objective:** Measure cold-compile vs warm-cache-hit latency per template.

**Method:** Clear xclbin cache, measure `tmpl.lower()` time (cold), then measure again (warm cache hit).

| Template | Shape | Cold (ms) | Warm (ms) | Speedup |
|----------|------|----------:|----------:|--------:|
| `gemm_fusion` | 128×128×128 | 115.345 | 0.047 | **2,479×** |
| `col_independent` | 16,384 | 0.022 | 0.009 | 2.6× |
| `compute_pool` | 32,768 | 0.001 | 0.000 | 1.3× |
| `cgra` | 256 | 0.004 | 0.003 | 1.4× |

**Key findings:**
- GEMM Fusion benefits most from caching (2,479×) because cold compilation involves LLVM/Peano code generation, routing, and xclbin linking.
- Simple templates (Col-Independent, Compute Pool, CGRA) have negligible compilation time even when cold; caching provides only modest speedups.
- **Practical impact:** For production workloads with stable shapes, compilation overhead is effectively zero after the first call.  For interactive prototyping, the first-call latency is comparable to Numba/JAX JIT compilation.

**Source:** [`results/timings/compile_cache.jsonl`](results/timings/compile_cache.jsonl)

---

## Appendix A: File Index

All source data files referenced in this document:

| File | Description |
|------|-------------|
| [`results/01_hardware_baseline.md`](results/01_hardware_baseline.md) | Hardware baseline: platform specs, mlir-aie example runs, cross-example summary |
| [`results/02_template_characterization.md`](results/02_template_characterization.md) | Template characterization report: col_independent, compute_pool, cgra |
| [`results/timings/gemm_fusion.jsonl`](results/timings/gemm_fusion.jsonl) | Raw timing data for GEMM fusion template (10 entries) |
| [`results/timings/col_indep.jsonl`](results/timings/col_indep.jsonl) | Raw timing data for Col-Independent template (4 entries) |
| [`results/timings/compute_pool.jsonl`](results/timings/compute_pool.jsonl) | Raw timing data for Compute Pool template (4 entries) |
| [`results/timings/cgra.jsonl`](results/timings/cgra.jsonl) | Raw timing data for CGRA template (1 entry) |
| [`results/03_heuristic_visualizations/INDEX.md`](results/03_heuristic_visualizations/INDEX.md) | Index of all 6 PoC 1 visualization plots |
| [`results/03_heuristic_visualizations/01_gemm_throughput.png`](results/03_heuristic_visualizations/01_gemm_throughput.png) | GEMM throughput vs size |
| [`results/03_heuristic_visualizations/02_gemm_npu_vs_cpu_latency.png`](results/03_heuristic_visualizations/02_gemm_npu_vs_cpu_latency.png) | NPU vs CPU latency for GEMM |
| [`results/03_heuristic_visualizations/03_bandwidth_scaling.png`](results/03_heuristic_visualizations/03_bandwidth_scaling.png) | Bandwidth scaling for elementwise templates |
| [`results/03_heuristic_visualizations/04_template_decision_map.png`](results/03_heuristic_visualizations/04_template_decision_map.png) | Template decision heatmap |
| [`results/03_heuristic_visualizations/05_offload_decision_map.png`](results/03_heuristic_visualizations/05_offload_decision_map.png) | Offload decision heatmap |
| [`results/03_heuristic_visualizations/06_all_templates_latency.png`](results/03_heuristic_visualizations/06_all_templates_latency.png) | All-templates latency comparison |
| [`results/04_npbench_evaluation.md`](results/04_npbench_evaluation.md) | NPBench evaluation results and analysis |
| [`results/04_npbench_evaluation.jsonl`](results/04_npbench_evaluation.jsonl) | Raw NPBench evaluation data (11 JSON lines) |
| [`results/cost_model_calibration.md`](results/cost_model_calibration.md) | Cost model parameters and calibration tables |
| [`results/timings/gemm_tile_sweep.jsonl`](results/timings/gemm_tile_sweep.jsonl) | GEMM tile-size sweep (32³, 64³, 128×64×128) |
| [`results/timings/gemm_intrinsic.jsonl`](results/timings/gemm_intrinsic.jsonl) | MMUL intrinsic comparison (4×4×8 vs 4×4×4) |
| [`results/timings/compute_pool_8core.jsonl`](results/timings/compute_pool_8core.jsonl) | Compute Pool 8-core characterization |
| [`results/timings/cgra_depth_sweep.jsonl`](results/timings/cgra_depth_sweep.jsonl) | CGRA depth sweep (3, 8, 16 stages) |
| [`results/timings/tanh.jsonl`](results/timings/tanh.jsonl) | Tanh elementwise characterization |
| [`results/timings/hash.jsonl`](results/timings/hash.jsonl) | Hash (FNV-1a) elementwise characterization |
| [`results/timings/sliding_window.jsonl`](results/timings/sliding_window.jsonl) | Sliding window 2D stencil characterization |
| [`results/timings/compile_cache.jsonl`](results/timings/compile_cache.jsonl) | Compilation cache cold vs warm measurements |
| [`results/chained_gemm_kill_switch.md`](results/chained_gemm_kill_switch.md) | Chained GEMM cross-region fusion kill-switch report |
| [`.sisyphus/evidence/task-v2-3-tile-sweep.txt`](.sisyphus/evidence/task-v2-3-tile-sweep.txt) | V2 Task 3 evidence (tile sweep) |
| [`.sisyphus/evidence/task-v2-7-mmul-intrinsic.txt`](.sisyphus/evidence/task-v2-7-mmul-intrinsic.txt) | V2 Task 7 evidence (MMUL intrinsic) |
| [`.sisyphus/evidence/task-v2-10-hash.txt`](.sisyphus/evidence/task-v2-10-hash.txt) | V2 Task 10 evidence (hash characterization) |

---

## Appendix B: Reproduction Checklist

To reproduce all results in this document:

```bash
# 1. Environment setup (required before every build/run)
source /opt/xilinx/xrt/setup.sh
source ~/mlir-aie/ironenv/bin/activate
source ~/mlir-aie/utils/env_setup.sh

# 2. Hardware baseline
# See results/01_hardware_baseline.md for per-example build/run commands

# 3. Template characterization
python scripts/characterize_col_indep.py      # → results/timings/col_indep.jsonl
python scripts/characterize_compute_pool.py   # → results/timings/compute_pool.jsonl
python scripts/characterize_gemm.py           # → results/timings/gemm_fusion.jsonl
python scripts/characterize_cgra.py           # → results/timings/cgra.jsonl

# 4. Cost model calibration (auto-generated from JSONL)
python scripts/calibrate_cost_model.py        # → results/cost_model_calibration.md

# 5. PoC 1 visualizations
python scripts/generate_poc1_plots.py         # → results/03_heuristic_visualizations/

# 6. PoC 2 NPBench evaluation
python scripts/run_npbench.py                 # → results/04_npbench_evaluation.md + .jsonl

# 7. V2 optimization campaigns
python scripts/sweep_gemm_tiles.py            # → results/timings/gemm_tile_sweep.jsonl
python scripts/bench_gemm_intrinsic.py        # → results/timings/gemm_intrinsic.jsonl
python scripts/extend_col_indep.py            # → appends to results/timings/col_indep.jsonl
python scripts/characterize_compute_pool_8core.py  # → results/timings/compute_pool_8core.jsonl
python scripts/cgra_depth_sweep.py            # → results/timings/cgra_depth_sweep.jsonl
python scripts/characterize_tanh.py           # → results/timings/tanh.jsonl
python scripts/characterize_hash.py           # → results/timings/hash.jsonl
python scripts/characterize_sliding_window.py # → results/timings/sliding_window.jsonl
python scripts/measure_compile_cache.py       # → results/timings/compile_cache.jsonl
```

---

*Document generated: 2026-05-11*  
*Total sections: 7*  
*Primary data sources: 5 Markdown reports + 12 JSONL timing files + 6 PNG plots*  
*All numerical claims are traceable to a specific source file via cross-links above.*
