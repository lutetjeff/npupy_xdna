# Template Characterization Report

Hardware: AMD Ryzen AI 7 350 (Krackan Point), XDNA2 NPU  
Date: Mon May 11 2026  
Methodology: 5 warmup + 10 measured iterations, `np.random.default_rng(42)`, int16 relu (elementwise_unary)

---

## Column-Independent Template (`col_independent`)

Architecture: 8 columns × 4 cores = 32 cores. Each column handles `N/8` elements via split/join FIFOs; cores process 512-element line tiles. Data path: shim DMA → col FIFO (split) → 4 core FIFOs → relu kernel → join → output.

Script: `scripts/characterize_col_indep.py`  
Data: `results/timings/col_indep.jsonl`

| Size (int16) | NPU median (µs) | NPU min (µs) | NPU max (µs) | CPU median (µs) | BW (GB/s) | Speedup |
|-------------:|----------------:|-------------:|-------------:|----------------:|----------:|--------:|
|       16 384 |           330.1 |            — |            — |             8.8 |      0.20 |   0.03× |
|       65 536 |           319.7 |            — |            — |            29.5 |      0.82 |   0.09× |
|      262 144 |           330.2 |            — |            — |            74.7 |      3.18 |   0.23× |
|    1 048 576 |           387.8 |            — |            — |           294.0 |     10.81 |   0.76× |

**Observations:**

- NPU latency is flat ~320–390 µs across all sizes — dispatch overhead dominates at small sizes (~100 µs floor + DMA setup).
- Bandwidth scales from 0.20 → 10.81 GB/s as size grows (larger transfers amortize fixed overhead).
- CPU outperforms NPU at all tested sizes; NPU would become competitive beyond ~1 M elements where bandwidth exceeds ~11 GB/s.
- The ~320 µs baseline reflects xclbin dispatch + DMA initiation cost independent of data volume.

---

## Compute Pool Template (`compute_pool`)

Architecture: 8 columns × 4 rows = 32 fully independent cores. Each core receives a flat chunk of `N/32` elements via per-column FIFOs (split from column-level fifos). No inter-core dependencies; purely parallel elementwise dispatch.

Script: `scripts/characterize_compute_pool.py`  
Data: `results/timings/compute_pool.jsonl`

| Size (int16) | NPU median (µs) | NPU min (µs) | NPU max (µs) | CPU median (µs) | BW (GB/s) | Speedup |
|-------------:|----------------:|-------------:|-------------:|----------------:|----------:|--------:|
|       32 768 |        15 814.0 |            — |            — |            10.9 |      0.01 |   0.00× |
|      131 072 |        16 042.7 |            — |            — |            57.9 |      0.03 |   0.00× |
|      524 288 |        15 183.3 |            — |            — |           148.3 |      0.14 |   0.01× |
|    2 097 152 |        15 328.7 |            — |            — |           588.8 |      0.55 |   0.04× |

**Observations:**

- NPU latency is flat ~15–16 ms across all sizes — the ComputePool kernel has a much higher dispatch floor than ColIndependent (~15 ms vs ~320 µs).
- Bandwidth peaks at only 0.55 GB/s even at 2 M elements; the template is bottlenecked by dispatch/DMA overhead, not compute.
- CPU outperforms NPU by 100–1000× at all tested sizes.
- The high fixed cost suggests the current ComputePool xclbin configuration incurs significant kernel launch overhead; optimization target: reduce per-dispatch overhead or batch larger workloads.

---

*Bandwidth formula: `2 × N × 2 bytes / (npu_median_us × 1e-6) / 1e9` (read + write, int16 = 2 bytes)*
