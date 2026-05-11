# Cost Model Calibration — V2 (Wave-2 + New Templates)

Generated: 2026-05-11

## Summary of Changes (V1 → V2)

- Added `ci_low` / `ci_high` fields to `CostEstimate` (reporting-only; offload decision unchanged)
- Added **tanh** predictor calibrated from `tanh.jsonl` (4 data points)
- Added **hash** predictor (placeholder — no `hash.jsonl` yet)
- Added **sliding_window** CI (placeholder)
- Updated **CGRA** predictor with depth-interpolated model from `cgra_depth_sweep.jsonl`

## Per-Template Parameters

| Template       | Model Type       | dispatch_floor_us | Peak Perf          | CI rel-std | Source                        |
|----------------|-----------------|-------------------|--------------------|-----------:|-------------------------------|
| gemm_fusion    | Compute-bound   | 500 µs            | 5 159 GOPS         | 10.1%      | 10-pt calibration (min/max)   |
| col_independent| Bandwidth-bound | 300 µs            | 10.81 GB/s         | 14.6%      | 6-pt calibration (min/max)    |
| compute_pool   | Bandwidth-bound | 15 000 µs         | 0.55 GB/s          |  2.9%      | 4-pt calibration (stable)     |
| cgra           | Depth-interp    | see table below   | N/A                | 18.0%      | 3-depth sweep (actual timings)|
| sliding_window | Compute-bound   | 500 µs            | 1 000 GOPS         | 15.0%      | placeholder (no jsonl)        |
| tanh           | Bandwidth-bound | 300 µs            | 5.95 GB/s          |  9.9%      | 4-pt calibration (min/max)    |
| hash           | Bandwidth-bound | 300 µs            | 3.00 GB/s          | 15.0%      | placeholder (no jsonl)        |

CI half-width = 2 × rel-std × predicted_latency_us. CI is **reporting only**.

---

## GEMM Fusion

**Model:** `latency_us = max(500, 2·M·K·N / (5159 · 1000))`
**CI:** ±20.2% of prediction (rel-std 10.1%, calibrated from (max−min)/4/median across 10 points)

| Shape       | Epilogue | Measured (µs) | Predicted (µs) | GOPS (meas) | Error   |
|-------------|----------|---------------|----------------|-------------|---------|
| 256³        | none     | 620.6         | 500            | 54.1        | -19.4%  |
| 256³        | relu     | 729.3         | 500            | 46.0        | -31.4%  |
| 512³        | none     | 925.7         | 500            | 290.0       | -46.0%  |
| 512³        | relu     | 1082.0        | 500            | 248.1       | -53.8%  |
| 1024³       | none     | 2686.5        | 2147           | 799.4       | -20.1%  |
| 1024³       | relu     | 2691.0        | 2147           | 798.0       | -20.2%  |
| 2048³       | none     | 3330.0        | 3330           | 5159.1      | ~0%     |
| 2048³       | relu     | 3364.4        | 3330           | 5106.3      | -1.0%   |
| 4096³       | none     | 35666.8       | 26640          | 3853.4      | -25.3%  |
| 4096³       | relu     | 34824.1       | 26640          | 3946.7      | -23.5%  |

---

## Col-Independent

**Model:** `latency_us = max(300, N · 4 / (10.8145 · 1000))`
**CI:** ±29.2% of prediction (rel-std 14.6%)

| Size (elements) | Measured (µs) | Predicted (µs) | BW (meas, GB/s) | Error  |
|-----------------|---------------|----------------|-----------------|--------|
| 16 384          | 330.1         | 300            | 0.199           | -9.1%  |
| 65 536          | 319.7         | 300            | 0.820           | -6.2%  |
| 262 144         | 330.2         | 300            | 3.175           | -9.2%  |
| 1 048 576       | 387.8         | 387.8          | 10.815          | ~0%    |
| 2 097 152       | 524.8         | 775.7          | 15.984          | +47.8% |
| 4 194 304       | 720.7         | 1551.3         | 23.279          | +115%  |

**Note:** Model saturates at 10.81 GB/s; larger sizes exceed predicted (CPU already wins at all sizes).

---

## Compute Pool

**Model:** `latency_us = max(15000, N · 4 / (0.5472 · 1000))`
**CI:** ±5.8% of prediction (rel-std 2.9%, very stable floor-dominated behaviour)

| Size (elements) | Measured (µs) | Predicted (µs) | BW (meas, GB/s) | Error  |
|-----------------|---------------|----------------|-----------------|--------|
| 32 768          | 15 814        | 15 000         | 0.008           | -5.1%  |
| 131 072         | 16 043        | 15 000         | 0.033           | -6.5%  |
| 524 288         | 15 183        | 15 000         | 0.138           | -1.2%  |
| 2 097 152       | 15 329        | 15 329         | 0.547           | ~0%    |

**DESIGN ISSUE:** 15 ms dispatch floor dominates all sizes. CPU wins at all measured sizes.

---

## CGRA (V2: depth-interpolated)

**Model:** Linear interpolation over depth-sweep table; depth inferred from `len(region.inputs)`.
**CI:** ±36% of prediction (rel-std 18.0% from actual per-sample timings across 3 depths)

| Depth | n_elements | Measured median (µs) | Measured std (µs) | rel-std |
|------:|-----------:|---------------------:|------------------:|--------:|
| 3     | 256        | 240.36               | 9.8               | 4.1%    |
| 8     | 256        | 351.19               | 65.7              | 18.7%   |
| 16    | 256        | 364.77               | 113.3             | 31.1%   |

**Interpolation example (depth=4):** 240.36 + (4−3)/(8−3)·(351.19−240.36) = **262.5 µs**

**Note (V1→V2):** Previous constant was 190.0 µs from a single cgra.jsonl data point. Depth-sweep shows 240–365 µs depending on chain depth. The single-point measurement (189.3 µs at depth inferred ≈1) was a lighter kernel; depth ≥3 chains are costlier.

---

## Tanh (NEW in V2)

**Model:** `latency_us = max(300, N · 4 / (5.9516 · 1000))`
**CI:** ±19.8% of prediction (rel-std 9.9% from tanh.jsonl (max−min)/4/median)

| Size (elements) | Measured median (µs) | Predicted (µs) | Eff BW (GB/s) | Error  |
|-----------------|---------------------:|---------------:|--------------:|--------|
| 65 536          | 328.8                | 300            | 0.797         | -8.7%  |
| 262 144         | 454.6                | 300            | 2.307         | -34%   |
| 1 048 576       | 866.0                | 705.1          | 4.844         | -18.6% |
| 4 194 304       | 2 819.0              | 2819.5         | 5.952         | ~0%    |

**Note:** Compute-bound at large sizes (tanh involves exp, capped at ~5.95 GB/s vs. col_indep's 10.81 GB/s). Large error at 262K elements is the dispatch-to-compute transition regime.

---

## Hash (NEW in V2 — placeholder)

**Model:** `latency_us = max(300, N · 4 / (3.0 · 1000))`
**CI:** ±30% of prediction (placeholder 15% rel-std)

No `hash.jsonl` data collected. Model assumes compute-bound at 3.0 GB/s (more expensive than tanh, less than tanh at peak). To be refined once hardware measurements are available.

---

## Sliding Window (placeholder — unchanged from V1)

**Model:** `latency_us = 500 + H·W·5 / (1000 · 1000)`
**CI:** ±30% of prediction (placeholder 15% rel-std)

No `sliding_window.jsonl` data. Placeholder maintained.

---

## CPU Baseline Model

| Op Type         | Model                    | Reference                                 |
|-----------------|--------------------------|-------------------------------------------|
| matmul/matmul_fused | O(flops), anchored at 1024³ | 9.4 ms @ 256³, 80 ms @ 512³, 1.27 s @ 1024³ |
| elementwise_*   | O(N) at 2.80e-4 µs/elem  | col_indep 1M measurement (294 µs)         |

---

## Crossover Summary

| Template        | NPU wins when               | Notes                                               |
|-----------------|-----------------------------|-----------------------------------------------------|
| gemm_fusion     | Shape ≥ 512³               | 15–471× speedup at 512³–1024³                      |
| col_independent | ≥ 2M elements              | Crossover around 2M (speedup 1.12×)                |
| compute_pool    | Never                       | 15 ms floor; CPU always faster                      |
| cgra            | Never at measured sizes     | 240–365 µs vs CPU 4–48 µs                          |
| tanh            | ≥ 1M elements              | Crossover ~1M (speedup 1.68×); 2.92× at 4M         |
| hash            | Unknown (no data)           | Placeholder — needs calibration                     |
| sliding_window  | Unknown (no data)           | Placeholder — needs calibration                     |
