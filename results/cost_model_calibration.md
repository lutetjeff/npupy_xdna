# Cost Model Calibration — Wave-2 Characterisation Data

Generated: 2026-05-11

## Per-Template Parameters

| Template       | Model Type       | dispatch_floor_us | Peak Perf          | Confidence Notes                                   |
|----------------|-----------------|-------------------|--------------------|----------------------------------------------------|
| gemm_fusion    | Compute-bound   | 500 µs            | 5 159 GOPS         | High for N≥2048³; dispatch-floor-dominated for N≤512³ |
| col_independent| Bandwidth-bound | 300 µs            | 10.81 GB/s         | Approaches 1.0 at 1M elements; low at ≤65K elements  |
| compute_pool   | Bandwidth-bound | 15 000 µs         | 0.55 GB/s          | Always near 0 — dispatch floor dominates all sizes   |
| cgra           | Constant        | 190 µs (total)    | N/A                | Fixed 0.6 — single data point                        |

## GEMM Fusion

**Model:** `latency_us = max(500, 2·M·K·N / (5159 · 1000))`

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

**Note:** The model underestimates at small sizes (dispatch floor is empirically higher than 500 µs — actual overhead at 256³ is ~620 µs) and at 4096³ (peak GOPS drops from 5159 to ~3900, likely due to off-chip memory pressure). The calibrated peak (2048³) is the optimal operating point.

## Col-Independent

**Model:** `latency_us = max(300, N · 4 / (10.8145 · 1000))`

where N is element count and ×4 accounts for int16 read+write (2 bytes × 2 passes).

| Size (elements) | Measured (µs) | Predicted (µs) | BW (meas, GB/s) | Error  |
|-----------------|---------------|----------------|-----------------|--------|
| 16 384          | 330.1         | 300            | 0.199           | -9.1%  |
| 65 536          | 319.7         | 300            | 0.820           | -6.2%  |
| 262 144         | 330.2         | 300            | 3.175           | -9.2%  |
| 1 048 576       | 387.8         | 387.9          | 10.815          | ~0%    |

**Note:** All sizes ≤262K are in the dispatch-floor regime. Only at 1M elements does the bandwidth model dominate. CPU beats NPU at all measured sizes (CPU at 1M: 294 µs vs NPU 388 µs).

## Compute Pool

**Model:** `latency_us = max(15000, N · 4 / (0.5472 · 1000))`

| Size (elements) | Measured (µs) | Predicted (µs) | BW (meas, GB/s) | Error  |
|-----------------|---------------|----------------|-----------------|--------|
| 32 768          | 15 814        | 15 000         | 0.008           | -5.1%  |
| 131 072         | 16 043        | 15 000         | 0.033           | -6.5%  |
| 524 288         | 15 183        | 15 000         | 0.138           | -1.2%  |
| 2 097 152       | 15 329        | 15 329         | 0.547           | ~0%    |

**DESIGN ISSUE:** Compute Pool has a ~15 ms dispatch floor that dominates all characterised sizes. This is 50× higher than Col-Independent (300 µs) for the same element-wise ReLU kernel. The design appears to have a serialisation bottleneck in the dispatch path. **CPU wins at all measured sizes** — the NPU scheduler should never dispatch element-wise work to Compute Pool until this is resolved.

## CGRA

**Model:** `latency_us = 190` (constant — single data point at 256 elements)

| Size | Measured (µs) | Predicted (µs) | CPU (µs) | Winner |
|------|---------------|----------------|----------|--------|
| 256  | 189.3         | 190            | 4.4      | CPU    |

**Note:** At 256 elements the NPU dispatch floor (~190 µs) is 43× slower than CPU (4.4 µs). CGRA offload is not beneficial at any characterised size. A crossover would require the NPU kernel to process ~40 000+ elements to amortise dispatch, which is outside the current CGRA kernel design.

## CPU Baseline Model

| Op Type         | Model                    | Reference                              |
|-----------------|--------------------------|----------------------------------------|
| matmul/matmul_fused | O(flops), anchored at 1024³ | 9.4 ms @ 256³, 80 ms @ 512³, 1.27 s @ 1024³ |
| elementwise_*   | O(N) at 2.80e-4 µs/elem  | col_indep 1M measurement (294 µs)      |

## Crossover Summary

| Template        | NPU wins when               | Notes                                          |
|-----------------|-----------------------------|------------------------------------------------|
| gemm_fusion     | Shape ≥ 512³               | 15–471× speedup at 512³–1024³                 |
| col_independent | Never at measured sizes     | CPU beats NPU at all 4 characterised sizes     |
| compute_pool    | Never                       | 15 ms floor makes NPU 1000–26000× slower       |
| cgra            | Never at measured sizes     | 43× slower than CPU at 256 elements            |
