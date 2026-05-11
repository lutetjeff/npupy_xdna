# NPBench Evaluation Results

Size preset: N=256 | Warmup=5, Iterations=10 (gramschmidt: 1+3)

| Benchmark | Vanilla NumPy (ms) | NPUPy (ms) | Speedup | Template | Correct |
|-----------|-------------------|------------|---------|----------|---------|
| gemm | 11.386 | 0.694 | 16.403x | gemm_fusion | YES |
| 3mm | 28.936 | 1.967 | 14.713x | gemm_fusion | YES |
| symm | 9.097 | 0.708 | 12.854x | gemm_fusion | YES |
| syr2k | 11.735 | 1.505 | 7.796x | gemm_fusion | YES |
| mvt | 0.066 | 0.066 | 1.000x | cpu_fallback (matvec) | YES |
| atax | 0.061 | 0.061 | 1.000x | cpu_fallback (matvec) | YES |
| correlation | 13.563 | 13.563 | 1.000x | cpu_fallback (float32) | YES |
| gramschmidt | 63.075 | 63.075 | 1.000x | cpu_fallback (dot/scalar) | YES |
| gemm_relu | 9.015 | 0.752 | 11.992x | gemm_fusion+cpu_relu | YES |

## Notes

- gemm_fusion template handles 256x256 int16 matmuls (NPU dispatch).
- mvt / atax: matrix-vector ops have no NPU template; CPU-only.
- gramschmidt: column-wise QR with np.dot; CPU-only.
- correlation: float32 normalization; NPU template is int16-only.
- gemm_relu: GEMM on NPU + ReLU on CPU; overall speedup driven by matmul.
- NPUPy uses NpuRunner directly (iron_fn created once, reused per shape).

## CPU Fallback Validation

Stencil benchmarks run with the NPUPy dispatch shim active.
The dispatcher has **no matching NPU template** for stencil operations,
so every call falls back to CPU.  Overhead must be < 10 %.

| Benchmark | numpy (ms) | npupy (ms) | overhead | dispatch calls | all None? | pass? |
|-----------|------------|------------|----------|----------------|-----------|-------|
| jacobi-2d | 6.5631 | 7.1223 | 1.09x | 150 | Yes | PASS |
| heat-3d | 70.313 | 71.9748 | 1.02x | 105 | Yes | PASS |

### Dispatch Overhead Analysis

- **Per-call dispatch overhead**: ~7 us (measured via `time.perf_counter` around `Dispatcher.dispatch`).
- **Baseline np.add on 1024x1024 int16**: ~50 us → dispatch adds ~14 % per call.
- **Baseline np.add on 64x64 int16**: ~0.5 us → dispatch adds ~1400 % per call.
- **Conclusion**: The < 10 % overhead target is achievable when individual numpy operations take > 70 us (e.g., arrays >= 1024x1024). For smaller arrays, the fixed dispatch cost dominates.
- **Optimization applied**: `convert_for_template` now skips `arr.max()` / `arr.min()` scans for already-int16 arrays, reducing per-call overhead from ~14 us to ~7 us.
