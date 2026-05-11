# NPBench V2 Evaluation Results

Size preset: N=256 | Warmup=5, Iterations=10

| Benchmark | int16 CPU (ms) | BLAS CPU (ms) | NPUPy (ms) | vs int16 | vs BLAS | Template | Correct |
|-----------|---------------|--------------|------------|---------|--------|----------|---------|
| gemm | 9.400 | 10.959 | 0.766 | 12.27x | 14.31x | gemm_fusion | YES |
| 3mm | 30.019 | 32.665 | 2.089 | 14.37x | 15.64x | gemm_fusion | YES |
| symm | 9.691 | 10.964 | 0.756 | 12.81x | 14.50x | gemm_fusion | YES |
| syr2k | 12.308 | 22.035 | 1.441 | 8.54x | 15.29x | gemm_fusion | YES |
| mvt | 0.056 | 0.089 | 0.056 | 1.00x | 1.57x | cpu_fallback (matvec) | YES |
| atax | 0.068 | 0.089 | 0.068 | 1.00x | 1.31x | cpu_fallback (matvec) | YES |
| correlation | 13.551 | 13.551 | 13.551 | 1.00x | 1.00x | cpu_fallback (float32) | YES |
| gramschmidt | 63.989 | 63.989 | 63.989 | 1.00x | 1.00x | cpu_fallback (dot/scalar) | YES |
| gemm_relu | 9.576 | 10.991 | 0.790 | 12.12x | 13.91x | gemm_fusion+cpu_relu | YES |
| jacobi-2d | 0.319 | — | 0.319 | 1.00x | — | cpu_fallback (sliding_window unavailable: RuntimeError) | YES |
| horner_poly | 1.274 | — | 1.274 | 1.00x | — | cpu_fallback (no NPU template) | YES |

## Notes

- **int16_cpu_ms**: NumPy int16 baseline (no BLAS acceleration – numpy doesn't dispatch int16 to BLAS).
- **blas_cpu_ms**: Honest baseline: int16→f32 cast, BLAS matmul, clip→int16. `—` = no BLAS path.
- **speedup_vs_int16**: NPUPy speedup vs int16 numpy (favorable; int16 numpy is slow).
- **speedup_vs_blas**: NPUPy speedup vs BLAS round-trip (honest; BLAS is fast). `—` = not applicable.
- mvt/atax: matvec; no NPU template → speedup_vs_blas reflects BLAS matvec round-trip.
- correlation/gramschmidt: already float32 path internally → blas_cpu_ms ≈ int16_cpu_ms.
- jacobi-2d: 5-point stencil; SlidingWindowTemplate attempted (256×256 is supported shape).
- horner_poly: degree-8 polynomial, 256K elements; no NPU dispatch path.
