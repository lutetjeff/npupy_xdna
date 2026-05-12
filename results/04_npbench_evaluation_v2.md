# NPBench V2 Evaluation Results

Size preset: N=256 | Warmup=5, Iterations=10

| Benchmark | int16 CPU (ms) | BLAS CPU (ms) | NPUPy (ms) | vs int16 | vs BLAS | Template | Correct |
|-----------|---------------|--------------|------------|---------|--------|----------|---------|
| gemm | 9.420 | 0.242 | 0.774 | 12.17x | 0.31x | gemm_fusion | YES |
| 3mm | 28.968 | 0.843 | 2.930 | 9.89x | 0.29x | gemm_fusion | YES |
| symm | 9.937 | 0.257 | 2.674 | 3.72x | 0.10x | gemm_fusion | YES |
| syr2k | 12.169 | 0.797 | 2.143 | 5.68x | 0.37x | gemm_fusion | YES |
| mvt | 0.076 | 0.063 | 0.076 | 1.00x | 0.83x | cpu_fallback (matvec) | YES |
| atax | 0.077 | 0.055 | 0.077 | 1.00x | 0.71x | cpu_fallback (matvec) | YES |
| correlation | 13.238 | 13.238 | 13.238 | 1.00x | 1.00x | cpu_fallback (float32) | YES |
| gramschmidt | 64.127 | 64.127 | 64.127 | 1.00x | 1.00x | cpu_fallback (dot/scalar) | YES |
| gemm_relu | 9.485 | 0.284 | 1.697 | 5.59x | 0.17x | gemm_fusion+cpu_relu | YES |
| jacobi-2d | 0.417 | — | 0.417 | 1.00x | — | cpu_fallback (sliding_window unavailable: RuntimeError) | YES |
| horner_poly | 1.433 | — | 1.433 | 1.00x | — | cpu_fallback (no NPU template) | YES |

## Notes

- **int16_cpu_ms**: NumPy int16 baseline (no BLAS acceleration – numpy doesn't dispatch int16 to BLAS).
- **blas_cpu_ms**: Honest baseline: int16→f32 cast, BLAS matmul, clip→int16. `—` = no BLAS path.
- **speedup_vs_int16**: NPUPy speedup vs int16 numpy (favorable; int16 numpy is slow).
- **speedup_vs_blas**: NPUPy speedup vs BLAS round-trip (honest; BLAS is fast). `—` = not applicable.
- mvt/atax: matvec; no NPU template → speedup_vs_blas reflects BLAS matvec round-trip.
- correlation/gramschmidt: already float32 path internally → blas_cpu_ms ≈ int16_cpu_ms.
- jacobi-2d: 5-point stencil; SlidingWindowTemplate attempted (256×256 is supported shape).
- horner_poly: degree-8 polynomial, 256K elements; no NPU dispatch path.
