# kernels/
C++ AIE kernel sources compiled by Peano (llvm-aie) for XDNA2 AIE cores.
- `gemm_int16_*.cc` — MMUL-based matmul kernels with epilogue variants
- `relu_int16.cc` — vector ReLU for int16
- `tanh_int16.cc` — Horner polynomial tanh approximation
- `hash_int16.cc` — FNV-1a hash (high arithmetic intensity)
- `stencil_5pt_int16.cc` — 5-point 2D stencil
- `cgra_*.cc` — individual pipeline stage kernels
