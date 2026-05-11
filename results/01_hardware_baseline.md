# NPU Hardware Baseline — Preliminary Results
**Date:** 2026-05-11  
**Machine:** ASUS Vivobook S16 M5606KA  
**CPU:** AMD Ryzen AI 7 350 w/ Radeon 860M (16 logical cores)  
**NPU:** NPU Krackan 1 (XDNA2 architecture, BDF 0000:64:00.1)  
**RAM:** 14,580 MB  
**OS:** Ubuntu Resolute Raccoon (Linux 6.19.0-5-generic, x86_64)

## Environment

| Component | Version / Path |
|---|---|
| XRT | 2.23.0 (hash 99b8b976) |
| amdxdna driver | 2.23.0_20260218 |
| NPU Firmware | 255.0.11.69 |
| mlir-aie | ~/mlir-aie (IRON interface) |
| Peano (llvm-aie) | ~/mlir-aie/ironenv/lib/python3.13/site-packages/llvm-aie |
| Host compiler | g++-15 (system default; g++-13 absent) |
| Device target | `npu2` (required for Krackan/Strix Point — NOT `npu`/Phoenix) |

### Environment Setup (required before every build/run)
```bash
source /opt/xilinx/xrt/setup.sh
source ~/mlir-aie/ironenv/bin/activate
source ~/mlir-aie/utils/env_setup.sh
```

---

## Examples Run

All examples from `~/mlir-aie/programming_examples/`. Built with `NPU2=1` Makefile flag.  
Methodology: 5 warmup iterations discarded, 10 measured iterations, numerical correctness verified (PASS) for all runs.

---

## Example 1 — `basic/matrix_multiplication/whole_array`

**Operation:** Integer GEMM, A(MxK) @ B(KxN) → C(MxN)  
**Dtype:** int8 input → int32 output  
**Hardware config:** 8 AIE columns × 4 rows = 32 cores, 64×64×64 per-tile kernel  
**B layout:** column-major (`--b_col_maj 1`) — required flag, see Gotchas  
**Theoretical peak:** 50 TOPS (int8 on Krackan)

| Size (M=K=N) | Avg Latency | Avg GOPS | Min Latency | Peak GOPS | % of 50 TOPS |
|---|---|---|---|---|---|
| 1024 | 1,475 µs | 1,456 | 331 µs | 6,488 | 13.0% |
| 2048 | 2,171 µs | 7,914 | 1,997 µs | 8,603 | **17.2%** |
| 4096 | 50,480 µs | 2,723 | 22,490 µs | 6,111 | 5.4% |

**GOPS formula:** `2 * M * K * N / latency_us / 1000`

**Notes:**
- 2048³ is the throughput sweet spot. DMA overhead amortizes well (~24 MB total A+B+C transfer), variance is tight (1997–2303 µs across 10 runs).
- 4096³ degrades badly. A+B+C at i8/i32 is ~96 MB; the average is dragged down by memory-bandwidth saturation. Best-case single-run still achieves 6.1 TOPS.
- 1024³ is DMA-launch dominated. The 331 µs minimum represents near-pure compute; the 1,475 µs average shows the dispatch + DMA round-trip overhead dominates at this scale. Wide variance (331–3115 µs) even after 5 warmup iterations.

**Build command (per size — xclbin must match M/K/N exactly):**
```bash
cd ~/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
make NPU2=1 M=2048 K=2048 N=2048 build/final_2048x2048x2048_64x64x64_8c.xclbin
```

**Run command:**
```bash
./whole_array.exe \
  -x build/final_2048x2048x2048_64x64x64_8c.xclbin \
  -i build/insts_2048x2048x2048_64x64x64_8c.txt \
  -k MLIR_AIE -M 2048 -K 2048 -N 2048 \
  --b_col_maj 1 --warmup 5 --iters 10 -v 0
```

---

## Example 2 — `ml/relu`

**Operation:** ReLU activation, element-wise max(0, x)  
**Dtype:** bfloat16  
**Hardware config:** 4 AIE columns, 2 DMA channels, streaming

| Elements | Data (2-way) | Latency | Effective Bandwidth |
|---|---|---|---|
| 16,384 | 64 KB | 155 µs | 0.42 GB/s |
| 65,536 | 256 KB | 116 µs | 2.26 GB/s |
| 262,144 | 1 MB | 138 µs | 7.60 GB/s |
| 1,048,576 | 4 MB | 210 µs | **19.97 GB/s** |

**Bandwidth formula:** `2 * N * sizeof(bf16) / latency_us / 1000` (input + output)

**Notes:**
- Small sizes (≤64K elements) are entirely overhead-dominated; the ~115 µs launch floor swamps useful bandwidth.
- At 1M elements, bandwidth saturates near 20 GB/s — approaching DRAM bandwidth limits for this UMA system. ReLU is compute-trivial so this is effectively a DMA throughput ceiling measurement.
- xclbin must be rebuilt per size (length is baked into DMA descriptors).

**Build command (per length):**
```bash
cd ~/mlir-aie/programming_examples/ml/relu
make NPU2=1 length=1048576 build/final.xclbin
```

**Run command:**
```bash
./_build/relu -x build/final.xclbin -i build/insts.elf -k MLIR_AIE -l 1048576
```

---

## Example 3 — `ml/eltwise_add`

**Operation:** Element-wise addition, A + B → C  
**Dtype:** bfloat16  
**Size:** 65,536 elements (3 buffers × 128 KB = 384 KB total transfer)

| Metric | Value |
|---|---|
| Avg Latency | 108.6 µs |
| Min Latency | 72 µs |
| Max Latency | 118 µs |
| Derived bandwidth @ min | ~5.3 GB/s |

**Notes:**
- Nearly identical profile to relu at the same size class — both are single-pass memory-bandwidth-bound ops.
- 72 µs minimum is close to the NPU dispatch floor for this buffer size.
- The eltwise_add test does not report GOPS/bandwidth natively; the values above are derived.

**Build command:**
```bash
cd ~/mlir-aie/programming_examples/ml/eltwise_add
make NPU2=1 build/final.xclbin
# Host exe: cmake with -DCMAKE_CXX_COMPILER=g++-15 (see Gotchas)
```

**Run command:**
```bash
./_build/eltwise_add -x build/final.xclbin -i build/insts.bin -k MLIR_AIE --warmup 5 --iters 10
```

---

## Example 4 — `basic/memcpy` (DMA passthrough baseline)

**Operation:** No compute — pure shim DMA passthrough (read in, write out unchanged)  
**Config:** 1 AIE column, bypass=True

| Buffer Size | Latency | Bidirectional BW |
|---|---|---|
| 16 KB | 114 µs | ~0.3 GB/s |
| 4 MB | 1,404 µs | ~5.7 GB/s |

**Notes:**
- The ~114 µs floor at small sizes is the irreducible NPU dispatch + DMA setup cost, visible across all examples.
- At 4 MB, a single shim DMA channel saturates around 5–6 GB/s bidirectional.
- This is a useful baseline: any workload whose compute fits within the DMA time is not NPU-worth offloading.

---

## Cross-Example Summary

| Metric | Observed |
|---|---|
| NPU dispatch floor | ~72–115 µs (irreducible per kernel launch) |
| Single-channel DMA peak (1 col) | ~5.7 GB/s |
| Multi-channel DMA peak (4 cols, streaming) | ~20 GB/s |
| Peak GEMM (int8, 2048³, 32 cores) | **8,603 GOPS = 8.6 TOPS** |
| Peak GEMM efficiency vs 50 TOPS | ~17% |
| NPU vs CPU GEMM crossover | ~1024³–2048³ for int8 |

---

## Learnings and Gotchas

### 1. xclbin dimensions are statically compiled — M/K/N are NOT runtime parameters
The most important finding. MLIR-AIE compiles DMA buffer descriptors with **hardcoded element counts** into the xclbin. The `-M`, `-K`, `-N` flags on the test host executable only control **host-side buffer allocation**. If you pass `-M 512` with an xclbin compiled for 4096, the NPU DMA will read/write 64× beyond the allocated buffer into arbitrary kernel memory.

**This caused a kernel panic** when attempting to sweep matrix sizes by passing different `-M`/`-K`/`-N` values against a single 4096×4096×4096 xclbin.

**Rule:** Each (M, K, N) size requires its own separately compiled xclbin. The xclbin filename encodes this: `final_4096x4096x4096_64x64x64_8c.xclbin`.

### 2. `--b_col_maj 1` is required for the whole_array matmul
The xclbin for whole_array is compiled with B in column-major layout. Omitting `--b_col_maj 1` at runtime causes 100% verification failures (computed values are numerically wrong) but the run completes and reports timing — a silent correctness bug. Always match runtime flags to compile-time flags.

### 3. Device target: `npu2` not `npu`
Krackan (Ryzen AI 300/400 series) is XDNA2 and requires `NPU2=1` on the make command, which sets `devicename=npu2`. This affects:
- Peano compiler target (`aie2p-none-unknown-elf` vs `aie2-none-unknown-elf`)
- MLIR device annotation (`aie.device(npu2)`)
- DMA instruction encoding

Running an `npu` (Phoenix) xclbin on npu2 hardware, or vice versa, produces wrong results.

### 4. Host compiler: only g++-15 present; CMakeLists.txt defaults to g++-13
All example `CMakeLists.txt` files hardcode `set(CMAKE_CXX_COMPILER g++-13)` as default. g++-13 is absent on this machine; only g++-15 is available.

**Fix:** Build host executables manually with cmake overrides instead of `make <target>.exe`:
```bash
rm -rf _build && mkdir -p _build
cd _build && cmake .. \
  -DTARGET_NAME=<name> \
  -DCMAKE_CXX_COMPILER=g++-15 \
  -DCMAKE_C_COMPILER=gcc-15 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### 5. Kernel (.o) compilation is cached; only MLIR and xclbin need rebuilding per size
For the matrix_multiplication examples, the per-core kernel `mm_64x64x64.o` (Peano-compiled) does not change across different M/K/N global sizes — only the data movement MLIR changes. Make correctly reuses the cached `.o`, so rebuilding at a new size is much faster than the initial build.

### 6. NPU dispatch overhead is ~100 µs and is the dominant cost for small workloads
Every kernel invocation incurs an irreducible ~72–115 µs overhead regardless of payload size. This matches the midterm report's finding that chained matmuls lose to CPU because each kernel launch adds ~318 µs of fixed overhead. Any offload decision must account for this floor.

### 7. `insts` file format differs between examples
Some examples generate `insts.elf` (newer, `--aie-generate-elf`), others generate `insts.bin` or `insts.txt` (older, `--aie-generate-npu-insts`). The `-i` flag on the host exe must match the actual file generated. Check the Makefile to confirm which format is used.

### 8. DMA bandwidth scales with number of shim columns
- 1 active shim column (memcpy): ~5.7 GB/s peak
- 4 active shim columns (relu, streaming): ~20 GB/s peak
This ~4× scaling confirms that shim DMA bandwidth is roughly proportional to active column count, which aligns with the XDNA2 architecture description.
