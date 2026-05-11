# NPUPy XDNA Reference
> **Audience:** Agent that knows Python but has never seen IRON/MLIR-AIE.
> Read this before implementing any template. Hardware: AMD Ryzen AI 7 350 (Krackan), XDNA2, 32 AIE cores (8 cols × 4 rows).

---

## Template → mlir-aie Example Crosswalk

Four planned templates, each mapped to the closest upstream example to extract from.

| Template | Closest mlir-aie Example | File Path |
|---|---|---|
| **GEMM Fusion** | `whole_array` (whole-array GEMM) | `programming_examples/basic/matrix_multiplication/whole_array/whole_array_iron.py` |
| **Column-Independent** | `ml/relu` (streaming elementwise) | `programming_examples/ml/relu/relu.py` |
| **Compute Pool** | `ml/relu` extended to 32 independent cores | `programming_examples/ml/relu/relu.py` (8-col variant) |
| **CGRA** | `basic/chaining_channels` (tile-to-tile pipeline) | `programming_examples/basic/chaining_channels/chaining_channels_placed.py` |

### GEMM Fusion → `whole_array_iron.py`

**What it demonstrates:** Distributing a large GEMM across all 32 cores (8 cols × 4 rows). A tiles are broadcast down each column (all 4 rows get the same A block), B tiles are distributed across columns (each column works on a different N slice). C tiles are collected back via join. B must be in column-major layout at runtime (`b_col_maj=1`).

**Key code pattern to extract:**
- `ObjectFifo.cons().split(offsets, obj_types, dims_to_stream, placement)` — distributes one shim FIFO to 4 per-row FIFOs with optional layout transform
- `ObjectFifo.prod().join(offsets, obj_types, names, placement)` — collects 4 per-row C FIFOs into one shim FIFO
- `TensorTiler2D.step_tiler((N,K), (n,k), tile_group_repeats=..., tile_group_steps=(n_aie_cols,1))` — col-major B tiling
- `Kernel(func_name, "mm_{m}x{k}x{n}.o", [A_l1_ty, B_l1_ty, C_l1_ty])` — pre-compiled vectorized kernel
- `Worker(..., placement=Tile(tile_col, tile_row), stack_size=0xD00)` — explicit tile placement required for 32-core design
- Makefile default: `n_aie_cols=8`, `b_col_maj=1` — these are compile-time flags that must match runtime flags exactly

**npu2 specifics:** `dev_ty = NPU2()` (not `NPU1()`). When `n_aie_cols=8 > n_aie_rows=4`, only 4 shim/mem tiles handle A distribution (alternating columns: `Tile(2*i, 1)`).

### Column-Independent → `relu.py`

**What it demonstrates:** Streaming elementwise op across multiple columns. Each column independently processes its own data slice — no broadcast, no join. Two DMA channels per column (`num_channels=2`) for double-buffering.

**Key code pattern to extract:**
- `ObjectFifo(line_type, name=f"in{i}_{j}")` — one FIFO per (column, channel); no linking between columns
- `TensorAccessPattern((1,size), offset=chunk*i*num_channels + chunk*j, sizes=[1,1,1,chunk], strides=[0,0,0,1])` — each column's TAP covers its own non-overlapping slice
- `rt.start(*my_workers)` with a single `task_group` containing all fills and drains — all columns run in parallel within one task group
- Extend to 8 columns for npu2: `num_columns = 8 if isinstance(dev, NPU2) else 4`

### Compute Pool → `relu.py` (32 independent cores, no grouping)

**What it demonstrates:** Extending relu to 32 fully independent cores (8 cols × 4 rows). Critically: **NO column grouping, NO broadcast, NO join**. Each of the 32 cores handles its own independent data chunk.

**Key distinction from relu.py:** relu uses 4 cols × 2 channels = 8 workers. Compute Pool uses 8 cols × 4 rows = 32 workers. Each worker gets its own input/output ObjectFifo with its own TensorAccessPattern slice.

**Key code pattern to extract:**
- Outer loop over `range(n_cols)`, inner loop over `range(n_rows)` — 2D worker indexing
- `chunk = total_elements // (n_cols * n_rows)` — partition data evenly
- Each worker's TAP: offset = `chunk * (col * n_rows + row)`, size = `[1,1,1,chunk]`
- `Worker(core_fn, [of_ins[i].cons(), of_outs[i].prod(), kernel], placement=Tile(col, row+2))` — note row offset: shim=0, mem=1, compute starts at row 2

### CGRA → `chaining_channels_placed.py`

**What it demonstrates:** Tile-to-tile pipeline via MemTile buffering. Data flows Shim → MemTile → ComputeTile as a chain. Uses the **lower-level placed MLIR dialect API** (`aie.dialects.aie`), not the IRON high-level API.

**Key code pattern to extract:**
- `tile(col, row)` — explicit tile references (ShimTile=row 0, MemTile=row 1, ComputeTile=row 2+)
- `buffer(tile, datatype, initial_value=...)` — pre-loaded MemTile static buffers
- `lock(tile, lock_id, init, sym_name)` — explicit synchronization via lock acquire/release
- For CGRA templates needing multi-stage pipelines: chain `ObjectFifo` endpoints through MemTile intermediates

**Warning:** `chaining_channels_placed.py` uses low-level dialects, not IRON. For IRON-based CGRA, use `ObjectFifo` chaining: `fifo1.cons().forward(..., placement=MemTile)` to route through MemTile.

---

## IRON API Cheat-Sheet

### Imports

```python
import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program, ExternalFunction, Kernel
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern, TensorAccessSequence
```

### ObjectFifo

Represents a data movement channel. Automatically maps to shim DMA + mem tile buffering based on placement.

```python
# Create a FIFO for moving (m, k) int16 tiles
fifo = ObjectFifo(np.ndarray[(m, k), np.dtype[np.int16]], name="A_L3L2", depth=2)

# Consumer-side operations
fifo.cons()                                    # consumer endpoint (pass to Worker)
fifo.cons().forward(dims_to_stream=..., name="A_L2L1")  # re-tile as it passes through MemTile
fifo.cons().split(offsets, obj_types, dims_to_stream, placement)  # 1→N split (broadcast/distribute)

# Producer-side operations
fifo.prod()                                    # producer endpoint (pass to rt.fill)
fifo.prod().join(offsets, obj_types, names, depths, placement)  # N→1 join (collect)
```

`depth=2` enables double-buffering (producer and consumer can overlap).

### Worker

Binds a Python core function to an AIE compute tile. The function runs indefinitely on the tile.

```python
def core_fn(of_a, of_b, of_c, matmul_kernel):
    for _ in range_(n_tiles):           # range_ = IRON loop (NOT Python range)
        elem_out = of_c.acquire(1)      # acquire one buffer slot
        for _ in range_(K // k):
            elem_in_a = of_a.acquire(1)
            matmul_kernel(elem_in_a, elem_out)
            of_a.release(1)
        of_c.release(1)

worker = Worker(
    core_fn,
    [fifo_A.cons(), fifo_C.prod(), matmul_kernel],  # args passed positionally to core_fn
    placement=Tile(col, row),           # optional: explicit tile placement
    stack_size=0xD00,                   # optional: needed for large kernels
)
```

### Runtime

Describes host-side DMA sequencing.

```python
rt = Runtime()
with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):  # declare host tensor handles
    rt.start(*workers)                  # launch all workers

    tg = rt.task_group()               # group DMA ops for batched sync
    rt.fill(fifo.prod(), A, tap, task_group=tg)      # DMA: host→device
    rt.drain(fifo.cons(), C, tap, wait=True, task_group=tg)  # DMA: device→host
    rt.finish_task_group(tg)           # sync point: wait for all in tg to complete
```

`wait=True` on drain means this task group blocks until the drain is done.
Multiple fill+drain within one task_group run in parallel on separate DMA channels.

### TensorTiler2D

Generates a sequence of TensorAccessPatterns for tiled DMA.

```python
# Tile (M,K) matrix into (m,k) tiles, grouped by row
a_taps = TensorTiler2D.group_tiler(
    (M, K),          # full tensor shape
    (m, k),          # tile shape
    (1, K // k),     # group: 1 tile row at a time, all K tiles
    pattern_repeat=N // n,  # repeat for each N tile
)
# a_taps[tile_row] gives the TAP for that row's DMA

# Step tiler for column-major B
b_tiles = TensorTiler2D.step_tiler(
    (N, K),                            # B transposed shape
    (n, k),                            # tile shape
    tile_group_repeats=(N//n//n_cols, K//k),
    tile_group_steps=(n_cols, 1),      # stride by n_cols in column dimension
)
```

### TensorAccessPattern

Low-level 4D DMA descriptor. Used when TensorTiler2D doesn't give the right shape.

```python
tap = TensorAccessPattern(
    tensor_dims=(1, size),             # full tensor shape (1D viewed as 2D)
    offset=chunk * i,                  # byte offset into tensor
    sizes=[1, 1, 1, chunk],            # 4 dimension sizes (outermost first)
    strides=[0, 0, 0, 1],             # strides in elements
)
```

Constraint: `sizes` and `strides` are always length 4 (XDNA2 DMA has 4D descriptors).

### Program + SequentialPlacer

```python
my_program = Program(iron.get_current_device(), rt)
return my_program.resolve_program(SequentialPlacer())
# SequentialPlacer: assigns tiles/channels sequentially without manual placement
```

### @iron.jit

```python
@iron.jit(is_placed=False)    # is_placed=False → let SequentialPlacer decide
def my_kernel(input0, input1, output):
    # ObjectFifo / Worker / Runtime definitions here
    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())
```

Caches compiled xclbin at `~/.iron/cache/`. Recompiles only when the IRON program changes.

### ExternalFunction vs Kernel

```python
# ExternalFunction: compile C++ source with Peano at JIT time
ef = ExternalFunction(
    "matrix_multiplication",
    source_file="/path/to/mm.cc",
    arg_types=[a_ty, b_ty, c_ty],
    include_dirs=[cxx_header_path()],
)

# Kernel: link pre-compiled .o (already Peano-compiled by Makefile)
k = Kernel("matmul_i8_i32", "mm_64x64x64.o", [a_ty, b_ty, c_ty])
```

Use `ExternalFunction` during development (auto-compiled), `Kernel` in Makefile-driven flows (faster rebuilds).

### range_ vs range

```python
# range_ = IRON control flow — generates AIE tile code
for _ in range_(K // k):
    elem = fifo.acquire(1)
    kernel(elem)
    fifo.release(1)

# range (Python) = host-side loop — runs on CPU during IRON program construction
for tile_row in range(M // m):
    rt.fill(...)   # this generates MLIR DMA ops, not tile code
```

**Rule:** Inside `core_fn` (the function passed to `Worker`), use `range_`. Outside (in the `rt.sequence` block), use `range`.

---

## Gotchas

Each gotcha below caused a real failure on this machine. Listed in order of pain severity.

### 1. `b_col_maj=1` required for whole_array matmul

**Symptom:** Matmul completes with no error, timing looks normal, but 100% of output elements fail numerical verification. Silent correctness bug — no crash, no warning.

**Root cause:** `whole_array_iron.py` is compiled with `b_col_maj=1` by default (the Makefile sets `b_col_maj?=1`). The xclbin DMA descriptors expect B in column-major layout. If you pass B in row-major (NumPy default) at runtime, the NPU reads the wrong strides and produces garbage output.

**Fix:** Always pass `--b_col_maj 1` to the host executable when running `whole_array`. When building programmatically, ensure your host-side B array is in column-major layout or pass a transposed B. In the GEMM Fusion template, pass `tile_group_col_major=True` to `TensorTiler2D` and set `b_col_maj=True` in `dims_to_stream`.

**Verification:** Run with `whole_array.exe -v 1` (verbose verify). Output should report `PASS`.

---

### 2. `copy=True` on `output.numpy()` (silent buffer corruption)

**Symptom:** NPU output arrays look correct immediately after the kernel call but contain stale or corrupted data by the time you compare against the NumPy reference. Intermittent failures under repeated calls.

**Root cause:** `output.numpy()` returns a view into the XRT buffer's memory without copying. If the XRT runtime reuses or unmaps that buffer before you finish reading, the view is dangling.

**Fix:**
```python
# Wrong
result = output.numpy()
np.testing.assert_array_equal(result, reference)

# Correct
result = np.array(output.numpy(), copy=True)
np.testing.assert_array_equal(result, reference)
```

---

### 3. xclbin dimensions are baked at compile time — not runtime parameters

**Symptom:** Passing `-M 512` to a host executable that was compiled with an xclbin for `M=4096` causes a **kernel panic** (hard crash of the host, possible kernel oops). The NPU DMA reads/writes 64× beyond the allocated buffer into arbitrary kernel memory.

**Root cause:** MLIR-AIE compiles DMA buffer descriptors with **hardcoded element counts** into the xclbin. The `-M`, `-K`, `-N` flags on the host executable only control host-side buffer allocation. There is no runtime parameterization.

**Fix:** Each (M, K, N) combination requires a separately compiled xclbin. The xclbin filename encodes this: `final_2048x2048x2048_64x64x64_8c.xclbin`. Never reuse an xclbin for a different problem size. In the template system, the `@iron.jit` cache handles this automatically by keying the cache on all dimension parameters.

---

### 4. `g++-13` absent; `g++-15` required

**Symptom:** Running `make <target>.exe` fails with: `g++-13: command not found` or CMake error during host executable compilation.

**Root cause:** All example `CMakeLists.txt` files hardcode `set(CMAKE_CXX_COMPILER g++-13)` or default to `g++-13`. The system only has `g++-15`.

**Fix:** Build host executables manually instead of via `make`:
```bash
rm -rf _build && mkdir -p _build
cd _build && cmake .. \
  -DTARGET_NAME=<name> \
  -DCMAKE_CXX_COMPILER=g++-15 \
  -DCMAKE_C_COMPILER=gcc-15 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

The Python IRON flow (`@iron.jit` + `ExternalFunction`) uses Peano (`llvm-aie`) for AIE kernel compilation and does NOT invoke g++ — this issue only affects host executables (test runners).

---

### 5. `NPU2=1` required for Krackan — omitting it silently targets wrong device

**Symptom:** Build succeeds but produces incorrect results or crashes. MLIR output contains `aie.device(npu)` instead of `aie.device(npu2)`.

**Root cause:** Makefile defaults to `npu` (Phoenix/Hawk, XDNA1). Krackan is `npu2` (XDNA2, `aie2p` Peano target). The two use different:
- Peano compiler target: `aie2p-none-unknown-elf` vs `aie2-none-unknown-elf`
- MLIR device annotation: `aie.device(npu2)` vs `aie.device(npu)`
- DMA instruction encoding and firmware ABI

**Fix:** Always build with `NPU2=1`:
```bash
make NPU2=1 M=2048 K=2048 N=2048 build/final_2048x2048x2048_64x64x64_8c.xclbin
```

In Python IRON: use `NPU2()` from `aie.iron.device`, not `NPU1()`.

---

### 6. Verify MLIR header contains `aie.device(npu2)` not `aie.device(npu)`

**Symptom:** Runtime errors, firmware assertion failures, or wrong tile counts when an xclbin is loaded on npu2 hardware.

**Root cause:** If the device annotation in the compiled MLIR is wrong, the firmware applies incorrect DMA register maps and tile row/column offsets.

**Fix:** After generating MLIR (via `python whole_array_iron.py --dev npu2 ...`), inspect the first line:
```
module @name { aie.device(npu2) { ...
```
If it says `npu` instead of `npu2`, the `--dev` argument was wrong or `NPU2=1` was omitted.

---

### 7. `insts` file format differs between examples

**Symptom:** `FileNotFoundError: build/insts.elf` or `build/insts.bin` — one exists, the other does not. Or: host exe fails silently reading wrong format.

**Root cause:** Newer examples use `--aie-generate-elf` → `insts.elf`; older examples use `--aie-generate-npu-insts` → `insts.bin` or `insts.txt`.

**Fix:** Check the example's Makefile for which flag is used:
```bash
grep aie-generate Makefile
```
Then pass the correct file to `-i` on the host executable. The `whole_array` example generates `insts_*.txt`; `relu` and `eltwise_add` generate `insts.elf`.

---

## Verified Hardware Configurations

**Machine:** ASUS Vivobook S16 M5606KA  
**CPU:** AMD Ryzen AI 7 350 (Krackan Point), Radeon 860M, 16 logical cores  
**NPU:** XDNA2 (AIE2P), BDF `0000:64:00.1`, 8 cols × 4 rows = 32 AIE cores  
**RAM:** 14,580 MB (UMA — shared CPU+NPU memory)  
**OS:** Ubuntu Resolute Raccoon, Linux 6.19.0-5-generic

### Environment Versions

| Component | Version |
|---|---|
| XRT | 2.23.0 (hash 99b8b976) |
| amdxdna driver | 2.23.0_20260218 |
| NPU Firmware | 255.0.11.69 |
| mlir-aie | `~/mlir-aie` (IRON interface) |
| Peano (llvm-aie) | in `~/mlir-aie/ironenv/lib/python3.13/site-packages/llvm-aie` |
| Host compiler | g++-15 (g++-13 absent) |
| Device target | `npu2` — required for Krackan |
| Python venv | `~/mlir-aie/ironenv` (Python 3.13) |

**Environment setup (required before every build/run):**
```bash
source /opt/xilinx/xrt/setup.sh
source ~/mlir-aie/ironenv/bin/activate
source ~/mlir-aie/utils/env_setup.sh
```

### GEMM Performance (int8, `whole_array`, 8-col, b_col_maj=1)

Methodology: 5 warmup iterations discarded, 10 measured iterations, correctness PASS.

| Size (M=K=N) | Avg Latency | Avg GOPS | Min Latency | Peak GOPS | % of 50 TOPS |
|---|---|---|---|---|---|
| 1024 | 1,475 µs | 1,456 | 331 µs | 6,488 | 13.0% |
| **2048** | **2,171 µs** | **7,914** | **1,997 µs** | **8,603** | **17.2%** |
| 4096 | 50,480 µs | 2,723 | 22,490 µs | 6,111 | 5.4% |

**Peak verified: 8,603 GOPS = 8.6 TOPS** (int8, 2048³, 32 cores)

**Sweet spot:** 2048³ — DMA amortizes well (~24 MB A+B+C), tight variance (1997–2303 µs).  
**Crossover vs CPU:** ~1024³–2048³ for int8. Below 1024³, NPU dispatch overhead dominates.  
**4096³ degrades:** A+B+C ≈ 96 MB at i8/i32 saturates UMA memory bandwidth.

### DMA / Bandwidth Baselines

| Config | Operation | Peak Bandwidth | Notes |
|---|---|---|---|
| 1 col, bypass | memcpy (DMA passthrough) | ~5.7 GB/s | Single shim channel ceiling |
| 4 cols, streaming | relu (bfloat16) | ~20 GB/s | 1M elements, 4 DMA channels |
| 8 cols, streaming | eltwise_add (bfloat16) | est. 20 GB/s | Scales ~linearly with active columns |
| any | small buffer (≤64K elements) | <0.5 GB/s | Overhead-dominated |

**DMA bandwidth scales linearly with active shim columns (~5 GB/s per column).**

### Dispatch Floor

Every kernel invocation has an **irreducible ~72–115 µs overhead** regardless of payload:
- `72 µs` — minimum observed (small relu, already warm)
- `100 µs` — practical planning value
- `115 µs` — p99 floor for cold launches

**Implication for offload decisions:** Any workload whose compute fits inside 100 µs is not NPU-worth offloading. For GEMM, the crossover is roughly 512³–768³ for int8.

### GEMM Formula Reference
```
GOPS = 2 * M * K * N / latency_µs / 1000
DMA_BW = 2 * N * sizeof(dtype) / latency_µs / 1000   # for elementwise (input + output)
```

---

## NPBench Framework Contract

> **Source:** Task 7 (ran in parallel). Documented from live exploration of `/home/lutet/ece511/npbench/`.

---

### 1. JSON Schema: `framework_info/<name>.json`

Every framework requires exactly one JSON file at `npbench/framework_info/<name>.json`.
The file is loaded by `Framework.__init__` and stored as `self.info`.

**Minimum required schema (numpy uses only these fields):**

```json
{
    "framework": {
        "simple_name": "numpy",
        "full_name": "NumPy",
        "prefix": "np",
        "postfix": "numpy",
        "class": "Framework",
        "arch": "cpu"
    }
}
```

**All fields and their roles:**

| Field | Required | Used by | Notes |
|-------|----------|---------|-------|
| `simple_name` | yes | `generate_framework()`, CLI | Matches filename stem; passed as `fname` to `Framework.__init__` |
| `full_name` | yes | `Test.run()` output lines | Display name in benchmark report |
| `prefix` | yes | `Framework.args()`, `mutable_args()`, `inout_args()` | Used to name vars: `__npb_{prefix}_{arg}` |
| `postfix` | recommended | `Framework.implementations()`, `impl_files()` | Suffix for finding `{module}_{postfix}.py` |
| `class` | yes | `generate_framework()` | Class imported from `npbench.infrastructure`; custom classes must be registered there |
| `arch` | recommended | metadata only | `"cpu"` or `"gpu"` |

---

### 2. Python Class Interface

**Base class:** `npbench.infrastructure.Framework`  
**Location:** `npbench/npbench/infrastructure/framework.py`

```python
class NPUPyFramework(Framework):
    def __init__(self, fname: str):
        super().__init__(fname)   # loads framework_info/<fname>.json → self.info
```

**Methods to override for a new framework:**

| Method | Signature | Override when |
|--------|-----------|---------------|
| `version()` | `() -> str` | Package name differs from `fname` |
| `imports()` | `() -> Dict[str, Any]` | Framework needs to inject modules into benchmark exec context |
| `copy_func()` | `() -> Callable` | Framework uses a different array type |
| `copy_back_func()` | `() -> Callable` | Results must be transferred to host |
| `impl_files()` | `(bench) -> Sequence[Tuple[str, str]]` | Multiple implementations per benchmark |
| `implementations()` | `(bench) -> Sequence[Tuple[Callable, str]]` | Custom import logic |

---

### 3. Benchmark Implementation File Convention

```
npbench/npbench/benchmarks/{relative_path}/{module_name}_{postfix}.py
```

The file must expose a function named exactly `bench_info["func_name"]` (usually `kernel`):

```python
# gemm_npupy.py
import npupy_xdna as npupy

def kernel(alpha, beta, C, A, B):
    return npupy.dispatch_gemm(alpha, beta, C, A, B)
```

---

### 4. Bench Info JSON Schema: `bench_info/<name>.json`

```json
{
    "benchmark": {
        "name": "General matrix-matrix multiplication",
        "short_name": "gemm",
        "relative_path": "polybench/gemm",
        "module_name": "gemm",
        "func_name": "kernel",
        "parameters": {
            "S": { "NI": 1000, "NJ": 1100, "NK": 1200 },
            "M": { "NI": 2500, "NJ": 2750, "NK": 3000 },
            "L": { "NI": 7000, "NJ": 7500, "NK": 8000 }
        },
        "init": {
            "func_name": "initialize",
            "input_args": ["NI", "NJ", "NK"],
            "output_args": ["alpha", "beta", "C", "A", "B"]
        },
        "input_args": ["alpha", "beta", "C", "A", "B"],
        "array_args": ["C", "A", "B"],
        "output_args": ["C"]
    }
}
```

---

### 5. CLI Invocation

```bash
cd /home/lutet/ece511/npbench
python run_benchmark.py -b gemm -f numpy -p S -r 3
python run_benchmark.py -b gemm -f npupy -p S -r 5 -v True
```

**Note:** `python -m npbench` does NOT work — no `__main__.py`.

---

### 6. How `generate_framework()` Resolves the Class

```python
info = json.load(frmwrk_path)["framework"]
module = importlib.import_module("npbench.infrastructure")
cls = getattr(module, info["class"])
```

The class name in `framework_info/npupy.json` must be exported from `npbench/npbench/infrastructure/__init__.py`.

---

### 7. NPBench Installation Gotchas

1. **`pkg_resources` missing on Python 3.13+**: Fix: `pip install "setuptools<67"` in ironenv (setuptools-66.1.1 installed).
2. **`pygount` missing**: Fix: `pip install pygount` (pulls in gitpython, chardet).
3. **Namespace collision**: Running `python -c "import npbench"` from `/home/lutet/ece511/` hits the directory as a namespace package. Run from `/tmp` or any other directory.
4. **PyXRT warning**: `import npupy_xdna` emits `Failed to import PyXRT: No module named 'pyxrt'` — expected in dev environments. Return code is still 0.

---

### 8. Template for `framework_info/npupy.json` (Task 24)

```json
{
    "framework": {
        "simple_name": "npupy",
        "full_name": "NPUPy (XDNA)",
        "prefix": "npupy",
        "postfix": "npupy",
        "class": "NPUPyFramework",
        "arch": "cpu"
    }
}
```

Place at: `/home/lutet/ece511/npbench/framework_info/npupy.json`  
Implement: `/home/lutet/ece511/npbench/npbench/infrastructure/npupy_framework.py`  
Register: add `from .npupy_framework import *` to `npbench/npbench/infrastructure/__init__.py`
