# NPUPy Examples

Working examples demonstrating NPU offloading on AMD Ryzen AI (XDNA2).

## Prerequisites

```bash
source /opt/xilinx/xrt/setup.sh
source ~/mlir-aie/ironenv/bin/activate
source ~/mlir-aie/utils/env_setup.sh
```

## Running

Examples handle path setup automatically — run from any directory:

```bash
python /home/lutet/ece511/npupy_xdna/examples/01_gemm.py
```

Or run all examples at once from the workspace root:

```bash
cd /home/lutet/ece511
for f in npupy_xdna/examples/0*.py; do
    echo "=== $f ==="
    timeout 60 python "$f" 2>&1 | tail -8
    echo
done
```

## Examples

| # | File | What it demonstrates | Expected speedup |
|---|------|---------------------|-----------------|
| 1 | `01_gemm.py` | Basic int16 GEMM offload (512×512) | ~9× vs CPU |
| 2 | `02_gemm_relu.py` | GEMM with fused ReLU epilogue (512×512) | ~9× vs CPU |
| 3 | `03_tanh.py` | High-intensity elementwise tanh (1M elements) | ~32× vs CPU |
| 4 | `04_relu.py` | Simple elementwise ReLU (1M elements) | ~2× vs CPU |
| 5 | `05_hash.py` | Compute-bound FNV-1a hash (1M elements) | ~49× vs CPU |
| 6 | `06_transparent_dispatch.py` | Full transparent `np.matmul` interception | automatic |

## Notes

- First run of each example may take 1–2 minutes (xclbin compilation; cached afterward)
- All examples use int16 data (project scope)
- NPU execution requires XRT environment (see Prerequisites)
- Without XRT, examples fall back to CPU gracefully with an informational message
- Examples 1–5 print CPU vs NPU timing and a PASS/FAIL correctness check
- GEMM examples use 512×512 by default for fast demo; 2048×2048 shows larger NPU speedup
