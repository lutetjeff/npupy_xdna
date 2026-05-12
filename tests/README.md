# tests/
pytest test suite. Run: `source ~/mlir-aie/ironenv/bin/activate && pytest npupy_xdna/tests/ -v`
NPU-execution tests require XRT: `source /opt/xilinx/xrt/setup.sh`
- `test_region.py`, `test_cpu_runner.py`, `test_bench.py` — core infrastructure
- `test_template_*.py` — per-template correctness (GEMM, Col-Indep, Compute Pool, CGRA, Sliding Window)
- `test_classifier.py`, `test_cost_model.py`, `test_offload.py` — heuristic modules
- `test_array_shim.py`, `test_extract.py`, `test_dispatcher.py` — dispatch pipeline
- `test_dispatch_tanh_hash.py` — end-to-end tanh/hash dispatch verification
