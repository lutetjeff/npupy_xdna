# runtime/
NPU execution infrastructure — runners, safety checks, concurrency control.
- `runner.py` — RunResult dataclass, Runner protocol
- `cpu_runner.py` — CPU baseline via numpy
- `npu_runner.py` — NPU execution via IRON JIT with timeout enforcement
- `iron_jit.py` — xclbin compilation + atomic cache
- `preflight.py` — 5-point safety check before first NPU run (prevents kernel panics)
- `npu_lock.py` — fcntl.flock mutex for NPU exclusivity across parallel agents
