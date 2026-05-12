# bench/
Benchmarking utilities and synthetic benchmarks.
- `timer.py` — BenchmarkConfig, Timer context manager, run_benchmark()
- `baselines.py` — CPU baselines: int16 (no BLAS) + int16↔f32 scipy BLAS round-trip
- `paths.py` — canonical result directory paths
- `seed.py` — deterministic seeding (seed=42 project-wide)
- `synthetic/` — synthetic benchmark implementations (Horner polynomial, tanh, hash)
