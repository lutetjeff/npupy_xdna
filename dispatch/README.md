# dispatch/
PoC 2 — Transparent NumPy backend via __array_function__ protocol.
- `array_shim.py` — NPUPyArray subclass intercepting numpy operations
- `extract.py` — maps numpy function calls to Region objects
- `dispatcher.py` — orchestrates: extract → classify → offload → lower → run
- `dtype_convert.py` — f64→int16 conversion with overflow detection
- `correctness_gate.py` — bit-exact verification for int16 results
