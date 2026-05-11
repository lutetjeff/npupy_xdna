# Chained GEMM (cross-region fusion) — KILL-SWITCH report

**V2 Task 15** · **Path B outcome** · **Time spent: ~75 minutes of the 2-hour
hard budget**

## TL;DR

The intended fusion — `D = (A @ B) @ C` with the intermediate `T` held in
memtile (NOT round-tripped to host) — is **not implementable on the stock
mlir-aie IRON high-level API** as it stands today. The walls are concrete and
each one is independently sufficient to block a working 256³ chain.

The template file `npupy_xdna/templates/chained_gemm.py` ships in this commit
with `ChainedGemmTemplate.ENABLED = False`, so `match()` returns False
unconditionally and the dispatcher ignores it.

This document records what was tried, what blocked it, and what V3 would have
to do differently.

---

## What was attempted

A single-xclbin two-phase IRON program on the npu2 (Krackan, 4 rows × 8 cols)
following this dataflow graph:

```
                 host A          host B
                   │                │
       shim col 0  │                │
                   ▼                ▼
                 memtile col 0    memtile col 0
                   │ split          │ forward
                   ▼                ▼
         ┌──── L1 row 0..3 col 0 (Phase-1 cores) ─────┐
         │  T_tile = A_tile @ B_tile  (MAC accum)     │
         └────────────────┬───────────────────────────┘
                          │ join
                          ▼
                 memtile col 0 (T-resident FIFO)
                          │ forward (cross-col)
                          ▼
                 memtile col 1 (T staging)        host C
                          │ split                  │
                          ▼                        ▼
         ┌──── L1 row 0..3 col 1 (Phase-2 cores) ─────┐
         │  D_tile = T_tile @ C_tile  (MAC accum)     │
         └────────────────┬───────────────────────────┘
                          │ join
                          ▼
                 memtile col 1
                          │ drain
                          ▼
                       host D
```

Concrete shapes for the target 256³ chain:
- M = K1 = N1 = K2 = N2 = 256
- m = k = n = 64 (single MAC-aligned tile)
- n_aie_rows = 4, n_aie_cols-per-phase = 1, two phases ⇒ 8 cores total
- Intermediate T ∈ int16, shape (256, 256), size = 256·256·2 = **128 KB**
  (fits comfortably in 512 KB memtile budget)

---

## Why it doesn't work (root causes)

### Wall #1 — `ObjectFifo` is single-consume; GEMM2 re-reads A

The standard `whole_array_iron.py` GEMM streams its `A` input via:

```python
A_tiles = TensorTiler2D.group_tiler(
    (M, K),
    (m * n_A_tiles_per_shim, k),
    (1, K // k),
    pattern_repeat=N // n // n_aie_cols,   #  ← this
    prune_step=False,
)
```

`pattern_repeat = N // n // n_aie_cols` means that **for each output N-column
of D, the entire A matrix is re-streamed from host**. With N=256, n=64,
n_aie_cols=1 (single-column-per-phase design) this is **4 full re-reads of A**.

`pattern_repeat` is a *shim-DMA-descriptor* level repeat. It is implemented by
the shim DMA controller re-walking its host-side descriptor list. When `A` is
the host buffer this is fine — host memory is random-access.

But for the chained design, GEMM2's `A` is the GEMM1 output `T` residing in a
memtile. **An IRON `ObjectFifo` has single-consume semantics**: each release
permanently advances the consumer pointer; there is no "rewind" operation in
the high-level API. A memtile FIFO cannot satisfy `pattern_repeat × 4`.

The only escape hatches on the current API are:
- **(a)** Have GEMM1 produce T four times (defeats the purpose; identical to
  running two kernels back-to-back through host memory).
- **(b)** Rewrite GEMM2 to consume A exactly once. This requires changing the
  N-outer/K-inner tile traversal order in GEMM2's `core_fn` *and* changing the
  MAC accumulator usage so that one A-tile contributes to ALL N-output-tiles
  before being released. The latter is incompatible with the
  `matmul_i16_i16` external kernel's accumulator semantics (one accumulator
  per `(m, n)` tile, not per `(m, K)` row-band). It would require a new
  hand-vectorised C++ kernel — out of scope for a 2-hour task.

### Wall #2 — MAC accumulator layout vs row-major layout

GEMM1's C output sits in L1 in the **MAC accumulator layout** (interleaved
4×8 sub-tiles to feed the AIE2P vector MAC). The standard whole_array design
converts this back to row-major when crossing the L1→memtile boundary, via
`dims_to_stream=[(m//r, r*n), (r, t), (n//t, r*t), (t, 1)]` on the C_L2L3
ObjectFifo.

GEMM2's A input must arrive in L1 in a *different* MAC-input layout, applied
by `dims_to_stream=[(m//r, r*k), (k//s, s), (r, k), (s, 1)]` on the
A_L3L2-to-A_L2L1 path.

For the chained design these two transforms must be **composed back-to-back
through a single memtile-resident FIFO**. IRON's `ObjectFifo.forward()` does
support `dims_to_stream` AND `dims_from_stream`, so this is plausible in
principle — but:

1. Whether `dims_from_stream` on a memtile that was itself fed by a `join`
   with `dims_to_stream` correctly resolves to a single DMA descriptor with
   composed access pattern is **not documented** and was not verified on
   npu2.
2. Even if it composes correctly at IR level, the resulting access pattern is
   likely 5-D or 6-D and AIE2P shim/memtile DMAs cap out at 4-D nested
   strides. Anything beyond that fails at MLIR lowering.

### Wall #3 — Cross-column memtile-to-memtile routing

GEMM2's A_L2L1 split must be on the same column as the Phase-2 compute tiles.
If Phase-1 is col 0 and Phase-2 is col 1, T must move memtile-col-0 →
memtile-col-1. The npu2 stream switch supports this but the high-level IRON
API requires it to be expressed as a single ObjectFifo whose producer is on
one memtile and consumer on a different memtile — which the API does NOT
naturally express. (`.forward(placement=Tile(1, 1))` *should* place the
forward op on col 1's memtile, pulling data from col 0's memtile, but the
ObjectFifo allocator currently fuses both ends to the same memtile in the
naive case.)

A single-column variant (Phase-1 and Phase-2 both on col 0) sidesteps this,
but then the same 4 cores can't be in two `Worker` definitions simultaneously
— `rt.start(*workers)` would over-subscribe them and IRON rejects the
configuration.

### Wall #4 — Even if it compiled, the chained variant would be SLOWER

Per the V1 results, the existing whole_array GEMM at 256³ hits ~30 GOPS using
**all 32 cores**. The chained design above uses only **8 cores total** (4 per
phase, single-column each). That's a 4× hardware-utilisation reduction. The
intermediate-on-chip benefit (saving one 128 KB round-trip ≈ 8 µs at 16 GB/s
shim DMA) is dwarfed by losing 24 of 32 cores for compute.

The "use cols 0–3 for Phase-1 and cols 4–7 for Phase-2" alternative restores
full hardware utilisation but each phase only has **4 cores × 4 rows = 16**
not 32, and the cross-column memtile routing of Wall #3 becomes more severe
(T must distribute from 4 memtiles to 4 different memtiles, a 4-to-4 shuffle
that IRON does not express).

---

## What it would take to make this work (V3 path)

In rough increasing order of effort:

1. **Drop to placed-IR / direct MLIR-AIE**. Use `aie.device(NPU2) → aie.tile`
   and `aie.objectfifo` primitives directly, programming the memtile DMA
   descriptors as raw `aie.dma_bd` blocks. This bypasses the
   `pattern_repeat=N/n/cols` shim-side replay by emitting an explicit memtile
   DMA descriptor that re-walks T four times from a stationary memtile
   allocation. Reference example for the placed style is
   `mlir-aie/programming_examples/basic/matrix_scalar_mul/matrix_scalar_mul_placed.py`
   and for raw memtile DMA control:
   `mlir-aie/programming_examples/basic/dma_transpose/`.
   Estimated effort: 1–2 weeks for someone fluent in MLIR-AIE.

2. **Custom fused C++ kernel for the (m, k, n) inner tile**. Have a single
   core compute `(A_tile @ B_tile) @ C_tile` in L1, with the intermediate
   `(m, n)` tile held in vector accumulator registers. This avoids the
   memtile-routing problem entirely but caps the chain shape at the per-core
   L1 budget (~64 KB), so the maximum chained dims are roughly
   `m·n + m·k + k·n + n·n_C ≤ 32K i16` words. For m=k=n=64 this is
   ~32 KB of L1 — feasible — but the test target is 256³, which requires
   tiling, which brings memtile routing back into the picture.
   Estimated effort: 3–5 days, plus a new entry in
   `npupy_xdna/kernels/gemm_i16_all.cc`.

3. **Wait for IRON `ObjectFifo.replay()` / multi-consume support upstream**.
   The fundamental limitation is in the dataflow type system. The Vitis AIE
   roadmap mentions "persistent" memtile buffers; if/when they land they
   would unblock this template without any low-level work on our side.

---

## Decision

Given the 2-hour hard budget on V2 T15, **Path B (kill-switch) is the
correct outcome**. The walls above are individually documented enough that
V3 planning can pick up the most promising of (1)–(3) without re-doing this
investigation.

This task carries a real positive value for the V2 report even in the
kill-switch state: it converts a hand-wavey "cross-region fusion should help"
hypothesis into a concrete documented blocker, and it lets §5 of the report
honestly report the 14.7× 3mm speedup as a *ceiling* under the high-level
IRON API — not as a soft "we didn't try" floor.

---

## Artifacts

- `npupy_xdna/templates/chained_gemm.py` — template with `ENABLED=False`
- `npupy_xdna/tests/test_template_chained_gemm.py` — skipped pytest suite
- `.sisyphus/evidence/task-v2-15-killswitch.md` — evidence summary

## References

- `mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/whole_array_iron.py`
- `mlir-aie/python/iron/dataflow/objectfifo.py` — `.join()`, `.split()`, `.forward()`
- `npupy_xdna/templates/gemm_fusion.py` — single GEMM template this was built on
- V1 3mm benchmark result (FINAL_REPORT_DATA.md): 14.7× speedup via 3 kernel launches
