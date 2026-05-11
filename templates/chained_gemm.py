"""
ChainedGemmTemplate: cross-region fusion attempt for D = (A @ B) @ C.

STATUS: KILL-SWITCHED (Path B). See `npupy_xdna/results/chained_gemm_kill_switch.md`
for the full design exploration and the root-cause analysis that justifies disabling
this template until lower-level (MLIR-AIE memref + manual DMA) infrastructure is
available.

`match()` returns False unconditionally. `config_space()` and `estimated_cost()`
remain implemented so that the dispatcher and cost-model tooling can still inspect
the template metadata (used by `results/` reporting and the V3 plan).

Design intent (not realised):
    Phase 1 cores compute T = A @ B; T is joined into a memtile-resident object
    FIFO instead of being drained to host. Phase 2 cores read T from the same
    memtile (forwarded across columns by the stream switch) as their A input and
    compute D = T @ C. Only A, B, C are filled from host; only D is drained.

Why it fails on stock IRON (summary; full version in the kill-switch doc):
  1. IRON `ObjectFifo` is single-consume FIFO. GEMM2 re-reads its A input
     `N // n // n_aie_cols` times across the natural iteration order (the
     `pattern_repeat` flag on `TensorTiler2D`). Re-reading from memtile is
     unsupported in the high-level API.
  2. The MAC accumulator layout of GEMM1's C output (m,n in tile-major) is not
     the same layout as GEMM2's A input (m,k in MAC-vectorised stream), forcing
     a memtile-side `dims_from_stream` re-layout pass. Plumbing both a `join`
     `dims_to_stream` AND a downstream `split` `dims_from_stream` chained
     through `forward` works in principle but has not been verified on npu2.
  3. Workers in IRON run perpetually (`rt.start(*workers)`); chaining two
     full whole-array GEMMs in one xclbin doubles the core count needed (32
     for GEMM1 + 32 for GEMM2 = 64) but the NPU2 only has 32 cores. A
     single-column-per-phase variant fits (4 + 4 = 8 cores) but produces too
     few `m_row_groups` to amortise DMA setup, which makes the chained variant
     SLOWER than two separate kernel launches by a wide margin (the very
     opposite of the speedup the experiment was meant to demonstrate).

To revisit in V3:
  * Drop down to direct MLIR-AIE (`aie.device` / `aie.tile` / `aie.objectfifo`
    primitives) and program the memtile DMA descriptors by hand so T can be
    written-once / read-many. Reference: `mlir-aie/programming_examples/basic/
    matrix_scalar_mul/matrix_scalar_mul_placed.py` for placed-IR style and
    `programming_examples/basic/dma_transpose/` for memref DMA control.
  * Or: implement at the kernel-C++ level — fuse both matmuls inside one core's
    inner loop with T held in L1 scratch (256×256 int16 = 128KB does NOT fit in
    one core's L1 ~64KB, so this only works for very small chains, e.g.
    `(64×64×64) @ (64×64×64)` where T = 64×64×2 = 8KB, fits trivially).
"""

from __future__ import annotations

from typing import Callable

from npupy_xdna.regions.region import Region
from npupy_xdna.templates.protocol import CostEstimate, Config


# ---- Supported chain shapes (only 2-chain at 256³ was in scope for V2 T15) ----
# Each entry is a tuple (M, K1, N1==K2, N2) describing
#     T = A(M,K1) @ B(K1,N1);  D = T(M,K2) @ C(K2,N2)
# Even if `match()` returns False today, listing the canonical target shape lets
# downstream tooling (`cost_model.py`, `results/` reports) reference the design.
SUPPORTED_CHAIN_SHAPES: list[tuple[int, int, int, int]] = [
    (256, 256, 256, 256),
]

# Number of AIE rows in the npu2 array (used by `config_space` for reporting).
_N_AIE_ROWS = 4


class ChainedGemmTemplate:
    """Cross-region fusion template for D = (A @ B) @ C — KILL-SWITCHED."""

    name = "chained_gemm"

    # Set this constant to True once a working IRON program (or lower-level
    # MLIR-AIE program) lands. The dispatcher honours this via `match()`.
    ENABLED: bool = False

    def match(self, region: Region) -> bool:
        """Always False (kill-switch). The chained pattern is recognised at the
        dispatcher level — `dispatcher.py` is responsible for detecting back-to-
        back matmul regions and would route them here once `ENABLED` flips.

        See `npupy_xdna/results/chained_gemm_kill_switch.md` for the rationale.
        """
        if not type(self).ENABLED:
            return False
        # The body below documents what `match()` WOULD do once the template
        # is implemented. It is unreachable as long as `ENABLED == False`.
        if region.op != "matmul_fused":
            return False
        meta = region.metadata or {}
        chain = meta.get("chain")
        if not chain:
            return False
        return tuple(chain) in SUPPORTED_CHAIN_SHAPES

    # ----- Tile / config space (kept for reporting even while disabled) -----

    # MAC intrinsic constraint (npu2 i16, r=4 s=4 t=8) requires
    #     tile_m % r == 0, tile_k % s == 0, tile_n % t == 0
    # Same constraint applies to BOTH GEMMs in the chain because both use the
    # same `matmul_i16_i16` external function.
    TILE_SIZES: list[tuple[int, int, int]] = [
        (64, 64, 64),
    ]

    def config_space(self, region: Region) -> list[Config]:
        """Return the (empty, by design) config space for the chained template.

        Kept non-empty so downstream tooling can show the *intended* config when
        reporting `chained_gemm` even though `match()` rejects every region.
        """
        configs: list[Config] = []
        for m, k, n in self.TILE_SIZES:
            # 1 column per phase × 4 rows × 2 phases = 8 cores total.
            n_cores = _N_AIE_ROWS * 2
            configs.append(
                Config(
                    tile=(m, k, n),
                    n_cores=n_cores,
                    extra={
                        "phases": 2,
                        "intermediate_in_memtile": True,
                        "enabled": type(self).ENABLED,
                        "notes": "kill-switched, see results/chained_gemm_kill_switch.md",
                    },
                )
            )
        return configs

    def lower(self, region: Region, config: Config) -> Callable:
        """Raises until the template is re-enabled.

        The dispatcher never calls `lower()` for templates whose `match()`
        returned False, so this is only reachable if someone bypasses the
        dispatcher and tries to lower the template directly.
        """
        raise NotImplementedError(
            "ChainedGemmTemplate is kill-switched. "
            "See npupy_xdna/results/chained_gemm_kill_switch.md for the root-cause analysis "
            "and the path to re-enabling this template in V3."
        )

    def estimated_cost(self, region: Region, config: Config) -> CostEstimate:
        """Provide a placeholder cost so that comparison tooling has a number to
        show even while the template is disabled. The number reflects the
        *hypothesised* cost if the design worked (used by V2 reporting to
        contrast against the V1 3mm baseline of 14.7×)."""
        # Crude back-of-the-envelope:
        #   - 2 × M*K*N MACs per GEMM = 2 × 256^3 = 33.6M MACs per GEMM
        #   - 32-core npu2 i16 MAC peak ≈ 32 × 4 × 4 × 8 / cycle = ~16 GMAC/s/core?
        #     The actual measured throughput for the single GEMM at 256³ in V2 is
        #     ~30 GOPS sustained for the whole array; assume the chain runs at the
        #     same sustained rate => latency_us ≈ 2 × 256^3 × 2 / 30e9 * 1e6 ≈ 2.2µs
        # We bump confidence to 0.0 so the cost model treats this prediction as
        # an unreliable hypothesis.
        return CostEstimate(
            predicted_latency_us=2.2,
            predicted_gops=30.0,
            confidence=0.0,
        )
