#define NOCPP

#include <stdint.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

// Default sizes for 64×64 grid with 8-column strip decomposition.
// STENCIL_W and STENCIL_STRIP_H are injected as compile-time #defines
// by sliding_window.py when generating shape-specific kernels.
#ifndef STENCIL_W
#define STENCIL_W 64
#endif
#ifndef STENCIL_STRIP_H
#define STENCIL_STRIP_H 8
#endif

static constexpr int _W = STENCIL_W;
static constexpr int _SH = STENCIL_STRIP_H;

// in  : (_SH + 2) * _W int16 elements  (one halo row above and below the strip)
// out : _SH * _W int16 elements
//
// 5-point stencil: out[i][j] = (center + top + bot + left + right) / 5
// Boundary columns (j=0 and j=W-1) are zeroed.
// Division: C truncation-toward-zero (same as Python `int(s / 5)`).
//
// Implementation: pad the center row with zeros at both ends so the inner
// loop runs exactly _W iterations (a multiple of 32 for any supported W),
// giving the Peano auto-vectorizer two full 32-element passes with no tail.
// #pragma clang loop vectorize(disable) is ignored by Peano on AIE2 targets.
extern "C" void stencil_5pt_int16(int16_t *restrict in, int16_t *restrict out) {
    for (int i = 0; i < _SH; i++) {
        const int16_t *in_top = in + i * _W;
        const int16_t *in_ctr = in + (i + 1) * _W;
        const int16_t *in_bot = in + (i + 2) * _W;
        int16_t *out_row = out + i * _W;

        // Build padded center row: [0, in_ctr[0..W-1], 0]
        // padded[j] is the left neighbor of column j (padded[0]=0 → col 0 has no left neighbor).
        // padded[j+2] is the right neighbor (padded[W+1]=0 → col W-1 has no right neighbor).
        int16_t padded[_W + 2];
        padded[0] = 0;
#pragma clang loop min_iteration_count(1)
        for (int k = 0; k < _W; k++) padded[k + 1] = in_ctr[k];
        padded[_W + 1] = 0;

        // Inner loop: exactly _W iterations (= 64 = 2×32, no tail for the vectorizer).
        // Boundary columns (j=0 and j=W-1) are zeroed after the loop.
#pragma clang loop min_iteration_count(1)
        for (int j = 0; j < _W; j++) {
            int32_t s = (int32_t)in_ctr[j]
                      + (int32_t)in_top[j]
                      + (int32_t)in_bot[j]
                      + (int32_t)padded[j]
                      + (int32_t)padded[j + 2];
            int32_t v = s / 5;
            if (v > 32767)  v = 32767;
            if (v < -32768) v = -32768;
            out_row[j] = (int16_t)v;
        }

        out_row[0]      = (int16_t)0;
        out_row[_W - 1] = (int16_t)0;
    }
}
