#define STENCIL_W 128
#define STENCIL_STRIP_H 16
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
// For each output row i and column j:
//   - if j == 0 or j == _W-1 (left/right boundary): out[i][j] = 0
//   - else: out[i][j] = (center + top + bottom + left + right) / 5
//     using int32 accumulator; result is clipped to int16 range.
//
// Division uses C truncation-toward-zero (same as `s / 5` in C).
extern "C" void stencil_5pt_int16(int16_t *restrict in, int16_t *restrict out) {
    for (int i = 0; i < _SH; i++) {
        const int16_t *in_top = in + i * _W;
        const int16_t *in_ctr = in + (i + 1) * _W;
        const int16_t *in_bot = in + (i + 2) * _W;
        int16_t *out_row = out + i * _W;

        out_row[0] = (int16_t)0;

        for (int j = 1; j < _W - 1; j++) {
            int32_t s = (int32_t)in_ctr[j]
                      + (int32_t)in_top[j]
                      + (int32_t)in_bot[j]
                      + (int32_t)in_ctr[j - 1]
                      + (int32_t)in_ctr[j + 1];
            int32_t v = s / 5;
            if (v > 32767)  v = 32767;
            if (v < -32768) v = -32768;
            out_row[j] = (int16_t)v;
        }

        out_row[_W - 1] = (int16_t)0;
    }
}
