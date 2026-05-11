#define NOCPP

#include <stdint.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

// INT16_TILE_SIZE must match LINE_SIZE in col_independent.py.
static constexpr int INT16_TILE_SIZE = 512;

// Pade [3/3] rational approximant for tanh:
//   tanh(x) ≈ x*(27 + x^2) / (27 + 9*x^2)   accurate for |x| <= 3
//
// Fixed-point scheme (input int16, output int16):
//   xs = input >> 8   → scales int16 range to [-128, 127]
//   For |xs| >= 4: output saturates to ±32767 (tanh(4) ≈ 0.9993)
//   Result scaled back: out = num * 32767 / den  → int16 range
//
// Arithmetic intensity: 7 ops/element (vs relu 1 op/element)
static void tanh_scalar_loop(int16_t *restrict a, int16_t *restrict c,
                              const int32_t size) {
  for (int i = 0; i < size; i++) {
    int32_t xs = (int32_t)a[i] >> 8;

    if (xs >= 4) {
      c[i] = 32767;
      continue;
    }
    if (xs <= -4) {
      c[i] = (int16_t)(-32768);
      continue;
    }

    int32_t xs2 = xs * xs;
    int32_t num = xs * (27 + xs2);
    int32_t den = 27 + 9 * xs2;

    // |num| <= 108, so 108*32767 = 3,538,836 < INT32_MAX — no overflow
    int32_t out = (num * 32767) / den;

    if (out > 32767) out = 32767;
    if (out < -32768) out = -32768;

    c[i] = (int16_t)out;
  }
}

extern "C" {

void tanh_int16(int16_t *restrict a_in, int16_t *restrict c_out,
                int32_t size) {
  tanh_scalar_loop(a_in, c_out, size);
}

void int16_tanh(int16_t *a_in, int16_t *c_out) {
  tanh_scalar_loop(a_in, c_out, INT16_TILE_SIZE);
}

}
